mod direct;
pub mod hcpe;
pub mod hcpe3;
mod montybinpack;
mod rng;
pub mod sfbinpack;
pub mod shogipack;
mod text;
pub mod viribinpack;

pub use direct::{CanBeDirectlySequentiallyLoaded, DirectSequentialDataLoader};
pub use hcpe::HcpeDataLoader;
pub use hcpe3::Hcpe3DataLoader;
pub use montybinpack::MontyBinpackLoader;
pub use sfbinpack::SfBinpackLoader;
pub use shogipack::ShogiPackLoader;
pub use text::InMemoryTextLoader;
pub use viribinpack::{ViriBinpackLoader, ViriFilter};

use bulletformat::BulletFormat;
use rayon::prelude::*;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::game::{inputs::SparseInputType, outputs::OutputBuckets};

/// 破損レコードの通算数。学習を止めないかわりに、混入率が見えるよう報告する。
static CORRUPT_ENTRIES: AtomicU64 = AtomicU64::new(0);

/// 破損レコードを 1 件計上し、初回と 10 万件ごとに警告を出す。
/// ★黙って捨てると混入率が分からなくなるので、必ず可視化する。
/// 率が無視できない大きさなら、データでなくデコード側のバグを疑うこと。
#[cold]
fn report_corrupt_entry() {
    let n = CORRUPT_ENTRIES.fetch_add(1, Ordering::Relaxed) + 1;
    if n == 1 || n % 100_000 == 0 {
        println!("[warn] 破損レコードを健全レコードの複製で置換: 通算 {n} 件");
    }
}

/// これまでに置き換えた破損レコード数。訓練終了時の健全性レポート用。
pub fn corrupt_entries_skipped() -> u64 {
    CORRUPT_ENTRIES.load(Ordering::Relaxed)
}

use super::Wgt;

unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::ChessBoard {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::AtaxxBoard {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::chess::CudADFormat {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::chess::MarlinFormat {}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    Loss = 0,
    Draw = 1,
    Win = 2,
}

pub trait LoadableDataType: Sized {
    fn score(&self) -> i16;

    fn result(&self) -> GameResult;
}

impl<T: BulletFormat + 'static> LoadableDataType for T {
    fn result(&self) -> GameResult {
        [GameResult::Loss, GameResult::Draw, GameResult::Win][self.result_idx()]
    }

    fn score(&self) -> i16 {
        <Self as BulletFormat>::score(self)
    }
}

/// Dictates how data is read from a file into the expected datatype.
/// This allows for the file format to be divorced from the training
/// data format.
pub trait DataLoader<T>: Clone + Send + Sync + 'static {
    fn data_file_paths(&self) -> &[String];

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_chunks<F: FnMut(&[T]) -> bool>(&self, start_position: usize, f: F);
}

pub(crate) type B<I> = fn(&<I as SparseInputType>::RequiredDataType, f32) -> f32;

#[derive(Clone)]
pub struct DefaultDataLoader<I: SparseInputType, O, D> {
    input_getter: I,
    output_getter: O,
    blend_getter: B<I>,
    weight_getter: Option<Wgt<I>>,
    use_win_rate_model: bool,
    wdl: bool,
    scale: f32,
    /// `Some(cap)` のとき `|score| >= cap` の局面を loss から除外（weight を 0 にする）。
    /// 特徴量デコードはそのまま走るが GPU 側で勾配寄与ゼロ。
    score_drop_abs: Option<u16>,
    loader: D,
}

impl<I: SparseInputType, O, D> DefaultDataLoader<I, O, D> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_getter: I,
        output_getter: O,
        blend_getter: B<I>,
        weight_getter: Option<Wgt<I>>,
        use_win_rate_model: bool,
        wdl: bool,
        scale: f32,
        score_drop_abs: Option<u16>,
        loader: D,
    ) -> Self {
        if use_win_rate_model && !wdl {
            initialise_win_rate_model_score_table();
        }
        Self {
            input_getter,
            output_getter,
            blend_getter,
            weight_getter,
            use_win_rate_model,
            wdl,
            scale,
            score_drop_abs,
            loader,
        }
    }
}

impl<I, O, D> DefaultDataLoader<I, O, D>
where
    I: SparseInputType,
    O: OutputBuckets<I::RequiredDataType>,
    D: DataLoader<I::RequiredDataType>,
    I::RequiredDataType: LoadableDataType,
{
    pub fn load_and_map_batches<F: FnMut(&[I::RequiredDataType]) -> bool>(
        &self,
        start_batch: usize,
        batch_size: usize,
        f: F,
    ) {
        self.load_and_map_batches_from_position(start_batch * batch_size, batch_size, f);
    }

    pub fn load_and_map_batches_from_position<F: FnMut(&[I::RequiredDataType]) -> bool>(
        &self,
        start_position: usize,
        batch_size: usize,
        mut f: F,
    ) {
        let mut incomplete_buf = Vec::new();

        self.loader.map_chunks(start_position, |chunk| {
            let remainder = if !incomplete_buf.is_empty() {
                let remainder = batch_size - incomplete_buf.len();

                if chunk.len() >= remainder {
                    incomplete_buf.extend_from_slice(&chunk[..remainder]);
                    let should_break = f(&incomplete_buf);
                    incomplete_buf.clear();

                    if should_break {
                        return true;
                    }
                } else {
                    incomplete_buf.extend_from_slice(chunk);
                }

                remainder
            } else {
                0
            };

            if chunk.len() >= remainder {
                let chunks = chunk[remainder..chunk.len()].chunks_exact(batch_size);
                incomplete_buf.extend_from_slice(chunks.remainder());

                for batch in chunks {
                    let should_break = f(batch);

                    if should_break {
                        return true;
                    }
                }
            }

            false
        });
    }

    pub fn prepare(&self, data: &[I::RequiredDataType], threads: usize, blend: f32) -> PreparedData<I, O> {
        PreparedData::new(
            self.input_getter.clone(),
            self.output_getter,
            self.blend_getter,
            self.weight_getter,
            self.use_win_rate_model,
            self.wdl,
            data,
            threads,
            blend,
            self.scale,
            self.score_drop_abs,
        )
    }

    pub fn prepare_with_pool(
        &self,
        data: &[I::RequiredDataType],
        pool: &rayon::ThreadPool,
        threads: usize,
        blend: f32,
    ) -> PreparedData<I, O> {
        PreparedData::new_with_pool(
            self.input_getter.clone(),
            self.output_getter,
            self.blend_getter,
            self.weight_getter,
            self.use_win_rate_model,
            self.wdl,
            data,
            threads,
            blend,
            self.scale,
            self.score_drop_abs,
            Some(pool),
        )
    }
}

/// A batch of data, in the correct format for the GPU.
pub struct PreparedData<I: SparseInputType, O> {
    pub(crate) input_getter: I,
    pub(crate) output_getter: O,
    pub(crate) batch_size: usize,
    pub(crate) stm: Vec<i32>,
    pub(crate) nstm: Vec<i32>,
    pub(crate) buckets: Vec<i32>,
    pub(crate) targets: Vec<f32>,
    pub(crate) weights: Vec<f32>,
    /// HandCount dense auxiliary input。`I::hand_count_dims()` が `Some` のとき
    /// `Some(hand_count_dim * batch_size)` 長の flat Vec を保持する (列方向 = batch index)。
    /// 次元数 (`hand_count_dim`) はここでは持たず、consumer (model 定義側) が
    /// `input_getter.hand_count_dims()` から取得する前提。
    pub(crate) hand_count: Option<Vec<f32>>,
}

impl<I, O> PreparedData<I, O>
where
    I: SparseInputType,
    O: OutputBuckets<I::RequiredDataType>,
    I::RequiredDataType: LoadableDataType,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_getter: I,
        output_getter: O,
        blend_getter: B<I>,
        weight_getter: Option<Wgt<I>>,
        use_win_rate_model: bool,
        wdl: bool,
        data: &[I::RequiredDataType],
        threads: usize,
        blend: f32,
        scale: f32,
        score_drop_abs: Option<u16>,
    ) -> Self {
        Self::new_with_pool(
            input_getter,
            output_getter,
            blend_getter,
            weight_getter,
            use_win_rate_model,
            wdl,
            data,
            threads,
            blend,
            scale,
            score_drop_abs,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_pool(
        input_getter: I,
        output_getter: O,
        blend_getter: B<I>,
        weight_getter: Option<Wgt<I>>,
        use_win_rate_model: bool,
        wdl: bool,
        data: &[I::RequiredDataType],
        threads: usize,
        blend: f32,
        scale: f32,
        score_drop_abs: Option<u16>,
        pool: Option<&rayon::ThreadPool>,
    ) -> Self {
        let rscale = 1.0 / scale;
        let batch_size = data.len();
        let max_active = input_getter.max_active();
        let chunk_size = batch_size.div_ceil(threads);
        let input_size = input_getter.num_inputs();
        let output_size = if wdl { 3 } else { 1 };
        let sparse_size = max_active * batch_size;
        let hand_count_dims = input_getter.hand_count_dims();
        let hand_count_dim = hand_count_dims.unwrap_or(0);

        let hand_count_init = hand_count_dims.map(|dims| vec![0.0; dims * batch_size]);

        let mut prep = Self {
            input_getter,
            output_getter,
            batch_size,
            stm: vec![-1; sparse_size],
            nstm: vec![-1; sparse_size],
            buckets: vec![0; batch_size],
            targets: vec![0.0; output_size * batch_size],
            weights: vec![0.0; batch_size],
            hand_count: hand_count_init,
        };

        let sparse_chunk_size = max_active * chunk_size;

        if hand_count_dim == 0
            && let Some(pool) = pool
            && threads > 1
            && batch_size > 1
        {
            pool.install(|| {
                data.par_chunks(chunk_size)
                    .zip(prep.stm.par_chunks_mut(sparse_chunk_size))
                    .zip(prep.nstm.par_chunks_mut(sparse_chunk_size))
                    .zip(prep.buckets.par_chunks_mut(chunk_size))
                    .zip(prep.targets.par_chunks_mut(output_size * chunk_size))
                    .zip(prep.weights.par_chunks_mut(chunk_size))
                    .for_each(
                        |(((((data_chunk, stm_chunk), nstm_chunk), buckets_chunk), results_chunk), weights_chunk)| {
                            let inp = &prep.input_getter;
                            let out = &prep.output_getter;

                            // 破損レコードの位置と、複製元にする健全レコードの位置。
                            let mut corrupt_indices: Vec<usize> = Vec::new();
                            let mut donor: Option<usize> = None;
                            let chunk_len = data_chunk.len();

                            for i in 0..data_chunk.len() {
                                let pos = &data_chunk[i];
                                let mut j_stm: usize = 0;
                                let mut j_nstm: usize = 0;
                                let sparse_offset = max_active * i;

                                // 破損レコード耐性 (bullet-shogi から移植, HANDOFF-2026-07-20 §14)。
                                // 駒数が不正な PSV は特徴数が max_active を超えたり範囲外 index を
                                // 出したりする。★書き込みは必ず境界内に抑える — 溢れてから assert
                                // すると、はみ出した分が隣のレコードの領域を踏んだ後に発覚する
                                // (= 静かなデータ汚染)。破損は corrupt フラグで後段に伝える。
                                let mut corrupt = false;

                                inp.map_features_split(pos, |our_opt, opp_opt| {
                                    if let Some(our) = our_opt {
                                        if our < input_size && j_stm < max_active {
                                            stm_chunk[sparse_offset + j_stm] = our as i32;
                                            j_stm += 1;
                                        } else {
                                            corrupt = true;
                                        }
                                    }
                                    if let Some(opp) = opp_opt {
                                        if opp < input_size && j_nstm < max_active {
                                            nstm_chunk[sparse_offset + j_nstm] = opp as i32;
                                            j_nstm += 1;
                                        } else {
                                            corrupt = true;
                                        }
                                    }
                                });

                                // 未使用スロットを -1 で埋める (STM/NSTM 独立)
                                for j in j_stm..max_active {
                                    stm_chunk[sparse_offset + j] = -1;
                                }
                                for j in j_nstm..max_active {
                                    nstm_chunk[sparse_offset + j] = -1;
                                }

                                if corrupt {
                                    // ★特徴を切り詰めたまま実 target を付けて流してはいけない。
                                    //   「欠けた入力 → ある値」を学習させてバイアスを汚染する。
                                    //   weight=0 も効かない (weight_getter 未設定の実験ではグラフに
                                    //   繋がらない)。位置だけ記録し、ループ後に健全レコードの
                                    //   複製で置き換える。out.bucket(pos) は破損局面でパニックしうる
                                    //   ので呼ばずに離脱する。
                                    corrupt_indices.push(i);
                                    report_corrupt_entry();
                                    continue;
                                }
                                if donor.is_none() {
                                    donor = Some(i);
                                }

                                if O::BUCKETS > 1 {
                                    buckets_chunk[i] = out.bucket(pos) as i32;
                                }
                                let mut weight = weight_getter.map_or(1.0, |w| w(pos));
                                if let Some(cap) = score_drop_abs {
                                    if pos.score().unsigned_abs() >= cap {
                                        weight = 0.0;
                                    }
                                }
                                weights_chunk[i] = weight;

                                if wdl {
                                    results_chunk[output_size * i + usize::from(pos.result() as u8)] = 1.0;
                                } else {
                                    let score = if use_win_rate_model {
                                        win_rate_model_score(pos.score())
                                    } else {
                                        let score = f32::from(pos.score());
                                        sigmoid(rscale * score)
                                    };
                                    let result = f32::from(pos.result() as u8) / 2.0;
                                    let blend = blend_getter(pos, blend);
                                    assert!((0.0..=1.0).contains(&blend), "WDL proportion must be in [0, 1]");
                                    results_chunk[i] = blend * result + (1. - blend) * score;
                                }
                            }

                            replace_corrupt_with_donor(
                                &corrupt_indices, donor, chunk_len, max_active, output_size, wdl,
                                stm_chunk, nstm_chunk, buckets_chunk, results_chunk, weights_chunk,
                                None,
                            );
                        },
                    );
            });

            return prep;
        }

        // HandCount 用の並列チャンクを事前に materialise。Option は並列ループ内で扱う。
        let hand_count_chunk_size = hand_count_dim * chunk_size;
        let num_chunks = batch_size.div_ceil(chunk_size);
        let hand_count_slices: Vec<Option<&mut [f32]>> = if let Some(hc) = prep.hand_count.as_mut() {
            hc.chunks_mut(hand_count_chunk_size).map(Some).collect()
        } else {
            (0..num_chunks).map(|_| None).collect()
        };

        std::thread::scope(|s| {
            data.chunks(chunk_size)
                .zip(prep.stm.chunks_mut(sparse_chunk_size))
                .zip(prep.nstm.chunks_mut(sparse_chunk_size))
                .zip(prep.buckets.chunks_mut(chunk_size))
                .zip(prep.targets.chunks_mut(output_size * chunk_size))
                .zip(prep.weights.chunks_mut(chunk_size))
                .zip(hand_count_slices)
                .for_each(
                    |(
                        (((((data_chunk, stm_chunk), nstm_chunk), buckets_chunk), results_chunk), weights_chunk),
                        hand_count_chunk,
                    )| {
                        let inp = &prep.input_getter;
                        let out = &prep.output_getter;
                        s.spawn(move || {
                            let chunk_len = data_chunk.len();
                            let mut hand_count_chunk = hand_count_chunk;

                            // 破損レコードの位置と、複製元にする健全レコードの位置。
                            let mut corrupt_indices: Vec<usize> = Vec::new();
                            let mut donor: Option<usize> = None;

                            for i in 0..chunk_len {
                                let pos = &data_chunk[i];

                                if let Some(hc_slice) = hand_count_chunk.as_deref_mut() {
                                    let offset = hand_count_dim * i;
                                    let end = offset + hand_count_dim;
                                    // 事前に 0 で埋め済み。fill_hand_count は
                                    // 書き込みのみで読まないので再初期化は不要。
                                    inp.fill_hand_count(pos, &mut hc_slice[offset..end]);
                                }
                                // STM と NSTM は独立カウンタで管理: 非対称 feature
                                // (HandThreat defensive 等) で |STM_active| != |NSTM_active|
                                // を許可するため。symmetric な input type は
                                // map_features_split の default impl 経由で
                                // 両側同時に進むので従来挙動と一致する。
                                let mut j_stm: usize = 0;
                                let mut j_nstm: usize = 0;
                                let sparse_offset = max_active * i;

                                // 破損レコード耐性 (bullet-shogi から移植, HANDOFF-2026-07-20 §14)。
                                // 駒数が不正な PSV は特徴数が max_active を超えたり範囲外 index を
                                // 出したりする。★書き込みは必ず境界内に抑える — 溢れてから assert
                                // すると、はみ出した分が隣のレコードの領域を踏んだ後に発覚する
                                // (= 静かなデータ汚染)。破損は corrupt フラグで後段に伝える。
                                let mut corrupt = false;

                                inp.map_features_split(pos, |our_opt, opp_opt| {
                                    if let Some(our) = our_opt {
                                        if our < input_size && j_stm < max_active {
                                            stm_chunk[sparse_offset + j_stm] = our as i32;
                                            j_stm += 1;
                                        } else {
                                            corrupt = true;
                                        }
                                    }
                                    if let Some(opp) = opp_opt {
                                        if opp < input_size && j_nstm < max_active {
                                            nstm_chunk[sparse_offset + j_nstm] = opp as i32;
                                            j_nstm += 1;
                                        } else {
                                            corrupt = true;
                                        }
                                    }
                                });

                                // 未使用スロットを -1 で埋める (STM/NSTM 独立)
                                for j in j_stm..max_active {
                                    stm_chunk[sparse_offset + j] = -1;
                                }
                                for j in j_nstm..max_active {
                                    nstm_chunk[sparse_offset + j] = -1;
                                }

                                if corrupt {
                                    // ★特徴を切り詰めたまま実 target を付けて流してはいけない。
                                    //   「欠けた入力 → ある値」を学習させてバイアスを汚染する。
                                    //   weight=0 も効かない (weight_getter 未設定の実験ではグラフに
                                    //   繋がらない)。位置だけ記録し、ループ後に健全レコードの
                                    //   複製で置き換える。out.bucket(pos) は破損局面でパニックしうる
                                    //   ので呼ばずに離脱する。
                                    corrupt_indices.push(i);
                                    report_corrupt_entry();
                                    continue;
                                }
                                if donor.is_none() {
                                    donor = Some(i);
                                }

                                if O::BUCKETS > 1 {
                                    buckets_chunk[i] = out.bucket(pos) as i32;
                                }
                                let mut weight = weight_getter.map_or(1.0, |w| w(pos));
                                if let Some(cap) = score_drop_abs {
                                    if pos.score().unsigned_abs() >= cap {
                                        weight = 0.0;
                                    }
                                }
                                weights_chunk[i] = weight;

                                if wdl {
                                    results_chunk[output_size * i + usize::from(pos.result() as u8)] = 1.0;
                                } else {
                                    let score = if use_win_rate_model {
                                        win_rate_model_score(pos.score())
                                    } else {
                                        let score = f32::from(pos.score());
                                        sigmoid(rscale * score)
                                    };
                                    let result = f32::from(pos.result() as u8) / 2.0;
                                    let blend = blend_getter(pos, blend);
                                    assert!((0.0..=1.0).contains(&blend), "WDL proportion must be in [0, 1]");
                                    results_chunk[i] = blend * result + (1. - blend) * score;
                                }
                            }

                            replace_corrupt_with_donor(
                                &corrupt_indices, donor, chunk_len, max_active, output_size, wdl,
                                stm_chunk, nstm_chunk, buckets_chunk, results_chunk, weights_chunk,
                                hand_count_chunk.as_deref_mut().map(|hc| (hc, hand_count_dim)),
                            );
                        });
                    },
                );
        });

        prep
    }
}

/// 破損レコードを、同じチャンク内の健全レコードの複製で置き換える。
///
/// 特徴を切り詰めたまま実 target を付けて流すと「欠けた入力 → ある値」を学習させて
/// バイアスを汚染する。weight=0 も効かない (`weight_getter` 未設定の実験では
/// weight がグラフに繋がらない)。実質は健全局面をごくわずかに重複サンプルするだけなので、
/// 破損率が低い限り学習への影響は無視できる。
/// 由来: bullet-shogi の同名処理 (HANDOFF-2026-07-20 §14.2)。
#[allow(clippy::too_many_arguments)]
fn replace_corrupt_with_donor(
    corrupt_indices: &[usize],
    donor: Option<usize>,
    chunk_len: usize,
    max_active: usize,
    output_size: usize,
    wdl: bool,
    stm_chunk: &mut [i32],
    nstm_chunk: &mut [i32],
    buckets_chunk: &mut [i32],
    results_chunk: &mut [f32],
    weights_chunk: &mut [f32],
    mut hand_count: Option<(&mut [f32], usize)>,
) {
    if corrupt_indices.is_empty() {
        return;
    }
    if let Some(d) = donor {
        let src = max_active * d;
        let res_src = output_size * d;
        for &i in corrupt_indices {
            stm_chunk.copy_within(src..src + max_active, max_active * i);
            nstm_chunk.copy_within(src..src + max_active, max_active * i);
            results_chunk.copy_within(res_src..res_src + output_size, output_size * i);
            buckets_chunk[i] = buckets_chunk[d];
            weights_chunk[i] = weights_chunk[d];
            if let Some((hc, dim)) = hand_count.as_mut() {
                let dim = *dim;
                let hc_src = dim * d;
                hc.copy_within(hc_src..hc_src + dim, dim * i);
            }
        }
    } else {
        // チャンク全体が破損。複製元が無いので中立値に倒す。
        // データが根本的に壊れている状況なので大きく警告する。
        println!(
            "[warn] チャンク {chunk_len} 件すべてが破損レコードでした。\
             複製元が無いため中立 target に倒します (データを確認してください)"
        );
        for &i in corrupt_indices {
            for j in 0..max_active {
                stm_chunk[max_active * i + j] = -1;
                nstm_chunk[max_active * i + j] = -1;
            }
            buckets_chunk[i] = 0;
            weights_chunk[i] = 0.0;
            if wdl {
                // Draw を 1.0 (情報を持たない選択)
                for k in 0..output_size {
                    results_chunk[output_size * i + k] = if k == 1 { 1.0 } else { 0.0 };
                }
            } else {
                results_chunk[i] = 0.5;
            }
            if let Some((hc, dim)) = hand_count.as_mut() {
                let dim = *dim;
                for j in 0..dim {
                    hc[dim * i + j] = 0.0;
                }
            }
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

static WIN_RATE_MODEL_SCORE_TABLE: OnceLock<Box<[f32]>> = OnceLock::new();

pub(crate) fn initialise_win_rate_model_score_table() -> &'static [f32] {
    WIN_RATE_MODEL_SCORE_TABLE.get_or_init(|| {
        // 定数 (旧: offset=270, out_scaling=380 固定) は wrm_params から取る。
        // このテーブルは OnceLock キャッシュなので、set_wrm_params は必ず
        // 初回のデータロードより前に呼ぶこと (wrm_params 側の使用済みガード参照)。
        let params = crate::value::wrm_params::wrm_params();
        let mut values = Vec::with_capacity(usize::from(u16::MAX) + 1);
        for raw_score in i32::from(i16::MIN)..=i32::from(i16::MAX) {
            let score = raw_score as f32;
            values.push(crate::value::wrm_params::wrm_target(&params, score));
        }
        values.into_boxed_slice()
    })
}

pub(crate) fn win_rate_model_score(score: i16) -> f32 {
    let table = initialise_win_rate_model_score_table();
    let index = (i32::from(score) - i32::from(i16::MIN)) as usize;
    table[index]
}

#[cfg(test)]
mod tests {
    use crate::{
        game::{inputs::SparseInputType, outputs::OutputBuckets},
        value::loader::{GameResult, LoadableDataType, PreparedData, corrupt_entries_skipped},
    };

    #[derive(Clone, Copy)]
    struct TinyPos {
        a: usize,
        b: usize,
        score: i16,
        result: GameResult,
    }

    impl LoadableDataType for TinyPos {
        fn score(&self) -> i16 {
            self.score
        }

        fn result(&self) -> GameResult {
            self.result
        }
    }

    #[derive(Clone)]
    struct TinyInput;

    impl SparseInputType for TinyInput {
        type RequiredDataType = TinyPos;

        fn num_inputs(&self) -> usize {
            8
        }

        fn max_active(&self) -> usize {
            2
        }

        fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
            f(pos.a, pos.b);
            f((pos.a + 1) % 8, (pos.b + 1) % 8);
        }

        fn shorthand(&self) -> String {
            "tiny".to_string()
        }

        fn description(&self) -> String {
            "Tiny test input".to_string()
        }
    }

    #[derive(Clone, Copy, Default)]
    struct TinyBuckets;

    impl OutputBuckets<TinyPos> for TinyBuckets {
        const BUCKETS: usize = 3;

        fn bucket(&self, pos: &TinyPos) -> usize {
            pos.a % 3
        }
    }

    // ---------------------------------------------------------------------
    // 破損レコード耐性 (HANDOFF-2026-07-20 §14 / 2026-08-16 に BulletOu へ移植)
    //
    // 駒数が不正な PSV は特徴数が max_active を超える。旧コードはここで panic し、
    // 1 レコードのために十数時間の学習が落ちていた。さらに境界チェックが書き込みの
    // **後**にあったため、はみ出した分が隣のレコードの領域を踏んでいた。
    // ---------------------------------------------------------------------

    /// `n_features` で発火する特徴数を指定できるテスト用データ点。
    #[derive(Clone, Copy)]
    struct FakePos {
        n_features: usize,
        feature_base: usize,
        score: i16,
    }

    impl LoadableDataType for FakePos {
        fn score(&self) -> i16 {
            self.score
        }
        fn result(&self) -> GameResult {
            GameResult::Draw
        }
    }

    /// max_active=4 / num_inputs=100。境界超過を意図的に作れる。
    #[derive(Clone)]
    struct FakeInput;

    impl SparseInputType for FakeInput {
        type RequiredDataType = FakePos;

        fn num_inputs(&self) -> usize {
            100
        }
        fn max_active(&self) -> usize {
            4
        }
        fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
            for k in 0..pos.n_features {
                let idx = pos.feature_base + k;
                f(idx, idx);
            }
        }
        fn shorthand(&self) -> String {
            "fake".into()
        }
        fn description(&self) -> String {
            "test input".into()
        }
    }

    #[derive(Clone, Copy, Default)]
    struct FakeBuckets;

    impl OutputBuckets<FakePos> for FakeBuckets {
        const BUCKETS: usize = 1;
        fn bucket(&self, _pos: &FakePos) -> usize {
            0
        }
    }

    fn prepare_fake(data: &[FakePos], threads: usize) -> PreparedData<FakeInput, FakeBuckets> {
        PreparedData::new(
            FakeInput,
            FakeBuckets,
            (|_, blend| blend) as fn(&FakePos, f32) -> f32,
            None,
            false,
            false,
            data,
            threads,
            0.0,
            400.0,
            None,
        )
    }

    fn healthy(base: usize, score: i16) -> FakePos {
        FakePos { n_features: 4, feature_base: base, score }
    }

    fn broken(base: usize) -> FakePos {
        // max_active=4 に対して 6 個発火する = 駒数が不正な PSV 相当
        FakePos { n_features: 6, feature_base: base, score: 0 }
    }

    #[test]
    fn corrupt_entry_does_not_panic_and_is_counted() {
        let before = corrupt_entries_skipped();
        let data = [healthy(0, 100), broken(10), healthy(20, -100)];
        let _ = prepare_fake(&data, 1);
        // ★他テストと並列に走ると通算値に混ざるので、増加の向きだけを見る。
        assert!(corrupt_entries_skipped() >= before + 1, "破損件数が計上される");
    }

    #[test]
    fn corrupt_entry_does_not_corrupt_neighbours() {
        let data = [healthy(0, 100), broken(10), healthy(20, -100)];
        let prep = prepare_fake(&data, 1);
        let ma = FakeInput.max_active();

        // 隣接する健全レコードの特徴が、破損レコードのはみ出しで壊れていないこと。
        assert_eq!(&prep.stm[0..ma], &[0, 1, 2, 3], "前の健全レコードが無傷");
        assert_eq!(&prep.stm[2 * ma..3 * ma], &[20, 21, 22, 23], "後の健全レコードが無傷");
    }

    #[test]
    fn corrupt_entry_is_replaced_by_donor_not_left_empty() {
        let data = [healthy(0, 100), broken(10), healthy(20, -100)];
        let prep = prepare_fake(&data, 1);
        let ma = FakeInput.max_active();
        let slot = &prep.stm[ma..2 * ma];

        // ★特徴が全部 -1 (空入力) のまま実 target が付いて流れると
        //   「欠けた入力 → ある値」を学習してバイアスが汚染される。
        //   健全レコード (donor) の複製で埋まっていること。
        assert_ne!(slot, &[-1, -1, -1, -1], "空入力のまま流してはいけない");
        assert_eq!(slot, &prep.stm[0..ma], "donor (先頭の健全レコード) の複製");
        assert_eq!(prep.targets[1], prep.targets[0], "target も donor のもの");
        assert_eq!(prep.weights[1], prep.weights[0], "weight も donor のもの");
    }

    #[test]
    fn corrupt_entry_handled_in_parallel_path_too() {
        // rayon プール経路 (site 1) にも同じ処理が入っていること。
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        let data = [healthy(0, 100), broken(10), healthy(20, -100), healthy(30, 50)];
        let prep = PreparedData::new_with_pool(
            FakeInput,
            FakeBuckets,
            (|_, blend| blend) as fn(&FakePos, f32) -> f32,
            None,
            false,
            false,
            &data,
            2,
            0.0,
            400.0,
            None,
            Some(&pool),
        );
        let ma = FakeInput.max_active();
        assert_eq!(&prep.stm[0..ma], &[0, 1, 2, 3], "健全レコードが無傷");
        assert_ne!(&prep.stm[ma..2 * ma], &[-1, -1, -1, -1], "破損は donor で埋まる");
    }

    #[test]
    fn prepare_with_pool_matches_scoped_prepare() {
        let data: Vec<_> = (0..16)
            .map(|i| TinyPos {
                a: i % 7,
                b: (i * 3) % 7,
                score: (i as i16) * 10 - 80,
                result: [GameResult::Loss, GameResult::Draw, GameResult::Win][i % 3],
            })
            .collect();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();

        let baseline = PreparedData::new(
            TinyInput,
            TinyBuckets,
            (|_, blend| blend) as fn(&TinyPos, f32) -> f32,
            None,
            true,
            false,
            &data,
            4,
            0.0,
            400.0,
            None,
        );
        let pooled = PreparedData::new_with_pool(
            TinyInput,
            TinyBuckets,
            (|_, blend| blend) as fn(&TinyPos, f32) -> f32,
            None,
            true,
            false,
            &data,
            4,
            0.0,
            400.0,
            None,
            Some(&pool),
        );

        assert_eq!(baseline.stm, pooled.stm);
        assert_eq!(baseline.nstm, pooled.nstm);
        assert_eq!(baseline.buckets, pooled.buckets);
        assert_eq!(baseline.targets, pooled.targets);
        assert_eq!(baseline.weights, pooled.weights);
        assert_eq!(baseline.hand_count, pooled.hand_count);
    }
}
