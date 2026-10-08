//! #99 progress bucket: 学習側 (BulletOu) の layer stack bucket を局面ごとに書き出し、探索側 (やねうら王 `nnuedump` の
//! bucket 欄) と突き合わせる。progress.bin は本家 e47d460f (tatara) 形式 (ヘッダ無し little-endian double × 81 × 1548、bias 無し)。
//!
//! usage:
//!   dump_layerstack_buckets --psv <in.bin> [--progress-bin <progress.bin>] [--max N] [--king k3k3|none] [--progress 1|2|3|4|8|16|32]
//! stdout: 1 局面 1 行 `idx,bucket,progress255,game_ply` (idx = ファイル内の順番)。

use std::io::{Read, Write};

use bulletou_lib::game::outputs::{
    SHOGI_SFNN_PROGRESS_WEIGHT_COUNT, ShogiSfnnHandBucketKind, ShogiSfnnKingBucketKind,
    ShogiSfnnLayerStackBucketKind, ShogiSfnnProgressBucketKind, ShogiSfnnProgressQ16Params,
    set_shogi_sfnn_progress_q16_params, shogi_sfnn_progress_0_to_255,
};
use bulletou_lib::shogi::packed_sfen::PackedSfenValue;

fn load_progress_bin(path: &str) -> Result<ShogiSfnnProgressQ16Params, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("failed to read {path}: {e}"))?;
    let expected = SHOGI_SFNN_PROGRESS_WEIGHT_COUNT * 8;
    if bytes.len() != expected {
        return Err(format!("{path}: size {} != {expected} (double x 81 x 1548)", bytes.len()));
    }
    // 探索側 Progress::Parameters::ReadParameters と同じ q16 化
    let weights = bytes
        .chunks_exact(8)
        .map(|c| {
            let v = f64::from_le_bytes(c.try_into().unwrap());
            (v * 65_536.0).round().clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
        })
        .collect::<Vec<_>>();
    ShogiSfnnProgressQ16Params::new(0, weights)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let mut psv_path = None;
    let mut progress_bin = None;
    let mut max_n = usize::MAX;
    let mut king = ShogiSfnnKingBucketKind::KingRank9;
    let mut progress = ShogiSfnnProgressBucketKind::Progress8;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--psv" => {
                psv_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--progress-bin" => {
                progress_bin = Some(args[i + 1].clone());
                i += 2;
            }
            "--max" => {
                max_n = args[i + 1].parse().map_err(|e| format!("--max: {e}"))?;
                i += 2;
            }
            "--king" => {
                king = match args[i + 1].as_str() {
                    "k3k3" => ShogiSfnnKingBucketKind::KingRank9,
                    "none" => ShogiSfnnKingBucketKind::None,
                    other => return Err(format!("--king {other}: k3k3|none")),
                };
                i += 2;
            }
            "--progress" => {
                progress = match args[i + 1].as_str() {
                    "1" => ShogiSfnnProgressBucketKind::None,
                    "2" => ShogiSfnnProgressBucketKind::Progress2,
                    "3" => ShogiSfnnProgressBucketKind::Progress3,
                    "4" => ShogiSfnnProgressBucketKind::Progress4,
                    "8" => ShogiSfnnProgressBucketKind::Progress8,
                    "16" => ShogiSfnnProgressBucketKind::Progress16,
                    "32" => ShogiSfnnProgressBucketKind::Progress32,
                    other => return Err(format!("--progress {other}: 1|2|3|4|8|16|32")),
                };
                i += 2;
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }
    let psv_path = psv_path.ok_or("--psv is required")?;
    if let Some(p) = progress_bin.as_deref() {
        set_shogi_sfnn_progress_q16_params(load_progress_bin(p)?)?;
    }
    let kind = ShogiSfnnLayerStackBucketKind::new(ShogiSfnnHandBucketKind::None, king, progress);
    eprintln!("layerstack: {} stacks (king {} x progress {})", kind.num_stacks(), king.bucket_count(), progress.bucket_count());

    let mut file = std::fs::File::open(&psv_path).map_err(|e| format!("failed to open {psv_path}: {e}"))?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let mut buf = [0u8; 40];
    let mut idx = 0usize;
    let mut hist = vec![0usize; kind.num_stacks()];
    while idx < max_n {
        if file.read_exact(&mut buf).is_err() {
            break;
        }
        let psv: PackedSfenValue = unsafe { std::mem::transmute(buf) };
        let b = kind.bucket(&psv);
        let p = shogi_sfnn_progress_0_to_255(&psv);
        hist[b] += 1;
        writeln!(out, "{idx},{b},{p},{}", psv.game_ply()).map_err(|e| e.to_string())?;
        idx += 1;
    }
    out.flush().map_err(|e| e.to_string())?;
    eprintln!("{idx} positions; bucket histogram: {hist:?}");
    Ok(())
}
