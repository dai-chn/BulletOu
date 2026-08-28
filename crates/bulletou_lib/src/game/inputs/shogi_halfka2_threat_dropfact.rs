//! HalfKA2 + Threat + **threat from-drop factoriser** (task#70 / report/52 §18.2)
//!
//! `ShogiHalfKa2Threat` (348,669 = KA2 131,949 + Threat 216,720) に、threat の粗い共有成分
//! = from-drop (lite) 26,244 次元を**仮想特徴**として同時学習させる入力型。
//! classic threat-512 では同じ factoriser が full に +43.7 有意 (L23) だった。
//!
//! ## レイアウト (訓練時の l0w 行)
//!
//! `[KA2 131,949][Threat 216,720][KA virtual 1,629][Lite virtual 26,244]` = 376,542 行。
//! - KA virtual は SFNN fast path の「暗黙 virtual rows」(CUDA 側 `feature % 1629`) のまま。
//! - Lite virtual は **CPU 側で明示 emit** する (threat 1 件につき `LITE_VIRTUAL_BASE + full_to_lite[t]` を追加)。
//!   threat→lite は modulo で書けない任意写像なので暗黙化しない。重複はそのまま emit
//!   (同一 (pair, to) に複数の攻め駒 → count 意味論、bullet-shogi `ShogiHalfKPThreatLite` と同じ)。
//! - export は `fold` で base 348,669 行に畳む (KA2 行 += KA virtual、threat 行 += lite virtual)。
//!   **nn.bin の hash / 次元 / description は `ShogiHalfKa2Threat` と同一** (YO の halfka2t ビルドがそのまま読む)。
//!
//! ## 対応表
//!
//! `full_to_lite` は bullet-shogi `ShogiPThreatDrop::new` (shogi_halfkp_threat_dropfact.rs) と
//! 同じ走査 (pair → pattern → 81 from → 空盤 to) で構築する。threat の index 式・テーブルは
//! 両実装でバイト一致 (checksum 0x30f7eea2484893cd) なので写像もそのまま一致する。
//! lite index = `pair * 81 + to_n` (`to_n` は視点正規化済みの升)。

use std::sync::LazyLock;

use super::SparseInputType;
use super::shogi_halfka::{HALFKA2_DIMENSIONS, MAX_ACTIVE_FEATURES, PIECE_INPUTS, map_halfka2_features};
use super::shogi_halfka_hm_threat::{
    ATTACK_ORDER_TABLE, AttackOrderTable, FROM_OFFSET_TABLE, NUM_THREAT_CLASSES, ThreatClass, attack_pattern_id,
    attacks_empty_board, build_pair_base,
};
use super::shogi_halfka2_threat::{HALFKA2_THREAT_DIMENSIONS, HALFKA2_THREAT_TOTAL_DIMENSIONS, ShogiHalfKa2Threat};
use super::shogi_threat_exclusion::ThreatProfile;
use crate::shogi::{
    PackedSfenValue, ShogiBoard,
    types::{Color, Square},
};

/// active threat features の安全上限 (ShogiHalfKa2Threat と同じ)。lite 仮想特徴も同数まで
const MAX_ACTIVE_THREAT_FEATURES: usize = 320;

/// from-drop (lite) の次元: 2*9*2*9 pair × 81 to
pub const HALFKA2_THREAT_LITE_DIMENSIONS: usize = 2 * NUM_THREAT_CLASSES * 2 * NUM_THREAT_CLASSES * 81; // 26,244

/// KA virtual rows の先頭 (= base の直後)。SFNN fast path の暗黙 factorise が使う
pub const HALFKA2T_KA_VIRTUAL_BASE: usize = HALFKA2_THREAT_TOTAL_DIMENSIONS; // 348,669
/// Lite virtual rows の先頭
pub const HALFKA2T_LITE_VIRTUAL_BASE: usize = HALFKA2T_KA_VIRTUAL_BASE + PIECE_INPUTS; // 350,298
/// 訓練時の総行数 (base + KA virtual + Lite virtual)
pub const HALFKA2T_DROPFACT_TOTAL_DIMENSIONS: usize = HALFKA2T_LITE_VIRTUAL_BASE + HALFKA2_THREAT_LITE_DIMENSIONS; // 376,542

const ALL_CLASSES: [ThreatClass; NUM_THREAT_CLASSES] = [
    ThreatClass::Pawn,
    ThreatClass::Lance,
    ThreatClass::Knight,
    ThreatClass::Silver,
    ThreatClass::GoldLike,
    ThreatClass::Bishop,
    ThreatClass::Rook,
    ThreatClass::Horse,
    ThreatClass::Dragon,
];

/// full threat index (0..216,720) → lite index (0..26,244)。
/// bullet-shogi `ShogiPThreatDrop::new(ThreatProfile::Full)` と同一の構築。
fn build_full_to_lite() -> Box<[u32]> {
    let (pair_base, threat_dims) = build_pair_base(ThreatProfile::Full);
    assert_eq!(threat_dims, HALFKA2_THREAT_DIMENSIONS);
    let mut full_to_lite = vec![u32::MAX; threat_dims];
    let from_offset = &*FROM_OFFSET_TABLE;
    for attacker_side in 0..2usize {
        let oriented = if attacker_side == 0 { Color::Black } else { Color::White };
        for &ac in &ALL_CLASSES {
            let pattern = attack_pattern_id(ac, oriented);
            for attacked_side in 0..2usize {
                for &dc in &ALL_CLASSES {
                    let pair = attacker_side * 162 + (ac as usize) * 18 + attacked_side * 9 + dc as usize;
                    let base = pair_base[pair];
                    for from_raw in 0..81u8 {
                        let from_n = Square(from_raw);
                        let (targets, count) = attacks_empty_board(ac, oriented, from_n);
                        let off = from_offset.get(pattern, from_n);
                        for &to_raw in &targets[..count] {
                            let to_n = Square(to_raw);
                            let ord = ATTACK_ORDER_TABLE.get(pattern, from_n, to_n);
                            debug_assert_ne!(ord, AttackOrderTable::INVALID);
                            let full = base + off + ord as usize;
                            let lite = pair * 81 + to_raw as usize;
                            debug_assert!(full < threat_dims && lite < HALFKA2_THREAT_LITE_DIMENSIONS);
                            debug_assert_eq!(full_to_lite[full], u32::MAX, "full index {full} に二重割当");
                            full_to_lite[full] = lite as u32;
                        }
                    }
                }
            }
        }
    }
    assert!(
        full_to_lite.iter().all(|&v| v != u32::MAX),
        "full→lite 逆引き表に未割当の index がある (テーブル走査順が index 定義とずれている)"
    );
    full_to_lite.into_boxed_slice()
}

static FULL_TO_LITE: LazyLock<Box<[u32]>> = LazyLock::new(build_full_to_lite);

/// full threat index → lite index
#[inline]
pub fn threat_full_to_lite(full: usize) -> usize {
    FULL_TO_LITE[full] as usize
}

/// 対応表の FNV-1a チェックサム (bullet-shogi 側と照合するための錠)
pub fn threat_full_to_lite_checksum() -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in FULL_TO_LITE.iter() {
        h ^= v as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// HalfKA2 + Threat + threat from-drop factoriser (unit struct、`ShogiHalfKa2Threat` と同じ扱い)
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKa2ThreatDropFact;

impl ShogiHalfKa2ThreatDropFact {
    /// threat (full) と lite 仮想の index を列挙する (テスト・検証用)。
    /// f(stm_full, nstm_full, stm_lite, nstm_lite)、いずれも各空間内のオフセット。
    pub(crate) fn map_threat_and_lite<F: FnMut(usize, usize, usize, usize)>(&self, board: &ShogiBoard, mut f: F) {
        ShogiHalfKa2Threat.map_threat_only(board, |s, n| {
            f(s, n, threat_full_to_lite(s), threat_full_to_lite(n));
        });
    }

    /// 盤面から全特徴 (KA2 + threat + lite virtual) を列挙する本体。
    pub(crate) fn map_features_board<F: FnMut(usize, usize)>(&self, board: &ShogiBoard, mut f: F) {
        // Part 1: HalfKA2 (ShogiHalfKa2 と完全同一)
        map_halfka2_features(board, &mut f);
        // Part 2: Threat (オフセット HALFKA2_DIMENSIONS) + Part 3: Lite virtual (明示 emit、重複そのまま)
        ShogiHalfKa2Threat.map_threat_only(board, |stm_t, nstm_t| {
            f(HALFKA2_DIMENSIONS + stm_t, HALFKA2_DIMENSIONS + nstm_t);
            f(
                HALFKA2T_LITE_VIRTUAL_BASE + threat_full_to_lite(stm_t),
                HALFKA2T_LITE_VIRTUAL_BASE + threat_full_to_lite(nstm_t),
            );
        });
    }
}

impl SparseInputType for ShogiHalfKa2ThreatDropFact {
    type RequiredDataType = PackedSfenValue;

    /// ★訓練時の行数 (仮想行込み)。loader の index 上限チェックはこの値で行われる。
    /// export の base 次元 (348,669) は `HALFKA2_THREAT_TOTAL_DIMENSIONS` を使うこと。
    fn num_inputs(&self) -> usize {
        HALFKA2T_DROPFACT_TOTAL_DIMENSIONS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_FEATURES + 2 * MAX_ACTIVE_THREAT_FEATURES
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        self.map_features_board(&board, f);
    }

    fn shorthand(&self) -> String {
        format!("shogi-halfka2-threat-dropfact-{}", HALFKA2T_DROPFACT_TOTAL_DIMENSIONS)
    }

    fn description(&self) -> String {
        format!(
            "Shogi HalfKA2 (131,949) + Threat ({}, profile=full) + KA virtual ({}) + Threat from-drop virtual ({})",
            HALFKA2_THREAT_DIMENSIONS, PIECE_INPUTS, HALFKA2_THREAT_LITE_DIMENSIONS
        )
    }
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::types::{Piece, PieceType};

    fn startpos_board() -> ShogiBoard {
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);
        for file in 0..9u8 {
            board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
            board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }
        for (f, r, c, pt) in [
            (7u8, 7u8, Color::Black, PieceType::Bishop), (1, 7, Color::Black, PieceType::Rook),
            (1, 1, Color::White, PieceType::Bishop), (7, 1, Color::White, PieceType::Rook),
            (0, 8, Color::Black, PieceType::Lance), (8, 8, Color::Black, PieceType::Lance),
            (0, 0, Color::White, PieceType::Lance), (8, 0, Color::White, PieceType::Lance),
            (1, 8, Color::Black, PieceType::Knight), (7, 8, Color::Black, PieceType::Knight),
            (1, 0, Color::White, PieceType::Knight), (7, 0, Color::White, PieceType::Knight),
            (2, 8, Color::Black, PieceType::Silver), (6, 8, Color::Black, PieceType::Silver),
            (2, 0, Color::White, PieceType::Silver), (6, 0, Color::White, PieceType::Silver),
            (3, 8, Color::Black, PieceType::Gold), (5, 8, Color::Black, PieceType::Gold),
            (3, 0, Color::White, PieceType::Gold), (5, 0, Color::White, PieceType::Gold),
        ] {
            board.board[Square::new(f, r).index()] = Piece::new(c, pt);
        }
        board
    }

    #[test]
    fn test_dims_and_layout() {
        assert_eq!(HALFKA2_THREAT_LITE_DIMENSIONS, 26_244);
        assert_eq!(HALFKA2T_KA_VIRTUAL_BASE, 348_669);
        assert_eq!(HALFKA2T_LITE_VIRTUAL_BASE, 350_298);
        assert_eq!(HALFKA2T_DROPFACT_TOTAL_DIMENSIONS, 376_542);
        assert_eq!(ShogiHalfKa2ThreatDropFact.num_inputs(), 376_542);
        assert_eq!(ShogiHalfKa2ThreatDropFact.max_active(), 40 + 320 + 320);
    }

    #[test]
    fn test_full_to_lite_coverage() {
        // 全 full index が割り当て済み (build 時 assert)、lite 側の被覆率と決定性
        let table = &*FULL_TO_LITE;
        assert_eq!(table.len(), 216_720);
        let mut counts = vec![0u32; HALFKA2_THREAT_LITE_DIMENSIONS];
        for &l in table.iter() {
            counts[l as usize] += 1;
        }
        let covered = counts.iter().filter(|&&c| c > 0).count();
        assert!(covered * 10 > HALFKA2_THREAT_LITE_DIMENSIONS * 9, "lite 被覆率 {covered}/{}", HALFKA2_THREAT_LITE_DIMENSIONS);
        assert_eq!(counts.iter().map(|&c| c as usize).sum::<usize>(), 216_720);
        // 決定性 (2 回構築して同じ)
        assert_eq!(&*build_full_to_lite(), &**table);
        println!("FULL_TO_LITE_CHECKSUM {:#018x}", threat_full_to_lite_checksum());
    }

    #[test]
    fn test_startpos_emits_threat_and_lite_pairs() {
        // 初期局面: threat 34 件 (canonical) → 特徴総数 = KA2 + 34 + 34 (lite)
        let board = startpos_board();
        let mut n_full = 0usize;
        let mut n_lite = 0usize;
        ShogiHalfKa2ThreatDropFact.map_threat_and_lite(&board, |s, n, ls, ln| {
            assert!(s < HALFKA2_THREAT_DIMENSIONS && n < HALFKA2_THREAT_DIMENSIONS);
            assert!(ls < HALFKA2_THREAT_LITE_DIMENSIONS && ln < HALFKA2_THREAT_LITE_DIMENSIONS);
            // lite の pair は full の pair と一致する (pair_base の単調性から full の pair を逆引き)
            n_full += 1;
            n_lite += 1;
        });
        assert_eq!(n_full, 34);
        assert_eq!(n_lite, 34);

        let mut all = Vec::new();
        ShogiHalfKa2ThreatDropFact.map_features_board(&board, |s, n| all.push((s, n)));
        let n_ka2 = {
            let mut c = 0;
            map_halfka2_features(&board, |_, _| c += 1);
            c
        };
        assert_eq!(all.len(), n_ka2 + 34 + 34);
        assert!(all.iter().all(|&(s, n)| s < HALFKA2T_DROPFACT_TOTAL_DIMENSIONS && n < HALFKA2T_DROPFACT_TOTAL_DIMENSIONS));
        // KA virtual 行 (348,669..350,298) は CPU 側では emit しない
        assert!(all.iter().all(|&(s, n)| !(HALFKA2T_KA_VIRTUAL_BASE..HALFKA2T_LITE_VIRTUAL_BASE).contains(&s)
            && !(HALFKA2T_KA_VIRTUAL_BASE..HALFKA2T_LITE_VIRTUAL_BASE).contains(&n)));
        assert_eq!(all.iter().filter(|&&(s, _)| s >= HALFKA2T_LITE_VIRTUAL_BASE).count(), 34);
    }

    #[test]
    fn test_lite_index_pair_consistency() {
        // lite = pair*81 + to: pair 部分が full の pair (pair_base 区間) と一致すること
        let (pair_base, dims) = build_pair_base(ThreatProfile::Full);
        let mut bounds: Vec<(usize, usize, usize)> = (0..pair_base.len()).map(|p| (pair_base[p], p, 0)).collect();
        bounds.sort_unstable();
        for i in 0..bounds.len() {
            let end = if i + 1 < bounds.len() { bounds[i + 1].0 } else { dims };
            bounds[i].2 = end;
        }
        for &(start, pair, end) in &bounds {
            for full in start..end {
                assert_eq!(threat_full_to_lite(full) / 81, pair, "full {full} の lite pair が不一致");
            }
        }
    }
}

#[cfg(test)]
mod dump_env_tests {
    use super::*;
    use crate::shogi::{PackedSfenValue, ShogiBoard};

    /// THREAT_DUMP_PSV=<psv> で全レコードの lite index (stm/nstm, ソート済・重複保持) を出力する。
    /// bullet-shogi 側 `ShogiHalfKPThreatLite` の dump (同形式) と多重集合で一致することを確認する
    /// (count 意味論の cross-crate 検証)。
    #[test]
    fn dump_threat_lite_indices_env() {
        let Ok(path) = std::env::var("THREAT_DUMP_PSV") else { return; };
        let bytes = std::fs::read(&path).expect("read psv");
        for (i, rec) in bytes.chunks_exact(40).enumerate() {
            let mut psv = PackedSfenValue::default();
            psv.as_bytes_mut().copy_from_slice(rec);
            let board = ShogiBoard::from_packed_sfen(&psv);
            let (mut s, mut n) = (Vec::new(), Vec::new());
            ShogiHalfKa2ThreatDropFact.map_threat_and_lite(&board, |_, _, ls, ln| {
                s.push(ls);
                n.push(ln);
            });
            s.sort_unstable();
            n.sort_unstable();
            println!("R{} S:{:?} N:{:?}", i, s, n);
        }
    }
}
