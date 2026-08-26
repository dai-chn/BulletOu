//! HalfKA2 + Threat 連結特徴量 (task#54 / report/52 実行キュー)
//!
//! HalfKA2 (131,949 次元) と Threat (full profile で 216,720 次元) を連結した
//! sparse input 型。SFNN アーキ (FT1024 + pairwise) に threat を足すための
//! **Phase-1 (HalfKP+Threat, +140.4) の SFNN 版最小差分**:
//! KA2 部は `ShogiHalfKa2` と完全同一 index、threat 部は
//! `shogi_halfka_hm_threat.rs` の index 計算 (rshogi threat_spec 由来) を再利用する。
//!
//! ## bullet-shogi 側 `ShogiHalfKPThreat` との対応
//!
//! - threat 部の index 計算・テーブルは**同一** (テーブル checksum 0x30f7eea2484893cd、
//!   YO 側 threat.cpp と照合済)。
//! - ベースが HalfKA2 (81 king bucket 非ミラー、敵玉 collapse) なので
//!   threat 部の升正規化も **HM ミラー無し** (`normalize_sq(sq, persp, false)`) —
//!   HalfKP 版と同じ選択。KA2 は左右対称化しないので threat だけミラーすると
//!   盤の左右で KA2 と threat の対応がねじれる。
//!
//! ## factorisation
//!
//! profile は **Full 固定** (threat factoriser は付けない — multifactor 劣化の
//! 実測 (−393.3) に基づく単因子方針)。KA 部の factorise は SFNN fast path の
//! 「暗黙 virtual rows」機構側で行う (このモジュールの責務外):
//! feature < 131,949 のみ virtual row (feature % 1629) に射影し、threat 部は射影しない。

use std::sync::LazyLock;

use super::SparseInputType;
use super::shogi_halfka::{HALFKA2_DIMENSIONS, MAX_ACTIVE_FEATURES, map_halfka2_features};
use super::shogi_halfka_hm_threat::{
    FROM_OFFSET_TABLE, NUM_PAIRS, Occupied, ThreatClass, ThreatParams, build_pair_base, for_each_attack, normalize_sq,
    threat_index,
};
use super::shogi_threat_exclusion::ThreatProfile;
use crate::shogi::{
    PackedSfenValue, ShogiBoard,
    types::{Color, PieceType, Square},
};

/// active threat features の安全上限 (KA_hm+Threat / KP+Threat 版と同じ値)
const MAX_ACTIVE_THREAT_FEATURES: usize = 320;

/// Full profile の threat 次元数 (2*9*2*9 pair × 幾何 from/to)
pub const HALFKA2_THREAT_DIMENSIONS: usize = 216_720;

/// HalfKA2 + Threat の総入力次元 (131,949 + 216,720)
pub const HALFKA2_THREAT_TOTAL_DIMENSIONS: usize = HALFKA2_DIMENSIONS + HALFKA2_THREAT_DIMENSIONS;

/// Full profile の pair_base テーブル (プロセス内共有)
static PAIR_BASE_FULL: LazyLock<(Box<[usize; NUM_PAIRS]>, usize)> = LazyLock::new(|| {
    let (pair_base, threat_dims) = build_pair_base(ThreatProfile::Full);
    assert_eq!(threat_dims, HALFKA2_THREAT_DIMENSIONS, "Full profile の threat 次元数が canonical 値と不一致");
    (Box::new(pair_base), threat_dims)
});

/// HalfKA2 + Threat 連結特徴量 (Full profile 固定、unit struct)
///
/// SFNN fast path の dispatch (値渡し) に合わせて `Copy` な unit struct にし、
/// pair_base テーブルは `LazyLock` の static を参照する。
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKa2Threat;

impl ShogiHalfKa2Threat {
    pub fn threat_dimensions(&self) -> usize {
        PAIR_BASE_FULL.1
    }

    /// テーブル群 (pair_base / from_offset / attack_order) の FNV-1a チェックサム。
    /// ★bullet-shogi `ShogiHalfKPThreat::tables_checksum` および YO 側 threat.cpp の
    /// Tables と**同じ走査順・同じ式**。3 実装の決定的一致の保証。
    pub fn tables_checksum(&self) -> u64 {
        use super::shogi_halfka_hm_threat::{ATTACK_ORDER_TABLE, NUM_ATTACK_PATTERNS};
        let mut h: u64 = 0xcbf29ce484222325;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x100000001b3);
        };
        for &b in PAIR_BASE_FULL.0.iter() {
            mix(b as u64);
        }
        for pat in 0..NUM_ATTACK_PATTERNS {
            for sq in 0..81u8 {
                mix(FROM_OFFSET_TABLE.get(pat, crate::shogi::types::Square(sq)) as u64);
            }
        }
        for pat in 0..NUM_ATTACK_PATTERNS {
            for from in 0..81u8 {
                for to in 0..81u8 {
                    mix(ATTACK_ORDER_TABLE.get(
                        pat,
                        crate::shogi::types::Square(from),
                        crate::shogi::types::Square(to),
                    ) as u64);
                }
            }
        }
        h
    }

    /// threat 特徴のみを列挙する (テスト・検証用にも公開)。
    /// index は threat 内オフセット (0..216,720)。呼び出し側で HALFKA2_DIMENSIONS を足す。
    /// ロジックは bullet-shogi `ShogiHalfKPThreat::map_threat_only` と同一 (HM ミラー無し)。
    pub(crate) fn map_threat_only<F: FnMut(usize, usize)>(&self, board: &ShogiBoard, mut f: F) {
        let stm = board.side_to_move;
        let nstm = stm.opponent();

        // 片玉/詰将棋データは KA2 部と同様スキップ (特徴集合の整合のため)
        if !board.king_square(stm).is_valid() || !board.king_square(nstm).is_valid() {
            return;
        }

        let pair_base = &PAIR_BASE_FULL.0;
        let threat_dims = PAIR_BASE_FULL.1;
        let from_offset_table = &*FROM_OFFSET_TABLE;
        let occ = Occupied::from_board(board);

        for sq_raw in 0..81u8 {
            let from_sq = Square(sq_raw);
            let pc = board.piece_on(from_sq);
            if pc.is_none() {
                continue;
            }
            let pt = pc.piece_type;
            let attacker_color = pc.color;
            if pt == PieceType::King {
                continue;
            }
            let attacker_class = match ThreatClass::from_piece_type(pt) {
                Some(c) => c,
                None => continue,
            };

            for_each_attack(pt, attacker_color, from_sq, &occ, |to_sq| {
                let target_pc = board.piece_on(to_sq);
                if target_pc.is_none() {
                    return;
                }
                if target_pc.piece_type == PieceType::King {
                    return;
                }
                let attacked_class = match ThreatClass::from_piece_type(target_pc.piece_type) {
                    Some(c) => c,
                    None => return,
                };
                let target_color = target_pc.color;

                // --- STM 視点 (HM ミラー無し) ---
                let stm_params = ThreatParams {
                    attacker_side: usize::from(attacker_color != stm),
                    attacker_class,
                    oriented_color: if stm == Color::Black { attacker_color } else { attacker_color.opponent() },
                    attacked_side: usize::from(target_color != stm),
                    attacked_class,
                    from_sq_n: normalize_sq(from_sq, stm, false),
                    to_sq_n: normalize_sq(to_sq, stm, false),
                };
                let Some(stm_idx) = threat_index(&stm_params, pair_base, from_offset_table) else {
                    return; // excluded pair (Full では発生しない)
                };

                // --- NSTM 視点 (HM ミラー無し) ---
                let nstm_params = ThreatParams {
                    attacker_side: usize::from(attacker_color != nstm),
                    attacker_class,
                    oriented_color: if nstm == Color::Black { attacker_color } else { attacker_color.opponent() },
                    attacked_side: usize::from(target_color != nstm),
                    attacked_class,
                    from_sq_n: normalize_sq(from_sq, nstm, false),
                    to_sq_n: normalize_sq(to_sq, nstm, false),
                };
                let Some(nstm_idx) = threat_index(&nstm_params, pair_base, from_offset_table) else {
                    return;
                };

                debug_assert!(stm_idx < threat_dims && nstm_idx < threat_dims);
                f(stm_idx, nstm_idx);
            });
        }
    }
}

impl SparseInputType for ShogiHalfKa2Threat {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        HALFKA2_THREAT_TOTAL_DIMENSIONS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_FEATURES + MAX_ACTIVE_THREAT_FEATURES
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        // Part 1: HalfKA2 (ShogiHalfKa2 と完全同一)
        map_halfka2_features(&board, &mut f);
        // Part 2: Threat (オフセット HALFKA2_DIMENSIONS)
        self.map_threat_only(&board, |stm_t, nstm_t| {
            f(HALFKA2_DIMENSIONS + stm_t, HALFKA2_DIMENSIONS + nstm_t)
        });
    }

    fn shorthand(&self) -> String {
        format!("shogi-halfka2-threat-{}", HALFKA2_THREAT_TOTAL_DIMENSIONS)
    }

    fn description(&self) -> String {
        format!(
            "Shogi HalfKA2 (131,949) + Threat ({}, profile=full)",
            HALFKA2_THREAT_DIMENSIONS
        )
    }
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn startpos_board() -> ShogiBoard {
        // 初期局面を手動構築 (shogi_halfka_hm_threat.rs のテストと同一手順)
        use crate::shogi::types::Piece;
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
    fn test_tables_checksum() {
        // YO 側 (threat.cpp) / bullet-shogi 側と照合する canonical 値。
        let cs = ShogiHalfKa2Threat.tables_checksum();
        assert_eq!(cs, 0x30f7eea2484893cd, "テーブル仕様が変わった (YO threat.cpp と再照合せよ)");
    }

    #[test]
    fn test_total_dimensions() {
        // full profile: 131,949 + 216,720 = 348,669
        let input = ShogiHalfKa2Threat;
        assert_eq!(input.threat_dimensions(), 216_720);
        assert_eq!(input.num_inputs(), 348_669);
    }

    #[test]
    fn test_startpos_active_counts() {
        // ★threat 数 34 は python-shogi (独立実装) の attackers() 列挙で照合済み (2026-08-20):
        //   香4 + 桂8 + 銀4 + 金4 + 角8 + 飛6 = 34 (玉が絡む関係は除外)。
        //   bullet-shogi ShogiHalfKPThreat のテストと同じ canonical 値。
        let input = ShogiHalfKa2Threat;
        let board = startpos_board();
        let mut th = 0usize;
        input.map_threat_only(&board, |_, _| th += 1);
        assert_eq!(th, 34, "初期局面の threat active 数が canonical 値と不一致");
    }

    #[test]
    fn test_indices_in_range() {
        // KA2 部・threat 部それぞれのレンジ検証 (KA2 部は map_halfka2_features を共有
        // しているので index 一致は構成的に保証される)
        let board = startpos_board();
        let mut n_ka2 = 0usize;
        map_halfka2_features(&board, |s, n| {
            assert!(s < HALFKA2_DIMENSIONS && n < HALFKA2_DIMENSIONS);
            n_ka2 += 1;
        });
        assert!(n_ka2 > 0);
        let input = ShogiHalfKa2Threat;
        let mut th = 0usize;
        input.map_threat_only(&board, |s, n| {
            assert!(s < input.threat_dimensions() && n < input.threat_dimensions());
            th += 1;
        });
        assert_eq!(th, 34);
    }


    #[test]
    fn test_dump_gold_faceoff_indices() {
        // 金当たり局面: 白金(4,4) 黒金(4,5) 王(4,0)/(4,8)。cross-side ペアの照合用 dump。
        use crate::shogi::types::Piece;
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);
        board.board[Square::new(4, 4).index()] = Piece::new(Color::White, PieceType::Gold);
        board.board[Square::new(4, 5).index()] = Piece::new(Color::Black, PieceType::Gold);
        let mut pairs = Vec::new();
        ShogiHalfKa2Threat.map_threat_only(&board, |s, n| pairs.push((s, n)));
        pairs.sort_unstable();
        println!("GOLD_FACEOFF_DUMP {:?}", pairs);
    }

    #[test]
    fn test_startpos_symmetry() {
        // 初期局面は先後対称なので、STM/NSTM の threat index 集合は一致するはず
        let input = ShogiHalfKa2Threat;
        let board = startpos_board();
        let mut stm_set = Vec::new();
        let mut nstm_set = Vec::new();
        input.map_threat_only(&board, |s, n| {
            stm_set.push(s);
            nstm_set.push(n);
        });
        stm_set.sort_unstable();
        nstm_set.sort_unstable();
        assert_eq!(stm_set, nstm_set);
    }
}

#[cfg(test)]
mod dump_env_tests {
    use super::*;
    use crate::shogi::{PackedSfenValue, ShogiBoard};

    /// THREAT_DUMP_PSV=<psv> で全レコードの threat index (stm/nstm, ソート済) を出力する
    /// (bullet-shogi 側の同名テストとの cross-crate 突き合わせ用)。
    #[test]
    fn dump_threat_indices_env() {
        let Ok(path) = std::env::var("THREAT_DUMP_PSV") else { return; };
        let bytes = std::fs::read(&path).expect("read psv");
        for (i, rec) in bytes.chunks_exact(40).enumerate() {
            let mut psv = PackedSfenValue::default();
            psv.as_bytes_mut().copy_from_slice(rec);
            let board = ShogiBoard::from_packed_sfen(&psv);
            let (mut s, mut n) = (Vec::new(), Vec::new());
            ShogiHalfKa2Threat.map_threat_only(&board, |a, b| { s.push(a); n.push(b); });
            s.sort_unstable(); n.sort_unstable();
            println!("R{} S:{:?} N:{:?}", i, s, n);
        }
    }
}
