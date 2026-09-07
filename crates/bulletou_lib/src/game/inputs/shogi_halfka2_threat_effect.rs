//! HalfKA2 + ThreatEffect (長/短利き数バケット) 連結特徴量 (task#73 王者移植, 2026-09-07, report/52 §18.10)
//!
//! classic 512 で ThreatEffect (26,244 次元) が threat lite/full と互角 (L30 +2.9) かつ
//! YO 側で列挙なし差分更新 (LONG_EFFECT_LIBRARY) できることが確認できたので、王者 SFNN
//! (HalfKA2 1024-7-64) に同じ特徴を連結する。
//!
//! ## レイアウト
//!
//! `[KA2 131,949][ThreatEffect 26,244]` = 158,193 (+ KA virtual 1,629 = 訓練 159,822 行)。
//! KA2 部は `ShogiHalfKa2` と完全同一 index、effect 部は bullet-shogi `ShogiHalfKPThreatEffect`
//! (crate 004z) と**同一の index 式・利き定義**:
//!   feature = (attacker_side, defender_side, defender_class, to_sq, long 0/1/2+, short 0/1/2+)
//!   effect_index = (((as*18 + ds*9 + dc) * 81 + to_n) * 3 + lb) * 3 + sb
//! - 攻撃側は玉を含む (YO の board_effect と同定義)、被弾側は玉を除外。
//! - 長い利き = 香/角/飛の全射程 + 馬の斜め射程 + 龍の縦横射程 (隣接升を含む)、短い利き = それ以外の 1 歩。
//! - 集合意味論: 被弾駒 1 つ × 攻撃側 2 色 = 常に 2 特徴 ((0,0) も emit、重複なし)。
//! - 升の正規化は KA2 と同じく HM ミラー無し (`normalize_sq(sq, persp, false)`)。
//!
//! ## factorisation
//!
//! KA2 部のみ SFNN fast path の「暗黙 virtual rows」(feature < 131,949 → feature % 1,629) で factorise。
//! effect 部は factorise しない (halfka2t の threat 部と同じ扱い)。

use super::SparseInputType;
use super::shogi_halfka::{HALFKA2_DIMENSIONS, MAX_ACTIVE_FEATURES, map_halfka2_features};
use super::shogi_halfka_hm_threat::{Occupied, ThreatClass, normalize_sq};
use crate::shogi::{
    PackedSfenValue, ShogiBoard,
    types::{Color, PieceType, Square},
};

/// (as, ds, dc) 36 組 × to 81 × long 3 × short 3
pub const HALFKA2_THREATEFFECT_DIMENSIONS: usize = 2 * 2 * 9 * 81 * 3 * 3; // 26,244

/// HalfKA2 + ThreatEffect の総次元
pub const HALFKA2_THREATEFFECT_TOTAL_DIMENSIONS: usize = HALFKA2_DIMENSIONS + HALFKA2_THREATEFFECT_DIMENSIONS; // 158,193

/// 1 視点の threat-effect 特徴の上限 (玉以外の駒 ≤ 38 × 攻撃側 2 色)
const MAX_ACTIVE_EFFECT_FEATURES: usize = 80;

#[inline]
fn bucket(n: u8) -> usize {
    if n >= 2 { 2 } else { n as usize }
}

/// effect index: (attacker_side, defender_side, defender_class, to_n, long_bucket, short_bucket)
#[inline]
fn effect_index(attacker_side: usize, defender_side: usize, dc: ThreatClass, to_n: Square, lb: usize, sb: usize) -> usize {
    let group = attacker_side * 18 + defender_side * 9 + dc as usize;
    ((group * 81 + to_n.0 as usize) * 3 + lb) * 3 + sb
}

/// 駒 pt (color) が from から届かせる実利きを列挙し、(to, is_long) を返す。
/// for_each_attack (threat 系) と同じ移動規則だが、★玉も含み、長短の区別を返す。
/// bullet-shogi `shogi_halfkp_threat_effect.rs::for_each_effect` と同一。
fn for_each_effect<F: FnMut(Square, bool)>(pt: PieceType, color: Color, from: Square, occ: &Occupied, mut cb: F) {
    let file = (from.0 / 9) as i8;
    let rank = (from.0 % 9) as i8;
    let fwd: i8 = if color == Color::Black { -1 } else { 1 };
    let inb = |f: i8, r: i8| (0..9).contains(&f) && (0..9).contains(&r);
    let mut step = |df: i8, dr: i8, cb: &mut F| {
        let (f, r) = (file + df, rank + dr);
        if inb(f, r) {
            cb(Square::new(f as u8, r as u8), false);
        }
    };
    let mut ray = |df: i8, dr: i8, cb: &mut F| {
        let (mut f, mut r) = (file + df, rank + dr);
        while inb(f, r) {
            let sq = Square::new(f as u8, r as u8);
            cb(sq, true);
            if occ.is_occupied(sq.0) {
                break;
            }
            f += df;
            r += dr;
        }
    };
    match pt {
        PieceType::Pawn => step(0, fwd, &mut cb),
        PieceType::Lance => ray(0, fwd, &mut cb),
        PieceType::Knight => {
            step(-1, 2 * fwd, &mut cb);
            step(1, 2 * fwd, &mut cb);
        }
        PieceType::Silver => {
            for (df, dr) in [(-1, fwd), (0, fwd), (1, fwd), (-1, -fwd), (1, -fwd)] {
                step(df, dr, &mut cb);
            }
        }
        PieceType::Gold | PieceType::ProPawn | PieceType::ProLance | PieceType::ProKnight | PieceType::ProSilver => {
            for (df, dr) in [(-1, fwd), (0, fwd), (1, fwd), (-1, 0), (1, 0), (0, -fwd)] {
                step(df, dr, &mut cb);
            }
        }
        PieceType::King => {
            for (df, dr) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] {
                step(df, dr, &mut cb);
            }
        }
        PieceType::Bishop => {
            for (df, dr) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                ray(df, dr, &mut cb);
            }
        }
        PieceType::Rook => {
            for (df, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                ray(df, dr, &mut cb);
            }
        }
        PieceType::Horse => {
            for (df, dr) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                ray(df, dr, &mut cb);
            }
            for (df, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                step(df, dr, &mut cb);
            }
        }
        PieceType::Dragon => {
            for (df, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                ray(df, dr, &mut cb);
            }
            for (df, dr) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                step(df, dr, &mut cb);
            }
        }
        PieceType::None => {}
    }
}

/// HalfKA2 + ThreatEffect 連結特徴量 (unit struct、SFNN fast path の値渡し dispatch 用)
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKa2ThreatEffect;

impl ShogiHalfKa2ThreatEffect {
    /// 各升の利き数 (長/短、色別) を数える。返り値 [color][sq] = (long, short)
    pub(crate) fn effect_counts(board: &ShogiBoard) -> [[(u8, u8); 81]; 2] {
        let occ = Occupied::from_board(board);
        let mut cnt = [[(0u8, 0u8); 81]; 2];
        for sq_raw in 0..81u8 {
            let from = Square(sq_raw);
            let pc = board.piece_on(from);
            if pc.is_none() {
                continue;
            }
            let ci = usize::from(pc.color == Color::White);
            for_each_effect(pc.piece_type, pc.color, from, &occ, |to, is_long| {
                let e = &mut cnt[ci][to.0 as usize];
                if is_long {
                    e.0 = e.0.saturating_add(1);
                } else {
                    e.1 = e.1.saturating_add(1);
                }
            });
        }
        cnt
    }

    /// threat-effect 特徴のみを列挙 (index は effect 内オフセット 0..26,244)。重複 emit なし。
    pub(crate) fn map_effect_only<F: FnMut(usize, usize)>(&self, board: &ShogiBoard, mut f: F) {
        let stm = board.side_to_move;
        let nstm = stm.opponent();
        // 片玉/詰将棋データは KA2 部と同様スキップ (特徴集合の整合のため)
        if !board.king_square(stm).is_valid() || !board.king_square(nstm).is_valid() {
            return;
        }
        let cnt = Self::effect_counts(board);
        for sq_raw in 0..81u8 {
            let sq = Square(sq_raw);
            let pc = board.piece_on(sq);
            if pc.is_none() || pc.piece_type == PieceType::King {
                continue;
            }
            let dc = match ThreatClass::from_piece_type(pc.piece_type) {
                Some(c) => c,
                None => continue,
            };
            let vc = pc.color;
            for attacker_color in [Color::Black, Color::White] {
                let (l, s) = cnt[usize::from(attacker_color == Color::White)][sq_raw as usize];
                let (lb, sb) = (bucket(l), bucket(s));
                let stm_idx = effect_index(
                    usize::from(attacker_color != stm),
                    usize::from(vc != stm),
                    dc,
                    normalize_sq(sq, stm, false),
                    lb,
                    sb,
                );
                let nstm_idx = effect_index(
                    usize::from(attacker_color != nstm),
                    usize::from(vc != nstm),
                    dc,
                    normalize_sq(sq, nstm, false),
                    lb,
                    sb,
                );
                debug_assert!(stm_idx < HALFKA2_THREATEFFECT_DIMENSIONS && nstm_idx < HALFKA2_THREATEFFECT_DIMENSIONS);
                f(stm_idx, nstm_idx);
            }
        }
    }
}

impl SparseInputType for ShogiHalfKa2ThreatEffect {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        HALFKA2_THREATEFFECT_TOTAL_DIMENSIONS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_FEATURES + MAX_ACTIVE_EFFECT_FEATURES
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        // Part 1: HalfKA2 (ShogiHalfKa2 と完全同一)
        map_halfka2_features(&board, &mut f);
        // Part 2: ThreatEffect (オフセット HALFKA2_DIMENSIONS)
        self.map_effect_only(&board, |s, n| f(HALFKA2_DIMENSIONS + s, HALFKA2_DIMENSIONS + n));
    }

    fn shorthand(&self) -> String {
        format!("shogi-halfka2-threateffect-{}", HALFKA2_THREATEFFECT_TOTAL_DIMENSIONS)
    }

    fn description(&self) -> String {
        format!("Shogi HalfKA2 (131,949) + ThreatEffect long/short buckets ({})", HALFKA2_THREATEFFECT_DIMENSIONS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::types::Piece;

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
    fn test_dimensions() {
        assert_eq!(HALFKA2_THREATEFFECT_DIMENSIONS, 26_244);
        assert_eq!(ShogiHalfKa2ThreatEffect.num_inputs(), 158_193);
    }

    #[test]
    fn test_startpos_emit_count() {
        // 玉以外 38 駒 × 攻撃側 2 色 = 76 特徴 / 視点、重複なし、先後対称
        let board = startpos_board();
        let (mut s, mut n) = (Vec::new(), Vec::new());
        ShogiHalfKa2ThreatEffect.map_effect_only(&board, |a, b| { s.push(a); n.push(b); });
        assert_eq!(s.len(), 76);
        let mut u = s.clone(); u.sort_unstable(); u.dedup();
        assert_eq!(u.len(), 76, "重複 emit があってはならない (集合意味論)");
        s.sort_unstable(); n.sort_unstable();
        assert_eq!(s, n, "初期局面は先後対称");
    }

    #[test]
    fn test_effect_counts_startpos() {
        // bullet-shogi 側テストと同じ canonical 値
        let board = startpos_board();
        let cnt = ShogiHalfKa2ThreatEffect::effect_counts(&board);
        // 9七の歩 (file 0, rank 6): 9九の香の長い利き 1 本 + 8九の桂の短い利き 1 本
        assert_eq!(cnt[0][Square::new(0, 6).0 as usize], (1, 1));
        // 6九の金 (file 3, rank 8): 5九の玉の短い利きのみ
        assert_eq!(cnt[0][Square::new(3, 8).0 as usize], (0, 1));
        assert_eq!(cnt[1][Square::new(3, 8).0 as usize], (0, 0));
    }

    #[test]
    fn test_indices_in_range_and_ka2_shared() {
        let board = startpos_board();
        let mut n_ka2 = 0usize;
        map_halfka2_features(&board, |s, n| {
            assert!(s < HALFKA2_DIMENSIONS && n < HALFKA2_DIMENSIONS);
            n_ka2 += 1;
        });
        assert!(n_ka2 > 0);
        let mut total = 0usize;
        let psv = PackedSfenValue::default();
        // map_features は片玉判定を含むので default psv (空盤) では effect 部が出ない
        ShogiHalfKa2ThreatEffect.map_features(&psv, |s, n| {
            assert!(s < HALFKA2_THREATEFFECT_TOTAL_DIMENSIONS && n < HALFKA2_THREATEFFECT_TOTAL_DIMENSIONS);
            total += 1;
        });
        let _ = total;
    }

    /// EFFECT_DUMP_PSV=<psv> で全特徴 (KA2 + effect、オフセット込み) の (stm, nstm) index 列を出力
    /// (YO 側 THREAT_EFFECT_DUMP ビルドとの照合用。bullet-shogi 側の同名テストと同じ書式)
    #[test]
    fn dump_effect_indices_env() {
        let Ok(path) = std::env::var("EFFECT_DUMP_PSV") else { return; };
        let bytes = std::fs::read(&path).expect("read psv");
        for (i, rec) in bytes.chunks_exact(40).enumerate() {
            let mut psv = PackedSfenValue::default();
            psv.as_bytes_mut().copy_from_slice(rec);
            let (mut s, mut n) = (Vec::new(), Vec::new());
            ShogiHalfKa2ThreatEffect.map_features(&psv, |a, b| { s.push(a); n.push(b); });
            println!("F{} S:{:?} N:{:?}", i, s, n);
        }
    }
}
