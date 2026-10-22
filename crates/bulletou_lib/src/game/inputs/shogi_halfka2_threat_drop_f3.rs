//! HalfKA2 + Threat + DropF3 (安全な打ち当たり) 連結特徴量
//!
//! 設計: shogi-nnue `docs/superpowers/specs/2026-10-23-drop-f3-feature-design.md` §3。
//! やねうら王 fork `eval/nnue/features/drop_f3.{h,cpp}` と**同一 index 仕様**。
//!
//! 視点 c の accumulator に、色 c の持ち駒 p を升 s に打つと相手の駒 (クラス vc) に当たる関係の行を入れる:
//! - 当たる相手が玉なら (p, s, 玉) を立てる (s に相手の利きがあっても = 王手の打ち)
//! - 当たる相手が歩 (不成) なら立てない
//! - それ以外は s に相手の利き (玉を含む) が無いときだけ立てる (打った駒がただで取られない)
//!
//! s は空き升で、歩・香は最終段、桂は最終 2 段、歩は二歩を除く (打ち歩詰め・王手回避は見ない)。
//! index = p (歩香桂銀金角飛 = 0..6) * 729 + s_n (81、後手視点は inverse) * 9 + vc (香 0 桂 1 銀 2 金類 3 角 4 馬 5 飛 6 龍 7 玉 8)。
//! 次元 7 * 81 * 9 = 5,103。(p, s) ごとに vc 昇順で重複なし。
//!
//! ## 非対称 emission
//! STM 視点には STM の打ちだけ、NSTM 視点には NSTM の打ちだけを入れるので |STM_active| != |NSTM_active|。
//! `map_features_split` で片側ずつ emit する (hand_threat_defensive と同じ流儀)。`map_features` (対称 API) は
//! KA2 + Threat 部分だけを emit する best-effort。
//!
//! レイアウト: [KA2 131,949][Threat 216,720][DropF3 5,103] = 353,772 (訓練時は KA 仮想行 1,629 を足して 355,401)。

use super::SparseInputType;
use super::shogi_halfka::{HALFKA2_DIMENSIONS, MAX_ACTIVE_FEATURES, map_halfka2_features};
use super::shogi_halfka_hm_threat::{Occupied, for_each_attack};
use super::shogi_halfka2_threat::{HALFKA2_THREAT_DIMENSIONS, HALFKA2_THREAT_TOTAL_DIMENSIONS, ShogiHalfKa2Threat};
use super::shogi_halfka2_threat_effect::ShogiHalfKa2ThreatEffect;
use crate::shogi::{
    PackedSfenValue, ShogiBoard,
    types::{Color, PieceType, Square},
};

/// DropF3 の次元数 (7 p × 81 s × 9 vc)
pub const DROP_F3_DIMENSIONS: usize = 7 * 81 * 9;

/// HalfKA2 + Threat + DropF3 の総入力次元 (131,949 + 216,720 + 5,103 = 353,772)
pub const HALFKA2_THREAT_DROPF3_TOTAL_DIMENSIONS: usize = HALFKA2_THREAT_TOTAL_DIMENSIONS + DROP_F3_DIMENSIONS;

/// active threat features の安全上限 (shogi_halfka2_threat と同じ)
const MAX_ACTIVE_THREAT_FEATURES: usize = 320;

/// active DropF3 features の上限 (片視点。実測の最大は両視点合計 150)
pub const MAX_ACTIVE_DROPF3_FEATURES: usize = 192;

/// 持ち駒の順 (index の p)。★変更禁止 (やねうら王側と不一致になる)
const HAND_ORDER: [PieceType; 7] = [
    PieceType::Pawn,
    PieceType::Lance,
    PieceType::Knight,
    PieceType::Silver,
    PieceType::Gold,
    PieceType::Bishop,
    PieceType::Rook,
];

/// 当たる駒のクラス vc。歩は None (行を立てない)
#[inline]
fn target_class(pt: PieceType) -> Option<usize> {
    match pt {
        PieceType::Lance => Some(0),
        PieceType::Knight => Some(1),
        PieceType::Silver => Some(2),
        PieceType::Gold | PieceType::ProPawn | PieceType::ProLance | PieceType::ProKnight | PieceType::ProSilver => Some(3),
        PieceType::Bishop => Some(4),
        PieceType::Horse => Some(5),
        PieceType::Rook => Some(6),
        PieceType::Dragon => Some(7),
        PieceType::King => Some(8),
        _ => None, // Pawn, None
    }
}

/// 行きどころの無い駒の判定 (歩・香は最終段、桂は最終 2 段)
#[inline]
fn drop_rank_ok(pt: PieceType, color: Color, sq: Square) -> bool {
    let rank = sq.rank();
    match pt {
        PieceType::Pawn | PieceType::Lance => {
            if color == Color::Black { rank != 0 } else { rank != 8 }
        }
        PieceType::Knight => {
            if color == Color::Black { rank >= 2 } else { rank <= 6 }
        }
        _ => true,
    }
}

/// `color` の不成の歩がある筋のビット集合 (二歩判定)
fn pawn_files(board: &ShogiBoard, color: Color) -> u16 {
    let mut files = 0u16;
    for sq_raw in 0..81u8 {
        let pc = board.piece_on(Square(sq_raw));
        if !pc.is_none() && pc.color == color && pc.piece_type == PieceType::Pawn {
            files |= 1 << Square(sq_raw).file();
        }
    }
    files
}

/// HalfKA2 + Threat + DropF3 連結特徴量 (unit struct、SFNN fast path の値渡し dispatch 用)
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKa2ThreatDropF3;

impl ShogiHalfKa2ThreatDropF3 {
    /// 色 `color` の DropF3 index (DropF3 内オフセット 0..5,103) を、p 昇順 → s 昇順 (回転前の升番号) → vc 昇順で列挙する。
    /// `enemy_attacked[sq]` = 相手 (¬color) の利きが sq にあるか (玉を含む)。
    pub(crate) fn for_each_drop_f3<F: FnMut(usize)>(board: &ShogiBoard, color: Color, occ: &Occupied, enemy_attacked: &[bool; 81], mut f: F) {
        let hand = board.hand(color);
        let opp = color.opponent();
        let mut own_pawn_files: Option<u16> = None;
        for (pi, &pt) in HAND_ORDER.iter().enumerate() {
            if hand.count(pt) == 0 {
                continue;
            }
            for sq_raw in 0..81u8 {
                let s = Square(sq_raw);
                if !board.piece_on(s).is_none() {
                    continue;
                }
                if !drop_rank_ok(pt, color, s) {
                    continue;
                }
                if pt == PieceType::Pawn {
                    let files = *own_pawn_files.get_or_insert_with(|| pawn_files(board, color));
                    if files & (1 << s.file()) != 0 {
                        continue;
                    }
                }
                let mut mask: u16 = 0;
                let safe = !enemy_attacked[sq_raw as usize];
                for_each_attack(pt, color, s, occ, |t| {
                    let tpc = board.piece_on(t);
                    if tpc.is_none() || tpc.color != opp {
                        return;
                    }
                    let Some(vc) = target_class(tpc.piece_type) else { return };
                    if vc == 8 {
                        mask |= 1 << 8; // 王手の打ち: 安全性を問わない
                    } else if safe {
                        mask |= 1 << vc;
                    }
                });
                if mask == 0 {
                    continue;
                }
                let s_n = if color == Color::Black { s } else { s.inverse() };
                for vc in 0..9usize {
                    if mask & (1 << vc) != 0 {
                        let idx = pi * 729 + usize::from(s_n.0) * 9 + vc;
                        debug_assert!(idx < DROP_F3_DIMENSIONS);
                        f(idx);
                    }
                }
            }
        }
    }

    /// DropF3 部分の非対称 emission (index は DropF3 内オフセット)。STM の打ちは (Some, None)、NSTM の打ちは (None, Some)。
    pub(crate) fn map_drop_f3_only<F: FnMut(Option<usize>, Option<usize>)>(&self, board: &ShogiBoard, mut f: F) {
        let stm = board.side_to_move;
        let nstm = stm.opponent();
        // 片玉/詰将棋データは KA2 部と同様スキップ (特徴集合の整合のため)
        if !board.king_square(stm).is_valid() || !board.king_square(nstm).is_valid() {
            return;
        }
        let occ = Occupied::from_board(board);
        let cnt = ShogiHalfKa2ThreatEffect::effect_counts(board);
        let mut attacked = [[false; 81]; 2];
        for ci in 0..2 {
            for sq in 0..81 {
                attacked[ci][sq] = cnt[ci][sq] != (0, 0);
            }
        }
        // 視点 c には色 c の打ちだけ。その安全性は ¬c の利きで決まる。
        let nstm_ci = usize::from(nstm == Color::White);
        Self::for_each_drop_f3(board, stm, &occ, &attacked[nstm_ci], |idx| f(Some(idx), None));
        let stm_ci = usize::from(stm == Color::White);
        Self::for_each_drop_f3(board, nstm, &occ, &attacked[stm_ci], |idx| f(None, Some(idx)));
    }
}

impl SparseInputType for ShogiHalfKa2ThreatDropF3 {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        HALFKA2_THREAT_DROPF3_TOTAL_DIMENSIONS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_FEATURES + MAX_ACTIVE_THREAT_FEATURES + MAX_ACTIVE_DROPF3_FEATURES
    }

    /// 対称 API (後方互換用 best-effort): KA2 + Threat 部分だけを emit する。DropF3 は非対称なので
    /// 通常の経路は `map_features_split` を使う。
    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        map_halfka2_features(&board, &mut f);
        ShogiHalfKa2Threat.map_threat_only(&board, |s, n| f(HALFKA2_DIMENSIONS + s, HALFKA2_DIMENSIONS + n));
    }

    fn map_features_split<F: FnMut(Option<usize>, Option<usize>)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        // Part 1: HalfKA2 (対称)
        map_halfka2_features(&board, |s, n| f(Some(s), Some(n)));
        // Part 2: Threat (対称、オフセット HALFKA2_DIMENSIONS)
        ShogiHalfKa2Threat.map_threat_only(&board, |s, n| f(Some(HALFKA2_DIMENSIONS + s), Some(HALFKA2_DIMENSIONS + n)));
        // Part 3: DropF3 (非対称、オフセット HALFKA2_THREAT_TOTAL_DIMENSIONS)
        self.map_drop_f3_only(&board, |s, n| {
            f(s.map(|x| HALFKA2_THREAT_TOTAL_DIMENSIONS + x), n.map(|x| HALFKA2_THREAT_TOTAL_DIMENSIONS + x))
        });
    }

    fn shorthand(&self) -> String {
        format!("shogi-halfka2-threat-dropf3-{}", HALFKA2_THREAT_DROPF3_TOTAL_DIMENSIONS)
    }

    fn description(&self) -> String {
        format!(
            "Shogi HalfKA2 (131,949) + Threat ({}) + DropF3 safe drop attacks ({}) (asymmetric)",
            HALFKA2_THREAT_DIMENSIONS, DROP_F3_DIMENSIONS
        )
    }
}
