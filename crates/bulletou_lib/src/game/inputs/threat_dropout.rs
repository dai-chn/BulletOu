//! threat dropout (lazy threat A1 の同時学習、2026-10-02、report/52 §21.18)
//!
//! サンプル単位で確率 p で「threat 行 (index >= threat_begin) を全部落とした」特徴集合を emit する入力アダプタ。
//! 王者の FT は threat 行と KA2 行が共適応していて、threat 平面を落とすと評価値が平均 +184 cp ずれ、互角帯で符号が
//! 46% 反転する (nnuedump v3 eval_ka2)。学習時に一部のサンプルを threat 無しで見せると、同じ下流層が KA2 平面だけでも
//! 読める方向に FT/下流が動く。推論側 (YaneuraOu) は threat 平面を落とした値を絶対閾値ゲート (|v| > T) に使い、
//! 決定できない局面だけ threat 行を適用する。export される nn.bin の形式は変わらない。
//!
//! 落とす/落とさないの決定はサンプルの通し番号 (loader スレッド共有の atomic カウンタ) の splitmix64 ハッシュで行う
//! (位置に依存させると同じ局面が常に同じ扱いになるため)。factoriser の仮想行は KA2 部だけなので影響しない。
use std::sync::atomic::{AtomicU64, Ordering};

use super::SparseInputType;

static SAMPLE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[derive(Clone)]
pub struct ThreatDropout<T: Clone> {
    pub inner: T,
    /// この index 以上が threat 行 (HalfKA2Threat なら HALFKA2_DIMENSIONS)
    pub threat_begin: usize,
    /// 落とす確率 (Q16 固定小数、0..=65536)
    pub prob_q16: u32,
    pub seed: u64,
}

impl<T: Clone> ThreatDropout<T> {
    pub fn new(inner: T, threat_begin: usize, prob: f32, seed: u64) -> Self {
        let prob_q16 = (prob.clamp(0.0, 1.0) * 65536.0).round() as u32;
        Self { inner, threat_begin, prob_q16, seed }
    }

    #[inline]
    fn drop_this_sample(&self) -> bool {
        if self.prob_q16 == 0 {
            return false;
        }
        let n = SAMPLE_COUNTER.fetch_add(1, Ordering::Relaxed);
        (splitmix64(n ^ self.seed.rotate_left(17)) & 0xFFFF) < u64::from(self.prob_q16)
    }
}

impl<T: SparseInputType> SparseInputType for ThreatDropout<T> {
    type RequiredDataType = T::RequiredDataType;

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        if !self.drop_this_sample() {
            self.inner.map_features(pos, f);
            return;
        }
        let tb = self.threat_begin;
        self.inner.map_features(pos, |stm, nstm| {
            if stm < tb && nstm < tb {
                f(stm, nstm);
            }
        });
    }

    fn map_features_split<F: FnMut(Option<usize>, Option<usize>)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        if !self.drop_this_sample() {
            self.inner.map_features_split(pos, f);
            return;
        }
        let tb = self.threat_begin;
        self.inner.map_features_split(pos, |stm, nstm| {
            let keep = |x: Option<usize>| x.filter(|&i| i < tb);
            let (s, n) = (keep(stm), keep(nstm));
            if s.is_some() || n.is_some() {
                f(s, n);
            }
        });
    }

    fn max_active(&self) -> usize {
        self.inner.max_active()
    }

    fn num_inputs(&self) -> usize {
        self.inner.num_inputs()
    }

    fn description(&self) -> String {
        format!("{} + threat dropout p={:.3}", self.inner.description(), f64::from(self.prob_q16) / 65536.0)
    }

    fn shorthand(&self) -> String {
        self.inner.shorthand()
    }

    fn hand_count_dims(&self) -> Option<usize> {
        self.inner.hand_count_dims()
    }

    fn fill_hand_count(&self, pos: &Self::RequiredDataType, out: &mut [f32]) {
        self.inner.fill_hand_count(pos, out)
    }

    fn is_factorised(&self) -> bool {
        self.inner.is_factorised()
    }

    fn merge_factoriser(&self, unmerged: Vec<f32>) -> Vec<f32> {
        self.inner.merge_factoriser(unmerged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct Fake;
    impl SparseInputType for Fake {
        type RequiredDataType = u8;
        fn num_inputs(&self) -> usize {
            1000
        }
        fn max_active(&self) -> usize {
            4
        }
        fn map_features<F: FnMut(usize, usize)>(&self, _pos: &u8, mut f: F) {
            f(5, 7);
            f(12, 3);
            f(500, 600);
            f(900, 901);
        }
        fn shorthand(&self) -> String {
            "fake".into()
        }
        fn description(&self) -> String {
            "fake".into()
        }
    }

    fn collect<I: SparseInputType<RequiredDataType = u8>>(i: &I) -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        i.map_features(&0, |a, b| v.push((a, b)));
        v
    }

    #[test]
    fn prob_one_drops_only_threat_rows() {
        let d = ThreatDropout::new(Fake, 100, 1.0, 1);
        for _ in 0..8 {
            assert_eq!(collect(&d), vec![(5, 7), (12, 3)]);
        }
    }

    #[test]
    fn prob_zero_is_identity() {
        let d = ThreatDropout::new(Fake, 100, 0.0, 1);
        assert_eq!(collect(&d), collect(&Fake));
        assert_eq!(d.num_inputs(), 1000);
        assert_eq!(d.max_active(), 4);
    }

    #[test]
    fn prob_half_drops_about_half() {
        let d = ThreatDropout::new(Fake, 100, 0.5, 7);
        let dropped = (0..4000).filter(|_| collect(&d).len() == 2).count();
        assert!((1600..=2400).contains(&dropped), "dropped {dropped} of 4000");
    }
}
