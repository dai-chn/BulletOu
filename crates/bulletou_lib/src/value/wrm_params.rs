//! Win-Rate-Model (nnue-pytorch 系) 損失のパラメータ化。
//!
//! upstream BulletOu は WRM の定数を nodchip nnue-pytorch の値
//! (nnue2score=600, offset=270, in_scaling=340, out_scaling=380, pow=2.5) に
//! ハードコードしていた。shogi-nnue プロジェクトの本番レシピは
//! `offset=0, in_scaling=out_scaling=285, pow=2` (exp004 系。offset=0 のとき
//! WRM は `sigmoid(score/285)` に代数的に縮退する) なので、そのまま使うと
//! **静かにレシピが変わる**。ここで実行時パラメータに昇格させる。
//!
//! - 既定値は nodchip 定数のまま → 何も設定しなければ upstream と同一挙動。
//! - `set_wrm_params()` は **データローダ/損失の初回使用より前に 1 回だけ**呼ぶ。
//!   ターゲット変換テーブル (`loader::win_rate_model_score`) は OnceLock で
//!   キャッシュされるため、使用後の変更は反映されない。誤用は panic で検出する。
//!
//! GPU カーネル側 (`bulletou_cuda_cpp`) には同名の setter があり、
//! trainer 起動時に両方へ同じ値を渡す (examples/bulletou.rs 参照)。

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// WRM 損失の全パラメータ。
///
/// ネット側:   `q  = sigmoid((output*nnue2score − offset) / in_scaling)`
/// ターゲット側: `p = sigmoid((teacher_score − offset) / out_scaling)`
/// 損失:       `|prediction − target|^pow_exp`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WrmParams {
    pub nnue2score: f32,
    pub offset: f32,
    pub in_scaling: f32,
    pub out_scaling: f32,
    pub pow_exp: f32,
}

impl Default for WrmParams {
    /// nodchip nnue-pytorch の定数 (= upstream BulletOu のハードコード値)。
    fn default() -> Self {
        Self { nnue2score: 600.0, offset: 270.0, in_scaling: 340.0, out_scaling: 380.0, pow_exp: 2.5 }
    }
}

static WRM_PARAMS: OnceLock<WrmParams> = OnceLock::new();
static WRM_PARAMS_USED: AtomicBool = AtomicBool::new(false);

/// 現在の WRM パラメータ。未設定なら nodchip 既定値。
///
/// 呼んだ時点で「使用済み」となり、以後の `set_wrm_params` は panic する
/// (テーブルキャッシュ済みの値と食い違う設定を静かに握り潰さないため)。
pub fn wrm_params() -> WrmParams {
    WRM_PARAMS_USED.store(true, Ordering::SeqCst);
    WRM_PARAMS.get().copied().unwrap_or_default()
}

/// WRM パラメータを設定する。データローダ/損失の初回使用より前に 1 回だけ。
///
/// # Panics
/// 既にどこかが `wrm_params()` を読んだ後 (= 古い値でテーブル等が構築済み) や、
/// 二重設定のとき。
pub fn set_wrm_params(params: WrmParams) {
    assert!(
        !WRM_PARAMS_USED.load(Ordering::SeqCst),
        "set_wrm_params must be called before the first use of wrm_params() \
         (the target-conversion table is cached with the old values)"
    );
    assert!(WRM_PARAMS.set(params).is_ok(), "set_wrm_params must be called at most once");
}

/// ターゲット側変換: 教師 score → 期待勝率 (WRM)。
/// `offset=0` のとき `sigmoid(score/out_scaling)` に縮退する。
pub fn wrm_target(params: &WrmParams, score: f32) -> f32 {
    let p = sigmoid((score - params.offset) / params.out_scaling);
    let pm = sigmoid((-score - params.offset) / params.out_scaling);
    0.5 * (1.0 + p - pm)
}

/// ネット側の損失と勾配 (純関数、パラメータ明示版)。
///
/// upstream の `nnue_pytorch_wrm_loss_and_gradient` と同じ数式で、定数だけを
/// `params` に置き換えたもの。既定パラメータでの数値一致は fast_loss.rs の
/// テストが upstream 由来の期待値で確認する。
pub fn wrm_loss_and_gradient(params: &WrmParams, output: f32, target: f32) -> (f32, f32) {
    let scorenet = output * params.nnue2score;
    let q = sigmoid((scorenet - params.offset) / params.in_scaling);
    let qm = sigmoid((-scorenet - params.offset) / params.in_scaling);
    let prediction = (1.0 + q - qm) * 0.5;
    let error = prediction - target;
    let abs_error = error.abs();
    let loss = abs_error.powf(params.pow_exp);
    let q_prime = q * (1.0 - q);
    let qm_prime = qm * (1.0 - qm);
    let prediction_gradient = 0.5 * (params.nnue2score / params.in_scaling) * (q_prime + qm_prime);
    let loss_gradient = params.pow_exp * error.signum() * abs_error.powf(params.pow_exp - 1.0);
    (loss, loss_gradient * prediction_gradient)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// shogi-nnue 本番レシピ (offset=0, in=out=285, pow=2)。
    fn ours() -> WrmParams {
        WrmParams { nnue2score: 600.0, offset: 0.0, in_scaling: 285.0, out_scaling: 285.0, pow_exp: 2.0 }
    }

    /// offset=0 のとき WRM は素の sigmoid に縮退する (qm = 1 − q)。
    /// 損失・勾配とも sigmoid-MSE (scale = nnue2score/in_scaling) と一致すること。
    #[test]
    fn offset_zero_reduces_to_sigmoid_mse() {
        let p = ours();
        let inv_scale = p.nnue2score / p.in_scaling;
        for &(output, target) in
            &[(0.0f32, 0.5f32), (0.3, 0.9), (-0.7, 0.1), (1.5, 0.0), (-2.0, 1.0), (0.05, 0.55)]
        {
            let (loss, grad) = wrm_loss_and_gradient(&p, output, target);
            let pred = sigmoid(output * inv_scale);
            let err = pred - target;
            let mse_loss = err * err;
            let mse_grad = 2.0 * err * pred * (1.0 - pred) * inv_scale;
            assert!((loss - mse_loss).abs() < 1e-6, "loss {loss} != {mse_loss} at ({output},{target})");
            assert!((grad - mse_grad).abs() < 1e-5, "grad {grad} != {mse_grad} at ({output},{target})");
        }
    }

    /// ターゲット側も offset=0 で sigmoid(score/285) に縮退。
    #[test]
    fn target_offset_zero_reduces_to_sigmoid() {
        let p = ours();
        for &score in &[0.0f32, 100.0, -100.0, 285.0, -1000.0, 3000.0] {
            let t = wrm_target(&p, score);
            let s = sigmoid(score / p.out_scaling);
            assert!((t - s).abs() < 1e-6, "target {t} != sigmoid {s} at score {score}");
        }
    }

    /// 既定パラメータ = nodchip 定数 (upstream との互換性の要)。
    #[test]
    fn default_is_nodchip() {
        let d = WrmParams::default();
        assert_eq!(
            (d.nnue2score, d.offset, d.in_scaling, d.out_scaling, d.pow_exp),
            (600.0, 270.0, 340.0, 380.0, 2.5)
        );
    }
}
