# Modifications from upstream BulletOu

Upstream: https://github.com/yaneurao/BulletOu (branch `shogi-support`)
Base commit: `6251eef` (2026-08 取得)
License: MIT (`LICENSE`, Copyright (c) 2023 Jamie Whiting)

shogi-nnue プロジェクト (Kisou Engine) の学習器として使うための改変。
MIT なので開示義務は無いが、YaneuraOu fork と同じ規律で改変を記録する。

## WRM 損失のパラメータ化 (2026-08-15)

upstream は WRM (nnue-pytorch 系) の定数を nodchip 値にハードコードしていた:
`nnue2score=600, offset=270, in_scaling=340 (ネット側), out_scaling=380 (ターゲット側), pow=2.5`。

shogi-nnue の本番レシピは `offset=0, in_scaling=out_scaling=285, pow=2`
(offset=0 のとき WRM は `sigmoid(score/285)` に代数的に縮退する)。
そのまま使うと**静かにレシピが変わる**ため、実行時パラメータへ昇格させた。
**既定値は nodchip 定数のまま** — 何も指定しなければ upstream と同一挙動。

- `crates/bulletou_lib/src/value/wrm_params.rs` (新規):
  `WrmParams` + set-once グローバル + 純関数 (`wrm_target` / `wrm_loss_and_gradient`)。
  縮退性のテスト付き (offset=0 で sigmoid-MSE と損失・勾配が一致すること)。
- `crates/bulletou_lib/src/value/fast_loss.rs`: CPU golden 実装が `wrm_params()` を読む形に。
- `crates/bulletou_lib/src/value/loader.rs`: ターゲット変換テーブル (270/380 固定だった) を
  パラメータ化。テーブルは OnceLock キャッシュなので **set は初回データロードより前**
  (誤用は panic で検出)。
- `crates/bulletou_lib/src/validate.rs`: 検証側 3 箇所を同じパラメータで。
- `crates/cuda_cpp/cpp/bulletou_cuda_backend.cu`:
  `loss_nnue_pytorch_wrm_reduce_kernel` の constexpr をカーネル引数化。
  extern C 3 本 (`scalar_loss_device_with_finalize` / `_device` / `_host`) と
  内部 launcher に 4 引数 (`wrm_nnue2score/offset/in_scaling/pow_exp`) を追加。
- `crates/cuda_cpp/src/lib.rs`: `set_wrm_loss_params()` (set-once、既定 = nodchip)。
  公開 Rust API の署名は不変 (呼び出し側の変更ゼロ)。
- `examples/bulletou.rs`: `--wrm-constants n2s,off,in,out,pow` を追加し、
  bulletou_lib と cuda_cpp の両方へ同時に設定する。

## ビルド注意 (このマシン固有)

CUDA 12.8 の nvcc は MSVC 2026 (VS18) を弾く。
`NVCC_APPEND_FLAGS=-allow-unsupported-compiler` を付けてビルドする
(既存の動作実績あり)。

## 未着手 / 既知の穴 (作業メモ)

- `ft_factorize` (HalfKP piece-factorizer): **ライブラリには完備**
  (`ShogiHalfKPPieceFactorizer` + `Factorised` 結線 + `merge_factoriser` の単体テスト) だが、
  trainer CLI からは到達不能 (`examples/bulletou.rs` でハードコード false)。
  ★有効化する前に **quantised save が仮想行を fold するか**の数値検証が必須
  (shogi-nnue 側で「ハッシュ通過・対局可能・ただ弱い」export 事故の前例がある)。
- 移行のパリティ照合 (同一データ・同一レシピで既存学習器と loss/重みを照合) は未実施。
