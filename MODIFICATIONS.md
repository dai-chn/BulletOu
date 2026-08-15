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

## 検証記録 (2026-08-15)

### HalfKP factorization は「暗黙」実装で完備 — 追加実装不要と確認

調査の結論: `ft_factorize: false` は正しい設定だった。HalfKP fast trainer は
**常に factorized (input 126936 = 125388 + 1548)** で訓練し、CUDA カーネル
`nnue_halfkp_factorized_feature` が raw index から base (=1548+feat) と
virtual (=feat%1548) を**暗黙に両方 gather** する。batch 側は raw index のみでよい。
save は `fold_halfkp_piece_factorized_l0w` (fold → 量子化の順、単体テスト付き) で
125388 行へ畳んでから nn.bin に書く。

### export 経路の独立数値検証: PASS

30 step の smoke 訓練 (512x2-16-32, `--wrm-constants 600 0 285 285 2`) で export した
nn.bin を、shogi-nnue 側の独立実装 (`tools/nnue_eval.py` の整数忠実 forward) と
実配布エンジン (KisouEngine.exe) で照合し **60/60 局面が ±2cp 一致**。
implicit factorization → fold → 量子化 → nn.bin の全経路が数値的に正しい。
throughput 参考値: 4.27M pos/s (RTX 5060 Ti, 30 step の短時間測定)。

### ビルドの罠

`cargo test` は examples も**フィーチャー無しで再ビルド**し、
`--features cuda-cpp-backend` で作った exe を上書きする。テスト後は example を
作り直すこと (7.0MB → 1.1MB になっていたら feature 無し版)。

## 未着手 (作業メモ)

- loss レベルのパリティ照合 (同一データ・同一レシピで既存学習器 004d と
  loss 曲線・最終重みを比較) — GPU が空き次第 (768 訓練後)。
