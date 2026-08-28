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
throughput: ★未確立。smoke の pos/s は train_elapsed (GPU step のみ、データ準備除外)
ベースで、かつ測定時 GPU は別の訓練と共有状態だった。30 step で 4.27M / 300 step で
445k と 10 倍ばらつく = 競合下の数字はどちら向きにも信用できない。
アイドル GPU・同一アーキ・同一データ・壁時計での実測はパリティ照合と同時に行う。

### ビルドの罠

`cargo test` は examples も**フィーチャー無しで再ビルド**し、
`--features cuda-cpp-backend` で作った exe を上書きする。テスト後は example を
作り直すこと (7.0MB → 1.1MB になっていたら feature 無し版)。

## パリティ照合の結果と追加修正 (2026-08-16)

同一データ (ryfamate quiet 28 shard, 240M 局面)・同一レシピ (batch 16384 / LR 0.001 固定 /
WRM 600,0,285,285,2 / decay 0.01 / beta1 0.99) で bullet-shogi (crate 004d) と 2sb 比較。
初回は **loss +10〜14% / holdout MSE +12〜28% の劣化**を検出した。切り分けの結果:

| 仮説 | 判定 |
|---|---|
| weight decay | ✗ (0 にしても不変) |
| beta1 0.9→0.99 | ✗ (0.3% のみ。ただし bullet と揃える意味で 0.99 推奨) |
| backward の仮想行散布 | ✗ 正常 |
| **init スケール** | △ sb1 の差を説明 → `--nnue-init bullet-kaiming` を追加 |
| RAdam の (1−β2^t) 欠落 | ✗ 原因ではないが標準形からの逸脱 → 修正 (下記) |
| データ順 | ✗ 両者とも厳密逐次 |
| Lookahead 式 | ✗ 同一 |
| TF32 | ✗ 既定 OFF |
| **`--use_fast_math`** | **★真因。** 近似 exp/sqrt/除算の系統誤差が収束品質を蝕む |

**修正一式 (全部込みで 004d 同等以上を確認: sb2 loss 0.983×、holdout 0.958〜0.982×):**

1. `crates/cuda_cpp/build.rs`: `--use_fast_math` を除去 (bullet-shogi は不使用)。
   ローカルではローダ律速のため wall throughput はほぼ不変 (688k→665k pos/s)。
2. `crates/cuda_cpp/src/lib.rs` `step_scale()`: RAdam の v バイアス補正 `(1−β2^t)` を追加
   (bullet-shogi / PyTorch と同形。upstream は欠落しており序盤ステップが最大 ~3 倍過大)。
3. `examples/bulletou.rs`: `--nnue-init bullet-kaiming` を追加
   (重み N(0, sqrt(2/fan_in)) / バイアス 0。既定 `tatara-simple` は不変)。
4. `crates/bulletou_lib/src/teacher_path.rs`: `.bin` 拡張子を PSV として受理
   (既存プールの全 shard が `.bin` のため)。**2 箇所**要る:
   `infer_data_format()` (単一ファイル/明示リスト用) と `TEACHER_EXTS`
   (ディレクトリ走査用)。前者だけ直すとディレクトリ指定で
   "no teacher files found" になる (実際に踏んだ)。
   ★1925 shard をカンマ連結すると 146,248 文字となり Windows のコマンドライン上限
   (32,767) を超えてプロセスが無言で起動しないので、大規模プールは
   **ディレクトリ指定**が必須。使う前に「走査結果 == 意図した shard 集合」を
   照合すること (今回は 1925 = 1925 差分ゼロを確認してから使った)。

★004d レシピを BulletOu で再現するときの必須フラグ:
`--nnue-init bullet-kaiming --optimizer-beta1 0.99 --optimizer-weight-decay 0.01
 --wrm-constants 600 0 285 285 2 --nnue-pytorch-wrm-loss --lambda 1.0`

## HalfKA2+Threat 入力 (SFNN_halfka2t) の追加 (2026-08-22, task#54)

Phase-1 (HalfKP+Threat, 512x2, +140.4 実測) を SFNN アーキに載せるための入力追加。
レイアウトは `[HalfKA2 131,949][Threat 216,720][KA virtual 1,629]` = 訓練時 350,298 行。
threat 部は factorise しない (multifactor 劣化 −393.3 の実測に基づく単因子方針)。

- `crates/bulletou_lib/src/game/inputs/shogi_halfka_hm_threat.rs`:
  bullet-shogi 側と同一の可視性変更 (`pub(super)` 化) のみ。テーブルロジックは
  upstream と不変 (checksum 0x30f7eea2484893cd で bullet-shogi / YO threat.cpp と一致検証)。
- `crates/bulletou_lib/src/game/inputs/shogi_halfka2_threat.rs` (新規):
  `ShogiHalfKa2Threat` (Full profile 固定 unit struct)。KA2 部は `map_halfka2_features`
  を共有、threat 部は既存テーブルを再利用。tables_checksum + startpos 34 テスト付き。
- `crates/bulletou_lib/src/game/inputs/shogi_halfka.rs`: `map_halfka2_features` を pub(super) に。
- `crates/bulletou_lib/src/value/fast_sfnn.rs`: `SFNN_HALFKA2T_*` 定数 +
  `halfka2_ft_factorized_virtual_feature` に threat 分岐 (KA2 部のみ virtual へ)。
- `crates/bulletou_lib/src/value/nnue_save.rs`: `NnueFeatureSet::HalfKa2Threat`。
  hash = `FEATURE_HASH_HALFKA2 ^ 0x54485254 ("THRT") ^ 0 (full)` = 0x0b6b1eec。
  YO 側 `FeatureSet<Threat, HalfKA2>` は Threat::kHashValue = 0xB52D879C で逆算一致
  (HalfKP 版 0xB3F22C9C と同じ手順、既知値で検算済)。
- `crates/cuda_cpp/cpp/bulletou_cuda_backend.cu`: `SFNN_HALFKA2T_*` constexpr、
  `sfnn_factorized_virtual_feature` に threat 分岐、backward の reduce kernel を
  (ka_rows, virtual_base) 引数化 (threat 行は virtual 勾配に寄与しない)。
- `examples/bulletou.rs`: arch `SFNN_halfka2t_<FT>_<H1>_<H2>[_k3k3...]`、
  `EvalType::SfnnHalfka2Threat` / `CudaCppSfnnFeatureKind::Halfka2Threat`、
  `ka_base_rows()`、fold を threat-aware 化 (KA2 行のみ virtual 畳み込み)。
  テスト 4 本 (parse / 次元 / 初期重みレイアウト / fold ミニチュア)。

## `--rescore-psv` モード (2026-08-22, task#56)

threat ラベラ自己蒸留 (report/52 §10) の推論経路。訓練済み SFNN net で PSV shard の
score を差し替える standalone モード。特徴写像 (`build_sfnn_validation_fast_batch`)・
fold (`cuda_cpp_sfnn_weights_for_cpu_validation`)・forward (`sfnn_forward_device`) は
訓練/検証と同一コードを使うため、train/infer パリティが構造的に保たれる。

- `examples/bulletou.rs`: `--rescore-psv <file|dir|list>` + `--rescore-out <dir>` +
  `run_cuda_cpp_sfnn_rescore` (SFNN 全 feature kind 対応)。
  cp = round(forward × `--scale`)、±29999 clamp、score 以外の 38 byte は不変。
  出力は .tmp → rename (アトミック)。
  使用例:
  `bulletou --arch SFNN_halfka2t_1024_7_64_k3k3 --backend cuda-cpp --teacher /dev/null      --cuda-cpp-weights-bin <ckpt>/cuda-cpp-direct/weights.bin      --rescore-psv D:/pool/dir --rescore-out D:/pool-rescored --test-batch-size 16384`

### 検証記録 (2026-08-22)

- 10,000 局面 rescore で **score 以外の 38 byte が全件不変**を numpy 照合で確認。
- 訓練済み sfnn-3way (240sb) の fp32 rescore vs YO (量子化 nn.bin, FV16) 15 局面:
  大 |cp| で比 1.19±0.03 に収束 = **定数の export 量子化スケール** + 小値の量子化ノイズ。
  機能的不一致なし。ラベルは訓練単位 (forward×600) が正であり、YO の表示 cp とは
  規約差 (FV16 で ~1.19 倍) がある — これは既知の FV 較正の話であり rescore のバグではない。
- スループット: debug ビルド + batch 4096 で 29k pos/s (10k 局面 0.34s)。
  release + batch 16384 で大幅向上見込み (GPU forward 自体は同じ)。

## 2026-08-29: HalfKA2+Threat + threat from-drop factoriser (`SFNN_halfka2tdf_*`, task#70)

classic threat-512 で from-drop factoriser が full に +43.7 有意 (shogi-nnue report/52 §18.2) だったので、王者 halfka2t にも移植した。

- `crates/bulletou_lib/src/game/inputs/shogi_halfka2_threat_dropfact.rs` (新規): `ShogiHalfKa2ThreatDropFact`。
  full threat index → lite (from-drop, `pair*81 + to`, 26,244 次元) の対応表を bullet-shogi `ShogiPThreatDrop` と同じ走査で構築し、
  threat 1 件ごとに lite 仮想 index (`350,298 + full_to_lite[t]`) を **CPU 側で明示 emit** する (重複そのまま = count 意味論)。
  訓練時の行レイアウトは `[KA2 131,949][Threat 216,720][KA virtual 1,629][Lite virtual 26,244]` = 376,542、`max_active` = 680。
- `crates/bulletou_lib/src/value/fast_sfnn.rs`: `SFNN_HALFKA2TDF_FT_FACTORIZED_INPUT_SIZE` と CPU 参照 forward の分岐 (暗黙 virtual は KA2 部のみ)。
- `crates/cuda_cpp/cpp/bulletou_cuda_backend.cu`: `SFNN_HALFKA2TDF_FACTORIZED_INPUT_SIZE`、`sfnn_factorized_virtual_feature` の分岐、
  inverse-index backward で lite 行は通常 gather (n_features = 全行)、KA virtual の reduce は halfka2t と同じ base で実行。新 kernel 無し。
- `examples/bulletou.rs`: arch 名 `halfka2tdf`、`EvalType::SfnnHalfka2ThreatDropFact`、`CudaCppSfnnFeatureKind::Halfka2ThreatDropFact`
  (`base_input_size` は 348,669 のまま = nn.bin の hash / 次元 / description は halfka2t と同一)、
  `fold_sfnn_halfka2_threat_dropfact_l0w` (KA2 行 += KA virtual、threat 行 += lite virtual)、検証バッチの上限を訓練行数に。
- 検証: lib/example テスト、bullet-shogi `ShogiHalfKPThreatLite` との lite index 多重集合照合 400 局面一致 (重複 561 件含む)。
- ビルド注意: .cu を再コンパイルする環境では `NVCC_APPEND_FLAGS=-allow-unsupported-compiler` (上記参照)。
