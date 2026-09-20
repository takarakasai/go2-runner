# GRU 契約の実行系（go2-runner / misa-policy-runner）

2026-09-20。go2_rl `doc/gru_runtime_contract_audit_20260920.md` が挙げた
**非対応・不一致の 4 点を実装で潰した**記録。対象は
`outputs/history_gru_v1/curriculum_force10_stage2_v1/model_49.onnx` と
凍結推定器 `outputs/history_high_speed_stability_v1/estimator/estimator_42.onnx`。

## 何が足りていなかったか → 何を入れたか

| audit の指摘 | 入れたもの |
|---|---|
| `OnnxPolicy` は 1 入力 1 出力。隠れ状態を持てない | `misa_policy_runner::policy::RecurrentOnnxPolicy`（2 入力 2 出力、形は batch 1 に束縛） |
| 速度 73:76 に接地脚オドメトリを供給していた | `gru::HistoryVelocityEstimator`（6×73 → 3 の凍結グラフ）を契約の内側に持つ |
| リセットは前回行動だけ | `PureGruController::reset` が隠れ状態・前回行動・速度履歴を同時に落とす |
| 指令域が Pure の固定クランプ | `clamp_gru_cmd` = vx [−0.16, 2.0] / **vy ≡ 0** / wz ±0.8（チェックポイントの `env.yaml`） |
| 入力幅 76 だけで Pure と GRU を判別 | `graph_input_arity` で**入力の本数**から判別（1 本 = MLP、2 本 = GRU） |

推論は tract。GRU オペレータはそのまま通り、**推論 29 µs（中央値）/ 101 µs
(p99)** — 50 Hz の枠 20 ms に対して桁で余る（x86 実測。aarch64 は未測定）。

## リセット規則（契約の一部）

隠れ状態を持つランタイムでは「いつゼロに戻すか」が契約になる。
`PureGruController::reset()` は次を**同時に**行う:

- 隠れ状態 128 → 0
- 前回行動（観測 37:73）→ 0
- 速度履歴 → 破棄（次の tick が現在フレーム 6 枚で埋め直す）

`policy` / `policy --sim` は既定姿勢へのランプ直後、方策ループに入る前に
これを呼ぶ。**推論が失敗した tick では何も進めない** — 隠れ状態も履歴も
据え置き、直前の指令を保持して次の tick でやり直す（壊れた隠れ状態は以後
ずっと尾を引くため）。

## パリティ試験（Python 参照 ⇄ Rust、tick 単位）

プラントを挟まず、**同一の観測列**を両ランタイムに通して obs 76 / action 36
/ hidden 128 を全部突き合わせる。プラント差が混ざらないので、差が出たら
それはランタイムの差。

```bash
G=~/work/dp/go2_rl
python3 scripts/gru_parity_reference.py \
  --model $G/outputs/history_gru_v1/curriculum_force10_stage2_v1/model_49.onnx \
  --estimator $G/outputs/history_high_speed_stability_v1/estimator/estimator_42.onnx \
  --generate /tmp/frames.txt --ticks 400 --out /tmp/python.txt

cargo run --release --manifest-path ../misa-policy-runner/Cargo.toml \
  --example gru_parity -- \
  --model ... --estimator ... --frames /tmp/frames.txt --out /tmp/rust.txt

python3 scripts/gru_parity_reference.py --model ... --estimator ... \
  --frames /tmp/frames.txt --compare /tmp/rust.txt
```

結果（400 tick、隠れ状態を通しで持ち回り）:

| ブロック | max \|diff\| |
|---|---|
| observation (76) | 1.22e−5 |
| action (36) | 1.38e−5 |
| hidden (128) | 8.19e−6 |

tick 0 で 1.8e−7、平均 4.1e−6、tick 399 で 1.4e−5 — **増えるが発散しない**。
PyTorch→ONNX のエクスポート誤差自体が 1.4e−6（`model_49.json`）なので、
これは onnxruntime と tract の f32 の差が再帰で積む分。

既定姿勢（観測の関節オフセット）は Python 側も go2_rl の
`mit_rl/natural_walk.py` から計算しているので、この試験は**軌道 + IK の
Rust 移植のズレも観測差として拾う**（今回は上表のとおり出ていない）。

## MuJoCo 閉ループ（go2.misa プラント、粘性 0.1、足摩擦 0.8）

`policy --sim`、15 s、転倒なし:

| 指令 | 真値 v̄x | v̄y | ω̄z | |roll|max / |pitch|max |
|---|---|---|---|---|
| 停止 | +0.002 | −0.000 | +0.002 | 1.00° / 0.50° |
| vx 0.3（推定器） | +0.235 (78 %) | **−0.104** | −0.055 | 1.56° / 2.09° |
| vx 0.3（`--gru-host-velocity`） | +0.245 | **−0.144** | −0.070 | 1.35° / 2.15° |
| vx 0.6（推定器） | +0.359 (60 %) | **−0.440** | −0.107 | 2.14° / 2.69° |
| vx 0.3（推定器、足摩擦 0.4） | +0.263 | **−0.123** | −0.058 | 1.78° / 2.46° |

最後の行は `--friction` が実際に効くようになってから取り直したもの
（それ以前の `--friction` は足に届いていなかった。README の切り分け記録を
参照）。**横流れは摩擦を半分にしても消えない** ので、接地の滑りではない。

同じプラント・同じ条件で Pure76（Grip2）は vx 0.3 → **+0.310（103 %）、
v̄y −0.035**。つまり横流れは**このプラントでの GRU チェックポイントの性質**で、
ランタイムの不一致ではない（上のパリティ試験が同一入力での一致を示している）。
Python 参照プラント（go2_rl の go2.xml）では audit が vx 0.287 / vy +0.015 を
測っており、**プラント間で横流れの出方が違う**。

→ **この model_49 は運用採用しない。** 49 iteration の stage-2 チェックポイント
で、横指令は学習していない（vy ≡ 0）。採用の可否は go2_rl 側で、
プラント差込みの横流れを潰してから。ランタイム側は契約どおりで、
別の GRU チェックポイントをそのまま受けられる。

## 実機へ出す前に

1. `--hold` で観測だけ見る（符号・単位・関節順）。
2. 吊るした状態で `--duration` 短めの空中試験。
3. 指令は 0 から。`--vx-max` を併用する（GRU の学習域上限は 2.0 m/s で、
   テレオペの数字キーは一気に飛ぶ）。
4. 横指令は効かない（0 に固定される）。旋回は ±0.8 まで。
