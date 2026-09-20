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

## MuJoCo 閉ループ（go2.misa プラント、粘性 0.1、足摩擦 0.4、15 s、転倒なし）

**指令は体座標系なので追従も体座標系で測る。** 最初にここへ書いた
「横に −0.104 m/s 流れる」は世界系の平均を読んだ誤りで、方策がヨーに流れて
円弧を描くと前進が過小・横が過大に出る（namiashi で一度踏んだのと同じ罠）。
`policy --sim` は体座標系を主、世界系を参考として出すように直した。
下表はすべて体座標系。

| モデル | 指令 | v̄x | v̄y | ω̄z | tilt max |
|---|---|---|---|---|---|
| Pure76 Grip2（比較） | vx 0.3 | +0.325 (108 %) | +0.007 | −0.016 | 0.72° |
| GRU force10_stage2 | vx 0.3 | +0.297 (**99 %**) | +0.017 | **−0.058** | 2.46° |
| GRU lateral_escape | vx 0.3 | +0.301 (100 %) | +0.019 | −0.061 | 2.39° |
| GRU axis_state_timed | vx 0.3 | +0.301 (100 %) | +0.016 | −0.058 | 2.38° |
| GRU axis_force20 | vx 0.3 | +0.297 (99 %) | +0.019 | −0.060 | 2.36° |
| GRU force10_stage2 | vx 0.6 | +0.640 (107 %) | +0.017 | **−0.102** | 6.24° |
| GRU force10_stage2 | wz 0.4 | +0.006 | −0.012 | +0.150 (**37 %**) | 1.56° |
| GRU force10_stage2 | 停止 | +0.002 | −0.000 | +0.002 | 1.00° |

### Python 参照プラントとの突き合わせ

同じ ONNX・同じ指令・同じ粘性/摩擦で
`sim2sim_mit_go2_mujoco.py --pure76-gru --velocity-estimator`（go2_rl の
go2.xml シーン）と比べる:

| | Rust (go2.misa) | Python 参照 (go2.xml) |
|---|---|---|
| v̄x（体系） | +0.297 | +0.287 |
| v̄y（体系） | +0.017 | +0.015 |
| ω̄z | −0.058 | −0.057 |
| 変位 (x, y) | (+4.02, −1.60) m | (+3.92, −1.49) m |
| tilt max | 2.46° | 2.30° |

**プラントもランタイムも一致している。**（tick 単位のパリティ試験が同一入力
での一致を示し、閉ループのこの表が積分後の一致を示す。）

### 速度入力を 3 通りに振る（凍結推定器 / 脚オドメトリ / 真値）

`--estimator` / `--gru-host-velocity` / `GO2_SIM_TRUTH_VEL=1` で、観測 73:76 に
入る値だけを差し替える（cmd 0.3、15 s、他は同条件）。`policy --sim` は
**方策に実際に入った速度**も出すので、系統誤差がそのまま読める:

| 速度入力 | 入力の系統誤差 v̄x | 真値 v̄x | ω̄z |
|---|---|---|---|
| 凍結推定器（契約） | **+0.052**（+18 %） | +0.297 | −0.058 |
| 脚オドメトリ | **−0.071**（−22 %） | +0.319 | −0.068 |
| MuJoCo 真値 | 0.000 | +0.309 | −0.064 |

- **推定器は過大、脚オドメトリは過小**に読み、真値を両側から挟む。脚
  オドメトリの過小は接地足の滑り（Pure76 でも既知）、推定器の過大はこの
  プラントが学習分布と違うぶん。
- **ヨー流れは 3 通りとも −0.06 rad/s で変わらない。** 速度入力を完全な真値に
  してもヨーは直らないので、**ヨーバイアスの原因は速度入力ではなく方策側**。
  動画の HUD で推定器が 3 割過大に見えたのは事実だが、ヨーの犯人ではない。
- 前進追従は入力を ±20 % 誤らせても 99–106 % に収まる。実機では推定器の
  誤差の出方が sim とは違うはずなので、この鈍感さは素直に朗報。

### 読み取れること

- 前進追従は 99–108 %。**「78 %」「60 %」は世界系で読んだ誤りだった。**
- 実際の弱点は**ヨー**: 指令 0 で −0.058 rad/s（vx 0.6 では −0.102）流れ、
  wz 0.4 の指令には +0.150 rad/s（37 %）しか応えない。指令 0 のバイアスと
  指令への不足が同じ向きに揃っているので、原因は 1 つと見てよい。
- 押し耐性の系統（lateral_escape / axis_state_timed / axis_force20）は
  この指令域では親（force10_stage2）と挙動が同じ。外力を入れない限り差は
  出ない = 歩行性能を落とさずに入っている、とは言える。
- 停止は完全に静止（0.002 m/s、tilt 1.0°）。
- 採否は go2_rl 側の判断。ランタイムからの申し送りは「前進は出来ている、
  ヨーが残っている」。

## 実機へ出す前に

1. `--hold` で観測だけ見る（符号・単位・関節順）。
2. 吊るした状態で `--duration` 短めの空中試験。
3. 指令は 0 から。`--vx-max` を併用する（GRU の学習域上限は 2.0 m/s で、
   テレオペの数字キーは一気に飛ぶ）。
4. 横指令は効かない（0 に固定される）。旋回は ±0.8 まで。
