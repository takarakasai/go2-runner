# go2-runner

Unitree Go2 を **misa-runner** で動かす実行ファイル。モデルベース
（quadruped-gait の歩容 + WBC）と RL ベース（**misa-policy-runner** の
MIT モード方策）の両方を 1 本で持つ。namiashi-runner2 / hayaashi-runner と
同じ「薄い main + 機体固有 Backend」の形。

```
機体固有のもの:
  robots/go2.toml        プロファイル（kind = "ros2"、mit_gains、歩容の包絡）
  models/unitree_go2     go2.misa とメッシュ（go2-gait-runner から複製）
  src/go2_plant.rs       rt/lowcmd / rt/lowstate の DDS ブリッジ（misa Plant）
  src/backend.rs         Go2Backend（run/stance/bridge に差さる）+ sport_mode 解除
  src/policy.rs          RL 方策の 500 Hz ループ（50 Hz 推論 + 毎周期 τ_ff）
  src/estimator.rs       支持脚レンチ用の脚オドメトリ（高さ・世界系速度）
```

## モデルベース（misa-runner のサブコマンドそのまま）

```bash
cargo run -- check --robot robots/go2.toml          # 設定とモデルの検証
cargo run -- dump  --robot robots/go2.toml --gait trot --vx 0.1
cargo run --features sim -- sim --robot robots/go2.toml --pilot keys   # MuJoCo
GO2_IFACE=eth0 cargo run -- run --robot robots/go2.toml --pilot keys   # 実機
```

- Backend が呼ばれるのは `kind = "ros2"` のとき（misa-runner の仕様）。
  Go2 の低レベルは素の CycloneDDS なので分類として正しい。
- **NIC は `GO2_IFACE` 環境変数**（プロファイルに置き場が無い）。
- sport_mode は接続時に自動解除。`GO2_NO_RELEASE=1` で抑止。
- 位置指令は `[hardware.mit_gains]` の kp/kd で MIT に写る。**実機の歩容は
  未検証** — まず `dump` → `sim` → 吊り上げた状態の `run` の順で。

## RL ベース（`policy` サブコマンド）

```bash
GO2_IFACE=eth0 cargo run --release -- policy \
  --model /path/to/go2_rl/logs/rsl_rl/go2_mit_natural_h30/<run>/exported/policy.onnx
```

Natural 契約（go2_rl `doc/mit_natural.md`）を misa-policy-runner が実装:
50 Hz 推論 + 500 Hz MIT（q_ref/Kp/Kd）+ 毎周期の支持脚レンチ τ_ff。
既定は **NaturalH30 標準**（stride 1.55 / yaw 2.0 / height 0.30 m）。
チェックポイントを替えるときは `--stride-gain / --yaw-stride-gain /
--body-height を学習時の値に合わせる**こと（違っても歩く。悪く。
エラーは出ない）。

- `--hold`: 方策を走らせず既定姿勢を保持し観測だけ表示（符号の実機確認）。
- W/S = vx、A/D = wz、R/F = vy、Space = 停止、q/Esc = 伏せて終了。
- 安全側: 観測異常・推論失敗は直前指令の保持（10 連続で中断）、
  傾き 35° で即脱力。終了は常に伏せ姿勢へのランプ経由。

## RL ポリシーを MuJoCo で回す（`policy --sim` + articara 可視化）

実機と**同じ** NaturalController・脚オドメトリ・デコードを MuJoCo
（misa-plant-mujoco、物理 2 ms × 10 = 50 Hz 推論）で閉ループにする。
articara へは planned（指令、ゴースト）と measured（MuJoCo 実測）の
2 ストリームを Zenoh で配信する。

```bash
# 端末 1: シミュレーション + キーボード操縦 + 配信
./scripts/policy_sim.sh            # 既定 = NaturalH30 標準チェックポイント

# 端末 2: articara（viz 付きビルド）でモデルを開いて購読
cd ../articara
MUJOCO_DYNAMIC_LINK_DIR=$HOME/.mujoco/mujoco-3.8.0/lib \
  cargo run --release --features viz -- --model ../go2-runner/models/unitree_go2/go2.misa
#   GUI の「Live feed (Zenoh)」ウィンドウ:
#   キーは既定のまま（go2/gait/planned / go2/gait/measured）、
#   topology = Connect、endpoint = tcp/127.0.0.1:7447 で Subscribe を ON
```

キー: W/S = 前後、A/D = 旋回、R/F = 横、Space = 停止、q/Esc = 終了。
ヘッドレス検証は `--no-keyboard --duration S --vx V`（実測 2026-09-12:
vx 0.12 指令で 10 s に +1.10 m、高さ 0.298 m 維持、転倒なし —
Python `sim2sim_mit_go2_mujoco.py --natural-walk` と同等の挙動）。

## ビルドの注意

- `libddsc.so.0`（CycloneDDS）は cyclonedds-sys がビルドし、`cargo run` は
  パスを自動設定する。バイナリを直接叩くときは
  `LD_LIBRARY_PATH=target/<profile>/build/cyclonedds-sys-*/out` を通す。
- misa-policy-runner は**まだ path 依存**（`../misa-policy-runner` に
  チェックアウトを置く）。公開したら git 依存へ。
- rustc 1.94 では `cargo update kstring@2.0.4 --precise 2.0.2` が要る
  （Cargo.lock に記録済み）。

## まだやっていないこと

- 実機での歩容/WBC・policy の実走（MuJoCo 閉ループまでは検証済み）。
  実機手順: `check` → `bridge`（受信確認）→ `policy --hold`（観測符号）
  → 吊って `policy` → 接地。
- misa-runner にコントローラ差し込み口（seam）が入ったら、`policy` の
  自前ループを `run` 系へ統合する。

## 別 PC での立ち上げ（依存の用意）

```bash
git clone git@github.com:takarakasai/go2-runner.git && cd go2-runner
cargo build --release                 # 実機のみ（MuJoCo 不要）
./scripts/build_sim.sh                # MuJoCo 閉ループつき（--features sim）
```

依存はすべて git（misa-runner / misa-policy-runner / articara /
unitree-sdk-rs）なので、隣にチェックアウトを置く必要は無い。
misa-policy-runner を更新したときは
`cargo update -p misa-policy-runner` で Cargo.lock を進める。

### 受動粘性の既定は 0.1（Unitree 公式値）

`policy --sim` は go2.misa を読むが、その `damping = 2.0` は MuJoCo
Menagerie 由来の保守値で、**Unitree 公式の MuJoCo モデル**
（[unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) の
`unitree_robots/go2/go2.xml`）は `damping = 0.1`。armature 0.01 /
frictionloss 0.2 / ctrlrange 23.7・45.43 / cone elliptic / impratio 100 は
両者一致で、この 1 点だけ 20 倍違う。学習側（Isaac）も受動粘性ゼロなので、
**既定をメーカー値 0.1 に寄せた**（起動時に「受動粘性を 0.1 に差し替えました」
と出る）。Menagerie の保守値で頑健性を見たいときは `--joint-damping 2.0`。

これにより Pure 契約の到達速度が上がったので、`|vx|` の自動上限（0.6）は
粘性を下げているときは掛からない。

```bash
./scripts/policy_sim.sh <policy.onnx>                    # 粘性 0.1（既定）
./scripts/policy_sim.sh <policy.onnx> --joint-damping 2.0 # 従来の悲観プラント
```

`build_sim.sh` は MuJoCo 3.8.0 を `MUJOCO_DYNAMIC_LINK_DIR` →
`$MUJOCO_HOME/lib` → `~/.mujoco/mujoco-3.8.0/lib` の順で探し、**どこにも無ければ
`sim-autodownload` feature で `~/.mujoco` へ自動ダウンロードして**ビルドする
（要ネットワーク、約 5 MB）。`cargo build --features sim` を直に叩くと
mujoco-rs のビルドスクリプトが置き場を見つけられず
`failed to run custom build command for mujoco-rs` で落ちるので、sim を使う
ときはこのスクリプト経由にするか `MUJOCO_DYNAMIC_LINK_DIR` を自分で export する。

学習済み ONNX はリポジトリに入っていないので別途コピーする
（`~/work/dp/go2_rl/logs/rsl_rl/<実験名>/<run>/exported/policy.onnx`）。
引数を省いたときの既定は下の**推奨チェックポイント**で、置き場は
`GO2_POLICY` でも上書きできる。見つからなければ起動前に止まる。

### viz のポートが埋まっているとき

`--viz-endpoint` は **go2-run 側が待ち受ける**エンドポイントなので、その
ポートが空いている必要がある（articara は Connect 側）。埋まっていると
起動時に原因と逃げ道つきで止まる。掴んでいる相手は

```bash
ss -tlnp | grep 7447        # または lsof -i :7447
```

で判る（`ps aux | grep 7447` ではポート番号はプロセス名に出ないので見つからない）。
よくあるのは **ROS 2 の rmw_zenoh ルータ `rmw_zenohd`**（7447 は zenoh ルータの
既定ポート。ROS 2 を使うなら止めずに別番号へ逃がす）、前回の go2-run の残り、
articara の Live feed を Listen 側にしている。別番号で逃げるなら:

```bash
GO2_VIZ_ENDPOINT=tcp/127.0.0.1:7448 ./scripts/policy_sim.sh   # articara 側も 7448 に
```

### テレオペで前進すると転ぶとき

**粘性 2.0 のまま**（`--joint-damping 2.0`）だと go2.misa は出せる速度が低い。
実測（Pure 契約 Rival、2026-09-13）:

| 指令 vx | 粘性 2.0 の go2.misa |
|---|---|
| 0.3 | 真値 0.20 m/s、安定 |
| 0.6 | 真値 0.46 m/s、安定（横流れあり） |
| 0.7 | 歩くが横に大きく流れる |
| 0.8 以上 | 転倒 |

そのため粘性を下げていないときだけ **|vx| を 0.6 に自動で抑える**（Pure 契約の
ように学習域がそれより広い場合のみ。起動時にその旨を表示する）。変更・解除は
`--vx-max V`。実機側では抑えない。Natural H30 契約は学習域自体が 0.16 なので
影響しない。

**既定（粘性 0.1）ではこの制限は掛からず、上限まで出る。** 推奨
チェックポイント Grip2 の実測（go2.misa、粘性 0.1、12 s、`--no-keyboard`）:

| 指令 vx | Grip2 真値（オドメトリ） | Rival 真値 | 横流れ Grip2 / Rival |
|---|---|---|---|
| 0.3 | 0.311（0.232） | — | — |
| 0.6 | 0.618（0.520） | — | −1.46 m |
| 0.8 | **0.792**（0.693） | 0.736 | −2.41 / −3.24 m |
| 1.0 | **0.957**（0.855） | 0.736 | −3.36 / **−6.50** m |

いずれも転倒せず、傾きは |roll| 1.9° / |pitch| 1.7° 以内。その場旋回
（wz 0.8）は 12 s で変位 0.02 m = **本当にその場**、横移動（vy 0.2）は
12 s で 2.29 m = 0.19 m/s。**Grip2 は高速側で Rival より速く、横流れが半分**。

オドメトリが真値より 10〜25 % 低いのは接地足の滑りによる系統的な過小評価で、
これが 76 入力方策の 74〜76 番目に入る。Grip2 はこの誤差も Rival より小さい。

原因の切り分け記録:

- **受動粘性は主因ではない。** `--joint-damping` で 2.0 / 0.5 / 0.2 / 0.1 と
  下げても cmd 1.0 では 1〜3 s で転倒した（go2.misa の 2.0 は MuJoCo
  Menagerie 由来の保守値で、実機 Go2 が 2.5 m/s 以上出る事実とは両立しない
  が、ここでの限界を決めているのはこれではない）。
- **接触モデルは効いた。** 既定の `pyramidal` / `impratio = 1` では接地足が
  滑る（misa-plant-mujoco の `SimOptions::impratio` のコメントどおり）。
  Python 参照と同じ `cone = elliptic` / `impratio = 100` / 足摩擦
  `0.8 0.02 0.01` に合わせたところ、Natural H30 の速度追従が 112 % → 101〜
  103 %、横流れが −0.12 m → +0.01 m（12 s）になり Python 側の数値と一致した。
  `--impratio` / `--cone` / `--friction` で触れる。

  **`--friction` は 2026-09-20 まで足に届いていなかった**（μ 0.1 と 0.8 で
  軌跡がビット単位で一致する、という形で出た）。`SimOptions::friction` は
  MJCF の `<default><geom friction=…/>` にしか入らないのに、go2.misa の足
  geom は `friction = [0.8, 0.02, 0.01]` を明示したうえ `priority = 1` を
  持つため、per-geom 値が `<default>` を上書きし接触ペアの摩擦を単独で決める。
  いまは `--friction` が一時 .misa 側の slide 成分（4 面）を書き換える。
  **これ以前に Rust sim で取った摩擦スイープの結果は無効**（.misa の 0.8 で
  測っていたことになる）。Python 参照プラント側の摩擦知見は影響を受けない。
- 1 m/s は**実機で確認する話**。Python 参照プラント（現実的な粘性）では同じ
  ONNX が 1.04 m/s を出している（go2_rl `doc/mit_pure.md`）。

## Pure 契約（ネットワークのみ）

`policy` / `policy --sim` は **ONNX のグラフ形状で契約を自動判別**する。
まず入力の**本数**、次に幅を見る:

| 入力 | 契約 | ランタイムが持つもの |
|---|---|---|
| 1 本 × 39 | Natural | リファレンス軌道 + IK + 残差 + 30 ms フィルタ + τ_ff |
| 1 本 × 73 | Pure | ネットワーク + アフィンデコードのみ（前回行動を帰還） |
| 1 本 × 76 | Pure + 体速度 | 上に体座標系の速度推定（脚オドメトリ）を追加入力 |
| 2 本 × (76, 128) | GRU | 上に**隠れ状態**と**凍結速度推定器**（下記） |

**幅だけで Pure76 と GRU を見分けてはいけない** — どちらも観測 76 で、隠れ
状態・速度の出どころ・指令域がまるごと違う。だから判別は入力の本数から入る。

Pure 契約では τ_ff は送らず（ゼロ）、既定姿勢は 0.30 m のしゃがみ、指令域は
vx ∈ [−0.16, 1.0] / vy ±0.30 / wz ±0.80 にクランプされる（`clamp_pure_cmd`。
これは**契約ではなく推奨チェックポイントの学習域**で、ONNX 側にメタ情報は
無い）。脚オドメトリの接地マスクは歩容計画が無いので足裏力センサ（実機）/
接触センサ（sim）で測る。

### 推奨チェックポイント: Grip2（既定）

```bash
./scripts/policy_sim.sh    # 引数なしで Grip2 を読む
```

`go2_rl/logs/rsl_rl/go2_mit_pure_vel_grip2/2026-09-14_00-12-30_grip2_norm/exported/policy.onnx`
（76 入力 / 36 出力、パリティ 1.9e-6）。Python 参照プラント（Unitree 公式値の
粘性 0.1 / 足摩擦 0.4）での実測は go2_rl `doc/mit_pure.md`:

| 指令 | 到達 | 接触点滑り |
|---|---|---|
| vx 0.3 | 0.303（101 %） | 0.074（25 %） |
| vx 0.6 | 0.614（102 %） | 0.136（22 %） |
| vx 1.0 | 1.001（100 %） | 0.216（22 %） |
| wz 0.8（その場旋回） | 0.623（78 %） | 0.084 |
| vy 0.2 | 0.198（99 %） | 0.091 |

前任の Rival から**接触点滑りが半減**しており（速度は落ちていない）、
レッグオドメトリの系統誤差が小さい = 76 番目の入力が汚れにくい。
滑りをさらに減らした Grip3 もあるが、静止から指令 1.0 を階段状に入れると
倒れかけるので**実機・テレオペには使わない**（数字キーは一気に飛ぶ）。

go2.misa は粘性ダンピング 2.0 の**悲観**プラントなので、そこでの到達速度は
指令の 7〜8 割で、cmd 1.0 は転倒する（Python 側の同条件プラントと一致する
既知の限界。現実的な粘性 0.1 のモデルでは 1.0 → 1.04 m/s）。

診断: `GO2_SIM_TRUTH_VEL=1` で 76 入力の体速度を MuJoCo の真値に差し替え、
脚オドメトリ誤差の影響を切り分けられる（実測: 真値 0.199 に対し推定 0.064
と 3 倍ずれても歩行結果はほぼ不変）。

## GRU 契約（再帰方策）

```bash
cargo run --release -- policy \
  --model  <run>/model_49.onnx \
  --estimator <run>/estimator_42.onnx
```

go2_rl `export_gru_policy.py` が出す**2 入力 2 出力**のグラフ
（`observation[1,76]` + `hidden_in[1,1,128]` → `action[1,36]` +
`hidden_out[1,1,128]`）。Pure76 との違いは 4 つで、どれも外せない:

1. **ランタイムが状態を持つ**。前周期の `hidden_out` が次周期の `hidden_in`。
   リセット時は隠れ状態・前回行動・速度履歴を**同時に**ゼロへ戻す。
2. **観測 73:76 の速度は凍結推定器が作る** — 脚オドメトリではない。
   73 次元（基底 37 + 前回行動 36）を 6 フレーム、古い順に並べた 438 を
   `--estimator` のグラフへ通す。起動直後は現在フレームを 6 枚で埋める。
3. **指令域は vx ∈ [−0.16, 2.0] / vy ≡ 0 / wz ±0.8**（採用チェックポイントの
   `env.yaml`）。横指令は**学習していない**ので 0 に固定する（悪くなるので
   はなく、見たことのない入力になる）。
4. τ_ff は無し、既定姿勢は Pure と同じ 0.30 m のしゃがみ、デコードも同じ
   （`q = default + 0.25a`、`Kp = 45 + 12.5a`、`Kd = 2 + 0.5a`）。

`--gru-host-velocity` で推定器の代わりに脚オドメトリを使える。**契約から
外れる診断用**（別の入力分布になる。go2_rl の audit で MuJoCo 挙動が変わる
ことが測られている）。`--estimator` との併用はエラー。

契約の詳細・パリティ試験の手順・実測値は `doc/gru_deploy.md`。
