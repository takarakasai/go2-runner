# go2-runner

Unitree Go2 を **misa-runner** で動かす実行ファイル。モデルベース
（quadruped-gait の歩容 + WBC）と RL ベース（**misa-policy-runner** の
MIT モード方策）の両方を 1 本で持つ。namiashi-runner2 / keel-runner と
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

`build_sim.sh` は MuJoCo 3.8.0 を `MUJOCO_DYNAMIC_LINK_DIR` →
`$MUJOCO_HOME/lib` → `~/.mujoco/mujoco-3.8.0/lib` の順で探し、**どこにも無ければ
`sim-autodownload` feature で `~/.mujoco` へ自動ダウンロードして**ビルドする
（要ネットワーク、約 5 MB）。`cargo build --features sim` を直に叩くと
mujoco-rs のビルドスクリプトが置き場を見つけられず
`failed to run custom build command for mujoco-rs` で落ちるので、sim を使う
ときはこのスクリプト経由にするか `MUJOCO_DYNAMIC_LINK_DIR` を自分で export する。

学習済み ONNX はリポジトリに入っていないので別途コピーする
（`~/work/dp/go2_rl/logs/rsl_rl/<実験名>/<run>/exported/policy.onnx`）。

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

## Pure 契約（ネットワークのみ）

`policy` / `policy --sim` は **ONNX の入力幅で契約を自動判別**する:

| 入力幅 | 契約 | ランタイムが持つもの |
|---|---|---|
| 39 | Natural | リファレンス軌道 + IK + 残差 + 30 ms フィルタ + τ_ff |
| 73 | Pure | ネットワーク + アフィンデコードのみ（前回行動を帰還） |
| 76 | Pure + 体速度 | 上に体座標系の速度推定（脚オドメトリ）を追加入力 |

Pure 契約では τ_ff は送らず（ゼロ）、既定姿勢は 0.30 m のしゃがみ、指令域は
vx ∈ [−0.16, 1.0] / vy ±0.10 / wz ±0.40 にクランプされる。脚オドメトリの接地
マスクは歩容計画が無いので足裏力センサ（実機）/ 接触センサ（sim）で測る。

```bash
# 1 m/s まで対応する Pure76（ネットワーク 1 枚だけの契約）
./scripts/policy_sim.sh ~/work/dp/go2_rl/logs/rsl_rl/go2_mit_pure_vel_speed/2026-09-13_00-02-36_pure76_speed100_lr1e4/exported/policy.onnx
```

go2.misa は粘性ダンピング 2.0 の**悲観**プラントなので、そこでの到達速度は
指令の 7〜8 割で、cmd 1.0 は転倒する（Python 側の同条件プラントと一致する
既知の限界。現実的な粘性 0.1 のモデルでは 1.0 → 1.04 m/s）。

診断: `GO2_SIM_TRUTH_VEL=1` で 76 入力の体速度を MuJoCo の真値に差し替え、
脚オドメトリ誤差の影響を切り分けられる（実測: 真値 0.199 に対し推定 0.064
と 3 倍ずれても歩行結果はほぼ不変）。
