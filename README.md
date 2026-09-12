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

## ビルドの注意

- `libddsc.so.0`（CycloneDDS）は cyclonedds-sys がビルドし、`cargo run` は
  パスを自動設定する。バイナリを直接叩くときは
  `LD_LIBRARY_PATH=target/<profile>/build/cyclonedds-sys-*/out` を通す。
- misa-policy-runner は**まだ path 依存**（`../misa-policy-runner` に
  チェックアウトを置く）。公開したら git 依存へ。
- rustc 1.94 では `cargo update kstring@2.0.4 --precise 2.0.2` が要る
  （Cargo.lock に記録済み）。

## まだやっていないこと

- 実機での歩容/WBC・policy の実走（このリポジトリはまだ机上 + `check` /
  `dump` / 単体テストまで）。実機手順: `check` → `bridge`（受信確認）→
  `policy --hold`（観測符号）→ 吊って `policy` → 接地。
- policy モードの MuJoCo 閉ループ検証（Python の
  `sim2sim_mit_go2_mujoco.py --natural-walk` と Rust 実装の突き合わせ）。
- misa-runner にコントローラ差し込み口（seam）が入ったら、`policy` の
  自前ループを `run` 系へ統合する。
