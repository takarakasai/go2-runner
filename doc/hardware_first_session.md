# 実機 1 回目のセッション手順（Go2）

SBC が用意できた日に、**この順で 1 セッションに収める**ためのチェックリスト。
溜まっている未測定は 3 件で、どれも数分だが**前提が他の結果に効く**ので順番が
ある。所要はおよそ 40 分（機体の組み立て・ネットワークを除く）。

すべて go2_rl / go2-runner の既存成果物で、**新しく作るものは無い**。

## 0. 前提の確認（5 分）

```bash
ping <GO2_IP>                      # SBC へ到達するか
ssh unitree@<GO2_IP> uname -m      # aarch64 であること
```

機体は**吊るすか、伏せた状態から始める**。`policy` は既定姿勢へ 3 s ランプして
から歩き出すので、床に置くなら周囲 2 m を空ける。

## 1. IMU ヨーのドリフト（静止 5 分、10 分）

**最初にやる。** 方位サーボ（`--heading-hold`）の参照が漂うと、ドリフトは
1:1 で実際の曲率になる。要件は **1.0 m/s で ≤ 9°/min（0.3 m/s なら 30°/min）**。

```bash
ssh unitree@<GO2_IP>
cd ~/go2_gru_latency          # §2 の一式を先に送っておいてもよい
go2-run policy --hold --duration 300 --no-keyboard 2>&1 | tee yaw_drift.log
```

`--hold` は方策を走らせず既定姿勢を保持して観測だけ出す。ログの `tilt` と
状態表示からヨーの変化を読む。**機体は完全に静止させること**（台の上でよい）。

判定: 5 分で **45°（0.3 m/s 用）を超えたら方位サーボは実機では使えない**。
その場合は補正の参照を IMU ヨーではなく別の源（オドメトリ融合など）にする
必要があり、それ自体が別の課題になる。

## 2. GRU 推論のレイテンシ（5 分）

`go2_gru_latency.tar.gz`（送付済み。再生成は misa-policy-runner
`scripts/build_aarch64.sh --example gru_parity`）。静的リンクなので SBC 側に
依存は要らない。

```bash
scp -r go2_gru_latency unitree@<GO2_IP>:~/
ssh unitree@<GO2_IP> 'cd ~/go2_gru_latency && ./gru_parity \
    --model model_49.onnx --estimator estimator_42.onnx \
    --frames frames.txt --out rust_aarch64.txt'
```

判定: 最後の行の `inference … µs median / … p99` が **50 Hz の枠 20 000 µs** に
対してどれだけ余るか。参考値（x86、同じ frames.txt）は **28 µs / 39 µs**。
p99 が 2 000 µs（枠の 10 %）を超えるようなら、方策の推論だけで余裕が無い。

持ち帰った `rust_aarch64.txt` は x86 と数値比較できる:

```bash
python3 scripts/gru_parity_reference.py --model … --estimator … \
    --frames frames.txt --compare rust_aarch64.txt
```

## 3. 推定器バイアスの実測（静止、10 分）

sim で測った **+0.069 m/s** は**プラント固有**なので、実機の値は実機で測る。
**静止させて、推定器が返す vx を読む**（静止なら真値は 0 なのでバイアスそのもの）。

```bash
go2-run policy --hold --duration 60 --no-keyboard \
    --model <gru.onnx> --estimator <estimator.onnx> 2>&1 | tee est_bias.log
```

`policy --sim` と同じく「方策への速度入力」を出すので、その平均がバイアス。

判定: 得られた値を `--estimator-bias VALUE` に入れる。**静止中は引かないこと**
（方策が後退と誤認して前に這う。実装済みだが、ログで確認する）。

## 4. 歩かせる（15 分）

上の 3 件が済んでから。**指令は 0 から**、`--vx-max` を併用する。

```bash
go2-run policy --model <policy.onnx> [--estimator …] \
    --estimator-bias <§3 の値> --heading-hold 2 0.5 --vx-max 0.3
```

- 指令のレート制限は既定で入る（`--cmd-accel 0.5`）。**外さないこと** —
  sim では 1 tick のステップで 0.9 m/s から転んでいる。
- 異常・推論失敗は直前指令の保持（10 連続で中断）、傾き 35° で即脱力。
- 終了は q / Esc。伏せ姿勢へランプしてから脱力する。

## 記録すること

| 項目 | 期待 | 実測 |
|---|---|---|
| IMU ヨーのドリフト | ≤ 45°/5 min | |
| 推論レイテンシ（中央値 / p99） | ≪ 20 ms | |
| 推定器バイアス（静止 vx） | sim は +0.069 | |
| 0.3 m/s の追従 | sim は 91 %（バイアス補正後） | |
| 方位サーボ ON の横流れ | sim は ≤ 0.3 m / 15 s | |

実測が sim と食い違ったら、**sim 側の結論を実測で上書きする**こと。
このリポジトリの数値はすべて sim であり、実機で測り直した項目だけが実機の値。
