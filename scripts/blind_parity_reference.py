#!/usr/bin/env python3
"""盲目段差契約の参照実装（Python 側）— Rust の `examples/blind_parity` と突き合わせる。

    python3 scripts/blind_parity_reference.py --model p.onnx --frames frames.txt --out py.txt

入出力の並びは Rust 側と同じ:
  frames.txt: cmd(3) quat_wxyz(4) gyro(3) accel(3) q_isaac(12) dq_isaac(12) vel_body(3) = 40
  out:        obs(48·H) q_des_isaac(12)

契約は go2_rl `artifacts/GO2_BLIND_STAIRS20.md`。加速度はこの契約では使わない
（重力方向を四元数から出す）が、他の照合台と列を揃えるために読む。
"""
import argparse
import numpy as np
import onnxruntime as ort

DEFAULT_ISAAC = np.array([0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5])
POS_SCALE = 0.25
TERM_DIMS = [3, 3, 3, 3, 12, 12, 12]
N_FRAME = 48


def projected_gravity(q):
    w, x, y, z = q
    return np.array([2 * (w * y - x * z), -2 * (y * z + w * x), 2 * (x * x + y * y) - 1.0])


ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--frames', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()

sess = ort.InferenceSession(a.model, providers=['CPUExecutionProvider'])
n_in = sess.get_inputs()[0].shape[1]
assert n_in % N_FRAME == 0, f'入力 {n_in} は {N_FRAME} の倍数でない'
H = n_in // N_FRAME
name = sess.get_inputs()[0].name

rows, frames, last_action = [], None, np.zeros(12)
for line in open(a.frames):
    if not line.strip():
        continue
    v = np.array([float(t) for t in line.split()])
    assert v.size == 40, f'1 行 40 個であること: {v.size}'
    cmd, quat, gyro = v[0:3], v[3:7], v[7:10]
    q_isaac, dq_isaac, vel_body = v[13:25], v[25:37], v[37:40]
    frame = np.concatenate([
        vel_body, gyro, projected_gravity(quat), cmd,
        q_isaac - DEFAULT_ISAAC, dq_isaac, np.clip(last_action, -100.0, 100.0),
    ]).astype(np.float32)
    frames = np.repeat(frame[None, :], H, 0) if frames is None else \
        np.concatenate([frames[1:], frame[None, :]], 0)
    # 項目ごとに [古い…新しい]（IsaacLab の CircularBuffer と同じ並び）
    parts, off = [], 0
    for d in TERM_DIMS:
        parts.append(frames[:, off:off + d].reshape(-1))
        off += d
    obs = np.concatenate(parts).astype(np.float32)
    action = sess.run(None, {name: obs[None, :]})[0][0].astype(np.float64)
    q_des = DEFAULT_ISAAC + POS_SCALE * action        # **クランプしない**
    last_action = action
    rows.append(' '.join(f'{x:.17e}' for x in np.concatenate([obs, q_des])))

open(a.out, 'w').write('\n'.join(rows) + '\n')
print(f'[reference] 履歴 {H} フレーム、{len(rows)} ティック -> {a.out}')
