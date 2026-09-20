#!/usr/bin/env python3
"""Python side of the GRU-contract parity harness (see the Rust example
`misa-policy-runner/examples/gru_parity.rs`).

Three jobs, one script:

  1. `--generate frames.txt` writes a deterministic, plausible sensor
     stream (seeded; no simulator, so both runtimes see *identical* inputs
     and any difference is the runtime's, not the plant's).
  2. `--out python.txt` replays that stream through onnxruntime using the
     rules of the reference implementation
     (`go2_rl/sim2sim_mit_go2_mujoco.py --pure76-gru --velocity-estimator`):
     obs = base37 + clip(last_action, ±100) + estimator(6×73 history),
     GRU hidden carried across ticks, all state zeroed at the start.
  3. `--compare rust.txt` reports the max-abs difference per block
     (observation / action / hidden state).

The default pose (the 0.30 m crouch that the joint-position observation is
measured against) is computed from go2_rl's own `mit_rl/natural_walk.py`,
NOT hardcoded — so a divergence between the Python reference and the Rust
port of the trajectory/IK shows up as an observation difference here.

    python3 scripts/gru_parity_reference.py --go2-rl ~/work/dp/go2_rl \\
        --model model_49.onnx --estimator estimator_42.onnx \\
        --generate /tmp/frames.txt --ticks 400 --out /tmp/python.txt
    cargo run --release --example gru_parity --manifest-path \\
        ../misa-policy-runner/Cargo.toml -- --model model_49.onnx \\
        --estimator estimator_42.onnx --frames /tmp/frames.txt --out /tmp/rust.txt
    python3 scripts/gru_parity_reference.py ... --compare /tmp/rust.txt
"""

import argparse
import importlib.util
import os

import numpy as np
import onnxruntime as ort

N_BASE = 37
N_ACT = 36
N_FEATURES = 73
N_OBS = 76
N_HIDDEN = 128
VEL_FRAMES = 6
ACCEL_CLIP = 30.0
LAST_ACTION_CLIP = 100.0
BODY_HEIGHT = 0.30


def default_pose_isaac(go2_rl):
    """ik(trajectory(0, cmd=0, body_height=0.30)) — the contract's default."""
    import torch
    spec = importlib.util.spec_from_file_location(
        "natural_walk", os.path.join(go2_rl, "mit_rl", "natural_walk.py"))
    natural = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(natural)
    feet, _, _ = natural.trajectory(torch.zeros(1), torch.zeros(1, 3), body_height=BODY_HEIGHT)
    return natural.ik(feet)[0].numpy().astype(np.float64)


def generate(path, ticks, default, seed, host_velocity):
    """A plausible standing-and-wobbling stream: nothing here is a plant, the
    point is only that both runtimes consume the same numbers."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(ticks):
        t = k / 50.0
        cmd = [0.3 + 0.2 * np.sin(2 * np.pi * 0.2 * t), 0.0, 0.3 * np.sin(2 * np.pi * 0.1 * t)]
        # small tilt about x and y, normalized, w >= 0 left to the runtime
        rp, ry = 0.05 * np.sin(2 * np.pi * 0.7 * t), 0.04 * np.cos(2 * np.pi * 0.5 * t)
        quat = np.array([1.0, rp, ry, 0.02 * np.sin(2 * np.pi * 0.3 * t)])
        quat /= np.linalg.norm(quat)
        gyro = 0.3 * np.array([np.sin(2 * np.pi * 1.1 * t), np.cos(2 * np.pi * 0.9 * t), 0.2])
        accel = np.array([0.4 * np.sin(2 * np.pi * 1.3 * t), 0.3, 9.81]) + rng.normal(0, 0.05, 3)
        q = default + 0.15 * np.sin(2 * np.pi * 1.5 * t + np.arange(12)) + rng.normal(0, 0.01, 12)
        dq = 1.5 * np.cos(2 * np.pi * 1.5 * t + np.arange(12)) + rng.normal(0, 0.05, 12)
        row = np.concatenate([cmd, quat, gyro, accel, q, dq])
        if host_velocity:
            row = np.concatenate([row, [0.3, 0.0, 0.0] + rng.normal(0, 0.02, 3)])
        rows.append(row)
    np.savetxt(path, np.array(rows), fmt="%.17e")
    return np.array(rows)


def clamp_cmd(cmd):
    """The adopted checkpoint's envelope (env.yaml): vx [-0.16, 2.0],
    vy identically 0 (never trained), wz +-0.8."""
    return np.array([np.clip(cmd[0], -0.16, 2.0), 0.0, np.clip(cmd[2], -0.8, 0.8)])


def replay(frames, default, model_path, estimator_path, host_velocity):
    actor = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    assert len(actor.get_inputs()) == 2, "not a recurrent graph"
    obs_name, hidden_name = (i.name for i in actor.get_inputs())
    estimator = None
    if not host_velocity:
        estimator = ort.InferenceSession(estimator_path, providers=["CPUExecutionProvider"])
        assert estimator.get_inputs()[0].shape[-1] == VEL_FRAMES * N_FEATURES

    hidden = np.zeros((1, 1, N_HIDDEN), dtype=np.float32)
    last_action = np.zeros(N_ACT, dtype=np.float32)
    history = None
    rows = []
    for row in frames:
        cmd = clamp_cmd(row[0:3])
        quat = row[3:7].copy()
        if quat[0] < 0.0:
            quat = -quat  # quat_unique, w >= 0
        obs = np.concatenate([
            cmd,
            quat,
            row[7:10],
            np.clip(row[10:13], -ACCEL_CLIP, ACCEL_CLIP),
            row[13:25] - default,
            row[25:37],
        ]).astype(np.float32)[None, :]
        obs = np.concatenate((obs, np.clip(last_action, -LAST_ACTION_CLIP, LAST_ACTION_CLIP)[None, :]), axis=-1)
        if host_velocity:
            velocity = row[37:40].astype(np.float32)[None, :]
        else:
            frame = obs[:, :N_FEATURES]
            history = (np.repeat(frame[:, None, :], VEL_FRAMES, axis=1) if history is None
                       else np.concatenate((history[:, 1:], frame[:, None, :]), axis=1))
            velocity = estimator.run(None, {estimator.get_inputs()[0].name:
                                            history.reshape(1, VEL_FRAMES * N_FEATURES)})[0]
        obs = np.concatenate((obs, velocity), axis=-1)
        assert obs.shape[-1] == N_OBS
        action, hidden = actor.run(None, {obs_name: obs, hidden_name: hidden})
        if not np.isfinite(hidden).all() or not np.isfinite(action).all():
            raise RuntimeError("non-finite GRU output")
        last_action = action[0].astype(np.float32)
        rows.append(np.concatenate([obs[0], action[0], hidden.reshape(-1)]).astype(np.float64))
    return np.array(rows)


def compare(mine, other_path):
    other = np.loadtxt(other_path)
    if other.ndim == 1:
        other = other[None, :]
    if other.shape != mine.shape:
        raise SystemExit(f"shape mismatch: python {mine.shape} vs rust {other.shape}")
    blocks = [("observation", 0, N_OBS), ("action", N_OBS, N_OBS + N_ACT),
              ("hidden", N_OBS + N_ACT, N_OBS + N_ACT + N_HIDDEN)]
    worst = 0.0
    for name, lo, hi in blocks:
        d = np.abs(mine[:, lo:hi] - other[:, lo:hi])
        tick, col = np.unravel_index(int(np.argmax(d)), d.shape)
        worst = max(worst, float(d.max()))
        print(f"{name:12s} max |diff| = {d.max():.3e}  (tick {tick}, column {col}), "
              f"final tick {np.abs(mine[-1, lo:hi] - other[-1, lo:hi]).max():.3e}")
    print(f"overall max |diff| = {worst:.3e}")
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--go2-rl", default=os.path.expanduser("~/work/dp/go2_rl"))
    p.add_argument("--model", help="GRU actor ONNX (2 inputs / 2 outputs)")
    p.add_argument("--estimator", help="frozen 438 -> 3 velocity estimator ONNX")
    p.add_argument("--host-velocity", action="store_true",
                   help="skip the estimator; read the velocity from the frames file")
    p.add_argument("--generate", help="write a deterministic sensor stream here")
    p.add_argument("--frames", help="read the sensor stream from here")
    p.add_argument("--ticks", type=int, default=400)
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--out", help="write this side's obs/action/hidden log here")
    p.add_argument("--compare", help="the Rust log to diff against")
    p.add_argument("--tolerance", type=float, default=1e-4)
    a = p.parse_args()

    default = default_pose_isaac(a.go2_rl)
    print("default pose (isaac):", np.array2string(default, precision=6))
    if a.generate:
        frames = generate(a.generate, a.ticks, default, a.seed, a.host_velocity)
        print(f"wrote {len(frames)} ticks to {a.generate}")
    elif a.frames:
        frames = np.loadtxt(a.frames)
    else:
        raise SystemExit("--generate or --frames is required")

    if not a.model:
        return
    rows = replay(frames, default, a.model, a.estimator, a.host_velocity)
    if a.out:
        np.savetxt(a.out, rows, fmt="%.17e")
        print(f"wrote {rows.shape} to {a.out}")
    if a.compare:
        worst = compare(rows, a.compare)
        if worst > a.tolerance:
            raise SystemExit(f"PARITY FAILED: {worst:.3e} > {a.tolerance:.1e}")
        print(f"PARITY OK (<= {a.tolerance:.1e})")


if __name__ == "__main__":
    main()
