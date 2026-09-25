#!/usr/bin/env bash
# 盲目段差契約の Rust ↔ Python 照合。
#
#   ./scripts/blind_parity.sh <policy.onnx> [ティック数]
#
# **判定は「同一観測での 1 ティック差」で行う。** 素の行ごとの差は、前回行動が
# 観測に入る契約なので tract と onnxruntime の float32 演算差（1e-6 程度）が
# ティックを追うごとに蓄積し、60 ティックで 1e-4 程度になる。これは実装の
# 不一致ではないので、観測を揃えて 1 ティックぶんだけを比べる。
set -eu
cd "$(dirname "$0")/.."
MODEL="$(realpath "$1")"; N="${2:-60}"
OUT="$(mktemp -d)"
source /home/takara/work/install/isaac_5_1_0/env_isaaclab/bin/activate
python3 - "$OUT/frames.txt" "$N" <<'PY'
import sys, numpy as np
rng = np.random.default_rng(7); rows = []
for _ in range(int(sys.argv[2])):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    rows.append(' '.join(f'{x:.9f}' for x in np.concatenate([
        [0.6, 0.0, 0.0], q, rng.normal(0, 0.3, 3), [0, 0, 9.81] + rng.normal(0, 0.2, 3),
        np.array([0.1,-0.1,0.1,-0.1,0.8,0.8,1.0,1.0,-1.5,-1.5,-1.5,-1.5]) + rng.normal(0, 0.2, 12),
        rng.normal(0, 1.0, 12), rng.normal(0, 0.2, 3)])))
open(sys.argv[1], 'w').write('\n'.join(rows) + '\n')
PY
(cd ../misa-policy-runner && cargo run --release --quiet --example blind_parity -- \
  --model "$MODEL" --frames "$OUT/frames.txt" --out "$OUT/rust.txt")
python3 scripts/blind_parity_reference.py --model "$MODEL" \
  --frames "$OUT/frames.txt" --out "$OUT/py.txt"
python3 - "$MODEL" "$OUT" <<'PY'
import sys, numpy as np, onnxruntime as ort
model, out = sys.argv[1], sys.argv[2]
r = np.loadtxt(f'{out}/rust.txt'); p = np.loadtxt(f'{out}/py.txt')
n = r.shape[1] - 12
assert r.shape == p.shape, (r.shape, p.shape)
# 1 行目は履歴も前回行動も無いので、実装が一致していればここは厳密に同じ。
first = abs(r[0] - p[0]).max()
s = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
name = s.get_inputs()[0].name
DEF = np.array([0.1,-0.1,0.1,-0.1,0.8,0.8,1.0,1.0,-1.5,-1.5,-1.5,-1.5])
d = np.array([abs(DEF + 0.25 * s.run(None, {name: r[k, :n].astype(np.float32)[None, :]})[0][0]
                  - r[k, n:]).max() for k in range(r.shape[0])])
drift = abs(r[:, :n] - p[:, :n]).max()
print(f'1 行目（履歴なし）の差       : {first:.3e}')
print(f'同一観測での 1 ティック差    : 中央値 {np.median(d):.3e} / 最大 {d.max():.3e}')
print(f'（参考）蓄積後の観測差       : {drift:.3e}')
ok = first < 1e-6 and d.max() < 1e-3
print('判定:', '一致' if ok else '不一致')
sys.exit(0 if ok else 1)
PY
rm -rf "$OUT"
