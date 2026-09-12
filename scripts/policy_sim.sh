#!/usr/bin/env bash
# 学習済みポリシーを MuJoCo で回し、articara へ配信しつつキーボードで操縦する。
#
#   端末1: ./scripts/policy_sim.sh [policy.onnx] [追加フラグ...]
#   端末2: cd ../articara && cargo run --release --features viz -- \
#            --model ../go2-runner/models/unitree_go2/go2.misa
#          GUI の「Live feed (Zenoh)」で topology=Connect,
#          endpoint tcp/127.0.0.1:7447 のまま Subscribe を ON
#
# キー: W/S = 前後, A/D = 旋回, R/F = 横, Space = 停止, q/Esc = 終了
set -eu
cd "$(dirname "$0")/.."
export MUJOCO_DYNAMIC_LINK_DIR="${MUJOCO_DYNAMIC_LINK_DIR:-$HOME/.mujoco/mujoco-3.8.0/lib}"
# libmujoco は cargo run でも自動では載らない（libddsc は載る）。
export LD_LIBRARY_PATH="$MUJOCO_DYNAMIC_LINK_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
MODEL="${1:-$HOME/work/dp/go2_rl/logs/rsl_rl/go2_mit_natural_h30/2026-09-12_14-27-39_h30_v1/exported/policy.onnx}"
[ $# -gt 0 ] && shift
exec cargo run --release --features sim -- policy --sim --model "$MODEL" \
  --viz --viz-endpoint tcp/127.0.0.1:7447 "$@"
