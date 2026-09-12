#!/usr/bin/env bash
# `--features sim`（MuJoCo 閉ループ）をビルドする。MuJoCo 3.8.0 の置き場を
# 自分で探し、無ければ mujoco-rs に取ってこさせるので、新しい PC でも
# これ 1 本で通る。
#
#   ./scripts/build_sim.sh            # release ビルド
#   ./scripts/build_sim.sh --debug    # debug ビルド
#
# 優先順位:
#   1. MUJOCO_DYNAMIC_LINK_DIR が設定済み  → それを使う
#   2. $MUJOCO_HOME/lib か ~/.mujoco/mujoco-3.8.0/lib に libmujoco がある
#                                          → それを使う
#   3. どれも無い                          → sim-autodownload feature で
#                                            ~/.mujoco へ取得してビルド
#
# 実行時のライブラリパスは scripts/policy_sim.sh が同じ規則で解決する。
set -eu
cd "$(dirname "$0")/.."

PROFILE=--release
[ "${1:-}" = "--debug" ] && { PROFILE=""; shift; }

MJ_VER=3.8.0
CANDIDATES=(
  "${MUJOCO_DYNAMIC_LINK_DIR:-}"
  "${MUJOCO_HOME:-}/lib"
  "$HOME/.mujoco/mujoco-$MJ_VER/lib"
)
FOUND=""
for d in "${CANDIDATES[@]}"; do
  [ -n "$d" ] && [ -e "$d/libmujoco.so.$MJ_VER" ] && { FOUND="$d"; break; }
done

if [ -n "$FOUND" ]; then
  echo "build_sim: MuJoCo $MJ_VER を $FOUND で見つけました"
  export MUJOCO_DYNAMIC_LINK_DIR="$FOUND"
  exec cargo build $PROFILE --features sim "$@"
fi

# 見つからない: mujoco-rs にダウンロードさせる（要ネットワーク、~5 MB）。
# 展開先は ~/.mujoco なので、次回からは上の 2. で拾われる。
DL="${MUJOCO_DOWNLOAD_DIR:-$HOME/.mujoco}"
mkdir -p "$DL"
echo "build_sim: MuJoCo $MJ_VER が見つかりません — $DL へダウンロードして使います"
export MUJOCO_DOWNLOAD_DIR="$DL"
exec cargo build $PROFILE --features sim-autodownload "$@"
