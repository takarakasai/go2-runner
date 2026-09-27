#!/usr/bin/env bash
# Go2 の SBC（aarch64）向けに、**sudo なしで**クロスビルドする。
#
#   ./scripts/build_aarch64.sh [追加の cargo 引数...]
#
# misa-policy-runner の同名スクリプトと同じ仕掛け（あちらの冒頭に理由が詳しい）:
# tract-linalg が aarch64 のアセンブラカーネルを持っていて cc-rs 経由で
# `aarch64-linux-gnu-gcc` を探すが、クロス gcc の導入には sudo が要る。
# **clang はマルチターゲット**なので `--target=` を渡せばそのまま使え、
# リンクは musl ターゲットに rustup が同梱する crt + libc.a と rust-lld で済む。
# 出来上がりは**静的リンク**なので SBC 側に何も入れずに走る。
#
# **`--no-default-features` にしている理由**: 既定の `viz`（Zenoh 配信）は
# 実機では要らないうえ依存が重い。実機で使うのは `policy` サブコマンドと
# Go2 の DDS バックエンドだけ。
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=aarch64-unknown-linux-musl
AR=$(command -v llvm-ar || command -v llvm-ar-18 || true)
[ -n "$AR" ] || { echo "llvm-ar が見つかりません" >&2; exit 1; }
command -v clang >/dev/null || { echo "clang が見つかりません" >&2; exit 1; }
rustup target list --installed | grep -qx "$TARGET" || rustup target add "$TARGET"

TOOLCHAIN=$(rustup default | cut -d' ' -f1)
export CC_aarch64_unknown_linux_musl=clang
export CFLAGS_aarch64_unknown_linux_musl="--target=$TARGET"
export AR_aarch64_unknown_linux_musl="$AR"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=rust-lld
export PATH="$HOME/.rustup/toolchains/$TOOLCHAIN/lib/rustlib/x86_64-unknown-linux-gnu/bin:$PATH"

cargo build --release --target "$TARGET" --no-default-features "$@"
echo
BIN="target/$TARGET/release/go2-run"
file "$BIN" 2>/dev/null || true
