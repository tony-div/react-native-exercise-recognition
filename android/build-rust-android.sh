#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUST_DIR="$ROOT_DIR/rust"

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found"
  exit 1
fi

if ! command -v rustup >/dev/null 2>&1; then
  echo "rustup not found"
  exit 1
fi

TARGETS=(
  aarch64-linux-android
  armv7-linux-androideabi
  i686-linux-android
  x86_64-linux-android
)

for target in "${TARGETS[@]}"; do
  rustup target add "$target" >/dev/null 2>&1 || true
  cargo build --manifest-path "$RUST_DIR/Cargo.toml" --target "$target" --release
done
