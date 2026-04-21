#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUST_DIR="$ROOT_DIR/rust"
PREBUILT_DIR="$ROOT_DIR/android/prebuilt"

ABI_MAP=(
  "aarch64-linux-android:arm64-v8a"
  "armv7-linux-androideabi:armeabi-v7a"
  "i686-linux-android:x86"
  "x86_64-linux-android:x86_64"
)

has_all_prebuilt=true
for pair in "${ABI_MAP[@]}"; do
  target="${pair%%:*}"
  abi="${pair##*:}"
  if [ ! -f "$PREBUILT_DIR/$abi/libexercise_recognition_rust.a" ]; then
    has_all_prebuilt=false
    break
  fi
done

if [ "$has_all_prebuilt" = true ]; then
  echo "Using prebuilt binaries"
  exit 0
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found"
  exit 1
fi

if ! command -v rustup >/dev/null 2>&1; then
  echo "rustup not found"
  exit 1
fi

for pair in "${ABI_MAP[@]}"; do
  target="${pair%%:*}"
  abi="${pair##*:}"
  rustup target add "$target" >/dev/null 2>&1 || true
  cargo build --manifest-path "$RUST_DIR/Cargo.toml" --target "$target" --release
  mkdir -p "$PREBUILT_DIR/$abi"
  cp "$RUST_DIR/target/$target/release/libexercise_recognition_rust.a" "$PREBUILT_DIR/$abi/"
done
