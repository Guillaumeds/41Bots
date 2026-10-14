#!/usr/bin/env bash
# Native-versus-WASM determinism check (docs/PLAN.md §6.1; M0 gate "WASM results equal native").
#
# Builds crates/wasm twice, as a native binary and as a wasm32-unknown-unknown module, has both
# play the same seeded random matches, and compares the FNV-1a hashes of everything that happened
# (every event, public and private, in order, plus the money). Exits non-zero on any difference.
# Needs the pinned Rust toolchain (rust-toolchain.toml lists the wasm32 target) and Node 16+.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

cargo build -p wasm --release --target wasm32-unknown-unknown --lib
cargo build -p wasm --release --bin wasm-native

target_dir="${CARGO_TARGET_DIR:-target}"
module="$target_dir/wasm32-unknown-unknown/release/wasm.wasm"
native="$target_dir/release/wasm-native"
if [ ! -x "$native" ] && [ -x "$native.exe" ]; then
  native="$native.exe" # Git Bash on Windows
fi

echo "node $(node --version); module $module ($(wc -c < "$module") bytes)"
node tools/wasm_run.mjs "$module" --describe

failed=0
for run in "1 200" "12345 50"; do
  read -r seed matches <<< "$run"
  native_hash=$("$native" "$seed" "$matches")
  wasm_hash=$(node tools/wasm_run.mjs "$module" "$seed" "$matches")
  if [[ ! "$native_hash" =~ ^[0-9]+$ ]] || [ "$native_hash" != "$wasm_hash" ]; then
    verdict="DIFFERENT"
    failed=1
  else
    verdict="same"
  fi
  echo "seed=$seed matches=$matches native=$native_hash wasm=$wasm_hash $verdict"
done

if [ "$failed" -ne 0 ]; then
  echo "FAIL: native and WASM results differ"
  exit 1
fi
echo "PASS: WASM results equal native"
