#!/usr/bin/env bash
# Native-versus-WASM determinism check (docs/PLAN.md §6.1; M0 gate "WASM results equal native").
#
# Builds crates/wasm twice, as a native binary and as a wasm32-unknown-unknown module, has both
# play the same seeded random matches, and compares the FNV-1a hashes of everything that happened
# (every event, public and private, in order, plus the money). Exits non-zero on any difference.
# Needs the pinned Rust toolchain (rust-toolchain.toml lists the wasm32 target) and Node 18+.
#
# Then the app's API (docs/WASM-API.md), when the equity table and the trick-3 network are given
# (WASM_EQUITY=<equity .npy> WASM_NET3=<O_3 .safetensors>; skipped otherwise, they are not in the
# repo): the check's requests (`engine-native fixtures`: a refused trick 1, a robber's rob and a
# dealer's steal, trick 2 with O_3, tricks 3 and 4 exact, two forced cards, two positions with a
# pair turned; and seven of them again with `method: "student"` and a seeded random student network)
# are valued natively and as WebAssembly, and the results must be the same bytes;
# every request is also split into 2, 3, 4 and 7 shards in Node and merged, and must give the
# same bytes again. WASM_STUDENT=<file> uses a real student network for the `-student` requests; WASM_STUDENT1 / WASM_STUDENT2 load per-trick students (S1, S2) too. WASM_SIMD=1 also builds the module with simd128 and compares it the same way.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# The engine version string in every result names this; both builds must carry the same.
export ENGINE_COMMIT="${ENGINE_COMMIT:-check}"
export CARGO_PROFILE_RELEASE_DEBUG="${CARGO_PROFILE_RELEASE_DEBUG:-0}"

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

if [ -z "${WASM_EQUITY:-}" ] || [ -z "${WASM_NET3:-}" ]; then
  echo "SKIP: the API check needs WASM_EQUITY and WASM_NET3 (the equity table and O_3)"
  exit 0
fi
cargo build -p wasm --release --bin engine-native
api_native="$target_dir/release/engine-native"
if [ ! -x "$api_native" ] && [ -x "$api_native.exe" ]; then
  api_native="$api_native.exe"
fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
"$api_native" fixtures "$work/req" 2> /dev/null
# A seeded random student network for the `-student` requests (the same bytes natively and as
# WebAssembly is what is checked, never a bot's strength).
# WASM_STUDENT=<student .safetensors> checks a real network instead (an export check before it ships).
if [ -n "${WASM_STUDENT:-}" ]; then
  cp "$WASM_STUDENT" "$work/student.safetensors"
else
  "$api_native" random-student "$work/student.safetensors" 2> /dev/null
fi
# WASM_STUDENT1 / WASM_STUDENT2=<student .safetensors>: also load a student for trick 1 / trick 2 only (S1, S2), which then serve their
# trick ahead of the default student above.
student_args=(--student "$work/student.safetensors")
[ -n "${WASM_STUDENT1:-}" ] && student_args+=(--student1 "$WASM_STUDENT1")
[ -n "${WASM_STUDENT2:-}" ] && student_args+=(--student2 "$WASM_STUDENT2")
files=("$work"/req/*.json)
echo "API check: ${#files[@]} requests ($(cd "$work/req" && echo *.json))"
"$api_native" --equity "$WASM_EQUITY" --net "$WASM_NET3" "${student_args[@]}" eval "${files[@]}" \
  > "$work/native.txt" 2> /dev/null
modules=("$module")
if [ "${WASM_SIMD:-0}" = "1" ]; then
  CARGO_TARGET_DIR="$target_dir/wasm-simd" RUSTFLAGS="-C target-feature=+simd128" \
    cargo build -p wasm --release --target wasm32-unknown-unknown --lib
  modules+=("$target_dir/wasm-simd/wasm32-unknown-unknown/release/wasm.wasm")
fi
for m in "${modules[@]}"; do
  node tools/wasm_api_run.mjs "$m" --equity "$WASM_EQUITY" --net "$WASM_NET3" "${student_args[@]}" \
    eval "${files[@]}" > "$work/wasm.txt" 2> /dev/null
  if cmp -s "$work/native.txt" "$work/wasm.txt"; then
    echo "$m: every result the same bytes as native"
  else
    echo "$m: results DIFFER from native"
    failed=1
  fi
done
for k in 2 3 4 7; do
  node tools/wasm_api_run.mjs "$module" --equity "$WASM_EQUITY" --net "$WASM_NET3" "${student_args[@]}" \
    shards "$k" "${files[@]}" > "$work/shards.txt" 2> /dev/null
  if cmp -s "$work/native.txt" "$work/shards.txt"; then
    echo "$k shards merged: the same bytes"
  else
    echo "$k shards merged: DIFFERENT"
    failed=1
  fi
done
if [ "$failed" -ne 0 ]; then
  echo "FAIL: the API's results differ"
  exit 1
fi
echo "PASS: the API's WASM results equal native, for any split"
