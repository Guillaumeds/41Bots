#!/usr/bin/env bash
# Builds the app's engine module (docs/WASM-API.md §7) into crates/wasm/dist/: engine.wasm (release,
# wasm32-unknown-unknown, no target features beyond the baseline: the measured choice, see the doc),
# the loader engine.mjs, and SHA256SUMS. The module carries the source commit it was built from
# (ENGINE_COMMIT, 12 hex digits; "-dirty" when crates/ or Cargo.lock have uncommitted changes) in
# every result's `engine` string. Build from a committed tree, then commit dist/.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

commit=$(git rev-parse --short=12 HEAD)
if [ -n "$(git status --porcelain -- crates Cargo.lock Cargo.toml rust-toolchain.toml)" ]; then
  commit="$commit-dirty"
fi
export ENGINE_COMMIT="$commit"
export CARGO_PROFILE_RELEASE_DEBUG=0

cargo build -p wasm --release --target wasm32-unknown-unknown --lib

target_dir="${CARGO_TARGET_DIR:-target}"
dist=crates/wasm/dist
mkdir -p "$dist"
cp "$target_dir/wasm32-unknown-unknown/release/wasm.wasm" "$dist/engine.wasm"
cp crates/wasm/js/engine.mjs "$dist/engine.mjs"
(cd "$dist" && sha256sum engine.wasm engine.mjs > SHA256SUMS)
echo "built $dist with ENGINE_COMMIT=$ENGINE_COMMIT"
cat "$dist/SHA256SUMS"
ls -l "$dist"
