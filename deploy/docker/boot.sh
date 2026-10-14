#!/usr/bin/env bash
# Boot script baked into the public image. Everything it needs comes from the environment the
# launcher sets on the instance: RUN_ID, BIN_SHA, R2_BUCKET, RCLONE_CONFIG_R2_*, NCPU, DPH.
# vast.ai adds CONTAINER_ID and CONTAINER_API_KEY (a key that can only manage this instance).
# 1. pick the x86-64-v3 build if the CPU supports it, else v2
# 2. download bin/<BIN_SHA>/<build>.tar.zst and its manifest.json, check every SHA-256
# 3. download runs/<RUN_ID>/manifest.json and run `runner run` (which uploads and self-destroys)
# Any failure before the runner starts is reported as runs/<RUN_ID>/FAILED and the instance is
# destroyed, so nothing is left billing.
set -euo pipefail
# Never print the environment: it holds the R2 secret and vast logs are fetched into GitHub logs.
echo "boot: started as $(id -un) on $(hostname) at $(date -u +%FT%TZ)"

: "${RUN_ID:?}" "${BIN_SHA:?}" "${R2_BUCKET:?}" "${RCLONE_CONFIG_R2_TYPE:?}"
R="r2:${R2_BUCKET}"
W="${HOME}/w"
mkdir -p "$W"
cd "$W"
RC=(--s3-no-check-bucket --retries 5)

die() {
  echo "boot: FAILED: $*" >&2
  printf '{"stage":"boot","error":"%s"}\n' "$*" >FAILED || true
  rclone copyto "${RC[@]}" FAILED "$R/runs/$RUN_ID/FAILED" || true
  if [[ -n "${CONTAINER_ID:-}" && -n "${CONTAINER_API_KEY:-}" ]]; then
    vastai destroy instance "$CONTAINER_ID" -y --api-key "$CONTAINER_API_KEY" || true
  fi
  exit 1
}
trap 'die "boot script error at line $LINENO"' ERR

v3=1
for f in avx avx2 bmi1 bmi2 fma f16c movbe abm; do
  grep -qw "$f" /proc/cpuinfo || v3=0
done
if [[ $v3 == 1 ]]; then BUILD=x86-64-v3; else BUILD=x86-64-v2; fi
export BIN_BUILD="$BUILD"
echo "boot: run $RUN_ID, binaries $BIN_SHA/$BUILD, NCPU=${NCPU:-?}"

rclone copyto "${RC[@]}" "$R/bin/$BIN_SHA/manifest.json" bin-manifest.json || die "no release manifest for $BIN_SHA"
rclone copyto "${RC[@]}" "$R/bin/$BIN_SHA/$BUILD.tar.zst" bin.tar.zst || die "no $BUILD build for $BIN_SHA"
rclone copyto "${RC[@]}" "$R/runs/$RUN_ID/manifest.json" run-manifest.json || die "no run manifest"

python3 - "$BUILD" <<'PY' || die "binary hash check failed"
import hashlib, json, os, subprocess, sys
build = sys.argv[1]
m = json.load(open("bin-manifest.json"))["builds"][build]
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if h("bin.tar.zst") != m["sha256"]:
    sys.exit("tarball hash mismatch")
os.makedirs("bin", exist_ok=True)
subprocess.run(["tar", "--zstd", "-xf", "bin.tar.zst", "-C", "bin"], check=True)
for name, want in m["files"].items():
    if h(os.path.join("bin", name)) != want:
        sys.exit(f"{name}: hash mismatch")
print("boot: all binary hashes match")
PY

trap - ERR
# The runner uploads its outputs, writes DONE or FAILED last, and destroys the instance.
status=0
./bin/runner run run-manifest.json --work-dir "$W/work" || status=$?
# Only reached if the runner could not destroy the instance itself (e.g. it crashed).
echo "boot: runner exited with $status" >&2
if [[ -n "${CONTAINER_ID:-}" && -n "${CONTAINER_API_KEY:-}" ]]; then
  vastai destroy instance "$CONTAINER_ID" -y --api-key "$CONTAINER_API_KEY" || true
fi
exit "$status"
