#!/usr/bin/env bash
# Boot script baked into the public image. Everything it needs comes from the environment the
# launcher sets on the instance: RUN_ID, BIN_SHA, R2_BUCKET, RCLONE_CONFIG_R2_*, NCPU, DPH, and
# VAST_DESTROY_KEY (the run's destroy-only key, when the launcher could create one).
# vast.ai adds CONTAINER_ID and CONTAINER_API_KEY (documented as a key that can only start, stop or
# destroy this instance; in args launch mode vast.ai answered it with "401: Invalid user key").
# 1. pick the x86-64-v3 build if the CPU supports it, else v2
# 2. download bin/<BIN_SHA>/<build>.tar.zst and its manifest.json, check every SHA-256
# 3. download runs/<RUN_ID>/manifest.json and run `runner run` (which uploads and self-destroys);
#    a format-2 manifest resumes from the run's done-list, so a replacement machine boots the same way
# Any failure before the runner starts is reported as runs/<RUN_ID>/FAILED and the instance is
# destroyed, so nothing is left billing.
set -euo pipefail
# Never print the environment: it holds the R2 secret and vast logs are fetched into GitHub logs.
echo "boot: started as $(id -un) on $(hostname) at $(date -u +%FT%TZ)"

: "${RUN_ID:?}" "${BIN_SHA:?}" "${R2_BUCKET:?}" "${RCLONE_CONFIG_R2_TYPE:?}"
# Which destroy keys arrived (a short SHA-256 prefix, never the key: the launcher logs the same
# prefix of the per-instance key vast.ai returned when it created this instance).
fingerprint() {
  local v="${!1:-}"
  if [[ -n "$v" ]]; then
    printf 'set (sha256 %s)' "$(printf '%s' "$v" | sha256sum | cut -c1-12)"
  else
    printf 'unset'
  fi
}
echo "boot: instance ${CONTAINER_ID:-unset}; CONTAINER_API_KEY $(fingerprint CONTAINER_API_KEY); VAST_DESTROY_KEY $(fingerprint VAST_DESTROY_KEY)"

# Asks vast.ai to destroy this instance, trying the per-instance key, then the run's destroy-only
# key. A key reaches vastai only through its VAST_API_KEY environment variable (never a command
# line), and vastai's output is not echoed (its hints show a key's last characters). vastai exits
# 0 even when the API refuses, so success is its "destroying instance <id>." line. The launcher
# destroys the instance anyway, and vast-cleanup.yml catches anything left.
self_destroy() {
  [[ -n "${CONTAINER_ID:-}" ]] || return 0
  local name key out why
  for name in CONTAINER_API_KEY VAST_DESTROY_KEY; do
    key="${!name:-}"
    [[ -n "$key" ]] || continue
    out=$(VAST_API_KEY="$key" VASTAI_NO_UPDATE_CHECK=1 vastai destroy instance "$CONTAINER_ID" -y </dev/null 2>&1) || true
    if grep -qxF "destroying instance ${CONTAINER_ID}." <<<"$out"; then
      echo "boot: vast.ai accepted the destroy of instance $CONTAINER_ID ($name)"
      return 0
    fi
    why=$(grep -m1 '^Failed with error' <<<"$out" || true)
    why=${why//"$key"/***}
    echo "boot: self-destroy with $name refused: ${why:-no confirmation from vastai}" >&2
  done
  echo "boot: instance $CONTAINER_ID could not destroy itself; the launcher destroys it on DONE/FAILED or a stale heartbeat, vast-cleanup.yml after 2 hours" >&2
}

R="r2:${R2_BUCKET}"
W="${HOME}/w"
mkdir -p "$W"
cd "$W"
RC=(--s3-no-check-bucket --retries 5)

die() {
  echo "boot: FAILED: $*" >&2
  printf '{"stage":"boot","error":"%s"}\n' "$*" >FAILED || true
  rclone copyto "${RC[@]}" FAILED "$R/runs/$RUN_ID/FAILED" || true
  self_destroy || true
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

# The compiler that built the binaries, for the run's final manifest.
BIN_RUSTC=$(python3 -c 'import json; print(json.load(open("bin-manifest.json")).get("rustc", ""))' 2>/dev/null) || BIN_RUSTC=""
export BIN_RUSTC

trap - ERR
# The runner uploads its outputs, writes DONE or FAILED last, and destroys the instance.
# A SIGTERM to this script (PID 1: `docker stop`, a stopped instance) is passed on to the runner,
# which then stops its work, uploads what it finished and writes its marker
# (crates/runner/src/stop.rs); this script waits for that before it exits.
status=0
./bin/runner run run-manifest.json --work-dir "$W/work" &
runner_pid=$!
trap 'kill -TERM "$runner_pid" 2>/dev/null || true' TERM INT HUP
while true; do
  wait "$runner_pid" && status=0 || status=$?
  # `wait` also returns when a trapped signal arrives; go on waiting while the runner lives.
  kill -0 "$runner_pid" 2>/dev/null || break
done
trap - TERM INT HUP
# Only reached if the runner could not destroy the instance itself (e.g. it crashed).
echo "boot: runner exited with $status" >&2
self_destroy || true
exit "$status"
