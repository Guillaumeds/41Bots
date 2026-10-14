#!/usr/bin/env python3
"""Rent one on-demand vast.ai machine, run one manifest chunk on it, watch it, destroy it.

docs/M1-SPEC.md section 7 and docs/PLAN.md section 12. Standard library only; vast.ai is driven
through the `vastai` CLI (pip package `vastai`) and R2 through the `aws` CLI (S3 API).

    launch.py plan --min-cores 16                      # dry run: rank offers, print the choice
    launch.py run --manifest m.json --min-cores 16 --max-minutes 30 \
                  --image ghcr.io/guillaumeds/41bots-base:<tag> --bin-sha <git sha>

Credentials come only from the environment:
    VAST_API_KEY                                     vast.ai account key (never sent to the machine)
    R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,          the bucket-scoped R2 key (the machine gets it)
    R2_BUCKET, R2_S3_ENDPOINT

Rules applied:
- On-demand only (`--type on-demand`, never a bid price).
- Rank by estimated dollars per unit of work: dph / (effective cores x measured matches per second
  per core for that CPU model, from R2 bench/; unknown models use the median of known ones, or 1).
- Owner's rule: among offers within 20% (--premium) of the cheapest cost per unit of work, take
  the one doing the most work per hour.
- Watch through the vast API and the R2 heartbeat every 60 s, never SSH. Destroy when: no
  heartbeat within --startup-minutes; heartbeat older than 3 minutes; CPU use below 85% for 5
  minutes while the chunk is running (unless its heartbeat says progress.partial_cpu_ok); --max-minutes reached; DONE/FAILED seen. Always destroy,
  never stop.
- Format-2 manifests (docs/M3-SPEC.md section 7.3) resume: when the heartbeat goes missing for 3
  minutes, the machine never starts, or the instance dies before DONE, the launcher destroys it
  and rents a replacement (another machine) with the same manifest, which carries on from the
  done-list. At most 3 replacements per run, then it stops and reports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

POLL_S = 60
HEARTBEAT_STALE_S = 180
MAX_REPLACEMENTS = 3
LOW_CPU_PCT = 85.0
LOW_CPU_S = 300


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%SZ')}] {msg}", flush=True)


def need_env(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        sys.exit(f"launch: missing environment variables: {', '.join(missing)}")


# ------------------------------------------------------------------ vast.ai

def vast(*args: str, raw: bool = True, check: bool = True, tail: tuple[str, ...] = ()):
    # `tail` goes last: `--args` (argparse REMAINDER) swallows every token after it.
    cmd = ["vastai", *args, "--api-key", os.environ["VAST_API_KEY"]]
    if raw:
        cmd.append("--raw")
    cmd.extend(tail)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check and p.returncode != 0:
        # Never echo the command line: it holds keys.
        raise RuntimeError(f"vastai {args[0]} {args[1] if len(args) > 1 else ''} failed: {p.stderr.strip()[:500]}")
    if not raw:
        return p.stdout
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return None


def offer_query(a) -> str:
    return " ".join([
        "reliability>0.98", "num_gpus>=1", "gpu_frac=1", "rentable=true", "verified=true",
        f"cpu_cores_effective>={a.min_cores}", f"cpu_ram>={a.min_ram_gb}",
        "compute_cap>=600", "compute_cap<1000", "cuda_vers>=12.6",
        f"disk_space>={a.disk_gb}", f"inet_down_cost<={a.max_net_cost}", f"inet_up_cost<={a.max_net_cost}",
    ] + ([f"dph_total<={a.max_dph}"] if a.max_dph else []))


def search_offers(a) -> list[dict]:
    # --type on-demand: fixed-price rental only. Interruptible (bid) machines are never used.
    rows = vast("search", "offers", offer_query(a), "--type", "on-demand", "--order", "dph_total", "--limit", "200")
    return [o for o in (rows or []) if not o.get("is_bid") and o.get("dph_total") and o.get("cpu_cores_effective")
            and (not a.max_dph or float(o["dph_total"]) <= a.max_dph)
            and (not a.cpu_match or re.search(a.cpu_match, o.get("cpu_name") or "", re.I))
            and o.get("machine_id", o.get("id")) not in getattr(a, "exclude_machines", set())]


# ------------------------------------------------------------------ R2 (aws CLI, S3 API)

def _aws_env() -> dict:
    e = dict(os.environ)
    e.update(AWS_ACCESS_KEY_ID=os.environ["R2_ACCESS_KEY_ID"], AWS_SECRET_ACCESS_KEY=os.environ["R2_SECRET_ACCESS_KEY"],
             AWS_DEFAULT_REGION="auto", AWS_EC2_METADATA_DISABLED="true")
    return e


def _aws(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["aws", "--endpoint-url", os.environ["R2_S3_ENDPOINT"], *args]
    p = subprocess.run(cmd, capture_output=True, env=_aws_env())
    if check and p.returncode != 0:
        raise RuntimeError(f"aws {' '.join(args[:2])} failed: {p.stderr.decode()[:500]}")
    return p


def r2_get(key: str) -> bytes | None:
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "x")
        p = _aws("s3", "cp", f"s3://{os.environ['R2_BUCKET']}/{key}", f, "--only-show-errors", check=False)
        if p.returncode != 0:
            return None
        with open(f, "rb") as fh:
            return fh.read()


def r2_put(key: str, data: bytes) -> None:
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "x")
        with open(f, "wb") as fh:
            fh.write(data)
        _aws("s3", "cp", f, f"s3://{os.environ['R2_BUCKET']}/{key}", "--only-show-errors")


def r2_list(prefix: str) -> list[str]:
    p = _aws("s3api", "list-objects-v2", "--bucket", os.environ["R2_BUCKET"], "--prefix", prefix, "--output", "json")
    out = p.stdout.decode().strip()
    if not out:
        return []
    return [c["Key"] for c in json.loads(out).get("Contents", [])]


# ------------------------------------------------------------------ ranking

def cpu_slug(name: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "unknown").lower()).strip("-") or "unknown"


def load_bench(use_r2: bool) -> dict[str, float]:
    """CPU model slug -> median measured difftest matches per second per core."""
    if not use_r2:
        return {}
    per: dict[str, list[float]] = {}
    for key in r2_list("bench/"):
        raw = r2_get(key)
        try:
            b = json.loads(raw or b"")
            per.setdefault(key.split("/")[1], []).append(float(b["matches_per_sec_per_core"]))
        except (ValueError, KeyError, IndexError, TypeError):
            continue
    return {k: statistics.median(v) for k, v in per.items() if v}


def rank(offers: list[dict], bench: dict[str, float], premium: float) -> tuple[dict | None, list[dict]]:
    default_rate = statistics.median(bench.values()) if bench else 1.0
    for o in offers:
        slug = cpu_slug(o.get("cpu_name"))
        o["_rate"] = bench.get(slug, default_rate)
        o["_measured"] = slug in bench
        o["_work_per_hour"] = o["cpu_cores_effective"] * o["_rate"]
        o["_cost_per_work"] = o["dph_total"] / o["_work_per_hour"]
    offers.sort(key=lambda o: (o["_cost_per_work"], o["id"]))
    if not offers:
        return None, offers
    cheapest = offers[0]["_cost_per_work"]
    near = [o for o in offers if o["_cost_per_work"] <= cheapest * (1.0 + premium)]
    best = max(near, key=lambda o: (o["_work_per_hour"], -o["_cost_per_work"]))
    return best, offers


def describe(o: dict) -> str:
    unit = "matches/s/core measured" if o["_measured"] else "assumed rate"
    return (f"offer {o['id']}: ${o['dph_total']:.3f}/h, {o['cpu_cores_effective']:.0f} eff. cores, "
            f"{o.get('cpu_name')}, {o.get('num_gpus')}x {o.get('gpu_name')}, rel {o.get('reliability2', 0):.3f}, "
            f"rate {o['_rate']:.2f} ({unit}), cost/work {o['_cost_per_work']:.5f}")


def choose(a) -> dict | None:
    offers = search_offers(a)
    have_r2 = all(os.environ.get(n) for n in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_S3_ENDPOINT"))
    bench = load_bench(have_r2)
    best, ranked = rank(offers, bench, a.premium)
    if a.cheapest and ranked:
        # Test runs: the lowest hourly price, whatever its speed.
        best = min(ranked, key=lambda o: (float(o["dph_total"]), o["id"]))
    log(f"{len(ranked)} on-demand offers with >= {a.min_cores} effective cores; {len(bench)} CPU models benchmarked")
    for o in ranked[:10]:
        log(("  * " if o is best else "    ") + describe(o))
    if best:
        how = "cheapest per hour (--cheapest)" if a.cheapest else f"within {a.premium:.0%} of cheapest cost per work, fastest"
        log(f"chosen ({how}): {describe(best)}")
    return best


# ------------------------------------------------------------------ run

def instance_env(run_id: str, ncpu: int, dph: float, bin_sha: str) -> str:
    kv = {
        "RUN_ID": run_id, "NCPU": str(ncpu), "DPH": f"{dph:.5f}", "BIN_SHA": bin_sha,
        "R2_BUCKET": os.environ["R2_BUCKET"], "R2_S3_ENDPOINT": os.environ["R2_S3_ENDPOINT"],
        "RCLONE_CONFIG_R2_TYPE": "s3", "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
        "RCLONE_CONFIG_R2_ENDPOINT": os.environ["R2_S3_ENDPOINT"],
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID": os.environ["R2_ACCESS_KEY_ID"],
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": os.environ["R2_SECRET_ACCESS_KEY"],
        "RCLONE_CONFIG_R2_NO_CHECK_BUCKET": "true",
    }
    for k, v in kv.items():
        if not re.fullmatch(r"[A-Za-z0-9_.:/+=-]*", v):
            sys.exit(f"launch: environment value for {k} has characters vast's --env cannot carry")
    return " ".join(f"-e {k}={v}" for k, v in kv.items())


def destroy(instance_id: int) -> None:
    for attempt in range(5):
        try:
            # -y: without it vastai asks "Are you sure?" on stdin, reads EOF and silently does nothing.
            vast("destroy", "instance", str(instance_id), "-y", raw=False)
            time.sleep(5)
            info = instance_status(instance_id)
            if info is None or info.get("actual_status") in (None, "exited", "offline", "destroyed"):
                log(f"destroyed instance {instance_id} (verified)")
                return
            log(f"destroy attempt {attempt + 1}: instance {instance_id} still {info.get('actual_status')}")
        except RuntimeError as e:
            log(f"destroy attempt {attempt + 1} failed: {e}")
            time.sleep(10)
    log(f"::error::could not destroy instance {instance_id}; destroy it by hand in the vast.ai console")


BOOT_PATH = "/usr/local/bin/41bots-boot"


def instance_status(instance_id: int) -> dict | None:
    try:
        info = vast("show", "instance", str(instance_id))
    except RuntimeError as e:
        log(f"show instance failed: {e}")
        return None
    return info if isinstance(info, dict) else None


def status_line(info: dict | None) -> str:
    if not info:
        return "status unknown"
    msg = str(info.get("status_msg") or "").strip().replace("\n", " ")[:300]
    return (f"actual_status={info.get('actual_status')} intended_status={info.get('intended_status')} "
            f"cur_state={info.get('cur_state')} status_msg={msg!r}")


def print_logs(instance_id: int, lines: int = 200) -> None:
    """Print the tail of the container log (and the daemon log) into the GitHub log before destroying."""
    for label, extra in (("container", ()), ("daemon", ("--daemon-logs",))):
        try:
            text = vast("logs", str(instance_id), "--tail", str(lines), *extra, raw=False)
        except RuntimeError as e:
            log(f"could not fetch {label} logs: {e}")
            continue
        secret = os.environ.get("R2_SECRET_ACCESS_KEY")
        if secret:
            text = text.replace(secret, "***")
        log(f"::group::vast {label} log of instance {instance_id} (last {lines} lines)")
        print(text.rstrip(), flush=True)
        log("::endgroup::")


def instance_gone(instance_id: int) -> bool:
    try:
        info = vast("show", "instance", str(instance_id))
    except RuntimeError:
        return True
    return not info or (info.get("actual_status") in (None, "exited", "offline") and info.get("intended_status") == "stopped")


def latest_heartbeat(run_id: str) -> tuple[str | None, dict | None]:
    keys = sorted(k for k in r2_list(f"runs/{run_id}/heartbeat/") if k.endswith(".json"))
    if not keys:
        return None, None
    raw = r2_get(keys[-1])
    try:
        return keys[-1], json.loads(raw or b"")
    except ValueError:
        return keys[-1], None


def cmd_run(a) -> int:
    need_env("VAST_API_KEY", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_S3_ENDPOINT")
    text = a.manifest if a.manifest.lstrip().startswith("{") else open(a.manifest, encoding="utf-8").read()
    manifest = json.loads(text)
    run_id = manifest.get("run_id") or f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    manifest["run_id"] = run_id
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        sys.exit("launch: run_id may only use letters, digits, - _ .")
    if r2_list(f"runs/{run_id}/"):
        sys.exit(f"launch: runs/{run_id}/ already exists in R2; choose a new run_id")
    if not r2_get(f"bin/{a.bin_sha}/manifest.json"):
        sys.exit(f"launch: no release binaries at bin/{a.bin_sha}/ (run release-binaries.yml first)")

    best = choose(a)
    if not best:
        sys.exit("launch: no offer matches the filters")
    # No "threads" here: the runner derives the usable cores from the container's cgroup CPU quota
    # (vast's cpu_cores_effective is a fraction of the host; NCPU below is recorded for comparison).
    manifest.setdefault("deadline_utc", datetime.fromtimestamp(time.time() + a.max_minutes * 60 - 120, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    if a.dry_run:
        log("dry run: would upload manifest and create the instance:\n" + json.dumps(manifest, indent=2))
        return 0
    r2_put(f"runs/{run_id}/manifest.json", json.dumps(manifest, indent=2).encode())
    resumable = manifest.get("format") == 2
    run_start = time.time()
    attempts: list[dict] = []
    cpu_samples: list[float] = []
    a.exclude_machines = set()
    while True:
        ncpu = int(best["cpu_cores_effective"])
        dph = float(best["dph_total"])
        instance_id = create_instance(a, best, run_id, ncpu, dph)
        created = time.time()
        final, reason, replaceable = "", "", False
        try:
            final, reason, replaceable = watch(a, run_id, instance_id, created, run_start, cpu_samples)
        finally:
            if final != "DONE":
                log(f"last status before destroy: {status_line(instance_status(instance_id))}")
                print_logs(instance_id)
            if final:
                # The runner destroys itself after uploading; give it two minutes, then make sure.
                for _ in range(4):
                    if instance_gone(instance_id):
                        break
                    time.sleep(30)
            destroy(instance_id)
        ended = time.time()
        attempts.append({"instance_id": instance_id, "offer_id": best["id"], "cpu_name": best.get("cpu_name"),
                         "cores_effective": ncpu, "dph": dph, "minutes": round((ended - created) / 60, 1),
                         "dollars_est": round((ended - created) / 3600 * dph, 4), "ended": final or reason})
        log(f"instance {instance_id} ended: {reason}")
        if final or not (resumable and replaceable):
            break
        if len(attempts) > MAX_REPLACEMENTS:
            reason = f"{reason}; {MAX_REPLACEMENTS} replacements used, stopping"
            log(f"::error::{reason}")
            break
        a.exclude_machines.add(best.get("machine_id", best.get("id")))
        best = choose(a)
        if not best:
            reason = f"{reason}; no replacement offer"
            break
        log(f"replacement {len(attempts)} of {MAX_REPLACEMENTS}: renting another machine; it resumes from the done-list")

    log(f"run ended: {reason}")
    result = json.loads(r2_get(f"runs/{run_id}/result.json") or r2_get(f"runs/{run_id}/merged.json") or b"{}")
    bench = json.loads(r2_get(f"runs/{run_id}/benchmark.json") or b"{}")
    dollars = sum(x["dollars_est"] for x in attempts)
    last = attempts[-1]
    matches = result.get("matches") or 0
    report = {
        "run_id": run_id, "instance_id": last["instance_id"], "offer_id": last["offer_id"], "outcome": final or "KILLED",
        "reason": reason, "cpu_name": last["cpu_name"], "gpu_name": best.get("gpu_name") if best else None,
        "cores_effective": last["cores_effective"], "dph": last["dph"],
        "elapsed_min": round((time.time() - run_start) / 60, 1), "dollars_est": round(dollars, 4), "matches": matches,
        "matches_per_dollar": round(matches / dollars) if dollars > 0 and matches else None,
        "avg_cpu_pct": round(statistics.mean(cpu_samples), 1) if cpu_samples else None,
        "attempts": attempts, "replacements": len(attempts) - 1,
        "benchmark": bench, "result": result, "bin_sha": a.bin_sha, "image": a.image,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    r2_put(f"runs/{run_id}/report.json", json.dumps(report, indent=2).encode())
    if bench.get("matches_per_sec_per_core"):
        r2_put(f"bench/{cpu_slug(bench.get('cpu_model') or last['cpu_name'])}/{run_id}.json", json.dumps(bench).encode())
    print(json.dumps(report, indent=2))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(f"## vast.ai run {run_id}: {report['outcome']}\n\n```json\n{json.dumps(report, indent=2)}\n```\n")
    return 0 if final == "DONE" else 1


def create_instance(a, best: dict, run_id: str, ncpu: int, dph: float) -> int:
    # Entrypoint ("args") launch mode. Without --args the CLI defaults to the ssh runtype, and vast's
    # ssh/jupyter modes replace the image's ENTRYPOINT/CMD with their own init, so the boot script
    # never ran (run 38509301077). --entrypoint names the boot script; an empty --args (must be the
    # very last token) selects runtype "args". On-demand: never pass --bid_price.
    res = vast("create", "instance", str(best["id"]), "--image", a.image, "--disk", str(a.disk_gb),
               "--env", instance_env(run_id, ncpu, dph, a.bin_sha), "--label", f"41bots-{run_id}",
               "--entrypoint", BOOT_PATH, "--cancel-unavail", tail=("--args",))
    instance_id = (res or {}).get("new_contract")
    if not instance_id:
        sys.exit(f"launch: create failed: {res}")
    log(f"created instance {instance_id} for run {run_id}")
    if a.state_file:
        # Always the live instance, so the workflow's cleanup step destroys the right one.
        with open(a.state_file, "w") as fh:
            fh.write(str(instance_id))
    return instance_id


def watch(a, run_id: str, instance_id: int, created: float, run_start: float,
          cpu_samples: list[float]) -> tuple[str, str, bool]:
    """Watch one instance. Returns (DONE/FAILED or "", reason, whether a replacement may resume)."""
    low_since, last_key, last_change = None, None, created
    # Heartbeats of an earlier machine of this run do not count as this one's.
    first_key, _ = latest_heartbeat(run_id)
    while True:
        time.sleep(POLL_S)
        now = time.time()
        for marker in ("DONE", "FAILED"):
            if r2_get(f"runs/{run_id}/{marker}") is not None:
                return marker, marker, False
        if now - run_start > a.max_minutes * 60:
            return "", f"max_minutes ({a.max_minutes}) reached", False
        key, hb = latest_heartbeat(run_id)
        if key == first_key:
            key = None
        if key != last_key:
            last_key, last_change = key, now
        if key is None:
            if now - created > a.startup_minutes * 60:
                return "", f"no heartbeat within {a.startup_minutes} minutes of renting", True
            info = instance_status(instance_id)
            if instance_gone(instance_id) or (info or {}).get("actual_status") == "exited":
                return "", "instance stopped before the first heartbeat", True
            log(f"waiting for the machine to start (image download and boot): {status_line(info)}")
            continue
        if now - last_change > HEARTBEAT_STALE_S:
            return "", "heartbeat missing for 3 minutes", True
        if instance_gone(instance_id):
            return "", "instance died before DONE", True
        hb = hb or {}
        cpu = hb.get("cpu_util_pct")
        if cpu is None:
            cpu = hb.get("host_cpu_busy_pct")
        log(f"heartbeat {hb.get('seq')}: phase {hb.get('phase')}, cpu {cpu}%, elapsed {hb.get('elapsed_s')} s, est ${hb.get('est_cost_usd')}")
        # A chunk may declare a stretch of deliberately partial CPU use (e.g. the solver
        # measurement, which times single-thread solves); the low-CPU rule skips it.
        partial_ok = bool((hb.get("progress") or {}).get("partial_cpu_ok"))
        if hb.get("phase") in ("benchmark", "chunk", "chunks") and cpu is not None and not partial_ok:
            cpu_samples.append(float(cpu))
            if cpu < LOW_CPU_PCT:
                low_since = low_since or now
                if now - low_since >= LOW_CPU_S:
                    return "", f"CPU use below {LOW_CPU_PCT:.0f}% for 5 minutes while work remains", False
            else:
                low_since = None


def cmd_plan(a) -> int:
    need_env("VAST_API_KEY")
    log("query: " + offer_query(a) + " (type on-demand)")
    return 0 if choose(a) else 1


def cmd_destroy(a) -> int:
    need_env("VAST_API_KEY")
    destroy(int(a.instance_id))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def offer_args(s):
        s.add_argument("--min-cores", type=int, default=16)
        s.add_argument("--min-ram-gb", type=int, default=32)
        s.add_argument("--disk-gb", type=int, default=20)
        s.add_argument("--max-net-cost", type=float, default=0.02, help="max $/GB up or down")
        s.add_argument("--premium", type=float, default=0.20, help="speed premium margin (owner's 20%% rule)")
        s.add_argument("--max-dph", type=float, default=None, help="max total $/hour for the offer")
        s.add_argument("--cpu-match", default="", help="regex the offer's CPU name must match (e.g. EPYC)")
        s.add_argument("--cheapest", action="store_true", help="test runs: take the cheapest $/hour offer, not the best cost per work")

    s = sub.add_parser("plan", help="dry run: rank offers and print the choice")
    offer_args(s)
    s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("run", help="rent, run, watch, destroy, report")
    offer_args(s)
    s.add_argument("--manifest", required=True, help="path to a manifest JSON file, or the JSON itself")
    s.add_argument("--image", required=True)
    s.add_argument("--bin-sha", required=True)
    s.add_argument("--max-minutes", type=int, default=30)
    s.add_argument("--startup-minutes", type=int, default=15)
    s.add_argument("--state-file", help="write the instance id here (for a cleanup step)")
    s.add_argument("--dry-run", action="store_true", help="choose the offer, but create nothing")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("destroy", help="destroy one instance")
    s.add_argument("instance_id")
    s.set_defaults(fn=cmd_destroy)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
