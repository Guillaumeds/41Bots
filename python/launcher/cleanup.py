"""Run-aware safety net for vast.ai machines (the vast-cleanup workflow, every 15 minutes).

    python3 python/launcher/cleanup.py [--all] [--dry-run]

A run may last longer than GitHub's 6-hour job limit: the launcher then hands its machine over
(runs/<run>/watch.json says so, and watch/<run>.json lists it here) and this script takes over
the watching. A machine labelled 41bots-<run> is left alone while its run is healthy; it is
destroyed only when one of these holds:

- its run wrote DONE or FAILED more than 10 minutes ago (the runner should have destroyed itself);
- its run has heartbeated since the machine started, but not for 20 minutes, or it never
  heartbeated and is older than 45 minutes; in both cases only when vast shows it idle (CPU below
  50%) or not running. A silent machine that vast shows running and busy (CPU at least 50%, or no
  CPU reading) is treated as an upload or network problem (2026-10-15): it is asked to stop and
  upload what it finished (runs/<run>/STOP) and destroyed only if still silent 15 minutes later;
- vast shows it exited or offline;
- the run is 30 minutes past its own deadline (the runner stops itself at the deadline);
- the run's machines cost more than its dollar cap (watch.json): first it is asked to stop and
  upload what it finished (runs/<run>/STOP), then destroyed if still alive 15 minutes later;
- --all (the workflow's "all" input).

Machines without the 41bots- label are never touched. Before any destroy (except --all) the
machine is read again from vast (`show instance`) and the decision re-made on the fresh record;
if vast cannot be read, nothing is destroyed this round (a healthy machine is never destroyed on
a single bad answer). A destroy is verified by a later status call and logged as unverified
when that cannot be read. For each handed-over run whose machine is gone, it writes
runs/<run>/closed.json, saying whether the machine destroyed itself.
Self-contained on purpose (vastai and aws CLIs only): it runs from main, whatever launch.py is there.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

LABEL = "41bots-"
MARKER_GRACE_S = 10 * 60
SILENT_S = 20 * 60
NO_HEARTBEAT_S = 45 * 60
STOP_GRACE_S = 15 * 60
DEADLINE_GRACE_S = 30 * 60
DEAD_STATES = ("exited", "offline")
BUSY_CPU_PCT = 50.0


def log(msg: str) -> None:
    print(msg, flush=True)


def utc(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(s) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except ValueError:
        try:
            return parsedate_to_datetime(str(s)).timestamp()
        except (TypeError, ValueError):
            return None


# ------------------------------------------------------------------ the decision (pure, tested)

def cpu_pct(inst: dict) -> float | None:
    """vast's `cpu_util` as a percentage of the container, or None when absent or unreadable."""
    try:
        v = float(inst.get("cpu_util"))
    except (TypeError, ValueError):
        return None
    return None if v < 0 else min(v, 100.0)


def decide(inst: dict, state: dict, now: float, everything: bool = False) -> tuple[str, str]:
    """("keep" | "stop" | "destroy", why) for one labelled instance.

    `state`: "markers" {"DONE"/"FAILED": time}, "stop" (time of runs/<run>/STOP or None),
    "heartbeats" (times of the run's heartbeat objects), "watch" (runs/<run>/watch.json or {})."""
    if everything:
        return "destroy", "the 'all' option"
    status = inst.get("actual_status")
    if status in DEAD_STATES:
        return "destroy", f"vast shows it {status}"
    for marker, t in sorted((state.get("markers") or {}).items()):
        if t is not None and now - t > MARKER_GRACE_S:
            return "destroy", f"its run wrote {marker} {(now - t) / 60:.0f} min ago and the machine did not destroy itself"
    start = parse_time(inst.get("start_date")) or now
    # Heartbeats of an earlier machine of the same run do not count as this one's.
    beats = [t for t in state.get("heartbeats") or [] if t >= start - 60]
    silent_why = None
    if beats:
        silent = now - max(beats)
        if silent > SILENT_S:
            silent_why = f"no heartbeat for {silent / 60:.0f} min"
    elif now - start > NO_HEARTBEAT_S:
        silent_why = f"no heartbeat at all in its {(now - start) / 60:.0f} min of life"
    if silent_why:
        cpu = cpu_pct(inst)
        cpu_text = "no CPU reading" if cpu is None else f"CPU {cpu:.0f}%"
        # Running and busy (or with no CPU reading to say otherwise): an upload or network
        # problem, not a dead run. Ask it to stop and upload; destroy only if still silent later.
        if status == "running" and (cpu is None or cpu >= BUSY_CPU_PCT):
            stop = state.get("stop")
            if stop is None:
                return "stop", f"{silent_why} but vast shows it running and busy ({cpu_text}): asking it to stop and upload"
            if now - stop > STOP_GRACE_S:
                return "destroy", (f"{silent_why}, busy ({cpu_text}) but still silent "
                                   f"{(now - stop) / 60:.0f} min after the stop request")
            return "keep", f"{silent_why}, busy ({cpu_text}): stop requested {(now - stop) / 60:.0f} min ago, waiting for its upload"
        return "destroy", f"{silent_why} and vast shows it {status} with {cpu_text}"
    w = state.get("watch") or {}
    deadline = parse_time(w.get("deadline_utc"))
    if deadline and now - deadline > DEADLINE_GRACE_S:
        return "destroy", f"its run is {(now - deadline) / 60:.0f} min past its deadline {w.get('deadline_utc')}"
    cap = float(w.get("max_dollars") or 0)
    if cap > 0:
        dph = float(inst.get("dph_total") or w.get("dph") or 0)
        spent = float(w.get("spent_before") or 0) + max(0.0, now - start) / 3600 * dph
        if spent >= cap:
            stop = state.get("stop")
            if stop is None:
                return "stop", f"${spent:.2f} spent of its ${cap:.2f} cap: asking it to stop and upload"
            if now - stop > STOP_GRACE_S:
                return "destroy", (f"${spent:.2f} spent of its ${cap:.2f} cap and still alive "
                                   f"{(now - stop) / 60:.0f} min after the stop request")
            return "keep", f"${spent:.2f} of ${cap:.2f}: stop requested {(now - stop) / 60:.0f} min ago, waiting for its upload"
    return "keep", "healthy"


# ------------------------------------------------------------------ vast and R2

def vast(*args: str, raw: bool = True):
    cmd = ["vastai", *args, "--api-key", os.environ["VAST_API_KEY"]] + (["--raw"] if raw else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"vastai {' '.join(args[:2])} failed: {p.stderr.strip()[:300]}")
    if not raw:
        return p.stdout
    return json.loads(p.stdout or "null")


def aws(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ, AWS_ACCESS_KEY_ID=os.environ["R2_ACCESS_KEY_ID"],
               AWS_SECRET_ACCESS_KEY=os.environ["R2_SECRET_ACCESS_KEY"], AWS_DEFAULT_REGION="auto",
               AWS_EC2_METADATA_DISABLED="true")
    p = subprocess.run(["aws", "--endpoint-url", os.environ["R2_S3_ENDPOINT"], *args], capture_output=True, env=env)
    if check and p.returncode != 0:
        raise RuntimeError(f"aws {' '.join(args[:2])} failed: {p.stderr.decode()[:300]}")
    return p


class R2:
    def __init__(self):
        self.bucket = os.environ["R2_BUCKET"]

    def list(self, prefix: str) -> list[tuple[str, float]]:
        p = aws("s3api", "list-objects-v2", "--bucket", self.bucket, "--prefix", prefix, "--output", "json")
        out = p.stdout.decode().strip()
        rows = json.loads(out).get("Contents", []) if out else []
        return [(c["Key"], parse_time(c.get("LastModified")) or 0.0) for c in rows]

    def mtime(self, key: str) -> float | None:
        p = aws("s3api", "head-object", "--bucket", self.bucket, "--key", key, "--output", "json", check=False)
        if p.returncode != 0:
            return None
        return parse_time(json.loads(p.stdout.decode() or "{}").get("LastModified"))

    def get(self, key: str) -> bytes | None:
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "x")
            if aws("s3", "cp", f"s3://{self.bucket}/{key}", f, "--only-show-errors", check=False).returncode != 0:
                return None
            with open(f, "rb") as fh:
                return fh.read()

    def put(self, key: str, data: bytes) -> None:
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "x")
            with open(f, "wb") as fh:
                fh.write(data)
            aws("s3", "cp", f, f"s3://{self.bucket}/{key}", "--only-show-errors")

    def delete(self, key: str) -> None:
        aws("s3", "rm", f"s3://{self.bucket}/{key}", "--only-show-errors", check=False)


def get_json(r2, key: str) -> dict:
    try:
        return json.loads(r2.get(key) or b"{}")
    except ValueError:
        return {}


def run_state(r2, run_id: str) -> dict:
    return {
        "markers": {m: r2.mtime(f"runs/{run_id}/{m}") for m in ("DONE", "FAILED")},
        "stop": r2.mtime(f"runs/{run_id}/STOP"),
        "heartbeats": [t for k, t in r2.list(f"runs/{run_id}/heartbeat/") if k.endswith(".json")],
        "watch": get_json(r2, f"runs/{run_id}/watch.json"),
    }


def instance_now(instance_id) -> dict | None:
    """A fresh `show instance` record, or None when vast cannot be read (never a guess)."""
    try:
        info = vast("show", "instance", str(instance_id))
    except RuntimeError as e:
        log(f"show instance {instance_id} failed: {e}")
        return None
    return info if isinstance(info, dict) else None


def destroy(instance_id) -> bool:
    """Destroys the instance; True when the destroy call went through. A later status call
    verifies it (logged as verified, or as unverified when vast cannot be read in 3 tries)."""
    for attempt in range(3):
        try:
            # -y: without it vastai asks "Are you sure?" on stdin and does nothing.
            vast("destroy", "instance", str(instance_id), "-y", raw=False)
        except RuntimeError as e:
            log(f"destroy attempt {attempt + 1} of {instance_id} failed: {e}")
            time.sleep(10)
            continue
        time.sleep(5)
        for _ in range(3):
            info = instance_now(instance_id)
            if info is not None:
                status = info.get("actual_status")
                word = "verified" if status in DEAD_STATES + (None, "destroyed") else f"NOT verified: still {status}"
                log(f"destroyed instance {instance_id} ({word})")
                return True
            time.sleep(10)
        log(f"destroyed instance {instance_id} (unverified: vast's status could not be read in 3 tries; check the console)")
        return True
    return False


def close_run(r2, run_id: str, w: dict, marker: str | None, self_destroyed: bool, why: str, now: float) -> None:
    """Records how a handed-over run ended and stops watching it."""
    start = parse_time(w.get("created_utc"))
    rec = {"run_id": run_id, "instance_id": w.get("instance_id"), "outcome": marker or "none",
           "self_destroyed": self_destroyed, "why": why, "closed_utc": utc(now),
           "dollars_est_upper": round(float(w.get("spent_before") or 0)
                                      + (max(0.0, now - start) / 3600 * float(w.get("dph") or 0) if start else 0), 2)}
    r2.put(f"runs/{run_id}/closed.json", json.dumps(rec, indent=2).encode())
    if w.get("destroy_key_id"):
        try:
            vast("delete", "api-key", str(w["destroy_key_id"]), raw=False)
        except RuntimeError as e:
            log(f"could not delete the destroy key of {run_id}: {e}")
    r2.delete(f"watch/{run_id}.json")
    log(f"closed handed-over run {run_id}: {json.dumps(rec)}")


def sweep(r2, instances: list[dict], now: float, everything: bool, dry_run: bool) -> list[tuple]:
    rows = []
    destroyed: dict[str, str] = {}
    live_ids = set()
    for inst in instances:
        label = inst.get("label") or ""
        if not label.startswith(LABEL):
            continue
        run_id = label[len(LABEL):]
        live_ids.add(int(inst["id"]))
        state = run_state(r2, run_id)
        action, why = decide(inst, state, now, everything)
        if action == "destroy" and not everything and not dry_run:
            # Never on one reading: read the machine again and decide on the fresh record; if
            # vast cannot be read, leave it until the next round.
            fresh = instance_now(inst["id"])
            if fresh is None:
                action, why = "keep", f"would destroy ({why}) but vast could not be read again; left for the next round"
            else:
                again, why2 = decide({**inst, **fresh}, state, now)
                if again != "destroy":
                    action, why = again, f"on a second reading {why2} (the first said: {why})"
        rows.append((inst["id"], run_id, inst.get("actual_status"), action, why))
        log(f"instance {inst['id']} run {run_id} ({inst.get('actual_status')}): {action.upper()} - {why}")
        if dry_run or action == "keep":
            continue
        if action == "stop":
            r2.put(f"runs/{run_id}/STOP", json.dumps({"reason": f"cleanup: {why}", "utc": utc(now)}).encode())
        elif destroy(inst["id"]):
            destroyed[run_id] = why
    # Handed-over runs: record how each ended once its machine is gone.
    for key, _ in r2.list("watch/"):
        run_id = key[len("watch/"):].removesuffix(".json")
        w = get_json(r2, key)
        iid = int(w.get("instance_id") or 0)
        if run_id in destroyed or iid not in live_ids:
            markers = [m for m in ("DONE", "FAILED") if r2.mtime(f"runs/{run_id}/{m}") is not None]
            if dry_run:
                log(f"(dry run) would close handed-over run {run_id}")
                continue
            close_run(r2, run_id, w, markers[0] if markers else None, run_id not in destroyed,
                      destroyed.get(run_id, "the machine was gone"), now)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="destroy every 41bots- instance")
    p.add_argument("--dry-run", action="store_true", help="decide and print, change nothing")
    a = p.parse_args(argv)
    instances = vast("show", "instances") or []
    rows = sweep(R2(), instances, time.time(), a.all, a.dry_run)
    if not rows:
        log("no 41bots- instances")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write("| instance | run | vast status | action | why |\n|---|---|---|---|---|\n")
            fh.writelines(f"| {i} | {r} | {s} | {act} | {why} |\n" for i, r, s, act, why in rows)
            if not rows:
                fh.write("| - | - | - | - | no 41bots- instances |\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
