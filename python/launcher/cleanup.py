"""Run-aware safety net for vast.ai machines (the vast-cleanup workflow, every 15 minutes).

    python3 python/launcher/cleanup.py [--all] [--dry-run]

A run may last longer than GitHub's 6-hour job limit: the launcher then hands its machine over
(runs/<run>/watch.json says so, and watch/<run>.json lists it here) and this script takes over
the watching. When it closes a handed-over run it also marks the run's machine-cost record
(machines/<run>.json, written by the runner every 10 minutes) final, with the outcome, minutes
and dollars from the run's last heartbeat or its launcher data, if the runner could not do that
itself. A machine labelled 41bots-<run> is left alone while its run is healthy; it is
destroyed only when one of these holds:

- its run wrote DONE or FAILED more than 10 minutes ago (the runner should have destroyed itself);
- its run has heartbeated since the machine started, but not for 20 minutes, or it never
  heartbeated and is older than 60 minutes; in both cases only when vast shows it idle (CPU below
  50%, or below the run's own busy threshold from watch.json when lower, e.g. 1% for a run that
  keeps only a few cores busy) or not running. A silent machine that vast shows running and busy (CPU at least 50%, or no
  CPU reading) is treated as an upload or network problem (2026-10-15): it is asked to stop and
  upload what it finished (runs/<run>/STOP) and destroyed only if still silent 15 minutes later;
- vast shows it exited, or offline with no heartbeat for 20 minutes (vast losing touch with a
  host that still heartbeats to R2 is not a dead run);
- the run is 30 minutes past its own deadline (the runner stops itself at the deadline);
- the run's machines cost more than its dollar cap (watch.json): first it is asked to stop and
  upload what it finished (runs/<run>/STOP), then destroyed if still alive 15 minutes later;
- --all (the workflow's "all" input).

A kept machine (keep/<instance>.json, written by a launcher run with machine.keep, `launch.py` keep_machine) is
left alone, whatever its run's markers say, until the record's keep_until_utc plus 5 minutes; then it is destroyed
(an idle machine bills for nothing). A takeover deletes the record. A record whose machine is gone is deleted.

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
# A machine that has never heartbeated is given 60 minutes (was 45, until 2026-10-16). The launcher lets an image pull
# that is still advancing run for up to 40 minutes from renting (launch.py PULL_CEILING_MIN) and then allows 15 more
# minutes from the container's start to the first heartbeat, so a healthy but slow-starting machine needs up to 55;
# 60 leaves a margin. test_cleanup.py checks this stays above the launcher's two numbers.
NO_HEARTBEAT_S = 60 * 60
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


KEEP_GRACE_S = 5 * 60


def decide_kept(inst: dict, rec: dict, now: float, everything: bool = False) -> tuple[str, str]:
    """("keep" | "destroy", why) for a machine with a keep record: left up for the next worker until its
    keep_until_utc (plus a grace), destroyed after, or at once when it is dead or the record cannot be read."""
    if everything:
        return "destroy", "the 'all' option"
    status = inst.get("actual_status")
    if str(inst.get("intended_status")) == "stopped":
        # Stopped on purpose (the owner's console stop): the rental and its disk are the owner's to keep at the
        # storage rate; the cleanup never destroys it (EXPERIMENTS A11, 2026-10-25).
        return "keep", "stopped by the owner (intended_status stopped): left as it is"
    if status in DEAD_STATES:
        return "destroy", f"kept machine, but vast shows it {status}"
    until = parse_time(rec.get("keep_until_utc"))
    if until is None:
        return "destroy", "its keep record has no readable keep_until_utc"
    if now > until + KEEP_GRACE_S:
        return "destroy", f"kept machine idle past its keep time {rec.get('keep_until_utc')} (run {rec.get('run_id')})"
    return "keep", f"kept for the next worker until {rec.get('keep_until_utc')} (run {rec.get('run_id')})"


def keep_record_of(run_id: str, instance_id: int, state: dict, inst: dict, marker_t: float) -> dict:
    """keep/<instance>.json for a handed-over kept run that ended at `marker_t` (the same shape as launch.py
    keep_record: the launcher is gone, so the cleanup writes it)."""
    watch = state.get("watch") or {}
    hours = keep_asked(state)[1]
    dph = float(watch.get("dph") or inst.get("dph_total") or 0)
    until = marker_t + hours * 3600
    return {"instance_id": int(instance_id), "run_id": run_id, "dph": round(dph, 5),
            "why": "run ended after the launcher's handover (kept by the cleanup)",
            "cpu_name": watch.get("cpu_name") or inst.get("cpu_name") or "", "hours": hours,
            "kept_utc": utc(marker_t), "keep_until_utc": utc(until),
            "next": (f"deploy the next job onto it: a manifest with \"machine\": {{\"instance_id\": {int(instance_id)}}} "
                     f"through the vast-run workflow (docs/STATUS.md, 'Deploy onto a kept machine'), or destroy it "
                     f"(vast-stop workflow). Until then it bills ${dph:.3f}/h; the vast-cleanup workflow destroys it at "
                     f"keep_until_utc.")}


def decide(inst: dict, state: dict, now: float, everything: bool = False) -> tuple[str, str]:
    """("keep" | "stop" | "destroy", why) for one labelled instance.

    `state`: "markers" {"DONE"/"FAILED": time}, "stop" (time of runs/<run>/STOP or None),
    "heartbeats" (times of the run's heartbeat objects), "watch" (runs/<run>/watch.json or {})."""
    if everything:
        return "destroy", "the 'all' option"
    status = inst.get("actual_status")
    start = parse_time(inst.get("start_date")) or now
    # A taken-over instance keeps its original start_date (the earlier rental), but its new run's container only
    # started when the run was launched (runs/<run>/manifest.json is written then): the silence rules count from
    # there. Without this a takeover of an old machine was destroyed as "no heartbeat in 928 min of life" while
    # its container was still booting (2026-10-24, the 3060 box, S1 training).
    run_start = state.get("run_start")
    if run_start is not None:
        start = max(start, run_start)
    # Heartbeats of an earlier machine of the same run do not count as this one's.
    beats = [t for t in state.get("heartbeats") or [] if t >= start - 60]
    # Offline means vast has lost touch with the host; a run that still heartbeats to R2 is
    # working (review E, 2026-10-16: destroying it lost the work since its last upload). It goes
    # through the other rules (silence, markers, cap) like a running machine.
    working_offline = status == "offline" and bool(beats) and now - max(beats) <= SILENT_S
    if str(inst.get("intended_status")) == "stopped":
        # The owner stopped it from the console: theirs to keep or destroy (EXPERIMENTS A11).
        return "keep", "stopped by the owner (intended_status stopped): left as it is"
    if status in DEAD_STATES and not working_offline:
        return "destroy", f"vast shows it {status}"
    w = state.get("watch") or {}
    for marker, t in sorted((state.get("markers") or {}).items()):
        if t is not None and now - t > MARKER_GRACE_S:
            if keep_asked(state)[0]:
                # A kept run (machine.keep) that ended after its launcher had handed it over: the runner parked
                # (KEEP_MACHINE=1); the machine is the next job's, so the keep record is written now (the sweep)
                # and the keep time runs from the marker (2026-10-25: before, such a run was destroyed here).
                return "kept", f"its run wrote {marker} {(now - t) / 60:.0f} min ago and asked to keep the machine (machine.keep)"
            return "destroy", f"its run wrote {marker} {(now - t) / 60:.0f} min ago and the machine did not destroy itself"
    # The run's own idea of busy (launch.py busy_threshold): a run declared to use few cores
    # (--low-cpu-pct 1) is busy at 17% CPU, so a hang of its uploads is not taken for a dead run.
    try:
        busy_pct = min(BUSY_CPU_PCT, float(w.get("busy_cpu_pct") if w.get("busy_cpu_pct") is not None else BUSY_CPU_PCT))
    except (TypeError, ValueError):
        busy_pct = BUSY_CPU_PCT
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
        if status == "running" and (cpu is None or cpu >= busy_pct):
            stop = state.get("stop")
            if stop is None:
                return "stop", f"{silent_why} but vast shows it running and busy ({cpu_text}): asking it to stop and upload"
            if now - stop > STOP_GRACE_S:
                return "destroy", (f"{silent_why}, busy ({cpu_text}) but still silent "
                                   f"{(now - stop) / 60:.0f} min after the stop request")
            return "keep", f"{silent_why}, busy ({cpu_text}): stop requested {(now - stop) / 60:.0f} min ago, waiting for its upload"
        return "destroy", f"{silent_why} and vast shows it {status} with {cpu_text}"
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
        "run_start": r2.mtime(f"runs/{run_id}/manifest.json"),
        "heartbeats": [t for k, t in r2.list(f"runs/{run_id}/heartbeat/") if k.endswith(".json")],
        "watch": get_json(r2, f"runs/{run_id}/watch.json"),
        # The run's own manifest (its `machine` block says keep): read for runs whose launcher wrote its watch record
        # before the record carried keep (any run launched before 2026-10-25 and handed over).
        "manifest": get_json(r2, f"runs/{run_id}/manifest.json"),
    }


def keep_asked(state: dict) -> tuple[bool, float]:
    """Whether the run asked to keep its machine (watch.json keep, else the manifest's machine.keep), and for how
    many hours (watch.json keep_hours, else the manifest's machine.keep_hours, else 6)."""
    w = state.get("watch") or {}
    m = ((state.get("manifest") or {}).get("machine") or {}) if isinstance(state.get("manifest"), dict) else {}
    keep = w.get("keep") if isinstance(w.get("keep"), bool) else m.get("keep") is True
    hours = w.get("keep_hours") if w.get("keep_hours") is not None else m.get("keep_hours")
    try:
        hours = float(hours) if hours is not None else 6.0
    except (TypeError, ValueError):
        hours = 6.0
    return bool(keep), hours


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


def last_heartbeat(r2, run_id: str) -> dict:
    keys = sorted(k for k, _ in r2.list(f"runs/{run_id}/heartbeat/") if k.endswith(".json"))
    return get_json(r2, keys[-1]) if keys else {}


def finalise_machine_record(rec: dict, run_id: str, outcome: str, w: dict, hb: dict, now: float) -> dict | None:
    """The final form of a run's machine-cost record (machines/<run>.json, see crates/runner/src/machine.rs)
    when the runner did not write one itself: minutes and dollars from the run's last heartbeat (the
    runner's own clock and price), else from the launcher's watch record (an upper bound); the cost per
    node is recomputed from them. None when the record is already final."""
    if rec.get("state") == "final":
        return None
    out = dict(rec)
    start = parse_time(w.get("created_utc"))
    if hb.get("elapsed_s"):
        minutes = float(hb["elapsed_s"]) / 60
        usd = float(hb.get("est_cost_usd") or 0) or minutes / 60 * float(w.get("dph") or rec.get("dph") or 0)
        basis = "the run's last heartbeat"
    elif start:
        minutes = max(0.0, now - start) / 60
        usd = float(w.get("spent_before") or 0) + minutes / 60 * float(w.get("dph") or 0)
        basis = "the launcher's watch record (an upper bound: the machine may have died earlier)"
    else:
        minutes, usd, basis = rec.get("minutes") or 0.0, rec.get("usd") or 0.0, "the runner's last record"
    p = hb.get("progress") or {}
    nodes = int(rec.get("nodes_total") or p.get("nodes_total") or 0)
    out.update({"format": "41bots-machine-record", "version": 1, "run_id": run_id, "state": "final",
                "outcome": outcome, "utc": utc(now), "minutes": round(minutes, 1), "usd": round(usd, 4),
                "nodes_total": nodes, "worlds_finished": rec.get("worlds_finished") or p.get("worlds_finished") or 0,
                "usd_per_1e12_nodes": round(usd / nodes * 1e12, 3) if nodes else None,
                "finalised_by": "cleanup", "finalised_basis": basis})
    out.setdefault("dph", w.get("dph"))
    out.setdefault("instance_id", w.get("instance_id"))
    out.setdefault("offer_id", w.get("offer_id"))
    out.setdefault("cpu_model", w.get("cpu_name"))
    out.setdefault("nodes_basis", "measured" if nodes else "none")
    return out


def close_run(r2, run_id: str, w: dict, marker: str | None, self_destroyed: bool, why: str, now: float) -> None:
    """Records how a handed-over run ended and stops watching it; marks its machine record final."""
    start = parse_time(w.get("created_utc"))
    rec = {"run_id": run_id, "instance_id": w.get("instance_id"), "outcome": marker or "none",
           "self_destroyed": self_destroyed, "why": why, "closed_utc": utc(now),
           "dollars_est_upper": round(float(w.get("spent_before") or 0)
                                      + (max(0.0, now - start) / 3600 * float(w.get("dph") or 0) if start else 0), 2)}
    r2.put(f"runs/{run_id}/closed.json", json.dumps(rec, indent=2).encode())
    machine = finalise_machine_record(get_json(r2, f"machines/{run_id}.json"), run_id,
                                      marker or "KILLED", w, last_heartbeat(r2, run_id), now)
    if machine is not None:
        r2.put(f"machines/{run_id}.json", json.dumps(machine, indent=2).encode())
        log(f"machine record of {run_id} marked final: {machine['minutes']} min, ${machine['usd']}, "
            f"{machine['nodes_total']} nodes ({machine['finalised_basis']})")
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
        kept = get_json(r2, f"keep/{inst['id']}.json")

        def decide_now(i: dict) -> tuple[str, str]:
            return decide_kept(i, kept, now, everything) if kept else decide(i, state, now, everything)

        action, why = decide_now(inst)
        if action == "destroy" and not everything and not dry_run:
            # Never on one reading: read the machine again and decide on the fresh record; if
            # vast cannot be read, leave it until the next round.
            fresh = instance_now(inst["id"])
            if fresh is None:
                action, why = "keep", f"would destroy ({why}) but vast could not be read again; left for the next round"
            else:
                again, why2 = decide_now({**inst, **fresh})
                if again != "destroy":
                    action, why = again, f"on a second reading {why2} (the first said: {why})"
        rows.append((inst["id"], run_id, inst.get("actual_status"), action, why))
        log(f"instance {inst['id']} run {run_id} ({inst.get('actual_status')}): {action.upper()} - {why}")
        if dry_run or action == "keep":
            continue
        if action == "kept":
            # The keep record of a handed-over kept run, from the marker's time (launch.py keep_record's shape).
            marker_t = min(t for t in (state.get("markers") or {}).values() if t is not None)
            rec = keep_record_of(run_id, int(inst["id"]), state, inst, marker_t)
            r2.put(f"keep/{inst['id']}.json", json.dumps(rec, indent=2).encode())
            r2.put(f"runs/{run_id}/KEPT.json", json.dumps(rec, indent=2).encode())
            log(f"::warning::MACHINE KEPT RUNNING, ACTION NEEDED: instance {inst['id']} (run {run_id}) is kept until "
                f"{rec['keep_until_utc']}; deploy the next job onto it (machine.instance_id) or destroy it (vast-stop)")
            continue
        if action == "stop":
            r2.put(f"runs/{run_id}/STOP", json.dumps({"reason": f"cleanup: {why}", "utc": utc(now)}).encode())
        elif destroy(inst["id"]):
            destroyed[run_id] = why
    # Keep records whose machine is gone (destroyed by hand, taken over under another label): delete them.
    for key, _ in r2.list("keep/"):
        rec = get_json(r2, key)
        iid = int(rec.get("instance_id") or 0)
        if iid and iid not in live_ids and not dry_run:
            fresh = instance_now(iid)
            if fresh is None or fresh.get("actual_status") in DEAD_STATES + (None, "destroyed"):
                log(f"keep record {key}: instance {iid} is gone; deleted")
                r2.delete(key)
    # Handed-over runs: record how each ended once its machine is gone.
    for key, _ in r2.list("watch/"):
        run_id = key[len("watch/"):].removesuffix(".json")
        w = get_json(r2, key)
        iid = int(w.get("instance_id") or 0)
        if run_id not in destroyed and iid not in live_ids and not dry_run:
            # Absent from the listing: make sure with a reading of its own before closing the run
            # (an incomplete listing would otherwise close a working run and delete its key).
            fresh = instance_now(iid) if iid else None
            if fresh and fresh.get("actual_status") not in DEAD_STATES + (None, "destroyed"):
                log(f"handed-over run {run_id}: instance {iid} missing from the listing but vast shows it "
                    f"{fresh.get('actual_status')}: not closed")
                continue
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
