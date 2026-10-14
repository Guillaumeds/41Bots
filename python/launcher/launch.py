#!/usr/bin/env python3
"""Rent one on-demand vast.ai machine, run one manifest chunk on it, watch it, destroy it.

docs/M1-SPEC.md section 7 and docs/PLAN.md section 12. Standard library only; vast.ai is driven
through the `vastai` CLI (pip package `vastai`) and R2 through the `aws` CLI (S3 API).

    launch.py plan --min-cores 16                      # dry run: rank offers, print the choice
    launch.py run --manifest m.json --min-cores 16 --max-minutes 30 \
                  --image ghcr.io/guillaumeds/41bots-base:<tag> --bin-sha <git sha>
    launch.py put-input data/equity-v0.npy             # upload a manifest input by content hash

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
- Watch through the vast API and the R2 heartbeat every --poll-s (60) seconds, never SSH. Destroy
  when: no heartbeat within --startup-minutes; heartbeat older than 3 minutes; CPU use below
  --low-cpu-pct (85%) for --low-cpu-s (300) seconds while the chunk is running (unless its
  heartbeat says progress.partial_cpu_ok); --max-minutes reached; DONE/FAILED seen. Always
  destroy, never stop.
- Container start failures are caught at the next poll: while waiting for the first heartbeat, an
  instance whose status shows a container error ("Error response from daemon", "failed to
  inject", "OCI runtime", or actual_status exited/offline) is destroyed at once and the next offer
  is rented (counted as a replacement, for format 1 too: nothing ran on it), instead of waiting
  for --startup-minutes.
- A silent machine (no new heartbeat for 3 minutes) is checked through the vast API first, and its
  container log tail printed (redacted): gone, exited or idle (vast CPU below --busy-cpu-pct, 50%)
  is destroyed at once; running and busy is treated as an upload problem (run m3-agree-t1-b lost
  172 minutes of work that way): the launcher waits up to --silent-busy-min (15) more minutes for
  heartbeats to resume, logging each poll, then asks it to stop and upload as below.
- Budget: --max-dollars (0 = none) stops the run (stop and upload, then destroy) once the money
  spent on its machines reaches it, and rents no replacement past it. A heartbeat's progress.eta
  (done_frac, eta_utc, projected_usd, basis) is shown on each poll line, with a warning once if
  the projected cost exceeds the budget by more than 20%.
- One log line per poll: minutes elapsed, dollars so far, CPU, progress, ETA (UTC and Mauritius
  time, UTC+4) and projected dollars. At the end a short markdown summary (outcome, reason,
  minutes, dollars, the run's files in R2) goes to $GITHUB_STEP_SUMMARY with the full report.
- Keep finished work (docs/M3-SPEC.md §13.1 item 11): before destroying a machine that is still
  working (--max-minutes, low CPU, Ctrl-C) the launcher writes runs/<run_id>/STOP; the runner
  reads it at its next heartbeat, stops, uploads what it finished, writes DONE or FAILED, and the
  launcher waits up to --stop-wait-s (300 s) for that marker before destroying. The report's
  outcome is then STOPPED. A machine whose heartbeat stopped or that died is destroyed at once.
  When the workflow is cancelled this process is killed within seconds, so it only writes STOP
  and leaves the wait, the destroy and the key deletion to the workflow's cleanup steps
  (`launch.py stop RUN_ID --instance ID`; the run id is kept in <state-file>.run-id).
- If creating the instance fails (typically the offer was taken between search and create, which
  vast reports as an empty answer), the next-best offer is tried, up to 5 offers.
- Format-2 manifests (docs/M3-SPEC.md section 7.3) resume: when the heartbeat goes missing for 3
  minutes, the machine never starts, or the instance dies before DONE, the launcher destroys it
  and rents a replacement (another machine) with the same manifest, which carries on from the
  done-list. At most 3 replacements per run, then it stops and reports. A format-2 manifest's
  `inputs` (sha256 content hashes) must already be in R2 at inputs/sha256/<hex> (`put-input`);
  the launcher checks before renting anything.
- Destroying: the launcher always destroys the instance itself (on DONE/FAILED, a stale
  heartbeat, a dead instance or --max-minutes), and vast-cleanup.yml destroys any 41bots-
  instance left after 2 hours: that is the mechanism that does not depend on the machine. The
  machine also tries to destroy itself last. vast.ai documents a per-instance key in
  CONTAINER_API_KEY that "can only start, stop, or destroy that specific instance"
  (docs.vast.ai/api-reference/hello-world), but in args launch mode it answered that key with
  "401: Invalid user key" (2026-10-14). So for each run the launcher also creates an API key
  limited to destroying instances (vast.ai scoped keys, docs.vast.ai/api-reference/permissions;
  the instance id is not known before the create call, so it cannot be limited to the one
  instance), passes it as VAST_DESTROY_KEY with --env, and deletes it when the run ends. If
  vast.ai refuses to create it, the run goes ahead without it. The key never appears in logs.
  A host that reads it could only destroy the account's instances while the run lasts (no
  renting, no billing, no reading); --no-destroy-key leaves it out.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone

POLL_S = 60  # default; --poll-s overrides
HEARTBEAT_STALE_S = 180
MAX_REPLACEMENTS = 3
MAX_CREATE_TRIES = 5
LOW_CPU_PCT = 85.0  # default; --low-cpu-pct overrides per run
LOW_CPU_S = 300  # default; --low-cpu-s overrides
STOP_WAIT_S = 300  # default; --stop-wait-s overrides: how long a stopped machine may take to upload
STOP_POLL_S = 10
SILENT_BUSY_MIN = 15  # default; --silent-busy-min overrides
BUSY_CPU_PCT = 50.0  # default; --busy-cpu-pct overrides
MAURITIUS = timezone(timedelta(hours=4))


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%SZ')}] {msg}", flush=True)


def need_env(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        sys.exit(f"launch: missing environment variables: {', '.join(missing)}")


# ------------------------------------------------------------------ vast.ai

# Keys the launcher created during this process (the run's destroy-only key), redacted like the
# environment's.
SECRETS: list[str] = []


def redact(text: str) -> str:
    """Removes every credential the launcher holds from text that will be logged."""
    for v in [os.environ.get(k) for k in ("VAST_API_KEY", "R2_SECRET_ACCESS_KEY", "R2_ACCESS_KEY_ID")] + SECRETS:
        if v:
            text = text.replace(v, "***")
    return text


def fingerprint(secret: str) -> str:
    """A short SHA-256 prefix to tell keys apart in logs without showing them."""
    return hashlib.sha256(secret.encode()).hexdigest()[:12]


LAST_VAST_OUTPUT = ""


def vast(*args: str, raw: bool = True, check: bool = True, tail: tuple[str, ...] = ()):
    # `tail` goes last: `--args` (argparse REMAINDER) swallows every token after it.
    global LAST_VAST_OUTPUT
    cmd = ["vastai", *args, "--api-key", os.environ["VAST_API_KEY"]]
    if raw:
        cmd.append("--raw")
    cmd.extend(tail)
    p = subprocess.run(cmd, capture_output=True, text=True)
    LAST_VAST_OUTPUT = redact(f"{p.stdout.strip()} {p.stderr.strip()}".strip())[:500]
    if check and p.returncode != 0:
        # Never echo the command line: it holds keys.
        raise RuntimeError(f"vastai {args[0]} {args[1] if len(args) > 1 else ''} failed: {redact(p.stderr.strip())[:500]}")
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
    ] + ([] if a.cpu_only else ["compute_cap>=600", "compute_cap<1000", "cuda_vers>=12.6"]) + [
        f"disk_space>={a.disk_gb}", f"inet_down_cost<={a.max_net_cost}", f"inet_up_cost<={a.max_net_cost}",
    ] + ([f"dph_total<={a.max_dph}"] if a.max_dph else []))


def search_offers(a) -> list[dict]:
    # --type on-demand: fixed-price rental only. Interruptible (bid) machines are never used.
    rows = vast("search", "offers", offer_query(a), "--type", "on-demand", "--order", "dph_total", "--limit", "200")
    return [o for o in (rows or []) if not o.get("is_bid") and o.get("dph_total") and o.get("cpu_cores_effective")
            and (not a.max_dph or float(o["dph_total"]) <= a.max_dph)
            and (not a.cpu_match or re.search(a.cpu_match, o.get("cpu_name") or "", re.I))
            and o.get("id") not in getattr(a, "exclude_offers", set())
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


def r2_exists(key: str) -> bool:
    return _aws("s3api", "head-object", "--bucket", os.environ["R2_BUCKET"], "--key", key, check=False).returncode == 0


def r2_put_file(key: str, path: str) -> None:
    _aws("s3", "cp", path, f"s3://{os.environ['R2_BUCKET']}/{key}", "--only-show-errors")


def r2_list(prefix: str) -> list[str]:
    return [k for k, _ in r2_list_sized(prefix)]


def r2_list_sized(prefix: str) -> list[tuple[str, int]]:
    """(key, bytes) of every object under `prefix` (the aws CLI follows every page)."""
    p = _aws("s3api", "list-objects-v2", "--bucket", os.environ["R2_BUCKET"], "--prefix", prefix, "--output", "json")
    out = p.stdout.decode().strip()
    if not out:
        return []
    return [(c["Key"], int(c.get("Size") or 0)) for c in json.loads(out).get("Contents", [])]


# ------------------------------------------------------------------ ranking

def cpu_slug(name: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "unknown").lower()).strip("-") or "unknown"


_BENCH: dict[str, float] | None = None


def load_bench(use_r2: bool) -> dict[str, float]:
    """CPU model slug -> median measured difftest matches per second per core (read once)."""
    global _BENCH
    if not use_r2:
        return {}
    if _BENCH is not None:
        return _BENCH
    per: dict[str, list[float]] = {}
    for key in r2_list("bench/"):
        raw = r2_get(key)
        try:
            b = json.loads(raw or b"")
            per.setdefault(key.split("/")[1], []).append(float(b["matches_per_sec_per_core"]))
        except (ValueError, KeyError, IndexError, TypeError):
            continue
    _BENCH = {k: statistics.median(v) for k, v in per.items() if v}
    return _BENCH


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

def instance_env(run_id: str, ncpu: int, dph: float, bin_sha: str, image: str = "", destroy_key: str = "") -> str:
    kv = {
        "RUN_ID": run_id, "NCPU": str(ncpu), "DPH": f"{dph:.5f}", "BIN_SHA": bin_sha,
        "R2_BUCKET": os.environ["R2_BUCKET"], "R2_S3_ENDPOINT": os.environ["R2_S3_ENDPOINT"],
        "RCLONE_CONFIG_R2_TYPE": "s3", "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
        "RCLONE_CONFIG_R2_ENDPOINT": os.environ["R2_S3_ENDPOINT"],
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID": os.environ["R2_ACCESS_KEY_ID"],
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": os.environ["R2_SECRET_ACCESS_KEY"],
        "RCLONE_CONFIG_R2_NO_CHECK_BUCKET": "true",
    }
    # Recorded in the run's final manifest; left out if vast's --env cannot carry it (a digest's @).
    if image and re.fullmatch(r"[A-Za-z0-9_.:/+=-]*", image):
        kv["IMAGE"] = image
    if destroy_key:
        kv["VAST_DESTROY_KEY"] = destroy_key
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


# The run's destroy-only key: the destroy endpoint and nothing else (no reads, no creating
# instances, no billing). Format: docs.vast.ai/api-reference/permissions ("api" → category →
# endpoint → optional constraints; the instance id is unknown before the create call).
DESTROY_KEY_PERMISSIONS = {"api": {"instance_write": {"api.instance.destroy": {}}}}
KEY_CHARS = r"[A-Za-z0-9_.:/+=-]+"


def parse_created_key(out: str) -> dict | None:
    """The {"id", "key"} of a `vastai create api-key` answer: vastai 1.8.3 prints
    `api-key created <the API's JSON as a Python dict>` (fields id, key, permissions)."""
    m = re.search(r"api-key created (\{.*\})", out, re.DOTALL)
    if not m:
        return None
    try:
        d = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        try:
            d = json.loads(m.group(1))
        except ValueError:
            return None
    if not isinstance(d, dict) or not isinstance(d.get("id"), int) or not isinstance(d.get("key"), str):
        return None
    if not re.fullmatch(KEY_CHARS, d["key"]):
        return None
    return {"id": d["id"], "key": d["key"]}


def create_destroy_key(run_id: str) -> dict | None:
    """Creates the run's destroy-only key ({"id", "key"}), or None (logged) if vast.ai refuses."""
    global LAST_VAST_OUTPUT
    name = f"41bots-destroy-{run_id}"
    out, err = "", ""
    with tempfile.TemporaryDirectory() as d:
        perm = os.path.join(d, "destroy-only.json")
        with open(perm, "w") as fh:
            json.dump(DESTROY_KEY_PERMISSIONS, fh)
        try:
            out = vast("create", "api-key", "--name", name, "--permission_file", perm, raw=False) or ""
        except RuntimeError as e:
            err = str(e)
    LAST_VAST_OUTPUT = ""  # it may hold the new key
    created = parse_created_key(out)
    if created:
        SECRETS.append(created["key"])
        log(f"created destroy-only API key {created['id']} ({name}, sha256 {fingerprint(created['key'])}) for the run's machines")
        return created
    if "api-key created" in out:
        why = f"vast created a key but its answer could not be read; delete the key named {name} in the vast.ai console"
    else:
        # vastai prints the API error (no key in it) on one line; never echo anything else.
        why = err or next((ln.strip() for ln in out.splitlines() if "rror" in ln), "no answer")
    log(f"::warning::no destroy-only key for this run ({redact(why)[:300]}); machines can only be destroyed by the launcher")
    return None


def delete_destroy_key(key_id: int) -> None:
    try:
        out = vast("delete", "api-key", str(key_id), raw=False) or ""
        # vastai prints the API's answer (exit status 0 even on a refusal).
        ok = not any(s in out for s in ("Failed with error", "rror", "'success': False", '"success": false'))
    except RuntimeError as e:
        out, ok = str(e), False
    if ok:
        log(f"deleted destroy-only API key {key_id}")
    else:
        log(f"::warning::could not delete destroy-only API key {key_id} ({redact(out.strip())[:200]}); "
            f"delete it in the vast.ai console (it can only destroy instances)")


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
        log(f"::group::vast {label} log of instance {instance_id} (last {lines} lines)")
        print(redact(text or "").rstrip(), flush=True)
        log("::endgroup::")


def instance_gone(instance_id: int) -> bool:
    try:
        info = vast("show", "instance", str(instance_id))
    except RuntimeError:
        return True
    return not info or (info.get("actual_status") in (None, "exited", "offline") and info.get("intended_status") == "stopped")


# Signs in an instance's status that its container cannot start: the image or its configuration
# failed on the host (e.g. 2026-10-14: "Error response from daemon: failed to create task for
# container: failed to inject ... OCI runtime create failed"). Such an instance never heartbeats.
CONTAINER_ERRORS = ("error response from daemon", "failed to inject", "oci runtime")
START_FAILED = "container failed to start"


def container_start_error(info: dict | None) -> str | None:
    """Why an instance that has not heartbeated yet cannot start, or None: a container error in
    its status message, or a stopped state (actual_status exited or offline)."""
    if not info:
        return None
    msg = str(info.get("status_msg") or "").strip().replace("\n", " ")
    if any(e in msg.lower() for e in CONTAINER_ERRORS):
        return msg[:300]
    status = info.get("actual_status")
    if status in ("exited", "offline"):
        return f"actual_status {status}" + (f": {msg[:250]}" if msg else "")
    return None


def vast_cpu_pct(info: dict | None) -> float | None:
    """The instance's CPU use in percent from vast's `cpu_util`, or None when absent. Observed on
    our rentals (2026-10-14) as a percentage of the whole container: 95.9 for a run whose own
    heartbeat said 98%, 0 for an idle one; so a small value is a small percentage, never a
    fraction. Values over 100 are clamped. The raw value is always logged too."""
    try:
        v = float((info or {}).get("cpu_util"))
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    return min(v, 100.0)


def silent_state(info: dict | None, busy_pct: float) -> tuple[str, str]:
    """What a machine whose heartbeats stopped is doing, from vast's view of it: ("gone" |
    "idle" | "busy", detail). Busy means running with CPU at or above `busy_pct`; no CPU reading
    counts as idle (destroyed at once, as before)."""
    if not info:
        return "gone", "vast no longer reports it"
    status = info.get("actual_status")
    if status != "running":
        return "gone", f"actual_status {status}"
    pct = vast_cpu_pct(info)
    raw = info.get("cpu_util")
    if pct is None:
        return "idle", "running, no CPU reading from vast"
    detail = f"running, vast CPU {pct:.0f}% (cpu_util {raw})"
    return ("busy" if pct >= busy_pct else "idle"), detail


def eta_text(eta: dict) -> str:
    """`progress.eta` for a log line: done share, ETA in UTC and Mauritius time, projected $."""
    parts = []
    frac = eta.get("done_frac")
    if isinstance(frac, (int, float)):
        parts.append(f"{100 * frac:.1f}% done")
    utc = eta.get("eta_utc")
    if isinstance(utc, str):
        try:
            t = datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            m = t.astimezone(MAURITIUS)
            parts.append(f"ETA {t:%d %b %H:%M} UTC ({m:%d %b %H:%M} Mauritius)")
        except ValueError:
            parts.append(f"ETA {utc}")
    usd = eta.get("projected_usd")
    if isinstance(usd, (int, float)):
        parts.append(f"projected ${usd:.2f}")
    if eta.get("basis"):
        parts.append(f"basis: {eta['basis']}")
    return ", ".join(parts)


def poll_line(hb: dict, minutes: float, spent: float, cpu) -> str:
    """The one log line per poll."""
    p = hb.get("progress") if isinstance(hb.get("progress"), dict) else {}
    parts = [f"heartbeat {hb.get('seq')}", f"phase {hb.get('phase')}"
             + (f" ({p['window']} {p.get('t_s')} s)" if isinstance(p.get("window"), str) else ""),
             f"{minutes:.1f} min", f"${spent:.2f} so far", f"CPU {cpu}%" if cpu is not None else "CPU ?"]
    eta = p.get("eta")
    if isinstance(eta, dict):
        text = eta_text(eta)
        if text:
            parts.append(text)
    return ", ".join(parts)


def latest_heartbeat(run_id: str) -> tuple[str | None, dict | None]:
    keys = sorted(k for k in r2_list(f"runs/{run_id}/heartbeat/") if k.endswith(".json"))
    if not keys:
        return None, None
    raw = r2_get(keys[-1])
    try:
        return keys[-1], json.loads(raw or b"")
    except ValueError:
        return keys[-1], None


def manifest_inputs(manifest: dict) -> dict[str, str]:
    """Format-2 inputs: name -> lowercase sha256 hex (the runner fetches inputs/sha256/<hex>)."""
    out = {}
    for name, h in (manifest.get("inputs") or {}).items():
        hexd = h[len("sha256:"):].lower() if isinstance(h, str) and h.startswith("sha256:") else ""
        if not re.fullmatch(r"[0-9a-f]{64}", hexd):
            sys.exit(f"launch: input {name!r} must be \"sha256:<64 hex digits>\"")
        out[name] = hexd
    return out


def cmd_run(a) -> int:
    need_env("VAST_API_KEY", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_S3_ENDPOINT")
    text = a.manifest if a.manifest.lstrip().startswith("{") else open(a.manifest, encoding="utf-8").read()
    manifest = json.loads(text)
    run_id = manifest.get("run_id") or f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    manifest["run_id"] = run_id
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        sys.exit("launch: run_id may only use letters, digits, - _ .")
    resumable = manifest.get("format") == 2
    if resumable:
        # The runner refuses binaries other than the manifest's.
        if manifest.get("bin_sha") and manifest["bin_sha"] != a.bin_sha:
            sys.exit(f"launch: manifest bin_sha {manifest['bin_sha']} differs from --bin-sha {a.bin_sha}")
        manifest["bin_sha"] = a.bin_sha
    if r2_list(f"runs/{run_id}/"):
        sys.exit(f"launch: runs/{run_id}/ already exists in R2; choose a new run_id")
    if not r2_get(f"bin/{a.bin_sha}/manifest.json"):
        sys.exit(f"launch: no release binaries at bin/{a.bin_sha}/ (run release-binaries.yml first)")
    missing = [f"{n} (inputs/sha256/{h})" for n, h in manifest_inputs(manifest).items()
               if not r2_exists(f"inputs/sha256/{h}")]
    if missing:
        sys.exit("launch: manifest inputs missing in R2: " + ", ".join(missing)
                 + "; upload each with: launch.py put-input FILE")

    # No "threads" here: the runner derives the usable cores from the container's cgroup CPU quota
    # (vast's cpu_cores_effective is a fraction of the host; NCPU below is recorded for comparison).
    manifest.setdefault("deadline_utc", datetime.fromtimestamp(time.time() + a.max_minutes * 60 - 120, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    a.exclude_machines = set()
    a.exclude_offers = set()
    best = choose(a)
    if not best:
        sys.exit("launch: no offer matches the filters")
    if a.dry_run:
        log("dry run: would upload manifest and create the instance:\n" + json.dumps(manifest, indent=2))
        return 0
    r2_put(f"runs/{run_id}/manifest.json", json.dumps(manifest, indent=2).encode())
    run_start = time.time()
    attempts: list[dict] = []
    cpu_samples: list[float] = []
    dkey = None if a.no_destroy_key else create_destroy_key(run_id)
    a.destroy_key = dkey["key"] if dkey else ""
    a.stopped = False
    if a.state_file:
        # For the workflow's cleanup step (`launch.py stop`), should this process be cancelled.
        with open(f"{a.state_file}.run-id", "w") as fh:
            fh.write(run_id)
    if dkey and a.state_file:
        # For the workflow's cleanup step, should this process not reach its own deletion.
        with open(f"{a.state_file}.destroy-key", "w") as fh:
            fh.write(str(dkey["id"]))
    try:
        final, reason = rent_and_watch(a, run_id, best, resumable, run_start, attempts, cpu_samples)
    finally:
        # A cancelled run handed to the cleanup step keeps its key until the machine is gone.
        if dkey and not getattr(a, "handed_over", False):
            delete_destroy_key(dkey["id"])
            if a.state_file:
                with contextlib.suppress(OSError):
                    os.remove(f"{a.state_file}.destroy-key")
    return report_run(a, run_id, resumable, run_start, attempts, cpu_samples, final, reason)


def rent_and_watch(a, run_id: str, best: dict | None, resumable: bool, run_start: float,
                   attempts: list[dict], cpu_samples: list[float]) -> tuple[str, str]:
    """Rents, watches and destroys machines until the run ends; returns (DONE/FAILED or "", reason)."""
    final, reason = "", ""
    while True:
        spent = sum(x["dollars_est"] for x in attempts)
        if a.max_dollars and spent >= a.max_dollars:
            reason = (f"{reason}; " if reason else "") + f"budget of ${a.max_dollars:.2f} used (${spent:.2f}): no replacement"
            log(f"::error::{reason}")
            break
        best, instance_id = rent(a, run_id, best)
        if not best:
            reason = (f"{reason}; " if reason else "") + "no offer could be rented"
            log(f"::error::{reason}")
            break
        ncpu = int(best["cpu_cores_effective"])
        dph = float(best["dph_total"])
        created = time.time()
        final, reason, replaceable, graceful = "", "", False, False
        try:
            final, reason, replaceable, graceful = watch(a, run_id, instance_id, created, run_start, cpu_samples,
                                                         dph=dph, spent_before=spent)
        except KeyboardInterrupt:
            # A cancelled workflow or Ctrl-C: let the machine upload what it finished first.
            reason, graceful = "cancelled", True
            if a.state_file:
                # Under the workflow, this process is killed seconds after the interrupt; its
                # cleanup step (`launch.py stop`) waits for the upload, then destroys the
                # machine and deletes the run's key. Here: only ask the runner to stop.
                request_stop(run_id, reason)
                a.handed_over = True
                raise
        finally:
            if not getattr(a, "handed_over", False):
                if graceful and not final:
                    # A machine still working: it uploads what it finished, then writes its marker.
                    try:
                        marker = stop_and_wait(run_id, reason, a.stop_wait_s)
                    except KeyboardInterrupt:
                        marker = ""
                        log("second interrupt: destroying now")
                    if marker:
                        final = marker
                        reason = f"{reason}; stopped by the launcher, the runner uploaded what it finished ({marker})"
                        a.stopped = True
                if final != "DONE":
                    log(f"last status before destroy: {redact(status_line(instance_status(instance_id)))}")
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
                         "gpu_name": best.get("gpu_name"), "cores_effective": ncpu, "dph": dph,
                         "minutes": round((ended - created) / 60, 1),
                         "dollars_est": round((ended - created) / 3600 * dph, 4), "ended": final or reason})
        log(f"instance {instance_id} ended: {reason}")
        # A container that never started ran nothing, so even a format-1 run is safe to retry.
        start_failed = reason.startswith(START_FAILED)
        if final or not ((resumable and replaceable) or start_failed):
            break
        if len(attempts) > MAX_REPLACEMENTS:
            reason = f"{reason}; {MAX_REPLACEMENTS} replacements used, stopping"
            log(f"::error::{reason}")
            break
        # Another machine: the same host may have the same fault.
        a.exclude_machines.add(best.get("machine_id", best.get("id")))
        best = None
        log(f"replacement {len(attempts)} of {MAX_REPLACEMENTS}: renting another machine; it resumes from the done-list")
    return final, reason


def report_run(a, run_id: str, resumable: bool, run_start: float, attempts: list[dict],
               cpu_samples: list[float], final: str, reason: str) -> int:
    log(f"run ended: {reason}")
    if not attempts:
        return 1
    result = json.loads(r2_get(f"runs/{run_id}/result.json") or b"{}")
    if resumable:
        # The merged report lists every output; the cost report keeps the per-stream summary.
        merged = json.loads(r2_get(f"runs/{run_id}/merged.json") or b"{}")
        result = {"complete": merged.get("complete"), "streams": [
            {k: st.get(k) for k in ("name", "kind", "chunks_done", "chunks_total", "left", "sprt") if k in st}
            for st in merged.get("streams", [])]}
    benches = run_benchmarks(run_id, resumable)
    bench = benches[-1] if benches else {}
    dollars = sum(x["dollars_est"] for x in attempts)
    last = attempts[-1]
    matches = result.get("matches") or 0
    report = {
        "run_id": run_id, "instance_id": last["instance_id"], "offer_id": last["offer_id"],
        "outcome": "STOPPED" if getattr(a, "stopped", False) else final or "KILLED",
        "reason": reason, "cpu_name": last["cpu_name"], "gpu_name": last["gpu_name"],
        "cores_effective": last["cores_effective"], "dph": last["dph"],
        "elapsed_min": round((time.time() - run_start) / 60, 1), "dollars_est": round(dollars, 4), "matches": matches,
        "matches_per_dollar": round(matches / dollars) if dollars > 0 and matches else None,
        "avg_cpu_pct": round(statistics.mean(cpu_samples), 1) if cpu_samples else None,
        "attempts": attempts, "replacements": len(attempts) - 1,
        "benchmark": bench, "result": result, "bin_sha": a.bin_sha, "image": a.image,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    r2_put(f"runs/{run_id}/report.json", json.dumps(report, indent=2).encode())
    for i, b in enumerate(benches):
        if b.get("matches_per_sec_per_core"):
            name = run_id if len(benches) == 1 else f"{run_id}-{i}"
            r2_put(f"bench/{cpu_slug(b.get('cpu_model') or last['cpu_name'])}/{name}.json", json.dumps(b).encode())
    print(json.dumps(report, indent=2))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(summary_markdown(run_id, report, saved_files(run_id)))
    return 0 if final == "DONE" and not getattr(a, "stopped", False) else 1


def saved_files(run_id: str) -> list[tuple[str, int, int]]:
    """(prefix, files, bytes) of what the run saved in R2: its outputs (out/) and, for stopped
    format-2 chunks, their unfinished outputs (partial/)."""
    out = []
    for sub in ("out/", "partial/"):
        prefix = f"runs/{run_id}/{sub}"
        try:
            items = r2_list_sized(prefix)
        except (RuntimeError, OSError, ValueError) as e:
            log(f"could not list {prefix}: {e}")
            continue
        if items or sub == "out/":
            out.append((prefix, len(items), sum(n for _, n in items)))
    return out


def human_bytes(n: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def summary_markdown(run_id: str, report: dict, saved: list[tuple[str, int, int]]) -> str:
    """The short markdown summary for $GITHUB_STEP_SUMMARY, then the full report folded."""
    machines = len(report.get("attempts") or [])
    lines = [f"## vast.ai run {run_id}: {report['outcome']}", "",
             f"- **Outcome:** {report['outcome']}: {report.get('reason')}",
             f"- **Time:** {report.get('elapsed_min')} minutes on {machines} machine(s); "
             f"**cost:** about ${report.get('dollars_est', 0):.2f}"]
    for prefix, files, size in saved:
        what = f"{files} file(s), {human_bytes(size)}" if files else "nothing (no outputs were saved)"
        lines.append(f"- **Saved in R2** `{prefix}`: {what}")
    lines += ["", "<details><summary>Full report</summary>", "", "```json", json.dumps(report, indent=2), "```",
              "", "</details>", ""]
    return "\n".join(lines) + "\n"


def run_benchmarks(run_id: str, resumable: bool) -> list[dict]:
    """The run's start-up benchmarks: one per machine for format 2 (benchmark/<utc>.json)."""
    keys = sorted(r2_list(f"runs/{run_id}/benchmark/")) if resumable else [f"runs/{run_id}/benchmark.json"]
    out = []
    for k in keys:
        try:
            b = json.loads(r2_get(k) or b"")
        except ValueError:
            continue
        if isinstance(b, dict):
            out.append(b)
    return out


def rent(a, run_id: str, first: dict | None = None) -> tuple[dict | None, int | None]:
    """Create the instance on the best offer (`first` if given, else a fresh search); when that
    fails (the offer was taken meanwhile, or vast refuses it), drop the offer and take the next
    best from a fresh search, up to MAX_CREATE_TRIES offers."""
    for attempt in range(1, MAX_CREATE_TRIES + 1):
        best = first if attempt == 1 and first else choose(a)
        if not best:
            return None, None
        try:
            instance_id = create_instance(a, best, run_id, int(best["cpu_cores_effective"]), float(best["dph_total"]))
            why = f"vast answered {LAST_VAST_OUTPUT!r}"
        except RuntimeError as e:
            instance_id, why = None, str(e)
        if instance_id:
            return best, instance_id
        log(f"create on offer {best['id']} failed ({why}); trying the next-best offer ({attempt}/{MAX_CREATE_TRIES})")
        a.exclude_offers.add(best["id"])
    return None, None


def create_instance(a, best: dict, run_id: str, ncpu: int, dph: float) -> int | None:
    # Entrypoint ("args") launch mode. Without --args the CLI defaults to the ssh runtype, and vast's
    # ssh/jupyter modes replace the image's ENTRYPOINT/CMD with their own init, so the boot script
    # never ran (run 38509301077). --entrypoint names the boot script; an empty --args (must be the
    # very last token) selects runtype "args". On-demand: never pass --bid_price.
    env = instance_env(run_id, ncpu, dph, a.bin_sha, a.image, getattr(a, "destroy_key", ""))
    res = vast("create", "instance", str(best["id"]), "--image", a.image, "--disk", str(a.disk_gb),
               "--env", env, "--label", f"41bots-{run_id}",
               "--entrypoint", BOOT_PATH, "--cancel-unavail", tail=("--args",))
    instance_id = res.get("new_contract") if isinstance(res, dict) else None
    if not instance_id:
        return None
    # The per-instance key vast.ai injects as CONTAINER_API_KEY; the boot script logs the same
    # fingerprint of the key it receives, so a mismatch shows in the container log.
    ikey = res.get("instance_api_key") if isinstance(res, dict) else None
    ikey_note = f"per-instance key sha256 {fingerprint(ikey)}" if ikey else "no per-instance key returned"
    log(f"created instance {instance_id} on offer {best['id']} for run {run_id} ({ikey_note})")
    if a.state_file:
        # Always the live instance, so the workflow's cleanup step destroys the right one.
        with open(a.state_file, "w") as fh:
            fh.write(str(instance_id))
    return instance_id


def request_stop(run_id: str, reason: str) -> None:
    """Asks the runner to stop: it uploads what it finished, then writes DONE or FAILED
    (crates/runner/src/stop.rs reads runs/<run_id>/STOP at every heartbeat)."""
    try:
        r2_put(f"runs/{run_id}/STOP", json.dumps({"reason": reason, "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}).encode())
        log(f"asked run {run_id} to stop ({reason}); it uploads what it finished, then ends")
    except RuntimeError as e:
        log(f"could not ask run {run_id} to stop: {e}")


def stop_and_wait(run_id: str, reason: str, wait_s: int) -> str:
    """Requests a stop (unless a marker is already there) and waits up to `wait_s` seconds for
    the runner's DONE or FAILED marker; returns the marker, or "" if none came in time."""
    for marker in ("DONE", "FAILED"):
        if r2_get(f"runs/{run_id}/{marker}") is not None:
            return marker
    request_stop(run_id, reason)
    deadline = time.time() + max(0, wait_s)
    while time.time() < deadline:
        time.sleep(STOP_POLL_S)
        for marker in ("DONE", "FAILED"):
            if r2_get(f"runs/{run_id}/{marker}") is not None:
                log(f"run {run_id} stopped and wrote {marker}")
                return marker
    log(f"::warning::run {run_id} wrote no marker within {wait_s} s of the stop request; destroying anyway")
    return ""


def watch(a, run_id: str, instance_id: int, created: float, run_start: float,
          cpu_samples: list[float], dph: float = 0.0, spent_before: float = 0.0) -> tuple[str, str, bool, bool]:
    """Watch one instance. Returns (DONE/FAILED or "", reason, whether a replacement may resume,
    whether to ask the machine to stop and upload before destroying it: a machine that is still
    working, ended for the time limit, the budget, low CPU or silent while busy; a dead, idle or
    gone one is destroyed at once). `dph` is this machine's price, `spent_before` the dollars of
    the run's earlier machines."""
    low_since, last_key, last_change = None, None, created
    warned: set[str] = set()
    silent_logged = False
    eta_warned = False
    # Heartbeats of an earlier machine of this run do not count as this one's.
    first_key, _ = latest_heartbeat(run_id)
    while True:
        time.sleep(a.poll_s)
        now = time.time()
        for marker in ("DONE", "FAILED"):
            if r2_get(f"runs/{run_id}/{marker}") is not None:
                return marker, marker, False, False
        if now - run_start > a.max_minutes * 60:
            return "", f"max_minutes ({a.max_minutes}) reached", False, True
        spent = spent_before + (now - created) / 3600 * dph
        if a.max_dollars and spent >= a.max_dollars:
            return "", f"budget reached: ${spent:.2f} of ${a.max_dollars:.2f} spent", False, True
        key, hb = latest_heartbeat(run_id)
        if key == first_key:
            key = None
        if key != last_key:
            last_key, last_change = key, now
        if key is None:
            if now - created > a.startup_minutes * 60:
                return "", f"no heartbeat within {a.startup_minutes} minutes of renting", True, False
            info = instance_status(instance_id)
            bad = container_start_error(info)
            if bad:
                log(f"::warning::instance {instance_id} cannot start its container: {redact(bad)}")
                return "", f"{START_FAILED} ({redact(bad)[:200]})", True, False
            if instance_gone(instance_id):
                return "", "instance stopped before the first heartbeat", True, False
            log(f"waiting for the machine to start (image download and boot): {redact(status_line(info))}")
            continue
        silent_s = now - last_change
        if silent_s > HEARTBEAT_STALE_S:
            info = instance_status(instance_id)
            state, detail = silent_state(info, a.busy_cpu_pct)
            if not silent_logged:
                silent_logged = True
                log(f"::warning::no new heartbeat for {silent_s / 60:.1f} minutes; vast: {redact(status_line(info))}; "
                    f"{redact(detail)}")
                print_logs(instance_id, 100)
            if state != "busy":
                return "", f"heartbeat missing for {silent_s / 60:.0f} minutes and the instance is {state} ({redact(detail)})", True, False
            waited_min = (silent_s - HEARTBEAT_STALE_S) / 60
            if waited_min >= a.silent_busy_min:
                # Still busy but silent: its uploads are stuck. Ask it to stop and upload.
                return "", (f"heartbeat missing for {silent_s / 60:.0f} minutes while the machine stayed busy "
                            f"({redact(detail)}): uploads stuck"), True, True
            log(f"::warning::no heartbeat for {silent_s / 60:.1f} min but the instance is busy ({redact(detail)}): "
                f"treating it as an upload problem; waiting {a.silent_busy_min - waited_min:.0f} more min "
                f"for heartbeats to resume (${spent:.2f} so far)")
            continue
        if silent_logged:
            silent_logged = False
            log("heartbeats resumed")
        if instance_gone(instance_id):
            return "", "instance died before DONE", True, False
        hb = hb or {}
        cpu = hb.get("cpu_util_pct")
        if cpu is None:
            cpu = hb.get("host_cpu_busy_pct")
        line = poll_line(hb, (now - run_start) / 60, spent, cpu)
        log(line if now == last_change else f"no new heartbeat for {now - last_change:.0f} s; last: {line}")
        eta = (hb.get("progress") or {}).get("eta")
        usd = eta.get("projected_usd") if isinstance(eta, dict) else None
        if (a.max_dollars and isinstance(usd, (int, float)) and spent_before + usd > 1.2 * a.max_dollars
                and not eta_warned):
            eta_warned = True
            log(f"::warning::projected cost ${spent_before + usd:.2f} is more than 20% over the budget of "
                f"${a.max_dollars:.2f} (the run stops when the budget is reached)")
        # Warnings the runner reports (e.g. a transposition table lowered to fit memory), once each.
        for w in (hb.get("progress") or {}).get("warnings") or []:
            if isinstance(w, str) and w not in warned:
                warned.add(w)
                log(f"::warning::runner: {redact(w)[:500]}")
        # A chunk may declare a stretch of deliberately partial CPU use (e.g. the solver
        # measurement, which times single-thread solves); the low-CPU rule skips it.
        partial_ok = bool((hb.get("progress") or {}).get("partial_cpu_ok"))
        if hb.get("phase") in ("benchmark", "chunk", "chunks") and cpu is not None and not partial_ok:
            cpu_samples.append(float(cpu))
            if cpu < a.low_cpu_pct:
                low_since = low_since or now
                if now - low_since >= a.low_cpu_s:
                    return "", f"CPU use below {a.low_cpu_pct:.0f}% for {a.low_cpu_s} s while work remains", False, True
            else:
                low_since = None


def cmd_plan(a) -> int:
    need_env("VAST_API_KEY")
    log("query: " + offer_query(a) + " (type on-demand)")
    return 0 if choose(a) else 1


def cmd_put_input(a) -> int:
    """Uploads a file to inputs/sha256/<hex> (if not there yet) and prints the manifest value."""
    need_env("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_S3_ENDPOINT")
    h = hashlib.sha256()
    with open(a.file, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    key = f"inputs/sha256/{h.hexdigest()}"
    if r2_exists(key):
        log(f"{key} already in R2")
    else:
        r2_put_file(key, a.file)
        log(f"uploaded {a.file} to {key}")
    print(f"sha256:{h.hexdigest()}")
    return 0


def cmd_destroy(a) -> int:
    need_env("VAST_API_KEY")
    destroy(int(a.instance_id))
    return 0


def cmd_stop(a) -> int:
    """Stops a run cleanly: asks the runner to stop (unless it has already ended), waits up to
    --wait-s seconds for it to upload what it finished and write its marker, then destroys the
    instance. The vast-run workflow's cleanup step uses it when a run is cancelled."""
    need_env("VAST_API_KEY", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_S3_ENDPOINT")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", a.run_id):
        sys.exit("launch: run_id may only use letters, digits, - _ .")
    marker = stop_and_wait(a.run_id, a.reason, a.wait_s)
    log(f"run {a.run_id}: {marker or 'no marker'}")
    if a.instance:
        destroy(int(a.instance))
    return 0


def cmd_delete_key(a) -> int:
    need_env("VAST_API_KEY")
    delete_destroy_key(int(a.key_id))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def offer_args(s):
        s.add_argument("--min-cores", type=int, default=16)
        s.add_argument("--min-ram-gb", type=int, default=32)
        s.add_argument("--disk-gb", type=int, default=20)
        s.add_argument("--max-net-cost", type=float, default=0.10, help="max $/GB up or down (owner 2026-10-14: 0.10; our traffic is tiny)")
        s.add_argument("--premium", type=float, default=0.20, help="speed premium margin (owner's 20%% rule)")
        s.add_argument("--max-dph", type=float, default=None, help="max total $/hour for the offer")
        s.add_argument("--cpu-match", default="", help="regex the offer's CPU name must match (e.g. EPYC)")
        s.add_argument("--cheapest", action="store_true", help="test runs: take the cheapest $/hour offer, not the best cost per work")
        s.add_argument("--cpu-only", action="store_true",
                       help="CPU-only job: drop the GPU compute-capability/CUDA filters (they only matter for PyTorch training)")

    s = sub.add_parser("plan", help="dry run: rank offers and print the choice")
    offer_args(s)
    s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("run", help="rent, run, watch, destroy, report")
    offer_args(s)
    s.add_argument("--manifest", required=True, help="path to a manifest JSON file, or the JSON itself")
    s.add_argument("--image", required=True)
    s.add_argument("--bin-sha", required=True)
    s.add_argument("--max-minutes", type=int, default=30)
    s.add_argument("--low-cpu-pct", type=float, default=LOW_CPU_PCT,
                   help="destroy if CPU stays below this for --low-cpu-s (lower only for runs with known serial tails)")
    s.add_argument("--low-cpu-s", type=int, default=LOW_CPU_S,
                   help="seconds of CPU below --low-cpu-pct before the machine is stopped and destroyed")
    s.add_argument("--poll-s", type=int, default=POLL_S,
                   help="seconds between checks of the machine (vast status, R2 heartbeat and markers)")
    s.add_argument("--silent-busy-min", type=float, default=SILENT_BUSY_MIN,
                   help="a machine silent for 3 minutes but busy (vast CPU >= --busy-cpu-pct) gets this many more "
                        "minutes for its heartbeats to resume before it is asked to stop and upload")
    s.add_argument("--busy-cpu-pct", type=float, default=BUSY_CPU_PCT,
                   help="vast CPU use at or above which a silent machine counts as busy")
    s.add_argument("--max-dollars", type=float, default=0.0,
                   help="stop the run (stop and upload, then destroy) once its machines have cost this much (0 = no limit)")
    s.add_argument("--startup-minutes", type=int, default=15)
    s.add_argument("--stop-wait-s", type=int, default=STOP_WAIT_S,
                   help="before destroying a machine that is still working (time limit, low CPU, Ctrl-C), ask it to "
                        "stop and wait this long for it to upload what it finished")
    s.add_argument("--state-file", help="write the instance id here (for a cleanup step)")
    s.add_argument("--dry-run", action="store_true", help="choose the offer, but create nothing")
    s.add_argument("--no-destroy-key", action="store_true",
                   help="do not create the run's destroy-only key (machines are then destroyed by the launcher only)")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("put-input", help="upload a manifest input to R2 by content hash")
    s.add_argument("file")
    s.set_defaults(fn=cmd_put_input)
    s = sub.add_parser("destroy", help="destroy one instance")
    s.add_argument("instance_id")
    s.set_defaults(fn=cmd_destroy)
    s = sub.add_parser("stop", help="ask a run to stop, wait for its upload (bounded), then destroy its instance")
    s.add_argument("run_id")
    s.add_argument("--instance", default="", help="the instance to destroy afterwards")
    s.add_argument("--wait-s", type=int, default=STOP_WAIT_S)
    s.add_argument("--reason", default="stopped by launch.py stop")
    s.set_defaults(fn=cmd_stop)
    s = sub.add_parser("delete-key", help="delete one API key by id (a run's destroy-only key)")
    s.add_argument("key_id")
    s.set_defaults(fn=cmd_delete_key)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
