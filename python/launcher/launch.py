#!/usr/bin/env python3
"""Rent one on-demand vast.ai machine, run one manifest chunk on it, watch it, destroy it.

docs/M1-SPEC.md section 7 and docs/PLAN.md section 12. Standard library only; vast.ai is driven
through the `vastai` CLI (pip package `vastai`) and R2 through the `aws` CLI (S3 API).

    launch.py plan --min-cores 190                     # dry run: rank offers, print the choice and what it would wait for
    launch.py run --manifest m.json --min-cores 190 --max-dollars 20 \
                  --image ghcr.io/guillaumeds/41bots-base:<tag> --bin-sha <git sha>
    launch.py put-input data/equity-v0.npy             # upload a manifest input by content hash

Credentials come only from the environment:
    VAST_API_KEY                                     vast.ai account key (never sent to the machine)
    R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,          the bucket-scoped R2 key (the machine gets it)
    R2_BUCKET, R2_S3_ENDPOINT

Rules applied:
- On-demand only (`--type on-demand`, never a bid price).
- Verified hosts are preferred, not required (owner, 2026-10-16: "I am OK with using unverified or
  deverified [vast.ai hosts], but prefer verified. The risk is small enough."). Every search asks
  for verified hosts first (`verified=true`); only when no verified offer passes every other filter
  (--cpu-match, GPU needs, price caps, exclusions, the reliability floor) is the search repeated
  with the verified filter lifted (`verified=any`) and the best of those taken, with a log line
  saying so. The same holds for each replacement machine. Each offer's line shows its host
  status. `--verified-only` (workflow input `verified_only`) keeps the strict behaviour. The
  GPU filter stays `compute_cap<1000`: no Blackwell cards (owner, 2026-10-16).
- Rank by estimated dollars per unit of work: dph / (effective cores x measured matches per second
  per core for that CPU model, from R2 bench/; unknown models use the median of known ones, or 1).
- Owner's rule: among offers within 20% (--premium) of the cheapest cost per unit of work, take
  the one doing the most work per hour.
- Watch through the vast API and the R2 heartbeat every --poll-s (60) seconds, never SSH. Destroy
  when: no heartbeat within --startup-minutes (the image-pull rule below can extend this);
  heartbeat older than 3 minutes; CPU use below
  --low-cpu-pct (85%) for --low-cpu-s (300) seconds while the chunk is running (unless its
  heartbeat says progress.partial_cpu_ok); the dollar cap --max-dollars ($20 by default) reached;
  DONE/FAILED seen. Always destroy, never stop. No time limit by default (owner, 2026-10-16: a run
  ends only when it finishes, at its dollar cap, or when a person stops it); --max-minutes N > 0
  sets one on purpose (then also the manifest's deadline_utc).
- Container start failures are caught at the next poll: while waiting for the first heartbeat, an
  instance whose status shows a container error ("Error response from daemon", "failed to
  inject", "OCI runtime") is destroyed at once and the next offer is rented (counted as a
  replacement, for format 1 too: nothing ran on it), instead of waiting for --startup-minutes;
  one that vast shows exited, offline or not listed is treated the same once that is confirmed.
- Image pulls take time (2026-10-16, run m3-hybrid-r1-c16m-2: the base image grew to 5-6 GB with
  PyTorch baked in, two machines with slow downlinks were still pulling it at 15 minutes and were
  destroyed, 33 minutes and $0.14 lost). So the 15 minutes (--startup-minutes) to the first
  heartbeat are counted like this: (1) once vast's status_msg reads "success, running ..." the
  container has started, and the first heartbeat is due within --startup-minutes of THAT moment;
  (2) before that, a machine that is still pulling the image when the 15 minutes from renting are
  used up is given more time, as long as vast's status_msg shows pull progress (a "Pulling fs
  layer", "Downloading", "Download complete", "Extracting" or "Pull complete" line) and has
  changed within the last --pull-stall-min (10) minutes; it is destroyed as before when its
  status stops changing for that long, or --pull-ceiling-min (40) minutes after renting. A
  machine that never showed pull progress is destroyed at 15 minutes as before; nothing is ever
  destroyed earlier than before. Each poll logs "still pulling the image (N min): <status_msg>".
- Fast downlinks are preferred: every search carries inet_down>=--min-inet-down (default 100 Mbps,
  owner 2026-10-16: the 5-6 GB image pulls in about 7 minutes at 100 Mbps; 0 = no filter; vast's offer
  field `inet_down` is in Mb/s, vastai 1.8.3 `search offers`). Unless --cheapest, an offer whose link is
  under 200 Mbps (SLOW_LINK_MBPS) has its cost per unit of work multiplied by 1 + (200 - Mbps) / 200
  (x1.5 at 100 Mbps, x2 at 0), so the ranking avoids it; an offer that reports no link speed gets no
  penalty. --offer-id reports a link slower than --min-inet-down in the log and rents it anyway (the
  owner named it).
- Floors (owner, 2026-10-16): at least --min-cores threads (default 190), at most --max-dph dollars per
  hour (default 1.70; 0 = no cap; the cap holds for --offer-id too), a GPU job at least --gpus GPUs (a
  2-GPU box is fine, the price is per machine; the trainer uses cuda:0), always the whole machine. The
  search asks vast for all of them and the launcher checks them again on what comes back, so nothing below
  a floor is ever rented, in a replacement search either; no floor is relaxed to find a machine.
- Waiting instead of failing (owner, 2026-10-16: "the launcher must wait until it gets a cheap 190+ thread
  machine; there are many, they come every few minutes"): when no offer matches (the verified search and
  the verified=any fallback both empty) the launcher does not exit. It searches again every
  --offer-poll-s (60) seconds for up to --wait-for-offer-min (120; 0 = the old behaviour: one search, at
  the cap, then "no offer matches the filters"), and logs one line per attempt: the price step, the
  attempt, the elapsed minutes and the closest miss (an offer that fits but costs more than this step,
  or the biggest one under the thread floor). The same waiting applies to every replacement search after
  a start failure. The waiting search is a price ladder, so that the cheapest is looked for first:
  --price-steps (default 1.2,1.45,1.7 $/h) with --tries-per-step (3) attempts at each step, then the cap
  (--max-dph) until the wait runs out; steps above the cap are dropped and the last rung is always the
  cap. The first offer that matches at the current rung is rented (ranked by cost per unit of work among
  the matches at that rung); a replacement search starts the ladder again from the first step. The wait
  also stops at --max-minutes and at the hand-over time. plan and --dry-run do not wait: they search at
  the first step and print what a real run would wait for, with the closest miss and what the cap holds.
  The workflow has no inputs for these settings (GitHub allows 25 inputs); the defaults apply.
- Machines that did not start are remembered: an instance that ends as "container failed to start" or that
  never heartbeats within the start timeout is recorded (machine id, host, offer, reason, time) in R2
  launcher/bad-machines.json, and every search (first, replacement, plan) leaves the machines listed in
  the last 24 hours out, logging "excluding machine M (host H): <reason> at <utc>". --offer-id on such a
  machine warns and goes ahead (the owner's explicit choice). A missing file is fine; an unreadable one
  is left alone.
- The thresholds and checks (reliability floor, download floor, network cost cap, GPU limits, what counts
  as a verified host) live in offer_rules.py, next to this file; the market tools import the same
  module for their "can the launcher rent it" labels, so the two cannot drift apart.
- No single answer from vast can say a machine is gone (2026-10-15): a failed call or an
  unreadable reply means "unknown" and changes nothing; a machine is declared gone (dead before
  DONE, stopped before its first heartbeat, destroyed itself) only after 3 consecutive readings
  showing it exited, offline or absent from the instance list, spanning at least 90 s.
- A silent machine (no new heartbeat for 3 minutes) is checked through the vast API first, and its
  container log tail printed (redacted): confirmed gone, or running but idle (vast CPU below
  --busy-cpu-pct, 50%), is destroyed at once; running and busy is treated as an upload problem
  (run m3-agree-t1-b lost 172 minutes of work that way), and so is a machine vast cannot tell us
  about: the launcher waits up to --silent-busy-min (15) more minutes for heartbeats to resume,
  logging each poll, then asks it to stop and upload as below. A failed R2 call while polling
  counts as "no new heartbeat this poll" (logged), never as an error that ends the watch.
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
  vast reports as an empty answer), the account's instances are listed first: one labelled
  41bots-<run_id> means vast rented it after all and it is adopted (never a second rental); if
  the list cannot be read either, the run stops. Otherwise the next-best offer is tried, up to 5.
- --offer-id (an offer the owner picked on vast's website): one search by that id checks the
  same things as the normal search (reliability at least 0.90, the whole machine, on-demand,
  the run's cores and RAM, and the GPU needs below); an offer that fails any of them, or that
  cannot be checked, is not rented. An unverified or deverified host is accepted (and logged)
  unless --verified-only.
- GPU jobs (network training, docs/m3-hybrid/vast.md): --gpus N (at least N GPUs on the machine; the price is
  per machine, a 2-GPU box is fine, the trainer uses cuda:0),
  --min-gpu-ram-gb, --min-compute-cap (vast's units: 600 = compute capability 6.0, the lowest the
  PyTorch 2.14.1+cu126 build supports), always with compute_cap<1000 and cuda_vers>=12.6 (the
  driver must support CUDA 12.6; no Blackwell). A manifest may carry the same needs, and the
  machine's, in a top-level "machine" object: {"gpus": 1, "min_gpu_ram_gb": 4,
  "min_compute_cap": 600, "min_cores": 128, "min_ram_gb": 64, "disk_gb": 80,
  "min_inet_down": 100}; each minimum is the larger of the manifest's and the command line's, so a
  manifest can raise min_cores above the 190 default but not lower it (to run on smaller machines lower
  the min_cores input; except min_inet_down, in Mbps: the manifest's value replaces the default of 100,
  0 = no filter, and the larger wins only if --min-inet-down is given too). Also "max_dph" (the price cap
  of 1.70: a manifest can raise or lower it), "wait_for_offer_min", "offer_poll_s", "price_steps" (a list
  of $/h) and "tries_per_step": the command line wins when it gave one, else the manifest, else the
  default. A GPU count given both ways must agree, a GPU need overrides --cpu-only, and any other key is
  refused before anything is rented. Every other rule (the preference for verified hosts above, reliability, whole machine,
  on-demand only, dollar cap) is unchanged.
- GPU models: "gpu_names" (a manifest machine key, or --gpu-names "RTX 4090,RTX 5090") is a non-empty list of GPU
  model names exactly as vast reports them in an offer's `gpu_name` ("RTX 4090", "H100 SXM", "RTX PRO 6000 S"); the
  training runs in strict float32, where GPUs of a similar price differ 2-4x in speed. When it is given, an offer is
  acceptable only if its gpu_name is in the list (case and runs of spaces ignored): in the search, in every
  replacement search, in plan and --dry-run, and for --offer-id (refused, with the reason). It is not a minimum:
  the command line wins when it gave one, else the manifest's; absent, nothing changes. An empty list, a list that
  is not one, or an entry that is not text is refused before anything is rented. The check is made on what vast
  returns (the search query does not carry it), and the waiting log's closest miss names the cheapest offer that
  fits but for its GPU model ("GPU RTX 3090 not in gpu_names").
- Destroying is verified: "destroyed (verified)" only once a later status call shows the instance
  exited, offline or absent, in two readings 10 s apart (one reading, such as an empty instance
  list, is never enough); when that call fails 3 times the destroy is reported as unverified
  (in the log, the report and the step summary), never counted as done.
- Format-2 manifests (docs/M3-SPEC.md section 7.3) resume: when the heartbeat goes missing for 3
  minutes, the machine never starts, or the instance dies before DONE, the launcher destroys it
  and rents a replacement (another machine) with the same manifest, which carries on from the
  done-list. At most 3 replacements per run, then it stops and reports. A format-2 manifest's
  `inputs` (sha256 content hashes) must already be in R2 at inputs/sha256/<hex> (`put-input`);
  the launcher checks before renting anything.
- Long runs (--handover-min): GitHub ends a job after 6 hours, so a longer run is handed over
  before that: the launcher stops watching, leaves the machine working, and writes
  runs/<run_id>/watch.json and watch/<run_id>.json (instance, price, dollar cap, deadline).
- Destroying: the launcher destroys the instance itself (on DONE/FAILED, a stale heartbeat, a
  dead instance or --max-minutes) unless it handed it over, and vast-cleanup.yml (cleanup.py,
  every 15 minutes) destroys a 41bots- instance whose run ended 10 minutes ago, went silent for
  20 minutes, never heartbeated in 60 minutes, passed its deadline or its dollar cap (after a
  stop request): that is the mechanism that does not depend on the machine. The
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
import copy
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

# The thresholds and checks for which offers may be rented live in offer_rules.py (next to this file),
# shared with the market tools so that their "the launcher can rent it" labels cannot drift.
from offer_rules import (  # noqa: E402,F401  (some names are re-exported for the tests)
    MAX_COMPUTE_CAP, MAX_DPH, MAX_NET_COST, MIN_COMPUTE_CAP, MIN_CUDA, MIN_INET_DOWN, MIN_RELIABILITY, SLOW_LINK_MBPS,
    check_failures, gpu_failures, gpu_name_failures, host_status, inet_down_of, is_verified, link_factor)

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
GONE_CONFIRMATIONS = 3  # readings showing an instance exited/offline/absent before it counts as gone
GONE_SPAN_S = 90  # ...spanning at least this long
DESTROY_RECHECK_S = 10  # destroy(): a "dead" reading counts only when a second one this much later agrees
ADOPT_READINGS = 2  # readable instance listings without the run's label before a failed create counts as "nothing rented"
MIN_CORES = 190  # default --min-cores (owner, 2026-10-16: "at least 190+"; many such machines come up every few minutes)
WAIT_FOR_OFFER_MIN = 120  # default --wait-for-offer-min: how long to look for an offer before giving up (0 = do not wait)
OFFER_POLL_S = 60  # default --offer-poll-s: seconds between two searches while waiting
PRICE_STEPS = (1.2, 1.45, 1.7)  # default --price-steps ($/h): look for the cheapest first, then raise the price (the last is the cap)
TRIES_PER_STEP = 3  # default --tries-per-step: searches at one price step before the next
BAD_MACHINES_KEY = "launcher/bad-machines.json"  # R2: machines whose container did not start (or never heartbeated)
BAD_MACHINE_HOURS = 24  # a listed machine is left out of every search for this long
BAD_MACHINE_KEEP_DAYS = 7  # entries older than this are dropped when the file is rewritten
PULL_CEILING_MIN = 40  # default; --pull-ceiling-min overrides: the latest an image pull is allowed to end, from renting
PULL_STALL_MIN = 10  # default; --pull-stall-min overrides: vast's status unchanged this long while pulling = dead
MAX_DOLLARS = 20.0  # the default dollar cap of a run (owner, 2026-10-16)
# GitHub ends a job after 6 hours (vast-run.yml timeout-minutes: 360). Longer runs are handed over
# before that: the machine keeps working and the run-aware cleanup (cleanup.py, every 15 minutes)
# watches it from then on, through runs/<run>/watch.json and watch/<run>.json.
HANDOVER = "handed over to the cleanup"
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


VAST_TIMEOUT_S = 180  # a vastai call that hangs (no answer) counts as failed; see `vast`


def vast(*args: str, raw: bool = True, check: bool = True, tail: tuple[str, ...] = (),
         timeout: float | None = VAST_TIMEOUT_S):
    # `tail` goes last: `--args` (argparse REMAINDER) swallows every token after it.
    # A hung call (vastai sets no time limit of its own) would stop the watch: no polls, no
    # hand-over before GitHub's 6-hour limit. It counts as a failed call instead, which the
    # callers already treat as "unknown". `create instance` passes timeout=None: a create cut
    # short could still rent a machine (`adopt_instance` covers an unreadable answer).
    global LAST_VAST_OUTPUT
    cmd = ["vastai", *args, "--api-key", os.environ["VAST_API_KEY"]]
    if raw:
        cmd.append("--raw")
    cmd.extend(tail)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        LAST_VAST_OUTPUT = ""
        raise RuntimeError(f"vastai {args[0]} {args[1] if len(args) > 1 else ''} gave no answer in {timeout:.0f} s")
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


def gpu_job(a) -> bool:
    return bool(getattr(a, "gpus", 0))


def offer_query(a, verified: bool = True) -> str:
    # `verified=any`, not simply leaving the token out: `vastai search offers` adds `verified=true`
    # to every query by itself (its default query), and only `verified=any` removes that filter.
    gpus = int(getattr(a, "gpus", 0) or 0)
    if gpus:
        cc = max(MIN_COMPUTE_CAP, int(getattr(a, "min_compute_cap", MIN_COMPUTE_CAP) or MIN_COMPUTE_CAP))
        gpu = [f"num_gpus>={gpus}", f"compute_cap>={cc}", f"compute_cap<{MAX_COMPUTE_CAP}", f"cuda_vers>={MIN_CUDA}"]
        if getattr(a, "min_gpu_ram_gb", 0):
            gpu.append(f"gpu_ram>={a.min_gpu_ram_gb}")
    else:
        gpu = ["num_gpus>=1"] + ([] if a.cpu_only else [f"compute_cap>={MIN_COMPUTE_CAP}",
                                                        f"compute_cap<{MAX_COMPUTE_CAP}", f"cuda_vers>={MIN_CUDA}"])
    return " ".join([
        f"reliability>{a.min_reliability}", gpu[0], "gpu_frac=1", "rentable=true",
        "verified=true" if verified else "verified=any",
        f"cpu_cores_effective>={a.min_cores}", f"cpu_ram>={a.min_ram_gb}",
    ] + gpu[1:] + [
        f"disk_space>={a.disk_gb}", f"inet_down_cost<={a.max_net_cost}", f"inet_up_cost<={a.max_net_cost}",
    ] + ([f"dph_total<={a.max_dph}"] if a.max_dph else []) + inet_down_clause(a))


def inet_down_clause(a) -> list[str]:
    """The query token asking for a fast downlink (`inet_down` is the offer's download speed in
    Mb/s), or nothing when --min-inet-down is 0. Every search carries it: the base image is big."""
    floor = inet_floor(a)
    return [f"inet_down>={floor:g}"] if floor > 0 else []


def inet_floor(a) -> float:
    """The run's minimum download speed in Mbps: --min-inet-down and/or the manifest's
    machine.min_inet_down (see apply_machine); MIN_INET_DOWN (500) when neither was given; 0 = none."""
    v = getattr(a, "min_inet_down", None)
    return MIN_INET_DOWN if v is None else max(float(v), 0.0)


def gpu_check_failures(o: dict, a) -> list[str]:
    """Why an offer fails a GPU job's needs (--gpus, --min-gpu-ram-gb, --min-compute-cap, and the
    PyTorch build's limits; offer_rules.gpu_failures), from vast's raw record (gpu_ram in MB,
    compute_cap x 100), and, whatever the job, the GPU models the run accepts (--gpu-names or the
    manifest's gpu_names; offer_rules.gpu_name_failures). Empty for an offer that passes."""
    failed = []
    if gpu_job(a):
        failed = gpu_failures(o, int(a.gpus), getattr(a, "min_gpu_ram_gb", 0), getattr(a, "min_compute_cap", MIN_COMPUTE_CAP))
    return failed + gpu_name_failures(o, getattr(a, "gpu_names", None))


def parse_gpu_names(text: str) -> list[str]:
    """--gpu-names "RTX 4090,RTX 5090" as a list of GPU model names, spelled as vast reports them."""
    names = [x.strip() for x in str(text).split(",")]
    if any(not x for x in names):
        raise argparse.ArgumentTypeError(f"must be GPU model names separated by commas, like \"RTX 4090,RTX 5090\" (got {text!r})")
    return names


# The keys of a manifest's "machine" object (what the launcher rents for it).
MACHINE_KEYS = ("gpus", "min_gpu_ram_gb", "min_compute_cap", "min_cores", "min_ram_gb", "disk_gb", "min_inet_down",
                "max_dph", "wait_for_offer_min", "offer_poll_s", "price_steps", "tries_per_step", "gpu_names")


def apply_machine(a, manifest: dict) -> None:
    """Merges a manifest's "machine" needs into the offer settings `a` (see the module notes):
    minimums take the larger value, a GPU count given both ways must agree, and a GPU need turns
    the GPU filters on even with --cpu-only. machine.min_inet_down (Mbps) replaces the default
    floor of 100, and is the larger of the two when --min-inet-down is given too. The settings that
    are not minimums (max_dph, wait_for_offer_min, offer_poll_s, price_steps, tries_per_step, gpu_names) are the
    command line's when it gave one, else the manifest's, else the default (finalize_settings; gpu_names: none).
    Exits (nothing rented) on an unknown key or a bad value (negative numbers included)."""
    m = manifest.get("machine")
    if m is None:
        return
    if not isinstance(m, dict):
        sys.exit('launch: the manifest\'s "machine" must be an object')
    bad = [k for k in m if k not in MACHINE_KEYS]
    if bad:
        sys.exit(f"launch: the manifest's \"machine\" has unknown keys {bad}; it takes only {', '.join(MACHINE_KEYS)}")
    steps = m.get("price_steps")
    if "price_steps" in m:
        if (not isinstance(steps, list) or not steps
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or x <= 0 for x in steps)):
            sys.exit(f"launch: machine.price_steps must be a list of dollars per hour above 0, like [1.2, 1.45, 1.7] (got {steps!r})")
    names = m.get("gpu_names")
    if "gpu_names" in m:
        if not isinstance(names, list) or not names or any(not isinstance(x, str) or not x.strip() for x in names):
            sys.exit(f"launch: machine.gpu_names must be a list of GPU model names as vast reports them, like "
                     f"[\"RTX 4090\", \"RTX 5090\"] (got {names!r})")
    for k, v in m.items():
        if k in ("price_steps", "gpu_names"):
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            sys.exit(f"launch: machine.{k} must be a number, 0 or more (got {v!r})")
    if "tries_per_step" in m and (not float(m["tries_per_step"]).is_integer() or m["tries_per_step"] < 1):
        sys.exit(f"launch: machine.tries_per_step must be a whole number, 1 or more (got {m['tries_per_step']!r})")
    if "offer_poll_s" in m and m["offer_poll_s"] < 1:
        sys.exit(f"launch: machine.offer_poll_s must be at least 1 second (got {m['offer_poll_s']!r})")
    if not float(m.get("gpus", 0)).is_integer():
        sys.exit(f"launch: machine.gpus must be a whole number (got {m['gpus']!r})")
    gpus = int(m.get("gpus", 0))
    if gpus and getattr(a, "gpus", 0) and int(a.gpus) != gpus:
        sys.exit(f"launch: the manifest asks for {gpus} GPUs but --gpus says {a.gpus}")
    if gpus:
        a.gpus = gpus
    for k in ("min_gpu_ram_gb", "min_compute_cap", "min_cores", "min_ram_gb", "disk_gb"):
        if k in m:
            setattr(a, k, max(m[k], getattr(a, k, 0) or 0))
    if "min_inet_down" in m:
        # The command line's default (100) is not a choice: the manifest's value replaces it (0 = no
        # filter, a lower value lets slower hosts in); a --min-inet-down given too makes the larger win.
        given = getattr(a, "min_inet_down", None)
        a.min_inet_down = float(m["min_inet_down"]) if given is None else max(float(m["min_inet_down"]), float(given))
    # max_dph (the price cap: a manifest can raise it, or lower it), the wait and the price steps: the command line
    # wins when it was given, else the manifest's value is used; whatever is still unset gets its default later.
    for k in ("max_dph", "wait_for_offer_min", "offer_poll_s", "tries_per_step"):
        if k in m and getattr(a, k, None) is None:
            setattr(a, k, m[k])
    if "price_steps" in m and getattr(a, "price_steps", None) is None:
        a.price_steps = sorted({float(x) for x in steps})
    if "gpu_names" in m and getattr(a, "gpu_names", None) is None:
        a.gpu_names = list(names)
    if gpu_job(a) and a.cpu_only:
        log(f"the run needs {a.gpus} GPU(s): the GPU filters apply (--cpu-only is ignored)")
        a.cpu_only = False


def search_offers(a, verified: bool = True) -> list[dict]:
    # --type on-demand: fixed-price rental only. Interruptible (bid) machines are never used.
    rows = vast("search", "offers", offer_query(a, verified), "--type", "on-demand", "--order", "dph_total", "--limit", "200")
    # The query asks vast for every floor, and the launcher checks them again here: whatever vast returns, an offer
    # below a floor (threads, RAM, download link, price) or of a GPU model outside gpu_names (checked only here, by
    # `gpu_check_failures`: the query does not carry it) is never rented, in a replacement search either.
    floor = inet_floor(a)
    return [o for o in (rows or []) if not o.get("is_bid") and o.get("dph_total") and o.get("cpu_cores_effective")
            and (not a.max_dph or float(o["dph_total"]) <= a.max_dph)
            and float(o["cpu_cores_effective"]) >= a.min_cores
            and (o.get("cpu_ram") is None or float(o["cpu_ram"]) >= a.min_ram_gb * 1000)
            and (floor <= 0 or inet_down_of(o) is None or inet_down_of(o) >= floor)
            and (not a.cpu_match or re.search(a.cpu_match, o.get("cpu_name") or "", re.I))
            and o.get("id") not in getattr(a, "exclude_offers", set())
            and o.get("machine_id", o.get("id")) not in getattr(a, "exclude_machines", set())
            and o.get("machine_id", o.get("id")) not in getattr(a, "bad_machines", {})
            and not gpu_check_failures(o, a)]


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


def r2_delete(key: str) -> None:
    _aws("s3", "rm", f"s3://{os.environ['R2_BUCKET']}/{key}", "--only-show-errors")


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


def link_note(o: dict) -> str:
    """The offer's download link for a log line: ", link 850 Mbps", ", link 120 Mbps (slow: cost/work x1.40)"
    or ", link speed not reported"."""
    mbps = inet_down_of(o)
    if mbps is None:
        return ", link speed not reported"
    f = o.get("_link_factor", link_factor(o))
    return f", link {mbps:.0f} Mbps" + (f" (slow: cost/work x{f:.2f})" if f > 1.0 else "")


def rank(offers: list[dict], bench: dict[str, float], premium: float,
         link_penalty: bool = True) -> tuple[dict | None, list[dict]]:
    default_rate = statistics.median(bench.values()) if bench else 1.0
    for o in offers:
        slug = cpu_slug(o.get("cpu_name"))
        o["_rate"] = bench.get(slug, default_rate)
        o["_measured"] = slug in bench
        o["_work_per_hour"] = o["cpu_cores_effective"] * o["_rate"]
        o["_link_factor"] = link_factor(o) if link_penalty else 1.0
        o["_cost_per_work"] = o["dph_total"] / o["_work_per_hour"] * o["_link_factor"]
    offers.sort(key=lambda o: (o["_cost_per_work"], o["id"]))
    if not offers:
        return None, offers
    cheapest = offers[0]["_cost_per_work"]
    near = [o for o in offers if o["_cost_per_work"] <= cheapest * (1.0 + premium)]
    best = max(near, key=lambda o: (o["_work_per_hour"], -o["_cost_per_work"]))
    return best, offers


def describe(o: dict) -> str:
    unit = "matches/s/core measured" if o["_measured"] else "assumed rate"
    host = "host verified" if is_verified(o) else f"host NOT verified ({host_status(o)})"
    return (f"offer {o['id']}: ${o['dph_total']:.3f}/h, {o['cpu_cores_effective']:.0f} eff. cores, "
            f"{o.get('cpu_name')}, {o.get('num_gpus')}x {o.get('gpu_name')}"
            + (f" ({o.get('gpu_ram')} MB, cc {o.get('compute_cap')}, CUDA {o.get('cuda_max_good')})"
               if o.get("gpu_ram") is not None else "")
            + f", {host}, rel {o.get('reliability2', 0):.3f}, "
            f"rate {o['_rate']:.2f} ({unit}), cost/work {o['_cost_per_work']:.5f}")


def offer_check_failures(o: dict, floor: float = MIN_RELIABILITY, verified_only: bool = False) -> list[str]:
    """Why an offer fails the launcher's safety checks (offer_rules.check_failures): reliability at least
    0.90, the whole machine, on-demand and rentable; with `verified_only` also a verified host. Empty when
    it passes."""
    return check_failures(o, floor, verified_only)


def named_text(o: dict) -> str:
    """One line on a named offer as vast lists it (price, threads, CPU, RAM, GPU, host, link), logged before
    its checks so that a refusal still says what the machine is."""
    def num(k, scale=1.0, fmt="{:.0f}"):
        try:
            return fmt.format(float(o[k]) / scale)
        except (KeyError, TypeError, ValueError):
            return "?"
    return (f"${num('dph_total', fmt='{:.3f}')}/h, {num('cpu_cores_effective')} eff. cores of {num('cpu_cores')}, "
            f"{o.get('cpu_name')}, {num('cpu_ram', 1000)} GB RAM, {o.get('num_gpus')}x {o.get('gpu_name')} "
            f"({num('gpu_ram', 1000)} GB, cc {o.get('compute_cap')}), host {host_status(o)}, "
            f"rel {num('reliability2', fmt='{:.3f}')}, link {num('inet_down')} Mbps, {o.get('geolocation')}")


def why_offer_missing(oid: int) -> str:
    """When the lookup by id finds nothing, a few wider searches (any host, external listings, not
    rentable, interruptible) to say why (2026-10-17: four offers the owner named all read as gone).
    Returns text to append to the refusal; never raises."""
    tries = (("any host, external and unrentable listings included", f"id={oid} verified=any external=any rentable=any", "on-demand"),
             ("interruptible", f"id={oid} verified=any external=any rentable=any", "bid"))
    found = []
    for label, q, kind in tries:
        try:
            rows = vast("search", "offers", q, "--type", kind, "--limit", "5")
        except RuntimeError as e:
            found.append(f"{label}: the search failed ({redact(str(e))[:120]})")
            continue
        hit = next((o for o in rows if isinstance(o, dict) and o.get("id") == oid), None) if isinstance(rows, list) else None
        if hit:
            found.append(f"{label}: found ({named_text(hit)}; rentable {hit.get('rentable')}, rented {hit.get('rented')}, "
                         f"external {hit.get('external')}, interruptible {hit.get('is_bid')})")
    if not found:
        return ("; no wider search finds it either (any host, external, unrentable, interruptible): the number may not be "
                "an offer id (vast's offer id is the number to rent by; the m: number is the machine)")
    return "; " + "; ".join(found)


def lookup_offer(a) -> dict:
    """The one offer the owner named (--offer-id), read back from vast by its id (one search,
    `id=N`, on-demand) and checked like every searched offer (`offer_check_failures`). Stops the
    run with a plain message when the offer is gone, fails a check, or cannot be read: the checks
    are never skipped."""
    oid = int(a.offer_id)
    try:
        # `verified=any`: vast's search adds `verified=true` to every query by itself (see `offer_query`), so
        # without it an offer on an unverified host reads as gone (2026-10-17, offer 31231107).
        rows = vast("search", "offers", f"id={oid} verified=any", "--type", "on-demand", "--limit", "5")
    except RuntimeError as e:
        sys.exit(f"launch: offer {oid} could not be checked (vast's search failed: {redact(str(e))[:200]}); the checks "
                 f"(reliability >= {MIN_RELIABILITY}, whole machine) cannot be skipped, so nothing was rented")
    if not isinstance(rows, list):
        sys.exit(f"launch: offer {oid} could not be checked (vast's search answered {LAST_VAST_OUTPUT!r}); nothing was rented")
    row = next((o for o in rows if isinstance(o, dict) and o.get("id") == oid), None)
    if row is None:
        sys.exit(f"launch: offer {oid} is not on offer (on-demand) any more; nothing was rented"
                 f"{why_offer_missing(oid)}")
    log(f"named offer {oid}: {named_text(row)}")
    verified_only = bool(getattr(a, "verified_only", False))
    failed = offer_check_failures(row, float(a.min_reliability), verified_only)
    # The run's own needs, as the normal search filters them (`offer_query`): vast's raw records
    # give cpu_ram in MB. Review E (2026-10-16): the named offer was never checked against
    # --min-ram-gb, so a table run needing 480 GB could be rented on a 251 GB machine.
    try:
        ram_mb = float(row.get("cpu_ram")) if row.get("cpu_ram") is not None else None
    except (TypeError, ValueError):
        ram_mb = None
    if ram_mb is not None and ram_mb < a.min_ram_gb * 1000:
        failed.append(f"{ram_mb / 1000:.0f} GB of RAM, below --min-ram-gb {a.min_ram_gb}")
    try:
        cores = float(row.get("cpu_cores_effective")) if row.get("cpu_cores_effective") is not None else None
    except (TypeError, ValueError):
        cores = None
    if cores is not None and cores < a.min_cores:
        failed.append(f"{cores:.0f} effective cores, below --min-cores {a.min_cores}")
    # The price cap holds for a named offer too (owner, 2026-10-16); --max-dph (or the manifest) raises it.
    cap = MAX_DPH if getattr(a, "max_dph", None) is None else float(a.max_dph)
    price = float(row.get("dph_total") or 0)
    if cap > 0 and price > cap:
        failed.append(f"price ${price:.2f}/h above the cap ${cap:.2f}/h (--max-dph raises it)")
    failed += gpu_check_failures(row, a)
    if failed:
        sys.exit(f"launch: offer {oid} fails the launcher's checks: {'; '.join(failed)}; nothing was rented")
    if not is_verified(row):
        log(f"offer {oid} is on a host that is {host_status(row)}, not verified: accepted, because the owner named it "
            f"(owner rule 2026-10-16: unverified or deverified hosts are acceptable; every other check passed)")
    load_bad_machines(a)
    bad = getattr(a, "bad_machines", {}).get(row.get("machine_id", row.get("id")))
    if bad:
        log(f"::warning::offer {oid} is on machine {row.get('machine_id')} (host {bad.get('host_id')}), which did not start a run "
            f"({bad.get('reason')} at {bad.get('utc')}); going ahead, because the owner named it")
    link = inet_down_of(row)
    floor = inet_floor(a)
    if link is not None and floor > 0 and link < floor:
        log(f"::warning::offer {oid} reports a download link of {link:.0f} Mbps, below --min-inet-down {floor:g}: "
            f"the image pull may be slow (the launcher waits while it advances, up to {getattr(a, 'pull_ceiling_min', PULL_CEILING_MIN)} minutes); "
            f"accepted, because the owner named it")
    elif link is None and floor > 0:
        log(f"offer {oid} reports no download link speed (inet_down); --min-inet-down {floor:g} could not be checked; "
            f"accepted, because the owner named it")
    best = dict(row)
    best.setdefault("machine_id", a.offer_machine or oid)
    if best.get("reliability2") is None:
        best["reliability2"] = float(best.get("reliability") or 0)
    best["dph_total"] = float(best.get("dph_total") or a.offer_dph or 0)
    best["cpu_cores_effective"] = float(best.get("cpu_cores_effective") or a.offer_cores or 0)
    if not (best["dph_total"] and best["cpu_cores_effective"]):
        sys.exit(f"launch: offer {oid}: vast reports no price or no cores for it; nothing was rented")
    for name, given, have in (("--offer-dph", a.offer_dph, best["dph_total"]),
                              ("--offer-cores", a.offer_cores, best["cpu_cores_effective"])):
        if given and abs(float(given) - have) > 0.1 * max(have, 1e-9):
            log(f"::warning::{name} {given} differs from what vast reports for offer {oid} ({have}); vast's value is used")
    return best


def choose(a, quiet: bool = False) -> dict | None:
    """The best offer for the run's filters, or None. `quiet` (repeat searches while waiting for an offer)
    leaves out the log lines about the verified/unverified fallback and the empty list when nothing matches."""
    if getattr(a, "offer_id", None):
        # An offer the owner picked on vast's own website (2026-10-15): one search by its id checks it
        # like any other (reliability, whole machine; a verified host only with --verified-only), nothing else is searched. It is offered
        # once; a failed create, or a replacement after the machine dies, finds nothing and the run stops
        # instead of falling back to a full search.
        if getattr(a, "_offer_given_used", False):
            return None
        a._offer_given_used = True
        best = lookup_offer(a)
        best["_rate"], best["_measured"] = 1.0, False
        best["_link_factor"] = 1.0  # a named offer is not ranked
        best["_work_per_hour"] = best["cpu_cores_effective"]
        best["_cost_per_work"] = best["dph_total"] / max(best["cpu_cores_effective"], 1e-9)
        log(f"offer {best['id']} chosen by the owner and checked by id (reliability "
            f"{float(best.get('reliability2') or 0):.3f}, whole machine, on-demand): {describe(best)}{link_note(best)}")
        return best
    offers = search_offers(a)
    if not offers:
        # Owner, 2026-10-16: prefer verified hosts, accept unverified or deverified ones when no
        # verified offer fits. Same filters, only `verified` lifted; a replacement machine
        # (rent() calls choose() again) starts from the verified search again.
        if getattr(a, "verified_only", False):
            if not quiet:
                log("no verified offer matches (--verified-only: unverified and deverified hosts are not searched)")
        else:
            if not quiet:
                log("no verified offer matches; searching again with unverified/deverified hosts allowed (owner rule 2026-10-16)")
            offers = search_offers(a, verified=False)
            if offers:
                n = sum(1 for o in offers if not is_verified(o))
                log(f"no verified offer matches; using unverified/deverified offers (owner rule 2026-10-16): "
                    f"{len(offers)} offers found, {n} on unverified or deverified hosts")
            elif not quiet:
                log("no offer matches even with unverified/deverified hosts allowed")
    have_r2 = all(os.environ.get(n) for n in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_S3_ENDPOINT"))
    bench = load_bench(have_r2)
    best, ranked = rank(offers, bench, a.premium, link_penalty=not a.cheapest)
    if a.cheapest and ranked:
        # Test runs: the lowest hourly price, whatever its speed.
        best = min(ranked, key=lambda o: (float(o["dph_total"]), o["id"]))
    if quiet and not ranked:
        return None
    log(f"{len(ranked)} on-demand offers with >= {a.min_cores} effective cores; {len(bench)} CPU models benchmarked")
    for o in ranked[:10]:
        log(("  * " if o is best else "    ") + describe(o) + link_note(o))
    if best:
        how = "cheapest per hour (--cheapest)" if a.cheapest else f"within {a.premium:.0%} of cheapest cost per work, fastest"
        log(f"chosen ({how}): {describe(best)}{link_note(best)}")
    return best


# ------------------------------------------------------------------ waiting for an offer, the price ladder

def parse_price_steps(text) -> list[float]:
    """--price-steps "1.2,1.45,1.7" as ascending dollars per hour; "" / "off" / "none" = no steps."""
    t = str(text).strip().lower()
    if t in ("", "off", "none"):
        return []
    try:
        steps = sorted({float(x) for x in t.split(",") if x.strip()})
    except ValueError:
        sys.exit(f"launch: --price-steps must be dollars per hour separated by commas, like 1.2,1.45,1.7 (got {text!r})")
    if any(x <= 0 for x in steps):
        sys.exit(f"launch: --price-steps must all be above 0 (got {text!r})")
    return steps


def finalize_settings(a) -> None:
    """Turns the settings left unset on the command line (None) into their defaults, once the manifest has had its
    say: price cap 1.70 $/h (0 = no cap), price steps, tries per step, how long to wait for an offer (minutes) and
    how often to search (seconds). Exits on a value that makes no sense. Says which GPU models the run accepts, if
    it names any (gpu_names)."""
    cap = getattr(a, "max_dph", None)
    a.max_dph = MAX_DPH if cap is None else float(cap)
    steps = getattr(a, "price_steps", None)
    a.price_steps = list(PRICE_STEPS) if steps is None else (list(steps) if isinstance(steps, (list, tuple)) else parse_price_steps(steps))
    for name, default in (("tries_per_step", TRIES_PER_STEP), ("wait_for_offer_min", WAIT_FOR_OFFER_MIN), ("offer_poll_s", OFFER_POLL_S)):
        v = getattr(a, name, None)
        setattr(a, name, default if v is None else v)
    if a.max_dph < 0:
        sys.exit(f"launch: --max-dph must be 0 (no cap) or more (got {a.max_dph})")
    if int(a.tries_per_step) < 1:
        sys.exit(f"launch: --tries-per-step must be at least 1 (got {a.tries_per_step})")
    if float(a.offer_poll_s) < 1:
        sys.exit(f"launch: --offer-poll-s must be at least 1 second (got {a.offer_poll_s})")
    if float(a.wait_for_offer_min) < 0:
        sys.exit(f"launch: --wait-for-offer-min must be 0 (do not wait) or more (got {a.wait_for_offer_min})")
    if getattr(a, "gpu_names", None):
        log(f"only these GPU models are rented: {', '.join(a.gpu_names)} (checked on every offer vast returns, a named offer included)")


def price_ladder(a) -> list[float]:
    """The price limits a waiting search goes through, cheapest first: the steps below the cap, then the cap itself
    (--max-dph, 1.70 by default; 0 = no cap, shown as the last rung 0). With no steps, the cap alone."""
    cap = float(a.max_dph or 0)
    return [x for x in a.price_steps if not cap or x < cap] + [cap]


def rung_text(price: float) -> str:
    return f"${price:.2f}/h" if price else "no price limit"


def with_price(a, price: float):
    """A copy of the settings whose price limit is `price` (0 = none): one rung of the ladder."""
    b = copy.copy(a)
    b.max_dph = price
    return b


def floors_text(a) -> str:
    """The floors a search holds to, for the log: threads, RAM, link, price cap, GPUs, GPU models, CPU name."""
    cap = float(a.max_dph or 0)
    parts = [f">= {a.min_cores} threads", f">= {a.min_ram_gb} GB RAM", f"reliability >= {a.min_reliability}"]
    link = inet_floor(a)
    parts.append(f"download >= {link:g} Mbps" if link > 0 else "any download speed")
    parts.append(f"price cap {rung_text(cap)}")
    if getattr(a, "gpus", 0):
        parts.append(f">= {a.gpus} GPU(s)")
    if getattr(a, "gpu_names", None):
        parts.append(f"GPU model one of {', '.join(a.gpu_names)}")
    if a.cpu_match:
        parts.append(f"CPU matching {a.cpu_match!r}")
    return ", ".join(parts)


def attempt_reaching(a, price: float, tries: int | None = None) -> int | None:
    """The attempt of a waiting search at which the price limit first reaches `price`, or None above the cap."""
    tries = int(tries if tries is not None else a.tries_per_step)
    for i, rung in enumerate(price_ladder(a)):
        if not rung or price <= rung:
            return i * tries + 1
    return None


def closest_miss(a, rung: float) -> str:
    """One log phrase about the nearest thing to a match, from one extra search that keeps every filter except that it
    lifts the price to the cap and halves the thread floor (so the cheapest 200 offers are not all small machines):
    the cheapest offer that meets everything but this step's price, the biggest one under the thread floor, and, when
    the run names GPU models (gpu_names), the cheapest offer that meets everything at this step but its GPU model."""
    b = copy.copy(a)
    floor = int(a.min_cores)
    b.min_cores = max(1, floor // 2)
    b.max_dph = float(a.max_dph or 0)
    names = getattr(a, "gpu_names", None)
    b.gpu_names = None  # lifted for this look, so that an offer of another GPU model can be named
    try:
        rows = search_offers(b, verified=False)
    except RuntimeError as e:
        return f"closest miss unknown (the search failed: {redact(str(e))[:120]})"
    cores = lambda o: float(o["cpu_cores_effective"])  # noqa: E731
    dph = lambda o: float(o["dph_total"])  # noqa: E731
    other_gpu = [o for o in rows if gpu_name_failures(o, names)]
    rows = [o for o in rows if not gpu_name_failures(o, names)]
    wrong_gpu = [o for o in other_gpu if cores(o) >= floor and (not rung or dph(o) <= rung)]
    over_price = [o for o in rows if cores(o) >= floor and rung and dph(o) > rung]
    under_floor = [o for o in rows if cores(o) < floor and (not rung or dph(o) <= rung)]
    parts = []
    if wrong_gpu:
        o = min(wrong_gpu, key=lambda o: (dph(o), o["id"]))
        more = f" (and {len(wrong_gpu) - 1} more like it)" if len(wrong_gpu) > 1 else ""
        parts.append(f"offer {o['id']} has {cores(o):.0f} threads and costs ${dph(o):.2f}/h but {gpu_name_failures(o, names)[0]}{more}")
    if over_price:
        o = min(over_price, key=lambda o: (dph(o), o["id"]))
        n = attempt_reaching(a, dph(o))
        parts.append(f"offer {o['id']} has {cores(o):.0f} threads but costs ${dph(o):.2f}/h, above this step's {rung_text(rung)}"
                     + (f" (the ladder reaches it at attempt {n})" if n else ""))
    if under_floor:
        o = max(under_floor, key=lambda o: (cores(o), -dph(o)))
        parts.append(f"offer {o['id']} costs ${dph(o):.2f}/h but has only {cores(o):.0f} threads (floor {floor})")
    if not parts:
        return (f"closest miss: none (no offer meets the other filters even with the price at the cap and "
                f"the thread floor halved to {b.min_cores})")
    return "closest miss: " + "; ".join(parts)


def choose_or_wait(a) -> dict | None:
    """`choose`, and when no offer matches, wait and search again instead of giving up (owner, 2026-10-16: "the launcher
    must wait until it gets a cheap 190+ thread machine; there are many, they come every few minutes").

    Every attempt is one search at one rung of the price ladder: --tries-per-step (3) attempts at each price step
    (--price-steps 1.2,1.45,1.7 $/h), --offer-poll-s (60) seconds apart, then the cap (--max-dph, 1.70) until
    --wait-for-offer-min (120) minutes are used up; the first offer that matches at the current rung is rented
    (ranked by cost per unit of work among the matches at that rung). Every floor of the first search (threads, RAM,
    GPU, link, reliability, the price cap) holds for every later one, a replacement search included: nothing is
    relaxed to find a machine, and each replacement search starts the ladder again from the first step.
    --wait-for-offer-min 0 is the old behaviour: one search, at the cap. A named offer (--offer-id) is looked at once.
    Returns None when the wait is used up (or the run's own time limit comes first)."""
    if getattr(a, "offer_id", None):
        return choose(a)
    load_bad_machines(a)
    ladder = price_ladder(a)
    wait_min, poll, tries = float(a.wait_for_offer_min), float(a.offer_poll_s), int(a.tries_per_step)
    if wait_min <= 0:
        return choose(with_price(a, ladder[-1]))
    start = time.time()
    end, why_end = start + wait_min * 60, f"{wait_min:g} min"
    rs = getattr(a, "run_start", None)
    if rs is not None:
        for label, t in (("the run's time limit (--max-minutes)", a.max_minutes * 60 if getattr(a, "max_minutes", 0) > 0 else None),
                         ("the hand-over time", a.handover_min * 60 if getattr(a, "handover_min", 0) else None)):
            if t is not None and rs + t < end:
                end, why_end = rs + t, label
    attempt, failures = 0, 0
    while True:
        attempt += 1
        step = min((attempt - 1) // tries, len(ladder) - 1)
        rung = ladder[step]
        elapsed = (time.time() - start) / 60
        try:
            best = choose(with_price(a, rung), quiet=attempt > 1)
            failures = 0
        except RuntimeError as e:
            failures += 1
            if attempt == 1 or failures >= 5:
                raise
            log(f"::warning::search failed at attempt {attempt} ({redact(str(e))[:200]}); counted as no offer")
            best = None
        if best:
            if attempt > 1:
                log(f"offer found at price step {step + 1}/{len(ladder)} ({rung_text(rung)}) after waiting {elapsed:.1f} min "
                    f"(attempt {attempt})")
            return best
        now = time.time()
        if now >= end:
            log(f"gave up waiting for an offer after {(now - start) / 60:.0f} min ({attempt} attempts; limit: {why_end})")
            return None
        if attempt == 1:
            log(f"no offer matches now; waiting up to {wait_min:g} min for one, searching every {poll:g} s "
                f"({tries} attempts per price step, steps {', '.join(rung_text(x) for x in ladder)}); floors: {floors_text(a)}")
        log(f"waiting for an offer: price step {step + 1}/{len(ladder)} ({rung_text(rung)}), attempt {attempt}, "
            f"{elapsed:.1f} of {wait_min:g} min; no match; {closest_miss(a, rung)}")
        time.sleep(min(poll, end - now))


def plan_search(a) -> dict | None:
    """What a run would do first, without waiting or renting (plan and --dry-run): the search at the first price step;
    if that finds nothing, say what a real run would wait for, with the closest miss, and look once at the price cap to
    show whether a real run would rent something after climbing the ladder. Returns the offer found, if any."""
    load_bad_machines(a)
    ladder = price_ladder(a)
    wait_min = float(a.wait_for_offer_min)
    first = ladder[0] if wait_min > 0 else ladder[-1]
    best = choose(with_price(a, first))
    if best or getattr(a, "offer_id", None):
        return best
    if wait_min <= 0:
        log(f"no offer matches (floors: {floors_text(a)}); with --wait-for-offer-min 0 a run stops here")
        return None
    log(f"no offer matches at the first price step ({rung_text(first)}); a run would wait up to {wait_min:g} min for one, "
        f"searching every {float(a.offer_poll_s):g} s, {int(a.tries_per_step)} attempts per price step "
        f"({', '.join(rung_text(x) for x in ladder)}); floors: {floors_text(a)}; {closest_miss(a, first)}")
    if len(ladder) > 1:
        log(f"looking at the price cap {rung_text(ladder[-1])} to see what the ladder would reach")
        best = choose(with_price(a, ladder[-1]), quiet=True)
        if best:
            n = attempt_reaching(a, float(best["dph_total"]))
            log(f"at the price cap the best offer is {describe(best)}; a run would reach it at attempt {n} "
                f"(about {(n - 1) * float(a.offer_poll_s) / 60:.0f} min from the start)")
        else:
            log("nothing matches even at the price cap right now")
    return best


# ------------------------------------------------------------------ machines that did not start

def have_r2_env() -> bool:
    return all(os.environ.get(n) for n in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_S3_ENDPOINT"))


def utc_to_epoch(text) -> float | None:
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def epoch_to_utc(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_bad_machines() -> tuple[list[dict], bool]:
    """(entries, readable) of R2 launcher/bad-machines.json. A missing file is readable and empty; a file that is
    there but cannot be read (R2 error, not JSON) is not readable, and nothing is written over it."""
    if not have_r2_env():
        return [], False
    try:
        raw = r2_get(BAD_MACHINES_KEY)
        if raw is None:
            return [], not r2_exists(BAD_MACHINES_KEY)
        d = json.loads(raw)
    except (RuntimeError, OSError, ValueError) as e:
        log(f"::warning::could not read {BAD_MACHINES_KEY} ({redact(str(e))[:150]}); no machine is excluded on its account")
        return [], False
    ms = d.get("machines") if isinstance(d, dict) else None
    if not isinstance(ms, list):
        log(f"::warning::{BAD_MACHINES_KEY} has no machine list; ignored")
        return [], False
    return [m for m in ms if isinstance(m, dict) and m.get("machine_id") is not None], True


def load_bad_machines(a) -> None:
    """Puts the machines listed in launcher/bad-machines.json in the last 24 hours into `a.bad_machines` (machine id ->
    entry), which every search leaves out (the first, the replacements, plan); each is logged once per run."""
    entries, _ = read_bad_machines()
    now = time.time()
    bad = {}
    for e in entries:
        t = utc_to_epoch(e.get("utc"))
        if t is not None and now - t <= BAD_MACHINE_HOURS * 3600:
            bad[e["machine_id"]] = e
    a.bad_machines = bad
    logged = getattr(a, "_bad_logged", None)
    if logged is None:
        logged = a._bad_logged = set()
    for mid, e in bad.items():
        if mid not in logged:
            logged.add(mid)
            log(f"excluding machine {mid} (host {e.get('host_id')}): {e.get('reason')} at {e.get('utc')}")


NO_START_PREFIX = "no heartbeat within"  # the reasons watch() gives when the first heartbeat never came


def start_failure_kind(reason: str) -> str | None:
    """"container failed to start" or "never heartbeated" for the reasons of a machine that did not get going,
    else None (a machine that failed later, or finished)."""
    if reason.startswith(START_FAILED):
        return "container failed to start"
    if reason.startswith(NO_START_PREFIX):
        return "never heartbeated"
    return None


def record_bad_machine(run_id: str, instance_id: int, best: dict, kind: str, detail: str) -> None:
    """Remembers a machine whose container did not start (or that never heartbeated) in R2
    launcher/bad-machines.json, so that no search leaves it in for 24 hours (read, add, write; a missing file is
    fine, an unreadable one is left alone). A failed write is only logged."""
    if not have_r2_env():
        return
    entries, readable = read_bad_machines()
    if not readable:
        log(f"::warning::{BAD_MACHINES_KEY} could not be read, so machine {best.get('machine_id')} is not recorded")
        return
    mid = best.get("machine_id", best.get("id"))
    now = time.time()
    keep = []
    for e in entries:
        t = utc_to_epoch(e.get("utc"))
        if e.get("machine_id") != mid and t is not None and now - t <= BAD_MACHINE_KEEP_DAYS * 86400:
            keep.append(e)
    keep.append({"machine_id": mid, "host_id": best.get("host_id"), "offer_id": best.get("id"), "instance_id": instance_id,
                 "run_id": run_id, "reason": kind, "detail": redact(detail)[:300], "utc": epoch_to_utc(now)})
    try:
        r2_put(BAD_MACHINES_KEY, json.dumps({"format": 1, "machines": keep}, indent=2).encode())
    except (RuntimeError, OSError) as e:
        log(f"::warning::could not write {BAD_MACHINES_KEY}: {redact(str(e))[:150]}")
        return
    log(f"remembered machine {mid} (host {best.get('host_id')}) as bad for {BAD_MACHINE_HOURS} h: {kind}")


# ------------------------------------------------------------------ run

def instance_env(run_id: str, ncpu: int, dph: float, bin_sha: str, image: str = "", destroy_key: str = "",
                 offer: dict | None = None) -> str:
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
    # The offer, host and machine numbers, for the runner's machine-cost record (machines/<run>.json).
    for key, field in (("OFFER_ID", "id"), ("HOST_ID", "host_id"), ("MACHINE_ID", "machine_id")):
        if offer and offer.get(field) is not None:
            kv[key] = str(offer[field])
    for k, v in kv.items():
        if not re.fullmatch(r"[A-Za-z0-9_.:/+=-]*", v):
            sys.exit(f"launch: environment value for {k} has characters vast's --env cannot carry")
    return " ".join(f"-e {k}={v}" for k, v in kv.items())


def destroy_reading(instance_id: int) -> tuple[str, dict | None]:
    """One reading for destroy(): like instance_state(), except that "dead" must be seen twice,
    DESTROY_RECHECK_S seconds apart. A single listing has been wrong (2026-10-16, 09:56 UTC: vast's
    `show instances` held no 41bots instance, or returned an empty list, while the machine was
    running and sending heartbeats), and destroy() decides on one reading where the Liveness class
    needs three over 90 s. If the second reading shows the instance alive, that is the answer (the
    first reading is forgotten); if it cannot be read ("unknown"), that is the answer: unknown is
    never dead."""
    state, info = instance_state(instance_id)
    if state != "dead":
        return state, info
    log(f"instance {instance_id} reads dead ({info_word(info)}); reading again in {DESTROY_RECHECK_S} s to be sure")
    time.sleep(DESTROY_RECHECK_S)
    state, info = instance_state(instance_id)
    if state == "alive":
        log(f"instance {instance_id} reads alive on the second reading ({info_word(info)}): the first was wrong")
    return state, info


def destroy(instance_id: int) -> str:
    """Destroys the instance and checks that it is gone. Returns "verified" (a later status call
    showed it exited, offline or absent from the instance list), "unverified" (the destroy call
    went through, but vast's status could not be read in 3 tries: check the console) or "failed"
    (the destroy call itself failed 5 times and the instance was not seen gone). An unreadable
    status is never counted as destroyed.

    A destroy call that fails is not the end of it: the machine may already have destroyed itself
    (the runner does at the end of a run, or a stop raced it), and then vast no longer knows the
    instance, so every destroy call fails. After each failed call the status is read; if that
    reading shows the instance gone ("dead": exited, offline or absent from a listing that was
    read successfully) it counts as verified. If the status cannot be read ("unknown"), or shows
    the instance still there, the destroy is tried again.

    "Dead" counts only when two readings DESTROY_RECHECK_S seconds apart both say dead (see
    destroy_reading): one reading is never enough, after a failed call or after a successful one."""
    for attempt in range(5):
        try:
            # -y: without it vastai asks "Are you sure?" on stdin, reads EOF and silently does nothing.
            vast("destroy", "instance", str(instance_id), "-y", raw=False)
        except RuntimeError as e:
            log(f"destroy attempt {attempt + 1} failed: {e}")
            state, info = destroy_reading(instance_id)
            if state == "dead":
                log(f"instance {instance_id} was already gone when the destroy call failed "
                    f"({info_word(info)}); counting it as destroyed (verified)")
                return "verified"
            time.sleep(10)
            continue
        time.sleep(5)
        unread = 0
        while unread < 3:
            state, info = destroy_reading(instance_id)
            if state == "unknown":
                unread += 1
                time.sleep(10)
                continue
            if state == "dead":
                log(f"destroyed instance {instance_id} (verified: {info_word(info)})")
                return "verified"
            log(f"destroy attempt {attempt + 1}: instance {instance_id} still {info_word(info)}")
            break
        else:
            log(f"::warning::destroy of instance {instance_id} was sent, but vast's status could not be read in "
                f"3 tries, so it is NOT verified: check the vast.ai console (the cleanup workflow also checks)")
            return "unverified"
    log(f"::error::could not destroy instance {instance_id}; destroy it by hand in the vast.ai console")
    return "failed"


def info_word(info: dict | None) -> str:
    if info is None:
        return "not listed"
    return f"actual_status {info.get('actual_status')}"


def instance_listed(instance_id: int) -> dict | None:
    """The instance's row in `vastai show instances`: {} when a successful listing does not hold
    it (not listed), None when the listing failed or could not be read (unknown)."""
    try:
        rows = vast("show", "instances")
    except RuntimeError as e:
        log(f"show instances failed: {e}")
        return None
    if not isinstance(rows, list):
        return None
    for r in rows:
        if isinstance(r, dict) and r.get("id") == instance_id:
            return r
    return {}


def instance_state(instance_id: int) -> tuple[str, dict | None]:
    """One reading of an instance: ("alive" | "dead" | "unknown", vast's record). Dead means
    vast shows it exited or offline, or a successful listing of the account's instances does not
    hold it. Unknown means no call could be read; it is never taken for dead.

    A record whose actual_status is None is not decided on its own: a machine still loading shows
    None, and so does one that was just destroyed (seen 2026-10-16, run m3-t1-leader-layouts-9654:
    five destroy calls went through, every follow-up reading said "actual_status None", and the
    job failed). So the account's instance list is read: not listed (in a list that was read
    successfully) is dead; listed is judged by the listed row, where None is alive (loading); an
    unreadable list is unknown. A record that says intended_status "stopped" is dead as before."""
    info = instance_status(instance_id)
    if info is not None and info.get("actual_status") is None and info.get("intended_status") != "stopped":
        info = None
    if info is None:
        row = instance_listed(instance_id)
        if row is None:
            return "unknown", None
        if not row:
            return "dead", None
        info = row
    status = info.get("actual_status")
    if status in ("exited", "offline", "destroyed") or (status is None and info.get("intended_status") == "stopped"):
        return "dead", info
    return "alive", info


class Liveness:
    """Whether one instance is still there, read so that no single answer from vast can say
    "gone": it is declared gone only after GONE_CONFIRMATIONS consecutive readings showing it dead
    (exited, offline or absent from the instance list) spanning at least GONE_SPAN_S seconds. A
    reading that shows it alive clears the count; a failed call or an unreadable reply is
    "unknown" and changes nothing."""

    def __init__(self, instance_id: int):
        self.id = instance_id
        self.count = 0
        self.first: float | None = None
        self.detail = ""

    def check(self) -> tuple[str, dict | None]:
        """("alive" | "gone" | "dying" | "unknown", vast's record); dying = dead by this reading
        but not yet confirmed."""
        now = time.time()
        state, info = instance_state(self.id)
        if state == "unknown":
            self.detail = "vast could not be read"
            return "unknown", None
        if state == "alive":
            self.count, self.first = 0, None
            self.detail = ""
            return "alive", info
        self.count += 1
        self.first = now if self.first is None else self.first
        span = now - self.first
        self.detail = f"{info_word(info)}: reading {self.count} of {GONE_CONFIRMATIONS} over {span:.0f} s"
        if self.count >= GONE_CONFIRMATIONS and span >= GONE_SPAN_S:
            return "gone", info
        return "dying", info


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


# Signs in an instance's status that its container cannot start: the image or its configuration
# failed on the host (e.g. 2026-10-14: "Error response from daemon: failed to create task for
# container: failed to inject ... OCI runtime create failed"). Such an instance never heartbeats.
CONTAINER_ERRORS = ("error response from daemon", "failed to inject", "oci runtime")
START_FAILED = "container failed to start"


def container_start_error(info: dict | None) -> str | None:
    """Why an instance that has not heartbeated yet cannot start, or None: a container error in
    its status message. (An exited or offline state is not read here: one reading never counts,
    `Liveness` confirms it.) No record means unknown, not an error."""
    if not info:
        return None
    msg = str(info.get("status_msg") or "").strip().replace("\n", " ")
    if any(e in msg.lower() for e in CONTAINER_ERRORS):
        return msg[:300]
    return None


# What vast's status_msg looks like while it pulls the image: docker's own progress lines, one per
# layer ("<id>: Pulling fs layer", "Waiting", "Downloading [===>  ] 12MB/90MB", "Verifying
# Checksum", "Download complete", "Extracting ...", "Pull complete", "Already exists"). A started
# container reads "success, running <image> ...".
PULL_WORDS = ("pulling", "downloading", "download complete", "pull complete", "extracting",
              "verifying checksum", "already exists")
LAYER_WAITING = re.compile(r"\b[0-9a-f]{6,}: waiting\b")
STARTED_PREFIX = "success, running"


def status_msg_of(info: dict | None) -> str | None:
    """vast's status_msg of an instance record as one trimmed string, or None when the record has
    none to read (no record, or a record without that key, such as a row of the instance list)."""
    if not info or "status_msg" not in info:
        return None
    return str(info.get("status_msg") or "").strip()


def pull_message(msg: str) -> bool:
    """Whether a status_msg shows the image being pulled."""
    low = msg.lower()
    return any(w in low for w in PULL_WORDS) or bool(LAYER_WAITING.search(low))


def container_started(msg: str) -> bool:
    """Whether a status_msg says the container started ("success, running ...")."""
    return msg.lower().startswith(STARTED_PREFIX)


class StartClock:
    """When the first heartbeat of one machine is due, given how far its start has got (2026-10-16:
    the base image grew to 5-6 GB and machines with slow downlinks were destroyed while still
    pulling it). Fed one reading of vast's record per poll (`see`), asked once per poll whether to
    give up (`verdict`).

    - Container started (status_msg "success, running ..."; latched): the first heartbeat is due
      within `startup_min` minutes of that moment.
    - Not started, `startup_min` minutes from renting not used up: keep waiting, as before.
    - Not started, those minutes used up: the machine is given more time only if it showed image
      pull progress and vast's status_msg changed within the last `stall_min` minutes, and only up
      to `ceiling_min` minutes from renting. Otherwise it is given up on, with the old reason
      ("no heartbeat within N minutes of renting") plus the detail.
    A reading without a status_msg (vast unreadable, or a list row) changes nothing. Nothing here
    ever gives up earlier than the old rule (N minutes from renting) did."""

    def __init__(self, created: float, startup_min: float, ceiling_min: float = PULL_CEILING_MIN,
                 stall_min: float = PULL_STALL_MIN):
        self.created = created
        self.startup_min = startup_min
        self.base_s = startup_min * 60
        self.ceiling_s = max(ceiling_min, startup_min) * 60
        self.stall_s = stall_min * 60
        self.started_at: float | None = None
        self.msg: str | None = None
        self.last_change = created
        self.seen_pull = False  # a pull-progress message was read at some poll
        self.pulling = False  # ...and the latest one is such a message

    def see(self, now: float, info: dict | None) -> None:
        msg = status_msg_of(info)
        if msg is None:
            return
        if msg != self.msg:
            self.msg, self.last_change = msg, now
        self.pulling = pull_message(msg)
        self.seen_pull = self.seen_pull or self.pulling
        if self.started_at is None and container_started(msg):
            self.started_at = now

    def verdict(self, now: float) -> str | None:
        """None while the machine may go on starting, else why it is given up on."""
        n = self.startup_min
        if self.started_at is not None:
            if now - self.started_at > self.base_s:
                return f"no heartbeat within {n} minutes of the container starting"
            return None
        if now - self.created <= self.base_s:
            return None
        if not self.seen_pull:
            return f"no heartbeat within {n} minutes of renting"
        quiet = now - self.last_change
        if quiet >= self.stall_s:
            return (f"no heartbeat within {n} minutes of renting: the image pull stalled "
                    f"(vast's status unchanged for {quiet / 60:.0f} minutes)")
        if now - self.created > self.ceiling_s:
            return (f"no heartbeat within {n} minutes of renting: the image was still pulling "
                    f"{self.ceiling_s / 60:g} minutes after renting (the ceiling)")
        return None

    def line(self, now: float, info: dict | None, state: str, detail: str) -> str:
        """The log line of one poll while the first heartbeat is awaited."""
        minutes = (now - self.created) / 60
        if self.started_at is None and self.pulling:
            text = f"still pulling the image ({minutes:.0f} min): {redact(self.msg or '')[:300]}"
            if now - self.created > self.base_s:
                text += (f"; past the usual {self.startup_min} minutes, waiting while the pull moves "
                         f"(gives up after {self.stall_s / 60:g} min unchanged, or {self.ceiling_s / 60:g} min in all)")
        else:
            text = f"waiting for the machine to start (image download and boot): {redact(status_line(info))}"
            if self.started_at is not None:
                text += (f"; container started {(now - self.started_at) / 60:.0f} min ago, first heartbeat due within "
                         f"{self.startup_min} minutes of that")
        return text + (f"; {detail}" if state != "alive" else "")


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


def busy_threshold(a) -> float:
    """The vast CPU reading at or above which a silent machine counts as busy (an upload problem,
    given time) rather than idle (destroyed at once): --busy-cpu-pct, but never above the run's
    own --low-cpu-pct. A run declared to use few cores (e.g. --low-cpu-pct 1: 32 of 190 threads
    busy, about 17%) would otherwise read as idle and lose its unsaved work the moment its
    heartbeat uploads hang, the failure that cost $18 on 2026-10-14 (review E, 2026-10-16)."""
    return min(float(a.busy_cpu_pct), float(getattr(a, "low_cpu_pct", a.busy_cpu_pct)))


def silent_state(info: dict | None, busy_pct: float) -> tuple[str, str]:
    """What a machine whose heartbeats stopped is doing, from one reading of vast's record of it
    (a reading that shows it alive, see `Liveness`): ("unknown" | "idle" | "busy", detail). Busy
    means running with CPU at or above `busy_pct`; running with no CPU reading counts as idle
    (destroyed at once, as before); no record, or a state other than running (loading, created),
    is unknown and gets the busy machine's grace."""
    if not info:
        return "unknown", "vast gave no readable answer"
    status = info.get("actual_status")
    if status != "running":
        return "unknown", f"actual_status {status}"
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
    steps = p.get("chunk_steps")
    if isinstance(steps, dict):
        for name, st in sorted(steps.items()):
            if isinstance(st, dict) and isinstance(st.get("step"), str):
                parts.append(f"{name}: {st['step']}" + step_detail(st))
    eta = p.get("eta")
    if isinstance(eta, dict):
        text = eta_text(eta)
        if text:
            parts.append(text)
    return ", ".join(parts)


def step_detail(st: dict) -> str:
    """The counts of a chunk's step for a log line (`progress.chunk_steps`, e.g. a hybrid_rung's
    label rows, golden worlds or training epoch), or ""."""
    bits = []
    for done, total, word in (("rows_done", "rows_total", "rows"), ("worlds_done", "worlds_total", "worlds"),
                              ("train_batches_done", "train_batches", "batches")):
        if isinstance(st.get(done), int) and isinstance(st.get(total), int):
            bits.append(f"{st[done]:,}/{st[total]:,} {word}")
    if isinstance(st.get("train_epoch"), int):
        bits.insert(0, f"epoch {st['train_epoch'] + 1}" + (f"/{st['train_epochs']}"
                                                          if isinstance(st.get("train_epochs"), int) else ""))
    if isinstance(st.get("device"), str):
        bits.append(f"on {st['device']}")
    return f" ({', '.join(bits)})" if bits else ""


def latest_heartbeat(run_id: str) -> tuple[str | None, dict | None]:
    """(key, record) of the run's newest heartbeat, or (None, None). Raises RuntimeError when R2
    cannot be listed; `watch` treats that as no new heartbeat this poll."""
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


# hybrid_rung settings that name a stored input by its key (crates/runner/src/hybrid.rs): the
# chunk fetches them itself from inputs/sha256/<hex>, so they are checked here with the
# manifest's `inputs`, before anything is rented.
HYBRID_INPUT_PARAMS = ("harvest_net", "golden_decisions_input", "golden_worlds_input",
                       "audit_predicates_input", "t1_decisions_input")
# The trick-2 loop's rough test: a list of {"run": LABEL, "input": KEY} (the saved trick-1 worlds).
HYBRID_INPUT_LISTS = ("t1_saved_inputs",)


def param_inputs(manifest: dict) -> dict[str, str]:
    """Stored inputs named in hybrid_rung streams' params: "<stream>.<param>" (a list's entries as
    "<stream>.<param>[<run>]") -> lowercase sha256 hex. A value that is not
    "inputs/sha256/<64 hex>" (or "sha256:<64 hex>") stops the launch."""
    out = {}
    key = re.compile(r"(?:inputs/sha256/|sha256:)([0-9a-fA-F]{64})")
    for st in manifest.get("streams") or []:
        if not isinstance(st, dict) or st.get("kind") != "hybrid_rung":
            continue
        params = st.get("params") or {}
        named = [(k, params.get(k)) for k in HYBRID_INPUT_PARAMS]
        for k in HYBRID_INPUT_LISTS:
            entries = params.get(k)
            if entries is None:
                continue
            if not isinstance(entries, list):
                sys.exit(f"launch: stream {st.get('name')!r}: params.{k} must be a list of {{\"run\", \"input\"}}")
            for e in entries:
                run = e.get("run") if isinstance(e, dict) else None
                named.append((f"{k}[{run}]", e.get("input") if isinstance(e, dict) else e))
        for k, v in named:
            if v is None:
                continue
            m = key.fullmatch(v) if isinstance(v, str) else None
            if not m:
                sys.exit(f"launch: stream {st.get('name')!r}: params.{k} must be \"inputs/sha256/<64 hex digits>\"")
            out[f"{st.get('name')}.{k}"] = m.group(1).lower()
    return out


# The keys of a format-2 stream (crates/runner/src/sched.rs STREAM_KEYS); the runner refuses
# any other, so a manifest with one is refused here before anything is rented.
STREAM_KEYS = ("name", "priority", "kind", "params", "chunks")
PARAMS_KEYS = ("threads", "chunk_sizes", "est_seconds", "blocks_per_chunk", "sprt")


def manifest_stream_errors(manifest: dict) -> list[str]:
    """Stream keys the runner refuses (review E, 2026-10-16: a stream-level "threads" was
    ignored, so every chunk ran on one thread). Before the runner refused them, a refused
    manifest also cost up to 4 rentals: the runner exits at once, the machine destroys itself
    before its first heartbeat, and the launcher took that for a start failure and rented again."""
    out = []
    for st in manifest.get("streams") or []:
        if not isinstance(st, dict):
            out.append("a stream is not an object")
            continue
        for k in st:
            if k not in STREAM_KEYS:
                hint = f" (it belongs in params: params.{k})" if k in PARAMS_KEYS else ""
                out.append(f"stream {st.get('name')!r}: unknown key {k!r}{hint}")
    return out


def cmd_run(a) -> int:
    need_env("VAST_API_KEY", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_S3_ENDPOINT")
    text = a.manifest if a.manifest.lstrip().startswith("{") else open(a.manifest, encoding="utf-8").read()
    manifest = json.loads(text)
    run_id = manifest.get("run_id") or f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    manifest["run_id"] = run_id
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        sys.exit("launch: run_id may only use letters, digits, - _ .")
    apply_machine(a, manifest)
    resumable = manifest.get("format") == 2
    if resumable:
        bad = manifest_stream_errors(manifest)
        if bad:
            sys.exit("launch: the runner would refuse this manifest (nothing rented): " + "; ".join(bad))
        # The runner refuses binaries other than the manifest's.
        if manifest.get("bin_sha") and manifest["bin_sha"] != a.bin_sha:
            sys.exit(f"launch: manifest bin_sha {manifest['bin_sha']} differs from --bin-sha {a.bin_sha}")
        manifest["bin_sha"] = a.bin_sha
    if r2_list(f"runs/{run_id}/"):
        sys.exit(f"launch: runs/{run_id}/ already exists in R2; choose a new run_id")
    if not r2_get(f"bin/{a.bin_sha}/manifest.json"):
        sys.exit(f"launch: no release binaries at bin/{a.bin_sha}/ (run release-binaries.yml first)")
    wanted = {**manifest_inputs(manifest), **(param_inputs(manifest) if resumable else {})}
    missing = [f"{n} (inputs/sha256/{h})" for n, h in wanted.items()
               if not r2_exists(f"inputs/sha256/{h}")]
    if missing:
        sys.exit("launch: manifest inputs missing in R2: " + ", ".join(missing)
                 + "; upload each with: launch.py put-input FILE")

    # No "threads" here: the runner derives the usable cores from the container's cgroup CPU quota
    # (vast's cpu_cores_effective is a fraction of the host; NCPU below is recorded for comparison).
    # No time limit unless one is asked for (--max-minutes > 0): the owner's rule (2026-10-16) is
    # that a run ends when it finishes, at its dollar cap, or when a person stops it.
    if a.max_minutes > 0:
        manifest.setdefault("deadline_utc", datetime.fromtimestamp(time.time() + a.max_minutes * 60 - 120, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    a.deadline_utc = manifest.get("deadline_utc")
    a.exclude_machines = set()
    a.exclude_offers = set()
    a.run_start = time.time()
    finalize_settings(a)
    if a.dry_run:
        # Nothing is waited for or rented: say what a real run would do.
        best = plan_search(a)
        if not best:
            if float(a.wait_for_offer_min) <= 0:
                sys.exit("launch: no offer matches the filters")
            log("dry run: no offer to rent right now; a real run would wait as described above")
        log("dry run: would upload manifest and create the instance:\n" + json.dumps(manifest, indent=2))
        return 0
    best = choose_or_wait(a)
    if not best:
        sys.exit("launch: no offer matches the filters")
    r2_put(f"runs/{run_id}/manifest.json", json.dumps(manifest, indent=2).encode())
    run_start = time.time()
    attempts: list[dict] = []
    cpu_samples: list[float] = []
    dkey = None if a.no_destroy_key else create_destroy_key(run_id)
    a.destroy_key = dkey["key"] if dkey else ""
    a.destroy_key_id = dkey["id"] if dkey else None
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
        a.known_instances = getattr(a, "known_instances", set()) | {instance_id}
        ncpu = int(best["cpu_cores_effective"])
        dph = float(best["dph_total"])
        created = time.time()
        write_watch(a, run_id, instance_id, best, dph, spent, created)
        final, reason, replaceable, graceful = "", "", False, False
        self_destroyed = None
        destroyed = "left running"  # handed over: the cleanup watches it
        try:
            final, reason, replaceable, graceful = watch(a, run_id, instance_id, created, run_start, cpu_samples,
                                                         dph=dph, spent_before=spent)
            if reason == HANDOVER:
                hand_over(a, run_id, instance_id, best, dph, spent, created)
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
                    # The runner destroys itself after uploading; give it a few readings over 90 s
                    # (gone only when confirmed, `Liveness`), then make sure.
                    self_destroyed = False
                    live = Liveness(instance_id)
                    for _ in range(4):
                        if live.check()[0] == "gone":
                            self_destroyed = True
                            break
                        time.sleep(30)
                    log(f"instance {instance_id} {'destroyed itself (' + live.detail + ')' if self_destroyed else 'did NOT destroy itself (confirmed) within 90 s of ' + final}")
                destroyed = destroy(instance_id)
        ended = time.time()
        attempts.append({"instance_id": instance_id, "offer_id": best["id"], "cpu_name": best.get("cpu_name"),
                         "gpu_name": best.get("gpu_name"), "cores_effective": ncpu, "dph": dph,
                         "minutes": round((ended - created) / 60, 1),
                         "dollars_est": round((ended - created) / 3600 * dph, 4), "ended": final or reason,
                         "self_destroyed": self_destroyed, "destroy": destroyed})
        if reason == HANDOVER:
            log(f"instance {instance_id} keeps working; the vast-cleanup workflow watches it from now on")
            break
        log(f"instance {instance_id} ended: {reason}")
        kind = None if final else start_failure_kind(reason)
        if kind:
            # Its container did not start, or it never heartbeated: no search takes this machine again for 24 hours.
            record_bad_machine(run_id, instance_id, best, kind, reason)
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
        # A stop request written for the machine just destroyed (e.g. silent while busy) would
        # stop the replacement at its first heartbeat: it would write its marker at once and the
        # run would end with its work undone (review E, 2026-10-16).
        clear_stop(run_id)
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
        "outcome": ("STOPPED" if getattr(a, "stopped", False) else final
                    or ("HANDED_OVER" if reason == HANDOVER else "KILLED")),
        "self_destroyed": last.get("self_destroyed"),
        "destroy": last.get("destroy"),
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
    if reason != HANDOVER:
        finalise_machine_record(run_id, report["outcome"], attempts)
    for i, b in enumerate(benches):
        if b.get("matches_per_sec_per_core"):
            name = run_id if len(benches) == 1 else f"{run_id}-{i}"
            r2_put(f"bench/{cpu_slug(b.get('cpu_model') or last['cpu_name'])}/{name}.json", json.dumps(b).encode())
    print(json.dumps(report, indent=2))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(summary_markdown(run_id, report, saved_files(run_id)))
    return 0 if (final == "DONE" or reason == HANDOVER) and not getattr(a, "stopped", False) else 1


def finalise_machine_record(run_id: str, outcome: str, attempts: list[dict]) -> None:
    """Marks the run's machine-cost record (machines/<run>.json) final when the runner could not
    (a machine destroyed silent, idle or dead): before (review E, 2026-10-16) such a record stayed
    `running` for ever, and `machines.py --live` listed the machine as running, stale. Minutes and
    dollars come from the run's last heartbeat, else the launcher's own times (cleanup.py's rule)."""
    try:
        import cleanup  # the same directory: one rule for the launcher and the cleanup
        rec = json.loads(r2_get(f"machines/{run_id}.json") or b"{}")
        if not isinstance(rec, dict) or not rec or rec.get("state") == "final":
            return
        _, hb = latest_heartbeat(run_id)
        last = attempts[-1]
        w = {"dph": last["dph"], "spent_before": round(sum(x["dollars_est"] for x in attempts[:-1]), 4),
             "created_utc": datetime.fromtimestamp(time.time() - last["minutes"] * 60, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "instance_id": last["instance_id"], "offer_id": last["offer_id"], "cpu_name": last.get("cpu_name")}
        out = cleanup.finalise_machine_record(rec, run_id, outcome, w, hb or {}, time.time())
        if out is not None:
            out["finalised_by"] = "launcher"
            r2_put(f"machines/{run_id}.json", json.dumps(out, indent=2).encode())
            log(f"machine record of {run_id} marked final ({outcome}): {out['minutes']} min, ${out['usd']}")
    except (RuntimeError, OSError, ValueError, KeyError) as e:
        log(f"::warning::could not mark the machine record of {run_id} final: {e}")


def watch_record(a, run_id: str, instance_id: int, best: dict, dph: float, spent_before: float,
                 created: float, handed_over: bool) -> dict:
    return {"run_id": run_id, "instance_id": instance_id, "offer_id": best.get("id"), "cpu_name": best.get("cpu_name"),
            "dph": dph, "max_dollars": a.max_dollars, "spent_before": round(spent_before, 4),
            "created_utc": datetime.fromtimestamp(created, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "deadline_utc": getattr(a, "deadline_utc", None), "handed_over": handed_over,
            "busy_cpu_pct": busy_threshold(a), "low_cpu_pct": a.low_cpu_pct,
            "destroy_key_id": getattr(a, "destroy_key_id", None),
            "github_run": os.environ.get("GITHUB_RUN_ID"),
            "utc": datetime.fromtimestamp(time.time(), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def write_watch(a, run_id: str, instance_id: int, best: dict, dph: float, spent_before: float, created: float) -> None:
    """runs/<run>/watch.json: what the cleanup needs to enforce the run's dollar cap and deadline,
    should the launcher be gone (written for every machine, handed over or not)."""
    try:
        r2_put(f"runs/{run_id}/watch.json",
               json.dumps(watch_record(a, run_id, instance_id, best, dph, spent_before, created, False), indent=2).encode())
    except RuntimeError as e:
        log(f"::warning::could not write runs/{run_id}/watch.json: {e}")


def hand_over(a, run_id: str, instance_id: int, best: dict, dph: float, spent_before: float, created: float) -> None:
    """Leaves the machine working: the cleanup watches it from now on (watch/<run>.json lists it),
    and the workflow's safety-net steps must not destroy it or delete its key."""
    rec = json.dumps(watch_record(a, run_id, instance_id, best, dph, spent_before, created, True), indent=2).encode()
    r2_put(f"runs/{run_id}/watch.json", rec)
    r2_put(f"watch/{run_id}.json", rec)
    a.handed_over = True
    if a.state_file:
        for suffix in ("", ".run-id", ".destroy-key"):
            with contextlib.suppress(OSError):
                os.remove(f"{a.state_file}{suffix}")
    spent = spent_before + (time.time() - created) / 3600 * dph
    log(f"::notice::handed instance {instance_id} over after {a.handover_min} minutes (${spent:.2f} so far): it runs "
        f"on until DONE{', its deadline ' + a.deadline_utc if getattr(a, 'deadline_utc', None) else ''} or its cap "
        f"${a.max_dollars:.2f}; the "
        f"vast-cleanup workflow (every 15 minutes) watches it and writes runs/{run_id}/closed.json when it is gone")


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
    unverified = [str(x.get("instance_id")) for x in report.get("attempts") or [] if x.get("destroy") in ("unverified", "failed")]
    if unverified:
        lines.append(f"- **WARNING: destroy not verified** for instance(s) {', '.join(unverified)}: check the vast.ai console "
                     f"(the vast-cleanup workflow also checks every 15 minutes)")
    for prefix, files, size in saved:
        what = f"{files} file(s), {human_bytes(size)}" if files else "nothing (no outputs were saved)"
        lines.append(f"- **Saved in R2** `{prefix}`: {what}")
    lines += ["", "<details><summary>Full report</summary>", "", "```json", json.dumps(report, indent=2), "```",
              "", "</details>", ""]
    return "\n".join(lines) + "\n"


def run_benchmarks(run_id: str, resumable: bool) -> list[dict]:
    """The run's start-up benchmarks: one per machine for format 2 (benchmark/<utc>.json)."""
    try:
        keys = sorted(r2_list(f"runs/{run_id}/benchmark/")) if resumable else [f"runs/{run_id}/benchmark.json"]
    except (RuntimeError, OSError, ValueError) as e:
        log(f"could not list the run's benchmarks: {e}")
        return []
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
        try:
            best = first if attempt == 1 and first else choose_or_wait(a)
        except RuntimeError as e:
            # e.g. vast's daily search quota (429): nothing more is rented; the run's report is
            # still written (before, the exception skipped it).
            log(f"::error::the offer search failed ({redact(str(e))[:200]}); nothing more is rented")
            return None, None
        if not best:
            return None, None
        try:
            instance_id = create_instance(a, best, run_id, int(best["cpu_cores_effective"]), float(best["dph_total"]))
            why = f"vast answered {LAST_VAST_OUTPUT!r}"
        except RuntimeError as e:
            instance_id, why = None, str(e)
        if instance_id:
            return best, instance_id
        # vast may have rented the machine although its answer could not be read: never rent twice.
        adopted = adopt_instance(a, run_id, best)
        if adopted:
            return best, adopted
        log(f"create on offer {best['id']} failed ({why}); trying the next-best offer ({attempt}/{MAX_CREATE_TRIES})")
        a.exclude_offers.add(best["id"])
    return None, None


def adopt_instance(a, run_id: str, best: dict) -> int | None:
    """After a create call whose answer could not be read (or that failed): lists the account's
    instances and adopts one labelled 41bots-<run_id>, which means vast rented it after all.
    Returns None when a readable listing holds none. When the listing cannot be read in 3 tries
    the run stops: renting another offer could be a second rental."""
    label = f"41bots-{run_id}"
    # Never an earlier machine of this run (a replacement is being rented because it died).
    known = getattr(a, "known_instances", set())
    # A machine vast is still creating may be missing from the first listing: "nothing rented" is
    # concluded only after ADOPT_READINGS readable listings 10 s apart (else a second offer could
    # be rented while the first comes up: two machines on one run).
    readable = 0
    for attempt in range(3 + ADOPT_READINGS):
        rows = None
        try:
            rows = vast("show", "instances")
        except RuntimeError as e:
            log(f"show instances failed ({attempt + 1}): {e}")
        if isinstance(rows, list):
            for r in rows:
                if (isinstance(r, dict) and r.get("label") == label and r.get("id") and int(r["id"]) not in known
                        and r.get("actual_status") not in ("exited", "offline", "destroyed")):
                    iid = int(r["id"])
                    log(f"adopting instance {iid}: vast rented it for run {run_id} (offer {best['id']}) although its "
                        f"answer to the create call could not be read")
                    if a.state_file:
                        with open(a.state_file, "w") as fh:
                            fh.write(str(iid))
                    return iid
            readable += 1
            if readable >= ADOPT_READINGS:
                return None
        time.sleep(10)
        if attempt >= 2 and readable == 0:
            break
    sys.exit(f"launch: the create call's answer could not be read, and neither could the instance list, so the "
             f"launcher cannot tell whether a machine was rented; renting another could be a second rental. "
             f"Check the vast.ai console for an instance labelled {label}.")


def create_instance(a, best: dict, run_id: str, ncpu: int, dph: float) -> int | None:
    # Entrypoint ("args") launch mode. Without --args the CLI defaults to the ssh runtype, and vast's
    # ssh/jupyter modes replace the image's ENTRYPOINT/CMD with their own init, so the boot script
    # never ran (run 38509301077). --entrypoint names the boot script; an empty --args (must be the
    # very last token) selects runtype "args". On-demand: never pass --bid_price.
    env = instance_env(run_id, ncpu, dph, a.bin_sha, a.image, getattr(a, "destroy_key", ""), offer=best)
    res = vast("create", "instance", str(best["id"]), "--image", a.image, "--disk", str(a.disk_gb),
               "--env", env, "--label", f"41bots-{run_id}",
               "--entrypoint", BOOT_PATH, "--cancel-unavail", tail=("--args",), timeout=None)
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


def clear_stop(run_id: str) -> None:
    """Removes runs/<run_id>/STOP (before a replacement machine is rented)."""
    try:
        if r2_exists(f"runs/{run_id}/STOP"):
            r2_delete(f"runs/{run_id}/STOP")
            log(f"removed the stop request of run {run_id}, so the replacement does not stop at once")
    except RuntimeError as e:
        log(f"::warning::could not remove runs/{run_id}/STOP ({e}); the replacement may stop at once")


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
    last_hb: dict | None = None
    warned: set[str] = set()
    silent_logged = False
    eta_warned = False
    live = Liveness(instance_id)
    start = StartClock(created, a.startup_minutes, getattr(a, "pull_ceiling_min", PULL_CEILING_MIN),
                       getattr(a, "pull_stall_min", PULL_STALL_MIN))
    # Heartbeats of an earlier machine of this run do not count as this one's.
    first_key = None
    for attempt in range(3):
        try:
            first_key, _ = latest_heartbeat(run_id)
            break
        except (RuntimeError, OSError, ValueError) as e:
            log(f"::warning::could not read the run's heartbeats ({attempt + 1}/3): {redact(str(e))[:200]}")
            time.sleep(5)
    while True:
        time.sleep(a.poll_s)
        now = time.time()
        for marker in ("DONE", "FAILED"):
            if r2_get(f"runs/{run_id}/{marker}") is not None:
                return marker, marker, False, False
        if a.max_minutes > 0 and now - run_start > a.max_minutes * 60:
            return "", f"max_minutes ({a.max_minutes}) reached", False, True
        spent = spent_before + (now - created) / 3600 * dph
        if a.max_dollars and spent >= a.max_dollars:
            return "", f"budget reached: ${spent:.2f} of ${a.max_dollars:.2f} spent", False, True
        if getattr(a, "handover_min", 0) and now - run_start > a.handover_min * 60:
            return "", HANDOVER, False, False
        try:
            key, hb = latest_heartbeat(run_id)
        except (RuntimeError, OSError, ValueError) as e:
            # R2 could not be read: no new heartbeat this poll, nothing more.
            log(f"::warning::could not read the run's heartbeats this poll ({redact(str(e))[:200]}); "
                f"counting it as no new heartbeat")
            key, hb = last_key, last_hb
        if key == first_key:
            key = None
        if key != last_key:
            last_key, last_change = key, now
        last_hb = hb
        if key is None:
            state, info = live.check()
            bad = container_start_error(info)
            if bad:
                log(f"::warning::instance {instance_id} cannot start its container: {redact(bad)}")
                return "", f"{START_FAILED} ({redact(bad)[:200]})", True, False
            if state == "gone":
                # Nothing ran on it, so even a format-1 run is safe to retry on another offer.
                return "", f"{START_FAILED}: instance gone before the first heartbeat ({live.detail})", True, False
            start.see(now, info)
            why = start.verdict(now)
            if why:
                log(f"::warning::giving up on instance {instance_id}: {why}; last status: {redact(status_line(info))}")
                return "", why, True, False
            log(start.line(now, info, state, live.detail))
            continue
        silent_s = now - last_change
        if silent_s > HEARTBEAT_STALE_S:
            state, info = live.check()
            if state == "alive":
                state, detail = silent_state(info, busy_threshold(a))
            else:
                detail = live.detail
            if not silent_logged:
                silent_logged = True
                log(f"::warning::no new heartbeat for {silent_s / 60:.1f} minutes; vast: {redact(status_line(info))}; "
                    f"{state}: {redact(detail)}")
                print_logs(instance_id, 100)
            if state in ("gone", "idle"):
                return "", f"heartbeat missing for {silent_s / 60:.0f} minutes and the instance is {state} ({redact(detail)})", True, False
            # Busy (an upload problem), dying (dead by one reading, not yet confirmed) or unknown
            # (vast could not be read): give it time, then ask it to stop and upload.
            waited_min = (silent_s - HEARTBEAT_STALE_S) / 60
            if waited_min >= a.silent_busy_min:
                return "", (f"heartbeat missing for {silent_s / 60:.0f} minutes while the machine stayed {state} "
                            f"({redact(detail)}): uploads stuck"), True, True
            log(f"::warning::no heartbeat for {silent_s / 60:.1f} min but the instance is {state} ({redact(detail)}): "
                f"treating it as an upload problem; waiting {a.silent_busy_min - waited_min:.0f} more min "
                f"for heartbeats to resume (${spent:.2f} so far)")
            continue
        if silent_logged:
            silent_logged = False
            log("heartbeats resumed")
        state, _ = live.check()
        if state == "gone":
            return "", f"instance died before DONE ({live.detail})", True, False
        if state == "dying":
            log(f"instance {instance_id} looks dead ({live.detail}); not yet confirmed")
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
    if a.manifest:
        text = a.manifest if a.manifest.lstrip().startswith("{") else open(a.manifest, encoding="utf-8").read()
        apply_machine(a, json.loads(text))
    a.exclude_machines = set()
    a.exclude_offers = set()
    finalize_settings(a)
    log("query: " + offer_query(with_price(a, price_ladder(a)[0])) + " (type on-demand; the first price step)")
    if not getattr(a, "verified_only", False) and not getattr(a, "offer_id", None):
        log("if no verified offer matches, the search is repeated with verified=any (owner rule 2026-10-16; --verified-only turns that off)")
    return 0 if plan_search(a) or float(a.wait_for_offer_min) > 0 else 1


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
    return 0 if destroy(int(a.instance_id)) == "verified" else 1


def cmd_stop(a) -> int:
    """Stops a run cleanly: asks the runner to stop (unless it has already ended), waits up to
    --wait-s seconds for it to upload what it finished and write its marker, then destroys the
    instance. The vast-run workflow's cleanup step uses it when a run is cancelled."""
    need_env("VAST_API_KEY", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_S3_ENDPOINT")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", a.run_id):
        sys.exit("launch: run_id may only use letters, digits, - _ .")
    marker = stop_and_wait(a.run_id, a.reason, a.wait_s)
    log(f"run {a.run_id}: {marker or 'no marker'}")
    if a.instance and destroy(int(a.instance)) != "verified":
        return 1
    return 0


def cmd_delete_key(a) -> int:
    need_env("VAST_API_KEY")
    delete_destroy_key(int(a.key_id))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def offer_args(s):
        s.add_argument("--min-cores", type=int, default=MIN_CORES,
                       help=f"effective cores (threads) the machine must have (default {MIN_CORES}, owner 2026-10-16); the launcher "
                            "waits for such a machine rather than take a smaller one, a replacement search included")
        s.add_argument("--min-ram-gb", type=int, default=32)
        s.add_argument("--min-reliability", type=float, default=MIN_RELIABILITY,
                       help="the host reliability floor (default 0.90, the owner's rule of 2026-10-16); raise it for a run that wants a steadier host")
        s.add_argument("--disk-gb", type=int, default=20)
        s.add_argument("--max-net-cost", type=float, default=MAX_NET_COST, help="max $/GB up or down (owner 2026-10-14: 0.10; our traffic is tiny)")
        s.add_argument("--min-inet-down", type=float, default=None,
                       help="minimum download speed in Mbps an offer must report (vast's inet_down; default 500: the base "
                            "image is several GB; 0 = no filter; a manifest's machine.min_inet_down replaces the default, "
                            "or the larger of the two wins when both are given). Under 200 Mbps the ranking adds a "
                            "penalty (without --cheapest); --offer-id only reports a slower link")
        s.add_argument("--premium", type=float, default=0.20, help="speed premium margin (owner's 20%% rule)")
        s.add_argument("--max-dph", type=float, default=None,
                       help=f"price cap in $/hour for every search and for --offer-id (default {MAX_DPH:.2f}, owner 2026-10-16; "
                            "0 = no cap); the price steps below it are tried first")
        s.add_argument("--price-steps", default=None,
                       help="look for the cheapest first: the price limits ($/h, ascending, comma-separated) of the waiting "
                            "search, --tries-per-step attempts at each, then the cap (--max-dph) until the wait runs out "
                            f"(default {','.join(f'{x:g}' for x in PRICE_STEPS)}; 'off' = the cap alone)")
        s.add_argument("--tries-per-step", type=int, default=None,
                       help=f"attempts at each price step before the next (default {TRIES_PER_STEP}), --offer-poll-s apart")
        s.add_argument("--wait-for-offer-min", type=float, default=None,
                       help=f"when no offer matches, wait and search again for up to this many minutes (default {WAIT_FOR_OFFER_MIN}; "
                            "0 = do not wait: one search at the cap, then stop). Also applies to every replacement search")
        s.add_argument("--offer-poll-s", type=float, default=None,
                       help=f"seconds between two searches while waiting for an offer (default {OFFER_POLL_S})")
        s.add_argument("--cpu-match", default="", help="regex the offer's CPU name must match (e.g. EPYC)")
        s.add_argument("--cheapest", action="store_true", help="test runs: take the cheapest $/hour offer, not the best cost per work")
        s.add_argument("--cpu-only", action="store_true",
                       help="CPU-only job: drop the GPU compute-capability/CUDA filters (they only matter for PyTorch training)")
        s.add_argument("--gpus", type=int, default=0,
                       help="GPU job: at least this many GPUs on the machine (0 = no GPU need; the manifest's machine.gpus also sets it)")
        s.add_argument("--min-gpu-ram-gb", type=int, default=0, help="GPU job: memory per GPU, in GB")
        s.add_argument("--gpu-names", type=parse_gpu_names, default=None,
                       help="rent only these GPU models, comma-separated and spelled as vast reports them in an offer's gpu_name, "
                            "e.g. \"RTX 4090,RTX 5090\" (case and extra spaces ignored); the manifest's machine.gpu_names "
                            "applies when this is not given; checked on every offer, --offer-id included")
        s.add_argument("--min-compute-cap", type=int, default=MIN_COMPUTE_CAP,
                       help="GPU job: lowest compute capability x 100 (default 600 = 6.0, Pascal; below 1000 always)")
        s.add_argument("--verified-only", action="store_true",
                       help="rent only from vast.ai-verified hosts (the strict behaviour). Without it, verified hosts are "
                            "still preferred, but when no verified offer fits, unverified or deverified hosts are searched "
                            "too (owner rule 2026-10-16); this also applies to --offer-id")
        s.add_argument("--offer-id", type=int, default=None,
                       help="rent exactly this vast.ai offer: one search by its id checks it (reliability >= 0.90, whole "
                            "machine, on-demand; an unverified or deverified host is accepted unless --verified-only) and "
                            "nothing else is searched; an offer failing a check is not rented")
        s.add_argument("--offer-dph", type=float, default=None,
                       help="with --offer-id: the price per hour you saw (optional; a warning if vast reports another)")
        s.add_argument("--offer-cores", type=int, default=None,
                       help="with --offer-id: the CPU cores you saw (optional; a warning if vast reports another)")
        s.add_argument("--offer-machine", type=int, default=None, help="with --offer-id: its machine id (optional)")
        s.add_argument("--offer-cpu", default="", help="with --offer-id: the CPU name, for the report (optional)")

    s = sub.add_parser("plan", help="dry run: rank offers and print the choice")
    offer_args(s)
    s.add_argument("--manifest", default="", help="optional: a manifest (path or JSON) whose \"machine\" needs apply")
    s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("run", help="rent, run, watch, destroy, report")
    offer_args(s)
    s.add_argument("--manifest", required=True, help="path to a manifest JSON file, or the JSON itself")
    s.add_argument("--image", required=True)
    s.add_argument("--bin-sha", required=True)
    s.add_argument("--max-minutes", type=int, default=0,
                   help="a time limit on purpose (stop, upload, destroy); 0 (the default) = none: the owner's rule is "
                        "that a run ends when it finishes, at its dollar cap, or when a person stops it")
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
    s.add_argument("--max-dollars", type=float, default=MAX_DOLLARS,
                   help=f"stop the run (stop and upload, then destroy) once its machines have cost this much "
                        f"(default ${MAX_DOLLARS:.0f}; 0 = no limit)")
    s.add_argument("--startup-minutes", type=int, default=15,
                   help="minutes to the first heartbeat: from renting, or from the container starting once vast's "
                        "status says 'success, running' (a machine still pulling the image is given longer, see "
                        "--pull-ceiling-min)")
    s.add_argument("--pull-ceiling-min", type=float, default=PULL_CEILING_MIN,
                   help="a machine still pulling the image when --startup-minutes from renting are used up is given "
                        "more time while vast's status shows the pull advancing, but never past this many minutes "
                        "from renting")
    s.add_argument("--pull-stall-min", type=float, default=PULL_STALL_MIN,
                   help="...and it is destroyed when vast's status has not changed for this many minutes")
    s.add_argument("--stop-wait-s", type=int, default=STOP_WAIT_S,
                   help="before destroying a machine that is still working (time limit, low CPU, Ctrl-C), ask it to "
                        "stop and wait this long for it to upload what it finished")
    s.add_argument("--state-file", help="write the instance id here (for a cleanup step)")
    s.add_argument("--handover-min", type=float, default=0,
                   help="after this many minutes, stop watching but leave the machine working: the vast-cleanup "
                        "workflow watches it from then on (GitHub ends jobs at 6 hours; 0 = never hand over)")
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
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
