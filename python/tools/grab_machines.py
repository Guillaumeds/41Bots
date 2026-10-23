"""Grab chosen vast.ai machines the moment they are rentable, and keep them (owner 2026-10-23).

    R2_S3_ENDPOINT=$R2_ENDPOINT python3 python/tools/grab_machines.py [--once] [--dry-run] [--max-held 3]

One pass over the wanted machines (in priority order). A machine that is rentable now, under its price cap, and not yet
grabbed is rented on that very offer through the `vast-run.yml` workflow with a tiny hold job (`difftest`) and
`machine.keep: true`: the job ends in about two minutes and the machine stays up (Jupyter + SSH, the Connect button) for
`keep_hours`, billing its hourly rate, until a later run is deployed onto it (`machine.instance_id`) or the cleanup
destroys it. State is kept in R2 (`grab/state.json`), so a restart of the session never grabs a machine twice.
`--max-held` caps how many held machines are billing at once (a held machine bills while idle).
"""

import argparse
import json
import subprocess
import sys
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "launcher"))
import launch  # noqa: E402

# machine id, what it is, the price cap ($/h) above which it is not taken
WANTED = [
    (68507, "RTX 3060 + AMD EPYC 7B12 (256 threads, 129 GB), Thailand, verified: the O_2 audit-loop box (owner killed it by accident 2026-10-23)", 0.90),
    (38144, "RTX 3070 + AMD EPYC 7742 (256 threads, 258 GB), verified: the O_2 box the owner killed (host 20284)", 1.20),
    (151960, "RTX A2000 + AMD EPYC 9654 (368 threads, 129 GB), British Columbia, verified: the O_2 label box, taken back once o2-vol40-1 ends and destroys it (fallback host of the audit loop)", 1.20),
    (144582, "RTX 5090 + AMD EPYC 7742 (256 threads, 258 GB), Hawaii, verified", 0.90),
    (138819, "RTX 3070 + AMD EPYC 7742 (256 threads, 258 GB), Taiwan, verified", 0.90),
    (43901, "RTX 3080 + AMD EPYC 7B12 (256 threads, 258 GB), Thailand, verified", 0.90),
    (153489, "L40S + AMD Threadripper PRO 7995WX (192 threads, 515 GB), Taiwan, unverified", 0.90),
    (57333, "RTX 3090 + AMD EPYC 7B13 (256 threads, 516 GB), Germany, unverified", 0.90),
]
BIN = "dd19e4d66cefbd60eb0f0b662e94f828d5c189a0"  # a runner build with `keep` (merged main, 2026-10-23)
IMAGE = "ghcr.io/guillaumeds/41bots-base:20261017-bc692e446939"
STATE_KEY = "grab/state.json"


def rentable_offer(machine_id):
    body = json.dumps({"machine_id": {"eq": machine_id}, "type": "on-demand", "rentable": {"eq": True}, "limit": 5})
    p = subprocess.run(["curl", "-sSL", "-m", "30", "-X", "POST", "https://console.vast.ai/api/v0/bundles/",
                        "-H", "Content-Type: application/json", "-d", body], capture_output=True, text=True)
    try:
        offers = json.loads(p.stdout).get("offers", [])
    except ValueError:
        return None
    offers = [o for o in offers if o.get("machine_id") == machine_id and o.get("rentable")]
    return min(offers, key=lambda o: o["dph_total"]) if offers else None


GENERIC_MAX_DPH = 1.50  # owner 2026-10-23: 190+ threads under $1.50/h
GENERIC_MIN_THREADS = 190
GENERIC_MIN_RAM_MB = 120_000  # the O_3 decision-loss experiment holds 46M rows (87 GB) in RAM
GENERIC_GOOD_CPUS = ("EPYC", "Threadripper")  # a Xeon Phi 7250 (272 slow threads) was taken by the plain thread rule on 2026-10-23
GENERIC_BAD_CPUS = ("9B45",)  # lists 384 threads, gave about 46 (docs/VAST-MACHINES.md)


def generic_offer(exclude_machines):
    """The cheapest-per-thread rentable offer that fits the owner's floor (190+ effective threads, a GPU of 6 GB or more, at most
    $1.20/h, reliability 0.90, 100 Mbps, 200 GB disk), not one of `exclude_machines` and not a CPU known to give far fewer threads."""
    body = json.dumps({"type": "on-demand", "rentable": {"eq": True}, "num_gpus": {"gte": 1}, "gpu_ram": {"gte": 6000},
                       "cpu_cores_effective": {"gte": GENERIC_MIN_THREADS}, "dph_total": {"lte": GENERIC_MAX_DPH}, "cpu_ram": {"gte": GENERIC_MIN_RAM_MB},
                       "reliability": {"gte": 0.90}, "inet_down": {"gte": 500}, "disk_space": {"gte": 200},
                       "order": [["dph_total", "asc"]], "limit": 60})
    p = subprocess.run(["curl", "-sSL", "-m", "30", "-X", "POST", "https://console.vast.ai/api/v0/bundles/",
                        "-H", "Content-Type: application/json", "-d", body], capture_output=True, text=True)
    try:
        offers = json.loads(p.stdout).get("offers", [])
    except ValueError:
        return None
    ok = [o for o in offers if o.get("rentable") and o.get("machine_id") not in exclude_machines
          and o.get("cpu_cores_effective", 0) >= GENERIC_MIN_THREADS and o.get("dph_total", 9) <= GENERIC_MAX_DPH
          and any(g in str(o.get("cpu_name")) for g in GENERIC_GOOD_CPUS)
          and not any(b in str(o.get("cpu_name")) for b in GENERIC_BAD_CPUS)]
    return min(ok, key=lambda o: o["dph_total"] / o["cpu_cores_effective"]) if ok else None


def load_state():
    raw = launch.r2_get(STATE_KEY)
    return json.loads(raw) if raw else {}


def save_state(st):
    launch.r2_put(STATE_KEY, json.dumps(st, indent=1).encode())


def direct_rent(o, what):
    """Rents the offer at once through vast's API (one call, about a second), the way the console's Rent button does: vast's own
    Jupyter + SSH image, 200 GB disk, labelled in the owner's style. The workflow route below needs 30-60 s to start, and offers of
    machines that come and go (the Thailand and Taiwan boxes) were gone by then (2026-10-23). Returns the new instance id or None."""
    body = {"client_id": "me", "image": "vastai/base-image:@vastai-automatic-tag", "disk": 200,
            "label": f"41bots - hold {what} - free for the next job", "runtype": "jupyter_direc ssh_direc ssh_proxy",
            "use_jupyter_lab": False, "cancel_unavail": True, "env": {}}
    p = subprocess.run(["curl", "-sSL", "-m", "30", "-X", "PUT", f"https://console.vast.ai/api/v0/asks/{o['id']}/",
                        "-H", "Content-Type: application/json", "-d", json.dumps(body)], capture_output=True, text=True)
    try:
        r = json.loads(p.stdout)
    except ValueError:
        return None
    return r.get("new_contract") if isinstance(r, dict) and r.get("success") else None


def launch_hold(st, mid, o, now, a, lines, generic):
    """Rents the offer (directly through the API; the tiny hold job is the fallback) and records it in the state."""
    run_id = f"hold-{mid}-{int(now) % 100000}"
    if not a.dry_run:
        iid = direct_rent(o, f"machine {mid}")
        if iid:
            lines.append(f"{mid}: RENTED DIRECTLY: instance {iid}, ${o['dph_total']:.3f}/h, Jupyter + SSH, 200 GB disk; billing until you destroy it")
            st[str(mid)] = {"state": "kept", "run_id": run_id, "t": now, "offer": o["id"], "dph": o["dph_total"], "generic": generic,
                            "instance": iid}
            save_state(st)
            return
        lines.append(f"{mid}: direct rent did not go through; trying the hold job")
    if not generic:
        lines.append(f"{mid}: RENTABLE: offer {o['id']} ${o['dph_total']:.3f}/h -> grabbing as {run_id}")
    if a.dry_run:
        return
    m = {"run_id": run_id, "chunk": {"difftest": {"matches": 20, "seed": 3}},
         "machine": {"keep": True, "keep_hours": a.keep_hours, "disk_gb": 200}}
    cmd = ["gh", "workflow", "run", "vast-run.yml", "-R", "Guillaumeds/41Bots", "--ref", "main",
           "-f", "manifest=" + json.dumps(m, separators=(",", ":")), "-f", "min_cores=150", "-f", "min_ram_gb=32",
           "-f", "disk_gb=200", "-f", "cpu_only=false", "-f", "gpus=1", "-f", "min_gpu_ram_gb=6", "-f", "low_cpu_pct=1",
           "-f", "max_dollars=2", "-f", "any_host=true", "-f", f"offer_id={o['id']}", "-f", f"offer_dph={o['dph_total']:.4f}",
           "-f", f"bin_sha={BIN}", "-f", f"image={IMAGE}", "-f", "dry_run=false"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines.append("   " + (r.stdout + r.stderr).strip()[-200:])
    if r.returncode == 0:
        gh_run = (r.stdout.strip().rsplit("/", 1) + [""])[-1] if "/runs/" in r.stdout else ""
        st[str(mid)] = {"state": "launched", "run_id": run_id, "t": now, "offer": o["id"], "dph": o["dph_total"], "generic": generic,
                        "gh_run": gh_run}
        save_state(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-held", type=int, default=3)
    ap.add_argument("--keep-hours", type=float, default=8)
    ap.add_argument("--generic", action="store_true", help=f"also take the cheapest-per-thread machine with {GENERIC_MIN_THREADS}+ threads under ${GENERIC_MAX_DPH}/h")
    ap.add_argument("--generic-only", action="store_true", help="skip the named machines: only the generic rule")
    ap.add_argument("--max-generic", type=int, default=1, help="how many generic machines may be held at once")
    a = ap.parse_args()
    st = load_state()
    now = time.time()
    held = [m for m, e in st.items() if e.get("state") in ("launched", "kept") and now - e["t"] < a.keep_hours * 3600 + 900]
    live = None
    if any(e.get("instance") for e in st.values()):
        r = subprocess.run(["curl", "-sSL", "-m", "30", "https://console.vast.ai/api/v1/instances/"], capture_output=True, text=True)
        try:
            live = {i["id"] for i in json.loads(r.stdout).get("instances", [])}
        except (ValueError, KeyError, AttributeError):
            live = None
    for mid, e in list(st.items()):
        # a machine rented directly that is not in vast's instance list any more (destroyed, by hand or by accident): forget it so
        # the watcher takes the machine again when it is back on the market
        if live is not None and e.get("instance") and e["instance"] not in live and now - e["t"] > 120:
            del st[mid]
            continue
        # a hold job whose workflow run already failed before any heartbeat (the offer was taken by another renter in the seconds
        # between the check and the rent, seen 2026-10-23): forget it at once so the machine can be grabbed again
        if e.get("state") == "launched" and e.get("gh_run") and now - e["t"] > 60:
            r = subprocess.run(["gh", "run", "view", str(e["gh_run"]), "-R", "Guillaumeds/41Bots", "--json", "conclusion", "-q", ".conclusion"],
                               capture_output=True, text=True)
            if r.stdout.strip() == "failure" and launch.r2_get(f"runs/{e['run_id']}/heartbeat/000000.json") is None:
                e["state"] = "failed"
        # a launch that ended without a KEPT record failed (the offer vanished, ...): forget it so it can be retried
        if e.get("state") == "launched" and now - e["t"] > 900:
            if launch.r2_get(f"runs/{e['run_id']}/KEPT.json") is not None:
                e["state"] = "kept"
            elif (launch.r2_get(f"runs/{e['run_id']}/DONE") is not None or launch.r2_get(f"runs/{e['run_id']}/FAILED") is not None
                  or launch.r2_get(f"runs/{e['run_id']}/heartbeat/000000.json") is None):
                # ended without a KEPT record, or never started (the workflow run failed before any heartbeat, e.g. the offer
                # was taken by someone else: seen 2026-10-23 on machine 68507): forget it so it can be grabbed again
                e["state"] = "failed"
                e["tries"] = e.get("tries", 0) + 1
    lines = []
    for mid, what, cap in ([] if a.generic_only else WANTED):
        e = st.get(str(mid), {})
        if e.get("state") in ("launched", "kept"):
            lines.append(f"{mid}: {e['state']} ({e['run_id']})")
            continue
        if e.get("tries", 0) >= 3:
            lines.append(f"{mid}: gave up after 3 failed grabs")
            continue
        o = rentable_offer(mid)
        if not o:
            lines.append(f"{mid}: not rentable ({what})")
            continue
        if o["dph_total"] > cap:
            lines.append(f"{mid}: rentable but ${o['dph_total']:.3f}/h is over the cap ${cap}")
            continue
        n_held = sum(1 for x in st.values() if x.get("state") in ("launched", "kept") and now - x["t"] < a.keep_hours * 3600 + 900)
        if n_held >= a.max_held:
            lines.append(f"{mid}: RENTABLE (offer {o['id']}, ${o['dph_total']:.3f}/h) but {n_held} machines are held already (max {a.max_held})")
            continue
        launch_hold(st, mid, o, now, a, lines, generic=False)
    if a.generic:
        n_held = sum(1 for x in st.values() if x.get("state") in ("launched", "kept") and now - x["t"] < a.keep_hours * 3600 + 900)
        n_gen = sum(1 for x in st.values() if x.get("generic") and x.get("state") in ("launched", "kept")
                    and now - x["t"] < a.keep_hours * 3600 + 900)
        if n_gen >= a.max_generic or n_held >= a.max_held:
            lines.append(f"generic: {n_gen} generic and {n_held} machines held already (max {a.max_generic} / {a.max_held})")
        else:
            o = generic_offer({int(k) for k in st if k.isdigit()} | ({mid for mid, _, _ in WANTED} if not a.generic_only else set()))
            if not o:
                lines.append(f"generic: no rentable machine with {GENERIC_MIN_THREADS}+ threads under ${GENERIC_MAX_DPH}/h")
            else:
                lines.append(f"generic: machine {o['machine_id']} {o.get('gpu_name')} + {o.get('cpu_name')} "
                             f"({o['cpu_cores_effective']:.0f} threads, ${o['dph_total']:.3f}/h) -> grabbing")
                if not a.dry_run:
                    launch_hold(st, o["machine_id"], o, now, a, lines, generic=True)
    if not a.dry_run:
        save_state(st)
    print(f"held now: {len(held)} (max {a.max_held})")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
