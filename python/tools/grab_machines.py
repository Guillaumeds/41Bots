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


def load_state():
    raw = launch.r2_get(STATE_KEY)
    return json.loads(raw) if raw else {}


def save_state(st):
    launch.r2_put(STATE_KEY, json.dumps(st, indent=1).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-held", type=int, default=3)
    ap.add_argument("--keep-hours", type=float, default=8)
    a = ap.parse_args()
    st = load_state()
    now = time.time()
    held = [m for m, e in st.items() if e.get("state") in ("launched", "kept") and now - e["t"] < a.keep_hours * 3600 + 900]
    for mid, e in list(st.items()):
        # a launch that ended without a KEPT record failed (the offer vanished, ...): forget it so it can be retried
        if e.get("state") == "launched" and now - e["t"] > 900:
            if launch.r2_get(f"runs/{e['run_id']}/KEPT.json") is not None:
                e["state"] = "kept"
            elif launch.r2_get(f"runs/{e['run_id']}/DONE") is not None or launch.r2_get(f"runs/{e['run_id']}/FAILED") is not None:
                e["state"] = "failed"
                e["tries"] = e.get("tries", 0) + 1
    lines = []
    for mid, what, cap in WANTED:
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
        run_id = f"hold-{mid}-{int(now) % 100000}"
        lines.append(f"{mid}: RENTABLE: offer {o['id']} ${o['dph_total']:.3f}/h -> grabbing as {run_id}")
        if a.dry_run:
            continue
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
            st[str(mid)] = {"state": "launched", "run_id": run_id, "t": now, "offer": o["id"], "dph": o["dph_total"]}
            save_state(st)
    if not a.dry_run:
        save_state(st)
    print(f"held now: {len(held)} (max {a.max_held})")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
