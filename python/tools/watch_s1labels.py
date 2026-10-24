"""Watch the vast.ai market for a cheap many-thread machine and start the S1 label run on it (owner 2026-10-24).

    R2_S3_ENDPOINT=$R2_ENDPOINT python3 python/tools/watch_s1labels.py [--dry-run] [--once] [--every 60]
        [--manifest docs/vast/o2-s1more.manifest.json] [--max-dph 1.0] [--other-max-dph 0.40]

Every `--every` seconds: the rentable on-demand offers with at least 190 effective threads, at most `--max-dph` dollars an hour, at least
96 GB RAM, one GPU of 6 GB or more, reliability 0.90, 100 Mbps, an AMD EPYC or Threadripper (the Xeon Phi 7250 and EPYC 9B45 trap
of docs/VAST-MACHINES.md are excluded). Preferred: a machine in Thailand or Taiwan (the cheap EPYC 7B12 boxes, $0.33/h, 12 hours for
250,000 labels); any other country is taken only at `--other-max-dph` or less. The cheapest per thread wins. The run is started on
that offer by id through the `vast-run.yml` workflow (a normal rent, no takeover), with the manifest's `keep` settings. The decision is
recorded in R2 `grab/s1labels.json`, so a restart never starts the run twice. Needs the `gh` CLI and R2/vast access as the other tools.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "launcher"))
import launch  # noqa: E402

STATE_KEY = "grab/s1labels.json"
BIN = "dd19e4d66cefbd60eb0f0b662e94f828d5c189a0"
IMAGE = "ghcr.io/guillaumeds/41bots-base:20261017-bc692e446939"
MIN_THREADS = 190
MIN_RAM_MB = 96_000
GOOD_CPUS = ("EPYC", "Threadripper")
BAD_CPUS = ("9B45",)
PREFERRED_COUNTRIES = ("TH", "TW")  # the last two letters of vast's geolocation, "Thailand, TH"


def offers(max_dph):
    body = json.dumps({"type": "on-demand", "rentable": {"eq": True}, "num_gpus": {"gte": 1}, "gpu_ram": {"gte": 6000},
                       "cpu_cores_effective": {"gte": MIN_THREADS}, "dph_total": {"lte": max_dph}, "cpu_ram": {"gte": MIN_RAM_MB},
                       "reliability": {"gte": 0.90}, "inet_down": {"gte": 100}, "disk_space": {"gte": 150},
                       "order": [["dph_total", "asc"]], "limit": 100})
    p = subprocess.run(["curl", "-sSL", "-m", "30", "-X", "POST", "https://console.vast.ai/api/v0/bundles/",
                        "-H", "Content-Type: application/json", "-d", body], capture_output=True, text=True)
    try:
        return json.loads(p.stdout).get("offers", [])
    except ValueError:
        return []


def country(o):
    return str(o.get("geolocation") or "")[-2:]


def choose(rows, max_dph, other_max_dph):
    """The offer to take, or None: preferred countries up to max_dph, others up to other_max_dph, cheapest per thread first."""
    ok = []
    for o in rows:
        cpu = str(o.get("cpu_name"))
        if not o.get("rentable") or o.get("cpu_cores_effective", 0) < MIN_THREADS:
            continue
        if not any(g in cpu for g in GOOD_CPUS) or any(b in cpu for b in BAD_CPUS):
            continue
        cap = max_dph if country(o) in PREFERRED_COUNTRIES else other_max_dph
        if o.get("dph_total", 9) > cap:
            continue
        ok.append(o)
    ok.sort(key=lambda o: (country(o) not in PREFERRED_COUNTRIES, o["dph_total"] / o["cpu_cores_effective"]))
    return ok[0] if ok else None


def launch_run(manifest_path, o):
    with open(manifest_path) as fh:
        m = json.load(fh)
    m["machine"].pop("instance_id", None)
    m["machine"].pop("force_take", None)
    cmd = ["gh", "workflow", "run", "vast-run.yml", "-R", "guillaumeds/41bots", "--ref", "main",
           "-f", "manifest=" + json.dumps(m), "-f", f"offer_id={o['id']}", "-f", f"offer_dph={o['dph_total'] * 1.1:.3f}",
           "-f", "cpu_only=false", "-f", "gpus=1", "-f", "min_gpu_ram_gb=6", "-f", f"min_cores={MIN_THREADS}", "-f", "min_ram_gb=96",
           "-f", "disk_gb=100", "-f", "low_cpu_pct=1", "-f", "max_dollars=20", "-f", "any_host=true",
           "-f", f"bin_sha={BIN}", "-f", f"image={IMAGE}", "-f", "dry_run=false"]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    return p.returncode == 0, (p.stdout + p.stderr).strip()[-300:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--manifest", default=os.path.join(REPO, "docs/vast/o2-s1more.manifest.json"))
    ap.add_argument("--max-dph", type=float, default=1.0)
    ap.add_argument("--other-max-dph", type=float, default=0.40)
    a = ap.parse_args()
    while True:
        raw = launch.r2_get(STATE_KEY)
        if raw and json.loads(raw).get("started"):
            print("already started:", raw.decode()[:300], flush=True)
            return
        o = choose(offers(a.max_dph), a.max_dph, a.other_max_dph)
        stamp = time.strftime("%H:%M:%SZ", time.gmtime())
        if o is None:
            print(f"[{stamp}] no machine yet (190+ threads, preferred TH/TW up to ${a.max_dph}/h, others up to ${a.other_max_dph}/h)", flush=True)
        else:
            what = (f"offer {o['id']} machine {o.get('machine_id')} {o.get('geolocation')} {o.get('cpu_name')} {o.get('cpu_cores_effective'):.0f} threads, "
                    f"{o.get('gpu_name')}, ${o['dph_total']:.3f}/h")
            print(f"[{stamp}] FOUND {what}", flush=True)
            if a.dry_run:
                return
            ok, out = launch_run(a.manifest, o)
            print(f"[{stamp}] launch {'started' if ok else 'FAILED'}: {out}", flush=True)
            if ok:
                launch.r2_put(STATE_KEY, json.dumps({"started": True, "utc": stamp, "offer": o["id"], "what": what}).encode())
                return
        if a.once:
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
