"""Grab a vast.ai machine the moment it comes back on the market, and run the O_2 22M-label training on it.

    R2_S3_ENDPOINT=$R2_ENDPOINT python3 python/tools/o2_grab.py --once [--dry-run]

Meant to be called every few minutes (a scheduled wake-up of the session). One pass:
 1. is the label run (`--label-run`, default o2-vol-3) DONE in R2? (its machine is destroyed by it)
 2. is machine `--machine-id` (default 38144, the EPYC 7742 host 20284) rentable again? (vast's market, searched
    through the console API; credentials come from the environment's proxy)
 3. if both: build the training manifest of all ten streams (`o2_train_manifest.py --vol-run`), and launch it on that
    very offer through the `vast-run.yml` workflow (`offer_id`), once (a marker file stops a second launch).
Prints one status line; exit 0 always. Needs `gh` and the R2 variables.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "launcher"))
import launch  # noqa: E402

BIN = "ab0b4642c49e79bec35284dce44f439f8f87c06e"
IMAGE = "ghcr.io/guillaumeds/41bots-base:20261017-bc692e446939"


def offer_of(machine_id):
    body = json.dumps({"machine_id": {"eq": machine_id}, "type": "on-demand", "rentable": {"eq": True}, "limit": 5})
    p = subprocess.run(["curl", "-sSL", "-m", "30", "-X", "POST", "https://console.vast.ai/api/v0/bundles/",
                        "-H", "Content-Type: application/json", "-d", body], capture_output=True, text=True)
    try:
        offers = json.loads(p.stdout).get("offers", [])
    except ValueError:
        return None
    offers = [o for o in offers if o.get("machine_id") == machine_id and o.get("rentable")]
    return offers[0] if offers else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--machine-id", type=int, default=38144)
    ap.add_argument("--label-run", default="o2-vol-3")
    ap.add_argument("--run-id", default="o2-volnet-10")
    ap.add_argument("--marker", default="/tmp/o2_grab.launched")
    a = ap.parse_args()
    if os.path.exists(a.marker):
        print(f"already launched: {open(a.marker).read().strip()}")
        return
    done = launch.r2_get(f"runs/{a.label_run}/DONE") is not None
    failed = launch.r2_get(f"runs/{a.label_run}/FAILED") is not None
    offer = offer_of(a.machine_id)
    if offer:
        print(f"machine {a.machine_id} is on the market: offer {offer['id']}, ${offer['dph_total']:.4f}/h")
    else:
        print(f"machine {a.machine_id} is not rentable yet")
    print(f"label run {a.label_run}: done={done} failed={failed}")
    if not (done and offer):
        return
    mf = os.path.join(REPO, "docs", "vast", f"{a.run_id}.manifest.json")
    if a.dry_run:
        print("dry run: would build the manifest and launch on the offer")
        return
    subprocess.run([sys.executable, os.path.join(HERE, "o2_train_manifest.py"), "--vol-run", a.label_run, "--vol-streams", "10",
                    "--run-id", a.run_id, "--out", mf], check=True)
    m = json.load(open(mf))
    m["machine"].update(min_gpu_ram_gb=6, min_cores=120, disk_gb=200, max_dph=1.2)
    json.dump(m, open(mf, "w"), indent=1)
    cmd = ["gh", "workflow", "run", "vast-run.yml", "-R", "Guillaumeds/41Bots", "--ref", "claude/brave-newton-3pkcpf",
           "-f", "manifest=" + json.dumps(m, separators=(",", ":")), "-f", "min_cores=120", "-f", "min_ram_gb=64",
           "-f", "disk_gb=200", "-f", "cpu_only=false", "-f", "gpus=1", "-f", "min_gpu_ram_gb=6", "-f", "low_cpu_pct=1",
           "-f", "max_dollars=8", "-f", "any_host=true", "-f", f"offer_id={offer['id']}",
           "-f", f"offer_dph={offer['dph_total']:.4f}", "-f", f"bin_sha={BIN}", "-f", f"image={IMAGE}", "-f", "dry_run=false"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-300:])
    if r.returncode == 0:
        open(a.marker, "w").write(f"{a.run_id} on offer {offer['id']}")


if __name__ == "__main__":
    main()
