"""Start the next S1 label batch on the same kept machine as soon as the running batch has finished (owner 2026-10-24, asleep).

    R2_S3_ENDPOINT=$R2_ENDPOINT python3 -u python/tools/chain_next_batch.py \\
        --after o2-s1more-3 --manifest docs/vast/o2-s1more3.manifest.json --instance 57517251

Waits until `runs/<after>/merged.json` exists in R2 (the running batch finished; its runner then parks, the machine stays up). Then it takes
the instance over for the next manifest through `vast-run.yml`, exactly as the first takeover did, and checks the new run's heartbeat.

Why the takeover is done twice when needed (found 2026-10-24 on instance 57517251): `vast update instance` stores the new environment, but
the container that `recycle instance` builds right after it still got the settings of the PREVIOUS update (the stored settings said
RUN_ID=o2-s1more-4 while the container ran o2-s1more-3). So the first takeover starts a container with the old settings; a second takeover
with the same manifest (a second update, then a recycle) builds the container from the new ones. The script launches the second one
when the new run has not heartbeated after `--settle` seconds, after cancelling the first launcher (a cancelled run with machine.keep leaves
the machine up). Progress and every decision go to stdout, one line each; state is kept in R2 `chain/<run id>.json`, so a restart never
takes the machine over twice in a row.
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

BIN = "dd19e4d66cefbd60eb0f0b662e94f828d5c189a0"
IMAGE = "ghcr.io/guillaumeds/41bots-base:20261017-bc692e446939"


def say(msg):
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def instance_alive(iid):
    p = subprocess.run(["curl", "-s", "-m", "30", "https://console.vast.ai/api/v1/instances/"], capture_output=True, text=True)
    try:
        return any(i["id"] == iid for i in json.loads(p.stdout).get("instances", []))
    except ValueError:
        return True  # unreadable: never act on a guess


def heartbeat_of(run):
    r = launch.latest_heartbeat(run)
    return r[1] if r and r[1] else None


def dispatch(manifest_path, iid):
    with open(manifest_path) as fh:
        m = json.load(fh)
    m["machine"]["instance_id"] = iid
    m["machine"]["force_take"] = True
    cmd = ["gh", "workflow", "run", "vast-run.yml", "-R", "guillaumeds/41bots", "--ref", "main",
           "-f", "manifest=" + json.dumps(m), "-f", f"instance_id={iid}", "-f", "min_cores=190", "-f", "min_ram_gb=96",
           "-f", "disk_gb=100", "-f", "cpu_only=false", "-f", "gpus=1", "-f", "min_gpu_ram_gb=6", "-f", "low_cpu_pct=1",
           "-f", "max_dollars=20", "-f", "any_host=true", "-f", f"bin_sha={BIN}", "-f", f"image={IMAGE}", "-f", "dry_run=false"]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    out = (p.stdout + p.stderr).strip()
    say(f"takeover dispatched ({'ok' if p.returncode == 0 else 'FAILED'}): {out[-160:]}")
    return out.rsplit("/", 1)[-1] if p.returncode == 0 and "/runs/" in out else None


def cancel(gh_run):
    if gh_run:
        p = subprocess.run(["gh", "run", "cancel", gh_run, "-R", "guillaumeds/41bots"], capture_output=True, text=True)
        say(f"cancelled the first launcher {gh_run} (the kept machine stays up): {(p.stdout + p.stderr).strip()[-80:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", required=True, help="run id whose merged.json says it is finished")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--instance", type=int, required=True)
    ap.add_argument("--settle", type=int, default=420, help="seconds to wait for the new run's first heartbeat after a takeover")
    ap.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()
    new = json.load(open(a.manifest))["run_id"]
    key = f"chain/{new}.json"
    st = json.loads(launch.r2_get(key) or b"{}")
    say(f"waiting for runs/{a.after}/merged.json, then the next batch {new} on instance {a.instance}")
    while not launch.r2_exists(f"runs/{a.after}/merged.json"):
        if not instance_alive(a.instance):
            say(f"ALERT: instance {a.instance} is gone before {a.after} finished")
            return 2
        time.sleep(a.poll)
    say(f"{a.after} finished (merged.json found); machine alive: {instance_alive(a.instance)}")
    time.sleep(90)  # let the runner write DONE and park
    for attempt in (1, 2):
        if heartbeat_of(new):
            break
        gh_run = dispatch(a.manifest, a.instance)
        st[f"takeover{attempt}"] = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gh_run": gh_run}
        launch.r2_put(key, json.dumps(st).encode())
        t0 = time.time()
        while time.time() - t0 < a.settle and not heartbeat_of(new):
            time.sleep(20)
        if heartbeat_of(new):
            say(f"{new} is heartbeating after takeover {attempt}")
            break
        say(f"no heartbeat of {new} after takeover {attempt} ({a.settle} s): the container probably ran the previous settings")
        cancel(gh_run)
        time.sleep(30)
    hb = heartbeat_of(new)
    if hb:
        say(f"DONE: {new} running: {hb.get('utc')} {hb.get('phase')} {(hb.get('progress') or {}).get('queue')}")
        return 0
    say(f"ALERT: {new} did not start after two takeovers; machine alive: {instance_alive(a.instance)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
