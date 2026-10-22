"""Progress of a vast.ai run from its R2 heartbeats: step, rows, speed, ETA, cost.

    R2_S3_ENDPOINT=$R2_ENDPOINT python3 python/tools/o2watch.py RUN_ID [--back N]

Reads the newest heartbeat and one 10 minutes older (heartbeats come every 30 s, key
``runs/RUN/heartbeat/NNNNNN.json``); prints cost so far, the streams finished and waiting, and for
the running step the rows done, rows per second and the time left in that step.
Needs the aws CLI and the R2_* variables (python/launcher/launch.py helpers).
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "launcher"))
import launch  # noqa: E402


def hb_at(keys, seq):
    k = f"runs/{RUN}/heartbeat/{seq:06d}.json"
    return json.loads(launch.r2_get(k) or b"null") if k in keys else None


def hms(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


def main():
    global RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--back", type=int, default=20, help="heartbeats back for the speed (20 = 10 min)")
    a = ap.parse_args()
    RUN = a.run
    keys = sorted(k for k in launch.r2_list(f"runs/{RUN}/heartbeat/") if k.endswith(".json"))
    if not keys:
        sys.exit("no heartbeat yet")
    now = json.loads(launch.r2_get(keys[-1]))
    old = hb_at(set(keys), max(0, now["seq"] - a.back))
    p = now["progress"]
    print(f"{RUN}: seq {now['seq']} {now['utc']}  elapsed {hms(now['elapsed_s'])}  cost ${now['est_cost_usd']:.2f}"
          f"  threads {now['threads']} usable {now['usable_cores']}  cpu {now['cpu_util_pct']}%")
    done = [q["stream"] for q in p.get("queue", []) if q["left"] == 0 and q["running"] == 0]
    wait = [q["stream"] for q in p.get("queue", []) if q["left"] > 0 and q["running"] == 0]
    print(f"finished {len(done)}: {' '.join(done) or '-'}   waiting {len(wait)}: {' '.join(wait) or '-'}")
    for name, c in p.get("chunk_steps", {}).items():
        line = f"running {name}: {c.get('step')}"
        tot, got = c.get("rows_total"), c.get("rows_done")
        if tot and got is not None:
            line += f"  {got:,}/{tot:,} ({100 * got / tot:.0f}%)"
            if old:
                oc = old["progress"].get("chunk_steps", {}).get(name)
                if oc and oc.get("step") == c.get("step") and oc.get("rows_done") is not None:
                    dt = now["elapsed_s"] - old["elapsed_s"]
                    rate = (got - oc["rows_done"]) / dt if dt > 0 else 0
                    if rate > 0:
                        line += f"  {rate:.1f} rows/s  step ETA {hms((tot - got) / rate)}"
        print(line)
    for w in p.get("warnings", []):
        print("warning:", w)


if __name__ == "__main__":
    main()
