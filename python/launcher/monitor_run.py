"""Progress of a running student_teacher run from its R2 heartbeat and finished parts.

    R2_S3_ENDPOINT=$R2_ENDPOINT python monitor_run.py RUN_ID [--total-decisions N] [--part-size N]

Prints the latest heartbeat's step, usable threads, elapsed time and dollars, the parts stored so far
(`runs/RUN/out/<stream>/*.tar.zst`), the measured speed (decisions an hour, from parts finished) and
the ETA and projected cost at that speed. Read-only.
"""

import argparse
import json
import re
import time

import launch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--total-decisions", type=int, default=250_000)
    ap.add_argument("--part-size", type=int, default=1000)
    a = ap.parse_args()
    _, hb = launch.latest_heartbeat(a.run)
    if not hb:
        print("no heartbeat yet")
        return
    prog = hb.get("progress", {})
    el, cost = hb.get("elapsed_s", 0.0), hb.get("est_cost_usd", 0.0)
    keys = [k for k in launch.r2_list(f"runs/{a.run}/out/") if k.endswith(".tar.zst")]
    parts = len(set(re.sub(r"-[0-9a-f]+\.tar\.zst$", "", k) for k in keys))
    done = parts * a.part_size
    cur = prog.get("chunk_steps", {})
    print(f"{hb.get('utc')}  phase {hb.get('phase')}  threads {hb.get('threads')} (usable {hb.get('usable_cores')})"
          f"  cpu {hb.get('cpu_util_pct')}%  upload failures {hb.get('upload_failures')}")
    print(f"elapsed {el / 3600:.2f} h  cost ${cost:.2f}  steps {json.dumps(cur)}")
    print(f"parts stored {parts}/{a.total_decisions // a.part_size} = {done} decisions")
    q = prog.get("queue", [])
    if q:
        print("queue:", ", ".join(f"{x['stream']} done {x['done']} left {x['left']} median {x.get('median_s')}s" for x in q))
    if done > 0 and el > 0:
        # Time before the first part includes boot and setup: use the median part time as well.
        rate = done / el * 3600
        left = a.total_decisions - done
        eta = left / rate
        print(f"speed {rate:.0f} decisions/h (whole run)  ETA {eta:.1f} h  projected total ${cost + eta * cost / el * 3600:.2f}")
    print(f"now {time.strftime('%H:%M:%S', time.gmtime())} UTC")


if __name__ == "__main__":
    main()
