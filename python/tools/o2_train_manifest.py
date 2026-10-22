"""Manifest of the training-only run of the O_2 G2 arm (docs/EXPERIMENTS.md D6).

    R2_S3_ENDPOINT=$R2_ENDPOINT python3 python/tools/o2_train_manifest.py [--out FILE] [--run-id ID]

Reads the stored shards of the label runs `o2-g2arm-1` (part 1 and the s601 validation shard) and `o2-g2arm-2` (parts 1-4)
from their `shards.json` (the `label_inputs` lists the runner wrote), and writes a one-stream `hybrid_rung` manifest that
trains the 2048-1024-256 network on every stored G2 part (`label_inputs`, nothing generated), early stopping on the exact
s601 shard, like the other Stage A arms. Refuses unless 5 random + 5 rules-v1 hybrid shards (750,000 rows) are found.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "launcher"))
import launch  # noqa: E402

O3 = "f5766aa98a1d769e31122c94cc4da9790c84508fab5c432d0c0358fe3124faad"
G2 = "20c41678c64681855003e5b8660e0a2e1fb9e8d552f7d898dd956055c767a9d5"
BIN = "ab0b4642c49e79bec35284dce44f439f8f87c06e"
SOURCES = [("o2-g2arm-1", "g75"), ("o2-g2arm-2", "g2parts")]


def stored_entries():
    out = []
    for run, stream in SOURCES:
        raw = launch.r2_get(f"runs/{run}/partial/{stream}/000000/out/hybrid/shards.json")
        if raw is None:
            sys.exit(f"no shards.json for {run}")
        s = json.loads(raw)
        if s.get("label_inputs_pending"):
            sys.exit(f"{run}: shards still waiting to be stored: {s['label_inputs_pending']}")
        out += [(run, e) for e in s["label_inputs"]]
    return out


def build(run_id):
    ent = stored_entries()
    train = [e for _, e in ent if e["set"] in ("random", "rules")]
    val = [e for run, e in ent if e["set"] == "val" and run == "o2-g2arm-1"]
    rnd = sum(e["rows"] for e in train if e["set"] == "random")
    rul = sum(e["rows"] for e in train if e["set"] == "rules")
    if len(val) != 1 or val[0]["seed"] != 601:
        sys.exit("the s601 validation shard of o2-g2arm-1 is missing")
    if (rnd, rul) != (600_000, 150_000):
        sys.exit(f"stored G2 labels: {rnd} random + {rul} rules-v1 rows; need 600,000 + 150,000 (the label runs are not done)")
    train.sort(key=lambda e: (e["set"], e["seed"]))
    params = dict(threads="all", cut=2, check=False, widths="2048,1024,256", head="outcome", epochs=40, patience=6, batch=1024,
                  lr=0.001, train_seed=1, max_minutes=0, device="cuda", overlap=False, upload_shards=False,
                  table={"hands": 1000000, "seed": 1}, label_inputs=train + val)
    return {"format": 2, "run_id": run_id, "run_seed": 1, "threads": None, "benchmark_seconds": 0, "upload_every_s": 300,
            "heartbeat_every_s": 30,
            "note": "O_2 stage A arm 'g75' training only: the 2048-1024-256 network (seed 1, outcome head, early stopping on the exact s601 "
                    "shard) on all 750,000 stored G2 hybrid labels (600k random + 150k rules-v1; parts of o2-g2arm-1 and o2-g2arm-2), "
                    "like arms e075/e15/p15/p60. Judged on the 250k exact test set with fortyone_nn.netcmp.",
            "machine": {"gpus": 1, "min_gpu_ram_gb": 8, "min_cores": 120, "min_ram_gb": 32, "min_inet_down": 100, "max_dph": 1.7, "disk_gb": 80},
            "inputs": {},
            "streams": [{"name": "g75full", "priority": 1, "kind": "hybrid_rung", "chunks": 1, "params": params}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(HERE)), "docs", "vast", "o2-g2arm-train.manifest.json"))
    ap.add_argument("--run-id", default="o2-g2full-1")
    a = ap.parse_args()
    m = build(a.run_id)
    with open(a.out, "w") as f:
        json.dump(m, f, indent=1)
    p = m["streams"][0]["params"]["label_inputs"]
    print(f"wrote {a.out}: {len(p)} label_inputs ({sum(e['rows'] for e in p if e['set'] != 'val'):,} training rows)")


if __name__ == "__main__":
    main()
