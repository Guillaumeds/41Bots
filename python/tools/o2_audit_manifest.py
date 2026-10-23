"""Manifest of the O_2 audit loop on stored plain labels (docs/m3-hybrid/path-forward.md section 0, EXPERIMENTS D8-D10).

    R2_S3_ENDPOINT=$R2_ENDPOINT python3 python/tools/o2_audit_manifest.py --run-id o2-audit-4 --out docs/vast/o2-audit-4.manifest.json \\
        --source o2-vol-3:1-10 --source o2-vol40-1:11-18 [--max-rows N] [--smoke]

Each `--source RUN:FIRST-LAST` takes the stored streams volNN (NN = FIRST..LAST) of the label run RUN: their random, rules-v1 and harvest
shards as `label_inputs`, plus the exact s601 validation shard. Every key is checked to exist in R2, and the disk the shards need
(bytes of every shard, from the runner's own shards.json) is printed against `--disk-gb`. The loop settings are those of
`docs/vast/o2-audit.manifest.json` (plain O_3 labels for new harvests, 10% exact anchor, exact audit shards, capacity ladder), with
`--target-rows` (rows of each round's targeted harvest) and `--rounds`. `--smoke` shrinks everything (2 rounds, 20,000 audit rows,
50,000 targeted rows, gamma 0.01) to test the machinery end to end on a rented machine, never to judge a network.
`--force-take` adds `force_take` (the owner confirmed a hand-labelled machine may be taken over; launcher guard).
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "launcher"))
import launch  # noqa: E402

BASE = os.path.join(REPO, "docs", "vast", "o2-audit.manifest.json")
ROW_BYTES = 1883  # 3,012,800,256 bytes for 1.6M rows (measured on the stored random shards)


def shards_json(run, k):
    """The stream's shards.json: from its final output tar when it has one (the partial snapshot of a finished stream can be
    older, as for vol11 of o2-vol40-1), else from the partial upload."""
    import io
    import tarfile

    tars = launch.r2_list(f"runs/{run}/out/vol{k:02d}/")
    if tars:
        from compression import zstd  # Python 3.14 (run with `uv run --python 3.14`)

        t = tarfile.open(fileobj=io.BytesIO(zstd.decompress(launch.r2_get(tars[0]))))
        name = [n for n in t.getnames() if n.endswith("hybrid/shards.json")]
        if name:
            return json.loads(t.extractfile(name[0]).read())
    raw = launch.r2_get(f"runs/{run}/partial/vol{k:02d}/000000/out/hybrid/shards.json")
    if raw is None:
        sys.exit(f"stream vol{k:02d} of {run} is not stored yet")
    return json.loads(raw)


def stream_entries(run, k):
    s = shards_json(run, k)
    if s.get("label_inputs_pending"):
        sys.exit(f"vol{k:02d} of {run}: shards still waiting to be stored")
    got = {e["set"] for e in s["label_inputs"]}
    if not {"random", "rules", "harvest"} <= got:
        sys.exit(f"vol{k:02d} of {run}: incomplete (has {sorted(got)}): the stream is still being made")
    by_key = {x["key"]: x for x in s.get("shards", [])}
    out = []
    for e in s["label_inputs"]:
        if e["set"] in ("random", "rules", "harvest"):
            out.append((e, by_key.get(e["shard"], {}).get("bytes", e["rows"] * ROW_BYTES)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", required=True, help="RUN:FIRST-LAST")
    ap.add_argument("--max-rows", type=int, default=0, help="stop adding streams past this many training rows (disk)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--target-rows", type=int, default=3_000_000)
    ap.add_argument("--audit-rows", type=int, default=200_000)
    ap.add_argument("--disk-gb", type=int, default=80)
    ap.add_argument("--min-gpu-ram-gb", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force-take", action="store_true")
    a = ap.parse_args()

    train, nbytes = [], 0
    for src in a.source:
        run, rng = src.split(":")
        lo, hi = (int(x) for x in rng.split("-"))
        for k in range(lo, hi + 1):
            ent = stream_entries(run, k)
            if a.max_rows and sum(e["rows"] for e in train) + sum(e["rows"] for e, _ in ent) > a.max_rows:
                print(f"stopping before {run} vol{k:02d}: --max-rows {a.max_rows:,}")
                break
            train += [e for e, _ in ent]
            nbytes += sum(b for _, b in ent)
    val = json.loads(launch.r2_get("runs/o2-g2arm-1/partial/g75/000000/out/hybrid/shards.json"))["label_inputs"]
    val = [e for e in val if e["set"] == "val" and e["seed"] == 601]
    if len(val) != 1:
        sys.exit("the s601 validation shard of o2-g2arm-1 is missing")
    for e in train + val:
        for f in ("shard", "sidecar"):
            if not launch.r2_exists(e[f]):
                sys.exit(f"missing in R2: {e[f]}")
    rows = sum(e["rows"] for e in train)

    m = json.load(open(BASE))
    m["run_id"] = a.run_id
    p = m["streams"][0]["params"]
    p["label_inputs"] = train + val
    p["audit_rounds"] = 2 if a.smoke else a.rounds
    p["audit_rows"] = 20_000 if a.smoke else a.audit_rows
    p["audit_target_rows"] = 50_000 if a.smoke else a.target_rows
    if a.smoke:
        p["audit_gamma"] = 0.01
        p["epochs"] = 3
        p["patience"] = 2
    m["machine"].update(disk_gb=a.disk_gb, min_gpu_ram_gb=a.min_gpu_ram_gb)
    if a.force_take:
        m["machine"]["force_take"] = True
    target_bytes = (50_000 if a.smoke else a.target_rows) * ROW_BYTES * (2 if a.smoke else a.rounds)
    need_gb = (nbytes + target_bytes) / 1e9 + 8  # + image, networks, logs
    m["note"] = (f"{'SMOKE TEST of the machinery: ' if a.smoke else ''}O_2 audit loop at trick 2 on {rows:,} stored plain hybrid labels "
                 f"({a.source}), exact s601 validation, new harvests plain O_3 with a 10% exact anchor, exact audit shards, "
                 f"{p['audit_rounds']} rounds, targeted harvest {p['audit_target_rows']:,} rows a round at weight 2. "
                 f"Shards on disk about {nbytes / 1e9:.1f} GB + targeted shards {target_bytes / 1e9:.1f} GB + 8 GB image and networks = "
                 f"{need_gb:.0f} GB against a disk of {a.disk_gb} GB. The run is taken over onto a kept or hand-rented machine with vast-run.yml instance_id.")
    json.dump(m, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(train)} training shards, {rows:,} rows, shards {nbytes / 1e9:.1f} GB; disk need about {need_gb:.0f} GB "
          f"of {a.disk_gb} GB{'  *** TOO SMALL ***' if need_gb > a.disk_gb else ''}")


if __name__ == "__main__":
    main()
