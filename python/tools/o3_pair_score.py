"""Scores the arms of the O_3 decision-loss experiment (docs/vast/o3-pair.manifest.json, run o3-pair-1; docs/m3-hybrid/sibrank.md).

    R2_S3_ENDPOINT=$R2_ENDPOINT uv run --python 3.14 --project python/nn python python/tools/o3_pair_score.py \\
        [--run o3-pair-1] [--work DIR] [--streams base_s1,w1_s1,...]

For every arm found under `runs/RUN/out/ARM/` (the final bundle, else the live `partial/` files): its float network and its golden check
(`hybrid/hc-t2-o3.txt`: 2,000 trick-2 decisions x 320 worlds, the O_3 recipe's decision loss). Each network is then judged on the 100,000 held-out EXACT
sibling rows of `sib3` (seed 7903, the o3-ladder-1 set) with `fortyone_nn.netcmp`: the sibling regret (what a wrong ordering of the cards costs, percent of the
money chance), the value SD and the largest team bias. Prints one table, sibling regret first (the sensitive judge), golden second (the confirming one).
Needs Python 3.14 (compression.zstd). Reads R2 only; writes under --work.
"""

import argparse
import io
import json
import os
import re
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "launcher"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "nn"))
import launch  # noqa: E402

SIB = {  # the held-out exact trick-3 sibling set of o3-ladder-1 (EXPERIMENTS D10)
    "sib3.npy": "inputs/sha256/5cdf183ba80dee0ded3e3dd8ffb67670bf90f6ff27372becd608de98e62aa841",
    "sib3.npy.groups": "inputs/sha256/d2854861706e26c639e3ed9c7a98038a5a7ec7ff77752d935642ce17b5df6234",
    "sib3.npy.json": "inputs/sha256/c4fb5318d3019c037cc9f998b9fd661243598a6fe5da9595f60e7f519fc9ead3",
}
F5766A = "o3 f5766a (adopted)"
ALL_RE = re.compile(r"^all\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+(yes|no)\s+([\d.]+)%\s+\d+\s+([+-][\d.]+)%\s+([+-][\d.]+)%\s+([+-][\d.]+)%\s+([+-][\d.]+)%\s+([\d.]+)%")


def fetch_sib(work):
    for name, key in SIB.items():
        path = os.path.join(work, name)
        if not os.path.exists(path):
            raw = launch.r2_get(key)
            if raw is None:
                sys.exit(f"missing in R2: {key}")
            with open(path, "wb") as f:
                f.write(raw)
    return os.path.join(work, "sib3.npy")


def arm_files(run, arm):
    """(float network bytes, golden text) of one arm, from its final bundle or its live files."""
    from compression import zstd

    tars = launch.r2_list(f"runs/{run}/out/{arm}/")
    if tars:
        t = tarfile.open(fileobj=io.BytesIO(zstd.decompress(launch.r2_get(tars[0]))))
        net = [n for n in t.getnames() if n.endswith("net/float.safetensors")]
        gold = [n for n in t.getnames() if n.endswith("hc-t2-o3.txt")]
        return (t.extractfile(net[0]).read() if net else None, t.extractfile(gold[0]).read().decode() if gold else None)
    base = f"runs/{run}/partial/{arm}/000000/out/hybrid/"
    net = launch.r2_get(base + "net/float.safetensors")
    gold = launch.r2_get(base + "hc-t2-o3.txt")
    return net, (gold.decode() if gold else None)


def golden_row(text):
    for line in (text or "").splitlines():
        m = ALL_RE.match(line.strip())
        if m:
            return {"decisions": int(m[1]), "mean_loss_pct": float(m[2]), "upper95_pct": float(m[3]), "own_bias_pct": float(m[6]), "sd_pct": float(m[10])}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="o3-pair-1")
    ap.add_argument("--work", default="/tmp/o3-pair-score")
    ap.add_argument("--streams", default="base_s1,w1_s1,base_s2,w1_s2,w3_s1")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    sib = fetch_sib(a.work)
    nets, gold = {}, {}
    for arm in a.streams.split(","):
        net, g = arm_files(a.run, arm)
        if net is None:
            print(f"{arm}: no network stored yet")
            continue
        path = os.path.join(a.work, f"{arm}.float.safetensors")
        with open(path, "wb") as f:
            f.write(net)
        nets[arm] = path
        gold[arm] = golden_row(g)
    if not nets:
        sys.exit("no arm has a stored network")
    from fortyone_nn.netcmp import compare

    rep = compare(nets, [sib], ref=None, siblings=sib)
    rows = []
    for arm, r in rep["nets"].items():
        g = gold.get(arm) or {}
        rows.append((arm, r["siblings"]["mean_regret_pct"], r["siblings"]["mean_regret_pm95_pct"], r["sd_pct"], r["largest_abs_bias_pct"],
                     g.get("mean_loss_pct"), g.get("upper95_pct")))
    print(f"| arm | sibling regret % (95% +-) | sibling SD % | largest bias % | golden loss % (95% upper) |")
    print("|---|---|---|---|---|")
    for arm, reg, pm, sd, bias, gl, gu in rows:
        print(f"| {arm} | {reg:.4f} +-{pm:.4f} | {sd:.3f} | {bias:.4f} | {gl if gl is not None else 'n/a'} ({gu if gu is not None else 'n/a'}) |")
    print(f"| {F5766A} (reference, D10 ladder) | 100k exact siblings: see docs/m3-hybrid/o3-ladder-calibration.json | | | 0.0056 (0.0068) |")
    with open(os.path.join(a.work, "report.json"), "w") as f:
        json.dump({"netcmp": rep, "golden": gold}, f, indent=1)


if __name__ == "__main__":
    main()
