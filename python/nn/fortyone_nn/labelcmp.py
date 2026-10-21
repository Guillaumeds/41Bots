"""Label quality (docs/m3-hybrid/o3-rounds.md): how far hybrid labels are from exact ones on the
same positions. O_2's training labels are hybrid (exact search through trick 2, then O_3 at the
start of trick 3); this measures, per team, their error against the exact labels of the stored
exact shard made from the same seed (so the same positions in the same order).

    uv run python -m fortyone_nn.labelcmp --exact EXACT.npy --hybrid HYBRID.npy --out REPORT.json

numpy only. Every sum is a whole number (the values are fixed-point money chances, units of
2^-32), so the report is the same on every machine. Values are in the shards' rotated team order:
team 0 is the leader's team, then the next, the opposite and the previous.

What matters most for training: the **bias** per team (a network learns one-sided label errors
faithfully; random errors average out over many labels), then the error spread (SD).
"""

import argparse
import json

import numpy as np

from .formats import ONE, load_shard, race_classes, RACE_CLASS_NAMES

TEAMS = ("leader's team", "next team", "opposite team", "previous team")


def errors(exact, hybrid):
    """The per-team errors (hybrid - exact, units of 2^-32, int64), after checking the two shards
    hold the same positions in the same order."""
    if len(exact) != len(hybrid):
        raise SystemExit(f"the shards hold {len(exact)} and {len(hybrid)} rows: not the same positions")
    if not (np.array_equal(exact["active"], hybrid["active"]) and np.array_equal(exact["dense"], hybrid["dense"])):
        raise SystemExit("the shards' positions differ: the hybrid shard must be made from the exact shard's seed and source")
    return hybrid["value"].astype(np.int64) - exact["value"].astype(np.int64)


def summary(e):
    """Bias and SD per team, the overall SD (root mean square over every team and row) and the
    mean absolute error, in percent of money chance; the largest absolute error."""
    n = e.shape[0]
    if n == 0:
        return {"rows": 0}
    sums = [int(x) for x in e.sum(axis=0, dtype=np.int64)]
    sq = [int(x) for x in (e.astype(object) ** 2).sum(axis=0)]
    pct = 100.0 / ONE
    bias = [s / n * pct for s in sums]
    sd = [((q / n) - (s / n) ** 2) ** 0.5 * pct for s, q in zip(sums, sq)]
    rms = (sum(sq) / (4 * n)) ** 0.5 * pct
    absum = int(np.abs(e).sum(dtype=np.int64))
    return {
        "rows": n,
        "bias_pct": dict(zip(TEAMS, bias)),
        "sd_pct": dict(zip(TEAMS, sd)),
        "largest_abs_bias_pct": max(abs(b) for b in bias),
        "rms_pct": rms,
        "mean_abs_pct": absum / (4 * n) * pct,
        "max_abs_pct": int(np.abs(e).max()) * pct,
        "rows_exact": int((np.abs(e).sum(axis=1) == 0).sum()),
    }


def compare(exact_path, hybrid_path):
    exact, hybrid = load_shard(exact_path), load_shard(hybrid_path)
    e = errors(exact, hybrid)
    rep = {"what": "hybrid labels minus exact labels on the same positions (per team, the leader's team first)",
           "exact": str(exact_path), "hybrid": str(hybrid_path), "all": summary(e), "by_race_class": {}}
    cls = race_classes(exact)
    for k, name in enumerate(RACE_CLASS_NAMES):
        sel = cls == k
        if sel.any():
            rep["by_race_class"][str(k)] = {"name": name, **summary(e[sel])}
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fortyone_nn.labelcmp", description=__doc__.split("\n\n")[0])
    ap.add_argument("--exact", required=True)
    ap.add_argument("--hybrid", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = compare(a.exact, a.hybrid)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    s = rep["all"]
    print(f"labelcmp: {s['rows']} rows; rms {s['rms_pct']:.4f}%, mean |error| {s['mean_abs_pct']:.4f}%, "
          f"largest |bias| {s['largest_abs_bias_pct']:.4f}%; exact rows {s['rows_exact']}")


if __name__ == "__main__":
    main()
