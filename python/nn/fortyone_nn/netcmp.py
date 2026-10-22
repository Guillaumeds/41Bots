"""Network comparison on exact shards (docs/m3-hybrid/o2-labels.md): how far each candidate
network is from the exact values of the same positions, per team, and how the candidates differ
**on the same rows** (paired).

    uv run python -m fortyone_nn.netcmp --shard T1.npy --shard T2.npy \\
        --net a=A/float.safetensors --net b=B/float.safetensors --ref a \\
        [--siblings S.npy] --out REPORT.json

PyTorch on the CPU, float weights (the integer twin differs from them by under 1e-6 of a money
chance). Per network: rows, bias per team (leader's team first) with its standard error, SD,
RMS and mean absolute error, all in percent of the money chance. Against ``--ref``: the mean
change in squared error and in each team's bias, with a 95% bound (normal, rows independent).
``--siblings``: the exact sibling shard and its groups sidecar; for each group the network's
favourite for the mover is picked and its regret against the exact best is averaged (percent of
the money chance), the decision-level number: a bias common to all siblings cancels in it.
"""

import argparse
import json

import numpy as np
import torch

from .formats import ONE, load_groups, load_shard
from .train import Net, columns, evaluate, read_init, set_layers

TEAMS = ("leader's team", "next team", "opposite team", "previous team")
Z95 = 1.96


def load_net(path, widths=None, head="outcome"):
    """(model, layers): the float network at ``path`` (a trainer's ``float.safetensors``)."""
    from .formats import read_safetensors

    t, _ = read_safetensors(path)
    n = sum(1 for k in t if k.startswith("w"))
    widths = widths or [int(t[f"w{k}"].shape[1]) for k in range(n - 1)]
    layers, _, n_out = read_init(path, widths, head)
    model = Net(widths, n_out)
    set_layers(model, layers)
    return model


def predict(model, shard, batch=8192, head="outcome"):
    """The network's money values of every row of ``shard`` (float64 [n, 4], units of 2^-32)."""
    out = []
    for i in range(0, len(shard), batch):
        part = shard[i : i + batch]
        data = [torch.from_numpy(x) for x in columns(part)]
        _, v, _ = evaluate(model, head, data, 1.0, batch=batch)
        out.append(v)
    return np.concatenate(out) if out else np.zeros((0, 4))


def errors(pred, shard):
    """Predicted minus exact, percent of the money chance, float64 [n, 4]."""
    return (pred - np.asarray(shard["value"]).astype(np.float64)) * (100.0 / ONE)


def summary(e):
    n = len(e)
    bias = e.mean(axis=0)
    return {
        "rows": n,
        "bias_pct": dict(zip(TEAMS, map(float, bias))),
        "bias_se_pct": dict(zip(TEAMS, map(float, e.std(axis=0, ddof=1) / np.sqrt(n)))),
        "largest_abs_bias_pct": float(np.abs(bias).max()),
        "sd_pct": float(e.std()),
        "rms_pct": float(np.sqrt((e**2).mean())),
        "mean_abs_pct": float(np.abs(e).mean()),
    }


def paired(e_a, e_b):
    """Change from reference ``e_b`` to ``e_a`` on the same rows: mean squared error (all teams)
    and each team's bias, each with a 95% half-width."""
    n = len(e_a)
    d = (e_a**2).mean(axis=1) - (e_b**2).mean(axis=1)
    db = e_a - e_b
    return {
        "rows": n,
        "mean_sq_error_change": float(d.mean()),
        "mean_sq_error_change_pm95": float(Z95 * d.std(ddof=1) / np.sqrt(n)),
        "bias_change_pct": dict(zip(TEAMS, map(float, db.mean(axis=0)))),
        "bias_change_pm95_pct": dict(zip(TEAMS, map(float, Z95 * db.std(axis=0, ddof=1) / np.sqrt(n)))),
    }


def sibling_regret(pred, shard, groups):
    """Mean regret and misorder rate over the groups: the mover's exact value (percent of the
    money chance) of the best sibling less that of the sibling the network likes best for the
    mover (ties to the first row). Groups of rows ``groups['group']`` with ``groups['mover']``."""
    g = np.asarray(groups["group"]).astype(np.int64)
    mover = np.asarray(groups["mover"]).astype(np.int64)
    exact = np.asarray(shard["value"]).astype(np.float64) * (100.0 / ONE)
    pred = pred * (100.0 / ONE)
    rows = np.arange(len(g))
    ev, pv = exact[rows, mover], pred[rows, mover]
    cut = np.flatnonzero(np.diff(g)) + 1
    regrets, wrong = [], 0
    for r in np.split(rows, cut):
        if len(r) < 2:
            continue
        pick = r[np.argmax(pv[r])]
        best = ev[r].max()
        regrets.append(best - ev[pick])
        wrong += int(best - ev[pick] > 1e-9)
    x = np.array(regrets)
    n = len(x)
    return {
        "groups": n,
        "mean_regret_pct": float(x.mean()) if n else None,
        "mean_regret_pm95_pct": float(Z95 * x.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
        "misordered_share": wrong / n if n else None,
    }


def compare(nets, shard_paths, ref=None, siblings=None):
    shards = [load_shard(p) for p in shard_paths]
    shard = np.concatenate([np.asarray(s) for s in shards]) if len(shards) > 1 else np.asarray(shards[0])
    rep = {"shards": [str(p) for p in shard_paths], "nets": {}, "paired_vs_ref": {}, "ref": ref}
    errs = {}
    for name, path in nets.items():
        model = load_net(path)
        p = predict(model, shard)
        errs[name] = errors(p, shard)
        rep["nets"][name] = {"file": str(path), **summary(errs[name])}
        if siblings:
            sh = np.asarray(load_shard(siblings))
            groups = load_groups(siblings, len(sh))
            if groups is None:
                raise SystemExit(f"{siblings}: no groups sidecar")
            rep["nets"][name]["siblings"] = sibling_regret(predict(model, sh), sh, groups)
    if ref:
        for name in nets:
            if name != ref:
                rep["paired_vs_ref"][name] = paired(errs[name], errs[ref])
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fortyone_nn.netcmp", description=__doc__.split("\n\n")[0])
    ap.add_argument("--shard", action="append", required=True)
    ap.add_argument("--net", action="append", required=True, help="NAME=float.safetensors")
    ap.add_argument("--ref")
    ap.add_argument("--siblings")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    nets = dict(x.split("=", 1) for x in a.net)
    if a.ref and a.ref not in nets:
        raise SystemExit(f"--ref {a.ref} is not one of the nets")
    rep = compare(nets, a.shard, a.ref, a.siblings)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    for name, s in rep["nets"].items():
        sib = s.get("siblings", {}).get("mean_regret_pct")
        print(f"{name}: rows {s['rows']}, sd {s['sd_pct']:.4f}%, largest |bias| {s['largest_abs_bias_pct']:.4f}%"
              + (f", sibling regret {sib:.4f}%" if sib is not None else ""))


if __name__ == "__main__":
    main()
