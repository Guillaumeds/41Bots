"""The leaf-risk predictor and its threshold (docs/m3-hybrid/rung2.md "Learned exact leaves"):
which trick-3 frontier positions the hybrid H_3 solves exactly instead of taking the oracle
network O_3's value, learned from data instead of chosen by hand.

    uv run python -m fortyone_nn.leafrisk prepare --net O3.safetensors --shard H.npy --out H.npz
    uv run python -m fortyone_nn.leafrisk train --data TRAIN.npz [--data ...] --out MODEL.json
        [--trees 150] [--depth 6] [--rate 0.05] [--min-leaf 400] [--subsample 0.5] [--seed 1]
        [--holdout 0.1]
    uv run python -m fortyone_nn.leafrisk calibrate --model MODEL.json --cal CAL.npz
        --test TEST.npz --out REPORT.json [--delta 0.05] [--targets 1.6,1.4,...]

numpy only. The parts and their research:

- **Learning to defer / selective prediction** (Geifman & El-Yaniv 2017; Mozannar & Sontag 2020):
  a second model decides, per input, whether the cheap predictor (the network) is trusted or the
  case goes to the expensive expert (the exact solver). Its score here is the network's
  **expected squared error** at the position, summed over the four teams.
- **Value of computation** (Russell & Wefald 1991; Hay, Russell, Tolpin & Shimony 2012): spend a
  computation where it can change a decision. A leaf's error `e` changes a comparison between
  two values whose true gap `g` is smaller than `|e|` and of the other sign; with a density
  `f(0)` of gaps near zero the expected regret is `f(0) e^2 / 2`. So the expected squared error
  is the myopic value of an exact solve up to the factor `f(0)` (the decision's sensitivity),
  which a leaf-local rule cannot see and which is taken as constant (module notes in
  ``crates/search/src/exact_leaves.rs``).
- **The model**: gradient-boosted regression trees (Friedman 2001, squared loss; row
  subsampling, Friedman 2002; histogram splits on at most 64 bins a feature, as LightGBM, Ke et
  al. 2017), every split a whole-number threshold and every leaf a whole number, so Rust
  (``nn::leafrisk``) predicts bit for bit what Python does (``predict``).
- **The threshold** with a statistical guarantee (Learn then Test, Angelopoulos, Bates, Candès,
  Jordan & Lei 2021; conformal risk control, Angelopoulos, Bates, Fisch, Lei & Schuster 2022):
  the loss of a threshold `t` at a position is its squared error when it is left to the network
  (score below `t`), 0 when solved. The loss only grows with `t`, so the thresholds are tested in
  order from "solve everything" upwards (fixed-sequence testing, which needs no multiplicity
  correction) and the largest `t` whose upper confidence bound on the risk is within the target
  is kept: then P(risk(t̂) <= target) >= 1 - delta. The risk is per **decision** (the mean loss
  over a decision's harvested positions, then the mean over decisions), because positions of one
  decision are not independent and decisions are what the hybrid is judged on; decisions are
  recovered as runs of rows with the same match (``decision_runs``: neighbouring decisions with
  the same match merge, which only makes the units larger, still independent). Two bounds:
  ``bernstein`` (empirical Bernstein, Maurer & Pontil 2009: finite-sample, valid for any
  distribution of losses in [0, 2], the most a squared money error can be; it needs hundreds of
  thousands of decisions to be tight) and ``clt`` (the normal approximation, asymptotically
  valid, as Learn then Test allows). Conformal risk control's expectation form (risk at most the
  target on average, `(n R + B) / (n + 1) <= target`) is reported too.

**Rows**: a harvest label shard at trick 3 (``search labels --source harvest``, the audit shards'
recipe: positions the hybrid's own search reaches from trick-2 training decisions, exactly
solved), with O_3's integer money chance computed here (``audit.ExactNet``, the network Rust
runs, bit for bit). Errors are whole numbers: ``sse`` = sum over the teams of (network - exact)^2
in units of 2^-64, shifted down 24 bits to units of 2^-40 (``SCORE_SHIFT``).
"""

import argparse
import json
import math
import time

import numpy as np

from . import audit, intnet
from .formats import N_BINARY, N_DENSE, N_FEATURES, OFF_GAMES, PAD, load_shard, sha256_file

FORMAT = "41bots-leafrisk-v1"
SCORE_SHIFT = 40
ONE = 1 << 32
GAME_WON = 4
EXTRA_NAMES = [
    "race class",
    "network: leader's team",
    "network: next team",
    "network: opposite team",
    "network: previous team",
    "network: top share",
    "network: top-two margin",
    "network: Gini impurity",
    "L1 to the nearest possible outcome",
    "L1 to the farthest possible outcome",
    "possible outcomes",
    "teams that can win the game",
    "widest money range of one team",
]
N_EXTRAS = len(EXTRA_NAMES)
N_INPUTS = N_FEATURES + N_EXTRAS
MAX_BINS = 64
# The largest loss of a position: sum over teams of a squared difference of two money vectors.
LOSS_BOUND = 2.0


# ---------------------------------------------------------------------------------------------
# Rows.


def race_class_of(mask, active):
    win = np.stack([(np.asarray(mask, dtype=np.uint64) >> np.uint64(t)) & np.uint64(1) for t in range(4)], axis=1)
    win = win.astype(bool)
    game = np.stack([(active == OFF_GAMES + t).any(axis=1) for t in range(4)], axis=1)
    k = win.sum(axis=1)
    money = (win & game).any(axis=1)
    return np.where(k == 0, 0, np.where(~money, np.where(k == 1, 1, 2), np.where(k == 1, 3, 4))).astype(np.int64)


def extras(active, mask, outcomes, money):
    """The extra inputs (``EXTRA_NAMES``, ``nn::leafrisk::extras``) of rows: active lists, possible
    outcomes (mask, vectors [n, V, 4]) and the network's money chance [n, 4], all rotated."""
    active = np.asarray(active)
    m = np.asarray(money, dtype=np.uint64)
    n = len(m)
    vecs = np.asarray(outcomes, dtype=np.uint64).astype(np.int64)
    v = vecs.shape[1]
    possible = (np.asarray(mask, dtype=np.uint64)[:, None] >> np.arange(v, dtype=np.uint64)) & np.uint64(1)
    possible = possible.astype(bool)
    mi = m.astype(np.int64)
    srt = np.sort(mi, axis=1)
    pairs = np.zeros(n, dtype=np.uint64)
    for t in range(4):
        for u in range(t + 1, 4):
            pairs += m[:, t] * m[:, u]
    l1 = np.abs(vecs - mi[:, None, :]).sum(axis=2)
    big = np.int64(1) << 62
    near = np.where(possible, l1, big).min(axis=1)
    far = np.where(possible, l1, -1).max(axis=1)
    hi = np.where(possible[:, :, None], vecs, -1).max(axis=1)
    lo = np.where(possible[:, :, None], vecs, big).min(axis=1)
    rng = (hi - lo).max(axis=1)
    msk = np.asarray(mask, dtype=np.uint64)
    pop = np.array([bin(int(x)).count("1") for x in msk], dtype=np.int64)
    gpop = np.array([bin(int(x) & 0xF).count("1") for x in msk], dtype=np.int64)
    out = np.stack(
        [
            race_class_of(mask, active),
            mi[:, 0],
            mi[:, 1],
            mi[:, 2],
            mi[:, 3],
            srt[:, 3],
            srt[:, 3] - srt[:, 2],
            (pairs >> np.uint64(32)).astype(np.int64),
            near,
            far,
            pop,
            gpop,
            rng,
        ],
        axis=1,
    )
    return out.astype(np.int64)


def sse_units(money, value):
    """Sum over the teams of (money - value)^2 in units of 2^-64, floored to units of 2^-40."""
    e = np.asarray(money, dtype=np.uint64).astype(np.int64) - np.asarray(value, dtype=np.uint64).astype(np.int64)
    a = np.abs(e).astype(object)
    return np.array([int(x) >> (64 - SCORE_SHIFT) for x in (a * a).sum(axis=1)], dtype=np.int64)


def outcome_uncertainty(z, mask):
    """Diagnostics of the outcome head (floats, not inputs of the deployed model): the entropy
    (bits) of the softmax over the possible outcomes and its top probability."""
    z = np.asarray(z, dtype=np.float64) / float(1 << intnet.LOGIT_SHIFT)
    v = z.shape[1]
    possible = ((np.asarray(mask, dtype=np.uint64)[:, None] >> np.arange(v, dtype=np.uint64)) & np.uint64(1)).astype(bool)
    z = np.where(possible, z, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=1)
    return h.astype(np.float32), p.max(axis=1).astype(np.float32)


def decision_runs(active):
    """Unit ids: runs of consecutive rows with the same match. Rows of a decision are consecutive
    in a harvest shard and share its trump, which teams have a game and which team deals; the
    scores are not used (a row's scores include the hand's tricks so far, which differ between
    the lines of one decision). The features are in each row's own rotation, so the teams' (game,
    dealer) pairs are compared as a sorted list. Neighbouring decisions that agree on these merge
    into one unit, which only makes units larger (still independent of each other)."""
    a = audit
    act = np.asarray(active).astype(np.int64)
    valid = act != PAD
    trump = np.full(len(act), -1, dtype=np.int64)
    games = np.zeros((len(act), 4), dtype=np.int64)
    dealer = np.zeros((len(act), 4), dtype=np.int64)
    for k in range(4):
        trump = np.where((valid & (act == a.OFF_TRUMP + k)).any(axis=1), k, trump)
        games[:, k] = (valid & (act == OFF_GAMES + k)).any(axis=1)
        dealer[:, k] = (valid & (act == a.OFF_DEALER + k)).any(axis=1)
    pairs = np.sort(games * 2 + dealer, axis=1)
    key = trump * 256 + pairs[:, 0] * 64 + pairs[:, 1] * 16 + pairs[:, 2] * 4 + pairs[:, 3]
    change = np.concatenate([[0], (key[1:] != key[:-1]).astype(np.int64)])
    return np.cumsum(change)


def prepare(net_path, shard_path, out, chunk=4096, log=print):
    """O_3's value, the inputs and the squared error of every row of a harvest shard, into a
    ``.npz``: ``active``, ``dense``, ``extras``, ``sse``, ``race``, ``unit`` (``decision_runs``),
    ``oent`` and ``otop`` (outcome-head diagnostics)."""
    layers, meta = intnet.load(net_path)
    head = meta.get("head", "money4")
    if head != "outcome-v1":
        raise ValueError("the leaf-risk predictor is for outcome-head networks")
    net = audit.ExactNet(layers)
    s = load_shard(shard_path)
    n = len(s)
    parts = {k: [] for k in ("active", "dense", "extras", "sse", "race", "oent", "otop")}
    t0 = time.time()
    checked = False
    for i in range(0, n, chunk):
        r = np.asarray(s[i : i + chunk])
        z = net.logits(r["active"], r["dense"])
        m = audit.money_outcomes(z, r["mask"], r["outcomes"])
        if not checked:
            ref = audit.reference_values(layers, head, r[:64])
            if not np.array_equal(ref, m[:64]):
                raise AssertionError("the fast integer path differs from intnet")
            checked = True
        x = extras(r["active"], r["mask"], r["outcomes"], m)
        parts["active"].append(np.asarray(r["active"]))
        parts["dense"].append(np.asarray(r["dense"]))
        parts["extras"].append(x)
        parts["sse"].append(sse_units(m, r["value"]))
        parts["race"].append(x[:, 0].astype(np.int8))
        h, top = outcome_uncertainty(z, r["mask"])
        parts["oent"].append(h)
        parts["otop"].append(top)
        if (i // chunk) % 20 == 0:
            log(f"prepare: {min(i + chunk, n)} of {n} rows, {time.time() - t0:.0f} s")
    d = {k: np.concatenate(v) for k, v in parts.items()}
    d["unit"] = decision_runs(d["active"])
    d["net_sha256"] = np.array(sha256_file(net_path))
    d["shard_sha256"] = np.array(sha256_file(shard_path))
    np.savez_compressed(out, **d)
    return d


def load_rows(paths):
    ds = [np.load(p) for p in paths]
    out = {}
    for k in ("active", "dense", "extras", "sse", "race", "oent", "otop"):
        out[k] = np.concatenate([d[k] for d in ds])
    units, base = [], 0
    for d in ds:
        units.append(d["unit"] + base)
        base += int(d["unit"].max()) + 1
    out["unit"] = np.concatenate(units)
    out["net_sha256"] = {str(d["net_sha256"]) for d in ds}
    return out


# ---------------------------------------------------------------------------------------------
# The trees.


def column_values(d):
    """The non-binary inputs: dense features then extras (int64 [n, N_DENSE + N_EXTRAS])."""
    return np.concatenate([np.asarray(d["dense"], dtype=np.int64), np.asarray(d["extras"], dtype=np.int64)], axis=1)


def bin_thresholds(x, max_bins=MAX_BINS):
    """Up to ``max_bins - 1`` thresholds per column: values of the data at evenly spaced
    quantiles (a split ``x <= thr`` at each)."""
    out = []
    for j in range(x.shape[1]):
        q = np.quantile(x[:, j], np.linspace(0, 1, max_bins + 1)[1:-1], method="lower")
        thr = np.unique(q.astype(np.int64))
        thr = thr[thr < x[:, j].max()]
        out.append(thr)
    return out


def codes(x, thr):
    c = np.zeros(x.shape, dtype=np.uint8)
    for j, t in enumerate(thr):
        c[:, j] = np.searchsorted(t, x[:, j], side="left")
    return c


def binary_pairs(active):
    active = np.asarray(active)
    r, c = np.nonzero(active != PAD)
    f = active[r, c].astype(np.int64)
    order = np.argsort(f, kind="stable")
    rows_by_f = r[order].astype(np.int64)
    starts = np.searchsorted(f[order], np.arange(N_BINARY + 1))
    return r.astype(np.int64), f, rows_by_f, starts


def fit(d, trees=150, depth=6, rate=0.05, min_leaf=400, subsample=0.5, lam=1.0, seed=1, holdout=0.1, log=print):
    """Gradient-boosted regression trees on ``sse`` (module notes). Returns the model dict
    (``FORMAT``) with the gain of every input (``importance``) and the held-out error curve; the
    number of trees is the one with the least held-out squared error."""
    y_all = np.asarray(d["sse"], dtype=np.float64)
    n_all = len(y_all)
    rng = np.random.default_rng(seed)
    # Hold out whole units (decisions).
    units = np.asarray(d["unit"])
    hold_u = rng.random(int(units.max()) + 1) < holdout
    hold = hold_u[units]
    x = column_values(d)
    thr = bin_thresholds(x[~hold])
    C = codes(x, thr)
    K = C.shape[1]
    nthr = np.array([len(t) for t in thr])
    rr, ff, rows_by_f, starts = binary_pairs(d["active"])
    base = int(round(float(y_all[~hold].mean())))
    F = np.full(n_all, float(base))
    model_trees = []
    gain_by_input = np.zeros(N_INPUTS)
    curve = []
    best = (math.inf, 0)
    t0 = time.time()
    on_cache = {}

    def rows_on(f):
        if f not in on_cache:
            m = np.zeros(n_all, dtype=bool)
            m[rows_by_f[starts[f] : starts[f + 1]]] = True
            on_cache[f] = m
        return on_cache[f]

    for k in range(trees):
        g = y_all - F
        insample = (~hold) & (rng.random(n_all) < subsample)
        pos = np.zeros(n_all, dtype=np.int64)  # node of every row
        feat, thrv, left, right, value = [0], [0], [0], [0], [0.0]
        level = [0]
        node_gain = {}
        for _depth in range(depth + 1):
            if not level:
                break
            nl = len(level)
            local = np.full(len(feat), -1, dtype=np.int64)
            local[np.array(level)] = np.arange(nl)
            loc = local[pos]
            act = insample & (loc >= 0)
            idx = np.nonzero(act)[0]
            li = loc[idx]
            gi = g[idx]
            G = np.bincount(li, weights=gi, minlength=nl)
            H = np.bincount(li, minlength=nl).astype(np.float64)
            split = [None] * nl
            if _depth < depth:
                # Non-binary inputs: histograms over (node, column, bin).
                key = (li[:, None] * (K * MAX_BINS) + np.arange(K)[None, :] * MAX_BINS + C[idx]).ravel()
                Gd = np.bincount(key, weights=np.repeat(gi, K), minlength=nl * K * MAX_BINS).reshape(nl, K, MAX_BINS)
                Hd = np.bincount(key, minlength=nl * K * MAX_BINS).reshape(nl, K, MAX_BINS).astype(np.float64)
                GL, HL = np.cumsum(Gd, axis=2), np.cumsum(Hd, axis=2)
                GR, HR = G[:, None, None] - GL, H[:, None, None] - HL
                parent = (G * G / (H + lam))[:, None, None]
                gain_d = GL * GL / (HL + lam) + GR * GR / (HR + lam) - parent
                ok = (HL >= min_leaf) & (HR >= min_leaf) & (np.arange(MAX_BINS)[None, None, :] < nthr[None, :, None])
                gain_d = np.where(ok, gain_d, -np.inf)
                # Binary inputs: right = on.
                pm = act[rr]
                lb = loc[rr[pm]]
                keyb = lb * N_BINARY + ff[pm]
                Gb = np.bincount(keyb, weights=g[rr[pm]], minlength=nl * N_BINARY).reshape(nl, N_BINARY)
                Hb = np.bincount(keyb, minlength=nl * N_BINARY).reshape(nl, N_BINARY).astype(np.float64)
                GLb, HLb = G[:, None] - Gb, H[:, None] - Hb
                gain_b = GLb * GLb / (HLb + lam) + Gb * Gb / (Hb + lam) - (G * G / (H + lam))[:, None]
                gain_b = np.where((HLb >= min_leaf) & (Hb >= min_leaf), gain_b, -np.inf)
                for j in range(nl):
                    bd = int(np.argmax(gain_d[j]))
                    bb = int(np.argmax(gain_b[j]))
                    gd = gain_d[j].ravel()[bd]
                    gb = gain_b[j][bb]
                    if max(gd, gb) <= 0 or not np.isfinite(max(gd, gb)):
                        continue
                    if gb >= gd:
                        split[j] = (bb, 0, None, gb)
                    else:
                        col, b = divmod(bd, MAX_BINS)
                        split[j] = (N_BINARY + col, int(thr[col][b]), (col, b), gd)
            nxt = []
            for j, nd in enumerate(level):
                if split[j] is None:
                    feat[nd] = -1
                    value[nd] = rate * G[j] / (H[j] + lam)
                    continue
                f, t, cb, gn = split[j]
                gain_by_input[f] += gn
                at = pos == nd
                if cb is None:
                    go_right = at & rows_on(f)
                else:
                    go_right = at & (C[:, cb[0]] > cb[1])
                lc, rc = len(feat), len(feat) + 1
                feat += [0, 0]
                thrv += [0, 0]
                left += [0, 0]
                right += [0, 0]
                value += [0.0, 0.0]
                feat[nd], thrv[nd], left[nd], right[nd] = f, t, lc, rc
                pos[at] = lc
                pos[go_right] = rc
                nxt += [lc, rc]
            level = nxt
        for nd in level:
            feat[nd] = -1
        vals = [int(round(v)) if feat[i] < 0 else 0 for i, v in enumerate(value)]
        F += np.asarray(vals, dtype=np.float64)[pos]
        model_trees.append({"f": feat, "t": thrv, "l": left, "r": right, "v": vals})
        mse = float(np.mean((y_all[hold] - np.maximum(F[hold], 0)) ** 2)) if hold.any() else 0.0
        curve.append(mse)
        if mse < best[0]:
            best = (mse, k + 1)
        if k % 20 == 0 or k == trees - 1:
            log(f"train: tree {k + 1} of {trees}, held-out MSE {mse:.4e} (best {best[0]:.4e} at {best[1]}), {time.time() - t0:.0f} s")
    keep = best[1] if hold.any() else trees
    return {
        "format": FORMAT,
        "encoder": "oracle-v2",
        "n_inputs": N_INPUTS,
        "extras": EXTRA_NAMES,
        "unit": "predicted sum over the four teams of the network's squared money-chance error, units of 2^-40",
        "base": base,
        "trees": model_trees[:keep],
        "training": {
            "method": "gradient-boosted regression trees (Friedman 2001), squared loss, shrinkage, row subsampling (Friedman 2002), histogram splits",
            "rows": int((~hold).sum()),
            "held_out_rows": int(hold.sum()),
            "trees_grown": trees,
            "trees_kept": keep,
            "depth": depth,
            "rate": rate,
            "min_leaf": min_leaf,
            "subsample": subsample,
            "lambda": lam,
            "seed": seed,
            "held_out_mse_curve": curve,
        },
        "importance": importance(gain_by_input),
    }


def input_name(i):
    if i < N_BINARY:
        return audit.binary_name(i)
    if i < N_FEATURES:
        return audit.dense_name(i - N_BINARY)
    return EXTRA_NAMES[i - N_FEATURES]


def importance(gain, top=40):
    total = float(gain.sum()) or 1.0
    order = np.argsort(-gain, kind="stable")
    return [
        {"input": int(i), "name": input_name(int(i)), "share": float(gain[i] / total)}
        for i in order[:top]
        if gain[i] > 0
    ]


def predict(model, d):
    """The model's integer score of every row, as ``nn::leafrisk::RiskModel::predict``: the base
    plus every tree's leaf, clamped at 0 (int64)."""
    x = column_values(d)
    active = np.asarray(d["active"])
    n = len(x)
    on = np.zeros((n, N_BINARY), dtype=bool)
    r, c = np.nonzero(active != PAD)
    on[r, active[r, c].astype(np.int64)] = True

    def get(rows, f):
        if f < N_BINARY:
            return on[rows, f].astype(np.int64)
        return x[rows, f - N_BINARY]

    total = np.full(n, int(model["base"]), dtype=object)
    for t in model["trees"]:
        node = np.zeros(n, dtype=np.int64)
        f = np.asarray(t["f"])
        for _ in range(len(f)):
            internal = f[node] >= 0
            if not internal.any():
                break
            for nd in np.unique(node[internal]):
                rows = np.nonzero(node == nd)[0]
                go_left = get(rows, int(f[nd])) <= int(t["t"][nd])
                node[rows] = np.where(go_left, t["l"][nd], t["r"][nd])
        total = total + np.asarray(t["v"], dtype=np.int64)[node].astype(object)
    return np.array([max(0, min(int(v), (1 << 63) - 1)) for v in total], dtype=np.int64)


# ---------------------------------------------------------------------------------------------
# Thresholds with a guarantee.


def unit_losses(score, sse, unit, t):
    """Per unit (decision): the mean over its rows of the loss at threshold ``t`` (sse when the
    score is below ``t``, else 0), in squared money (sse / 2^40)."""
    loss = np.where(score < t, sse, 0).astype(np.float64) / float(1 << SCORE_SHIFT)
    u = np.asarray(unit)
    _, inv, cnt = np.unique(u, return_inverse=True, return_counts=True)
    return np.bincount(inv, weights=loss) / cnt


def ucb(x, delta, kind, bound=LOSS_BOUND):
    """An upper confidence bound at level 1 - delta on the mean of i.i.d. ``x`` in [0, bound]:
    ``bernstein`` (Maurer & Pontil 2009, Theorem 4) or ``clt`` (normal approximation)."""
    n = len(x)
    m = float(np.mean(x))
    if n < 2:
        return math.inf
    v = float(np.var(x, ddof=1))
    if kind == "bernstein":
        lg = math.log(2.0 / delta)
        return m + math.sqrt(2.0 * v * lg / n) + 7.0 * bound * lg / (3.0 * (n - 1))
    if kind == "clt":
        return m + audit.normal_isf(delta) * math.sqrt(v / n)
    raise ValueError(kind)


def candidate_thresholds(score, steps=400):
    """Thresholds to test, ascending: 0 (solve everything) and scores at evenly spaced
    quantiles."""
    q = np.quantile(score, np.linspace(0, 1, steps + 1), method="higher")
    return np.unique(np.concatenate([[0], q.astype(np.int64), [np.iinfo(np.int64).max]]))


def ltt_threshold(score, sse, unit, alpha, delta, kind="bernstein", grid=None):
    """Learn then Test with fixed-sequence testing: walk the thresholds from 0 upwards and keep
    the last one whose upper bound on the per-decision risk is at most ``alpha`` (stop at the
    first that is not). ``None`` when even t = 0 fails (it never does: its loss is 0)."""
    grid = candidate_thresholds(score) if grid is None else grid
    kept = None
    for t in grid:
        if ucb(unit_losses(score, sse, unit, t), delta, kind) <= alpha:
            kept = int(t)
        else:
            break
    return kept


def crc_threshold(score, sse, unit, alpha, grid=None, bound=LOSS_BOUND):
    """Conformal risk control (expectation form): the largest grid threshold with
    ``(n * risk + bound) / (n + 1) <= alpha`` (the loss grows with ``t``)."""
    grid = candidate_thresholds(score) if grid is None else grid
    kept = None
    for t in grid:
        x = unit_losses(score, sse, unit, t)
        n = len(x)
        if (n * float(np.mean(x)) + bound) / (n + 1) <= alpha:
            kept = int(t)
        else:
            break
    return kept


def risk_coverage(score, sse, unit=None, shares=None):
    """The risk-coverage curve: for each share of positions solved (the highest scores first),
    the share of the squared error left and the error SD left per team (root of the mean
    squared error over all positions and teams, the auditor's SD)."""
    shares = np.linspace(0, 1, 21) if shares is None else shares
    order = np.argsort(-np.asarray(score), kind="stable")
    s = np.asarray(sse, dtype=np.float64)[order] / float(1 << SCORE_SHIFT)
    total = float(s.sum())
    n = len(s)
    left = total - np.concatenate([[0.0], np.cumsum(s)])
    out = []
    for c in shares:
        k = int(round(c * n))
        out.append({"solved": float(c), "error_left": float(left[k] / total), "sd_left": math.sqrt(left[k] / (4 * n))})
    return out


def policy_point(solved, sse):
    """Share solved, share of the squared error left and SD left of a fixed set of solved rows."""
    s = np.asarray(sse, dtype=np.float64) / float(1 << SCORE_SHIFT)
    left = float(s[~solved].sum())
    return {"solved": float(solved.mean()), "error_left": left / float(s.sum()), "sd_left": math.sqrt(left / (4 * len(s)))}


def quantile_table(score, steps=1000):
    """``budget.quantiles``: entry ``k - 1`` is the score at or above which a share ``k / steps`` of
    the rows lie (non-increasing)."""
    s = np.sort(np.asarray(score, dtype=np.int64))[::-1]
    n = len(s)
    return [int(s[max(0, math.ceil(k * n / steps) - 1)]) for k in range(1, steps + 1)]


def calibrate(model, cal, test, targets_sd, delta=0.05, log=print):
    """Scores on the calibration and test splits; per target error SD left (percent per team) the
    Learn-then-Test thresholds (Bernstein and CLT) and the conformal-risk-control one, each with
    its share solved and error left on the test split; the risk-coverage curves of the model, of
    the hand-picked race classes and of the best possible order (by the true error); the budget
    table."""
    sc, st = predict(model, cal), predict(model, test)
    grid = candidate_thresholds(sc)
    out = {"delta": delta, "cal_rows": len(sc), "cal_units": int(len(np.unique(cal["unit"]))),
           "test_rows": len(st), "test_units": int(len(np.unique(test["unit"]))), "targets": []}
    base_risk = float(np.mean(unit_losses(sc, cal["sse"], cal["unit"], np.iinfo(np.int64).max)))
    out["cal_risk_none"] = base_risk
    out["cal_sd_none_pct"] = 100 * math.sqrt(base_risk / 4)
    for sd in targets_sd:
        alpha = 4 * (sd / 100) ** 2
        row = {"target_sd_pct": sd, "alpha": alpha}
        for name, t in (
            ("ltt_bernstein", ltt_threshold(sc, cal["sse"], cal["unit"], alpha, delta, "bernstein", grid)),
            ("ltt_clt", ltt_threshold(sc, cal["sse"], cal["unit"], alpha, delta, "clt", grid)),
            ("crc", crc_threshold(sc, cal["sse"], cal["unit"], alpha, grid)),
        ):
            if t is None:
                row[name] = None
                continue
            solved = st >= t
            risk = float(np.mean(unit_losses(st, test["sse"], test["unit"], t)))
            row[name] = {"t": t, "cal_solved": float((sc >= t).mean()), "test": policy_point(solved, test["sse"]),
                         "test_risk": risk, "test_risk_sd_pct": 100 * math.sqrt(risk / 4), "test_within": risk <= alpha}
        out["targets"].append(row)
        log(f"calibrate: target SD {sd}%: " + ", ".join(
            f"{k} t={row[k]['t']} solves {100 * row[k]['test']['solved']:.1f}% (test SD left {row[k]['test_risk_sd_pct']:.3f}%)"
            if row[k] else f"{k} none" for k in ("ltt_bernstein", "ltt_clt", "crc")))
    race = np.asarray(test["race"])
    out["curve_model"] = risk_coverage(st, test["sse"])
    out["curve_oracle"] = risk_coverage(test["sse"], test["sse"])
    out["race_policies"] = {
        name: policy_point(np.isin(race, cls), test["sse"])
        for name, cls in (("race:4", [4]), ("race:2,4", [2, 4]), ("race:1-4", [1, 2, 3, 4]))
    }
    for name, p in out["race_policies"].items():
        k = int(round(p["solved"] * len(st)))
        order = np.argsort(-st, kind="stable")
        solved = np.zeros(len(st), dtype=bool)
        solved[order[:k]] = True
        p["learned_at_same_share"] = policy_point(solved, test["sse"])
    out["budget_quantiles"] = quantile_table(st)
    # The outcome head's entropy as a lone score (a diagnostic: not an input of the model).
    out["curve_outcome_entropy"] = risk_coverage(np.asarray(test["oent"]), test["sse"])
    out["curve_race_class"] = risk_coverage(race.astype(np.int64) * 10 + 0, test["sse"])
    return out


# ---------------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fortyone_nn.leafrisk", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--net", required=True)
    p.add_argument("--shard", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--trees", type=int, default=150)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--rate", type=float, default=0.05)
    p.add_argument("--min-leaf", type=int, default=400)
    p.add_argument("--subsample", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--holdout", type=float, default=0.1)
    p = sub.add_parser("calibrate")
    p.add_argument("--model", required=True)
    p.add_argument("--cal", action="append", required=True)
    p.add_argument("--test", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--targets", default="1.6,1.5,1.4,1.3,1.2,1.1,1.0,0.9,0.8")
    p.add_argument("--write-budget", action="store_true", help="store the test split's budget table in the model")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare(a.net, a.shard, a.out)
    elif a.cmd == "train":
        d = load_rows(a.data)
        if len(d["net_sha256"]) != 1:
            raise SystemExit("the rows come from different networks")
        m = fit(d, trees=a.trees, depth=a.depth, rate=a.rate, min_leaf=a.min_leaf, subsample=a.subsample,
                seed=a.seed, holdout=a.holdout)
        m["net_sha256"] = next(iter(d["net_sha256"]))
        m["trick"] = 3
        m["budget"] = None
        with open(a.out, "w") as f:
            json.dump(m, f)
    else:
        with open(a.model) as f:
            model = json.load(f)
        cal, test = load_rows(a.cal), load_rows(a.test)
        targets = [float(x) for x in a.targets.split(",")]
        rep = calibrate(model, cal, test, targets, a.delta)
        rep["model_sha256"] = sha256_file(a.model)
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
        if a.write_budget:
            model["budget"] = {"what": "score quantiles of the test split: entry k-1 is the score at or above which a share k/1000 of its rows lie",
                               "quantiles": rep["budget_quantiles"]}
            with open(a.model, "w") as f:
                json.dump(model, f)


if __name__ == "__main__":
    main()
