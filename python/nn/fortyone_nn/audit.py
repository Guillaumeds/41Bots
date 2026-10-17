"""The auditor (docs/m3-hybrid/audit-loop.md §2-§4 and §6, build B1): finds the groups of
positions where an integer oracle network is weak, without being told what to look for.

    uv run python -m fortyone_nn.audit --net int.safetensors --shard AUDIT.npy [--shard ...]
        --out DIR [--gamma 0.001] [--alpha-bias 0.0025] [--alpha-sd 0.01] [--max-depth 6]
        [--max-conj 3] [--fdr 0.05] [--screen 0.01] [--seed 1] [--chunk 0] [--workers 0]
        [--work DIR] [--keep-work] [--overall-only]

``--overall-only`` (docs/m3-hybrid/audit-loop.md §10): the network's errors and the whole shard's
numbers only (``overall``: SD, bias per team, p99.9, by race class; the same function and the same
rows as a full audit's ``overall``), no groups: the additive loop at trick 2 compares the best
network so far and a new one on the same fresh audit shard. ``report.json`` then holds
``overall_only: true`` and no ``audit``; no ``predicates.json`` is written.

numpy only (no PyTorch), so it runs wherever the label shards are.

**At scale** (docs/m3-hybrid/audit-loop.md, decisions of 2026-10-16 night: gamma 0.0001, so
audit shards of 10-20M rows): every pass runs on ``--workers`` processes (default: one per CPU
this process may use; forkserver start, one BLAS thread each) over tasks of a fixed size (row
blocks of 65,536, histogram tasks of 2^18 rows, triple tasks of 2^19 rows of one item) that do
not depend on the number of workers; the shard is memory-mapped and read block by block; shared
arrays are ``.npy`` work files (``--work``, default ``OUT/work``, removed after unless
``--keep-work``): the error columns, the race classes, half A's active lists, dense values and
bins, its rows' items and each item's rows as a packed bitset, and the confirmed groups' members
as packed bitsets. Memory is bounded by the task sizes (no per-group full-length mask is kept;
groups are confirmed 512 at a time per block). Every sum is a whole number (float64 products
and bincounts over at most 2^16 rows of values below 2^34 are exact, then added as int64), so
every output is byte-identical for any number of workers (tested: 1 and 4).

**Errors.** Every row's error is the integer network's money chance (``intnet``'s scheme, the one
Rust runs, computed here by a faster exact path that is checked against ``intnet.money_rows`` on
the first rows of every run) minus the exact label, per team, in units of 2^-32. Statistics use
whole-number sums of these (and of their squares, floored to units of 2^-32 of a squared money
chance), so every count, bias and SD is the same on any machine; floats appear only in the final
divisions. Teams are in the shard's order: relative to the player who leads the position's trick
(the leader's team, the next team, the opposite team, the previous team). The shard does not
record which seat made the decision whose search reached a position, so the hybrid check's
decider-relative view (own / next / opposite / previous) cannot be formed per row here.

Per group: ``weight`` (its share of rows), ``bias`` (mean error per team), ``sd`` (the error over
all four teams together, as the trainer and the check report it: errors sum to zero over the teams,
so this is the root mean square), ``p999`` (99.9th percentile of the absolute error over the four
teams' entries).

**Split.** Row ``i`` (rows of the shards one after another) is in half A (discovery) when the top
bit of the ``i``-th output of SplitMix64 seeded with ``--seed`` is 0, else in half B (confirmation).

**Groups** (the class C of plan §2), found on half A only; ``gamma`` sets the smallest group
(``ceil(gamma * rows of A)`` rows):

- regression trees (CART: Breiman, Friedman, Olshen, Stone 1984): one fitted to the signed error
  (the four teams' errors together, squared-error impurity summed over the teams), one to the
  absolute error (the mean over the teams of |error|); depth at most ``--max-depth``, every node at
  least the smallest group; a binary feature splits on/off, a dense feature at one of up to 63
  thresholds (quantiles of half A, each a value of the data); the best split by squared-error
  reduction, ties to the lower feature and threshold. Every node below the root is a group
  (the conjunction of the conditions on its path; SliceFinder's coarse-to-fine lattice, so a weak
  region is met at the coarsest node that holds it);
- the DivExplorer-style enumeration (Pastor, de Alfaro, Baralis 2021): every conjunction of up to
  ``--max-conj`` (1-3) items with support at least the smallest group on half A. The items are
  "binary feature i is on" for every binary feature, and "is off" for the yes/no features (a
  team's game, its best-trump bonus, the 5 of trumps played; for the one-hot blocks "off" is the
  union of the block's other values; an item on in every row defines no group and is not
  extended). Pruned by support (Apriori: a conjunction is counted only when every
  sub-conjunction is frequent), as SliceFinder does a candidate is not extended (a group inside
  it is reported through it), and by a valid bound on what any extension can reach
  (``Screen.extendable``: the optimistic estimate of exhaustive subgroup discovery; it never
  changes the candidates). The sums are sparse and exact: pairs from every row's list of items
  (a row of half A holds about 46 of them), triples per item over that item's rows; so the
  candidates are exactly those of whole-number arithmetic on any machine and worker count
  (``Enumeration``).

**Candidates.** A group whose half-A statistics exceed a threshold (``|bias| > alpha_bias`` for
some team, or ``sd > alpha_sd``) and whose one-sided test on half A gives p <= ``--screen`` (the
SliceFinder definition of a problematic slice: effect size and significance; normal
approximation; for the bias the variance is bounded by 3 times the mean square, as every team's
squared error is at most 3 times the four teams' mean square when the errors sum to zero).
``--screen 1`` takes every group over a threshold, the plan's literal rule; with heavy-tailed
errors that is mostly groups holding one large error, which half B then rejects.

**Confirmation** on half B, independent of the selection: per candidate and criterion it was
flagged for, a one-sided test of the null "within the threshold": the bias of the team and
direction chosen on half A (Student t), the SD by the mean squared error (Student t on the per-row
mean square: the test of a mean, which is what a bootstrap of the mean square approximates; the
F-test of variances is not used because it assumes normal errors, and these are far heavier
tailed). Benjamini-Hochberg at ``--fdr`` over all the hypotheses; a group is confirmed when one of
its hypotheses is. Confirmed groups contained in a more general confirmed group (its conditions a
subset, or bounds of the same dense features at least as tight: its conditions imply the other's)
are listed in report.json but left out of predicates.json (SliceFinder's minimality); the general
group covers their positions.

**Outputs** in DIR (the same files for the same inputs and seed):

- ``predicates.json``: the plan §6 format (below), the confirmed groups (an empty list when none);
- ``report.txt``: plain English: rows audited, candidates, confirmed, the worst groups with their
  numbers, the verdict ("flat" when nothing is confirmed);
- ``report.json``: everything: thresholds, split seed, counts, every confirmed group with its
  half-A and half-B statistics, p-values, race-class mix (a diagnostic: race classes are not
  features), timings.

**Predicate format** (shared with Rust ``nn::predicate``; fixture under ``tests/fixtures``): a JSON
list of groups ``{"name": str, "conditions": [{"feature": i, "op": op, "value": v}, ...], ...}``.
``feature`` indexes the encoder's feature vector (``crates/nn/src/encode.rs``, ``oracle-v2``):
0..808 the binary features, 809..908 the dense features (dense ``j`` is ``809 + j``). For a binary
feature ``op`` is ``"=="`` and ``value`` 0 or 1 (the feature is on when its index is in the row's
active list). For a dense feature ``op`` is ``"<="`` or ``">"`` and ``value`` a whole number that
fits a signed 32-bit integer, in the feature's own units (``log2(p) / 32`` in units of 2^-24, as
the shard stores it), compared with the row's dense value as integers. A row is in the group when
every condition holds (an empty list: every row). Other keys (``weight``, ``bias``, ``sd``, ...)
are information; readers ignore them.
"""

import argparse
import json
import math
import os
import time

import numpy as np

from . import intnet
from .formats import (
    INPUT_SHIFT, MAX_ACTIVE, N_BINARY, N_DENSE, N_FEATURES, ONE, PAD, RACE_CLASS_NAMES, VALUE_FAULTS,
    load_shard, race_classes, sha256_file, value_fault,
)

FORMAT = "41bots-audit-v1"

# ---------------------------------------------------------------------------------------------
# The encoder's layout (crates/nn/src/encode.rs, oracle-v2), for names and the yes/no items.

SLOTS = 54
PLACES = 5
OFF_SLOT_SEAT = 0
OFF_SLOT_PLACE = OFF_SLOT_SEAT + SLOTS * 8
OFF_TRUMP = OFF_SLOT_PLACE + SLOTS * PLACES
OFF_TRICKS_DONE = OFF_TRUMP + 4
N_SCORES = 15
SCORE_VALUES = (0, 5, 10, 11, 15, 16, 20, 21, 25, 26, 30, 31, 35, 36, 40)
OFF_SCORE = OFF_TRICKS_DONE + 5
OFF_FLAG = OFF_SCORE + 4 * N_SCORES
OFF_TRICKS = OFF_FLAG + 4
OFF_FIVE_PLAYED = OFF_TRICKS + 4 * 5
OFF_END_BONUS = OFF_FIVE_PLAYED + 1
OFF_GAMES = OFF_END_BONUS + 5
OFF_DEALER = OFF_GAMES + 4
assert OFF_DEALER + 4 == N_BINARY
DENSE_NOW, DENSE_GAME, DENSE_NEXT1, DENSE_NEXT2 = 0, 4, 20, 36
CLASS_BASE = (0, 1, 4, 15, 28, 41)
# Binary features whose "off" is itself a natural item: games won, best-trump bonus flags, the
# 5 of trumps played.
YES_NO = tuple(range(OFF_GAMES, OFF_GAMES + 4)) + tuple(range(OFF_FLAG, OFF_FLAG + 4)) + (OFF_FIVE_PLAYED,)

TEAMS = ("the leader's team", "the next team", "the opposite team", "the previous team")
SUITS = ("clubs", "diamonds", "hearts", "spades")


def _ordinal(k):
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(k, f"{k}th")


def _best(k):
    return "best" if k == 1 else f"{_ordinal(k)}-best"


def seat_name(s):
    return "the leader" if s == 0 else f"the {_ordinal(s + 1)} player"


def slot_name(i):
    if i == 0:
        return "the 5 of trumps"
    if i < CLASS_BASE[2]:
        return f"the {_best(i - CLASS_BASE[1] + 1)} live top trump (J, Joker, A of hearts)"
    if i < CLASS_BASE[3]:
        return f"the {_best(i - CLASS_BASE[2] + 1)} live ordinary trump"
    for c in (5, 4, 3):
        if i >= CLASS_BASE[c]:
            return f"the {_best(i - CLASS_BASE[c] + 1)} live card of plain suit {c - 2}"
    raise AssertionError(i)


def binary_name(f, on=True):
    """Plain English for binary feature ``f`` being on (or off)."""
    if f < OFF_SLOT_PLACE:
        i, s = divmod(f - OFF_SLOT_SEAT, 8)
        return f"{seat_name(s)} {'holds' if on else 'does not hold'} {slot_name(i)}"
    if f < OFF_TRUMP:
        i, p = divmod(f - OFF_SLOT_PLACE, PLACES)
        return f"{slot_name(i)} is {'' if on else 'not '}card {p + 1} of its holder's hand (card order)"
    if f < OFF_TRICKS_DONE:
        return f"trumps are {'' if on else 'not '}{SUITS[f - OFF_TRUMP]}"
    if f < OFF_SCORE:
        k = f - OFF_TRICKS_DONE
        return f"{k} trick{'' if k == 1 else 's'} done" + ("" if on else " is false")
    if f < OFF_FLAG:
        t, i = divmod(f - OFF_SCORE, N_SCORES)
        return f"{TEAMS[t]} has {'' if on else 'not '}{SCORE_VALUES[i]} points"
    if f < OFF_TRICKS:
        t = f - OFF_FLAG
        return f"{TEAMS[t]} has {'had' if on else 'had no'} best-trump bonus this game"
    if f < OFF_FIVE_PLAYED:
        t, k = divmod(f - OFF_TRICKS, 5)
        return f"{TEAMS[t]} has {'' if on else 'not '}won {k} trick{'' if k == 1 else 's'} this hand"
    if f == OFF_FIVE_PLAYED:
        return f"the 5 of trumps has {'' if on else 'not '}been played"
    if f < OFF_GAMES:
        k = f - OFF_END_BONUS
        if k == 0:
            return "no team would take the end-of-hand bonus" if on else "some team would take the end-of-hand bonus"
        return f"{TEAMS[k - 1]} would {'' if on else 'not '}take the end-of-hand bonus"
    if f < OFF_DEALER:
        t = f - OFF_GAMES
        return f"{TEAMS[t]} has {'won a game' if on else 'no game'}"
    if f < N_BINARY:
        return f"the dealer is {'' if on else 'not '}in {TEAMS[f - OFF_DEALER]}"
    raise ValueError(f"feature {f} is not binary")


def dense_name(j):
    if j < DENSE_GAME:
        return f"{TEAMS[j]}'s money chance if the hand ended now"
    if j < DENSE_NEXT1:
        t, u = divmod(j - DENSE_GAME, 4)
        return f"{TEAMS[u]}'s money chance if {TEAMS[t]} won the game now"
    if j < DENSE_NEXT2:
        a, u = divmod(j - DENSE_NEXT1, 4)
        return f"{TEAMS[u]}'s money chance after {TEAMS[a]} wins the next trick"
    a, r = divmod(j - DENSE_NEXT2, 16)
    b, u = divmod(r, 4)
    return f"{TEAMS[u]}'s money chance after {TEAMS[a]} then {TEAMS[b]} win the next two tricks"


def dense_chance(v):
    """A dense value (log2(p) / 32 in units of 2^-24) as the money chance p."""
    return 2.0 ** (int(v) / float(1 << (INPUT_SHIFT - 5)))


def _pct(p):
    x = 100 * p
    if x >= 0.1:
        return f"{x:.3g}%"
    return f"{x:.2e}%"


def condition_text(c):
    f, op, v = c["feature"], c["op"], c["value"]
    if f < N_BINARY:
        return binary_name(f, v == 1)
    word = "at most" if op == "<=" else "above"
    return f"{dense_name(f - N_BINARY)} is {word} {_pct(dense_chance(v))}"


def group_name(conditions):
    return "; ".join(condition_text(c) for c in conditions) if conditions else "every position"


# ---------------------------------------------------------------------------------------------
# Predicates (plan §6).

OPS = ("==", "<=", ">")


def check_condition(c):
    """Raises ValueError unless ``c`` is a valid condition of the predicate format."""
    if not isinstance(c, dict) or set(c) != {"feature", "op", "value"}:
        raise ValueError(f"a condition is {{feature, op, value}}: {c!r}")
    f, op, v = c["feature"], c["op"], c["value"]
    if type(f) is not int or not 0 <= f < N_FEATURES:
        raise ValueError(f"feature {f!r}: a whole number 0..{N_FEATURES - 1}")
    if type(v) is not int:
        raise ValueError(f"value {v!r}: a whole number")
    if f < N_BINARY:
        if op != "==" or v not in (0, 1):
            raise ValueError(f"binary feature {f}: op '==' and value 0 or 1, not {op!r} {v!r}")
    elif op not in ("<=", ">") or not -(1 << 31) <= v < (1 << 31):
        raise ValueError(f"dense feature {f}: op '<=' or '>' and a value that fits 32 bits, not {op!r} {v!r}")


def load_predicates(path):
    """The groups of a predicate file (a JSON list), checked."""
    with open(path) as fh:
        groups = json.load(fh)
    if not isinstance(groups, list):
        raise ValueError(f"{path}: a predicate file is a JSON list of groups")
    for g in groups:
        if not isinstance(g, dict) or not isinstance(g.get("conditions"), list):
            raise ValueError(f"{path}: a group is an object with a list of conditions")
        for c in g["conditions"]:
            check_condition(c)
    return groups


def binary_matrix(active):
    """bool [n, N_BINARY]: binary feature f of row r is on (``f`` in its active list)."""
    active = np.asarray(active)
    n = len(active)
    out = np.zeros((n, N_BINARY), dtype=bool)
    rows = np.repeat(np.arange(n), active.shape[1])
    cols = active.reshape(-1).astype(np.int64)
    m = cols != PAD
    if np.any(cols[m] >= N_BINARY):
        raise ValueError("an active index that is neither a binary feature nor the padding")
    out[rows[m], cols[m]] = True
    return out


def members(conditions, binary, dense):
    """bool [n]: the rows in the group (``binary`` from ``binary_matrix``, ``dense`` the shard's
    int32 dense values)."""
    m = np.ones(len(binary), dtype=bool)
    for c in conditions:
        f, op, v = c["feature"], c["op"], c["value"]
        if f < N_BINARY:
            m &= binary[:, f] if v == 1 else ~binary[:, f]
        else:
            x = dense[:, f - N_BINARY]
            m &= (x <= v) if op == "<=" else (x > v)
    return m


def canonical(conditions):
    """Conditions in a fixed order with redundant bounds merged (a dense feature keeps its
    tightest "<=" and ">")."""
    bin_, le, gt = {}, {}, {}
    for c in conditions:
        f, op, v = c["feature"], c["op"], c["value"]
        if f < N_BINARY:
            bin_[f] = v
        elif op == "<=":
            le[f] = min(v, le.get(f, v))
        else:
            gt[f] = max(v, gt.get(f, v))
    out = [{"feature": f, "op": "==", "value": bin_[f]} for f in sorted(bin_)]
    for f in sorted(set(le) | set(gt)):
        if f in gt:
            out.append({"feature": f, "op": ">", "value": gt[f]})
        if f in le:
            out.append({"feature": f, "op": "<=", "value": le[f]})
    return out


def cond_key(conditions):
    return tuple((c["feature"], c["op"], c["value"]) for c in canonical(conditions))


def implies(key_g, key_h):
    """Whether the conjunction ``key_g`` implies ``key_h`` (``cond_key`` forms): every condition of
    ``h`` follows from one of ``g`` (the same binary condition; for a dense feature, ``x <= a``
    gives ``x <= b`` when a <= b, ``x > a`` gives ``x > b`` when a >= b). Then every position in
    group g is in group h."""
    for f, op, v in key_h:
        ok = False
        for f2, op2, v2 in key_g:
            if f2 == f and op2 == op and (v2 == v if op == "==" else v2 <= v if op == "<=" else v2 >= v):
                ok = True
                break
        if not ok:
            return False
    return True


# ---------------------------------------------------------------------------------------------
# The integer network, exactly (``intnet``'s numbers), faster.

F64_EXACT = 53


class ExactNet:
    """The integer forward pass of ``intnet.logits``, bit for bit, organised for speed: the first
    layer's binary part as one float64 product of the 0/1 inputs (at most 96 weights below 2^31
    are summed: exact), its dense part and the later layers through ``intnet.exact_matmul``'s
    pieces, cut once per layer for the largest inputs the layer can get (so no piece's partial
    sums reach 2^53)."""

    def __init__(self, layers):
        self.layers = layers
        w0, _, _ = layers[0]
        self.wb = w0[:N_BINARY].astype(np.float64)
        self.wd = self._pieces(w0[N_BINARY:], N_DENSE * (1 << INPUT_SHIFT))
        self.hidden = [(self._pieces(w, w.shape[0] * intnet.CLIP), b, s) for w, b, s in layers[1:]]

    @staticmethod
    def _pieces(w, bound):
        bits = F64_EXACT - int(bound).bit_length()
        if bits < 8:
            raise ValueError("a layer too wide for exact float64 pieces")
        out, rest, off = [], w.astype(np.int64), 0
        while True:
            top = not rest.size or max(-int(rest.min()), int(rest.max())) < (1 << bits)
            piece = rest if top else rest & ((1 << bits) - 1)
            out.append((piece.astype(np.float64), off))
            if top:
                return out
            rest = rest >> bits
            off += bits

    @staticmethod
    def _product(hf, pieces):
        out = None
        for p, off in pieces:
            x = (hf @ p).astype(np.int64)
            if off:
                x *= np.int64(1 << off)
            out = x if out is None else out + x
        return out

    def logits(self, active, dense):
        xb = binary_matrix(active).astype(np.float64)
        _, b0, s0 = self.layers[0]
        acc = (xb @ self.wb).astype(np.int64) * np.int64(1 << INPUT_SHIFT)
        acc += self._product(np.asarray(dense, dtype=np.float64), self.wd)
        acc += b0
        h = intnet.round_shift(acc, s0)
        last = len(self.layers) - 1
        if last:
            np.clip(h, 0, intnet.CLIP, out=h)
        for k, (pieces, b, s) in enumerate(self.hidden, start=1):
            h = intnet.round_shift(self._product(h.astype(np.float64), pieces) + b, s)
            if k != last:
                np.clip(h, 0, intnet.CLIP, out=h)
        return h


_M32 = np.uint64(0xFFFFFFFF)
_D_CAP = 1 << 31  # d >= 2^31 makes n >= 63 in ``intnet.exp_weight``: weight 0


def _mulshr62(w, r):
    """floor(w * r / 2^62), exactly, for uint64 ``w`` < 2^63 and a whole number ``r`` < 2^62."""
    b1, b0 = np.uint64(r >> 32), np.uint64(r & 0xFFFFFFFF)
    a1, a0 = w >> np.uint64(32), w & _M32
    lo = a0 * b0
    mid = a1 * b0 + a0 * b1  # < 2^63 + 2^62
    low = lo + (mid << np.uint64(32))
    high = a1 * b1 + (mid >> np.uint64(32)) + (low < lo).astype(np.uint64)
    return (high << np.uint64(2)) | (low >> np.uint64(62))


def exp_weights(d):
    """``intnet.exp_weight`` of every element of int64 ``d >= 0`` (uint64)."""
    d = np.minimum(np.asarray(d, dtype=np.int64), _D_CAP).astype(np.uint64)
    y = d * np.uint64(intnet.LOG2E_Q32)
    n = y >> np.uint64(56)
    f = (y >> np.uint64(32)) & np.uint64(0xFFFFFF)
    w = np.full(d.shape, 1 << 62, dtype=np.uint64)
    for m in range(23, -1, -1):
        sel = ((f >> np.uint64(m)) & np.uint64(1)).astype(bool)
        if sel.any():
            w[sel] = _mulshr62(w[sel], intnet.R[24 - m])
    out = np.zeros(d.shape, dtype=np.uint64)
    ok = n < np.uint64(63)
    out[ok] = w[ok] >> n[ok]
    return out


def _largest_remainder(q, r):
    """``q`` + one unit for the ``2^32 - sum(q)`` largest remainders ``r`` (ties to the lower team),
    rows of Python whole numbers; uint64 [n, 4]."""
    missing = (ONE - q.sum(axis=1)).astype(np.int64)
    rank = np.zeros(q.shape, dtype=np.int64)
    for i in range(4):
        for j in range(4):
            if j != i:
                beat = r[:, j] > r[:, i]
                if j < i:
                    beat = beat | (r[:, j] == r[:, i])
                rank[:, i] += beat.astype(np.int64)
    return (q + (rank < missing[:, None]).astype(object)).astype(np.uint64)


def money4(z):
    """``intnet.money`` of every row of logits."""
    z = np.asarray(z, dtype=np.int64)
    w = exp_weights(z.max(axis=1, keepdims=True) - z).astype(object)
    num = w * ONE
    total = w.sum(axis=1)[:, None]
    return _largest_remainder(num // total, num % total)


def money_outcomes(z, mask, vecs):
    """``intnet.money_outcomes`` of every row (possible outcomes from ``mask``)."""
    z = np.asarray(z, dtype=np.int64)
    v = z.shape[1]
    bits = np.asarray(mask).astype(np.uint64)[:, None] >> np.arange(v, dtype=np.uint64)
    possible = (bits & np.uint64(1)).astype(bool)
    top = np.where(possible, z, np.iinfo(np.int64).min).max(axis=1, keepdims=True)
    w = exp_weights(np.where(possible, top - z, _D_CAP)).astype(object)
    num = (w[:, :, None] * np.asarray(vecs).astype(object)).sum(axis=1)
    den = w.sum(axis=1)[:, None]
    return _largest_remainder(num // den, num % den)


def network_values(layers, head, shards, chunk=2048):
    """The integer network's money chance of every row of the shards (uint64 [n, 4])."""
    net = ExactNet(layers)
    out = []
    for s in shards:
        for i in range(0, len(s), chunk):
            r = np.asarray(s[i : i + chunk])
            z = net.logits(r["active"], r["dense"])
            out.append(money4(z) if head == "money4" else money_outcomes(z, r["mask"], r["outcomes"]))
    return np.concatenate(out) if out else np.zeros((0, 4), dtype=np.uint64)


def reference_values(layers, head, rows):
    """``intnet.money_rows`` (the path the trainer checks against Rust) of record rows."""
    if head == "money4":
        return intnet.money_rows(layers, rows["active"], rows["dense"])
    return intnet.money_rows(layers, rows["active"], rows["dense"], mask=rows["mask"], outcomes=rows["outcomes"])


# ---------------------------------------------------------------------------------------------
# Errors as whole numbers.


def _square_split(x):
    """(hi, lo) with x^2 = hi * 2^32 + lo, 0 <= lo < 2^32, for int64 |x| <= 2^32."""
    a = np.abs(x)
    ah, al = a >> 16, a & 0xFFFF
    mid = (2 * ah * al << 16) + al * al
    return ah * ah + (mid >> 32), mid & 0xFFFFFFFF


class Errors:
    """Per-row whole-number columns of the errors (units of 2^-32 of a money chance):

    - ``e`` [n, 4]: network minus label, per team;
    - ``e2`` [n, 4]: floor(e^2 / 2^32) (squares, units of 2^-32 of a squared money chance);
    - ``q`` [n]: floor(sum over teams of e^2 / 2^34): the four teams' mean square, same units;
    - ``qq`` [n]: floor(q^2 / 2^32).

    Every entry is below 2^33 in size, so sums over up to 2^20 rows are exact in float64 and
    int64."""

    def __init__(self, pred, value):
        self._set(np.asarray(pred).astype(np.int64) - np.asarray(value).astype(np.int64))

    @classmethod
    def of(cls, e):
        """From the errors themselves (int64 [n, 4], |e| <= 2^32)."""
        out = cls.__new__(cls)
        out._set(np.asarray(e, dtype=np.int64))
        return out

    def _set(self, e):
        if e.size and np.abs(e).max() > ONE:
            raise ValueError("an error beyond 2^32")
        hi, lo = _square_split(e)
        self.e = e
        self.e2 = hi
        self.q = (hi.sum(axis=1) + (lo.sum(axis=1) >> 32)) >> 2
        qh, _ = _square_split(self.q)
        self.qq = qh

    def __len__(self):
        return len(self.q)

    def columns(self):
        """int64 [n, 11]: 1, e (4), e2 (4), q, qq: what ``Stats`` is made of."""
        return np.column_stack([np.ones(len(self.q), dtype=np.int64), self.e, self.e2, self.q, self.qq])

    def fractions(self):
        return self.e.astype(np.float64) / ONE


SCREEN_COLS = (0, 1, 2, 3, 9, 10)  # 1, e0, e1, e2, q, qq: enough for the half-A screen


class Stats:
    """A group's statistics from whole-number sums (``Errors.columns`` summed over its rows)."""

    def __init__(self, sums):
        s = [int(x) for x in sums]
        self.n = s[0]
        self.sums = s
        n = max(self.n, 1)
        self.bias = [s[1 + t] / n / ONE for t in range(4)]
        ms = [s[5 + t] / n / ONE for t in range(4)]
        self.var = [max(ms[t] - self.bias[t] ** 2, 0.0) * n / max(n - 1, 1) for t in range(4)]
        self.msq = s[9] / n / ONE
        self.sd = math.sqrt(self.msq)
        self.var_q = max(s[10] / n / ONE - self.msq**2, 0.0) * n / max(n - 1, 1)

    def worst_team(self):
        """(team, sign) of the largest |bias| (ties to the lower team)."""
        t = max(range(4), key=lambda k: (abs(self.bias[k]), -k))
        return t, (1 if self.bias[t] >= 0 else -1)

    def p_sd(self, alpha_sd):
        """One-sided p-value of "mean square <= alpha_sd^2" (Student t, n - 1 degrees of freedom)."""
        return t_test_p(self.msq - alpha_sd**2, self.var_q, self.n)

    def p_bias(self, team, sign, alpha_bias):
        return t_test_p(sign * self.bias[team] - alpha_bias, self.var[team], self.n)

    def as_dict(self, total=None):
        d = {"rows": self.n, "bias": self.bias, "sd": self.sd}
        if total:
            d["weight"] = self.n / total
        return d


def sums_of(cols, rows=None):
    """Exact column sums (Python ints) of int64 ``cols`` over ``rows`` (indices or a mask)."""
    x = cols if rows is None else cols[rows]
    return [int(v) for v in x.sum(axis=0, dtype=np.int64)]


# ---------------------------------------------------------------------------------------------
# Tests: Student t, normal, Benjamini-Hochberg.


def _betacf(a, b, x):
    """The continued fraction of the regularised incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a, b, x):
    """The regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """P(T > t) for Student's t with ``df`` degrees of freedom."""
    if df <= 0:
        return 0.5
    if t == 0:
        return 0.5
    p = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return p if t > 0 else 1.0 - p


def t_test_p(diff, var, n):
    """One-sided p-value for "mean - bound > 0" with sample variance ``var`` over ``n`` rows."""
    if n < 2:
        return 1.0
    if var <= 0.0:
        return 0.0 if diff > 0 else 1.0
    return t_sf(diff / math.sqrt(var / n), n - 1)


def normal_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def normal_isf(p):
    """z with P(Z > z) = p (bisection; p in (0, 1))."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if normal_sf(mid) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def benjamini_hochberg(p, fdr):
    """bool [len(p)]: the hypotheses rejected by Benjamini-Hochberg at ``fdr`` (ties by index)."""
    p = np.asarray(p, dtype=np.float64)
    k = len(p)
    if not k:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((np.arange(k), p))
    ok = p[order] <= fdr * np.arange(1, k + 1) / k
    out = np.zeros(k, dtype=bool)
    if ok.any():
        out[order[: int(np.flatnonzero(ok)[-1]) + 1]] = True
    return out


class Screen:
    """The half-A candidate rule, vectorised over many groups' exact screening sums (columns
    ``SCREEN_COLS``: n, e0, e1, e2, q, qq; the fourth team's sum is minus the others')."""

    def __init__(self, alpha_bias, alpha_sd, minsup, screen):
        self.alpha_bias, self.alpha_sd, self.minsup = alpha_bias, alpha_sd, minsup
        self.z = normal_isf(screen) if screen < 1 else -math.inf

    def _decide(self, n, ab, msq_sd, var_q, msq_b):
        """(sd flag, bias flag) from: rows, the largest |bias|, the mean square and the variance
        of the per-row mean square (SD criterion), the mean square bounding the bias's variance."""
        nn = np.maximum(n, 2.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_sd = (msq_sd - self.alpha_sd**2) / np.sqrt(var_q / nn)
            var_b = np.maximum(3.0 * msq_b - ab**2, 0.0) * nn / (nn - 1)
            z_b = (ab - self.alpha_bias) / np.sqrt(var_b / nn)
        z_sd = np.where(var_q > 0, z_sd, np.where(msq_sd > self.alpha_sd**2, np.inf, -np.inf))
        z_b = np.where(var_b > 0, z_b, np.where(ab > self.alpha_bias, np.inf, -np.inf))
        ok = n >= self.minsup
        sd = ok & (msq_sd > self.alpha_sd**2) & (z_sd >= self.z)
        bi = ok & (ab > self.alpha_bias) & (z_b >= self.z)
        return sd, bi

    def __call__(self, s):
        """Exact sums int64 [k, 6] -> (candidate, sd flag, bias flag), bool [k] each."""
        s = np.asarray(s, dtype=np.float64).reshape(-1, 6)
        n = s[:, 0]
        nn = np.maximum(n, 2.0)
        e = s[:, 1:4]
        ab = np.abs(np.column_stack([e, -e.sum(axis=1)])).max(axis=1) / nn / ONE
        msq = s[:, 4] / nn / ONE
        var_q = np.maximum(s[:, 5] / nn / ONE - msq**2, 0.0) * nn / (nn - 1)
        sd, bi = self._decide(n, ab, msq, var_q, msq)
        return sd | bi, sd, bi

    def extendable(self, sq, sm):
        """bool [k]: whether a conjunction extending a group could still be a candidate, from the
        group's exact sums ``sq`` of q and ``sm`` of m (the largest |e_t| of a row) over its rows.

        The bound (the optimistic estimate of exhaustive subgroup discovery with a minimum support:
        Grosskreutz, Rüping, Wrobel 2008; Lemmerich, Atzmueller, Puppe 2016): any subset C of the
        group P with at least ``minsup`` rows has mean square sum_C q / n_C <= sum_P q / minsup
        (q >= 0), and every team's |bias| = |sum_C e_t| / n_C <= sum_C m / n_C <= sum_P m / minsup.
        The candidate rule needs mean square > alpha_sd^2 or |bias| > alpha_bias, so when both
        bounds are at or below (with a relative margin of 1e-9, far above the float64 rounding of
        the rule's own arithmetic) no extension can be a candidate and P is not extended. The
        bound is valid, never a heuristic: the candidates are the same with and without it."""
        sq = np.asarray(sq, dtype=np.float64)
        sm = np.asarray(sm, dtype=np.float64)
        keep = 1.0 - 1e-9
        return (sq / self.minsup / ONE > self.alpha_sd**2 * keep) | (sm / self.minsup / ONE > self.alpha_bias * keep)


# ---------------------------------------------------------------------------------------------
# Half split.


def half_b_range(seed, start, stop):
    """bool [stop - start]: rows start..stop-1 of ``half_b``."""
    i = np.arange(start + 1, stop + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed & 0xFFFFFFFFFFFFFFFF) + i * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(63)).astype(bool)


def half_b(seed, n):
    """bool [n]: row i is in half B when the top bit of the i-th output of SplitMix64(seed) is 1."""
    return half_b_range(seed, 0, n)


# ---------------------------------------------------------------------------------------------
# Whole-number columns of a row's error, as the passes store them.

NCOL = 12  # 1, e (4), e2 (4), q, qq, m
COL_M = 11  # m = max over the teams of |e_t|: the bound on every team's |bias| (``Screen.extendable``)
# The enumeration's sums: count, e0, e1, e2, q, qq (``SCREEN_COLS``), then m.
ENUM_W = (1, 2, 3, 9, 10, COL_M)


def error_columns(e):
    """int64 [n, NCOL] of the errors ``e`` (int64 [n, 4], |e| <= 2^32): ``Errors.columns`` and m."""
    err = Errors.of(e)
    return np.column_stack([err.columns(), np.abs(err.e).max(axis=1) if len(err) else np.zeros(0, np.int64)])


# ---------------------------------------------------------------------------------------------
# Parallel passes (the auditor at scale).
#
# Every pass splits the rows (or the work) into tasks of a fixed size that does not depend on the
# number of workers; every sum a task returns is a whole number (float64 bincounts and products
# are taken over at most 2^16 rows of values below 2^34, so every partial sum is below 2^50 and
# exact; they are then added as int64), so the totals, and everything decided from them, are the
# same for any number of workers. Shared arrays are ``.npy`` files in a work directory, opened
# memory-mapped by each worker.

BLOCK = 1 << 16  # rows of a block (a multiple of 8, so packed bitsets of blocks align)
SUB = 1 << 14  # rows of one bincount or product
KEYS = 1 << 22  # pair keys made at once
L3_ROWS = 1 << 19  # rows of one triple task
HIST_ROWS = 1 << 18  # rows of one histogram task
P999_ROWS = 1 << 21  # p99.9 of a group's |error| over at most this many half-B rows (the first)
KI = 96 + 9  # items per row at most: the active features, and the yes/no features' "off"
_BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")

_CACHE = {}


def _open(path, write=False):
    """A work file, memory-mapped (cached per process)."""
    key = (path, write)
    a = _CACHE.get(key)
    if a is None:
        a = np.load(path, mmap_mode="r+" if write else "r", allow_pickle=False)
        _CACHE[key] = a
    return a


def _json(path):
    key = ("json", path)
    a = _CACHE.get(key)
    if a is None:
        with open(path) as fh:
            a = json.load(fh)
        _CACHE[key] = a
    return a


def _new(path, shape, dtype, fill=None):
    """A new work file of at least one element per axis."""
    shape = tuple(max(int(x), 1) for x in shape)
    a = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    if fill is not None:
        a[...] = fill
    a.flush()
    del a
    return path


def _save(path, arr):
    np.save(path, np.asarray(arr), allow_pickle=False)
    return path


def _call(job):
    fn, i, task = job
    return i, fn(task)


class Workers:
    """``workers`` processes (forkserver start, one BLAS thread each), or this process for 1.
    ``map`` returns the results in task order whatever order they finish in."""

    def __init__(self, workers):
        self.n = max(1, int(workers))
        self.pool = None
        if self.n > 1:
            import multiprocessing as mp

            ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
            old = {k: os.environ.get(k) for k in _BLAS_ENV}
            os.environ.update({k: "1" for k in _BLAS_ENV})
            try:
                self.pool = ctx.Pool(self.n)
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    def map(self, fn, tasks, cost=None):
        """``fn`` of every task, in task order; ``cost``: run the dearest first (balance only)."""
        tasks = list(tasks)
        if self.pool is None or len(tasks) <= 1:
            return [fn(t) for t in tasks]
        order = range(len(tasks)) if cost is None else sorted(range(len(tasks)), key=lambda i: (-cost[i], i))
        out = [None] * len(tasks)
        for i, r in self.pool.imap_unordered(_call, [(fn, i, tasks[i]) for i in order]):
            out[i] = r
        return out

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None


def _blocks(n, size=BLOCK):
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def _bool_matrix(active):
    """bool [n, N_BINARY] from active lists (any width, PAD-padded)."""
    active = np.asarray(active)
    out = np.zeros((len(active), N_BINARY), dtype=bool)
    r, c = np.nonzero(active != PAD)
    out[r, active[r, c].astype(np.int64)] = True
    return out


_NETS = {}


def _exact_net(path):
    if path not in _NETS:
        layers, meta = intnet.load(path)
        _NETS[path] = (ExactNet(layers), meta.get("head", "money4"))
    return _NETS[path]


def _records(ctx, s, t):
    """Rows s..t-1 of the shards, one after another."""
    parts = []
    for path, lo, hi in ctx["shards"]:
        a, b = max(s, lo), min(t, hi)
        if a < b:
            parts.append(np.asarray(load_shard(path)[a - lo : b - lo]))
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _feats(ctx, s, t):
    """(active lists, dense values) of rows s..t-1: from the shards, or the work copies."""
    if ctx.get("shards"):
        r = _records(ctx, s, t)
        return np.asarray(r["active"]), np.asarray(r["dense"])
    return np.asarray(_open(ctx["act"])[s:t]), np.asarray(_open(ctx["dense"])[s:t])


def _write_block(ctx, s, t, a0, cols, races, active, dense):
    """Stores a block's columns, and its half-A rows' features; returns its sums."""
    _open(ctx["cols"], True)[s:t] = cols
    _open(ctx["races"], True)[s:t] = races
    isb = np.asarray(_open(ctx["isb"])[s:t])
    a = ~isb
    na = int(a.sum())
    act = np.asarray(active)[a]
    if na:
        _open(ctx["actA"], True)[a0 : a0 + na, : act.shape[1]] = act
        _open(ctx["colsA"], True)[a0 : a0 + na] = cols[a]
        _open(ctx["denseA"], True)[:, a0 : a0 + na] = np.asarray(dense)[a].T
    flat = act[act != PAD].astype(np.int64)
    race_sums = np.zeros((len(RACE_CLASS_NAMES), NCOL), dtype=np.int64)
    for k in range(len(RACE_CLASS_NAMES)):
        race_sums[k] = cols[races == k].sum(axis=0)
    return {"on": np.bincount(flat, minlength=N_BINARY), "race_sums": race_sums, "rows_a": na}


def _task_prep(args):
    """A block of the shards: checks, the integer network, the error columns, race classes."""
    ctx, s, t, a0 = args
    r = _records(ctx, s, t)
    tricks = sorted(int(x) for x in np.unique(r["trick"]))
    fault = value_fault(r)
    if fault is not None:
        return {"fault": fault, "tricks": tricks}
    net, head = _exact_net(ctx["net"])
    pred = []
    for i in range(0, len(r), ctx["chunk"]):
        b = r[i : i + ctx["chunk"]]
        z = net.logits(b["active"], b["dense"])
        pred.append(money4(z) if head == "money4" else money_outcomes(z, b["mask"], b["outcomes"]))
    e = np.concatenate(pred).astype(np.int64) - np.asarray(r["value"]).astype(np.int64)
    out = _write_block(ctx, s, t, a0, error_columns(e), race_classes(r).astype(np.int8), r["active"], r["dense"])
    out.update({"fault": None, "tricks": tricks})
    return out


def column_thresholds(x):
    """``dense_thresholds`` of one column."""
    x = np.sort(np.asarray(x, dtype=np.int64))
    if not len(x):
        return np.zeros(0, dtype=np.int64)
    idx = (np.arange(1, N_BINS) * len(x)) // N_BINS
    t = np.unique(x[np.minimum(idx, len(x) - 1)])
    return t[t < x[-1]]


def _task_thresholds(args):
    ctx, j = args
    return column_thresholds(_open(ctx["denseA"])[j, : ctx["n_a"]])


def _task_items(args):
    """A block of half A: the dense bins, the rows' items (sorted, ``F`` after the last), the
    items' packed bitsets and their exact sums (count, then ``ENUM_W``)."""
    ctx, s, t = args
    thr, lens = _open(ctx["thr"]), _open(ctx["thr_len"])
    dT = _open(ctx["denseA"])
    bins = np.empty((t - s, N_DENSE), dtype=np.uint8)
    for j in range(N_DENSE):
        bins[:, j] = np.searchsorted(thr[j, : lens[j]], np.asarray(dT[j, s:t], dtype=np.int64), side="left")
    _open(ctx["binsA"], True)[s:t] = bins
    feat, on = _open(ctx["item_feature"]), _open(ctx["item_on"])
    f = int(ctx["n_items"])
    xb = _bool_matrix(_open(ctx["actA"])[s:t])
    x = xb[:, feat[:f]] == on[:f][None, :] if f else np.zeros((t - s, 0), dtype=bool)
    if f:
        _open(ctx["ibits"], True)[:f, s // 8 : s // 8 + (t - s + 7) // 8] = np.packbits(x, axis=0).T
    r, c = np.nonzero(x)
    pos = np.arange(len(r)) - np.searchsorted(r, r, side="left")
    if len(pos) and int(pos.max()) >= KI:
        raise ValueError(f"a row with more than {KI} items")
    items = np.full((t - s, KI), f, dtype=np.int16)
    items[r, pos] = c
    _open(ctx["itemsA"], True)[s:t] = items
    w = np.asarray(_open(ctx["colsA"])[s:t])[:, ENUM_W]
    out = np.zeros((max(f, 1), 7), dtype=np.int64)
    for i in range(0, t - s, SUB):
        xs = x[i : i + SUB].astype(np.float64)
        out[:f, 0] += np.rint(xs.sum(axis=0)).astype(np.int64)
        out[:f, 1:] += np.rint(xs.T @ w[i : i + SUB].astype(np.float64)).astype(np.int64)
    return out[:f]


def _task_pairs(args):
    """A range of half-A rows: exact sums (count, then ``ENUM_W``) of every pair of grow items
    (positions a < b, key a * G + b) over the rows holding both."""
    ctx, s, t = args
    g = int(ctx["n_grow"])
    lut = _open(ctx["pos_lut"])
    items, colsA = _open(ctx["itemsA"]), _open(ctx["colsA"])
    acc = np.zeros((7, g * g), dtype=np.int64)
    for i in range(s, t, SUB):
        j = min(i + SUB, t)
        gp = lut[np.asarray(items[i:j]).astype(np.int64)]
        gp.sort(axis=1)
        k = int((gp < g).sum(axis=1).max()) if len(gp) else 0
        if k < 2:
            continue
        gp = gp[:, :k].astype(np.int64)
        w = np.asarray(colsA[i:j])[:, ENUM_W].astype(np.float64)
        iu, ju = np.triu_indices(k, 1)
        step = max(1, KEYS // len(iu))
        for r0 in range(0, len(gp), step):
            blk = gp[r0 : r0 + step]
            hi = blk[:, ju]
            ri, pi = np.nonzero(hi < g)
            key = blk[ri, iu[pi]] * g + hi[ri, pi]
            acc[0] += np.bincount(key, minlength=g * g)
            ww = w[r0 + ri]
            for c in range(6):
                acc[1 + c] += np.rint(np.bincount(key, weights=ww[:, c], minlength=g * g)).astype(np.int64)
    return acc


def _task_triples(args):
    """Rows lo..hi-1 of grow item ``a``'s rows (half A): exact sums (count, e0, e1, e2, q, qq) of
    every triple (a, b, c), b < c among the items extending ``a`` whose pair is extendable too, in
    the order of ``np.flatnonzero`` of that pair table."""
    ctx, a, lo, hi = args
    g = int(ctx["n_grow"])
    grow2 = _open(ctx["grow2"])
    ba = np.asarray(grow2[a, :g]).copy()
    ba[: a + 1] = False
    nb = int(ba.sum())
    sub = np.asarray(grow2[:g, :g])[np.ix_(ba, ba)]
    want = np.flatnonzero(np.triu(sub, 1).reshape(-1))
    if nb < 2 or not len(want):
        return np.zeros((6, 0), dtype=np.int64)
    rank = np.full(g + 1, nb, dtype=np.int64)
    rank[np.flatnonzero(ba)] = np.arange(nb)
    lut = _open(ctx["pos_lut"])
    item = int(_open(ctx["item_of_pos"])[a])
    rows = np.flatnonzero(np.unpackbits(np.asarray(_open(ctx["ibits"])[item]), count=int(ctx["n_a"])))[lo:hi]
    items, colsA = _open(ctx["itemsA"]), _open(ctx["colsA"])
    acc = np.zeros((6, nb * nb), dtype=np.int64)
    for i in range(0, len(rows), SUB):
        r = rows[i : i + SUB]
        loc = rank[lut[np.asarray(items[r]).astype(np.int64)]]
        loc.sort(axis=1)
        k = int((loc < nb).sum(axis=1).max()) if len(loc) else 0
        if k < 2:
            continue
        loc = loc[:, :k]
        w = np.asarray(colsA[r])[:, ENUM_W[:5]].astype(np.float64)
        iu, ju = np.triu_indices(k, 1)
        step = max(1, KEYS // len(iu))
        for r0 in range(0, len(loc), step):
            blk = loc[r0 : r0 + step]
            hi_ = blk[:, ju]
            ri, pi = np.nonzero(hi_ < nb)
            b, c = blk[ri, iu[pi]], hi_[ri, pi]
            ok = sub[b, c]
            ri, b, c = ri[ok], b[ok], c[ok]
            key = b * nb + c
            acc[0] += np.bincount(key, minlength=nb * nb)
            ww = w[r0 + ri]
            for col in range(5):
                acc[1 + col] += np.rint(np.bincount(key, weights=ww[:, col], minlength=nb * nb)).astype(np.int64)
    return acc[:, want]


def _task_hist(args):
    """A node's rows (half-A positions): exact histograms of the tree's targets (``kind``
    "signed": the four teams' errors; "abs": the sum of their absolute values) per binary
    feature and per dense feature and bin, the targets' totals and the screening sums."""
    ctx, rows, kind = args
    act, bins, colsA = _open(ctx["actA"]), _open(ctx["binsA"]), _open(ctx["colsA"])
    d = 4 if kind == "signed" else 1
    nb = N_BINS
    n_on = np.zeros(N_BINARY, dtype=np.int64)
    s_on = np.zeros((N_BINARY, d), dtype=np.int64)
    cnt = np.zeros(N_DENSE * nb, dtype=np.int64)
    sums = np.zeros((N_DENSE * nb, d), dtype=np.int64)
    total = np.zeros(d, dtype=np.int64)
    scr = np.zeros(len(SCREEN_COLS), dtype=np.int64)
    for i in range(0, len(rows), SUB):
        r = rows[i : i + SUB]
        a = np.asarray(act[r])
        c = np.asarray(colsA[r])
        y = c[:, 1:5] if d == 4 else np.abs(c[:, 1:5]).sum(axis=1, keepdims=True)
        ri, ci = np.nonzero(a != PAD)
        flat = a[ri, ci].astype(np.int64)
        n_on += np.bincount(flat, minlength=N_BINARY)
        b = (np.arange(N_DENSE)[None, :] * nb + np.asarray(bins[r]).astype(np.int64)).reshape(-1)
        cnt += np.bincount(b, minlength=N_DENSE * nb)
        for k in range(d):
            yk = y[:, k].astype(np.float64)
            s_on[:, k] += np.rint(np.bincount(flat, weights=yk[ri], minlength=N_BINARY)).astype(np.int64)
            sums[:, k] += np.rint(np.bincount(b, weights=np.repeat(yk, N_DENSE), minlength=N_DENSE * nb)).astype(np.int64)
        total += y.sum(axis=0)
        scr += c[:, list(SCREEN_COLS)].sum(axis=0)
    return [n_on, s_on, cnt.reshape(N_DENSE, nb), sums.reshape(N_DENSE, nb, d), total, len(rows), scr]


CAND_CHUNK = 512  # groups whose members are held at once in a confirmation block


def _task_confirm(args):
    """A block of all rows: each group's exact column sums over its rows in half A and in half B;
    with ``bits`` (a work file), its members are written there as packed bits instead. Groups are
    taken ``CAND_CHUNK`` at a time, so a worker's memory does not grow with their number."""
    ctx, s, t, groups_path, bits = args
    groups = _json(groups_path)
    active, dense = _feats(ctx, s, t)
    xb = _bool_matrix(active)
    dense = np.asarray(dense, dtype=np.int64)
    cols = np.asarray(_open(ctx["cols"])[s:t]).astype(np.float64)
    isb = np.asarray(_open(ctx["isb"])[s:t])
    k = len(groups)
    out = np.zeros((k, 2, NCOL), dtype=np.int64)
    halves = [(i, sel) for i in range(0, t - s, SUB) for sel in (~isb[i : i + SUB], isb[i : i + SUB])]
    for g0 in range(0, k, CAND_CHUNK):
        g1 = min(k, g0 + CAND_CHUNK)
        m = np.empty((t - s, g1 - g0), dtype=bool)
        for g in range(g0, g1):
            m[:, g - g0] = members(groups[g], xb, dense)
        if bits is not None:
            _open(bits, True)[g0:g1, s // 8 : s // 8 + (t - s + 7) // 8] = np.packbits(m, axis=0).T
            continue
        for h, (i, sel) in enumerate(halves):
            if sel.any():
                mi = m[i : i + SUB][sel].T.astype(np.float64)
                out[g0:g1, h % 2] += np.rint(mi @ cols[i : i + SUB][sel]).astype(np.int64)
    return None if bits is not None else out


def _task_detail(args):
    """A confirmed group (row ``g`` of the bits file): its race-class mix, its example row (the
    half-B row of largest mean square, the first such) and the 99.9th percentile of |error| over
    its half-B rows (the first ``P999_ROWS`` of them in row order)."""
    ctx, bits, g = args
    n = int(ctx["n"])
    rows = np.flatnonzero(np.unpackbits(np.asarray(_open(bits)[g]), count=n))
    races = np.bincount(np.asarray(_open(ctx["races"])[rows]).astype(np.int64), minlength=len(RACE_CLASS_NAMES))
    rows_b = rows[np.asarray(_open(ctx["isb"])[rows])]
    out = {"races": [int(x) for x in races], "example_row": None, "p999": None}
    if len(rows_b):
        cols = _open(ctx["cols"])
        q = np.asarray(cols[rows_b, 9])
        out["example_row"] = int(rows_b[int(np.argmax(q))])
        eb = np.abs(np.asarray(cols[rows_b[:P999_ROWS], 1:5])).astype(np.float64) / ONE
        out["p999"] = float(np.quantile(eb, 0.999))
    return out


# ---------------------------------------------------------------------------------------------
# Regression trees.

N_BINS = 64  # a dense feature splits at up to N_BINS - 1 thresholds


def dense_thresholds(dense):
    """Per dense feature: the sorted distinct values of its quantiles k / N_BINS (k = 1..63) on
    the given rows, each a value of the data; a split is "x <= threshold"."""
    return [column_thresholds(np.asarray(dense)[:, j]) for j in range(np.asarray(dense).shape[1])]


def dense_bins(dense, thresholds):
    """uint8 [n, N_DENSE]: the number of thresholds strictly below the value, so that
    ``x <= t_k`` exactly when the bin is at most ``k``."""
    out = np.zeros(dense.shape, dtype=np.uint8)
    for j, t in enumerate(thresholds):
        out[:, j] = np.searchsorted(t, np.asarray(dense[:, j], dtype=np.int64), side="left")
    return out


def fit_tree(d, pool, kind, min_leaf, max_depth):
    """A CART regression tree on half A (squared-error impurity summed over the targets: the
    four teams' errors for ``kind`` "signed", the sum of their absolute values for "abs"); every
    histogram exact (``_task_hist``, in tasks of ``HIST_ROWS`` rows), so the splits are the same
    for any number of workers. A binary feature splits on/off, a dense feature at one of up to 63
    thresholds (quantiles of half A); the best split by squared-error reduction, ties to the lower
    feature and threshold. Returns the nodes below the root: (path conditions, half-A positions,
    depth, screening sums). A node's histograms are counted for the smaller child of each split
    and taken from the parent for the other (histogram subtraction, exact here)."""
    ctx, thresholds = d.ctx, d.thresholds
    nb = N_BINS

    def hist(rows):
        parts = pool.map(_task_hist, [(ctx, rows[i : i + HIST_ROWS], kind) for i in range(0, len(rows), HIST_ROWS)])
        if not parts:
            parts = [_task_hist((ctx, rows, kind))]
        return [sum(p[k] for p in parts) for k in range(7)]

    exists = np.arange(nb - 1)[None, :] < np.array([len(t) for t in thresholds])[:, None]
    bits_of = {f: i for i, (f, v) in enumerate(d.items) if v == 1}
    out = []
    queue = [([], np.arange(d.n_a, dtype=np.int64), 0, None)]
    while queue:
        conds, rows, depth, h = queue.pop(0)
        if h is None:
            h = hist(rows)
        n_on, s_on, cnt, sums, total, k, scr = h
        if depth > 0:
            out.append((conds, rows, depth, scr))
        if depth >= max_depth or k < 2 * min_leaf:
            continue
        totf = total.astype(np.float64)
        base = float((totf**2).sum()) / k
        n_onf = n_on.astype(np.float64)
        s_onf = s_on.astype(np.float64)
        n_off = k - n_onf
        s_off = totf[None, :] - s_onf
        okb = (n_onf >= min_leaf) & (n_off >= min_leaf)
        with np.errstate(divide="ignore", invalid="ignore"):
            gb = (s_onf**2).sum(axis=1) / n_onf + (s_off**2).sum(axis=1) / n_off - base
        gb = np.where(okb, gb, -np.inf)
        n_le = np.cumsum(cnt, axis=1)[:, : nb - 1].astype(np.float64)
        s_le = np.cumsum(sums, axis=1)[:, : nb - 1, :].astype(np.float64)
        n_gt = k - n_le
        s_gt = totf[None, None, :] - s_le
        okd = exists & (n_le >= min_leaf) & (n_gt >= min_leaf)
        with np.errstate(divide="ignore", invalid="ignore"):
            gd = (s_le**2).sum(axis=2) / n_le + (s_gt**2).sum(axis=2) / n_gt - base
        gd = np.where(okd, gd, -np.inf)
        gains = np.concatenate([gb, gd.reshape(-1)])
        best = int(np.argmax(gains))
        if not gains[best] > 0:
            continue
        if best < N_BINARY:
            f = best
            on = np.unpackbits(np.asarray(_open(ctx["ibits"])[bits_of[f]]), count=d.n_a).astype(bool)
            left_mask = on[rows]
            cl = [{"feature": f, "op": "==", "value": 1}]
            cr = [{"feature": f, "op": "==", "value": 0}]
        else:
            j, b = divmod(best - N_BINARY, nb - 1)
            t = int(thresholds[j][b])
            left_mask = np.asarray(_open(ctx["binsA"])[rows, j]) <= b
            cl = [{"feature": N_BINARY + j, "op": "<=", "value": t}]
            cr = [{"feature": N_BINARY + j, "op": ">", "value": t}]
        lrows, rrows = rows[left_mask], rows[~left_mask]
        if depth + 1 < max_depth:
            small_left = len(lrows) <= len(rrows)
            hs = hist(lrows if small_left else rrows)
            hl = [p - c for p, c in zip(h, hs)]
            hleft, hright = (hs, hl) if small_left else (hl, hs)
        else:
            hleft = hright = None
        queue.append((conds + cl, lrows, depth + 1, hleft))
        queue.append((conds + cr, rrows, depth + 1, hright))
    return out


# ---------------------------------------------------------------------------------------------
# The conjunction enumeration.


def enumeration_items(on, n_rows, minsup):
    """The items: (feature, 1) for every binary feature on in at least ``minsup`` of ``n_rows``
    rows (``on``: per feature, the rows it is on in), (feature, 0) for the yes/no features off in
    at least ``minsup`` rows; in feature order, "on" before "off"."""
    items = []
    for f in range(N_BINARY):
        if on[f] >= minsup:
            items.append((f, 1))
        if f in YES_NO and n_rows - on[f] >= minsup:
            items.append((f, 0))
    return items


class Enumeration:
    """Every conjunction of up to ``max_conj`` items with support >= ``minsup`` on half A, with
    its candidate decision; candidates are not extended (see the module notes), nor are items on
    in every row, nor groups whose sums rule out every extension (``Screen.extendable``, a valid
    bound). Every sum is exact: single items and pairs by row blocks (``_task_items``,
    ``_task_pairs``), triples per item over its rows (``_task_triples``, an item's rows split into
    tasks of ``L3_ROWS``). A triple (i, j, k), positions in increasing support (ties by item
    index), is summed over item i's rows, among the items j, k whose pairs with i, and with each
    other, are extendable. ``sums1``: the items' exact sums (count, then ``ENUM_W``)."""

    def __init__(self, d, pool, screen, max_conj, sums1):
        if not 1 <= max_conj <= 3:
            raise ValueError("--max-conj is 1, 2 or 3")
        self.screen = screen
        self.counted = [0, 0, 0]
        self.frequent = [0, 0, 0]
        self.pruned = [0, 0]  # groups not extended because the bound rules out every extension
        self.found = []  # (item tuple, sd flag, bias flag)
        f = len(d.items)
        s1 = np.asarray(sums1, dtype=np.int64).reshape(f, 7)
        cand, sdf, bif = screen(s1[:, :6])
        freq1 = s1[:, 0] >= screen.minsup
        self.counted[0], self.frequent[0] = f, int(freq1.sum())
        self._keep(np.arange(f)[:, None], cand, sdf, bif)
        ext = screen.extendable(s1[:, 4], s1[:, 6])
        grow = np.flatnonzero(freq1 & ~cand & (s1[:, 0] < d.n_a) & ext)
        self.pruned[0] = int((freq1 & ~cand & (s1[:, 0] < d.n_a) & ~ext).sum())
        idx = grow[np.lexsort((grow, s1[grow, 0]))]
        g = len(idx)
        if max_conj < 2 or g < 2:
            return
        ctx = dict(d.ctx, n_grow=g)
        pos_lut = np.full(f + 1, g, dtype=np.int16)
        pos_lut[idx] = np.arange(g)
        ctx["pos_lut"] = _save(os.path.join(d.work, "pos_lut.npy"), pos_lut)
        ctx["item_of_pos"] = _save(os.path.join(d.work, "item_of_pos.npy"), idx)
        tasks = [(ctx, s, t) for s, t in _blocks(d.n_a, max(BLOCK, -(-d.n_a // (4 * pool.n))))]
        s2 = sum(pool.map(_task_pairs, tasks)).reshape(7, g, g)
        iu, ju = np.triu_indices(g, 1)
        p2 = s2[:, iu, ju].T
        fr2 = p2[:, 0] >= screen.minsup
        c2, d2, b2 = screen(p2[:, :6])
        c2, d2, b2 = c2 & fr2, d2 & fr2, b2 & fr2
        self.counted[1] += len(p2)
        self.frequent[1] += int(fr2.sum())
        self._keep(np.column_stack([idx[iu], idx[ju]]), c2, d2, b2)
        ext2 = screen.extendable(p2[:, 4], p2[:, 6])
        self.pruned[1] = int((fr2 & ~c2 & ~ext2).sum())
        grow2 = np.zeros((g, g), dtype=bool)
        grow2[iu, ju] = fr2 & ~c2 & ext2
        if max_conj < 3:
            return
        ctx["grow2"] = _save(os.path.join(d.work, "grow2.npy"), grow2)
        support = s1[idx, 0]
        jobs, cost = [], []
        for a in range(g):
            ba = grow2[a].copy()
            nb = int(ba.sum())
            if nb < 2 or not np.triu(grow2[np.ix_(ba, ba)], 1).any():
                continue
            for lo in range(0, int(support[a]), L3_ROWS):
                jobs.append((ctx, a, lo, min(lo + L3_ROWS, int(support[a]))))
                cost.append(min(L3_ROWS, int(support[a]) - lo) * nb)
        res = pool.map(_task_triples, jobs, cost=cost)
        per_a = {}
        for (_, a, _, _), r in zip(jobs, res):
            per_a[a] = per_a[a] + r if a in per_a else r
        for a in sorted(per_a):
            ba = grow2[a].copy()
            js = np.flatnonzero(ba)
            sub = np.triu(grow2[np.ix_(ba, ba)], 1)
            bi, ci = np.nonzero(sub)
            s3 = per_a[a].T
            fr3 = s3[:, 0] >= screen.minsup
            c3, d3, b3 = screen(s3)
            self.counted[2] += len(s3)
            self.frequent[2] += int(fr3.sum())
            tup = np.column_stack([np.full(len(bi), idx[a]), idx[js[bi]], idx[js[ci]]])
            self._keep(tup, c3 & fr3, d3 & fr3, b3 & fr3)

    def _keep(self, tuples, cand, sdf, bif):
        for k in np.flatnonzero(cand):
            self.found.append((tuple(int(v) for v in tuples[k]), bool(sdf[k]), bool(bif[k])))


# ---------------------------------------------------------------------------------------------
# The audit.


class AuditData:
    """The work files of one audit (in ``work``) and what the parent keeps: rows, half-A rows,
    the half split, per-feature supports on half A, race-class sums, the items, the thresholds."""

    def __init__(self, work, n, n_a, ctx):
        self.work, self.n, self.n_a, self.ctx = work, n, n_a, ctx
        self.on = np.zeros(N_BINARY, dtype=np.int64)
        self.race_sums = np.zeros((len(RACE_CLASS_NAMES), NCOL), dtype=np.int64)
        self.items = []
        self.thresholds = []

    @classmethod
    def create(cls, work, is_b, k_active, shards=None, net=None, chunk=None):
        os.makedirs(work, exist_ok=True)
        n = len(is_b)
        n_a = int((~is_b).sum())
        p = lambda name: os.path.join(work, name)  # noqa: E731
        ctx = {"n": n, "n_a": n_a, "work": work}
        ctx["isb"] = _save(p("isb.npy"), is_b)
        ctx["cols"] = _new(p("cols.npy"), (n, NCOL), np.int64)
        ctx["races"] = _new(p("races.npy"), (n,), np.int8)
        ctx["actA"] = _new(p("actA.npy"), (n_a, k_active), np.uint16)
        ctx["colsA"] = _new(p("colsA.npy"), (n_a, NCOL), np.int64)
        ctx["denseA"] = _new(p("denseA.npy"), (N_DENSE, n_a), np.int32)
        if shards:
            ctx["shards"] = shards
            ctx["net"] = net
            ctx["chunk"] = chunk
        return cls(work, n, n_a, ctx)

    def add(self, r):
        self.on += r["on"]
        self.race_sums += r["race_sums"]

    def a_offsets(self, blocks):
        isb = _open(self.ctx["isb"])
        out, a0 = [], 0
        for s, t in blocks:
            out.append(a0)
            a0 += int((~np.asarray(isb[s:t])).sum())
        return out


def prepare_memory(work, err, active, dense, is_b, races):
    """The work files of an audit of rows held in memory (``err``: ``Errors``; ``active``: active
    lists [n, K] PAD-padded, or a bool matrix [n, N_BINARY]; ``dense`` [n, N_DENSE])."""
    active = np.asarray(active)
    if active.dtype == bool:
        r, c = np.nonzero(active)
        k = int(np.bincount(r, minlength=len(active)).max()) if len(r) else 1
        lists = np.full((len(active), max(k, 1)), PAD, dtype=np.uint16)
        lists[r, np.arange(len(r)) - np.searchsorted(r, r, side="left")] = c
        active = lists
    d = AuditData.create(work, np.asarray(is_b, dtype=bool), active.shape[1])
    d.ctx["act"] = _save(os.path.join(work, "act.npy"), active)
    d.ctx["dense"] = _new(os.path.join(work, "dense.npy"), (d.n, N_DENSE), np.int32)
    dn = _open(d.ctx["dense"], True)
    races = np.asarray(races).astype(np.int8)
    blocks = _blocks(d.n)
    for (s, t), a0 in zip(blocks, d.a_offsets(blocks)):
        dn[s:t] = np.asarray(dense[s:t])
        cols = error_columns(err.e[s:t])
        d.add(_write_block(d.ctx, s, t, a0, cols, races[s:t], active[s:t], dense[s:t]))
    return d


def prepare_shards(work, paths, shards, net, seed, pool, chunk, log=print):
    """The work files of an audit of label shards (the network's errors computed in the pass);
    returns (data, frontier trick)."""
    n = sum(len(s) for s in shards)
    spans, lo = [], 0
    for p, s in zip(paths, shards):
        spans.append((p, lo, lo + len(s)))
        lo += len(s)
    d = AuditData.create(work, half_b(seed, n), MAX_ACTIVE, shards=spans, net=net, chunk=chunk)
    blocks = _blocks(n)
    res = pool.map(_task_prep, [(d.ctx, s, t, a0) for (s, t), a0 in zip(blocks, d.a_offsets(blocks))])
    tricks = set()
    for (s, _), r in zip(blocks, res):
        tricks |= set(r["tricks"])
        if r["fault"] is not None:
            p = next(p for p, a, b in spans if a <= s < b)
            raise SystemExit(f"{p}: {VALUE_FAULTS[r['fault']]}")
        d.add(r)
    if len(tricks) != 1:
        raise SystemExit(f"the shards hold tricks {sorted(tricks)}: one frontier trick per audit")
    return d, tricks.pop()


def prepare_items(d, pool, minsup):
    """The items of half A (``enumeration_items``), the dense thresholds and bins, the rows'
    items, the items' bitsets; returns the items' exact sums (count, then ``ENUM_W``)."""
    n_a, ctx, work = d.n_a, d.ctx, d.work
    d.items = enumeration_items(d.on, n_a, minsup)
    f = len(d.items)
    ctx["n_items"] = f
    ctx["item_feature"] = _save(os.path.join(work, "item_feature.npy"), [x for x, _ in d.items] or [0])
    ctx["item_on"] = _save(os.path.join(work, "item_on.npy"), [v == 1 for _, v in d.items] or [True])
    ctx["ibits"] = _new(os.path.join(work, "ibits.npy"), (f, (n_a + 7) // 8), np.uint8)
    ctx["itemsA"] = _new(os.path.join(work, "itemsA.npy"), (n_a, KI), np.int16)
    ctx["binsA"] = _new(os.path.join(work, "binsA.npy"), (n_a, N_DENSE), np.uint8)
    d.thresholds = pool.map(_task_thresholds, [(ctx, j) for j in range(N_DENSE)])
    thr = np.zeros((N_DENSE, N_BINS - 1), dtype=np.int64)
    for j, t in enumerate(d.thresholds):
        thr[j, : len(t)] = t
    ctx["thr"] = _save(os.path.join(work, "thr.npy"), thr)
    ctx["thr_len"] = _save(os.path.join(work, "thr_len.npy"), [len(t) for t in d.thresholds])
    sums1 = pool.map(_task_items, [(ctx, s, t) for s, t in _blocks(n_a)])
    return sum(sums1) if sums1 else np.zeros((f, 7), dtype=np.int64)


def _stats_dict(s, total):
    return Stats(s[:11]).as_dict(total)


def audit_data(d, pool, gamma=0.001, alpha_bias=0.0025, alpha_sd=0.01, max_depth=6, max_conj=3, fdr=0.05,
               screen=0.01, log=print):
    """The discovery (half A) and confirmation (half B) of weak groups on prepared work files
    (``prepare_memory``, ``prepare_shards``). Returns a dict (the report's body; ``confirmed``
    and ``predicates``)."""
    t0 = time.time()
    timing = {}
    n, n_a, ctx, work = d.n, d.n_a, d.ctx, d.work
    minsup = max(2, math.ceil(gamma * n_a))
    scr = Screen(alpha_bias, alpha_sd, minsup, screen)
    sums1 = prepare_items(d, pool, minsup)
    f = len(d.items)
    timing["items"] = round(time.time() - t0, 3)

    # Trees on half A.
    candidates = {}  # condition key -> {conditions, sources, flags}
    tree_nodes = {}
    for name, kind in (("tree-signed", "signed"), ("tree-abs", "abs")):
        nodes = fit_tree(d, pool, kind, minsup, max_depth) if n_a else []
        tree_nodes[name] = len(nodes)
        if nodes:
            sums = np.array([s for _, _, _, s in nodes], dtype=np.int64)
            cand, sdf, bif = scr(sums)
            for k in np.flatnonzero(cand):
                conds = canonical(nodes[k][0])
                key = cond_key(conds)
                c = candidates.setdefault(key, {"conditions": conds, "sources": [], "sd": False, "bias": False})
                c["sources"].append(name)
                c["sd"] |= bool(sdf[k])
                c["bias"] |= bool(bif[k])
    timing["trees"] = round(time.time() - t0 - timing["items"], 3)
    log(f"audit: trees on half A ({n_a} rows): {tree_nodes} nodes, {len(candidates)} candidates so far "
        f"({timing['trees']:.1f} s)")

    # The enumeration on half A.
    t1 = time.time()
    en = Enumeration(d, pool, scr, max_conj, sums1)
    for tup, sdf, bif in en.found:
        conds = canonical([{"feature": d.items[i][0], "op": "==", "value": d.items[i][1]} for i in tup])
        key = cond_key(conds)
        c = candidates.setdefault(key, {"conditions": conds, "sources": [], "sd": False, "bias": False})
        c["sources"].append(f"conj-{len(tup)}")
        c["sd"] |= sdf
        c["bias"] |= bif
    timing["enumeration"] = round(time.time() - t1, 3)
    log(f"audit: enumeration: {f} items, conjunctions counted by size {en.counted}, frequent "
        f"{en.frequent}, not extended by the bound {en.pruned}, {len(en.found)} candidates "
        f"({timing['enumeration']:.1f} s)")

    # Confirmation on half B.
    t2 = time.time()
    keys = sorted(candidates)
    gpath = os.path.join(work, "candidates.json")
    with open(gpath, "w") as fh:
        json.dump([candidates[k]["conditions"] for k in keys], fh)
    sums = np.zeros((len(keys), 2, NCOL), dtype=np.int64)
    if keys:
        for r in pool.map(_task_confirm, [(ctx, s, t, gpath, None) for s, t in _blocks(n)]):
            sums += r
    hyp = []  # (candidate index, criterion, team, sign, p)
    for ci, key in enumerate(keys):
        c = candidates[key]
        sa, sb = Stats(sums[ci, 0, :11]), Stats(sums[ci, 1, :11])
        if c["sd"]:
            hyp.append((ci, "sd", None, None, sb.p_sd(alpha_sd)))
        if c["bias"]:
            team, sign = sa.worst_team()
            hyp.append((ci, "bias", team, sign, sb.p_bias(team, sign, alpha_bias)))
    rejected = benjamini_hochberg([h[4] for h in hyp], fdr)
    confirmed = {}
    for h, rej in zip(hyp, rejected):
        ci, crit, team, sign, p = h
        if rej:
            confirmed.setdefault(ci, []).append({"criterion": crit, "team": team, "sign": sign, "p": p})
    conf = sorted(confirmed)
    details = []
    bits = None
    if conf:
        cpath = os.path.join(work, "confirmed.json")
        with open(cpath, "w") as fh:
            json.dump([candidates[keys[ci]]["conditions"] for ci in conf], fh)
        bits = _new(os.path.join(work, "members.npy"), (len(conf), (n + 7) // 8), np.uint8)
        pool.map(_task_confirm, [(ctx, s, t, cpath, bits) for s, t in _blocks(n)])
        details = pool.map(_task_detail, [(ctx, bits, g) for g in range(len(conf))])
    timing["confirmation"] = round(time.time() - t2, 3)

    # Minimality: a confirmed group inside a more general confirmed group is covered by it.
    conf_keys = {keys[ci] for ci in conf}
    n_b = n - n_a
    records = []
    for g, ci in enumerate(conf):
        key = keys[ci]
        c = candidates[key]
        sa, sb = sums[ci, 0], sums[ci, 1]
        rec = {"name": group_name(c["conditions"]), "conditions": c["conditions"], "source": c["sources"]}
        rec["whole"] = _stats_dict(sa + sb, n)
        rec["half_a"] = _stats_dict(sa, n_a)
        rec["half_b"] = _stats_dict(sb, n_b)
        det = details[g]
        if det["p999"] is not None:
            rec["half_b"]["p999"] = det["p999"]
            rec["example_row"] = det["example_row"]
        rec["race_classes"] = det["races"]
        rec["confirmed_by"] = confirmed[ci]
        rec["p_values"] = {h[1]: h[4] for h in hyp if h[0] == ci}
        covered = next((h for h in sorted(conf_keys) if h != key and implies(key, h)), None)
        rec["covered_by"] = None if covered is None else group_name(
            [{"feature": f_, "op": op, "value": v} for f_, op, v in covered])
        sbs = Stats(sb[:11])
        rec["severity"] = max(sbs.sd / alpha_sd, max(abs(b) for b in sbs.bias) / alpha_bias)
        rec["_bits_row"] = g
        records.append(rec)
    records.sort(key=lambda r: (-r["severity"], -r["whole"]["rows"], r["name"]))
    predicates = []
    inside = np.zeros((n + 7) // 8, dtype=np.uint8)
    for r in records:
        g = r.pop("_bits_row")
        if r["covered_by"] is None:
            predicates.append({
                "name": r["name"], "conditions": r["conditions"], "weight": r["whole"]["weight"],
                "bias": r["half_b"]["bias"], "sd": r["half_b"]["sd"], "p999": r["half_b"].get("p999"),
                "rows": r["whole"]["rows"], "example_row": r.get("example_row"),
                "criteria": [x["criterion"] for x in r["confirmed_by"]], "source": r["source"],
            })
            inside |= np.asarray(_open(bits)[g])
    in_rows = np.unpackbits(inside, count=n).astype(bool)
    races = np.asarray(_open(ctx["races"]))[:n].astype(np.int64)
    coverage = []
    rows_by_class = np.bincount(races, minlength=len(RACE_CLASS_NAMES))
    in_by_class = np.bincount(races[in_rows], minlength=len(RACE_CLASS_NAMES))
    for k in range(len(RACE_CLASS_NAMES)):
        coverage.append({"class": k, "rows": int(rows_by_class[k]),
                         "share_inside_groups": float(in_by_class[k] / rows_by_class[k]) if rows_by_class[k] else None})
    timing["total"] = round(time.time() - t0, 3)
    n_tree = sum(1 for c in candidates.values() if any(s.startswith("tree") for s in c["sources"]))
    n_conj = sum(1 for c in candidates.values() if any(s.startswith("conj") for s in c["sources"]))
    return {
        "rows": n,
        "half_a_rows": int(n_a),
        "half_b_rows": int(n_b),
        "min_group_rows": int(minsup),
        "tree_nodes": tree_nodes,
        "items": f,
        "conjunctions_counted": en.counted,
        "conjunctions_frequent": en.frequent,
        "not_extended_by_bound": en.pruned,
        "candidates": len(candidates),
        "candidates_from_trees": n_tree,
        "candidates_from_conjunctions": n_conj,
        "hypotheses": len(hyp),
        "confirmed": len(records),
        "confirmed_general": len(predicates),
        "groups": records,
        "rows_inside_groups": int(in_rows.sum()),
        "race_class_coverage": coverage,
        "predicates": predicates,
        "seconds": timing,
    }


def overall(d):
    """The whole shard's numbers: SD, bias, p99.9 of |error|, and the same by race class."""
    tot = d.race_sums.sum(axis=0)
    s = Stats(tot[:11])
    cols = _open(d.ctx["cols"])
    e = np.abs(np.asarray(cols[: d.n, 1:5])).astype(np.float64) / ONE
    out = {"rows": int(d.n), "bias": s.bias, "sd": s.sd, "p999": float(np.quantile(e, 0.999)) if d.n else None}
    out["by_race_class"] = []
    for k, name in enumerate(RACE_CLASS_NAMES):
        rk = d.race_sums[k]
        sk = Stats(rk[:11])
        out["by_race_class"].append({"class": k, "name": name, "rows": int(rk[0]),
                                     "bias": sk.bias if rk[0] else None, "sd": sk.sd if rk[0] else None})
    return out


def _cleanup(work, keep):
    for k in [k for k in _CACHE if (k[0] == "json" and str(k[1]).startswith(work)) or str(k[0]).startswith(work)]:
        _CACHE.pop(k, None)
    if not keep:
        import shutil

        shutil.rmtree(work, ignore_errors=True)


def audit(err, binary, dense, is_b, races, gamma=0.001, alpha_bias=0.0025, alpha_sd=0.01, max_depth=6,
          max_conj=3, fdr=0.05, screen=0.01, workers=1, work=None, log=print):
    """``audit_data`` of rows held in memory (``err``: ``Errors`` of all rows; ``binary``: their
    active lists [n, K] PAD-padded, or the bool matrix of ``binary_matrix``; ``dense``: their
    dense values; ``is_b``: ``half_b``; ``races``: their race classes, a diagnostic only), on
    ``workers`` processes, with the work files in ``work`` (a new temporary directory, removed
    after, by default)."""
    import tempfile

    own = work is None
    work = tempfile.mkdtemp(prefix="audit-") if own else work
    try:
        d = prepare_memory(work, err, binary, dense, is_b, races)
        with Workers(workers) as pool:
            body = audit_data(d, pool, gamma=gamma, alpha_bias=alpha_bias, alpha_sd=alpha_sd, max_depth=max_depth,
                              max_conj=max_conj, fdr=fdr, screen=screen, log=log)
            pool.close()
        body["overall"] = overall(d)
        return body
    finally:
        _cleanup(work, keep=not own)


def _fmt_bias(b):
    return " / ".join(f"{100 * x:+.3f}%" for x in b)


def text_report(rep):
    th = rep["thresholds"]
    body = rep["audit"]
    lines = [
        f"Audit of the network {rep['net']['path']} (sha256 {rep['net']['sha256'][:16]}..., trick {rep['trick']}, "
        f"{rep['net']['head']} head).",
        f"Rows audited: {body['rows']:,} from {len(rep['shards'])} shard(s); half A (discovery) "
        f"{body['half_a_rows']:,}, half B (confirmation) {body['half_b_rows']:,}, split seed {rep['seed']}.",
        f"Integer network checked against the reference integer path on {rep['parity_rows']} rows: identical.",
        "Teams are named relative to the player who leads the position's trick: the leader's team / the next "
        "team / the opposite team / the previous team (the shard does not say which seat decided).",
        "",
        f"Overall: SD {100 * rep['overall']['sd']:.3f}%, bias {_fmt_bias(rep['overall']['bias'])}, "
        f"99.9th percentile of |error| {100 * rep['overall']['p999']:.2f}%.",
        "By race class (a diagnostic; the audit does not use race classes):",
    ]
    for c in rep["overall"]["by_race_class"]:
        if c["rows"]:
            lines.append(f"  {c['class']} ({c['name']}): {c['rows']:,} rows ({100 * c['rows'] / body['rows']:.1f}%), "
                         f"SD {100 * c['sd']:.3f}%, bias {_fmt_bias(c['bias'])}")
    lines += [
        "",
        f"Standard: no group of weight at least {100 * th['gamma']:g}% ({body['min_group_rows']} rows of half A) with "
        f"|bias| over {100 * th['alpha_bias']:g}% for some team or SD over {100 * th['alpha_sd']:g}%; "
        f"trees of depth up to {th['max_depth']}, conjunctions of up to {th['max_conj']} binary conditions; "
        f"candidates screened on half A at p <= {th['screen']:g}; Benjamini-Hochberg at {100 * th['fdr']:g}% "
        "on half B.",
        f"Searched on half A: tree nodes {body['tree_nodes']}; {body['items']} items, conjunctions counted by size "
        f"{body['conjunctions_counted']} (frequent {body['conjunctions_frequent']}; items and pairs not extended "
        f"because their sums rule out every extension: {body['not_extended_by_bound']}).",
        f"Candidates found on half A: {body['candidates']} ({body['candidates_from_trees']} from the trees, "
        f"{body['candidates_from_conjunctions']} from the conjunctions); hypotheses tested on half B: "
        f"{body['hypotheses']}.",
        f"Confirmed on half B: {body['confirmed']} group(s); {body['confirmed_general']} after leaving out groups "
        "inside a more general confirmed group (these go to predicates.json).",
        "",
    ]
    if body["confirmed"]:
        cov = ", ".join(f"{c['class']}: {100 * c['share_inside_groups']:.0f}%" for c in body["race_class_coverage"]
                        if c["rows"])
        lines.append(f"Positions inside the confirmed groups: {body['rows_inside_groups']:,} "
                     f"({100 * body['rows_inside_groups'] / body['rows']:.1f}%); by race class, the share of its "
                     f"positions inside them: {cov}.")
        lines.append("")
    if not body["confirmed"]:
        lines.append("Verdict: flat. No weak group was confirmed.")
    else:
        lines.append(f"Verdict: not flat. {body['confirmed_general']} weak group(s) remain. The worst:")
        for i, g in enumerate([g for g in body["groups"] if g["covered_by"] is None][:15], 1):
            b = g["half_b"]
            mix = g["race_classes"]
            tot = max(sum(mix), 1)
            by = ", ".join(f"{x['criterion']} (p {x['p']:.1e})" for x in g["confirmed_by"])
            lines.append(
                f" {i:2d}. {g['name']}\n"
                f"     weight {100 * g['whole']['weight']:.4f}% ({g['whole']['rows']:,} rows); on half B: SD "
                f"{100 * b['sd']:.3f}%, bias {_fmt_bias(b['bias'])}, p99.9 |error| {100 * (b.get('p999') or 0):.2f}%; "
                f"confirmed by {by}; "
                f"found by {', '.join(sorted(set(g['source'])))}; example row {g.get('example_row')}; "
                f"race classes 0-4: {', '.join(f'{100 * x / tot:.0f}%' for x in mix)}")
    lines.append("")
    lines.append(f"Time: {rep['seconds']}")
    return "\n".join(lines) + "\n"


def default_chunk(layers):
    """Rows per network batch: about 2^23 values of the widest layer (memory per worker)."""
    widest = max(w.shape[1] for w, _, _ in layers)
    return int(max(256, min(8192, (1 << 23) // max(widest, 1))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True, help="the integer network (int.safetensors)")
    ap.add_argument("--shard", action="append", required=True, help="an audit label shard (repeat for more)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gamma", type=float, default=0.001, help="smallest group, as a share of half A")
    ap.add_argument("--alpha-bias", type=float, default=0.0025)
    ap.add_argument("--alpha-sd", type=float, default=0.01)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--max-conj", type=int, default=3)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--screen", type=float, default=0.01,
                    help="half-A significance a candidate needs (1: every group over a threshold)")
    ap.add_argument("--seed", type=int, default=1, help="the half split's seed")
    ap.add_argument("--chunk", type=int, default=0, help="rows per batch of the network (0: by its width)")
    ap.add_argument("--workers", type=int, default=0,
                    help="worker processes (0: one per CPU this process may use); never changes a result")
    ap.add_argument("--work", default=None, help="the work files' directory (default OUT/work, removed after)")
    ap.add_argument("--keep-work", action="store_true", help="keep the work files")
    ap.add_argument("--parity-rows", type=int, default=512,
                    help="rows checked against the reference integer path (intnet.money_rows)")
    ap.add_argument("--overall-only", action="store_true",
                    help="only the network's overall numbers on the shards (no groups)")
    args = ap.parse_args(argv)
    t0 = time.time()
    seconds = {}
    os.makedirs(args.out, exist_ok=True)
    workers = args.workers or (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())
    work = args.work or os.path.join(args.out, "work")
    layers, meta = intnet.load(args.net)
    head = meta.get("head", "money4")
    shards = [load_shard(p) for p in args.shard]
    if len({s.dtype["outcomes"].shape[0] for s in shards}) != 1:
        raise SystemExit("the shards hold different outcome vocabularies: one frontier trick per audit")
    first = np.asarray(shards[0][: args.parity_rows])
    ref = reference_values(layers, head, first)
    if not np.array_equal(ref, network_values(layers, head, [first], 512)):
        raise SystemExit("the fast integer path differs from intnet.money_rows: a bug, stopping")
    seconds["load"] = round(time.time() - t0, 3)
    chunk = args.chunk or default_chunk(layers)
    try:
        with Workers(workers) as pool:
            t = time.time()
            d, trick = prepare_shards(work, [os.path.abspath(p) for p in args.shard], shards, os.path.abspath(args.net),
                                      args.seed, pool, chunk)
            if "trick" in meta and int(meta["trick"]) != trick:
                raise SystemExit(f"{args.net} is a trick-{meta['trick']} network; the shards are trick {trick}")
            seconds["network"] = round(time.time() - t, 3)
            print(f"audit: {d.n} rows, trick {trick}, network {head} and error columns in {seconds['network']:.1f} s "
                  f"on {pool.n} worker(s)", flush=True)
            if args.overall_only:
                pool.close()
                ov = overall(d)
                write_overall_only(args, meta, head, shards, trick, ov, seconds, t0)
                return
            body = audit_data(d, pool, gamma=args.gamma, alpha_bias=args.alpha_bias, alpha_sd=args.alpha_sd,
                              max_depth=args.max_depth, max_conj=args.max_conj, fdr=args.fdr, screen=args.screen,
                              log=lambda s: print(s, flush=True))
            pool.close()
        ov = overall(d)
    finally:
        _cleanup(work, keep=args.keep_work)
    seconds["audit"] = body["seconds"]
    seconds["total"] = round(time.time() - t0, 3)
    seconds["workers"] = workers
    rep = {
        "format": FORMAT,
        "encoder": "oracle-v2",
        "net": {"path": args.net, "sha256": sha256_file(args.net), "head": head, "trick": meta.get("trick")},
        "shards": [{"path": p, "rows": int(len(s)), "sha256": sha256_file(p)} for p, s in zip(args.shard, shards)],
        "trick": int(trick),
        "seed": args.seed,
        "split": "row i (shards in order) is in half B when the top bit of the i-th SplitMix64(seed) output is 1",
        "thresholds": {"gamma": args.gamma, "alpha_bias": args.alpha_bias, "alpha_sd": args.alpha_sd,
                       "max_depth": args.max_depth, "max_conj": args.max_conj, "fdr": args.fdr, "screen": args.screen},
        "teams": list(TEAMS),
        "parity_rows": int(len(first)),
        "overall": ov,
        "audit": {k: v for k, v in body.items() if k != "predicates"},
        "seconds": seconds,
    }
    with open(os.path.join(args.out, "predicates.json"), "w") as fh:
        json.dump(body["predicates"], fh, indent=1)
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    txt = text_report(rep)
    with open(os.path.join(args.out, "report.txt"), "w") as fh:
        fh.write(txt)
    print(txt, flush=True)


def write_overall_only(args, meta, head, shards, trick, ov, seconds, t0):
    """``--overall-only``'s report: the network, the shards and their ``overall`` numbers."""
    seconds["total"] = round(time.time() - t0, 3)
    rep = {
        "format": FORMAT,
        "encoder": "oracle-v2",
        "overall_only": True,
        "net": {"path": args.net, "sha256": sha256_file(args.net), "head": head, "trick": meta.get("trick")},
        "shards": [{"path": p, "rows": int(len(s)), "sha256": sha256_file(p)} for p, s in zip(args.shard, shards)],
        "trick": int(trick),
        "teams": list(TEAMS),
        "overall": ov,
        "seconds": seconds,
    }
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"overall: SD {100 * ov['sd']:.4f}%, bias {_fmt_bias(ov['bias'])} on {ov['rows']:,} rows", flush=True)


if __name__ == "__main__":
    main()
