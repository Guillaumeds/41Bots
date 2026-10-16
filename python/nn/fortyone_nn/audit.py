"""The auditor (docs/m3-hybrid/audit-loop.md §2-§4 and §6, build B1): finds the groups of
positions where an integer oracle network is weak, without being told what to look for.

    uv run python -m fortyone_nn.audit --net int.safetensors --shard AUDIT.npy [--shard ...]
        --out DIR [--gamma 0.001] [--alpha-bias 0.0025] [--alpha-sd 0.01] [--max-depth 6]
        [--max-conj 3] [--fdr 0.05] [--screen 0.01] [--seed 1] [--chunk 2048]

numpy only (no PyTorch), so it runs wherever the label shards are.

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
  sub-conjunction is frequent) and, as SliceFinder does, a candidate is not extended (a group
  inside it is reported through it). The sums are matrix products: single items exactly (float64
  on whole numbers whose partial sums stay below 2^53), pairs and triples in float32 with a
  rigorous bound on every sum's error, the candidate rule applied at both ends of each sum's
  range, and a conjunction the bound leaves undecided summed again exactly; so the candidates are
  exactly those of whole-number arithmetic on any machine and thread count (``Enumeration``).

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
    """The half-A candidate rule, vectorised over many groups' screening sums (columns
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
        s = np.asarray(s, dtype=np.float64)
        n = s[:, 0]
        nn = np.maximum(n, 2.0)
        e = s[:, 1:4]
        ab = np.abs(np.column_stack([e, -e.sum(axis=1)])).max(axis=1) / nn / ONE
        msq = s[:, 4] / nn / ONE
        var_q = np.maximum(s[:, 5] / nn / ONE - msq**2, 0.0) * nn / (nn - 1)
        sd, bi = self._decide(n, ab, msq, var_q, msq)
        return sd | bi, sd, bi

    def bounded(self, n, s, gamma):
        """The rule on approximate sums ``s`` (float64 [k, 6], column 0 exact) each within
        ``gamma`` times the sum of the absolute values summed (float32 products): (decided,
        candidate, sd flag, bias flag); where not decided, the exact sums must be used. The
        decision is taken at both ends of every sum's range (the flags grow with the mean square
        and the largest |bias|, and fall with the variance and, for the bias, the mean square)."""
        nn = np.maximum(n, 2.0)
        sq, sqq = s[:, 4], s[:, 5]
        dq = gamma * np.abs(sq) / (1 - gamma)
        dqq = gamma * np.abs(sqq) / (1 - gamma)
        sq_hi, sq_lo = sq + dq, np.maximum(sq - dq, 0.0)
        sqq_hi, sqq_lo = sqq + dqq, np.maximum(sqq - dqq, 0.0)
        # Every team's squared error is below 3 * 2^32 * (q + 1) per row; Cauchy-Schwarz bounds
        # the sum of |e_t| over the rows.
        de = gamma * np.sqrt(nn * 3.0 * ONE * (sq_hi + nn)) / (1 - gamma)
        e = s[:, 1:4]
        e4 = np.column_stack([e, -e.sum(axis=1)])
        d4 = np.column_stack([de, de, de, 3 * de])
        ab_hi = (np.abs(e4) + d4).max(axis=1) / nn / ONE
        ab_lo = np.maximum(np.abs(e4) - d4, 0.0).max(axis=1) / nn / ONE
        m_hi, m_lo = sq_hi / nn / ONE, sq_lo / nn / ONE
        v_hi = np.maximum(sqq_hi / nn / ONE - m_lo**2, 0.0) * nn / (nn - 1)
        v_lo = np.maximum(sqq_lo / nn / ONE - m_hi**2, 0.0) * nn / (nn - 1)
        sd_hi, bi_hi = self._decide(n, ab_hi, m_hi, v_lo, m_lo)
        sd_lo, bi_lo = self._decide(n, ab_lo, m_lo, v_hi, m_hi)
        decided = (sd_hi == sd_lo) & (bi_hi == bi_lo)
        return decided, sd_lo | bi_lo, sd_lo, bi_lo


# ---------------------------------------------------------------------------------------------
# Half split.


def half_b(seed, n):
    """bool [n]: row i is in half B when the top bit of the i-th output of SplitMix64(seed) is 1."""
    i = np.arange(1, n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed & 0xFFFFFFFFFFFFFFFF) + i * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(63)).astype(bool)


# ---------------------------------------------------------------------------------------------
# Regression trees.

N_BINS = 64  # a dense feature splits at up to N_BINS - 1 thresholds


def dense_thresholds(dense):
    """Per dense feature: the sorted distinct values of its quantiles k / N_BINS (k = 1..63) on
    the given rows, each a value of the data; a split is "x <= threshold"."""
    out = []
    for j in range(dense.shape[1]):
        x = np.sort(np.asarray(dense[:, j], dtype=np.int64))
        if not len(x):
            out.append(np.zeros(0, dtype=np.int64))
            continue
        idx = (np.arange(1, N_BINS) * len(x)) // N_BINS
        t = np.unique(x[np.minimum(idx, len(x) - 1)])
        out.append(t[t < x[-1]])  # a threshold at the maximum splits nothing off
    return out


def dense_bins(dense, thresholds):
    """uint8 [n, N_DENSE]: the number of thresholds strictly below the value, so that
    ``x <= t_k`` exactly when the bin is at most ``k``."""
    out = np.zeros(dense.shape, dtype=np.uint8)
    for j, t in enumerate(thresholds):
        out[:, j] = np.searchsorted(t, np.asarray(dense[:, j], dtype=np.int64), side="left")
    return out


def _histograms(act, bins, y, rows):
    """A node's sums: per binary feature, rows on and target sums; per dense feature and bin, rows
    and target sums; the node's target total and rows."""
    k, d = len(rows), y.shape[1]
    nb = N_BINS
    yr = y[rows]
    a = act[rows]
    valid = a != PAD
    flat = a[valid]
    rpos = np.broadcast_to(np.arange(k)[:, None], a.shape)[valid]
    n_on = np.bincount(flat, minlength=N_BINARY).astype(np.float64)
    s_on = np.stack([np.bincount(flat, weights=yr[rpos, t], minlength=N_BINARY) for t in range(d)], axis=1)
    bflat = (np.arange(N_DENSE)[None, :] * nb + bins[rows].astype(np.int64)).reshape(-1)
    cnt = np.bincount(bflat, minlength=N_DENSE * nb).reshape(N_DENSE, nb).astype(np.float64)
    sums = np.stack([np.bincount(bflat, weights=np.repeat(yr[:, t], N_DENSE), minlength=N_DENSE * nb)
                     .reshape(N_DENSE, nb) for t in range(d)], axis=2)
    return [n_on, s_on, cnt, sums, yr.sum(axis=0), float(k)]


def fit_tree(active, bins, thresholds, y, min_leaf, max_depth):
    """A CART regression tree on the rows given (``active`` [m, 96] the shard's active lists,
    ``bins`` from ``dense_bins``, ``y`` [m, d] float64 targets, squared-error impurity summed over
    the d targets). Returns its nodes below the root: (path conditions, row indices, depth). A
    node's sums are counted for the smaller child of each split and taken from the parent for the
    other (the histogram subtraction of gradient-boosting libraries; the same splits up to
    rounding)."""
    m, d = y.shape
    act = np.asarray(active).astype(np.int64)
    nb = N_BINS
    exists = np.arange(nb - 1)[None, :] < np.array([len(t) for t in thresholds])[:, None]
    out = []
    queue = [([], np.arange(m), 0, None)]
    while queue:
        conds, rows, depth, hist = queue.pop(0)
        if depth > 0:
            out.append((conds, rows, depth))
        if depth >= max_depth or len(rows) < 2 * min_leaf:
            continue
        if hist is None:
            hist = _histograms(act, bins, y, rows)
        n_on, s_on, cnt, sums, total, k = hist
        base = float((total**2).sum()) / k
        # Binary features: left = on.
        n_off = k - n_on
        s_off = total[None, :] - s_on
        okb = (n_on >= min_leaf) & (n_off >= min_leaf)
        with np.errstate(divide="ignore", invalid="ignore"):
            gb = (s_on**2).sum(axis=1) / n_on + (s_off**2).sum(axis=1) / n_off - base
        gb = np.where(okb, gb, -np.inf)
        # Dense features: left = bin <= b (x <= thresholds[j][b]).
        n_le = np.cumsum(cnt, axis=1)[:, : nb - 1]
        s_le = np.cumsum(sums, axis=1)[:, : nb - 1, :]
        n_gt = k - n_le
        s_gt = total[None, None, :] - s_le
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
            left_mask = (act[rows] == f).any(axis=1)
            cl = [{"feature": f, "op": "==", "value": 1}]
            cr = [{"feature": f, "op": "==", "value": 0}]
        else:
            j, b = divmod(best - N_BINARY, nb - 1)
            t = int(thresholds[j][b])
            left_mask = bins[rows, j] <= b
            cl = [{"feature": N_BINARY + j, "op": "<=", "value": t}]
            cr = [{"feature": N_BINARY + j, "op": ">", "value": t}]
        lrows, rrows = rows[left_mask], rows[~left_mask]
        if depth + 1 < max_depth:
            small_left = len(lrows) <= len(rrows)
            hs = _histograms(act, bins, y, lrows if small_left else rrows)
            hl = [p - c for p, c in zip(hist, hs)]
            hleft, hright = (hs, hl) if small_left else (hl, hs)
        else:
            hleft = hright = None
        queue.append((conds + cl, lrows, depth + 1, hleft))
        queue.append((conds + cr, rrows, depth + 1, hright))
    return out


# ---------------------------------------------------------------------------------------------
# The conjunction enumeration.

EXACT_ROWS = 1 << 17  # rows per float64 product: |sum| < 2^17 * 2^33 = 2^50, exact
FLOAT_ROWS = 1 << 14  # rows per float32 product (the error bound grows with it)
U32 = 2.0**-24
# The bound on a float32 dot product of FLOAT_ROWS terms (Higham, Accuracy and Stability of
# Numerical Algorithms, 2002, §3.1: |error| <= gamma_n * sum |x_i y_i| for any order of the
# additions), with one more rounding for the inputs' conversion to float32 and a margin for the
# float64 sums of the blocks' results.
GAMMA32 = (FLOAT_ROWS + 1) * U32 / (1 - (FLOAT_ROWS + 1) * U32) + 1e-9


def _exact_tdot(a, b, rows=EXACT_ROWS):
    """``a.T @ b`` for 0/1 or whole-number float64 matrices whose entries are below 2^33 in size,
    exactly (int64), in blocks of at most ``EXACT_ROWS`` rows."""
    out = None
    for i in range(0, max(len(a), 1), rows):
        x = np.rint(a[i : i + rows].T @ b[i : i + rows]).astype(np.int64)
        out = x if out is None else out + x
    return out


class Workspace:
    """Buffers reused by every ``cross32`` call (fresh arrays of this size each call cost more in
    page faults than the products)."""

    def __init__(self, items, columns, rows=FLOAT_ROWS):
        self.x = np.empty((rows, max(items, 1)), dtype=np.float32)
        self.st = np.empty((rows, columns * max(items, 1)), dtype=np.float32)


def cross32(xbool, w, cols, ws=None):
    """float64 [c, m, m]: for every weight column c, the sum over the rows of
    ``w[r, c] * x[r, j] * x[r, k]`` (j, k in ``cols``) by float32 products of blocks of
    ``FLOAT_ROWS`` rows (one stacked product a block): each within ``GAMMA32`` times the sum of
    ``|w[r, c]|`` over the same rows; column 0 must be the 0/1 count, which is exact."""
    c = w.shape[1]
    m = len(cols)
    ws = ws or Workspace(m, c)
    out = np.zeros((c * m, m), dtype=np.float64)
    for i in range(0, len(xbool), FLOAT_ROWS):
        blk = xbool[i : i + FLOAT_ROWS]
        k = len(blk)
        xa = ws.x[:k, :m]
        xa[...] = blk if m == blk.shape[1] and np.array_equal(cols, np.arange(m)) else blk[:, cols]
        st = ws.st[:k, : c * m]
        np.multiply(xa[:, None, :], w[i : i + k].astype(np.float32)[:, :, None], out=st.reshape(k, c, m))
        out += st.T @ xa
    return out.reshape(c, m, m)


def exact_pairs(xbool, w, ja, jb):
    """int64 [len(ja), c]: the exact sums of ``w`` over the rows where items ``ja[p]`` and
    ``jb[p]`` are both on."""
    out = np.zeros((len(ja), w.shape[1]), dtype=np.int64)
    wf = w.astype(np.float64)
    step = 256
    for p in range(0, len(ja), step):
        both = (xbool[:, ja[p : p + step]] & xbool[:, jb[p : p + step]]).astype(np.float64)
        out[p : p + step] = _exact_tdot(both, wf)
    return out


class Enumeration:
    """Every conjunction of up to ``max_conj`` items with support >= ``minsup`` on the rows
    given, with its candidate decision; candidates are not extended (see the module notes).
    Level 1 sums are exact float64 products. Levels 2 and 3 are float32 products with a rigorous
    error bound (``cross32``, ``Screen.bounded``); a conjunction the bound leaves undecided is
    summed again exactly (``exact_pairs``), so the candidates are exactly those of whole-number
    arithmetic, on any machine. A triple (i, j, k) is summed over item i's rows, among the items
    j, k whose pairs with i, and with each other, are frequent and not candidates."""

    def __init__(self, items, xbool, cols, screen, max_conj):
        if not 1 <= max_conj <= 3:
            raise ValueError("--max-conj is 1, 2 or 3")
        self.items, self.screen = items, screen
        self.counted = [0, 0, 0]
        self.frequent = [0, 0, 0]
        self.rechecked = [0, 0, 0]
        self.found = []  # (item tuple, sd flag, bias flag)
        f = len(items)
        w = cols.astype(np.int64)
        s1 = np.zeros((f, w.shape[1]), dtype=np.int64)
        for r in range(0, len(xbool), FLOAT_ROWS):
            xr = xbool[r : r + FLOAT_ROWS].T.astype(np.float64)
            s1 += np.rint(xr @ w[r : r + FLOAT_ROWS].astype(np.float64)).astype(np.int64)
        cand, sdf, bif = screen(s1)
        freq1 = s1[:, 0] >= screen.minsup
        self.counted[0], self.frequent[0] = f, int(freq1.sum())
        self._keep(np.arange(f)[:, None], cand, sdf, bif)
        # Extended: frequent items that are not candidates and not on in every row (those define
        # no group of their own), in increasing support (ties by index), so that a triple is
        # summed over the rows of its rarest item.
        grow = np.flatnonzero(freq1 & ~cand & (s1[:, 0] < len(xbool)))
        idx = grow[np.lexsort((grow, s1[grow, 0]))]
        if max_conj < 2 or len(idx) < 2:
            return
        # Pairs (a, b) of positions in ``idx`` with a < b.
        ws = Workspace(len(idx), w.shape[1])
        iu, ju = np.triu_indices(len(idx), 1)
        s2 = cross32(xbool, w, idx, ws)[:, iu, ju].T
        c2, fr2 = self._level(1, s2, xbool, w, idx[iu], idx[ju])
        self._keep(np.column_stack([idx[iu], idx[ju]]), *c2)
        grow2 = np.zeros((len(idx), len(idx)), dtype=bool)
        grow2[iu, ju] = fr2 & ~c2[0]
        if max_conj < 3:
            return
        for a in range(len(idx)):
            bs = np.flatnonzero(grow2[a])
            if len(bs) < 2:
                continue
            iu, ju = np.triu_indices(len(bs), 1)
            ok = grow2[bs[iu], bs[ju]]
            if not ok.any():
                continue
            iu, ju = iu[ok], ju[ok]
            js = idx[bs]
            r = np.flatnonzero(xbool[:, idx[a]])
            xr, wr = xbool[np.ix_(r, js)], w[r]
            local = np.arange(len(js))
            s3 = cross32(xr, wr, local, ws)[:, iu, ju].T
            c3, _ = self._level(2, s3, xr, wr, iu, ju)
            self._keep(np.column_stack([np.full(len(iu), idx[a]), js[iu], js[ju]]), *c3)

    def _level(self, k, s, xbool, w, ja, jb):
        """Decisions for one level's approximate sums; the undecided ones summed exactly."""
        n = np.rint(s[:, 0])
        decided, cand, sdf, bif = self.screen.bounded(n, s, GAMMA32)
        fr = n >= self.screen.minsup
        und = np.flatnonzero(~decided & fr)
        if len(und):
            ex = exact_pairs(xbool, w, ja[und], jb[und])
            c, d, b = self.screen(ex)
            cand[und], sdf[und], bif[und] = c, d, b
        self.counted[k] += len(s)
        self.frequent[k] += int(fr.sum())
        self.rechecked[k] += len(und)
        return (cand & fr, sdf & fr, bif & fr), fr

    def _keep(self, tuples, cand, sdf, bif):
        for k in np.flatnonzero(cand):
            self.found.append((tuple(int(v) for v in tuples[k]), bool(sdf[k]), bool(bif[k])))


def enumeration_items(binary_a, minsup):
    """The items: (feature, 1) for every binary feature on in at least ``minsup`` rows, (feature,
    0) for the yes/no features off in at least ``minsup`` rows; and their bool columns."""
    on = binary_a.sum(axis=0)
    items, cols = [], []
    for f in range(N_BINARY):
        if on[f] >= minsup:
            items.append((f, 1))
            cols.append(binary_a[:, f])
        if f in YES_NO and len(binary_a) - on[f] >= minsup:
            items.append((f, 0))
            cols.append(~binary_a[:, f])
    x = np.column_stack(cols) if cols else np.zeros((len(binary_a), 0), dtype=bool)
    return items, x


# ---------------------------------------------------------------------------------------------
# The audit.


def _group_record(conds, source, members_all, is_b, err, cols, races, total_rows):
    """Statistics of a group over the whole shard and each half."""
    rows = np.flatnonzero(members_all)
    rows_a = rows[~is_b[rows]]
    rows_b = rows[is_b[rows]]
    sa, sb, sw = Stats(sums_of(cols, rows_a)), Stats(sums_of(cols, rows_b)), Stats(sums_of(cols, rows))
    rec = {"name": group_name(conds), "conditions": conds, "source": source}
    rec["whole"] = sw.as_dict(total_rows)
    rec["half_a"] = sa.as_dict(int((~is_b).sum()))
    rec["half_b"] = sb.as_dict(int(is_b.sum()))
    if len(rows_b):
        eb = np.abs(err.e[rows_b]).astype(np.float64) / ONE
        rec["half_b"]["p999"] = float(np.quantile(eb, 0.999))
        worst = rows_b[int(np.argmax(err.q[rows_b]))]
        rec["example_row"] = int(worst)
    rec["race_classes"] = [int(x) for x in np.bincount(races[rows], minlength=len(RACE_CLASS_NAMES))]
    return rec, sa, sb


def audit(err, binary, dense, is_b, races, gamma=0.001, alpha_bias=0.0025, alpha_sd=0.01, max_depth=6,
          max_conj=3, fdr=0.05, screen=0.01, log=print):
    """The discovery (half A) and confirmation (half B) of weak groups. ``err``: ``Errors`` of all
    rows; ``binary``: ``binary_matrix`` of their active lists; ``dense``: their dense values;
    ``is_b``: ``half_b``; ``races``: their race classes (a diagnostic only). Returns a dict (the
    report's body; ``confirmed`` and ``predicates``)."""
    t0 = time.time()
    timing = {}
    n = len(err)
    a_rows = np.flatnonzero(~is_b)
    n_a = len(a_rows)
    minsup = max(2, math.ceil(gamma * n_a))
    cols = err.columns()
    screen_cols = cols[:, list(SCREEN_COLS)]
    scr = Screen(alpha_bias, alpha_sd, minsup, screen)

    # Trees on half A.
    act_a = np.full((n_a, MAX_ACTIVE), PAD, dtype=np.int64)
    bin_a = binary[a_rows]
    # Rebuild the active lists from the binary matrix (ascending), the tree's input.
    r_idx, f_idx = np.nonzero(bin_a)
    pos = np.arange(len(r_idx)) - np.searchsorted(r_idx, r_idx, side="left")
    act_a[r_idx, pos] = f_idx
    dense_a = np.asarray(dense)[a_rows]
    thr = dense_thresholds(dense_a)
    bins_a = dense_bins(dense_a, thr)
    ef = err.e[a_rows].astype(np.float64) / ONE
    trees = {"tree-signed": ef, "tree-abs": np.abs(ef).mean(axis=1, keepdims=True)}
    candidates = {}  # condition key -> {conditions, sources, flags}
    tree_nodes = {}
    for name, y in trees.items():
        nodes = fit_tree(act_a, bins_a, thr, y, minsup, max_depth)
        tree_nodes[name] = len(nodes)
        if nodes:
            sums = np.array([sums_of(screen_cols, a_rows[rows]) for _, rows, _ in nodes], dtype=np.int64)
            cand, sdf, bif = scr(sums)
            for k in np.flatnonzero(cand):
                conds = canonical(nodes[k][0])
                key = cond_key(conds)
                c = candidates.setdefault(key, {"conditions": conds, "sources": [], "sd": False, "bias": False})
                c["sources"].append(name)
                c["sd"] |= bool(sdf[k])
                c["bias"] |= bool(bif[k])
    timing["trees"] = round(time.time() - t0, 3)
    log(f"audit: trees on half A ({n_a} rows): {tree_nodes} nodes, {len(candidates)} candidates so far "
        f"({timing['trees']:.1f} s)")

    # The enumeration on half A.
    t1 = time.time()
    items, x_a = enumeration_items(bin_a, minsup)
    en = Enumeration(items, x_a, screen_cols[a_rows], scr, max_conj)
    for tup, sdf, bif in en.found:
        conds = canonical([{"feature": items[i][0], "op": "==", "value": items[i][1]} for i in tup])
        key = cond_key(conds)
        c = candidates.setdefault(key, {"conditions": conds, "sources": [], "sd": False, "bias": False})
        c["sources"].append(f"conj-{len(tup)}")
        c["sd"] |= sdf
        c["bias"] |= bif
    timing["enumeration"] = round(time.time() - t1, 3)
    log(f"audit: enumeration: {len(items)} items, conjunctions counted by size {en.counted}, frequent "
        f"{en.frequent}, summed again exactly {en.rechecked}, {len(en.found)} candidates "
        f"({timing['enumeration']:.1f} s)")

    # Confirmation on half B.
    t2 = time.time()
    keys = sorted(candidates)
    hyp = []  # (candidate index, criterion, team, sign, p)
    mem = {}
    for ci, key in enumerate(keys):
        c = candidates[key]
        m = members(c["conditions"], binary, dense)
        mem[key] = m
        sa = Stats(sums_of(cols, m & ~is_b))
        sb = Stats(sums_of(cols, m & is_b))
        if c["sd"]:
            hyp.append((ci, "sd", None, None, sb.p_sd(alpha_sd)))
        if c["bias"]:
            team, sign = sa.worst_team()
            hyp.append((ci, "bias", team, sign, sb.p_bias(team, sign, alpha_bias)))
    rejected = benjamini_hochberg([h[4] for h in hyp], fdr)
    timing["confirmation"] = round(time.time() - t2, 3)

    confirmed = {}
    for h, rej in zip(hyp, rejected):
        ci, crit, team, sign, p = h
        if rej:
            confirmed.setdefault(ci, []).append({"criterion": crit, "team": team, "sign": sign, "p": p})
    # Minimality: a confirmed group inside a more general confirmed group is covered by it.
    conf_keys = {keys[ci] for ci in confirmed}
    records = []
    for ci in sorted(confirmed):
        key = keys[ci]
        c = candidates[key]
        rec, _, sb = _group_record(c["conditions"], c["sources"], mem[key], is_b, err, cols, races, n)
        rec["confirmed_by"] = confirmed[ci]
        rec["p_values"] = {h[1]: h[4] for h in hyp if h[0] == ci}
        covered = next((h for h in sorted(conf_keys) if h != key and implies(key, h)), None)
        rec["covered_by"] = None if covered is None else group_name(
            [{"feature": f, "op": op, "value": v} for f, op, v in covered])
        rec["severity"] = max(sb.sd / alpha_sd, max(abs(b) for b in sb.bias) / alpha_bias)
        records.append(rec)
    records.sort(key=lambda r: (-r["severity"], -r["whole"]["rows"], r["name"]))
    predicates = []
    for r in records:
        if r["covered_by"] is None:
            predicates.append({
                "name": r["name"], "conditions": r["conditions"], "weight": r["whole"]["weight"],
                "bias": r["half_b"]["bias"], "sd": r["half_b"]["sd"], "p999": r["half_b"].get("p999"),
                "rows": r["whole"]["rows"], "example_row": r.get("example_row"),
                "criteria": [x["criterion"] for x in r["confirmed_by"]], "source": r["source"],
            })
    inside = np.zeros(n, dtype=bool)
    for p in predicates:
        inside |= mem[cond_key(p["conditions"])]
    coverage = []
    for k, name in enumerate(RACE_CLASS_NAMES):
        sel = races == k
        coverage.append({"class": k, "rows": int(sel.sum()),
                         "share_inside_groups": float(inside[sel].mean()) if sel.any() else None})
    timing["total"] = round(time.time() - t0, 3)
    n_tree = sum(1 for c in candidates.values() if any(s.startswith("tree") for s in c["sources"]))
    n_conj = sum(1 for c in candidates.values() if any(s.startswith("conj") for s in c["sources"]))
    return {
        "rows": n,
        "half_a_rows": int(n_a),
        "half_b_rows": int(n - n_a),
        "min_group_rows": int(minsup),
        "tree_nodes": tree_nodes,
        "items": len(items),
        "conjunctions_counted": en.counted,
        "conjunctions_frequent": en.frequent,
        "conjunctions_rechecked_exactly": en.rechecked,
        "candidates": len(candidates),
        "candidates_from_trees": n_tree,
        "candidates_from_conjunctions": n_conj,
        "hypotheses": len(hyp),
        "confirmed": len(records),
        "confirmed_general": len(predicates),
        "groups": records,
        "rows_inside_groups": int(inside.sum()),
        "race_class_coverage": coverage,
        "predicates": predicates,
        "seconds": timing,
    }


def overall(err, idx, races):
    e = err.e[idx].astype(np.float64) / ONE
    s = Stats(sums_of(err.columns(), idx))
    out = {"rows": int(len(idx)), "bias": s.bias, "sd": s.sd,
           "p999": float(np.quantile(np.abs(e), 0.999)) if len(e) else None}
    out["by_race_class"] = []
    for k, name in enumerate(RACE_CLASS_NAMES):
        sel = idx[races[idx] == k]
        sk = Stats(sums_of(err.columns(), sel))
        out["by_race_class"].append({"class": k, "name": name, "rows": int(len(sel)),
                                     "bias": sk.bias if len(sel) else None, "sd": sk.sd if len(sel) else None})
    return out


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
        f"{body['conjunctions_counted']} (frequent {body['conjunctions_frequent']}).",
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
                f"     weight {100 * g['whole']['weight']:.2f}% ({g['whole']['rows']:,} rows); on half B: SD "
                f"{100 * b['sd']:.3f}%, bias {_fmt_bias(b['bias'])}, p99.9 |error| {100 * b.get('p999', 0):.2f}%; "
                f"confirmed by {by}; "
                f"found by {', '.join(sorted(set(g['source'])))}; example row {g.get('example_row')}; "
                f"race classes 0-4: {', '.join(f'{100 * x / tot:.0f}%' for x in mix)}")
    lines.append("")
    lines.append(f"Time: {rep['seconds']}")
    return "\n".join(lines) + "\n"


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
    ap.add_argument("--chunk", type=int, default=2048, help="rows per batch of the network")
    ap.add_argument("--parity-rows", type=int, default=512,
                    help="rows checked against the reference integer path (intnet.money_rows)")
    args = ap.parse_args(argv)
    t0 = time.time()
    seconds = {}
    os.makedirs(args.out, exist_ok=True)
    layers, meta = intnet.load(args.net)
    head = meta.get("head", "money4")
    shards = [load_shard(p) for p in args.shard]
    tricks, vocab = set(), set()
    for p, s in zip(args.shard, shards):
        for i in range(0, len(s), 1 << 16):
            blk = s[i : i + (1 << 16)]
            f = value_fault(blk)
            if f is not None:
                raise SystemExit(f"{p}: {VALUE_FAULTS[f]}")
            tricks |= set(np.unique(blk["trick"]).tolist())
        vocab.add(s.dtype["outcomes"].shape[0])
    if len(tricks) != 1 or len(vocab) != 1:
        raise SystemExit(f"the shards hold tricks {sorted(tricks)}: one frontier trick per audit")
    trick = tricks.pop()
    if "trick" in meta and int(meta["trick"]) != trick:
        raise SystemExit(f"{args.net} is a trick-{meta['trick']} network; the shards are trick {trick}")
    seconds["load"] = round(time.time() - t0, 3)

    t = time.time()
    pred = network_values(layers, head, shards, args.chunk)
    seconds["network"] = round(time.time() - t, 3)
    first = np.asarray(shards[0][: args.parity_rows])
    ref = reference_values(layers, head, first)
    if not np.array_equal(ref, pred[: len(first)]):
        raise SystemExit("the fast integer path differs from intnet.money_rows: a bug, stopping")
    value = np.concatenate([np.asarray(s["value"]) for s in shards])
    active = np.concatenate([np.asarray(s["active"]) for s in shards])
    dense = np.concatenate([np.asarray(s["dense"]) for s in shards])  # int32: predicates' values fit it
    races = np.concatenate([race_classes(s) for s in shards])
    err = Errors(pred, value)
    binary = binary_matrix(active)
    del active
    is_b = half_b(args.seed, len(err))
    print(f"audit: {len(err)} rows, trick {trick}, network {head} in {seconds['network']:.1f} s", flush=True)
    body = audit(err, binary, dense, is_b, races, gamma=args.gamma, alpha_bias=args.alpha_bias, alpha_sd=args.alpha_sd,
                 max_depth=args.max_depth, max_conj=args.max_conj, fdr=args.fdr, screen=args.screen,
                 log=lambda s: print(s, flush=True))
    seconds["audit"] = body["seconds"]
    seconds["total"] = round(time.time() - t0, 3)
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
        "overall": overall(err, np.arange(len(err)), races),
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


if __name__ == "__main__":
    main()
