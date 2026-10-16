"""The auditor (``fortyone_nn.audit``, docs/m3-hybrid/audit-loop.md build B1).

numpy only (runs in the CI's parity job). Run: ``cd python/nn && uv run python -m unittest
discover -s tests -v``.

- the integer network: ``ExactNet`` and the vectorised softmax give exactly ``intnet``'s numbers
  (golden networks of ``crates/nn``, quantised random networks, both heads, softmax edges), so the
  auditor's errors are those of the network Rust runs;
- whole-number error columns against Python's big integers; the half split against SplitMix64;
  Student's t against closed forms; Benjamini-Hochberg against its definition;
- the enumeration with float32 products and error bounds gives exactly the candidates of a plain
  whole-number enumeration (also with a bound so wide that most decisions are summed again);
- synthetic shards: a planted conjunction of two binary features with a bias and a planted
  dense-threshold group with a larger spread are confirmed and named, and every confirmed group
  really breaks a threshold; on clean shards nothing is claimed, and with every group exactly at
  the thresholds (the null's boundary) claims stay within the false-discovery rate over 20 runs;
- predicates: the format's checks, the round trip through predicates.json, and the shared fixture
  (``tests/fixtures``: ``predicate-rows.json``, 48 rows of a real trick-4 harvest shard as JSON
  (the repository keeps no .npy files); ``predicates.json``; ``predicate-members.json``, the
  expected members of each group, which Rust's ``nn::predicate`` also tests against), checked by
  a second, plain evaluator too;
- the command line end to end on a small shard (both heads): the files, and the same files again
  for the same seed.
"""

import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fortyone_nn import audit as au
from fortyone_nn import intnet
from fortyone_nn.formats import (
    MAX_ACTIVE, N_BINARY, N_DENSE, N_FEATURES, ONE, PAD, VOCAB, load_shard, shard_dtype,
)
from fortyone_nn.quantize import quantize, write_int

from test_intnet import GOLDEN, random_float_net

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------------------------
# The integer network.


class ExactNetwork(unittest.TestCase):
    def test_logits_are_intnets(self):
        d = json.loads(GOLDEN.read_text())
        active, dense = intnet.golden_features(d["seed"] + 1, d["count"])
        x = intnet.inputs(active, dense)
        for layers in (intnet.golden_net(d["seed"]), intnet.golden_net(d["seed"], 14),
                       quantize(random_float_net(4, (256, 64, 32), 14)), quantize(random_float_net(5, (32,), 4))):
            np.testing.assert_array_equal(au.ExactNet(layers).logits(active, dense), intnet.logits(layers, x))

    def test_golden_money(self):
        d = json.loads(GOLDEN.read_text())
        active, dense = intnet.golden_features(d["seed"] + 1, d["count"])
        z = au.ExactNet(intnet.golden_net(d["seed"])).logits(active, dense)
        np.testing.assert_array_equal(au.money4(z), np.array(d["outputs"], dtype=np.uint64))
        n_out = d["outcome"]["n_out"]
        tables = intnet.golden_outcomes(d["seed"] + 2, d["count"], n_out)
        z = au.ExactNet(intnet.golden_net(d["seed"], n_out)).logits(active, dense)
        mask = np.array([m for m, _ in tables], dtype=np.uint64)
        vecs = np.array([v for _, v in tables], dtype=np.uint64)
        self.assertEqual(au.money_outcomes(z, mask, vecs).tolist(), d["outcome"]["outputs"])

    def test_softmax_edges(self):
        rng = np.random.default_rng(3)
        d = rng.integers(0, 1 << 34, size=5000)
        d[:64] = np.arange(64)
        d[64:128] = (1 << 31) - np.arange(64)
        d[128:228] = 731_000_000 + np.arange(0, 10_000_000, 100_000)  # around the n >= 63 edge
        want = np.array([intnet.exp_weight(int(x)) for x in d], dtype=np.uint64)
        np.testing.assert_array_equal(au.exp_weights(d), want)
        z = rng.integers(-(1 << 40), 1 << 40, size=(2000, 4))
        z[:400] = rng.integers(-(1 << 26), 1 << 26, size=(400, 4))
        z[400:500] = 0
        z[500:600, 1:] = z[500:600, :1]  # ties
        np.testing.assert_array_equal(au.money4(z), np.array([intnet.money(r) for r in z], dtype=np.uint64))

    def test_outcome_head_on_rows(self):
        rows = fixture_rows()
        layers = quantize(random_float_net(7, (64, 16), VOCAB[4]))
        got = au.network_values(layers, "outcome-v1", [rows], chunk=7)
        np.testing.assert_array_equal(got, au.reference_values(layers, "outcome-v1", rows))
        layers = quantize(random_float_net(8, (64, 16), 4))
        got = au.network_values(layers, "money4", [rows], chunk=11)
        np.testing.assert_array_equal(got, au.reference_values(layers, "money4", rows))


# ---------------------------------------------------------------------------------------------
# Whole-number columns, split, tests.


class Columns(unittest.TestCase):
    def test_error_columns(self):
        rng = np.random.default_rng(5)
        e = rng.integers(-(1 << 32), (1 << 32) + 1, size=(300, 4))
        e[:5] = [[ONE, -ONE, 0, 0], [-ONE, ONE, 0, 0], [0, 0, 0, 0], [1, -1, 3, -3], [ONE, 0, 0, -ONE]]
        err = au.Errors.of(e)
        for r in range(len(e)):
            sq = [int(x) ** 2 for x in e[r]]
            self.assertEqual([int(x) for x in err.e2[r]], [x >> 32 for x in sq])
            q = sum(sq) >> 34
            self.assertEqual(int(err.q[r]), q)
            self.assertEqual(int(err.qq[r]), (q * q) >> 32)
        with self.assertRaises(ValueError):
            au.Errors.of(np.array([[ONE + 1, -ONE - 1, 0, 0]]))

    def test_stats(self):
        rng = np.random.default_rng(6)
        e = rng.normal(0, 0.01, size=(5000, 4))
        e -= e.mean(axis=1, keepdims=True)
        e[:, 1] += 0.002
        e[:, 3] -= 0.002  # errors sum to zero over the teams, as the network's and the labels' do
        err = au.Errors.of(np.rint(e * ONE).astype(np.int64))
        s = au.Stats(au.sums_of(err.columns()))
        f = err.e / ONE
        np.testing.assert_allclose(s.bias, f.mean(axis=0), rtol=0, atol=1e-12)
        # The four teams' errors sum to zero, so the SD is the root mean square (each row's mean
        # square is floored to a whole number of 2^-32: below 2^-32 too low).
        self.assertAlmostEqual(s.sd, f.std(), places=7)
        np.testing.assert_allclose(s.var, f.var(axis=0, ddof=1), rtol=0, atol=2.5e-10)  # squares floored
        self.assertEqual(s.worst_team(), (1, 1))

    def test_half_split_is_splitmix64(self):
        for seed in (1, 9145, (1 << 64) - 1):
            sm = intnet.SplitMix64(seed)
            want = [sm.next() >> 63 == 1 for _ in range(1000)]
            self.assertEqual(au.half_b(seed, 1000).tolist(), want)
        self.assertAlmostEqual(float(au.half_b(1, 100000).mean()), 0.5, delta=0.01)

    def test_student_t(self):
        for t in (0.3, 1.0, 2.5, 7.0, -1.2):
            self.assertAlmostEqual(au.t_sf(t, 1), 0.5 - math.atan(t) / math.pi, places=12)
            self.assertAlmostEqual(au.t_sf(t, 2), 0.5 - t / (2 * math.sqrt(t * t + 2)), places=12)
        self.assertAlmostEqual(au.t_sf(2.0, 10), 0.036694017385370, places=10)
        self.assertAlmostEqual(au.t_sf(1.959963984540054, 10**7), 0.025, places=6)
        self.assertAlmostEqual(au.t_sf(0.0, 5), 0.5)
        self.assertAlmostEqual(au.normal_isf(0.01), 2.3263478740408408, places=9)
        self.assertEqual(au.t_test_p(1.0, 0.0, 10), 0.0)
        self.assertEqual(au.t_test_p(-1.0, 0.0, 10), 1.0)

    def test_benjamini_hochberg(self):
        p = [0.01, 0.04, 0.03, 0.005, 0.2, 0.041]
        # Sorted: 0.005, 0.01, 0.03, 0.04, 0.041, 0.2 against 0.05 k / 6 = 0.0083, 0.0167, 0.025,
        # 0.0333, 0.0417, 0.05: the largest k with p_(k) under its line is 5 (0.041), so the first
        # five are rejected although 0.03 and 0.04 are above their own lines.
        self.assertEqual(au.benjamini_hochberg(p, 0.05).tolist(), [True, True, True, True, False, True])
        self.assertEqual(au.benjamini_hochberg([0.03, 0.2], 0.05).tolist(), [False, False])
        self.assertEqual(au.benjamini_hochberg([0.5, 0.9], 0.05).tolist(), [False, False])
        self.assertEqual(au.benjamini_hochberg([], 0.05).tolist(), [])


# ---------------------------------------------------------------------------------------------
# Synthetic shards.

# Binary features of the synthetic shards and how often each is on: card holdings, scores, games,
# flags, the dealer (indices of the real layout, so the names read as positions).
FEATURES = [(3, 0.12), (40, 0.3), (129, 0.2), (226, 0.1), (300, 0.5), (433, 0.4), (520, 0.25),
            (au.OFF_TRUMP + 1, 0.25), (au.OFF_SCORE + 9, 0.15), (au.OFF_SCORE + 15 + 12, 0.1),
            (au.OFF_SCORE + 30 + 4, 0.3), (au.OFF_FLAG + 2, 0.35), (au.OFF_TRICKS + 6, 0.4),
            (au.OFF_FIVE_PLAYED, 0.5), (au.OFF_END_BONUS + 2, 0.2), (au.OFF_GAMES + 0, 0.3),
            (au.OFF_GAMES + 3, 0.3), (au.OFF_DEALER + 1, 0.25)]
BIAS_PAIR = (40, au.OFF_GAMES + 3)  # on together in about 9% of rows
SPREAD_DENSE = 37  # dense feature 37: "after the leader's team then the leader's team win ..."


def synthetic(n, seed, rms=0.004, bias=None, spread=None, team_bias=0.0):
    """(Errors, binary, dense, truth): random binary features (``FEATURES``), dense features, and
    sum-zero Student-t(5) errors of pooled RMS ``rms``; ``bias`` = (features, size), or a list of
    them: the rows with all those features on get +size on team 1 and -size on team 3; ``spread`` =
    (dense feature, quantile, factor): rows at or below the quantile get their noise multiplied;
    ``team_bias``: +/- on teams 1/3 everywhere.
    ``truth``: per row, the expected mean square and the expected bias vector."""
    rng = np.random.default_rng(seed)
    binary = np.zeros((n, N_BINARY), dtype=bool)
    for f, p in FEATURES:
        binary[:, f] = rng.random(n) < p
    dense = rng.integers(-(1 << 24), 1, size=(n, N_DENSE))
    dense[:, 5] = rng.choice([-(1 << 24), -1453013, -887539, 0], size=n)  # few distinct values
    z = rng.standard_t(5, size=(n, 4)) / math.sqrt(5 / 3)
    z -= z.mean(axis=1, keepdims=True)  # each team's variance is now 3/4
    sigma = np.full(n, rms / math.sqrt(0.75))
    b = np.zeros((n, 4))
    b[:, 1] += team_bias
    b[:, 3] -= team_bias
    planted = {}
    if spread is not None:
        j, qtl, k = spread
        thr = int(np.quantile(dense[:, j], qtl, method="lower"))
        m = dense[:, j] <= thr
        sigma[m] *= k
        planted["spread"] = (m, thr)
    if bias is not None:
        for feats, size in [bias] if isinstance(bias[1], float) else bias:
            m = np.all(binary[:, list(feats)], axis=1)
            b[m, 1] += size
            b[m, 3] -= size
            planted.setdefault("bias", m)
    e = z * sigma[:, None] + b
    err = au.Errors.of(np.clip(np.rint(e * ONE), -ONE, ONE).astype(np.int64))
    truth = {"msq": sigma**2 * 0.75 + (b**2).mean(axis=1), "bias": b, "planted": planted}
    return err, binary, dense, truth


def run(err, binary, dense, seed=1, **kw):
    kw.setdefault("gamma", 0.005)
    is_b = au.half_b(seed, len(err))
    with contextlib.redirect_stdout(io.StringIO()):
        return au.audit(err, binary, dense, is_b, np.zeros(len(err), dtype=np.int64), log=lambda s: None, **kw)


def true_violation(group, binary, dense, truth, alpha_bias=0.0025, alpha_sd=0.01):
    m = au.members(group["conditions"], binary, dense)
    sd = math.sqrt(truth["msq"][m].mean())
    bias = np.abs(truth["bias"][m].mean(axis=0)).max()
    return sd > alpha_sd or bias > alpha_bias


class Planted(unittest.TestCase):
    def test_planted_groups_are_found_and_named(self):
        err, binary, dense, truth = synthetic(40000, 11, bias=(BIAS_PAIR, 0.006), spread=(SPREAD_DENSE, 0.04, 5.0))
        body = run(err, binary, dense)
        preds = body["predicates"]
        self.assertTrue(preds)
        # The bias pair, exactly, by the enumeration, named by both features.
        want = [{"feature": f, "op": "==", "value": 1} for f in sorted(BIAS_PAIR)]
        hits = [p for p in preds if p["conditions"] == want]
        self.assertEqual(len(hits), 1, [p["name"] for p in preds])
        g = hits[0]
        self.assertIn("bias", g["criteria"])
        self.assertIn(au.binary_name(40), g["name"])
        self.assertIn(au.binary_name(au.OFF_GAMES + 3), g["name"])
        self.assertIn("the next team has won a game", au.binary_name(au.OFF_GAMES + 1))
        self.assertAlmostEqual(g["bias"][1], 0.006, delta=0.0015)
        self.assertAlmostEqual(g["bias"][3], -0.006, delta=0.0015)
        # The spread group, by a dense threshold close to the planted one.
        m_true, thr = truth["planted"]["spread"]
        best = None
        for p in preds:
            m = au.members(p["conditions"], binary, dense)
            jac = (m & m_true).sum() / max((m | m_true).sum(), 1)
            if best is None or jac > best[0]:
                best = (jac, p)
        jac, p = best
        self.assertGreater(jac, 0.8, p["name"])
        self.assertIn("sd", p["criteria"])
        self.assertTrue(any(c["feature"] == N_BINARY + SPREAD_DENSE and c["op"] == "<=" for c in p["conditions"]))
        self.assertIn(au.dense_name(SPREAD_DENSE), p["name"])
        self.assertGreater(p["sd"], 0.015)
        # Nothing false: every confirmed group (general or not) really breaks a threshold.
        for grp in body["groups"]:
            self.assertTrue(true_violation(grp, binary, dense, truth), grp["name"])
        # Statistics recorded: weight, rows, an example row inside the group, p99.9.
        m = au.members(g["conditions"], binary, dense)
        self.assertEqual(g["rows"], int(m.sum()))
        self.assertAlmostEqual(g["weight"], m.mean())
        self.assertTrue(m[g["example_row"]])
        self.assertGreater(g["p999"], 0)

    def test_clean_shards_claim_nothing(self):
        for seed in (21, 22, 23, 24, 25):
            err, binary, dense, _ = synthetic(30000, seed, rms=0.005)
            body = run(err, binary, dense, seed=seed)
            self.assertEqual(body["predicates"], [], seed)
            self.assertEqual(body["confirmed"], 0)

    def test_false_discoveries_at_the_thresholds(self):
        """Every group exactly at both thresholds (the null's boundary): runs with any claim stay
        within the false-discovery rate (expected at most 5% of runs; 20 fixed runs, at most 3)."""
        claims = 0
        for seed in range(31, 51):
            # The pooled RMS includes the bias: noise for the rest.
            noise = math.sqrt(0.01**2 - 0.0025**2 / 2)
            err, binary, dense, truth = synthetic(16000, seed, rms=noise, team_bias=0.0025)
            body = run(err, binary, dense, seed=seed, gamma=0.01)
            claims += body["confirmed"] > 0
        self.assertLessEqual(claims, 3)

    def test_deterministic(self):
        err, binary, dense, _ = synthetic(20000, 12, bias=(BIAS_PAIR, 0.006), spread=(SPREAD_DENSE, 0.05, 4.0))
        a = run(err, binary, dense, seed=3)
        b = run(err, binary, dense, seed=3)
        a.pop("seconds"), b.pop("seconds")
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        c = run(err, binary, dense, seed=4)
        self.assertNotEqual(a["half_a_rows"], c["half_a_rows"])


# ---------------------------------------------------------------------------------------------
# The enumeration against a plain whole-number one.


def plain_enumeration(items, xbool, cols, screen, max_conj):
    """The candidates (frozenset of items -> (sd, bias)) by the rules of ``Enumeration``, with
    exact sums of boolean masks."""
    def sums(m):
        return np.array([au.sums_of(cols, m)], dtype=np.int64)

    found = {}
    s1 = np.array([au.sums_of(cols, xbool[:, i]) for i in range(len(items))], dtype=np.int64)
    c1, d1, b1 = screen(s1)
    for i in np.flatnonzero(c1):
        found[frozenset([i])] = (bool(d1[i]), bool(b1[i]))
    grow = [i for i in range(len(items)) if s1[i, 0] >= screen.minsup and not c1[i] and s1[i, 0] < len(xbool)]
    order = sorted(grow, key=lambda i: (s1[i, 0], i))
    pos = {i: k for k, i in enumerate(order)}
    grow2 = set()
    for a in order:
        for b in order:
            if pos[a] < pos[b]:
                s = sums(xbool[:, a] & xbool[:, b])
                c, d, bi = screen(s)
                if c[0]:
                    found[frozenset([a, b])] = (bool(d[0]), bool(bi[0]))
                elif s[0, 0] >= screen.minsup:
                    grow2.add((a, b))
    if max_conj < 3:
        return found
    for a in order:
        for b in order:
            for c in order:
                if pos[a] < pos[b] < pos[c] and (a, b) in grow2 and (a, c) in grow2 and (b, c) in grow2:
                    s = sums(xbool[:, a] & xbool[:, b] & xbool[:, c])
                    cc, d, bi = screen(s)
                    if cc[0]:
                        found[frozenset([a, b, c])] = (bool(d[0]), bool(bi[0]))
    return found


class EnumerationExact(unittest.TestCase):
    def check(self, screen_level, minsup, gamma=None):
        # A bias on a triple (8% of rows) that its pairs (20% or more) dilute below the threshold,
        # a pair's bias, a spread.
        triple = ((300, 433, au.OFF_TRICKS + 6), 0.005)
        err, binary, dense, _ = synthetic(6000, 13, rms=0.006, bias=[triple, (BIAS_PAIR, 0.004)],
                                          spread=(SPREAD_DENSE, 0.03, 2.5))
        items, x = au.enumeration_items(binary, minsup)
        cols = err.columns()[:, list(au.SCREEN_COLS)]
        scr = au.Screen(0.0025, 0.01, minsup, screen_level)
        old = au.GAMMA32
        try:
            if gamma is not None:
                au.GAMMA32 = gamma
            en = au.Enumeration(items, x, cols, scr, 3)
        finally:
            au.GAMMA32 = old
        got = {frozenset(t): (d, b) for t, d, b in en.found}
        want = plain_enumeration(items, x, cols, scr, 3)
        self.assertEqual(got, want)
        self.assertTrue(any(len(k) == 3 for k in want))
        self.assertTrue(any(len(k) == 2 for k in want))
        return en

    def test_bounded_float32_gives_the_exact_candidates(self):
        self.check(1.0, 30)
        self.check(0.2, 30)

    def test_undecided_sums_are_summed_again_exactly(self):
        en = self.check(1.0, 30, gamma=0.05)
        self.assertGreater(sum(en.rechecked), 100)


# ---------------------------------------------------------------------------------------------
# Predicates and the shared fixture.


def plain_member(conditions, active, dense):
    """A second evaluator, row by row: binary feature f is on when f is in the active list; dense
    feature 809 + j is compared as an integer."""
    act = set(int(a) for a in active if a != PAD)
    for c in conditions:
        f, op, v = c["feature"], c["op"], c["value"]
        if f < N_BINARY:
            if (f in act) != (v == 1):
                return False
        else:
            x = int(dense[f - N_BINARY])
            if not (x <= v if op == "<=" else x > v):
                return False
    return True


# The fixture's predicates: every kind of condition, inclusive and exclusive bounds at values the
# rows hold, an empty conjunction, an empty group, and conditions of the kind the auditor's trees
# confirmed on a real trick-4 shard (a team's money chance is nil if a given team wins the next two
# tricks: that team wins the match).
FIXTURE_PREDICATES = [
    {"name": "every position", "conditions": []},
    {"conditions": [{"feature": 443, "op": "==", "value": 1}]},
    {"conditions": [{"feature": au.OFF_GAMES + 1, "op": "==", "value": 0}]},
    {"conditions": [{"feature": 809, "op": "<=", "value": -1029656}]},
    {"conditions": [{"feature": 809, "op": ">", "value": -1029656}]},
    {"conditions": [{"feature": 703, "op": "==", "value": 1}, {"feature": au.OFF_FLAG + 3, "op": "==", "value": 0},
                    {"feature": 831, "op": ">", "value": -2697097}, {"feature": 831, "op": "<=", "value": -1088006}]},
    {"conditions": [{"feature": 814, "op": "<=", "value": -16777217}]},
    {"conditions": [{"feature": 814, "op": "<=", "value": -16777216}]},
    {"conditions": [{"feature": 846, "op": "<=", "value": -16777216},
                    {"feature": 906, "op": "<=", "value": -16777216}]},
    {"conditions": [{"feature": 886, "op": ">", "value": -16777216}, {"feature": 643, "op": "==", "value": 0},
                    {"feature": au.OFF_GAMES + 3, "op": "==", "value": 1}]},
]


def fixture_rows(path=FIXTURES / "predicate-rows.json"):
    """The fixture's rows as a label-shard record array (``predicate-rows.json``: per row the
    active binary features, ascending, without the padding; the 100 dense values; and the label
    fields: value, source, trick, label, mask, outcomes)."""
    d = json.loads(Path(path).read_text())
    v = len(d["rows"][0]["outcomes"])
    a = np.zeros(len(d["rows"]), dtype=shard_dtype(v))
    for i, r in enumerate(d["rows"]):
        a["active"][i, :] = PAD
        a["active"][i, : len(r["active"])] = r["active"]
        for k in ("dense", "value", "source", "trick", "label", "mask", "outcomes"):
            a[k][i] = r[k]
    return a


def build_fixture(source, out_dir=FIXTURES):
    """Writes the fixture from a trick-4 audit shard (run by hand; the files are committed):
    16 rows of race class 4 and 32 others from the shard's first 2,000, the predicates (named by
    ``group_name``) and their memberships."""
    from fortyone_nn.formats import race_classes

    a = load_shard(source)
    rc = race_classes(a[:2000])
    pick = np.sort(np.concatenate([np.flatnonzero(rc == 4)[:16], np.flatnonzero(rc != 4)[:32]]))
    rows = np.asarray(a[pick])
    os.makedirs(out_dir, exist_ok=True)
    out = {"encoder": "oracle-v2", "from": f"rows {pick.tolist()} of {os.path.basename(source)}", "rows": [
        {"active": [int(x) for x in r["active"] if x != PAD], "dense": r["dense"].tolist(),
         "value": r["value"].tolist(), "source": int(r["source"]), "trick": int(r["trick"]),
         "label": int(r["label"]), "mask": int(r["mask"]), "outcomes": r["outcomes"].tolist()}
        for r in rows]}
    with open(os.path.join(out_dir, "predicate-rows.json"), "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    rows = fixture_rows(os.path.join(out_dir, "predicate-rows.json"))
    preds = [{"name": au.group_name(p["conditions"]), "conditions": p["conditions"]} for p in FIXTURE_PREDICATES]
    with open(os.path.join(out_dir, "predicates.json"), "w") as fh:
        json.dump(preds, fh, indent=1)
    b = au.binary_matrix(rows["active"])
    d = rows["dense"].astype(np.int64)
    members = [np.flatnonzero(au.members(p["conditions"], b, d)).tolist() for p in preds]
    with open(os.path.join(out_dir, "predicate-members.json"), "w") as fh:
        json.dump({"rows_file": "predicate-rows.json", "predicates": "predicates.json", "rows": len(rows),
                   "members": members}, fh, indent=1)


class Predicates(unittest.TestCase):
    def test_fixture_memberships(self):
        rows = fixture_rows()
        preds = au.load_predicates(FIXTURES / "predicates.json")
        want = json.loads((FIXTURES / "predicate-members.json").read_text())
        self.assertEqual(want["rows"], len(rows))
        self.assertEqual([p["conditions"] for p in preds], [p["conditions"] for p in FIXTURE_PREDICATES])
        b = au.binary_matrix(rows["active"])
        d = rows["dense"].astype(np.int64)
        for p, w in zip(preds, want["members"]):
            self.assertEqual(np.flatnonzero(au.members(p["conditions"], b, d)).tolist(), w, p["name"])
            plain = [r for r in range(len(rows)) if plain_member(p["conditions"], rows["active"][r], rows["dense"][r])]
            self.assertEqual(plain, w, p["name"])
            self.assertEqual(p["name"], au.group_name(p["conditions"]))
        sizes = [len(w) for w in want["members"]]
        self.assertEqual(sizes[0], len(rows))
        self.assertEqual(sizes[3] + sizes[4], len(rows))  # "<=" and ">" at the same value
        self.assertEqual(sizes[6], 0)
        self.assertTrue(all(0 < s < len(rows) for s in sizes[1:6] + sizes[7:]), sizes)

    def test_format_checks(self):
        ok = [{"feature": 0, "op": "==", "value": 1}, {"feature": 808, "op": "==", "value": 0},
              {"feature": 809, "op": "<=", "value": -5}, {"feature": N_FEATURES - 1, "op": ">", "value": 0}]
        for c in ok:
            au.check_condition(c)
        bad = [{"feature": 809, "op": "==", "value": 1}, {"feature": 3, "op": "<=", "value": 1},
               {"feature": 3, "op": "==", "value": 2}, {"feature": N_FEATURES, "op": ">", "value": 0},
               {"feature": -1, "op": "==", "value": 1}, {"feature": 3, "op": "==", "value": True},
               {"feature": 900, "op": "<=", "value": -0.3}, {"feature": 900, "op": "<", "value": -3},
               {"feature": 3, "op": "==", "value": 1, "extra": 0}, {"feature": 900, "op": ">", "value": 1 << 31}]
        for c in bad:
            with self.assertRaises(ValueError, msg=c):
                au.check_condition(c)

    def test_implication(self):
        g = au.cond_key([{"feature": 5, "op": "==", "value": 1}, {"feature": 820, "op": "<=", "value": -9},
                         {"feature": 821, "op": ">", "value": -50}])
        for h, want in (([{"feature": 5, "op": "==", "value": 1}], True),
                        ([{"feature": 820, "op": "<=", "value": -9}], True),
                        ([{"feature": 820, "op": "<=", "value": -3}], True),
                        ([{"feature": 820, "op": "<=", "value": -10}], False),
                        ([{"feature": 821, "op": ">", "value": -60}], True),
                        ([{"feature": 821, "op": ">", "value": -40}], False),
                        ([{"feature": 5, "op": "==", "value": 0}], False),
                        ([{"feature": 6, "op": "==", "value": 1}], False),
                        ([], True)):
            self.assertEqual(au.implies(g, au.cond_key(h)), want, h)

    def test_round_trip(self):
        err, binary, dense, _ = synthetic(20000, 14, bias=(BIAS_PAIR, 0.006), spread=(SPREAD_DENSE, 0.05, 5.0))
        body = run(err, binary, dense)
        self.assertTrue(body["predicates"])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "predicates.json")
            with open(p, "w") as fh:
                json.dump(body["predicates"], fh)
            back = au.load_predicates(p)
        self.assertEqual(back, json.loads(json.dumps(body["predicates"])))
        for g in back:
            self.assertEqual(int(au.members(g["conditions"], binary, dense).sum()), g["rows"])

    def test_names(self):
        self.assertEqual(au.binary_name(0), "the leader holds the 5 of trumps")
        self.assertEqual(au.binary_name(8 * 4 + 6), "the 7th player holds the best live ordinary trump")
        self.assertEqual(au.binary_name(au.OFF_SCORE + 15 * 2 + 13), "the opposite team has 36 points")
        self.assertEqual(au.binary_name(au.OFF_GAMES + 2, False), "the opposite team has no game")
        self.assertEqual(au.dense_name(au.DENSE_NEXT2 + 16 * 2 + 4 * 2 + 1),
                         "the next team's money chance after the opposite team then the opposite team win the next two tricks")
        self.assertEqual(au.dense_chance(0), 1.0)
        self.assertEqual(au.dense_chance(-(1 << 19)), 0.5)
        for f in range(N_BINARY):
            self.assertTrue(au.binary_name(f) and au.binary_name(f, False))
        for j in range(N_DENSE):
            self.assertTrue(au.dense_name(j))


# ---------------------------------------------------------------------------------------------
# The command line.


class CommandLine(unittest.TestCase):
    def test_end_to_end(self):
        rows = fixture_rows()
        big = np.tile(rows, 60)
        with tempfile.TemporaryDirectory() as d:
            shard = os.path.join(d, "audit.npy")
            np.save(shard, big)
            for head, n_out in (("outcome-v1", VOCAB[4]), ("money4", 4)):
                net = os.path.join(d, f"{head}.safetensors")
                write_int(net, quantize(random_float_net(9, (32, 16), n_out)), 4, {"test": "audit"}, head=head)
                outs = []
                for k in range(2):
                    out = os.path.join(d, f"out-{head}-{k}")
                    with contextlib.redirect_stdout(io.StringIO()):
                        au.main(["--net", net, "--shard", shard, "--shard", shard, "--out", out, "--gamma", "0.02",
                                 "--max-conj", "2", "--seed", "5"])
                    outs.append(out)
                preds = [Path(o, "predicates.json").read_text() for o in outs]
                self.assertEqual(preds[0], preds[1])
                reps = [json.loads(Path(o, "report.json").read_text()) for o in outs]
                for r in reps:
                    r.pop("seconds")
                    r["audit"].pop("seconds")
                self.assertEqual(reps[0], reps[1])
                rep = reps[0]
                self.assertEqual(rep["audit"]["rows"], 2 * len(big))
                self.assertEqual(rep["net"]["head"], head)
                self.assertEqual(rep["thresholds"]["max_conj"], 2)
                self.assertEqual(rep["seed"], 5)
                self.assertIsInstance(json.loads(preds[0]), list)
                txt = Path(outs[0], "report.txt").read_text()
                self.assertIn("Rows audited: 5,760", txt)
                self.assertIn("Verdict:", txt)
                au.load_predicates(Path(outs[0], "predicates.json"))


if __name__ == "__main__":
    unittest.main()
