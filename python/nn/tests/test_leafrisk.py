"""The leaf-risk predictor (``fortyone_nn.leafrisk``, docs/m3-hybrid/rung2.md "Learned exact
leaves").

numpy only. Run: ``cd python/nn && uv run python -m unittest discover -s tests -v``.

- whole-number squared errors against Python's big integers;
- the extras of a hand-made case (the same case as Rust's ``extras_by_hand``);
- the trees: a planted rule (a binary feature and an extra together) is found first, ranked
  first by importance, and lowers the held-out error; ``predict`` is the base plus the leaves;
- the thresholds on toy cases worked by hand: conformal risk control's expectation form, Learn
  then Test's fixed-sequence walk (it stops at the first threshold that fails), the bounds;
  ``risk_coverage`` and the budget table;
- the decision units (runs of rows with the same match);
- the shared fixture (``tests/fixtures/leafrisk-fixture.json``: 32 real trick-3 harvest rows with
  O_3's values, a model's first trees, and the expected extras and scores, which Rust's
  ``nn::leafrisk`` also tests against).
"""

import json
import math
import unittest
from pathlib import Path

import numpy as np

from fortyone_nn import leafrisk as lr
from fortyone_nn.formats import MAX_ACTIVE, N_BINARY, N_DENSE, OFF_GAMES, PAD

FIXTURE = Path(__file__).parent / "fixtures" / "leafrisk-fixture.json"
ONE = 1 << 32


def rows(active_lists, money, mask, vecs, dense=None):
    n = len(active_lists)
    act = np.full((n, MAX_ACTIVE), PAD, dtype=np.uint16)
    for i, a in enumerate(active_lists):
        act[i, : len(a)] = sorted(a)
    d = np.zeros((n, N_DENSE), dtype=np.int32) if dense is None else np.asarray(dense, dtype=np.int32)
    return act, d, np.asarray(mask, dtype=np.uint64), np.asarray(vecs, dtype=np.uint64), np.asarray(money, dtype=np.uint64)


class Errors(unittest.TestCase):
    def test_sse_is_whole_number_exact(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            a = rng.integers(0, ONE + 1, size=4, dtype=np.uint64)
            b = rng.integers(0, ONE + 1, size=4, dtype=np.uint64)
            want = sum((int(x) - int(y)) ** 2 for x, y in zip(a, b)) >> 24
            self.assertEqual(int(lr.sse_units(a[None, :], b[None, :])[0]), want)
        # The largest error: two certain winners that differ.
        self.assertEqual(int(lr.sse_units([[ONE, 0, 0, 0]], [[0, ONE, 0, 0]])[0]), 2 << 40)


class Extras(unittest.TestCase):
    def test_by_hand(self):
        q = ONE // 4
        vecs = [[ONE, 0, 0, 0], [0, ONE, 0, 0], [0, 0, ONE, 0], [0, 0, 0, ONE], [ONE // 2, ONE // 2, 0, 0], [q] * 4]
        act, _, mask, v, m = rows([[]], [[2 * q, q, q, 0]], [0b100011], [vecs])
        e = lr.extras(act, mask, v, m)[0]
        self.assertEqual(int(e[0]), 2)
        self.assertEqual([int(x) for x in e[1:5]], [2 * q, q, q, 0])
        self.assertEqual((int(e[5]), int(e[6])), (2 * q, q))
        self.assertEqual(int(e[7]), 5 << 28)
        self.assertEqual((int(e[8]), int(e[9])), (2 * q, 6 * q))
        self.assertEqual((int(e[10]), int(e[11]), int(e[12])), (3, 2, ONE))
        act, _, mask, v, m = rows([[OFF_GAMES]], [[q] * 4], [0b11], [[[ONE, 0, 0, 0], [0, ONE, 0, 0]]])
        self.assertEqual(int(lr.extras(act, mask, v, m)[0][0]), 4)


def synthetic(n, seed):
    """Rows whose error is large exactly when binary feature 7 is on and the race class is 4."""
    rng = np.random.default_rng(seed)
    act = np.full((n, MAX_ACTIVE), PAD, dtype=np.uint16)
    on7 = rng.random(n) < 0.3
    for i in range(n):
        f = sorted(set(rng.choice(np.arange(10, 700), size=20, replace=False).tolist()) | ({7} if on7[i] else set()))
        act[i, : len(f)] = f
    dense = rng.integers(-(1 << 24), 0, size=(n, N_DENSE)).astype(np.int32)
    ex = np.zeros((n, lr.N_EXTRAS), dtype=np.int64)
    ex[:, 0] = rng.integers(0, 5, size=n)
    ex[:, 5] = rng.integers(ONE // 4, ONE, size=n)
    hot = on7 & (ex[:, 0] == 4)
    sse = np.where(hot, 40_000_000, 1_000_000) + rng.integers(0, 2_000_000, size=n)
    return {"active": act, "dense": dense, "extras": ex, "sse": sse.astype(np.int64), "unit": np.arange(n) // 8}, hot


class Trees(unittest.TestCase):
    def test_planted_rule(self):
        d, hot = synthetic(6000, 1)
        logs = []
        m = lr.fit(d, trees=30, depth=3, rate=0.3, min_leaf=50, subsample=0.8, seed=2, holdout=0.2, log=logs.append)
        top = [x["input"] for x in m["importance"][:2]]
        self.assertEqual(sorted(top), sorted([7, lr.N_FEATURES + 0]))
        curve = m["training"]["held_out_mse_curve"]
        self.assertLess(min(curve), 0.2 * curve[0])
        s = lr.predict(m, d)
        self.assertEqual(s.dtype, np.int64)
        self.assertTrue((s >= 0).all())
        self.assertGreater(s[hot].min(), s[~hot].max())
        # The base plus the leaves, by hand for one row.
        x = lr.column_values(d)
        row = 0
        on = set(int(f) for f in d["active"][row] if f != PAD)
        total = m["base"]
        for t in m["trees"]:
            nd = 0
            while t["f"][nd] >= 0:
                f = t["f"][nd]
                v = int(f in on) if f < N_BINARY else int(x[row, f - N_BINARY])
                nd = t["l"][nd] if v <= t["t"][nd] else t["r"][nd]
            total += t["v"][nd]
        self.assertEqual(int(s[row]), max(0, total))
        # Thresholds are whole numbers; every child comes after its parent.
        for t in m["trees"]:
            for i, f in enumerate(t["f"]):
                self.assertIsInstance(t["t"][i], int)
                if f >= 0:
                    self.assertTrue(t["l"][i] > i and t["r"][i] > i)


class Thresholds(unittest.TestCase):
    # Four decisions of one position each; losses 0.1 .. 0.4 at scores 1 .. 4.
    score = np.array([1, 2, 3, 4], dtype=np.int64)
    sse = (np.array([0.1, 0.2, 0.3, 0.4]) * (1 << 40)).astype(np.int64)
    unit = np.arange(4)
    grid = np.arange(6)

    def test_unit_losses(self):
        risk = [float(lr.unit_losses(self.score, self.sse, self.unit, t).mean()) for t in self.grid]
        for got, want in zip(risk, [0, 0, 0.025, 0.075, 0.15, 0.25]):
            self.assertAlmostEqual(got, want, places=9)
        # Two positions in one decision: their mean, then the mean over decisions.
        x = lr.unit_losses(self.score, self.sse, np.array([0, 0, 1, 1]), 5)
        self.assertTrue(np.allclose(x, [0.15, 0.35]))

    def test_conformal_risk_control(self):
        # (4 R + 0.4) / 5 <= 0.2 holds up to t = 4 (exactly 0.2), not at t = 5 (0.28).
        self.assertEqual(lr.crc_threshold(self.score, self.sse, self.unit, 0.2, self.grid, bound=0.4), 4)
        self.assertEqual(lr.crc_threshold(self.score, self.sse, self.unit, 0.13, self.grid, bound=0.4), 2)
        self.assertEqual(lr.crc_threshold(self.score, self.sse, self.unit, 0.01, self.grid, bound=0.4), None)

    def test_bounds(self):
        x = np.array([0.1, 0.2, 0.3, 0.4])
        m, v, n = 0.25, float(np.var(x, ddof=1)), 4
        lg = math.log(2 / 0.1)
        self.assertAlmostEqual(lr.ucb(x, 0.1, "bernstein", 1.0), m + math.sqrt(2 * v * lg / n) + 7 * lg / (3 * (n - 1)))
        self.assertAlmostEqual(lr.ucb(x, 0.1, "clt"), m + 1.2815515655446004 * math.sqrt(v / n), places=6)
        self.assertEqual(lr.ucb(x[:1], 0.1, "clt"), math.inf)

    def test_learn_then_test_walks_up_and_stops(self):
        rng = np.random.default_rng(5)
        n = 4000
        score = rng.integers(0, 1000, size=n)
        sse = ((score + rng.integers(0, 200, size=n)) * 1_000_000).astype(np.int64)
        unit = np.arange(n)
        grid = np.arange(0, 1001, 50)
        alpha = 4e-5
        t = lr.ltt_threshold(score, sse, unit, alpha, 0.05, "clt", grid)
        # Every threshold up to t passes, the next fails.
        for g in grid[grid <= t]:
            self.assertLessEqual(lr.ucb(lr.unit_losses(score, sse, unit, g), 0.05, "clt"), alpha)
        nxt = grid[grid > t][0]
        self.assertGreater(lr.ucb(lr.unit_losses(score, sse, unit, nxt), 0.05, "clt"), alpha)
        # A smaller delta (more confidence) never keeps a larger threshold; Bernstein is the most
        # careful.
        self.assertLessEqual(lr.ltt_threshold(score, sse, unit, alpha, 0.001, "clt", grid), t)
        tb = lr.ltt_threshold(score, sse, unit, alpha, 0.05, "bernstein", grid)
        self.assertTrue(tb is None or tb <= t)
        # A walk that would pass again later still stops at the first failure.
        bumpy = np.array([0, 1, 2, 3, 4, 5, 0], dtype=np.int64)
        self.assertEqual(lr.ltt_threshold(self.score, self.sse, self.unit, 0.03, 0.5, "clt", bumpy), 2)

    def test_coverage_and_budget(self):
        c = lr.risk_coverage(self.score, self.sse, shares=[0, 0.5, 1])
        self.assertAlmostEqual(c[0]["error_left"], 1.0)
        self.assertAlmostEqual(c[1]["error_left"], 0.3)
        self.assertAlmostEqual(c[2]["error_left"], 0.0)
        self.assertAlmostEqual(c[0]["sd_left"], math.sqrt(1.0 / 16))
        self.assertEqual(lr.quantile_table(self.score, 4), [4, 3, 2, 1])
        self.assertEqual(lr.quantile_table(np.array([5, 5, 1, 9]), 2), [5, 1])


class Units(unittest.TestCase):
    def test_runs_of_one_match(self):
        a = lr.audit
        base = [a.OFF_TRUMP + 1, a.OFF_SCORE + 3, a.OFF_SCORE + a.N_SCORES + 0, OFF_GAMES + 1, a.OFF_DEALER + 2]
        # The same match seen from another leader (teams rotate) and after other tricks (scores
        # differ): the same unit.
        rot = [a.OFF_TRUMP + 1, a.OFF_SCORE + 5, OFF_GAMES + 3, a.OFF_DEALER + 0]
        other = [a.OFF_TRUMP + 2, a.OFF_SCORE + 3, OFF_GAMES + 1, a.OFF_DEALER + 2]
        nogame = [a.OFF_TRUMP + 2, a.OFF_SCORE + 3, a.OFF_DEALER + 2]
        act, *_ = rows([base, base + [3], rot, other, nogame, base], [[ONE, 0, 0, 0]] * 6, [1] * 6, [[[ONE, 0, 0, 0]]] * 6)
        self.assertEqual(lr.decision_runs(act).tolist(), [0, 0, 0, 1, 2, 3])


@unittest.skipUnless(FIXTURE.exists(), "the shared fixture")
class Fixture(unittest.TestCase):
    def test_shared_fixture(self):
        fx = json.loads(FIXTURE.read_text())
        r = fx["rows"]
        act = np.full((len(r), MAX_ACTIVE), PAD, dtype=np.uint16)
        for i, x in enumerate(r):
            act[i, : len(x["active"])] = x["active"]
        dense = np.array([x["dense"] for x in r], dtype=np.int32)
        mask = np.array([x["mask"] for x in r], dtype=np.uint64)
        vecs = np.array([x["outcomes"] for x in r], dtype=np.uint64)
        money = np.array([x["net"] for x in r], dtype=np.uint64)
        ex = lr.extras(act, mask, vecs, money)
        self.assertEqual(ex.tolist(), [x["extras"] for x in r])
        s = lr.predict(fx["model"], {"active": act, "dense": dense, "extras": ex})
        self.assertEqual(s.tolist(), [x["score"] for x in r])
        self.assertEqual(lr.sse_units(money, [x["value"] for x in r]).tolist(), [x["sse"] for x in r])


if __name__ == "__main__":
    unittest.main()
