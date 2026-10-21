"""Label quality (``fortyone_nn.labelcmp``): hybrid minus exact labels per team, worked by hand.

numpy only. Run: ``cd python/nn && uv run python -m unittest discover -s tests -v``.
"""

import json
import os
import tempfile
import unittest

import numpy as np

from fortyone_nn import labelcmp
from fortyone_nn.formats import MAX_ACTIVE, ONE, PAD, VOCAB, shard_dtype


def shard(values):
    a = np.zeros(len(values), dtype=shard_dtype(VOCAB[3]))
    a["active"][:] = PAD
    for i in range(len(values)):
        a["active"][i, :2] = [i, 100 + i]
    a["value"] = np.array(values, dtype=np.uint64)
    return a


Q = ONE // 4


class LabelCmp(unittest.TestCase):
    def test_bias_and_spread_by_hand(self):
        exact = shard([[Q, Q, Q, Q], [Q, Q, Q, Q]])
        # Row 0: the leader's team 1% (of money chance) too low, the next 1% too high; row 1 exact.
        d = ONE // 100
        hybrid = shard([[Q - d, Q + d, Q, Q], [Q, Q, Q, Q]])
        s = labelcmp.summary(labelcmp.errors(exact, hybrid))
        self.assertEqual(s["rows"], 2)
        self.assertEqual(s["rows_exact"], 1)
        self.assertAlmostEqual(s["bias_pct"]["leader's team"], -0.5, places=6)
        self.assertAlmostEqual(s["bias_pct"]["next team"], 0.5, places=6)
        self.assertAlmostEqual(s["bias_pct"]["opposite team"], 0.0)
        self.assertAlmostEqual(s["sd_pct"]["leader's team"], 0.5, places=6)
        self.assertAlmostEqual(s["largest_abs_bias_pct"], 0.5, places=6)
        # rms over 8 team-values: two of 1%, so sqrt(2/8) = 0.5%.
        self.assertAlmostEqual(s["rms_pct"], 0.5, places=6)
        self.assertAlmostEqual(s["mean_abs_pct"], 0.25, places=6)

    def test_other_positions_are_refused(self):
        exact = shard([[Q, Q, Q, Q]])
        other = shard([[Q, Q, Q, Q]])
        other["active"][0, 0] = 7
        with self.assertRaises(SystemExit):
            labelcmp.errors(exact, other)
        with self.assertRaises(SystemExit):
            labelcmp.errors(exact, shard([[Q, Q, Q, Q]] * 2))

    def test_command_writes_the_report(self):
        with tempfile.TemporaryDirectory() as d:
            e, h, out = (os.path.join(d, n) for n in ("e.npy", "h.npy", "r.json"))
            np.save(e, shard([[Q, Q, Q, Q]] * 3))
            np.save(h, shard([[Q, Q, Q, Q]] * 3))
            labelcmp.main(["--exact", e, "--hybrid", h, "--out", out])
            with open(out) as f:
                rep = json.load(f)
            self.assertEqual(rep["all"]["rows"], 3)
            self.assertEqual(rep["all"]["rms_pct"], 0.0)
            self.assertIn("0", rep["by_race_class"])


if __name__ == "__main__":
    unittest.main()
