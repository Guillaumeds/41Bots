"""Network comparison (``fortyone_nn.netcmp``): the paired and sibling arithmetic, worked by hand.

Needs PyTorch (the module imports the trainer); skipped where only numpy is installed.
"""

import unittest

import numpy as np

try:
    from fortyone_nn import netcmp
    from fortyone_nn.formats import GROUPS_DTYPE, ONE
except ImportError:  # pragma: no cover
    netcmp = None


@unittest.skipIf(netcmp is None, "PyTorch not installed")
class NetCmp(unittest.TestCase):
    def test_summary_and_paired(self):
        e = np.array([[1.0, -1.0, 1.0, -1.0], [3.0, -3.0, 3.0, -3.0]])
        s = netcmp.summary(e)
        self.assertAlmostEqual(s["bias_pct"]["leader's team"], 2.0)
        self.assertAlmostEqual(s["rms_pct"], np.sqrt(5.0))
        p = netcmp.paired(e, e * 0)
        self.assertAlmostEqual(p["mean_sq_error_change"], 5.0)
        self.assertAlmostEqual(p["bias_change_pct"]["next team"], -2.0)
        same = netcmp.paired(e, e)
        self.assertEqual(same["mean_sq_error_change"], 0.0)

    def test_sibling_regret(self):
        # two groups; the mover (team 0) exact values 50/60/40 and 30/20; the net prefers row 0 and row 3
        exact = np.zeros((5, 4), dtype=np.uint64)
        for i, v in enumerate([50, 60, 40, 30, 20]):
            exact[i, 0] = int(ONE * v / 100)
        pred = np.zeros((5, 4))
        for i, v in enumerate([70, 60, 40, 35, 10]):
            pred[i, 0] = ONE * v / 100
        groups = np.zeros(5, dtype=GROUPS_DTYPE)
        groups["group"] = [0, 0, 0, 1, 1]
        r = netcmp.sibling_regret(pred, {"value": exact}, groups)
        self.assertEqual(r["groups"], 2)
        self.assertAlmostEqual(r["mean_regret_pct"], 5.0, places=4)  # group 0 regret 10, group 1 regret 0
        self.assertAlmostEqual(r["misordered_share"], 0.5)


if __name__ == "__main__":
    unittest.main()
