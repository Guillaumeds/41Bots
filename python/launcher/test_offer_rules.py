"""Tests of offer_rules.py, the launcher's rules for which offers it may rent (shared with the market tools).

    python3 -m unittest discover -s python/launcher -p "test_*.py"
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import offer_rules as R  # noqa: E402


def offer(**kw) -> dict:
    o = {"id": 1, "reliability2": 0.99, "verification": "verified", "gpu_frac": 1.0, "num_gpus": 1, "is_bid": False,
         "rentable": True, "inet_down": 900.0, "inet_down_cost": 0.01, "inet_up_cost": 0.01}
    o.update(kw)
    return o


class OfferRulesTest(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual((R.MIN_RELIABILITY, R.MAX_NET_COST, R.MIN_INET_DOWN, R.SLOW_LINK_MBPS), (0.90, 0.10, 500.0, 200.0))
        self.assertEqual((R.MIN_COMPUTE_CAP, R.MAX_COMPUTE_CAP, R.MIN_CUDA), (600, 1000, 12.6))

    def test_a_good_offer_passes_every_check(self):
        self.assertEqual(R.check_failures(offer()), [])
        self.assertEqual(R.search_failures(offer()), [])

    def test_reliability_floor_is_inclusive(self):
        for rel, ok in ((0.89, False), (0.8999, False), (0.90, True), (0.95, True)):
            self.assertEqual(R.check_failures(offer(reliability2=rel)) == [], ok, rel)
        self.assertEqual(R.check_failures(offer(reliability2=0.85)), ["reliability 0.850 below 0.9"])
        # reliability2 missing: the older `reliability` field is used; neither: 0.
        self.assertEqual(R.check_failures(offer(reliability2=None, reliability=0.95)), [])
        self.assertEqual(R.check_failures(offer(reliability2=None)), ["reliability 0.000 below 0.9"])

    def test_whole_machine_on_demand_rentable(self):
        self.assertIn("shared machine (gpu_frac 0.5)", R.check_failures(offer(gpu_frac=0.5))[0])
        self.assertIn("interruptible", R.check_failures(offer(is_bid=True))[0])
        self.assertEqual(R.check_failures(offer(rentable=False)), ["not rentable"])

    def test_verified_hosts_are_preferred_not_required(self):
        for word in ("unverified", "deverified"):
            o = offer(verification=word)
            self.assertEqual(R.check_failures(o), [])
            self.assertEqual((R.is_verified(o), R.host_status(o), R.host_flag(o)), (False, word, word))
            self.assertIn("host not verified", R.check_failures(o, verified_only=True)[0])
        v = offer()
        self.assertEqual((R.is_verified(v), R.host_flag(v)), (True, ""))
        self.assertTrue(R.is_verified({"verified": True}))
        self.assertEqual(R.host_status({"verified": False}), "unverified")
        self.assertEqual(R.host_status({}), "unknown")

    def test_download_speed(self):
        self.assertEqual(R.search_failures(offer(inet_down=300.0)), ["download link 300 Mbps below 500"])
        self.assertEqual(R.search_failures(offer(inet_down=500.0)), [])
        self.assertEqual(R.search_failures(offer(inet_down=300.0), min_inet_down=250), [])
        self.assertEqual(R.search_failures(offer(inet_down=300.0), min_inet_down=0), [])
        self.assertEqual(R.search_failures(offer(inet_down=900.0), min_inet_down=1000), ["download link 900 Mbps below 1000"])
        o = offer()
        del o["inet_down"]
        self.assertEqual(R.search_failures(o), ["download speed (inet_down) not reported"])
        self.assertEqual(R.search_failures(o, min_inet_down=0), [])

    def test_network_cost_and_gpu_count(self):
        self.assertEqual(len(R.search_failures(offer(inet_up_cost=0.2))), 1)
        self.assertEqual(len(R.search_failures(offer(inet_down_cost=0.11))), 1)
        self.assertEqual(R.search_failures(offer(inet_down_cost=0.10)), [])
        self.assertEqual(len(R.search_failures(offer(num_gpus=0))), 1)

    def test_gpu_job_limits(self):
        gpu = dict(num_gpus=1, gpu_ram=6138, compute_cap=860, cuda_max_good=12.8)
        self.assertEqual(R.search_failures(offer(**gpu), gpus=1, min_gpu_ram_gb=4), [])
        for kw, word in (({"num_gpus": 2}, "2 GPUs, not the 1 asked for"), ({"compute_cap": 1200}, "compute capability 1200 outside 600..999"),
                         ({"compute_cap": 599}, "compute capability 599"), ({"cuda_max_good": 12.4}, "driver supports CUDA 12.4"),
                         ({"gpu_ram": 3000}, "GPU memory 3000 MB, below --min-gpu-ram-gb 4")):
            why = R.search_failures(offer(**{**gpu, **kw}), gpus=1, min_gpu_ram_gb=4)
            self.assertTrue(any(word in w for w in why), (kw, why))
        # A CPU-only job leaves the compute-capability and CUDA limits out; a job that is not CPU-only keeps them.
        blackwell = offer(compute_cap=1200, cuda_max_good=12.0)
        self.assertEqual(R.search_failures(blackwell), [])
        self.assertEqual(len(R.search_failures(blackwell, cpu_only=False)), 2)

    def test_link_factor(self):
        self.assertEqual([R.link_factor({"inet_down": v}) for v in (900, 500, 200, "850")], [1.0] * 4)
        self.assertAlmostEqual(R.link_factor({"inet_down": 100}), 1.5)
        self.assertAlmostEqual(R.link_factor({"inet_down": 0}), 2.0)
        self.assertEqual([R.link_factor(o) for o in ({}, {"inet_down": None}, {"inet_down": "n/a"})], [1.0] * 3)


if __name__ == "__main__":
    unittest.main()
