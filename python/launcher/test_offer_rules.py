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
        self.assertEqual((R.MIN_RELIABILITY, R.MAX_NET_COST, R.MIN_INET_DOWN, R.SLOW_LINK_MBPS, R.MAX_DPH),
                         (0.90, 0.10, 100.0, 200.0, 1.70))
        self.assertEqual((R.MIN_COMPUTE_CAP, R.BLACKWELL_COMPUTE_CAP, R.MIN_CUDA, R.MIN_CUDA_BLACKWELL),
                         (600, 1000, 12.6, 13.0))
        self.assertFalse(hasattr(R, "MAX_COMPUTE_CAP"))   # no upper compute-capability limit any more (2026-10-17)

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
        # The floor is 100 Mbps (owner, 2026-10-16): 300 is rentable, 80 is not.
        self.assertEqual(R.search_failures(offer(inet_down=300.0)), [])
        self.assertEqual(R.search_failures(offer(inet_down=80.0)), ["download link 80 Mbps below 100"])
        self.assertEqual(R.search_failures(offer(inet_down=100.0)), [])
        self.assertEqual(R.search_failures(offer(inet_down=80.0), min_inet_down=50), [])
        self.assertEqual(R.search_failures(offer(inet_down=80.0), min_inet_down=0), [])
        self.assertEqual(R.search_failures(offer(inet_down=900.0), min_inet_down=1000), ["download link 900 Mbps below 1000"])
        o = offer()
        del o["inet_down"]
        self.assertEqual(R.search_failures(o), ["download speed (inet_down) not reported"])
        self.assertEqual(R.search_failures(o, min_inet_down=0), [])

    def test_price_cap(self):
        # $1.70 per hour by default (owner, 2026-10-16); a run can raise it, 0 means no cap.
        self.assertEqual(R.search_failures(offer(dph_total=1.70)), [])
        self.assertEqual(R.search_failures(offer(dph_total=1.90)), ["price $1.90/h above the cap $1.70/h"])
        self.assertEqual(R.search_failures(offer(dph_total=1.90), max_dph=2.5), [])
        self.assertEqual(R.search_failures(offer(dph_total=9.0), max_dph=0), [])
        self.assertEqual(R.search_failures(offer(dph_total=1.0), max_dph=0.9), ["price $1.00/h above the cap $0.90/h"])

    def test_network_cost_and_gpu_count(self):
        self.assertEqual(len(R.search_failures(offer(inet_up_cost=0.2))), 1)
        self.assertEqual(len(R.search_failures(offer(inet_down_cost=0.11))), 1)
        self.assertEqual(R.search_failures(offer(inet_down_cost=0.10)), [])
        self.assertEqual(len(R.search_failures(offer(num_gpus=0))), 1)

    def test_gpu_job_limits(self):
        gpu = dict(num_gpus=1, gpu_ram=6138, compute_cap=860, cuda_max_good=12.8)
        self.assertEqual(R.search_failures(offer(**gpu), gpus=1, min_gpu_ram_gb=4), [])
        for kw, word in (({"num_gpus": 0}, "0 GPUs, fewer than the 1 asked for"),
                         ({"compute_cap": 1200}, "compute capability 1200 (Blackwell) runs the PyTorch 2.14.1+cu130 build, "
                                                 "which needs a driver that supports CUDA 13.0; this host's supports 12.8"),
                         ({"compute_cap": 599}, "compute capability 599 below 600"), ({"cuda_max_good": 12.4}, "driver supports CUDA 12.4"),
                         ({"gpu_ram": 3000}, "GPU memory 3000 MB, below --min-gpu-ram-gb 4")):
            why = R.search_failures(offer(**{**gpu, **kw}), gpus=1, min_gpu_ram_gb=4)
            self.assertTrue(any(word in w for w in why), (kw, why))
        # At least the GPUs asked for: a 2-GPU machine is fine for a 1-GPU job (the price is per machine), 1 is not for 2.
        self.assertEqual(R.search_failures(offer(**{**gpu, "num_gpus": 2}), gpus=1, min_gpu_ram_gb=4), [])
        self.assertEqual(R.search_failures(offer(**{**gpu, "num_gpus": 1}), gpus=2, min_gpu_ram_gb=4),
                         ["1 GPUs, fewer than the 2 asked for"])
        # A CPU-only job leaves the compute-capability and CUDA limits out; a job that is not CPU-only keeps them.
        blackwell = offer(compute_cap=1200, cuda_max_good=12.0)
        self.assertEqual(R.search_failures(blackwell), [])
        self.assertEqual(len(R.search_failures(blackwell, cpu_only=False)), 1)   # the CUDA 12.6 floor (no second Blackwell line)

    def test_blackwell_cards_need_a_cuda_13_driver(self):
        # Owner, 2026-10-17: compute capability 10.0 and above (B200, RTX 50-series) is rentable, with a driver that
        # supports CUDA 13.0 (the PyTorch 2.14.1+cu130 build); every card below 10.0 is as before (cu126).
        gpu = dict(num_gpus=1, gpu_ram=16311, compute_cap=1200, cuda_max_good=13.1)
        for cc in (1000, 1030, 1200, 1210):
            for cuda, ok in ((13.1, True), (13.0, True), (12.9, False), (12.8, False), (12.6, False)):
                o = offer(**{**gpu, "compute_cap": cc, "cuda_max_good": cuda})
                self.assertEqual(R.search_failures(o, gpus=1, min_gpu_ram_gb=4) == [], ok, (cc, cuda))
                self.assertEqual(R.blackwell_driver_failure(o) is None, ok, (cc, cuda))
        why = R.search_failures(offer(**{**gpu, "cuda_max_good": 12.8}), gpus=1, min_gpu_ram_gb=4)
        self.assertEqual(why, ["compute capability 1200 (Blackwell) runs the PyTorch 2.14.1+cu130 build, which needs a "
                               "driver that supports CUDA 13.0; this host's supports 12.8"])
        # Below 10.0 nothing changed: cc 8.6 and 9.0 on a CUDA 12.6 driver are fine, 12.4 is not.
        for cc in (600, 750, 860, 890, 900):
            self.assertEqual(R.search_failures(offer(**{**gpu, "compute_cap": cc, "cuda_max_good": 12.6}), gpus=1), [], cc)
            self.assertIsNone(R.blackwell_driver_failure(offer(**{**gpu, "compute_cap": cc, "cuda_max_good": 12.6})))
            self.assertEqual(len(R.search_failures(offer(**{**gpu, "compute_cap": cc, "cuda_max_good": 12.4}), gpus=1)), 1)
        # An offer that reports no compute capability fails the generic check, never the Blackwell line; one that
        # reports a Blackwell card and no CUDA version fails the generic driver check once.
        self.assertIsNone(R.blackwell_driver_failure(offer(cuda_max_good=12.0)))
        no_cuda = R.search_failures(offer(**{**gpu, "cuda_max_good": None}), gpus=1)
        self.assertEqual(len(no_cuda), 1)
        self.assertIn("driver supports CUDA None, below 12.6", no_cuda[0])
        # A raised floor still applies below the Blackwell range.
        self.assertTrue(R.search_failures(offer(**{**gpu, "compute_cap": 860, "cuda_max_good": 12.8}), gpus=1, min_compute_cap=890))

    def test_gpu_names_accept_only_the_listed_models(self):
        # The names are spelled as vast reports them in an offer's gpu_name; case and runs of spaces do not matter.
        names = ["RTX 4090", "RTX PRO 6000 S"]
        for model in ("RTX 4090", "rtx 4090", "  RTX   4090 ", "RTX\t4090", "RTX PRO 6000 S", "rtx pro 6000 s"):
            self.assertEqual(R.gpu_name_failures(offer(gpu_name=model), names), [], model)
            self.assertEqual(R.search_failures(offer(gpu_name=model), gpu_names=names), [], model)
        # A different model, a longer or shorter name of the same family, or no model at all: refused, with the model named.
        for model in ("RTX 3090", "RTX 4090 D", "RTX PRO 6000 WS", "RTX PRO 6000", "RTX4090", "L40S"):
            self.assertEqual(R.gpu_name_failures(offer(gpu_name=model), names), [f"GPU {model} not in gpu_names"], model)
            self.assertEqual(R.search_failures(offer(gpu_name=model), gpu_names=names), [f"GPU {model} not in gpu_names"], model)
        for missing in (offer(), offer(gpu_name=None), offer(gpu_name=""), offer(gpu_name=4090)):
            self.assertEqual(R.gpu_name_failures(missing, names), ["GPU model not reported, and gpu_names is set"], missing)
        # The names given may differ in case and spacing from the offer's too.
        self.assertEqual(R.gpu_name_failures(offer(gpu_name="RTX 5090"), ["rtx  5090"]), [])
        # No list (None or empty): no filter, as before; the other checks still apply beside it.
        for none in (None, []):
            self.assertEqual(R.gpu_name_failures(offer(gpu_name="RTX 3090"), none), [])
            self.assertEqual(R.search_failures(offer(gpu_name="RTX 3090"), gpu_names=none), [])
        self.assertEqual(R.search_failures(offer(gpu_name="RTX 3090")), [])
        self.assertEqual(R.search_failures(offer(gpu_name="RTX 4090", dph_total=1.9), gpu_names=names),
                         ["price $1.90/h above the cap $1.70/h"])
        self.assertEqual(len(R.search_failures(offer(gpu_name="RTX 3090", dph_total=1.9), gpu_names=names)), 2)

    def test_link_factor(self):
        self.assertEqual([R.link_factor({"inet_down": v}) for v in (900, 500, 200, "850")], [1.0] * 4)
        self.assertAlmostEqual(R.link_factor({"inet_down": 100}), 1.5)
        self.assertAlmostEqual(R.link_factor({"inet_down": 0}), 2.0)
        self.assertEqual([R.link_factor(o) for o in ({}, {"inet_down": None}, {"inet_down": "n/a"})], [1.0] * 3)


if __name__ == "__main__":
    unittest.main()
