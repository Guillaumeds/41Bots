"""Tests of vast_market.py's ranking and speed-factor bookkeeping (no network).

    python3 -m unittest discover -s python/tools -p "test_*.py"
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vast_market as vm  # noqa: E402

SPEED = {
    "format": "41bots-vast-speed-factors", "version": 1, "quota_ratio": 0.96,
    "default": {"compute": 0.6, "tt": 0.5, "uncertainty": 0.5, "basis": "unknown"},
    "machines": {"7": {"usable_threads": 50, "note": "half quota"}},
    "models": [
        {"name": "Fast", "match": r"Fast\b", "arch": "x", "cores_per_socket": 64, "memory": "12 channels DDR5",
         "estimate": {"compute": 2.0, "tt": 1.0, "uncertainty": 0.15, "basis": "SPEC", "sources": []},
         "measured": None},
        {"name": "Slow", "match": r"Slow\b", "arch": "y", "cores_per_socket": 32, "memory": "8 channels DDR4",
         "estimate": {"compute": 1.0, "tt": 1.0, "uncertainty": 0.15, "basis": "SPEC", "sources": []},
         "measured": {"compute": 1.1, "runs": []}},
    ],
}


def offer(i, cpu, threads, dph, ram=512.0, machine=None, cores=None):
    return {"id": i, "machine_id": machine or 1000 + i, "cpu_name": cpu, "cpu_cores": cores or threads,
            "cpu_cores_effective": threads, "cpu_ram_gb": ram, "dph_total": dph, "gpu_frac": 1,
            "reliability2": 0.99, "geolocation": "X, XX", "query": "eligible"}


class VastMarketTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "speed.json")
        with open(self.path, "w") as f:
            json.dump(SPEED, f)
        self.sp = vm.load_speed(self.path)

    def tearDown(self):
        self.dir.cleanup()

    def test_reliability_floor_is_90_percent_like_the_launcher(self):
        # Owner, 2026-10-16: the launcher accepts hosts above 0.9; the market view labels offers the same way.
        q = vm.queries(16, 32)
        self.assertEqual(q["eligible"]["reliability"], {"gt": 0.90})
        self.assertEqual(q["reliability_0.80-0.90"]["reliability"], {"gt": 0.80, "lte": 0.90})
        self.assertNotIn("reliability_0.90-0.98", q)

    def test_search_follows_the_launchers_current_rules(self):
        # Owner, 2026-10-16: verified hosts are preferred, not required; the launcher also asks for a fast downlink.
        q = vm.queries(16, 32)
        e = q["eligible"]
        self.assertNotIn("verified", e)                      # every host is asked for; unverified ones are flagged
        self.assertEqual(e["inet_down"], {"gte": 100.0})     # the launcher's inet_down>=100 (offer_rules.MIN_INET_DOWN)
        self.assertEqual(e["dph_total"], {"lte": 1.7})       # and its price cap (offer_rules.MAX_DPH)
        self.assertEqual(e["reliability"], {"gt": 0.9})
        self.assertEqual(e["inet_down_cost"], {"lte": 0.10})
        self.assertEqual(e["num_gpus"], {"gte": 1})
        self.assertEqual(e["gpu_frac"], {"eq": 1})
        self.assertNotIn("unverified", q)                    # no longer a reason to leave an offer out
        self.assertEqual(q["slow_link"]["inet_down"], {"lt": 100.0})
        self.assertEqual({k: v for k, v in q["slow_link"].items() if k != "inet_down"},
                         {k: v for k, v in e.items() if k != "inet_down"})
        self.assertEqual(q["price_over_cap"]["dph_total"], {"gt": 1.7})     # kept visible as "not rentable"
        # The value passed on the command line replaces the default; 0 removes the filter (and the "slow" search).
        self.assertEqual(vm.queries(16, 32, 300)["eligible"]["inet_down"], {"gte": 300.0})
        self.assertNotIn("inet_down", vm.queries(16, 32, 0)["eligible"])
        self.assertNotIn("slow_link", vm.queries(16, 32, 0))
        self.assertEqual(vm.queries(16, 32, None, 2.5)["eligible"]["dph_total"], {"lte": 2.5})
        self.assertNotIn("dph_total", vm.queries(16, 32, None, 0)["eligible"])
        self.assertNotIn("price_over_cap", vm.queries(16, 32, None, 0))

    def test_thresholds_are_the_launchers_own(self):
        import offer_rules
        self.assertIs(vm.R, offer_rules)
        self.assertEqual((vm.MIN_RELIABILITY, vm.NET_CAP, vm.MIN_INET_DOWN),
                         (offer_rules.MIN_RELIABILITY, offer_rules.MAX_NET_COST, offer_rules.MIN_INET_DOWN))

    def test_unverified_eligible_offers_are_flagged_in_the_report(self):
        eligible = {"reliability": {"gt": 0.9}, "inet_down": {"gte": 100.0}, "dph_total": {"lte": 1.7}, "cpu_cores_effective": {"gte": 16}}
        offers = [dict(offer(1, "Fast", 128, 1.0), verification="verified", inet_down=900.0),
                  dict(offer(2, "Fast", 128, 1.2), verification="unverified", inet_down=900.0)]
        snap = {"utc": "t", "queries": {"eligible": eligible}, "offers": offers}
        out = io.StringIO()
        a = type("A", (), dict(markdown=False, min_cores=16, premium=0.2, top=5, t1_min_ram_gb=480, t1_min_gb_per_thread=1.0,
                               t1_table_gb=64, include_other=False))()
        text = vm.report(snap, self.sp, a)
        self.assertIn("any host (verified preferred, unverified ones flagged)", text)
        self.assertIn("download >= 100 Mbps, price <= $1.70/h", text)
        flagged = [ln for ln in text.splitlines() if "unverified host" in ln and ln.lstrip().startswith("2 ")]
        self.assertTrue(flagged, text)
        self.assertFalse([ln for ln in text.splitlines() if ln.lstrip().startswith("1 ") and "unverified host" in ln])
        # An old snapshot (asked for verified hosts only, no download filter) is described as it was made.
        old = {"utc": "t", "queries": {"eligible": {"verified": {"eq": True}, "reliability": {"gt": 0.9}}},
               "offers": [offer(1, "Fast", 128, 1.0)]}
        old_text = vm.report(old, self.sp, a)
        self.assertIn("verified host, reliability > 0.9, network", old_text)
        self.assertNotIn("download >=", old_text)

    def test_factor_precedence(self):
        self.assertEqual(vm.factor(self.sp, "AMD Fast 64-Core", "compute")[:2], (2.0, "estimate"))
        self.assertEqual(vm.factor(self.sp, "Slow chip", "compute")[:2], (1.1, "measured"))
        self.assertEqual(vm.factor(self.sp, "Slow chip", "tt")[:2], (1.0, "estimate"))  # measured has no tt
        self.assertEqual(vm.factor(self.sp, "Mystery", "compute")[:2], (0.6, "default"))

    def test_threads_quota_and_machine_quirk(self):
        a = vm.annotate(offer(1, "Fast", 100, 1.0), self.sp)
        self.assertAlmostEqual(a["_threads"], 96.0)
        self.assertAlmostEqual(a["_compute_cost"], 1.0 / (96 * 2.0) * 1000)
        b = vm.annotate(offer(2, "Fast", 100, 1.0, machine=7), self.sp)
        self.assertEqual(b["_threads"], 50.0)

    def test_owner_rule(self):
        offers = [vm.annotate(o, self.sp) for o in (
            offer(1, "Slow", 100, 0.50),   # 0.5/(96*1.1): the cheapest per work
            offer(2, "Fast", 400, 8.00),   # 8/(384*2): 2.2x the cheapest, too dear
            offer(3, "Fast", 300, 3.00),   # 3/(288*2): 1.10x the cheapest, and the most work per hour
            offer(4, "Slow", 50, 0.80),    # 0.8/(48*1.1): 3.2x the cheapest
        )]
        ranked, near = vm.rank(offers, "compute", 0.20)
        self.assertEqual(ranked[0]["id"], 1)
        self.assertEqual([o["id"] for o in near], [3, 1])

    def test_trick1_ram_filter_and_memory_flag(self):
        ok = vm.annotate(offer(1, "Fast", 256, 5.0, ram=1000), self.sp)
        small = vm.annotate(offer(2, "Fast", 256, 5.0, ram=200), self.sp)
        self.assertTrue(vm.t1_ok(ok, 480, 1.0))
        self.assertFalse(vm.t1_ok(small, 480, 1.0))
        # 256 threads of a 64-core part = 2 sockets x 12 channels x 16 GB = 384 GB for every channel filled.
        self.assertEqual(ok["_mem_flag"], "")
        self.assertNotEqual(small["_mem_flag"], "")

    def test_record_writes_measured_and_report_prints(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            vm.main(["record", "--cpu", "AMD Fast 64-Core Processor", "--compute", "1.7", "--run", "b1",
                     "--file", self.path])
            vm.main(["record", "--cpu", "Brand New 9999", "--tt", "0.8", "--run", "b2", "--file", self.path])
        sp = vm.load_speed(self.path)
        self.assertEqual(vm.factor(sp, "Fast", "compute")[:2], (1.7, "measured"))
        self.assertEqual(vm.factor(sp, "Fast", "tt")[:2], (1.0, "estimate"))
        self.assertEqual(vm.factor(sp, "Brand New 9999", "tt")[:2], (0.8, "measured"))
        snap = {"utc": "t", "queries": {"eligible": {}}, "offers": [offer(1, "Fast", 128, 1.0)]}
        snap_path = os.path.join(self.dir.name, "snap.json")
        with open(snap_path, "w") as f:
            json.dump(snap, f)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            vm.main(["--speed", self.path, "--from", snap_path, "--markdown"])
        self.assertIn("| 1 | Fast |", out.getvalue())


if __name__ == "__main__":
    unittest.main()
