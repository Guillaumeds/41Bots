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
