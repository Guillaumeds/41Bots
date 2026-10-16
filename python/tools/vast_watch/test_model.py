"""Tests for the watcher's cost model, categories and records: python3 -m unittest test_model (in this folder)."""
import json
import os
import tempfile
import unittest

import model as M


def offer(id, cpu, threads, ram, dph, machine=None, **kw):
    o = {"id": id, "machine_id": machine or id, "cpu_name": cpu, "threads": threads, "ram_gb": ram, "dph_total": dph,
         "reliability2": 0.995, "verification": "verified", "gpu_frac": 1.0, "num_gpus": 1, "gpu_name": "RTX 4090",
         "geolocation": "Norway, NO", "inet_down_cost": 0.01, "inet_up_cost": 0.01}
    o.update(kw)
    return o


class Cost(unittest.TestCase):
    def setUp(self):
        self.m = M.Model()

    def test_measured_9995wx_reproduces_the_bench(self):
        # Wave 1/3: 184 threads, 241 GB, $1.27/h measured $0.283-0.287 per 10^12 nodes.
        a = M.annotate(self.m, offer(1, "AMD Ryzen Threadripper PRO 9995WX 96-Cores", 192, 251, 1.2685))
        self.assertEqual(a["source"], "measured")
        self.assertAlmostEqual(a["usable"], 184.0, delta=0.5)
        self.assertAlmostEqual(a["cost"], 0.285, delta=0.012)

    def test_every_measured_machine_is_reproduced_within_8_percent(self):
        runs = {  # model, effective threads, RAM GB, $/h, measured $ per 10^12
            "AMD EPYC 9754 128-Core Emb Processor": (256, 503, 3.7366, 0.910),
            "AMD EPYC 9654P 96-Core Processor": (192, 251, 2.9361, 0.799),
            "AMD EPYC 9734 112-Core Processor": (224, 125, 3.3339, 0.779),
            "AMD EPYC 9554 64-Core Processor": (256, 504, 6.935, 1.134),
            "Xeon® E5-2695 v4 ": (72, 504, 0.4067, 0.519),
            "Xeon® Gold 5220R ": (96, 47, 0.699, 0.571),
        }
        for cpu, (thr, ram, dph, want) in runs.items():
            a = M.annotate(self.m, offer(1, cpu, thr, ram, dph))
            self.assertEqual(a["source"], "measured", cpu)
            self.assertAlmostEqual(a["cost"] / want, 1.0, delta=0.08, msg=f"{cpu}: {a['cost']:.3f} vs {want}")

    def test_unmeasured_chip_is_an_estimate(self):
        a = M.annotate(self.m, offer(1, "AMD EPYC 7J13 64-Core Processor", 128, 252, 1.23))
        self.assertEqual(a["source"], "estimated")
        self.assertTrue(0.4 < a["cost"] < 1.0, a["cost"])

    def test_small_tables_cost_more(self):
        big = M.annotate(self.m, offer(1, "AMD EPYC 7J13 64-Core Processor", 128, 512, 1.0))
        small = M.annotate(self.m, offer(2, "AMD EPYC 7J13 64-Core Processor", 128, 16, 1.0))
        self.assertGreater(small["cost"], big["cost"] * 1.5)
        self.assertAlmostEqual(M.table_time(448), 1.0)
        self.assertAlmostEqual(M.table_time(128), 1.158, places=3)
        self.assertGreater(M.table_time(16), M.table_time(32))

    def test_quirk_host_gives_half_its_threads(self):
        a = M.annotate(self.m, offer(1, "AMD Eng Sample: 100-000000894-04", 384, 251, 1.89, machine=137718))
        self.assertAlmostEqual(a["usable"], 192.0, delta=0.1)

    def test_launcher_rules(self):
        self.assertTrue(M.launcher_ok(offer(1, "x", 128, 100, 1.0)))
        self.assertFalse(M.launcher_ok(offer(1, "x", 128, 100, 1.0, gpu_frac=0.5)))
        self.assertFalse(M.launcher_ok(offer(1, "x", 128, 100, 1.0, verification="deverified")))
        self.assertTrue(M.launcher_ok(offer(1, "x", 128, 100, 1.0, reliability2=0.95)))   # floor is 0.90 (owner, 2026-10-16)
        self.assertFalse(M.launcher_ok(offer(1, "x", 128, 100, 1.0, reliability2=0.85)))
        self.assertFalse(M.launcher_ok(offer(1, "x", 128, 100, 1.0, inet_up_cost=0.2)))


class Categories(unittest.TestCase):
    def setUp(self):
        self.m = M.Model()
        self.offers = [M.annotate(self.m, o) for o in [
            offer(1, "AMD Ryzen Threadripper PRO 9995WX 96-Cores", 192, 251, 1.27, machine=10),
            offer(2, "AMD Ryzen Threadripper PRO 9995WX 96-Cores", 192, 251, 1.40, machine=11),
            offer(3, "AMD EPYC 9554 64-Core Processor", 256, 504, 6.94, machine=12),
            offer(4, "Xeon® E5-2695 v4 ", 72, 504, 0.41, machine=13),
            offer(5, "AMD EPYC 9754 128-Core Processor", 256, 503, 3.75, machine=14, gpu_frac=0.5),   # shared: out
            offer(6, "AMD EPYC 9654P 96-Core Processor", 192, 251, 2.94, machine=15),
            offer(7, "AMD EPYC 9654P 96-Core Processor", 192, 251, 2.94, machine=15),                 # same machine again
        ]]

    def test_ranking(self):
        c = M.live_categories(self.offers)
        self.assertEqual([r["id"] for r in c["cheapest"]], [1, 2])
        self.assertEqual(c["cheapest"][0]["machine_id"], 10)
        self.assertEqual([r["id"] for r in c["fast_cheap"]][0], 1)       # the E5 is too slow for this category
        self.assertNotIn(4, [r["id"] for r in c["fast_cheap"]])
        self.assertEqual(len(c["balanced"]), 2)
        for cat in c.values():
            self.assertEqual(len({r["machine_id"] for r in cat}), len(cat))  # never two listings of one machine
        self.assertNotIn(5, [r["id"] for k in c.values() for r in k])        # shared machines are never ranked

    def test_min_threads(self):
        c = M.live_categories(self.offers, min_threads=100)
        self.assertNotIn(4, [r["id"] for r in c["cheapest"]])


class Records(unittest.TestCase):
    def setUp(self):
        self.m = M.Model()
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "history.json")

    def cats(self, offers, **kw):
        return M.live_categories([M.annotate(self.m, o) for o in offers], **kw)

    def test_baseline_then_new_record_then_persist(self):
        h = M.History(self.path)
        ev = h.update(self.cats([offer(1, "AMD EPYC 9654P 96-Core Processor", 192, 251, 2.94, machine=15)]), now=100)
        self.assertTrue(ev and all(e["why"] == "baseline" for e in ev))
        # an hour later a much cheaper machine appears
        ev = h.update(self.cats([offer(1, "AMD EPYC 9654P 96-Core Processor", 192, 251, 2.94, machine=15),
                                 offer(2, "AMD Ryzen Threadripper PRO 9995WX 96-Cores", 192, 251, 1.27, machine=10)]), now=3700)
        cheap = [e for e in ev if e["category"] == "cheapest"]
        self.assertEqual(len(cheap), 1)
        self.assertEqual((cheap[0]["rank"], cheap[0]["why"], cheap[0]["record"]["id"]), (1, "new machine", 2))
        # the cheap machine is gone: the record stays, and a file survives a restart
        h.update(self.cats([offer(1, "AMD EPYC 9654P 96-Core Processor", 192, 251, 2.94, machine=15)]), now=3800)
        h2 = M.History(self.path)
        recs = h2.data["records"]["cheapest"]
        self.assertEqual([r["id"] for r in recs], [2, 1])
        view = h2.view({1})
        self.assertEqual([r["still_listed"] for r in view["records"]["cheapest"]], [False, True])

    def test_same_machine_cheaper_is_an_event_but_not_a_second_entry(self):
        h = M.History(self.path)
        h.update(self.cats([offer(1, "AMD EPYC 9654P 96-Core Processor", 192, 251, 2.94, machine=15)]), now=1)
        ev = h.update(self.cats([offer(9, "AMD EPYC 9654P 96-Core Processor", 192, 251, 2.00, machine=15)]), now=2)
        self.assertEqual([e["why"] for e in ev if e["category"] == "cheapest"], ["same machine, lower cost"])
        self.assertEqual(len(h.data["records"]["cheapest"]), 1)
        self.assertEqual(h.data["records"]["cheapest"][0]["id"], 9)

    def test_nothing_new_no_events(self):
        h = M.History(self.path)
        c = self.cats([offer(1, "AMD EPYC 9654P 96-Core Processor", 192, 251, 2.94, machine=15)])
        h.update(c, now=1)
        self.assertEqual(h.update(c, now=2), [])


if __name__ == "__main__":
    unittest.main()
