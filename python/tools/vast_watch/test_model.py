"""Tests for the watcher's cost model, categories and records: python3 -m unittest test_model (in this folder)."""
import json
import os
import tempfile
import unittest

import model as M


def offer(id, cpu, threads, ram, dph, machine=None, **kw):
    o = {"id": id, "machine_id": machine or id, "cpu_name": cpu, "threads": threads, "ram_gb": ram, "dph_total": dph,
         "reliability2": 0.995, "verification": "verified", "gpu_frac": 1.0, "num_gpus": 1, "gpu_name": "RTX 4090",
         "geolocation": "Norway, NO", "inet_down_cost": 0.01, "inet_up_cost": 0.01, "inet_down": 900.0}
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
        self.assertTrue(M.launcher_ok(offer(1, "x", 128, 100, 1.0, reliability2=0.95)))   # floor is 0.90 (owner, 2026-10-16)
        self.assertFalse(M.launcher_ok(offer(1, "x", 128, 100, 1.0, reliability2=0.85)))
        self.assertFalse(M.launcher_ok(offer(1, "x", 128, 100, 1.0, inet_up_cost=0.2)))
        self.assertFalse(M.launcher_ok(offer(1, "x", 128, 100, 1.0, inet_down_cost=0.2)))

    def test_reliability_floor_is_0_90_inclusive(self):
        for rel, ok in ((0.89, False), (0.8999, False), (0.90, True), (0.91, True), (0.995, True)):
            self.assertEqual(M.launcher_ok(offer(1, "x", 128, 100, 1.0, reliability2=rel)), ok, rel)
        why = M.launcher_failures(offer(1, "x", 128, 100, 1.0, reliability2=0.89))
        self.assertEqual(why, ["reliability 0.890 below 0.9"])

    def test_a_verified_host_with_a_slow_link_is_not_rentable(self):
        o = offer(1, "x", 128, 100, 1.0, verification="verified", inet_down=300.0)
        self.assertFalse(M.launcher_ok(o))
        self.assertEqual(M.launcher_failures(o), ["download link 300 Mbps below 500"])
        a = M.annotate(self.m, o)
        self.assertFalse(a["launcher_ok"])
        self.assertIn("download link 300 Mbps below 500", a["launcher_why"])
        # Exactly 500 passes; the floor can be passed in (this run's --min-inet-down) either way.
        self.assertTrue(M.launcher_ok(offer(1, "x", 128, 100, 1.0, inet_down=500.0)))
        self.assertTrue(M.launcher_ok(o, min_inet_down=300))
        self.assertTrue(M.launcher_ok(o, min_inet_down=0))
        self.assertFalse(M.launcher_ok(offer(1, "x", 128, 100, 1.0, inet_down=900.0), min_inet_down=1000))
        m = M.Model(rules={"min_inet_down": 200})
        self.assertTrue(M.annotate(m, o)["launcher_ok"])
        # A link speed vast does not report cannot be matched by the search's inet_down filter.
        no_speed = offer(1, "x", 128, 100, 1.0)
        del no_speed["inet_down"]
        self.assertFalse(M.launcher_ok(no_speed))
        self.assertTrue(M.launcher_ok(no_speed, min_inet_down=0))

    def test_an_unverified_fast_host_is_rentable_and_flagged(self):
        for word in ("unverified", "deverified"):
            o = offer(1, "x", 128, 100, 1.0, verification=word, inet_down=900.0)
            self.assertTrue(M.launcher_ok(o), word)
            a = M.annotate(self.m, o)
            self.assertEqual((a["launcher_ok"], a["launcher_flag"], a["launcher_why"]), (True, word, ""))
        v = M.annotate(self.m, offer(1, "x", 128, 100, 1.0))
        self.assertEqual((v["launcher_ok"], v["launcher_flag"]), (True, ""))
        # The strict setting of the launcher (--verified-only) refuses them.
        self.assertFalse(M.launcher_ok(offer(1, "x", 128, 100, 1.0, verification="unverified"), verified_only=True))
        # An unverified host with a slow link is not rentable, and carries no "yes" flag.
        slow = M.annotate(self.m, offer(1, "x", 128, 100, 1.0, verification="unverified", inet_down=100.0))
        self.assertEqual((slow["launcher_ok"], slow["launcher_flag"]), (False, ""))

    def test_gpu_rules_are_the_launchers(self):
        base = dict(num_gpus=1, compute_cap=860, cuda_max_good=12.8, gpu_ram=6138)
        self.assertTrue(M.launcher_ok(offer(1, "x", 128, 100, 1.0, **base), gpus=1, min_gpu_ram_gb=4))
        for kw, word in (({"num_gpus": 2}, "2 GPUs, not the 1 asked for"), ({"compute_cap": 1200}, "compute capability 1200"),
                         ({"compute_cap": 520}, "compute capability 520"), ({"cuda_max_good": 12.4}, "driver supports CUDA 12.4"),
                         ({"gpu_ram": 3000}, "GPU memory 3000 MB")):
            why = M.launcher_failures(offer(1, "x", 128, 100, 1.0, **{**base, **kw}), gpus=1, min_gpu_ram_gb=4)
            self.assertTrue(any(word in w for w in why), (kw, why))
        # A CPU-only job (the default) leaves the compute-capability and CUDA limits out; a machine with no GPU at
        # all is still not found by the launcher's search (it asks num_gpus>=1).
        self.assertTrue(M.launcher_ok(offer(1, "x", 128, 100, 1.0, compute_cap=1200, cuda_max_good=12.0)))
        self.assertFalse(M.launcher_ok(offer(1, "x", 128, 100, 1.0, compute_cap=1200, cuda_max_good=12.0), cpu_only=False))
        self.assertFalse(M.launcher_ok(offer(1, "x", 128, 100, 1.0, num_gpus=0)))

    def test_the_thresholds_are_the_launchers_own(self):
        # One place (python/launcher/offer_rules.py): the tools import the numbers, and the launcher's query is built
        # from the same ones, so a change in one is a change in both.
        import launch
        import offer_rules as R
        self.assertEqual((M.MIN_RELIABILITY, M.NET_CAP, M.MIN_INET_DOWN), (R.MIN_RELIABILITY, R.MAX_NET_COST, R.MIN_INET_DOWN))
        self.assertEqual((launch.MIN_RELIABILITY, launch.MIN_INET_DOWN, launch.MAX_NET_COST),
                         (R.MIN_RELIABILITY, R.MIN_INET_DOWN, R.MAX_NET_COST))
        self.assertIs(launch.offer_check_failures.__globals__["check_failures"], R.check_failures)
        import argparse
        a = argparse.Namespace(min_reliability=R.MIN_RELIABILITY, cpu_only=True, min_cores=16, min_ram_gb=32, disk_gb=20,
                               max_net_cost=R.MAX_NET_COST, max_dph=None, min_inet_down=None)
        q = launch.offer_query(a).split()
        self.assertIn(f"reliability>{R.MIN_RELIABILITY}", q)
        self.assertIn(f"inet_down>={R.MIN_INET_DOWN:g}", q)
        self.assertIn(f"inet_down_cost<={R.MAX_NET_COST}", q)
        # And a record the launcher's checks refuse is refused by the label, for the same reason.
        o = offer(1, "x", 128, 100, 1.0, reliability2=0.85)
        self.assertEqual(launch.offer_check_failures(o), R.check_failures(o))
        self.assertTrue(set(launch.offer_check_failures(o)) <= set(M.launcher_failures(o)))


class Dashboard(unittest.TestCase):
    """What the dashboard server passes to the page: the fields the launcher's rules need are kept, and the
    "can rent" label arrives with its flag and its reason."""

    def test_raw_offers_keep_the_fields_the_launcher_rules_look_at(self):
        import watch as W
        raw = {"id": 7, "machine_id": 70, "cpu_name": "x", "cpu_cores_effective": 128, "cpu_ram": 256 * 1024, "dph_total": 1.0,
               "reliability2": 0.95, "verification": "unverified", "gpu_frac": 1.0, "num_gpus": 1, "gpu_name": "g",
               "inet_down_cost": 0.01, "inet_up_cost": 0.01, "inet_down": 812.5, "gpu_ram": 6138, "compute_cap": 860,
               "cuda_max_good": 12.8, "is_bid": False}
        t = W.trim(raw)
        self.assertEqual((t["inet_down"], t["gpu_ram"], t["compute_cap"], t["cuda_max_good"]), (812.5, 6138, 860, 12.8))
        m = M.Model(rules={"min_inet_down": 800})
        row = W.Watcher._row(M.annotate(m, t))
        self.assertEqual((row["launcher_ok"], row["launcher_flag"], row["launcher_why"], row["inet_down"]),
                         (True, "unverified", "", 812.5))
        row = W.Watcher._row(M.annotate(M.Model(rules={"min_inet_down": 900}), t))
        self.assertEqual((row["launcher_ok"], row["launcher_flag"]), (False, ""))
        self.assertEqual(row["launcher_why"], "download link 812 Mbps below 900")


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
