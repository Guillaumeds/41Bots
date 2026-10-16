"""Tests of machines.py against fixture records, a fake R2 and a fake market (no network).

    python3 -m unittest discover -s python/tools -p "test_*.py"
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import machines as mc  # noqa: E402
import vast_market as vm  # noqa: E402

NOW = dt.datetime(2026, 10, 16, 12, 0, tzinfo=dt.timezone.utc)
TR = "AMD Ryzen Threadripper PRO 9995WX 96-Cores"
EPYC = "AMD EPYC 9754 128-Core Processor"


def rec(run, cpu, state="final", nodes=1e12, usd=0.3, minutes=60, worlds=100, total=100, sched="plan", per=6.5e6, **kw):
    r = {"run_id": run, "state": state, "outcome": "DONE" if state == "final" else None, "utc": "2026-10-16T11:50:00Z",
         "cpu_model": cpu, "threads": 184, "ram_gb": 241.0, "dph": 1.27, "minutes": minutes, "usd": usd,
         "nodes_total": nodes, "nodes_basis": "measured", "worlds_finished": worlds, "worlds_total": total,
         "usd_per_1e12_nodes": round(usd / nodes * 1e12, 3) if nodes else None, "nodes_per_s_per_thread": per,
         "schedule": sched, "trick": 1}
    r.update(kw)
    return r


FIXTURES = [
    rec("a1", TR, nodes=2e12, usd=0.60),                                   # $0.300
    rec("a2", TR, nodes=1e12, usd=0.32, per=6.3e6),                        # $0.320
    rec("a3", TR, state="running", nodes=1e11, usd=0.05, worlds=5, total=100, utc="2026-10-16T11:58:00Z"),  # settling
    rec("a4", TR, state="running", nodes=5e11, usd=0.15, worlds=50, total=100, utc="2026-10-16T11:00:00Z"),  # stale
    rec("b1", EPYC, nodes=1e12, usd=0.95, per=4.7e6),
    rec("b2", EPYC, nodes=1e12, usd=1.10, nodes_basis="estimated: thread-seconds x speed"),           # estimate: excluded
    rec("c1", "Intel(R) Xeon(R) Gold 5220R CPU @ 2.20GHz", nodes=0, usd=0.2, worlds=0, total=None, sched=None, per=None),
    rec("t1", TR, sched="bench", total=None, nodes=3.5e11, usd=0.15, per=6.77e6,
        windows={"fill": {"nodes": 2.8e11, "nodes_per_s_per_thread": 6.3e6, "usd_per_1e12_nodes": 0.304},
                 "prod": {"nodes": 6.8e10, "nodes_per_s_per_thread": 3.0e6, "usd_per_1e12_nodes": 0.629}}),
]


class FakeR2:
    def __init__(self, store: dict[str, bytes]):
        self.store = store
        self.written: dict[str, bytes] = {}

    def ls(self, prefix):
        return [(k, "2026-10-16T00:00:00.000Z") for k in sorted(self.store) if k.startswith(prefix)]

    def json(self, key):
        try:
            return json.loads(self.store[key])
        except (KeyError, ValueError):
            return None

    def text(self, key):
        b = self.store.get(key)
        return b.decode() if b is not None else None

    def put(self, key, data):
        self.written[key] = data
        self.store[key] = data
        return 200


def offer(i, cpu, threads, dph, ram=256.0):
    return {"id": i, "machine_id": 1000 + i, "cpu_name": cpu, "threads": threads, "ram_gb": ram, "dph_total": dph,
            "gpu_frac": 1, "num_gpus": 1, "verification": "verified", "reliability2": 0.99, "geolocation": "X, FR",
            "inet_down_cost": 0, "inet_up_cost": 0, "inet_down": 900.0}


class RecordsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        for r in FIXTURES:
            with open(os.path.join(self.dir.name, r["run_id"] + ".json"), "w") as f:
                json.dump(r, f)
        self.recs = mc.load_records(from_dir=self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_cpu_key_matches_cpuinfo_and_vast_names(self):
        self.assertEqual(mc.cpu_key(TR), "threadripper pro 9995wx")
        self.assertEqual(mc.cpu_key(EPYC), mc.cpu_key("AMD EPYC 9754"))
        self.assertEqual(mc.cpu_key("Intel(R) Xeon(R) Gold 5220R CPU @ 2.20GHz"), "xeon gold 5220r")
        self.assertEqual(mc.cpu_key(None), "")

    def test_table_marks_estimates_settling_and_stale(self):
        rows = {r[0]: r for r in mc.table_rows(self.recs, NOW)}
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows["a1"][11], "0.300")
        self.assertTrue(rows["b2"][11].startswith("~"))
        self.assertIn("nodes estimated", rows["b2"][-1])
        self.assertIn("settling", rows["a3"][-1])
        self.assertIn("stale", rows["a4"][-1])
        self.assertEqual(rows["c1"][11], "-")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            mc.main(["--from", self.dir.name, "--markdown"])
        self.assertIn("| a1 | final |", out.getvalue())

    def test_live_lists_running_machines_only(self):
        lines = mc.live_lines(self.recs, NOW)
        self.assertEqual(len(lines), 2)
        a3 = next(x for x in lines if x.startswith("a3:"))
        self.assertIn("settling", a3)
        self.assertIn("worlds 5/100", a3)
        self.assertIn("$0.500 per 1e12", a3)
        a4 = next(x for x in lines if x.startswith("a4:"))
        self.assertIn("STALE", a4)
        self.assertNotIn("settling", a4)
        self.assertEqual(mc.live_lines([r for r in self.recs if r["state"] != "running"], NOW)[0][:10], "no machine")

    def test_by_cpu_means_use_measured_settled_records_and_bench_windows(self):
        rows = {(r["cpu"], r["schedule"]): r for r in mc.by_cpu(self.recs)}
        tr = rows[("threadripper pro 9995wx", "plan")]
        self.assertEqual((tr["runs"], tr["measured_runs"]), (4, 3))      # a3 is settling; a4 is half done and counts
        self.assertAlmostEqual(tr["cost_mean"], (0.300 + 0.320 + 0.300) / 3, places=4)
        self.assertAlmostEqual(tr["cost_best"], 0.30, places=3)
        self.assertAlmostEqual(tr["per_thread_mean"], (6.5e6 + 6.3e6 + 6.5e6) / 3)
        ep = rows[("epyc 9754", "plan")]
        self.assertEqual((ep["runs"], ep["measured_runs"]), (2, 1))
        self.assertAlmostEqual(ep["cost_mean"], 0.95, places=3)
        self.assertEqual(ep["estimated_costs"], [1.1])
        self.assertIn(("threadripper pro 9995wx", "fill"), rows)
        self.assertAlmostEqual(rows[("threadripper pro 9995wx", "fill")]["cost_mean"], 0.304)
        self.assertAlmostEqual(rows[("threadripper pro 9995wx", "prod")]["per_thread_mean"], 3.0e6)
        self.assertNotIn("xeon gold 5220r", {k[0] for k in rows})
        text = vm.table(mc.BY_CPU_HEAD, mc.by_cpu_rows(mc.by_cpu(self.recs)), True)
        self.assertIn("| EPYC 9754 | plan | 2 | 1 | 0.950 | 0.950 |", text)

    def test_rank_uses_measured_speed_where_known(self):
        model = mc.M.Model(None, None, rules={"max_dph": 0})  # no speed files; no price cap (the EPYC offer costs $3.74)
        offers = [offer(1, TR, 192, 1.27), offer(2, EPYC, 256, 3.74), offer(3, "Some New Chip 9999", 128, 0.50),
                  offer(4, TR, 32, 0.10), dict(offer(5, TR, 192, 0.50), inet_down=80.0),
                  dict(offer(6, TR, 192, 1.50), verification="unverified")]
        ranked = mc.rank_offers(offers, self.recs, model, min_threads=64)
        self.assertEqual(ranked[0]["id"], 1)                           # measured, cheapest first
        self.assertTrue(ranked[0]["measured"] and ranked[0]["source"].startswith("measured (3 run(s), plan)"))
        per = (6.5e6 + 6.3e6 + 6.5e6) / 3
        self.assertAlmostEqual(ranked[0]["cost"], 1.27 / (192 * 0.958 * per * 3600) * 1e12, places=3)
        ep = next(a for a in ranked if a["id"] == 2)
        self.assertTrue(ep["measured"])
        self.assertAlmostEqual(ep["cost"], 3.74 / (256 * 0.958 * 4.7e6 * 3600) * 1e12, places=3)
        new = next(a for a in ranked if a["id"] == 3)
        self.assertFalse(new["measured"])
        self.assertIn("estimate", new["source"])
        self.assertNotIn(4, [a["id"] for a in ranked])   # too few threads
        self.assertNotIn(5, [a["id"] for a in ranked])   # the launcher cannot rent it: its download link is 80 Mbps
        # With the launcher's own price cap ($1.70/h, owner 2026-10-16) the $3.74 EPYC offer is not rentable either.
        capped = mc.rank_offers(offers, self.recs, mc.M.Model(None, None), min_threads=64)
        self.assertNotIn(2, [a["id"] for a in capped])
        self.assertIn(1, [a["id"] for a in capped])
        six = next(a for a in ranked if a["id"] == 6)    # an unverified host is allowed (owner, 2026-10-16), and flagged
        self.assertEqual(six["launcher_flag"], "unverified")
        rows = {r[0]: r for r in mc.rank_rows(ranked)}
        self.assertTrue(rows[1][5].startswith("0.") and rows[3][5].startswith("~"))
        self.assertTrue(rows[6][7].endswith("; unverified host") and "unverified" not in rows[1][7])

    def test_record_speeds_writes_measured_tt_factors(self):
        speed = os.path.join(self.dir.name, "speed.json")
        with open(speed, "w") as f:
            json.dump({"format": "41bots-vast-speed-factors", "version": 1, "quota_ratio": 0.96,
                       "default": {"compute": 0.6, "tt": 0.5, "uncertainty": 0.5},
                       "machines": {},
                       "models": [{"name": "AMD EPYC 9754", "match": r"EPYC 9754\b", "arch": "x", "cores_per_socket": 128,
                                   "memory": "12 channels", "estimate": {"compute": 1.2, "tt": 1.1, "uncertainty": 0.15,
                                                                        "sources": []}, "measured": None}]}, f)
        measured = os.path.join(self.dir.name, "measured.json")
        with open(measured, "w") as f:
            json.dump({"default_k_nodes_per_s_per_speed_unit": 5.0e6, "models": []}, f)
        done = mc.record_speeds(self.recs, speed, measured, quiet=True)
        self.assertEqual(sorted(done), ["epyc 9754", "threadripper pro 9995wx"])
        sp = vm.load_speed(speed)
        self.assertEqual(vm.factor(sp, EPYC, "tt")[:2], (0.94, "measured"))          # 4.7e6 / 5e6
        self.assertEqual(vm.factor(sp, TR, "tt")[:2], (1.287, "measured"))           # 6.433e6 / 5e6 (plan preferred)
        runs = sp["models"][-1]["measured"]["runs"][0]
        self.assertEqual(runs["run"], "a1,a2,a4")
        self.assertEqual(vm.factor(sp, EPYC, "compute")[:2], (1.2, "estimate"))      # compute untouched


class BackfillTest(unittest.TestCase):
    def make_r2(self):
        store: dict[str, bytes] = {}

        def put(key, obj):
            store[key] = (obj if isinstance(obj, (bytes, str)) else json.dumps(obj)).encode() if not isinstance(obj, bytes) else obj

        # A bench run with results.
        put("runs/t1bench-x/manifest.json", {"run_id": "t1bench-x", "chunk": {"t1bench": {}}})
        put("runs/t1bench-x/hardware.json", {"cpu_model": TR, "threads_default": 184, "mem_total_kb": 263193936, "dph_env": "1.27",
                                             "usable_cores": 184.0, "bin_sha": "abc"})
        put("runs/t1bench-x/report.json", {"elapsed_min": 7.0, "dollars_est": 0.148, "outcome": "DONE", "offer_id": 5, "utc": "2026-10-15T13:00:00Z"})
        put("runs/t1bench-x/DONE", {})
        put("runs/t1bench-x/out/t1bench/results.json", {
            "fill": {"nodes": 2.8e11, "nodes_per_s": 1.16e9, "nodes_per_s_per_thread": 6.3e6, "worlds_finished": 439, "threads": 184,
                     "tt_mb": 448, "dollars_per_tera_nodes": 0.304, "cpu_pct": 99.3, "wall_s": 241.5},
            "prod": {"nodes": 6.8e10, "nodes_per_s": 5.6e8, "nodes_per_s_per_thread": 3.0e6, "worlds_finished": 1, "threads": 184,
                     "dollars_per_tera_nodes": 0.629, "cpu_pct": 93.6, "wall_s": 121.8},
            "sweep": {"runs": [{"nodes_per_s": 1.0e9, "window_s": 90.0, "worlds_finished": 80}]},
            "score": {"nodes_per_s_per_thread": 6.77e6, "dollars_per_tera_nodes": 0.283}})
        # A trick-1 measurement killed mid-way, resumed from an earlier run: worlds in the checkpoint
        # (one restored from the earlier run) and one complete decision in the worlds file.
        dec = [{"id": 1, "trick": 1, "position": 0, "stratum": 0}, {"id": 2, "trick": 1, "position": 5, "stratum": 0}]
        for run, resume in (("m-a", None), ("m-b", "m-a")):
            man = {"run_id": run, "chunk": {"search_measure": {"trick": 1, "count": 32, "to": 4, "ids": [1, 2]}}}
            if resume:
                man["resume_from"] = resume
            put(f"runs/{run}/manifest.json", man)
            put(f"runs/{run}/hardware.json", {"cpu_model": TR, "threads_default": 184, "mem_total_kb": 263193936, "dph_env": "1.27"})
            put(f"runs/{run}/report.json", {"elapsed_min": 60.0, "dollars_est": 1.27, "outcome": "KILLED"})
            put(f"runs/{run}/partial/out/measure/decisions-t1.jsonl", "\n".join(json.dumps(d) for d in dec))
            put(f"runs/{run}/partial/out/measure/worlds.log",
                'worlds: position-aware plan {"classes":[{"position":0,"threads":46},{"position":5,"threads":1}]}\n')
        put("runs/m-a/partial/out/measure/exact-t1-n4.jsonl.ckpt", json.dumps({"id": 1, "w": 0, "v": [], "us": 100e6}))
        put("runs/m-a/partial/out/measure/exact-t1-n4.jsonl", "")
        ck = [{"id": 1, "w": 0, "v": [], "us": 100e6}, {"id": 1, "w": 1, "v": [], "us": 50e6}]
        put("runs/m-b/partial/out/measure/exact-t1-n4.jsonl.ckpt", "\n".join(json.dumps(x) for x in ck))
        put("runs/m-b/partial/out/measure/exact-t1-n4.jsonl",
            json.dumps({"id": 2, "from": 0, "rows": [[], [], [], []]}))
        put("runs/m-b/partial/out/measure/exact-t1-n4.jsonl.times.jsonl",
            json.dumps({"id": 2, "world_us": 40e6, "threads": 1, "worlds": 4}))
        return FakeR2(store)

    def test_backfill_builds_bench_and_measure_records(self):
        r2 = self.make_r2()
        model = mc.M.Model(None, None)
        model.models = [{"name": "TR", "match": "9995WX", "per_thread_nodes_per_s": 6.0e6, "confidence": "measured",
                         "_re": mc.re.compile("9995WX")}]
        with contextlib.redirect_stderr(io.StringIO()):
            recs = mc.backfill(r2, ["t1bench-x", "m-b", "nope"], model, write=True, out_dir=None)
        self.assertEqual([r["run_id"] for r in recs], ["t1bench-x", "m-b"])
        b = recs[0]
        self.assertEqual((b["schedule"], b["state"], b["outcome"], b["cpu_model"], b["threads"]), ("bench", "final", "DONE", TR, 184))
        self.assertEqual(b["nodes_total"], round(2.8e11 + 6.8e10 + 9.0e10))
        self.assertEqual(b["worlds_finished"], 520)
        self.assertAlmostEqual(b["usd_per_1e12_nodes"], 0.148 / 4.38e11 * 1e12, places=3)
        self.assertEqual(b["windows"]["fill"]["usd_per_1e12_nodes"], 0.304)
        self.assertEqual(b["nodes_per_s_per_thread"], 6.77e6)
        self.assertEqual(b["ram_gb"], 251.0)
        m = recs[1]
        # World (1, 0) was taken over from m-a; (1, 1) ran here on 46 threads, decision 2's 4 worlds on 1.
        self.assertEqual((m["worlds_finished"], m["worlds_taken_over"], m["worlds_total"]), (5, 1, 7))
        self.assertEqual(m["world_thread_s"], round(50 * 46 + 40))
        self.assertEqual(m["nodes_total"], round((50 * 46 + 40) * 6.0e6))
        self.assertTrue(m["nodes_basis"].startswith("estimated"))
        self.assertFalse(mc.measured(m))
        self.assertEqual((m["schedule"], m["trick"], m["resumed_from"], m["usd"]), ("plan", 1, "m-a", 1.27))
        self.assertAlmostEqual(m["usd_per_1e12_nodes"], 1.27 / m["nodes_total"] * 1e12, places=3)
        self.assertNotIn("_done_keys", m)
        self.assertEqual(set(r2.written), {"machines/t1bench-x.json", "machines/m-b.json"})
        self.assertEqual(json.loads(r2.written["machines/m-b.json"])["run_id"], "m-b")

    def test_plan_threads_and_jsonl_tolerate_junk(self):
        self.assertEqual(mc.plan_threads('x\nworlds: position-aware plan {"classes":[{"position":2,"threads":6},{"position":2,"threads":9}]}'),
                         {2: 6})
        self.assertEqual(mc.plan_threads(None), {})
        self.assertEqual(mc.jsonl('{"a":1}\n{"b":2\n'), [{"a": 1}])


if __name__ == "__main__":
    unittest.main()
