"""The O3G2 round manifests (round_manifest.py) and taking over an instance rented by hand
(launch.py --instance-id). Standard library only: ``python3 -m unittest`` in python/launcher."""

from __future__ import annotations

import json
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import launch  # noqa: E402
import round_manifest as rm  # noqa: E402


class Amounts(unittest.TestCase):
    def test_amounts(self):
        self.assertEqual(rm.parse_amounts("500k,1M,2M"), [500000, 1000000, 2000000])
        self.assertEqual(rm.parse_amounts("1.5M"), [1500000])
        for bad in ("", "2M,1M", "1M,1M", "1M,2M,3M,4M", "two", "0"):
            with self.assertRaises(ValueError, msg=bad):
                rm.parse_amounts(bad)


class Build(unittest.TestCase):
    def test_round_one_from_the_seed(self):
        m = rm.build(1, rm.seed_record(), [2000000, 4000000, 8000000])
        p = m["streams"][0]["params"]
        r = p["round"]
        self.assertEqual(m["run_id"], "o3r-1-1")
        self.assertEqual((r["id"], r["seed"]), ("o3r-1", 90100))
        self.assertEqual([c["rows"] for c in r["candidates"]], [2000000, 4000000, 8000000])
        self.assertEqual(r["risk_model"], rm.seed_record()["next"]["risk_model"])
        self.assertEqual([s["seed"] for s in r["g2"]["splits"]], [90111, 90112, 90113])
        self.assertEqual(p["init_net"], rm.seed_record()["next"]["init_net"])
        self.assertEqual(len(p["label_inputs"]), 4)
        self.assertEqual(p["train_mode"], "additive")
        # 40.1M stored rows and 8M new ones: 80 + 3.5 x 48.1 GB.
        self.assertEqual(m["machine"]["disk_gb"], 249)
        # The launcher finds every stored input it must check before renting.
        names = launch.param_inputs(m)
        self.assertIn("rung.round.risk_model", names)
        self.assertIn("rung.round.label_check.shard", names)
        self.assertIn("rung.init_net", names)

    def test_options(self):
        m = rm.build(2, {"id": "o3r-1", "next": rm.seed_record()["next"]}, [1000000, 2000000],
                     scratch_last="4096-2048-1024-256", risk_power="0.5", g2="rebuild")
        r = m["streams"][0]["params"]["round"]
        self.assertEqual(r["candidates"][1], {"rows": 2000000, "mode": "scratch", "widths": "4096,2048,1024,256"})
        self.assertEqual(r["risk_power"], "0.5")
        self.assertNotIn("risk_model", r)
        self.assertEqual(r["seed"], 90200)
        with self.assertRaises(ValueError):
            rm.build(2, {"next": {}}, [1])
        with self.assertRaises(ValueError):
            rm.build(1, rm.seed_record(), [1], scratch_last="9000")

    def test_previous_round_from_r2(self):
        rec = {"next": rm.seed_record()["next"], "winner": {"name": "c1"}}
        got = rm.previous("o3r-1", lambda k: json.dumps(rec).encode() if k == "rounds/o3r-1/round.json" else None)
        self.assertEqual(got["id"], "o3r-1")
        with self.assertRaises(ValueError):
            rm.previous("o3r-9", lambda k: None)

    def test_committed_round_one_manifest_is_the_builders(self):
        path = os.path.join(rm.ROUNDS_DIR, "..", "o3-round-1.manifest.json")
        with open(path) as f:
            committed = json.load(f)
        built = rm.build(1, rm.seed_record(), [2000000, 4000000, 8000000], note="Round 1: three additive candidates.")
        self.assertEqual(committed, built)


class TakeOver(unittest.TestCase):
    def args(self, **kw):
        a = types.SimpleNamespace(instance_id=123, disk_gb=250, gpus=1, min_gpu_ram_gb=16, min_cores=190,
                                  cpu_only=False)
        a.__dict__.update(kw)
        return a

    def test_checks(self):
        info = {"id": 123, "disk_space": 1488, "num_gpus": 1, "gpu_ram": 12288, "cpu_cores_effective": 256,
                "dph_total": 0.304, "actual_status": "running"}
        best, why = launch.take_check(self.args(), info)
        self.assertEqual(len(why), 1, why)
        self.assertIn("its GPU has 12 GB", why[0])
        best, why = launch.take_check(self.args(min_gpu_ram_gb=12), info)
        self.assertEqual(why, [])
        self.assertEqual((best["id"], best["dph_total"], best["cpu_cores_effective"]), (123, 0.304, 256))
        _, why = launch.take_check(self.args(min_gpu_ram_gb=12), {**info, "disk_space": 32})
        self.assertTrue(any("its disk is 32 GB" in w for w in why), why)
        _, why = launch.take_check(self.args(), None)
        self.assertTrue(why)

    def test_env_dict(self):
        self.assertEqual(launch.env_dict("-e A=1 -e B=x=y"), {"A": "1", "B": "x=y"})


if __name__ == "__main__":
    unittest.main()
