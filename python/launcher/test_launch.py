"""Tests of the launcher's renting and replacement logic against a simulated vast.ai and R2.

    python3 -m unittest discover -s python/launcher -p "test_*.py"

Nothing here talks to vast.ai or R2: `vast`, the R2 helpers and the clock are replaced by an
in-memory world in which each rented instance follows a script (heartbeat, stall, finish).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch  # noqa: E402

RUN = "t-run"


class World:
    def __init__(self, offers: list[dict], scripts: list[str], create_fail: dict[int, str] | None = None):
        self.t = 1_000_000.0
        self.store: dict[str, bytes] = {"bin/abc/manifest.json": b"{}"}
        self.offers = offers
        self.scripts = list(scripts)  # one per successfully created instance, in order
        self.create_fail = create_fail or {}
        self.instances: dict[int, dict] = {}
        self.next_id = 100
        self.hb_seq = 0
        self.created_on: list[int] = []

    # ---- clock
    def time(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt
        if dt >= launch.POLL_S:
            for inst in self.instances.values():
                if inst["alive"]:
                    self.step(inst)

    def step(self, inst: dict) -> None:
        inst["polls"] += 1
        kind, n = inst["script"].split(":")
        n = int(n)
        if kind == "never":
            return
        if kind == "stall" and inst["polls"] > n:
            return
        if kind == "die" and inst["polls"] > n:
            inst["alive"] = False
            return
        self.store[f"runs/{RUN}/heartbeat/{self.hb_seq:06}.json"] = json.dumps(
            {"seq": self.hb_seq, "phase": "chunks", "cpu_util_pct": 99.0}).encode()
        self.hb_seq += 1
        if kind == "done" and inst["polls"] >= n:
            self.store[f"runs/{RUN}/merged.json"] = json.dumps({"complete": True, "streams": [
                {"name": "gate", "kind": "arena_blocks", "chunks_done": 3, "chunks_total": 145, "left": 0,
                 "sprt": {"decision": "Accept"}, "chunks": [{"index": 0}]}]}).encode()
            self.store[f"runs/{RUN}/DONE"] = b"{}"
            inst["alive"] = False  # the runner destroys its own instance

    # ---- vast CLI
    def vast(self, *args, raw=True, check=True, tail=()):
        cmd = args[:2]
        if cmd == ("search", "offers"):
            return [dict(o) for o in self.offers]
        if cmd == ("create", "instance"):
            oid = int(args[2])
            how = self.create_fail.get(oid)
            if how == "none":
                return None  # vast printed something that is not JSON: the offer is gone
            if how == "raise":
                raise RuntimeError("vastai create instance failed: no_such_ask")
            iid = self.next_id
            self.next_id += 1
            self.instances[iid] = {"alive": True, "polls": 0, "script": self.scripts.pop(0), "offer": oid}
            self.created_on.append(oid)
            return {"success": True, "new_contract": iid}
        if cmd == ("show", "instance"):
            inst = self.instances.get(int(args[2]))
            if not inst or not inst["alive"]:
                return {"actual_status": "exited", "intended_status": "stopped"}
            return {"actual_status": "running", "intended_status": "running"}
        if cmd == ("destroy", "instance"):
            inst = self.instances.get(int(args[2]))
            if inst:
                inst["alive"] = False
            return ""
        if args[0] == "logs":
            return ""
        raise AssertionError(f"unexpected vast call {args}")

    # ---- R2
    def r2_get(self, key: str):
        return self.store.get(key)

    def r2_put(self, key: str, data: bytes) -> None:
        self.store[key] = data

    def r2_list(self, prefix: str) -> list[str]:
        return sorted(k for k in self.store if k.startswith(prefix))

    def r2_exists(self, key: str) -> bool:
        return key in self.store


def offer(i: int, machine: int, dph: float) -> dict:
    return {"id": i, "machine_id": machine, "dph_total": dph, "cpu_cores_effective": 32, "cpu_name": "EPYC 7B12",
            "num_gpus": 1, "gpu_name": "GTX 1080", "reliability2": 0.99, "is_bid": False}


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.env = mock_env()
        self.env.__enter__()
        launch._BENCH = None

    def tearDown(self):
        self.env.__exit__(None, None, None)

    def run_launch(self, world: World, manifest: dict, *extra: str) -> tuple[int, dict]:
        saved = {n: getattr(launch, n) for n in ("vast", "r2_get", "r2_put", "r2_list", "r2_exists", "time")}
        launch.vast = world.vast
        launch.r2_get, launch.r2_put = world.r2_get, world.r2_put
        launch.r2_list, launch.r2_exists = world.r2_list, world.r2_exists
        launch.time = world
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                code = launch.main(["run", "--manifest", json.dumps(manifest), "--image", "img:1", "--bin-sha", "abc",
                                    "--max-minutes", "600", "--startup-minutes", "5", *extra])
        finally:
            for n, v in saved.items():
                setattr(launch, n, v)
        report = json.loads(world.store.get(f"runs/{RUN}/report.json", b"{}"))
        return code, report

    def test_vanished_offer_falls_back_to_next_best(self):
        offers = [offer(1, 11, 1.0), offer(2, 12, 1.5), offer(3, 13, 2.0)]
        w = World(offers, ["done:2"], create_fail={1: "none", 2: "raise"})
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}})
        self.assertEqual(code, 0)
        self.assertEqual(w.created_on, [3])
        self.assertEqual(report["outcome"], "DONE")
        self.assertEqual(report["offer_id"], 3)
        self.assertEqual(report["replacements"], 0)

    def test_no_offer_left_stops(self):
        w = World([offer(1, 11, 1.0)], [], create_fail={1: "none"})
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}})
        self.assertEqual(code, 1)
        self.assertEqual(w.created_on, [])

    def test_format2_stale_machine_is_replaced_on_another_host(self):
        offers = [offer(1, 11, 1.0), offer(2, 11, 1.1), offer(3, 12, 1.5)]
        w = World(offers, ["stall:3", "done:2"])
        m = {"format": 2, "run_id": RUN, "run_seed": 1, "streams": []}
        code, report = self.run_launch(w, m)
        self.assertEqual(code, 0)
        self.assertEqual(report["outcome"], "DONE")
        self.assertEqual(report["replacements"], 1)
        # Offer 2 is on the stalled machine 11, so the replacement is offer 3.
        self.assertEqual(w.created_on, [1, 3])
        self.assertIn("heartbeat missing", report["attempts"][0]["ended"])
        self.assertTrue(all(not i["alive"] for i in w.instances.values()))
        self.assertEqual(report["result"]["streams"][0]["sprt"]["decision"], "Accept")
        self.assertNotIn("chunks", report["result"]["streams"][0])
        # The launcher recorded the binaries in the manifest the machines run.
        self.assertEqual(json.loads(w.store[f"runs/{RUN}/manifest.json"])["bin_sha"], "abc")

    def test_format2_dead_instance_is_replaced(self):
        w = World([offer(1, 11, 1.0), offer(2, 12, 1.5)], ["die:2", "done:1"])
        code, report = self.run_launch(w, {"format": 2, "run_id": RUN, "run_seed": 1, "streams": []})
        self.assertEqual((code, report["replacements"]), (0, 1))
        self.assertIn("died", report["attempts"][0]["ended"])

    def test_at_most_three_replacements(self):
        offers = [offer(i, 10 + i, 1.0 + i / 10) for i in range(1, 7)]
        w = World(offers, ["never:0"] * 6)
        code, report = self.run_launch(w, {"format": 2, "run_id": RUN, "run_seed": 1, "streams": []})
        self.assertEqual(code, 1)
        self.assertEqual(len(w.created_on), 4)  # the first machine and 3 replacements
        self.assertEqual(report["replacements"], 3)
        self.assertEqual(report["outcome"], "KILLED")

    def test_format1_is_not_replaced(self):
        w = World([offer(1, 11, 1.0), offer(2, 12, 1.5)], ["stall:2", "done:1"])
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}})
        self.assertEqual(code, 1)
        self.assertEqual(w.created_on, [1])

    def test_missing_input_stops_before_renting(self):
        w = World([offer(1, 11, 1.0)], ["done:1"])
        m = {"format": 2, "run_id": RUN, "run_seed": 1, "streams": [], "inputs": {"equity": "sha256:" + "ab" * 32}}
        with self.assertRaises(SystemExit) as e:
            self.run_launch(w, m)
        self.assertIn("put-input", str(e.exception))
        self.assertEqual(w.created_on, [])
        w.store["inputs/sha256/" + "ab" * 32] = b"x"
        code, _ = self.run_launch(w, m)
        self.assertEqual((code, w.created_on), (0, [1]))

    def test_bin_sha_mismatch_refused(self):
        w = World([offer(1, 11, 1.0)], ["done:1"])
        with self.assertRaises(SystemExit):
            self.run_launch(w, {"format": 2, "run_id": RUN, "run_seed": 1, "streams": [], "bin_sha": "other"})

    def test_redact(self):
        os.environ["R2_SECRET_ACCESS_KEY"] = "s3cr3t"
        self.assertEqual(launch.redact("key s3cr3t here"), "key *** here")


@contextlib.contextmanager
def _env(values: dict[str, str]):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def mock_env():
    return _env({"VAST_API_KEY": "vk", "R2_ACCESS_KEY_ID": "ak", "R2_SECRET_ACCESS_KEY": "sk",
                 "R2_BUCKET": "b", "R2_S3_ENDPOINT": "https://example.invalid"})


if __name__ == "__main__":
    unittest.main()
