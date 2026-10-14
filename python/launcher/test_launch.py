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
        self.envs: list[str] = []  # the --env string of each created instance
        self.key_answer = "api-key created {'id': 77, 'key': 'dk-0123456789abcdef', 'permissions': {'api': {}}}"
        self.keys: dict[int, str] = {}  # destroy-only keys alive in the account
        self.deleted_keys: list[int] = []
        self.poll = launch.POLL_S  # the launcher's poll interval in this test (--poll-s)
        self.destroyed_at: dict[int, float] = {}  # instance id -> clock at its destroy

    # ---- clock
    def time(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt
        # A working runner sees the launcher's STOP at its next heartbeat: it uploads what it
        # finished and writes FAILED with `stopped` (then destroys itself).
        if f"runs/{RUN}/STOP" in self.store:
            for inst in self.instances.values():
                if inst["alive"] and inst["script"].split(":")[0] in ("work", "idle"):
                    self.store[f"runs/{RUN}/out/measure/exact.jsonl"] = b'{"id": 0}\n'
                    self.store[f"runs/{RUN}/FAILED"] = json.dumps({"stopped": "the launcher asked"}).encode()
                    inst["alive"] = False
        if dt >= self.poll:
            for inst in self.instances.values():
                if inst["alive"]:
                    self.step(inst)

    def step(self, inst: dict) -> None:
        inst["polls"] += 1
        kind, n = inst["script"].split(":")
        n = int(n)
        if kind in ("never", "broken"):
            return
        if kind in ("work", "idle"):
            # Works on and on (at full CPU, or at 10%), never finishing by itself.
            self.store[f"runs/{RUN}/heartbeat/{self.hb_seq:06}.json"] = json.dumps(
                {"seq": self.hb_seq, "phase": "chunk", "cpu_util_pct": 99.0 if kind == "work" else 10.0}).encode()
            self.hb_seq += 1
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
            self.instances[iid] = {"alive": True, "polls": 0, "script": self.scripts.pop(0), "offer": oid,
                                   "created": self.t}
            self.created_on.append(oid)
            self.envs.append(args[args.index("--env") + 1])
            return {"success": True, "new_contract": iid, "instance_api_key": "ik-secret"}
        if cmd == ("create", "api-key"):
            with open(args[args.index("--permission_file") + 1]) as fh:
                perm = json.load(fh)
            assert perm == {"api": {"instance_write": {"api.instance.destroy": {}}}}, perm
            assert args[args.index("--name") + 1] == f"41bots-destroy-{RUN}"
            self.keys[77] = "dk-0123456789abcdef"
            return self.key_answer
        if cmd == ("delete", "api-key"):
            self.keys.pop(int(args[2]), None)
            self.deleted_keys.append(int(args[2]))
            return "{'success': True}"
        if cmd == ("show", "instance"):
            inst = self.instances.get(int(args[2]))
            if not inst or not inst["alive"]:
                return {"actual_status": "exited", "intended_status": "stopped"}
            if inst["script"].startswith("broken"):
                # The 2026-10-14 failure: the host cannot start the container (the message here
                # also carries a secret, which must never be logged).
                return {"actual_status": "created", "intended_status": "running",
                        "status_msg": "Error response from daemon: failed to create task for container: "
                                      "failed to inject CDI devices: OCI runtime create failed: sk-test-secret-3"}
            return {"actual_status": "running", "intended_status": "running"}
        if cmd == ("destroy", "instance"):
            inst = self.instances.get(int(args[2]))
            if inst:
                inst["alive"] = False
                self.destroyed_at.setdefault(int(args[2]), self.t)
            return ""
        if args[0] == "logs":
            return ""
        raise AssertionError(f"unexpected vast call {args}")

    def instances_created_at(self, iid: int) -> float:
        return self.instances[iid]["created"]

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
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                code = launch.main(["run", "--manifest", json.dumps(manifest), "--image", "img:1", "--bin-sha", "abc",
                                    "--max-minutes", "600", "--startup-minutes", "5", *extra])
        finally:
            for n, v in saved.items():
                setattr(launch, n, v)
        self.log = out.getvalue()
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

    def test_destroy_key_is_passed_and_deleted(self):
        w = World([offer(1, 11, 1.0), offer(2, 12, 1.5)], ["die:2", "done:1"])
        launch.SECRETS.clear()
        code, _ = self.run_launch(w, {"format": 2, "run_id": RUN, "run_seed": 1, "streams": []})
        self.assertEqual(code, 0)
        # Every machine of the run (the replacement too) gets the key; it is deleted at the end.
        self.assertEqual(len(w.envs), 2)
        self.assertTrue(all("-e VAST_DESTROY_KEY=dk-0123456789abcdef" in e for e in w.envs))
        self.assertEqual((w.keys, w.deleted_keys), ({}, [77]))
        # It is redacted from anything logged.
        self.assertEqual(launch.redact("x dk-0123456789abcdef y"), "x *** y")
        launch.SECRETS.clear()

    def test_run_goes_ahead_without_a_destroy_key(self):
        w = World([offer(1, 11, 1.0)], ["done:1"])
        w.key_answer = "An unexpected error occurred: 400 Client Error: Bad Request for url: https://x/api/v0/auth/apikeys/"
        code, _ = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}})
        self.assertEqual(code, 0)
        self.assertNotIn("VAST_DESTROY_KEY", w.envs[0])
        self.assertEqual(w.deleted_keys, [])
        # --no-destroy-key: no key asked for at all.
        w = World([offer(1, 11, 1.0)], ["done:1"])
        w.key_answer = "unused"
        code, _ = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}},
                                  "--no-destroy-key")
        self.assertEqual((code, w.keys), (0, {}))
        self.assertNotIn("VAST_DESTROY_KEY", w.envs[0])

    def test_time_limit_stops_and_keeps_finished_work(self):
        # The run is still working at --max-minutes: the launcher asks it to stop, waits for its
        # marker (it uploads what it finished first), then destroys it.
        w = World([offer(1, 11, 1.0)], ["work:0"])
        code, report = self.run_launch_minutes(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}}, 5)
        self.assertEqual(code, 1)
        self.assertEqual(report["outcome"], "STOPPED")
        self.assertIn(f"runs/{RUN}/STOP", w.store)
        self.assertIn(f"runs/{RUN}/out/measure/exact.jsonl", w.store)
        self.assertIn("uploaded what it finished", report["reason"])
        self.assertTrue(all(not i["alive"] for i in w.instances.values()))

    def test_low_cpu_stops_gracefully(self):
        w = World([offer(1, 11, 1.0)], ["idle:0"])
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}})
        self.assertEqual((code, report["outcome"]), (1, "STOPPED"))
        self.assertIn("CPU use below 85% for 300 s", report["reason"])
        self.assertIn(f"runs/{RUN}/STOP", w.store)
        default_minutes = report["attempts"][0]["minutes"]
        # --low-cpu-s shortens the wait (here 2 polls instead of 5).
        w = World([offer(1, 11, 1.0)], ["idle:0"])
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}},
                                       "--low-cpu-s", "120")
        self.assertEqual((code, report["outcome"]), (1, "STOPPED"))
        self.assertIn("CPU use below 85% for 120 s", report["reason"])
        self.assertLess(report["attempts"][0]["minutes"], default_minutes - 2.5)

    def test_poll_interval(self):
        # --poll-s 30: the run's DONE is seen 30 s after it is written, not 60.
        times = {}
        for poll in (60, 30):
            w = World([offer(1, 11, 1.0)], ["done:2"])
            w.poll = poll
            code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}},
                                           "--poll-s", str(poll))
            self.assertEqual((code, report["outcome"]), (0, "DONE"))
            times[poll] = report["attempts"][0]["minutes"]
        self.assertAlmostEqual(times[60] - times[30], 1.0, places=1)

    def test_container_error_is_replaced_at_once(self):
        # The container cannot start: destroyed at the first poll (not after --startup-minutes)
        # and the next offer is rented, for a format-1 run too (nothing ran on the first).
        w = World([offer(1, 11, 1.0), offer(2, 12, 1.5)], ["broken:0", "done:2"])
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}})
        self.assertEqual((code, report["outcome"]), (0, "DONE"))
        self.assertEqual(w.created_on, [1, 2])
        self.assertEqual(report["replacements"], 1)
        self.assertIn("container failed to start", report["attempts"][0]["ended"])
        self.assertIn("OCI runtime create failed", report["attempts"][0]["ended"])
        # Destroyed at the first check, a minute after it was created.
        self.assertLessEqual(w.destroyed_at[100] - w.instances_created_at(100), launch.POLL_S + 1)
        # The status message is logged, its secret redacted.
        self.assertIn("cannot start its container: Error response from daemon", self.log)
        self.assertNotIn("sk-test-secret-3", self.log)
        self.assertIn("OCI runtime create failed: ***", self.log)
        # Still at most 3 replacements.
        w = World([offer(i, 10 + i, 1.0 + i / 10) for i in range(1, 7)], ["broken:0"] * 6)
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}})
        self.assertEqual(code, 1)
        self.assertEqual(len(w.created_on), 4)
        self.assertEqual(report["replacements"], 3)

    def test_container_start_error(self):
        e = launch.container_start_error
        self.assertIsNone(e(None))
        self.assertIsNone(e({"actual_status": "loading", "status_msg": "Pulling image"}))
        self.assertIsNone(e({"actual_status": "running", "status_msg": "success, running img"}))
        self.assertIn("OCI runtime", e({"actual_status": "created", "status_msg": "x: OCI runtime create failed"}))
        self.assertIn("failed to inject", e({"status_msg": "Failed To Inject CDI devices"}).lower())
        self.assertEqual(e({"actual_status": "exited", "status_msg": ""}), "actual_status exited")
        self.assertTrue(e({"actual_status": "offline", "status_msg": "host down"}).startswith("actual_status offline"))

    def test_silent_machine_is_destroyed_without_waiting(self):
        w = World([offer(1, 11, 1.0)], ["stall:2"])
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}})
        self.assertEqual((code, report["outcome"]), (1, "KILLED"))
        self.assertNotIn(f"runs/{RUN}/STOP", w.store)

    def test_stop_subcommand_waits_then_destroys(self):
        w = World([offer(1, 11, 1.0)], ["work:0"])
        w.instances[100] = {"alive": True, "polls": 0, "script": "work:0", "offer": 1}
        saved = {n: getattr(launch, n) for n in ("vast", "r2_get", "r2_put", "time")}
        launch.vast, launch.r2_get, launch.r2_put, launch.time = w.vast, w.r2_get, w.r2_put, w
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launch.main(["stop", RUN, "--instance", "100", "--wait-s", "60"]), 0)
            self.assertIn(f"runs/{RUN}/STOP", w.store)
            self.assertIn(f"runs/{RUN}/FAILED", w.store)
            self.assertFalse(w.instances[100]["alive"])
            # A run that already ended is not asked again.
            del w.store[f"runs/{RUN}/STOP"]
            with contextlib.redirect_stdout(io.StringIO()):
                launch.main(["stop", RUN, "--wait-s", "60"])
            self.assertNotIn(f"runs/{RUN}/STOP", w.store)
        finally:
            for n, v in saved.items():
                setattr(launch, n, v)

    def test_cancel_under_workflow_leaves_wait_to_cleanup(self):
        # Interrupted with a state file: only STOP is written; the cleanup step waits and destroys,
        # so the instance and the run's key are left alone here.
        import tempfile
        w = World([offer(1, 11, 1.0)], ["work:0"])
        saved_watch = launch.watch

        def interrupted(*args, **kw):
            raise KeyboardInterrupt
        launch.watch = interrupted
        try:
            with tempfile.TemporaryDirectory() as d:
                state = os.path.join(d, "instance-id.txt")
                with self.assertRaises(KeyboardInterrupt):
                    self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}},
                                    "--state-file", state)
                with open(f"{state}.run-id") as fh:
                    self.assertEqual(fh.read(), RUN)
                self.assertTrue(os.path.exists(f"{state}.destroy-key"))
            self.assertIn(f"runs/{RUN}/STOP", w.store)
            self.assertTrue(w.instances[100]["alive"])
            self.assertEqual(w.deleted_keys, [])
            # Without a state file (a person's Ctrl-C): wait for the upload, then destroy.
            w = World([offer(1, 11, 1.0)], ["work:0"])
            code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}})
            self.assertEqual((code, report["outcome"]), (1, "STOPPED"))
            self.assertFalse(w.instances[100]["alive"])
            self.assertEqual(w.deleted_keys, [77])
        finally:
            launch.watch = saved_watch

    def run_launch_minutes(self, world: World, manifest: dict, minutes: int) -> tuple[int, dict]:
        saved = {n: getattr(launch, n) for n in ("vast", "r2_get", "r2_put", "r2_list", "r2_exists", "time")}
        launch.vast = world.vast
        launch.r2_get, launch.r2_put = world.r2_get, world.r2_put
        launch.r2_list, launch.r2_exists = world.r2_list, world.r2_exists
        launch.time = world
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                code = launch.main(["run", "--manifest", json.dumps(manifest), "--image", "img:1", "--bin-sha", "abc",
                                    "--max-minutes", str(minutes), "--startup-minutes", "5"])
        finally:
            for n, v in saved.items():
                setattr(launch, n, v)
        return code, json.loads(world.store.get(f"runs/{RUN}/report.json", b"{}"))

    def test_parse_created_key(self):
        p = launch.parse_created_key
        self.assertEqual(p("api-key created {'id': 5, 'key': 'abc123', 'permissions': {'api': {}}}\n"),
                         {"id": 5, "key": "abc123"})
        self.assertEqual(p('api-key created {"id": 5, "key": "abc123", "permissions": false}'),
                         {"id": 5, "key": "abc123"})
        self.assertIsNone(p("An unexpected error occurred: 400 Client Error"))
        self.assertIsNone(p("api-key created {'id': 5}"))
        self.assertIsNone(p("api-key created {'id': 5, 'key': 'has space'}"))


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
    return _env({"VAST_API_KEY": "vk-test-key-1", "R2_ACCESS_KEY_ID": "ak-test-id-2",
                 "R2_SECRET_ACCESS_KEY": "sk-test-secret-3",
                 "R2_BUCKET": "b", "R2_S3_ENDPOINT": "https://example.invalid"})


if __name__ == "__main__":
    unittest.main()
