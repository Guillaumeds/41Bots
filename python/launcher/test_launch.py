"""Tests of the launcher's renting and replacement logic against a simulated vast.ai and R2.

    python3 -m unittest discover -s python/launcher -p "test_*.py"

Nothing here talks to vast.ai or R2: `vast`, the R2 helpers and the clock are replaced by an
in-memory world in which each rented instance follows a script (heartbeat, stall, finish).
"""

from __future__ import annotations

import argparse
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
        self.searches = 0
        self.id_lookups = 0  # searches by offer id (`id=N`), the one search --offer-id makes
        self.forbid_search = False  # the account's daily search quota is used up: any full search is a bug
        self.status_fails = False  # `show instance` fails (vast's API down)
        self.listing_fails = False  # `show instances` fails
        self.r2_list_fails_until = 0.0  # r2_list raises while the clock is below this
        self.poll = launch.POLL_S  # the launcher's poll interval in this test (--poll-s)
        self.progress = None  # a heartbeat's progress object (e.g. with an eta), if set
        self.outputs: dict[str, bytes] = {}  # files a finishing run uploads before its DONE
        self.destroyed_at: dict[int, float] = {}  # instance id -> clock at its destroy
        self.id_lookup_fails = False  # the search by offer id fails
        self.search_fails = False  # a full search fails (vast's daily search quota: 429)
        self.hidden_listings = 0  # `show instances` leaves out the newest instance this many more times
        self.destroy_fails = 0  # the next this many `destroy instance` calls fail (vast no longer knows the instance)
        self.destroy_calls = 0  # how many `destroy instance` calls were made
        self.queries: list[str] = []  # the query of every full search
        self.disks: list[str] = []  # the --disk of each created instance

    # ---- clock
    def time(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt
        # A working runner sees the launcher's STOP at its next heartbeat: it uploads what it
        # finished and writes FAILED with `stopped` (then destroys itself).
        if f"runs/{RUN}/STOP" in self.store:
            for inst in self.instances.values():
                if inst["alive"] and inst["script"].split(":")[0] in ("work", "idle", "silentbusy", "silentlow",
                                                                       "obeys"):
                    self.store[f"runs/{RUN}/out/measure/exact.jsonl"] = b'{"id": 0}\n'
                    self.store[f"runs/{RUN}/FAILED"] = json.dumps({"stopped": "the launcher asked"}).encode()
                    inst["alive"] = False
        if dt >= self.poll:
            for inst in self.instances.values():
                if inst["alive"]:
                    self.step(inst)

    def heartbeat(self, phase: str, cpu: float) -> None:
        hb = {"seq": self.hb_seq, "phase": phase, "cpu_util_pct": cpu}
        if self.progress is not None:
            hb["progress"] = self.progress
        self.store[f"runs/{RUN}/heartbeat/{self.hb_seq:06}.json"] = json.dumps(hb).encode()
        self.hb_seq += 1

    def step(self, inst: dict) -> None:
        inst["polls"] += 1
        kind, *args = inst["script"].split(":")
        n = int(args[0])
        if kind in ("never", "broken"):
            return
        if kind in ("work", "idle"):
            # Works on and on (at full CPU, or at 10%), never finishing by itself.
            self.heartbeat("chunk", 99.0 if kind == "work" else 10.0)
            return
        if kind in ("silentbusy", "silentidle", "silentlow", "stuck"):
            # Heartbeats for n polls, then none: its uploads hang (busy) or it stopped working (idle).
            if inst["polls"] <= n:
                self.heartbeat("chunk", 99.0)
            return
        if kind == "hiccup":
            # Silent (busy) from poll n+1 to poll m, then heartbeats again, and done 3 polls later.
            m = int(args[1])
            if n < inst["polls"] <= m:
                return
            self.heartbeat("chunk", 99.0)
            if inst["polls"] >= m + 3:
                self.store[f"runs/{RUN}/DONE"] = b"{}"
                inst["alive"] = False
            return
        if kind == "obeys" and inst["polls"] >= n:
            # A replacement that sees no STOP finishes its work.
            self.store[f"runs/{RUN}/DONE"] = b"{}"
            inst["alive"] = False
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
            self.store.update(self.outputs)
            self.store[f"runs/{RUN}/merged.json"] = json.dumps({"complete": True, "streams": [
                {"name": "gate", "kind": "arena_blocks", "chunks_done": 3, "chunks_total": 145, "left": 0,
                 "sprt": {"decision": "Accept"}, "chunks": [{"index": 0}]}]}).encode()
            self.store[f"runs/{RUN}/DONE"] = b"{}"
            inst["alive"] = False  # the runner destroys its own instance

    # ---- vast CLI
    def vast(self, *args, raw=True, check=True, tail=(), timeout=None):
        cmd = args[:2]
        if cmd == ("search", "offers"):
            if args[2].startswith("id="):
                self.id_lookups += 1
                if self.id_lookup_fails:
                    raise RuntimeError("vastai search offers failed: 429 Too Many Requests")
                return [dict(o) for o in self.offers if o["id"] == int(args[2][3:])]
            self.searches += 1
            self.queries.append(args[2])
            if self.forbid_search:
                raise AssertionError("a search was made")
            if self.search_fails:
                raise RuntimeError("vastai search offers failed: 429 Too Many Requests")
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
                                   "created": self.t, "label": args[args.index("--label") + 1]}
            self.created_on.append(oid)
            self.envs.append(args[args.index("--env") + 1])
            self.disks.append(args[args.index("--disk") + 1])
            if how == "unreadable":
                return None  # vast rented it, but printed something that is not JSON
            if how == "late":
                # vast rented it, its answer was an error, and the first listing misses it.
                self.hidden_listings = 1
                raise RuntimeError("vastai create instance failed: 504 Gateway Timeout")
            return {"success": True, "new_contract": iid, "instance_api_key": "ik-secret"}
        if cmd == ("show", "instances"):
            if self.listing_fails:
                raise RuntimeError("vastai show instances failed: 502 Bad Gateway")
            # Destroyed instances leave the list; one that died (or ended) but was not destroyed stays, exited.
            hide = None
            if self.hidden_listings > 0:
                self.hidden_listings -= 1
                hide = max(self.instances)
            return [{"id": iid, "label": inst["label"], "actual_status": "running" if inst["alive"] else "exited",
                     "intended_status": "running" if inst["alive"] else "stopped"}
                    for iid, inst in self.instances.items() if not inst.get("destroyed") and iid != hide]
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
            if self.status_fails:
                raise RuntimeError("vastai show instance failed: 502 Bad Gateway")
            inst = self.instances.get(int(args[2]))
            if not inst or not inst["alive"]:
                return {"actual_status": "exited", "intended_status": "stopped"}
            cpu = {"silentbusy": 95.0, "hiccup": 97.0, "silentidle": 2.0, "silentlow": 17.0,
                   "stuck": 95.0}.get(inst["script"].split(":")[0])
            if cpu is not None:
                return {"actual_status": "running", "intended_status": "running", "cpu_util": cpu,
                        "cpu_cores_effective": 32, "status_msg": "success, running img:1"}
            if inst["script"].startswith("broken"):
                # The 2026-10-14 failure: the host cannot start the container (the message here
                # also carries a secret, which must never be logged).
                return {"actual_status": "created", "intended_status": "running",
                        "status_msg": "Error response from daemon: failed to create task for container: "
                                      "failed to inject CDI devices: OCI runtime create failed: sk-test-secret-3"}
            return {"actual_status": "running", "intended_status": "running"}
        if cmd == ("destroy", "instance"):
            self.destroy_calls += 1
            if self.destroy_fails > 0:
                self.destroy_fails -= 1
                raise RuntimeError("vastai destroy instance failed: 404 Not Found: no such instance")
            inst = self.instances.get(int(args[2]))
            if inst:
                inst["alive"] = False
                inst["destroyed"] = True
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
        if self.t < self.r2_list_fails_until and "heartbeat" in prefix:
            raise RuntimeError("aws s3api list-objects-v2 failed: Could not connect to the endpoint URL")
        return sorted(k for k in self.store if k.startswith(prefix))

    def r2_list_sized(self, prefix: str) -> list[tuple[str, int]]:
        return [(k, len(self.store[k])) for k in self.r2_list(prefix)]

    def r2_exists(self, key: str) -> bool:
        return key in self.store

    def r2_delete(self, key: str) -> None:
        self.store.pop(key, None)


def offer(i: int, machine: int, dph: float, **kw) -> dict:
    o = {"id": i, "machine_id": machine, "dph_total": dph, "cpu_cores_effective": 32, "cpu_name": "EPYC 7B12",
         "num_gpus": 1, "gpu_name": "GTX 1080", "reliability2": 0.99, "is_bid": False, "verification": "verified",
         "gpu_frac": 1.0, "rentable": True}
    o.update(kw)
    return o


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.env = mock_env()
        self.env.__enter__()
        launch._BENCH = None

    def tearDown(self):
        self.env.__exit__(None, None, None)

    @contextlib.contextmanager
    def patched(self, world: World):
        """The launcher's vast, R2 and clock replaced by `world`."""
        saved = {n: getattr(launch, n) for n in ("vast", "r2_get", "r2_put", "r2_list", "r2_list_sized", "r2_exists",
                                                 "r2_delete", "time")}
        launch.vast = world.vast
        launch.r2_get, launch.r2_put = world.r2_get, world.r2_put
        launch.r2_list, launch.r2_exists = world.r2_list, world.r2_exists
        launch.r2_list_sized = world.r2_list_sized
        launch.r2_delete = world.r2_delete
        launch.time = world
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                yield
        finally:
            for n, v in saved.items():
                setattr(launch, n, v)

    def run_launch(self, world: World, manifest: dict, *extra: str, time_limit: bool = True) -> tuple[int, dict]:
        saved = {n: getattr(launch, n) for n in ("vast", "r2_get", "r2_put", "r2_list", "r2_list_sized", "r2_exists",
                                                 "r2_delete", "time")}
        launch.vast = world.vast
        launch.r2_get, launch.r2_put = world.r2_get, world.r2_put
        launch.r2_list, launch.r2_exists = world.r2_list, world.r2_exists
        launch.r2_list_sized = world.r2_list_sized
        launch.r2_delete = world.r2_delete
        launch.time = world
        out = io.StringIO()
        limit = ["--max-minutes", "600"] if time_limit else []
        try:
            with contextlib.redirect_stdout(out):
                code = launch.main(["run", "--manifest", json.dumps(manifest), "--image", "img:1", "--bin-sha", "abc",
                                    *limit, "--startup-minutes", "5", *extra])
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

    def test_offer_chosen_by_id_is_checked_by_one_lookup_and_rented_without_a_search(self):
        w = World([offer(1, 11, 0.5), offer(500, 50, 1.013, cpu_cores_effective=384)], ["done:2"])
        w.forbid_search = True
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}},
                                       "--offer-id", "500", "--offer-dph", "1.013", "--offer-cores", "384")
        self.assertEqual(code, 0)
        self.assertEqual(w.created_on, [500])
        self.assertEqual((w.searches, w.id_lookups), (0, 1))
        self.assertEqual((report["offer_id"], report["cores_effective"]), (500, 384))
        self.assertIn("chosen by the owner and checked by id", self.log)

    def test_offer_chosen_by_id_that_is_gone_stops_without_searching(self):
        w = World([offer(1, 11, 1.0), offer(500, 50, 1.013)], [], create_fail={500: "none"})
        w.forbid_search = True
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}},
                                       "--offer-id", "500", "--offer-dph", "1.013", "--offer-cores", "384")
        self.assertEqual(code, 1)
        self.assertEqual(w.created_on, [])
        self.assertEqual(w.searches, 0)

    def test_offer_chosen_by_id_must_pass_the_same_checks(self):
        # Unverified, shared, unreliable or interruptible: refused by name. Not on offer, or a
        # lookup that fails: refused too (the checks are never skipped). Nothing is rented.
        cases = [({"verification": "unverified"}, "host not verified"), ({"gpu_frac": 0.5}, "shared machine"),
                 ({"reliability2": 0.95}, "reliability 0.950 below 0.98"), ({"is_bid": True}, "interruptible")]
        for kw, word in cases:
            w = World([offer(500, 50, 1.0, **kw)], ["done:1"])
            w.forbid_search = True
            with self.assertRaises(SystemExit) as e:
                self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}}, "--offer-id", "500")
            self.assertIn("fails the launcher's checks", str(e.exception), kw)
            self.assertIn(word, str(e.exception), kw)
            self.assertEqual(w.created_on, [], kw)
        w = World([offer(1, 11, 1.0)], ["done:1"])
        with self.assertRaises(SystemExit) as e:
            self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}}, "--offer-id", "500")
        self.assertIn("not on offer", str(e.exception))
        w = World([offer(500, 50, 1.0)], ["done:1"])
        w.id_lookup_fails = True
        with self.assertRaises(SystemExit) as e:
            self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}}, "--offer-id", "500")
        self.assertIn("could not be checked", str(e.exception))
        self.assertEqual(w.created_on, [])
        self.assertEqual(launch.offer_check_failures(offer(7, 7, 1.0)), [])

    def test_unreadable_create_answer_adopts_the_rented_instance(self):
        # vast rented the machine but its answer could not be read: the launcher finds it by its
        # label and adopts it instead of renting the next offer too.
        w = World([offer(1, 11, 1.0), offer(2, 12, 1.5)], ["done:2"], create_fail={1: "unreadable"})
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}})
        self.assertEqual((code, report["outcome"]), (0, "DONE"))
        self.assertEqual(w.created_on, [1])
        self.assertEqual(len(w.instances), 1)
        self.assertIn("adopting instance 100", self.log)
        # An unreadable answer and no machine: the next offer, as before.
        w = World([offer(1, 11, 1.0), offer(2, 12, 1.5)], ["done:2"], create_fail={1: "none"})
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}})
        self.assertEqual((code, w.created_on), (0, [2]))
        # An unreadable answer and an unreadable instance list: stop, never rent another.
        w = World([offer(1, 11, 1.0), offer(2, 12, 1.5)], ["done:2"], create_fail={1: "none"})
        w.listing_fails = True
        with self.assertRaises(SystemExit) as e:
            self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}})
        self.assertIn("second rental", str(e.exception))
        self.assertEqual(w.created_on, [])

    def test_replacement_with_unreadable_create_answer_adopts_the_new_machine_only(self):
        # Format 2: the first machine dies; the replacement's create answer cannot be read, but
        # vast rented it. The new machine is adopted (never the dead one, still listed as exited
        # had its destroy not gone through) and the run finishes on it.
        w = World([offer(1, 11, 1.0), offer(2, 12, 1.5)], ["die:2", "done:1"], create_fail={2: "unreadable"})
        code, report = self.run_launch(w, {"format": 2, "run_id": RUN, "run_seed": 1, "streams": []})
        self.assertEqual((code, report["outcome"], report["replacements"]), (0, "DONE", 1))
        self.assertEqual(w.created_on, [1, 2])
        self.assertEqual([x["instance_id"] for x in report["attempts"]], [100, 101])
        self.assertIn("adopting instance 101", self.log)
        self.assertNotIn("adopting instance 100", self.log)

    def test_vast_errors_never_mean_gone(self):
        # vast's API is down for the whole run: a healthy, working machine is never declared
        # gone; the run ends at its time limit (stop and upload), and the destroy, whose
        # verification cannot be read, is reported as unverified, not as done.
        w = World([offer(1, 11, 1.0)], ["work:0"])
        w.status_fails = w.listing_fails = True
        code, report = self.run_launch_minutes(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}}, 5)
        self.assertEqual((code, report["outcome"]), (1, "STOPPED"))
        self.assertIn("max_minutes", report["reason"])
        self.assertNotIn("died", report["reason"])
        self.assertEqual(report["destroy"], "unverified")
        self.assertEqual(report["attempts"][0]["destroy"], "unverified")
        self.assertFalse(w.instances[100]["alive"])  # the destroy call itself went through
        md = launch.summary_markdown(RUN, report, [])
        self.assertIn("destroy not verified** for instance(s) 100", md)
        # Verified when the status can be read.
        w = World([offer(1, 11, 1.0)], ["done:1"])
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}})
        self.assertEqual((code, report["destroy"]), (0, "verified"))
        self.assertIn("(verified: actual_status exited)", self.log)

    def test_dead_machine_is_declared_gone_only_after_three_readings_over_90_s(self):
        w = World([offer(1, 11, 1.0)], ["die:2"])
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}})
        self.assertEqual((code, report["outcome"]), (1, "KILLED"))
        self.assertIn("instance died before DONE (actual_status exited: reading 3 of 3 over 120 s)", report["reason"])
        self.assertIn("looks dead (actual_status exited: reading 1 of 3 over 0 s); not yet confirmed", self.log)
        self.assertGreaterEqual(report["attempts"][0]["minutes"], 2 + 2)
        # The pure parts: one reading never says gone; an unreadable one changes nothing.
        with self.patched(w):
            self.assertEqual(launch.instance_state(100)[0], "dead")
            w.status_fails = w.listing_fails = True
            self.assertEqual(launch.instance_state(100), ("unknown", None))
            live = launch.Liveness(100)
            self.assertEqual(live.check()[0], "unknown")
            w.status_fails = w.listing_fails = False
            self.assertEqual(live.check()[0], "dying")
            w.status_fails = True
            self.assertEqual(live.check()[0], "dying")  # not listed: a reading, 2 of 3
            w.t += 100
            self.assertEqual(live.check()[0], "gone")
            self.assertIn("not listed: reading 3 of 3 over 100 s", live.detail)
            # A reading that shows it alive clears the count.
            w.instances[100]["alive"] = True
            w.status_fails = False
            self.assertEqual(live.check()[0], "alive")
            self.assertEqual(live.count, 0)

    def test_r2_failure_while_polling_is_no_new_heartbeat(self):
        # R2 cannot be listed for the first three polls: logged, counted as no new heartbeat,
        # and the run goes on to DONE (no exception reaches the destroy).
        w = World([offer(1, 11, 1.0)], ["done:5"])
        w.r2_list_fails_until = w.t + 3 * launch.POLL_S + 1
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}})
        self.assertEqual((code, report["outcome"]), (0, "DONE"))
        self.assertIn("could not read the run's heartbeats this poll", self.log)
        self.assertIn("counting it as no new heartbeat", self.log)
        self.assertNotIn("Traceback", self.log)

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

    def test_silent_busy_machine_gets_time_then_stops_and_uploads(self):
        # Heartbeats stop but vast shows the machine busy (95% CPU): an upload problem, not a dead
        # machine. The launcher prints its log, waits 15 more minutes, then asks it to stop and
        # upload what it finished, and only then destroys it.
        w = World([offer(1, 11, 1.0)], ["silentbusy:2"])
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}})
        self.assertEqual((code, report["outcome"]), (1, "STOPPED"))
        self.assertIn("stayed busy", report["reason"])
        self.assertIn("vast CPU 95%", report["reason"])
        self.assertIn(f"runs/{RUN}/STOP", w.store)
        self.assertIn(f"runs/{RUN}/out/measure/exact.jsonl", w.store)
        self.assertGreaterEqual(report["attempts"][0]["minutes"], 2 + 3 + 15)
        self.assertIn("treating it as an upload problem", self.log)
        self.assertIn("vast container log of instance 100", self.log)
        # --silent-busy-min shortens the wait.
        w = World([offer(1, 11, 1.0)], ["silentbusy:2"])
        code, short = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}},
                                      "--silent-busy-min", "5")
        self.assertEqual(short["outcome"], "STOPPED")
        self.assertLess(short["attempts"][0]["minutes"], report["attempts"][0]["minutes"] - 8)

    def test_silent_busy_machine_that_recovers_is_kept(self):
        # Six silent minutes while busy, then heartbeats resume and the run finishes.
        w = World([offer(1, 11, 1.0)], ["hiccup:2:8"])
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}})
        self.assertEqual((code, report["outcome"]), (0, "DONE"))
        self.assertNotIn(f"runs/{RUN}/STOP", w.store)
        self.assertIn("heartbeats resumed", self.log)

    def test_silent_idle_or_gone_machine_is_destroyed_at_once(self):
        for script, word in (("silentidle:2", "idle (running, vast CPU 2%"), ("stall:2", "no CPU reading"),
                             ("die:2", "instance died")):
            w = World([offer(1, 11, 1.0)], [script])
            code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}})
            self.assertEqual((code, report["outcome"]), (1, "KILLED"), script)
            self.assertIn(word, report["reason"], script)
            self.assertNotIn(f"runs/{RUN}/STOP", w.store, script)
            self.assertLess(report["attempts"][0]["minutes"], 2 + 3 + 3, script)

    def test_vast_cpu_reading(self):
        c = launch.vast_cpu_pct
        self.assertIsNone(c({}))
        self.assertIsNone(c({"cpu_util": None}))
        self.assertEqual(c({"cpu_util": 0.9}), 0.9)  # a small percentage: an idle machine
        self.assertEqual(c({"cpu_util": 75.0}), 75.0)
        self.assertEqual(c({"cpu_util": 103.0}), 100.0)  # clamped
        st = launch.silent_state
        # One reading never says gone: no record or a state other than running is unknown.
        self.assertEqual(st(None, 50)[0], "unknown")
        self.assertEqual(st({"actual_status": "exited", "cpu_util": 99}, 50)[0], "unknown")
        self.assertEqual(st({"actual_status": "running", "cpu_util": 60}, 50)[0], "busy")
        self.assertEqual(st({"actual_status": "running", "cpu_util": 10}, 50)[0], "idle")
        self.assertEqual(st({"actual_status": "running"}, 50)[0], "idle")

    def test_budget_stops_and_uploads_and_warns_on_projection(self):
        w = World([offer(1, 11, 1.0)], ["work:0"])
        w.progress = {"eta": {"done_frac": 0.25, "eta_utc": "2026-10-14T15:20:00Z", "projected_usd": 0.5,
                              "basis": "chunks done"}}
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}},
                                       "--max-dollars", "0.1")
        self.assertEqual((code, report["outcome"]), (1, "STOPPED"))
        self.assertIn("budget reached: $0.10 of $0.10 spent", report["reason"])
        self.assertIn(f"runs/{RUN}/STOP", w.store)
        # Six minutes at $1/hour, then the stop.
        self.assertLess(report["attempts"][0]["minutes"], 6 + 2)
        # The projection is shown on every poll line and warned about once.
        self.assertEqual(self.log.count("::warning::projected cost $0.50 is more than 20% over the budget"), 1)
        self.assertIn("25.0% done, ETA 14 Oct 15:20 UTC (14 Oct 19:20 Mauritius), projected $0.50, basis: chunks done",
                      self.log)
        self.assertRegex(self.log, r"heartbeat 0, phase chunk, [0-9.]+ min, \$0\.0[0-9] so far, CPU 99\.0%")
        # No budget (0; the default is $20): no warning, no stop for money.
        w = World([offer(1, 11, 1.0)], ["done:3"])
        w.progress = {"eta": {"projected_usd": 50.0}}
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}},
                                       "--max-dollars", "0")
        self.assertEqual((code, report["outcome"]), (0, "DONE"))
        self.assertNotIn("over the budget", self.log)

    def test_budget_rents_no_replacement(self):
        # The first machine dies at 3 minutes and is confirmed gone at 6 ($1.20 at $12/hour), just
        # under the budget; with its destroy it has used the budget, so no replacement is rented.
        w = World([offer(1, 11, 12.0), offer(2, 12, 12.0)], ["die:3", "done:1"])
        code, report = self.run_launch(w, {"format": 2, "run_id": RUN, "run_seed": 1, "streams": []},
                                       "--max-dollars", "1.21")
        self.assertEqual(code, 1)
        self.assertEqual(w.created_on, [1])
        self.assertIn("no replacement", report["reason"])

    def test_step_summary_lists_what_was_saved(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "summary.md")
            with _env({"GITHUB_STEP_SUMMARY": path}):
                w = World([offer(1, 11, 1.0)], ["done:2"])
                w.outputs = {f"runs/{RUN}/out/t1bench/results.json": b"x" * 2048,
                             f"runs/{RUN}/out/t1bench/summary.txt": b"y" * 100}
                code, _ = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}})
            with open(path) as fh:
                md = fh.read()
        self.assertEqual(code, 0)
        self.assertIn(f"## vast.ai run {RUN}: DONE", md)
        self.assertIn("- **Outcome:** DONE: DONE", md)
        self.assertIn("minutes on 1 machine(s); **cost:** about $0.0", md)
        self.assertIn(f"- **Saved in R2** `runs/{RUN}/out/`: 2 file(s), 2.1 KB", md)
        self.assertIn("<details><summary>Full report</summary>", md)
        self.assertNotIn("partial/", md)

    def test_container_start_error(self):
        e = launch.container_start_error
        self.assertIsNone(e(None))
        self.assertIsNone(e({"actual_status": "loading", "status_msg": "Pulling image"}))
        self.assertIsNone(e({"actual_status": "running", "status_msg": "success, running img"}))
        self.assertIn("OCI runtime", e({"actual_status": "created", "status_msg": "x: OCI runtime create failed"}))
        self.assertIn("failed to inject", e({"status_msg": "Failed To Inject CDI devices"}).lower())
        # Exited or offline is not a container error by one reading: `Liveness` confirms it.
        self.assertIsNone(e({"actual_status": "exited", "status_msg": ""}))
        self.assertIsNone(e({"actual_status": "offline", "status_msg": "host down"}))

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

    def test_long_run_is_handed_over_and_keeps_working(self):
        # GitHub's 6-hour limit: after --handover-min the launcher leaves the machine working,
        # tells the cleanup (watch/<run>.json) and clears the workflow's safety-net files.
        import tempfile
        w = World([offer(1, 11, 1.0)], ["work:0"])
        with tempfile.TemporaryDirectory() as d:
            state = os.path.join(d, "instance-id.txt")
            code, report = self.run_launch(w, {"format": 2, "run_id": RUN, "run_seed": 1, "streams": []},
                                           "--state-file", state, "--handover-min", "30", "--max-dollars", "20")
            self.assertEqual([f for f in os.listdir(d)], [])
        self.assertEqual((code, report["outcome"]), (0, "HANDED_OVER"))
        self.assertTrue(w.instances[100]["alive"])
        self.assertNotIn(f"runs/{RUN}/STOP", w.store)
        self.assertEqual(w.deleted_keys, [])  # the machine still needs its key to destroy itself
        rec = json.loads(w.store[f"watch/{RUN}.json"])
        self.assertEqual(rec, json.loads(w.store[f"runs/{RUN}/watch.json"]))
        self.assertEqual((rec["instance_id"], rec["dph"], rec["max_dollars"], rec["handed_over"], rec["destroy_key_id"]),
                         (100, 1.0, 20.0, True, 77))
        self.assertTrue(rec["deadline_utc"])

    def test_every_machine_gets_a_watch_record_and_self_destroy_is_reported(self):
        w = World([offer(1, 11, 1.0)], ["done:2"])
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}},
                                       "--max-dollars", "5")
        self.assertEqual((code, report["outcome"], report["self_destroyed"]), (0, "DONE", True))
        rec = json.loads(w.store[f"runs/{RUN}/watch.json"])
        self.assertEqual((rec["max_dollars"], rec["handed_over"]), (5.0, False))
        self.assertNotIn(f"watch/{RUN}.json", w.store)

    # ---- review E (2026-10-16)

    def test_no_time_limit_by_default_and_a_20_dollar_cap(self):
        # The owner's rule: a run ends when it finishes, at its dollar cap, or when a person stops
        # it. Without --max-minutes, a run working for 50 minutes (the old default stopped it at
        # 30) finishes, and its manifest carries no deadline for the runner to stop at.
        w = World([offer(1, 11, 1.0)], ["done:50"])
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}},
                                       time_limit=False)
        self.assertEqual((code, report["outcome"]), (0, "DONE"))
        self.assertGreater(report["attempts"][0]["minutes"], 50)
        self.assertNotIn(f"runs/{RUN}/STOP", w.store)
        self.assertNotIn("deadline_utc", json.loads(w.store[f"runs/{RUN}/manifest.json"]))
        rec = json.loads(w.store[f"runs/{RUN}/watch.json"])
        self.assertEqual((rec["max_dollars"], rec["deadline_utc"]), (20.0, None))
        # The cap is what ends a run that never finishes: $20 at $10/hour is 2 hours.
        w = World([offer(1, 11, 10.0)], ["work:0"])
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}},
                                       time_limit=False)
        self.assertEqual((code, report["outcome"]), (1, "STOPPED"))
        self.assertIn("budget reached", report["reason"])
        self.assertAlmostEqual(report["attempts"][0]["minutes"], 120, delta=8)

    def test_stop_of_a_stopped_machine_does_not_stop_its_replacement(self):
        # Format 2: the first machine goes silent while busy and ignores the stop request (its
        # network is stuck), so it is destroyed and replaced. The replacement must not find that
        # STOP: before, it stopped at its first heartbeat and the run ended with its work undone.
        w = World([offer(1, 11, 1.0), offer(2, 12, 1.1)], ["stuck:2", "obeys:3"])
        code, report = self.run_launch(w, {"format": 2, "run_id": RUN, "run_seed": 1, "streams": []})
        self.assertEqual((code, report["outcome"], report["replacements"]), (0, "DONE", 1))
        self.assertNotIn(f"runs/{RUN}/FAILED", w.store)
        self.assertIn("removed the stop request", self.log)

    def test_silent_machine_of_a_low_cpu_run_gets_the_busy_grace(self):
        # A run that keeps 32 of 190 threads busy (--low-cpu-pct 1) reads about 17% in vast. When
        # its heartbeat uploads hang it is treated as busy (wait, then stop and upload), not as
        # idle (destroyed at once with its unsaved work).
        w = World([offer(1, 11, 1.0)], ["silentlow:2"])
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}},
                                       "--low-cpu-pct", "1")
        self.assertEqual((code, report["outcome"]), (1, "STOPPED"))
        self.assertIn("stayed busy", report["reason"])
        self.assertIn(f"runs/{RUN}/out/measure/exact.jsonl", w.store)
        self.assertEqual(json.loads(w.store[f"runs/{RUN}/watch.json"])["busy_cpu_pct"], 1.0)
        # With the default threshold the same machine is idle: destroyed at once, as before.
        w = World([offer(1, 11, 1.0)], ["silentlow:2"])
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}})
        self.assertEqual((code, report["outcome"]), (1, "KILLED"))

    def test_stream_keys_the_runner_refuses_are_refused_before_renting(self):
        w = World([offer(1, 11, 1.0)], ["done:1"])
        m = {"format": 2, "run_id": RUN, "run_seed": 1, "streams": [
            {"name": "tt12g", "priority": 1, "kind": "search_measure", "threads": 32, "chunks": 1,
             "params": {"tt_mb": 12288}}]}
        with self.assertRaises(SystemExit) as e:
            self.run_launch(w, m)
        self.assertIn("unknown key 'threads' (it belongs in params: params.threads)", str(e.exception))
        self.assertEqual(w.created_on, [])
        m["streams"][0]["params"]["threads"] = m["streams"][0].pop("threads")
        self.assertEqual(launch.manifest_stream_errors(m), [])

    def test_gpu_job_query_keeps_every_safety_filter(self):
        a = argparse.Namespace(
            min_reliability=0.98, min_cores=128, min_ram_gb=64, disk_gb=80, max_net_cost=0.1, max_dph=None,
            cpu_only=True, gpus=1, min_gpu_ram_gb=4, min_compute_cap=600)
        q = launch.offer_query(a).split()
        for want in ("num_gpus=1", "gpu_ram>=4", "compute_cap>=600", "compute_cap<1000", "cuda_vers>=12.6",
                     "gpu_frac=1", "verified=true", "rentable=true", "reliability>0.98", "cpu_cores_effective>=128",
                     "cpu_ram>=64", "disk_space>=80"):
            self.assertIn(want, q)
        self.assertNotIn("num_gpus>=1", q)
        # A higher capability floor is kept; one below the build's 6.0 is raised to it.
        a.min_compute_cap = 860
        self.assertIn("compute_cap>=860", launch.offer_query(a).split())
        a.min_compute_cap = 500
        self.assertIn("compute_cap>=600", launch.offer_query(a).split())
        # CPU jobs are unchanged: any GPU, and the PyTorch filters only without --cpu-only.
        a.gpus, a.min_gpu_ram_gb = 0, 0
        q = launch.offer_query(a).split()
        self.assertIn("num_gpus>=1", q)
        self.assertFalse(any(x.startswith(("compute_cap", "cuda_vers", "gpu_ram")) for x in q), q)
        a.cpu_only = False
        self.assertIn("cuda_vers>=12.6", launch.offer_query(a).split())

    def test_manifest_machine_needs_rent_one_gpu_machine(self):
        # The hybrid rung's manifest asks for 1 GPU with 4 GB, 128 cores and 80 GB of disk. Under the
        # workflow's default --cpu-only the GPU filters still apply; offers without a usable GPU
        # (too little memory, a Blackwell card the cu126 build cannot run, an old driver, two GPUs)
        # are skipped even though the search returned them, and the cheapest one that fits is rented.
        a2000 = dict(num_gpus=1, gpu_name="RTX A2000", gpu_ram=6138, compute_cap=860, cuda_max_good=12.8,
                     cpu_cores_effective=384, cpu_name="AMD EPYC 9654 96-Core Processor")
        offers = [offer(1, 11, 0.4, **{**a2000, "gpu_ram": 2048}),
                  offer(2, 12, 0.5, **{**a2000, "compute_cap": 1200, "gpu_name": "RTX 5090"}),
                  offer(3, 13, 0.6, **{**a2000, "cuda_max_good": 12.4}),
                  offer(4, 14, 0.7, **{**a2000, "num_gpus": 2}),
                  offer(5, 15, 1.02, **a2000),
                  offer(6, 16, 1.5, **a2000)]
        w = World(offers, ["done:2"])
        m = {"format": 2, "run_id": RUN, "run_seed": 1,
             "machine": {"gpus": 1, "min_gpu_ram_gb": 4, "min_cores": 128, "disk_gb": 80},
             "streams": [{"name": "rung", "priority": 1, "kind": "hybrid_rung", "chunks": 1,
                          "params": {"threads": "all"}}]}
        code, report = self.run_launch(w, m, "--cpu-only", "--cheapest")
        self.assertEqual((code, report["outcome"]), (0, "DONE"))
        self.assertEqual(w.created_on, [5])
        self.assertEqual(w.disks, ["80"])
        q = w.queries[0].split()
        for want in ("num_gpus=1", "gpu_ram>=4", "compute_cap>=600", "compute_cap<1000", "cuda_vers>=12.6",
                     "cpu_cores_effective>=128", "disk_space>=80", "gpu_frac=1", "verified=true"):
            self.assertIn(want, q)
        self.assertIn("--cpu-only is ignored", self.log)
        self.assertIn("RTX A2000 (6138 MB, cc 860, CUDA 12.8)", self.log)

    def test_offer_chosen_by_id_must_meet_the_gpu_needs(self):
        a2000 = dict(num_gpus=1, gpu_name="RTX A2000", gpu_ram=6138, compute_cap=860, cuda_max_good=12.8)
        m = {"run_id": RUN, "machine": {"gpus": 1, "min_gpu_ram_gb": 4},
             "chunk": {"difftest": {"matches": 1, "seed": 1}}}
        cases = [({"compute_cap": 1200}, "compute capability 1200 outside 600..999"),
                 ({"compute_cap": 520}, "compute capability 520"),
                 ({"gpu_ram": 3000}, "GPU memory 3000 MB, below --min-gpu-ram-gb 4"),
                 ({"num_gpus": 4}, "4 GPUs, not the 1 asked for"),
                 ({"cuda_max_good": 12.2}, "driver supports CUDA 12.2, below 12.6")]
        for kw, word in cases:
            w = World([offer(500, 50, 1.0, **{**a2000, **kw})], ["done:1"])
            w.forbid_search = True
            with self.assertRaises(SystemExit) as e:
                self.run_launch(w, m, "--offer-id", "500")
            self.assertIn(word, str(e.exception), kw)
            self.assertEqual(w.created_on, [], kw)
        w = World([offer(500, 50, 1.0, **a2000)], ["done:1"])
        w.forbid_search = True
        code, _ = self.run_launch(w, m, "--offer-id", "500")
        self.assertEqual((code, w.created_on), (0, [500]))
        # The same needs from the command line.
        w = World([offer(500, 50, 1.0, **{**a2000, "gpu_ram": 3000})], ["done:1"])
        w.forbid_search = True
        with self.assertRaises(SystemExit):
            self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}},
                            "--offer-id", "500", "--gpus", "1", "--min-gpu-ram-gb", "4")
        self.assertEqual(w.created_on, [])

    def test_manifest_machine_object_is_checked_before_renting(self):
        base = {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}}
        for machine, word in (({"gpu": 1}, "unknown keys ['gpu']"), ({"gpus": "one"}, "machine.gpus must be a number"),
                              ({"disk_gb": -1}, "machine.disk_gb must be a number"),
                              ({"gpus": 1.5}, "machine.gpus must be a whole number"), ([1], "must be an object")):
            w = World([offer(1, 11, 1.0)], ["done:1"])
            with self.assertRaises(SystemExit) as e:
                self.run_launch(w, {**base, "machine": machine})
            self.assertIn(word, str(e.exception), machine)
            self.assertEqual((w.created_on, w.searches), ([], 0), machine)
        w = World([offer(1, 11, 1.0)], ["done:1"])
        with self.assertRaises(SystemExit) as e:
            self.run_launch(w, {**base, "machine": {"gpus": 1}}, "--gpus", "2")
        self.assertIn("the manifest asks for 1 GPUs but --gpus says 2", str(e.exception))
        # Minimums take the larger of the two.
        a = argparse.Namespace(gpus=0, min_gpu_ram_gb=8, min_compute_cap=600, min_cores=256,
                                                          min_ram_gb=32, disk_gb=20, cpu_only=False)
        with contextlib.redirect_stdout(io.StringIO()):
            launch.apply_machine(a, {"machine": {"gpus": 1, "min_gpu_ram_gb": 4, "min_cores": 128, "min_ram_gb": 64,
                                                 "disk_gb": 80}})
        self.assertEqual((a.gpus, a.min_gpu_ram_gb, a.min_cores, a.min_ram_gb, a.disk_gb), (1, 8, 256, 64, 80))

    def test_poll_line_shows_each_chunks_step(self):
        hb = {"seq": 7, "phase": "chunks", "progress": {"chunk_steps": {
            "rung/0": {"step": "train", "train_epoch": 3, "train_epochs": 60, "train_batches_done": 1200,
                       "train_batches": 3907, "device": "cuda"}}}}
        line = launch.poll_line(hb, 12.0, 0.2, 2.0)
        self.assertIn("rung/0: train (epoch 4/60, 1,200/3,907 batches, on cuda)", line)
        hb["progress"]["chunk_steps"]["rung/0"] = {"step": "labels random (1 of 3)", "rows_done": 250000,
                                                   "rows_total": 3200000}
        self.assertIn("rung/0: labels random (1 of 3) (250,000/3,200,000 rows)", launch.poll_line(hb, 1.0, 0.0, 99.0))

    def test_offer_chosen_by_id_must_have_the_ram_and_cores_asked_for(self):
        # Review E: --min-ram-gb and --min-cores apply to a named offer too (vast's cpu_ram in MB).
        m = {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}}
        w = World([offer(500, 50, 1.0, cpu_ram=257_000)], ["done:1"])
        w.forbid_search = True
        with self.assertRaises(SystemExit) as e:
            self.run_launch(w, m, "--offer-id", "500", "--min-ram-gb", "480")
        self.assertIn("257 GB of RAM, below --min-ram-gb 480", str(e.exception))
        with self.assertRaises(SystemExit) as e:
            self.run_launch(w, m, "--offer-id", "500", "--min-cores", "64")
        self.assertIn("32 effective cores, below --min-cores 64", str(e.exception))
        self.assertEqual(w.created_on, [])
        w = World([offer(500, 50, 1.0, cpu_ram=515_000)], ["done:1"])
        w.forbid_search = True
        code, _ = self.run_launch(w, m, "--offer-id", "500", "--min-ram-gb", "480")
        self.assertEqual((code, w.created_on), (0, [500]))

    def test_machine_record_of_a_destroyed_run_is_marked_final(self):
        # Review E: a machine destroyed silent left its record `running` for ever.
        w = World([offer(1, 11, 1.0)], ["silentidle:2"])
        w.store[f"machines/{RUN}.json"] = json.dumps({"run_id": RUN, "state": "running", "nodes_total": 10 ** 12,
                                                      "usd": 0.01, "minutes": 1.0}).encode()
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}})
        self.assertEqual((code, report["outcome"]), (1, "KILLED"))
        rec = json.loads(w.store[f"machines/{RUN}.json"])
        self.assertEqual((rec["state"], rec["outcome"], rec["finalised_by"]), ("final", "KILLED", "launcher"))
        self.assertIsNotNone(rec["usd_per_1e12_nodes"])
        # A final record is left alone.
        w = World([offer(1, 11, 1.0)], ["done:2"])
        w.store[f"machines/{RUN}.json"] = json.dumps({"run_id": RUN, "state": "final", "outcome": "DONE"}).encode()
        self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}})
        self.assertEqual(json.loads(w.store[f"machines/{RUN}.json"]), {"run_id": RUN, "state": "final", "outcome": "DONE"})

    def test_failed_search_for_a_replacement_still_reports(self):
        # vast's daily search quota runs out (429) when a replacement is needed: nothing more is
        # rented and the run's report is written (before, the exception skipped it).
        w = World([offer(1, 11, 1.0), offer(2, 12, 1.5)], ["die:2", "done:1"])
        orig = w.vast

        def vast(*args, **kw):
            if args[:2] == ("destroy", "instance"):
                w.search_fails = True
            return orig(*args, **kw)
        w.vast = vast
        code, report = self.run_launch(w, {"format": 2, "run_id": RUN, "run_seed": 1, "streams": []})
        self.assertEqual(code, 1)
        self.assertEqual(w.created_on, [1])
        self.assertIn("no offer could be rented", report["reason"])
        self.assertIn("the offer search failed", self.log)

    def test_create_error_whose_machine_appears_late_is_adopted_not_rented_twice(self):
        # The create call fails, but vast rented the machine and lists it only from its second
        # listing: it is adopted; the next offer is not rented as well.
        w = World([offer(1, 11, 1.0), offer(2, 12, 1.5)], ["done:2", "done:2"], create_fail={1: "late"})
        code, report = self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}})
        self.assertEqual((code, report["outcome"]), (0, "DONE"))
        self.assertEqual(w.created_on, [1])
        self.assertIn("adopting instance 100", self.log)

    def destroy_world(self, alive: bool, destroyed: bool = False) -> World:
        """A world holding only instance 100: running, or ended (and, if `destroyed`, gone from vast)."""
        w = World([], [])
        w.instances[100] = {"alive": alive, "polls": 0, "script": "work:0", "offer": 1, "created": w.t,
                            "label": f"41bots-{RUN}", "destroyed": destroyed}
        return w

    def run_destroy(self, w: World) -> tuple[str, str]:
        """launch.destroy(100) against `w`: its answer and what it logged."""
        out = io.StringIO()
        with self.patched(w):
            with contextlib.redirect_stdout(out):
                result = launch.destroy(100)
        return result, out.getvalue()

    def test_destroy_of_a_machine_that_already_destroyed_itself_is_verified(self):
        # 2026-10-16: the runner's own self-destroy (or a stop that raced it) removed the instance,
        # so vast no longer knows it and every destroy call fails. That is a success, not an error:
        # once the listing (read successfully) does not hold the instance, it is verified.
        for status_fails in (True, False):  # vast's `show instance` errors for an unknown id, or says exited
            with self.subTest(status_fails=status_fails):
                w = self.destroy_world(alive=False, destroyed=True)
                w.destroy_fails = 10 ** 6
                w.status_fails = status_fails
                result, out = self.run_destroy(w)
                self.assertEqual(result, "verified")
                self.assertEqual(w.destroy_calls, 1)  # no pointless retries
                self.assertIn("destroy attempt 1 failed", out)
                self.assertIn("was already gone when the destroy call failed", out)
                self.assertNotIn("::error::", out)
                self.assertNotIn("::warning::", out)

    def test_failed_destroy_with_an_unreadable_status_is_retried_and_ends_failed(self):
        # The safety rule: an unreadable status (the instance listing itself fails) is never counted
        # as destroyed, even when the machine really is gone. 5 attempts, then "failed" with the error.
        w = self.destroy_world(alive=False, destroyed=True)
        w.destroy_fails = 10 ** 6
        w.status_fails = w.listing_fails = True
        result, out = self.run_destroy(w)
        self.assertEqual(result, "failed")
        self.assertEqual(w.destroy_calls, 5)
        self.assertIn("destroy attempt 5 failed", out)
        self.assertIn("::error::could not destroy instance 100; destroy it by hand", out)
        self.assertNotIn("already gone", out)
        self.assertNotIn("verified", out)

    def test_failed_destroy_of_an_instance_still_there_is_retried_and_ends_failed(self):
        # The destroy call fails and a clean reading shows the instance still running: not gone.
        w = self.destroy_world(alive=True)
        w.destroy_fails = 10 ** 6
        result, out = self.run_destroy(w)
        self.assertEqual(result, "failed")
        self.assertEqual(w.destroy_calls, 5)
        self.assertIn("::error::could not destroy instance 100", out)
        self.assertNotIn("already gone", out)
        # A destroy call that fails twice and then goes through is the normal path after a retry.
        w = self.destroy_world(alive=True)
        w.destroy_fails = 2
        result, out = self.run_destroy(w)
        self.assertEqual((result, w.destroy_calls), ("verified", 3))
        self.assertIn("destroyed instance 100 (verified: actual_status exited)", out)
        self.assertNotIn("::error::", out)

    def test_destroy_normal_path_is_unchanged(self):
        # The call goes through and a later reading shows the instance gone: verified, one call.
        w = self.destroy_world(alive=True)
        result, out = self.run_destroy(w)
        self.assertEqual((result, w.destroy_calls), ("verified", 1))
        self.assertIn("destroyed instance 100 (verified: actual_status exited)", out)
        self.assertNotIn("::error::", out)
        self.assertNotIn("already gone", out)
        self.assertTrue(w.instances[100]["destroyed"])
        # The call goes through but vast's status cannot be read: unverified, never "verified".
        w = self.destroy_world(alive=True)
        w.status_fails = w.listing_fails = True
        result, out = self.run_destroy(w)
        self.assertEqual((result, w.destroy_calls), ("unverified", 1))
        self.assertIn("::warning::destroy of instance 100 was sent, but vast's status could not be read", out)
        self.assertNotIn("::error::", out)

    def test_instance_state_is_dead_only_after_a_successful_reading(self):
        w = self.destroy_world(alive=False, destroyed=True)
        with self.patched(w):
            # Absent from a successfully read listing (and `show instance` erroring): dead.
            w.status_fails = True
            self.assertEqual(launch.instance_state(100), ("dead", None))
            # Neither call can be read: unknown, whether the machine is there or not.
            w.listing_fails = True
            self.assertEqual(launch.instance_state(100), ("unknown", None))
            # An unreadable (non-list) listing is unknown too.
            w.listing_fails = False
            orig = w.vast
            w.vast = lambda *a, **kw: {"error": "x"} if a[:2] == ("show", "instances") else orig(*a, **kw)
            launch.vast = w.vast
            self.assertEqual(launch.instance_state(100), ("unknown", None))

    def run_launch_minutes(self, world: World, manifest: dict, minutes: int) -> tuple[int, dict]:
        saved = {n: getattr(launch, n) for n in ("vast", "r2_get", "r2_put", "r2_list", "r2_exists", "r2_delete", "time")}
        launch.vast = world.vast
        launch.r2_get, launch.r2_put = world.r2_get, world.r2_put
        launch.r2_list, launch.r2_exists = world.r2_list, world.r2_exists
        launch.r2_delete = world.r2_delete
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
    # GITHUB_STEP_SUMMARY empty: tests run under CI must not write into the job's summary.
    return _env({"VAST_API_KEY": "vk-test-key-1", "R2_ACCESS_KEY_ID": "ak-test-id-2",
                 "R2_SECRET_ACCESS_KEY": "sk-test-secret-3",
                 "R2_BUCKET": "b", "R2_S3_ENDPOINT": "https://example.invalid", "GITHUB_STEP_SUMMARY": ""})


if __name__ == "__main__":
    unittest.main()
