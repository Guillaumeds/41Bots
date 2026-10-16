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
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch  # noqa: E402

RUN = "t-run"
# The settings the older tests were written for: machines of 16+ threads, no price cap to speak of, no price steps,
# and no waiting when nothing matches. Tests of the new behaviour pass their own values after these.
LEGACY = ["--min-cores", "16", "--max-dph", "1000", "--price-steps", "off", "--wait-for-offer-min", "0"]


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
        self.queries = []  # the query text of every full search
        self.forbid_search = False  # the account's daily search quota is used up: any full search is a bug
        self.status_fails = False  # `show instance` fails (vast's API down)
        self.listing_fails = False  # `show instances` fails
        self.status_none = False  # `show instance` answers a record whose actual_status is None (loading, or just destroyed)
        self.listed_status_none = False  # ...and the instance list shows None for the instances it holds
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
        self.listing_script: list[str] = []  # answers of the next `show instances` calls, one each: "empty" (a
        #                                      successfully read list without the instance), "fail", or "normal"
        self.listing_times: list[float] = []  # the clock at each `show instances` call
        self.status_times: list[float] = []  # ...and at each `show instance` call
        self.status_msg_fn = None  # f(instance, minutes since it was rented) -> the status_msg `show instance` answers
        self.search_empty_until = 0.0  # full searches answer an empty list while the clock is below this (no machine on offer)
        self.offer_appears = []  # [(clock, offer)]: an offer that is added to the market once the clock reaches `clock`

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
        if kind == "late":
            # Silent for n polls (its image was still being pulled), then heartbeats, done m polls later.
            if inst["polls"] <= n:
                return
            self.heartbeat("chunk", 99.0)
            if inst["polls"] >= n + int(args[1]):
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
            for when, o in list(self.offer_appears):
                if self.t >= when:
                    self.offers.append(o)
                    self.offer_appears.remove((when, o))
            rows = [dict(o) for o in self.offers]
            if self.t < self.search_empty_until:
                rows = []
            for tok in args[2].split():
                if tok.startswith("inet_down>="):
                    # vast filters on the download speed; an offer that reports none is kept here.
                    rows = [o for o in rows if o.get("inet_down") is None or o["inet_down"] >= float(tok[11:])]
                elif tok.startswith("dph_total<="):
                    rows = [o for o in rows if o["dph_total"] <= float(tok[11:])]
                elif tok.startswith("cpu_cores_effective>="):
                    rows = [o for o in rows if o["cpu_cores_effective"] >= float(tok[21:])]
            if "verified=true" in args[2].split():
                # vast's search then returns only verified hosts; `verified=any` lifts that.
                rows = [o for o in rows if o.get("verification", "verified") == "verified"]
            return rows
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
            if how == "unreadable":
                return None  # vast rented it, but printed something that is not JSON
            if how == "late":
                # vast rented it, its answer was an error, and the first listing misses it.
                self.hidden_listings = 1
                raise RuntimeError("vastai create instance failed: 504 Gateway Timeout")
            return {"success": True, "new_contract": iid, "instance_api_key": "ik-secret"}
        if cmd == ("show", "instances"):
            self.listing_times.append(self.t)
            step = self.listing_script.pop(0) if self.listing_script else "normal"
            if step == "empty":
                return []
            if self.listing_fails or step == "fail":
                raise RuntimeError("vastai show instances failed: 502 Bad Gateway")
            # Destroyed instances leave the list; one that died (or ended) but was not destroyed stays, exited.
            hide = None
            if self.hidden_listings > 0:
                self.hidden_listings -= 1
                hide = max(self.instances)
            return [{"id": iid, "label": inst["label"],
                     "actual_status": None if self.listed_status_none else "running" if inst["alive"] else "exited",
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
            self.status_times.append(self.t)
            if self.status_fails:
                raise RuntimeError("vastai show instance failed: 502 Bad Gateway")
            if self.status_none:
                return {"actual_status": None, "intended_status": "running", "status_msg": ""}
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
            if self.status_msg_fn is not None:
                msg = self.status_msg_fn(inst, (self.t - inst["created"]) / 60)
                return {"actual_status": "running" if msg.startswith("success") else "loading",
                        "intended_status": "running", "status_msg": msg}
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


def big(i: int, machine: int, dph: float, **kw) -> dict:
    """An offer that meets the launcher's thread floor (190): 224 threads."""
    return offer(i, machine, dph, **{"cpu_cores_effective": 224, **kw})


# The launcher's own defaults, spelled out (the older tests override them with LEGACY): 190 threads, $1.70 cap with
# the price steps below it, three attempts per step, a wait of two hours with a search every minute.
NEW = ["--min-cores", "190", "--max-dph", "1.7", "--price-steps", "1.2,1.45,1.7", "--tries-per-step", "3",
       "--wait-for-offer-min", "120", "--offer-poll-s", "60"]


def workflow_inputs(path: str) -> dict[str, dict]:
    """The `workflow_dispatch` inputs of a workflow file: name -> {field: value}, read as plain text (no YAML library)."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "inputs:")
    out, cur = {}, None
    for ln in lines[start + 1:]:
        if ln.strip() and not ln.startswith(" "):
            break
        m = re.match(r"^      ([A-Za-z0-9_]+):\s*$", ln)
        if m:
            cur = out[m.group(1)] = {}
            continue
        m = re.match(r"^        ([a-z_]+):\s*(.*)$", ln)
        if m and cur is not None:
            cur[m.group(1)] = m.group(2).strip().strip('"').strip("'")
        elif ln.strip() and not ln.startswith("        "):
            break
    return out


def pull_msgs(until: float | None = None, start: float | None = None):
    """What vast's status_msg says over time for a machine pulling the image: a docker layer line
    whose byte count grows each minute until minute `until` (then it freezes; None = never), and
    "success, running ..." from minute `start` (the container started; None = never)."""
    def msg(inst, minutes):
        if start is not None and minutes >= start:
            return "success, running img:1"
        m = minutes if until is None else min(minutes, until)
        return f"a1b2c3d4e5f6: Downloading [===>   ] {int(m * 10)}MB/900MB"
    return msg


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
                                    *limit, "--startup-minutes", "5", *LEGACY, *extra])
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
        # Shared, unreliable or interruptible (and unverified with --verified-only): refused by
        # name. Not on offer, or a lookup that fails: refused too (the checks are never skipped).
        # Nothing is rented.
        cases = [({"verification": "unverified"}, "host not verified", ["--verified-only"]),
                 ({"verification": "deverified"}, "host not verified", ["--verified-only"]),
                 ({"gpu_frac": 0.5}, "shared machine", []),
                 ({"reliability2": 0.85}, "reliability 0.850 below 0.9", []), ({"is_bid": True}, "interruptible", [])]
        for kw, word, extra in cases:
            w = World([offer(500, 50, 1.0, **kw)], ["done:1"])
            w.forbid_search = True
            with self.assertRaises(SystemExit) as e:
                self.run_launch(w, {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}}, "--offer-id", "500",
                                *extra)
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

    # ---- verified hosts are preferred, not required (owner, 2026-10-16)

    MANIFEST = {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}}

    def test_a_verified_offer_is_chosen_and_there_is_no_fallback_search(self):
        # The unverified offer is cheaper, but a verified offer fits: it is rented, in one search.
        w = World([offer(1, 11, 0.5, verification="unverified"), offer(2, 12, 2.0)], ["done:1"])
        code, report = self.run_launch(w, self.MANIFEST)
        self.assertEqual((code, w.created_on, w.searches), (0, [2], 1))
        self.assertIn("verified=true", w.queries[0].split())
        self.assertNotIn("no verified offer matches", self.log)
        self.assertIn("host verified", self.log)
        self.assertNotIn("NOT verified", self.log)

    def test_no_verified_offer_falls_back_to_unverified_offers_and_says_so(self):
        # The only verified offer is refused by --cpu-match, so nothing verified fits: the search is
        # repeated without the verified filter, and the best unverified/deverified offer is taken.
        w = World([offer(3, 13, 0.4, cpu_name="Xeon E5"),
                   offer(1, 11, 1.0, verification="unverified"), offer(2, 12, 0.5, verification="deverified")],
                  ["done:1"])
        code, report = self.run_launch(w, self.MANIFEST, "--cpu-match", "EPYC")
        self.assertEqual((code, w.created_on, w.searches), (0, [2], 2))
        self.assertIn("verified=true", w.queries[0].split())
        self.assertIn("verified=any", w.queries[1].split())
        self.assertNotIn("verified=true", w.queries[1].split())
        self.assertIn("no verified offer matches; using unverified/deverified offers (owner rule 2026-10-16)", self.log)
        self.assertIn("host NOT verified (deverified)", self.log)
        # Every other filter is unchanged in the second search, the Blackwell cut-off included.
        first, second = w.queries
        self.assertEqual(first.replace("verified=true", "verified=any"), second)
        self.assertIn("compute_cap<1000", second.split())
        self.assertIn("reliability>0.9", second.split())

    def test_fallback_search_with_nothing_either_stops_the_run(self):
        w = World([], ["done:1"])
        with self.assertRaises(SystemExit) as e:
            self.run_launch(w, self.MANIFEST)
        self.assertIn("no offer matches the filters", str(e.exception))
        self.assertEqual(w.searches, 2)
        self.assertIn("no offer matches even with unverified/deverified hosts allowed", self.log)

    def test_verified_only_keeps_the_strict_search(self):
        w = World([offer(1, 11, 1.0, verification="unverified")], ["done:1"])
        with self.assertRaises(SystemExit) as e:
            self.run_launch(w, self.MANIFEST, "--verified-only")
        self.assertIn("no offer matches the filters", str(e.exception))
        self.assertEqual((w.searches, w.created_on), (1, []))
        self.assertIn("--verified-only", self.log)
        self.assertNotIn("verified=any", " ".join(w.queries))
        # With a verified offer on the market, --verified-only still rents it.
        w = World([offer(1, 11, 1.0, verification="unverified"), offer(2, 12, 2.0)], ["done:1"])
        code, _ = self.run_launch(w, self.MANIFEST, "--verified-only")
        self.assertEqual((code, w.created_on, w.searches), (0, [2], 1))

    def test_offer_chosen_by_id_on_an_unverified_host_is_accepted_and_logged(self):
        for word in ("unverified", "deverified"):
            w = World([offer(500, 50, 1.013, cpu_cores_effective=384, verification=word)], ["done:2"])
            w.forbid_search = True
            code, report = self.run_launch(w, self.MANIFEST, "--offer-id", "500")
            self.assertEqual((code, w.created_on, w.searches, w.id_lookups), (0, [500], 0, 1), word)
            self.assertIn(f"offer 500 is on a host that is {word}, not verified: accepted", self.log)
            self.assertIn(f"host NOT verified ({word})", self.log)
        # Every other check still applies to it.
        w = World([offer(500, 50, 1.0, verification="unverified", reliability2=0.85)], ["done:1"])
        with self.assertRaises(SystemExit) as e:
            self.run_launch(w, self.MANIFEST, "--offer-id", "500")
        self.assertIn("reliability 0.850 below 0.9", str(e.exception))
        self.assertNotIn("host not verified", str(e.exception))
        self.assertEqual(w.created_on, [])

    def test_a_replacement_machine_follows_the_same_rule(self):
        # Format 2: the first machine (found by the fallback) dies. The replacement starts from the
        # verified search again, finds nothing verified, and falls back again, on another machine.
        w = World([offer(1, 11, 1.0, verification="unverified"), offer(2, 12, 1.5, verification="deverified")],
                  ["die:2", "done:1"])
        code, report = self.run_launch(w, {"format": 2, "run_id": RUN, "run_seed": 1, "streams": []})
        self.assertEqual((code, report["replacements"], w.created_on), (0, 1, [1, 2]))
        self.assertEqual(["verified=true" in q.split() for q in w.queries], [True, False, True, False])
        self.assertEqual(self.log.count("no verified offer matches; using unverified/deverified offers"), 2)
        # A verified machine that turns up before the replacement is preferred to the fallback.
        w = World([offer(1, 11, 1.0, verification="unverified")], ["die:2", "done:1"])
        orig = w.vast

        def vast_with_a_new_verified_offer(*args, **kw):
            if args[:2] == ("search", "offers") and not args[2].startswith("id=") and w.created_on:
                w.offers.append(offer(9, 19, 3.0))
            return orig(*args, **kw)

        w.vast = vast_with_a_new_verified_offer
        code, report = self.run_launch(w, {"format": 2, "run_id": RUN, "run_seed": 1, "streams": []})
        self.assertEqual((code, report["replacements"], w.created_on), (0, 1, [1, 9]))

    def test_plan_prints_the_host_status_of_the_offers(self):
        w = World([offer(1, 11, 1.0, verification="unverified")], [])
        saved = {n: getattr(launch, n) for n in ("vast", "r2_get", "r2_list", "r2_exists", "time")}
        launch.vast, launch.r2_get, launch.r2_list, launch.r2_exists, launch.time = w.vast, w.r2_get, w.r2_list, w.r2_exists, w
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                code = launch.main(["plan", "--min-cores", "16"])
        finally:
            for n, v in saved.items():
                setattr(launch, n, v)
        self.assertEqual(code, 0)
        self.assertIn("host NOT verified (unverified)", out.getvalue())
        self.assertIn("if no verified offer matches", out.getvalue())

    def test_reliability_floor_is_90_percent(self):
        # Owner, 2026-10-16: hosts with reliability of at least 90% are accepted, for every rental.
        self.assertEqual(launch.MIN_RELIABILITY, 0.90)
        manifest = {"run_id": RUN, "chunk": {"difftest": {"matches": 1, "seed": 1}}}
        # The default search asks vast for reliability above 0.9 (not 0.98); a run can still ask for more.
        w = World([offer(1, 11, 1.0)], ["done:1"])
        code, _ = self.run_launch(w, manifest)
        self.assertEqual(code, 0)
        self.assertIn("reliability>0.9", w.queries[0].split())
        self.assertNotIn("reliability>0.98", w.queries[0])
        w = World([offer(1, 11, 1.0)], ["done:1"])
        self.run_launch(w, manifest, "--min-reliability", "0.98")
        self.assertIn("reliability>0.98", w.queries[0].split())
        # The checks for a named offer (--offer-id): 0.95 and exactly 0.90 pass, 0.85 fails.
        for rel, ok in ((0.95, True), (0.90, True), (0.85, False)):
            self.assertEqual(launch.offer_check_failures(offer(7, 7, 1.0, reliability2=rel)),
                             [] if ok else ["reliability 0.850 below 0.9"], rel)
            w = World([offer(500, 50, 1.0, reliability2=rel)], ["done:1"])
            w.forbid_search = True
            if ok:
                code, report = self.run_launch(w, manifest, "--offer-id", "500")
                self.assertEqual((code, w.created_on), (0, [500]), rel)
            else:
                with self.assertRaises(SystemExit) as e:
                    self.run_launch(w, manifest, "--offer-id", "500")
                self.assertIn("reliability 0.850 below 0.9", str(e.exception))
                self.assertEqual(w.created_on, [], rel)

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

    # ---- waiting for an offer, the price ladder, floors that are never relaxed (owner, 2026-10-16, 18:10-18:20)

    def run_cli(self, world: World, argv: list[str]) -> tuple[int, str]:
        """launch.main(argv) against `world`; returns (exit code, what it printed)."""
        names = ("vast", "r2_get", "r2_put", "r2_list", "r2_list_sized", "r2_exists", "r2_delete", "time")
        saved = {n: getattr(launch, n) for n in names}
        launch.vast = world.vast
        launch.r2_get, launch.r2_put, launch.r2_list = world.r2_get, world.r2_put, world.r2_list
        launch.r2_list_sized, launch.r2_exists, launch.r2_delete = world.r2_list_sized, world.r2_exists, world.r2_delete
        launch.time = world
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                code = launch.main(argv)
        finally:
            for n, v in saved.items():
                setattr(launch, n, v)
        return code, out.getvalue()

    def rungs(self, world: World) -> list[float]:
        """The price limit (dph_total<=X) of each verified search, in order: one per waiting attempt."""
        return [float(re.search(r"dph_total<=([0-9.]+)", q).group(1)) for q in world.queries if "verified=true" in q.split()]

    def test_defaults_of_the_waiting_search(self):
        a = launch.build_parser().parse_args(["run", "--manifest", "{}", "--image", "x", "--bin-sha", "y"])
        self.assertEqual((a.min_cores, a.max_dph, a.price_steps, a.wait_for_offer_min), (190, None, None, None))
        launch.finalize_settings(a)
        self.assertEqual((a.max_dph, a.price_steps, a.tries_per_step, a.wait_for_offer_min, a.offer_poll_s),
                         (1.7, [1.2, 1.45, 1.7], 3, 120, 60))
        self.assertEqual(launch.price_ladder(a), [1.2, 1.45, 1.7])
        self.assertEqual(launch.MIN_CORES, 190)
        # The last rung is always the cap; steps at or above it are dropped; 0 = no cap; "off" = the cap alone.
        for argv, ladder in ((["--max-dph", "1.3"], [1.2, 1.3]), (["--max-dph", "3"], [1.2, 1.45, 1.7, 3.0]),
                             (["--max-dph", "0"], [1.2, 1.45, 1.7, 0.0]), (["--price-steps", "off"], [1.7]),
                             (["--price-steps", "1.5,1.0"], [1.0, 1.5, 1.7]), (["--max-dph", "0.9"], [0.9])):
            b = launch.build_parser().parse_args(["plan", *argv])
            launch.finalize_settings(b)
            self.assertEqual(launch.price_ladder(b), ladder, argv)
        self.assertEqual(launch.rung_text(0), "no price limit")
        for argv in (["--price-steps", "a,b"], ["--price-steps", "0"], ["--tries-per-step", "0"], ["--offer-poll-s", "0"],
                     ["--wait-for-offer-min", "-1"], ["--max-dph", "-1"]):
            with self.assertRaises(SystemExit, msg=argv):
                launch.finalize_settings(launch.build_parser().parse_args(["plan", *argv]))

    def test_the_workflow_asks_for_190_threads_and_ranks_by_cost_per_work_by_default(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            ".github", "workflows", "vast-run.yml")
        inputs = workflow_inputs(path)
        self.assertEqual(inputs["min_cores"]["default"], "190")
        self.assertEqual(inputs["cheapest"]["default"], "false")
        self.assertIn("cost per unit", inputs["cheapest"]["description"])
        self.assertIn("lowest $/hour", inputs["cheapest"]["description"])
        self.assertLessEqual(len(inputs), 25)  # GitHub's limit; the wait and the price steps go through the manifest, not inputs
        self.assertFalse({"wait_for_offer_min", "offer_poll_s", "price_steps", "tries_per_step", "max_dph"} & set(inputs))
        self.assertEqual(launch.build_parser().parse_args(["run", "--manifest", "{}", "--image", "x", "--bin-sha", "y"]).cheapest, False)

    def test_no_offer_waits_and_rents_when_one_comes(self):
        # Nothing on offer for 3.5 minutes: attempts at 0, 1, 2, 3 minutes find nothing, the one at 4 minutes rents.
        w = World([big(1, 11, 1.0)], ["done:2"])
        w.search_empty_until = w.t + 3.5 * 60
        code, report = self.run_launch(w, self.FORMAT1, *NEW, "--wait-for-offer-min", "10")
        self.assertEqual((code, report["outcome"], w.created_on), (0, "DONE", [1]))
        self.assertEqual(w.instances_created_at(100) - 1_000_000.0, 240)
        self.assertIn("no offer matches now; waiting up to 10 min for one, searching every 60 s (3 attempts per price step, "
                      "steps $1.20/h, $1.45/h, $1.70/h); floors: >= 190 threads, >= 32 GB RAM, reliability >= 0.9, "
                      "download >= 100 Mbps, price cap $1.70/h", self.log)
        self.assertIn("waiting for an offer: price step 1/3 ($1.20/h), attempt 1, 0.0 of 10 min; no match; closest miss: none", self.log)
        self.assertIn("waiting for an offer: price step 2/3 ($1.45/h), attempt 4, 3.0 of 10 min; no match", self.log)
        self.assertIn("offer found at price step 2/3 ($1.45/h) after waiting 4.0 min (attempt 5)", self.log)

    def test_the_wait_gives_up_at_the_limit_with_the_old_message(self):
        w = World([big(1, 11, 1.0)], ["done:1"])
        w.search_empty_until = w.t + 10_000_000
        with self.assertRaises(SystemExit) as e:
            self.run_launch(w, self.FORMAT1, *NEW, "--wait-for-offer-min", "8")
        self.assertEqual(str(e.exception), "launch: no offer matches the filters")
        self.assertEqual(w.created_on, [])
        self.assertIn("gave up waiting for an offer after 8 min (9 attempts; limit: 8 min)", self.log)
        self.assertEqual(self.rungs(w), [1.2] * 3 + [1.45] * 3 + [1.7] * 3)   # the ladder: 1.2 x3, 1.45 x3, then the cap
        # --wait-for-offer-min 0 is the old behaviour: one search, at the cap, then the same exit.
        w = World([big(1, 11, 1.0)], ["done:1"])
        w.search_empty_until = w.t + 10_000_000
        with self.assertRaises(SystemExit) as e:
            self.run_launch(w, self.FORMAT1, *NEW, "--wait-for-offer-min", "0")
        self.assertEqual((str(e.exception), self.rungs(w), w.t), ("launch: no offer matches the filters", [1.7], 1_000_000.0))

    def test_the_price_ladder_rents_a_1_40_offer_on_attempt_4_and_never_a_1_90_one(self):
        w = World([big(1, 11, 1.40)], ["done:2"])
        code, report = self.run_launch(w, self.FORMAT1, *NEW)
        self.assertEqual((code, w.created_on), (0, [1]))
        self.assertEqual(self.rungs(w), [1.2] * 3 + [1.45])             # not before attempt 4
        self.assertEqual(w.instances_created_at(100) - 1_000_000.0, 180)
        self.assertIn("offer found at price step 2/3 ($1.45/h) after waiting 3.0 min (attempt 4)", self.log)
        self.assertIn("closest miss: offer 1 has 224 threads but costs $1.40/h, above this step's $1.20/h "
                      "(the ladder reaches it at attempt 4)", self.log)
        # $1.90 is above the cap: waited for until the limit, never rented, whatever the step.
        w = World([big(1, 11, 1.90)], ["done:1"])
        with self.assertRaises(SystemExit):
            self.run_launch(w, self.FORMAT1, *NEW, "--wait-for-offer-min", "12")
        self.assertEqual(w.created_on, [])
        self.assertEqual(set(self.rungs(w)), {1.2, 1.45, 1.7})
        # ...unless the cap is raised: the ladder then goes on to it (here at attempt 10, after 9 minutes).
        w = World([big(1, 11, 1.90)], ["done:2"])
        code, _ = self.run_launch(w, self.FORMAT1, *NEW, "--max-dph", "2.5")
        self.assertEqual((code, w.created_on), (0, [1]))
        self.assertEqual(self.rungs(w), [1.2] * 3 + [1.45] * 3 + [1.7] * 3 + [2.5])
        self.assertEqual(w.instances_created_at(100) - 1_000_000.0, 540)
        # Among the offers that match at the current step, the launcher's own ranking decides: cost per unit of work, not
        # price per hour (offer 2 costs more per hour but gives twice the threads); --cheapest is the hourly price.
        offers = [big(1, 11, 1.00, cpu_cores_effective=200), big(2, 12, 1.19, cpu_cores_effective=400)]
        w = World(offers, ["done:2"])
        self.run_launch(w, self.FORMAT1, *NEW)
        self.assertEqual((w.created_on, self.rungs(w)), ([2], [1.2]))
        w = World(offers, ["done:2"])
        self.run_launch(w, self.FORMAT1, *NEW, "--cheapest")
        self.assertEqual(w.created_on, [1])

    def test_a_replacement_search_keeps_every_floor_and_waits_instead_of_taking_a_small_machine(self):
        # The first machine (224 threads) dies. Only a 120-thread machine is left: below the 190 floor, so it is not
        # rented; the launcher waits (the ladder starts again at its first step) and takes the big one that appears.
        m = {"format": 2, "run_id": RUN, "run_seed": 1, "streams": []}
        w = World([big(1, 11, 1.0), offer(2, 12, 0.5, cpu_cores_effective=120)], ["die:2", "done:1"])
        w.offer_appears = [(w.t + 9 * 60, big(3, 13, 1.1))]
        code, report = self.run_launch(w, m, *NEW)
        self.assertEqual((code, report["outcome"], w.created_on, report["replacements"]), (0, "DONE", [1, 3], 1))
        self.assertIn("closest miss: offer 2 costs $0.50/h but has only 120 threads (floor 190)", self.log)
        rungs = self.rungs(w)
        self.assertEqual(rungs[0], 1.2)                      # the first search
        self.assertEqual(rungs[1:4], [1.2, 1.2, 1.2])        # the replacement's ladder starts again from the first step
        # Nothing ever comes: the run ends with "no offer could be rented", and the 120-thread machine is never rented.
        w = World([big(1, 11, 1.0), offer(2, 12, 0.5, cpu_cores_effective=120)], ["die:2", "done:1"])
        code, report = self.run_launch(w, m, *NEW, "--wait-for-offer-min", "5")
        self.assertEqual((code, w.created_on, report["outcome"]), (1, [1], "KILLED"))
        self.assertIn("no offer could be rented", report["reason"])
        # Whatever vast returns, the launcher itself drops an offer under a floor (threads, RAM, link).
        rows = [offer(1, 11, 1.0, cpu_cores_effective=100), offer(2, 12, 1.0, cpu_cores_effective=224, cpu_ram=16_000),
                offer(3, 13, 1.0, cpu_cores_effective=224, inet_down=50.0), big(4, 14, 1.0, cpu_ram=64_000, inet_down=300.0)]
        a = launch.build_parser().parse_args(["plan"])
        launch.finalize_settings(a)
        saved = launch.vast
        launch.vast = lambda *args, **kw: [dict(r) for r in rows]
        try:
            self.assertEqual([o["id"] for o in launch.search_offers(a)], [4])
        finally:
            launch.vast = saved

    def test_plan_and_dry_run_say_what_they_would_wait_for_and_do_not_wait(self):
        # Only a 96-thread machine at $0.80: below the floor. Plan does not sleep; it prints what a run would wait for.
        w = World([offer(5, 15, 0.8, cpu_cores_effective=96)], [])
        code, out = self.run_cli(w, ["plan"])
        self.assertEqual((code, w.t), (0, 1_000_000.0))
        self.assertIn("no offer matches at the first price step ($1.20/h); a run would wait up to 120 min for one, searching "
                      "every 60 s, 3 attempts per price step ($1.20/h, $1.45/h, $1.70/h); floors: >= 190 threads", out)
        self.assertIn("closest miss: offer 5 costs $0.80/h but has only 96 threads (floor 190)", out)
        self.assertIn("nothing matches even at the price cap right now", out)
        # A machine at $1.50: the first step finds nothing, the cap finds it, and the ladder reaches it at attempt 7... here 4.
        w = World([big(6, 16, 1.40)], [])
        code, out = self.run_cli(w, ["plan"])
        self.assertEqual(code, 0)
        self.assertIn("at the price cap the best offer is offer 6", out)
        self.assertIn("a run would reach it at attempt 4 (about 3 min from the start)", out)
        self.assertEqual(w.t, 1_000_000.0)
        # --wait-for-offer-min 0: plan says a run would stop, and exits 1 as before.
        w = World([offer(5, 15, 0.8, cpu_cores_effective=96)], [])
        code, out = self.run_cli(w, ["plan", "--wait-for-offer-min", "0"])
        self.assertEqual(code, 1)
        self.assertIn("with --wait-for-offer-min 0 a run stops here", out)
        # run --dry-run: the same text, nothing waited for or rented, exit 0.
        w = World([offer(5, 15, 0.8, cpu_cores_effective=96)], ["done:1"])
        code, out = self.run_cli(w, ["run", "--manifest", json.dumps(self.FORMAT1), "--image", "img:1", "--bin-sha", "abc", "--dry-run"])
        self.assertEqual((code, w.created_on, w.t), (0, [], 1_000_000.0))
        self.assertIn("dry run: no offer to rent right now; a real run would wait as described above", out)

    def test_a_named_offer_is_not_waited_for_and_the_price_cap_holds_for_it(self):
        w = World([big(500, 50, 1.90)], ["done:1"])
        w.forbid_search = True
        with self.assertRaises(SystemExit) as e:
            self.run_launch(w, self.FORMAT1, *NEW, "--offer-id", "500")
        self.assertIn("price $1.90/h above the cap $1.70/h (--max-dph raises it)", str(e.exception))
        self.assertEqual(w.created_on, [])
        code, _ = self.run_launch(w, self.FORMAT1, *NEW, "--offer-id", "500", "--max-dph", "2")
        self.assertEqual((code, w.created_on, w.t >= 1_000_000.0), (0, [500], True))
        # Gone: stops at once, no waiting.
        w = World([big(1, 11, 1.0)], ["done:1"])
        with self.assertRaises(SystemExit) as e:
            self.run_launch(w, self.FORMAT1, *NEW, "--offer-id", "500")
        self.assertIn("not on offer", str(e.exception))

    # ---- machines that did not start are remembered for 24 hours (R2 launcher/bad-machines.json)

    BAD = "launcher/bad-machines.json"

    def bad_entry(self, w: World, machine: int, hours_ago: float, host: int = 60400, reason: str = "container failed to start"):
        return {"machine_id": machine, "host_id": host, "offer_id": 1, "instance_id": 5, "run_id": "old", "reason": reason,
                "detail": "x", "utc": launch.epoch_to_utc(w.t - hours_ago * 3600)}

    def test_a_machine_whose_container_did_not_start_is_recorded(self):
        w = World([offer(1, 11, 1.0, host_id=60400), offer(2, 12, 1.5, host_id=60500)], ["broken:0", "done:2"])
        code, report = self.run_launch(w, self.FORMAT1)
        self.assertEqual((code, w.created_on), (0, [1, 2]))
        d = json.loads(w.store[self.BAD])
        self.assertEqual([e["machine_id"] for e in d["machines"]], [11])
        e = d["machines"][0]
        self.assertEqual((e["host_id"], e["offer_id"], e["instance_id"], e["run_id"], e["reason"]),
                         (60400, 1, 100, RUN, "container failed to start"))
        self.assertTrue(w.instances_created_at(100) + 60 <= launch.utc_to_epoch(e["utc"]) <= w.instances_created_at(100) + 300)
        self.assertNotIn("sk-test-secret-3", e["detail"])
        self.assertIn("remembered machine 11 (host 60400) as bad for 24 h: container failed to start", self.log)
        # A machine that never heartbeated is recorded too (its old entry, 30 hours old, is replaced); one that failed later is not.
        w = World([offer(1, 11, 1.0, host_id=60400), offer(2, 12, 1.5, host_id=60500), offer(3, 13, 2.0, host_id=60600)],
                  ["never:0", "die:2", "done:1"])
        w.store[self.BAD] = json.dumps({"format": 1, "machines": [self.bad_entry(w, 99, 2), self.bad_entry(w, 11, 30, host=1)]}).encode()
        code, report = self.run_launch(w, {"format": 2, "run_id": RUN, "run_seed": 1, "streams": []})
        self.assertEqual((code, w.created_on), (0, [1, 2, 3]))
        d = json.loads(w.store[self.BAD])
        by = {e["machine_id"]: e for e in d["machines"]}
        self.assertEqual(sorted(by), [11, 99])                       # 99 kept, 11 replaced (not duplicated), 12 (died later) not added
        self.assertEqual((by[11]["reason"], by[11]["host_id"]), ("never heartbeated", 60400))
        self.assertEqual(by[99]["host_id"], 60400)

    def test_listed_machines_are_left_out_of_every_search(self):
        entry = None
        w = World([offer(1, 11, 0.5), offer(2, 12, 1.0)], ["done:2"])
        entry = self.bad_entry(w, 11, 1)
        w.store[self.BAD] = json.dumps({"format": 1, "machines": [entry]}).encode()
        code, _ = self.run_launch(w, self.FORMAT1)
        self.assertEqual((code, w.created_on), (0, [2]))
        self.assertIn(f"excluding machine 11 (host 60400): container failed to start at {entry['utc']}", self.log)
        self.assertEqual(self.log.count("excluding machine 11"), 1)
        # Plan mode leaves it out too.
        w = World([offer(1, 11, 0.5), offer(2, 12, 1.0)], [])
        w.store[self.BAD] = json.dumps({"format": 1, "machines": [self.bad_entry(w, 11, 1)]}).encode()
        code, out = self.run_cli(w, ["plan", "--min-cores", "16"])
        self.assertEqual(code, 0)
        self.assertIn("excluding machine 11 (host 60400): container failed to start at", out)
        self.assertIn("chosen (", out)
        self.assertNotIn("offer 1:", out.split("excluding machine 11")[1])
        # A replacement search leaves it out as well (listed by another run after this one started renting).
        w = World([offer(1, 11, 1.0), offer(2, 12, 2.0), offer(3, 13, 3.0)], ["die:2", "done:1"])
        orig = w.vast

        def listing_machine_12_after_the_first_rental(*args, **kw):
            if args[:2] == ("create", "instance"):
                w.store[self.BAD] = json.dumps({"format": 1, "machines": [self.bad_entry(w, 12, 0.1, host=7)]}).encode()
            return orig(*args, **kw)

        w.vast = listing_machine_12_after_the_first_rental
        code, report = self.run_launch(w, {"format": 2, "run_id": RUN, "run_seed": 1, "streams": []})
        self.assertEqual((code, w.created_on), (0, [1, 3]))

    def test_an_old_entry_is_ignored_and_an_unreadable_file_is_left_alone(self):
        w = World([offer(1, 11, 0.5), offer(2, 12, 1.0)], ["done:1"])
        w.store[self.BAD] = json.dumps({"format": 1, "machines": [self.bad_entry(w, 11, 25)]}).encode()
        code, _ = self.run_launch(w, self.FORMAT1)
        self.assertEqual((code, w.created_on), (0, [1]))
        self.assertNotIn("excluding machine", self.log)
        # Just inside 24 hours still counts.
        w = World([offer(1, 11, 0.5), offer(2, 12, 1.0)], ["done:1"])
        w.store[self.BAD] = json.dumps({"format": 1, "machines": [self.bad_entry(w, 11, 23.9)]}).encode()
        self.run_launch(w, self.FORMAT1)
        self.assertEqual(w.created_on, [2])
        # A file that is not JSON: warned about, nothing excluded, and a new failure does not overwrite it.
        w = World([offer(1, 11, 1.0), offer(2, 12, 1.5)], ["broken:0", "done:2"])
        w.store[self.BAD] = b"not json"
        code, _ = self.run_launch(w, self.FORMAT1)
        self.assertEqual((code, w.created_on, w.store[self.BAD]), (0, [1, 2], b"not json"))
        self.assertIn("could not be read, so machine 11 is not recorded", self.log)

    def test_a_named_offer_on_a_listed_machine_is_rented_with_a_warning(self):
        w = World([offer(500, 50, 1.0, host_id=60400)], ["done:1"])
        w.forbid_search = True
        w.store[self.BAD] = json.dumps({"format": 1, "machines": [self.bad_entry(w, 50, 1)]}).encode()
        code, _ = self.run_launch(w, self.FORMAT1, "--offer-id", "500")
        self.assertEqual((code, w.created_on), (0, [500]))
        self.assertIn("offer 500 is on machine 50 (host 60400), which did not start a run (container failed to start at", self.log)
        self.assertIn("going ahead, because the owner named it", self.log)

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

    # ---- image pulls and download speed (2026-10-16, run m3-hybrid-r1-c16m-2)

    FORMAT1 = {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}}

    def test_pull_progress_extends_the_wait_and_the_container_start_resets_the_clock(self):
        # The harness gives 5 minutes (--startup-minutes 5). The image is still being pulled at
        # minute 5, 6, ... and vast's status keeps changing: the machine is not destroyed. The
        # container starts at minute 20 and the first heartbeat comes at minute 23, within 5
        # minutes of the start (and long past 5 minutes from renting).
        w = World([offer(1, 11, 1.0), offer(2, 12, 1.5)], ["late:22:3", "done:1"])
        w.status_msg_fn = pull_msgs(start=20)
        code, report = self.run_launch(w, self.FORMAT1)
        self.assertEqual((code, report["outcome"]), (0, "DONE"))
        self.assertEqual((w.created_on, report["replacements"]), ([1], 0))
        self.assertGreaterEqual(report["attempts"][0]["minutes"], 23)
        self.assertIn("still pulling the image (12 min): a1b2c3d4e5f6: Downloading", self.log)
        self.assertIn("past the usual 5 minutes, waiting while the pull moves (gives up after 10 min unchanged, or 40 min in all)", self.log)
        self.assertIn("container started 2 min ago, first heartbeat due within 5 minutes of that", self.log)
        # Without the extension (ceiling no later than the startup minutes) the same machine dies
        # at minute 6, which is what the old rule did.
        w = World([offer(1, 11, 1.0)], ["late:22:3"])
        w.status_msg_fn = pull_msgs(start=20)
        code, report = self.run_launch(w, self.FORMAT1, "--pull-ceiling-min", "5")
        self.assertEqual(code, 1)
        self.assertIn("still pulling", report["attempts"][0]["ended"])
        self.assertLess(report["attempts"][0]["minutes"], 8)

    def test_a_stalled_pull_still_dies(self):
        # The status changes until minute 6 and then freezes: past the 5 minutes, the machine is
        # waited for until its status has not changed for 10 minutes (minute 16), then destroyed.
        w = World([offer(1, 11, 1.0)], ["never:0"])
        w.status_msg_fn = pull_msgs(until=6)
        code, report = self.run_launch(w, self.FORMAT1)
        self.assertEqual(code, 1)
        ended = report["attempts"][0]["ended"]
        self.assertTrue(ended.startswith("no heartbeat within 5 minutes of renting"), ended)
        self.assertIn("image pull stalled (vast's status unchanged for 10 minutes)", ended)
        self.assertTrue(15.5 <= report["attempts"][0]["minutes"] <= 17.5, report["attempts"][0]["minutes"])
        self.assertIn("still pulling the image (10 min)", self.log)
        # --pull-stall-min shortens it.
        w = World([offer(1, 11, 1.0)], ["never:0"])
        w.status_msg_fn = pull_msgs(until=6)
        code, report = self.run_launch(w, self.FORMAT1, "--pull-stall-min", "3")
        self.assertIn("unchanged for 3 minutes", report["attempts"][0]["ended"])
        self.assertTrue(8.5 <= report["attempts"][0]["minutes"] <= 10.5, report["attempts"][0]["minutes"])

    def test_the_pull_ceiling_ends_a_pull_that_keeps_changing(self):
        self.assertEqual((launch.PULL_CEILING_MIN, launch.PULL_STALL_MIN, launch.MIN_INET_DOWN), (40, 10, 100))
        w = World([offer(1, 11, 1.0)], ["never:0"])
        w.status_msg_fn = pull_msgs()
        code, report = self.run_launch(w, self.FORMAT1, "--pull-ceiling-min", "12")
        self.assertEqual(code, 1)
        ended = report["attempts"][0]["ended"]
        self.assertIn("the image was still pulling 12 minutes after renting (the ceiling)", ended)
        self.assertTrue(12.5 <= report["attempts"][0]["minutes"] <= 14.5, report["attempts"][0]["minutes"])

    def test_without_pull_progress_the_old_rule_applies(self):
        # A status that never shows a pull (even if its text changes): destroyed 5 minutes after
        # renting, with the old reason word for word. Nothing is ever destroyed earlier than before.
        w = World([offer(1, 11, 1.0)], ["never:0"])
        w.status_msg_fn = lambda inst, minutes: f"Loading, {int(minutes)} min"
        code, report = self.run_launch(w, self.FORMAT1)
        self.assertEqual(code, 1)
        self.assertEqual(report["attempts"][0]["ended"], "no heartbeat within 5 minutes of renting")
        self.assertTrue(5.5 <= report["attempts"][0]["minutes"] <= 7.5, report["attempts"][0]["minutes"])
        # No status text at all (the plain fake): the same.
        w = World([offer(1, 11, 1.0)], ["never:0"])
        code, report = self.run_launch(w, self.FORMAT1)
        self.assertEqual(report["attempts"][0]["ended"], "no heartbeat within 5 minutes of renting")
        # A pull that froze before the 5 minutes were used up: still destroyed at 5, not later.
        w = World([offer(1, 11, 1.0)], ["never:0"])
        w.status_msg_fn = pull_msgs(until=0)
        code, report = self.run_launch(w, self.FORMAT1, "--startup-minutes", "9", "--pull-stall-min", "2")
        self.assertIn("image pull stalled", report["attempts"][0]["ended"])
        self.assertTrue(9.5 <= report["attempts"][0]["minutes"] <= 11.5, report["attempts"][0]["minutes"])

    def test_container_start_gives_the_first_heartbeat_five_minutes_from_the_start(self):
        # Started at minute 4, first heartbeat at minute 8: after the 5 minutes from renting, within
        # 5 minutes of the start.
        w = World([offer(1, 11, 1.0)], ["late:7:3"])
        w.status_msg_fn = pull_msgs(start=4)
        code, report = self.run_launch(w, self.FORMAT1)
        self.assertEqual((code, report["outcome"], w.created_on), (0, "DONE", [1]))
        # Started at minute 4 and silent: destroyed 5 minutes after the start (minute 9), not before.
        w = World([offer(1, 11, 1.0)], ["never:0"])
        w.status_msg_fn = pull_msgs(start=4)
        code, report = self.run_launch(w, self.FORMAT1)
        self.assertEqual(code, 1)
        self.assertEqual(report["attempts"][0]["ended"], "no heartbeat within 5 minutes of the container starting")
        self.assertTrue(9.5 <= report["attempts"][0]["minutes"] <= 11.5, report["attempts"][0]["minutes"])
        # After a long pull (container started at minute 20), the 5 minutes count from minute 20.
        w = World([offer(1, 11, 1.0)], ["never:0"])
        w.status_msg_fn = pull_msgs(start=20)
        code, report = self.run_launch(w, self.FORMAT1)
        self.assertEqual(report["attempts"][0]["ended"], "no heartbeat within 5 minutes of the container starting")
        self.assertTrue(25.5 <= report["attempts"][0]["minutes"] <= 27.5, report["attempts"][0]["minutes"])

    def test_pull_message_and_start_clock_readings(self):
        for msg in ("a1b2c3d4e5f6: Pulling fs layer", "a1b2c3d4e5f6: Waiting", "a1b2c3d4e5f6: Download complete",
                    "a1b2c3d4e5f6: Pull complete", "a1b2c3d4e5f6: Extracting [==>  ] 1.2GB/3.5GB",
                    "Verifying Checksum", "a1b2c3d4e5f6: Already exists", "Pulling from library/x"):
            self.assertTrue(launch.pull_message(msg), msg)
        for msg in ("", "Loading", "waiting for host", "success, running img:1", "created"):
            self.assertFalse(launch.pull_message(msg), msg)
        self.assertTrue(launch.container_started("success, running ghcr.io/x/y:1"))
        self.assertTrue(launch.container_started("Success, running img"))
        self.assertFalse(launch.container_started("a1b2c3d4e5f6: Pull complete"))
        # A reading with no status_msg (vast unreadable, or a row of the instance list) changes nothing:
        # in particular it does not restart the "unchanged for 10 minutes" clock.
        c = launch.StartClock(0.0, 15, 40, 10)
        c.see(60.0, {"status_msg": "a1b2c3d4e5f6: Downloading 1MB/9MB"})
        c.see(120.0, {"actual_status": "running"})
        c.see(180.0, None)
        self.assertEqual((c.last_change, c.pulling, c.started_at), (60.0, True, None))
        c.see(240.0, {"status_msg": "success, running img:1"})
        self.assertEqual((c.started_at, c.pulling), (240.0, False))
        c.see(900.0, {"status_msg": "something else"})
        self.assertEqual(c.started_at, 240.0)  # once started, always started
        self.assertIsNone(c.verdict(240.0 + 900.0))
        self.assertIn("of the container starting", c.verdict(240.0 + 901.0))

    def test_every_search_asks_for_a_fast_downlink(self):
        w = World([offer(1, 11, 1.0)], ["done:1"])
        self.run_launch(w, self.FORMAT1)
        self.assertIn("inet_down>=100", w.queries[0].split())
        # The fallback search (unverified hosts allowed) and a replacement's search carry it too.
        w = World([offer(1, 11, 1.0, verification="unverified")], ["done:1"])
        self.run_launch(w, self.FORMAT1, "--min-inet-down", "800")
        self.assertEqual(len(w.queries), 2)
        self.assertTrue(all("inet_down>=800" in q.split() for q in w.queries), w.queries)
        # 0 turns the filter off.
        w = World([offer(1, 11, 1.0)], ["done:1"])
        self.run_launch(w, self.FORMAT1, "--min-inet-down", "0")
        self.assertNotIn("inet_down", w.queries[0].replace("inet_down_cost", ""))
        # An offer reporting a link under the floor (100 Mbps) is not returned by the (simulated) search.
        w = World([offer(1, 11, 0.5, inet_down=80.0), offer(2, 12, 2.0, inet_down=900.0)], ["done:1"])
        code, report = self.run_launch(w, self.FORMAT1)
        self.assertEqual((code, w.created_on), (0, [2]))

    def test_a_slow_link_adds_a_penalty_to_the_ranking_unless_cheapest(self):
        lf = launch.link_factor
        self.assertEqual([lf({"inet_down": v}) for v in (500, 200, 199.99 + 0.01, "850")], [1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(lf({"inet_down": 100}), 1.5)
        self.assertAlmostEqual(lf({"inet_down": 0}), 2.0)
        self.assertAlmostEqual(lf({"inet_down": 150.0}), 1.25)
        self.assertEqual(lf({}), 1.0)  # no speed reported: nothing is known, no penalty
        self.assertEqual(lf({"inet_down": None}), 1.0)
        self.assertEqual(lf({"inet_down": "n/a"}), 1.0)
        # The slow offer is 13% cheaper per hour but its link is 100 Mbps (x1.5): the fast one wins.
        offers = [offer(1, 11, 1.00, inet_down=100.0), offer(2, 12, 1.15, inet_down=900.0)]
        w = World(offers, ["done:1"])
        code, report = self.run_launch(w, self.FORMAT1, "--min-inet-down", "0")
        self.assertEqual((code, w.created_on), (0, [2]))
        self.assertIn("link 100 Mbps (slow: cost/work x1.50)", self.log)
        self.assertIn("link 900 Mbps", self.log)
        # --cheapest takes the lowest hourly price, link or not.
        w = World(offers, ["done:1"])
        code, report = self.run_launch(w, self.FORMAT1, "--min-inet-down", "0", "--cheapest")
        self.assertEqual((code, w.created_on), (0, [1]))
        self.assertNotIn("slow: cost/work", self.log)
        # Equal links: the plain cost-per-work ranking, as before.
        w = World([offer(1, 11, 1.00, inet_down=900.0), offer(2, 12, 1.15, inet_down=900.0)], ["done:1"])
        self.run_launch(w, self.FORMAT1, "--min-inet-down", "0")
        self.assertEqual(w.created_on, [1])

    def test_offer_chosen_by_id_with_a_slow_link_is_accepted_and_reported(self):
        w = World([offer(500, 50, 1.0, inet_down=80.0)], ["done:1"])
        w.forbid_search = True
        code, report = self.run_launch(w, self.FORMAT1, "--offer-id", "500")
        self.assertEqual((code, w.created_on, w.searches, w.id_lookups), (0, [500], 0, 1))
        self.assertIn("offer 500 reports a download link of 80 Mbps, below --min-inet-down 100", self.log)
        self.assertIn("accepted, because the owner named it", self.log)
        # A fast link: nothing to report. --min-inet-down 0: nothing to report either.
        w = World([offer(500, 50, 1.0, inet_down=900.0)], ["done:1"])
        self.run_launch(w, self.FORMAT1, "--offer-id", "500")
        self.assertNotIn("below --min-inet-down", self.log)
        w = World([offer(500, 50, 1.0, inet_down=80.0)], ["done:1"])
        self.run_launch(w, self.FORMAT1, "--offer-id", "500", "--min-inet-down", "0")
        self.assertNotIn("below --min-inet-down", self.log)
        # A link speed vast does not report: said in the log, not refused.
        w = World([offer(500, 50, 1.0)], ["done:1"])
        code, _ = self.run_launch(w, self.FORMAT1, "--offer-id", "500")
        self.assertEqual((code, w.created_on), (0, [500]))
        self.assertIn("reports no download link speed", self.log)
        # Every other check still refuses.
        w = World([offer(500, 50, 1.0, inet_down=80.0, reliability2=0.5)], ["done:1"])
        with self.assertRaises(SystemExit):
            self.run_launch(w, self.FORMAT1, "--offer-id", "500")

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
                # Two dead readings, at least DESTROY_RECHECK_S apart (the listing alone when `show instance` errors).
                times = w.listing_times if status_fails else w.status_times
                self.assertEqual(len(times), 2)
                self.assertGreaterEqual(times[1] - times[0], launch.DESTROY_RECHECK_S)
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
        self.assertEqual(len(w.listing_times), 5)  # one unreadable reading per attempt: never a pair
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
        # Attempts 1 and 2: one reading each (still running, so no second one); attempt 3: a pair.
        self.assertEqual(len(w.status_times), 4)

    def test_destroy_normal_path_is_unchanged(self):
        # The call goes through and a later reading shows the instance gone: verified, one call.
        w = self.destroy_world(alive=True)
        result, out = self.run_destroy(w)
        self.assertEqual((result, w.destroy_calls), ("verified", 1))
        self.assertIn("destroyed instance 100 (verified: actual_status exited)", out)
        self.assertNotIn("::error::", out)
        self.assertNotIn("already gone", out)
        self.assertTrue(w.instances[100]["destroyed"])
        self.assertEqual(len(w.status_times), 2)  # the extra reading: two, at least DESTROY_RECHECK_S apart
        self.assertGreaterEqual(w.status_times[1] - w.status_times[0], launch.DESTROY_RECHECK_S)
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

    def test_status_none_is_decided_by_the_instance_list(self):
        # 2026-10-16 (run m3-t1-leader-layouts-9654): after a destroy, `show instance` still answered a
        # record whose actual_status was None, so the machine never read as gone and the job failed. A
        # None record is judged by the account's instance list. (A machine still loading shows None too.)
        # None, and absent from a successfully read list: dead.
        w = self.destroy_world(alive=False, destroyed=True)
        w.status_none = True
        with self.patched(w):
            self.assertEqual(launch.instance_state(100), ("dead", None))
        # None, and listed with None as well (a fresh machine loading): alive, judged by the listed row.
        w = self.destroy_world(alive=True)
        w.status_none = w.listed_status_none = True
        with self.patched(w):
            state, info = launch.instance_state(100)
            self.assertEqual((state, info["id"], info["actual_status"]), ("alive", 100, None))
        # None, and listed as running: alive. Listed as exited: dead.
        w = self.destroy_world(alive=True)
        w.status_none = True
        with self.patched(w):
            self.assertEqual(launch.instance_state(100)[0], "alive")
            w.instances[100]["alive"] = False
            self.assertEqual(launch.instance_state(100)[0], "dead")
        # None, and the list cannot be read (a failed call, or an answer that is not a list): unknown, never dead.
        w = self.destroy_world(alive=False, destroyed=True)
        w.status_none = w.listing_fails = True
        with self.patched(w):
            self.assertEqual(launch.instance_state(100), ("unknown", None))
            w.listing_fails = False
            orig = w.vast
            w.vast = lambda *a, **kw: {"error": "x"} if a[:2] == ("show", "instances") else orig(*a, **kw)
            launch.vast = w.vast
            self.assertEqual(launch.instance_state(100), ("unknown", None))
        # A None record that says intended_status "stopped" is dead without reading the list, as before.
        w = self.destroy_world(alive=True)
        w.status_none = True
        orig = w.vast
        w.vast = lambda *a, **kw: ({"actual_status": None, "intended_status": "stopped"} if a[:2] == ("show", "instance")
                                   else (_ for _ in ()).throw(AssertionError("the list was read")) if a[:2] == ("show", "instances")
                                   else orig(*a, **kw))
        with self.patched(w):
            self.assertEqual(launch.instance_state(100)[0], "dead")

    def test_destroy_is_verified_when_the_status_reads_none_but_the_list_no_longer_holds_it(self):
        # The destroy call goes through; `show instance` keeps answering actual_status None; the list
        # (read successfully) no longer holds the instance: verified, one call, no error.
        w = self.destroy_world(alive=True)
        w.status_none = True
        result, out = self.run_destroy(w)
        self.assertEqual((result, w.destroy_calls), ("verified", 1))
        self.assertIn("destroyed instance 100 (verified: not listed)", out)
        self.assertEqual(len(w.listing_times), 2)
        self.assertGreaterEqual(w.listing_times[1] - w.listing_times[0], launch.DESTROY_RECHECK_S)
        self.assertNotIn("still actual_status None", out)
        self.assertNotIn("::error::", out)
        self.assertTrue(w.instances[100]["destroyed"])
        # The same, but the list cannot be read: never counted as destroyed (unverified, not verified).
        w = self.destroy_world(alive=True)
        w.status_none = w.listing_fails = True
        result, out = self.run_destroy(w)
        self.assertEqual((result, w.destroy_calls), ("unverified", 1))
        self.assertIn("NOT verified", out)

    def test_one_empty_listing_is_not_enough_after_a_failed_destroy_call(self):
        # 2026-10-16, 09:56 UTC: vast's `show instances` once held no 41bots instance while the machine
        # ran on. The destroy call failed (vast hiccup), the first reading says "not listed" (dead), the
        # second, 10 s later, lists the instance running: not verified, and the retry goes on. The next
        # destroy call goes through and two clean dead readings then verify it.
        w = self.destroy_world(alive=True)
        w.destroy_fails = 1
        w.status_fails = True  # `show instance` unreadable: the listing decides
        w.listing_script = ["empty"]  # the first listing misses the instance, every later one is right
        result, out = self.run_destroy(w)
        self.assertEqual((result, w.destroy_calls), ("verified", 2))
        self.assertIn("destroy attempt 1 failed", out)
        self.assertNotIn("already gone", out)
        self.assertIn("reads alive on the second reading", out)
        self.assertIn("destroyed instance 100 (verified: not listed)", out)
        self.assertNotIn("::error::", out)
        self.assertGreaterEqual(w.listing_times[1] - w.listing_times[0], launch.DESTROY_RECHECK_S)
        self.assertEqual(len(w.listing_times), 4)  # attempt 1: empty + alive; attempt 2: dead + dead
        # The instance stayed alive through the first attempt: nothing marked it destroyed before call 2.
        self.assertTrue(w.instances[100]["destroyed"])
        # And a machine that is really alive is never counted as destroyed by a lone empty listing.
        w = self.destroy_world(alive=True)
        w.destroy_fails = 10 ** 6
        w.status_fails = True
        w.listing_script = ["empty"]
        result, out = self.run_destroy(w)
        self.assertEqual((result, w.destroy_calls), ("failed", 5))
        self.assertNotIn("already gone", out)
        self.assertIn("::error::could not destroy instance 100", out)

    def test_one_empty_listing_is_not_enough_after_a_successful_destroy_call(self):
        # The same on the follow-up check of a destroy call that went through: the first reading is an
        # empty list, the second shows the instance running (the call had not taken effect): not
        # verified, "still ..." is logged and the destroy is called again, then verified by a clean pair.
        w = self.destroy_world(alive=True)
        w.status_fails = True
        w.listing_script = ["empty"]
        # Make the first destroy call a no-op on vast's side (it answers success, nothing happens).
        real = w.vast
        calls = []

        def vast(*a, **kw):
            if a[:2] == ("destroy", "instance") and not calls:
                calls.append(a)
                w.destroy_calls += 1
                return ""
            return real(*a, **kw)

        w.vast = vast
        result, out = self.run_destroy(w)
        self.assertEqual((result, w.destroy_calls), ("verified", 2))
        self.assertIn("reads alive on the second reading", out)
        self.assertIn("destroy attempt 1: instance 100 still actual_status running", out)
        self.assertIn("destroyed instance 100 (verified: not listed)", out)
        self.assertGreaterEqual(w.listing_times[1] - w.listing_times[0], launch.DESTROY_RECHECK_S)

    def test_two_dead_readings_ten_seconds_apart_verify_a_destroy(self):
        # After a failed call and after a successful one: dead, 10 s later dead again: verified.
        for fails in (10 ** 6, 0):
            with self.subTest(destroy_fails=fails):
                w = self.destroy_world(alive=False, destroyed=True)
                w.destroy_fails = fails
                w.status_fails = True
                result, out = self.run_destroy(w)
                self.assertEqual((result, w.destroy_calls), ("verified", 1))
                self.assertEqual(len(w.listing_times), 2)
                self.assertGreaterEqual(w.listing_times[1] - w.listing_times[0], launch.DESTROY_RECHECK_S)
                self.assertNotIn("::error::", out)

    def test_dead_then_unreadable_is_not_verified(self):
        # Unknown is never dead: a dead reading followed by an unreadable one decides nothing. After a
        # failed call the destroy is tried again; after a successful one the pair is read afresh.
        w = self.destroy_world(alive=False, destroyed=True)
        w.destroy_fails = 10 ** 6
        w.status_fails = True
        w.listing_script = ["empty", "fail"]  # dead, unknown; then (attempt 2) dead, dead
        result, out = self.run_destroy(w)
        self.assertEqual((result, w.destroy_calls), ("verified", 2))
        self.assertEqual(len(w.listing_times), 4)
        w = self.destroy_world(alive=False, destroyed=True)
        w.status_fails = True
        w.listing_script = ["empty", "fail", "empty", "fail", "empty", "fail"]  # never a clean pair
        result, out = self.run_destroy(w)
        self.assertEqual((result, w.destroy_calls), ("unverified", 1))
        self.assertIn("NOT verified", out)
        self.assertNotIn("(verified", out)

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
                                    "--max-minutes", str(minutes), "--startup-minutes", "5", *LEGACY])
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
