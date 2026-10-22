"""Kept machines and takeover (owner, 2026-10-22), against the launcher's fake vast and R2 (test_launch.World).

A manifest's machine.keep leaves the machine running when the run ends (the runner parks, KEEP_MACHINE=1) and records
it in keep/<instance>.json; machine.instance_id takes an instance over instead of renting one. Nothing here touches
the network.

    python3 -m unittest discover -s python/launcher -p "test_*.py"
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch  # noqa: E402
import test_launch as tl  # noqa: E402

RUN = tl.RUN
BODY = {"run_id": RUN, "chunk": {"difftest": {"matches": 10, "seed": 1}}}


def stamp(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class KWorld(tl.World):
    """test_launch.World with a machine that parks after DONE (keep), and instances that already exist."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.keep = False  # the runner parks (stays alive) after its DONE
        self.preset: dict[int, dict] = {}  # `show instance` of instances that exist before the run
        self.calls: list[tuple] = []  # update / recycle / label calls
        self.takeover_script = "done:2"
        self.start_fails = False

    def add_machine(self, iid: int, label: str, script: str = "idle:0", **info) -> None:
        self.instances[iid] = {"alive": True, "polls": 0, "script": script, "offer": 0, "created": self.t, "label": label}
        self.preset[iid] = {"actual_status": "running", "intended_status": "running", "label": label,
                            "disk_space": 60.0, "cpu_cores_effective": 224.0, "dph_total": 0.7,
                            "start_date": self.t - 7200, **info}

    def vast(self, *args, raw=True, check=True, tail=(), timeout=None):
        cmd = args[:2]
        if cmd == ("show", "instance") and int(args[2]) in self.preset:
            inst = self.instances[int(args[2])]
            if self.preset[int(args[2])].get("actual_status") == "exited":
                return dict(self.preset[int(args[2])])  # stopped by its owner: the record keeps its details
            if not inst["alive"]:
                return {"actual_status": "exited", "intended_status": "stopped"}
            return dict(self.preset[int(args[2])])
        if cmd == ("update", "instance"):
            self.calls.append(("update", int(args[2]), args[args.index("--env") + 1], args[args.index("--image") + 1]))
            return {"success": True}
        if cmd == ("start", "instance"):
            iid = int(args[2])
            self.calls.append(("start", iid))
            if self.start_fails:
                raise RuntimeError("vastai start instance failed: no resources: the machine's GPU is taken")
            self.instances[iid].update(alive=True, polls=0, script=self.takeover_script, created=self.t)
            self.preset[iid]["actual_status"] = "running"
            return ""
        if cmd == ("recycle", "instance"):
            iid = int(args[2])
            self.calls.append(("recycle", iid))
            self.instances[iid].update(alive=True, polls=0, script=self.takeover_script, created=self.t)
            return ""
        if cmd == ("label", "instance"):
            iid = int(args[2])
            self.calls.append(("label", iid, args[3]))
            self.preset.setdefault(iid, {})["label"] = args[3]
            self.instances[iid]["label"] = args[3]
            return ""
        return super().vast(*args, raw=raw, check=check, tail=tail, timeout=timeout)

    def step(self, inst: dict) -> None:
        super().step(inst)
        if self.keep and f"runs/{RUN}/DONE" in self.store and not inst.get("destroyed"):
            inst["alive"] = True  # parked: it does not destroy itself

    def heartbeat_of(self, run: str, t: float) -> None:
        self.store[f"runs/{run}/heartbeat/000001.json"] = json.dumps({"seq": 1, "utc": stamp(t)}).encode()


class KeepTestBase(unittest.TestCase):
    setUp = tl.LauncherTest.setUp
    tearDown = tl.LauncherTest.tearDown
    run_launch = tl.LauncherTest.run_launch
    patched = tl.LauncherTest.patched

    def manifest(self, **machine) -> dict:
        return {**BODY, "machine": machine} if machine else dict(BODY)


class SettingsTest(unittest.TestCase):
    def apply(self, machine: dict, **given) -> argparse.Namespace:
        a = argparse.Namespace(gpus=0, cpu_only=True, **given)
        launch.apply_machine(a, {"run_id": RUN, "machine": machine})
        return a

    def test_the_new_machine_keys_are_set(self):
        a = self.apply({"keep": True, "keep_hours": 3, "instance_id": 57253570})
        self.assertEqual((a.keep, a.keep_hours, a.instance_id), (True, 3.0, 57253570))
        with contextlib.redirect_stdout(io.StringIO()):
            launch.finalize_settings(a)
        self.assertEqual((a.keep, a.keep_hours), (True, 3.0))

    def test_defaults_keep_nothing_and_a_command_line_value_wins(self):
        a = self.apply({"min_cores": 1})
        with contextlib.redirect_stdout(io.StringIO()):
            launch.finalize_settings(a)
        self.assertEqual((a.keep, a.keep_hours, getattr(a, "instance_id", None)), (False, launch.KEEP_HOURS, None))
        b = self.apply({"instance_id": 5, "keep_hours": 2}, instance_id=9, keep_hours=1.5)
        self.assertEqual((b.instance_id, b.keep_hours), (9, 1.5))

    def test_bad_values_are_refused(self):
        for machine in ({"keep": "yes"}, {"keep": 1}, {"instance_id": 0}, {"instance_id": -3}, {"instance_id": "57"},
                        {"instance_id": True}, {"instance_id": 1.5}, {"keep_hours": 0}, {"keep_hours": -1},
                        {"keep_hours": launch.KEEP_HOURS_MAX + 1}, {"keep_hours": "6"}, {"keep_hourz": 6}):
            with self.assertRaises(SystemExit, msg=str(machine)):
                self.apply(machine)

    def test_the_runner_is_told_to_park_only_when_the_run_keeps_its_machine(self):
        with tl.mock_env():
            kept = launch.instance_env("r", 8, 0.7, "abc", "img:1", "key1", keep=True)
            plain = launch.instance_env("r", 8, 0.7, "abc", "img:1", "key1")
        self.assertIn("-e KEEP_MACHINE=1", kept)
        self.assertNotIn("KEEP_MACHINE", plain)

    def test_keep_applies_only_to_a_run_that_ended_by_itself(self):
        a = argparse.Namespace(keep=True, stopped=False)
        self.assertTrue(launch.keep_applies(a, "DONE", "DONE"))
        self.assertTrue(launch.keep_applies(a, "FAILED", "FAILED"))
        self.assertFalse(launch.keep_applies(a, "", "heartbeat missing"))
        self.assertFalse(launch.keep_applies(a, "DONE", "budget reached: $5.01 of $5.00 spent"))
        a.stopped = True  # the launcher stopped it (low CPU, silence, time): destroyed as before
        self.assertFalse(launch.keep_applies(a, "FAILED", "CPU use below 85%"))
        self.assertFalse(launch.keep_applies(argparse.Namespace(keep=False, stopped=False), "DONE", "DONE"))

    def test_the_record_and_the_notice_say_what_to_do(self):
        rec = launch.keep_record("r1", 57253570, 0.723, "run DONE", 6.0, "EPYC 7B12", now=1_800_000_000.0)
        self.assertEqual(rec["kept_utc"], stamp(1_800_000_000.0))
        self.assertEqual(rec["keep_until_utc"], stamp(1_800_000_000.0 + 6 * 3600))
        note = launch.keep_notice(rec)
        for word in ("MACHINE KEPT RUNNING, ACTION NEEDED", "57253570", "$0.723/h", "instance_id", rec["keep_until_utc"]):
            self.assertIn(word, note)


class KeepRunTest(KeepTestBase):
    def test_a_finished_run_leaves_its_machine_up_and_says_so(self):
        w = KWorld([tl.offer(1, 11, 1.0)], ["done:2"])
        w.keep = True
        code, report = self.run_launch(w, self.manifest(keep=True, keep_hours=3))
        self.assertEqual(code, 0)
        self.assertEqual((report["outcome"], report["destroy"]), ("DONE", "kept"))
        self.assertEqual(w.destroy_calls, 0)
        self.assertTrue(w.instances[100]["alive"] and not w.instances[100].get("destroyed"))
        self.assertIn("-e KEEP_MACHINE=1", w.envs[0])
        rec = json.loads(w.store["keep/100.json"])
        self.assertEqual((rec["run_id"], rec["instance_id"], rec["hours"]), (RUN, 100, 3.0))
        self.assertEqual(json.loads(w.store[f"runs/{RUN}/KEPT.json"]), rec)
        self.assertIn("MACHINE KEPT RUNNING, ACTION NEEDED", self.log)
        self.assertIn("::warning::MACHINE KEPT", self.log)

    def test_without_keep_nothing_is_kept_and_the_runner_is_not_told_to_park(self):
        w = KWorld([tl.offer(1, 11, 1.0)], ["done:2"])
        code, report = self.run_launch(w, self.manifest())
        self.assertEqual(code, 0)
        self.assertNotIn("KEEP_MACHINE", w.envs[0])
        self.assertNotIn("keep/100.json", w.store)
        self.assertNotIn(f"runs/{RUN}/KEPT.json", w.store)

    def test_the_budget_still_destroys_a_machine_that_was_to_be_kept(self):
        w = KWorld([tl.offer(1, 11, 1.0)], ["work:0"])
        w.keep = True
        code, report = self.run_launch(w, self.manifest(keep=True), "--max-dollars", "0.2")
        self.assertTrue(w.instances[100].get("destroyed"))
        self.assertNotIn("keep/100.json", w.store)
        self.assertIn("budget", report["reason"])

    def test_the_command_line_flag_keeps_too(self):
        w = KWorld([tl.offer(1, 11, 1.0)], ["done:2"])
        w.keep = True
        code, report = self.run_launch(w, self.manifest(), "--keep", "--keep-hours", "2")
        self.assertEqual(report["destroy"], "kept")
        self.assertEqual(json.loads(w.store["keep/100.json"])["hours"], 2.0)

    def test_the_state_files_are_removed_so_the_workflow_safety_net_cannot_destroy_it(self):
        w = KWorld([tl.offer(1, 11, 1.0)], ["done:2"])
        w.keep = True
        with tempfile.TemporaryDirectory() as d:
            state = os.path.join(d, "instance-id.txt")
            code, report = self.run_launch(w, self.manifest(keep=True), "--state-file", state)
            self.assertEqual(report["destroy"], "kept")
            self.assertEqual(sorted(os.listdir(d)), [])  # the id, the run id and the destroy key files are all gone


class TakeoverTest(KeepTestBase):
    def take(self, w: KWorld, **extra) -> tuple[int, dict]:
        w.forbid_search = True
        return self.run_launch(w, self.manifest(instance_id=500, **extra))

    def test_an_unlabelled_machine_rented_by_hand_is_taken_over_not_rented(self):
        w = KWorld([tl.offer(1, 11, 1.0)], [])
        w.add_machine(500, "")
        code, report = self.take(w)
        self.assertEqual(code, 0)
        self.assertEqual(w.created_on, [])
        self.assertEqual(w.searches, 0)
        kinds = [c[0] for c in w.calls]
        self.assertEqual(kinds, ["update", "recycle", "label"])
        env = json.loads(w.calls[0][2])
        self.assertEqual((env["RUN_ID"], env["BIN_SHA"]), (RUN, "abc"))
        self.assertNotIn("KEEP_MACHINE", env)
        self.assertEqual(w.calls[0][3], "img:1")
        self.assertEqual(w.calls[2][2], f"41bots-{RUN}")
        self.assertEqual(report["outcome"], "DONE")
        self.assertEqual(report["instance_id"], 500)

    def test_a_kept_machine_is_taken_over_and_its_record_goes(self):
        w = KWorld([], [])
        w.add_machine(500, "41bots-old-run")
        w.store["keep/500.json"] = json.dumps({"instance_id": 500, "run_id": "old-run"}).encode()
        code, report = self.take(w)
        self.assertEqual(code, 0)
        self.assertNotIn("keep/500.json", w.store)
        self.assertEqual([c[0] for c in w.calls], ["update", "recycle", "label"])

    def test_the_machine_of_an_ended_run_may_be_taken(self):
        w = KWorld([], [])
        w.add_machine(500, "41bots-old-run")
        w.store["runs/old-run/DONE"] = b"{}"
        code, _ = self.take(w)
        self.assertEqual(code, 0)

    def test_another_runs_live_machine_is_refused_and_nothing_is_changed(self):
        w = KWorld([], [])
        w.add_machine(500, "41bots-other-run")
        w.heartbeat_of("other-run", w.t - 60)
        with self.assertRaises(SystemExit) as e:
            self.take(w)
        self.assertIn("still working on it", str(e.exception))
        self.assertEqual(w.calls, [])
        self.assertTrue(w.instances[500]["alive"])

    def test_a_silent_old_run_no_longer_protects_its_machine(self):
        w = KWorld([], [])
        w.add_machine(500, "41bots-other-run")
        w.heartbeat_of("other-run", w.t - 3600)
        code, _ = self.take(w)
        self.assertEqual(code, 0)

    def test_a_machine_that_has_just_been_started_by_another_run_is_refused(self):
        w = KWorld([], [])
        w.add_machine(500, "41bots-other-run", start_date=w.t - 300)
        with self.assertRaises(SystemExit) as e:
            self.take(w)
        self.assertIn("probably booting", str(e.exception))

    def test_a_label_that_is_not_ours_is_refused(self):
        w = KWorld([], [])
        w.add_machine(500, "my-other-service")
        with self.assertRaises(SystemExit) as e:
            self.take(w)
        self.assertIn("belongs to something else", str(e.exception))
        self.assertEqual(w.calls, [])

    def test_too_small_a_disk_or_too_few_threads_or_a_dead_machine_is_refused(self):
        for info in ({"disk_space": 10.0}, {"cpu_cores_effective": 2.0}):
            w = KWorld([], [])
            w.add_machine(500, "", **info)
            with self.assertRaises(SystemExit, msg=str(info)) as e:
                self.run_launch(w, self.manifest(instance_id=500), "--min-cores", "16", "--disk-gb", "40")
            self.assertIn("cannot take this run", str(e.exception))
            self.assertEqual(w.calls, [], info)
        w = KWorld([], [])
        w.add_machine(500, "")
        w.instances[500]["alive"] = False
        with self.assertRaises(SystemExit):
            self.take(w)

    def test_a_stopped_machine_is_updated_then_started_not_recycled(self):
        w = KWorld([], [])
        w.add_machine(500, "", actual_status="exited", intended_status="stopped")
        w.instances[500]["alive"] = False
        code, report = self.take(w)
        self.assertEqual(code, 0)
        self.assertEqual([c[0] for c in w.calls], ["update", "start", "label"])
        self.assertEqual(report["instance_id"], 500)
        self.assertIn("started", self.log)

    def test_a_stopped_machine_that_cannot_start_ends_the_run_without_renting_another(self):
        w = KWorld([tl.offer(1, 11, 1.0)], ["done:2"])
        w.add_machine(500, "", actual_status="exited", intended_status="stopped")
        w.instances[500]["alive"] = False
        w.start_fails = True
        code, report = self.take(w)
        self.assertEqual(w.created_on, [])
        self.assertNotEqual(code, 0)
        self.assertIn("free slot", self.log)

    def test_an_offline_machine_is_refused(self):
        w = KWorld([], [])
        w.add_machine(500, "", actual_status="offline")
        with self.assertRaises(SystemExit):
            self.take(w)
        self.assertEqual(w.calls, [])

    def test_a_machine_that_is_taken_over_can_be_kept_again(self):
        w = KWorld([], [])
        w.keep = True
        w.add_machine(500, "")
        code, report = self.take(w, keep=True)
        self.assertEqual(report["destroy"], "kept")
        self.assertEqual(json.loads(w.calls[0][2])["KEEP_MACHINE"], "1")
        self.assertEqual(json.loads(w.store["keep/500.json"])["run_id"], RUN)
        self.assertEqual(w.destroy_calls, 0)


class StopKeepTest(KeepTestBase):
    """`launch.py stop --keep`: the workflow's cancel step for a run with machine.keep."""

    def stop(self, w: KWorld, *extra: str) -> int:
        with self.patched(w):
            return launch.main(["stop", RUN, "--instance", "500", "--wait-s", "0", *extra])

    def test_a_cancelled_run_with_a_marker_keeps_its_machine(self):
        w = KWorld([], [])
        w.add_machine(500, f"41bots-{RUN}")
        w.store[f"runs/{RUN}/DONE"] = b"{}"
        self.assertEqual(self.stop(w, "--keep", "--keep-hours", "4"), 0)
        self.assertEqual(w.destroy_calls, 0)
        self.assertEqual(json.loads(w.store["keep/500.json"])["hours"], 4.0)
        self.assertIn(f"runs/{RUN}/KEPT.json", w.store)

    def test_without_the_flag_or_without_a_marker_the_machine_is_destroyed(self):
        w = KWorld([], [])
        w.add_machine(500, f"41bots-{RUN}")
        w.store[f"runs/{RUN}/DONE"] = b"{}"
        self.stop(w)
        self.assertEqual(w.destroy_calls, 1)
        self.assertNotIn("keep/500.json", w.store)
        w2 = KWorld([], [])
        w2.add_machine(500, f"41bots-{RUN}")  # no marker within 0 s: the runner did not stop
        self.stop(w2, "--keep")
        self.assertEqual(w2.destroy_calls, 1)
        self.assertNotIn("keep/500.json", w2.store)


if __name__ == "__main__":
    unittest.main()
