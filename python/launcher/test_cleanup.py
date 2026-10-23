"""Tests of the run-aware cleanup against an in-memory R2 and instance list (no network).

    python3 -m unittest discover -s python/launcher -p "test_*.py"
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cleanup  # noqa: E402

NOW = 2_000_000_000.0
MIN = 60.0


def inst(start_min_ago: float, status: str = "running", dph: float = 1.0, label: str = "41bots-r1", iid: int = 9,
         cpu=0.0):
    """An instance row; `cpu` is vast's cpu_util (idle by default; None = no reading)."""
    return {"id": iid, "label": label, "actual_status": status, "start_date": NOW - start_min_ago * MIN, "dph_total": dph,
            "cpu_util": cpu}


def state(beat_min_ago=None, markers=None, stop_min_ago=None, watch=None):
    return {"markers": {k: NOW - v * MIN for k, v in (markers or {}).items()},
            "stop": None if stop_min_ago is None else NOW - stop_min_ago * MIN,
            "heartbeats": [] if beat_min_ago is None else [NOW - beat_min_ago * MIN],
            "watch": watch or {}}


class DecideTest(unittest.TestCase):
    def d(self, i, s, everything=False):
        return cleanup.decide(i, s, NOW, everything)[0]

    def test_long_healthy_run_is_kept(self):
        # 12 hours old, heartbeating, within its cap: the old 2-hour rule killed exactly this.
        self.assertEqual(self.d(inst(720), state(1, watch={"max_dollars": 20, "dph": 1.0})), "keep")

    def test_finished_run_gets_ten_minutes_to_destroy_itself(self):
        self.assertEqual(self.d(inst(300), state(1, markers={"DONE": 5})), "keep")
        self.assertEqual(self.d(inst(300), state(1, markers={"DONE": 11})), "destroy")
        self.assertEqual(self.d(inst(300), state(1, markers={"FAILED": 11})), "destroy")

    def test_silence(self):
        self.assertEqual(self.d(inst(300), state(19)), "keep")
        self.assertEqual(self.d(inst(300), state(21)), "destroy")

    def test_silent_but_busy_is_asked_to_stop_first(self):
        # Silent for 21 min but vast shows it running at 95% CPU (or gives no CPU reading): an
        # upload problem. Stop first; destroy only if still silent 15 min after the stop request.
        for cpu in (95.0, 50.0, None):
            self.assertEqual(self.d(inst(300, cpu=cpu), state(21)), "stop", cpu)
            self.assertEqual(self.d(inst(300, cpu=cpu), state(21, stop_min_ago=10)), "keep", cpu)
            self.assertEqual(self.d(inst(300, cpu=cpu), state(36, stop_min_ago=16)), "destroy", cpu)
            # The same for a machine that never heartbeated.
            self.assertEqual(self.d(inst(61, cpu=cpu), state()), "stop", cpu)
            self.assertEqual(self.d(inst(62, cpu=cpu), state(stop_min_ago=16)), "destroy", cpu)
        # Idle, or not running: destroyed as before.
        self.assertEqual(self.d(inst(300, cpu=49.9), state(21)), "destroy")
        self.assertEqual(self.d(inst(300, "loading", cpu=99.0), state(21)), "destroy")
        self.assertIn("busy (CPU 95%)", cleanup.decide(inst(300, cpu=95.0), state(21), NOW)[1])
        self.assertIn("no CPU reading", cleanup.decide(inst(300, cpu=None), state(21), NOW)[1])

    def test_heartbeats_of_an_earlier_machine_do_not_count(self):
        # Replacement machine started 30 min ago; the only heartbeat is from before it.
        self.assertEqual(self.d(inst(30), state(40)), "keep")
        self.assertEqual(self.d(inst(50), state(60)), "keep")  # still within its 60 minutes to a first heartbeat
        self.assertEqual(self.d(inst(62), state(70)), "destroy")

    def test_never_heartbeated(self):
        # 60 minutes (2026-10-16): a slow image pull may take 40 minutes, then 15 to the first heartbeat.
        self.assertEqual(self.d(inst(40), state()), "keep")
        self.assertEqual(self.d(inst(46), state()), "keep")
        self.assertEqual(self.d(inst(59), state()), "keep")
        self.assertEqual(self.d(inst(61), state()), "destroy")
        self.assertIn("no heartbeat at all in its 61 min of life", cleanup.decide(inst(61), state(), NOW)[1])

    def test_never_heartbeated_limit_covers_the_launchers_slowest_start(self):
        # The launcher waits for an advancing image pull up to PULL_CEILING_MIN minutes from renting and
        # then --startup-minutes (15) from the container start; the safety net must not cut that short.
        import launch
        self.assertEqual(cleanup.NO_HEARTBEAT_S, 60 * 60)
        self.assertGreaterEqual(cleanup.NO_HEARTBEAT_S, (launch.PULL_CEILING_MIN + 15) * 60)

    def test_exited_or_offline(self):
        self.assertEqual(self.d(inst(5, "exited"), state(1)), "destroy")
        # Offline in vast's eyes but still heartbeating to R2: working, kept (review E).
        self.assertEqual(self.d(inst(300, "offline"), state(1)), "keep")
        self.assertEqual(self.d(inst(300, "offline"), state(21)), "destroy")
        self.assertEqual(self.d(inst(50, "offline"), state()), "destroy")
        # ... and its dollar cap still applies.
        w = {"max_dollars": 2, "dph": 1.0}
        self.assertEqual(self.d(inst(180, "offline"), state(1, watch=w)), "stop")

    def test_busy_threshold_of_a_low_cpu_run(self):
        # Review E: a run declared to use few cores (watch.json busy_cpu_pct 1, from
        # --low-cpu-pct 1) reads 17% in vast; silent, it is asked to stop and upload first.
        w = {"busy_cpu_pct": 1.0}
        self.assertEqual(self.d(inst(300, cpu=17.0), state(21, watch=w)), "stop")
        self.assertEqual(self.d(inst(300, cpu=17.0), state(36, stop_min_ago=16, watch=w)), "destroy")
        self.assertEqual(self.d(inst(300, cpu=0.0), state(21, watch=w)), "destroy")
        self.assertEqual(self.d(inst(300, cpu=17.0), state(21)), "destroy", "default threshold 50%")

    def test_all(self):
        self.assertEqual(self.d(inst(5), state(1), everything=True), "destroy")

    def test_budget_stops_first_then_destroys(self):
        w = {"max_dollars": 20, "spent_before": 3.0, "dph": 1.0}
        self.assertEqual(self.d(inst(16 * 60), state(1, watch=w)), "keep")       # $19
        self.assertEqual(self.d(inst(17 * 60 + 1), state(1, watch=w)), "stop")    # $20.02
        self.assertEqual(self.d(inst(17 * 60 + 5), state(1, stop_min_ago=10, watch=w)), "keep")
        self.assertEqual(self.d(inst(17 * 60 + 20), state(1, stop_min_ago=16, watch=w)), "destroy")

    def test_deadline(self):
        w = {"deadline_utc": cleanup.utc(NOW - 20 * MIN)}
        self.assertEqual(self.d(inst(600), state(1, watch=w)), "keep")
        w = {"deadline_utc": cleanup.utc(NOW - 31 * MIN)}
        self.assertEqual(self.d(inst(600), state(1, watch=w)), "destroy")

    def test_parse_time_formats(self):
        self.assertEqual(cleanup.parse_time("2033-05-18T03:33:20+00:00"), NOW)
        self.assertEqual(cleanup.parse_time("Wed, 18 May 2033 03:33:20 GMT"), NOW)
        self.assertEqual(cleanup.parse_time(NOW), NOW)


class FakeR2:
    def __init__(self):
        self.store: dict[str, tuple[bytes, float]] = {}
        self.deleted: list[str] = []

    def put(self, key, data, t=NOW):
        self.store[key] = (data, t)

    def get(self, key):
        return self.store[key][0] if key in self.store else None

    def mtime(self, key):
        return self.store[key][1] if key in self.store else None

    def list(self, prefix):
        return sorted((k, t) for k, (_, t) in self.store.items() if k.startswith(prefix))

    def delete(self, key):
        self.store.pop(key, None)
        self.deleted.append(key)


class SweepTest(unittest.TestCase):
    def setUp(self):
        self.destroyed: list = []
        self.vast_calls: list = []
        self.fresh: dict | None = {}  # what the second reading of an instance adds (None: vast unreadable)
        self.saved = cleanup.destroy, cleanup.vast, cleanup.instance_now
        cleanup.destroy = lambda i: self.destroyed.append(i) or True
        cleanup.vast = lambda *a, raw=True: self.vast_calls.append(a)
        cleanup.instance_now = lambda i: self.fresh

    def tearDown(self):
        cleanup.destroy, cleanup.vast, cleanup.instance_now = self.saved

    def sweep(self, r2, instances, dry_run=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return cleanup.sweep(r2, instances, NOW, False, dry_run)

    def test_other_instances_are_never_touched(self):
        r2 = FakeR2()
        rows = self.sweep(r2, [inst(9999, "exited", label="someone-else", iid=1), inst(9999, "exited", label="", iid=2)])
        self.assertEqual(rows, [])
        self.assertEqual(self.destroyed, [])

    def test_budget_writes_stop(self):
        r2 = FakeR2()
        r2.put("runs/r1/heartbeat/000001.json", b"{}", NOW - MIN)
        r2.put("runs/r1/watch.json", json.dumps({"max_dollars": 2, "dph": 1.0}).encode())
        self.sweep(r2, [inst(180)])
        self.assertIn("cleanup", json.loads(r2.get("runs/r1/STOP"))["reason"])
        self.assertEqual(self.destroyed, [])

    def test_destroy_needs_a_second_reading(self):
        # Silent 30 min: destroyed only when a fresh reading agrees; when vast cannot be read
        # again, nothing happens this round; when the fresh reading shows it busy, STOP is
        # written instead.
        r2 = FakeR2()
        r2.put("runs/r1/heartbeat/000001.json", b"{}", NOW - 30 * MIN)
        rows = self.sweep(r2, [inst(180)])
        self.assertEqual((self.destroyed, rows[0][3]), ([9], "destroy"))
        self.destroyed.clear()
        self.fresh = None
        rows = self.sweep(r2, [inst(180)])
        self.assertEqual((self.destroyed, rows[0][3]), ([], "keep"))
        self.assertIn("could not be read again", rows[0][4])
        self.fresh = {"cpu_util": 97.0}
        rows = self.sweep(r2, [inst(180)])
        self.assertEqual((self.destroyed, rows[0][3]), ([], "stop"))
        self.assertIn("cleanup", json.loads(r2.get("runs/r1/STOP"))["reason"])
        self.assertIn("on a second reading", rows[0][4])

    def test_dry_run_changes_nothing(self):
        r2 = FakeR2()
        r2.put("runs/r1/DONE", b"{}", NOW - 60 * MIN)
        self.sweep(r2, [inst(180)], dry_run=True)
        self.assertEqual(self.destroyed, [])

    def test_handed_over_run_that_destroyed_itself_is_closed(self):
        r2 = FakeR2()
        w = {"instance_id": 9, "dph": 1.0, "created_utc": cleanup.utc(NOW - 600 * MIN), "destroy_key_id": 77}
        r2.put("watch/r1.json", json.dumps(w).encode())
        r2.put("runs/r1/DONE", b"{}", NOW - 5 * MIN)
        self.sweep(r2, [])
        closed = json.loads(r2.get("runs/r1/closed.json"))
        self.assertEqual(closed["outcome"], "DONE")
        self.assertTrue(closed["self_destroyed"])
        self.assertEqual(closed["dollars_est_upper"], 10.0)
        self.assertIn("watch/r1.json", r2.deleted)
        self.assertEqual(self.vast_calls, [("delete", "api-key", "77")])

    def test_handed_over_run_destroyed_by_cleanup_says_so(self):
        r2 = FakeR2()
        r2.put("watch/r1.json", json.dumps({"instance_id": 9, "dph": 1.0}).encode())
        r2.put("runs/r1/heartbeat/000001.json", b"{}", NOW - MIN)
        r2.put("runs/r1/DONE", b"{}", NOW - 30 * MIN)
        self.sweep(r2, [inst(600)])
        self.assertEqual(self.destroyed, [9])
        closed = json.loads(r2.get("runs/r1/closed.json"))
        self.assertFalse(closed["self_destroyed"])
        self.assertIn("DONE", closed["why"])

    def test_closing_finalises_the_machine_record(self):
        # The runner wrote a running record and heartbeats, then its machine was destroyed: the record
        # becomes final with the heartbeat's minutes and dollars, and the cost per node recomputed.
        r2 = FakeR2()
        r2.put("watch/r1.json", json.dumps({"instance_id": 9, "dph": 2.0, "offer_id": 55,
                                            "created_utc": cleanup.utc(NOW - 600 * MIN)}).encode())
        r2.put("runs/r1/FAILED", b"{}", NOW - 30 * MIN)
        r2.put("runs/r1/heartbeat/000001.json", json.dumps({"elapsed_s": 60, "est_cost_usd": 0.03}).encode())
        r2.put("runs/r1/heartbeat/000120.json", json.dumps(
            {"elapsed_s": 7200, "est_cost_usd": 4.0, "progress": {"nodes_total": 5e12}}).encode())
        r2.put("machines/r1.json", json.dumps({"run_id": "r1", "state": "running", "nodes_total": 4e12,
                                               "worlds_finished": 40, "cpu_model": "X", "minutes": 100,
                                               "usd": 3.33, "usd_per_1e12_nodes": 0.8325}).encode())
        self.sweep(r2, [])
        m = json.loads(r2.get("machines/r1.json"))
        self.assertEqual((m["state"], m["outcome"], m["minutes"], m["usd"]), ("final", "FAILED", 120.0, 4.0))
        self.assertEqual(m["nodes_total"], 4e12)  # the runner's own count, not the heartbeat's
        self.assertEqual(m["usd_per_1e12_nodes"], 1.0)
        self.assertEqual((m["cpu_model"], m["offer_id"], m["finalised_by"]), ("X", 55, "cleanup"))
        # A record the runner already made final is left alone; a run without any record gets one
        # from the launcher's watch data (an upper bound, said so).
        r2.put("watch/r2.json", json.dumps({"instance_id": 10, "dph": 1.0, "created_utc": cleanup.utc(NOW - 120 * MIN)}).encode())
        r2.put("machines/r2.json", json.dumps({"state": "final", "usd": 1.0, "minutes": 60}).encode())
        r2.put("watch/r3.json", json.dumps({"instance_id": 11, "dph": 1.0, "created_utc": cleanup.utc(NOW - 120 * MIN)}).encode())
        r2.put("runs/r3/DONE", b"{}", NOW - 30 * MIN)
        self.sweep(r2, [])
        self.assertEqual(json.loads(r2.get("machines/r2.json")), {"state": "final", "usd": 1.0, "minutes": 60})
        m3 = json.loads(r2.get("machines/r3.json"))
        self.assertEqual((m3["state"], m3["outcome"], m3["minutes"], m3["usd"], m3["nodes_total"]), ("final", "DONE", 120.0, 2.0, 0))
        self.assertIsNone(m3["usd_per_1e12_nodes"])
        self.assertIn("upper bound", m3["finalised_basis"])

    def test_handed_over_run_missing_from_the_listing_but_running_is_not_closed(self):
        # Review E: an incomplete instance listing must not close a working handed-over run
        # (closing deletes its destroy key and marks its machine record final).
        r2 = FakeR2()
        r2.put("watch/r1.json", json.dumps({"instance_id": 9, "dph": 1.0, "destroy_key_id": 77}).encode())
        self.fresh = {"actual_status": "running"}
        self.sweep(r2, [])
        self.assertIsNone(r2.get("runs/r1/closed.json"))
        self.assertIsNotNone(r2.get("watch/r1.json"))
        self.assertEqual(self.vast_calls, [])
        self.fresh = {"actual_status": "exited"}
        self.sweep(r2, [])
        self.assertIsNotNone(r2.get("runs/r1/closed.json"))

    def test_handed_over_run_still_working_stays_watched(self):
        r2 = FakeR2()
        r2.put("watch/r1.json", json.dumps({"instance_id": 9, "dph": 1.0}).encode())
        r2.put("runs/r1/heartbeat/000001.json", b"{}", NOW - MIN)
        self.sweep(r2, [inst(600)])
        self.assertEqual(self.destroyed, [])
        self.assertIsNone(r2.get("runs/r1/closed.json"))
        self.assertIsNotNone(r2.get("watch/r1.json"))

class KeptTest(SweepTest):
    """A machine with a keep record (keep/<instance>.json) is left for the next worker until its keep time."""

    def record(self, until_min_from_now: float, run="r1", iid=9) -> bytes:
        until = cleanup.utc(NOW + until_min_from_now * MIN)
        return json.dumps({"instance_id": iid, "run_id": run, "keep_until_utc": until}).encode()

    def test_decide_kept(self):
        rec = json.loads(self.record(60))
        self.assertEqual(cleanup.decide_kept(inst(300), rec, NOW)[0], "keep")
        old = json.loads(self.record(-10))  # 10 minutes past: beyond the 5-minute grace
        self.assertEqual(cleanup.decide_kept(inst(300), old, NOW)[0], "destroy")
        soon = json.loads(self.record(-2))  # inside the grace
        self.assertEqual(cleanup.decide_kept(inst(300), soon, NOW)[0], "keep")
        self.assertEqual(cleanup.decide_kept(inst(300, "exited"), rec, NOW)[0], "destroy")
        self.assertEqual(cleanup.decide_kept(inst(300), {"run_id": "r1"}, NOW)[0], "destroy")
        self.assertEqual(cleanup.decide_kept(inst(300), rec, NOW, everything=True)[0], "destroy")

    def test_a_kept_machine_of_a_finished_run_is_not_destroyed(self):
        # Without the record the run's DONE marker, 30 minutes old, would have it destroyed.
        r2 = FakeR2()
        r2.put("runs/r1/DONE", b"{}", NOW - 30 * MIN)
        r2.put("keep/9.json", self.record(120))
        rows = self.sweep(r2, [inst(600)])
        self.assertEqual((self.destroyed, rows[0][3]), ([], "keep"))
        self.assertIn("kept for the next worker", rows[0][4])
        r2b = FakeR2()
        r2b.put("runs/r1/DONE", b"{}", NOW - 30 * MIN)
        rows = self.sweep(r2b, [inst(600)])
        self.assertEqual((self.destroyed, rows[0][3]), ([9], "destroy"))

    def test_a_kept_machine_past_its_time_is_destroyed_after_a_second_reading(self):
        r2 = FakeR2()
        r2.put("keep/9.json", self.record(-30))
        rows = self.sweep(r2, [inst(600)])
        self.assertEqual((self.destroyed, rows[0][3]), ([9], "destroy"))
        self.destroyed.clear()
        self.fresh = None  # vast unreadable on the second reading: nothing is destroyed
        rows = self.sweep(r2, [inst(600)])
        self.assertEqual((self.destroyed, rows[0][3]), ([], "keep"))

    def test_a_record_of_a_vanished_machine_is_deleted(self):
        r2 = FakeR2()
        r2.put("keep/9.json", self.record(120))
        self.fresh = {"actual_status": "exited"}
        self.sweep(r2, [])
        self.assertIn("keep/9.json", r2.deleted)
        r3 = FakeR2()
        r3.put("keep/9.json", self.record(120))
        self.fresh = {"actual_status": "running"}  # missing from the listing but alive: kept
        self.sweep(r3, [])
        self.assertEqual(r3.deleted, [])

    def test_a_kept_machine_of_someone_elses_label_is_never_touched(self):
        r2 = FakeR2()
        r2.put("keep/9.json", self.record(-300))
        rows = self.sweep(r2, [inst(600, label="mine-not-41bots")])
        self.assertEqual((rows, self.destroyed), ([], []))


if __name__ == "__main__":
    unittest.main()
