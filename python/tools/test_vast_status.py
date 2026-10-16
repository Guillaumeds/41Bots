"""Tests for the second-machine check of vast_status.py: python3 -m unittest test_vast_status (in python/tools)."""
import os
import re
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vast_watch"))
import model as M  # noqa: E402
import vast_status as V  # noqa: E402


def offer(id, cpu, threads, ram, dph, **kw):
    o = {"id": id, "machine_id": id, "cpu_name": cpu, "threads": threads, "ram_gb": ram, "dph_total": dph, "reliability2": 0.995,
         "verification": "verified", "gpu_frac": 1.0, "num_gpus": 1, "gpu_name": "RTX 4090", "geolocation": "Norway, NO",
         "inet_down_cost": 0.01, "inet_up_cost": 0.01}
    o.update(kw)
    return o


HW = {"cpu_model": "AMD Ryzen Threadripper PRO 9995WX 96-Cores", "cgroup_quota_cores": 184.32, "available_parallelism": 184,
      "mem_total_kb": 263193936}


class Comparable(unittest.TestCase):
    def setUp(self):
        self.m = M.Model()
        self.ref = V.reference_machine(self.m, HW, 1.2685)

    def test_reference_is_the_measured_9995wx(self):
        self.assertEqual(self.ref["source"], "measured")
        self.assertAlmostEqual(self.ref["cost"], 0.285, delta=0.015)
        self.assertAlmostEqual(self.ref["nps"] / 1e9, 1.24, delta=0.06)

    def test_no_when_only_dearer_machines(self):
        market = [offer(1, "AMD EPYC 9654P 96-Core Processor", 192, 251, 2.94), offer(2, "AMD EPYC 9754 128-Core Processor", 256, 503, 3.75)]
        ok, closest = V.comparable(self.m, market, self.ref, 1.3, 0.7)
        self.assertEqual(ok, [])
        self.assertEqual(len(closest), 2)

    def test_yes_for_a_second_machine_of_the_same_class(self):
        market = [offer(1, "AMD EPYC 9654P 96-Core Processor", 192, 251, 2.94),
                  offer(7, "AMD Ryzen Threadripper PRO 9995WX 96-Cores", 192, 251, 1.30)]
        ok, closest = V.comparable(self.m, market, self.ref, 1.3, 0.7)
        self.assertEqual([o["id"] for o in ok], [7])
        self.assertEqual(closest, [])

    def test_unverified_shared_and_slow_machines_never_count(self):
        cpu = "AMD Ryzen Threadripper PRO 9995WX 96-Cores"
        market = [offer(1, cpu, 192, 251, 1.0, verification="deverified"), offer(2, cpu, 192, 251, 1.0, gpu_frac=0.5),
                  offer(3, cpu, 192, 251, 1.0, reliability2=0.85), offer(4, cpu, 64, 80, 0.4)]   # 64 threads: too slow
        ok, _ = V.comparable(self.m, market, self.ref, 1.3, 0.7)
        self.assertEqual(ok, [])

    def test_measured_only_filter(self):
        market = [offer(5, "AMD EPYC 7J13 64-Core Processor", 128, 252, 0.3)]   # an absurdly cheap unbenchmarked machine
        self.assertEqual([o["id"] for o in V.comparable(self.m, market, self.ref, 5.0, 0.3)[0]], [5])
        self.assertEqual(V.comparable(self.m, market, self.ref, 5.0, 0.3, measured_only=True)[0], [])


class Endpoint(unittest.TestCase):
    def test_r2_reads_the_workflow_endpoint_name_first(self):
        base = {"R2_BUCKET": "b", "R2_ACCESS_KEY_ID": "k", "R2_SECRET_ACCESS_KEY": "s"}
        old = {k: os.environ.pop(k, None) for k in ("R2_S3_ENDPOINT", "R2_ENDPOINT", *base)}
        try:
            os.environ.update(base, R2_S3_ENDPOINT="https://s3.example.invalid/")
            self.assertEqual(V.R2().ep, "https://s3.example.invalid")
            del os.environ["R2_S3_ENDPOINT"]
            os.environ["R2_ENDPOINT"] = "https://old.example.invalid"
            self.assertEqual(V.R2().ep, "https://old.example.invalid")
            del os.environ["R2_ENDPOINT"]
            with self.assertRaises(SystemExit) as e:
                V.R2()
            self.assertIn("R2_S3_ENDPOINT", str(e.exception))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


WORKFLOW = os.path.join(os.path.dirname(os.path.dirname(HERE)), ".github", "workflows", "vast-status.yml")


def workflow_step_script(step_name: str) -> str:
    """The `run: |` script of one step of vast-status.yml, dedented, read from the workflow file itself
    so the test runs exactly what GitHub runs (no YAML library needed: the step is plain text)."""
    with open(WORKFLOW, encoding="utf-8") as f:
        lines = f.read().split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.strip() == f"- name: {step_name}")
    run = next(i for i in range(start, len(lines)) if re.match(r"\s+run: \|\s*$", lines[i]))
    indent = len(lines[run]) - len(lines[run].lstrip())
    body = []
    for ln in lines[run + 1:]:
        if ln.strip() and len(ln) - len(ln.lstrip()) <= indent:
            break
        body.append(ln)
    return textwrap.dedent("\n".join(body))


FAKE_AWS = r"""#!/bin/bash
# A stand-in for the aws CLI. FAKE_MODE: beats (two heartbeats), empty (what `aws s3 ls` really does for a folder
# with nothing in it: prints nothing, exits 1), error (a real failure: exit 255 and a message on stderr).
args="$*"
case "$args" in
  *"s3 ls s3://bkt/runs/r1/heartbeat/")
    case "$FAKE_MODE" in
      beats) echo "2026-10-16 10:00:00        100 000001.json"; echo "2026-10-16 10:01:00        100 000002.json";;
      empty) exit 1;;
      error) echo "An error occurred (AccessDenied) when calling the ListObjectsV2 operation" >&2; exit 255;;
    esac;;
  *"s3 ls s3://bkt/runs/r1/")
    case "$FAKE_MODE" in
      beats) echo "                           PRE heartbeat/"; echo "2026-10-16 09:59:00        300 manifest.json";;
      empty) exit 1;;
      error) exit 255;;
    esac;;
  *"s3 cp s3://bkt/runs/r1/heartbeat/"*) k="${args##*/heartbeat/}"; k="${k%% *}"; echo "{\"file\": \"$k\"}";;
  *) echo "unexpected aws call: $args" >&2; exit 99;;
esac
"""


class HeartbeatStep(unittest.TestCase):
    """The workflow's "Run heartbeats" step: a run that was only just rented (still pulling its image) has no
    heartbeat folder yet, and that must not fail the job."""

    def run_step(self, mode: str, run_id: str = "r1"):
        with tempfile.TemporaryDirectory() as d:
            aws = os.path.join(d, "aws")
            with open(aws, "w") as f:
                f.write(FAKE_AWS)
            os.chmod(aws, os.stat(aws).st_mode | stat.S_IXUSR)
            env = dict(os.environ, PATH=d + os.pathsep + os.environ["PATH"], FAKE_MODE=mode, RUN_ID=run_id,
                       ENDPOINT="https://example.invalid", BUCKET="bkt")
            return subprocess.run(["bash", "-c", workflow_step_script("Run heartbeats")], env=env, capture_output=True, text=True)

    def test_an_empty_heartbeat_folder_is_not_a_failure(self):
        p = self.run_step("empty")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("no heartbeats yet for r1", p.stdout)

    def test_heartbeats_are_printed_when_there_are_some(self):
        p = self.run_step("beats")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn('{"file": "000001.json"}', p.stdout)
        self.assertIn('{"file": "000002.json"}', p.stdout)
        self.assertNotIn("no heartbeats yet", p.stdout)

    def test_a_real_listing_error_still_fails_the_job(self):
        p = self.run_step("error")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("could not list the heartbeats of r1", p.stdout + p.stderr)
        self.assertNotIn("no heartbeats yet", p.stdout)

    def test_a_bad_run_id_is_refused(self):
        p = self.run_step("beats", run_id="r1; rm -rf /")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("bad run_id", p.stdout)


if __name__ == "__main__":
    unittest.main()
