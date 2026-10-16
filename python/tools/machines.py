#!/usr/bin/env python3
"""The machine database: what every machine we rent really costs per 10^12 solver nodes.

    python3 python/tools/machines.py                  # every record, one row per run
    python3 python/tools/machines.py --live           # one line per machine running now
    python3 python/tools/machines.py --by-cpu         # per CPU model and schedule: runs, mean and best cost, speed
    python3 python/tools/machines.py --rank           # today's vast.ai offers ranked by MEASURED cost where we have it
    python3 python/tools/machines.py --record-speeds  # measured thread speeds into docs/vast/speed-factors.json
    python3 python/tools/machines.py --backfill [--runs a,b] [--write]   # records for past runs from their R2 files
    options: --from DIR (local record files instead of R2), --markdown, --schedule plan|fill|prod|split|bench

Records are `machines/<run_id>.json` in R2, written by the runner every 10 minutes and at the end of
a run (crates/runner/src/machine.rs), or by the cleanup script when it closes a handed-over run, or
by `--backfill` for runs made before the records existed. Each holds the machine (CPU model,
threads, RAM, $/hour), the run so far (minutes, dollars, worlds finished of total, solver nodes) and
the two figures that matter for renting: dollars per 10^12 solver nodes [positions the exact solver
examined; all our work is counted in them] and nodes per second per thread. `schedule` says how the
worlds were spread: `plan` (the position-aware schedule of the real runs), `fill` (one thread per
world, the bench's full-machine window), `split` (one world over several threads), `prod` (the
bench's 2-minute production window) or `bench` (a whole t1bench run).

Caveats. Early figures read high: the work queue runs the hardest deals first, so a running record
is marked "settling" until 10% of its worlds are done. Backfilled records of runs made before
2026-10-16 have no node counts: their nodes are ESTIMATED from the worlds' thread-seconds and the
measured thread speed of the CPU model (marked `~`), and they are left out of the measured means.

Reads R2 with the launcher's variables (R2_S3_ENDPOINT or R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY_ID,
R2_SECRET_ACCESS_KEY); `--rank` searches vast.ai offers (no key needed). Standard library only.
Nothing here rents or destroys anything.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import statistics
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vast_watch"))
import vast_market as vm  # noqa: E402  (speed factors, the `record` command)
import vast_status as vs  # noqa: E402  (R2 reads, the market search)
import model as M  # noqa: E402  (the cost model of a listing)

MU = dt.timezone(dt.timedelta(hours=4))  # Mauritius
STALE_S = 20 * 60
SETTLED_SHARE = 0.10
SCHEDULES = ("plan", "prod", "fill", "split", "bench")
DEFAULT_BACKFILL = ["m3-agree-t1-b2", "m3-agree-t1-a7", "m3-agree-t1-b3", "m3-agree-t1-b",
                    "m3-agree-t2d", "m3-agree-t3", "m3-agree-t4c"]


# ------------------------------------------------------------------ R2 (the read-only class, plus a PUT)

class R2(vs.R2):
    def put(self, key: str, data: bytes) -> int:
        """Writes one object (SigV4, like the GETs of vast_status.R2). Returns the HTTP status."""
        now = dt.datetime.now(dt.timezone.utc)
        amz, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
        cp = urllib.parse.quote(f"/{self.bucket}/{key}", safe="/")
        ph = hashlib.sha256(data).hexdigest()
        hd = f"host:{self.host}\nx-amz-content-sha256:{ph}\nx-amz-date:{amz}\n"
        sh = "host;x-amz-content-sha256;x-amz-date"
        creq = f"PUT\n{cp}\n\n{hd}\n{sh}\n{ph}"
        scope = f"{day}/auto/s3/aws4_request"
        sts = f"AWS4-HMAC-SHA256\n{amz}\n{scope}\n{hashlib.sha256(creq.encode()).hexdigest()}"
        k = self._h(self._h(self._h(self._h(("AWS4" + self.sk).encode(), day), "auto"), "s3"), "aws4_request")
        sig = hmac.new(k, sts.encode(), hashlib.sha256).hexdigest()
        auth = f"AWS4-HMAC-SHA256 Credential={self.ak}/{scope}, SignedHeaders={sh}, Signature={sig}"
        req = urllib.request.Request(f"{self.ep}{cp}", data=data, method="PUT",
                                     headers={"x-amz-date": amz, "x-amz-content-sha256": ph, "Authorization": auth,
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    def text(self, key: str) -> str | None:
        st, b = self.get(f"/{self.bucket}/{key}")
        return b.decode("utf-8", "replace") if st == 200 else None


def load_records(r2=None, from_dir: str | None = None) -> list[dict]:
    """Every machine record (the mutable ones; the history copies are left out), by run id."""
    recs = []
    if from_dir and os.path.isfile(from_dir):
        # One JSON file holding {"records": [...]} (docs/vast/measured-machines.json).
        with open(from_dir, encoding="utf-8") as f:
            recs = list(json.load(f).get("records") or [])
    elif from_dir:
        for name in sorted(os.listdir(from_dir)):
            p = os.path.join(from_dir, name)
            if name.endswith(".json") and os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    try:
                        recs.append(json.load(f))
                    except ValueError:
                        pass
    else:
        for key, _ in r2.ls("machines/"):
            if key.endswith(".json") and not key.startswith("machines/history/"):
                rec = r2.json(key)
                if isinstance(rec, dict):
                    recs.append(rec)
    recs = [r for r in recs if r.get("run_id")]
    recs.sort(key=lambda r: (str(r.get("utc") or ""), r["run_id"]))
    return recs


# ------------------------------------------------------------------ record arithmetic

def cpu_key(name) -> str:
    """One short name per CPU model, the same from /proc/cpuinfo and from vast.ai's listing."""
    s = (name or "").lower()
    s = re.sub(r"\(r\)|\(tm\)|®|™", " ", s)
    s = re.sub(r"@.*$", " ", s)
    s = re.sub(r"\b\d+-cores?\b|\bprocessor\b|\bcpu\b|\bamd\b|\bintel\b|\bryzen\b|\bgenuine\b", " ", s)
    return " ".join(s.split())


def short_cpu(name) -> str:
    return cpu_key(name).upper() or "?"


def nodes_of(rec: dict) -> float:
    return float(rec.get("nodes_total") or 0)


def cost_of(rec: dict) -> float | None:
    c = rec.get("usd_per_1e12_nodes")
    if c is None and nodes_of(rec) > 0 and rec.get("usd") is not None:
        c = float(rec["usd"]) / nodes_of(rec) * 1e12
    return float(c) if c is not None else None


def measured(rec: dict) -> bool:
    return nodes_of(rec) > 0 and (rec.get("nodes_basis") or "measured").startswith("measured")


def settling(rec: dict) -> bool:
    if rec.get("settling") is not None:
        return bool(rec["settling"])
    total = rec.get("worlds_total")
    return bool(total) and float(rec.get("worlds_finished") or 0) < SETTLED_SHARE * float(total)


def parse_utc(s) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def age_s(rec: dict, now: dt.datetime) -> float | None:
    t = parse_utc(rec.get("utc"))
    return (now - t).total_seconds() if t else None


def notes(rec: dict, now: dt.datetime) -> str:
    out = []
    if rec.get("state") == "running":
        a = age_s(rec, now)
        if a is not None and a > STALE_S:
            out.append(f"stale: last heard {a / 60:.0f} min ago")
        if settling(rec):
            out.append("settling")
    elif settling(rec) and nodes_of(rec) > 0:
        out.append("under 10% of its worlds: figures read high")
    if nodes_of(rec) > 0 and not measured(rec):
        out.append("nodes estimated")
    return "; ".join(out)


def fmt(x, nd=3, dash="-") -> str:
    return dash if x is None else f"{x:.{nd}f}"


def sci(x) -> str:
    return "-" if not x else f"{float(x):.2e}"


# ------------------------------------------------------------------ views

TABLE_HEAD = ["run", "state", "outcome", "CPU", "threads", "RAM GB", "$/h", "min", "$", "worlds", "nodes",
              "$ per 1e12", "M nodes/s/thread", "schedule", "trick", "notes"]


def table_rows(recs: list[dict], now: dt.datetime) -> list[list]:
    rows = []
    for r in recs:
        worlds = f"{r.get('worlds_finished', '-')}/{r.get('worlds_total') if r.get('worlds_total') else '-'}"
        c = cost_of(r)
        tilde = "" if measured(r) or c is None else "~"
        per = r.get("nodes_per_s_per_thread")
        rows.append([r["run_id"], r.get("state", "?"), r.get("outcome") or "-", short_cpu(r.get("cpu_model")),
                     r.get("threads", "-"), fmt(r.get("ram_gb"), 0), fmt(r.get("dph"), 3), fmt(r.get("minutes"), 0),
                     fmt(r.get("usd"), 2), worlds, tilde + sci(nodes_of(r)), tilde + fmt(c), fmt(per / 1e6 if per else None, 2),
                     r.get("schedule") or "-", r.get("trick") if r.get("trick") is not None else "-", notes(r, now)])
    return rows


def live_lines(recs: list[dict], now: dt.datetime) -> list[str]:
    """One line per machine running now (its record says `running`; a record not refreshed for 20
    minutes is shown as stale, since a run that died without its final record stays `running`)."""
    out = []
    for r in recs:
        if r.get("state") != "running":
            continue
        a = age_s(r, now)
        c = cost_of(r)
        cost = f"${c:.3f} per 1e12 nodes" if c is not None else "no nodes yet"
        if settling(r):
            cost += " (settling: under 10% of the worlds done, reads high)"
        stale = f"   STALE, last heard {a / 60:.0f} min ago" if a is not None and a > STALE_S else ""
        out.append(f"{r['run_id']}: {short_cpu(r.get('cpu_model'))} · {r.get('threads')} threads · {fmt(r.get('ram_gb'), 0)} GB · "
                   f"${float(r.get('dph') or 0):.3f}/h · {float(r.get('minutes') or 0):.0f} min · ${float(r.get('usd') or 0):.2f} so far · "
                   f"worlds {r.get('worlds_finished', 0)}/{r.get('worlds_total') or '?'} · {cost}{stale}")
    return out or ["no machine is running (no record in state `running`)"]


def samples(recs: list[dict]) -> list[dict]:
    """Per-schedule samples for the CPU summary: a run's record once, except a whole-bench record,
    which gives one sample per window (fill, prod) from the figures it kept."""
    out = []
    for r in recs:
        if nodes_of(r) <= 0:
            continue
        base = {"run_id": r["run_id"], "cpu": cpu_key(r.get("cpu_model")), "cpu_model": r.get("cpu_model"),
                "measured": measured(r) and not settling(r), "settling": settling(r)}
        wins = r.get("windows") or {}
        if r.get("schedule") == "bench" and wins:
            for w in ("fill", "prod"):
                x = wins.get(w) or {}
                if x.get("usd_per_1e12_nodes") is not None:
                    out.append(dict(base, schedule=w, cost=float(x["usd_per_1e12_nodes"]),
                                    per_thread=float(x.get("nodes_per_s_per_thread") or 0) or None))
            continue
        out.append(dict(base, schedule=r.get("schedule") or "?", cost=cost_of(r), per_thread=r.get("nodes_per_s_per_thread")))
    return [s for s in out if s["cost"] is not None]


def by_cpu(recs: list[dict]) -> list[dict]:
    """Per CPU model and schedule: runs, the mean and best measured $ per 10^12 nodes and the mean
    nodes per second per thread (measured, settled records only; the others are counted)."""
    groups: dict[tuple, list[dict]] = {}
    for s in samples(recs):
        groups.setdefault((s["cpu"], s["schedule"]), []).append(s)
    rows = []
    for (cpu, sched), ss in groups.items():
        good = [s for s in ss if s["measured"]]
        costs = [s["cost"] for s in good]
        pers = [s["per_thread"] for s in good if s["per_thread"]]
        rows.append({"cpu": cpu, "cpu_model": ss[0]["cpu_model"], "schedule": sched, "runs": len(ss), "measured_runs": len(good),
                     "cost_mean": statistics.mean(costs) if costs else None, "cost_best": min(costs) if costs else None,
                     "per_thread_mean": statistics.mean(pers) if pers else None,
                     "estimated_costs": [s["cost"] for s in ss if not s["measured"]],
                     "run_ids": [s["run_id"] for s in ss]})
    rows.sort(key=lambda r: (r["cost_mean"] if r["cost_mean"] is not None else 1e9, r["cpu"], r["schedule"]))
    return rows


BY_CPU_HEAD = ["CPU", "schedule", "runs", "measured", "mean $ per 1e12", "best $ per 1e12", "mean M nodes/s/thread",
               "estimated-only $ per 1e12", "runs"]


def by_cpu_rows(rows: list[dict]) -> list[list]:
    return [[r["cpu"].upper(), r["schedule"], r["runs"], r["measured_runs"], fmt(r["cost_mean"]), fmt(r["cost_best"]),
             fmt(r["per_thread_mean"] / 1e6 if r["per_thread_mean"] else None, 2),
             ", ".join(f"~{c:.3f}" for c in r["estimated_costs"]) or "-", " ".join(r["run_ids"])] for r in rows]


def measured_speeds(recs: list[dict], prefer: tuple = SCHEDULES) -> dict[str, dict]:
    """Per CPU key: the measured nodes per second per thread (mean of the settled, measured
    records under the first schedule of `prefer` that has any), with the runs and schedule."""
    out = {}
    rows = by_cpu(recs)
    for sched in prefer:
        for r in rows:
            if r["schedule"] == sched and r["per_thread_mean"] and r["cpu"] not in out:
                out[r["cpu"]] = {"per_thread": r["per_thread_mean"], "n": r["measured_runs"], "schedule": sched,
                                 "cpu_model": r["cpu_model"],
                                 "runs": sorted(s["run_id"] for s in samples(recs) if s["cpu"] == r["cpu"] and s["schedule"] == sched and s["measured"])}
    return out


def rank_offers(offers: list[dict], recs: list[dict], model: M.Model, min_threads: float = 64, prefer: tuple = SCHEDULES) -> list[dict]:
    """Offers our launcher can rent, cheapest per 10^12 nodes first: by the measured thread speed of
    the CPU model where we have a record (applied to the offer's price and threads), else by the
    cost model's estimate. Each row says which."""
    speeds = measured_speeds(recs, prefer)
    out = []
    for o in offers:
        a = M.annotate(model, o)
        if not a["launcher_ok"] or (o.get("threads") or 0) < min_threads:
            continue
        m = speeds.get(cpu_key(o.get("cpu_name")))
        if m and a["usable"] > 0:
            nps = a["usable"] * m["per_thread"]
            a["nps"] = nps
            a["cost"] = float(o.get("dph_total") or 0) / (nps * 3600.0) * 1e12
            a["source"] = f"measured ({m['n']} run(s), {m['schedule']})"
            a["measured"] = True
        else:
            a["source"] = "estimate" if a.get("source") == "estimated" else f"estimate ({a.get('source')})"
            a["measured"] = False
        if a["cost"] is None:
            continue
        out.append(a)
    out.sort(key=lambda a: (a["cost"], -a["nps"]))
    return out


RANK_HEAD = ["offer", "CPU", "threads", "RAM GB", "$/h", "$ per 1e12", "G nodes/s", "basis", "country"]


def rank_rows(ranked: list[dict]) -> list[list]:
    return [[a["id"], short_cpu(a.get("cpu_name")), round(a.get("usable") or a.get("threads") or 0), fmt(a.get("ram_gb"), 0),
             fmt(a.get("dph_total"), 3), ("" if a["measured"] else "~") + fmt(a["cost"]), fmt(a["nps"] / 1e9, 2),
             a["source"] + ("; " + a["launcher_flag"] + " host" if a.get("launcher_flag") else ""),
             (a.get("geolocation") or "?").split(",")[-1].strip()] for a in ranked]


def record_speeds(recs: list[dict], speed_file: str, measured_file: str | None, prefer: tuple = SCHEDULES, quiet: bool = False) -> list[str]:
    """Writes each measured CPU model's thread speed into the speed-factor file through
    `vast_market.py record` (the tt factor = nodes/s per thread divided by the k of
    docs/vast/measured-speeds.json, the EPYC-7502-relative unit), so the market tools use the
    measured value. Returns the models written."""
    k = 4.5e6
    if measured_file and os.path.isfile(measured_file):
        with open(measured_file, encoding="utf-8") as f:
            k = float(json.load(f).get("default_k_nodes_per_s_per_speed_unit") or k)
    done = []
    for cpu, m in measured_speeds(recs, prefer).items():
        argv = ["record", "--cpu", m["cpu_model"] or cpu, "--tt", f"{m['per_thread'] / k:.3f}", "--run", ",".join(m["runs"]),
                "--note", f"machines.py: {m['per_thread'] / 1e6:.2f} M nodes/s per thread over {m['n']} run(s), schedule {m['schedule']}",
                "--file", speed_file]
        if quiet:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                vm.main(argv)
        else:
            vm.main(argv)
        done.append(cpu)
    return done


# ------------------------------------------------------------------ backfill

def plan_threads(log_text: str | None) -> dict[int, int]:
    """Threads per world by position from `search worlds`' plan line in worlds.log."""
    out: dict[int, int] = {}
    for line in (log_text or "").splitlines():
        if "position-aware plan" in line and "{" in line:
            try:
                plan = json.loads(line[line.index("{"):])
            except ValueError:
                continue
            for c in plan.get("classes", []):
                out.setdefault(int(c.get("position", 0)), int(c.get("threads") or 1))
    return out


def jsonl(text: str | None) -> list[dict]:
    rows = []
    for line in (text or "").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass  # a torn last line
    return rows


def measure_work(r2, run_id: str, prefix: str) -> dict:
    """The worlds a search_measure run finished and their thread-seconds, from its worlds files,
    checkpoints and timing files under `prefix` (`out/measure` or `partial/out/measure`)."""
    # The exact worlds only: rollout worlds search no nodes, and the totals count exact worlds.
    keys = [k for k, _ in r2.ls(f"runs/{run_id}/{prefix}/exact-") if k.endswith(".jsonl") and not k.endswith(".times.jsonl")]
    done: dict[tuple, float] = {}  # (file stem, id, w) -> thread-seconds
    worlds_files = []
    for key in keys:
        name = key.rsplit("/", 1)[-1]
        rows = jsonl(r2.text(key))
        times = {t["id"]: t for t in jsonl(r2.text(key + ".times.jsonl"))}
        in_file = set()
        for rec in rows:
            n = len(rec.get("rows") or [])
            t = times.get(rec["id"]) or {}
            per = (float(t.get("world_us") or 0) / 1e6 * float(t.get("threads") or 1)) / n if n else 0.0
            for i in range(n):
                done[(name, rec["id"], int(rec.get("from") or 0) + i)] = per
                in_file.add(rec["id"])
        worlds_files.append(name)
    decisions = {d["id"]: d for d in jsonl(r2.text(f"runs/{run_id}/{prefix}/" + next(
        (k.rsplit("/", 1)[-1] for k, _ in r2.ls(f"runs/{run_id}/{prefix}/decisions-") if k.endswith(".jsonl")), "none")) or "")}
    threads_by_pos = plan_threads(r2.text(f"runs/{run_id}/{prefix}/worlds.log"))
    for key, _ in r2.ls(f"runs/{run_id}/{prefix}/exact-"):
        if not key.endswith(".ckpt"):
            continue
        name = key.rsplit("/", 1)[-1][:-len(".ckpt")]
        for row in jsonl(r2.text(key)):
            k = (name, row["id"], row["w"])
            if k in done:
                continue
            pos = int((decisions.get(row["id"]) or {}).get("position", 0))
            done[k] = float(row.get("us") or 0) / 1e6 * threads_by_pos.get(pos, 1)
    return {"done": done, "files": worlds_files, "trick": next((int(d.get("trick")) for d in decisions.values()), None)}


def build_record(r2, run_id: str, model: M.Model, depth: int = 0) -> dict | None:
    """A machine record for a past run from its R2 files. None when the run has no manifest."""
    man = r2.json(f"runs/{run_id}/manifest.json")
    if not isinstance(man, dict):
        return None
    hw = r2.json(f"runs/{run_id}/hardware.json") or {}
    rep = r2.json(f"runs/{run_id}/report.json") or {}
    res = r2.json(f"runs/{run_id}/result.json") or {}
    hbs = sorted(k for k, _ in r2.ls(f"runs/{run_id}/heartbeat/") if k.endswith(".json"))
    hb = (r2.json(hbs[-1]) if hbs else None) or {}
    files = {k.split("/", 2)[2] for k, _ in r2.ls(f"runs/{run_id}/")}
    chunk = next(iter((man.get("chunk") or {}).keys()), None)
    params = (man.get("chunk") or {}).get(chunk) or {}
    threads = hw.get("threads_default") or hb.get("threads") or rep.get("cores_effective")
    try:
        dph = float(hw.get("dph_env") or rep.get("dph") or 0)
    except ValueError:
        dph = float(rep.get("dph") or 0)
    if rep.get("elapsed_min") and rep.get("dollars_est"):
        minutes, usd, basis = float(rep["elapsed_min"]), float(rep["dollars_est"]), "the launcher's report (the whole rental)"
    elif hb.get("elapsed_s"):
        minutes, usd = float(hb["elapsed_s"]) / 60, float(hb.get("est_cost_usd") or 0)
        basis = "the run's last heartbeat (the runner's own clock)"
    else:
        minutes, usd, basis = 0.0, 0.0, "unknown"
    outcome = rep.get("outcome") or ("DONE" if "DONE" in files else "FAILED" if "FAILED" in files else "KILLED")
    rec = {"format": "41bots-machine-record", "version": 1, "run_id": run_id, "state": "final", "outcome": outcome,
           "utc": rep.get("utc") or hb.get("utc"), "cpu_model": hw.get("cpu_model") or rep.get("cpu_name"),
           "threads": threads, "usable_cores": hw.get("usable_cores"),
           "ram_gb": round(float(hw["mem_total_kb"]) / 1048576, 1) if hw.get("mem_total_kb") else None,
           "dph": dph, "minutes": round(minutes, 1), "usd": round(usd, 4), "minutes_basis": basis,
           "chunk": chunk, "bin_sha": hw.get("bin_sha") or rep.get("bin_sha"), "image": rep.get("image"),
           "offer_id": rep.get("offer_id"), "instance_id": rep.get("instance_id") or hw.get("vast_container_id"),
           "backfilled": True}
    if chunk == "t1bench":
        results = r2.json(f"runs/{run_id}/out/t1bench/results.json") or {}
        nodes, worlds, windows = 0.0, 0, {}
        for w in ("fill", "prod"):
            x = results.get(w) or {}
            if x.get("nodes"):
                nodes += float(x["nodes"])
                worlds += int(x.get("worlds_finished") or 0)
                windows[w] = {k: x.get(k) for k in ("nodes", "nodes_per_s", "nodes_per_s_per_thread", "worlds_finished", "threads",
                                                    "tt_mb", "cpu_pct", "wall_s")}
                windows[w]["usd_per_1e12_nodes"] = x.get("dollars_per_tera_nodes")
        for s in ((results.get("sweep") or {}).get("runs") or []):
            if s.get("nodes_per_s") and s.get("window_s"):
                nodes += float(s["nodes_per_s"]) * float(s["window_s"])
                worlds += int(s.get("worlds_finished") or 0)
        score = results.get("score") or {}
        rec.update({"nodes_total": round(nodes), "nodes_basis": "measured" if nodes else "none", "worlds_finished": worlds,
                    "worlds_total": None, "schedule": "bench", "trick": 1, "settling": False,
                    "usd_per_1e12_nodes": round(usd / nodes * 1e12, 3) if nodes and usd else None,
                    "nodes_per_s_per_thread": score.get("nodes_per_s_per_thread"), "windows": windows,
                    "tt_mb": (results.get("fill") or {}).get("tt_mb")})
        return rec
    if chunk != "search_measure":
        rec.update({"nodes_total": 0, "nodes_basis": "none", "worlds_finished": 0, "worlds_total": None,
                    "usd_per_1e12_nodes": None, "schedule": None, "settling": False})
        return rec
    prefix = "out/measure" if any(f.startswith("out/measure/") for f in files) else "partial/out/measure"
    work = measure_work(r2, run_id, prefix)
    resumed_from = man.get("resume_from") or params.get("resume_from")
    taken_over = set()
    if resumed_from and depth < 3:
        earlier = build_record(r2, str(resumed_from), model, depth + 1)
        taken_over = set(map(tuple, (earlier or {}).get("_done_keys") or []))
    own = {k: v for k, v in work["done"].items() if k not in taken_over}
    thread_s = sum(own.values())
    est = model.lookup(rec["cpu_model"])
    if est:
        per = float(est["per_thread_nodes_per_s"])
        speed_src = f"measured thread speed of {est['name']} ({est.get('confidence', 'measured')})"
    else:
        f, src, _, _ = vm.factor(model.sp, rec["cpu_model"], "tt") if model.have_speed_file else (0.6, "default", 0.5, "")
        per = f * model.k
        speed_src = f"speed-factor {src} ({f:.2f} x k)"
    nodes = thread_s * per
    to, frm = int(params.get("to") or 0), int(params.get("from") or 0)
    ids = params.get("ids")
    n_dec = len(ids) if isinstance(ids, list) else int(params.get("count") or 0)
    sched = "split" if isinstance(params.get("solve_threads"), int) and params["solve_threads"] > 1 else "plan"
    if sched == "plan" and work["trick"] and work["trick"] >= 2 and params.get("worlds_at_once") in (None, "auto"):
        sched = "fill"  # at tricks 2-4 the plan gives every world one thread
    rec.update({"nodes_total": round(nodes), "nodes_basis": f"estimated: {thread_s:.0f} world thread-seconds x {speed_src} "
                                                            "(no node counts were recorded before 2026-10-16)",
                "worlds_finished": len(own), "worlds_total": max(0, n_dec * (to - frm) - len(taken_over)) or None,
                "usd_per_1e12_nodes": round(usd / nodes * 1e12, 3) if nodes and usd else None,
                "nodes_per_s_per_thread": None, "schedule": sched, "trick": work["trick"] or params.get("trick"),
                "tt_mb": (hb.get("progress") or {}).get("tt_mb") or (res.get("tt") or {}).get("tt_mb"),
                "resumed_from": resumed_from, "worlds_taken_over": len(taken_over), "world_thread_s": round(thread_s),
                "_done_keys": sorted(work["done"].keys())})
    rec["settling"] = settling(rec)
    return rec


def backfill(r2, run_ids: list[str], model: M.Model, write: bool, out_dir: str | None) -> list[dict]:
    recs = []
    for run_id in run_ids:
        rec = build_record(r2, run_id, model)
        if rec is None:
            print(f"{run_id}: no manifest in R2, skipped", file=sys.stderr)
            continue
        rec.pop("_done_keys", None)
        recs.append(rec)
        text = json.dumps(rec, indent=2).encode()
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"{run_id}.json"), "wb") as f:
                f.write(text)
        if write:
            st = r2.put(f"machines/{run_id}.json", text)
            print(f"{run_id}: {'written' if st in (200, 201) else f'PUT failed ({st})'}", file=sys.stderr)
    return recs


def bench_run_ids(r2) -> list[str]:
    return sorted({k.split("/")[1] for k, _ in r2.ls("runs/t1bench-") if k.endswith("/out/t1bench/results.json")})


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="one line per machine running now")
    ap.add_argument("--by-cpu", action="store_true", help="per CPU model and schedule")
    ap.add_argument("--rank", action="store_true", help="rank today's vast.ai offers by measured cost")
    ap.add_argument("--record-speeds", action="store_true", help="write measured thread speeds into the speed-factor file")
    ap.add_argument("--backfill", action="store_true", help="build records for past runs from their R2 files")
    ap.add_argument("--runs", help="with --backfill: run ids, comma-separated (default: the listed runs plus every t1bench run with results)")
    ap.add_argument("--write", action="store_true", help="with --backfill: write the records to R2 (machines/<run>.json)")
    ap.add_argument("--out", help="with --backfill: also save the records as JSON files in this directory")
    ap.add_argument("--from", dest="from_dir", help="read records from this directory of JSON files (or one records file) instead of R2")
    ap.add_argument("--markdown", action="store_true", help="markdown tables")
    ap.add_argument("--schedule", default=",".join(SCHEDULES), help="schedules to prefer for --rank/--record-speeds, best first")
    ap.add_argument("--min-threads", type=float, default=64)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--speed-file", default=vm.SPEED_FILE)
    ap.add_argument("--measured-file", default=M.DEFAULT_MEASURED)
    a = ap.parse_args(argv)
    prefer = tuple(s.strip() for s in a.schedule.split(",") if s.strip())
    now = dt.datetime.now(dt.timezone.utc)
    r2 = None if a.from_dir and not a.backfill else R2()
    model = M.Model(a.measured_file, a.speed_file)
    if a.backfill:
        runs = [x.strip() for x in a.runs.split(",") if x.strip()] if a.runs else DEFAULT_BACKFILL + bench_run_ids(r2)
        recs = backfill(r2, runs, model, a.write, a.out)
    else:
        recs = load_records(r2, a.from_dir)
    if a.live:
        print(f"machines running at {now.astimezone(MU):%H:%M} Mauritius ({now:%H:%M} UTC):")
        print("\n".join("  " + x for x in live_lines(recs, now)))
        return 0
    if a.by_cpu:
        print(vm.table(BY_CPU_HEAD, by_cpu_rows(by_cpu(recs)), a.markdown))
        return 0
    if a.rank:
        offers = vs.search_market()
        ranked = rank_offers(offers, recs, model, a.min_threads, prefer)
        print(f"{len(ranked)} rentable offers at {now.astimezone(MU):%H:%M} Mauritius; '~' = estimate (no record of that CPU model yet)")
        print(vm.table(RANK_HEAD, rank_rows(ranked[:a.top]), a.markdown))
        return 0
    if a.record_speeds:
        done = record_speeds(recs, a.speed_file, a.measured_file, prefer)
        print(f"recorded {len(done)} CPU model(s) into {a.speed_file}: {', '.join(done) or 'none'}")
        return 0
    print(vm.table(TABLE_HEAD, table_rows(recs, now), a.markdown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
