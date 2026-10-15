#!/usr/bin/env python3
"""Terminal view of a vast.ai run, plus a check for a second, comparable machine.

    python3 python/tools/vast_status.py                 # the newest run, once
    python3 python/tools/vast_status.py RUN_ID --watch  # refresh every 30 s until Ctrl-C
    python3 python/tools/vast_status.py RUN_ID --watch --every 15 --cap 4.00
    python3 python/tools/vast_status.py --market-only   # only the second-machine check (no R2 keys; used by the
                                                        # "market-check" GitHub workflow)
    python3 python/tools/vast_status.py --market-only --watch   # keep checking every 30 s, bell + banner on a YES

Shows: where the run is (decisions, worlds, percent), time left and the finish time in Mauritius time,
cost so far and the forecast, speed, CPU use, upload health; and whether the market now has a second
machine of comparable cost and speed that our launcher can rent (yes/no, and if yes its offer number
and details), so you can say "start run A on that one".

Reads the run's files from R2 (environment: R2_S3_ENDPOINT, or R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY_ID,
R2_SECRET_ACCESS_KEY, the same as the launcher) and searches vast.ai offers (no key needed).
Read-only: it never rents or destroys anything. Standard library only.

"Comparable" = a verified, whole machine our launcher can rent whose estimated cost per 10^12 solver
nodes is at most --max-cost-ratio (default 1.3) times this run's machine and whose estimated speed is at
least --min-speed-ratio (default 0.7) times. Estimates come from our own benchmark runs
(docs/vast/measured-speeds.json); a machine of an unbenchmarked chip is labelled "estimate".
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vast_watch"))
import model as M  # noqa: E402  (cost model, shared with the dashboard)

MU = dt.timezone(dt.timedelta(hours=4))  # Mauritius


# ------------------------------------------------------------------ R2 (read-only, SigV4)

class R2:
    def __init__(self):
        try:
            # The workflows and the launcher call it R2_S3_ENDPOINT; R2_ENDPOINT is the older name.
            self.ep = (os.environ.get("R2_S3_ENDPOINT") or os.environ["R2_ENDPOINT"]).rstrip("/")
            self.bucket = os.environ["R2_BUCKET"]
            self.ak, self.sk = os.environ["R2_ACCESS_KEY_ID"], os.environ["R2_SECRET_ACCESS_KEY"]
        except KeyError as e:
            sys.exit(f"missing environment variable {e}: set R2_S3_ENDPOINT (or R2_ENDPOINT), R2_BUCKET, R2_ACCESS_KEY_ID, "
                     f"R2_SECRET_ACCESS_KEY")
        self.host = urllib.parse.urlparse(self.ep).netloc

    @staticmethod
    def _h(k, m):
        return hmac.new(k, m.encode(), hashlib.sha256).digest()

    def get(self, path, query=None):
        query = query or {}
        now = dt.datetime.now(dt.timezone.utc)
        amz, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
        cq = "&".join(f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(str(v), safe='')}" for k, v in sorted(query.items()))
        cp = urllib.parse.quote(path, safe="/")
        ph = hashlib.sha256(b"").hexdigest()
        hd = f"host:{self.host}\nx-amz-content-sha256:{ph}\nx-amz-date:{amz}\n"
        sh = "host;x-amz-content-sha256;x-amz-date"
        creq = f"GET\n{cp}\n{cq}\n{hd}\n{sh}\n{ph}"
        scope = f"{day}/auto/s3/aws4_request"
        sts = f"AWS4-HMAC-SHA256\n{amz}\n{scope}\n{hashlib.sha256(creq.encode()).hexdigest()}"
        k = self._h(self._h(self._h(self._h(("AWS4" + self.sk).encode(), day), "auto"), "s3"), "aws4_request")
        sig = hmac.new(k, sts.encode(), hashlib.sha256).hexdigest()
        auth = f"AWS4-HMAC-SHA256 Credential={self.ak}/{scope}, SignedHeaders={sh}, Signature={sig}"
        req = urllib.request.Request(f"{self.ep}{cp}" + (f"?{cq}" if cq else ""),
                                     headers={"x-amz-date": amz, "x-amz-content-sha256": ph, "Authorization": auth})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def ls(self, prefix):
        out, token = [], None
        while True:
            q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if token:
                q["continuation-token"] = token
            st, b = self.get(f"/{self.bucket}", q)
            if st != 200:
                raise RuntimeError(f"R2 list failed ({st})")
            ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
            root = ET.fromstring(b)
            out += [(c.find("s:Key", ns).text, c.find("s:LastModified", ns).text) for c in root.findall("s:Contents", ns)]
            if root.findtext("s:IsTruncated", namespaces=ns) != "true":
                return out
            token = root.findtext("s:NextContinuationToken", namespaces=ns)

    def json(self, key):
        st, b = self.get(f"/{self.bucket}/{key}")
        if st != 200:
            return None
        try:
            return json.loads(b)
        except ValueError:
            return None


def latest_run(r2: R2) -> str:
    best = None
    for k, t in r2.ls("runs/"):
        parts = k.split("/")
        if len(parts) == 3 and parts[2] == "manifest.json" and (best is None or t > best[0]):
            best = (t, parts[1])
    if not best:
        sys.exit("no runs found in R2")
    return best[1]


# ------------------------------------------------------------------ the comparable-machine check

def search_market() -> list[dict]:
    import watch as W  # the dashboard's own vast.ai search (no key needed)
    return W.search()


def reference_machine(model: M.Model, hw: dict, dph: float) -> dict | None:
    """Annotate the machine the run is on, from its hardware.json and the price it is billed at."""
    cpu = hw.get("cpu_model") or hw.get("cpu_name")
    usable = hw.get("cgroup_quota_cores") or hw.get("available_parallelism")
    ram = (hw.get("mem_total_kb") or 0) / 1048576.0 or hw.get("mem_total_gb") or hw.get("ram_gb")
    if not (cpu and usable and dph):
        return None
    if not ram:
        ram = 256.0
    o = {"id": 0, "machine_id": -1, "cpu_name": cpu, "threads": float(usable) / model.usable_ratio, "ram_gb": float(ram),
         "dph_total": float(dph), "gpu_frac": 1.0, "verification": "verified", "reliability2": 1.0,
         "inet_down_cost": 0, "inet_up_cost": 0}
    return M.annotate(model, o)


def comparable(model: M.Model, offers: list[dict], ref: dict, max_cost_ratio: float, min_speed_ratio: float,
               measured_only: bool = False, min_threads: float = 64):
    """(matches, closest): offers our launcher can rent that are within the cost and speed limits, best first;
    and the closest few when there are none."""
    ann = [M.annotate(model, o) for o in offers]
    pool = [o for o in ann if o["launcher_ok"] and o["cost"] is not None and (o["threads"] or 0) >= min_threads
            and (not measured_only or o["source"] != "estimated")]
    ok = [o for o in pool if o["cost"] <= ref["cost"] * max_cost_ratio and o["nps"] >= ref["nps"] * min_speed_ratio]
    ok.sort(key=lambda o: (o["cost"], -o["nps"]))
    gap = lambda o: max(o["cost"] / ref["cost"] / max_cost_ratio, ref["nps"] * min_speed_ratio / o["nps"])
    closest = sorted(pool, key=gap)[:3] if not ok else []
    return ok, closest


# ------------------------------------------------------------------ display

class C:
    def __init__(self, on):
        self.on = on

    def __call__(self, code, s):
        return f"\033[{code}m{s}\033[0m" if self.on else s


def hms(s):
    s = int(max(0, s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def clock(t: dt.datetime) -> str:
    return t.astimezone(MU).strftime("%H:%M")


def bar(frac, width=30):
    frac = min(1.0, max(0.0, frac or 0.0))
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def words(o):
    cpu = (o.get("cpu_name") or "?").replace("AMD Ryzen ", "AMD ").replace("®", "").replace("™", "").strip()
    return f"{cpu} · {round(o.get('usable') or o.get('threads') or 0)} threads · {round(o.get('ram_gb') or 0)} GB RAM · ${o.get('dph_total', 0):.2f}/h"


def run_panel(r2: R2, run: str, cap: float | None, c: C) -> tuple[list[str], dict | None]:
    lines, ref = [], None
    files = {k.split("/", 2)[2] for k, _ in r2.ls(f"runs/{run}/") if "/heartbeat/" not in k and k.count("/") == 2}
    hbs = sorted(k for k, _ in r2.ls(f"runs/{run}/heartbeat/"))
    hb = r2.json(hbs[-1]) if hbs else None
    hw = r2.json(f"runs/{run}/hardware.json") or {}
    report = r2.json(f"runs/{run}/report.json") if "report.json" in files else None
    failed = r2.json(f"runs/{run}/FAILED") if "FAILED" in files else None
    state = "FINISHED OK" if "DONE" in files else "FAILED" if "FAILED" in files else "RUNNING" if hb else "STARTING"
    col = {"FINISHED OK": "32", "FAILED": "31", "RUNNING": "36", "STARTING": "33"}[state]
    now = dt.datetime.now(dt.timezone.utc)
    lines.append(c("1", f"RUN {run}") + "    " + c("1;" + col, state) + f"    now {clock(now)} Mauritius ({now:%H:%M} UTC)")
    if not hb:
        lines.append("No heartbeat yet: the machine is still being rented or booting (usually 1-3 minutes).")
        return lines, None
    p = hb.get("progress") or {}
    el = float(hb.get("elapsed_s") or 0)
    cost = float(hb.get("est_cost_usd") or 0)
    dph = cost / el * 3600 if el > 60 and cost else None
    hb_age = (now - dt.datetime.fromisoformat(hb["utc"].replace("Z", "+00:00"))).total_seconds() if hb.get("utc") else None
    cpu = hw.get("cpu_model") or "?"
    lines.append(f"Machine   {cpu} · {hb.get('threads')} threads usable" + (f" · about ${dph:.2f}/h" if dph else ""))
    eta = p.get("eta") or {}
    frac = eta.get("done_frac")
    if frac is None and p.get("worlds_total"):
        frac = (p.get("worlds_done") or 0) / p["worlds_total"]
    lines.append(f"Progress  [{bar(frac)}] {100 * (frac or 0):.1f}%   worlds {p.get('worlds_done')}/{p.get('worlds_total')}   decisions {p.get('decisions_done')}/{p.get('decisions_total')}   step: {p.get('step', hb.get('phase'))}")
    if state == "RUNNING" and eta.get("remaining_s") is not None:
        fin = dt.datetime.fromisoformat(eta["eta_utc"].replace("Z", "+00:00")) if eta.get("eta_utc") else None
        basis = str(eta.get("basis") or "")
        trust = c("32", "measured forecast") if basis.startswith("measured") else c("33", "rough forecast (needs 10% done)")
        lines.append(f"Time      elapsed {hms(el)}   remaining about {hms(eta['remaining_s'])}   finish about {clock(fin) if fin else '?'} Mauritius   [{trust}]")
    else:
        lines.append(f"Time      elapsed {hms(el)}")
    proj = eta.get("projected_usd")
    capnote = ""
    if cap and proj:
        capnote = c("31", f"   OVER the ${cap:.2f} cap") if proj > cap else f"   (cap ${cap:.2f})"
    lines.append(f"Cost      so far ${cost:.2f}" + (f"   projected total ${proj:.2f}" if proj else "") + capnote)
    done = p.get("worlds_done") or 0
    rate = f"{done / (el / 60):.1f} worlds/min" if el > 120 and done else "–"
    cpu_pct = hb.get("cpu_util_pct") or hb.get("cgroup_cpu_pct")
    cpu_txt = f"{cpu_pct:.0f}%" if cpu_pct is not None else "–"
    lines.append(f"Speed     CPU {c('32' if (cpu_pct or 0) >= 85 else '31', cpu_txt)}   {rate}" + (f"   ${cost / done:.4f} per world" if done and cost else ""))
    up = f"uploads OK ({hb.get('upload_failures', 0)} failures)" if not hb.get("upload_failures") else c("31", f"{hb.get('upload_failures')} upload failures")
    lines.append(f"Health    last heartbeat {hb_age:.0f} s ago · {up} · last saved {hb.get('last_upload_ok_s', '?')} s ago" +
                 (c("31", "   <- heartbeat is late") if hb_age and hb_age > 150 and state == "RUNNING" else ""))
    for w in (p.get("warnings") or [])[:3]:
        lines.append(c("33", "Warning   " + str(w)[:160]))
    if failed:
        lines.append(c("31", "Failed    " + str(failed.get("error") or failed)[:200]))
    if report:
        lines.append(f"Report    outcome {report.get('outcome')} · {report.get('elapsed_min')} min · about ${report.get('dollars_est')}")
    if dph:
        ref = {"hw": hw, "dph": dph, "state": state}
    return lines, ref


def market_panel(model: M.Model, ref: dict | None, a, c: C, plain: list[str] | None = None) -> list[str]:
    """The second-machine check. `ref` = {cost, nps, source} of the machine in use. `plain` collects the same
    result as plain text lines (for the GitHub run summary)."""
    lines = ["", c("1", "SECOND MACHINE FOR RUN A")]
    if not ref:
        return lines + ["  (cannot read this machine's hardware yet; try again in a minute)"]
    try:
        offers = search_market()
    except Exception as e:
        return lines + [c("31", f"  could not search vast.ai: {e}")]
    ok, closest = comparable(model, offers, ref, a.max_cost_ratio, a.min_speed_ratio, a.measured_only)
    lines.append(f"  This run's machine: est. ${ref['cost']:.3f} per 10¹² nodes, {ref['nps'] / 1e9:.2f} billion nodes/s ({ref['source']}).")
    lines.append(f"  Comparable = cost at most ${ref['cost'] * a.max_cost_ratio:.3f} and speed at least {ref['nps'] * a.min_speed_ratio / 1e9:.2f} billion nodes/s, verified whole machine.")
    now = dt.datetime.now(dt.timezone.utc)
    stamp = f"checked {clock(now)} Mauritius, {len(offers)} machines on the market"
    if plain is not None:
        plain.append(f"checked {clock(now)} Mauritius ({now:%H:%M} UTC), {len(offers)} machines on the market. Comparable = at most "
                     f"${ref['cost'] * a.max_cost_ratio:.3f} per 10¹² nodes and at least {ref['nps'] * a.min_speed_ratio / 1e9:.2f} billion nodes/s, verified whole machine.")
    if ok:
        lines.append(c("1;32", f"  AVAILABLE: YES  ({len(ok)} found; {stamp})"))
        for i, o in enumerate(ok[:3], 1):
            lines.append(f"   {i}. offer {c('1', str(o['id']))}: {words(o)} · {o.get('geolocation')}")
            lines.append(f"      est. ${o['cost']:.3f} per 10¹² nodes, {o['nps'] / 1e9:.2f} billion nodes/s ({o['source']}), reliability {o.get('reliability2'):.3f}")
            if plain is not None:
                plain.append(f"{i}. **offer {o['id']}**: {words(o)} · {o.get('geolocation')}: est. ${o['cost']:.3f} per 10¹² nodes, {o['nps'] / 1e9:.2f} billion nodes/s ({o['source']})")
        lines.append(c("32", f"  -> tell Claude: \"start run A on offer {ok[0]['id']}\""))
        lines.append("RESULT: YES offer " + str(ok[0]["id"]))
    else:
        lines.append(c("1;31", f"  AVAILABLE: NO  ({stamp})"))
        for i, o in enumerate(closest, 1):
            txt = f"offer {o['id']}: {words(o)} · {o.get('geolocation')} · est. ${o['cost']:.3f} per 10¹² nodes, {o['nps'] / 1e9:.2f} bn nodes/s ({o['source']})"
            lines.append(f"   closest {i}: {txt}")
            if plain is not None:
                plain.append(f"closest {i}: {txt}")
        lines.append("RESULT: NO")
    if plain is not None:
        plain.insert(0, "YES" if ok else "NO")
    return lines


def github_summary(plain: list[str]):
    """Put the result at the top of the GitHub run page (and as a notice when a machine is available)."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path or not plain:
        return
    verdict = plain[0]
    md = [f"## Second machine for run A: {'YES, available' if verdict == 'YES' else 'NO, nothing comparable right now'}", ""]
    md += [f"- {x}" for x in plain[1:]]
    if verdict == "YES":
        md += ["", "Tell Claude: **start run A on offer <the first offer number above>**."]
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    if verdict == "YES":
        print("::notice title=Comparable machine available::" + (plain[2] if len(plain) > 2 else plain[1])[:200].replace("\n", " "))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", nargs="?", help="run id (default: the newest run)")
    ap.add_argument("--watch", action="store_true", help="refresh until Ctrl-C")
    ap.add_argument("--every", type=float, default=30.0, help="seconds between refreshes with --watch (default 30)")
    ap.add_argument("--cap", type=float, default=None, help="the run's dollar cap, to flag a forecast above it")
    ap.add_argument("--max-cost-ratio", type=float, default=1.3, help="a second machine may cost this many times per node (default 1.3)")
    ap.add_argument("--min-speed-ratio", type=float, default=0.7, help="and must be at least this fast, relative (default 0.7)")
    ap.add_argument("--measured-only", action="store_true", help="only count machines whose chip we have benchmarked")
    ap.add_argument("--no-market", action="store_true", help="skip the second-machine check")
    ap.add_argument("--market-only", action="store_true", help="only the second-machine check, against --ref-cost/--ref-gn (no R2 keys needed)")
    ap.add_argument("--ref-cost", type=float, default=0.285, help="with --market-only: the running machine's $ per 10^12 nodes (default 0.285, the 9995WX)")
    ap.add_argument("--ref-gn", type=float, default=1.24, help="with --market-only: its speed in billion nodes/s (default 1.24)")
    ap.add_argument("--color", choices=["auto", "on", "off"], default="auto")
    a = ap.parse_args(argv)
    c = C(a.color == "on" or (a.color == "auto" and sys.stdout.isatty()))
    model = M.Model()
    if a.market_only:
        ref = {"cost": a.ref_cost, "nps": a.ref_gn * 1e9, "source": "given reference"}
        last_yes = None
        while True:
            plain: list[str] = []
            lines = market_panel(model, ref, a, c, plain)
            if a.watch and c.on:
                sys.stdout.write("\033[H\033[2J")
            print("\n".join(lines))
            github_summary(plain)
            result = next((x for x in lines if x.startswith("RESULT:")), "RESULT: ?")
            if not a.watch:
                return 0
            if result.startswith("RESULT: YES") and result != last_yes:
                sys.stdout.write("\a")  # terminal bell, once per new best machine
                print(c("1;32", "\n*** A COMPARABLE MACHINE IS AVAILABLE: tell Claude to start run A on it ***"))
            last_yes = result if result.startswith("RESULT: YES") else None
            print(c("2", f"\n(checks every {a.every:.0f} s; Ctrl-C to stop. Nothing is rented by this view.)"), flush=True)
            try:
                time.sleep(a.every)
            except KeyboardInterrupt:
                return 0
    r2 = R2()
    run = a.run or latest_run(r2)
    while True:
        try:
            lines, ref = run_panel(r2, run, a.cap, c)
            if not a.no_market and ref and ref.get("state") == "RUNNING":
                lines += market_panel(model, reference_machine(model, ref["hw"], ref["dph"]), a, c)
            elif not a.no_market and ref:
                lines += ["", c("1", "SECOND MACHINE FOR RUN A"), "  (the run has ended; ask for a fresh market check when you want to start run A)"]
        except Exception as e:  # keep the loop alive through a network blip
            lines = [c("31", f"error: {type(e).__name__}: {e}")]
        if a.watch and c.on:
            sys.stdout.write("\033[H\033[2J")
        print("\n".join(lines))
        if not a.watch:
            return 0
        print(c("2", f"\n(refreshes every {a.every:.0f} s; Ctrl-C to stop watching. The run keeps going without this view.)"), flush=True)
        try:
            time.sleep(a.every)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
