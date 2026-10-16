#!/usr/bin/env python3
"""Snapshot vast.ai's on-demand market and rank machines for our CPU-heavy jobs (docs/VAST-MACHINES.md).

    python3 python/tools/vast_market.py                      # live snapshot, ranked tables
    python3 python/tools/vast_market.py --save docs/vast/market-$(date -u +%F).json
    python3 python/tools/vast_market.py --from docs/vast/market-2026-10-14.json     # re-rank a saved snapshot
    python3 python/tools/vast_market.py --markdown            # tables ready to paste into a doc
    python3 python/tools/vast_market.py record --cpu "AMD EPYC 9754 128-Core Processor" \\
        --compute 1.21 --tt 0.93 --run m3-bench-9754 --note "fixed trick-2 set, 512 threads"

Read-only: it searches offers and never rents anything. Standard library only.

Speed factors come from docs/vast/speed-factors.json: the speed of ONE hardware thread (a vast.ai
"effective core") of each CPU model with every thread busy, relative to one thread of an AMD EPYC 7502
(1.00). `compute` is for jobs with small tables (exact solves from trick 2 on, equity sweeps, rollouts);
`tt` is for jobs dominated by huge transposition tables (exact trick-1 solves). A `measured` value written
by `record` (from our own benchmarks) always replaces the `estimate`.

One unit of work = one hour of one EPYC 7502 thread with all threads busy. For each offer:
    threads    = usable threads (96% of vast's effective cores, the CPU quota every container got so far,
                 unless the speed file knows the machine gives less)
    work/hour  = threads x factor
    $ per work = $/hour / work/hour   (shown per 1,000 units)

Rankings (owner's rule as in python/launcher/launch.py, 20% premium by default):
  A  cheapest per unit of compute work;
  B  among offers within +20% of A's best $ per work, the most work per hour;
  C  exact trick-1 solves: the tt factor, only machines with enough RAM for several big tables at once
     (--t1-min-ram-gb, --t1-min-gb-per-thread); cheapest first, then the 20% rule.

Credentials: VAST_API_KEY, if set, is sent as a Bearer token (searching offers does not need it).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SPEED_FILE = os.path.join(REPO, "docs", "vast", "speed-factors.json")
API = "https://console.vast.ai/api/v0/bundles/"

# The offer fields kept in a saved snapshot.
KEEP = ["id", "machine_id", "host_id", "cpu_name", "cpu_cores", "cpu_cores_effective", "cpu_ghz", "cpu_ram",
        "dph_total", "reliability2", "verification", "geolocation", "num_gpus", "gpu_name", "gpu_frac",
        "disk_space", "inet_down_cost", "inet_up_cost", "mobo_name", "hosting_type", "vms_enabled", "static_ip",
        "end_date"]


# ------------------------------------------------------------------ queries

# Max $/GB up or down, matching launch.py --max-net-cost (owner 2026-10-14: 0.10; our traffic is tiny).
NET_CAP = 0.10
# Host reliability floor, matching launch.py MIN_RELIABILITY (owner, 2026-10-16: 90%).
MIN_RELIABILITY = 0.90

def base_query(min_cores: int, min_ram_gb: int) -> dict:
    """The launcher's CPU-only filter (launch.py offer_query with --cpu-only), as API JSON."""
    return {
        "verified": {"eq": True}, "external": {"eq": False}, "rentable": {"eq": True},
        "reliability": {"gt": MIN_RELIABILITY}, "num_gpus": {"gte": 1}, "gpu_frac": {"eq": 1},
        "cpu_cores_effective": {"gte": min_cores}, "cpu_ram": {"gte": min_ram_gb * 1000},
        "disk_space": {"gte": 20}, "inet_down_cost": {"lte": NET_CAP}, "inet_up_cost": {"lte": NET_CAP},
        "order": [["dph_total", "asc"]], "type": "on-demand", "limit": 2000, "allocated_storage": 20,
    }


def queries(min_cores: int, min_ram_gb: int) -> dict[str, dict]:
    """Named searches, run in this order; an offer is tagged with the first one that finds it.
    Only 'eligible' offers can be rented by launch.py; the others show the rest of the market."""
    eligible = base_query(min_cores, min_ram_gb)
    net = {k: v for k, v in eligible.items() if k not in ("inet_down_cost", "inet_up_cost")}
    partial = dict(eligible, gpu_frac={"lt": 1})
    low_rel = dict(eligible, reliability={"gt": 0.80, "lte": MIN_RELIABILITY})
    unverified = dict(eligible, verified={"eq": False})
    return {"eligible": eligible, "network_cost_over_cap": net, "partial_machine": partial,
            "reliability_0.80-0.90": low_rel, "unverified": unverified}


WHY = {"network_cost_over_cap": f"network > ${NET_CAP}/GB", "partial_machine": "shared machine",
       "reliability_0.80-0.90": f"reliability < {MIN_RELIABILITY}", "unverified": "unverified"}


def search(q: dict) -> list[dict]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if os.environ.get("VAST_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VAST_API_KEY"]
    req = urllib.request.Request(API, data=json.dumps(q).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r).get("offers", [])


def trim(o: dict, query: str) -> dict:
    t = {k: o.get(k) for k in KEEP}
    t["cpu_ram_gb"] = round((o.get("cpu_ram") or 0) / 1024, 1)
    del t["cpu_ram"]
    t["dph_total"] = round(float(o.get("dph_total") or 0), 4)
    t["query"] = query
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()}


def snapshot(min_cores: int, min_ram_gb: int) -> dict:
    qs = queries(min_cores, min_ram_gb)
    offers, seen = [], set()
    for name, q in qs.items():
        for o in search(q):
            if o.get("is_bid") or not o.get("dph_total") or o.get("id") in seen:
                continue
            seen.add(o["id"])
            offers.append(trim(o, name))
    return {"format": "41bots-vast-market", "version": 1,
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "api": API, "queries": qs, "offers": offers}


# ------------------------------------------------------------------ speed factors

def load_speed(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        sp = json.load(f)
    for m in sp["models"]:
        m["_rx"] = re.compile(m["match"], re.I)
    return sp


def find_model(sp: dict, cpu_name: str | None) -> dict | None:
    for m in sp["models"]:
        if m["_rx"].search(cpu_name or ""):
            return m
    return None


def factor(sp: dict, cpu_name: str | None, workload: str) -> tuple[float, str, float, str]:
    """(factor, 'measured'|'estimate'|'default', uncertainty, model name) for one CPU and workload."""
    m = find_model(sp, cpu_name)
    if m is None:
        d = sp["default"]
        return d[workload], "default", d["uncertainty"], "(unknown: " + (cpu_name or "?").strip() + ")"
    meas = m.get("measured") or {}
    if meas.get(workload) is not None:
        return float(meas[workload]), "measured", float(meas.get("uncertainty", 0.05)), m["name"]
    e = m["estimate"]
    return float(e[workload]), "estimate", float(e["uncertainty"]), m["name"]


def channels_per_socket(m: dict | None) -> int | None:
    g = re.match(r"(\d+) channels", (m or {}).get("memory", ""))
    return int(g.group(1)) if g else None


def annotate(o: dict, sp: dict) -> dict:
    """Add the derived columns (prefixed _) to an offer."""
    m = find_model(sp, o.get("cpu_name"))
    quirk = sp.get("machines", {}).get(str(o.get("machine_id")))
    eff = float(o.get("cpu_cores_effective") or 0)
    threads = float(quirk["usable_threads"]) if quirk and quirk.get("usable_threads") else eff * sp["quota_ratio"]
    o["_threads"] = threads
    o["_quirk"] = f"host gives {quirk['usable_threads']} threads" if quirk else ""
    o["_model"] = m["name"] if m else "(unknown: " + (o.get("cpu_name") or "?").strip() + ")"
    o["_arch"] = m["arch"] if m else "?"
    for w in ("compute", "tt"):
        f, src, unc, _ = factor(sp, o.get("cpu_name"), w)
        o[f"_{w}"], o[f"_{w}_src"], o[f"_{w}_unc"] = f, src, unc
        o[f"_{w}_work_h"] = threads * f
        o[f"_{w}_cost"] = o["dph_total"] / (threads * f) * 1000 if threads * f > 0 else math.inf
    o["_ram_per_thread"] = o["cpu_ram_gb"] / threads if threads else 0
    # Fewer DIMMs than memory channels? Full population needs sockets x channels DIMMs of >= 16 GB.
    o["_mem_flag"] = ""
    cps, ch = (m or {}).get("cores_per_socket"), channels_per_socket(m)
    cores = o.get("cpu_cores") or 0
    if cps and ch and o.get("gpu_frac") == 1 and cores % (2 * cps) == 0:
        sockets = cores // (2 * cps)
        if o["cpu_ram_gb"] < 0.9 * sockets * ch * 16:
            o["_mem_flag"] = "RAM below full population: some memory channels likely empty"
    return o


# ------------------------------------------------------------------ ranking

def rank(offers: list[dict], workload: str, premium: float) -> tuple[list[dict], list[dict]]:
    """(all offers cheapest first, offers within the premium sorted by work per hour)."""
    cost = f"_{workload}_cost"
    ranked = sorted(offers, key=lambda o: (o[cost], o["id"]))
    if not ranked:
        return [], []
    near = [o for o in ranked if o[cost] <= ranked[0][cost] * (1 + premium)]
    near.sort(key=lambda o: (-o[f"_{workload}_work_h"], o[cost], o["id"]))
    return ranked, near


def t1_ok(o: dict, min_ram_gb: float, min_gb_per_thread: float) -> bool:
    return o["cpu_ram_gb"] >= min_ram_gb and o["_ram_per_thread"] >= min_gb_per_thread


def model_summary(offers: list[dict]) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for o in offers:
        by.setdefault(o["_model"], []).append(o)
    rows = []
    for name, os_ in by.items():
        best = min(os_, key=lambda o: o["_compute_cost"])
        rows.append(dict(
            model=name, arch=os_[0]["_arch"], offers=len(os_),
            threads=sorted({int(o.get("cpu_cores_effective") or 0) for o in os_}),
            dph_min=min(o["dph_total"] for o in os_), dph_med=statistics.median(o["dph_total"] for o in os_),
            thread_h=statistics.median(o["dph_total"] / o["_threads"] for o in os_),
            compute=os_[0]["_compute"], tt=os_[0]["_tt"], src=os_[0]["_compute_src"], unc=os_[0]["_compute_unc"],
            cost_best=best["_compute_cost"], work_h_best=best["_compute_work_h"],
            tt_cost_best=min(o["_tt_cost"] for o in os_),
            ram_per_thread=statistics.median(o["_ram_per_thread"] for o in os_),
            ram_max=max(o["cpu_ram_gb"] for o in os_)))
    rows.sort(key=lambda r: r["cost_best"])
    return rows


# ------------------------------------------------------------------ printing

def table(headers: list[str], rows: list[list], markdown: bool) -> str:
    rows = [[str(c) for c in r] for r in rows]
    if markdown:
        out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out)
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = lambda r: "  ".join(c.ljust(w[i]) for i, c in enumerate(r))  # noqa: E731
    return "\n".join([line(headers), line(["-" * x for x in w])] + [line(r) for r in rows])


def offer_rows(offers: list[dict], workload: str, table_gb: float = 0.0) -> list[list]:
    rows = []
    for o in offers:
        extra = []
        if table_gb:
            n = int(o["cpu_ram_gb"] * 0.9 // table_gb)
            extra = [n, f"{o['_threads'] / n:.0f}" if n else "-"]
        f, src = o[f"_{workload}"], o[f"_{workload}_src"]
        flags = "; ".join(x for x in (o["_quirk"], o["_mem_flag"]) if x)
        rows.append([o["id"], o["_model"], f"{int(o.get('cpu_cores_effective') or 0)}", f"{o['_threads']:.0f}",
                     f"{o['dph_total']:.2f}", f"{f:.2f}{'' if src == 'measured' else '~'}",
                     f"{o[f'_{workload}_work_h']:.0f}", f"{o[f'_{workload}_cost']:.2f}", f"{o['cpu_ram_gb']:.0f}",
                     f"{o['_ram_per_thread']:.1f}", f"{o.get('reliability2') or 0:.3f}",
                     (o.get("geolocation") or "?").split(",")[-1].strip()] + extra + [flags])
    return rows


OFFER_HEAD = ["offer", "CPU model", "eff. cores", "threads used", "$/h", "speed/thread", "work/h",
              "$ per 1k work", "RAM GB", "GB/thread", "reliab.", "country", "flags"]
T1_HEAD = OFFER_HEAD[:-1] + ["deals at once", "threads per deal", "flags"]


def report(snap: dict, sp: dict, a) -> str:
    out = []
    md = a.markdown
    offers = [annotate(dict(o), sp) for o in snap["offers"]]
    elig = [o for o in offers if o["query"] == "eligible"]
    min_cores = snap["queries"].get("eligible", {}).get("cpu_cores_effective", {}).get("gte", a.min_cores)
    floor = snap["queries"].get("eligible", {}).get("reliability", {}).get("gt", MIN_RELIABILITY)  # a saved snapshot keeps its own floor
    out.append(f"vast.ai on-demand market at {snap['utc']}: {len(elig)} offers launch.py can rent "
               f"(whole machine, verified, reliability > {floor}, network <= ${NET_CAP}/GB, >= {min_cores} effective cores)"
               + "".join(f"; {sum(o['query'] == q for o in offers)} {q}" for q in snap["queries"] if q != "eligible"))
    out.append("Speed: per hardware thread with all threads busy, EPYC 7502 = 1.00; '~' = estimate, not yet measured. "
               "1 unit of work = 1 hour of one EPYC 7502 thread.\n")

    out.append("## By CPU model (eligible offers, cheapest compute work first)\n")
    rows = []
    for r in model_summary(elig):
        rows.append([r["model"], r["arch"], r["offers"], "/".join(map(str, r["threads"])),
                     f"{r['dph_min']:.2f}" + ("" if r["dph_med"] - r["dph_min"] < 0.005 else f"-{r['dph_med']:.2f}"),
                     f"{r['thread_h'] * 1000:.1f}", f"{r['compute']:.2f}{'' if r['src'] == 'measured' else '~'}",
                     f"{r['tt']:.2f}", f"±{r['unc'] * 100:.0f}%", f"{r['cost_best']:.2f}", f"{r['work_h_best']:.0f}",
                     f"{r['tt_cost_best']:.2f}", f"{r['ram_per_thread']:.1f}", f"{r['ram_max']:.0f}"])
    out.append(table(["CPU model", "architecture", "offers", "eff. cores", "$/h (min-median)", "$ per 1k thread-h",
                      "compute speed", "tt speed", "uncertainty", "$ per 1k work (best)", "work/h (that offer)",
                      "$ per 1k tt-work (best)", "GB/thread", "max RAM GB"], rows, md))

    ranked, near = rank(elig, "compute", a.premium)
    out.append(f"\n## A. Cheapest per unit of work (compute jobs: trick 2-5 solves, equity sweeps, rollouts), top {a.top}\n")
    out.append(table(OFFER_HEAD, offer_rows(ranked[:a.top], "compute"), md))
    out.append(f"\n## B. Fastest within +{a.premium * 100:.0f}% of the cheapest $ per work (owner's rule), top {a.top}\n")
    out.append(table(OFFER_HEAD, offer_rows(near[:a.top], "compute"), md))

    t1 = [o for o in elig if t1_ok(o, a.t1_min_ram_gb, a.t1_min_gb_per_thread)]
    ranked_t, near_t = rank(t1, "tt", a.premium)
    out.append(f"\n## C. Exact trick-1 solves (tt speed; RAM >= {a.t1_min_ram_gb:.0f} GB and >= "
               f"{a.t1_min_gb_per_thread:.1f} GB per thread): {len(t1)} of {len(elig)} offers qualify\n")
    out.append(f"'deals at once' = how many {a.t1_table_gb:.0f} GB tables fit in 90% of the RAM; fewer threads per "
               "deal waste less (one deal on 368 threads ran at about 1/7 of the per-thread speed of 4 threads).\n")
    out.append("Cheapest per unit of tt-work:\n")
    out.append(table(T1_HEAD, offer_rows(ranked_t[:a.top], "tt", a.t1_table_gb), md))
    out.append(f"\nFastest within +{a.premium * 100:.0f}% of that:\n")
    out.append(table(T1_HEAD, offer_rows(near_t[:a.top], "tt", a.t1_table_gb), md))

    if a.include_other:
        other = [o for o in offers if o["query"] in ("network_cost_over_cap", "partial_machine")]
        for o in other:
            o["_quirk"] = WHY[o["query"]] + ("; " + o["_quirk"] if o["_quirk"] else "")
        ranked_o, _ = rank(other, "compute", a.premium)
        out.append(f"\n## Not rentable by launch.py's current rules (network cost over the cap, or part of a "
                   f"machine shared with other renters), cheapest compute work first, top {a.top}\n")
        out.append(table(OFFER_HEAD, offer_rows(ranked_o[:a.top], "compute"), md))
    unknown = sorted({o["_model"] for o in offers if o["_compute_src"] == "default"})
    if unknown:
        out.append("\nCPU models with no entry in the speed file (default factor used): " + ", ".join(unknown))
    return "\n".join(out)


# ------------------------------------------------------------------ record a measurement

def cmd_record(a) -> int:
    with open(a.file, encoding="utf-8") as f:
        sp = json.load(f)
    target = None
    for m in sp["models"]:
        if re.search(m["match"], a.cpu, re.I):
            target = m
            break
    if target is None:
        core = re.sub(r"\s+\d+-Core.*$", "", a.cpu.strip())
        target = {"name": core, "match": re.escape(core) + r"\b", "arch": "?", "cores_per_socket": None,
                  "clock_ghz": {"base": None, "boost": None}, "l3_mb_per_socket": None, "l3_mb_per_core": None,
                  "l3_layout": "?", "memory": "?",
                  "estimate": dict(sp["default"], sources=[]), "measured": None, "caveats": "added by record"}
        sp["models"].append(target)
        print(f"no model matched; added {core!r}", file=sys.stderr)
    meas = dict(target.get("measured") or {})
    if a.compute is not None:
        meas["compute"] = a.compute
    if a.tt is not None:
        meas["tt"] = a.tt
    if a.uncertainty is not None:
        meas["uncertainty"] = a.uncertainty
    runs = list(meas.get("runs", []))
    runs.append({"run": a.run, "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "cpu_name": a.cpu,
                 "compute": a.compute, "tt": a.tt, "note": a.note})
    meas["runs"] = runs
    target["measured"] = meas
    sp["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with open(a.file, "w", encoding="utf-8") as f:
        json.dump(sp, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{target['name']}: measured {json.dumps({k: v for k, v in meas.items() if k != 'runs'})}")
    return 0


# ------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speed", default=SPEED_FILE, help="speed-factor file (default docs/vast/speed-factors.json)")
    p.add_argument("--from", dest="src", help="re-rank a saved snapshot instead of searching vast.ai")
    p.add_argument("--save", help="write the trimmed snapshot to this JSON file")
    p.add_argument("--min-cores", type=int, default=64)
    p.add_argument("--min-ram-gb", type=int, default=32)
    p.add_argument("--premium", type=float, default=0.20, help="owner's speed premium (default 20%%)")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--t1-min-ram-gb", type=float, default=480.0,
                   help="trick-1 ranking: minimum machine RAM in GB as vast reports it (a 512 GB machine shows ~503)")
    p.add_argument("--t1-table-gb", type=float, default=64.0,
                   help="trick-1 ranking: RAM per deal solved at once (the worst trick-1 deal so far peaked at 65 GB)")
    p.add_argument("--t1-min-gb-per-thread", type=float, default=1.0, help="trick-1 ranking: minimum RAM per thread")
    p.add_argument("--include-other", action="store_true",
                   help="also rank offers launch.py refuses today (network cost over the cap, shared machines)")
    p.add_argument("--markdown", action="store_true", help="print markdown tables")
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("record", help="store a measured speed factor for a CPU model")
    r.add_argument("--cpu", required=True, help="vast.ai cpu_name (or any text the model's pattern matches)")
    r.add_argument("--compute", type=float, help="measured compute factor (per thread, EPYC 7502 = 1)")
    r.add_argument("--tt", type=float, help="measured tt factor (per thread, EPYC 7502 = 1)")
    r.add_argument("--uncertainty", type=float, help="relative error of the measurement (default 0.05)")
    r.add_argument("--run", required=True, help="run id of the benchmark")
    r.add_argument("--note", default="")
    r.add_argument("--file", default=SPEED_FILE)
    a = p.parse_args(argv)
    if a.cmd == "record":
        if a.compute is None and a.tt is None:
            p.error("record needs --compute and/or --tt")
        return cmd_record(a)
    sp = load_speed(a.speed)
    if a.src:
        with open(a.src, encoding="utf-8") as f:
            snap = json.load(f)
    else:
        snap = snapshot(a.min_cores, a.min_ram_gb)
    if a.save:
        with open(a.save, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"saved {len(snap['offers'])} offers to {a.save}", file=sys.stderr)
    print(report(snap, sp, a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
