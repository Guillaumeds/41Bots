"""Cost model, category ranking and all-time records for the vast.ai watcher (pure functions).

No network here, so everything can be tested. `watch.py` feeds it market listings.

A listing is turned into an estimated cost per 10^12 solver nodes [positions the exact solver
examines; our work is counted in them]:

    usable threads  = effective cores x 0.958 (the CPU quota every container got), or a known host quirk
    speed per thread = measured on that CPU model (docs/vast/measured-speeds.json), else
                       4.5e6 x the model's 'tt' factor (docs/vast/speed-factors.json), an estimate
    table penalty   = how much slower the machine is for having small solver tables, from its RAM
                       per thread (sweep measured on 2026-10-15), relative to the table used when
                       the speed was measured
    cost per 10^12  = $/hour / (usable threads x speed x 3600) x 10^12

Three categories (each ranks launcher-rentable whole machines with enough threads):
    cheapest      lowest cost per 10^12 nodes, speed does not matter
    fast_cheap    lowest cost among machines that solve at least `fast_nps` nodes per second
    balanced      lowest cost x time (cost per 10^12 divided by speed): cheap AND quick
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
import vast_market as vm  # noqa: E402  (speed factors and the model/quirk lookup)

CATEGORIES = ("cheapest", "fast_cheap", "balanced")
LABELS = {"cheapest": "Cheapest (speed does not matter)",
          "fast_cheap": "Cheapest among the fast",
          "balanced": "Best balanced (cost x time)"}
DEFAULT_MEASURED = os.path.join(REPO, "docs", "vast", "measured-speeds.json")
DEFAULT_SPEED = os.path.join(REPO, "docs", "vast", "speed-factors.json")
NET_CAP = 0.10
# Host reliability floor, matching launch.py MIN_RELIABILITY (owner, 2026-10-16: 90%).
MIN_RELIABILITY = 0.90


# ------------------------------------------------------------------ table-size penalty

DEFAULT_TABLE_TIME = [[32, 2.296], [64, 1.541], [128, 1.158], [256, 1.06], [448, 1.0], [512, 1.0]]


def table_time(mb: float, curve=None) -> float:
    """Relative wall time at `mb` MB per thread (448 MB = 1.0): log-linear between measured points,
    1.5x per halving below the smallest, 1.0 from the largest on."""
    curve = curve or DEFAULT_TABLE_TIME
    if mb <= 0:
        return curve[0][1] * 3.0
    if mb >= curve[-1][0] or mb >= 448:
        return 1.0
    if mb <= curve[0][0]:
        return curve[0][1] * 1.5 ** math.log2(curve[0][0] / mb)
    for (m0, t0), (m1, t1) in zip(curve, curve[1:]):
        if m0 <= mb <= m1:
            f = (math.log2(mb) - math.log2(m0)) / (math.log2(m1) - math.log2(m0))
            return t0 + f * (t1 - t0)
    return 1.0


def table_mb_estimate(ram_gb: float, usable_threads: float) -> float:
    """Table size per thread the bench would pick on this machine (fits the memory plan), capped at 512."""
    if usable_threads <= 0:
        return 0.0
    return min(512.0, 0.88 * ram_gb * 1024.0 / (2.4 * usable_threads))


# ------------------------------------------------------------------ loading

def load_json(path: str | None):
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Model:
    def __init__(self, measured_path: str | None = DEFAULT_MEASURED, speed_path: str | None = DEFAULT_SPEED):
        m = load_json(measured_path) or {}
        self.usable_ratio = float(m.get("usable_thread_ratio", 0.958))
        self.k = float(m.get("default_k_nodes_per_s_per_speed_unit", 4.5e6))
        self.curve = m.get("table_time_vs_448mb") or DEFAULT_TABLE_TIME
        self.models = []
        for e in m.get("models", []):
            self.models.append(dict(e, _re=re.compile(e["match"], re.I)))
        if speed_path and os.path.isfile(speed_path):
            self.sp, self.have_speed_file = vm.load_speed(speed_path), True
        else:
            self.sp, self.have_speed_file = {"models": [], "machines": {}, "quota_ratio": self.usable_ratio}, False

    def lookup(self, cpu_name: str | None):
        for e in self.models:
            if e["_re"].search(cpu_name or ""):
                return e
        return None

    def estimate(self, o: dict) -> dict:
        """Computed fields for one listing: usable threads, speed, table, source, cost per 10^12."""
        eff = float(o.get("threads") or 0)
        quirk = (self.sp.get("machines") or {}).get(str(o.get("machine_id")))
        usable = float(quirk["usable_threads"]) if quirk and quirk.get("usable_threads") else eff * self.usable_ratio
        ram = float(o.get("ram_gb") or 0)
        e = self.lookup(o.get("cpu_name"))
        if e:
            per, ref_mb, source = float(e["per_thread_nodes_per_s"]), float(e.get("tt_mb_during_measurement") or 512), e.get("confidence", "measured")
            unc = 0.03 if source == "measured" else 0.12
            n_runs = e.get("n", 1)
            note = f"{n_runs} bench run(s) of this CPU model" if source == "measured" else "measured on a sibling model"
        else:
            if self.have_speed_file:
                f, src, unc, _ = vm.factor(self.sp, o.get("cpu_name"), "tt")
            else:
                f, src, unc = 0.6, "default", 0.5
            per, ref_mb, source = f * self.k, 512.0, "estimated"
            unc = max(0.15, float(unc or 0.15))
            note = f"estimate from the speed model ({src})"
        tt_mb = table_mb_estimate(ram, usable)
        penalty = table_time(tt_mb, self.curve) / table_time(ref_mb, self.curve)
        nps = usable * per / penalty if usable > 0 and per > 0 else 0.0
        dph = float(o.get("dph_total") or 0)
        cost = dph / (nps * 3600.0) * 1e12 if nps > 0 else None
        return {"usable": round(usable, 1), "per_thread": round(per), "tt_mb": round(tt_mb), "penalty": round(penalty, 3),
                "nps": nps, "cost": cost, "source": source, "unc": round(unc, 2), "note": note,
                "gb_per_thread": round(ram / usable, 2) if usable else None,
                "model_name": e["name"] if e else None}


def launcher_ok(o: dict) -> bool:
    """Can python/launcher/launch.py rent it: whole machine, verified, reliable, cheap network."""
    return ((o.get("gpu_frac") or 0) >= 1 and o.get("verification") == "verified"
            and (o.get("reliability2") or 0) > MIN_RELIABILITY and (o.get("inet_down_cost") or 0) <= NET_CAP
            and (o.get("inet_up_cost") or 0) <= NET_CAP)


def annotate(model: Model, o: dict) -> dict:
    d = dict(o)
    d.update(model.estimate(o))
    d["launcher_ok"] = launcher_ok(o)
    d["score"] = (d["cost"] / (d["nps"] / 1e9)) if d.get("cost") and d["nps"] > 0 else None  # cost x time index
    return d


# ------------------------------------------------------------------ categories

def metric(rec: dict, cat: str):
    return rec.get("score") if cat == "balanced" else rec.get("cost")


def compact(o: dict) -> dict:
    keys = ["id", "machine_id", "cpu_name", "threads", "usable", "ram_gb", "dph_total", "gpu_name", "num_gpus", "geolocation",
            "reliability2", "verification", "cost", "nps", "score", "source", "tt_mb", "note", "unc"]
    r = {k: o.get(k) for k in keys}
    r["cost"] = round(r["cost"], 4) if r["cost"] is not None else None
    r["score"] = round(r["score"], 4) if r["score"] is not None else None
    r["nps"] = round(r["nps"]) if r["nps"] else None
    return r


def top2(recs: list[dict], cat: str) -> list[dict]:
    """The two best records of `cat`, never two listings of the same physical machine."""
    out, seen = [], set()
    for r in sorted((r for r in recs if metric(r, cat) is not None), key=lambda r: (metric(r, cat), r.get("id") or 0)):
        mid = r.get("machine_id") if r.get("machine_id") is not None else ("id", r.get("id"))
        if mid in seen:
            continue
        seen.add(mid)
        out.append(r)
        if len(out) == 2:
            break
    return out


def live_categories(annotated: list[dict], min_threads: float = 64, fast_nps: float = 1.0e9) -> dict:
    scope = [o for o in annotated if o["launcher_ok"] and o["cost"] is not None and (o["threads"] or 0) >= min_threads]
    fast = [o for o in scope if o["nps"] >= fast_nps]
    return {"cheapest": [compact(o) for o in top2(scope, "cheapest")],
            "fast_cheap": [compact(o) for o in top2(fast, "fast_cheap")],
            "balanced": [compact(o) for o in top2(scope, "balanced")]}


# ------------------------------------------------------------------ all-time records

class History:
    """The best two machines ever seen in each category, saved to a JSON file on every change, plus a
    log of the times a record was set. A record keeps the machine's whole configuration and offer id."""

    def __init__(self, path: str | None):
        self.path = path
        self._saved_at = 0.0
        self.data = {"records": {c: [] for c in CATEGORIES}, "events": [], "next_event": 1}
        d = load_json(path)
        if d:
            for c in CATEGORIES:
                self.data["records"][c] = d.get("records", {}).get(c, [])
            self.data["events"] = d.get("events", [])
            self.data["next_event"] = d.get("next_event", 1)

    def update(self, live: dict, now: float | None = None) -> list[dict]:
        """Fold the current market's best machines into the records; returns the new events.
        An event is a machine entering a category's top two, or the same machine getting cheaper.
        The very first fill of a category is logged as a baseline (the page does not notify for it)."""
        now = now or time.time()
        events = []
        for cat in CATEGORIES:
            old = self.data["records"][cat]
            pool = {}  # one record per physical machine: the best one seen
            for r in old:
                pool[r.get("machine_id")] = dict(r)
            improved = set()
            for c in live.get(cat, []):
                mid = c.get("machine_id")
                rec = dict(c, seen_at=now, last_seen=now)
                prev = pool.get(mid)
                if prev is None:
                    pool[mid] = rec
                elif (metric(rec, cat) or 1e18) < (metric(prev, cat) or 1e18) * 0.999:
                    rec["seen_at"] = prev.get("seen_at", now)
                    pool[mid] = rec
                    improved.add(mid)
                else:
                    prev["last_seen"] = now
                    prev["last_offer"] = c["id"]
            new = top2(list(pool.values()), cat)
            old_ids = {r.get("machine_id") for r in old}
            for rank, r in enumerate(new, start=1):
                mid = r.get("machine_id")
                if mid not in old_ids:
                    why = "baseline" if not old else "new machine"
                    replaced = old[rank - 1] if rank - 1 < len(old) else None
                    events.append(self._event(cat, rank, r, replaced, why, now))
                elif mid in improved:
                    prev = next(o for o in old if o.get("machine_id") == mid)
                    events.append(self._event(cat, rank, r, prev, "same machine, lower cost", now))
            self.data["records"][cat] = new
        if events:
            self.data["events"] = (self.data["events"] + events)[-500:]
        if events or now - self._saved_at > 60:  # last_seen alone is saved at most once a minute
            self.save()
            self._saved_at = now
        return events

    def _event(self, cat, rank, rec, replaced, why, now):
        e = {"id": self.data["next_event"], "t": now, "category": cat, "rank": rank, "why": why,
             "record": {k: rec.get(k) for k in ("id", "machine_id", "cpu_name", "threads", "usable", "ram_gb", "dph_total", "gpu_name",
                                                 "num_gpus", "geolocation", "cost", "nps", "score", "source")},
             "replaced": ({k: replaced.get(k) for k in ("id", "cpu_name", "cost", "score")} if replaced else None)}
        self.data["next_event"] += 1
        return e

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)

    def view(self, listed_ids: set) -> dict:
        recs = {c: [dict(r, still_listed=(r.get("id") in listed_ids or r.get("last_offer") in listed_ids)) for r in self.data["records"][c]]
                for c in CATEGORIES}
        return {"records": recs, "events": self.data["events"][-60:][::-1], "last_event": self.data["next_event"] - 1}
