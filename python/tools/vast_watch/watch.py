#!/usr/bin/env python3
"""Vast.ai machine dashboard: a local web page that watches the whole vast.ai market, estimates what each
machine would cost for our exact-solver work, ranks the best in three categories, remembers the best
ever seen, and sends browser desktop notifications.

    python3 python/tools/vast_watch/watch.py             # then open http://127.0.0.1:8765
    python3 python/tools/vast_watch/watch.py --refresh-s 5 --data-dir ~/vast-watch-data
    python3 python/tools/vast_watch/watch.py --fake offers.json     # test with made-up offers

How it fits together
- This program asks vast.ai for every on-demand machine (one search takes a few seconds; it repeats as
  soon as the last finished and at least `--refresh-s` seconds apart, default 10), turns each listing
  into an estimated cost per 10^12 solver nodes (python/tools/vast_watch/model.py, calibrated on our
  own benchmark runs in docs/vast/measured-speeds.json), ranks the three categories, and updates the
  all-time records saved in `--data-dir` (history.json). It keeps doing that even when no page is open.
- The page (index.html) asks this program for the latest state every few seconds and shows the table,
  the categories, the records and the notifications. Browsers do not let a page call vast.ai directly,
  which is why this program is in between.
- The page has a "Watching" switch. Off = the page stops asking and this program stops searching vast.ai: nothing
  runs (start with --start-paused to begin that way). "Stop program" ends the program itself. Without either, the
  program keeps searching and saving records while the page is closed (--idle-pause-min N pauses it by itself).
- Standard library only. Read-only: it searches offers and never rents anything.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://console.vast.ai/api/v0/bundles/"
MIN_THREADS_SEARCH = 8

# Every on-demand machine that can be rented; the dashboard filters further. Cheapest first.
QUERY = {
    "rentable": {"eq": True}, "external": {"eq": False}, "type": "on-demand",
    "cpu_cores_effective": {"gte": MIN_THREADS_SEARCH},
    "order": [["dph_total", "asc"]], "limit": 3000, "allocated_storage": 20,
}
KEEP = ["id", "machine_id", "cpu_name", "cpu_cores_effective", "dph_total", "reliability2", "verification",
        "gpu_frac", "num_gpus", "gpu_name", "geolocation", "inet_down_cost", "inet_up_cost", "cpu_ghz"]


def trim(o: dict) -> dict:
    t = {k: o.get(k) for k in KEEP}
    t["ram_gb"] = round((o.get("cpu_ram") or 0) / 1024, 1)
    t["dph_total"] = round(float(o.get("dph_total") or 0), 4)
    t["threads"] = t.pop("cpu_cores_effective")
    return t


def search() -> list[dict]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if os.environ.get("VAST_API_KEY"):  # optional; searching offers does not need a key
        headers["Authorization"] = "Bearer " + os.environ["VAST_API_KEY"]
    req = urllib.request.Request(API, data=json.dumps(QUERY).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        offers = json.load(r).get("offers", [])
    return [trim(o) for o in offers if not o.get("is_bid") and o.get("dph_total")]


class Watcher:
    """Background refresher: market -> estimates -> categories -> records; the newest state is served."""

    def __init__(self, a):
        self.a = a
        self.model = M.Model(a.measured, a.speed_file)
        self.history = M.History(os.path.join(a.data_dir, "history.json") if a.data_dir else None)
        self.series_path = os.path.join(a.data_dir, "series.jsonl") if a.data_dir else None
        self.lock = threading.Lock()
        self.state = {"ready": False, "error": "starting"}
        self.first_seen: dict[int, float] = {}
        self.last_top: str | None = None
        self.last_series = 0.0
        self.stop = threading.Event()
        self.on = threading.Event()  # set = watching; cleared = paused: no search, nothing runs
        if not a.start_paused:
            self.on.set()
        self.token = secrets.token_hex(16)  # the page proves it is ours before it can pause or stop us
        self.last_ask = time.time()
        self.searches = 0

    def fetch(self) -> list[dict]:
        if self.a.fake:
            with open(self.a.fake, encoding="utf-8") as f:
                return json.load(f)
        return search()

    def refresh_once(self):
        t0 = time.time()
        err = None
        try:
            raw = self.fetch()
        except Exception as e:  # keep serving the last good state
            raw, err = None, f"{type(e).__name__}: {e}"
        if raw is None:
            with self.lock:
                if self.state.get("ready"):
                    self.state = dict(self.state, error=err)
                else:
                    self.state = {"ready": False, "error": err}
            return
        now = time.time()
        self.searches += 1
        offers = [M.annotate(self.model, o) for o in raw]
        for o in offers:
            first = self.first_seen.setdefault(o["id"], now)
            o["listed_s"] = round(now - first)
        gone = [i for i in self.first_seen if i not in {o["id"] for o in offers}]
        for i in gone:
            del self.first_seen[i]
        live = M.live_categories(offers, self.a.min_threads, self.a.fast_gn * 1e9)
        self.history.update(live, now)
        self._series(live, now)
        listed = {o["id"] for o in offers}
        state = {"ready": True, "error": None, "checked_at": now, "took_s": round(time.time() - t0, 1),
                 "count": len(offers), "offers": [self._row(o) for o in offers], "live": live,
                 "history": self.history.view(listed),
                 "params": {"min_threads": self.a.min_threads, "fast_gn": self.a.fast_gn, "refresh_s": self.a.refresh_s,
                            "labels": M.LABELS, "data_dir": self.a.data_dir, "speed_models": len(self.model.models)}}
        with self.lock:
            self.state = state

    @staticmethod
    def _row(o: dict) -> dict:
        keys = ["id", "machine_id", "cpu_name", "threads", "usable", "ram_gb", "dph_total", "gpu_name", "num_gpus", "gpu_frac",
                "geolocation", "reliability2", "verification", "inet_down_cost", "inet_up_cost", "launcher_ok", "cost", "nps",
                "score", "source", "unc", "note", "tt_mb", "gb_per_thread", "listed_s"]
        r = {k: o.get(k) for k in keys}
        r["cost"] = round(r["cost"], 4) if r["cost"] is not None else None
        r["score"] = round(r["score"], 4) if r["score"] is not None else None
        r["nps"] = round(r["nps"]) if r["nps"] else None
        return r

    def _series(self, live: dict, now: float):
        """Append a line to series.jsonl when the live top two of any category changed, or every 10 minutes."""
        if not self.series_path:
            return
        top = {c: [(r["id"], r["cost"]) for r in live[c]] for c in M.CATEGORIES}
        sig = json.dumps(top, sort_keys=True)
        if sig == self.last_top and now - self.last_series < 600:
            return
        self.last_top, self.last_series = sig, now
        os.makedirs(os.path.dirname(os.path.abspath(self.series_path)), exist_ok=True)
        line = {"t": round(now), "live": {c: [{k: r[k] for k in ("id", "machine_id", "cpu_name", "threads", "ram_gb", "dph_total", "cost", "nps", "source")}
                                              for r in live[c]] for c in M.CATEGORIES}}
        with open(self.series_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")

    def set_watching(self, on: bool):
        if on:
            self.on.set()
        else:
            self.on.clear()

    def loop(self):
        while not self.stop.is_set():
            if not self.on.wait(timeout=1.0):
                continue  # paused: sleep, do not touch the network
            t0 = time.time()
            self.refresh_once()
            wait = max(0.0, self.a.refresh_s - (time.time() - t0))
            self.stop.wait(wait)
            idle = self.a.idle_pause_min
            if idle and self.on.is_set() and time.time() - self.last_ask > idle * 60:
                self.on.clear()  # nobody has looked for a while: pause by itself

    def get(self) -> dict:
        with self.lock:
            st = dict(self.state)
        st["watching"] = self.on.is_set()
        st["searches"] = self.searches
        return st


def make_handler(w: Watcher):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    html = f.read().replace(b"__TOKEN__", w.token.encode())
                self._send(200, html, "text/html; charset=utf-8")
            elif path == "/api/state":
                w.last_ask = time.time()
                self._send(200, json.dumps(w.get()).encode(), "application/json")
            elif path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                n = min(int(self.headers.get("Content-Length") or 0), 4096)
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return self._send(400, b"bad request", "text/plain")
            if not secrets.compare_digest(str(body.get("token", "")), w.token):
                return self._send(403, b"forbidden", "text/plain")
            if path == "/api/watching":
                w.set_watching(bool(body.get("on")))
                w.last_ask = time.time()
                self._send(200, json.dumps({"watching": w.on.is_set()}).encode(), "application/json")
            elif path == "/api/quit":
                self._send(200, b'{"stopping": true}', "application/json")
                w.on.clear()
                w.stop.set()
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self._send(404, b"not found", "text/plain")

    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this computer only (default)")
    ap.add_argument("--refresh-s", type=float, default=10.0, help="seconds between market searches (at least 3; each takes a few seconds)")
    ap.add_argument("--data-dir", default="vast-watch-data", help="where the all-time records are saved ('' = do not save)")
    ap.add_argument("--min-threads", type=float, default=64, help="smallest machine the three categories rank")
    ap.add_argument("--fast-gn", type=float, default=1.0, help="billion nodes per second a machine needs to count as fast")
    ap.add_argument("--measured", default=M.DEFAULT_MEASURED, help="docs/vast/measured-speeds.json")
    ap.add_argument("--speed-file", default=M.DEFAULT_SPEED, help="docs/vast/speed-factors.json")
    ap.add_argument("--start-paused", action="store_true", help="start with watching switched off (nothing is searched until the page switches it on)")
    ap.add_argument("--idle-pause-min", type=float, default=0, help="pause by itself when no page has asked for this many minutes (0 = never)")
    ap.add_argument("--fake", help="use offers from this JSON file instead of vast.ai (testing)")
    a = ap.parse_args(argv)
    a.refresh_s = max(3.0, a.refresh_s)
    w = Watcher(a)
    threading.Thread(target=w.loop, daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(w))
    print(f"vast dashboard on http://{a.host}:{a.port}   (market search every {a.refresh_s:.0f} s; records in "
          f"{a.data_dir or 'memory only'}; Ctrl-C to stop)", file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    w.stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
