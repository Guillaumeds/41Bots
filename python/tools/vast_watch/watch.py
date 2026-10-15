#!/usr/bin/env python3
"""Vast.ai machine watcher: a small local web page that tells you (browser desktop notification)
when a machine you are waiting for appears on the vast.ai market.

    python3 python/tools/vast_watch/watch.py            # then open http://127.0.0.1:8765
    python3 python/tools/vast_watch/watch.py --port 9000
    python3 python/tools/vast_watch/watch.py --fake offers.json    # test with made-up offers

Why a program and not just a page: vast.ai does not let a web page call its search directly, so this
program asks vast.ai for you (one search every few seconds at most, shared by all open pages) and the
page asks this program. Standard library only. Read-only: it searches offers and never rents anything.

The page does the rest: you choose what to wait for (CPU name, threads, RAM, price), switch
notifications on or off, and while the page is open (a minimised or background tab is fine, a closed
one is not) it checks every 15-60 seconds and pops up a desktop notification for each machine that
newly matches. Machines already there when you open the page are the baseline and do not notify.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://console.vast.ai/api/v0/bundles/"
CACHE_S = 10  # one vast.ai search at most this often, however many pages ask
MIN_THREADS = 8  # the search drops tiny machines

# Every on-demand machine that can be rented; the page filters further. Sorted cheapest first.
QUERY = {
    "rentable": {"eq": True}, "external": {"eq": False}, "type": "on-demand",
    "cpu_cores_effective": {"gte": MIN_THREADS},
    "order": [["dph_total", "asc"]], "limit": 3000, "allocated_storage": 20,
}

KEEP = ["id", "machine_id", "cpu_name", "cpu_cores_effective", "dph_total", "reliability2",
        "verification", "gpu_frac", "num_gpus", "gpu_name", "geolocation", "inet_down_cost",
        "inet_up_cost", "cpu_ghz"]


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


class Source:
    """Offers from vast.ai (cached), or from a JSON file with --fake (re-read on every request)."""

    def __init__(self, fake: str | None):
        self.fake, self.lock, self.at, self.cached, self.error = fake, threading.Lock(), 0.0, [], None

    def get(self) -> tuple[list[dict], str | None, float]:
        if self.fake:
            with open(self.fake, encoding="utf-8") as f:
                return json.load(f), None, time.time()
        with self.lock:
            if time.time() - self.at >= CACHE_S:
                try:
                    self.cached, self.error = search(), None
                except Exception as e:  # keep the last good list; the page shows the error
                    self.error = f"{type(e).__name__}: {e}"
                self.at = time.time()
            return self.cached, self.error, self.at


def make_handler(src: Source):
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
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif path == "/api/offers":
                offers, err, at = src.get()
                body = json.dumps({"checked_at": at, "error": err, "count": len(offers), "offers": offers})
                self._send(200, body.encode(), "application/json")
            elif path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            else:
                self._send(404, b"not found", "text/plain")

    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this computer only (default)")
    ap.add_argument("--fake", help="serve offers from this JSON file instead of vast.ai (testing)")
    a = ap.parse_args(argv)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(Source(a.fake)))
    print(f"vast watcher on http://{a.host}:{a.port}  (Ctrl-C to stop)", file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
