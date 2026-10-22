"""Root-restricted solve: exactness and speed through the app API (path-forward.md, session 0c).

For each request in DIR (bench-t{trick}-p{place}-{k}.json from `engine-native bench`), value the
decision in full, then again with a `shortlist` of the K best cards (by the full run), and check that
every shortlisted card has the byte-identical sums; then time both with `engine-native time`.

    python tools/shortlist_time.py ENGINE_NATIVE DIR OUT.json --equity F --net F --tricks 2 3 --ks 2 3
"""
import argparse
import json
import os
import subprocess
import tempfile
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("native")
ap.add_argument("dir")
ap.add_argument("out")
ap.add_argument("--equity", required=True)
ap.add_argument("--net", action="append", default=[])
ap.add_argument("--tricks", type=int, nargs="+", default=[2, 3])
ap.add_argument("--ks", type=int, nargs="+", default=[2, 3])
ap.add_argument("--repeat", type=int, default=2)
a = ap.parse_args()
base = [a.native, "--equity", a.equity] + [x for n in a.net for x in ("--net", n)]


def run(cmd, files):
    p = subprocess.run(base + cmd + files, capture_output=True, text=True, check=True)
    return [l for l in p.stdout.splitlines() if l.strip()]


def card(c):
    return c["action"]["card"]


work = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.out)))
rows = []
for trick in a.tricks:
    names = sorted(f for f in os.listdir(a.dir) if f.startswith(f"bench-t{trick}-"))
    full = {}
    for n in names:
        full[n] = json.loads(run(["eval"], [f"{a.dir}/{n}"])[0])
    for k in a.ks:
        files, meta = [], {}
        for n in names:
            r = full[n]
            cs = r["candidates"]
            if r["status"] != "ok" or len(cs) <= k:
                continue
            order = sorted(range(len(cs)), key=lambda i: -cs[i]["value"])[:k]
            req = json.load(open(f"{a.dir}/{n}"))
            req["shortlist"] = sorted(card(cs[i]) for i in order)
            p = f"{work}/{n[:-5]}-k{k}.json"
            json.dump(req, open(p, "w"))
            sub = json.loads(run(["eval"], [p])[0])
            assert sub["status"] == "ok", sub
            want = {card(cs[i]): cs[i]["sum_fp"] for i in order}
            got = {card(c): c["sum_fp"] for c in sub["candidates"]}
            assert got == want, (n, k, "shortlisted sums differ from the full solve")
            files.append(p)
            meta[p] = n
        # time: full and shortlisted, same requests
        for kind, fs in (("full", [f"{a.dir}/{meta[p]}" for p in files]), (f"k{k}", files)):
            for l in run(["time", "--repeat", str(a.repeat)], fs):
                f = dict(x.split("=", 1) for x in l.split("\t")[1:])
                nm = os.path.basename(l.split("\t")[0])
                place = int(nm.split("-p")[1].split("-")[0].split(".")[0])
                rows.append({"trick": trick, "k": k, "kind": kind, "place": place, "file": nm,
                             "ms": float(f["ms"]), "deals": int(f["deals"]),
                             "cands": len(full[meta.get(l.split("\t")[0], os.path.basename(l.split("\t")[0]))]["candidates"]) if False else None})
json.dump(rows, open(a.out, "w"))
agg = defaultdict(lambda: defaultdict(list))
for r in rows:
    agg[(r["trick"], r["k"], r["place"])][r["kind"]].append(r["ms"] / r["deals"])
print("trick  k  place  ms/deal full -> shortlisted   (ratio)   [positions]")
for (t, k, p), d in sorted(agg.items()):
    f = sum(d["full"]) / len(d["full"])
    s = sum(d[f"k{k}"]) / len(d[f"k{k}"])
    print(f"{t:>5} {k:>2} {p:>6}  {f:9.3f} -> {s:9.3f}   ({s/f:.2f})   [{len(d['full'])}]")
for t in a.tricks:
    for k in a.ks:
        f = sum(sum(d["full"]) for (tt, kk, _), d in agg.items() if (tt, kk) == (t, k))
        s = sum(sum(d[f"k{k}"]) for (tt, kk, _), d in agg.items() if (tt, kk) == (t, k))
        if f:
            print(f"trick {t} k={k}: total ms/deal summed over places {f:.2f} -> {s:.2f} (ratio {s/f:.2f})")
