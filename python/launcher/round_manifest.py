"""The manifest of one O3G2 round (docs/m3-hybrid/o3-rounds.md) from the previous round's record.

A round starts from the previous round's ``round.json`` (stored by the runner at
``rounds/<id>/round.json`` in R2) or, for round 1, from ``docs/vast/o3-rounds/seed.json``; both
hold a ``next`` block: the incumbent's integer and float networks, its G2 (leaf-risk model), the
stored training set (``label_inputs``) and the golden set. The fixed settings come from
``docs/vast/o3-rounds/defaults.json``. Only the round's own choices are given here: its number,
the new rows of each candidate, optionally a wider network trained from scratch for the last
candidate, the risk power and whether to rebuild G2.

    python3 python/launcher/launch.py round-manifest --round 1 --from seed --amounts 2M,4M,8M
    python3 python/launcher/launch.py round-manifest --round 2 --from o3r-1 --amounts 2M,4M,8M \\
        --scratch-last 4096,2048,1024,256

Seeds: round ``n`` uses ``90000 + 100 n`` (the risk profile), ``+1 .. +3`` (the new parts) and
``+11 .. +13`` (the next G2's train, cal and test splits): never another round's, never the base's.
Standard library only.
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROUNDS_DIR = os.path.join(HERE, "..", "..", "docs", "vast", "o3-rounds")
SEED_BASE = 90000
SEED_STEP = 100

_AMOUNT = re.compile(r"^(\d+(?:\.\d+)?)([kKmM]?)$")


def parse_amount(text):
    """``500k`` -> 500000, ``2M`` -> 2000000, ``1.5M`` -> 1500000, ``750000`` -> 750000."""
    m = _AMOUNT.match(text.strip())
    if not m:
        raise ValueError(f"amount {text!r}: a number of rows like 500k, 2M or 1.5M")
    mult = {"": 1, "k": 1000, "m": 1000000}[m.group(2).lower()]
    v = float(m.group(1)) * mult
    if v != int(v) or v <= 0:
        raise ValueError(f"amount {text!r}: a whole number of rows above 0")
    return int(v)


def parse_amounts(text):
    """One to three growing amounts: the candidates' new rows (nested: each candidate trains on the
    earlier ones' new rows and more)."""
    v = [parse_amount(x) for x in text.split(",") if x.strip()]
    if not 1 <= len(v) <= 3:
        raise ValueError(f"amounts {text!r}: one to three, e.g. 2M,4M,8M")
    if any(b <= a for a, b in zip(v, v[1:])):
        raise ValueError(f"amounts {text!r}: each larger than the one before (they are nested)")
    return v


def parse_widths(text):
    w = [int(x) for x in text.replace("-", ",").split(",") if x.strip()]
    if not w or any(x < 1 or x > 4096 for x in w):
        raise ValueError(f"widths {text!r}: hidden-layer sizes of 1 to 4096, e.g. 4096,2048,1024,256")
    return ",".join(str(x) for x in w)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def defaults():
    return load_json(os.path.join(ROUNDS_DIR, "defaults.json"))


def seed_record():
    return load_json(os.path.join(ROUNDS_DIR, "seed.json"))


def round_id(n):
    return f"o3r-{n}"


def round_seed(n):
    return SEED_BASE + SEED_STEP * n


def dataset_rows(entries):
    return sum(int(e.get("rows") or 0) for e in entries)


def build(n, prev, amounts, scratch_last=None, risk_power=None, g2="auto", dflt=None, note=""):
    """The manifest of round ``n`` from ``prev`` (a round.json or the seed record)."""
    d = dflt or defaults()
    if n < 1:
        raise ValueError("the round number is 1 or more")
    nxt = prev.get("next") or {}
    for k in ("init_net", "init_net_float", "label_inputs", "golden_decisions_input", "golden_worlds_input"):
        if not nxt.get(k):
            raise ValueError(f"the previous round's record has no next.{k}: it did not finish; start from its last good round")
    if g2 not in ("auto", "rebuild"):
        raise ValueError("g2: auto (reuse the previous round's G2 of this network) or rebuild")
    seed = round_seed(n)
    r = dict(d["round"])
    g = r.pop("g2")
    counts = g.pop("counts")
    g["splits"] = [{"name": name, "count": counts[name], "seed": seed + 11 + i}
                   for i, name in enumerate(("train", "cal", "test"))]
    cands = [{"rows": a} for a in amounts]
    if scratch_last:
        cands[-1] = {"rows": amounts[-1], "mode": "scratch", "widths": parse_widths(scratch_last)}
    rnd = {"id": round_id(n), "seed": seed, **r, "candidates": cands, "g2": g}
    if risk_power is not None:
        rnd["risk_power"] = str(risk_power)
    if g2 == "auto" and nxt.get("risk_model"):
        rnd["risk_model"] = nxt["risk_model"]
    params = dict(d["rung"])
    params.update({
        "init_net": nxt["init_net"],
        "init_net_float": nxt["init_net_float"],
        "label_inputs": nxt["label_inputs"],
        "golden_decisions_input": nxt["golden_decisions_input"],
        "golden_worlds_input": nxt["golden_worlds_input"],
        "round": rnd,
    })
    rows = dataset_rows(nxt["label_inputs"]) + amounts[-1]
    disk = d["disk"]
    machine = dict(d["machine"])
    machine["disk_gb"] = int(disk["base_gb"] + disk["gb_per_million_rows"] * rows / 1e6 + 0.999)
    text = (f"O3G2 round {n} (docs/m3-hybrid/o3-rounds.md), from {prev.get('id') or 'the previous round'}: "
            f"new G2-chosen rows {', '.join(str(a) for a in amounts)} (nested), "
            f"{'last candidate ' + cands[-1]['widths'] + ' from scratch, ' if scratch_last else ''}"
            f"risk power {rnd['risk_power']}, plain floor {rnd['risk_floor']}; G2 "
            f"{'reused' if rnd.get('risk_model') else 'built first'}. {note}").strip()
    return {
        "format": 2,
        "run_id": f"{round_id(n)}-1",
        "run_seed": 1,
        "threads": None,
        "benchmark_seconds": 0,
        "upload_every_s": 300,
        "heartbeat_every_s": 30,
        "note": text,
        "machine": machine,
        "streams": [{"name": "rung", "priority": 1, "kind": "hybrid_rung", "chunks": 1, "params": params}],
    }


def previous(from_id, fetch):
    """The previous round's record: the seed for ``seed``, else ``rounds/<from>/round.json`` read by
    ``fetch(key) -> bytes | None``."""
    if from_id == "seed":
        return seed_record()
    data = fetch(f"rounds/{from_id}/round.json")
    if data is None:
        raise ValueError(f"no rounds/{from_id}/round.json in R2: did round {from_id} finish?")
    rec = json.loads(data)
    rec.setdefault("id", from_id)
    return rec
