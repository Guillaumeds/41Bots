"""The student's data and integer network without PyTorch (docs/STUDENT.md section 5): feature
shards (written by ``student encode``, Rust) and their sidecars, training columns, the decision loss
against the teacher's means, and the integer forward pass exactly as ``crates/student`` computes it
(the oracle's ``intnet`` on the student's inputs). The parity tests run with numpy only.
"""

import json

import numpy as np

from . import intnet
from .formats import ONE, write_safetensors

FEATURES_FORMAT = "41bots-student-features-v1"
ENCODER = "student-v1"
HEAD = "student-money4"
PAD = 0xFFFF
CLIP_F = 64.0


def load_features(path):
    """A feature shard (memory-mapped record array) and its sidecar, checked: format, encoder."""
    with open(path + ".json") as f:
        side = json.load(f)
    if side.get("format") != FEATURES_FORMAT:
        raise SystemExit(f"{path}: not a student feature shard ({side.get('format')})")
    if side.get("encoder") != ENCODER:
        raise SystemExit(f"{path}: encoder {side.get('encoder')}, not {ENCODER}")
    a = np.load(path, mmap_mode="r")
    if len(a) != side["records"]:
        raise SystemExit(f"{path}: {len(a)} records, the sidecar says {side['records']}")
    return a, side


def check_same(sides, names):
    keys = ("encoder_sha256", "n_view", "n_binary", "n_dense", "v_max", "k_max", "c_max")
    for k in keys:
        vals = {s[k] for s in sides}
        if len(vals) != 1:
            raise SystemExit(f"shards {names} disagree on {k}: {sorted(map(str, vals))}")


class Columns:
    """Training columns of decisions (numpy), compact: the view's and the candidates' active
    indices as stored (uint16, 0xFFFF the padding: ``indices`` turns a block into the network's
    int64 inputs), the dense values (raw and as floats), the candidate mask, the targets (fixed
    point and float), and per decision its trick, place, stratum and position key."""

    def __init__(self, recs, n_binary):
        self.n_binary = n_binary
        self.view = np.asarray(recs["view"]).astype(np.uint16)
        self.dense_raw = np.asarray(recs["dense"]).astype(np.int64)
        self.dense = self.dense_raw.astype(np.float32) / float(1 << 24)
        nc = np.asarray(recs["n_cands"]).astype(np.int64)
        self.n_cands = nc
        cmax = recs.dtype["cand"].shape[0]
        self.mask = np.arange(cmax)[None, :] < nc[:, None]
        self.cand = np.asarray(recs["cand"]).astype(np.uint16)
        self.target_fp = np.asarray(recs["target"]).astype(np.uint64)
        self.target = (self.target_fp.astype(np.float64) / float(ONE)).astype(np.float32)
        self.trick = np.asarray(recs["trick"]).astype(np.int64)
        self.place = np.asarray(recs["place"]).astype(np.int64)
        self.stratum = np.asarray(recs["stratum"]).astype(np.int64)
        self.pos_key = np.asarray(recs["pos_key"]).astype(np.uint64)

    def __len__(self):
        return len(self.n_cands)

    def take(self, idx):
        c = Columns.__new__(Columns)
        for k, v in self.__dict__.items():
            setattr(c, k, v if k == "n_binary" else v[idx])
        return c


def indices(a, n_binary):
    """Stored active indices (uint16, padding 0xFFFF; ascending, the padding last) as int64 inputs
    of an embedding bag, the padding turned into ``n_binary`` (its zero row) and the columns cut
    to the longest list of the block."""
    a = np.asarray(a)
    width = int((a != PAD).sum(axis=-1).max()) if a.size else 0
    out = a[..., : max(width, 1)].astype(np.int64)
    out[out == PAD] = n_binary
    return out


def concat(cols):
    c = Columns.__new__(Columns)
    for k in cols[0].__dict__:
        if k == "n_binary":
            c.n_binary = cols[0].n_binary
        else:
            setattr(c, k, np.concatenate([getattr(x, k) for x in cols]))
    return c


def choose(own, mask):
    """Each decision's choice: the first candidate with the largest own-pair value."""
    v = np.where(mask, own, -np.inf)
    return np.argmax(v, axis=1)  # numpy's argmax returns the first maximum


def decision_report(own_student, c):
    """Decision loss (fraction of money chance) and agreement against the teacher's means, from
    the student's own-pair values [n, C] (any monotone score: logits' softmax or exact money)."""
    t = c.target_fp[:, :, 0].astype(np.float64)  # the deciding pair is pair 0
    t_masked = np.where(c.mask, t, -1.0)
    best = t_masked.max(axis=1)
    pick = choose(own_student, c.mask)
    got = t[np.arange(len(t)), pick]
    loss = (best - got) / float(ONE)
    return {
        "decisions": int(len(t)),
        "decision_loss_pct": float(100.0 * loss.mean()) if len(t) else 0.0,
        "agreement": float((loss == 0).mean()) if len(t) else 0.0,
    }


def by_groups(own_student, c):
    out = {"all": decision_report(own_student, c)}
    for name, col, vals in (("trick", c.trick, range(1, 6)), ("place", c.place, range(8)),
                            ("stratum", c.stratum, range(1, 5))):
        for v in vals:
            sel = col == v
            if sel.any():
                label = f"{name} {v + 1}" if name == "place" else f"{name} {v}"
                out[label] = decision_report(own_student[sel], c.take(sel))
    return out


def int_inputs(c, n_binary, n_features, rows):
    """The integer input matrix of the candidates of decisions ``rows``: each candidate's row has
    its decision's view features and its own at 2^24 and the dense values."""
    d, k = np.nonzero(c.mask[rows])
    d_abs = np.asarray(rows)[d] if not isinstance(rows, slice) else np.arange(len(c))[rows][d]
    x = np.zeros((len(d), n_features), dtype=np.int64)
    one = 1 << 24
    for j, (dd, kk) in enumerate(zip(d_abs, k)):
        v = c.view[dd]
        x[j, v[v != PAD].astype(np.int64)] = one
        cc = c.cand[dd, kk]
        x[j, cc[cc != PAD].astype(np.int64)] = one
        x[j, n_binary:] = c.dense_raw[dd]
    return x, d, k


def int_logits(layers, c, n_binary, n_features, chunk=256):
    """Every candidate's logits [n, C, 4] (int64, units of 2^-24, rotated pair order; 0 past a
    decision's candidates): the exact integer forward pass of ``crates/student`` (products by
    ``intnet.exact_matmul``), what ``student eval`` writes."""
    out = np.zeros(c.mask.shape + (4,), dtype=np.int64)
    for i in range(0, len(c), chunk):
        rows = slice(i, min(i + chunk, len(c)))
        x, d, k = int_inputs(c, n_binary, n_features, rows)
        if len(x):
            out[i + d, k] = intnet.logits(layers, x)
    return out


def int_scores(layers, c, n_binary, n_features, chunk=256):
    """Own-pair money chance (exact integer forward pass and softmax, as Rust) [n, C], -inf past a
    decision's candidates."""
    z = int_logits(layers, c, n_binary, n_features, chunk)
    own = np.full(c.mask.shape, -np.inf)
    d, k = np.nonzero(c.mask)
    for dd, kk in zip(d, k):
        own[dd, kk] = intnet.money(z[dd, kk])[0]
    return own


def write_student_int(path, int_layers, meta, side):
    tensors = {}
    for k, (w, b, _) in enumerate(int_layers):
        tensors[f"w{k}"] = w
        tensors[f"b{k}"] = b
    m = dict(meta)
    m.update({
        "format": intnet.FORMAT,
        "encoder": ENCODER,
        "encoder_sha256": side["encoder_sha256"],
        "n_features": str(side["n_binary"] + side["n_dense"]),
        "head": HEAD,
        "shifts": json.dumps([int(s) for _, _, s in int_layers]),
    })
    write_safetensors(path, tensors, m)


