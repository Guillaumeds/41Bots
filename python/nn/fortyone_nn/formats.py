"""File formats shared with Rust (PLAN section 12.7).

Label shards are ``.npy`` record arrays written by ``crates/nn/src/shard.rs``; weights are
safetensors files (a little-endian u64 header length, a JSON header, raw little-endian data).
Only numpy is needed here, so the parity check runs without PyTorch.
"""

import hashlib
import json
import struct

import numpy as np

# The encoder's layout (crates/nn/src/encode.rs, encoder "oracle-v2"). Python never encodes; these
# only size the network and expand the shards' active-index lists.
ENCODER = "oracle-v2"
N_BINARY = 809
N_DENSE = 100
N_FEATURES = N_BINARY + N_DENSE
MAX_ACTIVE = 96
PAD = 0xFFFF
INPUT_SHIFT = 24  # a binary feature that is on, and a dense value of 1, are 2^24

# The binary feature of "team t (rotated order) has won a game" is OFF_GAMES + t
# (crates/nn/src/encode.rs).
OFF_GAMES = 801

# The widest layer of the integer inference (crates/nn/src/mlp.rs `MAX_WIDTH`, audited; raised from
# 1024 to 4096 on 2026-10-16, the proof in that module's notes). A network with a wider layer cannot be
# loaded by Rust, so training one only wastes the time: the trainer refuses it at once
# (``parse_widths``). Tested against Rust: the source line and ``nn-tool limits``.
MAX_WIDTH = 4096


def width_error(k, n):
    """The refusal of hidden layer ``k`` (from 0) with ``n`` units, in the words of Rust's
    ``nn::mlp::width_error`` (tested to be the same text)."""
    return (f"layer {k}: {n} units, wider than {MAX_WIDTH} (the limit of the integer inference, "
            f"`MAX_WIDTH` in crates/nn/src/mlp.rs); use layers of at most {MAX_WIDTH} units")


def parse_widths(text):
    """The hidden widths of ``--widths`` ("1024,512,128") as a list of ints; ``ValueError`` with the
    reason for anything that is not a list of positive whole numbers, or for a layer wider than
    ``MAX_WIDTH`` (named by its number, from 0, and the limit). Checked before any data is loaded."""
    parts = [x.strip() for x in str(text).split(",")]
    try:
        widths = [int(x) for x in parts]
    except ValueError:
        raise ValueError('hidden-layer sizes like "1024,512,128" are needed') from None
    if not widths or any(w < 1 for w in widths):
        raise ValueError('hidden-layer sizes like "1024,512,128" are needed (each at least 1)')
    for k, n in enumerate(widths):
        if n > MAX_WIDTH:
            raise ValueError(width_error(k, n))
    return widths


# Outcome-head vocabulary size by frontier trick (crates/nn/src/outcome.rs).
VOCAB = {5: 8, 4: 14, 3: 24, 2: 39}

# Race classes (crates/nn/src/outcome.rs `race_class`): what can be decided in the hand.
RACE_CLASS_NAMES = (
    "no game can end this hand",
    "one team can win the game, not the match",
    "two or more teams can win the game, not the match",
    "one team can win the match, nobody else the game",
    "a team can win the match and another the game",
)


def shard_dtype(v):
    """The label-shard record type with ``v`` outcomes (crates/nn/src/shard.rs)."""
    return np.dtype(
        [
            ("active", "<u2", (MAX_ACTIVE,)),
            ("dense", "<i4", (N_DENSE,)),
            ("value", "<u8", (4,)),
            ("source", "u1"),
            ("trick", "u1"),
            ("label", "u1"),
            ("mask", "<u8"),
            ("outcomes", "<u8", (v, 4)),
        ]
    )


ONE = 1 << 32


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_shard(path, mmap=True):
    """A label shard as a numpy record array (checked: record type)."""
    a = np.load(path, mmap_mode="r" if mmap else None)
    names = a.dtype.names or ()
    if "outcomes" not in names or a.dtype != shard_dtype(a.dtype["outcomes"].shape[0]):
        raise ValueError(f"{path}: record type {a.dtype} is not a label shard")
    return a


def mask_bits(mask, v):
    """The possible-outcome bits as a boolean array [n, v]."""
    m = np.asarray(mask).astype(np.uint64)
    return ((m[:, None] >> np.arange(v, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(bool)


def race_classes(rows):
    """The race class (0-4) of every row, as ``nn::shard::Row::race_class`` computes it: from the
    possible game-won outcomes (mask bits 0-3) and the teams with a game won (binary features
    ``OFF_GAMES + t``), both in the rotated team order. 0: no team can win the game this hand;
    1: one can, not the match; 2: two or more can, none the match; 3: one team can win the match
    and nobody else the game; 4: a team can win the match and another the game."""
    mask = np.asarray(rows["mask"]).astype(np.uint64)
    act = np.asarray(rows["active"])
    win = np.stack([(mask >> np.uint64(t)) & np.uint64(1) for t in range(4)], axis=1).astype(bool)
    game = np.stack([(act == OFF_GAMES + t).any(axis=1) for t in range(4)], axis=1)
    k = win.sum(axis=1)
    money = (win & game).any(axis=1)
    return np.where(k == 0, 0, np.where(~money, np.where(k == 1, 1, 2), np.where(k == 1, 3, 4))).astype(np.int64)


# What ``check_values`` checks, in its order.
VALUE_FAULTS = (
    "a target that does not sum to 2^32",
    "a label outside the vocabulary",
    "a label that is not a possible outcome",
    "a label whose money vector is not the target",
)


def value_fault(a):
    """The first of ``check_values``' checks (an index into ``VALUE_FAULTS``) that rows ``a`` fail,
    or None. Every check is row by row, so the first fault of a whole array is the smallest of its
    blocks' first faults."""
    s = a["value"].astype(np.uint64).sum(axis=1, dtype=np.uint64)
    if not np.all(s == np.uint64(ONE)):
        return 0
    v = a["outcomes"].shape[1]
    lab = np.asarray(a["label"]).astype(np.int64)
    if np.any(lab >= v):
        return 1
    rows = np.arange(len(a))
    if not np.all(mask_bits(a["mask"], v)[rows, lab]):
        return 2
    if not np.array_equal(np.asarray(a["outcomes"])[rows, lab], np.asarray(a["value"])):
        return 3
    return None


def check_values(a):
    """Targets sum to 2^32, and every row's label is a possible outcome whose money vector is
    exactly the target (the outcome vocabulary's completeness, spec 6a)."""
    f = value_fault(a)
    if f is not None:
        raise ValueError(VALUE_FAULTS[f])


_DTYPES = {"I32": "<i4", "I64": "<i8", "F32": "<f4"}


def read_safetensors(path):
    """(tensors by name as numpy arrays, metadata dict)."""
    with open(path, "rb") as f:
        data = f.read()
    (n,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + n])
    body = data[8 + n :]
    meta = header.pop("__metadata__", {})
    tensors = {}
    for name, t in header.items():
        a, b = t["data_offsets"]
        tensors[name] = np.frombuffer(body[a:b], dtype=_DTYPES[t["dtype"]]).reshape(t["shape"]).copy()
    return tensors, meta


def write_safetensors(path, tensors, meta):
    """Write tensors (name -> numpy array of int32, int64 or float32) in name order; the header is
    padded with spaces to a multiple of 8 bytes (as the reference implementation and Rust do)."""
    names = sorted(tensors)
    header = {}
    if meta:
        header["__metadata__"] = {k: str(v) for k, v in sorted(meta.items())}
    off = 0
    blobs = []
    for name in names:
        a = np.ascontiguousarray(tensors[name])
        kind = {np.dtype("int32"): "I32", np.dtype("int64"): "I64", np.dtype("float32"): "F32"}[a.dtype]
        raw = a.astype(a.dtype.newbyteorder("<")).tobytes()
        header[name] = {"dtype": kind, "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    h = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
