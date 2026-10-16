"""File formats shared with Rust (PLAN section 12.7).

Label shards are ``.npy`` record arrays written by ``crates/nn/src/shard.rs``; weights are
safetensors files (a little-endian u64 header length, a JSON header, raw little-endian data).
Only numpy is needed here, so the parity check runs without PyTorch.
"""

import hashlib
import json
import struct

import numpy as np

# The encoder's layout (crates/nn/src/encode.rs, encoder "oracle-v1"). Python never encodes; these
# only size the network and expand the shards' active-index lists.
ENCODER = "oracle-v1"
N_BINARY = 809
N_DENSE = 20
N_FEATURES = N_BINARY + N_DENSE
MAX_ACTIVE = 96
PAD = 0xFFFF
INPUT_SHIFT = 15  # a binary feature that is on, and a dense value of 1, are 2^15

SHARD_DTYPE = np.dtype(
    [
        ("active", "<u2", (MAX_ACTIVE,)),
        ("dense", "<u2", (N_DENSE,)),
        ("value", "<u8", (4,)),
        ("source", "u1"),
        ("trick", "u1"),
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
    """A label shard as a numpy record array (checked: record type, values summing to 2^32)."""
    a = np.load(path, mmap_mode="r" if mmap else None)
    if a.dtype != SHARD_DTYPE:
        raise ValueError(f"{path}: record type {a.dtype} is not a label shard")
    return a


def check_values(a):
    s = a["value"].astype(np.uint64).sum(axis=1, dtype=np.uint64)
    if not np.all(s == np.uint64(ONE)):
        raise ValueError("a target that does not sum to 2^32")


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
