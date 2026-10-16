"""Per-row loss weights: the sidecar ``NAME.npy.rowweights`` (``fortyone_nn.formats.load_row_weights``,
``crates/nn/src/rowweights.rs``).

- the format: ``numpy.save`` of a ``<u2`` array writes the bytes Rust's writer writes (the golden
  hash in the Rust source), and the constants (256 is a weight of 1, the suffix) are the Rust ones;
- ``load_row_weights``: none without a sidecar; refused: another type, another shape, another
  length than the shard's rows, a weight of 0;
- parity with Rust (when the ``nn-tool`` binary is there, as ``test_intnet``): ``nn-tool
  row-weights`` writes the sidecar this module reads, byte for byte what ``numpy.save`` writes,
  and its summary (rows, rows not 1, sum, min, max, SHA-256) is numpy's; bad weights and rows are
  refused.

numpy only (the CI's parity job): ``cd python/nn && uv run python -m unittest discover -s tests -v``.
"""

import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import unittest

import numpy as np

from fortyone_nn.formats import ROW_WEIGHT_ONE, ROW_WEIGHTS_SUFFIX, load_row_weights, sha256_file, shard_dtype

from test_intnet import ROOT, nn_tool

RUST = ROOT / "crates" / "nn" / "src" / "rowweights.rs"


def npy_bytes(a):
    f = io.BytesIO()
    np.save(f, a)
    return f.getvalue()


def empty_shard(path, rows, v=14):
    """A label shard of ``rows`` zero records (the sidecar tools read only its header)."""
    np.save(path, np.zeros(rows, dtype=shard_dtype(v)))
    return path


class Format(unittest.TestCase):
    def test_numpy_writes_the_rust_bytes(self):
        src = RUST.read_text()
        m = re.search(r'const GOLDEN_SHA: &str = "([0-9a-f]{64})";', src)
        self.assertIsNotNone(m, "crates/nn/src/rowweights.rs no longer holds GOLDEN_SHA like this")
        b = npy_bytes(np.array([256, 512, 384, 1, 65535], dtype="<u2"))
        self.assertEqual(hashlib.sha256(b).hexdigest(), m.group(1))
        self.assertIn(f"pub const ROW_WEIGHT_ONE: u16 = {ROW_WEIGHT_ONE};", src)
        self.assertIn(f'pub const SUFFIX: &str = "{ROW_WEIGHTS_SUFFIX}";', src)

    def test_load_row_weights(self):
        with tempfile.TemporaryDirectory() as d:
            shard = os.path.join(d, "s.npy")
            self.assertIsNone(load_row_weights(shard, 3))
            side = shard + ROW_WEIGHTS_SUFFIX
            with open(side, "wb") as f:
                np.save(f, np.array([256, 512, 1], dtype="<u2"))
            self.assertEqual(load_row_weights(shard, 3).tolist(), [256, 512, 1])
            for bad, rows in ((np.array([256, 512, 1], dtype="<u2"), 4), (np.array([256, 0, 1], dtype="<u2"), 3),
                              (np.array([256, 512, 1], dtype="<u4"), 3), (np.array([256, 512, 1], dtype=">u2"), 3),
                              (np.array([1.0, 2.0, 1.0], dtype="<f4"), 3),
                              (np.array([[256], [512], [1]], dtype="<u2"), 3)):
                with open(side, "wb") as f:
                    np.save(f, bad)
                with self.assertRaises(ValueError, msg=f"{bad.dtype} {bad.shape} for {rows} rows"):
                    load_row_weights(shard, rows)


@unittest.skipIf(nn_tool() is None, "no nn-tool binary (cargo build --release -p nn)")
class ParityWithRust(unittest.TestCase):
    def tool(self, *args, ok=True):
        r = subprocess.run([nn_tool(), "row-weights", *args], capture_output=True, text=True)
        if ok:
            self.assertEqual(r.returncode, 0, r.stderr)
            return json.loads(r.stdout)
        self.assertEqual(r.returncode, 2, r.stdout)
        return r.stderr

    def test_nn_tool_writes_what_numpy_reads(self):
        with tempfile.TemporaryDirectory() as d:
            shard = empty_shard(os.path.join(d, "t4-harvest-s7.npy"), 9)
            self.assertEqual(self.tool("--shard", shard), {"shard": shard, "rows": 9, "sidecar": None})
            idx = os.path.join(d, "rows.npy")
            np.save(idx, np.array([5, 0, 5, 8], dtype="<u8"))
            got = self.tool("--shard", shard, "--weight", "2", "--rows", idx)
            want = np.full(9, 256, dtype="<u2")
            want[[0, 5, 8]] = 512
            side = shard + ROW_WEIGHTS_SUFFIX
            self.assertEqual(load_row_weights(shard, 9).tolist(), want.tolist())
            with open(side, "rb") as f:
                self.assertEqual(f.read(), npy_bytes(want), "numpy.save's bytes")
            summary = {"shard": shard, "rows": 9, "sidecar": side, "sha256": sha256_file(side), "rows_not_1": 3,
                       "sum": int(want.sum()), "min": 256, "max": 512}
            self.assertEqual(got, summary)
            self.assertEqual(self.tool("--shard", shard), summary, "the summary of the sidecar there")
            # A sidecar numpy wrote reads the same way (through a file object: given a name,
            # numpy.save would add ".npy" to it).
            with open(side, "wb") as f:
                np.save(f, np.array([1, 384, 65535, 256, 256, 256, 256, 256, 256], dtype="<u2"))
            s = self.tool("--shard", shard)
            self.assertEqual((s["rows_not_1"], s["min"], s["max"], s["sha256"]), (3, 1, 65535, sha256_file(side)))
            # Refused: a weight that is not a multiple of 1/256 or out of range, a row past the
            # end, two-dimensional rows, --rows alone, a sidecar of another length, no shard.
            for w in ("1.001", "0", "256", "-1", "x"):
                self.assertIn("--weight", self.tool("--shard", shard, "--weight", w, "--rows", idx, ok=False))
            np.save(idx, np.array([9], dtype="<u8"))
            self.assertIn("row 9 of a shard of 9 rows", self.tool("--shard", shard, "--weight", "2", "--rows", idx,
                                                                  ok=False))
            np.save(idx, np.array([[1, 2]], dtype="<u8"))
            self.assertIn("one-dimensional", self.tool("--shard", shard, "--weight", "2", "--rows", idx, ok=False))
            self.tool("--shard", shard, "--rows", idx, ok=False)
            with open(side, "wb") as f:
                np.save(f, np.full(4, 256, dtype="<u2"))
            self.assertIn("4 row weights for 9 rows", self.tool("--shard", shard, ok=False))
            self.tool("--shard", os.path.join(d, "none.npy"), ok=False)


if __name__ == "__main__":
    unittest.main()
