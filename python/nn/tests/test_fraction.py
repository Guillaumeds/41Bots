"""A fraction of the training rows (``fortyone_nn.train --train-fraction F``; owner, 2026-10-19: the
learning-curve test of whether more base rows help):

- ``prefix_rows`` is ceil(F x rows) on F as written, and the range (0, 1] is enforced;
- ``--train-fraction 1`` trains exactly the network of no flag (the same files, byte for byte);
- 0.5 trains, in the resident, staged and disk feeds and from streamed shards, exactly the network
  of the shards cut to their first ceil(rows / 2) rows with their row weights cut to match (the same
  float and integer tensors), never touching the validation shard; the disk feed writes only the
  prefix rows; the report gives the fraction and the rows in the files and used.
"""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:  # the numpy-only CI job
    raise unittest.SkipTest("PyTorch is not installed") from None

from fortyone_nn import train
from fortyone_nn.formats import load_shard, read_safetensors, sha256_file

from test_train import set_row_weights, synthetic_shard


def tensors(path):
    """The tensors of a safetensors file (its metadata left out: it names the files trained on)."""
    t, _ = read_safetensors(path)
    return {k: np.asarray(v) for k, v in t.items()}


class PrefixRowsTest(unittest.TestCase):
    def test_prefix_rows(self):
        self.assertEqual(train.prefix_rows(1.0, 7), 7)
        self.assertEqual(train.prefix_rows(0.5, 7), 4)
        self.assertEqual(train.prefix_rows(0.5, 8), 4)
        self.assertEqual(train.prefix_rows(0.5, 1), 1)
        self.assertEqual(train.prefix_rows(0.1, 10), 1, "0.1 as written, not the binary float above it")
        self.assertEqual(train.prefix_rows(0.3, 10), 3)
        self.assertEqual(train.prefix_rows(1e-9, 5), 1)
        self.assertEqual(train.prefix_rows(0.5, 25_600_000), 12_800_000)

    def test_out_of_range_is_refused(self):
        for f in (0.0, -0.5, 1.5, float("nan"), float("inf")):
            self.assertIsNotNone(train.check_fraction(f), f)
        for f in (1.0, 0.5, 1e-6):
            self.assertIsNone(train.check_fraction(f), f)
        with tempfile.TemporaryDirectory() as d:
            a = synthetic_shard(os.path.join(d, "a.npy"), 50, 1)
            v = synthetic_shard(os.path.join(d, "v.npy"), 20, 2)
            for f in ("0", "1.5", "-1", "nan"):
                with self.subTest(f=f), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    train.main(["--train", a, "--val", v, "--out", os.path.join(d, "o"), "--device", "cpu",
                                "--train-fraction", f])
            self.assertFalse(os.path.exists(os.path.join(d, "o")), "refused before anything is made")


class FractionTrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        d = cls.dir.name
        cls.rows = os.path.join(d, "rows")
        full = os.path.join(d, "full")
        cut = os.path.join(d, "cut")
        os.makedirs(full)
        os.makedirs(cut)
        # Odd row counts: the prefix is ceil(rows / 2). Shard a has a shard weight, b row weights.
        cls.full = [synthetic_shard(os.path.join(full, "a.npy"), 701, 1),
                    synthetic_shard(os.path.join(full, "b.npy"), 301, 2)]
        with open(cls.full[0] + ".weight", "w") as f:
            f.write("2")
        rw = np.full(301, 256, dtype="<u2")
        rw[1::3] = 768
        rw[::7] = 384
        set_row_weights(cls.full[1], rw)
        # The same shards cut by hand to their first 351 and 151 rows, the row weights too.
        cls.cut = []
        for p, k in zip(cls.full, (351, 151)):
            q = os.path.join(cut, os.path.basename(p))
            np.save(q, np.asarray(load_shard(p))[:k])
            cls.cut.append(q)
        shutil.copy(cls.full[0] + ".weight", cls.cut[0] + ".weight")
        set_row_weights(cls.cut[1], rw[:151])
        cls.val = synthetic_shard(os.path.join(d, "v.npy"), 200, 3)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def run_train(self, name, files, *extra):
        out = os.path.join(self.dir.name, name)
        with contextlib.redirect_stdout(io.StringIO()):
            train.main(["--train", *files, "--val", self.val, "--out", out, "--widths", "32,16", "--epochs", "3",
                        "--batch", "64", "--seed", "5", "--threads", "1", "--device", "cpu", "--rows-dir", self.rows,
                        "--prefetch-threads", "3", "--prefetch-depth", "4", *extra])
        with open(os.path.join(out, "report.json")) as f:
            rep = json.load(f)
        return out, rep

    def test_one_is_every_row_byte_for_byte(self):
        a, rep_a = self.run_train("plain", self.full, "--feed", "resident")
        b, rep_b = self.run_train("one", self.full, "--feed", "resident", "--train-fraction", "1")
        for f in ("float.safetensors", "int.safetensors"):
            self.assertEqual(sha256_file(os.path.join(b, f)), sha256_file(os.path.join(a, f)), f)
        self.assertEqual(rep_b["data"]["train_fraction"], 1.0)
        self.assertEqual(rep_b["data"]["train_rows_used"], [701, 301])
        self.assertEqual(rep_b["data"]["train_rows_in_files"], [701, 301])
        self.assertEqual([h["val_loss"] for h in rep_b["history"]], [h["val_loss"] for h in rep_a["history"]])

    def test_half_is_the_prefix_in_every_feed(self):
        ref, rep_ref = self.run_train("cut-resident", self.cut, "--feed", "resident")
        want = (tensors(os.path.join(ref, "float.safetensors")), tensors(os.path.join(ref, "int.safetensors")))
        for feed in ("resident", "staged", "disk"):
            with self.subTest(feed=feed):
                out, rep = self.run_train(f"half-{feed}", self.full, "--feed", feed, "--train-fraction", "0.5")
                got = (tensors(os.path.join(out, "float.safetensors")), tensors(os.path.join(out, "int.safetensors")))
                for w, g in zip(want, got):
                    self.assertEqual(sorted(w), sorted(g))
                    for k in w:
                        np.testing.assert_array_equal(g[k], w[k], err_msg=f"{feed} {k}")
                self.assertEqual([h["val_loss"] for h in rep["history"]],
                                 [h["val_loss"] for h in rep_ref["history"]])
                d = rep["data"]
                self.assertEqual(d["train_fraction"], 0.5)
                self.assertEqual(d["train_rows_in_files"], [701, 301])
                self.assertEqual(d["train_rows_used"], [351, 151])
                self.assertEqual(d["train_weights"], [2.0, 1.0])
                self.assertEqual(d["train_row_weights"][1]["rows"], 151, "the row weights cut to the prefix")
                self.assertEqual(rep["device"]["feed"]["mode"], feed)
                self.assertEqual(rep["device"]["feed"]["rows"], 502)
                if feed == "disk":
                    self.assertEqual(rep["device"]["feed"]["bytes"], rep_ref["device"]["feed"]["bytes"],
                                     "only the prefix rows are written")
                _, meta = read_safetensors(os.path.join(out, "int.safetensors"))
                self.assertEqual(meta["train_rows"], "502")
                self.assertEqual(meta["train_fraction"], "0.5")
                self.assertEqual(json.loads(meta["train_sha256"]), [sha256_file(p) for p in self.full])
                self.assertEqual(d["val"], sha256_file(self.val), "the validation shard is whole")

    def test_half_of_streamed_shards(self):
        ref, _ = self.run_train("cut-ref", self.cut, "--feed", "resident")
        for feed in ("resident", "disk"):
            with self.subTest(feed=feed):
                w = os.path.join(self.dir.name, f"watch-{feed}")
                os.makedirs(w)
                for p in self.full:
                    for suffix in (".weight", ".rowweights", ""):
                        if os.path.exists(p + suffix):
                            shutil.copy(p + suffix, os.path.join(w, os.path.basename(p) + suffix))
                out = os.path.join(self.dir.name, f"stream-{feed}")
                with contextlib.redirect_stdout(io.StringIO()):
                    train.main(["--watch-dir", w, "--expect", "2", "--val", self.val, "--out", out, "--widths",
                                "32,16", "--epochs", "3", "--batch", "64", "--seed", "5", "--threads", "1",
                                "--device", "cpu", "--watch-poll", "0.05", "--rows-dir", self.rows, "--feed", feed,
                                "--prefetch-threads", "3", "--prefetch-depth", "4", "--train-fraction", "0.5"])
                with open(os.path.join(out, "report.json")) as f:
                    rep = json.load(f)
                for f in ("float.safetensors", "int.safetensors"):
                    want, got = tensors(os.path.join(ref, f)), tensors(os.path.join(out, f))
                    for k in want:
                        np.testing.assert_array_equal(got[k], want[k], err_msg=f"{feed} {f} {k}")
                self.assertEqual([(p["rows"], p["rows_in_file"]) for p in rep["stream"]["parts"]],
                                 [(351, 701), (151, 301)])
                self.assertEqual(rep["data"]["train_rows_used"], [351, 151])
                self.assertEqual(rep["data"]["train_rows_in_files"], [701, 301])


if __name__ == "__main__":
    unittest.main()
