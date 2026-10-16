"""The trainer's device handling and data loading (``fortyone_nn.train``, ``fortyone_nn.device``).

Needs PyTorch (``cd python/nn && uv run python -m unittest discover -s tests -v``); skipped where
only numpy is installed (the CI's parity job). On a machine with a usable GPU the last test also
trains on it and compares with the CPU.

- the GPU check's architecture rule (``arch_supported``) and its CPU answer without a GPU;
- the compact training columns built a block at a time from memory-mapped shards, expanded,
  equal the full-width columns of the concatenated record array (the trainer's former way), for
  any block size, and an active index that is neither a feature nor the padding is refused;
- the feed gives exactly the former batches (the same indices and tensors, in order, for a seed,
  over two epochs), both resident and staged;
- training on the CPU is bit-for-bit repeatable, ``--device auto`` without a usable GPU and
  ``--feed staged`` with a batch profile give the same network as ``--device cpu``, and the
  progress file and the report record the device, the feed and the profile.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:  # the numpy-only CI job
    raise unittest.SkipTest("PyTorch is not installed") from None

from fortyone_nn import device, train
from fortyone_nn.formats import MAX_ACTIVE, N_BINARY, N_DENSE, ONE, PAD, VOCAB, load_shard, shard_dtype, sha256_file


def synthetic_shard(path, rows, seed, trick=4):
    """A label shard of random but valid rows: every row's label is a possible outcome whose
    money vector is the target, and every vector sums to 2^32."""
    v = VOCAB[trick]
    rng = np.random.default_rng(seed)
    a = np.zeros(rows, dtype=shard_dtype(v))
    act = np.full((rows, MAX_ACTIVE), PAD, dtype=np.uint16)
    for i in range(rows):
        k = rng.integers(10, 40)
        act[i, :k] = np.sort(rng.choice(N_BINARY, size=k, replace=False))
    a["active"] = act
    a["dense"] = rng.integers(-(1 << 24), 1 << 25, size=(rows, N_DENSE), dtype=np.int32)
    w = rng.integers(1, 1 << 20, size=(rows, v, 4)).astype(np.uint64)
    out = (w * np.uint64(ONE)) // w.sum(axis=2, keepdims=True, dtype=np.uint64)
    out[:, :, 0] += np.uint64(ONE) - out.sum(axis=2, dtype=np.uint64)
    a["outcomes"] = out
    lab = rng.integers(0, v, size=rows)
    a["label"] = lab
    a["mask"] = np.uint64((1 << v) - 1)
    a["value"] = out[np.arange(rows), lab]
    a["trick"] = trick
    np.save(path, a)
    return path


class DeviceTest(unittest.TestCase):
    def test_arch_rule(self):
        arch = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
        self.assertTrue(device.arch_supported((8, 6), arch))  # RTX A2000
        self.assertTrue(device.arch_supported((6, 1), arch))  # GTX 1080 Ti: sm_60 runs on 6.1
        self.assertTrue(device.arch_supported((8, 9), arch))  # Ada: sm_86 runs on 8.9
        self.assertFalse(device.arch_supported((12, 0), arch))  # Blackwell: no sm_12x, no PTX
        self.assertTrue(device.arch_supported((12, 0), arch + ["compute_90"]))  # PTX compiles on the spot
        self.assertFalse(device.arch_supported((8, 0), ["sm_86"]))  # a newer minor never runs on an older one
        self.assertTrue(device.arch_supported((9, 0), ["sm_90a"]))
        self.assertFalse(device.arch_supported((9, 1), ["sm_90a"]))  # arch-specific code: exactly 9.0

    def test_cpu_answers(self):
        f = device.probe("cpu")
        self.assertEqual((f["device"], f["want"]), ("cpu", "cpu"))
        self.assertEqual(f["torch"], torch.__version__)
        if not torch.cuda.is_available():
            f = device.probe("auto")
            self.assertEqual(f["device"], "cpu")
            self.assertFalse(f["usable_gpu"])
            self.assertTrue(f["reason"])
        with self.assertRaises(ValueError):
            device.probe("gpu")


class TrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        d = cls.dir.name
        cls.train = [synthetic_shard(os.path.join(d, "a.npy"), 700, 1), synthetic_shard(os.path.join(d, "b.npy"), 300, 2)]
        cls.val = synthetic_shard(os.path.join(d, "v.npy"), 200, 3)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def whole(self):
        """The former training columns: ``columns`` of the concatenated shards, as tensors."""
        return train.tensors(np.concatenate([np.asarray(load_shard(p)) for p in self.train]))

    def test_compact_columns_by_blocks_expand_to_the_concatenation(self):
        whole = self.whole()
        for block in (7, 300, 1 << 18):
            comp = train.load_compact([load_shard(p) for p in self.train], block=block)
            self.assertLess(sum(c.nbytes for c in comp), 0.6 * sum(x.numel() * x.element_size() for x in whole))
            got = train.expand(*(torch.from_numpy(c) for c in comp))
            for x, y in zip(whole, got):
                self.assertEqual((x.dtype, x.shape), (y.dtype, y.shape))
                self.assertTrue(torch.equal(x, y), block)
        idx = whole[0]
        self.assertEqual(int(idx.max()), N_BINARY, "padding becomes the zero row")
        bad = np.asarray(load_shard(self.train[0]))[:5].copy()
        bad["active"][3, 0] = N_BINARY
        with self.assertRaises(ValueError):
            train.compact(bad)

    def test_feed_gives_the_former_batches(self):
        whole = self.whole()
        n = len(whole[0])
        cpu = torch.device("cpu")
        for batch in (64, 100):
            for mode in ("auto", "resident", "staged"):
                feed = train.Feed(train.load_compact([load_shard(p) for p in self.train]), cpu, batch, mode)
                self.assertEqual(feed.facts["mode"], "staged" if mode == "staged" else "resident")
                rng = np.random.default_rng(5)
                for _ in range(2):
                    perm = torch.from_numpy(rng.permutation(n))
                    order = feed.order(perm)
                    for i in range(0, n, batch):
                        b_old = perm[i : i + batch]  # the former loop's indices and batch
                        old = [c[b_old] for c in whole]
                        b = order[i : i + batch]
                        self.assertTrue(torch.equal(b, b_old))
                        for x, y in zip(old, feed(b)):
                            self.assertEqual((x.dtype, x.shape), (y.dtype, y.shape))
                            self.assertTrue(torch.equal(x, y), (batch, mode, i))

    def run_train(self, name, *extra):
        out = os.path.join(self.dir.name, name)
        prog = out + ".progress.json"
        with contextlib.redirect_stdout(io.StringIO()):
            train.main(["--train", *self.train, "--val", self.val, "--out", out, "--widths", "32,16", "--epochs", "2",
                        "--batch", "64", "--seed", "5", "--threads", "1", "--progress", prog, *extra])
        with open(os.path.join(out, "report.json")) as f:
            report = json.load(f)
        with open(prog) as f:
            progress = json.load(f)
        return sha256_file(os.path.join(out, "int.safetensors")), report, progress

    def test_cpu_training_repeats_and_auto_without_gpu_is_the_cpu(self):
        a, rep, prog = self.run_train("cpu1", "--device", "cpu")
        b, _, _ = self.run_train("cpu2", "--device", "cpu")
        self.assertEqual(a, b)
        self.assertEqual(rep["device"]["device"], "cpu")
        self.assertEqual(rep["device"]["feed"]["mode"], "resident")
        self.assertEqual(len(rep["seconds"]["epochs"]), 2)
        self.assertNotIn("profile", rep["seconds"], "no batch profile on the CPU by default")
        s, rep_s, _ = self.run_train("staged", "--device", "cpu", "--feed", "staged", "--profile-batches", "3")
        self.assertEqual(s, a, "the staged feed, and profiling batches, train exactly the same network")
        self.assertEqual(rep_s["device"]["feed"]["mode"], "staged")
        p = rep_s["seconds"]["profile"]
        self.assertNotIn("error", p)
        self.assertEqual((p["batches"], p["first_batch"]), (3, 8))
        self.assertGreater(p["host_ms_per_batch"], 0)
        self.assertTrue(any("embedding_bag" in r["op"] for r in p["top_host_ms_per_batch"]))
        self.assertEqual(rep["int"]["sha256"], a)
        self.assertEqual((prog["step"], prog["device"], prog["int_sha256"]), ("done", "cpu", a))
        self.assertEqual(len(rep["history"]), 2)
        if not torch.cuda.is_available():
            c, rep, _ = self.run_train("auto", "--device", "auto")
            self.assertEqual(c, a, "auto falls back to exactly the CPU training")
            self.assertEqual(rep["device"]["want"], "auto")
            self.assertTrue(rep["device"]["reason"])

    @unittest.skipUnless(torch.cuda.is_available(), "no GPU here")
    def test_gpu_training_follows_the_cpu(self):
        _, cpu, _ = self.run_train("cpu-ref", "--device", "cpu")
        _, gpu, prog = self.run_train("gpu", "--device", "cuda")
        self.assertEqual((gpu["device"]["device"], prog["device"]), ("cuda", "cuda"))
        for c, g in zip(cpu["history"], gpu["history"]):
            self.assertAlmostEqual(c["val_loss"], g["val_loss"], delta=1e-3 * max(1.0, abs(c["val_loss"])))


if __name__ == "__main__":
    unittest.main()
