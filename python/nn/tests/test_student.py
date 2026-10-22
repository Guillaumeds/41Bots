"""The student network's Python side against Rust ``crates/student`` (docs/STUDENT.md section 5).

- parity (numpy only): on real encoded positions (``student fixture``) and a quantised random
  network, the Rust integer forward pass (``student eval``) and ``fortyone_nn.studentint`` give
  bit-identical logits, and the integer network computes the float one closely;
- training (needs PyTorch): a tiny student trains on fixture shards, its integer file loads in Rust
  with the same logits as Python's integer pass, the trainer refuses the test set as training data,
  and a warm start reports the keep rule.

Needs the ``student`` binary: ``$STUDENT_TOOL``, else ``target/release/student`` or
``target/debug/student``; skipped when there is none.

Run: ``cd python/nn && uv run python -m unittest discover -s tests -v``.
"""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fortyone_nn import intnet, studentint
from fortyone_nn.quantize import quantize

ROOT = Path(__file__).resolve().parents[3]


def student_tool():
    env = os.environ.get("STUDENT_TOOL")
    if env:
        return env
    for p in (ROOT / "target" / "release" / "student", ROOT / "target" / "debug" / "student"):
        if p.exists():
            return str(p)
    return None


TOOL = student_tool()


def fixture(d, name, count, seed, role="fixture"):
    p = os.path.join(d, name)
    subprocess.run([TOOL, "fixture", "--count", str(count), "--seed", str(seed), "--out", p, "--role", role],
                   check=True, capture_output=True)
    return p


def random_float_layers(n_features, widths, seed):
    rng = np.random.default_rng(seed)
    dims = [n_features, *widths, 4]
    out = []
    for k in range(len(dims) - 1):
        w = rng.normal(0.0, 1.0 / np.sqrt(dims[k] if k else 60), size=(dims[k], dims[k + 1]))
        b = rng.normal(0.0, 0.3, size=dims[k + 1])
        out.append((w, b))
    return out


def float_logits(layers, x_int):
    h = x_int.astype(np.float64) / (1 << 24)
    for k, (w, b) in enumerate(layers):
        h = h @ w + b
        if k < len(layers) - 1:
            h = np.clip(h, 0.0, 64.0)
    return h


@unittest.skipIf(TOOL is None, "no student binary (cargo build -p student)")
class Parity(unittest.TestCase):
    def test_rust_and_python_integer_logits_are_identical(self):
        with tempfile.TemporaryDirectory() as d:
            f = fixture(d, "f.npy", 60, 5)
            a, side = studentint.load_features(f)
            n_binary, n_features = side["n_binary"], side["n_binary"] + side["n_dense"]
            c = studentint.Columns(a, n_binary)
            fl = random_float_layers(n_features, (32, 16), 3)
            il = quantize(fl, n_features=n_features)
            net = os.path.join(d, "int.safetensors")
            studentint.write_student_int(net, il, {"note": "random"}, side)
            out = os.path.join(d, "z.npy")
            subprocess.run([TOOL, "eval", "--net", net, "--features", f, "--out", out], check=True,
                           capture_output=True)
            rust = np.load(out)
            py = studentint.int_logits(il, c, n_binary, n_features)
            np.testing.assert_array_equal(rust, py)
            self.assertGreater(int(c.mask.sum()), 100)
            # The integer network computes the float one closely.
            x, dd, kk = studentint.int_inputs(c, n_binary, n_features, slice(0, len(c)))
            zf = float_logits(fl, x)
            zi = py[dd, kk].astype(np.float64) / (1 << 24)
            self.assertLess(np.abs(zf - zi).max(), 1e-3)
            # Money chances sum to 2^32.
            self.assertEqual(sum(intnet.money(py[0, 0])), 1 << 32)

    def test_a_file_for_another_encoder_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            f = fixture(d, "f.npy", 4, 6)
            _, side = studentint.load_features(f)
            n_features = side["n_binary"] + side["n_dense"]
            il = quantize(random_float_layers(n_features, (8,), 1), n_features=n_features)
            net = os.path.join(d, "int.safetensors")
            studentint.write_student_int(net, il, {}, dict(side, encoder_sha256="0" * 64))
            r = subprocess.run([TOOL, "eval", "--net", net, "--features", f, "--out", os.path.join(d, "z.npy")],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("encoder hash", r.stderr)


try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except ImportError:  # the numpy-only CI job
    HAVE_TORCH = False


@unittest.skipIf(TOOL is None or not HAVE_TORCH, "needs the student binary and PyTorch")
class Training(unittest.TestCase):
    def test_a_tiny_student_trains_exports_and_loads_in_rust(self):
        from fortyone_nn import student

        with tempfile.TemporaryDirectory() as d:
            tr = fixture(d, "train.npy", 120, 7)
            te = fixture(d, "test.npy", 30, 8, role="test")
            out = os.path.join(d, "s1")
            rep = student.main(["--train", tr, "--test", te, "--out", out, "--widths", "16,8", "--epochs", "3",
                                "--batch", "16", "--val-permille", "200", "--threads", "1"])
            self.assertTrue(os.path.exists(os.path.join(out, "int.safetensors")))
            self.assertEqual(len(rep["history"]), 3)
            self.assertIn("decision_loss_pct", rep["test"]["int"]["all"])
            # Rust loads the trained network and computes Python's integer logits.
            z = os.path.join(d, "z.npy")
            subprocess.run([TOOL, "eval", "--net", os.path.join(out, "int.safetensors"), "--features", te,
                            "--out", z], check=True, capture_output=True)
            a, side = studentint.load_features(te)
            c = studentint.Columns(a, side["n_binary"])
            n_features = side["n_binary"] + side["n_dense"]
            from fortyone_nn.formats import read_safetensors

            t, meta = read_safetensors(os.path.join(out, "int.safetensors"))
            layers = [(t[f"w{k}"], t[f"b{k}"], s) for k, s in enumerate(json.loads(meta["shifts"]))]
            np.testing.assert_array_equal(np.load(z), studentint.int_logits(layers, c, side["n_binary"], n_features))
            # A warm start reports the keep rule.
            out2 = os.path.join(d, "s2")
            rep2 = student.main(["--train", tr, "--test", te, "--out", out2, "--widths", "16,8", "--epochs", "1",
                                 "--batch", "16", "--val-permille", "200", "--threads", "1",
                                 "--init", os.path.join(out, "float.safetensors"), "--shrink-perturb", "0.8"])
            self.assertIn("keep", rep2["keep"])
            # The test set is never trained on.
            with self.assertRaises(SystemExit):
                student.main(["--train", te, "--out", os.path.join(d, "s3"), "--epochs", "1"])

    def test_limit_takes_a_prefix_and_progress_is_written(self):
        from fortyone_nn import student

        with tempfile.TemporaryDirectory() as d:
            a = fixture(d, "a.npy", 40, 11)
            b = fixture(d, "b.npy", 40, 12)
            prog = os.path.join(d, "p.json")
            rep = student.main(["--train", a, b, "--limit", "50", "--out", os.path.join(d, "l1"), "--widths", "8",
                                "--epochs", "2", "--batch", "8", "--val-permille", "200", "--threads", "1",
                                "--progress", prog])
            self.assertEqual(rep["data"]["decisions"], 50)
            self.assertEqual(rep["data"]["train_rows"] + rep["data"]["val_rows"], 50)
            with open(prog) as f:
                p = json.load(f)
            self.assertEqual(p["step"], "done")
            self.assertEqual(p["int_sha256"], rep["int_sha256"])
            # The whole of the first shard as a limit is the first shard alone.
            r1 = student.main(["--train", a, b, "--limit", "40", "--out", os.path.join(d, "l2"), "--widths", "8",
                               "--epochs", "1", "--batch", "8", "--val-permille", "200", "--threads", "1"])
            r2 = student.main(["--train", a, "--out", os.path.join(d, "l3"), "--widths", "8", "--epochs", "1",
                               "--batch", "8", "--val-permille", "200", "--threads", "1"])
            self.assertEqual(r1["history"][0]["train_loss"], r2["history"][0]["train_loss"])
            with self.assertRaises(SystemExit):
                student.main(["--train", a, "--limit", "41", "--out", os.path.join(d, "l4"), "--epochs", "1"])

    def test_training_is_repeatable(self):
        from fortyone_nn import student
        from fortyone_nn.formats import sha256_file

        with tempfile.TemporaryDirectory() as d:
            tr = fixture(d, "train.npy", 60, 9)
            shas = []
            for k in range(2):
                out = os.path.join(d, f"r{k}")
                student.main(["--train", tr, "--out", out, "--widths", "8", "--epochs", "2", "--batch", "8",
                              "--val-permille", "200", "--threads", "1"])
                shas.append(sha256_file(os.path.join(out, "int.safetensors")))
            self.assertEqual(shas[0], shas[1])

    def test_rank_term_matches_a_direct_kl_and_zero_weight_changes_nothing(self):
        import math

        import torch

        from fortyone_nn import student
        from fortyone_nn.formats import sha256_file

        # Two decisions, 3 and 1 candidates; the KL is computed here one decision at a time.
        z = torch.tensor([[0.3, 0.1, -0.2, 0.0], [1.0, -1.0, 0.5, 0.2], [0.0, 0.4, 0.1, -0.3], [0.7, 0.2, 0.1, 0.0]],
                         dtype=torch.float64)
        tgt = torch.tensor([[0.40, 0.20, 0.30, 0.10], [0.35, 0.25, 0.25, 0.15], [0.45, 0.15, 0.2, 0.2],
                            [0.5, 0.2, 0.2, 0.1]], dtype=torch.float64)
        dec = torch.tensor([0, 0, 0, 1])
        T = 0.05
        got = float(student.rank_term(z, tgt, dec, torch.tensor([3.0, 1.0]), T))
        vs = torch.softmax(z, dim=1)[:, 0].numpy() / T
        vt = tgt[:, 0].numpy() / T
        kl = []
        for rows in ([0, 1, 2], [3]):
            ls = vs[rows] - math.log(np.exp(vs[rows]).sum())
            lt = vt[rows] - math.log(np.exp(vt[rows]).sum())
            kl.append(float((np.exp(lt) * (lt - ls)).sum()))
        self.assertAlmostEqual(got, sum(kl) / 2, places=9)
        self.assertEqual(kl[1], 0.0)  # one candidate: nothing to rank
        with tempfile.TemporaryDirectory() as d:
            tr = fixture(d, "train.npy", 60, 9)
            shas = []
            for k, extra in enumerate(([], ["--rank-weight", "0"], ["--rank-weight", "1", "--stop-on", "decision"])):
                out = os.path.join(d, f"w{k}")
                student.main(["--train", tr, "--out", out, "--widths", "8", "--epochs", "2", "--batch", "8",
                              "--val-permille", "200", "--threads", "1", *extra])
                shas.append(sha256_file(os.path.join(out, "int.safetensors")))
            self.assertEqual(shas[0], shas[1])  # weight 0 is the old loss, bit for bit
            self.assertNotEqual(shas[0], shas[2])


if __name__ == "__main__":
    unittest.main()
