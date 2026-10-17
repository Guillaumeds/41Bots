"""Additive training: the trainer's ``--init`` (``fortyone_nn.train``; docs/m3-hybrid/audit-loop.md §10).

Needs PyTorch (``cd python/nn && uv run python -m unittest discover -s tests -v``); skipped where only
numpy is installed (the CI's parity job).

- the float layers round-trip through ``set_layers`` exactly (``Net.float_layers`` of the loaded
  network are the stored ones);
- a network of another shape, head or trick, an integer network, and an ``--init-int`` the float
  file does not quantise to are refused, the shape and head before any data is read;
- ``--epochs 0`` takes the network as it is: its float and integer files byte for byte, measured on
  the validation shard like the training that made it;
- a continued training (with and without shrink and perturb) repeats bit for bit for a seed, the
  perturbation moves with its seed, and the network's metadata and the report record the start;
- the toy case of catastrophic forgetting: continued on old and new data together, a network keeps
  what it knew of the old data; continued on the new data alone, it loses more of it.
"""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:  # the numpy-only CI job
    raise unittest.SkipTest("PyTorch is not installed") from None

from fortyone_nn import train
from fortyone_nn.formats import N_BINARY, read_safetensors, sha256_file

from test_train import synthetic_shard


def quiet_main(argv):
    with contextlib.redirect_stdout(io.StringIO()):
        train.main(argv)


class InitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        d = cls.dir.name
        cls.a = synthetic_shard(os.path.join(d, "a.npy"), 192, seed=11)
        cls.b = synthetic_shard(os.path.join(d, "b.npy"), 192, seed=12)
        cls.val = synthetic_shard(os.path.join(d, "v.npy"), 64, seed=13)
        cls.base = os.path.join(d, "base")
        quiet_main(["--train", cls.a, "--val", cls.val, "--out", cls.base, "--widths", "24,12", "--epochs", "2",
                    "--batch", "64", "--seed", "3", "--threads", "1"])
        cls.float = os.path.join(cls.base, "float.safetensors")
        cls.int = os.path.join(cls.base, "int.safetensors")

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def run_train(self, name, *extra, train_files=None, widths="24,12"):
        out = os.path.join(self.dir.name, name)
        quiet_main(["--train", *(train_files or [self.a, self.b]), "--val", self.val, "--out", out, "--widths", widths,
                    "--batch", "64", "--seed", "3", "--threads", "1", *extra])
        with open(os.path.join(out, "report.json")) as f:
            return out, json.load(f)

    def test_layers_round_trip(self):
        layers, meta, n_out = train.read_init(self.float, [24, 12], "outcome")
        net = train.Net([24, 12], n_out)
        train.set_layers(net, layers)
        for (w, b), (w2, b2) in zip(layers, net.float_layers()):
            self.assertTrue(np.array_equal(w, w2) and np.array_equal(b, b2))
        self.assertTrue(torch.equal(net.emb.weight[N_BINARY], torch.zeros(24)), "the padding row stays 0")
        self.assertEqual(train.requantised_sha256(layers, meta), sha256_file(self.int),
                         "the float file quantises to its integer network, byte for byte")

    def test_mismatches_are_refused(self):
        def refused(*extra, widths="24,12", head="outcome"):
            with mock.patch.object(train, "load_shard", side_effect=AssertionError("read a shard")):
                with self.assertRaises(SystemExit) as e:
                    quiet_main(["--train", self.a, "--val", self.val, "--out", os.path.join(self.dir.name, "no"),
                                "--widths", widths, "--head", head, "--epochs", "1", *extra])
            self.assertFalse(os.path.exists(os.path.join(self.dir.name, "no", "int.safetensors")))
            return str(e.exception)

        self.assertIn("hidden widths are 24,12, but --widths is 32,12", refused("--init", self.float, widths="32,12"))
        self.assertIn("hidden widths are 24,12, but --widths is 24,12,8", refused("--init", self.float,
                                                                                widths="24,12,8"))
        self.assertIn("head is 'outcome-v1'", refused("--init", self.float, head="money4"))
        self.assertIn("not a float network", refused("--init", self.int))
        other = os.path.join(self.dir.name, "other-int.safetensors")
        shutil.copyfile(self.int, other)
        with open(other, "ab") as f:
            f.write(b" ")
        self.assertIn("not that network's weights", refused("--init", self.float, "--init-int", other))
        # Another trick: refused once the data's trick is known (before any training).
        t3 = synthetic_shard(os.path.join(self.dir.name, "t3.npy"), 64, seed=14, trick=3)
        with self.assertRaises(SystemExit) as e:
            quiet_main(["--train", t3, "--val", t3, "--out", os.path.join(self.dir.name, "t3"), "--widths", "24,12",
                        "--epochs", "1", "--init", self.float])
        self.assertIn("trained for trick 4 positions, but the data is at trick 3", str(e.exception))
        for argv in (["--epochs", "0"], ["--init-int", self.int], ["--shrink-perturb", "0.9"],
                     ["--init", self.float, "--epochs", "0", "--shrink-perturb", "0.9"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                quiet_main(["--train", self.a, "--val", self.val, "--out", os.path.join(self.dir.name, "x"),
                            "--widths", "24,12", *argv])
        with self.assertRaises(SystemExit) as e:
            self.run_train("lam", "--init", self.float, "--shrink-perturb", "1.0", "--epochs", "1")
        self.assertIn("lambda must be above 0 and below 1", str(e.exception))

    def test_check_init_only(self):
        """``--check-init-only``: the start checked (shape, head, the quantisation proof) without
        any data; a mismatch stops it."""
        out = os.path.join(self.dir.name, "check")
        with mock.patch.object(train, "load_shard", side_effect=AssertionError("read a shard")):
            quiet_main(["--out", out, "--widths", "24,12", "--init", self.float, "--init-int", self.int,
                        "--check-init-only"])
            with open(os.path.join(out, "init-check.json")) as f:
                rec = json.load(f)
            self.assertEqual((rec["int_sha256"], rec["outputs"], rec["inputs"]), (sha256_file(self.int), 14, 909))
            other = os.path.join(self.dir.name, "other2.safetensors")
            shutil.copyfile(self.int, other)
            with open(other, "ab") as f:
                f.write(b" ")
            for argv, want in ((["--init-int", other, "--widths", "24,12"], "not that network's weights"),
                               (["--init-int", self.int, "--widths", "24,8"], "hidden widths are 24,12")):
                with self.assertRaises(SystemExit) as e:
                    quiet_main(["--out", out, "--init", self.float, "--check-init-only", *argv])
                self.assertIn(want, str(e.exception))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            quiet_main(["--out", out, "--widths", "24,12", "--init", self.float, "--check-init-only"])

    def test_epochs_zero_takes_the_network_as_it_is(self):
        with mock.patch.object(train, "load_compact", side_effect=AssertionError("read the training rows")):
            out, rep = self.run_train("as-is", "--init", self.float, "--init-int", self.int, "--epochs", "0")
        for f in ("float.safetensors", "int.safetensors"):
            with open(os.path.join(out, f), "rb") as x, open(os.path.join(self.base, f), "rb") as y:
                self.assertEqual(x.read(), y.read(), f)
        with open(os.path.join(self.base, "report.json")) as f:
            base = json.load(f)
        self.assertEqual(rep["float"]["val_loss"], base["float"]["val_loss"], "the same network, the same measure")
        self.assertEqual(rep["int"]["error"], base["int"]["error"])
        self.assertEqual(rep["int"]["sha256"], sha256_file(self.int))
        self.assertEqual((rep["history"], rep["best_epoch"]), ([], -1))
        self.assertTrue(rep["init"]["evaluated_only"])
        self.assertEqual(rep["init"]["int_sha256"], sha256_file(self.int))

    def test_continued_training_repeats_and_records_its_start(self):
        a, rep = self.run_train("c1", "--init", self.float, "--init-int", self.int, "--epochs", "2", "--lr", "3e-4")
        b, _ = self.run_train("c2", "--init", self.float, "--init-int", self.int, "--epochs", "2", "--lr", "3e-4")
        sha = lambda d: sha256_file(os.path.join(d, "int.safetensors"))  # noqa: E731
        self.assertEqual(sha(a), sha(b), "the same seed and data: the same network")
        self.assertNotEqual(sha(a), sha256_file(self.int), "it trained")
        _, meta = read_safetensors(os.path.join(a, "int.safetensors"))
        self.assertEqual(meta["init_float_sha256"], sha256_file(self.float))
        self.assertEqual(meta["init_int_sha256"], sha256_file(self.int))
        self.assertNotIn("shrink_perturb", meta)
        self.assertEqual(rep["init"]["float_sha256"], sha256_file(self.float))
        self.assertFalse(rep["init"]["evaluated_only"])
        # Shrink and perturb: seeded, so repeatable; another seed, another network.
        sp = ("--init", self.float, "--epochs", "1", "--shrink-perturb", "0.9")
        p1, rep_p = self.run_train("p1", *sp, "--perturb-seed", "7")
        p2, _ = self.run_train("p2", *sp, "--perturb-seed", "7")
        p3, _ = self.run_train("p3", *sp, "--perturb-seed", "8")
        self.assertEqual(sha(p1), sha(p2))
        self.assertNotEqual(sha(p1), sha(p3))
        _, meta = read_safetensors(os.path.join(p1, "int.safetensors"))
        self.assertEqual(json.loads(meta["shrink_perturb"]), {"lambda": 0.9, "seed": 7})
        s = rep_p["init"]["shrink_perturb"]
        self.assertEqual((s["lambda"], s["seed"]), (0.9, 7))
        self.assertAlmostEqual(s["sigma"]["hidden.0.weight"], 0.1 / np.sqrt(3 * 24))
        self.assertAlmostEqual(s["sigma"]["emb.weight"], 0.1 / np.sqrt(3 * 909))

    def test_shrink_perturb_rule(self):
        torch.manual_seed(0)
        net = train.Net([8, 4], 14)
        before = {k: v.clone() for k, v in net.state_dict().items()}
        sig = train.shrink_perturb(net, 0.5, 21)
        net2 = train.Net([8, 4], 14)
        net2.load_state_dict(before)
        train.shrink_perturb(net2, 0.5, 21)
        for k, v in net.state_dict().items():
            self.assertTrue(torch.equal(v, net2.state_dict()[k]), k)
        self.assertTrue(torch.equal(net.emb.weight[N_BINARY], torch.zeros(8)))
        # Mostly the shrink: the noise is half a fresh initialisation's spread.
        d = (net.out.weight.detach() - 0.5 * before["out.weight"]).flatten()
        self.assertLess(abs(float(d.std()) - sig["out.weight"]), 0.5 * sig["out.weight"])


class ForgettingTest(unittest.TestCase):
    """The toy case: a network fitted to old data A, continued from its weights on the new data B
    alone, forgets A; continued on A and B mixed, it keeps A."""

    def test_old_and_new_data_together_keep_the_old(self):
        with tempfile.TemporaryDirectory() as d:
            a = synthetic_shard(os.path.join(d, "a.npy"), 256, seed=21)
            b = synthetic_shard(os.path.join(d, "b.npy"), 256, seed=22)

            def run(name, files, val, *extra):
                out = os.path.join(d, name)
                quiet_main(["--train", *files, "--val", val, "--out", out, "--widths", "64,32", "--batch", "32",
                            "--seed", "4", "--threads", "1", "--patience", "100", *extra])
                return os.path.join(out, "float.safetensors")

            def loss_on(net_path, shard):
                layers, _, n_out = train.read_init(net_path, [64, 32], "outcome")
                net = train.Net([64, 32], n_out)
                train.set_layers(net, layers)
                return train.evaluate(net, "outcome", train.tensors(np.asarray(train.load_shard(shard))), 1.0)[0]

            old = run("old", [a], a, "--epochs", "25", "--lr", "3e-3")
            start = loss_on(old, a)
            new_only = run("new-only", [b], b, "--init", old, "--epochs", "20", "--lr", "3e-3")
            mixed = run("mixed", [a, b], b, "--init", old, "--epochs", "20", "--lr", "3e-3")
            lost_new_only, lost_mixed = loss_on(new_only, a) - start, loss_on(mixed, a) - start
            # Measured: about 0.13 lost on B alone, 0.0004 mixed (from a loss of 1.21 on A).
            self.assertGreater(lost_new_only, 0.05, "trained on B alone, the network loses some of A")
            self.assertLess(lost_mixed, 0.25 * lost_new_only,
                            f"A and B together keep A: {lost_mixed:.4f} lost against {lost_new_only:.4f}")


if __name__ == "__main__":
    unittest.main()
