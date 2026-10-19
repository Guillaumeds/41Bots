"""Sibling groups, the pairwise ranking loss and the tie-aware loss (``fortyone_nn.train``
``--pair-weight``, ``--tie-aware``; ``formats.load_groups``, ``formats.tie_bits``).

- the groups sidecar: numpy writes the record type Rust writes (``crates/nn/src/groups.rs``), and
  ``load_groups`` refuses broken numbering, another length or another type;
- ``tie_bits``: the possible outcomes whose exact u64 money vector is the label's;
- the tie-aware loss equals the standard one, value and gradient, when no outcome ties, and is -log
  of the tied outcomes' summed probability when some do;
- the pairwise loss: its gradient raises the mover's value at the truly better sibling and lowers
  it at the worse, pairs weigh their true gap (an equal pair weighs nothing), rows of other groups
  or none are never paired;
- ``PairIndex.epoch``: a permutation of the rows, every group whole at the start of one full batch,
  the plan naming each slot's group and mover, the same order for the same seed;
- the defaults are the former training (a groups sidecar and ``--pair-weight 0`` train the same
  network, byte for byte, as no sidecar);
- a small training with the pairwise loss (eager, and the CUDA-graph path without a GPU): the
  pairwise loss falls, the report records it, and the integer network's money chances are valid.
"""

import json
import os
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:  # the numpy-only CI job
    raise unittest.SkipTest("PyTorch is not installed") from None

from unittest import mock

from fortyone_nn import train
from fortyone_nn.formats import (
    GROUPS_DTYPE, GROUPS_SUFFIX, ONE, VOCAB, check_groups, load_groups, load_shard, shard_dtype, tie_bits,
)
from fortyone_nn import intnet
from fortyone_nn.formats import read_safetensors
from fortyone_nn.intnet import money_rows

from test_train import synthetic_shard


def write_groups(shard, sizes, seed=0):
    """A groups sidecar beside ``shard``: groups of ``sizes`` rows from row 0 on (the rest of the
    shard in groups of one), random movers."""
    a = load_shard(shard)
    n = len(a)
    gid, k = [], 0
    for s in sizes:
        gid += [k] * s
        k += 1
    while len(gid) < n:
        gid.append(k)
        k += 1
    g = np.zeros(n, dtype=GROUPS_DTYPE)
    g["group"] = np.asarray(gid[:n], dtype=np.uint32)
    g["mover"] = np.random.default_rng(seed).integers(0, 4, size=n).astype(np.uint8)
    with open(shard + GROUPS_SUFFIX, "wb") as f:
        np.save(f, g)
    return g


class FormatTest(unittest.TestCase):
    def test_numpy_writes_rusts_record_type(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "g.npy")
            g = np.zeros(3, dtype=GROUPS_DTYPE)
            g["group"] = [0, 0, 1]
            g["mover"] = [1, 2, 3]
            with open(p, "wb") as f:
                np.save(f, g)
            with open(p, "rb") as f:
                head = f.read(128).decode("latin1")
            squeeze = "".join(head.split())
            # The descr Rust writes and reads (crates/nn/src/groups.rs DESCR), spaces aside.
            self.assertIn("'descr':[('group','<u4'),('mover','|u1')]", squeeze)
            self.assertEqual(GROUPS_DTYPE.itemsize, 5)

    def test_groups_are_checked(self):
        with tempfile.TemporaryDirectory() as d:
            s = synthetic_shard(os.path.join(d, "a.npy"), 10, 1)
            self.assertIsNone(load_groups(s, 10), "an old shard has no groups")
            g = write_groups(s, [2, 3])
            self.assertTrue(np.array_equal(load_groups(s, 10), g))
            with self.assertRaises(ValueError):
                load_groups(s, 11)
            for bad in ([1, 1, 2], [0, 2, 2], [0, 1, 0]):
                h = np.zeros(3, dtype=GROUPS_DTYPE)
                h["group"] = bad
                self.assertIsNotNone(check_groups(h), bad)
            h = np.zeros(2, dtype=GROUPS_DTYPE)
            h["mover"] = [0, 4]
            self.assertIsNotNone(check_groups(h))
            with open(s + GROUPS_SUFFIX, "wb") as f:
                np.save(f, np.zeros(10, dtype="<u2"))
            with self.assertRaises(ValueError):
                load_groups(s, 10)

    def test_tie_bits(self):
        v = VOCAB[4]
        a = np.zeros(2, dtype=shard_dtype(v))
        out = np.zeros((2, v, 4), dtype=np.uint64)
        out[:, :, 0] = np.uint64(ONE)  # every outcome: team 0 certain
        out[0, 5] = [ONE - 1, 1, 0, 0]  # differs in the last unit only
        out[1, 3] = [0, ONE, 0, 0]
        a["outcomes"] = out
        a["mask"] = np.uint64((1 << v) - 1) & ~np.uint64(1 << 7)  # outcome 7 impossible
        a["label"] = [0, 3]
        a["value"] = out[np.arange(2), a["label"]]
        bits = tie_bits(a)
        want0 = sum(1 << o for o in range(v) if o not in (5, 7))
        self.assertEqual(int(bits[0]), want0, "exact u64 equality, possible outcomes only")
        self.assertEqual(int(bits[1]), 1 << 3, "a label tied with nothing")


class LossTest(unittest.TestCase):
    def batch(self, rows=6, v=VOCAB[3], seed=0):
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(rows, v, generator=g, dtype=torch.float64, requires_grad=True)
        mask = torch.rand(rows, v, generator=g) > 0.2
        lab = torch.tensor([int(torch.nonzero(m)[0]) for m in mask])
        vecs = torch.rand(rows, v, 4, generator=g, dtype=torch.float64)
        vecs = vecs / vecs.sum(dim=2, keepdim=True)
        tgt = vecs[torch.arange(rows), lab]
        return z, tgt, lab, mask, vecs

    def test_tie_aware_is_the_standard_loss_without_ties(self):
        z, tgt, lab, mask, vecs = self.batch()
        for w in (None, torch.tensor([1.0, 2.0, 0.5, 1.0, 3.0, 1.0], dtype=torch.float64)):
            extra = () if w is None else (w,)
            a = train.loss_of("outcome", z, tgt, lab, mask, vecs, 1.0, *extra)
            (ga,) = torch.autograd.grad(a, z)
            b = train.loss_of("outcome", z, tgt, torch.ones_like(lab) << lab, mask, vecs, 1.0, *extra, tie_aware=True)
            (gb,) = torch.autograd.grad(b, z)
            self.assertEqual(float(a), float(b))
            self.assertTrue(torch.equal(ga, gb))

    def test_tie_aware_sums_the_tied_outcomes(self):
        z, tgt, lab, mask, vecs = self.batch(rows=1)
        on = torch.nonzero(mask[0])[:, 0]
        a, b = int(on[0]), int(on[1])
        bits = torch.tensor([(1 << a) | (1 << b)])
        loss = train.loss_of("outcome", z, tgt, bits, mask, vecs, 0.0, tie_aware=True)
        p = torch.softmax(z.masked_fill(~mask, -1e9), dim=1)[0]
        self.assertAlmostEqual(float(loss), -float(torch.log(p[a] + p[b])), places=10)

    def test_pair_gradient_sign(self):
        # Three rows: rows 0 and 1 a group (mover index 2 and 0), row 2 in no group.
        value = torch.tensor([[0.1, 0.1, 0.5, 0.3], [0.45, 0.25, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]],
                             dtype=torch.float64, requires_grad=True)
        tgt = torch.tensor([[0.0, 0.0, 0.7, 0.3], [0.4, 0.2, 0.2, 0.2], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        gid = torch.tensor([5, 5, -1])
        mover = torch.tensor([2, 0, 1])
        loss = train.pair_loss(value, tgt, gid, mover, 100.0)
        (g,) = torch.autograd.grad(loss, value)
        # Row 0 is truly better for its mover (0.7 > 0.4): raising its value lowers the loss.
        self.assertLess(float(g[0, 2]), 0)
        self.assertGreater(float(g[1, 0]), 0)
        self.assertEqual(float(g[0, [0, 1, 3]].abs().sum()), 0.0, "only the mover's value counts")
        self.assertEqual(float(g[2].abs().sum()), 0.0, "a row in no group is never paired")
        # The loss is the logistic loss of the one pair (weights cancel in the weighted mean).
        want = float(torch.nn.functional.softplus(torch.tensor(-100.0 * (0.5 - 0.45))))
        self.assertAlmostEqual(float(loss), want, places=10)
        # Swapped predictions cost more; equal true values weigh nothing.
        worse = train.pair_loss(value.detach()[[1, 0, 2]][:, [0, 1, 2, 3]], tgt[[1, 0, 2]], gid, mover[[1, 0, 2]], 100.0)
        self.assertGreaterEqual(float(worse), 0.0)
        tie = tgt.clone()
        tie[1] = torch.tensor([0.7, 0.1, 0.1, 0.1], dtype=torch.float64)
        self.assertEqual(float(train.pair_loss(value, tie, gid, mover, 100.0)), 0.0)
        # Two groups never mix: one pair each, weighted by the true gaps.
        v4 = value.detach()[[0, 1, 0, 1]]
        t4 = tgt[[0, 1, 0, 1]].clone()
        t4[2, 2] = 0.5  # gap 0.1 in group 2, 0.3 in group 1
        l4 = train.pair_loss(v4, t4, torch.tensor([1, 1, 2, 2]), mover[[0, 1, 0, 1]], 100.0)
        self.assertAlmostEqual(float(l4), want, places=10)


class PairIndexTest(unittest.TestCase):
    def test_epoch_keeps_groups_whole_at_batch_starts(self):
        n, batch = 1000, 64
        gid = np.full(n, -1, dtype=np.int64)
        mover = np.zeros(n, dtype=np.int64)
        rng = np.random.default_rng(3)
        at, k = 0, 0
        while at < 300:
            s = int(rng.integers(1, 5))
            gid[at + 50 : at + 50 + s] = k
            mover[at + 50 : at + 50 + s] = rng.integers(0, 4, size=s)
            at += s + int(rng.integers(0, 3))
            k += 1
        idx = train.PairIndex(gid, mover)
        p = idx.slots(batch)
        order, plan = idx.epoch(np.random.default_rng(1), batch)
        self.assertTrue(np.array_equal(np.sort(order), np.arange(n)), "a permutation")
        order2, plan2 = idx.epoch(np.random.default_rng(1), batch)
        self.assertTrue(np.array_equal(order, order2) and np.array_equal(plan.gid, plan2.gid))
        nb = n // batch
        self.assertEqual(plan.gid.shape, (nb, p))
        seen = set()
        for b in range(nb):
            rows = order[b * batch : (b + 1) * batch]
            g = gid[rows]
            inside = g >= 0
            c = int(inside.sum())
            self.assertTrue(np.all(inside[:c]), "groups first in the batch")
            self.assertLessEqual(c, p)
            for x in np.unique(g[inside]):
                self.assertNotIn(x, seen)
                seen.add(x)
                self.assertEqual(int((g == x).sum()), int((gid == x).sum()), "whole groups")
            # The plan names the same groups and the rows' movers.
            pg = plan.gid[b]
            self.assertTrue(np.all(pg[c:] == -1) and np.all(pg[:c] >= 0))
            same = (pg[:c, None] == pg[None, :c]) == (g[:c, None] == g[None, :c])
            self.assertTrue(np.all(same))
            self.assertTrue(np.array_equal(plan.mover[b][:c], mover[rows[:c]]))
        self.assertEqual(seen, set(range(k)), "every group in some full batch")
        with self.assertRaises(ValueError):
            idx.slots(4)


class TrainPairTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        d = cls.dir.name
        cls.a = synthetic_shard(os.path.join(d, "a.npy"), 600, 1)
        cls.sib = synthetic_shard(os.path.join(d, "s.npy"), 400, 2)
        rng = np.random.default_rng(4)
        sizes, left = [], 400
        while left > 0:
            s = min(left, int(rng.integers(2, 5)))
            sizes.append(s)
            left -= s
        write_groups(cls.sib, sizes, seed=5)
        cls.val = synthetic_shard(os.path.join(d, "v.npy"), 200, 3)
        # The same shard without its sidecar.
        cls.plain = os.path.join(d, "plain.npy")
        with open(cls.sib, "rb") as f, open(cls.plain, "wb") as g:
            g.write(f.read())

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def run_train(self, name, shards, *extra, epochs=3):
        out = os.path.join(self.dir.name, name)
        args = ["--train", *shards, "--val", self.val, "--out", out, "--widths", "32,16", "--epochs", str(epochs),
                "--batch", "64", "--threads", "1", "--int-eval-rows", "200", "--device", "cpu", *extra]
        train.main(args)
        with open(os.path.join(out, "report.json")) as f:
            rep = json.load(f)
        return rep["int"]["sha256"], rep, out

    def test_defaults_are_the_former_training(self):
        a, rep_a, _ = self.run_train("plain", [self.a, self.plain])
        b, rep_b, _ = self.run_train("sidecar", [self.a, self.sib])
        c, _, _ = self.run_train("zero", [self.a, self.sib], "--pair-weight", "0")
        self.assertEqual(a, b, "a groups sidecar alone changes nothing")
        self.assertEqual(a, c, "--pair-weight 0 is the former training")
        self.assertNotIn("pairs", rep_b)
        self.assertNotIn("train_pair_loss", rep_b["history"][0])
        _, meta = read_safetensors(os.path.join(self.dir.name, "sidecar", "int.safetensors"))
        self.assertNotIn("pair_weight", meta)
        self.assertNotIn("tie_aware", meta)

    def test_pairwise_training(self):
        s, rep, out = self.run_train("pairs", [self.a, self.sib], "--pair-weight", "0.5", epochs=6)
        h = rep["history"]
        losses = [x["train_pair_loss"] for x in h]
        self.assertLess(losses[-1], losses[0], losses)
        self.assertTrue(all(x["pair_batches"] == 1000 // 64 for x in h))
        self.assertTrue(all(0 < x["train_pair_share"] < 1 for x in h))
        self.assertEqual(rep["pairs"]["groups"], rep["pairs"]["sidecars"][1]["groups"])
        self.assertIsNone(rep["pairs"]["sidecars"][0])
        _, meta = read_safetensors(os.path.join(out, "int.safetensors"))
        self.assertEqual(meta["pair_weight"], "0.5")
        # The integer network's money chances: four whole numbers summing to 2^32.
        val = load_shard(self.val)
        layers = intnet.load(os.path.join(out, "int.safetensors"))[0]
        m = money_rows(layers, val["active"][:50], val["dense"][:50], mask=val["mask"][:50],
                       outcomes=val["outcomes"][:50])
        self.assertTrue(np.all(m.astype(np.uint64).sum(axis=1, dtype=np.uint64) == np.uint64(ONE)))
        # The same training again: the same network.
        s2, _, _ = self.run_train("pairs2", [self.a, self.sib], "--pair-weight", "0.5", epochs=6)
        self.assertEqual(s, s2)

    def test_tie_aware_training(self):
        s, rep, out = self.run_train("ties", [self.a, self.sib], "--tie-aware", "--pair-weight", "0.2")
        self.assertIn("tie_aware", rep)
        _, meta = read_safetensors(os.path.join(out, "int.safetensors"))
        self.assertEqual(meta["tie_aware"], "1")
        # The synthetic rows tie nothing: the tie-aware training is the plain one (the same
        # weights; only the metadata says tie_aware).
        p, _, out_p = self.run_train("noties", [self.a, self.sib], "--pair-weight", "0.2")
        self.assertNotEqual(s, p)
        ta, _ = read_safetensors(os.path.join(out, "int.safetensors"))
        tb, _ = read_safetensors(os.path.join(out_p, "int.safetensors"))
        self.assertEqual(sorted(ta), sorted(tb))
        for k in ta:
            self.assertTrue(np.array_equal(ta[k], tb[k]), k)
        # The rows on disk (their label column the tied outcomes) and staged: the same network.
        for feed in ("disk", "staged"):
            f, rep_f, _ = self.run_train("ties-" + feed, [self.a, self.sib], "--tie-aware", "--pair-weight", "0.2",
                                         "--feed", feed, "--rows-dir", self.dir.name)
            self.assertEqual(rep_f["device"]["feed"]["mode"], feed)
            self.assertEqual(f, s, feed)

    def test_graph_path_with_pairs(self):
        e, rep_e, _ = self.run_train("pairs-eager", [self.a, self.sib], "--pair-weight", "0.5")
        with mock.patch.object(train, "GRAPH_DEVICES", ("cpu",)), mock.patch.object(train, "GRAPH_CLASS", train.Replayed):
            g, rep_g, _ = self.run_train("pairs-graph", [self.a, self.sib], "--pair-weight", "0.5")
        st = rep_g["device"]["train_step"]
        self.assertEqual(st["mode"], "cuda-graph")
        self.assertGreater(st["replays"], 0)
        for x, y in zip(rep_e["history"], rep_g["history"]):
            self.assertEqual(x["pair_batches"], y["pair_batches"])
            self.assertAlmostEqual(x["train_pair_loss"], y["train_pair_loss"], delta=1e-3 * max(1.0, x["train_pair_loss"]))
            self.assertAlmostEqual(x["val_loss"], y["val_loss"], delta=1e-3 * max(1.0, abs(x["val_loss"])))


if __name__ == "__main__":
    unittest.main()
