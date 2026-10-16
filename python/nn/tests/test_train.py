"""The trainer's device handling and data loading (``fortyone_nn.train``, ``fortyone_nn.device``).

Needs PyTorch (``cd python/nn && uv run python -m unittest discover -s tests -v``); skipped where
only numpy is installed (the CI's parity job). On a machine with a usable GPU the ``Gpu`` tests
also run: the CUDA-graph step against the eager step, and training on the GPU against the CPU.

- the GPU check's architecture rule (``arch_supported``) and its CPU answer without a GPU;
- the compact training columns built a block at a time from memory-mapped shards, expanded,
  equal the full-width columns of the concatenated record array (the trainer's former way), for
  any block size and number of threads, and an active index that is neither a feature nor the
  padding is refused;
- the shards checked a block at a time on threads give ``check_values``' verdict on each shard
  in turn (the same first error, from the first shard that has one) and the tricks they hold;
- the feed gives exactly the former batches (the same indices and tensors, in order, for a seed,
  over two epochs), both resident and staged;
- training on the CPU is bit-for-bit repeatable, ``--device auto`` without a usable GPU,
  ``--feed staged`` with a batch profile and ``--cuda-graph off`` give the same network as
  ``--device cpu``, and the progress file and the report record the device, the feed, the
  training step's mode and the profile;
- the CUDA-graph step's bookkeeping, without a GPU (``train.Replayed`` stands in for the graph:
  each replay runs the captured step again on the graph's input buffers): the same losses, loss
  sums and network as eager steps, bit for bit, over epochs with partial batches and a new
  learning rate (a new capture), for both feeds; a failed capture goes on eagerly; the trial
  passes a working graph and catches a broken one or a failed capture, without touching the
  network; the trainer's own graph path (the trial, the two profiles, the report and progress
  fields, ``--cuda-graph off``, a capture failing after the trial: the same network, eagerly);
- on a GPU (skipped here): the replayed CUDA graph against eager steps for about 300 steps (two
  new learning rates, both feeds): losses within 1e-5 (relative) with the same Adam, within 1e-3
  with the former Adam; and a training run reports the graph, its trial and the replayed
  batches' profile.
"""

import contextlib
import copy
import io
import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:  # the numpy-only CI job
    raise unittest.SkipTest("PyTorch is not installed") from None

from fortyone_nn import device, train
from fortyone_nn.formats import (
    MAX_ACTIVE, N_BINARY, N_DENSE, ONE, PAD, ROW_WEIGHTS_SUFFIX, VOCAB, check_values, load_shard, read_safetensors,
    shard_dtype, sha256_file,
)


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


class WidthGuardTest(unittest.TestCase):
    """``--widths`` over the integer inference's limit (formats.MAX_WIDTH, 1024) is refused before anything
    else happens: no shard is read, no GPU probed, no output directory made."""

    def main_args(self, d, widths, device="cpu"):
        return ["--train", "t.npy", "--val", "v.npy", "--out", os.path.join(d, "net"), "--widths", widths,
                "--device", device]

    def test_a_layer_over_the_limit_is_refused_first(self):
        with tempfile.TemporaryDirectory() as d:
            for widths, layer, units in (("2048,1024,256", 0, 2048), ("512,1025", 1, 1025)):
                with mock.patch.object(train, "load_shard", side_effect=AssertionError("read a shard")), \
                        mock.patch.object(train, "probe", side_effect=AssertionError("probed the GPU")):
                    with self.assertRaises(SystemExit) as e:
                        train.main(self.main_args(d, widths, device="auto"))
                msg = str(e.exception)
                self.assertTrue(msg.startswith(f"--widths {widths}: layer {layer}: {units} units, wider than 1024"), msg)
                self.assertIn("MAX_WIDTH", msg)
                self.assertFalse(os.path.exists(os.path.join(d, "net")))

    def test_malformed_widths_are_refused_first_too(self):
        with tempfile.TemporaryDirectory() as d:
            for text in ("1024,,128", "x", "0,4"):
                with mock.patch.object(train, "load_shard", side_effect=AssertionError("read a shard")):
                    with self.assertRaises(SystemExit) as e:
                        train.main(self.main_args(d, text))
                self.assertIn(f"--widths {text}:", str(e.exception))

    def test_a_width_at_the_limit_starts_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            sentinel = RuntimeError("stop: the guard let the run start")
            with mock.patch.object(train, "load_shard", side_effect=sentinel) as load:
                with self.assertRaises(RuntimeError) as e:
                    train.main(self.main_args(d, "1024,512,128"))
            self.assertIs(e.exception, sentinel)
            load.assert_called_once_with("t.npy")


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
        for block, threads in ((7, 1), (7, 4), (300, 1), (300, 3), (64, 8), (1 << 15, 2), (1 << 18, 1)):
            comp = train.load_compact([load_shard(p) for p in self.train], block=block, threads=threads)
            self.assertLess(sum(c.nbytes for c in comp), 0.6 * sum(x.numel() * x.element_size() for x in whole))
            got = train.expand(*(torch.from_numpy(c) for c in comp))
            for x, y in zip(whole, got):
                self.assertEqual((x.dtype, x.shape), (y.dtype, y.shape))
                self.assertTrue(torch.equal(x, y), (block, threads))
        idx = whole[0]
        self.assertEqual(int(idx.max()), N_BINARY, "padding becomes the zero row")
        bad = np.asarray(load_shard(self.train[0]))[:5].copy()
        bad["active"][3, 0] = N_BINARY
        with self.assertRaises(ValueError):
            train.compact(bad)
        shards = [load_shard(self.train[0]), bad]
        for threads in (1, 3):
            with self.assertRaises(ValueError):
                train.load_compact(shards, block=2, threads=threads)

    def test_checked_shards_by_blocks_give_check_values_verdicts(self):
        good = [np.asarray(load_shard(p)) for p in self.train]
        for block, threads in ((1 << 15, 1), (7, 1), (7, 4), (64, 3)):
            self.assertEqual(train.check_shards(good, block=block, threads=threads), {4})
        mixed = [good[0], good[1].copy()]
        mixed[1]["trick"][5] = 3
        self.assertEqual(train.check_shards(mixed, block=50, threads=2), {3, 4})

        # Faults placed in different blocks: the verdict is check_values' on the whole shard.
        for rows_at, faults in (([3], [2]), ([3, 250], [3, 0]), ([10, 400], [1, 2]), ([100, 600], [2, 1])):
            a = good[0][:700].copy()
            v = a["outcomes"].shape[1]
            for i, f in zip(rows_at, faults):
                if f == 0:
                    a["value"][i, 0] += np.uint64(1)
                elif f == 1:
                    a["label"][i] = v
                elif f == 2:
                    a["mask"][i] &= ~(np.uint64(1) << np.uint64(int(a["label"][i])))
                else:
                    a["label"][i] = (int(a["label"][i]) + 1) % v
            with self.assertRaises(ValueError) as want:
                check_values(a)
            for block, threads in ((1 << 15, 1), (64, 1), (64, 4), (7, 3)):
                for shards in ([a], [good[1], a], [a, good[1]]):
                    with self.assertRaises(ValueError) as got:
                        train.check_shards(shards, block=block, threads=threads)
                    self.assertEqual(str(got.exception), str(want.exception), (rows_at, faults, block, threads))
        # The first shard with a fault gives the error.
        b = good[1].copy()
        b["value"][0, 0] += np.uint64(1)
        c = good[0].copy()
        c["label"][699] = c["outcomes"].shape[1]
        for threads in (1, 4):
            with self.assertRaisesRegex(ValueError, "outside the vocabulary"):
                train.check_shards([c, b], block=64, threads=threads)
            with self.assertRaisesRegex(ValueError, "sum to 2"):
                train.check_shards([b, c], block=64, threads=threads)

    def test_feed_gives_the_former_batches(self):
        whole = self.whole()
        n = len(whole[0])
        cpu = torch.device("cpu")
        for batch in (64, 100):
            for mode in ("auto", "resident", "staged"):
                feed = train.Feed(train.load_compact([load_shard(p) for p in self.train]), cpu, batch, mode)
                self.assertEqual(feed.facts["mode"], "staged" if mode == "staged" else "resident")
                bufs = feed.inputs(batch)
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
                        if len(b) == batch:  # the CUDA-graph step's way, through fixed buffers
                            feed.put(b, bufs)
                            for x, y in zip(old, feed.read(bufs)):
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
        self.assertEqual(len(rep["seconds"]["ms_per_batch"]), 2)
        self.assertNotIn("profile", rep["seconds"], "no batch profile on the CPU by default")
        st = rep["device"]["train_step"]
        self.assertEqual((st["mode"], st["reason"], st["trial"]), ("eager", "the CPU", None))
        self.assertEqual((st["captures"], st["replays"], st["eager_steps"]), (0, 0, 32))
        self.assertEqual(prog["step_mode"], "eager (the CPU)")
        o, _, _ = self.run_train("graph-off", "--device", "cpu", "--cuda-graph", "off")
        self.assertEqual(o, a, "--cuda-graph off is the same eager step")
        s, rep_s, _ = self.run_train("staged", "--device", "cpu", "--feed", "staged", "--profile-batches", "3")
        self.assertEqual(s, a, "the staged feed, and profiling batches, train exactly the same network")
        self.assertEqual(rep_s["device"]["feed"]["mode"], "staged")
        p = rep_s["seconds"]["profile"]
        self.assertNotIn("error", p)
        self.assertEqual((p["batches"], p["first_batch"], p["step"]), (3, 8, "eager"))
        self.assertNotIn("graphed", p)
        self.assertGreater(p["host_ms_per_batch"], 0)
        self.assertTrue(any("embedding_bag" in r["op"] for r in p["top_host_ms_per_batch"]))
        self.assertEqual(rep["int"]["sha256"], a)
        self.assertEqual((prog["step"], prog["device"], prog["int_sha256"]), ("done", "cpu", a))
        self.assertEqual(len(rep["history"]), 2)
        # The validation error by race class: five classes, every validation row counted once.
        for key in ("float", "int"):
            by = rep[key]["error_by_race_class"]
            self.assertEqual([c["class"] for c in by], [0, 1, 2, 3, 4])
            self.assertEqual(sum(c["rows"] for c in by), rep[key].get("rows", 200))
            for c in by:
                self.assertEqual(c["sd"] is None, c["rows"] == 0)
        self.assertEqual(len(rep["history"][0]["val_sd_by_race_class"]), 5)
        if not torch.cuda.is_available():
            c, rep, _ = self.run_train("auto", "--device", "auto")
            self.assertEqual(c, a, "auto falls back to exactly the CPU training")
            self.assertEqual(rep["device"]["want"], "auto")
            self.assertTrue(rep["device"]["reason"])

    def test_the_trainers_graph_path_without_a_gpu(self):
        a, cpu, _ = self.run_train("cpu-ref2", "--device", "cpu")
        with mock.patch.object(train, "GRAPH_DEVICES", ("cpu",)), mock.patch.object(train, "GRAPH_CLASS", train.Replayed):
            g, rep, prog = self.run_train("cpu-graph", "--device", "cpu", "--profile-batches", "3")
            o, rep_o, _ = self.run_train("cpu-graph-off", "--device", "cpu", "--cuda-graph", "off")
        FailsAfterTrial.made = 0
        with mock.patch.object(train, "GRAPH_DEVICES", ("cpu",)), mock.patch.object(train, "GRAPH_CLASS", FailsAfterTrial):
            f, rep_f, prog_f = self.run_train("cpu-graph-fails", "--device", "cpu", "--profile-batches", "3")
        st = rep_f["device"]["train_step"]
        self.assertEqual(st["trial"]["max_rel_loss_difference"], 0.0)
        self.assertEqual(st["mode"], "eager")
        self.assertTrue(st["reason"].startswith("capturing the step failed: RuntimeError"), st["reason"])
        self.assertEqual(prog_f["step_mode"], f"eager ({st['reason']})")
        self.assertEqual((st["captures"], st["replays"], st["eager_steps"]), (0, 0, 32))
        self.assertEqual(f, g, "after a failed capture the same training goes on eagerly")
        self.assertEqual(o, a, "--cuda-graph off: the former step and Adam")
        self.assertEqual(rep_o["device"]["train_step"]["reason"], "--cuda-graph off")
        st = rep["device"]["train_step"]
        self.assertEqual((st["mode"], st["reason"], prog["step_mode"]), ("cuda-graph", None, "cuda-graph"))
        self.assertEqual(st["optimizer"], "Adam(fused=True, capturable=True)")
        self.assertEqual(st["trial"]["max_rel_loss_difference"], 0.0)
        # As on a GPU (test_gpu_training_follows_the_cpu): 15 full batches an epoch, 8 + 3 eager
        # (the profile), a capture at the 12th, the replays' profile from the 14th.
        self.assertEqual((st["captures"], st["replays"], st["eager_steps"]), (1, 15 - 11 + 15, 11 + 2))
        p = rep["seconds"]["profile"]
        self.assertEqual((p["step"], p["first_batch"], p["batches"]), ("eager", 8, 3))
        self.assertEqual((p["graphed"]["step"], p["graphed"]["first_batch"], p["graphed"]["batches"]), ("cuda-graph", 13, 2))
        for c, x in zip(cpu["history"], rep["history"]):
            self.assertAlmostEqual(c["val_loss"], x["val_loss"], delta=1e-4 * max(1.0, abs(c["val_loss"])))

    @unittest.skipUnless(torch.cuda.is_available(), "no GPU here")
    def test_gpu_training_follows_the_cpu(self):
        _, cpu, _ = self.run_train("cpu-ref", "--device", "cpu")
        # 15 full batches an epoch: 8 + 3 eager (the profile), a capture at the 12th, 3 batches
        # later the replays' profile (2 batches: the 16th is partial).
        _, gpu, prog = self.run_train("gpu", "--device", "cuda", "--profile-batches", "3")
        self.assertEqual((gpu["device"]["device"], prog["device"]), ("cuda", "cuda"))
        st = gpu["device"]["train_step"]
        self.assertEqual((st["mode"], st["reason"], prog["step_mode"]), ("cuda-graph", None, "cuda-graph"))
        self.assertLessEqual(st["trial"]["max_rel_loss_difference"], train.TRIAL_TOLERANCE)
        self.assertEqual((st["captures"], st["replays"], st["eager_steps"]), (1, 15 - 11 + 15, 11 + 2))
        p = gpu["seconds"]["profile"]
        self.assertEqual((p["step"], p["first_batch"], p["batches"]), ("eager", 8, 3))
        g = p["graphed"]
        self.assertNotIn("error", g)
        self.assertEqual((g["step"], g["first_batch"], g["batches"]), ("cuda-graph", 13, 2))
        self.assertGreater(g["host_ms_per_batch"], 0)
        _, eager, prog_e = self.run_train("gpu-eager", "--device", "cuda", "--cuda-graph", "off")
        self.assertEqual((eager["device"]["train_step"]["mode"], prog_e["step_mode"]),
                         ("eager", "eager (--cuda-graph off)"))
        for run in (gpu, eager):
            for c, g in zip(cpu["history"], run["history"]):
                self.assertAlmostEqual(c["val_loss"], g["val_loss"], delta=1e-3 * max(1.0, abs(c["val_loss"])))


class NoGraph:
    """A capture that fails, as on a driver without CUDA graphs."""

    def __init__(self, fn, stream):
        raise RuntimeError("operation not permitted when stream is capturing")


class FailsAfterTrial(train.Replayed):
    """Captures that work in the trial (its one capture) and fail afterwards."""

    made = 0

    def __init__(self, fn, stream):
        type(self).made += 1
        if type(self).made > 1:
            raise RuntimeError("capture failed on this driver")
        super().__init__(fn, stream)


class Idle(train.Replayed):
    """A broken graph: its replays compute nothing."""

    def replay(self):
        self.out = torch.zeros(())


def drive(step, n, epochs, cuts=(), seed=9):
    """Trains with ``step`` as the trainer does (a permutation an epoch, the loss sum on the
    device), the learning rate cut by 0.3 at the start of the epochs ``cuts``: (every step's loss,
    every epoch's loss sum)."""
    rng = np.random.default_rng(seed)
    losses, totals = [], []
    for epoch in range(epochs):
        if epoch in cuts:
            step.opt.param_groups[0]["lr"] *= 0.3
        order = step.feed.order(torch.from_numpy(rng.permutation(n)))
        step.total.zero_()
        for i in range(0, n, step.batch):
            step(order[i : i + step.batch])
            losses.append(float(step.loss))
        totals.append(float(step.total))
    return losses, totals


class StepCase(unittest.TestCase):
    """Training rows and a network for the training-step tests."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        d = cls.dir.name
        tr = [synthetic_shard(os.path.join(d, "a.npy"), 700, 1), synthetic_shard(os.path.join(d, "b.npy"), 300, 2)]
        cls.comp = train.load_compact([load_shard(p) for p in tr])
        cls.n = len(cls.comp[0])
        torch.manual_seed(4)
        cls.net = train.Net([32, 16], VOCAB[4])

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def stepper(self, feed, graphed, graph_class=train.Replayed, not_before=0, fused=True, on_fail=None):
        m = copy.deepcopy(self.net)
        return train.Step(m, train.adam(m.parameters(), 1e-3, fused), "outcome", 1.0, feed, 64,
                          graphed=graphed, not_before=not_before, graph_class=graph_class, on_fail=on_fail)

    def same_networks(self, a, b):
        for (k, x), y in zip(a.model.state_dict().items(), b.model.state_dict().values()):
            self.assertTrue(torch.equal(x, y), k)


class GraphStepTest(StepCase):
    """The CUDA-graph step's bookkeeping without a GPU (``train.Replayed`` for the graph)."""

    def test_replayed_steps_are_the_eager_steps(self):
        cpu = torch.device("cpu")
        for mode in ("resident", "staged"):
            feed = train.Feed(self.comp, cpu, 64, mode)
            eager = self.stepper(feed, False)
            graph = self.stepper(feed, True, not_before=5)
            e = drive(eager, self.n, 4, cuts=(2,))
            g = drive(graph, self.n, 4, cuts=(2,))
            self.assertEqual(e, g, mode)
            self.same_networks(eager, graph)
            # 1000 rows: 15 full batches and one of 40 an epoch. Epoch 0: 5 eager (not_before), a
            # capture, 10 replays, the partial batch; epoch 1: 15 replays; epoch 2 (a new rate):
            # 3 eager (warm-up), a capture, 12 replays; epoch 3: 15 replays.
            self.assertEqual(graph.counts, {"captures": 2, "replays": 10 + 15 + 12 + 15, "eager_steps": 6 + 1 + 4 + 1})
            self.assertEqual((graph.describe(), graph.graph_lr), ("cuda-graph", 1e-3 * 0.3))
            self.assertEqual(eager.counts, {"captures": 0, "replays": 0, "eager_steps": 64})

    def test_a_failed_capture_goes_on_eagerly(self):
        feed = train.Feed(self.comp, torch.device("cpu"), 64, "resident")
        eager = self.stepper(feed, False)
        told = []
        failed = self.stepper(feed, True, graph_class=NoGraph, on_fail=told.append)
        self.assertEqual(drive(eager, self.n, 2), drive(failed, self.n, 2))
        self.same_networks(eager, failed)
        self.assertFalse(failed.graphed)
        self.assertIn("operation not permitted", failed.reason)
        self.assertTrue(failed.describe().startswith("eager (capturing the step failed: RuntimeError"))
        self.assertEqual(told, [failed.reason])
        self.assertEqual(failed.counts, {"captures": 0, "replays": 0, "eager_steps": 32})

    def test_the_trial(self):
        cpu = torch.device("cpu")
        before = {k: x.clone() for k, x in self.net.state_dict().items()}
        for mode in ("resident", "staged"):
            feed = train.Feed(self.comp, cpu, 64, mode)
            why, facts = train.graph_trial(self.net, "outcome", 1.0, feed, 64, 1e-3, graph_class=train.Replayed)
            self.assertIsNone(why, facts)
            self.assertEqual((facts["steps"], facts["captures"], facts["replays"], facts["eager_steps"]), (6, 1, 3, 3))
            self.assertEqual(facts["max_rel_loss_difference"], 0.0)
            why, facts = train.graph_trial(self.net, "outcome", 1.0, feed, 64, 1e-3, graph_class=Idle)
            self.assertIn("differ from the eager steps'", why)
            why, _ = train.graph_trial(self.net, "outcome", 1.0, feed, 64, 1e-3, graph_class=NoGraph)
            self.assertIn("capturing the step failed", why)
        for k, x in self.net.state_dict().items():
            self.assertTrue(torch.equal(before[k], x), "the trial works on copies")


def drop(src, dst_dir):
    """Puts a copy of the shard ``src`` in ``dst_dir`` as ``search labels`` does: written under
    another name, then renamed."""
    dst = os.path.join(dst_dir, os.path.basename(src))
    with open(src, "rb") as f, open(dst + ".part", "wb") as g:
        g.write(f.read())
    os.replace(dst + ".part", dst)


class StreamTest(unittest.TestCase):
    """Streamed shards (``--watch-dir``, class ``Watcher``): with every shard there from the start,
    exactly the ``--train`` training; with shards appearing over time (here one after each partial
    epoch, ``--partial-epochs 1``), each joins between epochs, the partial epochs count for nothing
    (no patience, scheduler or best network), and the report says which epoch saw which shards."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        d = cls.dir.name
        cls.src = os.path.join(d, "src")
        os.makedirs(cls.src)
        cls.shards = [synthetic_shard(os.path.join(cls.src, n), r, s)
                      for n, r, s in (("a.npy", 700, 1), ("b.npy", 300, 2), ("c.npy", 500, 4))]
        cls.val = synthetic_shard(os.path.join(d, "v.npy"), 200, 3)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def run_train(self, name, *args):
        out = os.path.join(self.dir.name, name)
        prog = out + ".progress.json"
        with contextlib.redirect_stdout(io.StringIO()):
            train.main(["--val", self.val, "--out", out, "--widths", "32,16", "--batch", "64", "--seed", "5",
                        "--threads", "1", "--device", "cpu", "--progress", prog, "--watch-poll", "0.05", *args])
        with open(os.path.join(out, "report.json")) as f:
            return sha256_file(os.path.join(out, "int.safetensors")), json.load(f)

    def test_every_shard_there_from_the_start_is_the_plain_training(self):
        w = os.path.join(self.dir.name, "all")
        os.makedirs(w)
        for p in self.shards:
            drop(p, w)
        a, rep = self.run_train("plain", "--train", *self.shards, "--epochs", "3")
        b, srep = self.run_train("streamed-all", "--watch-dir", w, "--expect", "3", "--epochs", "3")
        self.assertEqual(a, b)
        st = srep["stream"]
        self.assertEqual((st["first_full_epoch"], st["partial_epochs"]), (0, 0))
        self.assertEqual([p["first_epoch"] for p in st["parts"]], [0, 0, 0])
        self.assertEqual(srep["data"]["train"], rep["data"]["train"])
        self.assertTrue(all(h["all_in"] for h in srep["history"]))
        self.assertNotIn("stream", rep)

    def streamed(self, name):
        """A streamed run where shard k + 1 appears once the trainer waits after its partial
        epoch on the first k."""
        w = os.path.join(self.dir.name, name + "-in")
        os.makedirs(w)
        drop(self.shards[0], w)
        prog = os.path.join(self.dir.name, name + ".progress.json")

        def feeder():
            for k in (1, 2):
                while True:
                    try:
                        with open(prog) as f:
                            p = json.load(f)
                        if p.get("step") == "wait" and p.get("shards_in") == k:
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(0.02)
                drop(self.shards[k], w)

        t = threading.Thread(target=feeder, daemon=True)
        t.start()
        out = self.run_train(name, "--watch-dir", w, "--expect", "3", "--partial-epochs", "1", "--epochs", "2",
                             "--lr", "0.01")
        t.join(timeout=5)
        return out

    def test_shards_join_between_epochs(self):
        a, rep = self.streamed("s1")
        b, _ = self.streamed("s2")
        self.assertEqual(a, b, "the same arrivals train the same network")
        h = rep["history"]
        self.assertEqual([(x["shards"], x["rows"], x["all_in"]) for x in h],
                         [(1, 700, False), (2, 1000, False), (3, 1500, True), (3, 1500, True)])
        self.assertEqual({x["lr"] for x in h}, {0.01}, "no scheduler step on partial epochs")
        st = rep["stream"]
        self.assertEqual([(p["name"], p["first_epoch"]) for p in st["parts"]], [("a.npy", 0), ("b.npy", 1), ("c.npy", 2)])
        self.assertEqual([p["weight"] for p in st["parts"]], [1.0, 1.0, 1.0])
        self.assertEqual((st["first_full_epoch"], st["partial_epochs"]), (2, 2))
        self.assertEqual(st["epochs_trained_on"], {"a.npy": 4, "b.npy": 3, "c.npy": 2})
        self.assertGreaterEqual(rep["best_epoch"], 2, "the best network has seen every shard")
        self.assertEqual(rep["data"]["train"], [sha256_file(p) for p in self.shards])
        self.assertEqual(len(rep["seconds"]["epochs"]), 4)

    def test_wrong_shards_are_refused(self):
        w = os.path.join(self.dir.name, "two")
        os.makedirs(w)
        for p in self.shards[:2]:
            drop(p, w)
        with self.assertRaises((SystemExit, ValueError)):
            self.run_train("too-many", "--watch-dir", w, "--expect", "1", "--epochs", "1")
        t5 = os.path.join(self.dir.name, "t5")
        os.makedirs(t5)
        synthetic_shard(os.path.join(t5, "x.npy"), 100, 9, trick=5)
        with self.assertRaises(SystemExit):
            self.run_train("other-trick", "--watch-dir", t5, "--expect", "1", "--epochs", "1")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.run_train("no-expect", "--watch-dir", t5, "--epochs", "1")


class WeightTest(unittest.TestCase):
    """Shard weights (``NAME.npy.weight``): the weighted mean of each batch's loss; a weight of 1
    everywhere is the unweighted training, byte for byte; a bad weight is refused."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        d = cls.dir.name
        cls.shards = [synthetic_shard(os.path.join(d, n), r, s) for n, r, s in (("a.npy", 300, 1), ("b.npy", 200, 2))]
        cls.val = synthetic_shard(os.path.join(d, "v.npy"), 100, 3)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def run_train(self, name, *args):
        out = os.path.join(self.dir.name, name)
        with contextlib.redirect_stdout(io.StringIO()):
            train.main(["--train", *self.shards, "--val", self.val, "--out", out, "--widths", "16,8", "--epochs", "2",
                        "--batch", "64", "--seed", "5", "--threads", "1", "--device", "cpu", *args])
        with open(os.path.join(out, "report.json")) as f:
            return sha256_file(os.path.join(out, "int.safetensors")), json.load(f)

    def set_weight(self, shard, text):
        p = shard + ".weight"
        if text is None:
            if os.path.exists(p):
                os.remove(p)
        else:
            with open(p, "w") as f:
                f.write(text)

    def test_weighted_loss(self):
        z = torch.randn(6, VOCAB[4])
        data = train.tensors(np.asarray(load_shard(self.shards[0]))[:6])
        _, _, tgt, lab, mask, vecs = data
        for head in ("outcome", "money4"):
            zz = z if head == "outcome" else z[:, :4]
            plain = train.loss_of(head, zz, tgt, lab, mask, vecs, 0.5)
            ones = train.loss_of(head, zz, tgt, lab, mask, vecs, 0.5, torch.ones(6))
            self.assertAlmostEqual(float(plain), float(ones), places=5)
            # Weight 2 on the first row is the mean over the rows with the first one twice.
            w = torch.tensor([2.0, 1, 1, 1, 1, 1])
            rows = [0, 0, 1, 2, 3, 4, 5]
            twice = train.loss_of(head, zz[rows], tgt[rows], lab[rows], mask[rows], vecs[rows], 0.5)
            self.assertAlmostEqual(float(train.loss_of(head, zz, tgt, lab, mask, vecs, 0.5, w)), float(twice), places=5)

    def test_shard_weights_in_training(self):
        plain, rep = self.run_train("plain")
        self.assertNotIn("train_weights", rep["data"])
        for s in self.shards:
            self.set_weight(s, "1\n")
        ones, rep1 = self.run_train("ones")
        self.assertEqual(ones, plain, "weights of 1 are the unweighted training")
        self.assertNotIn("train_weights", rep1["data"])
        self.set_weight(self.shards[1], "2")
        two, rep2 = self.run_train("two")
        self.assertNotEqual(two, plain)
        self.assertEqual(rep2["data"]["train_weights"], [1.0, 2.0])
        # The CUDA-graph step's way (without a GPU: ``Replayed``) reads the weights too.
        self.set_weight(self.shards[1], "2")
        with mock.patch.object(train, "GRAPH_DEVICES", ("cpu",)), mock.patch.object(train, "GRAPH_CLASS", train.Replayed):
            _, repg = self.run_train("two-graph")
        self.assertEqual(repg["device"]["train_step"]["mode"], "cuda-graph")
        for a, b in zip(rep2["history"], repg["history"]):
            self.assertAlmostEqual(a["val_loss"], b["val_loss"], delta=1e-4 * max(1.0, abs(a["val_loss"])))
        for bad in ("x", "0", "-1", "nan"):
            self.set_weight(self.shards[1], bad)
            with self.assertRaises(SystemExit):
                self.run_train("bad")
        for s in self.shards:
            self.set_weight(s, None)

    def test_a_weighted_shard_joins_a_streamed_training(self):
        w = os.path.join(self.dir.name, "watch")
        os.makedirs(w)
        drop(self.shards[0], w)
        with open(os.path.join(w, "b.npy.weight"), "w") as f:
            f.write("3")
        prog = os.path.join(self.dir.name, "w.progress.json")

        def feeder():
            while True:
                try:
                    with open(prog) as f:
                        if json.load(f).get("step") == "wait":
                            break
                except (OSError, ValueError):
                    pass
                time.sleep(0.02)
            drop(self.shards[1], w)

        t = threading.Thread(target=feeder, daemon=True)
        t.start()
        out = os.path.join(self.dir.name, "streamed-weighted")
        with contextlib.redirect_stdout(io.StringIO()):
            train.main(["--watch-dir", w, "--expect", "2", "--partial-epochs", "1", "--val", self.val, "--out", out,
                        "--widths", "16,8", "--epochs", "2", "--batch", "64", "--seed", "5", "--threads", "1",
                        "--device", "cpu", "--progress", prog, "--watch-poll", "0.05"])
        t.join(timeout=5)
        with open(os.path.join(out, "report.json")) as f:
            rep = json.load(f)
        self.assertEqual([(p["name"], p["weight"]) for p in rep["stream"]["parts"]], [("a.npy", 1.0), ("b.npy", 3.0)])
        self.assertEqual(rep["data"]["train_weights"], [1.0, 3.0])
        self.assertEqual(len(rep["history"]), 3)


def set_row_weights(shard, w):
    """Writes (or with None removes) the shard's row-weights sidecar, as a writer does: whole,
    under another name, then renamed (through a file object: given a name, numpy.save adds .npy)."""
    p = shard + ROW_WEIGHTS_SUFFIX
    if w is None:
        if os.path.exists(p):
            os.remove(p)
        return
    with open(p + ".part", "wb") as f:
        np.save(f, np.asarray(w))
    os.replace(p + ".part", p)


class RowWeightTest(unittest.TestCase):
    """Row weights (``NAME.npy.rowweights``): each row's weight times its shard's, the weighted mean
    of each batch's loss on the eager and the CUDA-graph step; all 256 is the unweighted network,
    byte for byte; all 512 is a shard weight of 2; recorded in the report and the network's
    metadata; a streamed shard brings its own; bad sidecars are refused."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        d = cls.dir.name
        cls.shards = [synthetic_shard(os.path.join(d, n), r, s) for n, r, s in (("a.npy", 300, 1), ("b.npy", 200, 2))]
        cls.val = synthetic_shard(os.path.join(d, "v.npy"), 100, 3)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def tearDown(self):
        for s in self.shards:
            set_row_weights(s, None)
            if os.path.exists(s + ".weight"):
                os.remove(s + ".weight")

    def run_train(self, name, *args):
        out = os.path.join(self.dir.name, name)
        with contextlib.redirect_stdout(io.StringIO()):
            train.main(["--train", *self.shards, "--val", self.val, "--out", out, "--widths", "16,8", "--epochs", "2",
                        "--batch", "64", "--seed", "5", "--threads", "1", "--device", "cpu", *args])
        with open(os.path.join(out, "report.json")) as f:
            rep = json.load(f)
        return sha256_file(os.path.join(out, "int.safetensors")), rep, out

    def test_the_weight_column(self):
        comp = [np.zeros(5)]
        self.assertIs(train.with_weights(comp, [(3, 1.0, None), (2, 1.0, np.full(2, 256, dtype="<u2"))]), comp)
        cols = train.with_weights(comp, [(3, 1.0, None), (2, 2.0, np.array([256, 640], dtype="<u2"))])
        self.assertEqual(cols[-1].dtype, np.float32)
        self.assertEqual(cols[-1].tolist(), [1.0, 1.0, 1.0, 2.0, 5.0], "the shard's weight times the row's")
        cols = train.with_weights(comp, [(3, 1.5, None), (2, 1.0, np.array([1, 65535], dtype="<u2"))])
        self.assertEqual(cols[-1].tolist(), [1.5, 1.5, 1.5, 1 / 256, 65535 / 256], "exact in float32")

    def test_row_weights_in_training(self):
        plain, rep, _ = self.run_train("plain")
        self.assertNotIn("train_row_weights", rep["data"])
        for s in self.shards:
            set_row_weights(s, np.full(300 if s.endswith("a.npy") else 200, 256, dtype="<u2"))
        ones, rep1, _ = self.run_train("ones")
        self.assertEqual(ones, plain, "row weights of 1 are the unweighted training, byte for byte")
        self.assertEqual([r["rows_not_1"] for r in rep1["data"]["train_row_weights"]], [0, 0])
        # All 512 on b is b's shard weight of 2: the same column, so the same network.
        set_row_weights(self.shards[0], None)
        set_row_weights(self.shards[1], np.full(200, 512, dtype="<u2"))
        sha_512 = sha256_file(self.shards[1] + ROW_WEIGHTS_SUFFIX)
        _, rep_rw, out_rw = self.run_train("rows-512")
        set_row_weights(self.shards[1], None)
        with open(self.shards[1] + ".weight", "w") as f:
            f.write("2")
        _, rep_sw, out_sw = self.run_train("shard-2")
        os.remove(self.shards[1] + ".weight")
        for f in ("int.safetensors", "float.safetensors"):
            a, meta_rw = read_safetensors(os.path.join(out_rw, f))
            b, _ = read_safetensors(os.path.join(out_sw, f))
            self.assertEqual(sorted(a), sorted(b))
            for k in a:
                self.assertTrue(np.array_equal(a[k], b[k]), f"{f} {k}")
        self.assertEqual([h["val_loss"] for h in rep_rw["history"]], [h["val_loss"] for h in rep_sw["history"]])
        self.assertEqual(json.loads(meta_rw["train_row_weights_sha256"]), [None, sha_512])
        # Some rows weighted: another network, recorded.
        wa = np.full(300, 256, dtype="<u2")
        wa[::3] = 512
        wb = np.full(200, 256, dtype="<u2")
        wb[:50] = 128
        set_row_weights(self.shards[0], wa)
        set_row_weights(self.shards[1], wb)
        some, rep2, out2 = self.run_train("some")
        self.assertNotEqual(some, plain)
        self.assertEqual(rep2["data"]["train_row_weights"], [
            {"file": "a.npy.rowweights", "sha256": sha256_file(self.shards[0] + ROW_WEIGHTS_SUFFIX), "rows": 300,
             "rows_not_1": 100, "mean": float(wa.mean() / 256)},
            {"file": "b.npy.rowweights", "sha256": sha256_file(self.shards[1] + ROW_WEIGHTS_SUFFIX), "rows": 200,
             "rows_not_1": 50, "mean": float(wb.mean() / 256)}])
        self.assertNotIn("train_weights", rep2["data"])
        _, meta = read_safetensors(os.path.join(out2, "int.safetensors"))
        self.assertEqual(json.loads(meta["train_row_weights_sha256"]),
                         [r["sha256"] for r in rep2["data"]["train_row_weights"]])
        # The CUDA-graph step's way (without a GPU: ``Replayed``) reads them too.
        with mock.patch.object(train, "GRAPH_DEVICES", ("cpu",)), mock.patch.object(train, "GRAPH_CLASS", train.Replayed):
            _, repg, _ = self.run_train("some-graph")
        self.assertEqual(repg["device"]["train_step"]["mode"], "cuda-graph")
        for a, b in zip(rep2["history"], repg["history"]):
            self.assertAlmostEqual(a["val_loss"], b["val_loss"], delta=1e-4 * max(1.0, abs(a["val_loss"])))
        # Refused: another length, a 0, another type.
        for bad in (np.full(199, 256, dtype="<u2"), np.where(np.arange(200) == 7, 0, 256).astype("<u2"),
                    np.full(200, 1.0, dtype="<f4")):
            set_row_weights(self.shards[1], bad)
            with self.assertRaises(SystemExit):
                self.run_train("bad")

    def test_a_streamed_shard_brings_its_row_weights(self):
        w = os.path.join(self.dir.name, "watch")
        os.makedirs(w)
        drop(self.shards[0], w)
        wb = np.full(200, 256, dtype="<u2")
        wb[1::2] = 768
        set_row_weights(os.path.join(w, "b.npy"), wb)
        prog = os.path.join(self.dir.name, "rw.progress.json")

        def feeder():
            while True:
                try:
                    with open(prog) as f:
                        if json.load(f).get("step") == "wait":
                            break
                except (OSError, ValueError):
                    pass
                time.sleep(0.02)
            drop(self.shards[1], w)

        t = threading.Thread(target=feeder, daemon=True)
        t.start()
        out = os.path.join(self.dir.name, "streamed-rows")
        with contextlib.redirect_stdout(io.StringIO()):
            train.main(["--watch-dir", w, "--expect", "2", "--partial-epochs", "1", "--val", self.val, "--out", out,
                        "--widths", "16,8", "--epochs", "2", "--batch", "64", "--seed", "5", "--threads", "1",
                        "--device", "cpu", "--progress", prog, "--watch-poll", "0.05"])
        t.join(timeout=5)
        with open(os.path.join(out, "report.json")) as f:
            rep = json.load(f)
        sha = sha256_file(os.path.join(w, "b.npy" + ROW_WEIGHTS_SUFFIX))
        self.assertEqual([(p["name"], p["row_weights_sha256"]) for p in rep["stream"]["parts"]],
                         [("a.npy", None), ("b.npy", sha)])
        rw = rep["data"]["train_row_weights"]
        self.assertIsNone(rw[0])
        self.assertEqual((rw[1]["rows_not_1"], rw[1]["mean"]), (100, 2.0))
        self.assertEqual(len(rep["history"]), 3)


@unittest.skipUnless(torch.cuda.is_available(), "no GPU here")
class GpuGraphStepTest(StepCase):
    """The real CUDA graph against eager steps on the GPU, about 300 steps."""

    def test_graphed_steps_follow_the_eager_steps(self):
        cuda = torch.device("cuda")
        device.deterministic(cuda)
        self.net = copy.deepcopy(type(self).net).to(cuda)
        for mode in ("resident", "staged"):
            feed = train.Feed(self.comp, cuda, 64, mode)
            # 19 epochs of 16 batches (304 steps), new learning rates at epochs 6 and 12.
            graph = self.stepper(feed, True, graph_class=train.CudaGraph)
            g, gt = drive(graph, self.n, 19, cuts=(6, 12))
            self.assertEqual((graph.describe(), graph.counts["captures"]), ("cuda-graph", 3))
            self.assertEqual(graph.counts["replays"], 19 * 15 - 3 * train.WARMUP)
            for name, fused, tol in (("eager, the same Adam", True, 1e-5), ("eager, the former Adam", False, 1e-3)):
                e, et = drive(self.stepper(feed, False, fused=fused), self.n, 19, cuts=(6, 12))
                for k, (x, y) in enumerate(zip(e + et, g + gt)):
                    self.assertLessEqual(abs(x - y), tol * abs(x), (mode, name, k, x, y))


if __name__ == "__main__":
    unittest.main()
