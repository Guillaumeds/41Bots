"""The Python integer network against Rust ``crates/nn``.

- golden: the seeded golden network and inputs (same recipe on both sides) give the outputs in
  ``crates/nn/tests/data/intnet-golden.json``, which the Rust tests also check;
- quantisation: the integer network computes the float one closely, and safetensors round-trip;
  wide networks (2048 and 4096 units) of a trained network's scale keep all 24 fraction bits with
  the bound to spare, and weights too large for them get fewer, the bound holding exactly;
- parity: on real encoded positions (``nn-tool fixture``) and a quantised random network, the
  Rust integer forward pass (``nn-tool eval``) and this one give bit-identical money chances,
  also at widths 2048 (2048-1024-256) and 4096 (4096-512).
  Needs the ``nn-tool`` binary: ``$NN_TOOL``, else ``target/release/nn-tool`` or
  ``target/debug/nn-tool`` (CI builds it); skipped when there is none.

Run: ``cd python/nn && uv run python -m unittest discover -s tests -v``.
"""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fortyone_nn import intnet
from fortyone_nn.formats import N_FEATURES, VOCAB, check_values, load_shard, read_safetensors, write_safetensors
from fortyone_nn.quantize import quantize, write_int

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "crates" / "nn" / "tests" / "data" / "intnet-golden.json"


def nn_tool():
    env = os.environ.get("NN_TOOL")
    if env:
        return env
    for p in (ROOT / "target" / "release" / "nn-tool", ROOT / "target" / "debug" / "nn-tool"):
        if p.exists():
            return str(p)
    return None


def random_float_net(seed, widths=(64, 16), n_out=4):
    rng = np.random.default_rng(seed)
    dims = [N_FEATURES, *widths, n_out]
    layers = []
    for k in range(len(dims) - 1):
        w = rng.normal(0.0, 1.0 / np.sqrt(dims[k] if k else 60), size=(dims[k], dims[k + 1]))
        b = rng.normal(0.0, 0.3, size=dims[k + 1])
        layers.append((w, b))
    return layers


def float_logits(layers, x_int):
    h = x_int.astype(np.float64) / (1 << 24)
    for k, (w, b) in enumerate(layers):
        h = h @ w + b
        if k < len(layers) - 1:
            h = np.clip(h, 0.0, 64.0)
    return h


class Golden(unittest.TestCase):
    def test_golden_outputs(self):
        d = json.loads(GOLDEN.read_text())
        layers = intnet.golden_net(d["seed"])
        active, dense = intnet.golden_features(d["seed"] + 1, d["count"])
        got = intnet.money_rows(layers, active, dense)
        want = np.array(d["outputs"], dtype=np.uint64)
        np.testing.assert_array_equal(got, want)
        self.assertTrue(np.all(got.sum(axis=1, dtype=np.uint64) == np.uint64(1 << 32)))

    def test_golden_outcome_head(self):
        d = json.loads(GOLDEN.read_text())
        n_out = d["outcome"]["n_out"]
        layers = intnet.golden_net(d["seed"], n_out)
        active, dense = intnet.golden_features(d["seed"] + 1, d["count"])
        tables = intnet.golden_outcomes(d["seed"] + 2, d["count"], n_out)
        z = intnet.logits(layers, intnet.inputs(active, dense))
        got = [intnet.money_outcomes(z[i], m, vecs) for i, (m, vecs) in enumerate(tables)]
        self.assertEqual(got, d["outcome"]["outputs"])

    def test_softmax_edges(self):
        self.assertEqual(intnet.money([0, 0, 0, 0]), [1 << 30] * 4)
        self.assertEqual(intnet.money([1 << 40, -(1 << 40), 0, 0]), [1 << 32, 0, 0, 0])
        self.assertEqual(intnet.exp_weight(0), 1 << 62)
        self.assertEqual(intnet.R[1], 3260954456333195553)


class ExactProduct(unittest.TestCase):
    """``exact_matmul`` (float64 BLAS on pieces of the weights) is numpy's int64 product."""

    def test_pieces_give_the_int64_product(self):
        rng = np.random.default_rng(11)
        lo32, hi32 = -(1 << 31), (1 << 31) - 1
        # (rows, inputs, outputs, input range): layer 0 (binary 2^24, dense -2^24..0), hidden
        # layers (0..2^26), a wide layer, inputs too large for pieces (the int64 product, wrapping),
        # inputs that wrap int64 in the pieces, empty shapes.
        for n, k, m, lo, hi in ((64, N_FEATURES, 96, -(1 << 24), 1 << 24), (64, 1024, 64, 0, 1 << 26),
                                (16, 4096, 8, 0, 1 << 26), (16, 64, 8, -(1 << 62), 1 << 62),
                                (16, N_FEATURES, 8, -(1 << 34), 1 << 34), (0, 5, 3, 0, 1), (3, 4, 0, 0, 9)):
            h = rng.integers(lo, hi, size=(n, k), endpoint=True, dtype=np.int64)
            w = rng.integers(lo32, hi32, size=(k, m), endpoint=True, dtype=np.int64).astype(np.int32)
            if n and m:
                h[0, :] = hi  # the largest sums
                w[:, 0] = lo32
                w[:, -1] = hi32
            got = intnet.exact_matmul(h, w)
            self.assertEqual(got.dtype, np.int64)
            np.testing.assert_array_equal(got, h @ w.astype(np.int64), err_msg=str((n, k, m)))

    def test_logits_equal_the_int64_logits(self):
        d = json.loads(GOLDEN.read_text())
        active, dense = intnet.golden_features(d["seed"] + 1, d["count"])
        x = intnet.inputs(active, dense)
        for layers in (intnet.golden_net(d["seed"]), intnet.golden_net(d["seed"], 14),
                       quantize(random_float_net(4, (256, 64, 32), 14))):
            np.testing.assert_array_equal(intnet.logits(layers, x), intnet.logits_int64(layers, x))


class Quantise(unittest.TestCase):
    def test_integer_network_follows_the_float_one(self):
        fl = random_float_net(3)
        ints = quantize(fl)
        active, dense = intnet.golden_features(5, 50)
        x = intnet.inputs(active, dense)
        zf = float_logits(fl, x)
        zi = intnet.logits(ints, x).astype(np.float64) / (1 << 24)
        self.assertLess(np.abs(zf - zi).max(), 1e-4)

    def test_wide_networks_keep_every_fraction_bit(self):
        # Trained-scale weights (normal, 1/sqrt(fan-in)) at widths 2048 and 4096: the largest scale
        # s = 24 passes the 2^62 bound with room (the L1 norm per output is about 0.8 sqrt(F), far
        # inside the 2^11 that s = 24 allows; crates/nn/src/mlp.rs, "Widths and the overflow
        # bound"), so the shifts are those of a narrow network, and the integer network follows
        # the float one as closely.
        active, dense = intnet.golden_features(5, 40)
        x = intnet.inputs(active, dense)
        for seed, widths in ((7, (2048, 1024, 256)), (8, (2048, 2048, 1024, 256)), (9, (4096, 2048, 64))):
            fl = random_float_net(seed, widths, 14)
            ints = quantize(fl)
            self.assertEqual([s for _, _, s in ints], [28] + [24] * (len(widths) - 1) + [20], widths)
            for k, (w, b, _) in enumerate(ints):
                max_in = 1 << 24 if k == 0 else intnet.CLIP
                worst = max(abs(int(b[j])) + int(np.abs(w[:, j].astype(np.int64)).sum()) * max_in
                            for j in range(w.shape[1]))
                self.assertLessEqual(worst, intnet.SUM_BOUND // 2**4, f"{widths} layer {k}: room to spare")
            zf = float_logits(fl, x)
            zi = intnet.logits(ints, x).astype(np.float64) / (1 << 24)
            self.assertLess(np.abs(zf - zi).max(), 1e-4, widths)

    def test_large_wide_weights_get_fewer_fraction_bits(self):
        # Weights of 1 on all 4096 inputs of an output: an L1 norm of 2^12, over the 2^11 that 24
        # fraction bits allow, so the quantiser takes 23 (shift 23), and the bound holds exactly
        # at half (2^12 * 2^23 * 2^26 = 2^61); a little more takes 22.
        rng = np.random.default_rng(3)
        for scale, frac in ((1.0, 23), (1.01, 22)):
            fl = [(rng.normal(0, 0.05, size=(N_FEATURES, 4096)), np.zeros(4096)),
                  (np.full((4096, 4), scale), np.zeros(4))]
            ints = quantize(fl)
            self.assertEqual(ints[0][2], 28)
            self.assertEqual(ints[1][2], frac + 20 - 24, scale)
            w = ints[1][0].astype(np.int64)
            self.assertLessEqual(int(np.abs(w).sum(axis=0).max()) * intnet.CLIP, intnet.SUM_BOUND // 2)

    def test_safetensors_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.safetensors")
            t = {
                "a": np.arange(6, dtype=np.int32).reshape(2, 3) - 3,
                "b": np.array([-(1 << 62), 5], dtype=np.int64),
                "c": np.array([0.5], dtype=np.float32),
            }
            write_safetensors(p, t, {"k": "v"})
            t2, m = read_safetensors(p)
            self.assertEqual(m, {"k": "v"})
            for k in t:
                np.testing.assert_array_equal(t[k], t2[k])
            with open(p, "rb") as f:
                self.assertEqual(int.from_bytes(f.read(8), "little") % 8, 0)


@unittest.skipIf(nn_tool() is None, "no nn-tool binary (cargo build --release -p nn)")
class Parity(unittest.TestCase):
    def test_rust_and_python_integer_networks_agree(self):
        tool = nn_tool()
        count = int(os.environ.get("NN_PARITY_COUNT", "3000"))
        with tempfile.TemporaryDirectory() as d:
            for trick in (3, 4, 5):
                shard = os.path.join(d, f"fixture-{trick}.npy")
                subprocess.run(
                    [tool, "fixture", "--trick", str(trick), "--count", str(count // 3), "--seed", "17", "--out", shard],
                    check=True,
                )
                rows = np.asarray(load_shard(shard))
                # Every row's outcome label carries exactly its target (completeness, spec 6a).
                check_values(rows)
                self.assertEqual(rows["outcomes"].shape[1], VOCAB[trick])
                nets = [(1, (64, 16), "money4"), (2, (128, 32, 16), "outcome-v1")]
                # Wide networks (MAX_WIDTH 4096): 2048 at trick 4, 4096 at trick 5.
                nets += {3: [], 4: [(3, (2048, 1024, 256), "outcome-v1")], 5: [(4, (4096, 512), "outcome-v1")]}[trick]
                for seed, widths, head in nets:
                    n_out = 4 if head == "money4" else VOCAB[trick]
                    net = os.path.join(d, f"net{seed}-{trick}.safetensors")
                    write_int(net, quantize(random_float_net(seed, widths, n_out)), trick, {"test": "parity"}, head=head)
                    out = os.path.join(d, f"out{seed}-{trick}.npy")
                    subprocess.run([tool, "eval", "--net", net, "--shard", shard, "--out", out, "--reps", "1"], check=True)
                    rust = np.load(out)
                    layers, meta = intnet.load(net)
                    self.assertEqual(meta["head"], head)
                    if head == "money4":
                        py = intnet.money_rows(layers, rows["active"], rows["dense"])
                    else:
                        py = intnet.money_rows(
                            layers, rows["active"], rows["dense"], mask=rows["mask"], outcomes=rows["outcomes"]
                        )
                    self.assertEqual(rust.shape, (len(rows), 4))
                    np.testing.assert_array_equal(rust, py)
                    # Not degenerate: the outputs spread.
                    self.assertGreater(len(np.unique(rust[:, 0])), len(rows) // 3)

if __name__ == "__main__":
    unittest.main()
