"""The Python integer network against Rust ``crates/nn``.

- golden: the seeded golden network and inputs (same recipe on both sides) give the outputs in
  ``crates/nn/tests/data/intnet-golden.json``, which the Rust tests also check;
- quantisation: the integer network computes the float one closely, and safetensors round-trip;
- parity: on real encoded positions (``nn-tool fixture``) and a quantised random network, the
  Rust integer forward pass (``nn-tool eval``) and this one give bit-identical money chances.
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


class Quantise(unittest.TestCase):
    def test_integer_network_follows_the_float_one(self):
        fl = random_float_net(3)
        ints = quantize(fl)
        active, dense = intnet.golden_features(5, 50)
        x = intnet.inputs(active, dense)
        zf = float_logits(fl, x)
        zi = intnet.logits(ints, x).astype(np.float64) / (1 << 24)
        self.assertLess(np.abs(zf - zi).max(), 1e-4)

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
                for seed, widths, head in ((1, (64, 16), "money4"), (2, (128, 32, 16), "outcome-v1")):
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
