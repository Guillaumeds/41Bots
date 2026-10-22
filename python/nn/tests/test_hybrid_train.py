"""Training on hybrid-labelled shards (docs/m3-hybrid/audit-loop.md section 12): the target is the
hybrid's mixture of endings, so no outcome need have exactly its money vector. The trainer's value
check skips that one test for shards whose sidecar says ``label_source`` ``"hybrid"`` and keeps it
for every other shard (found 2026-10-22: the first hybrid training run died on it).
"""

import json
import os
import tempfile
import unittest

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:
    raise unittest.SkipTest("PyTorch is not installed") from None

from fortyone_nn import train
from fortyone_nn.formats import ONE, is_hybrid_shard, load_shard, value_fault

from test_train import synthetic_shard


def hybridise(path):
    """Turn a synthetic exact shard into a hybrid one: the target becomes the mean of the label's
    vector and another outcome's (it still sums to 2^32), and the sidecar says hybrid."""
    a = np.load(path)
    other = np.roll(a["outcomes"], 1, axis=1)[np.arange(len(a)), a["label"]]
    mix = (a["value"] // np.uint64(2)) + (other // np.uint64(2))
    mix[:, 0] += np.uint64(ONE) - mix.sum(axis=1, dtype=np.uint64)
    a["value"] = mix
    np.save(path, a)
    with open(path + ".json", "w") as f:
        json.dump({"label_source": "hybrid"}, f)
    return path


class HybridTrainTest(unittest.TestCase):
    def test_check_and_training(self):
        with tempfile.TemporaryDirectory() as d:
            h = hybridise(synthetic_shard(os.path.join(d, "h.npy"), 200, 1))
            e = synthetic_shard(os.path.join(d, "e.npy"), 200, 2)
            v = synthetic_shard(os.path.join(d, "v.npy"), 50, 3)
            self.assertTrue(is_hybrid_shard(h))
            self.assertFalse(is_hybrid_shard(e))
            a = np.asarray(load_shard(h))
            self.assertEqual(value_fault(a), 3, "strict check refuses a mixture target")
            self.assertIsNone(value_fault(a, exact=False))
            with self.assertRaises(ValueError):
                train.check_shards([a], exact=[True])
            train.check_shards([a], exact=[False])
            # a hybrid shard whose targets do not sum to 2^32 is still refused
            bad = a.copy()
            bad["value"][0, 0] += np.uint64(1)
            self.assertEqual(value_fault(bad, exact=False), 0)
            out = os.path.join(d, "o")
            train.main(["--train", h, e, "--val", v, "--out", out, "--device", "cpu", "--widths", "16,8",
                        "--epochs", "1", "--batch", "64", "--head", "outcome"])
            self.assertTrue(os.path.exists(os.path.join(out, "float.safetensors")))


class SidecarRaceTest(unittest.TestCase):
    def test_sidecar_appearing_late_is_waited_for(self):
        import threading
        import time

        with tempfile.TemporaryDirectory() as d:
            path = synthetic_shard(os.path.join(d, "h.npy"), 100, 4)
            a = np.load(path)
            other = np.roll(a["outcomes"], 1, axis=1)[np.arange(len(a)), a["label"]]
            mix = (a["value"] // np.uint64(2)) + (other // np.uint64(2))
            mix[:, 0] += np.uint64(ONE) - mix.sum(axis=1, dtype=np.uint64)
            a["value"] = mix
            np.save(path, a)
            sh = load_shard(path)

            def late():
                time.sleep(1.0)
                with open(path + ".json", "w") as f:
                    json.dump({"label_source": "hybrid"}, f)

            t = threading.Thread(target=late)
            t.start()
            tricks = train.check_shards_waiting_for_sidecar(sh, path, 1, wait_s=10, poll_s=0.1)
            t.join()
            self.assertEqual(tricks, {4})
            # an exact-looking mixture with no sidecar ever is refused after the wait
            os.remove(path + ".json")
            with self.assertRaises(ValueError):
                train.check_shards_waiting_for_sidecar(sh, path, 1, wait_s=0.5, poll_s=0.1)


if __name__ == "__main__":
    unittest.main()
