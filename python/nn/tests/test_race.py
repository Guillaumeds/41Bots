"""Race classes of label rows (``fortyone_nn.formats.race_classes``) against Rust.

- hand-made rows: every class 0-4 from the possible game-won outcomes (mask bits 0-3) and the
  teams with a game won (binary features ``OFF_GAMES + t``), the hand-over outcomes ignored;
- parity: on real encoded positions (``nn-tool fixture``), the classes equal Rust's
  ``nn::outcome::race_class_of`` of each row's position decoded from its features (``nn-tool
  classes``: not from the row's mask), so the trainer's per-class report counts the classes the
  harvest selects by. Needs the ``nn-tool`` binary (as ``test_intnet``); skipped without it.

numpy only (runs in the CI's parity job). Run: ``cd python/nn && uv run python -m unittest
discover -s tests -v``.
"""

import os
import subprocess
import tempfile
import unittest

import numpy as np

from fortyone_nn.formats import MAX_ACTIVE, OFF_GAMES, PAD, RACE_CLASS_NAMES, VOCAB, load_shard, race_classes, shard_dtype

from test_intnet import nn_tool


def rows(cases):
    """Rows with the given (mask, teams with a game) and nothing else that matters."""
    a = np.zeros(len(cases), dtype=shard_dtype(VOCAB[4]))
    for i, (mask, games) in enumerate(cases):
        act = [5, 700] + [OFF_GAMES + t for t in games]
        a["active"][i, :] = PAD
        a["active"][i, : len(act)] = sorted(act)
        a["mask"][i] = mask
    return a


class RaceClasses(unittest.TestCase):
    def test_hand_made_rows(self):
        cases = [
            (0b1111_0000, ()),  # no game-won outcome possible (hand-over splits only)
            (0b0100, ()),  # one team can win the game
            (0b0100, (0,)),  # one team can win the game; a team with a game cannot
            (0b0110, ()),  # two can, nobody would win the match
            (0b0100, (2,)),  # one team can win the match
            (0b0101, (0,)),  # a team can win the match and another the game
            (0b1111, (0, 1)),
        ]
        self.assertEqual(list(race_classes(rows(cases))), [0, 1, 1, 2, 3, 4, 4])
        self.assertEqual(len(RACE_CLASS_NAMES), 5)
        self.assertEqual(MAX_ACTIVE, rows(cases)["active"].shape[1])


@unittest.skipIf(nn_tool() is None, "no nn-tool binary (cargo build --release -p nn)")
class Parity(unittest.TestCase):
    def test_python_classes_are_rusts(self):
        tool = nn_tool()
        count = int(os.environ.get("NN_PARITY_COUNT", "3000")) // 3
        seen = set()
        with tempfile.TemporaryDirectory() as d:
            for trick in (3, 4, 5):
                shard = os.path.join(d, f"fixture-{trick}.npy")
                subprocess.run(
                    [tool, "fixture", "--trick", str(trick), "--count", str(count), "--seed", "29", "--out", shard],
                    check=True,
                )
                out = os.path.join(d, f"classes-{trick}.npy")
                subprocess.run([tool, "classes", "--shard", shard, "--out", out], check=True)
                a = np.asarray(load_shard(shard))
                rust = np.load(out)
                self.assertEqual(rust.shape, (len(a),))
                np.testing.assert_array_equal(race_classes(a), rust.astype(np.int64))
                seen.update(int(x) for x in np.unique(rust))
        # The fixtures (random play in a match's first hands) reach several classes.
        self.assertGreaterEqual(len(seen), 3, seen)


if __name__ == "__main__":
    unittest.main()
