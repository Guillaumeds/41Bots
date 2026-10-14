"""Read a teacher `decisions-<sha16>.npy` shard (docs/M3-SPEC.md section 5.10), numpy only.

Usage: python read_decisions.py decisions-XXXX.npy
Prints the row count and the first row's main fields.
"""

import sys

import numpy as np

FIELDS = [
    "hand_id", "action_index", "seat", "trick", "kind", "method", "belief",
    "n_worlds", "log2_world_count_q8", "cand_action", "values", "own_sd", "chosen",
]


def read(path):
    a = np.load(path, allow_pickle=False)
    assert list(a.dtype.names) == FIELDS, a.dtype.names
    # Every candidate's money chance sums to exactly 2^32.
    n = (a["cand_action"] != 0).sum(axis=1)
    for i in range(len(a)):
        sums = a["values"][i, : n[i]].sum(axis=1, dtype=np.uint64)
        assert (sums == np.uint64(1 << 32)).all(), (i, sums)
    return a


if __name__ == "__main__":
    a = read(sys.argv[1])
    print(len(a), "decisions")
    if len(a):
        r = a[0]
        print({k: (r[k].tolist() if k not in ("values", "own_sd", "cand_action") else None) for k in FIELDS})
