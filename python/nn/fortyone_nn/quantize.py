"""Float network -> the Rust integer scheme (``41bots-int-mlp-v1``, see ``intnet``).

Each layer's weights are stored as int32 ``round(w * 2^s)`` with the largest ``s <= 24`` such that
every weight fits int32 and no sum can leave +-2^62 for any input; biases as int64
``round(b * 2^(s + in))`` (``in`` = 24 for the first layer, 20 after); the layer's right shift is
``s + in - out`` (``out`` = 20 for hidden layers, 24 for the logits). With 24 fraction bits per
weight the integer network computes the float one to about 1e-7 per weight; the remaining
difference (activation and logit rounding) is measured, never assumed away.
"""

import json

import numpy as np

from .formats import ENCODER, INPUT_SHIFT, N_FEATURES, write_safetensors
from .intnet import ACT_SHIFT, CLIP, FORMAT, LOGIT_SHIFT, SUM_BOUND, check_layers

MAX_FRAC = 24


def quantize(float_layers, n_features=N_FEATURES):
    """``float_layers``: [(w float [n_in, n_out] input-major, b float [n_out])]; returns
    [(w int32, b int64, shift)]. ``n_features``: the first layer's inputs (the oracle encoder's by
    default; the student's when given: the same scheme, ``crates/student``)."""
    out = []
    last = len(float_layers) - 1
    for k, (w, b) in enumerate(float_layers):
        w = np.asarray(w, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        s_in = INPUT_SHIFT if k == 0 else ACT_SHIFT
        s_out = LOGIT_SHIFT if k == last else ACT_SHIFT
        max_in = 1 << INPUT_SHIFT if k == 0 else CLIP
        for s in range(MAX_FRAC, -1, -1):
            if s + s_in < s_out:
                raise ValueError(f"layer {k}: weights too large to quantise")
            wi = np.rint(w * 2.0**s)
            bi = np.rint(b * 2.0 ** (s + s_in))
            if np.abs(wi).max() >= 2**31 - 1:
                continue
            worst = np.abs(wi).sum(axis=0) * max_in + np.abs(bi)
            if worst.max() > SUM_BOUND / 2:  # room for float rounding in this check
                continue
            out.append((wi.astype(np.int32), bi.astype(np.int64), s + s_in - s_out))
            break
    check_layers(out, n_features)
    return out


def write_int(path, int_layers, trick, meta, head="money4"):
    """Write an integer network in the format ``crates/nn`` reads (``head``: ``money4`` or
    ``outcome-v1``)."""
    tensors = {}
    for k, (w, b, _) in enumerate(int_layers):
        tensors[f"w{k}"] = w
        tensors[f"b{k}"] = b
    m = dict(meta)
    m.update(
        {
            "format": FORMAT,
            "encoder": ENCODER,
            "n_features": str(N_FEATURES),
            "trick": str(trick),
            "shifts": json.dumps([int(s) for _, _, s in int_layers]),
            "head": head,
        }
    )
    write_safetensors(path, tensors, m)
