"""The integer network, bit for bit as ``crates/nn`` computes it (format ``41bots-int-mlp-v1``).

- Inputs: binary features that are on are 2^24, dense values -2^24..0.
- Layer: ``acc = b + x @ w`` in int64 (exact: the files are checked so that no sum can leave
  +-2^62), then ``y = (acc + 2^(k-1)) >> k`` (arithmetic shift, round half up); hidden layers clip
  ``y`` to ``[0, 64 * 2^20]``; the output layer gives logits in units of 2^-24.
- Softmax: ``d = max(z) - z``; ``y = d * LOG2E_Q32``; ``n = y >> 56``; ``f = (y >> 32) mod 2^24``;
  the weight is ``2^62`` times the product, over the set bits m of f from 23 down to 0, of
  ``R[24 - m] = 2^(-1/2^(24-m))`` (units of 2^-62, each product floored), shifted right by n (0
  when n >= 63). ``R[0] = 2^61``, ``R[k] = isqrt(R[k-1] * 2^62)``. Then largest-remainder
  normalisation to 2^32, ties to the lower index (``equity::MoneyChance::from_weights``).
"""

import json
import math

import numpy as np

from .formats import INPUT_SHIFT, MAX_ACTIVE, N_BINARY, N_DENSE, N_FEATURES, PAD, read_safetensors

FORMAT = "41bots-int-mlp-v1"
ACT_SHIFT = 20
LOGIT_SHIFT = 24
CLIP = 64 << ACT_SHIFT
SUM_BOUND = 1 << 62
LOG2E_Q32 = 6196328019
ONE = 1 << 32

R = [1 << 61]
for _k in range(1, 25):
    R.append(math.isqrt(R[-1] << 62))


def round_shift(x, k):
    if k == 0:
        return x
    return (x + (1 << (k - 1))) >> k


def check_layers(layers):
    """The Rust checks: chained shapes, N_FEATURES inputs, 4 to 64 outputs, shifts < 63, and no
    sum can leave +-2^62 for any input."""
    if layers[0][0].shape[0] != N_FEATURES or not 4 <= layers[-1][0].shape[1] <= 64:
        raise ValueError("shapes")
    for k, (w, b, shift) in enumerate(layers):
        if k and w.shape[0] != layers[k - 1][0].shape[1]:
            raise ValueError(f"layer {k}: shape")
        if w.dtype != np.int32 or b.dtype != np.int64 or not 0 <= shift < 63:
            raise ValueError(f"layer {k}: types or shift")
        max_in = 1 << INPUT_SHIFT if k == 0 else CLIP
        worst = [abs(int(b[j])) + int(np.abs(w[:, j].astype(np.int64)).sum()) * max_in for j in range(w.shape[1])]
        if max(worst) > SUM_BOUND:
            raise ValueError(f"layer {k}: a sum could leave 2^62")


def load(path):
    """(layers [(w int32 [n_in, n_out], b int64 [n_out], shift)], metadata)."""
    t, meta = read_safetensors(path)
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: not {FORMAT}")
    shifts = json.loads(meta["shifts"])
    layers = [(t[f"w{k}"], t[f"b{k}"], s) for k, s in enumerate(shifts)]
    check_layers(layers)
    return layers, meta


def inputs(active, dense):
    """The integer input matrix (int64) from shard columns."""
    active = np.asarray(active)
    n = active.shape[0]
    x = np.zeros((n, N_FEATURES), dtype=np.int64)
    rows = np.repeat(np.arange(n), MAX_ACTIVE)
    cols = active.reshape(-1).astype(np.int64)
    m = cols != PAD
    x[rows[m], cols[m]] = 1 << INPUT_SHIFT
    x[:, N_BINARY:] = np.asarray(dense, dtype=np.int64)
    return x


F64_EXACT_BITS = 53  # every integer of magnitude up to 2^53 is exactly a float64


def exact_matmul(h, w):
    """``h @ w`` for integer ``h`` [n, k] and ``w`` [k, m]: exactly numpy's int64 product (the
    same int64 values, wrap-around included), computed through float64 BLAS, about a hundred
    times faster than numpy's integer product (which has no BLAS).

    ``w`` is cut into pieces of ``bits`` bits (the low pieces unsigned, the top one signed), with
    ``bits`` small enough that k * max|h| * 2^bits <= 2^53. Then every partial sum of ``h`` times
    a piece, in whatever order BLAS adds, is an integer of magnitude below 2^53, which float64
    holds exactly; the pieces' products are put back together in int64, each times 2^offset.
    When ``h`` is too large for pieces of at least 8 bits, numpy's int64 product is used."""
    h = np.asarray(h, dtype=np.int64)
    w = np.asarray(w, dtype=np.int64)
    hmax = max(-int(h.min()), int(h.max())) if h.size else 0
    bits = F64_EXACT_BITS - (h.shape[1] * hmax).bit_length()
    if bits < 8:
        return h @ w
    hf = h.astype(np.float64)
    out = np.zeros((h.shape[0], w.shape[1]), dtype=np.int64)
    rest, offset = w, 0
    while True:
        top = not rest.size or max(-int(rest.min()), int(rest.max())) < (1 << bits)
        piece = rest if top else rest & ((1 << bits) - 1)
        out += (hf @ piece.astype(np.float64)).astype(np.int64) * np.int64(1 << offset)
        if top:
            return out
        rest = rest >> bits
        offset += bits


def logits(layers, x):
    """Logits (int64, units of 2^-24) for an integer input matrix (products by
    ``exact_matmul``: the same integers as ``logits_int64``)."""
    h = x
    last = len(layers) - 1
    for k, (w, b, shift) in enumerate(layers):
        acc = exact_matmul(h, w) + b
        y = round_shift(acc, shift)
        h = y if k == last else np.clip(y, 0, CLIP)
    return h


def logits_int64(layers, x):
    """``logits`` with numpy's plain int64 product: the reference ``logits`` is tested against."""
    h = x
    last = len(layers) - 1
    for k, (w, b, shift) in enumerate(layers):
        acc = h @ w.astype(np.int64) + b
        y = round_shift(acc, shift)
        h = y if k == last else np.clip(y, 0, CLIP)
    return h


def exp_weight(d):
    y = d * LOG2E_Q32
    n = y >> 56
    if n >= 63:
        return 0
    f = (y >> 32) & 0xFFFFFF
    w = 1 << 62
    for m in range(23, -1, -1):
        if (f >> m) & 1:
            w = (w * R[24 - m]) >> 62
    return w >> n


def from_weights(w):
    total = sum(w)
    q = [(x << 32) // total for x in w]
    r = [(x << 32) % total for x in w]
    missing = ONE - sum(q)
    for i in sorted(range(4), key=lambda i: (-r[i], i))[:missing]:
        q[i] += 1
    return q


def money(z):
    """Money chance (four ints summing to 2^32) of one row of logits."""
    z = [int(v) for v in z]
    m = max(z)
    return from_weights([exp_weight(m - v) for v in z])


def mix_wide(terms):
    """``equity::MoneyChance::mix_wide``: the weighted mean of money vectors, u128 sums, then
    largest remainder (ties to the lower team); zero weights ignored."""
    num = [0, 0, 0, 0]
    den = 0
    for w, c in terms:
        if w == 0:
            continue
        den += w
        for t in range(4):
            num[t] += w * int(c[t])
    q = [x // den for x in num]
    r = [x % den for x in num]
    missing = ONE - sum(q)
    for i in sorted(range(4), key=lambda i: (-r[i], i))[:missing]:
        q[i] += 1
    return q


def money_outcomes(z, mask, vecs):
    """The outcome head's money chance: integer softmax over the possible outcomes' logits, then
    the mixture of their money vectors (``nn::outcome::money_from_outcomes``)."""
    mask = int(mask)
    idx = [o for o in range(len(vecs)) if (mask >> o) & 1]
    m = max(int(z[o]) for o in idx)
    return mix_wide([(exp_weight(m - int(z[o])), vecs[o]) for o in idx])


def money_rows(layers, active, dense, chunk=1024, mask=None, outcomes=None):
    """Money chances (uint64 [n, 4]) of shard rows: the four-logit head, or with ``mask`` and
    ``outcomes`` the outcome head."""
    out = []
    for i in range(0, len(active), chunk):
        z = logits(layers, inputs(active[i : i + chunk], dense[i : i + chunk]))
        if mask is None:
            out.extend(money(row) for row in z)
        else:
            out.extend(money_outcomes(row, mask[i + j], outcomes[i + j]) for j, row in enumerate(z))
    return np.array(out, dtype=np.uint64).reshape(-1, 4)


class SplitMix64:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)


def golden_net(seed, n_out=4):
    """The golden network of ``crates/nn/src/synthetic.rs`` (same recipe)."""
    sm = SplitMix64(seed)
    dims = [N_FEATURES, 32, 16, n_out]
    shifts = [27, 23, 16]
    layers = []
    for k in range(3):
        n_in, n_out = dims[k], dims[k + 1]
        w = np.array([(sm.next() % (1 << 21)) - (1 << 20) for _ in range(n_in * n_out)], dtype=np.int32)
        b = np.array([(sm.next() % (1 << 40)) - (1 << 39) for _ in range(n_out)], dtype=np.int64)
        layers.append((w.reshape(n_in, n_out), b, shifts[k]))
    check_layers(layers)
    return layers


def golden_features(seed, count):
    """The golden inputs of ``crates/nn/src/synthetic.rs``: (active [count, 96], dense [count, 20])."""
    sm = SplitMix64(seed)
    active = np.full((count, MAX_ACTIVE), PAD, dtype=np.uint16)
    dense = np.zeros((count, N_DENSE), dtype=np.int32)
    for r in range(count):
        n = 0
        for i in range(N_BINARY):
            if sm.next() % 10 == 0 and n < MAX_ACTIVE:
                active[r, n] = i
                n += 1
        for d in range(N_DENSE):
            dense[r, d] = -(sm.next() % ((1 << 24) + 1))
    return active, dense


def golden_outcomes(seed, count, v):
    """The golden outcome tables of ``crates/nn/src/synthetic.rs``: [(mask, vecs)]."""
    sm = SplitMix64(seed)
    out = []
    for _ in range(count):
        mask = sm.next() % (1 << v)
        mask |= 1 << (sm.next() % v)
        vecs = []
        for _ in range(v):
            a, b, c = sm.next() % (1 << 30), sm.next() % (1 << 30), sm.next() % (1 << 30)
            vecs.append([a, b, c, ONE - a - b - c])
        out.append((mask, vecs))
    return out
