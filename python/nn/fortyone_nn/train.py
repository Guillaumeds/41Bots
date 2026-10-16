"""Train an oracle value network O_k (docs/M3-HYBRID-SPEC.md section 6).

    uv run python -m fortyone_nn.train --train A.npy [B.npy ...] --val V.npy --out DIR
        [--widths 256,64] [--epochs 40] [--batch 1024] [--lr 1e-3] [--patience 4]
        [--seed 1] [--threads 4] [--max-minutes M] [--trick K]

A plain MLP: the binary features through an embedding-bag sum (exactly a linear layer on the
one-hot inputs), plus the dense features through a linear layer, ReLU clipped at 64 (as the
integer inference), hidden layers, four logits. Loss: cross-entropy of the four-way softmax
against the exact four-vector (soft targets; PLAN section 6.5 value-head rule). Adam, seeded,
early stopping on the validation shard. Writes to DIR:

- ``float.safetensors``: the float weights (input-major, as the integer file), with metadata;
- ``int.safetensors``: the quantised network ``crates/nn`` loads (format 41bots-int-mlp-v1);
- ``report.json``: settings, data hashes, per-epoch losses, and validation metrics of the float
  and the integer network (cross-entropy, KL to the targets, and the error of each team's money
  chance: mean, SD, 99.9th percentile of the absolute value, max).
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .formats import (
    INPUT_SHIFT,
    MAX_ACTIVE,
    N_BINARY,
    N_DENSE,
    ONE,
    PAD,
    check_values,
    load_shard,
    sha256_file,
    write_safetensors,
)
from .intnet import money_rows
from .quantize import quantize, write_int

CLIP_F = 64.0


class Net(nn.Module):
    def __init__(self, widths):
        super().__init__()
        w0 = widths[0]
        self.emb = nn.EmbeddingBag(N_BINARY + 1, w0, mode="sum", padding_idx=N_BINARY)
        bound = 1.0 / math.sqrt(N_BINARY + N_DENSE)
        with torch.no_grad():
            self.emb.weight.uniform_(-bound, bound)
            self.emb.weight[N_BINARY].zero_()
        self.dense = nn.Linear(N_DENSE, w0)
        self.hidden = nn.ModuleList(nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1))
        self.out = nn.Linear(widths[-1], 4)

    def forward(self, idx, dense):
        h = torch.clamp(self.emb(idx) + self.dense(dense), 0.0, CLIP_F)
        for layer in self.hidden:
            h = torch.clamp(layer(h), 0.0, CLIP_F)
        return self.out(h)

    def float_layers(self):
        """[(w [n_in, n_out] input-major, b [n_out])] in the integer file's layout."""
        w0 = torch.cat([self.emb.weight[:N_BINARY], self.dense.weight.t()], dim=0)
        layers = [(w0, self.dense.bias)]
        layers += [(l.weight.t(), l.bias) for l in self.hidden]
        layers.append((self.out.weight.t(), self.out.bias))
        return [(w.detach().double().numpy(), b.detach().double().numpy()) for w, b in layers]


def tensors(a):
    """(index tensor with padding -> N_BINARY, dense float tensor, target float tensor)."""
    idx = np.asarray(a["active"]).astype(np.int64)
    idx[idx == PAD] = N_BINARY
    dense = np.asarray(a["dense"]).astype(np.float32) / float(1 << INPUT_SHIFT)
    tgt = np.asarray(a["value"]).astype(np.float64) / float(ONE)
    return torch.from_numpy(idx), torch.from_numpy(dense), torch.from_numpy(tgt.astype(np.float32))


def soft_ce(z, t):
    return -(t * torch.log_softmax(z, dim=1)).sum(dim=1).mean()


def entropy(t):
    t = t.clamp_min(1e-30)
    return float(-(t * torch.log(t)).sum(dim=1).mean())


def error_stats(pred, target):
    """Errors of each team's money chance (fractions): mean, SD, 99.9th percentile of |e|, max."""
    e = (pred.astype(np.float64) - target.astype(np.float64)) / ONE
    a = np.abs(e)
    return {
        "bias": float(e.mean()),
        "sd": float(e.std()),
        "mean_abs": float(a.mean()),
        "p999_abs": float(np.quantile(a, 0.999)),
        "max_abs": float(a.max()),
    }


def evaluate(model, data, batch=8192):
    idx, dense, tgt = data
    model.eval()
    zs = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            zs.append(model(idx[i : i + batch], dense[i : i + batch]))
    z = torch.cat(zs)
    ce = float(soft_ce(z, tgt))
    p = torch.softmax(z.double(), dim=1).numpy() * ONE
    return ce, p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--widths", default="256,64")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-minutes", type=float, default=0.0, help="stop training after this long (0 = no limit)")
    ap.add_argument("--trick", type=int, default=0, help="the frontier trick (default: read from the shards)")
    ap.add_argument("--int-eval-rows", type=int, default=20000, help="validation rows for the integer network")
    args = ap.parse_args(argv)

    t0 = time.time()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    widths = [int(x) for x in args.widths.split(",")]
    os.makedirs(args.out, exist_ok=True)

    train = np.concatenate([np.asarray(load_shard(p)) for p in args.train])
    val = np.asarray(load_shard(args.val))
    check_values(train)
    check_values(val)
    tricks = set(np.unique(train["trick"]).tolist()) | set(np.unique(val["trick"]).tolist())
    trick = args.trick or (tricks.pop() if len(tricks) == 1 else 0)
    if not trick:
        raise SystemExit(f"shards hold tricks {sorted(tricks)}: pass --trick")
    hashes = {"train": [sha256_file(p) for p in args.train], "val": sha256_file(args.val)}
    print(f"train {len(train)} rows, val {len(val)} rows, trick {trick}, widths {widths}", flush=True)

    tr = tensors(train)
    va = tensors(val)
    model = Net(widths)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=1)
    val_entropy = entropy(va[2])
    best = (math.inf, None, -1)
    history = []
    n = len(train)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(n))
        total = 0.0
        for i in range(0, n, args.batch):
            b = perm[i : i + args.batch]
            loss = soft_ce(model(tr[0][b], tr[1][b]), tr[2][b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
        vce, _ = evaluate(model, va)
        sched.step(vce)
        history.append({"epoch": epoch, "train_ce": total / n, "val_ce": vce, "val_kl": vce - val_entropy,
                        "lr": opt.param_groups[0]["lr"], "minutes": (time.time() - t0) / 60})
        print(json.dumps(history[-1]), flush=True)
        if vce < best[0]:
            best = (vce, {k: v.clone() for k, v in model.state_dict().items()}, epoch)
        elif epoch - best[2] >= args.patience:
            break
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            print("time limit reached", flush=True)
            break
    model.load_state_dict(best[1])
    vce, pf = evaluate(model, va)

    layers = model.float_layers()
    meta = {
        "widths": json.dumps(widths),
        "train_sha256": json.dumps(hashes["train"]),
        "val_sha256": hashes["val"],
        "train_rows": str(len(train)),
        "seed": str(args.seed),
        "best_epoch": str(best[2]),
        "val_ce": repr(vce),
        "torch": torch.__version__,
    }
    ft = {}
    for k, (w, b) in enumerate(layers):
        ft[f"w{k}"] = w.astype(np.float32)
        ft[f"b{k}"] = b.astype(np.float32)
    write_safetensors(os.path.join(args.out, "float.safetensors"), ft, {**meta, "trick": str(trick)})
    ints = quantize(layers)
    int_path = os.path.join(args.out, "int.safetensors")
    write_int(int_path, ints, trick, meta)

    m = min(args.int_eval_rows, len(val))
    pi = money_rows(ints, val["active"][:m], val["dense"][:m]).astype(np.float64)
    report = {
        "args": vars(args),
        "trick": trick,
        "data": hashes,
        "history": history,
        "best_epoch": best[2],
        "val_entropy": val_entropy,
        "float": {"val_ce": vce, "val_kl": vce - val_entropy, "error": error_stats(pf, val["value"])},
        "int": {
            "rows": m,
            "error": error_stats(pi, val["value"][:m]),
            "vs_float": error_stats(pi, pf[:m]),
            "sha256": sha256_file(int_path),
            "shifts": [int(s) for _, _, s in ints],
        },
        "minutes": (time.time() - t0) / 60,
    }
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("float", "int", "minutes")}, indent=1))


if __name__ == "__main__":
    main()
