"""Train the student network (docs/STUDENT.md sections 1, 5, 6, 7).

    uv run python -m fortyone_nn.student --train F1.npy [F2.npy ...] --out DIR
        [--test T.npy] [--widths 256,128] [--epochs 40] [--batch 256] [--lr 1e-3] [--patience 4]
        [--seed 1] [--val-permille 50] [--threads 4] [--device cpu|cuda|auto] [--max-minutes M]
        [--init FLOAT [--shrink-perturb L] [--perturb-seed S]] [--limit N] [--progress FILE]
        [--rank-weight W] [--rank-temperature T] [--stop-on val_loss|decision]

The data are feature shards written by ``student encode`` (Rust, the only encoder; format
``41bots-student-features-v1``): one record per decision with the view's active features, the
dense block, each candidate's features and the teacher's mean money chance of each candidate in the
rotated pair order (the deciding pair first). The sizes come from the shards' sidecars, never from
constants here, so Python has no copy of the encoder.

The network (one scorer shared by every candidate): the view's binary features and the candidate's
through one embedding-bag sum (exactly a linear layer on the one-hot inputs), the dense features
through a linear layer, ReLU clipped at 64 (as the integer inference), hidden layers, four logits.
The view's part of the first layer is computed once per decision and added to each candidate's.

Loss: per candidate, the cross-entropy of the four-way softmax against the teacher's mean
four-vector (the oracle head's loss); the mean over a decision's candidates, then over decisions.
Decision-aware term (``--rank-weight W`` above 0, ``--rank-temperature T``; policy distillation,
Rusu et al. 2016, docs/STUDENT.md section 5 step 3): a choice depends only on the differences
between the cards of one decision, so each decision adds W times the KL divergence from the
teacher's softmax over its candidates to the student's, both taken on the deciding pair's money
chance divided by T (T in money-chance units: 0.01 is one point of chance). W = 0 (the default) is
exactly the loss above. ``--stop-on decision`` stops on the validation *decision* loss instead of
the value cross-entropy.

Validation: decisions whose position key (``pos_key``, the first 8 bytes of the position's SHA-256)
is below ``--val-permille`` per thousand (``pos_key mod 1000``): a position is never on both sides.
Early stopping on the validation loss (``--patience`` epochs), the best epoch kept.

Warm start (``--init`` a previous student's ``float.safetensors``; ``--shrink-perturb L``: Ash &
Adams 2020, ``fortyone_nn.train.SIGMA_RULE``): the run continues that network; train on all the data
selected so far, never on the new data alone (docs/STUDENT.md section 6).

``--limit N`` trains on the first N decisions of the training shards, in the order given (a
deterministic prefix: the runner's learning curve, ``student_train``), before the validation split;
more than the shards hold is refused. ``--progress FILE`` is rewritten (atomically) at each step and
about every 20 s within an epoch: ``step`` (load, train, quantise, measure, done), ``epoch``,
``epochs``, ``batches_done``, ``batches`` and ``minutes``.

Writes to DIR: ``float.safetensors`` (float32 layers in the integer layout), ``int.safetensors``
(the quantised network ``crates/student`` loads: format ``41bots-int-mlp-v1``, encoder ``student-v1``
with its hash, head ``student-money4``) and ``report.json`` (settings, data hashes, history, and
for validation and ``--test``: the cross-entropy, the decision loss against the teacher's means and
the agreement, of the float network and of the integer one computed exactly as Rust does). With
``--init`` and ``--test`` the report says whether the new network is no worse than the initial one
on the test set (``keep``): docs/STUDENT.md section 5's keep rule.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .device import deterministic
from .formats import parse_widths, read_safetensors, sha256_file, write_safetensors
from .quantize import quantize
from .studentint import (
    CLIP_F,
    ENCODER,
    HEAD,
    Columns,
    by_groups,
    check_same,
    concat,
    decision_report,
    indices,
    int_scores,
    load_features,
    write_student_int,
)

class Progress:
    """The ``--progress`` file: rewritten atomically (a temporary file, then a rename)."""

    def __init__(self, path, t0):
        self.path, self.t0, self.state = path, t0, {}

    def __call__(self, **kv):
        self.state.update(kv)
        self.state["minutes"] = round((time.time() - self.t0) / 60, 3)
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.path)


class StudentNet(nn.Module):
    def __init__(self, n_binary, n_dense, widths):
        super().__init__()
        w0 = widths[0]
        self.n_binary = n_binary
        self.emb = nn.EmbeddingBag(n_binary + 1, w0, mode="sum", padding_idx=n_binary)
        bound = 1.0 / math.sqrt(n_binary + n_dense)
        with torch.no_grad():
            self.emb.weight.uniform_(-bound, bound)
            self.emb.weight[n_binary].zero_()
        self.dense = nn.Linear(n_dense, w0)
        self.hidden = nn.ModuleList(nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1))
        self.out = nn.Linear(widths[-1], 4)

    def forward(self, view, dense, cand, dec):
        """Logits [N, 4] of N candidates; ``dec[i]`` is candidate i's decision (a row of
        ``view``/``dense``)."""
        hv = self.emb(view) + self.dense(dense)
        h = torch.clamp(hv[dec] + self.emb(cand), 0.0, CLIP_F)
        for layer in self.hidden:
            h = torch.clamp(layer(h), 0.0, CLIP_F)
        return self.out(h)

    def float_layers(self):
        """[(w [n_in, n_out] input-major, b [n_out])] in the integer file's layout (inputs: the
        binary features, then the dense ones)."""
        w0 = torch.cat([self.emb.weight[: self.n_binary], self.dense.weight.t()], dim=0)
        layers = [(w0, self.dense.bias)]
        layers += [(l.weight.t(), l.bias) for l in self.hidden]
        layers.append((self.out.weight.t(), self.out.bias))
        return [(w.detach().cpu().double().numpy(), b.detach().cpu().double().numpy()) for w, b in layers]

    def set_layers(self, layers):
        (w0, b0), last = layers[0], layers[-1]
        f32 = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
        with torch.no_grad():
            self.emb.weight[: self.n_binary].copy_(f32(w0[: self.n_binary]))
            self.emb.weight[self.n_binary].zero_()
            self.dense.weight.copy_(f32(w0[self.n_binary :].T))
            self.dense.bias.copy_(f32(b0))
            for layer, (w, b) in zip(self.hidden, layers[1:-1]):
                layer.weight.copy_(f32(w.T))
                layer.bias.copy_(f32(b))
            self.out.weight.copy_(f32(last[0].T))
            self.out.bias.copy_(f32(last[1]))


def batch_tensors(c, device):
    """Tensors of a Columns batch: view, dense, the valid candidates flattened, their decision,
    targets, and the candidates per decision."""
    dec_idx, cand_pos = np.nonzero(c.mask)
    return (
        torch.from_numpy(indices(c.view, c.n_binary)).to(device),
        torch.from_numpy(c.dense).to(device),
        torch.from_numpy(indices(c.cand[dec_idx, cand_pos], c.n_binary)).to(device),
        torch.from_numpy(dec_idx.astype(np.int64)).to(device),
        torch.from_numpy(c.target[dec_idx, cand_pos]).to(device),
        torch.from_numpy(c.n_cands.astype(np.float32)).to(device),
    )


def loss_of(z, target, dec, n_cands):
    """Per candidate soft cross-entropy, mean per decision, then mean over decisions."""
    ce = -(target * torch.log_softmax(z, dim=1)).sum(dim=1)
    per = torch.zeros(len(n_cands), dtype=ce.dtype, device=ce.device).index_add_(0, dec, ce)
    return (per / n_cands).mean()


def rank_term(z, target, dec, n_cands, temperature):
    """Mean over decisions of KL(teacher softmax || student softmax) across a decision's candidates,
    both softmaxes over the deciding pair's money chance / ``temperature``."""
    n = len(n_cands)
    vs = torch.softmax(z, dim=1)[:, 0] / temperature
    vt = target[:, 0] / temperature

    def log_softmax_groups(x):
        top = torch.full((n,), -math.inf, dtype=x.dtype, device=x.device).scatter_reduce(0, dec, x, "amax")
        e = torch.exp(x - top[dec])
        tot = torch.zeros(n, dtype=x.dtype, device=x.device).index_add_(0, dec, e)
        return x - top[dec] - torch.log(tot)[dec]

    lt, ls = log_softmax_groups(vt), log_softmax_groups(vs)
    kl = torch.zeros(n, dtype=z.dtype, device=z.device).index_add_(0, dec, torch.exp(lt) * (lt - ls))
    return kl.mean()


def float_scores(model, c, device, batch=1024):
    """Own-pair value (softmax) [n, C] and the mean loss of the float network."""
    model.eval()
    own = np.full(c.mask.shape, -np.inf)
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(c), batch):
            b = c.take(slice(i, i + batch))
            view, dense, cand, dec, tgt, nc = batch_tensors(b, device)
            z = model(view, dense, cand, dec)
            total += float(loss_of(z, tgt, dec, nc)) * len(b)
            n += len(b)
            p = torch.softmax(z, dim=1)[:, 0].double().cpu().numpy()
            d, k = np.nonzero(b.mask)
            own[i + d, k] = p
    return own, total / max(n, 1)


def write_float(path, model, meta):
    tensors = {}
    for k, (w, b) in enumerate(model.float_layers()):
        tensors[f"w{k}"] = w.astype(np.float32)
        tensors[f"b{k}"] = b.astype(np.float32)
    write_safetensors(path, tensors, meta)


def read_init(path, n_features, widths):
    t, meta = read_safetensors(path)
    n = sum(1 for k in t if k.startswith("w"))
    if n < 2 or set(t) != {f"{p}{k}" for k in range(n) for p in "wb"}:
        raise SystemExit(f"--init {path}: not a float student network")
    if t["w0"].shape[0] != n_features:
        raise SystemExit(f"--init {path}: takes {t['w0'].shape[0]} inputs, the data {n_features}")
    hidden = [int(t[f"w{k}"].shape[1]) for k in range(n - 1)]
    if hidden != list(widths):
        raise SystemExit(f"--init {path}: hidden widths {hidden}, --widths {widths}")
    return [(t[f"w{k}"].astype(np.float64), t[f"b{k}"].astype(np.float64)) for k in range(n)], meta


def shrink_perturb(model, lam, seed, n_features):
    """Ash & Adams 2020 (as ``fortyone_nn.train.shrink_perturb``): every parameter
    ``lam * p + N(0, sigma^2)``, sigma = (1 - lam) / sqrt(3 * fan_in); padding row stays 0."""
    if not (0.0 < lam < 1.0):
        raise SystemExit(f"--shrink-perturb {lam}: lambda must be above 0 and below 1")
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for name, p in model.named_parameters():
            mod = name.rsplit(".", 1)[0]
            fan_in = n_features if mod == "emb" else model.get_submodule(mod).in_features
            sigma = (1.0 - lam) / math.sqrt(3.0 * fan_in)
            p.mul_(lam).add_(torch.randn(tuple(p.shape), generator=g, dtype=torch.float32) * sigma)
        model.emb.weight[model.n_binary].zero_()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--widths", default="256,128")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--val-permille", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--init")
    ap.add_argument("--shrink-perturb", type=float)
    ap.add_argument("--perturb-seed", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--progress")
    ap.add_argument("--rank-weight", type=float, default=0.0)
    ap.add_argument("--rank-temperature", type=float, default=0.01)
    ap.add_argument("--stop-on", default="val_loss", choices=["val_loss", "decision"])
    args = ap.parse_args(argv)
    t0 = time.time()
    progress = Progress(args.progress, t0)
    progress(step="load")
    try:
        widths = parse_widths(args.widths)
    except ValueError as e:
        raise SystemExit(f"--widths: {e}") from None
    if args.rank_weight < 0 or args.rank_temperature <= 0:
        raise SystemExit("--rank-weight must be at least 0 and --rank-temperature above 0")
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    det = deterministic(torch.device(device))

    loaded = [load_features(p) for p in args.train]
    sides = [s for _, s in loaded]
    for p, s in zip(args.train, sides):
        if s.get("role") == "test":
            raise SystemExit(f"{p} is the fixed test set: it is never trained on")
    check_same(sides, args.train)
    side = sides[0]
    n_binary, n_dense = side["n_binary"], side["n_dense"]
    n_features = n_binary + n_dense
    cols = concat([Columns(a, n_binary) for a, _ in loaded])
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(cols):
            raise SystemExit(f"--limit {args.limit}: the training shards hold {len(cols)} decisions")
        cols = cols.take(np.arange(args.limit))
    val_sel = (cols.pos_key % np.uint64(1000)) < np.uint64(args.val_permille)
    train_c, val_c = cols.take(~val_sel), cols.take(val_sel)
    test_c = None
    if args.test:
        ta, ts = load_features(args.test)
        if ts.get("role") != "test":
            raise SystemExit(f"--test {args.test}: not a test-set shard (role {ts.get('role')})")
        check_same([side, ts], [args.train[0], args.test])
        test_c = Columns(ta, n_binary)

    model = StudentNet(n_binary, n_dense, widths)
    init_info = None
    if args.init:
        layers, meta = read_init(args.init, n_features, widths)
        model.set_layers(layers)
        init_info = {"float": args.init, "float_sha256": sha256_file(args.init)}
        if args.shrink_perturb is not None:
            seed = args.perturb_seed if args.perturb_seed is not None else args.seed
            shrink_perturb(model, args.shrink_perturb, seed, n_features)
            init_info["shrink_perturb"] = {"lambda": args.shrink_perturb, "seed": seed}
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    best = (math.inf, None, -1)
    history = []
    bad = 0
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(len(train_c))
        total, n = 0.0, 0
        batches = (len(perm) + args.batch - 1) // args.batch
        progress(step="train", epoch=epoch, epochs=args.epochs, batches_done=0, batches=batches)
        last = time.time()
        for k, i in enumerate(range(0, len(perm), args.batch)):
            if time.time() - last > 20:
                progress(batches_done=k)
                last = time.time()
            b = train_c.take(perm[i : i + args.batch])
            view, dense, cand, dec, tgt, nc = batch_tensors(b, device)
            z = model(view, dense, cand, dec)
            loss = loss_of(z, tgt, dec, nc)
            if args.rank_weight > 0:
                loss = loss + args.rank_weight * rank_term(z, tgt, dec, nc, args.rank_temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
            n += len(b)
        own, vloss = float_scores(model, val_c, device)
        rep = decision_report(own, val_c)
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_loss": vloss,
                        "val_decision_loss_pct": rep["decision_loss_pct"], "val_agreement": rep["agreement"],
                        "minutes": (time.time() - t0) / 60})
        print(f"epoch {epoch}: train {total / max(n, 1):.6f} val {vloss:.6f} "
              f"loss {rep['decision_loss_pct']:.4f}% agree {rep['agreement']:.3f}", flush=True)
        stop_metric = vloss if args.stop_on == "val_loss" else rep["decision_loss_pct"]
        if stop_metric < best[0] - 1e-9:
            best = (stop_metric, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            break
    if best[1] is not None:
        model.load_state_dict(best[1])
    progress(step="quantise", best_epoch=best[2])
    os.makedirs(args.out, exist_ok=True)
    data_meta = {
        "train_shards": json.dumps([os.path.basename(p) for p in args.train]),
        "train_decisions": str(len(cols)),
        "train_sha256": json.dumps([sha256_file(p) for p in args.train]),
        "widths": args.widths,
        "seed": str(args.seed),
        "best_epoch": str(best[2]),
    }
    write_float(os.path.join(args.out, "float.safetensors"), model, {**data_meta, "head": HEAD, "encoder": ENCODER,
                                                                     "encoder_sha256": side["encoder_sha256"]})
    int_layers = quantize(model.float_layers(), n_features=n_features)
    int_path = os.path.join(args.out, "int.safetensors")
    write_student_int(int_path, int_layers, data_meta, side)

    def measure(c, name):
        own_f, l = float_scores(model, c, device)
        own_i = int_scores(int_layers, c, n_binary, n_features)
        return {"set": name, "loss": l, "float": by_groups(own_f, c), "int": by_groups(own_i, c)}

    progress(step="measure")
    report = {
        "settings": vars(args),
        "data": {"decisions": len(cols), "limit": args.limit, "train_rows": len(train_c), "val_rows": len(val_c),
                 "encoder_sha256": side["encoder_sha256"],
                 "teacher": side.get("teacher_stamps")},
        "history": history,
        "best_epoch": best[2],
        "val": measure(val_c, "val"),
        "int_sha256": sha256_file(int_path),
        "device": {"device": device, "deterministic": det, "torch": torch.__version__},
        "init": init_info,
        "seconds": time.time() - t0,
    }
    if test_c is not None:
        report["test"] = measure(test_c, "test")
        if args.init:
            layers, _ = read_init(args.init, n_features, widths)
            old = int_scores(quantize(layers, n_features=n_features), test_c, n_binary, n_features)
            old_loss = decision_report(old, test_c)["decision_loss_pct"]
            new_loss = report["test"]["int"]["all"]["decision_loss_pct"]
            report["keep"] = {"new_test_loss_pct": new_loss, "init_test_loss_pct": old_loss,
                              "keep": bool(new_loss <= old_loss)}
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    progress(step="done", int_sha256=report["int_sha256"])
    v = report["val"]["int"]["all"]
    print(f"int network {report['int_sha256'][:16]}: val decision loss {v['decision_loss_pct']:.4f}% "
          f"agreement {v['agreement']:.3f} ({v['decisions']} decisions)")
    if test_c is not None:
        t = report["test"]["int"]["all"]
        print(f"test: decision loss {t['decision_loss_pct']:.4f}% agreement {t['agreement']:.3f} "
              f"({t['decisions']} decisions)")
    return report


if __name__ == "__main__":
    main()
