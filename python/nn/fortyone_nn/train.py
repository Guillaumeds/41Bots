"""Train an oracle value network O_k (docs/M3-HYBRID-SPEC.md sections 6 and 6a).

    uv run python -m fortyone_nn.train --train A.npy [B.npy ...] --val V.npy --out DIR
        [--head outcome|money4] [--mix-weight 1.0] [--widths 512,256,64] [--epochs 40]
        [--batch 512] [--lr 1e-3] [--patience 4] [--seed 1] [--threads 4] [--max-minutes M]
        [--device cpu|cuda|auto] [--feed auto|resident|staged] [--cuda-graph auto|off]
        [--profile-batches N] [--progress FILE]

A plain MLP: the binary features through an embedding-bag sum (exactly a linear layer on the
one-hot inputs), plus the dense features through a linear layer, ReLU clipped at 64 (as the
integer inference), hidden layers, then the head:

- ``outcome`` (spec 6a, default): one logit per outcome of the trick's vocabulary; the impossible
  outcomes of each position are masked; loss = cross-entropy on the outcome label plus
  ``--mix-weight`` times the cross-entropy of the mixture (the outcomes' exact money vectors
  weighted by the softmax) against the exact four-vector. Value = the mixture.
- ``money4`` (spec 6): four logits, soft cross-entropy against the exact four-vector.

Adam, seeded, early stopping on the validation shard.

Device (``--device``, default ``cpu``): ``cuda`` trains on the GPU (an error if it fails the
checks of ``device.probe``), ``auto`` on the GPU when it passes them, else on the CPU with the
reason printed and recorded. The network is initialised on the CPU under the seed and then
moved, and the batches are the same rows in the same order on every device, so a GPU run differs
from a CPU run only by float rounding (different kernels add in different orders): it is not
bit-identical across devices or GPU models, which is accepted because the quantised integer
network is what is validated and its SHA-256 is recorded (docs/M3-HYBRID-SPEC.md §6). PyTorch's
deterministic switches are set (``device.deterministic``; on a GPU also cuDNN's, and float32
products at full precision, no TF32). The CPU path computes exactly what it computed before the
GPU path existed (same networks, byte for byte).

Data: the training shards are read memory-mapped and converted a block of rows at a time into
one compact array per input (``compact``: 847 bytes a row at trick 4; the full record array is
never held in memory). The blocks are checked (``check_shards``) and converted (``load_compact``)
on ``--load-threads`` threads (numpy lets other threads run while it works on a block; each block
fills its own rows), and the shards' SHA-256 are computed on threads of their own meanwhile: the
same arrays, checks and hashes as one thread gives. Each batch is expanded to the network's inputs (``expand``) on the
training device; the batches are exactly those of the former full-width columns (tested).
``--feed`` (class ``Feed``): ``resident`` keeps the compact rows in the device's memory and
gathers each batch there, so within an epoch the host only queues GPU work and never waits for
it; ``staged`` gathers each batch on the host into page-locked ("pinned") staging buffers and
copies it while the GPU works on the previous batch; ``auto`` (default) is resident when the rows
fit in the GPU's free memory with ``RESERVE`` to spare (always on the CPU). The batch size is
``--batch`` everywhere, so results stay comparable between devices.

Training step (``--cuda-graph``, class ``Step``): on a GPU, by default, the whole step for a full
batch (gathering its rows, forward, loss, backward, Adam, the loss sum) is captured once as a CUDA
graph and replayed, so the host launches one graph a batch instead of about 80 small GPU
operations and a slow host CPU no longer holds the GPU back. The batches, their order, the loss
and Adam's settings are unchanged; Adam runs in its fused, capturable form (``adam``: step counts
on the GPU), which rounds differently. Before training, a trial on copies of the network
(``graph_trial``) checks that replayed steps give the eager steps' losses; if the trial or a
capture fails, the step is eager (as before) and the log, the progress file (``step_mode``) and
the report (``device.train_step``) say why. ``off``, and the CPU: the eager step, exactly the
former loop.

``--progress FILE`` is rewritten (atomically) at each step and about every 20 s within an
epoch: ``step`` (load, train, quantise, int-eval, done), epoch, batches done, the last epoch's
losses, minutes, device, the training step's mode (``step_mode``: ``cuda-graph``, or ``eager``
and why); the runner's heartbeat reads it.

Writes to DIR ``float.safetensors``,
``int.safetensors`` (the quantised network ``crates/nn`` loads, with its head) and
``report.json`` (settings, data hashes, per-epoch losses, and validation metrics of the float and
the integer network: outcome accuracy, and the error of every team's money chance: bias per
team, SD, 99.9th percentile of the absolute error, max; wall-clock seconds of loading, of each
epoch's training loop and of the integer check, each epoch's milliseconds a batch, and a profile
of a few batches (``BatchProfile``: on a GPU by default, ``--profile-batches``; with the CUDA graph
also of a few replayed batches, under ``graphed``); and the device facts of ``device.probe``: GPU
name, driver, PyTorch and CUDA versions, the deterministic settings, the feed's mode and sizes,
and the training step (``train_step``: its mode and why, the trial, captures, replays and eager
steps). The integer check runs the integer network exactly (``intnet.exact_matmul``).
"""

import argparse
import copy
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

from .formats import (
    INPUT_SHIFT,
    N_BINARY,
    N_DENSE,
    ONE,
    PAD,
    VALUE_FAULTS,
    load_shard,
    mask_bits,
    sha256_file,
    value_fault,
    write_safetensors,
)
from .device import deterministic, facts_base, probe
from .intnet import money_rows
from .quantize import quantize, write_int

CLIP_F = 64.0


class Net(nn.Module):
    def __init__(self, widths, n_out):
        super().__init__()
        w0 = widths[0]
        self.emb = nn.EmbeddingBag(N_BINARY + 1, w0, mode="sum", padding_idx=N_BINARY)
        bound = 1.0 / math.sqrt(N_BINARY + N_DENSE)
        with torch.no_grad():
            self.emb.weight.uniform_(-bound, bound)
            self.emb.weight[N_BINARY].zero_()
        self.dense = nn.Linear(N_DENSE, w0)
        self.hidden = nn.ModuleList(nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1))
        self.out = nn.Linear(widths[-1], n_out)

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
        return [(w.detach().cpu().double().numpy(), b.detach().cpu().double().numpy()) for w, b in layers]


def columns(a):
    """Index array (padding -> N_BINARY), dense, target, label, mask, outcome vectors (numpy) of a
    record array or a slice of one."""
    idx = np.asarray(a["active"]).astype(np.int64)
    idx[idx == PAD] = N_BINARY
    dense = np.asarray(a["dense"]).astype(np.float32) / float(1 << INPUT_SHIFT)
    tgt = (np.asarray(a["value"]).astype(np.float64) / float(ONE)).astype(np.float32)
    v = a.dtype["outcomes"].shape[0]
    lab = np.asarray(a["label"]).astype(np.int64)
    mask = mask_bits(a["mask"], v)
    vecs = (np.asarray(a["outcomes"]).astype(np.float64) / float(ONE)).astype(np.float32)
    return idx, dense, tgt, lab, mask, vecs


def tensors(a):
    """``columns`` of a whole record array, as CPU tensors."""
    return [torch.from_numpy(x) for x in columns(a)]


def compact(a):
    """The training columns of a record array (or a slice of one) in compact form: the active
    indices as int16 (the shard's u16 values; the padding 0xFFFF reads -1), dense, target, the
    label as uint8, mask, outcome vectors (dense, target, mask and vectors by ``columns``' own
    expressions). ``expand`` of rows of these is exactly ``columns`` of the same rows."""
    act = np.asarray(a["active"])
    if np.any((act >= N_BINARY) & (act != PAD)):
        raise ValueError(f"an active index outside 0..{N_BINARY - 1} that is not the padding")
    dense = np.asarray(a["dense"]).astype(np.float32) / float(1 << INPUT_SHIFT)
    tgt = (np.asarray(a["value"]).astype(np.float64) / float(ONE)).astype(np.float32)
    v = a.dtype["outcomes"].shape[0]
    lab = np.asarray(a["label"]).astype(np.uint8)
    mask = mask_bits(a["mask"], v)
    vecs = (np.asarray(a["outcomes"]).astype(np.float64) / float(ONE)).astype(np.float32)
    return act.view(np.int16), dense, tgt, lab, mask, vecs


# Rows a block of the loading steps: small enough that a block's temporary arrays (about 50 MB)
# are reused rather than mapped afresh, and that many threads share the work evenly.
LOAD_BLOCK = 1 << 15


def load_threads_default():
    """Threads for loading: the usable CPUs, at most 32 (the loading is bound by memory long
    before, and each thread holds a block's temporaries)."""
    n = os.process_cpu_count() if hasattr(os, "process_cpu_count") else os.cpu_count()
    return max(1, min(n or 1, 32))


def on_threads(fn, jobs, threads):
    """``fn`` of every job, on ``threads`` threads (in order on one); the results in job order. An
    exception in a job is raised here."""
    if threads <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with ThreadPoolExecutor(min(threads, len(jobs))) as ex:
        return list(ex.map(fn, jobs))


def row_blocks(shards, block):
    """(shard index, first row, rows before the shard) of every block of ``block`` rows, shard by
    shard."""
    out = []
    before = 0
    for k, s in enumerate(shards):
        out += [(k, j, before) for j in range(0, len(s), block)]
        before += len(s)
    return out


def check_shards(shards, block=LOAD_BLOCK, threads=1):
    """``check_values`` of each shard in turn (raising the error it raises first, from the first
    shard that has one) and the set of tricks the shards hold, worked out a block of rows at a
    time on ``threads`` threads."""

    def one(job):
        k, j, _ = job
        a = shards[k][j : j + block]
        return value_fault(a), set(np.unique(a["trick"]).tolist())

    jobs = row_blocks(shards, block)
    res = on_threads(one, jobs, threads)
    for k in range(len(shards)):
        faults = [f for (kk, _, _), (f, _) in zip(jobs, res) if kk == k and f is not None]
        if faults:
            raise ValueError(VALUE_FAULTS[min(faults)])
    tricks = set()
    for _, t in res:
        tricks |= t
    return tricks


def load_compact(shards, block=LOAD_BLOCK, threads=1):
    """``compact`` of the shards one after another (as of one concatenated array), built a block
    of rows at a time from the memory-mapped shards (the full record array is never held in
    memory) on ``threads`` threads, each block into its own rows, as numpy arrays: 847 bytes a row
    at trick 4 (``columns`` takes 1430). The same arrays for any block size and thread count."""
    n = sum(len(s) for s in shards)
    first = compact(shards[0][:1])
    out = [np.empty((n,) + c.shape[1:], dtype=c.dtype) for c in first]

    def fill(job):
        k, j, before = job
        part = compact(shards[k][j : j + block])
        at = before + j
        for o, p in zip(out, part):
            o[at : at + len(p)] = p

    on_threads(fill, row_blocks(shards, block), threads)
    return out


def expand(act, dense, tgt, lab, mask, vecs):
    """Rows of the compact columns (tensors, on any device) -> exactly ``columns`` of those rows:
    the active indices as int64 with the padding (-1) -> N_BINARY, the labels as int64."""
    idx = act.to(torch.int64)
    idx.masked_fill_(idx < 0, N_BINARY)
    return [idx, dense, tgt, lab.to(torch.int64), mask, vecs]


# GPU memory kept free beside resident training columns (this, or a tenth of the GPU if more):
# the network, Adam's state, a batch's activations, a validation batch, cuBLAS's workspaces.
RESERVE = 1 << 30


class Feed:
    """The training batches on the training device: ``feed(b)`` is ``columns`` of the rows ``b``,
    a slice of ``feed.order(perm)`` (the epoch's permutation). Two ways, the same rows:

    - ``resident`` (always on the CPU; on a GPU whose free memory holds the compact columns with
      ``RESERVE`` to spare): the compact columns live on the device, the epoch's permutation is
      copied there once, and each batch is gathered (``index_select``) and expanded there. No
      host work, no host-to-device copy and nothing that makes the host wait for the GPU within
      an epoch, so the host queues work ahead while the GPU computes.
    - ``staged`` (a GPU too small for that, or ``mode="staged"``): the compact columns stay on the
      host; each batch's rows are gathered there into page-locked staging buffers (two sets, used
      in turn) and copied asynchronously, so a copy overlaps the GPU's work on the previous batch
      (a set is reused only once its last copy has finished); then expanded on the device.
    """

    def __init__(self, comp, device, batch, mode="auto"):
        if mode not in ("auto", "resident", "staged"):
            raise ValueError(f"feed {mode!r}: auto, resident or staged")
        self.device = device
        self.cuda = device.type == "cuda"
        need = sum(int(c.nbytes) for c in comp)
        self.facts = {"rows": int(len(comp[0])), "bytes": need}
        fits = True
        if self.cuda:
            free, total = torch.cuda.mem_get_info(device)
            spare = max(RESERVE, total // 10)
            self.facts.update(gpu_free_bytes=int(free), gpu_total_bytes=int(total), reserve_bytes=int(spare))
            fits = need + spare <= free
        self.resident = fits if mode == "auto" else mode == "resident"
        self.facts["mode"] = "resident" if self.resident else "staged"
        if self.resident:
            self.cols = [torch.from_numpy(c).to(device) for c in comp]
            return
        self.cols = [torch.from_numpy(c) for c in comp]

        def buffer(c):
            t = torch.empty((batch,) + tuple(c.shape[1:]), dtype=c.dtype)
            return t.pin_memory() if self.cuda else t

        self.stage = [[buffer(c) for c in self.cols] for _ in range(2)]
        self.copied = [None, None]
        self.turn = 0

    def order(self, perm):
        """The epoch's permutation (an int64 CPU tensor), where ``__call__``'s slices come from."""
        return perm.to(self.device) if self.resident else perm

    def __call__(self, b):
        if self.resident:
            return expand(*(c.index_select(0, b) for c in self.cols))
        k, rows = self._staged(b)
        out = [s.to(self.device, non_blocking=True) for s in rows]
        self._copied(k)
        return expand(*out)

    def _staged(self, b):
        """The rows ``b`` gathered on the host into the next set of staging buffers, once that
        set's last copy has finished: (the set, the filled slices)."""
        k = self.turn
        self.turn ^= 1
        if self.copied[k] is not None:
            self.copied[k].synchronize()
        n = len(b)
        for c, s in zip(self.cols, self.stage[k]):
            torch.index_select(c, 0, b, out=s[:n])
        return k, [s[:n] for s in self.stage[k]]

    def _copied(self, k):
        """Marks the copies from staging set ``k`` just queued (the set is reused once they end)."""
        if self.cuda:
            ev = torch.cuda.Event()
            ev.record()
            self.copied[k] = ev

    # The CUDA-graph step (class ``Step``) reads its batch from fixed buffers on the device.

    def inputs(self, batch):
        """Fixed device buffers for a batch of ``batch`` rows, which a captured step reads: the
        row indices (resident) or the compact rows (staged)."""
        if self.resident:
            return [torch.zeros(batch, dtype=torch.int64, device=self.device)]
        return [torch.zeros(s.shape, dtype=s.dtype, device=self.device) for s in self.stage[0]]

    def put(self, b, bufs):
        """Fills ``inputs``' buffers with the full batch ``b`` (queued on the current stream)."""
        if self.resident:
            bufs[0].copy_(b, non_blocking=True)
            return
        k, rows = self._staged(b)
        for d, s in zip(bufs, rows):
            d.copy_(s, non_blocking=True)
        self._copied(k)

    def read(self, bufs):
        """``columns`` of the batch in ``inputs``' buffers: what ``self(b)`` gives after
        ``put(b, bufs)``."""
        if self.resident:
            return expand(*(c.index_select(0, bufs[0]) for c in self.cols))
        return expand(*bufs)


class BatchProfile:
    """Where a training batch's time goes, measured on ``batches`` batches of the first epoch
    (from its 9th, when it has enough) with PyTorch's profiler; on a GPU with its kernel and copy
    times (CUPTI): the device's busy milliseconds a batch, the host's milliseconds a batch over
    the window (the profiler adds a little), and the operations that keep the device (and the
    host) busiest. It only observes: the training is the same with or without it. A failure is
    recorded in the report, never raised. With the CUDA-graph step a second window (``start``)
    profiles replayed steps, after the first, which profiles eager steps."""

    def __init__(self, batches, steps, device, start=None):
        if start is None:
            self.n = max(0, min(batches, steps))
            self.start = 8 if steps >= 8 + self.n else 0
        else:  # a window from batch ``start`` among the first ``steps``
            self.n = max(0, min(batches, steps - start))
            self.start = start
        # Batches before this one are the profile's or before it (0 when there is no profile).
        self.end = self.start + self.n if self.n else 0
        self.device, self.prof, self.result = device, None, None

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def before(self, k):
        if not self.n or k != self.start:
            return
        try:
            acts = [torch.profiler.ProfilerActivity.CPU]
            if self.device.type == "cuda":
                acts.append(torch.profiler.ProfilerActivity.CUDA)
            self.sync()
            self.prof = torch.profiler.profile(activities=acts)
            self.prof.start()
            self.t = time.perf_counter()
        except Exception as e:  # noqa: BLE001 (observing must never stop the training)
            self.prof, self.result = None, {"error": f"{type(e).__name__}: {str(e)[:300]}"}

    def after(self, k):
        if self.prof is None or k != self.start + self.n - 1:
            return
        try:
            self.sync()
            wall = time.perf_counter() - self.t
            self.prof.stop()
            self.result = self.summary(wall)
        except Exception as e:  # noqa: BLE001
            self.result = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        self.prof = None

    def summary(self, wall):
        def device_us(e):
            for name in ("self_device_time_total", "self_cuda_time_total"):
                v = getattr(e, name, None)
                if v is not None:
                    return float(v)
            return 0.0

        ev = list(self.prof.key_averages())
        on_device = [e for e in ev if "CUDA" in str(getattr(e, "device_type", ""))]
        on_host = [e for e in ev if "CUDA" not in str(getattr(e, "device_type", ""))]
        n = self.n

        def top(rows):
            rows = sorted((r for r in rows if r[0] > 0), key=lambda r: -r[0])[:12]
            return [{"op": key[:100], "ms": round(us / 1000 / n, 3), "calls": round(c / n, 2)} for us, c, key in rows]

        return {
            "batches": n,
            "first_batch": self.start,
            "host_ms_per_batch": round(1000 * wall / n, 3),
            "device_events": len(on_device),
            "device_busy_ms_per_batch": round(sum(device_us(e) for e in on_device) / 1000 / n, 3),
            # A kernel's time goes to the innermost operation that launched it.
            "top_device_ms_per_batch": top((device_us(e), e.count, e.key) for e in on_host),
            "top_host_ms_per_batch": top((float(e.self_cpu_time_total), e.count, e.key) for e in on_host),
        }


class Progress:
    """The ``--progress`` file: rewritten atomically (a temporary file, then a rename)."""

    def __init__(self, path, t0, device):
        self.path, self.t0, self.state = path, t0, {"device": device}

    def __call__(self, **kv):
        self.state.update(kv)
        self.state["minutes"] = round((time.time() - self.t0) / 60, 3)
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.path)


def outcome_terms(z, mask, vecs):
    """Masked log-probabilities of the outcomes and the mixture (the value)."""
    logp = torch.log_softmax(z.masked_fill(~mask, -1e9), dim=1)
    mix = torch.einsum("bv,bvt->bt", logp.exp(), vecs)
    return logp, mix


def loss_of(head, z, tgt, lab, mask, vecs, mix_weight):
    if head == "money4":
        return -(tgt * torch.log_softmax(z, dim=1)).sum(dim=1).mean()
    logp, mix = outcome_terms(z, mask, vecs)
    ce = -logp.gather(1, lab[:, None]).mean()
    soft = -(tgt * torch.log(mix.clamp_min(1e-12))).sum(dim=1).mean()
    return ce + mix_weight * soft


# Eager steps (on a side stream) before each capture of the training step, as PyTorch's CUDA-graph
# documentation does: lazy initialisations (Adam's state, library workspaces) happen outside it.
WARMUP = 3
# The trial's largest relative difference between the losses of replayed and eager steps.
TRIAL_TOLERANCE = 1e-5


def adam(params, lr, graphed):
    """Adam as before (PyTorch's default form: ``foreach`` kernels on a GPU, step counts and bias
    corrections on the host), or for the CUDA-graph step its ``fused`` form with ``capturable``
    (step counts and bias corrections on the GPU, as a captured step needs): the same update with
    the same settings, rounded differently."""
    if not graphed:
        return torch.optim.Adam(params, lr=lr)
    opt = torch.optim.Adam(params, lr=lr, fused=True, capturable=True)
    # Its eager steps (warm-up, the last partial batch, after a failed capture) are intended.
    opt._warned_capturable_if_run_uncaptured = True
    return opt


class CudaGraph:
    """``fn()`` captured as a CUDA graph on ``stream``: capturing records the GPU work without
    running any of it; ``replay()`` runs it all on the current stream with one launch; ``out`` is
    what ``fn`` returned (in the graph's memory, rewritten by every replay)."""

    def __init__(self, fn, stream):
        self.g = torch.cuda.CUDAGraph()
        # The outer context puts the current stream back even if ending the capture raises
        # (torch.cuda.graph then leaves its capture stream current).
        with torch.cuda.stream(torch.cuda.current_stream()), torch.cuda.graph(self.g, stream=stream):
            self.out = fn()

    def replay(self):
        self.g.replay()


class Replayed:
    """``CudaGraph``'s stand-in without a GPU (the tests): capturing runs nothing, each replay runs
    ``fn`` again."""

    def __init__(self, fn, stream):
        self.fn, self.out = fn, None

    def replay(self):
        self.out = self.fn()


# The devices where the trainer uses the CUDA-graph step, and its graph (the tests put the CPU
# and ``Replayed`` here to run the trainer's graph path without a GPU).
GRAPH_DEVICES = ("cuda",)
GRAPH_CLASS = CudaGraph


class Step:
    """One training step on the rows ``b`` (a slice of ``feed.order``'s permutation): forward,
    loss, backward, Adam, and the loss added to ``total`` (the epoch's sum, float64 on the device;
    each term the batch's loss times its rows, as before). ``loss`` is the last step's loss.

    - Eager (the CPU; ``--cuda-graph off``; a graph that failed): exactly the former loop's
      operations, one kernel launch each (about 80 on a GPU).
    - CUDA graph (``graphed``): the whole step for a full batch (gathering its rows from the
      resident columns or reading the staged ones, expanding them, forward, loss, backward, Adam,
      the loss sum) is captured once and replayed: per batch the host only copies the batch's
      row indices (or staged rows) into the graph's input buffers (``Feed.put``) and launches the
      graph. So the host's speed no longer limits the GPU. Before a capture, at least ``WARMUP``
      steps, and every step up to ``not_before`` (the eager batch profile), run eagerly on a side
      stream. Adam's learning rate is a constant of the captured update, so a new rate (the
      plateau scheduler) means a new warm-up and capture. A partial batch (an epoch's last) runs
      eagerly. Replayed steps compute what eager steps compute on the same rows in the same
      order; only the kernels' rounding may differ (``graph_trial`` checks it before training).
      A capture that fails runs nothing, so the network and Adam are untouched: the training
      goes on eagerly from that batch, ``reason`` says why and ``on_fail(reason)`` is called.
    """

    def __init__(self, model, opt, head, mix_weight, feed, batch, graphed=False, not_before=0,
                 graph_class=CudaGraph, on_fail=None):
        self.model, self.opt, self.head, self.mix_weight = model, opt, head, mix_weight
        self.feed, self.batch, self.device = feed, batch, feed.device
        self.total = torch.zeros((), dtype=torch.float64, device=self.device)
        self.loss = None
        self.graphed, self.reason = graphed, None
        self.not_before, self.graph_class, self.on_fail = not_before, graph_class, on_fail
        self.graph, self.graph_lr, self.warm, self.calls = None, None, 0, 0
        self.counts = {"captures": 0, "replays": 0, "eager_steps": 0}
        self.side = self.capture_stream = None
        if graphed:
            if self.device.type == "cuda":
                self.side = torch.cuda.Stream(self.device)
                self.capture_stream = torch.cuda.Stream(self.device)
            self.bufs = feed.inputs(batch)

    def describe(self):
        return "cuda-graph" if self.graphed else f"eager ({self.reason})" if self.reason else "eager"

    def __call__(self, b):
        self.calls += 1
        if not self.graphed or len(b) != self.batch:
            return self.eager(b)
        if self.graph is not None and self.opt.param_groups[0]["lr"] != self.graph_lr:
            self.graph, self.warm = None, 0  # a new learning rate: warm up and capture again
        if self.graph is None:
            if self.warm < WARMUP or self.calls <= self.not_before:
                self.warm += 1
                return self.eager(b, side=True)
            if not self.capture():
                return self.eager(b)
        self.feed.put(b, self.bufs)
        self.graph.replay()
        self.loss = self.graph.out
        self.counts["replays"] += 1

    def eager(self, b, side=False):
        if not (side and self.side is not None):
            return self._eager(b)
        here = torch.cuda.current_stream(self.device)
        self.side.wait_stream(here)
        with torch.cuda.stream(self.side):
            self._eager(b)
        here.wait_stream(self.side)

    def _eager(self, b):
        idx, dense, tgt, lab, mask, vecs = self.feed(b)
        z = self.model(idx, dense)
        loss = loss_of(self.head, z, tgt, lab, mask, vecs, self.mix_weight)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.total += loss.detach().double() * len(b)
        self.loss = loss.detach()
        self.counts["eager_steps"] += 1

    def _captured(self):
        """The step on the batch in the input buffers, as captured. The gradients are made by the
        captured backward in the graph's memory and rewritten (not added to) by every replay."""
        self.opt.zero_grad(set_to_none=True)
        idx, dense, tgt, lab, mask, vecs = self.feed.read(self.bufs)
        z = self.model(idx, dense)
        loss = loss_of(self.head, z, tgt, lab, mask, vecs, self.mix_weight)
        loss.backward()
        self.opt.step()
        self.total += loss.detach().double() * self.batch
        return loss.detach()

    def capture(self):
        """Captures the step; False (and the eager step from now on) if that fails."""
        lr = self.opt.param_groups[0]["lr"]
        try:
            g = self.graph_class(self._captured, self.capture_stream)
        except Exception as e:  # noqa: BLE001 (a failed capture must not stop the training)
            self.opt.zero_grad(set_to_none=True)  # gradients made while capturing hold nothing
            self.graphed, self.graph = False, None
            self.reason = f"capturing the step failed: {type(e).__name__}: {str(e)[:300]}"
            if self.on_fail:
                self.on_fail(self.reason)
            return False
        self.graph, self.graph_lr = g, lr
        self.counts["captures"] += 1
        return True


def graph_trial(model, head, mix_weight, feed, batch, lr, graph_class=CudaGraph, steps=2 * WARMUP):
    """The CUDA-graph step tried before training, on two copies of the network (the network
    itself is not touched): ``steps`` full batches (rows 0, 1, ... cycling) through ``Step``
    (``WARMUP`` eager steps, a capture, replays) on one copy and all eager on the other, with the
    same Adam. Returns (None, facts) when every loss agrees within ``TRIAL_TOLERANCE`` (relative),
    else (the reason, facts)."""
    t = time.time()
    n = feed.facts["rows"]
    rows = [feed.order(torch.arange(j * batch, (j + 1) * batch) % n) for j in range(steps)]
    losses, gst = [], None
    facts = {"steps": steps}
    try:
        for g in (False, True):
            m = copy.deepcopy(model)
            st = Step(m, adam(m.parameters(), lr, True), head, mix_weight, feed, batch, graphed=g,
                      graph_class=graph_class)
            ls = []
            for b in rows:
                st(b)
                ls.append(float(st.loss))
            losses.append(ls)
            gst = st
    except Exception as e:  # noqa: BLE001 (a failed trial means: train eagerly)
        facts["seconds"] = round(time.time() - t, 3)
        return f"the trial failed: {type(e).__name__}: {str(e)[:300]}", facts
    facts["seconds"] = round(time.time() - t, 3)
    facts.update(gst.counts)
    if gst.reason:
        return gst.reason, facts
    eager, replayed = losses
    diff = max(abs(a - b) / max(abs(a), 1e-6) for a, b in zip(eager, replayed))
    facts["max_rel_loss_difference"] = diff
    if not all(math.isfinite(x) for x in eager + replayed):
        return f"the trial's losses are not finite: {replayed}", facts
    if gst.counts["replays"] != steps - WARMUP or not diff <= TRIAL_TOLERANCE:
        return (f"the replayed steps' losses differ from the eager steps' by {diff:.3g} (relative; "
                f"tolerance {TRIAL_TOLERANCE}; {gst.counts['replays']} replays)"), facts
    return None, facts


def error_stats(pred, target):
    """Errors of every team's money chance (fractions; team 0 is the leader's): bias per team,
    and over all teams the SD, mean and 99.9th percentile of the absolute error, and the max."""
    e = (pred.astype(np.float64) - target.astype(np.float64)) / ONE
    a = np.abs(e)
    return {
        "bias_by_team": [float(x) for x in e.mean(axis=0)],
        "sd": float(e.std()),
        "mean_abs": float(a.mean()),
        "p999_abs": float(np.quantile(a, 0.999)),
        "max_abs": float(a.max()),
    }


def evaluate(model, head, data, mix_weight, batch=8192, device=None):
    """(mean loss, money values [n, 4] in units of 2^-32, outcome accuracy) on ``data`` (CPU
    tensors, moved to ``device`` a batch at a time)."""
    device = device or torch.device("cpu")
    model.eval()
    losses, vals, hits = [], [], 0
    n = len(data[0])
    with torch.no_grad():
        for i in range(0, n, batch):
            idx, dense, tgt, lab, mask, vecs = (x[i : i + batch].to(device) for x in data)
            z = model(idx, dense)
            losses.append(float(loss_of(head, z, tgt, lab, mask, vecs, mix_weight)) * len(z))
            if head == "money4":
                vals.append(torch.softmax(z.double(), dim=1).cpu())
            else:
                logp, _ = outcome_terms(z.double(), mask, vecs.double())
                vals.append(torch.einsum("bv,bvt->bt", logp.exp(), vecs.double()).cpu())
                hits += int((logp.argmax(dim=1) == lab).sum())
    v = torch.cat(vals).numpy() * ONE
    return sum(losses) / n, v, hits / n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", choices=["outcome", "money4"], default="outcome")
    ap.add_argument("--mix-weight", type=float, default=1.0, help="outcome head: weight of the mixture loss")
    ap.add_argument("--widths", default="512,256,64")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--load-threads", type=int, default=0,
                    help="threads that check and convert the shards (0: the usable CPUs, at most 32)")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="stop training after this long (0 = no limit)")
    ap.add_argument("--int-eval-rows", type=int, default=20000, help="validation rows for the integer network")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu",
                    help="cpu (default), cuda (the GPU, or an error), auto (the GPU if it passes the checks, else the CPU)")
    ap.add_argument("--progress", help="rewrite this JSON file with the training's progress")
    ap.add_argument("--feed", choices=["auto", "resident", "staged"], default="auto",
                    help="where the training rows live on a GPU (class Feed): auto (resident if they fit), resident, staged")
    ap.add_argument("--profile-batches", type=int, default=None,
                    help="profile this many batches of the first epoch (class BatchProfile; default 16 on a GPU, 0 on the CPU)")
    ap.add_argument("--cuda-graph", choices=["auto", "off"], default="auto",
                    help="the training step on a GPU (class Step): auto (captured as a CUDA graph and replayed, "
                         "eager if a trial or a capture fails, with the reason), off (eager); eager on the CPU")
    args = ap.parse_args(argv)

    t0 = time.time()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    widths = [int(x) for x in args.widths.split(",")]
    os.makedirs(args.out, exist_ok=True)
    if args.device == "cpu":
        facts = facts_base()
        facts.update(want="cpu", device="cpu", reason="the CPU was asked for")
    else:
        facts = probe(args.device)
        if args.device == "cuda" and facts["device"] != "cuda":
            raise SystemExit(f"--device cuda: no usable GPU: {facts['reason']}")
        if facts["device"] != "cuda":
            print(f"no usable GPU ({facts['reason']}): training on the CPU with {args.threads} threads", flush=True)
    device = torch.device(facts["device"])
    # Seeded again: the probe's test step drew from the generator. Seeding a fresh generator
    # twice gives the same state, so the CPU path is unchanged; manual_seed seeds every GPU too.
    torch.manual_seed(args.seed)
    facts["deterministic"] = deterministic(device)
    if device.type == "cuda":
        print(f"training on {facts['gpu_name']} (compute capability {facts['capability']}, "
              f"torch {facts['torch']}, CUDA {facts['torch_cuda']})", flush=True)
    progress = Progress(args.progress, t0, facts["device"])
    progress(step="load")

    load_threads = args.load_threads if args.load_threads > 0 else load_threads_default()
    shards = [load_shard(p) for p in args.train]
    val = np.asarray(load_shard(args.val))
    # The files' SHA-256 (for the report and the network's metadata), each on a thread of its own
    # while the rows are checked and converted (hashlib lets other threads run as it hashes).
    hashing = ThreadPoolExecutor(len(args.train) + 1)
    train_sha = [hashing.submit(sha256_file, p) for p in args.train]
    val_sha = hashing.submit(sha256_file, args.val)
    tricks = check_shards(shards + [val], threads=load_threads)
    if len(tricks) != 1:
        raise SystemExit(f"shards hold tricks {sorted(tricks)}: one frontier trick per network")
    trick = tricks.pop()
    v = shards[0].dtype["outcomes"].shape[0]
    n_out = v if args.head == "outcome" else 4
    head_name = "outcome-v1" if args.head == "outcome" else "money4"
    n = sum(len(sh) for sh in shards)
    print(
        f"train {n} rows, val {len(val)} rows, trick {trick}, head {head_name} "
        f"({n_out} outputs), widths {widths}, device {device.type}",
        flush=True,
    )

    comp = load_compact(shards, threads=load_threads)
    del shards
    hashes = {"train": [f.result() for f in train_sha], "val": val_sha.result()}
    hashing.shutdown()
    va = tensors(val)
    feed = Feed(comp, device, args.batch, args.feed)
    del comp
    facts["feed"] = feed.facts
    print(f"training rows {feed.facts['mode']} ({feed.facts['bytes'] / 2**20:.0f} MiB compact)"
          + (f", GPU free {feed.facts['gpu_free_bytes'] / 2**20:.0f} MiB" if device.type == "cuda" else ""),
          flush=True)
    # Wall-clock seconds: loading, each epoch's training loop (not its validation), the integer
    # check; and each epoch's milliseconds per batch.
    seconds = {"load": round(time.time() - t0, 3), "epochs": [], "ms_per_batch": []}
    # Initialised on the CPU under the seed (the same weights on every device), then moved.
    model = Net(widths, n_out).to(device)
    # The training step (class Step): a CUDA graph on a GPU unless --cuda-graph off, once a trial
    # on copies of the network agrees with eager steps; else eager, as before.
    graphed, why, trial = False, "the CPU", None
    if device.type in GRAPH_DEVICES:
        if args.cuda_graph == "off":
            why = "--cuda-graph off"
        elif n < args.batch:
            why = f"fewer training rows ({n}) than a batch ({args.batch})"
        else:
            why, trial = graph_trial(model, args.head, args.mix_weight, feed, args.batch, args.lr, GRAPH_CLASS)
            graphed = why is None
    opt = adam(model.parameters(), args.lr, graphed)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=1)
    best = (math.inf, None, -1)
    history = []
    steps = (n + args.batch - 1) // args.batch
    last_note = time.time()
    nprof = args.profile_batches if args.profile_batches is not None else (16 if device.type == "cuda" else 0)
    prof = BatchProfile(nprof, steps, device)
    def fell_back(reason):
        print(f"training step: eager from now on ({reason})", flush=True)
        progress(step_mode=step.describe())

    # The first capture waits for the warm-up and the eager profile; replays are profiled from
    # the capture's third (the first replays also load the graph), among the full batches.
    step = Step(model, opt, args.head, args.mix_weight, feed, args.batch, graphed=graphed, not_before=prof.end,
                graph_class=GRAPH_CLASS, on_fail=fell_back)
    gprof = BatchProfile(nprof, n // args.batch, device, start=max(WARMUP, prof.end) + 2) if graphed else None
    step_facts = {"cuda_graph": args.cuda_graph, "mode": "cuda-graph" if graphed else "eager",
                  "reason": why, "trial": trial, "warmup": WARMUP,
                  "optimizer": "Adam(fused=True, capturable=True)" if graphed else "Adam (default form)"}
    facts["train_step"] = step_facts
    if not graphed:
        step.reason = why
    print(f"training step: {step.describe()}" + (f"; trial {json.dumps(trial)}" if trial else ""), flush=True)
    progress(step="train", epoch=0, epochs=args.epochs, batches_done=0, batches=steps, step_mode=step.describe())
    for epoch in range(args.epochs):
        model.train()
        te = time.time()
        order = feed.order(torch.from_numpy(rng.permutation(n)))
        # The epoch's loss sum in float64 on the device (each term float(loss) * rows, exactly the
        # Python float sum), so the GPU is not stopped to read every batch's loss.
        step.total.zero_()
        for k, i in enumerate(range(0, n, args.batch)):
            if epoch == 0:
                prof.before(k)
                if gprof:
                    gprof.before(k)
            step(order[i : i + args.batch])
            if epoch == 0:
                prof.after(k)
                if gprof:
                    gprof.after(k)
            if k % 64 == 0 and time.time() - last_note >= 20:
                last_note = time.time()
                progress(step="train", epoch=epoch, epochs=args.epochs, batches_done=k, batches=steps,
                         step_mode=step.describe())
        total = float(step.total)
        seconds["epochs"].append(round(time.time() - te, 3))
        seconds["ms_per_batch"].append(round(1000 * seconds["epochs"][-1] / steps, 3))
        if epoch == 0 and prof.result is not None:
            seconds["profile"] = dict(prof.result, step="eager")
            if gprof and gprof.result is not None:
                seconds["profile"]["graphed"] = dict(gprof.result, step=step.describe())
            for name, p in (("batch profile", prof.result), ("batch profile, replayed", gprof and gprof.result)):
                if p:
                    print(f"{name}: {json.dumps({k: x for k, x in p.items() if not k.startswith('top')})}", flush=True)
        vloss, vv, acc = evaluate(model, args.head, va, args.mix_weight, device=device)
        sched.step(vloss)
        sd = float(((vv - val["value"].astype(np.float64)) / ONE).std())
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": vloss, "val_acc": acc,
                        "val_sd": sd, "lr": opt.param_groups[0]["lr"], "minutes": (time.time() - t0) / 60})
        print(json.dumps(history[-1]), flush=True)
        progress(step="train", epoch=epoch, epochs=args.epochs, batches_done=steps, batches=steps,
                 last=history[-1], best_epoch=best[2] if vloss >= best[0] else epoch, step_mode=step.describe())
        if vloss < best[0]:
            best = (vloss, {k: x.clone() for k, x in model.state_dict().items()}, epoch)
        elif epoch - best[2] >= args.patience:
            break
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            print("time limit reached", flush=True)
            break
    model.load_state_dict(best[1])
    step_facts.update(step.counts)
    if graphed and not step.graphed:  # a capture failed during the training: eager from then on
        step_facts.update(mode="eager", reason=step.reason)
    vloss, pf, acc = evaluate(model, args.head, va, args.mix_weight, device=device)
    progress(step="quantise", best_epoch=best[2])

    layers = model.float_layers()
    meta = {
        "widths": json.dumps(widths),
        "train_sha256": json.dumps(hashes["train"]),
        "val_sha256": hashes["val"],
        "train_rows": str(n),
        "seed": str(args.seed),
        "best_epoch": str(best[2]),
        "val_loss": repr(vloss),
        "mix_weight": repr(args.mix_weight),
        "torch": torch.__version__,
    }
    ft = {}
    for k, (w, b) in enumerate(layers):
        ft[f"w{k}"] = w.astype(np.float32)
        ft[f"b{k}"] = b.astype(np.float32)
    write_safetensors(os.path.join(args.out, "float.safetensors"), ft, {**meta, "trick": str(trick), "head": head_name})
    ints = quantize(layers)
    int_path = os.path.join(args.out, "int.safetensors")
    write_int(int_path, ints, trick, meta, head=head_name)

    progress(step="int-eval")
    ti = time.time()
    m = min(args.int_eval_rows, len(val))
    if args.head == "outcome":
        pi = money_rows(ints, val["active"][:m], val["dense"][:m], mask=val["mask"][:m], outcomes=val["outcomes"][:m])
    else:
        pi = money_rows(ints, val["active"][:m], val["dense"][:m])
    pi = pi.astype(np.float64)
    seconds["int_eval"] = round(time.time() - ti, 3)
    report = {
        "args": vars(args),
        "trick": trick,
        "head": head_name,
        "outcomes": v,
        "data": hashes,
        "history": history,
        "best_epoch": best[2],
        "float": {"val_loss": vloss, "outcome_accuracy": acc, "error": error_stats(pf, val["value"])},
        "int": {
            "rows": m,
            "error": error_stats(pi, val["value"][:m]),
            "vs_float": error_stats(pi, pf[:m]),
            "sha256": sha256_file(int_path),
            "shifts": [int(s) for _, _, s in ints],
        },
        "minutes": (time.time() - t0) / 60,
        "seconds": seconds,
        "device": facts,
    }
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    progress(step="done", best_epoch=best[2], int_sha256=report["int"]["sha256"])
    print(json.dumps({k: report[k] for k in ("float", "int", "minutes")}, indent=1))


if __name__ == "__main__":
    main()
