# Running a hybrid rung on a rented machine (vast.ai, one GPU)

Branch `m3-hybrid-vtrain`, 2026-10-16 (Mauritius). Spec: `docs/M3-HYBRID-SPEC.md`. The rung-1 run on
GitHub (`.github/workflows/hybrid-rung1.yml`, run 38868893704) spent about 5 minutes on 4M labels,
2 minutes on the trick-3 golden set and up to 200 minutes training on 4 CPU cores. The same rung now
runs as one chunk of the existing runner on one rented machine with many CPU threads and one GPU:
labels, golden set, training on the GPU, the hybrid check, and every result uploaded to R2.
Nothing here has been rented yet: each run is priced and approved by the owner first.

## In one paragraph

A manifest (a small file saying what to run) asks for a `hybrid_rung` chunk and for a machine with
one GPU. The launcher rents such a machine (on-demand only, verified host, the whole machine, the
dollar cap, as for every run). On the machine the runner: gets the equity table; installs the
training software (PyTorch 2.14.1 for CUDA 12.6, pinned by hash, through `uv`, which the release
bundle carries); checks the GPU (PyTorch must see it, support its generation, and run one tiny
training step); makes the labels with every thread; builds the golden set (exact values); trains
the network on the GPU; turns it into the whole-number network; runs `search hybrid-check`; uploads
the reports and networks; and destroys the machine. If the GPU check fails, it trains on the CPU
with every thread instead and says so in the heartbeat, the launcher's log and the report (never in
silence).

## How to launch

One-time, for each new commit of this branch: run `release-binaries.yml` on it (Actions → release-
binaries → Run workflow, branch `m3-hybrid-vtrain`). The bundle it uploads now also carries the
training code (`python/nn`: its source, `pyproject.toml`, `uv.lock`) and `uv` 0.13.0 (PLAN §11.2,
checked against its published SHA-256), all hash-checked by the boot script.

The image changes (owner, 2026-10-16): it now carries `uv` 0.13.0 (checksum-verified) and a uv cache
with Python 3.14 and the whole cu126 group of `python/nn/uv.lock` (torch 2.14.1+cu126, by hash)
already in it, so `uv sync --locked` on the machine installs from disk in seconds instead of
downloading about 3.7 GB (`UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` are fixed in the image's ENV).
If the lock used by a run differs from the one baked in, uv downloads only what differs; the run
still works. Rebuild the image (Actions → image) whenever `python/nn/uv.lock` changes. Image tag:
`ghcr.io/guillaumeds/41bots-base:20261016-2bb16f03f96d` (digest `sha256:c77a2f5f84ddf128bceedc0c19e35a1cc5b111ec0fa636dcd4fe81ecdcdb65ae`, built at 2bb16f0). Size: the uv cache alone is 6.3 GB on disk (wheels plus their unpacked copies), so expect about 4-6 GB compressed (the old image was under 1 GB); the
vast.ai pull is not billed. Its `nvidia/cuda:12.6.3-base` base is enough:
the PyTorch cu126 packages bring their own CUDA libraries (cuBLAS, cuDNN and the rest, about 3.7 GB
downloaded per run), and only the GPU driver comes from the host, which the launcher's filter
`cuda_vers>=12.6` guarantees is new enough. `uv` installs Python 3.14 and the packages as the
image's normal user; nothing needs root. The locked PyTorch build (`torch-2.14.1+cu126`, SHA-256
`eb7564fd…`) is compiled for GPU generations sm_50, 60, 70, 75, 80, 86 and 90 (read from its
`libtorch_cuda.so` on this VM): the RTX A2000 (8.6) is covered, as are Pascal, Volta, Turing, Ada
(8.9 runs the 8.6 code) and Hopper; Blackwell (10.0 and 12.0) is not, hence the launcher's
`compute_cap<1000`. (Checked by reasoning and on this VM without a GPU; the smoke run below is the
first real test.)

Then Actions → vast-run → Run workflow, first with `dry_run` ticked (it only prints the machine it
would rent), then without:

| input | smoke run | full run |
|---|---|---|
| manifest | `docs/vast/m3-hybrid-smoke.manifest.json` | `docs/vast/m3-hybrid-rung1b.manifest.json` |
| image | `ghcr.io/guillaumeds/41bots-base:20261016-2bb16f03f96d` | same |
| bin_sha | the branch commit released above | same |
| max_dollars | 2 | 6 |
| offer_id | the offer the owner picked (or blank: the search) | same |
| cheapest | as the owner prefers (blank offer_id only) | same |
| everything else | defaults | defaults |

The manifests carry the machine's needs in a `machine` object: one GPU (`gpus` 1) with at least
4 GB, compute capability 6.0 or newer (and below 10.0: the cu126 build has no Blackwell), at least
32 CPU threads, 32 GB of RAM and 80 GB of disk (lowered from 128 threads and 64 GB on
2026-10-16: no 128-thread GPU machine was on offer, and a trick-4 or trick-3 rung needs the GPU, not
the threads; raise `min_cores` for trick-2 labels). These apply even with the workflow's default
`cpu_only` box ticked; the `gpus`, `min_gpu_ram_gb` and `disk_gb` inputs can raise them. An offer
picked by `offer_id` is checked against the same needs. The run ids are `m3-hybrid-smoke-1` and
`m3-hybrid-rung1b-1`; a second run of the same file needs a new `run_id` in it.

The launcher's log shows one line per minute with the step, e.g. `rung/0: labels random (1 of 3)
(250,000/3,200,000 rows)` or `rung/0: train (epoch 4/60, 1,200/3,907 batches, on cuda)`.
The low-CPU rule is not applied to this kind of run: the long stretches (the download, the GPU
training) leave the CPU idle by design, and on hundreds of threads the CPU steps take seconds.
Instead the download, the GPU check and the training have time limits (30 minutes per download
attempt, 10 minutes, `max_minutes` + 30 minutes), and the dollar cap bounds everything.

## What the two runs do

- **Smoke run** (about 10 minutes): 500,000 trick-4 labels and 20,000 for validation, a small
  trick-3 golden set (32 decisions x 32 worlds), the real network shape (1024-512-128, outcome
  head) for 3 epochs on the GPU, then the same 3 epochs on the CPU for comparison (`compare_cpu`),
  the check, and the label shards uploaded to R2 (`upload_shards`, to test that path; about
  0.5 GB). It proves the GPU path end to end and measures seconds per epoch on the GPU, which
  prices the full run's training.
- **Full run** (rung 1b): exactly the GitHub job's settings: 3.2M random-play and 0.8M rules-v1
  trick-4 labels (seeds 101, 102), 100,000 validation labels (seed 201), the trick-3 golden set
  of 400 decisions x 320 worlds (seed 1), the 1024-512-128 outcome-head network (60 epochs at
  most, batch 1024, learning rate 0.001, patience 5, seed 1, 200 minutes at most), and H_4 checked
  against exact on the golden set. Label shards are not uploaded (4.4 GB; they are remade
  byte-identical in about a minute); set `"upload_shards": true` to keep them.

## Adding harvested labels (spec §5 source (c))

Four optional settings add a shard of positions harvested from the hybrid's own searches to the
training set (what and why: `docs/m3-hybrid/rung1.md`, "Harvested positions"):
`labels_harvest` (rows; default 0, none), `labels_harvest_seed` (default 103; never 1 and never
another shard's seed), `harvest_cap_per_decision` (default 16) and `harvest_positions` (the shares
of decisions for the decider's positions 1–2, 3–4 and 5–8, as text, default `"0.25,0.25,0.5"`).
The harvested shard is made after the random-play and rules-v1 shards and trained on with them.

`docs/vast/m3-hybrid-harvest.manifest.json` (a proposal; not run): the full run with 6.4M
random-play and 1.6M rules-v1 labels (twice rung 1b's) plus 4M harvested labels with positions 1–4
oversampled (`"0.4,0.4,0.2"`). Expected extra cost against rung 1b: labels about 2 more minutes
(writing 13 GB of shards), training about three times longer per epoch (12.1M rows; 8 s an epoch
for 4M on the RTX 4090 of `m3-hybrid-rung1b-1`). The 10.2 GB of compact training rows stay in the
GPU's memory on a GPU of 12 GB or more; on a smaller one the trainer feeds them from the host
("staged"), slower. Peak memory of the label step is about 15 GB (the 6.4M random-play shard).

## Rung 2 (O_3): a network-driven harvest and a stored golden set

Three more optional settings (`docs/m3-hybrid/rung2.md`): `harvest_net` (the key
`inputs/sha256/<hex>` of an earlier run's `int.safetensors`, which then drives the harvest's
search: the way to harvest at trick 3), and `golden_decisions_input` with `golden_worlds_input`
(the keys of a stored golden set, reused instead of generated, with the check's 1% recheck;
`verify_pct` at least 1). The runner fetches and hash-checks them before anything else; the
launcher checks they are in R2 before renting. A generated golden set is now also uploaded to
`inputs/sha256/<hex>`, and `rung.json`'s `golden.reuse_params` gives the settings that reuse it.
Proposal: `docs/vast/m3-hybrid-t3-round2.manifest.json` (placeholders to fill in first).

## Time and money (estimates for the AMD EPYC 9654 (384 threads) + RTX A2000 class at about $1/h)

These are estimates from this VM's and GitHub's measurements, not measurements on that machine;
the smoke run measures them.

| step | full run | smoke run |
|---|---|---|
| boot: image (not billed while it downloads), binaries, start | 1–3 min | 1–3 min |
| equity table (from R2, or built from the model) | under 1 min | under 1 min |
| Python environment (about 3.7 GB downloaded; depends on the host's network) | 2–5 min | 2–5 min |
| GPU check | under 1 min | under 1 min |
| labels (4.1M rows; most of the time is writing 4.4 GB) | 1–2 min | under 1 min |
| golden set (128,000 exact trick-3 worlds, about 3.5 ms each) | about 1 min | seconds |
| training on the GPU (about 10–20 s an epoch for 4M rows; early stopping) | 5–15 min (20 at most) | under 1 min, plus 1–2 min on the CPU |
| hybrid check (128,000 worlds) | about 1 min | seconds |
| uploads | under 1 min | 1–2 min (the shards) |
| **total** | **15–30 min, about $0.25–0.50** | **about 10 min, about $0.20** |

If the GPU check fails, the full run trains on the CPU with every thread (roughly 30–90 s an epoch
for 4M rows, an estimate) for up to 200 minutes: about $3.50 more, inside the $6 cap.

The next rungs are mostly label cost: trick-3 labels (O_3) at about 9 ms each, 1–3M of them, are
2.5–7.5 thread-hours (about 1–3 minutes on 384 threads, plus writing); trick-2 labels (O_2) at
about 3 s each, 0.5–1M, are 400–800 thread-hours (2–4 hours on 384 threads, $2–4). A run that long
should be split into several label chunks first (one machine lost would otherwise redo it all);
that is not built yet.

## What it records

Everything is in the run's bundle `runs/<run_id>/out/rung/000000-<hash>.tar.zst` in R2 (the
`r2-fetch` workflow downloads a run; `tar --zstd -xf` or Python 3.14's `compression.zstd` opens it),
under `hybrid/`:

- `rung.json`: the settings; each step's seconds; the Python environment (uv version, the
  dependency group, the `uv.lock` SHA-256); the device facts; the label files (rows, seeds,
  SHA-256); the golden files' SHA-256; the network's SHA-256 (integer and float), best epoch,
  validation curve and errors; the check's verdict (PASS/FAIL) and its overall row; every warning.
- `device.json`: the GPU check: GPU name, memory and compute capability, driver and the CUDA
  version it supports, PyTorch, CUDA and cuDNN versions, and why the CPU was used if it was.
- `labels/*.npy.json`: each label shard's sidecar (encoder, values version, binary and table
  SHA-256, seed, counts, the shard's SHA-256). With `upload_shards`, `shards.json` lists the
  shards' R2 keys (`inputs/sha256/<hex>`) in the form a later manifest's `inputs` takes.
- `decisions-t3.jsonl`, `exact-t3-n320.jsonl`: the regenerated trick-3 golden set.
- `net/float.safetensors`, `net/int.safetensors`, `net/report.json`: the networks and the
  training report (per-epoch losses, errors of the float and integer networks, device facts and
  the deterministic settings). `net-cpu/` too with `compare_cpu`.
- `hc-t3-o4.json`, `hc-t3-o4.txt`: the hybrid check (spec §3), as JSON and as printed tables.

The logs of every step (uv, the GPU check, each label shard, the golden set, the training, the
check) are in `runs/<run_id>/logs/rung/000000.tar.zst`.

## Reproducibility

The labels, the golden set and the check are byte-identical on every machine and thread count.
Training on a GPU is not bit-for-bit repeatable across machines: different GPU kernels add numbers
in different orders. That is accepted, because what the yardstick uses is the whole-number network
made from it, which the check validates against exact search, and whose SHA-256 is recorded (and
named in the hybrid values version, `2+O4:<sha256>`). The trainer still does what PyTorch allows:
everything seeded, the network created on the CPU under the seed and then moved (so it starts
from the same weights everywhere), the same batches in the same order as on the CPU (the batch size
stays the configured one, so results are comparable), PyTorch's deterministic mode
(`torch.use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
cuDNN deterministic), and full 32-bit precision (no TF32 shortcut). Training on the CPU gives
exactly the same networks as before these changes (checked: same SHA-256 on this VM for both
heads).

PLAN §11.2 also lists mixed precision on GPUs (bf16/TF32 on Ampere and newer). It is not used:
it changes the network, and our standard is 0.01% money chance; it would need its own comparison
first.

## The training step on a GPU (CUDA graph)

Branch `m3-hybrid-gstep`, 2026-10-16. Measured on vast.ai today: training the 1024-512-128
network (batch 1024) was held back by the host CPU, not the GPU. Each batch is about 80 small
GPU operations, which the host launches one at a time: on an i9-14900K + RTX 4090 about 2.0 ms a
batch (the GPU busy for 1.3 ms of it); on old 2.2–2.3 GHz Xeons + RTX 3090-class GPUs 11–14 ms;
on an EPYC 7302P + 3090 with the 512-256-64 network 5.6 ms.

Now, on a GPU, the trainer records the whole training step of a full batch once as a CUDA graph
[a recording of all the GPU operations of one step, which the GPU replays from a single launch by
the host] and replays it for every batch: per batch the host only copies the batch's row numbers
into the recording's input and launches it, so a slow host CPU no longer matters. What is trained
does not change: the same batches in the same order, the same loss, the same Adam settings (batch
size and learning rate unchanged). Adam runs in its "fused, capturable" form [the whole update as
one GPU operation, its step counter kept on the GPU, as a recording needs], which rounds the last
digits differently; GPU training was already not bit-identical across machines, and the integer
network's SHA-256 is what gets validated. On the CPU nothing changes (the same networks byte for
byte: checked on this VM for both heads, both feeds, 1–4 threads and the real network shape).

- **Safety.** Before training, a trial on copies of the network runs 6 batches both ways (3
  normal steps, a recording, 3 replays) and requires the same losses to 1e-5. If the trial or a
  later recording fails (an unusual driver, say), the training runs the normal way, as before,
  and says why in the log (`training step: eager (...)`), the heartbeat (`train_step_mode`) and
  the report (`device.train_step`). It never stops the run.
- **When it records.** In the first epoch after 3 warm-up steps (PyTorch's documented procedure)
  and after the 16 profiled normal steps; again after each learning-rate cut (the plateau
  scheduler), since the rate is fixed inside a recording. The last, partial batch of each epoch
  runs normally. `--cuda-graph off` turns it off.
- **What the report shows.** `seconds.ms_per_batch` (each epoch's milliseconds a batch: the real
  measure), `seconds.profile` (16 normal steps, as before) and `seconds.profile.graphed` (up to
  16 replayed steps: milliseconds a batch, GPU busy time, top operations), and
  `device.train_step` (mode and why, the trial, recordings, replays and normal steps).
- **Expected speed (estimates from today's profile, not measurements; the next run measures
  them).** About 1.2–1.4 ms a batch on a 4090 and 1.7–2.2 ms on a 3090, whatever the host CPU
  (the GPU's own busy time, plus a little; the fused Adam saves a little of it): about 5 s and
  7–9 s an epoch for 4M rows, against 43–55 s on the old Xeons today. If a 3090 stays above
  2 ms, the replayed profile will show whether the first layer's embedding-bag backward pass
  (0.5 of the 1.3 ms on the 4090) dominates; replacing it by an equivalent dense product is the
  next candidate, not done yet.

## Pieces (for maintainers)

- `crates/runner/src/hybrid.rs`: the `hybrid_rung` chunk kind (settings, steps, records; any
  unknown setting is refused, also when the runner reads the manifest).
- `python/nn/fortyone_nn/device.py`: the GPU check; `train.py`: `--device cpu|cuda|auto` (default
  `cpu`), `--progress FILE`, memory-mapped shards read a block at a time into compact columns
  (847 bytes a row at trick 4), held in the GPU's memory when they fit with 1 GiB to spare
  (`--feed auto|resident|staged`; else pinned staging buffers), the same batches as before; a
  profile of 16 batches on a GPU and the seconds of loading, each epoch and the integer check in
  `report.json` (`seconds`); the integer check through exact float64 products
  (`intnet.exact_matmul`, the same integers, ~25-50x faster than numpy's int64 product); the
  training step (`Step`, `--cuda-graph auto|off`: a CUDA graph on a GPU after `graph_trial`,
  eager otherwise; `Feed.inputs/put/read` give the graph its batch).
- `python/nn/pyproject.toml`, `uv.lock`: PyTorch in two dependency groups, `cpu` (the default:
  `uv sync`, this VM, the laptop, GitHub) and `cu126` (`uv sync --locked --no-default-groups
  --group cu126`), both pinned with hashes.
- `python/launcher/launch.py`: `--gpus`, `--min-gpu-ram-gb`, `--min-compute-cap` and the
  manifest's `machine` object; the step of each running chunk in the log line.
- `.github/workflows/release-binaries.yml` (uv and `python/nn` in the bundle),
  `.github/workflows/vast-run.yml` (inputs `gpus`, `min_gpu_ram_gb`, `disk_gb`).
