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

The manifests carry the machine's needs in a `machine` object: at least one GPU (`gpus` 1; a machine with
two GPUs is fine, the price is per machine, and the trainer uses one, `cuda:0`) with at least 4 GB,
compute capability 6.0 or newer (and below 10.0: the cu126 build has no Blackwell), 32 GB of RAM and
80 GB of disk. These apply even with the workflow's default `cpu_only` box ticked; the `gpus`,
`min_gpu_ram_gb` and `disk_gb` inputs can raise them. An offer picked by `offer_id` is checked against
the same needs.

**Floors and waiting (owner rules of 2026-10-16, 18:10-18:30 Mauritius).** The launcher wants at least 190
CPU threads (`min_cores`, workflow input default "190"; a manifest's `min_cores` can raise it but not
lower it, so to run a smoke test on a smaller machine lower the `min_cores` input), at most $1.70 an hour
(`machine.max_dph` in the manifest can raise or lower the cap; an offer picked by `offer_id` is held to
it too), and a download link of at least 100 Mbps (`machine.min_inet_down`; the 5-6 GB image pulls in
about 7 minutes at 100 Mbps; an offer picked by `offer_id` with a slower link is rented and the log says
so). A machine still pulling the image after 15 minutes is waited for while vast's status shows the pull
advancing (40 minutes at most, 10 minutes without change ends it). When no machine fits, the launcher
does not fail: it searches again every minute for up to two hours and logs one line per attempt with the
price step, the attempt, the elapsed minutes and the closest miss. It looks for the cheapest first: three
searches at $1.20 an hour, three at $1.45, then at the $1.70 cap, renting the first offer that fits at the
current step (the best cost per unit of CPU work among those), and a replacement search after a start
failure waits the same way, with the same floors, starting again at $1.20. The workflow has no inputs for
the wait (it is near GitHub's limit of 25); put it in the manifest's `machine` object when it needs
changing: `wait_for_offer_min` (120; 0 = fail at once, as before), `offer_poll_s` (60), `price_steps`
(a list, default `[1.2, 1.45, 1.7]`), `tries_per_step` (3) and `max_dph`. A machine whose container did
not start, or that never sent a heartbeat, is remembered in R2 (`launcher/bad-machines.json`) and left
out of every search for 24 hours (an `offer_id` pointing at it is rented anyway, with a warning). The
run ids are `m3-hybrid-smoke-1` and `m3-hybrid-rung1b-1`; a second run of the same file needs a new
`run_id` in it.

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

**Network widths: at most 4096 units per layer** (the `widths` setting of `hybrid_rung`, the trainer's
`--widths`). The integer inference that loads, checks and uses the network refuses a layer wider
than `MAX_WIDTH` = 4096 (`crates/nn/src/mlp.rs`). The limit was 1024 until 2026-10-16, when a
trick-3 run trained 2048-1024-256 for 27 minutes and then stopped at the check with
`layer 0: wider than 1024`. It was raised to 4096 the same day (owner: "go 2048 wide"), with the
proof written in that module's notes ("Widths and the overflow bound"):
- every sum stays exact because each network's own weights are checked against the `2^62` bound
  when it is loaded, at any width; the trainer's quantiser picks the scale so that the bound holds
  (trained-scale weights keep all 24 fraction bits at widths 2048 and 4096, far inside it);
- what grows with the width is memory only (the fast pass's scratch, and the per-thread cache of
  first-layer parts: 4 MiB a thread at width 2048);
- every network up to 1024 wide gives byte-identical outputs to before (16 networks, 4 of them
  trained), and the fast pass equals the plain reference at 1024, 2048 and 4096 (tests).

Every place that accepts a width still refuses one over the limit at once, naming the layer
(counted from 0) and the limit: the runner's `hybrid_rung` settings (before anything is installed
or trained), the trainer's `--widths` (before any data is loaded), and loading a network file
(`search labels --harvest-net`, `search hybrid-check`). The trainer's copy of the number
(`MAX_WIDTH` in `python/nn/fortyone_nn/formats.py`) is tested against the Rust source and `nn-tool
limits`.

Cost of width (one frontier evaluation, random networks of a trained network's scale, this
project's build VM, x86-64-v3 code as on the EPYC machines; `nn-tool bench`, 1,000 real frontier
positions, best of 3; the VM is shared, so about ±15%):

| network | trick 4 | trick 3 | file (`int.safetensors`) |
|---|---|---|---|
| 1024-512-128 | 42 µs | 37 µs | 6.1 MB |
| 2048-1024-256 | 225 µs (×5.4) | 203 µs (×5.4) | 16.9 MB |
| 2048-2048-1024-256 | 602 µs (×14) | 552 µs (×15) | 33.7 MB |

Trained networks are faster than random ones of the same shape (more of their hidden units are
zero): the trained 1024-512-128 networks take 12–17 µs.

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

## The race harvest (lead-4)

Why: `docs/m3-hybrid/rung1.md`, "Why seats 3-4 fail: the race for the money". A fourth optional
label source, made after the harvest shard and trained on after it: `labels_race` (rows; default 0,
none), `labels_race_seed` (default 105; never 1 and never another shard's seed), `race_classes`
(default `"2,4"`: the race classes whose reached positions are all kept), `race_cap` (default 256
of them per decision), `race_strata` (default `"4"`; `"2,4"` also works: the strata the decisions'
match states are drawn from), `race_positions` (the shares of decisions by the decider's seat band,
default the harvest's `harvest_positions`) and `race_other_cap` (default 0: positions of the other
classes kept per decision, so the shard holds race positions only). It is `search labels --source
harvest --harvest-cap <race_other_cap> --harvest-positions <race_positions> --harvest-strata
<race_strata> --harvest-race-classes <race_classes> --harvest-race-cap <race_cap>`, with
`--harvest-net` when `harvest_net` is given (then a race harvest alone may use it). The race
settings are refused when `labels_race` is 0. `rung.json` records them under `params.race`, and
each shard's rows by race class (from its sidecar's `rows_by_race_class`) under `labels`; the
training report gives the validation error by race class (`float.error_by_race_class`,
`int.error_by_race_class`, and each epoch's `val_sd_by_race_class` in the history).

Cost: 1M race rows at trick 4 take 0.08 thread-seconds per 1,000 rows (27 s on the 4 threads of
the build VM; about 2 s for 10M on 384 threads plus writing the 10.8 GB shard): cheaper than the
plain harvest per row, because race-rich stratum-4 decisions give 43 rows each. They are 52% class
4 and 48% class 2.

Proposal `docs/vast/m3-hybrid-lead-4.manifest.json` (not run): 16M random play (seed 151), 4M
rules-v1 (152), 10M harvest (154, cap 64, seats 0.3/0.55/0.15), 10M race (155, classes 2 and 4,
cap 256, stratum 4, the same seat shares, no other classes), 100,000 validation (201),
1024-512-128, patience 6, the trick-3 golden set (400 x 320, seed 1) generated as before. Same
size, time and machine as lead-3c (40M rows). The golden set is byte-identical when regenerated;
to reuse lead-3c's files instead, upload them (`put-input`) and add `golden_decisions_input`
`inputs/sha256/e195e7739ce257d0fdaf99cfb6780e9d7b1954a922f22771c721ea62cff6e3b4` and
`golden_worlds_input` `inputs/sha256/c1404d874352b6d46546322891033e87b23c6a2218f55b4af10b98d37fd13911`
(`decisions-t3.jsonl` and `exact-t3-n320.jsonl` of lead-3c's bundle).

## Train only (O_2), and the trained network shared

`"check": false` makes a train-only rung: labels and training as above, but no golden set and no
hybrid check; the verdict reads "not checked (train only)". It is for O_2, whose check is the
trick-1 check on the saved exact worlds (a generated trick-1 golden set is out of reach), run
afterwards as a `hybrid_t1check` chunk (`docs/m3-hybrid/rung2.md` §4–5). With it the golden set's
and the check's settings (`golden_*`, `verify_pct`, `tt_mb`) are refused rather than ignored.
Every rung now also uploads its integer network to `inputs/sha256/<hex>` after training
(`rung.json` `net.int_key`, `result.json` `net_int_key`), so the next run names it as an input
without a manual `put-input`. Proposal: `docs/vast/m3-hybrid-t2-round1.manifest.json` (cut 2,
1M random-play and 250,000 rules-v1 labels, 50,000 for validation, 1024-512-128; runnable as it
is; priced in `rung2.md` §5 before anything is rented).

## The audit loop's settings (decisions of 2026-10-16 night)

The audit loop (`audit_rounds` above 0; `docs/m3-hybrid/audit-loop.md`) takes, beside its earlier
settings:

- `audit_gamma` (default 0.001; **tonight 0.0001**): the smallest weak group the auditor looks
  for, as a share of positions. At 0.0002 or less the audit shard grows by itself to 20M rows at
  trick 4 and 10M at trick 3 (`audit_rows` overrides it), so a group of 0.01% holds 1,000-2,000
  rows. The shard is deleted once the auditor has read it (about 22 GB at trick 4, 14 GB at
  trick 3; its record stays in `round-<r>/`). The auditor uses every thread of the step and has
  90 minutes plus 15 per million rows above 2 million.
- `audit_standard` (default 0.01) and `audit_push_standard` (none by default; **tonight 0.005 at
  trick 4, 0.0075 at trick 3**), in percent of money chance: every golden check, slice and mini
  set is judged by the standard (`search hybrid-check --standard`); once the loop is flat at
  `audit_standard` it judges the same round again by the push standard and carries on, on the same
  machine and dollar cap, until flat at that too. The heartbeat's `standard` says which phase it
  is in; the summary has a line "flat at 0.01% after round 3; pushing to 0.005%".
- `audit_mini_candidates` (default 200 × `audit_mini_decisions`): the most candidates one mini
  golden set may draw. A group lighter than 0.1% keeps decisions with at least 500 × its weight of
  their reached positions in it (at least 5%) instead of a half. A set's verdict needs at least
  `audit_mini_min` decisions (default 30): with fewer (none included), its group is reported
  "unmeasurable at the decision level (N decisions kept)" in every round report and the summary,
  never passed or failed, never failing and never blocking "flat" (the overall row covers it);
  it is targeted only when the auditor confirms it again. `audit_mini_sets` (new sets a round, default 4) has no upper limit: at trick 4 a set costs
  seconds, so a loop may measure every waiting group each round (e.g. 400).
- Stops added: *diminishing returns* (two rounds in a row on the largest shape lowering neither
  the worst failing group's bound nor the overall bound by `audit_min_gain`; never while the shape
  can still grow; the summary lists the groups still failing with their weight and SD). It
  replaces "capacity ladder exhausted".
- The capacity ladder keeps only shapes larger than the rung's `widths` (a rung at 2048,1024,256
  steps to 2048,2048,1024,256, then 4096,2048,1024,256).
- The chunk's network (`rung.json` `net`, `check`) is the best round's, not the last (flat at the
  tightest standard first, then the lowest overall bound); `audit-summary.txt` says "best: round
  k"; a stop keeps it too.

## The audit loop at trick 2 (O_2)

Branch `m3-hybrid-t2loop`, 2026-10-16 (Mauritius). The audit loop (`docs/m3-hybrid/audit-loop.md`,
module notes of `crates/runner/src/hybrid/audit.rs`) runs at cut 2 too, in one rental. The owner's
decision of 2026-10-16 (spec §3): **no trick-1 golden set is ever built**, so at trick 2:

- **The stop is position-level only.** A round is *flat* when the auditor confirms no weak group on
  that round's fresh audit shard of exactly labelled trick-2 positions. There are no golden slices
  and no mini golden sets; a group confirmed in a round is targeted in that round's harvest only
  (no group is ever "failing" at trick 2).
- **The rough test** runs every round and is reported only: H_2 against the saved exact trick-1
  worlds (`search hybrid-check --trick 1 --cut 2 --saved …`, the `hybrid_t1check` chunk's command,
  now with `--slices`), with a slice per group the loop has met. It never decides the stop and
  never makes a group failing. Its evidence is thin and every report says so: the saved decisions
  file holds 32 trick-1 decisions, but saved exact worlds exist for only 8 of them (decisions 0-7,
  seats 1-2 only): 8 decisions with 31 worlds under the searched burn and 5 decisions with 355
  worlds under the rules-v1 burn. Seats 3-8 have no saved trick-1 decision at all. Each group's
  slice is labelled "thin: N decisions".
- **Settings**: `t1_decisions_input` and `t1_saved_inputs` (a list of `{"run": LABEL, "input":
  "inputs/sha256/<hex>"}`; required at cut 2, refused at other cuts), fetched and SHA-256 checked
  before anything is installed (the launcher checks they exist before renting); `t1_verify_pct`
  (3), `t1_verify_rule` (`rules-v1`), `t1_verify_solve_threads` (46), `t1_verify_tt_mb` (8192): the
  verification sample of round 1 (8 rules-v1 worlds recomputed exactly first; about half an hour,
  `rung2.md` §4); later rounds reuse the same worlds. At cut 2 the golden set's settings
  (`golden_*`, `verify_pct`), the mini sets' (`audit_mini_decisions`, `audit_mini_sets`,
  `audit_mini_candidates`, `audit_mini_min`) and the decision-level standards (`audit_standard`,
  `audit_push_standard`: nothing is judged at the decision level at trick 2, so there is no flat
  to push from; the rough test reports against 0.01%) are refused, and a cut-2 rung with a check
  but no loop (a generated trick-1 golden set) is refused. `audit_gamma` applies (the auditor is
  trick 2's standard); the audit shard stays at 200,000 rows at any gamma unless `audit_rows` says
  otherwise (a label costs seconds).
- **Round 1** has no earlier network to drive the search (the manifest gives no `harvest_net`, and an
  exact-driven trick-2 harvest would solve every reached position, thousands a decision at seconds
  each), so its audit shard is made after round 1's training, driven by round 1's own O_2; later
  rounds' shards are driven by the previous round's O_2, beside the training, and the targeted
  harvest by the round's own (`--harvest-tricks 1` is the default at cut 2: trick-1 decisions).
- With `upload_shards`, the loop's audit and targeted shards are shared content-addressed as well
  (at trick 2 only: each is hours of exact labels).

**Merged with the overnight loops' builds** (branch `m3-hybrid-t2merge`, 2026-10-17; the trick-2
design above holds, and so do the builds of the overnight trick-3 and trick-4 loops):

- *harvest_net.* The rule "an audit loop at cut 3 or earlier needs `harvest_net`" (eng-runfix) is
  now cut 3 only: at cut 3 round 1's audit shard is harvested beside round 1's training, before any
  network of the loop's own; at cut 2 it waits for round 1's network by design. The reason stays
  right for the base labels, so `labels_harvest` and `labels_race` at cut 3 or 2 need `harvest_net`
  (made before any network of the run exists). A `harvest_net` must value the cut's own frontier:
  the runner reads its trick when it fetches it and refuses another, before anything is installed
  (`search labels` refused it only when the first harvest started, hours into a trick-2 run). So
  O_3 cannot drive a trick-2 harvest, and the trick-2 manifest names no driver.
- *The stall watch* (the trainer killed only when its progress file stops changing while it is not
  waiting for data) and *the bounded targeted harvest* (`search labels --max-decisions`, by default
  max(1,000, 50 x rows / keep cap) decisions) apply at cut 2 as everywhere. A bounded shard counts
  the rows it wrote.
- *Shards kept.* The audit shard is deleted once the auditor has read it (eng-auditscale), except
  at trick 2: there it is shared when it is made (`upload_shards`), deleted only once shared, and
  kept on the disk with a warning when it was not (`local_file` in the round's `audit_shard`).
  Targeted shards stay on the disk for the next rounds' training and are shared too.
- *The best round* is the chunk's result, as at tricks 3 and 4, but without a golden set the
  ranking reads the position level: flat (no group confirmed on its own audit shard) first, then
  the lowest overall SD (RMS error) of the round's network on its own fresh, never-trained audit
  shard of exactly labelled trick-2 positions (the auditor's `overall.sd`; the validation shard's
  SD when the auditor gave none), then the lowest SD of a group it confirmed, then the earlier
  round. Not the rough test: 8 decisions at seats 1-2 are too few to choose by, and the owner made
  it report-only. `audit-summary.txt` says "best: round k (…; overall SD x% on its own fresh audit
  shard)"; `rung.json` `check.rough_test` is the kept round's.
- *Diminishing returns* at the position level: two rounds in a row on the largest shape of the
  ladder whose worst confirmed group's SD fell by less than `audit_min_gain` end the loop (never
  while the shape can still step up; the targeted harvest of the second is not made). The
  decision-level rule of the overnight loops has nothing to read at trick 2. The ladder only grows.
- *partial_epochs* is accepted with the audit loop (it needed `label_parts` above 1): each round's
  training on a GPU starts beside the round before's targeted harvest, which takes hours at trick
  2; uncapped, the trainer would make dozens of epochs on the earlier shards (`overlap.md`: 10
  partial epochs a part cost 7% of validation loss, 2 about nothing). The trick-2 manifest sets 2.

**Defaults per cut** (`audit_rows`, `audit_target_rows`; both can be set in the manifest):

| cut | audit shard rows | targeted harvest rows | decision-level stage |
|---|---|---|---|
| 4 and 5 | 1,000,000 (20,000,000 at `audit_gamma` 0.0002 or less) | 10,000,000 | golden slices, mini golden sets (gate) |
| 3 | 1,000,000 (10,000,000 at `audit_gamma` 0.0002 or less) | 4,000,000 | golden slices, mini golden sets (gate) |
| 2 | 200,000 at any gamma | 1,000,000 (the manifest: 250,000) | rough test on the saved trick-1 worlds (report only) |

**The manifest** `docs/vast/m3-hybrid-t2-loop.manifest.json` (run `m3-hybrid-t2-loop-1`; not run
yet): cut 2, scratch mode as the overnight loops; base labels 1,200,000 random play (s511), 300,000
rules-v1 self-play (s512), 50,000 validation (s601), no harvest (no O_2 exists, and O_3 cannot drive
one); 2048-1024-256 on the ladder `<widths>;2048,2048,1024,256;4096,2048,1024,256`, epochs 40,
patience 6, `max_minutes` 0, `overlap` with `partial_epochs` 2; 4 rounds of 200,000 audit rows
(seed 221, gamma 0.001) and 250,000 targeted rows (weight 2); `upload_shards`; the rough test on
the saved trick-1 worlds by the SHA-256 of the t1check manifest. Machine (owner, 2026-10-17: a
cheap machine with at least 300 threads): 1 GPU of 20 GB or more (any model: the run is CPU-bound),
at least 300 threads and 192 GB (labels hold about 200 MB a thread with 256 MB tables, at most 98 GB
at 384 threads; the trainer about 3 GB of rows; the rough test's verification table 8 GB), 80 GB of
disk (about 4 GB of shards), $2.00/h at most on the ladder 1.40 / 1.70 / 2.00 with ten searches a
step. Launch with `--max-dollars 35` (the owner's trick-2 budget).

**Projection** (rented thread-seconds a label: the build VM's measurements below with item A,
random 3.65, rules 3.5-4.2, harvested 1.4-3.5 (1.4 in item A's gate with a network driver, 3.5
on 2026-10-16 on 416 labels of every seat band), times 1.0-1.3 for a rented thread; about 2 hours
of setup, round 1's training, round 1's rough test with its verification sample, the auditors and
slack; the later rounds' training runs beside the CPU):

| step | 300 threads | 384 threads |
|---|---|---|
| base labels (1.55M) | 5.2-7.0 h | 4.1-5.5 h |
| an audit shard (200,000) | 0.26-0.83 h | 0.20-0.65 h |
| a targeted harvest (250,000) | 0.32-1.04 h | 0.25-0.81 h |
| flat after round 1 | 6.7-9.0 h ($7-9 at $1.00/h, $9-13 at $1.40/h, $13-18 at $2.00/h) | 5.5-7.3 h ($5-7, $8-10, $11-15) |
| flat after round 2 | 7.6-11.3 h ($8-11, $11-16, $15-23) | 6.3-9.2 h ($6-9, $9-13, $13-18) |
| all 4 rounds (3 targeted harvests) | 9.2-15.5 h ($9-15, $13-22, $18-31) | 7.6-12.5 h ($8-13, $11-18, $15-25) |

No corner of the projection reaches $35, even at the ladder's top price; the dollar cap is a
safety net (it would stop the loop with every finished round uploaded and the best round kept).

**Cost, measured on the build VM** (2026-10-16; 4 vCPU Intel Xeon at 2.1 GHz, all 4 busy; solver
nodes per label are the same on any machine):

| label | labels measured | solver nodes a label | thread-seconds a label here |
|---|---|---|---|
| harvested trick-2 positions (audit shard, targeted harvest), every seat band | 416 | 31.5 million | 3.5 |
| random play (round 0, validation) | 60 | 36.4 million | 4.0 |
| rules-v1 self-play (round 0) | 40 | 43.6 million | 4.8 |

A busy thread here did about 9 million nodes a second. The search that drives the harvest is under
2% of its time even with a 2048-1024-256 network (3.2 s for a trick-1 leader decision, about 23,000
evaluations at 140 µs, against 16 labels of 3.5 s). A rented thread of the 184-thread Threadripper
is taken as 1.0-1.3 times slower (trick-3 labels: 9 ms here, 11.3 ms on that machine in round 1 of
rung 2). So on 184 threads:

| step | time |
|---|---|
| round 0: 1.3M labels | 8.2-10.7 h |
| audit shard, 200,000 rows | 1.1-1.4 h |
| targeted harvest, 1,000,000 rows | 5.3-6.9 h |
| training (GPU; beside the CPU steps after round 1) | not measured for 2048-1024-256: an estimate of 5-25 min a round |
| auditor, rough test | minutes (the rough test's verification sample: about 30 min, round 1 only) |
| **one round with a targeted harvest** | **6.6-8.5 h** |

The whole rental: flat after round 1, 10-13 h ($11-21 at $1.1-1.6 an hour); after round 2, 17-22 h
($19-35); all 4 rounds, 31-39 h ($34-43 at $1.1, $49-62 at $1.6). **The $35 cap does not hold four
rounds with 1,000,000 targeted rows**; it holds them only with fewer targeted rows (500,000: 23-29 h,
$25-46; 250,000: 19-23 h, $20-37), or the dollar cap stops the loop in round 3 or 4 with every
finished round uploaded. The plan's "200k positions (~10 minutes on 184 threads)" for the audit
shard (`audit-loop.md` §7) is seven to eight times too low: 200,000 labels at 3.5-4.6 thread-seconds
are 1.1-1.4 hours on 184 threads.

## The GPU and the CPU at the same time

`overlap` (true): on a GPU, a golden set to generate is made while the network trains (on the
threads the trainer leaves; answer-neutral); a trick-3 run saves about 9 of 43 minutes.
`label_parts` (1) and `partial_epochs` (0): the labels in parts, the training starting after the
first; small gains at best and a measured accuracy cost when many epochs run on the early parts,
so off by default (`docs/m3-hybrid/overlap.md`, which also describes the interfaces the audit loop
uses: `Rung::concurrent`, the background training, `train.py --watch-dir`).

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

### Time and money, measured (2026-10-16)

Nine rentals ran the same day (the smoke run, the 4M-label run and seven more; what each trained
and what it found: `docs/m3-hybrid/rung1.md`, "Results, 2026-10-16"). Against the estimates above:

| step | estimate above | measured |
|---|---|---|
| boot, image, start, uploads, tear-down | 1-3 min plus under 1 min | 3-8 min of each rental fall outside the rung's own steps |
| Python environment | 2-5 min | 19-56 s with the older image; 1-2 s with the image that carries the packages (from harvest-2 on) |
| GPU check | under 1 min | 5-18 s |
| labels | 1-2 min for 4.1M rows | 53 s for 4.1M rows (30 threads); 2.3 min for 28.1M rows, 30 GB (46 threads) |
| golden set (400 decisions x 320 worlds, exact) | about 1 min | 4-25 s (184 threads to 38 threads) |
| training on the GPU | 5-15 min for 4M labels | 2.1 min for 4M labels (8 s an epoch); 5.2 min for 8M; 10.4 min for 16M; 12.5 min for 28M (22 epochs at 29.5 s) |
| check | about 1 min | 51 s in the first runs, 5 s in the last |
| whole rental | 15-30 min, $0.25-0.50 (smoke: about 10 min, $0.20) | 4M labels: 8.7 min, $0.10 (7.4 min, $0.21 on a dearer host); 28M labels: 21.3 min, $0.27; smoke: 10.2 min, $0.17 |

The CUDA-graph training step ran at 1.1 ms a batch on the RTX 6000 Ada card and 1.9-2.0 ms on the
RTX 3090 and 3090 Ti cards behind three different host chips (estimate for the 3090: 1.7-2.2 ms).
Three runs predate the change (4M labels twice on an RTX 4090, and 16M labels with the 512-256-64
network on an RTX 3090 behind a 16-core EPYC 7302P); the last took 83 s an epoch, about 5 ms a
batch, and 41 minutes in all. The nine rentals together cost about $2.01 (minutes times the
offer's hourly price, the launcher's estimate rather than an invoice); the dearest single run was
that 41-minute one ($0.45), and the 16M run's total includes two machines that never reported
within 15 minutes and were replaced (33 minutes, about $0.14).

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
