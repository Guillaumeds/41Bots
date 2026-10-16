# The GPU and the CPU at the same time (hybrid rungs)

Branch `m3-hybrid-overlap`, 2026-10-16 (Mauritius). The owner's request (18:30): "build 1/4 on CPU
then feed the GPU while the CPU builds the next quarter". In every rented run that day the GPU sat
idle for 20–110 minutes of CPU work. Nothing was rented for this build; the measurements below were
made on the build VM (4 CPU threads, no GPU).

**In one paragraph.** Two overlaps were built into `hybrid_rung`:
- **The golden set beside the training.** It is on by default and changes no answer. On a GPU,
  the golden set is made while the network trains, and so is any later step of the same kind.
  For a trick-3 run it saves about 9 of 43 minutes.
- **Label parts streamed into the training** (`label_parts`, off by default). Training starts
  once the first quarter of the labels exists, and later quarters join as they are made. It saves
  at most a minute or two, and only when nothing else is longer. In a small measurement it cost
  accuracy: when the network trains many rounds on the first quarters before the rest arrives,
  its final validation loss was 7% worse.

So the golden overlap is the useful part, and the streamed labels stay off unless a run shows
they pay. Both mechanisms are general, for the audit loop's CPU steps beside the GPU
(`docs/m3-hybrid/audit-loop.md` §4); their interfaces are in the last section.

## 1. The golden set beside the training (`overlap`, default true)

On a GPU, when the golden set is to be generated (not a stored one):
1. The trainer starts as soon as the labels are made. It runs as a child process beside the
   steps, with its usual 8 CPU threads.
2. `search decisions` and `search worlds` for the golden set then run on the remaining threads
   (`threads − 8`, at least 1).
3. The hybrid check runs after both, on every thread, as before.

The golden set and the check are byte-identical whatever runs beside them, because each child
gets a fixed thread count and the results never depend on the thread count.

The heartbeat names both steps at once (`"golden worlds + train"`), with the worlds' counts and
the trainer's epoch and batches (`progress.chunk_steps`).

If either fails, the other is stopped at once, the chunk fails with the reason, and nothing is
left running (tested both ways).

`rung.json` gets `overlap`:
- `golden_with_training`;
- the two thread counts;
- `train_s` and `golden_s` (how long each ran);
- `both_s` (how long they ran together).

The step list in `rung.json` is in the same order as before.

On the CPU (no GPU) nothing changes: the two would only share the same threads.

## 2. Label parts streamed into the training (`label_parts`, default 1)

With `label_parts` P above 1:
- **How the labels are split.** The random-play and rules-v1 labels are made in P shards each.
  Part k of a set with seed s has its share of the rows and the seed s × 100 + k (part 2 of seed
  101 is seed 10102). So the rows are a pure function of the manifest, whatever the machine.
- **The order they are made in.** First the validation shard, then part 1 of each set, then
  part 2, and so on. A harvest shard, if any, comes last as one shard. Training shards made in
  parts go into `hybrid-work/labels/train/`.
- **On a GPU:**
  - The trainer starts once part 1 is made. It runs `train.py --watch-dir … --expect N`
    (section 5), and the later parts are made on the threads it leaves.
  - Each new shard joins the training between epochs.
  - Epochs before every shard is in are "partial": they do not count towards the epoch limit,
    the patience or the learning-rate scheduler, and they cannot provide the best network. The
    network kept is always one trained on every shard.
  - The trainer's time limit counts from the moment the last shard is made.
  - `partial_epochs` K caps the epochs on each partial set of shards; after K the trainer waits
    for the next part.
- **On the CPU**, the same shards are made, then the training runs on all of them, as before.
- **With P = 1**, the step list and the files are exactly as before.

`rung.json` records:
- under `stream`: when the training started and how many threads the labels kept;
- per shard (in `labels`): its part and its threads;
- the trainer's own `stream` record under `net`: which epoch first saw each shard, the first
  epoch with every shard, and how many epochs trained on each.

## 3. Measured on this VM

**Quality.** All runs used the same trick-4 label shards: 4 parts of 50,000 random-play rows
(seeds 10101–10104) and 20,000 validation rows (seed 201). Settings:
- network 256-128-64 with the outcome head;
- batch 512, learning rate 0.001, patience 3;
- 4 CPU threads.

Only the moment the shards arrived differed. In a streamed run, part k+1 arrived once the
trainer had done its partial epochs on the first k parts (`--partial-epochs`).

| run | partial epochs | full epochs | best validation loss | outcome accuracy | value error SD |
|---|---|---|---|---|---|
| all shards at once, seed 1 | – | 12 | 1.5457 | 85.6% | 2.88% |
| all shards at once, seed 2 (noise reference) | – | 13 | 1.5534 | 85.4% | 2.83% |
| streamed, 2 partial epochs per part | 6 | 8 | 1.5574 | 84.7% | 2.98% |
| streamed, 10 partial epochs per part | 30 | 4 | 1.6605 | 83.4% | 3.17% |

**What the table says:**
- **With 10 partial epochs per part:** the network first fitted the early parts. When the rest
  arrived, its validation loss stopped improving after the first full epoch, so the patience
  ended the training after 4 full epochs, 7% worse than training on everything from the start.
  This is the known cost of warm-starting [training first on part of the data, then on all of
  it] (Ash & Adams, "On warm-starting neural network training", NeurIPS 2020).
- **With 2 partial epochs per part:** the cost is small. It sits at the edge of the run-to-run
  noise (the two seeds differ by 0.008 in loss; the streamed run is 0.004–0.012 worse).

**Which regime a real run is in:**
- **Trick 3:** a part of the labels takes about 4 minutes to make, while a GPU epoch on a quarter
  of the labels takes seconds. Unless capped, the trainer would make dozens of partial epochs
  per part: the bad regime.
- **Trick 4:** a part takes about 20 seconds, which is a few partial epochs at most.

**Times.** Times could not be measured here: there is no GPU, and on 4 CPU threads the trainer
and the labels share the same cores. The mechanics (order, threads, failures, records) are tested
with fake programs. The expected savings below are worked out from the step times of the 7995WX
runs (184 threads).

## 4. Expected wall-time savings (estimates)

| run | before | golden beside training | plus 4 label parts |
|---|---|---|---|
| trick 3: labels 16 min, golden 17 min, training 10 min | 43 min | 33.8 min (labels 16, then the golden set on 176 threads, 17.8 min, with the training inside it): **−9 min, −21%** | 34.3 min: no further gain (the golden set is the longer step; the later parts lose 8 threads) |
| trick 3 with a stored golden set: labels 16, training 10 | 26 min | 26 min | 16.5 min plus the full-data training: at best about −4 min if the warm start halves it, else nothing. This is the regime of dozens of partial epochs per part, which costs accuracy unless capped (and capped, the GPU mostly waits anyway) |
| trick 4: labels 1–2 min, training 15 min | 16.5 min | the same, or −2–3 min when the trick-3 golden set is generated (it then runs beside the training) | −1 to −1.5 min (training starts after about 0.4 min), at the small quality cost of the 2-partial-epoch row |

These do not count the check, which comes after both and is unchanged.

## 5. Where the owner's idea costs quality

- **The golden set beside the training: nothing.** Every child has a fixed thread count, and no
  value depends on the thread count or on what runs beside it.
- **Streamed labels: the warm start.** A network that trains many epochs on the first parts
  before the rest arrives ends up worse (table above, +7% validation loss with 10 partial epochs
  per part). The trainer limits the damage: the patience, the scheduler and the choice of the
  best network count only epochs on every shard, and `partial_epochs` caps the partial epochs.
  But the cap also caps the gain, because the GPU then waits for the CPU anyway.
- **Different labels.** With parts, the labels come from P seeds instead of one. That is the
  same distribution, but a different sample, so a run with `label_parts` 4 is not
  label-for-label comparable with one with 1.

**Recommendation:** keep `overlap` on and `label_parts` at 1, as the committed manifests do.
Streamed labels are worth trying only at trick 4 with `partial_epochs` 1–2, and only once a
rented run shows the gain.

## 6. Interfaces (for the audit loop and later steps)

**Runner, several steps at once:** `Rung::concurrent(steps)` in `crates/runner/src/hybrid.rs`.
- **Input:** each `ConcurrentStep` gives:
  - `step` (its name);
  - `cmd` (the child, given its own `--threads`);
  - `threads` (its share);
  - `log`;
  - `progress` (a JSON file the child rewrites, or none);
  - `prefix` (for its progress fields);
  - `limit`.
- **What it does:**
  - All the steps start at once.
  - Shares that add up to more than the machine's threads are refused. A step already running
    beside counts towards this, e.g. the training's 8 threads.
  - The heartbeat names them all (`"a + b"`), with each one's progress fields as
    `<prefix>_<field>`; `chunk_steps` shows fields that end in `_done` or `_total`.
  - When one fails, the others are killed at once and the call fails with the reason.
  - Each step is recorded in `rung.json`'s `steps` with its threads, `started_s` and `ended_s`,
    and the group under `concurrent`.
- **Returns:** when each step ran.

**Runner, a long step beside a sequence:** `Rung::start_train(&Feed)` and `Rung::finish_train()`.
- **The training as a step beside others.** Its threads are kept out of every later step's
  share (`Rung::step_threads`). Every later step's heartbeat shows `+ train` and the trainer's
  `train_*` fields.
- **The trainer's input:** `Feed::Files` (given shards) or `Feed::Watch { dir, expect, val }`
  (streamed shards, below).
- **Underneath:** `Ctx::spawn_bg`, `poll_bg`, `wait_bg`, `kill_bg` and `bg_limit_from_now`
  (`crates/runner/src/main.rs`). Several named children can run beside the current step. Any one
  failing or overrunning its limit stops all of them and the current step, and a stop or the
  deadline stops them too.

**Trainer, streamed shards:** `python -m fortyone_nn.train --watch-dir DIR --expect N --val V
[--partial-epochs K] …`.
- **Naming:** any `*.npy` in DIR not starting with a dot; nothing else in DIR is read. When
  several shards are ready together they join in name order. The validation shard is outside
  DIR. `--train` files may be given too; they count in N and come first.
- **Complete:** a shard counts once it exists under its final name. Write it under another name
  (`search labels` writes `NAME.npy.part`) and rename it.
- **Weight:** `NAME.npy.weight`, a text file with a positive number, in place before the shard is
  renamed into DIR; without one the weight is 1. Every row of the shard counts that much in its
  batch's loss (the weighted mean). With every weight 1 the network is byte-identical to the
  unweighted training (tested). `--train` files take weights the same way. The report gives them
  (`data.train_weights`, `stream.parts[].weight`), and the network's metadata gives
  `train_weights` when one is not 1.
- **Row weights:** `NAME.npy.rowweights`, one weight per row, also in place before the shard is
  renamed into DIR (or beside a `--train` file). A row then counts its shard's weight times its
  own.
  - Format: a `.npy` file of a 1-D `<u2` array (little-endian 16-bit whole numbers), one per row in
    row order, in units of 1/256: 256 is a weight of 1, 512 of 2, 384 of 1.5; 1 to 65535, never
    0. Whole numbers, so a weight has one exact encoding and Rust writes the same bytes as
    `numpy.save` (tested against a golden hash).
  - Writers: Rust `nn::rowweights::write(shard_path, &weights)` (whole and atomic: a `.part` name,
    then a rename; `nn::rowweights::fixed(2.0)` gives 512, `chosen_rows(rows, &indices, 512)` the
    array where the listed rows weigh 2). From the command line: `nn-tool row-weights --shard
    NAME.npy --weight 2 --rows ROWS.npy` (ROWS a 1-D `<u8` array of row indices); without
    `--weight` it prints the sidecar's summary. From Python: `numpy.save` on an open file (given a
    name, numpy adds `.npy` to it), then rename.
  - A sidecar of all 256 trains the byte-identical network; all 512 trains the same network as a
    shard weight of 2 (both tested). The validation shard's sidecar, if any, is not read.
  - Recorded: `data.train_row_weights` (per training shard: file, SHA-256, rows, rows not 1, mean;
    `null` without a sidecar), `stream.parts[].row_weights_sha256`, and the network's metadata
    `train_row_weights_sha256` when a row weighs other than 1.
- **Start:**
  - The training waits for the validation shard and at least one shard, then starts with every
    shard present.
  - Later shards are checked, converted and hashed on a thread of their own while an epoch
    trains, and join between epochs.
  - More than N shards is an error.
  - When all N are there from the start, the training is exactly the `--train` one (tested).
- **Report:** `stream`:
  - `parts[]`: name, rows, SHA-256, weight, row weights' SHA-256, seconds until ready, first epoch;
  - `first_full_epoch`;
  - `partial_epochs`;
  - `epochs_trained_on` (per shard).

  Each epoch in `history` gives its `shards`, `rows` and `all_in`.
