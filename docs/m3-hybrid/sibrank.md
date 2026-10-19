# Fix E: sibling groups and a pairwise ranking loss for O_3

Owner, 2026-10-19. O_3 trained on 80M rows instead of 40M had a smaller value error (SD 1.85% →
1.76%) but no better golden decision loss (overall 95% bound 0.0068% vs 0.0080%, leader 0.0201% vs
0.0264%: within noise). A decision depends on the **differences** between the positions its cards
lead to, not on each value alone. Decision-focused learning (Elmachtoub & Grigas, "Smart
Predict-then-Optimize", 2022) says: train on what the decision needs; RankNet (Burges et al., ICML
2005) is the proven loss for an order. Fix E adds both, and an A/B against today's training.

## 1. Sibling groups (`search labels --source siblings`)

`search::oracle_labels::sibling_plan`, through the harvest machinery (same decisions, pooled
exact labels, streamed shard, determinism):

- **Decisions:** harvest decisions (`harvest_root`): the yardstick generator at trick k−1, the
  decider's position from `--harvest-positions`, a world drawn from the decider's belief.
- **The walk:** in that world, the hybrid's own search (solver max^n, the frontier at the start of
  trick k valued by the driver: `--harvest-net`, else exact values). From the decision, every seat
  still to play in the trick is a **choice point** in turn, along the line the search expects
  (each seat plays the best card for its own team by those values; the solver's tie rule).
- **A group:** at a choice point, every legal card is followed to the frontier along its principal
  line (every later card the best for its seat's team); the distinct frontier positions reached
  (canonical form) are the group; the seat's team is the **mover**. Groups of fewer than two
  positions are dropped (one card; every card leading to the same position; lines on which the
  game ends before the frontier). At most `--harvest-cap` groups a decision (runner default 8).
- **Why this rule:** when the hybrid decides, a card's value is the network's value at the end of
  that card's principal line (max^n copies a leaf's vector up unchanged), and every later seat in
  the trick chooses the same way for its own team. A decision goes wrong when the network orders
  those positions differently from their exact values. So the groups are exactly the positions the
  search compares, and every position is labelled exactly. With the network as driver the lines
  are the ones H_k plays; with exact values they are exact best responses (both supported; the
  runner uses `harvest_net`). Tested: with exact values a decision's first group's mover values are
  exactly its cards' exact values.
- **Rows:** ordinary label rows, source tag 4 (`SOURCE_SIBLINGS`); decisions 0, 1, 2, … in order,
  groups in walk order, the shard cut at the row count (its last group may be cut). A pure function
  of (seed, row index), the driver and the binary: tested for thread counts 1/3, through the
  streamed writer, and as a prefix (41 rows = the first 41 of 90).
- **Format:** the sidecar `NAME.npy.groups` (`crates/nn/src/groups.rs`, `formats.load_groups`):
  one packed record a row, `group` <u4 (0, 1, 2, … in row order, rows of a group consecutive) and
  `mover` u1 (the mover's index **in that row's own rotated value order**: siblings can have
  different trick-k leaders). Written atomically before the shard is renamed into place. The
  shard's JSON sidecar gets a `siblings` record (groups, sizes, the sidecar's SHA-256, the rule).
  Old shards have no sidecar and read exactly as before; old readers ignore it.

## 2. Trainer (`fortyone_nn.train`)

- `--pair-weight W` (default 0: none) and `--pair-scale S` (default 100). Loss = the usual loss of
  every row + W × `pair_loss`: over each pair (i, j) of a group, d = true money chance of the mover
  at i minus at j (each row's own mover index), p = the same from the predicted outcome
  distribution (the mixture, as the value is derived everywhere), loss softplus(−S·sign(d)·p)
  weighted by |d| (a pair of equal true values weighs nothing), the weighted mean over the batch's
  pairs. S = 100 turns a gap of 1% into a logit of 1 (measured nonzero sibling gaps: median 2.9%,
  quartiles 1.0%–5.8%).
- **Mixed batches** (`PairIndex.epoch`): each epoch, the groups in a random order are cut into the
  full batches by their running row count, whole groups at the start of each batch (at most
  `slots` = ⌈group rows / full batches⌉ + largest group rows), the rest of each batch ordinary rows
  in a random order, the last partial batch ordinary only. Every row is still seen once an epoch
  by the cross-entropy, so absolute calibration is kept. The epoch's plan is put on the device
  once; the CUDA-graph step reads a batch's groups from fixed buffers (tested through the graph
  path without a GPU, and with the resident, staged and disk feeds).
- `--tie-aware` (default off): the outcome target is the set of possible outcomes whose exact u64
  money vectors equal the label's (`formats.tie_bits`, computed from the u64 vectors, not floats),
  loss −log of their summed probability; with one outcome in the set it is the standard loss,
  value and gradient exactly (tested). Validation loss and accuracy follow the same rule.
- Defaults: byte-identical networks to the trainer before this change (checked against commit
  a011cba's trainer, resident and disk feeds: the same int SHA-256; and in the tests: a groups
  sidecar with `--pair-weight 0` trains the same network as none). The network's metadata names
  `pair_weight`, `pair_scale`, `groups_sha256` and `tie_aware` only when they are on; the report
  gains `pairs` and per-epoch `train_pair_loss`, `pair_batches`, `train_pair_share`.
- Groups come from whole `--train` shards only (refused with `--watch-dir`).

## 3. Runner (`hybrid_rung`)

`labels_siblings` (rows), `labels_siblings_seed` (107), `siblings_positions` (default the harvest's
shares), `siblings_cap` (8); needs `harvest_net` at cut ≤ 3; one shard after the race shard.
`pair_weight`, `pair_scale`, `tie_aware` pass the trainer flags only when set (refused without a
sibling shard, with `label_parts` > 1 or the audit loop). `label_inputs` takes set `siblings` (and
a `groups` key, required for it); `upload_shards` stores the groups sidecar and lists it.

## 4. Local measurements (this VM: 4 vCPU shared with other agents, load 4–9)

- **Speed:** 10,000 sibling rows at trick 3 driven by the stored O_3 (`b89b26d6…`): 17 s wall on 4
  threads, 6.2 thread-ms a row (search 0.85, exact labels 5.4); random-play labels on the same VM
  13 thread-ms a row. About 160 rows per thread-second.
- **Groups:** 934 decisions, 3,375 groups, 10.7 rows a decision; sizes 2: 36%, 3: 32%, 4: 32%
  (mean 2.96). 28% of pairs have unequal true values for the mover (the rest weigh nothing).
- **Ties:** in 44,000 trick-3 rows (random, siblings, validation) no two possible outcomes share a
  money vector, so `--tie-aware` is expected to change nothing at trick 3.
- **Smoke training** (30k random + 12k sibling rows, 256-128-64, batch 256, W 1, tie-aware, CPU):
  the pairwise loss fell 0.78 → 0.65 → 0.47 → … → 0.20 over 8 epochs, 19% of the loss in epoch 1;
  integer network valid (money chances sum to 2^32, int vs float max 5.8e-7).
- **Small A/B on this VM** (the same 42k rows, 10 epochs, no tie-aware; judged on 6,000 held-out
  sibling rows, seed 7103, by the integer networks): A (cross-entropy only) best epoch 5, val loss
  2.280, value SD 5.02%, gap-weighted misordered pairs 15.6%, mean regret of the best-looking
  sibling 0.278%; B (W 1) best epoch 8, val loss 2.432, SD 5.83%, misordered 18.3%, regret 0.310%.
  **B was worse here**: its training pair loss fell to 0.14 while held-out order got worse, i.e.
  with 12k sibling rows and a small network the pairwise term overfits the training pairs. This
  says nothing reliable about 6M rows and the full network, but it is a warning: if B loses on the
  golden set, the next step is a smaller weight (0.25) before dropping the idea.

## 5. The A/B (`docs/vast/o3-sibrank-{a,b}.manifest.json`; not launched)

Both: the stored 40M trick-3 base as `extra` sets plus the validation shard, 6M sibling rows
(seed 351, `siblings_positions` 0.5,0.25,0.25 so the leader's choice points are not under-sampled:
a decision at position p gives groups at p…8), driven by the stored O_3 `b89b26d6…`, stored
(`upload_shards`); 2048-1024-256 from scratch, seed 1, 40 epochs, patience 4, batch 1024; the
stored 2,000 × 320 trick-2 golden check. A: cross-entropy only. B: + `pair_weight` 1,
`pair_scale` 100, `tie_aware`. The manifests differ only in run id, note and B's three settings
(tested). Both make the identical sibling shard (pure in seed, network and binary), so they can run
side by side; a later run reuses it through `label_inputs` (set `siblings` with its `groups` key
from `out/hybrid/shards.json`). B's slots: 138 of each batch of 1,024.

**Cost estimate**, from o3-base80-1 (Threadripper PRO 9995WX, 192 threads, RTX 5090 D, $0.64/h:
fetch of the stored 40M 12.5 min; 115 s an epoch for 80M rows; 24 epochs; check about 35 min;
129 min in all): per run about 3 min boot, 12.5 min fetch, 4–6 min for the 6M sibling rows, about
66 s an epoch for 46M rows × 25–35 epochs ≈ 30–40 min, 35 min check: **about 1.5 hours, $1.0 at
$0.64/h (up to $2.6 at the $1.70 cap); both runs about $2–5.**
