# Hybrid yardstick, rung 2 (O_3): what changes for trick 3

Branch `m3-hybrid-t3`, 2026-10-16 (Mauritius). Spec: `docs/M3-HYBRID-SPEC.md` (§3, §4, §5 source
(c)). Rung 2 is the trick-3 network O_3: the hybrid searches tricks 1 and 2 exactly and asks O_3 at
the start of trick 3 (24 cards left, 24 possible hand endings). It is checked on the trick-2
golden set (400 decisions × 320 worlds). This note covers the two pieces built for it; nothing
was rented.

## 1. Harvesting at trick 3, driven by a network

**The problem.** Harvesting [collecting training positions from the hybrid's own searches,
`rung1.md` "Harvested positions"] gave every position the search reaches its exact value. At
trick 4 that is free (0.04 ms a solve). At trick 3 an exact solve costs about 9 ms, and a trick-2
leader decision reaches about 2,200 trick-3 positions, so the old way costs about 13 seconds per
leader decision.

**The fix.** The search is now driven by a network [the network's values are used at the cut,
exactly as the real hybrid H_3 runs], normally the first-round O_3. Every position it reaches is
recorded in the order reached; duplicates are dropped; the same seeded sample of at most 16 per
decision is taken as before; and **only the sampled positions are solved exactly** for their
labels. The labels are exact either way; only the choice of which positions to visit uses the
network.

```
search labels --trick 3 --source harvest --harvest-net o3-round1/int.safetensors \
  --count 1000000 --seed 103 --harvest-positions 0.4,0.4,0.2 --out t3-harvest-s103.npy \
  --equity equity-v0.npy --threads T
```

Without `--harvest-net` nothing changes: trick 4 keeps the exact-driven harvest (checked: the same
shard bytes as before this change). The shard format is unchanged (source tag "harvest"). The
sidecar now says which driver was used (`harvest.frontier_driver`: `exact`, or `net` with the
network's SHA-256, file name, trick, head, widths and training metadata) and adds the mean
positions reached and network evaluations per decision for each seat (`distinct_mean_by_position`,
`evaluations_mean_by_position`).

**Measured on this VM** (4 vCPU; a small O_3 trained here for timing only: 120,000 trick-3 labels,
512-256-64, 3 epochs, 17 s, outcome accuracy 63%, net `dc5a406f…`):

| | net-driven | exact-driven (before) |
|---|---|---|
| time per 1,000 harvested labels (thread-seconds), positions uniform | 9.6 (2.5 s on 4 threads) | not measured whole; by band below |
| same, positions 1–4 oversampled (0.4,0.4,0.2) | 8.6 (2.3 s on 4 threads) | |
| leaders only (3 decisions) / seats 3–4 (10) / seats 5–8 (20), same decisions both ways | 11.5 / 14.0 / 12.4 | 818 / 191 / 31 |
| share of the time in the searches | about 7% (the rest is the exact solves of the sample, about 9 ms each) | 100% |

So 1 million harvested trick-3 labels cost about 2.7 thread-hours (about 25 s on 384 threads,
5 minutes on 32), instead of hundreds of thread-hours.

**Positions reached per trick-2 decision, by the decider's seat** (net-driven, 809 decisions,
about 100 per seat; mean distinct positions, with network evaluations in brackets):

| seat 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| 2,209 (2,784) | 684 (815) | 274 (308) | 135 (152) | 46 (50) | 19 (20) | 6 (7) | 3 (3) |

Median 48, 90th percentile 1,152, largest 13,090 (a leader with many choices). The search itself
takes about 8 ms per decision with this network.

**Does the network change which positions are visited?** Hardly, as at trick 4. On 96 trick-2
decisions (12 per seat) the network-driven search reached 99.94% of the positions the exact-driven
search reaches (and 3% more: a weaker value prunes a little less), and 91 of the 96 decisions
gave byte-identical label rows either way. This settles the "O_3 caveat" of `rung1.md`: driving
the search with the first-round network is as good as driving it with exact values, at about 1%
of the cost.

**Determinism.** Each decision's search runs on one thread, so the positions reached and their
order never depend on the thread count; the network's caches never change a value. Tested: the
same shard bytes on 1 and 4 threads (library tests at tricks 3 and 4, and the command at trick 3:
`64177332…` both times).

**Tests** (`crates/search/src/oracle_labels.rs`): the network-driven search is exactly the hybrid
H_k with that network (same candidate values), it reaches only real start-of-trick positions (each
in the set of all positions reachable from the decision's world), one per canonical form, each
valued by the network during the search; the rows are a sample of at most the cap in reach order,
each labelled with the exact solver's value of its own position (and its solver nodes); a second
run with fresh memory gives the same rows; 1 and 4 threads give the same shard; where the
network's search reaches what the exact search reaches, the rows are the exact driver's rows.

## 2. Passing the network and a stored golden set to `hybrid_rung`

Three new optional settings of the runner's `hybrid_rung` chunk, each a stored file named by its
key in R2, `inputs/sha256/<SHA-256 of the file>` (the form `sha256:<hex>` is accepted too):

- `harvest_net`: the first-round `int.safetensors` (from that run's bundle, `hybrid/net/`). It
  is passed to the harvest's `search labels --harvest-net` only (needs `labels_harvest` above 0).
- `golden_decisions_input` and `golden_worlds_input` (both or neither): a stored golden set,
  reused instead of generated. Regenerating the trick-2 set costs about 90 thread-hours per run.
  `verify_pct` must then be at least 1: the check recomputes 1% of the stored worlds with the
  current binary and stops on any byte difference (spec §4). The decisions file must hold
  `golden_count` decisions at `golden_trick`, or the chunk stops.

The runner fetches them first, before installing anything, with the same code that fetches a
manifest's `inputs` (`remote::fetch_input`), and checks each file's SHA-256: a missing key or a
file with other bytes stops the chunk at once. The launcher also checks, before renting, that
the keys exist in R2 (and refuses a value that is not a key, such as a placeholder).
`rung.json` records each input (key, SHA-256, size) under `inputs`, the network's SHA-256 beside
the harvest shard, and `golden.stored: true` with both keys.

**Sharing a generated golden set.** When a run generates its golden set, the runner now uploads
the decisions and worlds files to `inputs/sha256/<hex>` as well (as `upload_shards` does for
shards), and `rung.json`'s `golden.reuse_params` gives the two settings that reuse them in the next
run. A failed upload is only a warning: the files are in the run's bundle too.

## 3. The proposed round-2 run

`docs/vast/m3-hybrid-t3-round2.manifest.json` (a proposal; not run): cut 3; 2.4M random-play and
0.6M rules-v1 trick-3 labels (seeds 101, 102), 1M harvested labels driven by the round-1 O_3
(seed 103, cap 16, positions 1–4 oversampled), 100,000 for validation (seed 201); the 1024-512-128
outcome-head network as rung 1b; checked against the stored trick-2 golden set (400 × 320, seed
1) with a 1% recheck. Three placeholders must be filled in first (the runner and the launcher
refuse them as they are):

1. `harvest_net`: `inputs/sha256/<SHA-256 of round 1's int.safetensors>`, after
   `launch.py put-input int.safetensors`. Round 1 is the same file without `labels_harvest`,
   `labels_harvest_seed`, `harvest_cap_per_decision`, `harvest_positions` and `harvest_net`.
2. `golden_decisions_input` and `golden_worlds_input`: the trick-2 golden set of run
   `m3-agree-t2d` (spec §4), uploaded with `launch.py put-input`. Its files (copies in
   `/home/user/41Bots-golden/t2d/m3-agree-t2d/out/measure/` on the build VM) have these SHA-256:
   - `decisions-t2.jsonl` (223,935 bytes): `159d59c6ab7c58816688c9bbb9f8acdde575723038f7b0127bf4f6eef787a6c8`
   - `exact-t2-n320.jsonl` (18,415,006 bytes): `121c535fcf5f5eb4b2380bb3cdb22c4246f0f86e05da36182ae8c9795405966b`

   They were made by the belief sampler (they carry world counts), so the check's default
   `--source belief` applies.

Estimated cost (from this VM's speeds; not measured on a rented machine): labels about 11
thread-hours (3.1M random-play, rules-v1 and validation rows at about 9 ms each, 1M harvested at
about 10 ms; about 20 minutes on 32 threads, 5 minutes on 128); 4.1M rows of 1.4 KB make about 5.8 GB of shards; training as rung
1b (GPU); the check about 2 thread-hours (the hybrid on 128,000 trick-2 worlds, plus the 1%
recheck: about 1,280 exact trick-2 worlds at about 2.5 s). Without the stored golden set the
check's exact worlds would add about 90 thread-hours.

## Open points (for the owner)

- **When the stored golden set is rechecked.** The 1% recheck runs inside the hybrid check, the
  last step, after the labels and the training. If the stored worlds fail it (a value changed
  since `m3-agree-t2d` was made), the run's network is still saved and uploaded, but the check has
  to be run again on recomputed worlds (about 90 thread-hours). An early recheck step before the
  labels was not built; it would cost the same 1,280 trick-2 solves, done first.
- **Machine size.** The proposal keeps `min_cores` 32 (labels about 20 minutes); with 64 or more
  threads on offer the labels take 10 minutes or less.
