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

## Results, 2026-10-16

**What was run.** Two trainings of the trick-3 oracle network O_3 [the network that values a
position at the start of trick 3, with 24 cards left], each checked as the hybrid H_3 [exact search
through tricks 1 and 2, the network from the start of trick 3] against exact search on the
trick-2 golden set [a fixed test set]: the 400 trick-2 decisions of seed 1, each valued on 320
worlds [guessed deals of the hidden cards], 1,697,280 values in all. Each ran as the runner's
`hybrid_rung` chunk (`docs/m3-hybrid/vast.md`) on one rented vast.ai machine with 184 threads and one
graphics card (GPU); each check recomputed 1% of the stored exact worlds (1,284 worlds) with the
current binary and found them byte-identical. The reports (the check's text and JSON, the training
curve, the step timings, the machine) are in each run's R2 bundle, `runs/<run_id>/out/rung/`.

**How to read the tables.** Seats, decision loss [the money chance given away per decision by
playing the hybrid's card instead of exact search's, shown as its 95% upper bound, in percent of
money chance], bias, agreement and the standard are as in `rung1.md`, "Results, 2026-10-16": **at
most 0.0100% overall and in every band (seats 1-2, 3-4, 5-8), and every team's bias at most
0.0100%**; bold marks a number over the line. Seats 1, 4 and 5 are not part of the standard; they
are shown because they are the weakest seats here (48-52 decisions each, so their bounds are
coarse). *Labels* are the training positions with their exact values: random play, self-play
(rules-v1 bots against each other) and harvested [positions the hybrid's own search reaches,
labelled exactly; section 1 above]. *Cap* is the most harvested positions kept per decision; the
three percentages are the shares of harvested decisions whose decider sits at seats 1-2 / 3-4 /
5-8. Both networks were validated on the same 100,000 random-play positions (seed 401). Epochs are
counted from 0 as the trainer reports them, so "best 13" is the fourteenth pass.

**Table 1: what each run was.**

| run | labels: random play + self-play + harvested | network widths | machine | golden set | training | whole rental |
|---|---|---|---|---|---|---|
| round 1 (`m3-hybrid-t3-r1c`) | 12.8M + 3.2M + none; seeds 311, 312 | 1024-512-128 | AMD Ryzen Threadripper PRO 7995WX 96-core workstation chip (184 threads) + RTX 3090 graphics card (24 GB), $1.135/h | generated in the run: 400 x 320 exact trick-2 worlds in 1,057 s on 184 threads | 9.7 min (582 s), 19 epochs (best 13) | 48 min, about $0.90 |
| round 2 (`m3-hybrid-t3-r2`) | 12.8M + 3.2M + 8M harvested (cap 32 per decision; decisions at seats 1-2 / 3-4 / 5-8: 40/30/30%; search driven by round 1's network `b89b26d6...`); seeds 311, 312, 313 | 1024-512-128 | AMD Ryzen Threadripper PRO 7995WX 96-core workstation chip (184 threads) + RTX 4090 graphics card (24 GB) | stored, reused from round 1 (fetched in 3 s; 1% recheck byte-identical) | 10.8 min (650 s), 25 epochs (best 18) | 42 min, about $0.79 |

**Table 2: what the check found** (95% upper bounds of decision loss in percent of money chance;
bold = over the standard).

| run | overall | seats 1-2 | seats 3-4 | seats 5-8 | largest team \|bias\| (%) | agreement | seat 1 | seat 4 | seat 5 | standard met? |
|---|---|---|---|---|---|---|---|---|---|---|
| round 1 (`m3-hybrid-t3-r1c`) | **0.0105** | **0.0138** | 0.0091 | **0.0133** | **0.0108** | 84.5% | **0.0240** | 0.0062 | **0.0355** | no (overall, 1-2, 5-8, bias) |
| round 2 (`m3-hybrid-t3-r2`) | **0.0111** | **0.0156** | **0.0171** | **0.0109** | 0.0092 | 83.2% | **0.0280** | **0.0229** | **0.0224** | no (overall, 1-2, 3-4, 5-8) |

Notes. Mean decision loss (not the upper bound) was 0.0073% in round 1 and 0.0082% in round 2.
Seat by seat (95% upper bound, %, seats 1 to 8): round 1: 0.0240, 0.0068, 0.0143, 0.0062, 0.0355,
0.0183, 0.0140, 0.0027; round 2: 0.0280, 0.0066, 0.0175, 0.0229, 0.0224, 0.0186, 0.0149, 0.0025.
Team biases (the decider's own team, the next, the opposite, the previous; %): round 1 -0.0090,
+0.0108, +0.0030, -0.0047; round 2 -0.0092, +0.0079, +0.0061, -0.0049. The networks are `b89b26d6...`
(round 1) and `22c76276...` (round 2). Where the time went (seconds): round 1: labels random 786,
rules 202, validation 7, golden worlds 1,057, training 582, check 64 (the chunk took 2,705 s);
round 2: stored inputs 4, labels random 777, rules 199, harvest 600, validation 7, training 650,
check 63 (2,307 s).

**Speed of the hybrid.** At trick 2 the hybrid values a world in 12.7 ms (873 search nodes, 559
network evaluations at 21.5 microseconds each) where exact search takes 1,289 ms (9,786,559 nodes):
about 100 times faster per world (12.5 ms against 1,288 ms in round 2, 103 times). The exact figure
is from the 1,284 worlds recomputed in the check.

**The golden set, generated once and reused.** Round 1 generated the 400 decisions (240,642 bytes)
and their exact worlds (18,495,472 bytes) in 1,057 s on 184 threads, about 54 thread-hours. Round 2
fetched the same two files from storage (same SHA-256: decisions `7cfc09e0...`, worlds
`644c5506...`) instead of generating them, and its 1% recheck (1,284 worlds, recomputed with the
current binary) was byte-identical. These are not the `m3-agree-t2d` files named in section 3 above
(there the decisions file is `159d59c6...`); round 1 made its own.

**The harvest of round 2** (sidecar `labels/t3-harvest-s313.npy.json`). The search was driven by
round 1's network. It covered 311,348 decisions, 219,362 of them capped at 32 positions (70%), and
kept 8,000,000 labelled positions, in 600 s (391 million solver nodes, 250 million network
evaluations). Distinct positions reached per decision, by the decider's seat, 1 to 8: 2110.6,
759.0, 301.8, 117.8, 46.7, 18.2, 7.0, 2.7 (mean over all decisions 642.5, median 144, largest
27,511). Rows kept by seat: 1,978,062; 1,967,853; 1,457,209; 1,385,134; 594,188; 389,535; 164,277;
63,742.

**What the numbers say.**

1. Round 1, from random-play and rules-v1 labels alone, is a near miss: overall bound 0.0105%
   against 0.0100%, seats 3-4 inside the line (0.0091%), seats 1-2 at 0.0138% and seats 5-8 at
   0.0133%, and the largest team bias 0.0108%.
2. The harvest at cap 32 did not help. The overall bound went from 0.0105% to 0.0111% and seats
   1-2 from 0.0138% to 0.0156%, both small; seats 3-4 went from 0.0091% to 0.0171%, the one clear
   step the wrong way; seats 5-8 improved from 0.0133% to 0.0109%, and the largest team bias from
   0.0108% to 0.0092% (inside the line now). There is no repeat of a trick-3 training yet, so
   the size of single-run noise is not measured here (at trick 4 it was up to about 0.0025% overall).
3. On the validation positions the second network is a little better (validation loss 1.4971
   against 1.5131, outcome accuracy 87.2% against 86.4%), yet its decisions in the check are not,
   so the validation loss does not predict the check here. (Round 2 also had 24M rows against
   16M, and ran 25 epochs against 19.)
4. Trick 3 is a much harder function than trick 4: the network's outcome accuracy [how often it
   gets the ending of the hand right] is 87% here against 98% at trick 4.
5. Round 3 is running to test capacity: 32M random-play and rules-v1 labels plus 8M harvested,
   and a network of 2048-1024-256 units. Its result will be added here.
