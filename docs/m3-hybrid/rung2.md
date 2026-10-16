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

## 4. Trick-1 checks on the saved exact worlds (spec §3, "Trick-1 checks for O_3 and O_2")

Branch `m3-hybrid-t1check`, 2026-10-16. Once O_3 passes on the trick-2 golden set, the spec asks
for H_3 on trick-1 worlds: first the known hard world, then every exact trick-1 world earlier runs
saved. Those worlds cost 10–160×10⁹ solver nodes each, so they are read back, not solved again.
Nothing was rented and nothing was written to R2 for this.

### What the saved files hold

All four runs used the same decisions file (`decisions-t1.jsonl`, 32 trick-1 decisions, seed 1,
16,088 bytes, SHA-256 `b23c8b0d…`). Their worlds files are empty or partial: every world is in the
checkpoint files (`<worlds file>.ckpt`, one line per world: the candidates' exact values, the
microseconds and, from 2026-10-16, the solver nodes it took).

| run (R2 `runs/<id>/…`) | file | bytes | worlds | values version (burn rule) | how it was solved |
|---|---|---|---|---|---|
| `m3-t1-tables-1` | `partial/tt12g/000000/out/measure/exact-t1-n4.jsonl.ckpt` | 10,644 | 26: decisions 0–7 (leader and second seat), worlds 0–3 but six | 2 (searched burn) | one thread a world, 12 GB tables, EPYC 7542; decision 0 world 3: 13,057 s, 79.3×10⁹ nodes |
| `m3-t1-leader-layouts-1` | `partial/big4x46/…/exact-t1-n4.jsonl.ckpt` and `exact-t1-n4.jsonl` | 6,615 and 2,731 | 15: decisions 0–3 (the file holds decisions 0 and 1 whole) | 2 | 46 threads a world |
| `m3-t1-leader-layouts-9654` | `partial/big2x192/…` (same names) | 6,616 and 2,731 | the same 15 (its worlds file has the same bytes as layouts-1's) | 2 | 192 threads a world |
| `m3-agree-t1-ab1` | `out/measure/exact-t1-n80.jsonl.ckpt` | 117,404 | 355: decisions 0 (77), 1 (79), 3 (80), 4 (40), 6 (79) | 1 (rules-v1 burn; binary `56851ce`; no table, nodes or version recorded) | position-aware threads, up to 92 a world |

Joined: **31 distinct worlds valued with the searched burn** (decisions 0–7, worlds 0–3, except
decision 2 world 2, which no run finished) and **355 with the rules-v1 burn**. Every world that
two or three runs valued has identical values in each. 19 worlds are in both sets: 17 have the same
values under both rules, 2 differ (decision 3 world 2 and decision 4 world 2).

### What was built

- **`search::saved`**: reads checkpoint lines and worlds records (both `search::measure`'s own
  formats, one JSON line each, told apart by their fields; a torn line is skipped and counted),
  joins a world found in several files (refusing different values), and splits the worlds into one
  set per burn rule: values version 1 (absent) → `rules_robber_burn = true`, the current version
  → the searched burn, anything else refused. Each world is checked against the decisions file
  (decision fingerprint, candidates, world source, equity table where recorded; what is not
  recorded is listed in the report) and regenerated from (seed, label, decision, world number)
  exactly as the yardstick draws it.
- **`search hybrid-check --saved RUN=FILE …`** (`search::hybrid::check_worlds`, the same code as
  the golden-set check, now for any chosen worlds): the sets run at once, the threads shared out by
  worlds, one world per thread, the largest saved worlds first. Per world: every candidate's exact
  and H_k value, the largest error over the four teams, H_k's time, nodes (by trick) and network
  evaluations, whether another seat robs later in trick 1, and every saved time and node count.
  Per decision: the choices and the decision loss (a decision with one world has a loss too; the
  tables below count every decision). Per set: the §3 tables exactly as `hybrid-check` prints them
  (decision loss with its bootstrap bound, bias per team, SD, 99.9th percentile, largest, overall,
  by band and by position), the time line and the verdict. Then the `--hard` worlds on their own
  lines with the saved exact times and the ratio, and the peak memory. `--only D:W,…` restricts
  the check (the hard world alone: `--only 0:3`); `--plan-only` prints what would run and the
  verification sample, solving nothing.
- **Verification** (spec: "after confirming a sample is byte-identical on the current exact solver
  with that switch"): `--verify-pct P` recomputes P% of the saved worlds (drawn by seed, decision
  and world, as the golden-set check), `--verify-rule rules-v1` only in the rules-v1 set, each
  solve on `--verify-solve-threads` threads with a `--verify-tt-mb` table, before the hybrid; a set
  with any difference is not compared, and the report lists the worlds. Without it (the default,
  0) the report says "NOT re-verified" for the set, and for the rules-v1 set that the identity on
  the current solver is not confirmed.
- **Runner**: a format-2 chunk kind `hybrid_t1check` (`crates/runner/src/t1check.rs`). The network,
  the decisions file and the saved files are the manifest's `inputs` (content-addressed
  `inputs/sha256/<hex>`, fetched with `remote::fetch_input` and SHA-256 checked before the chunk
  starts; the launcher checks they exist before renting), named in the params. Outputs:
  `out/hybrid/t1check.txt` and `t1check.json` (every world's values in full); each world's line in
  the chunk's log as it finishes, so a stopped run keeps what finished. `partial_cpu_ok` only in the
  genuine tail (fewer worlds left than threads). Proposal: `docs/vast/m3-hybrid-t1check.manifest.json`
  (the O_3 placeholder must be filled in; not run).

**Tests**: the reader on synthetic records (both forms, an old line, a torn line, joins and
refusals); the real files (small copies in `crates/search/testdata/t1-saved`, 32 KB: the decisions
file, `m3-t1-tables-1`'s checkpoint, the layouts' worlds file and nine `m3-agree-t1-ab1` lines; the
hard world reads back with 13,056,957,069 µs and 79,320,348,281 nodes); at trick 3 with the frontier
at trick 4 the saved-worlds check gives exactly `hybrid-check`'s numbers (stored or recomputed,
1 and 4 threads, chosen worlds regenerated exactly); the burn rule reaches the hybrid's and the exact
solves (at trick 1, frontier at trick 2, worlds where a later seat robs: values follow the switch and
differ between the rules); with the production table (`FORTYONE_EQUITY`), `m3-agree-t1-ab1`'s
decision 6 world 25 (seat 2 robs later) is byte-identical on this solver with `rules_robber_burn` and
differs without it; the runner chunk with a fake `search`, its settings, and the committed manifest.

### Measured on this VM

Small timing network (512-256-64, 120,000 labels, 63% outcome accuracy, `dc5a406f…`; its values are
not a candidate), one thread a world, 64 MB tables. The VM was shared with another session's jobs,
so some times may be up to about twice too long.

**Verification sample (rules-v1 set):** five `m3-agree-t1-ab1` worlds (6:25, 3:73, 4:33, 6:4, 6:63,
the cheapest by their saved time; three with a later robber) recomputed on this binary with
`rules_robber_burn`: all byte-identical (0.2–10 s each). With the switch off, 6:25 differs. This is
a cheap, not a random, sample; the rental should still draw one (below).

**H_3 time per trick-1 world, searched-burn set** (24 of the 31 worlds; the saved exact time is
`m3-t1-tables-1`'s, one thread):

| world | exact (s) | H_3 (s) | world | exact (s) | H_3 (s) |
|---|---|---|---|---|---|
| 6:1 | 11 | 0.5 | 7:1 | 1,287 | 37 |
| 4:0 | 62 | 1.6 | 3:1 | 1,315 | 63 |
| 3:3 (robber) | 67 | 21–26 | 2:1 | 1,458 | 99 |
| 3:0 | 129 | 4.5–6 | 5:0 | 1,518 | 30 |
| 4:3 | 214 | 6.1 | 5:1 | 1,820 | 45 |
| 7:3 | 230 | 5.5 | 1:3 | 2,009 | 93 |
| 2:0 (robber) | 249 | 14–16 | 7:0 | 2,072 | 67 |
| 6:0 | 285 | 20 | 0:1 | 2,310 | 76 |
| 6:2 | 527 | 16 | 5:2 | 2,684 | 70 |
| 6:3 | 586 | 22 | 5:3 | 3,078 | 168 |
| 7:2 | 662 | 11 | 2:3 | 3,701 | 146 |
| 4:2 | 1,072 | 24 | 0:2 | 4,483 | 64 |

Together 1,103 s against 31,785 s exact: about 29 times faster (3 to 70 times by world; slowest
relative to exact where a later seat robs). The other seven did not finish within the 5-minute
limit: **the hard world 0:3** (exact 13,057 s), 4:1 (12,648 s) and the five largest (0:0, 1:0, 1:1,
1:2, 3:2: 1.5–1.6×10¹¹ nodes, solved only on 46–192 threads). At about 29 times the hard world
should take 5–15 minutes on one thread; the five largest perhaps 10–30 minutes each.

**Rules-v1 set sample** (decisions 0, 1, 3 worlds 5/25/45/65 and 4, 6 worlds 5/25/35): leaders
8–162 s (11 finished, mean 75 s; 1:65 over 250 s), second seat 0.02–20 s (mean 7 s).

**Time goes to the network**: about 90% of H_3's time at trick 1 is its network evaluations (from
a few thousand to 17 million a world, 8–14 µs each here); the solver's nodes are 10⁴–5×10⁷ a
world. A 1024-512-128 O_3
is about four times slower per evaluation (`nn-speed.md`), so H_3 with it about 3.5–4 times slower.

**Memory per world**: about 70 MB a thread (its 64 MB table, which the large worlds fill, and a few
MB of network caches), plus about 60 MB for the process (the equity table, 26 MB): 184 threads
about 13 GB. A larger table barely helps (64 → 512 MB: 3–7% faster).

### Cost of the full check (estimate, not measured on a rented machine)

Work, small network: the searched-burn set about 1.5–3 thread-hours (1,103 s measured + the seven
largest), the rules-v1 set about 5–10 (236 leader worlds at about 75 s, 119 others at about 7 s,
with a long tail): **7–13 thread-hours**; with a 1024-512-128 O_3, 25–50. On a 184-thread machine
that is 3–5 minutes of full load (10–20 with the large network), but every world runs on one thread,
so the wall time is set by the largest single worlds: about 0.5 hour with the small network, 1.5–2.5
hours with the large one, plus about 10 minutes of start-up. At about $1.1/h: **$0.5–1 (small
network), $2–3 (large)**.

**Verification sample** (proposed: 3% of the rules-v1 set, `--plan-only` lists it): 8 worlds
(0:31, 0:54, 0:64, 1:3, 1:24, 4:18, 6:24, 6:54), which took 7–739 s on the old run's many threads
a world. Solved first, three at a time on 46 threads each with 8 GB tables, they should take about
half an hour (0:64 sets it), before the rules-v1 set's hybrid starts: about $0.5 more. The
searched-burn set has the current values version and is not re-verified in the proposal
(`verify_rule` `rules-v1`); its 31 worlds were each solved by up to three runs with identical bytes.

**To run it** (owner's approval first; nothing was rented): train or pick the O_3 network, then
upload the inputs (`python launch.py put-input FILE` for the network and for each saved file,
downloaded from the R2 paths above; the decisions file and the saved files are already named by
their SHA-256 in the manifest), put the network's SHA-256 in place of the placeholder, and launch
`docs/vast/m3-hybrid-t1check.manifest.json`. The known hard world alone first, if wanted: the same
manifest with `"only": "0:3"` and `"verify_pct": 0` (one world, one thread: a small machine
suffices).

## 5. O_2 round 1: a train-only rung, checked on the saved trick-1 worlds

O_2 (exact trick 1, the network from the start of trick 2) cannot have a generated golden set: its
golden set would be trick-1 decisions with exact worlds. Its check is §4's, on the saved worlds,
with `--cut 2`. So `hybrid_rung` has a train-only mode (`"check": false`, `vast.md`), and every
rung now shares its integer network content-addressed (`net.int_key`), ready to be the
`hybrid_t1check`'s `o3`/`o2` input.

**Cut 2 end to end, on this VM:** `search labels --trick 2` (random play and rules-v1), the Python
trainer on those shards (39 hand endings at trick 2), and `search hybrid-check --trick 1 --cut 2
--saved …` with the resulting tiny network on saved worlds (the hard world included) all run.
With a tiny network H_2 takes 0.3 s on the hard world (67,000 network evaluations): the O_2 trick-1
check will be cheap; the labels are the cost.

**Price of a trick-2 label** (one thread, 20 labels, this VM, shared with other jobs): random play
2.67 s a label, rules-v1 3.22 s. The proposal's 1,300,000 labels (1M random play, 250,000 rules-v1,
50,000 validation) are then about 1,000 thread-hours: about 5.5 hours on 184 threads, about $6 at
$1.1/h if a rented thread is as fast as this VM's; a hyper-threaded thread on a fully loaded machine
may be up to half again slower (about 8 hours, $9). Training on the GPU adds up to `max_minutes`
(200). To be priced with the owner before running (spec §5).
