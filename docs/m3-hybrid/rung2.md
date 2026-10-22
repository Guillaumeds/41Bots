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
5. Round 3 was launched to test capacity: 32M random-play and rules-v1 labels plus 8M harvested,
   and a network of 2048-1024-256 units. It failed on a chunk error (run `m3-hybrid-t3-r3`) and was
   superseded by the overnight audit loop (`docs/EXPERIMENTS.md` C1-C2).

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

**The trick-2 audit loop** (branch `m3-hybrid-t2loop`, merged on 2026-10-17 as `m3-hybrid-t2merge`
with the overnight loops' builds; no trick-1 golden set, owner 2026-10-16): `docs/m3-hybrid/vast.md`,
"The audit loop at trick 2 (O_2)": position-level stop, the saved trick-1 worlds above as a rough
test reported every round (with a slice per group), the per-cut defaults, the manifest
`docs/vast/m3-hybrid-t2-loop.manifest.json` and its projected cost.

## Adopted O_3 (owner, 2026-10-17)
The overnight trick-3 audit loop's round 2 is the current O_3: 2048-1024-256, outcome head,
`inputs/sha256/f5766aa98a1d769e31122c94cc4da9790c84508fab5c432d0c0358fe3124faad`, trained on 40M trick-3 labels
(random 25.6M s321, rules 6.4M s322, harvest 8M s313) plus round 1's targeted shard. On 2,000 trick-2 golden decisions
x 320 exact worlds (generated overnight, stored): 95% upper bound 0.0068% overall, seats 1-2 0.0122%, 3-4 0.0080%,
5-8 0.0052%, own-team bias -0.0085% (the overall row passes; the leader's band does not). Later loops start from it
(additive mode) and must beat it on the same golden set.

## Exact leaves (G) (build of 2026-10-19)

**Why.** Fix "G" of the 2026-10-19 diagnosis (owner approved): O_3 is least reliable in the race
classes (`nn::outcome::race_class`: 2, several teams can win the game; 4, a team can win the match and
another the game; auditor SD 2.0% and 4.0% against about 0.95% in classes 0-1), and at trick 4 exact
values at the race-class frontier took H_4's bound from 0.0057% to 0.0016% (rung1.md §6). An exact solve
from the start of trick 3 costs milliseconds, so H_3 (trick-2 decisions: the trick-2 teacher, O_2's cheap
labels and later the app's trick-2 search) can afford one at the leaves where O_3 is weakest. This spends
compute for quality; no standard is lowered.

**Design** (`crates/search/src/exact_leaves.rs`; `search hybrid-check --exact-leaves SPEC
[--exact-leaf-tt-mb M]`):
- **The policy** is a union of terms joined by `+`: `race:LIST` (every leaf of those race classes;
  `race:4`, `race:2,4`, `race:1-4`), `band:LO-HI` (every leaf where O_3 gives the decider's team LO% to
  HI%, the bounds turned into units of 2^-32 once and compared as whole numbers), `all` (every leaf: H_3
  becomes exact search, for tests and ceilings), `none` (the default). Only whole numbers decide: the race
  class comes from the outcome mask and the games won, the band from the network's integer money chance.
  Tie-decided leaves are not a term: telling them apart needs two exact solves of every leaf (the cost of
  `all`); the race classes hold almost all of them (rung1.md §5).
- **An exact leaf's value** is `oracle_labels::exact_value` of the position as the search first reached
  it: the labels' exact solver, tie rule and fixed-point money chance, so an exact leaf equals its label
  exactly (tested against a solve from empty tables). Each world's search has its own `ExactLeaves`
  evaluator that decides once per distinct canonical form and memoises the value (as the network's values
  are), so a value depends only on the canonical form and the match, as the solvers require.
- **Speed without changing a value:** the leaf solves of one world share one table
  (`solver::SolveFamily::of_match` of the world: the values of solves from empty tables, fewer nodes), on
  a second per-thread solver memory kept from world to world (`--exact-leaf-tt-mb`, default 64 MB at cut 3;
  the runner counts a second table per thread in its memory plan). Every value is byte-identical for any
  thread count and machine; the leaf solves' times are reports only.
- **`none` is today's check byte for byte:** without a policy the network is not wrapped at all, and a
  policy that picks no leaf gives the same values, choices and errors (tested).
- **Report** (text and JSON `exact_leaves`): per race class the distinct leaves a world and the share
  solved exactly; the leaf solves' nodes and thread time a world; the hybrid's thread time a world with
  them and without them. `values.hybrid` ends `+exact:<policy>`. `hybrid-paired` also prints both checks'
  thread time a world. The runner's `hybrid_checks` takes `exact_leaves` per check (not with `truth`
  `"net"`).

**Measured on this VM** (4 threads, the stored trick-2 golden set, decisions 0-31, one per seat and
stratum, x 32 worlds, the adopted O_3 f5766a; distinct leaves per world's search: 355, of which class 0
159, class 1 117, class 2 64, class 3 0.3, class 4 14.7):

| policy | leaves exact | leaf solves a world | ms a solve | thread-ms a world | x none | bound all | own bias all | own bias seats 1-2 |
|---|---|---|---|---|---|---|---|---|
| none | 0% | 0 | - | 21.4 | 1.0 | 0.0241% | -0.127% | -0.160% |
| race:4 | 4.1% | 14.7 | 3.4 | 73.3 | 3.4 | 0.0241% | -0.053% | -0.155% |
| race:2,4 | 22.1% | 78.3 | 1.9 | 169.3 | 7.9 | 0.0241% | -0.058% | -0.124% |
| race:1-4 | 55.2% | 195.8 | 2.5 | 513.3 | 24.0 | 0.0132% | -0.028% | -0.021% |

Thirty-two decisions cannot judge the standard (the bounds above are for the sample only; one decision,
seat 3, keeps its loss under every policy). The paired comparison of race:1-4 against none on them: loss
-0.0057% (interval -0.0148% .. 0), own-team bias +0.099% (seats 1-2 +0.139%, interval +0.032% ..
+0.271%): the exact leaves remove most of the own-team undervaluation in this sample. With `--exact-leaves
all` on six golden decisions x 4 worlds, H_3's values equal the stored exact worlds in every world (loss 0,
every error 0): 1.18 s a world, 3.7 ms a leaf solve.

**Price of the rented run** (`docs/vast/o3-exactleaf.manifest.json`, run `o3-exactleaf-1`; written before launch, it ran on 2026-10-19: `docs/EXPERIMENTS.md` C10):
2,000 decisions x 320 worlds per check, (a) none, (b) race:4, (c) race:2,4, (d) race:1-4, and (b)-(d)
paired against (a). On this VM the four checks cost 0.78 thread-seconds per world-set together, 0.50 M
thread-seconds; a rented hyper-threaded thread ran the baseline 2.4 times slower (51 against 21 ms a
world, o3-curse-ensemble-1), so about 1.2 M thread-seconds: about 1.4 hours on 245 threads, 2.6 hours on
128 (est_seconds 9,000).

**For the app** (trick-2 search, PIMC over many deals with H_3): race:4 multiplies H_3's time a world by
about 3.4, race:2,4 by about 8 and race:1-4 by about 24 on this VM. Which policy is worth it is for the
rented run's paired comparisons to show.

## Learned exact leaves (build of 2026-10-19)

The owner asked for G's choice of leaves to be found in the data rather than set by hand. A learned
leaf-risk model (`nn::leafrisk`: gradient-boosted integer trees over O_3's own inputs and cheap
extras of its answer, trained on fresh network-driven trick-3 harvest rows with O_3's errors)
predicts O_3's squared error at each frontier position; `--exact-leaves learned:MODEL:t=T` solves
the leaves scored at or above `T`, a threshold chosen by Learn then Test on a separate calibration
split. On the held-out test split, at race:1-4's share of positions solved it leaves 1.0% of O_3's
squared error against race:1-4's 9.6%; in the hybrid on 32 golden decisions, at race:1-4's thread
time it solves 35% of the leaves and takes the value SD from 0.61% to 0.27%. Design, research,
numbers and the manifest (`docs/vast/o3-learnedleaf.manifest.json`; ran 2026-10-19, result below):
`docs/m3-hybrid/learned-leaves.md`.

### Result on the golden set (run `o3-learnedleaf-2`, 2026-10-19; adopted)

The adopted O_3 (f5766a) on the stored 2,000 trick-2 decisions x 320 worlds, one machine (AMD EPYC 7B12,
245 threads, $0.30/h; the run cost about $0.65). Run 1 (`o3-learnedleaf-1`) was stopped at its second check
for speed; its baseline reproduced `o3-exactleaf-1`'s byte for byte, so race:1-4 was not re-run and is read
from `o3-exactleaf-1`. Decision loss, mean / 95% upper bound:

| exact leaves | thread-ms a world | leaves solved | all | pos 1-2 | pos 1 | largest \|bias\| | verdict |
|---|---|---|---|---|---|---|---|
| none | 252 | 0% | 0.0056% / 0.0068% | | 0.0154% / 0.0201% | 0.0239% | FAIL |
| race:1-4 (G v1) | 1,692 | 56.8% | 0.0014% / 0.0018% | 0.0032% (upper) | 0.0040% / 0.0058% | 0.0041% | PASS |
| **learned, t=277338260** | **1,550** | **40.2%** | **0.0005% / 0.0008%** | **0.0009% / 0.0013%** | **0.0015% / 0.0022%** | **0.0019%** | **PASS** |
| learned, t=1057825477 | 1,083 | 20.0% | 0.0023% / 0.0030% | 0.0041% / 0.0057% | 0.0071% / 0.0101% | 0.0082% | FAIL (pos 1) |

- At slightly less thread time than race:1-4, the learned rule loses about a third as much money overall and at
  the leader, and every seat (largest upper bound 0.0031%, pos 5), score group and race group passes.
- The half-cost threshold saves only 30% of the time (leaf solves on the golden set cost more each than on the
  build VM's sample) and fails at pos 1.
- **Adopted:** `learned:<leafrisk-o3-v1>:t=277338260` is H_3's exact-leaf rule for trick-2 decisions and for
  hybrid trick-2 labels (`docs/vast/m3-hybrid-t2-hlabels.manifest.json`). Files: R2
  `runs/o3-learnedleaf-2/out/checks/`.
