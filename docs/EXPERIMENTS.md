# Experiments: every experiment we ran, its method and its result

This is the single collected record of what the project has measured, built to measure, tried and
stopped, from the first engine test (2026-10-13) to the specialist test (2026-10-22). Each entry is
a short study: the question, the method, the result in numbers, the cost, the limits of the evidence,
and where the files are. It reports what was done and what came out. It does not recommend what to
do next; where the project made a decision at the time, the decision is stated as a fact, with its
date. Numbers were read from the runs' own stored outputs in R2 (`runs/<run_id>/...`,
`rounds/<id>/...`, `inputs/sha256/<hex>`) or, where a run's files were not re-read, from the
document named under "Sources".

Companion records: `docs/RESEARCH.md` holds the outside research (published papers, other programs,
the game family) that the methods below come from; `docs/README.md` lists every document with a
one-line outcome; design and code detail stay in each experiment's own document (named under
"Sources"). A new result is added here, with a one-line summary in `docs/README.md`.

## 0. How to read the numbers

**Terms.**
- **Money chance**: each partnership's (team's) chance of winning the match money, four numbers
  summing to 1, stored as whole numbers in units of 2^-32. Every value, label and loss below is in
  percent of money chance, the decider's team's unless said otherwise.
- **World**: one complete guess of the hidden cards consistent with what the deciding seat knows,
  drawn exactly uniformly by the world generator (§A5).
- **Exact search**: the max^n solver [each seat picks the card best for its own team, all cards
  face up] to the end of the hand, with the documented tie rule (first strictly best card in
  increasing card index).
- **O_k**: a network that values a position at the start of trick k (all cards face up). **H_k**:
  exact search to the start of trick k, then O_k at every position reached there (the "frontier"
  or "leaves"). H_k plus exact solves of some leaves is written H_k + G.
- **Outcome head**: the network predicts how the hand ends (8, 14, 24 and 39 possible endings at
  tricks 5, 4, 3, 2); each ending's money is computed exactly from the equity table.
- **Golden set**: fixed decisions, each valued on 320 worlds by exact search, never trained on.
  The trick-3 set (for O_4) has 400 or 4,000 decisions; the trick-2 set (for O_3) has 400, later
  2,000 decisions (seed 1, keys 635e9aef… / 6ddc59b1…). No trick-1 golden set exists (owner
  decision 2026-10-16).
- **Decision loss**: per decision, the exact mean value of the exact best card minus the exact mean
  value of the card the tested method picks (both over the same worlds). Reported as the mean over
  decisions and as the **95% bound** (the upper end of a bootstrap over decisions, 1,000 replicates,
  seed 1). **The standard**: 95% bound at most 0.01% overall and in each seat band (seats 1-2, 3-4,
  5-8), and every team's mean value error (**bias**) at most 0.01%.
- **Seat** (`pos` in program output): the decider's place in the trick, seat 1 leads. Seat-band
  bounds rest on 104-504 decisions; single-seat bounds on 48-252 decisions and are coarse.
- **Value SD**: spread of the network's (or method's) value error per position and team, on a named
  set of positions.
- **Race class** of a position (from the outcome head's exact mask): 0 no team can win the game
  this hand; 1 one team can win the game, not the match; 2 two or more teams can win the game, none
  the match; 3 one team can win the match, nobody else the game; 4 a team can win the match and
  another the game (a race for the money).
- **Score stratum S1-S4** of a decision: the hand's start state; S4 is a team close to the money.
- **Paired comparison**: two methods on the same decisions and worlds, bootstrap of the
  difference; "interval" is its 95% range.
- **Thread-ms a world**: CPU time to value one world of one decision with all candidates.

**Noise levels measured** (useful when comparing rows below):
- Retraining the same recipe with another seed moved the trick-3 golden 95% bound (400 decisions)
  by up to about 0.0025 points (trick-4 rung 1, §B2).
- Rounds of the trick-4 loop with growing data moved the 4,000-decision bound 0.0016 → 0.0021 →
  0.0016 → 0.0024% (§B5).
- The O3G2 rounds' adoption rule uses a tolerance of 0.002 points for this reason (§C12).

**Costs** are dollars of vast.ai rental (whole rental unless "training only" is said), at the
machine's hourly rate. Rented threads ran 2.4-4.6 times slower than the 4-vCPU build VM under full
load (§A8).

---

## Part A. Foundations and measurements

### A1. Rules engine and independent reference (M0, 2026-10-13/14)
- **Question.** Are the rules implemented exactly, including the owner's house rules (RULES H)?
- **Method.** Engine (`crates/engine`) and a naive independent reference (`crates/reference`,
  written separately from RULES.md, no shared internals) driven by identical random legal actions;
  every legal-action set, trick winner, score, event stream, seat view and money result compared at
  every step (`crates/difftest`). One test per numbered rule; about 25 plain-English transcripts read
  and approved by the owner; property tests; RNG golden vectors on Windows, Linux and wasm32.
- **Result.** 10,000,000 matches (5 shards on GitHub CI, run 38480695319): zero disagreements, rare
  branches covered. After the 2026-10-14 rule changes: 100,000-match reruns, zero disagreements. Owner
  approved the transcripts 2026-10-14. Later adversarial review (2026-10-15/16, review D) found and
  fixed one gap (hand-start states accepted unreachable scores 1 and 6); 150,000 further
  differential matches (65M actions) showed zero disagreements.
- **Sources.** `M0-SPEC.md`, `TRANSCRIPTS.md`, `history/STATUS-history.md` (M0 gate).

### A2. Fixed-point money chance, exact match-equity table, arena (M1, 2026-10-13/14)
- **Method.** Money chances as u64 per team in units of 2^-32 summing exactly to 2^32;
  largest-remainder rounding. Match-equity table v0: an exact sweep over about 810,000 match states,
  with per-hand transition probabilities estimated from 1M rules-v1 self-play hands (seed 1). Arena:
  14-way duplicate blocks, a 0.000 canary.
- **Result.** Table checks pass (sums, rotation, monotonicity, Monte Carlo |z| ≤ 2.52); table
  sha256 64740dab…6808 (used by every later run). Arena canary exactly 0 on every block. Spread of a
  block, rules-v1 against random play: σ_b = 0.146 money chance (1.17 stakes), mean +0.26. rules-v1
  wins about 56% of the money against random play.
- **Limit.** The table's hand-transition model comes from rules-v1 play, not from strong play.
- **Sources.** `M1-SPEC.md`, `history/STATUS-history.md` (M1).

### A3. What an exact solve costs (M2 spike and optimised solver, 2026-10-14)
- **Question.** At which trick is exact search affordable, per world?
- **Method.** Naive max^n solver, then an optimised one (bounded transposition table, equivalent
  cards with the game's exceptions, Sturtevant shallow pruning, optional move ordering). Identity
  gate: optimised = naive on every root value, card value and chosen card, all 16 switch
  combinations, both tie orders (150,000 worlds at tricks 4 and 5, 3,000 at trick 3, 64 at trick 2),
  on an Intel Xeon build VM and an AMD EPYC 7502.
- **Result** (optimised, one thread, median / p90 per world): trick 5 0.003 ms; trick 4 0.04 / 0.11
  ms; trick 3 8.8 / 30 ms (71k nodes); trick 2 3.1 s / 15.4 s (worst 29 s, 18.5M nodes); trick 1:
  none of 8 worlds in 60 s, none of 2 in 300 s (over 1.5 billion nodes, table full at 3 GB).
  Equivalent cards gave most of the gain (3.3× fewer nodes at trick 3, 5× at trick 2); shallow
  pruning fired at about 1 node in 10,000; move ordering cost 15% and saved nothing (left off).
  82-85% of decision nodes have tied best cards; reversing the tie order never changed a root's
  money vector at tricks 3-4 (3,000 roots). Same costs within machine variation on the EPYC 7502.
- **Trick 1 on a big machine** (run `m2-trick1-1`, AMD EPYC 9654, 368 threads, parallel solver, 86
  GB table cap, $0.14): four worlds took 7, 32, 140 and 25 s (1.2-28.9 billion nodes, up to 65 GB
  of table). One world spread over 368 threads ran each thread about 7 times slower than a thread of
  the 4-thread VM.
- **Sources.** `M2-SPIKE.md`, `m2-epyc/`.

### A4. Trick-1 exact cost at scale: benches, table sizes, layouts (2026-10-15/16)
- **Question.** What does an exact trick-1 world cost on rented machines, and what table per world
  is needed?
- **Method.** `t1bench` runner chunk: a 240-s "fill" window (one world per thread, every thread
  busy) and a 120-s production-schedule window; 11 machine types in three waves; a table-size sweep
  on the same worlds; run `m3-t1-tables-1` (AMD EPYC 7542, 122 threads, $1.92/h) solving trick-1
  worlds one per thread at 12, 3 and 1 GB tables.
- **Result.** Cost per 10^12 solver nodes at fill speed: AMD Threadripper PRO 9995WX (184 threads,
  241 GB, $1.27/h) $0.28-0.31, repeat within 2%; Xeon E5-2695 v4 (69 threads, $0.41/h) $0.53;
  Xeon Gold 5220R $0.59; EPYC 9734 $0.83; EPYC 9654P $0.84; EPYC 9754 $0.94; EPYC 9554 $1.20; a Zen 5
  engineering sample (491 threads, $13.66/h) $1.50. The production schedule ran at 0.34-0.54 of the
  fill rate. Table sweep (9995WX, against 448 MB a thread): 256 MB 1.07× the nodes, 128 MB 1.21×, 64
  MB 1.88×, 32 MB 3.2×; 1-2 GB on an old Xeon changed nothing for short worlds. Leader (seat 1)
  worlds in `m3-t1-tables-1`: 0.6-79 ×10^9 nodes, median 12 ×10^9, 10 of 16 finished, the other 6
  past 95 ×10^9 at the cap, so the mean leader world is at least 47 ×10^9 nodes; seat-2 worlds mean
  12 ×10^9. Trick 2 is about 1.8 ×10^7 nodes a world, so trick 1 costs 1,000-2,500 times more per
  world. One hard world (decision 0, world 3): 79.3 ×10^9 nodes, 13,057 s on one thread.
- **Consequence recorded at the time.** A full exact trick-1 golden set (≥ 25 decisions per cell, N
  about 500 worlds) was estimated at about 2 ×10^15 nodes for the leader position alone, about
  $300-700 and two machine-weeks. On 2026-10-16 the owner decided that no trick-1 golden set is
  built.
- **Sources.** `history/STATUS-history.md` (2026-10-15 waves), `history/plan-log-2026-10.md` (Phase 4),
  `VAST-MACHINES.md` ("Measured on our runs"), `vast/measured-machines.json`.

### A5. Card tracker and world generator (M3, 2026-10-14)
- **Method.** Facts each seat knows for certain (voids, trump limits, ace facts, burns), the exact
  number of consistent worlds (u128), max-flow feasibility, exactly uniform sampling.
- **Result.** The true deal was never excluded (20,000 mixed matches, every seat at every decision;
  plus rare-branch hands); on small late positions the tracker's worlds equal the engine's rebuilt
  deals exactly; chi-square tests against uniform and against a rejection oracle pass. World counts:
  about 10^36 at the start of trick 1, 10^14 at trick 4, 10^8 at trick 5. Cost: 0.1-0.5 ms median a
  decision for tracker + count + 100 samples.
- **Sources.** `M3-SPEC.md` §3-4, `history/STATUS-history.md`.

### A6. How many worlds a decision needs (N-curves, tricks 2-4, 2026-10-14)
- **Question.** For PIMC with exact search on each world, how does the decision loss fall with the
  number of worlds N, and what N reaches 0.01%?
- **Method.** 400 decisions per trick, 320 exact worlds each; the reference is the best card on
  K = 160 held-out worlds; loss fitted as a·N^-b on N ≤ 80 with 400 random splits and bootstrap;
  held-out and extrapolation checks.
- **Result.** Shared exponent b: trick 2 0.738 [0.61, 0.94], trick 3 0.709, trick 4 0.819. N for 0.01%
  loss (95% range): trick 2 243 [119, 563], trick 3 179 [84, 456], trick 4 60 [36, 127]. Held-out
  checks: 63/63 inside the band at trick 2 (61/63 trick 3, 57/63 trick 4); extrapolation from 160 to
  320 worlds 134/135 inside at trick 2. Hardest groups at trick 2: leader (N 252), seat 6 (361), score
  stratum S4 (365), two or more top trumps held (654), 3+ trumps (465). Rollout PIMC with rules-v1
  playouts at trick 2 lost about 0.35% per decision, about 13 times exact PIMC; the plan was amended
  (A3', 2026-10-14) to use exact PIMC early.
- **Limits.** 52 decisions per seat; the reference itself has noise (K = 160).
- **Sources.** `m3-ncurves/` (`curves-t2.txt`, `-t3`, `-t4`, `score-t2.txt`, `index.html`).

### A7. Trick-1 deal counts (attempted, 2026-10-14/16)
- **Question.** The same N-curve at trick 1.
- **Method.** 32 trick-1 decisions × 80 exact worlds (sets A and B), on several rentals.
- **Result.** Not completed. Runs `m3-agree-t1-b2`, `-a7`, `-b3` were destroyed by the project's own
  hourly cleanup job (a 2-hour age rule, since removed) after saving 278 + 58 of 2,560 worlds
  ($9.30); run B earlier lost about $18 to a hung upload. Run `m3-agree-t1-ab1` (old binary, rules-bot
  robber burn; 5 decisions × 80 worlds) showed one world disagreeing with another by 0.5-1.5% on the
  best-minus-second difference, and the two halves of 40 worlds disagreeing by 0.04-0.26% (4-26 times
  0.01%). The golden N for a trick-1 leader decision was estimated at 250-2,000 worlds (not measured).
  Saved exact trick-1 worlds that remain: 32 decisions in the decisions file, 8 with worlds (31
  worlds, searched burn) plus 5 decisions × 355 worlds valued with the older rules-bot burn.
- **Sources.** `history/STATUS-history.md` (2026-10-15), `history/plan-log-2026-10.md` (Phase 4).

### A8. Rented machines: speed and price
- **Facts measured.** Containers get about 96% of advertised threads; some hosts give far fewer (an
  AMD EPYC 9B45 listed with 384 threads gave 46 usable threads on every offer tried: twice on
  2026-10-17, three offers on 2026-10-22; the launcher stops such machines). Rented threads under full load ran
  2.4-4.6 times slower than the build VM (H_3 on trick-2 decisions: 46.9 ms a world on a loaded
  224-thread EPYC 7663 against 16.9 ms on the idle VM; exact trick-2 label 11.3 thread-s on an EPYC
  7B12 against 3.65 s on the VM). A speed model from SPEC benchmarks predicted three new machines
  within 11%, the Xeon Gold 5220R 40% too pessimistically and the Zen 5 sample about 20% too
  optimistically. Pricing is set by the graphics cards; CPU-heavy work is cheapest on big CPUs with
  cheap cards.
- **Run-operation failures recorded:** hourly cleanup killing healthy runs (2026-10-15); a hung
  upload (run B); slow image pulls killing machines (fixed with a pull-aware timeout); a broken
  NVIDIA runtime rented twice (now remembered for 24 h); the vast balance reaching zero stopped every
  machine at once (2026-10-18).
- **Every machine's measured cost per unit of work** (exact trick-2/3 solves, labels, H_3, teachers,
  trick-1 benches): `VAST-MACHINES.md`, first section.
- **Sources.** `VAST-MACHINES.md`, `vast/measured-*.json`, `history/STATUS-history.md`,
  `history/plan-log-2026-10.md` (2026-10-22 entries).

### A9. Exact robber burn and the adversarial reviews (2026-10-15/16)
- **Change.** When another seat robs in a sampled world, its burn is searched inside the exact solve
  (its team picks the best burn) instead of being chosen by rules-v1. Another seat robs in about 74%
  of sampled trick-1 worlds (count 32) and 65% (count 128).
- **Reviews.** Five independent reviews (solver, equity leaf and scoring, PIMC and measurement,
  engine and reference, runner and cost safety). Found and fixed, each with a failing test first:
  equivalent-card merging ignored the tie order (wrong values for other teams, sometimes the mover's;
  fix costs about 3% more nodes); a pending rob's turned card treated as dead; decision sets drawing
  the play right after a seat's own burn; the rollout leaf reading the match winner from money (wrong
  with stake 0); runner and launcher safety issues. Records now carry `VALUES_VERSION` = 2; worlds
  made before 2026-10-15 cannot be extended by later binaries.
- **Sources.** `history/STATUS-history.md` (2026-10-15/16), `reviews/`.

---

## Part B. The trick-4 network O_4 (values the start of trick 4; checked on trick-3 decisions)

### B1. First networks and the outcome head (rung 1 and 1b, 2026-10-16, build VM)
- **Method.** Labels: exact values of trick-4 positions from random play and rules-v1 self-play.
  MLP 512-256-64 with four logits (rung 1), then with the outcome head (rung 1b). Checked on 40
  trick-3 golden decisions × 64 worlds.
- **Result.** Four-logit O_4 (900k labels): value SD 3.0%, bias −0.16%, decision loss mean 0.022%
  (bound 0.043%). Outcome head, same data: value SD 2.08%, outcome accuracy 90.1%, loss mean 0.019%
  (bound 0.040%). O_5 with the outcome head: 99.96% outcome accuracy, mean |error| 0.016%; H_5 on 40
  trick-4 decisions: loss 0. An encoder without the equity table's successor values was worse even at
  trick 5 (SD 2.7% against 1.9%). Network evaluation then cost 64-75 µs.
- **Sources.** `m3-hybrid/rung1.md`.

### B2. Rung-1 scaling sweep and harvested positions (ten trainings, 2026-10-16)
- **Question.** What moves H_4's decision loss: more ordinary labels, network size, harvested
  positions [frontier positions the hybrid's own search reaches, labelled exactly], harvest weighted
  to leaders?
- **Method.** 1024-512-128 (one run 512-256-64), outcome head; trick-3 golden set 400 decisions × 320
  worlds; one rental each (about $0.10-0.45 each, nine rentals about $2.0).
- **Result** (95% bounds, %; all / seats 1-2 / 3-4 / 5-8 / largest bias):

| run | labels (random + self-play + harvested) | all | 1-2 | 3-4 | 5-8 | bias |
|---|---|---|---|---|---|---|
| 4M (`m3-hybrid-rung1b-1`) | 3.2M + 0.8M | 0.0108 | 0.0309 | 0.0174 | 0.0016 | 0.0130 |
| 4M, training seed 2 | same | 0.0097 | 0.0247 | 0.0177 | 0.0014 | 0.0186 |
| GitHub CPU run | same | 0.0083 | 0.0113 | 0.0145 | 0.0073 | 0.0168 |
| 8M | 6.4M + 1.6M | 0.0086 | 0.0281 | 0.0113 | 0.0008 | 0.0089 |
| 16M | 12.8M + 3.2M | 0.0076 | 0.0211 | 0.0115 | 0.0007 | 0.0091 |
| 16M, 512-256-64 | same | 0.0084 | 0.0224 | 0.0161 | 0.0001 | 0.0077 |
| harvest-2 | 6.4M + 1.6M + 4M (cap 16, seats 40/40/20%) | 0.0090 | 0.0210 | 0.0059 | 0.0098 | 0.0094 |
| lead-1 | 12.8M + 3.2M + 8M (cap 64, seats 60/25/15%) | 0.0037 | 0.0069 | 0.0109 | 0.0002 | 0.0075 |
| lead-2 | 12.8M + 3.2M + 12M (cap 64, 45/40/15%) | 0.0033 | 0.0046 | 0.0103 | 0.0002 | 0.0079 |
| lead-3c | 16M + 4M + 20M (cap 64, 30/55/15%) | 0.0057 | 0.0097 | 0.0159 | 0.0007 | 0.0057 |

- **Observed.** Three trainings on the same 4M labels spread 0.0083-0.0108% overall. Leader-weighted
  harvest (lead-1) halved the overall bound against 16M ordinary labels; lead-1 changed cap, amount
  and seat shares together, so the parts are not separated. A seat-1 decision reaches about 500
  trick-4 positions on average, a seat-4 decision about 40. In lead-2 two seat-4 decisions (143 and
  335) carried 91% of seat 4's loss, and seven of eight networks missed each at exactly the same cost.
  Lead-2's validation loss rose for five epochs after its best epoch (overfitting).
- **Sources.** `m3-hybrid/rung1.md` ("Results, 2026-10-16"); R2 `runs/<run_id>/out/rung/`.

### B3. Why seats 3-4 failed: the race for the money (diagnosis, 2026-10-16)
- **Method.** `search hybrid-check --explain` on every lossy decision of lead-2 and lead-3c, and on
  golden decisions 0-99; error by race class on golden frontier positions (18.6M values) and on a
  fresh 300k harvest shard.
- **Result.** Race class 4 was 7% of frontier positions and carried 75% of lead-3c's decision loss
  (86% of the seats 3-4 loss). Value SD by class on the golden frontier: class 0 0.33%, 1 0.28%, 2
  1.65%, 3 0.62%, 4 2.46% (fresh shard: class 4 2.74%, lead-2 3.04%). Class-4 positions were present
  in the training data (about 7% of rows, about 1.4M of 20M harvested). Example (decision 143): the
  network over-rated the team that could win the match (team 1 bias +0.62%), i.e. under-rated the
  other teams' blocking. Decision 335 was decided by the solver's tie rule in one world (tie-decided
  positions: 0.015% of the frontier, network error SD 59% there). **Ceiling with exact values at
  chosen classes** (lead-3c, 95% bounds all / 1-2 / 3-4 / 5-8): none 0.0057 / 0.0097 / 0.0159 /
  0.0007; class 4 exact 0.0016 / 0.0041 / 0.0023 / 0.0007; classes 2 and 4 0.0015 / 0.0041 / 0.0015 /
  0.0007; classes 1-4 and tie-decided 0.0005 / 0.0002 / 0.0013 / 0.0007. With 104 decisions in a band,
  one miss of about 0.35% fails the band alone.
- **Sources.** `m3-hybrid/rung1.md` ("Why seats 3-4 fail").

### B4. Race harvest and network shape (lead-4, lead-5c/6a/6b, 2026-10-16)
- **Method.** lead-4: lead-3c's mix with half the harvest replaced by race rows (16M random s151,
  4M rules s152, 10M harvest s154 cap 64, 10M race-class-2/4 rows s155): class-4 rows from about 7.5%
  to about 19% of 40M. Then three shapes on the same 40M: 1024-1024-512-128, 2048-1024-256,
  2048-2048-1024-256.
- **Result.** lead-4 (1024-512-128; read from its R2 bundle for this record): 95% bound all 0.0075%,
  seats 1-2 0.0231%, 3-4 0.0106%, 5-8 0.0002%, seat 1 0.0395%; integer value SD 0.92% overall, by
  race class 0.39 / 0.50 / 1.33 / 0.73 / 2.58% (lead-3c's class 4: 2.74%). Shapes:

| shape | val loss | class-4 SD (int / float) | all | 1-2 | 3-4 | 5-8 | bias |
|---|---|---|---|---|---|---|---|
| 1024-1024-512-128 (lead-5c) | 1.19817 | 2.33 / 2.99% | 0.0072 | 0.0210 | 0.0109 | 0.0001 | 0.0083 |
| 2048-1024-256 (lead-6a) | 1.20811 | 3.60 / 3.34% | 0.0074 | 0.0222 | 0.0104 | 0.0002 | 0.0097 |
| 2048-2048-1024-256 (lead-6b) | 1.19772 | 3.29 / 3.02% | 0.0076 | 0.0239 | 0.0102 | 0.0003 | 0.0060 |

- **Observed.** 2.5× the class-4 data moved class-4 SD by about 6-10%; width did not lower the float
  class-4 SD (about 3.0% for all three). 1024-1024-512-128 was chosen as the loop's starting shape.
- **Sources.** `m3-hybrid/rung1.md` (shape comparison); R2 `runs/m3-hybrid-harvest-lead-4/out/rung/`.

### B5. The trick-4 audit loop (mechanism tests and the overnight loop, 2026-10-16/17)
- **Question.** Can weak groups of positions be found automatically and trained away until none
  over the standard remains ("flat")?
- **Method** (`m3-hybrid/audit-loop.md`): each round trains, then an auditor searches conjunctions of
  up to 6 encoder-feature conditions on a fresh exactly-labelled audit shard (weight ≥ γ; confirmed on
  a held-out half with Benjamini-Hochberg false-discovery control at 5%; thresholds α_bias 0.25%, α_sd
  1.0%), measures confirmed groups at the decision level (golden slices; "mini golden sets" of 100
  decisions by rejection sampling), harvests positions of failing groups, retrains from scratch, with
  a capacity ladder. Research basis: multiaccuracy (Hébert-Johnson et al. 2018; Kim et al. 2019),
  slice discovery (SliceFinder, DivExplorer, Domino), boosting/targeted sampling.
- **Runs.** Mechanism tests `m3-hybrid-t4-audit-mech-1` (EPYC 9654, 384 threads, $0.60) and `-2`
  (EPYC 7B12, 122 threads, $1.25); loop `m3-hybrid-t4-loop-3` (AMD EPYC 7B13, 122 threads, 504 GB,
  $0.73/h, about $4.2; two earlier launches about $1.4). Golden set: 4,000 trick-3 decisions × 320
  worlds; 10M-row audit shards at γ = 0.0001; at most 60 mini sets a round.
- **Result** (95% bound all / 1-2 / 3-4 / 5-8 / largest bias; groups confirmed; failing at decision
  level):

| round | shape | all | 1-2 | 3-4 | 5-8 | bias | confirmed | failing |
|---|---|---|---|---|---|---|---|---|
| 1 | 1024-1024-512-128 (b28750e3) | 0.0016 | 0.0026 | 0.0023 | 0.0012 | 0.0025 | 1,551 | 8 (1,473 not measured) |
| 2 | same | 0.0021 | 0.0055 | 0.0026 | 0.0008 | 0.0030 | 3,787 | 14 |
| 3 | 2048-2048-1024-256 (6d72e18c) | 0.0016 | 0.0040 | 0.0020 | 0.0009 | 0.0035 | 2,724 | 13 |
| 4 | 4096-2048-1024-256 (stopped) | 0.0024 | 0.0059 | 0.0023 | 0.0012 | 0.0045 | 3,026 | 13 |

- **Observed.** On 4,000 decisions every seat band was under 0.006%; the earlier 400-decision failures
  of seats 1-2 did not recur. The auditor confirmed thousands of groups because its SD threshold (1%)
  was close to the network's overall SD (0.8-1.0%); 92.5% of positions lay inside some confirmed
  group; 60 mini sets a round measured a few percent of them, so "flat" could not be reached as
  defined. Position-level SD predicted decision-level failure with AUC 0.77-0.79. Of the groups
  still failing, 62% were race-for-the-money conditions. Retraining from scratch made rounds noisy.
- **Decision recorded.** O_4 = round 1 (b28750e3…) adopted (owner, 2026-10-17); no more trick-4 runs
  (an exact trick-3 solve, about 6 ms a deal, is cheaper than H_4, about 18 ms).
- **Sources.** `m3-hybrid/audit-loop.md` §8-9.

---

## Part C. The trick-3 network O_3 (values the start of trick 3; checked on trick-2 decisions)

All O_3 golden numbers below are on the trick-2 golden set (2,000 decisions × 320 exact worlds,
seed 1) unless "400 decisions" is said. Reference figures for the adopted O_3 f5766a alone: mean
0.0056%, 95% bound 0.0068%; seats 1-2 0.0122%; seat 1 0.0154% / 0.0201%; own-team bias −0.0085%
(seats 1-2 −0.0239%); value SD 1.85%; 85.7% agreement. It fails the standard on seats 1-2 only.
The same figures were reproduced exactly by the incumbent's check in at least five later runs (C7, C10, C11, C12, C13).

### C1. First O_3 rounds (2026-10-16)
- **Method.** 1024-512-128 on 12.8M random + 3.2M rules-v1 labels (round 1); plus 8M harvested
  positions driven by round 1's network, cap 32, seats 40/30/30% (round 2). 400-decision golden set
  generated in round 1 (1,057 s on 184 threads), reused in round 2. AMD Threadripper PRO 7995WX (184
  threads), $0.90 and $0.79.
- **Result** (95% bounds all / 1-2 / 3-4 / 5-8 / largest bias): round 1 0.0105 / 0.0138 / 0.0091 /
  0.0133 / 0.0108; round 2 0.0111 / 0.0156 / 0.0171 / 0.0109 / 0.0092. Round 2's validation loss was
  better (1.4971 vs 1.5131; outcome accuracy 87.2% vs 86.4%) but its check was not. O_3's outcome
  accuracy (87%) against O_4's (98%) indicates trick 3 is a harder function. Hybrid speed at trick 2:
  12.7 ms a world against 1,289 ms exact (about 100×). A round 3 (2048-1024-256, 40M labels) failed
  on a chunk error and was not repeated as such.
- **Sources.** `m3-hybrid/rung2.md` ("Results, 2026-10-16").

### C2. The trick-3 audit loop, scratch mode (overnight 2026-10-16/17)
- **Method.** As B5, at trick 3; golden set 2,000 trick-2 decisions × 320 worlds generated in the run
  (about 3 s an exact world) and stored (635e9aef / 6ddc59b1); 10M-row audit shards. Run
  `m3-hybrid-t3-loop-1`: AMD EPYC 7663 (224 threads, 2× A100 40 GB, $1.60/h), about $9.9.
- **Result** (95% bound all / 1-2 / 3-4 / 5-8 / own-team bias; confirmed; failing):

| round | shape | all | 1-2 | 3-4 | 5-8 | own bias | confirmed | failing |
|---|---|---|---|---|---|---|---|---|
| 1 | 2048-1024-256 (b34e12b5) | 0.0088 | 0.0153 | 0.0071 | 0.0088 | −0.0125 | 628 | 56 |
| 2 | 2048-1024-256 (f5766a) | 0.0068 | 0.0122 | 0.0080 | 0.0052 | −0.0085 | 630 | 46 |
| 3 | 2048-2048-1024-256 (a099061b) | 0.0074 | 0.0164 | 0.0057 | 0.0056 | −0.0096 | 629 | 47 |
| 4 | 2048-2048-1024-256 (stopped) | 0.0076 | 0.0153 | 0.0060 | 0.0061 | −0.0142 | 629 | 47 |

- **Observed.** Seats 1-2 stayed at 0.012-0.016% at both sizes; the leader's own team was
  under-valued by about 0.02-0.03% in that band. Failing groups were mostly ordinary groups with small
  slices (median 89 decisions against 299 for passing groups) around a mean of 0.006%.
- **Decision recorded.** O_3 = round 2, f5766aa98a1d… (2048-1024-256, trained on 40M labels: random
  25.6M s321, rules 6.4M s322, harvest 8M s313, plus round 1's targeted shard) adopted 2026-10-17.
- **Sources.** `m3-hybrid/audit-loop.md` §8; `m3-hybrid/rung2.md` ("Adopted O_3").

### C3. The trick-3 audit loop, additive mode (run `m3-hybrid-t3-additive-2`, 2026-10-18/19)
- **Question.** Does continuing f5766a on old plus new targeted data (warm start with replay) improve
  it, where retraining from scratch was noisy?
- **Method.** Round 1 = f5766a unchanged; round 2 continues it for 5 epochs on 48M rows (the 40M
  base plus an 8M weak-spot shard); 20M-row audit shards; a round is adopted unless it loses a pass
  the best had or is worse than the noise tolerance (0.002 points). Research basis: warm start with
  replay; Ash & Adams 2020 (warm-start generalisation gap).
- **Result** (read from R2 for this record). Round 1 (f5766a): golden 0.0056% / 0.0068%, 637 groups
  confirmed, 46 over 0.01% at the decision level, 489 not measured. Round 2 (a97ac177…): integer
  validation SD 1.83% (f5766a 1.85%), golden mean 0.0059%, bound 0.0071%; own-team bias −0.0121%
  (fails the 0.01% limit; f5766a −0.0085%), so **not adopted**; 53 groups over 0.01%. The auditor's
  worst groups were score states (a team on 35, 36 or 40 points; SD 4.4-5.2%). No larger shape was
  left on the ladder. The run was stopped by the owner on 2026-10-19 ("the additive rounds made O_3
  no better; stop and build an 80M base instead"). An earlier attempt, `m3-hybrid-t3-additive-1`,
  was stopped for low vast credit.
- **Sources.** R2 `runs/m3-hybrid-t3-additive-2/partial/rung/000000/out/hybrid/audit/round-{1,2}.txt`,
  `STOP`.

### C4. O_3 at an 80M base (run `o3-base80-1`, 2026-10-19)
- **Question.** Does doubling the training data (40M → 80M) improve O_3's decisions?
- **Method.** 2048-1024-256 from scratch (seed 1, batch 1024, lr 0.001, patience 4), outcome head, on
  the stored 40M base plus 40M new rows (random 25.6M s331, rules 6.4M s332, harvest 8M s323 driven by
  O_3 round 1 `b89b26d6`, cap 32, seats 0.4/0.3/0.3). Validation 100k (s401). AMD Threadripper PRO
  9995WX (192 threads, RTX 5090 D, $0.64/h): fetch 12.5 min, new labels 30 min, training 48 min (best
  epoch 19 of 24), 128.8 min in all, $1.37. Network 7eeb1f88…, stored.
- **Result.**

| | f5766a (40M) | 80M (7eeb1f88) |
|---|---|---|
| integer value SD, validation rows | 1.85% | 1.82% (float 1.84%; outcome accuracy 89.9%) |
| value SD on the golden frontier | 1.850% | 1.756% |
| class-4 SD, validation | 3.6-4.0% | 3.70% |
| decision loss mean / 95% bound | 0.0056 / 0.0068% | 0.0064 / 0.0080% |
| seats 1-2 bound | 0.0122% | 0.0149% |
| seat 1 mean / bound | 0.0154 / 0.0201% | 0.0204 / 0.0264% |
| own-team bias | −0.0085% | −0.0080% |
| verdict | fails (seats 1-2) | fails (seats 1-2 and 3-4: 0.0105%) |

- **Observed.** Value error fell slightly; decision loss did not fall (differences within the
  measured retraining noise). This result was used by the owner on 2026-10-19 to motivate fix E
  (sibling ranking, C9) and the optimizer's-curse test (C7).
- **Sources.** R2 `runs/o3-base80-1/out/rung/` (`hybrid/hc-t2-o3.txt`, `hybrid/net/report.json`).

### C5. O_3 base-size curve, 50% vs 100% of the 40M base (runs `o3-basecurve-50-1`, `-100-1`)
- **Question.** Train from scratch on 50% and 100% of the stored 40M base, same settings; compare
  validation SD and golden loss.
- **Result.** **Not run.** Both runs were cancelled by their workflow 37 s after start (2026-10-19
  14:40 UTC), before any training; the owner's diagnosis plan of that afternoon skipped the 40M size
  tests and folded the size question into the 80M run (C4). No result exists.
- **Sources.** R2 `runs/o3-basecurve-{50,100}-1/DONE`, `STOP`; `history/plan-log-2026-10.md` ("Diagnosis
  fixes A and B").

### C6. Wider and deeper O_3 (manifests `o3-size-wide`, `o3-size-deep`)
- **Result.** **Never launched** (no run in R2). Network size was tested elsewhere: 2048-2048-1024-256
  in the scratch loop (C2: rounds 3-4 no better than round 2) and 4096-2048-1024-256 from scratch on
  56M rows in O3G2 round 2 (C12: worse on seats 1-2 and bias).

### C7. The optimizer's-curse test (run `o3-curse-ensemble-1`, 2026-10-19)
- **Question.** H_3 under-values the decider's own team at the decision level (seat 1 −0.033%, seat
  2 −0.021%, seat 3 −0.032%, seat 4 −0.006%, seats 5-8 about 0) although O_3's position-level bias is
  about 0 (leader's team −0.003%). Can search over noisy but unbiased leaf values alone produce this
  [the optimizer's curse: picking the best of several noisy values favours the one whose noise
  flatters it]?
- **Method.** "Truth" = H_3 with O_3's own noise-free values at the trick-3 leaves; "player" = the
  same search with seeded, exactly zero-mean noise added to each leaf's 4-team vector (transfers
  between pairs of teams, so the vector sums to 2^32 and stays non-negative), with SD per race class
  set to the auditor's measured O_3 error SD (0.98 / 0.92 / 2.01 / 1.52 / 4.05% for classes 0-4),
  calibrated. The noise is independent between leaves and has no heavy tail; O_3's real errors are
  heavy-tailed (99.9th percentile 23%) and correlated between neighbouring positions. Same golden
  decisions and worlds; AMD EPYC 7B12 (256 threads, $0.30/h), 263 s wall; run $0.19.
- **Result.** Realised noise SD 1.46% overall (classes 3-4 shrunk to 1.24% and 2.96% because teams
  with little money cannot take full transfers).

| measure | noise-only player | real O_3 against exact |
|---|---|---|
| decision loss mean / 95% bound | 0.0157 / 0.0182% | 0.0056 / 0.0068% |
| agreement | 72.3% | 85.7% |
| own-team bias, all | −0.33% | −0.0085% |
| own-team bias seats 1-2 / 3-4 / 5-8 | −0.57 / −0.03 / −0.35% | −0.024 / −0.017 / +0.004% |
| own-team bias seat 1 / 2 / 3 / 4 / 8 | −0.69 / −0.41 / −0.15 / +0.09 / +0.00% | −0.024 / −0.024 / −0.030 / −0.003 / +0.004% |
| next team's bias, all | +0.29% | +0.0054% |
| worst race group (loss mean) | race 4: 0.0302% | race 4: 0.0126% |

- **Observed.** Zero-mean independent noise alone produced a negative own-team bias, largest at seat
  1 and at seat 5 (each leads a sub-sequence of four opponents), near zero at seats 4 and 8, and a
  positive bias for the next team, with the same sign pattern as the real H_3 at about 20-40 times
  the size. The real network's loss is about a third of the noise player's. The noise's size per
  race class matches O_3's measured SD but its correlation and tails do not.
- **Sources.** R2 `runs/o3-curse-ensemble-1/partial/checks/000000/out/hybrid/checks/curse.txt`,
  `single.txt`; design `m3-hybrid/curse-ensemble.md`.

### C8. Ensemble of four O_3 networks (runs `o3-curse-ensemble-1` step 3, `o3-ensemble4-1`)
- **Question.** Does the integer mean of four trained O_3s (f5766a, a099061b, bfd1bec2, a97ac177)
  lower H_3's decision loss?
- **Method.** `--net A,B,C,D`: each member values the leaf, u128 sum, largest-remainder rounding
  (byte-identical; one member equals that network). Paired against f5766a alone.
- **Result.** **Not measured.** On the rented AMD EPYC 7B12 the ensemble check ran at about 10
  worlds a second, about 100 times slower per world than one network: the four members' weights
  (about 70 MB) do not fit the chip's 16 MB last-level cache per four cores, while one network
  (about 9 MB) does. A batching fix cut passes over the weights 15× on a 16 × 32 test (values
  byte-identical); the rerun (`o3-ensemble4-1`, binary 20e8680) was still at about 10 worlds a second
  and was stopped after 14 min ($0.07). Both runs stored the single-network check (identical to
  f5766a's record). No ensemble number exists.
- **Sources.** R2 `runs/o3-curse-ensemble-1/STOP`, `runs/o3-ensemble4-1/STOP`;
  `m3-hybrid/curse-ensemble.md` ("Speed of an ensemble").

### C9. Sibling groups and a pairwise ranking loss (fix E, 2026-10-19)
- **Question.** A decision depends on differences between the positions its cards lead to. Does a
  pairwise ranking term (RankNet, Burges et al. 2005; decision-focused learning, Elmachtoub & Grigas
  2022) over "sibling" positions [the frontier positions a choice point's legal cards lead to along
  their principal lines] improve O_3's decisions?
- **Method.** Sibling shards (`search labels --source siblings`), trainer `--pair-weight W
  --pair-scale 100`, optional `--tie-aware`. Rented A/B designed (6M sibling rows + 40M base, A
  cross-entropy only, B W = 1), about $2-5.
- **Result.** Only the build-VM A/B ran: 30k random + 12k sibling rows, 256-128-64 for the smoke and
  a small network for the A/B, judged on 6,000 held-out sibling rows. A (cross-entropy): validation
  loss 2.280, value SD 5.02%, gap-weighted misordered pairs 15.6%, mean regret of the best-looking
  sibling 0.278%. B (W = 1): 2.432, 5.83%, 18.3%, 0.310%. B's training pair loss fell to 0.14 while
  held-out ordering got worse. Measured facts on sibling groups: 10.7 rows a decision, mean group size
  2.96, 28% of pairs have unequal true values; no two possible outcomes shared a money vector in 44,000
  trick-3 rows (so `--tie-aware` changes nothing at trick 3). **The rented A/B
  (`o3-sibrank-a/-b`) was never launched**; no golden-set result exists.
- **Sources.** `m3-hybrid/sibrank.md`.

### C10. Exact leaves by race class, G v1 (run `o3-exactleaf-1`, 2026-10-19)
- **Question.** If the trick-3 leaves of chosen race classes are solved exactly inside H_3, how much
  loss goes and at what time cost?
- **Method.** `--exact-leaves race:4`, `race:2,4`, `race:1-4`; a second per-thread solver table of 64
  MB. AMD EPYC 7B12 (256 threads, $0.30/h), 121.5 min, $0.60.
- **Result** (f5766a; distinct leaves a world 413: class 0 178, 1 115, 2 58, 3 24, 4 38):

| exact leaves | leaves solved | thread-ms a world | mean / 95% bound | seat 1 mean / bound | largest bias | standard |
|---|---|---|---|---|---|---|
| none | 0% | 51 (this run) | 0.0056 / 0.0068% | 0.0154 / 0.0201% | 0.0085% | fails (1-2) |
| race:4 | 9.2% | 232 | 0.0037 / 0.0047% | 0.0119 / 0.0161% | 0.0097% | passes |
| race:2,4 | 23.3% | 668 | 0.0031 / 0.0040% | 0.0085 / 0.0118% | 0.0068% | passes |
| race:1-4 | 56.8% | 1,692 | 0.0014 / 0.0018% | 0.0040 / 0.0058% | 0.0041% | passes |

- **Observed.** Each class added lowered the loss and the own-team bias; race:4 alone passed the
  standard (seat 1, not part of the standard, stayed above 0.01%). An exact leaf solve cost 4-6 ms.
- **Sources.** R2 `runs/o3-exactleaf-1/partial/checks/000000/out/hybrid/checks/`; `m3-hybrid/rung2.md`
  ("Exact leaves (G)").

### C11. Learned exact leaves, G v2 (build and runs `o3-learnedleaf-1/-2`, 2026-10-19)
- **Question.** Can the choice of leaves to solve exactly be learned from data, with a guarantee?
- **Method.** A predictor of O_3's squared error at a position ("learning to defer": Geifman &
  El-Yaniv 2017; Mozannar & Sontag 2020): gradient-boosted integer regression trees (Friedman 2001;
  150 trees, depth 6) over O_3's 909 inputs plus 13 extras from O_3's own answer (race class, its
  shares, top share, Gini impurity, distance to the nearest and farthest possible ending, number of
  possible endings). Score = predicted squared error (value of computation, Russell & Wefald 1991).
  Threshold by Learn then Test (Angelopoulos et al. 2021) with a CLT bound; conformal risk control
  and empirical Bernstein reported. Data: fresh harvest shards driven by f5766a: train 400k (15,564
  decisions), calibration 100k (3,559), test 100k (3,592). Model leafrisk-o3-v1, 20c41678…
- **Result, position level (test split).** O_3's squared error is concentrated: the worst 1% of
  positions hold 66% of it, the worst 10% hold 96%; race class 4 holds 57%, class 0 12%. Largest split
  gains: distance to the nearest possible ending 42%, number of possible endings 12.5%, distance to the
  farthest ending 5.4%. Share of squared error left after solving a share of positions:

| solved | learned | best possible order | race class alone | outcome entropy alone |
|---|---|---|---|---|
| 5% | 25.3% | 8.8% | 73.1% | 77.1% |
| 10% | 16.6% | 2.9% | 53.1% | 61.2% |
| 20% | 8.9% | 0.3% | 34.0% | 36.5% |
| 50% | 1.4% | 0.0% | 12.0% | 0.6% |

  At race:1-4's share (61.8%) the learned order leaves 1.0% of the error against race:1-4's 9.6%.
  Every Learn-then-Test threshold held on the test split; the finite-sample bounds certified no useful
  target at this sample size (they need hundreds of thousands of decisions).
- **Result, golden set** (`o3-learnedleaf-2`, AMD EPYC 7B12, 256 threads, $0.30/h, $1.07; `-1` was
  stopped on a 128-thread machine, $0.70; race:1-4 read from C10):

| exact leaves | thread-ms a world | leaves solved | mean / 95% bound | seats 1-2 bound | seat 1 mean / bound | largest bias | standard |
|---|---|---|---|---|---|---|---|
| none | 252 | 0% | 0.0056 / 0.0068% | 0.0122% | 0.0154 / 0.0201% | 0.0239% | fails |
| race:1-4 | 1,692 | 56.8% | 0.0014 / 0.0018% | 0.0032% | 0.0040 / 0.0058% | 0.0041% | passes |
| learned t=277338260 | 1,550 | 40.2% | 0.0005 / 0.0008% | 0.0013% | 0.0015 / 0.0022% | 0.0019% | passes |
| learned t=1057825477 | 1,083 | 20.0% | 0.0023 / 0.0030% | 0.0057% | 0.0071 / 0.0101% | 0.0082% | fails (seat 1) |

  (Times as each run reported them: this run gives 252 thread-ms for "none", C10's run 51 ms for
  the same check; the two figures were not reconciled.) Every seat, score group and race group passed at t=277338260; the
  largest seat bound was 0.0031% (seat 5).
- **Decision recorded.** G v2 at t=277338260 adopted for H_3 (owner, 2026-10-19).
- **Sources.** `m3-hybrid/learned-leaves.md`; `data/leafrisk/leafrisk-o3-v1*.json`; R2
  `runs/o3-learnedleaf-2/out/checks/`.

### C12. O3G2 rounds: G2 chooses O_3's new training positions (rounds o3r-1 and o3r-2, 2026-10-21)
- **Question.** O_3 alone is what values the trick-3 positions in O_2's training labels and in the
  planned bronze set, so the target was O_3 alone at ≤ 0.0025% (95% bound) overall and at every seat.
  Does adding positions chosen by G2 [the learned error predictor of C11] bring O_3 alone closer to
  O_3 + G2?
- **Method.** Each round: a pilot of 2,000 decisions scores every trick-3 position the incumbent's
  H_3 reaches with G2; 20 equal-count risk bands; target share of new rows per band
  (1 − f)·frequency·risk^α/sum + f·frequency with α = 1, f = 0.2; positions kept by a seeded coin per
  band before labelling, at most 1,000 a decision, from trick-2 decisions. Research basis: Learning Loss
  for Active Learning (Yoo & Kweon 2019), prioritised replay (Schaul et al. 2016), importance sampling
  by loss (Katharopoulos & Fleuret 2018), KataGo policy-surprise weighting, defensive importance
  sampling (Hesterberg 1995). Three nested candidates a round (2M, 4M, 8M new rows); additive
  candidates continue the incumbent 8 epochs at lr 0.0003 on base + parts; a candidate is adopted
  unless it loses a pass the incumbent had or is worse beyond 0.002 points; label check of each
  network on the stored 50k exact trick-2 validation shard (s601) as O_2's labels would be made
  (exact through trick 2, then O_3); next G2 built on the winner.
- **Runs.** Round 1 first attempt `o3r-1-1` (AMD EPYC 7B12, 256 threads, RTX 3060, $0.36/h): the
  instance died, $0.45. `o3r-1-3` (AMD EPYC 9B14, 184 usable threads, RTX 5090, $1.34/h): 51 min, $1.14.
  `o3r-2-1` (same machine type): 95 min, $2.10. Round 2's base was the 40M base plus round 1's 8M
  stored rows.
- **Mix made.** New rows in the riskiest 2 of 20 bands: round 1 57.8% (aimed 63.3%), round 2 58.7%
  (aimed 62.1%). Race class 4 was 34% of round 1's part 1 (it is 9% of the distinct leaves H_3 reaches on the golden set, C10).
- **Result** (golden 95% bound; seat 1 bound; own-team bias; label check against exact trick-2 labels:
  RMS error and largest team bias):

| round | candidate | new rows | golden bound | seats 1-2 | seat 1 | own bias | label RMS / largest bias | adopted |
|---|---|---|---|---|---|---|---|---|
| — | incumbent f5766a | — | 0.0068% | 0.0122% | 0.0201% | −0.0085% | 2.113 / 0.0132% | — |
| 1 | c1 additive | 2M | 0.0071% | 0.0133% | 0.0227% | −0.0121% | 2.122 / 0.0141% | no (bias) |
| 1 | c2 additive | 4M | 0.0080% | 0.0169% | 0.0288% | −0.0098% | 2.093 / 0.0140% | no (seats 1-2) |
| 1 | c3 additive | 8M | 0.0073% | 0.0141% | 0.0229% | −0.0111% | 2.104 / 0.0159% | no (bias) |
| 2 | c1 additive | 2M | 0.0069% | 0.0135% | 0.0233% | −0.0100% | 2.112 / 0.0142% | no (bias at the line) |
| 2 | c2 additive | 4M | 0.0070% | 0.0135% | 0.0223% | −0.0085% | 2.104 / 0.0169% | **yes** (1057e9e9) |
| 2 | c3 scratch 4096-2048-1024-256 | 8M | 0.0079% | 0.0182% | 0.0325% | −0.0155% | 2.258 / 0.0082% | no |

  Paired against the incumbent (mean loss B − A, 95% interval): round 1 c1 +0.0003 (−0.0006..+0.0011),
  c2 +0.0010 (−0.0001..+0.0023); round 2 c1 +0.0001 (−0.0010..+0.0011), c2 +0.0001
  (−0.0010..+0.0012), c3 +0.0006 (−0.0008..+0.0023). Own-team bias at seats 1-2 was more negative than
  the incumbent's for every candidate (e.g. round 1 c1 −0.0300% vs −0.0239%, interval of the difference
  −0.0109..−0.0015). Every additive candidate's validation loss was lowest after its first epoch (best
  epoch 0; validation SD 1.93-1.95% at that point, integer SD on 20k rows 1.75-1.85%). The scratch
  4096-wide network reached validation SD 2.06% (best epoch 9) and cost 77.5 ms a world in H_3
  against 23.9 ms. Round 2's adopted c2 (1057e9e9…) is within noise of f5766a on every figure; the
  next G2 (5f7e6792…) was built on it.
- **Status.** The target (≤ 0.0025%) was not approached; O_3 alone stayed at 0.0068-0.0080% in every
  candidate. Round 2's rule-winner c2 was not switched in as O_3 (owner record, 2026-10-22: within
  retrain noise). No round 3 has run.
- **Sources.** R2 `rounds/o3r-1/`, `rounds/o3r-2/` (`report.txt`, `round.json`, `profile.json`),
  `runs/o3r-1-3/`, `runs/o3r-2-1/` (`round/c*/paired.txt`, `net/report.json`); design
  `m3-hybrid/o3-rounds.md`.

### C13. Specialist network S on G2-flagged positions (runs `specialist-s2`, `golden-routed-s2`, `net-error-o3-val-1`, 2026-10-22)
- **Question** (owner). If a network S is trained only on the positions G2 flags as risky, and used
  on those positions in play instead of exact solves, does it score close to exact?
- **Method.** `search risk-filter` at the play rule's threshold t=277338260 over the stored 40M base:
  kept random 11,077,690 of 25.6M (43.3%), rules 2,235,652 of 6.4M (34.9%), harvest 3,162,386 of 8M
  (39.5%): 16,475,728 rows; held-out 43,681 of the 100k validation rows. S: 2048-1024-256 from random
  weights, 20 epochs, lr 0.001, batch 1024, RTX 5090 (8.8 min training; run $0.56; first attempt
  `specialist-s1` failed on a runner registration bug, $0.68). Routed golden check: G2 picks the
  leaves, S values them, O_3 the rest; compared in the same run with O_3 alone and with G2 + exact
  (AMD EPYC 9B14, 184 usable threads, $1.34/h, 43 min, $0.95). Then `search net-error`: O_3's error on the
  same held-out rows (3 min, $0.07).
- **Result, position level** (held-out flagged rows; S float from the trainer's report, O_3 integer):

| measure | S | O_3 f5766a |
|---|---|---|
| value SD | 3.03% | 2.94% |
| mean \|error\| | 1.09% | 1.08% |
| 99.9th percentile \|error\| | 34.0% | 31.2% |
| largest team bias | 0.027% | 0.033% |
| SD by race class 0-4 | 1.6 / 1.6 / 3.1 / 2.2 / 5.1% | 1.6 / 1.6 / 3.3 / 2.0 / 5.0% |

  O_3 on all 100k validation rows: SD 1.96%, mean |error| 0.49%. S's outcome accuracy 75.1% (O_3's
  not measured on these rows).
- **Result, golden set.** O_3 alone 0.0056 / 0.0068%, seat 1 0.0201%; G2 + exact 0.0005 / 0.0008%,
  seat 1 0.0022% (passes); routed (S on flagged leaves) 0.0064 / 0.0079%, seat 1 0.0216%, own-team bias
  0.0179% (seats 1-2 0.0411%), fails. Paired: routed − G2 exact +0.0059 (+0.0044..+0.0075); routed −
  O_3 alone +0.0008 (−0.0006..+0.0024). Speed: routed 43 ms a world, G2 exact 586 ms, O_3 alone about
  24 ms.
- **Sources.** `history/plan-log-2026-10.md` (2026-10-22 entries); R2 `runs/specialist-s2/`,
  `runs/golden-routed-s2/out/checks/`, `runs/net-error-o3-val-1/`; design `m3-hybrid/specialist.md`.

### C14. Network evaluation speed (answer-identical engineering)
- **Result.** Integer forward pass with runtime-dispatched AVX2/AVX-512 kernels and a memo of recent
  positions: 9-12× faster per evaluation for 512-wide networks, about 7× for 1024-wide; H_4 0.94 ms a
  world (was 10.1), H_3 17 ms (was 175); byte-identical (old code kept as the reference). Trained O_3
  2048-1024-256: 21.5-36 µs an evaluation on the build VM or a Threadripper, 84-90 µs on loaded EPYC
  7B12/7663 threads. CPU/GPU overlap (golden set built beside training; label parts streamed into
  training) is in `m3-hybrid/overlap.md`.
- **Sources.** `m3-hybrid/nn-speed.md`, `m3-hybrid/overlap.md`.

---

## Part D. The trick-2 network O_2 and trick 1

### D1. The trick-2 audit loop and the provisional O_2 (runs `m3-hybrid-t2-loop-1`, `-2`, 2026-10-17/19)
- **Method.** Exact trick-2 labels (about 3 s each on the build VM, 11.3 thread-s on a rented EPYC
  7B12): 1.2M random (s511) + 300k rules-v1 (s512), validation 50k (s601), all stored. 2048-1024-256,
  outcome head (39 endings). No golden set at trick 2: the loop's standard is position-level (200k
  audit shards from trick-1 decisions, γ 0.1%); the only decision-level evidence is the "rough test"
  on the saved exact trick-1 worlds (8 decisions / 31 worlds with the searched burn; 5 decisions /
  355 worlds with the old rules-bot burn; seats 1-2 only). Loop-1: AMD EPYC 7742 (245 threads, RTX
  4060 Ti, $0.72/h); it stopped when the vast balance reached zero (2026-10-18 15:30 Irish). Loop-2
  resumed from the stored labels (AMD EPYC 7742, 330.8 min, $6.99); stopped by the owner after round
  3 ("round 3 no better; vast credit low").
- **Result.** Provisional O_2 = loop-1 round 1, 505a0c7c… (owner, 2026-10-17/18): float validation
  loss 2.0456, outcome accuracy 68.2%, value SD 3.62%, mean |error| 1.46%, team bias within ±0.12%;
  integer SD 3.75%. O_3's for comparison: 1.85%. Loop-2 rounds (integer validation SD; groups
  confirmed; rough test worst set's 95% bound):

| round | network | rows | int SD | confirmed | rough test (rules-burn set, 5 decisions) |
|---|---|---|---|---|---|
| 1 | e2a8602f | 1.5M | 3.655% | 681 | 0.1283% (searched-burn set 0.4837%) |
| 2 | 20188d24 | 1.75M | 3.733% | 678 | 0.1498% (searched-burn set 0.0000%) |
| 3 | 92b705dd | 2.0M | 3.851% | 681 | 0.1159% (searched-burn set 0.0000%) |

  The worst groups were score states (a team on 35, 36 or 40 points; SD 6-15%). Validation SD rose
  slightly each round while targeted rows were added.
- **Limits.** The rough test is 5-8 decisions at seats 1-2; its bounds are coarse and the rules-burn
  worlds were made by an older binary.
- **Sources.** `m3-hybrid/rung3.md`; R2 `runs/m3-hybrid-t2-loop-2/partial/rung/000000/out/hybrid/audit/round-*.txt`.

### D2. Cheaper trick-2 labels: hybrid labels, G v2 labels and the label test (2026-10-19/20)
- **Question.** Can O_2's training labels come from H_3 (exact through trick 2, then O_3; TreeStrap,
  Veness et al. 2009) instead of exact solves, and at what cost and quality?
- **Measured cost per trick-2 label.** Build VM: exact 3,650 thread-ms; plain hybrid (O_3 only) 86
  thread-ms; hybrid + race:1-4 exact leaves 2,800; hybrid + G v2 2,000. Rented EPYC 7B12: exact 11.3
  thread-s, G v2 9.3 thread-s (G v2 labels about 1.2× cheaper than exact; an exact trick-2 solve is
  mostly its ~400 trick-3 leaf solves and G v2 still solves 40% of them).
- **Runs.** `o2-gv2-labels-2` (AMD EPYC 7B12) and `-3`: O_2 on 10M G v2 labels, stopped by the owner
  after about an hour (no part finished; $0.34 for -2); `-3` made a 5k exact validation shard (s603).
  `o2-labeltest-1` (plain hybrid vs exact labels at 600k, same exact validation) was cancelled after
  8 min, before the exact-label part and before any training: it made the 600k plain hybrid labels
  (358.6 s on an Intel engineering sample, 184 threads, $0.40/h: 9.1 labels per thread-second, the
  shard not stored) and a 20k exact validation shard (s602). STATUS-history records "O_2 on
  G v2 labels: rejected" (2026-10-20) on the cost figure.
- **Label quality measured instead** (C12's label check): O_3-based trick-2 labels against exact on
  50k positions: RMS error about 2.1%, largest team bias 0.013-0.017% (leader's team negative).
- **Sources.** `m3-hybrid/audit-loop.md` §12, `m3-hybrid/learned-leaves.md`, `history/STATUS-history.md`
  (2026-10-20).

### D3. O_2's own error predictor, "O_2 G v2" (run `o2-leafrisk-1`, 2026-10-20)
- **Question.** Is O_2's error as concentrated as O_3's, so that deferring a few trick-2 leaves of a
  trick-1 search to an expert removes most of it?
- **Method.** Trick-1 decisions searched by H_2 with the provisional O_2 (505a0c7c); trick-2 frontier
  positions harvested and labelled exactly: train 300k (s79001), calibration 150k (s79002, 5,039
  decision units), test 50k (s79003, 1,684 units); the same tree model and calibration as C11 (110
  of 150 trees kept). AMD EPYC 7B12 (256 threads, $0.30/h), 330 min, $1.63. Model 76865b9c… (file),
  calibration's model SHA 04c02935…; the model was not stored content-addressed by the run (`stored`:
  null).
- **Result** (test split; share of O_2's squared error left after solving a share of positions;
  O_3's figures from C11 for comparison):

| solved | O_2: learned | O_2: best possible order | O_3: learned |
|---|---|---|---|
| 5% | 49.1% | 29.7% | 25.3% |
| 10% | 36.6% | 18.7% | 16.6% |
| 20% | 23.9% | 8.7% | 8.9% |
| 30% | 16.5% | 4.0% | 4.7% |
| 40% | 11.6% | 1.7% | 2.2% |
| 50% | 7.7% | 0.6% | 1.4% |
| 75% | 1.4% | 0.0% | 0.5% |

  O_2's unsolved SD on these positions: 3.28% (calibration 3.51%; O_3: 1.92%). Race policies on O_2's
  positions: race:4 solves 24.1% and leaves 40.2% (learned at that share: 20.6%); race:2,4 solves
  52.4% and leaves 19.6% (learned: 7.0%). Learn-then-Test thresholds (CLT), target SD left → share
  solved: 2.0% → 12%, 1.6% → 21.7%, 1.2% → 39.1%, 1.0% → 49.3%, 0.5% → 75.5%. Most important inputs:
  distance to the nearest possible ending 44.8%, number of possible endings 13.3%, widest money range
  of one team 3.9%.
- **Not done.** The trick-1 decision test of O_2 + G v2 (planned against a silver set, D4) has not
  run.
- **Sources.** R2 `runs/o2-leafrisk-1/out/leafrisk/` (`model.json`, `calibration.json`);
  `m3-hybrid/learned-leaves.md` ("O_2 G v2").

### D4. The trick-1 silver set (built 2026-10-17; attempts 2026-10-20/21)
- **Design.** A reference for trick-1 decisions without exact trick-1 solves: H_3 (exact through
  tricks 1-2, then O_3) on the same decisions and worlds as H_2 (exact trick 1, then O_2). Identity
  recorded in the design: true loss of H_2 = loss against silver + H_3's own true loss + (δ_s* − δ_h),
  so a pass against silver bounds H_2 relative to H_3, not relative to exact.
- **Pilot cost** (build VM, 4 decisions a seat, world 0): an H_3 leader world searches about 47M solver
  nodes and asks O_3 about 10.9M times, 712-1,800 s; seat 2 330 s mean; seats 4-8 seconds. Estimated
  on an AMD EPYC 9654 (368 threads, $1.01/h): 400 × 320 $42-82; 400 × 80 $5-10. 65-75% of the cost is
  seat 1.
- **Attempts.** `o2-silver-1` (AMD Threadripper PRO 9995WX, 192 threads, $0.60/h): FAILED after 79
  min, $0.79, 143 worlds checkpointed of 32,000 (400 × 80). `o2-silver-2/-3/-4` and `sx50_*`,
  `s2_50` were part of the work of 2026-10-19/20 that the owner discarded (branch
  `archive/silver-shared`); they checkpointed a few hundred worlds each (210, 192, 471). **No silver
  set exists.** The optional exact anchor (`m3-hybrid-t1-anchor`) was not run.
- **Sources.** `m3-hybrid/silver.md`; R2 `runs/o2-silver-*/`.

### D5. Trick-1 checks on the saved exact worlds (rough test)
- **Result.** See D1: H_2 with the trick-2 loop's networks on 5-8 saved trick-1 decisions; 95% bounds
  0.12-0.48% on the worst set, team biases 0.32-0.59%. These are the only trick-1 decision-level
  numbers measured against exact values.

### D6. O_2 label source and volume, Stage A (runs `o2-stagea-full-1/-2`, `g2-labelcheck-1`, `o2-g2arm-1/-2`, 2026-10-22)
- **Question.** Which labels train O_2 best per dollar: exact, plain hybrid (exact through trick 2, O_3 f5766a
  after) or G2 hybrid (plain plus the learned exact leaves)? Judged on a new exact test set, never trained on.
- **Method.** Same network everywhere (2048-1024-256, outcome head, seed 1, early stopping on the exact s601
  shard), 80/20 random/rules-v1 positions. Arms: exact 0.3M (`e03`), 0.75M (`e075`), 1.5M (`e15`) from the stored
  s511+s512 shards (`train_fraction`); plain 1.5M (`p15`, seeds 541/542) and 6M (`p60`, seeds 545/546). Test set:
  250,000 exact trick-2 positions in ten parts (val seeds 604-613, stored; keys in each step's `worker.log`),
  scored with `fortyone_nn.netcmp` (paired differences, 95% bounds, normal approximation, rows independent), plus
  4,479 exact sibling groups (15,000 positions, seed 7604, shard 5734d7f0) for the mover's mean regret.
  Machine: AMD Threadripper PRO 7995WX (184 threads, $1.27/h), 262 min, $4.73 (the first attempt, `-1`, lost $1.21:
  the trainer refused hybrid-labelled shards; fixed, `test_hybrid_train`).
- **Result** (error of the money chance, percent; `docs/m3-hybrid/o2-stagea-netcmp.json`):

| arm | labels | SD | paired change in squared error vs `e15` | largest team bias | sibling regret (95%) |
|---|---|---|---|---|---|
| `e03` | exact 0.3M | 4.303 | +5.27 +-0.33 | 0.412 | 0.324 +-0.055 |
| `e075` | exact 0.75M | 3.931 | +2.20 +-0.30 | 0.180 | 0.327 +-0.056 |
| `e15` | exact 1.5M | 3.640 | reference | 0.210 | 0.295 +-0.056 |
| `p15` | plain 1.5M | 3.668 | +0.20 +-0.29 (tie) | 0.116 | 0.295 +-0.055 |
| `p60` | plain 6M | 3.132 | -3.44 +-0.30 | 0.126 | 0.247 +-0.051 |

  Standard error of a team bias about 0.01%. Sibling-regret differences between arms are inside the noise.
- **Reading.** At equal volume plain labels tie exact ones; at equal money (6M plain about $0.4 against 1.5M exact
  about $12) plain wins clearly; volume has not flattened (SD 3.67 to 3.13 for 4x data; extrapolated 40M about
  2.5%, not measured). Every network's own team bias (0.1-0.2%) is far above the plain labels' 0.013-0.017%.
  O_2's best SD 3.13% is far from O_3's 1.85%.
- **Label check** (`g2-labelcheck-1`, EPYC 7663 215 threads, $0.51): the same 10,000 positions labelled three ways.
  RMS error against exact: plain 2.145% (largest bias 0.0155%, 16% of rows identical to exact), G2 0.479% (bias
  0.0063%, 55% identical). Speed measured on 215 threads: exact 341.6 s (7.3 thread-s a label), G2 172.4 s (3.7),
  plain 9.7 s (0.21). Wider-machine speeds: plain 1,607 labels/s on the 7995WX (40M plain labels about 7 h, about
  $9, not the $2.7 once estimated); G2 about 50 labels/s on an EPYC 7B12 (245 threads).
- **G2 arm** (0.75M G2 labels, same recipe): `o2-g2arm-1` (EPYC 7663, $1.60/h) made part 1 (120k random + 30k rules,
  seeds 561/562) and was stopped; `o2-g2arm-2` (EPYC 7B12 taken over by hand, instance 57260975, $0.73/h) makes the
  other 600k (seeds 563/564) in four parts; a training-only run on all stored parts plus s601 then gives the
  network. Result pending (see STATUS).
- **Limits.** One seed per arm; the test set is random-play positions, not the positions trick-1 search reaches;
  no trick-1 decision-level number exists, so O_2 can only be audit-clean on exact trick-2 positions.
- **Tools made.** `fortyone_nn.netcmp`, `python/tools/o2watch.py` (progress, speed, ETA from R2 heartbeats),
  `vast-run.yml` input `instance_id` (take over a machine rented by hand; the instance needs the run's disk, 80 GB).

---

## Part E. Students (small networks for the app's tricks 1-2)

### E1. Pipeline and the first local student (2026-10-17/18)
- **Method.** A plain MLP over one seat's view (encoder `student-v1`, 3,292 inputs) scoring every
  legal candidate, integer inference (byte-identical natively and in WASM), trained on the teacher's
  per-card values in every deal (Hinton et al. 2015; Rusu et al. 2016; Expert Iteration, Anthony et
  al. 2017). Teacher at trick 2: PIMC over D deals with H_3 (O_3 f5766a). Generator speed: 4.7
  thread-s a trick-2 decision at D = 320 on the build VM.
- **Result.** First tiny student (3,000 decisions at 16 deals; test 512 decisions at 64 deals): random
  card 0.520%, rules-v1 0.351%, student 256-128 0.366%, 512-256 0.289%. In WASM a 512-256 student
  request took 0.4-0.7 ms (hybrid PIMC trick-2 request of 4 deals: 12.6 ms natively).
- **Sources.** `STUDENT.md` §11.

### E2. Learning curve and width (runs `student-t2-base-1`, `-curve-1`, `-more-1/-2`, `-curve-1m-1`, 2026-10-18/19)
- **Method.** 512-256 students on rules-bot and mixed trick-2 decisions at 320 or 64 deals; fixed test
  set 4,096 decisions at 1,024 deals.
- **Result.** Decision loss against the teacher: 50k 0.186%, 100k 0.134%, 250k 0.078% (rules-v1:
  0.40% on the same test), 1M 0.057%. The last 4× of data gained 27%, consistent with loss ∝ N^-0.23.
  1024-512 at 1M: 0.056% (no measurable gain over 512-256).
- **Sources.** `history/plan-log-2026-10.md` ("Student to 0.01"), `STUDENT.md` §14.

### E3. Deals per decision at equal teacher cost (run `student-t2-dealtest-1`, 2026-10-18, $0.67)
- **Result** (512-256; loss, 95% bound):

| positions | deals | teacher cost | loss | agree |
|---|---|---|---|---|
| 126,000 | 320 | 1× | 0.1140% (0.1317) | 63.7% |
| 315,000 | 128 | 1× | 0.0695% (0.0771) | 64.2% |
| 625,000 | 64 | 1× | 0.0760% (0.0875) | 63.8% |
| 625,000 | 32 | 0.5× | 0.0776% (0.0862) | 63.3% |
| 126,000 | 64 | 0.2× | 0.1463% (0.1670) | 60.0% |
| 250,000 | 320 | 2× | 0.0784% (0.0912) | 64.4% |

- **Decision recorded.** New student data at 64 deals (2026-10-18).
- **Sources.** `STUDENT.md` §13.

### E4. The ruler: split-half floor and the top-k shortlist (2026-10-19, build VM)
- **Method.** Even deals = half A, odd = half B; pick on one half, score on the other (no winner's
  curse). "Floor" = a 512-deal teacher scored against the other half. Top-k: the student shortlists k
  cards and the teacher picks among them.
- **Result** (1M, 512-256, 64 deals): full loss 0.0566%; split student 0.0586%; floor 0.0080%; gap
  0.0506% (90% range 0.0443-0.0575); top-2 0.0174% full / 0.0233% split; top-3 0.0050% / 0.0122%.
  Leaders: floor 0.0177%, full loss 0.0838%, top-3 0.0194%. S4: floor 0.0140%, loss 0.0889%. The
  1024-512 students' gaps overlap the 512-256's.
- **Sources.** `STUDENT.md` §14.

### E5. The student against the exact golden set (2026-10-20)
- **Result.** Trick-2 student (512-256, 1M, plain H_3 teacher; 954223e2…) on the 2,000-decision
  exact golden set: alone 0.073% (95% bound 0.088%), seat 1 0.18%; top-3 shortlist with a perfect
  final pick 0.0073% (bound 0.013%), seat 1 0.0148% (bound 0.024%).
- **Sources.** `history/STATUS-history.md` (2026-10-20).

### E6. Trick-1 teacher cost (run `student-t1-cost-2`, 2026-10-18)
- **Method.** 2,000 rules-v1 trick-1 decisions at 64 deals, teacher H_2 with the provisional O_2, in 4
  parts. AMD EPYC 7B13 (256 threads, $1.05/h).
- **Result.** 12,489 decisions an hour (part 0: 500 decisions in 144 s); 5.28 candidates a decision;
  run 18 min, $0.32. That is about $84 per million trick-1 decisions at that price. Build VM estimate
  had been 1.2 thread-s a deal (77 thread-s a decision).
- **Sources.** R2 `runs/student-t1-cost-2/out/rules/`; `history/plan-log-2026-10.md`.

### E7. Trick-2 student with a G v2 teacher (runs `student-t2-gv2-1/-2`, 2026-10-20)
- **Result.** No data produced: `-1` failed on a runner bug (fixed in 6c7acf5), `-2` was stopped by the
  idle watchdog during setup.

### E8. Does a shortlist make the exact solve cheaper? And a trick-1 H_3 world (2026-10-22, build VM)
- **Question.** If the student names 2 or 3 cards, does the search over them cost less per deal? What
  does one trick-1 deal cost when the search is exact through tricks 1 and 2 and O_3 values trick 3?
- **Method.** `engine-native time` on 32 trick-2 positions (4 per place, 16 deals each, one thread,
  native x86-64), with the candidate list cut to the first or last 2 or 3 cards by a local, uncommitted
  patch of `crates/wasm/src/api.rs`; the network was the 80M-base O_3 (`o3-base80-1`, same
  2048-1024-256 shape, not the adopted f5766a, which is not on the VM), the equity table rebuilt from
  `data/equity` (SHA-256 matches the pinned 64740dab). Separately `search teacher --trick 1 --deals 1
  --threads 1` with that network (exact through tricks 1 and 2, then O_3) on one seed-7 decision.
- **Result.** All cards, ms a deal by place 1-8: 82, 31, 7.3, 5.2, 2.8, 1.0, 0.6, 0.3 (the stored
  `WASM-API.md` §8 native figures are 73, 25, 6.7, 5.0, 3.1, 1.2, 0.75, 0.38). With 2 or 3 cards the
  same within noise (leader 76-83): the exact solver values every root card in one search
  (`search::pimc::exact_world`, "one solve values every root card"), so the candidate list does not
  change the cost. The candidate lists had 2-4 cards (leader 4 in all four samples). The trick-1 H_3
  deal was **not finished when stopped** (one run at 590 s, another at its background limit, up to 50
  minutes, of one thread); D4 records 10-30 thread-minutes for a leader world on the build VM.
- **Limits.** 4 positions a place; the stand-in O_3; the trick-1 timing is one decision and a lower
  bound. A root-restricted solver was then built (`FastOptions::root_keep`,
  `crates/solver/tests/root_keep.rs`): on 20 tricks 3-5 test positions one root card cost 509,266
  nodes against 1,353,126 for all cards, kept cards keep byte-identical values; its saving at tricks 1-2
  with a network at the leaves is not yet measured (the search barely prunes, 106 of 81,000 nodes at
  trick 3, `ENGINEERING-2026-10-17.md`, so cost should follow the number of root cards kept).
- **Sources.** `docs/m3-hybrid/path-forward.md`.

### E9. Shortlist + search at the app's deal counts (`search student-check --sub-deals`, 2026-10-22, build VM, no rental)
- **Question.** At the 25-100 deals a phone can afford, does "the student shortlists 3 cards, a short
  search picks" beat the student alone?
- **Method.** S2 (954223e2) and O_3 (f5766aa9) on the 2,000-decision exact trick-2 golden set (320 exact
  worlds each). Per decision and per repeat (8, seed 1) a seeded shuffle of the 320 worlds; the first N
  are the sample (N = 10, 25, 50, 100; nested, so one shuffle serves every N), the method picks on the
  sample and is **scored on the other 320-N worlds**. Methods: the student alone, search over all cards,
  top-2 and top-3 of the student searched. "Corrected loss" = the student's loss on all 320 worlds less
  the method's gain over the student on the scoring worlds, so it is on the golden check's scale (the
  student's own is 0.0733%, the golden check's number to the digit); the raw numbers (best of the scoring
  worlds less the pick) are in the JSON. Bounds: bootstrap 95% over decisions (1,000 replicates, seed 1).
  `crates/search/src/shortlist.rs`, 4 unit tests; the run takes 6 s.
- **Result** (loss in % of the money chance; 95% bounds in brackets):

| deals | student alone | search all cards | top-2 + search | top-3 + search |
|---|---|---|---|---|
| 10 | 0.0733 | 0.1164 | 0.0836 | 0.1030 |
| 25 | 0.0733 | 0.0619 | **0.0542** | 0.0596 |
| 50 | 0.0733 | **0.0384** | 0.0422 | 0.0402 |
| 100 | 0.0733 | **0.0254** | 0.0338 | 0.0293 |

  Gain over the student alone (positive = better): top-3 + search +0.0137 [+0.0018, +0.0259] at 25
  deals, +0.0331 [+0.0210, +0.0459] at 50, +0.0441 [+0.0319, +0.0570] at 100; at 10 deals it is
  **worse** by 0.0297 [0.0175, 0.0427]. Seat 1 (the leader, where the student loses most): student
  0.179; top-3 + search 0.146 / 0.097 / 0.073 / 0.061 at 10 / 25 / 50 / 100 deals. Stratum S4: student
  0.120; top-3 + search 0.171 / 0.100 / 0.072 / 0.055.
  **The shortlist against searching every card:** top-2 beats search-all at 10 deals (+0.033) and 25
  deals (+0.0077 [+0.0013, +0.0138]), no difference at 50, and is **worse** at 100 (-0.0084 [-0.0149,
  -0.0027]); top-3 is within noise of search-all at 25-100 (+0.0023, -0.0018, -0.0039, every bound
  across zero). So at the app's deal counts a shortlist buys cost only, not accuracy.
- **Verdict.** Yes: at 25 deals top-3 + search is better than S2 alone (by 0.014 points, bound just
  above zero), clearly at 50 and 100, no at 10. It stays far above the "perfect pick" 0.0073% of E5,
  which needed all 320 worlds. The break-even lies between 10 and 25 deals a decision.
- **Limits.** Exact world values: no leaf error, optimistic for any real search. The sample is drawn
  from 320 worlds without replacement (31% of them at N = 100), which lowers its variance by about
  17% against independent draws, so the 100-deal numbers are a little optimistic. The truth proxy
  (320-world means) is itself noisy (a 512-deal teacher's floor is 0.008%, E4). The stored golden set
  is trick 2 only. The phone cost is a separate matter: the leader's trick-2 deal is 368 ms in
  WebAssembly on the build VM (`WASM-API.md` §8), so 7 s on one thread is about 19 deals for the
  leader (27 with a 3-card shortlist, E11), and a phone is 2-4 times slower: the leader may not reach
  the break-even at all, exactly where the student loses most.
- **Sources.** `docs/m3-hybrid/path-forward.md` section 3 step 0a.

### E10. How concentrated is S2's loss, and can a router find the risky decisions? (same run as E9, plus `tools/router_fit.py`, 2026-10-22)
- **Question.** If the loss sits in a few decisions, the app can spend its 7 seconds only there.
- **Concentration** (2,000 decisions; 547 have any loss, 1,453 (72.7%) lose nothing): the worst 1% of
  decisions hold 38.8% of S2's total loss, the worst 5% hold 75.1%, 10% hold 90.9%, 20% hold 99.1%,
  40% hold all of it. Concentrated, as G2's idea assumed.
- **The margin does not find them.** Sorting by S2's own margin (top value less second, for the
  deciding pair) and searching the lowest f% with top-3 + N deals, at 100 deals (all searched: 0.0293):
  f = 10% 0.0711, 30% 0.0671, 50% 0.0588, against the student alone 0.0733. The correlation of margin
  and loss is 0.01 and the 10% with the lowest margin hold 5.4% of the loss, below their share: a low
  margin is a near-tie, and a near-tie costs little either way.
- **A learned router does a little better, nowhere near the oracle.** Gradient boosting on what the
  student shows (place, stratum, candidate count, its top values, margins and spread; 5-fold
  cross-validation, 5 seeds; `tools/router_fit.py`): correlation with the true loss 0.048; the
  predicted-riskiest 10% / 20% / 50% hold 23.9% / 40.4% / 72.1% of the loss. Router at 100 deals
  (loss %; searched share, 95% bounds of the model router): 10% 0.0613 [0.052, 0.071], 20% 0.0548, 30%
  0.0504 [0.044, 0.063], 50% 0.0423 [0.035, 0.051], 100% 0.0293; i.e. 30% of the searches give 52% of the
  gain, 50% give 70%. At 25 deals: 30% gives 71% of the gain (0.0636 against 0.0596 for all). The
  oracle router (search the decisions with the largest true loss): 5% 0.0267, 10% 0.0193, 20% 0.0174,
  better than searching everything (0.0293) because a search cannot hurt where the student is already
  right.
- **Verdict.** The concentration is real and large (the oracle needs to search 5-10% of decisions to
  beat searching all of them), but nothing the student shows on its own finds them: do **not** build
  the margin router. A risk model is worth testing only with better signals (a second student's
  disagreement, a G2-style predictor trained on the student's losses on held-out teacher decisions);
  the cheap-feature model above is the floor. This router is the assistant's idea (not a published
  recipe): by the project's rule a duplicate tournament would decide before it ships.
- **Limits.** One golden set, 2,000 decisions (the router was fitted and scored by cross-validation on
  the same set); trick 2 only; the router's cost (a fast network call) is not counted.
- **Sources.** `tools/router_fit.py`; the JSON of the run (`per_decision`) is regenerated by the E9
  command with `--json`.

### E11. The root-restricted solve (2026-10-22, build VM)
- **Question.** Does searching only the shortlisted cards make the exact solve cheaper (E8 said the
  candidate list alone changes nothing)?
- **Built.** `FastOptions::root_keep` in the solver (earlier commit) is now passed by
  `search::pimc::exact_world` from the candidates' cards (all legal cards = the full solve; burns pass
  none), and the app API takes an optional `shortlist` of plays (`WASM-API.md` §3; refusals; the key
  includes it; shards of one shortlist merge byte for byte; test
  `a_shortlist_values_its_cards_exactly_as_the_full_search_does`).
- **Method.** `tools/shortlist_time.py` over the API's bench requests (`engine-native bench`, 16 deals,
  4 positions a place; native x86-64, one thread; real O_3 and the pinned equity table): value each
  position in full, then with a shortlist of its best 2 or 3 cards, assert that every shortlisted card
  has the same sums as the full run, and time both (best of 2).
- **Result.** Values identical in every case. Time as a ratio of the full search: **trick 2, 2 cards
  0.64, 3 cards 0.70** (summed over places); by place for 2 / 3 cards: leader 0.66 / 0.71, second 0.57 /
  0.65, third 0.41, 4th-5th 0.57-0.76; seats 6-8 are too cheap to matter (about 0.1 ms a deal, ratio
  0.85-1.08, within noise). Trick 3 (exact), 2 cards: 0.69 (leader 0.64). Absolute, leader at trick 2:
  100 ms a deal falls to 66 (2 cards); 116 to 82 (3 cards).
- **Verdict.** Exact, but a **1.4-1.55 times** speed-up, not the 2-3 times the share of cards kept
  suggests: most of the work below the root is not specific to one root card. With E9 (a 3-card
  shortlist is no more accurate than searching all cards at the same deals) the shortlist is worth
  about 40% more deals in the same time, nothing more. Trick 1 was not measured (no bench request
  exists for it and a trick-1 deal is minutes on this VM, E8).
- **Limits.** 3-4 positions per place and trick; timings on a shared build VM (best of 2); the
  shortlists were the true best cards, the cost depends on which cards are kept.
- **Sources.** `tools/shortlist_time.py`, `crates/wasm/tests/api.rs`.

---

## Part F. Runs and builds without a result, and work set aside

| item | state | what exists |
|---|---|---|
| O_3 base curve 50% / 100% (`o3-basecurve-*`) | cancelled at 37 s, 2026-10-19 | nothing |
| O_3 wider / deeper (`o3-size-*`) | never launched | manifests |
| Sibling ranking A/B (`o3-sibrank-a/-b`) | never launched | build-VM A/B only (C9) |
| Four-network ensemble | stopped twice for speed | single-network checks only (C8) |
| Hybrid-label test (`o2-labeltest-1`) | cancelled at 8 min | the hybrid labels' speed; 20k exact validation shard (s602) |
| O_2 on G v2 labels (`o2-gv2-labels-2/-3`) | stopped by owner | 5k exact validation shard (s603) |
| Trick-2 hybrid-label run (`m3-hybrid-t2-hlabels`) | never launched | manifest |
| Silver set and exact anchor (D4) | incomplete / not run | a few hundred checkpointed worlds |
| O_2 G v2 trick-1 test (D3) | not run | the predictor model |
| Trick-1 deal-count curves (A7) | incomplete | 31 + 355 saved exact worlds |
| PIMC-v0 vs rules-v1 SPRT gate (M3-SPEC §8.3), PIMC-suitability report (§6) | not run | — |
| Speed tests of AMD EPYC 9B45 offers (2026-10-22) | stopped: 46 usable threads | no speed figure |
| Copilot work of 2026-10-19/20 (`silver-shared`, `sx50_*`, `s2_50`, `o2-silver-2..4`, `o3v2-*`, `o3g2-linux-verify-*`, `o3-targeted2m-recovery-*`) | discarded by the owner | files in R2, not analysed here; branch `archive/silver-shared` |

---

## Part G. Measured costs, side by side

| what | measured cost | where |
|---|---|---|
| exact solve, start of trick 4 / 3 / 2 (one thread, build VM) | 0.04 ms / 8.8 ms / 3.1 s median | A3 |
| exact trick-2 world on a loaded EPYC 7B12 thread | about 2.5 s | C10 |
| exact trick-1 world, leader | ≥ 47 ×10^9 nodes mean; one world 13,057 s on one thread | A4 |
| H_3 a trick-2 world (2048-1024-256 O_3), loaded EPYC 7B12 | 51 ms; Threadripper 21 ms | C7, C4 |
| H_3 + G v2 (t=277338260) a trick-2 world | 1,550 thread-ms (586 ms on a 9B14) | C11, C13 |
| H_3 + race:4 / race:2,4 / race:1-4 | 232 / 668 / 1,692 thread-ms | C10 |
| H_3 with S on flagged leaves | 43 ms | C13 |
| H_3 at trick 1 (silver), leader world | 10-30 thread-minutes (build VM) | D4 |
| exact trick-3 label | about 9 ms (build VM) | A3 |
| exact trick-2 label | 3.65 s build VM; 11.3 thread-s EPYC 7B12 | D2 |
| G v2 trick-2 label | 2.0 s build VM; 9.3 thread-s EPYC 7B12 | D2 |
| plain hybrid trick-2 label | 86 thread-ms build VM; 110 thread-ms (9.1 a thread-second) on an Intel engineering sample, 184 threads | D2 |
| trick-2 teacher decision at 320 deals (H_3) | 4.7 thread-s build VM | E1 |
| trick-1 teacher decision at 64 deals (H_2) | 12,489 decisions an hour on 256 EPYC 7B13 threads | E6 |
| student request in WASM | 0.4-0.7 ms | E1 |

---

## Part H. Stored networks and data referred to above (R2 `inputs/sha256/<hex>`)

| what | key (prefix) |
|---|---|
| O_4 adopted (trick-4 loop round 1) | b28750e3d447… |
| O_3 adopted f5766a (2048-1024-256, 40M) | f5766aa98a1d…, float b5679677… |
| O_3 at 80M base | 7eeb1f8862cc… |
| O_3 other loop networks | b34e12b5 (t3 loop r1), a099061b (r3), bfd1bec2 (r4), a97ac177 (additive-2 r2), b89b26d6 (rung-2 round 1) |
| O3G2 round 1 candidates | e9fabe22 (c1), 1ba7f1ff (c2), 23c1d23f (c3) |
| O3G2 round 2 candidates | 32042aef (c1), 1057e9e9 (c2, adopted by the round), c2d1f2fa (c3, 4096-wide) |
| G v2 for O_3 (leafrisk-o3-v1) | 20c41678c646…; next G2 on round 2's winner 5f7e6792… |
| Specialist S | int d30ef496…, float f5f6768d… |
| O_2 provisional | 505a0c7c6514… |
| O_2 leaf-risk model | in `runs/o2-leafrisk-1/out/leafrisk/model.json` (file SHA 76865b9c…) |
| trick-2 golden set (2,000 × 320) | decisions 635e9aef…, worlds 6ddc59b1… |
| trick-3 base (40M) | random fd846dad…, rules 91c6e001…, harvest a80cc287…, validation d51e22ab… |
| trick-3 80M additions | random a27d0ce2…, rules 79ed2709…, harvest 9ea6a23a… |
| exact trick-2 labels | random 1.2M fb51c217…, rules 0.3M 2160fefc…, validation 50k 38de4835…, 20k 855ba467…, 5k 46ba92fc… |
| trick-2 student (1M, 512-256) | 954223e2… |
| equity table v0 | 64740dab…6808 |
