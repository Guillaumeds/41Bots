# Experiments: every experiment we ran, its method and its result

This is the single collected record of what the project has measured, built to measure, tried and
stopped, from the first engine test (2026-10-13) to the trick-1 teacher data of 2026-10-23. Each entry is
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

### A10. Taking over, renting and watching machines (2026-10-22/23)
- **Method.** The O_2 work of 2026-10-22/23 ran on many rentals. What the launcher (`python/launcher/launch.py`), the vast-run workflow and
  the watcher (`python/tools/grab_machines.py`) did on them is recorded here from the runs' own logs and R2 records.
- **Result.**
  - **Every rental is Jupyter + SSH** since 2026-10-23 (`--jupyter --direct`, the console's Connect button; the boot script starts from the
    on-start command because vast's own init replaces the image's entrypoint). The SSH trial `ssh-trial-1` ran 80 s and kept the machine. The
    agent's network can reach only an allow-list of hosts: a machine's Jupyter port and SSH could not be opened from the session (connection reset).
  - **Takeover of a machine rented by hand** (`instance_id`): vast's `update instance` stores the new image, environment and start command and
    `recycle instance` rebuilds the container. First heartbeat 3-4 minutes after the call on the New York machine and the cheap EPYC 7B12 box.
    Failures: the EPYC 7742 box 57346368 (container started on vast's base image, no heartbeat in 15 minutes); the Threadripper 9995WX box
    57386924 (the recreate sent 25 s after creation had no effect for 25 minutes; the container then started with the first attempt's environment,
    `RUN_ID=o3-pair-1`, while the retry `o3-pair-2` had updated the stored settings; both launcher jobs gave up after 15 minutes without a
    heartbeat and kept the machine, the runner went on and wrote to `runs/o3-pair-1/`).
  - **Keep.** A cancelled run of a machine with `keep` destroyed the machine on 2026-10-23 (the keep marker was written only when the launcher's
    interrupt handler ran, and the workflow kills the launcher within seconds); the marker is now written when a kept run starts. The launcher
    refuses a machine whose label is not `41bots-<run>` (`force_take` in the manifest's machine block overrides it); two sessions took the same kept
    machine within two minutes on 2026-10-23.
  - **Renting.** The workflow route (check the offer, then `vast-run.yml`) needs 30-60 s; offers of machines that come and go were gone by then
    (the EPYC 7742 3070 offer 56279046 on 2026-10-23). One API call (`PUT /api/v0/asks/<offer id>/` with vast's Jupyter + SSH image, 200 GB disk)
    rents in about a second; it rented the Threadripper 9995WX box (instance 57386924) and the Norway RTX 4090 box (57373788) that way.
  - **Watcher.** `grab_machines.py` polls named machines and, with `--generic`, any AMD EPYC or Threadripper offer with 190+ effective threads, 120+ GB
    RAM and 500+ Mbps at most $1.50/h. Its first thread-only rule took a Xeon Phi 7250 box (272 slow threads, instance 57365260, $1.14/h); the direct rent does not go through the launcher's refusal of Xeon Phi hosts.
    The vast API `/api/v0/instances/` is deprecated (`/api/v1/instances/`); labels are set with `PUT /api/v0/instances/<id>/`.
  - **Hosts that failed.** An unverified RTX 5090 + EPYC 7H12 host (60% reliability) pulled the image at 6 Mbps; a Michigan 4 x RTX 5090 box could not
    start containers (`failed to inject CDI devices: unresolvable CDI devices`); the EPYC 9B45 gave 46 threads again (A8).
  - **Every chunk fetches its inputs again** (a finished chunk's folder, inputs included, is deleted): in `o3-pair-1` each of the five arms read the
    same 62 GB (about 20 minutes at 50-65 MB/s, GPU idle). A `student_teacher` chunk of 500 decisions spends about 40 s of its 105 s in setup (D13).
  - **Nine failed S1 training launches (2026-10-24, `student-t1-250k-1` to `-9`).** Machines on the Thailand 3060 host (taken over three times,
    rented once), a Hungary P100 box, a UK 2080 Ti box and the Nevada 4x RTX 5070 Ti offer each disappeared from the account 1-7 minutes after
    their container started, in both launch modes (Jupyter + SSH and `entrypoint_mode`). Cause: the launches used `bin_sha` 2c0774d (2026-10-22
    06:48); the trainer's per-point `rank_weight`, `rank_temperature` and `stop_on` options came in commit cf47745 (2026-10-22 13:16) and are in
    `bin_sha` dd19e4d (2026-10-23 06:50), which the S2 runs used. The older runner refused the manifest (exit status 2), `deploy/docker/boot.sh`
    wrote `runs/<id>/FAILED` (`{"stage":"manifest", ...}`) and the machine destroyed itself; the launcher reports that as "instance gone before the
    first heartbeat" or "destroyed itself (not listed)". The first of these machines was also destroyed by the cleanup job (below). The
    manifest-resolution test `manifests_resolve_into_settings_their_chunks_accept` does not cover `student_train`, and the local `runner run` check
    that would have shown the refusal was made with the current source, not with the released binary.
  - **The cleanup job destroyed a taken-over machine (2026-10-24 09:10 UTC).** `python/launcher/cleanup.py` counted silence from the instance's
    `start_date`, which a takeover keeps from the original rental: "no heartbeat at all in its 928 min of life", with the new container still
    booting and vast showing CPU 0%. Now the count starts at the run's launch (`runs/<run>/manifest.json`); test
    `test_taken_over_old_machine_counts_from_its_runs_launch`. Machines whose label does not begin `41bots-` are never touched by the cleanup.
  - **Other launcher facts from that day.** A cancelled launcher with `machine.keep` leaves the machine up (verified on three cancels); the launcher
    marks a machine "bad" for 24 hours after a start failure, and a named offer on such a machine is still rented ("because the owner named it");
    `offer_id` fails with "not on offer any more" when the offer was rented in the meantime; the cancelled generic search had already rented a UK
    box before the cancel reached it.
  - **A takeover's settings reach the container one recycle late (found 2026-10-24/25, instance 57517251).** `vast update instance` stores the new image,
    environment and start command; the `recycle instance` the launcher sends right after builds the container from the settings stored by the PREVIOUS
    update: on 57517251 the stored `RUN_ID` was `o2-s1more-4` while the container ran `o2-s1more-3` (read from `GET /api/v0/instances/<id>/`, field
    `extra_env`, and the runs' heartbeat names). Each takeover therefore started the previous takeover's run: the launcher watched a run name that never
    heartbeated, gave up after 15 minutes ("no heartbeat ... takeover did not start") and kept the machine, while the work went on under another run name
    (here 7 chunks were redone). The first takeover on a fresh machine shows the vast.ai base image. To start run B on a machine: take it over with B (the
    container runs the previous settings), then take it over once more under a new run name C (the launcher refuses a name that exists in R2:
    `runs/C/ already exists`); that recycle builds the container from B's stored settings, B's manifest is read from R2 and B heartbeats. Done at
    01:47-01:56 UTC on 2026-10-25 (`python/tools/chain_next_batch.py` waits for the running batch's `merged.json`; its second step used the same run name and
    was refused, so the correcting takeover `o2-s1more3-2` was sent by hand). The stored settings now name `o2-s1more3-2`: never recycle that machine
    again while a run is going. Two more things the same night: (1) `o2-s1more3-1` started and stopped after 16 s, because cancelling its first launcher had left
    `runs/o2-s1more3-1/STOP` in R2 and the run reads that; a run name whose launcher was cancelled is spoiled, so the next takeover used a new name
    (`o2-s1more3-3`); (2) that takeover's recycle used its OWN update (no lag), so the container ran `o2-s1more3-3`, not the name stored before it: the lag is
    not reliable, and the only certain way to know which run a container has is the heartbeat name in R2.
  - **A directly rented instance (PUT `/api/v0/asks/<offer>/`, `grab_machines.direct_rent`) took two takeovers** (the first left vast's base image),
    where the console-rented 57404109 took one. The cleanup job destroys any instance labelled `41bots-<run>` whose run never heartbeats within 60 minutes
    of the run's manifest upload; a label with a space after `41bots` (`41bots - ...`) is never touched by it.
- **Cost.** About $3 of rentals went to failed or abandoned launches of `o2-audit-4` to `-9`; the Xeon Phi box billed $1.14/h until it was destroyed.
  The nine S1 launches cost under $1 in rentals; the owner re-rented the Thailand box three times.
- **Limits.** Counts are from one session; the causes of the two takeover failures of 2026-10-22/23 are read from vast's status records and the launcher
  log, not from inside the machines. For the nine S1 launches the cause is read from the `FAILED` marker and the commit dates; the container logs of the
  destroyed machines were empty.
- **Sources.** `python/launcher/launch.py`, `python/launcher/test_keep.py`, `python/tools/grab_machines.py`, `.github/workflows/vast-run.yml`, GitHub
  runs 39925013210, 39926028226, 39975581166, 39979256595; R2 `runs/o2-audit-*/KEPT.json`, `runs/o3-pair-*/`; `CLAUDE.md` (machine rules).

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
  (`o3-sibrank-a/-b`) was never launched**; no golden-set result existed (it was run on 2026-10-23, D12).
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
- **Observed.** At equal volume plain labels tie exact ones; at equal money (6M plain about $0.4 against 1.5M exact
  about $12) plain wins clearly; volume has not flattened (SD 3.67 to 3.13 for 4x data; extrapolated 40M about
  2.5%, not measured). Every network's own team bias (0.1-0.2%) is far above the plain labels' 0.013-0.017%.
  O_2's best SD 3.13% is far from O_3's 1.85%.
- **Label check** (`g2-labelcheck-1`, EPYC 7663 215 threads, $0.51): the same 10,000 positions labelled three ways.
  RMS error against exact: plain 2.145% (largest bias 0.0155%, 16% of rows identical to exact), G2 0.479% (bias
  0.0063%, 55% identical). Speed measured on 215 threads: exact 341.6 s (7.3 thread-s a label), G2 172.4 s (3.7),
  plain 9.7 s (0.21). Wider-machine speeds: plain 1,607 labels/s on the 7995WX (40M plain labels about 7 h, about
  $9, not the $2.7 once estimated); G2 about 50 labels/s on an EPYC 7B12 (245 threads).
- **G2 arm** (0.75M G2 labels, same recipe): `o2-g2arm-1` (EPYC 7663, $1.60/h) made part 1 (120k random + 30k rules,
  seeds 561/562) and was stopped; `o2-g2arm-2` (EPYC 7B12 taken over by hand, instance 57260975, $0.73/h) made 4 of the
  other 5 parts (seeds 563/564); a training-only run on all stored parts plus s601 gave the network. **Result** (`o2-g2arm-1` + `o2-g2arm-2`, `o2-g2full-1`; the label run `-2` died on a sidecar race after 4 of 5 parts, fixed
  in the trainer, `test_hybrid_train`): the network was trained on the 420,000 stored G2 labels (360k random + 60k rules-v1;
  `docs/m3-hybrid/o2-stagea-netcmp-g75.json`, same test set as above, 16 min, $0.23 on an EPYC 7B12 + RTX 3090):

| arm | labels | SD | paired change in squared error vs `p60` | largest team bias | sibling regret |
|---|---|---|---|---|---|
| `e03` | exact 0.3M | 4.303 | +8.71 +-0.37 | 0.412 | 0.324 |
| `g75` (partial) | **G2 0.42M** | **4.152** | +7.43 +-0.37 | 0.161 | 0.388 |
| `e075` | exact 0.75M | 3.931 | +5.65 +-0.33 | 0.180 | 0.327 |
| `p15` | plain 1.5M | 3.668 | +3.64 +-0.29 | 0.116 | 0.295 |
| `p60` | plain 6M | 3.132 | reference | 0.126 | 0.247 |

  G2 0.42M sits exactly where exact labels of the same count would (log-interpolating `e03` and `e075` gives 4.16): G2
  labels are as good as exact ones per label, not better, at about half the cost of exact (3.7 against 7.3 thread-s) and 18 times
  the cost of plain. At equal money plain volume won by a wide margin (6M plain about $0.4 of labels against 0.42M G2 about
  $3). Decision (owner, 2026-10-22): no niche for G2 labels as a bulk source (the 2026-10-20 note stands).
- **Limits.** One seed per arm; the test set is random-play positions, not the positions trick-1 search reaches;
  no trick-1 decision-level number exists, so O_2 can only be audit-clean on exact trick-2 positions.
- **Tools made.** `fortyone_nn.netcmp`, `python/tools/o2watch.py` (progress, speed, ETA from R2 heartbeats),
  `vast-run.yml` input `instance_id` (take over a machine rented by hand; the instance needs the run's disk, 80 GB).

### D7. O_2 on 11M plain labels (Stage B volume, runs `o2-vol-3` (labels), `o2-volnet-5` (network), 2026-10-23)
- **Question.** Does plain-label volume keep helping past 6M, and by how much? (D6: plain 6M beat exact 1.5M.)
- **Method.** Labels: `o2-vol-3`, CPU-only on an AMD EPYC 7742 (245 threads, $0.775/h), ten stored streams of 2.2M plain hybrid labels
  (1.6M random + 0.4M rules-v1 + 0.2M harvested positions, harvest driven by the provisional O_2; smoke step first), seeds
  5711-5760; about 32 min and $0.41 per stream (1,170 labels/s). Network: the first five streams (11M rows: 8M random, 2M rules-v1,
  1M harvest) plus the exact s601 validation shard, 2048-1024-256, outcome head, seed 1, early stopping (best epoch 7 of 14):
  `o2-volnet-5` on an EPYC 7B12 + RTX 3090, 18 min, $0.19 (`python/tools/o2_train_manifest.py`).
- **Result** (same 250k exact test set and 4,479 sibling groups as D6; `docs/m3-hybrid/o2-stageb-netcmp-v11.json`):

| arm | labels | validation loss / accuracy (s601) | test SD | paired change in squared error vs `p60` | largest team bias | sibling regret (95%) |
|---|---|---|---|---|---|---|
| `p15` | plain 1.5M | 2.069 / 67.1% | 3.668 | +3.64 +-0.29 | 0.116 | 0.295 +-0.055 |
| `p60` | plain 6M | not read | 3.132 | reference | 0.126 | 0.247 +-0.051 |
| `v11` | plain 11M (5 streams) | 1.861 / 74.7% | **2.932** | **-1.21 +-0.25** | **0.054** | **0.199 +-0.038** |

- **Observed.** More plain volume still helps: SD 3.132 to 2.932 for 1.83x the data (exponent about -0.11 on rows, flatter than the
  -0.12 read off 1.5M to 6M); bias and sibling regret improved too. Extrapolated at this exponent: 22M about 2.71%, 40M about
  2.55% (not measured); reaching O_3's 1.85% would need about 700M rows at this exponent (extrapolated).
- **Cost.** Labels $0.41 per 2.2M rows; the 11M network $0.19.
- **Limits.** One seed per arm; random-play test positions; sibling-regret differences inside the noise except v11 against p15.

### D8. O_2 on 19.8M plain labels (run `o2-volnet-9k`, 2026-10-23)
- **Question.** Does the volume curve keep going with nine of the ten D7 label streams (19.8M rows)?
- **Method.** As D7 (2048-1024-256, outcome head, seed 1, early stopping on the exact s601 shard) on an AMD EPYC 7742 (245 threads, RTX 3070,
  $0.775/h): 15 epochs, best epoch 8; 46 min, $0.60 (machine kept 8 h). Scored on the same 250k exact test set and 4,479 sibling groups
  (`docs/m3-hybrid/o2-stageb-netcmp-v20.json`). Network: `runs/o2-volnet-9k/out/volnet/`.
- **Result.**

| arm | rows | validation loss / accuracy (s601) | test SD | paired change in squared error vs `v11` | largest team bias | sibling regret (95%) |
|---|---|---|---|---|---|---|
| `v11` | 11M | 1.861 / 74.7% | 2.932 | reference | 0.054 | 0.199 +-0.038 |
| `v20` | 19.8M | 1.795 / 77.2% | **2.748** | **-1.04 +-0.21** | **0.028** | **0.176 +-0.033** |

- **Observed.** SD exponent about -0.11 on rows for 1.8x the data (predicted 2.71%, measured 2.75%); sibling regret falls more slowly
  (exponent about -0.20, 0.199% to 0.176%, inside the noise). At these exponents sibling regret 0.11% would need about 10x the rows
  (extrapolated; about 200M rows, 380 GB of shards).
- **Cost.** Network $0.60 (labels: D7).
- **Limits.** One seed; random-play test positions; the sibling-regret change is inside the noise.
- **Sources.** R2 `runs/o2-volnet-9k/`; `docs/vast/o2-volnet-9k.manifest.json`; `docs/m3-hybrid/o2-stageb-netcmp-v20.json`.

### D9. O_2 capacity and seed noise on the 11M plain labels (run `o2-volcap-1`, 2026-10-23)
- **Question.** Does a deeper or wider network help O_2 at fixed data, and how large is run-to-run noise?
- **Method.** The 11M plain labels of D7 (first five streams of `o2-vol-3`) and s601 validation; outcome head, early stopping. Arms: seed 2 of
  2048-1024-256; deeper 2048-2048-1024-256; wider 4096-2048-1024-256. One RTX 3060 on an AMD EPYC 7B12 (256 threads, $0.328/h), about 5 h,
  $1.5. Scored on the same 250k exact test set and 4,479 sibling groups as D6-D8 (`docs/m3-hybrid/o2-capacity-netcmp.json`,
  `o2-capacity-wide-netcmp.json`).
- **Result.**

| arm | best epoch | validation loss / accuracy | test SD | paired change in squared error vs `v11` | largest team bias | sibling regret (95%) |
|---|---|---|---|---|---|---|
| `v11` (2048-1024-256, seed 1) | 7 | 1.861 / 74.7% | 2.932 | reference | 0.054 | 0.199 +-0.038 |
| seed 2 | 5 | 1.857 / 74.5% | 2.950 | +0.11 +-0.22 | 0.055 | 0.190 |
| deeper | 8 | 1.829 / 76.7% | 2.791 | -0.80 +-0.23 | 0.084 | 0.202 |
| wider | 8 | 1.845 / 76.2% | 2.830 | -0.59 +-0.22 | **0.018** | 0.219 |

- **Observed.** Seed noise is about 0.02 in SD (seed 2 against seed 1 is inside the paired bound). Depth and width lower the value error
  (deeper about as much as 1.6x the data would), while sibling regret does not improve (0.202 and 0.219 against 0.199; 95% half-width about
  0.035) and the deeper network's bias is larger; the wider network has the smallest bias. As for O_3 (C2, C12) and O_4 (B4), capacity did
  not move decision-level numbers.
- **Cost.** $1.5.
- **Limits.** One seed per shape; 11M rows only (capacity at 40M rows was not tested); random-play test positions.
- **Sources.** R2 `runs/o2-volcap-1/out/{seed2,deep,wide}/`; `docs/vast/o2-volcap.manifest.json`.

### D10. Calibration: what a network's error costs in decisions (run `o3-ladder-1`, 2026-10-23)
- **Question.** No trick-1 golden set exists (owner decision, 2026-10-16). What trick-1 decision loss does a given O_2 error imply, read from the
  same relation one trick later, where exact answers exist?
- **Method.** Four O_3-recipe networks (2048-1024-256, outcome head, seed 1, exact trick-3 labels) trained on 1M, 3M, 10M and 20M rows (0.025,
  0.075, 0.25 and 0.5 of the stored 40M base). Each judged two ways: (a) the trick-2 golden check (2,000 decisions x 320 worlds, 1% re-solved;
  H_3 with that network at the leaves against exact) for decision loss; (b) position SD and sibling regret on 100,000 exact trick-3 sibling
  positions (`sib3`, seed 7903, driven by O_3 f5766a). Log-log lines through the four points. AMD Threadripper PRO 9995WX, 96 min, $2.36
  (`docs/vast/o3-ladder.manifest.json`).
- **Result.**

| network (rows) | position SD | sibling regret | golden decision loss (95% upper) | largest team bias (golden) |
|---|---|---|---|---|
| lad1 (1M) | 3.02% | 0.190% | 0.0293% (0.0491) | 0.033% |
| lad3 (3M) | 2.65% | 0.154% | 0.0143% (0.0179) | 0.032% |
| lad10 (10M) | 2.28% | 0.111% | 0.0089% (0.0110) | 0.025% |
| lad20 (20M) | 1.93% | 0.085% | 0.0071% (0.0085) | 0.016% |

  Decision loss grows as sibling regret to the power 1.7 (residual SD 0.22 in log) and as position SD to the power 3.1-3.2 (residual 0.15-0.22).
  Applying the fits to O_2 networks (trick-2 position SD and sibling regret on the 250k exact test set, D6-D11): `v11` (11M) 0.022-0.026%;
  `v20` (19.8M) 0.019-0.022%; round 1 of the audit loop (39.6M, D11) 0.016% by SD and 0.024% by sibling regret; round 2 0.016% and 0.020%;
  each interval a factor of 1.5 either way. The fits reach 0.01% at sibling regret about 0.11% or position SD about 2.25-2.3%.
- **Observed.** The 0.01% standard failed for every ladder network on bias (largest 0.016-0.033%) even where decision loss was close.
- **Limits.** An estimate, not a measurement: four points, one seed each; the relation is read across two tricks (trick-2 decisions with trick-3
  leaves, applied to trick-1 decisions with trick-2 leaves; trick 1 has more cards and larger spreads, so the true loss may be above or below);
  the golden losses carry their own noise (the upper bounds above); the sibling positions are random-play positions.
- **Sources.** `docs/m3-hybrid/o3-ladder/lad*-hc-t2-o3.txt`, `docs/m3-hybrid/o3-ladder-calibration.json`; R2 `runs/o3-ladder-1/`.

### D11. The O_2 audit loop at trick 2 (run `o2-audit-10`, 2026-10-23)
- **Question.** Does the audit loop (find weak groups on fresh exact positions, harvest more of them at double weight, retrain from scratch) improve
  O_2 beyond plain volume?
- **Method.** `docs/vast/o2-audit-4.manifest.json` (built by `python/tools/o2_audit_manifest.py`): the 39.6M stored plain hybrid labels (`o2-vol-3`
  streams 1-10, `o2-vol40-1` streams 11-18) plus the exact s601 validation shard; 2048-1024-256, outcome head; audit shards of 200,000 exact trick-2
  positions as the hybrid reaches them from trick-1 decisions (seeds 22101, 22102); targeted harvest of 2,000,000 rows a round at weight 2 (plain O_3
  labels, 10% exact); 3 rounds set. New York machine (AMD EPYC 9B14, 184 usable threads, RTX 5090, 128 GB RAM, $1.41/h), 5.7 h, about $8. Stopped by
  the owner (2026-10-23) after round 2's audit report, before its harvest. Earlier launches (`o2-audit-4` to `-9`) failed or were abandoned: a host
  downloading at 6 Mbps, boxes destroyed before the takeover, a box whose GPU could not start containers (CDI error); about $3 of rentals.
- **Result.** Round 1 network `13152c24bc27` (best epoch 9; validation loss 1.754, accuracy 78.4%); round 2 network `235ed448acc7` (best epoch 20;
  1.767, 78.3%). On the 250k exact test set and 4,479 sibling groups:

| | round 1 | round 2 | `v20` (19.8M, D8) |
|---|---|---|---|
| test SD | 2.603% | 2.624% (paired change +0.11 +-0.14) | 2.748% |
| largest team bias | 0.019% | 0.022% | 0.028% |
| sibling regret (95%) | 0.188% +-0.044 | 0.171% +-0.036 | 0.176% +-0.033 |

  The audits on fresh exact shards: round 1 overall SD 2.531%, 668 weak groups, worst "the leader's team has 40 points" (weight 0.19%, SD 7.05%,
  bias -0.88%); round 2 overall SD 2.524%, 667 weak groups, worst the same group (weight 0.21%, SD 6.84%); race class 4 SD 4.07% then 4.08%. The auditor's
  verdict was "not flat" both times. Round 1's rough test on the saved exact trick-1 worlds (8 decisions): mean loss 0.16% (95% upper 0.48%).
- **Observed.** The 39.6M network beat the 19.8M one on error (2.60% against 2.75%) and bias; sibling regret did not change beyond the noise. The
  targeted round changed nothing measurable: the same score-state groups (30, 31, 36, 40 points) stayed at SD 4-7%. As for O_3 (C2, C3, C12), more
  audit rounds did not move the network beyond retraining noise. Decision (owner, 2026-10-23): round 1's network (integer `13152c24`) is the O_2 for
  the trick-1 work.
- **Cost.** Loop about $8; stored base labels $6.1 (`o2-vol-3`) and $2.9 (`o2-vol40-1`).
- **Limits.** One seed per round; round 3 and round 2's harvest were not run; round 2's own audit shard was not stored; the rough test is 8 decisions.
- **Sources.** R2 `runs/o2-audit-10/partial/rung/000000/out/hybrid/` (networks, `audit/round-N/`), round-1 audit shard `inputs/sha256/c8a8a392...`,
  targeted shard `inputs/sha256/6fceb941...`; `docs/m3-hybrid/o2-audit-10/` (auditor reports).

### D12. A sibling ranking loss on O_3 (run `o3-pair-1`, 2026-10-23)
- **Question.** Does a pairwise ranking loss over sibling positions move O_3's decisions, as S2's decision-aware loss did for the student (E12)?
- **Method.** `docs/vast/o3-pair.manifest.json`, judged by `python/tools/o3_pair_score.py`: the O_3 recipe (2048-1024-256, outcome head, cut 3) on the
  stored 40M trick-3 base plus 6M exact sibling rows (seed 7911, driven by O_3 f5766a), identical in every arm; arms differ in `pair_weight` (the
  trainer's gap-weighted RankNet term, scale 100, Fix E, `docs/m3-hybrid/sibrank.md`) and seed. Judged on the 100,000 held-out exact sibling
  positions `sib3` and on the in-run golden check (2,000 trick-2 decisions x 320 worlds). AMD Threadripper PRO 9995WX (184 usable threads), RTX
  3090, $1.54/h; two of five arms finished (`base_s1`, `w1_s1`); the others were stopped by the owner (2026-10-23) after the result.
- **Result.**

| arm | sibling regret (95%) | sibling SD | largest bias | golden loss (95% bound) | epochs run (best) | validation loss / outcome accuracy |
|---|---|---|---|---|---|---|
| `base_s1` (weight 0) | 0.0720% +-0.0094 | 1.787% | 0.0242% | 0.0066% (0.0081%) | 17 (13) | 1.456 / 88.6% |
| `w1_s1` (weight 1) | 0.1274% +-0.0128 | 2.276% | 0.0067% | 0.0089% (0.0108%) | 40 (39, at the cap) | 1.653 / 81.2% |
| adopted O_3 f5766a (D10 / C2) | not scored on this set | | | 0.0056% (0.0068%) | | |

- **Observed.** Weight 1 had sibling regret 77% above the control (outside the noise) and a higher golden loss (inside the 0.0025-point retraining
  noise); only the team bias improved. It ran to the 40-epoch cap with its best epoch at 39, and its validation loss and outcome accuracy were worse than the control's (1.653 and 81.2% against 1.456 and 88.6%, 46,000,000 rows each); the trainer stops on value validation, not
  on decisions (E12: the same ranking term scored 0.095% without and 0.059% with a stop on decision loss). The control reproduced f5766a within
  retraining noise. Cost of the experiment: about $6.
- **Limits.** One seed per arm (`base_s2`, `w1_s2` and `w3_s1` were not run); one weight; the sibling set is built with the same rule as the training
  sibling rows (principal lines driven by f5766a); a smaller weight and a stop rule on sibling regret were not tried.
- **Sources.** R2 `runs/o3-pair-1/out/base_s1/`, `out/w1_s1/`; `docs/m3-hybrid/o3-pair-score.json`; `docs/m3-hybrid/sibrank.md` section 6.

### D13. Trick-1 teacher data with the adopted O_2 (run `o2-copper-s1-1`, 2026-10-23/24)
- **Question.** What does the trick-1 teacher cost on a cheap machine with the adopted O_2, and what do the copper set and the first S1 training
  labels look like?
- **Method.** `docs/vast/o2-copper-s1.manifest.json`: `student_teacher` chunks of 500 trick-1 decisions, teacher H_2 with O_2 round 1 of D11 (integer
  `13152c24`, `inputs/sha256/13152c24...`); copper = 4 chunks at 128 deals (seed 7801, rules-v1 positions); S1 labels = 300 chunks of rules-v1
  positions (seed 7811) and 200 chunks of mixed positions (seed 7812), 64 deals each, 250,000 decisions. AMD EPYC 7B12 (245 usable threads), RTX 3060,
  $0.328/h, taken over by hand (instance 57404109; takeover 3 min). The S1 test set (`docs/vast/o2-s1test.manifest.json`, run `o2-s1test-2`): 4 chunks of
  500 decisions at 128 deals, role test (the trainer's test seed 32343; copper has role train and cannot serve), rules-v1 positions, same teacher, on an
  AMD EPYC 7K62 (192 usable threads), $1.05/h.
- **Result.** All 504 chunks finished (copper 4, `s1rules` 300, `s1mixed` 200); the run's elapsed time was 54,385 s (15.1 h), about $4.95 at $0.328/h
  (the launcher's report shows only its first 331 minutes, $1.81, before it handed the machine over). Chunks of 105 s median for 500 decisions at 64
  deals; copper chunks 107-124 s. Rules-v1 chunks: 65 s of compute for 500 decisions (about 29,700 decisions an hour) plus about 40 s of setup
  (`setup: table v0`, the equity table is rebuilt in every chunk): 16,800-17,100 decisions an hour, **$19.5 per million decisions** on this machine.
  The earlier measurement (E6) was 12,489 decisions an hour at 64 deals on an EPYC 7B13 ($1.05/h, about $84 per million). The test set took 20.6
  minutes and $0.36 (the first attempt, `o2-s1test-1`, was stopped by the low-CPU watchdog after one chunk, 15.5 minutes, $0.27).
- **Second batch (2026-10-24/25, `docs/vast/o2-s1more.manifest.json`, run name `o2-s1more-3`).** 250,000 more trick-1 decisions at 64 deals (seeds 7821
  and 7822, 150 + 100 chunks of 1,000 decisions, same teacher), on the Thailand AMD EPYC 7B12 (245 usable threads, RTX 3060, $0.350/h as billed, rented
  directly through the API): 250 chunks, median 158.6 s a chunk (22,700 decisions an hour with the larger chunk), 40,035 s on the run's own clock, about
  $3.89, **$15.6 per million decisions**; chunks under R2 `runs/o2-s1more-3/out/{s1rules2,s1mixed2}/`, merged report `runs/o2-s1more-3/merged.json`.
  With the first batch S1 now has 500,000 labelled decisions. A third batch (seeds 7831 / 7832, run name `o2-s1more3-3`) started on the same machine
  at 02:01 UTC on 2026-10-25 (the names `o2-s1more3-1` and `-2` did not run it: see A10).
- **Limits.** One machine type; the rate holds for rules-v1 and mixed positions at 64 deals without racing or the restricted solve; the cost per
  million decisions assumes the same machine is available.
- **Sources.** R2 `runs/o2-copper-s1-1/out/{copper,s1rules,s1mixed}/`, `runs/o2-s1test-2/out/test/`; E6.

### D14. S1, the trick-1 student: plain against decision-aware loss, and the first learning-curve points (run `student-t1-250k-10`, 2026-10-24)
- **Question.** Does the decision-aware loss of E12 (`rank_weight` 1, `rank_temperature` 0.01, stop on the validation decision loss) lower a trick-1
  student's loss against its teacher as it did at trick 2, and how does that loss fall with the number of decisions and of teacher deals?
- **Method.** `docs/vast/student-t1-250k.manifest.json`: `student_train` on the 250,000 stored decisions of D13 (300 `s1rules` + 200 `s1mixed` parts,
  spread order), students 512,256 (`student-v1` inputs), batch 256, lr 0.001, patience 4, 40 epochs at most, 5% validation split. Nine networks: plain
  loss and decision-aware loss, seeds 1-3, at 250,000 decisions and 64 deals; the decision-aware loss at seed 1 with 50,000 and 100,000 decisions
  (64 deals) and with 250,000 decisions cut to 32 deals. Judged by `search student-check` on the 2,000-decision test set of D13 (128 deals, role test),
  beside the rules-v1 and random baselines. Nevada machine, 4x RTX 5070 Ti (one used), AMD EPYC 7K62 (192 usable threads), $1.34/h, rented by offer
  id, not kept; binaries `dd19e4d66cef` (2026-10-23).
- **Result.** Loss = the teacher's best mean value less the value of the player's card, % of the deciding pair's money chance (95% upper bound in
  brackets; agree = % of decisions on the teacher's best card):

| player | decisions | deals | seeds | loss % | agree % |
|---|---|---|---|---|---|
| random | | | | 0.770 [0.876] | 30.9 |
| rules-v1 | | | | 0.343 [0.392] | 47.7 |
| S1 plain loss | 250,000 | 64 | 1 / 2 / 3 | 0.0735 / 0.0876 / 0.0947, mean 0.0853 (SD 0.011) | 60.2 / 58.5 / 58.7 |
| **S1 decision-aware loss** | 250,000 | 64 | 1 / 2 / 3 | **0.0469 / 0.0445 / 0.0503, mean 0.0472 (SD 0.003)** | 65.5 / 64.6 / 64.8 |
| S1 decision-aware loss | 100,000 | 64 | 1 | 0.0679 [0.0808] | 61.7 |
| S1 decision-aware loss | 50,000 | 64 | 1 | 0.0824 [0.1029] | 60.1 |
| S1 decision-aware loss | 250,000 | 32 | 1 | 0.0512 [0.0607] | 63.3 |

  Every decision-aware seed is below every plain seed; the mean is 45% lower. By place in the trick the decision-aware leader-seat loss is 0.063 / 0.057 /
  0.067 against 0.107 / 0.112 / 0.135 plain; in score class S4, 0.094 / 0.082 / 0.097 against 0.110 / 0.113 / 0.178. The decision-aware loss at seed 1
  falls from 0.0824 (50,000) to 0.0679 (100,000) to 0.0469 (250,000): a power-law slope of about 0.35-0.40 in the number of decisions, not flattened.
  Halving the deals (64 to 32) raised it from 0.0469 to 0.0512. Integer networks (content-addressed in R2, `runs/student-t1-250k-10/out/s1/`):
  decision-aware seeds 1 `6ec1b9cc38b4...`, 2 `97df2a54e1c3...`, 3 `b20445c31bde...` (full keys in the run's `done.json` files); plain seeds 1
  `70812487304221...`, 2 `b591b5d90571...`, 3 `bf8f5082049e...`; 100,000-decision `8e01a3e74fe9...`, 50,000-decision `31643f7f6b49...`, 32-deal `0e6e233dfe5b...`.
- **Copper check (owner, 2026-10-24: "test against the copper set"; the bronze set never finished, one complete decision).** The nine integer networks on
  the stored copper set (2,000 trick-1 decisions, 128 deals, seed 7801, role train, never in any training input), `search student-check --boot 1000` on the
  build VM (0.7 s a network; `docs/m3-hybrid/s1-copper-check.json`). Loss % (95% upper bound), agreement with the teacher:

| player | decisions | seeds | loss % | agree % |
|---|---|---|---|---|
| random | | | 0.762 [0.874] | 31.9 |
| rules-v1 | | | 0.371 [0.454] | 48.7 |
| S1 plain loss | 250,000 | 1 / 2 / 3 | 0.0784 / 0.0805 / 0.0709, mean 0.0766 | 60.8 / 60.4 / 60.6 |
| **S1 decision-aware loss** | 250,000 | 1 / 2 / 3 | **0.0404 [0.0461] / 0.0386 [0.0442] / 0.0457 [0.0524], mean 0.0416** | 66.1 / 65.9 / 65.3 |
| S1 decision-aware loss | 100,000 | 1 | 0.0564 [0.0653] | 62.6 |
| S1 decision-aware loss | 50,000 | 1 | 0.0689 [0.0790] | 60.4 |
| S1 decision-aware loss | 250,000, 32 deals | 1 | 0.0472 [0.0549] | 65.8 |

  The ordering is the same as on the test set above: the decision-aware mean is 46% below the plain mean, each decision-aware seed below each plain seed;
  the leader-seat loss of the decision-aware seeds is 0.082 / 0.065 / 0.074 against 0.127 / 0.106 / 0.113 plain. The slope in the number of decisions is
  0.0689 (50,000), 0.0564 (100,000), 0.0404 (250,000). Copper's teacher is the same H_2 + O_2 at the same 128 deals as the test set: copper measures how
  well S1 copies the teacher, and says nothing about the teacher's own error against exact play (no trick-1 golden set exists).
- **Ship preparation (2026-10-24).** Candidate: the decision-aware seed 2 network `97df2a54e1c368a13d69ffd887de62f45c6c3f244605450579b803f31f00e134`
  (7,349,312 bytes, `inputs/sha256/<key>` in R2): the lowest loss on both sets (copper 0.0386%, test set 0.0445%); seeds 1 and 3 are within noise of it. The
  engine had one student slot for tricks 1 and 2; it now has one per trick (`load_student_trick`, `loadStudent(bytes, sha, trick)`; a default student still
  serves a trick without its own; test `per_trick_students_serve_their_trick`, 11 + 16 wasm tests pass, clippy clean). Export check
  (`tools/wasm_check.sh` with `WASM_STUDENT1` = this S1 and `WASM_STUDENT2` = S2 `3d1189e3...`, build VM, with the equity table and O_3): the 21 requests of the
  API check, including the trick-1 play and rob requests valued by S1, and 250 random matches give the same bytes natively and in WebAssembly, and again for
  2, 3, 4 and 7 shards; PASS. The same check with S1 alone as the default student also passed. New module `crates/wasm/dist/engine.wasm`
  `8366f7d2...` (913,674 bytes, built from 6e8c5f6a2afc) and `engine.mjs` `d0d8ab56...`; the app session's steps are in STATUS and `docs/WASM-API.md` section 7.
- **Cost.** 84.2 minutes, $1.87: 20 minutes reading the 504 part bundles one at a time (about 2 s each), 30 minutes encoding 4 parts at a time
  (`fetch_jobs` default 4 of 192 threads; 32 is allowed), 2.5-3.4 minutes training each network. Both steps are fixed afterwards: the inputs step reads
  `fetch_jobs` at a time (commit `f08888c`) and the manifest sets `fetch_jobs` 32.
- **Limits.** The loss is measured against the teacher's own values (H_2 + O_2, 128 deals on 2,000 decisions), not against exact truth: no trick-1
  golden set exists, and the bronze set was not used; a student better than its teacher scores worse on it. One seed for the three slope points; the
  slope is from one seed per size; no 500,000 or 1,000,000-decision point has been trained. The 32-deal difference (0.0043) is about the size of the
  decision-aware seed spread of the 250,000 points together (SD 0.003).
- **Sources.** `docs/vast/student-t1-250k.manifest.json`, R2 `runs/student-t1-250k-10/out/s1/` (`student/curve.txt` and the per-network `done.json`),
  E12, D13. Nine earlier launches of this run failed for a reason unrelated to training: A10.

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

### E9. Shortlist plus search at the app's deal counts (`search student-check --sub-deals`, 2026-10-22, build VM)
- **Question.** At the 10-100 deals [worlds searched per decision] a phone can afford, does "the
  student names its top 2 or 3 cards and a short search picks among them" lose less money chance than
  the student alone?
- **Method.** The trick-2 student S2 (954223e2…) and O_3 (f5766aa9…) on the exact trick-2 golden set
  (2,000 decisions x 320 exact worlds). For each decision, 8 repeats (seed 1) of a seeded shuffle of its
  320 worlds; the first N form the sample (N = 10, 25, 50, 100; the samples are nested, so one shuffle
  serves every N). A method picks a card from the sample and is **scored on the other 320-N worlds**,
  so a lucky sample cannot flatter its own pick. Methods: the student alone; search over all cards;
  top-2 and top-3 of the student searched. Losses are put on the golden check's scale: the student's
  loss on all 320 worlds less the method's gain over the student on the scoring worlds (the student
  alone then reads 0.0733%, the same as the plain golden check; the uncorrected figures are in the
  run's JSON). Bounds: bootstrap 95% over decisions (1,000 replicates, seed 1). Code
  `crates/search/src/shortlist.rs` (4 unit tests), run as `search student-check ... --sub-deals
  10,25,50,100 --sub-repeats 8 --sub-seed 1`.
- **Result** (loss in % of money chance; the student alone is 0.0733% at every N):

| deals | search all cards | top-2 + search | top-3 + search |
|---|---|---|---|
| 10 | 0.1164 | 0.0836 | 0.1030 |
| 25 | 0.0619 | 0.0542 | 0.0596 |
| 50 | 0.0384 | 0.0422 | 0.0402 |
| 100 | 0.0254 | 0.0338 | 0.0293 |

  Gain of top-3 + search over the student alone (positive = better): -0.0297 [-0.0427, -0.0175] at 10
  deals, +0.0137 [+0.0018, +0.0259] at 25, +0.0331 [+0.0210, +0.0459] at 50, +0.0441 [+0.0319, +0.0570]
  at 100. Seat 1 (the leader), student 0.179: top-3 + search 0.146 / 0.097 / 0.073 / 0.061 at 10 / 25 /
  50 / 100 deals. Stratum S4, student 0.120: 0.171 / 0.100 / 0.072 / 0.055. Top-2 against searching all
  cards (positive = top-2 better): +0.0328 [+0.0256, +0.0400] at 10, +0.0077 [+0.0013, +0.0138] at 25,
  -0.0038 [-0.0103, +0.0023] at 50, -0.0084 [-0.0149, -0.0027] at 100. Top-3 against all cards:
  +0.0134 [+0.0077, +0.0186] at 10, then +0.0023, -0.0018, -0.0039, each with its bound across zero.
- **Cost.** No rental; the run takes 6 s on the build VM.
- **Limits.** The values are exact world values, so no leaf-network error is included: any real search
  does worse. Trick-2 decisions have only 2-4 candidates (493 with 2, 421 with 3, 1,086 with 4, mean
  3.3), so top-3 differs from "all cards" only in the 4-candidate decisions, by one card (E11). The
  sample is drawn from 320 worlds without replacement (31% of them at N = 100), which lowers its
  variance by about 17% against independent draws. The truth proxy, a 320-world mean, is itself
  noisy (a 512-deal teacher's floor is 0.008%, E4). Trick 2 only. WebAssembly cost per deal is
  separate: the leader's trick-2 deal is 368 ms on the build VM (`WASM-API.md` §8), about 19 deals in 7 s
  on one thread, and a phone is expected to be 2-4 times slower.
- **Sources.** `docs/m3-hybrid/path-forward.md` section 3 (step 0a).

### E10. How concentrated is S2's loss, and does a router find the risky decisions? (same run as E9, `tools/router_fit.py`, 2026-10-22)
- **Question.** If S2's loss sits in a few decisions, a search could be spent only on those. Is the
  loss concentrated, and can a rule or a model that sees only what the student shows find them?
- **Method.** The 2,000 golden decisions of E9. Concentration: the share of the total loss held by the
  worst 1, 5, 10, 20, 40% of decisions. Router: decisions ranked by a score; the top f% by risk get
  top-3 + N deals (E9's corrected losses), the rest keep the student's card. Scores: (a) the student's
  margin [its best value less its second-best, for the deciding team]; (b) a gradient-boosted model
  [an ensemble of small decision trees] on place, stratum, candidate count, the student's top values,
  margins and spread, scored by 5-fold cross-validation over 5 seeds (every decision is predicted by a
  model that never saw it); (c) an oracle that knows the true loss (the upper bound). Bounds:
  bootstrap over decisions, 1,000 replicates.
- **Result.** 547 of 2,000 decisions have any loss (1,453, or 72.7%, lose nothing). The worst 1% of
  decisions hold 38.8% of S2's loss, the worst 5% hold 75.1%, 10% hold 90.9%, 20% hold 99.1%, 40% hold
  all of it. The margin does not find them: its correlation with the loss is 0.01 and the 10% of
  decisions with the lowest margin hold 5.4% of the loss (a low margin is a near-tie, which costs little
  either way). The model's correlation is 0.048; its predicted-riskiest 10% / 20% / 50% hold 23.9% /
  40.4% / 72.1% of the loss. Loss in % of money chance with f% of decisions searched at 100 deals
  (student alone 0.0733, all searched 0.0293):

| searched | margin router | model router (95% range) | oracle |
|---|---|---|---|
| 10% | 0.0711 | 0.0613 (0.052-0.071) | 0.0193 |
| 30% | 0.0671 | 0.0504 (0.044-0.063) | 0.0186 |
| 50% | 0.0588 | 0.0423 (0.035-0.051) | 0.0208 |

  At 100 deals the model router gets 52% of the gain for 30% of the searches and 70% for 50%; at 25
  deals 71% for 30% (0.0636 against 0.0596 for all). The oracle at 5% / 10% / 20% searched reads
  0.0267 / 0.0193 / 0.0174, below searching everything (0.0293), because a search cannot hurt where the
  student is already right.
- **Cost.** No rental; seconds on the build VM.
- **Limits.** One golden set of 2,000 decisions, and the model was fitted and scored by
  cross-validation on that same set; trick 2 only; the cost of the router's own network call is not
  counted. The router is the assistant's idea, not a published recipe (PLAN §9: duplicate tournaments
  would decide between it and a proven option). The margin router was not built into any bot.
- **Sources.** `tools/router_fit.py`; the per-decision records are in the E9 run's JSON
  (`sub_deals.per_decision`, written with `--json`).

### E11. The root-restricted solve: exactness and cost (2026-10-22, build VM)
- **Question.** E8 found that cutting the candidate list changes nothing, because one solve values every
  root card. Once the solver searches only chosen root cards, does it cost less, and how much?
- **Built.** `FastOptions::root_keep` in the solver (an earlier commit, tested) is now passed by
  `search::pimc::exact_world` from the candidates' cards (all legal cards = the full solve, burns pass
  none). The app API takes an optional `shortlist` of plays (`WASM-API.md` §3): each listed card keeps
  the full search's sums byte for byte, the search key includes the shortlist, shards of one shortlist
  merge to the whole, and bad lists are refused (test `a_shortlist_values_its_cards_exactly_as_the_full_search_does`).
- **Method 1 (first timing).** `tools/shortlist_time.py` on the API's bench positions (`engine-native
  bench`, 16 deals, 4 positions a place; native x86-64, one thread; real O_3): each position valued in
  full, then with a shortlist of its best 2 or 3 cards; every shortlisted card's sums checked equal to
  the full run's; both timed (best of 2).
- **Result 1.** Values identical in every case. Time as a ratio of the full search, summed over places:
  trick 2, 2 cards 0.64 and 3 cards 0.70 (leader 0.66 / 0.71); trick 3 (exact), 2 cards 0.69.
- **Method 2 (breakdown).** `crates/wasm/src/bin/shortlist_cost.rs`: trick-2 bench positions with at
  least 4 candidates (36 positions, 8 deals each, 144 deals). Each variant run as a request of its own
  (cold network memo, fresh table); solver nodes, network (frontier) calls and time counted; the setup
  no root card changes (belief preparation once, the draw of each deal) timed apart. Kept cards = the
  best by mean over those deals.
- **Result 2.** Ratio to the full solve:

| cards kept | solver nodes | network calls | time (two runs) |
|---|---|---|---|
| 1 | 0.24 | 0.25 | 0.25 / 0.29 |
| 2 | 0.49 | 0.49 | 0.55 / 0.58 |
| 3 | 0.80 | 0.80 | 0.86 / 0.90 |

  Cost follows the number of cards kept. The bench positions list 2-4 candidates (leader [4, 4, 4, 4];
  trick 3: 2-3 everywhere), so 2 kept cards are half of 4 and 3 kept are three quarters; Result 1's
  summed ratios included positions with 3 candidates, where 2 kept cards are 0.67 of the cards and 3
  kept are all of them. The setup costs 0.02-0.3 ms a deal: under 1% of a deal at seats 1-4 and 23% at
  the 8th seat, where the whole solve is 0.5 ms. In the golden set (E9) the mix is 493 / 421 / 1,086
  decisions with 2 / 3 / 4 candidates, so keeping 2 cards saves about 40% of the work over the mix and
  keeping 3 about 15%.
- **Cost.** No rental; a few minutes on the build VM.
- **Limits.** 3-4 positions a place and trick on a shared VM (timings vary by a few percent between
  runs; node and call counts are exact); the kept cards were the best ones, and cost depends on which
  cards are kept; no trick-3 position had 4 candidates, so the breakdown is trick 2 only; trick 1 (hands
  of 5) not measured.
- **Sources.** `tools/shortlist_time.py`, `crates/wasm/src/bin/shortlist_cost.rs`,
  `crates/wasm/tests/api.rs`, E8.

### E12. A decision-aware loss for the trick-2 student (runs `student-t2-plain-1` and `-plain-2`, 2026-10-22/23)
- **Question.** Does training the student on the ordering of a decision's candidate cards, and stopping on its card
  choices, lower its loss against the exact golden set, and by how much against the retraining noise?
- **Method.** Student 512-256 (3,292 inputs, `student-v1`), labels from the plain H_3 teacher (O_3 f5766a, 64 deals): the first
  250,000 decisions of the stored base shards (320 deals cut to 64, exact subsets, E3), and for the 1,000,000-decision networks the base,
  more-1 and more-2 shards the existing S2 was trained on. The decision-aware term (`fortyone_nn.student --rank-weight W
  --rank-temperature T`) adds W times the per-decision KL divergence from the teacher's softmax over the candidates to the student's,
  both on the deciding pair's money chance divided by T; W = 0 is the old loss bit for bit; `--stop-on decision` stops on the validation
  decision loss instead of the value cross-entropy (tests: `python/nn/tests/test_student.py`). `student_train` points take `seed`,
  `rank_weight`, `rank_temperature` and `stop_on`. Two GPU runs (`docs/vast/student-t2-plain-arm.manifest.json`, `student-t2-plain-2.manifest.json`;
  AMD EPYC 7502, RTX 5070 Ti, $0.325/h). Each integer network judged on the exact trick-2 golden set (2,000 decisions x 320 worlds) with
  `search student-check --golden-decisions --golden-worlds --top-k 3`: loss alone, at seat 1 (the leader), and with the teacher's perfect
  pick among the student's top 3. The stored S2 (954223e2) reproduced its recorded 0.0733% on this check.
- **Result.** Loss in % of the money chance (95% upper bound in brackets); seeds 1-3 where three were trained:

| network | decisions | seeds | alone | seat 1 | top-3 |
|---|---|---|---|---|---|
| existing S2 (plain loss) | 1,000,000 | 1 | 0.0733 [0.0876] | 0.179 | 0.0073 [0.0130] |
| plain loss, 64 deals | 250,000 | 3 | 0.1195 (SD 0.020) | 0.161 | 0.0100 |
| plain loss, 320 deals | 250,000 | 1 | 0.0936 [0.1089] | 0.140 | 0.0083 |
| plain loss, 64 deals, 100,000 / 50,000 decisions | 100,000 / 50,000 | 1 | 0.1314 / 0.2320 | 0.168 / 0.252 | 0.0087 / 0.0150 |
| stop on decision loss, no rank term | 250,000 | 1 | 0.1001 | 0.120 | 0.0219 |
| rank W=1, T=0.01, no stop rule | 250,000 | 1 | 0.0951 [0.1319] | 0.126 | 0.0235 [0.0564] |
| rank W=1, T=0.003 | 250,000 | 3 | 0.0855 (SD 0.014) | 0.132 | 0.0073 |
| rank W=1, T=0.003, stop on decision loss | 250,000 | 3 | 0.0780 (SD 0.006) | 0.132 | 0.0056 |
| **rank W=1, T=0.01, stop on decision loss** | 250,000 | 3 | **0.0587 (SD 0.003)** | **0.100** | **0.0045** |
| rank W=1, T=0.003, stop on decision loss | 1,000,000 | 1 | 0.0495 [0.0601] | 0.092 | 0.0036 |
| **rank W=1, T=0.01, stop on decision loss** | **1,000,000** | 3 | **0.0456 (SD 0.006)** | **0.090** | **0.0037** |

  Per seed (alone): plain 0.1217 / 0.1386 / 0.0980; T=0.003 0.0692 / 0.0966 / 0.0907; T=0.003 + stop 0.0770 / 0.0725 / 0.0845;
  T=0.01 + stop 0.0599 / 0.0550 / 0.0613. Retraining noise at the same data and settings is SD 0.020 for the plain loss. T=0.01 with the stop
  rule is below every plain seed and halves the plain mean, at SD 0.003; from 250,000 decisions it is below the existing S2, which used four
  times the data, and its leader-seat loss is 0.100 against 0.179. At 1,000,000 decisions (seeds 1-3, run `student-t2-seeds-1m-2` for seeds 2 and 3) it reaches 0.0456% alone (0.0389 / 0.0491 / 0.0487;
  seed 1 was the luckiest), 0.090% at seat 1 (0.080 / 0.094 / 0.097) and 0.0037% for the top-3 pick (0.0030 / 0.0025 / 0.0057), the lowest measured; the
  95% upper bounds alone are 0.0467 / 0.0646 / 0.0580. All three are below the existing S2 (0.0733%, 0.179%, 0.0073%), and the mean is 22% below the
  best 250,000-decision cell's 0.0587%: four times the data helps, less than the loss change did. The stop rule alone and plain at 320 deals lie inside the plain noise (one seed each). The same ordering
  holds on the runner's fixed 4,096-decision test set (plain seed 1 0.092, rank T=0.01 0.067, 250,000 decisions). Integer networks (content-addressed in R2):
  1M T=0.01 + stop seed 1 `3d1189e3042dfc9537b1f575f87e42b15e91628541e2ff90678ba319959c6e03`, seed 2 `01c7bd58e339ae5817ec6fde03820253826f61eed8b7323214add8183c5dcff7` and seed 3 `2f269718f62198e97ce0fbb1a063536a6596bb5e2d275e97148c522fb5b76a2b` (the last two in the run's out bundle `runs/student-t2-seeds-1m-2/out/seeds/`), 1M T=0.003 + stop
  `0cd6d08264dd99fe96ba0e6c0b496b6cc96b5604a0ba44302df7764e9d336db7`; the others in the runs' out bundles (`runs/student-t2-plain-{1,2}/out/plain/`).
- **Cost.** `-plain-1` 45 minutes, about $0.25 (9 networks); `-plain-2` 68 minutes, $0.37 (10 networks); each golden check a few seconds on the build VM.
  `student-t2-seeds-1m-2` (seeds 2 and 3 of the 1,000,000-decision network, taken over onto the hand-rented AMD EPYC 7742 server chip, 256 threads,
  257 GB RAM, RTX 3070 8 GB, $0.796/h): 53.7 minutes, $0.71, of which 33 minutes were fetching and encoding the 25 GB of labels (once for both seeds) and
  about 9 minutes training each network. The machine was kept afterwards.
- **Limits.** Three seeds for the 1,000,000-decision network of T=0.01 (one for T=0.003); one seed for the stop rule alone, plain at 320 deals and the learning-curve points; the plain-320 difference
  (0.094 against 0.119) is within the noise; the weight and temperature were tried on a small grid (W = 1; T = 0.003 and 0.01) only; every network is judged
  on the same 2,000-decision golden set, which the training and the stop rule's validation split never touch; the labels are the plain H_3 teacher's
  (noisy at 64 deals, E4). Export check (2026-10-23, build VM, `WASM_STUDENT=<seed-1 network 3d1189e3...> tools/wasm_check.sh`, with the equity table and O_3): the 21
  requests of the API check, seven of them `method: "student"` with this network, give the same bytes natively and as WebAssembly, and again when split into 2, 3, 4 and 7
  shards; PASS. (The app-path check is E16.)
- **Sources.** `python/nn/fortyone_nn/student.py`, `crates/runner/src/student_train.rs`, `docs/vast/student-t2-plain-arm.manifest.json`,
  `docs/vast/student-t2-plain-2.manifest.json`, R2 `runs/student-t2-plain-{1,2}/`, E2-E5.

### E13. Does a G2-labelled teacher give a better trick-2 student? (runs `student-t2-g2arm-50k-1` and `student-t2-plainarm-50k-1`, 2026-10-23)
- **Question.** Labelled by H_3 with G2's learned exact leaves instead of plain H_3 (same positions, same 64 deals), does the trick-2 student get a lower
  golden-set loss, with the plain loss and with the decision-aware loss of E12?
- **Method.** The first 10 parts of 5,000 decisions of the rules stream (seed 401, decisions 0-49,999), the same positions in both arms. Plain labels:
  the base run's shards cut to 64 deals. G2 labels: `docs/vast/student-t2-gv2-250k.manifest.json` (O_3 f5766a, G v2 t=277338260, 64 deals), parts from runs
  `student-t2-gv2-250k-4`, `-6` and `-8` (E14); a part of 1,000 decisions holds exactly the records of the same decisions inside a part of 5,000 (every field
  equal except the offset into the values file). Each arm: 512-256, 50,000 decisions, seeds 1-3, with the plain loss and with the decision-aware
  loss (W=1, T=0.01, stop on decision loss); judged on the golden set as in E12 (`docs/vast/student-t2-{g2,plain}arm-50k.manifest.json`; the G2 arm
  needs `allow_stamps: teacher` against the plain test set). The planned 250,000-decision arm was not made: the G2 labelling was cancelled at 60,000 decisions.
- **Result.** Loss alone in % of the money chance (seat 1 and top-3 are means over the three seeds):

| labels | loss | alone, per seed | alone, mean (SD) | seat 1 | top-3 |
|---|---|---|---|---|---|
| plain H_3 | plain | 0.2718 / 0.2436 / 0.2140 | 0.2431 (0.029) | 0.342 | 0.0167 |
| H_3 + G2 | plain | 0.2392 / 0.2312 / 0.2406 | 0.2370 (0.005) | 0.272 | 0.0139 |
| plain H_3 | decision-aware | 0.1095 / 0.1306 / 0.1249 | **0.1217 (0.011)** | 0.143 | 0.0089 |
| H_3 + G2 | decision-aware | 0.1323 / 0.1545 / 0.1505 | 0.1458 (0.012) | 0.150 | 0.0132 |

  G2 labels minus plain labels (difference of means, standard error): plain loss -0.006 (0.017), no difference; decision-aware loss +0.024 (0.009),
  the G2-labelled network worse by about 2.6 standard errors, its top-3 pick worse too (0.0132 against 0.0089). The decision-aware loss halves the
  loss on both label sets (0.122 against 0.243 on plain labels, 0.146 against 0.237 on G2 labels).
- **Cost.** The G2 arm 13 minutes, $0.05; the plain arm about 20 minutes, about $0.2; the G2 labels used here are part of the $5.3 of E14.
- **Limits.** 50,000 decisions only and three seeds per cell, so an effect that appears with more data is not excluded (the second look at 100,000 decisions was
  not made); one teacher setting (G v2 at t=277338260, 64 deals); the labels differ only in the teacher; judged on the 2,000-decision golden set only.
- **Sources.** `docs/vast/student-t2-g2arm-50k.manifest.json`, `docs/vast/student-t2-plainarm-50k.manifest.json`, R2 `runs/student-t2-{g2arm,plainarm}-50k-1/`, E7, E12, E14.

### E14. Throughput of the G2 teacher on rented machines (runs `student-t2-gv2-250k-3`, `-4`, `-6`, `-8`, 2026-10-22/23)
- **Question.** What does a trick-2 decision at 64 deals cost with G2, and does the teacher keep a 245-thread
  machine busy?
- **Method.** Four rentals of the same manifest (`docs/vast/student-t2-gv2-250k.manifest.json`), each with
  every usable thread: `-3` AMD EPYC 7B12, 245 threads, $0.708/h, parts of 1,000 decisions; `-4` AMD EPYC
  7742, 245 threads, $0.735/h, parts of 5,000; `-6` AMD EPYC 7B12, 245 threads, $0.723/h, parts of 5,000, the
  ordered writer's window widened to the whole part (binaries 258ef77). A local smoke test (4 vCPU build VM): 6
  decisions at 8 deals, plain against G2. Part times from the stored parts' timestamps in R2, CPU use from the
  30-second heartbeats.
- **Result.**

| run | machine and price | part | part time | CPU use | outcome |
|---|---|---|---|---|---|
| build VM | 4 vCPU | 6 decisions x 8 deals | 3.0 s plain, 45.9 s G2 (15 times) | - | smoke test |
| `-3` | 7B12, $0.708/h | 1,000 decisions | not finished in 28 min | 100% for 5 min, then falling to 2%; average 58% | stopped by the idle watchdog, $0.33, nothing stored |
| `-4` | 7742, $0.735/h | part 0, 5,000 | 41 min | 100% swinging to 0-5% (ordered window of 4 x threads + 16) | part 0 stored; part 1 not finished in 28 min; cancelled, about $0.93 in all |
| `-6` | 7B12, $0.723/h | part 1, 5,000 | 38 min | 100% for about 25 min, then one straggler decision at about 0.5% | part 1 stored; part 2 cancelled; about $0.55 in all |
| `-8` | 7B12, $0.723/h, both fixes (binaries c6d8c08) | parts 2-12, 5,000 each | 28 min (median 1,683 s) | steady 100% | 12 parts stored (60,000 decisions) in 4.9 h, $3.49; cancelled at 02:47 UTC on 2026-10-23 by someone other than the assistant (stop and no-replacement markers) |

  From `-8` (both fixes): 5,000 decisions in 28 minutes on 245 threads, 10,700 decisions an hour, **$68 per million decisions** at $0.723/h
  (the fixes cut a part from 38 to 28 minutes, 27%), so about $17 and 23 hours for 250,000. From `-6`: 5,000 decisions in 38 minutes on 245 threads is 112 thread-seconds a decision, 1.75 thread-seconds
  a deal (the docs' 1.55 on a 7B12 plus the idle tail): about 7,900 decisions an hour and $92 per million
  decisions at $0.723/h, so about $23 and 32 hours for 250,000. Causes found: (1) the ordered writer let workers
  run only 4 x threads + 16 decisions ahead, so one slow decision stalled every worker behind it (fixed:
  window = the whole part); (2) a decision's 64 deals ran on one thread, so the heaviest decisions (leader
  seats) leave the machine idle for the last 10-20 minutes of a part (fixed: a decision's deals are worked in
  blocks of 8 on any thread and joined, `DEAL_BLOCK`; records byte-identical, test
  `blocks_of_deals_join_into_the_whole_decisions_records` for 19, 8, 3 and 16 deals on 1-4 threads).
- **Cost.** `-3` $0.33, `-4` about $0.93, `-6` about $0.55, `-8` $3.49; about $5.3 in all, 60,000 decisions stored.
- **Limits.** The block fix's effect is measured only together with the window fix (`-8` against `-6`: 28 against 38 minutes a part); part times include about 1 minute of setup; one machine type (Zen 2)
  measured; the estimate in E1/path-forward ($15-30 per million) was for a Zen 4 machine at $1.34/h.
- **Sources.** R2 `runs/student-t2-gv2-250k-{3,4,6}/` (`report.json`, `heartbeat/`, `out/`), `crates/search/src/student_teacher.rs`,
  `crates/search/src/ordered.rs`, `docs/VAST-MACHINES.md` section 5.

### E15. Keeping a machine after a run and taking it over (session 1, 2026-10-22)
- **Question.** Can the next job be deployed onto a machine without renting another (an hour's search found the
  one cheap 245-thread machine), and can a run leave its machine up when it ends or fails?
- **Method.** `machine.keep` / `machine.keep_hours` / `machine.instance_id` in the manifest (`launch.py`
  `keep_machine`, `take_instance`; the runner parks on `KEEP_MACHINE=1`, `crates/runner` `end_machine`;
  `cleanup.py` `decide_kept`): the machine is left running when a run ends (DONE or FAILED) or the workflow is
  cancelled, recorded in `keep/<instance>.json` and `runs/<run>/KEPT.json` with a warning and a job-summary
  banner; the hourly cleanup destroys it at its keep time; a takeover updates the instance's image,
  environment and binaries and recycles the container (a stopped instance is updated and started). Refused: another run's live
  machine (heartbeat within 15 minutes), a machine another run started under an hour ago, a label that is not
  `41bots-`. Tested with 25 tests against the launcher's fake vast and R2 (`python/launcher/test_keep.py`, 36
  cleanup tests) and tried on a spare machine (GTX TITAN X, Xeon E5-2630 v4, 20 threads, $0.04/h) with
  `difftest` jobs: run `keep-trial-1` with `keep: true`, then `keep-trial-2` with `instance_id` of that machine.
- **Result.** `keep-trial-1` ended DONE after 20.5 minutes (mostly the image pull on a 278 Mbps link), $0.014:
  its machine stayed up (instance 57255649 running afterwards), `keep/57255649.json`, `runs/keep-trial-1/KEPT.json`
  and the warning were written. **`keep-trial-2` (the takeover of that machine) failed:** the instance was
  updated, recycled and relabelled `41bots-keep-trial-2` and its keep record deleted, but the recycled container
  booted the **old** environment (its log: `boot: run keep-trial-1 ... runner: run keep-trial-1`, re-running the
  first job and failing to upload over its immutable files), so no heartbeat came for `keep-trial-2`; after 15
  minutes the launcher destroyed the machine ($0.013). The same takeover method (`vastai update instance --env`
  then `recycle instance`, built for `o3-round.yml`, `--instance-id`) had also failed on 2026-10-21 (`o3r-1-1`,
  cause then not found); so the stored environment is not what the recycled container receives. Cause not
  found (trials 3 and 4 were stopped before they could be probed). The owner then reported that the O2 session's takeover of the
  hand-rented instance 57259029 succeeded; whether it was stopped or running at the time, and which calls were made, is not recorded here. So a
  takeover of a hand-rented machine has worked once and one of a launcher-created kept machine has failed once; the cause of the difference is
  not established (an untested guess: the kept machine's environment was fixed when the launcher created it). Afterwards added: a taken-over machine that never starts is kept and recorded, never
  destroyed (tested against the fake vast). Starting a *stopped* instance is tested against the fake vast only.
  A second kind of failure, 2026-10-23: the S2 session (run `student-t2-seeds-1m-1`, 08:27:35 UTC) and the O_2 session (run `o2-audit-1`, 08:31:29 UTC) took over the same
  kept machine (57338218) four minutes apart; the second takeover's container replaced the first's (`RUN_ID` was `o2-audit-1`, which heartbeated from 08:31:57), and the
  first run, left without a container, would on a normal cancel have run `stop`, which destroys the instance. It was force-cancelled (`gh run cancel --force`, which
  skips the cleanup steps); the machine, running the O_2 audit, was untouched. Added: a claim (`claims/<instance>.json`, written by `take_instance`, read back after 8 s,
  the last writer wins; a fresh claim of a run that has not ended blocks a second takeover for 45 minutes, also in `take_guard`); four tests (`test_keep.py`).
  Later the same day, with the claim lock and `force_take` in place: a takeover of the owner's hand-rented, *running* instance 57346368 (AMD EPYC 7742, RTX 3070, no label,
  the machine of the collision after the owner re-rented it) worked: run `student-t2-seeds-1m-2` was launched at 08:38:52 UTC, `watch.json` and the first heartbeat
  (`RUN_ID` the new run's) followed at 08:41, the run finished (DONE) and the machine was kept, not destroyed. A second takeover (57349848, a hand-rented unlabelled
  instance, 09:19 UTC) had relabelled the instance and was recycling its container when the owner stopped it by hand; the workflow was force-cancelled, its claim
  deleted and the instance relabelled, nothing destroyed. So a takeover of a hand-rented running machine has now worked twice (once per session) and failed to complete never.
- **Cost.** About $0.05 for the trial machine.
- **Limits.** One spare machine type; **a takeover of a launcher-created kept machine failed on vast** (above; one of a hand-rented machine was reported to work, stopped or running not recorded); a
  machine with a different disk size is not tried; a boot failure (missing manifest, bad binary hash) still destroys the machine because
  the image's boot script does (an image change would be needed); the runner image and run workflow were not
  changed beyond the manifest keys and the cancel steps in `vast-run.yml`.
- **Sources.** `python/launcher/launch.py`, `cleanup.py`, `test_keep.py`, `monitor_run.py`, `.github/workflows/vast-run.yml`,
  R2 `runs/keep-trial-{1,2}/`.

### E16. The new trick-2 student in the app's shortlist-plus-search path, and the router (`search student-check --sub-deals`, `tools/router_fit.py`, 2026-10-23, build VM)
- **Question.** With the decision-aware network of E12 (1,000,000 decisions, `rank_weight` 1, `rank_temperature` 0.01, stop on decision loss), at how many
  deals does "the student shortlists, a short search picks" beat the student alone (E9), and can the app find the decisions where the student loses and search only
  those (E10)?
- **Method.** As E9 and E10, with the new network (`3d1189e3...`, E12) and, as a check, the old S2 (954223e2) on the same settings: the 2,000-decision exact
  golden set (320 exact worlds each), per decision and repeat (8, seed 1) a seeded shuffle of the worlds, the first N (10, 25, 50, 100) the sample, the method picks on
  the sample and is scored on the other 320 - N worlds; methods: the student alone, search over all cards, top-2 and top-3 of the student searched; 95% bounds by
  bootstrap over decisions. The router: `tools/router_fit.py` (gradient-boosted risk model on the student's own values, 5-fold cross-validation, 3 seeds).
- **Result.** Loss in % of the money chance (corrected, on the golden check's scale); the old S2 reproduced E9's numbers (alone 0.0733; top-3 + search 0.1030 / 0.0596 / 0.0402 / 0.0293):

| deals | new S2 alone | search all cards | top-2 + search | top-3 + search | old S2 top-3 + search |
|---|---|---|---|---|---|
| 10 | **0.0389** | 0.1165 | 0.0746 | 0.0983 | 0.1030 |
| 25 | **0.0389** | 0.0623 | 0.0476 | 0.0554 | 0.0596 |
| 50 | 0.0389 | 0.0389 | 0.0360 | 0.0360 | 0.0402 |
| 100 | 0.0389 | 0.0265 | 0.0286 | **0.0258** | 0.0293 |

  At seat 1 (the leader) the new S2 alone loses 0.0795 (the old: 0.1794); top-2 + search 0.1062 / 0.0739 / 0.0556 / 0.0489 and top-3 + search 0.1427 / 0.0916 /
  0.0606 / 0.0499 at 10 / 25 / 50 / 100 deals; search over all cards 0.1996 / 0.1193 / 0.0782 / 0.0642. The loss is still concentrated: the worst 1 / 5 / 10 / 20% of
  decisions hold 38.9 / 78.8 / 93.1 / 99.8% of it (old S2 38.8 / 75.1 / 90.9 / 99.1%); 465 of the 2,000 decisions lose anything. The risk model's cross-validated correlation
  with the true loss is 0.055 (the student's margin: 0.037); its predicted-riskiest 10 / 20 / 50% of decisions hold 20.0 / 36.5 / 71.3% of the loss (the margin's lowest 10 / 20 / 50%: 8.1 /
  17.3 / 49.6%). Router, loss when the f% predicted riskiest are searched (top-3 + N deals) and the rest keep the student's card: at 10 deals 0.0438 (5%) rising to 0.0983
  (100%); at 25 deals 0.0400 (5%), 0.0419 (20%), 0.0554 (100%); at 50 deals 0.0366 (10%), 0.0356 (20%), **0.0338 (50%)**, 0.0360 (100%); at 100 deals 0.0331 (20%), 0.0284
  (50%), 0.0258 (100%); the oracle router (sorted by true loss) 0.0145 at 10% at 50 deals and 0.0124 at 10% at 100 deals.
- **Cost.** No rental; a few minutes on the build VM (the router fit about 10 minutes).
- **Limits.** Exact world values: no leaf error, optimistic for any real search (E9); the sample is drawn without replacement from 320 worlds, which lowers its variance by
  about 17% at 100 deals; the router was fitted on one network and one golden set with cross-validation, not on fresh decisions; its features are the student's values, place,
  stratum and candidate count only; one seed of the network (E12); the phone's cost of a deal is separate (a leader deal is 368 ms in WebAssembly, `WASM-API.md` section 8).
- **Sources.** `crates/search/src/shortlist.rs`, `tools/router_fit.py`, E9, E10, E12.

### E17. A weight and temperature search for the decision-aware loss (run `student-t2-losssearch-1`, 2026-10-23)
- **Question.** Is there a `rank_weight` and `rank_temperature` better than the weight 1, temperature 0.01 of E12, and how much does the choice matter?
- **Method.** As E12: 512-256, the first 250,000 plain-H_3 decisions at 64 deals, stop on the validation decision loss; weight 0.3 / 1 / 3 against temperature
  0.005 / 0.01 / 0.02 / 0.03, seeds 1 and 2 for every cell except (1, 0.01), which is E12's three seeds (22 networks; `docs/vast/student-t2-losssearch.manifest.json`, one
  GPU, a Xeon E5-2697 v4 box at $0.134/h). Each network judged on the exact golden set as in E12.
- **Result.** Loss alone in % of the money chance, seeds 1 / 2, then the 2-seed mean; seat 1 and top-3 are means:

| weight | temperature | alone, seed 1 / 2 | alone, mean | seat 1 | top-3 |
|---|---|---|---|---|---|
| 0.3 | 0.005 | 0.0839 / 0.0865 | 0.0852 | 0.096 | 0.0160 |
| 0.3 | **0.01** | 0.0515 / 0.0636 | **0.0576** | 0.091 | 0.0054 |
| 0.3 | 0.02 | 0.0637 / 0.0654 | 0.0646 | 0.085 | 0.0061 |
| 0.3 | 0.03 | 0.0732 / 0.0653 | 0.0692 | 0.103 | 0.0051 |
| 1 | 0.005 | 0.0854 / 0.0757 | 0.0806 | 0.087 | 0.0132 |
| **1** | **0.01** (E12, 3 seeds) | 0.0599 / 0.0550 / 0.0613 | **0.0587** | 0.100 | 0.0045 |
| 1 | 0.02 | 0.0720 / 0.0736 | 0.0728 | 0.092 | 0.0043 |
| 1 | 0.03 | 0.0871 / 0.0652 | 0.0762 | 0.104 | 0.0040 |
| 3 | 0.005 | 0.0524 / 0.0651 | 0.0588 | 0.091 | 0.0048 |
| 3 | **0.01** | 0.0550 / 0.0684 | 0.0617 | 0.090 | **0.0032** |
| 3 | 0.02 | 0.0746 / 0.0567 | 0.0657 | 0.089 | 0.0044 |
| 3 | 0.03 | 0.0677 / 0.0726 | 0.0701 | 0.112 | 0.0040 |

  No cell is clearly better than weight 1, temperature 0.01 (0.0587): the nearest are weight 0.3 at 0.01 (0.0576), weight 3 at 0.005 (0.0588) and weight 3 at 0.01 (0.0617),
  all within the noise (the two seeds of one cell differ by up to 0.017). By temperature, over the weights: 0.005 gives 0.0852 / 0.0806 / 0.0588, 0.01 gives 0.0576 /
  0.0587 / 0.0617, 0.02 gives 0.0646 / 0.0728 / 0.0657, 0.03 gives 0.0692 / 0.0762 / 0.0701 (weights 0.3 / 1 / 3): temperature 0.01 is the best column at every weight but 3,
  where 0.005 ties it; 0.02 and 0.03 are worse; the weight matters little between 0.3 and 3 at temperature 0.01, and a small weight with temperature 0.005 is poor (0.085). The top-3
  pick is lowest at weight 3, temperature 0.01 (0.0032) and highest at temperature 0.005 with weight 0.3 or 1 (0.0160, 0.0132).
- **Cost.** 128 minutes, $0.29 (22 networks); the golden checks about 3 minutes on the build VM.
- **Limits.** Two seeds per cell (the seed SD within a cell is 0.003-0.010, E12), so differences under about 0.01 between cells are not resolved; one set of 250,000
  plain-H_3 decisions; the grid is coarse (a factor of 2-3 in each setting) and the stop rule is on in all cells; a third seed was not run for the best cells; the
  1,000,000-decision network of E12 was not retrained with another setting.
- **Sources.** `docs/vast/student-t2-losssearch.manifest.json`, R2 `runs/student-t2-losssearch-1/out/search/`, E12.

---

## Part F. Runs and builds without a result, and work set aside

| item | state | what exists |
|---|---|---|
| O_3 base curve 50% / 100% (`o3-basecurve-*`) | cancelled at 37 s, 2026-10-19 | nothing |
| O_3 wider / deeper (`o3-size-*`) | never launched | manifests |
| Sibling ranking A/B (`o3-sibrank-a/-b`) | never launched; run on 2026-10-23 as `o3-pair-1` (D12) | build-VM A/B only (C9) |
| Four-network ensemble | stopped twice for speed | single-network checks only (C8) |
| Hybrid-label test (`o2-labeltest-1`) | cancelled at 8 min | the hybrid labels' speed; 20k exact validation shard (s602) |
| O_2 on G v2 labels (`o2-gv2-labels-2/-3`) | stopped by owner | 5k exact validation shard (s603) |
| Trick-2 hybrid-label run (`m3-hybrid-t2-hlabels`) | never launched | manifest |
| Silver set and exact anchor (D4) | incomplete / not run | a few hundred checkpointed worlds |
| O_2 G v2 trick-1 test (D3) | not run | the predictor model |
| Trick-1 deal-count curves (A7) | incomplete | 31 + 355 saved exact worlds |
| PIMC-v0 vs rules-v1 SPRT gate (M3-SPEC §8.3), PIMC-suitability report (§6) | not run | — |
| Speed tests of AMD EPYC 9B45 offers (2026-10-22) | stopped: 46 usable threads | no speed figure |
| O_2 test-set parts 1-3 (`o2-testset-1/-2/-3`) | stopped, one failed on a chunk error, 2026-10-22 | replaced by `o2-stagea-full-2` (D6) |
| O_2 volume labels as streamed parts (`o2-vol-1`, `o2-vol-2`) | no machine found in 50 min / label parts hit the trainer's sidecar race | replaced by ten stored streams (`o2-vol-3`, D7) |
| O_2 on 19.8M labels as `o2-volnet-9` | superseded by the same run with `keep` (`o2-volnet-9k`, D8) | manifest |
| O_2 audit loop launches `o2-audit-4` to `-9` | failed or abandoned (a 6 Mbps host, boxes destroyed before the takeover, a box whose GPU could not start containers) | A10, D11 |
| O_2 audit loop round 3 and round 2's harvest | not run (stopped by the owner after round 2's audit) | D11 |
| O_3 sibling ranking A/B arms `base_s2`, `w1_s2`, `w3_s1` of `o3-pair-1` | not run (stopped by the owner after weight 1) | D12 |
| O_3 sibling ranking A/B launch `o3-pair-2` | container never started under that name (A10) | the experiment ran as `o3-pair-1` |
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
| plain hybrid trick-2 label, 245 EPYC 7742 threads | 1,170 labels/s (4.8 a thread-second), $0.41 per 2.2M rows | D7 |
| plain hybrid trick-2 label, 368 EPYC 9654 threads | about 2,000 labels/s (5.4 a thread-second) | D11 |
| exact trick-2 audit label, 184 EPYC 9B14 threads | 41 rows/s (about 4.5 thread-s) | D11 |
| targeted harvest row (plain labels at trick 2 driven by the network), same machine | 320-335 rows/s (about 0.55 thread-s, about 3x a random plain label) | D11 |
| O_2 training, 39.6M rows, 2048-1024-256 | about 1.7 min an epoch on an RTX 5090 (38,672 batches of 1,024) | D11 |
| O_3 training, 46M rows, 2048-1024-256 | about 3.4 min an epoch on an RTX 3090 (44,922 batches) | D12 |
| trick-1 teacher decision (H_2 + O_2 round 1), 245 EPYC 7B12 threads | 64 deals: 65 s of compute a chunk of 500 (29,700 an hour), 105 s with setup (16,800 an hour); 128 deals: 108-124 s (15,600 an hour) | D13 |
| R2 download of 62-75 GB of shards | 12-20 min (50-65 MB/s; 6 Mbps on one host) | A10 |
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
| O_2 networks of D6-D9 | `p60` runs/o2-stagea-full-2/out/p60; `v11` runs/o2-volnet-5/out/volnet; `v20` runs/o2-volnet-9k/out/volnet; capacity arms runs/o2-volcap-1/out/{seed2,deep,wide} |
| **O_2 adopted (audit loop round 1, 39.6M plain labels)** | integer 13152c24bc27…, float and report under runs/o2-audit-10/partial/rung/000000/out/hybrid/net/ |
| O_2 audit loop round 2 network | 235ed448acc7…, under .../audit/round-2/net/ |
| O_2 plain labels (22M and 17.6M rows) | per-stream `shards.json` of runs/o2-vol-3 (vol01-vol10) and runs/o2-vol40-1 (vol11-vol18); 54 shards, 74.6 GB |
| O_2 exact test set (250k) and sibling groups (4,479) | ten parts in runs/o2-stagea-full-2/partial/test01-test10; sibling shard 5734d7f0… |
| trick-3 exact sibling set `sib3` (100k, seed 7903) | shard 5cdf183b…, groups d2854861…, sidecar c4fb5318… |
| O_2 audit round-1 shard (200k exact) and targeted shard (2M) | c8a8a392…, 6fceb941… |
| O_3 sibling A/B arms | runs/o3-pair-1/out/base_s1 and out/w1_s1 |
| O_3 ladder networks (1M, 3M, 10M, 20M rows) | runs/o3-ladder-1/out/lad{1,3,10,20}/ |
| copper and S1 teacher labels with O_2 round 1 | runs/o2-copper-s1-1/out/{copper,s1rules,s1mixed}/ |
| equity table v0 | 64740dab…6808 |
