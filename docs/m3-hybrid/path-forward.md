# The path forward after the O3G2 rounds (proposed 2026-10-22, for the owner's approval)

Status: **proposal**, written from the owner's sequence of 2026-10-22 and our measured results. Nothing
here is running. Every number is cited to `docs/EXPERIMENTS.md` (written "E4", "D2" ...) or marked
"estimate". Where this page and STATUS differ, STATUS is current.

## 1. Where we stand (facts)
- O_3 (trick-3 network): 0.0068% alone (seat 1 0.0201%), 0.0008% with G2 and exact solving (seat 1
  0.0022%). Three attempts to lower O_3 alone (O3G2 rounds, 80M base, specialist S) did not move it
  (C4, C12, C13).
- O_2 (trick-2 network) is provisional: 1.5M exact labels, position error SD 3.6% against O_3's 1.85%
  (D1). It has had **far less data than O_3's 40M positions**.
- Trick-2 student S2 (512-256, 1M decisions, plain H_3 teacher): 0.073% alone on the exact golden set;
  **top-3 shortlist with a perfect final pick 0.0073%** (bound 0.013%), seat 1 0.0148% (E4-E5).
- Trick-1 student: none. Trick-1 teacher (H_2 with O_2) costs about **$84 per million decisions at 64
  deals** (E6, 12,489 decisions an hour on a 256-thread EPYC 7B13, $1.05/h, 5.28 candidates a decision).
- No trick-1 golden, silver or bronze set exists: a trick-1 world solved exactly through tricks 1 and 2
  then O_3 costs 10-30 thread-minutes on the build VM (D4), and one sample measured on 2026-10-22 was
  still unfinished after 50 minutes of one thread (E8).
- App budget: 7 s a decision on a phone. In WebAssembly a trick-2 deal costs 368 ms (leader) to 0.9 ms
  (8th seat) per thread (`WASM-API.md` §8); the search over N deals loses about 0.01% x (243/N)^0.738 at
  trick 2 (A6: N for 0.01% is 243), i.e. about 0.05% at 25 deals, 0.03% at 50, 0.02% at 100 (estimate
  from the fitted curve; all candidates, no shortlist).
- A shortlist alone does **not** make the exact solve cheaper today: the solver values every root card in
  one search, so dropping cards from the list changes nothing (E8, measured).

## 2. How to use the two big gains
Two measured gains of about 8-10 times: the **top-3 shortlist** (student alone 0.073% to 0.0073% with a
perfect pick) and **G2** (O_3 alone 0.0068% to 0.0008%). They are not equally usable:

| gain | where it can run | what is needed to collect it |
|---|---|---|
| top-3 shortlist (about 10x) | online and offline | a **root-restricted solve** (search only the shortlisted first cards; expected cost about the share of root cards kept, to be measured) and enough deals for the final pick: the "perfect pick" used 320 exact worlds, the app has 25-100 |
| G2 + exact leaves (about 8.5x) | **offline only**: 586 thread-ms a deal on a 9B14 against 24 for O_3 alone, far beyond the 7 s phone budget | use it only where compute is free: teachers for students on close calls, yardsticks, the final label pass |
| both, as a router | online | a learned risk model for the **student's** decision (G2's idea moved to decision level): spend the 7 s of search only where the student's top card is likely wrong, skip it elsewhere. G2 shows error concentrates (75% of O_3's error in its riskiest 5% of positions); the student's loss is concentrated by place and score class (leader 0.084%, 5th seat 0.088%, S4 0.089%, E4) and its per-decision concentration is **not yet measured** |

## 3. The path, in order, with cost and the judge of each step
Costs are rental dollars; "estimate" means not measured.

**Step 0: free reads on the build VM (no rental).**
- 0a. **Shortlist + search at the app's deal counts.** Sample N = 25, 50, 100 of the 320 stored exact
  worlds per golden decision; the student shortlists 3 cards; pick the best of them by mean over the N
  worlds; score against the full golden values. Needs a small extension of `student-check`
  (`--deals N`). It uses exact world values, so it isolates the sampling noise and is optimistic about
  leaf error. Decides whether shortlist + search beats the student alone at app deal counts.
- 0b. **How concentrated is the student's loss per decision** (the router's feasibility), from the same
  golden check's per-decision losses.
- 0c. **Root-restricted solve.** The solver half is **built and on `main`** (`FastOptions::root_keep`,
  `crates/solver/tests/root_keep.rs`): kept cards keep byte-identical values, one root card cost 509,266
  nodes against 1,353,126 for all cards over 20 tricks 3-5 positions. What is left: wire it through
  `search::pimc::exact_world` and the WASM API, then measure the saving at trick 2 and trick 1 (see
  section 6, session 0).

**Step 1: S2.** (owner's step 1: "train S2 on O3G2")
- Cost of the teacher: H_3 + G2 is about 24 times a plain H_3 world (586 against 24 thread-ms on a 9B14,
  E-table G), so about **$15-30 per million decisions at 64 deals** against about $0.8 plain (estimate
  from the 4.7 thread-s decision cost, E1).
  **Measured 2026-10-22 (E13):** 112 thread-seconds a decision on a 245-thread EPYC 7B12 at $0.723/h, i.e.
  7,900 decisions an hour and **about $92 per million decisions**, so about $23 and 32 hours for the 250,000 of
  this A/B. The estimate above was for a Zen 4 machine at $1.34/h. Two throughput bugs (ordered-writer window,
  a decision's deals on one thread) wasted about 40% of the first rentals; both are fixed (E13).
- Run it as a **cheap A/B, not a build**: 250k decisions plain against 250k with the G2 teacher, same
  network and seed, judged on the exact golden set (alone and top-3), about $5-10 plus training.
- Also A/B the decision-aware loss (`rank_weight`, planned step 3 of the student plan, about $0.5).
- Why a student-fit limit is suspected: its gap to a 512-deal teacher is 0.0506% against O_3's own
  0.0068% error, a 1024-512 network was no better and more data helps only as N^-0.23 (E4). The A/B tests
  whether a cleaner teacher moves it.
- The student audit loop (harvest where the student loses most) is **low priority**: the same method did
  not move O_3 (C2, C12).

**Step 2: O_2 (the long pole).** O_2 must exist before the trick-1 teacher, copper and S1.
- Labels are **positions at the start of trick 2, all cards face up**, valued exactly through trick 2 and
  then by plain O_3 ("hybrid labels", D2). Bot play (random, rules-v1, H_2-driven harvest) only chooses
  which positions to label; it never supplies the label.
- Cost per label (D2): plain hybrid about 0.1 thread-s, **exact 11.3 thread-s, G2 labels 9.3 thread-s**.
  So per million labels: plain about $0.07-0.15, exact about $7-9 (estimate at 0.0022-0.003 $/thread-h),
  G2 labels the same order as exact. **G2 labels have no niche**: they cost 1.2x less than exact, which is
  perfect. The real choice is plain (volume) against exact (truth); label quality of plain labels against
  exact: RMS 2.1%, largest team bias 0.013-0.017% (D2). That is the unrun **label test at equal money**
  (`o2-labeltest-1` was cancelled at 8 minutes): judged on the exact validation shards s601/s602/s603,
  bias first.
- Then the audit loop on O_2 as for O_3, standard 0.01%. O_2 is judged on exact trick-2 positions, **not
  on copper** (copper is labelled by O_2 itself).
- Decision to make: adopt whichever label source wins the equal-money test, then the volume (O_3 had 40M).

**Step 3: trick-1 yardsticks** (after O_2).
- **Copper** (new name for the set below bronze): about 2,000 trick-1 decisions labelled by the teacher
  H_2 + O_2 at 128 deals. About 10-20 minutes on a 256-thread machine, **about $0.2-0.5** (E6 rate, estimate
  at 128 deals). Measures how well S1 copies its teacher. It rewards copying: a student better than its
  teacher would score *worse* on it.
- **Playout yardstick** (to judge "better than the teacher"): for each of a few thousand trick-1
  positions, play each of the top-3 cards on many dealt hands with the same fixed strong players
  finishing the hand (S2 or H_3, then exact), compare the mean money chance of the cards on the same
  deals. Independent of O_2 and of PIMC's perfect-information assumption. Cost depends on the playout
  speed; resolving 0.1% takes about 10^5 playouts per position-card pair (estimate); design and a pilot
  first.
- **Small silver/bronze anchor**: a few dozen decisions x a few dozen worlds, to measure the chain
  (S1 vs copper) + (O_2 vs bronze) + (the reference's own error). Estimate $5-15; trick-1 world costs have
  heavy tails (A4, D4), so cap it. Not a full set (the first attempt was stopped at about $100 for about
  10%).

**Step 4: S1.**
- Labels from the teacher H_2 + O_2, 64 deals, racing (extra deals only for close calls). **About $84 per
  million decisions** at the E6 rate and the provisional O_2's width; racing and a restricted root should
  lower it (estimate $40-100 per million). S2 needed 1M decisions and still loses 0.057%, so plan for 1M
  and a learning-curve point at 250k.
- Judged on copper, the playout yardstick and the small anchor; the decision-aware loss as in step 1.

**Step 5: ship S1 and S2 alone** (milliseconds a decision, `method: "student"`, WASM 0.4-0.7 ms).

## 4. Improvements, by expected gain against work
1. **Root-restricted solve and the router** (step 0 decides): turn the shortlist's 10x into speed and
   accuracy online. Trick 2 first (O_3 at the leaves, no G2 online), then trick 1 (needs a smaller leaf
   network: a trick-1 deal is 0.2-0.6 s on the build VM and about 5x that in WebAssembly, so 2-8 deals in
   7 s as things stand).
2. **Disagreement set from the pros** (7 pros x 60 matches, about 420 matches): their moves are not the
   truth. Use them as realistic positions and as a finder: where the bot and a pro disagree, spend heavy
   offline compute (many deals, G2 on close calls) to see who is right. Expected about 6,000 trick-1
   decisions (estimate).
3. **Self-play on very thin slices** (e.g. trick 1 with 2-3 trumps close to the money), judged by the
   **playout yardstick**, not copper. If it wins in a slice, a predictor chooses between S1 and the
   self-play network (the same learned-routing idea as G2). A DouZero-type trial from scratch needs about a
   billion decisions, so start from the teacher's network with a decaying teacher penalty (PLAN phase B);
   estimate $100-300 for a first trial.
4. **Public matches** (thousands of matches from hundreds of people): tens of thousands of decisions, far
   too few for self-play volume. Use for the position distribution humans create and for the disagreement
   finder (item 2).

## 5. Decisions for the owner
- Approve step 0 (free) first; it decides the router and the final-pick design.
- Approve the S2 teacher A/B ($5-10) or skip it.
- Approve the O_2 label test at equal money (plain volume against exact), then the volume.
- Trick-1 yardsticks: copper (cheap) + playout yardstick pilot + a capped small anchor, instead of any
  full silver/bronze set.

## 6. Session briefs: one new session and branch per step
How to work (owner, 2026-10-22): each step below is started in its own session on its own branch from
`main`, and pushes back to `main` when done. Every session reads `CLAUDE.md`, `docs/STATUS.md` and this
page first. Before a commit: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo test --workspace`. Results go into `docs/EXPERIMENTS.md` (a new entry), `docs/README.md` (one
line) and `docs/STATUS.md` (the present state). Costs: the build VM is for building and tests; anything
over about 5 minutes runs on vast.ai (`vast-run.yml`, see STATUS "How to run things").

**Getting stored inputs onto the VM.** Workflow `input-fetch.yml` (inputs: space-separated SHA-256 keys)
copies `inputs/sha256/<hex>` out of R2 into an artifact, hash-checked: `gh workflow run input-fetch.yml
-f keys="..."`, then `gh run download <run id> -n inputs -D <dir>`. Keys used by step 0 (all small):
the golden decisions `635e9aef196c816380f3f15c05ede90b38b875fa4817061a57e6d39550065ab0` (1.2 MB), the
golden worlds `6ddc59b191e1fd534f260f9dfe103f742d6026768ca74acdbbb4fd430eb8265e` (92 MB), the trick-2
student S2 `954223e2f56ecade906ba28ad2ea07ff70fee70b841e0743d4c911217afb6ce6` (7 MB), O_3
`f5766aa98a1d769e31122c94cc4da9790c84508fab5c432d0c0358fe3124faad` (17 MB). The equity table is rebuilt
locally in 7 s: `equity build --model data/equity/model-rules-v1-1M-seed1.json --out equity.npy` (its
SHA-256 must be `64740dab...`, `data/equity/equity-v0-rules-v1.sha256`).

### Session 0: the free checks (no rental)
**Done 2026-10-22: results in `docs/EXPERIMENTS.md` E9-E11 and the STATUS verdict. The brief below is kept as built (the sub-deal draws are nested prefixes of one shuffle per repeat rather than one shuffle per N; the router table uses a learned model beside the margin).**
Goal: the three questions of section 3, step 0, answered on the build VM and recorded.
- **0a Shortlist + search at the app's deal counts.** Extend `search student-check` (golden form:
  `--golden-decisions D --golden-worlds W --trick 2`, `crates/search/src/bin/search.rs`, `student_check`)
  with `--sub-deals 10,25,50,100 --sub-repeats R --sub-seed S`. For each decision and each repeat, draw N
  of the 320 exact worlds with `engine::rng::Rng::derive` (seed, decision index, N, repeat) and shuffle;
  pick on the N worlds and **score on the other 320-N** (no winner's curse; best of the complement less the
  pick's value on the complement). Variants per N: the student alone (its choice, scored the same way),
  search over all cards, top-2 and top-3 of the student (`search::curves::top_k_choice`; helpers
  `curves::choose_among`, `curves::totals_over`). Report per place (seat 1 = leader) and per score stratum with the
  existing `student::check::report_measure` bootstrap (1,000 replicates, seed 1). Use the real O_3 and S2
  above. Unit tests on tiny synthetic rows (the same draws for any thread count; N = all worlds reduces
  to the existing top-k loss).
  Question answered: at 25-100 deals does top-3 + search beat S2 alone (0.073%, seat 1 0.18%), and at what N?
  Known context: plain PIMC loses about 0.01% x (243/N)^0.738 at trick 2 (A6).
- **0b Concentration and the router.** From the same pass: the share of S2's total loss held by its worst
  1, 5, 10, 20, 40% of decisions; the share of decisions where S2's top card is the teacher's best; and a
  **router table**: sort decisions by S2's own margin (top-1 minus top-2 value for the decider's team),
  search the lowest f% (top-3 + N deals, as 0a) and keep S2's card for the rest, f = 0, 5, 10, 20, 30,
  50, 75, 100%, with 95% bounds. An oracle router (sorted by true loss) as the upper bound. Question
  answered: can the app spend its 7 s only where S2 is likely wrong?
- **0c Root-restricted solve, finish and measure.** Solver half done (see step 0c above). Remaining: pass
  the candidate cards' mask from `search::pimc::exact_world` (`root_keep` is `0` for burns and other
  non-Play candidates, and equals the full solve when all legal cards are candidates; assert
  byte-identical results for the full list in a test), add an optional `shortlist` request field to
  `crates/wasm` (a contract change: update `docs/WASM-API.md` and the contract test, the engine's
  `engine.wasm` hash), and measure ms a deal at trick 2 (by place) for 2 and 3 kept cards with the real
  O_3 through `engine-native time` (the E8 method, without the local patch). Expected: about the share of
  root cards kept; at trick 1 measure with whatever O_2 exists.
- **Deliverables:** the tool, its tests, an EXPERIMENTS entry with the tables, and a verdict line in
  STATUS (which of the router design and the final-pick design to build).

### Session 1: S2 (A/B tests)
- Teacher A/B: 250k trick-2 decisions at 64 deals with plain H_3 against with G2 (`student_teacher`
  runner chunk; `docs/vast/student-t2-gv2.manifest.json` is the G2 form, its first run failed on a runner
  bug fixed in 6c7acf5; the plain form is `student-t2-base`), same 512-256 network and seed, judged on
  the exact golden set (alone and top-3, 95% bounds) with `student-check`. Estimated $5-10 plus
  training.
- Decision-aware loss: `rank_weight` and `rank_temperature` in `python/nn/fortyone_nn/student.py`, A/B at
  equal data (about $0.5), stop on the validation decision loss.
- The student audit loop stays low priority.
- Record in EXPERIMENTS (Part E): E12 (the A/B, open), E13 (labelling throughput), E14 (keep and takeover).
- Outcome 2026-10-23 (E12-E13): the decision-aware loss is the S2 win; G2 labels gave no gain at 50,000 decisions and the G2 labelling was
  cancelled at 60,000 (12 parts stored: `runs/student-t2-gv2-250k-{4,6,8}/out/rules/`); S2 continues with the plain H_3 teacher.

### Session 2: O_2
- The label test at equal money: plain hybrid labels (`label_source: hybrid`, `label_exact_leaves:
  none`, about 0.1 thread-s each) against exact labels (11.3 thread-s each) for the same dollars, judged
  on the exact validation shards s601, s602, s603 (keys in STATUS) for value SD and team bias, bias
  first. `o2-labeltest-1` was cancelled at 8 minutes; its manifest exists.
- Then volume (O_3 had 40M positions; O_2 has 1.5M) and the audit loop as for O_3, standard 0.01%
  (`docs/m3-hybrid/audit-loop.md`, `rung3.md`). O_2 is judged on exact trick-2 positions, never on
  copper.
- Decision to record: the label source and the volume.

### Session 3: trick-1 yardsticks (after O_2 exists)
- Copper: about 2,000 trick-1 decisions labelled by H_2 + the adopted O_2 at 128 deals
  (`search teacher --trick 1`, runner kind `student_teacher`; measured rate 12,489 decisions an hour on
  256 threads at 64 deals, E6).
- Playout yardstick: design and a pilot (see section 3, step 3); the fixed finishing players are S2 or
  H_3 and then exact.
- A small capped silver/bronze anchor (a few dozen decisions x a few dozen worlds; `docs/m3-hybrid/silver.md`),
  with a dollar cap; the first silver attempt was stopped at about $100 for about 10%.
- No trick-1 golden set (owner, 2026-10-16).

### Session 4: S1
- Teacher labels H_2 + O_2 at 64 deals with racing (extra deals for close calls) and, once session 0
  has it, the restricted solve; about $84 a million decisions at the E6 rate; learning-curve point at
  250k, then 1M. Judge on copper, the playout yardstick and the anchor. **Same loss options as session 1: S2 found the decision-aware
  loss (`rank_weight` 1, `rank_temperature` 0.01, `stop_on` decision) halves the golden-set loss (E12); test plain against that loss for
  S1 at 250,000 decisions, 3 seeds, before spending on more data.**

### Session 5: ship, then the improvements
- Ship S1 and S2 alone (`method: "student"`, `docs/WASM-API.md`).
- Then in order: the restricted solve and the router (session 0's verdict), the pros' disagreement set,
  self-play on thin slices judged by the playout yardstick, public matches. Each starts with a design
  note on this page and a cost estimate for the owner.

### Notes for every session
- Two parts are the assistant's ideas, not published recipes: the **router** and the **playout
  yardstick**. By the project's rule (PLAN §9) duplicate tournaments decide between proven options; flag
  both as proposals in any result.
- Trick-1 time budget: 7 s a decision in the app; a trick-1 deal is about 5x the native time in
  WebAssembly (E8), so the leaf network and the restricted solve decide whether trick 1 fits.
- Machines: `docs/VAST-MACHINES.md` (measured cost per unit of work). Read `usable_cores` in the first
  heartbeat before trusting a machine's listed threads.
