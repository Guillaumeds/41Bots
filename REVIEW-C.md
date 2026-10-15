# Review C: the trick-1 yardstick (sampling, resume, aggregation, curves)

Branch `review-rC` (from `origin/m3-pimc` at ac3c7c6). Fix commit **52c0253** (not pushed).
All probes are in `crates/search/src/review_c_tests.rs`. Run with `cargo test -p search --lib review_c_tests`, about 15 s.

## Findings, by severity

### 1. Non-reproducible: worlds valued by an older binary were resumed, extended and joined silently (FIXED)
- **Where:** `crates/search/src/measure.rs`, the `RowFp` / `CkptRow` fingerprints (`read_ckpt` around line 1411, `check_fingerprints` around line 1791, `merge_rows` around line 1852).
- **What was wrong:** the fingerprint held the seed, the decision, the world source and the equity-table hash, but nothing about the code that values a world. Tonight's commits changed the values of every world where another seat robs later in trick 1:
  - 13d3858 and 65b6a96: that seat's burn is now searched by the solver instead of chosen by `rules-v1`;
  - 7253b48: a solver bug fix.

  A checkpoint or worlds file uploaded by an earlier run (STATUS plans `b5` to resume `b3` and `a9` to resume `a7`), resumed or extended on the new binary, was accepted as the same run. It would have mixed two different yardsticks in one file with no warning. `merge_rows` also joined a record that had no fingerprint to one that had.
- **Probes (both failed before the fix):**
  - `stale::checkpoint_from_an_older_binary_is_refused`: an old-format checkpoint line was resumed, and the output file differed from a fresh run's.
  - `stale::records_from_an_older_binary_are_not_extended`: old records were resumed, and were also joined to new worlds 3..5.
- **Fix:** `VALUES_VERSION = 2` (measure.rs:398) is now written into every record fingerprint and every checkpoint line. A resume, extension or join with another version, or with no version, is refused by name. Records that have no fingerprint can no longer be joined to records that have one. **Bump the constant** whenever the solver, the leaf rule, the world generator, the rebuild or the candidates can change a recorded value.
- **Consequence:** all checkpoints and worlds files made before this commit can no longer be resumed or extended. They must be redone, and that is correct: their values came from the old burn rule.

### 2. Statistical weakness (it biases the decision population): plays straight after the decider's own rob burn were drawn as decisions (FIXED)
- **Where:** `build_decisions` (measure.rs:273); the new filter is `play_after_own_rob` (measure.rs:336).
- **What was wrong:** a seat that robs at its turn in trick 1 has two decision points at its position:
  - the joint burn-and-play decision;
  - the play straight after its burn.

  The bot searches only the first (M3-SPEC §5.5). In a 256-decision trick-1 set (seed 1), 11 decisions were these post-burn plays, against 18 joint decisions. So about 40% of the robber decisions drawn were moves the bot never makes, and the "robber" band and the position cells were skewed.
- **Probe:** `decision_set_robber_nodes` failed before the fix (11 found) and passes now.
- **Consequence:** a decision set built with the new binary and the same seed differs from an old one. Decision fingerprints keep the two from being mixed.

### 3. Statistical weaknesses (reported, not changed)
- **Extrapolating the curve fit:**
  - The golden N comes from a power-law fit over N ≤ K/2 (curves.rs `analyse_trick`), extrapolated to the 0.01% threshold.
  - Each decision's own loss falls off exponentially once its best card stands out, so a power law can be pessimistic or optimistic beyond the measured N. The held-out and half-data checks (`heldout_inside`, `half_inside`, `b_by_range`) must be read before trusting N > K.
  - `fit_points` drops grid points whose mean is not positive. That pushes the fitted a up (more worlds than needed, so it errs on the safe side).
  - A bootstrap replicate with no positive point gets N = 1, which errs the other way, but this is rare.
- **"95% upper bound":** this is the 97.5th percentile of a bootstrap over decisions (one-sided 97.5%), so it is a little conservative. That is fine, but state it in the report.
- **`read_jsonl` (measure.rs:1248):** it skips any line it cannot read, not only a torn last line. A corrupt line in the middle of a worlds file makes `agree`/`curves` drop that decision with no message.
- **`score` (measure.rs:2113):** it matches yardstick and method records by id alone. Ids repeat across tricks, so a mixed-trick yardstick file could be matched wrongly. It panics if the candidates differ, so it is only silent when the candidates happen to be equal.

## Checked and found sound (evidence)
- **World sampling at trick 1** (`trick1_belief_marginals_match_engine_oracle`):
  - **Views:** 48 hands, 6,510 tests (card by place). They cover:
    - a pending robber, a rob at turn, a turned Joker then an ordinary card (dealer may rob 8th, Joker burn may be the future ace), Joker then ace, and a turned ace;
    - every position in the trick, including the decider's own rob.
  - **Exact against sampled:** for the ace of trumps, every taken card, the Joker and two other cards, the exact probability of each place came from the belief counter (by restricting one card's places). The belief sampler and the engine-based rejection oracle both matched it, 5,000 draws each; worst |z| 3.75.
  - **Nothing missing:** every rejection-oracle world also passed `Tracker::consistent`, so the tracker misses no trick-1 fact and wrongly forbids nothing.
  - **Uniform prior:** uniform over placements is the right prior, since a rob or Joker burn always has a fixed number of legal burns (5 or 6).
- **Rebuild:** `same_view` checks the scores, games, dealer, leader, trump, best trump and turned/taken cards against the real view, and `world_rebuild_view_identity` exercises it. Reviewed: `from_match` gives `MatchCtx` (dealer team, games) and the leader, and `solver_position` sets the pending other robber's `Rob` (never the decider: its own rob is resolved first).
- **Candidates:** `joint_candidates_are_exactly_the_engines`: over 400 deals, the joint candidates equal the engine's legal burns × its legal plays after each burn.
- **Aggregation:**
  - Fixed-point `MoneyChance::mix` is used in world order, and `mean_of_totals` in curves (tested equal).
  - The loss is in the decider's team entry, against the reference choice, with ties to the first candidate.
- **Split-half referee:** an independent Python simulation (3 candidates, Gaussian, M = 64) gave E[cv(N)] = L(N) − L(K) within 1.5 SE at every N. This agrees with `cross_fitted_is_unbiased`, and the fit recovers a synthetic power law exactly.
- **Thread, schedule and chunk independence** (`belief_worlds_files_identical_under_threads_schedules_and_chunks`, with the production belief source): the worlds files are the same bytes on 1 and 4 threads and under the position-aware plan, and each record is identical when the decisions run in two chunks. World w depends only on (seed, label, id, w).
- **Determinism:** no `HashMap` iteration (belief memo lookup only, FNV) and no floats in the value path (floats only in plan/ETA/curves).
- **Not covered:** exact trick-1 solves themselves (too slow for this VM). The solver review owns parallel-solve identity with `Rob` nodes.
