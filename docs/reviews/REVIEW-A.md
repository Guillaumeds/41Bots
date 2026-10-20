# Review A: exact solver correctness (2026-10-15, branch review-rA from origin/m3-pimc ac3c7c6)

Scope: `crates/solver` (lib.rs naive oracle, fast.rs, tt.rs, equiv.rs, denial.rs, worlds.rs,
tests) and its callers in `crates/search/src/pimc.rs`. Every finding below has a test; new
probe tests are in `crates/solver/tests/probe.rs`.

## Findings (by severity)

### 1. WRONG ANSWER (fixed): equivalent-card merging ignored the tie order
- Where: `crates/solver/src/equiv.rs` (old `emit`, line 301-324 at ac3c7c6; the rule in the
  module notes, line 25). Used by `fast.rs` `candidates_of` (390), `burn_node` (614) and
  `root_groups` (746), so the sequential, parallel and burn paths were all affected.
- What: two rank-adjacent cards of one class in the mover's hand were treated as
  interchangeable. The proof swapped them everywhere, but forgot that the naive solver breaks
  ties by **card index**, and a tie for the mover's team can hide different values for the
  other teams. If another card of the same hand lies between the two in card index, keeping
  one or the other breaks a later tie differently.
- Example (seed 52, trick 3 + 5 plays, clubs trumps, mover seat 3 holds 4C 5C 7C, 6C gone):
  4C and 7C were merged; the 5C lies between them in index. Naive values: 4C =
  [0.2, 0.4, 0.2, 0.2], 7C = [0.5, 0.5, 0, 0]; fast copied the 4C value onto the 7C, so the
  **root mover's own entry** was wrong (0.2 instead of 0), not only the others'.
- Fix: commit `64738d4`. A group is now also cut wherever another card of the hand lies between
  two rank-consecutive members in card index (hands only lose cards later; the robber's burn
  groups use its hand with the turned card). The test oracle `groups_walk` got the same rule.
  Cost: about 3% more nodes (40 trick-3 roots: 1.477M -> 1.517M nodes; 4 trick-2 roots:
  67.1M -> 68.9M).
- Tests: `merged_cards_have_identical_naive_values` (checks the merging claim itself against
  the naive solver: failed before the fix), `identity_when_tie_order_separates_merged_cards`
  (fast, all 16 combos and parallel vs naive at that position: failed before, passes after),
  negative decks added to `equivalent_cards_negative_decks` in `tests/fast.rs`.
- How often it bites: rarely with the real equity table. `real_table_identity` (ignored; needs
  `PROBE_EQUITY=<table>`) found 0 differences in 4,000 random tricks-3-to-5 roots and 3,000 more
  with games won and scores >= 30, before the fix. It needs an exact tie for one team with
  different vectors, which the real table rarely produces (certain-money leaves late in a match
  are the likely source). Note: the random fast-vs-naive sweeps did NOT catch it, even with a
  degenerate leaf over 600 positions (`extreme_leaf_sweep_long` passes on the old code); only the
  direct claim check did. Worth keeping that style of test for every future merge rule.
- Design note for the owner/Fable: an alternative that keeps every merge is to make the tie order
  rank-based (by class and rank instead of card index) in both solvers and in PIMC. That changes
  the documented tie rule (M2, M3-SPEC 5.5), so it is a design decision, not a review fix.

### 2. Nondeterminism: none found.
### 3. Waste / diagnostics (minor, not fixed)
- `fast.rs` `node()` never counts `decision_nodes` or `tied_nodes` (only `burn_node`, line 647,
  counts decision nodes), so these two counters are not comparable with the naive solver's. They
  are diagnostics only; no answer depends on them.
- `tt.rs` `Packed::pack` (line 129) only debug-asserts that a value is valid. An invalid leaf
  value (sum not 2^32) would be silently "repaired" by the table, so table on/off would differ.
  The equity table checks every entry on load (`from_npy`), so this needs a broken leaf; noted only.

## Checked and found sound (with evidence)
- **Naive oracle rules** (legal plays via `engine::rules`, trick winner, best-trump tracking, 5's
  immediate bonus, end bonus 6/5 by flag, 41 at any point, jink after the 41 check, pending bonus
  lost on an early game end, two teams at 41, game won -> games+1, scores reset, deal moves left,
  second game = certain money): `naive_known_answers_last_trick` (10 hand-made last tricks with
  known end states), `naive_known_answers_two_cards` (max^n team backup, trump-led forcing,
  "suit or trump" on a plain lead, black trump order), `late_match_naive_matches_engine_brute_force`
  (up to 160 late-match positions at tricks 4-5, 3 leaf tables, every root child vs the engine brute
  force; many certain-money leaves).
- **Tie rule and backup** (first strictly best in tie order, team entry maximised): naive vs
  engine brute force above, plus `same()` also compares `root_tied`.
- **TT key** (trick boundaries, no pending rob; cards-left word + leader, tricks per team, 5
  played, best rank/team, flags; scores derivable from the root): sound by argument; probed by all
  identity tests with `tt_mb` 0 (one bucket: constant replacement) and 64, both orders.
- **Packed value encoding**: lossless proof checked; existing `packed_values_round_trip`; the
  degenerate leaf (certain wins, exact halves, 1-unit values) through every identity test.
- **Shallow pruning and certain-win stop**: bound logic re-derived (strict/non-strict cases,
  filter when parent team equals child team, forced chains, burn nodes ignore the bound);
  degenerate leaf hits the equality branches: `extreme_leaf_identity_tricks_3_to_5`,
  `game_won_mid_hand_identity`, `late_match_identity_*` (all 16 combos, both orders).
- **Forced-play chains, last-trick shortcut, leaf cache**: same identity tests; leaf cache compares
  the whole key.
- **Pending rob / burn merging**: `pending_rob_identity_with_degenerate_leaves` (any dead card
  turned, incl. Joker, robber given the ace of trumps on even seeds), `merged_burns_have_identical_naive_values`.
- **Parallel solver**: same value, chosen card and children for 1-4 threads and split sizes 1, 3,
  64, three switch combos, both orders: `par_identity_threads_and_splits_late_match`; also inside
  every `identity()` call.
- **Table reuse / replacement / generation wrap**: existing tt.rs tests (slot-for-slot oracle);
  tiny tables in every probe identity.
- **Fixed point**: backups only copy values (no arithmetic), so sums stay exactly 2^32:
  `every_reported_value_sums_to_one`. No HashMap iteration (naive table is lookup-only), no
  floats in the solve path (only `TableMemory::peak_factor`, a diagnostic).
- **pimc.rs callers**: `solver_position` sets `Rob` only for a seat other than the mover; the
  mover's own burn is solved per burn after `Action::Burn`; `from_match` detects the pending rob
  correctly after a taken Joker (engine: taking the Joker is not a rob).

## Probe tests added (`crates/solver/tests/probe.rs`)
naive_known_answers_last_trick, naive_known_answers_two_cards,
late_match_naive_matches_engine_brute_force, late_match_identity_tricks_4_and_5_every_combination,
late_match_identity_trick_3, game_won_mid_hand_identity, extreme_leaf_identity_tricks_3_to_5,
merged_cards_have_identical_naive_values, identity_when_tie_order_separates_merged_cards,
merged_burns_have_identical_naive_values, pending_rob_identity_with_degenerate_leaves,
par_identity_threads_and_splits_late_match, every_reported_value_sums_to_one;
ignored: real_table_identity, equiv_cost_measurement, extreme_leaf_sweep_long.
`cargo test -p solver`: all pass (probe file about 45 s at the test profile); `cargo test -p search`
passes; fmt and clippy -D warnings clean.
