# Review B: scoring, match transitions, money chance and the equity table (2026-10-15)

Branch `review-rB` at `origin/m3-pimc` (ac3c7c6) plus two commits (not pushed):
- `0c39384` search: a rollout that ends the match is certain money for the two-game team (fix + failing test)
- `bd1edb7` equity: independent checks of hand scoring, rotation and the table values (tests only)

Commands run: `cargo test -p equity -p engine -p reference` (baseline, green); after the
changes `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings` (clean),
`cargo test -p equity -p engine -p solver` (green), `cargo test -p search --lib` (54 passed).

## Findings by severity

### 1. Wrong value (latent): rollout leaf names team 0 the match winner when the stake is 0
- `crates/search/src/pimc.rs:217-223` (`leaf_now`, before the fix) read the match winner as
  the team with the largest money entry, ties to the lowest team. With `Rules { stake: 0 }`
  every entry is 0, so every rollout that ended the match returned `certain(0)` whoever won.
  (A negative stake would invert it.) Not triggered by today's runs (`Rules::HOUSE`, stake 5),
  but `stake` is a public, configurable field (H6) and nothing rejects 0.
- Probe: `crates/search/src/tests.rs`
  `rollout_match_end_is_certain_for_the_two_game_team_at_any_stake`; before the fix it failed
  with `left: certain(0)`, `right: certain(1)`.
- Fix `0c39384`: read the winner from `games_won` via `MatchState::winner()`, as the solver
  (`solver/src/lib.rs` `game_won`), the sweep and the test brute force already do.
- Related, loud not silent: `crates/arena/src/block.rs:176` divides by the stake
  (`m % stake`), so an arena run with stake 0 panics. Suggest the engine refuse a stake <= 0.

### 2. Nondeterminism
None found. Table v0 rebuilt here from `data/equity/model-rules-v1-1M-seed1.json` with 1 and
4 threads: both sha256 `64740dab...6808`, equal to `data/equity/equity-v0-rules-v1.sha256`.
The 1,000,000-hand rules-v1 model re-estimated here (seed 1, 3 threads, 47 s) is
byte-identical to the committed model, so the committed model is the current engine and bot.

### 3. Approximation (approved, for the record)
- Table v0's hand-outcome model is stationary and score-blind (M1-SPEC §6.3, "v0"), while the
  rules-v1 bot it was measured from is score-aware (clinch defence). PLAN A1's self-consistent
  table (bots using the successor equities, iterated) is due before the M4 teacher run. Every
  "exact" solver value in the golden run is exact relative to table v0, not to perfect
  equities. The model's `trick_winner` counts include tricks of hands cut short by a game end
  (documented in the model JSON). Both are in the approved plan; no change proposed.

### 4. Waste / weak tests
- `validate::check_rotation_symmetry` (`crates/equity/src/validate.rs:36`) is a tautology:
  `lookup` is defined by the rotation it checks, so it cannot fail. The real rotation checks are
  the Monte-Carlo test (simulator in real team order) and the new independent recursion below.
- `MoneyChance::mix_core` (`crates/equity/src/money.rs:180`) adds `den += w` before its
  `< 2^96` assert; a single `mix_wide` weight near 2^128 would wrap in release before the
  assert. No caller comes near it (sweep weights <= 2^64). Nit.

## Checked and found sound (with evidence)

**Hand scoring vs RULES 27-33 and the end checks** (`equity/src/hand.rs`, engine
`finish_trick`/`end_of_hand`, `game.rs:1076-1205`): trick 5 points; the 5 of trumps' bonus at
once (rule 29); first bonus 6, later 5 per team per game (rule 28, flag = score mod 5 == 1);
41 or more ends the game at once, mid-hand (rule 32), only the scoring team can cross so no
ties; after trick 5: trick points -> 41 -> jink -> end bonus to the team that played the
highest trump (it always wins its trick) -> 41; a pending end bonus is lost when the game ends
earlier; rule 31 no bonus. Evidence: new `in_hand_scoring_equals_rules_by_hand` (RULES
re-written by hand vs `InHand`, 120 start scores x 1024 trick sequences x 6 positions of the 5
x 4 bonus teams), `hand_computed_scoring_cases` (11 hand-computed cases), existing engine
`rules_by_number` tests 27-36, H5, H6, and `engine_parity_on_self_play_hands`.

**Match transitions at the solver leaf** (`solver/src/lib.rs:417-436`, `fast.rs:337-374`):
game won -> games+1, scores and flags reset, dealer team +1 (H5), 2 games -> `certain`; hand
over -> scores after the end bonus, dealer team +1. `from_match` takes scores, explicit flags,
tricks, the 5 (completed tricks only), best trump and dealer from the engine. Covered against
the engine by the brute-force tests with a deliberately rotation-asymmetric leaf (`brute.rs`,
`fast.rs`), and by the new recursion test.

**Money chance** (`equity/src/money.rs`): u128 accumulation, power-of-two shift path equals
division, largest remainder (ties to the lower index), sum exactly 2^32 after every
constructor; `rotate` is a relabelling; `expected_money` = (8p - 2) x stake (H6, PLAN 5.3).
New property test `money_chance_sum_invariant_property` (3,000 random `mix_wide` vectors,
rotations, chained mixes; every share floor or ceiling). TT packing (`solver/src/tt.rs:113-154`)
is lossless (a non-maximal entry is at most 2^31).

**Table index and rotation** (`equity/src/state.rs`): the state (15 scores per team, games
0/1 per team, dealer team) is complete for a stationary model (dealer seat within the team is a
true symmetry; only cyclic rotations are used, correctly not reflections). In: rotate by
4 - d; out: `rotate(d)`. New `table_equals_independent_recursion_*`: canonical index and
rotation written by hand, each sampled state (random dealer team) recomputed by plain
recursion over every trick sequence from the model and the table's successor entries; the
synthetic table (300 states) and the **real table v0** (3,000 states, release, ignored test
with `EQUITY_TABLE`/`EQUITY_MODEL`) agree within 1e-7 and `lookup` equals the hand-made
rotation bit for bit. This checks the sweep's dynamic program, scoring, dealer and game
transitions and rotation in one go.

**Table used by runs**: `runner/src/spike.rs:296-466` accepts a table only if its sha256 is the
recorded one (manifest path, `inputs/sha256/<hex>` in R2, or rebuilt from the repository's
model), unless `allow_unverified_table`; format-2 inputs are verified by hash on download
(`runner/src/v2.rs:586-601`); search worlds files carry the table sha in their fingerprint. No
`tables/<hands>/<seed>` path exists on this branch.

**Reference** (`crates/reference`): independent naive scoring, bonus, jink, end-bonus,
game/match end, H5 dealer move and money; `crates/difftest` compares scores, games, bonus flags,
money, dealer and every event after every action. Not covered by the reference: `equity::hand`
(`InHand`), the solver's `Position::play`, the sweep and `MoneyChance`; these are checked against
the engine (parity tests) and now against the independent recursion, not against the reference.
