# Review D: are the engine's rules exact? (2026-10-15, branch review-rD)

Scope: `docs/RULES.md` (rules 1–41, H1–H7, interpretation notes and owner decisions), `docs/M0-SPEC.md`,
`crates/engine` (cards, rules, game, hand_start, view, rng), `crates/reference`, `crates/difftest`,
the engine tests, and the hidden-card assumptions in `crates/search/src/worlds.rs` and
`crates/belief/src/tracker.rs`. Also checked: the in-hand scoring the solver uses
(`crates/equity/src/hand.rs`, `crates/solver/src/lib.rs` `Position::play`), since solver values
rest on it.

## Verdict

No rule of the game is implemented wrongly. I found no legal action the engine forbids, no illegal
action it allows, and no wrong state change (rob, burn, Joker, the 5, trick winner, trumps,
scoring, game end, match end, dealer, leader, hand number). One low-severity validation gap was
found and fixed (L1). The differential test passed 150,000 matches.

## Findings by severity

| Sev | Where | Finding | Probe test | Fix |
|---|---|---|---|---|
| Low | `crates/engine/src/game/hand_start.rs:52` (`HandStartState::validate`) | Accepted the unreachable scores 1 and 6 (both `≡ 1 mod 5`). RULES "Reachable scores": a score is `5k` or `5k+6` with `k ≥ 1`. No real match or view can produce them; only a hand-built teacher/stratified start state could. | `crates/engine/tests/review_d_probes.rs::hand_start_refuses_the_unreachable_scores_one_and_six` (failed before the fix) | one-line check `s == 1 \|\| s == 6`; commit 7ec127b on review-rD |
| Info | `crates/engine/src/rules.rs:211` (`beats`) | Used by the fast solver's pruning but had no direct test. Now checked exhaustively against `trick_winner` (all 4 trump suits × every led card × every 2nd and 3rd card; >500k cases): it agrees. | `review_d_probes.rs::beats_agrees_with_trick_winner_exhaustively_on_short_tricks` | none needed |
| Info | `crates/solver/src/lib.rs` (`Search.tt`), `crates/belief/src/space.rs` (`Memo`) | `HashMap`s, but only for lookups (never iterated), the belief one with a fixed FNV hasher. Deterministic. | – | none |

## Rule checklist (coverage)

Tests: `RBN` = `crates/engine/tests/rules_by_number.rs`, `T` = `crates/engine/tests/transcripts/*.txt`
(run by `transcripts.rs`), `P` = `crates/engine/tests/properties.rs`, `D` = my new
`crates/engine/tests/review_d_probes.rs`, `REF` = `crates/reference/src/**` tests, `DT` = difftest.
Every rule has a direct test; none were uncovered. Where I judged the coverage thin for a
combination of rules, I added a probe (D).

| Rule | Covered by | Note |
|---|---|---|
| 1 seats, partners opposite | RBN `rule_01_eight_players_in_pairs`, `decision_table_of_eight_in_four_partnerships` | |
| 2 53 cards | RBN `rule_02_one_pack_of_53_cards` | |
| 3 first dealer random, deal moves left | RBN `rule_03_…`, `decision_first_dealer_drawn_once_and_fresh_deck_every_hand` | |
| 4 deal 2 then 3 from dealer's left | RBN `rule_04_…`; D `layout` (own deck layout) | |
| 5 turned card fixes trumps | RBN `rule_05_…` | |
| 6 trump order red/black | RBN `rule_06_…` (`assert_order`, every pair in a real trick); engine unit `trump_order_red_and_black` | |
| 7 Joker, A♥ always trumps | RBN `rule_07_…` | |
| 8 hearts trumps: 14 trumps, A♥ 4th | RBN `rule_08_…`; D `hearts_trumps_rob_then_renege_against_the_king` | |
| 9 red plain order, ace lowest | RBN `rule_09_…`; D `the_five_of_a_plain_suit_is_an_ordinary_number` | |
| 10 black plain order | RBN `rule_10_…`; D same | |
| 11 A♥ never a plain heart | RBN `rule_11_…`; T 22 | |
| 12 left of dealer leads, then winner | RBN `rule_12_…`; `check_tricks` in random matches | |
| 13 any card may be led | RBN `rule_13_…` | |
| 14 follow plain suit or trump | RBN `rule_14_…`; `rulebook_legal` oracle in `rule_37` random matches | |
| 15 void: any card | RBN `rule_15_…` | |
| 16 trump led: must trump, top four may renege | RBN `rule_16_…`; D hearts-trumps probes | |
| 17 higher trump led forces | RBN `rule_17_…`; T 19, 20, 23; D `hearts_trumps_ace_of_hearts_led_forces_the_king_not_the_joker` | |
| 18 only holdable trumps: anything | RBN `rule_18_…`; T 21; D `joker_turned_then_the_leader_robs_before_leading` | |
| 19 trick winner | RBN `rule_19_…`; T 24, 25; P `naive_winner`; D `beats_…` | |
| 20 ace holder robs before first card | RBN `rule_20_…`, `house_h2_…`; T 10, 11, 17 | |
| 21 ace of trumps never burned; burn hidden | RBN `rule_21_…`, `house_h3_…`; T 13 | |
| 22 turned ace: dealer robs at once | RBN `rule_22_…`, `house_h4_…`; T 08, 09 | |
| 23 ace undealt: nobody robs | RBN `rule_23_…`; T 07, 18 | |
| 24 turned Joker | RBN `rule_24_…`, `house_h4_…`, `decision_two_takes_in_one_hand`; T 14, 15, 16, 17, 18; D `joker_turned_then_the_leader_robs_before_leading` | |
| 25, 26 penalties for not robbing | RBN `rule_25_…`, `rule_26_…` (forced take; refusal) | H1 |
| 27 trick = 5 | RBN `rule_27_…` | |
| 28 best-trump trick 11, later 10 | RBN `rule_28_…`, `decision_bonus_six_then_five`; T 03, 04 | |
| 29 5 of trumps: bonus at once | RBN `rule_29_…`; T 01, 06 | |
| 30 end-of-hand bonus to highest trump | RBN `rule_30_…`; T 02, 05; D `end_bonus_goes_to_the_trick_with_the_highest_trump` | |
| 31 no trump, no bonus | RBN `rule_31_…`, `decision_rule_31_cannot_occur_in_a_completed_hand`; DT branch count 0 | |
| 32 41 ends the game at once | RBN `rule_32_…`, `decision_reaching_41_means_41_or_more`, `decision_order_of_checks_…`, `decision_pending_bonus_lost_…`, `decision_two_teams_over_41_…` | |
| 33 jink | RBN `rule_33_…`; T 26 | |
| 34 all others turned | RBN `rule_34_…` | H1: cannot occur |
| 35 match = first to 2 games | RBN `rule_35_…` | |
| 36 scores reset, games kept | RBN `rule_36_…`; T 04 | |
| 37–41 penalties | RBN `rule_37_…` … `rule_41_…` (illegal actions refused, match unchanged) | H1 |
| H1 illegal plays impossible | RBN `house_h1_…`, `rule_37_…` (every offered action checked against the rule-text oracle; every other card refused) | |
| H2 rob at own first turn in trick 1 | RBN `house_h2_…`; T 10, 11 | |
| H3 any card but the ace of trumps may be burned | RBN `house_h3_…` (both tests); T 13, 14; engine unit `any_card_may_be_burned` | |
| H4 dealer robs at once (amended) | RBN `house_h4_…`, `decision_dealer_holding_the_ace_robs_as_the_eighth_card`; T 15, 16 | |
| H5 deal moves left after a mid-hand game end | RBN `house_h5_…` | |
| H6 money | RBN `house_h6_…` | |
| H7 money is the objective | RBN `house_h7_…` | (not a rule of play) |
| Notes: reachable scores, flag = `score mod 5 == 1` | RBN `decision_reachable_scores_and_the_bonus_flag`; D `hand_start_refuses_…` (L1) | |
| Notes: Joker / A♥ leads are trump leads | RBN `decision_joker_and_ace_of_hearts_leads_are_trump_leads` | |
| Notes: ace facts known incrementally | RBN `decision_ace_of_trumps_facts_known_incrementally` | |
| Notes: card counts, hidden 12 or 13 | `decision_two_takes_in_one_hand`; difftest compares `hidden_count` every action | |

## Reference implementation and difftest

- **Independent?** Yes. `crates/reference/Cargo.toml` has no dependencies; nothing in it names the
  engine. It uses enums for suit and rank, `Vec<Card>` hands, every order written out as a list
  best-first (no rank arithmetic), and computes the end-of-hand bonus a different way (it searches
  the finished tricks for the best trump and gives the bonus to *that trick's winner*, while the
  engine tracks the best trump as cards are played and gives it to the *player's team*). The two
  agree because the highest trump in the hand always wins its own trick (rule 19).
- **Does difftest compare everything after every action?** Yes: after every action it compares
  phase, seat to act, the full legal-action list, dealer, hand/game index, scores, games, bonus
  flags, money, every field of `PublicState` (trump, table card, turned cards and takers, Joker
  flag, hand sizes, burn counts, burn order, hidden count, cards played, trick number, leader,
  current trick, trick winners, tricks per team, best trump, five played, robber, proved-no-ace,
  ace-undealt), every seat's hand and burns, and the undealt pack. Each seat's full event list is
  compared at every deal and at the match end (logs only grow, so every event is compared), and
  on any difference it reruns comparing everything after every action. It also replays each match
  from seed and actions. The one thing it does not exercise is `apply` refusing an illegal action;
  that is covered because both sides refuse exactly what is not in their (compared) legal lists,
  and by RBN `rule_37`/`house_h1`.
- **Result (this review):** `difftest 150000 20261015 --threads 2`: **150,000 matches, 482,703 games,
  1,765,538 hands, 64,944,598 actions, zero disagreements, every match replayed identically,
  coverage minimums met, PASS** (361 s). Branch rates match the combinatorics (e.g. Joker then ace
  0.144% vs 1/53 × 4/52 = 0.145%; non-dealer robbed 62.2% vs 49/53 × 35/52 = 62.2%).

## Shared assumptions (both implementations follow M0-SPEC; a shared misreading would pass difftest)

| # | Shared reading | Rule text | Verdict |
|---|---|---|---|
| 1 | Deal: 2 then 3 from the dealer's left; after a turned Joker the next turned card is deck position 41 | Rules 4, 24 ("other bundles are fine") | Right |
| 2 | Robber robs at their own first turn in trick 1, after seeing earlier trick-1 cards; a robbing leader robs before leading | H2 | Right |
| 3 | Turned ace (and an ace turned after a Joker) robbed by the dealer at once, before trick 1; after Joker + ordinary card, a dealer holding the ace robs as the 8th card | Rules 22, 24, H4 (amended 2026-10-14) | Right |
| 4 | Burns: anything but the ace of trumps (aces, A♥ when not trumps, the Joker, the card just taken); the Joker-take burn: anything | Rules 21, 24, H3 | Right |
| 5 | Dealer's pre-trump burn may hold the future ace; then nobody robs, card stays up | H3 (owner, 2026-10-14) | Right |
| 6 | Reneging: a trump may be held back iff it is one of 5, J, Joker, A♥ and ranks above the led trump; the A♥ is a top trump also when hearts are trumps | Rules 16–17 list the A♥ unconditionally | Right |
| 7 | Joker led and A♥ led (hearts not trumps) are trump leads | Rule 17 ("The Joker led forces…", "The ace of hearts led forces out the ace of trumps…") | Right |
| 8 | When forced to trump, any trump may be played, held-back ones included | Rule 16 ("must play a trump") | Right |
| 9 | End-of-hand bonus goes to the team of the seat that played the highest trump | Rules 28, 30 say "the trick holding" it; that trump always wins its trick (rule 19) | Right (identical) |
| 10 | Order after a trick: 5 points → 5-of-trumps bonus → 41; after trick 5: jink → end bonus → 41 | RULES "End checks", owner defaults | Right; jink vs 41 can only differ in the reason label, never the winner |
| 11 | A pending end bonus is lost if the game ends earlier | Rule 32 (default) | Right, and money-neutral: scores reset when the game ends |
| 12 | Two teams over 41 after trick 5: the trick's winner wins | Rule 32 (default) | Right (the trick is scored before the end-of-hand bonus) |
| 13 | Money: winner +6 stakes, each loser −2 stakes | H6 | Right |
| 14 | "Left" = seat + 1; play and the deal move that way | Interpretation note | Right |
| 15 | Hand index counts across games; dealer moves left after every hand, also after a game ends mid-hand | Rule 3, H5 | Right |

## Robbing, burning and what others can know (rules 20–24, H2–H4) vs search and belief

- Engine: the burned card is a `Private(seat)` `Burned` event plus a public `BurnDone`; only the
  burner's view has it (`own_burns`). Public facts: who took which turned card (`Took`,
  `turned`), the kind of each burn (`burn_order`), `robber`, `ProvedNoAce` per seat, `AceUndealt`.
  The pending robber is never exposed before its turn.
- `crates/search/src/worlds.rs` (`Hidden::of`, `RejectionSource`): treats as known only own hand,
  own burns, cards played and the face-up turned card; fixes the ace of trumps in the robber's hand
  (correct: it can never be burned and, if played, it is known); then rejects any world whose
  rebuilt match does not show the viewer exactly the same view. Correct by construction.
- `crates/belief/src/tracker.rs`: taken cards stay with the taker (hand, or a burn made after the
  take); the ace of trumps is never in a burn made with trumps known; it may be in the dealer's
  Joker-take burn only while nobody has robbed; a seat that played trick 1 (or is asked to play
  it) without robbing lacks it; voids come from `legal_plays` card by card (exact, since legality
  is decided card by card and a seat never gains a card after its first play). These match the
  engine and the rule text. No missing or wrong public fact found.

## Determinism

- RNG: `tools/rng_golden.py` regenerated today gives a file byte-identical to
  `crates/engine/tests/data/rng_golden.txt`; the golden test passes; SplitMix64(0) first output is
  the published 0xE220A8397B1DCDAF; xoshiro256** matches the published algorithm.
- No floats in engine, reference, belief or solver code (only in belief's statistical tests).
- `HashMap` only for lookups (see Info row above). Engine and reference use arrays/`Vec` only.

## Questions for the owner

None found that change play or money. Every point the rule text leaves open is already settled by
section H or by an owner/default decision, and I agree with each reading (table above).

## Commands run

- `cargo run --release -p difftest -- 150000 20261015 --threads 2` (PASS, above)
- `cargo test --release -p engine --test review_d_probes` (7 tests; L1 probe failed before the fix); the required suites all pass (exit 0)
- `cargo fmt --all`; `cargo clippy --workspace --all-targets -- -D warnings`;
  `cargo test -p engine -p reference -p difftest -p solver -p search`
