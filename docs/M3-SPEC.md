# M3 specification: card tracker, world generator, PIMC teacher, suitability measurement, full runner

Derived from `docs/PLAN.md` (amendments A3, A4, A5, A6, A8, A9; §6.3, §6.4, §6.8, §6.9, §8 phase A, §9, §12, and the revised §13 M3 row), `docs/RULES.md` (rules 20–24, H2–H4 and the interpretation notes), and the measured solver costs in `docs/M2-SPIKE.md`. Where this file and PLAN.md or RULES.md disagree, they win and this file must be fixed. M0–M2 are the base: `crates/engine` (`Match`, `SeatView`, `rules::legal_burns`, `legal_plays`), `crates/bot`, `crates/bots` (`rules-v1`), `crates/tablemgr`, `crates/arena`, `crates/equity` (`MoneyChance`, `EquityTable`, `MatchState`), `crates/solver` (`solve_fast`, `from_match`), `crates/runner`, `python/launcher`.

**M3 gate (PLAN §13):** PIMC beats the rules bot decisively on the 14-way block (GSPRT, section 8.3); the PIMC-suitability properties are reported per trick (section 6); the resume test passes (section 7.4).

Terms used below, in plain words:
- **world** [one complete guess of where every hidden card is, consistent with everything the deciding seat has seen];
- **PIMC** [Perfect-Information Monte Carlo: score each legal action in many worlds as if all cards were face up, average, play the best];
- **rollout** [play the action, then let simulated players finish the hand];
- **hard fact** [something certain from the rules and the cards seen, e.g. "seat 3 has no clubs"].

## 1. Shared conventions

- Everything in M1-SPEC §1 still applies: integer logic, no `HashMap`/`HashSet` iteration in anything that affects play, no `unsafe`, randomness only from `engine::rng::Rng` by labelled derivation, JSON Lines and `.npy` formats, content-addressed immutable files.
- **The money type.** Every value produced by tracker-, world-, PIMC- or teacher-code is an `equity::MoneyChance` (four `u64` entries in units of 2^-32 summing to exactly `ONE = 2^32`; CLAUDE.md's "u32 per team" is the same scale, stored widened so the sum fits). Averages over worlds use `MoneyChance::mix` / `mix_wide` with the terms in **world index order**. Floats appear only in reports (SPRT, standard errors, plots).
- **Hash maps.** A memo table that is only *looked up* (never iterated to produce output) may use our own fixed hash (as `solver::tt` does); `std::collections::HashMap` with its random seed is still forbidden in these crates.
- **Thread independence.** Parallel work writes into pre-sized slots indexed by world (or hand, or block) and is reduced sequentially in index order. Every public function that takes `threads` must return byte-identical results for 1, 2 and 4 threads (tested).
- **Labels for seed derivation** (new constants, in `belief::labels` and `search::labels`): `WORLD = 0x3071D`, `ROLLOUT = 0x2011`, `DECISION = 0xDEC1`, `TEACHER_HAND = 0x7EAC`, `STRATUM = 0x5742`, `MEASURE = 0x3EA5`, `AGREE = 0xA62E`.
- **Crate dependencies** (new crates follow PLAN §13's layout):
  - `crates/belief`: `engine`, `equity` (only for `MatchState` in tests), no others;
  - `crates/search`: `engine`, `bot`, `bots`, `equity`, `solver`, `belief`, `serde`, `serde_json`;
  - `arena` gains a dependency on `search` (for the bot registry, section 8.1).
  - Nothing else without the reviewer's agreement. `libm` is allowed in `search` for the optional weighting arm only (section 5.2).

## 2. Engine additions (small, tested, in `crates/engine`)

1. **Start a hand from a match state.**
   ```rust
   pub struct HandStartState { pub dealer: u8, pub scores: [u8; 4], pub bonus_taken: [bool; 4],
                               pub games_won: [u8; 4], pub hand_index: u32, pub game_index: u8 }
   impl Match { pub fn from_hand_start(s: HandStartState, deck: [Card; DECK_SIZE], seed: u64, rules: Rules)
                    -> Result<Match, StateError>; }
   ```
   It validates the state (reachable scores; `bonus_taken[t] == (scores[t] % 5 == 1)`; no team at 41 or more; games won ≤ 1 each (any number of teams may hold one game while the match continues); `dealer < 8`) and deals `deck` as hand `hand_index`. Later hands are shuffled from `Rng::new(seed)`. `MatchRecord` gains an optional `start: Option<HandStartState>` (format version 2; version-1 records still read) so a record starting mid-match replays.
   - Test `from_hand_start_equals_played_match`: for 500 seeds, play a match with `rules-v1` to the start of some hand, rebuild it with `from_hand_start` from that hand's public state and deck, then play both on: every event and every `SeatView` agree.
2. **Hidden burns in a view.** `PublicState` already has `burn_counts`; add `burn_order: Vec<(u8 /*seat*/, BurnKind)>` with `enum BurnKind { JokerTake, AceRobAtOnce, RobAtTurn }`, in burn order, so the tracker can tell the dealer's pre-trump burn (`JokerTake`) from a rob burn without parsing events. Covered by the difftest (the reference derives the same field) and a transcript check on transcripts 15–18.

No other engine change. The tracker must take its rules facts from the engine's own functions (`legal_plays`, `legal_burns`, `ace_of_trumps`, `trump_mask`, `rules::effective_suit`, the `ProvedNoAce`/`AceUndealt` events), never from re-derived copies (A6).

## 3. `crates/belief`: the card tracker (A6, PLAN §6.3)

### 3.1 Locations
A hidden card can be in one of up to 18 **locations**:
```rust
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Loc { Hand(u8 /*seat*/), Undealt, Burn(u8 /*seat*/, BurnKind) }
```
Each location has a **capacity** known to every seat: `Hand(s)` = `hand_sizes[s]` (the view's own hand is fully known and is not a location for the viewer); `Undealt` = `hidden_count − Σ burn_counts`; `Burn(s, k)` = 1 per entry of `burn_order` (the viewer's own burns are known and are not locations). The second turned card after a Joker is public once turned. During the dealer's pre-trump burn decision only, `Undealt` holds 12 cards, one of which will be turned next (section 4.5).

### 3.2 Facts
```rust
pub struct Tracker {
    pub seat: u8,
    pub trump: Option<Suit>,
    pub known: Bits,                    // own hand + own burns + played + face-up table card + public turned cards not taken
    pub unknown: Bits,                  // the 53 cards minus `known`
    pub locs: Vec<(Loc, u8)>,           // every location and its capacity, sorted by Loc
    pub allowed: [u32; 53],             // per card: bit i set ⇔ the card may be in locs[i]; 0 for known cards
}
impl Tracker {
    pub fn new(view: &SeatView) -> Tracker;          // from the hand's start events, then `update` for each later event
    pub fn update(&mut self, ev: &Event, view: &SeatView);   // incremental: one event at a time
    pub fn from_view(view: &SeatView) -> Tracker;    // = new + update over all events of the current hand (tested equal)
    pub fn consistent(&self, hands: &[Bits; 8], burns_by_loc: &[(Loc, Bits)], undealt: Bits) -> bool;
}
```
Each fact narrows `allowed` with one rule. All of them are certainties under RULES H1 (illegal plays are impossible):

| # | Fact (when it arises) | Effect on `allowed` | Source |
|---|---|---|---|
| F1 | A card becomes known (dealt to me, played, turned and left face up, my burn) | removed from `unknown` | events |
| F2 | **Void:** seat s plays a card whose effective suit is not the plain suit S that was led | no card of plain suit S in `Hand(s)` | rule 14, `effective_suit`, `legal_plays` |
| F3 | **Trump limit:** a trump L is led and seat s plays a non-trump | `Hand(s)` holds no trump except top trumps (5, J, Joker, A♥) ranked above L | rules 16–18; checked against `legal_plays` |
| F4 | **No ace (incremental):** `ProvedNoAce{s}` | ace of trumps not in `Hand(s)`; nor in any `Burn(s, RobAtTurn)` (s never robbed) | A6, H2 |
| F5 | **Ace robbed at once** (turned ace, or ace turned after a Joker) | the ace is known: in `Hand(dealer)` until played (it can never be burned) | rules 21–22, H4 |
| F6 | **Rob at turn:** `Took{s, c}` then `BurnDone{s}` | the ace of trumps is in `Hand(s)` (until played); the taken card c is in `Hand(s)` or `Burn(s, RobAtTurn)`; that burn is never the ace of trumps | rules 20–21, H3 |
| F7 | **Dealer's Joker take** | the Joker is in `Hand(dealer)` or `Burn(dealer, JokerTake)`; that burn may hold **any** card, even the card that turns out to be the ace of trumps | rule 24, H3 (owner, 2026-10-14) |
| F8 | **Dealer's at-once rob burn** after a turned ace | that burn is never the ace of trumps | rule 21, H3 |
| F9 | **Pending robber before their turn** (trick 1, nobody has robbed, turned card on the table) | the ace of trumps may be in `Hand(s)` only for seats without `ProvedNoAce`, or `Undealt`, or `Burn(dealer, JokerTake)` | A6 |
| F10 | **Dealer holding the ace after a Joker + ordinary card** | nothing proved at once: the dealer is treated like every other seat and robs (if at all) as the eighth card of trick 1 | H4 amended 2026-10-14 |
| F11 | `AceUndealt` (trick 1 completed with no rob) | the ace of trumps is in `Undealt` or `Burn(dealer, JokerTake)` | A6 |
| F12 | **Hidden out-of-play cards** | Σ capacities of `Undealt` and all `Burn` = `hidden_count` ∈ {12, 13} (asserted) | RULES card counts |

Notes:
- The plain aces, the A♥ when hearts are not trumps, the taken card and the Joker may all be burned (H3); the tracker never forbids them in a burn.
- A trump-led follow with a trump reveals nothing about plain suits; trumping a plain lead reveals nothing (rule 14). Following suit reveals nothing.
- The tracker never uses soft information (what a player "would" do). That is belief weighting (section 5).

### 3.3 Tracker tests
- `tracker_never_excludes_truth` (**the A6 property test**, proptest): 20,000 random matches (random and rules-v1 players, a mixture so voids and reneges occur) plus every hand-made transcript deal; at every decision, for every seat's tracker, the true deal from `Match::truth()` (hands, burns by location, undealt) satisfies `consistent`. Also run with the forced rare branches of the difftest (turned Joker, Joker then ace, Joker with the dealer holding the new ace, dealer burning the future ace after a Joker, turned A♥).
- `tracker_incremental_equals_batch`: `from_view` equals `new` + `update` per event at every decision of 2,000 matches.
- `tracker_capacity_sums`: Σ capacities = |unknown| at every decision; `hidden_count` ∈ {12, 13}.
- One named test per fact F2–F11 on a hand-made deck (built with `engine::game::build_deck`), each asserting the exact `allowed` bits, including:
  - `joker_burn_may_be_future_ace` (F7);
  - `robbed_ace_never_in_burn` (F5, F6, F8);
  - `dealer_after_joker_robs_eighth` (F10: after Joker + ordinary card, a dealer holding the ace is not excluded until they play, and robs as the eighth card);
  - `plain_ace_may_be_burned`;
  - `reneging_top_trumps_allowed` (F3 keeps 5, J, Joker, A♥ above L);
  - `a_hearts_led_not_trumps` (A♥ led when hearts are not trumps is a trump lead, rule 17).
- `tracker_matches_engine_legality`: for each fact F2–F3, a seat's actual remaining hand always makes its observed play legal under `legal_plays`. This is a cross-check that the fact is derived from the engine's function.

## 4. `crates/belief`: the world generator (A5, PLAN §6.4)

### 4.1 The class matrix (the suit-length matrix, generalised)
Group the unknown cards into **classes**: maximal sets of cards with identical `allowed` masks. At most 16 classes; typically 4–8:
- the plain suits;
- the trumps split by trump-limit thresholds;
- the ace of trumps, the taken card and the Joker when they carry their own facts.

A **class matrix** `M[c][l]` gives how many cards of class c sit in location l, with:
- row sums = class sizes;
- column sums = capacities;
- `M[c][l] = 0` whenever l is not allowed for c.

This is the bridge "suit-length matrix" with locations for columns, extended to the 41 locations. Two worlds with the same matrix differ only by which cards of a class went where.

### 4.2 Counting (A4 counted threshold)
The number of worlds consistent with the facts is
`N = Σ_M Π_c ( n_c! / Π_l M[c][l]! )`.
It is computed by a dynamic program over locations in `Loc` order, where the state is the vector of cards still unplaced per class:
- the transition fills location l with any count vector `k` with `Σ k = cap(l)`, `k_c ≤ remaining_c` and `k_c = 0` where l is forbidden;
- its weight is `Π_c C(remaining_c, k_c)`.

`count(state after the last location) = 1` iff all remaining are 0. The arithmetic is `u128` with checked operations: the largest count is below 2 × 10^36 < 2^128, and an overflow panics as a bug. Memoised by state in a fixed-hash table.
```rust
pub struct WorldSpace { /* classes, locations, memo */ }
impl WorldSpace {
    pub fn new(t: &Tracker) -> Result<WorldSpace, Infeasible>;   // max-flow check first (4.3)
    pub fn count(&self) -> u128;                                 // exact N
    pub fn sample(&self, rng: &mut Rng) -> World;                // exact uniform (4.4)
    pub fn enumerate(&self, limit: u64) -> Option<Vec<World>>;   // all worlds in canonical order iff N ≤ limit
}
pub struct World { pub hands: [Bits; 8], pub burns: Vec<(Loc, Card)>, pub undealt: Bits }
```
`hands[viewer]` is the viewer's real hand. `World` derives `Ord`; canonical order is the derived order.

### 4.3 Feasibility by max-flow
Before counting, build the network source → class c (capacity n_c) → location l if allowed (capacity ∞) → sink (capacity cap(l)), and run integer Dinic or Edmonds–Karp. A maximum flow below |unknown| means the facts are contradictory (`Infeasible`, a bug that must never happen; the property test asserts it). The same max-flow, run per DP state, prunes infeasible branches of the counting DP. This is a pure speed-up: an infeasible branch counts 0 either way, and a test asserts the count with pruning equals the count without it.

### 4.4 Exact uniform sampling, and why it is uniform
1. Draw an integer `r` uniformly in `[0, N)`. Take two `u64`s into a `u128` and reject values ≥ `⌊2^128 / N⌋ · N`, so there is no modulo bias.
2. Walk the DP from the first location. At each location, enumerate its transitions `k` in a fixed order (lexicographic over classes), each with mass `w(k) · count(next state)`. Subtract masses from `r` until it falls inside one, and take that `k`.
3. This yields a matrix M with probability exactly `Π_c (n_c! / Π_l M[c][l]!) / N`.
4. Then, within each class, shuffle its cards with `Rng::shuffle` and deal them to the locations in `Loc` order according to `M[c][·]`.

**Why the result is uniform.** Every consistent world has exactly one matrix M. Given M, the shuffle produces each of the `Π_c n_c! / Π_l M[c][l]!` distinct card placements with probability `Π_l M[c][l]! / n_c!` per class: a uniform permutation, cut into consecutive groups whose internal order does not matter. So each world has probability `[Π_c n_c!/Π M!] / N × Π_c [Π M!/n_c!] = 1/N`. The `World` uses sets, not orders, so no ordering is counted twice.

Random draws: `Rng::derive(decision_seed, &[WORLD, w])` for world number `w`, so world `w` is the same however many worlds are drawn and whichever thread draws it.

### 4.5 Special cases
- **Pending robber.** Worlds only place cards; they do not choose burns. In a world where the ace of trumps lies in `Hand(s)` of a seat that has yet to play in trick 1, that seat robs at its turn inside the rollout, choosing its burn with the inner player's policy. This is the "burn branch" of A4/A5: every world names its burned cards once the rollout reaches them.
- **The dealer's pre-trump burn** (after a turned Joker, trump unknown). Worlds also draw which unknown card in `Undealt` is turned next, uniformly from that world's undealt cards (`Rng::derive(decision_seed, &[WORLD, w, 41])`). This is exact: deck position 41 is a uniform position of the shuffled pack.
- **Undealt order.** It never matters for play (leakage test 1 of M1), so `World::undealt` is a set. Rebuilding a match (section 5.1) orders it ascending.

### 4.6 World-generator tests
- `uniform_vs_rejection_oracle` (chi-square, the A5 test oracle):
  - On 40 small hand-made late-hand positions with N ≤ 5,000 (enumerated), draw 200·N worlds with `sample` and compare the counts per world with uniform: Pearson χ² at p ≥ 0.001.
  - Also compare with a **rejection oracle**: deal the unknown cards uniformly at random into the locations by capacity, ignoring the facts, and keep the deals the tracker accepts. Both histograms must match at p ≥ 0.001.
- `count_equals_enumeration`:
  - on 500 random late positions with N ≤ 10^4, `count()` equals the length of `enumerate`, every enumerated world is `consistent`, and no duplicates appear;
  - on 200 positions, `count()` also equals the number of acceptances when **every** placement is enumerated by brute force (small N only).
- `true_deal_has_positive_mass`: in the A6 property run, `count() ≥ 1` and the true deal is in `enumerate` whenever N ≤ 10^4.
- `count_prune_identity`: counts with and without max-flow pruning are equal.
- `sample_thread_independent`: worlds 0..1000 identical for 1/2/4 threads and for any draw order.
- `world_rebuild_view_identity` (the strongest consistency check): for every sampled world at 2,000 decisions, the match rebuilt from the world (section 5.1) gives the deciding seat a `SeatView` equal to the real one.
- Golden counts: N at the start of trick 1, 4 and 5 for three fixed seeds, stored as test data. Expect the orders of magnitude of A4: ~10^36 at trick 1 down to ~10^8 at trick 5.
- **Cost bench** (`belief-bench` binary, not a unit test): `new` + `count` + 100 `sample` per trick, median and p90 ms, on 2,000 positions per trick. Recorded in the M3 report. There is no pass/fail figure, but trick-1 counting must finish (if it is too slow, raise it with the reviewer; never replace it with an approximation).

## 5. `crates/search`: the PIMC teacher (A3, PLAN §6.7–6.9, §8 phase A)

### 5.1 Rebuilding a world as a match
`fn rebuild(view: &SeatView, world: &World, seed: u64) -> Match`:
1. Build the hand's deck under the frozen deal convention. A seat's original 5 cards are its world hand, plus the cards it played this hand, plus its world burns, minus a card it took. Turned cards go at positions 40/41 and the undealt set (ascending) after them.
2. Call `Match::from_hand_start` with the hand's start state, taken from the view's `HandStart` event and the public scores.
3. Re-apply the hand's public actions in order. Another seat's `Burn` is taken from the world's burn for that location; the viewer's own actions come from its events.

The rebuilt match is the world. It is never shown to a bot except through `view(seat)` (no peeking).

### 5.2 Belief at M3 (section 4 of the task; PLAN A3, A5, §6.4)
No network exists before M4 (A3: "no network exists at M3"). So:
- **Default (`belief = uniform`):** worlds drawn uniformly from the hard-constraint world space (section 4), all weights equal. This is the bottom rung of A5's fallback ladder ("constraint-only uniform belief") and the GIB constrained-deal generator of §6.4, without inference.
- **Measured arm (`belief = rules-lik(ε)`):** importance weighting by the rules bot's policy likelihood, as §6.4 and A7 prescribe for the interim "human-like" model ("the rules bot … as the interim human-like predictor in belief weighting only").
  - `rules-v1` is deterministic, so its likelihood is smoothed with the mandatory floor (§6.4, A5: "a model's opinion never drives a world's weight to exactly zero"): `p(a | world) = (1 − ε)·[a = rules_v1(view_in_world)] + ε / |legal|`.
  - The weight of world w is `Π p` over every action of the **other** seats this hand. The deciding seat's own actions and forced single-card plays are excluded, since `p` is equal across worlds there.
  - Weights are computed in log2 fixed point (`libm::log2` / `exp2`, Q32), normalised by the largest weight, rounded to `u64` and fed to `MoneyChance::mix_wide`. `libm` is pinned pure Rust, so this is identical on every platform; a golden test fixes three weight vectors.
  - Reported per decision: Kish ESS = (Σw)² / Σw² in f64, for reports only.
  - ε ∈ {0.1, 0.3}, pre-registered.
  - This arm enters the tournament only through its own SPRT against `uniform` (section 8.3). It is never adopted without that test (CLAUDE: proven methods, SPRT decides).
- The particle filter, the belief head and MCMC rejuvenation are M4+ and out of M3.

### 5.3 Evaluation methods by phase of the hand (A3, A2)
| Decision point | Method (default) | Inner player / evaluator |
|---|---|---|
| Dealer's at-once burns (trick 0), any decision in tricks 1–2 | **rollout-PIMC** | every seat, the decider included after its candidate action, played by `rules-v1` from its own `SeatView` in the rebuilt world to the end of the hand; end valued by the leaf rule (5.4) |
| Trick 2 (comparison arm `t2 = exact-few`) | **few-worlds exact PIMC** | `solve_fast` per world, defaults switches, `tt_mb = 512` |
| Trick 3 and 4 | **exact PIMC** | `solve_fast` per world; every root card's value from `Solution::children` |
| Trick 5 | none: decision-free (one card each; A4) | — |
| Forced moves | none: a single legal action is returned at once without drawing worlds (§6.8) | — |

Equivalent root cards (the solver's `Classes`) are evaluated once in exact mode. In rollout mode every legal action is rolled out: the rules bot may play equivalent cards differently.

**Rollout-PIMC is genuine PIMC with a realistic inner player.** The world is fixed per sample, but inside the rollout no seat sees another's cards; each sees only its world `SeatView`. This is Bouzy et al. 2020's rollout form, which A3 cites.

### 5.4 Leaf rule (the money)
A rollout ends when the hand ends, or the game or match ends:
- `hand over, game continues` → `table.lookup(MatchState::from_public(next hand's start))`;
- `game won` → the successor state's value, or `MoneyChance::certain(team)` if the match is won.

This is the same as the solver's `Step` leaves (tested equal on 2,000 random rollouts by solving each rollout's final position with 0 cards left).

### 5.5 The robber's joint (burn, first card) decision (A4, §6.9, RULES "burn then first card")
When the view's phase is `AwaitBurn(me)` with `trick_no == 1` (a rob at turn):
- the candidates are every pair (b, p): b ∈ `legal_burns`, and p ∈ the legal plays of the hand after the burn, given the current trick;
- all candidates are evaluated over **one** shared world set (CRN), by rollout-PIMC;
- the best pair is chosen; `act` returns `Burn(b)` and stores p;
- the next `act` (same hand, phase `AwaitPlay(me)`, events unchanged except our `Burned`/`BurnDone`) returns `Play(p)` without new search. A debug assertion checks p is legal.

The record still shows two actions. The dealer's at-once burns (turned ace, turned Joker) are single decisions: nothing of theirs is played straight after, since trick 1 is led by dealer+1.

### 5.6 Common random numbers and determinism (A4)
- Decision seed: `dseed = Rng::derive(bot_seed, &[DECISION, hand_index, actions_so_far_in_match]).next_u64()`.
- World `w`: `Rng::derive(dseed, &[WORLD, w])`.
- Any randomness inside a rollout (none for `rules-v1`; kept for later inner players): `Rng::derive(dseed, &[ROLLOUT, w, seat, candidate_index])`.
- The same worlds serve every candidate (CRN, §6.8).
- Budgets are counts of worlds, never wall time (A4); a wall-clock cap exists only as a safety stop that marks the decision `timed_out` in reports and is off in teacher and arena runs.
- Parallelism: worlds in pre-sized slots, reduced in world order (section 1).
- Test `pimc_thread_independent`: 300 decisions per trick, 1/2/4 threads → byte-identical action-value vectors and choices.

### 5.7 Choice and recorded values
```rust
pub struct DecisionValues {
    pub kind: DecisionKind,                 // Play | Burn | JointBurnPlay | DealerBurn
    pub method: Method,                     // RolloutPimc | ExactPimc | Forced
    pub n_worlds: u32,
    pub world_count: Option<u128>,          // exact N when computed (always for reports)
    pub belief: BeliefMode,                 // Uniform | RulesLik { eps_ppm: u32 }
    pub candidates: Vec<(Action, Option<Card> /*joint play*/, MoneyChance)>,  // in action order
    pub chosen: usize,
    pub own_sd: Vec<u64>,                   // per candidate, sample SD of own-team entry, 2^-32 units, integer sqrt
}
```
- **Choice:** the candidate with the largest entry for the decider's team. Ties go to the first in candidate order (actions ascending by card index; pairs by (burn, play)). This is the solver's documented tie rule (M2).
- **Exactness.** Because averages are exact fixed point over the same worlds, ties are real ties.
- `own_sd` uses integer arithmetic (u128 sums of squares, integer square root) so records stay platform-identical.

### 5.8 World counts, chosen by decision-agreement curves (A4)
World counts are parameters, `w12` (rollout worlds, tricks 0–2) and `w34` (exact worlds, tricks 3–4), set from this measurement (`search agree` binary):
- **Decision sets:** for each of trick 0/1 (incl. robber joint decisions), 2, 3, 4, take 1,000 decisions sampled from `rules-v1` self-play matches started from stratified match states (5.9). Seed `MEASURE`; stored as (record, action index) so they replay.
- **Reference:** for each decision, PIMC with `W_ref` worlds (rollout: 2,560; exact: 640) drawn with an **independent** seed label (`AGREE`, 1).
- **Grid:** W ∈ {5, 10, 20, 40, 80, 160, 320, 640, 1280} (exact: up to 320), two seeds each.
- **Metrics per (trick, W):**
  - `agreement(W)` = fraction of decisions where the W-world choice is among the reference's tied-best candidates;
  - `loss(W)` = mean over decisions of (reference value of the reference choice − reference value of the W-world choice), the decider's team entry, in money chance, with its standard error.
- **Rule (pre-registered):** `w12`/`w34` = the smallest W on the grid with `loss(W) ≤ loss(next grid W) + 0.0005` **and** `loss(W) ≤ 0.002`. Quality first (CLAUDE): the teacher is offline, so cost is not a criterion here. The tier budgets of A4 are set at M5.
- The published plateaus (20 worlds for 5-card solver PIMC, 160 in Skat; A4) are reported beside the curve for comparison.
- **Output:** `agree.json` and a text table (trick × W: agreement, loss ± SE, ms per decision).

### 5.9 The trick-2 comparison arm (A3: "few-worlds exact PIMC as a measured comparison arm")
On the 1,000 trick-2 decisions of 5.8:
- exact PIMC with W ∈ {2, 4, 8, 16};
- reference: exact PIMC with 64 worlds;
- report agreement and loss as in 5.8 against that reference, for both exact-few and rollout-PIMC at `w12`.

**Decision rule:** if exact-few at some W has a loss lower than rollout-PIMC by more than 2 SE, the arm `t2 = exact-few(W)` is entered as a teacher variant and decided by SPRT on the 14-way block (section 8.3). Otherwise rollout-PIMC stays. This is a report that steers tournament time, not an automatic switch (A3).

### 5.10 Teacher hands (`search teacher` binary and runner chunk)
- **Unit:** one hand, started with `Match::from_hand_start`, all 8 seats played by `pimc-teacher` (the PIMC bot at teacher budgets `w12`, `w34`, `belief = uniform`). Every decision's `DecisionValues` is recorded.
- **Stratification** (A3: "stratified over match states and tricks so the late-match blocking states are covered"):
  - Four strata of start states:
    - S1 opening: games 0-0-0-0, all scores ≤ 20;
    - S2 mid: no team at 1 game with ≥ 26;
    - S3 one team at 1 game: max score 0–25;
    - S4 clinch-near: some team at 1 game and ≥ 26, or any team ≥ 31.
  - Equal weight per stratum (25% each). Within a stratum, start states are drawn from the empirical distribution of hand-start states visited by 1,000,000 `rules-v1` self-play hands (seed 1, the equity v0 data), stored as `strata-v0.json` (state, frequency, stratum).
  - The hand's deck: `Rng::derive(chunk_seed, &[TEACHER_HAND, i])` shuffle; the dealer comes from the state.
  - The per-trick coverage (decisions per trick per stratum) is reported.
- **Outputs per chunk:**
  - `hands-<sha16>.jsonl`: one line per hand: format version, rules version, equity table sha256, start state, deck, actions, final money chance or successor state, bot config string, git commit.
  - `decisions-<sha16>.npy`: structured, little-endian:
    - `hand_id u64`, `action_index u16`, `seat u8`, `trick u8`, `kind u8`, `method u8`, `belief u8`, `n_worlds u32`;
    - `log2_world_count_q8 u16` (⌊256·log2 N⌋);
    - `cand_action u16[32]` (burn card · 64 + play card + 1, or play card + 1; 0 = empty);
    - `values u64[32][4]`, `own_sd u64[32]`, `chosen u8`.
  - The network inputs are **not** stored. M4's Rust encoder rebuilds them by replaying `hands` (PLAN §6.5: "the encoder is written once, in Rust"). Records stay action-only (A7).
  - A Python round-trip test (`python/tools/read_decisions.py`, numpy only) reads one shard and checks a golden row.
- **Seeds:** chunk seed = `Rng::derive(run_seed, &[stream_index, chunk_index])`, so any chunk can be redone identically (PLAN §12.4).
- **M3 delivers the generator and a 10,000-hand smoke run.** The 10^5–2×10^6-hand teacher run belongs to M4 (A3, §13 M4 row).

### 5.11 PIMC tests
- `rollout_no_peeking`: inside rollouts each inner bot's `SeatView` passes the M1 view-purity check against the rebuilt world's truth.
- `exact_pimc_equals_mean_of_solves`: for 200 trick-3/4 decisions, the recorded candidate values equal `MoneyChance::mix` of the per-world `solve_fast` children, recomputed independently with the naive `solver::solve`.
- `enumerated_pimc_at_small_n`: when N ≤ 10^4 at trick 4, PIMC with `enumerate` equals the exact expectation over all worlds.
- `joint_burn_then_play`: the Play after a joint decision equals the stored card, and the pair maximises over the full candidate grid (recomputed).
- `forced_no_search`: single legal action → zero worlds drawn.
- `teacher_chunk_reproducible`: the same chunk seed twice, and with 1 and 4 threads, gives byte-identical `hands` and `decisions` files (same sha256).
- `leaf_rule_matches_solver`: see 5.4.

## 6. The pre-registered PIMC-suitability measurement (A3, PLAN §5.1, §9; Long, Sturtevant, Buro & Furtak, AAAI 2010)

Long et al. explain PIMC's success with three properties of the game tree. Their definitions are for two-player win/loss trees. Here is the 41 form, fixed **before** any data is seen. Binary: `search suitability`.

**Sample.** 10,000 hands from `rules-v1` self-play, started from stratified start states (5.10 strata, equal weights), seed `MEASURE`. The positions are the decision nodes (≥ 2 legal cards, any seat) on the actual path of each hand, grouped by trick t = 1..5. Trick 5 is decision-free and is reported as such.

**Metrics per trick t**, each with its count and a standard error clustered by hand:
1. **Leaf correlation `lc_t`.** Long: "the probability that sibling terminal nodes have the same payoff". For a decision node, solve the true deal open-handed (`solve_fast`); the node is *correlated* if every legal child has the identical `MoneyChance` (exact fixed-point equality). `lc_t` = fraction of correlated nodes. Also reported:
   - `lc_t(δ)`: equality of the mover's team entry within δ = 0.001;
   - `lc_term`: Long's terminal form, which is the trick-4 nodes whose children are all one step from the hand's end (2 cards left per seat).

   Tricks 3–4 use all nodes. Trick 2 uses a 1,000-hand subsample (cost). Trick 1 is reported as **not measurable** (no exact solve in 5 min, M2-SPIKE) with the rollout-based proxy beside it, clearly labelled.
2. **Bias `b_t`.** Long: "the probability that the game favours a particular player". 41 form: from each node, play 64 uniformly random legal play-outs on the true deal to the end of the hand, scoring each leaf by the leaf rule. Let the leaf's *favoured team* be the team whose entry rose most from the node's exact value (ties excluded and counted). `b_t` = mean over nodes of the largest share any single team has among that node's favoured-team outcomes, so it runs from 0.25 (no bias) to 1. Per-stratum values are also reported.
3. **Disambiguation `df_t`.** Long: "how quickly hidden information is revealed". 41 form, from exact world counts:
   - for each seat's view at the start of trick t and t+1, `N` = `WorldSpace::count()`;
   - `df_t` = mean over seats and hands of `(log2 N_t − log2 N_{t+1}) / log2 N_t`, the share of the remaining (uniform-belief) uncertainty that trick t removes;
   - also the raw medians, p10 and p90 of `log2 N_t`.
4. **Unresolved at trick 4 `u_4`.** At each trick-4 decision node, take the decider's view:
   - enumerate all consistent worlds if N ≤ 10^4, else sample 1,000;
   - solve each world with `solve_fast`;
   - the node is *resolved* if some legal card is among the tied-best cards in **every** world; `u_4` = fraction unresolved.

   The same is reported for trick 3 (sampled worlds only).

**Report** (`suitability.json` plus a plain-English text table in `docs/M3-REPORT.md`):

| trick | nodes | lc | lc(δ) | bias | df | log2 N median (p10–p90) | unresolved |
|---|---|---|---|---|---|---|---|

Each value carries its SE. The report adds:
- the published comparison points (Long et al.: high correlation, low bias and moderate disambiguation favour PIMC; their Skat and Hearts readings);
- a plain-English paragraph on what the numbers say about where tournament time should go (e.g. a high `u_4` with low `df_t` argues for weighting worlds by inference sooner).

**It is a report, not an engine switch** (A3).

**Tests:**
- the metrics on hand-made toy positions with known answers (all-tied node → lc 1; a forced node excluded; counts from 4.6);
- reproducibility: same seed gives the same JSON.

## 7. `crates/runner` and `python/launcher`: the full runner (PLAN §12, A8)

### 7.1 Manifest (format 2; format 1 still runs as a single-chunk stream)
```json
{ "format": 2, "run_id": "m3-teacher-smoke", "run_seed": 7, "deadline_utc": "2026-10-20T06:00:00Z",
  "threads": null, "bin_sha": "…", "inputs": { "equity": "sha256:64740dab…6808", "strata": "sha256:…" },
  "upload_every_s": 600, "heartbeat_every_s": 30,
  "streams": [
    { "name": "teacher", "priority": 1, "kind": "teacher", "params": { "hands_per_chunk": 200, "config": "pimc-teacher" }, "chunks": 50 },
    { "name": "gate", "priority": 2, "kind": "arena_blocks", "params": { "a": "pimc-v0", "b": "rules-v1", "block14": true, "blocks_per_chunk": 4, "sprt": { "s0": 0.02, "s1": 0.06, "alpha": 0.05, "beta": 0.05, "max_blocks": 578 } }, "chunks": "until_decided" },
    { "name": "suitability", "priority": 3, "kind": "suitability", "params": { "hands_per_chunk": 250 }, "chunks": 40 } ] }
```
- Chunk kinds: `difftest`, `equity_sweep`, `benchmark`, `solver_spike` (existing), and new `teacher`, `arena_blocks`, `suitability`, `agree`, `belief_bench`.
- Inputs are fetched by content hash and verified.
- Chunk `i` of stream `k` has seed `Rng::derive(run_seed, &[k, i])`.

### 7.2 Scheduling: priorities, chunks, deadline
- Workers: `threads / threads_per_chunk` child processes, each running one chunk at a time, single-threaded inside unless the kind asks for more.
- Next chunk = the lowest priority number with work left. Within a stream, chunks go in index order. For fixed lists with known sizes, longest first (LPT).
- Chunk size target: 2–5 minutes, from the start-up benchmark and the running median of finished chunks of that kind.
- **Deadline:** no chunk is issued if `now + 1.5 × median duration` passes the deadline.
- **`until_decided` (SPRT streams):** the LLR is recomputed in block order over the contiguous finished prefix; once it crosses a bound, or hits the cap, no further chunks of that stream are issued. In-flight chunks finish and are recorded but excluded from the decision beyond the decided prefix, as in Fishtest. The decision is therefore independent of completion order.

### 7.3 Heartbeat, done-list, checkpoints, uploads
- **Heartbeat** as in M1 (`runs/<run>/heartbeat/<seq>.json`), plus queue state (chunks done, running, left per stream).
- **The network never blocks the runner** (added 2026-10-14, after run `m3-agree-t1-b` lost about $18 of finished work: an rclone call hung, the heartbeats stopped while the solver kept working, and the launcher destroyed the machine). The local `heartbeat.json` is written every interval whatever the network does; its upload, the `STOP` check and the partial uploads run on background threads (the newest heartbeat wins; a check or upload still going is not started again), and every rclone call has hard limits: rclone's own short timeouts (`--contimeout 10s --timeout 30s --low-level-retries 2 --retries 1`) and a kill after 45 s plus one second per MiB moved. Other uploads run beside the heartbeat and are tried up to 3 times. Heartbeats report `upload_failures`, `other_net_failures` and `last_upload_ok_s` (seconds since the last good upload).
- **Partial uploads and per-world resume.** `search worlds` appends every finished world to a checkpoint next to its output (`<worlds file>.ckpt`; deleted once every decision is written), so a resumed run redoes only the worlds that were being solved and still writes the same bytes. Every `upload_every_s` (format 1: default 300 s; format 2: its setting, at most 600 s) the chunk's changed output files are uploaded, checkpoints first, to mutable keys: `runs/<run>/partial/out/…` (format 1) or `runs/<run>/partial/<stream>/<index:06>/out/…` (format 2). A stop uploads the checkpoints first, then the rest. Format 1 resumes with `"resume_from": "<earlier run id>"` (manifest or chunk params, `search_measure` only, with a new `run_id`): it downloads that run's final `out/`, else its `partial/out/`, before the chunk starts. Format 2 restores a `search_measure` chunk's live files when it starts the chunk again on a replacement machine.
- **Honest progress.** During a worlds step the heartbeat carries `progress.eta = {done_frac, remaining_s, eta_utc, projected_usd, basis}`: work done weighted by each world's expected cost (the scheduler's own model per trick and position, refined by this run's measured times; worlds still running count for their elapsed time, up to their expected cost), the end time at this session's rate, the run's projected cost at the machine's price, and `basis` = `measured` once at least 10% of the work is done and every position with work left has at least 2 worlds timed, else `rough: …` naming what is missing. Format 2 passes each running chunk's estimate on (`progress.chunk_eta`).
- **Machine-cost records (owner request, 2026-10-16).** Every rented machine is measured on the run itself. The runner follows the solver nodes [positions the exact solver examined; all our work is counted in them] of the worlds finished on this machine (`search worlds` reports them in its progress file and writes them into every checkpoint line and timing record; a `t1bench` window reports its own) and adds to every heartbeat's `progress`: `nodes_total`, `worlds_finished`, `usd_so_far` (elapsed × the machine's price), `usd_per_1e12_nodes` (null until a node is counted) and `nodes_per_s_per_thread`; the compact log line shows the cost per node too (`$/1e12 0.312 (settling)`). Every 10 minutes and at the end it writes `machines/<run_id>.json` (bucket root, mutable; an immutable copy under `machines/history/<run_id>/<utc>.json`; the same file under the run's work root) with the machine (CPU model, threads, RAM, $/hour, offer/host/instance numbers, binary, image), the run so far (minutes, dollars, worlds finished of total, nodes, schedule kind `plan`/`fill`/`split`, table size or budget, trick, chunk kind) and the two renting figures; the final record carries `state: "final"` and the outcome. The cleanup script marks the record of a handed-over run final (from its last heartbeat, else the launcher's watch data) when the runner could not. Uploads go on the background network lane: they never block the runner (`crates/runner/src/machine.rs`).
- **The live view and the database** (`python/tools/machines.py`): `--live` prints one line per running machine (run, CPU, threads, RAM, $/h, minutes, $ so far, worlds finished/total, $ per 10^12 nodes so far); the default prints every record; `--by-cpu` gives, per CPU model and schedule, the runs, the mean and best measured $ per 10^12 nodes and nodes/s per thread; `--rank` ranks today's vast.ai offers by the measured thread speed of their CPU model applied to their price (estimate otherwise, marked `~`); `--record-speeds` writes the measured speeds into `docs/vast/speed-factors.json` through `vast_market.py record`; `--backfill` builds records for runs made before the records existed. **Caveat:** early figures read high, because the work queue runs the hardest deals first; a record is marked `settling` until 10% of its worlds are done, and `--live` says so.
- **Done-list:** after each chunk's outputs are uploaded and verified, append its entry `{stream, index, seed, outputs: [{path, sha256, bytes}], seconds}`. Every `upload_every_s` (≤ 600 s) write `done/<run>/<seq:08>-<utc>.json`, the full list, as a new versioned key. Readers take the highest sequence number.
- **Uploads:** output bundles go to `runs/<run>/out/<stream>/<index:06>-<sha16>.tar.zst` with `rclone copy --immutable`.
- **End:** `rclone check` → manifest uploaded last (git commit, image digest, binary build v2/v3, seeds, threads, versions) → `DONE` → self-destroy (M1 behaviour).
- **Resume:** on start, the runner reads the highest done-list. It skips listed chunks (verifying their remote sha256) and redoes any chunk that is missing or incomplete from its seed. Partial local outputs of unlisted chunks are deleted first.
- **Launcher** (`python/launcher/launch.py`):
  - on a missing heartbeat (3 min) or a dead instance before `DONE`, it destroys the instance and rents a replacement with the same manifest, which resumes;
  - at most 3 replacements per run, then it stops and reports;
  - on-demand only; the 20% speed-premium rule as in M1.

### 7.4 Resume test (gate item)
`resume_identity` (Rust integration test in `crates/runner/tests`, with the remote as a local directory through the same `Remote` trait the rclone backend implements):
1. Run a manifest of three small streams (teacher 6 chunks × 2 hands, arena_blocks 3 chunks × 1 block of `rules-v1` vs `random`, suitability 2 chunks × 3 hands) uninterrupted; record the sha256 of every output and of the final merged reports.
2. Run it again from scratch, killing the runner process (SIGKILL on Unix, `TerminateProcess` on Windows) after chunk 2 and again after chunk 5. A test hook triggers the kill at a named point: also once in the middle of an upload, and once between upload and done-list write.
3. Restart each time. Final outputs and reports must be byte-identical to step 1, and no chunk may be listed twice.

A manual vast.ai rehearsal (kill the instance from the launcher mid-run; the replacement finishes) is run once and logged in STATUS.

### 7.5 Runner tests
- manifest parsing (format 1 and 2);
- priority order;
- LPT order;
- deadline cut-off with a fake clock;
- SPRT stream stops at the bound and ignores out-of-order completions beyond the prefix;
- done-list versioning picks the highest sequence;
- `resume_identity`.

## 8. Bot integration and the M3 gate

### 8.1 Registry
`search::factory(name) -> Option<Box<dyn BotFactory + Send + Sync>>` knows `pimc-v0`, `pimc-teacher` and parameterised variants, and falls back to `bots::factory`. `arena` switches its registry lookup to `search::factory`, so `random` and `rules-v1` still work.
- **Name grammar:** `pimc-v0[:key=value,…]` with keys:
  - `w12` (rollout worlds), `w34` (exact worlds);
  - `t2=rollout|exact-few`, `t2w`;
  - `belief=uniform|rules-lik`, `eps`;
  - `threads`.

  The canonical form (keys sorted, defaults omitted) is the recorded bot name.
- **Equity table:** the factory loads the table once (shared `Arc<EquityTable>`) from the path in `FORTYONE_EQUITY` or `arena --equity <path>`, verifies its sha256 against `data/equity/` and records it in every record.

### 8.2 Budgets
- **`pimc-teacher`:** `w12`, `w34` from section 5.8, `threads = 1` (teacher parallelism is across hands).
- **`pimc-v0`** (the arena and M3b bot): the same world counts, capped so a decision fits the owner's think time on the 4-vCPU VM on one thread (8 s trick 1, 5 s trick 2, ≤ 3 s tricks 3–4; PLAN A4, owner decision 2026-10-13).
  - The caps are expressed in worlds, never time, so play is deterministic: `w12_cap = ⌊8 s / measured p90 rollout-decision cost per world⌋`, `w34_cap` likewise for trick 3.
  - They are fixed once from the `belief-bench`/`agree` timings and written into the bot's defaults with the measurement date.
  - If the cap is below the agreement-curve plateau, the report says so plainly (strength is then budget-limited at M3; M5 decides tier budgets).

### 8.3 The gate: PIMC beats rules-v1 decisively
- **Test:** `arena sprt --a pimc-v0 --b rules-v1 --block14`, GSPRT on block scores (M1-SPEC §5.2), in money-chance units:
  - **s0 = 0.02, s1 = 0.06**, α = β = 0.05 (bounds ≈ ±2.944);
  - minimum 20 blocks;
  - cap = `default_max_blocks(σ_b = 0.146)` = ⌈5 × 8.67 × (0.146/0.04)²⌉ = **578 blocks**, then "inconclusive".
- **"Decisively"** means the test rejects a gain of 2 percentage points of match-win chance or less in favour of 6 or more. Even at H0's edge PIMC would be worth +0.16 stake per team per match (0.08 per player: a match swings 8 stakes for a team, 4 for each player, +3 or −1). The expected length is about 115 blocks in the worst case, fewer if the edge is large.
- **σ_b = 0.146** was measured for rules-v1 vs random; it sets only the cap. The GSPRT uses the sample variance, and the actual σ for this pair is reported.
- **Two stages (A9):** stage 1 at half the world counts, reject-only; then stage 2 at full budget.
- **Variant arms, each its own SPRT after the gate:**
  - `belief=rules-lik(0.1)` vs `uniform`, `[s0, s1] = [0, 0.02]`;
  - `t2=exact-few` if 5.9 triggers it.
- **Canary:** `pimc-v0` vs itself, 5 blocks, every block exactly 0 (in `cargo test` with tiny world counts) and 50 blocks in the gate run.
- **Gate report** (`docs/M3-REPORT.md`):
  - SPRT result, per-mix breakdown (1v3, 2v2, 3v1), σ, blocks, cost;
  - the suitability table (section 6);
  - the agreement curves (5.8) and the trick-2 arm (5.9);
  - the world-generator cost bench;
  - the resume test result.

## 9. Tests and CI
- `cargo test --workspace` stays fast (each new unit test under a few seconds; large sweeps `#[ignore]` or in binaries).
- CI adds:
  - `cargo test -p belief -p search` on Linux and Windows;
  - a `cargo build -p belief --target wasm32-unknown-unknown` check (M3b runs the generator in a Worker);
  - the canary step for `pimc-v0` with tiny budgets;
  - the `resume_identity` test on both OSes.
- `belief` must stay free of threads and file I/O so it builds for wasm32.

## 10. Out of M3
- Networks, the belief head, the particle filter and MCMC rejuvenation (M4).
- Recursive rollouts, SPARTA thresholds, truncated rollouts and tier budgets (M5).
- The Cloudflare slice (M3b).
- The large teacher run (M4).
- AIVAT (M4).

## 11. Estimated compute (estimates until measured; `rules-v1` rollout cost assumed ≈ 100 µs, exact trick-3 world 9 ms median per M2-SPIKE)

| Measurement | Core-hours (est.) | Cheapest sensible place |
|---|---|---|
| Unit and property tests incl. A6 property (20k matches), χ² oracle | < 0.5 | this VM / CI |
| World-generator cost bench (2,000 positions × 5 tricks) | ~0.5 (trick-1 counting is the unknown) | this VM |
| Agreement curves (5.8): 4,000 decisions, references 2,560 rollout / 640 exact worlds, grid × 2 seeds | ~6 (dominated by trick-3 exact references: 1,000 × 640 × 9 ms ≈ 1.6 h, plus grid) | vast.ai (≈ $0.10) or this VM in background chunks |
| Trick-2 exact-few arm (5.9): 1,000 decisions × (64 + 30) worlds × ~3 s | ~80 | vast.ai, 32–64 cores (≈ $1–2) |
| PIMC suitability (6): 10,000 hands; lc trick 3–4 ~0.3 h, lc trick 2 (1,000 hands × 8 × 3 s) ~7 h, u_4 ~1.5 h, counts and bias ~1 h | ~10 | vast.ai (≈ $0.20) |
| Teacher smoke run: 10,000 hands at ~5 core-s per hand (tricks 1–2 ~1.6 s, trick 3 ~3 s) | ~14 | vast.ai, as the first full-runner run (≈ $0.30) |
| M3 gate GSPRT: ≤ 578 blocks × ~100 hands × ~1.2 core-s | ≤ 19 (expected ~4–10) | vast.ai, same run (≈ $0.20–0.50) |
| Resume test, rehearsal | < 0.5 | CI + one short vast.ai run |
| **Total** | **~130 core-hours, ≈ $3–5** | |

Every figure is re-measured by the start-up benchmark and the first chunks; the launcher's cost report goes into STATUS.

## 12. Open questions for the owner (plain English)
1. **How big a win counts as "decisive"?** I propose that the new bot must show it wins the match at least 2 percentage points more often than the rules bot, testing for 6 points. At the pub stake (€5) that is a gain of at least 80 cents per team per match, 40 cents per player. Is that the bar you want, or should it be higher?
2. **Practice games for the "teacher".** The new bot's teacher will be trained mostly on ordinary positions, but we plan to give a quarter of its practice hands to positions where someone is close to winning the money (one game up and 26 or more points), so it learns blocking. Does a quarter sound right to you, or should near-the-end positions get even more?
3. **Early tricks.** In tricks 1 and 2 the bot imagines everyone playing like the simple rules bot when it tests a card, because exact analysis is far too slow that early. Do you see any common situations in the first two tricks where good players play very differently from the plain rules (for example around robbing or holding back top trumps) that we should test specially?
4. **Budget.** The M3 measurements need about €5 of rented machine time in total. Is that fine to spend without asking each time?
5. *(Already answered, for the record.)* The dealer may burn, after a turned Joker, the card that turns out to be the ace of trumps (RULES H3: legal). The tracker allows for it. STATUS still lists it as open and should be updated.

## 13. Amendment (2026-10-14): early-trick quality, from the owner and the trick-1 measurement
**Owner:** in tricks 1–2, good players often play differently from the plain rules. The bots must find the best plays themselves, so the early-trick searches must come as close as possible to a true solve.

**Measurement** (`m2-trick1-1`, AMD EPYC 9654, 368 threads, parallel exact solver): four trick-1 worlds solved exactly in **7 s, 32 s, 140 s and 25 s**, using up to 65 GB. Exact trick-1 world solves are therefore affordable offline on a large machine. They are not affordable inside a timed decision.

Changes to this spec:
1. **Exact early-trick reference set (new, required for the gate report).** Build 300 trick-1 and 300 trick-2 decisions, stratified as §5. Value each with **exact PIMC**: N exact world solves per decision, with N taken from the agreement curve. Trick 1 runs on a large vast.ai machine with the parallel solver (`solve_fast_par`); trick 2 runs on ordinary cores. This set is the yardstick for every cheaper early-trick method.
2. **Report against the yardstick.** For rollout-PIMC (rules-v1 inner player) and every other early-trick arm, report per trick:
   - the agreement of the chosen card with exact PIMC;
   - the mean money-chance loss of the chosen card under the exact-PIMC values (in 2^-32 units and in stake units);
   - the loss distribution.
3. **Trick 2 teacher default.** The few-worlds exact trick-2 arm stops being only a comparison. It becomes the **default trick-2 teacher** if its mean loss against the yardstick is lower than rollout-PIMC's by more than 2 standard errors at a compute cost within the M3 budget. Otherwise rollouts stay the default, and the gap is reported.
4. **Trick 1 teacher.** Rollout-PIMC stays the scalable trick-1 teacher at M3. Its measured loss against the exact yardstick is a pre-registered number the later phases must reduce: M4 network rollouts, M5 recursive search and M6 self-play. Each of these is scored against the same yardstick in addition to tournaments.
5. **Scope note.** Exact PIMC is itself an approximation of true imperfect-information play, because it plays each world with all cards face up. It ignores what each card reveals and how it is inferred. The yardstick measures how close the cheap searches come to exact PIMC. Strength beyond PIMC comes from M5 search and M6 self-play, judged by money in tournaments (PLAN §8–9).
6. **Note (Joker burn, answered by the owner 2026-10-14).** The dealer's burn after a turned Joker, made before trumps are known, may turn out to be the ace of trumps: legal (RULES H3), a poor play; nobody robs and the ace is announced out of play after trick 1. The tracker's fact F7 (§3.2) allows for it. The open question in STATUS and PLAN is closed.

### 13.1 Sizing the yardstick (owner-approved 2026-10-14; replaces the fixed "300 + 300")
The yardstick is for **measuring only, not training**. Its size is set by measurement:
1. **Deals per decision (N): agreement curve first.**
   - **Trick 2:** 200 decisions with N = 10, 20, 40, 80, 160, 320 (nested world sets, common seeds).
   - **Golden-N rule (replaces the plateau rule, 2026-10-15):** the golden N per position is the smallest N whose fitted money loss (the power-law fit of item 10, judged by the split-half referee) is at most 0.01% money chance per decision at the 95% upper bound. The earlier plateau rule ("doubling N changes the chosen card in under 1% of decisions and the mean chosen-card value by under 0.001 money chance") could not be used: the card changes kept happening at every N measured, and the value change says nothing about the money lost; the rule actually used is the fitted-loss one.
   - **Trick 1:** 20 decisions on a large machine with `solve_fast_par`, N = 10, 20, 40, 80, to measure trick 1's **own** plateau N, using the same rule. Trick 1 has more hidden cards than trick 2, so it may need a larger N. If there is no plateau by N = 80, extend to 160 and 320 (priced with the owner first). Trick 1 never borrows trick 2's N.
   - 20 is only the published plateau for 5-card PIMC, and Skat needed 160, so nothing is assumed.
2. **Number of decisions.** Methods are compared **paired**, on the same decisions. Size the set so that a 3-point difference in card agreement between two methods is detected at 95% confidence; the expected range is 300–500. The disagreement rate is measured on the trick-2 data before any trick-1 money is spent.
3. **Trick 2:** 1,000 decisions at the plateau N, costing under $1.
4. **Trick 1:** cost = decisions × N × measured seconds per deal. The 384-core EPYC measured 7–140 s per deal, about $0.28 per decision at N = 20. **The trick-1 set is priced with the owner before it runs.**
5. **Repeatability:** re-solve 20 decisions with fresh deals at the chosen N. The chosen card must match in at least 95% of them, or N is raised.
6. **Position in the trick (owner, 2026-10-14).** The plateau N is measured and set **per trick and per position in the trick** (1st to 8th to play). The leader of trick 1 has the most unknowns; the 8th player has seen 7 cards. Decisions are stratified equally across positions. Positions may be pooled into pairs (1–2, 3–4, 5–6, 7–8) only if their plateaus agree within one doubling step. Time per deal is also reported per position.
7. **Tricks 3 and 4 too (owner, 2026-10-14).** The same agreement curve, with 200 decisions (25 per position) and N up to 320, is measured for tricks 3 and 4. Exact solves are cheap there, about 9 ms per world at trick 3. This replaces the assumed N for the exact-PIMC teacher at tricks 3–4 with a measured one. Trick 5 has no decisions, since each player has one card left, so it needs no N.
8. **Match situation and uncertainty (owner, 2026-10-14).**
   - Decisions are stratified equally over the four match-state strata S1–S4 (§5), as well as over positions, and the plateau is reported **per stratum** as well as per position. Near the money, values are more extreme, so a small difference can flip the choice.
   - The plateau is also reported against the decision's **number of consistent worlds** (log2, from the belief counter): what N must cover is the uncertainty about the hidden cards.
   - **Rule:** N is set per trick and position. If any stratum or world-count band needs a larger N than that, the larger N is used for that band.
   - The decision count is raised until every reported cell has at least 25 decisions.
   - The golden set is built with the same stratification.
9. **Hand type (owner, 2026-10-14).** Every decision record carries the deciding seat's hand type, computed from cards it can see: number of trumps held, number of top trumps held (5, J, Joker, A♥, ace of trumps), and the longest plain suit.
   - **Added features (owner, 2026-10-14):** holds the 5 of trumps (yes/no); robber (yes/no: took a turned card this hand, including a dealer who took a turned Joker or robbed a turned ace, from the public burn order); void plain suits (0, 1, 2+). Partners always sit across, so there is no partner feature. `agree` and `curves` report these groups too, and `curves` compares their N in pairs (paired bootstrap ratio). Older worlds files get them by `--decisions FILE`, which replays each decision; no re-solving.
   - The agreement output reports the plateau per hand-type band, alongside position and stratum.
   - **Tricks 2–4 run first**, to learn which hand types need the most worlds. The expensive trick-1 set is then sampled with more decisions in the bands that need them.
   - Each band is weighted back to its true frequency, from the 20,000-hand start pool, when the overall N and the golden set's summary are reported. Sampling more of the hard bands never biases the answer.
10. **N-curves: the money lost at each N, and the N each scenario needs (2026-10-14).** The saved per-world values are turned into curves without solving again (`search curves`, `crates/search/src/curves.rs`; charts and report by `python/tools/ncurves_report.py`; first report in `docs/m3-ncurves/`).
   - **Grid.** Any N up to the saved M (default every 10). The bot's choice with N worlds is re-made from N of the M saved worlds, many times (seeded random subsets without replacement), exactly as the bot chooses (fixed-point means).
   - **Fair referee.** Each repetition splits the M worlds in halves: the N-world choice and the K = M/2-world reference choice come from the first half and are both scored on the second, which neither saw. On average this is exactly L(N) − L(K), the true loss at N minus the true loss at K. The plain reading (subsets judged against all M) and the single nested first-N reading are reported too; the plain one reads low as N nears M.
   - **Fit.** L(N) = a·N^(−b): one b per trick, shared by the 8 positions, and its own a for every group (position, stratum, hand-type band, and position × stratum × trumps cells with at least 10 decisions). Weighted least squares on log scale over N ≤ K/2. Ranges come from a bootstrap of decisions within position × stratum cells, refitting everything each time.
   - **Checks.** Held-out N (fit on N ≤ K/2, predict up to K), and extrapolation (the same recipe on the first M/2 worlds must predict the full data).
   - **Thresholds.** N for an expected loss of 0.01%, 0.005% and 0.002% money chance per decision, per group, with ranges.
   - **Extension.** `search worlds --ids 3,17,42|FILE --from-world M --to 2M` values more worlds for chosen decisions; the new records continue the saved ones exactly (same seeds), and `agree`, `curves` and `score` join all records of a decision (`--worlds` may be given several times). The `search_measure` chunk takes `ids` (or `ids_input` in format 2), `from`, a fine `grid` (e.g. `"10:320:10"`), and runs `search curves` after `search agree`; an extension chunk (`from` > 0) only values the new worlds.
   - **Decision sets.** `search decisions` refuses a `--count` that is not a multiple of 32 (4 strata × 8 positions), unless `--allow-uneven` is given (chunk: `"allow_uneven": true`), and prints the position × stratum table it made. The default count is 224 (7 per cell).
11. **Spreading exact trick-1 solves over a big machine, and keeping finished work (2026-10-14).** Exact trick-1 worlds cost from seconds to hours of one core, mostly by position in the trick, and a parallel solve does far less work per thread on hundreds of threads than on a few (about 8–10 times less on 368 than on 4). One-thread costs on the 4-vCPU Xeon: position 4 about 200 s a world (about 1 billion solver nodes), position 6 about 16 s, positions 7–8 10–50 s; the leader's worlds 1–29 billion nodes (M2 runs).
   - **Bench** (`t1bench` chunk; `search bench-t1`; `crates/search/src/t1bench.rs`; rebuilt 2026-10-14 after two rented machines sat mostly idle under the old configuration sweep). Its only job is to rank machines by exact trick-1 solving per dollar, with every core busy the whole time. After setup (table v0 and the stratified decision set, count 32, seed 1; the only stretch marked `partial_cpu_ok`, with a warning past 90 s), a **fill window** (`fill_s`, 240 s) runs exactly one world per usable thread, thread k on yardstick worlds of position (k mod 8)+1 in a fixed order, every table `fill_tt_mb` (512 MB, lowered to fit memory with a warning), and stops every solve at the end, counting the nodes [positions the solver examined] of unfinished worlds too. A **production window** (`prod_s`, 120 s) then runs the real position-aware schedule on a queue too deep to run dry, to check that production keeps the machine full and to give its raw node rate (which includes the extra work of splitting a world over many threads). Throughout both windows, from 20 s in, CPU below `min_cpu_pct` (80%) of the usable cores over the last 30 s, or fewer than 95% of its threads working for more than 10 s in a row (the tolerance lets the production plan leave a few threads unassigned), stops the bench at once: it writes what it has and the chunk ends FAILED with the reason (e.g. `low CPU: 41% in fill at 50 s`), so the machine uploads and destroys itself. Heartbeats show the window, seconds into it, threads working, nodes and node rate (`partial_cpu_ok: false`), and the runner prints one line per heartbeat to the machine's log. Outputs `out/t1bench/results.json` and `summary.txt` (plain English): per window nodes per second for the machine and per thread, CPU use, share of time every thread was working, peak memory, table size and dollars per 10^12 nodes; per position the fill rate; one machine score (the fill rates weighted by each position's share of real trick-1 work, `position_nodes`, defaults from the M2 runs); and the projected hours and dollars of a 128 × 200 golden set at one world per thread. A world finished in both windows must have byte-identical values, or the chunk fails.
   - **Table-size sweep** (added 2026-10-15; `t1bench` settings `sweep_mb`, a list of table sizes in MB per thread, and `sweep_s`, default 90). Nodes per second cannot show what a small table costs (a table that forgets more makes the solver search positions again, and that still counts as nodes), so after the main windows the bench runs one more fill window per size on the **same worlds in the same order** as the fill window, and compares them on the worlds that finished in both: the nodes and wall time those worlds took, against the fill window's table size, and their values, which must be byte-identical (table size changes speed only; a difference fails the chunk). Results: `out/t1bench/fill-sweep-<mb>.json` and the `sweep` section of `results.json` and `summary.txt`. A size the memory plan lowers to one already run is skipped with a warning. Only worlds short enough to finish in the windows are compared (worlds from every position, but the longest worlds of the early positions are usually missing), so this measures the small-table regime; the 12-31 GB tables of positions 1-3 are not covered and need a long run of a few worlds. Tested by `analyse_sweep` unit test (`search`) and a local end-to-end run.
   - **Position-aware schedule** (`crates/search/src/plan.rs`; `search worlds --schedule auto`; the runner's default when `worlds_at_once` is `"auto"` or neither `worlds_at_once` nor `solve_threads` is given; explicit numbers keep their old meaning). Each world gets threads from its expected cost (per position; joint burn-and-play decisions count one solve per burn) so that worlds take about the same time, at most a quarter of the run, and more threads if its share of the memory (shared out per thread) would not hold the table it needs; worlds start longest first while threads are free; solver memory is reused per shape and freed when its shape is no longer queued; tables in use never exceed the memory budget. The worlds file is byte-identical under any schedule (tested on 1/3/8 threads, several plans, 1 MB tables, resumed runs). Manifest `schedule` overrides the plan's settings (e.g. `{"t1_costs_s": [...]}` measured by the yardstick run). The trick-1 bench's production window checks that this plan keeps a machine full (see the bench above).
   - **Stopping keeps finished work** (`crates/runner/src/stop.rs`). SIGTERM/SIGINT/SIGHUP (the boot script passes its own SIGTERM on), the launcher's `runs/<run>/STOP` object (checked at every heartbeat, in the background) or a local `STOP` file stop the run: format 1 uploads its checkpoints first, then its whole `out/` (the worlds file holds every finished decision, the checkpoint every world finished since), and writes `FAILED` with `stopped` and the file list; format 2 uploads each stopped chunk's changed files to its live prefix (checkpoints first; what a replacement continues from, §7.3), then its unfinished `out/` to `runs/<run>/partial/<stream>/<index>-<utc>.tar.zst` (not in the done-list) and ends `DONE` with `stopped`. Before destroying a machine that is still working (time limit, low CPU, Ctrl-C) the launcher writes `STOP` and waits up to `--stop-wait-s` (300 s) for the marker; outcome `STOPPED`. A cancelled workflow's cleanup step does the same (`launch.py stop RUN --instance ID`). A machine whose heartbeats stop is checked through vast first (run m3-agree-t1-b lost 172 minutes of work when a busy machine with a stuck upload was destroyed at once): gone, exited or idle is destroyed at once; running and busy (vast CPU at least 50%) gets up to `--silent-busy-min` (15) more minutes for heartbeats to resume, then is asked to stop and upload like the cases above. `--max-dollars` (workflow `max_dollars`) stops and uploads the same way once the run's machines have cost that much, and no replacement is rented past it. Tested by `crates/runner/tests/stop_partial.rs` (SIGTERM mid-step, format 1; `STOP` mid-step, format 2) and the launcher tests; the network's hard limits by the runner test `hung_network_never_blocks` (an rclone that hangs on every call: heartbeats keep coming, each call is killed at its limit, the chunk ends on time); per-world resume by `search`'s `worlds_resume_per_world` and `crates/runner/tests/resume_worlds.rs` (SIGKILL mid-step with decisions part-done, then `resume_from` in format 1 and a replacement in format 2: only the worlds missing from the uploaded copy are valued, and the worlds file is byte-identical).
   - **Table v0 without the 1,000,000-hand estimate** (`table.source` `"auto"`, the default of `t1bench` and `search_measure`): a format-2 input, else `inputs/sha256/<recorded hash>` in R2, else `equity build` from `data/equity/model-rules-v1-1M-seed1.json` (kept in the runner; about 8 s on 4 threads); always checked against the recorded hash; a table built on a machine is shared to R2 for the next ones.
   - **Running the bench on a vast.ai machine** (one run per machine type; `dph` comes from the launcher):
     ```json
     { "run_id": "t1bench-<machine>-1", "benchmark_seconds": 0, "chunk": { "t1bench": {} } }
     ```
     `vast-run` inputs: this manifest, `min_cores` 256 (or the type wanted), `cpu_match` e.g. `EPYC`, `max_minutes` 20, `cpu_only` true, `cheapest` false, `dry_run` false. The measured part takes 6 minutes (240 s + 120 s) on any machine, plus 2–5 minutes of boot, table and decisions. The launcher's `poll_s` (60) and `low_cpu_s` (300) set how often it checks the machine and how long low CPU is tolerated; a container that cannot start is destroyed at the next check and another offer rented.
