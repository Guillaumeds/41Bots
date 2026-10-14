# M0 specification: engine, reference implementation and differential test

This is the build specification for checkpoint M0. It is derived from `docs/RULES.md` (authoritative) and `docs/PLAN.md` §6.1–6.3 and the amendments. Where this file and RULES.md disagree, RULES.md wins and this file must be fixed.

Already built in `crates/engine`: `cards.rs` (card indices and bitboards), `rng.rs` (SplitMix64 + xoshiro256**, bounded, shuffle; golden-tested against `tools/rng_golden.py`), `rules.rs` (orders, legal plays, legal burns, trick winner). Read them before building `game.rs`; do not change their public functions without updating this spec.

## 1. Shared conventions (engine, reference and difftest)

- **Seats** 0–7 clockwise; "left of seat s" is `(s + 1) % 8`. **Team** of a seat is `seat % 4`; seats s and s+4 are partners. Every trick is played in the order leader, leader+1, …, so teams always cycle A, B, C, D, A, B, C, D.
- **Card index** (`u8`): `suit * 13 + (rank − 1)` with suits clubs 0, diamonds 1, hearts 2, spades 3 and ranks ace 1, 2…10, jack 11, queen 12, king 13. The Joker is 52. The reference may use any internal representation but must convert to and from this index.
- **Card names** in canonical strings: rank then suit letter: `A`, `2`…`10`, `J`, `Q`, `K` and `C D H S`; the Joker is `JK`. Example: `5H`, `10S`, `AH`.
- **Orders** (RULES 6–10): trumps 5, J, Joker, A♥, ace of trumps, K, Q, then red 10 9 8 7 6 4 3 2 or black 2 3 4 6 7 8 9 10. Plain red suit K Q J 10…2 A; plain black suit K Q J A 2…10. Plain hearts has no ace. With hearts trumps there are 14 trumps and the A♥ is the ace of trumps.
- **Deal convention (frozen).** A deck is the 53 card indices in shuffled order. Positions 0–15 go two at a time to seats dealer+1, dealer+2, …, dealer+8 (i.e. seat `(dealer + 1 + i) % 8` gets positions `2i, 2i+1`); positions 16–39 three at a time in the same seat order (seat `(dealer + 1 + i) % 8` gets `16 + 3i .. 16 + 3i + 3`); position 40 is turned; position 41 is the second turned card if the first was the Joker; the rest, in order, are the undealt pack. Every hand uses a freshly shuffled deck.
- **Rules switches** (`Rules` struct, both crates): `stake: i32` (default 5). The bonuses are fixed constants `BONUS_FIRST = 6` and `BONUS_LATER = 5` (the `bonus_first`/`bonus_later` switches were removed on 2026-10-14). Money is settled per team in stake units × stake: the winning team receives `+6 × stake`, each of the three losing teams `−2 × stake` (each player of a losing team pays the stake; each winner receives three stakes).
- **Actions:** `Burn(card)` and `Play(card)`. Robbing and taking are never actions; the engine performs them automatically and only asks for the burn. A forced single legal card is still submitted as an action. `legal_actions()` is sorted by card index and is never empty before the match is over.
- **Determinism:** integer logic only; no `HashMap`/`HashSet`; `Vec` and arrays only; no floating point anywhere.

## 2. The hand, step by step (both implementations must follow this exactly)

Let `dealer` be the dealer seat, `D = dealer`.

1. **Deal** per the convention. Emit `HandStart`, one private `Dealt` per seat, then `Turned(card40)`.
2. **If the first turned card is the Joker** (rule 24, H4):
   a. The dealer takes it: emit `Took{seat: D, card: JK}`. The trump suit is not yet known. Phase `AwaitBurn(D)`; legal burns are all of the dealer's 6 cards, aces and the Joker included: trumps are not yet known, so nothing is forbidden (H3).
   b. After the burn: turn position 41: emit `Turned(card41)`. The trump suit is now the suit of card41 (it cannot be the Joker).
   c. If card41 is an ace (any suit; an A♥ makes hearts trumps and is then the ace of trumps): the dealer takes it (`Took`), phase `AwaitBurn(D)` (any of the 6 cards except the ace of trumps, rule 21); after the burn go to step 5.
   d. *(Removed 2026-10-14, H4 amended: a dealer holding the ace of trumps of an ordinary second card no longer robs at once; step 2e applies and the dealer is the pending robber.)*
   e. Else: card41 stays face up on the table; whichever seat holds the ace of trumps, the dealer included, is the pending robber (step 6 handles it); if nobody holds it (it is undealt, or the dealer burned it in step 2a), nobody robs. Nothing is proved about the dealer here; the dealer proves they lack the ace like everyone else, by playing their first card of trick 1 without robbing. Go to step 5.
3. **Else if the first turned card is an ace** (rule 22, H4): the trump suit is its suit; the dealer takes it (`Took`), phase `AwaitBurn(D)` (any of the 6 cards except the turned ace, now the ace of trumps). After the burn go to step 5.
4. **Else:** the trump suit is the turned card's suit; the card stays face up; the seat holding the ace of trumps, if any, is the pending robber; go to step 5.
5. **Trick 1** is led by seat `D + 1`. Set `trick_no = 1`, `leader = D + 1`, `to_act = leader`.
6. **Before each seat's first play of trick 1** (only in trick 1, and only if nobody has robbed yet and the turned card is still on the table): if `to_act` is the pending robber, the engine gives them the turned card (`Took{seat, card}`), and the phase becomes `AwaitBurn(seat)` with legal burns = their 6 cards minus the ace of trumps; after the burn, phase `AwaitPlay(seat)`. If `to_act` is **not** the pending robber and has not proved it before, emit `ProvedNoAce{seat}` the moment they play their first card (they played without robbing); this includes a dealer who took a turned Joker.
7. **A play** (`Play(card)`): must be in `legal_plays(hand, led, trump)`. Remove the card from the hand, add it to `played`, append `(seat, card)` to the current trick, emit `Played`. If it is a trump and beats the best trump so far (or there was none), update `best_trump = (card, seat)`. If it is the 5 of trumps, set `five_played`. Then `to_act` moves left, unless the trick has 8 cards.
8. **When the 8th card of a trick is played:** winner = `trick_winner` of the 8 cards in play order; emit `TrickWon{trick, seat: winner, team, points: 5}`; `tricks_won[team] += 1`; `scores[team] += 5`. **If the 5 of trumps was played in this trick** (it necessarily won it): award the best-trump bonus now: `points = bonus_first` if the team has not yet taken a bonus this game, else `bonus_later`; set the team's bonus flag; emit `Bonus{team, points, kind: Immediate, card: 5 of trumps}`. **Then check 41:** if the winning team's score ≥ 41, the game ends: reason `FortyOneByImmediateBonus` if the score was below 41 before this trick's bonus and the bonus was awarded in this trick, otherwise `FortyOneByTrick`; go to step 11. **If the trick completes trick 1** and nobody has robbed and no seat holds the ace of trumps (it is undealt): emit `AceUndealt`. If `trick_no < 5`: `trick_no += 1`, `leader = winner`, `to_act = winner`, go to step 6/7. If `trick_no == 5`: go to step 9.
9. **End of hand, after trick 5:** if one team won all 5 tricks: the game ends with reason `Jink` (rule 33); go to step 11. Else if `five_played` is false and a trump was played in the hand: award the end-of-hand bonus to the team of the seat that played `best_trump` (`bonus_first` or `bonus_later` as above, set the flag), emit `Bonus{kind: EndOfHand, card}`; then if that team's score ≥ 41 the game ends with reason `FortyOneByEndBonus`; go to step 11. (If `five_played` is true the bonus was already awarded in step 8. If no trump was played there is no bonus, rule 31.)
10. **Hand over, game continues:** emit `HandEnd{hand_index, aborted: false}`; `dealer = D + 1`; `hand_index += 1`; go to step 1.
11. **Game end:** emit `GameWon{team, reason, scores}` ; `games_won[team] += 1`; emit `HandEnd{hand_index, aborted: hand not finished}`. If `games_won[team] == 2`: the match is over; compute money; emit `MatchWon{team, money}`; phase `MatchOver`. Else: `game_index += 1`; all four scores → 0; all four bonus flags → false; `dealer = D + 1` (H5); `hand_index += 1`; go to step 1.

Pinned details (both implementations must agree; settled 2026-10-13 after the reference build):
- **Event order.** `ProvedNoAce{seat}` is emitted immediately before that seat's `Played`. `Burned` (private) is emitted immediately before its public twin `BurnDone`. (Until 2026-10-14 step 2e emitted the dealer's `ProvedNoAce` at once; it no longer does.)
- **Second turned card.** During the dealer's first burn after a turned Joker, position 41 is still in the undealt pack: `undealt` has 12 cards (positions 41–52) and `hidden_count` is 12. Only after that burn is position 41 turned (`undealt` becomes positions 42–52, 11 cards).
- **Trick number and leader.** `trick_no` is 0 during the dealer's at-once burns (steps 2–3) and 1 from step 5 onward, including during a leading robber's burn. `leader` is `dealer + 1` from the start of the hand. `trick_cards` is cleared the moment a trick's eighth card is played (also for trick 5 and for a game-ending trick); `leader` and `to_act` move to the winner only when a next trick starts.
- **Robber.** `robber` is set by every `Took` except the dealer's take of the Joker: the dealer robbing a turned ace (step 3 or 2c) makes the dealer the robber; so does a dealer robbing the second card at their own turn in trick 1. The turned-cards list records the Joker as taken by the dealer; `table_card` is the second card if it stays.
- **HandEnd.aborted** is true exactly when fewer than 5 tricks were completed. A game ending on trick 5 is not aborted, even if another team's pending end-of-hand bonus is lost.
- **After MatchOver** nothing is reset and the dealer does not move; `hand_index` and `game_index` start at 0 and `hand_index` counts across games. Between hands within a match the engine starts the next hand immediately; the reference waits for a deck.
- **`Rules`** derives `Clone, Copy, Debug, PartialEq, Eq, Default` (6, 5, 5) in both crates.
- **Rule 31 is unreachable with 8 players:** 40 cards are in play and at most 13 are out of play, but there are 14 or 15 trumps, so every completed hand has at least one trump played. The difftest reports the "no-trump hand" branch (always 0) and a property test asserts it.

Notes. A turned card that nobody takes stays known and out of play. The dealer's pre-trump burn (step 2a) happens before the trump suit exists; any card may be burned there, aces included, since trumps are not yet known (H3); if the dealer burns the card that becomes the ace of trumps, nobody robs. Hidden cards at any time = undealt (12, or 11 after a Joker) + all burns (0–2): always 12 or 13. A seat's hand is 5 cards at every play of trick 1 after any take and burn.

## 3. Engine API (`crates/engine`)

```rust
pub mod cards; pub mod rng; pub mod rules; pub mod game; pub mod view;   // in lib.rs

// game.rs
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub struct Rules { pub stake: i32 }  // Default: 5
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)] pub enum Action { Burn(Card), Play(Card) }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Phase { AwaitBurn(u8), AwaitPlay(u8), MatchOver }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum GameEndReason { FortyOneByTrick, FortyOneByImmediateBonus, FortyOneByEndBonus, Jink }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum BonusKind { Immediate, EndOfHand }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Visibility { Public, Private(u8) }
#[derive(Clone, Debug, PartialEq, Eq)] pub enum Event {
    MatchStart { first_dealer: u8 },
    HandStart { hand_index: u32, game_index: u8, dealer: u8 },
    Dealt { seat: u8, cards: Bits },                 // Private(seat)
    Turned { card: Card },
    Took { seat: u8, card: Card },
    Burned { seat: u8, card: Card },                 // Private(seat)
    BurnDone { seat: u8 },                           // Public twin of Burned
    Played { seat: u8, card: Card },
    ProvedNoAce { seat: u8 },
    AceUndealt,
    TrickWon { trick: u8, seat: u8, team: u8, points: u8 },
    Bonus { team: u8, points: u8, kind: BonusKind, card: Card },
    GameWon { team: u8, reason: GameEndReason, scores: [u8; 4] },
    HandEnd { hand_index: u32, aborted: bool },
    MatchWon { team: u8, money: [i32; 4] },
}
#[derive(Clone, Debug, PartialEq, Eq)] pub struct IllegalAction { pub action: Action, pub reason: &'static str }
pub struct Truth { pub deck: [Card; 53], pub hands: [Bits; 8], pub burns: [Bits; 8], pub undealt: Bits }  // Clone, Debug
pub struct Match { .. }
impl Match {
    pub fn new(seed: u64, rules: Rules) -> Match;                     // first dealer = rng.bounded(8); each hand's deck = 0..53 shuffled by rng.shuffle
    pub fn with_decks(first_dealer: u8, decks: Vec<[Card; 53]>, seed: u64, rules: Rules) -> Match;  // uses the given decks first, then seeded shuffles
    pub fn replay(seed: u64, rules: Rules, actions: &[Action]) -> Result<Match, IllegalAction>;
    pub fn rules(&self) -> Rules;  pub fn seed(&self) -> u64;
    pub fn phase(&self) -> Phase;  pub fn to_act(&self) -> Option<u8>;
    pub fn legal_mask(&self) -> Bits;  pub fn legal_actions(&self) -> Vec<Action>;
    pub fn apply(&mut self, action: Action) -> Result<(), IllegalAction>;   // never panics on an illegal action
    pub fn events(&self) -> &[(Visibility, Event)];  pub fn actions(&self) -> &[Action];
    pub fn dealer(&self) -> u8; pub fn hand_index(&self) -> u32; pub fn game_index(&self) -> u8;
    pub fn scores(&self) -> [u8; 4]; pub fn games_won(&self) -> [u8; 4]; pub fn bonus_taken(&self) -> [bool; 4];
    pub fn money(&self) -> Option<[i32; 4]>;
    pub fn truth(&self) -> &Truth;                                    // hidden information, for tests and the difftest only
    pub fn public(&self) -> view::PublicState;  pub fn view(&self, seat: u8) -> view::SeatView;
}
impl Event { pub fn canonical(&self) -> String; }   // format in section 5
```

Order of `events()` entries within a hand: exactly the order the steps above emit them. `MatchStart` is first. `Dealt` events are emitted seat by seat in deal order starting at dealer+1.

`view.rs`:

```rust
#[derive(Clone, Debug, PartialEq, Eq)] pub struct PublicState {
    pub phase: Phase, pub to_act: Option<u8>,
    pub dealer: u8, pub hand_index: u32, pub game_index: u8,
    pub scores: [u8; 4], pub games_won: [u8; 4], pub bonus_taken: [bool; 4],
    pub trump: Option<Suit>,                // None only during the dealer's pre-trump burn
    pub table_card: Option<Card>,           // the turned card currently face up and not taken
    pub turned: Vec<(Card, Option<u8>)>,    // every turned card this hand and who took it
    pub joker_turned: bool,
    pub hand_sizes: [u8; 8], pub burn_counts: [u8; 8], pub hidden_count: u8,
    pub played: Bits, pub trick_no: u8,     // 0 before the first lead, else 1..=5
    pub leader: u8, pub trick_cards: Vec<(u8, Card)>, pub trick_winners: Vec<u8>, pub tricks_won: [u8; 4],
    pub best_trump: Option<(Card, u8)>, pub five_played: bool,
    pub robber: Option<u8>, pub proved_no_ace: [bool; 8], pub ace_undealt: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)] pub struct SeatView { pub seat: u8, pub hand: Bits, pub own_burns: Bits, pub public: PublicState, pub events: Vec<Event> }
```
`view(seat).events` contains every `Public` event and the `Private(seat)` events, in order. `hand_sizes[s]` is 6 while seat s must burn.

## 4. Reference implementation (`crates/reference`)

Written independently and naively from RULES.md and section 2 of this file, with its own types (`Vec<Card>` hands, enums for suit and rank, ordered lists). **It must not depend on, read or copy the engine crate** (`[dependencies]` stays empty; a CI step fails if `cargo tree -p reference` lists `engine`). It receives decks from the outside instead of shuffling.

```rust
pub struct Rules { pub stake: i32 }   // Default 5
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Action { Burn(u8), Play(u8) }   // card indices
pub struct Match { .. }
impl Match {
    pub fn new(first_dealer: u8, rules: Rules) -> Match;   // starts in "needs a deck"
    pub fn needs_deck(&self) -> bool;                       // true between hands and before the first
    pub fn start_hand(&mut self, deck: &[u8; 53]);          // deals per the convention and runs steps 1–5
    pub fn phase(&self) -> Phase; pub fn to_act(&self) -> Option<u8>;   // Phase as in the engine (own enum)
    pub fn legal_actions(&self) -> Vec<Action>;             // sorted by card index
    pub fn apply(&mut self, action: Action) -> Result<(), String>;
    pub fn dealer(&self) -> u8; pub fn hand_index(&self) -> u32; pub fn game_index(&self) -> u8;
    pub fn scores(&self) -> [u8; 4]; pub fn games_won(&self) -> [u8; 4]; pub fn bonus_taken(&self) -> [bool; 4];
    pub fn money(&self) -> Option<[i32; 4]>; pub fn match_over(&self) -> bool;
    pub fn hand(&self, seat: u8) -> Vec<u8>; pub fn burns(&self, seat: u8) -> Vec<u8>; pub fn undealt(&self) -> Vec<u8>;   // sorted indices
    pub fn public(&self) -> Public;                          // same fields as the engine's PublicState, with Vec<u8> (sorted) for card sets and Option<u8> for the trump suit index
    pub fn events_for(&self, seat: u8) -> Vec<String>;       // canonical strings (section 5) visible to that seat, in order
}
```
When a hand ends without ending the match, the reference moves the dealer, updates indices and returns to `needs_deck() == true`; the difftest then calls `start_hand` with the engine's next deck.

## 5. Canonical event strings (both crates must produce these exactly)

One line per event, fields space-separated, card sets in ascending index order:
```
MatchStart first_dealer=<n>
HandStart hand=<hand_index> game=<game_index> dealer=<n>
Dealt seat=<n> cards=<names joined by ,>
Turned card=<name>
Took seat=<n> card=<name>
Burned seat=<n> card=<name>
BurnDone seat=<n>
Played seat=<n> card=<name>
ProvedNoAce seat=<n>
AceUndealt
TrickWon trick=<1..5> seat=<n> team=<n> points=5
Bonus team=<n> points=<n> kind=<immediate|end> card=<name>
GameWon team=<n> reason=<trick|immediate_bonus|end_bonus|jink> scores=<a>,<b>,<c>,<d>
HandEnd hand=<hand_index> aborted=<true|false>
MatchWon team=<n> money=<a>,<b>,<c>,<d>
```

## 6. Differential test (`crates/difftest`)

Depends on `engine` and `reference` only. Library function `run(seed: u64, games: u64, rules) -> Report` plus a `--release` binary `difftest <games> [seed] [--threads N]`.

For each match: `engine = engine::game::Match::new(seed_i, rules)`; `reference = reference::Match::new(engine.dealer(), rules)`. Loop: if `reference.needs_deck()`, call `reference.start_hand(&engine.truth().deck)`. Compare the snapshot (phase, to_act, legal actions, public state, every seat's hand and burns, undealt set, every seat's canonical event list, scores, games, bonus flags, money). Stop with a detailed report on the first disagreement (seed, hand, action index, the two snapshots). If the engine is `MatchOver`, record the money and go to the next match. Otherwise choose the k-th legal action with `k = rng.bounded(len)` from a difftest RNG derived from the seed (the engine's own `rng::Rng::derive`), apply it to both, and continue. Count rule branches per match (turned Joker; Joker then ace; Joker with the dealer holding the ace; turned A♥; turned 5 of trumps; turned plain ace; dealer robbed at once; non-dealer robbed; nobody robbed; jink; game ended mid-hand; immediate bonus; end bonus; no-trump hand; five-game match) and print the counts; `cargo test` runs 200 matches; the binary asserts the minimum coverage listed in STATUS.md for runs of ≥ 100,000 matches. Also replay every engine match from its seed and actions and assert the event log is identical.

## 7. Tests the engine must ship with (M0)

- One test per numbered rule in RULES.md (rules 1–41; rules 25, 26, 34, 37–41 assert that the illegal action is rejected or that the forced take happens).
- The named transcripts listed in `docs/STATUS.md` task 5, each built with `Match::with_decks` and asserted play by play (deal, turned card, actions, scores after every trick, result). Each transcript also renders to plain English (a helper `transcript(&Match) -> String`) for owner review.
- Property tests (proptest, dev-dependency) over random seeds for the invariants listed in STATUS.md task 5.
- Replay: every random match replays byte-identically from seed plus actions (event logs equal).

## 8. Hand-made transcripts (owner-review material and test input)

Each transcript is one UTF-8 text file under `crates/engine/tests/transcripts/<nn>-<slug>.txt`, written before the engine exists, straight from RULES.md. A later harness (`crates/engine/tests/transcripts.rs`) replays each file against the engine with `Match::with_decks` and asserts every expectation; disagreements are adjudicated against RULES.md by the reviewer (the transcript may be the one that is wrong). Format, one directive per line, `#` starts a comment:

```
title: <one line>
story: <plain-English description of what the deal tests and why, may repeat on several story: lines>
rules: stake=5                        # optional; default 5
dealer: 3                             # first dealer of the match (seat 0-7)
hand 4: 5H JK AD 2C 9S                # five cards per seat, every seat 0-7, any order of lines
...
turned: 9D                            # the card at deck position 40
second: AD                            # optional: the card at position 41 (required when turned is JK; must be undealt otherwise)
undealt: 2D 3D ...                    # the remaining cards in pack order (position 41 or 42 onward); omit cards already placed; the harness verifies the 53-card permutation
then hand 5: ...                      # optional: a second deal for the next hand, same directives prefixed by "then "; as many as needed
> Burn 2C                             # an action by the seat to act (Burn or Play), applied in order
= Took seat=3 card=JK                 # expected canonical event(s) produced by the preceding action, in order (public events only; section 5 format)
= BurnDone seat=3
scores: 5,0,0,0                       # expected scores after the preceding action
phase: AwaitPlay 4                    # optional expected phase after the preceding action (AwaitBurn n | AwaitPlay n | MatchOver)
legal: 2C 9S                          # optional: the exact legal cards for the seat to act, before the next action
result: game team=2 reason=jink       # optional, after the action that ends a game; or "result: match team=1 money=30,-10,-10,-10"
```
Expectation lines (`=`, `scores:`, `phase:`, `legal:`, `result:`) always refer to the state right after the most recent `>` action (or right after the deal, if they appear before the first action). A transcript need not assert every event; it must assert every rule it exists to test. Seats that must rob or burn are chosen by the engine; the transcript's `>` lines must supply those burns in the right place.

