# The app's engine contract, answered by the engine

Reply of 2026-10-16 (Fable, the engine's reviewer) to `fivefingers:app-modes docs/engine-contract.md`
(eee5d32 / 581b2cd). The app's author did not know the 2026-10-16 engine work (the exact solver
with the oracle network, "the hybrid"), so this says what the new engine gives, field by field,
what stays the app's, and where the app's design should change. Nothing here merges `app-modes`
or deploys anything; Guillaume decides that.

## 0. Where the new engine stands (2026-10-16)

- **Exact solver** (`crates/solver`): the whole hand solved exactly for four teams (max^n with the
  documented tie rule: the first card by index among equals), in whole numbers; every value is the
  four pairs' money chance [chance of winning the match] as `u64` in units of 2⁻³² summing to
  exactly 2³². Identical bytes on every machine, thread count and in WASM; our own pinned random
  numbers; index-ordered merging; no clocks anywhere.
- **The hybrid H_k** (`crates/search/src/hybrid.rs`): exact search from the decision to the start
  of trick *k*, then the oracle network O_k (integer inference, `crates/nn`) for the rest of the
  hand. Exact from trick 3 today; O_4 nearly proven, O_3 in its first rounds, O_2 next
  (`docs/M3-HYBRID-SPEC.md`, `docs/m3-hybrid/`). Measured on a desktop core: a trick-2 deal
  12.7 ms with H_3 (exact 1,289 ms); a trick-4 deal 0.04 ms exact.
- **PIMC over deals** (`crates/search/src/pimc.rs`): a decision is valued over N deals drawn from
  the seat's own information (the belief sampler; likelihood weights available), each deal solved
  by the hybrid or exactly; the candidates' four-vectors are the index-ordered means.
- **Not yet built:** the student network that plays fast (PLAN M4–M6: distilled from this
  teacher), the WASM export of the search API (only `crates/wasm`'s determinism check exists),
  the plan strip (§2), the §1 adapter. All are small except the student, which is the plan's next
  checkpoints.

## 1. Field mapping (the names and meanings of the app's §1, kept)

The engine's WASM API will expose one call, `evaluate(match_view, seat, budget, seed) ->
Result`, with these fields. Types are what the JSON across the WASM boundary carries; the app's
TypeScript names are kept exactly.

| App field | Engine source | Notes |
|---|---|---|
| `candidates[]` best first | `pimc::candidates` + the PIMC means, sorted by `value` with the documented tie rule | Equivalent cards (same rank class in the solver's canonical form) are one candidate, as today's engine does; `action` names the lowest card of the class and lists the class in `equivalent` (new, optional). |
| `candidates[i].action` | `engine::Action` (`Play(card)` or `Burn(card)`) | Same two kinds as the app. Card ids: the engine's 0..52 index (53 cards, one Joker = 52); the adapter maps the app's numbers (§6.1). |
| `candidates[i].mean[4]` | the candidate's money-chance four-vector, pairs in the app's seat order from the seat's own view | `number[4]` in 0..1 (the fixed-point value ÷ 2³²); the fixed-point bytes are also given (`mean_fp`, four `u64` as decimal strings) for anyone who wants byte identity. |
| `candidates[i].value` | `mean[own pair]` | |
| `candidates[i].se` | standard error of the own-pair value over the deals (likelihood-weighted) | |
| `candidates[i].n` | deals valued | the work budget's N, or fewer when the position is fully known |
| `candidates[i].ess` | effective sample size of the likelihood weights (`(Σw)²/Σw²`) | = `n` when weights are uniform |
| `candidates[i].trickWin` | share of deals whose principal line has the acting seat's pair winning the current trick after this candidate | from the solved line of each deal |
| `candidates[i].trickWinner[8]` | share of deals in which each seat wins the current trick | same source |
| `candidates[i].expectedPoints` | mean over deals of the acting pair's points at the end of the hand on the principal line | exact where the line is exact; at tricks 1–2 the network's expected outcome beyond the cut (the outcome head gives the distribution over hand endings) |
| `candidates[i].gameEnds` | share of deals whose principal line ends a game in this hand | same |
| `candidates[i].plan` | §2 | new build |
| `best` | index 0 of `candidates` | |
| `worlds` | `n` of the best candidate; **0 for a forced card** (one legal action), with empty `candidates` numbers | convention kept |
| `stake`, `floor` | **the app's** (`objective.ts`: `matchStake`, `matchFloor`) | not the engine's; the engine does not compute "games in play" |
| `meanHorizon` | tricks searched exactly before the network takes over: `5 − tricksDone` when exact to the end (trick 3 on), else `k − trick` | the review's "N tricks ahead" label; "to the end" when exact |
| `seat view` | the engine's `SeatView` from the app's `maskMatch`-style knowledge: the seat's cards, the hands' sizes, the cards played, the scores, games, dealer, turned card, robs | nothing more is needed; the belief sampler draws the rest |

Everything is from the seat's own view, as the app requires. A position's value from one seat's
view while another seat is to act (the app's §5) is the same call with `seat` = the viewer and
the acting seat taken from the position: free.

## 2. `CandidateStats.plan`: the engine's plan strip

`plan[c][t]` = the weighted share of deals whose principal line, after this candidate, plays the
acting seat's card `c` in trick `t` (0-based); the candidate's own card is 1 in the current
trick; robs not counted; a card's shares sum to at most 1.

- The principal line of each deal comes from the solver (the best reply at every node under the
  documented tie rule). Where the solve is exact (trick 3 on, and tricks 1–2 once O_2/O_3 are in
  use with the cut at the start of trick k), the line is walked to the end of the hand; at the
  network's cut the line stops, and the shares beyond it are simply absent (sum < 1), exactly the
  app's rule.
- Merged over shards by `n`, as `mergeResults` does; `n` is the deal count per shard.
- Cost: nothing measurable; the line is already in the search.

## 3. The day's match: the same bots on every device

This is the engine's design, not a feature to add:

- Budgets are **work**: N deals per decision (and the exact solve has no cap at all). There is no
  clock in the engine; `capMs: Infinity` is the only mode it has.
- **Byte-identical** results for any thread count and worker split: deals are seeded
  `derive(seed, decision, world)` and merged in index order, so 1, 2, 4 or 7 workers give the same
  bytes; the app's sharding by a fixed count of 8 is honoured but not needed (any split gives the
  same answer). `tests/ai/duplicate.test.ts` is the engine's normal state.
- **No state between searches**: a transposition table per exact solve, the network's memo per
  thread that never changes a value, no move-ordering memory.
- **Ties**: one rule, first card by index, nowhere seeded, identical everywhere. (The engine's
  values in positions where a seat is indifferent depend on that rule. `ties: average` belongs to
  today's TypeScript engine (forge e4e7f8c) and goes with it; the app's own tie rule in its grades,
  `TIE_LOSS` = half the standard error, is the app's and stays.)

## 4. What cannot be met, and the design change

A trick-2 deal costs the hybrid about 12.7 ms on a fast desktop core — about 50 ms on a phone in
WASM; a trick-1 deal costs seconds; a decision wants hundreds of deals; a hand has about 35 bot
decisions. **No engine plays full-strength PIMC bots on a phone within 2 minutes a hand**, and the
engine work of M3 is not aimed at that. The plan's own answer (M4–M6, A3/A4) is the design the app
should take:

1. **Bots that play = the student network** (the policy distilled from this teacher) plus exact
   search from trick 3: milliseconds per decision, integer inference, identical on every device,
   no budget, no clock, no shards. The Club's "same bots for everyone" becomes trivial, and the
   bots are as strong as the teacher taught them. Until the student exists the app's two-trick Pro
   stays; it is the only honest option today.
2. **The engine that judges = the hybrid PIMC with a work budget** (N deals per decision, fixed
   seeds): Play, par, the cards' costs, the pair odds, the plan strip, the lab, Replay — after the
   hand or on demand, with "thinking…" shown; grades are comparable across devices because the
   numbers are byte-identical. Reviews that must be instant on a phone run on the server (same
   binary, same bytes; `worker/` is where the Club already lives).
3. `DUPLICATE_BUDGET` therefore becomes the **review** budget (deals, never seconds), not a bot
   budget. The engine's budget type is `{ deals: N, horizon: k }`; it will report `n`, `ess` and
   `meanHorizon` for every call so the app can label the work done.
4. The swap on a Monday (UTC) is fine with the engine; the engine's version (commit, network
   SHA-256, values version) comes in every `Result` so `club_hands` can record it.

## 5. What the engine will deliver, in order

1. The WASM API with the §1 fields and `plan` (Opus, after the current hybrid rounds): a thin
   adapter over `pimc` + `hybrid`; the app's `tests/coach/plan.test.ts` and
   `tests/ai/duplicate.test.ts` run against it unchanged.
2. The pace numbers from a phone (`scripts/bot-timing.ts` with the engine's budgets at deals
   32 / 64 / 128), so Guillaume can set the review budget.
3. The student network (M4–M6) for the bots that play.

Questions for the app's author are in §6; answers go into this file.

## 6. Answers from the app (2026-10-16) and the engine's decisions

The app is deployed: fivefingers `main` = `app-modes` 581b2cd since 2026-10-16 (Guillaume
approved). The engine swap goes into `main`, on a Monday (UTC).

1. **Cards.** The app's card is a number 0..53, `suit*13 + (rank−1)`, ace 1 … king 13, Jokers 52
   and 53, with suits **hearts 0, diamonds 1, clubs 2, spades 3** (`src/engine/types.ts`); the
   engine's suits are **clubs 0, diamonds 1, hearts 2, spades 3**, one Joker 52
   (`crates/engine/src/cards.rs`). The adapter swaps the blocks 0–12 ↔ 26–38 and leaves 13–25
   (diamonds), 39–51 (spades) and 52 (the Joker) unchanged; tested both ways (the app's A♥ 0 is
   the engine's 26, and back) over all 53 cards. A match with a second Joker (`config.jokers 2`,
   the app's 53) is refused, as is any table that is not 8 seats.
   **Actions.** App `play{card}` → `Play(card)`; app `rob{discard}` (the ace-of-trumps holder,
   rule 20) and `steal{discard}` (the dealer taking a turned ace or Joker, rules 22 and 24) →
   `Burn(card)`, told apart by the phase; `leaveJoker` is illegal and never a candidate.
   **`declineRob` is not a candidate.** The owner's rule H1 (`docs/RULES.md`): illegal plays are
   impossible and robbing is forced, so rules 25, 26, 34 and 37–41 (turning a partnership, passed
   tricks) are not part of the engine and it cannot value a refusal. The app grades a decline as a
   penalty, as it offered. If Guillaume ever wants a decline valued, that is an owner decision to
   change H1 first.
2. **`trickWinner`**: per seat, length 8, indexed by seat number; `trickWin` stays the pair's
   share.
3. **`expectedPoints` at tricks 1–2**: always a number, never null, on the exact scale (the pair's
   points in the hand, the 11/10 as credited). Exact along the principal line up to the network's
   cut; beyond it, the mean over the outcome head's endings: a hand-over ending gives every team's
   points exactly (5 a trick, the bonus fixed by the position); a game-won ending credits the
   winning pair the points that reach 41 (the game ends there, rule 32) and the other pairs their
   points at the cut (a lower bound, stated in the API doc). Labelled by `meanHorizon`. Passed
   tricks (rule 39) never happen in the engine (H1).
4. **Engine version**: every `Result` carries `engine` = one string, `"<commit>+<network SHA-256 or
   'exact'>+v<values version>"`, for `club_hands.engine` from the first day of the swap.
5. **Server-side review on Cloudflare Workers Paid** (approved 2026-10-16: up to 5 min CPU and
   128 MB per request, 64 MiB code). What it means for the engine:
   - Memory: the equity table (26 MB), the network (6 MB), the WASM code and a transposition table
     per solve must fit in 128 MB. The table size never changes a value (exact search), only the
     time, so the server build uses a 16–32 MB table; the API takes `tt_mb`.
   - Time: a review is split by the seeded shards into requests that each stay under the CPU limit
     and merged by index: byte-identical whatever the split. The engine gives a `plan_only` call
     that returns the shard list with a cost estimate per shard, so the app can size the requests.
   - Trick 1 needs the trick-2 network O_2 (H_2: tenths of a second a deal); with only O_3 (H_3,
     seconds to minutes a deal at trick 1) a trick-1 review does not fit Workers and stays on the
     two-trick Pro until O_2 is proven.
6. **Guide's lab at the player's turn** (today about 4 s on a laptop): the pace numbers (§5.2) are
   given separately for trick 1 and trick 2 decisions, and for tricks 3–5, at 32 / 64 / 128 deals,
   on a phone and on a Worker.

## 7. The questions as asked (answered in §6)

- Card naming: the engine's card index table is in `crates/engine/src/cards.rs`; confirm the app's
  `card` type (string or number) so the adapter maps it once.
- `trickWinner`: per seat or per pair? (The engine can give either.)
- `expectedPoints` at tricks 1–2: the network's expectation beyond the cut, or "unknown" (null)?
  The engine prefers the expectation, labelled by `meanHorizon`.
- Does the Club need the engine's version in `club_hands` from day one of the swap? (Recommended.)

## 7. Turned partnerships (owner decision, 2026-10-17)
The engine's own players never act illegally (house rule H1), but the app's Real Life mode lets a human renege and be
caught, and then rules 39-41 (and 34) apply for the rest of the hand. Decision:
- The engine and the independent reference implementation take a position in which one or more partnerships have
  been turned (the app says which, and when) and apply rules 39-41 and 34 exactly: the turned pair's cards no longer
  count, its won tricks pass with their points to the standing pair that played the strongest card in each, a turned
  leader's lead passes to the next player still in, the turned pair scores nothing for the hand, and a lone standing
  pair takes every trick (a jink). Rules 37-38 (when a pair is caught) stay the app's job.
- The solver skips turned seats; positions after a turn are valued by search (smaller games: 6 or 4 players), not by
  the networks, which have never seen such a position. A separate network for turned hands, or a "turned" input added
  to a network with zero-initialised weights and trained additively, only if measurement shows search is not enough.
- Until this is built and released, the app's own bot plays any hand after a turn.

### 7.1 Built (branch `eng-turned`, 2026-10-17, with the owner's rulings of the same day)
The engine now referees a hand exactly as the app's `src/engine/game.ts` does (`penalise`,
`credit`, `catchHiddenAces`, `resolveTrick`, `advance`, `finishHand`, `afterTurn`), so an app game
replays in the engine card for card. Our own players still never act illegally (H1): nothing below
is among `legal_actions`, and the solver and search never cause it.

**The app's three ways in** (`crates/engine/src/game/referee.rs`; each refused with a code and the
match unchanged when it cannot happen; all recorded in `Match::extras()`, so a record is
`actions()` + `extras()`, replayed by `Match::apply_record` / `Match::replay_record`):
- `turn_team(team)`: a pair turned before the play the match awaits, or, for the pair whose rob is
  due, instead of that rob. The engine's automatic take is undone, as it never happened at the
  app's table: the robber not robbing (Guide, cause `no_rob`), the dealer refusing a turned ace
  (`declineRob`, `declined_rob`: the ace stays on the pack, out of play), the dealer leaving a
  turned Joker (`leaveJoker`, `left_joker`: the Joker stays face up on the deck and the next card
  is turned; an ace turned next stays unrobbed, the app's `afterTurn`).
- `play_then_turn(card, team)`, ruling (a) "play, then turn": the legal card is placed, the pair
  turned, then the trick goes on or is won without the turned cards (cause `with_play`).
- `let_through(lapse)`, for replaying app records only: an illegal card, a play without robbing, a
  rob without the ace (it may burn any card it then holds but the ace of trumps, H3), a burned
  ace. Each is kept as a hidden renege and the engine turns the pair exactly when the app does: a
  later card of that seat proves it (rule 37, `proved`), the ace of trumps shows from another hand
  (`ace_shown`), the pair would reach 41 (rule 38: the credit is not made and the pair is turned
  instead, `claim`, at every credit: a trick, a passed trick, the bonus, the jink), or the hand
  ends (`hand_end`).

Events: `PairTurned { team, score, cause }`; `TrickPassed { trick, team, points }` (0 when the
receiver is caught at the claim); `TrickWon` with 0 points when its winner is caught at the claim
or only turned cards were in it; game-end reason `passed_trick`; `turned_teams` in the public state
(left out of the JSON when nobody is turned). Hands where the app does nothing are unchanged: every
test, the WASM hashes and the normal differential output are identical to the parent commit.

**The rulings as built:**
- **Q2, "first card played takes it"** (the app's `standingWinnerIndex`): a trick to pass goes to
  the card of a pair still in play that beats the others against the card led; when none of their
  cards is a trump or of the suit led, to the first of them played. A trick in which no pair still
  in play played stays with nobody, for no points; when the last pairs are turned mid-trick, the
  trick goes to its best card for no points (the app's `resolveTrick`) and the hand ends with
  nobody scoring it. In the trick being played the case cannot arise (its leader is in play: Q1).
- **Q3, "same as the app"**: passed tricks are credited one at a time in trick order, each checked
  at the claim; the first pair to reach 41 wins the game (`passed_trick`) and later passes are not
  credited.
- **Rule 34 as the app plays it**: the last pair in play plays the hand out alone and wins with a
  jink at the end of the hand (unless caught at the claim). A hand with a turn or a let-through
  ends in the app's order: hidden false robs and burned aces caught, then the bonus (one caught at
  the claim is lost with the pair, as in the app), then the jink (worth what the pair still needs,
  at least 10). After a trick whose winner was turned, the next seat on its left still in play
  leads (the app).

**Still refused: Q1, "ask the pros first"** (`led_trick_in_progress`): a pair turned while the
trick it led is being played, up to the moment that trick is won, whatever turns it (a turn, a
play then a turn, a card that proves its own renege, the ace of trumps showing). The app keeps the
turned card as the card led; the engine waits for the pub's ruling and refuses the step, the match
unchanged; a legal card that would do it is refused by `apply` too, and the app's own bot plays on.
It is not refused when every pair is then turned, or when a passed trick ends the game first. In
the differential test's random referee play (far more reneges than any real table), a legal card is
refused this way in about 8% of hands, and about 12% of matches reach a seat all of whose cards are
refused (the run stops that match there).

**Differences from the app that cannot change the money** (documented, not restructured):
- In a hand where the app did nothing, the engine checks a jink before the end-of-hand bonus (the
  app: bonus, then jink). The jinking pair holds the best trump's trick, so the same pair wins the
  game; only the recorded reason (`jink` or `end_bonus`) can differ.
- `GameWon.scores` never adds a jink's credit (the app's hand points do).
- When a passed trick ends the game the engine stops there; the app's `penalise` still rewrites the
  later tricks' winners and points, and may turn further pairs after the game is won.
- `trick_winners` keeps who won each trick at the table; after passes the holders are in
  `tricks_won` and the `TrickPassed` events (the app rewrites `trick.winner`).
- H3: the engine lets a robber burn the card it has just taken; the app's rob burns from the hand
  before the take. (A real difference in what can be recorded, not in the money of a recorded app
  game: the app never sends such a burn.)
- `ProvedNoAce` and `AceUndealt` are the table's inferences: after a play without robbing let
  through, they can be false until the renege is caught, exactly as at a real table.

**The app's rules text must follow its code (for the app session, not done here):** fivefingers
`docs/rules.md` lines 113-115 say "a card of another suit never wins"; the owner's ruling and the
app's own `standingWinnerIndex` give the trick to the first card played among the pairs still in
play. That text must be corrected in the app repo (and `rulesText.ts`, which holds the same words).
This branch does not edit /home/user/fivefingers.

**Reference** (`crates/reference/src/game/referee.rs`): written separately after the app's code (a
pairwise "beats" from the first standing card, the trick kept whole, its own renege kinds), with
the same events, refusal codes and hidden reneges. **Differential test** (`difftest --turns N`):
random referee steps (turns before a play, with a play and while a rob is due; all four lapses)
before about one play in N and one rob in three, both implementations asked the same, every answer
and the full snapshot (state, hidden reneges, every seat's events) compared after each, and every
match replayed from its record. 20,000 matches (seed 1, `--turns 12`): 378,492 turns and 261,815
lapses tried, 254,743 pairs turned (141,751 before a play, 64,511 with a play, 24,245 proved,
2,126 by the ace shown, 1,127 at a claim, 7,828 at the hand end, 10,603 for not robbing, 2,043
declined robs, 509 left Jokers), 98,298 tricks passed, 5,700 games won by passed tricks, 4,705
hands with every pair turned: all agree, and every match replays from its record; 2,498 matches
stopped at a seat all of whose cards the open question refuses. Without referee steps the run
(20,000 matches) is byte for byte the parent commit's.

**Belief and search.** The card tracker no longer trusts a turned seat's show-outs or "no ace"
proofs (its face-down cards keep its hand size and may be of any suit), nor the robber facts for a
turned robber or for one the cards seen show without the ace (its own plays, the ace seen
elsewhere, the ace taken from the table at its turn, another seat turned for not robbing). It keeps
what a turn itself makes public: the seat turned for not robbing holds the ace; a robber caught at
the claim or the hand end does not; a seat asked to play in trick 1 when a pair was turned before or
with its card lacks it (the engine robs by itself). When the facts still admit no placement (an
uncaught renege the cards give away), every other seat is treated as possibly having broken the
rules. `search::worlds::rebuild` replays a view step by step to match: each turn at its moment
(`turn_team`, or `Match::observe_play` with the card it came with), a turned or untrusted seat's
card as the table saw it (`observe_play`, also undoing a rob the world would make), a turned or
provably false robber's rob as the lapse the world needs, and the engine catches the rest itself.
Tests: every sampled world of 5,000 random referee matches rebuilds to the seat's exact view, the
true deal is never excluded once every renege's pair is turned, turned seats get their face-down
cards with no void constraints (`referee_views_rebuild_and_turned_seats_have_no_voids`,
`a_turned_seats_show_out_proves_nothing`), the tracker's worlds equal the engine's on 25 small
positions (2,500 in the long run) after the referee (`belief_equals_engine_oracle_after_the_referee`),
and exact PIMC on referee views equals the mean of naive solves
(`exact_pimc_on_referee_views_equals_mean_of_solves`).

**Solver**: positions with turned pairs are searched exactly among the seats still in play (6, 4,
or 2 when one pair plays the hand out) and never shown to a network (`FrontierCut::applies`
refuses them, `canonical` panics on them). Exact solve cost on this VM (one thread, default
switches): trick 2 mean 3.0 s with 8 seats, 68 ms with 6, 0.9 ms with 4; trick 3 6.7 / 0.5 /
0.04 ms; trick 4 under 0.03 ms; trick 1 with 6 seats 17 s (max 42 s), with 4 seats 29 ms
(`cargo run --release -p solver --example turned_cost`).
