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
| `candidates[i].action` | `engine::Action` (`Play(card)` or `Burn(card)`) | Same two kinds as the app. Card ids: the engine's 0..53 index (54 cards with the Joker); a table in the API doc maps them to the app's names. |
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
  values in positions where a seat is indifferent depend on that rule; the app's `ties: average`
  is its own smoothing on top and may stay.)

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

## 6. Open questions for the app

- Card naming: the engine's card index table is in `crates/engine/src/card.rs`; confirm the app's
  `card` type (string or number) so the adapter maps it once.
- `trickWinner`: per seat or per pair? (The engine can give either.)
- `expectedPoints` at tricks 1–2: the network's expectation beyond the cut, or "unknown" (null)?
  The engine prefers the expectation, labelled by `meanHorizon`.
- Does the Club need the engine's version in `club_hands` from day one of the swap? (Recommended.)
