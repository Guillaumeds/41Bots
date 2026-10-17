# The engine in the browser: the WebAssembly API

**Status: built (2026-10-17).** Stage A, the bots' fields, and Stage B, the coach's fields from each
deal's principal line (`trickWin`, `trickWinner`, `expectedPoints`, `gameEnds`, `plan`; asked for
with `lines: true`), tested and measured. The design follows `docs/APP-ENGINE-CONTRACT.md` §1–§3, §5, §6 and §7.1; this file is its item §5.1,
bots first. Changes since the Stage 0 draft are listed at the end.

## 1. What the app gets

One WebAssembly module, `engine.wasm` [WebAssembly: compiled code a browser or Node runs at near
native speed, the same bytes everywhere], and a tiny loader, `engine.mjs` (plain JavaScript, no
dependencies; a browser page, a Web Worker, Node 18+, a Cloudflare Worker). The app sends its own
seat view as JSON and gets the engine's valuation of every legal action back as JSON. Nothing
else: no clocks, no threads, no network access, no randomness of its own, no imports at all. The
same request gives the same bytes on every device, natively or in WebAssembly, for any split of
the work (tested, §10).

| Decision | How each deal [one guessed layout of the hidden cards] is valued | Network needed |
|---|---|---|
| The dealer's burn before trick 1 (`steal`, the dealer's `rob`) | refused, `not_built` | (method to come) |
| Trick 1, the robber's rob included | H_2: exact search to the start of trick 2, then the trick-2 network O_2; refused `no_network_for_trick` until O_2 is loaded | O_2 |
| Trick 2, all 8 seats in play | H_3: exact search to the start of trick 3, then the trick-3 network O_3 | O_3 |
| Tricks 3–5 | exact solve to the end of the hand | none |
| After a pair was turned | exact among the seats still in play (networks never see these); trick 1 with six seats in play is refused (`too_slow`) | none |
| One legal action | nothing searched (`method: "forced"`) | none |

The search is the engine's own PIMC [Perfect-Information Monte Carlo: value every candidate in many
guessed deals as if the cards were face up, and average], exactly as the native search runs it:
the candidates of `search::pimc::candidates`, deal *w* drawn by the card tracker from
`Rng::derive(seed, [WORLD, w])` and rebuilt by `search::pimc::world_match`, each deal valued by
`search::pimc::exact_world` (with the network as the frontier for H_k). Nothing in the API computes
a value of its own.

## 2. The loader (`engine.mjs`)

```js
import { load } from "./engine.mjs";

const engine = await load(wasmBytesOrModule);          // ArrayBuffer | typed array | WebAssembly.Module
engine.loadEquity(equityBytes, "64740dab…");            // -> { sha256, bytes }; SHA-256 checked when given
engine.loadNet(o3Bytes, "f5766aa9…");                   // -> { sha256, trick: 3, bytes }
const resultJson = engine.evaluate(requestObjectOrJson); // -> string (JSON)
const mergedJson = engine.merge([result0, result1]);     // objects or JSON strings -> string (JSON)
engine.info();  // -> { engine, api: 1, valuesVersion: 2, equity, nets: { "3": sha }, memoryBytes }
```

- `load` is the only asynchronous call. Every other call is synchronous: run `evaluate` in a Web
  Worker, never on the page's main thread.
- `loadEquity` comes before `loadNet` (the network reads the equity table for its inputs); loading
  the table again drops the networks. A wrong file throws an `Error` whose message starts with its
  code: `bad_file`, `sha256_mismatch` (the expected SHA-256 is checked before anything is loaded),
  `no_equity`.
- `loadNet` reads the cut trick from the file's own metadata (`trick`): O_3 serves trick-2
  decisions, O_2 trick-1 decisions; several can be loaded; a second network for the same trick
  replaces the first.
- `evaluate` and `merge` never throw on a bad request: they return a refusal (§5).
- If the module ever traps (an engine bug), that engine object refuses every later call with
  `engine_dead: …`: load a new one. A loader older or newer than the module throws `api_mismatch`.
- A compiled `WebAssembly.Module` can be posted to each Web Worker and passed to `load`.

### The module's raw exports (for anyone not using the loader)

The module imports nothing. Bytes cross through two buffers the module owns: JavaScript asks for an
input buffer of `len` bytes (`inbox(len)` returns its address in the exported `memory`), writes the
bytes there, then calls the operation with `len`; the output is read at `outbox()` for
`outbox_len()` bytes. Why this way: Rust never reads or writes memory it does not own (the two
vectors live in statics behind a mutex), so the crate needs no `unsafe` block, only the
`#[unsafe(no_mangle)]` attribute on each export (allowed item by item under the crate's `deny`);
one copy each way, so even the 26 MB table loads in a quarter of a second.

| Export | Signature | Does |
|---|---|---|
| `memory` | WebAssembly.Memory | the module's memory |
| `api_version` | `() -> u32` | 1 |
| `inbox` | `(len) -> u32` | makes the input buffer `len` bytes long; returns its address |
| `expect_sha256` | `(len) -> u32` | the input (64 lowercase hex digits) is the SHA-256 the next loaded file must have: 0, or 1 with a refusal in the output |
| `load_equity` | `(len) -> u32` | the input is the equity table: 0 (output `{sha256, bytes}`) or 1 (output: a refusal) |
| `load_net` | `(len) -> u32` | the input is a network: 0 (output `{sha256, trick, bytes}`) or 1 (output: a refusal) |
| `evaluate` | `(len) -> u32` | the input is a request (UTF-8 JSON); the result goes to the output; returns its length |
| `merge` | `(len) -> u32` | the input is a JSON array of results; the merged result goes to the output |
| `info` | `() -> u32` | the engine's version, the loaded files and the memory in use, as JSON in the output |
| `outbox`, `outbox_len` | `() -> u32` | the output's address and length |

(`result_hash` is the M0 determinism check's export, not part of the API.)

## 3. The request

Built from the app's own data, so the adapter is one line:

```ts
const request = { v: 1, seat, deals: 64, seed: String(seed), match: maskMatch(m, seat) };
```

| Field | Type | Meaning |
|---|---|---|
| `v` | 1 | request format version |
| `seat` | 0..7 | the seat whose view this is: must be the seat to act (`match.hand.toAct`), else `not_to_act` |
| `deals` | 1..4096 | N, the deals of the whole search (all shards together) |
| `seed` | decimal string (u64) or a number below 2^53 | the decision's seed. Deal *w* is drawn from it and *w* alone. For the day's match: the same seed on every device, e.g. a hash of the day's seed and the position. |
| `shard`, `shards` | optional, default 0 and 1; `shards` ≤ `deals` | this call values shard `shard` of `shards` (§6) |
| `ttMb` | optional, 1..1024, default 16 | the exact search's memory table in MB: only the speed, never a value (measured: 2 to 64 MB change the time by about 10%) |
| `ownBurns` | optional, app card ids, oldest first | the cards this seat burned this hand, when `match.hand.discards` hides them |
| `lines` | optional, default false | also the coach's fields from each deal's principal line (§4.1): every candidate then has its own row. Costs about 10% more at trick 2 and about 3× at tricks 3–5 (measured, §8); the bots leave it off. |
| `match` | the app's `MatchState` | the seat's view: `maskMatch(m, seat)`. The unmasked match gives the same bytes (tested): the engine reads nothing a seat may not see. |

**What the engine reads from `match`** (everything else is ignored):
- `config`: `n` must be 8 (`not_8_seats`), `jokers` 1 (`two_jokers`); `target` 41, `gamesToWin` 2,
  `trickPoints` 5, `bonusExtra` 6, `jinkWins` true, `allowDiscardTrumpAce`, `partnerPlaysOn` and
  `mustTrumpWhenVoid` false, when present, else `not_house_rules`.
- `status` (`"playing"`), `gameScores` (the scores before this hand), `gamesWon`, `gameIndex`,
  `handIndex`, `dealer`.
- `hand`: `phase`, `toAct`, `dealer`, `trump` (checked), `turnedCard`, `turned`, `turnedTakenBy`,
  `jokerStolen`, `jokerLeft`, `hands[seat]` (the seat's own cards; every other hand only for its
  length, checked against the plays), the `seat`/`card` of every play in `tricks[]` and `current`
  in order, `penalised`, the caught `reneges` (`seat`, `kind`, `caughtAt` ≥ 0), and the seat's own
  entries of `discards`.
- The robber's turned card and a stolen Joker are known to everyone: the engine finds them from
  `turnedTakenBy`, `turnedCard` and `jokerStolen`, never from another seat's hand.

**Own burns.** `maskMatch` hides every discard, the seat's own included. A seat that robbed or
stole knows its burned card, and the engine needs it (that card is in nobody's hand). Solo play
can pass the unmasked match (the engine reads only the seat's own entries of `discards`); a table
with friends must add `ownBurns`, or the server's `maskMatch` should keep a seat's own discards
(an app change). Missing when needed: `own_burns_hidden`.

**Cards and actions.** The app's numbering is used throughout (card = suit×13 + rank−1, hearts 0,
diamonds 1, clubs 2, spades 3, the Joker 52); the engine converts inside (its suits are clubs 0,
diamonds 1, hearts 2, spades 3: the blocks 0–12 and 26–38 swap; tested both ways over all 53
cards). Actions:
- `{type:"play", card}` ↔ the engine's `Play(card)`;
- `{type:"rob", discard}` (the ace-of-trumps holder at its turn, the dealer robbing a turned ace)
  and `{type:"steal", discard}` (the dealer taking a turned Joker) ↔ the engine's `Burn(card)`,
  told apart by the app's phase. The engine takes the turned card first and then burns, so it could
  burn the card it has just taken; the app's rob burns from the hand before the take (contract
  §7.1, H3): that burn has no app action and is never listed or searched.
- A robber's rob is valued together with the card it then plays (the engine's joint decision):
  each candidate is `{type:"rob", discard}` with `then: {type:"play", card}`. Make the rob; the
  play after it is a new request (its value may differ a little: the view now has the burn).
- `declineRob` and `leaveJoker` are never candidates (house rule H1: robbing is forced).

**How the view is rebuilt.** The engine plays the hand in itself from one deal that fits what the
seat saw (every hidden card placed so that each play was legal unless the cards seen give it away
as a renege, the robber holds the ace of trumps, nobody who played in trick 1 while the turned
card lay on the table holds it), with the app's referee steps where the app made them (a pair
turned at its rob, a refused ace, a left Joker; the lapses now known: a caught false rob, burned
ace or play without robbing, a card the seen cards prove illegal). The engine catches reneges
itself exactly as the app does (contract §7.1). The replay must then agree with the app's own
state (the seat to act, the phase, the pairs turned, every hand's size, the seat's cards), else
`inconsistent_view`. The deal is only a vehicle: any deal that fits gives the same view (tested
against the engine's own view of the seat on every decision of random matches, with and without
turned pairs), and the search draws its own deals from that view.

### Example: a real trick-2 position

Dealer seat 7, diamonds trumps (the 7♦ turned, robbed by seat 2 with the A♦). Trick 1: seat 0
led the K♠, seat 5 won it with the J♦. Seat 5 led the K♣ in trick 2, seat 6 followed with the
2♣, and seat 7 (pair 3) is to play holding K♦ 10♣ 9♥ 6♣ (its legal cards: 10♣, 6♣, K♦).
Pair 1 has 16 (its 11 spent) and a game; the others none. (This exact request is a test.)

```json
{
  "v": 1, "seat": 7, "deals": 64, "seed": "20261017",
  "match": {
    "config": { "n": 8, "target": 41, "gamesToWin": 2, "trickPoints": 5, "bonusExtra": 6,
      "jinkWins": true, "allowDiscardTrumpAce": false, "partnerPlaysOn": false, "jokers": 1,
      "jokerMode": "firstWins", "mustTrumpWhenVoid": false },
    "seed": 0, "pairs": 4, "gameScores": [10, 16, 5, 0], "gamesWon": [0, 1, 0, 0],
    "gameIndex": 1, "handIndex": 7, "dealer": 7, "status": "playing",
    "lastGameWinner": 1, "matchWinner": -1,
    "hand": {
      "index": 7, "seed": 0, "n": 8, "dealer": 7, "trump": 1,
      "turned": -1, "turnedCard": 19, "jokerTurned": 0, "jokerStolen": [], "jokerLeft": [],
      "jokerMode": "firstWins", "turnedTakenBy": 2,
      "hands": [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, 19], [-1, -1, -1, -1],
                [-1, -1, -1, -1], [-1, -1, -1], [-1, -1, -1], [25, 35, 8, 31]],
      "pack": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], "discards": [-1], "turnedDown": [],
      "tricks": [{ "leader": 0, "winner": 5, "points": 5, "bonus": false, "bonusTo": -1,
        "plays": [{ "seat": 0, "card": 51 }, { "seat": 1, "card": 42 }, { "seat": 2, "card": 14 },
                  { "seat": 3, "card": 50 }, { "seat": 4, "card": 47 }, { "seat": 5, "card": 23 },
                  { "seat": 6, "card": 45 }, { "seat": 7, "card": 40 }] }],
      "current": { "leader": 5, "winner": -1, "points": 0, "bonus": false, "bonusTo": -1,
        "plays": [{ "seat": 5, "card": 38 }, { "seat": 6, "card": 27 }] },
      "penalised": [false, false, false, false, false, false, false, false],
      "robbed": [false, false, true, false, false, false, false, false],
      "voidedPairs": [false, false, false, false], "reneges": [], "handPoints": [0, 5, 0, 0],
      "bestTrump": { "card": 23, "seat": 5, "trickIndex": 0 }, "bonusSettled": false,
      "jink": -1, "phase": "play", "toAct": 7
    }
  }
}
```

## 4. The result

A real result (the trick-2 check request `t2-hybrid-b`: seat 7 to play, 3 deals, O_3; the key
and the three other candidates shortened):

```json
{"v":1,"status":"ok","engine":"<commit>+f5766aa98a1d769e31122c94cc4da9790c84508fab5c432d0c0358fe3124faad+v2",
 "key":"d76c276b…","seat":7,"pair":3,"trick":2,"method":"hybrid","meanHorizon":1,
 "deals":3,"shard":0,"shards":1,"dealsFrom":0,"dealsTo":3,"worlds":3,
 "worldCount":"152385422204181534568200000000","best":0,
 "candidates":[
  {"action":{"type":"play","card":17},"equivalent":[{"action":{"type":"play","card":17}}],
   "mean":[0.17431352566927671,0.17309287888929248,0.17657288792543113,0.4760207075159997],
   "mean_fp":["748670892","743428254","758374779","2044493371"],
   "value":0.4760207075159997,"se":0.042460351483896375,"n":3,"ess":3,
   "sum_fp":["2246012677","2230284762","2275124337","6133480112"],"sumsq_fp":"12739403186191007062"},
  …]}
```

| Field | Meaning |
|---|---|
| `status` | `"ok"` or `"refused"` (§5) |
| `engine` | `"<commit>+<network SHA-256, or exact>+v<values version>"` (contract §6.4): the source commit the module was built from (12 hex digits, `-dirty` if built from uncommitted code, `unversioned` for a plain `cargo` build), the network that valued this decision (`exact` when none did: tricks 3–5, turned pairs, forced cards), and the values version (2, `search::measure::VALUES_VERSION`). For `club_hands.engine`. |
| `key` | SHA-256 of what the engine read (its version string, the seat's view as the engine rebuilt it, the seat, `deals`, `seed`): equal for every shard of one search; `merge` refuses different keys |
| `seat`, `pair` | the viewer and its pair (`seat % 4`, the app's `pairOf`) |
| `trick` | 1..5 (1 also for the dealer's burn before trick 1) |
| `method` | `"hybrid"` (H_k), `"exact"` (to the end of the hand) or `"forced"` |
| `meanHorizon` | tricks searched exactly before the network takes over (`k − trick`: 1 at trick 2 with O_3), or `null` when exact to the end of the hand or forced |
| `deals`, `shard`, `shards`, `dealsFrom`, `dealsTo` | the request's N and the deals this result covers (end excluded) |
| `worlds` | `n` of the best candidate; **0 for a forced card** |
| `worldCount` | the exact number of deals consistent with the seat's view, from the card tracker (a decimal string; `null` for a forced card) |
| `best` | always 0: `candidates` is sorted best first |
| `candidates[]` | by `value`, high first; ties to the engine's first card in its own card order (clubs, diamonds, hearts, spades), the engine's tie rule. **Do not re-sort.** |
| `.action` | the app's action to make (§3) |
| `.then` | robs only: the card the engine plays after the rob |
| `.equivalent` | every action whose value was exactly this one's in every deal searched (the action itself first): the solver's equivalent cards and any card that never made a difference. One row stands for all of them. |
| `.mean[4]` | the four pairs' chance of the money [chance of winning the match], by the app's pair (`seat % 4`), 0..1, summing to 1 (exact: each is a fixed-point value ÷ 2³²) |
| `.mean_fp[4]` | the same as fixed-point numbers (units of 2⁻³², summing to exactly 2³²), decimal strings |
| `.value` | `mean[pair]`: the acting pair's chance |
| `.se` | standard error of `value` over the deals, computed in whole numbers (0 with one deal) |
| `.n`, `.ess` | deals valued; effective sample size of the deals' weights (`(Σw)²/Σw²`), equal to `n` (the deals are equally weighted today) |
| `.sum_fp[4]`, `.sumsq_fp` | exact sums over the deals of each pair's fixed-point value and of the square of the acting pair's: what `merge` adds (strings) |
| `.trickWin`, `.trickWinner[8]`, `.expectedPoints[4]`, `.gameEnds`, `.plan`, `.lines_fp` | with `lines: true` only (§4.1) |

### 4.1 The coach's fields (`lines: true`)

The **principal line** of a candidate in a deal is the line its value comes from: after the
candidate every seat plays the card the solver chooses there (its pair's best, ties to the first
card, the engine's tie rule), a robber burns its pair's best card. The engine walks it by solving
again at each step with the same evaluator and frontier, and the tests check that each line ends
exactly where the deal's value comes from (the leaf after the hand, or the network's value at the
frontier). The line stops at the end of the hand (exact search) or at the network's frontier
(the hybrid); beyond the frontier the network's outcome head [its estimate of how likely each way
of ending the hand is] gives the endings' chances.

| Field | Meaning |
|---|---|
| `.trickWinner[8]` | per seat (0..7), the share of deals whose line has that seat win the current trick |
| `.trickWin` | the share of deals whose line has the acting seat's pair win it (the sum of its two seats) |
| `.expectedPoints[4]` | per pair (`seat % 4`), the mean over deals of its points in this hand on the line: exact where the line reaches the end of the hand (11/10 as credited, a jink as the app credits it: what the pair needs, at least 10); beyond the frontier the mean over the network's endings: a hand that ends gives every pair its points exactly (5 a trick, the bonus fixed by the position); a game won by a pair credits it the points that reach 41 and every other pair its points at the frontier (a lower bound, contract §6.3). A turned pair scores 0. |
| `.gameEnds` | the share of deals whose line ends a game in this hand (beyond the frontier: the network's chance of a game-won ending) |
| `.plan` | `{ "<app card>": [5 shares] }` for every card of the acting seat's hand: the share of deals whose line plays that card in trick 1..5 (index 0..4). The candidate's own card is 1 in the current trick; robs are not counted; a card the line does not reach before the frontier has shares summing below 1 (contract §2) |
| `.lines_fp` | the exact sums behind them (`winner` counts per seat, `points` per pair in units of 2⁻³² points, `gameEnds` in units of 2⁻³², `plan` counts): what `merge` adds |

Every share and mean is computed in whole numbers (the network's chances from its integer softmax
weights) and divided once at the end, so merged shards give the same bytes. With `lines: true`
every candidate has its own row (its line is its own), and `equivalent` still lists the actions of
equal value in every deal. Results with and without the lines never merge.

A forced card gives `method: "forced"`, `worlds: 0`, one candidate with `mean`, `mean_fp`,
`value`, `se`, `sum_fp`, `sumsq_fp` null and `n` 0; it needs no file loaded. `stake` and `floor`
stay the app's (`matchStake`, `matchFloor`).

## 5. Refusals

`{ "v": 1, "status": "refused", "code": "<code>", "reason": "<plain English>", "engine": "<commit>" }`.
The app's own bot plays any refused decision.

| Code | When |
|---|---|
| `bad_request` | not JSON, a field missing, of the wrong type or out of range (`deals` 0 or over 4096, `shards` over `deals`, a seed that is no u64) |
| `not_8_seats` | `config.n` is not 8 |
| `two_jokers` | `config.jokers` is 2, or a card 53 appears |
| `not_house_rules` | another rule switch differs from the 41 house rules (or a `trumpWhenVoid` renege appears) |
| `not_playing` | the match is not `"playing"`, the hand is over, or nobody is to act |
| `not_to_act` | `seat` is not the seat to act (a view from another seat: coming) |
| `no_equity` | the equity table is not loaded |
| `no_network_for_trick` | the decision needs a network not loaded: trick 1 needs O_2, trick 2 (all eight seats in play) needs O_3 |
| `not_built` | the dealer's burn before trick 1 (`steal`, or robbing a turned ace at once) |
| `too_slow` | after a turn, a trick-1 decision with six seats in play (an exact deal takes about 17 s natively) |
| `led_trick_in_progress` | contract §7.1 Q1: the view needs a pair turned while the trick it led is being played (open rules question) |
| `own_burns_hidden` | the seat burned a card and neither `ownBurns` nor the seat's `discards` says which |
| `inconsistent_view` | the view cannot happen under the rules (no deal fits it, the engine refuses one of its steps, or the replay disagrees with the app's own state), or the seat itself has a renege nobody has caught |
| `merge_mismatch` | `merge`: results of different searches (`key`), shards missing or twice, or refusals that differ (identical refusals merge to that refusal) |
| `bad_file`, `sha256_mismatch` | `loadEquity` / `loadNet`: not a complete table or a network this engine reads; not the expected SHA-256 |

## 6. Several Web Workers: shards and `merge`

Deal *w* (0 ≤ *w* < N) is drawn from `seed` and *w* only, and every value is a whole number. So:
- shard *i* of *K* values the deals `floor(i·N/K)` to `floor((i+1)·N/K)` (end excluded; K ≤ N);
- run shard *i* in worker *i* (`{...request, shard: i, shards: K}`), then `engine.merge([r0, …])`
  in any worker, in any order (the loaded files are not needed to merge);
- the merged bytes equal the single call's (`shards: 1`) for every K: tested for K = 1, 2, 3, 4
  and 7, natively, in WebAssembly, and with four real worker threads. Sums are added exactly; the
  means, standard errors, equivalences and the order are computed once from the totals.
- Each worker needs its own instance with the files loaded (§8 for memory). Contiguous shards can
  be uneven (deals differ in cost); the slowest shard sets the time.

## 7. Files the app hosts

| File | SHA-256 | Bytes | Note |
|---|---|---|---|
| `engine.wasm` | `7d928e322b31f5be57154d8843d6936e23f80bf6e20809b520e5dc66a5a0a290` (built from a708c7691e0d; the current one is in `crates/wasm/dist/SHA256SUMS`) | 754,647 | built by `tools/wasm_build.sh` from a committed tree; committed in `crates/wasm/dist/` |
| `engine.mjs` | `512882afaa788b08c0c067dbdaebb1f954cb4a2d5282cc8e61f3461cf007400e` | 4,691 | the loader (`crates/wasm/js/engine.mjs`) |
| equity table `equity-v0.npy` | `64740dab7845f0502c93ce195d721e1ffb8aa6ab141739ef43f754690de16808` | 25,920,128 (24.7 MiB) | under Cloudflare's 25 MiB static-file limit; rebuilding it takes 25 s natively, so it is downloaded, not rebuilt |
| O_3, the trick-3 network (2048-1024-256) | `f5766aa98a1d769e31122c94cc4da9790c84508fab5c432d0c0358fe3124faad` | 16,936,384 | from R2 `inputs/sha256/<sha>`; serves trick-2 decisions |
| O_2, the trick-2 network | when its run delivers | about 17 MB | serves trick-1 decisions |

The network and the table are not in the repo; the engine's main session hands them over from R2.

**A new network replaces the old by its name**: host every file under its SHA-256
(`nets/<sha256>.safetensors`, `equity/<sha256>.npy`, `engine/<sha256>.wasm`), list the current
ones in a small manifest the app reads (`{ wasm, equity, nets: { "3": sha, "2": sha } }`), and
change the manifest to swap. Files never change under a name, so they can be cached for ever.
`loadNet(bytes, sha)` checks the SHA-256 before loading, and every result names the network that
valued it (`engine`).

## 8. Memory and speed (measured 2026-10-17 on the build VM: 4 vCPU, Node 22, one thread per instance)

**Loading** (WebAssembly): the module instantiates in under 10 ms; the equity table loads in
0.23–0.27 s and O_3 in 0.16–0.22 s (both with their SHA-256 checked).

**Memory per instance** (one per Web Worker): 106 MB of WebAssembly memory once the table and O_3
are loaded (the table 26 MB, the network's weights and their fast copy, the peak of reading the
table), 106–123 MB after searching with the default `ttMb` 16; `ttMb` 64 takes it to 174 MB. The
memory of a WebAssembly instance never shrinks. Four workers: about 0.5 GB.

**Milliseconds per deal, one instance**, by place in the trick (the mean of 4 random positions per
place; native = this crate's `engine-native`, x86-64 with the network's AVX2 kernels; WebAssembly
plain and with `+simd128`):

| | 1st (leader) | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th |
|---|---|---|---|---|---|---|---|---|
| trick 2, H_3: native / wasm / wasm+simd128 | 73 / 368 / 431 | 25 / 128 / 141 | 6.7 / 37 / 39 | 5.0 / 24 / 31 | 3.1 / 12 / 17 | 1.2 / 4.1 / 4.6 | 0.75 / 2.5 / 2.8 | 0.38 / 0.87 / 0.98 |
| trick 3, exact: native / wasm / wasm+simd128 | 6.5 / 11.4 / 13.7 | 2.2 / 4.1 / 5.5 | 1.2 / 2.4 / 3.3 | 0.82 / 1.6 / 2.2 | 0.48 / 0.93 / 1.2 | 0.19 / 0.33 / 0.43 | 0.16 / 0.29 / 0.38 | 0.10 / 0.18 / 0.23 |
| trick 4, exact: native / wasm | 0.06 / 0.12 | 0.03 / 0.05 | 0.02 / 0.04 | 0.03 / 0.06 | 0.02 / 0.03 | 0.01 / 0.03 | 0.01 / 0.03 | 0.01 / 0.03 |

- **Target features: plain (no `simd128`) is shipped.** The `+simd128` build gives the same bytes
  (checked on 107 requests) but is 5–35% *slower*: the compiler's automatic vectorising of the
  network's 64-bit integer sums does not pay in WebAssembly.
- Where the time goes at trick 2 in WebAssembly: 89% in the network's two integer kernels
  (`nn::kernel`, no SIMD there); natively those kernels use AVX2, hence the 5×. A hand-written
  WebAssembly SIMD kernel (same integers, same bytes) is the one lever left; it is not built.
- **The coach's lines** (`lines: true`), one instance in WebAssembly, ms per deal without / with:
  trick 2 leader 337 / 375, second 230 / 253, fourth 31 / 31 (the line's searches to the frontier
  mostly reuse the network's memo); trick 3 leader 17 / 54, second 8.9 / 27, fourth 3.7 / 5.8
  (the line is searched to the end of the hand).
- The network's memo of recent positions lives as long as the instance: a later request that
  meets the same frontier positions (the same deals again, or the next decision of the same hand)
  runs faster. It never changes a value.
- Several workers (measured with real worker threads, same bytes as one): trick 2 leader 1.6× /
  2.1× / 2.2× faster with 2 / 3 / 4 workers (deals cost unevenly, the slowest shard decides, and
  four copies of the network share the memory bus); trick-2 second seat 2.2× / 2.5× / 3.4×;
  trick 3 2.0× / 3.1× / 3.9×.

**Deals that fit the bots' budget on this VM** (5 s at trick 2, 3 s at tricks 3–5; from the
slowest of the 4 positions per place and the measured worker speed-ups above; capped at 4096):

| trick | workers | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 11 | 22 | 98 | 172 | 262 | 806 | 1556 | 4096 |
| 2 | 2 | 17 | 35 | 155 | 272 | 414 | 1274 | 2459 | 4096 |
| 2 | 3 | 23 | 47 | 205 | 360 | 548 | 1685 | 3252 | 4096 |
| 2 | 4 | 25 | 49 | 216 | 379 | 577 | 1774 | 3424 | 4096 |
| 3 | 1 | 194 | 497 | 875 | 1514 | 2051 | 4096 | 4096 | 4096 |
| 3 | 2 | 381 | 975 | 1716 | 2967 | 4020 | 4096 | 4096 | 4096 |
| 3 | 3 | 602 | 1543 | 2715 | 4096 | 4096 | 4096 | 4096 | 4096 |
| 3 | 4 | 754 | 1931 | 3398 | 4096 | 4096 | 4096 | 4096 | 4096 |
| 4, 5 | 1–4 | 4096 | 4096 | 4096 | 4096 | 4096 | 4096 | 4096 | 4096 |

**Caveat: a phone must be measured by the app** (expect 2–4× slower than this VM, and fewer
cores). Budgets are deals, never time (contract §3): for the same bots everywhere the app fixes a
table of deals by trick and place that its slowest supported phone meets, e.g. from
`tools/wasm_workers.mjs` run in the phone's browser, or by timing a few requests per place once.

## 9. What is coming

- Trick 1 works as soon as O_2 is loaded (plays and the robber's rob, H_2); the dealer's burn
  before trick 1 needs a method first (`not_built`).
- A view from another seat than the one to act (the app's §5), `plan_only` (contract §6.5), and a
  WebAssembly SIMD kernel for the network.

## 10. Tests (`cargo test -p wasm`, `tools/wasm_check.sh`)

- **Native and WebAssembly give the same bytes** (`tools/wasm_check.sh` with
  `WASM_EQUITY=<equity .npy> WASM_NET3=<O_3>`): eleven requests from `engine-native fixtures` (a
  dealer's steal, a robber's rob and a trick-1 play, all refused; trick 2 with O_3 twice; tricks 3
  and 4 exact; two forced cards; a pair turned at tricks 2 and 3), valued natively, as plain
  WebAssembly and with `simd128` (`WASM_SIMD=1`): identical. The 96 speed requests too.
- **Any split gives the same bytes**: 2, 3, 4 and 7 shards merged in Node equal the native single
  call; in `cargo test`, 1, 2, 3, 4 and 7 shards in any order, as objects or strings, at trick 2
  (hybrid), 3, 3 with a pair turned, and 5 (forced); missing or doubled shards and different
  searches are refused.
- **The API's values are the native search's**: on the seat's own engine view, the same seed and
  deals, the candidates' fixed-point means equal `search::pimc::evaluate` (exact, tricks 3–4) and
  the yardstick's per-deal `exact_world` with the network as frontier (trick 2 with O_3, trick 1
  and the robber's joint rob with O_2, a turned pair), and the best is the native choice.
- **The rebuilt view is the engine's own**: on every decision of random matches (12 plain, 40
  with the app's referee steps; the long run: 1,600 matches, 566,910 views, 34,648 of them with a
  pair turned: pairs turned at a rob, refused aces, left Jokers, illegal cards,
  plays without robbing, false robs, burned aces), the view rebuilt from the app's JSON equals the
  engine's view of the seat (hand, burns, public state, every event of the hand), masked or not;
  a hidden own burn is refused `own_burns_hidden`, a seat's own uncaught renege
  `inconsistent_view`. A longer run: `WASM_API_MATCHES=400 cargo test --release -p wasm --test
  api -- --ignored`.
- **The coach's lines**: each line's end value equals its deal's value for every candidate and
  deal (trick 2 with O_3, tricks 3 and 4, a turned pair, trick 1 with O_2 including a robber's
  rob); the shares add up (the candidate's card in its trick, each card at most once, the trick's
  winners at most 1); 2, 3, 4 and 7 shards merge to the same bytes; three of the check's requests
  ask for the lines, natively and in WebAssembly.
- **Cards and actions**: all 53 cards both ways; plays in the app's numbers; a robber's candidates
  are a rob of each of its five cards but the ace of trumps, never of the card it takes, each with
  its play.
- **Refusals**: every code above from a request that causes it; every code the sources use is
  listed here.

## Changes since the Stage 0 draft

- New refusal codes `not_built` (the dealer's burn before trick 1), `bad_file`,
  `sha256_mismatch`; the loader's `api_mismatch` and `engine_dead`; the export `expect_sha256`.
- Trick 1 is valued as soon as O_2 is loaded (it was "coming").
- `merge` of identical refusals returns that refusal.
- `worldCount` and the forced card's fields are `null` for a forced card.
- Stage B: the request's `lines` and the fields of §4.1 (`expectedPoints` per pair, as the app's
  `CandidateStats` has it).
- A dealer who stole the Joker and burned it when robbing is read (the long test found it); for
  another seat that burn stays a face-down card in the engine's view, as the engine's own view has
  it (the app shows the Joker gone from the dealer's hand: slightly less information, never a
  wrong one).
