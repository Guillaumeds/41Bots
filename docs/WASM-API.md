# The engine in the browser: the WebAssembly API

**Status: DRAFT (Stage 0, 2026-10-17).** This is the interface the app (fivefingers) builds
against while the engine side is built. It becomes final when Stage A lands (the measured
numbers below are filled in then, and any change to a field is listed in a "Changes" section at
the end). The design follows `docs/APP-ENGINE-CONTRACT.md` §1–§3, §5, §6 and §7.1; this file is
its item §5.1, bots first.

## 1. What the app gets

One WebAssembly module, `engine.wasm` [WebAssembly: compiled code a browser or Node runs at near
native speed, the same bytes everywhere], and a tiny loader, `engine.mjs` (plain JavaScript, no
dependencies, works in a browser page, a Web Worker, Node 18+ and Cloudflare Workers). The app
sends its own seat view as JSON and gets the engine's valuation of every legal action back as
JSON. Nothing else: no clocks, no threads, no network access, no randomness of its own. The same
request gives the same bytes on every device.

What it plays today (Stage A):

| Decision | How each deal [one guessed layout of the hidden cards] is valued | Network needed |
|---|---|---|
| Trick 1 (and the dealer's or robber's burn) | refused, `no_network_for_trick` (H_2 once the trick-2 network O_2 exists) | O_2 |
| Trick 2, all 8 seats in play | H_3: exact search to the start of trick 3, then the trick-3 network O_3 | O_3 |
| Tricks 3–5 | exact solve to the end of the hand | none |
| Any trick after a pair was turned | exact among the seats still in play (networks never see these) | none |

## 2. The loader (`engine.mjs`)

```js
import { load } from "./engine.mjs";

const engine = await load(wasmBytesOrModule);        // ArrayBuffer | Uint8Array | WebAssembly.Module
engine.loadEquity(equityBytes, "64740dab…");          // the equity table; SHA-256 checked when given
const net = engine.loadNet(o3Bytes, "f5766aa9…");     // -> { sha256, trick: 3, bytes }
const resultJson = engine.evaluate(requestJsonOrObject);   // -> string (JSON)
const mergedJson = engine.merge([resultJson1, resultJson2]); // -> string (JSON)
engine.info();                                        // -> { engine, api: 1, equity, nets, memoryBytes }
```

- `load` is the only asynchronous call (it instantiates the module). Every other call is
  synchronous and returns or throws at once; run `evaluate` in a Web Worker, never on the page's
  main thread.
- `loadEquity` must come before `loadNet` (the network reads the equity table for its inputs).
  Both throw an `Error` whose message starts with a refusal code (`bad_file`, `sha256_mismatch`).
- `loadNet` reads the cut trick from the file's own metadata (`trick`), so several networks can
  be loaded at once: O_3 serves trick-2 decisions, O_2 (later) trick-1 decisions. Loading a
  second network for the same trick replaces the first.
- `evaluate` and `merge` never throw on a bad request: they return a refusal result (§5).
- A `WebAssembly.Module` (compiled once) can be posted to each Web Worker and passed to `load`.

### The module's raw exports (for anyone not using the loader)

The module imports nothing. Bytes cross the boundary through one Rust-owned buffer: JavaScript
asks for an input buffer of `len` bytes (`inbox(len)`, which returns its address in the module's
`memory`), writes the bytes there, then calls the operation with `len`; the operation's output is
left in a Rust-owned output buffer whose address and length JavaScript reads with `outbox()` and
`outbox_len()`. Rust never reads memory it does not own, so the crate needs no `unsafe` block
(only the `#[unsafe(no_mangle)]` export attribute, per item).

| Export | Signature | Does |
|---|---|---|
| `memory` | WebAssembly.Memory | the module's memory |
| `api_version` | `() -> u32` | 1 |
| `inbox` | `(len: u32) -> u32` | makes the input buffer `len` bytes long; returns its address |
| `load_equity` | `(len: u32) -> u32` | reads the inbox as the equity table; 0 = ok, else the outbox holds an error JSON |
| `load_net` | `(len: u32) -> u32` | reads the inbox as a network; 0 = ok (the outbox holds `{sha256, trick, bytes}`), else an error JSON |
| `evaluate` | `(len: u32) -> u32` | reads the inbox as a request (UTF-8 JSON); the result JSON goes to the outbox; returns its length |
| `merge` | `(len: u32) -> u32` | reads the inbox as a JSON array of results; the merged result goes to the outbox |
| `info` | `() -> u32` | the engine's version, the loaded files and the memory in use, as JSON in the outbox |
| `outbox` | `() -> u32` | address of the output |
| `outbox_len` | `() -> u32` | length of the output |

## 3. The request

The request is built from the app's own data, so the adapter is one line:

```ts
const request = { v: 1, seat, deals: 64, seed: String(seed), match: maskMatch(m, seat) };
```

| Field | Type | Meaning |
|---|---|---|
| `v` | 1 | request format version |
| `seat` | 0..7 | the seat whose view this is. Stage A: must be the seat to act (`match.hand.toAct`), else `not_to_act` |
| `deals` | 1..4096 | N, the deals of the whole search (all shards together) |
| `seed` | string (decimal u64) or number below 2^53 | the decision's seed. Deal *w* is drawn from it and *w* alone, so any split of the deals gives the same values (§6). For the day's match: the same seed on every device. |
| `shard`, `shards` | optional, default 0 and 1 | this call values shard `shard` of `shards` (§6) |
| `ttMb` | optional, 1..256, default 16 | the exact search's memory table in MB. Changes only the speed, never a value |
| `ownBurns` | optional, app card ids | the cards this seat burned this hand, oldest first. Needed only when `match.hand.discards` hides them (see below) |
| `match` | the app's `MatchState` | the seat's view: `maskMatch(m, seat)` (the unmasked match gives the same bytes: the engine reads nothing a seat may not see) |

**What the engine reads from `match`** (everything else is ignored):
- `config`: `n` must be 8 (`not_8_seats`), `jokers` must be 1 (`two_jokers`); `target` 41,
  `gamesToWin` 2, `trickPoints` 5, `bonusExtra` 6, `jinkWins` true, `allowDiscardTrumpAce`,
  `partnerPlaysOn` and `mustTrumpWhenVoid` false, else `not_house_rules`.
- `status` (must be `"playing"`), `gameScores` (the scores before this hand), `gamesWon`,
  `gameIndex`, `handIndex`, `dealer`.
- `hand`: `phase`, `toAct`, `turnedCard`, `turned`, `turnedTakenBy`, `jokerStolen`, `jokerLeft`,
  `hands[seat]` (the seat's own cards; every other hand is read for its size only), the plays of
  `tricks[]` and of `current` (seat and card, in order), `penalised`, `reneges` (the caught
  ones), and the seat's own entries of `discards`.
- The robber's turned card and a stolen Joker are known to everyone; the engine finds them from
  `turnedTakenBy`, `turnedCard` and `jokerStolen`, never from another seat's hand.

**Own burns.** `maskMatch` hides every discard, the seat's own included. A seat that robbed or
stole knows its burned card, and the engine needs it (it is not in anyone's hand). Solo play can
pass the unmasked match (the engine reads only the seat's own entries of `discards`); a table
with friends must add `ownBurns` (the runner's view from the server hides it: the server's
`maskMatch` should keep a seat's own discards). Missing when needed: `own_burns_hidden`.

**Cards and actions.** The app's numbering is used throughout the request and the result (card
= suit×13 + rank−1, hearts 0, diamonds 1, clubs 2, spades 3, the Joker 52). The engine converts
inside (its suits are clubs 0, diamonds 1, hearts 2, spades 3: blocks 0–12 and 26–38 swap,
diamonds, spades and the Joker stay; tested both ways over all 53 cards). Actions:
- `{type:"play", card}` ↔ the engine's `Play(card)`;
- `{type:"rob", discard}` (the ace-of-trumps holder at its turn, or the dealer robbing a turned
  ace) and `{type:"steal", discard}` (the dealer taking a turned Joker) ↔ the engine's
  `Burn(card)`, told apart by the phase. The engine takes the turned card first and then burns,
  so it could burn the card it has just taken; the app's rob burns from the hand before the take
  (contract §7.1, H3). Such a burn has no app action and is never listed.
- `declineRob` and `leaveJoker` are never candidates (house rule H1: robbing is forced).
- A robber's rob is valued together with the card it then plays (the engine's joint decision):
  each candidate is `{type:"rob", discard}` with `then: {type:"play", card}`. (Trick 1: coming
  with O_2.)

### Example: a real trick-2 position

Dealer seat 7, diamonds trumps (the 7♦ turned, robbed by seat 2 with the A♦). Trick 1: seat 0
led the K♠, seat 5 won it with the J♦. Seat 5 led the K♣ in trick 2, seat 6 followed with the
2♣, and seat 7 (pair 3) is to play holding K♦ 10♣ 9♥ 6♣ (its legal cards: 10♣, 6♣, K♦).
Pair 1 has 16 (its 11 spent) and a game; the others none.

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

```json
{
  "v": 1, "status": "ok",
  "engine": "1a2b3c4d5e6f+f5766aa98a1d769e31122c94cc4da9790c84508fab5c432d0c0358fe3124faad+v2",
  "key": "<64 hex: the request's fingerprint>",
  "seat": 7, "pair": 3, "trick": 2, "method": "hybrid", "meanHorizon": 1,
  "deals": 64, "shard": 0, "shards": 1, "dealsFrom": 0, "dealsTo": 64,
  "worlds": 64, "worldCount": "91372305120", "best": 0,
  "candidates": [
    { "action": { "type": "play", "card": 25 }, "equivalent": [{ "type": "play", "card": 25 }],
      "mean": [0.2, 0.31, 0.19, 0.3], "mean_fp": ["858993459", "1331439862", "816043786", "1288490189"],
      "value": 0.3, "se": 0.004, "n": 64, "ess": 64,
      "sum_fp": ["…", "…", "…", "…"], "sumsq_fp": "…" }
  ]
}
```
(The numbers above are illustrative, not measured.)

| Field | Meaning |
|---|---|
| `status` | `"ok"` or `"refused"` (§5) |
| `engine` | `"<commit>+<network SHA-256, or exact>+v<values version>"` (contract §6.4): the source commit the module was built from (12 hex digits), the network that valued this decision (`exact` when none did), and the values version (2 today, `search::measure::VALUES_VERSION`). For `club_hands.engine`. |
| `key` | SHA-256 of what the engine read (its version, the seat's view as the engine rebuilt it, the seat, `deals`, `seed`): equal for every shard of one search; `merge` refuses results with different keys |
| `seat`, `pair` | the viewer and its pair (`seat % 4`, the app's `pairOf`) |
| `trick` | 1..5, the trick of the decision (1 also for the dealer's burn before trick 1) |
| `method` | `"hybrid"` (H_k: exact to the start of trick k, then the network), `"exact"` (to the end of the hand), or `"forced"` (one legal action: nothing searched) |
| `meanHorizon` | tricks searched exactly before the network takes over (`k − trick`: 1 at trick 2 with O_3), or `null` when exact to the end of the hand (the review's "to the end") |
| `deals`, `shard`, `shards`, `dealsFrom`, `dealsTo` | the request's N, and the deals this result covers (`dealsFrom`..`dealsTo`, end excluded) |
| `worlds` | `n` of the best candidate; **0 for a forced card** |
| `worldCount` | the exact number of deals consistent with the seat's view (decimal string), from the card tracker |
| `best` | index of the best candidate in `candidates`: always 0 |
| `candidates[]` | best first: by `value` (high first), ties to the engine's first card (its own card order, clubs, diamonds, hearts, spades), the engine's tie rule. Do not re-sort. |
| `.action` | the app's action (§3) to make |
| `.then` | robs only: the card the engine plays after it (trick 1, coming) |
| `.equivalent` | every action that had exactly this candidate's value in every deal (the action itself included, the engine's choice first): the solver's equivalent cards and any card that never differs. One row stands for them all. |
| `.mean[4]` | the four pairs' chance of the money [chance of winning the match], indexed by the app's pair (`seat % 4`), 0..1, summing to 1 |
| `.mean_fp[4]` | the same as exact fixed-point numbers (units of 2⁻³², summing to exactly 2³²), decimal strings |
| `.value` | `mean[pair]`: the acting pair's chance |
| `.se` | standard error of `value` over the deals (from integers; 0 with one deal) |
| `.n`, `.ess` | deals valued; effective sample size of the deals' weights, equal to `n` (the deals are equally weighted today) |
| `.sum_fp[4]`, `.sumsq_fp` | exact sums over the deals of each pair's fixed-point value, and of the square of the acting pair's: what `merge` adds. Strings. |
| `.trickWin`, `.trickWinner[8]`, `.expectedPoints`, `.gameEnds`, `.plan` | **coming (Stage B)**: contract §1 and §2, from each deal's principal line, merged by `n` |

A forced card (one legal action) gives `method: "forced"`, `worlds: 0`, and one candidate with
`mean`, `mean_fp`, `value` and `se` null and `n` 0. `stake` and `floor` stay the app's
(`matchStake`, `matchFloor`): the engine does not compute games in play.

## 5. Refusals

`{ "v": 1, "status": "refused", "code": "<code>", "reason": "<plain English>", "engine": "…" }`.
The app's own bot plays any refused decision.

| Code | When |
|---|---|
| `bad_request` | not JSON, a field missing, of the wrong type or out of range |
| `not_8_seats` | `config.n` is not 8 |
| `two_jokers` | `config.jokers` is 2 |
| `not_house_rules` | another rule switch differs from the 41 house rules |
| `not_playing` | the match is not `"playing"` or the hand is over |
| `not_to_act` | `seat` is not the seat to act (a view from another seat: coming) |
| `no_equity` | `loadEquity` was not called |
| `no_network_for_trick` | the decision needs a network not loaded: trick 1 (and the burns before and in it) needs O_2, trick 2 with all eight seats in play needs O_3 |
| `too_slow` | after a turn, a trick-1 decision with six seats in play (an exact deal takes about 17 s natively) |
| `led_trick_in_progress` | contract §7.1 Q1: a pair turned while the trick it led is being played (open rules question) |
| `own_burns_hidden` | the seat burned a card and neither `ownBurns` nor `discards` says which |
| `inconsistent_view` | no deal fits the view under the rules, or the view's step the engine refused (e.g. `toAct` not the seat the rules say) |
| `merge_mismatch` | `merge`: results of different requests (`key`), overlapping or missing shards, or a refusal among them |

## 6. Several Web Workers: shards and `merge`

Deal *w* (0 ≤ *w* < N) is drawn from `seed` and *w* only (the native search's
`Rng::derive(seed, [WORLD, w])`), and every value is a whole number. So a search can be split
into `shards` parts and merged with identical bytes:
- shard *i* of *K* values the deals `floor(i·N/K)` to `floor((i+1)·N/K)` (end excluded);
- run shard *i* in worker *i* (`{...request, shard: i, shards: K}`), then
  `engine.merge([r0, r1, …])` in any worker (any order; the loaded files are not needed);
- the merged bytes equal the single call's (`shards: 1`) for every K: tested for K = 1, 2, 3, 4
  and 7. Sums are added exactly, then the means, standard errors and the order are computed
  once from the totals.

## 7. Files the app hosts

| File | SHA-256 | Bytes | Note |
|---|---|---|---|
| `engine.wasm` | in `crates/wasm/dist/SHA256SUMS` | (Stage A) | built from this repo, committed |
| `engine.mjs` | in `crates/wasm/dist/SHA256SUMS` | (Stage A) | the loader, committed |
| equity table | `64740dab7845f0502c93ce195d721e1ffb8aa6ab141739ef43f754690de16808` | 25,920,128 (24.7 MiB) | under Cloudflare's 25 MiB static-file limit; rebuilding it takes 25 s natively, so it is downloaded, not rebuilt |
| O_3, the trick-3 network (2048-1024-256) | `f5766aa98a1d769e31122c94cc4da9790c84508fab5c432d0c0358fe3124faad` | 16,936,384 | from R2 `inputs/sha256/<sha>`; serves trick-2 decisions |
| O_2, the trick-2 network | (about 15 h after the trick-2 run starts) | about 17 MB | serves trick-1 decisions |

The network and the table are not in the repo; the engine's main session hands them over from R2.

**A new network replaces the old by its name**: host every file under its SHA-256
(`nets/<sha256>.safetensors`, `equity/<sha256>.npy`), list the current ones in a small manifest
the app reads (`{ wasm, equity, nets: { "3": sha, "2": sha } }`), and change the manifest to
swap. Files never change under a name, so they can be cached for ever. `loadNet` checks the
SHA-256 when given it, and every result names the network that valued it (`engine`).

## 8. Memory and speed

Per instance (one per Web Worker): the equity table (about 26 MB), each network (weights and
their fast copies: about 2× the file), the transposition table (`ttMb`), and the code. Measured
in Stage A, with a table of the deals that fit 5 s (trick 2) and 3 s (tricks 3–5) for 1–4
workers on the build machine; a phone must be measured by the app (expect 2–4× slower).

## 9. What is coming

- Stage B: `trickWin`, `trickWinner[8]`, `expectedPoints`, `gameEnds`, `plan` (contract §1, §2).
- Trick 1 (H_2) once O_2 exists, the dealer's burn and the robber's joint rob included.
- A view from another seat than the one to act (the app's §5), and `plan_only` (contract §6.5).
