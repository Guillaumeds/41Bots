# Status

**Current checkpoint: M0** (PLAN §13, as revised on 2026-10-13): the rules engine, independent reference implementation, deterministic RNG, CI.

## Done
- Research, design, tech stack and run-operations plan approved (`docs/PLAN.md`).
- Authoritative rules and house rules recorded (`docs/RULES.md`).
- Repo skeleton: pinned Rust toolchain, Cargo workspace, CI workflow, `.gitattributes`.
- **2026-10-13: plan reviewed and amended** (`docs/REVIEW-2026-10-13.md`; amendments at the top of `docs/PLAN.md`). Owner decisions: no ace may ever be burned (H3); think time 8 s / 5 s on tricks 1–2; M3b playable demo; equal tournament mix weights; `bonus_later` switch.

## M0 task list (do in order)
1. **`crates/engine`: deterministic RNG.**
   - SplitMix64 for seed expansion, plus xoshiro256** for generation.
   - An unbiased bounded integer function using `u64` rejection sampling, with no `usize` inside the RNG.
   - Our own Fisher–Yates shuffle.
   - Golden vectors, cross-checked against an independent Python implementation (`tools/rng_golden.py`, stdlib only), stored as test data.
2. **`crates/engine`: cards and rules.**
   - 53-card bitboards (`u64`).
   - Effective suit, trump order (red/black, hearts = 14 trumps), plain-suit orders.
   - Legal-move generation, covering plain leads, trump leads (the Joker and the A♥ led are trump leads), the reneging test, and rule 18.
   - Trick winner.
   - Legal burns: **never an ace of any suit** (H3); the taken card and the Joker may be burned.
3. **`crates/engine`: hand, game and match flow.**
   - Frozen deal convention: deck positions 0–15 two at a time to seats dealer+1 … dealer+8, positions 16–39 three at a time, position 40 turned, 41 the second turned card after a Joker, the rest the undealt pack in order; first dealer = `bounded(8)` from the seeded RNG before the first shuffle; every hand reshuffles all 53.
   - Robbing per RULES H2–H4, with burn decisions as actions and forced takes done automatically; a non-ace-holder is never offered a burn.
   - Scoring: 5 per trick; best-trump bonus 11 first / 10 later per team per game (`bonus_later` switch: 5 house, 6 flat-11); immediate bonus when the 5 of trumps is played; end-of-hand bonus; no bonus if no trump was played; per-trick check order from the RULES notes.
   - Game ends the instant a team reaches 41 or more; jink; match = first to 2 games (at most five); next deal always moves left; money settlement (stake configurable). Record a game-end reason (41-by-trick, 41-by-immediate-bonus, 41-by-end-bonus, jink).
   - An event stream with **visibility tags** (public / private to seat s) on every event; each seat's view is derived from the tags. Events carry the incremental hard facts (who has proved they lack the ace of trumps; burn locations and counts; hidden-card count 12 or 13).
   - A `rules_version` field in every record; a test-only constructor for custom deals.
4. **`crates/reference`.** A naive, independent second implementation of the same rules (Vec-based, ordered lists, written straight from RULES.md). Empty `[dependencies]`; it must never import `engine` (a CI step runs `cargo tree -p reference` and fails if `engine` appears). It emits the same event sequence and derives seat views from the same tags.
5. **Tests.**
   - One test per numbered rule (rules 25, 26, 34, 37–41: assert illegal actions are rejected and forced robs happen).
   - **About 25 named plain-English transcripts**, written before coding and doubling as owner-review material: the 5-of-trumps bonus ending a game mid-hand; 41 reached only by the +6; a second bonus worth 10 while another team's first is 11, with the flag resetting in a new game; no trump played; the only trump being the A♥ or a black 2; a turned 5 or J robbed vs unrobbed; a turned A♥; a turned plain ace; the dealer holding the ace with an ordinary turned card (robs eighth); the ace holder leading trick 1; the A♥ holder not robbing when hearts are not trumps; an ace never burnable (all four aces, by trump suit); the dealer burning the Joker; turned Joker then turned ace (two burns, 11 undealt); turned Joker with the dealer holding the ace; turned Joker then an ordinary card with a non-dealer robbing (two burns); the four rule-17 forcing variants; both rule-18 branches; plain hearts led with the A♥ as a trump; the A♥ led when hearts are not trumps; off-suit never winning; black 2 beats black 10 and red 10 beats red 2; plain ace positions; jink and 41 in one hand counting as one game; two teams at 41 after trick 5 (trick winner wins); a game ending at trick 3 with another team's bonus pending (lost); scores of 45, 46, 47; dealer rotation across games; a five-game match; a non-default stake.
   - Property tests (proptest): 53 cards always accounted for; every trick has 8 cards in team order A,B,C,D,A,B,C,D; 5 cards per seat at the start of trick 1 after any robs; hidden cards ∈ {12, 13}; burns ∈ {0, 1, 2} (dealer ≤ 2, non-dealer ≤ 1) and never an ace; the ace of trumps robbed exactly once per hand when dealt; each team's bonus flag equals (`score mod 5 == 1`) under the house rule; no non-terminal score exceeds 40; a game lasts at most 7 hands and a match at most 5 games; exactly one game-end reason per game; money across the four teams sums to zero and equals +6/−2 stake units per team; the legal action set is never empty; every random game replays byte-identically from seed plus actions; the tracker's hard facts never exclude the true deal.
6. **`crates/difftest`.** Drives engine and reference with identical random legal actions from the same explicit deck and dealer (the reference must not share the engine's RNG), comparing legal-action sets, trick winners, scores, phases, event streams, seat views, hidden-card and burn counts, and match/money results at every step; run under every rule-switch setting.
   - `cargo test` runs a small count.
   - A `--release` binary takes a game count, reports rule-branch counts and asserts minimum coverage of rare branches (turned Joker ≈ 1 in 53 hands; Joker then turned ace ≈ 0.15%; Joker with the dealer holding the ace ≈ 0.18%; turned A♥ ≈ 1.9%; turned 5 of trumps ≈ 7.5%); also runs on the hand-made deals and on deals seeded to force each branch.
   - **Gate:** zero disagreements over 10,000,000 games. Run it as the manual sharded CI job (5 × 2M, seeds from (run, shard), resumable), on vast.ai, or overnight on the laptop, and record the result here.
7. **WASM.** `cargo build -p engine --target wasm32-unknown-unknown`, plus a determinism check: the same seeded random matches give an identical result hash on native and wasm. Node 22 is available in cloud sessions to run the wasm.
8. **CI green** on Linux and Windows (including the wasm build): Windows on PRs and pushes to main plus a nightly run; the 10M-game differential job on manual dispatch.
9. **Owner review.** The owner reads and approves the engine-generated plain-English transcripts of every hand-made case (the engine and the reference are both written from the same text, so the differential gate cannot catch a shared misreading).

## Gate to pass M0 (PLAN §13)
- Every numbered rule has passing tests.
- RNG golden vectors are identical on Windows, Linux and wasm32.
- Zero engine-vs-reference disagreements over 10 million random games, with the rare-branch coverage minimums met.
- WASM results equal native.
- The owner has approved the transcripts.

## Next after M0 (see the revised §13 table in PLAN.md)
M1 adds the table manager, the arena (14-way duplicate block, generalised SPRT, Plackett-Luce rating, the 0.000 canary), the **fixed-point money-chance type and the exact match-equity table v0**, and thin run-ops. M2 starts with a **solver measurement spike** before any pruning work.

## Log
- 2026-10-13: Repo initialised from the approved plan. Rust 1.99.0 pinned; Rust 1.99.1 is due 2026-10-15 (x86-64 FFI fix). Bump the pin when it's out.
- 2026-10-13: Plan reviewed (8 research sweeps, 5 lenses, 3 sceptics per change). Verdict: keep the recipe; re-scope the exact solver (cheap late only), make the match-equity table exact at M1, specify the learning pipeline precisely, add M3b. H3 tightened: no ace may ever be burned. Details in `docs/REVIEW-2026-10-13.md`.

## M0 gate result (2026-10-14 00:37 UTC)
- 10M-match differential gate: 5 shards × 2M matches on GitHub CI (run 38480695319), all PASS, zero engine/reference disagreements.
- All M0 gate items met **except the owner's reading and approval of the transcripts** (`docs/TRANSCRIPTS.md`, `crates/engine/tests/transcripts/`).
