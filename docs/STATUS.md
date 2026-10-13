# Status

**Current checkpoint: M0** (PLAN §13): the rules engine, independent reference implementation, deterministic RNG, CI.

## Done
- Research, design, tech stack and run-operations plan approved (`docs/PLAN.md`).
- Authoritative rules and house rules recorded (`docs/RULES.md`).
- Repo skeleton: pinned Rust toolchain, Cargo workspace, CI workflow, `.gitattributes`.

## M0 task list (do in order)
1. **`crates/engine`: deterministic RNG.**
   - SplitMix64 for seed expansion, plus xoshiro256** for generation.
   - An unbiased bounded integer function using `u64` rejection sampling, with no `usize` inside the RNG.
   - Our own Fisher–Yates shuffle.
   - Golden vectors, cross-checked against an independent Python implementation (`tools/rng_golden.py`, stdlib only), stored as test data.
2. **`crates/engine`: cards and rules.**
   - 53-card bitboards (`u64`).
   - Effective suit, trump order (red/black, hearts = 14 trumps), plain-suit orders.
   - Legal-move generation, covering plain leads, trump leads, the reneging test, and rule 18.
   - Trick winner.
3. **`crates/engine`: hand, game and match flow.**
   - Seeded deal (2 then 3 cards, starting left of dealer); turned card; undealt pack order.
   - Robbing per RULES H2–H4, with burn decisions as actions and forced takes done automatically.
   - Scoring: 5 per trick; best-trump bonus 11 first / 10 later per team per game; immediate bonus when the 5 of trumps is played; end-of-hand bonus; no bonus if no trump was played.
   - Game ends the instant a team reaches 41; jink; match = first to 2 games; next deal always moves left; money settlement (stake configurable).
   - An event stream and a test-only constructor for custom deals.
4. **`crates/reference`.** A naive, independent second implementation of the same rules (Vec-based, ordered lists, written straight from RULES.md).
5. **Tests.**
   - One test per numbered rule (rules 25, 26, 34, 37–41: assert illegal actions are rejected and forced robs happen).
   - Hand-made cases: turned Joker then a turned ace; dealer holding the ace of trumps after a turned Joker; hearts as trumps; higher trump forcing out top trumps; robbing at own first turn; burning the taken card; game ending mid-hand at 41 with the next deal moving left; a jink; money settlement.
   - Property tests (proptest): 53 cards always accounted for; scores consistent.
6. **`crates/difftest`.** Drives engine and reference with identical random legal actions, comparing legal-action sets, trick winners, scores, phases and match/money results at every step.
   - `cargo test` runs a small count.
   - A `--release` binary takes a game count.
   - **Gate:** zero disagreements over 10,000,000 games. Run it on vast.ai or overnight on the laptop, and record the result here.
7. **WASM.** `cargo build -p engine --target wasm32-unknown-unknown`, plus a determinism check: the same seeded random matches give an identical result hash on native and wasm. Node 22 is available in cloud sessions to run the wasm.
8. **CI green** on Linux and Windows (including the wasm build).

## Gate to pass M0 (PLAN §13)
- Every numbered rule has passing tests.
- RNG golden vectors are identical on Windows, Linux and wasm32.
- Zero engine-vs-reference disagreements over 10 million random games.
- WASM results equal native.

## Log
- 2026-10-13: Repo initialised from the approved plan. Rust 1.99.0 pinned; Rust 1.99.1 is due 2026-10-15 (x86-64 FFI fix). Bump the pin when it's out.
