# Status

**Current checkpoint: M0** (PLAN §13, as revised on 2026-10-13): the rules engine, independent reference implementation, deterministic RNG, CI.

## Done
- Research, design, tech stack and run-operations plan approved (`docs/PLAN.md`).
- Authoritative rules and house rules recorded (`docs/RULES.md`).
- Repo skeleton: pinned Rust toolchain, Cargo workspace, CI workflow, `.gitattributes`.
- **2026-10-13: plan reviewed and amended** (`docs/REVIEW-2026-10-13.md`; amendments at the top of `docs/PLAN.md`). Owner decisions: no ace may ever be burned (H3; withdrawn 2026-10-14); think time 8 s / 5 s on tricks 1–2; M3b playable demo; equal tournament mix weights; `bonus_later` switch.

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
   - Legal burns: **any card except the ace of trumps** (rules 21, 24, H3; the 2026-10-13 no-ace rule is withdrawn); plain aces, the taken card and the Joker may be burned.
3. **`crates/engine`: hand, game and match flow.**
   - Frozen deal convention: deck positions 0–15 two at a time to seats dealer+1 … dealer+8, positions 16–39 three at a time, position 40 turned, 41 the second turned card after a Joker, the rest the undealt pack in order; first dealer = `bounded(8)` from the seeded RNG before the first shuffle; every hand reshuffles all 53.
   - Robbing per RULES H2–H4, with burn decisions as actions and forced takes done automatically; a non-ace-holder is never offered a burn.
   - Scoring: 5 per trick; best-trump bonus 11 first / 10 later per team per game (fixed: +6 first, +5 later; the flat-11 switch was removed on 2026-10-14); immediate bonus when the 5 of trumps is played; end-of-hand bonus; no bonus if no trump was played; per-trick check order from the RULES notes.
   - Game ends the instant a team reaches 41 or more; jink; match = first to 2 games (at most five); next deal always moves left; money settlement (stake configurable). Record a game-end reason (41-by-trick, 41-by-immediate-bonus, 41-by-end-bonus, jink).
   - An event stream with **visibility tags** (public / private to seat s) on every event; each seat's view is derived from the tags. Events carry the incremental hard facts (who has proved they lack the ace of trumps; burn locations and counts; hidden-card count 12 or 13).
   - A `rules_version` field in every record; a test-only constructor for custom deals.
4. **`crates/reference`.** A naive, independent second implementation of the same rules (Vec-based, ordered lists, written straight from RULES.md). Empty `[dependencies]`; it must never import `engine` (a CI step runs `cargo tree -p reference` and fails if `engine` appears). It emits the same event sequence and derives seat views from the same tags.
5. **Tests.**
   - One test per numbered rule (rules 25, 26, 34, 37–41: assert illegal actions are rejected and forced robs happen).
   - **About 25 named plain-English transcripts**, written before coding and doubling as owner-review material: the 5-of-trumps bonus ending a game mid-hand; 41 reached only by the +6; a second bonus worth 10 while another team's first is 11, with the flag resetting in a new game; no trump played; the only trump being the A♥ or a black 2; a turned 5 or J robbed vs unrobbed; a turned A♥; a turned plain ace; the dealer holding the ace with an ordinary turned card (robs eighth); the ace holder leading trick 1; the A♥ holder not robbing when hearts are not trumps; an ace never burnable (all four aces, by trump suit); the dealer burning the Joker; turned Joker then turned ace (two burns, 11 undealt); turned Joker with the dealer holding the ace; turned Joker then an ordinary card with a non-dealer robbing (two burns); the four rule-17 forcing variants; both rule-18 branches; plain hearts led with the A♥ as a trump; the A♥ led when hearts are not trumps; off-suit never winning; black 2 beats black 10 and red 10 beats red 2; plain ace positions; jink and 41 in one hand counting as one game; two teams at 41 after trick 5 (trick winner wins); a game ending at trick 3 with another team's bonus pending (lost); scores of 45, 46, 47; dealer rotation across games; a five-game match; a non-default stake.
   - Property tests (proptest): 53 cards always accounted for; every trick has 8 cards in team order A,B,C,D,A,B,C,D; 5 cards per seat at the start of trick 1 after any robs; hidden cards ∈ {12, 13}; burns ∈ {0, 1, 2} (dealer ≤ 2, non-dealer ≤ 1); the ace of trumps robbed exactly once per hand when dealt; each team's bonus flag equals (`score mod 5 == 1`) under the house rule; no non-terminal score exceeds 40; a game lasts at most 7 hands and a match at most 5 games; exactly one game-end reason per game; money across the four teams sums to zero and equals +6/−2 stake units per team; the legal action set is never empty; every random game replays byte-identically from seed plus actions; the tracker's hard facts never exclude the true deal.
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

## M1 progress (2026-10-13 night)
- Committed on `m1-tables`: `crates/tablemgr` (tables, records, protocol, leakage tests), `crates/bots` (random, rules-v1: 56% money vs random), engine block decks + serde feature, `crates/equity` (fixed-point MoneyChance, exact 810k-state match-equity sweep, ~7 s on 4 threads).
- Match-equity table v0: built from 1M rules-v1 self-play hands (`data/equity/model-rules-v1-1M-seed1.json`); all checks pass (sums, rotation, monotone, flag rule, Monte Carlo |z| ≤ 2.60). The 26 MB `.npy` is rebuilt from the model by `equity build`; its sha256 is in `data/equity/`.
- **Must fix before untrusted seats (M3b):** (1) the seat seed can be inverted to the match seed — derive it one-way or from a secret; (2) the `end` message carries the full record, revealing the other decks of a duplicate block — send records only after the block, or strip seeds/burns.
- Next: arena (spec §5), run-ops (spec §7), first vast.ai smoke run, σ_b.
## M0 gate result (2026-10-14 00:37 UTC)
- 10M-match differential gate: 5 shards × 2M matches on GitHub CI (run 38480695319), all PASS, zero engine/reference disagreements.
- All M0 gate items met **except the owner's reading and approval of the transcripts** (`docs/TRANSCRIPTS.md`, `crates/engine/tests/transcripts/`).
- 2026-10-14: the owner read and approved the transcripts. **M0 gate fully met.**
- 2026-10-14: **Owner rule changes** (`RULES_VERSION` 1 → 2; version-1 records are refused):
  - **H3: the 2026-10-13 "no ace may be burned" rule is withdrawn.** Any card may be burned, aces included (the ace of trumps too), by the dealer and the robber alike. If the dealer burns, after a turned Joker, the card that becomes the ace of trumps, nobody robs and the ace is announced out of play after trick 1 (`AceUndealt` now means "undealt or burned").
  - **H4: after a turned Joker and an ordinary card, a dealer holding the ace of trumps robs at their own first turn (as the eighth card), not at once.** Only a turned ace (also one after a Joker) is robbed at once. The dealer is no longer known to lack the ace straight after the Joker sequence; they prove it only by playing their first card.
  - **The `bonus_later` switch is removed:** bonuses are always +6 first, +5 later per partnership per game; the flat-11 variant was rejected. The `Rules` struct now holds only the stake (records, protocol and model JSON lose the two bonus fields).
  - Wording fixes in RULES rule 17 (every forcing lead also forces all lower trumps), transcripts 19, 20, 22, 24, 26, 32 and `docs/TRANSCRIPTS.md`; transcript 13 now shows an ace being burned; transcript 31 (flat rule) deleted.
  - Checks: fmt, clippy, all tests, WASM check pass; difftest 100,000 matches seed 1: zero disagreements, PASS (16.4% of hands burn an ace; the "Joker, dealer holds the new ace" branch now robs in turn).
  - Equity v0 rebuilt from 1M rules-v1 hands (seed 1): all checks pass (Monte Carlo largest |z| 2.52); new table sha256 `00bdc49f…68da` (was `bc7e7293…5ead9a`).
  - Arena canary: 50 blocks, every block exactly 0. **σ_b (rules-v1 vs random, 200 Block14 blocks): 0.1538 money chance (1.231 stakes), mean +0.2774** (before the rule changes: 0.1501, mean +0.2642).
- 2026-10-14 (later): **Owner chose "back to the original H3"** (supersedes the H3 line of the entry above): any card may be burned **except the ace of trumps** (rules 21 and 24), so plain aces, the A♥ when hearts are not trumps, the card just taken and the Joker may go. `RULES_VERSION` stays 2 (no version-2 records were ever published).
  - **Open question for the owner:** after a turned Joker the dealer burns before trumps are known. If that burned card turns out to be the ace of the suit turned next, is that allowed? The engine forbids nothing retroactively: nobody robs, the turned card stays face up, and after trick 1 the ace is announced out of play (`AceUndealt` = "undealt or in the dealer's Joker burn").
  - Checks: fmt, clippy, all tests, WASM check pass; difftest 100,000 matches seed 1: zero disagreements, PASS (ace burned in 3.9% of hands).
  - Equity v0 rebuilt (1M rules-v1 hands, seed 1): all checks pass (Monte Carlo largest |z| 2.52); table sha256 `64740dab…6808`.
  - Arena canary: every block exactly 0. **σ_b (rules-v1 vs random, 200 blocks): 0.1463 money chance (1.170 stakes), mean +0.2606.**

## M1 gate result (2026-10-14)
- First vast.ai run end to end (GitHub run 38512961545): cheapest ≥16-core on-demand box, 20M-hand equity sweep, DONE uploaded, instance self-destroyed and verified gone; CPU 98.6% (benchmark), 77% (chunk start), 93% (last reading). Cost < $0.01.
- Table v0 checks pass; σ_b = 0.146 money chance (1.17 stakes) per 14-way block, rules-v1 vs random; canary exactly 0.
- Leakage and replay tests pass. **M1 gate met.** Open owner question: see the early Joker-burn ace question above.
- Next: M2 (exact solver) — measurement spike first (PLAN §13).

## M2 progress (2026-10-14)
- **Measurement spike done** (`crates/solver`, binary `solver-spike`, results in `docs/M2-SPIKE.md`): naive exact max^n in money chance, optional plain transposition table; matches an engine-driven brute force on 1-, 2-, 3-card endgames; table on = off.
- Naive costs on the 4-vCPU Xeon: trick 3 median 18 ms (p90 55 ms) with the table; trick 2 median 9 s, 7% over 60 s, 365 MB; trick 1 never finished in 60 s. 82–85% of decisions have tied best cards; reversing the tie order never changed a root value at tricks 3–4.
- Next: equivalent cards, Sturtevant pruning (on = off under the tie order), move ordering; repeat the spike on a second CPU type (vast.ai EPYC) for the M2 gate.
- **2026-10-14 (later): optimised exact solver** (`solver::solve_fast`, details and tables in `docs/M2-SPIKE.md`). Four switchable speed-ups: bounded transposition table keyed per A2, equivalent cards with the 41 exceptions, move ordering, Sturtevant shallow pruning (proof in `fast.rs`; deep/speculative pruning left out, no proof for 4 teams with our tie rule).
  - **Gate: identical answers to the naive solver** (root value, every root card's value, chosen card, both tie orders, all 16 switch combinations): 0 differences over 150,000 worlds each at tricks 4 and 5, 3,000 at trick 3 (three leaf tables), 64 at trick 2 (table-on combinations). Four-layer validation in `crates/solver/tests/`: engine parity (found and fixed a `from_match` bug: a 5 of trumps already in the current trick was marked as scored), brute force on 1–3-card endgames, on/off identity, best response at the root. Forced-denial check (test-only) added and tested.
  - Costs (4-vCPU Xeon): trick 3 median 8.8 ms (naive 17.6), p90 30 ms (55); trick 2 median 3.1 s (naive 9.2 s), all 100 worlds solved, worst 29 s (naive: 7 over 60 s); trick 1 still unsolved in 60 s. Equivalent cards give almost all of it (3.3× fewer nodes at trick 3, 5× at trick 2); shallow pruning fires at ~1 node in 10,000; move ordering costs 15% and saves nothing, so it is off by default.
  - Next for the M2 gate: repeat the cost tables on a second CPU type (vast.ai EPYC).

## M2 gate result (2026-10-14)
- Optimised solver identical to the naive solver (0 mismatches on the Xeon and on an AMD EPYC 7502); pruning on/off identical; brute force on 1–3-card endgames passes; costs published for both CPU types (`docs/M2-SPIKE.md`). **M2 gate met.**
- Follow-up in progress: a multi-core single-world solve to measure the true trick-1 duration on a high-core machine (owner request).
- Next: M3 (card tracker, suit-length-matrix world generator, PIMC teacher: exact from trick 3, rollout-PIMC with the rules bot at tricks 1–2; pre-registered PIMC-suitability measurement; full runner).

- 2026-10-14: owner answered the Joker-burn question (the dealer's burn after a turned Joker may turn out to be the ace of trumps: legal, a poor play; RULES H3). No longer open.
- M3 started: spec `docs/M3-SPEC.md` on branch `m3-pimc`.
- 2026-10-14: owner approved the M3 gate bar (≥2 points more match wins, test for 6), the 25% near-clinch teacher stratum, and ~€5 of M3 rented compute.

## M3 progress: card tracker and world generator (2026-10-14)
- **Engine (M3-SPEC §2):** `PublicState.burn_order` lists every burn of the hand with its kind (the dealer's burn after a turned Joker, the dealer's at-once rob burn, a rob at the robber's turn); the reference derives it too and the difftest compares it (100,000 matches seed 11: zero disagreements, coverage minimums met). Match records are format 2: a match started mid-way (`Match::from_hand_start`) records its start state and deck and replays; format-1 records still read.
- **`crates/belief` (§3–4):** the card tracker [what each seat knows for certain about where every hidden card can be] and the world generator [complete guesses of the hidden cards consistent with those facts].
  - Facts F1–F12 from the engine's own functions: voids and trump limits from `legal_plays`, ace facts from `legal_burns` and the `ProvedNoAce`/`AceUndealt` events, the Joker-burn and dealer-robs-eighth rules of 2026-10-14, 12 or 13 hidden cards.
  - The exact number of consistent worlds (whole numbers, `u128`), a max-flow feasibility check, exactly uniform sampling and full listing when small. Counts: about 10^36 at the start of trick 1, 10^14 at trick 4, 10^8 at trick 5 (golden values stored).
  - Checks: the true deal is never excluded (20,000 `rules-v1`/random mixed matches, every seat at every decision: pass; 2,000 random matches with world counts and listings plus 5,000 forced rare-branch hands: pass; every transcript deal); on small late positions the tracker's worlds are **exactly** the deals the engine rebuilds into the same view; sampled worlds rebuild into the seat's real view at 2,000 decisions; count = listing = brute force; chi-square of the sampler against uniform and against a rejection oracle (40 positions up to 5,000 worlds, 200 draws per world: pass).
  - Cost (`belief-bench`, 2,000 positions per trick, 4-vCPU Xeon): tracker + count + 100 samples take a median 0.1–0.5 ms and at most ~10 ms per decision at every trick; trick-1 counting is no bottleneck.
  - Full-size sweeps are `#[ignore]` tests (`cargo test --release -p belief -p search -- --ignored`); the 20,000-match belief sweep with counts and listings takes ~20 min on 4 cores, so it belongs on vast.ai or CI.
- **`crates/search`:** the belief sampler is now the default world source (`BeliefSource`); it reports each decision's exact world count, so the agreement output's world-count bands (§13.1 item 8) and the teacher records' `log2_world_count_q8` are filled. `--source rejection` keeps the old engine-based sampler as an oracle.
