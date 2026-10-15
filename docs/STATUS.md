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
- 2026-10-14: owner accepted the per-run destroy-only vast.ai key (launcher creates it, passes it as VAST_DESTROY_KEY, deletes it at run end). Residual risk: a host could destroy our other instances during a run; no spend or data access.
- 2026-10-14: early-trick yardstick measured at tricks 2–4 (docs/m3-ncurves/): loss falls as about N^-0.75; N for 0.01% is about 240 (t2), 180 (t3), 60 (t4); hardest: leader, near the money, robber, holder of the 5, 2+ voids. Rollout-PIMC (rules-v1) loses about 0.35% per decision at trick 2, about 13× exact, so exact PIMC is required early. Trick 1 (32 decisions to 80 deals) is running on two machines.
- 2026-10-14: **trick-1 speed and cost tools** (M3-SPEC §13.1 item 11):
  - `t1bench` runner chunk (`search bench-t1`), rebuilt 2026-10-14 after the configuration sweep left rented machines mostly idle: a 240-s fill window (one world per thread, positions spread evenly) and a 120-s production-schedule window, both at full CPU, stopped at once and FAILED if CPU drops below 80% or a thread sits idle for more than 10 s; reports nodes per second, a machine score for the real trick-1 mix, $ per 10^12 nodes and the golden set's projected cost (M3-SPEC §13.1 item 11). Manifest: `{"run_id": "t1bench-<machine>-1", "benchmark_seconds": 0, "chunk": {"t1bench": {}}}`.
  - Position-aware schedule for exact worlds (`worlds_at_once: "auto"`, now the default): each world gets threads and a table by its expected cost; values byte-identical (tested).
  - Stopping keeps finished work: SIGTERM, the launcher's `STOP` request or the deadline make the runner upload what it finished before the machine is destroyed; the launcher waits up to 5 minutes for that. The boot-script part needs a new image build.
  - Equity table v0 is fetched from R2 or rebuilt from the stored model in seconds (hash-checked) instead of re-estimated on every machine.
  - Measured on the 4-vCPU Xeon: one-thread trick-1 worlds take about 200 s at position 4 and 10–50 s at positions 6–8; one thread per world did 20–25% more worlds per hour than 2 or 4 threads per world.

## 2026-10-15 (Mauritius morning): runner-robustness review and rebuild
- Reviewed commit a41cf21 (merged as 46e555d, head of `m3-pimc`; nothing to fix, so no new code). Checked in a private worktree: `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace` pass; `python3 -m unittest test_launch` 27 tests pass.
- The agent's claims, each run myself: a hung rclone does not block heartbeats and every hung call is killed at its limit (`hung_network_never_blocks`); `kill -9` mid-run then resume values only the unsaved worlds and the worlds file is byte-identical, in format 1 (`resume_from`: 9 restored, 39 valued of 48) and format 2 (replacement: 9 restored, 39 valued), `resume_worlds` (release, ignored by default, 141 s); per-world resume `worlds_resume_per_world` passes; ETA within 15% once measured (`eta_within_15_percent_once_measured`, a synthetic 20x-cost queue, not yet a real rented run).
- Release binaries for 46e555d are in R2 (`bin/46e555dfd90b7c0724126fa1b374fb5d4183c365/`, v2 and v3). **The base image was stale** (last built at a91b87b, before `boot.sh` changed in 221b768); rebuilt from 46e555d: `ghcr.io/guillaumeds/41bots-base:20261015-46e555dfd90b` (image run 38703688651, success).
- Market re-ranked (`docs/vast/market-2026-10-15.json`): the 768-core EPYC 9965 ($2.40/h) and the 384-core EPYC 9654 ($4.27/h) are **gone**. Nothing rented; vast shows 0 instances.

## 2026-10-15: speed-test wave 1 (clean; about $0.60 in total)
- Runs `t1bench-9754-w1` (offer 56038183, EPYC 9754, 245 threads, $3.737/h) and `t1bench-9995wx-w1` (offer 55530776, Threadripper PRO 9995WX, 184 threads, 241 GB, $1.269/h). Both: every thread busy (100%), CPU 95–99%, no low-CPU stop, worlds finished in both windows byte-identical, DONE written, instance destroyed by the run (vast showed 0 instances afterwards). About 7 and 6.5 minutes of rental each. Results in R2 `runs/<run>/out/t1bench/`.
- **Cost per 10^12 solver nodes: 9995WX $0.287 (fill) / $0.634 (real schedule); 9754 $0.910 / $1.750.** The 9995WX is about 3 times cheaper. Per thread 6.7 vs 4.7 million nodes/s.
- Golden set (128 decisions x 200 worlds, 5.9e13 nodes) at fill speed: $16.9 / 13.3 h on the 9995WX, $53.5 / 14.3 h on the 9754. The real schedule runs at about half the fill rate on both (splitting a world over threads costs extra nodes; only 1 of 5–6 production worlds finished in the 2-minute window, so that rate is rough).
- Memory was not the limit: 83–105 GB used of 241 GB on the 9995WX; its fill tables were lowered 512 -> 448 MB per thread to fit.
- The owner's wave-2 offers 53756039 (9995WX, 503 GB, $2.81) and 56056919 (EPYC 9754, 512 threads, $9.08) were gone from the market by 08:40 UTC. Nothing rented since wave 1.

## 2026-10-15 (Mauritius afternoon): speed-test wave 2 (about $0.80) and the table-size sweep
- Wave 2 (runs `t1bench-9654p-w2`, `t1bench-9734-w2`, `t1bench-e5-w2`; 15-minute limit, $1.50 cap; vast showed 0 instances afterwards): all three kept every thread busy; the E5's production window was stopped by the 80% CPU check at 83% (its fill results are complete), as designed.

  | Machine | $/h | Threads | RAM | Fill $/10^12 nodes | Score $/10^12 | Per thread (M nodes/s) | Production / fill rate |
  |---|---|---|---|---|---|---|---|
  | Threadripper PRO 9995WX (wave 1) | 1.27 | 184 | 241 GB | 0.308 | 0.287 | 6.7 | 0.49 |
  | EPYC 9734 (Zen 4c dense) | 3.34 | 215 | 125 GB | 0.826 | 0.779 | 5.2 | 0.44 |
  | EPYC 9654P (Zen 4) | 2.94 | 184 | 251 GB | 0.843 | 0.799 | 5.3 | 0.53 |
  | EPYC 9754 (wave 1) | 3.74 | 245 | 483 GB | 0.944 | 0.910 | 4.7 | 0.54 |
  | Xeon E5-2695 v4 | 0.41 | 69 | 504 GB | 0.542 | 0.527 | 3.0 | stopped at 83% CPU |

- The speed model predicted the three new machines within 11% ($/10^12: 9654P 0.72 -> 0.80, 9734 0.78 -> 0.78, E5 0.58 -> 0.53), so offers of known CPU models can be ranked without testing each one; test new CPU models only.
- The 9734 ran at only 224 MB tables per thread (0.6 GB of RAM per thread) with no loss of nodes per second, but nodes per second is blind to a small table making the solver search positions again: that is what the new sweep measures.
- **Table-size sweep added to the bench** (M3-SPEC §13.1 item 11): `sweep_mb`, `sweep_s`; `analyse_sweep` in `search::t1bench`; fmt, clippy -D warnings and `cargo test --workspace` pass; local end-to-end run on 4 threads works mechanically (no world finishes in 12 s there, so nothing paired).

## 2026-10-15 (Mauritius afternoon): speed-test wave 3 (about $3) and the first table-size sweeps
Five machines (15–30 minute limits, dollar caps, checked every 30 s; all destroyed, vast showed 0 instances afterwards). Binaries: commit 56851ce (adds the sweep). Results in R2 `runs/t1bench-*-w3/out/t1bench/`.

| Machine | Threads, RAM, rate | Fill $/10^12 nodes | Score $/10^12 | Total speed (billion nodes/s) | Golden set at fill speed |
|---|---|---|---|---|---|
| AMD Threadripper PRO 9995WX workstation chip (repeat of wave 1) | 184, 241 GB, $1.27/h | 0.304 | **0.283** (wave 1: 0.287) | 1.25 | 13.1 h, $17 |
| Intel Xeon E5-2695 v4 (2016 server) | 69, 504 GB, $0.41/h | 0.533 | 0.519 | 0.21 | 76 h, $31 |
| Intel Xeon Gold 5220R (2020 server) | 92, 45 GB, $0.70/h | 0.594 | 0.571 | 0.34 | 48 h, $34 |
| AMD EPYC 9554 server chip (Zen 4) | 245, 504 GB, $6.94/h | 1.202 | 1.134 | 1.70 | 9.6 h, $67 |
| AMD pre-release Zen 5 "Turin" engineering sample | 491, 1.1 TB, $13.66/h | 1.501 | 1.436 | 2.64 | 6.2 h, $85 |

- **Repeatability:** the 9995WX repeat gave 0.283 against 0.287: results repeat within 2%. It is 3 to 5 times cheaper per node than everything else tested, and the cost per node rises with machine size (the fastest machine costs 5 times the cheapest per node).
- **Real schedule vs full-machine window:** production runs at 0.34 (Xeon Gold) to 0.40 (EPYC 9554), 0.36 (Turin sample) of the fill rate on the biggest machines, against 0.49–0.54 at 184–245 threads, so the golden-set figures above are optimistic by 2 to 3 times. Only 1 of 6 production worlds finishes in the 2-minute window, so these ratios are rough.
- **Minimum RAM:** the 45 GB machine (0.5 GB per thread) ran the real schedule at 95% CPU: the biggest world got a 14 GB table. The floor is not a hard stop.
- **Table-size sweep (9995WX, same worlds, against 448 MB per thread):** 256 MB: 1.07x the nodes, 1.06x the time; 128 MB: 1.21x, 1.16x; 64 MB: 1.88x, 1.54x; 32 MB: 3.2x, 2.3x. Nodes per second go UP as tables shrink (1.13 -> 1.63 billion/s), which is why nodes per second alone misleads. All values byte-identical.
- **Larger tables (old Xeon, 504 GB, against 512 MB):** 1024 MB and 2048 MB: 0.99x the nodes, 1.00-1.03x the time. More RAM does not help the short worlds; tables of 256-512 MB per thread are enough there.
- Corrections for small tables: the 45 GB machine ran 212 MB tables (about +8% cost, so about $0.62 per 10^12) and the 125 GB EPYC 9734 of wave 2 ran 224 MB (about +7%, about $0.83).
- **Open question (79% of the work is positions 1-2):** the sweep covers worlds short enough to finish in 90-150 s (all eight positions appear among them, but not the long ones). Production gives position-1 worlds 14-32 GB tables, and what smaller tables cost there is unmeasured. Next test: a few position-1 worlds at several large table sizes on a big-RAM machine.
- The sweep's summary line says "mostly positions 5-8"; wording to fix at the next code change.
- Models: the speed model predicted the Zen 4 server and old Intel machines within about 11%; it was 40% too pessimistic for the Xeon Gold 5220R (no entry in the speed file) and about 20% too optimistic for the Turin sample.

## 2026-10-15: no 9995WX available; vast watcher built; trick-1 redo scoped
- The AMD Threadripper PRO 9995WX workstation chip (184 threads, 251 GB, $1.27/h) was gone again at 09:59 UTC; the dry run found 0 offers, so nothing was rented. No Threadripper with 128 or more threads is on the market.
- **Vast machine watcher** (`python/tools/vast_watch/`): `python3 python/tools/vast_watch/watch.py` then open http://127.0.0.1:8765. A page with an on/off switch for browser desktop notifications, filters (CPU name, threads, RAM, price, whole machine, verified, reliable), presets for the 9995WX/7995WX class, a "NEW" badge and a "can our launcher rent it" column; it checks every 15/30/60 s while the page is open. Machines present when the page opens are the baseline and do not notify. Tested in a real browser with a fake offer feed (new match notifies; non-match, switch off and filter changes do not; the switch survives a reload) and against live vast.ai data.
- **Trick-1 agreement data: nothing was saved from the earlier runs.** R2 holds no worlds output for `m3-agree-t1-a5` (42 of 1,280 worlds done, 0 of 16 decisions) or `m3-agree-t1-b` (5 of 16 decisions and 433 of 1,280 worlds done, killed after 168 minutes), so all 32 decisions x 80 worlds (2,560 world solves) are still to do, not 21 decisions. Manifest drafted (format 1, `search_measure`, `count` 32, `to` 80, automatic position-aware schedule and table sizes): run id `m3-agree-t1-c1`. At the benchmark rates the whole set is about 5.9e12 nodes, roughly 2 to 4 hours and $3 to $5 on a 9995WX-class machine, about 2 to 3 times that on other machines; the real cost will show in the run's ETA after 10% is done.

## 2026-10-15 (Mauritius evening): trick-1 runs killed by the old cleanup; long runs fixed
- **What went wrong:** runs `m3-agree-t1-b2`, `-a7` and `-b3` were all destroyed by `vast-cleanup.yml` on main, which destroyed every 41bots- machine older than 2 hours (it ran at 12:37, 14:28 and 15:28 UTC, exactly when they died). The work was not lost: each run saved its finished worlds (`runs/<run>/partial/out/measure/exact-t1-n80.jsonl.ckpt`): B 278 of 1,280 worlds (b3, including b2's 83), A 58 of 1,280 (a7). Spent: b2 $2.84, a7 $3.21, b3 $3.25.
- **Long runs now (owner-approved):**
  - `launch.py --handover-min` (vast-run passes 330): before GitHub's 6-hour job limit the launcher stops watching but leaves the machine working, writes `runs/<run>/watch.json` and `watch/<run>.json` (instance, price, dollar cap, deadline) and clears the workflow's safety-net files. Every machine gets a `watch.json`. Reports say whether the machine destroyed itself (`self_destroyed`); outcome `HANDED_OVER`.
  - `python/launcher/cleanup.py` + `vast-cleanup.yml` (on main, every 15 minutes) replace the 2-hour rule. A machine is destroyed only when its run wrote DONE/FAILED 10 minutes ago, went 20 minutes without a heartbeat, never heartbeated in 45 minutes, is exited/offline, is 30 minutes past its deadline, or passed its dollar cap (STOP first, destroyed 15 minutes later). For handed-over runs it writes `runs/<run>/closed.json` (outcome, whether the machine destroyed itself). Tests: 16 cleanup + 2 new launcher tests (47 in all) pass; a dry run against the real R2 data read b3's 303 heartbeats correctly.
  - The `--offer-id` mode rents a named offer with no search (the vast daily search quota of 20,000 rows was used up by polling; it resets at 00:00 UTC).
- **Restart attempt 20:16 UTC (00:16 Mauritius) failed, nothing rented:** both machines the owner approved were taken before we could rent them: the AMD EPYC 7B13 server, UK (245 usable threads, 258 GB, $1.058/h, offer 54173274) and the AMD EPYC 7663 server, Norway (215 usable threads, 258 GB, $1.008/h, offer 54278014). The run names `m3-agree-t1-b4` and `-a8` are used up (their manifests are in R2). The market search is blocked until 00:00 UTC (daily quota). A scheduled task at 00:10 UTC does one market check and, if verified machines at about $0.37 per 10^12 nodes or less are available, starts B as `m3-agree-t1-b5` (resumes b3) and A as `m3-agree-t1-a9` (resumes a7), each with a $20 cap and a 24-hour deadline.
- Manual dry run of the new cleanup on GitHub (main, run 38800533132): success.
