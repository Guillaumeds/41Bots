# Status history (2026-10-13 to 2026-10-20)

Moved out of `docs/STATUS.md` on 2026-10-21 so that STATUS holds only the current state. Nothing was edited.

## Earlier: state and next steps (2026-10-20, written for the next session or model)

Read this section, then `docs/m3-hybrid/learned-leaves.md` (G v2 and O_2 G v2), `docs/STUDENT.md`
§11-14 and `docs/m3-hybrid/silver.md`. The dated sections further down are history. The working plan
log of 2026-10-16 to 2026-10-20 is `docs/plan-log-2026-10.md`. Branch: everything is on `m3-hybrid`.

### The goal and the chain (owner)
One objective: the chance of the money. Target: about 0.01% decision loss even at the hardest trick-1
situations, with every layer below it. The chain, top down:
1. **Trick 3-5 decisions:** exact search (cheap). Done.
2. **O_3** (trick-3 network) + **G v2** (a learned predictor of O_3's error picks the trick-3 positions to
   solve exactly): trick-2 decisions at 0.0008% (seat 1 0.0022%) on the exact golden set. Done, adopted.
3. **O_2** (trick-2 network, used by trick-1 search): provisional only. **Now: O_2 G v2** (the same
   method one trick earlier, below).
4. **Teachers** (slow, offline): trick 2 = PIMC over 64+ deals with H_3 + G v2; trick 1 = PIMC over 64
   deals (128+ on close calls) with H_2 + O_2 (+ O_2 G v2).
5. **Students** (fast networks for the app): learn from the teachers; in the app they shortlist their
   top 3 cards and a small search picks among them.
The owner wants the teachers and their deal counts decided last; first O_2 G v2 and its score.

### Stored networks, models and data (R2 `inputs/sha256/<hex>`)
| what | key | notes |
|---|---|---|
| O_3 adopted (2048-1024-256) | f5766aa98a1d769e31122c94cc4da9790c84508fab5c432d0c0358fe3124faad | trained on 40M trick-3 positions |
| O_3 leaf-risk model (G v2) | 20c41678c64681855003e5b8660e0a2e1fb9e8d552f7d898dd956055c767a9d5 | `data/leafrisk/leafrisk-o3-v1.json`; threshold t=277338260 adopted |
| O_2 provisional (2048-1024-256) | 505a0c7c65147ffa9565e06059f47664ea9f1bfdf0c5ea2f8d392b0e6e1337b0 | 1.5M exact labels; position error SD 3.6% (O_3: 1.85%) |
| trick-2 golden set: decisions / worlds | 635e9aef196c816380f3f15c05ede90b38b875fa4817061a57e6d39550065ab0 / 6ddc59b191e1fd534f260f9dfe103f742d6026768ca74acdbbb4fd430eb8265e | 2,000 decisions x 320 exact worlds, seed 1 |
| exact trick-2 labels, random 1.2M (s511) | shard fb51c217b5406e4b4ba1454f14b8884bced45cff51a0e29bcca7f4440f6f0931, sidecar bc4165ed9408f83ff342ab6fa157992c5910654513e417a9271a700fbbb77117 | O_2's training data |
| exact trick-2 labels, rules-v1 0.3M (s512) | shard 2160fefc6fcfd07822292f330f140a0618d25e0b89fb4ff8ed13d62e4559f1bd, sidecar d052d1a172fcf1b09d6e1cd4e95e1a2814b60e197864e1d5b18e7ab692f265f2 | |
| exact trick-2 validation 50k (s601) | shard 38de48356209cbec0f9ce41ceba28e6d5df52428abc0c866d443f9eb57964327, sidecar b39dbd6bf8f00a9b71fa800ccdaf9dd7de0ef25bc5dd3fa7b8b9e718fe895df8 | |
| exact trick-2 validation 20k (s602) | shard 855ba467ee6bd7521a442cf668980af8cefacf00701939237895d7c749dfdeea, sidecar 6e346cbf05b9f2cc0d4368e224e4dd81a11a7c2aaab1fd5af0c52a8ad177c86d | made 2026-10-20 |
| exact trick-2 validation 5k (s603) | shard 46ba92fcdf8e… (see `runs/o2-gv2-labels-3/partial/.../shards.json`) | made 2026-10-20 |
| trick-2 student (512-256, 1M lessons, plain H_3 teacher) | 954223e2f56ecade906ba28ad2ea07ff70fee70b841e0743d4c911217afb6ce6 | see below |
| equity table v0 | sha256 64740dab7845f0502c93ce195d721e1ffb8aa6ab141739ef43f754690de16808 | rebuilt by every run from `table: {hands: 1000000, seed: 1}` |

### Measured today
- **Trick-2 student against the exact golden set** (`search student-check --golden-decisions …
  --golden-worlds … --split-half --top-k 2,3`, free on the VM): alone 0.073% (95% upper 0.088%), seat 1
  0.18%; **top-3 shortlist with a perfect final pick 0.0073% (upper 0.013%)**, seat 1 0.0148%
  (upper 0.024%). The student as a shortlister is already near 0.01%; its own fit, not its teacher,
  is the limit.
- **Rented speeds (AMD EPYC 7B12, all threads busy):** exact trick-2 label 11.3 thread-s, G v2 trick-2
  label 9.3 thread-s, harvested exact trick-2 row (H_2-driven) about 2 thread-s on the VM. Rented threads
  are 3-4.6x slower than the build VM: price from rented measurements only.

### Running now
- `o2-leafrisk-1` (O_2's predictor: 500k harvested exact trick-2 rows, then train and calibrate;
  cap $2.5; started 03:52 UTC). When DONE: fetch `runs/o2-leafrisk-1/out/leafrisk/` (model.json,
  calibration.json), store the model (`python/launcher/launch.py put-input`), read the risk table
  (how concentrated O_2's error is, error left at each share solved).

### Next steps, in order (each costs money: check the vast balance with the owner first)
1. **O_2 G v2 trick-1 test** (`learned-leaves.md`, "Next: the trick-1 decision test"): the silver set
   (`docs/vast/o2-silver.manifest.json`, 400 x 80, about $3-5; never launched yet) plus checks of
   H_2 with `--exact-leaves learned:<O_2 model>:t=<T> --defer-expert exact` at two thresholds, paired
   against plain H_2. Pick the cheapest threshold that passes 0.01% at every seat against the silver.
   Two rented runs, no new code: (a) the silver set with `search_measure` (`o2-silver.manifest.json`;
   it stores the decisions and the silver worlds at `inputs/sha256/`, listed in its `result.json`
   `shared.reuse_params`); (b) a `hybrid_checks` chunk (`crates/runner/src/checks.rs`, model
   `docs/vast/o3-learnedleaf.manifest.json`) with `trick: 1`, `cut: 2`, `decisions_input` /
   `worlds_input` = the stored silver files, `reference: {cut: 3, net_input: "o3"}`, and checks
   `none` (net o2), `learned` (net o2, `exact_leaves: "learned:<o2 model input>:t=<T1>"`,
   `defer_expert: "exact"`), a second threshold, plus `paired` comparisons against `none`. Add the new
   manifest to the resolve test before launching.
2. **Then decide O_2's own improvement:** more exact labels (11.3 thread-s each rented, about $3.8 per
   million) and retrain, using the stored 1.5M + new; a label-type test was designed
   (`docs/vast/o2-labeltest.manifest.json`: plain-hybrid vs exact at 600k, same exact validation) but
   stopped before a result.
3. **Teachers, then students** (`STUDENT.md`): trick-2 teacher with G v2 (`search teacher --trick 2
   --exact-leaves learned:… `, runner `student_teacher` param `exact_leaves`; about 99 thread-s a
   decision at 64 deals, so 1M lessons is about $34: relabel only what the student needs); trick-1
   teacher with the final O_2 (+ deferral if the test says so), measured about 70 thread-s a decision
   at 64 deals plain; trick-1 test set and student.
4. **App:** WASM module with the students, phone timing, duplicate tournament against today's bots.

### How to run things (GitHub Actions, no vast key on the VM)
- **Launch:** workflow `vast-run.yml` on `m3-hybrid` with inputs: `manifest` (the JSON inline, one
  line), `min_cores`, `min_ram_gb`, `disk_gb`, `cpu_only` ("true" for CPU jobs), `gpus`, `min_gpu_ram_gb`,
  `max_dollars` (the cap; the launcher stops and destroys at it), `any_host` ("true"), optionally
  `offer_id` (an owner-found offer: the `Type #` number) or `cpu_match` (e.g. "7B12"), `low_cpu_pct`
  ("1" for jobs with GPU-training or other single-thread phases, else the idle watchdog kills them),
  `bin_sha` (a commit with released binaries: **6c7acf50709117d224dda5363d516f9b367cc399** is the
  latest; dispatch `release-binaries.yml` on `m3-hybrid` for a newer commit and wait for
  `bin/<sha>/manifest.json` in R2), `image` **ghcr.io/guillaumeds/41bots-base:20261017-bc692e446939**,
  `dry_run` **"false"** (the default is true!). A Blackwell GPU (RTX 50xx) needs `"blackwell": true` in
  the manifest's `machine`.
- **Watch:** `python/launcher/launch.py` helpers (`latest_heartbeat(run)`, `r2_list`, `r2_get`) with
  `R2_S3_ENDPOINT=$R2_ENDPOINT`; heartbeats every 30 s carry the step, rows done and dollars.
- **Stop:** cancel the workflow run (it asks the run to stop, waits up to 5 minutes for uploads, then
  destroys the machine) or `vast-stop.yml` (run_id, instance_id); `vast-status.yml` lists every
  instance on the account. Confirm nothing is left billing.
- **From a laptop (no R2 or vast keys there):** everything goes through GitHub Actions with the `gh` CLI:
  `gh workflow run vast-run.yml --ref m3-hybrid -f manifest="$(jq -c . docs/vast/NAME.manifest.json)"
  -f min_cores=190 -f min_ram_gb=64 -f disk_gb=60 -f cpu_only=true -f max_dollars=3 -f any_host=true
  -f bin_sha=6c7acf50709117d224dda5363d516f9b367cc399 -f image=ghcr.io/guillaumeds/41bots-base:20261017-bc692e446939
  -f dry_run=false` (GPU jobs: `-f cpu_only=false -f gpus=1 -f min_gpu_ram_gb=8 -f low_cpu_pct=1`);
  `gh run list --workflow vast-run.yml`, `gh run view <id> --log` (the launcher's log shows the machine,
  price and progress lines), `gh run cancel <id>` (stop and destroy); `gh workflow run vast-status.yml`
  (every instance on the account); `gh workflow run r2-fetch.yml -f run_id=<run>` then
  `gh run download <id>` for a run's files (results bundles `out/*.tar.zst`, unpack with Python 3.14
  `compression.zstd`); `gh workflow run release-binaries.yml --ref m3-hybrid` for new binaries.
- **With keys on the laptop (`.env`, git-ignored; names in `.env.example`):** install the `aws` CLI
  and `vastai` (pip), load the file (`set -a; source .env; set +a`), then from `python/launcher`:
  `uv run python -c "import launch; print(launch.latest_heartbeat('RUN'))"`, `launch.r2_list('runs/RUN/')`,
  `launch.r2_get(key)`, `python launch.py put-input FILE` (store a model or shard content-addressed),
  `vastai show instances`. Prefer launching through the `vast-run.yml` workflow even then: its
  launcher watches and enforces the dollar cap for up to 6 hours without the laptop staying awake (a
  run outliving it is finished by its own self-destroy and the hourly `vast-cleanup`).
- **Test a manifest before launching:** add it to `v2::tests::manifests_resolve_into_settings_their_chunks_accept`
  (`crates/runner/src/v2.rs`; kinds hybrid_rung, student_teacher, hybrid_checks, leafrisk) and run
  `cargo test --release -p runner manifests_resolve`.
- **Pitfalls met:** `hybrid_rung` with `check: false` refuses `tt_mb`; `label_inputs` cannot be combined
  with `label_parts` above 1; vast offer numbers change after a machine is released (use `cpu_match`);
  a label part is stored only when complete (size parts to about an hour); the vast balance hitting zero
  stops every machine at once.

### 2026-10-20 runs that were stopped (no results beyond the stored validation shards)
`o2-gv2-labels-2` (Thailand) and `-3` (Greece): O_2 on 10M G v2 labels, stopped by the owner after
about an hour (no part finished; G v2 labels proved barely cheaper than exact at trick 2). `o2-labeltest-1`,
`o2-silver-1` (never rented), `student-t2-gv2-1/-2` (the first failed on a runner bug, fixed in
6c7acf5; the second was stopped by the idle watchdog during setup). Their manifests are in `docs/vast/`
and are valid; reuse them deliberately, not by default.


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
  - **Question for the owner (answered 2026-10-14: allowed, a poor play but legal; see the 2026-10-14 entry below and M3-SPEC §13 note 6):** after a turned Joker the dealer burns before trumps are known. If that burned card turns out to be the ace of the suit turned next, is that allowed? The engine forbids nothing retroactively: nobody robs, the turned card stays face up, and after trick 1 the ace is announced out of play (`AceUndealt` = "undealt or in the dealer's Joker burn").
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

## 2026-10-15 (Mauritius night): safety fixes from the code review (branch `p0-safety`, merged into `m3-pimc`; cleanup files also on `main`)
- **Launcher never destroys on one bad answer** (`python/launcher/launch.py`): a failed vast call or an unreadable reply is "unknown", never "gone"; a machine counts as gone only after 3 consecutive readings showing it exited, offline or absent from the instance list, spanning at least 90 s (`Liveness`). A failed R2 call while polling is "no new heartbeat this poll" (logged), never an exception that skips the report. A create call whose answer cannot be read lists the account's instances first and adopts one labelled with the run (no double rent); if the list cannot be read either, the run stops. `--offer-id` now does one search by that id and applies the normal checks (reliability at least 0.90, whole machine, on-demand; a verified host is preferred, but an unverified or deverified one is allowed: owner rule of 2026-10-16, which replaced the first rule of a verified host and 0.98); `vast-run.yml`'s dry run honours `offer_id`, and it has a `resume_from` input (put into the manifest unless it already has one). A destroy is "verified" only when a later status call shows the instance gone; unreadable status 3 times is reported as unverified in the log, the report and the step summary.
- **Cleanup stricter** (`python/launcher/cleanup.py`, on main too): a machine silent for 20 minutes (or never heartbeated in 45) that vast shows running and busy (CPU at least 50%, or no CPU reading) is asked to stop and upload first and destroyed only if still silent 15 minutes later; every destroy needs a second reading of the machine, and an unreadable vast answer destroys nothing that round; destroys are verified by a later status call (logged).
- **Resume safety** (`crates/search/src/measure.rs`, `crates/runner`): every worlds-file record and checkpoint line carries a fingerprint (seed, decision, world source, equity table SHA-256); a resume or extension under other settings is refused by name before the file is touched, and records made before fingerprints existed are read but never resumed. Two checkpoint lines for the same world must agree (else the checkpoint is refused as corrupt). The runner's `search_measure` chunk writes `out/measure/fingerprint.json` (method, label, seed, trick, count, start_hands, ids, first/last, from, to, table hash) and refuses a resume whose fields differ. An rclone listing that fails is now an error (only "directory not found" lists as empty): a format-1 `resume_from` fails loudly instead of restoring nothing, a format-2 chunk retries 3 times then says so loudly; partial uploads and restores skip `*.tmp` and `*.restore-tmp` files. Byte-identical worlds files on a matching resume still hold (`worlds_resume_per_world`, `resume_worlds` extended with refused wrong-seed and wrong-table resumes).
- **Small fixes:** `vast_status.py` reads `R2_S3_ENDPOINT` (falls back to `R2_ENDPOINT`); CI runs `resume_worlds` (release, `--ignored`) and the `python/tools` tests; the table-size sweep note says what it covers (worlds short enough to finish, from every position); M3-SPEC's guard text says 95% of threads; `v2.rs` no longer claims byte-identical bundles for a resumed chunk (the worlds file is, the timing file and `resumed` counts are not); the per-player stake slips fixed (a match swings 8 stakes per team, 4 per player: +3/−1).
- **Docs:** PLAN amendment A3' (exact PIMC required at tricks 1–2 for the yardstick and teacher targets; rollout-PIMC measured at trick 2 to lose about 13× more money per decision; rollouts only as a scalable fallback), CLAUDE.md to match; STATUS header now M3 (in progress); the Joker-burn question marked answered (STATUS, PLAN, M3-SPEC §13 note 6); VAST-MACHINES network cap $0.10/GB; M3-SPEC §13.1 item 1 now states the golden-N rule actually used (smallest N whose fitted loss is at most 0.01% per decision at the 95% upper bound).
- Tests: launcher 37 + cleanup 18 + tools 11 (Python); `cargo test --workspace` green; `resume_worlds` (release) green.

## 2026-10-16 (Mauritius early morning): machine-cost database and live view (branch `p0b-machines`, merged into `m3-pimc`; cleanup also on `main`)
Owner request: a database of every machine we rent with its real cost per 10^12 solver nodes [positions the exact solver examined; all our work is counted in them], its specs and a live view while it runs, so future rentals are picked from measured data.
- **Runner** (`crates/runner/src/machine.rs`, hooks in `main.rs`, `net.rs`, `measure.rs`): `search worlds` now counts the solver nodes of every world (in each checkpoint line, timing record and its progress file; `crates/search/src/measure.rs`, `pimc.rs`; values unchanged, the worlds file unchanged). The runner tallies the nodes and worlds finished on this machine per step (`search_measure` steps and `t1bench` windows) and adds to every heartbeat's `progress`: `nodes_total`, `worlds_finished`, `usd_so_far`, `usd_per_1e12_nodes` (null until a node is counted), `nodes_per_s_per_thread`, `settling`; the compact log line ends `$/1e12 0.312 (settling)`. Every 10 minutes and at the end it writes `machines/<run_id>.json` (mutable, bucket root; immutable history under `machines/history/<run_id>/<utc>.json`; the same file under the run root) with the machine (CPU, threads, RAM, $/h, offer/host/instance numbers now passed by the launcher as `OFFER_ID`/`HOST_ID`/`MACHINE_ID`, binary, image), the run so far (minutes, $, worlds finished/total, nodes, schedule kind `plan`/`fill`/`split`, table budget, trick, chunk) and the two renting figures; the final record carries `state: final` and the outcome (`DONE`/`FAILED`/`STOPPED`). Uploads use the background network lane (never block). Each worlds step also sends its final counts in a heartbeat of its own, so a step shorter than a heartbeat interval still counts. Not covered: `solver_spike` chunks and format-2 runs (their records stay local).
- **Cleanup** (`python/launcher/cleanup.py`, also on `main`): closing a handed-over run marks its machine record final (outcome, minutes and dollars from the run's last heartbeat, else the launcher's watch data as an upper bound; cost per node recomputed), unless the runner already did.
- **Tool** (`python/tools/machines.py`, stdlib, R2 variables as `vast_status.py`): `--live` (one line per running machine: run, CPU, threads, RAM, $/h, minutes, $ so far, worlds finished/total, $ per 10^12 nodes so far, "settling" while under 10% of the worlds are done, "STALE" when a running record is 20 minutes old), default table of every record, `--by-cpu` (per CPU model and schedule: runs, mean and best measured $ per 10^12, nodes/s per thread; estimated-only records listed apart), `--rank` (today's offers through `vast_watch`'s search and cost model, re-costed with the measured thread speed where a CPU model has a record, estimates marked `~`), `--record-speeds` (measured speeds into `docs/vast/speed-factors.json` through `vast_market.py record`), `--backfill [--runs] [--write] [--out]`.
- **Backfill run and committed:** 18 past runs (the four killed trick-1 runs b2/a7/b3/b, t2d/t3/t4c, and eleven `t1bench-*` runs) now have records in R2 under `machines/`, copied to `docs/vast/measured-machines.json`, and summarised in `docs/VAST-MACHINES.md` ("Measured on our runs"). The t1bench records are measured (their results' node totals); the older `search_measure` runs recorded no node counts, so their nodes are estimated from the worlds' thread-seconds and the measured thread speed of the CPU model (marked `~`, left out of the measured means). Run `m3-agree-t1-ab1` (in progress on the old binary) has no record.
- **Checks:** fmt, clippy -D warnings, `cargo test --workspace` green (runner +2 tally tests); launcher 37 + cleanup 19 + tools 21 Python tests; a local `search_measure` chunk (trick 3, 32 decisions × 48 worlds, `RUNNER_REMOTE_DIR`) wrote the running record, then the final one (two history copies), with `nodes_total` 4.87e7, 1,536 worlds and the cost per node in its heartbeats and log lines.
- **Caveat for readers:** early figures read high because the hardest deals run first; a run killed early keeps that figure. Short runs are dominated by boot and setup (m3-agree-t4c: 3 minutes, $0.07, about 4e7 nodes).

## 2026-10-15/16 (Mauritius night): exact robber burn, five adversarial reviews, new runs (branch `p3-burn`, pushed as `origin/m3-pimc`)
**Who builds what (owner, 2026-10-16, now in CLAUDE.md):** Fable 5.1 reviews, specifies and owns quality; Opus does the difficult builds (solver, search, anything touching correctness); Sonnet does the basic builds (scripts, docs, small fixes, tooling).

**Phase 3, exact robber burn, landed** (commits 13d3858 and 65b6a96, built by Opus, then reviewed). Another seat's rob burn is now searched inside the exact solve [its team picks the burn that is best for it] instead of being fixed in the world. Another seat robs in about 74% of sampled trick-1 worlds (count 32, seed 1, 4,000 worlds) and 65% with count 128, so this matters for most trick-1 values. The review found a wrong-answer bug, fixed in 7253b48: the turned card of a pending rob is still live (someone else can play it) until the burn is made.

**Five independent adversarial reviews** (A solver; B equity leaf and scoring; C PIMC and measurement; D engine rules and reference; E runner and cost safety), all merged into `m3-pimc` (commits 52c0253, a00d46d, 0c39384, bd1edb7, 64738d4, d2b7407, 7ec127b, c65a130, 4c58c06, 3e2dedc; merges c7aa75c, f40f7d8, 0a4f3e5). Every finding below was fixed with a test that failed before the fix.
- **Solver (A): wrong values.** Merging equivalent cards ignored the tie order: cards of one hand on either side of another card of that hand were merged, so values were wrong for other teams and sometimes the mover's own team. The fix cuts a group at any card of the hand standing between; about 3% more solver nodes.
- **Yardstick records (C).** Records and checkpoints now carry `VALUES_VERSION` (= 2, M3-SPEC §13.1 item 12). Records valued by another code version are refused on resume, extend or join, and records without a fingerprint cannot be joined with ones that have it. **Every worlds file made before 2026-10-15 cannot be extended by the new binaries** (their values differ: burn rule and merge fix).
- **Decision sets (C)** no longer draw the play straight after a seat's own rob burn (the bot decides burn and first card together); about 40% of the robber decisions drawn before were these. Sets rebuilt with the same seed differ from old ones; fingerprints keep them apart.
- **Rollout leaf (B):** the match winner is read from games won, not from money (it was wrong only with a stake of 0).
- **Engine (D):** hand-start states refuse the unreachable scores 1 and 6.
- **Runner, launcher, cleanup (E):**
  - No default time limit (`max_minutes` 0) and a $20 default dollar cap.
  - A replacement machine is not stopped by its predecessor's STOP.
  - The busy threshold for a silent machine follows `low_cpu_pct`, so low-CPU runs are not destroyed as idle.
  - A failed done-list write no longer ends a format-2 run; failed final uploads no longer self-destroy (format 1).
  - A failed create that appears late is adopted, not rented twice.
  - Cleanup keeps "offline" machines that still heartbeat and never closes a handed-over run on an incomplete listing.
  - `--offer-id` checks RAM and cores; a hung `vastai` call times out at 180 s.
  - Unknown stream keys in a format-2 manifest are refused by the runner and the launcher (a stream-level `threads` was silently ignored before).
  - Format-2 chunks run together only when their tables fit in memory (else lower priorities wait).
  - `boot.sh` writes FAILED when the runner refuses a manifest. Image rebuilt: `ghcr.io/guillaumeds/41bots-base:20261015-0a4f3e598996`.
  - The `cleanup.py` changes are also on main (8f2b903).
- **Found sound, with evidence:** every numbered rule and H rule has a direct test; 150,000 differential matches (65M actions) with zero disagreements; hand scoring recomputed by hand over about 3M cases; the equity table's index and rotation recomputed by independent recursion; the table rebuild is byte-identical on 1 and 4 threads; world sampling matches an exact counter (6,510 card-by-place checks) and the rejection sampler; outputs are byte-identical across threads and chunking; no floats or iterated HashMaps in value paths; RNG golden vectors match.

**Open items from the reviews (not done):**
- A format-2 run cannot be continued under its own run id after FAILED or out-of-memory. The golden run should use format 1 with `resume_from`, or a "continue" mode must be built.
- No machine-cost record (`machines/<run>.json`) for format-2 runs.
- A format-2 run stopped by the cap or a person writes DONE (not STOPPED).
- Recorded dollars leave out boot time and storage/network.
- The heartbeat ETA of a format-2 `search_measure` stream reads done_frac 1.0 before any world finishes (cosmetic).
- `vastai search offers id=N` reported two offers "not on offer" within a minute of a listing that showed them: the offer-id path is unreliable; the normal search with `cpu_match` worked.
- **Question for the owner:** breaking ties by rank instead of card index would make every equivalent-card merge valid (saves about 3% nodes) but changes the documented tie rule (M2, M3-SPEC). Not changed.
- `read_jsonl` skips unreadable lines silently; `score` matches records by id alone and ids repeat across tricks; the curve fit drops grid points with mean at or below 0; the "95% upper bound" is a bootstrap 97.5th percentile (wording).

**Runs in progress (times Mauritius, UTC+4):**
- `m3-agree-t1-ab1` (the other thread's run; old binary from the 56851ce era, so old values and the rules burn; useful only for the deal-count curves) is still running on instance 56149309: AMD EPYC 9654 server chip (368 threads, $1.02/h). At 04:08 on 16 Oct: 352 of 2,560 worlds, $3.78 spent. The runner's rough ETA is the evening of 17 Oct at about $45; the $20 cap would stop it at about 20:00 on 16 Oct. It can be resumed only with its own binary.
- **Phase 4 table measurement `m3-t1-tables-1`** launched 04:11 on 16 Oct on instance 56173856: AMD EPYC 7542 server chip (122 usable threads, $1.92/h), cap $10, `low_cpu_pct` 1, binaries 0a4f3e5 and the image above. Format-2 manifest (`docs/vast/m3-t1-tables-1.manifest.json`) with three `search_measure` streams on worlds 0-3 of the trick-1 decision set (count 32, seed 1; ids 0-7, one world per thread, 32 at once) with tables of 12 GB, 3 GB and 1 GB; the 12 GB stream runs alone, then the 3 GB and 1 GB streams together. Purpose: nodes and seconds per deal against table size for the one-deal-per-thread scheduler (Phase 4) and for pricing the golden set.

## 2026-10-16 (Mauritius morning): hybrid yardstick rung 1 (O_4) built (branch `m3-hybrid`)
- Built per `docs/M3-HYBRID-SPEC.md`: the solver's optional frontier at the start of trick k (naive, fast and parallel solvers identical; the evaluator sees a canonical form so equivalent-card merging and the table stay exact), `crates/nn` (encoder `oracle-v2`, whole-number MLP and softmax, safetensors, label shards), `search labels`, `python/nn` (trainer, quantiser, Rust/Python bit-for-bit parity in CI), `search hybrid-check` (spec §3).
- Small run on this VM: H_4 vs exact on 40 trick-3 decisions × 64 worlds: mean loss 0.022% (95% upper 0.043%), agreement 82.5%, value bias −0.16%, SD 3.0%: **fails** the 0.01% standard, as a first small network should. 65–70 µs per network evaluation. Details, the full-run commands (about 2–8 core-hours, off this VM) and the spec questions: `docs/m3-hybrid/rung1.md`.
- Found: the trick-4 golden set `m3-agree-t4c` was drawn by the rejection sampler (binary `bc4835a`, before the belief sampler); its worlds reproduce byte-identically only with `--source rejection`.
- Rung 1b (spec §6a, same day): the outcome head [the network says how the hand ends; the money chances of each ending come exactly from the equity table]. Completeness checked (every label has an ending with exactly its value). Same small run: O_5 values nearly exact (mean |error| 0.016%), H_5 vs exact on 40 golden trick-4 decisions: loss 0, agreement 100%, bias of two other teams 0.02% (fails); H_4 vs exact on 40 trick-3 decisions: loss 0.019% (upper 0.040%), value SD 2.1% (was 3.0%): still fails, O_4 needs more labels. Full-run workflow for review: `.github/workflows/hybrid-rung1.yml` (manual, not run).

**2026-10-16: launcher fix, a machine that is already gone no longer fails the job** (`python/launcher/launch.py`, `destroy()`). When the machine had destroyed itself (the runner does this at the end of a run) or a stop raced it, vast no longer knew the instance, so every destroy call failed and the job ended red with "could not destroy instance ...; destroy it by hand" although nothing was left to destroy (seen 3 times on 16 Oct: `m3-agree-t1-ab1`, `m3-t1-tables-1`, and the vast-stop of `m3-t1-leader-layouts-1`). Now, when a destroy call fails, the launcher reads the instance's status; if vast shows it gone [exited, offline, or missing from an instance list that was read successfully] it logs "already gone" and counts it as destroyed. The safety rule is unchanged: if the status cannot be read, that is never counted as destroyed, and the destroy is retried and ends "failed" after 5 attempts. Five new launcher tests (71 in all) pass; the cleanup workflow's own `destroy()` (`cleanup.py`) retries quietly and closes the run next round as "the machine was gone", so it was not changed.

**2026-10-16: the host reliability floor is now 90% (owner rule: "you can drop always to 90%")**. The launcher (`python/launcher/launch.py`, `MIN_RELIABILITY`) accepts vast.ai hosts with reliability of at least 0.90 for every rental, where it was 0.98: the default of `--min-reliability`, the offer search (`reliability>0.9`) and the `--offer-id` checks all use it. `vast-run.yml` (`min_reliability` input, default 0.90) and `ops-check.yml` follow, and so do the market tools (`vast_market.py`, `vast_watch`), so their "can the launcher rent it" labels match. `--min-reliability` stays, so a single run can still ask for a steadier host. The market snapshot now lists a "reliability_0.80-0.90" band (just under the floor) instead of "reliability_0.90-0.98"; saved snapshots and past run records are unchanged, and older entries above that mention 0.98 describe what was true then.

**2026-10-16: launcher fix, a destroyed machine that still answers "status None" counts as gone** (`python/launcher/launch.py`, `instance_state()`). After a vast-stop of `m3-t1-leader-layouts-9654` (GitHub job 116915522929, 09:54 UTC) the destroy call went through 5 times, but vast's `show instance` kept answering a record with `actual_status` None, so every follow-up reading said "still there" and the job ended with "could not destroy instance". A record with status None (and not marked stopped) is now not decided on its own: the account's instance list is read. Not in a list that was read successfully means gone; in the list means the listed row decides (None there too counts as alive, since a machine still loading shows None as well); a list that cannot be read stays "unknown", which is never taken for gone. Three new checks in the launcher tests, plus one for `destroy()` ending "verified".

**2026-10-16: launcher fix, `destroy()` needs two dead readings 10 s apart** (`python/launcher/launch.py`, `destroy_reading()`). At 09:56 UTC vast's `show instances` once held no 41bots instance (and can return an empty list) while the machine of run `m3-hybrid-smoke-1` (56229982) was still running and sending heartbeats. The last two fixes made "absent from a successfully read listing" count as gone, and `destroy()` decided on that single reading (after a failed destroy call, and when a record showed `actual_status` None), so one wrong listing could have counted a live machine as destroyed. `Liveness` was already safe (3 readings over at least 90 s). Now `destroy()` counts "dead" only when two readings at least 10 s apart (`DESTROY_RECHECK_S`) both say dead, after a failed call and after a successful one; an "alive" second reading resets it, "unknown" is never dead. The other rules are unchanged (5 attempts, "unverified" when the status cannot be read, "failed" otherwise). Four new launcher tests and the existing destroy tests updated (78 in all).

**2026-10-16: launcher fix, slow image pulls no longer kill a machine, and fast downlinks are preferred** (`python/launcher/launch.py`, tests in `test_launch.py`). The base image grew to 5-6 GB (PyTorch baked in). In run `m3-hybrid-r1-c16m-2` (GitHub job 116965722006) two rented machines were still pulling it [downloading the image's layers] after 15 minutes, and the launcher's rule "no heartbeat within 15 minutes of renting" destroyed both: about $0.14 and 33 minutes lost, while the third machine pulled in about 6 minutes and finished. Three changes. (1) **Pull-aware start timeout:** once vast's status message reads "success, running ..." (the container started), the 15 minutes to the first heartbeat count from that moment. A machine that is still pulling when the 15 minutes from renting are used up is given more time as long as vast's status message shows pull progress ("Pulling fs layer", "Downloading", "Download complete", "Extracting", "Pull complete") and has changed in the last 10 minutes (`--pull-stall-min`), up to 40 minutes from renting (`--pull-ceiling-min`); after that, or after 10 minutes of an unchanged message, it is destroyed as before. A machine that never showed pull progress still dies at 15 minutes, and nothing is destroyed earlier than before. Every poll logs "still pulling the image (N min): <status message>". (2) **Fast downlinks:** every offer search now carries `inet_down>=500` [vast's download speed field, in Mb/s; checked in the vastai 1.8.3 query list] through `--min-inet-down` (default 500; 0 = no filter), in the verified search, the fallback search and every replacement's search. `--offer-id` reports a slower link in the log and rents it anyway (the owner named it). (3) **Ranking:** without `--cheapest`, an offer under 200 Mbps has its cost per unit of work multiplied by 1 + (200 - Mbps) / 200 (x1.5 at 100 Mbps, x2 at 0), so the chooser avoids it; an offer that reports no speed gets no penalty. `vast-run.yml` got a comment only: it has no new input (25-input limit), so the defaults apply, and on the hybrid branch a manifest's `machine` object takes `min_inet_down` (Mbps; its value replaces the default 500, 0 = no filter, the larger of the two wins if `--min-inet-down` is given too; a negative number is refused before anything is searched or rented). Not changed: the cleanup workflow still destroys a machine that has never heartbeated after 45 minutes (`NO_HEARTBEAT_S`), only 5 minutes above the new 40-minute pull ceiling. Nine new launcher tests (94 launcher and cleanup tests in all; 101 on m3-hybrid with two more for `machine.min_inet_down`), the 20 tools tests pass.

**2026-10-16: three follow-ups to the slow-pull fix** (`python/launcher/cleanup.py`, `.github/workflows/vast-status.yml`, the market tools). (1) **Cleanup waits longer for a first heartbeat:** a machine that has never heartbeated is now destroyed after 60 minutes, not 45 (`NO_HEARTBEAT_S`). The launcher can wait up to 40 minutes for an image pull that is still advancing and then 15 more for the first heartbeat, so a healthy but slow machine can need 55 minutes; a test keeps the limit above the launcher's pull ceiling plus 15. (2) **vast-status no longer fails on a run with no heartbeat yet:** the "Run heartbeats" step listed the heartbeat folder with `aws s3 ls`, which prints nothing and exits 1 for an empty folder, and the job's strict shell settings turned that into a failed job for a run that was only just rented and still pulling. Now exit 1 from the listing means "none yet": the step prints "no heartbeats yet for <run_id>" and exits 0, and any other aws failure (bad keys, no network) still fails the job. The failure was reproduced first with a fake empty listing in the tools tests (they run the workflow's own step text against a fake `aws`), then fixed. (3) **One place for the launcher's offer rules, and the "can the launcher rent it" labels follow them:** the numbers and checks now live in `python/launcher/offer_rules.py`, which `launch.py` imports for its search and its `--offer-id` checks, and which the machine dashboard (`vast_watch`), the market snapshots (`vast_market.py`), the second-machine check (`vast_status.py`) and the machine ranking (`machines.py`) import too, so the two cannot drift apart again. The label now means: reliability at least 0.90, the whole machine, a download link of at least 500 Mbps (or the value passed with `--min-inet-down`; 0 = no filter), network cost at most $0.10 per GB, at least one GPU, and for a GPU job exactly the GPUs asked for with compute capability 600 to below 1000 and CUDA 12.6 or newer. A verified host is preferred, but an unverified or deverified host is allowed: it shows "yes, unverified host" and comes after verified ones in the second-machine list. A verified host with a 300 Mbps link, and a host at reliability 0.89, are "no" (hover a "no" in the dashboard for the reason). New market snapshots no longer filter on verified hosts and do carry the link speed; old saved snapshots keep describing themselves as they were made. Tests: launcher 104 (cleanup and the new rules module included), tools 28, dashboard model 17.

**2026-10-16 (18:10-18:30 Mauritius): the launcher waits for a machine instead of failing, with a price ladder; new floors; bad machines are remembered** (`python/launcher/launch.py`, `offer_rules.py`, `vast-run.yml`, the market tools). Owner rules of that evening, which replace the numbers in the entries above (a 500 Mbps link, and "exactly N GPUs"). (1) **Waiting:** when no offer matches, the launcher no longer exits. It searches again every `--offer-poll-s` (60 s) for up to `--wait-for-offer-min` (120; 0 = the old behaviour: one search, then stop), one log line per attempt with the price step, the attempt number, the elapsed minutes and the closest miss [the nearest thing to a match: an offer that fits but costs more than this step, or the biggest machine under the thread floor]. A replacement search after a start failure waits the same way and keeps every floor of the first search; the launcher also re-checks the threads, RAM, link and price on every offer vast returns, so nothing below a floor is rented. `plan` and `--dry-run` do not wait: they print what a real run would wait for. (2) **Floors:** at least 190 threads (`--min-cores`, workflow input `min_cores` default "190"), at most $1.70 an hour (`--max-dph`, also for `--offer-id`; a manifest's machine block can raise it), download link at least 100 Mbps (was 500; the baked image pulls in about 7 minutes at 100 Mbps and the pull-aware start timeout allows 40), and a GPU job needs at least the GPUs asked for (a 2-GPU machine is fine, the trainer uses cuda:0), always the whole machine. The ranking penalty for links under 200 Mbps is unchanged. (3) **Price ladder:** the waiting search looks for the cheapest first: `--price-steps` 1.2,1.45,1.7 dollars an hour with `--tries-per-step` 3 attempts each, a minute apart, then the cap until the wait runs out; the first offer that matches at the current step is rented (ranked by cost per unit of work among the matches at that step), and a replacement search starts again from the first step. An offer at $1.40 is rented on attempt 4; one at $1.90 never, unless the cap is raised. (4) **Defaults in the workflow:** `min_cores` "190", and `cheapest` is now off, so the launcher ranks by cost per unit of CPU work (its own rule) unless a test run asks for the lowest hourly price. There are no new inputs (the wait and the price steps are launcher options and, on the hybrid branch, manifest machine-block keys), because the workflow is near GitHub's limit of 25 inputs. (5) **Bad machines are remembered:** an instance that ends as "container failed to start" (the case that rented the same broken host twice: a broken NVIDIA container runtime on machine 58709, host 60400) or that never heartbeats is recorded in R2 `launcher/bad-machines.json`, and every search leaves machines listed in the last 24 hours out ("excluding machine M (host H): container failed to start at <utc>"); `--offer-id` on such a machine warns and goes ahead. (6) **Market tools:** the "can the launcher rent it" labels use the same numbers (the shared `offer_rules.py`): 100 Mbps, the $1.70 cap, at least N GPUs; a verified host with a 300 Mbps link is rentable, one with 80 Mbps is not; an offer above $1.70 is listed under "not rentable" (`price_over_cap`) in new market snapshots. Tests: launcher 117 (cleanup and rules included), tools 29, dashboard model 18. On `m3-hybrid` the launcher tests number 127; its manifest `machine` block also takes `max_dph`, `wait_for_offer_min`, `offer_poll_s`, `price_steps` and `tries_per_step` (the command line wins, else the manifest, else the default), a manifest's `min_cores` can raise the 190 floor but not lower it, and a 2-GPU machine is rentable for a 1-GPU job.

## 2026-10-16 (Mauritius afternoon): hybrid rungs on one rented GPU machine (branch `m3-hybrid-vtrain`, nothing rented)
- New runner chunk kind `hybrid_rung` (`crates/runner/src/hybrid.rs`): the rung-1 GitHub job as one chunk on one vast.ai machine with many CPU threads and one GPU: equity table, Python environment (PyTorch 2.14.1+cu126 through `uv` 0.13.0, pinned by hash in `python/nn/uv.lock`, group `cu126`; the `cpu` group stays the default for this VM, the laptop and GitHub), GPU check (PLAN §12.3 step 5), labels on every thread, trick-3 golden set, training on the GPU [graphics card used for network training], quantising, `search hybrid-check`, reports and networks uploaded. No usable GPU: training on the CPU with every thread, with a warning in the heartbeat, the launcher log and `rung.json`.
- Trainer: `--device cpu|cuda|auto` (default `cpu`: the CPU path gives the same networks as before, SHA-256 checked), seeded and deterministic switches on the GPU, memory-mapped shards read a block at a time, `--progress`. GPU float training is not bit-identical across machines; the integer network's SHA-256 is what is validated and recorded.
- Launcher: `--gpus`, `--min-gpu-ram-gb`, `--min-compute-cap` and a manifest `machine` object (all the old checks kept); vast-run inputs `gpus`, `min_gpu_ram_gb`, `disk_gb`. Release bundle: `uv` and `python/nn`. Image unchanged (`20261015-0a4f3e598996`).
- Manifests ready, not launched (price and owner approval first): `docs/vast/m3-hybrid-smoke.manifest.json` (about 10 min, about $0.20) and `docs/vast/m3-hybrid-rung1b.manifest.json` (the GitHub run's settings: 4M trick-4 labels, 400 x 320 trick-3 golden set, 1024-512-128 outcome head; about 15–30 min, about $0.25–0.50; estimates). How to launch, what it records, time per step: `docs/m3-hybrid/vast.md`.
- Checked on this VM (no GPU): the chunk end to end from a release-like bundle with uv 0.13.0 and CPU PyTorch (30–90 s at small counts), runner and launcher tests, the Python tests.

## 2026-10-16 (end of day): the hybrid yardstick, what was built, rentals and open items (branch `m3-hybrid`)

**Strategy.** The yardstick is the hybrid [exact search down to the start of trick k, then a network that values the position at that point], which is about 100 times faster per world than exact search at trick 2 (12.7 ms against 1,289 ms). Each network is checked against exact search on a fixed golden set. One network per rung: O_4 (values the start of trick 4; checked on trick-3 decisions), O_3 (checked on trick-2 decisions), then O_2 (to be checked on trick-1 decisions). The standard is the same for each: decision loss at most 0.0100% overall and in every seat band (95% upper bound), every team's bias at most 0.0100%. Details: `docs/m3-hybrid/rung1.md`, `rung2.md`, `vast.md`.

**Where each rung stands.**
- **O_4 (rung 1):** `rung1.md` has the nine rented trainings; the best is lead-3c: overall 0.0057%, seats 1-2 0.0097%, seats 3-4 **0.0159%**, seats 5-8 0.0007%, largest team bias 0.0057%, agreement 94.5%. Only seats 3-4 are over the line, because of a few systematic decisions (as with lead-2, `rung1.md` note 7); analysis is under way.
- **O_3 (rung 2):** round 1 (16M random-play and self-play labels): overall 0.0105%, seats 1-2 0.0138%, 3-4 0.0091%, 5-8 0.0133%, bias 0.0108%, agreement 84.5%, a near miss. Round 2 (+ 8M harvested positions, cap 32): overall 0.0111%, seats 1-2 0.0156%, 3-4 0.0171%, 5-8 0.0109%, bias 0.0092%, agreement 83.2%: the harvest did not help. The network's outcome accuracy is 87% at trick 3 against 98% at trick 4, so the trick-3 function is much harder. Round 2 reused round 1's golden set from storage (generation took 1,057 s on 184 threads), with the 1% recheck byte-identical. Round 3 (32M random-play and self-play + 8M harvested, network 2048-1024-256, testing capacity) is running. Tables: `rung2.md`, "Results, 2026-10-16".
- **O_2 (the trick-2 network):** not started. It needs the train-only runner mode and the trick-1 check tool, both being built. A trick-2 label costs 1.3 s on one 7995WX thread, so round 1 with 1.25M labels is about $3; awaiting the owner.

**Built and merged on `m3-hybrid` today.**
- *Yardstick:* GPU training on vast.ai (a CUDA-graph training step), the fast exact integer check, streamed label shards (hashed and written while the rows are made), the harvest of frontier positions, the network-driven harvest at trick 3 and reuse of a stored golden set, and the baked PyTorch image `ghcr.io/guillaumeds/41bots-base:20261016-2bb16f03f96d`.
- *Launcher and tools:* the host reliability floor 0.90, unverified or deverified hosts allowed (verified preferred), the pull-aware start timeout, the waiting search with the price ladder, the floors of 190 threads / $1.70 an hour / 100 Mbps, the bad-machine memory, the cleanup limit of 60 minutes for a machine that never heartbeated, the vast-status fix for a run with no heartbeat yet, and the offer rules shared with the market tools (entries above).

**Owner's rules of the day** (the "Where compute runs" and "Parallel rentals" rules are in `CLAUDE.md`).
- 2026-10-16, 18:10 Mauritius, on renting: "the launcher must wait until it gets a cheap 120+ thread machine, actually at least 190+; there are many, they come every few minutes." Then, by 18:20: the thread floor is 190, the price cap is $1.70 an hour (also for a named offer, unless a manifest raises it), the link floor is 100 Mbps, a GPU job needs at least the GPUs it asks for (the price is per machine), and the launcher should "look for the cheapest a few times before hiking the price" (3 searches at $1.20, 3 at $1.45, then the cap).
- Earlier the same day: hosts at 90% reliability are fine; unverified or deverified hosts are fine, verified preferred.

**Rentals and spend today (hybrid yardstick).** The nine trick-4 runs of `rung1.md`: about $2.01 (whole rentals). Then lead-3 $0.35 (cancelled), lead-3b $0.03 (start failure: machine #55016300, whose NVIDIA container runtime is broken, was rented a second time five minutes after its first failure, which is why the launcher now remembers such machines for 24 hours), lead-3c about $0.40, t3-r1b $0.58 (cancelled), t3-r1c $0.90 (O_3 round 1), t3-r2 $0.79 (O_3 round 2). Sum of these figures: about $5.06. O_3 round 3 is running and not counted.

**Open items.**
- Seats 3-4 of O_4: finish the analysis of the few systematic decisions (`rung1.md`, notes 7-9).
- The trick-1 checks, and the train-only runner mode that O_2 needs.
- Start trick 2 (O_2) once the owner agrees to the cost.
- Overlap the golden-set generation with training (round 1 spent 1,057 s generating it before training started), and feed the labels to training a quarter at a time as the owner proposed.
- Expect merge conflicts in the workflow files with `main` when `m3-hybrid` goes to `main`.
- The vast API key cannot create destroy-only keys (the launcher logs it and goes on; machines are destroyed by the launcher and the cleanup workflow only).

## 2026-10-17 (overnight, owner asleep): the audit loops on tricks 4 and 3

**Built and merged on `m3-hybrid` (binaries fcb3e73):** the auditor down to groups of 0.01% (`audit_gamma`) with the
standard and push standard (`audit_standard`, `audit_push_standard`), unmeasurable mini sets, the best round as the result,
audit shards deleted after use, a capacity ladder that only grows; trainer memory released at each streamed join, a stall
watch instead of the runner's fixed training limit, a bound on targeted harvests, `harvest_net` required at cut 3; every
`docs/vast` manifest checked by `cargo test`; the launcher's `gpu_names` (fast GPUs by name); item A (cheaper exact labels,
byte-identical) and item B (batched wide-network evaluation, byte-identical). `main` got the cleanup's 60-minute first
heartbeat rule.

**Runs.** Two mechanism tests (three rounds and two rounds, $0.6 and $1.25), then the trick-4 loop (third launch:
AMD EPYC 7B13, 122 threads, 504 GB, $0.73/h, about $4.2; the first two launches stopped for a mini-set cap and a price
ladder, about $1.4) and the trick-3 loop (AMD EPYC 7663, 224 threads, 2x A100, $1.60/h, about $9.9). Both stopped on the
owner's diminishing-returns rule after four rounds. Overnight spend about $17.5. Tables: `docs/m3-hybrid/audit-loop.md` §8.

**Results.**
- **O_4: 0.0016%** (95% upper bound on 4,000 trick-3 decisions x 320 exact worlds), seats 1-2 0.0026%, 3-4 0.0023%,
  5-8 0.0012%, largest bias 0.0025% (round 1, 1024-1024-512-128, `inputs/sha256/b28750e3…`; round 3 at 2048-2048-1024-256
  ties it). The earlier "seats 3-4 fail" was the 400-decision set's small sample.
- **O_3: 0.0068%** overall (2,000 trick-2 decisions; round 2, 2048-1024-256, `inputs/sha256/f5766aa9…`), but seats 1-2
  at 0.012-0.016% and the leader's own team under-valued by about 0.02% in that band: not yet 0.01% everywhere.
- "Flat" was not reached by either loop: the auditor confirms thousands of small position-level groups at 10M rows and
  flat needs each measured at the decision level (60 a round at trick 4); the push to 0.005% / 0.0075% did not start.

**Decisions for the owner.** (a) Calibrate the auditor's position-level thresholds against the decision-level results
each round records, so that it confirms the groups that can matter for decisions; (b) cut the round-to-round noise of
retraining from scratch (warm start, or several seeds); (c) a targeted study of trick 3's leader positions and own-team
bias; (d) adopt O_4 round 1 and O_3 round 2 as the current best. Not started: the trick-2 loop (branch `m3-hybrid-t2loop`,
built), items C and D.

## 2026-10-17: the trick-1 silver set (branch `eng-silver`, nothing rented)

O_2 checked at the decision level on trick-1 decisions without exact trick-1 solves
(`docs/m3-hybrid/silver.md`). The reference is H_3 [exact search through tricks 1 and 2, the adopted
O_3 at the start of trick 3] on the golden sets' own decisions and deals; the thing tested is H_2
[exact trick 1, O_2 at the start of trick 2] on the same deals; the score is the money H_2's choice
loses under H_3's values, with the golden checks' statistics plus score and race groups.
- **Built:** `search worlds --method hybrid --cut 3 --net O3` (resumable, byte-identical for any
  thread count; records marked `hybrid` with the network's SHA-256, refused wherever an exact golden
  set is required); `search hybrid-check --reference silver --reference-net O3 [--allow-exact]
  [--race-groups]`; the runner's `search_measure` chunk takes `reference`, `share`, `worlds_input`
  and `check`. Manifests, not launched: `m3-hybrid-t1-silver-mech` (16 × 8, about $1),
  `m3-hybrid-t1-silver` (400 × 320), `m3-hybrid-t1-silver-check` (a given O_2 against the stored
  set), `m3-hybrid-t1-anchor` (optional exact anchor, about $3).
- **Pilot (build VM):** H_3 at trick 1 is almost all network time: a leader world asks the
  2048-1024-256 O_3 about 10 million times (10–30+ thread-minutes here); seat 2 averages about 5.5
  minutes, seat 3 3.5, seats 4–8 seconds. **Price on an AMD EPYC 9654 (368 threads, $1.01/h):
  400 × 320 about 42–81 hours, $42–82; 400 × 160 or 200 × 320 $21–41; 200 × 160 $11–21; 400 × 80
  $5–10.** Worlds are nested, so a small set extends later. The mechanism test should confirm the
  rate on the real machine first.
- **For the owner:** which size to run (or a smaller first stage), and whether to add the exact
  anchor that measures the silver's own error at trick 1.

## 2026-10-17 (afternoon and evening, Irish time): the trick-2 run, the app's engine API, the student started

**The trick-2 loop is running** (`m3-hybrid-t2-loop-1`, manifest `docs/vast/m3-hybrid-t2-loop.manifest.json`, binaries
`ebc5d7a`, image `20261017-bc692e446939`, cap $35): AMD EPYC 7742 (245 usable threads, 251 GB, RTX 4060 Ti 16 GB,
$0.72/h), labels from 19:06 Irish time at about 1,730 rows a minute with the CPU steady at 100% (1.2M random rows, then
the rules and validation shards, training, the audit rounds). Expected about 35 hours and about $25. Round 1's network is
the provisional O_2 (out of spec until the loop and the silver set say otherwise), about 15 hours in.
- **Before it:** seven mechanism tests (about $4.6 in all). Fixes they forced: a shard upload that fails is retried, not
  fatal; the pooled harvest schedule (`eebf917`); the plain labels' look-ahead as a 1 GiB byte budget (`ebc5d7a`: with
  one-row chunks at trick 2 the old count window stalled every thread behind one slow row; run 6 swung between 3% and
  96% CPU).
- **Launcher guards added after bad rentals:** a machine whose container gets far fewer threads than listed is destroyed
  at its first heartbeat, recorded and replaced (an AMD EPYC 9B45 gave 46 of 384 threads, twice); Xeon Phi chips are
  never rented; an operator's stop is never followed by a replacement (`NO-REPLACE`); an upload-speed floor
  (`min_inet_up`); Blackwell GPUs by opt-in with the CUDA 13 image (`cu130` group of PyTorch 2.14.1).
- **Turned pairs** (rules 34, 37-41) are in the engine and the reference (branch `eng-turned`, merged; APP-ENGINE-CONTRACT
  §7.1); the owner's ruling Q1 (a pair turned while its led trick is played) waits for the pros.

**The app's engine API is built and merged** (`docs/WASM-API.md`; `crates/wasm/dist/engine.wasm` 834,515 bytes): the app
sends its own `maskMatch` view; PIMC with H_3 at trick 2, exact from trick 3, trick 1 refused until an O_2 is loaded;
the coach's fields with `lines: true`; byte-identical natively and in WASM and for any split across workers. In WASM on
the build VM a trick-2 deal costs 368 ms when leading, 128 ms second, under 40 ms later; four workers fit about 25 deals
at a trick-2 lead in 5 s. The owner sent the app session a prompt to wire it behind a switch and test it by duplicate
tournament against today's Pro.

**The student network build has started** (branch `eng-student`): a small network that values every legal card from the
seat's view, trained on stored teacher records (each decision's raw position, every card's value in every deal, stamped
with the teacher's network, the equity table and the rules), so later rounds only add data (DAgger, expert iteration).
First data run (trick 2, 250k decisions at 320 deals) about $1-2 per 250k on the 245-thread machine, to be priced and
approved when the pipeline is built.

**Silver set merged** (`eng-silver`, with the per-seat time priors fixed); its size is still the owner's choice.
