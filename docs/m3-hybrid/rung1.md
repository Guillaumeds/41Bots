# Hybrid yardstick, rung 1 (O_4): build report

Branch `m3-hybrid`, 2026-10-16 (Mauritius). Spec: `docs/M3-HYBRID-SPEC.md`. Rung 1 (below) was
followed the same day by **rung 1b, the outcome head** (spec §6a): see its section after the
first small run.

**In one paragraph.** The whole pipeline now exists and runs end to end: the exact solver can stop
at the start of a chosen trick and ask a network instead; the network's inputs are produced by one
encoder in Rust; labels (positions with their exact value) are generated and saved; a network is
trained in Python, turned into whole-number arithmetic, and loaded back in Rust where it gives
byte-identical answers on every machine; and `search hybrid-check` measures the hybrid against
exact search exactly as spec §3 asks. The small networks trained here (a few minutes on this VM)
are **not** good enough yet: the hybrid H_4 loses 0.022% money chance per trick-3 decision (95%
upper bound 0.043%) against the 0.01% standard, and the network's values have a bias of −0.16%.
That is the expected starting point of a rung; spec §3 says what comes next (more labels, a
larger network, harvested positions).

## What was built

1. **Solver frontier** (`crates/solver/src/frontier.rs`). An optional "frontier" [a cut-off point]
   at the start of trick k: every position the search reaches there is valued by an evaluator
   instead of being searched further. It works in the plain reference solver, the fast solver and
   the parallel solver, which give identical answers (tested under every switch combination, both
   tie orders, 1–4 threads, and trick-1 roots with a rob still to come). Without a frontier nothing
   changes (all existing identity tests pass).
   - The evaluator is shown a **canonical form** of the position [a standard rewriting that keeps
     exactly what the exact value depends on]: the live cards by suit class and relative rank, who
     holds each and its place in that hand's card order (the tie rule depends on it), scores,
     flags, tricks, the 5 of trumps, and the team that would take the end-of-hand bonus. This is
     what keeps the fast solver's shortcuts exact with a network at the frontier: two lines that
     differ only by which of two equivalent cards was played reach the same canonical form. The
     proof is in the module notes; tests check that equal canonical forms always have equal exact
     values (about 9,000 reached positions, 7,600 of them sharing a form, plus moved dead cards).
2. **`crates/nn`** (no `unsafe`):
   - the **encoder** `oracle-v2`: 809 on/off inputs (cards by slot and holder and hand place,
     trumps, scores, flags, tricks, the 5, bonus team, games, dealer; seats turned so the leader is
     seat 0) and 100 number inputs from the match-equity table: each team's chance if the hand
     ended now, if a team won the game now, and after each possible winner of the next trick and of
     the next two tricks (PLAN §6.5 "reachable successor equities"), given as log₂ of the chance so
     the network can pass a chosen outcome straight through;
   - **whole-number inference**: integer weights, 64-bit integer sums that are proven not to
     overflow, ReLU capped at 64, four output scores, then a fixed whole-number rule (an integer
     exponential built from integer square roots, then the money type's largest-remainder rounding)
     to a money chance summing to exactly 2^32;
   - weights read and written as safetensors (a small reader and writer of our own instead of the
     `safetensors` crate, so nothing new is pulled in), label shards as `.npy` files of fixed
     little-endian records;
   - `nn-tool` (test fixtures, evaluating a shard, golden outputs).
3. **Labels**: `search labels --trick K --count N --seed S --source random|rules --out FILE.npy
   --equity TABLE [--threads T]`. "random": a yardstick-style decision (all 8 positions, strata
   S1–S4, decision tricks 1..K−1), a world drawn from the decider's belief, then random legal play
   to the start of trick K. "rules": rules-v1 self-play to the start of trick K. The target is the
   exact solver's value (tie rule: first strictly best card by card number). A JSON sidecar records
   encoder, values version, tie rule, binary and table SHA-256, seed, counts and the shard's
   SHA-256. Same bytes for any thread count; seed 1 (the golden sets') is refused.
4. **Python trainer** (`python/nn`, uv, Python 3.14, PyTorch 2.14.1 CPU): plain MLP [layers of
   weighted sums], soft cross-entropy against the exact four-vector, Adam, seeded, early stopping
   on a validation shard; saves float and integer safetensors with the training-data hashes; the
   quantiser [turning float weights into the Rust whole-number scheme] and a numpy integer forward
   pass. **Parity**: Rust and Python integer networks agree bit for bit on 3,000 encoded real
   positions and two random networks, and both reproduce a committed golden file; CI runs this
   (numpy only, no PyTorch).
5. **`search hybrid-check`** (spec §3): for every decision and world, every candidate valued
   exactly and by H_k on the same world; decision loss with a 1,000-replicate bootstrap (seed 1)
   95% upper bound, overall and per position band (1–2, 3–4, 5–8); value error (bias, SD, 99.9th
   percentile, max) overall, per band and per position; agreement; time per world and per network
   evaluation. Worlds are regenerated exactly as the yardstick draws them; exact values are
   recomputed, or read from a stored worlds file after a sample is recomputed and found
   byte-identical (any difference stops the check).

## The small run (this VM, each step under 5 minutes)

| step | what | time |
|---|---|---|
| labels O_4 | trick 4: 720,000 random (seed 101) + 180,000 rules-v1 (seed 102); validation 20,000 random (seed 201) | 61 s on 4 threads |
| train O_4 | MLP 512-256-64, batch 512, stopped by the 4.5-minute limit after 14 epochs (best epoch 9) | 5.3 min |
| labels O_5 | trick 5: 240,000 random (301) + 60,000 rules-v1 (302); validation 20,000 (401) | 29 s |
| train O_5 | MLP 512-256-64, 22 epochs (best 17) | 2.9 min |

Validation (20,000 positions, each team's chance): O_4 error SD 3.0%, mean |error| 1.3%; O_5 SD
1.1%, mean 0.5%. The integer networks differ from the float ones by at most 0.00015%.

**Check 1, the rung's own measure (small form): H_4 against exact on trick-3 decisions.** The first
40 decisions of the trick-3 golden set (seed 1), 64 worlds each, regenerated (belief sampler),
exact values recomputed. Net `e9fdee7dac84…`.

| group | decisions | mean loss | 95% upper | ≤ 0.01%? | agree | values | bias | SD | 99.9% \|e\| | max \|e\| |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 40 | 0.0217% | 0.0431% | no | 82.5% | 6,784 | −0.163% | 3.05% | 31.9% | 53.9% |
| pos 1–2 | 16 | 0.0501% | 0.1030% | no | 62.5% | 2,816 | −0.183% | 3.43% | 49.0% | 53.9% |
| pos 3–4 | 8 | 0.0000% | 0.0000% | yes | 100% | 1,216 | −0.162% | 3.90% | 31.1% | 31.1% |
| pos 5–8 | 16 | 0.0043% | 0.0129% | no | 93.8% | 2,752 | −0.143% | 2.03% | 27.8% | 35.5% |

Per position (loss, agreement): 1: 0.091%, 38%; 2: 0.009%, 88%; 3, 4, 6, 7, 8: 0, 100%; 5:
0.017%, 75%. Time: exact 3.9 ms a world, hybrid 15–16 ms a world (231 network evaluations a
world), **64–69 µs per network evaluation** (the forward pass alone 62 µs, measured by `nn-tool
eval`; the rest is encoding). Result: FAIL. The same numbers come out on 2 and 4 threads.

**Check 2, sanity: H_5 against exact on the trick-4 golden set** (decisions 0–39, 64 worlds each,
stored exact values from `m3-agree-t4c`; a 5% sample, 145 worlds, recomputed and byte-identical).
Net O_5 `78ba4ec1f7c2…`.

| group | decisions | mean loss | 95% upper | ≤ 0.01%? | agree | bias | SD | 99.9% \|e\| |
|---|---|---|---|---|---|---|---|---|
| all | 40 | 0.0037% | 0.0094% | yes | 95.0% | −0.106% | 2.01% | 40.6% |
| pos 1–2 | 16 | 0.0044% | 0.0131% | no | 93.8% | +0.019% | 0.98% | 8.1% |
| pos 3–4 | 8 | 0.0095% | 0.0286% | no | 87.5% | −0.059% | 0.61% | 3.8% |
| pos 5–8 | 16 | 0.0000% | 0.0000% | yes | 100% | −0.255% | 2.99% | 40.7% |

Hybrid 2.0 ms a world (26 network evaluations), 70–76 µs per evaluation. Result: FAIL (bias, and two
bands). With 40 decisions the bounds are wide; these are pipeline checks, not verdicts.

**What the numbers say.** The loss comes from a few large value errors (99.9th percentile 30–50%)
in positions where a game or the match is decided in the last tricks; the bias is negative. The
first version of the encoder (without the successor equities) was worse even at trick 5, where
the play is trivial, which showed the network was struggling to learn the match-equity table, not
the cards; giving it the successor equities cut the trick-5 error SD from 2.7% to 1.9% with the
same network (1.1% with a deeper one). Both networks were still improving when stopped.

## Rung 1b: the outcome head (spec §6a)

**What changed.** The network no longer guesses four money chances. It says how the hand ends,
and the money chances of each ending are computed exactly from the match-equity table. This works
because in an all-cards-face-up position the exact value is always the value of one ending of the
hand. From the start of trick k an ending is either "team t wins the game" (4 of them; the value
depends only on t) or "the hand is over" with a given split of the remaining tricks between the
four teams (the bonus still to come goes to a team the position already fixes: the holder of a
live 5 of trumps, or the end-of-hand bonus team). That gives 8, 14, 24 and 39 possible endings at
tricks 5, 4, 3 and 2 (`crates/nn/src/outcome.rs`, with the proof). Endings no line can reach are
masked, exactly: a team that cannot reach 41, a split where someone would reach 41 or win all five
tricks, or a split where the team holding the highest live trump wins no trick. The network gives
a score per ending; the value is the softmax-weighted mean of the endings' exact money vectors, in
whole numbers (u128 sums, largest-remainder rounding, exactly 2^32). The encoder inputs are
unchanged (`oracle-v2`).

**Completeness checked three ways.** Every label row needs an ending whose money vector equals the
exact value, or `search labels` stops. A test solves 4,300 random positions at tricks 3–5 under
both tie orders (8,600 solves) and finds one every time, asking the table only for valid states.
The Python trainer checks every row of every shard before training. All 1.24 million labels of the
small run below passed.

**Small run, same seeds and sizes as rung 1** (O_5: 300,000 labels; O_4: 900,000; 20,000 for
validation each; MLP 512-256-64; this VM):

| net | outcome accuracy | mean \|error\| | SD | 99.9% \|e\| | bias by team (leader's, next, opposite, previous) | training |
|---|---|---|---|---|---|---|
| O_5 (8 endings) | 99.96% | 0.016% | 0.80% | 1.0% | −0.006%, −0.002%, −0.001%, +0.009% | 1.8 min, best epoch 5 |
| O_4 (14 endings) | 90.1% | 0.53% | 2.31% | 29% | +0.078%, −0.025%, −0.010%, −0.044% | 4.2 min, best epoch 5 |

(Rung 1, four-logit head: O_5 mean |error| 0.51%, SD 1.1%; O_4 1.35%, 3.0%.) The integer networks
match the float ones to 0.0001%. O_4 started to overfit after epoch 5 (training loss still
falling, validation loss rising): it needs more labels.

**Check 1: H_4 against exact, first 40 trick-3 decisions × 64 worlds** (seed 1, regenerated, exact
recomputed; net `5b26d0baa852…`). Value error now over all four teams' chances (the other seats
choose by their own team's chance inside the search); bias per team relative to the decider.

| group | dec | mean loss | 95% upper | ≤ 0.01%? | agree | bias: own / next / opposite / previous | SD | 99.9% \|e\| | max \|e\| |
|---|---|---|---|---|---|---|---|---|---|
| all | 40 | 0.0189% | 0.0399% | no | 82.5% | −0.024% / −0.008% / −0.009% / +0.041% | 2.08% | 25.3% | 75.1% |
| pos 1–2 | 16 | 0.0357% | 0.0882% | no | 68.8% | −0.081% / −0.023% / +0.007% / +0.098% | 1.84% | 20.3% | 50.5% |
| pos 3–4 | 8 | 0.0000% | 0.0000% | yes | 100% | +0.149% / +0.110% / −0.294% / +0.035% | 3.51% | 60.0% | 75.1% |
| pos 5–8 | 16 | 0.0117% | 0.0294% | no | 87.5% | −0.043% / −0.044% / +0.102% / −0.015% | 1.33% | 17.2% | 39.3% |

Decider's team alone: SD 1.99%, 99.9% |e| 25.5%. Time: exact 3.4 ms a world, hybrid 17.5 ms a
world (229 network evaluations), 75 µs per evaluation (the outcome table adds about 5–15 µs to the
four-logit head). Result: FAIL (loss and bias), value SD down from 3.05% to 2.08%.

**Check 2: H_5 against exact, trick-4 golden decisions 0–39 × 64 worlds** (stored exact values,
`--source rejection`, 145 worlds recomputed byte-identical; net `d740d066ea63…`):

| group | dec | mean loss | 95% upper | ≤ 0.01%? | agree | bias: own / next / opposite / previous | SD | 99.9% \|e\| | max \|e\| |
|---|---|---|---|---|---|---|---|---|---|
| all | 40 | 0.0000% | 0.0000% | yes | 100% | −0.000% / +0.019% / +0.004% / −0.023% | 0.42% | 2.3% | 28.2% |
| pos 1–2 | 16 | 0.0000% | 0.0000% | yes | 100% | +0.000% / −0.000% / +0.000% / +0.000% | 0.003% | 0.009% | 0.21% |
| pos 3–4 | 8 | 0.0000% | 0.0000% | yes | 100% | +0.000% / −0.000% / +0.000% / −0.000% | 0.0004% | 0.002% | 0.017% |
| pos 5–8 | 16 | 0.0000% | 0.0000% | yes | 100% | −0.000% / +0.048% / +0.010% / −0.057% | 0.66% | 10.2% | 28.2% |

Decider's team alone: SD 0.004%. The decision loss passes everywhere; the bias of two other teams
(+0.019%, −0.023%) fails, all of it from one decision at position 5 where an ending is misclassified.
Result: FAIL on bias only (rung 1: loss 0.0037%, bias −0.11%, SD 2.0%).

**What this says.** The outcome head fixes what it was meant to fix: once the cards decide
the ending (trick 5), the values are nearly exact. At trick 4 the remaining error is the network
not yet predicting who wins the last two tricks (90% accuracy), and it is data-limited. The next
round (spec §3): many more labels (they are cheap), a larger network, harvested frontier positions,
then the full trick-3 check.

## Full rung-1 run (off this VM; nothing rented)

(2026-10-16, later: the same run on one rented machine with a GPU, as the runner's `hybrid_rung`
chunk: `docs/m3-hybrid/vast.md` and `docs/vast/m3-hybrid-rung1b.manifest.json`.)

A ready-to-run GitHub Actions job is in `.github/workflows/hybrid-rung1.yml` (manual trigger
only; not run). One standard CPU runner (4 vCPU, 16 GB): equity table, 1.05M labels (~2 min),
the trick-3 golden set (~2–3 min), training capped at 200 minutes (1024-512-128, outcome head),
the check (~10–30 min), outputs kept as an artifact (networks, reports, sidecars, the new trick-3
golden decisions and exact worlds, the check). About 4 hours of wall time at most, inside the
6-hour limit. The same steps by hand (`T` = threads):

```
cargo build --release -p search -p nn -p equity
equity build --model data/equity/model-rules-v1-1M-seed1.json --out equity-v0.npy   # SHA-256 64740dab…

# 1. labels, ~1M trick-4 positions (~0.3 ms per row per core: ~0.1 core-hours; ~1.1 GB)
search labels --trick 4 --count 800000 --seed 101 --source random --out t4-random-s101.npy --equity equity-v0.npy --threads T
search labels --trick 4 --count 200000 --seed 102 --source rules  --out t4-rules-s102.npy  --equity equity-v0.npy --threads T
search labels --trick 4 --count 50000  --seed 201 --source random --out t4-val-s201.npy    --equity equity-v0.npy --threads T

# 2. training, outcome head (~25 s per epoch per 1M rows for 512-256-64 on 4 CPU threads;
#    1024-512-128 about 4x: 2-14 core-hours on CPU, minutes on a GPU)
cd python/nn && uv sync && uv run python -m fortyone_nn.train --head outcome \
  --train t4-random-s101.npy t4-rules-s102.npy --val t4-val-s201.npy --out o4-full \
  --widths 1024,512,128 --epochs 60 --batch 1024 --lr 1e-3 --patience 5 --seed 1 --threads T

# 3. the trick-3 golden set: 400 decisions x 320 worlds, seed 1 (128,000 exact worlds at ~3.5 ms:
#    ~0.13 core-hours), stored in the yardstick's worlds format
search decisions --trick 3 --count 400 --seed 1 --allow-uneven --out decisions-t3.jsonl
search worlds --decisions decisions-t3.jsonl --method exact --to 320 --out exact-t3-n320.jsonl --equity equity-v0.npy --threads T

# 4. the check (hybrid ~17 ms a world with 512-256-64, ~40-50 ms with 1024-512-128:
#    0.6-1.8 core-hours; 1% of exact worlds recomputed)
search hybrid-check --trick 3 --cut 4 --net o4-full/int.safetensors --equity equity-v0.npy \
  --decisions decisions-t3.jsonl --worlds exact-t3-n320.jsonl --to 320 --threads T --json hc-t3-o4-full.json
```

Total about 3–16 core-hours, nearly all of it training.

## Spec questions and the answers (review of 2026-10-16)

1. **Successor equities in the encoder** (`oracle-v2`): approved.
2. **The evaluator sees a canonical form**: approved.
3. **No suit symmetries** (the card-number tie rule breaks them): confirmed.
4. **The trick-4 golden set was drawn by the rejection sampler** (binary `bc4835a`); keep
   `--source rejection` for it. The new trick-3 set uses the current default (belief sampler).
5. **"95% upper bound"** = 97.5th percentile of 1,000 bootstrap means (seed 1), as `search
   curves`: accepted.
6. **Value error for all four teams**: done (rung 1b tables): bias per team relative to the
   decider, SD and tails over all four teams' chances; the decider's own SD is printed too.
7. **Synthetic robs in later tricks**: the solver tests now keep a synthetic rob only where the
   burn merge is proved sound (`solver/tests/common::burn_merge_sound`: no trump played in an
   earlier trick that is still the best lies between two trumps the robber could burn; noted in
   the test files). Among 154 positions where the proof does not hold, no wrong value was found.
8. **Hybrid values version** is written as `2+O4:<net sha256>` in the check's JSON.

## Open questions

- **Speed for O_3.** About 75 µs per evaluation is far too slow for O_3 (an exact trick-3 solve is
  ~4–9 ms; H_3 from trick 2 will make tens of thousands of evaluations a world). The forward pass is
  plain 64-bit arithmetic; the plan's path (32-bit SIMD kernel with runtime dispatch, smaller first
  layer, incremental first layer) should give 10x or more. The outcome table (up to 39 equity
  lookups at trick 2) should be cached per score state.
  **Done (branch `m3-hybrid-nnspeed`, `docs/m3-hybrid/nn-speed.md`):** byte-identical answers,
  one evaluation 9–12 times faster for 512-256-64 (6.4–6.8 µs for the trained O_4), about 7 times
  for 1024-512-128 with random weights; H_4 0.94 ms a world instead of 10.1 ms. More for
  1024-wide networks needs a new integer format (a decision for the owner, see that note).
- **Accuracy.** O_4 overfits 900k labels; labels are cheap (1M in about a minute here), so the next
  round should use several million, a larger network, and harvested frontier positions from H_4's
  own searches (`SOURCE_HARVEST` is reserved in the shard format). **Harvesting is built** (branch
  `m3-hybrid-harvest`, next section).
- The CI parity step and the rung-1 workflow have not been run (no workflows triggered from here).

## Harvested positions (spec §5 source (c); branch `m3-hybrid-harvest`, 2026-10-16)

**Why.** Two trainings of the rung-1b recipe (4M labels from random play and rules-v1 self-play)
came out at the edge of the standard on the trick-3 golden set (decision loss 95% upper bound
0.0083% and 0.0108%; positions 1–2 and 3–4 failing; a team bias of 0.013–0.017%). The hybrid H_4
asks the network about *every* position its search reaches at the start of trick 4, including
lines no sensible player would choose (the search must look at every card to know it is bad).
Random play and self-play give one position per deal; the search asks about hundreds of
neighbouring positions of the same deal, a different mix. Harvesting [collecting training
positions from the search itself] trains the network on exactly what it will be asked. It is the
proven remedy (DAgger [train on the situations your own method gets into], Ross et al. 2011).

**How** (`search labels --source harvest`, `search::oracle_labels::harvest_rows`):

1. A training decision drawn exactly as the yardstick draws its decisions (the generator of
   `search decisions`: a rules-v1 hand from a match state of each stratum S1–S4 in turn), at
   trick 3 by default (`--harvest-tricks`), with its own seed (seed 1, the golden sets', is
   refused).
2. One world drawn from the decider's belief, as for the random-play labels.
3. The hybrid's search from the decision, for every candidate, with the *exact* value at the
   frontier (the search H_4 makes when its network is right). Every position it reaches at the
   start of trick 4 is recorded through the solver's frontier hook, in the order reached.
4. A position met again (another line to the same canonical form) is kept once; then a seeded
   sample of at most 16 per decision (`--harvest-cap`), so the decisions that reach hundreds of
   positions do not dominate.
5. Each is labelled exactly like the other sources (same exact value, tie rule, outcome label and
   record format; source tag 3 in the shard).
6. `--harvest-positions 0.4,0.4,0.2` sets the shares of decisions for the decider's position
   bands 1–2, 3–4 and 5–8 (default 0.25,0.25,0.5: the eight positions equally). The sidecar
   records the settings and the counts reached: decisions and rows per position, per stratum and
   per trick, and the positions reached per decision.

The same bytes come out for any number of threads (tested on 1 and 4 threads; also checked on a
20,000-row shard).

**Measured on the build VM** (trick 3 decisions, frontier at trick 4):

| | |
|---|---|
| positions reached per decision (distinct) | mean 112, median 24, 90th percentile 324, max 2,184 (138 frontier evaluations on average) |
| by the decider's position 1 … 8 (400 decisions) | 502, 202, 96, 41, 28, 9, 5, 2 |
| share of all reachable positions that the search reaches | 95% (45,872 of 48,321 over 400 decisions; all of them in 385 decisions) |
| positions the trained rung-1b network's H_4 reaches that the harvest also reaches | 99.2% |
| time (one thread) | 0.33 s per 1,000 labels (positions uniform), 0.41 s with positions 1–4 oversampled (0.4,0.4,0.2); random-play labels 0.23 s, rules-v1 0.07 s; 4M harvested labels about 30 thread-minutes (seconds on a rented machine) |
| rows by position with 0.4,0.4,0.2 (50,000 rows) | 89% from positions 1–4 (a late player has few cards left to choose from, so its search reaches few positions) |
| overlap with the other sources | 0 of 20,000 harvested positions are among the first 100,000 random-play labels of the training seed (101), or among 100,000 rules-v1 labels |

The overlap is nil because a deal of 53 cards never repeats: no exact position recurs between
sources. What the harvest adds is the search's own neighbourhood of each deal (the third and
fourth lines above show that it is that neighbourhood, whatever values drive the search).

**Training** reads the harvested shard like the others (`--train random.npy rules.npy
harvest.npy`; the trainer ignores the source tag). The validation shard stays random play and the
verdict stays the golden set. The runner's `hybrid_rung` takes it with `labels_harvest` (rows),
`labels_harvest_seed`, `harvest_cap_per_decision` and `harvest_positions` (`docs/m3-hybrid/vast.md`).

**For O_3 later (not built).** With exact values at the frontier every reached position is solved
exactly: at trick 3 that is about 830 trick-3 positions per trick-2 decision (leader), 40 s on one
thread, about 2.5 s per label: too dear for millions. The O_3 harvest should drive its search with
the first-round network (or any fixed value) and solve only the sampled positions; at trick 4 the
reached sets are nearly the same either way (lines above).
**Built** (branch `m3-hybrid-t3`, `docs/m3-hybrid/rung2.md`): `search labels --harvest-net`, about
10 thread-seconds per 1,000 trick-3 labels; the network-driven search reaches 99.94% of what the
exact-driven one reaches at trick 3 too.
