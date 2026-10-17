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

## Results, 2026-10-16

**What was run.** Ten trainings of the trick-4 oracle network O_4 [the network that values a
position at the start of trick 4], each checked as the hybrid H_4 [exact search down to the start
of trick 4, the network from there on] against exact search on the same golden set [a fixed test
set]: the 400 trick-3 decisions of seed 1, each valued on 320 worlds [guessed deals of the hidden
cards], 1.4 million values in all. All but the first ran on one rented vast.ai machine each,
through the runner's `hybrid_rung` chunk (`docs/m3-hybrid/vast.md`); each of those checks
recomputed 1% of the stored exact worlds and found them byte-identical (1,284 worlds; 11 in the
smoke run). Each run's reports (the check's text and JSON, the training curve, the step timings,
the machine) are in its R2 bundle, `runs/<run_id>/out/rung/`.

**How to read the tables.**

- *Seat* is the decider's place in the order of play in the trick: seat 1 leads, seat 8 plays
  last (the program's tables call it `pos`). Seats 1-2, 3-4 and 5-8 are the three bands.
- *Decision loss* [the money chance given away per decision by playing the hybrid's card instead of
  exact search's] is shown as its 95% upper bound [the top of a 95% confidence range, found by
  resampling the 400 decisions 1,000 times], in percent of money chance. **The standard: at most
  0.0100% overall and in every band, and every team's bias at most 0.0100%** [bias: whether the
  network's values lean one way, the average error in each team's chance; the largest of the four
  teams' is shown]. Bold marks a number over the line. Seats 1 and 4 are not part of the standard;
  they are shown because they are the hardest seats (52 decisions each, so their bounds are coarse).
- *Agreement* is how often the hybrid and exact search choose the same card.
- *Network widths* [units in each layer]: 1024-512-128 is three layers of 1024, 512 and 128 units.
- *Labels* [training positions with their exact values]: random play, self-play (rules-v1 bots
  against each other), and harvested [positions the hybrid's own search reaches, labelled exactly;
  see "Harvested positions" above]. *Cap* is the most harvested positions kept per decision; the
  three percentages are the shares of harvested decisions whose decider sits at seats 1-2 / 3-4 /
  5-8. Every run also had 100,000 validation labels (seed 201) except the smoke run (20,000).
- *Epochs* [passes over all the training labels] are counted from 0 as the trainer reports them,
  so "best 8" is the ninth pass; the trainer keeps the best one. The graphics card (GPU) does the
  training.
- *Cost*: training minutes times the machine's hourly price from the launcher's record, "about";
  the *whole rental* (boot, uploads, tear-down included) is the launcher's estimate for the full
  rental at the same price.

**Table 1: what each run was** (in time order).

| run | started (Mauritius time) | labels: random play + self-play + harvested | network widths | machine | training | cost |
|---|---|---|---|---|---|---|
| GitHub run 38868893704 | before the rented runs | 3.2M + 0.8M + none (seeds 101, 102; the workflow's defaults) | 1024-512-128 | GitHub standard runner, CPU only (4 threads, 16 GB RAM), not rented | 52 min, on the CPU (epochs not kept here) | none (not rented) |
| smoke (`m3-hybrid-smoke-1`) | 13:48 | 0.4M + 0.1M + none; seeds 101, 102; 20,000 validation | 1024-512-128 | AMD EPYC 9654 96-core server chip (368 threads, 125 GB RAM, $1.00/h) [offer 53633715] + RTX A2000 graphics card (6 GB) | 1.6 min, 3 epochs (best 2) | about $0.03 (whole rental about $0.17) |
| 4M (`m3-hybrid-rung1b-1`) | 14:37 | 3.2M + 0.8M + none; seeds 101, 102 | 1024-512-128 | Intel Core i9-14900K desktop chip (30 threads, 63 GB RAM, $0.72/h) [offer 54905705] + RTX 4090 graphics card (24 GB) | 2.1 min, 14 epochs (best 8) | about $0.03 (whole rental about $0.10) |
| 16M, small network (`m3-hybrid-r1-d16m-small`) | 15:05 | 12.8M + 3.2M + none; seeds 121, 122 | 512-256-64 | AMD EPYC 7302P 16-core server chip (30 threads, 94 GB RAM, $0.51/h) [offer 47335235] + RTX 3090 graphics card (24 GB) | 41.1 min, 29 epochs (best 23) | about $0.35 (whole rental about $0.45) |
| 4M, training seed 2 (`m3-hybrid-r1-a4m-s2b`) | 15:24 | 3.2M + 0.8M + none; seeds 101, 102 | 1024-512-128 | AMD EPYC 9374F 32-core server chip (30 threads, 39 GB RAM, $1.68/h) [offer 55830933] + RTX 4090 graphics card (24 GB) | 2.4 min, 13 epochs (best 7) | about $0.07 (whole rental about $0.21) |
| harvest-2 (`m3-hybrid-harvest-2`) | 16:12 | 6.4M + 1.6M + 4M harvested (cap 16 per decision; decisions at seats 1-2 / 3-4 / 5-8: 40/40/20%); seeds 101, 102, 103 | 1024-512-128 | AMD Ryzen Threadripper PRO 7995WX 96-core workstation chip (184 threads, 251 GB RAM, $1.14/h) [offer 55689381] + RTX 3090 graphics card (24 GB) | 6.7 min, 16 epochs (best 10) | about $0.13 (whole rental about $0.26) |
| 8M (`m3-hybrid-r1-b8m-2`) | 16:12 | 6.4M + 1.6M + none; seeds 111, 112 | 1024-512-128 | older Intel Xeon E5-2698 v4 server chip, 2.2 GHz (38 threads, 126 GB RAM, $0.39/h) [offer 53914105] + RTX 3090 graphics card (24 GB) | 5.2 min, 16 epochs (best 10) | about $0.03 (whole rental about $0.10) |
| 16M (`m3-hybrid-r1-c16m-2`) | 16:44 | 12.8M + 3.2M + none; seeds 121, 122 | 1024-512-128 | AMD Ryzen 9 5950X 16-core desktop chip (30 threads, 63 GB RAM, $0.28/h) [offer 52369530] + RTX 3090 Ti graphics card (24 GB) | 10.4 min, 20 epochs (best 14) | about $0.05 (whole rental about $0.24) |
| lead-1 (`m3-hybrid-harvest-lead-1`) | 17:03 | 12.8M + 3.2M + 8M harvested (cap 64 per decision; decisions at seats 1-2 / 3-4 / 5-8: 60/25/15%); seeds 131, 132, 133 | 1024-512-128 | AMD EPYC 9654 96-core server chip (46 threads, 118 GB RAM, $0.70/h) [offer 55224468] + RTX 6000 Ada graphics card (48 GB) | 9.8 min, 20 epochs (best 14) | about $0.11 (whole rental about $0.21) |
| lead-2 (`m3-hybrid-harvest-lead-2`) | about 17:30 | 12.8M + 3.2M + 12M harvested (cap 64 per decision; decisions at seats 1-2 / 3-4 / 5-8: 45/40/15%); seeds 131, 132, 134 | 1024-512-128 | AMD EPYC 9654 96-core server chip (46 threads, $0.75/h; RAM not recorded, lead-1 on the same offer had 118 GB) [offer 55224468] + RTX 6000 Ada graphics card (48 GB) | 12.5 min, 22 epochs (best 16) | about $0.16 (whole rental about $0.27) |

**Table 2: what the check found** (95% upper bounds of decision loss in percent of money chance;
bold = over the standard).

| run | overall | seats 1-2 | seats 3-4 | seats 5-8 | largest team \|bias\| (%) | agreement | seat 1 | seat 4 | standard met? |
|---|---|---|---|---|---|---|---|---|---|
| GitHub run 38868893704 | 0.0083 | **0.0113** | **0.0145** | 0.0073 | **0.0168** | 88.2% | n/a | n/a | no (1-2, 3-4, bias) |
| smoke (`m3-hybrid-smoke-1`) | **0.2961** | **0.8868** | **0.4127** | **0.0441** | **0.0907** | 75.0% | **1.6611** | **0.0222** | no (all; a 32 x 32 pipeline test, not a verdict) |
| 4M (`m3-hybrid-rung1b-1`) | **0.0108** | **0.0309** | **0.0174** | 0.0016 | **0.0130** | 87.5% | **0.0519** | **0.0198** | no (overall, 1-2, 3-4, bias) |
| 16M, small network (`m3-hybrid-r1-d16m-small`) | 0.0084 | **0.0224** | **0.0161** | 0.0001 | 0.0077 | 92.0% | **0.0384** | **0.0202** | no (1-2, 3-4) |
| 4M, training seed 2 (`m3-hybrid-r1-a4m-s2b`) | 0.0097 | **0.0247** | **0.0177** | 0.0014 | **0.0186** | 86.8% | **0.0453** | **0.0214** | no (1-2, 3-4, bias) |
| harvest-2 (`m3-hybrid-harvest-2`) | 0.0090 | **0.0210** | 0.0059 | 0.0098 | 0.0094 | 92.0% | **0.0408** | 0.0088 | no (1-2) |
| 8M (`m3-hybrid-r1-b8m-2`) | 0.0086 | **0.0281** | **0.0113** | 0.0008 | 0.0089 | 91.2% | **0.0481** | **0.0196** | no (1-2, 3-4) |
| 16M (`m3-hybrid-r1-c16m-2`) | 0.0076 | **0.0211** | **0.0115** | 0.0007 | 0.0091 | 92.2% | **0.0373** | **0.0204** | no (1-2, 3-4) |
| lead-1 (`m3-hybrid-harvest-lead-1`) | 0.0037 | 0.0069 | **0.0109** | 0.0002 | 0.0075 | 93.8% | **0.0124** | **0.0198** | no (3-4) |
| lead-2 (`m3-hybrid-harvest-lead-2`) | 0.0033 | 0.0046 | **0.0103** | 0.0002 | 0.0079 | 93.5% | 0.0028 | **0.0196** | no (3-4) |

Notes. The GitHub row is the run's summary as reported (no files kept here, so no seat figures).
"4M, training seed 2" has the same labels as "4M" and differs only in the training seed; "16M,
small network" has the same labels as "16M". The 16M run's whole rental includes two machines that
never reported within 15 minutes and were replaced (33 minutes, about $0.14). The smoke run
trained 3 epochs and was checked on 32 decisions x 32 worlds only: it tests the pipeline, not
the network.

**What the numbers say.**

1. No run meets the whole standard yet, but the latest, lead-2, is close: overall bound 0.0033%,
   every team's bias at most 0.0079%, and only the seats 3-4 band over the line (0.0103% against
   0.0100%).
2. More ordinary labels help, slowly: with the same recipe the overall bound was 0.0083-0.0108% at
   4M labels (three trainings on the same labels, so differences of up to about 0.0025% between
   single runs are noise), 0.0086% at 8M and 0.0076% at 16M, while seats 1-2 and 3-4 stayed over
   the line (0.0211% and 0.0115% at 16M), and the smaller network on the same 16M labels did a
   little worse (0.0084%).
3. Harvested positions go at that directly: adding 4M of them to 8M ordinary labels (harvest-2,
   cap 16) took seats 3-4 from 0.0113% (the 8M run) to 0.0059%, but left the overall bound where
   it was (0.0090% against 0.0086%) and seats 1-2 at 0.0210%, and seats 5-8 slipped to 0.0098%,
   96% of that band's loss coming from one seat-5 decision.
4. Weighting the harvest towards the leaders was the big step: lead-1 (8M harvested on top of 16M
   ordinary labels, cap 64, 60% of harvested decisions at seats 1-2) roughly halved the overall
   bound against the 16M run without harvest (0.0076% to 0.0037%) and brought seats 1-2 under the
   line (0.0211% to 0.0069%; seat 1 alone from 0.0373% to 0.0124%).
5. It changed the cap, the amount and the shares together, so the gain cannot be split between
   them; the cap was raised because a cap of 16 keeps only a sliver of a leader's positions (a
   seat-1 decision reaches about 500 positions at trick 4 on average, a seat-4 decision about 40).
6. Lead-2 (12M harvested, shares 45/40/15) took seats 1-2 to 0.0046% (seat 1 to 0.0028%) but seats
   3-4 only from 0.0109% to 0.0103%, and seat 4 is the last to pass: its bound is 0.0196-0.0214%
   in every network except harvest-2 (0.0088%).
7. That seat-4 loss is concentrated: in lead-2, two of its 52 decisions (numbers 143 and 335 of
   the golden set) are 91% of the seat's loss and 54% of the whole loss, and seven of the eight
   networks with full files miss each of them at exactly the same cost (0.2533% and 0.0972% of
   money chance a decision), which points to a systematic gap, not training luck.
8. The lead-2 training curve shows overfitting [the network fits its training positions ever more
   closely while doing worse on positions it has not seen]: the best epoch is 16, with training
   loss 1.1775 [error on the positions it learns from] and validation loss 1.2088 [error on a
   separate set it never learns from], and by epoch 21 the training loss had fallen to 1.1673 while
   the validation loss rose in each of the five epochs after 16 (1.2114, 1.2150, 1.2188, 1.2227,
   1.2246), so the trainer stopped and kept epoch 16 (the two losses are measured on different mixes,
   harvested plus random-play positions for training and random play only for validation, so only
   the direction matters).
9. That argues for more and different positions, not more passes over the same 28M.
10. Lead-3 is therefore running as the next step: 16M random-play + 4M self-play + 20M harvested
    labels (40M in all), cap 64, with 55% of the harvested decisions at seats 3-4 (30% at seats 1-2,
    15% at seats 5-8), to see whether the seats 3-4 band and seat 4 finally pass; its result will be
    added here.

## Why seats 3-4 fail: the race for the money (2026-10-16)

**In one paragraph.** The seats 3-4 failure is not a shortage of data and not a quirk of the golden
set. It comes from one kind of position: at the start of trick 4 a team that already has a game
can win the match in this hand while another team can still win the game. There every other team
prefers the other winner (PLAN §5.3, "public enemy number one" and kingmaking), and who actually
gets through depends on whether the others hold the cards to steer each of the last two tricks.
The network is 7 to 9 times less accurate in these positions than elsewhere, on fresh training-type
positions as much as on the golden set's. They are 7% of the positions the hybrid asks the network
about, yet carry 75% of all the decision loss and 86% of the seats 3-4 loss of lead-3c; with exact
values there and the network everywhere else, every band passes with room to spare.

**How it was found.** `search hybrid-check … --explain IDS` (commit `5d69cd4`,
`crates/search/src/hybrid_explain.rs`) re-runs chosen golden decisions on the check's own worlds and
shows, for every world and candidate, the frontier position [the start-of-trick-4 position where the
search hands over to the network] whose value each search copied up to the root, the network's
error there, and the exact values under the reversed tie rule. Run on every lossy decision of
lead-2 and lead-3c (320 worlds each) and on golden decisions 0-99 as the typical case.

**Race classes** [what can be decided in the hand, from the outcome head's own exact mask:
`nn::outcome::race_class`]:

| class | meaning |
|---|---|
| 0 | no team can win the game in this hand |
| 1 | one team can win the game (reach 41 or make a jink), and it would not win the match |
| 2 | two or more teams can win the game, none of them would win the match (a race for the game) |
| 3 | one team can win the match in this hand and nobody else the game (public enemy number one) |
| 4 | a team can win the match in this hand and another team can win the game (a race for the money) |

**1. The loss is in class 4.** Share of the decision loss by the race class of the positions on the
two choices' principal lines:

| network | all decisions: class 4 | seats 3-4: class 4 | tie-rule worlds (all / seats 3-4) |
|---|---|---|---|
| lead-3c | 75% | 86% | 7% / 12% |
| lead-2 | 58% | 83% | 13% / 20% |

**2. The network is weak there everywhere.** Error of the network's money chances (all four teams)
at the frontier positions the hybrid reached in golden decisions 0-99 (lead-3c, 18.6M values), and
on a fresh harvest shard never trained on (seed 9144, lead-3c's harvest settings, 300,000 rows):

| class | share of frontier positions | SD [typical spread], golden frontier | 99.9th percentile, golden frontier | SD, fresh shard (lead-3c / lead-2) |
|---|---|---|---|---|
| 0 | 54% | 0.33% | 4.8% | 0.32% / 0.33% |
| 1 | 27% | 0.28% | 4.5% | 0.39% / 0.43% |
| 2 | 9% | 1.65% | 29.7% | 1.27% / 1.28% |
| 3 | 3% | 0.62% | 10.0% | 0.62% / 0.66% |
| 4 | 7% | 2.46% | 31.1% | 2.74% / 3.04% |
| all | | 0.88% | 11.4% | 0.87% / 0.95% |

On the fresh shard 1.05% of the class-4 values are off by more than 10 points (lead-2: 1.12%).
Lead-3c's 8M more harvested rows improved class 4 by only about 10%.

**3. Not for lack of such positions.** A census of the first 28,640 of lead-3c's 457,562 harvest
decisions (1.25M rows, 6.25%): 7.0% of its rows are class 4 (about 1.40M of the 20M), 8.0% class 2
(about 1.59M); 30% of the rows from stratum-4 decisions are class 4, while strata 1 and 3 give
almost none. The cap of 64 keeps only 38% of the class-4 positions the harvest's own searches reach.
The golden frontier has the same mix (class 4: 7.3%). Rows with exactly decision 143's score and
games state: about 240 in the 20M; decision 139's: about 800.

**4. The decisions** (money chance of the decider's team, 320 worlds):

| decision | situation | candidates: exact mean | lead-3c hybrid | lead-2 hybrid | loss |
|---|---|---|---|---|---|
| 143 (seat 4) | hearts trumps; team 1 has a game and stands on 26 with the Joker in hand, team 2 on 36 | 7S 52.6414%, **JK 52.8948%** | 7S 53.1494%, JK 52.7019% (picks 7S) | 7S 52.9542%, JK 52.6749% (7S) | 0.2533% (both) |
| 335 (seat 4) | clubs trumps; teams 0 and 1 have a game (36 and 40), team 2 on 36 | AD 41.4265%, 7D 41.4265%, **JD 41.5237%** | 40.6729%, 40.6815%, 40.4794% (7D) | 40.8114%, 40.8410%, 40.8231% (7D) | 0.0972% (both) |
| 139 (seat 3) | diamonds trumps; teams 2 and 3 have a game, team 1 (no game) on 36 | 10D 18.3305%, **6H 18.6574%** | 10D 18.6134%, 6H 18.5905% (10D) | 10D 18.6054%, 6H 18.7232% (6H) | 0.3269% (lead-3c) |

- 143: many worlds; 95 push towards 7S, 77 against, the top 10 carry 93% of the misranking (lead-2:
  132%), 5 of the top 10 are the same for both networks. In world 270, after 7S the partner wins
  trick 3 and team 1 leads on 31: the network gives team 1 the game 93.8%; exact search has team 2
  win it (team 1's chance 40.4%), because QS beats AS in a black plain suit (rule 10) and teams 0
  and 3 throw low rather than let team 1 win the match. The network's error in 143's class-4
  positions: SD 3.1%, and team 1's entries are biased +0.62% (lead-2: +0.81%): it overrates the
  public enemy, i.e. it underrates the blocking.
- 139: the top 10 worlds carry 105%, all class 4; for example the network gives team 1 the game
  98.7% where teams 0 and 3, holding the 5 and the ace of trumps, shut it out.
- 335: one world (216) carries 104% of the misranking (lead-2: 93%); see 5.

**5. Decision 335 is decided by the tie rule.** In world 216, after JD, team 3 gets nothing
whoever wins (team 0 or team 1 would win the match), so seat 3 is indifferent between its cards.
The solver's documented tie rule [first strictly best card by card index] makes seat 3 throw 10D
under seat 7's lead of the 7C, so seat 0's 7D takes the last trick: team 0 wins. With the reverse
order seat 3 keeps 10D, team 3 takes both tricks and team 1 wins at the end of the hand with its
first best-trump bonus (the Joker in trick 1: 40 + 6 = 46, rules 28, 30, 32). The network says
team 1 wins 98.4%. Under the reverse tie order exact search picks AD and the hybrid's 7D loses
nothing. Such tie-decided positions are 0.015% of the frontier, and the network's error there has
an SD of 59%. They need no fix for the standard (below).

**6. Ceiling.** The full check (400 decisions x 320 worlds, lead-3c) with exact values at the
frontier positions of some classes and the network everywhere else (95% upper bounds):

| exact at | overall | seats 1-2 | seats 3-4 | seats 5-8 |
|---|---|---|---|---|
| nothing (as checked) | 0.0057% | 0.0097% | **0.0159%** | 0.0007% |
| class 4 | 0.0016% | 0.0041% | 0.0023% | 0.0007% |
| classes 2 and 4 | 0.0015% | 0.0041% | 0.0015% | 0.0007% |
| classes 1-4 and tie-decided positions | 0.0005% | 0.0002% | 0.0013% | 0.0007% |

**7. The statistic.** The bootstrap's upper bound counts a single outlier about three times: in the
104-decision band, 143 alone contributes 0.0073% of the 0.0100% budget, and any single miss of
about 0.35% or more fails the band by itself (139 alone: 0.0094%). With 143 and 335 fixed lead-2's
seats 3-4 bound would be 0.0018% (only 143: 0.0039%); lead-3c's would be 0.0108% (139 still
missed), 0.0022% with 139 fixed too, and 0.0043% with 139 and 143 fixed but 335 still missed.

**What follows: the race harvest** (branch `m3-hybrid-race`). `search labels --source harvest`
gains `--harvest-strata` (the decisions' strata: decision `j` from stratum `strata[j mod s]`, its
decider at `schedule[(j div s) mod cycle]`, so rows stay a pure function of (seed, `j`); all four
strata is the old harvest, byte for byte) and `--harvest-race-classes 2,4` (every reached position
of those classes kept up to `--harvest-race-cap`, 256 a decision, the others up to `--harvest-cap`,
which may then be 0), with either driver; every shard's sidecar counts its rows by race class. The
runner takes it as a fourth source (`labels_race`, `docs/m3-hybrid/vast.md`), and every training
report now gives the validation error by race class. Measured on this VM: 1M race rows at trick 4
(stratum 4, classes 2 and 4, no other classes, seat shares 0.3/0.55/0.15) in 27 s on 4 threads,
0.08 thread-seconds per 1,000 rows (23,111 decisions, 58 race positions reached per decision);
52.2% class 4 and 47.8% class 2. Proposal `docs/vast/m3-hybrid-lead-4.manifest.json`: lead-3c's
mix with half of the harvest replaced by race rows (16M random play s151, 4M rules-v1 s152, 10M
harvest s154 cap 64, 10M race s155), which takes the class-4 rows of the 40M from about 3.0M (7.5%;
random play and rules-v1 self-play are 8.0% and 8.2% class 4) to about 7.5M (19%), and class 2 from
about 3.2M to about 7.2M. Whether
three times the class-4 data is enough is not proven: the one comparison we have (lead-2 to lead-3c)
improved class 4 slowly. If it falls short, the next lever traceable to the plan is the per-team
"can clinch this hand" inputs of PLAN §6.5 (an encoder change), then a larger network.

## Network shape comparison (2026-10-16 evening; the start of the audit loop's ladder)

Same data for all three (the lead-4 mix: 40M trick-4 labels, random 16M, rules 4M, harvest 10M, race 10M; validation
100k), the stored trick-3 golden set (400 decisions x 320 worlds), outcome head, batch 1024, lr 0.001. Bounds are the
95% upper bound of the decision loss (% money); class-4 SD is the integer network's value error SD on race-class-4 rows
(a team can win the match and another the game) of the validation shard.

| run | shape | GPU | val loss | class-4 SD (int / float) | all | pos 1-2 | pos 3-4 | pos 5-8 | largest bias (all) | agree |
|---|---|---|---|---|---|---|---|---|---|---|
| lead-5c | 1024-1024-512-128 | A100 40GB | 1.19817 | 2.33% / 2.99% | 0.0072 | 0.0210 | 0.0109 | 0.0001 | 0.0083 | 95.2% |
| lead-6a | 2048-1024-256 | L4 | 1.20811 | 3.60% / 3.34% | 0.0074 | 0.0222 | 0.0104 | 0.0002 | 0.0097 | 93.8% |
| lead-6b | 2048-2048-1024-256 | RTX 3090 | 1.19772 | 3.29% / 3.02% | 0.0076 | 0.0239 | 0.0102 | 0.0003 | 0.0060 | 94.2% |


Choice (rule: class-4 SD, then the worst band bound, then validation loss; a tie goes to the cheaper network):
**1024-1024-512-128**. Width alone does not lower the race-class error on this data (the float class-4 SD is 3.0% for
all three), and the two wide networks early-stopped at epochs 9 and 11. The lever is the audit loop's targeted data;
the ladder keeps the wider shapes for when the data has grown: `<widths>;2048,2048,1024,256;4096,2048,1024,256`.

## Adopted O_4 (owner, 2026-10-17)
The overnight trick-4 audit loop's round 1 is the current O_4: 1024-1024-512-128, outcome head,
`inputs/sha256/b28750e3d447400523d2af9cbe85e4fbbb414200474560a7c1038b7adf0ce5a0`, trained on the lead-4 label mix
(40M trick-4 labels). On 4,000 trick-3 golden decisions x 320 exact worlds: 95% upper bound 0.0016% overall, seats 1-2
0.0026%, 3-4 0.0023%, 5-8 0.0012%, largest bias 0.0025%; 13 auditor groups still fail at the decision level, most of
them race-for-the-money conditions (`audit-loop.md` §8). Later loops start from it (additive mode) and must beat it on
the same golden set.
