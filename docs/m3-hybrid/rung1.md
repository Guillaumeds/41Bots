# Hybrid yardstick, rung 1 (O_4): build report

Branch `m3-hybrid`, 2026-10-16 (Mauritius). Spec: `docs/M3-HYBRID-SPEC.md`.

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

## Full rung-1 run (off this VM; nothing rented)

Binaries: `cargo build --release -p search -p nn -p equity`. Table: `equity build --model
data/equity/model-rules-v1-1M-seed1.json --out equity-v0.npy` (SHA-256 `64740dab…`). `T` = threads.

```
# 1. labels, about 1M trick-4 positions (~0.27 ms per row per core: ~0.08 core-hours; 626 MB)
search labels --trick 4 --count 800000 --seed 101 --source random --out t4-random-s101.npy --equity equity-v0.npy --threads T
search labels --trick 4 --count 200000 --seed 102 --source rules  --out t4-rules-s102.npy  --equity equity-v0.npy --threads T
search labels --trick 4 --count 50000  --seed 201 --source random --out t4-val-s201.npy    --equity equity-v0.npy --threads T

# 2. training (~25 s per epoch per 1M rows for 512-256-64 on 4 CPU threads; 1024-512-128 about 4x:
#    40-60 epochs = 1.5-6 core-hours on CPU, minutes on a GPU)
cd python/nn && uv sync && uv run python -m fortyone_nn.train \
  --train t4-random-s101.npy t4-rules-s102.npy --val t4-val-s201.npy --out o4-full \
  --widths 1024,512,128 --epochs 60 --batch 1024 --lr 1e-3 --patience 5 --seed 1 --threads T

# 3. the trick-3 golden set: 400 decisions x 320 worlds, seed 1 (128,000 exact worlds at ~3.8 ms:
#    ~0.14 core-hours), stored in the yardstick's worlds format
search decisions --trick 3 --count 400 --seed 1 --allow-uneven --out decisions-t3.jsonl
search worlds --decisions decisions-t3.jsonl --method exact --to 320 --out exact-t3-n320.jsonl --equity equity-v0.npy --threads T

# 4. the check (hybrid ~15 ms a world with a 512-256-64 net, ~40 ms with 1024-512-128:
#    0.5-1.5 core-hours; 1% of exact worlds recomputed)
search hybrid-check --trick 3 --cut 4 --net o4-full/int.safetensors --equity equity-v0.npy \
  --decisions decisions-t3.jsonl --worlds exact-t3-n320.jsonl --to 320 --threads T --json hc-t3-o4-full.json
```

Total about 2–8 core-hours depending on the network size and where training runs: under 10
minutes of wall time on a 32-thread machine for steps 1, 3 and 4.

## Spec questions (what I decided, for review)

1. **Successor equities in the encoder.** Spec §6 lists the fields; I added, as number inputs, the
   match-equity values after each possible winner of the next one and two tricks (PLAN §6.5 lists
   "the best and worst reachable successor equities" as optional inputs; this gives the network all
   of them for the next two tricks). They are pure functions of fields the encoder already holds,
   so completeness is unchanged. Encoder version `oracle-v2`. Please confirm.
2. **The evaluator sees a canonical form, not the raw position** (see "What was built", 1). Without
   it the fast solver's equivalent-card merging would not stay exact once a network sits at the
   frontier. The form is complete (equal forms ⇒ equal exact values: proved and tested).
3. **Suit symmetries are not used.** Swapping clubs and spades (PLAN A8) changes the card numbers,
   and the solver's tie rule breaks ties by card number, so exact values can change: it is not an
   exact symmetry of the current solver. Only the team rotation is used (tested).
4. **The trick-4 golden set was drawn by the rejection sampler.** `m3-agree-t4c` was made by binary
   `bc4835a`, before the belief sampler became the default; its worlds are reproduced only with
   `--source rejection` (then byte-identical). The trick-2 set has world counts, so it used the
   belief sampler. The new trick-3 set will use the current default.
5. **"95% upper bound"** is taken as the upper end of the 95% bootstrap interval, the 97.5th
   percentile of 1,000 bootstrap means (decisions resampled with replacement, seed 1, each band on
   its own), as `search curves` does. It is the stricter reading (a one-sided bound would use the
   95th percentile).
6. **Value error** is reported in the decider's team's chance (as the decision loss is).
7. **Hybrid values version** is written as `2+O4:<net sha256>` in the check's JSON.

## Open questions

- **Speed for O_3.** About 65 µs per evaluation is far too slow for O_3 (an exact trick-3 solve is
  ~4–9 ms; H_3 from trick 2 will make tens of thousands of evaluations a world). The forward pass is
  plain 64-bit arithmetic; the plan's path (32-bit SIMD kernel with runtime dispatch, smaller first
  layer, incremental first layer) should give 10x or more.
- **Accuracy.** Next round per spec §3: more labels (they are cheap: 1M in about a minute here),
  a larger and longer-trained network (GPU), and harvested frontier positions from H_4's own
  searches (`SOURCE_HARVEST` is reserved in the shard format).
- **For the solver owner (not a real-game issue):** the burn-merge proof assumes the rob happens in
  trick 1 (no played card can lie between two burn candidates). Some solver tests put synthetic
  robs in tricks 3–5, where a played trump could lie between two merged burns and change the
  end-of-hand bonus.
- The new CI step (Python parity, `astral-sh/setup-uv@v6`) has not been run (no workflows were
  triggered from here).
