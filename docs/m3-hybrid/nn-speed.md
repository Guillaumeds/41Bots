# Faster network evaluations at the frontier (before rung O_3)

Branch `m3-hybrid-nnspeed` (from `m3-hybrid` at `eff6c4c`), 2026-10-16 (Mauritius). Answers
`docs/m3-hybrid/rung1.md`, "Open questions: Speed for O_3".

**In one paragraph.** The hybrid H_k asks the network O_k for the value of every position it
reaches at the start of trick k. One such evaluation now costs about a tenth of what it did for
the 512-256-64 network (9–12 times faster, 11 times for the trained O_4 of rung 1), and about a
seventh for 1024-512-128 with random weights. A new memo [a small table of recently valued
positions] answers repeated positions, so per request the gain is 9–17 times. H_4 now takes
0.94 ms a world instead of 10.1 ms, and H_3 (from trick 2) 17 ms instead of 175 ms. Every answer
is byte-identical to before: the old code is kept as the reference and compared with the new one
bit for bit in tests, by the Python parity check, and on whole `search hybrid-check` runs. The
10-times target is met for the 512-wide networks; for 1024-wide random networks it is not (about
7 times per evaluation), and getting there needs a change to the network's integer format, which
changes answers: a decision for the owner (end of this note).

## Before and after: one evaluation

Microseconds per evaluation on this VM (Xeon, 2.1 GHz, AVX-512, one thread), median of three runs
of the best of five passes over 5,000 real frontier positions in the order the solver asks for
them (`nn-tool bench`). The parts: **(a)** encoding [turning the position into the network's
inputs; for the fast path: the on/off inputs and a look-up of the equity inputs], **(b)** the
forward pass [the network's arithmetic], **(c)** the outcome table and the mixing of its money
vectors. "v3" is the AVX2 code [x86-64-v3: the vector instructions of the vast.ai EPYC machines],
"v4" the AVX-512 code [x86-64-v4: newer machines and this VM]; one binary picks the best the
machine has when it starts. Random networks have the trained network's number format and scale.

| trick | network | before: total = (a) + (b) + (c) | after v3: total (a / b / c) | × | after v4 | × | per request with the memo, v3 / v4 (share answered) |
|---|---|---|---|---|---|---|---|
| 4 | trained O_4 512-256-64 (rung 1) | 73.8 = 4.9 + 69.8 + 0.8 | 6.8 (0.2 / 6.1 / 0.3) | 10.9 | 6.4 | 11.5 | 5.6 / 5.5 (24%): ×13 |
| 4 | random 512-256-64 (14 endings) | 95.5 = 4.8 + 85.2 + 0.8 | 9.6 (0.2 / 9.7 / 0.3) | 9.9 | 8.0 | 12.0 | 7.6 / 6.3: ×13 / ×15 |
| 4 | random 1024-512-128 | 297.6 = 4.7 + 293.7 + 1.2 | 43.1 (0.3 / 38.8 / 0.3) | 6.9 | 40.4 | 7.4 | 33.1 / 31.1: ×9 / ×10 |
| 3 | random 512-256-64 (24 endings) | 93.1 = 5.0 + 92.6 + 1.2 | 10.4 (0.1 / 9.7 / 0.4) | 8.9 | 8.7 | 10.7 | 6.5 / 5.3 (42%): ×14 / ×17 |
| 3 | random 1024-512-128 | 308.7 = 4.8 + 307.8 + 1.2 | 43.4 (0.1 / 47.0 / 0.5) | 7.1 | 42.8 | 7.2 | 24.5 / 24.4: ×13 |
| 2 | random 512-256-64 (39 endings) | 94.1 = 4.6 + 88.6 + 1.6 | 10.8 (0.1 / 9.9 / 0.6) | 8.7 | 9.2 | 10.3 | 10.0 / 7.6 (9%): ×9 / ×12 |
| 2 | random 1024-512-128 | 300.9 = 4.5 + 312.2 + 1.5 | 44.3 (0.1 / 42.7 / 0.6) | 6.8 | 42.2 | 7.1 | 38.3 / 34.4: ×8 / ×9 |

The parts are timed in separate passes, so they do not add up exactly; runs on this VM vary by
about ±15%. On a machine without AVX2 the new code is still 2.5–4 times faster. The share of
requests the memo answers depends on the trick (repeats of a canonical form: 24% at trick 4 and
20–42% at trick 3; at trick 2 about 9–12%); in long runs it catches 97% or more of all repeats.

## Before and after: H_k per world (`search hybrid-check`)

Same decisions, worlds and settings before (binary of `eff6c4c`) and after:

| check | network | threads | before | after | × |
|---|---|---|---|---|---|
| H_4 on trick-3 decisions (32 decisions × 32 worlds, seed 1) | trained O_4 512-256-64 | 1 | 10.14 ms | 0.94 ms | 10.8 |
| same | same | 4 | 10.18 ms | 1.09 ms | 9.3 |
| same | random 512-256-64 | 4 | 13.2 ms | 1.59 ms | 8.3 |
| same | random 1024-512-128 | 4 | 48.8 ms | 5.34 ms | 9.1 |
| H_3 on trick-2 decisions (8 decisions × 4 worlds, seed 1; 1,578 frontier requests a world) | random O_3 512-256-64 | 4 | 174.7 ms | 17.5 ms | 10.0 |
| same | random O_3 1024-512-128 | 4 | 520 ms | 67.7 ms | 7.7 |

For comparison, the exact solve of those trick-2 worlds takes about 2.4 s each. With four
threads the per-request time is higher than with one (the threads share the memory caches).

## What changed (all in `crates/nn`, answer-identical engineering, PLAN §10)

1. **Fast forward pass** (`fast.rs`, `kernel.rs`). The plain forward pass (`IntMlp::forward`) is
   unchanged and remains the reference.
   - The first layer is split in two. Its **equity part** (the 100 number inputs, which depend
     only on the score and match state) is computed once per state and kept. Its **on/off part**
     (adding up the weight rows of the active inputs) is kept between evaluations and updated by
     only the rows that switched on or off since the previous position [an *incremental first
     layer*, as PLAN amendment A8 plans for the integer students]: about 22 rows instead of
     about 46–78.
   - Hidden activations are stored as 32-bit numbers (they never exceed 2^26), and weights in a
     "packed pair" layout [two weights per 64-bit word], so every product is one 32 × 32 → 64-bit
     multiply that the compiler turns into vector instructions [SIMD: many multiplications at
     once]. Only inputs that are not zero are read (as before). No memory is allocated per
     evaluation and there is no branch per unit.
   - The inner loops are compiled for x86-64-v2, v3 and v4 and the best is chosen when the program
     starts (`multiversion` 0.9.0, as PLAN §11.1 lists). Which loop order is used at each level was
     chosen by measurement; every level gives the same integers.
2. **Encoder and outcome table keyed by what they depend on.** `encode` is split into the on/off
   inputs and `dense_features(key)`, a function of a `DenseKey` alone (scores, flags, tricks,
   games, dealer, tricks done, the 5 of trumps, the end-bonus team, all in the encoder's rotated
   order); the outcome table likewise is `outcomes_of(key, top-trump team)`. Because these are
   functions of the key and nothing else, computing them once per key cannot change a value. A
   solve meets only a few dozen keys, so the 25 equity-table look-ups and 100 logarithms of the
   encoder, and the 4–39 look-ups of the outcome table, almost disappear: (a) went from 4.5–5 µs
   to 0.1–0.3 µs and (c) from 0.8–1.6 µs to 0.3–0.6 µs.
3. **Memo of recent positions** (`oracle.rs`). The solver's transposition table already values a
   repeated position once, but it is keyed by the cards themselves, and positions that differ
   only by equivalent cards (same canonical form, proven to have equal values in
   `solver/src/frontier.rs`) were each evaluated. The oracle now remembers the values of recent
   canonical forms (16,384 slots per thread, the whole key compared on a hit).
4. Smaller things: the integer exponential uses a single 64 × 64-bit multiply per step (the same
   integers; tested on 200,000 inputs against the old code, which is kept as
   `exp_weight_reference`), and the mixing step no longer allocates.

The caches live per thread and per loaded network, and are emptied when full; since each holds a
pure function of its full key, what they hold never affects a value.

## Overflow bounds

- The file check (unchanged) refuses a network unless, for every layer and output,
  |bias| + Σ |weight| × (largest input) ≤ 2^62 (largest input 2^24 for the first layer, 2^26
  after). Every partial sum the new code forms is a part of the plain sum's terms, so it stays
  within 2^62 whatever the order of additions: no 64-bit sum can overflow. Each product is at
  most 2^31 × 2^26 = 2^57; adding the rounding half (at most 2^61) before the shift stays below
  2^63.
- The on/off part of the first layer is summed in 32-bit arithmetic only when, for every output,
  the 96 largest |weights| (a position has at most 96 active inputs) add up to at most 2^31 − 1;
  then the true sum always fits, and wrapping arithmetic [addition that wraps around past the
  largest 32-bit number] gives it exactly, even when an update passes the limit on the way (a
  test drives updates across it). Otherwise the code sums in 64 bits. Both trained and random
  networks so far qualify.
- Activations fit 32 bits (0 to 2^26), and the equity inputs (−2^24 to 0) too.

## Proof of identity

- `crates/nn/tests/fast.rs`: the fast forward pass equals the plain one bit for bit at every
  instruction level the machine has, on random networks of several sizes (512-256-64 and
  1024-512-128 included, sparse and dense ones), on networks with weights and biases at the
  format's limits (weights of ±2^31, biases near 2^62, any shift, first layers that need the
  64-bit path), and on 4,500 real frontier positions at tricks 2, 3 and 4 in solver order and
  shuffled (about 190,000 comparisons on this VM); `Oracle::eval` (caches and memo) equals the plain
  `Oracle::eval_reference` for both output heads, memo on and off, several networks in one
  thread, and from three threads at once; the fast path reproduces the committed golden file.
- Kernel unit tests compare every loop at every level with plain loops (all widths, limits).
- The existing tests pass unchanged, among them the golden-output test and the hybrid identity
  test (naive, optimised and parallel solvers with the network at the frontier).
- `nn-tool eval` (used by the Python parity check) now evaluates with the fast path and stops if
  any row differs from the plain one; the Python tests pass (bit-identical to numpy).
- `search hybrid-check` writes byte-identical JSON before and after (timing fields and the
  binary's own hash aside) on all eight runs above, and for 1 and 4 threads.
- `cargo check -p nn --target wasm32-unknown-unknown` passes (the plain loops are used there).

## Limits, and what would need a decision

- **1024-512-128 is about 7 times faster per evaluation, not 10.** What remains is almost all
  the second layer: about 500 active inputs × 512 outputs = 260,000 exact 64-bit
  multiply-and-adds per evaluation, and its 2 MB of weights do not stay in a core's own cache.
  The compiler already produces the instruction sequence a hand-written kernel would use (checked
  in the disassembly), so intrinsics [hand-written processor instructions] would not help. A
  trained network is sparser (the trained O_4 has 28% of its first-layer units active against
  51% for random weights), so a trained 1024-wide network should do better; to be measured when
  the full run's network exists.
- **Going further changes answers** (proposals, not done):
  1. The integer scheme PLAN amendment A8 and §11.1 already name for later students: 16-bit
     first layer, 8-bit hidden weights, 32-bit sums (quantisation-aware training). Several times
     faster again (16–32 multiplications per instruction instead of 4–8), bit-identical
     everywhere, but a new weights format, retraining and a fresh validation of each rung.
  2. A narrower first hidden layer (for example 512 instead of 1024): the second layer's work
     falls in proportion; retraining needed, and accuracy decides (spec §6: size chosen by
     validation loss).
- **`multiversion` and `unsafe`.** Our source contains no `unsafe`, the workspace lint and the CI
  grep pass. The crate's macro does expand to feature-checked `unsafe` calls of the per-level
  copies (that is how runtime dispatch works in Rust); PLAN §11.1 and amendment A8 name this crate
  for this purpose.
  Flagged for review.
- `search hybrid-check` now reports "ns per frontier evaluation" (memo hits included) where it
  said "per network evaluation"; the JSON field names are unchanged.

## Reproduce

```
cargo build --release -p nn -p search
nn-tool bench --equity equity-v0.npy --trick 4 --worlds 40 --max 5000 --reps 5 \
  --widths 512,256,64 --widths 1024,512,128 [--net o4/int.safetensors]
nn-tool random --trick 3 --widths 512,256,64 --out o3-random.safetensors   # benchmarks only
search hybrid-check --trick 3 --cut 4 --net o4/int.safetensors --equity equity-v0.npy \
  --count 32 --seed 1 --to 32 --threads 1 --json hc.json
```
