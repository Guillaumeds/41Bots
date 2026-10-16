# Engineering plan of 2026-10-17: same answers, less cost

Owner's request (2026-10-17, 00:50 Mauritius): research, plan and build the engineering
opportunities proposed the evening before; the ones the trick-3 and trick-2 runs use must be ready
before each run starts. Every item below gives **byte-identical answers** (values, labels, networks'
outputs) or, where it changes an estimate's noise, keeps it unbiased and is gated like a method
change. None plays worse to save compute (CLAUDE.md, "Quality first").

## 0. Correction and what already exists

The solver items of the 2026-10-16 review are **already built and merged**: 32-byte table slots
(was 112, `ff202a8`), tables reserved at their cap (`19b19d2`), forced play resolved without nodes
(`f10e993`), leaf values memoised per solve and bit-parallel equivalent-card groups (`bb143b4`).
What is left on the solver is in item A.

## 1. Where the money goes now

| run | dominant cost | measured |
|---|---|---|
| trick 2 | exact labels of trick-2 positions | ~3 s a label on one thread (2.7 random, 3.2 rules) |
| trick 3 | exact labels (~10 ms each) and the network-driven harvest | labels 16.5 min, harvest 10 min of a 45-min run |
| trick 3/2 | network evaluations at the frontier (2048-wide: 5.4-14x the 1024 cost) | 1024: ~20-35 us per evaluation |
| trick 1 (app, teacher) | deals per decision x H_2 per deal | trick-1 noise 2-5x trick 2's per deal |

## 2. Items, order and gates

| item | what | needed before | owner of the build |
|---|---|---|---|
| A | Label cost: profile trick-2/3 label solves; share one table across the positions of one harvested decision; any further answer-identical solver lever the profile shows (move ordering from history, prefetch, huge pages) | trick 3 and trick 2 runs | Opus (branch `eng-labels`) |
| B | Network evaluation for wide nets: evaluate a node's frontier children as one batch (matrix-matrix, weights read once), and a faster integer kernel for 2048-4096 widths, differential-tested against the plain reference | trick 3 and trick 2 runs | Opus (branch `eng-nnbatch`) |
| C | Fewer deals for the same precision at trick 1: deals drawn in balanced groups (who holds the top trumps and the Joker; proportional stratification, unbiased) | the app's trick-1 search and the teacher (not the runs) | Opus (branch `eng-strata`), after A and B |
| D | The student network for the app's bots (PLAN M4-M6): distilled from the hybrid teacher | after O_2 is proven (the teacher must be the proven one) | plan now, build after the rungs |

Gates (each before merge):
- A: every label shard of fixed seeds at tricks 2, 3 and 4 (random, rules, harvest, race) byte-identical
  to today's binary (SHA-256), any thread count; solver identity tests; the measured cost per 1,000
  labels before and after on this VM and on one rented run.
- B: the fast kernel equals the plain reference at 1024, 2048 and 4096 (all instruction levels); every
  hybrid-check JSON of a fixed golden subset byte-identical before and after; measured evaluation time
  per position at 1024-512-128, 2048-1024-256 and 2048-2048-1024-256.
- C: unbiased by construction (proportional allocation, the same estimator); a test that the deal
  distribution of every stratum equals the sampler's; the variance gain measured on saved decisions;
  adopted only after a duplicate-tournament SPRT at equal deals (PLAN §9), as it changes the estimate's
  noise.
- D: PLAN M4-M6 as written, with the hybrid H_k as the teacher; promotion by SPRT.

## 3. Readiness

The trick-4 audit-loop test runs first (from about 02:00 Mauritius). A and B are merged and released
before the trick-3 and trick-2 loops start; if one is late, that loop starts on the current binary
and the item joins its next run (no answer changes, so mixing binaries across rounds is safe).
