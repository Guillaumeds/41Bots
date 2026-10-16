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

## Item B, done (branch `eng-nnbatch`, 2026-10-17)

**In one paragraph.** The hybrid asks its network for the value of every position it reaches at
the start of the frontier trick. Three changes make those values cheaper, all giving exactly the
same numbers as before: (1) the network's first layer for trained networks is now updated from
one position to the next instead of being recomputed (about half the work of one evaluation of
the trained O_4); (2) the solver now hands the network all the positions it is about to reach
from a few plays before the frontier, and the network values them as one **batch** [several
positions at once, so its weights are read from memory once for all of them instead of once per
position]; (3) a batched multiply for the hidden layers. Measured on this VM (one thread, the
trick-3 golden decisions 0-15): the trained O_4 (1024-512-128) hybrid goes from 2.78 to 2.32 ms a
world (1.2 times faster), a 2048-1024-256 network from 41.0 to 27.5 ms (1.5 times) and a
2048-2048-1024-256 network from 101.7 to 71.8 ms (1.4 times). Every hybrid-check JSON compared is
byte-identical before and after, timing fields aside. Batches are on by default only for networks
with a hidden layer of at least 2048 × 1024 weights; for 1024-wide networks they measured no
faster here (their weights stay in a core's own cache), so those keep one-at-a-time evaluation.

### Where the time goes (profile)

Microseconds per evaluation on this VM (Xeon, 2.1 GHz, one thread), real frontier positions of
trick 4 reached from trick-3 roots, one at a time (`nn-tool bench`, `nn-tool bench-batch`). The
encoding [turning a position into the network's inputs] costs 0.13–0.22 µs and the outcome table
with the softmax [turning the outputs into a money chance] 0.24–0.36 µs; the rest is the forward
pass [the network's arithmetic], by layer:

| network | level | first layer | hidden layer 1 | hidden layer 2 | hidden layer 3 | output | total |
|---|---|---|---|---|---|---|---|
| trained O_4 1024-512-128, before | v3 | 11.5 | 8.1 | 1.5 | – | 0.3 | 14.9 |
| same | v4 | 5.8 | 5.1 | 1.1 | – | 0.2 | 12.1 |
| trained O_4, after change 1 | v3 | 5.5–5.9 | 4.5–4.8 | 0.9 | – | 0.1 | 11.0–13.6 |
| same | v4 | 2.8–2.9 | 4.3–4.4 | 0.8 | – | 0.1 | 7.9–8.5 |
| random 1024-512-128 | v3 / v4 | 4.3 / 4.1 | 30.5 / 31.6 | 4.3 / 4.2 | – | 0.4 | 38 |
| random 2048-1024-256 | v3 / v4 | 10–12 / 8–10 | 154–171 / 138–181 | 22–26 / 21–28 | – | 0.8–1.4 | 169–209 |
| random 2048-2048-1024-256 (VM loaded) | v3 / v4 | 14 / 13 | 512 / 565 | 281 / 223 | 46 / 28 | 0.9 | 880–980 |

"v3" is the AVX2 code [x86-64-v3: the vector instructions of the rented EPYC machines], "v4" the
AVX-512 code [this VM]. The memo of recent positions answers 20% of the requests at trick 4. So:
the trained O_4 spent three quarters of its time (v3) in the first layer, because its first-layer
weights are too large for the narrow 32-bit sums and it took the slow path (a fresh 64-bit
product of every active input, every time); every wider network spends almost all its time in the
hidden layers' matrix products, the 2048 × 2048 layer alone 58–64%. The VM was shared with other
jobs (load 4–14 on 4 cores) through these measurements, so single numbers vary by ±15% and more;
the hybrid comparisons below are CPU time [time the processor spent on the job, not clock time],
which other jobs disturb less.

### What changed (all answer-identical, PLAN §10)

1. **First layer of trained networks** (`crates/nn/src/fast.rs`). When the 32-bit sums cannot be
   used, the sums of the active inputs' weight rows are now kept in 64-bit and updated by the rows
   that switched on and off since the previous position, as the 32-bit path already did, instead
   of a fresh 64-bit multiply of every active row. Why it is exact: every intermediate value is the
   sum of the rows of a set of at most 2 × 96 distinct inputs, each weight a 32-bit number, so it
   stays below 2^39 and no 64-bit sum can overflow; the final sum, shifted by 24 and added to the
   dense part, is the reference's own sum, within the network's checked 2^62.
2. **Batches from the solver** (`crates/solver/src/frontier.rs`, `crates/solver/src/fast.rs`).
   The `Frontier` trait has three new methods, each with a default that keeps every existing
   evaluator working unchanged: `value_batch` (several positions at once; the default asks
   `value` for each), `prefetch_depth` (from how many plays before the frontier the solver should
   gather; default 0 = never) and `prefetch` (the positions ahead; default: ignore them). At the
   first node of a line within `d` plays of the frontier (`d` = the evaluator's depth; counted
   after the node's forced plays, so a forced card into the frontier is covered), the solver
   gathers every frontier position reachable from there in the current trick, through the same
   cards its search plays (one card per group of equivalent cards, forced cards followed), at most
   256, and hands them to the evaluator. **How the values reach the search in the same order:**
   the search does not take values from the batch. It goes on exactly as before and asks the
   evaluator for each frontier position when it reaches it (`value_at`, same place, same order);
   the oracle has meanwhile valued the batch into its memo, so those requests are answered from
   there. So the order of visiting, every counter, every table entry and the shallow pruning are
   untouched, and a recording evaluator (the harvest's `Recorder`, the audit's `GroupRecorder`)
   records exactly what it recorded before (they pass `prefetch` on to their network and record
   only in `value_at`). What a batch can cost: positions the search then does not ask for (it
   stopped early, or found the position in its table) were valued for nothing.
3. **Batched evaluation in the network** (`crates/nn/src/oracle.rs`, `fast.rs`, `kernel.rs`).
   `Oracle::eval_batch` looks each position up in the memo (a hit, or a repeat inside the batch, is
   not evaluated again), runs the others through `FastMlp::forward_batch` and stores them in the
   memo like any evaluation. `forward_batch` runs the first layer position by position (the
   incremental sums going from each to the next; neighbours in a batch are siblings and differ by
   a few inputs) and each hidden layer as one product for up to 64 positions
   (`kernel::mac_pairs_batch_at`): the weight matrix is taken in chunks of 128 KiB [small enough
   for the second-level cache of the rented EPYC machines], and every position adds its own
   non-zero inputs' rows of the chunk to its sums, so each chunk comes from main memory once per
   batch. At v4, a matrix that stays in the second-level cache (1 MiB or less) is taken position by
   position by the existing register-blocked loop, which is faster there. Why it is exact: every
   output of every position is the same sum of the same terms as one at a time, only grouped
   differently; integer addition does not depend on the grouping, and every partial sum is the
   bias plus some of the terms, so it stays within the network's checked 2^62 (the bound proved in
   `crates/nn/src/mlp.rs`). Activations, rounding, the output head and the memo are the same code
   as one at a time.
4. **When batches are used** (`nn::oracle::default_prefetch`): depth 3 (`WIDE_PREFETCH`) for a
   network with a hidden layer of at least 2048 × 1024 weights (`WIDE_WEIGHTS`), none otherwise;
   `Oracle::set_prefetch` overrides it. Never with the memo off (the values would have nowhere to
   wait). Memory: a batch's buffers are made per thread and network on first use, 64 positions ×
   (6 bytes × the widest layer + 8 bytes × the widest hidden layer): 1.75 MiB at 2048 wide,
   3.5 MiB at 4096 (about 350 MB and 670 MB on 190 threads).

### Gate evidence

- **Fast kernel = plain reference** at widths 1024, 2048 and 4096 and at the format's limits, at
  every level this machine has (base, v2, v3, v4): the kernel tests (`kernel.rs`: every batch
  order, batches of 1 to 64 positions, empty and repeated rows, the matrix's last row, weights of
  ±2^31; the 64-bit first-layer sums from any start, rows added and removed), `tests/fast.rs` and
  `tests/wide.rs` (one at a time, then in batches in solver order and shuffled, one past the
  largest batch; random networks at trained scale and three times it, first layers needing 64-bit
  sums, networks at the limits).
- **Batched = unbatched on every frontier** of a fixed set of trick-3 and trick-2 decisions:
  `tests/fast.rs::hybrid_with_batches_equals_the_hybrid_without` (16 trick-3 roots at every place
  in the trick, frontier at trick 4; 3 trick-2 roots, frontier at trick 3; depths 0, 1, 2, 3, 5):
  the solve's answer, every counter, and the sequence of positions asked for with their values are
  identical, and the values equal the plain evaluation. `crates/solver/tests/frontier.rs::
  frontier_batches_change_nothing_and_cover_the_requests`: for depths 1 to 7, five switch
  combinations, both tie orders, trick-4, trick-3, trick-2 and trick-1 (pending rob) roots, the
  answer, every counter and the request sequence are unchanged, batches hold frontier positions
  only, and from depth 1 to 3 every position asked for was handed over in a batch first; the
  parallel solver's answers too. `oracle_batches_equal_the_plain_evaluation`: `eval_batch` equals
  `eval_reference` for both heads, a 64-bit first layer, repeats, memo hits and memo off.
- **Hybrid-check JSON byte-identical** (timing fields and the binary's hash aside): the
  trick-3 golden decisions 0-15 × 320 worlds with `lead3c/x/hybrid/net/int.safetensors` and
  `equity-v0.npy`, before (`b96a167`) and after, on 1 and 4 threads (batch depths 0 to 4 tried
  too, all identical); the same decisions × 16 and × 8 worlds with random 2048-1024-256 and
  2048-2048-1024-256 networks (batched by default), two runs each.
- **wasm32**: `cargo check -p nn --target wasm32-unknown-unknown` passes (the plain loops).

### Before and after

H_4 on the trick-3 golden decisions 0-15, CPU milliseconds per world on one thread (the mean of
two runs, the 0.32 s of loading subtracted), and per frontier request (the same divided by the
257–258 requests a world; it includes the solver's own time, about a tenth):

| network | before (`b96a167`) | after | × | per request before → after |
|---|---|---|---|---|
| trained O_4 1024-512-128 (no batches by default) | 2.78 ms | 2.32 ms | 1.20 | 10.8 → 9.0 µs |
| random 2048-1024-256 (batches from 3 plays) | 41.0 ms | 27.5 ms | 1.49 | 159 → 107 µs |
| random 2048-2048-1024-256 (batches from 3 plays) | 101.7 ms | 71.8 ms | 1.42 | 394 → 278 µs |

The batch depth, measured the same way before the last kernel choice: for 2048-1024-256, depth 0
/ 2 / 3 gave 42 / 33 / 30 ms a world; for 2048-2048-1024-256, 102 / 89 / 75; for the trained
1024-512-128, 2.5 / 2.8 / 2.7 (batches slower: hence none for it). The batched network alone
(`nn-tool bench-batch`, consecutive reached positions, µs per position): 2048-1024-256 from
169–209 one at a time to 127–148 in batches of 8–16 (v3 and v4); the trained 1024-512-128 about
the same either way (within 10%).

H_3 on trick-2 decisions (`hybrid-check --trick 2 --cut 3 --count 4 --seed 1 --to 2`: 4
decisions × 2 worlds, 1,974 frontier requests a world, random networks for trick 3), clock time per
world on one thread with the VM loaded (the exact solves, the same work in both runs, are given
as a gauge of the load); the JSONs are byte-identical before and after for all three networks:

| network | before | after | exact solve, before / after |
|---|---|---|---|
| random 1024-512-128 (no batches by default) | 86.2 ms | 83.1 ms | 3.67 / 3.63 s |
| random 2048-1024-256 | 410 ms | 257 ms (×1.6) | 3.17 / 3.16 s |
| random 2048-2048-1024-256 | 848 ms | 733 ms (×1.16; ×1.4 at the load of the run before) | 2.56 / 3.07 s |

### What did not work, and what is left

- A kernel that multiplies each weight by every position of a group at once (the union of the
  group's inputs, sums for 8 positions in registers) was 3–10 times slower: it reads each weight
  row in short pieces far apart in memory, and multiplies the zeros of inputs only some positions
  have. A plain block order (outputs outermost) was also slower. Both were dropped; the chunked
  order above was the best measured at v3 and v4.
- The gain is mostly memory traffic, so it should be larger on the rented machines (a 512 KiB
  second-level cache per core on Zen 2/3 EPYC, against 2 MiB here, and up to 190 threads sharing
  the memory) than on this 4-core VM. That could not be measured here (no rentals). Whether 1024-wide
  networks gain from batches there too is open: `Oracle::set_prefetch(3)` turns them on (answers
  unchanged); a first run on a rental could time it.
- The arithmetic itself (64-bit products of 32-bit weights and 26-bit activations) is unchanged;
  going faster again needs a narrower integer format (A8's 8/16-bit weights), which changes answers
  and is the owner's decision (docs/m3-hybrid/nn-speed.md, "Limits").

### Solver lines touched (for the other branches)

Only the frontier interface and the hybrid's hook, no search logic:
- `crates/solver/src/frontier.rs`: `Frontier` gains `value_batch`, `prefetch_depth`, `prefetch`
  (all with defaults); `FrontierCut` gains `prefetch_depth`, `prefetch` (delegating). No existing
  line changed.
- `crates/solver/src/fast.rs`: the `use crate::frontier` line; two constants after `LEAF_SLOTS`
  (`GATHER_CAP`, `SEATS`); two new methods of `Search` just before `fn node` (`prefetch_ahead`,
  `gather`); in `node`, `let arrival = p;` before the forced-play chain and
  `self.prefetch_ahead(arrival, p, false);` after the burn check that follows it (before the table
  probe); in `solve_fast_cut`, `s.prefetch_ahead(pos, pos, true);` before the root loop.
- Wrappers that pass batches on to their network: `search::hybrid::Timed` (counts the batch's
  time as evaluation time), `search::audit::GroupRecorder`, `search::oracle_labels::Recorder`
  (with a network driver only).
