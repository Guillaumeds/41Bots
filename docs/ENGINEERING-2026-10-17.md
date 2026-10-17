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

## Item A, done (branch `eng-labels`)

**Outcome.** Every label shard of the gate is byte-identical to the binary before (SHA-256, 1 and 4
threads). Exact labels cost 8-17% less at tricks 2-3 for the `random` and `rules` sources; a
harvest's labels cost 24-79% less at tricks 3-4 (trick 3: 24% with the network driver, 58% for the
race harvest, 4.6 times less with the exact driver); at trick 2 the harvests gain little (within the
noise of a shared VM). Only the sidecar's `solver_nodes` of a harvest changes (see lever 1).

### The profile (before; this VM, one thread, release build for x86-64-v3)

`perf` is not installed here, so the profile is the solver's own counters with a timer per solve
(`crates/search/examples/label_profile.rs`, the label solves exactly as `search labels` makes them)
and callgrind [an instruction-level simulator] for where the instructions and cache misses go.

| trick, source | ms a solve | nodes a solve | ns a node | table probes a solve | table hit rate | leaf-memo hit rate |
|---|---|---|---|---|---|---|
| 4 random | 0.028 | 131 | 218 | 58 | 0% | 96% |
| 4 rules | 0.029 | 137 | 208 | 59 | 0% | 97% |
| 4 harvest (every reached position, exact driver) | 0.020 | 112 | 180 | 49 | 0% | 95% |
| 3 random | 8.9 | 81,100 | 110 | 36,900 | 86.7% | 99.9% |
| 3 rules | 8.3 | 72,300 | 115 | 32,800 | 86.7% | 99.9% |
| 3 harvest (sampled positions, network driver) | 6.1 | 54,600 | 111 | 24,400 | 84.0% | 99.9% |
| 2 random | 3,920 | 36.2 M | 108 | 14.9 M | 96.6% | 100% |
| 2 rules | 4,720 | 43.3 M | 109 | 19.5 M | 97.6% | 100% |

Where a trick-3 solve's instructions went (callgrind, 40 `random` solves): node bookkeeping 32%,
playing a card and scoring a trick 16%, the trick's winner 15%, the groups of equivalent cards 14%,
the table 7% (but 80% of the first-level data-cache misses: a probe is a random memory access),
legal moves 4%, the leaf values under 1% (the per-solve leaf memo already answers 96-100%).
Shallow pruning stops 106 nodes of a trick-3 solve's 81,000, so move ordering (history or any
other) has almost nothing to prune faster: not taken. At trick 4 the table never hits within one
solve (a trick has no transpositions); at tricks 2-3 most probes are the last trick's boundaries.

### The levers taken (each gives the same answers; the reasons are in the code)

1. **One table for a decision's positions** (`solver::SolveFamily`, `SolveMem::share`;
   `oracle_labels::harvest_decision`). The exact solves of one harvest decision (every reached
   position with the exact driver, the sampled ones with a network; the race and targeted
   harvests too) run on one table and leaf memo that is no longer emptied between them. The table
   key leaves out the hands and scores because one root fixes them; across roots this holds when
   all roots descend from one ancestor in one world: the family records the ancestor's hands
   (disjoint, the turned card with the seat that must rob it), scoring state and match, and admits
   a root only if its hands lie within those, its tricks follow the ancestor's and its scores and
   flags are exactly the ancestor's plus 5 a trick won since and the 5 of trumps' bonus if played
   since. Then equal keys mean equal positions (proof at `SolveFamily`), the table holds only exact
   values, and no solve with a frontier ever shares. A root that is not admitted is solved from
   empty tables as before. **The sidecar's `solver_nodes` of a harvest becomes the nodes of the
   shared-table solves** (fewer: 8.5 M instead of 22.3 M for the gate's trick-3 harvest with the
   exact driver, 44.7 M instead of 93.0 M for its trick-3 race harvest); the per-label count from
   an empty table is not kept (it would cost the very solve the sharing saves).
   Rows are labelled in their original order (reach order); reordering the solves for more
   sharing was not needed.
2. **The leaf memo kept between solves** (`SolveMem`): emptied by a stamp instead of a new 40 KB
   allocation and fill per solve; it answers exactly as a new memo.
3. **The card orders as tables** (`solver::ranks`): the trick's winner and a trump's rank read from
   per-trump tables of the engine's own `trump_rank`, `plain_rank` and `effective_suit` (tested
   equal to `engine::rules::trick_winner` on every lead and card for every trump, and on 200,000
   random tricks); whether the 5 of trumps is in a trick is whether it won it. 15% fewer
   instructions per trick-3 solve.
4. **Groups of equivalent cards only when they matter** (`Classes::representatives`): when the
   legal cards hold at most one trump and one card of each plain suit, every legal card is its own
   representative, so the groups are not built (tested against the groups on random hands and
   legal sets). 10% fewer instructions per trick-3 solve.

Not taken: move ordering (above); prefetching the table (needs an `unsafe` intrinsic, and the table
is 7% of the instructions); a larger starting table (about 3%, and it would change which positions
the harvest's own search reaches, so it would have to apply to label solves only); huge pages
(already switchable with `RUNNER_HUGEPAGES=1` in `deploy/docker/boot.sh`: no difference measurable
here, 3.82 against 3.81 s and 4.04 against 3.87 s a trick-2 label, where the 128-256 MB table fits
this VM's 260 MB third-level cache; worth an A/B on one rented run, where the cache is smaller).

### The gate

Runs: the binary before and after, each at `--threads 1` and `--threads 4` (4 runs); 5: also the
committed binary (`172920a`) re-run at 4 threads. Every run of a row gave the one SHA-256 shown,
and each run's sidecar `shard_sha256` agrees with the file.

| trick | source | rows | shard SHA-256 (every run) | runs | before: s per 1,000 labels | after | change |
|---|---|---|---|---|---|---|---|
| 4 | random | 3,000 | `d7d927dab7338936872c6348295676681f5beb51c799fbfc9e36c57c621ea9f3` | 4 | 0.267 | 0.267 | +0% |
| 4 | rules | 3,000 | `30d408a48ac91a4c7b7f94adb3cdb0944c8f1bf4faaccefd8775cb53a22295ed` | 4 | 0.067 | 0.067 | +0% |
| 4 | harvest, exact driver | 3,000 | `6e39036ee5d9dff8358d188cf5f320c646355529664980a0062088caa862f1ad` | 5 | 0.334 | 0.201 | -40% |
| 4 | race harvest | 3,000 | `cf2410109dc4e9aceef708082351655edbdea1e66f9a6827fad62cf049228c36` | 5 | 0.267 | 0.134 | -50% |
| 3 | random | 1,000 | `7e14f715a1a609ebf30941b9fef43859617129af2084b0129d5b1266a4661c93` | 5 | 8.21 | 7.21 | -12% |
| 3 | rules | 1,000 | `0bbdb2c1c208754a7138c251589a4548f8fc2f6563de6a800998a12840e2a7bd` | 4 | 8.4 | 7 | -17% |
| 3 | harvest, network driver | 1,000 | `63d1114c195074596bef6ede83aae2ff6666a7984b1a8397f57f7261ec0acd3c` | 5 | 9.2 | 7 | -24% |
| 3 | race harvest | 1,000 | `6d12f1e67bce3edbdb9b99930cc4e0d203e876f55f5d0acb2363a3dcdadbcc9c` | 5 | 10.6 | 4.5 | -58% |
| 3 | harvest, exact driver | 320 | `25dca9480a7e42520f6460895ca63838ba6949cf515384f2235a38b2963881f4` | 4 | 288 | 61.6 | -79% |
| 2 | random | 48 | `69f101566540f1d752aed353b4250eb8dcac48ff7d37feaafed3bf0bb4e40a3e` | 4 | 3,946 | 3,644 | -8% |
| 2 | rules | 48 | `2a57b99c589604a80020ebcecb4938eba7c146322b00a9a2c5097e170d1011af` | 4 | 4,044 | 3,528 | -13% |
| 2 | harvest, network driver | 48 | `5ab6fa72025574b0b395882b63eebb94dc1d3c4295121de41fb5eae36cbb815f` | 4 | 1,429 | 1,396 | -2% |
| 2 | race harvest | 24 | `718286727b6bdfa3acbd076ae199c55567912972528d448fbd1037207d36013b` | 4 | 7,779 | 8,108 | +4% |

How it was run: `search labels`, release builds for x86-64-v3 of `b96a167` (before) and of this
branch (after), the same seeds, start states (20,000 hands) and equity table (`64740dab…`). The
harvests with a network use the trained trick-3 network `o3-small`
(`/home/user/t3work/o3-small/int.safetensors`) and, at trick 2, a random network
(`nn-tool random --trick 2 --widths 512,256,64 --seed 17`); the race harvests keep classes 2 and 4
(at trick 2 with caps of 8 and 4). Trick 2 has 48 rows, not 100: other jobs loaded the VM to 9-25
during these runs, and 96-row runs no longer fitted the 5-minute limit.

Cost: one thread, seconds per 1,000 labels: for a harvest, its searches and labelling in
thread-seconds (from its log); for the others, the run's time a row. Trick 4's times are coarse
(the run's clock steps by 0.2 s, and 3,000 rows take under a second); the profile tool gives the
trick-4 harvest's label solves at 0.020 ms before and 0.012 ms after. The trick-3 and trick-4 runs
had the VM nearly to themselves (load 2-3); the trick-2 runs did not (load 9-25), so their times
carry several percent of noise. Instruction counts are exact: a trick-3 `random` solve takes 24%
fewer (callgrind, the same 40 solves: 2.21 G before, 1.88 G with levers 2 and 3, 1.69 G with
lever 4 too).

At trick 2 the harvests gain little: the gate's trick-1 decisions reach about 5,700 trick-2
positions each, and the 16 sampled ones (12 in the small race run) share few futures (nodes 4% and
3% fewer); the trick-3 decisions reach about 600 and their samples share much more. The per-node
levers also gain less at trick 2 than at trick 3 (the 128-256 MB table's memory accesses weigh
more).

Solver identity (`solver-identity`, release, the production equity table unless said):
trick 4, 5,400 worlds, all 16 switch combinations, both tie orders: 0 mismatches; trick 3, 340
worlds, the lean 9 combinations (and 40 with the tie-heavy leaf, the 8 table combinations): 0
mismatches; the parallel solve on 4 threads against the sequential one, trick 3, 200 worlds: 0
mismatches. `cargo test --workspace` passes, with the new tests: the family's shared table against
solves from empty tables (`a_family_shares_one_table_without_changing_a_value`), what a family
admits (`a_family_admits_only_its_descendants`), the rank tables against the engine
(`ranks::tests`), and the representatives' shortcut against the groups
(`representatives_match_the_groups`).

Not done here: the cost on one rented run (no rentals in this session); the next trick-3 run
measures it.

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
- **The network-driven harvest** (`search labels --trick 4 --count 600 --seed 7 --source harvest
  --harvest-net …`, 2 threads; the `Recorder` with a network driver): the label shards are
  byte-identical (SHA-256) before and after with the trained O_4 and with a 2048-1024-256 network
  (batched), and the sidecars differ only in the seconds and the binary's hash.
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

## Harvest scheduling (branch `eng-harvestpar`, 2026-10-17)

**Outcome.** The harvests (the audit loop's audit shards and targeted harvests, the base and race
harvests) now share their exact labels out over every thread. Same shards byte for byte, same
decisions and sidecar values, apart from the timing fields, the new `harvest.label_solves` record
and, at trick 2 only, `solver_nodes`. On this VM a 200-row trick-2 harvest took 258 s instead of
589 s, at 9.05 M solver nodes per thread-second instead of 3.69 M. A single-thread plain label
does 9.7 M.

### The cause (the trick-2 mechanism test, 245 threads, 2026-10-17)

Each harvest decision's search, sample and every one of its labels ran on one thread
(`ordered` over decisions, a window of `4 × threads + 16` decisions ahead). At trick 2 a label is
seconds to minutes, so one decision's labels were 16 (audit shard) to 256 (targeted harvest,
`audit_keep_cap`) solves in a row on one thread. Two things followed:
- the shard is written in decision order, so it waited on its slowest decisions while the other
  threads made decisions up to the window ahead, nearly all past the row count, and thrown away;
- when the row count was reached, the decisions still being made ran to their end before the
  step could finish.

The logs show it (`hybrid/labels-audit-r1.log`, `labels-target-r1.log`):
- The rows counter counts the rows of every decision finished, including those past the count.
  It reached 5,000 at 100 s in the audit shard and at about 60 s in the targeted harvest.
- The audit shard ran 1,751.7 s on 245 threads (429,000 thread-seconds). The labels of the 368
  decisions it kept took 27,738 thread-seconds (6.5%) and their searches 160.
- Nothing came after the rows: hashing 0.0 s, writing 0.0 s, sidecar 0.3 s after the last row.
- The targeted harvest (237 threads) was still running at 5,823 s when the run was stopped.

The labels' table was not the cause. The sidecar's "table 64 MB" is the harvest's own search table
(`HARVEST_TT_MB`, fixed because what the search reaches may depend on it). The label solves always
used the trick's table, as the plain labels do (`tt_mb_for`: 256 MB at trick 2). The trick-1
searches with the network were cheap (0.6% of the thread time) and already ran in parallel, one
decision per thread.

### What changed (`search::ordered::pooled`, `oracle_labels::HarvestSchedule::Pooled`)

- Decisions are still searched and sampled one per thread, in order, exactly as before.
- The sampled positions of the decisions taken, cut at the row count, form one ordered list. Every
  thread takes the next unit of that list, whichever decision it came from, and a label always goes
  before a further decision.
- Units are one position at trick 2 and all of a decision's positions at tricks 3–5
  (`label_group_for`). At tricks 3–5 a label is milliseconds and sharing one table across a
  decision's labels saves a quarter to three quarters of the nodes, so the node counts there are
  exactly the old ones.
- No label is made for a decision past the row count. Finished labels may run up to 1 GiB ahead of
  the writing (`LABEL_WINDOW_BYTES`); a pending plan holds only its sampled positions.
- The rows counter now counts labelled rows of the shard only.
- The old schedule stays as `--harvest-schedule decision`, for comparisons and the tests.

Also:
- `search labels --chunk-rows` defaults to 1 at trick 2, 32 elsewhere. A trick-2 chunk of 32 rows
  was minutes on one thread: the mechanism test's 5,000-row rules and validation shards were 157
  chunks on 245 threads, one wave.
- `search labels --tt-mb` sets the label solves' table, and the runner lowers it only if one table
  per thread would not fit the machine (`hybrid::label_tables`, `crate::mem`).

### Identity

- Unit tests:
  - `oracle_labels::tests::harvest_schedules_give_the_same_shard`: the shard, its SHA-256, the
    decisions' facts, the group tallies and the row counts are identical between the old schedule
    and the new one. It covers 1, 2 and 4 threads; one position or a whole decision a unit; a 1 MB
    table; the exact and network drivers; tricks 3, 4 and 5; race, targeted and bounded harvests.
    Node counts equal the old ones with whole-decision units, and with one position a unit each
    row's nodes are its solve's from empty tables.
  - `label_table_size_never_changes_a_row` and `ordered::tests::pooled_*`.
- CLI (release binaries before = `4fd352d` and after, same seeds, `equity-v0.npy`):

| run (seed, rows) | shard SHA-256, every run | runs | `solver_nodes` before → after |
|---|---|---|---|
| trick 2 harvest, network driver, cap 8 (231, 24) | `318e10eac2d5043c021fb95bab3d21603e5427aba715b48fddb81b3c2cef2882` | before (4 threads); after on 1, 2 and 4 threads, with `--harvest-schedule decision`, with `--tt-mb` 16, 64 and 128 | 1,434,573,927 → 1,449,543,129 on 1, 2 and 4 threads (+1.0%); `decision`: 1,434,573,927 |
| trick 2 targeted harvest, the mechanism test's round-1 predicates, keep cap 8 (23151, 24) | `2edf4d97d2a4a4ef6a6f668f4994f554933de08bee44b44bf454ab0263605da0` | before; after on 1 and 4 threads | 635,436,146 → 642,348,210 (+1.1%) |
| trick 3 harvest, network driver `o3-small`, cap 16 (103, 600) | `08abb6394e347f0e4fb64ad8d90e78f41c219e9742982a646b5e096c0161924d` | before; after on 1, 2 and 4 threads | 44,211,404, unchanged |
| trick 4 harvest, exact driver, cap 16 (104, 3,000) | `d1d43771e3adaeb6fde252eb5cb982314f00c6a6d1aa6f75fd0bda16b4c485eb` | before; after on 1 and 4 threads | 429,724, unchanged |
| trick 4 race harvest, classes 2,4 (105, 3,000) | `a183c4b3c7eb979464529593c2b3e31db320d469345af5e5d99a3f85fcf5d230` | before; after | 377,688, unchanged |
| trick 2 random labels, chunk 32 before, 1 after (511, 12) | `93c0aaa6621a23944b8aed3f36d94e35eb29879139d6b94640ebb185b5eb0fe3` | before; after; after with `--tt-mb 16` | 317,173,637, unchanged (16 MB: +4%) |

  Every sidecar key is the same as before's except `binary_sha256`, `seconds`, the new
  `harvest.label_solves`, and `solver_nodes` where the table or the units changed (trick 2 pooled,
  or another `--tt-mb`, then recorded as `label_tt_mb`). The 200-row and 64-row harvests below
  have the same SHA-256 before and after too (`2eeb3c58…`, `5e966b27…`).
- Against the rented machine: the new binary's first 48 rows of the mechanism test's round-1 audit shard
  (seed 23101, its own network `6557aa67…`, 4 threads) are byte for byte the first 48 records of
  the shard the old binary made on 245 threads (`1811345d…`, fetched from R2).

### Speed (this VM, 4 threads, trick 2, network driver, cap 16, random 1024-512-128 O_2)

| harvest | before: s | after: s | speed-up | nodes per thread-second | CPU core-seconds |
|---|---|---|---|---|---|
| 200 rows, 13 decisions | 588.7 | 257.8 | 2.3× | 3.69 M → 9.05 M | 2,173 → 991 |
| 64 rows, 4 decisions | 543.8 | 86.7 | 6.3× | 1.36 M → 8.75 M | |
| 24 rows, cap 8, 3 decisions | 208.9 | 48.4 | 4.3× | | |
| targeted, 24 rows, keep cap 8 | 300.1 | 26.4 | 11.4× | | |
| plain random labels, 12 rows (chunk 32 → 1) | 33.7 | 11.5 | 2.9× | | |

- The 200-row run before: 4 cores busy until 525 s, then 2 cores until 589 s. The labels of the
  rows kept took 1,230 thread-seconds; the other 940 core-seconds went to decisions thrown away.
- After: 4 cores busy until 255 s of 258 s. The labels took 1,014 thread-seconds, all kept.
- The pooled labels are faster per node too (9.2 M against 7.1 M nodes a thread-second while
  labelling). The shared table of a decision's chain grew larger and missed the cache more, and
  each decision allocated a new 256 MB table. That more than pays for the 1–7% more nodes of
  solving each trick-2 label from an empty table.
- The table at trick 2 (24-row harvest, 4 threads): 16 MB 93.7 s (+20% nodes), 64 MB 47.2 s
  (+0.1%), 128 MB 46.2 s, 256 MB 48.4 s.

**Why it scales.** Take `n` labels of mean time `c` on `N` threads, the longest single label
`c_max`. With every label in one list, every thread is busy until the last label is claimed. The
step then ends within `c_max`: about `n·c/N + c_max`. Under the old schedule the unit was a
decision of `g` labels in a row (16 for an audit shard, up to 256 for a targeted harvest). The
shard waited on the slowest decision in its order, and the decisions in flight at the end ran on.
So its tail was the slowest decision's `g` labels, plus the work thrown away.

The 4-core VM hides most of this: 13 decisions are already 3 a thread. At 245–384 threads a
5,000-row shard is 1–1.5 decisions a thread, so the old schedule's time was its slowest decision.

For the run's 200,000 audit rows at 384 threads (about 7.4 thread-seconds a label on the rented
EPYC): `n·c/N` ≈ 3,850 s. The pooled tail is one trick-2 label, minutes at worst, against a
decision of 16 labels in a row or, for a targeted harvest, up to 256 (half an hour or more).

### Memory at 384 threads (the 129 GB EPYC 9654, offer #48544090)

Per thread:
- The label table: `tt_mb_for(2)` = 256 MiB at trick 2, 64 at trick 3, 8 at tricks 4–5. It is
  reserved at its cap and its pages are touched as it grows; a thread keeps the largest size it
  reached.
- The network's memo, about 2.5 MB, and the stack, 2 MiB.
- The harvest search's 64 MiB table is reserved but hardly touched (trick 1 has no trick boundary
  to store).

Shared: the equity table (26 MB), the network's weights (6–30 MB), finished labels ahead of the
writer (at most 1 GiB), and the pending decisions' samples (small).

Measured peaks:
- 1,123 MB for the 200-row trick-2 harvest on 4 threads (1,125 MB before), against 78 MB for
  the process alone. That is about 260 MB a thread: the table reaches its cap once a thread meets
  a hard position.
- 255 MB for 8 random labels.

Expected at 384 threads, for every trick-2 label step (random, rules, validation, audit shard,
targeted harvest), the same as before:
- tables 384 × 256 MiB = 96 GiB (103 GB);
- plus about 3 GB (memos, stacks, window, base);
- about 106 GB in all.

At trick 3 the tables take 24 GiB, at trick 4 3 GiB.

On the 129 GB EPYC 9654 (about 120 GiB) the runner's budget (the limit less 10% less the
process's other memory) is about 105 GiB. The tables fit, so the runner passes no `--tt-mb`
(unit test `label_tables_fit_the_machine`).

With the trainer beside (about 3 GB of host memory), the total is about 109 GB of 129 GB. When
the rough test runs beside with its 8 GB verification table, the label step has fewer threads and
fewer tables. The margin is about 10–15 GB. If more is wanted, a 128 MB table at trick 2 halves the
label tables at no measured cost (above); that would change only `solver_nodes`, and is not done
here.

A machine with less memory gets its label tables lowered by the runner to fit: about 145 MB at
384 threads on 64 GiB. It gets a warning, never a different label.

### What is not verified here

- No rented run: the speed-up at 245–384 threads follows from the argument above, not a
  measurement. The longest single trick-2 label on the rented machines, which bounds the pooled
  tail, is not known.
- The runner's lowering of the table is unit-tested on stated machine sizes only.
