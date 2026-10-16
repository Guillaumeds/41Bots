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
