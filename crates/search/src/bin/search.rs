//! `search`: the M3 PIMC measurement and teacher tools (docs/M3-SPEC.md §5.8–5.10, §13.1).
//!
//! ```text
//! search decisions --trick T --count N --out FILE [--seed S] [--start-hands H] [--allow-uneven]
//!                  [--threads T]
//! search decisions --keep PRED.json --trick T --count N --out FILE --equity TABLE
//!                  [--keep-share 0.5] [--keep-worlds 8] [--cut T+1] [--net WEIGHTS]
//!                  [--first-candidate 0] [--max-candidates 200N] [--seed 1] [--label 1]
//!                  [--tt-mb 64] [--source belief|rejection] [--threads T] [--start-hands H]
//! search worlds    --decisions FILE --method exact|rollout|hybrid --to N --out FILE
//!                  [--cut K --net WEIGHTS (hybrid: a silver reference)]
//!                  [--from-world A] [--label L] [--seed S] [--first I] [--last J]
//!                  [--ids 3,17,42 | --ids FILE] [--threads K] [--solve-threads K] [--tt-mb M]
//!                  [--schedule fixed|auto] [--tt-budget-mb B] [--sched-json JSON|FILE]
//!                  --equity TABLE [--progress FILE]
//! search robber-burn [--trick 1] [--count 32] [--seed 1] [--start-hands 20000] [--position 0]
//!                  [--share-worlds 1000] [--worlds 4] [--equity TABLE] [--tt-mb 512] [--label 1]
//!                  [--source belief|rejection]
//! search bench-t1  --mode fill|prod --decisions FILE --equity TABLE --json OUT --threads T
//!                  [--window-s 240] [--tt-mb 512 (fill)] [--tt-budget-mb B]
//!                  [--worlds-per-decision 800] [--sched-json …] (prod) [--seed 1] [--label 1]
//!                  [--usable-cores C] [--min-cpu-pct 80] [--ramp-s 20] [--busy-streak-s 10] [--min-busy-pct 95]
//!                  [--cpu-interval-s 30] [--no-guard] [--sample-ms 1000]
//!                  [--progress FILE] [--stop-file FILE]
//! search agree     --worlds FILE [--worlds FILE …] [--grid 10,20,40,80,160,320] [--json FILE]
//!                  [--decisions FILE …]
//! search curves    --worlds FILE [--worlds FILE …] [--decisions FILE …]
//!                  [--grid 10,20,…|10:320:10] [--m M]
//!                  [--splits 400] [--boot 1000] [--seed 1] [--ids …]
//!                  [--thresholds 0.0001,0.00005,0.00002] [--min-cell 10] [--threads K]
//!                  [--method exact] [--label 1] [--json FILE]
//! search score     --yardstick FILE --n N --method FILE:N [--method FILE:N …] [--json FILE]
//!                  [--boot 1000] [--seed 1] [--allow-silver]
//! search teacher   --hands N --out DIR --equity TABLE [--seed S] [--config NAME] [--start-hands H]
//! search teacher   --trick 1|2 --count N --deals D --seed S --net WEIGHTS|exact --equity TABLE
//!                  --out SHARD.npy [--source rules|mixed|random|student] [--first I] [--deal-from W]
//!                  [--role train|test] [--threads T] [--tt-mb 64] [--start-hands 20000]
//!                  [--progress FILE]
//! search labels    --trick K --count N --seed S --source random|rules|harvest --out FILE.npy
//!                  --equity TABLE [--threads T] [--start-hands 20000] [--chunk-rows 32 (1 at K=2)]
//!                  [--tt-mb M]
//!                  harvest only: [--harvest-cap 16] [--harvest-positions 0.25,0.25,0.5]
//!                  [--harvest-tricks K-1 | --harvest-tricks 2,3] [--harvest-net WEIGHTS]
//!                  [--harvest-strata 1,2,3,4] [--harvest-race-classes 2,4 [--harvest-race-cap 256]]
//!                  [--harvest-keep PRED.json [--keep-cap 256]] [--max-decisions N]
//!                  [--harvest-schedule pooled|decision]
//!                  [--seat-weights 2,1,1]
//!                  [--harvest-risk MODEL.json --risk-profile PROFILE.json]
//!                  [--label-net WEIGHTS [--exact-share 0.1] [--exact-leaves SPEC
//!                   [--exact-leaf-tt-mb N]]]
//! search risk-profile --trick K --seed S --harvest-net WEIGHTS --harvest-risk MODEL.json
//!                  --equity TABLE --out PROFILE.json [--decisions 2000] [--bands 20] [--power 1]
//!                  [--floor 0.2] [--harvest-tricks K-1] [--harvest-positions 0.25,0.25,0.5]
//!                  [--harvest-strata 1,2,3,4] [--start-hands 20000] [--threads T]
//! search student-check --net STUDENT.safetensors --teacher SHARD.npy [--teacher …] --equity TABLE
//!                     (or --golden-decisions D.jsonl --golden-worlds W.jsonl [--to N] for exact values)
//!                  [--allow-stamp KEY …] [--boot 1000] [--boot-seed 1] [--json OUT]
//!                  [--baseline rules] [--baseline random] [--split-half] [--top-k 2,3]
//! search hybrid-check --trick T --cut K --net WEIGHTS --equity TABLE
//!                  [--decisions FILE | --count N] [--seed 1] [--start-hands 20000]
//!                  [--first I] [--last J] [--ids …] [--to 320] [--label 1]
//!                  [--worlds FILE …] [--verify-pct 1] [--recompute] [--threads T] [--tt-mb 64]
//!                  [--boot 1000] [--boot-seed 1] [--json OUT] [--source belief|rejection]
//!                  [--verify-solve-threads 1] [--verify-tt-mb (--tt-mb)]
//!                  [--explain 143,335 [--explain-worlds 3] [--explain-json FILE]]
//!                  [--slices PRED.json [--slice-share 0.5]] [--standard 0.01]
//!                  [--reference exact|silver [--reference-net WEIGHTS] [--reference-cut R]
//!                   [--allow-exact]] [--race-groups [--race-share 0.5] [--min-group 30]]
//!                  [--progress FILE]
//! search hybrid-check --trick T --cut K --net WEIGHTS --equity TABLE --decisions FILE
//!                  --saved RUN=FILE [--saved RUN=FILE …] [--only D:W,…] [--hard D:W,…]
//!                  [--verify-pct 0] [--verify-rule all|rules-v1|searched]
//!                  [--verify-solve-threads 1] [--verify-tt-mb M] [--threads T] [--tt-mb 64]
//!                  [--boot 1000] [--boot-seed 1] [--json OUT] [--progress FILE] [--plan-only]
//!                  [--slices PRED.json [--slice-share 0.5]] [--standard 0.01]
//! search hybrid-check … --net A,B,… (or --net A --net B …: an ensemble, golden-set form only)
//! search hybrid-check … --truth net --noise-sd-by-class P0,P1,P2,P3,P4 [--noise-seed 1]
//!                  [--noise-calibrate N [--noise-calibrate-worlds 8] [--noise-calibrate-rounds 4]]
//! search hybrid-check … --exact-leaves none|all|race:4|race:2,4|race:1-4[+band:LO-HI]
//!                  [+learned:MODEL.json:t=T|t=inf|budget=S]
//!                  [--exact-leaf-tt-mb (64 at --cut 3)]
//! search hybrid-paired --a A.json --b B.json [--a-name A] [--b-name B] [--boot 1000]
//!                  [--boot-seed 1] [--json OUT]
//! ```
//! **Exact leaves** (`search::exact_leaves`, docs/m3-hybrid/rung2.md "Exact leaves (G)"):
//! `--exact-leaves` solves the hybrid's frontier positions the policy picks exactly (the labels'
//! exact solver and tie rule) instead of valuing them with the network: `race:LIST` every leaf of
//! those race classes, `band:LO-HI` every leaf the network gives the decider's team LO% to HI%,
//! `all` every leaf, joined by `+`; `none` (the default) is the hybrid as before, byte for byte.
//! The report adds, per race class, the distinct leaves a world and the share solved exactly, the
//! leaf solves' nodes and thread time a world, and the hybrid's thread time a world with and
//! without them (JSON `exact_leaves`); `values.hybrid` ends `+exact:<policy>`. Golden-set form
//! only (not `--saved`, `--explain` or `--truth net`). `hybrid-paired` also prints both checks'
//! thread time a world.
//! **Ensembles** (`nn::ensemble`): `--net` takes several networks of the cut's trick, comma-separated
//! or repeated; the leaf value is the integer mean of their money chances (u128 totals per team,
//! largest-remainder rounding to exactly 2^32, ties to the lower team), each member valuing its
//! batches with its own kernel. One network is exactly that network (the same values and report);
//! an ensemble's JSON adds `ensemble` (its id `ens<N>:<sha>+…` and the members) and names it in
//! `values.hybrid`. Not with `--saved` or `--explain`.
//! **The optimizer's-curse test** (`search::curse`): `--truth net` takes the network itself as
//! the truth (H_K with its noise-free values, recomputed; no `--worlds`, `--reference`,
//! `--saved`, `--explain`) and the player is the same search with seeded zero-mean noise on each
//! leaf's money vector, SD `--noise-sd-by-class` (five percentages of money chance, race classes
//! 0-4) keyed by (`--noise-seed`, decision, world, position); the report adds the noise asked for
//! and realised per race class (JSON `noise`); `--noise-calibrate N` first rescales the SDs asked
//! for on a sample of N decisions (every k-th; `--noise-calibrate-worlds` worlds, 8;
//! `--noise-calibrate-rounds` rounds, 4) so the realised SD per class matches the SDs given,
//! which become the targets (JSON `noise_calibration`); the tables are the usual ones (decision loss
//! by band and position, every team's bias per position). With zero noise the loss and every bias
//! are exactly 0.
//! `hybrid-paired` compares two checks' JSONs on the same decisions (`search::hybrid::paired`):
//! mean decision loss and own-team bias, overall, by band, seat 1 and seat 2, for A and B, B − A
//! and its bootstrap 95% interval (decisions resampled within each group, the same for both).
//! `teacher --trick` writes the student's teacher records (docs/STUDENT.md §3–§4,
//! `search::student_teacher`): decisions `--first`..`--first + --count` at trick 1 to 4 from
//! `--source` (stratum `i mod 4 + 1`, place `(i div 4) mod 8`), each valued by PIMC over deals
//! `--deal-from`..`--deal-from + --deals` with the hybrid H_k (`--net`, an O_k with k after the
//! trick; `exact` for exact search, tests and small runs), every candidate in every deal stored:
//! `SHARD.npy` (records), `SHARD.values.npy` (values) and `SHARD.npy.json` (stamps, settings,
//! both files' SHA-256), written as `.part` files and renamed once whole. Byte-identical for any
//! `--threads`. `--role test` (the fixed test set) needs `student::STUDENT_TEST_SEED`, which a
//! training shard may not use; seed 1 is refused. `--progress FILE` is rewritten every 2 s.
//! `student-check` measures a student network (`crates/student`, docs/STUDENT.md §7) against the
//! teacher's stored values: the teacher shards (the fixed test set, or any) are merged by deal
//! index as `student encode` merges them, each decision is encoded and valued by the integer
//! student, and the report gives the decision loss (the teacher's best mean minus the mean of the
//! student's choice, deciding pair, % of money chance) with a bootstrap 95% upper bound,
//! agreement and the student's value error, overall, by trick, by place and by stratum;
//! `--baseline rules` adds the same table for the rules bot (`rules-v1`) on the same decisions,
//! `--baseline random` for a random legal card (`student::check::judge_random`: a seeded pick per
//! decision, `--boot-seed`, with the exact expected loss of a random pick printed beside it); the
//! JSON's `baselines` lists each one asked for (`baseline` stays the rules bot's).
//! `--split-half` and `--top-k K` add the measures of docs/STUDENT.md §14 after everything else
//! (`search::curves::split_half`, `top_k_loss`): the noisy oracle's floor, the student's loss on
//! each half and the unbiased gap (deals split by deal-index parity, chosen on one half, scored on
//! the other, both ways), and the loss when the teacher picks among the student's top K (all
//! deals, and split-half with `--split-half`); the JSON gains `top_k` and `split_half`. Without
//! them the output is as before.
//! `--golden-decisions D --golden-worlds W` (instead of `--teacher`; `--trick` 2, `--to N` the
//! first N worlds) scores the student against a stored golden set's **exact** worlds (the files
//! `hybrid-check --decisions/--worlds` read, e.g. the trick-2 golden set, 2,000 decisions x 320
//! worlds; `search::student_teacher::golden_merged`): each decision's app candidates take their
//! exact values world by world, decisions a student never meets (one candidate, the play after
//! the seat's own rob) are skipped and counted, and the report is the same (the loss is the exact
//! best mean less the student's choice's; split-half halves are even and odd worlds). The worlds'
//! equity table must be `--equity`'s; the JSON's `role` is `golden` and `golden` names the files
//! and their SHA-256s.
//! `labels` writes training labels for the oracle network O_K (docs/M3-HYBRID-SPEC.md §5,
//! `search::oracle_labels`): a label shard (`nn::shard`, `.npy` records: encoded features, the exact
//! value as four u64, the source tag, the trick) and a JSON sidecar `FILE.npy.json` with the
//! encoder, values version, tie rule, this binary's SHA-256, the equity table's SHA-256, the seed,
//! counts and the shard's SHA-256. Byte-identical for any `--threads` and `--chunk-rows`; seed 1
//! (the golden sets') is refused. The shard is streamed
//! (`search::oracle_labels::write_label_shard`): chunks of `--chunk-rows` rows are made on every
//! thread, up to 1 GiB of finished rows ahead of the next one written (`label_window`: one slow
//! row never stops the other threads, whatever the chunk size), then hashed and written in row
//! order as they come, under the name `FILE.npy.part`, renamed to `FILE.npy` once whole. `--tt-mb M` sets the exact label solves' table (megabytes a
//! thread; default `oracle_labels::tt_mb_for` the trick, 256 at trick 2): the runner lowers it
//! when the machine's memory would not hold one per thread; it never changes a row, only the
//! solves' node counts (`solver_nodes`; the sidecar then records `label_tt_mb`). A harvest's own
//! search keeps its fixed table. The start states (`--start-hands`; `decisions` and
//! `hybrid-check` too) are played on `--threads` threads, the same states for any count
//! (`search::measure::collect_starts_with`). `--source harvest` (spec §5 source (c),
//! `search::oracle_labels::harvest_rows`): training decisions drawn by the yardstick's generator at the tricks `--harvest-tricks`
//! (default K−1), a world from the decider's belief each, the frontier positions the hybrid's
//! search from the decision reaches (exact values at the frontier by default), deduplicated by
//! canonical form, at most `--harvest-cap` per decision (a seeded sample); `--harvest-positions` gives the
//! shares of decisions for the decider's positions 1–2, 3–4 and 5–8 (default uniform over the
//! eight positions). `--harvest-net WEIGHTS` (an integer O_K of an earlier round, as
//! `hybrid-check` takes it) drives the search with that network's values at the frontier instead
//! of exact ones, and only the sampled positions are solved exactly, for their labels: the way to
//! harvest at trick 3, where solving every reached position is too dear. The sidecar adds the
//! harvest's settings, the driver (and the network's SHA-256) and counts per position, stratum
//! and decision trick. `--harvest-strata 4` (or `2,4`) draws the decisions from those strata
//! only: decision `j` from stratum `strata[j mod s]`, its decider at `schedule[(j div s) mod
//! cycle]` (`s` strata; all four, the default, is the yardstick's `j mod 4 + 1`), so the rows stay
//! a pure function of (seed, `j`) and the settings. `--harvest-race-classes 2,4` makes a race
//! harvest (docs/m3-hybrid/rung1.md, "Why seats 3-4 fail"): the reached positions of those race
//! classes (`nn::outcome::race_class`: 0 no game can end, 1 one team can win the game, 2 several
//! can, 3 one team can win the match, 4 a team can win the match and another the game) are a
//! seeded sample of at most `--harvest-race-cap` (256) per decision, the other classes' of at most
//! `--harvest-cap` (which may then be 0: race positions only). `--harvest-keep PRED.json` makes a
//! targeted harvest (docs/m3-hybrid/audit-loop.md §4 step 5; not with a race harvest): the
//! reached positions inside any group of the predicate file (`nn::predicate`, decided on each
//! position's encoded features) are a seeded sample of at most `--keep-cap` (256) per decision,
//! drawn first on the decision's stream, the others of at most `--harvest-cap` (0: in-group
//! positions only); the sidecar's `harvest.keep` gives the file's SHA-256 and the rows and
//! reached positions per group. `--max-decisions N` bounds the decisions a harvest plays (a
//! targeted harvest without it: `oracle_labels::default_max_decisions`, max(1000, ⌈50 × count /
//! keep-cap⌉)); a harvest stopped by its bound writes the rows it found (the first decisions'
//! rows, the same for any thread count; the shard is otherwise byte-identical), and a harvest's
//! sidecar says `rows` (written), `decisions_played`, `max_decisions` and `bounded`. Every shard's sidecar counts its
//! rows by race class (`rows_by_race_class`); a harvest's records its strata and race settings.
//! `--seat-weights w12,w34,w58` (harvest only; the audit loop's leader focus,
//! docs/m3-hybrid/audit-loop.md §10) gives each row the loss weight of its decision's seat band
//! (positions 1–2, 3–4, 5–8; multiples of 1/256): the per-row weights `FILE.npy.rowweights`
//! (`nn::rowweights`), written before the shard is renamed into place, and `seat_weights` in the
//! sidecar (the weights, the file's SHA-256, the rows by position).
//! `--harvest-schedule` (harvest only) says how the work is shared out over the threads, which
//! never changes a row, a decision's facts or their order (`search::oracle_labels::HarvestSchedule`):
//! `pooled` (the default) searches and samples each decision on one thread and pools the sampled
//! positions of every decision taken into one list whose exact labels every thread takes in turn
//! (one position a unit at trick 2, a decision's at tricks 3-5: `label_group_for`), so no thread
//! waits on one decision's labels; `decision` (before 2026-10-17, for comparisons) labels each
//! decision's positions on the thread that searched it. Only the label solves' node counts at
//! trick 2 (`solver_nodes`) depend on it. The sidecar's `harvest.label_solves` records it.
//! `--label-net WEIGHTS` (an integer O_{K+1}, any source; docs/m3-hybrid/audit-loop.md §12)
//! labels each row with the hybrid H_{K+1} instead of its exact value
//! (`search::oracle_labels::hybrid_value`: the exact solver through trick K, the network at the
//! start of trick K+1, as `hybrid-check` and `worlds --method hybrid` run it; the same fixed-point
//! value, tie rule and row format, the outcome label the possible outcome nearest the value);
//! `--exact-share X` (0 to 1, six decimals; default 0) labels that share of the rows exactly all
//! the same, drawn by each row's key (`oracle_labels::anchored`). The sidecar then says
//! `label_source` `hybrid`, the network's SHA-256 (`label_net_sha256`), `values_version`
//! `"<exact version>+O<K+1>:<sha256>"` (an exact shard keeps the plain number, so a reader of
//! exact labels refuses hybrid ones), and the rows by label (`rows_hybrid`, `rows_exact`).
//! Without it every row is exact, byte for byte as before, and the sidecar says
//! `label_source` `exact`. The runner never asks for hybrid labels on a validation, audit or
//! golden set.
//! `--exact-leaves SPEC` (with `--label-net` only; the syntax of `hybrid-check --exact-leaves`:
//! `race:LIST`, `band:LO-HI`, `learned:MODEL:t=T|budget=S`, `all`, `none`, joined by `+`) solves
//! the hybrid's trick-(K+1) frontier positions the policy picks exactly
//! (`search::exact_leaves::ExactLeaves`, the wrapper `hybrid-check` uses) and values the others
//! with the network; a band reads the money chance of the team to lead the labelled position; a
//! learned model must be one made for trick K+1 leaves of this network. `none` (the default) is
//! the hybrid labels as before, byte for byte; `all` gives the exact labels' values. Each leaf
//! solve has a table of `--exact-leaf-tt-mb` (default the smaller of `--tt-mb` and the trick
//! K+1 table; never changes a value). The sidecar then records `label_exact_leaves` (the
//! policy's normal form), `label_leafrisk_sha256` (a learned term's model), `exact_leaves`
//! (leaves seen and solved exactly, their nodes) and a values version
//! `"<exact>+O<K+1>:<sha256>+exact:<policy>[+leafrisk:<model sha256>]"`
//! (`oracle_labels::hybrid_values_version`).
//! `hybrid-check` measures the hybrid H_K (exact to the start of trick K, the oracle network
//! there) against exact search on the same worlds (docs/M3-HYBRID-SPEC.md §3, `search::hybrid`):
//! decision loss with its bootstrap 95% upper bound, value error and agreement, overall, per
//! position band and per position. Decisions come from `--decisions` (those at trick T) or are
//! built like `search decisions` (`--count`, `--seed`); worlds are regenerated as the yardstick
//! draws them (`--seed`, `--label`, worlds `0..--to`). Exact values are recomputed, or read from
//! `--worlds` files (a `--verify-pct` sample recomputed first and required byte-identical;
//! `--recompute` ignores them). Worlds files made before the belief sampler became the default
//! (no world counts, e.g. the trick-4 golden set `m3-agree-t4c`) were drawn by the rejection
//! sampler: check them with `--source rejection`.
//! With `--saved` (docs/M3-HYBRID-SPEC.md §3 "Trick-1 checks", `search::saved`) the worlds are
//! those earlier runs saved exact values for (worlds files or their `.ckpt` checkpoints, each
//! labelled by its run), split by the burn rule that valued them (values version 1: the
//! `rules-v1` burn, checked with `rules_robber_burn`; the current version: searched); the sets run
//! at once on `--threads` shared out by worlds, one world per thread, the largest saved first;
//! each world's line goes to stderr as it finishes. `--verify-pct` (default 0 here) recomputes
//! that share of the saved worlds first (`--verify-rule` picks the sets), each solve on
//! `--verify-solve-threads` threads with a `--verify-tt-mb` table, and a set with any difference
//! is not compared. The report: per world, per decision, the §3 tables per set, the `--hard`
//! worlds with their saved exact times, the peak memory; `--plan-only` prints the sets and the
//! verification sample and solves nothing.
//! `--explain IDS` (not with `--saved`) runs, instead of the check, the explanation of those
//! decisions (`search::hybrid_explain`): exact means in both tie orders and hybrid means per
//! candidate, how the hybrid's misranking spreads over the worlds and over race classes, the
//! network's error at every frontier position the hybrid reached (per race class, tie-decided
//! or not), and for the `--explain-worlds` most misleading worlds the frontier positions each
//! search copied up to each candidate, with the network's outcome probabilities there;
//! `--explain-json FILE` writes every world (the principal positions of the
//! `--explain-worlds` most misleading ones only).
//! `--slices PRED.json` (not with `--explain`; docs/m3-hybrid/audit-loop.md §3 item 2,
//! `search::hybrid::slices`): the hybrid's own search in every world also records the distinct
//! frontier positions it reaches and their groups (never changing a value); a decision is in a
//! group when at least `--slice-share` (0.5) of the positions its worlds' searches reach are in
//! it. The report adds the §3 table per group and for the decisions in no group (reported, not
//! judged), and the standard then also asks every group's slice with decisions to pass the
//! decision-loss bound (JSON: `slices`, `pass` with them, `pass_without_slices`). With `--saved`
//! (the audit loop's rough test at trick 2) each set gets its own slices, in its JSON entry
//! under `sets`.
//! **A silver reference** (docs/m3-hybrid/silver.md): `--reference silver --reference-net
//! WEIGHTS` checks H_K against a deeper hybrid H_R (`--reference-cut R`, by default the network's
//! trick, after `--cut`) instead of exact search, on the same worlds: its values are read from
//! `--worlds` files of `hybrid` records made with that very network (`search worlds --method
//! hybrid`; another network's are refused, a `--verify-pct` sample recomputed with H_R and
//! required byte-identical), or recomputed when no `--worlds` is given. A silver file is refused
//! by the default `--reference exact` (it is never an exact golden set), and exact records by
//! `--reference silver` unless `--allow-exact` (they then stand in for the silver values of their
//! decisions, the anchor set; the report counts them). The report says "H_K vs silver (H_R, net
//! …)", the JSON `reference` (`kind`, `cut`, the network, `values`, how many decisions against
//! each kind) and `values.reference`; a decision's `exact_*` fields hold its reference's values
//! (`silver` true). Not with `--saved` or `--explain`.
//! `--race-groups`: the §3 tables also per stratum of the hand's start state (S1–S4, the score
//! groups) and per race class (`nn::outcome::race_class_of`, the classes the label shards count)
//! of the frontier positions the hybrid's own search reaches (a decision is in a class when at
//! least `--race-share` of its reached positions are, else in "mixed"; `search::hybrid::
//! race_groups`); a group is judged only with at least `--min-group` decisions, and none of
//! them is part of the verdict (JSON `score_groups`, `race_groups`). `--progress FILE` (the
//! golden-set form too) is rewritten every 2 s with the reference solves and hybrid worlds done.
//! `--standard PCT` (default 0.01, a percentage of money chance; owner, 2026-10-16 night: the
//! audit loop's push standards 0.005 at trick 4 and 0.0075 at trick 3) is the standard every
//! verdict uses: the overall, band and position rows' decision-loss bound and bias, and every
//! slice; the report prints it and the JSON records it (`standard_pct`, `standard`).
//! `decisions` stratifies equally over the 4 strata and 8 positions, so `--count` must be a
//! multiple of 32 (`--allow-uneven` overrides); it prints the position × stratum table it made.
//! `decisions --keep PRED.json` draws a mini golden set by rejection (docs/m3-hybrid/
//! audit-loop.md §3 item 3; `search::audit::keep_decisions`): candidates are the generator's
//! decisions `--first-candidate`, the next, … (decision `i` is the one `search decisions` makes
//! as its `i`th with the same `--seed`, and keeps `i` as its id, so `search worlds` and
//! `hybrid-check` draw its worlds as for any decision set); each is searched on its worlds
//! `0..--keep-worlds` (`--label`), by the hybrid with the network `--net` (an O_cut) at the
//! frontier of trick `--cut`, or with exact values there when no network is given (cheap at
//! trick 4); a candidate is kept when at least `--keep-share` of the distinct frontier positions
//! its searches reach lie in one group of the predicate file, until `--count` are kept or
//! `--max-candidates` drawn. The file holds the kept decisions in the ordinary format;
//! `FILE.json` records the predicate file (SHA-256), the share, the frontier's driver, the
//! candidates drawn, and each kept decision's reach by group. Byte-identical for any `--threads`.
//! `worlds --method hybrid --cut K --net WEIGHTS` values every candidate by the hybrid H_K (exact
//! search to the start of trick K, the integer network there; decisions before trick K only):
//! a **silver reference** (docs/m3-hybrid/silver.md), never an exact golden set. Its records are
//! `hybrid` with `fp.frontier` (the cut, the network's SHA-256, the encoder) and so are its
//! checkpoint lines; a run never resumes, extends or joins records of another method or network.
//! The worlds are the yardstick's (seed, label, decision, world), so a silver set and an exact
//! set of the same decisions value the same deals; the file is byte-identical for any thread
//! count and any interruption, and its timing file counts each decision's network evaluations
//! (`evals`). Under `--schedule auto` a hybrid run takes one world per thread.
//! `score --yardstick` takes exact records only (a silver one with `--allow-silver`, said in the
//! report); `agree` refuses files mixing methods and notes a silver reference's.
//! `worlds` and `teacher` take `--source belief|rejection`: the world generator, by default
//! `crates/belief`'s counted uniform sampler (which also records each decision's exact world
//! count); `rejection` is the engine-based oracle (same distribution, no counts).
//! `worlds` is the yardstick builder and the method runner: every (decision, world) task of the
//! run goes through one work queue on `--threads` threads (longest expected first), and one JSON
//! line per decision is appended to `--out` in that order; each world is also appended to the
//! checkpoint `<out>.ckpt` as soon as it finishes. A rerun skips the decisions already in `--out`
//! and the worlds already in the checkpoint, so an interrupted chunk loses only the worlds that
//! were being solved; the checkpoint is deleted once every decision is in `--out`. The file is
//! byte-identical for any thread count and any interruptions; timings go to
//! `<out>.times.jsonl`. `--progress FILE` is rewritten every 2 s with the counts done, the worlds
//! restored at the start, and the progress estimate (`eta`: work done weighted by each world's
//! expected cost, refined by measured times, and the seconds left; `search::eta`) for the
//! runner's heartbeat. Exact worlds are the yardstick (label 1); a method's own worlds use
//! another label. `--ids` restricts the run to chosen decisions; with `--from-world A` (A = the
//! worlds already saved) it extends them: the new worlds are appended as new records (to the same
//! file or another), and `agree`, `curves` and `score` join every record of a decision, so they
//! see exactly the rows one longer run would have written (§13.1 item 10).
//! `--schedule auto` gives every exact world its own threads and table by position on `--threads`
//! threads within `--tt-budget-mb` (`search::plan`, §13.1 item 11); the file is the same bytes.
//! `bench-t1` is one timed window of the trick-1 machine bench (`search::t1bench`): `fill` (one
//! world per thread on `--threads` threads, positions spread evenly, `--tt-mb` tables) or `prod`
//! (the position-aware schedule within `--tt-budget-mb`, on every decision × `--worlds-per-decision`
//! worlds). It stops every solve at the window's end, or earlier when the guard sees low CPU or
//! idle threads, or when `--stop-file` appears; it writes every world's nodes (stopped ones so
//! far) and value hash, the samples, CPU time and peak memory to `--json`.
//! `agree` reports N against 2N on the grid, per trick, per position in the trick (§13.1 item 6),
//! per match-state stratum and per log2 world-count band (item 8) and per hand-type band (item 9:
//! trumps, top trumps, longest plain suit; and the hand features added on 2026-10-14, holds the 5
//! of trumps, robber, void plain suits). Worlds records made before those features existed lack
//! them: `--decisions FILE` (the decision set, repeatable) replays each decision from its deck and
//! actions and labels it again, with no new solves; `agree` and `curves` both take it.
//! `curves` turns the saved rows into money-loss curves on a fine N grid, with power-law fits,
//! held-out checks and the N needed per money-loss threshold (item 10, `search::curves`);
//! `score` reports per position and stratum.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use search::bot::{PimcFactory, PimcSpec, load_equity, load_equity_sha};
use search::curves::{CurveParams, N_CAP, TrickReport, analyse, score_groups};
use search::measure::{
    BatchProgress, DecisionSpec, DecisionTiming, Schedule, WorldRows, WorldsJob, agreement_curve,
    build_decisions, collect_starts, collect_starts_with, feature_band_name, paired, parse_ids,
    read_jsonl, read_worlds, relabel, score, stratum_of, summarise, timing_path, worlds_file_net,
};
use search::pimc::{Ctx, Method, SearchConfig, SharedLeaf};
use search::plan::AutoParams;
use search::t1bench::{Guard, Mode, Sample, WindowParams, run_window};
use search::teacher::{decisions_npy, teacher_deck, teacher_hand};
use search::worlds::{BeliefSource, RejectionSource, WorldSource};

struct Args(Vec<String>);

impl Args {
    fn get(&self, k: &str) -> Option<&str> {
        self.0
            .iter()
            .position(|a| a == k)
            .and_then(|i| self.0.get(i + 1))
            .map(String::as_str)
    }
    fn all(&self, k: &str) -> Vec<&str> {
        self.0
            .iter()
            .enumerate()
            .filter(|(_, a)| *a == k)
            .filter_map(|(i, _)| self.0.get(i + 1).map(String::as_str))
            .collect()
    }
    fn num(&self, k: &str, d: u64) -> Result<u64, String> {
        self.get(k).map_or(Ok(d), |v| {
            v.parse().map_err(|_| format!("{k}: not a number: {v}"))
        })
    }
    fn req(&self, k: &str) -> Result<&str, String> {
        self.get(k).ok_or_else(|| format!("missing {k}"))
    }
    fn has(&self, k: &str) -> bool {
        self.0.iter().any(|a| a == k)
    }
}

/// `--source belief|rejection` (default `belief`): the world generator.
fn world_source(a: &Args) -> Result<Arc<dyn WorldSource>, String> {
    match a.get("--source").unwrap_or("belief") {
        "belief" => Ok(Arc::new(BeliefSource::new())),
        "rejection" => Ok(Arc::new(RejectionSource::new())),
        s => Err(format!("unknown --source {s} (belief or rejection)")),
    }
}

fn read_lines<T: serde::de::DeserializeOwned>(path: &str) -> Result<Vec<T>, String> {
    read_jsonl(Path::new(path))
}

fn main() -> ExitCode {
    let a = Args(std::env::args().skip(1).collect());
    let r = match a.0.first().map(String::as_str) {
        Some("decisions") => decisions(&a),
        Some("worlds") => worlds(&a),
        Some("agree") => agree(&a),
        Some("curves") => curves(&a),
        Some("score") => score_cmd(&a),
        Some("teacher") => teacher(&a),
        Some("bench-t1") => bench_t1(&a),
        Some("robber-burn") => robber_burn(&a),
        Some("labels") => labels_cmd(&a),
        Some("risk-profile") => risk_profile_cmd(&a),
        Some("risk-filter") => risk_filter_cmd(&a),
        Some("net-error") => net_error_cmd(&a),
        Some("hybrid-check") => hybrid_check(&a),
        Some("hybrid-paired") => hybrid_paired(&a),
        Some("student-check") => student_check(&a),
        _ => Err(
            "usage: search decisions|worlds|agree|curves|score|teacher|bench-t1|robber-burn|labels|risk-profile|hybrid-check|hybrid-paired|student-check … (see the source docs)"
                .into(),
        ),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("search: {e}");
            ExitCode::from(2)
        }
    }
}

/// Decisions per stratified block: 4 strata × 8 positions.
const BLOCK: u64 = 32;

fn decisions(a: &Args) -> Result<(), String> {
    if a.has("--keep") {
        return keep_decisions_cmd(a);
    }
    for f in [
        "--keep-share",
        "--keep-worlds",
        "--net",
        "--cut",
        "--first-candidate",
        "--max-candidates",
    ] {
        if a.has(f) {
            return Err(format!("{f} is for --keep PRED.json (a mini golden set)"));
        }
    }
    let trick = a.num("--trick", 2)? as u8;
    let count = a.num("--count", 7 * BLOCK)?;
    let seed = a.num("--seed", 1)?;
    if count % BLOCK != 0 && !a.has("--allow-uneven") {
        let (lo, hi) = (count / BLOCK * BLOCK, (count / BLOCK + 1) * BLOCK);
        let use_ = if lo == 0 {
            hi.to_string()
        } else {
            format!("{lo} or {hi}")
        };
        return Err(format!(
            "--count {count} is not a multiple of {BLOCK}: decision i is stratum i mod 4 at \
             position (i / 4) mod 8, so positions and strata would not be covered equally \
             (use {use_}, or pass --allow-uneven)"
        ));
    }
    let threads = a.num("--threads", 1)?.max(1) as usize;
    let starts = collect_starts_with(a.num("--start-hands", 20_000)?, seed, threads);
    let ds = build_decisions(trick, count, seed, &starts);
    print_cells(trick, &ds);
    let mut out = String::new();
    for d in &ds {
        out.push_str(&serde_json::to_string(d).map_err(|e| e.to_string())?);
        out.push('\n');
    }
    std::fs::write(a.req("--out")?, out).map_err(|e| e.to_string())
}

/// The position × stratum table of a decision set.
fn print_cells(trick: u8, ds: &[DecisionSpec]) {
    let mut cells = [[0usize; 4]; 8];
    for d in ds {
        cells[usize::from(d.position)][usize::from(d.stratum - 1)] += 1;
    }
    println!(
        "trick {trick}: {} decisions by position (rows) and stratum (columns)",
        ds.len()
    );
    println!("        S1    S2    S3    S4  total");
    for (p, row) in cells.iter().enumerate() {
        println!(
            "pos{}  {:>4}  {:>4}  {:>4}  {:>4}  {:>5}",
            p + 1,
            row[0],
            row[1],
            row[2],
            row[3],
            row.iter().sum::<usize>()
        );
    }
}

/// `search decisions --keep PRED.json`: a mini golden set by rejection sampling (module notes).
fn keep_decisions_cmd(a: &Args) -> Result<(), String> {
    use search::audit::{KeepJob, keep_decisions};
    let trick = a.num("--trick", 2)? as u8;
    let cut = a.num("--cut", u64::from(trick) + 1)? as u8;
    if !(2..=5).contains(&cut) || trick == 0 || trick >= cut {
        return Err(format!(
            "--cut {cut} must be 2..=5 and after --trick {trick} (and --trick at least 1)"
        ));
    }
    let pred_path = a.req("--keep")?;
    let preds = nn::predicate::Predicates::load(Path::new(pred_path))?;
    if preds.is_empty() {
        return Err(format!("--keep {pred_path}: the file has no groups"));
    }
    let share_text = a.get("--keep-share").unwrap_or("0.5");
    let share = nn::predicate::parse_share(share_text).map_err(|e| format!("--keep-share: {e}"))?;
    let count = a.num("--count", 50)?;
    let threads = a.num("--threads", 1)?.max(1) as usize;
    let seed = a.num("--seed", 1)?;
    let job = KeepJob {
        trick,
        cut,
        seed,
        label: a.num("--label", 1)?,
        worlds: a.num("--keep-worlds", 8)?.max(1),
        share,
        count,
        first: a.num("--first-candidate", 0)?,
        max_candidates: a.num("--max-candidates", count.saturating_mul(200))?,
        threads,
        tt_mb: a.num("--tt-mb", 64)?.max(1) as usize,
    };
    let t0 = Instant::now();
    let (table, table_sha) = load_equity_sha(a.req("--equity")?)?;
    let leaf: Arc<SharedLeaf> = Arc::new(table);
    let oracle = match a.get("--net") {
        Some(p) => {
            let o = nn::Oracle::load(Path::new(p), leaf.clone())?;
            if o.trick != cut {
                return Err(format!(
                    "--net {p} was trained for trick {} positions, not trick {cut}",
                    o.trick
                ));
            }
            Some((o, p.to_string()))
        }
        None => None,
    };
    let start_hands = a.num("--start-hands", 20_000)?;
    let starts = collect_starts_with(start_hands, seed, threads);
    let src = world_source(a)?;
    let ctx = Ctx {
        leaf: &*leaf,
        source: &*src,
        rules: engine::game::Rules::HOUSE,
    };
    let net: Option<&dyn solver::frontier::Frontier> = oracle
        .as_ref()
        .map(|(o, _)| o as &dyn solver::frontier::Frontier);
    eprintln!(
        "decisions: mini golden set at trick {trick}, frontier at trick {cut} ({}), {} groups of \
         {pred_path} (sha256 {}), share {share_text}, {} worlds a candidate, {count} wanted from \
         candidate {} (at most {}), {threads} threads",
        oracle
            .as_ref()
            .map_or("exact values".to_string(), |(o, _)| format!(
                "network {}",
                &o.sha256[..12]
            )),
        preds.len(),
        &preds.sha256[..12],
        job.worlds,
        job.first,
        job.max_candidates
    );
    let done = std::sync::atomic::AtomicU64::new(0);
    let finished = AtomicBool::new(false);
    let t1 = Instant::now();
    let run = std::thread::scope(|s| {
        let (done, finished) = (&done, &finished);
        s.spawn(move || {
            let mut last = Instant::now();
            while !finished.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(200));
                if last.elapsed() >= Duration::from_secs(20) {
                    last = Instant::now();
                    eprintln!(
                        "decisions: {} candidates searched, {:.0} s",
                        done.load(Ordering::Relaxed),
                        t1.elapsed().as_secs_f64()
                    );
                }
            }
        });
        // Stops the progress thread however the search ends (a panic included).
        struct Finish<'a>(&'a AtomicBool);
        impl Drop for Finish<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        let _finish = Finish(finished);
        keep_decisions(&job, &preds, &starts, &ctx, net, Some(done))
    });
    let wall = t1.elapsed().as_secs_f64();
    let ds: Vec<DecisionSpec> = run.kept.iter().map(|k| k.d.clone()).collect();
    print_cells(trick, &ds);
    let names = preds.describe();
    for (g, line) in names.iter().enumerate() {
        println!(
            "group {line}: {} of {} candidates in it ({:.1}% of the reached positions), {} kept",
            run.drawn_in_group[g],
            run.drawn,
            100.0 * run.reach.share(g),
            run.kept
                .iter()
                .filter(|k| k.groups.contains(&(g as u32)))
                .count()
        );
    }
    let complete = ds.len() as u64 >= count;
    println!(
        "kept {} of {count} wanted after {} candidates ({:.1} s wall, {:.1} thread-s searching: \
         {:.2} thread-s per kept decision){}",
        ds.len(),
        run.drawn,
        wall,
        run.search_time.as_secs_f64(),
        run.search_time.as_secs_f64() / ds.len().max(1) as f64,
        if complete {
            String::new()
        } else {
            format!(
                "; INCOMPLETE: --max-candidates {} reached",
                job.max_candidates
            )
        }
    );
    let out = a.req("--out")?;
    let mut text = String::new();
    for d in &ds {
        text.push_str(&serde_json::to_string(d).map_err(|e| e.to_string())?);
        text.push('\n');
    }
    std::fs::write(out, text).map_err(|e| format!("{out}: {e}"))?;
    let side = serde_json::json!({
        "what": "a mini golden set (docs/m3-hybrid/audit-loop.md §3 item 3): decisions of the \
                 yardstick's generator (search::measure::build_decision_at: decision i of seed \
                 `seed` at trick `trick`, id i), kept when at least `share` of the distinct \
                 frontier positions that the hybrid's searches over the decision's worlds \
                 0..worlds reach lie in one group of the predicate file (nn::predicate)",
        "predicates_file": pred_path,
        "predicates_sha256": preds.sha256,
        "groups": preds.groups.iter().enumerate().map(|(g, x)| serde_json::json!({
            "name": x.name,
            "conditions": x.conditions_text(),
            "candidates_in_group": run.drawn_in_group[g],
            "kept": run.kept.iter().filter(|k| k.groups.contains(&(g as u32))).count(),
            "reached_share": run.reach.share(g),
            "audit": x.meta,
        })).collect::<Vec<_>>(),
        "share": share_text,
        "share_millionths": share,
        "trick": trick,
        "cut": cut,
        "seed": seed,
        "label": job.label,
        "worlds_searched": job.worlds,
        "world_source": src.name(),
        "frontier": match &oracle {
            None => serde_json::json!({ "kind": "exact" }),
            Some((o, p)) => serde_json::json!({ "kind": "net", "sha256": o.sha256, "file": p,
                                                "trick": o.trick, "widths": o.net().widths() }),
        },
        "tt_mb": job.tt_mb,
        "first_candidate": job.first,
        "max_candidates": job.max_candidates,
        "candidates_drawn": run.drawn,
        "count_wanted": count,
        "kept": ds.len(),
        "complete": complete,
        "decisions": run.kept.iter().map(|k| serde_json::json!({
            "id": k.d.id, "position": k.d.position + 1, "stratum": k.d.stratum,
            "reached": k.reach.reached, "in_group": k.reach.in_group, "groups": k.groups,
        })).collect::<Vec<_>>(),
        "start_hands": start_hands,
        "binary_sha256": binary_sha256(),
        "equity_sha256": table_sha,
        "seconds": ((t0.elapsed().as_secs_f64()) * 10.0).round() / 10.0,
        "search_thread_seconds": run.search_time.as_secs_f64(),
    });
    write_json(&format!("{out}.json"), &side)
}

/// Rewrite the progress file (write a temporary file, then rename it over the old one, so a
/// reader never sees half a file).
fn write_progress(
    path: &Path,
    p: &BatchProgress,
    job: &WorldsJob,
    cfg: &SearchConfig,
    t0: Instant,
    finished: bool,
) {
    let ld = |x: &std::sync::atomic::AtomicU64| x.load(Ordering::Relaxed);
    let v = serde_json::json!({
        "method": job.method_name(),
        "label": job.label,
        "from": job.from,
        "to": job.to,
        "threads": ld(&p.threads),
        "solve_threads": cfg.solve_threads,
        "decisions_done": ld(&p.decisions_done),
        "decisions_total": ld(&p.decisions_total),
        "worlds_done": ld(&p.worlds_done),
        "worlds_total": ld(&p.worlds_total),
        "decisions_resumed": ld(&p.decisions_resumed),
        "worlds_resumed": ld(&p.worlds_resumed),
        "worlds_valued": ld(&p.worlds_valued),
        "nodes_valued": ld(&p.nodes_valued),
        "evals_valued": ld(&p.evals_valued),
        "eta": p.eta().map(|e| serde_json::json!({
            "done_frac": e.done_frac, "remaining_s": e.remaining_s, "measured": e.measured,
            "basis": e.basis, "work_thread_s": e.work_s, "done_thread_s": e.done_s,
        })),
        "par_tasks_done": ld(&p.par.tasks_done),
        "par_tasks_total": ld(&p.par.tasks_total),
        "planned": p.planned.load(Ordering::Relaxed),
        "running": ld(&p.exec.running),
        "busy_threads": ld(&p.exec.busy_threads),
        "tail": p.exec.tail.load(Ordering::Relaxed),
        "elapsed_s": t0.elapsed().as_secs(),
        "finished": finished,
    });
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    if std::fs::write(&tmp, v.to_string()).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

/// The exact-leaves policy of `--{p}exact-leaves SPEC` (`search::exact_leaves`), with each leaf
/// solve's table `--{p}exact-leaf-tt-mb MB` (default by the trick) and the expert of the picked
/// leaves `--{p}defer-expert exact|hK:NET[:SPEC]` (default exact), for a hybrid whose frontier
/// is at the start of trick `cut` (`p` is `""` or `"reference-"`). `None` without the flag or
/// with `none`. Refused: the table or the expert without leaves, a learned model made for
/// another trick's leaves, a deeper expert that does not look at trick `cut + 1` or whose own
/// learned model is not its network's.
fn leaf_policy_arg(
    a: &Args,
    p: &str,
    cut: u8,
    leaf: &Arc<SharedLeaf>,
) -> Result<Option<search::exact_leaves::LeafPolicy>, String> {
    let (k_el, k_tt, k_de) = (
        format!("--{p}exact-leaves"),
        format!("--{p}exact-leaf-tt-mb"),
        format!("--{p}defer-expert"),
    );
    let Some(text) = a.get(&k_el) else {
        for k in [&k_tt, &k_de] {
            if a.has(k) {
                return Err(format!("{k} is for {k_el}"));
            }
        }
        return Ok(None);
    };
    let tt = a.num(&k_tt, search::oracle_labels::tt_mb_for(cut) as u64)? as usize;
    let Some(mut pol) = search::exact_leaves::LeafPolicy::parse(text, tt)? else {
        if a.has(&k_de) {
            return Err(format!(
                "{k_de} defers the leaves {k_el} picks: none are picked"
            ));
        }
        return Ok(None);
    };
    if let Some(e) = a.get(&k_de) {
        let tt_e = search::oracle_labels::tt_mb_for(cut + 1);
        pol = pol.with_expert(search::exact_leaves::Expert::parse(e, cut, leaf, tt_e)?);
    }
    pol.check_for(cut).map_err(|e| format!("{k_el}: {e}"))?;
    Ok(Some(pol))
}

fn worlds(a: &Args) -> Result<(), String> {
    let ds: Vec<DecisionSpec> = read_lines(a.req("--decisions")?)?;
    let hybrid = a.req("--method")? == search::measure::HYBRID;
    let method = match a.req("--method")? {
        "exact" => Method::ExactPimc,
        "rollout" => Method::RolloutPimc,
        // A silver reference (docs/m3-hybrid/silver.md): exact search to the start of trick
        // --cut, the network --net there.
        m if m == search::measure::HYBRID => Method::ExactPimc,
        m => return Err(format!("unknown method {m} (exact, rollout or hybrid)")),
    };
    if !hybrid {
        for k in ["--cut", "--net"] {
            if a.has(k) {
                return Err(format!("{k} is for --method hybrid (a silver reference)"));
            }
        }
    }
    let t0 = Instant::now();
    let (table, table_sha) = load_equity_sha(a.req("--equity")?)?;
    let leaf: Arc<SharedLeaf> = Arc::new(table);
    // The frontier's network of a hybrid run, and the fingerprint its records carry.
    let oracle = if hybrid {
        let cut = a.num("--cut", 0)? as u8;
        let path = a.req("--net")?;
        let o = nn::Oracle::load(Path::new(path), leaf.clone())?;
        if a.has("--cut") && o.trick != cut {
            return Err(format!(
                "--net {path} was trained for trick {} positions, not trick {cut}",
                o.trick
            ));
        }
        Some(o)
    } else {
        None
    };
    // Exact leaves of a hybrid run (a silver reference "H_3 with learned exact leaves").
    let leaves = match &oracle {
        Some(o) => leaf_policy_arg(a, "", o.trick, &leaf)?,
        None => {
            for k in ["--exact-leaves", "--exact-leaf-tt-mb", "--defer-expert"] {
                if a.has(k) {
                    return Err(format!("{k} is for --method hybrid (a silver reference)"));
                }
            }
            None
        }
    };
    if let (Some(p), Some(o)) = (&leaves, &oracle)
        && let Some(m) = p.net_mismatch(&o.sha256)
    {
        return Err(format!("--exact-leaves: {m}"));
    }
    let frontier = oracle.as_ref().map(|o| search::measure::FrontierFp {
        cut: o.trick,
        net_sha256: o.sha256.clone(),
        encoder: nn::encode::ENCODER.to_string(),
        exact_leaves: leaves
            .as_ref()
            .map(search::measure::FrontierFp::leaves_text),
    });
    let src = world_source(a)?;
    let ctx = Ctx {
        leaf: &*leaf,
        source: &*src,
        rules: engine::game::Rules::HOUSE,
    };
    let cfg = SearchConfig {
        threads: a.num("--threads", 1)? as usize,
        solve_threads: a.num("--solve-threads", 1)? as usize,
        tt_mb: a.num("--tt-mb", 256)? as usize,
        ..SearchConfig::default()
    };
    let sched = schedule(a, cfg.threads)?;
    let (from, to) = (a.num("--from-world", 0)?, a.num("--to", 10)?);
    if to < from {
        return Err(format!("--to {to} is below --from-world {from}"));
    }
    let job = WorldsJob {
        method,
        label: a.num("--label", 1)?,
        seed: a.num("--seed", 1)?,
        from,
        to,
        table: table_sha,
        frontier,
        exact_leaves: leaves,
    };
    let (first, last) = (a.num("--first", 0)?, a.num("--last", u64::MAX)?);
    let ids = a.get("--ids").map(parse_ids).transpose()?;
    if let Some(ids) = &ids
        && let Some(x) = ids.iter().find(|&&i| !ds.iter().any(|d| d.id == i))
    {
        return Err(format!("--ids: decision {x} is not in the decisions file"));
    }
    let ds: Vec<DecisionSpec> = ds
        .into_iter()
        .filter(|d| d.id >= first && d.id <= last)
        .filter(|d| ids.as_ref().is_none_or(|ids| ids.contains(&d.id)))
        .collect();
    let out = Path::new(a.req("--out")?);
    let progress_path = a.get("--progress").map(PathBuf::from);
    let prog = BatchProgress::default();
    let finished = AtomicBool::new(false);
    if let (Some(f), Some(o)) = (&job.frontier, &oracle) {
        eprintln!(
            "worlds: a silver reference: H_{} (exact search to the start of trick {}, network {} \
             {:?} there), records `{}`; never an exact golden set",
            f.cut,
            f.cut,
            &f.net_sha256[..12],
            o.net().widths(),
            job.method_name()
        );
    }
    match &sched {
        Schedule::Fixed => eprintln!(
            "worlds: {} decisions, worlds {from}..{to}, {} threads, {} solve threads (table loaded in {:.1} s)",
            ds.len(),
            cfg.threads,
            cfg.solve_threads,
            t0.elapsed().as_secs_f64()
        ),
        Schedule::Auto {
            threads, budget_mb, ..
        } => eprintln!(
            "worlds: {} decisions, worlds {from}..{to}, position-aware on {threads} threads with \
             {budget_mb} MB for tables (table loaded in {:.1} s)",
            ds.len(),
            t0.elapsed().as_secs_f64()
        ),
    }
    let n = std::thread::scope(|s| {
        if let Some(p) = &progress_path {
            let (prog, finished, job, cfg) = (&prog, &finished, &job, &cfg);
            s.spawn(move || {
                while !finished.load(Ordering::Relaxed) {
                    write_progress(p, prog, job, cfg, t0, false);
                    for _ in 0..20 {
                        if finished.load(Ordering::Relaxed) {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    }
                }
            });
        }
        let net = oracle
            .as_ref()
            .map(|o| o as &dyn solver::frontier::Frontier);
        let r = worlds_file_net(
            out,
            &ds,
            &job,
            &ctx,
            &cfg,
            &sched,
            Some(&prog),
            |r, t| {
                eprintln!(
                    "decision {} trick {} pos {} cands {}: {:.1} ms per world (slowest {:.1} ms; {:.3e} nodes{}); {}/{} decisions, {:.0} s",
                    r.id,
                    r.trick,
                    r.position + 1,
                    r.cands.len(),
                    t.world_us as f64 / 1000.0 / t.worlds.max(1) as f64,
                    t.max_world_us as f64 / 1000.0,
                    t.nodes as f64 / t.worlds.max(1) as f64,
                    if t.evals > 0 {
                        format!(
                            ", {:.0} network evaluations",
                            t.evals as f64 / t.worlds.max(1) as f64
                        )
                    } else {
                        String::new()
                    },
                    prog.decisions_done.load(Ordering::Relaxed) + 1,
                    prog.decisions_total.load(Ordering::Relaxed),
                    t0.elapsed().as_secs_f64()
                );
            },
            net,
        );
        finished.store(true, Ordering::Relaxed);
        r
    })?;
    if let Some(p) = &progress_path {
        write_progress(p, &prog, &job, &cfg, t0, true);
    }
    eprintln!(
        "worlds: {n} decisions valued in {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// `--schedule fixed|auto` (default fixed): with `auto`, every exact world gets its own threads
/// and table by position ([`search::plan`]) on `--threads` threads in all, with
/// `--tt-budget-mb` MB for all tables at their peak (default: 16384) and the plan's knobs from
/// `--sched-json` (a JSON object of `AutoParams` fields, or a file holding one).
fn schedule(a: &Args, threads: usize) -> Result<Schedule, String> {
    match a.get("--schedule").unwrap_or("fixed") {
        "fixed" => Ok(Schedule::Fixed),
        "auto" => Ok(Schedule::Auto {
            threads: threads.max(1),
            budget_mb: a.num("--tt-budget-mb", 16_384)?,
            params: Box::new(auto_params(a)?),
        }),
        s => Err(format!("unknown --schedule {s} (fixed or auto)")),
    }
}

/// `--sched-json`: the position-aware plan's knobs, inline JSON or a file.
fn auto_params(a: &Args) -> Result<AutoParams, String> {
    match a.get("--sched-json") {
        None => Ok(AutoParams::default()),
        Some(s) => {
            let text = if s.trim_start().starts_with('{') {
                s.to_string()
            } else {
                std::fs::read_to_string(s).map_err(|e| format!("--sched-json {s}: {e}"))?
            };
            serde_json::from_str(&text).map_err(|e| format!("--sched-json: {e}"))
        }
    }
}

/// Write `v` to `path` whole (a temporary file renamed over it, so a reader never sees half).
fn write_atomic(path: &Path, v: &serde_json::Value) {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    if std::fs::write(&tmp, v.to_string()).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

fn float(a: &Args, k: &str, d: f64) -> Result<f64, String> {
    a.get(k).map_or(Ok(d), |s| {
        s.parse::<f64>()
            .map_err(|_| format!("{k}: not a number: {s}"))
    })
}

/// `search bench-t1`: one window of the trick-1 machine bench (`search::t1bench`), written as
/// JSON to `--json`. `--progress FILE` is rewritten every sample (about every second) with the
/// window's state; a `--stop-file` appearing stops the window at once with the file's text as the
/// reason.
fn bench_t1(a: &Args) -> Result<(), String> {
    let mode = Mode::parse(a.req("--mode")?)?;
    let ds: Vec<DecisionSpec> = read_lines(a.req("--decisions")?)?;
    let table = load_equity(a.req("--equity")?)?;
    let leaf: Arc<SharedLeaf> = Arc::new(table);
    let src = world_source(a)?;
    let ctx = Ctx {
        leaf: &*leaf,
        source: &*src,
        rules: engine::game::Rules::HOUSE,
    };
    let threads = a.num("--threads", 1)?.max(1) as usize;
    let mut p = WindowParams::new(mode, float(a, "--window-s", 240.0)?, threads);
    p.seed = a.num("--seed", 1)?;
    p.label = a.num("--label", 1)?;
    p.tt_mb = a.num("--tt-mb", 512)? as usize;
    p.budget_mb = a.num("--tt-budget-mb", 16_384)?;
    p.auto = auto_params(a)?;
    p.worlds = a.num("--worlds-per-decision", 800)?;
    p.sample_ms = a.num("--sample-ms", 1000)?;
    let d = Guard::default();
    let guard = Guard {
        ramp_s: float(a, "--ramp-s", d.ramp_s)?,
        min_cpu_pct: float(a, "--min-cpu-pct", d.min_cpu_pct)?,
        cpu_interval_s: float(a, "--cpu-interval-s", d.cpu_interval_s)?,
        busy_streak_s: float(a, "--busy-streak-s", d.busy_streak_s)?,
        min_busy_pct: float(a, "--min-busy-pct", d.min_busy_pct)?,
        usable_cores: float(a, "--usable-cores", d.usable_cores)?,
    };
    // `--no-guard` keeps the usable cores (for the CPU share) but never stops the window.
    p.guard = Some(if a.has("--no-guard") {
        Guard {
            ramp_s: f64::INFINITY,
            ..guard
        }
    } else {
        guard
    });
    let progress_path = a.get("--progress").map(PathBuf::from);
    let stop_file = a.get("--stop-file").map(PathBuf::from);
    if let Some(f) = &stop_file {
        let _ = std::fs::remove_file(f);
    }
    let window_s = p.window_s;
    let state = |s: Option<&Sample>, started: bool, stopped: Option<&str>| {
        let (t, nodes) = s.map_or((0.0, 0), |s| (s.t, s.nodes));
        serde_json::json!({
            "window": mode.name(), "started": started, "window_s": window_s,
            "t_s": (t * 10.0).round() / 10.0, "threads": threads,
            "busy": s.map(|s| s.busy), "held": s.map(|s| s.held), "nodes": nodes,
            "finished_worlds": s.map(|s| s.finished),
            "rate": if t > 0.0 { (nodes as f64 / t).round() } else { 0.0 },
            "cpu_s": s.map(|s| (s.cpu_s * 10.0).round() / 10.0),
            "stopped": stopped,
        })
    };
    if let Some(path) = &progress_path {
        write_atomic(path, &state(None, false, None));
    }
    eprintln!(
        "bench-t1 {}: {} decisions, {threads} threads, {window_s} s",
        mode.name(),
        ds.len()
    );
    let mut on_sample = |s: &Sample| {
        if let Some(path) = &progress_path {
            write_atomic(path, &state(Some(s), true, None));
        }
        let f = stop_file.as_ref()?;
        let why = std::fs::read_to_string(f).ok()?;
        Some(match why.trim() {
            "" => "stopped by the runner".to_string(),
            w => w.to_string(),
        })
    };
    let run = run_window(&ds, &ctx, &p, &mut on_sample)?;
    if let Some(path) = &progress_path {
        write_atomic(
            path,
            &state(run.samples.last(), true, run.stopped.as_deref())
                .as_object()
                .cloned()
                .map(|mut o| {
                    o.insert("finished".into(), serde_json::json!(true));
                    serde_json::Value::Object(o)
                })
                .unwrap_or_default(),
        );
    }
    eprintln!(
        "bench-t1 {}: {} nodes in {:.1} s ({:.3e} a second), {} of {} worlds finished{}",
        mode.name(),
        run.nodes(),
        run.wall_s,
        run.nodes() as f64 / run.wall_s.max(1e-9),
        run.worlds.iter().filter(|w| w.finished).count(),
        run.worlds.len(),
        run.stopped
            .as_deref()
            .map_or(String::new(), |s| format!("; stopped early: {s}"))
    );
    let text = serde_json::to_string(&run).map_err(|e| e.to_string())?;
    std::fs::write(a.req("--json")?, text + "\n").map_err(|e| e.to_string())
}

/// `--grid`: a list (`10,20,40`) or a range `start:end:step` (`10:320:10`).
fn parse_grid(s: &str) -> Result<Vec<usize>, String> {
    let bad = || format!("bad grid {s}");
    if s.contains(':') {
        let p: Vec<usize> = s
            .split(':')
            .map(|x| x.trim().parse().map_err(|_| bad()))
            .collect::<Result<_, _>>()?;
        let [a, b, step] = p[..] else {
            return Err(bad());
        };
        if step == 0 || a == 0 {
            return Err(bad());
        }
        return Ok((a..=b).step_by(step).collect());
    }
    s.split(',')
        .map(|x| x.trim().parse().map_err(|_| bad()))
        .collect()
}

fn grid(a: &Args) -> Result<Vec<usize>, String> {
    parse_grid(a.get("--grid").unwrap_or("10,20,40,80,160,320"))
}

/// Every `--worlds` file, merged (extension records joined to the earlier worlds), restricted to
/// `--ids` when given, with hand types recomputed from the `--decisions` files when given.
fn worlds_args(a: &Args) -> Result<Vec<WorldRows>, String> {
    let paths: Vec<&Path> = a.all("--worlds").into_iter().map(Path::new).collect();
    if paths.is_empty() {
        return Err("missing --worlds".into());
    }
    let (rows, warnings) = read_worlds(&paths)?;
    for w in warnings {
        eprintln!("warning: {w}");
    }
    let ids = a.get("--ids").map(parse_ids).transpose()?;
    let mut rows: Vec<WorldRows> = rows
        .into_iter()
        .filter(|r| ids.as_ref().is_none_or(|ids| ids.contains(&r.id)))
        .collect();
    // Hand types recomputed from the decision sets (the 2026-10-14 features for old records).
    let decision_files = a.all("--decisions");
    if !decision_files.is_empty() {
        let mut ds: Vec<DecisionSpec> = Vec::new();
        for f in &decision_files {
            ds.extend(read_lines::<DecisionSpec>(f)?);
        }
        let n = relabel(&mut rows, &ds)?;
        eprintln!(
            "hand types recomputed for {n} of {} records from {}",
            rows.len(),
            decision_files.join(", ")
        );
    }
    if rows
        .iter()
        .any(|r| r.hand_type.is_some_and(|h| h.feature_bands().is_none()))
    {
        eprintln!(
            "note: some records predate the hand features (holds the 5 of trumps, robber, voids); \
             pass --decisions FILE (their decision set) to compute them"
        );
    }
    Ok(rows)
}

/// The fewest decisions a reported cell should have (§13.1 item 8).
const MIN_CELL: usize = 25;

fn agree(a: &Args) -> Result<(), String> {
    let rows = worlds_args(a)?;
    let methods: std::collections::BTreeSet<&str> =
        rows.iter().map(|r| r.method.as_str()).collect();
    if methods.len() > 1 {
        return Err(format!(
            "the --worlds files hold records of several methods ({}): agree one at a time",
            methods.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    if let Some(f) = rows
        .iter()
        .find_map(|r| r.fp.as_ref().and_then(|f| f.frontier.as_ref()))
    {
        eprintln!(
            "note: a silver reference's values ({}), not exact ones (docs/m3-hybrid/silver.md)",
            f.describe()
        );
    }
    // Timings, when the timing files are there: per decision, summed over its records (an
    // extension adds a record).
    let mut times: BTreeMap<u64, DecisionTiming> = BTreeMap::new();
    for path in a.all("--worlds") {
        if let Ok(ts) = read_jsonl::<DecisionTiming>(&timing_path(Path::new(path))) {
            for t in ts {
                times
                    .entry(t.id)
                    .and_modify(|x| {
                        x.worlds += t.worlds;
                        x.world_us += t.world_us;
                        x.max_world_us = x.max_world_us.max(t.max_world_us);
                    })
                    .or_insert(t);
            }
        }
    }
    let g = grid(a)?;
    // Groups by (trick, kind, index): all, position, position pair, stratum, log2 world-count band.
    let mut groups: BTreeMap<(u8, u8, u32), Vec<&WorldRows>> = BTreeMap::new();
    for r in &rows {
        let p = u32::from(r.position);
        let mut keys = vec![
            (0, 0),
            (1, p + 1),
            (2, p / 2 * 2 + 1),
            (3, u32::from(r.stratum)),
        ];
        if let Some(b) = r.log2_worlds {
            keys.push((4, b));
        }
        if let Some(h) = r.hand_type {
            let [t, top, plain] = h.bands();
            keys.extend([
                (5, u32::from(t)),
                (6, u32::from(top)),
                (7, u32::from(plain)),
            ]);
            if let Some(f) = h.feature_bands() {
                keys.extend((0..3).map(|j| (8 + j as u8, u32::from(f[j]))));
            }
        }
        for (kind, i) in keys {
            groups.entry((r.trick, kind, i)).or_default().push(r);
        }
    }
    let name = |kind: u8, i: u32| match kind {
        0 => "all".to_string(),
        1 => format!("pos{i}"),
        2 => format!("pair{i}-{}", i + 1),
        3 => format!("S{i}"),
        4 => format!("log2worlds{i}"),
        5 => format!("trumps{i}{}", if i == 3 { "+" } else { "" }),
        6 => format!("top{i}{}", if i == 2 { "+" } else { "" }),
        7 => match i {
            1 => "plain0-1".to_string(),
            3 => "plain3+".to_string(),
            _ => format!("plain{i}"),
        },
        _ => feature_band_name(usize::from(kind - 8), i as u8),
    };
    let mut report = serde_json::Map::new();
    let mut last_kind = None;
    for (&(trick, kind, i), rs) in &groups {
        if last_kind != Some((trick, kind)) {
            last_kind = Some((trick, kind));
            println!(
                "== trick {trick}, by {} ==",
                [
                    "all",
                    "position",
                    "position pair",
                    "stratum",
                    "log2 world count",
                    "hand type: trumps held",
                    "hand type: top trumps held (5, J, Joker, A♥, ace of trumps)",
                    "hand type: longest plain suit",
                    "hand type: holds the 5 of trumps",
                    "hand type: robber (took a turned card this hand)",
                    "hand type: void plain suits",
                ][usize::from(kind)]
            );
        }
        let group = name(kind, i);
        let owned: Vec<WorldRows> = rs.iter().map(|r| (*r).clone()).collect();
        let (steps, plateau) = agreement_curve(&owned, &g);
        let timed: Vec<f64> = rs
            .iter()
            .filter_map(|r| times.get(&r.id))
            .map(|t| t.world_us as f64 / 1000.0 / t.worlds.max(1) as f64)
            .collect();
        let ms = (!timed.is_empty()).then(|| timed.iter().sum::<f64>() / timed.len() as f64);
        let enough = rs.len() >= MIN_CELL;
        println!(
            "trick {trick} {group}: {} decisions{}, {} per world, plateau N = {}",
            rs.len(),
            if enough {
                String::new()
            } else {
                format!(" (under {MIN_CELL}: too few to set N)")
            },
            ms.map_or("n/a ms".to_string(), |m| format!("{m:.1} ms")),
            plateau.map_or("none on this grid".to_string(), |n| n.to_string())
        );
        println!("   N→2N  change%  |Δmean|   loss(±SE)");
        for s in &steps {
            println!(
                "  {:>3}→{:<3} {:>6.2}  {:.5}  {:.5}±{:.5} {}",
                s.n,
                s.n2,
                100.0 * s.change_rate,
                s.mean_value_change,
                s.mean_loss,
                s.loss_se,
                if s.plateau_ok { "ok" } else { "" }
            );
        }
        report.insert(
            format!("trick{trick}-{group}"),
            serde_json::json!({ "decisions": rs.len(), "enough_decisions": enough,
                                "ms_per_world": ms, "plateau": plateau, "steps": steps }),
        );
    }
    let tricks: std::collections::BTreeSet<u8> = rows.iter().map(|r| r.trick).collect();
    for t in tricks {
        if !rows.iter().any(|r| r.trick == t && r.log2_worlds.is_some()) {
            println!(
                "trick {t} by log2 world count: n/a (the world source cannot count worlds; use --source belief)"
            );
            report.insert(format!("trick{t}-log2worlds"), serde_json::json!("n/a"));
        }
    }
    if let Some(p) = a.get("--json") {
        std::fs::write(
            p,
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// A money chance as a percentage for the text report.
fn pct(x: f64) -> String {
    if x.is_finite() {
        format!("{:.4}%", 100.0 * x)
    } else {
        "n/a".into()
    }
}

/// A predicted N for the text report.
fn n_text(x: f64) -> String {
    if !x.is_finite() {
        "n/a".into()
    } else if x >= N_CAP * 0.999 {
        "never".into()
    } else if x > 1e6 {
        format!(">{:.0e}", 1e6)
    } else {
        format!("{x:.0}")
    }
}

fn print_trick(t: &TrickReport) {
    println!(
        "== trick {}: {} decisions ({} with fewer than {} worlds left out), M = {} worlds, \
         reference K = {}, fit on N <= {}; {} splits, {} bootstrap replicates, seed {}",
        t.trick,
        t.decisions,
        t.dropped,
        t.worlds,
        t.worlds,
        t.k,
        t.fit_max,
        t.splits,
        t.boots,
        t.seed
    );
    match &t.b {
        Some(b) => println!(
            "shared exponent b = {:.3} [95%: {:.3}, {:.3}]; half-data fit b = {}",
            b.v,
            b.lo,
            b.hi,
            t.b_half.map_or("n/a".into(), |x| format!(
                "{:.3} [{:.3}, {:.3}]",
                x.v, x.lo, x.hi
            ))
        ),
        None => println!("shared exponent b: no fit (too few points)"),
    }
    println!(
        "b fitted on N up to: {}",
        t.b_by_range
            .iter()
            .map(|(n, b)| format!("{n}: {b:.3}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "held-out check ({} < N < {}, all + positions): {}/{} predictions inside the measured 95% \
         band; mean predicted/measured = {:.2}",
        t.fit_max,
        t.k,
        t.heldout_inside.0,
        t.heldout_inside.1,
        t.heldout_log_ratio.exp()
    );
    println!(
        "extrapolation check (first {} worlds predict the {}-world curve, N < {}): {}/{} inside; \
         mean predicted/measured = {:.2}",
        t.worlds / 2,
        t.worlds,
        t.k,
        t.half_inside.0,
        t.half_inside.1,
        t.half_log_ratio.exp()
    );
    let at = |g: &search::curves::GroupReport, n: usize| {
        t.grid
            .iter()
            .position(|&x| x == n)
            .and_then(|i| g.cv.mean.get(i).copied())
            .map_or("-".into(), pct)
    };
    let quarter = t.fit_max / 2;
    print!(
        "{:<22} {:>4} {:>10} {:>10} {:>10} {:>11} {:>22}",
        "group",
        "dec",
        format!("cv({})", t.grid.first().copied().unwrap_or(0)),
        format!("cv({quarter})"),
        format!("cv({})", t.fit_max),
        "a",
        "free b [95%]"
    );
    for th in &t.thresholds {
        print!(" {:>24}", format!("N({}) [95%]", pct(*th)));
    }
    println!();
    for g in &t.groups {
        print!(
            "{:<22} {:>4} {:>10} {:>10} {:>10} {:>11} {:>22}",
            format!("{}{}", g.name, if g.enough { "" } else { "*" }),
            g.decisions,
            at(g, t.grid.first().copied().unwrap_or(0)),
            at(g, quarter),
            at(g, t.fit_max),
            g.a.map_or("n/a".into(), |a| format!("{:.3e}", a.v)),
            g.free_b.map_or(String::new(), |b| format!(
                "{:.2} [{:.2}, {:.2}]",
                b.v, b.lo, b.hi
            ))
        );
        for e in &g.n_for {
            print!(
                " {:>24}",
                format!("{} [{}, {}]", n_text(e.v), n_text(e.lo), n_text(e.hi))
            );
        }
        println!();
    }
    println!(
        "(* fewer than {} decisions; cv(N) = measured L(N) - L({}), money chance)",
        search::curves::ENOUGH,
        t.k
    );
    if !t.contrasts.is_empty() {
        println!(
            "hand features, N({}) of one group over another [95%, paired bootstrap]: {}",
            pct(t.thresholds.first().copied().unwrap_or(f64::NAN)),
            t.contrasts
                .iter()
                .map(|c| format!(
                    "{}/{} {:.2} [{:.2}, {:.2}]",
                    c.group, c.vs, c.n_ratio.v, c.n_ratio.lo, c.n_ratio.hi
                ))
                .collect::<Vec<_>>()
                .join("; ")
        );
    }
}

fn curves(a: &Args) -> Result<(), String> {
    let t0 = Instant::now();
    let method = a.get("--method").unwrap_or("exact");
    let label = a.num("--label", 1)?;
    let rows: Vec<WorldRows> = worlds_args(a)?
        .into_iter()
        .filter(|r| r.method == method && r.label == label)
        .collect();
    if rows.is_empty() {
        return Err(format!(
            "no {method} label {label} records in the --worlds files"
        ));
    }
    let d = CurveParams::default();
    let p = CurveParams {
        grid: a
            .get("--grid")
            .map_or(Ok((1..=1000).map(|i| 10 * i).collect()), parse_grid)?,
        worlds: a
            .get("--m")
            .map(|_| a.num("--m", 0))
            .transpose()?
            .map(|m| m as usize),
        splits: a.num("--splits", d.splits)?,
        boots: a.num("--boot", d.boots)?,
        seed: a.num("--seed", d.seed)?,
        thresholds: match a.get("--thresholds") {
            Some(s) => s
                .split(',')
                .map(|x| x.trim().parse().map_err(|_| format!("bad threshold {x}")))
                .collect::<Result<_, _>>()?,
            None => d.thresholds,
        },
        min_cell: a.num("--min-cell", d.min_cell as u64)? as usize,
        threads: a.num(
            "--threads",
            std::thread::available_parallelism().map_or(1, |n| n.get()) as u64,
        )? as usize,
    };
    let reports = analyse(&rows, &p);
    for t in &reports {
        print_trick(t);
    }
    if let Some(path) = a.get("--json") {
        let v = serde_json::json!({
            "method": method, "label": label,
            "files": a.all("--worlds"),
            "tricks": reports,
        });
        std::fs::write(path, serde_json::to_string(&v).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    }
    eprintln!("curves: done in {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
}

fn score_cmd(a: &Args) -> Result<(), String> {
    let (yard, _) = read_worlds(&[Path::new(a.req("--yardstick")?)])?;
    // The yardstick is exact PIMC; a silver reference (hybrid records, docs/m3-hybrid/silver.md)
    // only when asked for, and the report then says so.
    if let Some(r) = yard.iter().find(|r| r.method != "exact") {
        if !(a.has("--allow-silver") && r.method == search::measure::HYBRID) {
            return Err(format!(
                "--yardstick: decision {} holds `{}` records, not exact ones{}",
                r.id,
                r.method,
                if r.method == search::measure::HYBRID {
                    " (a silver reference: pass --allow-silver to score against it)"
                } else {
                    ""
                }
            ));
        }
        println!(
            "yardstick: a silver reference ({}), not exact PIMC",
            r.fp.as_ref()
                .and_then(|f| f.frontier.as_ref())
                .map_or("hybrid".to_string(), |f| f.describe())
        );
    }
    let n = a.num("--n", 80)? as usize;
    let mut scored = Vec::new();
    let mut reports = Vec::new();
    for m in a.all("--method") {
        let (path, nm) = m.rsplit_once(':').ok_or("--method FILE:N")?;
        let nm: usize = nm.parse().map_err(|_| "--method FILE:N")?;
        let (rows, _) = read_worlds(&[Path::new(path)])?;
        let s = score(&yard, n, &rows, nm);
        let sum = summarise(&s);
        println!(
            "{m}: {}",
            serde_json::to_string(&sum).map_err(|e| e.to_string())?
        );
        // Per position in the trick, then per stratum.
        let mut cells: BTreeMap<(u8, u8), Vec<search::measure::Scored>> = BTreeMap::new();
        for x in &s {
            if let Some(y) = yard.iter().find(|y| y.id == x.id) {
                cells
                    .entry((0, y.position + 1))
                    .or_default()
                    .push(x.clone());
                cells.entry((1, y.stratum)).or_default().push(x.clone());
            }
        }
        for (&(kind, i), v) in &cells {
            let ps = summarise(v);
            println!(
                "  {} {i}: n={} agree={:.3} loss={:.5}±{:.5}",
                if kind == 0 { "pos" } else { "S" },
                ps.decisions,
                ps.agreement.0,
                ps.mean_loss.0,
                ps.mean_loss.1
            );
        }
        if a.get("--json").is_some() {
            reports.push(score_groups(
                m,
                &yard,
                n,
                &rows,
                nm,
                a.num("--boot", 1000)?,
                a.num("--seed", 1)?,
            ));
        }
        scored.push((m.to_string(), s));
    }
    let mut pairs = Vec::new();
    for i in 0..scored.len() {
        for j in i + 1..scored.len() {
            let p = paired(&scored[i].1, &scored[j].1);
            println!(
                "{} vs {}: {}",
                scored[i].0,
                scored[j].0,
                serde_json::to_string(&p).map_err(|e| e.to_string())?
            );
            pairs.push(serde_json::json!({ "a": scored[i].0, "b": scored[j].0, "paired": p }));
        }
    }
    if let Some(path) = a.get("--json") {
        let v = serde_json::json!({ "yardstick": a.req("--yardstick")?, "n_ref": n,
                                    "methods": reports, "paired": pairs });
        std::fs::write(path, serde_json::to_string(&v).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn teacher(a: &Args) -> Result<(), String> {
    if a.has("--trick") {
        return student_teacher(a);
    }
    let hands = a.num("--hands", 10)?;
    let seed = a.num("--seed", 1)?;
    let spec = PimcSpec::parse(a.get("--config").unwrap_or("pimc-teacher"))?;
    let table = load_equity(a.req("--equity")?)?;
    let leaf: Arc<SharedLeaf> = Arc::new(table);
    let f = PimcFactory::new(&spec, leaf, world_source(a)?);
    let starts = collect_starts(a.num("--start-hands", 20_000)?, seed);
    let dir = std::path::Path::new(a.req("--out")?);
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let commit = option_env!("GIT_COMMIT").unwrap_or("unknown");
    let mut lines = String::new();
    let mut rows = Vec::new();
    for i in 0..hands {
        let st = &starts[(i % 4) as usize];
        let mut rng = engine::rng::Rng::derive(seed, &[search::labels::STRATUM, i]);
        let start = st[rng.bounded(st.len() as u64) as usize];
        debug_assert_eq!(u64::from(stratum_of(&start)), i % 4 + 1);
        let (line, ds) = teacher_hand(i, start, teacher_deck(seed, i), &f, seed ^ i, commit);
        lines.push_str(&line.to_string());
        lines.push('\n');
        rows.extend(ds);
    }
    let npy = decisions_npy(&rows);
    let h = |b: &[u8]| equity::sha256::sha256_hex(b)[..16].to_string();
    std::fs::write(
        dir.join(format!("hands-{}.jsonl", h(lines.as_bytes()))),
        &lines,
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(dir.join(format!("decisions-{}.npy", h(&npy))), &npy)
        .map_err(|e| e.to_string())?;
    println!("{hands} hands, {} decisions", rows.len());
    Ok(())
}

/// `search teacher --trick 1|2`: a shard of the student's teacher records.
fn student_teacher(a: &Args) -> Result<(), String> {
    use search::student_teacher::{Teacher, TeacherJob, belief_ctx, write_teacher_shard};
    let trick = a.num("--trick", 2)? as u8;
    let seed: u64 = a
        .req("--seed")?
        .parse()
        .map_err(|_| "--seed: not a number".to_string())?;
    let threads = a.num("--threads", 1)?.max(1) as usize;
    let role = a.get("--role").unwrap_or("train");
    if role != "train" && role != "test" {
        return Err("--role is train or test".into());
    }
    let job = TeacherJob {
        trick,
        count: a.num("--count", 0)?,
        first: a.num("--first", 0)?,
        deals: u32::try_from(a.num("--deals", 320)?).map_err(|_| "--deals")?,
        deal_from: u32::try_from(a.num("--deal-from", 0)?).map_err(|_| "--deal-from")?,
        seed,
        source: student::record::parse_source(a.get("--source").unwrap_or("rules"))?,
        threads,
        tt_mb: a.num("--tt-mb", 64)? as usize,
        start_hands: a.num("--start-hands", 20_000)?,
    };
    let (table, equity_sha) = load_equity_sha(a.req("--equity")?)?;
    let leaf: Arc<SharedLeaf> = Arc::new(table);
    // The playout teacher (`--playout --student1 S1 --student2 S2 [--student3 S3 [--student4 S4]]
    // [--playout-cut 3]`): the students play every seat from its own view to the start of the
    // cut trick, exact from there.
    let playout = a.has("--playout");
    let students_files = if playout {
        if a.has("--net") {
            return Err("--playout values deals by the students, not a network: no --net".into());
        }
        let load = |k: &str| -> Result<(student::net::StudentNet, String), String> {
            let path = a.req(k)?;
            let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
            let net = student::net::StudentNet::from_bytes(&bytes)?;
            Ok((net, equity::sha256::sha256_hex(&bytes)))
        };
        // `--student3 S3` plays trick 3 when the cut is 4 or later, `--student4 S4` trick 4 when
        // it is 5 (docs/STUDENT.md §6.3).
        let opt = |k: &str| -> Result<_, String> {
            if a.has(k) {
                Ok(Some(load(k)?))
            } else {
                Ok(None)
            }
        };
        Some((
            load("--student1")?,
            load("--student2")?,
            opt("--student3")?,
            opt("--student4")?,
        ))
    } else {
        for k in [
            "--student1",
            "--student2",
            "--student3",
            "--student4",
            "--playout-cut",
        ] {
            if a.has(k) {
                return Err(format!("{k} is for the playout teacher (--playout)"));
            }
        }
        None
    };
    let playout_cut = a.num("--playout-cut", 3)? as u8;
    let net_arg = if playout { "exact" } else { a.req("--net")? };
    let oracle = if net_arg == "exact" {
        None
    } else {
        Some(nn::Oracle::load(Path::new(net_arg), leaf.clone())?)
    };
    // Exact leaves of a hybrid teacher (`--exact-leaves`, `--defer-expert`: as `hybrid-check`'s;
    // the stamp names them).
    let leaves = match &oracle {
        Some(o) => leaf_policy_arg(a, "", o.trick, &leaf)?,
        None => {
            for k in ["--exact-leaves", "--exact-leaf-tt-mb", "--defer-expert"] {
                if a.has(k) {
                    return Err(format!("{k} is for a hybrid teacher (--net WEIGHTS)"));
                }
            }
            None
        }
    };
    if let (Some(p), Some(o)) = (&leaves, &oracle)
        && let Some(m) = p.net_mismatch(&o.sha256)
    {
        return Err(format!("--exact-leaves: {m}"));
    }
    let teacher = match (&oracle, &students_files) {
        (_, Some(((s1, sha1), (s2, sha2), s3, s4))) => Teacher::Playout {
            students: search::pimc::Students {
                s1,
                s2,
                s3: s3.as_ref().map(|(n, _)| n),
                s4: s4.as_ref().map(|(n, _)| n),
            },
            cut: playout_cut,
            sha1,
            sha2,
            sha3: s3.as_ref().map(|(_, h)| h.as_str()),
            sha4: s4.as_ref().map(|(_, h)| h.as_str()),
        },
        (None, None) => Teacher::Exact,
        (Some(o), None) => Teacher::Hybrid {
            net: o,
            cut: o.trick,
            sha256: &o.sha256,
            leaves: leaves.as_ref(),
        },
    };
    search::student_teacher::check_job(&job, &teacher, role == "test")?;
    let out = PathBuf::from(a.req("--out")?);
    if !out.to_string_lossy().ends_with(".npy") {
        return Err("--out must name a .npy file".into());
    }
    let t0 = Instant::now();
    let starts = collect_starts_with(job.start_hands, seed, threads);
    let source = BeliefSource::new();
    let ctx = belief_ctx(&*leaf, &source);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let progress = a.get("--progress").map(PathBuf::from);
    let write = |finished: bool| {
        if let Some(p) = &progress {
            let v = serde_json::json!({
                "kind": "student_teacher",
                "done": done.load(Ordering::Relaxed),
                "count": job.count,
                "elapsed_s": t0.elapsed().as_secs(),
                "finished": finished,
            });
            let tmp = PathBuf::from(format!("{}.tmp", p.display()));
            if std::fs::write(&tmp, v.to_string()).is_ok() {
                let _ = std::fs::rename(&tmp, p);
            }
        }
    };
    let info = std::thread::scope(|s| {
        s.spawn(|| {
            while !stop.load(Ordering::Relaxed) {
                write(false);
                for _ in 0..20 {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        });
        let r = write_teacher_shard(
            &job,
            &starts,
            &ctx,
            teacher,
            &equity_sha,
            &binary_sha256(),
            role,
            &out,
            Some(&done),
        );
        stop.store(true, Ordering::Relaxed);
        r
    })?;
    write(true);
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "{} decisions at trick {trick} x {} deals ({}): key {}, {} value rows, {:.1} s on {threads} threads, {:.0} decisions an hour",
        info.records,
        job.deals,
        teacher.stamp(),
        info.key,
        info.value_rows,
        secs,
        info.records as f64 * 3600.0 / secs.max(1e-9),
    );
    Ok(())
}

/// `search student-check`: a student network against the teacher's stored values.
fn student_check(a: &Args) -> Result<(), String> {
    let net_path = a.req("--net")?;
    let net = student::net::StudentNet::from_bytes(
        &std::fs::read(net_path).map_err(|e| format!("{net_path}: {e}"))?,
    )?;
    let paths: Vec<PathBuf> = a.all("--teacher").into_iter().map(PathBuf::from).collect();
    let golden = a.get("--golden-decisions");
    let (table, equity_sha) = load_equity_sha(a.req("--equity")?)?;
    // The decisions: teacher shards, or the stored golden set with its exact worlds.
    let mut shards_opt: Option<student::data::Shards> = None;
    let mut golden_set: Vec<student::data::Merged> = Vec::new();
    let mut golden_info = serde_json::Value::Null;
    match (golden, paths.is_empty()) {
        (Some(_), false) => {
            return Err("give --teacher shards or --golden-decisions, not both".into());
        }
        (None, true) => {
            return Err("give at least one --teacher shard (or --golden-decisions)".into());
        }
        (None, false) => {
            let allow: Vec<String> = a
                .all("--allow-stamp")
                .into_iter()
                .map(String::from)
                .collect();
            let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
            let shards = student::data::Shards::open(&refs, &allow)?;
            for r in &shards.readers {
                if r.sidecar["equity_sha256"] != equity_sha.as_str() {
                    return Err(format!(
                        "{}: made with another equity table",
                        r.path.display()
                    ));
                }
            }
            shards_opt = Some(shards);
        }
        (Some(dpath), true) => {
            let wpath = a.req("--golden-worlds")?;
            let trick = a.num("--trick", 2)? as u8;
            let to = a.num("--to", u64::MAX)?;
            let ds: Vec<DecisionSpec> = read_lines(dpath)?;
            let (ws, _) = search::measure::read_worlds(&[Path::new(wpath)])?;
            let by_id: std::collections::BTreeMap<u64, &search::measure::WorldRows> =
                ws.iter().map(|w| (w.id, w)).collect();
            let (mut skipped, mut worlds) = (0u64, None);
            for d in ds.iter().filter(|d| d.trick == trick) {
                let w = by_id
                    .get(&d.id)
                    .ok_or_else(|| format!("{wpath}: no worlds for decision {}", d.id))?;
                if let Some(fp) = &w.fp
                    && fp.table != equity_sha
                {
                    return Err(format!(
                        "{wpath}: decision {} was valued with another equity table",
                        d.id
                    ));
                }
                let mut w = (*w).clone();
                if w.from != 0 {
                    return Err(format!(
                        "{wpath}: decision {}: worlds from {}",
                        d.id, w.from
                    ));
                }
                w.rows.truncate(usize::try_from(to).unwrap_or(usize::MAX));
                if *worlds.get_or_insert(w.rows.len()) != w.rows.len() {
                    return Err(format!(
                        "{wpath}: decision {} has {} worlds, others {}",
                        d.id,
                        w.rows.len(),
                        worlds.unwrap_or(0)
                    ));
                }
                match search::student_teacher::golden_merged(d, &w)? {
                    Some(m) => golden_set.push(m),
                    None => skipped += 1,
                }
            }
            if golden_set.is_empty() {
                return Err(format!(
                    "{dpath}: no trick-{trick} decision a student meets"
                ));
            }
            println!(
                "golden set: {} trick-{trick} decisions x {} exact worlds ({skipped} the student \
                 never meets: one candidate, or the play after the seat's own rob)",
                golden_set.len(),
                worlds.unwrap_or(0)
            );
            golden_info = serde_json::json!({
                "decisions": dpath, "decisions_sha256": sha256_hex_file(dpath)?,
                "worlds": wpath, "worlds_sha256": sha256_hex_file(wpath)?,
                "trick": trick, "worlds_per_decision": worlds, "judged": golden_set.len(),
                "skipped": skipped, "values": "exact",
            });
        }
    }
    let shards: &mut dyn student::data::Decisions = match shards_opt.as_mut() {
        Some(s) => s,
        None => &mut golden_set,
    };
    let t0 = Instant::now();
    // The extra measures (docs/STUDENT.md §14): `--split-half` and `--top-k K` (repeated or
    // `2,3`), computed in the same pass over the decisions.
    let split = a.has("--split-half");
    let mut ks: Vec<usize> = Vec::new();
    for v in a.all("--top-k") {
        for k in v.split(',') {
            let k: usize = k
                .parse()
                .ok()
                .filter(|&k| k >= 1)
                .ok_or_else(|| format!("--top-k {v}: whole numbers of at least 1"))?;
            ks.push(k);
        }
    }
    // `--sub-deals 25,50,100 [--sub-repeats R] [--sub-seed S]`: shortlist + search at the app's
    // deal counts (search::shortlist).
    let mut sub_grid: Vec<usize> = Vec::new();
    for v in a.get("--sub-deals").into_iter() {
        for n in v.split(',') {
            sub_grid.push(
                n.parse()
                    .ok()
                    .filter(|&n| n >= 1)
                    .ok_or_else(|| format!("--sub-deals {v}: whole numbers of at least 1"))?,
            );
        }
    }
    let sub_repeats = a.num("--sub-repeats", 8)? as u32;
    let sub_seed = a.num("--sub-seed", 1)?;
    let mut sub: Vec<search::shortlist::SubDecision> = Vec::new();
    let mut sub_worlds = 0usize;
    let mut judged = Vec::new();
    let mut extra = Extra::default();
    let mut half: Option<u64> = None;
    student::check::judge_each(&net, shards, &table, |m, student, j| {
        judged.push(j);
        if !sub_grid.is_empty() {
            let team = engine::team_of(m.head.pos.seat);
            let ranking = student::check::rank(student, team);
            let rows: Vec<Vec<[u64; 4]>> = m
                .values
                .iter()
                .map(|r| r.iter().map(|v| v.0).collect())
                .collect();
            if let Some(&big) = sub_grid.iter().find(|&&n| n >= rows.len()) {
                return Err(format!(
                    "--sub-deals {big}: the decision holds only {} deals",
                    rows.len()
                ));
            }
            sub_worlds = rows.len();
            sub.push(search::shortlist::SubDecision {
                trick: j.trick,
                place: j.place,
                stratum: j.stratum,
                margin: search::shortlist::margin(student, team, &ranking),
                loss: j.loss,
                n_cands: j.n_cands,
                values: ranking.iter().map(|&c| student[c].get(team)).collect(),
                rec: search::shortlist::sub_deal_record(
                    &rows,
                    team,
                    &ranking,
                    &sub_grid,
                    sub_repeats,
                    sub_seed,
                    judged.len() as u64,
                ),
            });
        }
        if !split && ks.is_empty() {
            return Ok(());
        }
        let team = engine::team_of(m.head.pos.seat);
        let ranking = student::check::rank(student, team);
        let key = |v: Vec<i128>| student::check::Scored {
            trick: j.trick,
            place: j.place,
            stratum: j.stratum,
            v,
        };
        let means = m.means();
        extra.full_loss.push(key(vec![i128::from(j.loss)]));
        for (i, &k) in ks.iter().enumerate() {
            let l = search::curves::top_k_loss(&means, team, &ranking, k);
            extra.top_k_full.resize_with(ks.len(), Vec::new);
            extra.top_k_full[i].push(key(vec![i128::from(l)]));
        }
        if split {
            let rows: Vec<Vec<[u64; 4]>> = m
                .values
                .iter()
                .map(|r| r.iter().map(|v| v.0).collect())
                .collect();
            let sh = search::curves::split_half(&rows, &m.deals, team, &ranking, &ks)?;
            if *half.get_or_insert(sh.half) != sh.half {
                return Err("split-half: the decisions' halves differ in size".into());
            }
            let both = |x: [i128; 2]| vec![x[0], x[1], x[0] + x[1]];
            extra.floor.push(key(both(sh.floor)));
            extra.player.push(key(both(sh.player)));
            extra.gap.push(key(both(sh.gap)));
            extra.top_k_split.resize_with(ks.len(), Vec::new);
            for (i, x) in sh.top_k.into_iter().enumerate() {
                extra.top_k_split[i].push(key(both(x)));
            }
        }
        Ok(())
    })?;
    let rows = student::check::report(&judged, a.num("--boot", 1000)?, a.num("--boot-seed", 1)?);
    print!("{}", student::check::print(&rows));
    println!(
        "{} decisions, student {} ({:?} hidden), {:.1} s",
        judged.len(),
        net.sha256,
        net.hidden(),
        t0.elapsed().as_secs_f64()
    );
    // Baselines on the same decisions: the rules bot, and a random legal card (seeded picks,
    // `--boot-seed`; with the exact expected loss of a random pick beside them).
    let mut baselines = Vec::new();
    for b in a.all("--baseline") {
        let boots = a.num("--boot", 1000)?;
        let boot_seed = a.num("--boot-seed", 1)?;
        match b {
            "rules" => {
                let (j, missing) = student::check::judge_rules(shards)?;
                let brows = student::check::report(&j, boots, boot_seed);
                println!("baseline rules-v1 (its choice not among the candidates: {missing}):");
                print!("{}", student::check::print(&brows));
                baselines.push(
                    serde_json::json!({ "bot": "rules-v1", "not_judged": missing, "rows": brows }),
                );
            }
            "random" => {
                let (j, expected) = student::check::judge_random(shards, boot_seed)?;
                let brows = student::check::report(&j, boots, boot_seed);
                println!(
                    "baseline random (a seeded random legal card; expected loss {:.5} %):",
                    100.0 * expected
                );
                print!("{}", student::check::print(&brows));
                baselines.push(serde_json::json!({ "bot": "random", "pick_seed": boot_seed,
                    "expected_loss_pct": 100.0 * expected, "rows": brows }));
            }
            x => return Err(format!("--baseline {x}: rules or random")),
        }
    }
    let sub_json = if sub.is_empty() {
        serde_json::Value::Null
    } else {
        search::shortlist::report(
            &sub,
            &sub_grid,
            sub_worlds,
            sub_repeats,
            a.num("--boot", 1000)?,
            a.num("--boot-seed", 1)?,
        )
    };
    let extra_json = extra.print(
        split.then_some(half.unwrap_or(0)),
        &ks,
        a.num("--boot", 1000)?,
        a.num("--boot-seed", 1)?,
    );
    // `baseline`: the rules bot's (as before); `baselines`: every one asked for, in order.
    let baseline = baselines.iter().find(|b| b["bot"] == "rules-v1").cloned();
    if let Some(out) = a.get("--json") {
        let mut v = serde_json::json!({
            "baseline": baseline,
            "baselines": baselines,
            "net": net_path,
            "net_sha256": net.sha256,
            "hidden": net.hidden(),
            "encoder_sha256": student::encode::encoder_sha256(),
            "teacher_shards": shards_opt.as_ref().map(|s| s.readers.iter().map(|r| r.key.clone()).collect::<Vec<_>>()),
            "role": shards_opt.as_ref().map_or("golden".to_string(), |s| s.role()),
            "equity_sha256": equity_sha,
            "rows": rows,
            "decisions": judged
                .iter()
                .map(|d| serde_json::json!([d.trick, d.place, d.stratum, d.n_cands, d.best, d.chosen, d.loss]))
                .collect::<Vec<_>>(),
        });
        if !golden_info.is_null() {
            v["golden"] = golden_info;
        }
        if !sub_json.is_null() {
            v["sub_deals"] = sub_json;
            v["sub_deals_grid"] = serde_json::json!(sub_grid);
            v["sub_deals_repeats"] = serde_json::json!(sub_repeats);
            v["sub_deals_seed"] = serde_json::json!(sub_seed);
        }
        // The extra measures' keys appear only when asked for (the rest stays as before).
        if let serde_json::Value::Object(o) = extra_json {
            for (k, x) in o {
                v[k] = x;
            }
        }
        write_json(out, &v)?;
    }
    Ok(())
}

/// SHA-256 (hex) of a file's bytes.
fn sha256_hex_file(path: &str) -> Result<String, String> {
    Ok(equity::sha256::sha256_hex(
        &std::fs::read(path).map_err(|e| format!("{path}: {e}"))?,
    ))
}

/// `student-check`'s extra measures (docs/STUDENT.md §14), per decision, whole numbers.
#[derive(Default)]
struct Extra {
    /// The student's loss on all deals (the plain measure, for the identity check).
    full_loss: Vec<student::check::Scored>,
    /// Per K: the loss of the teacher's pick among the student's top K, all deals.
    top_k_full: Vec<Vec<student::check::Scored>>,
    /// `[A→B, B→A, sum]` of the noisy oracle's loss (the floor).
    floor: Vec<student::check::Scored>,
    /// `[A→B, B→A, sum]` of the student's loss on each half.
    player: Vec<student::check::Scored>,
    /// `[A→B, B→A, sum]` of the other half's best less the student's choice (the honest gap).
    gap: Vec<student::check::Scored>,
    /// Per K: `[A→B, B→A, sum]` of the loss of the top-K pick made on the other half.
    top_k_split: Vec<Vec<student::check::Scored>>,
}

impl Extra {
    /// Print the extra tables (after everything `student-check` printed before) and return them
    /// as JSON keys (`top_k`, `split_half`): an empty object when none was asked for.
    fn print(&self, half: Option<u64>, ks: &[usize], boots: u64, seed: u64) -> serde_json::Value {
        use student::check::{print_measure, report_measure};
        let mut out = serde_json::Map::new();
        if !ks.is_empty() {
            let cols = [("loss", 1u128)];
            let rows = report_measure(&self.full_loss, &cols, boots, seed);
            println!("student on all deals (the plain measure again, bounds of the loss):");
            print!("{}", print_measure(&rows, &["loss"]));
            let mut tk = Vec::new();
            for (i, &k) in ks.iter().enumerate() {
                let rows = report_measure(&self.top_k_full[i], &cols, boots, seed);
                println!(
                    "top-{k}: the teacher picks among the student's top {k} candidates, all deals:"
                );
                print!("{}", print_measure(&rows, &["loss"]));
                tk.push(serde_json::json!({ "k": k, "rows": rows }));
            }
            out.insert("top_k".into(), serde_json::json!(tk));
        }
        if let Some(h) = half {
            let h = u128::from(h);
            let cols = [("a_to_b", h), ("b_to_a", h), ("mean", 2 * h)];
            let names = ["a_to_b", "b_to_a", "mean"];
            let mut sh = serde_json::Map::new();
            sh.insert(
                "rule".into(),
                "half A = even deal index, half B = odd; chosen on one half, scored on the other"
                    .into(),
            );
            sh.insert("half_deals".into(), serde_json::json!(h as u64));
            for (key, title, ds) in [
                (
                    "floor",
                    "the noisy oracle: the other half's best, scored on this half's best (the floor)",
                    &self.floor,
                ),
                (
                    "student",
                    "the student's loss on each half (its choice fixed)",
                    &self.player,
                ),
                (
                    "gap",
                    "the other half's best less the student's choice (unbiased; can be negative)",
                    &self.gap,
                ),
            ] {
                let rows = report_measure(ds, &cols, boots, seed);
                println!("split-half ({h} deals a half), {title}:");
                print!("{}", print_measure(&rows, &names));
                sh.insert(key.into(), serde_json::json!(rows));
            }
            let mut tk = Vec::new();
            for (i, &k) in ks.iter().enumerate() {
                let rows = report_measure(&self.top_k_split[i], &cols, boots, seed);
                println!(
                    "split-half ({h} deals a half), top-{k}: picked among the student's top {k} on one half, scored on the other:"
                );
                print!("{}", print_measure(&rows, &names));
                tk.push(serde_json::json!({ "k": k, "rows": rows }));
            }
            sh.insert("top_k".into(), serde_json::json!(tk));
            out.insert("split_half".into(), serde_json::Value::Object(sh));
        }
        serde_json::Value::Object(out)
    }
}

/// `search robber-burn`: the cost of searching another seat's rob burn inside the exact solve.
/// From the decision set (`--trick`, `--count`, `--seed`, `--start-hands`, as `search decisions`)
/// the decisions at `--position` (0 = leader) are taken. For each, `--share-worlds` yardstick
/// worlds (label `--label`) are sampled to give the share where another seat robs later in the
/// trick (no solve). With `--equity TABLE`, the first `--worlds` worlds of each decision are also
/// solved exactly twice, with the burn searched (the yardstick) and with the `rules-v1` burn,
/// and the node counts compared (the multiplier is searched nodes / rules-burn nodes).
fn robber_burn(a: &Args) -> Result<(), String> {
    use search::pimc::{WorldSolve, candidates, exact_world_row, world_robber};
    let trick = a.num("--trick", 1)? as u8;
    let seed = a.num("--seed", 1)?;
    let label = a.num("--label", 1)?;
    let position = a.num("--position", 0)? as u8;
    let starts = collect_starts(a.num("--start-hands", 20_000)?, seed);
    let ds: Vec<DecisionSpec> = build_decisions(trick, a.num("--count", 32)?, seed, &starts)
        .into_iter()
        .filter(|d| d.position == position)
        .collect();
    let share_worlds = a.num("--share-worlds", 1000)?;
    let solve_worlds = a.num("--worlds", 4)?;
    let table = a.get("--equity").map(load_equity).transpose()?;
    let src = world_source(a)?;
    let fallback: Arc<SharedLeaf> = Arc::new(solver::FnLeaf(|_: &equity::MatchState| {
        equity::MoneyChance::even()
    }));
    let leaf: Arc<SharedLeaf> = match table {
        Some(t) => Arc::new(t),
        None => fallback,
    };
    let ctx = Ctx {
        leaf: &*leaf,
        source: &*src,
        rules: engine::game::Rules::HOUSE,
    };
    let cfg = SearchConfig {
        tt_mb: a.num("--tt-mb", 512)? as usize,
        ..SearchConfig::default()
    };
    println!(
        "robber-burn: trick {trick}, position {position}: {} decisions",
        ds.len()
    );
    let (mut robbed, mut total) = (0u64, 0u64);
    let (mut n_search, mut n_rules) = (0u64, 0u64);
    for d in &ds {
        let (_, view, legal) = d.replay();
        let (_, cands) = candidates(&view, &legal);
        let worlds = ctx.source.prepare(&view, ctx.rules);
        let dseed = search::measure::measure_seed(seed, label, d.id);
        let mut r = 0u64;
        for w in 0..share_worlds {
            r += u64::from(world_robber(&view, &*worlds, w, dseed, &ctx).is_some());
        }
        robbed += r;
        total += share_worlds;
        println!(
            "decision {}: another seat robs in {r} of {share_worlds} worlds",
            d.id
        );
        if a.get("--equity").is_none() {
            continue;
        }
        for w in 0..solve_worlds {
            let mut nodes = [0u64; 2];
            for (k, rules_burn) in [false, true].into_iter().enumerate() {
                let c = SearchConfig {
                    rules_robber_burn: rules_burn,
                    ..cfg
                };
                let mut mem = solver::SolveMem::new();
                let mut run = WorldSolve::default();
                let t = Instant::now();
                exact_world_row(
                    &view, &*worlds, &cands, w, dseed, &ctx, &c, None, &mut mem, &mut run,
                )
                .expect("not stopped");
                nodes[k] = run.stats.nodes;
                println!(
                    "  world {w} {}: robber {:?}, {} nodes, {:.1} s",
                    if rules_burn {
                        "rules burn "
                    } else {
                        "burn search"
                    },
                    run.robber,
                    run.stats.nodes,
                    t.elapsed().as_secs_f64()
                );
            }
            n_search += nodes[0];
            n_rules += nodes[1];
        }
    }
    println!(
        "share of worlds where another seat robs: {robbed} / {total} = {:.2}%",
        100.0 * robbed as f64 / total.max(1) as f64
    );
    if n_rules > 0 {
        println!(
            "nodes: burn searched {n_search}, rules burn {n_rules}, multiplier {:.3}",
            n_search as f64 / n_rules as f64
        );
    }
    Ok(())
}

/// SHA-256 (hex) of the running binary, for records (empty if it cannot be read).
fn binary_sha256() -> String {
    std::env::current_exe()
        .and_then(std::fs::read)
        .map(|b| equity::sha256::sha256_hex(&b))
        .unwrap_or_default()
}

/// `search risk-filter --shard IN.npy --out OUT.npy --equity TABLE --net O3.safetensors
/// --risk-model G2.json (--threshold T | --step K) [--threads N]`: keeps the trick-3 rows of a stored shard that G2
/// flags as riskiest at budget step `K` of the model (`search::risk_filter`), for the specialist
/// network S (docs/m3-hybrid/specialist.md). Prints the counts; writes the kept rows as a shard.
fn risk_filter_cmd(a: &Args) -> Result<(), String> {
    use search::risk_filter::{keep, score_rows};
    let (table, _) = load_equity_sha(a.req("--equity")?)?;
    let leaf: Arc<SharedLeaf> = Arc::new(table);
    let net_path = a.req("--net")?;
    let o = nn::Oracle::load(Path::new(net_path), leaf.clone())?;
    if o.trick != 3 {
        return Err(format!(
            "--net {net_path} is for trick {}, not trick 3",
            o.trick
        ));
    }
    let model_path = a.req("--risk-model")?;
    let model = nn::leafrisk::RiskModel::load(Path::new(model_path))?;
    if model.trick != 3 {
        return Err(format!(
            "--risk-model {model_path} is for trick {}, not 3",
            model.trick
        ));
    }
    if !model.net_sha256.is_empty() && model.net_sha256 != o.sha256 {
        return Err(format!(
            "--risk-model {model_path} was built for another network"
        ));
    }
    // The play rule's own threshold (`learned:MODEL:t=T`, as in play), or a budget step of the
    // model's table (`--step K`, `threshold_for_share`).
    let (threshold, step) = match (a.get("--threshold"), a.get("--step")) {
        (Some(t), None) => (
            t.parse::<i64>()
                .map_err(|_| format!("--threshold: not a number: {t}"))?,
            0,
        ),
        (None, Some(_)) => {
            let k = a.num("--step", 0)? as usize;
            let t = model
                .threshold_for_share(k)
                .ok_or("--step 0 keeps nothing: give a budget step from 1 up")?;
            (t, k)
        }
        _ => return Err("give exactly one of --threshold T and --step K".into()),
    };
    let threads = a.num("--threads", 1)?.max(1) as usize;
    let (in_path, out_path) = (a.req("--shard")?, a.req("--out")?);
    let rows =
        nn::shard::from_npy(&std::fs::read(in_path).map_err(|e| format!("{in_path}: {e}"))?)?;
    let scores = score_rows(&rows, threads, |r| {
        if r.trick != 3 {
            return Ok(None);
        }
        let (c, ctx) = nn::encode::decode(&r.features)
            .ok_or_else(|| format!("{in_path}: a row's features are not an encoding"))?;
        Ok(Some(model.score(&c, &ctx, &*leaf, &o.eval(&c, &ctx))))
    })?;
    let kept = keep(&rows, &scores, threshold);
    let v = rows.first().map_or(0, |r| r.outcomes.len());
    std::fs::write(out_path, nn::shard::to_npy(&kept, v))
        .map_err(|e| format!("{out_path}: {e}"))?;
    let t3 = scores.iter().filter(|s| s.is_some()).count();
    println!(
        "risk-filter: {} rows, {t3} trick-3 starts, step {step} threshold {threshold}: kept {} ({:.2}% of trick-3 starts) -> {out_path}",
        rows.len(),
        kept.len(),
        100.0 * kept.len() as f64 / t3.max(1) as f64
    );
    Ok(())
}

/// `search net-error --shard VAL.npy --net O3.safetensors --risk-model G2.json --equity TABLE
/// [--threshold T] [--threads N] [--out FILE]`: a network's value error against the exact values of
/// a stored trick-3 shard, on all its rows and on the rows G2 flags at the threshold (default: the
/// play rule's 277338260). Prints (and writes) JSON: the statistics of each set
/// (`search::net_error`; the held-out numbers of docs/m3-hybrid/specialist.md).
fn net_error_cmd(a: &Args) -> Result<(), String> {
    use search::net_error::{row_error, stats};
    use search::risk_filter::score_rows;
    let (table, table_sha) = load_equity_sha(a.req("--equity")?)?;
    let leaf: Arc<SharedLeaf> = Arc::new(table);
    let net_path = a.req("--net")?;
    let o = nn::Oracle::load(Path::new(net_path), leaf.clone())?;
    if o.trick != 3 {
        return Err(format!(
            "--net {net_path} is for trick {}, not trick 3",
            o.trick
        ));
    }
    let model_path = a.req("--risk-model")?;
    let model = nn::leafrisk::RiskModel::load(Path::new(model_path))?;
    if model.trick != 3 {
        return Err(format!(
            "--risk-model {model_path} is for trick {}, not 3",
            model.trick
        ));
    }
    if !model.net_sha256.is_empty() && model.net_sha256 != o.sha256 {
        return Err(format!(
            "--risk-model {model_path} was built for another network"
        ));
    }
    let threshold = a.get("--threshold").map_or(Ok(277_338_260i64), |t| {
        t.parse::<i64>()
            .map_err(|_| format!("--threshold: not a number: {t}"))
    })?;
    let threads = a.num("--threads", 1)?.max(1) as usize;
    let shard_path = a.req("--shard")?;
    let rows =
        nn::shard::from_npy(&std::fs::read(shard_path).map_err(|e| format!("{shard_path}: {e}"))?)?;
    let scores = score_rows(&rows, threads, |r| {
        if r.trick != 3 {
            return Ok(None);
        }
        let (c, ctx) = nn::encode::decode(&r.features)
            .ok_or_else(|| format!("{shard_path}: a row's features are not an encoding"))?;
        Ok(Some(model.score(&c, &ctx, &*leaf, &o.eval(&c, &ctx))))
    })?;
    let (mut all, mut flagged) = (Vec::new(), Vec::new());
    for (row, score) in rows.iter().zip(&scores) {
        let Some(score) = *score else { continue };
        let (c, ctx) =
            nn::encode::decode(&row.features).ok_or("a row's features are not an encoding")?;
        let (_, rotation) = nn::encode::encode(&c, &ctx, &*leaf);
        let e = row_error(row, o.eval(&c, &ctx), rotation % 4);
        all.push(e);
        if score >= threshold {
            flagged.push(e);
        }
    }
    let report = serde_json::json!({
        "shard": shard_path,
        "net_sha256": o.sha256,
        "equity_sha256": table_sha,
        "threshold": threshold,
        "all": stats(&all),
        "flagged": stats(&flagged),
    });
    let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    if let Some(out) = a.get("--out") {
        std::fs::write(out, &text).map_err(|e| format!("{out}: {e}"))?;
    }
    println!("{text}");
    Ok(())
}

/// `search risk-profile`: the pilot of a risk-guided harvest (`search::risk_harvest`): the
/// harvest's own search on `--decisions` decisions (a seed derived from `--seed`, never the
/// labels' decisions), every reached position scored by the leaf-risk model, cut into `--bands`
/// bands of equal count, and each band's acceptance for `--power` and `--floor`. Writes the
/// profile JSON that `search labels --risk-profile` reads; the same file for any thread count.
fn risk_profile_cmd(a: &Args) -> Result<(), String> {
    use search::oracle_labels::{
        Driver, GOLDEN_SEED, HARVEST_SHARES, HARVEST_STRATA, HarvestJob, parse_band_shares,
        parse_strata, position_schedule, tt_mb_for,
    };
    use search::risk_harvest::{
        RISK_BANDS, RISK_FLOOR_PPM, RISK_PILOT_DECISIONS, RiskProfile, build_profile, parse_power,
        parse_ppm, pilot_scores, pilot_seed,
    };
    let trick = a.num("--trick", 3)? as u8;
    if !(2..=5).contains(&trick) {
        return Err("--trick is 2..=5 (the start of that trick)".into());
    }
    let seed = a.num("--seed", 0)?;
    if seed == 0 || seed == GOLDEN_SEED {
        return Err("--seed: the labels' seed (not 0 or the golden seed 1)".into());
    }
    let threads = a.num("--threads", 1)?.max(1) as usize;
    let decisions = a.num("--decisions", RISK_PILOT_DECISIONS)?;
    let bands = a.num("--bands", RISK_BANDS as u64)? as usize;
    if decisions == 0 || !(1..=255).contains(&bands) {
        return Err("--decisions is at least 1 and --bands 1 to 255".into());
    }
    let power_q = parse_power(a.get("--power").unwrap_or("1"))?;
    let floor_ppm = match a.get("--floor") {
        None => RISK_FLOOR_PPM,
        Some(f) => parse_ppm(f).map_err(|e| format!("--floor {e}"))?,
    };
    let strata = match a.get("--harvest-strata") {
        None => HARVEST_STRATA.to_vec(),
        Some(x) => parse_strata(x).map_err(|e| format!("--harvest-strata: {e}"))?,
    };
    let shares = a.get("--harvest-positions").unwrap_or(HARVEST_SHARES);
    let weights = parse_band_shares(shares).map_err(|e| format!("--harvest-positions: {e}"))?;
    let tricks: Vec<u8> = match a.get("--harvest-tricks") {
        None => vec![trick - 1],
        Some(t) => t
            .split(',')
            .map(|x| {
                x.trim()
                    .parse::<u8>()
                    .ok()
                    .filter(|t| (1..trick).contains(t))
            })
            .collect::<Option<Vec<_>>>()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| format!("--harvest-tricks {t:?}: decision tricks before {trick}"))?,
    };
    let (table, _) = load_equity_sha(a.req("--equity")?)?;
    let leaf: Arc<SharedLeaf> = Arc::new(table);
    let net_path = a.req("--harvest-net")?;
    let o = nn::Oracle::load(Path::new(net_path), leaf.clone())?;
    if o.trick != trick {
        return Err(format!(
            "--harvest-net {net_path} was trained for trick {} positions, not trick {trick}",
            o.trick
        ));
    }
    let model_path = a.req("--harvest-risk")?;
    let model = nn::leafrisk::RiskModel::load(Path::new(model_path))?;
    if model.net_sha256 != o.sha256 {
        return Err(format!(
            "--harvest-risk {model_path} was made for the network {}, not {net_path} ({})",
            model.net_sha256, o.sha256
        ));
    }
    let ps = pilot_seed(seed);
    let job = HarvestJob {
        trick,
        count: 0,
        seed: ps,
        threads,
        cap: 1,
        tricks,
        schedule: position_schedule(weights),
        strata,
        race: None,
        keep: None,
        siblings: false,
        risk: None,
    };
    let t0 = Instant::now();
    let starts = collect_starts_with(a.num("--start-hands", 20_000)?, ps, threads);
    let scores = pilot_scores(
        &job,
        decisions,
        &model,
        &starts,
        &*leaf,
        Driver::Net(&o),
        tt_mb_for(trick),
        threads,
    );
    if scores.is_empty() {
        return Err("the pilot reached no position".into());
    }
    let (edges, counts, mean, accept) = build_profile(scores, bands, power_q, floor_ppm);
    let p = RiskProfile {
        model_sha256: model.sha256.clone(),
        net_sha256: o.sha256.clone(),
        power_q,
        floor_ppm,
        edges,
        counts,
        mean,
        accept,
        pilot_seed: ps,
        pilot_decisions: decisions,
    };
    let mut v = p.to_json();
    v["trick"] = serde_json::json!(trick);
    v["harvest_positions"] = serde_json::json!(shares);
    v["seconds"] = serde_json::json!((t0.elapsed().as_secs_f64() * 10.0).round() / 10.0);
    let out = a.req("--out")?;
    std::fs::write(
        out,
        serde_json::to_string_pretty(&v).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| format!("{out}: {e}"))?;
    eprintln!(
        "risk-profile: {} positions from {decisions} pilot decisions in {:.1} s; {} bands, power \
         {}, floor {floor_ppm} ppm; keeps {:.2}% of reached positions before the cap; target \
         shares (lowest risk first) {:?}",
        p.counts.iter().sum::<u64>(),
        t0.elapsed().as_secs_f64(),
        p.bands(),
        search::risk_harvest::power_text(power_q),
        p.yield_ppm() as f64 / 1e4,
        p.target_ppm()
    );
    Ok(())
}

/// `search labels`: a label shard for the oracle network O_K and its sidecar.
fn labels_cmd(a: &Args) -> Result<(), String> {
    use search::oracle_labels::{
        Driver, GOLDEN_SEED, HARVEST_CAP, HARVEST_SHARES, HARVEST_STRATA, HARVEST_TT_MB,
        HarvestJob, HarvestSchedule, HybridLabels, KEEP_CAP, LabelExec, LabelJob, Labeller,
        PredKeep, RACE_CAP, RaceKeep, Source, chunk_rows_for, default_max_decisions,
        parse_band_shares, parse_exact_share, parse_race_classes, parse_seat_weights, parse_strata,
        position_schedule, reheader_shard, seat_row_weights, tt_mb_for, write_harvest_shard_by,
        write_label_shard_by,
    };
    let trick = a.num("--trick", 4)? as u8;
    if !(2..=5).contains(&trick) {
        return Err("--trick is 2..=5 (the start of that trick)".into());
    }
    let seed: u64 = a
        .req("--seed")?
        .parse()
        .map_err(|_| "--seed: not a number".to_string())?;
    if seed == GOLDEN_SEED {
        return Err("--seed 1 is the golden sets' seed: use another".into());
    }
    let job = LabelJob {
        trick,
        count: a.num("--count", 1000)?,
        seed,
        source: Source::parse(a.req("--source")?)?,
        threads: a.num("--threads", 1)?.max(1) as usize,
    };
    let harvest_flags = [
        "--harvest-cap",
        "--harvest-positions",
        "--harvest-tricks",
        "--harvest-net",
        "--harvest-strata",
        "--harvest-race-classes",
        "--harvest-race-cap",
        "--harvest-keep",
        "--keep-cap",
        "--max-decisions",
        "--seat-weights",
        "--harvest-schedule",
        "--harvest-risk",
        "--risk-profile",
    ];
    // The rows' loss weights by their decision's seat band, written beside the shard
    // (`NAME.npy.rowweights`) before it is renamed into place.
    let seat_weights = match a.get("--seat-weights") {
        Some(w) if job.source == Source::Harvest => Some((w.to_string(), parse_seat_weights(w)?)),
        _ => None,
    };
    let siblings = job.source == Source::Siblings;
    if siblings {
        // A sibling run keeps its groups: the other harvests' selections do not apply.
        for f in [
            "--harvest-keep",
            "--keep-cap",
            "--harvest-race-classes",
            "--harvest-race-cap",
            "--seat-weights",
        ] {
            if a.has(f) {
                return Err(format!(
                    "{f} is not for --source siblings (its rows are whole groups)"
                ));
            }
        }
    }
    if a.has("--harvest-risk") || a.has("--risk-profile") {
        if !(a.has("--harvest-risk") && a.has("--risk-profile")) {
            return Err(
                "a risk-guided harvest needs both --harvest-risk MODEL.json and --risk-profile \
                 PROFILE.json (search risk-profile)"
                    .into(),
            );
        }
        for f in ["--harvest-keep", "--harvest-race-classes", "--seat-weights"] {
            if a.has(f) {
                return Err(format!(
                    "{f} and --harvest-risk: a harvest keeps by one rule only"
                ));
            }
        }
        if siblings || !a.has("--harvest-net") {
            return Err(
                "--harvest-risk scores the network's values: a harvest (--source harvest) driven \
                 by --harvest-net, the network the model was made for"
                    .into(),
            );
        }
    }
    let mut harvest = if matches!(job.source, Source::Harvest | Source::Siblings) {
        let keep = match a.get("--harvest-keep") {
            None => {
                if a.has("--keep-cap") {
                    return Err(
                        "--keep-cap is for a targeted harvest: give --harvest-keep PRED.json"
                            .into(),
                    );
                }
                None
            }
            Some(f) => {
                if a.has("--harvest-race-classes") {
                    return Err(
                        "--harvest-keep and --harvest-race-classes: a harvest keeps a predicate \
                         file's groups or race classes, not both"
                            .into(),
                    );
                }
                let preds = nn::predicate::Predicates::load(Path::new(f))
                    .map_err(|e| format!("--harvest-keep: {e}"))?;
                if preds.is_empty() {
                    return Err(format!("--harvest-keep {f}: the file has no groups"));
                }
                let cap = a.num("--keep-cap", u64::from(KEEP_CAP))?;
                if !(1..=u64::from(u32::MAX)).contains(&cap) {
                    return Err("--keep-cap is at least 1".into());
                }
                Some((
                    PredKeep {
                        preds: Arc::new(preds),
                        cap: cap as u32,
                    },
                    f.to_string(),
                ))
            }
        };
        let race = match a.get("--harvest-race-classes") {
            None => {
                if a.has("--harvest-race-cap") {
                    return Err("--harvest-race-cap is for a race harvest: give --harvest-race-classes (e.g. 2,4)"
                        .into());
                }
                None
            }
            Some(c) => {
                let classes =
                    parse_race_classes(c).map_err(|e| format!("--harvest-race-classes: {e}"))?;
                let cap = a.num("--harvest-race-cap", u64::from(RACE_CAP))?;
                if !(1..=u64::from(u32::MAX)).contains(&cap) {
                    return Err("--harvest-race-cap is at least 1".into());
                }
                Some(RaceKeep {
                    classes,
                    cap: cap as u32,
                })
            }
        };
        let cap = a.num("--harvest-cap", u64::from(HARVEST_CAP))?;
        if cap > u64::from(u32::MAX) || (cap == 0 && race.is_none() && keep.is_none()) {
            return Err(
                "--harvest-cap is at least 1 (0, keeping only the race classes or the groups, in \
                 a race or targeted harvest)"
                    .into(),
            );
        }
        let strata = match a.get("--harvest-strata") {
            None => HARVEST_STRATA.to_vec(),
            Some(x) => parse_strata(x).map_err(|e| format!("--harvest-strata: {e}"))?,
        };
        let shares = a.get("--harvest-positions").unwrap_or(HARVEST_SHARES);
        let weights = parse_band_shares(shares).map_err(|e| format!("--harvest-positions: {e}"))?;
        let tricks: Vec<u8> = match a.get("--harvest-tricks") {
            None => vec![trick - 1],
            Some(t) => t
                .split(',')
                .map(|x| {
                    x.trim()
                        .parse::<u8>()
                        .ok()
                        .filter(|t| (1..trick).contains(t))
                })
                .collect::<Option<Vec<_>>>()
                .filter(|v| !v.is_empty())
                .ok_or_else(|| {
                    let last = trick - 1;
                    format!(
                        "--harvest-tricks {t:?}: decision tricks from 1 to {last} (before the \
                         frontier), e.g. {last}"
                    )
                })?,
        };
        let total: u64 = weights.iter().sum();
        if total > 1 << 24 {
            return Err(format!(
                "--harvest-positions {shares:?}: the shares make a cycle of {total} decisions; use \
                 fewer decimals"
            ));
        }
        // The bound on decisions played: `--max-decisions N`, else for a targeted harvest
        // `default_max_decisions` (its rare groups could otherwise need millions of decisions);
        // none for other harvests unless asked.
        let max_decisions = match a.get("--max-decisions") {
            Some(_) => {
                let m = a.num("--max-decisions", 0)?;
                if m == 0 {
                    return Err("--max-decisions is at least 1".into());
                }
                Some(m)
            }
            None => keep
                .as_ref()
                .map(|(k, _)| default_max_decisions(job.count, k.cap)),
        };
        Some((
            HarvestJob {
                trick,
                count: job.count,
                seed,
                threads: job.threads,
                cap: cap as u32,
                tricks,
                schedule: position_schedule(weights),
                strata,
                race,
                keep: keep.as_ref().map(|(k, _)| k.clone()),
                siblings,
                // Set once the driving network is loaded (the model must be made for it).
                risk: None,
            },
            shares.to_string(),
            weights,
            keep.map(|(_, f)| f),
            max_decisions,
        ))
    } else {
        if let Some(f) = harvest_flags.iter().find(|f| a.has(f)) {
            return Err(format!("{f} is for --source harvest (or siblings) only"));
        }
        None
    };
    let out = PathBuf::from(a.req("--out")?);
    let chunk_rows = a.num("--chunk-rows", chunk_rows_for(trick))?.max(1);
    let t0 = Instant::now();
    let (table, table_sha) = load_equity_sha(a.req("--equity")?)?;
    let leaf: Arc<SharedLeaf> = Arc::new(table);
    let table_s = t0.elapsed().as_secs_f64();
    // The network that drives a harvest's search (`--harvest-net`): the integer O_k of an
    // earlier round, loaded as `search hybrid-check` loads it.
    let net = match a.get("--harvest-net") {
        Some(p) => {
            let o = nn::Oracle::load(Path::new(p), leaf.clone())?;
            if o.trick != trick {
                return Err(format!(
                    "--harvest-net {p} was trained for trick {} positions, not trick {trick}",
                    o.trick
                ));
            }
            eprintln!(
                "labels: harvest driven by the network {p} (sha256 {}, {} head, widths {:?}): \
                 only the sampled positions are solved exactly",
                o.sha256,
                o.head.name(),
                o.net().widths()
            );
            Some((o, p.to_string()))
        }
        None => None,
    };
    // The risk-guided harvest (`--harvest-risk`): the model and the profile, both for this
    // network.
    let risk_files = match (a.get("--harvest-risk"), a.get("--risk-profile"), &net) {
        (Some(m), Some(pf), Some((o, _))) => {
            let model = nn::leafrisk::RiskModel::load(Path::new(m))?;
            let text = std::fs::read_to_string(pf).map_err(|e| format!("{pf}: {e}"))?;
            let v: serde_json::Value =
                serde_json::from_str(&text).map_err(|e| format!("{pf}: {e}"))?;
            let profile = search::risk_harvest::RiskProfile::from_json(&v)
                .map_err(|e| format!("{pf}: {e}"))?;
            let keep = search::risk_harvest::RiskKeep::new(model, profile, &o.sha256)
                .map_err(|e| format!("--harvest-risk: {e}"))?;
            if let Some((h, ..)) = harvest.as_mut() {
                h.risk = Some(keep);
            }
            eprintln!("labels: risk-guided harvest: model {m}, profile {pf}");
            Some((m.to_string(), pf.to_string()))
        }
        _ => None,
    };
    let driver = net.as_ref().map_or(Driver::Exact, |(o, _)| Driver::Net(o));
    // Hybrid labels (`--label-net`): the network at the start of the next trick.
    let label_net = match a.get("--label-net") {
        Some(p) => {
            let o = nn::Oracle::load(Path::new(p), leaf.clone())?;
            if trick >= 5 || o.trick != trick + 1 {
                return Err(format!(
                    "--label-net {p} was trained for trick {} positions: hybrid labels at the \
                     start of trick {trick} need the network of trick {} (O_{})",
                    o.trick,
                    trick + 1,
                    trick + 1
                ));
            }
            eprintln!(
                "labels: hybrid labels: exact search through trick {trick}, the network {p} \
                 (sha256 {}, {} head, widths {:?}) at the start of trick {}",
                o.sha256,
                o.head.name(),
                o.net().widths(),
                trick + 1
            );
            Some((o, p.to_string()))
        }
        None => None,
    };
    let exact_ppm = match a.get("--exact-share") {
        None => 0,
        Some(_) if label_net.is_none() => {
            return Err(
                "--exact-share is the exact anchor of hybrid labels: give --label-net".into(),
            );
        }
        Some(x) => parse_exact_share(x).map_err(|e| format!("--exact-share {e}"))?,
    };
    // How the run is carried out (never changes a row): the exact label solves' table
    // (`--tt-mb`, the runner lowers it to fit the machine's memory) and how a harvest's work is
    // shared out over the threads (`--harvest-schedule`: `pooled`, the default, or `decision`,
    // before 2026-10-17, for comparisons).
    let mut exec = LabelExec::for_trick(trick);
    exec.tt_mb = a.num("--tt-mb", tt_mb_for(trick) as u64)?.max(1) as usize;
    exec.schedule = HarvestSchedule::parse(a.get("--harvest-schedule").unwrap_or("pooled"))
        .map_err(|e| format!("--harvest-schedule: {e}"))?;
    // Exact leaves of the hybrid labels (`--exact-leaves`, as `hybrid-check`'s): the frontier
    // positions the policy picks are solved exactly; `none` (the default) is the hybrid as before.
    let exact_leaves = match a.get("--exact-leaves") {
        None => {
            if a.has("--exact-leaf-tt-mb") {
                return Err("--exact-leaf-tt-mb is for --exact-leaves".into());
            }
            None
        }
        Some(_) if label_net.is_none() => {
            return Err(
                "--exact-leaves picks the hybrid labels' frontier positions to solve exactly: \
                 give --label-net (exact labels are exact everywhere)"
                    .into(),
            );
        }
        Some(text) => {
            let leaf_tt = a.num(
                "--exact-leaf-tt-mb",
                exec.tt_mb.min(tt_mb_for(trick + 1)) as u64,
            )? as usize;
            search::exact_leaves::LeafPolicy::parse(text, leaf_tt)?
        }
    };
    // The picked leaves' expert (`--defer-expert`, as `hybrid-check`'s): exact (the default) or
    // the deeper hybrid H_{trick+2} with its own exact leaves (learned deferral).
    let exact_leaves = match (exact_leaves, a.get("--defer-expert")) {
        (p, None) => p,
        (None, Some(_)) => {
            return Err("--defer-expert defers the leaves --exact-leaves picks: give them".into());
        }
        (Some(p), Some(e)) => Some(p.with_expert(search::exact_leaves::Expert::parse(
            e,
            trick + 1,
            &leaf,
            tt_mb_for(trick + 2),
        )?)),
    };
    if let (Some(p), Some((o, _))) = (&exact_leaves, &label_net) {
        p.check_for(trick + 1).map_err(|e| {
            format!("--exact-leaves: {e} (hybrid labels at trick {trick} have their leaves there)")
        })?;
        if let Some(m) = p.net_mismatch(&o.sha256) {
            return Err(format!("--exact-leaves: {m} (--label-net)"));
        }
    }
    if let Some(p) = &exact_leaves {
        eprintln!(
            "labels: hybrid labels: exact leaves {} (leaf tables {} MB)",
            p.describe(),
            p.tt_mb
        );
    }
    let labeller = match &label_net {
        Some((o, _)) => Labeller::Hybrid(HybridLabels {
            net: o,
            exact_ppm,
            exact_leaves: exact_leaves.as_ref(),
        }),
        None => Labeller::Exact,
    };
    let start_hands = a.num("--start-hands", 20_000)?;
    let starts = collect_starts_with(start_hands, seed, job.threads);
    eprintln!(
        "labels: trick {trick}, {} rows from {}, seed {seed}, {} threads (table and start states in {:.1} s: table {table_s:.1} s)",
        job.count,
        job.source.name(),
        job.threads,
        t0.elapsed().as_secs_f64()
    );
    let done = std::sync::atomic::AtomicUsize::new(0);
    let finished = AtomicBool::new(false);
    let t1 = Instant::now();
    let shard = std::thread::scope(|s| {
        let (done, finished) = (&done, &finished);
        s.spawn(move || {
            let mut last = Instant::now();
            while !finished.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(200));
                if last.elapsed() >= Duration::from_secs(20) {
                    last = Instant::now();
                    eprintln!(
                        "labels: {} of {} rows, {:.0} s",
                        (done.load(Ordering::Relaxed) as u64).min(job.count),
                        job.count,
                        t0.elapsed().as_secs_f64()
                    );
                }
            }
        });
        // Stops the progress thread however `label_rows` ends (a panic included).
        struct Finish<'a>(&'a AtomicBool);
        impl Drop for Finish<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        let _finish = Finish(finished);
        // The shard goes to a temporary name, renamed once whole: a stopped run leaves no
        // shard that looks finished.
        let mut part = out.as_os_str().to_owned();
        part.push(".part");
        let part = PathBuf::from(part);
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&part)
            .map_err(|e| format!("{}: {e}", part.display()))?;
        let mut summary = match &harvest {
            Some((h, _, _, _, max)) => write_harvest_shard_by(
                h,
                exec,
                *max,
                &starts,
                &*leaf,
                driver,
                labeller,
                Some(done),
                &mut file,
            ),
            None => write_label_shard_by(
                &job,
                exec,
                &starts,
                &*leaf,
                labeller,
                Some(done),
                chunk_rows,
                &mut file,
            ),
        }
        .map_err(|e| format!("{}: {e}", part.display()))?;
        if summary.rows < job.count {
            // Bounded: the header says the rows asked; it is set right for the rows found.
            summary.sha256 = reheader_shard(&mut file, summary.rows)
                .map_err(|e| format!("{}: {e}", part.display()))?;
        }
        drop(file);
        if let Some((_, w)) = &seat_weights {
            let weights = seat_row_weights(&summary.decisions, *w);
            if weights.len() as u64 != summary.rows {
                return Err(format!(
                    "--seat-weights: {} weights for {} rows",
                    weights.len(),
                    summary.rows
                ));
            }
            nn::rowweights::write(&out, &weights)?;
        }
        // A sibling run's groups (`NAME.npy.groups`), in place before the shard is renamed.
        if let Some(g) = &summary.siblings {
            if g.len() as u64 != summary.rows {
                return Err(format!("{} group rows for {} rows", g.len(), summary.rows));
            }
            nn::groups::write(&out, g)?;
        }
        std::fs::rename(&part, &out).map_err(|e| format!("{}: {e}", out.display()))?;
        Ok::<_, String>(summary)
    })?;
    let rows_s = t1.elapsed().as_secs_f64();
    let n_rows = shard.rows;
    let c = shard.counts;
    let (by_stratum, by_root, by_position) = (c.by_stratum, c.by_root_trick, c.by_position);
    let (retries, nodes) = (c.retries, c.nodes);
    let (hash_s, write_s) = (
        shard.hash_time.as_secs_f64(),
        shard.write_time.as_secs_f64(),
    );
    let (search_s, label_s) = (
        shard.search_time.as_secs_f64(),
        shard.label_time.as_secs_f64(),
    );
    let decisions = shard.decisions;
    let sibling_groups = shard.siblings;
    let secs = t0.elapsed().as_secs_f64();
    let mut side = serde_json::json!({
        "format": "41bots-label-shard-v1",
        "encoder": nn::encode::ENCODER,
        "n_features": nn::encode::N_FEATURES,
        "n_binary": nn::encode::N_BINARY,
        "n_dense": nn::encode::N_DENSE,
        "max_active": nn::encode::MAX_ACTIVE,
        "record": nn::shard::descr(nn::outcome::vocab(trick)),
        "outcomes": nn::outcome::vocab(trick),
        "value_order": "rotated: the leader is seat 0 (as the encoder's outputs)",
        "trick": trick,
        "source": job.source.name(),
        "count": job.count,
        "seed": seed,
        "start_hands": start_hands,
        "values_version": search::measure::VALUES_VERSION,
        "tie_rule": "first strictly best card in increasing card index (solver default)",
        "binary_sha256": binary_sha256(),
        "equity_sha256": table_sha,
        "shard_sha256": shard.sha256,
        "rows_by_stratum": by_stratum,
        "rows_by_race_class": c.by_class,
        "race_classes": nn::outcome::RACE_CLASS_NAMES,
        "rows_by_root_trick": &by_root[1..trick as usize],
        "replayed_lines": retries,
        "solver_nodes": nodes,
        "seconds": (secs * 10.0).round() / 10.0,
    });
    if job.source != Source::Rules {
        side["rows_by_position"] = serde_json::json!(&by_position[1..]);
    }
    // How the labels were made (docs/m3-hybrid/audit-loop.md §12): every exact shard says so,
    // a hybrid one names its network, its values version and its rows by label.
    side["label_source"] = serde_json::json!(labeller.name());
    if let Some((o, p)) = &label_net {
        side["values_version"] = serde_json::json!(search::oracle_labels::hybrid_values_version(
            trick,
            &o.sha256,
            exact_leaves.as_ref()
        ));
        side["label_net_sha256"] = serde_json::json!(o.sha256);
        side["label_net"] = serde_json::json!({
            "path": p, "trick": o.trick, "head": o.head.name(), "widths": o.net().widths(),
        });
        side["tie_rule"] = serde_json::json!(
            "first strictly best card in increasing card index (solver default), through trick \
             K; the network's value at the start of trick K+1"
        );
        side["exact_share_ppm"] = serde_json::json!(exact_ppm);
        side["rows_hybrid"] = serde_json::json!(c.hybrid);
        side["rows_exact"] = serde_json::json!(n_rows - c.hybrid);
        side["frontier_evals"] = serde_json::json!(c.evals);
        side["hybrid_label_tt_mb"] = serde_json::json!(search::oracle_labels::HYBRID_TT_MB);
        side["outcome_label"] = serde_json::json!(
            "the first possible outcome whose money vector is nearest the value (sum of absolute \
             differences); an exact value's own outcome"
        );
        if let Some(pol) = &exact_leaves {
            let l = c.leaves;
            side["label_exact_leaves"] = serde_json::json!(pol.describe());
            if let Some(m) = &pol.learned {
                side["label_leafrisk_sha256"] = serde_json::json!(m.model.sha256);
            }
            side["exact_leaves"] = serde_json::json!({
                "policy": pol.describe(),
                "what": "the hybrid searches' frontier positions the policy picks are solved \
                         exactly (search::exact_leaves, the labels' exact solver and tie rule) \
                         instead of being valued by the network; counts are distinct leaves per \
                         row's search, summed over the hybrid rows",
                "leaf_tt_mb": pol.tt_mb,
                "leaves": l.distinct,
                "leaves_exact": l.exact,
                "share_exact": if l.distinct > 0 { l.exact as f64 / l.distinct as f64 } else { 0.0 },
                "leaf_nodes": l.nodes,
                "learned": pol.learned.as_ref().map(|m| serde_json::json!({
                    "model_sha256": m.model.sha256, "net_sha256": m.model.net_sha256,
                    "trees": m.model.n_trees(), "t": m.t, "budget": m.budget,
                })),
            });
            eprintln!(
                "labels: exact leaves {}: {} of {} distinct leaves solved exactly ({} nodes)",
                pol.describe(),
                l.exact,
                l.distinct,
                l.nodes
            );
        }
        eprintln!(
            "labels: hybrid labels: {} rows by the hybrid ({} frontier evaluations), {} exact \
             (anchor share {exact_ppm} per million)",
            c.hybrid,
            c.evals,
            n_rows - c.hybrid
        );
    }
    if exec.tt_mb != tt_mb_for(trick) {
        // A table other than the trick's (`--tt-mb`): the solves' node counts are on it.
        side["label_tt_mb"] = serde_json::json!(exec.tt_mb);
    }
    if let Some((text, w)) = &seat_weights {
        let path = nn::rowweights::path_for(&out);
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        side["seat_weights"] = serde_json::json!({
            "weights": text, "fixed_256": w,
            "rowweights": path.file_name().map(|n| n.to_string_lossy().into_owned()),
            "rowweights_sha256": equity::sha256::sha256_hex(&bytes),
            "rows_by_position": &by_position[1..] });
    }
    if let Some(g) = &sibling_groups {
        let path = nn::groups::path_for(&out);
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let sm = nn::groups::summary(g);
        side["siblings"] = serde_json::json!({
            "groups": sm.groups,
            "groups_file": path.file_name().map(|n| n.to_string_lossy().into_owned()),
            "groups_sha256": equity::sha256::sha256_hex(&bytes),
            "groups_by_size": sm.by_size,
            "sizes": format!("index = rows in the group (the last entry counts {} or more)",
                             nn::groups::SIZES),
            "max_groups_per_decision": harvest.as_ref().map(|(h, ..)| h.cap),
            "what": "groups of frontier positions that one choice point separates                      (search::oracle_labels::sibling_plan): from each decision, every seat still                      to play in the trick is a choice point along the line the search expects;                      each legal card of that seat is followed to the start of the frontier trick                      along its principal line (every later card the best for its seat's team by                      the search's values: the network's at the frontier, or exact ones; the                      solver's tie rule), the distinct positions reached (canonical form) are the                      group, the seat's team the mover; groups of fewer than two positions are                      dropped; every position is labelled exactly",
            "mover": "an index into each row's own (rotated) value order",
            "last_group_cut": "the shard ends at the row count, so its last group may be cut",
        });
        eprintln!(
            "labels: siblings: {} groups in {} rows (groups by size 0-{}+: {:?})",
            sm.groups,
            sm.rows,
            nn::groups::SIZES,
            sm.by_size
        );
    }
    if let Some((h, shares, weights, keep_file, max)) = &harvest {
        // Rows written and decisions played: fewer rows than `count` only when the bound on
        // decisions stopped the harvest ("bounded").
        side["rows"] = serde_json::json!(n_rows);
        side["decisions_played"] = serde_json::json!(decisions.len());
        side["max_decisions"] = serde_json::json!(max);
        side["bounded"] = serde_json::json!(n_rows < job.count);
        if n_rows < job.count {
            eprintln!(
                "labels: bounded: {} decisions played (--max-decisions {}), {n_rows} of {} rows",
                decisions.len(),
                max.unwrap_or(0),
                job.count
            );
        }
        side["harvest"] = harvest_record(
            h,
            shares,
            weights,
            &decisions,
            HARVEST_TT_MB,
            net.as_ref().map(|(o, p)| (o, p.as_str())),
        );
        if let (Some(r), Some((m, pf))) = (&h.risk, &risk_files) {
            // What the risk profile asked for and what the shard holds, band by band.
            let mut rows = vec![0u64; r.profile.bands()];
            let mut reached = vec![0u64; r.profile.bands()];
            for d in &decisions {
                for (x, &n) in rows.iter_mut().zip(&d.risk_rows) {
                    *x += u64::from(n);
                }
                for (x, &n) in reached.iter_mut().zip(&d.risk_reached) {
                    *x += u64::from(n);
                }
            }
            let mut prof = r.profile.to_json();
            prof["file"] = serde_json::json!(pf);
            side["risk"] = serde_json::json!({
                "model": m, "model_sha256": r.model.sha256, "net_sha256": r.model.net_sha256,
                "profile": prof,
                "rows_by_band": rows,
                "reached_by_band": reached,
                "what": "a risk-guided harvest (search::risk_harvest): each reached position kept \
                         with its band's acceptance before labelling, then up to the cap per \
                         decision; rows_by_band counts the shard's rows, reached_by_band the \
                         decisions' distinct reached positions",
            });
            eprintln!("labels: risk-guided harvest: rows by band {rows:?}");
        }
        if h.siblings {
            side["harvest"]["what"] = serde_json::json!(
                "sibling groups from training decisions (see siblings): not a sample of the \
                 reached positions; cap_per_decision is the most groups a decision gives"
            );
        }
        // The label solves (the search's own table is `search` above).
        side["harvest"]["label_solves"] = serde_json::json!({
            "table_mb": exec.tt_mb,
            "schedule": exec.schedule.name(),
            "positions_a_unit": match exec.group {
                usize::MAX => serde_json::json!("all of a decision's"),
                n => serde_json::json!(n),
            },
            "what": match exec.schedule {
                HarvestSchedule::Pooled => "each sampled position solved exactly \
                    (solver::solve_fast_with) with its own table of table_mb a thread; the \
                    positions of positions_a_unit consecutive rows of one decision share one table \
                    (solver::SolveFamily); the units of every decision taken form one list \
                    labelled by every thread (search::ordered::pooled)",
                HarvestSchedule::PerDecision => "each decision's search and every one of its \
                    labels on one thread, its labels sharing one table of table_mb \
                    (solver::SolveFamily)",
            },
        });
        if let (Some(k), Some(t)) = (&h.keep, &shard.groups) {
            side["harvest"]["keep"] = keep_record(k, keep_file.as_deref(), h.cap, &decisions, t);
            eprintln!(
                "labels: targeted harvest ({} groups of {}, sha256 {}): {} of {} rows in a \
                 group; rows per group {:?}; {} of {} reached positions in a group",
                k.preds.len(),
                keep_file.as_deref().unwrap_or("?"),
                &k.preds.sha256[..12],
                t.rows_in_any,
                n_rows,
                t.rows,
                t.reached_in_any,
                t.reached_all
            );
        }
        let n = decisions.len().max(1) as f64;
        let distinct: u64 = decisions.iter().map(|d| u64::from(d.distinct)).sum();
        eprintln!(
            "labels: harvest: {} decisions (by position 1-8: {}), {:.1} distinct frontier \
             positions a decision ({:.1} frontier evaluations), {}; rows by race class 0-4: {:?}",
            decisions.len(),
            side["harvest"]["decisions_by_position"],
            distinct as f64 / n,
            decisions.iter().map(|d| d.evals).sum::<u64>() as f64 / n,
            match h.race {
                None if h.keep.is_some() => format!(
                    "{} decisions with more in-group positions than the keep cap",
                    decisions
                        .iter()
                        .filter(|d| h.keep.as_ref().is_some_and(|k| d.keep_distinct > k.cap))
                        .count()
                ),
                None if h.siblings => format!(
                    "{} sibling rows a decision (at most {} groups a decision)",
                    format_args!("{:.1}", n_rows as f64 / n),
                    h.cap
                ),
                None => format!(
                    "{} decisions capped at {}",
                    decisions.iter().filter(|d| d.distinct > h.cap).count(),
                    h.cap
                ),
                Some(r) => format!(
                    "{:.1} positions of race classes {:?} a decision, {} decisions capped at {} \
                     of them (other classes: at most {} a decision)",
                    decisions
                        .iter()
                        .map(|d| u64::from(d.race_distinct))
                        .sum::<u64>() as f64
                        / n,
                    r.list(),
                    decisions.iter().filter(|d| d.race_distinct > r.cap).count(),
                    r.cap,
                    h.cap
                ),
            },
            c.by_class
        );
        eprintln!(
            "labels: harvest: frontier driven by {}; searches {search_s:.1} thread-s, labelling \
             the sample {label_s:.1} thread-s ({:.2} thread-s per 1,000 rows in all)",
            match &net {
                Some((o, _)) => format!("the network {}", &o.sha256[..12]),
                None => "exact values".to_string(),
            },
            (search_s + label_s) * 1000.0 / n_rows.max(1) as f64
        );
    }
    let mut side_path = out.as_os_str().to_owned();
    side_path.push(".json");
    std::fs::write(
        &side_path,
        serde_json::to_string_pretty(&side).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| e.to_string())?;
    eprintln!(
        "labels: {} rows made, hashed and written in {:.1} s ({:.3} ms a row on {} threads; \
         hashing {:.1} s and writing {:.1} s alongside; {secs:.1} s in all); shard sha256 {}",
        n_rows,
        rows_s,
        rows_s * 1000.0 / n_rows.max(1) as f64,
        job.threads,
        hash_s,
        write_s,
        side["shard_sha256"].as_str().unwrap_or("")
    );
    Ok(())
}

/// The sidecar's `harvest.keep` record of a targeted harvest: the predicate file (path and
/// SHA-256), its groups with their conditions, the caps, and the rows and reached positions per
/// group.
fn keep_record(
    k: &search::oracle_labels::PredKeep,
    file: Option<&str>,
    other_cap: u32,
    ds: &[search::oracle_labels::HarvestDecision],
    t: &search::oracle_labels::GroupTally,
) -> serde_json::Value {
    let groups: Vec<serde_json::Value> = k
        .preds
        .groups
        .iter()
        .enumerate()
        .map(|(g, x)| {
            serde_json::json!({
                "name": x.name,
                "conditions": x.conditions_text(),
                "rows": t.rows[g],
                "reached": t.reached[g],
                "audit": x.meta,
            })
        })
        .collect();
    serde_json::json!({
        "what": "frontier positions inside any group of the predicate file (nn::predicate, \
                 decided on the position's encoded features) are a seeded sample of at most \
                 cap_per_decision of them, the others of at most other_cap_per_decision (drawn \
                 after them on the same stream); a position in several groups counts in each",
        "predicates_file": file,
        "predicates_sha256": k.preds.sha256,
        "cap_per_decision": k.cap,
        "other_cap_per_decision": other_cap,
        "groups": groups,
        "rows_in_any_group": t.rows_in_any,
        "rows_in_no_group": ds.iter().map(|d| u64::from(d.rows)).sum::<u64>() - t.rows_in_any,
        "reached_in_any_group": t.reached_in_any,
        "reached": t.reached_all,
        "decisions_keep_capped": ds.iter().filter(|d| d.keep_distinct > k.cap).count(),
    })
}

/// The sidecar's `harvest` record: the settings, the decisions' counts by position, stratum and
/// trick, and the distribution of the distinct frontier positions per decision.
fn harvest_record(
    h: &search::oracle_labels::HarvestJob,
    shares: &str,
    weights: &[u64; 8],
    ds: &[search::oracle_labels::HarvestDecision],
    tt_mb: usize,
    net: Option<(&nn::Oracle, &str)>,
) -> serde_json::Value {
    let mut by_pos = [0u64; 8];
    let mut rows_by_pos = [0u64; 8];
    let mut distinct_by_pos = [0u64; 8];
    let mut evals_by_pos = [0u64; 8];
    let mut by_stratum = [0u64; 4];
    let mut by_trick = [0u64; 6];
    let mut distinct: Vec<u32> = Vec::with_capacity(ds.len());
    for d in ds {
        let p = usize::from(d.position - 1);
        by_pos[p] += 1;
        rows_by_pos[p] += u64::from(d.rows);
        distinct_by_pos[p] += u64::from(d.distinct);
        evals_by_pos[p] += d.evals;
        by_stratum[usize::from(d.stratum - 1)] += 1;
        by_trick[usize::from(d.trick)] += 1;
        distinct.push(d.distinct);
    }
    let mean_by_pos = |tot: &[u64; 8]| -> Vec<f64> {
        tot.iter()
            .zip(&by_pos)
            .map(|(&t, &n)| (t as f64 / n.max(1) as f64 * 10.0).round() / 10.0)
            .collect()
    };
    distinct.sort_unstable();
    let q = |f: f64| {
        distinct
            .get(((distinct.len().max(1) - 1) as f64 * f).round() as usize)
            .copied()
            .unwrap_or(0)
    };
    let n = ds.len().max(1) as f64;
    let race = h.race.map(|r| {
        serde_json::json!({
            "classes": r.list(),
            "cap_per_decision": r.cap,
            "other_classes_cap_per_decision": h.cap,
            "what": "frontier positions of the race classes (nn::outcome::race_class) are a \
                     seeded sample of at most cap_per_decision of them, the other classes' of at \
                     most other_classes_cap_per_decision (drawn after them on the same stream)",
            "race_positions_reached": ds.iter().map(|d| u64::from(d.race_distinct)).sum::<u64>(),
            "decisions_race_capped": ds.iter().filter(|d| d.race_distinct > r.cap).count(),
        })
    });
    let shares_given: Vec<f64> = shares
        .split(',')
        .map(|x| x.trim().parse().unwrap_or(f64::NAN))
        .collect();
    let (what, frontier_values, driver) = match net {
        None => (
            "frontier positions (start of the trick) that the hybrid's search from training \
             decisions reaches for every candidate, with exact values at the frontier (H_k with a \
             perfect network), deduplicated by canonical form within a decision, then a seeded \
             sample of at most cap_per_decision of them, kept in the order reached",
            "exact (solver::solve_fast_with of the reached position, the label's own value)"
                .to_string(),
            serde_json::json!({ "kind": "exact" }),
        ),
        Some((o, path)) => (
            "frontier positions (start of the trick) that the hybrid's search from training \
             decisions reaches for every candidate, with the network's values at the frontier \
             (H_k with that network, as the hybrid runs), deduplicated by canonical form within \
             a decision, then a seeded sample of at most cap_per_decision of them, kept in the \
             order reached; only the sampled positions are solved exactly, for their labels",
            format!(
                "the network's (net sha256 {}: nn::Oracle, integer inference with its memo) during \
                 the search; each sampled position then solved exactly (solver::solve_fast_with, \
                 the label's value)",
                o.sha256
            ),
            serde_json::json!({
                "kind": "net",
                "drove_the_search": true,
                "net_sha256": o.sha256,
                "net_file": Path::new(path).file_name().map(|f| f.to_string_lossy().into_owned()),
                "net_trick": o.trick,
                "net_head": o.head.name(),
                "net_widths": o.net().widths(),
                "net_meta": o.meta,
            }),
        ),
    };
    serde_json::json!({
        "what": what,
        "decision_generator": "search::measure::build_decision (the yardstick's): stratum strata[j mod s] \
                               and decider position schedule[(j div s) mod position_cycle], s = the number \
                               of strata (all four: stratum j mod 4 + 1, as the yardstick), decision trick \
                               uniform over decision_tricks",
        "strata": h.strata,
        "race": race,
        "decision_seed": "Rng::derive(seed, [NN_LABEL, SOURCE_HARVEST, trick, j]); the same stream \
                          then draws the world (belief sampler) and the sample",
        "search": format!("pimc::exact_world, sequential solver, table {tt_mb} MB, frontier at the start of trick {}", h.trick),
        "frontier_values": frontier_values,
        "frontier_driver": driver,
        "cap_per_decision": h.cap,
        "decision_tricks": h.tricks,
        "position_shares": { "bands": ["1-2", "3-4", "5-8"], "given": shares, "shares": shares_given },
        "position_weights": weights,
        "position_cycle": h.schedule.len(),
        "decisions": ds.len(),
        "decisions_by_position": by_pos,
        "rows_by_position": rows_by_pos,
        "decisions_by_stratum": by_stratum,
        "decisions_by_trick": &by_trick[1..usize::from(h.trick)],
        "decisions_capped": ds.iter().filter(|d| d.distinct > h.cap && h.race.is_none() && h.keep.is_none()).count(),
        "frontier_evaluations": ds.iter().map(|d| d.evals).sum::<u64>(),
        "distinct_positions": distinct.iter().map(|&x| u64::from(x)).sum::<u64>(),
        "distinct_mean_by_position": mean_by_pos(&distinct_by_pos),
        "evaluations_mean_by_position": mean_by_pos(&evals_by_pos),
        "distinct_per_decision": {
            "mean": (distinct.iter().map(|&x| f64::from(x)).sum::<f64>() / n * 10.0).round() / 10.0,
            "min": q(0.0), "p10": q(0.1), "median": q(0.5), "p90": q(0.9), "max": q(1.0),
        },
        "search_nodes": ds.iter().map(|d| d.search_nodes).sum::<u64>(),
    })
}

/// `search hybrid-check`: H_K against exact on the same worlds (docs/M3-HYBRID-SPEC.md §3).
fn hybrid_check(a: &Args) -> Result<(), String> {
    use search::hybrid::{CheckJob, CheckProgress, Reference, report};
    use search::measure::FrontierFp;
    let std_pct = a.get("--standard").unwrap_or("0.01").to_string();
    let standard = search::hybrid::parse_standard(&std_pct)?;
    let std_txt = format!("{}%", std_pct.trim());
    let trick = a.num("--trick", 3)? as u8;
    let cut = a.num("--cut", u64::from(trick) + 1)? as u8;
    if !(2..=5).contains(&cut) || trick == 0 || trick >= cut {
        return Err(format!(
            "--cut {cut} must be 2..=5 and after --trick {trick} (and --trick at least 1)"
        ));
    }
    // The reference: exact (the golden sets) or silver (a deeper hybrid, docs/m3-hybrid/
    // silver.md). Neither stands in for the other unless asked.
    let silver = match a.get("--reference").unwrap_or("exact") {
        "exact" => false,
        "silver" => true,
        r => return Err(format!("--reference {r}: exact or silver")),
    };
    if !silver {
        for k in ["--reference-net", "--reference-cut", "--allow-exact"] {
            if a.has(k) {
                return Err(format!("{k} is for --reference silver"));
            }
        }
    }
    if !a.has("--race-groups") && (a.has("--race-share") || a.has("--min-group")) {
        return Err("--race-share and --min-group are for --race-groups".into());
    }
    let seed = a.num("--seed", 1)?;
    let t0 = Instant::now();
    let (table, table_sha) = load_equity_sha(a.req("--equity")?)?;
    let leaf: Arc<SharedLeaf> = Arc::new(table);
    // One network, or several (`--net A,B` or `--net A --net B`): an ensemble valued as the
    // integer mean of its members (`nn::ensemble`); one network is exactly that network.
    let net_paths: Vec<String> = a
        .all("--net")
        .iter()
        .flat_map(|v| v.split(','))
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect();
    if net_paths.is_empty() {
        return Err("missing --net".into());
    }
    let mut members = Vec::new();
    for p in &net_paths {
        let o = nn::Oracle::load(Path::new(p), leaf.clone())?;
        if o.trick != cut {
            return Err(format!(
                "{p} was trained for trick {} positions, not trick {cut}",
                o.trick
            ));
        }
        members.push(o);
    }
    let ensemble = nn::Ensemble::new(members)?;
    let oracle = &ensemble.members()[0];
    let net_path = net_paths[0].as_str();
    let n_nets = net_paths.len();
    if n_nets > 1 && (a.has("--saved") || a.has("--explain")) {
        return Err(
            "several networks (an ensemble) are for the golden-set form of the check, not \
             --saved or --explain"
                .into(),
        );
    }
    // The optimizer's-curse test (`search::curse`): the network itself as the truth, the player
    // the same search with seeded zero-mean noise at each leaf.
    let truth_net = match a.get("--truth").unwrap_or("exact") {
        "exact" => false,
        "net" => true,
        t => return Err(format!("--truth {t}: exact or net")),
    };
    let noise = if truth_net {
        for k in [
            "--worlds",
            "--reference",
            "--saved",
            "--explain",
            "--recompute",
        ] {
            if a.has(k) {
                return Err(format!(
                    "{k} is not for --truth net (the truth is the network itself, recomputed)"
                ));
            }
        }
        let text = a
            .get("--noise-sd-by-class")
            .ok_or("--truth net needs --noise-sd-by-class (five percentages, race classes 0-4)")?;
        Some(search::curse::NoiseSpec::parse(
            text,
            a.num("--noise-seed", 1)?,
        )?)
    } else {
        if a.has("--noise-sd-by-class") || a.has("--noise-seed") {
            return Err("--noise-sd-by-class and --noise-seed are for --truth net".into());
        }
        None
    };
    // Exact leaves (`search::exact_leaves`): the frontier positions the policy picks are solved
    // exactly instead of being valued by the network; `none` (the default) is the hybrid as
    // before, byte for byte.
    // A learned term's model must be one made for this cut's leaves; `--defer-expert` hands the
    // picked leaves to a deeper hybrid instead of the exact solver (learned deferral).
    let exact_leaves = leaf_policy_arg(a, "", cut, &leaf)?;
    if let Some(m) = exact_leaves
        .as_ref()
        .and_then(|p| p.net_mismatch(&ensemble.id()))
    {
        eprintln!("warning: --exact-leaves: {m}");
    }
    if exact_leaves.is_some() {
        for k in ["--saved", "--explain", "--truth"] {
            if a.has(k) {
                return Err(format!(
                    "--exact-leaves is for the golden-set form of the check against exact or \
                     silver values, not {k}"
                ));
            }
        }
    }
    // The silver reference's network and fingerprint (its records must carry exactly this).
    let reference = if silver {
        if a.has("--saved") || a.has("--explain") {
            return Err(
                "--reference silver is for the golden-set form of the check (not --saved or \
                 --explain): the saved trick-1 worlds and the explanation are exact"
                    .into(),
            );
        }
        let p = a.req("--reference-net")?;
        let o = nn::Oracle::load(Path::new(p), leaf.clone())?;
        let rc = a.num("--reference-cut", u64::from(o.trick))? as u8;
        if o.trick != rc {
            return Err(format!(
                "--reference-net {p} was trained for trick {} positions, not trick {rc}",
                o.trick
            ));
        }
        if rc <= cut {
            return Err(format!(
                "a silver reference must look deeper than the hybrid it checks: H_{rc} against \
                 H_{cut} (--reference-cut after --cut)"
            ));
        }
        // The reference's own exact leaves (a silver reference "H_3 with learned exact leaves").
        let rl = leaf_policy_arg(a, "reference-", rc, &leaf)?;
        if let Some(m) = rl.as_ref().and_then(|l| l.net_mismatch(&o.sha256)) {
            return Err(format!("--reference-exact-leaves: {m}"));
        }
        let fp = FrontierFp {
            cut: rc,
            net_sha256: o.sha256.clone(),
            encoder: nn::encode::ENCODER.to_string(),
            exact_leaves: rl.as_ref().map(FrontierFp::leaves_text),
        };
        Some((o, p.to_string(), fp, rl))
    } else {
        for k in [
            "--reference-exact-leaves",
            "--reference-exact-leaf-tt-mb",
            "--reference-defer-expert",
        ] {
            if a.has(k) {
                return Err(format!("{k} is for --reference silver"));
            }
        }
        None
    };
    let mut ds: Vec<DecisionSpec> = match a.get("--decisions") {
        Some(f) => read_lines::<DecisionSpec>(f)?
            .into_iter()
            .filter(|d| d.trick == trick)
            .collect(),
        None => {
            let threads = a.num("--threads", 1)?.max(1) as usize;
            let starts = collect_starts_with(a.num("--start-hands", 20_000)?, seed, threads);
            build_decisions(trick, a.num("--count", 32)?, seed, &starts)
        }
    };
    let (first, last) = (a.num("--first", 0)?, a.num("--last", u64::MAX)?);
    let ids = a.get("--ids").map(parse_ids).transpose()?;
    let selected =
        |id: u64| id >= first && id <= last && ids.as_ref().is_none_or(|i| i.contains(&id));
    ds.retain(|d| selected(d.id));
    if ds.is_empty() {
        return Err(format!("no decisions at trick {trick}"));
    }
    let tt_mb = a.num("--tt-mb", 64)? as usize;
    let mut job = CheckJob {
        cut,
        seed,
        label: a.num("--label", 1)?,
        to: a.num("--to", 320)?,
        threads: a.num("--threads", 1)?.max(1) as usize,
        tt_mb,
        verify_pct: float(a, "--verify-pct", if a.has("--saved") { 0.0 } else { 1.0 })?,
        recompute: a.has("--recompute"),
        rules_robber_burn: false,
        exact_solve_threads: a.num("--verify-solve-threads", 1)?.max(1) as usize,
        exact_tt_mb: a.num("--verify-tt-mb", tt_mb as u64)? as usize,
        noise: noise.clone(),
        exact_leaves: exact_leaves.clone(),
    };
    let src = world_source(a)?;
    let ctx = Ctx {
        leaf: &*leaf,
        source: &*src,
        rules: engine::game::Rules::HOUSE,
    };
    let boots = a.num("--boot", 1000)?;
    let boot_seed = a.num("--boot-seed", 1)?;
    let slice_preds = match a.get("--slices") {
        None => {
            if a.has("--slice-share") {
                return Err("--slice-share is for --slices PRED.json".into());
            }
            None
        }
        Some(f) => {
            if a.has("--explain") {
                return Err("--slices is for the check, not --explain".into());
            }
            let p = nn::predicate::Predicates::load(Path::new(f))?;
            if p.is_empty() {
                return Err(format!("--slices {f}: the file has no groups"));
            }
            Some((p, f.to_string()))
        }
    };
    let share_text = a.get("--slice-share").unwrap_or("0.5");
    let share =
        nn::predicate::parse_share(share_text).map_err(|e| format!("--slice-share: {e}"))?;
    let race = a.has("--race-groups");
    let race_share_text = a.get("--race-share").unwrap_or("0.5");
    let race_share =
        nn::predicate::parse_share(race_share_text).map_err(|e| format!("--race-share: {e}"))?;
    let min_group = a.num("--min-group", 30)? as usize;
    if a.has("--saved") {
        if race {
            return Err(
                "--race-groups is for the golden-set form of the check, not --saved".into(),
            );
        }
        return hybrid_check_saved(
            a,
            SavedRun {
                trick,
                ds: &ds,
                selected: &selected,
                job,
                ctx: &ctx,
                oracle,
                net_path,
                table_sha: &table_sha,
                boots,
                boot_seed,
                t0,
                slices: slice_preds
                    .as_ref()
                    .map(|(p, f)| (p, f.as_str(), share_text, share)),
            },
        );
    }
    // Stored reference values: exact records (the golden sets) or the silver reference's own
    // hybrid records; never one for the other unless allowed (`--allow-exact`: exact records stand
    // in for the silver reference, the anchor decisions).
    let allow_exact = a.has("--allow-exact");
    let stored: BTreeMap<u64, WorldRows> = if a.has("--worlds") {
        let paths: Vec<&Path> = a.all("--worlds").into_iter().map(Path::new).collect();
        let (rows, warnings) = read_worlds(&paths)?;
        for w in warnings {
            eprintln!("warning: {w}");
        }
        search::hybrid::stored_reference(
            rows,
            trick,
            &selected,
            reference.as_ref().map(|r| &r.2),
            allow_exact,
            &a.all("--worlds").join(", "),
        )?
    } else {
        BTreeMap::new()
    };
    if a.has("--worlds")
        && let Some(d) = ds.iter().find(|d| !stored.contains_key(&d.id))
    {
        return Err(format!(
            "decision {} has no stored {} worlds",
            d.id,
            if silver { "reference" } else { "exact" }
        ));
    }
    if src.name() != "rejection"
        && let Some(r) = stored.values().find(|r| r.log2_worlds.is_none())
    {
        return Err(format!(
            "decision {}: the stored worlds have no world count, so they were drawn by the \
             rejection sampler (the default before the belief sampler): pass --source rejection",
            r.id
        ));
    }
    if let Some(ids) = a.get("--explain") {
        return hybrid_explain(a, ids, &ds, &stored, &job, &ctx, oracle);
    }
    // The network's description: one, or the ensemble's members.
    let net_desc = if n_nets == 1 {
        format!("net {} {:?}", &oracle.sha256[..12], oracle.net().widths())
    } else {
        format!(
            "an ensemble of {n_nets} nets (integer mean): {}",
            ensemble
                .members()
                .iter()
                .map(|o| format!("{} {:?}", &o.sha256[..12], o.net().widths()))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let truth_fp = FrontierFp {
        cut,
        net_sha256: ensemble.id(),
        encoder: nn::encode::ENCODER.to_string(),
        exact_leaves: None,
    };
    let n_exact_stored = stored.values().filter(|r| r.method == "exact").count();
    let ref_name = match &reference {
        None if truth_net => format!(
            "the network itself (H_{cut} with its noise-free values; the player adds noise of SD \
             {}% by race class 0-4, seed {}: the optimizer's-curse test)",
            noise.as_ref().map_or(String::new(), |n| n.sd_pct.join("/")),
            noise.as_ref().map_or(0, |n| n.seed)
        ),
        None => "exact".to_string(),
        Some((o, _, fp, _)) => format!(
            "silver (H_{}, net {} {:?}{}){}",
            fp.cut,
            &fp.net_sha256[..12],
            o.net().widths(),
            fp.exact_leaves
                .as_ref()
                .map_or(String::new(), |l| format!(", exact leaves {l}")),
            if n_exact_stored > 0 {
                format!(", exact for the {n_exact_stored} decisions stored exact")
            } else {
                String::new()
            }
        ),
    };
    let ref_values = if silver || truth_net {
        "reference"
    } else {
        "exact"
    };
    let exact_from = if stored.is_empty() || job.recompute {
        "recomputed by this binary".to_string()
    } else {
        format!(
            "stored ({}), {}% sample recomputed",
            a.all("--worlds").join(", "),
            job.verify_pct
        )
    };
    let net_desc = match &exact_leaves {
        None => net_desc,
        Some(p) => format!(
            "{net_desc}, exact leaves {} (leaf tables {} MB)",
            p.describe(),
            p.tt_mb
        ),
    };
    eprintln!(
        "hybrid-check: {} decisions at trick {trick}, frontier at trick {cut}, {net_desc}, {} worlds each (seed {seed}, label {}, {} source), {} threads; against {ref_name}; {ref_values} values {exact_from}",
        ds.len(),
        job.to,
        job.label,
        src.name(),
        job.threads
    );
    // The curse test's calibration (`--noise-calibrate N`): on a sample of N decisions (every
    // k-th) and their first `--noise-calibrate-worlds` worlds, the SDs asked for are rescaled
    // round by round until the realised SD per race class matches the target (the SDs given).
    let mut calibration = Vec::new();
    let n_cal = a.num("--noise-calibrate", 0)? as usize;
    if n_cal > 0 && !truth_net {
        return Err("--noise-calibrate is for --truth net".into());
    }
    if let Some(spec) = job.noise.clone().filter(|s| n_cal > 0 && !s.is_zero()) {
        let stride = (ds.len() / n_cal).max(1);
        let sample: Vec<DecisionSpec> = ds.iter().step_by(stride).take(n_cal).cloned().collect();
        let mut cjob = job.clone();
        cjob.to = a.num("--noise-calibrate-worlds", 8)?.clamp(1, job.to);
        let rounds = a.num("--noise-calibrate-rounds", 4)?.max(1);
        let mut cur = spec;
        for r in 0..rounds {
            cjob.noise = Some(cur.clone());
            let run = search::hybrid::run_check_ref(
                &sample,
                &BTreeMap::new(),
                &cjob,
                &ctx,
                &ensemble,
                None,
                Some((
                    Reference {
                        cut,
                        net: &ensemble as &dyn solver::frontier::Frontier,
                        leaves: None,
                    },
                    &truth_fp,
                )),
                false,
                None,
            )?;
            let st = run.noise.unwrap_or_default();
            let rep = st.report(&cur);
            eprintln!(
                "noise calibration round {}: {} decisions x {} worlds; class: asked -> realised (target) {}",
                r + 1,
                sample.len(),
                cjob.to,
                rep.classes
                    .iter()
                    .map(|c| format!(
                        "{}: {:.4}% -> {:.4}% ({:.4}%)",
                        c.class, c.asked_sd_pct, c.realised_sd_pct, c.target_sd_pct
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            calibration.push(serde_json::json!({ "round": r + 1, "noise": rep }));
            cur = cur.calibrated(&st);
        }
        eprintln!(
            "noise calibrated: asked SDs {} (%, race classes 0-4)",
            cur.sd
                .iter()
                .map(|&x| format!("{:.4}", x as f64 / equity::ONE as f64 * 100.0))
                .collect::<Vec<_>>()
                .join(", ")
        );
        job.noise = Some(cur);
    }
    let progress_path = a.get("--progress").map(PathBuf::from);
    let prog = CheckProgress::default();
    let finished = AtomicBool::new(false);
    let t1 = Instant::now();
    let run = std::thread::scope(|s| {
        if let Some(p) = &progress_path {
            let (prog, finished) = (&prog, &finished);
            let threads = job.threads as u64;
            let ex_workers = (job.threads / job.exact_solve_threads.max(1)).max(1) as u64;
            s.spawn(move || {
                while !finished.load(Ordering::Relaxed) {
                    let ld = |x: &std::sync::atomic::AtomicU64| x.load(Ordering::Relaxed);
                    let (ed, et) = (ld(&prog.exact_done), ld(&prog.exact_total));
                    let (hd, ht) = (ld(&prog.hybrid_done), ld(&prog.hybrid_total));
                    // The finished worlds' list is for the saved form's log; not kept here.
                    prog.finished
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .clear();
                    write_atomic(
                        p,
                        &serde_json::json!({
                            "exact_done": ed, "exact_total": et,
                            "worlds_done": hd, "worlds_total": ht,
                            "tail": if ed < et { et - ed < ex_workers } else { ht - hd < threads },
                            "elapsed_s": t1.elapsed().as_secs_f64(),
                        }),
                    );
                    for _ in 0..20 {
                        if finished.load(Ordering::Relaxed) {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    }
                }
            });
        }
        struct Finish<'a>(&'a AtomicBool);
        impl Drop for Finish<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        let _finish = Finish(&finished);
        search::hybrid::run_check_ref(
            &ds,
            &stored,
            &job,
            &ctx,
            &ensemble,
            slice_preds.as_ref().map(|(p, _)| p),
            match &reference {
                Some((o, _, fp, rl)) => Some((
                    Reference {
                        cut: fp.cut,
                        net: o as &dyn solver::frontier::Frontier,
                        leaves: rl.as_ref(),
                    },
                    fp,
                )),
                // The curse test's truth: the same search with the same network, no noise.
                None if truth_net => Some((
                    Reference {
                        cut,
                        net: &ensemble as &dyn solver::frontier::Frontier,
                        leaves: None,
                    },
                    &truth_fp,
                )),
                None => None,
            },
            race,
            progress_path.as_ref().map(|_| &prog),
        )
    })?;
    let wall = t1.elapsed().as_secs_f64();
    if let Some(p) = &progress_path {
        write_atomic(
            p,
            &serde_json::json!({
                "exact_done": prog.exact_done.load(Ordering::Relaxed),
                "exact_total": prog.exact_total.load(Ordering::Relaxed),
                "worlds_done": prog.hybrid_done.load(Ordering::Relaxed),
                "worlds_total": prog.hybrid_total.load(Ordering::Relaxed),
                "tail": false, "elapsed_s": wall, "finished": true,
            }),
        );
    }
    if run.verified.1 > 0 {
        return Err(format!(
            "{} of {} recomputed worlds differ from the stored {ref_values} values: recompute them \
             (--recompute, or off this VM for trick 2 and earlier)",
            run.verified.1, run.verified.0
        ));
    }
    let mut groups = report(&run, boots, boot_seed);
    for (_, g) in &mut groups {
        search::hybrid::judge(g, standard);
    }
    let n_silver = run.decisions.iter().filter(|d| d.silver).count();
    println!(
        "H_{cut} vs {ref_name}, trick {trick}: {} decisions x {} worlds (seed {seed}, label {}, {} source); {net_desc}; {ref_values} values {exact_from}{}",
        run.decisions.len(),
        job.to,
        job.label,
        src.name(),
        if run.verified.0 > 0 {
            format!(" ({} worlds verified byte-identical)", run.verified.0)
        } else {
            String::new()
        }
    );
    if silver {
        println!(
            "(a silver reference: the loss is measured against H_{} with its network, not \
             against exact search; the true loss is at most this loss plus the reference's own \
             error, docs/m3-hybrid/silver.md; {n_silver} decisions against the silver values, \
             {} against exact ones)",
            reference.as_ref().map_or(0, |r| r.2.cut),
            run.decisions.len() - n_silver
        );
    }
    let noise_report = job
        .noise
        .as_ref()
        .zip(run.noise.as_ref())
        .map(|(spec, st)| st.report(spec));
    if let Some(nr) = &noise_report {
        println!(
            "(the optimizer's-curse test: the truth is H_{cut} with the network's own values, the \
             player the same search with zero-mean noise at each leaf; any loss or bias below \
             comes from the search's choices over noisy values alone)"
        );
        print_noise(nr);
    }
    print_groups(&groups, &std_txt);
    let sl = slice_preds.as_ref().map(|(p, _)| {
        let mut sl = search::hybrid::slices(&run, p, share, boots, boot_seed);
        for x in &mut sl {
            search::hybrid::judge(&mut x.summary, standard);
            if x.pass.is_some() {
                x.pass = Some(x.summary.loss_pass);
            }
        }
        sl
    });
    if let Some(sl) = &sl {
        print_slices(sl, share_text, &std_txt);
    }
    // Score (strata) and race groups: reported, judged only where enough decisions exist, and
    // never part of the verdict.
    let groups_extra = race.then(|| {
        let mut st = search::hybrid::strata(&run, &ds, boots, boot_seed);
        let mut rg = search::hybrid::race_groups(&run, race_share, boots, boot_seed);
        for (_, g) in st.iter_mut().chain(rg.iter_mut()) {
            search::hybrid::judge(g, standard);
        }
        (st, rg)
    });
    if let Some((st, rg)) = &groups_extra {
        print_score_groups(st, rg, race_share_text, min_group, &std_txt);
    }
    let time = check_time(&run, job.threads, wall);
    let leaves_json = job
        .exact_leaves
        .as_ref()
        .zip(run.exact_leaves.as_ref())
        .map(|(p, st)| print_exact_leaves(p, st, &time));
    if silver {
        // The reference's solves are the silver hybrid's, not exact ones.
        print_time_as(&time, "reference (silver)", "by the reference");
    } else {
        print_time(&time);
    }
    let (bands_pass, line) = verdict(&groups, &std_txt);
    let (pass, line) = match &sl {
        None => (bands_pass, line),
        Some(sl) => {
            // The slices the audit names (the decisions in no group are reported, not judged:
            // the overall row covers them).
            let failing: Vec<&str> = sl
                .iter()
                .filter(|x| x.group.is_some() && x.pass == Some(false))
                .map(|x| x.name.as_str())
                .collect();
            let pass = bands_pass && failing.is_empty();
            let line = format!(
                "{}; every slice's decision loss 95% upper bound <= {std_txt}: {}{}; with the slices: {}",
                line.trim_end_matches("; PASS").trim_end_matches("; FAIL"),
                if failing.is_empty() { "yes" } else { "no" },
                if failing.is_empty() {
                    String::new()
                } else if failing.len() <= 5 {
                    format!(" (failing: {})", failing.join(", "))
                } else {
                    format!(
                        " (failing: {} and {} more)",
                        failing[..5].join(", "),
                        failing.len() - 5
                    )
                },
                if pass { "PASS" } else { "FAIL" }
            );
            (pass, line)
        }
    };
    println!(
        "{line}{}",
        if silver {
            " (against the silver reference)"
        } else {
            ""
        }
    );
    if let Some(out) = a.get("--json") {
        // The hybrid's values: the network's, with the exact leaves' policy when there is one.
        let hybrid_values = format!(
            "{}+O{cut}:{}{}",
            search::measure::VALUES_VERSION,
            ensemble.id(),
            job.exact_leaves
                .as_ref()
                .map_or(String::new(), |p| format!("+exact:{}", p.describe()))
        );
        let group_rows = |g: &[(String, search::hybrid::Summary)]| {
            g.iter()
                .map(|(n, s)| {
                    serde_json::json!({
                        "group": n, "summary": s,
                        "judged": s.decisions >= min_group,
                    })
                })
                .collect::<Vec<_>>()
        };
        let v = serde_json::json!({
            "trick": trick,
            "cut": cut,
            "seed": seed,
            "label": job.label,
            "worlds": job.to,
            "world_source": src.name(),
            "reference": match &reference {
                None if truth_net => serde_json::json!({
                    "kind": "net",
                    "what": "the optimizer's-curse test (search::curse): the reference is H_cut \
                             with the network's own noise-free values, the player the same search \
                             with seeded zero-mean noise at each leaf; the `exact_*` fields of a \
                             decision hold the noise-free network's values",
                    "cut": cut,
                    "net": ensemble.id(),
                }),
                None => serde_json::json!({ "kind": "exact" }),
                Some((o, p, fp, _)) => serde_json::json!({
                    "kind": "silver",
                    "exact_leaves": fp.exact_leaves,
                    "what": "a silver reference (docs/m3-hybrid/silver.md): the deeper hybrid \
                             H_cut, exact search to the start of trick `cut` and the network \
                             there; the `exact_*` fields of a decision hold its reference's \
                             values (`silver` true), or exact values where an exact record \
                             stood in (--allow-exact)",
                    "cut": fp.cut,
                    "net": net_record(p, o),
                    "values": fp.values(),
                    "decisions_silver": n_silver,
                    "decisions_exact": run.decisions.len() - n_silver,
                }),
            },
            "exact_values": exact_from,
            "verified_worlds": run.verified.0,
            "net": net_record(net_path, oracle),
            "values": match &reference {
                None if truth_net => serde_json::json!({
                    "reference": format!("{}+O{cut}:{}", search::measure::VALUES_VERSION, ensemble.id()),
                    "hybrid": format!(
                        "{}+O{cut}:{}+noise:{}:{}",
                        search::measure::VALUES_VERSION,
                        ensemble.id(),
                        job.noise.as_ref().map_or(0, |n| n.seed),
                        job.noise.as_ref().map_or(String::new(), |n| n
                            .sd
                            .iter()
                            .map(u64::to_string)
                            .collect::<Vec<_>>()
                            .join("/"))
                    ),
                }),
                None => serde_json::json!({
                    "exact": search::measure::VALUES_VERSION,
                    "hybrid": hybrid_values,
                }),
                Some((_, _, fp, _)) => serde_json::json!({
                    "reference": fp.values(),
                    "hybrid": hybrid_values,
                }),
            },
            "equity_sha256": table_sha,
            "binary_sha256": binary_sha256(),
            "standard_pct": std_pct,
            "standard": standard,
            "boot": {"replicates": boots, "seed": boot_seed, "bound": "97.5th percentile of bootstrap means (upper end of the 95% interval)"},
            "groups": groups.iter().map(|(n, s)| serde_json::json!({"group": n, "summary": s})).collect::<Vec<_>>(),
            "pass": pass,
            "pass_without_slices": bands_pass,
            "score_groups": groups_extra.as_ref().map(|(st, _)| serde_json::json!({
                "what": "the decisions by the stratum of their hand's start state (S1-S4, search::measure::stratum_of); reported, not part of `pass`",
                "min_group": min_group,
                "groups": group_rows(st),
            })),
            "race_groups": groups_extra.as_ref().map(|(_, rg)| serde_json::json!({
                "what": "a decision is in race class c (nn::outcome::race_class_of, the classes the label shards count) when at least `share` of the distinct frontier positions the hybrid's searches over its worlds reach are of class c, else in `mixed`; reported, not part of `pass`",
                "share": race_share_text,
                "share_millionths": race_share,
                "min_group": min_group,
                "groups": group_rows(rg),
            })),
            "slices": slice_preds.as_ref().zip(sl.as_ref()).map(|((p, f), sl)| serde_json::json!({
                "what": "a decision is in a group when at least `share` of the distinct frontier \
                         positions the hybrid's searches over its worlds reach are in it \
                         (search::audit::GroupRecorder, nn::predicate); `no group`: the others",
                "predicates_file": f,
                "predicates_sha256": p.sha256,
                "share": share_text,
                "share_millionths": share,
                "groups": sl,
                "pass": sl.iter().all(|x| x.group.is_none() || x.pass != Some(false)),
                "failing": sl.iter().filter(|x| x.group.is_some() && x.pass == Some(false)).map(|x| x.name.clone()).collect::<Vec<_>>(),
                "empty": sl.iter().filter(|x| x.group.is_some() && x.pass.is_none()).map(|x| x.name.clone()).collect::<Vec<_>>(),
                "judged": "the groups' slices (every one with decisions must pass); `no group` is reported only (the overall row covers it)",
            })),
            "time": time_json(&time, t1.duration_since(t0).as_secs_f64()),
            "decisions": run.decisions,
        });
        let mut v = v;
        // Only an ensemble or a curse test adds these (a single network's JSON is as before).
        if n_nets > 1 {
            v["ensemble"] = serde_json::json!({
                "what": "the leaf value is the integer mean of the members' money chances \
                         (nn::ensemble: u128 totals per team, largest-remainder rounding to \
                         exactly 2^32, ties to the lower team)",
                "id": ensemble.id(),
                "members": ensemble.members().iter().zip(&net_paths).map(|(o, p)| net_record(p, o)).collect::<Vec<_>>(),
            });
        }
        if let Some(nr) = &noise_report {
            v["noise"] = serde_json::json!(nr);
        }
        if let Some(l) = &leaves_json {
            v["exact_leaves"] = l.clone();
        }
        if !calibration.is_empty() {
            v["noise_calibration"] = serde_json::json!({
                "what": "the SDs asked for, rescaled round by round on a sample of decisions (every k-th, the first worlds) until the realised SD per race class matches the target (search::curse::NoiseSpec::calibrated); the main run uses the last round's rescaled SDs",
                "decisions": n_cal,
                "rounds": calibration,
            });
        }
        write_json(out, &v)?;
    }
    Ok(())
}

/// `search hybrid-paired --a A.json --b B.json`: the paired comparison of two `hybrid-check`
/// JSONs on the same decisions (`search::hybrid::paired`).
fn hybrid_paired(a: &Args) -> Result<(), String> {
    let load =
        |k: &str| -> Result<(String, serde_json::Value, Vec<search::hybrid::PairedRow>), String> {
            let f = a.req(k)?;
            let text = std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?;
            let v: serde_json::Value =
                serde_json::from_str(&text).map_err(|e| format!("{f}: {e}"))?;
            let rows: Vec<search::hybrid::PairedRow> =
                serde_json::from_value(v["decisions"].clone()).map_err(|e| {
                    format!("{f}: decisions: {e} (a hybrid-check JSON of this binary or later)")
                })?;
            Ok((f.to_string(), v, rows))
        };
    let (fa, va, ra) = load("--a")?;
    let (fb, vb, rb) = load("--b")?;
    for k in ["trick", "cut", "seed", "label", "worlds"] {
        if va[k] != vb[k] {
            return Err(format!(
                "{fa} and {fb} differ in `{k}` ({} against {}): not the same decisions and worlds",
                va[k], vb[k]
            ));
        }
    }
    let name = |k: &str, v: &serde_json::Value| {
        a.get(k).map_or_else(
            || {
                v["values"]["hybrid"]
                    .as_str()
                    .unwrap_or("?")
                    .chars()
                    .take(40)
                    .collect::<String>()
            },
            str::to_string,
        )
    };
    let (na, nb) = (name("--a-name", &va), name("--b-name", &vb));
    let boots = a.num("--boot", 1000)?;
    let boot_seed = a.num("--boot-seed", 1)?;
    let ms = search::hybrid::paired(&ra, &rb, boots, boot_seed)?;
    println!(
        "paired comparison on {} decisions (trick {}, cut {}, {} worlds): A = {na} ({fa}), B = {nb} ({fb}); {boots} bootstrap replicates (seed {boot_seed}), the same decisions for both",
        ra.len(),
        va["trick"],
        va["cut"],
        va["worlds"]
    );
    let p = |x: f64| format!("{:+.4}%", 100.0 * x);
    println!(
        "{:<20} {:>5} {:>10} {:>10} {:>10} {:>23} {:>8} {:>8}",
        "measure", "dec", "A", "B", "B - A", "95% interval of B - A", "P(<0)", "P(>0)"
    );
    for m in &ms {
        println!(
            "{:<20} {:>5} {:>10} {:>10} {:>10} {:>23} {:>7.1}% {:>7.1}%",
            m.measure,
            m.decisions,
            p(m.a),
            p(m.b),
            p(m.diff),
            format!("{} .. {}", p(m.diff_lo95), p(m.diff_hi95)),
            100.0 * m.share_below,
            100.0 * m.share_above
        );
    }
    println!(
        "(loss: mean decision loss; own bias: the decider's own team's mean value error; B - A \
         below 0 means B loses less, or undervalues its own team less when A's bias is negative)"
    );
    // The hybrid's thread time per world of each (reports only), and the exact leaves' share.
    let hms = |v: &serde_json::Value| v["time"]["hybrid_ms_per_world"].as_f64();
    let time = match (hms(&va), hms(&vb)) {
        (Some(x), Some(y)) => {
            let leaves = |v: &serde_json::Value| {
                v["exact_leaves"]["policy"].as_str().map_or(
                    "no exact leaves".to_string(),
                    |p| {
                        format!(
                            "exact leaves {p}: {:.2}% of the leaves, {:.3} ms a world in leaf solves",
                            100.0 * v["exact_leaves"]["share_exact"].as_f64().unwrap_or(0.0),
                            v["exact_leaves"]["leaf_ms_per_world"].as_f64().unwrap_or(0.0)
                        )
                    },
                )
            };
            println!(
                "time: A {x:.3} thread-ms a world ({}), B {y:.3} ({}); B / A {:.2}",
                leaves(&va),
                leaves(&vb),
                y / x.max(1e-9)
            );
            serde_json::json!({ "a_hybrid_ms_per_world": x, "b_hybrid_ms_per_world": y,
                                "a_exact_leaves": va["exact_leaves"]["policy"],
                                "b_exact_leaves": vb["exact_leaves"]["policy"] })
        }
        _ => serde_json::Value::Null,
    };
    if let Some(out) = a.get("--json") {
        write_json(
            out,
            &serde_json::json!({
                "a": { "file": fa, "name": na, "values": va["values"], "net": va["net"]["sha256"], "ensemble": va["ensemble"]["id"] },
                "b": { "file": fb, "name": nb, "values": vb["values"], "net": vb["net"]["sha256"], "ensemble": vb["ensemble"]["id"] },
                "trick": va["trick"], "cut": va["cut"], "seed": va["seed"], "worlds": va["worlds"],
                "decisions": ra.len(),
                "boot": { "replicates": boots, "seed": boot_seed,
                          "what": "decisions resampled with replacement within each group, the same for both players; the interval is the 2.5th to 97.5th percentile of B - A" },
                "measures": ms,
                "time": time,
            }),
        )?;
    }
    Ok(())
}

/// The exact leaves' report (`search::exact_leaves`): per race class, the distinct frontier
/// positions the hybrid's searches reached (summed over worlds) and the share solved exactly;
/// the leaf solves' nodes and thread time per world, and the hybrid's thread time per world with
/// them and without them (the rest: the search and the network). Returns the JSON record.
fn print_exact_leaves(
    p: &search::exact_leaves::LeafPolicy,
    st: &search::exact_leaves::LeafStats,
    t: &CheckTime,
) -> serde_json::Value {
    let k = search::exact_leaves::CLASSES;
    let share = |e: u64, d: u64| e as f64 / d.max(1) as f64;
    let (d_all, e_all): (u64, u64) = (st.distinct.iter().sum(), st.exact.iter().sum());
    let hybrid_ms = t.hybrid_us as f64 / 1000.0 / t.worlds;
    let leaf_ms = st.us as f64 / 1000.0 / t.worlds;
    println!(
        "exact leaves {} (leaf tables {} MB): race class, distinct leaves a world, solved exactly a world, share",
        p.describe(),
        p.tt_mb
    );
    let mut classes = Vec::new();
    for c in 0..k {
        println!(
            "  class {c} {:<60} {:>10.2} {:>10.2} {:>7.2}%",
            search::hybrid_explain::RACE_CLASSES[c],
            st.distinct[c] as f64 / t.worlds,
            st.exact[c] as f64 / t.worlds,
            100.0 * share(st.exact[c], st.distinct[c])
        );
        classes.push(serde_json::json!({
            "class": c, "name": search::hybrid_explain::RACE_CLASSES[c],
            "distinct": st.distinct[c], "exact": st.exact[c],
            "share_exact": share(st.exact[c], st.distinct[c]),
        }));
    }
    println!(
        "  {:<68} {:>10.2} {:>10.2} {:>7.2}%",
        "all",
        d_all as f64 / t.worlds,
        e_all as f64 / t.worlds,
        100.0 * share(e_all, d_all)
    );
    println!(
        "  leaf solves: {:.0} nodes and {:.3} ms a world ({:.3} ms a solve); hybrid {:.3} thread-ms a world with them, {:.3} without them (search and network)",
        st.nodes as f64 / t.worlds,
        leaf_ms,
        st.us as f64 / 1000.0 / e_all.max(1) as f64,
        hybrid_ms,
        hybrid_ms - leaf_ms
    );
    if let Some(l) = &p.learned {
        println!(
            "  learned: model {} ({} trees, network {}), threshold {}{}; {:.1} leaves scored a world, {:.3} ms a world scoring",
            &l.model.sha256[..12],
            l.model.n_trees(),
            &l.model.net_sha256[..l.model.net_sha256.len().min(12)],
            l.t.map_or("inf".to_string(), |t| t.to_string()),
            l.budget
                .as_ref()
                .map_or(String::new(), |b| format!(" (budget {b})")),
            st.scored as f64 / t.worlds,
            st.score_us as f64 / 1000.0 / t.worlds
        );
    }
    let learned = p.learned.as_ref().map(|l| {
        serde_json::json!({
            "model_sha256": l.model.sha256, "net_sha256": l.model.net_sha256,
            "trees": l.model.n_trees(), "t": l.t, "budget": l.budget,
            "scored": st.scored, "score_ms_per_world": st.score_us as f64 / 1000.0 / t.worlds,
        })
    });
    serde_json::json!({
        "learned": learned,
        "what": "the frontier positions the policy picks are solved exactly (search::exact_leaves, the labels' exact solver and tie rule) instead of being valued by the network; counts are distinct canonical leaves per world's search, summed over worlds; times are thread time (reports only)",
        "policy": p.describe(),
        "leaf_tt_mb": p.tt_mb,
        "classes": classes,
        "distinct": d_all,
        "exact": e_all,
        "share_exact": share(e_all, d_all),
        "leaf_nodes": st.nodes,
        "leaf_nodes_per_world": st.nodes as f64 / t.worlds,
        "leaf_ms_per_world": leaf_ms,
        "leaf_ms_per_solve": st.us as f64 / 1000.0 / e_all.max(1) as f64,
        "hybrid_ms_per_world": hybrid_ms,
        "hybrid_ms_per_world_without_leaf_solves": hybrid_ms - leaf_ms,
    })
}

/// The curse test's noise per race class: asked for and realised (`search::curse`).
fn print_noise(nr: &search::curse::NoiseReport) {
    println!(
        "noise (seed {}): race class, distinct leaves, SD target, SD asked, SD realised, mean, mean |change|, share shrunk",
        nr.seed
    );
    for c in &nr.classes {
        println!(
            "  class {}: {:>11} {:>8.4}% {:>8.4}% {:>8.4}% {:>+9.5}% {:>8.4}% {:>6.2}%",
            c.class,
            c.leaves,
            c.target_sd_pct,
            c.asked_sd_pct,
            c.realised_sd_pct,
            c.realised_mean_pct,
            c.realised_mean_abs_pct,
            100.0 * c.shrunk_share
        );
    }
    println!(
        "  all:     {:>11}                     {:>8.4}% (every team of every distinct leaf; transfers \
         with a team of little money are shrunk, so the realised SD can fall short of the asked)",
        nr.leaves, nr.realised_sd_pct
    );
}

/// The score (stratum) and race-group tables of `hybrid-check --race-groups`
/// (`search::hybrid::strata`, `search::hybrid::race_groups`): reported, judged only where at
/// least `min` decisions fall in a group.
fn print_score_groups(
    st: &[(String, search::hybrid::Summary)],
    rg: &[(String, search::hybrid::Summary)],
    share: &str,
    min: usize,
    std: &str,
) {
    let spct = |x: f64| format!("{:+.4}%", 100.0 * x);
    for (title, rows) in [
        ("score groups (the stratum of the hand's start state)", st),
        (
            &*format!(
                "race groups (a decision is in a class when at least {share} of the distinct \
                 frontier positions its hybrid searches reach are of it)"
            ),
            rg,
        ),
    ] {
        println!("{title}; judged only with at least {min} decisions; not part of the verdict:");
        println!(
            "{:<62} {:>5} {:>10} {:>10} {:>9} {:>7} {:>38} {:>9}",
            "group",
            "dec",
            "mean loss",
            "95% upper",
            format!("<={std}"),
            "agree",
            "bias: own / next / opposite / previous",
            "SD"
        );
        for (name, s) in rows {
            let judged = if s.decisions == 0 {
                "-".to_string()
            } else if s.decisions < min {
                "too few".to_string()
            } else if s.loss_pass {
                "yes".to_string()
            } else {
                "no".to_string()
            };
            if s.decisions == 0 {
                println!("{name:<62} {:>5}", 0);
                continue;
            }
            println!(
                "{:<62} {:>5} {:>10} {:>10} {:>9} {:>6.1}% {:>38} {:>9}",
                name,
                s.decisions,
                pct(s.mean_loss),
                pct(s.loss_upper95),
                judged,
                100.0 * s.agreement,
                s.bias.map(spct).join(" "),
                pct(s.sd)
            );
        }
    }
}

/// The slices' table of `hybrid-check --slices` (`search::hybrid::slices`).
fn print_slices(sl: &[search::hybrid::Slice], share: &str, std: &str) {
    let spct = |x: f64| format!("{:+.4}%", 100.0 * x);
    println!(
        "slices (a decision is in a group when at least {share} of the distinct frontier positions its hybrid searches reach are in it):"
    );
    println!(
        "{:<28} {:>5} {:>9} {:>10} {:>10} {:>7} {:>7} {:>10} {:>9}  conditions",
        "slice",
        "dec",
        "reached",
        "mean loss",
        "95% upper",
        format!("<={std}"),
        "agree",
        "bias own",
        "SD"
    );
    for x in sl {
        let s = &x.summary;
        let name: String = x.name.chars().take(28).collect();
        if s.decisions == 0 {
            println!(
                "{:<28} {:>5} {:>9} {:>10} {:>10} {:>7} {:>7} {:>10} {:>9}  {}",
                name,
                0,
                x.reached_share
                    .map_or("-".into(), |r| format!("{:.2}%", 100.0 * r)),
                "-",
                "-",
                "n/a",
                "-",
                "-",
                "-",
                x.conditions
            );
            continue;
        }
        println!(
            "{:<28} {:>5} {:>9} {:>10} {:>10} {:>7} {:>6.1}% {:>10} {:>9}  {}",
            name,
            s.decisions,
            x.reached_share
                .map_or("-".into(), |r| format!("{:.2}%", 100.0 * r)),
            pct(s.mean_loss),
            pct(s.loss_upper95),
            if s.loss_pass { "yes" } else { "no" },
            100.0 * s.agreement,
            spct(s.bias[0]),
            pct(s.sd),
            x.conditions
        );
    }
    println!(
        "(reached: the group's share of every frontier position the hybrid reached on these decisions; n/a: no decision of this set is in it, so a group needs a mini golden set; `no group` is reported, not judged: the overall row covers it)"
    );
}

fn write_json(out: &str, v: &serde_json::Value) -> Result<(), String> {
    std::fs::write(
        out,
        serde_json::to_string_pretty(v).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| format!("{out}: {e}"))
}

fn net_record(net_path: &str, oracle: &nn::Oracle) -> serde_json::Value {
    serde_json::json!({
        "path": net_path,
        "sha256": oracle.sha256,
        "widths": oracle.net().widths(),
        "trick": oracle.trick,
        "meta": oracle.meta,
    })
}

/// The §3 table of `hybrid-check` (`search::hybrid::report`'s groups).
fn print_groups(groups: &[(String, search::hybrid::Summary)], std: &str) {
    let spct = |x: f64| format!("{:+.4}%", 100.0 * x);
    println!(
        "{:<9} {:>5} {:>10} {:>10} {:>7} {:>7} {:>8} {:>38} {:>9} {:>10} {:>9}",
        "group",
        "dec",
        "mean loss",
        "95% upper",
        format!("<={std}"),
        "agree",
        "values",
        "bias: own / next / opposite / previous",
        "SD",
        "p99.9|e|",
        "max|e|"
    );
    for (name, s) in groups.iter().filter(|(_, s)| s.decisions > 0) {
        println!(
            "{:<9} {:>5} {:>10} {:>10} {:>7} {:>6.1}% {:>8} {:>38} {:>9} {:>10} {:>9}",
            name,
            s.decisions,
            pct(s.mean_loss),
            pct(s.loss_upper95),
            if s.loss_pass { "yes" } else { "no" },
            100.0 * s.agreement,
            s.values,
            s.bias.map(spct).join(" "),
            pct(s.sd),
            pct(s.p999_abs),
            pct(s.max_abs)
        );
    }
    if let Some((_, s)) = groups.first() {
        println!(
            "(value error over all four teams' chances; bias per team relative to the decider; the decider's team alone: SD {}, p99.9|e| {})",
            pct(s.decider_sd),
            pct(s.decider_p999_abs)
        );
    }
}

/// The standard of §3 on `groups`: whether it passes, and the line that says so.
fn verdict(groups: &[(String, search::hybrid::Summary)], std: &str) -> (bool, String) {
    let loss_pass = groups.iter().take(4).all(|(_, s)| s.loss_pass);
    let bias_pass = groups.first().is_some_and(|(_, s)| s.bias_pass);
    let pass = loss_pass && bias_pass;
    let worst_bias = groups.first().map_or(0.0, |(_, s)| {
        s.bias.iter().fold(0.0f64, |m, b| m.max(b.abs()))
    });
    let line = format!(
        "standard: decision loss 95% upper bound <= {std} overall and in every band: {}; every team's |bias| <= {std}: {} (largest {}); {}",
        if loss_pass { "yes" } else { "no" },
        if bias_pass { "yes" } else { "no" },
        pct(worst_bias),
        if pass { "PASS" } else { "FAIL" }
    );
    (pass, line)
}

/// Times and counts of a check (reports only).
struct CheckTime {
    worlds: f64,
    exact_worlds: f64,
    exact_us: u64,
    hybrid_us: u64,
    exact_nodes: u64,
    hybrid_nodes: u64,
    evals: u64,
    ns_eval: f64,
    wall: f64,
    threads: usize,
}

fn check_time(run: &search::hybrid::CheckRun, threads: usize, wall: f64) -> CheckTime {
    let d = &run.decisions;
    CheckTime {
        worlds: d.iter().map(|d| d.worlds).sum::<u64>().max(1) as f64,
        exact_worlds: d.iter().map(|d| d.exact_worlds).sum::<u64>() as f64,
        exact_us: d.iter().map(|d| d.exact_us).sum(),
        hybrid_us: d.iter().map(|d| d.hybrid_us).sum(),
        exact_nodes: d.iter().map(|d| d.exact_nodes).sum(),
        hybrid_nodes: d.iter().map(|d| d.hybrid_nodes).sum(),
        evals: d.iter().map(|d| d.frontier).sum(),
        ns_eval: run.net.0 as f64 / run.net.1.max(1) as f64,
        wall,
        threads,
    }
}

fn print_time(t: &CheckTime) {
    print_time_as(t, "exact", "exactly");
}

/// [`print_time`] naming the reference's solves (`what`, `how`: "exact", "exactly").
fn print_time_as(t: &CheckTime, what: &str, how: &str) {
    println!(
        "time: {what} {:.3} ms a world ({:.0} nodes; {} worlds solved {how} here), hybrid {:.3} ms a world ({:.0} nodes, {:.0} frontier evaluations); {:.0} ns per frontier evaluation (encode + integer forward + softmax, or the oracle's memo of recent positions); {:.1} s wall on {} threads",
        t.exact_us as f64 / 1000.0 / t.exact_worlds.max(1.0),
        t.exact_nodes as f64 / t.exact_worlds.max(1.0),
        t.exact_worlds,
        t.hybrid_us as f64 / 1000.0 / t.worlds,
        t.hybrid_nodes as f64 / t.worlds,
        t.evals as f64 / t.worlds,
        t.ns_eval,
        t.wall,
        t.threads
    );
}

fn time_json(t: &CheckTime, setup_s: f64) -> serde_json::Value {
    serde_json::json!({
        "exact_worlds_solved": t.exact_worlds,
        "exact_ms_per_world": t.exact_us as f64 / 1000.0 / t.exact_worlds.max(1.0),
        "hybrid_ms_per_world": t.hybrid_us as f64 / 1000.0 / t.worlds,
        "exact_nodes_per_world": t.exact_nodes as f64 / t.exact_worlds.max(1.0),
        "hybrid_nodes_per_world": t.hybrid_nodes as f64 / t.worlds,
        "net_evals_per_world": t.evals as f64 / t.worlds,
        "ns_per_net_eval": t.ns_eval,
        "wall_s": t.wall,
        "threads": t.threads,
        "setup_s": setup_s,
    })
}

/// What `hybrid_check_saved` needs from the common set-up.
struct SavedRun<'a> {
    trick: u8,
    ds: &'a [DecisionSpec],
    selected: &'a dyn Fn(u64) -> bool,
    job: search::hybrid::CheckJob,
    ctx: &'a Ctx<'a>,
    oracle: &'a nn::Oracle,
    net_path: &'a str,
    table_sha: &'a str,
    boots: u64,
    boot_seed: u64,
    t0: Instant,
    /// `--slices`: the predicate file, its path, the share as written and in millionths.
    slices: Option<(&'a nn::predicate::Predicates, &'a str, &'a str, u64)>,
}

/// `search hybrid-check --saved LABEL=FILE …`: H_K against the exact values earlier runs saved
/// (`search::saved`; docs/M3-HYBRID-SPEC.md §3, "Trick-1 checks"), one set per burn rule, each
/// checked with the matching `rules_robber_burn`; per world, per decision, the §3 tables per
/// set, and the `--hard` worlds on their own lines with the saved exact times. With `--slices`
/// (the audit loop at trick 2, docs/m3-hybrid/audit-loop.md), each set also gets the slices of
/// the golden-set check (`search::hybrid::slices`: a decision is in a group when at least the
/// share of the frontier positions its hybrid searches reach are in it), judged the same way.
fn hybrid_check_saved(a: &Args, r: SavedRun) -> Result<(), String> {
    use search::hybrid::{CheckProgress, CheckRun, check_worlds_with, report};
    use search::saved::{BurnRule, join_saved, parse_pairs, plan, read_saved, saved_sets};
    for k in ["--worlds", "--recompute", "--to"] {
        if a.has(k) {
            return Err(format!(
                "{k} does not go with --saved (the saved files give the worlds; --verify-pct 100 \
                 recomputes them all)"
            ));
        }
    }
    let SavedRun { trick, ds, job, .. } = r;
    let cut = job.cut;
    let mut all = Vec::new();
    let mut files = Vec::new();
    for spec in a.all("--saved") {
        let (label, path) = spec.split_once('=').unwrap_or((spec, spec));
        let (w, info) = read_saved(Path::new(path), label)?;
        eprintln!(
            "saved {label}: {} lines ({} checkpoint lines, {} worlds records, {} torn), {} exact worlds, by values version {:?}{}",
            info.lines,
            info.ckpt_lines,
            info.records,
            info.torn,
            info.worlds,
            info.by_version,
            if info.other_method > 0 {
                format!(", {} worlds of other methods skipped", info.other_method)
            } else {
                String::new()
            }
        );
        files.push(info);
        all.extend(w);
    }
    let only = a.get("--only").map(parse_pairs).transpose()?;
    let hard = a
        .get("--hard")
        .map(parse_pairs)
        .transpose()?
        .unwrap_or_default();
    let sets = saved_sets(join_saved(all)?)?;
    let keep =
        |id: u64, w: u64| (r.selected)(id) && only.as_ref().is_none_or(|o| o.contains(&(id, w)));
    let verify_rule = a.get("--verify-rule").unwrap_or("all");
    if !["all", "rules-v1", "searched"].contains(&verify_rule) {
        return Err(format!(
            "--verify-rule {verify_rule}: all, rules-v1 or searched"
        ));
    }
    let both: Vec<(u64, u64, bool)> = search::saved::both_rules(&sets)
        .into_iter()
        .filter(|&(d, w, _)| keep(d, w))
        .collect();
    let src_name = r.ctx.source.name().to_string();
    // Each set's plan and settings (its burn rule; the verification where asked).
    let mut work = Vec::new();
    for (si, set) in sets.iter().enumerate() {
        let (pl, notes) = plan(ds, set, job.seed, &src_name, r.table_sha, &keep)?;
        if pl.is_empty() {
            continue;
        }
        let verify_this = match verify_rule {
            "rules-v1" => set.rule == BurnRule::RulesV1,
            "searched" => set.rule == BurnRule::Searched,
            _ => true,
        };
        let sjob = search::hybrid::CheckJob {
            rules_robber_burn: set.rule.rules_robber_burn(),
            verify_pct: if verify_this { job.verify_pct } else { 0.0 },
            ..job.clone()
        };
        work.push((si, set, pl, notes, sjob));
    }
    // The sets run at once, the threads shared out by worlds (never more threads than worlds,
    // never more in all than --threads, so no world's time is inflated by another's).
    let counts: Vec<usize> = work
        .iter()
        .map(|w| w.2.iter().map(|d| d.worlds.len()).sum())
        .collect();
    let split = split_threads(job.threads, &counts);
    for (w, &t) in work.iter_mut().zip(&split) {
        w.4.threads = t;
    }
    println!(
        "H_{cut} vs saved exact worlds, trick {trick}: net {} {:?}; {} source; {} threads (one world each; {}), tables {} MB; saved files: {}",
        &r.oracle.sha256[..12],
        r.oracle.net().widths(),
        src_name,
        job.threads,
        work.iter()
            .map(|w| format!("set {}: {} threads", w.0 + 1, w.4.threads))
            .collect::<Vec<_>>()
            .join(", "),
        job.tt_mb,
        files
            .iter()
            .map(|f| format!("{} ({} exact worlds)", f.from, f.worlds))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if !both.is_empty() {
        let differ: Vec<String> = both
            .iter()
            .filter(|b| !b.2)
            .map(|(d, w, _)| format!("{d}:{w}"))
            .collect();
        println!(
            "worlds saved under both burn rules: {}; values identical in {}{}",
            both.len(),
            both.len() - differ.len(),
            if differ.is_empty() {
                String::new()
            } else {
                format!(", different in {} ({})", differ.join(","), differ.len())
            }
        );
    }
    if a.has("--plan-only") {
        // What the check would do, with nothing solved: to price a run before renting.
        for (si, set, pl, notes, sjob) in &work {
            let mut verify = Vec::new();
            for d in pl {
                for t in &d.worlds {
                    if sjob.verify_sampled(d.d.id, t.w) {
                        verify.push(format!("{}:{} (saved cost {})", d.d.id, t.w, t.cost));
                    }
                }
            }
            let cost: u64 = pl.iter().flat_map(|d| &d.worlds).map(|t| t.cost).sum();
            println!(
                "plan: set {} ({}): {} decisions, {} worlds, saved cost {cost} (nodes where recorded, else microseconds), {} threads; verification ({}%): {} worlds{}{}",
                si + 1,
                set.rule.name(),
                notes.decisions,
                notes.worlds,
                sjob.threads,
                sjob.verify_pct,
                verify.len(),
                if verify.is_empty() { "" } else { ": " },
                verify.join(", ")
            );
        }
        return Ok(());
    }
    let progress_path = a.get("--progress").map(PathBuf::from);
    let progs: Vec<CheckProgress> = work.iter().map(|_| CheckProgress::default()).collect();
    let finished = AtomicBool::new(false);
    let t1 = Instant::now();
    let runs: Vec<(Result<CheckRun, String>, f64)> = std::thread::scope(|s| {
        let (progs, finished) = (&progs, &finished);
        let sum = move |f: fn(&CheckProgress) -> &std::sync::atomic::AtomicU64| -> u64 {
            progs.iter().map(|p| f(p).load(Ordering::Relaxed)).sum()
        };
        if let Some(p) = &progress_path {
            let threads = job.threads as u64;
            let ex_workers: u64 = work
                .iter()
                .map(|w| (w.4.threads / w.4.exact_solve_threads.max(1)).max(1) as u64)
                .sum();
            s.spawn(move || {
                while !finished.load(Ordering::Relaxed) {
                    let (ed, et) = (sum(|p| &p.exact_done), sum(|p| &p.exact_total));
                    let (hd, ht) = (sum(|p| &p.hybrid_done), sum(|p| &p.hybrid_total));
                    write_atomic(
                        p,
                        &serde_json::json!({
                            "exact_done": ed, "exact_total": et,
                            "worlds_done": hd, "worlds_total": ht,
                            // Fewer solves left than threads: some threads are idle by design.
                            "tail": if ed < et { et - ed < ex_workers } else { ht - hd < threads },
                            "elapsed_s": t1.elapsed().as_secs_f64(),
                        }),
                    );
                    for _ in 0..20 {
                        if finished.load(Ordering::Relaxed) {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    }
                }
            });
        }
        // Each world on stderr as it finishes, so a stopped run still shows what finished.
        let set_of: Vec<usize> = work.iter().map(|w| w.0 + 1).collect();
        s.spawn(move || {
            let mut last = false;
            while !last {
                last = finished.load(Ordering::Relaxed);
                for (p, set) in progs.iter().zip(&set_of) {
                    let done: Vec<[u64; 5]> =
                        std::mem::take(&mut *p.finished.lock().unwrap_or_else(|e| e.into_inner()));
                    for [d, w, us, nodes, evals] in done {
                        eprintln!(
                            "set {set} d{d}:w{w}: H_{cut} {:.2} s, {:.3e} nodes, {evals} evaluations ({}/{} worlds, {:.0} s)",
                            us as f64 / 1e6,
                            nodes as f64,
                            sum(|p| &p.hybrid_done),
                            sum(|p| &p.hybrid_total),
                            t1.elapsed().as_secs_f64()
                        );
                    }
                }
                if !last {
                    std::thread::sleep(Duration::from_millis(200));
                }
            }
        });
        let preds = r.slices.map(|x| x.0);
        let handles: Vec<_> = work
            .iter()
            .zip(progs)
            .map(|(w, p)| {
                s.spawn(move || {
                    let t = Instant::now();
                    let run = check_worlds_with(&w.2, &w.4, r.ctx, r.oracle, Some(p), preds);
                    (run, t.elapsed().as_secs_f64())
                })
            })
            .collect();
        let out = handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect();
        finished.store(true, Ordering::Relaxed);
        out
    });
    let mut set_docs = Vec::new();
    let mut hard_lines: Vec<serde_json::Value> = Vec::new();
    for ((si, set, _pl, notes, sjob), (run, wall)) in work.iter().zip(runs) {
        let run = run?;
        let n_worlds: usize = run.decisions.iter().map(|d| d.worlds as usize).sum();
        let found: usize = set
            .worlds
            .iter()
            .filter(|w| keep(w.id, w.w))
            .map(|w| w.found.len())
            .sum();
        println!();
        println!(
            "set {}: {} (values version {}, rules_robber_burn = {}): {} decisions, {} worlds (found {found} times in the files); {} threads",
            si + 1,
            set.rule.name(),
            set.values,
            sjob.rules_robber_burn,
            notes.decisions,
            if run.mismatched.is_empty() {
                n_worlds
            } else {
                notes.worlds
            },
            sjob.threads
        );
        if notes.table_unrecorded > 0 || notes.seed_unrecorded > 0 || notes.fp_unrecorded > 0 {
            println!(
                "  not recorded in the files (so not checked): the equity table of {} worlds, the seed of {} (seed {} assumed), the decision fingerprint of {}",
                notes.table_unrecorded, notes.seed_unrecorded, job.seed, notes.fp_unrecorded
            );
        }
        let verification = if run.verified.0 == 0 {
            if set.rule == BurnRule::RulesV1 {
                format!(
                    "stored, NOT re-verified on this binary (--verify-pct {}): these values were made by an older binary (values version 1); that the current solver with rules_robber_burn gives the same bytes is not confirmed",
                    sjob.verify_pct
                )
            } else {
                format!(
                    "stored, NOT re-verified on this binary (--verify-pct {})",
                    sjob.verify_pct
                )
            }
        } else if run.verified.1 == 0 {
            format!(
                "stored; {} worlds recomputed by this binary (--verify-pct {}, {} threads a solve) and byte-identical",
                run.verified.0, sjob.verify_pct, sjob.exact_solve_threads
            )
        } else {
            format!(
                "stored; {} of {} recomputed worlds DIFFER from this binary's (decision:world {}): this set's stored values are not the current solver's{}, so H_{cut} was not compared with them",
                run.verified.1,
                run.verified.0,
                run.mismatched
                    .iter()
                    .map(|(d, w)| format!("{d}:{w}"))
                    .collect::<Vec<_>>()
                    .join(","),
                if sjob.rules_robber_burn {
                    " with rules_robber_burn"
                } else {
                    ""
                }
            )
        };
        println!("  exact values: {verification}");
        let mut doc = serde_json::json!({
            "set": si + 1,
            "burn_rule": set.rule.name(),
            "values_version": set.values,
            "rules_robber_burn": sjob.rules_robber_burn,
            "threads": sjob.threads,
            "plan": notes,
            "exact_values": verification,
            "verification": {
                "verify_pct": sjob.verify_pct,
                "solve_threads": sjob.exact_solve_threads,
                "tt_mb": sjob.exact_tt_mb,
                "recomputed": run.verified.0,
                "differ": run.verified.1,
                "mismatched": run.mismatched,
                "done": run.verified.0 > 0,
            },
        });
        if !run.mismatched.is_empty() {
            set_docs.push(doc);
            continue;
        }
        let saved_of = |id: u64, w: u64| {
            set.worlds
                .iter()
                .find(|s| s.id == id && s.w == w)
                .map(|s| s.found.clone())
                .unwrap_or_default()
        };
        println!(
            "  per world: the decider's team's money chance for each candidate, exact | H_{cut} (%), the largest |H - exact| over the four teams; H_{cut}'s time on one thread, nodes and network evaluations; the saved exact time and nodes (as recorded by each run, with its own threads per world)"
        );
        let mut worlds_json = Vec::new();
        for d in &run.decisions {
            for wc in &d.world_checks {
                let saved = saved_of(d.id, wc.w);
                let ch = |v: &[equity::MoneyChance]| {
                    v.iter()
                        .map(|m| {
                            format!("{:.3}", 100.0 * m.get(d.team) as f64 / equity::ONE as f64)
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                let worst = wc
                    .exact
                    .iter()
                    .zip(&wc.hybrid)
                    .flat_map(|(e, h)| (0..4u8).map(move |t| h.get(t).abs_diff(e.get(t))))
                    .max()
                    .unwrap_or(0);
                let saved_txt = saved
                    .iter()
                    .map(|s| {
                        format!(
                            "{} {}{}",
                            s.from,
                            s.us.map_or("-".to_string(), |u| format!("{:.1} s", u as f64 / 1e6)),
                            s.nodes
                                .map_or(String::new(), |n| format!(" {:.3e} nodes", n as f64))
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                println!(
                    "  d{} w{} pos {} S{}{}: exact {} | H {} | max|e| {}; H_{cut} {:.2} s, {:.3e} nodes (tricks 1-5: {:?}), {} evals{}; saved exact: {saved_txt}",
                    d.id,
                    wc.w,
                    d.position,
                    d.stratum,
                    wc.robber
                        .map_or(String::new(), |s| format!(" (seat {s} robs later)")),
                    ch(&wc.exact),
                    ch(&wc.hybrid),
                    pct(worst as f64 / equity::ONE as f64),
                    wc.hybrid_us as f64 / 1e6,
                    wc.hybrid_nodes as f64,
                    wc.hybrid_nodes_by_trick,
                    wc.frontier,
                    match wc.verified {
                        Some(true) => "; exact recomputed: identical",
                        Some(false) => "; exact recomputed: DIFFERS",
                        None => "",
                    }
                );
                let raw = |v: &[equity::MoneyChance]| v.iter().map(|m| m.0).collect::<Vec<_>>();
                let wj = serde_json::json!({
                    "id": d.id, "w": wc.w, "position": d.position, "stratum": d.stratum,
                    "team": d.team, "cands": d.cands, "robber": wc.robber,
                    "exact": raw(&wc.exact), "hybrid": raw(&wc.hybrid),
                    "max_abs_error": worst,
                    "exact_solved": wc.exact_solved, "verified": wc.verified,
                    "exact_us": wc.exact_us, "exact_nodes": wc.exact_nodes,
                    "hybrid_us": wc.hybrid_us, "hybrid_nodes": wc.hybrid_nodes,
                    "hybrid_nodes_by_trick": wc.hybrid_nodes_by_trick,
                    "net_evals": wc.frontier, "hybrid_tt_bytes": wc.hybrid_tt_bytes,
                    "saved": saved,
                });
                if hard.contains(&(d.id, wc.w)) {
                    hard_lines.push(serde_json::json!({ "set": si + 1, "burn_rule": set.rule.name(), "world": wj.clone() }));
                }
                worlds_json.push(wj);
            }
        }
        println!(
            "  per decision (choices are candidate indices; loss in the decider's team's money chance):"
        );
        for d in &run.decisions {
            println!(
                "  d{} pos {} S{}: {} worlds, {} candidates; exact choice {}, H_{cut} choice {}, loss {}{}",
                d.id,
                d.position,
                d.stratum,
                d.worlds,
                d.cands,
                d.exact_choice,
                d.hybrid_choice,
                pct(d.loss as f64 / equity::ONE as f64),
                if d.agree { " (agree)" } else { "" }
            );
        }
        let mut groups = report(&run, r.boots, r.boot_seed);
        let std_pct = a.get("--standard").unwrap_or("0.01");
        let standard = search::hybrid::parse_standard(std_pct)?;
        for (_, g) in &mut groups {
            search::hybrid::judge(g, standard);
        }
        let std_txt = format!("{}%", std_pct.trim());
        print_groups(&groups, &std_txt);
        let sl = r.slices.map(|(p, _, text, share)| {
            let mut sl = search::hybrid::slices(&run, p, share, r.boots, r.boot_seed);
            // Judged by `--standard`, as the golden-set check's slices.
            for x in &mut sl {
                search::hybrid::judge(&mut x.summary, standard);
                if x.pass.is_some() {
                    x.pass = Some(x.summary.loss_pass);
                }
            }
            print_slices(&sl, text, &std_txt);
            sl
        });
        let time = check_time(&run, sjob.threads, wall);
        print_time(&time);
        let (bands_pass, line) = verdict(&groups, &std_txt);
        let failing: Vec<&str> = sl
            .iter()
            .flatten()
            .filter(|x| x.group.is_some() && x.pass == Some(false))
            .map(|x| x.name.as_str())
            .collect();
        let pass = bands_pass && failing.is_empty();
        match &sl {
            None => println!("{line}"),
            Some(_) => println!(
                "{}; every slice's decision loss 95% upper bound <= {std_txt}: {}{}; with the slices: {}",
                line.trim_end_matches("; PASS").trim_end_matches("; FAIL"),
                if failing.is_empty() { "yes" } else { "no" },
                if failing.is_empty() {
                    String::new()
                } else {
                    format!(" (failing: {})", failing.join(", "))
                },
                if pass { "PASS" } else { "FAIL" }
            ),
        }
        let o = doc.as_object_mut().expect("an object");
        o.insert(
            "values".into(),
            serde_json::json!({
                "exact": set.values,
                "hybrid": format!("{}+O{cut}:{}", search::measure::VALUES_VERSION, r.oracle.sha256),
            }),
        );
        o.insert(
            "groups".into(),
            groups
                .iter()
                .map(|(n, s)| serde_json::json!({"group": n, "summary": s}))
                .collect(),
        );
        o.insert("pass".into(), serde_json::json!(pass));
        o.insert("pass_without_slices".into(), serde_json::json!(bands_pass));
        if let (Some(sl), Some((p, f, text, share))) = (&sl, r.slices) {
            o.insert(
                "slices".into(),
                serde_json::json!({
                    "what": "a decision is in a group when at least `share` of the distinct frontier \
                             positions the hybrid's searches over its saved worlds reach are in it \
                             (search::audit::GroupRecorder, nn::predicate); `no group`: the others",
                    "predicates_file": f,
                    "predicates_sha256": p.sha256,
                    "share": text,
                    "share_millionths": share,
                    "groups": sl,
                    "pass": failing.is_empty(),
                    "failing": failing,
                    "empty": sl.iter().filter(|x| x.group.is_some() && x.pass.is_none()).map(|x| x.name.clone()).collect::<Vec<_>>(),
                    "judged": "the groups' slices (every one with decisions must pass); `no group` is reported only (the overall row covers it)",
                }),
            );
        }
        o.insert("time".into(), time_json(&time, 0.0));
        o.insert("decisions".into(), serde_json::json!(run.decisions));
        o.insert("worlds".into(), serde_json::json!(worlds_json));
        set_docs.push(doc);
    }
    println!();
    for h in &hard {
        let lines: Vec<&serde_json::Value> = hard_lines
            .iter()
            .filter(|l| l["world"]["id"] == h.0 && l["world"]["w"] == h.1)
            .collect();
        if lines.is_empty() {
            println!("hard world d{}:w{}: not among the checked worlds", h.0, h.1);
        }
        for l in lines {
            let w = &l["world"];
            let hs = w["hybrid_us"].as_u64().unwrap_or(0) as f64 / 1e6;
            let saved = w["saved"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|s| {
                    let us = s["us"].as_u64();
                    format!(
                        "{}: {}{}{}",
                        s["from"].as_str().unwrap_or("?"),
                        us.map_or("time not recorded".to_string(), |u| format!(
                            "{:.0} s",
                            u as f64 / 1e6
                        )),
                        s["nodes"]
                            .as_u64()
                            .map_or(String::new(), |n| format!(", {:.3e} nodes", n as f64)),
                        us.filter(|_| hs > 0.0).map_or(String::new(), |u| format!(
                            " ({:.1} times H_{cut}'s time)",
                            u as f64 / 1e6 / hs
                        ))
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            println!(
                "hard world d{}:w{} ({}): H_{cut} {hs:.1} s on one thread, {:.3e} nodes, {} network evaluations, largest |H - exact| {}; saved exact: {saved}",
                h.0,
                h.1,
                l["burn_rule"].as_str().unwrap_or("?"),
                w["hybrid_nodes"].as_u64().unwrap_or(0) as f64,
                w["net_evals"],
                pct(w["max_abs_error"].as_u64().unwrap_or(0) as f64 / equity::ONE as f64)
            );
        }
    }
    let peak = search::t1bench::proc_peak_rss_mb();
    let max_tt = set_docs
        .iter()
        .flat_map(|d| d["worlds"].as_array().into_iter().flatten())
        .filter_map(|w| w["hybrid_tt_bytes"].as_u64())
        .max()
        .unwrap_or(0);
    println!(
        "memory: peak resident {} MB for {} threads (largest hybrid table {:.0} MB of {} MB allowed)",
        peak.map_or("?".to_string(), |m| m.to_string()),
        job.threads,
        max_tt as f64 / f64::from(1 << 20),
        job.tt_mb
    );
    if let Some(out) = a.get("--json") {
        let v = serde_json::json!({
            "trick": trick,
            "cut": cut,
            "seed": job.seed,
            "world_source": src_name,
            "saved_files": files,
            "only": only,
            "net": net_record(r.net_path, r.oracle),
            "equity_sha256": r.table_sha,
            "binary_sha256": binary_sha256(),
            "boot": {"replicates": r.boots, "seed": r.boot_seed, "bound": "97.5th percentile of bootstrap means (upper end of the 95% interval)"},
            "threads": job.threads,
            "tt_mb": job.tt_mb,
            "peak_rss_mb": peak,
            "wall_s": t1.elapsed().as_secs_f64(),
            "setup_s": t1.duration_since(r.t0).as_secs_f64(),
            "hard": hard_lines,
            "both_rules": both.iter().map(|(d, w, same)| serde_json::json!({"id": d, "w": w, "identical": same})).collect::<Vec<_>>(),
            "verify_rule": verify_rule,
            "sets": set_docs,
        });
        write_json(out, &v)?;
    }
    Ok(())
}

/// `threads` shared out among checks of `n[i]` worlds: one each first, then one at a time to
/// the check with the most worlds per thread so far (ties to the first), never more than a
/// check's worlds. Every check gets at least one thread, even when that makes more than
/// `threads`.
fn split_threads(threads: usize, n: &[usize]) -> Vec<usize> {
    let mut t = vec![1usize; n.len()];
    for _ in n.len()..threads {
        let best = (0..n.len()).filter(|&i| t[i] < n[i]).max_by(|&i, &j| {
            // n[i] / t[i] against n[j] / t[j], exactly; ties to the first.
            (n[i] * t[j]).cmp(&(n[j] * t[i])).then(j.cmp(&i))
        });
        match best {
            Some(i) => t[i] += 1,
            None => break,
        }
    }
    t
}

/// `hybrid-check --explain IDS`: why the hybrid and exact search choose differently on the
/// decisions `IDS` (`search::hybrid_explain`), instead of the check. Exact values are recomputed
/// (and must equal the stored ones when `--worlds` is given); `--explain-worlds N` (default 3)
/// details the N most misleading worlds of each decision; `--explain-json FILE` writes every
/// world (the principal positions of those N only).
fn hybrid_explain(
    a: &Args,
    ids: &str,
    ds: &[DecisionSpec],
    stored: &BTreeMap<u64, WorldRows>,
    job: &search::hybrid::CheckJob,
    ctx: &Ctx,
    oracle: &nn::Oracle,
) -> Result<(), String> {
    use search::hybrid_explain::{
        ExplainJob, explain, pooled_frontier_error, render, render_frontier_error,
    };
    let ids = parse_ids(ids)?;
    let chosen: Vec<DecisionSpec> = ds.iter().filter(|d| ids.contains(&d.id)).cloned().collect();
    if chosen.len() != ids.len() {
        return Err(format!(
            "--explain: {} of the {} ids are among the decisions given",
            chosen.len(),
            ids.len()
        ));
    }
    let xjob = ExplainJob {
        cut: job.cut,
        seed: job.seed,
        label: job.label,
        to: job.to,
        threads: job.threads,
        tt_mb: job.tt_mb,
    };
    let t = Instant::now();
    let mut xs = explain(&chosen, &xjob, ctx, oracle, Some(oracle))?;
    for x in &xs {
        if let Some(r) = stored.get(&x.id) {
            for w in &x.worlds {
                if r.rows.get(w.world as usize) != Some(&w.exact) {
                    return Err(format!(
                        "decision {} world {}: the stored exact values differ from this binary's",
                        x.id, w.world
                    ));
                }
            }
        }
    }
    let top = a.num("--explain-worlds", 3)? as usize;
    println!(
        "H_{} vs exact, explained: {} decisions x {} worlds (seed {}, label {}, {} source); net {} {:?}; {:.1} s on {} threads{}",
        job.cut,
        xs.len(),
        job.to,
        job.seed,
        job.label,
        ctx.source.name(),
        &oracle.sha256[..12],
        oracle.net().widths(),
        t.elapsed().as_secs_f64(),
        job.threads,
        if stored.is_empty() {
            ""
        } else {
            "; exact values equal to the stored ones"
        }
    );
    println!(
        "network error at the hybrid's frontier positions, all explained decisions: {}",
        render_frontier_error(&pooled_frontier_error(&xs))
    );
    for x in &xs {
        println!();
        for l in render(x, top) {
            println!("{l}");
        }
    }
    if let Some(out) = a.get("--explain-json") {
        for x in &mut xs {
            x.keep_lines(top);
        }
        let v = serde_json::json!({
            "cut": job.cut,
            "seed": job.seed,
            "label": job.label,
            "worlds": job.to,
            "net_sha256": oracle.sha256,
            "binary_sha256": binary_sha256(),
            "decisions": xs,
        });
        std::fs::write(
            out,
            serde_json::to_string(&v).map_err(|e| e.to_string())? + "\n",
        )
        .map_err(|e| format!("{out}: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use search::oracle_labels::{HarvestDecision, HarvestJob, position_schedule};

    /// Threads are shared out by worlds, never more than a set's worlds, at most `threads` in
    /// all (each set at least one).
    #[test]
    fn threads_are_shared_by_worlds() {
        assert_eq!(split_threads(184, &[31, 355]), vec![15, 169]);
        assert_eq!(split_threads(184, &[31]), vec![31]);
        assert_eq!(split_threads(4, &[1, 9]), vec![1, 3]);
        assert_eq!(split_threads(1, &[5, 5]), vec![1, 1]);
        assert_eq!(split_threads(8, &[2, 2]), vec![2, 2]);
        assert_eq!(split_threads(3, &[]), Vec::<usize>::new());
    }

    fn decision(position: u8, distinct: u32, evals: u64, rows: u32) -> HarvestDecision {
        HarvestDecision {
            id: 0,
            trick: 2,
            position,
            stratum: 1,
            cands: 3,
            evals,
            distinct,
            rows,
            search_nodes: 10,
            race_distinct: distinct / 4,
            keep_distinct: 0,
            risk_reached: Vec::new(),
            risk_rows: Vec::new(),
        }
    }

    /// The harvest sidecar names its driver: `exact`, or `net` with the network's SHA-256, file,
    /// trick, head and widths (the network drove the search); and gives the mean positions
    /// reached and evaluations per decision for each seat.
    #[test]
    fn harvest_record_names_the_driver() {
        let h = HarvestJob {
            trick: 3,
            count: 40,
            seed: 7,
            threads: 4,
            cap: 16,
            tricks: vec![2],
            schedule: position_schedule([1; 8]),
            strata: vec![1, 2, 3, 4],
            race: None,
            keep: None,
            siblings: false,
            risk: None,
        };
        let ds = [
            decision(1, 2000, 2500, 16),
            decision(1, 1000, 1100, 16),
            decision(2, 300, 310, 8),
        ];
        let net = nn::synthetic::random_net(5, &[16, 8], nn::outcome::vocab(3), 100);
        let meta = BTreeMap::from([("head".to_string(), "outcome-v1".to_string())]);
        let bytes = nn::oracle::write_net(&net, 3, &meta);
        let o =
            nn::Oracle::from_bytes(&bytes, Arc::new(nn::synthetic::SyntheticLeaf)).expect("loads");
        let shares = "0.25,0.25,0.5";
        let w = [1; 8];
        let r = harvest_record(
            &h,
            shares,
            &w,
            &ds,
            64,
            Some((&o, "/runs/o3/int.safetensors")),
        );
        let d = &r["frontier_driver"];
        assert_eq!(d["kind"], "net");
        assert_eq!(d["drove_the_search"], true);
        assert_eq!(d["net_sha256"], equity::sha256::sha256_hex(&bytes).as_str());
        assert_eq!(
            (
                d["net_file"].as_str(),
                d["net_trick"].as_u64(),
                d["net_head"].as_str()
            ),
            (Some("int.safetensors"), Some(3), Some("outcome-v1"))
        );
        assert_eq!(d["net_widths"], serde_json::json!([16, 8, 24]));
        assert!(
            r["frontier_values"]
                .as_str()
                .is_some_and(|v| v.contains(&o.sha256) && v.contains("solved exactly")),
            "{r}"
        );
        assert!(
            r["what"]
                .as_str()
                .is_some_and(|v| v.contains("only the sampled"))
        );
        assert_eq!(
            r["distinct_mean_by_position"],
            serde_json::json!([1500.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(
            r["evaluations_mean_by_position"],
            serde_json::json!([1800.0, 310.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(
            r["decisions_by_position"],
            serde_json::json!([2, 1, 0, 0, 0, 0, 0, 0])
        );
        assert_eq!(
            r["rows_by_position"],
            serde_json::json!([32, 8, 0, 0, 0, 0, 0, 0])
        );
        let e = harvest_record(&h, shares, &w, &ds, 64, None);
        assert_eq!(e["frontier_driver"], serde_json::json!({ "kind": "exact" }));
        assert!(
            e["frontier_values"]
                .as_str()
                .is_some_and(|v| v.starts_with("exact"))
        );
        assert_eq!(
            e["distinct_mean_by_position"],
            r["distinct_mean_by_position"]
        );
        assert_eq!(e["strata"], serde_json::json!([1, 2, 3, 4]));
        assert_eq!(e["race"], serde_json::Value::Null);
        assert_eq!(e["decisions_capped"], 3);
        // A race harvest from stratum 4: its settings and the race positions reached.
        let race = HarvestJob {
            strata: vec![4],
            race: Some(search::oracle_labels::RaceKeep {
                classes: 0b10100,
                cap: 256,
            }),
            cap: 0,
            ..h
        };
        // A targeted harvest's record: the file, its groups with their rows and reached
        // positions, the caps.
        let preds = nn::predicate::Predicates::parse(
            br#"[{"name": "g", "conditions": [{"feature": 801, "op": "==", "value": 1}], "sd": 0.03}]"#,
        )
        .expect("parses");
        let keep = search::oracle_labels::PredKeep {
            preds: Arc::new(preds),
            cap: 64,
        };
        let mut kd = ds.clone();
        kd[0].keep_distinct = 100;
        let t = search::oracle_labels::GroupTally {
            rows: vec![30],
            rows_in_any: 30,
            reached: vec![150],
            reached_in_any: 150,
            reached_all: 3300,
        };
        let k = keep_record(&keep, Some("/w/preds.json"), 0, &kd, &t);
        assert_eq!(k["predicates_sha256"], keep.preds.sha256.as_str());
        assert_eq!(k["groups"][0]["conditions"], "f801 == 1");
        assert_eq!(k["groups"][0]["rows"], 30);
        assert_eq!(k["groups"][0]["audit"]["sd"], 0.03);
        assert_eq!(k["rows_in_no_group"], 40 - 30);
        assert_eq!(k["decisions_keep_capped"], 1);
        assert_eq!(
            (
                k["cap_per_decision"].as_u64(),
                k["other_cap_per_decision"].as_u64()
            ),
            (Some(64), Some(0))
        );
        let x = harvest_record(&race, shares, &w, &ds, 64, None);
        assert_eq!(x["strata"], serde_json::json!([4]));
        assert_eq!(x["race"]["classes"], serde_json::json!([2, 4]));
        assert_eq!(x["race"]["cap_per_decision"], 256);
        assert_eq!(x["race"]["other_classes_cap_per_decision"], 0);
        assert_eq!(x["race"]["race_positions_reached"], 500 + 250 + 75);
        assert_eq!(x["race"]["decisions_race_capped"], 1);
        assert_eq!(x["decisions_capped"], 0);
        assert!(
            x["decision_generator"]
                .as_str()
                .is_some_and(|g| g.contains("strata[j mod s]"))
        );
    }
}
