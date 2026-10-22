# Working plan log, 2026-10-16 to 2026-10-20 (copied from the session plan; newest sections first; history, not instructions)

# Plan: review verdict, repairs and the road to the trick-1 golden set

## O_2 ON G V2 LABELS: LAUNCH NOW (owner 2026-10-20 ~03:10 Irish: "only focus on an excellent O_2 and test it; kick off on offer 36863856")
The build agents are out until 22 Oct, 20:00 UTC (weekly limit). Everything below uses code already merged on m3-hybrid (9ed76f0 / 6c7acf5).

1. **Labelling run `o2-gv2-labels-1`** on offer 36863856:
   - Machine: Thailand, AMD EPYC 7B12, 245 threads, RTX 3060, $0.30/h. Launch with `offer_id`, cap $3.5.
   - Kind `hybrid_rung`, cut 2:
     - `label_source` "hybrid", `label_net` O_3 f5766a, `label_exact_leaves` learned:leafrisk-o3-v1:t=277338260, `hybrid_exact_share` 0.05;
     - `labels_random` 2.6M (new seed 531) and `labels_rules` 0.6M (seed 532), with `label_parts` 10 and `upload_shards` true, so every finished part is stored even if the cap stops the run;
     - `val_count` 20000 (exact, seed 602);
     - `check` false;
     - training on the GPU at 2048-1024-256 with `label_inputs` = the stored exact random 1.2M (as set `extra`, seed 511), so the run itself ends with a first G v2 O_2.
   - First, a local dry parse on the VM via `runner` tests: copy the manifest into docs/vast and add it to the `manifests_resolve...` test. The rented machine has the GPU, so labelling and training happen on one machine.
2. **Second machine if one appears at ≤ $0.40/h with ≥ 190 threads:** the same manifest with seeds 541/542 and labels only (tiny CPU training), to double the data. The owner found no other cheap machine, so this one is optional.
3. **Test** (after the run): O_2's value error against the exact validation, and decision-level checks with `search hybrid-check --trick 1 --cut 2` on the saved trick-1 worlds, plus the H_3+G v2 reference where affordable.
4. Monitor with heartbeats. Report the number of labels, the cost, and the test result.

## OVERNIGHT 2026-10-20 (owner asleep; "keep going, push t3, t2, t1 below 0.01, levelled"; credit ~$7.5, ~$1.3 spent)

### State at 00:22 UTC (01:22 Irish), after a container restart
- **G v2 is finished and adopted** (o3-learnedleaf-2, $0.65, machine self-destroyed). The learned rule t=277338260 gives:
  - 0.0005% mean / 0.0008% upper overall, and seat 1 at 0.0015% / 0.0022%;
  - at 1,550 thread-ms a world, against race:1-4's 0.0014% / 0.0018% at 1,692;
  - half cost fails at seat 1 (0.0101%).
  - Recorded in rung2.md, and the t2-hlabels manifest uses it: m3-hybrid 08b8b45.
- **Trick-2 hybrid labels are built and merged** (8b41f66). The release binaries were dispatched for 8b41f66; a new release is needed for the latest head.
- **"Deferral one level up" (eng-defer2) was interrupted by the restart.** About 1,450 lines of uncommitted work are in /home/user/41Bots-defer2:
  - runner leafrisk.rs, search exact_leaves/hybrid/pimc/student_teacher, solver frontier, python leafrisk;
  - O_2's leaf-risk model, with a deferral expert of H_3 plus learned exact leaves, for trick-1 decisions.
- **Not launched, on purpose:** the trick-2 audit loop (the known-broken loop: 100% flagged).

### Steps
1. Commit the defer2 work as WIP on eng-defer2 and push, so a restart can't lose it.
2. Relaunch one Opus agent to finish eng-defer2 from that WIP, with the brief as before:
   - tests: "all" equals full deferral, "none" equals plain H_2, thread identity, refusal of a model for the wrong trick;
   - docs;
   - two manifests, not launched: the O_2 leaf-risk row harvest, and the trick-1 measurement against the H_3+learned reference, reusing eng-silver if it exists;
   - full checks, then merge to m3-hybrid.
3. Re-arm the 2-hourly check-in (send_later). When the agent reports:
   - dispatch release-binaries;
   - if the estimated cost of the O_2 leaf-risk harvest is ≤ $1.5, launch it on the cheapest per-work machine (any_host, at least 190 threads, cap $1.5);
   - then train the model locally (fast) and run the trick-1 measurement only if the remaining credit is ≥ $2.5 after the reserve of $1;
   - otherwise leave both priced for the morning.
4. Morning report, 07:30–08:30 Irish:
   - the G v2 table;
   - what launched and its spend;
   - the priced plan to level trick 3, trick 2 and trick 1:
     - O_3 is now covered by search with G v2;
     - O_2 through deferral, plus a repaired audit loop (fix D);
     - trick 1 through more deals and deferral.

### Verification
- The agent's tests and full checks are green before the merge.
- Every rental is confirmed destroyed (DONE marker, or the vast-status workflow).
- Spend stays within credit minus $1.

## DIAGNOSIS FIXES A AND B (owner "Yes", 2026-10-19 ~16:30 Irish)

### Context
The research diagnosis found:
- The audit loops never targeted anything: the confirmed groups cover 100% of positions, because α_sd = 1% is below O_3's overall SD of 1.87%.
- O_3 has no position-level bias at the leader (−0.003%).
- The decision-level bias, own team undervalued −0.033% at seat 1, grows with the number of opponents still to play. That is the signature of the optimizer's curse: search maximising over noisy leaf values.
- Data, not width, is the main lever.

The owner approved building A and B now. The 40M size tests are skipped; sizes will be compared later at 80M, inside fix C. The 80M run (`o3-base80-1`, Greece 9995WX) keeps running.

### A. Prove the optimizer's curse (CPU only, ~$0.5)
- Run the H_3 golden check on the stored trick-2 golden set (2,000 decisions × 320 worlds, `golden_decisions_input` 635e9aef…, `golden_worlds_input` 6ddc59b1…), changing only what is used as "truth":
  - **truth** = O_3's own values at the trick-3 leaves (the exact search through trick 2 is kept);
  - **player** = O_3 values plus deterministic seeded noise with the measured SD per race class, from the round-1 auditor report.
- Build:
  - a `search hybrid-check` option, e.g. `--truth net --noise-sd-by-class a,b,c,d,e --noise-seed S`;
  - fixed-point noise drawn from the project's xoshiro RNG, keyed by (world, leaf) so it is deterministic;
  - a runner `hybrid_rung` check-only path, or a small new chunk kind, to run it on a rented CPU machine.
- **Decides:** if seat 1's own-team bias is about −0.03% and shrinks with seat (seat 4 ≈ −0.006, seats 5–8 ≈ 0), the leader problem is noise × search. Then the fixes are B, C and G, not features.

### B. Ensemble of existing O_3 networks (~$1–2)
- Average the four stored O_3 integer networks: b34e12…, f5766a… (adopted), a09906…, a97ac1… (round 2 today). Confirm the full keys from the round reports.
- The average is integer and index-ordered over the 4 outcome distributions, so results stay byte-identical across machines and WASM.
- Build:
  - `--net` accepts several keys, and the leaf value is their mean;
  - the H_3 evaluator in crates/search reuses the batched integer kernel per member;
  - a differential test against a single network when N = 1.
- Run the golden check for the ensemble against f5766a alone on the same 2,000 decisions, with a **paired bootstrap of the differences** (fix H).
- **Decides:** seat-1 bound, own-team bias and overall.
- Cost in play: leaf evaluations ×4. This must be measured for the app later (student shortlist search), but it costs the teacher and oracle offline only time.

### Order
1. One Opus build agent for A and B together (shared code: hybrid-check options and the multi-net leaf), on branch `eng-curse-ensemble`, with full checks.
2. Merge, release bundle, then one rental runs both checks (CPU-heavy, GPU not needed): any_host, cheapest per CPU work, cap $2.
3. Report to the owner, with the paired comparisons.
4. In parallel: the 80M run finishes, and its golden result is compared with O_3 at 40M.

### Verification
- Unit and differential tests:
  - N = 1 ensemble equals the single network;
  - noise is reproducible under its seed;
  - `--truth net` with zero noise gives decision loss 0 and bias 0.
- Full checks (fmt, clippy, workspace tests).
- The A and B results go into docs/m3-hybrid/rung2.md (O_3) with the numbers.

## ERROR BUDGET TO 0.01 AT THE HARDEST TRICK-1 SITUATIONS (owner agreed, 2026-10-19 ~15:00 Irish)
Hardest situations: the trick-1 leader, other pairs close to the money, 2–3 trumps in hand. Errors add down the chain, so each layer must prove its budget in these situations before the next layer starts. The shares are rough and agreed, not measured:

| Layer | Budget |
|---|---|
| O_3 | ≤ 0.0025 (owner target: flat) |
| O_2 | ≤ 0.003 of its own (≤ about 0.005 with O_3's share) |
| Trick-1 teacher deal noise | ≤ 0.001–0.002 |
| Student plus shortlist search | ≤ 0.003 |
| **Total** | **about 0.01** |

- **Reporting:** every layer's report gets fixed lines for the trick leader and the close-to-the-money scores.
- **Order:** top down: O_3 → O_2 (cheap labels from the improved O_3) → trick-1 teacher and student.
- **Base-size test for O_3:** train from scratch on 50% vs 100% of the stored 40M base, same settings; compare validation SD and golden decision loss.
  - If 100% is clearly better (owner, 2026-10-19), double the O_3 base to 80M or 120M, and use 16M or 24M weak-spot positions a round.
  - Otherwise the levers are weak-spot rounds and network capacity.

## CHEAP MACHINES FOR LABELLING (owner, 2026-10-19 ~11:30 Irish: "Yes")
- **Labelling jobs on cheap machines.** Big labelling jobs (trick-1 student data; O_2's cheap labels) go on cheap 64 GB machines such as the EPYC 9654 at $0.34/h in Russia (192 threads, 64 GB RAM, RTX 5080 16 GB). Training stays a short job on a big-RAM machine.
- **The memory test, before the first trick-1 student batch.**
  - A test of a few minutes and a few cents on such a machine, running the trick-1 teacher with the final O_2.
  - Measure peak RAM against the 64 GB.
  - If needed, lower the teacher's table size so it fits, then re-measure decisions per hour.
  - Go if peak RAM is under about 56 GB and decisions per $ beat the EPYC 7B13 rate (about 12,500/h at $1.06/h).
- **GPU floor:** 16 GB everywhere. The trainer streams rows when they don't fit the GPU, so the run's results don't change.
- **Not now:** the oracle audit loops stay on big machines. Splitting them round by round saves only about $5–7 a run.

## STUDENT TO 0.01 / 0.0075 / 0.005 (owner's question, 2026-10-19 ~08:00 Irish)

### Context
The trick-2 student is at 0.057% decision loss on 1M decisions; the target is 0.01, ideally 0.0075 or 0.005. What we measured:

| Decisions | Loss |
|---|---|
| 50k | 0.186% |
| 100k | 0.134% |
| 250k | 0.078% |
| 1M | 0.057% |

- The last step (4x the data) gained only 27%. That fits loss ∝ N^-0.23.
- **More data alone is hopeless:** at that rate, 0.01 needs about 1,000x the data (about 1 billion decisions).
- **More width does not help:** 1024-512 = 512-256 at 1M.
- **Signs that label noise sets the limit:** each label is a 64-deal average, so it carries noise, and more data helps only slowly. These two observations fit that.
- **The ruler's own floor is unmeasured.** The test set's teacher uses 1,024 deals, so its "best card" is itself noisy. Even a perfect player shows some loss against it (the winner's curse). If that floor is about 0.02–0.04%, then 0.01 cannot be seen with today's ruler at all.

### Steps, cheapest first (each with a go/no-go number)
1. **Measure the ruler.**
   - Cost: free on the VM. The test file stores every deal's value.
   - Method: split the 1,024 deals into halves A and B. Pick each decision's best on A, score it on B. This gives the floor of a near-perfect player.
   - Also re-measure today's students with that split, so their loss is unbiased (pick on A, score on B).
   - Small build: `search student-check --split-half`.
   - Decision: if the floor is above about 0.005%, extend the test set's deals by index, from 1,024 to 4,096. This reuses the stored records; cost about $0.15–0.3.
2. **Shortlist test.**
   - Cost: free.
   - Measure the loss when the student shortlists its top 2 or 3 cards and the teacher chooses among them.
   - If the top-3 loss is about 0.01 or lower, the app's bot = student shortlist + a small search over 2–3 cards. This is proven: AlphaZero-style policy-guided search and Pluribus.
   - Then measure on a phone (WASM) the seconds per decision by place in the trick. The leader is the costliest place and also the weakest one.
3. **Train on what the decision needs.**
   - Cost: about $0.5, one GPU run, same 1M data, an A/B against today's student.
   - Today's loss fits each card's absolute value separately. A card's choice depends only on the differences between cards in the same decision. Those differences are much less noisy, because all cards are valued on the same deals.
   - Add a per-decision term: the KL divergence [how far apart two probability spreads are] between the student's and the teacher's softmax over the candidates. This is policy distillation, Rusu et al. ICLR 2016, already in the plan's research.
   - Stop training on the validation *decision* loss, not the value loss.
   - Settings: `rank_weight` and `rank_temperature` in `python/nn/fortyone_nn/student.py`, plus the `student_train` params.
   - Keep the new loss only if it beats today's on the unbiased measure from step 1.
4. **Extend the close calls.**
   - Cost: about $2–5.
   - Records are extendable by deal index. Add 64–192 more deals only to decisions whose best and second-best are within the noise. Those are the labels that flip choices.
   - This is the owner's "add deals where needed" rule, now driven by the measured noise per decision.
   - Retrain and re-measure.
5. **A student audit loop.**
   - Cost: about $3–8 a round.
   - Harvest new positions where the student loses most. Today that is place 1 (0.084%), place 5 (0.088%) and score class S4 (0.089%), found by the existing auditor on the student's per-decision loss.
   - Add DAgger [positions from the student's own games] (STUDENT.md §12 item 3).
   - Retrain additively. Same machinery as the oracle loops.
6. **Optional:** an ensemble of 2–4 students averages away training noise. It costs 2–4x run time in the app, so only if steps 3–5 stall just above target.

### Order with task #9 (owner, 2026-10-19: "first push O_3 lower, then the student?")
Run them in parallel, with one hold.
- **Now:** student steps 1–3. They are free, then about $0.5, and they buy no new data.
- **Same time:** the trick-3 additive loop (task #9; cap about $15) pushes O_3 lower. It runs on its own machine.
- **Hold:** don't buy new trick-2 student data (steps 4–5) until O_3 is pushed. The student's labels come from the teacher (H_3 with O_3), so data bought now would need relabelling after a teacher upgrade.
- **Why not O_3 first and alone:** today the student's 0.057% is about 8 times O_3's 0.0068%. A lower O_3 would not move the student's number until steps 1–3 close most of that gap.
- **Trick 1:** the trick-1 student's teacher uses O_2, not the trick-2 student. A better trick-2 student does not improve trick-1 labels. What carries over is the method (the ruler, the decision loss, the shortlist), and a better O_2.

### Honest limits
- The student is measured against the teacher (H_3 with O_3), not against the truth. The bot's total error is about the student's loss plus the teacher's own error (O_3: 0.0068% at trick 3).
- 0.005 is only reachable if the teacher itself is pushed lower. That is the trick-3 additive loop (task #9).
- Order and spend: steps 1–2 are free (about half a day of build), step 3 about $0.5, then steps 4–5 as the numbers say. Steps 1–3 report before any data is bought.

### Verification
- Step 1:
  - the split-half floor and both students' unbiased loss, with 95% bounds;
  - an identity check: the full-1,024 measure reproduces today's 0.0566% exactly.
- Step 2: the top-k loss table by place and score class.
- Step 3: A/B on the unbiased measure, same data and seed.
- Steps 4–5: a curve point after each, with spend.

## OVERNIGHT 2026-10-19 (owner asleep: "keep finishing everything; 8 hours of work in the pipeline")
State at 00:03 UTC (01:03 Irish): the container restarted again and the monitors are gone. Both runs live on their own machines:
- `m3-hybrid-t2-loop-2`: round-1 targeted harvest and training, $2.53 spent at 23:58 UTC, cap $20.
- `student-t2-more-2`: 64-deal top-up of 375k trick-2 positions, cap $3.

Steps:
1. Re-arm t2watch on both runs as a background monitor. It is a waiting loop, so run it in the background and never under `timeout`.
2. When `student-t2-more-2` is DONE:
   - check its parts in R2 by key and size;
   - dispatch `docs/vast/student-t2-curve-1m.manifest.json` (bin 253ae3a, 1 GPU with at least 8 GB, at least 32 cores and 64 GB RAM, disk 60, cap $5, the image 20261017-bc692e446939);
   - if more-2 has no merged.json, list its parts as bundles, as the dealtest manifest does.
3. After each trick-2 round: if the run ranks the round best, update the pointer in `docs/m3-hybrid/rung3.md` and make a 7-day link for the app.
4. Safety check-in every ~2 h (trig_01UNRE3no9u2GZdLeqdw1s87 at 01:19 UTC stands). Check vast for stray machines.
5. Morning report at 07:30–08:30 Irish: the 1M student result, the trick-2 rounds, spend, and what comes next (trick-1 data on the final O_2).

## RESTART AFTER THE TOP-UP (owner, 2026-10-18 ~20:30 Irish: credit topped up; "free deal test first")

1. **Trick-2 loop resume.**
   - New run `m3-hybrid-t2-loop-2`: a copy of `docs/vast/m3-hybrid-t2-loop.manifest.json` with `label_inputs`:
     - the stored random shard `fb51c217…` (1.2M);
     - the stored rules shard `2160fefc…` (300k);
     - round 1's audit shard if the reuse format accepts it. Otherwise it is only re-made; that costs about 1 h.
   - Validation (50k) is re-made from its seed, about 30 min.
   - `upload_shards` true.
   - Runner change first (small, Sonnet): upload each base and targeted shard as soon as it is finished, not after all
     three base shards (owner's concern, 2026-10-18). Tests; full checks; release binaries.
   - Then dispatch: cpu_match "9654|7742", min_cores 240, max_dollars 20 (the run total stays under $35).
2. **Trick-1 cost test:** re-dispatch `docs/vast/student-t1-cost.manifest.json` (bump `run_id` to
   `student-t1-cost-2`), cap $2.
3. **Free deal-count test (the owner's choice before buying more student data).**
   - Small build: `student encode --max-deals N` uses each record's first N deals (an exact subset, the same as a
     shorter teacher run). Add a `deals` list to `student_train` (its curve takes `(size, width, deals)` triples).
   - Run on the stored 630k trick-2 positions (base + more-1's 75 parts):
     - 512-256 at an equal teacher budget: 32 deals × 630k, 64 × 630k, 128 × 315k, 320 × 126k;
     - also 64 vs 320 deals at an equal 126k positions.
   - The fixed 1,024-deal test set is unchanged. About $1, one GPU machine.
   - Report the loss for each. The choice of deal count and how much more data to buy (trick 2 and trick 1) follows.
4. The student top-up (`student-t2-more-2`) waits for step 3's verdict (owner).

Verification:
- Each launched run: first heartbeat with usable threads and rate; shard uploads confirmed in R2 by key and size.
- The deal test's table, with the identity check that `--max-deals 320` reproduces today's 250k curve point exactly.

## BUDGET TO FINISH THE STUDENTS (2026-10-18 ~15:55 Irish, after vast credit ran out at 15:30)

All three machines stopped when the vast balance hit zero, and the clean-up destroyed them. Saved:
- trick-2 base shards and the round-1 audit shard;
- the provisional O_2;
- 75 of 150 student-extension parts (375k).

Lost: about $1 of work. Estimate to finish both students, with the owner tops up vast first:

| Item | Cost |
|---|---|
| Trick-2 student: the last 375k positions (about $8) and the 1M retrain (about $1) | about $9 |
| O_2 final: resume the trick-2 loop from stored labels, rounds 1–4 | about $10–12 (run total about $25–27) |
| Trick-1 cost test | under $1 |
| Trick-1 student data: 500k–1M decisions at 64 deals | about $30–65 |
| Trick-1 test set (4,096 decisions at 256 deals) | about $4 |
| Trick-1 training | about $2 |
| **Total** | **about $55–90** |

How the trick-1 figure is reached:
- The build-VM measure is 1.2 thread-seconds a deal with the provisional O_2, so 77 thread-seconds a decision at 64 deals.
- 1M decisions is then about 21,000 thread-hours: about 87 h on a 245-thread machine at $0.72/h.
- The cost test pins this down. Fewer deals per decision, or a learning curve that flattens early, lowers it.

Order: top up, trick-2 resume and the cost test in parallel, then the student top-up, then the trick-1 data on the final O_2.

## NOW (2026-10-18 11:08 Irish, after a second container restart): keep both runs watched, then O_2 and the 1M student

State, unaffected by the restart (both runs are on their own machines):
- **Trick-2 run:** rules labels 257k/300k, no warnings, $11.58. Validation about 12:05 Irish, then the three base
  shards upload, then round-1 training. Provisional O_2 about 13:00 Irish.
- **Student extension** (`student-t2-more-1`, Threadripper PRO 7995WX, 184 threads, $1.14/h): part 18 of 150, about
  5 min a part. The $12 cap stops it near part 120, about 600k extra. The owner was offered a $3 top-up later.
- Done earlier: steps 1, 3 and 4 of the section below (student merged; WASM rebuilt; student base data 254k; curve:
  512-256 at 250k = 0.078% vs rules 0.40%).

Steps:
1. Re-arm the run monitors. They are lost on every restart; the 13:00 Irish check-in (scheduled) still stands.
2. At about 12:15 Irish, confirm the three base shards in R2 (key and size vs sidecar) and tell the owner.
3. At round 1: provisional O_2 (fetch, SHA check, put-input, rough test, docs). See below.
4. When `student-t2-more-1` ends:
   - verify its parts;
   - if the owner wants it, run a top-up with `skip_parts` / `first_part` for the missing parts;
   - then a learning-curve run on everything, with sizes including the full count and widths 512,256 and 1024,512
     (a new manifest from `student-t2-curve.manifest.json` with train_inputs from both runs).
5. Future runs: upload each base shard when it finishes, not after all three (an owner concern, 2026-10-18), as a small
   runner change before the next trick-2 or trick-3 run.

## AFTER THE CONTAINER RESTART (2026-10-17 22:55 Irish): finish the student merge, keep the run watched, start the student's data

### Context
The session's container restarted. The trick-2 run (`m3-hybrid-t2-loop-1`) is on its own machine and is unaffected:
- at 22:55 Irish, 369,061 of 1.2M random labels, about 1,620 a minute, CPU 100%, 245 threads, no warnings, $2.76 spent;
- random labels end about 07:30 Irish Sunday; the provisional O_2 (round 1) about 12:00–13:00 Sunday;
- the whole run ends about 01:00 Monday, for about $20–27.

The student pipeline is built and pushed (`eng-student`, 4 commits; docs/STUDENT.md). Its merge into `m3-hybrid` was in
progress when the container restarted: MERGE_HEAD 0af9aa2 in /home/user/41Bots-merge, with two conflicts.
- `crates/runner/src/sched.rs`: the silver set's `search_measure` input check and the student's `student_teacher` input
  check sit at the same spot.
- `crates/runner/src/v2.rs`: two doc bullets at the same spot.

Both sides are additions, so both are kept. The background checks and monitors died with the old container.

### Steps
1. **Finish the merge.** Keep both blocks in `sched.rs` (two `if kind == …` checks one after the other) and both bullets in
   `v2.rs`.
   - Commit, then park it on `m3-hybrid-staging`.
   - Run all checks: fmt, clippy `-D warnings`, `cargo test --workspace --release`, and python/nn `unittest discover`.
   - Push to `m3-hybrid` only when every check is green.
2. **Re-arm the watch on the trick-2 run.** The t2watch monitor, re-armed every 30 min, plus a safety check-in every 3–4 h.
   Act only on a stale heartbeat, a FAILED mark, or a rate drop below about 1,200 a minute.
3. **Rebuild `crates/wasm/dist`** (engine.wasm, engine.mjs, SHA256SUMS) so that `loadStudent` and `method: "student"` are
   in the shipped module. Run `tools/wasm_check.sh` with the real O_3 and table. Tell the owner the new SHA-256 for the app
   session.
4. **The student's first teacher data (owner approves this plan = yes; about $1.5 in all).**
   - Machine: CPU-only, at least 190 threads, at least 96 GB RAM.
   - Data needs no GPU, so it runs beside the trick-2 run on a second machine (parallel rentals allowed).
   - **Mechanism test:** `docs/vast/student-t2-mech.manifest.json` (a few minutes, cents).
   - **Base run:** `docs/vast/student-t2-base.manifest.json`, cap $5. It makes a test set of 4,096 decisions at 1,024 deals,
     plus 200k rules-bot and 50k mixed trick-2 decisions at 320 deals: about 1.4 h, about $1.
   - **Training on the build VM or a small GPU job:** learning curve at 50k, 100k and 250k decisions, sizes 256-128 and 512-256.
     Report the decision loss against the teacher on the test set, by place and score class, against the rules bot and
     random play.
5. **Provisional O_2 at round 1:** as in the section below.
6. **Then re-measure the trick-1 teacher's cost with the trained O_2.** The student's estimate of $40–150 per million
   trick-1 decisions used random networks. Trained networks evaluate about 3–5 times faster (O_3 trained: 28.7 µs per
   evaluation; random 2048-1024-256: 107–159 µs). Price trick-1 student data for the owner from the measured rate.

7. **Shard safety (owner, 2026-10-18).** As each label shard of the trick-2 run completes, confirm it in R2 under
   `inputs/sha256/<key>`, its size, and its listing in `shards.json`. The first is the random shard, about 08:30 Irish.
   Report each one to the owner. R2 Paid is confirmed, so going over 10 GB is only billed, never refused.
8. **No machine switch mid-run (owner asked twice: AMD EPYC 9K84 at $1.87/h, AMD EPYC 9654 with a 6 GB GPU at $1.01/h).**
   An in-progress shard exists only on the machine. The cost per unit of work is the same or worse. Both GPUs are under the
   16 GB floor. Stay on the AMD EPYC 7742.

### Verification
- Merge: all checks green before `m3-hybrid` moves; `git log` shows the merge commit with both branches.
- WASM: `tools/wasm_check.sh` identical native vs wasm with the student requests.
- Student data: shards present in R2 under `inputs/sha256/`; the sidecars' stamps (O_3 f5766aa9…, equity 64740dab…).
- Spend:
  - every rental is destroyed at the end; the vast list is empty except the trick-2 machine;
  - the trick-2 run's heartbeat is never older than 15 min unnoticed.

## PROVISIONAL O_2 (owner's question, 2026-10-17 ~16:40 Irish): ship trick 2's first network now, the in-spec one later

### Context
The owner asks whether trick 2's first trained network can be promoted as it is (out of spec) so the app can
use it, while the trick-2 run keeps looping, and the in-spec network replaces it later. Yes: it costs nothing
extra and never interrupts the run. Facts that shape it:
- **The run is unchanged:**
  - **Before it launches:** the stall fix (eng-harvestpar, in progress), merge and checks, then mechanism
    test 7 on an EPYC 9654.
  - **The run itself:** about 30 h and about $30, cap $35.
  - **Round 1's network:** trained after the base labels, about 14-15 h in. Later rounds follow about every
    5 h, and the run's own report names its best round.
- **The run's network can be read while it runs:** every 5 minutes the run uploads its partial outputs,
  including the network files (`partial/rung/000000/out/hybrid/net/{int,float}.safetensors`,
  `net/report.json`, `audit/round-r/net/...`), as mechanism test 2 did. The run also tries to share each
  integer network content-addressed (`inputs/sha256/<key>`). Mechanism test 5 showed that this share can
  time out on a slow uplink; it is only a warning, and the partial copy remains.
- **What the app can do with it:**
  - **Trick-1 reviews and Guide numbers** (the "judge", APP-ENGINE-CONTRACT.md §4.2 and §6.5) need O_2: H_2
    costs tenths of a second a deal, against seconds to minutes with only O_3.
  - **The bots that play** need the student network (contract §4.1, not built); a raw O_2 does not make the
    phone bots faster.
  - **The engine's WASM API** (contract §5.1, `evaluate(...)`) is not built yet. It was planned "after the
    current hybrid rounds". Without it the app cannot call any network.

### Steps
1. **Launch the trick-2 run exactly as planned**; no design change.
2. **At round 1** (about 14-15 h in):
   - fetch round 1's integer and float networks and its `report.json` from the run's partial uploads;
   - check the SHA-256 against the run's round-1 report;
   - store both content-addressed (`launch.py put-input`);
   - run a quick local check on the build VM: load it, run `search hybrid-check --trick 1 --cut 2` on the
     saved trick-1 worlds (the rough test; a few minutes).
3. **Record it as "O_2 provisional, round 1 — out of spec"** in a new `docs/m3-hybrid/rung3.md`, in
   `docs/STATUS.md`, and in APP-ENGINE-CONTRACT.md (network key and widths for the `engine` version
   string). Record its measured numbers too: validation loss, the audit's overall SD and bias, the
   confirmed groups, and the rough test.
4. **After each later round:** if the run ranks it best, update the provisional pointer the same way. A
   swap is one changed key in the contract, and the engine's `engine` string tells every result which
   network made it.
5. **At the run's end:**
   - the run's best network becomes the O_2 candidate;
   - test it against the trick-1 silver set (eng-silver; the owner chooses the size, from $5-10 for
     400 × 80 to $42-82 for 400 × 320);
   - adopt it as O_2 if it is in spec, else keep it provisional and run the additive loop on the stored
     labels.
6. **Recommended, so the app can actually use it:** start the engine's WASM API build (contract §5.1) now,
   in parallel, with no rental. Then the provisional O_2 plugs in the day it lands, and the app gets the
   API, the plan strip and the version string for O_3 and exact search too. Bots that play stay on the
   app's two-trick Pro until the student network exists.

### Verification
- The fetched network's SHA-256 equals the run's report.
- The integer network loads and gives byte-identical values on the VM and in WASM (`tools/wasm_check.sh`
  style).
- The rough test runs.
- The contract and STATUS name the key and say "provisional".
- The run is never stopped or slowed.

## NEXT PHASE (drafted 2026-10-17 06:50 Irish, after the overnight loops; owner to choose below)

### Context
The overnight audit loops ran four rounds each and were stopped on the owner's diminishing-returns rule; all
machines are destroyed (spend ~$17.5). Full tables: `docs/m3-hybrid/audit-loop.md` §8, `docs/STATUS.md`.
- **O_4 passes** at 0.0016% on a 4,000-decision golden set (every seat band ≤ 0.006%, bias ≤ 0.005%): round 1,
  1024-1024-512-128, `inputs/sha256/b28750e3…`. 13 groups still fail at the decision level (0.011-0.074%), and 62% of
  them are race-for-the-money conditions (8% of passing groups): the known weakness, concentrated.
- **O_3 does not yet pass everywhere**: best round 2 (2048-1024-256, `inputs/sha256/f5766aa9…`): 0.0068% overall
  but seats 1-2 at 0.012-0.016% and the leader's own team under-valued by ~0.02% there. Its failing groups are mostly
  ordinary groups whose slices are small (median 89 decisions vs 299 for passing ones) around a mean of 0.006%: the
  problem is the network's overall level at the leader's decisions, not particular situations.
- **"Flat" was out of reach by design**: 10M-row audits confirm thousands of position-level groups (SD > 1%), and flat
  needs each measured at decision level; position SD predicts decision failure only moderately (AUC 0.77-0.79), so a
  plain threshold change would not be sound on its own. Retraining from scratch each round is noisy (trick-4 bound
  0.0016 → 0.0021 → 0.0016 → 0.0024%) and more width did not help.
- O_4 is not on the critical path (exact search from trick 4 is as cheap as a network evaluation); O_3 proves the
  method at the hard trick; O_2 (loop built on branch `m3-hybrid-t2loop`) is what trick 1 will use.

### Owner's answers (2026-10-17 morning)
- Adopt O_4 = round 1 (b28750e3) and O_3 = round 2 (f5766aa9) as the current best: **yes**.
- "Each round retrains from scratch? I thought we add scenarios and keep training on them, not re-spend money on
  intelligence already won... can't we use the current network and just add to it?" → the loop becomes **additive**.
  Facts for the record: labels are never re-made within a run (only re-generated, deterministically, on a new launch;
  the golden sets are stored); what each round re-spent was training from scratch (trick 3: 289 of 558 step-minutes;
  trick 4: 133 of 413), and that also made rounds noisy.

### Owner's direction (2026-10-17, later)
"Our current approach gets a network pretty flat; from the real examples we predict what trick 2 will take. At some
threshold we switch to very focused, additive training that only fills the small out-of-spec groups — as a run setting;
trick 2 runs like tricks 3 and 4 first, and we decide together when to go additive-only."
So: one loop, two **modes chosen per run in the manifest**: `train_mode: "scratch"` (today's behaviour: broad
improvement, capacity steps) and `train_mode: "additive"` (start from a stored network `init_net`, fine-tune on old +
new data, keep only if no worse, harvest only failing / small groups). Switch point, decided together per trick: the
golden set passes overall and in every band, and the rounds' changes are within noise (trick 4 reached it in round 1,
trick 3 after round 2). Machines: the focused phase is *more* CPU-bound, not less — finding and proving rare groups is
CPU work (rejection sampling, exact labels, mini golden sets), while additive training shortens the GPU part; keep
many threads, any decent GPU (trick 4 alone wants a fast GPU while it still trains from scratch).
Trick 2 cost driver: exact labels (~3 s each, ~300x trick 3), not training; so for trick 2 the big saving is never
relabelling (`label_inputs`), and the additive switch matters mainly for noise.

### Trick-2 labels from the trick-3 network (owner's question, 2026-10-17; researched)
Idea: label trick-2 positions with H_3 (exact search through trick 2, O_3 at the start of trick 3) instead of a full
exact solve: ~12 ms (1024-wide O_3) to ~250 ms (2048-wide) a position against ~3 s exact, i.e. 10-250x cheaper.
- **Proven family:** training a value function on the values of a deeper search that uses a learned evaluation at its
  leaves is bootstrapping (TD learning, TD-Leaf); Veness, Silver, Uther & Blair (NeurIPS 2009, "TreeStrap") trained a
  chess evaluation this way from self-play to master level, outperforming TD-Leaf.
- **What it costs in quality (maths, not guesswork):** with frontier values each off by at most e, the deciding team's
  backed-up value is off by at most e (a maximum of numbers each off by ≤ e is off by ≤ e), so errors carry forward and
  add along the chain: O_2's error ≈ H_3's error + O_2's own fitting error. Caveat for four teams: when a near-tie flips
  a player's choice inside trick 2, the *other* teams' values in the label can jump by more than e.
- **Consequence:** H_3 must be well inside the standard before its labels can feed O_2 (0.0068% today leaves ~0.003% for
  O_2's own error); the owner's targets (trick 4 at 0.005%, trick 3 at 0.0075%) are the right direction, and ~0.005%
  at trick 3 would leave a comfortable margin.
- **Design that cannot lose quality silently:** cheap H_3 labels for training (millions), exact labels for the audit
  shard, mini golden sets and the decision-level checks (the standard stays exact), optionally 10-20% exact labels mixed
  into training as an anchor. Gate (PLAN §9 spirit: two proven options compete, measure): O_2 trained on H_3 labels vs
  O_2 trained on exact labels at equal money, compared on the same exact checks; adopt only if not worse.
- Setting: `label_source: "exact" | "hybrid"` with `label_net` (the O_3 key) for the training shards; audit always
  exact.

### Recommended next steps
1. Record the adopted networks in `rung1.md` / `rung2.md` (keys above) and in STATUS.
2. Builds (Opus; full checks; mechanism test before any long run):
   a. **Additive training** (proven: warm start with replay, as the self-play engines train continuously; Ash & Adams
      2020 for the warm-start generalisation gap and its fix, shrink-and-perturb): round r starts from the best network
      so far (the run's start can be a stored network, e.g. O_3 round 2), trains on old + new data mixed (never new data
      alone: that makes a network forget what it knew), at a lower learning rate, for a few epochs; optional
      shrink-and-perturb (λ≈0.9, small noise). **Never lose a won pass:** the new network is kept only if the golden set
      is no worse (overall and every seat band) — otherwise the best stays and the new data waits for the next round.
      A step up to a larger network is the one case that starts fresh (its weights cannot be carried over); it too is
      kept only if it beats the best.
      Settings: `train_mode` ("scratch" default | "additive"), `init_net` (stored network key; required for
      additive), `additive_epochs`, `additive_lr`, `shrink_perturb` (λ; off by default); in additive mode the targeted
      harvest takes only the failing and not-yet-passing groups. Both modes covered by the manifest test.
   b. **Reuse stored base labels** on a relaunch (setting `label_inputs`: the shards from a previous run by key), so a
      new run starts at round 2 with zero label cost (essential for trick 2, ~3 s a label).
   c. **Decision-level flat rule that can be met**: measure the riskiest groups first (fitted from each round's
      calibration pairs: SD, |bias|, weight, race flags), flat = every measured group passes + golden set passes +
      fitted chance that any unmeasured group fails < 5%.
   d. **Trick 3 at the leader**: more (and weighted) frontier positions from seat 1-2 decisions; report error by "who
      leads trick 3".
3. Runs: trick-3 additive loop from O_3 round 2 (stored golden set, stored labels, ~6 rounds, cap ~$15) towards
   0.0075% and lower. **No more trick-4 runs** (2026-10-17): O_4 is not needed in play (an exact solve at trick 3,
   ~6 ms a deal, is cheaper than H_4, ~18 ms) nor for any label (trick-3 labels are exact); it proved the method.
   What the app needs: exact search at tricks 3-5, O_3 for trick-2 decisions, O_2 for trick-1 decisions; the
   app's ~2 s per move is met by the small student networks of item D (distilled from O_3 / O_2, checked against
   them), since the 2048-wide O_3 costs several seconds a trick-2 decision even on a desktop. Measure exact trick-3
   decisions on a phone (WASM, 320 deals) before relying on "no network from trick 3".
   Student design (owner, 2026-10-17): one small student distilled from several teachers (multi-teacher policy
   distillation, Rusu et al. ICLR 2016; Hinton et al. 2015): O_2 for trick-2 starts, O_3 for trick-3 starts, exact
   search (or a turned-hand network, only if one is built) for turned hands; inputs tell it the trick start and the
   turned pairs; each teacher proven first; the student checked per task and on a phone for speed; a task that suffers
   from sharing gets its own small head (shared body), else its own small network.
4. Trick 2: build `label_source: "hybrid"` (training labels from H_3, audit and checks exact); then the gate run
   (O_2 on H_3 labels vs O_2 on exact labels, equal money, same exact checks) once O_3 is at about 0.005-0.0075%;
   then the trick-2 loop with the winner, scratch mode first, additive when flat enough (decided together).

### Verification
Unit tests for each build; a mechanism test (2 rounds, small data) before each real loop; per round: golden table by
seat band and bias, decision-level groups measured/failing, the fitted "unmeasured fails" probability; no stray machines.

## OVERNIGHT PROGRAM (owner, 2026-10-16 ~22:00 Irish = 21:00 UTC; owner asleep; times for the owner in IRISH time)
Owner decisions: only tricks 4 and 3 tonight (trick 2 waits); networks may grow to 4096 wide; the auditor must
find groups down to 0.01% of positions (gamma 0.0001); once flat at 0.01% push trick 4 to 0.005% and trick 3 to
0.0075% (owner wrote "0.075", read as 0.0075% since it must be tighter than 0.01%); stop on diminishing returns
unless an engineering fix clearly helps (then build it and try); up to $2/h per machine; search ~20 min for the
best deal at the floor (trick 3: 190 threads; trick 4: 120) before relaxing floors with judgement (e.g. trick 3
to 120 threads); start trick 3 even if trick 4 is not yet flat.
Builds that must be merged (green fmt/clippy/cargo test/python tests) before the real loops: eng-labels (item A,
cheaper labels), eng-nnbatch (item B, batched/wide network evaluation), eng-auditscale (gamma 0.0001 auditor at
scale, `audit_gamma`, `audit_standard`/`audit_push_standard`, diminishing-returns stop, hybrid-check
`--standard`). m3-hybrid-t2loop is not needed tonight. If a build is more than 1 h late after 02:00 Irish, start
without it (A/B are speed only; without E use gamma 0.001 and push later).
Steps: (1) shape comparison lead-5c/6a/6b → choose shape; (2) trick-4 audit loop mechanism test at current
settings (rounds 3, cap $5) as soon as the shape is chosen; (3) when the builds land: real trick-4 loop (gamma
0.0001, standard 0.01 then push 0.005, ladder from the chosen shape to 4096, up to 8 rounds, cap $20) and trick-3
loop (gamma 0.0001, standard 0.01 then push 0.0075, golden trick-2 set reused 7cfc…/644c…, harvest driven by the
best O_3 so far, ladder from the chosen shape, up to 8 rounds, cap $30); (4) check every ~40 min, fix and relaunch
on failure, no stray machines; (5) morning report in Irish time (rounds, groups found/fixed, bounds, spend).

## NOW (2026-10-16 19:26 UTC = 23:26 Mauritius): choose the network shape, prove the audit loop on trick 4, then tricks 3 and 2 together

### Context
The owner's goal: oracle networks that are "proven 0.01" in every situation of the game, found by the machine
(spec §3 amended, `docs/m3-hybrid/audit-loop.md`), with maximum quality first. Today's evidence: more data
did not fix the race-for-the-money positions (lead-4: class-4 error 2.74% -> 2.58% with 2.5x the data), so
capacity is the lever; the integer kernel now loads layers up to 4096 wide (proved, byte-identical below).

### State (all merged on `m3-hybrid` 147bfa4; release binaries for 147bfa4 are in R2)
- Audit loop B1-B3 (auditor, predicates, targeted harvest, mini golden sets, golden slices, the round loop
  with decision-level stop, failing groups re-targeted, capacity ladder 1024 -> 2048 -> 2048x2 -> 4096),
  CPU/GPU overlap, race harvest, width 4096, per-row weights, width guard. All tests green.
- Lead-5b (1024-1024-512-128, the depth test) has waited ~60 min: no >=190-thread GPU machine under
  $1.70/h; the launcher's closest miss is a 128-thread EPYC at $0.67/h.
- Trick-4 data and golden set reusable from R2 (lead-4 seeds; golden e195.../c140...).

### Steps
1. **Thread floor (owner, 2026-10-16 23:30 Mauritius): 120 threads for trick-4 runs; tricks 3 and 2
   keep 190.** Trick-2 audit loop budget: up to $35 (owner), stopping earlier if flat. Cancel the waiting
   lead-5b job (190 floor) and re-dispatch it at 120.
2. **Shape comparison at trick 4** on the lead-4 data (40M rows, golden reused), three runs in parallel,
   bin 147bfa4, cap $3-4 each: lead-5b 1024-1024-512-128 (re-dispatched under the chosen floor), lead-6a
   2048-1024-256, lead-6b 2048-2048-1024-256. Choose the shape by (in order) int class-4 error SD, the
   worst decision-level band bound, validation loss; on a tie the cheaper network. ~1 h, ~$4.
3. **Audit loop test on trick 4** with the chosen shape as the ladder's start: `audit_rounds` 4,
   `audit_rows` 2M, `audit_target_rows` 10M, the lead-4 labels as round-0 data, golden reused, cap ~$6.
   Report round by round (groups found/confirmed/failing, worst group, shape, decision bounds, stop reason).
   Success = it ends "flat" (every confirmed group's decision-level bound <= 0.01% and the golden set
   passes), or shows a clear downward trend with a stated reason if it hits the round limit.
4. **If it works, tricks 3 and 2 at the same time** (floor 190 threads, $1.70/h cap):
   - Trick 3: audit loop from round 3's data, golden trick-2 set reused, network-driven audit shards,
     4 rounds, ~$6-8.
   - Trick 2: one build first (Opus): the loop's decision-level stage at trick 2 is the trick-1 check on
     the saved exact worlds (`hybrid_t1check`, 386 worlds, 32 decisions; mini golden sets at trick 1 are
     unaffordable), so wire `audit` + `check: false` + t1check as the round's decision measure, and report
     honestly that trick 2's decision-level evidence is thin. Labels ~3 s each: round 0 (1.25M) ~5.5 h on
     184 threads, each targeted round ~1M rows ~4.5 h; 4 rounds ~$25-35 (owner approval of this budget).
5. Docs: rung1.md / rung2.md / audit-loop.md results per round, STATUS.md end of day.

### Also open
- App integration: the owner is starting a Remote Control session on the laptop (fivefingers);
  `docs/APP-ENGINE-CONTRACT.md` is the shared record; engine-side WASM API build after the rungs.
- `vast-run.yml` input `min_cores` default is 190; a trick-4 run passes its own floor.

### Verification
Each run: `hc-t3-o4.txt` / `hc-t2-o3.txt` tables, `net/report.json` class errors, the audit loop's
`round-<r>.txt` and `audit-summary.txt`, integer network SHA-256 recorded, vast shows no stray machine.

## EARLIER (2026-10-16 10:50 UTC = 14:50 Mauritius): close rung 1 (O_4), then trick 3
Context: the owner wants the trick-4, then trick-3, then trick-2 networks "proven 0.01" (M3-HYBRID-SPEC §3), fast, on vast GPU machines.

Results in hand:
- GitHub run 38868893704 (4M trick-4 labels, 1024-512-128 outcome head, CPU-trained 52 min, early-stopped): H_4 on the 400 x 320 trick-3 golden set: mean decision loss 0.0052%, 95% UB 0.0083% overall (PASS overall), agreement 88.2%; bands: pos 1-2 UB 0.0113% (fail), pos 3-4 UB 0.0145% (fail), pos 5-8 UB 0.0073% (pass); largest team bias 0.0168% (limit 0.01%). Verdict FAIL by a small margin (morning small run: UB 0.040%).
- vast run m3-hybrid-rung1b-1 (same recipe, RTX 4090 + i9-14900K, bin aea501d): DONE in 8.7 min, $0.10, ~10 s per training pass over 4M rows, machine self-destroyed and verified. Its report is in R2, not yet fetched.
- Built and reviewed today, on branches: m3-hybrid-nnspeed (exact network evaluation 7-12x faster, byte-identical), m3-hybrid-vtrain (GPU rung on vast: runner `hybrid_rung` kind, cu126 PyTorch 2.14.1, fast exact integer check, GPU-resident batches; launcher 90% reliability floor, status-None and two-reading destroy fixes, also on m3-pimc).

Next actions (in order):
1. Fetch the vast run's report (r2-fetch m3-hybrid-rung1b-1; unpack out/rung/*.tar.zst with Python 3.14 compression.zstd) and compare with the GitHub run (same recipe, GPU vs CPU training): §3 tables, epochs, batch profile.
2. Merge m3-hybrid-nnspeed and m3-hybrid-vtrain into m3-hybrid (Sonnet; resolve docs conflicts; fmt/clippy/test; CI green). Copy any workflow change main needs.
3. Close rung 1 with a scaling sweep on vast GPUs (parallel rentals allowed; each run ~10 min, ~$0.10-0.20; price list shown to the owner first): labels 4M / 8M / 16M (new disjoint seeds) x nets 512-256-64 and 1024-512-128; then add harvested frontier positions (H_4's own searches on training decisions; `SOURCE_HARVEST` reserved in the shard format: Opus build of `search labels --source harvest` + a hybrid_rung param) and extra weight where the bands fail (positions 1-4). Pick the smallest net that passes overall, every band and bias <= 0.01%; the golden set is never trained on.
4. Rung 2 (O_3): trick-3 labels (1-3M at ~9 ms each, plus harvest) on a many-core GPU machine; validate H_3 on the trick-2 golden set t2d (400 x 320); same sweep method. Then rung 3 (O_2), priced first (trick-2 labels ~3 s each).
5. Owner-visible report per rung in docs/m3-hybrid/ and STATUS.md.

Verification: §3 tables from `search hybrid-check` (decision loss 95% UB overall and per band, bias per team, agreement) on the untouched golden set; integer network SHA-256 recorded; vast shows 0 instances after every run.

## Context
The owner asked for a review of everything built since the plan critique (by Sonnet and Haiku): is the rationale right for the strongest 41 bots, is the quality good, what engineering was missed, is the run set-up correct, and what must be rebuilt. Three read-only reviews were run (docs/rationale, code, solver). Owner decisions taken on 2026-10-16 (Mauritius): finish the trick-1 deal-count measurement first with the saved sets, then do the speed engineering; the yardstick must search the robber's burn (no rules-bot shortcut).

## Verdict in short
- **Direction is right.** Measured evidence (trick 2: rollout-PIMC loses 13x more money per decision than exact PIMC) justified moving from cheap rollouts to exact solving early. It just never got written into PLAN/M3-SPEC/CLAUDE.md.
- **Core code is sound.** No change to solver answers or world sampling; per-deal saves and resume give byte-identical outputs; fixed-point money everywhere; no unsafe/HashMap/TODO in the new Rust.
- **The run set-up is what cost money.** All three trick-1 runs on 2026-10-15 were killed by *our own* hourly `vast-cleanup` job (2-hour cap), not by the hosts. Earlier, run B was lost to a hung upload (now fixed). The launcher can still destroy a healthy machine on one failed API call.
- **The yardstick statistics are thinner than the spec allows** and the golden-set size (128 x 200) is a code default, not a derived number; its cost rests on a leader-deal estimate that is probably 2-3x low.
- **Big engineering wins were missed, all answer-neutral:** table entries are 112 bytes where ~28 would do (3.5-4.7x more table per GB); tables are budgeted at 2-3x their real size; the scheduler splits deals over threads mainly to get them memory, which halves throughput; forced last tricks are searched through the table.
- **One quality gap:** in imagined deals where another seat robs, that seat's burn is chosen by `rules-v1`, not searched. Owner: search it.

## Owner's run rules (2026-10-16)
- **No time-based kills anywhere.** A run ends only when the machine's own runner finishes (DONE/FAILED, then self-destroy) or a person stops it. Machines are cheap (~$1/h); the owner accepts the risk of a runner crash leaving a machine billing until the next look.
- A per-machine dollar cap ($20) is optional: build it only if it is trivial and works with nobody watching (the runner itself must enforce it, not the launcher); otherwise skip it.
- **Monitoring at most hourly, and only while a run is up.** No 30-second polling, no watcher scripts, no reminders beyond one hourly check. It is fine if the session ends; the runner saves per deal and uploads every 5 minutes, so nothing depends on a session being awake.
- Runs A and B were started by the owner in the other thread on 2026-10-16; that thread owns them. This plan picks up at the curves step.

## Phase 0: run set-up repairs (before the next rental; main checkout, small PRs)
1. Remove every time-based kill:
   - `.github/workflows/vast-cleanup.yml` (on `main`): delete the "older than N hours" rule; keep only exited/offline instances.
   - `python/launcher/launch.py` / `vast-run.yml`: `max_minutes` becomes a warning, never a stop; the job's end (GitHub's 6-hour limit) must not destroy a machine that is still working: the final "destroy" step runs only when the run wrote DONE/FAILED or the person cancelled. A machine that outlives its launcher is finished by the runner's own self-destroy (`VAST_DESTROY_KEY`), which already exists.
   - Runner: no `deadline_utc` by default.
   - If trivial: a runner-side `max_usd` (elapsed x dph from the manifest) that stops-and-uploads at the cap. Otherwise leave it out.
2. `python/launcher/launch.py`: never destroy on a single failed call.
   - `instance_gone` / `silent_state` / `container_start_error`: an error or unparseable reply = "unknown", not "gone"; require 3 consecutive confirmations over >= 90 s before "gone".
   - `latest_heartbeat` / `r2_list`: catch failures inside `watch`; a failed listing is "no new heartbeat", never an exception that trips the `finally` destroy.
   - `create_instance` unparseable reply: re-list instances by label before renting another offer (no double rent).
   - `--offer-id` path: apply the verified / reliability >= 0.98 / whole-machine checks; dry-run must honour `offer_id`.
   - Add `resume_from` as a `vast-run` workflow input. Tests for each (fake vast/aws failures).
3. Resume safety (`crates/search/src/measure.rs` ~1322-1346, 1537, 1572; `crates/runner/src/net.rs` restore): the worlds file and checkpoint must carry and check seed, count, start_hands, ids, trick, to, and the equity table sha256; a mismatch is an error, never a silent reuse. Duplicate checkpoint lines for one world must be compared (equal or fail). `RcloneRemote::list_sized` (remote.rs:431) must fail on rclone error, not return empty.
4. Small fixes: `vast_status.py` read `R2_S3_ENDPOINT`; CI runs `resume_worlds` and `python/tools` tests; delete `*.tmp` from partial copies; fix "mostly positions 5-8" wording (t1bench.rs:1144); guard doc says 95% (M3-SPEC).
5. Docs sync: PLAN amendment A3' (exact PIMC early, owner approved 2026-10-14; rollout stays only as a scalable fallback), CLAUDE.md line 18, STATUS header (checkpoint M3), Joker-burn question closed, VAST-MACHINES net cap $0.10, v2.rs "byte-identical bundle" claim corrected (timing file and `resumed` differ).

## Phase 0b: machine database (owner request 2026-10-16; after Phase 0, same files)
- At the end of every run (runner `result.json`, and `cleanup.py` for handed-over runs) write `machines/<run_id>.json` to R2: cpu_model, threads, ram_gb, dph, minutes, dollars, nodes_total (sum of finished worlds' solver nodes), usd_per_1e12_nodes, worlds_finished, schedule kind (fill / split), tt_mb, trick, chunk kind, outcome, date, offer_id, host_id, reliability.
- Live: every heartbeat (30 s) carries nodes_total, usd_so_far and usd_per_1e12_nodes so far (and the compact log line); every 10 minutes the runner rewrites `machines/<run_id>.json` marked `running` (final write marked `final`), with worlds_finished so the owner can judge how settled the figure is (hard deals run first, so early figures read high).
- `python/tools/machines.py --live`: one line per running machine (cost per node so far, CPU, threads, RAM, $/h, minutes); plain `machines.py`: list records; per-CPU-model averages (n runs, mean/min cost per 1e12 nodes, nodes/s per thread); `--rank` current offers by measured cost (estimate only for unseen chips); `--record-speeds` writes measured factors into `docs/vast/speed-factors.json` so `vast_market.py` ranking improves after every run.
- Record the three 2026-10-15 trick-1 runs and the speed-test waves from R2 retroactively.
- A Cloudflare Worker page over the same records is optional, later.

## Phase 1: finish the trick-1 deal-count measurement (running now, other thread)
- Sets A and B are resuming on a cheap machine started by the owner. Check at most once an hour (vast instance list + the run's latest heartbeat); act only if a run has ended.
- When both are DONE: `search curves --decisions` on the joined files; regenerate `docs/m3-ncurves` with trick 1; report N per position.
- Honesty note in the report: 80 deals gives a referee of 40, so the trick-1 curve is fitted on N <= 20 and extrapolated; 4 decisions per position. If the extrapolation check (held-out) fails the 15% bar used at tricks 2-4, extend the hard positions to 160 deals (price first; option (b) agreed earlier).
- Write the rule that was applied silently: golden N is the N where the fitted loss is <= 0.01% money chance per decision at the 95% upper bound, per position (M3-SPEC §13.1 item 1 replaces the unusable plateau rule). Fix the "stakes per player" slip (4 per player, not 8).

## Phase 2: solver engineering, answers byte-identical (then re-gate)
All gated by the existing fast-vs-naive identity tests plus the M2 identity run (10k worlds, cross thread counts). Compare by wall time per finished deal, not nodes.
1. Compact transposition-table entry (`crates/solver/src/tt.rs`): key = remaining-cards set (53 bits) + the 27-bit state word (owner of each card is fixed by the root hands within one solve); value = 3 x u32 + 2-bit index of the largest entry; u16 generation. Target <= 32 bytes/slot (from 112). Full key still verified; never a partial key.
2. Reserve table capacity at creation (`tt.rs` growth, `runner/src/mem.rs` peak factor, `plan.rs:167-173`): peak = cap, so the memory budget stops reserving 2-3x.
3. Resolve forced tricks directly (all remaining plays forced -> compute trick + bonuses + leaf; no table probe/store).
4. Per-solve leaf cache (scores -> table value) and bit-parallel `Classes::groups` (`equiv.rs:112-132`).
5. Touch child buckets before searching the 8th card of a trick (plain loads; no unsafe). Try `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` in `boot.sh`.
6. Measure: local identity + speed on the 4-vCPU VM; one 10-minute `t1bench` on a 9995WX-class machine (~$0.30).
Expected: 2-4x cheaper per deal before any scheduler change.

## Phase 3: exact robber burn (owner: required before the golden set)
- `crates/search/src/pimc.rs:262-288` (`solver_position`): when another seat robs in a sampled deal, make the burn a decision node for that seat inside the exact solve (max over legal burns by that team's money chance; equivalent-card merging applies to burns; fixed tie order). Record in the row which seat robbed.
- Cost: those deals branch up to ~6x at one node near the root; measure the share of deals affected and the cost multiplier on 16 leader deals before pricing the golden set.
- Rules test for H3/H4 (any card but the trump ace may be burned; dealer's Joker-burn case).
- Note: sets A/B (Phase 1) keep the rules-bot burn; they are used for the deal-count only.

## Phase 4 (done as measurements, 2026-10-16): the trick-1 cliff is measured
- Measured with the reviewed binary (searched burn), 12 GB table, one thread (m3-t1-tables-1, EPYC 7542): leader worlds 0.6-79e9 nodes, median 12e9, 10 of 16 finished; the 6 unfinished were each past 95e9 at the cap, so the mean leader world is >= 47e9 nodes. Second-seat worlds mean 12e9 (16 of 16). Solver: 6.6-7.0M nodes/s per thread. Trick 2 was ~1.8e7 per world: trick 1 costs 1,000-2,500x per world.
- Per-decision noise at trick 1 (ab1, 5 decisions x 80 worlds, old burn rule): one world disagrees with another by 0.5-1.5% money on the best-vs-second difference; the two halves of 40 disagree by 0.04-0.26%, i.e. 4-26x the 0.01% target. 80 worlds cannot measure the curve; the golden N for a leader decision is probably 250-2,000 (trick 2: 243).
- Consequence: a full-exact trick-1 golden set (>= 25 decisions per cell, N ~500) is ~2e15 nodes for the leader position alone: ~$300-700 and ~2 machine-weeks at 2e9 nodes/s; the plan's $17-35 was 20-30x low. Old-binary evidence (run b vs ab1): a few worlds at a time sharing giant tables beats one world per thread by >10x; the two layout tests running now (m3-t1-leader-layouts-1 on a 7995WX, -9654 on the 9654) quantify it on the new binary.
- Scheduler change (plan.rs) is therefore the opposite of the earlier idea: few worlds at once, each split over many threads with one big shared table, never one world per thread for positions 1-2.

## Phase 5 (revised 2026-10-16): the trick-1 yardstick strategy, staged and measured
Owner decision needed: the full-exact trick-1 golden set is not affordable at the size the spec needs. The recommended path keeps exact solves as the anchor and uses the plan's own M5 method (a depth-limited exact search cut off by the oracle head on the sampled world, REVIEW R:175; precedent Sturtevant & White 2006, Learned Belief Search, AlphaZero distillation) as the cheap instrument, calibrated against exact. Every step below is cheap and has a go/no-go number.
1. **Layouts (no new rentals):** read the two layout tests; write thread-seconds and nodes per world per layout into docs/vast; pick the production layout for positions 1-2; update `plan.rs` admission accordingly (big shared tables, few worlds at once).
2. **Frontier count ($0, VM):** add a depth cut to the exact solver that counts the distinct positions reached at the start of trick 3 (ply 16) for the 26 saved trick-1 worlds (identity of full solves unchanged; the cut is a counter). Go if <= ~1e7 per world. A cut at trick 4 saves nothing (an exact trick-4 solve, ~0.04 ms, is as cheap as a network evaluation); if the trick-3 frontier is too big, the cut moves earlier (start of trick 2), not later.
3. **Labels and network ($1-5):** exact money-chance values (fixed-point, max^n with the documented tie rule) for 1-3M perfect-information positions at the start of trick 3 (8.8 ms each) sampled from the belief sampler's worlds at trick-1 and trick-2 decisions, plus a trick-4 set; train the plan's oracle head (4-way softmax to the money vector, A1) on them. Inference integer-quantised so results are byte-identical across machines (the plan's integer path); floats only inside training.
4. **Validation ($0):** (a) trick-2 golden set, 400 decisions x 320 exact worlds: recompute every world as "1 exact trick + net at trick 3" and report per-candidate value error, choice agreement, and the money-loss error of the hybrid yardstick; (b) trick 1: the 26 + 48 exact worlds from today's runs as "2 exact tricks + net": root-value error per world. Go if the root-value error is <= 0.05% money (sd) with no bias above 0.01%; else enlarge the network or the label set and repeat (the trick-4 cut is not a fallback, see step 2).
5. **Variance reduction, unbiased ($0):** stratify worlds by the big latents (holder of the trump ace, Joker, five; trump counts) and add a control variate (the net's value of the same world, regression-adjusted) to the exact and hybrid world means: fewer worlds for the same precision, answers unbiased.
6. **Trick-1 golden set with the hybrid yardstick (a few dollars):** N from the trick-1 curve measured with the hybrid (cheap: 1-10 s per world), stratified as the spec says; plus an exact-solved anchor subset (16-32 leader decisions at N 200-400 on the chosen layout, ~$50-150) reported alongside with the hybrid's measured error. Values carry a new values version and the network's hash.
7. Plan amendment for the owner: A3' (exact PIMC for teacher targets at tricks 1-2) is infeasible at trick 1 (>= 47e9 nodes per world x N worlds x 1e5+ hands); the trick-1 teacher uses the validated hybrid. The trick-1 yardstick is a diagnostic; duplicate tournaments with SPRT remain the decider between methods (PLAN §9).
8. Not recommended: paranoid or best-reply search (changes the answers), partition search for max^n (no literature), speculative pruning (no four-side proof), a full-exact trick-1 golden set at $300-700 unless the owner wants the gold standard regardless.

## Where things stand (2026-10-16 08:00 UTC = 12:00 Mauritius) and the next actions
Running now, nothing to fix:
- **Hybrid rung 1 full run** (GitHub Actions run 38868893704, ref m3-hybrid): 4M trick-4 labels done (07:48 UTC), trick-3 golden set 400 x 320 regenerated (07:50), training in progress (cap 200 min, so done by ~11:10 UTC), then the H_4-vs-exact check. Results ~15:45 Mauritius.
- **9654 layout run** (instance 56200954, run m3-t1-leader-layouts-9654): 15 of 16 big2x192 worlds saved as of the 06:57 fetch; deal 2 world 2 still running. Spend at 08:00 UTC ~$3.3 (rented 04:43 at $1.006/h; machine cap $10 = ~14:40 UTC).

Actions once plan mode ends (in order):
1. Hourly check of the 9654 run: r2-fetch; if the 16th world is saved or the run has moved to stream mid4x96, dispatch vast-stop (run_id m3-t1-leader-layouts-9654, instance_id 56200954, wait_s 600, reason "owner 2026-10-16: first layout round complete"); otherwise leave it unless spend passes $8 (~12:40 UTC). Report deal 2 world 2 nodes/time when done. Re-arm hourly only while it is up.
2. Schedule one check of the rung-1 run for ~11:45 UTC (15:45 Mauritius); 08:09 UTC check: all steps green so far, training under way. When the rung-1 run ends: if it failed, fix on m3-hybrid (copy workflow changes to main), re-dispatch. If it passed, read the §3 tables (decision loss 95% UB overall and per band 1-2/3-4/5-8, bias for all four teams, agreement), write `docs/m3-hybrid/rung1-full.md`, report to the owner in plain English.
3. Next builds (Opus; Fable reviews): (a) integer network evaluation ~10x faster in `crates/nn` (autovectorised integer loops; an audited SIMD module only if needed, with its differential test), byte-identical outputs; (b) O_3 labels at trick 3 and training on a rented 1-GPU high-core machine (EPYC 9654-class, ~$1/h, priced and approved first; runner needs a training chunk and a PyTorch 2.14.1 image); (c) validate H_3 on the trick-2 golden set t2d; (d) O_2 the same way on trick 1; (e) the hard trick-1 world (decision 0, world 3), then all saved trick-1 worlds.
4. Small builds (Sonnet): launcher treats "instance unknown to vast after self-destroy" as destroyed, not an error; STATUS.md update (hybrid strategy, layout results, spend, open review items).

## Spend so far (for the owner's budget decision)
Original M3 compute approval ~€5. Spent on trick-1 measurement and speed tests to date: about $18.4 (lost run B) + $9.3 (three cleanup-killed runs, work saved) + ~$4.4 (speed-test waves) + ~$1 (first wasted bench) ~= $33. Phases 1-4 above add ~$10. The golden set is the next big item and is not approved yet.

## Remaining M3 items after the golden set (unchanged from M3-SPEC)
Trick-2 default (exact-few vs rollout) decision; belief policy-likelihood arm (§5.2); suitability report (§6) and `docs/M3-REPORT.md`; 10k-hand teacher smoke run (§5.10); PIMC-v0 vs rules-v1 SPRT gate (§8.3); Rust 1.99.1 pin bump; vast_watch dashboard (paused) finished or deleted.

## Verification
- Phase 0: launcher tests with faked vast/aws failures pass; a resume with a wrong seed is refused; `resume_worlds` runs in CI; `vast-cleanup` on `main` has no age rule; a launcher test shows the job's end does not destroy a working machine.
- Phase 1: both runs end DONE; `search curves` output and the regenerated ncurves page; nothing billing afterwards (vast list empty).
- Phase 2: identity tests + 10k-world cross-thread identity run byte-identical; wall time per deal improves on the VM and on one rented bench.
- Phase 3: rules tests; identity against a naive burn-searching reference on small deals.
- Phase 4: thread-seconds per world per layout in docs/vast from the two layout runs; world values byte-identical across layouts.
- Phase 5: step 2 frontier counts per world; step 4 error tables on the trick-2 golden set and the trick-1 exact worlds, with the go/no-go thresholds applied; the exact anchor subset agrees with the hybrid within its stated error.
- Every rental: cheap machine (~$1/h), hourly checks at most, the runner's own self-destroy at the end.

## 2026-10-22: speedtest of offer 57070691 (RTX PRO 6000 S, AMD EPYC 9B45, listed 384 threads, $1.36/h)
- Run `speedtest-rtxpro6000-1` (bin acc0f26, `benchmark` chunk, 300 s). Instance 57165309, machine 154394, verified: no.
- **Result: no speed measured.** The launcher stopped it after 4.7 minutes, $0.11 spent: the container
  got **46 usable threads** of the 384 listed, below the 190-thread floor. No benchmark was run.
- Lesson: the listed thread count is not the container's usable count on this host. The launcher
  checks the container's count (`usable_cores`) before the benchmark, so the run was stopped correctly.
- Speed of this machine: unknown. Not added to `docs/vast/speed-factors.json`.

## 2026-10-22: specialist S (run specialist-s2): trained, not yet tested in play
- Filter (`search risk-filter`, threshold 277338260 = the play rule's cut, O_3 value and G2 score):
  base-random 11,077,690 of 25,600,000 rows (43.3%); base-rules 2,235,652 of 6,400,000 (34.9%);
  base-harvest 3,162,386 of 8,000,000 (39.5%). Total training rows 16,475,728. Held-out: 43,681 of
  100,000 validation rows (43.7%).
- Training: from random weights, 2048-1024-256, 20 epochs, lr 0.001, batch 1024, on the GPU
  (RTX 5090, cu130). 8.8 minutes of training; 16 minutes for the whole chunk.
- Result on the held-out flagged rows (float network): outcome accuracy 75.1%; value error sd 3.03%,
  mean |error| 1.09%, 99.9th percentile |error| 34.0%, max 99.9%; team biases all under 0.03%.
  Worst class: "a team can win the match and another the game" (sd 5.1%).
- Integer network (quantised, what play uses): on 20,000 rows, error sd 2.99%, mean |error| 1.10%,
  p99.9 35.0%; agrees with the float network to 3e-8 sd. Stored in R2 (sha256 of int.safetensors
  d30ef496...), float f5f6768d...
- **Not yet known:** O_3's error on the same flagged rows (needed to compare), and the play result.
  Next: the routed golden check (G2 sends flagged positions to S, O_3 answers the rest), against
  the stored golden set, next to O_3 alone (0.0068%) and G2 with exact solving (0.0008%).
- Cost of the training machine: about $0.55 (RTX 5090, $1.50/h, about 22 minutes).

## 2026-10-22: speedtests of the EPYC 9B45 offers (no speed measured)
- Offers 57070744, 57070738 (RTX PRO 6000 S, listed 384 threads, $1.46/h each), and the earlier
  57070691 (same card): every container got **46 usable threads**, below the 190 floor; the launcher
  stopped each after about 5 minutes (about $0.11-0.14 each). No benchmark ran.
- Offer 55530460 (Threadripper PRO 7995WX, 192 threads, verified, $1.59/h): not on offer when the
  run started; nothing rented.
