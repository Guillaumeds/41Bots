# Status

**Current checkpoint: M3 (in progress)** (PLAN §13, as revised on 2026-10-13): PIMC bots, the exact early-trick yardstick and its N-curves, the trick-1 runs on rented machines, the gate against rules-v1. M0 (engine, reference, RNG, CI), M1 (equity table, arena, launcher) and M2 (solver spike) are done; see `docs/history/STATUS-history.md`.

## Where we are (2026-10-25, the review session: the playout teacher and the money gate)

- **The finding (EXPERIMENTS I1-I2, `docs/RESEARCH.md` §15).** The live trick-1 student leads the J of trumps without the 5 in about 30% of such
  positions expecting +0.6% of the money, and the lead earns -0.45% against players who cash the 5. The lean is in the **reference**, not in any
  copy of it: every teacher so far values a deal with the cards face up, so the 5-holder is never surprised. The Jack lead is one example of a
  family (wasting trumps in one's own or the opponents' hands, not trumping the partner, when to give a trick up for the money): the fix is a
  teacher that values hidden-card play, not rules per pattern.
- **Built and merged to `main` on 2026-10-25 (the owner: "merge everything, you own this repo"):**
  - **#7 playout teacher** (`docs/STUDENT.md` §3.1): each deal is valued by playing it out with the students at every seat from its own view
    (S1 at trick 1, S2 at trick 2), exact from trick 3; `teacher: "playout"` in a `student_teacher` stream; the source `student` (DAgger,
    §4); fixed-thread teacher chunks share the machine (the memory planner). Manifests: `playout-labels` (run 1b), `playout-labels-2`
    (round 2), `student-playout-train` (S1'/S2').
  - **#8/#9 the app's player as an arena bot** (`search::bot::AppPlay`, STUDENT.md §6.1) with the money gate `arena-students.manifest.json`,
    and **deal weighting by the other seats' plays** (`search::inference`, §6.2, integer throughout) with its gate
    `arena-inference.manifest.json`. One 14-way block at a quarter of the app's deals took 164 s on one build-VM thread.
  - **#6 launcher safety:** a machine the owner stops from the console is never destroyed or replaced (A11 destroyed the kept labels box);
    a kept run that ends after the launcher's 330-minute handover keeps its machine (the cleanup writes the record, reading the stored
    manifest for runs launched before); offers both searches miss are read from vast's public market list; `machine.machine_id`.
  - **I3 (EXPERIMENTS):** against run 1's playout test sets the shipped S1 loses 0.17% a trick-1 decision and S2 0.10% a trick-2 decision
    (about three times their face-up loss; 0.12% / 0.075% above the sets' noise): the room run 2 is set to measure.
- **Machines (2026-10-25 07:30 UTC):** the labels box 57517251 is gone (A11, with the third S1 batch `o2-s1more3-3`; whatever parts it uploaded
  before 05:08 UTC are under `runs/o2-s1more3-3/out/`). The owner re-rented the same machine by hand: **instance 57615744** (AMD EPYC 7B12 server
  chip, 256 threads, 129 GB RAM, RTX 3060, $0.328/h), unlabelled, idle, for the runs below; `keep: true` on every manifest that uses it.

### State at 2026-10-27 15:45 UTC (read this first; the blocks below are older)
- **S3, the trick-3 student (docs/STUDENT.md §6.3), built and merged** on the owner's word (2026-10-27: the bots throw lone high
  trumps early; being late in a trick is worth nothing when every deal is played face up from trick 3): PR #12 (536bda1: S3 in the
  playout teacher with a cut of 4, the runner's trick-3 streams, `pimc::evaluate_playout`, the arena's `student3`) and PR #13
  (80b15f3: WASM `method: "playout"`, `loadStudent(bytes, sha, 3)`). Release binaries for 536bda1 are built.
- **Running / waiting:** `playout-labels-2b` (round 2 labels, 7B12 box 57918931, $0.32/h, nearly done; the 17:05 UTC check launches
  `student-playout-3`, the two-arm round-2 training, binaries ca6bed6); **`s3-labels-1`** (vast-run 40531862216, waiting for a 120+-thread
  box ≤ $0.75; then `s3-train.manifest.json` on a GPU box, then `arena-s3.manifest.json` with the chosen S3 filled in, then round 3 of
  S1/S2 labels with `student3` and cut 4); **`arena-inference-1`** (the deal weighting, shipped students in both arms; AMD Threadripper
  3990X, 128 threads, $0.67/h, started 15:00 UTC, about 5 h).
- Spend since 2026-10-25 about $14 of the owner's $30 (with the runs above about $20-22 when they finish).

### State at 2026-10-27 10:40 UTC
- **The money gate accepted the retrained students (EXPERIMENTS I5):** `arena-students-8` on the France box ($0.40, 5.4 h) decided
  **Accept** after 929 blocks: the pair the app plays since 2026-10-26 (S1 `3d511c54…`, S2 `dc5f8ad0…`) takes about **+1% of the money
  a match** (mean block score +0.0096, standard error 0.0039) off the pair before it. The shipping stands; nothing to roll back.
- **Round 2 labels `playout-labels-2b`** run since 04:55 on 57918931 (AMD EPYC 7B12, 245 usable threads, $0.32/h; handed over to the
  cleanup at 10:24, $1.63 so far): the 16 test chunks and the trick-1 streams well under way; ETA about 16:30 UTC, about $3.90. Then
  training (`student-playout-3`, the round-2 labels, the France-class GPU box) and the gate again, then `arena-inference`.
- Spend this round so far: $2.10 (the killed labels run) + $0.40 + the labels in progress; about $11 of the owner's $30 since 2026-10-25.

- 14:20 UTC (EXPERIMENTS I6): on round 2's test sets the shipped S1 beats the old one (0.164% against 0.183%), round 1's seed 3
  `b5850c38` a little more (0.160%); **at trick 2 the old S2 `3d1189e3` beats the shipped `dc5f8ad0`** (0.119% against 0.135%). Round 2's
  training is judged against all three per trick. PR #11 (warm start, round-2 manifest `student-playout-3`) merged at ca6bed6, release
  binaries built for it; the 17:05 UTC check launches the training once `playout-labels-2b` is DONE.

### State at 2026-10-27 04:55 UTC
- The owner topped up vast on the evening of 2026-10-26 and approved an unattended night: "set a watch ... take a machine to do the next
  pieces", within the $30 ceiling of 2026-10-25. Launched 23:14 UTC with a 5h20m waiting price ladder ($0.35 / 0.45 / 0.65 / 0.75 per h,
  5 tries a rung, manifests' `machine` blocks): **`arena-students-7`** never found a 180-thread box under $0.75 (closest: 128 threads at
  $0.65, and the $0.07 France box 57515171 at 128 threads) and gave up at 04:35. **`playout-labels-2`** rented 57893573 (AMD EPYC 7K62,
  184 threads, $0.78/h) at 00:45 and ran 168 min ($2.10): the 16 test chunks and t1rules 0-85 are in R2 (done-list under
  `done/playout-labels-2/`), then the launcher's low-CPU watchdog stopped and destroyed it at 03:35 (CPU under 85% for 300 s): the
  round-2 manifest had `threads: "all"` on every stream (one chunk at a time, a serial setup at 0% CPU between chunks), not run 1c's
  16-thread chunks fifteen at once. Fixed in 107dc4e (threads 16 everywhere).
- 04:50 UTC relaunched both: **`arena-students-8`** on any box of 120+ threads (ladder from $0.12; the France box at $0.07 is the target:
  the owner on 2026-10-25, "keep this france box for everything, no rush"; low_cpu_pct 20, low_cpu_s 900 for the wave tails; max_dollars
  6) and **`playout-labels-2`** again (resumes; 180 threads, 96 GB, ladder to $0.75; low_cpu_pct 40, low_cpu_s 900; max_dollars 10). At
  $0.78/h the labels cost about $11-12 for 14-15 h; the owner was told at 01:20 and can stop the box (nothing done is lost).
- 05:05 UTC: **`arena-students-8` runs on the France box** (57918536, AMD EPYC 7551 server chip, 122 usable threads, $0.09/h; it starts
  from chunk 0: a format-2 run resumes only its own run id, so arena-2's 325 chunks are not reused; about 13 h, $1.2, SPRT may stop it
  sooner). The `playout-labels-2` relaunch was refused ("runs/playout-labels-2/ already exists in R2"): **a launch can never reuse a run
  id**; the done-list resume serves only the launcher's own replacement machine. Relaunched as **`playout-labels-2b`** (the -2 chunks
  redone; max_dollars 14 so the cap cannot cut a $0.78/h run short, which would lose everything again).
- Lesson: **always pass `low_cpu_pct` / `low_cpu_s`** to vast-run.yml for teacher and arena runs (the default 85% / 300 s kills chunk
  turnover and wave tails), and **never `threads: "all"` on a label stream**.

### State at 2026-10-26 06:45 UTC
- **Run 1c done** (11.0 h, $3.62; labels 300/200/400/200 + tests 4/4 under `runs/playout-labels-1c/out/`). **Training `student-playout-2`
  done** (54 min, $0.07, a $0.10/h France box, destroyed by its launcher): on run 1c's playout test sets **S1p** (`3d511c54...`, seed 1)
  loses **0.0897%** a decision against the shipped S1's 0.17% (I3), **S2p** (`dc5f8ad0...`, seed 2) **0.0974%** against 0.10% (no clear
  gain). Only 2 of the 3 seeds per stream came out (the curve has seeds 1 and 2). Both load in the shipped WASM engine.
- **Shipped to players on the owner's word (2026-10-26 06:30 and 06:45 UTC), before the money arena judged them:** fivefingers
  `src/ai/engine41/config.ts` students[1] = S1p, students[2] = S2p (commits 65df9f0, 0d7728b). Rollback = the old keys `97df2a54...` (S1)
  and `3d1189e3...` (S2), still in R2; change them back in that file.
- **Arena `arena-students-2` (new pair against the shipped pair, SPRT [0,0.01], cap 3,000 blocks) is NOT decided**: 325 of 750 chunks done
  (outputs under `runs/arena-students-2/out/`, no verdict read) when the rented Nevada box (57743935, EPYC 7K62, $1.37/h) stopped itself at
  03:24 UTC; it also stopped at 00:47 UTC after run -2's first heartbeat, and vast would not start it again. **Cause: the vast account ran
  out of credit** (every new rental since 04:55 UTC answers 400 "Your account lacks credit"; -3 to -6 failed on it; the owner will top up in a
  day or two). The launcher read both stops as "stopped by the owner" and gave up, leaving the box to boot the old run late (-2 booted
  after the -3 takeover failed). The stopped Nevada box 57743935 still exists (storage billing only, not destroyed; the owner decides).
- **Next, when the owner says the credit is back:** relaunch the arena (next unused id after -6, `docs/vast/arena-students.manifest.json` on
  main, `bin_sha` e2a2c3a5, `vast-run.yml`, cpu_only, max_dollars 8; about 4-5 h, $6-7 on a 190+-thread box) then, if the pair is accepted,
  round 2 (`playout-labels-2.manifest.json`, fill its s1new/s2new with `3d511c54...` / `dc5f8ad0...`; the third seed of each stream was not run).
  If the arena rejects the pair, roll back in fivefingers and record it. Spend so far $3.8 of the owner's $30.
- Fixed meanwhile: `cleanup.py` no longer destroys a just-taken-over stopped instance before its first heartbeat (e2a2c3a; the owner's Thailand
  box 57615744 was destroyed that way at 23:21 UTC 2026-10-25 between his stop and the takeover's boot).

### Handoff (2026-10-25): the runs, in order
1. **Run 1c, playout labels: DONE** (`docs/vast/playout-labels.manifest.json`, run id **`playout-labels-1c`**; `bin_sha` eef2d56b, on
   57615744 from about 10:44 to **21:46 UTC 2026-10-25, 11.0 h, $3.62**, no error, no partial chunk, no warning; the stored parts match the
   training manifest exactly: `out/t1rules` 300, `t1mixed` 200, `t2rules` 400, `t2mixed` 200, `t1test` 4, `t2test` 4, 150 MB in all, checked
   from `r2-inventory`). Pace: the 500 trick-1 chunks by about 18:30 UTC, then about 121 trick-2 chunks an hour at 15 chunks of 16 threads,
   CPU 72-94% (the lower numbers are the trick-2 chunks' tails turning over). No launcher watched it (its launcher gave up before the late
   boot); the runner parked with `KEEP_MACHINE=1` and the box is **still running, idle, at $0.328/h**, relabelled
   `41bots - Review (labels 1c done, idle: stop me for the training takeover) - ETA` (exempt from the cleanup): **the owner stops it in the
   console, then step 2 takes it over** (a takeover of a running box boots late or not at all, see "A machine that already runs our image"
   below; the `stop` dispatch of `vast-instance.yml` is refused for a session on auto mode, so the stop is the owner's). 250k trick-1 and 300k trick-2 decisions at 256 deals with the shipped S1/S2 as the inner players, plus 2,000-decision test
   sets at 512 deals; **16 threads a chunk, about fifteen chunks at once** (the owner, 09:01: the box must run full). The first launch
   (`playout-labels-1`, 07:28-09:21 UTC, one chunk at a time on 245 threads) kept the box 55% busy at about 96 s a chunk (27 hours in all) and
   was stopped with `keep` (its 8 test parts and 42 t1rules parts are stored under its own run id; I3 used its test sets); the planner now lets
   fixed-thread teacher chunks share the machine, so the tails overlap and freed threads take the next chunk at once. Expected: about 15-22 h at
   near-full use (`vast-status.yml` with `run_id` playout-labels-1b shows the queue and `cpu_util_pct`). The run keeps the box (`keep_hours` 8)
   and #6 is on `main`, so the cleanup keeps it at DONE too; **within 8 hours of DONE launch step 2 or the cleanup destroys it at `keep_until_utc`.**
2. **Training S1'/S2': NEXT, ready to launch** (`docs/vast/student-playout-train.manifest.json`, on `main`; run id `student-playout-1`, two
   `student_train` streams, decision-aware loss, seeds 1-3, judged on run 1c's `t1test`/`t2test`; takeover of the same box with `keep`,
   `bin_sha` eef2d56b). The stored parts match its `parts` (300/200/400/200, tests 4/4, checked 22:00 UTC 2026-10-25). Launch through
   `vast-run.yml` once the box shows `stopped/stopped`; then check the heartbeats of `student-playout-1` **and** of `playout-labels-1c` (a
   late boot has started the old run's environment before). Judge **never on copper or the golden sets** (face-up labels): on the playout test sets and in run 2.
3. **Run 2, the money gate** (`arena-students.manifest.json`, run id `arena-students-1`): fill `s1new`/`s2new` with the stored keys
   (`store-input.yml`), then the app's player with the new students against the shipped ones, SPRT [0, 0.01], cap 3,000 blocks (about 500
   core-hours at the app's deals, a few hours on the box). Accept = ship (the owner's word), then **round 2**: replay run 1 with the new students
   as the inner players (the lean they carry is what round 1 removes) and a third of the positions from the students' own games
   (`docs/vast/playout-labels-2.manifest.json`, source `student`: DAgger, built on `playout-teacher`).
4. **Run 3, the inference gate** (`arena-inference.manifest.json`, built: #9) with whichever students run 2 settles on; then the
   equity-table refresh.
5. All four PRs are merged (#6, #7, #9 carrying #8; this docs branch last). Record every result in EXPERIMENTS Part I, this page and
   `docs/README.md`; binaries of `main` eef2d56b (release-binaries run 40179892378) carry everything run 1b and the training need.

## Where we are (2026-10-23, end of the O_2 and S2 sessions)

- **O_2 (trick 2) adopted:** round 1 of the audit loop, 39.6M plain hybrid labels, 2048-1024-256, integer network `13152c24bc27...`
  (`runs/o2-audit-10/partial/rung/000000/out/hybrid/net/`; owner decision 2026-10-23). On the 250k exact test set: error SD 2.60% (the provisional O_2
  had 3.6%), largest team bias 0.019%, sibling regret 0.188%. Fits read across from O_3 put its trick-1 decision loss at 0.016-0.024%, above the
  0.01% line; no trick-1 golden set exists, so that is an estimate (D10). What was learned: plain labels beat exact ones at equal money, volume
  helps the value error slowly and not the decisions, capacity does not help, the audit loop's targeted round did not (D6-D11).
- **O_3 (trick 3) unchanged:** f5766a. The sibling ranking loss at weight 1 was clearly worse than its control (D12).
- **S2 (trick-2 student) shipped** (2026-10-23): `3d1189e3...`, golden loss 0.0389% (3-seed mean 0.0456%); open items in "S2, state" below.
- **S1 (trick-1 student), first results (2026-10-24, D13-D14):** the 250,000 teacher labels (H_2 + O_2 round 1, 64 deals) are stored (run
  `o2-copper-s1-1`, $4.95); S1 trained at 250,000 decisions: the decision-aware loss gives **0.0472% (3 seeds) against 0.0853% plain** on the S1 test set
  (2,000 decisions, 128 deals; copies-its-teacher, not truth), rules-v1 0.343%, random 0.770%. Decision-aware seed 1 at 50,000 / 100,000 / 250,000 decisions:
  0.0824 / 0.0679 / 0.0469, still falling (about 0.03% extrapolated at 1M, one seed); half the deals (32) 0.0512. On the copper set (owner, 2026-10-24: the bronze set never finished, one complete decision):
  decision-aware 0.0416% (seeds 0.0404 / 0.0386 / 0.0457) against plain 0.0766%, rules-v1 0.371%. Not shipped. Next: the handoff below.
- **Machines:** none rented, no watcher on (2026-10-24 12:30 UTC). Rules (CLAUDE.md): rare cheap machines are stopped by hand, never killed by a cancel;
  labels read `41bots - <session> (<what>) - ETA <HH:MM Irish>`; every rental is Jupyter + SSH. **Binaries: `bin_sha` dd19e4d66cef (the 2c0774d of 2026-10-22
  refuses the S2/S1 trainer options).** What the launcher and the takeover did on 2026-10-22/24, with the failures and fixes: `docs/EXPERIMENTS.md` A10.
- **Records of these sessions:** `docs/EXPERIMENTS.md` D6-D13, A10 and the tables in parts F-H; `docs/RESEARCH.md` sections 8, 10, 12;
  `docs/m3-hybrid/path-forward.md` section 0; the older state paragraphs moved to `docs/history/STATUS-history.md` ("Moved from STATUS on 2026-10-23").

### Handoff (2026-10-24, after the first S1 results)
Done and documented: O_2 adopted; S1 teacher labels (250,000 decisions) and test set stored; S1 trained at 250,000 decisions (D13, D14). Open, in order:
1. **Judging S1: done on copper** (2026-10-24, D14; `docs/m3-hybrid/s1-copper-check.json`). The bronze set never finished (one complete decision; owner), so there is no
   trick-1 yardstick against exact play; copper and the S1 test set measure how well S1 copies its teacher. Report any result as an estimate (owner, 2026-10-16).
2. **S1 is ready to wire into the app (another session does it):** candidate `97df2a54e1c368a13d69ffd887de62f45c6c3f244605450579b803f31f00e134` (decision-aware,
   seed 2, 250,000 decisions; copper 0.0386%, D14), 7,349,312 bytes in R2 `inputs/sha256/<key>`. The engine now takes one student per trick: host the new
   `crates/wasm/dist/engine.wasm` (`8366f7d2...`) and `engine.mjs` (`d0d8ab56...`), then `engine.loadStudent(s1Bytes, s1Sha, 1)` and
   `engine.loadStudent(s2Bytes, s2Sha, 2)` (S2 `3d1189e3...`); `method: "student"` then serves both tricks. Export check passed with both loaded (D14). The
   app side (merge of the fivefingers branch `engine41-bots`, student switched on for tricks 1-2, a real-browser check with `?bots=engine41`) is the owner's;
   the app's student config today points at S2 for trick 2 only. A network trained on more labels (item 3) replaces S1 by its hash.
3. **More data for S1:** 500,000 decisions are labelled and complete (`o2-copper-s1-1` 250,000 and `o2-s1more-3` 250,000, 64 deals; D13). A third batch (run name
   `o2-s1more3-3`, seeds 7831/7832) was started on the Thailand box at 02:01 UTC 2026-10-25 and ended at about 05:08 UTC when the owner stopped that
   instance from the console to free the box for the review's playout labels (A11): **67 chunks of `s1rules3` (67,000 rules-v1 decisions at 64 deals) are stored**
   under `runs/o2-s1more3-3/out/s1rules3/`, with no merged report (to train on them list the bundles with `bundles`, as the S2 manifests do; the run's
   other chunks were not made). Another 250,000 take about 11 hours and $4 on a 245-thread 7B12 box (22,700 decisions an hour, D13); do not take that
   box over while another run is on it. To retrain S1 on more data: `student-t1-250k.manifest.json` copied: train
   inputs `o2-copper-s1-1` (`s1rules` 300, `s1mixed` 200), `o2-s1more-3` (`s1rules2` 150, `s1mixed2` 100) and optionally the 67 partial chunks (`group` 0 / 1 / 2),
   points at 500,000 (and 567,000) decisions (decision-aware loss, seeds 1-3), `fetch_jobs` 32, binaries newer than `f08888c` (the inputs step reads in
   parallel; dispatch `release-binaries.yml`), a machine with one good GPU and 32-64 threads; judge on copper (`docs/m3-hybrid/s1-copper-check.json` shows how:
   `search student-check` on the copper shards, 0.7 s a network) and on the test set of `o2-s1test-2`. 32-deal labels would halve the teacher cost (D14, 0.0472 against 0.0404 on copper at 250,000).
4. **Rentals:** launch with `vast-run.yml` and `bin_sha` dd19e4d66cef. Taking over a hand-rented instance works (the S1 failures were the old binaries,
   not the takeover), but the cleanup job destroyed one taken-over machine before 2026-10-24 09:25 (fixed). Do not relabel or call a machine while the
   launcher rebuilds its container unless needed. Check `vast-status.yml` before renting.
5. **Record** each result in `docs/EXPERIMENTS.md`, this page, `docs/README.md` and `path-forward.md`.

**Session 0 (free checks) done, 2026-10-22, no rental (E9-E11, branch `claude/confident-albattani-sck9ru`):**
1. *Shortlist + search at 25-100 deals* (S2, O_3, exact golden set): top-3 + search beats S2 alone
   (0.0733%) from about 25 deals: 0.0596 / 0.0402 / 0.0293 at 25 / 50 / 100; worse at 10. Seat 1: 0.18
   to 0.097 / 0.073 / 0.061. A shortlist is **no more accurate than searching all cards** at those
   counts (top-2 helps at 25, hurts at 100): its value is cost only.
2. *Concentration:* the worst 5% of decisions hold 75% of S2's loss, 10% hold 91%, 73% of decisions lose
   nothing. But S2's margin does **not** find them (it is anti-informative: low margin = near-tie = small
   loss). A learned router on cheap features gets about half the gain for 30% of the searches; the oracle
   needs only 5-10% searched. Verdict: do not build the margin router; a risk model needs better
   signals (a second student's disagreement, a G2-style predictor) and a test.
3. *Root-restricted solve:* built through `exact_world` and the app API (`shortlist`, tested, values
   identical). Cost is **proportional to the cards kept** (nodes and network calls, 4-card trick-2
   decisions: 1 card 0.24, 2 cards 0.49, 3 cards 0.80 of the full solve). The first timing read
   0.64 / 0.70 only because it mixed in 3-candidate positions. The catch is that trick-2 decisions
   have only 2-4 candidates (mean 3.3), so top-3 drops at most one card (1.25x on the 54% of decisions
   with 4 cards) and top-2 about half (about 40% less work on average). This also explains result 1:
   top-3 barely differs from searching all cards; top-2 is the real shortlist at trick 2 (it wins at 10-25
   deals, loses at 100). Trick 1 (5-card hands) not measured.
   Consequence for the app: the leader needs 368 ms a deal in WebAssembly, so about 19 deals in 7 s on
   the VM (about 35 with top-2) and fewer on a phone, at or below the break-even exactly where S2
   loses most: the lever there is the cost of a deal (WebAssembly network kernel) and top-2.
   Next: session 1 (S2 A/B tests) is unaffected; the open question for the app is the cost per leader deal.

**Session 1 (S2 teacher A/B), 2026-10-22/23 (E12-E15, branch `claude/s2-teacher-ab-test-tiv5yj`):** built the
trainer's rank loss and the `student_train` point options; the G2 teacher run (`docs/vast/student-t2-gv2-250k.manifest.json`,
250,000 decisions, parts of 5,000) has stored 2 of 50 parts (R2 `runs/student-t2-gv2-250k-4/out/rules/000000-...`
and `runs/student-t2-gv2-250k-6/out/rules/000000-,000001-...`; use `skip_parts`/`bundles`). Two throughput bugs
found and fixed (ordered-writer window; a decision's 64 deals on one thread: now blocks of 8, byte-identical).
**Session 1 result (S2 teacher A/B), 2026-10-23 (E12-E14):** the **decision-aware loss is the win** (`rank_weight` 1, `rank_temperature`
0.01, `stop_on` decision): on the exact golden set the trick-2 student's loss halves, 0.0587% (3 seeds, SD 0.003) against 0.1195% for
plain at 250,000 decisions, and at 1,000,000 decisions it reaches 0.0456% alone over 3 seeds (0.0389 / 0.0491 / 0.0487, SD 0.006), 0.090% at seat 1 and 0.0037% for the top-3 pick
(the existing S2: 0.0733%, 0.179%, 0.0073%); seed 1 was the luckiest, all three beat the existing S2. Its integer network is in R2 (key in E12). **G2 labels were no help at 50,000 decisions (E13)**
(plain loss -0.006 +/- 0.017, decision-aware loss +0.024 +/- 0.009 worse, 3 seeds); the G2 labelling was cancelled at 60,000 decisions
(12 parts, $3.49; 28 minutes a part after the two throughput fixes), and S2 goes on with the plain H_3 teacher (owner, 2026-10-23).
**To check on S1 (trick 1) later:** the same loss, plain against decision-aware at 250,000 decisions, 3 seeds.
**S2, state (2026-10-23, end of the S2 session):** done: the decision-aware loss (E12, E17), G2 labels (no help, E13-E14), the app-path checks (E16), 1M seeds 1-3 (mean 0.0456%), the export check (native = WASM), and the ship record (the new S2 `3d1189e3...` in WASM-API §7). **Open:** (a) the app side: merge the fivefingers branch `engine41-bots` and switch the student on for trick 2 (owner; branch `claude/dreamy-franklin-irw8kk`, d0ca40f, holds the one-line swap; a real-browser check with `?bots=engine41` first); (b) optional: a 2M-decision point with the new loss (the plain 64-deal teacher costs about $4-5 per million decisions); (c) S1 with the same loss, after O_2 (see above); (d) [done 2026-10-23] machine 57346368 was destroyed by the owner.

**Records:** `docs/EXPERIMENTS.md` (every experiment and its result), `docs/RESEARCH.md` (the outside
research), `docs/README.md` (index).

**Built and merged into `main` (the single current line; old branches are `archive/*` tags):**
`search risk-profile`, `search labels --harvest-risk`, `search risk-filter`, `search net-error`,
the `s:NET` expert in `search::exact_leaves`; runner kinds `hybrid_rung` with `round`
(`crates/runner/src/hybrid/round.rs`), `specialist`, `net_error`; `fortyone_nn.labelcmp`;
`launch.py round-manifest` and `--instance-id`; workflows `o3-round.yml` (now with
`min_reliability`), `store-input.yml`, `r2-peek.yml` (now with `show`), `repo-archive.yml`,
`r2-inventory.yml`. Latest released binaries: `dd19e4d66cefbd60eb0f0b662e94f828d5c189a0` (2026-10-23 06:50; the older `2c0774db...` of 2026-10-22 06:48 does NOT know the trainer's `rank_weight` / `rank_temperature` / `stop_on` point options and refuses such manifests: see EXPERIMENTS A10, "Nine failed S1 launches").

### The goal and the chain (owner)
One objective: the chance of the money. Target: about 0.01% decision loss even at the hardest trick-1
situations, with every layer below it. The chain, top down:
1. **Trick 3-5 decisions:** exact search (cheap). Done.
2. **O_3** (trick-3 network) + **G v2** (a learned predictor of O_3's error picks the trick-3 positions to
   solve exactly): trick-2 decisions at 0.0008% (seat 1 0.0022%) on the exact golden set. Done, adopted.
3. **O_3 alone close to O_3 + G2** (2026-10-21, now): O3G2 rounds (`docs/m3-hybrid/o3-rounds.md`),
   because O_2's training labels and its test set (the bronze set) use O_3 alone.
4. **O_2** (trick-2 network, used by trick-1 search): adopted 2026-10-23, trained on 39.6M hybrid labels
   (exact through trick 2, then O_3; plain labels won the label test at equal money, D6); round 1 of the audit loop (D11).
5. **Teachers** (slow, offline): trick 2 = PIMC over 64+ deals with H_3 + G v2; trick 1 = PIMC over 64
   deals (128+ on close calls) with H_2 + O_2 (+ O_2 G v2).
6. **Students** (fast networks for the app): learn from the teachers; in the app they shortlist their
   top 3 cards and a small search picks among them.
The owner wants the teachers and their deal counts decided last.

### Stored networks, models and data (R2 `inputs/sha256/<hex>`)
| what | key | notes |
|---|---|---|
| O_3 adopted (2048-1024-256) | f5766aa98a1d769e31122c94cc4da9790c84508fab5c432d0c0358fe3124faad | trained on 40M trick-3 positions |
| O_3 leaf-risk model (G v2) | 20c41678c64681855003e5b8660e0a2e1fb9e8d552f7d898dd956055c767a9d5 | `data/leafrisk/leafrisk-o3-v1.json`; threshold t=277338260 adopted |
| **O_2 adopted (2048-1024-256, 39.6M plain hybrid labels, 2026-10-23)** | integer 13152c24bc27 (sha256 13152c24bc27b5f8...; in R2 as `inputs/sha256/13152c24...`), float and report under `runs/o2-audit-10/partial/rung/000000/out/hybrid/net/` | error SD 2.60%, bias 0.019%, sibling regret 0.188% on the 250k exact test set; replaces the provisional one below (D11) |
| O_2 provisional (2048-1024-256) | 505a0c7c65147ffa9565e06059f47664ea9f1bfdf0c5ea2f8d392b0e6e1337b0 | 1.5M exact labels; position error SD 3.6% (O_3: 1.85%) |
| trick-2 golden set: decisions / worlds | 635e9aef196c816380f3f15c05ede90b38b875fa4817061a57e6d39550065ab0 / 6ddc59b191e1fd534f260f9dfe103f742d6026768ca74acdbbb4fd430eb8265e | 2,000 decisions x 320 exact worlds, seed 1 |
| exact trick-2 labels, random 1.2M (s511) | shard fb51c217b5406e4b4ba1454f14b8884bced45cff51a0e29bcca7f4440f6f0931, sidecar bc4165ed9408f83ff342ab6fa157992c5910654513e417a9271a700fbbb77117 | O_2's training data |
| exact trick-2 labels, rules-v1 0.3M (s512) | shard 2160fefc6fcfd07822292f330f140a0618d25e0b89fb4ff8ed13d62e4559f1bd, sidecar d052d1a172fcf1b09d6e1cd4e95e1a2814b60e197864e1d5b18e7ab692f265f2 | |
| exact trick-2 validation 50k (s601) | shard 38de48356209cbec0f9ce41ceba28e6d5df52428abc0c866d443f9eb57964327, sidecar b39dbd6bf8f00a9b71fa800ccdaf9dd7de0ef25bc5dd3fa7b8b9e718fe895df8 | |
| exact trick-2 validation 20k (s602) | shard 855ba467ee6bd7521a442cf668980af8cefacf00701939237895d7c749dfdeea, sidecar 6e346cbf05b9f2cc0d4368e224e4dd81a11a7c2aaab1fd5af0c52a8ad177c86d | made 2026-10-20 |
| exact trick-2 validation 5k (s603) | shard 46ba92fcdf8e… (see `runs/o2-gv2-labels-3/partial/.../shards.json`) | made 2026-10-20 |
| **trick-1 student S1, shipping candidate 2026-10-24** (512-256, 250,000 decisions, decision-aware loss, seed 2) | 97df2a54e1c368a13d69ffd887de62f45c6c3f244605450579b803f31f00e134 | copper loss 0.0386% (3-seed mean 0.0416%; plain 0.0766%), S1 test set 0.0445%; copies its teacher H_2 + O_2, not judged against exact play (no trick-1 golden set; the bronze set never finished); 7,349,312 bytes in R2 `inputs/sha256/<key>`; export check passed with S2 beside it; load with `loadStudent(bytes, sha, 1)` (D14, WASM-API §7); seeds 1 and 3 `6ec1b9cc38b4...`, `b20445c31bde...` |
| **trick-2 student S2, shipped 2026-10-23** (512-256, 1M decisions, decision-aware loss, seed 1) | 3d1189e3042dfc9537b1f575f87e42b15e91628541e2ff90678ba319959c6e03 | golden-set loss 0.0389% (3-seed mean 0.0456%); export check passed; `inputs/sha256/<sha>` in R2, 7,306,112 bytes; the app swaps by its manifest (WASM-API §7); app side: manifest key changed on branch `claude/dreamy-franklin-irw8kk` (fivefingers, d0ca40f, no PR), checked by loading, 875 decisions and 4 whole matches; **not live**: the engine is not on the app's `main`, its bots are off by default and its trick 2 uses search |
| previous S2 (512-256, 1M lessons, plain H_3 teacher) | 954223e2f56ecade906ba28ad2ea07ff70fee70b841e0743d4c911217afb6ce6 | replaced; 0.0733% |
| equity table v0 | sha256 64740dab7845f0502c93ce195d721e1ffb8aa6ab141739ef43f754690de16808 | rebuilt by every run from `table: {hands: 1000000, seed: 1}` |
| O3G2 round records | `rounds/<id>/round.json`, `report.txt`, `profile.json` | written by each round (docs/m3-hybrid/o3-rounds.md) |
| round 1 result (`o3r-1`) | init_net f5766aa9 (unchanged), G2 20c41678 (unchanged), dataset 8336ccf3ce45dba40fae05319cbb3698d128d0509cafa0add799bf70e3175daa | winner: the incumbent |
| round 2 winner c2 (`o3r-2`, not adopted as O_3) | init_net 1057e9e949f04f09e59e4ef052b564694f1cc2140fbbd81a146188e46342d593, G2 5f7e6792a70b4ecf724896edea1a7d19fd8b80e592e6d623a5f450b54829ffe6, dataset ea682abeeaab4ca3c60123e414e825a30f07411439e9802be79073d0b7b2b0d1 | 0.0070% against O_3's 0.0068%: noise |
| specialist S int / float | d30ef4963658bc3fab186b71eeb69e766a38780d05637d3bda74d7e7076d590c / f5f6768da729fffd55116b4597be7203cb90de3444272f3ca7c0d10908c1055a | `docs/m3-hybrid/specialist.md`; the int file is in `inputs/sha256/` (store-input.yml) |
| specialist run output | `runs/specialist-s2/` (record.json, networks) | the flagged shards were not stored (about 23 GB) |

### How to run things (GitHub Actions, no vast key on the VM)
- **Which machine:** measured cost per unit of work of every machine we have rented, by kind of work, is the first section of `docs/VAST-MACHINES.md` (updated 2026-10-22).
- **Launch:** workflow `vast-run.yml` on `main` with inputs: `manifest` (the JSON inline, one
  line), `min_cores`, `min_ram_gb`, `disk_gb`, `cpu_only` ("true" for CPU jobs), `gpus`, `min_gpu_ram_gb`,
  `max_dollars` (the cap; the launcher stops and destroys at it), `any_host` ("true"), optionally
  `offer_id` (an owner-found offer: the `Type #` number) or `cpu_match` (e.g. "7B12"), `low_cpu_pct`
  ("1" for jobs with GPU-training or other single-thread phases, else the idle watchdog kills them),
  `bin_sha` (a commit with released binaries: **dd19e4d66cefbd60eb0f0b662e94f828d5c189a0** is the
  latest; dispatch `release-binaries.yml` on `main` for a newer commit and wait for
  `bin/<sha>/manifest.json` in R2), `image` **ghcr.io/guillaumeds/41bots-base:20261017-bc692e446939**,
  `dry_run` **"false"** (the default is true!). A Blackwell GPU (RTX 50xx) needs `"blackwell": true` in
  the manifest's `machine`.
- **Watch:** `python/launcher/launch.py` helpers (`latest_heartbeat(run)`, `r2_list`, `r2_get`) with
  `R2_S3_ENDPOINT=$R2_ENDPOINT`; heartbeats every 30 s carry the step, rows done and dollars.
- **Stop:** cancel the workflow run (it asks the run to stop, waits up to 5 minutes for uploads, then
  destroys the machine) or `vast-stop.yml` (run_id, instance_id); `vast-status.yml` lists every
  instance on the account. Confirm nothing is left billing.
- **From a laptop (no R2 or vast keys there):** everything goes through GitHub Actions with the `gh` CLI:
  `gh workflow run vast-run.yml --ref main -f manifest="$(jq -c . docs/vast/NAME.manifest.json)"
  -f min_cores=190 -f min_ram_gb=64 -f disk_gb=60 -f cpu_only=true -f max_dollars=3 -f any_host=true
  -f bin_sha=dd19e4d66cefbd60eb0f0b662e94f828d5c189a0 -f image=ghcr.io/guillaumeds/41bots-base:20261017-bc692e446939
  -f dry_run=false` (GPU jobs: `-f cpu_only=false -f gpus=1 -f min_gpu_ram_gb=8 -f low_cpu_pct=1`);
  `gh run list --workflow vast-run.yml`, `gh run view <id> --log` (the launcher's log shows the machine,
  price and progress lines), `gh run cancel <id>` (stop and destroy); `gh workflow run vast-status.yml`
  (every instance on the account); `gh workflow run r2-fetch.yml -f run_id=<run>` then
  `gh run download <id>` for a run's files (results bundles `out/*.tar.zst`, unpack with Python 3.14
  `compression.zstd`); `gh workflow run release-binaries.yml --ref main` for new binaries.
- **With keys on the laptop (`.env`, git-ignored; names in `.env.example`):** install the `aws` CLI
  and `vastai` (pip), load the file (`set -a; source .env; set +a`), then from `python/launcher`:
  `uv run python -c "import launch; print(launch.latest_heartbeat('RUN'))"`, `launch.r2_list('runs/RUN/')`,
  `launch.r2_get(key)`, `python launch.py put-input FILE` (store a model or shard content-addressed),
  `vastai show instances`. Prefer launching through the `vast-run.yml` workflow even then: its
  launcher watches and enforces the dollar cap for up to 6 hours without the laptop staying awake (a
  run outliving it is finished by its own self-destroy and the hourly `vast-cleanup`).
- **Test a manifest before launching:** add it to `v2::tests::manifests_resolve_into_settings_their_chunks_accept`
  (`crates/runner/src/v2.rs`; kinds hybrid_rung, student_teacher, hybrid_checks, leafrisk) and run
  `cargo test --release -p runner manifests_resolve`.
- **Pitfalls met:** `hybrid_rung` with `check: false` refuses `tt_mb`; `label_inputs` cannot be combined
  with `label_parts` above 1; vast offer numbers change after a machine is released (use `cpu_match`);
  a label part is stored only when complete (size parts to about an hour); the vast balance hitting zero
  stops every machine at once.
- **Peek at a running round without SSH or R2 keys:** `r2-peek.yml` (run_id, e.g. `o3r-1-1`; optional
  `show`: small files to print, e.g. `report.json DONE FAILED`) prints the run's files, its latest
  heartbeat (step, rows, dollars) and those files in the job log. A run's big results are inside
  `out/<stream>/*.tar.zst`: `r2-fetch.yml`, then `gh run download <fetch run id> -R Guillaumeds/41Bots
  -n run-<run_id>` and unpack with Python `zstandard`.
- **Get stored inputs onto the VM:** `input-fetch.yml` (`keys`: SHA-256 keys, space-separated) copies
  `inputs/sha256/<hex>` into an artifact (hash-checked); `gh run download <run id> -n inputs -D <dir>`.
  The golden set, S2 and O_3 are small enough (1.2 MB, 92 MB, 7 MB, 17 MB).
- **Store a trained network as a manifest input:** `store-input.yml` (the r2-fetch run id, artifact
  name, the member path, the expected SHA-256) puts it in `inputs/sha256/<hex>`.
- **Deploy onto a kept machine, and keep a machine after a run (owner, 2026-10-22; built in session 1, `python/launcher/test_keep.py`).**
  Both are keys of the manifest's `machine` object, so the `vast-run.yml` call is the usual one (no new input):
  - `"keep": true` (and `"keep_hours"`, default 6, at most 72): when the run ends by itself (DONE or FAILED) the runner
    **parks** instead of destroying its instance (`KEEP_MACHINE=1`, `crates/runner` `end_machine`) and the launcher
    leaves it running, writes `keep/<instance>.json` (for the cleanup) and `runs/<run>/KEPT.json`, and prints a
    `::warning::MACHINE KEPT RUNNING, ACTION NEEDED ...` line and a job-summary banner. A **cancelled** workflow keeps
    it too (`stop --keep`). Not kept: a budget stop, a launcher-forced stop (low CPU, silence, time), a machine that
    died, and a boot failure (the image's boot script still destroys it: image change needed). The idle machine bills
    until it is taken over or destroyed; **`vast-cleanup` destroys it at `keep_until_utc`** (+5 min) and deletes the
    record of a vanished machine.
  - **Takeover status (2026-10-22):** the owner reports the O2 session's takeover of the hand-rented instance 57259029
    (the same physical machine as 57253570) **succeeded** (how it was invoked is not recorded here). My own trial
    `keep-trial-2`, a takeover of a *running* kept instance, **failed**: the recycled container booted the OLD environment,
    so the new run never heartbeated (and `o3r-1-1` on 2026-10-21 failed alike). **What is established:** a takeover of a hand-rented machine
    worked once (O2 session; whether it was stopped or running is not recorded), one of a launcher-created kept machine failed once, and starting a
    *stopped* instance (update then start) is tested only against the fake vast. The cause of the difference is not found. A taken-over machine that
    does not start is now kept, not destroyed. **Two sessions must not take over one machine** (it happened on 2026-10-23: the later takeover
    replaced the earlier's container): a takeover now writes `claims/<instance>.json` and refuses while another unfinished run's claim is under 45 minutes old
    (E15); to stop a run whose machine another run now owns, use `gh run cancel --force` (a normal cancel destroys the instance it recorded).
  - `"instance_id": N`: take that instance over instead of renting one (`launch.py` `take_instance`: `vastai update
    instance` with this run's image, environment and binaries, then `recycle instance`, which rebuilds the container in
    place; the disk and machine stay; the instance is relabelled `41bots-<run>` and its `keep/` record deleted). Use a
    **new run id** (a used one is refused). Needs a disk of at least `disk_gb` and the threads asked. **Refused (nothing
    changed):** another run's machine that heartbeated in the last 15 minutes and has no DONE/FAILED (cancel it with
    `keep` first, or wait), a machine another run started under an hour ago, a label that is not `41bots-...`; allowed:
    a kept machine, the machine of an ended run, an unlabelled one rented by hand.
  - **A machine that already runs our image cannot be taken over while running** (2026-10-25, twice: `keep-trial-2` on 2026-10-22 and
    `playout-labels-1b` at 09:23 UTC): `update instance` then `recycle` boots the old environment, the old run's runner finds its DONE and
    parks, and the launcher gives up after 15 minutes ("no heartbeat"); the machine is kept. **Stop it first** (`vast-instance.yml` with
    action `stop`, or the console), then launch with `instance_id`: a stopped instance is updated and started and boots from the new
    settings. A hand rental with another image can be taken over running (07:28 UTC worked). `vast-instance.yml` also relabels
    (action `label`, the owner's rule) and starts; it never destroys. **Correction, 10:40 UTC:** the stop-and-start (run
    `playout-labels-1c`, 10:19) booted the old environment too: `vastai update instance --env` never changed the container's
    RUN_ID (the old run's runner heartbeated under `playout-labels-1` at 10:22 and parked). The takeover now puts the whole
    environment on the on-start command (`env K=V … 41bots-boot`, `take_onstart`), which the boot script reads whatever the
    container's own environment says; if vast ignores on-start updates as well, the remaining route is to destroy the instance
    and rent the machine again (`machine_id`), which creates a fresh instance with the right environment. **What actually happened
    (14:30 UTC):** the 1c settings did take, late: after the 1d recycle at 10:44 the box booted **1c's** environment and run
    `playout-labels-1c` has run since (305 of 1,107 chunks by 14:30, CPU 84-88%, 15 chunks at once), while 1d never started.
    So vast applies a takeover's settings one update late and after a delay. **After any takeover attempt, read the heartbeats of
    every recent run id before concluding** (`vast-status.yml` with each `run_id`), and relabel a working box out of the cleanup's
    watch when no launcher owns it (done: `41bots - Review (playout labels 1c, running) - ETA 00:30`).
  - `"machine_id": M` (or `--machine-id M`): rent the cheapest rentable, unrented on-demand offer on vast machine M,
    read from the **public market list** (`POST console.vast.ai/api/v0/bundles/`, no key, no 512-offer cap), then
    checked exactly like a named offer. A named `offer_id` that both `vastai search offers id=N` and the launcher's
    market listing miss is looked up there too before it is called gone (2026-10-25, EXPERIMENTS A11: offer 36863856
    was live and rentable there while both searches said it was gone, and the owner had to rent the box by hand).
  - **A machine the owner stops from vast's console is never destroyed or replaced** (2026-10-25, EXPERIMENTS A11:
    the launcher read the stopped Thailand box as "died before DONE" and destroyed it). vast's `intended_status`
    tells a stop ("stopped") from a crash or an exit ("running"): the watch ends the run with the reason
    `stopped by the owner (vast intended_status stopped)`, records the machine as kept (`keep/<instance>.json`,
    `KEPT.json`, the banner) whether or not the manifest said `keep`, and rents no replacement; `vast-cleanup`
    leaves a stopped instance alone too (kept record or not, past its keep time or not; only its `all` option
    destroys it). Restart it with `instance_id` as before (update, then start).
  - **A kept run longer than the launcher's handover (330 min) keeps its machine too** (2026-10-25): the launcher is gone when such
    a run writes DONE, so no keep record existed and the cleanup's marker rule destroyed the machine 10 minutes after DONE (the
    playout labels run, about 20 hours, would have ended that way). Now `watch.json` carries `keep` and `keep_hours` (and the
    stored `runs/<run>/manifest.json`'s `machine.keep` counts for runs launched before), and at DONE/FAILED the cleanup keeps the
    machine, writes `keep/<instance>.json` and `KEPT.json` from the marker's time (action `kept`, a `::warning::` banner) and the
    keep time then runs as for any kept machine. **The cleanup workflow runs from `main`: this must be merged before such a run ends.**
  - **After a run with `keep`, the session must act** (deploy the next job with `instance_id`, or destroy it with
    `vast-stop.yml`): `monitor_run.py RUN` prints a banner when `runs/RUN/KEPT.json` exists and `monitor_run.py x --kept`
    lists every kept machine. Arm a `send_later` for the run's end when launching with `keep`.
- **Launch pitfalls met 2026-10-22 (each is a refusal before anything is rented, unless said):**
  a run id cannot be reused (a failed attempt leaves `runs/<id>/` in R2; give each attempt a new id);
  a manifest `machine` block takes only gpus, min_gpu_ram_gb, min_compute_cap, min_cores, min_ram_gb,
  disk_gb, min_inet_down, min_inet_up, max_dph, wait_for_offer_min, offer_poll_s, price_steps,
  tries_per_step, gpu_names, blackwell (and `max_dph` there caps a named offer: a stray $0.50 refused
  a $1.34 machine); the launcher never rents Xeon Phi chips; an offer id can vanish between listing
  and renting; an unknown runner kind is refused by the machine's runner only after the machine has
  booted (`specialist-s1` had loaded for 29 minutes, $0.68, before it said so) (register new kinds in `sched.rs` `KINDS`, `v2.rs`, `mem.rs`, `main.rs` and the manifest
  test); **a machine can list 384 threads and give the container 46** (every AMD EPYC 9B45 /
  RTX PRO 6000 offer tried on 2026-10-22: the launcher rents, sees `usable_cores`, stops it after
  about 5 minutes for about $0.12): prefer the AMD EPYC 9B14 server chip with an RTX 5090 (184 usable of 192 threads, 132 GB RAM, $1.34/h) that ran everything else.
- **Repository and storage housekeeping:** `repo-archive.yml` (old branches become `archive/<name>`
  tags; dry run by default), `r2-inventory.yml` (read-only list of the bucket).

## Where the history is
- `docs/README.md`: one line per experiment and document (what it was, the result, adopted or not).
- `docs/history/STATUS-history.md`: every dated status entry from 2026-10-13 to 2026-10-20.
- Archived branches: `git tag -l 'archive/*'`; `git checkout archive/<name>` restores any of them.
