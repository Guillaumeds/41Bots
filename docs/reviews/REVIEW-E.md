# Review E: run operations and money safety (2026-10-16)

Scope: `crates/runner` (main, v2, sched, measure, mem, net, remote, machine, stop), `crates/search` plan/measure (checkpoints), `deploy/docker/boot.sh`, `python/launcher/launch.py`, `cleanup.py`, `python/tools/machines.py`, workflows `vast-run`, `vast-cleanup`, `vast-status`, `release-binaries`. Worktree `/home/user/41Bots-rE`, branch `review-rE`, base `c7aa75c`. Line numbers marked "base" are in `c7aa75c`.

Commits on `review-rE` (not pushed):
- `c65a130` runner: unknown stream keys refused; concurrent chunks admitted only if their tables fit; network failures no longer end a run.
- `4c58c06` launcher/cleanup/workflow/boot: no time limit by default, $20 default cap, STOP cleared before a replacement, busy threshold follows `low_cpu_pct`, stream keys checked before renting, 429 during replacement handled, late-listed create adopted, vastai timeouts, offline-but-heartbeating kept, listing glitch does not close a run, boot writes FAILED on a refused manifest.
- `3e2dedc` launcher: `--offer-id` checked against `--min-ram-gb`/`--min-cores`; a destroyed run's machine record is marked final.

Checks run: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace` (53 suites ok), runner release integration tests `resume_identity`, `stop_partial`, `resume_worlds` with `--ignored`, Python `unittest` launcher 66 + tools 19, all pass. Every new Python test was also run against the base `launch.py`/`cleanup.py`: exactly the 9 new tests fail there (the others pass), so each test catches the bug it names.

**Deploy notes.** `cleanup.py` runs from `main`: its fixes act only once merged there. `boot.sh` is baked into the image: its fix needs an image rebuild. The runner fixes need new release binaries; **the p4 manifest names `bin_sha 7253b48`, which has none of them** (see question 3).

---

## Findings, by severity

### A. Money lost / work lost

**A1. p4 manifest: a stream-level `"threads": 32` is silently ignored. Every chunk runs on one thread.** (fixed `c65a130`, `4c58c06`)
- Evidence: `sched.rs:268` (base) reads only `params.threads` (default 1); unknown stream keys were not checked. The local smoke run of the same shape (`scratchpad/p4/remote/runs/p4-smoke/final-manifest.json`) records `"stream_threads": [{"tt12g": 1}, {"tt3g": 1}]`.
- Effect: one world at a time per stream (in `measure.rs` `spread`, `worlds_at_once` is ignored when `solve_threads` is 1), so 3 of about 190 threads work. The same work takes about 32 times as long. A plan of about $10 becomes several hundred dollars of machine time, and the $20 cap stops it with a small fraction done. The per-world timings would also be taken without the memory contention of 32 neighbours.
- Fix: the runner refuses any stream key other than name, priority, kind, params and chunks, and names `params.threads`. The launcher refuses the same before renting anything (`manifest_stream_errors`). Before this, a manifest the runner refused also cost up to 4 rentals: the runner exits with status 2 and writes no marker, and the launcher took the vanished machine for a start failure and rented again. `boot.sh` now writes FAILED on exit status 2 (needs an image rebuild).

**A2. Format-2 chunks run together whenever their *threads* fit. Each one sizes its tables as if it had the machine to itself, so they can exhaust memory together.** (fixed `c65a130`)
- Evidence: `mem.rs:27` (base): "The budget assumes the chunk has the machine's memory to itself". `v2.rs:940` (base) admits by threads only, and `pick` treats priority as an order, not a barrier.
- Effect: see question 3. With the obvious repair of A1 (`params.threads: 32`), the 12, 3 and 1 GB streams start together and need 32×(12+3+1) GiB = 512 GiB on a machine of about 466 GiB. The tables are only reserved up front, so they grow into memory, and the kernel's out-of-memory killer acts hours in. The run then ends FAILED.
- Fix: `mem::chunk_need` gives each chunk's peak memory: the tables its own plan will choose, plus its other memory; "auto" counts as the whole machine. `v2` starts a chunk beside running ones only within the budget (limit less margin), and lower priorities never jump the queue. Test `streams_whose_tables_do_not_fit_together_run_one_after_another` uses the p4 numbers: 385, 97 and 33 GiB. The 12 GB stream runs alone, then the 3 and 1 GB streams together.

**A3. A STOP left in R2 stops the replacement machine.** (fixed `4c58c06`)
- Evidence: `launch.py:902` (base). A format-2 machine that is silent but busy gets STOP, then is destroyed and replaced. Nothing deletes `runs/<run>/STOP`, and the new runner polls it at its first heartbeat.
- Effect: the replacement stops at once, writes its marker, and the run ends with its work undone (and the money for the replacement's boot is wasted). On the base code the test `test_stop_of_a_stopped_machine_does_not_stop_its_replacement` ends `FAILED`; with the fix it ends `DONE`.
- Fix: `clear_stop` runs before every replacement.

**A4. A silent machine of a low-CPU run is destroyed at once, which repeats the $18 loss.** (fixed `4c58c06`; cleanup half needs `main`)
- Evidence: `launch.py:1225`, `cleanup.py:117` (base). "Busy" meant vast CPU of at least 50%. The p4 run keeps 32 of 190 threads busy, about 17%.
- Effect: when its heartbeat uploads hang, both the launcher (after 3 minutes) and the cleanup (after 20) classed it idle and destroyed it without a STOP. Everything since the last partial upload was lost.
- Fix: the busy threshold is `min(50, low_cpu_pct)`. It is written into `watch.json` (`busy_cpu_pct`), and the cleanup reads it from there. Tests are on both sides.

**A5. Time-based kills remained, against the owner's rule. There was also no default cap.** (fixed `4c58c06`)
- Evidence:
  - `--max-minutes` defaulted to 30 (`launch.py:1365`), and `vast-run.yml:30` called it a "Hard time limit".
  - It also wrote a `deadline_utc` into the manifest (`launch.py:779`), which the runner stops at. The cleanup destroys 30 minutes after that deadline.
  - `--max-dollars` defaulted to 0, meaning no cap (`launch.py:1377`, `vast-run.yml:46`).
- Effect: a golden run started with the defaults would be stopped at about 28 minutes.
- Fix: there is no time limit and no deadline unless `max_minutes > 0` is given on purpose. The default cap is $20 in both the launcher and the workflow. Test: a 50-minute run finishes, and a run that never finishes stops at $20.
- Remaining limits, which are liveness checks rather than age limits:
  - the launcher: startup 15 min with no heartbeat ever; silent 3 + 15 min while busy; the stop-upload wait of 5 min;
  - the cleanup: never heartbeated 45 min; silent 20 min; 15 min after a STOP; a deadline only if one was set;
  - GitHub: the 360-minute job limit, handled by the hand-over at 330 minutes.

**A6. Format 2: one failed done-list write dropped every running chunk and ended the run FAILED.** (fixed `c65a130`)
- Evidence: `v2.rs:1037`/`899` (base) `self.write_done()?` sits inside the scheduler loop. The error returns at once, the `Child` handles are dropped, the workers see stdin close and exit, and the run writes FAILED.
- Effect: about a minute of network failure (3 tries 2 s apart) lost all unsaved work in every running chunk. The FAILED marker also means no replacement, and the run id cannot be relaunched (B2).
- Fix: the periodic write logs the failure and retries at the next interval. The same `?`-without-retry pattern on the start-up listings (`v2.rs:533`, resume) now retries. A finished chunk's bundle upload and check use `retry_patient(8)`, which waits about 3 minutes.

**A7. Format 1: failed final uploads, then the machine destroys itself anyway.** (fixed `c65a130`)
- Evidence: `main.rs:1043` (base).
- Effect: a one-minute network failure at the end of a long run lost `out/` and the work since the last partial upload (up to 300 s).
- Fix: up to 5 rounds, 60 s apart, with the heartbeat running, before the self-destroy. At $1/h that costs at most about $0.08.

**A8. A create error whose machine shows up late was rented twice.** (fixed `4c58c06`)
- Evidence: `adopt_instance` (base `launch.py:1101`) concluded "nothing rented" from one readable listing.
- Effect: if vast lists the new machine a few seconds late, a second offer is rented with the same label. Both run the same `run_id`, and the cleanup counts one machine's heartbeats for the other. On the base code the test `test_create_error_whose_machine_appears_late_is_adopted_not_rented_twice` rents offers [1, 2].
- Fix: two readable listings 10 s apart are needed before concluding.

**A9. Cleanup destroyed "offline" machines that were still working, and closed handed-over runs on an incomplete listing.** (fixed `4c58c06`, needs `main`)
- Evidence: `cleanup.py:97`, `:350` (base).
- Effect: offline means vast has lost touch with the host, not that the container died. A machine that still heartbeats to R2 is working. Separately, an instance listing that came back missing the machine closed the run: it deleted its destroy key and marked its machine record KILLED while it worked on.
- Fix: offline with a heartbeat in the last 20 minutes goes through the normal rules, and its cap still applies. Closing a run needs a reading of the instance itself that does not show it running.

### B. Work that cannot be continued (reported, not fixed: design changes)

**B1. After the hand-over at 330 minutes, nobody rents a replacement.**
- If the golden run's machine dies at hour 7, the cleanup only closes the run. The finished worlds are safe in R2 (`partial/`, `done/`), but continuing needs a person.

**B2. A format-2 run cannot be continued under its own run id.**
- `launch.py:767` (base) refuses any `run_id` already in R2. Format 2 resumes only through the launcher's in-job replacement. It has no `resume_from`.
- So a format-2 run that ends FAILED, or is lost after the hand-over, cannot resume. Its chunks would be redone under a new id; within a chunk, only the bundles and partials can be read.
- **Recommendation:** run the golden run as format 1 (`search_measure` with `resume_from` works across run ids), or add a launcher "continue run" mode. Such a mode must clear STOP and the old markers and keep the run id.

**B3. A chunk failure in format 2 (out-of-memory kill, a panic in a solver thread) ends the whole run FAILED.**
- The other chunks in flight finish; no new chunk starts, and there is no replacement.
- The failed chunk's live files (checkpoints) are in R2, but because of B2 nothing continues them.

### C. Wrong or missing numbers

**C1. Format-2 runs write no machine-cost record and no cost per node.**
- Chunk workers run with `net: None` (`main.rs:486` base). Their tally is written only inside the chunk directory, which is deleted when the chunk ends. The orchestrator's heartbeat has no `nodes_total` or `usd_per_1e12_nodes`.
- STATUS lists this as "Not covered". The p4 measurement is format 2, so `machines/<run>.json` will not exist. `machines.py --live` will not show it, and `--by-cpu`/`--rank` gain nothing.
- If handed over, the cleanup writes a record with `nodes_total 0` and a null cost. The per-world nodes and times are still in the worlds files' checkpoint and timing lines.
- **Recommendation:** the orchestrator sums the chunks' heartbeat tallies (`nodes_total`, `worlds_finished`), keeps the totals of finished chunks, and writes the record as format 1 does.

**C2. A record whose run was destroyed without a marker stayed `state: running` for ever.**
- This applies to launcher-watched runs; the cleanup only finalised handed-over ones. (fixed `3e2dedc`: the launcher finalises with the cleanup's rule)

**C3. A format-2 run stopped by its cap or by a person writes DONE, not FAILED.**
- `v2.rs:448` (base): `go()` returns Ok after keeping partials, so the marker is DONE with `stopped`.
- The launcher's report says STOPPED, because it knows it asked. The cleanup's `closed.json` and the finalised machine record say `DONE` for an incomplete run (`merged.json` has `complete: false`).
- **Recommendation:** read `stopped` from the marker body.

**C4. Dollar figures differ, and none matches vast's bill exactly.**
- The machine record's `usd` is the runner's elapsed time × `DPH`. It leaves out image pull and boot (2–5 min per machine) and the self-destroy tail, so the record understates $/node, most on short runs.
- The launcher's `dollars_est` runs from create to the verified destroy (up to about 2 extra minutes). The cleanup's `dollars_est_upper` is an upper bound.
- None counts vast storage (20 GB disk, negligible) or network charges. The image pull is billed at up to $0.10/GB under our filter.
- The cleanup's finalised record in format 2 uses the last machine's `est_cost_usd` only, not earlier replacements.
- The cap is per run (all machines of the run), which is stricter than "per machine".

**C5. In format 2 the low-CPU watchdog is off whenever the streams leave threads free.**
- `v2.rs:960` (base): `partial_ok = used < threads`.
- For p4 (3, or 96, of 190 threads) the launcher never checks CPU. So `low_cpu_pct=1` is harmless, but a chunk hung at 0% CPU while the orchestrator heartbeats is caught only by the $20 cap. After the hand-over the cleanup has no CPU rule at all. With A5's default cap, that is now the bound.

### D. Waste / small

- **D1.** No timeout on `vastai` calls. A hung call stopped the launcher's polls, so there was no hand-over before GitHub's 360-minute kill. Fixed: 180 s timeout, except `create`. `aws` calls still have none of ours (the CLI has its own connect/read timeouts).
- **D2.** A 429 (daily search quota) while renting a replacement raised out of `rent_and_watch`, so there was no report and no summary. Fixed: the run ends with "no offer could be rented" and the report written. A 429 on the first search exits before anything is rented (safe).
- **D3.** `--offer-id` skipped `--min-ram-gb`/`--min-cores`. Fixed: a 480 GB job could no longer be rented on a 251 GB offer. Units: raw `cpu_ram` is in MB, as `vast_market.py` uses it.
- **D4.** Two dispatches with the same `run_id` can both pass the "already exists" check: the check and the manifest upload are not atomic. That would mean two machines on one run. Low probability (a person double-clicking).
- **D5.** Instances are always labelled in the create call itself. The cleanup ignores unlabelled instances by design, so a label lost by vast would bill unnoticed. Low.
- **D6.** One run whose R2 listing fails aborts the whole cleanup sweep (no cap enforcement anywhere that round). This fails safe for destruction.
- **D7.** GitHub cron can run late under load, and the cleanup's cap enforcement after the hand-over is ≤15 min + cron lag + 15 min STOP grace. At $1/h the overshoot is about $0.5, more on dearer machines.
- **D8.** Messages in `main.rs`/`boot.sh` still cited the removed 2-hour cleanup rule. Fixed.

---

## End-to-end trace: step × failure → outcome (after the fixes; "base:" where the fixes changed it)

| Step | Failure | Outcome |
|---|---|---|
| Rent (search) | vast error / 429 | First search: launcher exits, nothing rented. Replacement search: run ends "no offer could be rented", report written (base: exception, no report). |
| Rent (create) | offer taken / error / unreadable answer | Lists instances; adopts a `41bots-<run>` instance; "nothing rented" only after 2 readable listings 10 s apart (base: 1). Listing unreadable 3 times: run stops (no second rental). Next offer, up to 5. |
| Rent | hung `vastai` | 180 s timeout → failed call → "unknown" (base: launcher hung). `create` has no timeout. |
| Rent | two dispatches, same run_id | Both may rent (D4). |
| boot.sh | binary or manifest download fails / sha mismatch | `die`: FAILED marker + self-destroy; the launcher sees FAILED → run ends (format 2 too, no replacement). |
| boot.sh | rclone hangs | rclone's own timeouts (5 retries); no heartbeat within 15 min → launcher destroys; format 2 is replaced. |
| boot.sh | container cannot start (OCI error) | Seen at the first poll; destroyed; next offer (counted as a replacement, format 1 too). |
| Manifest | refused by the runner (exit 2) | Base: no marker → taken for a start failure → up to 4 rentals. Now: the launcher refuses bad stream keys before renting; boot writes FAILED on exit 2 (image rebuild). |
| Chunks / streams | stream-level `threads` | Base: ignored, 1 thread per chunk (A1). Now refused. |
| Chunks / streams | tables of concurrent chunks exceed RAM | Base: started together → OOM hours in (A2). Now admitted by memory; waits in priority order. |
| Chunk | OOM / panic in a worker thread | The `search` process dies → chunk failed → no new chunks; in-flight chunks finish; FAILED; no replacement (B3); checkpoints in `partial/`. Format 1: FAILED with out/ and checkpoints uploaded; resume with `resume_from`. |
| Chunk | full disk | Outputs or tmp copies fail → chunk fails or heartbeats stop → STOP → destroy; work since the last partial upload lost. 20 GB disk is ample for p4. |
| Checkpoints | every world | Appended to `<worlds>.ckpt`; uploaded first in every partial upload (every 300 s) and on stop. |
| Uploads | transient network error | Heartbeat/STOP/partials on background lanes (45 s + 1 s/MiB kill). Done-list write retried next interval (base: run ended, A6). Bundle upload ~3 min patient retry. Format-1 final uploads 5 rounds (base: self-destroy, A7). |
| Uploads | rclone hang | Killed at its limit; heartbeats never blocked (`hung_network_never_blocks`). |
| Heartbeats | uploads stuck, machine busy | Launcher: 3 min + 15 min grace, then STOP, 5 min wait, destroy; format 2 replaced, STOP cleared first (base: replacement stopped, A3). Cleanup: 20 min → STOP → +15 min destroy. Busy = vast CPU ≥ min(50, low_cpu_pct) (base: 50, A4). |
| Heartbeats | R2 unreachable from the launcher | Counted as no new heartbeat; same path as above. |
| Heartbeats | R2 unreachable from cleanup | The sweep raises; nothing destroyed, nor capped, that round. |
| Machine | reboot | Container restarts, boot runs again, the runner resumes from local files (format 1 from its own out/, format 2 from live files); gap < 3 min usually; vast state "loading" counts as unknown (grace). |
| Machine | vast shows offline | Launcher: gone only after 3 readings over 90 s. Cleanup: kept while heartbeating (base: destroyed, A9). |
| Launcher | job cancelled | KeyboardInterrupt → STOP written; the workflow's cancelled step waits up to 5 min for the marker, destroys, deletes the key. |
| Launcher | hand-over at 330 min | `watch/<run>.json` written, safety-net files removed; cleanup every 15 min. No replacements after this (B1). |
| Resume | stale checkpoint (other values version / fingerprint) | `search worlds` refuses by name → FAILED within minutes (about $0.1). The b3/a7 checkpoints from older binaries are unusable by design. |
| Cap | fires mid-world | Launcher (≤1 poll): STOP → the runner stops the worlds in flight (those are lost), uploads checkpoints and partials → marker → destroy. Format 2 marks it DONE+stopped (C3). Cleanup after the hand-over: ≤15 min + cron lag, then STOP; destroyed 15 min later if alive. |
| DONE → result.json | normal | Format 1: checkpoints, records, out/, marker last. Format 2: bundles verified by read-back, `merged.json`, `final-manifest.json`, marker. |
| Machine record | normal / cap / cleanup | Format 1: running every 10 min, final on DONE/FAILED/STOPPED. Destroyed without a marker: finalised by the launcher (now) or the cleanup (after the hand-over). Format 2: none (C1). |
| Self-destroy | key refused / vastai hangs | The launcher destroys after DONE (90 s grace, verified); after the hand-over the cleanup destroys 10 min after the marker. A hung `vastai` in the runner's self-destroy has no timeout, but the marker is already up. |
| Destroy verification | status unreadable | "unverified" in the log, the report and the summary; the cleanup checks again every 15 min. |

---

## Question 3: the p4 manifest (three streams × 32 one-thread worlds at 12 / 3 / 1 GB tables)

**As written, with the named binaries `7253b48`:**
1. `parse_v2` ignores the stream-level `"threads": 32`, so each stream gets `threads = 1` (base `sched.rs:268`). The smoke run's `final-manifest.json` confirms `stream_threads` 1.
2. `schedule` (`v2.rs`): with nothing running, `free = MAX`, so it picks `tt12g`. Then free = 189 → `tt3g`; free = 188 → `tt1g`. **All three start together.** Priority orders the picks; it does not make streams wait for each other.
3. Each worker runs `Ctx::local(threads = 1)` and `measure::spread`. That gives `Fixed { workers: 1 }`, because with `solve_threads` 1, `worlds_at_once` is ignored. So each stream is one `search worlds --threads 1` with one table of 12, 3 or 1 GB: about 16 GB in all, so no OOM.
4. Each stream's 32 worlds run **one after another on one thread**: 3 of about 190 threads busy, about 32 times the planned wall time. The $20 cap would stop it early.
5. The launcher's CPU rule is skipped (`partial_ok = used < threads`), so there is no low-CPU stop even at 1.6% CPU.

**With `params.threads: 32` but the old binaries (also `7253b48`):**
- All three start together (96 ≤ 190 threads). Each plans its tables against the whole machine's budget (about 418 GiB on 500 GB), so none is lowered.
- 384 + 96 + 32 GiB = 512 GiB of tables against about 466 GiB of RAM. Tables are reserved up front but touched as they fill, so the run starts normally. The kernel then kills the largest process (tt12g's `search`) once the trick-1 leader worlds fill their tables.
- tt12g fails, tt3g and tt1g finish, and the run ends FAILED with no replacement (B2/B3). If the orchestrator is the one killed instead, the launcher rents replacements that repeat the same OOM, up to 3 times.

**After the fixes (needs binaries from `c65a130` or later):**
- The manifest as written is refused before renting (launcher) and by the runner.
- With `params.threads: 32`, memory admission runs `tt12g` alone (385 GiB), then `tt3g` + `tt1g` together (130 GiB).
- Measurement caveat for the owner: tt12g's worlds are then timed with 32 busy neighbours and the other two streams' worlds with 64. If the comparison must be under equal load, run the streams as separate runs (or one stream per run). Equal load is not production load either (all threads busy).

## Integration tests (release)

With release binaries built from `3e2dedc`, all three `--ignored` integration tests pass:
- `resume_identity`: kill points after_done:2, mid_upload:4, after_upload:5, after_done:5; byte-identical result.
- `stop_partial`: format 1 SIGTERM, and a format 2 stop request.
- `resume_worlds`: kill -9, then resume world by world in format 1 and format 2, 79 s.
