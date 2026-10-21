# Status

**Current checkpoint: M3 (in progress)** (PLAN §13, as revised on 2026-10-13): PIMC bots, the exact early-trick yardstick and its N-curves, the trick-1 runs on rented machines, the gate against rules-v1. M0 (engine, reference, RNG, CI), M1 (equity table, arena, launcher) and M2 (solver spike) are done; see `docs/history/STATUS-history.md`.

## START HERE (2026-10-21): O3G2 round 1 running

**Decision (owner, 2026-10-21):** before O_2, push O_3 *alone* close to O_3 + G2, because O_2's
training labels (exact through trick 2, then O_3, no G2) and the bronze set (O_2's test set: exact
through tricks 1-2, then O_3, G2 only if cheap by then) both use O_3 alone. Target: O_3 alone at
most 0.0025% (95% bound) overall and at every seat (today 0.0068%, seat 1 0.0201%; with G2 0.0008%,
seat 1 0.0022%). Method: **O3G2 rounds**, `docs/m3-hybrid/o3-rounds.md` (read it first): G2 chooses
the new training positions (frequency x risk^α, α = 1, 20% plain), up to three candidates a round
(nested amounts, e.g. 2M, 4M, 8M), the winner by the additive adoption rule on the golden set, a
label check of O_3's trick-2 labels, the next G2 on the winner, everything stored in R2
(`rounds/<id>/round.json`). The Copilot work of 2026-10-19/20 was discarded; this starts from
m3-hybrid 3935df4.

**Built (merged into `main` 2026-10-21; `main` is the single current line, old branches are
`archive/*` tags):** `search risk-profile` and `search labels
--harvest-risk` (`crates/search/src/risk_harvest.rs`); the round in `hybrid_rung`
(`crates/runner/src/hybrid/round.rs`); `fortyone_nn.labelcmp`; `launch.py round-manifest` and
`--instance-id` (take over a machine rented by hand: `vastai update instance`; its disk must
already be about 250 GB); `.github/workflows/o3-round.yml`; round 1's manifest
`docs/vast/o3-round-1.manifest.json`. All tests pass (fmt, clippy, workspace, Python).

**Running:** round 1, run `o3r-1-1` (started 2026-10-21 17:00 UTC; GitHub run 39639928333):
amounts 2M, 4M, 8M (all additive), on instance 57085233 (AMD EPYC 7B12, 245 usable threads, RTX 3060
12 GB, 250 GB disk, about $0.36/h, cap $3), taken over from a hand rental (`--instance-id`; vast's
`update instance` needed a `recycle instance`, now in the launcher). Watch it with `r2-peek.yml`
(run_id `o3r-1-1`); the result is `rounds/o3r-1/round.json` and `report.txt` in R2.

**Next, in order:**
1. Read round 1's report: the winner, the mix aimed at and made, the golden check of each
   candidate against the incumbent and against O3G2, the label check. Round 2 from it
   (`from=o3r-1`), with `scratch_last=4096,2048,1024,256` if data alone moved little.
2. Before round 2: fetch the base while the first parts are labelled (about 10 of the 15 fetch
   minutes saved a round).
3. Free reads still owed: the results of `o3-base80-1` (O_3 at 80M base) and the ensemble rerun
   were never written into the docs (`r2-fetch.yml`).
4. After about three rounds: a fresh golden set (seed 2) to confirm; then the trick-1 check of O_3
   alone against O_3 + G2 on about 100 trick-1 decisions (decides G2 for the bronze set).

### The goal and the chain (owner)
One objective: the chance of the money. Target: about 0.01% decision loss even at the hardest trick-1
situations, with every layer below it. The chain, top down:
1. **Trick 3-5 decisions:** exact search (cheap). Done.
2. **O_3** (trick-3 network) + **G v2** (a learned predictor of O_3's error picks the trick-3 positions to
   solve exactly): trick-2 decisions at 0.0008% (seat 1 0.0022%) on the exact golden set. Done, adopted.
3. **O_3 alone close to O_3 + G2** (2026-10-21, now): O3G2 rounds (`docs/m3-hybrid/o3-rounds.md`),
   because O_2's training labels and its test set (the bronze set) use O_3 alone.
4. **O_2** (trick-2 network, used by trick-1 search): provisional; next trained on hybrid labels
   (exact through trick 2, then O_3) after a label test at equal money against exact labels.
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
| O_2 provisional (2048-1024-256) | 505a0c7c65147ffa9565e06059f47664ea9f1bfdf0c5ea2f8d392b0e6e1337b0 | 1.5M exact labels; position error SD 3.6% (O_3: 1.85%) |
| trick-2 golden set: decisions / worlds | 635e9aef196c816380f3f15c05ede90b38b875fa4817061a57e6d39550065ab0 / 6ddc59b191e1fd534f260f9dfe103f742d6026768ca74acdbbb4fd430eb8265e | 2,000 decisions x 320 exact worlds, seed 1 |
| exact trick-2 labels, random 1.2M (s511) | shard fb51c217b5406e4b4ba1454f14b8884bced45cff51a0e29bcca7f4440f6f0931, sidecar bc4165ed9408f83ff342ab6fa157992c5910654513e417a9271a700fbbb77117 | O_2's training data |
| exact trick-2 labels, rules-v1 0.3M (s512) | shard 2160fefc6fcfd07822292f330f140a0618d25e0b89fb4ff8ed13d62e4559f1bd, sidecar d052d1a172fcf1b09d6e1cd4e95e1a2814b60e197864e1d5b18e7ab692f265f2 | |
| exact trick-2 validation 50k (s601) | shard 38de48356209cbec0f9ce41ceba28e6d5df52428abc0c866d443f9eb57964327, sidecar b39dbd6bf8f00a9b71fa800ccdaf9dd7de0ef25bc5dd3fa7b8b9e718fe895df8 | |
| exact trick-2 validation 20k (s602) | shard 855ba467ee6bd7521a442cf668980af8cefacf00701939237895d7c749dfdeea, sidecar 6e346cbf05b9f2cc0d4368e224e4dd81a11a7c2aaab1fd5af0c52a8ad177c86d | made 2026-10-20 |
| exact trick-2 validation 5k (s603) | shard 46ba92fcdf8e… (see `runs/o2-gv2-labels-3/partial/.../shards.json`) | made 2026-10-20 |
| trick-2 student (512-256, 1M lessons, plain H_3 teacher) | 954223e2f56ecade906ba28ad2ea07ff70fee70b841e0743d4c911217afb6ce6 | see below |
| equity table v0 | sha256 64740dab7845f0502c93ce195d721e1ffb8aa6ab141739ef43f754690de16808 | rebuilt by every run from `table: {hands: 1000000, seed: 1}` |
| O3G2 round records | `rounds/<id>/round.json`, `report.txt`, `profile.json` | written by each round (docs/m3-hybrid/o3-rounds.md) |

### How to run things (GitHub Actions, no vast key on the VM)
- **Launch:** workflow `vast-run.yml` on `m3-hybrid` with inputs: `manifest` (the JSON inline, one
  line), `min_cores`, `min_ram_gb`, `disk_gb`, `cpu_only` ("true" for CPU jobs), `gpus`, `min_gpu_ram_gb`,
  `max_dollars` (the cap; the launcher stops and destroys at it), `any_host` ("true"), optionally
  `offer_id` (an owner-found offer: the `Type #` number) or `cpu_match` (e.g. "7B12"), `low_cpu_pct`
  ("1" for jobs with GPU-training or other single-thread phases, else the idle watchdog kills them),
  `bin_sha` (a commit with released binaries: **6c7acf50709117d224dda5363d516f9b367cc399** is the
  latest; dispatch `release-binaries.yml` on `m3-hybrid` for a newer commit and wait for
  `bin/<sha>/manifest.json` in R2), `image` **ghcr.io/guillaumeds/41bots-base:20261017-bc692e446939**,
  `dry_run` **"false"** (the default is true!). A Blackwell GPU (RTX 50xx) needs `"blackwell": true` in
  the manifest's `machine`.
- **Watch:** `python/launcher/launch.py` helpers (`latest_heartbeat(run)`, `r2_list`, `r2_get`) with
  `R2_S3_ENDPOINT=$R2_ENDPOINT`; heartbeats every 30 s carry the step, rows done and dollars.
- **Stop:** cancel the workflow run (it asks the run to stop, waits up to 5 minutes for uploads, then
  destroys the machine) or `vast-stop.yml` (run_id, instance_id); `vast-status.yml` lists every
  instance on the account. Confirm nothing is left billing.
- **From a laptop (no R2 or vast keys there):** everything goes through GitHub Actions with the `gh` CLI:
  `gh workflow run vast-run.yml --ref m3-hybrid -f manifest="$(jq -c . docs/vast/NAME.manifest.json)"
  -f min_cores=190 -f min_ram_gb=64 -f disk_gb=60 -f cpu_only=true -f max_dollars=3 -f any_host=true
  -f bin_sha=6c7acf50709117d224dda5363d516f9b367cc399 -f image=ghcr.io/guillaumeds/41bots-base:20261017-bc692e446939
  -f dry_run=false` (GPU jobs: `-f cpu_only=false -f gpus=1 -f min_gpu_ram_gb=8 -f low_cpu_pct=1`);
  `gh run list --workflow vast-run.yml`, `gh run view <id> --log` (the launcher's log shows the machine,
  price and progress lines), `gh run cancel <id>` (stop and destroy); `gh workflow run vast-status.yml`
  (every instance on the account); `gh workflow run r2-fetch.yml -f run_id=<run>` then
  `gh run download <id>` for a run's files (results bundles `out/*.tar.zst`, unpack with Python 3.14
  `compression.zstd`); `gh workflow run release-binaries.yml --ref m3-hybrid` for new binaries.
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
- **Peek at a running round without SSH or R2 keys:** `r2-peek.yml` (run_id, e.g. `o3r-1-1`) prints
  the run's files and its latest heartbeat (step, rows, dollars) in the job log.
- **Repository and storage housekeeping:** `repo-archive.yml` (old branches become `archive/<name>`
  tags; dry run by default), `r2-inventory.yml` (read-only list of the bucket).

## Where the history is
- `docs/README.md`: one line per experiment and document (what it was, the result, adopted or not).
- `docs/history/STATUS-history.md`: every dated status entry from 2026-10-13 to 2026-10-20.
- Archived branches: `git tag -l 'archive/*'`; `git checkout archive/<name>` restores any of them.
