# Status

**Current checkpoint: M3 (in progress)** (PLAN §13, as revised on 2026-10-13): PIMC bots, the exact early-trick yardstick and its N-curves, the trick-1 runs on rented machines, the gate against rules-v1. M0 (engine, reference, RNG, CI), M1 (equity table, arena, launcher) and M2 (solver spike) are done; see `docs/history/STATUS-history.md`.

## O_2 Stage A (2026-10-22): plain labels win; G2 is no bulk source; volume run next
Results in EXPERIMENTS D6 (`docs/m3-hybrid/o2-stagea-netcmp*.json`): on the 250k exact test set plain 1.5M ties exact 1.5M
(SD 3.67% against 3.64%), plain 6M is clearly better (3.13%) at about 1/30 of the label cost, and G2 labels (0.42M) score the same as
exact labels of that count (4.15%): no niche. **Stage B result (2026-10-23, EXPERIMENTS D7): the 11M-plain-label network scores SD 2.93% (p60 3.13%), bias 0.05%, sibling regret 0.199%; keys: `runs/o2-volnet-5/out/volnet/`.** **Stage B started overnight:** `o2-vol-2` makes 22M plain labels (16M random + 4M
rules-v1 + 2M harvested) in ten stored streams; then the real 2048-1024-256 network is trained from the stored streams
(`python/tools/o2_train_manifest.py`, to be generalised), scored by `netcmp` against `p60`, followed by the audit loop. A
trainer race on a late sidecar (killed `o2-g2arm-2`) is fixed in `train.py` (new binaries needed for streamed-parts runs).
**To do before the audit loop:** reused validation with label_parts (path-forward.md, Session 2). Takeover of a hand-rented
machine works (`vast-run.yml` `instance_id`, disk at least 80 GB).

## START HERE (2026-10-22): more training data does not move O_3; choose the next lever

**Where we are.** O_3 f5766a is still the adopted trick-3 network: 0.0068% alone (seat 1 0.0201%),
0.0008% with G2 and exact solving (seat 1 0.0022%) on the stored trick-2 golden set. The goal
(owner, 2026-10-21) is O_3 *alone* at most 0.0025% overall and at every seat, because O_2's training
labels and the bronze set use O_3 alone. Three attempts on 2026-10-21/22 did not move it (full
record: `docs/m3-hybrid/o3-rounds.md` "Results" and `docs/m3-hybrid/specialist.md`):
1. **O3G2 rounds 1 and 2** (extra G2-chosen rows, +2M/+4M/+8M, additive): 0.0069-0.0080%, all
   within retrain noise of 0.0068%; round 2's wide network from scratch was worse (0.0079%, seat 1
   0.0325%). Round 2's rule-winner c2 (0.0070%) is stored but not adopted as O_3.
2. **Specialist S** (a network trained only on the 16.5M base rows G2 flags): on its own rows S is
   at O_3's level (SD 3.03% against 2.94%); routed in play it scored 0.0079%, against 0.0008% for
   the exact solve. Rejected for now.
3. Nothing is running: no rented machine was open at the last check (2026-10-22 07:05 UTC). About
   $6 went on rentals from round 1 attempt 3 onward (rounds $3.2, specialist training $0.55, routed
   check $1, the three stopped speedtests and the error measurement about $0.5, the refused start $0.7).

**Next, in order (owner chooses):**
1. The free reads are done (2026-10-22, `docs/EXPERIMENTS.md`): O_3 on an 80M base had lower value
   error but no better decisions (0.0080%, C4); the optimizer's-curse test reproduced the leader's
   own-team under-valuation from zero-mean noise alone (C7); the ensemble was never measured (C8).
2. The data-versus-capacity test that never ran: `o3-basecurve-50/100` (O_3 on 20M against 40M rows,
   golden set) were cancelled at the start on 2026-10-19; the manifests exist, untested since (C5).
3. A 2-4-network ensemble of O_3 for building the bronze set only (offline, so 2-4x network time is
   acceptable), judged by the same golden check. Its two earlier attempts ran at about 10 worlds a
   second on an AMD EPYC 7B12 server chip (256 threads, $0.30/h; members' weights exceed the chip's cache; C8).
4. A round 3 only if a new lever appears (it would start from round 2's c2:
   `rounds/o3r-2/round.json`). After about three rounds, or before claiming a number: a fresh golden
   set (seed 2).
5. Then the trick-1 check of O_3 alone against O_3 + G2 on about 100 trick-1 decisions (decides
   whether G2 is used for the bronze set).

**Path from here (2026-10-22, owner starts one new session and branch per step, each pushes to `main` when done):**
`docs/m3-hybrid/path-forward.md`, section 6 has a brief per session: 0 the free checks (shortlist + search at
the app's deal counts, concentration of S2's loss and the router, the root-restricted solve), 1 S2 A/B tests,
2 O_2 (label test at equal money, volume, audit loop), 3 trick-1 yardsticks (copper, playout yardstick, a
small capped anchor), 4 S1, 5 ship and the improvements. The solver half of the root-restricted solve
(`FastOptions::root_keep`) is already on `main`.

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
   identical). Measured **0.64x (2 cards) / 0.70x (3 cards)** of the time at trick 2: a 1.4-1.55x
   speed-up. With result 1 it is worth about 40% more deals, nothing more. Trick 1 not measured.
   Consequence for the app: the leader needs 368 ms a deal in WebAssembly, so about 19-27 deals in 7 s
   on the VM and fewer on a phone, at or below the break-even exactly where S2 loses most: the lever
   there is the cost of a deal (WebAssembly network kernel), not the shortlist.
   Next: session 1 (S2 A/B tests) is unaffected; the open question for the app is the cost per leader deal.

**Records:** `docs/EXPERIMENTS.md` (every experiment and its result), `docs/RESEARCH.md` (the outside
research), `docs/README.md` (index).

**Built and merged into `main` (the single current line; old branches are `archive/*` tags):**
`search risk-profile`, `search labels --harvest-risk`, `search risk-filter`, `search net-error`,
the `s:NET` expert in `search::exact_leaves`; runner kinds `hybrid_rung` with `round`
(`crates/runner/src/hybrid/round.rs`), `specialist`, `net_error`; `fortyone_nn.labelcmp`;
`launch.py round-manifest` and `--instance-id`; workflows `o3-round.yml` (now with
`min_reliability`), `store-input.yml`, `r2-peek.yml` (now with `show`), `repo-archive.yml`,
`r2-inventory.yml`. Latest released binaries: `2c0774db7b576dccd405240746065cd6f39766fd`.

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
  `bin_sha` (a commit with released binaries: **2c0774db7b576dccd405240746065cd6f39766fd** is the
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
  -f bin_sha=2c0774db7b576dccd405240746065cd6f39766fd -f image=ghcr.io/guillaumeds/41bots-base:20261017-bc692e446939
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
