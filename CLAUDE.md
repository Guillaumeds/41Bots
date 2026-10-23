# 41Bots: instructions for Claude sessions

You are building the strongest possible bots for the 8-player card game **41** (4 partnerships of 2), from a clean slate.

## Read first, every session
1. `docs/STATUS.md`: where we are, what to do next. Update it at the end of every session.
   `docs/README.md` lists every document and experiment with its outcome (adopted, rejected, open); `docs/EXPERIMENTS.md` records every experiment we ran (question, method, result, cost, limits; add each new result there) and `docs/RESEARCH.md` the outside research the methods come from (papers, other programs, the game family); old branches are `archive/*` tags. Keep both current: STATUS holds only the present, history goes to `docs/history/`.
2. `docs/RULES.md`: the authoritative rules, including the owner's house rules (section H). Rules questions are answered there, never guessed.
3. `docs/PLAN.md`: the approved research analysis, design, tech stack, run operations and checkpoints M0–M9. Follow it, and don't redesign. If something in the plan seems wrong, raise it with the owner before changing course.
4. `docs/REVIEW-2026-10-13.md`: the review that amended the plan (the amendments themselves are at the top of PLAN.md). Read it when an amendment needs its evidence.

## Non-negotiables
- **One objective everywhere: the chance of the money** (RULES H6/H7, PLAN §5.3). A match is the first partnership to win 2 games; losers pay the stake.
  - Solver, search, training targets, rewards and tournament ranking all score in a single "money chance" type: 4 numbers, one per team, summing to 1.
  - Points, tricks and games are inputs or diagnostics only, never optimised for their own sake.
  - The money chance is re-evaluated after every trick ("public enemy number one" can change mid-hand).
- **Proven methods only.** Every component must trace to the research in PLAN §2–4. When two proven options compete, duplicate tournaments with SPRT decide (PLAN §9). Don't invent.
- **Quality first.** Save compute only through engineering that gives identical answers (PLAN §10), never by playing worse.
- **The exact solver is cheap late in the hand only, but exact PIMC is still the reference at tricks 1–2** (PLAN amendment A3', owner 2026-10-14). Exact open-hand solves are the default from trick 3 and are required for the trick 1–2 yardstick and teacher targets, run offline on large machines with measured world counts; rollouts (measured at trick 2 to lose about 13 times more money per decision) stay only as a scalable fallback where exact is unaffordable. Never budget on "millisecond per world" assumptions (PLAN amendments A2–A4).
- **The hybrid replaces exact search early in the hand (owner, 2026-10-16; docs/M3-HYBRID-SPEC.md).** Oracle networks O_4, O_3, O_2 take over from exact search at the start of tricks 4, 3, 2; each is proven 0.01 by the audit loop (`docs/m3-hybrid/audit-loop.md`: every situation the auditor can find, not averages). **No trick-1 golden set is built:** trick-1 decisions are PIMC over many deals searched with H_2, and the saved exact trick-1 worlds are only a rough test. For trick 1, later improvements are of two kinds only: pushing tricks 2–4 further below 0.01, and more deals at trick 1 (other engineering opportunities elsewhere remain open).
- **Rental floors (owner, 2026-10-16):** at least 190 threads for trick-3 and trick-2 runs, 120 for trick-4 runs (GPU-bound); price cap $1.70/h with the launcher's cheapest-first price ladder; at least one GPU, 2-GPU machines allowed; download at least 100 Mbps.
- **Money chances are fixed-point from M1** (u64 per team in units of 2^-32 summing exactly to 2^32, u64/u128 accumulators, index-ordered reductions), so solver, teacher and tournament results are byte-identical across machines, thread counts and WASM. Floats live only inside the network forward pass.
- **The rules must be exact.** Every numbered rule gets tests. Illegal actions are impossible (H1).

## Engineering rules
- **Toolchain:** pinned in `rust-toolchain.toml` (Rust 1.99.x, edition 2024).
  - Python is 3.14 via `uv`, with PyTorch pinned to `2.14.1` (cu126 on vast.ai; cpu on the laptop). **Never 2.15+**: it drops the Pascal/Volta GPUs we rent. See PLAN §11.
- **Determinism:**
  - The engine uses whole numbers only, plus our own pinned random-number generator (SplitMix64 seeding + xoshiro256**), with golden-vector tests that must pass on Windows, Linux and wasm32.
  - No `rand::StdRng` for anything replayable.
  - No randomly-seeded `HashMap` iteration in game logic.
  - Use `libm` for any transcendental maths in shared code.
  - Game records store every action; replays re-apply actions and never re-run networks.
- **Portability:**
  - Never `target-cpu=native`.
  - No PEXT/PDEP in hot loops (slow on pre-Zen-3 EPYC).
  - Build release binaries for x86-64-v2 and v3, with runtime dispatch (`multiversion`) for hot kernels.
- **Avoid:** `bincode` (discontinued, RUSTSEC-2025-0141), and Rust-only data formats for anything Python must read. Formats are fixed in PLAN §12.7.
- **The reference implementation** (`crates/reference`) is written independently and naively, using different data structures. It must never import engine internals; it shares only the public action and seat types, if needed.
- **`unsafe` is forbidden workspace-wide**; `crates/nn` is the only crate that may opt out for code, for one audited SIMD kernel module with a differential test against the plain reference. The foreign-function boundary crates (`crates/wasm`, later `crates/pybind`) may carry the `#[unsafe(no_mangle)]` export attribute only, under a `deny` lint with a per-item allow; never an `unsafe` block or function. CI greps for both.
- **Before every commit:**
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
- **Where compute runs (owner, 2026-10-14):** this VM is for building and testing. Any measurement or calculation expected to take more than about 5 minutes here runs on vast.ai (on-demand, through the runner), not locally.
- **Parallel rentals (owner, 2026-10-14):** independent jobs may run on several vast.ai machines at once; each is monitored and destroyed when done.
- **SSH and Jupyter always on (owner, 2026-10-23):** the launcher (`python/launcher/launch.py`, `create_instance`) rents every instance as Jupyter + SSH (`--jupyter --direct`, the console's Connect button) and starts the boot script from the on-start command. Launch every run and every watcher through `vast-run.yml` so it applies; never use `"entrypoint_mode": true` (the old no-SSH mode) except to debug the launcher itself. Keep the `41bots-` label prefix on instances (the cleanup only watches those). To keep a machine after a run use `"keep": true` in the manifest's `machine` block; to reuse one use `"instance_id"`. Details: `docs/STATUS.md`, "SSH and Jupyter on every machine".
- **Never kill a rare cheap or fast machine automatically (owner, 2026-10-23).** The owner stops those by hand. Manifests that run on one use `"keep": true`
  (`keep_hours` up to 8); no session cancels a run (`gh run cancel`), writes STOP, or destroys an instance on such a machine without asking the owner first.
  A cancel has twice ended in a destroyed machine; the launcher now writes the keep marker at the start of a kept run (`launch.py`), but still verify
  `runs/<run id>/KEPT.json` and the vast instance list after any stop. Never take over a machine another session uses: check its label and the
  `docs/STATUS.md` note first (two sessions took over the same box within two minutes). Hand-rented machines are taken over with `instance_id`
  (the manifest's `min_gpu_ram_gb` and `disk_gb` must not exceed the instance's).
- **Machine labels (owner, 2026-10-23): easy for a person to read, `41bots - <session> (<what>) - ETA <HH:MM Irish>`**, for example
  `41bots - S2 (1M-2) - ETA` and `41bots - O2 (Network Capacity 11M) - ETA 10:30`. After launching or taking over a machine, relabel it this way
  (`vastai label instance <id> "..."`, or `PUT /api/v0/instances/<id>/` with `{"label": ...}`) and update the ETA when it changes (Irish time, see above).
  The launcher's own label is `41bots-<run id>` (no spaces) and only machines with that exact prefix are watched by `vast-cleanup`; the human-readable
  labels are therefore exempt from the cleanup, so a session that relabels a machine owns its destruction (the run still destroys or keeps it per its manifest).
- **Host reliability (owner, 2026-10-16):** the launcher accepts vast.ai hosts with reliability of at least 90%.
- **Verified hosts (owner, 2026-10-16):** prefer vast.ai-verified hosts; unverified or deverified hosts are accepted when no verified offer fits.
- **Times for the owner (owner, 2026-10-16):** give clock times in Irish time (Europe/Dublin: UTC+1 until Sunday 25 October 2026, then UTC+0), never Mauritius time.
- **Cloud sessions:**
  - The VM has 4 vCPU and 16 GB RAM, and commands time out after ≤10 min.
  - Run long jobs (such as 10M differential games) as `--release` binaries with a size argument, or leave them for vast.ai or CI. Unit tests stay fast.
- **Secrets:** never commit secrets or tokens. R2 and vast.ai keys come in at checkpoint M1 via environment variables or GitHub secrets.
- Work on a branch per checkpoint (e.g. `m0-engine`) and open a PR. CI must be green.

## Talking to the owner
- The owner knows the game deeply but not AI jargon. Write in plain English.
- The first time you use a technical term, explain it in [square brackets] in plain words.
- Be concise and state outcomes plainly, including failures.
- Never name a machine by its chip number alone. Say what it is in words and always give its usable threads (CPU), RAM and hourly rate, e.g. "AMD Threadripper PRO 9995WX workstation chip (184 threads, 251 GB RAM, $1.27/h)". The offer number may follow in brackets.
