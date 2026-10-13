# 41Bots: instructions for Claude sessions

You are building the strongest possible bots for the 8-player card game **41** (4 partnerships of 2), from a clean slate.

## Read first, every session
1. `docs/STATUS.md`: where we are, what to do next. Update it at the end of every session.
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
- **The exact solver is cheap late in the hand only.** Exact open-hand solves are the default from trick 3; tricks 1–2 use measured world counts or policy rollouts. Never budget on "millisecond per world" assumptions (PLAN amendments A2–A4).
- **Money chances are fixed-point from M1** (u32 per team in units of 2^-32, u64 accumulators, index-ordered reductions), so solver, teacher and tournament results are byte-identical across machines, thread counts and WASM. Floats live only inside the network forward pass.
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
- **`unsafe` is forbidden workspace-wide**; `crates/nn` is the only crate that may opt out, for one audited SIMD kernel module with a differential test against the plain reference.
- **Before every commit:**
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
- **Cloud sessions:**
  - The VM has 4 vCPU and 16 GB RAM, and commands time out after ≤10 min.
  - Run long jobs (such as 10M differential games) as `--release` binaries with a size argument, or leave them for vast.ai or CI. Unit tests stay fast.
- **Secrets:** never commit secrets or tokens. R2 and vast.ai keys come in at checkpoint M1 via environment variables or GitHub secrets.
- Work on a branch per checkpoint (e.g. `m0-engine`) and open a PR. CI must be green.

## Talking to the owner
- The owner knows the game deeply but not AI jargon. Write in plain English.
- The first time you use a technical term, explain it in [square brackets] in plain words.
- Be concise and state outcomes plainly, including failures.
