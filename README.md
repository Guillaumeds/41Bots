# 41Bots

Strongest-possible bots for the 8-player, 4-partnership trick-taking card game **41**. Every decision is made to win the money.

- `docs/RULES.md`: the authoritative rules and house rules
- `docs/PLAN.md`: research analysis, design, tech stack, run operations, checkpoints
- `docs/STATUS.md`: current checkpoint and next steps
- `docs/RESEARCH.md`: the outside research (papers, other programs, the game family)
- `docs/EXPERIMENTS.md`: every experiment we ran and its result
- `docs/README.md`: index of every document
- `CLAUDE.md`: working rules for Claude sessions (local and cloud)

## Build and test
```
cargo test --workspace
cargo build -p engine --target wasm32-unknown-unknown
```
The Rust toolchain is pinned in `rust-toolchain.toml`; `rustup` installs it automatically.
