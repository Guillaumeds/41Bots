# M1 specification: tables, bots, arena, money type, match-equity table, thin run-ops

Derived from `docs/PLAN.md` (§5.3, §6.2, §6.6, §9, §12, the amendments A1, A8, A9, and the revised §13 M1 row) and `docs/RULES.md`. Where this file and those disagree, PLAN.md and RULES.md win and this file must be fixed. M0 is the base: `crates/engine` (rules, `Match`, `SeatView`), `crates/reference`, `crates/difftest`, `crates/wasm`, and `crates/bot` (the `Bot` and `BotFactory` traits, read `crates/bot/src/lib.rs`).

**M1 gate (PLAN §13):** duplicate matches replay exactly; leakage tests pass; the first small vast.ai run goes end to end (rent → benchmark → work → checkpoint → DONE → self-destroy) with CPU use near 100%; match-equity table v0 passes the reachable-score and flag property tests; the block standard deviation σ_b is measured.

## 1. Shared conventions

- Determinism rules from CLAUDE.md apply to every crate: integer logic in anything that affects play or results; `Vec`/arrays only (no `HashMap`/`HashSet` in game logic; a `BTreeMap` is acceptable for reports); no `unsafe`; all randomness from `engine::rng::Rng` with labelled derivations (`Rng::derive(seed, &[labels...])`), never from thread or time.
- **Seeds.** A match seed is a `u64`. The bot at seat `s` of a match with seed `m` receives `Rng::derive(m, &[0xB07, s as u64]).next_u64()` as its `start_match` seed. A duplicate block has a block seed `b`; the deck for hand index `k` of any match in that block is the identity deck shuffled by `Rng::derive(b, &[0xDEC, k as u64])` (hand index counts across the games of the match), and the first dealer of every match in the block is `Rng::derive(b, &[0xDEA]).bounded(8)`. This needs one engine addition (section 2).
- **Files and formats** (PLAN §12.7): JSON Lines for records and metrics (`.jsonl`, one object per line, UTF-8, `\n` line endings), `.npy` for arrays (little-endian, written by Rust with a hand-built header; `uint64` or `uint8` dtypes), content-addressed names where a file is immutable (`sha256` hex prefix, 16 characters), UTC timestamps as `YYYY-MM-DDTHH:MM:SSZ`, lowercase paths with `/`.
- **Money and chances.** The only score any evaluator produces is the money-chance type of section 6. Points, tricks and games appear in reports only under a `diagnostics` key.
- **Crate dependencies.** `tablemgr`, `bots`, `arena`, `equity`, `runner` may depend on `engine`, `bot`, `serde` (derive) and `serde_json`; `runner` may also use `zstd`. Nothing else without the reviewer's agreement (no tokio/axum in M1; the networked table uses `std::net`).

## 2. Engine additions (small, tested, in `crates/engine`)

- `Match::new_block(block_seed: u64, hand_decks: impl Fn(u32) -> [Card; 53], first_dealer: u8, rules: Rules)`: like `with_decks` but the deck for hand index `k` is requested from the closure when the hand starts (unbounded). Provide the standard closure `engine::game::block_deck(block_seed, k)` implementing the shuffle rule above. Test: two matches built from the same block seed deal identical decks for every hand index; `block_deck` is byte-stable (golden values for block seed 1, hands 0–2).
- `SeatView` and `PublicState` must serialise: add `serde` as an **optional** engine dependency (`features = ["serde"]`, derive `Serialize`/`Deserialize` on `Action`, `Phase`, `Event`, `Visibility`, `GameEndReason`, `BonusKind`, `Rules`, `PublicState`, `SeatView`, `Suit`), enabled by the crates that need it. The default build and the wasm build stay dependency-free; CI adds a job that builds the engine with `--features serde`.

## 3. `crates/tablemgr`: the table, the observation filter, records, the protocol

### 3.1 In-process table
```rust
pub struct TableConfig { pub seed: u64, pub rules: Rules, pub block: Option<BlockSpec> }   // BlockSpec { block_seed, first_dealer }
pub fn play_match(cfg: &TableConfig, seats: &mut [Box<dyn bot::Bot>; 8]) -> Result<MatchRecord, TableError>;
```
The table creates the `engine::game::Match` (plain `new(seed, rules)` or `new_block` from the `BlockSpec`), calls `start_match` on every bot with the seat seed rule of section 1, then loops: `view = m.view(to_act)`, `legal = m.legal_actions()`, `action = bots[to_act].act(&view, &legal)`; if `action` is not in `legal`, stop with `TableError::IllegalAction { seat, bot, action, legal }` (never apply it); else `m.apply(action)`. It records every action, and at the end the money. **The bot never receives anything but its `SeatView`**; the table is the only code that holds the `Match`.

### 3.2 Records
```rust
#[derive(Serialize, Deserialize)] pub struct MatchRecord {
    pub format_version: u32,            // 1
    pub rules_version: u32,             // engine::game::RULES_VERSION
    pub rules: Rules,
    pub seed: u64, pub block: Option<BlockSpec>,
    pub bots: [String; 8],              // bot names per seat
    pub first_dealer: u8,
    pub actions: Vec<Action>,           // every action in order (the only thing needed to replay)
    pub games: Vec<GameSummary>,        // per game: winner team, reason, final scores, hands played
    pub money: [i32; 4],                // settled by the engine
    pub winner: u8,
    pub public_timeline: Vec<TrickPoint>,   // per completed trick: hand_index, trick, scores after, game_index; the money-chance columns are added at M4 (leave the field, fill with scores only)
}
```
`MatchRecord::replay(&self) -> Result<engine::game::Match, _>` rebuilds the match (`Match::replay` for plain seeds; `new_block` + actions for blocks) and `verify(&self)` asserts the rebuilt money, winner and games equal the record. Records are written as JSON Lines by `RecordWriter` (append, one record per line) and read by `RecordReader`. A round-trip test: write, read, replay, equal.

### 3.3 Leakage tests (M1 gate)
1. **Undealt permutation:** for 200 random seeds, build two matches from decks that differ only in a permutation of positions 42–52 (the undealt pack beyond the second turned card) and play both with the same deterministic bots (rules bot): every action and every `SeatView` must be identical throughout (the undealt pack can never influence anything).
2. **View purity:** `SeatView` for seat s, serialised to JSON, never contains any card that is in another seat's hand, another seat's burn, or the undealt pack (checked against `truth()` at every decision of 100 random matches), except cards already public (played, turned-and-on-table, taken turned cards) — assert by set arithmetic on the bitboards and by scanning the JSON for card names.
3. **Protocol purity:** in the networked mode (3.4), the bytes sent to seat s are exactly the JSON of `SeatView` for s plus the legal actions; a test runs a match over the in-memory transport and asserts every message's card set against `truth()` as in test 2.

### 3.4 Protocol and networked table
Versioned JSON messages, one per line, over a byte stream:
- table → seat: `{"v":1,"type":"hello","seat":3,"rules":{...},"seed":123}`; `{"v":1,"type":"act","view":{SeatView},"legal":[{"Play":17},...]}`; `{"v":1,"type":"end","record":{MatchRecord}}`.
- seat → table: `{"v":1,"type":"action","action":{"Play":17}}`.
Transport trait `SeatLink { fn send(&mut self, line: &str); fn recv(&mut self) -> Option<String>; }` with two implementations: in-memory channels (`std::sync::mpsc`) and TCP (`std::net::TcpStream`, blocking, one thread per seat). `serve_match(cfg, links: [Box<dyn SeatLink>; 8]) -> MatchRecord` on the table side and `run_seat(link, bot)` on the client side. A test plays the same seed in-process and over the in-memory transport and asserts the records are identical; another plays over TCP on localhost with 8 client threads. (The Cloudflare Durable Object port at M3b reuses the message format.)

## 4. `crates/bots`: the baseline bots

Both implement `bot::Bot` and a `BotFactory`; both are deterministic given their seed and views; both only read `SeatView`.

- **`RandomBot`**: uniform over `legal` using `Rng::derive(seed, &[decision_index])`.
- **`RulesBot` (`rules-v1`)**: the published 25/45-family heuristics (docs/REVIEW-2026-10-13.md part 1 §3.15 and part 4), made exact and deterministic. Decision order; the first applicable rule decides:
  1. **Burn:** burn the lowest-value card by this ranking: non-trump cards first (lowest plain rank first, prefer the suit the hand holds fewest of), then the lowest trump; never the taken card if it is a trump; aces are eligible by plain rank order like any card (any card may be burned, H3 as amended 2026-10-14).
  2. **Clinch defence (score-aware):** if some *other* team could reach 41 or more by winning this trick (its score + 5 + the bonus it would get if the 5 of trumps is in the trick or already played by it… use the exact engine arithmetic: `score + 5 + (bonus if the trick currently contains the 5 of trumps)` ≥ 41) and a card of mine would win the trick as it stands, play the lowest such card.
  3. **Partner is winning** the trick as it stands and nobody after me can beat it with what is public: play the lowest legal card that does not beat the partner (never "strip the partner's trumps").
  4. **Leading:** if the 5 and J of trumps are both gone (played or known out of play) and I hold the best remaining trump, lead it; else on tricks 1–3 lead the lowest plain card of my longest plain suit; on trick 4 lead my strongest card (fight for trick 4, to lead trick 5); on trick 5 lead my best card.
  5. **Following, can win cheaply:** play the lowest card that wins the trick as it stands, but do not use a top trump (5, J, Joker, A♥) before trick 4 unless rule 2 applies or it is the only way to follow legally.
  6. **Cannot or should not win:** play the lowest legal card (plain before trump), keeping a "saver" (the highest trump) for later.
  "Lowest/highest" use the engine's `trump_rank`/`plain_rank`. "Known out of play" = `played | table_card | burned-by-me`; the bot keeps a small card tracker from its own view (played cards, the turned card, its own burns, `proved_no_ace`, `ace_undealt`, `robber`).
  Tests: each rule on a hand-made view (use `engine::game::build_deck` + `Match::with_decks` to produce real views), determinism (same seed → same actions over 50 matches), and a 1,000-match smoke run against `RandomBot` where `rules-v1` must win clearly more money than random (report the margin; it is a diagnostic, not a gate).

## 5. `crates/arena`: duplicate blocks, the sequential test, ratings, the canary

### 5.1 Blocks
A **block** is one block seed replayed under a set of team-to-slot assignments. Slots are the four team positions `{0,4}, {1,5}, {2,6}, {3,7}`. The **14-way block** for candidate A versus incumbent B is every assignment of A/B to the four slots except AAAA and BBBB (4 × 1v3, 6 × 2v2, 4 × 3v1). The **4-rotation block** is A in one slot and B in the other three, rotated (the cheap pre-screen). The deck for hand k is the same across all assignments (section 1), so the cards every seat holds in hand k are identical across the block; only who plays them differs.
Block score (owner decision: equal weights): `s = mean over A-team seats of money_chance_A − mean over B-team seats of money_chance_B`, where a team's realised money chance in a match is 1 for the winning team and 0 otherwise; report also the money in stake units. Expectation under "no difference" is 0; |s| ≤ 1.

### 5.2 Generalised SPRT on blocks (Fishtest form)
`llr = (n/2) · ln( ((μ − s0)² + σ²) / ((μ − s1)² + σ²) )` over the block scores (sample mean μ, sample variance σ², n blocks), computed in f64 **in the arena only** (reports, never play). Bounds `ln(β/(1−α))` and `ln((1−β)/α)` with α = β = 0.05 (≈ ±2.944). Hypotheses in money-chance units; defaults: gainer test `[s0, s1] = [0, 0.02]`, non-regression `[−0.015, 0.005]`; configurable. Two stages: stage 1 at a cheap budget can only **reject**; stage 2 at the full budget decides. Hard cap: `max_blocks` (default 5× the worst-case expected duration `8.67 (σ_b/Δ)²`), after which the result is `Inconclusive`. No manual early stopping. Minimum 20 blocks before any decision.
`σ_b` is measured by `arena measure-sigma` (identical bots, 200 blocks) and stored in the report.

### 5.3 Ratings
Offline **Plackett-Luce** by maximum likelihood (Hunter's MM algorithm; top-1 Luce form: table winner ~ `exp(r_i)/Σ exp(r_j)`) over all recorded tables, f64, with standard errors from the observed information; output per program. Elo is not computed.

### 5.4 Canary and reports
- **Canary:** `arena canary` plays 50 blocks of identical bots in all 14 assignments and asserts the mean block score is exactly 0 and every block score is exactly 0 (identical bots with identical seeds make identical decisions, so A and B must tie exactly). Run in `cargo test` with 5 blocks.
- **Reports:** JSON (machine) + a plain-English text summary: blocks played, μ, σ, LLR, decision, per-mix breakdown (1v3, 2v2, 3v1), diagnostics (games, points, tricks), matches/s, and the record files written (`.jsonl`).
- `arena` binary: `arena sprt --a <bot> --b <bot> [--block14|--rot4] [--s0 --s1 --alpha --beta --max-blocks --threads] --out <dir>`, `arena canary`, `arena measure-sigma --a <bot> --blocks N`, `arena rate <records…>`. Bots are selected by name from the `bots` crate registry (`random`, `rules-v1`). Parallel over blocks with `std::thread`; results are merged in block order so the report is identical for any thread count.

## 6. `crates/equity`: the money-chance type and the exact match-equity table

### 6.1 Type
```rust
pub const ONE: u64 = 1 << 32;
#[derive(Clone, Copy, PartialEq, Eq, Debug)] pub struct MoneyChance(pub [u64; 4]);   // units of 2^-32; invariant: sum == ONE exactly
impl MoneyChance {
    pub fn certain(team: u8) -> Self;
    pub fn from_weights(w: [u128; 4]) -> Self;        // normalise by largest-remainder rounding so the sum is exactly ONE
    pub fn mix(terms: &[(u64 /*weight, 2^-32 units*/, MoneyChance)]) -> Self;   // Σ w_i·c_i / Σ w_i, u128 accumulation, largest-remainder rounding
    pub fn expected_money(&self, stake: i32) -> [i64; 4];   // round((8p − 2) × stake) in stake units... see H6: +6·stake with chance p, −2·stake otherwise ⇒ (8p − 2)·stake
    pub fn rotate(&self, by: u8) -> Self;              // team relabelling
}
```
Only `equity` constructs `MoneyChance` values; a `#[non_exhaustive]`-style private field would block pattern matching, so keep the field public but document that every constructor normalises and add `debug_assert!(sum == ONE)` in all methods. A unit test tries 10,000 random weight vectors and checks the sum is exactly ONE.

### 6.2 Match state and table layout
`MatchState { scores: [u8; 4], games: [u8; 4], dealer_team: u8 }` with reachable scores `{0,5,…,40} ∪ {11,16,…,36}` (index 0–14; bonus flag ⟺ score mod 5 == 1). Canonical form: rotate teams so `dealer_team == 0`; index = `games_bits (4 bits, team order after rotation) × 15^4 + score_idx[0]·15³ + … + score_idx[3]`; 16 × 50,625 = 810,000 entries. States with a team at 2 games are terminal and not stored. Stored as `equity-v<N>.npy` (`uint64`, shape `[16, 15, 15, 15, 15, 4]`), plus `equity-v<N>.json` with the hand-outcome model id, sample sizes, sweep statistics and the content hash. Lookup for an arbitrary state rotates, indexes, rotates back.

### 6.3 Hand-outcome model v0 (stationary, from the rules bot)
Estimated from N self-play hands of `rules-v1` (N ≥ 1,000,000, seeds documented), expressed relative to the dealer's team (offset 0 = dealer's team, 1 = the team that leads trick 1, …), as integer counts turned into fixed-point probabilities:
- `P(winner_t = offset | t, tricks_won_so_far[4])` for t = 1..5;
- `P(five_in_trick_t | t, five not yet played, winner_t)`;
- `P(end_bonus_team = offset | final tricks_won[4], five never played)` (a trump is always played in a completed hand, RULES note).
Every count table is written to `outcome-v0.json` with its sample sizes so the model is inspectable.

### 6.4 The sweep
Exact backward induction. Order the canonical states by (number of games won, descending) then (total points, descending) and solve each with a within-hand dynamic program: starting from the match state, expand trick by trick using the model's conditionals, collapsing equal intermediate states `(scores, flags, tricks_won, five_played, trick)` and applying the engine's exact scoring and end checks at every trick (award 5 → immediate bonus → 41 check; after trick 5: jink → end bonus → 41 check). A game end (41 or jink) moves to the successor match state (scores and flags reset, games +1, dealer team +1, or terminal `certain(team)` if that makes 2 games); a hand end without a game end moves to (scores updated, dealer team +1). Every successor is already solved by the ordering. All arithmetic in `MoneyChance::mix` (u128), so the table is byte-identical on every machine and thread count. Record per state nothing but the value; the model's sample sizes go in the JSON.
Validation (M1 gate): every entry sums to ONE; team-rotation symmetry (the value of a state equals the rotated value of the rotated state); a team at 1 game and 36 points with the others at 0–0 has a chance above 0.5 (sanity); monotonicity in own points holding everything else fixed (with the stationary model this must hold exactly; assert it); a direct Monte-Carlo estimate (10,000 matches from 20 sampled states, using the model as a simulator) agrees with the table within 3 standard errors; the sweep is reproducible byte for byte across 1, 2 and 4 threads.
Binary: `equity build --model outcome-v0.json --out equity-v0.npy`, `equity estimate-model --bot rules-v1 --hands N --out outcome-v0.json`, `equity lookup --table … --state "scores=.. games=.. dealer_team=.."`.

## 7. `crates/runner`, `deploy/docker`, `python/launcher`, workflows (thin run-ops)

- **Runner** (`runner run <manifest.json>`): the manifest names a `run_id`, a `chunk` (`difftest {matches, seed}`, `arena_block {…}`, `equity_sweep {…}`, `benchmark {seconds}`), `threads` (from `NCPU`), `deadline_utc` and an `out` prefix. It (1) records the hardware (CPU model and flags, `/sys/fs/cgroup/cpu.max`, RAM, `nvidia-smi` if present) to `hardware.json`; (2) runs the 60-second benchmark (`difftest` matches per second per core) to `benchmark.json`; (3) runs the chunk, writing `heartbeat.json` every 30 s (UTC time, CPU utilisation from `/proc/stat`, progress, elapsed, estimated cost = elapsed × `DPH`); (4) writes outputs under `out/`, then `DONE`; (5) uploads everything with `rclone copy` using `RCLONE_CONFIG_R2_*` environment variables (type s3, provider Cloudflare, endpoint, keys); (6) if `CONTAINER_ID` and `CONTAINER_API_KEY` are set, destroys the instance (`vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"`) as the very last step. Mutable keys in R2 are versioned (`runs/<run_id>/heartbeat/<seq>.json`); bundles and results are immutable.
- **Docker** (`deploy/docker/Dockerfile`): `FROM nvidia/cuda:12.6.3-base-ubuntu24.04`, apt `ca-certificates curl python3 python3-pip`, `pip install vastai`, rclone from the official tarball (pinned 1.75.2), a non-root `app` user, no project code, no secrets. Workflow `.github/workflows/image.yml` (dispatch + tags) builds and pushes `ghcr.io/guillaumeds/41bots-base:<date>-<sha>` and never `latest`.
- **Release binaries** (`.github/workflows/release-binaries.yml`, dispatch + tags): builds `runner`, `difftest`, `arena`, `equity` for `x86-64-v2` and `x86-64-v3` (`RUSTFLAGS=-C target-cpu=x86-64-v2|v3`, never native), tars them, uploads to R2 `bin/<sha>/<build>.tar.zst` with a `manifest.json` (sha256 of every file), using the repository's variables and secrets.
- **Launcher** (`python/launcher/launch.py`, stdlib + `vastai` SDK; run by `.github/workflows/vast-run.yml` with dispatch inputs `manifest`, `min_cores`, `max_minutes`): searches offers with PLAN §12.2's filters, on-demand only, ranks by estimated cost per unit of work (`dph_total / cpu_cores_effective` until benchmarks exist in R2, then by measured matches per second per dollar for that CPU model), applies the owner's 20% speed-premium rule, creates the instance with the base image, environment (`RUN_ID`, `NCPU`, `DPH`, `R2_*`, `RCLONE_CONFIG_R2_*`, `BIN_SHA`) and an `onstart` command that downloads `bin/<BIN_SHA>/<v2|v3 by CPU flags>.tar.zst` and the manifest from R2, verifies the hashes, and runs `runner run`. The workflow then polls the vast API and the R2 heartbeat every 60 s: kills and reports if the heartbeat is missing for 3 minutes, CPU utilisation is below 85% for 5 minutes while work remains, or `max_minutes` is reached; destroys the instance if it still exists after `DONE`; prints a cost report (dollars, elapsed, matches, matches per dollar, average CPU) and uploads it to R2 `runs/<run_id>/report.json`.
- **M1 gate run:** a `difftest` chunk of 2,000,000 matches on the cheapest acceptable machine with ≥ 16 cores, expected a few minutes, CPU near 100%, cost well under $0.10.

## 8. Tests and CI

`cargo test --workspace` stays under a minute. CI (`ci.yml`): add `cargo build -p engine --features serde` and a `cargo test -p arena -- canary` step; the networked TCP test runs on both OSes. The `differential` job continues to use `crates/difftest`.

## 9. What is explicitly out of M1

Belief, search, PIMC, networks, AIVAT (needs the oracle head at M4), the Cloudflare port (M3b), the full runner priority queue (M3). The `public_timeline` money columns are filled at M4.
