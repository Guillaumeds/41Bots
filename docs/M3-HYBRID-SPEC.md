# M3 hybrid yardstick: exact early tricks + oracle value network

Status: approved direction by the owner on 2026-10-16 (Mauritius). Spec by Fable 5.1; builds by Opus
(CLAUDE.md "Who builds what"). This document amends the trick-1 parts of M3-SPEC §5.5 and §13.1;
everything not mentioned here is unchanged.

## 1. Why

Measured on 2026-10-16 (docs/STATUS.md): an exact trick-1 leader world costs on average at least
57×10⁹ solver nodes (trick 2: about 1.8×10⁷), and trick-1 decisions are 2–5× noisier per world
than trick 2, so a spec-sized exact trick-1 golden set would cost $300–700 and weeks of machine
time. The cost is intrinsic to four-sided max^n search (only shallow pruning applies), not a bug.

The remedy is the plan's own M5 method, built early: a depth-limited exact search cut off by the
oracle value head on the sampled world (REVIEW-2026-10-13 R:175; PLAN §6.5 head 4; precedents
Sturtevant & White 2006, Learned Belief Search, AlphaZero-style value cut-offs). It is adopted for
a yardstick only after it is shown to agree with exact search to the standard below, and exact
solves remain the anchor.

## 2. Definitions

- **Hand:** five tricks of eight cards; trick 5 has no choice. "Start of trick k" = no card of trick
  k played yet (`Position::n_played == 0`, `hand.tricks_done == k − 1`). A pending rob exists only
  in trick 1, so frontier positions at k ≥ 2 never carry one.
- **Oracle net O_k:** a network that takes a perfect-information position at the start of trick k
  (all hands known, scores, flags, tricks per team, five played, best trump so far, leader, trump,
  games won, dealer team: every field the solver's value depends on, see `solver::Position`,
  `solver::MatchCtx`, `equity::hand::InHand`) and returns the four teams' money chances.
- **Hybrid H_k:** the exact solver (same max^n, same tie rule, same leaf rule at hand end) from the
  decision point down to the start of trick k, with O_k's value at every position reached there
  instead of searching further. H_k is defined only for decision points before trick k.
- **Rungs (owner's order):** O_4, then O_3, then O_2. O_4 gives no speed (an exact solve from trick 4
  costs about 0.04 ms) but proves the whole pipeline on the easiest case. O_3 is the working cut for
  the trick-1 yardstick (exact tricks 1–2, about 16 card plays). O_2 (exact trick 1 only) is the
  stretch goal.

## 3. "Proven 0.01" (the acceptance standard of each rung)

On a golden set of the trick before the cut (O_4 on trick 3, O_3 on trick 2, O_2 on trick 1),
for every decision and every world of that set, compute the candidates' values exactly and with
H_k on the *same* worlds (paired, so sampling noise cancels). Then report:

1. **Decision loss (the standard):** per decision, the hybrid's choice is the candidate with the
   best H_k mean over the decision's worlds; its loss is the exact mean of the exact-best candidate
   minus the exact mean of the hybrid's choice, in the decider's team's money chance. Pass when the
   mean loss over decisions is ≤ 0.01% money chance at the 95% upper bound (bootstrap over
   decisions, 1,000 replicates, seed 1), overall and in each position band (1–2, 3–4, 5–8).
2. **Value error (diagnostic):** per candidate and world, H_k minus exact; mean (bias), SD, 99.9th
   percentile, overall and per position. Bias must be ≤ 0.01% money chance in absolute value.
3. **Agreement (diagnostic):** share of decisions where the hybrid's choice equals exact's or ties
   it exactly.

A rung that fails gets more labels, a larger network or harvested frontier positions (§5), then is
measured again on a fresh validation split; the golden set itself is never trained on.

**Amendment, owner 2026-10-16 (21:00 Mauritius): every situation, found by the machine.** The
position bands above are one grouping among thousands and are no longer the standard. A rung is
proven only when the audit loop of `docs/m3-hybrid/audit-loop.md` ends "flat": on a fresh audit
shard of exactly-labelled positions, no group of weight ≥ 0.1% definable from the encoder's
features (found by the auditor, confirmed on a held-out half with false-discovery control) has a
bias or spread over the calibrated thresholds; the golden set passes item 1 overall and in every
slice the audit names; rare groups get mini golden sets of their own (never trained on). The loop
(train → audit → targeted harvest → retrain) runs autonomously in one rental and reports each
round; its stop reason is part of the rung's record. The bands stay in the reports as one view.

Trick-1 checks for O_3 and O_2 (owner's order): first one known hard world (decision 0, world 3 of
the trick-1 set count 32 seed 1: 79.3×10⁹ nodes, 13,057 s on one thread of an EPYC 7542, exact
value saved), comparing time and every candidate's value; then all saved exact trick-1 worlds:
the 26 of run `m3-t1-tables-1`, the layout runs' worlds, and the 355 of `m3-agree-t1-ab1` (valued
with the rules-bot burn by an older binary: compare them with H_k run with
`SearchConfig::rules_robber_burn = true`, after confirming a sample is byte-identical on the
current exact solver with that switch).

**Owner decision, 2026-10-16 (00:25 Mauritius): no trick-1 golden set.** Trick-1 golden sets are
too expensive and will not be built. Quality is pushed to 0.01 at tricks 4, 3 and 2 (each rung
proven by the audit loop); trick-1 decisions are then made by PIMC over many deals, each deal
searched with H_2 (exact through trick 1, O_2 from the start of trick 2), and trick 1 is accepted as
good enough on that basis. The saved exact trick-1 worlds above (32 decisions, 386 worlds) remain
the only trick-1 measurement: a rough test, reported with its thinness, never a gate. O_2's own
standard is the audit loop at the position level on trick-2 positions plus that rough test.
For trick 1, later improvements are of two kinds only: tricks 2, 3 and 4 pushed below 0.01 (the
loop run further), and more deals per trick-1 decision.

## 4. Golden sets and references

- Trick 4: run `m3-agree-t4c`; trick 2: `m3-agree-t2d` (each 400 decisions × 320 worlds, values
  version 1; copies in `/home/user/41Bots-golden/` on the build VM). Trick 3: the R2 run
  `m3-agree-t3` holds no output (killed 2026-10-14), so the trick-3 golden set (400 decisions ×
  320 worlds, seed 1, as `docs/m3-ncurves/curves-t3.txt`) is regenerated with the current binary
  (about 20 core-minutes: off this VM). At tricks 2–4 no rob is pending and today's fixes
  cannot change a value except through the tie-order merge fix, so the harness recomputes exact
  values on the current binary where that is cheap (tricks 3 and 4, locally) and for trick 2
  checks a 1% sample byte-identical before reusing the stored values (any mismatch: recompute
  off this VM).
- Worlds are regenerated deterministically from (seed, label, decision id, world index) exactly as
  the yardstick does (`search::measure`), never stored twice.

## 5. Labels

- **Positions:** start-of-trick-k positions drawn from three sources, recorded per row:
  (a) random legal play from yardstick decision roots (all positions 1–8, strata S1–S4, many
  scores) to the start of trick k, inside belief-sampled worlds; (b) rules-v1 self-play hands;
  (c) for the second round of each rung, frontier positions harvested from H_k's own searches on
  training decisions (the search visits many lines no sensible player chooses; the network must be
  right there too). Training, validation and golden-set decisions use disjoint seeds.
- **Target:** the exact solver's value of the position (fixed-point four-vector summing to 2³²),
  current binary, documented tie rule. Rows carry the values version, binary sha, equity-table
  sha256 and source.
- **Counts:** O_4 about 1M (seconds), O_3 1–3M (about 9 ms each: a few core-hours), O_2 0.5–1M
  (about 3 s each: hundreds of core-hours, priced with the owner before running).
- **Format:** `.npy` shards with fixed little-endian fields (PLAN §12.7), content-addressed.

## 6. Network, training, inference

- **Encoder:** one encoder in Rust (`crates/nn`, the single source of truth, PLAN §13 layout),
  exported to Python through the shards, never re-implemented. Features cover every field listed
  in §2; a test checks that two positions with equal encodings have equal exact values on a large
  random sample (completeness), and that each symmetry used for augmentation (team rotation; any
  card-suit symmetry that RULES.md allows; prove each with the solver before using it) maps exact
  values as claimed.
- **Model:** a plain MLP (PLAN §13: MLP inference in `crates/nn`), output four logits; loss is
  cross-entropy of the four-way softmax against the exact four-vector (PLAN §6.5 value-head rule).
  Size and depth chosen by validation loss; start small.
- **Training:** Python 3.14 via uv, PyTorch 2.14.1 CPU (never 2.15+), seeded, weights saved as
  safetensors with the training data hashes.
- **Inference (decides the yardstick's reproducibility):** integer-only in Rust, no `unsafe`:
  weights and activations quantised to fixed-point integers with integer accumulators; logits to
  money chances by a fixed integer rule (an integer exponential table and a normalisation that
  sums exactly to 2³² with a documented remainder rule). The quantised net is the one validated in
  §3: its outputs are byte-identical on every machine, thread count and WASM. Quantisation error is
  part of the measured error, never assumed away.
- **Solver hook:** a cut-off at the start of trick k that returns O_k's value (the leaf rule at
  hand end is unchanged); full exact solves (no cut) stay byte-identical to today, proven by the
  existing identity tests. Hybrid values carry a new values version naming the net's sha256.

## 6a. Rung 1b (2026-10-16, after the first small run): the outcome head

Finding of rung 1: even O_5 (one card each, no choice left) had a value error SD of 1.1%, and O_4
3.0% with a bias of −0.16%. The networks are failing to reproduce the match-equity table's
mapping from scores to money, not the card play.

Fact used: in a perfect-information position the exact max^n value is always exactly the leaf value
of one terminal state of the hand (backups only copy child vectors; Review A). So the value is one
of a small set of known vectors: the equity table (or certain money) of each reachable end-of-hand
or game-won state.

Design (replaces the four-logit output of §6):
- **Outcome vocabulary per cut trick k:** a fixed list of hand outcomes from the start of trick k
  (for example: the winning team of each remaining trick, which team scores the 5 of trumps and when,
  which team takes the best-trump bonus, and the game ending mid-hand), chosen so that the
  position plus the outcome determines the terminal match state, which `equity` turns into an
  exact money vector. Outcomes that lead to the same terminal state for a position are merged at
  evaluation time.
- **Label:** the outcome whose exact money vector equals the exact solver's value (first in a fixed
  order when several equal it). A test over every label shard asserts such an outcome always exists:
  that proves the vocabulary is complete.
- **Network:** a classifier over the vocabulary (softmax, cross-entropy on the outcome label, plus
  optionally the soft four-vector loss); value = sum over outcomes of p(outcome) x the exact money
  vector of its terminal state, computed in integers (u128 accumulation, largest-remainder rule,
  sums to exactly 2^32). Outcomes impossible for the position (for example a team with no card that
  can win a trick) may be masked when that is cheap and exact.
- Everything else in §3–§6 stands; the value-error report covers all four teams, not only the
  decider's.

## 7. Run rules

- Anything expected to take more than about 5 minutes on this VM runs elsewhere (vast.ai through
  the runner, or a CPU-only GitHub Actions job under its 6-hour limit). Every rental is priced and
  approved by the owner first.
- Each rung ends with a short report in `docs/m3-hybrid/` (tables of §3) before the next rung
  starts.
