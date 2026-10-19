# Learned exact leaves (build of 2026-10-19)

**Why.** Fix G v1 (`rung2.md` "Exact leaves (G)") solves exactly the trick-3 frontier positions of
hand-picked kinds (race classes, a value band). The owner asked (2026-10-19) for the choice to be
found in the data instead: "make it self-discover its classes, G not defined by us". This build
replaces the hand-picked rule with a **learned** one that carries a **statistical guarantee**,
each part taken from published, proven methods. Nothing was rented.

## The design

**A. A predictor of the network's error** (`nn::leafrisk`, trained by
`python/nn/fortyone_nn/leafrisk.py`). For a frontier position it predicts O_3's expected squared
error, summed over the four teams [how far the network's four money chances are from the exact
ones, squared and added]. This is *learning to defer* [a second model decides when the cheap
answer can be trusted and when to hand the case to the expensive expert] (Geifman & El-Yaniv 2017,
selective prediction; Mozannar & Sontag 2020, learning to defer to an expert); here the cheap
answer is the network and the expert the exact solver.

- **Model:** gradient-boosted regression trees [many small yes/no decision trees added together,
  each correcting the last] (Friedman 2001; row subsampling, Friedman 2002; histogram splits on
  at most 64 bins per input, as LightGBM, Ke et al. 2017). Trees were chosen over a small network
  because every split is a whole-number test and every leaf a whole number: Rust adds them up in
  integers, so the choice of leaves is byte-identical on every machine, thread count and WASM
  (tested against Python on a shared fixture), and trees say which inputs mattered.
- **Inputs:** O_3's own 909 inputs (cards, holders, scores, games, the equity-table values), plus
  13 cheap extras computed from the network's own answer and the position's possible endings: the
  race class, the network's four shares, its top share and top-two margin, its Gini impurity [a
  measure of spread like entropy, computed without logarithms so it stays exact], the distance
  from the network's answer to the nearest and to the farthest possible ending's exact money (the
  exact value is always one of those endings, so the farthest is a hard limit on the error), the
  number of possible endings and of teams that can still win the game, and the widest range of one
  team's money over the endings.
- **Outcome-head entropy** [how unsure the network is between the possible endings] needs the
  network's raw outputs, which the search's leaf interface does not pass on, and recomputing them
  would double the network's cost. It was measured on the same data as a lone score (below) instead
  of being an input; the money-based extras carry the part of that doubt that can move money.
  Passing the outcome probabilities out of the oracle's evaluation (its memo already holds the
  position) would make it a free input; not built here.
- **Ensemble disagreement** (Lakshminarayanan et al. 2017) is not an input: it costs one network
  evaluation per member at every leaf, which on cheap CPUs is memory-bound. The ensemble remains
  available as the leaf evaluator itself (`--net A,B,…`) and the learned rule works with it
  unchanged.

**B. Value of computation.** Russell & Wefald (1991) and Hay, Russell, Tolpin & Shimony (2012)
spend a computation where it can change a decision. If a leaf's value is off by `e`, a comparison
it takes part in flips only when the true gap `g` between the two options is smaller than `|e|`;
with a density `f(0)` of such gaps near zero the expected loss is about `f(0) · e² / 2`. So the
**expected squared error is the value of an exact solve**, up to the sensitivity `f(0)` of the
comparison the leaf enters. The sensitivity is taken as constant (score = predicted error x 1),
for two reasons: (1) a leaf's value must be a function of the position alone (the solvers memoise
it by canonical form, `solver::frontier`), so a sensitivity read from the search path would make
one position's value depend on where the search met it first; (2) the calibration data (harvested
positions) have no search context, so a context-weighted score could not carry the guarantee
below. A search-derived sensitivity is possible later as a two-pass search (a network-only pass
measures how close each decision's candidates are, a second pass spends exact solves only under
close ones); it would be judged by the same paired check.

**C. The threshold, with a guarantee.** A leaf is solved exactly when its score is at least `t`.
The loss of `t` at a position is its squared error when left to the network (0 when solved); it
only grows with `t`. **Learn then Test** (Angelopoulos, Bates, Candès, Jordan & Lei 2021) tests
the thresholds from "solve everything" upwards (fixed-sequence testing, which needs no
multiplicity correction) and keeps the largest `t` whose upper confidence bound on the risk is
within the target, so that P(risk ≤ target) ≥ 1 − δ. **Conformal risk control** (Angelopoulos,
Bates, Fisch, Lei & Schuster 2022) gives the in-expectation version, `(n·R(t) + B)/(n + 1) ≤
target`. Details:

- **The unit is the decision.** Positions harvested from one decision are not independent, and
  decisions are what the hybrid is judged on, so the risk is the mean over decisions of the mean
  loss over the decision's positions. Decisions are recovered as runs of consecutive shard rows
  with the same match (trump, every team's score, flag, game and dealer, read without the
  rotation); neighbouring decisions with the same match merge, which only makes units larger.
- **Bounds:** the empirical Bernstein bound (Maurer & Pontil 2009) is finite-sample and valid for
  any losses in [0, 2] (the most a squared money error can be); its range term is `7·2·ln(2/δ) /
  (3(n−1))`, which is below a tenth of a useful target only from about 500,000 decisions. The
  normal (CLT) bound is asymptotically valid, as Learn then Test allows, and tight at the local
  sample. Both are reported; the local thresholds use the CLT one.
- **The target** is written as the error SD left per team (the auditor's SD: the root of the mean
  squared error over every position and team, with solved positions counting 0): target SD `s`
  is risk `α = 4 s²`.

**Interface.** `search hybrid-check --exact-leaves learned:MODEL.json:t=T` (a whole-number
threshold in units of 2^-40 of a squared money chance; `t=inf` solves nothing),
`learned:MODEL.json:budget=S` (the threshold that solves a share `S` of the model's test rows,
from the model's budget table), combinable with the other terms (`race:4+learned:…`). The normal
form in reports and the values' fingerprint is `learned:<first 12 hex of the model's
SHA-256>:t=T`. The report adds the leaves scored and the scoring time per world. In the runner's
`hybrid_checks` a learned term names its model as a manifest input
(`"exact_leaves": "learned:leafrisk-o3-v1:t=T"`).

**Tests.** Rust: the trees, extras by hand, the budget table, refusals, and the Python fixture bit
for bit (`nn::leafrisk`); learned terms' parsing (`exact_leaves`); in the check, `t=inf` equals no
exact leaves byte for byte, `t=0` equals `all` byte for byte, a split model solves exactly the
leaves it scores at or above `t`, the same on 1 and 3 threads (`hybrid`); the runner's parsing and
command (`checks`) and the committed manifest. Python (`tests/test_leafrisk.py`): whole-number
errors, extras by hand, a planted rule found by the trees, the threshold arithmetic on toy cases
worked by hand (conformal risk control, Learn then Test's walk and stop, both bounds), the
decision units, the fixture.

## What was trained here (this VM, 4 threads)

**Data.** Three fresh trick-3 harvest shards made with the audit shards' recipe (`search labels
--trick 3 --source harvest --harvest-net` O_3 f5766a `--harvest-cap 32 --harvest-positions
0.4,0.3,0.3 --start-hands 20000`; the audit shards themselves were not stored, only the targeted
shards cut from them, which hold the auditor's weak groups only and so would bias both training and
calibration): **train** 400,000 rows (seed 77001, 15,564 decisions), **calibration** 100,000
(seed 77002, 3,559 decision units) and **test** 100,000 (seed 77003, 3,592 units): separate seeds,
so separate decisions. O_3's value on every row is the integer network Rust runs, bit for bit
(`audit.ExactNet`, checked against `intnet`). O_3's error SD on the training rows is 1.72% (the
auditor's 20M-row audit: 1.86%); by race class 0.97%, 0.92%, 1.85%, 1.60%, 3.62%, as the auditor
found. The squared error is very concentrated: **the worst 1% of positions hold 66% of it, the worst
10% hold 96%** (race class 4 holds 57%, class 0 still 12%). Harvest about 35 minutes for the three
shards, O_3's values 4 minutes, training 18 minutes (146 trees kept by the held-out 10% of
decisions, depth at most 6, at least 400 positions a leaf, rate 0.05, half the rows a tree).

**What the predictor learned** (share of the trees' total split gain): the distance from O_3's
answer to the nearest possible hand ending 42%, the number of possible endings 12.5%, the distance
to the farthest ending 5.4%, then the two-trick successor values of the equity table (2.8%, 2.1%,
2.1%, 1.8%, 1.8%, 1.6%, 1.5%: "the next team's money after the opposite team wins the next two
tricks" and similar: who can still reach 41 within two tricks), the widest money range of one
team 2.6% and O_3's Gini impurity 2.6%. In plain words: **O_3 is wrong where it hedges between
endings that pay different money, and where many endings are still open**; the race classes are a
coarse version of the second half of that, and the data found the first half, which the race
classes cannot see (class 0 positions where O_3 hedges are as risky as class 4 ones where it is
sure). The outcome head's entropy alone ranks far worse than the model while a fifth of the
positions or fewer are solved, but better from about 30%: at half solved it leaves 0.6% of the
error against the model's 1.4% (below). Giving the model that entropy as an input is the first
improvement to try (it needs the network's raw outputs at each leaf, design A).

**Risk and coverage on the test split** (share of positions solved, highest scores first; share
of O_3's squared error left; error SD left per team):

| solved | learned: error left | SD left | best possible order (by the true error) | race class alone | outcome entropy alone |
|---|---|---|---|---|---|
| 5% | 25.3% | 0.96% | 8.8% | 73.1% | 77.1% |
| 10% | 16.6% | 0.78% | 2.9% | 53.1% | 61.2% |
| 20% | 8.9% | 0.57% | 0.3% | 34.0% | 36.5% |
| 30% | 4.7% | 0.42% | 0.0% | 20.6% | 14.4% |
| 50% | 1.4% | 0.23% | 0.0% | 12.0% | 0.6% |

| G v1 policy | solved | error left | SD left | learned at the same share: error left | SD left |
|---|---|---|---|---|---|
| race:4 | 13.1% | 37.6% | 1.18% | 13.5% | 0.70% |
| race:2,4 | 27.6% | 19.2% | 0.84% | 5.5% | 0.45% |
| race:1-4 | 61.8% | 9.6% | 0.59% | 1.0% | 0.19% |

(No SD left with nothing solved: 1.92% on the test split.)

**Thresholds with a guarantee** (calibration split, 3,559 decisions, δ = 0.05; the risk per
decision; then checked on the test split):

| target SD left | Learn then Test (CLT): threshold | solves (test) | SD left (test) | conformal risk control | empirical Bernstein |
|---|---|---|---|---|---|
| 1.0% | 4,800,888,675 | 4.9% | 0.95% | none | none |
| 0.8% | 2,050,323,033 | 10.5% | 0.76% | none | none |
| 0.6% | 1,094,141,811 | 19.4% | 0.57% | none | none |
| 0.5% | 781,086,922 | 25.2% | 0.48% | none | none |
| 0.4% | 464,749,693 | 32.3% | 0.37% | none | none |

Every Learn-then-Test threshold held on the test split. At this sample the finite-sample bounds
cannot certify any useful target: conformal risk control's term `2/(n+1)` alone is 0.00056 (an SD
of 1.18%), and the Bernstein range term is larger; both need hundreds of thousands of decisions
(a rented calibration harvest, below). The CLT thresholds are asymptotically valid, as Learn then
Test allows; the whole calibration report is `data/leafrisk/leafrisk-o3-v1.calibration.json`.

**The model** is `data/leafrisk/leafrisk-o3-v1.json` (SHA-256 `20c41678c646…`, 146 trees, the test
split's budget table included), stored in R2 as input `inputs/sha256/20c41678…`.

**To reproduce** (equity table v0, O_3 f5766a as `o3.safetensors`; each step is deterministic):

```
search labels --trick 3 --count 400000 --seed 77001 --source harvest --out h-train.npy \
  --equity equity.npy --threads 4 --start-hands 20000 --harvest-cap 32 \
  --harvest-positions 0.4,0.3,0.3 --harvest-net o3.safetensors      # and 100000 at 77002, 77003
cd python/nn
uv run python -m fortyone_nn.leafrisk prepare --net o3.safetensors --shard h-train.npy --out train.npz
uv run python -m fortyone_nn.leafrisk train --data train.npz --out model.json      # 150 trees, seed 1
uv run python -m fortyone_nn.leafrisk calibrate --model model.json --cal cal.npz --test test.npz \
  --out calib.json --write-budget --targets 1.6,1.4,1.2,1.0,0.9,0.8,0.7,0.6,0.5,0.4
search hybrid-check --trick 2 --cut 3 --net o3.safetensors --equity equity.npy \
  --decisions golden-decisions.jsonl --worlds golden-worlds.jsonl --verify-pct 0 --first 0 \
  --last 31 --to 32 --seed 1 --threads 4 --exact-leaves learned:model.json:budget=0.40
```

## In the hybrid (golden decisions 0-31 x 32 worlds, this VM)

The learned policy picks a smaller share of the hybrid's leaves than of harvest rows at the same
threshold (budget 0.40 picks 35% of the hybrid's distinct leaves), and its leaves are dearer to
solve (they are the open, uncertain ones):

| policy | leaves exact | ms a solve | thread-ms a world | mean loss | 95% bound | own bias | value SD |
|---|---|---|---|---|---|---|---|
| none | 0% | - | 27.7 | 0.0101% | 0.0241% | -0.127% | 2.41% |
| G v1 race:1-4 | 55.2% | 2.63 | 542 | 0.0044% | 0.0132% | -0.028% | 0.61% |
| learned t=277338260 (budget 0.40) | 35.0% | 3.93 | 528 | 0 | 0 | -0.005% | 0.27% |
| learned t=1057825477 (budget 0.20) | 14.6% | 4.59 | 273 | 0.0287% | 0.0653% | -0.029% | 0.59% |
| learned t=12850753 (budget 0.618) | 65.2% | 3.38 | 824 | 0 | 0 | +0.004% | 0.10% |

Scoring every leaf costs about 8-10 thread-ms a world (25 µs a leaf). Paired comparisons (1,000
bootstrap replicates): **learned at race:1-4's cost against race:1-4**: loss -0.0044% (interval
-0.0132% .. 0), own bias +0.023% (-0.016% .. +0.063%), value SD 0.27% against 0.61%; against none:
loss -0.0101% (-0.0224% .. 0), own bias +0.122% (+0.007% .. +0.264%). At half the cost the loss
on these 32 decisions is worse than none (+0.0186%, interval -0.0070% .. +0.0505%), though the
own-team bias is cut as much as by race:1-4: solving a few of a node's leaves exactly while their
siblings keep O_3's noise can move a comparison either way. Thirty-two decisions cannot judge any
of this; the rented run can.

## The rented run (`docs/vast/o3-learnedleaf.manifest.json`, run `o3-learnedleaf-1`, not launched)

The same stored golden set (2,000 decisions x 320 worlds, the same inputs as fix G v1's run), the
adopted O_3: (a) none, (b) G v1 race:1-4, (c) learned at race:1-4's thread time
(`learned:leafrisk-o3-v1:t=277338260`), (d) learned at half of it (`t=1057825477`); (b), (c), (d)
against (a) and (c) against (b) by the paired comparison. On this VM the four checks cost 1.37
thread-seconds a world-set (27.7 + 542 + 528 + 273 ms); with the rented machines' 2.4x slower
threads about 2.1 M thread-seconds: about 2.4 hours on 245 threads, 4.6 on 128 (est_seconds
16,000).

**Training needs no rented job** (18 minutes here). **A finite-sample guarantee would**: about
500,000 decisions in the calibration split, a harvest like the auditor's 20M-row shard (778,000
decisions; 864 s on the audit machine) and O_3 on its rows (145 s there), under an hour of a
190-thread machine. No runner kind does "harvest, then `leafrisk prepare` and `calibrate`" yet;
that is the next build if the rented check adopts the learned policy.
