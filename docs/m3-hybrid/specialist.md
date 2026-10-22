# Specialist S: a network trained only on G2's risky trick-3 positions (2026-10-21/22)

**Outcome: rejected for now (no gain).** S learned the flagged positions about as well as O_3
already values them (slightly worse), and routing G2's flagged positions to S in play scored
0.0079% on the golden set against 0.0008% for the exact solve and 0.0068% for O_3 alone. The
method, the code and every number are below.

## The question (owner, 2026-10-21)
G2 [the learned predictor of where O_3 is wrong, `learned-leaves.md`] flags the trick-3 positions
that play solves exactly today (a position is solved when its G2 score is at least
`t = 277338260`, about 40% of the positions). What if a network S, trained only on those risky
positions, valued them instead of the exact solver? In play: G2 decides which network answers each
frontier position: flagged goes to S, everything else to O_3. If S were close to exact there, the
exact solves (586 ms a world) would be replaced by a network call (43 ms a world).

## Method
1. **Filter the stored base** (`search risk-filter`): the 40M trick-3 base (random 25.6M s321,
   rules 6.4M s322, harvest 8M s313) is decoded row by row (`nn::encode::decode`), O_3 values the
   position, G2 scores it (`nn::leafrisk`), and a row is kept when its score is at least
   `t = 277338260`, the play rule's own cut. Natural mix inside the flagged share (no re-weighting).
   Labels unchanged. Held-out set: the stored 100k validation shard (s401), filtered the same way.
2. **Train S from random weights** on the flagged rows only (`fortyone_nn.train`, 2048-1024-256,
   outcome head, 20 epochs, lr 0.001, batch 1024, seed 1, early stopping with patience 4).
   Nothing from the unflagged rows, nothing from the O3G2 rounds.
3. **Held-out error** of S on the flagged validation rows (trainer's own report), and **O_3's error
   on the same rows** (`search net-error`, written for this comparison).
4. **Routed golden check** (`hybrid_checks`, `defer_expert: "s:<S>"`): the trick-2 golden set (2,000
   decisions x 320 exact worlds), H_3 with O_3, three checks in one run: O_3 alone, G2 with exact
   solving (today's best), and routed (G2 flags the leaves, S values them directly with no search,
   O_3 values the rest).

## What was built (all on `main`; tests pass)
- `search risk-filter` (`crates/search/src/risk_filter.rs`): score a stored shard with G2, keep the
  rows at or above a threshold (`--threshold T`, or `--step K` of the model's budget table).
  Thread-independent. Tests: thread independence, order, the cut, a bad row.
- The expert form `s:NET` (`search::exact_leaves`, `Expert::Specialist`): a picked leaf is valued by
  one network evaluation at the leaf's own trick, no search ahead; the network must be a trick-3
  network for trick-3 leaves (`h3:NET` is the deeper-hybrid form and would need an O_4). Accepted by
  `hybrid_checks` as `defer_expert`.
- Runner kind `specialist` (`crates/runner/src/specialist.rs`): filter each stored shard, store
  nothing but the network, train S, record. Runner kind `net_error` (`crates/runner/src/net_error.rs`)
  around `search net-error` (`crates/search/src/net_error.rs`): the network's value (rotated into the
  encoder's team order, minus the row's exact value, per team) on all rows and on the flagged rows.
- Workflow `store-input.yml`: put one trained network from a finished run's artifact into R2 as
  `inputs/sha256/<hex>`, hash-checked. `r2-peek.yml` gained a `show` input (print named small files).
- Manifests: `docs/vast/specialist-s1.manifest.json` (run `specialist-s2`),
  `golden-routed-s2.manifest.json`, `net-error-o3-val.manifest.json`, `speedtest-*.manifest.json`.
- New chunk kinds must be registered in four places (found the hard way, 2026-10-22): the scheduler's
  `KINDS` (`sched.rs`) and its input-name check, `v2.rs` resolve, `mem.rs`, the dispatch in
  `main.rs`; and the manifest listed in `v2::tests::manifests_resolve_...`. The first rental
  (`specialist-s1`) was refused by the machine's manifest check because `KINDS` lacked the kind.

## Results

### Filter and training (run `specialist-s2`, 2026-10-22)
Machine: AMD EPYC 9B14, 184 usable threads, RTX 5090, $1.50/h, about 22 minutes (about $0.55).

| shard | rows | flagged | share |
|---|---|---|---|
| base random | 25,600,000 | 11,077,690 | 43.3% |
| base rules | 6,400,000 | 2,235,652 | 34.9% |
| base harvest | 8,000,000 | 3,162,386 | 39.5% |
| validation (held out) | 100,000 | 43,681 | 43.7% |

16,475,728 training rows. Training took 8.8 minutes on the card. S float on the 43,681 held-out
flagged rows: outcome accuracy 75.1%, val loss 1.771. S integer (what play uses) agrees with the float
network to an SD of 3e-8. Stored: S int `d30ef4963658bc3fab186b71eeb69e766a38780d05637d3bda74d7e7076d590c`,
S float `f5f6768da729fffd55116b4597be7203cb90de3444272f3ca7c0d10908c1055a` (the int file is in R2
`inputs/sha256/`; the run's output archive also holds both).

### Held-out error: S against O_3, the same 43,681 flagged rows (run `net-error-o3-val-1`)
| | O_3 (int) | S (float, trainer's report) |
|---|---|---|
| SD of the value error (rows x teams) | **2.94%** | 3.03% |
| mean absolute error | **1.08%** | 1.09% |
| 99.9th percentile absolute error | **31.2%** | 34.0% |
| largest absolute error | 99.9% | 99.9% |
| largest team bias | 0.033% | 0.027% |
| SD, race class 4 ("a team can win the match and another the game") | **5.0%** | 5.1% |

O_3 over all 100,000 validation rows: SD 1.96%, mean 0.49%, 99.9th percentile 24.2%. So the flagged
rows are the hard ones, and S is at O_3's level on them, not above. Caveats: S's numbers are the
trainer's report (its definitions may differ slightly from `net-error`'s); O_3's outcome accuracy was
not measured.

### Routed golden check (run `golden-routed-s2`, 2026-10-22)
Machine: 9B14, 184 usable threads, $1.35/h, about 41 minutes (about $1). Golden set 635e9aef /
6ddc59b1, 2,000 decisions x 320 worlds, H_3 with O_3 f5766a.

| check | mean loss | 95% bound | seat 1 bound | ms a world | verdict |
|---|---|---|---|---|---|
| O_3 alone | 0.0056% | 0.0068% | 0.0201% | 24 | fails seat 1 |
| G2 + exact solving (today's) | 0.0005% | **0.0008%** | 0.0022% | 586 | **passes** |
| routed: G2 flags, S values, O_3 the rest | 0.0064% | 0.0079% | 0.0216% | 43 | fails seat 1 and bias |

Paired: routed against G2+exact is +0.0059 points (95% interval +0.0044 to +0.0075), clearly worse;
routed against O_3 alone is +0.0008 (interval -0.0006 to +0.0024), no measurable difference. Own-team
bias of routed: 0.0179% overall, 0.0411% in positions 1 and 2 (limit 0.01%), the only check to fail
the bias limit. Speed: 14x faster than the exact solve, 1.8x O_3 alone.

## Reading
- S does not replace the exact solve. The exact solve removes about 88% of O_3's loss (0.0068% to
  0.0008%); S removes none of it.
- Why: the flagged positions are the ones O_3 itself is worst at; a network of the same shape trained
  on 16M of them reaches O_3's level and no better. The error there is not noise that more positions
  of the same kind obviously cure: three different attempts (O3G2 rounds with extra G2-chosen rows,
  a bigger network from scratch, and S) all land at O_3's level.
- Pushing S to the full 40M rows (the owner's question, 2026-10-22) is not supported by this: S is
  already at O_3's level on its own rows, and the routed check showed no gain from that level. The
  data-versus-capacity test that would settle it (O_3 trained on 20M rows against 40M, golden set) was
  prepared on 2026-10-19 (`o3-basecurve-50/100`) and **never ran** (both cancelled at the start).

## Not done / open
- The 20M-row O_3 (basecurve) and the 80M-base O_3 (`o3-base80-1`, result never written down).
- O_3's outcome accuracy on the flagged rows; S as an ensemble member for the bronze set.
- A faster routed check is possible if S is ever worth keeping (the check took 151 s against 2,042 s
  for the exact one).
