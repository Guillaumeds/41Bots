# 41Bots documents: what each is, and what we learned

Start every session with `CLAUDE.md`, then `docs/STATUS.md` (where we are, what is running, what is
next), then this page for anything older. **`docs/RESEARCH.md` is the collected research record:**
every experiment's question, method, result in numbers, cost and limits, in one place (section
numbers below, e.g. C7, point into it). Old branches are tags: `git tag -l 'archive/*'`, and
`git checkout archive/<name>` restores one (archived 2026-10-21 by `repo-archive.yml`). Stored data
and networks: the table in `docs/STATUS.md` and each experiment's document; everything in R2 is
content-addressed (`inputs/sha256/<hex>`), and nothing expensive to remake is ever deleted.

Status words: **adopted** (in use), **done** (a finished build or measurement), **rejected**
(tested, not used), **open** (built or run, verdict not yet written down), **superseded**, **not run** /
**never launched** / **incomplete** (no result exists).

## Foundations (all done)
| document | what | outcome |
|---|---|---|
| `PLAN.md`, `REVIEW-2026-10-13.md` | the approved plan and its review (amendments at the top of PLAN) | **adopted**: the plan |
| `RULES.md`, `TRANSCRIPTS.md` | the authoritative rules, the owner's house rules (H), the owner-approved transcripts | **adopted** |
| `M0-SPEC.md` | engine, independent reference, RNG, differential test | **done**: 0 disagreements in 10M matches |
| `M1-SPEC.md` | fixed-point money chance, exact match-equity table, arena, launcher | **done** |
| `M2-SPIKE.md`, `m2-epyc/` | exact solver speed | **done**: cheap from trick 3; trick 1 7-140 s a world on 384 cores |
| `M3-SPEC.md` | card tracker, world generator, PIMC teacher | **done** |
| `m3-agree-t1-*` runs | trick-1 N-curve (RESEARCH A7) | **incomplete**: runs killed by the old cleanup job; 31 + 355 saved exact trick-1 worlds remain |
| `m3-ncurves/` | the early-trick yardstick's N-curves (RESEARCH A6) | **done**: loss ∝ N^-0.75; N for 0.01% about 240 (t2), 180 (t3), 60 (t4); rollout PIMC 13× worse than exact, so exact PIMC early |
| `VAST-MACHINES.md`, `vast/market-*.json`, `vast/measured-*.json` | rented machines measured | **done**: price work from rented measurements only (rented threads 3-4.6× slower than the build VM) |
| `ENGINEERING-2026-10-17.md`, `m3-hybrid/nn-speed.md`, `m3-hybrid/overlap.md`, `m3-hybrid/vast.md` | run operations, network speed, CPU/GPU overlap, the runner on vast | **adopted** |
| `APP-ENGINE-CONTRACT.md`, `WASM-API.md` | the engine API for the app | **done** |

## The hybrid networks (O_k values the start of trick k)
| document | what | outcome |
|---|---|---|
| `M3-HYBRID-SPEC.md` | the hybrid yardstick: exact search, then a network | **adopted** |
| `m3-hybrid/rung1.md` | O_4 (trick 4) | **adopted**: 0.0016% (95% bound) on 4,000 golden decisions |
| `m3-hybrid/rung2.md` | O_3 (trick 3): rounds, harvest, adopted f5766a | **adopted**: O_3 f5766a, 0.0068% (seat 1 0.0201%); plain harvest at cap 32 did **not** help |
| `m3-hybrid/audit-loop.md` | find weak groups automatically, train them away (multiaccuracy) | **superseded** for O_3 by G2-guided rounds: the confirmed groups covered almost everything, so "flat" was out of reach; its additive mode and stored-shard reuse are **adopted** |
| `m3-hybrid/rung2.md` "Exact leaves (G)" | G v1: solve the race-class leaves exactly | **superseded** by G v2 (0.0018%) |
| `m3-hybrid/learned-leaves.md` | **G v2**: a learned predictor of O_3's error picks the leaves to solve | **adopted**: O_3 + G2 0.0008% (seat 1 0.0022%) at race:1-4's cost |
| `m3-hybrid/curse-ensemble.md` | optimizer's-curse test (RESEARCH C7) | **done**: zero-mean noise alone at O_3's error SD gave own-team bias −0.33% (seat 1 −0.69%), loss 0.0157% / bound 0.0182%: same sign pattern as real H_3, about 20-40× larger |
| `m3-hybrid/curse-ensemble.md` | 4-network O_3 ensemble (RESEARCH C8) | **not measured**: stopped twice, 100× slower per world on the EPYC 7B12 (weights exceed cache); no ensemble number |
| `m3-hybrid/sibrank.md` | sibling groups and a pairwise ranking loss (RESEARCH C9) | **not run at scale**: rented A/B never launched; build-VM A/B (42k rows): B worse (misordered pairs 18.3% vs 15.6%) |
| `vast/o3-base80.manifest.json` | O_3 at 80M base, run `o3-base80-1` (RESEARCH C4) | **done, not adopted**: value SD 1.85 → 1.76% on the golden frontier, decision loss not better (bound 0.0080% vs 0.0068%, seat 1 0.0264% vs 0.0201%) |
| `vast/o3-basecurve-*.manifest.json` | O_3 on 50% vs 100% of the 40M base (RESEARCH C5) | **not run**: both cancelled 37 s after start (2026-10-19); no result |
| `vast/o3-size-*.manifest.json` | wider and deeper O_3 (RESEARCH C6) | **never launched**; size tested elsewhere (C2 rounds 3-4, C12 round 2 c3: no better) |
| `m3-hybrid/audit-loop.md` §10, run `m3-hybrid-t3-additive-2` | additive O_3 loop (RESEARCH C3) | **done, not adopted**: round 2 bound 0.0071%, own-team bias −0.0121% (fails); stopped by owner 2026-10-19 |
| `m3-hybrid/o3-rounds.md` | **O3G2 rounds**: G2 chooses O_3's new training positions (RESEARCH C12) | **method adopted** 2026-10-21; rounds 1-2 done: no candidate below 0.0069%; round 1 kept f5766a, round 2 adopted c2 (1057e9e9, 0.0070%, within noise of f5766a) |
| `m3-hybrid/specialist.md` | specialist S on G2-flagged positions (RESEARCH C13) | **rejected** 2026-10-22: routed check 0.0079% (fails bias); S no better than O_3 on the flagged rows (SD 3.03% vs 2.94%) |
| `m3-hybrid/rung3.md` | O_2 (trick 2), provisional 505a0c (RESEARCH D1) | **adopted as provisional**: position error SD 3.6%; trick-2 loop-2 rounds 1-3 no better (int SD 3.66 → 3.73 → 3.85%), stopped 2026-10-19 |
| `m3-hybrid/audit-loop.md` §12 | hybrid labels: exact through trick k, then O_{k+1} (TreeStrap) | **adopted** for O_2's labels, after a label test; the label test (`o2-labeltest-1`) was cancelled before a result (RESEARCH D2) |
| `m3-hybrid/learned-leaves.md` "O_2 G v2" | learned deferral one trick earlier, run `o2-leafrisk-1` (RESEARCH D3) | **predictor done**: O_2's error less concentrated than O_3's (10% solved leaves 36.6% of it vs 16.6%; 40% leaves 11.6% vs 2.2%); trick-1 decision test not run |
| `m3-hybrid/silver.md` | trick-1 reference (exact through trick 2, then O_3) (RESEARCH D4) | **built, never completed**: `o2-silver-1` failed at 143 of 32,000 worlds; later attempts were part of the discarded Copilot work; no silver set exists. Its O_3-only form is the **bronze set** plan |
| STATUS history 2026-10-20 | O_2 on G v2 labels | **rejected**: G v2 labels only 1.2× cheaper than exact at trick 2 |

## Students (fast networks for the app)
| document | what | outcome |
|---|---|---|
| `STUDENT.md` | trick-2 student from the teacher's labels (RESEARCH E1-E5) | **done**: 0.057% alone at 1M decisions; top-3 shortlist + search 0.0073%; more data alone too slow (loss ∝ N^-0.23), more width no help |
| `STUDENT.md` §13 | deals per decision at equal teacher cost (RESEARCH E3) | **adopted**: 64 deals (128 deals × 315k best, 0.0695%) |
| `STUDENT.md` §14 | split-half ruler and top-k shortlist (RESEARCH E4) | **done**: test-set floor 0.008%; student gap 0.051%; top-3 0.005% full / 0.012% split |
| run `student-t1-cost-2` | trick-1 teacher cost, H_2 with O_2 at 64 deals (RESEARCH E6) | **done**: 12,489 decisions an hour on 256 EPYC 7B13 threads ($1.05/h), about $84 per million |

## Reviews and logs (history)
`reviews/REVIEW-A.md`, `-B.md`, `-E.md`, `REVIEW-C.md`, `REVIEW-D.md` (code reviews of 2026-10-15);
`plan-log-2026-10.md` (the working plan log, 2026-10-16 to 20); `history/STATUS-history.md` (every
dated status entry, 2026-10-13 to 20). The Copilot work of 2026-10-19/20 (branch `silver-shared`
and its `sx50` rentals) was discarded by the owner; its branch is `archive/silver-shared`.
