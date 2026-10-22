# 41Bots documents: what each is, and what we learned

Start every session with `CLAUDE.md`, then `docs/STATUS.md` (where we are, what is running, what is
next), then this page for anything older. Old branches are tags: `git tag -l 'archive/*'`, and
`git checkout archive/<name>` restores one (archived 2026-10-21 by `repo-archive.yml`). Stored data
and networks: the table in `docs/STATUS.md` and each experiment's document; everything in R2 is
content-addressed (`inputs/sha256/<hex>`), and nothing expensive to remake is ever deleted.

Status words: **adopted** (in use), **done** (a finished build or measurement), **rejected**
(tested, not used), **open** (built or run, verdict not yet written down), **superseded**.

## Foundations (all done)
| document | what | outcome |
|---|---|---|
| `PLAN.md`, `REVIEW-2026-10-13.md` | the approved plan and its review (amendments at the top of PLAN) | **adopted**: the plan |
| `RULES.md`, `TRANSCRIPTS.md` | the authoritative rules, the owner's house rules (H), the owner-approved transcripts | **adopted** |
| `M0-SPEC.md` | engine, independent reference, RNG, differential test | **done**: 0 disagreements in 10M matches |
| `M1-SPEC.md` | fixed-point money chance, exact match-equity table, arena, launcher | **done** |
| `M2-SPIKE.md`, `m2-epyc/` | exact solver speed | **done**: cheap from trick 3; trick 1 7-140 s a world on 384 cores |
| `M3-SPEC.md` | card tracker, world generator, PIMC teacher | **done** |
| `m3-ncurves/` | the early-trick yardstick's N-curves | **done**: loss ∝ N^-0.75; N for 0.01% about 240 (t2), 180 (t3), 60 (t4); rollout PIMC 13× worse than exact, so exact PIMC early |
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
| `m3-hybrid/curse-ensemble.md` | optimizer's-curse test and 4-network ensembles | **open**: built and run; result not written down |
| `m3-hybrid/sibrank.md` | sibling groups and a pairwise ranking loss | **open**: A/B built; small test said B worse |
| `vast/o3-base80`, `o3-basecurve-*`, `o3-size-*` manifests | O_3 at 80M base; 50% vs 100% base; wider and deeper O_3 | **open**: `o3-base80-1` ran, result not written down; `o3-basecurve-50/100` (the 20M-row O_3) were **cancelled at the start on 2026-10-19, never trained: no result**; `o3-size-*` not checked |
| `m3-hybrid/o3-rounds.md` | **O3G2 rounds**: G2 chooses O_3's new training positions; candidates, winner, next G2 each round | method **adopted** 2026-10-21; **result: rounds 1 and 2 (2026-10-21/22) did not improve O_3** (best candidate 0.0070% against 0.0068%, within noise; a wider network from scratch 0.0079%); O_3 f5766a stays |
| `m3-hybrid/specialist.md` | **Specialist S**: a network trained only on the trick-3 positions G2 flags, used instead of the exact solve in play | **rejected for now** (2026-10-22): S is at O_3's level on those rows (SD 3.03% against 2.94%); routed in play 0.0079% against 0.0008% for the exact solve and 0.0068% for O_3 alone |
| `m3-hybrid/rung3.md` | O_2 (trick 2), provisional 505a0c | **adopted as provisional**: position error SD 3.6% |
| `m3-hybrid/audit-loop.md` §12 | hybrid labels: exact through trick k, then O_{k+1} (TreeStrap) | **adopted** for O_2's labels, after a label test |
| `m3-hybrid/learned-leaves.md` "O_2 G v2" | learned deferral one trick earlier | **open**: built; predictor run launched 2026-10-20; result not written down |
| `m3-hybrid/silver.md` | trick-1 reference (exact through trick 2, then O_3) | **done** (built); its O_3-only form is the **bronze set** plan |
| STATUS history 2026-10-20 | O_2 on G v2 labels | **rejected**: G v2 labels only 1.2× cheaper than exact at trick 2 |

## Students (fast networks for the app)
| document | what | outcome |
|---|---|---|
| `STUDENT.md` | trick-2 student from the teacher's labels | **done**: 0.057% alone at 1M decisions; top-3 shortlist + search 0.0073%; more data alone too slow (loss ∝ N^-0.23), more width no help |

## Reviews and logs (history)
`reviews/REVIEW-A.md`, `-B.md`, `-E.md`, `REVIEW-C.md`, `REVIEW-D.md` (code reviews of 2026-10-15);
`plan-log-2026-10.md` (the working plan log, 2026-10-16 to 22); `history/STATUS-history.md` (every
dated status entry, 2026-10-13 to 20). The Copilot work of 2026-10-19/20 (branch `silver-shared`
and its `sx50` rentals) was discarded by the owner; its branch is `archive/silver-shared`.
