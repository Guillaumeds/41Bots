# 41Bots documents: where everything is

**Read in this order:** `CLAUDE.md` (working rules) → `docs/STATUS.md` (where we are, what is
running, what is next) → this page. Then, as needed:

| record | holds |
|---|---|
| `RULES.md`, `TRANSCRIPTS.md` | the authoritative rules and the owner's house rules (H); owner-approved transcripts |
| `PLAN.md` (amendments at the top), `REVIEW-2026-10-13.md` | the approved design and the review that amended it |
| **`RESEARCH.md`** | the outside research: papers, other programs and the game family, with their numbers and where each is used |
| **`EXPERIMENTS.md`** | every experiment we ran: question, method, result, cost, limits, files (sections A-F, cited below) |
| this page | one line per document and experiment, with its outcome |

Status words: **adopted** (in use), **done** (a finished build or measurement), **rejected** (tested,
not used), **superseded**, **open** (no verdict yet), **not run** / **never launched** /
**incomplete** (no result exists).

Data and networks live in R2, content-addressed (`inputs/sha256/<hex>`); the current keys are in
`STATUS.md`, every run's files under `runs/<run_id>/`, O3G2 round records under `rounds/<id>/`.
Old branches are tags (`git tag -l 'archive/*'`; `git checkout archive/<name>` restores one).

## Specifications and engineering (built)
| document | what | outcome |
|---|---|---|
| `M0-SPEC.md` | engine, independent reference, RNG, differential test | **done**: 0 disagreements in 10M matches (EXPERIMENTS A1) |
| `M1-SPEC.md` | fixed-point money chance, exact match-equity table, arena, launcher | **done** (A2) |
| `M2-SPIKE.md`, `m2-epyc/` | exact solver and its cost | **done**: cheap from trick 3, seconds at trick 2, very costly at trick 1 (A3) |
| `M3-SPEC.md` | card tracker, world generator, PIMC teacher, runner | **done** (A5); its PIMC-suitability measurement (§6) and rules-v1 gate (§8.3) not run |
| `M3-HYBRID-SPEC.md` | the hybrid yardstick: exact search to trick k, then a network O_k | **adopted** |
| `ENGINEERING-2026-10-17.md`, `m3-hybrid/nn-speed.md`, `m3-hybrid/overlap.md` | answer-identical speed work | **adopted** (C14) |
| `m3-hybrid/vast.md`, `VAST-MACHINES.md`, `vast/` | running on rented machines; **measured cost per unit of work of every machine we rented** (`VAST-MACHINES.md`, first section); run manifests | **adopted** (A8). `vast/*.manifest.json` are the runs' inputs and are checked by the runner's tests |
| `APP-ENGINE-CONTRACT.md`, `WASM-API.md` | the engine API for the app | **done** |
| `STUDENT.md` | the student pipeline | **done** (E1) |

## Measurements of the game and the solver
| document | what | outcome |
|---|---|---|
| `m3-ncurves/` | worlds needed per decision at tricks 2-4 | **done**: N for 0.01% about 240 / 180 / 60; rollout PIMC about 13× worse than exact at trick 2 (A6) |
| runs `m3-agree-t1-*` | the same curve at trick 1 | **incomplete** (A7) |
| `t1bench-*`, `m3-t1-tables-1` | trick-1 exact cost on rented machines | **done** (A4); no trick-1 golden set is built (owner, 2026-10-16) |

## The trick-4 network O_4
| document | what | outcome |
|---|---|---|
| `m3-hybrid/rung1.md` | first networks, outcome head, scaling sweep, the race-for-the-money diagnosis, lead-4, shapes | **done** (B1-B4) |
| `m3-hybrid/audit-loop.md` §8 | trick-4 audit loop | **adopted**: O_4 b28750e3, 0.0016% on 4,000 decisions (B5) |

## The trick-3 network O_3
| document | what | outcome |
|---|---|---|
| `m3-hybrid/rung2.md` | first rounds; adopted O_3 f5766a | **adopted**: f5766a, 0.0068%, seat 1 0.0201% (C1-C2) |
| `m3-hybrid/audit-loop.md` | find weak groups automatically, train them away | **superseded** for O_3: "flat" out of reach (C2); additive mode tried, not adopted (C3) |
| run `o3-base80-1` | O_3 on 80M rows | **done, not adopted**: lower value error, decisions not better (C4) |
| `vast/o3-basecurve-*` | O_3 on 50% vs 100% of the base | **not run** (C5) |
| `vast/o3-size-*` | wider / deeper O_3 | **never launched** (C6) |
| `m3-hybrid/curse-ensemble.md` | optimizer's-curse test; 4-network ensemble | curse test **done** (C7); ensemble **not measured** (C8) |
| `m3-hybrid/sibrank.md` | sibling ranking loss | **never launched** at scale (C9) |
| `m3-hybrid/rung2.md` "Exact leaves (G)" | G v1: race-class leaves solved exactly | **superseded** by G v2 (C10) |
| `m3-hybrid/learned-leaves.md` | **G v2**: learned predictor of O_3's error picks leaves to solve | **adopted**: O_3 + G2 0.0008%, seat 1 0.0022% (C11) |
| `m3-hybrid/o3-rounds.md` | O3G2 rounds: G2 chooses new training positions | method **adopted** 2026-10-21; rounds 1-2 did not improve O_3 (C12) |
| `m3-hybrid/specialist.md` | specialist S on G2-flagged positions | **rejected** 2026-10-22 (C13) |

## The trick-2 network O_2 and trick 1
| document | what | outcome |
|---|---|---|
| `m3-hybrid/rung3.md` | O_2 provisional 505a0c; trick-2 loop | **adopted as provisional**; loop rounds 1-3 no better (D1) |
| `m3-hybrid/audit-loop.md` §12 | trick-2 labels from O_3 (TreeStrap) | **adopted** in design; the label test never ran (D2) |
| STATUS history 2026-10-20 | O_2 on G v2 labels | **rejected**: only 1.2× cheaper than exact (D2) |
| `m3-hybrid/learned-leaves.md` "O_2 G v2" | O_2's error predictor | predictor **done**; trick-1 test **not run** (D3) |
| `m3-hybrid/silver.md` | trick-1 reference (exact through trick 2, then O_3) | built, **incomplete**: no silver set exists (D4) |

## Students (fast networks for the app)
| document | what | outcome |
|---|---|---|
| `STUDENT.md` §11 | trick-2 student, learning curve | **done**: 0.057% alone at 1M decisions (E2) |
| `STUDENT.md` §13 | deals per decision | **adopted**: 64 deals (E3) |
| `STUDENT.md` §14 | split-half ruler, top-k shortlist | **done**: top-3 shortlist about 0.005-0.012% (E4-E5) |
| run `student-t1-cost-2` | trick-1 teacher cost | **done**: about 12,500 decisions an hour on 256 threads (E6) |
| `m3-hybrid/path-forward.md` | **proposed path** from the O3G2 rounds to shipped students: free reads, S2 A/B, O_2 label test, trick-1 yardsticks (copper, playout), S1, then improvements | **open**: path approved in outline (2026-10-22); section 6 holds a brief for each of the six sessions; E8 is its first measurement |
| run `student-check --sub-deals` | shortlist + search at the app's deal counts (25-100) on the exact golden set | **done** (session 0a, E9): top-3 + search beats S2 alone from about 25 deals (0.0733% to 0.0596 / 0.0402 / 0.0293 at 25 / 50 / 100), loses at 10; top-3 is hard to tell from search-all because trick-2 decisions have only 2-4 candidates; top-2 is the real shortlist |
| `tools/router_fit.py` | concentration of S2's loss and the "router" | **done** (session 0b, E10): loss is concentrated (worst 5% of decisions hold 75%) but S2's margin does not find them (do not build the margin router); a learned router gets about half the gain for 30% of the searches |
| root-restricted solve (`shortlist` in the app API) | search only the shortlisted cards | **built and exact** (session 0c, E11): cost proportional to the cards kept (4-card trick-2 decisions: 2 kept 0.49, 3 kept 0.80 of the nodes); trick 2 has only 2-4 candidates, so little to drop |
| runs `student-t2-plain-1`, `-plain-2`, `student-t2-gv2-250k-*` | S2 teacher A/B (plain H_3 against H_3 + G2), with controls and the decision-aware loss (session 1) | **plain arm and the loss done** (E12): retraining noise SD 0.020; the decision-aware loss (`rank_weight` 1, `rank_temperature` 0.01, `stop_on` decision) halves the golden-set loss (0.0587% over 3 seeds against 0.1195% plain at 250,000 decisions) and at 1,000,000 decisions reaches 0.0389% alone / 0.080% at seat 1 against the existing S2's 0.0733% / 0.179%; early G2 comparison at 50,000 decisions: G2 labels no better with the plain loss (-0.006 +/- 0.017) and worse with the decision-aware loss (+0.024 +/- 0.009); labelling cancelled at 60,000 decisions, G2 not pursued for S2 now; to try the same loss on S1 (E12, E13) |
| same runs | throughput of the G2 teacher on rented machines | **done** (E13): 7,900 decisions an hour, $92 per million at $0.723/h on a 245-thread EPYC 7B12; two idle-CPU bugs found and fixed |
| `machine.keep` / `machine.instance_id` (`launch.py`, `test_keep.py`) | leave a machine up after a run and take it over, a stopped one included | keep **works**; takeover of a stopped instance **worked** (O2 session, reported), recycling a running one **failed** (E14: the container booted the old environment); a failed takeover now keeps the machine |

## History
- `history/STATUS-history.md`: every dated status entry, 2026-10-13 to 22.
- `history/plan-log-2026-10.md`: the working plan log, 2026-10-16 to 22 (instructions there are history,
  not current).
- `reviews/REVIEW-A.md` to `-E.md`: the five code reviews of 2026-10-15.
- The Copilot work of 2026-10-19/20 was discarded by the owner (branch `archive/silver-shared`;
  EXPERIMENTS Part F).
