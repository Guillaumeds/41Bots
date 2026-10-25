# 41Bots documents: where everything is

**Read in this order:** `CLAUDE.md` (working rules) → `docs/STATUS.md` (where we are, what is
running, what is next) → this page. Then, as needed:

| record | holds |
|---|---|
| `RULES.md`, `TRANSCRIPTS.md` | the authoritative rules and the owner's house rules (H); owner-approved transcripts |
| `PLAN.md` (amendments at the top), `REVIEW-2026-10-13.md` | the approved design and the review that amended it |
| **`RESEARCH.md`** | the outside research: papers, other programs and the game family, with their numbers and where each is used |
| **`EXPERIMENTS.md`** | every experiment we ran: question, method, result, cost, limits, files (parts A-H, cited below) |
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
| `m3-hybrid/sibrank.md` | sibling ranking loss for O_3 | **rejected**: the 2026-10-19 build-VM test was negative (C9) and the rented A/B of 2026-10-23 was clearly worse at weight 1 (D12) |
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
| `m3-hybrid/o2-stagea-netcmp.json` | O_2 Stage A: scores of five networks on the 250k exact test set (EXPERIMENTS D6) | **result: plain labels tie exact at equal volume and win at equal money; volume is the lever**; G2 arm pending |
| `m3-hybrid/o2-stageb-netcmp-v20.json` | O_2 on 19.8M plain labels, scored on the 250k exact test set (EXPERIMENTS D8) | **result: SD 2.75%, sibling regret 0.176%; volume alone will not reach 0.01%** |
| `m3-hybrid/o2-capacity-netcmp.json`, `o2-capacity-wide-netcmp.json` | O_2 capacity and seed noise on 11M plain labels (EXPERIMENTS D9) | **result: deeper/wider lower the SD but not sibling regret; seed noise about 0.02** |
| `m3-hybrid/o2-audit-10/` | O_2 audit loop at trick 2, rounds 1-2 audit reports (EXPERIMENTS D11) | **result: targeted rounds change nothing beyond noise; round 1 (40M, SD 2.60%) adopted as O_2** |
| `m3-hybrid/o3-pair-score.json` | O_3 sibling ranking loss A/B, control against weight 1 (EXPERIMENTS D12) | **result: weight 1 clearly worse (sibling regret 0.127% against 0.072%); rejected** |
| `m3-hybrid/o3-ladder-calibration.json`, `m3-hybrid/o3-ladder/` | Calibration: O_3-recipe networks at 1M-20M rows, decision loss against sibling regret / SD (EXPERIMENTS D10) | **result: decision loss about regret^1.7; O_2 v11 estimated at 0.022-0.026% at trick 1; 0.01% needs sibling regret about 0.11%** |
| runs `o2-vol-3`, `o2-vol40-1` | 39.6M plain hybrid trick-2 labels in 18 stored streams of 2.2M | **done** (D7, D11): stored under `runs/<run>/partial/volNN/`; the adopted O_2 was trained on them |
| `docs/vast/o2-audit-4.manifest.json`, `python/tools/o2_audit_manifest.py` | the O_2 audit loop on the 39.6M base and its manifest builder | **done, stopped after round 2** (D11) |
| run `o2-copper-s1-1` (`docs/vast/o2-copper-s1.manifest.json`) | trick-1 teacher data with the adopted O_2: copper (2,000 decisions, 128 deals) and 250,000 S1 labels (64 deals) | **done**: 15.1 h, about $4.95, $19.5 per million decisions (D13) |
| run `o2-s1test-2` (`docs/vast/o2-s1test.manifest.json`) | the S1 test set: 2,000 trick-1 decisions, 128 deals, role test | **done**: 20.6 min, $0.36 (D13) |
| run `student-t1-250k-10` (`docs/vast/student-t1-250k.manifest.json`) | S1 at 250,000 decisions: plain against decision-aware loss, seeds 1-3, plus 50k / 100k / 32-deal points | **done**: decision-aware 0.0472% against plain 0.0853% on the S1 test set; 84 min, $1.87 (D14); nine earlier launches failed on stale binaries (A10) |
| `python/tools/o3_pair_score.py`, `docs/vast/o3-pair.manifest.json` | the O_3 sibling ranking A/B and its scorer | **done**, 2 of 5 arms run (D12) |
| `python/tools/grab_machines.py`, `python/tools/o2watch.py`, `fortyone_nn.netcmp` | the machine watcher/renter, run progress from R2 heartbeats, paired comparison of networks on exact shards | **done**, in use (A10, D6) |
| `m3-hybrid/silver.md` | trick-1 reference (exact through trick 2, then O_3) | built, **incomplete**: no silver set exists (D4) |

## Students (fast networks for the app)
| document | what | outcome |
|---|---|---|
| `STUDENT.md` §11 | trick-2 student, learning curve | **done**: 0.057% alone at 1M decisions (E2) |
| `STUDENT.md` §13 | deals per decision | **adopted**: 64 deals (E3) |
| `STUDENT.md` §14 | split-half ruler, top-k shortlist | **done**: top-3 shortlist about 0.005-0.012% (E4-E5) |
| `STUDENT.md` §6.1 | the app's player as an arena bot; the money gate `docs/vast/arena-students.manifest.json` | **built** 2026-10-25, run 2 pending run 1's students |
| run `student-t1-cost-2` | trick-1 teacher cost | **done**: about 12,500 decisions an hour on 256 threads (E6) |
| `m3-hybrid/path-forward.md` | **proposed path** from the O3G2 rounds to shipped students: free reads, S2 A/B, O_2 label test, trick-1 yardsticks (copper, playout), S1, then improvements | **open**: path approved in outline (2026-10-22); steps 0-2 done by 2026-10-23 (free reads, S2, O_2); steps 3-5 (yardsticks, S1, ship) open; section 0 records what was done and the lessons; section 6 holds a brief for each session |
| run `student-check --sub-deals` | shortlist + search at the app's deal counts (25-100) on the exact golden set | **done** (session 0a, E9): top-3 + search beats S2 alone from about 25 deals (0.0733% to 0.0596 / 0.0402 / 0.0293 at 25 / 50 / 100), loses at 10; top-3 is hard to tell from search-all because trick-2 decisions have only 2-4 candidates; top-2 is the real shortlist |
| `tools/router_fit.py` | concentration of S2's loss and the "router" | **done** (session 0b, E10): loss is concentrated (worst 5% of decisions hold 75%) but S2's margin does not find them (do not build the margin router); a learned router gets about half the gain for 30% of the searches |
| root-restricted solve (`shortlist` in the app API) | search only the shortlisted cards | **built and exact** (session 0c, E11): cost proportional to the cards kept (4-card trick-2 decisions: 2 kept 0.49, 3 kept 0.80 of the nodes); trick 2 has only 2-4 candidates, so little to drop |
| runs `student-t2-plain-1`, `-plain-2` | a decision-aware loss for the trick-2 student (`rank_weight`, `rank_temperature`, `stop_on`) | **done, adopted as the finding** (E12): retraining noise SD 0.020; with `rank_weight` 1, `rank_temperature` 0.01 and `stop_on` decision the golden-set loss halves (0.0587% over 3 seeds against 0.1195% plain at 250,000 decisions) and at 1,000,000 decisions reaches 0.0456% alone over 3 seeds (0.0389 / 0.0491 / 0.0487; SD 0.006) and 0.090% at seat 1, against the existing S2's 0.0733% / 0.179%; not yet in the app |
| runs `student-t2-g2arm-50k-1`, `student-t2-plainarm-50k-1` | G2-labelled teacher against plain H_3 for the trick-2 student, at 50,000 decisions | **done, rejected for now** (E13): no gain with the plain loss (-0.006 +/- 0.017), worse with the decision-aware loss (+0.024 +/- 0.009); the 250,000-decision G2 arm was not made (labelling cancelled at 60,000 decisions) |
| run `student-check --sub-deals` and `tools/router_fit.py` on the decision-aware S2 | shortlist + search and the router with the new network | **done** (E16): the new S2 alone (0.0389%) beats every search up to 25 deals and equals search at 50; top-3 + search wins only at 100 deals (0.0258%); the risk model still does not find the risky decisions (correlation 0.055) |
| run `student-t2-losssearch-1` | weight and temperature search for the decision-aware loss (22 networks, 12 cells) | **done** (E17): no cell clearly beats weight 1, temperature 0.01 (0.0587%); temperature 0.01 is the best column, the weight (0.3 to 3) matters little, 0.005 and 0.02-0.03 are worse |
| runs `student-t2-gv2-250k-{3,4,6,8}` | throughput of the G2 teacher on rented machines | **done** (E14): 10,700 decisions an hour, $68 per million at $0.723/h on a 245-thread EPYC 7B12 after two idle-CPU bugs were fixed (38 down to 28 minutes a 5,000-decision part) |
| `machine.keep` / `machine.instance_id` (`launch.py`, `test_keep.py`) | leave a machine up after a run and take it over, a stopped one included | keep **works**; takeover of a hand-rented machine **worked** once (O2 session, reported; stopped or running not recorded), of a launcher-created kept machine **failed** once (E15: the container booted the old environment); a failed takeover now keeps the machine |

## History
- `history/STATUS-history.md`: every dated status entry, 2026-10-13 to 23 (the 2026-10-22/23 O_2 state paragraphs were moved there from STATUS on 2026-10-23).
- `history/plan-log-2026-10.md`: the working plan log, 2026-10-16 to 22 (instructions there are history,
  not current).
- `reviews/REVIEW-A.md` to `-E.md`: the five code reviews of 2026-10-15.
- The Copilot work of 2026-10-19/20 was discarded by the owner (branch `archive/silver-shared`;
  EXPERIMENTS Part F).
