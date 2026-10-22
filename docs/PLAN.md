# 41 Bots: Research Analysis, Engine Design, Tech Stack and Run Operations

*How to read this:* the first time a technical term appears it is explained in [square brackets]. Section 17 is a glossary.

---

## 0. Context

You want to start fresh: drop the old 41 engine, study the linked papers and the World Computer-Bridge Championship winners, find the "winning recipe", and design the strongest possible bots for 41 (8 players, 4 partnerships of 2). The design must use only proven research, adapted to 41. Compute is saved only through engineering, never by lowering quality.

This document has four parts:
1. what the research says (sections 1–4)
2. the bot design, and why each part is there (sections 5–10)
3. the tech stack, pinned to the latest compatible stable releases, and how runs use rented machines (sections 11–12)
4. the build order with pass/fail checkpoints, and how we prove it works (sections 13–16)

Nothing from the existing 41 app is reused.

### 0.1 Your decisions

| Topic | Decision | Effect on the design |
|---|---|---|
| **What bots play for** | **The money.** A match is won by the first partnership to win 2 games. Each player of every losing team pays the agreed stake to the winning team (pub stake €5: winners get €15 each, losers pay €5 each). | Every decision maximises the team's **expected money**, which equals maximising its chance of winning the *match* (section 5.3). Games and points matter only through that. |
| Who bots play with | **Mixed tables:** bots partner and oppose humans | Human compatibility is in scope (phase E, section 8) |
| Where bots run | **Anywhere:** on the player's device eventually; Cloudflare serverless now; CPU server, cloud GPU or home GPU if commercially viable | One engine, four deployment tiers (section 7). Strength versus cost is measured per tier. |
| Illegal plays | **Impossible in the app** | The engine only offers legal actions. Penalty rules 25–26 and 37–41 can never trigger. |
| Robbing (house rule) | The ace-of-trumps holder robs **at their own first turn** in trick 1, and **may burn any card except the ace of trumps** (rules 21, 24, RULES H3; the 2026-10-13 no-ace rule was withdrawn by the owner on 2026-10-14), including the card just taken | Engine (6.1) and card tracker (6.3). Only the dealer's rob of a turned ace (also one turned after a Joker) and its take of a turned Joker stay "at once, before the first lead" (rules 22, 24, H4). |
| Turned Joker + dealer holds the ace of trumps (rule 24) | The dealer robs the newly turned ordinary card **at their own first turn in trick 1, as the eighth card** (owner, 2026-10-14; it was "at once" before) | Fixed in the engine (`RULES_VERSION` 2) |
| Next dealer after a game ends mid-hand | The deal **always moves one seat left**, as after any hand | Fixed in the engine |
| Where to build | A clean-slate project in a **new folder in this working directory** (`C:\Users\gdeswardt1.cnslt\41Bots\`), fully independent of the app | Nothing from the app is copied or referenced |
| Repository | **A new private GitHub repo (`Guillaumeds/41Bots`), not a branch of the app repo (`fivefingers`)** | A branch shares the app's history, workflows and settings, and branches are meant to merge back; this project never should. It has its own Rust + Python toolchain, CI and release rhythm. **Secrets:** create *new*, least-privilege keys for this repo instead of reusing the app's Cloudflare token: an R2 key scoped to this project's bucket, and a Workers deploy token later (M8). Rented machines only ever get the bucket-scoped key. The app later consumes versioned releases of the bot engine (WASM package and bot Worker API) like any dependency. |
| Compute operations | Windows laptop for development and overnight serial jobs. Rented vast.ai Linux machines for heavy runs: a different, cheapest machine each run, destroyed afterwards. CPUs near 100%, the GPU used where it helps, and all work finishing together. Data must move freely between Windows, Linux and Cloudflare. | Sections 11 and 12 |

---

## Amendments from the review of 13 October 2026

*The plan below was reviewed on 13 October 2026 (8 research sweeps, 5 reviewer lenses, 3 sceptics per proposed change; record in `docs/REVIEW-2026-10-13.md`). The amendments in this section are in force. The sections that follow have been edited where their original wording would mislead; where an original sentence survives that conflicts with an amendment, the amendment wins. The checkpoint table in section 13 has been replaced in full; `docs/RULES.md` carries the rule change (H3) and the confirmed interpretation notes.*

### Verdict

**The plan is the right plan.** Its recipe (exact open-hand solver → sampled worlds weighted by inference → one level of recursive rollouts with realistic simulated players → shared learned blueprint → self-play and search distillation, all scored in one four-way money-chance type and promoted only through duplicate tournaments) is still the proven winning recipe for trick-taking games. Nothing published 2019–2026 overturns it, no 41 AI exists anywhere, and every pinned version in the tech stack is correct as of today. The match objective is already built in correctly everywhere; the review found no place where points or games leak in.

**Three load-bearing assumptions must change** before the plan can be executed as a schedule:
1. **The exact solver is not cheap at trick 1.** Three independent enumerations put a trick-1 world at ~10^12 raw paths and ~10^8–10^9 node expansions even with a perfect transposition table; four-sided max^n gets only weak pruning, and partition search has never been shown to work for four-way values. Realistic: 0.1–150 s per world at trick 1, ~100× cheaper per trick, sub-millisecond from trick 3. So "tens of millions of exact teacher hands" and "thousands of worlds" are unaffordable and unsupported. The solver's *roles* change (exact late, measured early); the solver itself does not.
2. **The match-equity model should be an exact table, built at M1.** Reachable scores per team are only 15 values ({0,5,…,40} ∪ {11,16,…,36}; the first bonus arrives with its trick as 11, so 6 is unreachable), the bonus flag equals `score mod 5 == 1`, and only the dealer's *team* matters: 15^4 × 16 games-won patterns × 4 = 3.24M states (~810k after the 4 team rotations). Every hand adds ≥25 points or ends a game, so the state graph is acyclic and one backward sweep is exact given each state's hand-outcome distribution (the 20-year-old backgammon method). As written the plan is circular: the M2 solver scores lines with a model first built at M4.
3. **The learning pipeline is under-specified where it matters:** the per-trick reward's exact form, the solver's tie rule, the belief model's form, the league, network sizes and throughput. Each gap could let the single money objective drift in implementation. The fixes are specification, not new methods.

Plus ~20 simplifications and engineering savings with identical or tournament-gated answers, and a handful of text corrections. Evidence: `docs/REVIEW-2026-10-13.md`.

### Owner decisions taken during this review (2026-10-13)
- **Stakes:** every loser pays the same (H6 as written). The four-chance money type stays exactly as planned.
- **Burning (rule change to H3, withdrawn 2026-10-14):** ~~no ace of any suit may ever be burned~~. *Withdrawn by the owner on 2026-10-14: back to the original H3, any card but the ace of trumps may be burned (see the 2026-10-14 decisions below).* Confirmed: the dealer must take a turned Joker; the ace-of-trumps holder must rob at their own turn in trick 1 (H2/H4 unchanged).
- **Thinking time:** up to **8 s on trick 1 and 5 s on trick 2**, the same on phone and server. Tricks 3–4 unstated: assume ≤3 s; trick 5 is forced. Tiers differ by compute per second, not by time.
- **Early playable demo:** yes, add **M3b**.
- **Rules cross-check:** 41 is the same family as the Irish 25 app's "41"; the app publishes no rulebook, but everything it documents matches RULES.md except that robbing is optional there (compulsory here) and the "6 then 5" bonus is documented nowhere else (all sources: flat 11). Keep the house rules; add a `bonus_later` rule switch (5 house / 6 flat-11) in the engine only. *(Removed 2026-10-14: the owner rejected the flat-11 variant; bonuses are always 6 then 5.)*
- **Tournament mixes:** equal weight on all 14 candidate/incumbent mixes.
- Defaults confirmed by RULES.md without asking: "reach 41" = 41 or more; the trick scores before any pending end-of-hand bonus; a pending bonus is lost if the game ends at an earlier trick; a dealer holding the ace with an ordinary turned card robs at their own first turn (H2); burn-then-first-card can be one internal decision.

### Owner decisions of 2026-10-14
- **The no-ace burn rule is withdrawn; back to the original H3 (RULES H3, rules 21 and 24).** Any card except the ace of trumps may be burned, plain aces, the card just taken and the Joker included. The hard fact for the card tracker and belief model is now "a burn is never the ace of trumps", except the dealer's burn after a turned Joker, made before trumps are known: that burn may hold any card, even the future ace of trumps (then nobody robs; the owner answered on 2026-10-14 that this is allowed, a poor play but legal: see STATUS 2026-10-14 and M3-SPEC §13 note 6).
- **Turned Joker + ordinary card + dealer holds the ace of trumps (RULES H4):** the dealer no longer robs at once; they rob at their own first turn in trick 1, as the eighth card. So after a turned Joker and an ordinary card the dealer is *not* known at once to lack the ace; like everyone else they prove it only by playing their first card of trick 1 without robbing.
- **The `bonus_later` switch is removed.** The flat-11 variant was considered and rejected by the owner; a partnership's first bonus in a game is 6, every later one 5, with no switch anywhere (engine, reference, difftest, equity, arena, table manager).
- The engine's `RULES_VERSION` is now 2; records of version 1 are refused.

### Amendments by theme (section numbers refer to the sections below; where an original sentence has not been rewritten, the amendment wins)

#### A1. Objective and match equity (§5.3, §6.6, §6.5, §8, §15)
- **Exact match-equity table** by backward induction over (15 scores per team, games won, dealer team), sweep order "more games won first, then more total points"; hand-outcome distributions from duplicate-seeded rollouts of bots that already use the finalised successor equities; iterate with a sweep cap, log the maximum change, call the result "self-consistent with bot version X"; stratified rollout counts plus a log-odds smoother for rarely reached states. v0 at M1 from the rules bot; refreshed before the M4 teacher run and per generation; table version hash in every record. Stored in the fixed-point money type (A7).
- **Within-hand money chance stays learned**, trained toward the **exact table value at the end of the current hand** (Mortal's convention), not the final match result. Feed deterministic **table-derived features** computed in the Rust encoder: each team's equity if the hand ended now, per-team "can clinch this hand" flags (jink, immediate 5-of-trumps bonus, deferred bonus, first/later bonus), best/worst reachable successor equity as a gated extra. The public money-chance head becomes a separate public-input sub-module (no private cards in its trunk).
- **Per-trick reward in explicit potential-based form:** γ = 1; for each team `r_t = Φ(s_{t+1}) − Φ(s_t)` with Φ = oracle head within a hand and the exact table at hand boundaries, frozen per iteration; Φ(terminal) = 0 with the actual money result as the only terminal reward; Φ values stored per trick in the game record; a telescoping unit test; a short unshaped fine-tune per generation; promotion on unshaped money only. Cite Lu, Schwartz & Givigi 2011 and Devlin & Kudenko 2011/2012 (general-sum, n-player) beside Ng 1999 (single-agent).
- **Calibration recipe:** four-way softmax heads with cross-entropy, distilled from the table at hand boundaries, temperature scaling on held-out matches, log loss/Brier/adaptive-binned reliability down to 0.1–1%; focal loss as a gated challenger only. Sensitivity test: perturb the table in **log-odds** (±5% relative), investigate flips as table defects first, then declare ties.
- **Below the search's resolution:** the SPARTA-form deviation threshold (k paired standard errors plus an absolute floor) keeps a team on the money-trained blueprint; log the paired standard error per decision. Scenario tests in §14 stay exact (solver or enumeration at the scenario's depth).
- Note in §15 that equilibrium *selection* in a four-team game is handled by the shared blueprint plus distillation; the invariance theorem protects the set of equilibria only.

#### A2. Exact solver (§5.1, §6.7, §10, §13 M2)
- Say the solver is cheap **late in the hand**, not per world. Default exact use from trick 3, for post-mortems, and as brute-force ground truth at every trick. **First M2 deliverable: a measurement spike** (naive max^n on the 4-vCPU box) before any phase-A restructuring is finalised; the M2 gate publishes nodes, hit rates, memory and ms per world by trick index and seat on two CPU types; every cost figure above is a placeholder until then.
- **Backup rule stated:** at our team's nodes maximise our entry; at the other teams' nodes, before M4 a fixed position-only successor order (the tie rule under which "pruning on == pruning off" is asserted), from M4 a policy-weighted expectation with a floor (Prob-max^n proper, restricted to tied-best or top-k children to control cost); re-prove or disable speculative pruning in expectation mode. Fix the text that calls Prob-max^n a tie-break. M2 tests: fraction of tied nodes; root-flip rate under a different tie order.
- **Transposition key** = (remaining hands, leader, tricks per team this hand, 5-played, best trump so far and its team, per-team bonus flags), keyed at trick boundaries; the money value is not additive in points, so the in-hand score state can never be dropped.
- **Power-equivalent cards** defined by exact rule semantics with 41 exceptions (5 vs J bonus timing; J/Joker/A♥ force different sets; only A♥ may be held back against the ace of trumps; hearts trumps has no separate ace of trumps), with positive and negative test decks. **Partition search moved out of the critical path** (no publication applies it to max^n or vector values); near-equal-card merging and ProbCut stay out.
- Test-only brute-force **forced-denial check** ("can the other three teams deny team A every remaining trick in this world") for the §14 scenarios; never a decision rule.
- Adopt the four-layer solver validation gate and the 0.000 canary from the Nova Scotia 45s project (anecdotal precedent, consistent with Sturtevant): engine parity on random play-outs; search vs brute force on 1-, 2-, 3-card endgames; reduction on/off identical root values; exhaustive best response from mid-hand engine states.

#### A3. Teacher and bootstrap (§8 phase A, §13 M3/M4)
- ~10^5–2×10^6 teacher hands (stratified over match states and tricks so the late-match blocking states are covered), **exact solves from trick 3; at tricks 1–2 rollout-PIMC with the rules-bot inner player** (no network exists at M3; Bouzy: parity with exact PIMC at 5 cards), with few-worlds exact PIMC as a measured comparison arm. Per-decision action-value vectors recorded; train to the validation plateau; gate by SPRT. Teacher generation continues as a background chunk type. Belief-head data comes from any self-play (true locations known), not the teacher.
- Start phase B with a KL penalty toward the teacher decaying to zero (AlphaStar's precedent) and run one budget-capped **from-scratch control arm** at M6 under the SPRT gate.
- **Pre-register the M3 PIMC-suitability measurement** (Long et al.: 10,000 hands; leaf correlation, bias, disambiguation per trick; fraction unresolved at trick 4) as a report that prioritises tournament time, not an automatic engine switch.

#### A3'. Exact PIMC at tricks 1–2 (2026-10-14, owner)
- **Exact PIMC is required at tricks 1–2 for the yardstick and the teacher targets.** The early tricks are where good players differ most from the plain rules, so the reference values there come from exact open-hand solves of sampled worlds [exact PIMC: every imagined deal is solved exactly, not played out by a simple bot], run offline on large rented machines with measured world counts (M3-SPEC §13, §13.1).
- **Measured at trick 2:** rollout-PIMC (the rules bot playing out each imagined deal) loses about **13 times more money per decision** than exact PIMC against the yardstick. That is far too much for a training target.
- **Rollouts stay only as a scalable fallback** where exact is unaffordable (inside a timed decision on a small device, or a trick-1 batch the budget cannot cover), and every such fallback is scored against the exact yardstick. This supersedes A3's "at tricks 1–2 rollout-PIMC with the rules-bot inner player" as the default teacher.

#### A4. Decision-time search (§6.8, §6.9, §7.1, §13 M5)
- **Recursion depth 1** by default (depth 2 an explicit experiment). Inner player: policy-network rollouts at tricks 1–2 on every tier; from trick 3 on **T2/T3 only**, one-level recursion with PIMC + exact solver inside (Furtak's RK form); T0/T1 use policy rollouts throughout. Both hybrids stay in the pool with the switch point set by SPRT.
- **World counts from measured decision-agreement curves** per tier (start at the published plateaus: 20 worlds for 5-card solver PIMC, 160 in Skat; T3 may exceed them while money rises); shared world pool for all candidates; SPARTA's 100-rollout floor then the 2-SD skip. Replace "exact late enumeration" (41 has ~1.5×10^36 worlds at trick 1, 6.6×10^15 at trick 4, 2.5×10^8 at trick 5) with a **counted threshold** (dynamic program over the suit-length matrix; enumerate only below a tier budget); **trick 5 is decision-free**.
- **Mandatory M5 experiments at every tier** (Learned Belief Search): truncated rollouts + value bootstrap vs full-hand rollouts; all seven bots searching independently vs one seat per team / tighter shared threshold; three-arm cut-off evaluator (oracle head on the sampled world; seat-view head; sum of seat-view heads). Head 4 reworded: "never sees the real hidden cards; inside a sampled world it may evaluate that hypothesis"; the firewall test asserts its inputs come only from the sampled world.
- **Blocking tested as a joint outcome:** a "leader can clinch this hand" deal library (exact by solver/oracle), replayed with all three blocking teams searching independently; report clinch rate and the blockers' own money change (a block that costs a blocker money is an error); MP-Mix "offensive against the leader" / "paranoid when leading" bots as arena baselines only.
- Pool additions: determinised per-world UCT with max^n backups (Sturtevant 2008 stopped moon-shooting 7.7% vs 42%); EPIMC as a small-tier experiment. ISMCTS/GO-MCTS demoted (no trick-taking result beats a solver- or rollout-based engine).
- **Burn as a hidden action:** the robber's decision is the joint (burn, first card) searched once over shared worlds, recorded as two actions; worlds carry burn branches for seats that have not yet robbed; a burn is never the ace of trumps (rule 21, H3), except the dealer's pre-trump burn after a turned Joker.
- **Common random numbers from counter-based seeds** `SplitMix64(decision, world, seat, card index)`; asynchronous (EnvPool-style) batching over-provisioned 2–3×; budgets defined in worlds/rollouts so decisions are byte-identical regardless of completion order; kernels must give per-row results independent of batch size.
- Time budget per trick from the owner (8 s / 5 s / ≤3 s assumed) is the T0–T3 wall-clock envelope; T3's budget in worlds/rollouts is fixed before M5 and used by the promotion gate.

#### A5. Belief (§6.3, §6.4, §6.5 head 3)
- **Autoregressive belief head** with hard-constraint masking (Learned Belief Search / Off-Team / Ataraxos form): assigns each unseen card to one of 7 hands, undealt, robber-burn, dealer-burn(s), masking full or forbidden locations; proposal probability = product of conditionals (the importance weight is otherwise undefined). Default on T2/T3; **marginal-plus-masking** default on T0/T1; per-card marginals as a diagnostic.
- **One resample-move particle filter:** multiplicative policy-likelihood weights, Kish ESS, resample below N/2, History-Filtering Metropolis-Hastings swap kernel for rejuvenation. The weight floor is a **correctness** requirement (full-support policies). Deal generator on a **suit-length matrix with max-flow feasibility**; rejection sampling only as the test oracle.
- Card tracker facts rewritten incrementally (A6). Belief inference conditions on the match state and table features; belief quality (TSSR, NLL vs grounded and MCMC references, ESS, valid-sample rate, calibration) reported per match-state class; oracle-belief ablation; the arena in money remains the gate.
- Belief head trained each generation on a **mixture of league trajectories** (champion, past versions, noisy and rules bots, later the human model); 99% valid-sample target with a fallback ladder (autoregressive → MCMC → constraint-only), the rung logged per decision.

#### A6. Card tracker (§6.3)
- Ace facts incremental: the dealer has taken it the moment they rob a turned ace at once; every player, the dealer included after a turned Joker and an ordinary card (H4 as amended 2026-10-14), lacks it in hand from their first non-robbing play; out of play (undealt, or in the dealer's Joker burn) only once trick 1 completes without a rob. The turned Joker reveals nothing about the ace. A robbed turned ace is certainly in the robber's hand (the ace of trumps may never be burned). The robber's hand is exactly 5 of its 6 cards. Hidden out-of-play cards are **12 or 13** (12 undealt, +1 burn; or 11 + 2 after a Joker). Burns never contain the ace of trumps, except possibly the dealer's pre-trump burn after a turned Joker. Facts derived from the engine's own legal-burn and forced-rob functions; a property test that hard facts never exclude the true deal.

#### A7. Networks, training, distillation (§6.5, §8, §7, §13 M4/M6/M7)
- **PPO + perfect-information critic** specified: GAE, γ = 1, λ ≈ 0.95, four-way softmax critic on the oracle input, value-loss clipping (Trinal-Clip), target-entropy control, actor-learner lag ≤ 1. Challenger: **DMC + annealed oracle feature dropout + Q-ratio clipping** (OADMCDou). Never distil from an all-cards-visible policy.
- **Residual MLP** in the M4 pool at matched FLOPs (plain MLP stays default until SPRT); no LSTM over history. Sizing table as the initial pool, **every number an estimate**: teacher 4–6 blocks × 512 (2–4M params); T2 student 2–3 × 256; T0/T1 student 2 × 128–256. M4 gate: measure and publish decisions/s/core for engine + actor inference on a cheap EPYC/Xeon and derive the cost of one phase-B generation before any GPU rental.
- **League specified:** K-best past champions by money rating + skill-level checkpoints (init, mid, converged) + rules and noisy bots; 10–30% of games against the pool; cycling detector (duplicate SPRT vs the frozen pool + exploitability probe each generation; a generation that beats its predecessor but loses to the pool is not promoted). Same pool = phase-E cross-play suite and M9 population (Fictitious Co-Play evidence).
- **Soft targets:** store per-card money-chance 4-vectors and sample counts in the training shards (records stay action-only); distil the full distribution (Expert Iteration: +50 Elo over chosen-action targets).
- **Diagnostics-only auxiliary heads** ("tricks per team this hand", "game ends this hand") as an M6 ablation, enforced outside the money-chance type and listed in the objective audit.
- **Phase C budget:** rollouts per decision from the M5 curve; KataGo playout-cap randomisation (A/B 100% vs 25% full-search decisions at equal compute); targets only from full-search decisions; ~1 core-minute per recorded hand, one million-hand generation in 3–7 days on 64 cores.
- **Phase E order:** population robustness first; the rules bot (published 25/45-family heuristics) as the interim human-like predictor in belief weighting only; behaviour cloning and piKL only above an owner-set human-data threshold, triggered by a held-out prediction test; Off-Belief Learning out. piKL's λ chosen by money won at mixed tables; difficulty levels are labelled non-champion configurations outside the audit.
- Compute budget line (estimate): phase B ≈ 10^9–10^10 decisions, 1–2 days per generation on a 64-core, 4×RTX 3090 box; first full pipeline pass ≈ 6–10 weeks of machine time, ≈ €1,000–2,000. A generation ends when its SPRT gate says so, spanning runs if needed.

#### A8. Engineering and determinism (§6.1, §10, §11, §12, §14, Cargo.toml)
- **Fixed-point money-chance type:** u32 per team in units of 2^-32, normalised by largest-remainder rounding, u64 accumulators, rayon maps into pre-sized slots with a sequential reduce; floats only inside the network forward pass. Determinism **scoped**: engine, solver and teacher data identical everywhere; network-dependent data identical per (binary, CPU class = dispatched feature level, recorded in every record); GPU training resumes from checkpoints with logged divergence (not byte-identical across GPU generations); the kill-the-machine identity test applies to data-generation chunks.
- **Active-feature-index encoder** designed at M4 (serves float and integer paths). **NNUE-style integer students** after M5: quantisation-aware training with weight clamping, int16 first layer / int8 hidden / int32 accumulation, incremental first-layer accumulator, kernels behind multiversion (AVX2, AVX-512-VNNI, simd128, plain v2); gated against f32 twins by SPRT at equal wall-clock per tier; "bit-identical native vs WASM" applies to the integer path only; float stays "within tolerance" (FMA vs simd128). Never post-training-quantise an unclamped net (Lc0: −300 Elo).
- Start the MLP on matrixmultiply plus a plain reference routine; a fused kernel only if profiling demands. **Drop `ort`; defer tract, ruzstd, arrow/parquet (Rust), mimalloc** until a consumer exists; the transformer challenger must enter the T2/T3 pool by the first M7 generation (dated, so deferral is not omission).
- Workspace lint stays `forbid(unsafe_code)`; `crates/nn` is the only crate allowed to opt out (own `deny`, one audited kernel module, differential test vs reference on v2/v3/wasm32, CI grep that no other crate contains `unsafe`).
- Repo layout: `crates/reference` and `crates/difftest` (no root `tests/`); `tools/` for the Python RNG golden generator; CI step `cargo tree -p reference` must not show `engine`.
- CI: Windows on PRs and pushes to main plus nightly; Docker only on tags or manual dispatch from a digest-pinned base; a manual sharded 10M-game differential job (5 × 2M, resumable) within the 6-hour cap.
- **Cloudflare:** design for the stricter of the conflicting figures the reviewers reported and **re-verify at M3b against live docs** (script size 10 MB compressed vs 64 MiB uncompressed; startup 400 ms vs 1 s; DO attachment 2 KB vs 16 KB): weights loaded lazily from R2/KV, table state persisted in DO storage after every action and reconstructed by replay, worlds/particles capped by a measured memory budget, no WASM threads. T2 decided at M8 between Containers (standard-4, 4 native threads, same $/vCPU-s, idle-billed) and Worker fan-out, by measured strength per dollar and latency; build neither before M8.
- **vast.ai:** no CPU-only compute product exists (CPU-only jobs go to CI, the laptop or cheap-GPU boxes); CPU/RAM proportional to GPU fraction (`gpu_frac=1`, filter `cpu_cores_effective`, `cpu_ram`); interruptible instances have processes **killed** when outbid → checkpoint ≤10 min mandatory; storage and image-download bandwidth are billed even while loading → slim image, low `inet_down_cost`; per-instance key is `CONTAINER_API_KEY`.
- **Docker image:** recommended public, secret-free base image (CUDA base, uv-locked env, rclone, vastai) with project binaries and the Python wheel fetched from R2 by content hash at start-up (resolves the private-image vs bucket-key-only contradiction; owner to confirm).
- rclone `--immutable` only for content-addressed bundles; done-list, heartbeat and manifest as versioned keys with zero-padded sequence numbers; R2 lifecycle rule for old heartbeats.
- Mixed tables: reply after a content-independent delay (forced moves answered instantly leak hand structure); a bot taking over a disconnected human's seat weights that seat's earlier plays with the human model; symmetry canonicalisation only clubs↔spades when trumps are red, applied identically to particles, keys and training data, with a test. Human game data is pseudonymous: consent text, retention and deletion handling are M8 deliverables.

#### A9. Evaluation (§9, §10, §13 M1, §17)
- **Duplicate block:** deck for hand k derived from (block seed, k); candidate-vs-incumbent = the **14-way block** (every A/B assignment to the four team slots except AAAA/BBBB), equal weights, per-mix diagnostics; the 4-rotation 1v3 block as a cheap pre-screen. The block is the unit of analysis.
- **Generalised SPRT** on block scores (Fishtest LLR from sample mean and variance), α = β = 0.05 pre-registered with the effect size in stake units, two stages with the cheap stage **reject-only**, a hard cap → "inconclusive"; block cost per tier published; σ_b measured at M1.
- **AIVAT** = per-hand chance control variate using the oracle head with a fresh-shuffle baseline plus the known-strategy action term (bot-only tables), from M4; null test; expect 20–60% SD reduction (the plan's "10–40×" is DeepStack's human-match figure).
- Ratings: offline **Plackett-Luce** maximum likelihood; online OpenSkill/TrueSkill with ranks [1,2,2,2]; Elo only for two-program diagnostics. Championship: round robin of 4-program tables replayed under all 24 permutations; final = one table.
- Diagnostics defined: post-mortem error vs the solver under the documented tie rule, labelled "penalises information-gathering and deception", never a gate; belief metrics (A5); calibration; curated problem-deal suite; exploitability probe = RL exploiter team vs three frozen champion teams; cross-play grid with ≥3 seeds (harness at M1); paired evaluation n ≥ 2000 × 2 seeds; identical-bots canary = exactly 0.000.

#### A10. Text corrections
- Sturtevant row: the numbers are right (IJCAI 2003 Table 12, 3-player perfect information); add the thesis Monte-Carlo imperfect-information results (Hearts 6.91 vs 10.32; 4-player 5.34 vs 8.81; "no performance advantage" for paranoid) and the 2008 UCT result.
- Furtak row: "saturation beyond 160 worlds", "+6.84 soloist points at ~8× cost, +2 more at 10× hardware", "vs information-set UCT 74.37 vs 53.99"; solver speed-ups 45×/58×/22×.
- WCBC ended 2019 (unofficial 2022 WBridge5, 2023 Q-Plus); GIBBO (Dec 2025) = neural net **plus** double dummy, +0.9/+0.4 IMP per board over GIB 43; NooK/Ben internals unpublished.
- "best of 3" → "first to two games, at most five"; "13–14 never revealed" → "12–13"; "up to 6 burn candidates" → "5, or 6 for the dealer's first burn after a Joker"; "orders of magnitude" (partition search) → measured factors; Sturtevant & White 2006 is not score-aware; §15 "CPU-only rentals undocumented" → "do not exist"; §4 add the Nova Scotia 45s project and the Irish 25 app as same-family evidence; §16 add Best-Reply Search/BRS+ (one-opponent-moves breaks follow-suit and trick order) and ReBeL/Player of Games (two-player zero-sum guarantees only) as considered and rejected.

---

## 1. The short answer

For 25 years the winning recipe in trick-taking card games (bridge, Skat, Hearts) has stayed the same at its core, and has been refined on top:

1. **A very fast exact solver for the "all cards face up" version** of the game.
2. **Monte Carlo sampling** [try many random guesses and average the results] of where the hidden cards are. Each guess must fit everything seen so far, and guesses are weighted by **inference** [reasoning back from what people played to what they probably hold].
3. **Score each possible card across those guessed deals** and play the best one.

The newest generation improves step 3. Instead of pretending everyone can see all the cards, it finishes each guessed deal with a learned, realistic player, a **neural network** [a program that learns patterns from millions of examples instead of following hand-written rules]. It learns that network by **self-play** [the bot plays millions of games against copies of itself and learns from the results].

For 41 we build this recipe, adapted to four things about 41:
- **four competing partnerships**, not two sides
- **the race to 41**, where points matter more near the finish
- **five-card hands**, where hiding and revealing cards matters (robbing, burning, reneging)
- **very short hands (5 tricks)**, which makes the expensive parts cheap enough for the strongest versions

The same code must also run anywhere, from a phone to a GPU server, as strong as each place's compute allows.

**Everything is optimised for one thing only: the best chance of winning the money**, meaning the match: the first partnership to win 2 games collects the stakes. Training, solving, searching and tournament ranking all use that single objective (section 5.3).

---

## 2. What each linked paper contributes

*Sections 2-4 are the research as read when the plan was written (2026-10-13). The collected,
updated record of all outside research, with the review's verified numbers, is `docs/RESEARCH.md`.*

| Paper | What it is | Key result | What 41 takes from it |
|---|---|---|---|
| **GIB**, Ginsberg, JAIR 2001 (the arXiv 1106.0669 link is the same paper) | The first expert-level bridge program. For each decision it deals 50–500 random versions of the hidden cards that fit what is known. It solves each one **double dummy** [as if all hands were face up] with exact search, then plays the card with the best total. Deals are weighted by how likely the opponents' actual plays would be, using **Bayes' rule** [the maths for updating odds with new evidence]. Its solver uses **partition search** [it remembers whole families of equivalent positions, not single positions]. | Partition search cut the search far beyond normal shortcuts: the branching factor fell from *b* to *b*^0.76. A full deal took about 18,000 search steps (about 1 second on a 2000-era PC). On Bridge Master problems GIB solved 100 versus Bridge Baron's 33, and it dominated the 1998 and 2000 championships. GIB states that "all competitive bridge programs switched" to its method. **Weakness:** it assumes it will know the hidden cards later, so it undervalues safe plays and never plays to gain information. Its declarer fix (achievable sets plus squeaky-wheel optimisation) added only about 0.1 IMP per deal [IMP = bridge's standard scoring unit], and in defence plain Monte Carlo was preferred. | The core method; an exact solver with partition search; weighting guesses by how likely the observed plays are |
| **αμ**, Cazenave & Ventos 2019 | A search that makes "my side" play the same card in every world it cannot tell apart. It assumes the opponents can see everything. | It fixes **strategy fusion** [sampling's flaw: it assumes you can play differently in worlds you cannot actually tell apart] and non-locality. On close 3NT deals, with 40 sampled worlds, the win rate rose from 62.4% (PIMC) to 63.2% (αμ looking 3 moves ahead); with 20 worlds it rose from 60.2% to 62.0%. Cost rises steeply with look-ahead. The paper also confirms GIB and WBridge5 use **PIMC** [Perfect-Information Monte Carlo, the GIB method above]. | Optional late-hand experiment only. In 41 *nobody* sees the hidden cards (opponents included), so our main fix for strategy fusion is recursive rollouts (next row). |
| **Furtak**, PhD thesis, Alberta 2013 (the DOI link) | The search engine of the Skat program Kermit | Solver improvements (symmetries, move ordering, lookup tables) made the exact solver 22–58× faster. Plain PIMC **stopped improving after about 160 sampled worlds**. **Recursive Monte Carlo** (IIMC) finishes each sampled world with a realistic player who sees only their own cards, not the all-seeing solver. It beat Kermit at Kermit's best (about +7 points per game as soloist) and **kept improving with more hardware**, while PIMC did not. It also clearly beat an information-set MCTS bot and was harder to exploit. | **Our main thinking method**, plus solver speed-ups |
| **Setrograde**, Stone, Sturtevant & Schaeffer (arXiv 2024; IJCAI 2025) | Solves bridge endgames backwards, working on sets of positions that share the same value | 1,000× fewer operations and 10,000× fewer stored entries. Every 7-trick bridge ending was solved, the largest endgame database in any game. The authors note that partition search still underpins state-of-the-art bridge play. | Use set-based reasoning inside our solver. We do **not** need a precomputed 41 database: in 41 the cost is the number of hidden-card guesses, not the depth of a hand. |
| **Sturtevant**, IJCAI 2003 | How to search games with 3 or more competing sides | With his pruning, **max^n** [every side picks what is best for itself] beat **paranoid** [assume all other sides gang up on you] in 3-player, all-cards-visible versions of Hearts (8.01 vs 8.99 points, lower is better) and Spades (5.72 vs 5.61, higher is better), even though paranoid searched deeper (IJCAI 2003 Table 12: 3-player perfect-information variants, 50 hands, 500k nodes per move). His thesis adds the stronger imperfect-information evidence: Monte-Carlo Hearts 6.91 vs 10.32 (4-player 5.34 vs 8.81) and "paranoid has a depth advantage but no performance advantage in card games"; his 2008 paper found determinised UCT stopped moon-shooting 7.7% of the time versus 42% for plain max^n | 41 has four sides, so our solver uses max^n with **last-branch and speculative pruning** [safely skip moves that cannot change the result; this works because the four teams' winning chances always add up to 100%] |

---

## 3. What the championship winners actually did

| Winner | How it chose cards |
|---|---|
| 1997 Bridge Baron | Hand-written human techniques plus planning. It was overtaken once search-based programs arrived. |
| 1998–2000 GIB | PIMC with a double-dummy solver and inference |
| 2001–2019 Jack, WBridge5, Shark, Micro Bridge | The same family: GIB reports that by 2001 every competitive program had switched its card play to this approach. WBridge5 is confirmed to use PIMC (αμ paper, 2019). Its 2016–2018 titles came after **seed boosting** [choosing the random-number settings that make its sampling play best] (Ventos et al. 2017). |
| 2023 Q-Plus (unofficial event) | Hans Leber's long-running German program. Its card-play method is not published. The official World Computer-Bridge Championship ended in 2019; 2022 (WBridge5) and 2023 (Q-Plus) were unofficial events. |
| 2022 NooK (NukkAI) | Beat 8 world champions at declarer play (67 of 80 sets) using a hybrid of rules and search. The details are unpublished. |
| 2022 onward: Ben, then BBO's GIBBO (live December 2025) | Neural networks **alongside** double dummy: GIBBO adds a network to GIB's double-dummy play and measured +0.9 IMP per board (basic) and +0.4 (advanced) over GIB 43; Ben's card play is human-trained neural play without double dummy except late in the hand. Neither method is published in detail. |

- **Bidding is the weak spot for all of them, and 41 has no bidding.** 41 is pure card play, which is where computers are strongest.
- **How championships are run (we copy this):**
  - round robin, then knockout
  - **duplicate boards** [the same deals replayed with the teams swapped between seats, so luck of the cards cancels out]
  - since 2001, each computer controls **one seat only** and connects to a **table manager** [the referee server] over the network (the Blue Chip Bridge Table Manager Protocol)

---

## 4. Other proven systems that matter for 41

| System | Game | The proven idea we reuse |
|---|---|---|
| Kermit, then card-location networks (Solinas 2019), then Policy Inference (Rebstock 2019) | Skat | A network that predicts where each card is: +3 to +5 tournament points per game over the old inference in suit and null games (the gain in grand games was not significant). Then scoring each guessed world by how likely the observed plays were, which beat the card-location network again. |
| Long et al., AAAI 2010 | Skat, Hearts | Explains *why* PIMC works in trick-taking games. Outcomes change little between neighbouring endings (**leaf correlation** 0.8–1.0), and every card played reveals information (**disambiguation** about 0.6). It also gives a method to measure both for a new game, which we will apply to 41. |
| GO-MCTS (Rebstock et al. 2024) | Hearts, Skat | **Transformer** [the neural-network design behind modern language models] plus search. It became the best Hearts bot, but **lost to Kermit in Skat**. Learning alone does not automatically beat a well-engineered search engine, so we use both. |
| SPARTA (Lerer et al. 2020) | Hanabi (a partnership game) | Search on top of a shared **blueprint** [the agreed default strategy every bot knows]. Overrule it only when clearly better (margin 0.05). They proved search cannot make the blueprint worse. **Warning:** partners searching independently, without safeguards, did worse. |
| DouZero (2021), PerfectDou (2022) | DouDizhu (a team card game) | **Deep reinforcement learning** [learning by trial and error from game results, using neural networks] at scale, with a small network design that runs fast. PerfectDou adds a training-only judge that sees all the cards. |
| Suphx (2020) | Mahjong (4 players, many rounds) | **Global reward prediction** [turn "points this round" into "chance of winning the whole match"]. Training-time "oracle" hints. It reached the top 0.01% of human players. |
| Pluribus (2019) | 6-player poker | Shows strong bots for more than two sides are possible. Its equilibrium method, **CFR** [an algorithm that computes unexploitable mixed strategies], has *not* worked for trick-taking games (Long et al. 2010; Rebstock et al. 2024), so we do not use it. |

---

## 5. The winning recipe, adapted to 41

### 5.1 How 41 differs from bridge, and what we change

| What is different in 41 | Why it matters | Adaptation (and where it comes from) |
|---|---|---|
| No bidding | The whole game is card play plus the burn choice | Bridge's strongest component applies directly |
| Four competing partnerships | The game is not two-sided; the other teams also fight each other | Each team maximises *its own chance of winning the match*: max^n in the solver (Sturtevant), a 4-way value in the network |
| Only 5 tricks | Exact search per guessed deal is cheap **late in the hand** (sub-millisecond from trick 3); at trick 1 a four-sided exact solve costs 0.1–150 s per world (to be measured at M2), because max^n cannot use the two-sided pruning that makes bridge and Skat solvers fast | Exact solving late, sampled play early; world counts from measured plateaus (20–160 in the literature), one level of recursion (amendments A2–A4) |
| 12–13 cards never revealed (12 undealt, or 11 after a turned Joker, plus 1–2 burned cards) | Less is revealed during play than in bridge | Inference matters more. In checkpoint M3 we measure 41's leaf correlation and disambiguation (Long et al.). |
| The money: the first partnership to win 2 games collects the stakes; each game is a race to 41; a game can end mid-hand; a jink wins a game at once | Raw points and single games are not the true value. 5 points at 0–0 are worth less than the 5 that take you from 36 to 41, and a game matters only for what it does to your chance of collecting the money. | **Match-equity model** [each team's chance of winning the match, and so the money, from any score] (sections 5.3 and 6.6): backgammon's match-equity tables, plus Suphx |
| Reneging with top trumps; robbing at the robber's own turn; hidden burns, which may be the taken card | Players hide and reveal information | Hard-fact tracking plus policy-based inference. Rollout search values deception and concealment; PIMC cannot. |
| Partners cannot talk, and some partners are humans | They must read each other through their plays | A shared blueprint, SPARTA's safeguards (6.10), and human-compatibility training (phase E) |

### 5.2 Architecture at a glance

**At the table** (live games and evaluation tournaments):

```
 TABLE MANAGER  (referee: shuffles, deals, enforces rules, scores, logs)
   Live app:  one Cloudflare Durable Object per table   Training: in-process
      |  sends each seat ONLY what that seat is allowed to see
      v
 SEAT AGENT  (one per bot seat; humans connect the same way)
   1. Card tracker  -> hard facts (played cards, voids, trump limits, robber)
   2. Belief        -> weighted "worlds" (complete guesses of every hidden card)
                       built with the network's belief head; weighted by its policy head
   3. Search        -> for each legal card x each world: play it, then all 8 seats
                       finish the hand using the network's policy, each seeing only its
                       own cards; the end of the hand is scored by the match-equity model
   4. Decide        -> best average chance of the money; overrule the blueprint
                       only when the gain is clearly bigger than noise
   Runs on: phone/browser | Cloudflare Worker | CPU server | GPU server  (same code)
      |  returns one card (or one burn)
      v
 TABLE MANAGER
```

**Training loop:**

```
 Rules engine -> PIMC "teacher" (exact solver) -> millions of games -> copy-learning -> Network v1
 Network vN  -> self-play league (reinforcement learning) ------------------------> Network vN+1
 Network vN+1 + full search -> search-played games -> "distillation" -------------> Network vN+2
 Big network -> "student" networks per deployment tier (phone, serverless, server)
 Every candidate -> duplicate tournaments + statistical test -> champion only if it wins
 Each generation also refreshes the belief head and the match-equity model
```

### 5.3 What the bots optimise: the money

**The payoff.**
- The winning team collects the stakes from every player of the three losing teams. With €5: each winner gets €15, so the team gets +€30.
- Each losing team pays €10 (−€5 per player), **no matter which other team wins**.
- So, for any team: **expected money = 40 × (chance of winning the match) − 10.**

Expected money is just a rescaled chance of winning the match. Maximising one is *exactly* maximising the other. Points, tricks and single games have no value of their own; they matter only through what they do to the chance of collecting the money. The same holds for any stake.

**One objective everywhere: the chance of the money.**
- No component anywhere optimises points, tricks or games. Points are only *inputs* that change the chance of the money.
- The code enforces this. The only score that the solver, search, training and tournaments can produce is a **"money chance" type**: four numbers, one per team, adding to 1. A **type** is a kind of value the compiler checks, so code that tries to score anything else won't even build. Nothing can quietly optimise points instead.

| Component | How it uses the money objective |
|---|---|
| Rules engine (6.1) | Plays complete matches and settles the money at the end (stake configurable; €5 by default) |
| Match-equity model (6.6) | *Is* each team's chance of the money from any match state |
| Exact solver (6.7) | Every finished line is scored by the match-equity model, and each team maximises its own chance of the money (max^n) |
| PIMC teacher (6.7, phase A) | Plays the card with the best average chance of the money across sampled worlds |
| Search (6.8) and burn choice (6.9) | Every rollout ends in a chance of the money; the bot plays the card or burn with the best average |
| Network (6.5) | Value and oracle heads predict the chance of the money; the policy head is trained toward money-maximising actions |
| Training (8) | Rewards are changes in each team's chance of the money **after every trick**, plus the final money result; distillation copies money-maximising search |
| Human compatibility (phase E) | Human models only *predict* how people play. The bot still maximises its money, and prefers human-style plays only where that costs nothing (piKL). |
| Tournaments (9) | Ranked by money won per match; SPRT and luck-removal are applied to money |
| Diagnostics | Post-mortem errors are measured as "chance of the money given away", not tricks lost |

**Trick by trick, not hand by hand.**
- Every trick changes the scores, and with them every team's chance of the money: points are scored the moment a trick is won, and a game can end mid-hand.
- So the chance of the money is re-calculated **after every trick**, and every card decision is searched afresh from the latest scores.
- **Example:** the team in second place takes a trick and becomes first place to the money. From that moment it is **public enemy number one**, and every other team targets it: the previous leader is now only second. This matters most when the new leader **can grab the money in this very hand**, for example one game already won and 36 points, where any single trick wins the money. Then every other team will spend whatever it takes (top trumps, overtaking a partner's card) to deny it *any* trick.
- **How the design guarantees it:**
  - the per-trick money chances (6.6) are part of every bot's inputs (6.5)
  - the search re-plans at every card with the current scores (6.8)
  - training rewards are given per trick (8)

**Behaviour that follows automatically** (it is what maximising your own money means when four teams compete). Each case gets a hand-built test in section 14:
- **Block the team closest to the money.** If team A has one game in and stands on 36, every trick A takes lowers every other team's chance of being paid. So B, C and D will each prefer anyone but A to take a trick, even at a cost in their own points. This is the max^n "each team for itself" model producing coalition-like blocking only when it pays, unlike paranoid search, which assumes everyone is always against you.
- **Keep the match alive when behind.** A team far behind still has a real chance (scores reset each game, and a jink wins a game at once), so it plays to stop anyone else finishing and to give itself more hands.
- **Choosing the "king" (kingmaking)** [a team that is unlikely to win influencing *which* other team wins]. A money-maximising team always prefers the outcome that keeps its *own* chance alive. That usually means helping prevent the leader from closing out the match, rather than handing it to anyone.
  - Because each losing team pays the same whoever wins, the bot never favours a particular opponent for its own sake; it favours whatever keeps its own route to the money open.
  - This needs the match-equity model to be accurate even for small chances (6.6), so these small but real differences aren't drowned out by noise.
- **Risk-taking when it pays.** Chasing a jink, or holding back a top trump, is valued by its effect on the chance of the money, not on points.

---

## 6. Components in detail

### 6.1 Rules engine (one source of truth)

- Written in **Rust** [a very fast, crash-safe programming language].
  - **Native builds** for Windows and Linux, and **WebAssembly** builds [WASM: a portable compiled format that runs at near-native speed in browsers, in phone apps via a web view, and inside Cloudflare Workers].
  - Training, search, tournaments and the live app all run this same code, so the rules can never drift apart between them.
- Cards are stored as **bitboards** [one 64-bit number per set of cards, one bit per card]. 53 cards fit in a single number, so a question like "which of my cards can follow suit?" takes one processor instruction.
- It encodes all 41 rules:
  - trump order (red versus black trump suits), and the non-trump orders
  - the Joker and the A♥ are always trumps; with hearts as trumps there are 14 trumps
  - reneging with the top four trumps, including when a higher trump forces them out
  - robbing:
    - your house rule: the ace-holder robs at their own first turn and may burn any card except the ace of trumps (rule 21, H3; the 2026-10-13 no-ace rule was withdrawn on 2026-10-14)
    - the dealer's turned-ace and turned-Joker sequences happen at once; but after a turned Joker and an ordinary card, a dealer holding the ace of trumps robs at their own first turn, as the eighth card (H4, amended 2026-10-14)
  - after a game ends mid-hand, the deal always moves one seat left
  - burned cards stay hidden
  - the best-trump bonus: 11 points the first time a team takes it in a game, 10 after that
  - the game ends the moment a team reaches 41, even mid-hand
  - a jink (all 5 tricks to one team) wins the game; the match is the first to two games (at most five); the deal moves left
- **Legal moves only:** the engine offers only legal actions. Robbing is forced automatically; the only choice is which card to burn.
- **Rule switches** [settings, so a house rule can change without changing code] for any new ambiguity found while writing tests.
- **Deterministic by construction** [the same inputs always give exactly the same game]:
  - the engine uses whole numbers only
  - it uses our own small, pinned random-number generator, verified with golden-vector tests on Windows, Linux and WASM [stored known-correct outputs that every platform must reproduce]
  - no randomly ordered containers are used in game logic
  - any game can be replayed exactly from its seed and recorded actions; this is needed for debugging and for duplicate tournaments

### 6.2 Table manager and network ("tables" and "network")

- Copies the computer bridge championship setup. The referee deals, enforces the rules, and sends each seat only that seat's view.
- Each bot or human controls exactly one seat.
  - **Why:** peeking bugs become impossible by construction.
  - Humans, bots and different bot versions can share a table.
  - Tournaments and the live app use the same path.
  - A bot can take over a disconnected human's seat.
- **In the live app:** one **Durable Object** per table [Cloudflare's "one tiny, always-consistent server per game room", built for multiplayer games; players connect over **WebSockets**, which are always-open two-way connections]. It runs the Rust engine as WASM.
- **Bot backends are pluggable.** The table sends a seat's view to a bot backend and gets back a card. The backend can be:
  - a separate Cloudflare Worker [**serverless**: code that runs on Cloudflare's machines only when called, paid per use]
  - a CPU or GPU server
  - the player's own device (see the fairness rule in 7.2)
- **Messages** are versioned JSON, so any language can join: Rust, the app's front end, and test scripts.
- **In training:** self-play runs the same engine inside one program for speed, using the same **observation filter** [the code that hides other players' cards]. A test proves both modes give identical results.
- Seating: 8 seats with partners opposite (seats 0 & 4, 1 & 5, 2 & 6, 3 & 7). Every trick therefore alternates teams A, B, C, D, A, B, C, D.
- Every event is written to a **game record** [a compressed text log, one game per line, recording the format version, rules version, seed, deal and every action], used for replays, analysis and training data, including human games for phase E.

### 6.3 Card tracker (hard facts)

Each bot keeps track of what is *certain*:
- **Known cards:** its own hand, the turned card, and every played card.
- **The robber:** once someone robs, everyone knows they hold the ace of trumps. The taken card is *either* still in their hand *or* is their burned card, because your house rule lets them burn it.
- **No rob (incremental):** each player proves they lack the ace of trumps the moment they play their first card without robbing; the dealer has taken it the moment they rob a turned ace at once; only when all eight have played to trick 1 without a rob is it certain to be out of play (in the undealt pack, or in the dealer's burn made after a turned Joker before trumps were known).
- **Voids:** if a player plays a plain card of another suit when a plain suit was led, they hold none of the led suit. Trumping does *not* reveal a void (rule 14).
- **Trump limits:** if a player plays a non-trump when a trump was led, they hold no trumps except top trumps *higher than the led trump* (rules 16–18).
- **Out of play:** the undealt pack (12 cards, or 11 after a turned Joker) and the 1–2 burned cards are unknown and never revealed: 12 or 13 hidden cards in all. A burned card is never the ace of trumps (rule 21, H3), except possibly the dealer's burn after a turned Joker, made before trumps are known, and the robber's hand is exactly 5 of its 6 cards (original hand plus the taken card).

The output is, for every unseen card, the list of places it could be, plus how many cards each place holds.

### 6.4 Belief: who holds what

This combines three proven pieces:

1. **Belief head** of the network: for each unseen card, the probability that it is in each other player's hand, in the undealt pack, or burned. It is trained on self-play, where the true locations are known.
   - Solinas et al. 2019 trained the same thing on 20 million human Skat games, with a significant strength gain.
   - Learned Belief Search (Hu et al. 2021) did it in Hanabi.
2. **Constraint-respecting deal generator.** It builds complete **worlds** [one specific guess of where every hidden card is] that obey every hard fact, guided by the belief head (GIB used a constrained deal generator).
   - Enumeration instead of sampling only below a **counted threshold**: a dynamic program over the suit-length matrix counts the consistent worlds, and only when the count is below a tier budget (about 10^4) are they all listed (Rebstock et al. 2019). In 41 the hidden pool keeps the count astronomically large for most of the hand (about 1.5 × 10^36 at trick 1, 6.6 × 10^15 at trick 4), so this rarely fires; trick 5 is decision-free.
   - The MCMC sampler of Solinas et al. 2023 [MCMC: a random-walk method for sampling from tightly constrained possibilities] is the particle filter's move step, not a separate fallback; the fallback ladder below it is the constraint-only uniform belief, with the rung used logged per decision.
3. **Policy-based inference.** Each world is scored by how likely the other players' *actual* plays would be if that world were true (for example: "if West really held the 5 of trumps here, would they have played the 7?").
   - Rebstock et al. 2019 beat the previous best Skat inference this way. GIB used the same idea.
   - The weight of a world is the product of the policy-head probabilities of everyone's observed plays, divided by how likely the generator was to propose that world. This is **importance sampling** [a standard statistical correction: sample from a convenient guess, but still get the right answer].

**Engineering choices:**
- The worlds are kept as one **resample-move particle filter** [a running set of weighted worlds updated after every card, instead of rebuilt from scratch]: weights are multiplied by the policy head's probability of each observed action, resampling happens when the effective sample size falls below half the particle count, and the History Filtering Metropolis-Hastings swap kernel (Solinas et al. 2023) rejuvenates the set. The deal generator works on a suit-length matrix with a max-flow feasibility check (GIB; History Filtering), never on card-level rejection, which is kept only as the test oracle. This is standard (POMCP, Silver & Veness 2010; Gilks & Berzuini 2001) and saves compute at no quality cost.
- **Robustness, which matters even more with human players:** a model's opinion never drives a world's weight to exactly zero. A small floor remains (also a correctness requirement: the Metropolis-Hastings kernel needs full-support policies), the weighting mixes in the rules bot built from the published 25/45-family heuristics as the interim "human-like" play model until phase E provides a real one, and world weighting uses the same encoder as the blueprint, including the match state and the table-derived features, so score-driven plays (a top trump thrown away to block a team on 36) are read correctly. SPARTA found that beliefs collapse when partners deviate and this protection is missing; the valid-sample rate is monitored with a 99% target.

### 6.5 The neural networks ("the network")

- **One shared network for all 8 seats.** Every seat follows the same rules, so each hand gives 8 seats' worth of training examples. The input is rotated so "me" is always seat 0 and my partner is seat 4.
- **Inputs:** a complete, fixed-size description of everything this seat has seen. For each of the 53 cards: in my hand / turned / played, by whom, in which trick, in what order. Plus robbing and burning facts, card-tracker facts, and the match state: all four scores, games won, the derived first-bonus flags, and the dealer seat. Also, recomputed **after every trick** from the public state and the exact match-equity table in one lookup: each team's equity if the hand ended now, per-team "can clinch this hand" flags (modelling the jink, the immediate 5-of-trumps bonus, the deferred end-of-hand bonus and the first/later bonus), and optionally the best and worst reachable successor equities. The encoder is a list of active feature indices (sparse), which serves both the float and the integer inference paths. This makes "public enemy number one" explicit to the network. Because every played card records who played it and when, nothing about the history is lost.
  - **The input encoder is written once, in Rust.** Python gets already-encoded inputs from data files or from the Rust bindings, so training and play can never disagree.
- **Design:**
  - **Default:** an **MLP** [the simplest neural-network design: layers of weighted sums]. It is cheap on CPUs, phones and serverless, as in DouZero and the Skat card-location networks.
  - **Challengers:** a residual MLP (2–6 residual blocks; DouRN's gain came from stacking blocks on top of DouZero's MLP) at matched cost from M4, and a small transformer, as in GO-MCTS, for T2/T3 only, entering the pool by the first M7 generation. No recurrent history encoder: the full history is already in the input.
  - The tournament keeps whichever plays better for the same compute.
- **Outputs ("heads"):**
  1. **Policy head** [probabilities over the legal actions: which card to play or burn]. This is the **blueprint** every bot shares.
  2. **Value head** [the estimated chance that each of the four teams wins the match, and so the money, from this seat's view]. Four numbers that add to 1. Trained toward the exact match-equity table's value at the end of the current hand (a game or match ending mid-hand maps to its successor state), with the eventual match result kept as a held-out calibration check; the loss is cross-entropy on a four-way softmax, calibrated afterwards by temperature scaling and reported with log loss, Brier and adaptive-binned reliability down to 0.1–1%.
  3. **Belief head** (section 6.4): autoregressive with hard-constraint masking on T2/T3 (it assigns each unseen card in turn to one of the 7 other hands, the undealt pack, the robber's burn or the dealer's burns, masking full or forbidden locations, so every sampled world is legal and its proposal probability is exact); a marginal-plus-masking generator on T0/T1; per-card marginals as a diagnostic. Trained each generation on a mixture of league trajectories so it stays usable with partners who do not play like the champion.
  4. **Oracle value head**, used *in training only*: the same estimate, but given all cards face up. It grades decisions more accurately so learning is faster (PerfectDou's "perfect-training, imperfect-execution"; Suphx's oracle guiding). It never sees the real hidden cards; inside a sampled world it may evaluate that hypothesised deal as a rollout cut-off (one of three M5 evaluator arms), and the information-firewall test asserts its inputs come only from the sampled world.
  5. **Public money-chance head:** a separate small sub-module with a public-only input slice (so it cannot leak private cards), estimating each team's chance of the money after every trick (6.6). Its per-trick timeline is written into every game record, so you can see when each team became public enemy number one; whether its estimate is also fed back as a network feature is decided by SPRT.
  6. **Diagnostics-only auxiliary heads** ("tricks per team this hand", "game ends this hand"), added as an M6 ablation; their outputs are not the money-chance type and no solver, search or arena code can consume them.
- **Teacher and student sizes:** the strongest "teacher" network is trained first. Smaller "student" networks for phones and serverless are then trained to copy it. This is **knowledge distillation** [Hinton et al. 2015: a big network teaches a small one], which keeps as much strength as possible at each size.
- **Running the network everywhere** (section 11):
  - The MLP is computed by a small inference routine inside the shared Rust core. A plain, exact "reference" version is used for tests. Fast versions use AVX2/AVX-512 on servers and WASM SIMD on Cloudflare and phones [**SIMD**: processor instructions that do many multiplications at once], and each is checked against the reference.
  - Weights travel as **safetensors** files [a simple, safe, standard format for network weights].
  - The transformer challenger would run through **tract** [a pure-Rust engine that runs standard **ONNX** network files, an exchange format for trained networks], on every tier.
- **Symmetries:** interchangeable suits (for example clubs and spades when neither is trumps) are **canonicalised** [rewritten into one standard form, so each situation is learned once] or used as **data augmentation** [extra training on mirrored copies]. Red/black ordering and the A♥ rule are always respected.

### 6.6 Match-equity model ("what is each team's chance of the money?")

- **Input:** all four scores, games won and the dealer **team** (partners are a symmetry, so the seat adds nothing). Whether a team has already taken its first best-trump bonus this game is implied by its score (`score mod 5 == 1`), so it is a derived feature, not an input. **Output:** each team's chance of winning the match, which by section 5.3 is the same thing as its expected money.
- **Accurate even for small chances:** the table is exact given its sampled hand-outcome distributions (sample sizes and standard errors published per state), and the heads are trained with cross-entropy on a four-way softmax [a strictly proper loss that lives on the log-odds scale, a scale that stretches out very small and very large probabilities so they are learned precisely], calibrated by temperature scaling and checked with adaptive-binned reliability down to 0.1–1%, because blocking and kingmaking decisions depend on small but real differences. A sensitivity test perturbs the table in log-odds (about ±5% relative) and requires the section-14 scenario decisions not to flip.
- **Proven precedents:**
  - backgammon's **match-equity tables** [standard for decades: the chance of winning a match from any score]
  - Suphx's global reward prediction, which was central to its Mahjong success
- **How it is built (amended 2026-10-13):** it is an **exact table** computed by **backward induction** [solving from the end backwards, step by step] over the discrete match state: 15 reachable scores per team, games won, dealer team: 3.24 million states (about 810,000 after the four team rotations). Every hand adds at least 25 points or ends a game, so one backward sweep is exact given each state's hand-outcome distribution, which is the only estimated ingredient: measured by duplicate-seeded rollouts of bots that already use the finalised equities of the successor states (the backgammon Rockwell-Kazaross / XG2 procedure), iterated with a sweep cap and a logged maximum change, and called "self-consistent with bot version X". Version 0 is built at M1 from the rules bot, refreshed before the M4 teacher run and every generation; the table version hash is in every game record. It ships as a lookup table in the fixed-point money type on every tier.
- **Trick-by-trick money chances.** Between hands the exact table gives each team's chance of the money. *Within* a hand, the chances are re-estimated after **every trick** by two network heads, both trained toward the table's value at the end of the current hand:
  - a **public money-chance head**, which uses only what everyone can see (scores after this trick, tricks left in the hand, who leads, the best trump played so far and which team holds it, cards gone, games won, bonus flags, dealer). Every bot shares its view of who is "public enemy number one" and how close they are to grabbing the money this hand.
  - the training-only **oracle head**, which sees all cards, for the most accurate training signal

  Agreement with the table at hand boundaries is a standing test, not a separate mechanism.
- **Three uses:**
  1. scoring the end of every simulated line inside the search: end of hand, or a game or match ending mid-hand at the exact trick
  2. the training reward: the change in each team's chance of the money **after every trick**. This is **reward shaping** [extra, earlier feedback that provably does not change which strategy is best].
  3. making bots score-aware: block a team on 36, take risks when far behind, protect a lead

### 6.7 Exact "all cards face up" solver

Given one world (every card known), it computes the result of perfect play by all four teams, scored as **each team's chance of the money**. Every finished line (the end of the hand, or a game or match ending mid-hand) is valued by the match-equity model.

- **Four-sided search:** max^n, with each team maximising its own chance of the money, plus Sturtevant's last-branch and speculative pruning. The four chances always add to 100%, which is exactly what his pruning needs.
- **Transposition table** [a memory of positions already solved, so a position reached by a different play order is not solved twice], indexed with **Zobrist hashing** [a fast fingerprint of a position].
- **Equivalent cards** [cards that provably make no difference are treated as one; for example, two cards in the same hand become equal once every card ranked between them is gone], defined by exact 41 rule semantics (the 5 and the J are never equivalent because their bonus timing differs; J, Joker and A♥ force different sets; only the A♥ may be held back against the ace of trumps). Precedent: Furtak 2013. **Partition search** (Ginsberg 2001; Setrograde 2024/25) is deferred out of the critical path: its published gains come from two-sided alpha-beta proof trees and no publication applies it to max^n or vector values; it returns only with its own exactness gate and measured factors.
- **Transposition key:** remaining hands, leader, tricks per team this hand, whether the 5 has been played, the best trump so far and its team, and the per-team bonus flags, keyed at trick boundaries. The money value is not additive in points, so the in-hand score state can never be dropped from the key.
- **Move ordering** [try the most promising card first, so more of the search can be skipped]: Furtak's "fastest-cut-first" ordering, and killer moves.
- **Backup rule and ties (amended 2026-10-13):** ties are pervasive in 41 because a line's value depends only on who won each trick, the bonus state and whether a game ended. At the searching team's nodes the solver maximises its own money-chance entry. At the other three teams' nodes it uses, before M4, a fixed position-only successor order (the documented tie rule under which "pruning on == pruning off" is asserted) and, from M4, a policy-weighted expectation with a small floor so every legal card has non-zero weight. That expectation is the actual Prob-max^n idea (Sturtevant, Zinkevich & Bowling 2006): an expected-value backup over a model of the other players, not a tie-break. Pruning is re-proved or disabled in expectation mode; M2 reports the fraction of tied nodes and how often the root choice flips under a different tie order.

**Roles:**
1. the "teacher" for bootstrap training
2. a baseline opponent in tournaments
3. a **post-mortem checker** [after a game, it finds every play that gave away chance of the money, judged with all cards visible; GIB and Furtak used this kind of check, counted in tricks or wins, to measure error rates]
4. a candidate engine for CPU-only tiers
5. exact play in the last tricks, but only if tournaments show it helps

### 6.8 How a bot thinks before each card (decision-time search)

**Main engine: recursive Monte Carlo with neural rollouts** [a rollout means "play the card, then simulate the rest of the hand"]:

1. Take the current weighted worlds from the belief. The number per tier comes from a measured decision-agreement curve, starting from the published plateaus (20 worlds for 5-card solver-based PIMC in bridge, 160 in Skat); servers may go higher while money still rises under SPRT. One shared world pool serves every candidate card.
2. For each legal card (at most 5) and each world: play that card, then let all 8 seats finish the hand using the policy head. Each seat sees only its own information in that world; nobody peeks. Score the end with the match-equity model.
3. Play the card with the best average chance of the money for my team.

Recursion depth is **one** by default (depth two is an explicit experiment): both published recursive wins used one level (Furtak & Buro 2013; Bouzy et al. 2020 at 5 cards: +0.24 tricks per deal with a solver inner player, nothing beyond level two). The inner player is the policy network at tricks 1–2 on every tier; from trick 3 on the T2/T3 tiers it is PIMC with the exact solver inside (Furtak's recursive form), and T0/T1 use policy rollouts throughout. The owner's time budget is 8 s on trick 1 and 5 s on trick 2 on every tier (tricks 3–4 assumed at most 3 s); tiers differ by compute per second, and the T3 budget is fixed in worlds and rollouts before M5.

**Why this is proven:**
- Furtak & Buro 2013: recursive Kermit beat PIMC Kermit at its best, and kept scaling with hardware.
- BBO GIBBO uses neural rollouts in production.
- SPARTA set the Hanabi record using rollouts with the blueprint.
- Suphx adapts its policy during play with simulations.

**Why it beats plain PIMC for 41:**
- There is no strategy fusion.
- Simulated opponents cannot magically see your hidden 5 of trumps, so holding back top trumps (reneging) and deception are valued correctly.
- Partner signalling is part of the simulation.

**Anytime by design** [it can stop whenever its time or CPU budget runs out and return the best card found so far]. This is what lets one engine serve a phone and a GPU server.

**Safeguards and speed-ups (all proven):**
- **Common random numbers** [every candidate card is tested on the same worlds with the same random choices, so differences reflect the card, not luck]. GIB tested every move on the same deals.
- **Adaptive sampling** [spend more simulations on close calls and stop early on clear ones]. UCB pruning [a standard rule for dividing trials between options] cut SPARTA's rollouts about 10×.
- **Deviation threshold:** overrule the blueprint only when the gain clearly exceeds noise (SPARTA).
- **Counted enumeration:** list every remaining world only when the counted number of consistent worlds is below the tier budget (section 6.4); trick 5 is decision-free.
- **Forced moves:** a single legal card, or interchangeable cards, needs no search.
- **Truncated rollouts** at every tier, not only small ones: stop the simulation early and let a value head estimate the rest (Learned Belief Search found 16-step rollouts bootstrapped by the value head slightly but consistently better than full rollouts). Which cut-off evaluator to use (the oracle head on the sampled world, the seat-view head, or the sum of seat-view heads), whether truncation beats full rollouts, and whether all seven bots searching independently beats one searcher per team are mandatory M5 experiments, decided by SPRT per tier.
- **Common random numbers from counter-based seeds:** the noise for (decision, world, seat, card index) is SplitMix64 of those indices, so every draw keeps the same role across the candidate cards; network batches are formed asynchronously from whichever rollouts are ready (over-provisioned 2–3× the thread count), and budgets are defined in worlds and rollouts, never wall time, so decisions are byte-identical regardless of completion order.

**Alternatives kept in the tournament pool, so data decides:**
- plain PIMC with the exact solver (the GIB/Kermit style)
- a hybrid that uses PIMC early in the hand and recursive search late (Furtak's suggestion)
- network-only play, with no search (DouZero-style; GO-MCTS showed this alone was strong in Hearts)
- an upgrade path: **MCTS** over my own future decisions (GO-MCTS / information-set MCTS) [MCTS = a search that grows a tree of the most promising move sequences]

### 6.9 The burn decision

- Burn choices are evaluated by the same search: up to 6 candidates (every card in the burner's hand except the ace of trumps: rule 21, H3). The robber's burn and first card are searched as one joint decision over the shared worlds (under H2 nothing happens between them), and recorded as two actions. Sampled worlds carry burn branches for seats that have not yet robbed, so every world names its burned cards.
- Because the ace-holder robs at their own first turn, they choose the burn after seeing the trick-1 cards already played. The robber's card tracker records the burned card. Everyone else treats it as one more hidden, out-of-play card, which may be the taken card.

### 6.10 Partner coordination safeguards

These address SPARTA's warning that partners searching independently can mislead each other:
- All bots share the blueprint, so partners can read each other.
- Search overrules the blueprint only for clear gains.
- Search choices are regularly **distilled** back into the blueprint (section 8, phase C), so the blueprint keeps matching how bots really play and partners' inferences stay accurate.
- Beliefs keep a small floor, so a surprising partner play, from a bot or a human, never breaks inference.

---

## 7. Run anywhere: one engine, four deployment tiers

### 7.1 The tiers

| Tier | Where | Network | Search budget | Built with (section 11) |
|---|---|---|---|---|
| **T0 Device** | Player's phone or browser | Smallest student | Small, time-limited | WASM via wasm-bindgen (packaged with wasm-pack). Single thread, with WASM SIMD. Only allowed at single-human tables (7.2). |
| **T1 Serverless** | Cloudflare Worker, called by the table's Durable Object | Small student | A CPU budget per decision, on a single thread (Workers run one thread; up to 5 min CPU per request on the paid plan) | `worker` crate, with WASM SIMD on. Pricing per Cloudflare docs (Oct 2026), estimate only: at 1 CPU-second per bot decision, about $0.00002 per decision, roughly 1 cent per full match with 7 bots. |
| **T2 CPU server** | Website server, Cloudflare Containers (up to 4 vCPU each), or a rented many-thread machine | Medium or teacher | Many worlds, using all cores | Native Rust bot server (tokio + axum). Monte Carlo is **embarrassingly parallel** [the guesses don't depend on each other], so it splits across cores easily. |
| **T3 GPU server** | Cloud or home GPU | Teacher (or the transformer, if it wins) | Full strength: thousands of worlds, network calls batched across all tables | Same server; the GPU is used only if a large network wins its tournaments |

### 7.2 Rules that make "any direction" safe

- **Same code and same model files everywhere.** Only two knobs change: network size and search budget. The interface to the table is identical on every tier.
- **Fairness rule:** a bot running on a player's device would put that bot's hidden cards on the player's device. So:
  - **on-device bots are used only at tables with a single human** (practice and offline play)
  - **at tables with two or more humans, bots run on Cloudflare or a server**

  This is the championship rule "a computer only ever sees its own seat", applied to the app.
- **Serverless scaling options, measured at M8:** (a) fan-out of one decision's worlds across parallel Worker calls; (b) a native bot server in a Cloudflare Container (standard-4: four native threads, 12 GiB, AVX2, the same price per busy vCPU-second, but memory billed while awake and 1–3 s cold starts) multiplexing several tables. Neither is built before M8; measured strength per dollar and latency decide. Worker limits (isolate memory, script size, startup time, Durable Object attachment size) are re-verified against live documentation at M3b and the design assumes the stricter published values: weights loaded lazily from R2/KV, table state persisted in Durable Object storage after every action and rebuilt by replay, no WASM threads.
- **Strength–cost curves:** the tournament system (section 9) measures each tier's strength against the champion at several budgets. You then see exactly what a server buys before paying for one.
- **Difficulty levels** come for free: smaller budgets, or blending toward the human-like model from phase E.

---

## 8. Training pipeline

**Phase A: bootstrap (no human data needed).**
- The PIMC teacher plays about 10^5 to 2 × 10^6 hands, stratified over match states and tricks so the late-match blocking states are covered: exact solves from trick 3, rollout-PIMC with the rules-bot inner player at tricks 1–2 (no network exists yet; Bouzy et al. 2020 found this on a par with exact-solver PIMC at 5 cards). Per-decision action-value vectors are recorded; training stops at the validation plateau; the candidate is gated by SPRT. Teacher generation continues as a background chunk type. (Amended 2026-10-13: the exact solver is unaffordable at trick 1, section 6.7.)
- The network is trained by **supervised learning** [learning to copy examples]:
  - the policy head copies the teacher's choices
  - the value and oracle heads learn the money outcome: which team eventually won the match. Hands are played inside full matches; for a hand not followed to the end of its match, the match-equity model's estimate is the target.
  - the belief head learns the true card locations
- Precedent: Skat policy and inference networks were trained this way from human games. Here the "expert" is the teacher.
- The first match-equity model is built from these games.

**Phase B: self-play reinforcement learning.**
- **Algorithm:** **PPO** [a standard, stable method that improves the strategy in small, safe steps] with a perfect-information critic (PerfectDou): generalised advantage estimation, discount 1.0, λ ≈ 0.95, a four-way softmax critic on the all-cards input, value-loss clipping (AlphaHoldem's Trinal-Clip), target-entropy control, actor-learner lag at most 1. The oracle input never reaches the policy head at execution, and no policy is ever distilled from an all-cards-visible policy (Suphx reports that fails).
- The challenger is **Deep Monte-Carlo** [learn each move's value directly from final game results] in its strongest published form, OADMCDou (IJCAI 2024): oracle features with dropout annealed from 1 to 0, and clipped value updates; tournaments decide.
- **Reward (precise form, amended 2026-10-13):** discount 1; for each team i and each trick, `r = Φ_i(after) − Φ_i(before)`, where Φ_i is that team's chance of the money from the oracle head within a hand and from the exact match-equity table at hand boundaries, frozen for the current iteration; Φ(match end) = 0 and the actual money result (1 or 0) is the only terminal reward, so every episode's shaped return equals (result − Φ(start)). The Φ values used are stored per trick in the game record so old data is never re-shaped with a newer head. A unit test checks that returns telescope; a short unshaped fine-tune ends each generation; promotion is on unshaped money only. A game or a trick is never rewarded for its own sake (5.3).
  - This is **potential-based reward shaping** [giving feedback as the change in an estimated value is proven not to change which strategies are best; it only makes learning faster and sharper]: Ng, Harada & Russell 1999 for one agent; Lu, Schwartz & Givigi 2011 (per-player potentials are necessary and sufficient in n-player general-sum games) and Devlin & Kudenko 2011/2012 (potentials that change during training keep the guarantee if each transition uses the potential in force when it was visited) for our four-team case. The theorem protects the *set* of equilibria; which one self-play reaches is governed by the shared blueprint and distillation.
  - The bot learns precisely which tricks, and which cards inside them, swung the money, including stopping a new leader the moment it takes first place.
- **League** [train against a pool of past versions and special "exploiter" bots, not only the latest self, to avoid going in circles or growing blind spots], specified: the K best past champions by money rating, plus checkpoints at several skill levels from each run (initial, mid-strength, converged), plus rule-based and noisy bots; partners and opponents are drawn from the pool for a tournament-tuned fraction of games (start at 10–30%). A cycling detector runs every generation: a duplicate SPRT against the frozen pool and the exploitability probe; a generation that beats its predecessor but loses money against the pool is not promoted. The same pool is the phase-E cross-play suite and population (Fictitious Co-Play: skill-level diversity is what makes a bot a good partner to strangers; AlphaHoldem's K-best self-play; AlphaStar). Phase B starts with a KL penalty toward the teacher that decays to zero (AlphaStar's precedent), and one budget-capped from-scratch arm runs as a control at M6.

**Phase C: search distillation.** This is **Expert Iteration**, the AlphaZero loop [the slow, strong search player generates games; the fast network learns to predict what the search chose; repeat, so a better network makes a better search].
- Proven by AlphaZero (chess, Go, shogi) and ExIt. The network is trained on the search's **full per-card distribution** (soft targets stored in the training shards; Expert Iteration measured about +50 Elo over chosen-action targets), the belief head is retrained on search-generated games each generation, and KataGo's playout-cap randomisation (full search and recording on a fraction of decisions, fast search elsewhere; 1.37× measured) is A/B-tested at M7.
- SPARTA proved search does not make the blueprint worse (apart from sampling error).

**Phase D: students and refresh.**
- Distil the teacher into tier-sized student networks.
- Retrain the belief head and the match-equity model each generation.

**Phase E: human compatibility (in scope, because tables are mixed).**
- **From launch, before any human data exists:**
  - robust inference (floor plus human-like mixture, 6.4) and the deviation threshold (6.8)
  - **cross-play tests** [pairing our bot with partners it was not trained with: older versions, rule-based bots, deliberately noisy bots], the standard way to check that a bot is a good partner to strangers (zero-shot coordination research, Hu et al. 2020)
- **As the app collects human games** (with players' consent):
  - train a **human-behaviour model** by **behaviour cloning** [learning to imitate recorded human decisions], used to read human partners and opponents (BBO GIBBO and Rebstock 2019 trained on human games)
  - add **human-regularised search** (piKL, Jacob et al. 2022) [search that still maximises the chance of the money, but prefers human-style plays when they cost little or nothing, so bots stay strong and understandable as partners]
  - test bots partnered with the human model and with real players

**Rule:** no phase replaces the champion unless it wins the tournament gate in section 9.

---

## 9. Tournaments and evaluation ("tournaments")

**The score that decides everything: money per match.**
- The primary result is average money won per match per team (in stake units), which is equivalent to match-win rate.
- Game wins, points and tricks are reported only as diagnostics. A bot that wins more games but less money is worse.

**Formats:**
- **Duplicate deals.** The same seeded decks are replayed with the teams rotated through the seats, so luck of the cards cancels out. The deck for hand k of a match is derived from (block seed, k), so rotations keep sharing cards after matches diverge. The unit of analysis is the **block**: for candidate-versus-incumbent, the 14-way block (every assignment of the two programs to the four team slots except all-candidate and all-incumbent; Sturtevant 2003; GO-MCTS 2024), with equal weights across the 1v3, 2v2 and 3v1 mixes (owner decision) and per-mix diagnostics; the 4-rotation 1v3 block is the cheap pre-screen. Identical bots must score exactly 0.000 against each other (the canary, re-run after any engine or evaluator change).
- **SPRT gate** [the sequential statistical test top chess engines use (Stockfish's Fishtest): keep playing only until it is statistically clear whether the new version wins more money, then stop], in its generalised form on block scores (log-likelihood ratio from the sample mean and variance), with α = β = 0.05 and the effect size in stake units fixed in advance, two stages (a cheap search budget that can only reject, then the full budget), and a hard cap after which a result is "inconclusive". Every change must pass it. The block standard deviation is measured at M1 so test sizes can be budgeted.
- **AIVAT-style luck removal** [an unbiased correction that subtracts the estimated luck of each deal]: a per-hand chance term using the training-only oracle head with a baseline averaged over fresh shuffles of the same state, plus the known-strategy action term at bot-only tables, available from M4 and checked with a null test. Burch et al. 2018 report a 3× lower standard deviation bot-versus-bot (the 44× figure is DeepStack's human match, where one strategy was fully known); MIVAT in six-player poker gave about 20%. 41's figure is measured at M1, not assumed.
- **Multi-team ratings** for long-term tracking: an offline Plackett-Luce model [a standard model for ranking several competitors at once, whose output is literally a predicted chance of winning the table] fitted by maximum likelihood, and OpenSkill or TrueSkill online with ranks [1, 2, 2, 2]; Elo [chess-style strength numbers] is two-player and appears only in two-program diagnostics.
- **Championship format** for major releases: a round robin of top candidates, then a knockout, scored on duplicate (as in the WCBC).
- **Tier ladders:** each deployment tier has its own champion at its own budget, plus strength–cost curves against the T3 champion (7.2).

**Measurements and diagnostics:**
- post-mortem error rate per trick, against the exact solver
- belief accuracy: how often the true deal is among the highly weighted worlds (Rebstock's measure)
- **calibration** [when the bot says 30%, is it right 30% of the time?]
- 41's leaf correlation, bias and disambiguation (Long et al.)
- cross-play scores with unfamiliar partners (phase E)
- **Exploitability probes** [a dedicated bot is trained purely to beat the frozen champion; an easy win reveals a weakness to fix] (approximate best response, Timbers et al. 2022; AlphaStar's exploiters)

**Baselines:** a random legal-move bot, a simple rules bot, the PIMC teacher, every previous champion, and later the human-behaviour model.

---

## 10. Engineering savings (faster, with identical answers)

| Technique | Saves | Why quality is unchanged |
|---|---|---|
| Bitboards, plus incremental hashing | Engine and solver time | Same rules, faster arithmetic |
| Equivalent cards, trick-boundary transposition table (key includes the in-hand score state), move ordering, speculative pruning; partition search deferred | Solver work, by factors measured at M2 (max^n admits only shallow pruning, so expect far less than bridge/Skat) | Skipped work provably cannot change the answer under the documented tie rule |
| Symmetry canonicalisation (seat rotation, interchangeable suits) | Training data and solver memory | Equivalent situations are learned and solved once |
| Particle-filter reuse | Belief rebuilding | The same posterior, updated incrementally |
| **Batching** [sending many world simulations through the network together] and incremental evaluation along a rollout | Rollout cost | The same network outputs, computed once |
| Counter-based common random numbers, adaptive sampling (100-rollout floor, then drop options two standard deviations behind), counted enumeration, asynchronous batching | Number of simulations and idle cores | The same decision, with less noise per simulation |
| Fixed-point money chances with index-ordered reductions | Nothing directly; it makes solver, teacher and tournament results byte-identical across thread counts, CPU classes and WASM, so re-runs and resumes are exact | Same numbers, exact by construction |
| Duplicate blocks, generalised SPRT, AIVAT | Tournament games, by a factor measured at M1 (duplicate is the big win; AIVAT 20–60% in standard deviation is the realistic expectation) | The same statistical confidence |
| **Actor–learner setup** [many CPU workers play games while the GPU (graphics card used for fast neural-network maths) trains at the same time] | Training wall-clock time | Standard practice (DouZero, IMPALA, AlphaZero); same learning |
| Distillation into tier-sized students | Runtime cost on phones and serverless | The best possible strength at each size. The full-strength teacher is unchanged. |
| Run operations (section 12): saturated CPUs, a busy GPU, everything finishing together, measured value per dollar | Rented-machine cost | The same work, without paying for idle machines |

---

## 11. Tech stack: latest compatible stable releases (researched 13 Oct 2026)

**Principles:**
- **Pin exact versions** in **lockfiles** [files that list the exact version of every dependency, so every machine installs the same]: Rust's `Cargo.lock` plus `rust-toolchain.toml`, and Python's `uv.lock`.
- **Upgrade deliberately:** one component at a time, and only if the tournament gate shows nothing got worse.
- Where the very latest release breaks compatibility with your hardware, we deliberately stay one step behind; this is called out below.

### 11.1 Rust side (engine, solver, search, belief, tables, bot server, WASM)

| Component | Choice | Version (release date) | Why / notes |
|---|---|---|---|
| Compiler | Rust stable, **edition 2024** | **1.99.1** (due 2026-10-15; use 1.99.0, from 2026-10-01, until then) | 1.99.1 fixes an x86-64 miscompilation of `bool` values returned across foreign-function calls, which affects the Python bindings. AVX-512 has been stable since 1.89. The toolchain is pinned in `rust-toolchain.toml`. |
| Parallelism | rayon | 1.12.0 (2026-04-14) | **Work-stealing thread pools** [idle cores automatically take work from busy ones]. Thread count is set explicitly (section 12). |
| Concurrency utilities | crossbeam | 0.8.5 (2026-09-05) | Queues between workers |
| Memory allocator | mimalloc, **deferred** until a profile shows allocator pressure (the engine allocates nothing in hot loops) | 0.1.52 (2026-05-22) | A faster allocator for many threads. It needs a C compiler: MSVC on Windows, gcc in Docker. |
| CPU-specific speed-ups | multiversion | 0.9.0 (2026-09-06) | Runtime choice of AVX2/AVX-512 versions of the hot loops, so one binary runs on any rented x86 CPU |
| Network maths | **Own MLP routine** using matrixmultiply | 0.3.11 (2026-07-14) | matrixmultiply has AVX-512 kernels and a WASM SIMD (simd128) matrix-multiply kernel. Our plain reference routine is used for tests. Float outputs are reproducible per (binary, CPU class) but not bit-identical between x86 (FMA kernels) and WASM simd128 (no FMA); the M4 gate is "within tolerance" for float. An NNUE-style integer path (int16 first layer, int8 hidden, int32 accumulation, incremental first-layer accumulator, weights clamped during training) is the gated student candidate after M5 and is bit-identical everywhere. |
| Network weights | safetensors (Rust and Python) | 0.8.0 (2026-06-09) | One small, safe file per model, readable everywhere |
| Transformer challenger | tract (`=0.23.8`, exact pin), **deferred** until a transformer challenger exists (it must enter the T2/T3 pool by the first M7 generation) | 0.23.8 (2026-09-21) | Pure Rust, runs ONNX on native and WASM with SIMD kernels. 0.23 renamed many APIs, hence the exact pin. |
| Native ONNX cross-check | *(dropped 2026-10-13)* | — | `ort` was still a release candidate with no wasm32 target; the Python side's onnxruntime 1.31 is the ONNX cross-check instead. |
| Deterministic maths | libm | 0.2.16 | Same `exp`/`tanh` results on every platform. Rust's built-in versions may differ by platform. |
| Random numbers | **Own pinned generator** (SplitMix64 seeding plus xoshiro256** or ChaCha8), own shuffle | — | The `rand` crate (0.10.3) explicitly makes no promise of identical sequences across versions. A tiny in-house generator with golden-vector tests gives replayable games forever. |
| Data (de)serialisation | serde 1.0.229, serde_json 1.0.154 | 2026-07 / 2026-10 | Game records and protocol messages as JSON |
| Compression | zstd 0.14.2 (native); ruzstd 0.9.1 (pure-Rust decoder for WASM), **deferred** until a Worker must read `.zst` | 2026-10-10 / 2026-10-11 | Python 3.14 reads zstd natively with `compression.zstd` |
| Analysis exports (optional) | arrow / parquet, **deferred**: Parquet stays on the Python side (polars/pyarrow) until a Rust consumer exists | 60.0.0 (2026-09-15) | Only for ad-hoc analysis tables |
| Python bindings | PyO3 0.29.3 + maturin 1.15.0 | 2026-09-30 / 2026-08-24 | Lets Python call the Rust core: the encoder, the engine for tests, data tools |
| Cloudflare | `worker` + `worker-build` | 0.8.7 (2026-09-25) | Rust Durable Objects **with WebSocket hibernation** [idle tables sleep cheaply but keep players connected] |
| Browser/phone packaging | wasm-bindgen 0.2.129; wasm-pack 0.15.0 | 2026-09-25 / 2026-05-15 | WASM SIMD on (`+simd128`). "Relaxed SIMD" stays off, because it gives non-deterministic results. Single-threaded. |
| Native bot server | tokio 1.53.2 (LTS to Sep 2027), axum 0.8.9 | 2026-10-03 / 2026-04-14 | HTTP/WebSocket server for T2/T3 |
| Testing | proptest 1.12.0; cargo-nextest 0.9.148; insta 1.49.0; criterion 0.8.2; cargo-fuzz 0.13.2 (Linux only) | 2026 | Property tests, a fast test runner, snapshot tests, benchmarks, and fuzzing [feeding random junk to find crashes] |
| Avoid | bincode (discontinued, RUSTSEC-2025-0141); `rand::StdRng` for anything replayable; randomly seeded `HashMap` in game logic; `target-cpu=native`; PEXT/PDEP bit instructions in hot loops (very slow on older AMD EPYC) | — | Each would break portability, determinism or speed on some rented machines |

### 11.2 Python side (training, analysis, launcher)

| Component | Choice | Version (release date) | Why / notes |
|---|---|---|---|
| Python | **3.14.x** (not 3.15) | 3.14 line | 3.15.0 came out on 9 Oct 2026 and isn't yet covered by onnxruntime or `torch.compile`. 3.14 is the newest fully supported line. |
| Env and package manager | **uv** | 0.13.0 (2026-10-09) | One cross-platform `uv.lock` for Windows and Linux; it also installs Python itself. uv 0.13 defaults to 3.15, so 3.14 is pinned in `.python-version` and `requires-python = "==3.14.*"`. |
| Deep learning | **PyTorch 2.14.1 exactly** | 2026-09-30 | **Deliberately not 2.15** (due ~2026-10-28), which drops Pascal and Volta GPUs (GTX 1080 Ti, P40, P100, V100), common cheap vast.ai cards. |
| — GPU build (vast.ai default) | `torch==2.14.1+cu126` | — | Supports GPU generations Maxwell to Hopper (compute capability 5.0–9.0). Needs a host driver that supports CUDA 12.6 (filter `cuda_vers>=12.6`). Does not run RTX 50-series (Blackwell) cards: those get the build below. [**CUDA**: NVIDIA's GPU software layer; each GPU generation needs a matching build] |
| — Newest-GPU build (Blackwell: RTX 50-series, B200; owner, 2026-10-17) | `torch==2.14.1+cu130` | — | Same PyTorch 2.14.1 for CUDA 13.0 (dependency group `cu130` of `python/nn/uv.lock`, in the same image as `cu126`). The runner picks it when the first GPU's compute capability is 10.0 or above and `cu126` below that (CUDA 13 drops cards older than Turing, 7.5, so `cu126` stays for the GTX 10-series and V100). Needs a host driver that supports CUDA 13.0 (vast's `cuda_max_good` ≥ 13.0), which the launcher checks for a Blackwell offer. **Opt-in per run until a smoke test has proven the image** (`machine.blackwell: true` in the manifest, or `--allow-blackwell`; `docs/vast/cu130-smoke.manifest.json`): by default the launcher still filters `compute_cap<1000` and refuses a Blackwell offer, so a run with an older runner binary never lands on one. |
| — Laptop | `torch==2.14.1+cpu` | — | Windows wheel confirmed |
| Number formats by GPU | FP32 on Pascal (1080 Ti/P40); FP16 autocast on Volta/Turing; BF16 + TF32 on Ampere+ | — | **Mixed precision** [using smaller number formats for speed where the GPU handles them well]. The trainer picks automatically from the detected GPU. `torch.compile` is not relied on (it won't run on Pascal). |
| ONNX export (transformer challenger only) | onnx 1.23.2; onnxscript 0.7.2; onnxruntime 1.31.0 (validation) | 2026-10 | The new exporter (`dynamo=True`, the default). Target opset 18–20, `external_data=False` (single file for tract/WASM), `dynamic_shapes` |
| Arrays and data | numpy 2.5.4; pyarrow 26.0.0; polars 2.0.x (fallback 1.44.2) | 2026-10 | Training shards are read memory-mapped [straight from disk without copying] |
| Experiment logs | Per-run folder: `config.json`, `metrics.jsonl`, final Parquet; optional TensorBoard 2.21.0 | — | Plain files that sync to cloud storage. No server needed on throwaway machines. |
| vast.ai control | `vastai` CLI / Python SDK | 1.8.3 (2026-10-02) | Search, rent and destroy machines from scripts |

### 11.3 Infrastructure

| Need | Choice | Notes |
|---|---|---|
| Code | **Git**, in a **new private GitHub repo `Guillaumeds/41Bots`**, separate from `fivefingers` (section 0.1) | `.gitattributes`: `* text=auto`, `*.sh text eol=lf`, so Windows line endings never break Linux scripts |
| Secrets | GitHub Actions repository secrets in the new repo. New least-privilege keys: an R2 key scoped to the project bucket; a Workers deploy token later (M8). The vast.ai API key stays on the laptop. | Never reuse the app's Cloudflare token. Rented machines only receive the bucket-scoped R2 key. |
| **CI** [automatic build and test on every change] | GitHub Actions | Builds and tests on Windows, Linux and WASM. Builds the Docker image. |
| **Docker image** [a packaged, ready-to-run copy of the whole software environment] | A multi-stage build on a slim `nvidia/cuda:12.6.3-base-ubuntu24.04` base, with the uv-locked Python env (cu126), rclone and the vastai CLI; **recommended: a public, secret-free base image** with the versioned Rust binaries (x86-64-v2 and v3 builds) and the project's Python wheel fetched from R2 by content hash at start-up, so rented machines never need a registry credential (owner to confirm) | Hosted on **GHCR** (GitHub's container registry; storage and bandwidth "currently free"). vast.ai does not bill rental time while an image downloads, but storage and inbound bandwidth are billed, so the image stays slim. Built on tags or manual dispatch from a digest-pinned base. Tagged with date and commit, never `latest`. |
| Permanent storage | **Cloudflare R2** | S3-compatible, **no download (egress) fees**, $0.015/GB-month. One bucket is the single source of truth for models, data, records and run results. vast.ai volumes stay on one physical machine, so they are useless with a new machine each day. |
| Sync tool | **rclone** 1.75.2 (2026-10-09) | One tool for Windows and Linux, with a dedicated R2 provider. On rented machines it is configured entirely from environment variables, so no config file is left behind. |
| Laptop | Rust (MSVC toolchain + Windows SDK), uv, rclone (`winget`), vastai CLI, Git. Optional: WSL2 + Docker Desktop to test the image locally | Docker Desktop is free under 250 employees and under $10M revenue |

---

## 12. Run operations: using rented machines efficiently

### 12.1 The model: a "run" is one self-contained package

- **Run manifest** [a small file describing one run]: what to do, which inputs (by content hash), which outputs, and a **deadline or budget**. For example: "generate teacher games, train network v3 on them, and run the SPRT tournament for candidates A/B, until 06:00".
- **The same manifest runs on the laptop or on vast.ai.** The runner adapts to whatever hardware it finds.
- **Everything persistent lives in R2.** A rented machine holds nothing that matters beyond its last checkpoint.

### 12.2 Choosing the cheapest machine for the work (launcher on your laptop)

- **Search offers:** the `vastai search offers` filters in the table below, plus a minimum `cpu_ram`, disk space and disk speed.

  | Filter | Why |
  |---|---|
  | `cpu_cores_effective >= N` | Only the cores actually rented count |
  | `gpu_frac=1` | **Renting all of a machine's GPUs is what gives you its full CPU.** Renting 1 of 4 GPUs gives only about ¼ of the CPUs. |
  | `compute_cap>=600 compute_cap<1000 cuda_vers>=12.6` (a run that opts in to Blackwell, `machine.blackwell: true`, drops `compute_cap<1000`, and the launcher then also requires `cuda_max_good` ≥ 13.0 for a card of 10.0 or above) | Matches our PyTorch builds (cu126 below 10.0, cu130 from 10.0, opt-in until the smoke test) |
  | `reliability>0.9` | Avoids flaky hosts (owner, 2026-10-16: the floor is 90%; it was 98%) |
  | low `inet_up_cost` / `inet_down_cost` | Hosts charge per GB, and we upload results |

- **Rank by price per unit of work, not price per hour:**
  - At first: `dph / cpu_cores_effective` [dph = dollars per hour].
  - Each run starts with a 30–60 second **micro-benchmark** [a quick speed test] and records "games per second per dollar" for that CPU model in R2.
  - The launcher learns which CPU models really give the best value.
  - **Owner's rule (2026-10-13):** choose by cost per unit of work, not by price per hour; but among offers whose cost per unit of work is within 20% of the cheapest, take the fastest (most work per hour), because finishing sooner is worth a small premium. The 20% margin is a launcher setting. On-demand only: no interruptible (bid) machines, which get killed when outbid.
- **Rental type: on-demand only** (owner decision 2026-10-13). **Interruptible** [cheaper bid machines whose processes are killed when outbid; the disk is kept but resumption is not guaranteed] is not used. If a machine dies anyway, the launcher rents a new one that resumes from the last checkpoint, instead of waiting.

### 12.3 Inside the machine: start-up

1. Record the hardware: CPU model and flags, the container's CPU quota (`/sys/fs/cgroup/cpu.max`), RAM, and the GPU (`nvidia-smi`).
   - Core-count functions can report the whole host rather than your share. So the launcher passes `NCPU` (the offer's effective cores), and the runner sets `RAYON_NUM_THREADS`, `PYTHON_CPU_COUNT` and the PyTorch thread count explicitly.
   - This prevents **oversubscription** [more busy threads than cores, which slows everything down].
2. Pick the binary build. Use v3 (AVX2/FMA/BMI) if the CPU flags allow, otherwise v2. AVX-512 is used automatically inside hot loops where present.
3. Pull the manifest, the done-list and the latest checkpoints from R2.
4. Run the micro-benchmark to size work chunks and batch sizes for this exact machine.
5. **GPU check:** confirm the installed PyTorch build lists this GPU's architecture (`torch.cuda.get_arch_list()`), then run one tiny training step. If either fails, run CPU-only work and flag the machine type in R2, so the launcher avoids it next time.

### 12.4 Inside the machine: keeping CPUs near 100% and the GPU busy

- **Small work chunks**, each about 2–5 minutes, with seeds derived from (run, chunk) so any chunk can be redone identically. Examples: 10,000 teacher hands, 200 duplicate matches, one batch of search games. Small chunks mean:
  - no core waits on one giant final job
  - a dead machine loses only minutes
  - all streams can stop together
- **Rust worker processes** pull chunks from a local queue, each using rayon pools sized to the cores. Priorities when choosing the next chunk:
  1. data the GPU trainer needs next
  2. tournament (SPRT) matches
  3. background diagnostics, such as post-mortem error analysis, to fill any idle cores
- **GPU trainer** (one Python process) runs at the same time, training on shards as they arrive. This is the IMPALA/DouZero **actor–learner** split, with data passed through the local disk. When one model is too small to fill the GPU, several trainings run side by side (teacher, students and belief variants), so the GPU isn't idle.
- **Typical run packages, designed so CPU and GPU are both busy until the deadline:**
  - **teacher data run:** CPUs play PIMC teacher games; the GPU trains the bootstrap network on them as they arrive
  - **reinforcement-learning run:** CPUs self-play; the GPU learns; spare cores run evaluation
  - **distillation run:** CPUs run full search games; the GPU distils them into the network and trains students
  - **arena run:** CPUs run tournaments; the GPU trains the next candidates from data already stored
- **Finishing together:**
  - The manifest's deadline drives everything. New chunks stop being issued once there isn't time to finish one.
  - Fixed task lists are ordered longest first (**LPT scheduling** [longest-task-first; proven to finish close to the best possible time]).
  - The trainer stops at the deadline with a final checkpoint.
  - The idle tail is minutes, not hours.

### 12.5 Saving, finishing and self-destructing

- **Every 10 minutes at most:** completed shards and checkpoints are uploaded to R2 as large content-addressed bundles, using `rclone copy --immutable` (never overwriting). The done-list, heartbeat and manifest are mutable state and are written as versioned keys instead (`done/<run>/<seq>-<utc>.json` with zero-padded sequence numbers; the launcher reads the highest; an R2 lifecycle rule expires old heartbeats). Large bundles keep R2 operation costs low and respect its "one write per second per key" limit.
- A **heartbeat** file [a periodic "I'm alive" signal] lets the laptop launcher spot a dead or stalled machine. It then destroys it and rents a replacement that resumes from the done-list.
- **At the deadline:**
  1. final upload, then `rclone check` [verifies every file arrived intact]
  2. upload the run manifest *last*, with the git commit, image digest, binary build, seeds, thread counts and all versions recorded
  3. write a `DONE` marker
  4. the machine destroys itself (`vastai destroy instance $CONTAINER_ID`, using its own restricted key)
- The launcher also destroys it on `DONE` or on a hard time limit, as a double safety. **Always destroy, never stop:** stopped machines still bill for storage.

### 12.6 The laptop's role

- **Development:** fast builds, rule tests, small benchmarks and code review.
- **Overnight serial jobs:** work that is small or can't be split across cores, such as analysis reports, model conversion and small checks. These run through the same runner and manifests.
  - The runner keeps Windows awake while it works (the `SetThreadExecutionState` system call).
  - The laptop must stay plugged in with the lid open: closing the lid still sleeps.
- **Launching and watching** vast.ai runs, and pulling results from R2.

### 12.7 Data portability rules (Windows ↔ Linux ↔ Cloudflare)

- **Content-addressed files** [named by a fingerprint of their bytes (SHA-256), so the same content always has the same name and is never silently replaced].
- **Explicit formats:**
  - **Game records:** `.jsonl.zst` (one game per line, versioned).
  - **Training shards:** `.npy` [NumPy's standard array file] with fixed little-endian fields; compressed (`.npy.zst`) for transfer, decompressed on arrival so Python can memory-map them.
  - **Weights:** safetensors. **Transformer:** single-file ONNX.
  - **Metrics:** JSONL, then Parquet.
- **Time and naming:** timestamps in UTC; lowercase file names with `/` separators.
- **Replays never depend on floating-point maths:** game records store every action. Network results may differ in the last decimal between CPU types, so a replay re-applies recorded actions instead of re-running networks.
- **Data protection:** human game data (phase E) is anonymised before it ever leaves Cloudflare. Rented machines only receive bucket-scoped R2 keys.

---

## 13. Build order with pass/fail checkpoints

**Project:** `C:\Users\gdeswardt1.cnslt\41Bots\`, a new Git repository. Nothing is reused from the old app.

```
41Bots/
  rust-toolchain.toml  Cargo.toml (workspace)  pyproject.toml  uv.lock  .python-version  .gitattributes
  crates/
    engine/      cards/bitboards, rules + switches, scoring, observation filter, records, own RNG
    reference/   naive, independent second rules implementation (no engine dependency; testing only)
    difftest/    engine-vs-reference differential harness (small count in cargo test; --release binary with a game count)
    solver/      max^n open-hand solver (pruning, transposition table, partition search)
    belief/      card tracker, constrained deal generator, particle filter
    nn/          input encoder (single source of truth), MLP inference (reference + SIMD), weights I/O
    search/      recursive rollout search + PIMC; anytime budgets; batching
    tablemgr/    table manager, seat protocol (versioned JSON), bot-backend interface, client SDK
    arena/       duplicate tournaments, SPRT, Elo, AIVAT, tier ladders, reports
    runner/      run-manifest executor: chunks, priorities, deadlines, checkpoints, heartbeat
    pybind/      PyO3 bindings (encoder, engine, data tools)
    wasm/        WASM builds for browser/phone and Cloudflare
  deploy/
    cloudflare/  Durable Object table host + bot Worker (T1), fan-out option
    server/      native bot server for CPU/GPU machines (T2/T3)
    docker/      multi-stage Dockerfile (v2/v3 binaries + uv env cu126 + rclone + vastai)
  python/
    train/       supervised, PPO, distillation, students, match-equity model, exports
    launcher/    vast.ai offer search, value-per-dollar ranking, rent/watch/destroy, R2 sync
  tools/         rng_golden.py (stdlib-only RNG golden-vector generator) and other small scripts
  (the reference implementation and the differential harness live in crates/reference and crates/difftest; tests live in each crate)
  .github/workflows/   CI: Windows + Linux + WASM builds/tests, Docker image to GHCR
```

| Checkpoint | Delivers | Gate |
|---|---|---|
| **M0** | Engine with own RNG, match and money scoring, **burn rule (H3; the no-ace version was withdrawn 2026-10-14), frozen deal convention, `rules_version`, incremental ace facts and burn locations in the event stream, event visibility tags**; `crates/reference`, `crates/difftest`; **~25 named plain-English transcripts, fuller property tests, branch-coverage difftest**; CI (Windows on PRs + nightly, manual sharded 10M job). | Every numbered rule tested; golden vectors identical on three platforms; zero disagreements over 10M games with minimum rare-branch coverage; WASM equals native; **owner has read and approved the transcripts**. |
| **M1** | Table manager (storage-backed, replayable), observation filter, protocol, records, random and rules bots (**rules bot from published 45-family heuristics**); arena with **14-way block, GSPRT, Plackett-Luce, canary**; **fixed-point money type and match-equity table v0**; **thin run-ops** (rent, run prebuilt chunk, upload, self-destroy) and R2 layout. | Duplicate matches replay exactly; leakage tests pass; first vast.ai run end to end near 100% CPU; **table v0 passes the reachable-score/flag property tests; σ_b measured**. |
| **M2** | Exact solver: **measurement spike first**; max^n with **stated backup rule and position-only order**, **score-state key at trick boundaries**, **equivalent cards with 41 exceptions**, Sturtevant pruning; **tie-fraction and flip tests**; **test-only forced-denial check**; **four-layer validation gate**. | Matches brute force; identical pruning on/off under the stated order; **published nodes, hit rates, memory, ms per world by trick and seat on two CPU types**. |
| **M3** | Card tracker, suit-length-matrix generator, PIMC teacher (**exact from trick 3, rollout-PIMC with the rules bot earlier**); **pre-registered PIMC-suitability measurement with world counting**; **full runner (manifests, priorities, heartbeat, resume) before the M4 run**. | PIMC beats the rules bot decisively on the 14-way block; properties reported per trick; resume test passes. |
| **M3b (new)** | Thin Cloudflare slice: Durable Object skeleton + PIMC bot as WASM in a Worker through the real protocol; **playable "measurement build"**; Workers deploy token; **live re-verification of Cloudflare limits**. | Worlds per CPU-second within the 8 s / 5 s budget, peak memory vs the isolate limit, native-vs-WASM identity reported (measurement, not a gate). |
| **M4** | Network: **active-feature encoder**, plain MLP + **residual MLP in the pool**, heads (policy, value, **autoregressive belief**, oracle, **separate public model**, **diagnostics-only aux**), **soft targets in shards**; **short bootstrap** (10^5–2×10^6 hands); **table refreshed from PIMC self-play**; inference on all targets. | Network close to the teacher; belief calibration good; Rust inference within tolerance on native and WASM; **decisions/s/core published and phase-B cost derived**. |
| **M5** | Recursive rollout search (**depth 1, policy rollouts early, solver-inside late on T2/T3**), safeguards, **counter-based CRN, async batching**, **mandatory experiments** (truncated vs full; independent vs single-seat; cut-off evaluator arms), **clinch-deal library with MP-Mix baselines**, world curves per tier, **T3 budget fixed in worlds**. | Beats the PIMC teacher on SPRT; clinch rates and experiment results published. |
| **M6** | Self-play RL: PPO + oracle critic + stabilisers, **precise potential reward with stored Φ and telescoping test**, **FCP-shaped league with cycling detector**, OADMCDou challenger, **from-scratch control arm**, **aux-head ablation**. | Beats the M5 champion on unshaped money; not worse against the frozen pool. |
| **M7** | Distillation loop with **playout-cap randomisation and a budget**, tier students, **integer-inference student candidate**, **transformer challenger enters the T2/T3 pool**; strength–cost curves. | Each generation beats the previous; curves for T0–T3 with memory-capped T1 world counts. |
| **M8** | Production Cloudflare host and T1 Worker; **T2 decided (Containers vs fan-out) by measured strength per dollar and latency**; server bot T2/T3; load tests; **human-data consent/retention handling**. | Meets the 8 s / 5 s / ≤3 s budgets under load; fairness rule enforced. |
| **M9** | Phase E: cross-play suite from the league pool; FCP variant; human model and piKL once the data threshold is met. | Better cross-play and human-partner money than M7 without losing against bots. |

---

## 14. Verification (how we will know it works, end to end)

- **Rules:**
  - rule-by-rule tests written directly from rules 1–41, including hand-made cases (turned Joker then a turned ace; dealer holding the ace of trumps after a turned Joker; hearts as trumps; a higher trump forcing out top trumps; robbing at the robber's own turn; burning the taken card; the game ending mid-hand at 41 and the next deal moving left; a jink)
  - **property tests** [millions of random games checking things that must always hold: 53 cards accounted for, correct scoring]
  - **differential testing** against the reference implementation [two independent implementations must agree on every game], and native against WASM
- **Determinism:**
  - RNG golden vectors on Windows, Linux and WASM
  - replaying a recorded game reproduces it exactly
  - re-running a chunk with the same seed produces byte-identical data
- **Information firewall:** the hidden cards are shuffled, and every bot decision must stay identical. Any change means a leak. The same test runs on the Cloudflare path.
- **Solver:** brute-force cross-checks, and an exact match with pruning on and off.
- **Beliefs:** calibration plots and true-world sampling rates.
- **Search:** hand-built positions with known right answers (for example: when to hold back the 5 of trumps, which card to burn after seeing trick-1 plays), plus A/B tournaments.
- **Objective audit:**
  - The money-chance type (fixed-point from M1: u32 per team in units of 2^-32, normalised by largest-remainder rounding, u64 accumulators, index-ordered reductions) is the only score the solver, search, training targets and tournament ranking can produce. This is checked when the code is built, plus a test that every evaluator's four chances add to exactly 2^32. Engine, solver and teacher data are byte-identical across machines, thread counts and WASM; network-dependent data is identical per (binary, CPU class), recorded in every record; GPU training resumes from checkpoints with logged divergence.
  - Any points- or tricks-based number is labelled diagnostics-only.
  - The money settled by the engine at match end must match the tournament's money totals exactly.
- **Money behaviour** (section 5.3) gets its own hand-built scenarios:
  - blocking a team that has one game in and stands on 36–40
  - a team that can grab the money in this very hand (any one trick wins it): every other team denies it any trick, even at the cost of top trumps or overtaking a partner
  - mid-hand overtaking: the second-place team takes a trick and becomes first to the money. On the very next card, every other team's blocking switches to the new leader.
  - giving a trick to a far-behind team rather than the leader
  - keeping the match alive when behind
  - going for a jink when only a jink saves the money
  - the second game of a match versus the first, where the same score is worth different amounts
  - **blocking as a joint outcome:** a library of "the leader could clinch this hand" deals (exact by the solver or oracle head, regenerated per generation), replayed with all three blocking teams searching independently; the leader's clinch rate and the blockers' own money change are reported per engine and tier (a block that costs a blocker more money than it saves is an error), with MP-Mix-style leader-targeting bots as reference behaviours only

  Each scenario's expected choice is checked against the match-equity model and confirmed in tournaments.
- **Statistics:** SPRT with error rates fixed in advance; confidence intervals reported with every result.
- **Run operations:**
  - CPU-use and GPU-use logs per run, with the target of CPU near 100% across the run and a short idle tail
  - a kill-the-machine test: destroy mid-run, resume on another machine, and get identical final results
  - a cost report per run
- **End to end:** a full tournament over the real network path (Durable Object table plus bot backends on separate connections), replayed from records, giving identical results. Then a mixed table with scripted "human" clients.

---

## 15. Risks and open questions

- **Rule details:** all known ambiguities are now resolved (section 0.1). Any new ones found while writing rule tests become engine switches and are raised with you.
- **Partner mismatch while searching** (SPARTA): handled by the safeguards in section 6.10 and by distillation, and measured in tournaments.
- **Learning vs engineered search** (GO-MCTS lost to Kermit in Skat): we keep both, and tournaments pick, per tier.
- **Device and serverless strength:** small budgets mean weaker bots there. This is unavoidable physics, not a design shortcut. The strength–cost curves make the trade-off explicit, and distillation keeps the gap as small as possible.
- **Older GPUs and PyTorch:** 2.14.1 is the last PyTorch line for Pascal/Volta, so we are pinned there.
  - If we later need PyTorch 2.15+ features, either restrict rentals to Turing or newer (`compute_cap>=750`), or switch the GPU environment to conda-forge builds (pixi), which still cover Pascal.
  - Our networks are small, so 2.14.1 is fully sufficient.
- **vast.ai unknowns:**
  - How it enforces each container's CPU share is undocumented. Hence explicit thread counts, plus the start-up benchmark to confirm real throughput.
  - CPU-only compute rentals do not exist (verified 13 Oct 2026: `num_gpus=0` returns storage offers only), so we rent cheap-GPU machines with many cores (filter on `cpu_cores_effective` and `cpu_ram`) and keep their GPU busy with training. Interruptible instances have their processes killed when outbid, so checkpoints every 10 minutes are mandatory.
- **Rust 1.99.1** is due on 15 Oct. Until then 1.99.0 is used, and the Python bindings are re-tested after upgrading.
- **R2 costs:** many small files drive up operation costs, which is why bundles are large.
- **Human data and privacy:** human games are collected only with player consent and anonymised. Phase E quality grows with the amount of data.
- **Kingmaking:** while a match is live, every team keeps a real (if small) chance of the money, so maximising its own chance gives sensible blocking and prolonging behaviour (5.3). This relies on the match-equity table being accurate for small chances (6.6). Ties are common, not rare (6.7), and are governed by the documented backup rule; below the search's noise floor the SPARTA-form deviation threshold keeps a team on the money-trained blueprint. The theorem behind reward shaping protects the set of equilibria, not which one self-play reaches: that selection is handled by the shared blueprint and distillation, and tested with the clinch-deal library (14). Validated in scenario tests and tournaments.
- **Honesty about novelty:** no published 41 AI exists, and an 8-player, four-partnership trick-taking game is unusual. Every component has a published track record; the *combination* is integration work, and every choice between proven options is settled by measured tournaments, not by assumption.
- **Prices and versions** were checked on 13 Oct 2026 and will drift. CI runs on the pinned versions, and upgrades go through the tournament gate.

---

## 16. What we deliberately do NOT do, and why

| Not used | Why |
|---|---|
| CFR / equilibrium solving for the whole game | It has not worked for trick-taking games. Moves depend on private cards, and the information sets are too large (Long et al. 2010; Rebstock et al. 2024). |
| Precomputed 41 endgame databases | 41's cost lies in the number of hidden-card worlds, not hand depth. Setrograde-style set reasoning is used inside the solver instead. |
| Hand-written strategy rules as the main engine | Bridge Baron lost to search-based programs. Rules bots exist only as test baselines. |
| Paranoid search | Max^n with Sturtevant's pruning beat it in Hearts and Spades |
| Plain PIMC as the final engine | It saturates (Furtak) and suffers strategy fusion. It stays as the teacher, a baseline, and a CPU-tier candidate. |
| Heavy cluster frameworks (Ray, Kubernetes) | One big machine per run doesn't need them. A small runner with chunk queues is simpler and keeps cores busier. |
| Compiling on the rented machine / `target-cpu=native` | That is paid machine time, and the binaries wouldn't be portable. Prebuilt v2/v3 builds plus runtime dispatch are used instead. |
| Best-Reply Search / BRS+ as the solver | They let only one opponent move at a time, which breaks mandatory follow-suit and the trick order (Nijssen & Winands). |
| ReBeL / Player of Games style public-belief search | Their guarantees are two-player zero-sum only; 41 is a four-team general-sum game with an astronomically large hidden pool. |
| Rounding money chances inside the solver to enable discrete-cut-off pruning | Any step fine enough to respect small-chance accuracy prunes nothing extra; a coarse step would erase the differences blocking and kingmaking depend on (rejected unanimously in the 2026-10-13 review). |

---

## 17. Glossary

| Term | Plain meaning |
|---|---|
| Monte Carlo | Try many random guesses and average the results |
| Double dummy / open-hand | Analysing a deal as if every hand were face up |
| PIMC | Guess the hidden cards many times, solve each guess face up, play the best average card |
| Strategy fusion | PIMC's flaw: it assumes it can play differently in situations it cannot actually tell apart |
| Recursive / imperfect-information Monte Carlo | Score a card by playing the rest of the hand with realistic players who see only their own cards |
| Rollout | Play the candidate card, then simulate the rest of the hand |
| Anytime search | Can stop at any moment and return the best card found so far |
| Inference / belief | Working out who probably holds which cards from what they played |
| World | One complete guess of where every hidden card is |
| Particle filter | A running set of weighted worlds, updated after every card |
| Importance sampling | A correction that lets us sample from a convenient guess but still get the right answer |
| max^n vs paranoid | Every team plays for itself, versus assuming everyone gangs up on you |
| Pruning | Safely skipping moves that cannot change the result |
| Transposition table | A memory of positions already solved |
| Partition search | Remembering families of equivalent positions, not single ones |
| Neural network / MLP / transformer | A program that learns from examples; MLP is the simplest design; a transformer reads sequences |
| Policy / value | "Which card would I play?" / "How likely is each team to win from here?" |
| Blueprint | The default strategy every bot shares and can predict |
| Supervised learning / behaviour cloning | Learning to copy examples, including recorded human decisions |
| Reinforcement learning (PPO, Deep Monte-Carlo) | Learning by trial and error from game results |
| Oracle judge | A training-only helper that sees all cards, to grade decisions faster |
| League | Training against many past versions and exploiter bots |
| Distillation / Expert Iteration | Teaching a fast network to copy a slow, strong search, or a small network to copy a big one |
| Match-equity model | The chance each team wins the match, and so the money, from any score |
| Expected money | 40 × (chance of winning the match) − 10 per team at €5 stakes; maximising it equals maximising the match-win chance |
| Kingmaking | A team unlikely to win influencing which other team wins |
| Log-odds | A probability scale that stretches out very small and very large chances so they're learned precisely |
| Public enemy number one | Whichever team is currently closest to the money, re-assessed after every trick |
| Potential-based reward shaping | Rewarding the change in estimated money chance after every trick; proven not to change the best strategy, only to speed up learning |
| Duplicate | Replaying the same deals with seats swapped so luck cancels out |
| SPRT | Play only until it is statistically clear which version is better |
| AIVAT | Subtracting estimated luck from results to need fewer games |
| Elo | A chess-style strength rating |
| Cross-play | Testing a bot with partners it was never trained with |
| Exploitability probe | A bot trained only to beat ours, to find weaknesses |
| CFR | An equilibrium algorithm that works in poker but not in trick-taking games |
| Bitboard | A set of cards stored as one 64-bit number |
| SIMD / AVX2 / AVX-512 | Processor instructions that do many calculations at once; newer CPUs have wider versions |
| WASM / WebAssembly | A portable compiled format that runs fast in browsers, phones and Cloudflare |
| Durable Object / Worker / serverless | Cloudflare's per-table mini-server / on-demand code / pay-per-use computing |
| ONNX / safetensors | Standard file formats for trained networks / for network weights |
| CUDA / compute capability | NVIDIA's GPU software layer / the GPU's generation number; they must match |
| Mixed precision | Using smaller number formats for speed where the GPU supports them |
| Lockfile / toolchain | Exact list of dependency versions / compiler and build tools |
| CI | Automatic build and test on every change |
| Docker image / registry (GHCR) | A packaged copy of the whole software environment / where images are stored online |
| R2 / rclone | Cloudflare's file storage (no download fees) / the tool that syncs files with it |
| On-demand / interruptible | Fixed-price rental / cheaper rental that can be stopped when someone outbids you |
| Run manifest / chunk / done-list | A run's instructions / a small unit of work / the record of finished units |
| Heartbeat | A periodic "I'm alive" signal from the rented machine |
| LPT scheduling | Doing the longest tasks first so everything finishes together |
| Oversubscription | More busy threads than cores, which slows everything down |
| Content hash (SHA-256) | A fingerprint of a file's bytes, used as its name |
| GPU | A graphics card, used for fast neural-network maths |

---

## 18. Sources

**Linked papers:**
- [GIB, JAIR](https://jair.org/index.php/jair/article/view/10279) (same paper as [arXiv 1106.0669](https://arxiv.org/abs/1106.0669))
- [αμ](https://arxiv.org/abs/1911.07960)
- [Furtak thesis](https://doi.org/10.7939/R3H98ZQ52)
- [Setrograde, IJCAI 2025](https://www.ijcai.org/proceedings/2025/995) and [arXiv 2411.09089](https://arxiv.org/abs/2411.09089)
- [Sturtevant 2003](https://mlanthology.org/ijcai/2003/sturtevant2003ijcai-last)

**Additional research:**
- [Long et al. 2010](https://ojs.aaai.org/index.php/AAAI/article/view/7562)
- [Policy Based Inference](https://arxiv.org/abs/1905.10911)
- [Card-location inference](https://arxiv.org/abs/1903.09604)
- [History filtering](https://arxiv.org/abs/2311.14651)
- [GO-MCTS](https://arxiv.org/abs/2404.13150)
- [SPARTA](https://arxiv.org/abs/1912.02318)
- [Perfect Information Monte Carlo with Postponing Reasoning](https://arxiv.org/abs/2408.02380)
- [WBridge5 seed boosting](https://hal.archives-ouvertes.fr/hal-01665867)
- [BBO GIBBO interview](https://news.bridgebase.com/?p=87069)
- [Ben](https://github.com/larrycai/ben)
- [NooK coverage](https://www.fortune.com/2022/03/30/artificial-intelligence-beats-humans-card-game-bridge)
- [DouZero](https://arxiv.org/abs/2106.06135)
- [PerfectDou](https://arxiv.org/abs/2203.16406)
- [Suphx](https://arxiv.org/abs/2003.13590)
- [IMPALA](https://arxiv.org/abs/1802.01561)
- [AIVAT](https://ojs.aaai.org/index.php/AAAI/article/view/11481)
- [SPRT](https://chessprogramming.org/Sequential_Probability_Ratio_Test)
- [Computer bridge history](https://en.wikipedia.org/wiki/Computer_bridge)

**Added by the review of 13 October 2026** (full list with notes in `docs/REVIEW-2026-10-13.md`):
- Furtak & Buro, Recursive Monte Carlo search for Skat, CIG 2013: https://skatgame.net/mburo/ps/recmc13.pdf
- Bouzy, Rimbaud & Ventos, Recursive Monte Carlo search for bridge card play (5-card results), IEEE CoG 2020: https://ieee-cog.org/2020/papers/paper_82.pdf
- Sturtevant, PhD thesis 2003: https://webdocs.cs.ualberta.ca/~nathanst/papers/multiplayergamesthesis.pdf ; Sturtevant, UCT in multi-player games, CG 2008: https://webdocs.cs.ualberta.ca/~nathanst/papers/multi-player_UCT.pdf
- Sturtevant, Zinkevich & Bowling, Prob-max^n, AAAI 2006: https://webdocs.cs.ualberta.ca/~nathanst/papers/probmaxn.pdf ; Sturtevant & Bowling, soft-max^n, AAMAS 2006: https://webdocs.cs.ualberta.ca/~nathanst/papers/softmaxn.pdf
- Korf, Multi-player alpha-beta pruning, 1991 ; Sturtevant & Korf, On pruning techniques for multi-player games, AAAI 2000: https://cdn.aaai.org/AAAI/2000/AAAI00-031.pdf
- Zuckerman, Felner & Kraus, MP-Mix, IJCAI 2009: https://www.ijcai.org/Proceedings/09/Papers/113.pdf ; Nijssen & Winands, multi-player search overview: https://dke.maastrichtuniversity.nl/m.winands/documents/Multi_Overview.pdf
- Sievers & Helmert, Doppelkopf ensemble-UCT, KI 2015: https://ai.dmi.unibas.ch/papers/sievers-helmert-ki2015.pdf ; Cowling, Powley & Whitehouse, ISMCTS 2012
- Backgammon match-equity tables: https://bkgm.com/articles/met.html ; Rockwell-Kazaross: https://bkgm.com/articles/Kazaross/RockwellKazarossMET/index.html ; Zare on wrong tables: https://bkgm.com/articles/Zare/WrongMatchEquityTable/index.html
- Lu, Schwartz & Givigi, potential-based shaping in n-player general-sum games, JAIR 2011: https://arxiv.org/pdf/1401.3907 ; Devlin & Kudenko, AAMAS 2011: https://eprints.whiterose.ac.uk/id/eprint/75111/1/aamas11.pdf and AAMAS 2012 (dynamic potentials): https://eprints.whiterose.ac.uk/75121/2/p433_devlin.pdf
- Mortal (Mahjong) reward convention: https://github.com/Equim-chan/Mortal ; Guo et al., temperature scaling 2017: https://arxiv.org/abs/1706.04599 ; Nixon et al., measuring calibration 2019: https://arxiv.org/abs/1904.01685
- Hu, Lerer, Brown & Foerster, Learned Belief Search 2021: https://arxiv.org/pdf/2106.09086 ; Off-Belief Learning 2021: https://arxiv.org/pdf/2103.04000 ; Off-Team Learning, NeurIPS 2022 ; Sokota et al., Ataraxos 2025/26: https://arxiv.org/html/2511.07312v2
- Gilks & Berzuini, resample-move particle filters, 2001 ; Strouse et al., Fictitious Co-Play 2021: https://ar5iv.labs.arxiv.org/html/2110.08176
- DouZero+ 2022: https://ar5iv.labs.arxiv.org/html/2204.02558 ; DouRN 2024: https://arxiv.org/html/2403.14102 ; DanZero+ 2023: https://arxiv.org/html/2312.02561v1 ; OADMCDou, IJCAI 2024: https://www.ijcai.org/proceedings/2024/0660.pdf ; AlphaHoldem (Trinal-Clip PPO, K-best self-play), AAAI 2022: https://ojs.aaai.org/index.php/AAAI/article/view/20394
- Anthony, Tian & Barber, Expert Iteration 2017: https://ar5iv.labs.arxiv.org/html/1705.08439 ; KataGo training techniques: https://ar5iv.labs.arxiv.org/html/1902.10565 ; Jacob et al., piKL 2022: https://ar5iv.labs.arxiv.org/html/2112.07544
- Fishtest mathematics (generalised SPRT): https://github.com/official-stockfish/fishtest/wiki/Fishtest-mathematics ; Van den Bergh, normalized Elo: https://www.cantate.be/Fishtest/normalized_elo_practical.pdf
- White & Bowling, MIVAT 2009: https://bowlingmh.github.io/papers/09ijcai-mivat.pdf ; Weng & Lin, Bayesian rating for multi-team games, JMLR 2011: https://jmlr.org/papers/v12/weng11a.html ; OpenSkill, JOSS 2024: https://joss.theoj.org/papers/10.21105/joss.05901
- Hu et al., Other-Play (cross-play protocol) 2020: https://arxiv.org/abs/2003.02979 ; Timbers et al., approximate exploitability 2022: https://arxiv.org/abs/2004.09677
- Stockfish NNUE documentation: https://raw.githubusercontent.com/official-stockfish/nnue-pytorch/master/docs/nnue.md ; EnvPool (asynchronous batching): https://ar5iv.labs.arxiv.org/html/2206.10558
- Nova Scotia 45s AI (same-family PIMC + exact-solver results, validation gates, canary): https://github.com/jitousig/fortyfives-ai ; Irish 25 app (41 variant): https://irish25cardgame.com/scoring/ ; Pagat Twenty-Five: https://www.pagat.com/spoil5/25.html
- Cloudflare Containers pricing and platform details: https://developers.cloudflare.com/containers/pricing/ , https://developers.cloudflare.com/containers/platform-details/ ; vast.ai billing: https://docs.vast.ai/guides/reference/billing.md ; vast.ai Docker environment (CPU share, CONTAINER_API_KEY): https://docs.vast.ai/guides/instances/docker-environment.md

**Tech stack:**
- [Rust releases](https://blog.rust-lang.org/releases/)
- [wasm32 target](https://doc.rust-lang.org/stable/rustc/platform-support/wasm32-unknown-unknown.html)
- [rustwasm org sunset](https://blog.rust-lang.org/inside-rust/2025/07/21/sunsetting-the-rustwasm-github-org)
- [workers-rs](https://github.com/cloudflare/workers-rs)
- [tract](https://github.com/sonos/tract)
- [ort](https://ort.pyke.io/)
- [matrixmultiply](https://github.com/bluss/matrixmultiply)
- [multiversion](https://docs.rs/multiversion)
- [rand reproducibility](https://rust-random.github.io/book/crate-reprod.html)
- [bincode advisory](https://rustsec.org/advisories/RUSTSEC-2025-0141)
- [PyO3 changelog](https://raw.githubusercontent.com/PyO3/pyo3/main/CHANGELOG.md)
- [maturin](https://www.maturin.rs/changelog.html)
- [PEP 784 (zstd in stdlib)](https://peps.python.org/pep-0784/)
- [Python versions](https://devguide.python.org/versions/)
- [PyTorch releases](https://github.com/pytorch/pytorch/releases)
- [PyTorch 2.15 drops Pascal/Volta](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)
- [CUDA 12.6 wheel index](https://download.pytorch.org/whl/cu126/torch/)
- [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [uv + PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [torch.onnx](https://docs.pytorch.org/docs/2.14/onnx.html)

**Infrastructure:**
- [vast.ai search offers](https://docs.vast.ai/cli/reference/search-offers.md)
- [vast.ai create instance](https://docs.vast.ai/cli/reference/create-instance.md)
- [vast.ai Docker environment](https://docs.vast.ai/guides/instances/docker-environment.md)
- [vast.ai storage](https://docs.vast.ai/guides/instances/storage/types.md)
- [vast.ai instance types](https://docs.vast.ai/guides/instances/choosing/instance-types.md)
- [R2 pricing](https://developers.cloudflare.com/r2/pricing/)
- [R2 limits](https://developers.cloudflare.com/r2/platform/limits/)
- [rclone R2](https://developers.cloudflare.com/r2/examples/rclone/)
- [Workers limits](https://developers.cloudflare.com/workers/platform/limits/)
- [Workers pricing](https://developers.cloudflare.com/workers/platform/pricing/)
- [Durable Objects WebSockets](https://developers.cloudflare.com/durable-objects/best-practices/websockets/)
- [Containers](https://developers.cloudflare.com/changelog/post/2025-10-01-new-container-instance-types/)
- [GitHub Packages billing](https://docs.github.com/en/billing/concepts/product-billing/github-packages)
- [x86-64 levels](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels)
- [LPT scheduling](https://en.wikipedia.org/wiki/Longest-processing-time-first_scheduling)
- [Windows SetThreadExecutionState](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate)
