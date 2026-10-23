# Research: the outside evidence this project builds on

This is the single collected record of the published research, other programs and game sources
the project relies on: what each source did, under what conditions, and what it measured. It is
meant to let a new reader (a person or an AI) learn the field as it bears on 41 and its family
(the "five fingers" games: 25, 31, 41, 45, 110 and relatives) without re-reading the papers. It
states what the sources report; it gives no advice. Where the project adopted, rejected or deferred
a method, that fact is recorded with its date and a pointer, so the reader can see which evidence
each design choice rests on; the reasoning and the amendments themselves are in `docs/PLAN.md` and
`docs/REVIEW-2026-10-13.md`, and our own measurements are in `docs/EXPERIMENTS.md`.

**Provenance of the facts.** Most entries were gathered in the plan's research (2026-10-13) and its
review the same day: eight web-research sweeps over about 150 primary sources, then a loose-ends
sweep in which the key papers (Sturtevant's three PDFs, Furtak's thesis, Furtak & Buro 2013, Bouzy
et al. 2020, Sturtevant 2008) were downloaded and their numbers read from the text. Later entries
come from the design documents of 2026-10-16 to 2026-10-22 (`docs/m3-hybrid/*.md`, `docs/STUDENT.md`)
and from checks made for this record on 2026-10-22 (game-family pages, Smith & Winkler 2006). Each
entry says **verified** when its numbers were read in the primary source, **secondary** when they come
from a summary, abstract, forum or encyclopaedia, and **design citation** when the source was cited as
the basis of a method without its numbers being re-read. Source links are in §14.

---

## 1. The game family ("five fingers" / Maw family)

**History and naming** (secondary: Wikipedia "Twenty-five", Parlett).
- The family descends from the Scottish game Maw (recorded from at least 1548), fashionable at the
  court of James I. Charles Cotton's *The Compleat Gamester* (1674) describes its ancestor "Five
  Cards", in which the five of trumps ("five fingers") is the highest card.
- Members: Spoil Five, Twenty-five (described in detail by H.E. Heather, 1876), Forty-five, 110
  (and 55/220 bidding forms). Twenty-five is called the Irish national card game; Canadian
  Forty-fives descends from it. Heather's 1876 Twenty-five had no jink and no best-trump bonus.

**Common core across sources** (secondary: Pagat "Twenty-Five", Wikipedia "Forty-fives").
- Five cards each, dealt 2+3 or 3+2; the next card is turned for trumps; deal passes left.
- Trump order: 5, J, A♥, A (when hearts are not trumps), K, Q, then numerals; red numerals rank
  high-to-low (10 best), black numerals low-to-high (2 best): "highest in red, lowest in black".
  Plain suits: K, Q, J, then numerals; the ace of diamonds is lowest of its suit; black aces sit
  between the J and the 2. The A♥ is always a trump.
- Reneging: the top trumps (5, J, A♥) may be withheld when a lower trump is led; some tables allow
  only the 5; some forbid reneging.
- Robbing: the holder of the ace of trumps may rob the turned card (discard one, take it) before the
  first card is played; the dealer may rob a turned ace. Traditional sources make robbing optional.
- 25: 5 points a trick, first to 25; the game can end mid-hand. 45: 5 a trick plus a 5-point bonus
  for the trick won by the best trump (10 for that trick), first to 45. Auction 45 (Canada): bids
  15-30, 30-for-60, game 120. 110/55/220 (Ireland): a kitty, bids 10-60, no robbing, the bidder names
  trumps. Jink: in Wikipedia's account the player who takes all five tricks wins the game "whether
  played for twenty-five or forty-five"; Pagat reports only a double stake for five tricks, and only
  with no previous score.
- Players: 2-10 individually, or pairs with partners opposite (Pagat: 4, 6, 8 or 10 in pairs).
- Joker: the traditional 45 uses one; Irish 31/41 players place it third, between the J of trumps and
  the A♥ (boards.ie 2010-2016); in 110 it is usually just below the J; the Nova Scotia variant omits
  it; one secondary site places it below the A♥.

**31 and 41** (secondary: boards.ie 2010-2016; the Irish 25 app). 31 and 41 are Cork/Kerry/Limerick
partnership offshoots of 45 in which the best-trump trick scores 11 instead of 10. No source found
online documents "6 the first time, 5 later" (the project's RULES rule 28); every source gives a flat
11. The Irish 25 app (Tricksmith Games, 2024-) lists "41: first to 41 points; +6 bonus for strongest
trump; includes Joker" and "31: first to 31; +6 bonus", with robbing optional; it publishes no 41
rulebook. A 2016 forum ruling on a 41 dispute gave the game to the pair that reached 41 by tricks
mid-hand ahead of an unconfirmed best-trump claim. The full point-by-point comparison with
`docs/RULES.md` (11 points, confidence per point) is `docs/REVIEW-2026-10-13.md`, Part 4; the owner's
answers are in RULES section H.

**No published AI for 41 exists** (review sweep, 2026-10-13). The closest same-family program is
§1.1.

### 1.1 Nova Scotia 45s AI (`jitousig/fortyfives-ai`, MIT licence, 2026) — verified (repository read)
- Game: Nova Scotia 45s, 4 players in two partnerships, auction, kitty, five tricks, pegging to 125.
  Two sides only, so its solver is two-sided alpha-beta minimax, not max^n.
- Exact open-hand solver: transposition table keyed by hands, trick, leader, tricks per side and the
  best trump so far; equivalent cards merged within pools, never across the bonus-relevant trump
  boundary, A♥ never merged. Fast port ≈2.7 ms per fresh 5-card deal (25× its first version).
- Four-layer solver validation gate: engine parity on random play-outs; search equals brute force on
  1-, 2-, 3-card endgames; move reduction on/off gives identical root values; exhaustive best
  response from real engine states.
- Evaluation discipline: rule bot against itself must score exactly +0.0000 (canary, rerun after
  every engine or evaluation change); wins confirmed only at n ≥ 2,000 hands on ≥ 2 seeds with the
  95% interval excluding 0; no "near-optimal" claim without an oracle upper bound.
- Results (points per hand against the rule-based table, n = 2,000, two seeds): PIMC with 20 worlds
  and rule-based playouts +1.97 / +1.76; PIMC with an exact per-world solve +3.19 / +2.81 (+1.22 /
  +1.05 over the playout version); a cheating best-response oracle +8.62 / +8.25; a paranoid
  double-dummy oracle +4.26 / +4.40, about half the best-response ceiling. Hard sampler constraints
  from public actions (discard counts, then auction-consistency) added +0.37 and +0.25 pooled. PIMC
  averages worlds in final-payoff space "because E[delta] ≠ f(E[raw]) at make-or-set decisions".
- Not documented: any outcome of its DQN/NFSP/PPO scripts.
- Project use: the canary, the four-layer gate, paired evaluation and the oracle-headroom idea were
  adopted (PLAN amendment A2, A9).

---

## 2. How strong card-play programs are built: the record

### 2.1 Bridge
- **GIB** (Ginsberg, JAIR 2001) — design citation, numbers from the plan's reading. For each decision
  it deals 50-500 hidden-card layouts consistent with what is known, solves each double dummy [all
  hands face up] exactly, and plays the card with the best total (perfect-information Monte Carlo,
  PIMC). Deals are weighted by how likely the opponents' actual plays would be (Bayes). Its solver
  uses partition search [storing families of equivalent positions], which cut the branching factor
  from b to b^0.76; a full deal took about 18,000 search steps (about 1 s in 2000). It solved 100
  Bridge Master problems against Bridge Baron's 33 and dominated the 1998 and 2000 championships;
  by 2001 every competitive program had switched to its method. Stated weakness: it assumes it will
  know the hidden cards later, so it undervalues safe plays and never plays to gain information.
- **Championship winners**: 1997 Bridge Baron (hand-written techniques plus planning); 1998-2000 GIB;
  2001-2019 Jack, WBridge5, Shark, Micro Bridge (PIMC family; WBridge5's 2016-2018 titles followed
  "seed boosting", Ventos et al. 2017); the official World Computer-Bridge Championship ended in
  2019 (unofficial 2022 WBridge5, 2023 Q-Plus, whose method is unpublished). Championships use
  round robin then knockout, duplicate boards, and since 2001 one program per seat over a table
  manager protocol.
- **NooK** (NukkAI, 2022): beat 8 world champions at declarer play (67 of 80 sets) with rules plus
  search; details unpublished.
- **Ben** and **GIBBO** (BBO, live December 2025): neural networks alongside double dummy; GIBBO
  reported +0.9 IMP per board (basic) and +0.4 (advanced) over GIB 43; methods not published in
  detail. Bidding is the weak spot of all bridge programs; 41 has no bidding.

### 2.2 Skat
- **Kermit** (Buro, Long, Furtak & Sturtevant, IJCAI 2009): PIMC with an exact solver and a learned
  inference; its inference was worth 996 against 779 points per 36 games (as reported in the review).
- **Card-location networks** (Solinas, Rebstock & Buro, AAAI 2019) — verified via the review: a
  network trained on 20 million human games predicts where each card is; +3 to +5 tournament points
  per game over the old inference in suit and null games (grand games not significant).
- **Policy-based inference** (Rebstock, Solinas, Buro & Sturtevant 2019): each world weighted by the
  product of the policy's probabilities of every observed play; beat the card-location network
  (+2.32 / +0.64 / +1.57 points per game in the three game types, as summarised in the review).
- **Edelkamp** — verified (two papers), secondary (others): Knowledge-Based Paranoia Search (2021,
  Skat): the declarer takes the worst case over all consistent worlds with both defenders as a
  coalition, triggered after the third trick; on 50,000 replayed human games declarer wins 42,228
  with it, 41,765 without, 41,283 by the humans. "Outer-learning" (arXiv 2512.15435, Dec 2025): 30M
  self-play games added to a human database; replay prediction accuracy 84.73% → 84.78% (humans
  76.7%). A general trick-taking framework (KI 2024; Belote, Tarot, Doppelkopf, Spades, Hearts,
  Euchre, Schafkopf) is paywalled; its multi-side method could not be read. Edelkamp's handling of
  more than two sides is paranoid worst case in a 1-vs-2 game; it is not evidence about max^n with
  four sides.
- **GO-MCTS** (Rebstock et al. 2024): a transformer plus search; best Hearts program, but lost to
  Kermit in Skat by 9.84 points per game at about 42 s per move.

### 2.3 Other games with lessons for 41
- **Doppelkopf** (Sievers & Helmert, KI 2015): an ensemble of per-world UCT trees beat a single
  shared tree (4.52 vs −1.25); splits of 5/2000, 10/1000 and 20/500 worlds/iterations were within
  error bars.
- **Hearts and Spades** (Sturtevant): §4.
- **Jass** (Niklaus thesis 2019), **Wizard** (DQN, 2022), **Big-2** (PPO study, 2026): cited in the
  review's evidence list; their numbers were not used.
- **DouDizhu** (DouZero 2021, PerfectDou 2022 and successors), **Mahjong** (Suphx 2020, Mortal),
  **poker** (Pluribus 2019, AlphaHoldem 2022), **Hanabi** (SPARTA 2020, Learned Belief Search 2021):
  §6-§8.
- **Pluribus** (6-player poker, 2019): shows strong bots for more than two sides; its equilibrium
  method (CFR) has not worked for trick-taking games (Long et al. 2010; Rebstock et al. 2024).

---

## 3. Perfect-information Monte Carlo (PIMC) and its limits

- **Why PIMC works in trick-taking games** (Long, Sturtevant, Buro & Furtak, AAAI 2010) — verified.
  Three measurable game properties: leaf correlation (how little outcomes change between
  neighbouring endings), bias, and disambiguation (how fast play reveals hidden information). Skat
  and Hearts measured leaf correlation 0.8-1.0 and disambiguation about 0.6 (10,000 games per type,
  1,000 near-leaf samples). 15% of Skat games were unresolved with three tricks left, where PIMC lost
  0.42 points per deal. PIMC's loss is not monotone in disambiguation (worst at mid-range). The paper
  gives a procedure to measure these properties for a new game. Project status: the M3 measurement
  for 41 was specified (M3-SPEC §6) and has not been run.
- **Strategy fusion** [PIMC assumes it can play differently in worlds it cannot tell apart] and
  **non-locality** are its two named errors.
- **αμ** (Cazenave & Ventos 2019): makes "my side" play the same card in every world it cannot
  distinguish, assuming the opponents see everything. On close 3NT deals: 40 worlds, 62.4% (PIMC) →
  63.2% (αμ, 3 moves ahead); 20 worlds, 60.2% → 62.0%. Cost rises steeply with look-ahead. Confirms
  GIB and WBridge5 use PIMC. "Trick costs for αμ" (Bounan & Edelkamp, ISAIM 2024) could not be read.
- **Extended PIMC / PIMC with postponing reasoning** (Arjonilla, Cazenave & Ventos 2024): gains
  reported on Dark Chess, not card games.
- **Saturation (world counts)**: plain PIMC in Skat stopped improving beyond 160 worlds (Furtak &
  Buro 2013, Fig. 3: K160 vs K160 55.19, K3200 vs K3200 54.57; 18,962 games). In bridge (Bouzy et al.
  2020) solver-based PIMC plateaued at 100 worlds with 13 cards and 20 worlds with 5 cards; random-
  rollout PIMC at about 3,000 and 600, and 2.7 tricks weaker. SPARTA used a 100-rollout floor, then
  dropped options more than two standard deviations behind (about 10× fewer rollouts).
- **41's world counts** (project measurement, `docs/EXPERIMENTS.md` A5): about 10^36 consistent
  worlds at trick 1, 10^14-10^16 at trick 4, 10^8 at trick 5, against Skat's 4.3 × 10^7 at the start of
  play; 12-13 cards are never revealed.

---

## 4. Searching games with more than two sides

- **max^n** [each side maximises its own entry of a value vector] and **paranoid** [assume all other
  sides are a coalition against you]. Korf 1991: multi-player alpha-beta gives no average-case pruning
  gain; Sturtevant & Korf 2000: pruning techniques for multi-player games.
- **Sturtevant, IJCAI 2003** (last-branch and speculative pruning) — verified. Requires the values to
  sum to a constant (true for 41's money chances). Table 12 (3-player perfect-information variants,
  50 hands, 500k nodes a move): speculative max^n vs paranoid, Hearts 8.01 vs 8.99 (lower better),
  Spades 5.72 vs 5.61 (higher better), although paranoid searched deeper (Table 11, Spades partial
  tree 15.4 vs 22.4 ply).
- **Sturtevant's thesis (2003)** — verified. Perfect-information Hearts: 3-player 8.22 vs 8.82,
  4-player 6.33 vs 7.04; Spades 5.65 vs 5.68 ("negligible", paranoid ~10 ply deeper). Monte Carlo
  imperfect-information Hearts (Table 7.10): 3 players 6.91 vs 10.32 (40 models) and 6.43 vs 10.92
  (20 models); 4 players 5.34 vs 8.81. Imperfect-information Spades 5.81 vs 5.52 (max^n better). §8.3:
  "paranoid has a large search depth advantage over maxn, but no performance advantage" in card
  games. Theorem 4.1: with a mis-modelled tie rule the error of a max^n value is unbounded; "the tie-
  breaking rule must be implicitly defined by the way we order successors". Same move order needed in
  every part of the tree.
- **Prob-max^n** (Sturtevant, Zinkevich & Bowling, AAAI 2006): an expected-value backup over a model
  of the other players' choices (not a tie-break). **Soft-max^n** (Sturtevant & Bowling 2006): in
  exhaustively searched 3-player Spades a wrong opponent model dropped a 74.7% win rate to 44.0%.
- **Game-level objective** (Sturtevant 2004, Hearts to 100, Spades to 200): a game-level objective beat
  a hand-points objective (54.5% game wins despite a worse average score). Transposition gains: 157×
  in Spades, 26× in Hearts.
- **UCT in multi-player games** (Sturtevant 2008) — verified; all card-game tests perfect-information.
  4-player Hearts, 3,244 hands where shooting the moon was possible: the shooter succeeded 7.70%
  against UCT (50,000 playouts) and 42.45% against simple max^n. UCT beat a learned Hearts player
  46.12 vs 67.30. In 3-player Spades UCT was on par with prob-max^n (52.8% wins). Chinese Checkers: UCT
  beat paranoid and max^n 92-96% at 250k expansions (only 16.7% vs paranoid at 1,600). No UCT-vs-
  paranoid card-game result exists.
- **MP-Mix** (Zuckerman, Felner & Kraus 2009): switches between "offensive against the leader",
  paranoid and max^n by game state. **Best-Reply Search / BRS+** (Schadd & Winands 2011; Nijssen &
  Winands): only one opponent moves at a time. **Information-set MCTS** (Cowling, Powley & Whitehouse
  2012): no large difference from determinised search in Dou Di Zhu.
- **Project status**: max^n with the documented tie rule and shallow pruning is the solver (PLAN
  §6.7). Measured in 41: shallow pruning fires at about 1 node in 10,000 and 82-85% of decision nodes
  have tied best cards (`docs/EXPERIMENTS.md` A3). Paranoid, BRS and ReBeL-type search are recorded as
  not used (PLAN §16).

---

## 5. Exact open-hand solvers

- **Partition search** (GIB): b → b^0.76 in bridge; **Furtak's thesis (2013)** — verified: Skat solver
  speed-ups of 45× (suit), 58× (grand), 22× (null) from symmetries, move ordering and tables; Kermit
  solves ≈68 worlds a second exactly (30 cards, one core), 237 with forward pruning. Kupferschmid &
  Helmert (2006) Skat solver uses a two-sided zero-window [yes/no] search. Published fast solvers rely
  on two-sided alpha-beta, which a four-way value vector cannot use; no publication applies partition
  search to max^n or vector values.
- **Setrograde** (Stone, Sturtevant & Schaeffer, IJCAI 2025): solves bridge endgames backwards over
  sets of positions sharing a value; 1,000× fewer operations, 10,000× fewer stored entries; every
  7-trick bridge ending solved.
- **Equivalent cards**: two cards in one hand are interchangeable once every card ranked between them
  is gone (Furtak). The 45s project merges within pools but never across the bonus-relevant trump.
- **Transposition key**: Skat solvers drop the running score because their score is additive;
  Edelkamp keys only after a completed trick.
- **Project status and measurements**: `docs/EXPERIMENTS.md` A3-A4 (41: trick 3 9 ms, trick 2 3 s,
  trick 1 seconds to hours per world).

---

## 6. Beliefs: who holds which card

- **Constraint-respecting deal generation** (GIB; History Filtering): build worlds on a suit-length
  matrix with a feasibility check; card-level rejection only as a test oracle.
- **Learned card-location beliefs** (Solinas et al. 2019) and **policy-based weighting** (Rebstock et
  al. 2019): §2.2. The weight of a world is the product of the policy's probabilities of everyone's
  observed actions divided by the proposal probability (importance sampling).
- **History Filtering** (Solinas et al., NeurIPS 2023): a Metropolis-Hastings swap kernel that
  rejuvenates a set of worlds; requires full-support policies.
- **Resample-move particle filters** (Gilks & Berzuini 2001; POMCP, Silver & Veness 2010): weights
  updated per observed action, resampling when the effective sample size falls below half.
- **Learned Belief Search** (Hu, Lerer, Brown & Foerster 2021) — verified. 2-player Hanabi: rollouts
  truncated at 16 steps and bootstrapped by a value head scored 19.27 / 24.04 / 24.48 / 24.62 against
  full rollouts' 18.88 / 23.95 / 24.42 / 24.59 for weak to best blueprints; belief-prediction failures
  0.0083% vs 0.1%. Two players both searching: 24.34 vs single-agent 24.48 ("noticeably worse",
  beliefs become inaccurate). Against SPARTA: 24.48 in 47 s vs 24.52 in 215 s (5 cards).
- **Off-Team Belief Learning** (NeurIPS 2022): under partner shift, plain belief sampling produced a
  valid sample on about 83% of steps; off-team learning stayed above 99.5% and cut the self-play/
  cross-play gap below 1%. **Off-Belief Learning** (2021) is restricted to common-payoff games.
- **Ataraxos** (Sokota et al. 2025/26): transformer-decoder belief; the dou dizhu belief network has
  3.3M parameters, trained 8 hours on 4 H100s.
- **Autoregressive beliefs** (LBS, Off-Team, Ataraxos): assign hidden cards one at a time, masking full
  or forbidden locations; the proposal probability is the product of the conditionals.
- **Project status**: hard facts and exactly uniform sampling are built (`docs/EXPERIMENTS.md` A5);
  learned beliefs and policy weighting are planned (PLAN §6.4, amendment A5) and not built.

---

## 7. Search with realistic players inside the worlds (recursive Monte Carlo)

- **Furtak & Buro, CIG 2013** — verified. Recursive Kermit (RK: 160 level-1 worlds, 10 level-2) as
  soloist +6.84 points per game over saturated PIMC Kermit-160 (about 80 points per list), at about
  8× the cost; RK1600,10 +8.97 ("10× more hardware gives roughly 2 more points", while PIMC did not
  benefit). As defender −1.35 to −4.42 (lower better). Against information-set UCT ("Bernie", 1,600
  playouts): soloist 74.37 ± 1.58 vs 53.99 ± 1.97, about 35× faster. On synthetic trees recursion
  hurts when disambiguation < 0.3 and helps above; the largest drop in exploitability comes from the
  first recursion level. A post-mortem showed recursive Kermit makes more early perfect-information
  errors yet wins; the paper suggests recursion only late in the game.
- **Bouzy, Rimbaud & Ventos, IEEE CoG 2020** — verified. Bridge card play, no-trump, 100 deals per
  experiment. 13 cards: one recursion level with a double-dummy inner player +0.57 to +0.58 tricks per
  deal over PIMC-100; a random inner player only reached parity (+0.10, +0.08). 5 cards (20-card deck):
  level 1 with DDS inner player +0.24 (σ 0.05), random inner player +0.10; level 2 +0.20, level 3
  +0.16 ("the third recursion is actually not beneficial"; "more worlds is better" fails, best first-
  level count 20-30). Each level multiplies cost by about worlds × moves × hand length (13 cards,
  (100,100): 12 h per deal).
- **SPARTA** (Lerer et al. 2020, Hanabi): search on top of a shared blueprint, deviating only when the
  gain exceeds a margin (0.05) after a 100-rollout floor; proved search does not make the blueprint
  worse apart from sampling error; partners searching independently without safeguards did worse.
- **Sturtevant & White 2006**: a learned open-hand evaluator inside sampled search (not score-aware).
- **Project status**: the decision-time engine of PLAN §6.8 is not built; the project's hybrid
  yardstick (exact search to trick k, a value network after) is the built form of a depth-limited
  search with a learned leaf (`docs/EXPERIMENTS.md` Parts B-D).

---

## 8. Learning: self-play, distillation, networks

- **Self-play RL in team card games.** DouZero (2021): Deep Monte-Carlo [values learned from final
  results] with a small MLP; 48 cores + four 1080Ti, passing DeltaDou at about 5 × 10^9 steps (30
  days); from-scratch RL overtook a 50M-sample supervised network within two days. PerfectDou (2022):
  PPO with a perfect-information critic ("perfect-training, imperfect-execution": γ 1.0, GAE λ 0.95,
  entropy 0.1, model lag 1); 880 cores + 8 GPUs, 2.5 × 10^9 steps; its ablation shows plain PPO
  without the oracle critic is weak. DouZero+ (2022): adds hand prediction. DouRN (2024): 2-6 residual
  blocks on top of DouZero's MLP raised win rate to 52-57% with 2-3× longer training; replacement at
  the same depth gave 50.3%. DanZero+ (2023): PPO under a day only because it started from a 30-day
  DMC model. OADMCDou (IJCAI 2024): fixed perfect features in DMC did not beat DouZero in 30 days; a
  linear anneal 1 → 0 over 20 days plus clipped Q-updates did (ADP +0.104 ± 0.041).
- **Mahjong.** Suphx (2020): "global reward prediction" turns per-round points into chance of winning
  the whole game; "oracle guiding" trains the policy on perfect features and anneals them; plain
  distillation from an oracle policy "does not work well"; 44 GPUs × 2 days per agent; top 0.01% of
  human players. Mortal: per-round reward = change in expected final payoff.
- **Poker.** AlphaHoldem (AAAI 2022): Trinal-Clip PPO, K-best self-play. Pluribus: CFR, §2.3.
- **Expert Iteration** (Anthony, Tian & Barber 2017): training on the search's full action
  distribution gave 50 ± 13 Elo over chosen-action targets, with top-1 accuracy almost unchanged.
  AlphaZero: the same loop at scale.
- **KataGo** (2019): playout-cap randomisation (full search on about 25% of moves) 1.37× training
  efficiency; auxiliary ownership/score heads 1.65×, auxiliary policy target 1.30×; policy-surprise
  weighting shows positions where the network was most wrong more often.
- **Distillation.** Hinton, Vinyals & Dean 2015 (soft targets); Rusu et al., ICLR 2016 (policy
  distillation, several teachers into one student).
- **Loss against data size.** Kaplan et al. 2020 and Hoffmann et al. 2022 (language models): test loss
  falls as a power law of the training-set size, with exponents of a few hundredths to a few tenths;
  they were not re-read on 2026-10-23 (cited from memory of the papers). Project measurement: the value
  error of the trick-2 oracle falls with an exponent of about -0.11 on the number of rows, its sibling
  regret about -0.20 (`docs/EXPERIMENTS.md` D6-D8, D11); O_3 at 40M against 80M rows lowered value error
  and not decision loss (C4).
- **Training on a teacher's noisy values.** Learning with noisy labels (Natarajan et al., NeurIPS 2013;
  Zhang et al., ICLR 2017 on memorisation of noisy labels; not re-read on 2026-10-23). Project
  measurement: hybrid labels (exact search through trick 2, O_3 at the leaves; root-mean-square error 2.1%
  against exact) trained the trick-2 oracle as well as exact labels of the same count and better at the
  same money (D6).
- **Partners and humans.** Fictitious Co-Play (Strouse et al. 2021): skill-level checkpoints make a
  bot a good partner to strangers (10.6 vs 4.7 with a human proxy; 8.6 vs 1.0 with random partners);
  architecture diversity added nothing; the population plateaued at about 32; 114 human participants
  preferred FCP partners. Other-Play (Hu et al. 2020): the cross-play protocol. piKL (Jacob et al.
  2022): search regularised toward a human model; its Hanabi imitation model used 235,954 games; no-
  press Diplomacy (Bakhtin et al. 2022) about 46,000 games. AlphaStar: KL penalty toward the
  supervised policy, exploiter agents.
- **Reward shaping.** Ng, Harada & Russell 1999: potential-based shaping preserves optimal policies
  (one agent). Lu, Schwartz & Givigi (JAIR 2011): per-player potentials are necessary and sufficient
  in n-player general-sum games. Devlin & Kudenko (AAMAS 2011/2012): shaping preserves the set of
  equilibria but "can alter the joint policy converged upon"; potentials that change during training
  keep the guarantee if each transition uses the potential in force when visited.
- **Integer inference.** Stockfish NNUE docs — verified: int16 first layer and accumulator, int8 hidden
  layers with int32 accumulation, clipped ReLU; weights must be clamped during training; quantisation
  error negligible for shallow nets; no Elo figure published. Lc0 (2018): post-training int8 of a
  large CNN cost about −300 Elo; 2024 claim of int8 without degradation (policy only). KataGo ships
  FP16. No card-game int8 measurement exists.
- **Project status**: phases B-E of PLAN §8 (self-play RL, league, human models) are not started;
  oracle networks, students by distillation and integer inference are built (`docs/EXPERIMENTS.md`
  Parts B-E).

---

## 9. Objective: the match, the money, and calibration

- **Backgammon match-equity tables** (MET): the chance of winning a match from any score, built by
  backward induction; Rockwell-Kazaross rolled out every score 38,880 times; Kazaross-XG2 100,000,
  "piecemeal so that all possible resulting scores had already been rolled out". Zare: a table wrong by
  a few tenths of a percent costs almost nothing (100,000 GNU Backgammon matches: 50.1% vs 49.9%).
- **Calibration**: cross-entropy on a softmax is a strictly proper scoring rule; temperature scaling
  (Guo et al. 2017); equal-width bins are unstable for rare probabilities that are never the top label
  (Nixon et al. 2019).
- **Project status**: the exact equity table is built and in every run (`docs/EXPERIMENTS.md` A2).

---

## 10. Networks as leaf evaluators, their errors, and spending exact work where it matters

These sources underlie the hybrid yardstick and its later fixes (2026-10-16 to 22).
- **TreeStrap** (Veness, Silver, Uther & Blair, NeurIPS 2009) and TD-Leaf: training a value function
  on the values of a deeper search that uses a learned evaluation at its leaves; TreeStrap trained a
  chess evaluation from self-play to master level and outperformed TD-Leaf. Errors in leaf values carry
  into the labels (the project's note: with frontier values off by at most e the backed-up value of the
  deciding team is off by at most e; with four teams a flipped near-tie can move other teams' values by
  more).
- **The optimizer's curse** (Smith & Winkler, Management Science 52(3), 2006) — verified (abstract and
  erratum): choosing the best of several alternatives by unbiased estimates yields systematic post-
  decision disappointment, because selection favours over-estimates; the authors propose Bayesian
  shrinkage; a 2007 erratum states the magnitudes in their Figure 3 were too small. Project test:
  `docs/EXPERIMENTS.md` C7.
- **Value of computation** (Russell & Wefald 1991; Hay, Russell, Tolpin & Shimony 2012): spend a
  computation where it can change a decision. With a leaf off by e and a density f(0) of option gaps
  near zero, the expected loss is about f(0)·e²/2, so expected squared error measures the value of an
  exact solve (derivation in `docs/m3-hybrid/learned-leaves.md`).
- **Learning to defer / selective prediction** (Geifman & El-Yaniv 2017; Mozannar & Sontag 2020): a
  second model decides when the cheap answer can be trusted and when to hand the case to an expert.
- **Gradient-boosted trees** (Friedman 2001; row subsampling 2002; histogram splits as in LightGBM,
  Ke et al. 2017).
- **Guarantees for a threshold**: Learn then Test (Angelopoulos, Bates, Candès, Jordan & Lei 2021):
  fixed-sequence testing of thresholds keeps the largest whose upper confidence bound on risk is
  within target, P(risk ≤ target) ≥ 1 − δ. Conformal risk control (Angelopoulos, Bates, Fisch, Lei &
  Schuster 2022): (n·R(t) + B)/(n + 1) ≤ target in expectation. Empirical Bernstein bound (Maurer &
  Pontil 2009). Project measurement: the finite-sample bounds need hundreds of thousands of decisions
  to certify a useful target (`docs/EXPERIMENTS.md` C11).
- **Finding weak groups**: multiaccuracy and multicalibration (Hébert-Johnson, Kim, Reingold &
  Rothblum, ICML 2018; Kim, Ghorbani & Zou, AIES 2019): if an auditor can find a group of weight ≥ γ
  with mean error > α, "find, correct, repeat" reaches multiaccuracy after at most about 1/(α²γ)
  corrections. Slice discovery: SliceFinder (Chung et al., ICDE 2019), DivExplorer (Pastor, de Alfaro &
  Baralis, SIGMOD 2021), Domino (Eyuboglu et al., ICLR 2022). False-discovery control: Benjamini-
  Hochberg. Boosting (Freund & Schapire 1997). Project use: `docs/EXPERIMENTS.md` B5, C2-C3.
- **Choosing training data by predicted error**: Learning Loss for Active Learning (Yoo & Kweon, CVPR
  2019); prioritised experience replay (Schaul et al., ICLR 2016); importance sampling by loss
  (Katharopoulos & Fleuret, ICML 2018); KataGo policy-surprise weighting; Neyman allocation; defensive
  importance sampling (Hesterberg 1995: keep a share drawn the plain way). Project use: C12.
- **Warm starts**: Ash & Adams 2020: a warm-started network can generalise worse than one trained from
  scratch; shrink-and-perturb is their remedy. Continual training on old plus new data (replay) is the
  practice of self-play engines. Project use: C3, C12.
- **Ranking losses**: RankNet (Burges et al., ICML 2005); LambdaRank (pairs weighted by what swapping
  them costs; Burges, Ragno & Le 2006); the listwise ListNet (Cao et al., ICML 2007: a softmax over the
  candidates, cross-entropy to the target's softmax); decision-focused learning, "Smart Predict-then-
  Optimize" (Elmachtoub & Grigas 2022), whose SPO loss is the regret of the decision taken. Project use:
  the trick-2 student's loss is a listwise KL over the candidate cards with temperature T and halved its
  golden-set loss (`docs/EXPERIMENTS.md` E12, E17); the same idea as a pairwise term on O_3's sibling
  positions made O_3 worse (C9, D12); the sibling regret used to judge oracles (D6-D12) is the SPO loss on
  the positions a choice point's cards lead to.
- **Surrogate loss against task loss**: minimising a convex surrogate need not minimise the task loss
  (Bartlett, Jordan & McAuliffe, JASA 2006, on classification-calibrated surrogates; not re-read on
  2026-10-23). Project measurement: stopping on the validation decision loss instead of the value
  cross-entropy lowered the student's golden-set loss by about 16% without a ranking term and the ranking
  term with that stop rule by about half (E12); O_3's ranking arm kept improving its value score for 40
  epochs while its decisions were worse (D12).
- **Ensembles**: deep ensembles' disagreement as uncertainty (Lakshminarayanan et al. 2017). Project
  use: C8.

---

## 11. Evaluation statistics

- **Duplicate play**: the same deals replayed with seats rotated. For four competitors, every seat
  assignment of two programs except all-A and all-B (14 arrangements; Sturtevant; GO-MCTS 2024 with a
  Wilcoxon signed-rank test over blocks).
- **SPRT / generalised SPRT** (Stockfish Fishtest): log-likelihood ratio from block mean and variance;
  stop when a pre-set α, β and effect size are reached.
- **AIVAT** (Burch, Schmid, Moravčík & Bowling, AAAI 2018): 85% standard-deviation cut (44× fewer
  games) in the DeepStack human match where one strategy was fully known; about 3× in SD bot-vs-bot;
  MIVAT in six-player poker 20-25%.
- **Ratings for multi-team tables**: Plackett-Luce; Weng & Lin (JMLR 2011); OpenSkill (JOSS 2024);
  TrueSkill. Elo is two-player.
- **Exploitability probes**: approximate best response (Timbers et al. 2022); AlphaStar exploiters.
- **Project status**: the arena with 14-way duplicate blocks, the canary and σ_b are built
  (`docs/EXPERIMENTS.md` A2); no tournament between bots has been run yet.

---

## 12. Infrastructure facts (as checked on the dates given)

- **vast.ai** (docs and live probe, 2026-10-13): no CPU-only compute product (`num_gpus=0` returns
  storage offers); CPU and RAM are allocated in proportion to the GPU fraction; interruptible instances
  are stopped when outbid (processes killed); storage and bandwidth are billed while loading. Project
  measurements of rented machines: `docs/EXPERIMENTS.md` A8, `docs/VAST-MACHINES.md`.
  Checked in use on 2026-10-22/23 (`docs/EXPERIMENTS.md` A10): `--jupyter --direct` is the runtype
  `jupyter_direc ssh_direc ssh_proxy`; `PUT /api/v0/asks/<offer id>/` rents an offer in one call;
  `update instance` followed by `recycle instance` rebuilds a rented machine's container in place and can
  take minutes to act; `/api/v0/instances/` is deprecated for `/api/v1/instances/`; an instance record
  carries `public_ipaddr`, `ports`, `jupyter_token`, `disk_usage`, `inet_down_billed` and the stored
  `extra_env`; a stopped instance can lose its GPU slot to another renter.
- **PyTorch 2.15** drops CUDA 12.6 wheels and Maxwell/Pascal/Volta GPUs; 2.14.1 is the last line for
  them.
- **Cloudflare** (as reported by the 2026-10-13 review; conflicting figures to be re-checked against
  live docs): Workers 128 MB per isolate, startup 400 ms or 1 s, script 10 MB compressed or 64 MiB
  uncompressed; Durable Object attachments 2 KB or 16 KB; Containers at $0.000020 per vCPU-second.

---

## 13. What the sources say against some methods (recorded as not used, PLAN §16)

| method | what the sources report |
|---|---|
| CFR / equilibrium solving | has not worked for trick-taking games (Long et al. 2010; Rebstock et al. 2024) |
| paranoid search | deeper but not stronger in card games (Sturtevant thesis §8.3); a paranoid oracle reached about half the best-response ceiling in 45s (§1.1) |
| plain PIMC as the final engine | saturates at ~100-160 worlds and suffers strategy fusion; recursion keeps gaining (§3, §7) |
| hand-written rules as the engine | Bridge Baron was overtaken by search programs (§2.1) |
| Best-Reply Search / BRS+ | only one opponent moves at a time (§4) |
| ReBeL / Player of Games | guarantees are two-player zero-sum only |
| endgame databases | Setrograde's gain is depth; 41's cost is the number of hidden-card worlds (§3, §5) |
| post-training int8 of an unclamped network | Lc0 −300 Elo (§8) |
| single-tree ISMCTS, GO-MCTS | no trick-taking result beats a solver- or rollout-based engine; GO-MCTS lost to Kermit (§2.2, §4) |

---

## 14. Sources

Primary papers and pages (most also listed with notes in `docs/PLAN.md` §18 and
`docs/REVIEW-2026-10-13.md` §9):
- GIB: https://arxiv.org/abs/1106.0669 · αμ: https://arxiv.org/abs/1911.07960 · Furtak thesis:
  https://doi.org/10.7939/R3H98ZQ52 · Furtak & Buro 2013: https://skatgame.net/mburo/ps/recmc13.pdf ·
  Bouzy et al. 2020: https://ieee-cog.org/2020/papers/paper_82.pdf · Setrograde:
  https://arxiv.org/abs/2411.09089 · Kermit: https://cs.du.edu/~sturtevant/papers/skat.pdf · Long et al.
  2010: https://ojs.aaai.org/index.php/AAAI/article/view/7562 · Kupferschmid & Helmert:
  https://gki.informatik.uni-freiburg.de/papers/kupferschmid-helmert-cg2006.pdf
- Sturtevant IJCAI 2003: https://webdocs.cs.ualberta.ca/~nathanst/papers/spec_prune.pdf · thesis:
  https://webdocs.cs.ualberta.ca/~nathanst/papers/multiplayergamesthesis.pdf · UCT 2008:
  https://webdocs.cs.ualberta.ca/~nathanst/papers/multi-player_UCT.pdf · Prob-max^n:
  https://webdocs.cs.ualberta.ca/~nathanst/papers/probmaxn.pdf · soft-max^n:
  https://webdocs.cs.ualberta.ca/~nathanst/papers/softmaxn.pdf · Sturtevant & Korf 2000:
  https://cdn.aaai.org/AAAI/2000/AAAI00-031.pdf · MP-Mix: https://www.ijcai.org/Proceedings/09/Papers/113.pdf
  · Nijssen & Winands: https://dke.maastrichtuniversity.nl/m.winands/documents/Multi_Overview.pdf ·
  Doppelkopf: https://ai.dmi.unibas.ch/papers/sievers-helmert-ki2015.pdf · Edelkamp KBPS:
  https://arxiv.org/abs/2104.05423 · outer learning: https://arxiv.org/abs/2512.15435 · GO-MCTS:
  https://arxiv.org/abs/2404.13150 · EPIMC: https://arxiv.org/abs/2408.02380
- Beliefs: card location https://arxiv.org/abs/1903.09604 · policy inference
  https://arxiv.org/abs/1905.10911 · History Filtering https://arxiv.org/abs/2311.14651 · SPARTA
  https://arxiv.org/abs/1912.02318 · LBS https://arxiv.org/abs/2106.09086 · OBL
  https://arxiv.org/abs/2103.04000 · Ataraxos https://arxiv.org/html/2511.07312v2 · POMCP
  https://proceedings.neurips.cc/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf
- Learning: DouZero https://arxiv.org/abs/2106.06135 · PerfectDou https://arxiv.org/abs/2203.16406 ·
  DouZero+ https://arxiv.org/abs/2204.02558 · DouRN https://arxiv.org/html/2403.14102 · DanZero+
  https://arxiv.org/html/2312.02561v1 · OADMCDou https://www.ijcai.org/proceedings/2024/0660.pdf ·
  Suphx https://arxiv.org/abs/2003.13590 · Mortal https://github.com/Equim-chan/Mortal · AlphaHoldem
  https://ojs.aaai.org/index.php/AAAI/article/view/20394 · ExIt https://arxiv.org/abs/1705.08439 ·
  KataGo https://arxiv.org/abs/1902.10565 · FCP https://arxiv.org/abs/2110.08176 · Other-Play
  https://arxiv.org/abs/2003.02979 · piKL https://arxiv.org/abs/2112.07544 · Diplomacy
  https://arxiv.org/abs/2210.05492 · Wizard https://arxiv.org/abs/2205.13834 · IMPALA
  https://arxiv.org/abs/1802.01561 · NNUE
  https://raw.githubusercontent.com/official-stockfish/nnue-pytorch/master/docs/nnue.md
- Shaping and calibration: Ng 1999 https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
  · Lu et al. https://arxiv.org/abs/1401.3907 · Devlin & Kudenko
  https://eprints.whiterose.ac.uk/id/eprint/75111/1/aamas11.pdf · Guo et al.
  https://arxiv.org/abs/1706.04599 · Nixon et al. https://arxiv.org/abs/1904.01685 · MET
  https://bkgm.com/articles/met.html · Zare https://bkgm.com/articles/Zare/WrongMatchEquityTable/index.html
- Leaf evaluation and deferral: Smith & Winkler 2006 https://doi.org/10.1287/mnsc.1050.0451 (erratum:
  https://jimsmith.host.dartmouth.edu/wp-content/uploads/2022/04/pds_correction.pdf) · TreeStrap
  (Veness et al., NeurIPS 2009) · Learn then Test https://arxiv.org/abs/2110.01052 · conformal risk
  control https://arxiv.org/abs/2208.02814 · learning to defer (Mozannar & Sontag)
  https://arxiv.org/abs/2006.01862 · multicalibration (Hébert-Johnson et al.)
  https://arxiv.org/abs/1711.08513 · Learning Loss https://arxiv.org/abs/1905.03677 · prioritised
  replay https://arxiv.org/abs/1511.05952 · Ash & Adams https://arxiv.org/abs/1910.08475 · deep
  ensembles https://arxiv.org/abs/1612.01474
- Evaluation: AIVAT https://arxiv.org/abs/1612.06915 · MIVAT
  https://bowlingmh.github.io/papers/09ijcai-mivat.pdf · Fishtest
  https://github.com/official-stockfish/fishtest/wiki/Fishtest-mathematics · OpenSkill
  https://joss.theoj.org/papers/10.21105/joss.05901 · exploitability https://arxiv.org/abs/2004.09677
- Game family: Pagat 25 https://www.pagat.com/spoil5/25.html · Wikipedia Twenty-five
  https://en.wikipedia.org/wiki/Twenty-five_(card_game) · Forty-fives
  https://en.wikipedia.org/wiki/Forty-fives · Parlett https://www.parlettgames.uk/histocs/maw.html ·
  boards.ie 31/41 https://www.boards.ie/discussion/2057461308/31-41-card-game-rules · Irish 25 app
  https://irish25cardgame.com/scoring/ · 45s AI https://github.com/jitousig/fortyfives-ai
- Added 2026-10-23 (cited from memory, not re-read): Kaplan et al. https://arxiv.org/abs/2001.08361 ·
  Hoffmann et al. https://arxiv.org/abs/2203.15556 · Zhang et al. https://arxiv.org/abs/1611.03530 ·
  Natarajan et al. (NeurIPS 2013, "Learning with noisy labels") · Cao et al., ListNet (ICML 2007) ·
  Burges, Ragno & Le, LambdaRank (NeurIPS 2006) · Elmachtoub & Grigas https://arxiv.org/abs/1710.08005 ·
  Bartlett, Jordan & McAuliffe (JASA 101, 2006)
- Not readable at the time (2026-10-13): Bounan & Edelkamp ISAIM 2024; full text of Edelkamp KI 2024;
  the Irish 25 app's in-game 41 rules.
