# The path forward after the O3G2 rounds (proposed 2026-10-22, for the owner's approval)

Status: **proposal**, written from the owner's sequence of 2026-10-22 and our measured results. Nothing
here is running. Every number is cited to `docs/EXPERIMENTS.md` (written "E4", "D2" ...) or marked
"estimate". Where this page and STATUS differ, STATUS is current.

## 1. Where we stand (facts)
- O_3 (trick-3 network): 0.0068% alone (seat 1 0.0201%), 0.0008% with G2 and exact solving (seat 1
  0.0022%). Three attempts to lower O_3 alone (O3G2 rounds, 80M base, specialist S) did not move it
  (C4, C12, C13).
- O_2 (trick-2 network) is provisional: 1.5M exact labels, position error SD 3.6% against O_3's 1.85%
  (D1). It has had **far less data than O_3's 40M positions**.
- Trick-2 student S2 (512-256, 1M decisions, plain H_3 teacher): 0.073% alone on the exact golden set;
  **top-3 shortlist with a perfect final pick 0.0073%** (bound 0.013%), seat 1 0.0148% (E4-E5).
- Trick-1 student: none. Trick-1 teacher (H_2 with O_2) costs about **$84 per million decisions at 64
  deals** (E6, 12,489 decisions an hour on a 256-thread EPYC 7B13, $1.05/h, 5.28 candidates a decision).
- No trick-1 golden, silver or bronze set exists: a trick-1 world solved exactly through tricks 1 and 2
  then O_3 costs 10-30 thread-minutes on the build VM (D4), and one sample measured on 2026-10-22 was
  still unfinished after 50 minutes of one thread (E8).
- App budget: 7 s a decision on a phone. In WebAssembly a trick-2 deal costs 368 ms (leader) to 0.9 ms
  (8th seat) per thread (`WASM-API.md` §8); the search over N deals loses about 0.01% x (243/N)^0.738 at
  trick 2 (A6: N for 0.01% is 243), i.e. about 0.05% at 25 deals, 0.03% at 50, 0.02% at 100 (estimate
  from the fitted curve; all candidates, no shortlist).
- A shortlist alone does **not** make the exact solve cheaper today: the solver values every root card in
  one search, so dropping cards from the list changes nothing (E8, measured).

## 2. How to use the two big gains
Two measured gains of about 8-10 times: the **top-3 shortlist** (student alone 0.073% to 0.0073% with a
perfect pick) and **G2** (O_3 alone 0.0068% to 0.0008%). They are not equally usable:

| gain | where it can run | what is needed to collect it |
|---|---|---|
| top-3 shortlist (about 10x) | online and offline | a **root-restricted solve** (search only the shortlisted first cards; expected cost about the share of root cards kept, to be measured) and enough deals for the final pick: the "perfect pick" used 320 exact worlds, the app has 25-100 |
| G2 + exact leaves (about 8.5x) | **offline only**: 586 thread-ms a deal on a 9B14 against 24 for O_3 alone, far beyond the 7 s phone budget | use it only where compute is free: teachers for students on close calls, yardsticks, the final label pass |
| both, as a router | online | a learned risk model for the **student's** decision (G2's idea moved to decision level): spend the 7 s of search only where the student's top card is likely wrong, skip it elsewhere. G2 shows error concentrates (75% of O_3's error in its riskiest 5% of positions); the student's loss is concentrated by place and score class (leader 0.084%, 5th seat 0.088%, S4 0.089%, E4) and its per-decision concentration is **not yet measured** |

## 3. The path, in order, with cost and the judge of each step
Costs are rental dollars; "estimate" means not measured.

**Step 0: free reads on the build VM (no rental).**
- 0a. **Shortlist + search at the app's deal counts.** Sample N = 25, 50, 100 of the 320 stored exact
  worlds per golden decision; the student shortlists 3 cards; pick the best of them by mean over the N
  worlds; score against the full golden values. Needs a small extension of `student-check`
  (`--deals N`). It uses exact world values, so it isolates the sampling noise and is optimistic about
  leaf error. Decides whether shortlist + search beats the student alone at app deal counts.
- 0b. **How concentrated is the student's loss per decision** (the router's feasibility), from the same
  golden check's per-decision losses.
- 0c. **Root-restricted solve**: design and measure (kept cards must keep byte-identical values).

**Step 1: S2.** (owner's step 1: "train S2 on O3G2")
- Cost of the teacher: H_3 + G2 is about 24 times a plain H_3 world (586 against 24 thread-ms on a 9B14,
  E-table G), so about **$15-30 per million decisions at 64 deals** against about $0.8 plain (estimate
  from the 4.7 thread-s decision cost, E1).
- Run it as a **cheap A/B, not a build**: 250k decisions plain against 250k with the G2 teacher, same
  network and seed, judged on the exact golden set (alone and top-3), about $5-10 plus training.
- Also A/B the decision-aware loss (`rank_weight`, planned step 3 of the student plan, about $0.5).
- Why a student-fit limit is suspected: its gap to a 512-deal teacher is 0.0506% against O_3's own
  0.0068% error, a 1024-512 network was no better and more data helps only as N^-0.23 (E4). The A/B tests
  whether a cleaner teacher moves it.
- The student audit loop (harvest where the student loses most) is **low priority**: the same method did
  not move O_3 (C2, C12).

**Step 2: O_2 (the long pole).** O_2 must exist before the trick-1 teacher, copper and S1.
- Labels are **positions at the start of trick 2, all cards face up**, valued exactly through trick 2 and
  then by plain O_3 ("hybrid labels", D2). Bot play (random, rules-v1, H_2-driven harvest) only chooses
  which positions to label; it never supplies the label.
- Cost per label (D2): plain hybrid about 0.1 thread-s, **exact 11.3 thread-s, G2 labels 9.3 thread-s**.
  So per million labels: plain about $0.07-0.15, exact about $7-9 (estimate at 0.0022-0.003 $/thread-h),
  G2 labels the same order as exact. **G2 labels have no niche**: they cost 1.2x less than exact, which is
  perfect. The real choice is plain (volume) against exact (truth); label quality of plain labels against
  exact: RMS 2.1%, largest team bias 0.013-0.017% (D2). That is the unrun **label test at equal money**
  (`o2-labeltest-1` was cancelled at 8 minutes): judged on the exact validation shards s601/s602/s603,
  bias first.
- Then the audit loop on O_2 as for O_3, standard 0.01%. O_2 is judged on exact trick-2 positions, **not
  on copper** (copper is labelled by O_2 itself).
- Decision to make: adopt whichever label source wins the equal-money test, then the volume (O_3 had 40M).

**Step 3: trick-1 yardsticks** (after O_2).
- **Copper** (new name for the set below bronze): about 2,000 trick-1 decisions labelled by the teacher
  H_2 + O_2 at 128 deals. About 10-20 minutes on a 256-thread machine, **about $0.2-0.5** (E6 rate, estimate
  at 128 deals). Measures how well S1 copies its teacher. It rewards copying: a student better than its
  teacher would score *worse* on it.
- **Playout yardstick** (to judge "better than the teacher"): for each of a few thousand trick-1
  positions, play each of the top-3 cards on many dealt hands with the same fixed strong players
  finishing the hand (S2 or H_3, then exact), compare the mean money chance of the cards on the same
  deals. Independent of O_2 and of PIMC's perfect-information assumption. Cost depends on the playout
  speed; resolving 0.1% takes about 10^5 playouts per position-card pair (estimate); design and a pilot
  first.
- **Small silver/bronze anchor**: a few dozen decisions x a few dozen worlds, to measure the chain
  (S1 vs copper) + (O_2 vs bronze) + (the reference's own error). Estimate $5-15; trick-1 world costs have
  heavy tails (A4, D4), so cap it. Not a full set (the first attempt was stopped at about $100 for about
  10%).

**Step 4: S1.**
- Labels from the teacher H_2 + O_2, 64 deals, racing (extra deals only for close calls). **About $84 per
  million decisions** at the E6 rate and the provisional O_2's width; racing and a restricted root should
  lower it (estimate $40-100 per million). S2 needed 1M decisions and still loses 0.057%, so plan for 1M
  and a learning-curve point at 250k.
- Judged on copper, the playout yardstick and the small anchor; the decision-aware loss as in step 1.

**Step 5: ship S1 and S2 alone** (milliseconds a decision, `method: "student"`, WASM 0.4-0.7 ms).

## 4. Improvements, by expected gain against work
1. **Root-restricted solve and the router** (step 0 decides): turn the shortlist's 10x into speed and
   accuracy online. Trick 2 first (O_3 at the leaves, no G2 online), then trick 1 (needs a smaller leaf
   network: a trick-1 deal is 0.2-0.6 s on the build VM and about 5x that in WebAssembly, so 2-8 deals in
   7 s as things stand).
2. **Disagreement set from the pros** (7 pros x 60 matches, about 420 matches): their moves are not the
   truth. Use them as realistic positions and as a finder: where the bot and a pro disagree, spend heavy
   offline compute (many deals, G2 on close calls) to see who is right. Expected about 6,000 trick-1
   decisions (estimate).
3. **Self-play on very thin slices** (e.g. trick 1 with 2-3 trumps close to the money), judged by the
   **playout yardstick**, not copper. If it wins in a slice, a predictor chooses between S1 and the
   self-play network (the same learned-routing idea as G2). A DouZero-type trial from scratch needs about a
   billion decisions, so start from the teacher's network with a decaying teacher penalty (PLAN phase B);
   estimate $100-300 for a first trial.
4. **Public matches** (thousands of matches from hundreds of people): tens of thousands of decisions, far
   too few for self-play volume. Use for the position distribution humans create and for the disagreement
   finder (item 2).

## 5. Decisions for the owner
- Approve step 0 (free) first; it decides the router and the final-pick design.
- Approve the S2 teacher A/B ($5-10) or skip it.
- Approve the O_2 label test at equal money (plain volume against exact), then the volume.
- Trick-1 yardsticks: copper (cheap) + playout yardstick pilot + a capped small anchor, instead of any
  full silver/bronze set.
