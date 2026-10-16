# The audit loop: finding the hybrid's weak spots automatically and training them away

Owner's requirement (2026-10-16, 21:00 Mauritius): the oracle network must be "proven 0.01" in
every situation of the game, not on average; the weak situations must be found by the machine,
without anyone naming them, harvested, trained on, and found again, until nothing above 0.01%
is left to find. Seat bands are one grouping among thousands and get no special status.

This document is the plan and the standard; `docs/M3-HYBRID-SPEC.md` §3 refers to it. Plain
English first, then the method, then the builds and the runs for trick 4 and trick 3.

## 1. What goes wrong with an average

The check of 2026-10-16 measured the trick-4 network at 0.0057% overall and 0.0159% at seats
3–4. Digging by hand found the cause: positions where one team can win the match in this hand
while another can still win the game ("race class 4"). They are 7% of the positions the search
hands to the network but carry 75% of the decision loss. The seat band happened to show it; a
weakness spread evenly over seats would have passed unseen. The same network was also 8× less
accurate on those positions on fresh, never-trained data, which is the clue to the method: the
weakness is visible on cheap labelled positions long before it is visible on 400 golden decisions.

## 2. The known mathematics

Three strands of published work say how to do this; the loop below is their combination.

- **Multiaccuracy and multicalibration** (Hébert-Johnson, Kim, Reingold, Rothblum, ICML 2018;
  Kim, Ghorbani, Zou, AIES 2019). A predictor `f` of a quantity `v` is *multiaccurate* for a class
  `C` of groups if for every group `g` in `C` of weight at least `γ` [the share of positions in
  it], the average error inside the group is small: `|E[f − v | g]| ≤ α`. Their key result: if an
  *auditor* [a learning procedure] can find any group in `C` where the error exceeds `α`, then
  "find a group, correct on it, repeat" reaches multiaccuracy after at most about `1/(α²γ)`
  corrections. The auditor can be any learner over the input features; the groups it can express
  are the class `C`. This is exactly "find weak scenarios automatically, fix, repeat until none".
  Multicalibration is the stronger version (also conditioned on the predicted value); we audit
  both the signed error [bias] and the spread [SD] per group.
- **Slice discovery / subgroup discovery.** SliceFinder (Chung et al., ICDE 2019) finds
  interpretable groups with high loss by searching feature conjunctions and tree splits, with a
  statistical test per group; DivExplorer (Pastor, de Alfaro, Baralis, SIGMOD 2021) enumerates
  every frequent conjunction of binary features (support ≥ `γ`) and ranks them by how much the
  loss inside diverges from the loss outside, with Shapley-style attribution of which feature
  drives it; Domino (Eyuboglu et al., ICLR 2022) finds slices in a learned embedding when raw
  features are not enough. Our encoder's features are a complete description of the position
  (`crates/nn/src/encode.rs`), so conjunctions over them can express any situation anyone could
  name: seat, race class, who robbed, score gaps, trump holdings, the end-bonus team, the jink,
  and unnamed ones.
- **Reweighting and targeted sampling** (boosting, Freund & Schapire 1997; hard-example mining;
  importance sampling). Once a weak group is found, training focuses on it by sampling more of its
  positions (our harvest, labelled exactly by the solver) and by weighting its rows in the loss.
  This is the "correct" step of the multiaccuracy loop; the labels never change, only how often
  the learner sees them.

Statistical honesty across thousands of candidate groups: discovery and confirmation on disjoint
halves of a fresh shard (cross-fitting), false-discovery control (Benjamini–Hochberg at 5%) over
the candidates, a minimum group weight `γ` (below it a group cannot be measured and is not
claimed), and a new audit shard every round (never the one just trained on, never the golden set).

## 3. The standard, restated

A rung is proven when, for the final network and a fresh audit shard of `N_audit` positions
drawn as the hybrid reaches them:

1. **No discoverable weak group.** No group in `C` (conjunctions of up to 6 feature conditions,
   weight ≥ γ = 0.1% of the shard) confirmed on the held-out half has |bias| > α_bias or
   SD > α_sd, where the thresholds are calibrated once from measured data so that a group at the
   threshold stays under 0.01% decision loss (starting values from 2026-10-16: α_bias = 0.25% and
   α_sd = 1.0% of money chance; the race-class-4 group, SD 2.7%, bias +0.6%, gave 0.0159% at the
   decision level, calm positions at SD 0.3% pass everywhere). The calibration is re-measured
   whenever a confirmed group's decision-level loss is also measured (item 3).
2. **The golden set passes**, overall and in every slice the audit names (a golden decision is in
   a slice when most of the positions its search reaches are), with the §3 bootstrap bound.
3. **Mini golden sets for rare groups**: a confirmed group with fewer than 50 golden decisions gets
   its own set of 50–100 decisions from the same yardstick generator by rejection sampling
   [keep drawing decisions, keep only those whose search reaches the group], solved exactly, never
   trained on; 95% upper bound ≤ 0.01% there too. At trick 3 one such set costs about 15–30
   minutes on a 192-thread machine.
4. **Every training run prints the audit** (groups found, their weight, bias, SD, and the
   decision-level numbers where known), so a network cannot pass by averaging.

Seat bands stay in the report as one view among others; they are no longer the standard.

## 4. The loop

For a rung k (trick-4 network, then trick 3), one round is:

1. **Train** on the current label set (random + rules + harvest shards, with the row weights of
   step 5 if any), as today.
2. **Audit shard.** Harvest `N_audit` fresh frontier positions (new seed, the network-driven search
   at trick 3, exact at trick 4), label them exactly, and compute the network's integer-inference
   error on each (the same evaluator the hybrid uses). Cost at trick 4: seconds; at trick 3: about
   10 thread-seconds per 1,000 rows, so 2M rows in about 2 minutes on 184 threads.
3. **Find the groups.** On half A of the shard, (a) a regression tree of depth ≤ 6 and minimum
   leaf weight γ on the signed error and on the absolute error, and (b) the DivExplorer-style
   enumeration of conjunctions of up to 3 binary features with support ≥ γ, each scored by the
   bias and the SD inside. Candidate groups are then measured on half B; those over the thresholds
   after Benjamini–Hochberg are the round's weak groups, each written as a predicate [a list of
   feature conditions] with its weight, bias, SD and an example position.
4. **Decision-level numbers** for each weak group on the golden set and, where it has fewer than
   50 decisions, on a mini golden set (§3 item 3). These are reported, and they recalibrate the
   thresholds of §3 item 1.
5. **Target.** For each weak group: harvest positions that satisfy its predicate (the harvest keeps
   every reached position inside any weak group, up to a cap per decision; drawing decisions from
   every stratum and seat), to lift the group's share of the training set to at least 3× its natural
   share; and give rows inside weak groups a loss weight of 2 (recorded in the report).
6. **Retrain and repeat** from step 1 with the enlarged set. Stop when every confirmed group's
   decision-level bound (step 4: its golden slice, or its mini golden set) is ≤ 0.01% and the
   golden set passes overall, or after `R_max` rounds (then the report says which groups remain
   and the next lever is network capacity).

**Capacity is part of the loop** (2026-10-16, after lead-4): 2.5× more race positions in the training
set lowered the class-4 error only from 2.74% to 2.58%, so data alone can stall. The loop therefore
grows the network by itself: the shape is taken from a ladder (`audit_shapes`, default
`"<start>;2048,1024,256;2048,2048,1024,256;4096,2048,1024,256"`, the start being the shape chosen by
the width/depth comparison, every width within the proven `nn::MAX_WIDTH`), and a round moves one
step up when the targeted harvest did not lower the worst confirmed group's SD by at least
`audit_min_gain` (default 15%) against the round before; the next round then retrains from scratch on
all the data gathered so far. The top of the ladder with no gain ends the loop with the stop reason
"capacity ladder exhausted". Every round's report names the shape it trained.

**Decision of 2026-10-16, after the first real audit** (auditor `2be1c56`, lead-3c's network on a
fresh 300k-row trick-4 shard): the auditor, told nothing, found the race positions by itself (its
tree groups are 100% race classes 3–4 and hold 93% of all class-4 positions, through the dense
features "a team wins the match by taking the last two tricks"), plus "a team on 36 points" for
each team, at SD 1.6–2.8% against 0.30% for calm positions. But with α_sd = 1% (barely above the
network's overall 0.93%) the confirmed groups covered 82% of the positions, and the smaller ones
were not stable across split seeds. So the roles are split:
- the **auditor's thresholds choose what to target and slice** (erring towards too many groups is
  cheap: they only steer the harvest and name the slices);
- the **standard is decided at the decision level** (golden slices and mini golden sets, the §3
  bootstrap bound ≤ 0.01%): a confirmed group whose decision-level bound passes is not "remaining";
- mini golden sets only for groups with fewer than 50 golden decisions, in order of weight × SD²
  (their expected share of the loss), at most `audit_mini_sets` (default 4) per round;
- thresholds are recalibrated each round from the measured pairs (group SD, group decision-level
  bound) and reported.
Other implementation choices accepted from that build: candidates must also pass a one-sided
p ≤ 0.01 screen on half A (SliceFinder's definition; the errors are very heavy-tailed, so a raw
threshold alone admits thousands of single-row groups); the SD test is a t-test on the per-row
mean square (no normality assumption); errors are per team relative to the trick's leader (the
rows do not record the decider; a decider byte per row can be added later if a slice needs it);
the dense-feature thresholds in predicates are whole numbers in the feature's own units.

**One run, one machine, autonomous** (owner, 2026-10-16 21:10 Mauritius): the whole loop runs
inside one `hybrid_rung` rental, round after round, and ends by itself when the network is
"flat": no confirmed weak group on the fresh audit shard, the golden set and every slice and mini
set under 0.01%. Nobody restarts it between rounds. It is capped by `audit_rounds` and by the
machine's dollar cap, and a stop asked from outside (the launcher's STOP) ends it after the
current step with everything so far uploaded.

**Reporting, so the owner can watch it from the console** (the heartbeat and the run's log):
every heartbeat carries `round` (r of R_max), `step` (train / audit shard / audit / golden slices /
targeted harvest), `groups_found` (candidates on half A), `groups_confirmed` (after half B and
the false-discovery control), `groups_remaining` (confirmed this round, i.e. still out of spec),
`worst_group` (name, weight, bias, SD), `slices_failing` (golden slices over 0.01%, with the
worst), and the overall check numbers of the last round; the run writes `out/hybrid/audit/round-
<r>.txt` (plain English: what was found, how large, how bad, what was harvested, how the network
changed since the previous round) and `round-<r>.json`, and a final `audit-summary.txt` with one
line per round (groups confirmed → groups remaining, overall and worst-slice bounds) and the stop
reason ("flat after round 3", "round limit", "dollar cap", "stopped").

**CPU and GPU in parallel** (owner, 2026-10-16 21:15 Mauritius): within a round the CPU work
that does not depend on the network being trained runs while the GPU trains. The audit shard's
positions and exact labels (CPU) are made during the training of the same round (only the
network's errors on them wait for the network); the mini golden sets' exact worlds and the
targeted harvest of round r (CPU, the largest chunks) run during the training of round r+1,
whose first epochs use the base set and the targeted shards already in, with the new shard
joining when it lands (the streamed-labels mechanism of the step-overlap build); the golden
slices (a CPU check, a minute) run while the next training starts. The heartbeat shows both
steps when two run at once. Neither device waits for the other when it has a big chunk to do.

Everything the loop decides is data: the predicates, the weights, the thresholds, the stop
condition. No rule about play is written anywhere; the labels are the solver's exact values
throughout, and the golden sets are never trained on.

## 5. Why this finds unknown situations, and what it cannot find

Any situation that can be stated in terms of the position and the match context is a conjunction
of the encoder's features, so it is in `C` once it has weight ≥ γ in the positions the hybrid
reaches. The tree and the enumeration search that class without being told what to look for;
seat, race class and robbing fall out of them as ordinary features. What the loop cannot see: a
group lighter than γ (0.1% of reached positions — at 2M rows that is 2,000 positions; rarer
situations are below the resolution of any check this size and are reported as such), and a
weakness that is not a function of the position and context at all, which cannot exist for a
deterministic exact value.

## 6. Builds (Opus; Fable reviews)

The predicate format, shared by every piece: a JSON list of groups, each
`{"name": "...", "conditions": [{"feature": i, "op": "==", "value": 1}, {"feature": j, "op": "<=",
"value": -0.3}, …], "weight": …, "bias": …, "sd": …}` where `feature` indexes the encoder's
feature vector (`nn::encode`, binary features then the dense log-equity features), so Rust and
Python evaluate the same predicate on the same features.

- **B1, the auditor** (`python/nn/fortyone_nn/audit.py`): error computation with the integer
  network (parity with Rust inference as today), the tree and the conjunction search with
  cross-fitting and Benjamini–Hochberg, predicate export, a plain-English report, tests on
  synthetic shards with planted weak groups (found) and without (nothing claimed).
- **B2, Rust**: `nn::predicate` (evaluate a predicate file on encoded features, tested against
  Python on fixtures); `search labels --source harvest --harvest-keep PRED.json [--keep-cap N]`
  (every reached position inside any group kept up to the cap, the rest sampled as today;
  byte-identical for any thread count; sidecar records rows per group); `search decisions
  --keep PRED.json --keep-share S` (rejection sampling of decisions whose searched positions fall
  in a group at share ≥ S: mini golden sets from the same generator); `search hybrid-check
  --slices PRED.json` (the §3 tables per group, a decision in a group by the share of its reached
  positions, with the bootstrap bound).
- **B3, runner** (`hybrid_rung`): settings `audit_rows`, `audit_seed`, `audit_rounds` (R_max),
  `audit_keep_cap`, `audit_weight`, `audit_mini_decisions` (50); one rental runs train → audit
  shard → audit → golden slices and mini sets → targeted harvest → retrain, round after round,
  until the stop condition of §4, uploading each round's network (content-addressed) and audit
  report as it goes, with the heartbeat fields and files of §4; the final check runs with the
  slices. The runner's existing stop file, dollar cap and upload-every-5-minutes behaviour apply
  to every round.
- **B4, docs**: `docs/M3-HYBRID-SPEC.md` §3 points here; `rung1.md`/`rung2.md` get the audit
  tables per round.

## 7. Applying it

- **Trick 4** (rung 1): start from the lead-4 network (race harvest). Audit shard 2M positions.
  Up to 4 rounds in one rental (each: labels a few minutes, training ~15 min on a 24 GB card);
  about $1–1.5 for the rental. Expected: the race groups confirmed in round 1 (already known),
  possibly groups nobody has named; every band and slice under 0.01% by round 2–3, else capacity.
- **Trick 3** (rung 2): start from round 3's network. Audit shard 1M positions (about 1.5 minutes
  on 184 threads with the network-driven search). Up to 4 rounds; training ~10 min each; the
  stored golden set reused; about $2 per rental.
- **Trick 2** (rung 3): the same loop once the trick-1 check tool gives the decision-level
  numbers; the audit shard at trick 2 costs 3 s per label, so 200k positions (~10 minutes on 184
  threads) per round.

Verification per round: the audit report (groups, weights, bias, SD), the golden slices, the mini
golden sets, the integer network's SHA-256, and the identity of every exact label with the solver
(the harvest's own check).
