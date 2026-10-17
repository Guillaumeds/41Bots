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

**Decisions of 2026-10-16 night** (owner, 22:00 Irish time):
- **gamma 0.0001.** The audit must find weak groups down to 0.01% of positions (ten times finer
  than 0.001). The runner's `audit_gamma` (default 0.001) reaches the auditor as `--gamma`; at
  0.0002 or less the audit shard defaults to 20M rows at trick 4 and 10M at trick 3, so a group of
  0.01% holds 1,000-2,000 rows (half of them for discovery). The shard is deleted once the auditor
  has read it.
- **The auditor at scale** (`fortyone_nn.audit`, numpy only): every pass runs on all threads in
  tasks of a fixed size, the shard memory-mapped, shared arrays as work files, every sum a whole
  number, so the output is byte-identical for any number of workers (tested 1 against 4). The
  enumeration counts pairs and triples sparsely from each row's ~46 items (not dense products);
  a group is not extended when its sums prove no extension can pass a threshold (the optimistic
  estimate of exhaustive subgroup discovery, Grosskreutz et al. 2008, Lemmerich et al. 2016: a
  subset of at least gamma's rows has mean square at most sum(q)/minsup and every team's |bias| at
  most sum(max_t |e_t|)/minsup; valid, it never changes the candidates, and in practice it prunes
  little). **Auditor timings** (this VM: 4 vCPU shared with three other agents, load 9-19, so
  wall times are inflated; trick 4, the lead-4 network 1024-512-128, gamma 0.0001, 4 workers):
  300k rows (minsup 15): network and error columns 43 s, items 2 s, trees 2 s, enumeration 990 s
  wall of which about 6 minutes paused (508 items, 121k pairs, 8.7M triples counted, 7.0M
  frequent), confirmation 7 s (191 candidates, 98 confirmed); 607 CPU-seconds in all, peak
  3.1 GB (resident, shared pages counted per process). 1M rows (four trick-4 shards incl. race
  harvests, minsup 50): network and error columns 73 s, trees 4 s, enumeration 792 s (8.1M
  triples, 12,772 candidates); the confirmation of 12,772 candidates was cut by the measurement's
  own 20-minute limit (since then confirmed 512 groups at a time per block). The triples dominate
  (about 85% of the CPU). Projection, linear in rows: about 1,700 CPU-seconds per million rows for
  the enumeration and 250-300 for the network at 1024 wide (about 12 times that at
  4096-2048-1024-256); so 20M rows at trick 4 on a 128-thread machine (about 80 cores' worth)
  about 7 minutes of enumeration plus 1 minute (1024 wide) to 14 minutes (4096 wide) of network,
  and half that for 10M rows at trick 3. Not yet measured on a rental; the auditor's limit grows
  with the shard (90 minutes plus 15 per million rows above 2M).
- **The standard is a setting.** `search hybrid-check --standard PCT` (default 0.01) judges the
  overall, band, position and slice rows; the runner's `audit_standard` (default 0.01) and
  `audit_push_standard` (none by default). Once the loop is flat at the standard and a push
  standard is set, the same round is judged again by the push standard (its golden slices and
  mini sets checked again, files `…-push`) and the loop carries on, same rental, same dollar cap,
  until flat at the push standard: **0.005% at trick 4 and 0.0075% at trick 3**. The heartbeat's
  `standard` and the summary ("flat at 0.01% after round 3; pushing to 0.005%") show the phase.
- **Mini golden sets for tiny groups.** A group lighter than 0.1% keeps decisions with at least
  500 × its weight of their reached positions in it (at least 5%: a group of 0.01% keeps
  decisions with 5% of their positions in it) instead of a half, so rejection sampling finds
  decisions at all; at most `audit_mini_candidates` (200 × `audit_mini_decisions`) candidates are
  drawn. A set's verdict needs at least `audit_mini_min` decisions (default 30; a handful that all
  agree would give a bound of 0): with fewer (none included), the group is reported "unmeasurable
  at the decision level (N decisions kept)" in every round report and the summary ("unmeasurable:
  k groups"), never passed or failed, never failing and never blocking "flat": the overall row
  covers it, and a group too rare to measure must not keep the loop from reaching flat (the push
  starts only after flat). It is targeted only when the auditor confirms it again.
- **Diminishing returns.** Two rounds in a row on the largest shape of the capacity ladder that
  lower neither the worst failing group's decision-level bound nor the overall bound by
  `audit_min_gain` end the loop ("diminishing returns"), with the groups still limiting it (weight,
  SD, bound) in the summary; never while the shape can still step up. It replaces "capacity ladder
  exhausted". The ladder holds only shapes larger than the rung's widths, each once.
- **The chunk keeps the best round**, not the last: flat at the tightest standard first, then the
  lowest golden overall bound, the lowest worst group bound, the earlier round; also on a stop
  ("best: round k" in the summary).

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
- **Trick 2** (rung 3): the same loop; the audit shard at trick 2 costs 3 s per label, so 200k
  positions (~10 minutes on 184 threads) per round. Its decision-level stage is only the rough test
  on the saved exact trick-1 worlds (32 decisions): no trick-1 golden set will be built (owner,
  2026-10-16), so at trick 2 "flat" means no confirmed weak group at the position level, with the
  rough test reported beside it. Budget: up to $35 (owner).

Verification per round: the audit report (groups, weights, bias, SD), the golden slices, the mini
golden sets, the integer network's SHA-256, and the identity of every exact label with the solver
(the harvest's own check).

## 8. First run on real hardware: the trick-4 mechanism test (2026-10-16 night)

Run `m3-hybrid-t4-audit-mech-1` (binary 147bfa4; AMD EPYC 9654, 384 threads, RTX A4000, $1.01/h; 36 minutes,
about $0.60; manifest `docs/vast/m3-hybrid-t4-audit-mech.manifest.json`): 9M trick-4 labels, a 1024-512-128 network,
three rounds, 1M-row audit shards at gamma 0.001, 1M-row targeted harvests, at most two mini golden sets a round, the
stored 400-decision golden set. Every step ran and the run ended DONE (`round limit`):

| round | found / confirmed | failing at the decision level | golden 95% upper | train | audit | slices | mini set | harvest |
|---|---|---|---|---|---|---|---|---|
| 1 | 433 / 343 | 19 | 0.0069% (fail on a band) | 6.5 min | 1.6 min | 5 s | 3-5 s | 2 s |
| 2 | 449 / 354 | 12 | 0.0055% (pass) | 8.1 min | 1.7 min | 5 s | 3-5 s | 2 s |
| 3 | 452 / 360 | 44 | 0.0108% (fail) | 8.9 min | 1.7 min | 5 s | 3-5 s | 2 s |

What it taught, and what changed before the overnight loops:
- At trick 4 everything but training costs seconds, so the overnight trick-4 loop measures every waiting group each round
  (mini sets up to 400 a round, 100 decisions each).
- A golden slice of 80-130 decisions cannot show 0.01% at 95% even for a sound group when the mean loss is 0.004%
  (round 1: "trumps are spades", 0.0176%): the bound is mostly the sample's noise. The overnight loops generate larger
  golden sets (4,000 decisions at trick 4, about a minute to make and 3 a round to check; 2,000 at trick 3, about 75
  minutes of CPU beside round 1's training), so a group's slice holds 5-10 times the decisions.
- Networks retrained from scratch differ from round to round (round 3 was worse than round 2 on the golden set): the
  loop now reports the best round's network as its result (eng-auditscale).
- A mini set that keeps only a few decisions is "unmeasurable" (fewer than `audit_mini_min`, 30), never a pass.

### The second mechanism test, and the start of the overnight loops (2026-10-16/17 night)

Run `m3-hybrid-t4-audit-mech-2` (binary fcb3e73, every build merged; AMD EPYC 7B12, 122 threads, $0.67/h; stopped by
hand after round 2's audit, about $1.25): gamma 0.0001 on a 4M-row audit shard, the generated 4,000-decision golden
set, the start shape 1024-1024-512-128, at most 400 mini sets a round of 100 decisions.

| step (round 1, 122 threads) | time |
|---|---|
| golden set, 4,000 decisions x 320 exact worlds | 0.7 min |
| audit shard, 4M rows | 0.2 min |
| training, 9M rows | 4.8 min |
| auditor, 4M rows at gamma 0.0001 (544 found, 401 confirmed) | 2.2 min |
| golden slices on 4,000 decisions | 2.1 min |
| targeted harvest, 2M rows | 4 s |
| a mini golden set (100 decisions) | about 17 s each, several hundred a round |

Round 1's network on the 4,000-decision golden set: mean decision loss 0.0031%, 95% upper bound 0.0049% (seats 1-2
0.0131%, 3-4 0.0047%, 5-8 0.0017%). Round 2 trained beside round 1's mini sets: the trainer waited for its data while
the CPU built them and the stall watch left it alone (the risk the eng-runfix build addressed). Round 2: 443 confirmed,
48 failing at the decision level. A stopped run keeps its rounds in the partial uploads (`partial/rung/000000/out/`)
and every round's network content-addressed; there is no final bundle.

The overnight trick-4 loop's first run (`m3-hybrid-t4-loop-1`, 10M-row audit shards) confirmed **1,551 groups** in
round 1 (4,858 candidates; 92.5% of positions inside a confirmed group; overall SD 0.773%, race class 4 SD 2.39%). At
about 20 s a mini set, 400 a round would have spent most of the night measuring rather than improving, so the loop was
relaunched with at most 60 new mini sets a round (the groups with the largest weight x SD^2 first; the rest wait and are
reported as not yet measured), and with a patient price ladder after a relaunch climbed to $1.97/h in ten minutes
(`m3-hybrid-t4-loop-3`: AMD EPYC 7B13, 122 threads, 504 GB, $0.73/h).

### Overnight results, 2026-10-17 (both loops stopped on diminishing returns)

Trick 4 (`m3-hybrid-t4-loop-3`; AMD EPYC 7B13, 122 threads, 504 GB, $0.73/h; 00:05-05:45 UTC). Golden set 4,000
trick-3 decisions x 320 exact worlds; 95% upper bound of the decision loss (all / pos 1-2 / pos 3-4 / pos 5-8), largest
|bias| on the all row; groups from 10M-row audit shards at gamma 0.0001:

| round | shape | network | all | 1-2 | 3-4 | 5-8 | bias | found / confirmed | failing at decision level |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1024-1024-512-128 | b28750e3d447 | **0.0016%** | 0.0026% | 0.0023% | 0.0012% | 0.0025% | 4,858 / 1,551 | 8 (44 unmeasurable, 1,473 not yet measured) |
| 2 | 1024-1024-512-128 | ba808dcf05dd | 0.0021% | 0.0055% | 0.0026% | 0.0008% | 0.0030% | 8,375 / 3,787 | 14 |
| 3 | 2048-2048-1024-256 | 6d72e18cfcf8 | **0.0016%** | 0.0040% | 0.0020% | 0.0009% | 0.0035% | 7,690 / 2,724 | 13 |
| 4 | 4096-2048-1024-256 | (stopped in its mini sets) | 0.0024% | 0.0059% | 0.0023% | 0.0012% | 0.0045% | - / 3,026 | 13 |

Trick 3 (`m3-hybrid-t3-loop-1`; AMD EPYC 7663, 224 threads, 2x A100 40GB, $1.60/h; 22:53-05:05 UTC). Golden set 2,000
trick-2 decisions x 320 exact worlds (generated tonight, about 3 s an exact world, shared content-addressed); groups
from 10M-row audit shards:

| round | shape | network | all | 1-2 | 3-4 | 5-8 | own-team bias | found / confirmed | failing |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2048-1024-256 | b34e12b51099 | 0.0088% | 0.0153% | 0.0071% | 0.0088% | -0.0125% (fails) | 802 / 628 | 56 |
| 2 | 2048-1024-256 | f5766aa98a1d | **0.0068%** | 0.0122% | 0.0080% | 0.0052% | -0.0085% | 820 / 630 | 46 |
| 3 | 2048-2048-1024-256 | a099061b6aed | 0.0074% | 0.0164% | 0.0057% | 0.0056% | -0.0096% | 812 / 629 | 47 |
| 4 | 2048-2048-1024-256 | (stopped in its mini sets) | 0.0076% | 0.0153% | 0.0060% | 0.0061% | -0.0142% (fails) | - / 629 | 47 |

Network keys: `inputs/sha256/<sha>` with b28750e3d447400523d2af9cbe85e4fbbb414200474560a7c1038b7adf0ce5a0 (trick 4,
round 1), 6d72e18cfcf800be9071afacea8ece38b8fdacb2c07757839656a6ef8431f157 (trick 4, round 3),
f5766aa98a1d769e31122c94cc4da9790c84508fab5c432d0c0358fe3124faad (trick 3, round 2).

What the night showed:
- **Trick 4 passes at 0.0016%, every seat band below 0.006%, every bias below 0.005%,** on ten times the old golden
  set. The old 400-decision set's failing seat bands (0.021% at seats 1-2) were small-sample tails, not the network.
- **Trick 3's leader is the open problem:** seats 1-2 stay at 0.012-0.016% and the leader's own team is under-valued by
  about 0.02-0.03% in that band (-0.0085% to -0.0142% overall), round after round, at both network sizes.
- **"Flat" was out of reach by construction.** At 10M rows the auditor confirms thousands of groups at trick 4 (the
  position-level SD threshold, 1%, is close to the network's overall SD, 0.8-1.0%), and flat needs every one measured at
  the decision level; 60 mini sets a round measure a few percent of them. So the push to 0.005% / 0.0075% never
  started.
- **Retraining from scratch every round is noisy:** the golden bound moved 0.0016 -> 0.0021 -> 0.0016 -> 0.0024% at trick
  4 while the data only grew, and the capacity rule stepped up almost every round (its "worst group" changes from round
  to round). More width did not help either loop.

## 9. Decisions of 2026-10-17 (owner, after the overnight loops)
- **Two training modes, chosen per run:** `scratch` (the loop as it ran overnight: a fresh network each round, the
  capacity ladder) gets a trick "pretty flat"; `additive` (start from a stored network, keep training it on old and new
  data together, keep the result only if no worse on the golden set, harvest only the groups still out of spec) fills
  the rest. The switch is decided together per trick; the rule of thumb is that the golden set passes overall and in
  every band and the rounds' changes are within noise. Labels are never made twice: a run stores its shards and a later
  run reuses them.
- **Machines:** the focused, additive phase is CPU-bound (finding and proving rare groups), not GPU-bound; trick 3 and
  trick 2 rent for CPU threads with any GPU; trick 4 in scratch mode is the one case that wants a fast GPU.
- **Trick 2 runs the same design first** (scratch, exact labels, shards stored), and its data then feeds the additive
  phase.
- **The trick-2 loop, merged** (2026-10-17, branch `m3-hybrid-t2merge`): the trick-2 design (no golden set,
  position-level stop, the saved trick-1 worlds a rough test) on top of the overnight loops' builds. At trick 2
  `audit_gamma` applies, the standards and the mini-set settings are refused, the best round is chosen at the
  position level (flat first, then the lowest overall SD on its own fresh audit shard), diminishing returns read
  the capacity check, and the audit and targeted shards are stored for later runs (an audit shard is deleted only
  once shared). Its manifest and cost: `docs/m3-hybrid/vast.md`, "The audit loop at trick 2 (O_2)".
- **Trick-2 labels from the trick-3 network** (bootstrapping, TreeStrap-style): allowed only through a gate (a trick-2
  network trained on hybrid labels against one on exact labels, same money, same exact checks) and only once trick 3 is
  near 0.005-0.0075%; the audit and every check stay exact. Errors carry forward and add along the chain, and with four
  teams a flipped near-tie can move the other teams' values in a label by more.

## 10. Training modes, reused labels and the leader focus (build of 2026-10-17, branch `eng-additive`)

The owner's decisions of §9, as run settings of `hybrid_rung` (module notes:
`crates/runner/src/hybrid/audit/additive.rs`, `crates/runner/src/hybrid/reuse.rs`; trainer:
`python/nn/fortyone_nn/train.py`, "Additive training").

**`train_mode`** — `"scratch"` (the default: the loop exactly as it ran overnight, a fresh network each
round, the capacity ladder; every existing test unchanged) or `"additive"`:
- `init_net` (required): the stored integer network to continue, `inputs/sha256/<hex>` (`rung.json`'s
  `net.int_key`); `init_net_float` (required): the same network's float weights (`net.float_key`).
  Checked at step 0, before anything is installed: an integer network for the rung's trick and head,
  taking the encoder's 909 features and giving the head's outputs at the cut; its hidden widths are the
  rung's (`widths`, when given, must agree; otherwise they are taken from it); the float file has the
  same layer shapes and the same training record. Then, once the Python environment is in and before
  any label is made, the init check (`train.py --check-init-only`) proves, byte for byte, that the
  float weights quantise to `init_net`: a wrong pair costs seconds, not hours of labels.
- **Round 1 is `init_net` as it is** (`train.py --init --epochs 0`: copied and measured on the
  validation shard; no training spent on what it already knows). Its audit, golden check, mini sets
  and targeted harvest run as in scratch mode; at trick 3 and earlier its audit shard is driven by
  `init_net` (no `harvest_net` needed). It is the best so far.
- **Round r ≥ 2 continues the best network so far** on the base shards and every targeted shard so
  far (old and new data together: new data alone makes a network forget), for `additive_epochs` (8)
  at `additive_lr` (0.0003), early stopping on the validation shard as before. `shrink_perturb` (λ,
  off by default; Ash & Adams, NeurIPS 2020): every weight λ·w + N(0, σ²) first, σ = (1 − λ) times the
  standard deviation of the layer's fresh initialisation (σ = (1 − λ)/√(3·fan_in)), drawn on the CPU
  from the seed `train_seed × 1000 + r`: the same seed gives the same network.
- **Never lose a won pass, and no worse overall** (rule as amended by the review of 2026-10-17). On the
  golden set, judged at the standard as the check judges `loss_pass` and `bias_pass`, round r is *worse*
  than the best when (a) it loses a pass the best had: the overall bound, a seat band's bound
  (positions 1-2, 3-4, 5-8) or a team's |bias| on the overall row; (b) a band or bias the best already
  fails gets worse by more than the noise tolerance; or (c) its overall bound is worse than the best's by
  more than the noise tolerance. Otherwise (ties and improvements included) it is adopted. The noise
  tolerance is `adopt_tolerance`, default 0.002 percentage points of money chance: about the overnight
  loops' round-to-round moves of the golden bound while the networks did not really change (trick 4:
  0.0016, 0.0021, 0.0016, 0.0024%). The first form of the rule (every raw number no worse, no
  tolerance) let noise reject most rounds, better ones included. A round not adopted leaves the best
  as it was: the summary and the round's report say "NOT adopted" with every reason, its targeted data
  still joins the next round, the failing groups are restored wholesale to the best's, and it cannot
  end the loop as flat. Every harvest (audit shards, targeted harvests) is driven by the adopted best,
  never by a rejected network. The chunk keeps the latest adopted round's network and never the
  scratch ranking: an adopted round gets a provisional result at adoption, so a run stopped before the
  round's judgement ends still keeps it; before round 1 is judged the chunk names `init_net`.
- **At trick 2** (no golden set) the same rule reads the round's fresh audit shard, with both networks
  on the same positions: right after round r's auditor, before the shard goes, the best network so far
  is measured on it (`python -m fortyone_nn.audit --overall-only`, the full audit's `overall` numbers),
  and the same rule reads it: a team's |bias| that loses its pass (within the standard) or, where the
  best already fails it, grows by more than `adopt_tolerance`, or an overall SD (no standard there)
  worse by more than `adopt_tolerance`, is worse. This is the measure the scratch trick-2 loop keeps its best round by (overall
  SD on the audit shard), made paired, because each round's shard is a different sample and comparing
  each network on its own shard would let sampling noise decide. No seat bands there (the shard's rows
  do not record the decider) and no failing groups. A round not adopted that confirms no group ends
  the loop ("nothing new to train on"). Cost: one network pass over the audit shard a round (the
  auditor's first step; minutes).
- The capacity ladder is off (the shape is `init_net`'s; `audit_shapes` refused). A step up to a larger
  network is not built in additive mode (it would start fresh and is refused at parse): run it in
  scratch mode. The rounds run
  one after another (a training starts from the network adopted after the previous round's golden
  check, so it cannot start beside that check; `overlap` refused). Also refused: `epochs`, `lr` (use
  the additive ones), `label_parts` above 1; additive mode needs `audit_rounds` above 0.
- **Focused harvest:** the targeted harvest takes the groups still failing, the groups confirmed this
  round without a pass at the decision level (failing, not yet measured, unmeasurable) and the groups
  confirmed for the first time; a known group that passes is not harvested again.

**Float networks.** Until this build only the integer network was stored content-addressed; the float
weights were only in each run's bundle and partial uploads. From now on every round's float network is
shared too (`net.float_key`), and `upload_shards` shares every shard's sidecar: in both modes, for the
owner's "store every label" and later additive starts. Neither can fail a step or stop a paid run:
each upload is tried three times, and a final failure is a warning in the heartbeat (the files are in
the run's bundle). Last night's networks: the float
files are in the partial uploads and quantise, byte for byte, to the adopted integer networks
(checked 2026-10-17):

| network | init_net (integer) | init_net_float (float) | where the float file is |
|---|---|---|---|
| O_3, trick-3 round 2 | f5766aa98a1d… | b567967707852e8ae02f64890f4445ceac21eedd356b1beb7a570c157f64d447 | runs/m3-hybrid-t3-loop-1/partial/rung/000000/out/hybrid/audit/round-2/net/float.safetensors |
| O_4, trick-4 round 1 | b28750e3d447… | 4b85237ee8e6ca3f925e43336f6a26ef493230889f3ef8a84c85c0d3f99e4046 | runs/m3-hybrid-t4-loop-3/partial/rung/000000/out/hybrid/net/float.safetensors |

Neither float file is under `inputs/sha256/` yet (the build had read-only access): before launching
`docs/vast/m3-hybrid-t3-additive.manifest.json` or `…-t4-additive…`, download the file and store it
with `launch.py put-input FILE` (the launcher refuses the run until every stored input is there).

**`label_inputs`** — stored shards reused instead of generated, a list of `{ "set", "seed", "shard",
"sidecar" [, "rows", "weight", "rowweights"] }` (`set`: random, rules, harvest, race, val, or extra for
any other training shard such as an earlier run's targeted harvest). A named set is not generated (its
`labels_<set>` and seed settings are refused); each file's SHA-256 is its key, and the sidecar must
record the shard's SHA-256, the format, this binary's encoder, feature count and values version, the
rung's trick and outcomes, the entry's seed, a source that fits the set, and the shard header's rows.
Reused seeds count in the seed-disjointness checks (training, validation, audit, targeted harvests,
golden). With `upload_shards`, a run now also shares each shard's sidecar (and row weights) and writes
ready-to-paste entries: `out/hybrid/shards.json` (`label_inputs`: the base shards) and, in the audit
loop (additive mode, or trick 2 where the targeted shards are shared anyway), `out/hybrid/audit/label_inputs.json`
(every targeted harvest, set `extra`, with its weight).
`label_inputs` works at every cut, trick 2 included (the sidecar must record the rung's trick and its
outcome vocabulary). A run on a binary before this build (the trick-2 loop m3-hybrid-t2-loop-1 on
27ccdc2) shares its shards with `upload_shards` but not their sidecars: to reuse them, store each
sidecar from the run's bundle (`out/hybrid/labels/<file>.json`, `out/hybrid/audit/round-<r>/<file>.json`)
with `launch.py put-input` and name it in the entry. Last night's runs had `upload_shards` false, so their base shards are made again from the same seeds
(byte-identical) by the example manifests, which store them this time.

**Leader focus (plan item 2d).** Every round report has a "Leader focus" line and `leader_focus` field:
the golden check by the decider's seat band and position, with the own team's bias (seat 1 leads the
trick). `harvest_seat_weights` (`"w12,w34,w58"`, e.g. `"2,1,1"`; multiples of 1/256) weighs every
harvested row by its decision's seat band (the rung's harvest and race shards and the targeted
harvests; not the audit shards), through the per-row weights `NAME.npy.rowweights` that
`search labels --seat-weights` writes before the shard is renamed into place (its sidecar records the
weights and the file's SHA-256). Not built: the audit shard's error by the decider's seat — the shard's
rows do not record the decider (only the sidecar's counts by position), so the auditor cannot split
its errors by seat; the golden check's bands are the decision-level measure.
