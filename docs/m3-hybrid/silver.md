# The trick-1 silver set: O_2 checked at the decision level without exact trick-1 solves

Branch `eng-silver` (from `m3-hybrid` at `b70b168`), 2026-10-17. Owner's goal of the day: test the
trick-2 network O_2 at the decision level on many trick-1 decisions, at every seat and in every
score and race group, although no exact trick-1 golden set is affordable (about 15–160×10⁹ solver
nodes per exact trick-1 world, `rung2.md` §4). Nothing was rented for this note.

## 1. In plain words

A golden set is a list of decisions with the exact value of every card in many deals [worlds:
complete guesses of where the hidden cards are]. At trick 1 the exact values cost too much. The
**silver set** replaces "exact" by the best affordable thing: **H_3**, the search that plays tricks
1 and 2 exactly and asks the proven trick-3 network O_3 at the start of trick 3. The network under
test, O_2, is used by **H_2**, which plays only trick 1 exactly and asks O_2 at the start of trick 2.
Both value the same decisions on the same deals; the score is how much money H_2's choice of card
loses when the cards are valued by H_3 [the decision loss against the silver reference], with the
same statistics as the golden-set checks: the 95% upper bound of the mean loss overall, by seat
band and by seat, the bias [average error] for each team, the agreement, and now also by score
group and race group.

## 2. What it proves, and what it does not

For one decision (its 320 deals fixed), write E_c for card c's exact value averaged over the deals
(the decider's team's money chance), S_c for H_3's, δ_c = E_c − S_c for H_3's error on that card;
e* the exact best card, s* the silver's best card, h the card H_2 chooses. Then, exactly,

```
true loss of H_2  =  loss of H_2 against the silver  +  H_3's own true loss  +  (δ_s* − δ_h)
   E_e* − E_h     =        S_s* − S_h                +     E_e* − E_s*       +  (δ_s* − δ_h)
```

(write δ_s* − δ_h out as E_s* − S_s* − E_h + S_h and the right-hand side collapses to the left). The last two terms are
the **silver's own error** at that decision: what H_3 itself loses, plus how differently H_3
misvalues the card it prefers and the card H_2 chose. When H_2 picks the silver's card, the true
loss is exactly H_3's own. So:

- **It proves**, over 400 decisions at all eight seats and every score and race group, how much
  money O_2 adds to what O_3 loses, on the trick-2 positions H_2 really meets in play (reached by
  exact trick-1 search from real trick-1 decisions), at a scale the 32 saved exact trick-1
  decisions (8 of them with worlds, seats 1–2 only) cannot reach. A pass at 0.01% against silver
  says: H_2 is within 0.01% of H_3 (95% upper bound).
- **The true loss is at most the loss against silver plus the silver's own error.** O_3's measured
  error on the 2,000-decision trick-2 golden set: 0.0068% overall (95% upper bound), 0.012–0.016%
  at seats 1–2, its own team under-valued by about 0.0085% overall (`audit-loop.md` §8, round 2).
  Those are trick-2 decisions; trick-1 decisions reach start-of-trick-3 positions after two exact
  tricks, a different mix, so the trick-2 figures are a guide, not a measurement, of the silver's
  own error at trick 1. The **exact anchor** (§4) measures it.
- **It does not prove** that H_2 is within 0.01% of exact search; it cannot see a weakness that O_2
  and O_3 share in the same situations (it would cancel: the race groups help show where the two
  networks are both weak); it says nothing about the number of deals (both searches use the same
  320, paired); and a group is only judged from 30 decisions (`--min-group`), so rarer situations
  are reported, not judged. It is never trained on (seed 1 decisions and worlds, like the golden
  sets; training labels use other seeds).

## 3. What was built

**Worlds** (`search worlds --method hybrid --cut K --net WEIGHTS`, `search::measure`): every
candidate of every decision valued by H_K in every world, on the golden sets' own decision and
world streams (seed, label 1, decision id, world number), so a silver set and an exact set of the
same decisions value exactly the same deals. Everything the exact worlds runner does still holds:
one work queue over all (decision, world) tasks, longest first; a checkpoint line per world as it
finishes; resume world by world (here or on another machine, from the uploaded checkpoint); the
file byte-identical for any thread count and any interruption. Under `--schedule auto` a hybrid
run takes one world per thread (the position-aware plan prices exact worlds). The timing file and
the checkpoint record each world's network evaluations as well as its solver nodes (`evals`), and
the progress file `evals_valued`.

**Marked as silver, never mistaken for a golden set.** The records say `"method": "hybrid"` and
carry `fp.frontier = {cut, net_sha256, encoder}` (every checkpoint line too); exact records keep
their exact bytes (the new fields are absent). Refusals, each tested:
- a worlds run never resumes, extends or joins records of another method or another network
  (records, checkpoint lines, and `read_worlds`' join of several files);
- `search hybrid-check` against exact values (the default, every golden-set check of the runner)
  refuses a silver file ("a silver file is never an exact golden set");
- `search hybrid-check --reference silver` refuses another network's silver file, and exact
  records unless `--allow-exact` (they then stand in for the silver values of their decisions:
  the anchor; the report counts them);
- `hybrid_rung` refuses a silver file as `golden_worlds_input` at step 0, before any label;
- the saved trick-1 worlds (`--saved`, `search::saved`) refuse a silver line (before, other
  methods were skipped quietly);
- `search score --yardstick` refuses a silver yardstick unless `--allow-silver` (and says so);
  `search curves` reads `exact` unless told `--method hybrid`; `search agree` refuses files that
  mix methods and notes a silver reference.

**Check** (`search hybrid-check --reference silver --reference-net O3 [--reference-cut 3]`,
`search::hybrid::run_check_ref`): H_K against the deeper H_R (R > K) on the same worlds, the
reference values read from a silver file (a `--verify-pct` sample recomputed with H_R first and
required byte-identical, as for golden sets) or recomputed when no file is given (the plain
H_2-vs-H_3 comparison, which the stored-file check reproduces exactly: tested). The report says
"H_2 vs silver (H_3, net f5766aa98a1d …)", the JSON `reference` and `values.reference`; a
decision's `exact_*` fields hold its reference's values (`silver` true).
`--race-groups` adds two tables, reported and never part of the verdict, each group judged only
from `--min-group` (30) decisions: the **score groups** (the stratum S1–S4 of the hand's start
state) and the **race groups** (`nn::outcome::race_class_of`, the five classes the label shards
count, of the trick-2 positions H_2's own search hands O_2; a decision is in a class when at least
half its reached positions are, else "mixed"). `--progress FILE` reports the golden-set form too.

**Runner** (`search_measure` chunk, `crates/runner/src/measure.rs`, "A silver reference"):
`reference: {cut, net_input}` makes the worlds step the silver set
(`silver-t1-c3-n<to>.jsonl`, resumable world by world like the golden sets, its fingerprint naming
the method and the network); `share` (default on) stores the decisions and the silver worlds at
`inputs/sha256/<hex>`; `worlds_input` starts from a stored set (a later O_2 costs no H_3 solve);
`check` (one, or a list with distinct cuts) runs `search hybrid-check` of a given network against
the chunk's worlds (silver or exact) with the score and race groups. Manifests (not launched):

| manifest | what |
|---|---|
| `docs/vast/m3-hybrid-t1-silver-mech.manifest.json` | mechanism test: 16 decisions (two a seat) × 8 worlds, then a toy O_2 (round 1 of `m3-hybrid-t2-mech-6`) checked against them |
| `docs/vast/m3-hybrid-t1-silver.manifest.json` | the silver set: 400 decisions × 320 worlds (count and to are the size), shared |
| `docs/vast/m3-hybrid-t1-silver-check.manifest.json` | a given O_2 against the stored silver set (fill in the set's and the O_2's SHA-256) |
| `docs/vast/m3-hybrid-t1-anchor.manifest.json` | the optional exact anchor (§4) |

**Tests**: silver worlds byte-identical on 1, 2 and 4 threads, under the position-aware schedule,
and after an interruption and a resume; every world's values the hybrid's own (H_3 with the cut,
not exact); silver and exact never mixed in a worlds run (records, checkpoints, joins, missing or
stray network, decisions after the cut); the silver check reproduces the plain H_2-vs-H_3
comparison on a tiny trick-1 case (recomputed or from a stored file, 1 and 3 threads, race groups
on or off); the refusals above; the runner's settings, inputs and the chunk with a fake `search`
(the worlds, the fingerprint, the sharing, the check's command and summary); every committed
manifest parses.

**By hand** (any machine; the files are byte-identical for any thread count):

```
search decisions --trick 1 --count 400 --seed 1 --allow-uneven --threads T --out decisions-t1.jsonl
search worlds --decisions decisions-t1.jsonl --method hybrid --cut 3 --net o3-int.safetensors \
  --to 320 --threads T --tt-mb 64 --equity equity-v0.npy --out silver-t1-c3-n320.jsonl --progress p.json
search hybrid-check --trick 1 --cut 2 --net o2-int.safetensors --decisions decisions-t1.jsonl \
  --worlds silver-t1-c3-n320.jsonl --reference silver --reference-net o3-int.safetensors \
  --to 320 --verify-pct 1 --race-groups --threads T --equity equity-v0.npy --json hc-t1-o2-silver.json
```

## 4. The exact anchor (optional)

A few dozen leader and second-seat decisions of the silver set solved exactly on the same deals
(`m3-hybrid-t1-anchor`: 24 decisions, ids 32–39, 64–71, 96–103, × 16 worlds, the position-aware
schedule), then H_3 and H_2 each checked against them. With the silver check's numbers for the
same decisions this measures, decision by decision, the silver's own error at trick 1 (H_3 against
exact) and how far the loss against silver is from the true loss; `hybrid-check --reference silver
--allow-exact` can also use the anchor's exact worlds in place of the silver ones for those
decisions. Exact trick-1 worlds at seats 1–2 cost about 10¹⁰ to 1.6×10¹¹ solver nodes each (the 31
saved: mean about 3.6×10¹⁰, a heavy tail); 384 of them are about 1.4×10¹³ nodes, about 3 hours of
an EPYC 9654 machine at 1.2×10⁹ nodes a second, about $3 (rough: the tail decides).

## 5. Cost (pilot on the build VM)

Measured on the build VM on 2026-10-17 (4 vCPU Intel Xeon at 2.1 GHz, shared with builds and
tests: times are about 1.2 times what an idle VM gives, checked by rerunning two worlds alone,
whose values came out byte-identical). Decisions of the full set's generator (`search decisions
--trick 1 --count 400 --seed 1`), four a seat (ids 32k + 4·(seat − 1) + (k mod 4), k = 1–4; ids
0–31, the saved trick-1 set with its known hardest worlds, left out), world 0 of each, the adopted
O_3 (f5766aa9…, 2048-1024-256) at the start of trick 3, one thread a world, 64 MB tables:

| seat | H_3 worlds | mean s | max s | mean solver nodes | mean network requests | H_2 mean s (4 worlds × 4) | H_2 max s |
|---|---|---|---|---|---|---|---|
| 1 (leader) | 2 finished, 2 stopped | 712 (finished); 1,085 counting the stopped at their stop | 743; stopped at 1,117 and 1,798 | 47×10⁶ | 10.9×10⁶ (decision 66) | 1.9 | 3.9 |
| 2 | 4 | 330 | 1,116 | 20×10⁶ | 10.7×10⁶ (2 recorded) | 0.53 | 1.7 |
| 3 | 4 | 211 | 580 | 16×10⁶ | 3.9×10⁶ | 0.11 | 0.40 |
| 4 | 4 | 30 | 100 | 3.2×10⁶ | 0.57×10⁶ | 0.04 | 0.28 |
| 5 | 4 | 13 | 31 | 0.8×10⁶ | 0.25×10⁶ | 0.02 | 0.05 |
| 6 | 4 | 1.1 | 1.5 | 0.08×10⁶ | 0.03×10⁶ | < 0.01 | 0.01 |
| 7 | 4 | 0.2 | 0.3 | 0.02×10⁶ | 0.01×10⁶ | < 0.01 | < 0.01 |
| 8 | 4 | 1.8 | 6.3 | 0.08×10⁶ | 0.05×10⁶ | < 0.01 | < 0.01 |

H_2 here is with the mechanism test's toy O_2 (1024-512-128, `0cb6d614…`): 34,000 network
requests a leader world, 7,800 at seat 2, a few at seats 6–8; a 2048-wide O_2 costs about three
times as much per request.

What the numbers say:
- **H_3 at trick 1 is almost all network time.** A leader world searches about 47 million solver
  nodes (about 14 s of a rented thread at 3.3 million nodes a thread-second) but asks the network
  about 10 million times; the rest of its 10–30 minutes is the 2048-1024-256 forward pass. The
  solver's table size changes nothing (64 MB and 512 MB: the same 45.2 million nodes, the same
  bytes, the same time: the extra requests at 64 MB are answered by the network's memo).
- **The tail is heavy and reaches seats 2 and 3**: single worlds of 10–30+ thread-minutes here
  (decision 33 world 0 was stopped after 30 minutes; decision 0 world 0 of the saved set after 27).
  One world per thread, leaders first (the runner's order), so the full run ends on cheap worlds;
  the mechanism test (128 worlds) splits each world over 8 threads (the parallel solver, the same
  values: tested) so that one heavy world does not hold the machine for hours.
- **A rented thread is slower**: measured with this very O_3, H_3 on trick-2 decisions took 46.9 ms
  a world (90 µs per network request) on the trick-3 loop's fully loaded 224-thread AMD EPYC 7663
  against 16.9 ms (36.5 µs) on this VM idle: 2.5–2.8 times; and the toy O_2 on the 384-thread
  EPYC 9654 of `m3-hybrid-t2-mech-6`, 86–100 µs per request against about 50 µs here loaded. So a
  thread of the AMD EPYC 9654 (368 usable threads, $1.01/h) is taken as **2.0 to 2.7 times** this
  pilot's thread-seconds.

**Price of the silver set** on that machine, per decision count × worlds (seat means as above; the
low end counts the stopped leader worlds at their stopping time and the 2.0 factor, the high end
at twice it and the 2.7 factor):

| set | build-VM thread-hours | hours on 368 threads | dollars at $1.01/h |
|---|---|---|---|
| 400 × 320 (the golden sets' size) | 7,700–11,100 | 42–81 | $42–82 |
| 400 × 160 | 3,900–5,500 | 21–41 | $21–41 |
| 200 × 320 | 3,900–5,500 | 21–41 | $21–41 |
| 200 × 160 | 1,900–2,800 | 10–20 | $11–21 |
| 400 × 80 | 1,000–1,400 | 5–10 | $5–10 |

Worlds are nested (N worlds are the first N of 2N, the same deals), so a smaller set extends later
to a larger one without redoing anything (`from`/`to`: the extension records join). About 65–75% of
the cost is seat 1, 15–20% seat 2. The mechanism test (16 decisions × 8 worlds) is about 40–70
minutes of a 190+-thread machine, about $1, its wall time set by its heaviest world on 8 threads;
it records the real machine's thread-seconds per seat (the timing file), which should replace this
pilot's before the full set is launched. A later O_2's check against the stored set (no H_3 solve)
is minutes of H_2 plus the 1% verification sample (about 1% of the set's cost): about 0.5–1 hour,
under $1. Not measured: the rented machine itself (hence the range), and the heavy tail beyond the
pilot's 32 worlds (four decisions a seat is a thin sample of it).
