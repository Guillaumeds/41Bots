# Specialist S: a network trained only on G2's risky trick-3 positions (plan, 2026-10-21)

## Question (owner)
What if we train a network only on the positions G2 [the predictor of where O_3 is wrong] flags as
risky, and in play use that network S on those positions instead of solving them exactly? Does S
score close to exact on those positions, and does the golden set show it?

## Plan (option A, owner-approved 2026-10-21)
1. **Score the stored base.** For every row of the 40M base (`label_inputs`, exact labels), decode
   the position (`nn::encode::decode`), run O_3 on it, and compute G2's score
   (`nn::leafrisk::RiskModel::score`). Rows in G2's riskiest share are kept; the cut is the same
   budget share today's play rule solves exactly.
2. **Write the selection as row weights, not a new shard.** Kept rows get weight 1, the rest weight
   0 would be refused by `rowweights::to_npy` (a weight of 0 is not allowed), so the first version
   writes the flagged rows into a new shard of their own (same format, same labels), and the
   training input is that shard alone. Natural mix inside the flagged share is kept (no
   re-weighting), so the distribution is G2's own.
3. **Train S from random weights** on that shard only (existing trainer, no O_3 start). Nothing
   from the rest of the base, nothing from the rounds' non-risky rows.
4. **Check S before play.** S's error and one-sided error (bias) against exact on held-out flagged
   rows (not used in training).
5. **Golden set in play, routed by G2.** In play, G2 decides which network answers each frontier
   position: a position G2 flags goes to S, every other position goes to O_3 (today's rule sends the
   flagged ones to the exact solver). This is the routed check, and it is what the run measures:
   the `--exact-leaves learned:G2.json` picks the flagged positions, `--defer-expert h3:S` answers
   them, and O_3 answers the rest. `search hybrid-check --exact-leaves learned:G2.json --defer-expert
   h3:S` against the stored golden set (the routed check), compared with: O_3 alone (0.0068%, seat 1 0.0201%), and
   G2 with exact solving (0.0008%, seat 1 0.0022%).

## Expected size and cost (estimates, to be measured)
- Scoring 40M rows: O_3 forward plus G2 per row, about 30–60 minutes on a 192-thread rental, about
  $1–1.5.
- Training S on the flagged shard: about 30 minutes on the graphics card, under $0.5.
- Held-out check and golden run: about 90 minutes, about $2.
- Cap $5.

## Honest risks
- Flagged positions are the hardest and fewest. S may learn them worse than exact, which is the
  answer the test is meant to give.
- Starting from random weights on one slice means no shared knowledge with O_3. That is the point
  of the test, but it lowers S's expected accuracy.

## Status
- Plan written (this file). Build not started yet beyond the code survey: `nn::encode::decode`,
  `nn::leafrisk::RiskModel::score`, `nn::rowweights` exist; a per-row scorer over stored shards
  does not.

## Build progress (2026-10-21)
- **Done:** `search risk-filter` (`crates/search/src/risk_filter.rs`, command in `bin/search.rs`):
  decodes each trick-3 row, scores it with G2 using O_3's own value, keeps rows at or above the
  model's budget threshold for `--step K`, writes them as a shard. Refuses a G2 model built for
  another network, a wrong trick, or step 0. Tests: thread independence, order, the threshold
  cut, an error on a bad row. Clippy and `search` tests pass.
- **Not yet built:** the runner entry that loops this over the 40M base's shards and rents a
  machine; the training of S on the flagged shard (existing trainer, random start); the held-out
  check; the routed golden run (`--exact-leaves learned:G2.json --defer-expert h3:S`).
- **Open choice:** which budget step `K` to use. Today's play rule's step has not been read from
  the code yet; the filter takes it as an argument, so nothing is decided by the build.
- **Not run on real data:** the command has not been run on a real O_3, G2 or base shard. The
  first real run is the rented scoring pass.
- **The cut (decided 2026-10-21):** today's play rule solves a position exactly when its G2 score
  is at least `t = 277338260` (`learned:…leafrisk-o3-v1…:t=277338260`, the 40% share;
  docs/m3-hybrid/learned-leaves.md). `search risk-filter --threshold 277338260` uses that number,
  so the flagged set is exactly the set play sends to the exact solver today. `--step K` remains
  for a budget step of the model's own table.
