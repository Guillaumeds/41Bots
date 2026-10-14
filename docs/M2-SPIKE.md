# M2 measurement spike: what an exact solve costs (2026-10-14)

PLAN amendment A2 asks for a **naive** exact solver to be measured before any speed-up is designed. This is that measurement. Every figure here is from a deliberately simple solver: no pruning [skipping branches that cannot change the answer], no move ordering, no "equivalent cards". The optional transposition table [a memory of positions already solved, so the same position reached by a different order of play is solved once] is plain and is only consulted at the start of each trick.

## What was measured
- **Solver** (`crates/solver`): exact max^n [each partnership, at its own turn, picks the card best for itself] with every card face up. Every finished line is valued as the four teams' **chance of the money**, from the match-equity table v0: a game ended by 41 or a jink moves to the next game (or ends the match); a normal hand end looks up the next hand's state. Legal cards and trick winners come from the engine; in-hand scoring from `equity::hand` (the engine's exact order).
- **Tie rule** (pre-M4): at each node the mover maximises their team's entry; cards are tried in increasing engine card number and a later card replaces the best only if strictly better, so among tied cards the lowest-numbered is kept.
- **Checks passed:** 1-, 2- and 3-card endgames agree exactly with an independent brute force that drives clones of the real engine (values of the root and of every root card); every value sums to exactly 1; repeat solves are identical; table on/off give identical values, choices and child values under both tie orders.
- **Worlds:** random deals (engine RNG, seed 1), 0–3 whole hands skipped at random so the match scores vary, then random legal play to the start of trick k. At trick 1 an outstanding rob is first resolved with a random legal burn. Table v0 rebuilt with `equity estimate-model --bot rules-v1 --hands 1000000 --seed 1` + `equity build` (sha256 `64740dab…6808`, identical to the recorded one).
- **Machine:** the cloud VM, `Intel(R) Xeon(R) Processor @ 2.10GHz` (lscpu), 4 vCPU, 16 GB. Tricks 3–5 ran one world at a time on one thread; tricks 1–2 ran 4 worlds at once (one per thread), so their times include some memory contention. The **second CPU type** demanded by the M2 gate is still to do (vast.ai EPYC).
- **Command:** `target/release/solver-spike --table <equity-v0.npy> --trick K --worlds N [--tt on] [--cap-ms 60000] [--threads 4] [--flip]`. Total measurement time about 17 minutes.

## Results
"Nodes" = positions where a card is chosen. "Root seat" made no material difference at any trick (per-seat medians within about ±25% of the overall median, no pattern), so it is summarised in one line per row; the per-seat lines are in the binary's output.

| Start of trick | Table | Worlds | Solved | ms median | ms p90 | ms max | Nodes median | Nodes p90 | Nodes max | Table hit rate | Largest table | Peak memory |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 (1 card each) | off | 2000 | 2000 | 0.001 | 0.001 | 0.04 | 8 | 8 | 8 | – | – | table only (76 MB) |
| 4 (2 cards) | off | 2000 | 2000 | 0.036 | 0.073 | 0.36 | 751 | 1,728 | 2,303 | – | – | 76 MB |
| 4 | on | 2000 | 2000 | 0.063 | 0.131 | 2.3 | 751 | 1,728 | 2,303 | 0% | 256 entries | 76 MB |
| 3 (3 cards) | off | 1000 | 1000 | 31 | 109 | 402 | 694k | 2.5M | 10.1M | – | – | 76 MB |
| 3 | on | 1000 | 1000 | 18 | 55 | 188 | 232k | 699k | 2.4M | 92.5% | 34k entries (3.7 MB) | 76 MB |
| 2 (4 cards) | off, 60 s cap | 8 | 3 | 27,000 | – | 47,500 | 623M | – | 750M | – | – | 76 MB |
| 2 | on, 60 s cap | 100 | 93 (7 timeouts) | 9,200 | 33,400 | 59,300 | 93M | 325M | 551M | 98.4% | 3.4M entries (365 MB) | 2.1 GB (4 worlds at once) |
| 1 (5 cards) | on, 60 s cap | 16 | **0 (16 timeouts)** | – | – | – | >470M in 60 s | | | 98–99% | 2–3.8M entries after 60 s | 2.9 GB (4 at once) |

Speed: about 22 million nodes per second per thread without the table, 10 million with it.

### Ties
| Start of trick | Decision nodes with ≥2 equally best cards | Root: share of legal cards tied for best | Roots with any tie | Root value changes when the tie order is reversed |
|---|---|---|---|---|
| 4 | 82% | 88% | 75% (1504 / 2000) | 0 / 2000 |
| 3 | 82% | 82% | 83% (828 / 1000) | 0 / 1000 |
| 2 | 83% | 77% | 85% (79 / 93) | not measured |

Reversing the tie order changed the chosen root card exactly in the worlds with a tied root (by construction) and **never changed the root's money-chance vector** at tricks 3–4. Ties are therefore everywhere, as A2 predicted: the move ordering, the pruning proof and, from M4, the policy-weighted backup must all be designed around them.

## What this means for the A2 assumptions
- **"Exact is cheap from trick 3": confirmed, with a caveat.** From trick 3 a naive solve takes a median 18 ms (31 ms without a table), worst seen 0.4 s. That is already usable for teacher data and post-mortems, but **not yet** for 20+ worlds inside an 8 s / 5 s / ≤3 s decision: 20 worlds × 55 ms (p90) ≈ 1.1 s per candidate card. Equivalent cards, pruning and move ordering are needed before in-search use at trick 3; tricks 4–5 are effectively free (under 0.1 ms).
- **Trick 2 is not cheap.** Median 9 s, p90 33 s and 7% over 60 s, even with the table, and 365 MB of table per world. Exact solving at trick 2 is a teacher/offline tool at best; at decision time trick 2 needs rollouts, as A3/A4 say.
- **Trick 1 is out of reach for the naive solver**: no world finished in 60 s (over 470 million nodes each). The plan's "0.1–150 s per world at trick 1" is too optimistic for this solver; until pruning and equivalent cards are measured, trick 1 must use rollouts (A3/A4 already say so).
- **Transposition table:** worthless at trick 4 (nothing repeats), 1.7× faster at trick 3, essential from trick 2 (98% hit rate). Memory, not time, becomes the limit at tricks 1–2.
- **Next steps (M2 proper):** equivalent cards with the 41 exceptions, Sturtevant's pruning with the stated tie rule (pruning on = off asserted), move ordering, the score-state key as already used here; repeat this table on a second CPU type (vast.ai EPYC) and after each speed-up.
