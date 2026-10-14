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

---

# M2 optimised solver vs the naive one (2026-10-14, same machine, same worlds)

The optimised solver (`solver::solve_fast`) adds four speed-ups, each with its own on/off switch. **The gate is that it gives exactly the naive solver's answers**: the same money-chance value for the root, the same value for every legal root card, and the same chosen card under the fixed tie order (lowest card number first; also tested with the order reversed).

## The speed-ups, in plain words
1. **Bounded transposition table** at the start of each trick, keyed exactly as amendment A2 says (remaining hands, leader, tricks per team this hand, 5 played, best trump so far and its team, bonus flags; the match state is fixed for the solve). It has a memory cap, our own fixed hash [a fingerprint of the position, computed the same way on every machine], and a fixed replacement rule [when full, forget the position with the fewest cards left, the cheapest to redo]. It stores only exact values and checks the whole key, so a lookup gives the true value or nothing: replacement can cost time, never change an answer.
2. **Equivalent cards** [two cards in one hand that must lead to the same result, so only one is searched]: same "class" and no live card between them. Classes: the 5 of trumps alone (its bonus counts at once); the other top trumps J, Joker, A♥ (they may be held back); the ordinary trumps (they may not, so the A♥ and the ace of trumps are never merged; with hearts trumps the A♥ is a top trump and the K♥ the best ordinary one); each plain suit in its red or black order. Cards played in earlier tricks are gone, so they make the remaining cards neighbours; cards on the table in the current trick still count. The searched card is the first of its group in the tie order, so the reported card is the naive one. The proof is in `crates/solver/src/equiv.rs`; 8 "must merge" and 9 "must not merge" hand-made decks are tested.
3. **Move ordering** [trying likely good cards first]: cheapest winning card first, low cards when partner is winning. The choice is still made by value then tie order, so it cannot change the answer.
4. **Shallow pruning** (Sturtevant and Korf) [skipping the rest of a branch once it provably cannot be chosen]: the four money chances always add to exactly one, so once the next player's team has found at least `1 − b` for itself, the previous player (whose best so far is `b`) can no longer gain from this card. With move ordering on, the rule becomes strict (`> 1 − b`) when the card comes earlier in the tie order than the current best, so ties are still resolved exactly as the naive solver resolves them. Also: a node stops once its team has a certain win. The proof is in `crates/solver/src/fast.rs`.
5. **Deep and speculative pruning: not used.** Deep pruning is known to be wrong for three or more sides (Korf 1991). Speculative pruning (Sturtevant 2003) has no published proof for four sides with our position-only tie rule and ties at 82% of nodes, so per the plan ("only where provably safe") it is left out. Searching root cards on several threads was not built either: whole worlds are already spread over the cores.

## The gate: identical answers
| Check | Worlds | Result |
|---|---|---|
| Same answers as naive, all 16 on/off combinations, both tie orders, real table v0 and two synthetic tables (one built to be full of ties) | trick 5: 150,000; trick 4: 150,000; trick 3: 3,000 (roots at the start of the trick or up to 7 cards into it) | **0 differences** (9.6 million solves at tricks 4–5, 96,000 at trick 3) |
| Same, the 8 combinations with the table on, real table, both tie orders | trick 2: 64 | **0 differences** (1,024 solves) |
| (a) The solver's own play-and-score step vs the engine, random play-outs | 40 matches, every card | equal; **it found one real bug** (below) |
| (b) Brute force by engine clones on 1-, 2-, 3-card endgames, every combination | 93 positions | equal |
| (d) Best response at the root: each root card's value equals what that card really earns when every seat then follows the solver in the engine, and no card beats the chosen one | 36 positions, tricks 3–5 | holds |
| Tiny table (forces constant replacement) | 6 worlds at trick 3 | same answers |
| Forced-denial check (test only: "can the other three teams stop team A winning any more tricks?") vs an engine brute force | 80 worlds at tricks 4–5, each team (low-score worlds only) | equal |

**Bug found by check (a):** when a solve started part-way through a trick in which the 5 of trumps had already been played, the conversion from a game position marked the 5 as "already scored". The answer was still right in release builds, but a debug build would have stopped with an error. Fixed in `from_match`.

Commands: `target/release/solver-identity --leaf <table.npy|fake|tie> --trick K --worlds N --threads 4 [--combos all|lean|tt] [--cap-ms 60000]`; unit tests in `crates/solver/tests/{brute,fast}.rs` (a longer trick-3 sweep is `#[ignore]`d).

## What it costs now
Same seed (1), same worlds, same machine as the spike above; tricks 3–5 one world at a time on one thread, tricks 1–2 four worlds at once. Default switches: table, equivalent cards and pruning on, ordering off (see below).

| Start of trick | Solver | Solved | ms median | ms p90 | ms max | Nodes median | Nodes p90 | Nodes max | Table hit rate | Table memory per world |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | naive | 2000/2000 | 0.001 | 0.001 | 0.02 | 8 | 8 | 8 | – | – |
| 5 | optimised | 2000/2000 | 0.003 | 0.004 | 0.08 | 8 | 8 | 8 | – | – |
| 4 | naive (table) | 2000/2000 | 0.062 | 0.133 | 2.5 | 751 | 1,728 | 2,303 | 0% | – |
| 4 | optimised | 2000/2000 | 0.042 | 0.109 | 2.2 | 419 | 1,156 | 2,303 | 0% | 0.1 MB |
| 3 | naive (table) | 1000/1000 | 17.6 | 55.1 | 164 | 232k | 699k | 2.36M | 92.5% | 3.7 MB |
| 3 | **optimised** | 1000/1000 | **8.8** | **30.0** | **113** | **71k** | **238k** | **839k** | 87.5% | 3.2 MB used, 14 MB reserved |
| 2 | naive (table), 60 s cap | 93/100 (7 over 60 s) | 9,200 | 33,400 | 59,300 | 93M | 325M | 551M | 98.4% | 365 MB |
| 2 | **optimised**, 60 s cap | **100/100** | **3,090** | **15,400** | **29,200** | **18.5M** | **93M** | **174M** | 97.1% | 156 MB used, 448 MB reserved |
| 1 | naive, 60 s cap | 0/16 | – | – | – | >470M in 60 s | | | | |
| 1 | optimised, 60 s cap | **0/8** | – | – | – | >280M in 60 s | | | 96–98% | up to 2 GB |
| 1 | optimised, 300 s cap, 2 at once | **0/2** | – | – | – | >1.55 billion in 300 s | | | 98% | 3 GB cap reached (11M entries) |

(The naive trick-2 median leaves out its 7 time-outs, so the real gap there is larger than 3×.) Peak memory for four trick-2 worlds at once: 1.7 GB (naive 2.1 GB). Per-seat medians at trick 2 range from 1.6 s to 5.5 s with no pattern, as before.

### Which speed-up does what (trick 3, 1000 worlds, one thread)
| Switches on | ms median | ms p90 | Nodes median | Pruning stops |
|---|---|---|---|---|
| table only | 20.7 | 64.8 | 232k | – |
| equivalent cards only | 11.9 | 44.5 | 176k | – |
| table + equivalent cards | 8.6 | 30.3 | 71k | – |
| table + pruning | 23.1 | 68.7 | 232k | 27,544 in 329M nodes |
| table + equivalent cards + pruning (**default**) | 8.8 | 30.0 | 71k | 9,303 in 108M nodes |
| all four (ordering too) | 10.1 | 36.0 | 71k | 15,576 |

- **Equivalent cards are the real win**: they skip about 3 cards in 4 at a typical node, cutting nodes 3.3× at trick 3 and 5× at trick 2.
- **Shallow pruning almost never fires** (about 1 node in 10,000). With four teams and a typical chance of the money near 25–35% each, the next team rarely reaches "1 minus my best" for itself. It is kept on because it is exact and costs nothing measurable; it matters more in late-match states (e.g. one team on the verge of the money).
- **Move ordering does not pay**: it adds 15% time and saves no nodes, because there is almost no pruning for it to speed up. It is off by default (switch kept and gated).
- The bounded table alone is slightly slower than the naive unbounded one (hashing, growth), and is what keeps memory bounded at tricks 1–2.

### Ties and root flips (optimised solver, trick 3)
Reversing the tie order changed the chosen card in exactly the 828 / 1000 roots that have tied best cards and **never changed the root's money-chance vector**, the same as the naive solver.

## What this means
- **Trick 3 now costs a median 9 ms per world (p90 30 ms)**, half the naive cost: 20 worlds × p90 ≈ 0.6 s per decision on one thread. Usable inside the decision search on T2/T3 from trick 3, as amendment A4 assumes.
- **Trick 2 is now always solvable offline** (all 100 worlds, worst 29 s) but still seconds per world: still a teacher/post-mortem tool, not decision-time. Rollouts stay the plan for tricks 1–2.
- **Trick 1 is still out of reach**: not one world finished in 5 minutes (over 1.5 billion nodes each, table full at 3 GB; 4 worlds at once with a 3 GB cap ran out of the VM's 16 GB). The plan's "0.1–150 s per world at trick 1" remains unconfirmed.
- Still to do for the M2 gate: the same tables on a second CPU type (vast.ai EPYC).

## Second CPU type: AMD EPYC 7502 (vast.ai, 2026-10-14)
Run `m2-spike-epyc`: 64 logical cores (61 usable), same seeds, worlds and caps as the Xeon runs, each timing run alone (no concurrency). Raw outputs: `docs/m2-epyc/`. The equity table rebuilt on the box has the same hash (`64740dab…6808`). The cost was $0.17.

| Start of trick | Naive median / p90 | Fast median / p90 / max | Notes |
|---|---|---|---|
| 3 | 27.9 ms / 98 ms (no table) | 8.8 ms / 30 ms / 120 ms | 1,000 worlds, all solved |
| 2 | 8.7 s / 33.8 s (table on), 5 of 100 over 60 s | 2.8 s / 13.4 s / 26.4 s | fast: all 100 solved, 156 MB table used |
| 1 | — | 0 of 8 within 60 s | as on the Xeon |

**Identity on EPYC:** solver-identity on the real table, all 16 switch combinations and both tie orders: trick 5 (50,000 worlds), trick 4 (50,000) and trick 3 (1,000), with **0 mismatches**.

**Verdict:** on both CPU types the costs agree with the Xeon within normal machine variation. Exact solving is cheap from trick 3, takes a few seconds at trick 2, and is out of reach within a minute at trick 1. The M2 gate's "published costs on two CPU types" item is met.
