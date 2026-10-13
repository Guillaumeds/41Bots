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
| Robbing (house rule) | The ace-of-trumps holder robs **at their own first turn** in trick 1, and **may burn any card except the ace of trumps**, including the card just taken | Engine (6.1) and card tracker (6.3). The dealer's turned-ace and turned-Joker robs stay "at once, before the first lead" (rules 22, 24). |
| Turned Joker + dealer holds the ace of trumps (rule 24) | The dealer **always robs the newly turned card at once** | Fixed in the engine |
| Next dealer after a game ends mid-hand | The deal **always moves one seat left**, as after any hand | Fixed in the engine |
| Where to build | A clean-slate project in a **new folder in this working directory** (`C:\Users\gdeswardt1.cnslt\41Bots\`), fully independent of the app | Nothing from the app is copied or referenced |
| Repository | **A new private GitHub repo (`Guillaumeds/41Bots`), not a branch of the app repo (`fivefingers`)** | A branch shares the app's history, workflows and settings, and branches are meant to merge back; this project never should. It has its own Rust + Python toolchain, CI and release rhythm. **Secrets:** create *new*, least-privilege keys for this repo instead of reusing the app's Cloudflare token: an R2 key scoped to this project's bucket, and a Workers deploy token later (M8). Rented machines only ever get the bucket-scoped key. The app later consumes versioned releases of the bot engine (WASM package and bot Worker API) like any dependency. |
| Compute operations | Windows laptop for development and overnight serial jobs. Rented vast.ai Linux machines for heavy runs: a different, cheapest machine each run, destroyed afterwards. CPUs near 100%, the GPU used where it helps, and all work finishing together. Data must move freely between Windows, Linux and Cloudflare. | Sections 11 and 12 |

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

| Paper | What it is | Key result | What 41 takes from it |
|---|---|---|---|
| **GIB**, Ginsberg, JAIR 2001 (the arXiv 1106.0669 link is the same paper) | The first expert-level bridge program. For each decision it deals 50–500 random versions of the hidden cards that fit what is known. It solves each one **double dummy** [as if all hands were face up] with exact search, then plays the card with the best total. Deals are weighted by how likely the opponents' actual plays would be, using **Bayes' rule** [the maths for updating odds with new evidence]. Its solver uses **partition search** [it remembers whole families of equivalent positions, not single positions]. | Partition search cut the search far beyond normal shortcuts: the branching factor fell from *b* to *b*^0.76. A full deal took about 18,000 search steps (about 1 second on a 2000-era PC). On Bridge Master problems GIB solved 100 versus Bridge Baron's 33, and it dominated the 1998 and 2000 championships. GIB states that "all competitive bridge programs switched" to its method. **Weakness:** it assumes it will know the hidden cards later, so it undervalues safe plays and never plays to gain information. Its declarer fix (achievable sets plus squeaky-wheel optimisation) added only about 0.1 IMP per deal [IMP = bridge's standard scoring unit], and in defence plain Monte Carlo was preferred. | The core method; an exact solver with partition search; weighting guesses by how likely the observed plays are |
| **αμ**, Cazenave & Ventos 2019 | A search that makes "my side" play the same card in every world it cannot tell apart. It assumes the opponents can see everything. | It fixes **strategy fusion** [sampling's flaw: it assumes you can play differently in worlds you cannot actually tell apart] and non-locality. On close 3NT deals, with 40 sampled worlds, the win rate rose from 62.4% (PIMC) to 63.2% (αμ looking 3 moves ahead); with 20 worlds it rose from 60.2% to 62.0%. Cost rises steeply with look-ahead. The paper also confirms GIB and WBridge5 use **PIMC** [Perfect-Information Monte Carlo, the GIB method above]. | Optional late-hand experiment only. In 41 *nobody* sees the hidden cards (opponents included), so our main fix for strategy fusion is recursive rollouts (next row). |
| **Furtak**, PhD thesis, Alberta 2013 (the DOI link) | The search engine of the Skat program Kermit | Solver improvements (symmetries, move ordering, lookup tables) made the exact solver 22–58× faster. Plain PIMC **stopped improving after about 160 sampled worlds**. **Recursive Monte Carlo** (IIMC) finishes each sampled world with a realistic player who sees only their own cards, not the all-seeing solver. It beat Kermit at Kermit's best (about +7 points per game as soloist) and **kept improving with more hardware**, while PIMC did not. It also clearly beat an information-set MCTS bot and was harder to exploit. | **Our main thinking method**, plus solver speed-ups |
| **Setrograde**, Stone, Sturtevant & Schaeffer (arXiv 2024; IJCAI 2025) | Solves bridge endgames backwards, working on sets of positions that share the same value | 1,000× fewer operations and 10,000× fewer stored entries. Every 7-trick bridge ending was solved, the largest endgame database in any game. The authors note that partition search still underpins state-of-the-art bridge play. | Use set-based reasoning inside our solver. We do **not** need a precomputed 41 database: in 41 the cost is the number of hidden-card guesses, not the depth of a hand. |
| **Sturtevant**, IJCAI 2003 | How to search games with 3 or more competing sides | With his pruning, **max^n** [every side picks what is best for itself] beat **paranoid** [assume all other sides gang up on you] in 3-player, all-cards-visible versions of Hearts (8.01 vs 8.99 points, lower is better) and Spades (5.72 vs 5.61, higher is better), even though paranoid searched deeper | 41 has four sides, so our solver uses max^n with **last-branch and speculative pruning** [safely skip moves that cannot change the result; this works because the four teams' winning chances always add up to 100%] |

---

## 3. What the championship winners actually did

| Winner | How it chose cards |
|---|---|
| 1997 Bridge Baron | Hand-written human techniques plus planning. It was overtaken once search-based programs arrived. |
| 1998–2000 GIB | PIMC with a double-dummy solver and inference |
| 2001–2019 Jack, WBridge5, Shark, Micro Bridge | The same family: GIB reports that by 2001 every competitive program had switched its card play to this approach. WBridge5 is confirmed to use PIMC (αμ paper, 2019). Its 2016–2018 titles came after **seed boosting** [choosing the random-number settings that make its sampling play best] (Ventos et al. 2017). |
| 2023 Q-Plus | Hans Leber's long-running German program. Its card-play method is not published in the sources found. |
| 2022 NooK (NukkAI) | Beat 8 world champions at declarer play (67 of 80 sets) using a hybrid of rules and search. The details are unpublished. |
| 2022 onward: Ben, then BBO's GIBBO | Neural networks. BBO's newest card play no longer relies on double dummy. Instead it will "simulate playing a card, then let the neural network play the rest of the hand", and it estimates how likely each hand is from the plays seen. |

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
| Only 5 tricks | Exact search per guessed deal is cheap | We can afford thousands of guesses and recursive search. (Skat needed 32 cores for this.) |
| 13–14 cards never revealed (the 12 undealt cards plus burned cards) | Less is revealed during play than in bridge | Inference matters more. In checkpoint M3 we measure 41's leaf correlation and disambiguation (Long et al.). |
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
    - your house rule: the ace-holder robs at their own first turn and may burn any card except the ace of trumps
    - the dealer's turned-ace and turned-Joker sequences happen at once, including the dealer robbing the newly turned card when they hold the ace of trumps (rule 24, always)
  - after a game ends mid-hand, the deal always moves one seat left
  - burned cards stay hidden
  - the best-trump bonus: 11 points the first time a team takes it in a game, 10 after that
  - the game ends the moment a team reaches 41, even mid-hand
  - a jink (all 5 tricks to one team) wins the game; the match is best of 3; the deal moves left
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
- **No rob:** if trick 1 ends with no rob, the ace of trumps is in the undealt pack.
- **Voids:** if a player plays a plain card of another suit when a plain suit was led, they hold none of the led suit. Trumping does *not* reveal a void (rule 14).
- **Trump limits:** if a player plays a non-trump when a trump was led, they hold no trumps except top trumps *higher than the led trump* (rules 16–18).
- **Out of play:** the 12 undealt cards and any burned cards are unknown and never revealed.

The output is, for every unseen card, the list of places it could be, plus how many cards each place holds.

### 6.4 Belief: who holds what

This combines three proven pieces:

1. **Belief head** of the network: for each unseen card, the probability that it is in each other player's hand, in the undealt pack, or burned. It is trained on self-play, where the true locations are known.
   - Solinas et al. 2019 trained the same thing on 20 million human Skat games, with a significant strength gain.
   - Learned Belief Search (Hu et al. 2021) did it in Hanabi.
2. **Constraint-respecting deal generator.** It builds complete **worlds** [one specific guess of where every hidden card is] that obey every hard fact, guided by the belief head (GIB used a constrained deal generator).
   - When only a few worlds remain, they are all listed exactly instead of sampled (Rebstock et al. 2019).
   - For hard cases, the MCMC sampler of Solinas et al. 2023 is the fallback [MCMC: a random-walk method for sampling from tightly constrained possibilities].
3. **Policy-based inference.** Each world is scored by how likely the other players' *actual* plays would be if that world were true (for example: "if West really held the 5 of trumps here, would they have played the 7?").
   - Rebstock et al. 2019 beat the previous best Skat inference this way. GIB used the same idea.
   - The weight of a world is the product of the policy-head probabilities of everyone's observed plays, divided by how likely the generator was to propose that world. This is **importance sampling** [a standard statistical correction: sample from a convenient guess, but still get the right answer].

**Engineering choices:**
- The worlds are kept as a **particle filter** [a running set of weighted worlds updated after every card, instead of rebuilt from scratch]. Impossible worlds are dropped, and new ones are drawn when too few good ones remain. This is standard (POMCP, Silver & Veness 2010) and saves compute at no quality cost.
- **Robustness, which matters even more with human players:** a model's opinion never drives a world's weight to exactly zero. A small floor remains, and the weighting mixes in a simpler "human-like" play model until phase E provides a real one. SPARTA found that beliefs collapse when partners deviate and this protection is missing.

### 6.5 The neural networks ("the network")

- **One shared network for all 8 seats.** Every seat follows the same rules, so each hand gives 8 seats' worth of training examples. The input is rotated so "me" is always seat 0 and my partner is seat 4.
- **Inputs:** a complete, fixed-size description of everything this seat has seen. For each of the 53 cards: in my hand / turned / played, by whom, in which trick, in what order. Plus robbing and burning facts, card-tracker facts, and the match state: all four scores, games won, which teams have already taken a best-trump bonus this game, and the dealer seat. Also, recomputed **after every trick**, each team's current chance of the money and whether it can grab the money in this hand (points needed versus tricks and bonus still available). This makes "public enemy number one" explicit to the network. Because every played card records who played it and when, nothing about the history is lost.
  - **The input encoder is written once, in Rust.** Python gets already-encoded inputs from data files or from the Rust bindings, so training and play can never disagree.
- **Design:**
  - **Default:** an **MLP** [the simplest neural-network design: layers of weighted sums]. It is cheap on CPUs, phones and serverless, as in DouZero and the Skat card-location networks.
  - **Challenger:** a small transformer, as in GO-MCTS.
  - The tournament keeps whichever plays better for the same compute.
- **Outputs ("heads"):**
  1. **Policy head** [probabilities over the legal actions: which card to play or burn]. This is the **blueprint** every bot shares.
  2. **Value head** [the estimated chance that each of the four teams wins the match, and so the money, from this seat's view]. Four numbers that add to 1.
  3. **Belief head** (section 6.4).
  4. **Oracle value head**, used *in training only*: the same estimate, but given all cards face up. It grades decisions more accurately so learning is faster (PerfectDou's "perfect-training, imperfect-execution"; Suphx's oracle guiding). It is never used to choose a card in a real game.
  5. **Public money-chance head:** each team's chance of the money from public information only, re-estimated after every trick (6.6). Its per-trick timeline is also written into every game record, so you can see when each team became public enemy number one.
- **Teacher and student sizes:** the strongest "teacher" network is trained first. Smaller "student" networks for phones and serverless are then trained to copy it. This is **knowledge distillation** [Hinton et al. 2015: a big network teaches a small one], which keeps as much strength as possible at each size.
- **Running the network everywhere** (section 11):
  - The MLP is computed by a small inference routine inside the shared Rust core. A plain, exact "reference" version is used for tests. Fast versions use AVX2/AVX-512 on servers and WASM SIMD on Cloudflare and phones [**SIMD**: processor instructions that do many multiplications at once], and each is checked against the reference.
  - Weights travel as **safetensors** files [a simple, safe, standard format for network weights].
  - The transformer challenger would run through **tract** [a pure-Rust engine that runs standard **ONNX** network files, an exchange format for trained networks], on every tier.
- **Symmetries:** interchangeable suits (for example clubs and spades when neither is trumps) are **canonicalised** [rewritten into one standard form, so each situation is learned once] or used as **data augmentation** [extra training on mirrored copies]. Red/black ordering and the A♥ rule are always respected.

### 6.6 Match-equity model ("what is each team's chance of the money?")

- **Input:** all four scores, games won, which teams already took a best-trump bonus this game (11 vs 10 points), and the dealer seat. **Output:** each team's chance of winning the match, which by section 5.3 is the same thing as its expected money.
- **Accurate even for small chances:** it is trained on the log-odds scale [a scale that stretches out very small and very large probabilities so they are learned precisely], and checked for calibration in those ranges, because blocking and kingmaking decisions depend on small but real differences.
- **Proven precedents:**
  - backgammon's **match-equity tables** [standard for decades: the chance of winning a match from any score]
  - Suphx's global reward prediction, which was central to its Mahjong success
- **How it is built:** learned from self-play match results, cross-checked by an exact **dynamic-programming** [solving from the end backwards, step by step] calculation over hand-outcome statistics, and refreshed every training generation. It is small enough to ship as a lookup table on every tier.
- **Trick-by-trick money chances.** Between hands the match-equity model gives each team's chance of the money. *Within* a hand, the chances are re-estimated after **every trick** by two network heads:
  - a **public money-chance head**, which uses only what everyone can see (scores after this trick, tricks left in the hand, who leads, the best trump played so far and which team holds it, cards gone, games won, bonus flags, dealer). Every bot shares its view of who is "public enemy number one" and how close they are to grabbing the money this hand.
  - the training-only **oracle head**, which sees all cards, for the most accurate training signal

  Both are trained on actual match results and are tied to the match-equity model at hand boundaries, so the numbers agree everywhere.
- **Three uses:**
  1. scoring the end of every simulated line inside the search: end of hand, or a game or match ending mid-hand at the exact trick
  2. the training reward: the change in each team's chance of the money **after every trick**. This is **reward shaping** [extra, earlier feedback that provably does not change which strategy is best].
  3. making bots score-aware: block a team on 36, take risks when far behind, protect a lead

### 6.7 Exact "all cards face up" solver

Given one world (every card known), it computes the result of perfect play by all four teams, scored as **each team's chance of the money**. Every finished line (the end of the hand, or a game or match ending mid-hand) is valued by the match-equity model.

- **Four-sided search:** max^n, with each team maximising its own chance of the money, plus Sturtevant's last-branch and speculative pruning. The four chances always add to 100%, which is exactly what his pruning needs.
- **Transposition table** [a memory of positions already solved, so a position reached by a different play order is not solved twice], indexed with **Zobrist hashing** [a fast fingerprint of a position].
- **Partition search and equivalent cards** [cards that provably make no difference are treated as one; for example, two cards in the same hand become equal once every card ranked between them is gone]. Precedents: Ginsberg 2001; Furtak 2013 (22–58× in Skat); Setrograde 2024/25.
- **Move ordering** [try the most promising card first, so more of the search can be skipped]: Furtak's "fastest-cut-first" ordering, and killer moves.
- **Ties:** when a team does not care which way a move goes, the policy head's preference decides. This is the Prob-max^n idea (Sturtevant, Zinkevich & Bowling 2006).

**Roles:**
1. the "teacher" for bootstrap training
2. a baseline opponent in tournaments
3. a **post-mortem checker** [after a game, it finds every play that gave away chance of the money, judged with all cards visible; GIB and Furtak used this kind of check, counted in tricks or wins, to measure error rates]
4. a candidate engine for CPU-only tiers
5. exact play in the last tricks, but only if tournaments show it helps

### 6.8 How a bot thinks before each card (decision-time search)

**Main engine: recursive Monte Carlo with neural rollouts** [a rollout means "play the card, then simulate the rest of the hand"]:

1. Take the current weighted worlds from the belief. The number is set per tier by tournaments: thousands on servers, fewer on phones.
2. For each legal card (at most 5) and each world: play that card, then let all 8 seats finish the hand using the policy head. Each seat sees only its own information in that world; nobody peeks. Score the end with the match-equity model.
3. Play the card with the best average chance of the money for my team.

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
- **Exact endgame:** in late tricks, list every remaining world instead of sampling.
- **Forced moves:** a single legal card, or interchangeable cards, needs no search.
- **Truncated rollouts** for small tiers: stop the simulation early and let the value head estimate the rest (the standard AlphaZero-style trade). It is used only where tournaments show it beats more worlds at that tier's budget.

**Alternatives kept in the tournament pool, so data decides:**
- plain PIMC with the exact solver (the GIB/Kermit style)
- a hybrid that uses PIMC early in the hand and recursive search late (Furtak's suggestion)
- network-only play, with no search (DouZero-style; GO-MCTS showed this alone was strong in Hearts)
- an upgrade path: **MCTS** over my own future decisions (GO-MCTS / information-set MCTS) [MCTS = a search that grows a tree of the most promising move sequences]

### 6.9 The burn decision

- Burn choices are evaluated by the same search: up to 6 cards, never the ace of trumps.
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
- **Fan-out option for serverless:** one decision's worlds can be split across several parallel Worker calls, so latency drops while total cost stays the same.
- **Strength–cost curves:** the tournament system (section 9) measures each tier's strength against the champion at several budgets. You then see exactly what a server buys before paying for one.
- **Difficulty levels** come for free: smaller budgets, or blending toward the human-like model from phase E.

---

## 8. Training pipeline

**Phase A: bootstrap (no human data needed).**
- The PIMC teacher (exact solver plus constrained sampling) plays tens of millions of hands across many score situations.
- The network is trained by **supervised learning** [learning to copy examples]:
  - the policy head copies the teacher's choices
  - the value and oracle heads learn the money outcome: which team eventually won the match. Hands are played inside full matches; for a hand not followed to the end of its match, the match-equity model's estimate is the target.
  - the belief head learns the true card locations
- Precedent: Skat policy and inference networks were trained this way from human games. Here the "expert" is the teacher.
- The first match-equity model is built from these games.

**Phase B: self-play reinforcement learning.**
- **Algorithm:** **PPO** [a standard, stable method that improves the strategy in small, safe steps] with the training-only oracle judge (PerfectDou; Suphx).
- DouZero's simpler **Deep Monte-Carlo** method [learn each move's value directly from final game results] is kept as the alternative; tournaments decide.
- **Reward:** the change in each team's chance of the money **after every trick** (measured by the oracle head), plus the final money result of the match. A game or a trick is never rewarded for its own sake (5.3).
  - This is **potential-based reward shaping** [Ng, Harada & Russell 1999: giving feedback as the change in an estimated value is proven not to change which strategy is best; it only makes learning faster and sharper].
  - The bot learns precisely which tricks, and which cards inside them, swung the money, including stopping a new leader the moment it takes first place.
- **League** [train against a pool of past versions and special "exploiter" bots, not only the latest self, to avoid going in circles or growing blind spots]: AlphaStar; GO-MCTS's population self-play.

**Phase C: search distillation.** This is **Expert Iteration**, the AlphaZero loop [the slow, strong search player generates games; the fast network learns to predict what the search chose; repeat, so a better network makes a better search].
- Proven by AlphaZero (chess, Go, shogi) and ExIt.
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
- **Duplicate deals.** The same seeded decks are replayed with the teams rotated through the seats, so luck of the cards cancels out. Example: one candidate team against three baseline teams, rotated through all four team positions on identical decks; or two candidate teams and two baseline teams, alternating and then swapped.
- **SPRT gate** [the sequential statistical test top chess engines use (Stockfish's Fishtest): keep playing only until it is statistically clear whether the new version wins more money, then stop]. Every change must pass it. This saves large numbers of games without weakening the conclusion.
- **AIVAT-style luck removal** [the value head estimates how much of each result was luck and subtracts it; unbiased]. In poker it cut the games needed 44× (Burch et al. 2018).
- **Elo ratings** [chess-style strength numbers] across all versions and tiers, for long-term tracking.
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
| Partition search, equivalent cards, transposition table, move ordering, speculative pruning | Solver work, by orders of magnitude | Skipped work provably cannot change the answer |
| Symmetry canonicalisation (seat rotation, interchangeable suits) | Training data and solver memory | Equivalent situations are learned and solved once |
| Particle-filter reuse | Belief rebuilding | The same posterior, updated incrementally |
| **Batching** [sending many world simulations through the network together] and incremental evaluation along a rollout | Rollout cost | The same network outputs, computed once |
| Common random numbers, adaptive sampling, exact late enumeration | Number of simulations | The same decision, with less noise per simulation |
| Duplicate deals, SPRT, AIVAT | Tournament games, often 10–40× | The same statistical confidence |
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
| Memory allocator | mimalloc | 0.1.52 (2026-05-22) | A faster allocator for many threads. It needs a C compiler: MSVC on Windows, gcc in Docker. |
| CPU-specific speed-ups | multiversion | 0.9.0 (2026-09-06) | Runtime choice of AVX2/AVX-512 versions of the hot loops, so one binary runs on any rented x86 CPU |
| Network maths | **Own MLP routine** using matrixmultiply | 0.3.11 (2026-07-14) | matrixmultiply has AVX-512 kernels and a WASM SIMD (simd128) matrix-multiply kernel. Our plain reference routine is used for tests. |
| Network weights | safetensors (Rust and Python) | 0.8.0 (2026-06-09) | One small, safe file per model, readable everywhere |
| Transformer challenger | tract (`=0.23.8`, exact pin) | 0.23.8 (2026-09-21) | Pure Rust, runs ONNX on native and WASM with SIMD kernels. 0.23 renamed many APIs, hence the exact pin. |
| Native ONNX cross-check (testing only) | ort | 2.0.0-rc.14 (2026-10-12) | Still a release candidate, and it doesn't run in WASM. Used only as a second opinion on native builds. |
| Deterministic maths | libm | 0.2.16 | Same `exp`/`tanh` results on every platform. Rust's built-in versions may differ by platform. |
| Random numbers | **Own pinned generator** (SplitMix64 seeding plus xoshiro256** or ChaCha8), own shuffle | — | The `rand` crate (0.10.3) explicitly makes no promise of identical sequences across versions. A tiny in-house generator with golden-vector tests gives replayable games forever. |
| Data (de)serialisation | serde 1.0.229, serde_json 1.0.154 | 2026-07 / 2026-10 | Game records and protocol messages as JSON |
| Compression | zstd 0.14.2 (native); ruzstd 0.9.1 (pure-Rust decoder for WASM) | 2026-10-10 / 2026-10-11 | Python 3.14 reads zstd natively with `compression.zstd` |
| Analysis exports (optional) | arrow / parquet | 60.0.0 (2026-09-15) | Only for ad-hoc analysis tables |
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
| Env and package manager | **uv** | 0.13.0 (2026-10-09) | One cross-platform `uv.lock` for Windows and Linux; it also installs Python itself. uv 0.13 defaults to 3.15, so 3.14 is pinned in `.python-version`. |
| Deep learning | **PyTorch 2.14.1 exactly** | 2026-09-30 | **Deliberately not 2.15** (due ~2026-10-28), which drops Pascal and Volta GPUs (GTX 1080 Ti, P40, P100, V100), common cheap vast.ai cards. |
| — GPU build (vast.ai default) | `torch==2.14.1+cu126` | — | Supports GPU generations Maxwell to Hopper (compute capability 5.0–9.0). Needs a host driver that supports CUDA 12.6 (filter `cuda_vers>=12.6`). **No RTX 50-series (Blackwell)**; filter `compute_cap<1000`. [**CUDA**: NVIDIA's GPU software layer; each GPU generation needs a matching build] |
| — Newest-GPU build (optional) | `torch==2.14.1` for CUDA 13.0 | — | Only for Blackwell hosts (driver ≥580) |
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
| **Docker image** [a packaged, ready-to-run copy of the whole software environment] | A multi-stage build on a slim `nvidia/cuda:12.6.3-base-ubuntu24.04` base, with Rust binaries (x86-64-v2 and v3 builds), the uv-locked Python env (cu126), rclone and the vastai CLI | Hosted on **GHCR** (GitHub's container registry; storage and bandwidth "currently free"). vast.ai doesn't bill while an image downloads. Tagged with date and commit, never `latest`. |
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
  | `compute_cap>=600 compute_cap<1000 cuda_vers>=12.6` | Matches our PyTorch build |
  | `reliability>0.98` | Avoids flaky hosts |
  | low `inet_up_cost` / `inet_down_cost` | Hosts charge per GB, and we upload results |

- **Rank by price per unit of work, not price per hour:**
  - At first: `dph / cpu_cores_effective` [dph = dollars per hour].
  - Each run starts with a 30–60 second **micro-benchmark** [a quick speed test] and records "games per second per dollar" for that CPU model in R2.
  - The launcher learns which CPU models really give the best value.
- **Default rental type:** on-demand. **Interruptible** [cheaper bid machines that can be stopped when outbid] is only for long throughput runs. If one is stopped, the launcher rents a new machine that resumes from the last checkpoint, instead of waiting.

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

- **Every 10–15 minutes:** completed shards and checkpoints are uploaded to R2 as large bundles, using `rclone copy --immutable` (never overwriting). Then the done-list is updated. Large bundles keep R2 operation costs low and respect its "one write per second per key" limit.
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
  reference/     small, slow, obviously-correct second rules implementation (testing only)
  tests/         one test per numbered rule, golden vectors, differential, leakage, replay tests
  .github/workflows/   CI: Windows + Linux + WASM builds/tests, Docker image to GHCR
```

| Checkpoint | Delivers | Gate to pass |
|---|---|---|
| **M0** | Private repo `Guillaumeds/41Bots` (created by the owner). Pinned toolchains; CI on Windows, Linux and WASM. Rules engine with its own RNG, including match and money scoring; reference implementation. | Every numbered rule has passing tests. RNG golden vectors are identical on all three platforms. Zero engine-vs-reference disagreements over 10 million random games. WASM results equal native. |
| **M1** | Table manager, observation filter, protocol, records, random and rules bots, duplicate arena with SPRT. **Run operations v1:** runner, Docker image, R2 layout, launcher. | Duplicate matches replay exactly. Leakage tests pass. A first small vast.ai run goes end to end (rent → benchmark → work → checkpoint → DONE → self-destroy) with CPU use near 100%. |
| **M2** | Exact solver | Matches brute force on random small positions. Results are identical with pruning on and off. Speed benchmarks recorded per CPU type. |
| **M3** | Card tracker, deal generator, PIMC teacher; measurement of 41's PIMC-suitability properties | PIMC beats the rules bot decisively. Measured properties are reported. |
| **M4** | Network, bootstrap training (teacher-data run), match-equity model v1, inference on all targets | The network alone plays close to the teacher. Belief calibration is good. Rust inference matches PyTorch within tolerance on native and WASM. |
| **M5** | Recursive rollout search with all safeguards | Beats the PIMC teacher on SPRT |
| **M6** | Self-play reinforcement learning with league (RL runs) | Beats the M5 champion |
| **M7** | Distillation loop and tier students | Each generation beats the previous one. Strength–cost curves are published for T0–T3. |
| **M8** | Cloudflare Durable Object table host plus T1 bot Worker; server bot (T2/T3); latency and load tests | Meets the agreed time per card under load. The fairness rule is enforced. |
| **M9** | Phase E: cross-play suite now; human model and piKL once human games exist | Better cross-play and human-partner scores than M7, without losing strength against bots |

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
  - The money-chance type is the only score the solver, search, training targets and tournament ranking can produce. This is checked when the code is built, plus a test that every evaluator's four chances add to 1.
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
  - CPU-only rentals are undocumented, so we rent cheap-GPU machines with many cores and keep their GPU busy with training.
- **Rust 1.99.1** is due on 15 Oct. Until then 1.99.0 is used, and the Python bindings are re-tested after upgrading.
- **R2 costs:** many small files drive up operation costs, which is why bundles are large.
- **Human data and privacy:** human games are collected only with player consent and anonymised. Phase E quality grows with the amount of data.
- **Kingmaking:** while a match is live, every team keeps a real (if small) chance of the money, so maximising its own chance gives sensible blocking and prolonging behaviour (5.3). This relies on the match-equity model being accurate for small chances (6.6). Exact indifference is rare; when it happens, the policy head's preference breaks the tie (6.7). Validated in scenario tests and tournaments.
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
