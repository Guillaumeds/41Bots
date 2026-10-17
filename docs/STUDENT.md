# The student network: the app's bots at tricks 1 and 2

Design of 2026-10-17 (branch `eng-student`). It answers PLAN amendment A7 ("soft targets", "T0/T1
student 2 × 128–256", the integer students of A8), PLAN §8 phases C and D, and
`docs/APP-ENGINE-CONTRACT.md` §4.1: **the bots that play are the student network plus exact search
from trick 3.** Everything not stated here follows the plan; where the plan is silent the choice
that keeps values byte-identical and stored data reusable is taken and written down (§10).

## 0. In plain words

The engine's best player for the first two tricks is slow: for each decision it imagines a few
hundred deals [complete guesses of the hidden cards consistent with what the seat has seen] and
searches each one (exact search to the end of the trick, then the trick network O_k), which takes
seconds on a big machine and far longer on a phone. The **student** is a small network [a learned
formula] that looks only at what one seat can see and gives every legal card its chance of the
money in a few milliseconds. It learns by copying the slow player [the **teacher**] on many
thousands of positions: for each position the teacher's value of every card is stored, deal by
deal, and the student is trained to give the same values. The app's bots then play tricks 1 and 2
with the student and tricks 3–5 with exact search, which is already fast.

The data is the expensive part, so it is stored once, in raw form, and only ever added to: a new
student design, a bigger network or a better training method reuses everything already paid for.

## 1. What the student is

- **Input:** the deciding seat's view only (`engine::view::SeatView`: its cards, every card
  played and by whom, the turned card and the robs, the burns it made, the scores, games and
  dealer, the card tracker's certain facts), plus one candidate action. Never another seat's
  cards (the plan's information firewall; the encoder reads a `SeatView` and nothing else).
- **Output, per candidate:** four logits [raw scores], turned into the four pairs' money chances by
  the oracle head's integer softmax (`nn::money::money_from_logits`): a fixed-point four-vector
  summing to exactly 2³², the same type as every teacher value (PLAN §5.3, CLAUDE.md "one objective
  everywhere").
- **The bot's choice:** the candidate with the highest value for its own pair; ties to the first
  candidate in the engine's candidate order (cards by index; a robber's (burn, play) pairs by burn
  then play), the engine's one tie rule (contract §3).
- **Candidates:** exactly the WASM API's (`search::pimc::candidates`, without the burn of the card
  just taken, which has no app action: `docs/WASM-API.md` §3). A robber's rob is valued together
  with the card it then plays (the joint decision of M3-SPEC §5.5). The dealer's burn before trick
  1 (trick 0) is not a student decision (the WASM API refuses it too, `not_built`).
- **Where it plays:** tricks 1 and 2 with all eight seats in play. Positions after a pair is turned
  never reach a network (contract §7: exact search among the seats still in play).
- **Architecture:** a plain MLP [layers of weighted sums with a clipped ReLU between them], one
  scorer shared by every candidate (the candidate's features are added into the first layer, so
  the view's part of the first layer is computed once per decision). PLAN §6.5 default and A7's
  sizes; the residual MLP stays a challenger under SPRT.
- **Inference:** integer-only (A8; the oracle's scheme `41bots-int-mlp-v1`: weights `i32`, sums in
  `i64`, ReLU clipped at 64, round-half-up shifts), so a bot's choice is byte-identical on every
  machine, thread count and in WASM. Floats exist only inside training.

Research basis (PLAN §2–4, A7): knowledge distillation into a small network (Hinton, Vinyals &
Dean 2015; policy distillation, Rusu et al., ICLR 2016); learning from a search expert's full
per-action values rather than its choice only (Expert Iteration, Anthony, Tian & Barber 2017:
about +50 Elo over chosen-action targets); MLPs for card games (DouZero 2021; the Skat
card-location networks); integer students (Stockfish NNUE).

## 2. The inputs (encoder `student-v1`, `crates/student/src/encode.rs`)

One encoder, in Rust, the single source of truth (PLAN §6.5): the WASM inference and the training
shards call the same function; Python never encodes. Seats are rotated so the deciding seat is 0
and its partner 4 (teams: own pair 0); cards are renumbered relative to trumps (§2.1). Binary
features are given as a list of active indices [the features that are "on"]; the dense block is
the oracle's equity block.

| Block | Features |
|---|---|
| Own hand | each card held |
| History | each card played this hand × the trick it was played in (1–5) × the seat that played it: who played what, when, and (with the leaders) in what order |
| Leaders | the seat that led each trick so far (the current one included) |
| Now | the current trick (1–5), the cards already in it (0–7: the seat's place), the trump suit |
| Turned cards | each card turned this hand (first, and the one after a turned Joker) and the seat that took it, if any |
| Burns | the cards this seat burned; every burn made this hand by seat and kind (`engine::view::BurnKind`) |
| Dealer | the dealer's seat |
| Match | each pair's score (one of the 15 reachable values), its bonus taken this game, its tricks this hand, its games won; the 5 of trumps played; the pair holding the best trump played so far |
| Card tracker | for every card the seat cannot see, each place it **cannot** be (the seven other hands, out of play): show-outs, trump limits, the ace-of-trumps facts, the robber's hand (`belief::Tracker`'s certain facts) |
| Ace facts | seats proved without the ace of trumps; the ace announced out of play |
| Dense (100) | the oracle's equity features (`nn::encode::dense_features`) for the score state at the start of the current trick, in the rotated pair order: each pair's money chance if the hand ended now, if a pair won the game now, and after each pair wins the next one and two tricks (the 5's holder counted only when it is the seat's own card) |
| Candidate | the card played; for a joint rob, the card burned; whether the card would win the trick as it stands |

### 2.1 Card numbering relative to trumps

A card is renumbered by its role for the hand's trump suit: the trumps first, strongest first (5,
J, Joker, A♥, ace, K, Q, then the numbers in their colour's order), then the other suit of the
trump's colour, then the two suits of the other colour (clubs before spades, diamonds before
hearts), each plain suit strongest first (`engine::rules::trump_rank`, `plain_rank`). For each
trump suit this is a one-to-one map of the 53 cards, and the trump suit is an input of its own,
so nothing is lost; "the second-best trump" is then one feature whatever the trump suit, so the
network learns it once instead of four times (PLAN §6.5 "Symmetries"; no symmetry between suits is
assumed, which would need a proof).

### 2.2 Versions

`ENCODER = "student-v1"` and an **encoder hash**: SHA-256 of the layout (every block's name,
offset and size) and of the encodings of 64 fixed positions (rules-bot self-play from fixed
seeds). Any change of what the encoder produces changes the hash, the engine's or the tracker's
included. Feature shards and networks record both; a network made for another encoder name or
hash is refused at load (WASM refusal `bad_file`).

## 3. The teacher and its stored record (the owner's data rule)

**Teacher** = PIMC over D deals drawn from the seat's view by the card tracker's uniform sampler
(`search::worlds::BeliefSource`), every candidate valued in every deal by the hybrid H_k
(`search::pimc::exact_world` with `solver::FrontierCut`: exact to the start of trick k, the oracle
O_k there): H_3 (O_3) for trick-2 decisions, H_2 (O_2) for trick-1 decisions once O_2 exists.
Deal `w` of a decision with seed `s` is `search::pimc::world_match(view, worlds, w, s)`: exactly the
deal the WASM API and the yardstick draw, so a stored value can be recomputed anywhere.

**Record** (`crates/student/src/record.rs`, format `41bots-student-teacher-v1`). A teacher shard is
two `.npy` files of fixed little-endian fields (PLAN §12.7) and a JSON sidecar:

- `NAME.npy`, one record per decision:
  - the position in raw form: the hand's start state (dealer, scores, bonus flags, games, hand
    and game index), the deck in deal order (53 bytes), every action of the hand before the
    decision (codes: play `c+1`, burn `65+c`), the deciding seat. Replaying them rebuilds the true
    deal (diagnostics) and the seat's exact `SeatView` (`Match::view`);
  - `pos_sha256` (the SHA-256 of those bytes: the position's identity), its source, generator
    seed and index, trick, place in the trick (0–7), match-state stratum (S1–S4), decision kind;
  - the candidates (code `burn·64 + play + 1`, as `search::teacher`), up to 32;
  - the deals: decision seed `s`, first deal index, count `n`, the tracker's world count;
  - the per-candidate means (the uniform mean over the record's deals; redundant, recomputed and
    checked by every reader) and per-candidate own-pair SD;
  - `values_at`: the record's first row in the values file, and `values_sha256`, the SHA-256 of
    its value rows.
- `NAME.values.npy`: `u64 [rows, 4]`: for each record, deal by deal, candidate by candidate, the
  four pairs' money chance in real pair order (units of 2⁻³², each row summing to 2³²).
- `NAME.npy.json`: the stamps: format, teacher (`hybrid` with the network's SHA-256 and cut, or
  `exact`), equity table SHA-256, `RULES_VERSION`, `VALUES_VERSION`, the binary's SHA-256, the
  sampler (`belief-uniform`), deals per decision, the generator's seed, source and settings, the
  test-set role if any, counts by trick/place/stratum, both files' SHA-256 and sizes.

The shard's key is the SHA-256 of `NAME.npy`; because every record carries the SHA-256 of its own
value rows, that one key covers the values too. Shards are named by their key, stored
content-addressed (`inputs/sha256/<key>` as the oracle labels), never rewritten; a later run lists
the stored keys and only adds new shards.

What the record makes possible without a new solve:
1. **A new encoder, network size or training method:** every input is rebuilt from the raw
   position (`student encode`).
2. **A card-reading model later** (PLAN §6.4): the deals are uniform draws from the hard facts, so
   a likelihood model re-weights them by importance weights (`p(deal | plays) / q(deal)`), deal by
   deal, from the stored values; the deal itself is regenerated from (view, seed, index).
3. **More deals on close calls:** a new shard with the same position, seed and later deal indices;
   `student encode` merges records of one position by deal index (refusing overlaps or gaps), so
   the mean is exactly the mean over all deals.
4. **The mean** is recomputed exactly (`MoneyChance::mix`, deal order).

Data is selected by filters on the stamps (trick, source, teacher, equity, rules); a shard whose
stamps differ from the run's is refused unless explicitly allowed (`--allow-stamp KEY`).

## 4. The generator (`search teacher --trick 1|2`)

```text
search teacher --trick 1|2 --count N --deals D --seed S --net FILE --equity FILE --out SHARD.npy
               [--source rules|mixed|random] [--threads T] [--first I] [--role train|test]
               [--start-hands 20000]
```

- **Positions (round 0, as the oracle labels' sources):** `rules`: the yardstick's generator
  (`search::measure::build_decision`): a rules-bot self-play hand from a start state of stratum
  S1–S4, a uniformly chosen decision node at that trick and place with two or more candidates;
  decision `i` from stratum `i mod 4 + 1` at place `(i div 4) mod 8`, so strata and places are
  equally represented. `mixed`: the same with every action before the decision a uniformly random
  legal one with probability 1/4 (exploration: positions the rules bot would not reach). `random`:
  every action random. Later rounds add the student's own games (DAgger) and the app's recorded
  hands (§6).
- **Values:** D deals per decision, every candidate in every deal by H_k; deals in index order.
- **Determinism:** decision `i` depends only on (seed, source, trick, `i`); decisions are made on
  any number of threads and written in order (`search::ordered`), so a shard is byte-identical for
  any thread count (tested). The values equal `search::pimc`'s for the same view, seed and deals
  (tested).
- **Resumable:** a shard is written as `NAME.npy.part` and renamed once whole; a run makes many
  small shards (parts of a few thousand decisions, `--first`), each a pure function of its
  settings, so a stopped run redoes only the unfinished part (the runner's chunk resume).
- Seed 1 (the golden sets') is refused; the test set has its own reserved seed (§7).

## 5. Training (`student encode`, `python/nn/fortyone_nn/student.py`)

- `student encode --teacher A.npy [B.npy …] --out F.npy`: records → one feature shard (per
  decision: the view's active features, the dense block, each candidate's features, each
  candidate's target four-vector in the rotated pair order, the teacher's own-pair SD per
  candidate, trick, place, stratum, source, the position key) and a sidecar (encoder name and
  hash, the teacher shards' keys and stamps). Deal re-weighting hook: `--deal-weights FILE`
  (one weight per stored deal) gives weighted means instead of plain ones.
- **Loss:** per candidate, the cross-entropy of the four-way softmax against the teacher's mean
  four-vector (the oracle head's loss, PLAN §6.5); the mean over a decision's candidates, then over
  decisions (a robber's 25 joint candidates do not outweigh a 3-card decision). This is
  AggreVaTe's reduction (Ross & Bagnell 2014) in its regression form: the targets are the expert's
  cost-to-go of every action, and the learner acts greedily on its estimates.
- **Validation:** split by position (a decision's position key), never by row; early stopping
  on the validation loss; the report gives the decision loss and agreement on validation too.
- **Warm start** (`--init` a previous student's float file, `--shrink-perturb λ`): Ash & Adams
  2020; never on the new data alone: every round trains on all the data selected so far (§6).
- **Keep rule:** a new student replaces the previous one only if it is no worse on the fixed test
  set (decision loss, `search student-check`), then by SPRT in duplicate tournaments (PLAN §9).
- Export: the quantiser of the oracle (`fortyone_nn.quantize`), the same format with encoder
  `student-v1`, `n_features`, the encoder hash and the training data's keys in the metadata.

## 6. The round loop

Round 0: one big base run of teacher records from the self-play sources (§4). Train a student.
Round r ≥ 1 (DAgger, Ross, Gordon & Bagnell 2011; Expert Iteration's loop):
1. Play games with the student at tricks 1–2 (exact search from trick 3) from stratified start
   states, and record the decisions it meets (the positions the student itself causes: the
   reason DAgger converges where plain copying drifts).
2. Label a sample of them with the teacher (new shards only).
3. Train on **all** shards selected so far (the old ones included: training on new data alone
   makes a network forget), warm-started from the previous student with shrink-and-perturb.
4. Measure on the fixed test set; keep the new student only if it is no worse; promote by SPRT.

Later sources: the app's recorded real hands (`hand_records`, the positions people meet), labelled
by the same teacher. A position's source is stamped on its record, so a run can weight or filter
sources.

## 7. The fixed test set and the measure

- **Test set:** decisions from the reserved seed `STUDENT_TEST_SEED` (`crates/student`), trick 1
  and trick 2 apart, valued with **1,024 deals** each (four times the training width, so the
  teacher's own noise on the test set is about half). Its sidecar carries `role: "test"`;
  `student encode` writes it apart and the trainer refuses to train on it. Never trained on, never
  regenerated (it changes only with a teacher upgrade, §9).
- **Measure** (`search student-check --net INT --teacher TEST.npy`): per decision, the student's
  choice and the teacher's best; **decision loss** = the teacher's mean value of its best
  candidate minus that of the student's choice, in the deciding pair's money chance; agreement =
  the share of decisions where the student's choice has the teacher's best value (ties count).
  Reported overall, by trick, by place in the trick (1–8) and by match-state stratum S1–S4, with a
  bootstrap 95% upper bound over decisions (1,000 replicates, seed 1, as the hybrid check).

## 8. Sizes, speed and cost

- **Sizes:** start with 256-128 and 512-256 hidden units (A7: T0/T1 student 2 × 128–256), chosen by
  the test-set decision loss; larger only if it pays. Inference cost at 512-256 is about 0.3M
  multiply-adds per candidate after the shared first layer: well under a millisecond a decision in
  WASM.
- **Teacher cost, trick 2** (H_3 with O_3, one VM thread, ms per deal by place: lead 73, 2nd 25,
  3rd 6.7, 4th 5.0, 5th 3.1, 6th 1.2, 7th 0.75, 8th 0.38; places equally represented): 14.4 ms a
  deal on average, **4.6 s of a thread per decision at D = 320**. On a 245-thread machine at
  $0.72/h that is about 190,000 decisions an hour, **about $4 per million decisions** (5.2 hours).
  The learning-curve plan's first 250k decisions: about 1.3 hours, about $1 plus the start-up.
- **Teacher cost, trick 1** (H_2 with O_2): not measured; estimate 2–6 times trick 2's ($8–25 per
  million), to be measured before any run.
- **Records:** about 30 KB a trick-2 decision at D = 320 (values dominate: 320 deals × about 3
  candidates × 32 bytes), so 250k decisions are about 8 GB raw, compressed for transfer.
- Measured on this VM (Stage 1): see §11.

## 9. A teacher upgrade (a better O_k, a new equity table)

The stored values stay valid for their stamps. A new O_3 (or O_2) changes the teacher: measure on a
2% sample of the stored decisions (re-valued with the new teacher, same deals) how many choices
change and by how much; relabel only the close calls (decisions whose best two candidates are
within the measured change) and the test set; train on the relabelled records where they exist.
A rules change or a new equity table invalidates every value (different stamps): everything is
relabelled; the raw positions are kept and reused.

## 10. Choices made here (the plan is silent)

1. Records are two `.npy` files plus a sidecar, keyed by the SHA-256 of the records file, which
   binds the values by a per-record SHA-256 (one key, byte-identical, readable from Python).
2. Values are stored as `u64 × 4` per (deal, candidate), the engine's own type, uncompressed in
   the record (compressed only in transit, `.npy.zst`), so a reader needs no codec and recomputes
   every mean exactly.
3. The candidate set is the WASM API's (no burn of the card just taken).
4. Cards are renumbered relative to trumps by a one-to-one map (§2.1); no suit symmetry is
   assumed.
5. The dense equity block reads the score state at the start of the current trick and counts the
   5 of trumps' holder only when it is the seat's own card (the seat cannot know more).
6. The loss weights decisions equally (mean over candidates, then decisions).
7. The test set: its own seed, 1,024 deals, role-stamped and refused by the trainer.
8. The WASM API takes `"method": "student"`; without it every request is valued as before (the
   coach keeps the hybrid PIMC). With it, tricks 1–2 are valued by the student and tricks 3–5 by
   exact search exactly as today.

## 11. Measurements and status

(Filled in as the stages are built.)
