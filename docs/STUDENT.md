# The student network: the app's bots at tricks 1 and 2

Design of 2026-10-17 (branch `eng-student`). It answers PLAN amendment A7 ("soft targets", "T0/T1
student 2 × 128–256", the integer students of A8), PLAN §8 phases C and D, and
`docs/APP-ENGINE-CONTRACT.md` §4.1: **the bots that play are the student network plus exact search
from trick 3.** Everything not stated here follows the plan; where the plan is silent the choice
that keeps values byte-identical and stored data reusable is taken and written down (§10).

## 0. In plain words

The engine's best player for the first two tricks is slow: for each decision it imagines a few
hundred deals [complete guesses of the hidden cards consistent with what the seat has seen] and
searches each one (exact search to the end of the current trick, then a network that values the
rest of the hand), which takes
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
- **Candidates:** exactly the WASM API's (`search::student_teacher::app_candidates`:
  `search::pimc::candidates` without the burn of the card just taken, which has no app action:
  `docs/WASM-API.md` §3). A robber's rob is valued together
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

### 3.1 The playout teacher (2026-10-25; PLAN §6.8, a recursive rollout at depth one)

The search teachers above value each deal with every card face up (exact search, then a network
that copies the exact solver). The review of 2026-10-25 (EXPERIMENTS I1-I2) measured what that
costs on one decision class: the live S1 leads the J of trumps without the 5 expecting +0.6% of
the money and loses 0.45% against players who cash the 5, because in a face-up deal the
5-holder is never surprised. The **playout teacher** values each deal by playing it with
hidden-card players instead:

1. the candidate is played (a robber's joint burn-then-play as one candidate, as always);
2. then every seat to act, the decider included, chooses from **its own `SeatView`** with the
   student network of its trick: S1 at trick 1, S2 at trick 2 (`search::pimc::student_candidate`:
   the app's candidates, `net.evaluate`, the team's best, ties to the first candidate; exactly the
   app's `method: "student"` decision); a seat with one legal action plays it without the
   network;
3. at the start of trick `cut` (default 3) the deal, now fully known, is solved exactly to the end
   of the hand (`search::pimc::exact_value`: the root mover's best under the solver's tie rule); a
   hand that ended first is valued by the leaf rule. `cut` 6 plays every trick out; `cut` at the
   decision's own trick plays nothing out and equals the exact teacher byte for byte (tested).

The values are averaged over the same deals for every candidate, in deal order, with
`MoneyChance::mix`, so a shard is byte-identical for any thread count (tested). Furtak and Buro
(2013) and Bouzy, Rimbaud and Ventos (2020) are the precedents (RESEARCH §7, §15); the inner
players are today's students, lean included, so a second round replays the deals with the
retrained ones.

- Rust: `search::pimc::{Students, student_candidate, playout_value, playout_world}`,
  `search::student_teacher::Teacher::Playout { students, cut, sha1, sha2 }`, stamp
  `playout:<S1 sha256>:<S2 sha256>:cut<k>` in the sidecar (`Method::PlayoutPimc` in records).
- CLI: `search teacher --trick 1|2 ... --playout --student1 S1 --student2 S2 [--playout-cut 3]`
  (no `--net`).
- Runner (`student_teacher` streams): `"teacher": "playout"`, `"student1"`, `"student2"` (manifest
  input names of the student files), `"playout_cut"` (3); no `net_input`. Shards are named
  `t<trick>-<role>-<source>-s<seed>-d<deals>-pl<cut>-p<part>.npy`. The first run is
  `docs/vast/playout-labels.manifest.json` (launched 2026-10-25 07:28 UTC on the owner's Thailand
  box, run id `playout-labels-1`); the students' retraining on its labels is
  `docs/vast/student-playout-train.manifest.json` (S1' and S2', decision-aware loss, three seeds,
  judged on run 1's own playout test sets), and the money gate `docs/vast/arena-students.manifest.json`
  (§6.1, branch `arena-app-bot`).
- Cost per deal (estimate, to be measured in that run): up to 16 student calls and one exact solve
  from trick 3, tens of milliseconds, against 0.2-0.6 s for an H_2 deal natively; so 256-512 deals
  a decision instead of 64.
- Judge: never copper or the exact golden sets (they are labelled face up); the paired playout
  yardstick (EXPERIMENTS I1) and the arena in money.

## 4. The generator (`search teacher --trick 1|2`)

```text
search teacher --trick 1|2 --count N --deals D --seed S --net WEIGHTS|exact --equity TABLE
               --out SHARD.npy [--source rules|mixed|random|student] [--first I] [--deal-from W]
               [--role train|test] [--threads T] [--tt-mb 64] [--start-hands 20000]
               [--progress FILE]
```

- **Positions (round 0, as the oracle labels' sources):** `rules`: the yardstick's generator
  (`search::measure::build_decision`): a rules-bot self-play hand from a start state of stratum
  S1–S4, a uniformly chosen decision node at that trick and place with two or more candidates;
  decision `i` from stratum `i mod 4 + 1` at place `(i div 4) mod 8`, so strata and places are
  equally represented. `mixed`: the same with every action before the decision a uniformly random
  legal one with probability 1/4 (exploration: positions the rules bot would not reach). `random`:
  every action random. `student` (2026-10-25, with the playout teacher only): the students play
  every action before the decision from their own view (S1 at trick 1, S2 at trick 2, a robber's
  burn and first card as one choice; the dealer's burn by `rules-v1`), so the positions are the
  ones the students themselves reach (DAgger, §6 step 1; `student_teacher::Inner`); the decision's
  own seed sequence is unchanged, so a `student` set and a `rules` set at the same seed share
  their start states and decks and differ only where the students play otherwise. Later: the app's
  recorded hands (§6).
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
  (one weight per stored deal) gives weighted means instead of plain ones. `--max-deals N`
  builds every target from the decision's first N deals only: exactly the feature shard a
  teacher run with `--deals N` and the same seeds would give (tested byte for byte), so how many
  deals a student needs is measured on stored data; refused when a decision holds fewer than N
  deals, and never applied to the test set (the measure always uses all its deals).
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
   reason DAgger converges where plain copying drifts). Built 2026-10-25 as the generator's
   source `student` (§4), for the playout teacher: round 2's labels take `"source": "student"`
   beside `rules` and `mixed`.
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
- **Honest measures** (`--split-half`, `--top-k K`): the floor of the test set, the unbiased gap to
  a 512-deal teacher and the loss of a top-K shortlist (§14).

**Against exact values (2026-10-20).** `search student-check --golden-decisions D --golden-worlds
W` scores a trick-2 student on the stored trick-2 golden set (inputs `sha256:635e9aef…` decisions,
`sha256:6ddc59b1…` worlds: 2,000 decisions x 320 worlds, every candidate's exact value in every
world) instead of teacher shards, with the same report (loss by place and score class, split-half
floor, unbiased gap, top-K). The values are exact, so this measures the student against the truth,
not against its teacher; it is how a student trained on G v2 labels (`student-t2-gv2`) is scored,
with no new test set bought. A random 16-8 network on the build VM: 1.6 s for all 2,000 decisions.

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
- **Teacher cost, trick 1** (H_2 with O_2; measured with random networks of O_2's possible sizes,
  §11): 0.17 s of a thread a deal with a 1024-512-128 O_2, 0.57 s with a 2048-1024-256 one, 11 to
  39 times trick 2's: **53–182 s a decision at D = 320, about $40–150 per million decisions** on
  the same machine. The exact search of trick 1 reaches many more frontier positions than trick 2's,
  and each costs a network evaluation, so O_2's width sets the price: trick-1 data is priced again
  once O_2 exists.
- **Records:** about 37 KB a trick-2 decision at D = 320 (measured: 35 KB of values, 320 deals ×
  3.45 candidates × 32 bytes, and a 1.6 KB record), so 250k decisions are about 9.2 GB raw; the
  feature shard of the same decisions about 0.7 GB.
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
9. A rented run adds data by parts of one generator stream (`first_part`, `skip_parts` in the
   runner's `student_teacher` chunk): a part is a pure function of its settings, so a lost
   machine's part is remade byte for byte and a later run never remakes a stored one; stored
   shards are found by their keys (`<name>.inputs.json` beside each part's records).
10. The trainer keeps a whole selection in memory (compact: about 1.2 KB a decision), enough for a
    few million decisions on the rented machines; streaming from disk comes when the data needs it.
11. The deal re-weighting hook is in `student encode` (Rust), so the weighted means are the same
    integers everywhere; the trainer only sees targets.

## 11. Measurements and status

**Stage 1 (the generator), this VM (4 vCPU Xeon, release build, `search teacher`):**

| What | Settings | Result |
|---|---|---|
| Trick 2, H_3 with the real O_3 (`f5766aa9…`, 2048-1024-256) | 64 decisions (8 per place, 16 per stratum), D = 320, 4 threads | 76.8 s: **3,000 decisions an hour**; 4.7 s of a thread a decision (the 4.6 s estimate of §8); 3.45 candidates a decision; 35 KB of values and 1.6 KB of record a decision |
| Byte identity | 32 decisions, D = 16, 1 and 4 threads | records, values and sidecar files identical (SHA-256) |
| Trick 1, H_2 with a random 2048-1024-256 O_2 | 32 decisions, D = 8, 4 threads | 0.57 s of a thread a deal (3 robs among the 32) |
| Trick 1, H_2 with a random 1024-512-128 O_2 | the same | 0.17 s of a thread a deal |

So a 245-thread machine makes about 187,000 trick-2 decisions an hour at D = 320 (245 × 3,600 /
4.7 s): **about $3.80 per million decisions** at $0.72/h, the first 250k in about 1.3 hours.

**Stage 2 (encoder, feature shards, trainer, integer inference, the check):**

- The encoder `student-v1`: 3,083 view features, 109 candidate features (3,192 binary), 100 dense;
  3,292 inputs. At trick 2 a view has at most 80 active features in the 3,000 local decisions
  (room for 448: §2's bound). Encoder hash `321d806d…` (`student hash` prints it and the layout).
  `student encode` turns 3,000 records into a feature shard in 0.4 s; a shard of a record is
  2,850 bytes.
- Integer inference: the view's first-layer sum once, each candidate's rows added (tested equal to
  the plain forward pass on real positions); Python's integer pass equals Rust's bit for bit
  (`python/nn/tests/test_student.py`, `student eval`); the float and integer networks agree within
  0.001 of a logit on random networks; a network for another encoder name or hash is refused.
- `search student-check` (the measure, §7) agrees exactly with the trainer's own integer number
  (0.36618% for the 256-128 student below, both ways).

**The first tiny student (local, to prove the path; trick 2):** training data 3,000 decisions at
only 16 deals (2,000 rules-bot positions, seed 201; 1,000 mixed, seed 202; 6 minutes on 4 threads),
test set 512 decisions at 64 deals (seed `STUDENT_TEST_SEED`). Measured against the teacher's means
on the test set (decision loss, % of money chance; bootstrap 95% upper bound; agreement):

| Player | Loss | Upper 95% | Agreement |
|---|---|---|---|
| A random legal card | 0.520% | | |
| The rules bot (`rules-v1`) | 0.351% | 0.461% | 50.4% |
| Student 256-128 (19 epochs, 10 s) | 0.366% | 0.454% | 45.7% |
| Student 512-256 (13 s) | 0.289% | 0.368% | 51.0% |

It is weak, as expected from 3,000 decisions with 16 deals each (the targets are mostly the
teacher's noise): it shows the path end to end, not a strength. Leaders (place 1) and the race
for the money (stratum S4) are where both the rules bot and the student lose most.

**Stage 3 (the app and the first rented run):**

- WASM: `"method": "student"` (docs/WASM-API.md §3–§5): tricks 1–2 with all eight seats in play
  valued by the student (`loadStudent`), everything else as before, byte for byte; new refusal
  `no_student`. Tested natively (`crates/wasm/tests/api.rs`) and native against WebAssembly
  (`tools/wasm_check.sh` with the real equity table and O_3: PASS, seven student requests with a
  seeded random student among the 21, every split). Speed of the 512-256 student above, a whole
  request (the view rebuilt from the app's JSON, the card tracker, the encoder, the network):
  0.3–0.5 ms natively and 0.4–0.7 ms as WebAssembly under Node on this VM (the hybrid PIMC's
  trick-2 request of 4 deals: 12.6 ms natively); its file is 7.3 MB and loads in 74 ms.
- Runner: chunk kind `student_teacher` (`crates/runner/src/student.rs`): part `first_part + i` of a
  generator stream on every thread, the shard uploaded to `inputs/sha256/<hex>` and listed in
  `out/student/<name>.inputs.json`; a later run adds parts (`first_part`) or skips stored ones
  (`skip_parts`).
- Manifests (not launched): `docs/vast/student-t2-mech.manifest.json` (a few minutes) and
  `docs/vast/student-t2-base.manifest.json`: the test set (2 × 2,048 decisions at 1,024 deals),
  then 200,000 rules-bot and 50,000 mixed decisions at 320 deals: about 1.4 hours on a 245-thread
  machine, about $1 at $0.72/h (more at the $1.70 cap's prices), about 9.5 GB of shards.

**Stage 4 (training on a rented machine; written before launch, results in §11-14 and `docs/EXPERIMENTS.md` E2-E4):**

- Runner chunk kind `student_train` (`crates/runner/src/student_train.rs`, one chunk a stream):
  the learning curve from stored teacher shards. Inputs are named by run and stream
  (`{ "run": "student-t2-base-1", "stream": "rules", "parts": 40 }`: the run's `merged.json`
  lists each part's bundle, whose `student/<file>.inputs.json` gives the shard's keys), or a
  stored shard directly in its `.inputs.json` form. The training parts are spread in proportion
  (four rules parts to one mixed one for the base run), each size is the first that many decisions
  in that order (the trainer's new `--limit`), and only the parts the largest size needs are
  fetched (SHA-256 and size checked) and encoded, each once. For each (size, width): the trainer on
  the GPU (the hybrid rung's Python environment and GPU check), `search student-check` on the test
  set with the rules bot and a random card as baselines (`--baseline random`: a seeded random
  legal card per decision, with the exact expected loss of a random pick printed beside it), the
  networks shared to `inputs/sha256/`; `out/student/curve.txt` and `curve.json` give the loss,
  its upper bound and the agreement overall, by place and by score class. A lost machine's
  finished pairs are restored from the chunk's live files and not trained again.
- Manifests (not launched): `docs/vast/student-t2-curve-mech.manifest.json` (the mechanism run's
  shards, minutes) and `docs/vast/student-t2-curve.manifest.json` (the base run's 250,000
  decisions; sizes 50,000, 100,000 and 250,000; widths 256-128 and 512-256; one GPU of 8 GB or
  more, 32 threads, 64 GB of RAM, 40 GB of disk).
- Tested on this VM: the chunk with fake programs (`cargo test -p runner student_train`); end to
  end with real programs on a local store (`crates/runner/tests/student_train.rs`, `--ignored`:
  a teacher run with a random O_3, then the curve on CPU, then a resume that skips a finished
  pair); and the mechanism manifest itself on CPU against a local copy of `student-t2-mech-1`'s
  bundles and shards (3 MB).

**The deal test (2026-10-18; result in §13):** `student_train` takes `points` (each a size,
widths and `deals`: the targets from the first `deals` deals, encoded once per deal count with
`--max-deals` into its own feature set; the point's directory and resume key gain `-d<deals>`,
`curve.txt` a deals column), entries in ordered `group`s (a later run's parts follow an earlier
run's without changing its prefix) and a run's stored part bundles listed directly (`bundles`, for
a run stopped before its merged report). `docs/vast/student-t2-dealtest.manifest.json`: the base
run's 250,000 decisions then the 75 rules parts `student-t2-more-1` stored before it stopped
(625,000 in all; it made no mixed parts), 512-256 students at 250,000 × 320 deals (the anchor:
the earlier curve point's exact data and settings), 126,000 × 320, 315,000 × 128, 625,000 × 64,
625,000 × 32 and 126,000 × 64; `student-t2-dealtest-mech.manifest.json` the same mechanism on
`student-t2-mech-1`'s shards.

## 12. Next

1. Launch `student-t2-mech` (minutes), then `student-t2-base` (about 1.4 hours, about $1 on a
   245-thread machine at $0.72/h): the main session reviews and launches.
2. Train 256-128 and 512-256 students on 50k, 100k and 250k decisions (the learning curve) and
   measure them on the test set with `search student-check --baseline rules --baseline random`:
   launch `student-t2-curve-mech`, then `student-t2-curve` (the runner's `student_train`, §11
   Stage 4).
3. A student bot for the arena (`search::bot`), so promotion is by SPRT in duplicate tournaments
   (PLAN §9), and the DAgger source (`--source student`: positions from the student's own games).
4. Trick 1 once O_2 exists: price it with the real O_2 first (§8: its width sets the cost).

## 13. How many deals per decision (measured 2026-10-18, run `student-t2-dealtest-1`, $0.67)

Trick-2 student 512-256 on the stored records (base + `student-t2-more-1`), each target cut to the first N
deals (`student encode --max-deals`, exact subsets), scored on the fixed 4,096-decision test set at 1,024 deals:

| positions | deals | teacher cost | loss % (95% upper) | agree |
|---|---|---|---|---|
| 126,000 | 320 | 1x | 0.1140 (0.1317) | 63.7% |
| 315,000 | 128 | 1x | **0.0695** (0.0771) | 64.2% |
| 625,000 | 64 | 1x | 0.0760 (0.0875) | 63.8% |
| 625,000 | 32 | 0.5x | 0.0776 (0.0862) | 63.3% |
| 126,000 | 64 | 0.2x | 0.1463 (0.1670) | 60.0% |
| 250,000 | 320 | 2x | 0.0784 (0.0912) | 64.4% (identical to the curve run: the check) |

At equal teacher cost, more positions at 64-128 deals beat 320 deals clearly; at equal positions more deals
still help. Decision (2026-10-18): new student data at **64 deals** (extend close calls later by deal index);
trick-1 data likewise at 64-128, confirmed on a first trick-1 batch the same way.

## 14. The ruler: how far from 0.01%? (`student-check --split-half --top-k 2,3`, measured 2026-10-19)

Today's measure (§7) compares the student's card with the teacher's best on the same 1,024 deals.
The teacher's best is partly luck (the card that happened to look best on those deals), so the
measure reads a little high: the "winner's curse". Two new switches on `search student-check`
measure it honestly and say how far the student could get:

- `--split-half`: each decision's deals are cut in two by a fixed rule: **even deal index = half A,
  odd = half B** (512 each; `search::curves::split_halves`). A card is chosen on one half and scored
  on the other, which never saw it, then the other way round; the tables give A→B, B→A and the
  mean. Three numbers:
  - **floor**: a teacher of 512 deals (the best card on A) scored on B against B's best. This is
    what a near-perfect player reads on this test set at half width: the test set cannot show a
    loss much below it.
  - **student, split**: the student's (fixed) card against each half's best.
  - **gap**: the A-best card less the student's card, scored on B (and the reverse). Nothing in it
    is lucky, so it is the true distance between the student and a teacher of 512 deals. It can be
    negative. Always gap = student split − floor.
- `--top-k K`: the student shortlists its K best cards and the teacher picks among them (on all
  deals; with `--split-half` also picked on one half and scored on the other). This is what a
  "student shortlists, search decides" player would lose.

Same groups and bootstrap as §7 (1,000 replicates, seed 1, the same resampled decisions in every
table). "lower95/upper95" are the 5% and 95% points of the bootstrap, so the two together are a
90% range. Without the switches the output is byte for byte as before; with them the full-deal
student loss is repeated and equals §7's (0.0566% for the 1M 512-256 student). Whole numbers
throughout (units of 2^-32, totals over the scoring half), % only when printed.

Results on the fixed test set (4,096 trick-2 decisions, % of the deciding pair's money chance):

| student | full loss | split-half student | floor | gap (90% range) | top-2 full | top-3 full | top-2 split | top-3 split |
|---|---|---|---|---|---|---|---|---|
| 1M, 512-256, 64 deals | 0.0566 | 0.0586 | 0.0080 | 0.0506 (0.0443–0.0575) | 0.0174 | 0.0050 | 0.0233 | 0.0122 |
| 1M, 1024-512, all deals | 0.0560 | 0.0579 | 0.0080 | 0.0500 (0.0426–0.0581) | 0.0131 | 0.0030 | 0.0190 | 0.0104 |
| 1M, 1024-512, 64 deals | 0.0575 | 0.0594 | 0.0080 | 0.0515 (0.0436–0.0602) | 0.0189 | 0.0040 | 0.0251 | 0.0115 |
| 250k, 512-256, 320 deals | 0.0784 | 0.0803 | 0.0080 | 0.0723 (0.0629–0.0849) | 0.0219 | 0.0046 | 0.0280 | 0.0118 |

1M, 512-256, 64 deals:

| group | full loss | split student | floor | gap | gap 90% range | top-2 full | top-3 full | top-2 split | top-3 split |
|---|---|---|---|---|---|---|---|---|---|
| all | 0.0566 | 0.0586 | 0.0080 | 0.0506 | 0.0443–0.0575 | 0.0174 | 0.0050 | 0.0233 | 0.0122 |
| P1 | 0.0838 | 0.0886 | 0.0177 | 0.0710 | 0.0520–0.0933 | 0.0392 | 0.0194 | 0.0517 | 0.0351 |
| P2 | 0.0412 | 0.0426 | 0.0064 | 0.0362 | 0.0169–0.0586 | 0.0072 | 0.0002 | 0.0126 | 0.0064 |
| P3 | 0.0385 | 0.0400 | 0.0060 | 0.0340 | 0.0203–0.0505 | 0.0166 | 0.0011 | 0.0213 | 0.0061 |
| P4 | 0.0625 | 0.0644 | 0.0100 | 0.0544 | 0.0346–0.0766 | 0.0197 | 0.0073 | 0.0262 | 0.0163 |
| P5 | 0.0879 | 0.0905 | 0.0104 | 0.0800 | 0.0590–0.1051 | 0.0276 | 0.0063 | 0.0345 | 0.0150 |
| P6 | 0.0508 | 0.0528 | 0.0078 | 0.0450 | 0.0296–0.0615 | 0.0099 | 0.0016 | 0.0167 | 0.0090 |
| P7 | 0.0499 | 0.0508 | 0.0035 | 0.0473 | 0.0327–0.0659 | 0.0061 | 0.0001 | 0.0088 | 0.0034 |
| P8 | 0.0385 | 0.0390 | 0.0021 | 0.0369 | 0.0236–0.0511 | 0.0132 | 0.0040 | 0.0147 | 0.0060 |
| S1 | 0.0325 | 0.0338 | 0.0046 | 0.0292 | 0.0223–0.0365 | 0.0095 | 0.0029 | 0.0132 | 0.0072 |
| S2 | 0.0557 | 0.0572 | 0.0060 | 0.0512 | 0.0399–0.0636 | 0.0145 | 0.0044 | 0.0190 | 0.0099 |
| S3 | 0.0495 | 0.0515 | 0.0074 | 0.0441 | 0.0319–0.0571 | 0.0161 | 0.0018 | 0.0213 | 0.0081 |
| S4 | 0.0889 | 0.0919 | 0.0140 | 0.0780 | 0.0609–0.0964 | 0.0296 | 0.0109 | 0.0397 | 0.0233 |

1M, 1024-512, all deals:

| group | full loss | split student | floor | gap | gap 90% range | top-2 full | top-3 full | top-2 split | top-3 split |
|---|---|---|---|---|---|---|---|---|---|
| all | 0.0560 | 0.0579 | 0.0080 | 0.0500 | 0.0426–0.0581 | 0.0131 | 0.0030 | 0.0190 | 0.0104 |
| P1 | 0.0800 | 0.0848 | 0.0177 | 0.0672 | 0.0500–0.0867 | 0.0230 | 0.0067 | 0.0360 | 0.0220 |
| P2 | 0.0269 | 0.0282 | 0.0064 | 0.0218 | 0.0132–0.0310 | 0.0042 | 0.0001 | 0.0084 | 0.0064 |
| P3 | 0.0423 | 0.0438 | 0.0060 | 0.0378 | 0.0220–0.0552 | 0.0130 | 0.0007 | 0.0172 | 0.0064 |
| P4 | 0.0699 | 0.0719 | 0.0100 | 0.0619 | 0.0271–0.1156 | 0.0054 | 0.0010 | 0.0118 | 0.0105 |
| P5 | 0.0637 | 0.0662 | 0.0104 | 0.0558 | 0.0422–0.0716 | 0.0172 | 0.0062 | 0.0260 | 0.0156 |
| P6 | 0.0518 | 0.0538 | 0.0078 | 0.0460 | 0.0307–0.0612 | 0.0107 | 0.0048 | 0.0167 | 0.0118 |
| P7 | 0.0406 | 0.0415 | 0.0035 | 0.0380 | 0.0245–0.0529 | 0.0102 | 0.0031 | 0.0130 | 0.0064 |
| P8 | 0.0728 | 0.0733 | 0.0021 | 0.0712 | 0.0459–0.0997 | 0.0214 | 0.0017 | 0.0228 | 0.0036 |
| S1 | 0.0375 | 0.0388 | 0.0046 | 0.0342 | 0.0265–0.0422 | 0.0080 | 0.0008 | 0.0115 | 0.0051 |
| S2 | 0.0462 | 0.0476 | 0.0060 | 0.0416 | 0.0321–0.0511 | 0.0160 | 0.0054 | 0.0207 | 0.0109 |
| S3 | 0.0447 | 0.0467 | 0.0074 | 0.0393 | 0.0300–0.0497 | 0.0103 | 0.0034 | 0.0160 | 0.0100 |
| S4 | 0.0956 | 0.0987 | 0.0140 | 0.0847 | 0.0586–0.1128 | 0.0183 | 0.0026 | 0.0278 | 0.0154 |

250k, 512-256, 320 deals:

| group | full loss | split student | floor | gap | gap 90% range | top-2 full | top-3 full | top-2 split | top-3 split |
|---|---|---|---|---|---|---|---|---|---|
| all | 0.0784 | 0.0803 | 0.0080 | 0.0723 | 0.0629–0.0849 | 0.0219 | 0.0046 | 0.0280 | 0.0118 |
| P1 | 0.0894 | 0.0942 | 0.0177 | 0.0766 | 0.0569–0.0971 | 0.0311 | 0.0142 | 0.0441 | 0.0302 |
| P2 | 0.0849 | 0.0862 | 0.0064 | 0.0798 | 0.0305–0.1524 | 0.0594 | 0.0046 | 0.0641 | 0.0106 |
| P3 | 0.0623 | 0.0638 | 0.0060 | 0.0578 | 0.0385–0.0793 | 0.0202 | 0.0033 | 0.0256 | 0.0089 |
| P4 | 0.0648 | 0.0667 | 0.0100 | 0.0567 | 0.0381–0.0787 | 0.0132 | 0.0017 | 0.0205 | 0.0105 |
| P5 | 0.1101 | 0.1126 | 0.0104 | 0.1022 | 0.0775–0.1279 | 0.0252 | 0.0073 | 0.0334 | 0.0163 |
| P6 | 0.0994 | 0.1015 | 0.0078 | 0.0937 | 0.0589–0.1363 | 0.0049 | 0.0013 | 0.0113 | 0.0083 |
| P7 | 0.0523 | 0.0533 | 0.0035 | 0.0497 | 0.0332–0.0680 | 0.0096 | 0.0040 | 0.0125 | 0.0073 |
| P8 | 0.0638 | 0.0643 | 0.0021 | 0.0622 | 0.0388–0.0907 | 0.0114 | 0.0001 | 0.0129 | 0.0021 |
| S1 | 0.0529 | 0.0542 | 0.0046 | 0.0496 | 0.0387–0.0623 | 0.0135 | 0.0024 | 0.0169 | 0.0065 |
| S2 | 0.0600 | 0.0615 | 0.0060 | 0.0555 | 0.0420–0.0672 | 0.0182 | 0.0068 | 0.0230 | 0.0124 |
| S3 | 0.0578 | 0.0598 | 0.0074 | 0.0525 | 0.0400–0.0680 | 0.0126 | 0.0030 | 0.0180 | 0.0093 |
| S4 | 0.1427 | 0.1458 | 0.0140 | 0.1318 | 0.0973–0.1736 | 0.0432 | 0.0061 | 0.0542 | 0.0190 |

**How to read it:**

1. **The winner's curse is small.** The full-deal loss (0.0566%) and the honest gap to a 512-deal
   teacher (0.0506%) differ by less than the floor: almost all of today's number is real loss, not
   teacher noise. The 1024-512 networks are not measurably better than the 512-256 one (their
   gaps overlap); the 250k student is clearly worse (0.072%).
2. **The test set's own floor is about 0.008%** at 512 deals a half (0.018% for leaders, 0.014% in
   the race for the money S4, 0.002–0.004% for the last places). A player at 0.01% is therefore
   only just measurable with this test set; showing a difference of 0.01% between two players
   needs the gap measure (paired, unbiased) rather than the raw loss.
3. **The right card is almost always in the student's top 3.** If a search chose among the
   student's top 3 cards, the loss would be about 0.003–0.005% on all deals and 0.010–0.012% split
   (where the floor alone is 0.008%): about 0.01%, the target, with the search looking at three
   cards instead of every legal one. Top 2 gives 0.013–0.019% on all deals (0.019–0.025% split).
   The student alone is about five times further away.
4. **Where the loss is:** the leader (place 1), place 5 and stratum S4 (a pair close to the money),
   as before; the leader also has the highest floor (its choices are the closest calls).
