# O3G2 rounds: G2 chooses O_3's next training positions (built 2026-10-21)

**Status (2026-10-22): two rounds run; neither improved O_3.** O_3 f5766a stays the adopted network.
Results, the lessons from the launches and the reading are in "Results" below.

## Why (owner, 2026-10-21)
O_3 (the trick-3 network) with G2 (the predictor of where O_3 is wrong; those positions are solved
exactly) plays trick-2 decisions at 0.0005% mean, 0.0008% 95% bound, seat 1 0.0022% on the
golden set. O_3 alone is at 0.0056% / 0.0068%, seat 1 0.0154% / 0.0201%. O_3 alone is what
values the trick-3 positions in O_2's training labels (exact through trick 2, then O_3, never G2:
G2 makes labels too slow), and in the bronze set (O_2's test set: exact through tricks 1 and 2,
then O_3, with G2 only if it has become cheap by then). So **O_3 alone must come close to
O_3 + G2**. Target: O_3 alone at most 0.0025% (95% bound) overall and at every seat, the O_3
share of the agreed error budget.

The idea: G2 already knows where O_3 is wrong, so it chooses the new training positions. Each round
adds G2-chosen positions, trains candidate networks, keeps the best, builds the next G2 on it,
and stores everything for the next round.

## The research
The same method appears in several proven forms; G2 plays the "loss predictor" in each:
- **Learning Loss for Active Learning** (Yoo & Kweon, CVPR 2019): a second model predicts the main
  model's loss; the examples it predicts worst are labelled next.
- **Prioritized experience replay** (Schaul et al., ICLR 2016) and **importance sampling by the loss**
  (Katharopoulos & Fleuret, ICML 2018): sample each example with probability proportional to its
  priority to a power α.
- **KataGo's policy-surprise weighting** (Wu 2019/2020): training positions where the network was
  most wrong are shown more often.
- **Neyman allocation** [stratified sampling: give each group samples in proportion to its size and
  its spread] and **defensive importance sampling** (Hesterberg 1995): always keep a share drawn
  the plain way, so a wrong predictor cannot starve any kind of position.
- Already in the project: the find-fix-repeat audit loop (`audit-loop.md` §2) and additive training
  with replay [continue a network on old and new data together] (`audit-loop.md` §10).

For the training labels (exact through trick 2, then O_3): a network trained on millions of labels
averages away random label errors, but learns one-sided errors (bias) faithfully. So each round
measures the bias of O_3's labels directly (the label check, below).

## How the positions are chosen (`search::risk_harvest`)
1. **Pilot** (`search risk-profile`, no labels, seconds): the harvest's own search (H_3 driven by
   the incumbent O_3) on 2,000 decisions of a seed of their own; G2 scores every trick-3 position
   it reaches.
2. **Risk profile**: the scores cut into 20 bands of equal count, lowest risk first; each band's
   natural share (how often it occurs) and its mean predicted squared error (how wrong O_3 is there).
3. **Target mix**: a band's share of the new rows is
   `(1 − f) × frequency × risk^α / sum + f × frequency`. **α = 1** (owner, 2026-10-21, on the
   argument below) and **f = 0.2** (a plain 20% of the new rows; not of the whole training set).
4. **Kept before labelling**: each reached position is kept with its band's acceptance (a coin
   drawn from the decision's own seeded stream, so the rows are identical on any machine and thread
   count). Only kept positions get an exact label, so scanning many is cheap. Then at most
   `risk_cap` (1,000) a decision.

**Why α = 1.** With G2 in play O_3's riskiest positions are solved exactly, which argued for α = 0.5.
But the labels and the bronze set use O_3 alone, and the gap between O_3 alone and O_3 + G2 is
exactly O_3's error on the positions G2 would solve: G2's riskiest 10% hold about 84% of it. α = 1
sends the new data there. α stays a setting (`risk_power`: 0, 0.25, 0.5, 0.75, 1); an A/B test of
0.5 against 1 at the same amount costs about $1 if rounds stall.

**Measured on the build VM (2026-10-21, a random trick-3 network, a two-band test model):**
- A trick-1 decision's search reaches about **600,000** trick-3 positions, a trick-2 decision about
  **700**. Half the rows from trick-1 decisions would cost hours of CPU a round and swamp the
  profile, so `risk_tricks` is **2** by default (`"1,2"` remains an option).
- A low per-decision cap distorts the mix: risky positions bunch in a few decisions. The profile
  asked for 86% of rows in the risky band; the shard held 56% at a cap of 32, 69% at 256, 80%
  uncapped (the rest is the small test's noise). So `risk_cap` is **1,000**. At 2M+ rows a round
  still draws on tens of thousands of decisions. Each round's report prints the mix aimed at and the
  mix made.
- Labels: about 8 thread-seconds per 1,000 rows here (rented threads run 3–4 times slower): 8M rows
  about 30 minutes on 256 rented threads.

## One round on one machine (`hybrid_rung` with `round`; `crates/runner/src/hybrid/round.rs`)
| step | CPU | GPU |
|---|---|---|
| 1 | fetch the stored base (40M positions), the networks, G2, the golden set | idle |
| 2 | risk profile; part 1 of the new labels | |
| 3 | parts 2 and 3; the incumbent's golden check and label check | **candidate 1** trains (base + part 1) |
| 4 | candidate 1: golden check, paired comparison with the incumbent, label check | **candidate 2** (base + parts 1–2) |
| 5 | the same for candidate 2 | **candidate 3** (base + parts 1–3) |
| 6 | the same for candidate 3; **the winner**; **its G2**; the record; uploads; the machine destroys itself | idle |

- **Nested parts:** the amounts (e.g. 2M, 4M, 8M) are cumulative: part 1 is 2M, part 2 another 2M,
  part 3 another 4M; each part is labelled once and stored as soon as it is made.
- **Candidates:** each continues the incumbent (additive, 8 epochs at 0.0003) on the base and its
  parts; optionally the last trains a wider network from scratch (`--scratch-last 4096,2048,1024,256`)
  to test whether size, not data, is the limit.
- **Golden check:** H_3 *without* exact leaves against the stored 2,000 trick-2 decisions × 320
  exact worlds (the incumbent's check recomputes 1% of the stored worlds first). The yardstick is
  O_3 alone; the stored O_3 + G2 result (0.0008%, seat 1 0.0022%) is the mark to approach.
- **The winner:** a candidate is adopted over the incumbent unless it loses a pass the incumbent had
  (overall, a seat band, a team's bias, at the 0.01% standard) or is worse by more than the noise
  tolerance (0.002 points): the additive loop's rule. Among the adopted, the lowest overall bound
  wins; none adopted keeps the incumbent (the new labels are stored and used next round anyway).
- **Label check:** each network labels the stored exact trick-2 validation shard's positions (50,000,
  seed 601) as O_2's training labels would be made, and `fortyone_nn.labelcmp` reports the error
  spread and the bias per team against the exact labels. About a minute a network.
- **The next G2** is built on the winner at the end (training 400k, calibration 100k, test 100k
  positions; the same recipe as G v2), so the next round starts at full speed.
- **Stored in R2:** every new part and its sidecar, every candidate's networks (integer and float),
  the new G2 and its calibration, the dataset list, and `rounds/<id>/round.json` (plus `report.txt`
  and `profile.json`): the incumbent, every candidate with its numbers, the winner, and `next` (the
  networks, G2, dataset and golden set the next round starts from).

Time and money (estimates): 3–5 hours a round, about $1.5–2.5 (a $0.30/h 256-thread EPYC with an
RTX 3060 is at the slow end for training, about 30–40 minutes a candidate).

## How to run a round
**From GitHub (phone or laptop):** Actions → **o3-round** → Run workflow:
- `round`: 1, 2, …; `from`: `seed` for round 1, else the previous round's id (`o3r-1`);
- `amounts`: e.g. `2M,4M,8M`; optional `scratch_last`, `risk_power`, `g2` (`auto` reuses the
  previous round's G2, `rebuild` builds it first);
- the machine: `offer_id` (an offer's Type # number, rented at once), or `cpu_match` (e.g. `7B12`),
  or nothing (the launcher's search), or **`instance_id`** (a machine you already rented by hand,
  below); `min_gpu_ram_gb` 16 (12 lets an RTX 3060 in for an all-additive round);
- `max_dollars` (6), `dry_run` (prints the manifest and the machine, rents nothing).

From a laptop with `gh`: `gh workflow run o3-round.yml --ref <branch> -f round=1 -f from=seed -f
amounts=2M,4M,8M -f offer_id=36863856 -f min_gpu_ram_gb=12`.

**Taking over a machine you rented by hand** (`instance_id`; `launch.py run --instance-id`): vast.ai's
documented `vastai update instance ID --image … --env … --onstart …` stores this run's image,
settings and start command ([docs](https://docs.vast.ai/cli/reference/update-instance)), and
`vastai recycle instance ID` rebuilds the container in place from them
([docs](https://docs.vast.ai/cli/reference/recycle-instance)): `update` alone left the old
container running (2026-10-21, instance 57085233). The launcher then watches, stops and destroys it
like one it rented. Two limits: **the disk is fixed when you rent** (a round needs about
250 GB; the launcher refuses a smaller one before touching it), and if the machine fails no other is
rented. First used on 2026-10-21 (round 1).

**Watch / stop / results:** as every run (STATUS "How to run things"): heartbeats name the step
(`risk profile`, `new labels (part k)`, `train (candidate i)`, `golden check (ci)`, `G2 (g2-winner)`);
cancelling the workflow run stops, uploads and destroys. Results: `rounds/<id>/report.txt` and
`round.json` in R2 (`gh workflow run r2-fetch.yml -f run_id=o3r-1-1` fetches the run's files).

## Settings (`round`, inside a `hybrid_rung` stream)
`id`, `seed` (the profile; parts `seed + k`; the next G2's splits from `g2.splits`), `risk_model`
(the incumbent's stored G2; left out: built first from `g2`), `risk_power` ("1"), `risk_floor`
("0.2"), `risk_bands` (20), `risk_pilot` (2,000), `risk_cap` (1,000), `risk_tricks` ("2"),
`risk_positions` ("0.4,0.3,0.3"), `candidates` (one to three `{rows, mode, widths}`, growing),
`scratch_epochs` (40), `scratch_lr` (0.001), `label_check` (`{shard, sidecar}`), `g2` (the
`leafrisk` settings without `trick` and `net_input`). The rung must be additive (`init_net`,
`init_net_float`), its whole base stored (`label_inputs` with the validation shard), the golden set
stored, `upload_shards` true, exact labels, no audit loop. Fixed values for every round:
`docs/vast/o3-rounds/defaults.json`; round 1's start: `docs/vast/o3-rounds/seed.json`;
`python/launcher/round_manifest.py` builds the manifest (`launch.py round-manifest`).

## Tests
- Rust: the profile arithmetic by hand, the band edges, thread-independence of the pilot and the
  shard, a zero-risk band never kept, a model for another network refused (`search`); the round's
  settings and refusals, and a whole round with fake programs (three candidates, one not adopted,
  the winner, its G2, the stored record, the commands' arguments, candidate 1 training beside the
  later parts) (`runner`); the round-1 manifest in the manifests' resolve test.
- Python: `labelcmp` by hand; the manifest builder and the take-over checks (`test_round_manifest`).
- Real binaries on the build VM: `search risk-profile` and a risk-guided `search labels` with a
  random network (the measurements above).

## Results

### Round 1 (`o3r-1`, run `o3r-1-3`, 2026-10-21): from the seed, three additive candidates
Machine: AMD EPYC 9B14 (184 usable threads), RTX 5090 32 GB, $1.34/h, 51 minutes, about $1.14.
New rows 2M, 4M, 8M (nested), risk power 1, plain floor 0.2. The mix made: 57.8% of the new rows
in the riskiest 2 of the 20 bands (aimed at: 63.3%).

| network | new rows | 95% bound (all) | seat 1 | adopted? |
|---|---|---|---|---|
| incumbent O_3 f5766a | none | 0.0068% | 0.0201% | stays |
| c1 | +2M | 0.0071% | 0.0227% | no: team 0's bias 0.0121% over 0.0100% |
| c2 | +4M | 0.0080% | 0.0288% | no: positions 1-2 bound 0.0169% over 0.0122% + 0.0020% |
| c3 | +8M | 0.0073% | 0.0229% | no: team 0's bias 0.0111% over 0.0100% |

Label check (the exact trick-2 validation rows labelled by each network, as O_2's labels would be):
RMS error 2.113% (incumbent), 2.123% / 2.094% / 2.105% (c1, c2, c3); largest team bias 0.013-0.016%.
Winner: the incumbent. The new rows (8M) are stored and used by round 2.

### Round 2 (`o3r-2`, run `o3r-2-1`, 2026-10-22): c1, c2 additive, c3 a wider network from scratch
Same machine type, 91 minutes, about $2.04. 58.7% of the new rows in the riskiest 2 bands (aimed
at 62.1%). c3 = 4096-2048-1024-256 from scratch on the base plus 8M new rows (40 epochs, lr 0.001).

| network | new rows | 95% bound (all) | seat 1 | adopted? |
|---|---|---|---|---|
| incumbent O_3 f5766a | none | 0.0068% | 0.0201% | |
| c1 | +2M | 0.0069% | 0.0233% | no: team 0's bias 0.0100%, at the limit |
| c2 | +4M | 0.0070% | 0.0223% | **yes** by the rule (passes every pass the incumbent had, within the 0.002-point tolerance) |
| c3 wide, from scratch | +8M | 0.0079% | 0.0325% | no: positions 1-2 bound 0.0182%; team 0 bias 0.0155%, team 1 0.0109% |

Label check: RMS 2.112% (c1), 2.104% (c2), 2.258% (c3, the wide network is worse; its bias is lower,
0.0082%). The wide network took about 3.3 times a candidate's compute. The round chose c2
(`init_net inputs/sha256/1057e9e949f04f09e59e4ef052b564694f1cc2140fbbd81a146188e46342d593`, its G2
`5f7e6792a70b4ecf724896edea1a7d19fd8b80e592e6d623a5f450b54829ffe6`, dataset
`ea682abeeaab4ca3c60123e414e825a30f07411439e9802be79073d0b7b2b0d1`) as the start of round 3, but
**c2 is not better than O_3**: 0.0070% against 0.0068% is inside the ±0.002-point noise of a retrain.
Nothing was switched to c2.

### Reading (the 2026-10-22 session's interpretation, not a measurement; facts in `docs/EXPERIMENTS.md` C12)
- Targeted extra data (G2-chosen rows, up to +8M on a 40M base) did not move the golden bound, and
  the candidates landed within noise of each other. A bigger network from scratch was clearly worse
  here. The 95% bound did not move toward the 0.0025% target.
- Together with the specialist test (`specialist.md`: a network trained only on G2's risky
  positions is no better than O_3 on them) three independent attempts all end at O_3's level.
- Not tried: the 20M-versus-40M base curve (prepared 2026-10-19, never ran). The 80M base, read on
  2026-10-22: lower value error, decisions not better (0.0080%; `docs/EXPERIMENTS.md` C4). The 4-network
  ensemble was never measured (C8).

### Launch lessons (each cost a rental or a retry)
- **A 36 GB shard downloaded on one stream exceeded the 30-minute limit** on a Thailand machine
  (`o3r-1-1`): downloads over 64 MiB now use 16 streams and the limit is 3 hours (commit ffbe660).
- **A named offer id can vanish before it is rented** (offer 36863856 did, 2026-10-21): the launcher
  refuses it and rents nothing; send a current listing. Attempt `o3r-1-2` (offer 35971321) rented
  nothing either; its cause was not looked up.
- **`vastai update instance` alone does not restart the container**: `recycle instance` applies it.
- **A host below the 90% reliability floor needs `min_reliability`** (the o3-round workflow got that
  input on 2026-10-22; 86% was accepted for round 1 on the owner's word).
- **A run id cannot be reused**: a failed attempt leaves `runs/<id>/` in R2 and the launcher refuses
  the same id; give each attempt a new id (`attempt` in the manifest builder).

## Honest risks
- Untargeted extra harvest did not help O_3 before, and retraining alone moves the golden bound by
  about ±0.002 points: a round can end "incumbent kept". The race-for-the-money positions improved
  little with 2.5× the data, so network size (`scratch_last`) or an ensemble for building the bronze
  set may be the bigger lever.
- Picking winners on one golden set round after round slowly flatters them: after about three
  rounds, or before claiming a final number, confirm on a fresh golden set (seed 2; about 1.5 hours
  on 190 threads, $0.5–1).
- A full round runs in 1 to 1.5 hours and $1-2 on a 184-thread machine with an RTX 5090 (rounds 1
  and 2), faster than the 3-5 hours first estimated. The base is fetched while the first parts are
  labelled; the golden checks are the long part.
