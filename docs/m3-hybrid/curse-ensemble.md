# The optimizer's-curse test and O_3 ensembles (build of 2026-10-19)

**Why.** On the stored trick-2 golden set (2,000 decisions x 320 exact worlds), H_3 [exact search through
trick 2, the integer O_3 network at each trick-3 leaf] undervalues the decider's own team: seat 1 -0.033%,
seat 2 -0.021%, seat 3 -0.032%, seat 4 -0.006%, seats 5-8 about 0. At the position level O_3's bias is
about 0 (leader's team -0.003%) and its error SD about 1.87%. The suspicion is the **optimizer's curse**
[a search that picks the best of several noisy values picks, on average, the one whose noise flatters it,
so the values it copies upwards are biased even when every leaf is unbiased]. Two builds test it and one
possible cure.

## A. The curse test (`search hybrid-check --truth net`, `crates/search/src/curse.rs`)

The network itself is the truth, so nothing but the search over noisy values can produce a loss or a bias:
- **truth**: H_3 with O_3's own, noise-free values at the leaves (the check's reference path, at the same cut);
- **player**: the same search with O_3's values plus seeded zero-mean noise at each leaf.

Everything else is the ordinary check: decision loss with its bootstrap bound, overall, by seat band, per
seat, and every team's bias per seat (the own team first).

**The noise** is added to the leaf's 4-team money vector as transfers between pairs of teams: for each
pair of teams that both have money, a bell-shaped whole-number draw (Irwin-Hall: twelve uniform 32-bit
numbers, exactly symmetric) scaled to SD `s / sqrt(m - 1)` (`m` teams with money), shrunk by 7/8 steps while
it exceeds `min(v_i, v_j) / (m - 1)`, added to one team and taken from the other. So each team's change
has SD `s` (before shrinking), the vector always sums to exactly 2^32 and no share goes below 0, with no
renormalisation step; and every step is odd in the draw, so the noise is exactly zero-mean given the leaf's
value (never clipped on one side). Teams with nothing take no noise.

**The SD per race class** is the auditor's measured per-team error SD of O_3 (f5766a) on a fresh 20M-row
trick-3 audit shard (run `m3-hybrid-t3-additive-2`, round 1, `auditor/report.json`, `overall.by_race_class`):
class 0 0.9779%, class 1 0.9184%, class 2 2.0105%, class 3 1.5191%, class 4 4.0498% (overall 1.865%).
Where a team has little money (race classes 3-4, a race near its end) the shrinking makes the realised SD
smaller than asked, so the check **calibrates** first (`--noise-calibrate N`): on every k-th of N decisions
and their first 8 worlds it measures the realised SD per class over every team of every distinct leaf and
rescales the SD asked for by target / realised, four rounds; the main run reports target, asked and
realised SD per class. The noise matches the auditor's SD per class, not its heavy tails (99.9th percentile
of |error| 23%) and not its correlation between neighbouring positions (a network's errors at sibling
leaves are alike, which spares the search much of the curse that independent noise causes).

**Seeding**: the noise of a leaf is `Rng::derive(seed, [NOISE, decision, world, the canonical position as
nine words])`, a pure function of (seed, decision, world, position): byte-identical on any thread count,
and the search sees one value per position.

## B. Ensembles (`--net A,B,…`, `crates/nn/src/ensemble.rs`)

Several integer networks of the same trick as one leaf: each member values the position (its own caches and
batched kernel), the four per-team values are summed in u128 over the members in the order given, and the
mean is rounded by largest remainder (each share rounded down, the at most three missing units to the
largest remainders, ties to the lower team: the rule of every money chance in the project), summing to
exactly 2^32. An ensemble of one network is that network, byte for byte; the mean of identical networks is
the network. The check's JSON names an ensemble `ens<N>:<sha>+<sha>+…` and lists its members.

`search hybrid-paired --a A.json --b B.json` compares two checks on the same decisions: mean decision loss
and own-team bias, overall, by seat band, seat 1 and seat 2, for both, B - A, and its bootstrap 95% interval
(decisions resampled within each group, the same draws for both, so what they share cancels).

## The run (`docs/vast/o3-curse-ensemble.manifest.json`, kind `hybrid_checks`, `crates/runner/src/checks.rs`)

One CPU machine (at least 128 cores, 60 GB), no GPU, no training, on the stored golden set:
1. `curse`: the curse test with f5766a, calibrated on 250 decisions;
2. `single`: f5766a against the exact golden worlds (1% of the worlds recomputed first, byte-identical);
3. `ensemble4`: the integer mean of the four trained O_3s (all 2048-wide, outcome head):
   f5766aa98a1d (loop-1 round 2, adopted), a099061b6aed (loop-1 round 3), bfd1bec2c647 (loop-1 round 4),
   a97ac1771b98 (additive-2 round 2, continued from f5766a);
4. the paired comparison of 2 and 3.

Network keys (`inputs/sha256/<sha>`, all confirmed stored): f5766aa98a1d769e31122c94cc4da9790c84508fab5c432d0c0358fe3124faad,
a099061b6aedc09d660aa006820c6d9f450dd93a301cfbee06a2e3582193c594, bfd1bec2c6473ed15bb7136f1af5857e35a4346c9ac6d714acce09f5a0d1f2ec,
a97ac1771b984c52c851501f56edf7bbf8332b57fe9e9c99c8e2392f33566283; also stored, not in the ensemble: loop-1 round 1
b34e12b51099c065e44ceefa5eb64aa4c725f70027b2d699a2c05adfc953f279 (the weakest, 0.0088%).

## Speed of an ensemble (fix of 2026-10-19, after the first rented run)

On the rented AMD EPYC 7B12 (two 64-core chips, 245 threads, 128 GB, $0.30/h) the single check took 271 s, but the
4-network ensemble ran at about 9 worlds a second, about 100 times slower per world than one network, where this
build VM (a Xeon with 260 MB of last-level cache) showed the ensemble costing about the sum of its members. The
cause: the members' weights together (about 70 MB) do not fit the EPYC's last-level cache (16 MB per group of four
cores), while one 2048-1024-256 network's (about 9 MB) does; so every batch of positions streamed the weights from
memory, and the solver's batches were small (about 13 positions a pass, from 3 plays before the frontier). The fix
makes far fewer passes over the weights, with the same positions valued and the same values: an ensemble of
several networks takes the solver's batches from 7 plays before the frontier (at trick 2 every frontier position
below the decision's first play; the search reaches almost all of them anyway), values them through each member
in passes of up to 256 positions (`nn::kernel::MAX_BATCH`, was 64), and keeps the batch's means in a per-thread
table the search's requests read; the solver's gather cap is 65,536 positions (was 256). On 16 golden decisions x
32 worlds the 4-network ensemble's passes over the weights fell from 108,708 to 7,384 (15 times fewer), the
network evaluations from 1,430,124 to 1,431,056 (+0.07%), and every value and table stayed byte-identical. One
network is unchanged.
