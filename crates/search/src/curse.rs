//! The optimizer's-curse test of the hybrid check (`search hybrid-check --truth net`; the
//! diagnosis of 2026-10-19: H_3 undervalues the decider's own team at seats 1-4 while O_3's own
//! position-level bias is about zero).
//!
//! The question: does a search that maximises over noisy leaf values, with zero-mean noise of the
//! size the auditor measured, by itself produce the own-team undervaluation the golden check sees?
//! [the optimizer's curse: each player picks the card whose noise flatters its own team, so the
//! values copied up the tree are biased even when every leaf is unbiased]. The check then takes
//! the network itself as the truth:
//! - **truth**: the hybrid's own search (exact to the start of the cut trick) with the network's
//!   noise-free values at the leaves (the check's silver-reference path with the same cut and the
//!   same network, `crate::hybrid::Reference`);
//! - **player**: the same search with the network's values plus [`NoisyLeaf`]'s noise at each
//!   leaf.
//!
//! Everything else (choices, decision loss, bootstrap bounds, bands, per-seat own-team bias) is
//! the ordinary check's.
//!
//! **The noise** ([`perturb`]), additive on the leaf's 4-team money vector `v` (units of 2^-32,
//! summing to exactly 2^32), integer only, with `s` the SD asked for the leaf's race class, made
//! of transfers between pairs of teams:
//! 1. the *live* teams are those with any money chance (`v_t > 0`), `m` of them; with fewer than
//!    two, no noise;
//! 2. for each of the six pairs `(i, j)` (`i < j`, in order) an independent draw
//!    `z = 2·(u_1 + … + u_12) − 12·(2^32 − 1)`, the `u_k` uniform 32-bit whole numbers from the
//!    project's generator (Irwin–Hall: a bell curve of SD 2^33, cut off at six SDs, exactly
//!    symmetric about 0); a pair with a dead team ignores its draw;
//! 3. the transfer `y = trunc(z · s · K_m / 2^65)`, `K_m = round(2^32 / √(m − 1))`, of SD
//!    `s / √(m − 1)`;
//! 4. while `|y|` exceeds `floor(min(v_i, v_j) / (m − 1))`, `y = trunc(7·y / 8)` (a pair with a
//!    nearly dead team moves little; the others are untouched);
//! 5. team `i` gains `y` and team `j` loses it. Each team takes part in `m − 1` transfers, so its
//!    change has SD `s` (before step 4) and is at most its own money in size: the result `v + d`
//!    is a money chance summing to exactly 2^32 with no renormalisation step.
//!
//! Every step maps `z` to `−z` onto `d` to `−d` (the live set and the bounds depend on `v` only,
//! truncation is toward zero, step 4 tests `|y|`), so the noise is exactly zero-mean given the
//! leaf's value and never clipped one-sidedly. Step 4 makes the realised spread smaller than `s`
//! where some team's money is small (races near their end); the run measures what it really was:
//! the realised SD per race class over every team of every distinct leaf a search values
//! ([`NoiseStats`]), reported beside the target, and the SDs asked for are calibrated so the
//! realised ones match the auditor's (docs/m3-hybrid/curse-ensemble.md). The auditor's errors are
//! heavier-tailed than a bell curve (its 99.9th percentile of |error| is about 23% at an overall
//! SD of 1.9%) and correlated between neighbouring positions; this noise matches the SD per
//! class, not the tails or the correlation.
//!
//! **The SD per race class** (`--noise-sd-by-class`, five percentages of money chance, classes
//! 0-4 of `nn::outcome::race_class_of`) is the auditor's measured per-team error SD of the network
//! by race class on a fresh audit shard of positions the hybrid's search reaches.
//!
//! **Seeding**: the noise of a leaf is `Rng::derive(seed, &[NOISE, decision id, world, the leaf's
//! canonical form as nine words])`, so it is a pure function of (seed, decision, world, position):
//! the same in every search that reaches the position in that world, on any thread count and
//! machine, and the search stays deterministic (its tables see one value per position).

use std::collections::BTreeSet;
use std::sync::Mutex;

use engine::rng::Rng;
use equity::MoneyChance;
use serde::Serialize;
use solver::frontier::{CanonPos, Frontier};
use solver::{Leaf, MatchCtx};

/// Seed label of the leaf noise.
const NOISE: u64 = 0x2015E;
/// The race classes (`nn::outcome::N_RACE_CLASSES`).
pub const CLASSES: usize = nn::outcome::N_RACE_CLASSES;
/// `K_m = round(2^32 / √(m − 1))` for `m` = 2, 3, 4 live teams (module notes, step 3).
const SCALE: [i128; 3] = [1 << 32, 3_037_000_500, 2_479_700_525];
/// The six pairs of teams, in order.
const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// The noise of the curse test: its seed and its SD per race class (units of 2^-32).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoiseSpec {
    /// The master seed of the noise.
    pub seed: u64,
    /// The SD of each team's noise asked for, per race class, in units of 2^-32 (the transfers'
    /// scale; after [`NoiseSpec::calibrated`], larger than the target where step 4 shrinks).
    pub sd: [u64; CLASSES],
    /// The SD the realised noise should have, per race class (units of 2^-32): the auditor's.
    pub target: [u64; CLASSES],
    /// The SDs as given (percentages of money chance), for reports.
    pub sd_pct: Vec<String>,
}

impl NoiseSpec {
    /// From `--noise-sd-by-class` (five percentages of money chance, comma-separated, each from 0
    /// to 20) and a seed.
    pub fn parse(text: &str, seed: u64) -> Result<NoiseSpec, String> {
        let parts: Vec<&str> = text.split(',').map(str::trim).collect();
        if parts.len() != CLASSES {
            return Err(format!(
                "--noise-sd-by-class {text:?}: {CLASSES} percentages (race classes 0-4), not {}",
                parts.len()
            ));
        }
        let mut sd = [0u64; CLASSES];
        for (k, p) in parts.iter().enumerate() {
            let ok = !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit() || b == b'.');
            let x: f64 = if ok {
                p.parse().unwrap_or(f64::NAN)
            } else {
                f64::NAN
            };
            if !(x.is_finite() && (0.0..=20.0).contains(&x)) {
                return Err(format!(
                    "--noise-sd-by-class: {p:?} is not a percentage from 0 to 20"
                ));
            }
            // Parsed once, correctly rounded; the integer SD is what every machine uses.
            sd[k] = (x / 100.0 * equity::ONE as f64).round() as u64;
        }
        Ok(NoiseSpec {
            seed,
            sd,
            target: sd,
            sd_pct: parts.iter().map(|s| s.to_string()).collect(),
        })
    }

    /// The spec with the SDs asked for rescaled so the realised SDs of `stats` (a run with this
    /// spec) move to the targets: each class with leaves and a realised SD has its SD multiplied
    /// by target / realised (at most 4 times the target, at most 20% of money chance). Computed
    /// from whole-number sums with IEEE arithmetic (+, ×, ÷, √ only), so the same everywhere.
    pub fn calibrated(&self, stats: &NoiseStats) -> NoiseSpec {
        let mut out = self.clone();
        for k in 0..CLASSES {
            let n = 4 * stats.leaves[k];
            if n == 0 || stats.sum_sq[k] == 0 || self.target[k] == 0 {
                continue;
            }
            let realised = (stats.sum_sq[k] as f64 / n as f64).sqrt();
            let want = self.sd[k] as f64 * self.target[k] as f64 / realised;
            let cap = (4 * self.target[k]).min(equity::ONE / 5) as f64;
            out.sd[k] = want.min(cap).round() as u64;
        }
        out
    }

    /// No noise at all (every SD 0): the player is the truth.
    pub fn is_zero(&self) -> bool {
        self.sd.iter().all(|&s| s == 0)
    }
}

/// The words of a canonical form, for the noise's seed (every field, in a fixed order).
pub fn canon_words(c: &CanonPos) -> [u64; 9] {
    let mut b = [0u8; 72];
    b[0] = c.trump as u8;
    b[1] = c.leader;
    b[2] = c.tricks_done;
    b[3..7].copy_from_slice(&c.scores);
    for t in 0..4 {
        b[7 + t] = u8::from(c.flags[t]);
    }
    b[11..15].copy_from_slice(&c.tricks);
    b[15] = u8::from(c.five_played);
    b[16] = c.end_bonus.unwrap_or(0xFF);
    b[17..17 + solver::frontier::SLOTS].copy_from_slice(&c.slots);
    core::array::from_fn(|i| u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().expect("8 bytes")))
}

/// `v` plus the noise of SD `sd` drawn from `rng` (module notes, steps 1-5); also whether step 5
/// shrank it.
pub fn perturb(v: MoneyChance, sd: u64, rng: &mut Rng) -> (MoneyChance, bool) {
    if sd == 0 || v.0.iter().filter(|&&x| x > 0).count() < 2 {
        return (v, false);
    }
    let z: [i128; 6] = core::array::from_fn(|_| {
        let s: i128 = (0..12).map(|_| i128::from(rng.next_u64() >> 32)).sum();
        2 * s - 12 * i128::from(u32::MAX)
    });
    perturb_with(v, sd, &z)
}

/// Steps 3-5 of the noise (module notes) for the six pairs' draws `z`.
fn perturb_with(v: MoneyChance, sd: u64, z: &[i128; 6]) -> (MoneyChance, bool) {
    let m = v.0.iter().filter(|&&x| x > 0).count();
    if sd == 0 || m < 2 {
        return (v, false);
    }
    let k = SCALE[m - 2];
    let mut d = [0i128; 4];
    let mut shrunk = false;
    for (p, &(i, j)) in PAIRS.iter().enumerate() {
        if v.0[i] == 0 || v.0[j] == 0 {
            continue;
        }
        let bound = u128::from(v.0[i].min(v.0[j]) / (m as u64 - 1));
        let mut y = z[p] * i128::from(sd) * k / (1i128 << 65);
        while y.unsigned_abs() > bound {
            y = y * 7 / 8;
            shrunk = true;
        }
        d[i] += y;
        d[j] -= y;
    }
    let out: [u64; 4] = core::array::from_fn(|t| (i128::from(v.0[t]) + d[t]) as u64);
    let r = MoneyChance::from_raw(out).expect("transfers keep the sum and every share");
    (r, shrunk)
}

/// The realised noise of a run, per race class, over the distinct leaves each world's search
/// valued (each team's change counted once per leaf and world).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NoiseStats {
    /// Leaves noised.
    pub leaves: [u64; CLASSES],
    /// Leaves where step 4 shrank some transfer.
    pub shrunk: [u64; CLASSES],
    /// Sum of the teams' changes (units of 2^-32).
    pub sum: [i128; CLASSES],
    /// Sum of their squares.
    pub sum_sq: [u128; CLASSES],
    /// Sum of their absolute values.
    pub sum_abs: [u128; CLASSES],
}

impl NoiseStats {
    /// Adds another run's counts (sums of whole numbers: the order never matters).
    pub fn add(&mut self, o: &NoiseStats) {
        for k in 0..CLASSES {
            self.leaves[k] += o.leaves[k];
            self.shrunk[k] += o.shrunk[k];
            self.sum[k] += o.sum[k];
            self.sum_sq[k] += o.sum_sq[k];
            self.sum_abs[k] += o.sum_abs[k];
        }
    }

    fn note(&mut self, class: usize, before: &MoneyChance, after: &MoneyChance, shrunk: bool) {
        self.leaves[class] += 1;
        self.shrunk[class] += u64::from(shrunk);
        for t in 0..4 {
            let d = i128::from(after.0[t]) - i128::from(before.0[t]);
            self.sum[class] += d;
            self.sum_sq[class] += (d * d) as u128;
            self.sum_abs[class] += d.unsigned_abs();
        }
    }

    /// The report: per class, the target SD, the realised SD (root mean square of each team's
    /// change), mean, mean |change| and the share shrunk, in percent of money chance.
    pub fn report(&self, spec: &NoiseSpec) -> NoiseReport {
        let unit = equity::ONE as f64 / 100.0;
        let classes = (0..CLASSES)
            .map(|k| {
                let n = (4 * self.leaves[k]).max(1) as f64;
                NoiseClass {
                    class: k as u8,
                    target_sd_pct: spec.target[k] as f64 / unit,
                    asked_sd_pct: spec.sd[k] as f64 / unit,
                    leaves: self.leaves[k],
                    realised_sd_pct: (self.sum_sq[k] as f64 / n).sqrt() / unit,
                    realised_mean_pct: self.sum[k] as f64 / n / unit,
                    realised_mean_abs_pct: self.sum_abs[k] as f64 / n / unit,
                    shrunk_share: self.shrunk[k] as f64 / self.leaves[k].max(1) as f64,
                }
            })
            .collect();
        let tot = |f: &dyn Fn(usize) -> f64| (0..CLASSES).map(f).sum::<f64>();
        let n = (4 * self.leaves.iter().sum::<u64>()).max(1) as f64;
        NoiseReport {
            seed: spec.seed,
            sd_by_class_pct: spec.sd_pct.clone(),
            sd_by_class_units: spec.sd,
            classes,
            realised_sd_pct: (tot(&|k| self.sum_sq[k] as f64) / n).sqrt() / unit,
            leaves: self.leaves.iter().sum(),
        }
    }
}

/// One race class's line of the noise report.
#[derive(Clone, Debug, Serialize)]
pub struct NoiseClass {
    /// The race class.
    pub class: u8,
    /// The SD the realised noise should have (% of money chance): the auditor's.
    pub target_sd_pct: f64,
    /// The SD asked for (the transfers' scale, after any calibration).
    pub asked_sd_pct: f64,
    /// Distinct leaves noised (summed over worlds).
    pub leaves: u64,
    /// The realised SD of each team's change.
    pub realised_sd_pct: f64,
    /// The realised mean change (0 in expectation).
    pub realised_mean_pct: f64,
    /// The realised mean absolute change.
    pub realised_mean_abs_pct: f64,
    /// The share of leaves whose noise was shrunk (a team's money smaller than the noise).
    pub shrunk_share: f64,
}

/// The noise report of a curse run.
#[derive(Clone, Debug, Serialize)]
pub struct NoiseReport {
    /// The noise's seed.
    pub seed: u64,
    /// The SDs per class as given (%).
    pub sd_by_class_pct: Vec<String>,
    /// The SDs per class in units of 2^-32.
    pub sd_by_class_units: [u64; CLASSES],
    /// Per class.
    pub classes: Vec<NoiseClass>,
    /// Over every class.
    pub realised_sd_pct: f64,
    /// Leaves in all.
    pub leaves: u64,
}

/// A frontier evaluator that adds the curse test's noise to `inner`'s values: one per world
/// (decision `id`, world `w`), recording the realised noise of each distinct leaf it values.
pub struct NoisyLeaf<'a> {
    inner: &'a dyn Frontier,
    leaf: &'a (dyn Leaf + Sync),
    spec: &'a NoiseSpec,
    id: u64,
    w: u64,
    state: Mutex<(BTreeSet<CanonPos>, NoiseStats)>,
}

impl<'a> NoisyLeaf<'a> {
    /// The noise of decision `id`, world `w` around `inner`; `leaf` (the equity table) gives the
    /// race class of each position.
    pub fn new(
        inner: &'a dyn Frontier,
        leaf: &'a (dyn Leaf + Sync),
        spec: &'a NoiseSpec,
        id: u64,
        w: u64,
    ) -> Self {
        NoisyLeaf {
            inner,
            leaf,
            spec,
            id,
            w,
            state: Mutex::new((BTreeSet::new(), NoiseStats::default())),
        }
    }

    /// The realised noise of this world's distinct leaves.
    pub fn into_stats(self) -> NoiseStats {
        self.state.into_inner().unwrap_or_else(|e| e.into_inner()).1
    }

    fn noisy(&self, v: MoneyChance, canon: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        let class = usize::from(nn::outcome::race_class_of(canon, ctx, self.leaf));
        let words = canon_words(canon);
        let mut labels = Vec::with_capacity(3 + words.len());
        labels.extend_from_slice(&[NOISE, self.id, self.w]);
        labels.extend_from_slice(&words);
        let mut rng = Rng::derive(self.spec.seed, &labels);
        let (out, shrunk) = perturb(v, self.spec.sd[class], &mut rng);
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if st.0.insert(*canon) {
            st.1.note(class, &v, &out, shrunk);
        }
        out
    }
}

impl Frontier for NoisyLeaf<'_> {
    fn value(&self, canon: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        let v = self.inner.value(canon, ctx);
        self.noisy(v, canon, ctx)
    }

    fn value_at(&self, pos: &solver::Position, canon: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        let v = self.inner.value_at(pos, canon, ctx);
        self.noisy(v, canon, ctx)
    }

    fn prefetch_depth(&self) -> u8 {
        self.inner.prefetch_depth()
    }

    /// Passed on (the network values the batch into its memo; the noise is added when each
    /// position is asked for).
    fn prefetch(&self, pos: &[CanonPos], ctx: &MatchCtx) {
        self.inner.prefetch(pos, ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use equity::ONE;

    #[test]
    fn noise_sums_to_one_is_symmetric_and_has_the_asked_sd() {
        let v = MoneyChance::from_raw([ONE / 4, ONE / 4, ONE / 4, ONE / 4]).expect("valid");
        let sd = ONE / 50; // 2%
        let (mut n, mut s, mut s2) = (0f64, 0f64, 0f64);
        for i in 0..20_000u64 {
            let mut rng = Rng::derive(3, &[i]);
            let (m, shrunk) = perturb(v, sd, &mut rng);
            assert!(!shrunk);
            assert_eq!(m.0.iter().sum::<u64>(), ONE);
            for t in 0..4 {
                let d = m.0[t] as f64 - v.0[t] as f64;
                n += 1.0;
                s += d;
                s2 += d * d;
            }
        }
        let rms = (s2 / n).sqrt() / sd as f64;
        assert!((rms - 1.0).abs() < 0.02, "realised SD / asked {rms}");
        assert!((s / n).abs() / (sd as f64) < 0.02);
        // Zero SD: unchanged.
        let mut rng = Rng::new(1);
        assert_eq!(perturb(v, 0, &mut rng), (v, false));
    }

    #[test]
    fn noise_stays_a_money_chance_and_spares_dead_teams() {
        // Team 3 holds nothing: it takes no noise. Team 1 holds 10 units: its transfers are
        // bounded by 10 / 2 each, the others' are not.
        let v = MoneyChance::from_raw([ONE / 2, 10, ONE / 2 - 10, 0]).expect("valid");
        let (mut moved, mut shrunk_any) = (0, false);
        for i in 0..2_000u64 {
            let mut rng = Rng::derive(9, &[i]);
            let (m, shrunk) = perturb(v, ONE / 10, &mut rng);
            assert_eq!(m.0.iter().sum::<u64>(), ONE);
            assert_eq!(m.0[3], 0);
            assert!(m.0[1] <= 20);
            moved += usize::from(m.0[0] != v.0[0]);
            shrunk_any |= shrunk;
        }
        assert!(moved > 1_900 && shrunk_any);
        // A draw and its mirror image give opposite changes (the noise is odd in its draw),
        // also when step 4 shrinks it.
        let w = MoneyChance::from_raw([ONE / 8, ONE / 8, ONE / 4 - 5, ONE / 2 + 5]).expect("ok");
        let u = MoneyChance::from_raw([ONE / 2, 7, 0, ONE / 2 - 7]).expect("valid");
        for (i, sd, x) in [
            (1u64, ONE / 25, w),
            (2, ONE / 9, w),
            (3, ONE / 3, w),
            (4, ONE / 9, u),
        ] {
            let z: [i128; 6] = core::array::from_fn(|k| {
                i128::from(i * 7_000_000_000 + k as u64 * 3_100_000_000) * (k as i128 - 2)
            });
            let neg = z.map(|x| -x);
            let (p, sp) = perturb_with(x, sd, &z);
            let (q, sq) = perturb_with(x, sd, &neg);
            assert_eq!(sp, sq);
            assert_ne!(p, x);
            for t in 0..4 {
                assert_eq!(
                    p.0[t] as i128 - x.0[t] as i128,
                    x.0[t] as i128 - q.0[t] as i128
                );
            }
        }
        // Edge: only one live team: unchanged.
        let mut a = Rng::derive(4, &[1]);
        let c = MoneyChance::certain(2);
        assert_eq!(perturb(c, ONE / 25, &mut a), (c, false));
    }

    #[test]
    fn spec_parses_and_refuses() {
        let s = NoiseSpec::parse("0.978, 0.918,2.010,1.519,4.050", 5).expect("valid");
        assert_eq!(s.sd[0], (0.00978f64 * ONE as f64).round() as u64);
        assert!(!s.is_zero());
        assert!(NoiseSpec::parse("0,0,0,0,0", 1).expect("valid").is_zero());
        assert!(NoiseSpec::parse("1,2,3,4", 1).is_err());
        assert!(NoiseSpec::parse("1,2,3,4,-5", 1).is_err());
        assert!(NoiseSpec::parse("1,2,3,4,50", 1).is_err());
    }
}
