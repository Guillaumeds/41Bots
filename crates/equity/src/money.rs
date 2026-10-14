//! The money-chance type (docs/M1-SPEC.md §6.1, PLAN §5.3 and amendment A8).
//!
//! A [`MoneyChance`] is four whole numbers, one per team, each a chance of winning the match (the
//! money) in units of 2^-32, always summing to exactly [`ONE`]. Every constructor normalises by
//! largest-remainder rounding [round every share down, then give the few units still missing to
//! the shares that lost the most in rounding], so the sum is exact on every machine, and all
//! accumulation is in `u128`, so no step can overflow or depend on floating point.

use core::fmt;

/// Certainty: a chance of 1 in units of 2^-32.
pub const ONE: u64 = 1 << 32;

/// Each team's chance of the money, in units of 2^-32. Invariant: the four entries sum to exactly
/// [`ONE`].
///
/// The field is public so that values can be pattern-matched and stored, but only this crate
/// builds values, and every constructor ([`MoneyChance::certain`], [`MoneyChance::from_weights`],
/// [`MoneyChance::mix`], [`MoneyChance::mix_wide`], [`MoneyChance::rotate`],
/// [`MoneyChance::from_raw`]) normalises or checks, so the sum is always exactly `ONE`. Every
/// method also re-checks the invariant in debug builds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MoneyChance(pub [u64; 4]);

/// Floor shares and remainders of `num[t] / den`, then largest-remainder rounding so the shares
/// sum to `target`. Ties between equal remainders go to the lower index (a fixed rule, so the
/// result is the same everywhere).
///
/// `num` must sum to `target * den`, `den > 0`.
fn largest_remainder<const N: usize>(q: [u64; N], r: [u128; N], target: u64) -> [u64; N] {
    let mut out = q;
    let floor_sum: u64 = q.iter().sum();
    debug_assert!(floor_sum <= target);
    let missing = (target - floor_sum) as usize;
    debug_assert!(missing < N.max(1));
    if missing == 0 {
        return out;
    }
    // Indices sorted by remainder, largest first, ties to the lower index.
    let mut order: [usize; N] = core::array::from_fn(|i| i);
    order.sort_by(|&a, &b| r[b].cmp(&r[a]).then(a.cmp(&b)));
    for &i in order.iter().take(missing) {
        out[i] += 1;
    }
    out
}

/// `floor(x * 2^32 / den)` and the remainder, for `x <= den`, without any intermediate overflow
/// (bit-by-bit long division; used only off the hot path).
fn scaled_share(x: u128, den: u128) -> (u64, u128) {
    debug_assert!(x <= den && den > 0);
    if x == den {
        return (ONE, 0);
    }
    let mut q: u64 = 0;
    let mut r = x;
    for _ in 0..32 {
        q <<= 1;
        // Double r modulo den without overflow: 2r >= den  <=>  r >= den - r.
        if r >= den - r {
            r -= den - r;
            q |= 1;
        } else {
            r <<= 1;
        }
    }
    (q, r)
}

/// Normalise non-negative weights to fixed-point shares summing to exactly `ONE` (2^32), by
/// largest-remainder rounding. Panics if every weight is zero or the total overflows `u128`.
pub(crate) fn normalise<const N: usize>(w: [u128; N]) -> [u64; N] {
    let mut total: u128 = 0;
    for &x in &w {
        total = total
            .checked_add(x)
            .expect("weights too large: their total overflows u128");
    }
    assert!(total > 0, "cannot normalise weights that are all zero");
    let mut q = [0u64; N];
    let mut r = [0u128; N];
    for i in 0..N {
        let (qi, ri) = scaled_share(w[i], total);
        q[i] = qi;
        r[i] = ri;
    }
    largest_remainder(q, r, ONE)
}

impl MoneyChance {
    /// The team wins the money for certain.
    ///
    /// # Panics
    /// If `team` is not 0..=3.
    pub fn certain(team: u8) -> Self {
        assert!(team < 4, "no such team");
        let mut v = [0u64; 4];
        v[usize::from(team)] = ONE;
        MoneyChance(v)
    }

    /// Each team equally likely (a quarter each).
    pub fn even() -> Self {
        MoneyChance([ONE / 4; 4])
    }

    /// Check raw values (for example read back from a file): `Some` only if they sum to exactly
    /// `ONE`.
    pub fn from_raw(v: [u64; 4]) -> Option<Self> {
        let mut sum: u64 = 0;
        for &x in &v {
            sum = sum.checked_add(x)?;
        }
        (sum == ONE).then_some(MoneyChance(v))
    }

    /// Proportional to the weights, normalised by largest-remainder rounding so the sum is
    /// exactly `ONE`.
    ///
    /// # Panics
    /// If all weights are zero, or their total overflows `u128`.
    pub fn from_weights(w: [u128; 4]) -> Self {
        let v = MoneyChance(normalise(w));
        v.check();
        v
    }

    /// The weighted average `Σ w_i·c_i / Σ w_i` (weights in any common unit, for example 2^-32),
    /// accumulated in `u128` and rounded by largest remainder so the sum is exactly `ONE`.
    /// Terms with weight zero are ignored.
    ///
    /// # Panics
    /// If the total weight is zero.
    pub fn mix(terms: &[(u64, MoneyChance)]) -> Self {
        Self::mix_core(terms.iter().map(|&(w, c)| (u128::from(w), c)))
    }

    /// Like [`MoneyChance::mix`] with wider weights, used when a weight is a product of two
    /// fixed-point probabilities (units of 2^-64). The total weight must stay below 2^96 so the
    /// `u128` accumulators cannot overflow.
    ///
    /// # Panics
    /// If the total weight is zero or not below 2^96.
    pub fn mix_wide(terms: &[(u128, MoneyChance)]) -> Self {
        Self::mix_core(terms.iter().copied())
    }

    fn mix_core(terms: impl Iterator<Item = (u128, MoneyChance)>) -> Self {
        let mut num = [0u128; 4];
        let mut den: u128 = 0;
        for (w, c) in terms {
            c.check();
            if w == 0 {
                continue;
            }
            den += w;
            assert!(den < 1u128 << 96, "mix weights too large");
            for (n, &x) in num.iter_mut().zip(&c.0) {
                *n += w * u128::from(x);
            }
        }
        assert!(den > 0, "mix needs at least one positive weight");
        let mut q = [0u64; 4];
        let mut r = [0u128; 4];
        if den.is_power_of_two() {
            // The usual case in the sweep (weights are exact products of fixed-point
            // probabilities summing to 2^64): division is a shift, with the same result.
            let k = den.trailing_zeros();
            let mask = den - 1;
            for t in 0..4 {
                q[t] = (num[t] >> k) as u64;
                r[t] = num[t] & mask;
            }
        } else {
            for t in 0..4 {
                q[t] = (num[t] / den) as u64;
                r[t] = num[t] % den;
            }
        }
        let v = MoneyChance(largest_remainder(q, r, ONE));
        v.check();
        v
    }

    /// Expected money per team for a stake per losing player (H6): the winning team receives six
    /// stakes (three losing teams of two players) and every losing team pays two, so team `t`
    /// expects `(8·p_t − 2)·stake`, rounded to the nearest whole unit of `stake` (halves away from
    /// zero), in integers only. Pass the stake in cents for cent precision. Each value is within
    /// half a unit of the exact one, so the four can miss summing to zero by at most 2 units.
    pub fn expected_money(&self, stake: i32) -> [i64; 4] {
        self.check();
        let stake = i128::from(stake);
        let half: i128 = 1 << 31;
        core::array::from_fn(|t| {
            // Exact value: n / 2^32 with n = (8·c_t − 2·2^32)·stake.
            let n = (8 * i128::from(self.0[t]) - 2 * i128::from(ONE)) * stake;
            let rounded = if n >= 0 {
                (n + half) >> 32
            } else {
                -((-n + half) >> 32)
            };
            rounded as i64
        })
    }

    /// Relabel the teams: the chance of team `t` becomes the chance of team `(t + by) mod 4`.
    pub fn rotate(&self, by: u8) -> Self {
        self.check();
        let by = usize::from(by % 4);
        let mut v = [0u64; 4];
        for t in 0..4 {
            v[(t + by) % 4] = self.0[t];
        }
        MoneyChance(v)
    }

    /// The chance of one team, in units of 2^-32.
    pub fn get(&self, team: u8) -> u64 {
        self.check();
        self.0[usize::from(team % 4)]
    }

    /// The four chances as decimals, for printing only (never for anything that is stored).
    pub fn to_f64_for_display(&self) -> [f64; 4] {
        self.check();
        self.0.map(|x| x as f64 / ONE as f64)
    }

    #[inline]
    fn check(&self) {
        debug_assert_eq!(
            self.0.iter().map(|&x| u128::from(x)).sum::<u128>(),
            u128::from(ONE),
            "money chance must sum to exactly ONE"
        );
    }
}

impl fmt::Display for MoneyChance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = self.to_f64_for_display();
        write!(
            f,
            "A {:.9}  B {:.9}  C {:.9}  D {:.9}",
            p[0], p[1], p[2], p[3]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::rng::Rng;

    fn sum(c: &MoneyChance) -> u128 {
        c.0.iter().map(|&x| u128::from(x)).sum()
    }

    #[test]
    fn ten_thousand_random_weight_vectors_sum_to_one() {
        let mut rng = Rng::derive(41, &[0x3017]);
        for i in 0..10_000u32 {
            // Mix magnitudes: tiny, 32-bit, 64-bit and huge 126-bit weights, and zeros.
            let w: [u128; 4] = core::array::from_fn(|_| match rng.bounded(5) {
                0 => 0,
                1 => u128::from(rng.bounded(10)),
                2 => u128::from(rng.next_u64() >> 32),
                3 => u128::from(rng.next_u64()),
                _ => (u128::from(rng.next_u64()) << 62) | u128::from(rng.next_u64()),
            });
            if w.iter().all(|&x| x == 0) {
                continue;
            }
            let c = MoneyChance::from_weights(w);
            assert_eq!(sum(&c), u128::from(ONE), "vector {i}: {w:?}");
            // Every share is the floor or the ceiling of the exact share.
            let total: u128 = w.iter().sum();
            for (&wt, &ct) in w.iter().zip(&c.0) {
                let (fl, rem) = scaled_share(wt, total);
                assert!(ct == fl || (rem > 0 && ct == fl + 1), "vector {i}");
            }
        }
    }

    #[test]
    fn from_weights_simple_cases() {
        assert_eq!(
            MoneyChance::from_weights([1, 0, 0, 0]),
            MoneyChance::certain(0)
        );
        assert_eq!(MoneyChance::from_weights([1, 1, 1, 1]), MoneyChance::even());
        // Thirds: 2^32 = 3·1431655765 + 1; the missing unit goes to the lowest index.
        let c = MoneyChance::from_weights([0, 1, 1, 1]);
        assert_eq!(c.0, [0, 1_431_655_766, 1_431_655_765, 1_431_655_765]);
        // Huge weights do not overflow.
        let c = MoneyChance::from_weights([u128::MAX / 4, u128::MAX / 4, 0, 0]);
        assert_eq!(c.0, [ONE / 2, ONE / 2, 0, 0]);
    }

    #[test]
    fn mix_matches_exact_average_and_sums_to_one() {
        let mut rng = Rng::derive(41, &[0x3018]);
        for _ in 0..2_000 {
            let n = 1 + rng.bounded(8) as usize;
            let terms: Vec<(u64, MoneyChance)> = (0..n)
                .map(|_| {
                    let w = rng.next_u64() >> rng.bounded(64);
                    let c = MoneyChance::from_weights(core::array::from_fn(|_| {
                        u128::from(rng.next_u64() >> 1) + 1
                    }));
                    (w, c)
                })
                .collect();
            if terms.iter().all(|&(w, _)| w == 0) {
                continue;
            }
            let m = MoneyChance::mix(&terms);
            assert_eq!(sum(&m), u128::from(ONE));
            let den: u128 = terms.iter().map(|&(w, _)| u128::from(w)).sum();
            for t in 0..4 {
                let num: u128 = terms
                    .iter()
                    .map(|&(w, c)| u128::from(w) * u128::from(c.0[t]))
                    .sum();
                let fl = (num / den) as u64;
                assert!(m.0[t] == fl || m.0[t] == fl + 1);
            }
        }
    }

    #[test]
    fn mix_power_of_two_path_equals_division_path() {
        // The same mixture with weights scaled so the total is, or is not, a power of two.
        let a = MoneyChance::from_weights([3, 5, 7, 11]);
        let b = MoneyChance::from_weights([13, 1, 2, 9]);
        let pow = MoneyChance::mix_wide(&[(1u128 << 62, a), (3u128 << 62, b)]);
        let div = MoneyChance::mix(&[(1, a), (3, b)]);
        assert_eq!(pow, div);
        let div3 = MoneyChance::mix(&[(3, a), (9, b)]);
        assert_eq!(pow, div3);
    }

    #[test]
    fn mix_of_one_term_is_identity_and_zero_weights_ignored() {
        let a = MoneyChance::from_weights([3, 5, 7, 11]);
        let b = MoneyChance::certain(2);
        assert_eq!(MoneyChance::mix(&[(17, a)]), a);
        assert_eq!(MoneyChance::mix(&[(0, b), (5, a)]), a);
        assert_eq!(MoneyChance::mix(&[(1, b), (1, b)]), b);
    }

    #[test]
    #[should_panic(expected = "positive weight")]
    fn mix_with_zero_total_panics() {
        let _ = MoneyChance::mix(&[(0, MoneyChance::even())]);
    }

    #[test]
    fn rotate_relabels_teams() {
        let a = MoneyChance::from_weights([1, 2, 3, 4]);
        let r = a.rotate(1);
        assert_eq!(r.0, [a.0[3], a.0[0], a.0[1], a.0[2]]);
        assert_eq!(a.rotate(4), a);
        assert_eq!(a.rotate(1).rotate(3), a);
        assert_eq!(MoneyChance::certain(1).rotate(2), MoneyChance::certain(3));
    }

    #[test]
    fn expected_money_matches_h6() {
        // Certain winner: +6 stakes; certain losers: −2 stakes each (pub stake €5: +30, −10).
        assert_eq!(
            MoneyChance::certain(1).expected_money(5),
            [-10, 30, -10, -10]
        );
        // Even chances: everyone expects nothing.
        assert_eq!(MoneyChance::even().expected_money(500), [0, 0, 0, 0]);
        // Halves round away from zero: p = 1/16 gives (0.5 − 2)·1 = −1.5 → −2; p = 5/16 → +0.5 → 1.
        let c = MoneyChance([ONE / 16, 5 * (ONE / 16), 5 * (ONE / 16), 5 * (ONE / 16)]);
        assert_eq!(c.expected_money(1), [-2, 1, 1, 1]);
        // Always the nearest whole unit, so within 2 units of zero-sum.
        let mut rng = Rng::derive(41, &[0x3019]);
        for _ in 0..2_000 {
            let c = MoneyChance::from_weights(core::array::from_fn(|_| {
                u128::from(rng.next_u64() >> rng.bounded(64))
            }));
            let stake = 1 + rng.bounded(1000) as i32;
            let m = c.expected_money(stake);
            assert!(m.iter().sum::<i64>().abs() <= 2);
            for t in 0..4 {
                let exact = (8.0 * c.0[t] as f64 / ONE as f64 - 2.0) * f64::from(stake);
                assert!(
                    (m[t] as f64 - exact).abs() <= 0.5 + 1e-9,
                    "{m:?} vs {exact}"
                );
            }
        }
    }

    #[test]
    fn from_raw_checks_the_sum() {
        assert!(MoneyChance::from_raw([ONE, 0, 0, 0]).is_some());
        assert!(MoneyChance::from_raw([ONE, 1, 0, 0]).is_none());
        assert!(MoneyChance::from_raw([u64::MAX, 2, 0, 0]).is_none());
    }
}
