//! From four logits to a money chance, by a fixed integer rule (docs/M3-HYBRID-SPEC.md §6): the
//! softmax `p_t = e^(z_t) / Σ_u e^(z_u)`, computed as follows (the Python side in `python/nn`
//! does exactly the same integer steps):
//!
//! 1. Logits `z_t` are integers in units of 2^-24 ([`crate::mlp::LOGIT_SHIFT`]); let
//!    `d_t = max_u z_u − z_t ≥ 0`.
//! 2. `e^(−d) = 2^(−d·log2 e)`: `y = d × LOG2E_Q32` (log2 e rounded to 32 fraction bits) has 56
//!    fraction bits; its integer part `n = y >> 56`, its top 24 fraction bits
//!    `f = (y >> 32) mod 2^24`.
//! 3. The weight is `2^62 · 2^(−f / 2^24)`, built from `w = 2^62` by multiplying in, for every
//!    bit `m` of `f` from the highest (23) to the lowest (0) that is set,
//!    `R[24 − m] = 2^(−1 / 2^(24−m))` in units of 2^-62 (`w ← (w × R) >> 62`, floor), then
//!    shifted right by `n` (0 when `n ≥ 63`). The roots are computed by integer square roots:
//!    `R[0] = 2^61` (one half), `R[k] = isqrt(R[k−1] · 2^62)` (floor), so no table and no
//!    floating point is involved anywhere.
//! 4. The four weights (each at most 2^62, the largest exactly 2^62) are normalised by
//!    [`MoneyChance::from_weights`]: every share is `floor(w_t · 2^32 / Σw)`, and the few units
//!    still missing go to the largest remainders, ties to the lower team index. The shares sum to
//!    exactly 2^32.
//!
//! Accuracy: the 24-bit fraction and the floors lose at most a few parts in 10^7 of each weight,
//! far below the network's own error; the measured error (spec §3) includes it.

use std::sync::OnceLock;

use equity::MoneyChance;

/// log2(e) in units of 2^-32, rounded: 6196328018.7194806… → 6196328019.
pub const LOG2E_Q32: u64 = 6_196_328_019;

/// `R[k] = 2^(−1/2^k)` in units of 2^-62 (floor), `k = 0..=24`, by integer square roots.
pub fn roots() -> &'static [u64; 25] {
    static R: OnceLock<[u64; 25]> = OnceLock::new();
    R.get_or_init(|| {
        let mut r = [0u64; 25];
        r[0] = 1 << 61;
        for k in 1..25 {
            r[k] = (u128::from(r[k - 1]) << 62).isqrt() as u64;
        }
        r
    })
}

/// `2^62 · e^(−d / 2^24)` by the integer rule above (`d ≥ 0` in units of 2^-24).
///
/// The running weight is held in a `u64` (it starts at `2^62` and each step multiplies it by a
/// root below `2^62` and divides by `2^62`, so it never grows), and each step is one 64 × 64 →
/// 128-bit multiply; the set bits of `f` are visited from the highest down, as in the rule. The
/// same integers as [`exp_weight_reference`] (tested).
pub fn exp_weight(d: u64) -> u64 {
    let y = u128::from(d) * u128::from(LOG2E_Q32);
    let n = y >> 56;
    if n >= 63 {
        return 0;
    }
    let mut f = ((y >> 32) & 0xFF_FFFF) as u32;
    let r = roots();
    let mut w: u64 = 1 << 62;
    while f != 0 {
        let m = 31 - f.leading_zeros();
        f &= !(1 << m);
        w = ((u128::from(w) * u128::from(r[24 - m as usize])) >> 62) as u64;
    }
    w >> n
}

/// [`exp_weight`] written as the rule reads (the plain reference for tests).
pub fn exp_weight_reference(d: u64) -> u64 {
    let y = u128::from(d) * u128::from(LOG2E_Q32);
    let n = y >> 56;
    if n >= 63 {
        return 0;
    }
    let f = ((y >> 32) & 0xFF_FFFF) as u32;
    let r = roots();
    let mut w: u128 = 1 << 62;
    for m in (0..24).rev() {
        if (f >> m) & 1 == 1 {
            w = (w * u128::from(r[24 - m])) >> 62;
        }
    }
    (w >> n) as u64
}

/// The money chance of four logits (units of 2^-24; the first four entries of `z`): the integer
/// softmax above.
pub fn money_from_logits(z: &[i64]) -> MoneyChance {
    let z = &z[..4];
    let m = *z.iter().max().expect("four logits");
    let w: [u128; 4] = core::array::from_fn(|t| u128::from(exp_weight((m - z[t]) as u64)));
    MoneyChance::from_weights(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use equity::ONE;

    #[test]
    fn roots_are_close_to_the_true_values() {
        let r = roots();
        for (k, &x) in r.iter().enumerate() {
            let exact = libm_free_pow2(-1.0 / f64::from(1u32 << k));
            let got = x as f64 / (1u64 << 62) as f64;
            assert!((got - exact).abs() < 1e-15, "k {k}: {got} vs {exact}");
        }
        assert_eq!(r[1], 3_260_954_456_333_195_553);
    }

    /// 2^x for small |x| by a series (test only; never used for a value).
    fn libm_free_pow2(x: f64) -> f64 {
        let y = x * core::f64::consts::LN_2;
        let (mut term, mut sum) = (1.0f64, 1.0f64);
        for i in 1..30 {
            term *= y / f64::from(i);
            sum += term;
        }
        sum
    }

    #[test]
    fn exp_weight_matches_the_exponential() {
        assert_eq!(exp_weight(0), 1 << 62);
        for d in [1u64, 1000, 1 << 20, 1 << 24, 3 << 24, 10 << 24, 40 << 24] {
            let exact = (-(d as f64) / (1u64 << 24) as f64).exp();
            let got = exp_weight(d) as f64 / (1u64 << 62) as f64;
            assert!(
                (got - exact).abs() <= exact * 2e-7 + 1e-18,
                "d {d}: {got} vs {exact}"
            );
        }
        assert_eq!(exp_weight(u64::from(u32::MAX) << 24), 0);
    }

    #[test]
    fn exp_weight_equals_the_reference() {
        let mut sm = engine::rng::SplitMix64::new(3);
        for k in 0..200_000u64 {
            // Every scale: small differences, around the cut-off at n = 63, and huge ones.
            let d = match k % 4 {
                0 => sm.next_u64() % (1 << 24),
                1 => sm.next_u64() % (64 << 24),
                2 => sm.next_u64() >> (sm.next_u64() % 64),
                _ => k,
            };
            assert_eq!(exp_weight(d), exp_weight_reference(d), "d {d}");
        }
        for d in [0, 1, u64::MAX, (43 << 24) + 7, 1 << 40] {
            assert_eq!(exp_weight(d), exp_weight_reference(d), "d {d}");
        }
    }

    #[test]
    fn softmax_sums_to_one_and_follows_the_logits() {
        let m = money_from_logits(&[0, 0, 0, 0]);
        assert_eq!(m, MoneyChance::even());
        let m = money_from_logits(&[5 << 24, 0, 0, 0]);
        let p0 = m.0[0] as f64 / ONE as f64;
        let e5 = 5f64.exp();
        assert!((p0 - e5 / (e5 + 3.0)).abs() < 1e-6);
        let m = money_from_logits(&[i64::MAX / 2, -(i64::MAX / 2), 0, 0]);
        assert_eq!(m, MoneyChance::certain(0));
        // Shifting every logit by the same amount changes nothing.
        let a = money_from_logits(&[123_456, -7_000_000, 33, 1 << 26]);
        let b = money_from_logits(&[123_456 + 99, -7_000_000 + 99, 33 + 99, (1 << 26) + 99]);
        assert_eq!(a, b);
    }
}
