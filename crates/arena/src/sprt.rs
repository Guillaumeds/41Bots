//! The generalised sequential probability ratio test on block scores (docs/M1-SPEC.md §5.2), in
//! the Fishtest normal-approximation form:
//! `llr = (n/2) · ln( ((μ − s0)² + σ²) / ((μ − s1)² + σ²) )`.
//!
//! Floats are used here only, for the report and the stop decision; their inputs are the exact
//! whole-number block scores summed in block order, so the decision is the same on every machine
//! and for any thread count.
//!
//! Choices where the spec is silent: σ² is the maximum-likelihood variance (divide by n, as
//! Fishtest does); a variance of exactly 0 is floored at 1e-18 so the formula stays finite.

use serde::{Deserialize, Serialize};

/// Minimum number of blocks before any decision.
pub const MIN_BLOCKS: u64 = 20;

/// Expected-duration constant of the worst case (docs/M1-SPEC.md §5.2).
pub const WORST_CASE_FACTOR: f64 = 8.67;

/// The test's settings, in money-chance units.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SprtParams {
    /// Null hypothesis: A's edge is `s0`.
    pub s0: f64,
    /// Alternative: A's edge is `s1`.
    pub s1: f64,
    /// False-pass rate.
    pub alpha: f64,
    /// False-fail rate.
    pub beta: f64,
}

impl SprtParams {
    /// The gainer test `[0, 0.02]`, α = β = 0.05.
    pub const GAINER: SprtParams = SprtParams {
        s0: 0.0,
        s1: 0.02,
        alpha: 0.05,
        beta: 0.05,
    };
    /// The non-regression test `[−0.015, 0.005]`, α = β = 0.05.
    pub const NON_REGRESSION: SprtParams = SprtParams {
        s0: -0.015,
        s1: 0.005,
        alpha: 0.05,
        beta: 0.05,
    };

    /// Lower bound `ln(β/(1−α))`.
    pub fn lower(&self) -> f64 {
        (self.beta / (1.0 - self.alpha)).ln()
    }
    /// Upper bound `ln((1−β)/α)`.
    pub fn upper(&self) -> f64 {
        ((1.0 - self.beta) / self.alpha).ln()
    }
    /// Default hard cap: 5 × 8.67 (σ_b/Δ)², Δ = s1 − s0, at least [`MIN_BLOCKS`].
    pub fn default_max_blocks(&self, sigma_b: f64) -> u64 {
        let d = self.s1 - self.s0;
        let x = 5.0 * WORST_CASE_FACTOR * (sigma_b / d) * (sigma_b / d);
        (x.ceil() as u64).max(MIN_BLOCKS)
    }
}

/// The outcome of a test.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    /// Still running (only inside the loop).
    Continue,
    /// LLR reached the upper bound: A passes (H1).
    Accept,
    /// LLR reached the lower bound: A fails (H0).
    Reject,
    /// The hard cap was reached first.
    Inconclusive,
}

/// Running exact sums of block scores (whole-number units).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Moments {
    /// Blocks.
    pub n: u64,
    /// Σ x.
    pub sum: i128,
    /// Σ x².
    pub sum_sq: i128,
}

impl Moments {
    /// Add one block score.
    pub fn push(&mut self, x: i64) {
        self.n += 1;
        self.sum += i128::from(x);
        self.sum_sq += i128::from(x) * i128::from(x);
    }
    /// Mean and maximum-likelihood variance, scaled by `1/unit`.
    pub fn mean_var(&self, unit: f64) -> (f64, f64) {
        if self.n == 0 {
            return (0.0, 0.0);
        }
        let n = self.n as i128;
        let mean = self.sum as f64 / self.n as f64 / unit;
        // n·Σx² − (Σx)², exact, then divide.
        let var_num = n * self.sum_sq - self.sum * self.sum;
        let var = var_num as f64 / (self.n as f64 * self.n as f64) / (unit * unit);
        (mean, var)
    }
}

/// The log-likelihood ratio for `n` blocks with mean `mu` and variance `var`.
pub fn llr(n: u64, mu: f64, var: f64, s0: f64, s1: f64) -> f64 {
    let v = var.max(1e-18);
    (n as f64 / 2.0) * (((mu - s0) * (mu - s0) + v) / ((mu - s1) * (mu - s1) + v)).ln()
}

/// The decision after a block. `reject_only` is stage 1 (it can only reject).
pub fn decide(n: u64, l: f64, p: &SprtParams, reject_only: bool, max_blocks: u64) -> Decision {
    if n >= MIN_BLOCKS {
        if l <= p.lower() {
            return Decision::Reject;
        }
        if !reject_only && l >= p.upper() {
            return Decision::Accept;
        }
    }
    if n >= max_blocks {
        return Decision::Inconclusive;
    }
    Decision::Continue
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_are_fishtest() {
        let p = SprtParams::GAINER;
        assert!((p.upper() - 2.944_438_979).abs() < 1e-8);
        assert!((p.lower() + 2.944_438_979).abs() < 1e-8);
    }

    #[test]
    fn llr_sign_and_zero() {
        assert!(llr(100, 0.03, 0.01, 0.0, 0.02) > 0.0);
        assert!(llr(100, -0.01, 0.01, 0.0, 0.02) < 0.0);
        assert!(llr(100, 0.01, 0.01, 0.0, 0.02).abs() < 1e-12);
    }

    #[test]
    fn min_blocks_and_cap() {
        let p = SprtParams::GAINER;
        assert_eq!(decide(19, 100.0, &p, false, 1000), Decision::Continue);
        assert_eq!(decide(20, 100.0, &p, false, 1000), Decision::Accept);
        assert_eq!(decide(20, 100.0, &p, true, 1000), Decision::Continue);
        assert_eq!(decide(20, -100.0, &p, true, 1000), Decision::Reject);
        assert_eq!(decide(1000, 0.0, &p, false, 1000), Decision::Inconclusive);
        assert_eq!(p.default_max_blocks(0.1), 1084);
    }

    #[test]
    fn moments_exact() {
        let mut m = Moments::default();
        for x in [1i64, 2, 3, 4] {
            m.push(x);
        }
        let (mu, v) = m.mean_var(1.0);
        assert_eq!(mu, 2.5);
        assert_eq!(v, 1.25);
    }
}
