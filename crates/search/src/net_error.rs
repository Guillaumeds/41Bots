//! `search net-error`: a network's value error against the exact values of a stored shard's rows,
//! on the rows a G2 model flags at a threshold (the play rule's cut): the comparison for the
//! specialist's held-out numbers (docs/m3-hybrid/specialist.md).
//!
//! Each row's exact value is in the encoder's rotated team order (`Row::value`), so the network's
//! answer (real team order, `Oracle::eval`) is rotated the same way the risk inputs rotate it
//! (`MoneyChance::rotate` by the encoder's rotation). An error is (network − exact) for one team,
//! in units of the whole match (2^32 = 1). The statistics follow the trainer's held-out report:
//! the bias per team, the standard deviation over all rows × teams, the mean and 99.9th percentile
//! of the absolute error, the largest absolute error, and the same SD per race class.

use std::collections::BTreeMap;

use equity::money::ONE;
use nn::shard::Row;

/// One row's error for each of the four teams (encoder order), with its race class.
pub type RowError = (u8, [f64; 4]);

/// The statistics of a set of row errors.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Stats {
    /// Rows.
    pub rows: usize,
    /// Mean error per team (encoder order).
    pub bias: [f64; 4],
    /// Standard deviation over all rows × teams.
    pub sd: f64,
    /// Mean absolute error over all rows × teams.
    pub mean_abs: f64,
    /// 99.9th percentile of the absolute error.
    pub p999_abs: f64,
    /// Largest absolute error.
    pub max_abs: f64,
    /// Race class → (rows, SD over that class's rows × teams).
    pub by_class: BTreeMap<u8, (usize, f64)>,
}

/// The error of each row: the network's value (real order, from `value_of`) rotated into the
/// encoder's order, minus the row's exact value, per team, in whole-match units.
pub fn row_error(row: &Row, network_value: equity::money::MoneyChance, rotation: u8) -> RowError {
    let v = network_value.rotate(rotation);
    let e = core::array::from_fn(|t| {
        (v.get(t as u8) as f64 - row.value.get(t as u8) as f64) / ONE as f64
    });
    (row.race_class(), e)
}

/// The statistics of `errs` (empty input: zeros).
pub fn stats(errs: &[RowError]) -> Stats {
    let n = errs.len() * 4;
    let all: Vec<f64> = errs.iter().flat_map(|(_, e)| e.iter().copied()).collect();
    let sd = |xs: &[f64]| {
        if xs.is_empty() {
            return 0.0;
        }
        let m = xs.iter().sum::<f64>() / xs.len() as f64;
        (xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64).sqrt()
    };
    let bias: [f64; 4] = core::array::from_fn(|t| {
        if errs.is_empty() {
            0.0
        } else {
            errs.iter().map(|(_, e)| e[t]).sum::<f64>() / errs.len() as f64
        }
    });
    let mut abs: Vec<f64> = all.iter().map(|x| x.abs()).collect();
    abs.sort_by(|a, b| a.total_cmp(b));
    let p999 = if abs.is_empty() {
        0.0
    } else {
        abs[((abs.len() as f64 * 0.999).ceil() as usize).clamp(1, abs.len()) - 1]
    };
    let mut classes: BTreeMap<u8, Vec<f64>> = BTreeMap::new();
    for (c, e) in errs {
        classes.entry(*c).or_default().extend_from_slice(e);
    }
    let by_class = classes
        .into_iter()
        .map(|(c, xs)| (c, (xs.len() / 4, sd(&xs))))
        .collect();
    Stats {
        rows: errs.len(),
        bias,
        sd: sd(&all),
        mean_abs: if n == 0 {
            0.0
        } else {
            abs.iter().sum::<f64>() / n as f64
        },
        p999_abs: p999,
        max_abs: abs.last().copied().unwrap_or(0.0),
        by_class,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(c: u8, e: [f64; 4]) -> RowError {
        (c, e)
    }

    #[test]
    fn statistics_of_known_errors() {
        let errs = vec![err(0, [0.1, -0.1, 0.0, 0.0]), err(1, [0.3, -0.3, 0.0, 0.0])];
        let s = stats(&errs);
        assert_eq!(s.rows, 2);
        assert!((s.bias[0] - 0.2).abs() < 1e-12, "mean of 0.1 and 0.3");
        assert!((s.bias[1] + 0.2).abs() < 1e-12);
        assert!(
            (s.mean_abs - 0.1).abs() < 1e-12,
            "abs values 0.1,0.1,0,0,0.3,0.3,0,0: mean 0.1"
        );
        assert!((s.max_abs - 0.3).abs() < 1e-12);
        assert!(
            (s.p999_abs - 0.3).abs() < 1e-12,
            "the 99.9th percentile of 8 values is the largest"
        );
        // SD over all 8 values: mean 0; squares 0.01,0.01,0,0,0.09,0.09,0,0 → mean 0.025 → sqrt.
        assert!((s.sd - 0.025f64.sqrt()).abs() < 1e-12);
        assert_eq!(s.by_class[&0].0, 1);
        assert_eq!(s.by_class[&1].0, 1);
    }

    #[test]
    fn empty_is_zero() {
        let s = stats(&[]);
        assert_eq!(s.rows, 0);
        assert_eq!(s.sd, 0.0);
        assert_eq!(s.max_abs, 0.0);
    }
}
