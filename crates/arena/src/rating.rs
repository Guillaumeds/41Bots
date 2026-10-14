//! Offline Plackett-Luce ratings (docs/M1-SPEC.md §5.3), top-1 Luce form: the winning team of a
//! table is chosen with probability `exp(r_w) / Σ_teams exp(r_t)`. Fitted by maximum likelihood
//! with Hunter's MM algorithm, standard errors from the observed information. Floats: reports only.
//!
//! Choices where the spec is silent:
//! - The four teams of a table are the entrants. A team's program is its bot name if both seats
//!   run the same bot, else the two names joined with `+` (a pairing counts as its own program).
//! - Tables where every team is the same program say nothing and are skipped.
//! - Ratings are anchored so the first program (in name order) has r = 0; standard errors are
//!   relative to that anchor. A program that never won has no finite rating; it is reported as
//!   `None` and left out of the fit.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use tablemgr::MatchRecord;

/// One program's rating.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rating {
    /// Program name.
    pub program: String,
    /// Team-entries at informative tables.
    pub entries: u64,
    /// Tables won.
    pub wins: u64,
    /// Log-strength relative to the anchor (natural log units); `None` if it never won.
    pub r: Option<f64>,
    /// Standard error of `r`; `None` for the anchor or a program without a rating.
    pub se: Option<f64>,
}

/// The program name of team `t`.
pub fn team_program(rec: &MatchRecord, t: usize) -> String {
    let (x, y) = (&rec.bots[t], &rec.bots[t + 4]);
    if x == y {
        x.clone()
    } else {
        let (p, q) = if x <= y { (x, y) } else { (y, x) };
        format!("{p}+{q}")
    }
}

/// Fit ratings to records.
pub fn rate(records: &[MatchRecord]) -> Vec<Rating> {
    let mut names: BTreeMap<String, usize> = BTreeMap::new();
    let mut tables: Vec<([usize; 4], usize)> = Vec::new();
    for rec in records {
        let progs: [String; 4] = core::array::from_fn(|t| team_program(rec, t));
        if progs.iter().all(|p| *p == progs[0]) {
            continue;
        }
        for p in &progs {
            let n = names.len();
            names.entry(p.clone()).or_insert(n);
        }
        let ids = core::array::from_fn(|t| names[&progs[t]]);
        tables.push((ids, usize::from(rec.winner % 4)));
    }
    // Re-index in name order (BTreeMap iteration is sorted, so this is deterministic).
    let order: Vec<String> = names.keys().cloned().collect();
    let remap: Vec<usize> = {
        let mut v = vec![0; order.len()];
        for (new, name) in order.iter().enumerate() {
            v[names[name]] = new;
        }
        v
    };
    let k = order.len();
    let tables: Vec<([usize; 4], usize)> = tables
        .into_iter()
        .map(|(ids, w)| (ids.map(|i| remap[i]), w))
        .collect();
    let mut wins = vec![0u64; k];
    let mut entries = vec![0u64; k];
    for (ids, w) in &tables {
        wins[ids[*w]] += 1;
        for &i in ids {
            entries[i] += 1;
        }
    }
    let active: Vec<bool> = wins.iter().map(|&w| w > 0).collect();
    // MM iterations on γ = exp(r). Programs that never won have γ = 0 (fixed).
    let mut g: Vec<f64> = active.iter().map(|&a| if a { 1.0 } else { 0.0 }).collect();
    for _ in 0..100_000 {
        let mut denom = vec![0.0f64; k];
        for (ids, _) in &tables {
            let tot: f64 = ids.iter().map(|&i| g[i]).sum();
            if tot <= 0.0 {
                continue;
            }
            for &i in ids {
                denom[i] += 1.0 / tot;
            }
        }
        let mut next: Vec<f64> = (0..k)
            .map(|i| {
                if active[i] && denom[i] > 0.0 {
                    wins[i] as f64 / denom[i]
                } else {
                    0.0
                }
            })
            .collect();
        // Normalise to geometric mean 1 over active programs.
        let la: Vec<f64> = (0..k)
            .filter(|&i| active[i])
            .map(|i| next[i].ln())
            .collect();
        let shift = la.iter().sum::<f64>() / la.len().max(1) as f64;
        for i in 0..k {
            if active[i] {
                next[i] = (next[i].ln() - shift).exp();
            }
        }
        let change = (0..k)
            .filter(|&i| active[i])
            .map(|i| (next[i].ln() - g[i].ln()).abs())
            .fold(0.0, f64::max);
        g = next;
        if change < 1e-12 {
            break;
        }
    }
    // Observed information in r for the active programs: Σ_tables diag(p) − p pᵀ, where p_i is
    // the table's total probability on program i.
    let idx: Vec<usize> = (0..k).filter(|&i| active[i]).collect();
    let m = idx.len();
    let pos = |i: usize| idx.iter().position(|&j| j == i);
    let mut info = vec![vec![0.0f64; m]; m];
    for (ids, _) in &tables {
        let tot: f64 = ids.iter().map(|&i| g[i]).sum();
        let mut p = vec![0.0f64; m];
        for &i in ids {
            if let Some(a) = pos(i) {
                p[a] += g[i] / tot;
            }
        }
        for a in 0..m {
            info[a][a] += p[a];
            for b in 0..m {
                info[a][b] -= p[a] * p[b];
            }
        }
    }
    // Anchor the first active program: drop its row/column and invert the rest.
    let se_sub = if m >= 2 { invert(&reduce(&info)) } else { None };
    let anchor = idx.first().copied();
    (0..k)
        .map(|i| {
            let r = if active[i] {
                Some(g[i].ln() - anchor.map_or(0.0, |a| g[a].ln()))
            } else {
                None
            };
            let se = match (&se_sub, pos(i)) {
                (Some(inv), Some(a)) if a > 0 => Some(inv[a - 1][a - 1].max(0.0).sqrt()),
                _ => None,
            };
            Rating {
                program: order[i].clone(),
                entries: entries[i],
                wins: wins[i],
                r,
                se,
            }
        })
        .collect()
}

fn reduce(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    m[1..].iter().map(|row| row[1..].to_vec()).collect()
}

/// Gauss-Jordan inverse with partial pivoting; `None` if singular.
fn invert(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();
    for c in 0..n {
        let p = (c..n).max_by(|&x, &y| m[x][c].abs().total_cmp(&m[y][c].abs()))?;
        if m[p][c].abs() < 1e-300 {
            return None;
        }
        m.swap(c, p);
        let d = m[c][c];
        for v in m[c].iter_mut() {
            *v /= d;
        }
        for r in 0..n {
            if r != c {
                let f = m[r][c];
                if f != 0.0 {
                    let pivot = m[c].clone();
                    for (v, pv) in m[r].iter_mut().zip(pivot) {
                        *v -= f * pv;
                    }
                }
            }
        }
    }
    Some(m.into_iter().map(|r| r[n..].to_vec()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_small() {
        let inv = invert(&[vec![4.0, 7.0], vec![2.0, 6.0]]).unwrap();
        assert!((inv[0][0] - 0.6).abs() < 1e-12 && (inv[1][0] + 0.2).abs() < 1e-12);
    }
}
