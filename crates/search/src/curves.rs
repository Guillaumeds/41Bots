//! N-curves (docs/M3-SPEC.md §13.1 item 10): from the saved per-world candidate values of
//! `search worlds`, the expected money loss of choosing a card with N worlds, for every N on a
//! fine grid up to the saved count, with no new solves; a power-law fit with one exponent per
//! trick; two held-out checks; and the N each scenario needs to reach a money-loss threshold.
//!
//! **Per decision** (team values in units of 2^-32; a choice is made exactly as the PIMC bot
//! makes it, [`crate::pimc::aggregate`]: the largest rounded mean, ties to the first candidate):
//! - *prefix*: the loss, under the means of all M worlds, of the choice made from the first N
//!   saved worlds (the one nested sequence the bot would have used).
//! - *plain*: the same averaged over `splits` random N-subsets of the M worlds, drawn without
//!   replacement (subset `s` is the first N of permutation `s`, seeded by
//!   `Rng::derive(seed, &[CURVES, id, s])`). Its reference, the M-world choice, is itself noisy
//!   and shares worlds with every subset, so it reads too low as N nears M.
//! - *cross-fitted* (`cv`): the same permutation cuts the M worlds into a choice half (its first
//!   K = M/2) and a scoring half. The N-world choice (the first N) and the K-world reference
//!   choice (the whole choice half) are both scored on the scoring half, which neither has seen.
//!   Its expectation is exactly L(N) − L(K): the true loss at N (against infinitely many worlds)
//!   less the true loss at K. No lucky-reference bias, whatever the shape of the curve. Scored
//!   values are exact integer differences of totals.
//! - *cv on half the data* (`cv_half`): the same on the first M/2 worlds only (K' = M/4), for the
//!   extrapolation check below.
//!
//! **Groups**: all decisions; each position, stratum and hand-type band; position × stratum,
//! position × trumps band, stratum × trumps band and position × stratum × trumps band cells with
//! at least `min_cell` decisions. A group's curve is the plain mean over its decisions.
//!
//! **Uncertainty**: a stratified bootstrap of decisions (each replicate resamples, with
//! replacement, the decisions of every position × stratum design cell; seeded), recomputing every
//! group mean and refitting everything. Bands and intervals are the 2.5% and 97.5% points.
//!
//! **Fit**: L(N) = a·N^(−b), so the cv curve is a·(N^(−b) − K^(−b)). Weighted least squares on
//! log scale over the grid points N ≤ K/2, each weighted by its share of log N (so every doubling
//! of N counts the same however fine the grid) over its squared relative standard error (from
//! the bootstrap); points with a non-positive mean are skipped. One b for the eight positions of
//! a trick (found by a scan then golden-section search on the profile; a closed form gives each
//! position's a for a given b); every other group gets its own a with that b. Each group's own
//! free b is reported as a check that one b fits all.
//!
//! **Checks**: (1) held-out N: the fit, which saw only N ≤ K/2, predicts the cv curve for
//! K/2 < N < K; (2) extrapolation: the same recipe run on only the first M/2 worlds (reference
//! K' = M/4, fit on N ≤ M/8) predicts the full data's cv curve, out to N = K. Each prediction is
//! marked inside or outside the measured 95% band.
//!
//! **Thresholds**: N(θ) = (a/θ)^(1/b) per group and threshold θ (money chance), capped at
//! [`N_CAP`].
//!
//! Everything here is a report: values and choices are integer, statistics are `f64`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use engine::rng::Rng;
use equity::{MoneyChance, ONE};
use serde::Serialize;

use crate::measure::WorldRows;

/// Seed label for the curves' world permutations and bootstrap.
pub const CURVES: u64 = 0xC0F5;
/// Seed label for the bootstrap of decisions.
const BOOT: u64 = 0xB007;
/// Largest N reported for a threshold (a flatter curve never reaches it in practice).
pub const N_CAP: f64 = 1e9;
/// Fewest decisions for a group's numbers to set N (§13.1 item 8).
pub const ENOUGH: usize = 25;

/// Settings of one analysis.
#[derive(Clone, Debug)]
pub struct CurveParams {
    /// World counts N to evaluate (values above M are dropped).
    pub grid: Vec<usize>,
    /// Worlds per decision used (M); `None`: the fewest any decision of the trick has.
    pub worlds: Option<usize>,
    /// Random splits (permutations) per decision.
    pub splits: u64,
    /// Bootstrap replicates of decisions.
    pub boots: u64,
    /// Seed of the permutations and the bootstrap.
    pub seed: u64,
    /// Money-loss thresholds (money chance, e.g. 0.0001 = 0.01%).
    pub thresholds: Vec<f64>,
    /// Smallest cell (two- and three-way groups) reported.
    pub min_cell: usize,
    /// Worker threads for the per-decision step.
    pub threads: usize,
}

impl Default for CurveParams {
    fn default() -> Self {
        CurveParams {
            grid: (1..=32).map(|i| 10 * i).collect(),
            worlds: None,
            splits: 400,
            boots: 1000,
            seed: 1,
            thresholds: vec![1e-4, 5e-5, 2e-5],
            min_cell: 10,
            threads: 1,
        }
    }
}

/// One decision's curves on the grid (sums over the splits).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionCurve {
    /// Decision id.
    pub id: u64,
    /// Position in the trick (0 = leader).
    pub position: u8,
    /// Stratum (1–4).
    pub stratum: u8,
    /// Hand-type bands (trumps, top trumps, longest plain suit), when known.
    pub bands: Option<[u8; 3]>,
    /// log2 world count, when known.
    pub log2_worlds: Option<u32>,
    /// Loss of the first-N choice under the M-world means (units of 2^-32), per grid N.
    pub prefix: Vec<u64>,
    /// Sum over splits of the loss of the random-N-subset choice under the M-world means.
    pub plain: Vec<u128>,
    /// Splits whose N-subset choice is not among the M-world tied-best.
    pub plain_wrong: Vec<u64>,
    /// Sum over splits of (scoring-half total of the K-world choice − that of the N-world
    /// choice), per grid N ≤ K; divide by `splits · (M − K) · 2^32` for money chance.
    pub cv: Vec<i128>,
    /// The same on the first M/2 worlds (K' = M/4), per grid N ≤ K'.
    pub cv_half: Vec<i128>,
}

/// The candidate the PIMC bot picks from per-candidate totals over `n` worlds.
fn choose(totals: &[[u64; 4]], n: u64, team: u8) -> usize {
    let mut best = 0;
    let mut best_v = 0;
    for (j, t) in totals.iter().enumerate() {
        let v = MoneyChance::mean_of_totals(*t, n).get(team);
        if j == 0 || v > best_v {
            best = j;
            best_v = v;
        }
    }
    best
}

/// One sweep over a permutation of worlds `perm` (indices into `rows`): the choice from each
/// prefix of length N on `grid` (grid values up to `perm.len()`), the reference choice from the
/// first `k` worlds, and each candidate's team total over the worlds after the first `k` (the
/// scoring half).
fn pass(
    rows: &[Vec<[u64; 4]>],
    perm: &[usize],
    grid: &[usize],
    k: usize,
    team: u8,
) -> (Vec<usize>, usize, Vec<u64>) {
    let c = rows[0].len();
    let mut totals = vec![[0u64; 4]; c];
    let mut choices = Vec::new();
    let mut gi = 0;
    let mut reference = 0;
    let mut score = vec![0u64; c];
    let t = usize::from(team);
    for (i, &w) in perm.iter().enumerate() {
        let n = i + 1;
        for (tot, v) in totals.iter_mut().zip(&rows[w]) {
            for x in 0..4 {
                tot[x] += v[x];
            }
        }
        if n > k {
            for (s, v) in score.iter_mut().zip(&rows[w]) {
                *s += v[t];
            }
        }
        while gi < grid.len() && grid[gi] == n {
            choices.push(choose(&totals, n as u64, team));
            gi += 1;
        }
        if n == k {
            reference = choose(&totals, n as u64, team);
        }
    }
    (choices, reference, score)
}

/// The curves of one decision over its first `m` worlds ([`DecisionCurve`]).
pub fn decision_curve(
    r: &WorldRows,
    m: usize,
    grid: &[usize],
    splits: u64,
    seed: u64,
) -> DecisionCurve {
    assert!(
        m >= 4 && r.rows.len() >= m,
        "decision {}: needs {m} worlds",
        r.id
    );
    let rows = &r.rows[..m];
    let team = r.team;
    let ng = grid.iter().filter(|&&n| n <= m).count();
    let grid = &grid[..ng];
    // The M-world means and each candidate's loss under them (as `measure::score`).
    let ident: Vec<usize> = (0..m).collect();
    let (prefix_choices, full, _) = pass(rows, &ident, grid, m, team);
    let c = rows[0].len();
    let totals: Vec<[u64; 4]> = (0..c)
        .map(|j| core::array::from_fn(|x| rows.iter().map(|w| w[j][x]).sum()))
        .collect();
    let vals: Vec<u64> = totals
        .iter()
        .map(|t| MoneyChance::mean_of_totals(*t, m as u64).get(team))
        .collect();
    let best = vals[full];
    let loss: Vec<u64> = vals.iter().map(|&v| best - v).collect();
    let prefix = prefix_choices.iter().map(|&j| loss[j]).collect();
    let k = m / 2;
    let kh = m / 4;
    let nk = grid.iter().filter(|&&n| n <= k).count();
    let nkh = grid.iter().filter(|&&n| n <= kh).count();
    let mut plain = vec![0u128; ng];
    let mut plain_wrong = vec![0u64; ng];
    let mut cv = vec![0i128; nk];
    let mut cv_half = vec![0i128; nkh];
    let mut perm: Vec<usize> = (0..m).collect();
    let mut perm_h: Vec<usize> = (0..m / 2).collect();
    for s in 0..splits {
        // Each split's permutation starts from the identity, so split `s` does not depend on
        // the splits before it.
        for (i, p) in perm.iter_mut().enumerate() {
            *p = i;
        }
        Rng::derive(seed, &[CURVES, r.id, s, 0]).shuffle(&mut perm);
        // The plain subsets are the prefixes of the whole permutation; the cross-fitted choices
        // are the same prefixes up to K, scored on the worlds after K.
        let (ch, rk, score) = pass(rows, &perm, grid, k, team);
        for (g, &j) in ch.iter().enumerate() {
            plain[g] += u128::from(loss[j]);
            plain_wrong[g] += u64::from(loss[j] > 0);
            if g < nk {
                cv[g] += i128::from(score[rk]) - i128::from(score[j]);
            }
        }
        for (i, p) in perm_h.iter_mut().enumerate() {
            *p = i;
        }
        Rng::derive(seed, &[CURVES, r.id, s, 1]).shuffle(&mut perm_h);
        let (chh, rh, score_h) = pass(rows, &perm_h, &grid[..nkh], kh, team);
        for (g, &j) in chh.iter().enumerate() {
            cv_half[g] += i128::from(score_h[rh]) - i128::from(score_h[j]);
        }
    }
    DecisionCurve {
        id: r.id,
        position: r.position,
        stratum: r.stratum,
        bands: r.hand_type.map(|h| h.bands()),
        log2_worlds: r.log2_worlds,
        prefix,
        plain,
        plain_wrong,
        cv,
        cv_half,
    }
}

/// [`decision_curve`] for every decision, on `threads` threads (results in input order, so the
/// output does not depend on the thread count).
pub fn decision_curves(
    rs: &[&WorldRows],
    m: usize,
    grid: &[usize],
    splits: u64,
    seed: u64,
    threads: usize,
) -> Vec<DecisionCurve> {
    let slots: Vec<std::sync::OnceLock<DecisionCurve>> =
        rs.iter().map(|_| std::sync::OnceLock::new()).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.max(1).min(rs.len().max(1)) {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= rs.len() {
                        break;
                    }
                    let c = decision_curve(rs[i], m, grid, splits, seed);
                    assert!(slots[i].set(c).is_ok(), "each decision once");
                }
            });
        }
    });
    slots
        .into_iter()
        .map(|c| c.into_inner().expect("every decision done"))
        .collect()
}

// ---------------------------------------------------------------- fitting

/// One group's points for a fit: grid N, log of the measured mean, weight.
#[derive(Clone, Debug, Default)]
pub struct FitPoints {
    /// ln N.
    pub ln_n: Vec<f64>,
    /// ln of the measured cv mean.
    pub ln_y: Vec<f64>,
    /// Weights.
    pub w: Vec<f64>,
}

fn ln_g(ln_n: f64, ln_k: f64, b: f64) -> f64 {
    ((-b * ln_n).exp() - (-b * ln_k).exp()).ln()
}

/// For a fixed b, each group's best ln a and the total weighted squared error.
fn profile(groups: &[&FitPoints], ln_k: f64, b: f64) -> (f64, Vec<f64>) {
    let mut rss = 0.0;
    let mut las = Vec::with_capacity(groups.len());
    for g in groups {
        let sw: f64 = g.w.iter().sum();
        if sw <= 0.0 {
            las.push(f64::NAN);
            continue;
        }
        let la = g
            .ln_n
            .iter()
            .zip(&g.ln_y)
            .zip(&g.w)
            .map(|((&x, &y), &w)| w * (y - ln_g(x, ln_k, b)))
            .sum::<f64>()
            / sw;
        rss += g
            .ln_n
            .iter()
            .zip(&g.ln_y)
            .zip(&g.w)
            .map(|((&x, &y), &w)| {
                let e = y - la - ln_g(x, ln_k, b);
                w * e * e
            })
            .sum::<f64>();
        las.push(la);
    }
    (rss, las)
}

/// Smallest and largest exponent searched.
pub const B_RANGE: (f64, f64) = (0.01, 3.0);

/// Fit one shared b and a per group to cv points: model ln y = ln a_g + ln(N^(−b) − K^(−b)).
/// Returns (b, ln a per group); `None` when no group has two points.
pub fn fit_shared(groups: &[&FitPoints], k: f64) -> Option<(f64, Vec<f64>)> {
    let usable: usize = groups.iter().filter(|g| g.ln_n.len() >= 2).count();
    if usable == 0 {
        return None;
    }
    let ln_k = k.ln();
    let f = |b: f64| profile(groups, ln_k, b).0;
    // Scan, then golden-section search around the best scan point.
    let steps = ((B_RANGE.1 - B_RANGE.0) / 0.01).round() as usize;
    let mut best = (f64::INFINITY, B_RANGE.0);
    for i in 0..=steps {
        let b = B_RANGE.0 + 0.01 * i as f64;
        let v = f(b);
        if v < best.0 {
            best = (v, b);
        }
    }
    let (mut lo, mut hi) = (
        (best.1 - 0.01).max(B_RANGE.0),
        (best.1 + 0.01).min(B_RANGE.1),
    );
    let gr = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..40 {
        let (c, d) = (hi - gr * (hi - lo), lo + gr * (hi - lo));
        if f(c) <= f(d) {
            hi = d;
        } else {
            lo = c;
        }
    }
    let b = (lo + hi) / 2.0;
    let b = if f(b) <= best.0 { b } else { best.1 };
    Some((b, profile(groups, ln_k, b).1))
}

/// A group's ln a for a given b.
pub fn fit_a(g: &FitPoints, k: f64, b: f64) -> f64 {
    profile(&[g], k.ln(), b).1[0]
}

/// Log-spacing weights of sorted points: each point's share of the log-N axis.
fn log_spacing(ln_n: &[f64]) -> Vec<f64> {
    let n = ln_n.len();
    (0..n)
        .map(|i| {
            let left = if i > 0 { ln_n[i] - ln_n[i - 1] } else { 0.0 };
            let right = if i + 1 < n {
                ln_n[i + 1] - ln_n[i]
            } else {
                0.0
            };
            let s = (left + right) / 2.0;
            if n == 1 { 1.0 } else { s }
        })
        .collect()
}

/// Fit points of a curve: grid values N ≤ `n_max` with a positive mean, weighted by log spacing
/// (over all fit-range points) over squared relative error (`rel_se`, floored at 5%).
fn fit_points(grid: &[usize], y: &[f64], rel_se: &[f64], n_max: usize) -> FitPoints {
    let idx: Vec<usize> = (0..y.len()).filter(|&i| grid[i] <= n_max).collect();
    let ln_all: Vec<f64> = idx.iter().map(|&i| (grid[i] as f64).ln()).collect();
    let sp = log_spacing(&ln_all);
    let mut p = FitPoints::default();
    for (j, &i) in idx.iter().enumerate() {
        if y[i] > 0.0 && y[i].is_finite() {
            let r = rel_se[i].max(0.05);
            p.ln_n.push(ln_all[j]);
            p.ln_y.push(y[i].ln());
            p.w.push(sp[j] / (r * r));
        }
    }
    p
}

// ---------------------------------------------------------------- groups and report

/// Percentiles (2.5%, 97.5%) of values (NaN ignored).
fn band(v: &mut [f64]) -> (f64, f64) {
    let mut x: Vec<f64> = v.iter().copied().filter(|a| !a.is_nan()).collect();
    if x.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    x.sort_by(f64::total_cmp);
    let q = |p: f64| x[((x.len() - 1) as f64 * p).round() as usize];
    (q(0.025), q(0.975))
}

fn sd(v: &[f64]) -> f64 {
    let x: Vec<f64> = v.iter().copied().filter(|a| a.is_finite()).collect();
    if x.len() < 2 {
        return f64::NAN;
    }
    let m = x.iter().sum::<f64>() / x.len() as f64;
    (x.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / (x.len() - 1) as f64).sqrt()
}

/// A measured curve with its 95% band, per grid N.
#[derive(Clone, Debug, Serialize)]
pub struct Measured {
    /// Mean over the group's decisions (money chance).
    pub mean: Vec<f64>,
    /// Bootstrap 2.5% point.
    pub lo: Vec<f64>,
    /// Bootstrap 97.5% point.
    pub hi: Vec<f64>,
}

/// A predicted value with its 95% interval.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Est {
    /// Point value (from the full sample).
    pub v: f64,
    /// Bootstrap 2.5% point.
    pub lo: f64,
    /// Bootstrap 97.5% point.
    pub hi: f64,
}

/// One prediction checked against the measured band.
#[derive(Clone, Debug, Serialize)]
pub struct Check {
    /// N.
    pub n: usize,
    /// Predicted cv value.
    pub pred: f64,
    /// Measured cv mean.
    pub measured: f64,
    /// Inside the measured 95% band.
    pub inside: bool,
}

/// One group's results.
#[derive(Clone, Debug, Serialize)]
pub struct GroupReport {
    /// `all`, `position`, `stratum`, `trumps`, `top_trumps`, `longest_plain`, `log2_worlds`,
    /// or a cell kind such as `position×stratum`.
    pub kind: String,
    /// Display name (`pos1`, `S2`, `trumps3+`, `pos1·S2`, …).
    pub name: String,
    /// Decisions.
    pub decisions: usize,
    /// At least [`ENOUGH`] decisions.
    pub enough: bool,
    /// Plain loss against the M-world choice (random N-subsets).
    pub plain: Measured,
    /// Share of random N-subsets whose choice is not among the M-world tied-best.
    pub plain_change: Vec<f64>,
    /// Loss of the first-N choice against the M-world choice.
    pub prefix: Measured,
    /// Cross-fitted loss L(N) − L(K), per grid N ≤ K.
    pub cv: Measured,
    /// ln a of the shared-b fit (L(N) = a·N^(−b)) and its interval as a, `None` without data.
    pub a: Option<Est>,
    /// The group's own exponent (free b) and its interval (main groups only).
    pub free_b: Option<Est>,
    /// The fit's cv curve per grid N ≤ K.
    pub fit_cv: Vec<f64>,
    /// Held-out check: grid N with K/2 < N < K.
    pub heldout: Vec<Check>,
    /// Extrapolation check: the half-data fit's prediction of this cv curve, grid N < K.
    pub half: Vec<Check>,
    /// Predicted loss L(N) at `TrickReport::pred_n`.
    pub loss: Vec<Est>,
    /// Predicted N for each threshold.
    pub n_for: Vec<Est>,
}

/// One trick's results.
#[derive(Clone, Debug, Serialize)]
pub struct TrickReport {
    /// The trick.
    pub trick: u8,
    /// Decisions analysed.
    pub decisions: usize,
    /// Decisions left out for having fewer than M worlds.
    pub dropped: usize,
    /// Worlds per decision used (M).
    pub worlds: usize,
    /// The cross-fitted reference size K = M/2.
    pub k: usize,
    /// Largest N in the fit (K/2).
    pub fit_max: usize,
    /// Grid (N ≤ M).
    pub grid: Vec<usize>,
    /// Random splits per decision.
    pub splits: u64,
    /// Bootstrap replicates.
    pub boots: u64,
    /// Seed.
    pub seed: u64,
    /// Thresholds (money chance).
    pub thresholds: Vec<f64>,
    /// The shared exponent b over the eight positions, with its interval.
    pub b: Option<Est>,
    /// The shared exponent of the half-data fit (extrapolation check).
    pub b_half: Option<Est>,
    /// The shared exponent fitted on N up to each listed value (a drift check).
    pub b_by_range: Vec<(usize, f64)>,
    /// N at which the loss is predicted (`GroupReport::loss`).
    pub pred_n: Vec<f64>,
    /// Held-out check over `all` and the positions: predictions inside the band, and in total.
    pub heldout_inside: (usize, usize),
    /// Extrapolation check over `all` and the positions: inside, total.
    pub half_inside: (usize, usize),
    /// Mean of ln(pred / measured) over the held-out points with a positive measurement.
    pub heldout_log_ratio: f64,
    /// The same for the extrapolation check.
    pub half_log_ratio: f64,
    /// Groups.
    pub groups: Vec<GroupReport>,
}

struct Group {
    kind: String,
    name: String,
    members: Vec<usize>,
    main: bool,
}

fn groups_of(cs: &[DecisionCurve], min_cell: usize) -> Vec<Group> {
    let tb = |b: u8| format!("trumps{b}{}", if b == 3 { "+" } else { "" });
    let mut map: BTreeMap<(u8, String, String), Vec<usize>> = BTreeMap::new();
    let mut add = |order: u8, kind: &str, name: String, i: usize| {
        map.entry((order, kind.to_string(), name))
            .or_default()
            .push(i);
    };
    for (i, c) in cs.iter().enumerate() {
        let p = format!("pos{}", c.position + 1);
        let s = format!("S{}", c.stratum);
        add(0, "all", "all".into(), i);
        add(1, "position", p.clone(), i);
        add(2, "stratum", s.clone(), i);
        if let Some([t, top, plain]) = c.bands {
            add(3, "trumps", tb(t), i);
            add(
                4,
                "top_trumps",
                format!("top{top}{}", if top == 2 { "+" } else { "" }),
                i,
            );
            add(
                5,
                "longest_plain",
                match plain {
                    1 => "plain0-1".to_string(),
                    3 => "plain3+".to_string(),
                    x => format!("plain{x}"),
                },
                i,
            );
            add(8, "position×trumps", format!("{p}·{}", tb(t)), i);
            add(9, "stratum×trumps", format!("{s}·{}", tb(t)), i);
            add(
                10,
                "position×stratum×trumps",
                format!("{p}·{s}·{}", tb(t)),
                i,
            );
        }
        if let Some(l) = c.log2_worlds {
            add(6, "log2_worlds", format!("log2worlds{l}"), i);
        }
        add(7, "position×stratum", format!("{p}·{s}"), i);
    }
    map.into_iter()
        .filter(|((o, _, _), m)| *o <= 6 || m.len() >= min_cell)
        .map(|((o, kind, name), members)| Group {
            kind,
            name,
            members,
            main: o <= 2,
        })
        .collect()
}

/// Bootstrap multiplicities: replicate `r` resamples each position × stratum cell.
fn boot_counts(cs: &[DecisionCurve], boots: u64, seed: u64, trick: u8) -> Vec<Vec<u32>> {
    let keys: Vec<(u8, u8)> = cs.iter().map(|c| (c.position, c.stratum)).collect();
    boot_cells(&keys, boots, seed, u64::from(trick))
}

/// Bootstrap multiplicities over decisions with design cells `keys` (position, stratum):
/// replicate `r` resamples each cell with replacement, seeded by `tag` and `r`.
fn boot_cells(keys: &[(u8, u8)], boots: u64, seed: u64, tag: u64) -> Vec<Vec<u32>> {
    let mut cells: BTreeMap<(u8, u8), Vec<usize>> = BTreeMap::new();
    for (i, k) in keys.iter().enumerate() {
        cells.entry(*k).or_default().push(i);
    }
    (0..boots)
        .map(|r| {
            let mut rng = Rng::derive(seed, &[CURVES, BOOT, tag, r]);
            let mut cnt = vec![0u32; keys.len()];
            for m in cells.values() {
                for _ in 0..m.len() {
                    cnt[m[rng.bounded(m.len() as u64) as usize]] += 1;
                }
            }
            cnt
        })
        .collect()
}

/// The per-decision curves as money-chance means, one row per decision: (plain, wrong, prefix,
/// cv, cv_half).
struct Means {
    plain: Vec<Vec<f64>>,
    wrong: Vec<Vec<f64>>,
    prefix: Vec<Vec<f64>>,
    cv: Vec<Vec<f64>>,
    cv_half: Vec<Vec<f64>>,
}

fn means_of(cs: &[DecisionCurve], m: usize, splits: u64) -> Means {
    let one = ONE as f64;
    let sp = splits as f64;
    let (k, kh) = (m / 2, m / 4);
    Means {
        plain: cs
            .iter()
            .map(|c| c.plain.iter().map(|&x| x as f64 / sp / one).collect())
            .collect(),
        wrong: cs
            .iter()
            .map(|c| c.plain_wrong.iter().map(|&x| x as f64 / sp).collect())
            .collect(),
        prefix: cs
            .iter()
            .map(|c| c.prefix.iter().map(|&x| x as f64 / one).collect())
            .collect(),
        cv: cs
            .iter()
            .map(|c| {
                c.cv.iter()
                    .map(|&x| x as f64 / sp / (m - k) as f64 / one)
                    .collect()
            })
            .collect(),
        cv_half: cs
            .iter()
            .map(|c| {
                c.cv_half
                    .iter()
                    .map(|&x| x as f64 / sp / (m / 2 - kh) as f64 / one)
                    .collect()
            })
            .collect(),
    }
}

/// Weighted mean over members of per-decision rows (weights = bootstrap multiplicities, or 1).
fn group_mean(rows: &[Vec<f64>], members: &[usize], w: Option<&[u32]>) -> Vec<f64> {
    let len = rows.first().map_or(0, Vec::len);
    let mut s = vec![0f64; len];
    let mut n = 0f64;
    for &i in members {
        let wi = w.map_or(1.0, |w| f64::from(w[i]));
        if wi == 0.0 {
            continue;
        }
        n += wi;
        for (a, &x) in s.iter_mut().zip(&rows[i]) {
            *a += wi * x;
        }
    }
    if n == 0.0 {
        return vec![f64::NAN; len];
    }
    s.iter().map(|a| a / n).collect()
}

/// A group's replicate means of one curve: `[replicate][grid index]`.
fn replicates(rows: &[Vec<f64>], members: &[usize], counts: &[Vec<u32>]) -> Vec<Vec<f64>> {
    counts
        .iter()
        .map(|c| group_mean(rows, members, Some(c)))
        .collect()
}

fn measured(rows: &[Vec<f64>], members: &[usize], reps: &[Vec<f64>]) -> Measured {
    let mean = group_mean(rows, members, None);
    let mut lo = Vec::new();
    let mut hi = Vec::new();
    for g in 0..mean.len() {
        let mut v: Vec<f64> = reps.iter().map(|r| r[g]).collect();
        let (l, h) = band(&mut v);
        lo.push(l);
        hi.push(h);
    }
    Measured { mean, lo, hi }
}

/// Relative standard error per grid index from replicates.
fn rel_se(mean: &[f64], reps: &[Vec<f64>]) -> Vec<f64> {
    (0..mean.len())
        .map(|g| {
            let v: Vec<f64> = reps.iter().map(|r| r[g]).collect();
            sd(&v) / mean[g].abs().max(1e-300)
        })
        .collect()
}

/// The N where a·N^(−b) falls to `theta`, between 1 and [`N_CAP`]; `ln a = −∞` (no measured
/// loss at all) gives 1.
fn n_for(la: f64, b: f64, theta: f64) -> f64 {
    if la.is_nan() || !b.is_finite() || b <= 0.0 {
        return f64::NAN;
    }
    ((la - theta.ln()) / b).exp().clamp(1.0, N_CAP)
}

fn est(v: f64, reps: &mut [f64]) -> Est {
    let (lo, hi) = band(reps);
    Est { v, lo, hi }
}

/// Analyse one trick's decisions (all with at least M worlds).
pub fn analyse_trick(trick: u8, rows: &[&WorldRows], p: &CurveParams) -> Option<TrickReport> {
    let m = p
        .worlds
        .unwrap_or_else(|| rows.iter().map(|r| r.rows.len()).min().unwrap_or(0));
    let m = m & !3; // a multiple of 4, so both halves and quarters are whole
    if m < 8 {
        return None;
    }
    let used: Vec<&WorldRows> = rows.iter().copied().filter(|r| r.rows.len() >= m).collect();
    let dropped = rows.len() - used.len();
    let mut grid: Vec<usize> = p
        .grid
        .iter()
        .copied()
        .filter(|&n| n >= 1 && n <= m)
        .collect();
    grid.sort_unstable();
    grid.dedup();
    let (k, kh) = (m / 2, m / 4);
    let (fit_max, fit_max_h) = (k / 2, kh / 2);
    let cs = decision_curves(&used, m, &grid, p.splits, p.seed, p.threads);
    let mm = means_of(&cs, m, p.splits);
    let groups = groups_of(&cs, p.min_cell);
    let counts = boot_counts(&cs, p.boots, p.seed, trick);
    let nk = grid.iter().filter(|&&n| n <= k).count();
    let nkh = grid.iter().filter(|&&n| n <= kh).count();
    let grid_k = &grid[..nk];
    let grid_kh = &grid[..nkh];

    // Shared b over the positions: original sample, then every replicate (fixed weights).
    let pos: Vec<&Group> = groups.iter().filter(|g| g.kind == "position").collect();
    let pos_reps: Vec<Vec<Vec<f64>>> = pos
        .iter()
        .map(|g| replicates(&mm.cv, &g.members, &counts))
        .collect();
    let pos_half_reps: Vec<Vec<Vec<f64>>> = pos
        .iter()
        .map(|g| replicates(&mm.cv_half, &g.members, &counts))
        .collect();
    let pos_mean: Vec<Vec<f64>> = pos
        .iter()
        .map(|g| group_mean(&mm.cv, &g.members, None))
        .collect();
    let pos_rse: Vec<Vec<f64>> = pos_mean
        .iter()
        .zip(&pos_reps)
        .map(|(m, r)| rel_se(m, r))
        .collect();
    let pts = |ys: &[Vec<f64>], n_max: usize| -> Vec<FitPoints> {
        ys.iter()
            .zip(&pos_rse)
            .map(|(y, r)| fit_points(grid_k, y, r, n_max))
            .collect()
    };
    let shared = |fp: &[FitPoints]| fit_shared(&fp.iter().collect::<Vec<_>>(), k as f64);
    let b0 = shared(&pts(&pos_mean, fit_max)).map(|x| x.0);
    let b_reps: Vec<f64> = (0..counts.len())
        .map(|r| {
            let ys: Vec<Vec<f64>> = pos_reps.iter().map(|g| g[r].clone()).collect();
            shared(&pts(&ys, fit_max)).map_or(f64::NAN, |x| x.0)
        })
        .collect();
    let b = b0.map(|v| est(v, &mut b_reps.clone()));
    // The exponent fitted over growing ranges of N (the cv curve is unbiased up to K, so every
    // range is a fair fit): a b that falls as the range grows means the curve flattens, and
    // predictions beyond the measured N are then optimistic.
    let below_k = grid_k.iter().copied().filter(|&n| n < k).max().unwrap_or(k);
    let mut b_by_range: Vec<(usize, f64)> = Vec::new();
    for n_max in [fit_max / 2, fit_max, below_k] {
        if n_max >= 1
            && !b_by_range.iter().any(|x| x.0 == n_max)
            && let Some((bv, _)) = shared(&pts(&pos_mean, n_max))
        {
            b_by_range.push((n_max, bv));
        }
    }
    // Half-data fit (extrapolation check).
    let pos_half_mean: Vec<Vec<f64>> = pos
        .iter()
        .map(|g| group_mean(&mm.cv_half, &g.members, None))
        .collect();
    let pos_half_rse: Vec<Vec<f64>> = pos_half_mean
        .iter()
        .zip(&pos_half_reps)
        .map(|(y, r)| rel_se(y, r))
        .collect();
    let half_pts = |ys: &[Vec<f64>]| -> Vec<FitPoints> {
        ys.iter()
            .zip(&pos_half_rse)
            .map(|(y, r)| fit_points(grid_kh, y, r, fit_max_h))
            .collect()
    };
    let shared_h = |fp: &[FitPoints]| fit_shared(&fp.iter().collect::<Vec<_>>(), kh as f64);
    let b_half = shared_h(&half_pts(&pos_half_mean)).map(|x| {
        let mut reps: Vec<f64> = (0..counts.len())
            .map(|r| {
                let ys: Vec<Vec<f64>> = pos_half_reps.iter().map(|g| g[r].clone()).collect();
                shared_h(&half_pts(&ys)).map_or(f64::NAN, |x| x.0)
            })
            .collect();
        est(x.0, &mut reps)
    });

    // Prediction grid: log-spaced from the smallest grid N to 10^6 (12 per decade).
    let n0 = grid.first().copied().unwrap_or(1) as f64;
    let pred_n: Vec<f64> = (0..)
        .map(|i| n0 * 10f64.powf(f64::from(i) / 12.0))
        .take_while(|&n| n <= 1e6 * 1.0001)
        .collect();

    let mut reports = Vec::new();
    let (mut ho_in, mut ho_tot, mut ho_lr, mut ho_lrn) = (0, 0, 0.0, 0);
    let (mut hf_in, mut hf_tot, mut hf_lr, mut hf_lrn) = (0, 0, 0.0, 0);
    for g in &groups {
        let plain_reps = replicates(&mm.plain, &g.members, &counts);
        let prefix_reps = replicates(&mm.prefix, &g.members, &counts);
        let cv_reps = replicates(&mm.cv, &g.members, &counts);
        let half_reps = replicates(&mm.cv_half, &g.members, &counts);
        let plain = measured(&mm.plain, &g.members, &plain_reps);
        let prefix = measured(&mm.prefix, &g.members, &prefix_reps);
        let cv = measured(&mm.cv, &g.members, &cv_reps);
        let plain_change = group_mean(&mm.wrong, &g.members, None);
        let rse = rel_se(&cv.mean, &cv_reps);
        let fp = fit_points(grid_k, &cv.mean, &rse, fit_max);
        let la = b0
            .filter(|_| !fp.ln_n.is_empty())
            .map(|b| fit_a(&fp, k as f64, b));
        // A replicate with no positive point in the fit range measured no loss at all: a = 0.
        let la_reps: Vec<f64> = (0..counts.len())
            .map(|r| {
                let fpr = fit_points(grid_k, &cv_reps[r], &rse, fit_max);
                if !b_reps[r].is_finite() {
                    f64::NAN
                } else if fpr.ln_n.is_empty() {
                    f64::NEG_INFINITY
                } else {
                    fit_a(&fpr, k as f64, b_reps[r])
                }
            })
            .collect();
        let a = la.map(|v| {
            let mut x: Vec<f64> = la_reps.iter().map(|l| l.exp()).collect();
            est(v.exp(), &mut x)
        });
        let free_b = if g.main && fp.ln_n.len() >= 3 {
            fit_shared(&[&fp], k as f64).map(|(bf, _)| {
                let mut reps: Vec<f64> = (0..counts.len())
                    .map(|r| {
                        let fpr = fit_points(grid_k, &cv_reps[r], &rse, fit_max);
                        if fpr.ln_n.len() < 3 {
                            f64::NAN
                        } else {
                            fit_shared(&[&fpr], k as f64).map_or(f64::NAN, |x| x.0)
                        }
                    })
                    .collect();
                est(bf, &mut reps)
            })
        } else {
            None
        };
        let model = |la: f64, b: f64, n: f64, kk: f64| la.exp() * (n.powf(-b) - kk.powf(-b));
        let fit_cv: Vec<f64> = grid_k
            .iter()
            .map(|&n| match (la, b0) {
                (Some(la), Some(b)) => model(la, b, n as f64, k as f64),
                _ => f64::NAN,
            })
            .collect();
        let checked = g.kind == "all" || g.kind == "position";
        let mut heldout = Vec::new();
        if let (Some(la), Some(b)) = (la, b0) {
            for (i, &n) in grid_k.iter().enumerate() {
                if n > fit_max && n < k {
                    let pred = model(la, b, n as f64, k as f64);
                    let inside = pred >= cv.lo[i] && pred <= cv.hi[i];
                    if checked {
                        ho_tot += 1;
                        ho_in += usize::from(inside);
                        if cv.mean[i] > 0.0 {
                            ho_lr += (pred / cv.mean[i]).ln();
                            ho_lrn += 1;
                        }
                    }
                    heldout.push(Check {
                        n,
                        pred,
                        measured: cv.mean[i],
                        inside,
                    });
                }
            }
        }
        let mut half = Vec::new();
        if let Some(bh) = b_half.map(|e| e.v) {
            let hm = group_mean(&mm.cv_half, &g.members, None);
            let hp = fit_points(grid_kh, &hm, &rel_se(&hm, &half_reps), fit_max_h);
            if !hp.ln_n.is_empty() {
                let lah = fit_a(&hp, kh as f64, bh);
                for (i, &n) in grid_k.iter().enumerate() {
                    if n < k {
                        let pred = model(lah, bh, n as f64, k as f64);
                        let inside = pred >= cv.lo[i] && pred <= cv.hi[i];
                        if checked {
                            hf_tot += 1;
                            hf_in += usize::from(inside);
                            if cv.mean[i] > 0.0 {
                                hf_lr += (pred / cv.mean[i]).ln();
                                hf_lrn += 1;
                            }
                        }
                        half.push(Check {
                            n,
                            pred,
                            measured: cv.mean[i],
                            inside,
                        });
                    }
                }
            }
        }
        let loss: Vec<Est> = match (la, b0) {
            (Some(la), Some(b)) => pred_n
                .iter()
                .map(|&n| {
                    let mut reps: Vec<f64> = la_reps
                        .iter()
                        .zip(&b_reps)
                        .map(|(&l, &br)| (l - br * n.ln()).exp())
                        .collect();
                    est((la - b * n.ln()).exp(), &mut reps)
                })
                .collect(),
            _ => Vec::new(),
        };
        let n_for_v: Vec<Est> = match (la, b0) {
            (Some(la), Some(b)) => p
                .thresholds
                .iter()
                .map(|&th| {
                    let mut reps: Vec<f64> = la_reps
                        .iter()
                        .zip(&b_reps)
                        .map(|(&l, &br)| n_for(l, br, th))
                        .collect();
                    // A replicate that cannot fit counts as "never reached".
                    for x in &mut reps {
                        if x.is_nan() {
                            *x = N_CAP;
                        }
                    }
                    est(n_for(la, b, th), &mut reps)
                })
                .collect(),
            _ => Vec::new(),
        };
        reports.push(GroupReport {
            kind: g.kind.clone(),
            name: g.name.clone(),
            decisions: g.members.len(),
            enough: g.members.len() >= ENOUGH,
            plain,
            plain_change,
            prefix,
            cv,
            a,
            free_b,
            fit_cv,
            heldout,
            half,
            loss,
            n_for: n_for_v,
        });
    }
    Some(TrickReport {
        trick,
        decisions: used.len(),
        dropped,
        worlds: m,
        k,
        fit_max,
        grid,
        splits: p.splits,
        boots: p.boots,
        seed: p.seed,
        thresholds: p.thresholds.clone(),
        b,
        b_half,
        b_by_range,
        pred_n,
        heldout_inside: (ho_in, ho_tot),
        half_inside: (hf_in, hf_tot),
        heldout_log_ratio: if ho_lrn > 0 {
            ho_lr / ho_lrn as f64
        } else {
            f64::NAN
        },
        half_log_ratio: if hf_lrn > 0 {
            hf_lr / hf_lrn as f64
        } else {
            f64::NAN
        },
        groups: reports,
    })
}

/// Analyse every trick in `rows` (merged worlds records of one method and label).
pub fn analyse(rows: &[WorldRows], p: &CurveParams) -> Vec<TrickReport> {
    let mut by: BTreeMap<u8, Vec<&WorldRows>> = BTreeMap::new();
    for r in rows {
        if r.cands.len() >= 2 {
            by.entry(r.trick).or_default().push(r);
        }
    }
    by.into_iter()
        .filter_map(|(t, rs)| analyse_trick(t, &rs, p))
        .collect()
}

// ---------------------------------------------------------------- methods against the yardstick

/// One group of a method scored against the yardstick (§13 item 2).
#[derive(Clone, Debug, Serialize)]
pub struct ScoreGroup {
    /// `all`, `pos1`…`pos8`, `S1`…`S4`.
    pub name: String,
    /// Decisions.
    pub decisions: usize,
    /// Share of decisions whose choice is among the yardstick's tied-best.
    pub agreement: Est,
    /// Mean money-chance loss of the choice under the yardstick's values.
    pub loss: Est,
    /// The largest single loss (money chance).
    pub max_loss: f64,
    /// The decision with that loss.
    pub max_loss_id: Option<u64>,
}

/// A method scored against the yardstick, per group, with bootstrap ranges.
#[derive(Clone, Debug, Serialize)]
pub struct MethodScore {
    /// Display name (`FILE:N` as given).
    pub method: String,
    /// The method's recorded name (`rollout`, `exact`).
    pub kind: String,
    /// The method's worlds used.
    pub n: usize,
    /// The yardstick's worlds used.
    pub n_ref: usize,
    /// The trick of the decisions (the first one's).
    pub trick: Option<u8>,
    /// Groups: all, the positions, the strata.
    pub groups: Vec<ScoreGroup>,
}

/// Score a method's first `n` worlds against the yardstick's first `n_ref` (as
/// [`crate::measure::score`]: agreement and loss under the yardstick's means), per group, with
/// 95% ranges from a bootstrap of decisions within position × stratum cells.
pub fn score_groups(
    name: &str,
    yard: &[WorldRows],
    n_ref: usize,
    method: &[WorldRows],
    n: usize,
    boots: u64,
    seed: u64,
) -> MethodScore {
    let scored = crate::measure::score(yard, n_ref, method, n);
    let meta: Vec<(u8, u8)> = scored
        .iter()
        .map(|s| {
            let y = yard
                .iter()
                .find(|y| y.id == s.id)
                .expect("scored on the yardstick");
            (y.position, y.stratum)
        })
        .collect();
    let agree: Vec<f64> = scored
        .iter()
        .map(|s| f64::from(u8::from(s.agree)))
        .collect();
    let loss: Vec<f64> = scored.iter().map(|s| s.loss as f64 / ONE as f64).collect();
    let counts = boot_cells(&meta, boots, seed, 0x5C0E);
    let mean = |x: &[f64], members: &[usize], w: Option<&[u32]>| {
        let mut s = 0.0;
        let mut k = 0.0;
        for &i in members {
            let wi = w.map_or(1.0, |w| f64::from(w[i]));
            s += wi * x[i];
            k += wi;
        }
        if k > 0.0 { s / k } else { f64::NAN }
    };
    let mut groups: Vec<(String, Vec<usize>)> = vec![("all".into(), (0..scored.len()).collect())];
    for p in 0..8u8 {
        groups.push((
            format!("pos{}", p + 1),
            (0..scored.len()).filter(|&i| meta[i].0 == p).collect(),
        ));
    }
    for s in 1..=4u8 {
        groups.push((
            format!("S{s}"),
            (0..scored.len()).filter(|&i| meta[i].1 == s).collect(),
        ));
    }
    let groups = groups
        .into_iter()
        .filter(|(_, m)| !m.is_empty())
        .map(|(gname, m)| {
            let e = |x: &[f64]| {
                let mut reps: Vec<f64> = counts.iter().map(|c| mean(x, &m, Some(c))).collect();
                est(mean(x, &m, None), &mut reps)
            };
            ScoreGroup {
                name: gname,
                decisions: m.len(),
                agreement: e(&agree),
                loss: e(&loss),
                max_loss: m.iter().map(|&i| loss[i]).fold(0.0, f64::max),
                max_loss_id: m
                    .iter()
                    .copied()
                    .max_by(|&i, &j| loss[i].total_cmp(&loss[j]).then(j.cmp(&i)))
                    .map(|i| scored[i].id),
            }
        })
        .collect();
    MethodScore {
        method: name.to_string(),
        kind: method.first().map_or(String::new(), |r| r.method.clone()),
        n,
        n_ref,
        trick: yard.first().map(|r| r.trick),
        groups,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::{HandType, mean_first, tied_best};
    use engine::game::Action;

    /// A money chance with team 0's entry `v`.
    fn mc0(v: u64) -> [u64; 4] {
        let q = (ONE - v) / 3;
        [v, ONE - v - 2 * q, q, q]
    }

    fn record(id: u64, position: u8, stratum: u8, rows: Vec<Vec<[u64; 4]>>) -> WorldRows {
        let c = rows[0].len();
        WorldRows {
            id,
            trick: 3,
            stratum,
            position,
            team: 0,
            method: "exact".into(),
            label: 1,
            from: 0,
            cands: (0..c).map(|j| (Action::Play(j as u8), None)).collect(),
            rows,
            log2_worlds: None,
            hand_type: Some(HandType {
                trumps: (id % 4) as u8,
                top_trumps: (id % 3) as u8,
                longest_plain: 2,
            }),
        }
    }

    /// Random decisions: `c` candidates, team-0 values around 0.4 with per-candidate offsets.
    fn random_set(n: u64, m: usize, seed: u64) -> Vec<WorldRows> {
        (0..n)
            .map(|id| {
                let mut rng = Rng::derive(seed, &[id]);
                let c = 2 + rng.bounded(2) as usize;
                let off: Vec<u64> = (0..c).map(|_| rng.bounded(1 << 22)).collect();
                let rows = (0..m)
                    .map(|_| {
                        let base = (ONE / 5) * 2 + rng.bounded(1 << 26);
                        (0..c)
                            .map(|j| mc0(base + off[j] + rng.bounded(1 << 25)))
                            .collect()
                    })
                    .collect();
                record(id, (id % 8) as u8, (id % 4) as u8 + 1, rows)
            })
            .collect()
    }

    #[test]
    fn prefix_choice_is_the_agreement_choice() {
        let grid: Vec<usize> = (1..=16).map(|i| 4 * i).collect();
        for r in random_set(12, 64, 3) {
            let c = decision_curve(&r, 64, &grid, 3, 1);
            let (vm, cm) = mean_first(&r, 64);
            for (g, &n) in grid.iter().enumerate() {
                let (_, cn) = mean_first(&r, n);
                assert_eq!(c.prefix[g], vm[cm].get(0) - vm[cn].get(0), "N = {n}");
                assert_eq!(c.prefix[g] == 0, tied_best(&vm, 0, cn));
            }
            assert_eq!(c.prefix[grid.len() - 1], 0, "N = M agrees with itself");
            assert_eq!(c.plain[grid.len() - 1], 0);
            assert_eq!(c.cv.len(), 8);
            assert_eq!(c.cv[7], 0, "N = K is the reference");
            assert_eq!(c.cv_half.len(), 4);
        }
    }

    #[test]
    fn equal_or_dominated_candidates_lose_nothing() {
        let grid = [1, 2, 5, 10, 20, 40];
        let mut rng = Rng::derive(5, &[]);
        let same: Vec<Vec<[u64; 4]>> = (0..40)
            .map(|_| {
                let v = mc0(rng.bounded(ONE / 2));
                vec![v; 3]
            })
            .collect();
        let dom: Vec<Vec<[u64; 4]>> = (0..40)
            .map(|_| {
                let v = rng.bounded(ONE / 2);
                vec![mc0(v), mc0(v + 1000), mc0(v + 7)]
            })
            .collect();
        for rows in [same, dom] {
            let c = decision_curve(&record(1, 0, 1, rows), 40, &grid, 50, 2);
            assert!(c.prefix.iter().all(|&x| x == 0));
            assert!(c.plain.iter().all(|&x| x == 0));
            assert!(c.plain_wrong.iter().all(|&x| x == 0));
            assert!(c.cv.iter().chain(&c.cv_half).all(|&x| x == 0));
        }
    }

    /// The plain estimate is the exact subset average: with one world favouring the second
    /// candidate (ties go to the first), an N-subset misses it with chance (M − N)/M.
    #[test]
    fn plain_matches_exact_subset_probability() {
        let m = 40;
        let d = 1u64 << 20;
        let rows: Vec<Vec<[u64; 4]>> = (0..m)
            .map(|w| {
                let v = ONE / 3;
                vec![mc0(v), mc0(if w == 7 { v + d } else { v })]
            })
            .collect();
        let grid = [5, 10, 20, 30];
        let splits = 4000;
        let c = decision_curve(&record(9, 0, 1, rows), m, &grid, splits, 3);
        let loss = d / m as u64;
        for (g, &n) in grid.iter().enumerate() {
            let want = (m - n) as f64 / m as f64;
            let got = c.plain[g] as f64 / (splits as f64 * loss as f64);
            assert!((got - want).abs() < 0.03, "N {n}: {got} vs {want}");
            assert_eq!(c.plain_wrong[g] as f64 / splits as f64, got);
        }
        // The scoring half never shows a difference both choices do not share.
        assert!(c.cv.iter().all(|&x| x == 0));
    }

    /// The cross-fitted estimate is unbiased for L(N) − L(K): two candidates whose value
    /// difference per world is +X with chance 0.6 and −X otherwise, where L(N) is known exactly
    /// (the first candidate, the worse, is chosen unless heads exceed N/2).
    #[test]
    fn cross_fitted_is_unbiased() {
        let (m, x) = (32usize, 1u64 << 24);
        let n_dec = 600;
        let grid = [1, 2, 3, 4, 6, 8, 12, 16];
        let mut rng = Rng::derive(77, &[]);
        let mut per: Vec<Vec<f64>> = vec![Vec::new(); grid.len()];
        for id in 0..n_dec {
            let rows: Vec<Vec<[u64; 4]>> = (0..m)
                .map(|_| {
                    let base = ONE / 3 + rng.bounded(1 << 28);
                    let heads = rng.bounded(10) < 6;
                    vec![mc0(base), mc0(if heads { base + x } else { base - x })]
                })
                .collect();
            let c = decision_curve(&record(id, 0, 1, rows), m, &grid, 64, 4);
            for (g, v) in c.cv.iter().enumerate() {
                per[g].push(*v as f64 / 64.0 / (m / 2) as f64 / x as f64);
            }
        }
        // L(N) in units of X: 0.2 · P(heads ≤ N/2), heads ~ Binomial(N, 0.6).
        let l = |n: usize| {
            let mut p = 0.0;
            for h in 0..=n / 2 {
                let mut b = 1.0;
                for i in 0..h {
                    b *= (n - i) as f64 / (i + 1) as f64;
                }
                p += b * 0.6f64.powi(h as i32) * 0.4f64.powi((n - h) as i32);
            }
            0.2 * p
        };
        for (g, &n) in grid.iter().enumerate() {
            let k = per[g].len() as f64;
            let mean = per[g].iter().sum::<f64>() / k;
            let se = sd(&per[g]) / k.sqrt();
            let want = l(n) - l(m / 2);
            assert!(
                (mean - want).abs() < 4.0 * se + 1e-9,
                "N {n}: {mean} vs {want} (se {se})"
            );
        }
    }

    #[test]
    fn fit_recovers_a_shared_power_law() {
        let k: f64 = 160.0;
        let (b, a) = (0.7, [3e-3, 1e-3, 5e-4]);
        let grid: Vec<usize> = (1..=8).map(|i| 10 * i).collect();
        let pts: Vec<FitPoints> = a
            .iter()
            .map(|&ai| {
                let y: Vec<f64> = grid
                    .iter()
                    .map(|&n| ai * ((n as f64).powf(-b) - k.powf(-b)))
                    .collect();
                fit_points(&grid, &y, &vec![0.1; grid.len()], 80)
            })
            .collect();
        let (bf, las) = fit_shared(&pts.iter().collect::<Vec<_>>(), k).expect("fits");
        assert!((bf - b).abs() < 1e-4, "{bf}");
        for (la, ai) in las.iter().zip(a) {
            assert!((la.exp() / ai - 1.0).abs() < 1e-3);
        }
        assert!((fit_a(&pts[1], k, b).exp() / a[1] - 1.0).abs() < 1e-9);
        assert!((n_for(a[0].ln(), b, 1e-4) - (a[0] / 1e-4f64).powf(1.0 / b)).abs() < 1e-6);
        assert_eq!(n_for(f64::NEG_INFINITY, b, 1e-4), 1.0);
    }

    #[test]
    fn analysis_is_thread_independent_and_deterministic() {
        let rs = random_set(64, 48, 11);
        let p = CurveParams {
            grid: (1..=12).map(|i| 4 * i).collect(),
            splits: 20,
            boots: 30,
            min_cell: 4,
            ..CurveParams::default()
        };
        let run = |threads: usize| {
            let r = analyse(
                &rs,
                &CurveParams {
                    threads,
                    ..p.clone()
                },
            );
            serde_json::to_string(&r).expect("json")
        };
        let one = run(1);
        assert_eq!(one, run(3));
        assert_eq!(one, run(1));
        let r = analyse(&rs, &p);
        assert_eq!(r.len(), 1);
        let t = &r[0];
        assert_eq!((t.worlds, t.k, t.fit_max, t.decisions), (48, 24, 12, 64));
        assert!(t.groups.iter().any(|g| g.name == "pos1·S1"));
        assert_eq!(t.groups[0].name, "all");
        assert_eq!(t.groups[0].n_for.len(), 3);
    }

    #[test]
    fn method_scores_match_the_plain_scorer() {
        // The method sees 12 other worlds of the same decisions.
        let all = random_set(48, 52, 21);
        let yard: Vec<WorldRows> = all
            .iter()
            .map(|r| WorldRows {
                rows: r.rows[..40].to_vec(),
                ..r.clone()
            })
            .collect();
        let method: Vec<WorldRows> = all
            .iter()
            .map(|r| WorldRows {
                method: "rollout".into(),
                label: 2,
                rows: r.rows[40..].to_vec(),
                ..r.clone()
            })
            .collect();
        let ms = score_groups("m:12", &yard, 40, &method, 12, 50, 1);
        let plain = crate::measure::summarise(&crate::measure::score(&yard, 40, &method, 12));
        let all = &ms.groups[0];
        assert_eq!((all.name.as_str(), all.decisions), ("all", 48));
        assert!((all.agreement.v - plain.agreement.0).abs() < 1e-12);
        assert!((all.loss.v - plain.mean_loss.0).abs() < 1e-12);
        assert!(all.loss.lo <= all.loss.v && all.loss.v <= all.loss.hi);
        assert_eq!(ms.groups.len(), 13);
        assert_eq!(ms.kind, "rollout");
        let pos_total: usize = ms.groups[1..9].iter().map(|g| g.decisions).sum();
        assert_eq!(pos_total, 48);
    }
}
