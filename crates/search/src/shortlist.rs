//! Shortlist + search at the app's deal counts, and the router (`search student-check
//! --sub-deals`, docs/m3-hybrid/path-forward.md section 6, session 0).
//!
//! A phone can afford only N = 25-100 deals a decision. Per golden decision the stored exact
//! worlds are sampled: N of them are drawn (a seeded shuffle, nested prefixes so one draw serves
//! every N), a method picks a card from those N, and the pick is **scored on the other worlds**
//! (no winner's curse). The methods: the student alone (no sample used), search over all
//! candidates, and the student's top 2 and top 3 candidates searched.
//!
//! Two scores per method, both whole numbers over the common denominator
//! `repeats * (M - N) * M` (M worlds, units of 2^-32):
//! * **raw**: the best candidate of the scoring worlds less the pick, on the scoring worlds. The
//!   best of a finite sample is itself noisy and biased upward, so raw levels sit above the
//!   full-world loss; they compare methods fairly but are not on the golden check's scale.
//! * **corrected**: the student's loss on all M worlds less the method's gain over the student on
//!   the scoring worlds (`pick - student`, both scored out of sample). The common best-of-sample
//!   term cancels, so this is an unbiased estimate of the method's loss on the golden scale
//!   (it equals the student's own loss for the student, by construction).

use engine::rng::Rng;
use equity::MoneyChance;

use crate::curves::{choose_among, top_k_choice, totals_over};

/// The methods compared at each N, in the order of [`SubRecord::scores`].
pub const METHODS: [&str; 4] = ["student", "search-all", "top-2+search", "top-3+search"];

/// One method at one N on one decision, summed over the repeats.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SubScore {
    /// Raw loss, `raw` in the module docs, times `M` (so the denominator is `repeats*(M-N)*M`).
    pub raw: i128,
    /// Corrected loss on the same denominator.
    pub corrected: i128,
}

/// One decision's sub-deal results: per N of the grid, per method of [`METHODS`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubRecord {
    /// `scores[n][m]` for grid entry `n`, method `m`.
    pub scores: Vec<[SubScore; 4]>,
}

/// The sub-deal measure of one decision.
///
/// `rows[world][cand]` are the exact values (real pair order), `team` the deciding pair,
/// `ranking` the student's candidates best first, `grid` the sub-deal counts (each below
/// `rows.len()`), `repeats` the draws per N, and `key` identifies the decision for the seeded
/// draws. The same arguments give the same record on any thread count.
///
/// # Panics
/// If a grid entry is zero or not below the number of worlds, or `rows` is empty.
pub fn sub_deal_record(
    rows: &[Vec<[u64; 4]>],
    team: u8,
    ranking: &[usize],
    grid: &[usize],
    repeats: u32,
    seed: u64,
    key: u64,
) -> SubRecord {
    let m = rows.len();
    let c = rows[0].len();
    let t = usize::from(team);
    let all: Vec<usize> = (0..m).collect();
    let totals_all = totals_over(rows, &all);
    let student = ranking[0];
    // The student's loss on all worlds, as a total (best total less its own).
    let best_all = choose_among(&totals_all, m as u64, team, 0..c);
    let student_full = i128::from(totals_all[best_all][t]) - i128::from(totals_all[student][t]);
    let mut scores = vec![[SubScore::default(); 4]; grid.len()];
    for r in 0..repeats {
        let mut rng = Rng::derive(seed, &[0x5B01, key, u64::from(r)]);
        let mut perm = all.clone();
        rng.shuffle(&mut perm);
        for (gi, &n) in grid.iter().enumerate() {
            assert!(n >= 1 && n < m, "sub-deal count {n} must be in 1..{m}");
            let (picked, rest) = perm.split_at(n);
            let tp = totals_over(rows, picked);
            let ts = totals_over(rows, rest);
            let nn = n as u64;
            let score = |j: usize| i128::from(ts[j][t]);
            let best_rest = choose_among(&ts, (m - n) as u64, team, 0..c);
            let picks = [
                student,
                choose_among(&tp, nn, team, 0..c),
                top_k_choice(&tp, nn, team, ranking, 2),
                top_k_choice(&tp, nn, team, ranking, 3),
            ];
            let width = (m - n) as i128;
            for (mi, &p) in picks.iter().enumerate() {
                let s = &mut scores[gi][mi];
                s.raw += (score(best_rest) - score(p)) * m as i128;
                // student_full is over m worlds: bring it to the (m - n) worlds' scale.
                s.corrected += student_full * width - (score(p) - score(student)) * m as i128;
            }
        }
    }
    SubRecord { scores }
}

/// The denominator of a [`SubScore`] of grid entry `n`: `repeats * (M - N) * M`.
pub fn denominator(m: usize, n: usize, repeats: u32) -> u128 {
    u128::from(repeats) * (m - n) as u128 * m as u128
}

/// The student's margin on a decision: its top value less its second for the deciding pair
/// (units of 2^-32; a single candidate has no margin and is never judged).
pub fn margin(student: &[MoneyChance], team: u8, ranking: &[usize]) -> u64 {
    match ranking {
        [a, b, ..] => student[*a].get(team).saturating_sub(student[*b].get(team)),
        _ => u64::MAX,
    }
}

/// Router: per decision the loss (whole numbers, units of 2^-32 times `den`) when the decisions
/// flagged `searched` use the searched method's loss and the rest keep the student's.
pub fn route(student_loss: &[i128], searched_loss: &[i128], searched: &[bool]) -> Vec<i128> {
    student_loss
        .iter()
        .zip(searched_loss)
        .zip(searched)
        .map(|((&s, &q), &on)| if on { q } else { s })
        .collect()
}

/// The `share` (per 100) of decisions with the lowest `key`, ties by index: those to search.
pub fn lowest_share(key: &[u64], per_100: u64) -> Vec<bool> {
    let mut order: Vec<usize> = (0..key.len()).collect();
    order.sort_by_key(|&i| (key[i], i));
    let k = (key.len() as u64 * per_100).div_ceil(100) as usize;
    let mut on = vec![false; key.len()];
    for &i in order.iter().take(k) {
        on[i] = true;
    }
    on
}

/// The share (in 1e-4) of the total of `loss` held by its worst `per_100` percent of items
/// (ties by index), `0` when the total is zero.
pub fn worst_share_bp(loss: &[u64], per_100: u64) -> u64 {
    let total: u128 = loss.iter().map(|&l| u128::from(l)).sum();
    if total == 0 {
        return 0;
    }
    let mut order: Vec<usize> = (0..loss.len()).collect();
    order.sort_by_key(|&i| (core::cmp::Reverse(loss[i]), i));
    let k = (loss.len() as u64 * per_100).div_ceil(100) as usize;
    let held: u128 = order.iter().take(k).map(|&i| u128::from(loss[i])).sum();
    (held * 10_000 / total) as u64
}

/// One judged decision, as the sub-deal report needs it.
pub struct SubDecision {
    /// Trick.
    pub trick: u8,
    /// Place in the trick (0 = leader).
    pub place: u8,
    /// Stratum (1-4).
    pub stratum: u8,
    /// The student's margin ([`margin`]).
    pub margin: u64,
    /// The student's loss on all worlds (units of 2^-32, a mean).
    pub loss: u64,
    /// Candidates.
    pub n_cands: u8,
    /// The student's own values for the deciding pair, best first (units of 2^-32): what a router
    /// can see without any search.
    pub values: Vec<u64>,
    /// The sub-deal results.
    pub rec: SubRecord,
}

/// The percentiles of the concentration table.
const WORST: [u64; 6] = [1, 5, 10, 20, 40, 100];
/// The shares of decisions searched in the router table.
const ROUTE: [u64; 8] = [0, 5, 10, 20, 30, 50, 75, 100];

/// Print the sub-deal tables (all methods per N; the paired gain over the student; the
/// concentration of the student's loss; the router) and return them as JSON.
///
/// `worlds` is the number of exact worlds per decision (M). Bounds are bootstrap 95% over
/// decisions (`boots` replicates from `boot_seed`).
pub fn report(
    ds: &[SubDecision],
    grid: &[usize],
    worlds: usize,
    repeats: u32,
    boots: u64,
    boot_seed: u64,
) -> serde_json::Value {
    use serde_json::json;
    use student::check::{Scored, print_measure, report_measure};
    let key = |d: &SubDecision, v: Vec<i128>| Scored {
        trick: d.trick,
        place: d.place,
        stratum: d.stratum,
        v,
    };
    let mut out = serde_json::Map::new();
    // Concentration of the student's own loss.
    let losses: Vec<u64> = ds.iter().map(|d| d.loss).collect();
    let nonzero = losses.iter().filter(|&&l| l > 0).count();
    println!(
        "concentration of the student's loss ({} decisions, {} with a loss):",
        ds.len(),
        nonzero
    );
    let mut conc = Vec::new();
    for p in WORST {
        let bp = worst_share_bp(&losses, p);
        println!(
            "  worst {p:>3}% of decisions hold {:>6.2}% of the loss",
            bp as f64 / 100.0
        );
        conc.push(json!({ "worst_pct": p, "share_pct": bp as f64 / 100.0 }));
    }
    out.insert("concentration".into(), json!(conc));
    let mut per_n = Vec::new();
    for (gi, &n) in grid.iter().enumerate() {
        let den = denominator(worlds, n, repeats);
        let cols = [("raw", den), ("corrected", den)];
        let mut methods = Vec::new();
        println!(
            "\n=== {n} deals a decision ({repeats} draws each; scored on the other {}) ===",
            worlds - n
        );
        for (mi, name) in METHODS.iter().enumerate() {
            let sc: Vec<Scored> = ds
                .iter()
                .map(|d| {
                    let s = d.rec.scores[gi][mi];
                    key(d, vec![s.raw, s.corrected])
                })
                .collect();
            let rows = report_measure(&sc, &cols, boots, boot_seed);
            println!(
                "{name}: raw loss (best of the scoring worlds less the pick) and corrected loss (golden scale):"
            );
            print!("{}", print_measure(&rows, &["raw", "corrected"]));
            methods.push(json!({ "method": name, "rows": rows }));
        }
        // The paired gain over the student: student minus method, corrected.
        let gcols = [("gain", den)];
        let mut gains = Vec::new();
        for (mi, name) in METHODS.iter().enumerate().skip(1) {
            let sc: Vec<Scored> = ds
                .iter()
                .map(|d| {
                    let s = d.rec.scores[gi][0].corrected - d.rec.scores[gi][mi].corrected;
                    key(d, vec![s])
                })
                .collect();
            let rows = report_measure(&sc, &gcols, boots, boot_seed);
            println!(
                "gain of {name} over the student alone (positive = better), bounds of the gain:"
            );
            print!("{}", print_measure(&rows, &["gain"]));
            gains.push(json!({ "method": name, "rows": rows }));
        }
        // A shortlist against searching every card: top-k minus search-all (positive = the
        // shortlist is better), the same draws.
        let mut vs_all = Vec::new();
        for (mi, name) in METHODS.iter().enumerate().skip(2) {
            let sc: Vec<Scored> = ds
                .iter()
                .map(|d| {
                    let s = d.rec.scores[gi][1].corrected - d.rec.scores[gi][mi].corrected;
                    key(d, vec![s])
                })
                .collect();
            let rows = report_measure(&sc, &gcols, boots, boot_seed);
            println!("gain of {name} over search-all (positive = the shortlist is better):");
            print!("{}", print_measure(&rows, &["gain"]));
            vs_all.push(json!({ "method": name, "rows": rows }));
        }
        // The router: search (top-3 + N deals) the lowest-margin share, keep the student's card
        // for the rest; and the oracle (the share with the largest true loss).
        let student: Vec<i128> = ds.iter().map(|d| d.rec.scores[gi][0].corrected).collect();
        let top3: Vec<i128> = ds.iter().map(|d| d.rec.scores[gi][3].corrected).collect();
        let margins: Vec<u64> = ds.iter().map(|d| d.margin).collect();
        let worst: Vec<u64> = ds.iter().map(|d| u64::MAX - d.loss).collect();
        println!(
            "router at {n} deals (top-3 + search on the share of decisions with the smallest student margin; oracle = the share with the largest true loss):"
        );
        println!(
            "{:>7} {:>11} {:>10} {:>10} {:>11} {:>10}",
            "searched", "margin %", "lo95", "hi95", "oracle %", "hi95"
        );
        let mut router = Vec::new();
        for f in ROUTE {
            let by_margin = route(&student, &top3, &lowest_share(&margins, f));
            let by_oracle = route(&student, &top3, &lowest_share(&worst, f));
            let (m, mlo, mhi) = boot_mean(&by_margin, den, boots, boot_seed, 0x5C50 + f);
            let (o, _, ohi) = boot_mean(&by_oracle, den, boots, boot_seed, 0x5C50 + f);
            println!("{f:>6}% {m:>11.5} {mlo:>10.5} {mhi:>10.5} {o:>11.5} {ohi:>10.5}");
            router.push(
                json!({ "searched_pct": f, "margin_router_pct": m, "lower95_pct": mlo,
                "upper95_pct": mhi, "oracle_router_pct": o, "oracle_upper95_pct": ohi }),
            );
        }
        per_n.push(json!({ "deals": n, "methods": methods, "gains": gains, "vs_search_all": vs_all, "router": router }));
    }
    out.insert("by_deals".into(), json!(per_n));
    // Per decision (for fitting a better router offline): place, stratum, candidates, the
    // student's values best first, its loss, and its / top-3+search's corrected loss per N.
    out.insert(
        "per_decision".into(),
        json!(
            ds.iter()
                .map(|d| json!({
                    "place": d.place, "stratum": d.stratum, "n_cands": d.n_cands,
                    "values": d.values, "loss": d.loss,
                    "top3_corrected": d.rec.scores.iter().map(|s| s[3].corrected.to_string()).collect::<Vec<_>>(),
                }))
                .collect::<Vec<_>>()
        ),
    );
    serde_json::Value::Object(out)
}

/// Mean of per-decision whole numbers (over `den`) in % of money chance with the bootstrap 5% and
/// 95% points of the mean over decisions.
fn boot_mean(v: &[i128], den: u128, boots: u64, seed: u64, tag: u64) -> (f64, f64, f64) {
    let n = v.len() as u64;
    let pct = |sum: i128| 100.0 * (sum as f64 / n as f64 / den as f64 / equity::ONE as f64);
    let total: i128 = v.iter().sum();
    if boots == 0 || n == 0 {
        return (pct(total), pct(total), pct(total));
    }
    let mut rng = Rng::derive(seed, &[0x5C4F, tag]);
    let mut reps: Vec<i128> = (0..boots)
        .map(|_| (0..n).map(|_| v[rng.bounded(n) as usize]).sum())
        .collect();
    reps.sort_unstable();
    (
        pct(total),
        pct(reps[(boots * 5 / 100) as usize]),
        pct(reps[((boots * 95).div_ceil(100) as usize).min(reps.len()) - 1]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three candidates, 8 worlds: candidate 1 is the best on average (team 0), the student
    /// plays candidate 0, whose mean is lower.
    fn rows() -> Vec<Vec<[u64; 4]>> {
        (0..8u64)
            .map(|w| {
                [900u64, 1000, 600]
                    .iter()
                    .map(|&v| {
                        let x = (v + w) * 1_000_000;
                        [x, equity::ONE - x, 0, 0]
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn the_student_scores_its_own_full_loss_corrected() {
        let rows = rows();
        let rec = sub_deal_record(&rows, 0, &[0, 1, 2], &[2, 4], 3, 7, 11);
        let full_loss: i128 = 8 * 100 * 1_000_000; // best total less the student's, over 8 worlds
        for (gi, n) in [2usize, 4].into_iter().enumerate() {
            let s = rec.scores[gi][0];
            assert_eq!(s.corrected, full_loss * 3 * (8 - n) as i128);
            // The student's raw loss: the best of the scoring worlds less candidate 0.
            assert_eq!(s.raw, 100 * 1_000_000 * 3 * (8 - n) as i128 * 8);
        }
    }

    #[test]
    fn a_clean_gap_makes_search_find_the_best_and_the_loss_vanish() {
        let rows = rows();
        let rec = sub_deal_record(&rows, 0, &[0, 1, 2], &[3], 4, 7, 11);
        // Search picks candidate 1 on every sample: raw loss 0; corrected: the student's loss
        // less the whole gap (both 100 per world), so 0.
        for (s, name) in rec.scores[0].iter().zip(METHODS).skip(1) {
            assert_eq!((s.raw, s.corrected), (0, 0), "{name}");
        }
    }

    #[test]
    fn draws_depend_on_the_seed_and_key_only() {
        let rows = rows();
        let a = sub_deal_record(&rows, 0, &[2, 0, 1], &[2, 3], 5, 1, 9);
        let b = sub_deal_record(&rows, 0, &[2, 0, 1], &[2, 3], 5, 1, 9);
        assert_eq!(a, b);
        // Grid entries share one shuffle per repeat: the N = 3 sample extends the N = 2 one.
        let one = sub_deal_record(&rows, 0, &[2, 0, 1], &[3], 5, 1, 9);
        assert_eq!(one.scores[0], a.scores[1]);
    }

    #[test]
    fn route_and_shares() {
        let on = lowest_share(&[5, 1, 3, 9], 50);
        assert_eq!(on, vec![false, true, true, false]);
        assert_eq!(lowest_share(&[5, 1, 3, 9], 0), vec![false; 4]);
        assert_eq!(lowest_share(&[5, 1, 3, 9], 100), vec![true; 4]);
        assert_eq!(route(&[10, 20, 30, 40], &[1, 2, 3, 4], &on), [10, 2, 3, 40]);
        // Losses 0, 0, 10, 30: the worst 25% holds 75% of the total.
        assert_eq!(worst_share_bp(&[0, 0, 10, 30], 25), 7_500);
        assert_eq!(worst_share_bp(&[0, 0, 10, 30], 100), 10_000);
        assert_eq!(worst_share_bp(&[0, 0], 50), 0);
    }
}
