//! The student against the teacher's stored values (docs/STUDENT.md §7, `search student-check`):
//! per decision the student's choice and the teacher's best; the **decision loss** (the teacher's
//! mean value of its best candidate minus that of the student's choice, in the deciding pair's
//! money chance) and **agreement** (the student's choice has the teacher's best value: ties count),
//! overall, by trick, by place in the trick and by match-state stratum, with a bootstrap 95% upper
//! bound of the mean loss over decisions. Sums are whole numbers (units of 2^-32); floats appear
//! only in what is printed.

use engine::rng::Rng;
use equity::{MoneyChance, ONE};
use serde_json::{Value, json};
use solver::Leaf;

use crate::data::{Merged, Shards, Uniform, feature_row};
use crate::net::{StudentNet, choose};

/// One decision's outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Judged {
    /// Trick.
    pub trick: u8,
    /// Place in the trick (0 = leader).
    pub place: u8,
    /// Stratum (1–4).
    pub stratum: u8,
    /// Candidates.
    pub n_cands: u8,
    /// The teacher's best candidate (ties to the first).
    pub best: u8,
    /// The student's choice.
    pub chosen: u8,
    /// The loss, units of 2^-32.
    pub loss: u64,
    /// Σ over candidates of |student value − teacher value| for the deciding pair, units of 2^-32.
    pub abs_err: u64,
}

/// Judge every decision of the shards (merged by deal index; plain means).
pub fn judge(
    net: &StudentNet,
    shards: &mut Shards,
    leaf: &dyn Leaf,
) -> Result<Vec<Judged>, String> {
    let mut out = Vec::with_capacity(shards.groups.len());
    judge_each(net, shards, leaf, |_, _, j| {
        out.push(j);
        Ok(())
    })?;
    Ok(out)
}

/// [`judge`], handing each decision to `visit` in order: the merged decision (every deal's
/// values), the student's values per candidate (real pair order) and its [`Judged`].
pub fn judge_each(
    net: &StudentNet,
    shards: &mut Shards,
    leaf: &dyn Leaf,
    mut visit: impl FnMut(&Merged, &[MoneyChance], Judged) -> Result<(), String>,
) -> Result<(), String> {
    for g in 0..shards.groups.len() {
        let m = shards.merged(g)?;
        let row = feature_row(&m, leaf, &Uniform)?;
        let team = engine::team_of(m.head.pos.seat);
        let mut view = row.view.clone();
        view.team = team;
        let student = net.evaluate(&view, &row.cands);
        let teacher = m.means();
        let chosen = choose(&student, team);
        let best = choose(&teacher, team);
        let own = |v: &MoneyChance| v.get(team);
        let abs_err = student
            .iter()
            .zip(&teacher)
            .map(|(s, t)| own(s).abs_diff(own(t)))
            .sum();
        let j = Judged {
            trick: m.head.trick,
            place: m.head.place,
            stratum: m.head.stratum,
            n_cands: m.head.cands.len() as u8,
            best: best as u8,
            chosen: chosen as u8,
            loss: own(&teacher[best]) - own(&teacher[chosen]),
            abs_err,
        };
        visit(&m, &student, j)?;
    }
    Ok(())
}

/// A player's candidates best first by its values for `team`, ties to the lower index:
/// `rank(v, team)[0] == choose(v, team)`.
pub fn rank(values: &[MoneyChance], team: u8) -> Vec<usize> {
    let mut r: Vec<usize> = (0..values.len()).collect();
    r.sort_by_key(|&i| core::cmp::Reverse(values[i].get(team)));
    r
}

/// Judge the rules bot (`rules-v1`, `bots::decide`) the same way: a baseline for the student's
/// numbers. A robber's joint decision is its burn then the card it plays after it. A choice that is
/// not among the stored candidates (the bot burning the card it has just taken, which the app's
/// candidates leave out) is not judged; the second number counts them.
pub fn judge_rules(shards: &mut Shards) -> Result<(Vec<Judged>, u64), String> {
    use engine::game::{Action, Rules};
    let mut out = Vec::with_capacity(shards.groups.len());
    let mut missing = 0;
    for g in 0..shards.groups.len() {
        let m = shards.merged(g)?;
        let r = m.head.pos.replay()?;
        let team = engine::team_of(m.head.pos.seat);
        let (a, _) = bots::decide(&r.view, &Rules::HOUSE, &r.legal);
        let cand = match a {
            Action::Burn(b) => {
                let mut mm = r.m.clone();
                mm.apply(a).map_err(|e| format!("{e:?}"))?;
                let v = mm.view(m.head.pos.seat);
                let legal = mm.legal_actions();
                (
                    Action::Burn(b),
                    Some(bots::decide(&v, &Rules::HOUSE, &legal).0.card()),
                )
            }
            Action::Play(c) => (Action::Play(c), None),
        };
        let Some(chosen) = m.head.cands.iter().position(|&c| c == cand) else {
            missing += 1;
            continue;
        };
        let teacher = m.means();
        let best = choose(&teacher, team);
        out.push(Judged {
            trick: m.head.trick,
            place: m.head.place,
            stratum: m.head.stratum,
            n_cands: m.head.cands.len() as u8,
            best: best as u8,
            chosen: chosen as u8,
            loss: teacher[best].get(team) - teacher[chosen].get(team),
            abs_err: 0,
        });
    }
    Ok((out, missing))
}

/// The seed tag of [`judge_random`]'s picks.
const RANDOM_TAG: u64 = 0x7A4D;

/// Judge a player that picks a candidate uniformly at random (a floor for every other number):
/// decision `g`'s pick is `Rng::derive(seed, [RANDOM_TAG, g]).bounded(candidates)`, so the
/// baseline is the same on every machine. The second number is the exact expected loss of a
/// random pick over all decisions (the mean over each decision's candidates of the best less that
/// candidate, then over decisions), in money chance (a fraction): the seeded picks' mean estimates
/// it.
pub fn judge_random(shards: &mut Shards, seed: u64) -> Result<(Vec<Judged>, f64), String> {
    let mut out = Vec::with_capacity(shards.groups.len());
    let mut expected = 0.0f64;
    for g in 0..shards.groups.len() {
        let m = shards.merged(g)?;
        let team = engine::team_of(m.head.pos.seat);
        let teacher = m.means();
        let best = choose(&teacher, team);
        let n = teacher.len() as u64;
        let chosen = Rng::derive(seed, &[RANDOM_TAG, g as u64]).bounded(n) as usize;
        let top = teacher[best].get(team);
        let sum: u128 = teacher.iter().map(|t| u128::from(top - t.get(team))).sum();
        expected += sum as f64 / n as f64 / ONE as f64;
        out.push(Judged {
            trick: m.head.trick,
            place: m.head.place,
            stratum: m.head.stratum,
            n_cands: m.head.cands.len() as u8,
            best: best as u8,
            chosen: chosen as u8,
            loss: top - teacher[chosen].get(team),
            abs_err: 0,
        });
    }
    let n = out.len().max(1) as f64;
    Ok((out, expected / n))
}

/// A group's summary.
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    /// Decisions.
    pub n: u64,
    /// Mean loss, in money chance (a fraction).
    pub mean_loss: f64,
    /// Bootstrap 95% upper bound of the mean loss.
    pub upper: f64,
    /// Share of decisions in agreement.
    pub agreement: f64,
    /// Mean |error| of the student's own-pair value per candidate.
    pub mae: f64,
}

/// Summarise decisions: the mean loss, its bootstrap 95% upper bound (`boots` replicates drawn
/// from `Rng::derive(boot_seed, [tag])`), agreement and the value error.
pub fn summarise(ds: &[&Judged], boots: u64, boot_seed: u64, tag: u64) -> Summary {
    let n = ds.len() as u64;
    if n == 0 {
        return Summary {
            n: 0,
            mean_loss: 0.0,
            upper: 0.0,
            agreement: 0.0,
            mae: 0.0,
        };
    }
    let total: u128 = ds.iter().map(|d| u128::from(d.loss)).sum();
    let agree = ds.iter().filter(|d| d.loss == 0).count() as f64;
    let cands: u64 = ds.iter().map(|d| u64::from(d.n_cands)).sum();
    let err: u128 = ds.iter().map(|d| u128::from(d.abs_err)).sum();
    let mut rng = Rng::derive(boot_seed, &[0x5C4E, tag]);
    let mut reps: Vec<u128> = (0..boots)
        .map(|_| {
            (0..n)
                .map(|_| u128::from(ds[rng.bounded(n) as usize].loss))
                .sum()
        })
        .collect();
    reps.sort_unstable();
    let upper = if boots == 0 {
        total
    } else {
        reps[((boots * 95).div_ceil(100) as usize).min(reps.len()) - 1]
    };
    let f = |x: u128| x as f64 / n as f64 / ONE as f64;
    Summary {
        n,
        mean_loss: f(total),
        upper: f(upper),
        agreement: agree / n as f64,
        mae: err as f64 / cands.max(1) as f64 / ONE as f64,
    }
}

fn row(name: &str, s: &Summary) -> Value {
    json!({ "group": name, "n": s.n, "mean_loss_pct": 100.0 * s.mean_loss,
            "upper95_pct": 100.0 * s.upper, "agreement": s.agreement,
            "value_mae_pct": 100.0 * s.mae })
}

/// The report: overall, by trick, by place (1–8), by stratum (S1–S4), as JSON rows.
pub fn report(ds: &[Judged], boots: u64, boot_seed: u64) -> Vec<Value> {
    let all: Vec<&Judged> = ds.iter().collect();
    let mut out = vec![row("all", &summarise(&all, boots, boot_seed, 0))];
    let mut add = |name: String, tag: u64, f: &dyn Fn(&Judged) -> bool| {
        let sel: Vec<&Judged> = ds.iter().filter(|d| f(d)).collect();
        if !sel.is_empty() {
            out.push(row(&name, &summarise(&sel, boots, boot_seed, tag)));
        }
    };
    for t in 1..=5u8 {
        add(format!("trick {t}"), 10 + u64::from(t), &|d| d.trick == t);
    }
    for p in 0..8u8 {
        add(format!("place {}", p + 1), 20 + u64::from(p), &|d| {
            d.place == p
        });
    }
    for s in 1..=4u8 {
        add(format!("stratum S{s}"), 30 + u64::from(s), &|d| {
            d.stratum == s
        });
    }
    out
}

/// The report as text lines.
pub fn print(rows: &[Value]) -> String {
    let mut s = format!(
        "{:<12} {:>7} {:>12} {:>12} {:>10} {:>12}\n",
        "group", "n", "loss %", "upper95 %", "agree", "value MAE %"
    );
    for r in rows {
        s.push_str(&format!(
            "{:<12} {:>7} {:>12.5} {:>12.5} {:>10.4} {:>12.4}\n",
            r["group"].as_str().unwrap_or(""),
            r["n"],
            r["mean_loss_pct"].as_f64().unwrap_or(0.0),
            r["upper95_pct"].as_f64().unwrap_or(0.0),
            r["agreement"].as_f64().unwrap_or(0.0),
            r["value_mae_pct"].as_f64().unwrap_or(0.0),
        ));
    }
    s
}

/// One decision's whole-number scores for an extra measure (`student-check --split-half`,
/// `--top-k`, docs/STUDENT.md §14): one value per column, signed, each in units of 2^-32 times
/// its column's divisor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scored {
    /// Trick.
    pub trick: u8,
    /// Place in the trick (0 = leader).
    pub place: u8,
    /// Stratum (1–4).
    pub stratum: u8,
    /// One value per column.
    pub v: Vec<i128>,
}

/// The groups of [`report`] (all, by trick, by place, by stratum) with their bootstrap tags.
fn groups_of<T>(ds: &[T], key: impl Fn(&T) -> (u8, u8, u8)) -> Vec<(String, u64, Vec<&T>)> {
    let mut out = vec![("all".to_string(), 0, ds.iter().collect())];
    let mut add = |name: String, tag: u64, f: &dyn Fn(&T) -> bool| {
        let sel: Vec<&T> = ds.iter().filter(|d| f(d)).collect();
        if !sel.is_empty() {
            out.push((name, tag, sel));
        }
    };
    for t in 1..=5u8 {
        add(format!("trick {t}"), 10 + u64::from(t), &|d| key(d).0 == t);
    }
    for p in 0..8u8 {
        add(format!("place {}", p + 1), 20 + u64::from(p), &|d| {
            key(d).1 == p
        });
    }
    for s in 1..=4u8 {
        add(format!("stratum S{s}"), 30 + u64::from(s), &|d| {
            key(d).2 == s
        });
    }
    out
}

/// A measure's report: per group (as [`report`]), each column's mean in % of money chance (`cols`
/// gives each column's name and divisor) and the bootstrap bounds of the **last** column's mean:
/// the 5% and 95% points of the same replicates [`summarise`] draws (the same seed, tags and
/// picks, so the decisions resampled are the same in every measure and the upper bound of a plain
/// loss equals `upper95_pct` of [`report`]). Whole-number sums; floats only in the rows.
pub fn report_measure(
    ds: &[Scored],
    cols: &[(&str, u128)],
    boots: u64,
    boot_seed: u64,
) -> Vec<Value> {
    let last = cols.len() - 1;
    groups_of(ds, |d| (d.trick, d.place, d.stratum))
        .into_iter()
        .map(|(name, tag, sel)| {
            let n = sel.len() as u64;
            // In the order `summarise` divides, so a plain loss prints exactly as `report`'s.
            let pct =
                |sum: i128, den: u128| 100.0 * (sum as f64 / n as f64 / den as f64 / ONE as f64);
            let mut rng = Rng::derive(boot_seed, &[0x5C4E, tag]);
            let mut reps: Vec<i128> = (0..boots)
                .map(|_| (0..n).map(|_| sel[rng.bounded(n) as usize].v[last]).sum())
                .collect();
            reps.sort_unstable();
            let total = |c: usize| sel.iter().map(|d| d.v[c]).sum::<i128>();
            let (lo, up) = if boots == 0 {
                (total(last), total(last))
            } else {
                (
                    reps[(boots * 5 / 100) as usize],
                    reps[((boots * 95).div_ceil(100) as usize).min(reps.len()) - 1],
                )
            };
            let mut r = json!({ "group": name, "n": n });
            for (c, (col, den)) in cols.iter().enumerate() {
                r[format!("{col}_pct")] = json!(pct(total(c), *den));
            }
            r["lower95_pct"] = json!(pct(lo, cols[last].1));
            r["upper95_pct"] = json!(pct(up, cols[last].1));
            r
        })
        .collect()
}

/// [`report_measure`]'s rows as text: the columns' means, then the last column's bounds.
pub fn print_measure(rows: &[Value], cols: &[&str]) -> String {
    let mut s = format!("{:<12} {:>7}", "group", "n");
    for c in cols {
        s.push_str(&format!(" {:>10}", format!("{c} %")));
    }
    s.push_str(&format!(" {:>10} {:>10}\n", "lower95 %", "upper95 %"));
    for r in rows {
        s.push_str(&format!(
            "{:<12} {:>7}",
            r["group"].as_str().unwrap_or(""),
            r["n"]
        ));
        for c in cols {
            s.push_str(&format!(
                " {:>10.5}",
                r[format!("{c}_pct")].as_f64().unwrap_or(0.0)
            ));
        }
        s.push_str(&format!(
            " {:>10.5} {:>10.5}\n",
            r["lower95_pct"].as_f64().unwrap_or(0.0),
            r["upper95_pct"].as_f64().unwrap_or(0.0)
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(loss: u64, place: u8) -> Judged {
        Judged {
            trick: 2,
            place,
            stratum: 1,
            n_cands: 3,
            best: 0,
            chosen: u8::from(loss > 0),
            loss,
            abs_err: 30,
        }
    }

    #[test]
    fn summaries_are_whole_number_means_with_an_upper_bound() {
        let ds = [d(0, 0), d(0, 1), d(ONE / 100, 0), d(0, 2)];
        let all: Vec<&Judged> = ds.iter().collect();
        let s = summarise(&all, 1000, 1, 0);
        assert_eq!(s.n, 4);
        assert!((s.mean_loss - 0.0025).abs() < 1e-9);
        assert!(s.upper >= s.mean_loss);
        assert!((s.agreement - 0.75).abs() < 1e-12);
        let r = report(&ds, 100, 1);
        assert_eq!(r[0]["group"], "all");
        assert!(r.iter().any(|x| x["group"] == "place 1" && x["n"] == 2));
        assert!(!print(&r).is_empty());
        // The same input, the same report.
        assert_eq!(report(&ds, 100, 1), r);
    }

    #[test]
    fn rank_is_best_first_ties_to_the_lower_index() {
        let v = |x: u64| MoneyChance([x, ONE - x, 0, 0]);
        let vals = [v(10), v(30), v(30), v(20)];
        assert_eq!(rank(&vals, 0), vec![1, 2, 3, 0]);
        assert_eq!(rank(&vals, 0)[0], choose(&vals, 0));
        assert_eq!(rank(&vals, 1), vec![0, 3, 1, 2]);
    }

    #[test]
    fn a_measure_report_reproduces_the_plain_one() {
        let ds = [d(0, 0), d(0, 1), d(ONE / 100, 0), d(ONE / 50, 2), d(3, 5)];
        let plain = report(&ds, 200, 7);
        let sc: Vec<Scored> = ds
            .iter()
            .map(|j| Scored {
                trick: j.trick,
                place: j.place,
                stratum: j.stratum,
                v: vec![i128::from(j.loss)],
            })
            .collect();
        let m = report_measure(&sc, &[("loss", 1)], 200, 7);
        assert_eq!(m.len(), plain.len());
        for (a, b) in m.iter().zip(&plain) {
            assert_eq!(a["group"], b["group"]);
            assert_eq!(a["loss_pct"], b["mean_loss_pct"]);
            assert_eq!(a["upper95_pct"], b["upper95_pct"]);
            assert!(a["lower95_pct"].as_f64() <= a["loss_pct"].as_f64());
        }
        // Signed columns with divisors: [x, y, x + y] over 2 is the mean of x and y.
        let sc: Vec<Scored> = [(-4i128, 2i128), (6, 0)]
            .iter()
            .map(|&(x, y)| Scored {
                trick: 2,
                place: 0,
                stratum: 1,
                v: vec![x, y, x + y],
            })
            .collect();
        let r = report_measure(&sc, &[("a", 1), ("b", 1), ("mean", 2)], 50, 1);
        let pct = |x: f64| 100.0 * x / ONE as f64;
        assert_eq!(r[0]["a_pct"].as_f64(), Some(pct(1.0)));
        assert_eq!(r[0]["b_pct"].as_f64(), Some(pct(1.0)));
        assert_eq!(r[0]["mean_pct"].as_f64(), Some(pct(1.0)));
        assert!(r[0]["lower95_pct"].as_f64().unwrap() < 0.0);
        assert!(!print_measure(&r, &["a", "b", "mean"]).is_empty());
    }
}
