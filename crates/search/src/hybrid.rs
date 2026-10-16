//! The hybrid yardstick check (docs/M3-HYBRID-SPEC.md §3, "proven 0.01"): on a decision set of
//! the trick before the cut, every candidate is valued exactly and by the hybrid H_k (exact
//! search to the start of trick `k`, the oracle network O_k there) on the *same* worlds, and the
//! three measures of §3 are reported:
//!
//! 1. **Decision loss** (the standard): per decision, the hybrid's choice is the candidate with
//!    the best H_k mean over the decision's worlds (ties to the first candidate, as PIMC); its
//!    loss is the exact mean of the exact-best candidate minus the exact mean of the hybrid's
//!    choice, in the decider's team's money chance. Reported as the mean over decisions and its
//!    95% upper bound: the upper end of the 95% bootstrap interval, the 97.5th percentile of
//!    1,000 bootstrap means (decisions resampled with replacement, `Rng::derive(boot_seed, …)`,
//!    seed 1; the convention of `crate::curves`, and the stricter reading), overall and per
//!    position band (1–2, 3–4, 5–8; each band resampled on its own). Pass: upper bound ≤ 0.01%
//!    everywhere.
//! 2. **Value error** (diagnostic): per candidate and world, H_k minus exact in the decider's
//!    team's entry: mean (bias), SD, 99.9th percentile of the absolute error, largest; overall,
//!    per band and per position. Bias must be within ±0.01%.
//! 3. **Agreement** (diagnostic): the share of decisions whose hybrid choice is the exact choice
//!    or ties it exactly (zero loss).
//!
//! **Worlds** are regenerated exactly as the yardstick draws them (`crate::measure`: decision seed
//! from (seed, label, id), world `w` from it), never stored twice. **Exact values** are computed
//! by the current solver, or read from an existing worlds file (`--worlds`), in which case a
//! sample of worlds (`verify_pct`, default 1%) is recomputed and must be byte-identical first
//! (spec §4); any mismatch stops the check.
//!
//! Every value is computed in fixed point; the result is byte-identical for any thread count.
//! Timings (milliseconds per world, nanoseconds per network evaluation) are reports only.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use engine::rng::Rng;
use equity::{MoneyChance, ONE};
use serde::Serialize;
use solver::frontier::{CanonPos, Frontier};
use solver::{FrontierCut, MatchCtx, SolveMem};

use crate::measure::{DecisionSpec, WorldRows, measure_seed};
use crate::pimc::{
    Candidate, Ctx, SearchConfig, WorldSolve, aggregate, best_index, candidates, exact_world,
    world_match,
};

/// Seed label of the verification sample.
const VERIFY: u64 = 0x5E21;
/// Seed label of the bootstrap.
const BOOT: u64 = 0xB007;
/// The standard: 0.01% money chance, in units of 2^-32.
pub const STANDARD: f64 = 1e-4;

/// One check run.
#[derive(Clone, Debug)]
pub struct CheckJob {
    /// The frontier trick.
    pub cut: u8,
    /// Master seed of the worlds (as the yardstick: 1).
    pub seed: u64,
    /// World label (1 = the yardstick's).
    pub label: u64,
    /// Worlds per decision.
    pub to: u64,
    /// Worker threads (never changes a value).
    pub threads: usize,
    /// Transposition-table megabytes per solve.
    pub tt_mb: usize,
    /// Percentage of stored worlds recomputed and compared (with stored exact values).
    pub verify_pct: f64,
    /// Recompute every exact value even when stored ones are given.
    pub recompute: bool,
}

/// A frontier evaluator that times its evaluations (reports only).
struct Timed<'a> {
    inner: &'a dyn Frontier,
    ns: AtomicU64,
    n: AtomicU64,
}

impl Frontier for Timed<'_> {
    fn value(&self, pos: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        let t = Instant::now();
        let v = self.inner.value(pos, ctx);
        self.ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.n.fetch_add(1, Ordering::Relaxed);
        v
    }
}

/// One world of one decision: exact and hybrid values of every candidate.
#[derive(Clone, Debug)]
struct WorldOut {
    exact: Vec<MoneyChance>,
    hybrid: Vec<MoneyChance>,
    exact_us: u64,
    hybrid_us: u64,
    exact_nodes: u64,
    hybrid_nodes: u64,
    frontier: u64,
    /// Stored exact values were compared with recomputed ones in this world: `Some(equal)`.
    verified: Option<bool>,
}

/// The check of one decision.
#[derive(Clone, Debug, Serialize)]
pub struct DecisionCheck {
    /// Decision id.
    pub id: u64,
    /// Position in the trick (1 = leader … 8).
    pub position: u8,
    /// Stratum (1–4).
    pub stratum: u8,
    /// The decider's team.
    pub team: u8,
    /// Candidates.
    pub cands: usize,
    /// Worlds.
    pub worlds: u64,
    /// Exact means' choice and the hybrid means' choice (candidate indices).
    pub exact_choice: usize,
    /// The hybrid's choice.
    pub hybrid_choice: usize,
    /// Loss of the hybrid's choice in exact values, decider's team, units of 2^-32.
    pub loss: u64,
    /// The hybrid's choice is exact's or ties it exactly.
    pub agree: bool,
    /// Per world and candidate: hybrid minus exact, decider's team, units of 2^-32.
    #[serde(skip)]
    pub errors: Vec<i64>,
    /// Worlds whose exact values this run computed (all of them unless stored values were
    /// given; then the verification sample).
    pub exact_worlds: u64,
    /// Exact solve time over those worlds (µs).
    pub exact_us: u64,
    /// Hybrid solve time over the decision's worlds (µs).
    pub hybrid_us: u64,
    /// Solver nodes, exact and hybrid.
    pub exact_nodes: u64,
    /// Solver nodes of the hybrid solves.
    pub hybrid_nodes: u64,
    /// Network evaluations.
    pub frontier: u64,
}

/// The result of a check.
#[derive(Clone, Debug)]
pub struct CheckRun {
    /// Per decision, in the order given.
    pub decisions: Vec<DecisionCheck>,
    /// Stored worlds recomputed for verification, and how many differed.
    pub verified: (u64, u64),
    /// Time and count of network evaluations (ns, n).
    pub net: (u64, u64),
}

/// Run the check on `ds` (decisions before the cut's trick). `stored`: exact worlds records by
/// decision id (from an existing worlds file), used instead of recomputing (after the
/// verification sample) unless `job.recompute`.
pub fn run_check(
    ds: &[DecisionSpec],
    stored: &BTreeMap<u64, WorldRows>,
    job: &CheckJob,
    ctx: &Ctx,
    net: &dyn Frontier,
) -> Result<CheckRun, String> {
    struct Prep<'a> {
        d: &'a DecisionSpec,
        view: engine::view::SeatView,
        cands: Vec<Candidate>,
        dseed: u64,
        rec: Option<&'a WorldRows>,
    }
    let mut preps = Vec::new();
    for d in ds {
        if d.trick >= job.cut || d.trick == 0 {
            return Err(format!(
                "decision {} is at trick {}: the hybrid with a frontier at trick {} is defined for \
                 decisions at tricks 1..{}",
                d.id, d.trick, job.cut, job.cut
            ));
        }
        let (_, view, legal) = d.replay();
        let (_, cands) = candidates(&view, &legal);
        let rec = stored.get(&d.id);
        let seed = rec
            .and_then(|r| r.fp.as_ref())
            .map_or(job.seed, |fp| fp.seed);
        let label = rec.map_or(job.label, |r| r.label);
        if let Some(r) = rec {
            if r.cands != cands {
                return Err(format!("decision {}: stored candidates differ", d.id));
            }
            if r.from != 0 || (r.rows.len() as u64) < job.to {
                return Err(format!(
                    "decision {}: the stored record has worlds {}..{}, not 0..{}",
                    d.id,
                    r.from,
                    r.from + r.rows.len() as u64,
                    job.to
                ));
            }
        }
        preps.push(Prep {
            d,
            view,
            cands,
            dseed: measure_seed(seed, label, d.id),
            rec,
        });
    }
    let worlds: Vec<_> = preps
        .iter()
        .map(|p| ctx.source.prepare(&p.view, ctx.rules))
        .collect();
    let timed = Timed {
        inner: net,
        ns: AtomicU64::new(0),
        n: AtomicU64::new(0),
    };
    let cut = FrontierCut::new(job.cut, &timed);
    let cfg = SearchConfig {
        tt_mb: job.tt_mb,
        ..SearchConfig::default()
    };
    let per = job.to as usize;
    let n_tasks = preps.len() * per;
    let slots: Mutex<Vec<Option<WorldOut>>> = Mutex::new(vec![None; n_tasks]);
    let next = AtomicUsize::new(0);
    let work = || {
        let mut mem = SolveMem::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n_tasks {
                break;
            }
            let (k, w) = (i / per, (i % per) as u64);
            let p = &preps[k];
            let m = world_match(&p.view, &*worlds[k], w, p.dseed, ctx.rules);
            let stored_row = p.rec.filter(|_| !job.recompute).map(|r| {
                r.rows[w as usize]
                    .iter()
                    .map(|&v| MoneyChance::from_raw(v).expect("stored values sum to one"))
                    .collect::<Vec<_>>()
            });
            let verify = stored_row.is_some()
                && Rng::derive(job.seed, &[VERIFY, p.d.id, w]).bounded(1_000_000)
                    < (job.verify_pct * 10_000.0) as u64;
            let (exact, exact_us, exact_nodes, verified) = match stored_row {
                Some(s) if !verify => (s, 0, 0, None),
                _ => {
                    let mut run = WorldSolve::default();
                    let t = Instant::now();
                    let e = exact_world(&m, &p.cands, ctx, &cfg, None, &mut mem, &mut run, None)
                        .expect("nothing stops a solve here");
                    let us = t.elapsed().as_micros() as u64;
                    let ok = stored_row.as_ref().map(|s| *s == e);
                    (e, us, run.stats.nodes, ok)
                }
            };
            let mut run = WorldSolve::default();
            let t = Instant::now();
            let hybrid = exact_world(&m, &p.cands, ctx, &cfg, None, &mut mem, &mut run, Some(cut))
                .expect("nothing stops a solve here");
            let out = WorldOut {
                exact,
                hybrid,
                exact_us,
                hybrid_us: t.elapsed().as_micros() as u64,
                exact_nodes,
                hybrid_nodes: run.stats.nodes,
                frontier: run.stats.frontier,
                verified,
            };
            slots.lock().unwrap_or_else(|e| e.into_inner())[i] = Some(out);
        }
    };
    let threads = job.threads.max(1).min(n_tasks.max(1));
    if threads == 1 {
        work();
    } else {
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(work);
            }
        });
    }
    let outs: Vec<WorldOut> = slots
        .into_inner()
        .unwrap_or_else(|e| e.into_inner())
        .into_iter()
        .map(|o| o.expect("every world valued"))
        .collect();
    let mut verified = (0u64, 0u64);
    let mut decisions = Vec::new();
    for (k, p) in preps.iter().enumerate() {
        let ws = &outs[k * per..(k + 1) * per];
        let team = engine::team_of(p.d.seat);
        let ex: Vec<Vec<MoneyChance>> = ws.iter().map(|o| o.exact.clone()).collect();
        let hy: Vec<Vec<MoneyChance>> = ws.iter().map(|o| o.hybrid.clone()).collect();
        let (em, ec, _) = aggregate(&ex, p.cands.len(), team);
        let (hm, _, _) = aggregate(&hy, p.cands.len(), team);
        let hc = best_index(&hm, team);
        let loss = em[ec].get(team) - em[hc].get(team);
        let mut errors = Vec::with_capacity(per * p.cands.len());
        for o in ws {
            for (e, h) in o.exact.iter().zip(&o.hybrid) {
                errors.push(h.get(team) as i64 - e.get(team) as i64);
            }
            if let Some(ok) = o.verified {
                verified.0 += 1;
                verified.1 += u64::from(!ok);
            }
        }
        decisions.push(DecisionCheck {
            id: p.d.id,
            position: p.d.position + 1,
            stratum: p.d.stratum,
            team,
            cands: p.cands.len(),
            worlds: job.to,
            exact_choice: ec,
            hybrid_choice: hc,
            loss,
            agree: loss == 0,
            errors,
            exact_worlds: ws
                .iter()
                .filter(|o| o.verified.is_some() || p.rec.is_none() || job.recompute)
                .count() as u64,
            exact_us: ws.iter().map(|o| o.exact_us).sum(),
            hybrid_us: ws.iter().map(|o| o.hybrid_us).sum(),
            exact_nodes: ws.iter().map(|o| o.exact_nodes).sum(),
            hybrid_nodes: ws.iter().map(|o| o.hybrid_nodes).sum(),
            frontier: ws.iter().map(|o| o.frontier).sum(),
        });
    }
    Ok(CheckRun {
        decisions,
        verified,
        net: (
            timed.ns.load(Ordering::Relaxed),
            timed.n.load(Ordering::Relaxed),
        ),
    })
}

/// The §3 measures of a group of decisions.
#[derive(Clone, Debug, Serialize)]
pub struct Summary {
    /// Decisions.
    pub decisions: usize,
    /// Mean decision loss (money chance, a fraction).
    pub mean_loss: f64,
    /// Its 95% upper bound (bootstrap 97.5th percentile).
    pub loss_upper95: f64,
    /// Agreement: share of decisions with zero loss.
    pub agreement: f64,
    /// Values compared (candidate × world).
    pub values: usize,
    /// Value error: mean (bias), as a fraction.
    pub bias: f64,
    /// Value error: standard deviation.
    pub sd: f64,
    /// Value error: 99.9th percentile of the absolute error.
    pub p999_abs: f64,
    /// Value error: the largest absolute error.
    pub max_abs: f64,
    /// Decision loss within the standard at the 95% upper bound.
    pub loss_pass: bool,
    /// Bias within ±0.01%.
    pub bias_pass: bool,
}

/// The §3 measures of `ds`; the bootstrap uses `Rng::derive(boot_seed, &[BOOT, tag, b])` for
/// replicate `b`.
pub fn summarise(ds: &[&DecisionCheck], boots: u64, boot_seed: u64, tag: u64) -> Summary {
    let n = ds.len();
    let unit = ONE as f64;
    let losses: Vec<u64> = ds.iter().map(|d| d.loss).collect();
    let mean_loss = losses.iter().map(|&l| l as f64).sum::<f64>() / n.max(1) as f64 / unit;
    let mut reps: Vec<f64> = (0..boots)
        .map(|b| {
            let mut rng = Rng::derive(boot_seed, &[BOOT, tag, b]);
            let s: u128 = (0..n)
                .map(|_| u128::from(losses[rng.bounded(n as u64) as usize]))
                .sum();
            s as f64 / n.max(1) as f64 / unit
        })
        .collect();
    reps.sort_by(f64::total_cmp);
    let upper = if reps.is_empty() {
        mean_loss
    } else {
        reps[((reps.len() as f64 * 0.975).ceil() as usize).clamp(1, reps.len()) - 1]
    };
    let mut errs: Vec<i64> = ds.iter().flat_map(|d| d.errors.iter().copied()).collect();
    let m = errs.len().max(1) as f64;
    let mean = errs.iter().map(|&e| e as f64).sum::<f64>() / m;
    let var = errs.iter().map(|&e| (e as f64 - mean).powi(2)).sum::<f64>() / m;
    let mut abs: Vec<u64> = errs.iter().map(|e| e.unsigned_abs()).collect();
    abs.sort_unstable();
    let q = |p: f64| {
        if abs.is_empty() {
            0
        } else {
            abs[((abs.len() as f64 * p).ceil() as usize).clamp(1, abs.len()) - 1]
        }
    };
    errs.clear();
    Summary {
        decisions: n,
        mean_loss,
        loss_upper95: upper,
        agreement: ds.iter().filter(|d| d.agree).count() as f64 / n.max(1) as f64,
        values: abs.len(),
        bias: mean / unit,
        sd: var.sqrt() / unit,
        p999_abs: q(0.999) as f64 / unit,
        max_abs: abs.last().copied().unwrap_or(0) as f64 / unit,
        loss_pass: upper <= STANDARD,
        bias_pass: (mean / unit).abs() <= STANDARD,
    }
}

/// The position bands of §3: 1–2, 3–4, 5–8.
pub const BANDS: [(&str, u8, u8); 3] = [("pos 1-2", 1, 2), ("pos 3-4", 3, 4), ("pos 5-8", 5, 8)];

/// Overall, per band and per position.
pub fn report(run: &CheckRun, boots: u64, boot_seed: u64) -> Vec<(String, Summary)> {
    let all: Vec<&DecisionCheck> = run.decisions.iter().collect();
    let mut out = vec![("all".to_string(), summarise(&all, boots, boot_seed, 0))];
    for (i, &(name, lo, hi)) in BANDS.iter().enumerate() {
        let g: Vec<&DecisionCheck> = all
            .iter()
            .copied()
            .filter(|d| (lo..=hi).contains(&d.position))
            .collect();
        out.push((
            name.to_string(),
            summarise(&g, boots, boot_seed, 1 + i as u64),
        ));
    }
    for p in 1..=8u8 {
        let g: Vec<&DecisionCheck> = all.iter().copied().filter(|d| d.position == p).collect();
        if !g.is_empty() {
            out.push((
                format!("pos {p}"),
                summarise(&g, boots, boot_seed, 10 + u64::from(p)),
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use equity::MatchState;
    use solver::FnFrontier;

    use crate::measure::{build_decisions, collect_starts};
    use crate::pimc::SharedLeaf;
    use crate::worlds::BeliefSource;

    fn leaf() -> Arc<SharedLeaf> {
        Arc::new(solver::FnLeaf(|s: &MatchState| {
            MoneyChance::from_weights(core::array::from_fn(|t| {
                1 + u128::from(s.scores[t]) + 40 * u128::from(s.games[t])
            }))
        }))
    }

    fn even(_: &CanonPos, _: &MatchCtx) -> MoneyChance {
        MoneyChance::even()
    }

    /// The check is thread-independent, and with stored exact values (made by the yardstick's
    /// own worlds code) it verifies them and gives the same result.
    #[test]
    fn check_is_deterministic_and_reads_stored_values() {
        let starts = collect_starts(300, 3);
        let ds = build_decisions(4, 8, 5, &starts);
        let l = leaf();
        let src = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: engine::game::Rules::HOUSE,
        };
        let f = FnFrontier(even as fn(&CanonPos, &MatchCtx) -> MoneyChance);
        let job = |threads| CheckJob {
            cut: 5,
            seed: 5,
            label: 1,
            to: 6,
            threads,
            tt_mb: 4,
            verify_pct: 100.0,
            recompute: false,
        };
        let a = run_check(&ds, &BTreeMap::new(), &job(1), &ctx, &f).expect("runs");
        let b = run_check(&ds, &BTreeMap::new(), &job(3), &ctx, &f).expect("runs");
        let key = |r: &CheckRun| -> Vec<(u64, usize, usize, u64, Vec<i64>)> {
            r.decisions
                .iter()
                .map(|d| {
                    (
                        d.id,
                        d.exact_choice,
                        d.hybrid_choice,
                        d.loss,
                        d.errors.clone(),
                    )
                })
                .collect()
        };
        assert_eq!(key(&a), key(&b));
        assert!(a.decisions.iter().any(|d| d.errors.iter().any(|&e| e != 0)));
        // Stored exact values from a yardstick-style worlds run.
        let cfg = SearchConfig {
            tt_mb: 4,
            ..SearchConfig::default()
        };
        let stored: BTreeMap<u64, WorldRows> = ds
            .iter()
            .map(|d| {
                let r = crate::measure::world_rows(
                    d,
                    crate::pimc::Method::ExactPimc,
                    1,
                    5,
                    0,
                    6,
                    "t",
                    &ctx,
                    &cfg,
                );
                (d.id, r)
            })
            .collect();
        let c = run_check(&ds, &stored, &job(2), &ctx, &f).expect("runs");
        assert_eq!(key(&a), key(&c));
        assert_eq!(c.verified, (8 * 6, 0));
        let s = report(&a, 50, 1);
        assert_eq!(s[0].1.decisions, 8);
        assert!(s[0].1.mean_loss <= s[0].1.loss_upper95 + 1e-12);
    }
}
