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
//! 2. **Value error** (diagnostic): per candidate, world and team, H_k minus exact: for each
//!    team relative to the decider (its own, the next, the opposite, the previous), the mean
//!    (bias), and over all four teams together the SD, the 99.9th percentile of the absolute
//!    error and the largest (other seats choose by their own team's entry inside the search, so
//!    every team's error matters); overall, per band and per position. Every team's bias must be
//!    within ±0.01%.
//! 3. **Agreement** (diagnostic): the share of decisions whose hybrid choice is the exact choice
//!    or ties it exactly (zero loss).
//!
//! **Worlds** are regenerated exactly as the yardstick draws them (`crate::measure`: decision seed
//! from (seed, label, id), world `w` from it), never stored twice. **Exact values** are computed
//! by the current solver, or read from an existing worlds file (`--worlds`), in which case a
//! sample of worlds (`verify_pct`, default 1%) is recomputed and must be byte-identical first
//! (spec §4); any mismatch stops the check.
//!
//! **Slices** ([`check_worlds_with`] with a predicate file, [`slices`]; docs/m3-hybrid/
//! audit-loop.md §3 item 2): the hybrid's own search in each world records the distinct frontier
//! positions it reaches and their groups (`crate::audit::GroupRecorder`, which never changes a
//! value); a decision is in a group when at least a share (default a half) of the positions its
//! worlds' searches reach are in it, and the §3 measures are reported per group and for the
//! decisions in no group.
//!
//! **Chosen worlds** ([`check_worlds`]): the same measures on any worlds of each decision, with
//! stored values where given, e.g. the exact trick-1 worlds earlier runs saved (`crate::saved`,
//! spec §3 "Trick-1 checks"), each set with the burn rule that valued it
//! ([`CheckJob::rules_robber_burn`]).
//!
//! Every value is computed in fixed point; the result is byte-identical for any thread count.
//! Timings (milliseconds per world, nanoseconds per network evaluation) are reports only.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use engine::rng::Rng;
use equity::{MoneyChance, ONE};
use nn::predicate::Predicates;
use serde::Serialize;
use solver::frontier::{CanonPos, Frontier};
use solver::{FrontierCut, MatchCtx, SolveMem};

use crate::audit::{GroupRecorder, Reach};
use crate::measure::{DecisionSpec, WorldRows, measure_seed};
use crate::pimc::{
    Candidate, Ctx, SearchConfig, WorldSolve, aggregate, best_index, candidates, exact_world,
    world_match,
};

/// Seed label of the verification sample.
const VERIFY: u64 = 0x5E21;
/// Seed label of the bootstrap.
const BOOT: u64 = 0xB007;
/// The standard: 0.01% money chance (a fraction), the default of `hybrid-check --standard`.
pub const STANDARD: f64 = 1e-4;

/// The standard of `--standard PCT`, a percentage of money chance ("0.01", "0.005"), as a
/// fraction: the decimal text shifted two places and parsed once (correctly rounded), so
/// "0.01" is exactly [`STANDARD`].
pub fn parse_standard(pct: &str) -> Result<f64, String> {
    let t = pct.trim();
    let ok = !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit() || b == b'.');
    let x: f64 = if ok {
        format!("{t}e-2").parse().unwrap_or(f64::NAN)
    } else {
        f64::NAN
    };
    if x.is_finite() && x > 0.0 && x <= 0.01 {
        Ok(x)
    } else {
        Err(format!(
            "--standard {pct:?}: a percentage of money chance above 0 and at most 1, e.g. 0.01"
        ))
    }
}

/// Judges a summary by `standard` (a fraction): the decision-loss bound and every team's bias.
pub fn judge(s: &mut Summary, standard: f64) {
    s.loss_pass = s.loss_upper95 <= standard;
    s.bias_pass = s.bias.iter().all(|b| b.abs() <= standard);
}

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
    /// Transposition-table megabytes per hybrid solve.
    pub tt_mb: usize,
    /// Percentage of stored worlds recomputed and compared (with stored exact values).
    pub verify_pct: f64,
    /// Recompute every exact value even when stored ones are given.
    pub recompute: bool,
    /// Another seat's rob burn later in trick 1 is the `rules-v1` burn, in the exact solves and
    /// the hybrid's alike (`SearchConfig::rules_robber_burn`): the rule of worlds valued before
    /// values version 2 (`crate::saved`).
    pub rules_robber_burn: bool,
    /// Threads inside each exact solve this run makes (`solve_fast_par`, the same answer as one
    /// thread); with more than one, the exact solves run first, `threads / exact_solve_threads`
    /// at a time. Trick-1 worlds need this and big tables to be recomputed in hours, not days.
    pub exact_solve_threads: usize,
    /// Transposition-table megabytes per exact solve.
    pub exact_tt_mb: usize,
}

impl CheckJob {
    /// Whether world `w` of decision `id`, with stored exact values, is in the verification
    /// sample (`verify_pct` percent, drawn by `(seed, id, w)`).
    pub fn verify_sampled(&self, id: u64, w: u64) -> bool {
        Rng::derive(self.seed, &[VERIFY, id, w]).bounded(1_000_000)
            < (self.verify_pct * 10_000.0) as u64
    }

    /// The settings of the exact solves (`exact`) or of the hybrid's, with the job's burn rule.
    pub fn config(&self, exact: bool) -> SearchConfig {
        SearchConfig {
            tt_mb: if exact { self.exact_tt_mb } else { self.tt_mb },
            solve_threads: if exact {
                self.exact_solve_threads.max(1)
            } else {
                1
            },
            rules_robber_burn: self.rules_robber_burn,
            ..SearchConfig::default()
        }
    }
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

/// One world to check: its number, the decision seed it is drawn from
/// ([`measure_seed`]`(seed, label, id)`) and its stored exact values, if any.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldTask {
    /// World number.
    pub w: u64,
    /// The decision seed the world is drawn from.
    pub dseed: u64,
    /// Stored exact values (candidate order).
    pub stored: Option<Vec<MoneyChance>>,
    /// How long the world is expected to take, in any unit common to the check's worlds (for
    /// example the solver nodes its saved exact solve took; 0 = unknown): the longest go first.
    /// It never changes a value.
    pub cost: u64,
}

/// The worlds of one decision to check (any world numbers, in the order given).
#[derive(Clone, Debug)]
pub struct DecisionWorlds<'a> {
    /// The decision.
    pub d: &'a DecisionSpec,
    /// Its worlds.
    pub worlds: Vec<WorldTask>,
}

/// One world of one decision: exact and hybrid values of every candidate.
#[derive(Clone, Debug)]
pub struct WorldCheck {
    /// World number.
    pub w: u64,
    /// Exact values (stored or computed here), candidate order.
    pub exact: Vec<MoneyChance>,
    /// H_k's values.
    pub hybrid: Vec<MoneyChance>,
    /// The exact values were computed by this run (no stored values, `recompute`, or the
    /// verification sample).
    pub exact_solved: bool,
    /// Stored exact values were compared with recomputed ones in this world: `Some(equal)`.
    pub verified: Option<bool>,
    /// Exact solve time (µs; 0 unless solved here).
    pub exact_us: u64,
    /// Exact solver nodes (0 unless solved here).
    pub exact_nodes: u64,
    /// Hybrid solve time (µs).
    pub hybrid_us: u64,
    /// Hybrid solver nodes.
    pub hybrid_nodes: u64,
    /// The hybrid's solver nodes by trick (1–5).
    pub hybrid_nodes_by_trick: [u64; 5],
    /// The hybrid's network evaluations.
    pub frontier: u64,
    /// The largest transposition table the hybrid's solves used (bytes).
    pub hybrid_tt_bytes: u64,
    /// The seat other than the decider that robs later in trick 1 in this world (the burn the
    /// burn rule decides), if any.
    pub robber: Option<u8>,
    /// With slices: the distinct frontier positions the hybrid's search reached, by group.
    pub reach: Option<Reach>,
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
    /// Per world, candidate and team relative to the decider (`(team − decider) mod 4`, the
    /// innermost index): hybrid minus exact, units of 2^-32.
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
    /// With slices: the distinct frontier positions the hybrid's searches reached (per world,
    /// summed), by group.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reach: Option<Reach>,
    /// Every world's values and counters, in the order given.
    #[serde(skip)]
    pub world_checks: Vec<WorldCheck>,
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
    /// Recomputed worlds whose stored exact values differ (decision, world). When there is any,
    /// the hybrid was not run and `decisions` is empty: the stored values are not this solver's.
    pub mismatched: Vec<(u64, u64)>,
}

/// Live counters of a running check, for progress reports (they never change a value).
#[derive(Debug, Default)]
pub struct CheckProgress {
    /// Exact solves this run makes, and those finished.
    pub exact_total: AtomicU64,
    /// See `exact_total`.
    pub exact_done: AtomicU64,
    /// Hybrid worlds, and those finished.
    pub hybrid_total: AtomicU64,
    /// See `hybrid_total`.
    pub hybrid_done: AtomicU64,
    /// Each hybrid world as it finishes, in finishing order: (decision, world, µs, solver
    /// nodes, network evaluations). Progress reports drain it; a stopped run keeps what finished.
    pub finished: Mutex<Vec<[u64; 5]>>,
}

/// Run the check on `ds` (decisions before the cut's trick), worlds `0..job.to` of each.
/// `stored`: exact worlds records by decision id (from an existing worlds file), used instead of
/// recomputing (after the verification sample) unless `job.recompute`.
pub fn run_check(
    ds: &[DecisionSpec],
    stored: &BTreeMap<u64, WorldRows>,
    job: &CheckJob,
    ctx: &Ctx,
    net: &dyn Frontier,
) -> Result<CheckRun, String> {
    run_check_slices(ds, stored, job, ctx, net, None)
}

/// [`run_check`], recording each world's reach in the groups of `slices` when given (module
/// notes, [`slices`]).
pub fn run_check_slices(
    ds: &[DecisionSpec],
    stored: &BTreeMap<u64, WorldRows>,
    job: &CheckJob,
    ctx: &Ctx,
    net: &dyn Frontier,
    slices: Option<&Predicates>,
) -> Result<CheckRun, String> {
    let mut plan = Vec::new();
    for d in ds {
        let rec = stored.get(&d.id);
        let seed = rec
            .and_then(|r| r.fp.as_ref())
            .map_or(job.seed, |fp| fp.seed);
        let label = rec.map_or(job.label, |r| r.label);
        if let Some(r) = rec {
            if d.trick >= 1 && d.trick < job.cut {
                let (_, view, legal) = d.replay();
                if r.cands != candidates(&view, &legal).1 {
                    return Err(format!("decision {}: stored candidates differ", d.id));
                }
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
        let dseed = measure_seed(seed, label, d.id);
        let worlds = (0..job.to)
            .map(|w| {
                Ok(WorldTask {
                    w,
                    dseed,
                    cost: 0,
                    stored: rec
                        .map(|r| {
                            r.rows[w as usize]
                                .iter()
                                .map(|&v| {
                                    MoneyChance::from_raw(v).ok_or_else(|| {
                                        format!(
                                            "decision {} world {w}: stored values do not sum \
                                             to one",
                                            d.id
                                        )
                                    })
                                })
                                .collect::<Result<Vec<_>, String>>()
                        })
                        .transpose()?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        plan.push(DecisionWorlds { d, worlds });
    }
    check_worlds_with(&plan, job, ctx, net, None, slices)
}

/// Runs `f(i, mem)` for every `i` of `order` on `threads` threads (a shared work queue in that
/// order, each thread with its own solver memory); the results by index (`None` for an index not
/// in `order`), `n` of them.
fn pool<T: Send + Clone>(
    threads: usize,
    order: &[usize],
    f: &(dyn Fn(usize, &mut SolveMem) -> T + Sync),
    n: usize,
) -> Vec<Option<T>> {
    let slots: Mutex<Vec<Option<T>>> = Mutex::new(vec![None; n]);
    let next = AtomicUsize::new(0);
    let work = || {
        let mut mem = SolveMem::new();
        loop {
            let q = next.fetch_add(1, Ordering::Relaxed);
            let Some(&i) = order.get(q) else { break };
            let out = f(i, &mut mem);
            slots.lock().unwrap_or_else(|e| e.into_inner())[i] = Some(out);
        }
    };
    let threads = threads.max(1).min(order.len().max(1));
    if threads == 1 {
        work();
    } else {
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(work);
            }
        });
    }
    slots.into_inner().unwrap_or_else(|e| e.into_inner())
}

/// Run the check on chosen worlds of each decision (any world numbers; stored exact values where
/// given). The same measures as [`run_check`], which is this with worlds `0..to`. Exact values
/// are computed for worlds without stored ones, for every world with `job.recompute`, and for
/// the verification sample (`job.verify_pct` of the stored worlds, drawn by `(seed, id, w)`);
/// those solves run first, then the hybrid's, one world per thread, the longest expected first
/// (the worlds' cost hints, then the earliest positions in the trick). If any recomputed world's
/// stored values differ, the hybrid is not run: the result lists those worlds (`mismatched`) and
/// has no decisions. `progress` receives live counts.
pub fn check_worlds(
    plan: &[DecisionWorlds],
    job: &CheckJob,
    ctx: &Ctx,
    net: &dyn Frontier,
    progress: Option<&CheckProgress>,
) -> Result<CheckRun, String> {
    check_worlds_with(plan, job, ctx, net, progress, None)
}

/// [`check_worlds`], recording each hybrid search's reach in the groups of `slices` when given
/// (`crate::audit::GroupRecorder` around the network: the same values).
pub fn check_worlds_with(
    plan: &[DecisionWorlds],
    job: &CheckJob,
    ctx: &Ctx,
    net: &dyn Frontier,
    progress: Option<&CheckProgress>,
    slices: Option<&Predicates>,
) -> Result<CheckRun, String> {
    struct Prep {
        view: engine::view::SeatView,
        cands: Vec<Candidate>,
    }
    let mut preps = Vec::new();
    for p in plan {
        let d = p.d;
        if d.trick >= job.cut || d.trick == 0 {
            return Err(format!(
                "decision {} is at trick {}: the hybrid with a frontier at trick {} is defined for \
                 decisions at tricks 1..{}",
                d.id, d.trick, job.cut, job.cut
            ));
        }
        let (_, view, legal) = d.replay();
        let (_, cands) = candidates(&view, &legal);
        for t in &p.worlds {
            if t.stored.as_ref().is_some_and(|s| s.len() != cands.len()) {
                return Err(format!(
                    "decision {} world {}: {} stored values for {} candidates",
                    d.id,
                    t.w,
                    t.stored.as_ref().map_or(0, Vec::len),
                    cands.len()
                ));
            }
        }
        preps.push(Prep { view, cands });
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
    // Every (decision, world), and the queue order: the longest expected first (by the worlds'
    // cost hints, then the earliest trick and position).
    let tasks: Vec<(usize, usize)> = plan
        .iter()
        .enumerate()
        .flat_map(|(k, p)| (0..p.worlds.len()).map(move |j| (k, j)))
        .collect();
    let mut order: Vec<usize> = (0..tasks.len()).collect();
    order.sort_by_key(|&i| {
        let (k, j) = tasks[i];
        let d = plan[k].d;
        (
            std::cmp::Reverse(plan[k].worlds[j].cost),
            d.trick,
            d.position,
            i,
        )
    });
    let verify_of = |i: usize| -> bool {
        let (k, j) = tasks[i];
        let t = &plan[k].worlds[j];
        t.stored.is_some() && !job.recompute && job.verify_sampled(plan[k].d.id, t.w)
    };
    let exact_needed = |i: usize| -> bool {
        let (k, j) = tasks[i];
        plan[k].worlds[j].stored.is_none() || job.recompute || verify_of(i)
    };
    let world_of = |i: usize| {
        let (k, j) = tasks[i];
        world_match(
            &preps[k].view,
            &*worlds[k],
            plan[k].worlds[j].w,
            plan[k].worlds[j].dseed,
            ctx.rules,
        )
    };
    // The exact solves.
    let ex_order: Vec<usize> = order.iter().copied().filter(|&i| exact_needed(i)).collect();
    if let Some(p) = progress {
        p.exact_total
            .store(ex_order.len() as u64, Ordering::Relaxed);
        p.hybrid_total.store(tasks.len() as u64, Ordering::Relaxed);
    }
    let cfg_ex = job.config(true);
    let solve_exact = |i: usize, mem: &mut SolveMem| -> (Vec<MoneyChance>, u64, u64) {
        let m = world_of(i);
        let mut run = WorldSolve::default();
        let t = Instant::now();
        let e = exact_world(
            &m,
            &preps[tasks[i].0].cands,
            ctx,
            &cfg_ex,
            None,
            mem,
            &mut run,
            None,
        )
        .expect("nothing stops a solve here");
        let us = t.elapsed().as_micros() as u64;
        if let Some(p) = progress {
            p.exact_done.fetch_add(1, Ordering::Relaxed);
        }
        (e, us, run.stats.nodes)
    };
    let ex_threads = (job.threads / job.exact_solve_threads.max(1)).max(1);
    let exact = pool(ex_threads, &ex_order, &solve_exact, tasks.len());
    // Stored values must be this solver's before anything is compared with them.
    let mut verified = (0u64, 0u64);
    let mut mismatched = Vec::new();
    for &i in &ex_order {
        let (k, j) = tasks[i];
        let t = &plan[k].worlds[j];
        if let (Some(s), Some((e, _, _)), false) = (&t.stored, &exact[i], job.recompute) {
            verified.0 += 1;
            if s != e {
                verified.1 += 1;
                mismatched.push((plan[k].d.id, t.w));
            }
        }
    }
    if !mismatched.is_empty() {
        return Ok(CheckRun {
            decisions: Vec::new(),
            verified,
            net: (0, 0),
            mismatched,
        });
    }
    // The hybrid's solves.
    let cfg_h = job.config(false);
    type HybridOut = (Vec<MoneyChance>, u64, WorldSolve, Option<Reach>);
    let solve_hybrid = |i: usize, mem: &mut SolveMem| -> HybridOut {
        let m = world_of(i);
        let mut run = WorldSolve::default();
        let t = Instant::now();
        let rec = slices.map(|p| GroupRecorder::new(&timed, p, ctx.leaf));
        let h = exact_world(
            &m,
            &preps[tasks[i].0].cands,
            ctx,
            &cfg_h,
            None,
            mem,
            &mut run,
            Some(match &rec {
                Some(r) => FrontierCut::new(job.cut, r),
                None => cut,
            }),
        )
        .expect("nothing stops a solve here");
        let reach = rec.map(GroupRecorder::into_reach);
        let us = t.elapsed().as_micros() as u64;
        if let Some(p) = progress {
            let (k, j) = tasks[i];
            p.finished.lock().unwrap_or_else(|e| e.into_inner()).push([
                plan[k].d.id,
                plan[k].worlds[j].w,
                us,
                run.stats.nodes,
                run.stats.frontier,
            ]);
            p.hybrid_done.fetch_add(1, Ordering::Relaxed);
        }
        (h, us, run, reach)
    };
    let hybrid = pool(job.threads, &order, &solve_hybrid, tasks.len());
    let mut outs: Vec<WorldCheck> = Vec::with_capacity(tasks.len());
    for (i, (hy, ex)) in hybrid.into_iter().zip(exact).enumerate() {
        let (h, hus, run, reach) = hy.expect("every world valued by the hybrid");
        let (k, j) = tasks[i];
        let t = &plan[k].worlds[j];
        let (exact, exact_us, exact_nodes, verified) = match (ex, &t.stored) {
            (Some((e, us, nodes)), s) => {
                let ok = s.as_ref().filter(|_| !job.recompute).map(|s| *s == e);
                (e, us, nodes, ok)
            }
            (None, Some(s)) => (s.clone(), 0, 0, None),
            (None, None) => unreachable!("worlds without stored values are solved"),
        };
        outs.push(WorldCheck {
            w: t.w,
            exact,
            hybrid: h,
            exact_solved: exact_needed(i),
            verified,
            exact_us,
            exact_nodes,
            hybrid_us: hus,
            hybrid_nodes: run.stats.nodes,
            hybrid_nodes_by_trick: run.stats.nodes_by_trick,
            frontier: run.stats.frontier,
            hybrid_tt_bytes: run.stats.tt_bytes,
            robber: run.robber,
            reach,
        });
    }
    let mut decisions = Vec::new();
    let mut at = 0;
    for (k, p) in plan.iter().enumerate() {
        let ws = &outs[at..at + p.worlds.len()];
        at += p.worlds.len();
        let cands = &preps[k].cands;
        let team = engine::team_of(p.d.seat);
        let ex: Vec<Vec<MoneyChance>> = ws.iter().map(|o| o.exact.clone()).collect();
        let hy: Vec<Vec<MoneyChance>> = ws.iter().map(|o| o.hybrid.clone()).collect();
        let (em, ec, _) = aggregate(&ex, cands.len(), team);
        let (hm, _, _) = aggregate(&hy, cands.len(), team);
        let hc = best_index(&hm, team);
        let loss = em[ec].get(team) - em[hc].get(team);
        let mut errors = Vec::with_capacity(4 * ws.len() * cands.len());
        for o in ws {
            for (e, h) in o.exact.iter().zip(&o.hybrid) {
                for rel in 0..4u8 {
                    let t = (team + rel) % 4;
                    errors.push(h.get(t) as i64 - e.get(t) as i64);
                }
            }
        }
        decisions.push(DecisionCheck {
            id: p.d.id,
            position: p.d.position + 1,
            stratum: p.d.stratum,
            team,
            cands: cands.len(),
            worlds: ws.len() as u64,
            exact_choice: ec,
            hybrid_choice: hc,
            loss,
            agree: loss == 0,
            errors,
            exact_worlds: ws.iter().filter(|o| o.exact_solved).count() as u64,
            exact_us: ws.iter().map(|o| o.exact_us).sum(),
            hybrid_us: ws.iter().map(|o| o.hybrid_us).sum(),
            exact_nodes: ws.iter().map(|o| o.exact_nodes).sum(),
            hybrid_nodes: ws.iter().map(|o| o.hybrid_nodes).sum(),
            frontier: ws.iter().map(|o| o.frontier).sum(),
            reach: slices.map(|p| {
                let mut r = Reach::new(p.len());
                for o in ws {
                    r.add(o.reach.as_ref().expect("every world recorded"));
                }
                r
            }),
            world_checks: ws.to_vec(),
        });
    }
    Ok(CheckRun {
        decisions,
        verified,
        net: (
            timed.ns.load(Ordering::Relaxed),
            timed.n.load(Ordering::Relaxed),
        ),
        mismatched,
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
    /// Values compared (candidate × world × team).
    pub values: usize,
    /// Value error: mean (bias) of each team, relative to the decider (its own team, the next,
    /// the opposite, the previous), as fractions.
    pub bias: [f64; 4],
    /// Value error over all four teams: standard deviation.
    pub sd: f64,
    /// Value error over all four teams: 99.9th percentile of the absolute error.
    pub p999_abs: f64,
    /// Value error over all four teams: the largest absolute error.
    pub max_abs: f64,
    /// The decider's team alone: SD and 99.9th percentile of the absolute error.
    pub decider_sd: f64,
    /// See `decider_sd`.
    pub decider_p999_abs: f64,
    /// Decision loss within the standard at the 95% upper bound.
    pub loss_pass: bool,
    /// Every team's bias within ±0.01% (or the check's `--standard`, [`judge`]).
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
    let errs: Vec<i64> = ds.iter().flat_map(|d| d.errors.iter().copied()).collect();
    let stats = |e: &[i64]| -> (f64, f64, f64, f64) {
        let m = e.len().max(1) as f64;
        let mean = e.iter().map(|&x| x as f64).sum::<f64>() / m;
        let var = e.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / m;
        let mut abs: Vec<u64> = e.iter().map(|x| x.unsigned_abs()).collect();
        abs.sort_unstable();
        let p999 = if abs.is_empty() {
            0
        } else {
            abs[((abs.len() as f64 * 0.999).ceil() as usize).clamp(1, abs.len()) - 1]
        };
        let max = abs.last().copied().unwrap_or(0);
        (
            mean / unit,
            var.sqrt() / unit,
            p999 as f64 / unit,
            max as f64 / unit,
        )
    };
    let rel: Vec<Vec<i64>> = (0..4)
        .map(|k| errs.iter().skip(k).step_by(4).copied().collect())
        .collect();
    let bias: [f64; 4] = core::array::from_fn(|k| stats(&rel[k]).0);
    let (_, sd, p999_abs, max_abs) = stats(&errs);
    let (_, decider_sd, decider_p999_abs, _) = stats(&rel[0]);
    Summary {
        decisions: n,
        mean_loss,
        loss_upper95: upper,
        agreement: ds.iter().filter(|d| d.agree).count() as f64 / n.max(1) as f64,
        values: errs.len(),
        bias,
        sd,
        p999_abs,
        max_abs,
        decider_sd,
        decider_p999_abs,
        loss_pass: upper <= STANDARD,
        bias_pass: bias.iter().all(|b| b.abs() <= STANDARD),
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

/// One slice of a check (docs/m3-hybrid/audit-loop.md §3 item 2): the decisions in a group of a
/// predicate file, or in none.
#[derive(Clone, Debug, Serialize)]
pub struct Slice {
    /// The group's index in the file (`None`: the decisions in no group).
    pub group: Option<usize>,
    /// The group's name (`no group`).
    pub name: String,
    /// The group's conditions, as text.
    pub conditions: String,
    /// The decisions in the slice (ids).
    pub ids: Vec<u64>,
    /// Over the check's decisions, the group's share of all the reached frontier positions
    /// (the group's weight as the hybrid reaches it; `None` for "no group").
    pub reached_share: Option<f64>,
    /// The §3 measures of the slice's decisions.
    pub summary: Summary,
    /// The decision-loss bound passes (`None` with no decisions: nothing to measure, a mini
    /// golden set is needed).
    pub pass: Option<bool>,
}

/// The slices of a check run with a predicate file ([`check_worlds_with`]): for each group, the
/// decisions in it (at least the share `micro`, in millionths, of their reached frontier
/// positions in the group: `crate::audit::Reach::groups_at`), then the decisions in no group;
/// each with the §3 measures (the bootstrap tagged `1000 + g`, and `999` for no group).
///
/// # Panics
/// If the run did not record reaches.
pub fn slices(
    run: &CheckRun,
    preds: &Predicates,
    micro: u64,
    boots: u64,
    boot_seed: u64,
) -> Vec<Slice> {
    let groups: Vec<Vec<u32>> = run
        .decisions
        .iter()
        .map(|d| {
            d.reach
                .as_ref()
                .expect("a check with slices records reaches")
                .groups_at(micro)
        })
        .collect();
    let mut total = Reach::new(preds.len());
    for d in &run.decisions {
        total.add(d.reach.as_ref().expect("recorded"));
    }
    let make = |g: Option<usize>, pick: &dyn Fn(&[u32]) -> bool| {
        let ds: Vec<&DecisionCheck> = run
            .decisions
            .iter()
            .zip(&groups)
            .filter(|(_, gs)| pick(gs))
            .map(|(d, _)| d)
            .collect();
        let summary = summarise(&ds, boots, boot_seed, g.map_or(999, |g| 1000 + g as u64));
        Slice {
            group: g,
            name: g.map_or("no group".to_string(), |g| preds.groups[g].name.clone()),
            conditions: g.map_or(String::new(), |g| preds.groups[g].conditions_text()),
            ids: ds.iter().map(|d| d.id).collect(),
            reached_share: g.map(|g| total.share(g)),
            pass: (!ds.is_empty()).then_some(summary.loss_pass),
            summary,
        }
    };
    let mut out: Vec<Slice> = (0..preds.len())
        .map(|g| make(Some(g), &|gs: &[u32]| gs.contains(&(g as u32))))
        .collect();
    out.push(make(None, &|gs: &[u32]| gs.is_empty()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// `--standard`: percent text to the fraction ("0.01" is exactly `STANDARD`), refusals, and
    /// the judgement of a summary by it (decision-loss bound and every team's bias).
    #[test]
    fn the_standard_is_a_setting() {
        assert_eq!(parse_standard("0.01"), Ok(STANDARD));
        assert_eq!(parse_standard("0.005"), Ok(5e-5));
        assert_eq!(parse_standard(" 0.0075 "), Ok(7.5e-5));
        assert_eq!(parse_standard("1"), Ok(0.01));
        for bad in ["0", "", "-0.01", "1.5", "abc", "1e-2", "0.01%"] {
            assert!(parse_standard(bad).is_err(), "{bad}");
        }
        let d = |loss: u64, e: i64| DecisionCheck {
            id: 0,
            position: 1,
            stratum: 1,
            team: 0,
            cands: 2,
            worlds: 1,
            exact_choice: 0,
            hybrid_choice: 1,
            loss,
            agree: loss == 0,
            errors: vec![e, -e, 0, 0],
            exact_worlds: 0,
            exact_us: 0,
            hybrid_us: 0,
            exact_nodes: 0,
            hybrid_nodes: 0,
            frontier: 0,
            reach: None,
            world_checks: Vec::new(),
        };
        // Losses of about 0.006%, a bias of +-0.003%.
        let ds: Vec<DecisionCheck> = (0..20).map(|i| d(250_000 + 1000 * i, 130_000)).collect();
        let refs: Vec<&DecisionCheck> = ds.iter().collect();
        let mut s = summarise(&refs, 200, 1, 0);
        assert!(s.loss_pass && s.bias_pass, "{s:?}");
        judge(&mut s, 5e-5);
        assert!(!s.loss_pass && s.bias_pass, "{s:?}");
        judge(&mut s, 2e-5);
        assert!(!s.loss_pass && !s.bias_pass, "{s:?}");
        let u = s.loss_upper95;
        judge(&mut s, u);
        assert!(s.loss_pass && s.bias_pass);
        judge(&mut s, STANDARD);
        assert!(s.loss_pass && s.bias_pass);
    }

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
            rules_robber_burn: false,
            exact_solve_threads: 1,
            exact_tt_mb: 4,
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

    /// With slices, the check's values are unchanged, each decision's reach is its hybrid
    /// searches' (`crate::audit::decision_reach` with the same evaluator and table), the same on
    /// 1 and 3 threads, and the slices hold the decisions at the share, then those in no group.
    #[test]
    fn slices_record_the_hybrids_own_reach() {
        let starts = collect_starts(300, 3);
        let ds = build_decisions(3, 8, 5, &starts);
        let l = leaf();
        let src = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: engine::game::Rules::HOUSE,
        };
        let f = FnFrontier(even as fn(&CanonPos, &MatchCtx) -> MoneyChance);
        let job = |threads| CheckJob {
            cut: 4,
            seed: 5,
            label: 1,
            to: 3,
            threads,
            tt_mb: 4,
            verify_pct: 0.0,
            recompute: false,
            rules_robber_burn: false,
            exact_solve_threads: 1,
            exact_tt_mb: 4,
        };
        let text = format!(
            r#"[{{"name": "next team no trick", "conditions": [{{"feature": {}, "op": "==", "value": 1}}]}},
                {{"name": "never", "conditions": [{{"feature": {}, "op": "==", "value": 1}},
                                                   {{"feature": {}, "op": "==", "value": 1}}]}}]"#,
            nn::encode::OFF_TRICKS + 5,
            nn::encode::OFF_TRUMP,
            nn::encode::OFF_TRUMP + 1
        );
        let p = Predicates::parse(text.as_bytes()).expect("parses");
        let plain = run_check(&ds, &BTreeMap::new(), &job(1), &ctx, &f).expect("runs");
        let a = run_check_slices(&ds, &BTreeMap::new(), &job(1), &ctx, &f, Some(&p)).expect("runs");
        let b = run_check_slices(&ds, &BTreeMap::new(), &job(3), &ctx, &f, Some(&p)).expect("runs");
        let vals = |r: &CheckRun| -> Vec<(u64, u64, Vec<i64>)> {
            r.decisions
                .iter()
                .map(|d| (d.id, d.loss, d.errors.clone()))
                .collect()
        };
        assert_eq!(vals(&plain), vals(&a), "slices never change a value");
        assert!(plain.decisions.iter().all(|d| d.reach.is_none()));
        let reaches = |r: &CheckRun| -> Vec<Reach> {
            r.decisions
                .iter()
                .map(|d| d.reach.clone().expect("recorded"))
                .collect()
        };
        assert_eq!(reaches(&a), reaches(&b), "thread-independent");
        let mut mem = SolveMem::new();
        for (d, x) in ds.iter().zip(&a.decisions) {
            let want = crate::audit::decision_reach(
                d,
                &p,
                &ctx,
                4,
                4,
                measure_seed(5, 1, d.id),
                3,
                Some(&f),
                &mut mem,
            );
            assert_eq!(x.reach.as_ref(), Some(&want), "decision {}", d.id);
            assert_eq!(want.in_group[1], 0);
        }
        let sl = slices(&a, &p, 500_000, 50, 1);
        assert_eq!(sl.len(), 3);
        assert_eq!(
            sl.iter().map(|x| x.name.as_str()).collect::<Vec<_>>(),
            ["next team no trick", "never", "no group"]
        );
        let half: Vec<u64> = a
            .decisions
            .iter()
            .filter(|d| {
                let r = d.reach.as_ref().expect("recorded");
                2 * r.in_group[0] >= r.reached
            })
            .map(|d| d.id)
            .collect();
        assert_eq!(sl[0].ids, half);
        assert!(sl[1].ids.is_empty() && sl[1].pass.is_none());
        assert_eq!(sl[0].ids.len() + sl[2].ids.len(), 8, "a partition here");
        assert_eq!(sl[0].summary.decisions, sl[0].ids.len());
        assert!(!sl[0].ids.is_empty() && !sl[2].ids.is_empty(), "{sl:?}");
        assert_eq!(
            sl[0].conditions,
            format!("f{} == 1", nn::encode::OFF_TRICKS + 5)
        );
    }
}
