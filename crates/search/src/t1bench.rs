//! The trick-1 bench (docs/M3-SPEC.md §13.1 item 11): which way of spreading exact trick-1
//! world solves over a big machine gives the most worlds per hour and per dollar, measured on a
//! small fixed set of real yardstick worlds, the same on every machine.
//!
//! **Sets** (`position:decisions:worlds` parts; decisions are the first ones at that position in
//! the stratified decision set, worlds `0..worlds` of the yardstick's world label, so every bench
//! world is a real yardstick world and the work is identical on every machine):
//! - [`MID_SET`]: 48 worlds at positions 4–6 (16 each), run under every configuration ("worlds
//!   at once × threads per world", and the position-aware plan): the ranking;
//! - [`LATE_SET`]: 128 worlds at positions 7–8, run with one thread per world, all at once, and
//!   with the best configuration of the mid set;
//! - [`LEAD_SET`]: one world at each of positions 1–3 (the leader's included), run once with the
//!   best configuration: these positions are most of the real work, and their solver node counts
//!   give each position's cost relative to the others.
//!
//! One-thread costs measured on the 4-vCPU Xeon (2026-10-14): position 4 about 200 s a world
//! (about 1 billion nodes), position 6 about 16 s, positions 7–8 10–50 s; the leader's worlds
//! of the M2 trick-1 runs 1–29 billion nodes.
//!
//! **Measures**, per run: each world's wall time, threads, solver nodes and table use; the
//! process's CPU time and peak memory. The **occupancy** of a world is its wall time × its
//! threads (the thread-seconds it held). In a long run the machine is kept full, so the steady
//! rate is `3600 · T / mean occupancy` worlds per hour on `T` threads: the bench's own tails do not
//! count against a configuration. The rate for the **real position mix** (equal numbers of
//! decisions at the 8 positions, as the stratified sets have) uses measured occupancies where the
//! run had the position, and elsewhere the run's own occupancy per solver node scaled by each
//! position's mean node count (from every run that solved worlds there).
//!
//! **Early stop**: runs on the same set are ranked by their total occupancy. A run stops as soon
//! as its occupancy so far (finished worlds plus the time held by running ones, a lower bound)
//! exceeds `stop_ratio` × the best total so far: it cannot catch up. Its worlds not finished are
//! then left out of the comparison and its rate is reported as an upper bound.
//!
//! **Check**: every world solved by more than one run must have byte-identical values in all of
//! them (threads, table sizes and schedules change speed only); any difference is reported as
//! a failure.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use solver::{ParProgress, SolveMem};

use crate::measure::{DecisionSpec, measure_seed};
use crate::pimc::{
    Candidate, Ctx, SearchConfig, WorldSolve, candidates, exact_world_row, solves_per_world,
};
use crate::plan::{AutoParams, PlanTask, WorldPlan, plan_auto, run_planned};

/// The ranking set: 16 worlds at each of positions 4–6 (4 decisions × 4 worlds).
pub const MID_SET: &str = "3:4:4,4:4:4,5:4:4";
/// The late set: 64 worlds at each of positions 7–8 (4 decisions × 16 worlds).
pub const LATE_SET: &str = "6:4:16,7:4:16";
/// One world at each of positions 1–3, the leader's included.
pub const LEAD_SET: &str = "0:1:1,1:1:1,2:1:1";

/// One part of a bench set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetPart {
    /// Position in the trick (0 = leader).
    pub position: u8,
    /// The first this many decisions at the position (decision id order).
    pub decisions: u32,
    /// Worlds `0..worlds` of each.
    pub worlds: u32,
}

/// Parses `position:decisions:worlds,…` (positions 0–7).
pub fn parse_set(s: &str) -> Result<Vec<SetPart>, String> {
    s.split(',')
        .filter(|x| !x.trim().is_empty())
        .map(|x| {
            let v: Vec<u32> = x
                .trim()
                .split(':')
                .map(|n| n.trim().parse::<u32>())
                .collect::<Result<_, _>>()
                .map_err(|_| format!("bad set part {x:?} (position:decisions:worlds)"))?;
            match v[..] {
                [p, d, w] if p < 8 && d > 0 && w > 0 => Ok(SetPart {
                    position: p as u8,
                    decisions: d,
                    worlds: w,
                }),
                _ => Err(format!(
                    "bad set part {x:?}: position 0–7, at least one decision and world"
                )),
            }
        })
        .collect()
}

/// The bench's tasks for `set`: (index into `ds`, world), decisions in id order per position.
pub fn select(ds: &[DecisionSpec], set: &[SetPart]) -> Result<Vec<(usize, u64)>, String> {
    let mut out = Vec::new();
    for part in set {
        let mut at: Vec<usize> = (0..ds.len())
            .filter(|&i| ds[i].position == part.position)
            .collect();
        at.sort_by_key(|&i| ds[i].id);
        if at.len() < part.decisions as usize {
            return Err(format!(
                "position {} has {} decisions, the set wants {}",
                part.position + 1,
                at.len(),
                part.decisions
            ));
        }
        for &i in &at[..part.decisions as usize] {
            for w in 0..u64::from(part.worlds) {
                out.push((i, w));
            }
        }
    }
    Ok(out)
}

/// How a bench run spreads its worlds.
#[derive(Clone, Debug, PartialEq)]
pub enum BenchSchedule {
    /// `worlds_at_once` worlds at a time, each on `solve_threads` threads with `tt_mb` tables.
    Fixed {
        worlds_at_once: usize,
        solve_threads: usize,
        tt_mb: usize,
    },
    /// The position-aware plan ([`crate::plan`]).
    Auto {
        threads: usize,
        budget_mb: u64,
        params: Box<AutoParams>,
    },
}

/// One world of a bench run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchWorld {
    pub id: u64,
    pub position: u8,
    pub world: u64,
    /// Exact solves the world takes (one per burn of a joint decision).
    pub solves: u32,
    pub threads: usize,
    pub tt_mb: usize,
    /// Whether it was started, and finished (stopped runs leave some unfinished or unstarted).
    pub started: bool,
    pub finished: bool,
    /// Start (seconds from the run's start) and wall time.
    pub start_s: f64,
    pub wall_s: f64,
    pub nodes: u64,
    pub tt_probes: u64,
    pub tt_hits: u64,
    /// The largest table of its solves at the end: entries and bytes in use.
    pub tt_entries: u64,
    pub tt_bytes: u64,
    pub tt_replaced: u64,
    /// Every candidate's value (four u64 per candidate), when finished.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<[u64; 4]>>,
}

impl BenchWorld {
    /// Thread-seconds held (a lower bound when unfinished).
    pub fn occupancy(&self) -> f64 {
        self.wall_s * self.threads as f64
    }

    /// The share of the table's cap in use at the end (0–1).
    pub fn tt_fill(&self) -> f64 {
        let cap = (self.tt_mb as f64) * (1u64 << 20) as f64;
        let slots = cap / solver::fast::FAST_TT_SLOT_BYTES as f64;
        if slots > 0.0 {
            (self.tt_entries as f64 / slots).min(1.0)
        } else {
            0.0
        }
    }
}

/// One bench run (one configuration on one set).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchRun {
    pub name: String,
    pub set: String,
    /// The schedule, as given or planned.
    pub schedule: Value,
    /// The machine's threads the run had.
    pub threads: usize,
    pub cap_s: f64,
    /// The occupancy at which the run stops (the ranking rule), if any.
    #[serde(default)]
    pub stop_occ_s: Option<f64>,
    pub wall_s: f64,
    /// CPU seconds the process used during the run, and its peak resident memory.
    #[serde(default)]
    pub cpu_s: Option<f64>,
    #[serde(default)]
    pub peak_rss_mb: Option<u64>,
    /// Why it stopped early: `cap` (its time limit) or `ranking` (it cannot be the best).
    #[serde(default)]
    pub stopped: Option<String>,
    pub worlds: Vec<BenchWorld>,
}

/// Live counters of a bench run (progress reports only).
#[derive(Debug, Default)]
pub struct BenchProgress {
    pub total: AtomicU64,
    pub done: AtomicU64,
    pub running: AtomicU64,
}

struct Prep {
    d: usize,
    view: engine::view::SeatView,
    cands: Vec<Candidate>,
    dseed: u64,
}

/// Run `set` of decisions `ds` (yardstick worlds of `seed` and `label`) under `sched` for at
/// most `cap_s` seconds, stopping early once the occupancy reaches `stop_occ_s`.
#[allow(clippy::too_many_arguments)]
pub fn run_bench(
    name: &str,
    ds: &[DecisionSpec],
    set: &str,
    seed: u64,
    label: u64,
    ctx: &Ctx,
    sched: &BenchSchedule,
    cap_s: f64,
    stop_occ_s: Option<f64>,
    progress: Option<&BenchProgress>,
) -> Result<BenchRun, String> {
    let parts = parse_set(set)?;
    let tasks = select(ds, &parts)?;
    let mut preps: Vec<Prep> = Vec::new();
    let mut prep_of: BTreeMap<usize, usize> = BTreeMap::new();
    for &(d, _) in &tasks {
        prep_of.entry(d).or_insert_with(|| {
            let (_, view, legal) = ds[d].replay();
            let (_, cands) = candidates(&view, &legal);
            preps.push(Prep {
                d,
                view,
                cands,
                dseed: measure_seed(seed, label, ds[d].id),
            });
            preps.len() - 1
        });
    }
    let worlds: Vec<_> = preps
        .iter()
        .map(|p| ctx.source.prepare(&p.view, ctx.rules))
        .collect();
    let task_prep: Vec<usize> = tasks.iter().map(|(d, _)| prep_of[d]).collect();
    let plan_tasks: Vec<PlanTask> = task_prep
        .iter()
        .map(|&j| PlanTask {
            trick: ds[preps[j].d].trick,
            position: ds[preps[j].d].position,
            solves: solves_per_world(&preps[j].cands),
        })
        .collect();
    let (plans, order, threads, budget, schedule) = match sched {
        BenchSchedule::Fixed {
            worlds_at_once,
            solve_threads,
            tt_mb,
        } => {
            let plan = WorldPlan {
                threads: (*solve_threads).max(1),
                tt_mb: *tt_mb,
            };
            // Longest first, as `search worlds` does: earlier positions first.
            let mut order: Vec<usize> = (0..tasks.len()).collect();
            order.sort_by_key(|&k| (plan_tasks[k].position, tasks[k].0, tasks[k].1));
            (
                vec![plan; tasks.len()],
                order,
                (*worlds_at_once).max(1) * plan.threads,
                u64::MAX,
                json!({ "kind": "fixed", "worlds_at_once": worlds_at_once,
                        "solve_threads": plan.threads, "tt_mb": tt_mb }),
            )
        }
        BenchSchedule::Auto {
            threads,
            budget_mb,
            params,
        } => {
            let pl = plan_auto(&plan_tasks, *threads, budget_mb << 20, params);
            (
                pl.plans,
                pl.order,
                *threads,
                budget_mb << 20,
                json!({ "kind": "auto", "plan": pl.record }),
            )
        }
    };
    let n = tasks.len();
    if let Some(p) = progress {
        p.total.store(n as u64, Ordering::Relaxed);
    }
    let stop = AtomicBool::new(false);
    let stopped: Mutex<Option<String>> = Mutex::new(None);
    let flags: Vec<ParProgress> = (0..n).map(|_| ParProgress::default()).collect();
    // Start time (µs from t0, +1; 0 = not started) and end flag of each task, for the monitor.
    let starts: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
    let ended: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();
    let occ_done_us = AtomicU64::new(0);
    let finished_all = AtomicBool::new(false);
    let t0 = Instant::now();
    let mut out: Vec<Option<BenchWorld>> = vec![None; n];
    let halt = |why: &str| {
        let mut s = stopped.lock().unwrap_or_else(|e| e.into_inner());
        if s.is_none() {
            *s = Some(why.to_string());
        }
        stop.store(true, Ordering::Relaxed);
        for f in &flags {
            f.abort.store(true, Ordering::Relaxed);
        }
    };
    let result = std::thread::scope(|s| {
        // The monitor: the time cap and the ranking rule.
        s.spawn(|| {
            while !finished_all.load(Ordering::Relaxed) {
                let now = t0.elapsed();
                if now.as_secs_f64() >= cap_s {
                    halt("cap");
                    break;
                }
                if let Some(limit) = stop_occ_s {
                    let now_us = now.as_micros() as u64;
                    let mut occ = occ_done_us.load(Ordering::Relaxed) as f64;
                    for k in 0..n {
                        let st = starts[k].load(Ordering::Relaxed);
                        if st > 0 && !ended[k].load(Ordering::Relaxed) {
                            occ += now_us.saturating_sub(st - 1) as f64 * plans[k].threads as f64;
                        }
                    }
                    if occ / 1e6 >= limit {
                        halt("ranking");
                        break;
                    }
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        let work = |k: usize, plan: WorldPlan, mem: &mut SolveMem| {
            if stop.load(Ordering::Relaxed) {
                return None;
            }
            starts[k].store(t0.elapsed().as_micros() as u64 + 1, Ordering::Relaxed);
            if let Some(p) = progress {
                p.running.fetch_add(1, Ordering::Relaxed);
            }
            let j = task_prep[k];
            let p = &preps[j];
            let cfg = SearchConfig {
                threads: 1,
                solve_threads: plan.threads,
                tt_mb: plan.tt_mb,
                ..SearchConfig::default()
            };
            let mut run = WorldSolve::default();
            let v = exact_world_row(
                &p.view,
                &*worlds[j],
                &p.cands,
                tasks[k].1,
                p.dseed,
                ctx,
                &cfg,
                Some(&flags[k]),
                mem,
                &mut run,
            );
            ended[k].store(true, Ordering::Relaxed);
            if let Some(p) = progress {
                p.running.fetch_sub(1, Ordering::Relaxed);
            }
            Some((v, run))
        };
        let r = run_planned(
            &order,
            &plans,
            threads,
            budget,
            &stop,
            None,
            work,
            |k, r, ran| {
                let d = &ds[preps[task_prep[k]].d];
                let mut w = BenchWorld {
                    id: d.id,
                    position: d.position,
                    world: tasks[k].1,
                    solves: plan_tasks[k].solves,
                    threads: ran.plan.threads,
                    tt_mb: ran.plan.tt_mb,
                    started: false,
                    finished: false,
                    start_s: ran.start_s,
                    wall_s: 0.0,
                    nodes: 0,
                    tt_probes: 0,
                    tt_hits: 0,
                    tt_entries: 0,
                    tt_bytes: 0,
                    tt_replaced: 0,
                    values: None,
                };
                if let Some((v, run)) = r {
                    w.started = true;
                    w.wall_s = ran.wall_s;
                    w.finished = v.is_some();
                    w.values = v.map(|v| v.into_iter().map(|x| x.0).collect());
                    let st = run.stats;
                    (w.nodes, w.tt_probes, w.tt_hits) = (st.nodes, st.tt_probes, st.tt_hits);
                    (w.tt_entries, w.tt_bytes, w.tt_replaced) =
                        (st.tt_entries, st.tt_bytes, st.tt_replaced);
                    occ_done_us.fetch_add(
                        (ran.wall_s * 1e6 * ran.plan.threads as f64) as u64,
                        Ordering::Relaxed,
                    );
                    if let Some(p) = progress {
                        p.done.fetch_add(1, Ordering::Relaxed);
                    }
                }
                out[k] = Some(w);
                Ok(())
            },
        );
        finished_all.store(true, Ordering::Relaxed);
        r
    });
    result?;
    let wall_s = t0.elapsed().as_secs_f64();
    let worlds_out: Vec<BenchWorld> = out
        .into_iter()
        .enumerate()
        .map(|(k, w)| {
            w.unwrap_or_else(|| {
                let d = &ds[preps[task_prep[k]].d];
                BenchWorld {
                    id: d.id,
                    position: d.position,
                    world: tasks[k].1,
                    solves: plan_tasks[k].solves,
                    threads: plans[k].threads,
                    tt_mb: plans[k].tt_mb,
                    started: false,
                    finished: false,
                    start_s: 0.0,
                    wall_s: 0.0,
                    nodes: 0,
                    tt_probes: 0,
                    tt_hits: 0,
                    tt_entries: 0,
                    tt_bytes: 0,
                    tt_replaced: 0,
                    values: None,
                }
            })
        })
        .collect();
    let stopped = stopped.into_inner().unwrap_or_else(|e| e.into_inner());
    Ok(BenchRun {
        name: name.to_string(),
        set: set.to_string(),
        schedule,
        threads,
        cap_s,
        stop_occ_s,
        wall_s,
        cpu_s: None,
        peak_rss_mb: None,
        stopped: if worlds_out.iter().all(|w| w.finished) {
            None
        } else {
            stopped
        },
        worlds: worlds_out,
    })
}

// ---------------------------------------------------------------- analysis

/// Inputs of [`analyse`] besides the runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AnalyseParams {
    /// The machine's usable threads.
    pub threads: usize,
    /// Dollars per hour of the machine (0 = unknown).
    pub dph: f64,
    /// For the projection: a yardstick of this many decisions × worlds per decision.
    pub golden_decisions: u64,
    pub golden_worlds: u64,
    /// Ratio of the best total occupancy at which runs were stopped (for the text).
    pub stop_ratio: f64,
    /// The machine, for the text.
    pub cpu_model: String,
    pub mem_gb: f64,
    /// The ranking set: the best single configuration is chosen among the runs on it.
    pub mid_set: String,
}

impl Default for AnalyseParams {
    fn default() -> Self {
        AnalyseParams {
            threads: 1,
            dph: 0.0,
            golden_decisions: 192,
            golden_worlds: 200,
            stop_ratio: 1.5,
            cpu_model: String::new(),
            mem_gb: 0.0,
            mid_set: MID_SET.to_string(),
        }
    }
}

/// Per-run numbers used by the runner while the bench goes on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunTotals {
    /// Total occupancy (thread-seconds; a lower bound if not every world finished).
    pub occupancy_s: f64,
    pub finished: usize,
    pub worlds: usize,
}

/// The totals of a run.
pub fn totals(r: &BenchRun) -> RunTotals {
    RunTotals {
        occupancy_s: r.worlds.iter().map(BenchWorld::occupancy).sum(),
        finished: r.worlds.iter().filter(|w| w.finished).count(),
        worlds: r.worlds.len(),
    }
}

fn mean(x: &[f64]) -> Option<f64> {
    (!x.is_empty()).then(|| x.iter().sum::<f64>() / x.len() as f64)
}

fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// The run that first solved a world, and its values.
type FirstSeen<'a> = (&'a str, &'a Vec<[u64; 4]>);

/// Occupancy per position (thread-seconds per world), and whether it was measured (else scaled
/// by nodes).
type PositionOcc = [Option<(f64, bool)>; 8];

/// Per-position measurements of one run.
#[derive(Clone, Debug, Default)]
struct PosStat {
    finished: usize,
    unfinished: usize,
    occ: Vec<f64>,
    wall: Vec<f64>,
    threads: Vec<f64>,
    fill: Vec<f64>,
    /// Table entries overwritten, and entries at the end (summed over worlds).
    replaced: u64,
    entries: u64,
}

fn pos_stats(r: &BenchRun) -> BTreeMap<u8, PosStat> {
    let mut m: BTreeMap<u8, PosStat> = BTreeMap::new();
    for w in r.worlds.iter().filter(|w| w.started) {
        let s = m.entry(w.position).or_default();
        if w.finished {
            s.finished += 1;
        } else {
            s.unfinished += 1;
        }
        s.occ.push(w.occupancy());
        s.wall.push(w.wall_s);
        s.threads.push(w.threads as f64);
        s.fill.push(w.tt_fill());
        s.replaced += w.tt_replaced;
        s.entries += w.tt_entries;
    }
    m
}

/// The bench's analysis (`results.json`): per run, the identity check, the position cost
/// ratios, the steady rates for the real position mix, dollars per 1000 worlds, the best
/// configuration overall and per position, suggested plan costs, and a projection for the
/// yardstick. `ok` is false when any two runs disagree on a world's values.
pub fn analyse(runs: &[BenchRun], p: &AnalyseParams) -> Value {
    let t = p.threads.max(1) as f64;
    // Identity: every (decision, world) solved by more than one run.
    let mut first: BTreeMap<(u64, u64), FirstSeen> = BTreeMap::new();
    let mut compared = 0u64;
    let mut mismatches = Vec::new();
    for r in runs {
        for w in &r.worlds {
            let Some(v) = &w.values else { continue };
            match first.get(&(w.id, w.world)) {
                None => {
                    first.insert((w.id, w.world), (&r.name, v));
                }
                Some((other, v0)) => {
                    compared += 1;
                    if *v0 != v {
                        mismatches.push(json!({ "decision": w.id, "world": w.world,
                                                "runs": [other, r.name] }));
                    }
                }
            }
        }
    }
    // Position cost by nodes: per world the fewest nodes any run needed (parallel solves search
    // a little more), averaged per position.
    let mut nodes_of: BTreeMap<(u64, u64), (u8, u64)> = BTreeMap::new();
    for w in runs.iter().flat_map(|r| &r.worlds).filter(|w| w.finished) {
        let e = nodes_of
            .entry((w.id, w.world))
            .or_insert((w.position, w.nodes));
        e.1 = e.1.min(w.nodes);
    }
    let mut by_pos: BTreeMap<u8, Vec<f64>> = BTreeMap::new();
    for &(pos, nodes) in nodes_of.values() {
        by_pos.entry(pos).or_default().push(nodes as f64);
    }
    // Positions without a finished world: from the nearest measured one at the model's ratio
    // (2.257 per position), flagged.
    let measured: BTreeMap<u8, f64> = by_pos
        .iter()
        .filter_map(|(&pos, v)| mean(v).map(|m| (pos, m)))
        .collect();
    let model_ratio = AutoParams::default().position_ratio;
    let node_cost: Vec<Option<(f64, bool)>> = (0..8u8)
        .map(|pos| {
            if let Some(&m) = measured.get(&pos) {
                return Some((m, true));
            }
            let (&q, &m) = measured
                .iter()
                .min_by_key(|(q, _)| (i16::from(**q) - i16::from(pos)).abs())?;
            Some((m * model_ratio.powi(i32::from(q) - i32::from(pos)), false))
        })
        .collect();
    let mut run_rows = Vec::new();
    // Per run: occupancy per position (measured, or extrapolated by nodes).
    let mut full: Vec<(String, PositionOcc, bool)> = Vec::new();
    for r in runs {
        let ps = pos_stats(r);
        let tot = totals(r);
        let complete = tot.finished == tot.worlds;
        // The run's occupancy per node over its measured positions.
        let mut occ_sum = 0.0;
        let mut node_sum = 0.0;
        let mut occ_p: PositionOcc = [None; 8];
        for (&pos, s) in &ps {
            if s.finished == 0 {
                continue;
            }
            let fin: Vec<f64> = r
                .worlds
                .iter()
                .filter(|w| w.position == pos && w.finished)
                .map(BenchWorld::occupancy)
                .collect();
            let m = mean(&fin).unwrap_or(0.0);
            occ_p[usize::from(pos)] = Some((m, true));
            if let Some((nc, _)) = node_cost[usize::from(pos)] {
                occ_sum += m;
                node_sum += nc;
            }
        }
        let per_node = (node_sum > 0.0).then(|| occ_sum / node_sum);
        for pos in 0..8 {
            if occ_p[pos].is_none()
                && let (Some(k), Some((nc, _))) = (per_node, node_cost[pos])
            {
                occ_p[pos] = Some((k * nc, false));
            }
        }
        let mix: Option<f64> = occ_p
            .iter()
            .map(|x| x.map(|(o, _)| o))
            .sum::<Option<f64>>()
            .map(|s| s / 8.0);
        let rate = mix.filter(|&m| m > 0.0).map(|m| 3600.0 * t / m);
        let per_k =
            |rate: Option<f64>| rate.filter(|_| p.dph > 0.0).map(|x| r3(1000.0 * p.dph / x));
        let set_rate =
            (tot.occupancy_s > 0.0).then(|| 3600.0 * t * tot.finished as f64 / tot.occupancy_s);
        let cpu_pct = r
            .cpu_s
            .filter(|_| r.wall_s > 0.0)
            .map(|c| r3(100.0 * c / (r.wall_s * t)));
        let positions: Vec<Value> = ps
            .iter()
            .map(|(&pos, s)| {
                json!({
                    "position": pos + 1, "finished": s.finished, "unfinished": s.unfinished,
                    "mean_threads": mean(&s.threads).map(r3),
                    "mean_wall_s": mean(&s.wall).map(r3),
                    "mean_occupancy_s": mean(&s.occ).map(r3),
                    "mean_table_fill": mean(&s.fill).map(r3),
                    // Table entries overwritten per entry kept: near 0 means the tables were
                    // large enough for these worlds.
                    "replaced_per_entry": (s.entries > 0).then(|| r3(s.replaced as f64 / s.entries as f64)),
                })
            })
            .collect();
        run_rows.push(json!({
            "name": r.name, "set": r.set, "schedule": r.schedule, "wall_s": r3(r.wall_s),
            "complete": complete, "stopped": r.stopped,
            "worlds": tot.worlds, "finished": tot.finished,
            "occupancy_s": r3(tot.occupancy_s),
            "cpu_util_pct": cpu_pct, "peak_rss_mb": r.peak_rss_mb,
            "max_table_mb": r.worlds.iter().map(|w| w.tt_bytes >> 20).max(),
            "mean_table_fill": mean(&r.worlds.iter().filter(|w| w.finished).map(BenchWorld::tt_fill).collect::<Vec<_>>()).map(r3),
            "set_worlds_per_hour": set_rate.map(|x| x.round()),
            "mix_worlds_per_hour": rate.map(|x| x.round()),
            "mix_rate_is_upper_bound": !complete,
            "dollars_per_1000_worlds": per_k(rate),
            "occupancy_by_position_s": occ_p.iter().map(|x| x.map(|(o, m)| json!({ "s": r3(o), "measured": m }))).collect::<Vec<_>>(),
            "positions": positions,
        }));
        full.push((r.name.clone(), occ_p, complete));
    }
    // The best per position, and the combined schedule: the lowest occupancy measured by a
    // complete run, else by any run (a stopped run's finished worlds), else scaled by nodes.
    let mut best_pos = Vec::new();
    let mut combined = 0.0;
    let mut combined_ok = true;
    let mut t1_costs = [0.0f64; 8];
    for pos in 0..8 {
        let cands: Vec<(f64, bool, &String, u8)> = full
            .iter()
            .filter_map(|(n, o, complete)| {
                o[pos].map(|(x, m)| {
                    let rank = match (m, complete) {
                        (true, true) => 0,
                        (true, false) => 1,
                        _ => 2,
                    };
                    (x, m, n, rank)
                })
            })
            .filter(|c| c.0 > 0.0)
            .collect();
        let top = cands.iter().map(|c| c.3).min();
        let best = cands
            .into_iter()
            .filter(|c| Some(c.3) == top)
            .map(|(x, m, n, _)| (x, m, n))
            .min_by(|a, b| a.0.total_cmp(&b.0));
        match best {
            Some((x, m, n)) => {
                combined += x / 8.0;
                t1_costs[pos] = r3(x);
                best_pos.push(json!({ "position": pos + 1, "run": n, "occupancy_s": r3(x),
                                      "measured": m }));
            }
            None => combined_ok = false,
        }
    }
    let combined_rate = (combined_ok && combined > 0.0).then(|| 3600.0 * t / combined);
    // The best single configuration for the real mix: complete runs first, then by rate.
    let best_run = runs
        .iter()
        .zip(&run_rows)
        .filter(|(r, _)| r.set == p.mid_set || runs.iter().all(|q| q.set == r.set))
        .filter_map(|(r, row)| {
            row["mix_worlds_per_hour"].as_f64().map(|x| {
                (
                    r.name.clone(),
                    x,
                    row["complete"].as_bool().unwrap_or(false),
                )
            })
        })
        .max_by(|a, b| a.2.cmp(&b.2).then(a.1.total_cmp(&b.1)));
    let golden = (p.golden_decisions * p.golden_worlds) as f64;
    let project = |rate: Option<f64>| {
        rate.map(|x| {
            let hours = golden / x;
            json!({ "hours": r3(hours), "dollars": (p.dph > 0.0).then(|| r3(hours * p.dph)) })
        })
    };
    let best_rate = best_run.as_ref().map(|b| b.1);
    json!({
        "ok": mismatches.is_empty(),
        "identity": { "worlds_compared": compared, "mismatches": mismatches },
        "machine": { "threads": p.threads, "dph": p.dph, "cpu_model": p.cpu_model, "mem_gb": p.mem_gb },
        "position_nodes": node_cost.iter().enumerate().map(|(i, x)| json!({
            "position": i + 1, "mean_nodes": x.map(|(n, _)| n.round()), "measured": x.map(|(_, m)| m) })).collect::<Vec<_>>(),
        "runs": run_rows,
        "best_config": best_run.as_ref().map(|(n, x, complete)| json!({
            "name": n, "mix_worlds_per_hour": x.round(), "complete": complete,
            "dollars_per_1000_worlds": (p.dph > 0.0).then(|| r3(1000.0 * p.dph / x)) })),
        "best_per_position": best_pos,
        "combined": combined_rate.map(|x| json!({
            "mix_worlds_per_hour": x.round(),
            "dollars_per_1000_worlds": (p.dph > 0.0).then(|| r3(1000.0 * p.dph / x)) })),
        "suggested_auto_params": combined_ok.then(|| json!({ "t1_costs_s": t1_costs })),
        "golden": { "decisions": p.golden_decisions, "worlds": p.golden_worlds,
                    "best_config": project(best_rate), "combined": project(combined_rate) },
    })
}

fn fmt_rate(v: &Value) -> String {
    v.as_f64().map_or("-".to_string(), |x| format!("{:.0}", x))
}

/// The plain-English summary of an [`analyse`] result.
pub fn summary_text(a: &Value, p: &AnalyseParams) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "Trick-1 bench on {} threads{}{}{}.\n",
        p.threads,
        if p.cpu_model.is_empty() {
            String::new()
        } else {
            format!(" ({})", p.cpu_model)
        },
        if p.mem_gb > 0.0 {
            format!(", {:.0} GB", p.mem_gb)
        } else {
            String::new()
        },
        if p.dph > 0.0 {
            format!(", ${:.3}/hour", p.dph)
        } else {
            ", price unknown".to_string()
        }
    ));
    let id = &a["identity"];
    if a["ok"].as_bool() == Some(true) {
        s.push_str(&format!(
            "Check: every world solved by more than one configuration had byte-identical values \
             ({} comparisons).\n\n",
            id["worlds_compared"]
        ));
    } else {
        s.push_str(&format!(
            "FAILURE: configurations disagree on {} world(s): this is a bug (see results.json, \
             identity.mismatches).\n\n",
            id["mismatches"].as_array().map_or(0, Vec::len)
        ));
    }
    s.push_str(
        "Rates are worlds per hour once the machine is kept full, for the real mix of positions \
         (equal shares of positions 1-8). Positions a run did not solve are scaled from its \
         own speed per solver node and each position's node count.\n\n",
    );
    s.push_str(&format!(
        "{:<14} {:>6} {:>10} {:>9} {:>8} {:>9} {:>7}  {}\n",
        "run", "wall s", "worlds/h", "$/1000 w", "CPU use", "peak MB", "table", "note"
    ));
    for r in a["runs"].as_array().into_iter().flatten() {
        let note = match (r["stopped"].as_str(), r["complete"].as_bool()) {
            (Some("ranking"), _) => "stopped: cannot beat the best (rate is an upper bound)",
            (Some("cap"), _) => "stopped at its time limit (rate is an upper bound)",
            (_, Some(false)) => "incomplete",
            _ => "",
        };
        s.push_str(&format!(
            "{:<14} {:>6.0} {:>10} {:>9} {:>7}% {:>9} {:>6}%  {}\n",
            r["name"].as_str().unwrap_or("?"),
            r["wall_s"].as_f64().unwrap_or(0.0),
            fmt_rate(&r["mix_worlds_per_hour"]),
            r["dollars_per_1000_worlds"]
                .as_f64()
                .map_or("-".into(), |x| format!("{x:.2}")),
            r["cpu_util_pct"]
                .as_f64()
                .map_or("-".into(), |x| format!("{x:.0}")),
            r["peak_rss_mb"]
                .as_u64()
                .map_or("-".into(), |x| x.to_string()),
            r["mean_table_fill"]
                .as_f64()
                .map_or("-".into(), |x| format!("{:.0}", 100.0 * x)),
            note
        ));
    }
    s.push_str("\n(\"table\" = how full the transposition table [the solver's memory of solved positions] was at the end, on average.)\n");
    for k in a["skipped"].as_array().into_iter().flatten() {
        s.push_str(&format!(
            "Not run: {} ({}).\n",
            k["run"].as_str().unwrap_or("?"),
            k["why"].as_str().unwrap_or("?")
        ));
    }
    s.push('\n');
    if let Some(b) = a["best_config"].as_object() {
        s.push_str(&format!(
            "Best single configuration: {} at {} worlds/hour{}.\n",
            b["name"].as_str().unwrap_or("?"),
            fmt_rate(&b["mix_worlds_per_hour"]),
            b["dollars_per_1000_worlds"]
                .as_f64()
                .map_or(String::new(), |x| format!(", ${x:.2} per 1000 worlds"))
        ));
    }
    if let Some(c) = a["combined"].as_object() {
        s.push_str(&format!(
            "Best configuration per position combined (what the position-aware schedule aims \
             for): {} worlds/hour{}.\n",
            fmt_rate(&c["mix_worlds_per_hour"]),
            c["dollars_per_1000_worlds"]
                .as_f64()
                .map_or(String::new(), |x| format!(", ${x:.2} per 1000 worlds"))
        ));
    }
    s.push_str("Fastest per position (thread-seconds per world):\n");
    for b in a["best_per_position"].as_array().into_iter().flatten() {
        s.push_str(&format!(
            "  position {}: {} ({:.1} thread-s{})\n",
            b["position"],
            b["run"].as_str().unwrap_or("?"),
            b["occupancy_s"].as_f64().unwrap_or(0.0),
            if b["measured"].as_bool() == Some(true) {
                ""
            } else {
                ", scaled by nodes"
            }
        ));
    }
    let g = &a["golden"];
    for (label, key) in [
        ("best single configuration", "best_config"),
        ("per-position combination", "combined"),
    ] {
        if let Some(x) = g[key].as_object() {
            s.push_str(&format!(
                "A yardstick of {} decisions x {} worlds would take {:.1} hours{} with the {}.\n",
                g["decisions"],
                g["worlds"],
                x["hours"].as_f64().unwrap_or(0.0),
                x["dollars"]
                    .as_f64()
                    .map_or(String::new(), |d| format!(" (${d:.2})")),
                label
            ));
        }
    }
    s
}

/// CPU ticks this process has used (`/proc/self/stat` utime + stime, all threads), Linux only.
pub fn proc_cpu_ticks() -> Option<u64> {
    let t = std::fs::read_to_string("/proc/self/stat").ok()?;
    let rest = &t[t.rfind(')')? + 1..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    // f[0] is field 3 (state): utime and stime are fields 14 and 15.
    Some(f.get(11)?.parse::<u64>().ok()? + f.get(12)?.parse::<u64>().ok()?)
}

/// This process's peak resident memory in MB (`VmHWM`), Linux only.
pub fn proc_peak_rss_mb() -> Option<u64> {
    let t = std::fs::read_to_string("/proc/self/status").ok()?;
    let l = t.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: u64 = l.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sets_parse() {
        assert_eq!(parse_set(MID_SET).expect("mid").len(), 3);
        assert_eq!(
            parse_set("0:1:2").expect("one"),
            vec![SetPart {
                position: 0,
                decisions: 1,
                worlds: 2
            }]
        );
        for bad in ["8:1:1", "0:0:1", "0:1", "a:b:c", "1:1:0"] {
            assert!(parse_set(bad).is_err(), "{bad}");
        }
        let total = |s: &str| {
            parse_set(s)
                .expect("set")
                .iter()
                .map(|p| p.decisions * p.worlds)
                .sum::<u32>()
        };
        assert_eq!(
            (total(LEAD_SET), total(MID_SET), total(LATE_SET)),
            (3, 48, 128)
        );
    }

    fn world(
        id: u64,
        pos: u8,
        w: u64,
        threads: usize,
        wall: f64,
        nodes: u64,
        v: u64,
    ) -> BenchWorld {
        BenchWorld {
            id,
            position: pos,
            world: w,
            solves: 1,
            threads,
            tt_mb: 64,
            started: true,
            finished: true,
            start_s: 0.0,
            wall_s: wall,
            nodes,
            tt_probes: 10,
            tt_hits: 5,
            tt_entries: 1000,
            tt_bytes: 1 << 20,
            tt_replaced: 0,
            values: Some(vec![[v, 0, 0, 0]]),
        }
    }

    fn run(name: &str, set: &str, worlds: Vec<BenchWorld>) -> BenchRun {
        BenchRun {
            name: name.into(),
            set: set.into(),
            schedule: json!({}),
            threads: 8,
            cap_s: 100.0,
            stop_occ_s: None,
            wall_s: 10.0,
            cpu_s: Some(60.0),
            peak_rss_mb: Some(100),
            stopped: None,
            worlds,
        }
    }

    /// Rates, extrapolation by nodes, the best run, and a disagreement flagged as a failure.
    #[test]
    fn analysis() {
        // A run over every position, nodes halving per position.
        let ratio = run(
            "all",
            "0:1:1,1:1:1,2:1:1,3:1:1,4:1:1,5:1:1,6:1:1,7:1:1",
            (0..8u8)
                .map(|p| world(u64::from(p), p, 0, 1, 1.0, 1 << (20 - p), 7))
                .collect(),
        );
        // Two runs on positions 3–6 (same worlds): b twice as efficient as a.
        let mid = |name: &str, threads: usize, wall: f64| {
            run(
                name,
                MID_SET,
                (2..6u8)
                    .map(|p| world(u64::from(p), p, 0, threads, wall, 1 << (20 - p), 7))
                    .collect(),
            )
        };
        let a = mid("a", 4, 2.0);
        let b = mid("b", 2, 2.0);
        let p = AnalyseParams {
            threads: 8,
            dph: 2.0,
            ..AnalyseParams::default()
        };
        let res = analyse(&[ratio.clone(), a.clone(), b.clone()], &p);
        assert_eq!(res["ok"], json!(true), "{res}");
        assert_eq!(res["identity"]["worlds_compared"], json!(8));
        assert_eq!(res["best_config"]["name"], json!("b"));
        let ra = res["runs"][1]["mix_worlds_per_hour"].as_f64().expect("a");
        let rb = res["runs"][2]["mix_worlds_per_hour"].as_f64().expect("b");
        assert!((rb / ra - 2.0).abs() < 1e-6, "{ra} {rb}");
        // Unmeasured positions are scaled by nodes: the run's occupancy per node over the
        // positions it measured, times the position's nodes.
        let occ = &res["runs"][2]["occupancy_by_position_s"];
        let o1 = occ[1]["s"].as_f64().expect("pos 2");
        assert_eq!(occ[1]["measured"], json!(false));
        assert_eq!(occ[2]["measured"], json!(true));
        let per_node = 4.0 * 4.0 / f64::from((1 << 18) + (1 << 17) + (1 << 16) + (1 << 15));
        assert!((o1 - per_node * f64::from(1 << 19)).abs() < 0.01, "{o1}");
        let text = summary_text(&res, &p);
        assert!(text.contains("byte-identical"), "{text}");
        assert!(text.contains("Best single configuration: b"), "{text}");
        // A disagreement fails loudly.
        let mut bad = b.clone();
        bad.worlds[0].values = Some(vec![[8, 0, 0, 0]]);
        let res = analyse(&[ratio, a, bad], &p);
        assert_eq!(res["ok"], json!(false));
        assert!(summary_text(&res, &p).contains("FAILURE"));
    }
}
