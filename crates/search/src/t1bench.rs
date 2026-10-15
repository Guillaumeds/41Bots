//! The trick-1 machine bench (docs/M3-SPEC.md §13.1 item 11): how much exact trick-1 solving a
//! machine does per second, and per dollar, with every thread busy the whole time. Its only job
//! is to rank machines; it never chooses how production spreads its worlds.
//!
//! Two timed windows, each run by `search bench-t1` as its own process (so each has its own peak
//! memory), on `T` threads (the runner's usable threads):
//! - **fill** ([`Mode::Fill`]): exactly `T` worlds at once, one solver thread each, from the first
//!   second to the window's end. Thread `k` solves only yardstick worlds (world label 1) of
//!   position `k mod 8` in the trick, so every machine solves the same mix of positions. The
//!   `j`-th world of a position is decision `j mod D` of its `D` decisions (by id), world
//!   `j / D`; thread `k` takes `j = k/8, k/8 + ⌈T/8⌉, k/8 + 2⌈T/8⌉, …` ([`fill_task`]), so no
//!   two threads take the same world and the supply never runs out. Every table has the same size.
//! - **prod** ([`Mode::Prod`]): the production schedule (`crate::plan`, exactly as
//!   `search worlds --schedule auto` runs it: same order, plan and memory budget) on a queue too
//!   deep to run dry (every decision × `worlds` worlds); it checks that production keeps this
//!   machine full, and gives its raw node rate (parallel solves search extra nodes).
//!
//! At the window's end every solve is stopped; a stopped world's nodes so far still count
//! (the solver reports them; nothing it returns changes).
//!
//! **Recorded** ([`WindowRun`]): every world started (decision, world, position, threads, nodes,
//! finished or not, and a hash of its values when finished); per fill thread its finished worlds
//! and nodes; every second a [`Sample`]: threads working (`busy`: per running world, its solver
//! threads still searching, from the solver's own counter, so a parallel solve's idle tail shows,
//! or else the one thread preparing it or between its solves), threads held by running worlds,
//! nodes so far, worlds finished, process CPU time; the window's CPU time and peak memory. The
//! threads of a window are all started before its clock starts, and released together.
//!
//! **Guard** ([`Guard`]): from `ramp_s` (20 s) into the window, it stops the window at once when
//! the process's CPU over the last `cpu_interval_s` (30 s; counted from half the ramp on) is
//! below `min_cpu_pct` of the usable cores, or when fewer than `min_busy_pct` (95%) of the `T`
//! threads have been working for more than `busy_streak_s` (10 s) in a row (the tolerance lets the
//! production plan leave a few threads unassigned, e.g. 9 worlds × 42 threads on 384). The reason is recorded and the runner ends the
//! chunk FAILED (`crates/runner/src/t1bench.rs`).
//!
//! **Analysis** ([`analyse`], [`summary_text`]): per window nodes per second for the machine and
//! per thread, CPU use, mean share of threads working (and share of samples with all of them),
//! peak memory, dollars per
//! 10^12 nodes; per position the fill rate per thread; the machine score (the fill rates
//! weighted by each position's share of real trick-1 work, [`DEFAULT_POSITION_NODES`]); a
//! projection for a golden set; and the identity check: a world finished in both windows must
//! have byte-identical values.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use equity::MoneyChance;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use solver::{ParProgress, SolveMem};

use crate::measure::{DecisionSpec, batch_key, measure_seed};
use crate::pimc::{
    Candidate, Ctx, SearchConfig, WorldSolve, candidates, exact_world_row, solves_per_world,
};
use crate::plan::{AutoParams, ExecProgress, PlanTask, plan_auto, run_planned};
use crate::worlds::Worlds;

/// Positions in a trick.
pub const POSITIONS: usize = 8;

/// Mean solver nodes [positions the exact solver examines] of one trick-1 world at each position
/// (leader first), from the evidence so far: the leader's worlds in the M2 trick-1 run on the
/// 384-core EPYC 1.2, 4.5, 28.9 and 4.3 billion (docs/M2-SPIKE.md; mean about 10 billion);
/// position 4 about 1 billion (about 200 s on one thread of the 4-vCPU Xeon, 2026-10-14);
/// positions 6, 7 and 8 about 16, 20 and 25 s there, about 80, 100 and 125 million at position
/// 4's 5 million nodes a second; positions 2–3 and 5 geometric steps between their neighbours.
/// They weight the positions in the machine score; the yardstick run will measure them
/// (`position_nodes` in the chunk replaces them).
pub const DEFAULT_POSITION_NODES: [f64; POSITIONS] =
    [1.0e10, 4.6e9, 2.2e9, 1.0e9, 2.8e8, 8.0e7, 1.0e8, 1.25e8];

/// Which window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// One world per thread, every thread busy, positions spread evenly.
    Fill,
    /// The production schedule on a deep queue.
    Prod,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Fill => "fill",
            Mode::Prod => "prod",
        }
    }

    pub fn parse(s: &str) -> Result<Mode, String> {
        match s {
            "fill" => Ok(Mode::Fill),
            "prod" => Ok(Mode::Prod),
            _ => Err(format!("unknown bench mode {s:?} (fill or prod)")),
        }
    }
}

/// When a window stops itself (see the module text).
#[derive(Clone, Debug, PartialEq)]
pub struct Guard {
    /// Seconds from the window's start before any check.
    pub ramp_s: f64,
    /// The lowest acceptable CPU use, percent of the usable cores (0 = no CPU check).
    pub min_cpu_pct: f64,
    /// The CPU is measured over this many seconds back (the heartbeat interval).
    pub cpu_interval_s: f64,
    /// Fewer than `min_busy_pct` of the `T` threads working for more than this many seconds in a
    /// row stops the window.
    pub busy_streak_s: f64,
    /// The share of the `T` threads, in percent, that must be working (rounded up to whole threads).
    pub min_busy_pct: f64,
    /// The cores the process may use (the CPU percentage's denominator).
    pub usable_cores: f64,
}

impl Default for Guard {
    fn default() -> Self {
        Guard {
            ramp_s: 20.0,
            min_cpu_pct: 80.0,
            cpu_interval_s: 30.0,
            busy_streak_s: 10.0,
            min_busy_pct: 95.0,
            usable_cores: std::thread::available_parallelism().map_or(1.0, |n| n.get() as f64),
        }
    }
}

/// The guard's memory between samples.
#[derive(Clone, Debug, Default)]
pub struct GuardState {
    /// When the current run of samples with too few threads working began.
    low_since: Option<f64>,
}

impl Guard {
    /// The reason to stop now, if any, given every sample so far (in time order, the first at 0 s
    /// with the CPU baseline). `t` is the window's thread count.
    pub fn check(
        &self,
        st: &mut GuardState,
        samples: &[Sample],
        mode: Mode,
        t: usize,
    ) -> Option<String> {
        let s = samples.last()?;
        if s.t < self.ramp_s {
            st.low_since = None;
            return None;
        }
        let need = ((self.min_busy_pct.clamp(0.0, 100.0) / 100.0) * t as f64).ceil() as usize;
        if (s.busy as usize) < need.clamp(1, t.max(1)) {
            let since = *st.low_since.get_or_insert(s.t);
            if s.t - since > self.busy_streak_s {
                return Some(format!(
                    "idle threads: only {} of {t} working for over {:.0} s in {} at {:.0} s",
                    s.busy,
                    self.busy_streak_s,
                    mode.name(),
                    s.t
                ));
            }
        } else {
            st.low_since = None;
        }
        if self.min_cpu_pct > 0.0 && self.usable_cores > 0.0 {
            let from = (s.t - self.cpu_interval_s).max(self.ramp_s / 2.0);
            // The latest sample at or before `from` (the first sample is at 0 s).
            let base = samples.iter().rev().find(|x| x.t <= from + 1e-9)?;
            let dt = s.t - base.t;
            if dt > 0.0 {
                let pct = 100.0 * (s.cpu_s - base.cpu_s) / (dt * self.usable_cores);
                if pct < self.min_cpu_pct {
                    return Some(format!(
                        "low CPU: {pct:.0}% in {} at {:.0} s",
                        mode.name(),
                        s.t
                    ));
                }
            }
        }
        None
    }
}

/// What one window does.
#[derive(Clone, Debug)]
pub struct WindowParams {
    pub mode: Mode,
    /// The window's length in seconds.
    pub window_s: f64,
    /// `T`: worlds at once (fill), or threads for the schedule (prod).
    pub threads: usize,
    /// The yardstick's seed and world label (worlds are the yardstick's own).
    pub seed: u64,
    pub label: u64,
    /// Fill: every table's size in MB.
    pub tt_mb: usize,
    /// Prod: the memory for all tables at their peak, in MB, and the plan's settings.
    pub budget_mb: u64,
    pub auto: AutoParams,
    /// Prod: worlds per decision in the queue.
    pub worlds: u64,
    /// Milliseconds between samples (1000; tests use less).
    pub sample_ms: u64,
    /// Stop early (see [`Guard`]); `None`: run to the end.
    pub guard: Option<Guard>,
}

impl WindowParams {
    pub fn new(mode: Mode, window_s: f64, threads: usize) -> Self {
        WindowParams {
            mode,
            window_s,
            threads,
            seed: 1,
            label: 1,
            tt_mb: 512,
            budget_mb: 16_384,
            auto: AutoParams::default(),
            worlds: 800,
            sample_ms: 1000,
            guard: None,
        }
    }
}

/// One sample of a window.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    /// Seconds into the window.
    pub t: f64,
    /// Threads working now (see the module text).
    pub busy: u64,
    /// Threads held by the worlds running now (fill: one per thread; prod: the plan's).
    pub held: u64,
    /// Nodes so far (live counters, which lag by up to 4096 per thread).
    pub nodes: u64,
    /// Worlds finished so far.
    pub finished: u64,
    /// Process CPU seconds since the window's start.
    pub cpu_s: f64,
}

/// One world of a window.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldRec {
    /// Decision id, its position in the trick (0 = leader), and the world.
    pub id: u64,
    pub position: u8,
    pub world: u64,
    /// Solver threads and table size (MB).
    pub threads: usize,
    pub tt_mb: usize,
    /// The fill thread that solved it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread: Option<usize>,
    /// Start (seconds into the window) and wall time.
    pub start_s: f64,
    pub wall_s: f64,
    pub finished: bool,
    /// Solver nodes (so far, when stopped).
    pub nodes: u64,
    /// The first 16 hex digits of the SHA-256 of its values (every candidate's four u64,
    /// little-endian), when finished.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
}

/// One fill thread.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThreadRow {
    pub k: usize,
    pub position: u8,
    pub worlds_finished: u64,
    pub nodes_finished: u64,
    /// Nodes of the world stopped at the window's end.
    pub nodes_partial: u64,
}

/// One window's record (`search bench-t1 --json`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WindowRun {
    pub mode: Mode,
    pub window_s: f64,
    /// From the window's start until every solve had stopped.
    pub wall_s: f64,
    pub threads: usize,
    #[serde(default)]
    pub usable_cores: f64,
    /// Fill: every table's size.
    #[serde(default)]
    pub tt_mb: Option<usize>,
    /// Prod: the plan (per class of world: threads and table) and worlds per decision.
    #[serde(default)]
    pub plan: Option<Value>,
    #[serde(default)]
    pub worlds_per_decision: Option<u64>,
    /// Why the window stopped before its end (the guard, a request), if it did.
    #[serde(default)]
    pub stopped: Option<String>,
    #[serde(default)]
    pub cpu_s: Option<f64>,
    #[serde(default)]
    pub peak_rss_mb: Option<u64>,
    pub samples: Vec<Sample>,
    #[serde(default)]
    pub thread_rows: Vec<ThreadRow>,
    pub worlds: Vec<WorldRec>,
}

impl WindowRun {
    /// Nodes of every world started (finished ones, and stopped ones so far).
    pub fn nodes(&self) -> u64 {
        self.worlds.iter().map(|w| w.nodes).sum()
    }
}

/// The `n`-th world (from 0) thread `k` of `t` takes in the fill window, at its position
/// (`k mod 8`) with `d` decisions there: (index among them, world).
pub fn fill_task(k: usize, n: u64, t: usize, d: usize) -> (usize, u64) {
    let stride = t.max(1).div_ceil(POSITIONS) as u64;
    let j = (k / POSITIONS) as u64 + n * stride;
    let d = d.max(1) as u64;
    ((j % d) as usize, j / d)
}

/// The hash recorded for a finished world's values.
pub fn values_hash(v: &[MoneyChance]) -> String {
    let mut b = Vec::with_capacity(v.len() * 32);
    for m in v {
        for x in m.0 {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    equity::sha256::sha256_hex(&b)[..16].to_string()
}

struct Prep {
    id: u64,
    trick: u8,
    position: u8,
    view: engine::view::SeatView,
    cands: Vec<Candidate>,
    dseed: u64,
}

/// What the monitor reads at each sample: (busy, held, nodes, finished).
type Probe<'a> = dyn Fn() -> (u64, u64, u64, u64) + Sync + 'a;

/// Threads working on one world: its solver threads still searching, or the one thread that is
/// preparing it (drawing the world, setting up the position) or between its solves.
fn working(p: &ParProgress, in_world: &AtomicBool) -> u64 {
    match p.active.load(Ordering::Relaxed) {
        0 => u64::from(in_world.load(Ordering::Relaxed)),
        a => a,
    }
}

/// A start gate: threads wait at it until it opens (so that every thread of a window starts at
/// once, and the window's clock starts when they do).
#[derive(Default)]
struct Gate {
    open: Mutex<bool>,
    cv: std::sync::Condvar,
}

impl Gate {
    fn wait(&self) {
        let mut g = self.open.lock().unwrap_or_else(|e| e.into_inner());
        while !*g {
            g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn open(&self) {
        *self.open.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.cv.notify_all();
    }
}

/// Samples every `sample_ms`, runs the guard and the caller's check, and stops the window at its
/// end or on a reason: sets `stop` (no new world starts), then `abort_all` (running solves stop).
/// It sleeps until each sample is due (woken early by an unpark once `done` is set). Returns the
/// samples and the reason, if the window stopped early.
#[allow(clippy::too_many_arguments)]
fn monitor(
    p: &WindowParams,
    t0: Instant,
    cpu0: Option<f64>,
    done: &AtomicBool,
    stop: &AtomicBool,
    probe: &Probe<'_>,
    abort_all: &(dyn Fn() + Sync),
    on_sample: &mut (dyn FnMut(&Sample) -> Option<String> + Send),
) -> (Vec<Sample>, Option<String>) {
    let t = p.threads.max(1);
    let window = Duration::from_secs_f64(p.window_s.max(0.0));
    let step = Duration::from_millis(p.sample_ms.max(10));
    let take = |at: f64| {
        let (busy, held, nodes, finished) = probe();
        Sample {
            t: at,
            busy,
            held,
            nodes,
            finished,
            cpu_s: match (cpu0, proc_cpu_seconds()) {
                (Some(a), Some(b)) => (b - a).max(0.0),
                _ => 0.0,
            },
        }
    };
    let mut samples = vec![Sample {
        t: 0.0,
        busy: 0,
        held: 0,
        nodes: 0,
        finished: 0,
        cpu_s: 0.0,
    }];
    let mut gs = GuardState::default();
    let mut reason = None;
    let mut next = step;
    loop {
        let due = next.min(window);
        loop {
            let now = t0.elapsed();
            if now >= due || done.load(Ordering::SeqCst) {
                break;
            }
            std::thread::park_timeout(due - now);
        }
        if done.load(Ordering::SeqCst) {
            let at = t0.elapsed().as_secs_f64();
            samples.push(take(at));
            reason = Some(format!(
                "the work ran out at {at:.0} s, before the window's end"
            ));
            break;
        }
        let s = take(t0.elapsed().as_secs_f64());
        samples.push(s);
        if let Some(r) = on_sample(&s) {
            reason = Some(r);
            break;
        }
        if due >= window {
            break;
        }
        if let Some(g) = &p.guard
            && let Some(r) = g.check(&mut gs, &samples, p.mode, t)
        {
            reason = Some(r);
            break;
        }
        // The next tick after now: ticks missed while this thread waited for a core are
        // skipped, not taken back to back.
        let ticks = t0.elapsed().as_nanos() / step.as_nanos().max(1) + 1;
        next = step * u32::try_from(ticks).unwrap_or(u32::MAX);
    }
    stop.store(true, Ordering::SeqCst);
    abort_all();
    (samples, reason)
}

/// Run one window over decisions `ds` (their yardstick worlds: seed `p.seed`, label `p.label`).
/// `on_sample` sees every sample (e.g. to write a progress file) and may stop the window by
/// returning a reason.
pub fn run_window(
    ds: &[DecisionSpec],
    ctx: &Ctx,
    p: &WindowParams,
    on_sample: &mut (dyn FnMut(&Sample) -> Option<String> + Send),
) -> Result<WindowRun, String> {
    if ds.is_empty() {
        return Err("bench: no decisions".into());
    }
    let mut preps: Vec<Prep> = ds
        .iter()
        .map(|d| {
            let (_, view, legal) = d.replay();
            let (_, cands) = candidates(&view, &legal);
            Prep {
                id: d.id,
                trick: d.trick,
                position: d.position,
                view,
                cands,
                dseed: measure_seed(p.seed, p.label, d.id),
            }
        })
        .collect();
    match p.mode {
        // By id, so a position's decisions are taken in id order.
        Mode::Fill => preps.sort_by_key(|q| q.id),
        // The production order (`measure::world_rows_sched`).
        Mode::Prod => {
            let key: BTreeMap<u64, _> = ds.iter().map(|d| (d.id, d)).collect();
            preps.sort_by_key(|q| batch_key(key[&q.id], q.cands.len()));
        }
    }
    let worlds: Vec<Box<dyn Worlds + '_>> = preps
        .iter()
        .map(|q| ctx.source.prepare(&q.view, ctx.rules))
        .collect();
    let mut run = match p.mode {
        Mode::Fill => fill(&preps, &worlds, ctx, p, on_sample)?,
        Mode::Prod => prod(&preps, &worlds, ctx, p, on_sample)?,
    };
    run.peak_rss_mb = proc_peak_rss_mb();
    Ok(run)
}

fn cpu_since(cpu0: Option<f64>) -> Option<f64> {
    match (cpu0, proc_cpu_seconds()) {
        (Some(a), Some(b)) => Some(b - a),
        _ => None,
    }
}

fn fill(
    preps: &[Prep],
    worlds: &[Box<dyn Worlds + '_>],
    ctx: &Ctx,
    p: &WindowParams,
    on_sample: &mut (dyn FnMut(&Sample) -> Option<String> + Send),
) -> Result<WindowRun, String> {
    let t = p.threads.max(1);
    let mut at: [Vec<usize>; POSITIONS] = Default::default();
    for (i, q) in preps.iter().enumerate() {
        at[usize::from(q.position.min(7))].push(i);
    }
    for (pos, list) in at.iter().enumerate().take(t.min(POSITIONS)) {
        if list.is_empty() {
            return Err(format!(
                "bench: the decision set has no decision at position {} (the fill window needs \
                 every position its threads take)",
                pos + 1
            ));
        }
    }
    let pr: Vec<ParProgress> = (0..t).map(|_| ParProgress::default()).collect();
    let in_world: Vec<AtomicBool> = (0..t).map(|_| AtomicBool::new(false)).collect();
    let fin: Vec<AtomicU64> = (0..t).map(|_| AtomicU64::new(0)).collect();
    let stop = AtomicBool::new(false);
    let done = AtomicBool::new(false);
    let gate = Gate::default();
    // The window's clock and CPU baseline, set just before the gate opens.
    let clock: OnceLock<Instant> = OnceLock::new();
    let cfg = SearchConfig {
        threads: 1,
        solve_threads: 1,
        tt_mb: p.tt_mb.max(1),
        ..SearchConfig::default()
    };
    let probe = || {
        let mut x = (0, t as u64, 0, 0);
        for k in 0..t {
            x.0 += working(&pr[k], &in_world[k]);
            x.2 += pr[k].nodes.load(Ordering::Relaxed);
            x.3 += fin[k].load(Ordering::Relaxed);
        }
        x
    };
    let abort_all = || {
        for f in &pr {
            f.abort.store(true, Ordering::SeqCst);
        }
    };
    let one = |k: usize| {
        gate.wait();
        let t0 = *clock.get().expect("the clock starts before the gate opens");
        let pos = k % POSITIONS;
        let list = &at[pos];
        let mut mem = SolveMem::new();
        let mut row = ThreadRow {
            k,
            position: pos as u8,
            worlds_finished: 0,
            nodes_finished: 0,
            nodes_partial: 0,
        };
        let mut recs = Vec::new();
        let mut n = 0u64;
        while !stop.load(Ordering::SeqCst) {
            let (di, w) = fill_task(k, n, t, list.len());
            n += 1;
            let qi = list[di];
            let q = &preps[qi];
            let start = t0.elapsed().as_secs_f64();
            in_world[k].store(true, Ordering::Relaxed);
            let mut solve = WorldSolve::default();
            let v = exact_world_row(
                &q.view,
                &*worlds[qi],
                &q.cands,
                w,
                q.dseed,
                ctx,
                &cfg,
                Some(&pr[k]),
                &mut mem,
                &mut solve,
            );
            in_world[k].store(false, Ordering::Relaxed);
            let nodes = solve.stats.nodes;
            if v.is_some() {
                row.worlds_finished += 1;
                row.nodes_finished += nodes;
                fin[k].fetch_add(1, Ordering::Relaxed);
            } else {
                row.nodes_partial += nodes;
            }
            recs.push(WorldRec {
                id: q.id,
                position: q.position,
                world: w,
                threads: 1,
                tt_mb: cfg.tt_mb,
                thread: Some(k),
                start_s: start,
                wall_s: t0.elapsed().as_secs_f64() - start,
                finished: v.is_some(),
                nodes,
                hash: v.as_deref().map(values_hash),
            });
        }
        (row, recs)
    };
    let (out, samples, reason, t0, cpu0) = std::thread::scope(|s| {
        let one = &one;
        // Every thread first (they wait at the gate, using no CPU), so that none starts late.
        let mut hs = Vec::with_capacity(t);
        let mut failed = None;
        for k in 0..t {
            match std::thread::Builder::new().spawn_scoped(s, move || one(k)) {
                Ok(h) => hs.push(h),
                Err(e) => {
                    failed = Some(format!("bench: cannot start thread {k} of {t}: {e}"));
                    break;
                }
            }
        }
        let cpu0 = proc_cpu_seconds();
        let t0 = Instant::now();
        let _ = clock.set(t0);
        let probe = &probe;
        let abort_all = &abort_all;
        let (stop, done) = (&stop, &done);
        let mon = if failed.is_none() {
            let m = std::thread::Builder::new().spawn_scoped(s, move || {
                monitor(p, t0, cpu0, done, stop, probe, abort_all, on_sample)
            });
            m.map_err(|e| failed = Some(format!("bench: cannot start the monitor thread: {e}")))
                .ok()
        } else {
            None
        };
        // Whatever failed to start, the threads already waiting are released, to stop at once.
        if failed.is_some() {
            stop.store(true, Ordering::SeqCst);
        }
        gate.open();
        let mut out = Vec::with_capacity(t);
        let mut panic = None;
        for h in hs {
            match h.join() {
                Ok(r) => out.push(r),
                Err(e) => {
                    // Stop the others, then raise it once they have stopped.
                    stop.store(true, Ordering::SeqCst);
                    abort_all();
                    panic.get_or_insert(e);
                }
            }
        }
        done.store(true, Ordering::SeqCst);
        let (samples, reason) = match mon {
            Some(m) => {
                m.thread().unpark();
                m.join().expect("bench monitor")
            }
            None => (Vec::new(), failed),
        };
        if let Some(e) = panic {
            std::panic::resume_unwind(e);
        }
        (out, samples, reason, t0, cpu0)
    });
    let wall_s = t0.elapsed().as_secs_f64();
    let cpu_s = cpu_since(cpu0);
    let (thread_rows, recs): (Vec<ThreadRow>, Vec<Vec<WorldRec>>) = out.into_iter().unzip();
    Ok(WindowRun {
        mode: Mode::Fill,
        window_s: p.window_s,
        wall_s,
        threads: t,
        usable_cores: p.guard.as_ref().map_or(0.0, |g| g.usable_cores),
        tt_mb: Some(cfg.tt_mb),
        plan: None,
        worlds_per_decision: None,
        stopped: reason,
        cpu_s,
        peak_rss_mb: None,
        samples,
        thread_rows,
        worlds: recs.into_iter().flatten().collect(),
    })
}

fn prod(
    preps: &[Prep],
    worlds: &[Box<dyn Worlds + '_>],
    ctx: &Ctx,
    p: &WindowParams,
    on_sample: &mut (dyn FnMut(&Sample) -> Option<String> + Send),
) -> Result<WindowRun, String> {
    let t = p.threads.max(1);
    let nw = p.worlds.max(1) as usize;
    // Exactly `measure::world_rows_sched`'s tasks, plan and order.
    let tasks: Vec<PlanTask> = preps
        .iter()
        .flat_map(|q| {
            let x = PlanTask {
                trick: q.trick,
                position: q.position,
                solves: solves_per_world(&q.cands),
            };
            std::iter::repeat_n(x, nw)
        })
        .collect();
    let budget = p.budget_mb.saturating_mul(1 << 20);
    let planned = plan_auto(&tasks, t, budget, &p.auto);
    let n = tasks.len();
    let flags: Vec<ParProgress> = (0..n).map(|_| ParProgress::default()).collect();
    let in_world: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();
    let exec = ExecProgress::default();
    let finished = AtomicU64::new(0);
    let stop = AtomicBool::new(false);
    let done = AtomicBool::new(false);
    let probe = || {
        let mut x = (0, exec.busy_threads.load(Ordering::Relaxed), 0, 0);
        for (f, w) in flags.iter().zip(&in_world) {
            x.0 += working(f, w);
            x.2 += f.nodes.load(Ordering::Relaxed);
        }
        x.3 = finished.load(Ordering::Relaxed);
        x
    };
    let abort_all = || {
        for f in &flags {
            f.abort.store(true, Ordering::SeqCst);
        }
    };
    let recs: Mutex<Vec<WorldRec>> = Mutex::new(Vec::new());
    let cpu0 = proc_cpu_seconds();
    let t0 = Instant::now();
    let work = |k: usize, plan: crate::plan::WorldPlan, mem: &mut SolveMem| {
        if stop.load(Ordering::SeqCst) {
            return None;
        }
        let (j, i) = (k / nw, k % nw);
        let q = &preps[j];
        let cfg = SearchConfig {
            threads: 1,
            solve_threads: plan.threads,
            tt_mb: plan.tt_mb,
            ..SearchConfig::default()
        };
        let start = t0.elapsed().as_secs_f64();
        in_world[k].store(true, Ordering::Relaxed);
        let mut solve = WorldSolve::default();
        let v = exact_world_row(
            &q.view,
            &*worlds[j],
            &q.cands,
            i as u64,
            q.dseed,
            ctx,
            &cfg,
            Some(&flags[k]),
            mem,
            &mut solve,
        );
        in_world[k].store(false, Ordering::Relaxed);
        if v.is_some() {
            finished.fetch_add(1, Ordering::Relaxed);
        }
        Some(WorldRec {
            id: q.id,
            position: q.position,
            world: i as u64,
            threads: plan.threads,
            tt_mb: plan.tt_mb,
            thread: None,
            start_s: start,
            wall_s: t0.elapsed().as_secs_f64() - start,
            finished: v.is_some(),
            nodes: solve.stats.nodes,
            hash: v.as_deref().map(values_hash),
        })
    };
    let (r, samples, reason) = std::thread::scope(|s| {
        let (probe, abort_all, stop, done) = (&probe, &abort_all, &stop, &done);
        let mon = s.spawn(move || monitor(p, t0, cpu0, done, stop, probe, abort_all, on_sample));
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_planned(
                &planned.order,
                &planned.plans,
                t,
                budget,
                stop,
                Some(&exec),
                work,
                |_, rec, _| {
                    if let Some(rec) = rec {
                        recs.lock().unwrap_or_else(|e| e.into_inner()).push(rec);
                    }
                    Ok(())
                },
            )
        }));
        done.store(true, Ordering::SeqCst);
        if r.is_err() {
            stop.store(true, Ordering::SeqCst);
            abort_all();
        }
        mon.thread().unpark();
        let (samples, reason) = mon.join().expect("bench monitor");
        match r {
            Ok(r) => (r, samples, reason),
            Err(e) => std::panic::resume_unwind(e),
        }
    });
    r?;
    let wall_s = t0.elapsed().as_secs_f64();
    let cpu_s = cpu_since(cpu0);
    let mut worlds_out = recs.into_inner().unwrap_or_else(|e| e.into_inner());
    worlds_out.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
    Ok(WindowRun {
        mode: Mode::Prod,
        window_s: p.window_s,
        wall_s,
        threads: t,
        usable_cores: p.guard.as_ref().map_or(0.0, |g| g.usable_cores),
        tt_mb: None,
        plan: Some(planned.record),
        worlds_per_decision: Some(nw as u64),
        stopped: reason,
        cpu_s,
        peak_rss_mb: None,
        samples,
        thread_rows: Vec::new(),
        worlds: worlds_out,
    })
}

// ---------------------------------------------------------------- analysis

/// Inputs of [`analyse`] besides the windows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AnalyseParams {
    /// `T`, and the cores the machine may use.
    pub threads: usize,
    pub usable_cores: f64,
    /// Dollars per hour of the machine (0 = unknown).
    pub dph: f64,
    /// The golden set projected: decisions (equal shares of the 8 positions) × worlds each.
    pub golden_decisions: u64,
    pub golden_worlds: u64,
    /// Mean nodes of a world per position, and where they come from.
    pub position_nodes: [f64; POSITIONS],
    pub position_nodes_source: String,
    /// The machine, for the text.
    pub cpu_model: String,
    pub mem_gb: f64,
    /// Samples before this many seconds into a window are left out of the busy share.
    pub busy_skip_s: f64,
}

impl Default for AnalyseParams {
    fn default() -> Self {
        AnalyseParams {
            threads: 1,
            usable_cores: 1.0,
            dph: 0.0,
            golden_decisions: 128,
            golden_worlds: 200,
            position_nodes: DEFAULT_POSITION_NODES,
            position_nodes_source: "default".into(),
            cpu_model: String::new(),
            mem_gb: 0.0,
            busy_skip_s: 5.0,
        }
    }
}

fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn r1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// Dollars per 10^12 nodes at `rate` nodes per second.
fn per_tera(dph: f64, rate: f64) -> Option<f64> {
    (dph > 0.0 && rate > 0.0).then(|| r3(dph / 3600.0 / rate * 1e12))
}

/// The common numbers of a window.
fn window_numbers(w: &WindowRun, p: &AnalyseParams) -> Value {
    let t = w.threads.max(1) as f64;
    let nodes = w.nodes();
    let rate = if w.wall_s > 0.0 {
        nodes as f64 / w.wall_s
    } else {
        0.0
    };
    let cores = if w.usable_cores > 0.0 {
        w.usable_cores
    } else {
        p.usable_cores
    };
    let after: Vec<&Sample> = w.samples.iter().filter(|s| s.t > p.busy_skip_s).collect();
    let share = |f: &dyn Fn(&Sample) -> f64| {
        (!after.is_empty())
            .then(|| r1(100.0 * after.iter().map(|s| f(s)).sum::<f64>() / after.len() as f64))
    };
    json!({
        "window_s": w.window_s,
        "wall_s": r3(w.wall_s),
        "threads": w.threads,
        "stopped": w.stopped,
        "worlds_started": w.worlds.len(),
        "worlds_finished": w.worlds.iter().filter(|x| x.finished).count(),
        "nodes": nodes,
        "nodes_per_s": rate.round(),
        "nodes_per_s_per_thread": (rate / t).round(),
        "dollars_per_tera_nodes": per_tera(p.dph, rate),
        "cpu_pct": w.cpu_s.filter(|_| w.wall_s > 0.0 && cores > 0.0)
            .map(|c| r1(100.0 * c / (w.wall_s * cores))),
        "cpu_s": w.cpu_s.map(r1),
        // Mean share of the threads working, and share of samples with all of them.
        "busy_pct": share(&|s| s.busy as f64 / t),
        "all_busy_pct": share(&|s| if s.busy as usize >= w.threads { 1.0 } else { 0.0 }),
        "min_busy": after.iter().map(|s| s.busy).min(),
        "held_pct": share(&|s| s.held as f64 / t),
        "samples_counted": after.len(),
        "peak_rss_mb": w.peak_rss_mb,
        "tt_mb": w.tt_mb,
        "plan": w.plan,
        "worlds_per_decision": w.worlds_per_decision,
    })
}

/// The fill window's per-position rates: (threads, nodes, nodes per second per thread).
fn fill_positions(w: &WindowRun) -> [(usize, u64, f64); POSITIONS] {
    let mut out = [(0usize, 0u64, 0.0f64); POSITIONS];
    for r in &w.thread_rows {
        let e = &mut out[usize::from(r.position.min(7))];
        e.0 += 1;
        e.1 += r.nodes_finished + r.nodes_partial;
    }
    for e in &mut out {
        if e.0 > 0 && w.wall_s > 0.0 {
            e.2 = e.1 as f64 / (e.0 as f64 * w.wall_s);
        }
    }
    out
}

/// The machine score: nodes per second for the real trick-1 mix, from the fill window's
/// per-thread rate at each position, each position weighted by its share of the real work
/// (`position_nodes`): the time the mix takes is the sum over positions of its nodes there over
/// the machine's speed there, so the score is `T / Σ (share / rate per thread)`. Positions the
/// window did not run (fewer than 8 threads) are left out and the shares of the rest rescaled.
pub fn machine_score(w: &WindowRun, position_nodes: &[f64; POSITIONS]) -> Option<f64> {
    let pos = fill_positions(w);
    let used: Vec<usize> = (0..POSITIONS)
        .filter(|&i| pos[i].2 > 0.0 && position_nodes[i] > 0.0)
        .collect();
    let total: f64 = used.iter().map(|&i| position_nodes[i]).sum();
    if used.is_empty() || total <= 0.0 {
        return None;
    }
    let inv: f64 = used
        .iter()
        .map(|&i| position_nodes[i] / total / pos[i].2)
        .sum();
    Some(w.threads.max(1) as f64 / inv)
}

/// The bench's analysis (`results.json`). `ok` is false when a world finished in both windows
/// has different values.
pub fn analyse(fill: Option<&WindowRun>, prod: Option<&WindowRun>, p: &AnalyseParams) -> Value {
    // Identity: every (decision, world) finished in both windows.
    let mut seen: BTreeMap<(u64, u64), &str> = BTreeMap::new();
    for w in fill.iter().flat_map(|r| &r.worlds) {
        if let Some(h) = &w.hash {
            seen.insert((w.id, w.world), h);
        }
    }
    let mut compared = 0u64;
    let mut mismatches = Vec::new();
    for w in prod.iter().flat_map(|r| &r.worlds) {
        if let (Some(h), Some(h0)) = (&w.hash, seen.get(&(w.id, w.world))) {
            compared += 1;
            if h != h0 {
                mismatches.push(json!({ "decision": w.id, "world": w.world,
                                        "fill": h0, "prod": h }));
            }
        }
    }
    let fill_v = fill.map(|w| {
        let mut v = window_numbers(w, p);
        let pos = fill_positions(w);
        v["positions"] =
            json!((0..POSITIONS).map(|i| {
            let fin: Vec<u64> = w.worlds.iter()
                .filter(|x| usize::from(x.position) == i && x.finished)
                .map(|x| x.nodes).collect();
            json!({
                "position": i + 1, "threads": pos[i].0, "nodes": pos[i].1,
                "nodes_per_s_per_thread": pos[i].2.round(),
                "worlds_finished": fin.len(),
                // Only the worlds that finished inside the window: biased toward quick ones.
                "finished_mean_nodes": (!fin.is_empty())
                    .then(|| (fin.iter().sum::<u64>() as f64 / fin.len() as f64).round()),
            })
        }).collect::<Vec<_>>());
        v["thread_rows"] = json!(w.thread_rows);
        v
    });
    let prod_v = prod.map(|w| window_numbers(w, p));
    let score = fill.and_then(|w| machine_score(w, &p.position_nodes));
    let total_pn: f64 = p.position_nodes.iter().sum();
    let per_pos = p.golden_decisions as f64 / POSITIONS as f64;
    let golden_nodes: f64 = p
        .position_nodes
        .iter()
        .map(|n| per_pos * p.golden_worlds as f64 * n)
        .sum();
    let golden_hours = score.map(|s| golden_nodes / s / 3600.0);
    json!({
        "ok": mismatches.is_empty(),
        "identity": { "worlds_compared": compared, "mismatches": mismatches },
        "machine": { "threads": p.threads, "usable_cores": p.usable_cores, "dph": p.dph,
                     "cpu_model": p.cpu_model, "mem_gb": r1(p.mem_gb) },
        "position_nodes": {
            "values": p.position_nodes, "source": p.position_nodes_source,
            "shares": p.position_nodes.iter().map(|n| r3(n / total_pn.max(1e-300))).collect::<Vec<_>>(),
        },
        "fill": fill_v,
        "prod": prod_v,
        "score": score.map(|s| json!({
            "nodes_per_s": s.round(),
            "nodes_per_s_per_thread": (s / p.threads.max(1) as f64).round(),
            "dollars_per_tera_nodes": per_tera(p.dph, s),
        })),
        "golden": {
            "decisions": p.golden_decisions, "worlds": p.golden_worlds,
            "nodes": golden_nodes,
            "hours": golden_hours.map(r3),
            "dollars": golden_hours.filter(|_| p.dph > 0.0).map(|h| r3(h * p.dph)),
            "assumes": "one world per thread with every thread busy, at the fill window's speed per position",
        },
    })
}

/// The table-size sweep (`t1bench` settings `sweep_mb`, `sweep_s`): extra fill windows that run the
/// same worlds in the same order as the reference fill window (`fill_task` is fixed), each with a
/// different table size. Speed in nodes per second cannot show what a small table costs (a
/// table that forgets more makes the solver re-search positions, which still counts as nodes),
/// so each run is compared with the reference on the worlds that **finished in both**: the nodes
/// and the wall time those worlds took, and whether their values are byte-identical (table size
/// must change speed only). Only worlds short enough to finish inside the windows are paired,
/// which are mostly positions 5-8; the big positions 1-3 are not covered.
pub fn analyse_sweep(reference: &WindowRun, runs: &[WindowRun]) -> Value {
    let key = |w: &WorldRec| (w.id, w.world);
    let base: BTreeMap<(u64, u64), &WorldRec> = reference
        .worlds
        .iter()
        .filter(|w| w.finished)
        .map(|w| (key(w), w))
        .collect();
    let one = |r: &WindowRun| {
        let (mut n, mut nodes, mut base_nodes, mut wall, mut base_wall) =
            (0u64, 0f64, 0f64, 0f64, 0f64);
        let mut positions = std::collections::BTreeSet::new();
        let mut mismatches = Vec::new();
        for w in r.worlds.iter().filter(|w| w.finished) {
            let Some(b) = base.get(&key(w)) else { continue };
            n += 1;
            nodes += w.nodes as f64;
            base_nodes += b.nodes as f64;
            wall += w.wall_s;
            base_wall += b.wall_s;
            positions.insert(u32::from(w.position) + 1);
            if let (Some(h), Some(h0)) = (&w.hash, &b.hash)
                && h != h0
            {
                mismatches.push(
                    json!({ "decision": w.id, "world": w.world, "tt_mb": r.tt_mb,
                                        "reference": h0, "sweep": h }),
                );
            }
        }
        let rate = r.nodes() as f64 / r.wall_s.max(1e-9);
        let ratio = |a: f64, b: f64| (b > 0.0).then(|| r3(a / b));
        (
            json!({
                "tt_mb": r.tt_mb,
                "window_s": r.window_s,
                "nodes_per_s": rate.round(),
                "nodes_per_s_per_thread": (rate / r.threads.max(1) as f64).round(),
                "worlds_finished": r.worlds.iter().filter(|w| w.finished).count(),
                "paired_worlds": n,
                "paired_positions": positions,
                "nodes_vs_reference": ratio(nodes, base_nodes),
                "wall_vs_reference": ratio(wall, base_wall),
            }),
            mismatches,
        )
    };
    let mut entries = Vec::new();
    let mut all_mismatches = Vec::new();
    for r in runs {
        let (e, m) = one(r);
        entries.push(e);
        all_mismatches.extend(m);
    }
    json!({
        "reference_tt_mb": reference.tt_mb,
        "runs": entries,
        "mismatches": all_mismatches,
        "note": "paired on worlds finished in both the reference and the sweep window (short worlds, \
                 mostly positions 5-8); ratios above 1 mean the smaller or larger table needed more",
    })
}

fn sci(x: f64) -> String {
    if x >= 1e9 {
        format!("{:.2} billion", x / 1e9)
    } else if x >= 1e6 {
        format!("{:.1} million", x / 1e6)
    } else {
        format!("{x:.0}")
    }
}

fn opt(v: &Value, f: impl Fn(f64) -> String) -> String {
    v.as_f64().map_or("-".to_string(), f)
}

/// The plain-English summary of an [`analyse`] result (with `failure`, `table` and `warnings`
/// added by the runner, if any).
pub fn summary_text(a: &Value) -> String {
    let m = &a["machine"];
    let mut s = String::new();
    s.push_str(&format!(
        "Trick-1 machine bench: {} threads on {} usable cores{}{}, {}.\n",
        m["threads"],
        opt(&m["usable_cores"], |x| format!("{x:.1}")),
        m["cpu_model"]
            .as_str()
            .filter(|x| !x.is_empty())
            .map_or(String::new(), |x| format!(" ({x})")),
        opt(&m["mem_gb"], |x| format!(", {x:.0} GB memory")),
        opt(&m["dph"], |x| if x > 0.0 {
            format!("${x:.3} per hour")
        } else {
            "price unknown".into()
        }),
    ));
    if let Some(f) = a["failure"].as_str() {
        s.push_str(&format!(
            "FAILED: {f}. The bench stopped at once; the numbers below are what it had.\n"
        ));
    }
    s.push_str(
        "\"Nodes\" are the positions the exact solver [the program that plays every card \
         perfectly when all hands are face up] examines; nodes per second measure how fast a \
         machine solves.\n\n",
    );
    match a["score"].as_object() {
        Some(sc) => s.push_str(&format!(
            "Machine score: {} nodes per second for the real trick-1 mix{}.\n\
             (Each position in the trick counts by its share of the real work: {}.)\n",
            sci(sc["nodes_per_s"].as_f64().unwrap_or(0.0)),
            opt(&sc["dollars_per_tera_nodes"], |x| format!(
                ", ${x:.3} per million million nodes"
            )),
            a["position_nodes"]["shares"]
                .as_array()
                .map(|v| v
                    .iter()
                    .enumerate()
                    .map(|(i, x)| format!(
                        "position {} {:.1}%",
                        i + 1,
                        100.0 * x.as_f64().unwrap_or(0.0)
                    ))
                    .collect::<Vec<_>>()
                    .join(", "))
                .unwrap_or_default()
        )),
        None => s.push_str("Machine score: not available (the fill window measured nothing).\n"),
    }
    for (key, title, what) in [
        (
            "fill",
            "Fill window",
            "one world per thread, every thread solving from the first second",
        ),
        (
            "prod",
            "Production window",
            "the real schedule: each world's threads and table chosen by its position",
        ),
    ] {
        let w = &a[key];
        if w.is_null() {
            s.push_str(&format!("\n{title}: not run.\n"));
            continue;
        }
        s.push_str(&format!(
            "\n{title} ({what}): {} s{}.\n",
            opt(&w["wall_s"], |x| format!("{x:.0}")),
            w["stopped"]
                .as_str()
                .map_or(String::new(), |r| format!(", stopped early: {r}"))
        ));
        s.push_str(&format!(
            "  {} nodes per second in all, {} per thread{}{}.\n",
            sci(w["nodes_per_s"].as_f64().unwrap_or(0.0)),
            sci(w["nodes_per_s_per_thread"].as_f64().unwrap_or(0.0)),
            opt(&w["dollars_per_tera_nodes"], |x| format!(
                "; ${x:.3} per million million nodes"
            )),
            if key == "prod" {
                " (a world split over many threads examines extra nodes, so this rate \
                 includes that overhead)"
            } else {
                ""
            }
        ));
        s.push_str(&format!(
            "  CPU use {} of the usable cores; threads working {} of the time (every thread \
             working in {} of the samples, fewest {}); peak memory {}.\n",
            opt(&w["cpu_pct"], |x| format!("{x:.1}%")),
            opt(&w["busy_pct"], |x| format!("{x:.1}%")),
            opt(&w["all_busy_pct"], |x| format!("{x:.1}%")),
            w["min_busy"]
                .as_u64()
                .map_or("-".into(), |x| format!("{x} of {}", w["threads"])),
            w["peak_rss_mb"]
                .as_u64()
                .map_or("-".into(), |x| format!("{:.1} GB", x as f64 / 1024.0)),
        ));
        s.push_str(&format!(
            "  Worlds finished: {} of {} started.",
            w["worlds_finished"], w["worlds_started"]
        ));
        match key {
            "fill" => s.push_str(&format!(
                " Table [the solver's memory of positions already solved]: {} MB per thread.\n",
                w["tt_mb"]
            )),
            _ => {
                s.push_str(&format!(
                    " Queue: every decision x {} worlds; tables sized from the memory as in \
                     production.\n",
                    w["worlds_per_decision"]
                ));
                let classes: Vec<String> = w["plan"]["classes"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|c| {
                        format!(
                            "{}: {} thread(s), {} MB{}",
                            c["position"].as_u64().unwrap_or(0) + 1,
                            c["threads"],
                            c["tt_mb"],
                            match c["solves"].as_u64() {
                                Some(n) if n > 1 => format!(" ({n} solves)"),
                                _ => String::new(),
                            }
                        )
                    })
                    .collect();
                if !classes.is_empty() {
                    s.push_str(&format!(
                        "  Plan per position (threads and table per world): {}.\n",
                        classes.join("; ")
                    ));
                }
            }
        }
        if key == "fill" {
            s.push_str("  Per position in the trick, nodes per second per thread: ");
            s.push_str(
                &w["positions"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|x| x["threads"].as_u64().unwrap_or(0) > 0)
                    .map(|x| {
                        format!(
                            "{}: {}",
                            x["position"],
                            sci(x["nodes_per_s_per_thread"].as_f64().unwrap_or(0.0))
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            s.push_str(".\n");
        }
    }
    let id = &a["identity"];
    if a["ok"].as_bool() == Some(true) {
        s.push_str(&format!(
            "\nCheck: {} world(s) finished in both windows; all had byte-identical values.\n",
            id["worlds_compared"]
        ));
    } else {
        s.push_str(&format!(
            "\nFAILURE: {} world(s) finished in both windows had different values: this is a bug \
             (results.json, identity.mismatches).\n",
            id["mismatches"].as_array().map_or(0, Vec::len)
        ));
    }
    let g = &a["golden"];
    if let Some(h) = g["hours"].as_f64() {
        s.push_str(&format!(
            "A golden set of {} decisions x {} worlds ({:.2e} nodes) would take about {:.1} hours{} \
             on this machine at the fill window's speed, assuming one world per thread with every \
             thread busy.\n",
            g["decisions"],
            g["worlds"],
            g["nodes"].as_f64().unwrap_or(0.0),
            h,
            opt(&g["dollars"], |d| format!(" (${d:.2})"))
        ));
    }
    if let Some(sw) = a.get("sweep").filter(|v| v.is_object()) {
        s.push_str(&format!(
            "\nTable-size sweep (the same worlds at each table size; reference window: {} MB per thread). \
             Compared on worlds finished in both runs, which are mostly the short ones (positions 5-8); \
             the big early-trick worlds are not covered:\n",
            sw["reference_tt_mb"]
        ));
        for r in sw["runs"].as_array().into_iter().flatten() {
            s.push_str(&format!(
                "  {} MB per thread: {} per second, {} worlds finished; against the reference on {} shared world(s): {} the nodes, {} the time.\n",
                r["tt_mb"],
                opt(&r["nodes_per_s"], sci),
                r["worlds_finished"],
                r["paired_worlds"],
                opt(&r["nodes_vs_reference"], |x| format!("{x:.2}x")),
                opt(&r["wall_vs_reference"], |x| format!("{x:.2}x")),
            ));
        }
    }
    if let Some(src) = a["position_nodes"]["source"].as_str() {
        s.push_str(&format!(
            "Position work shares come from the {src} node counts per world (results.json, \
             position_nodes).\n"
        ));
    }
    for w in a["warnings"].as_array().into_iter().flatten() {
        if let Some(w) = w.as_str() {
            s.push_str(&format!("Warning: {w}\n"));
        }
    }
    s
}

// ---------------------------------------------------------------- process measures

fn clk_tck() -> f64 {
    static TCK: OnceLock<f64> = OnceLock::new();
    *TCK.get_or_init(|| {
        std::process::Command::new("getconf")
            .arg("CLK_TCK")
            .output()
            .ok()
            .and_then(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .trim()
                    .parse::<f64>()
                    .ok()
            })
            .filter(|&x| x > 0.0)
            .unwrap_or(100.0)
    })
}

/// CPU ticks this process has used (`/proc/self/stat` utime + stime, all threads), Linux only.
pub fn proc_cpu_ticks() -> Option<u64> {
    let t = std::fs::read_to_string("/proc/self/stat").ok()?;
    let rest = &t[t.rfind(')')? + 1..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    // f[0] is field 3 (state): utime and stime are fields 14 and 15.
    Some(f.get(11)?.parse::<u64>().ok()? + f.get(12)?.parse::<u64>().ok()?)
}

/// CPU seconds this process has used, Linux only.
pub fn proc_cpu_seconds() -> Option<f64> {
    proc_cpu_ticks().map(|t| t as f64 / clk_tck())
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
    use crate::measure::{build_decisions, collect_starts};
    use crate::pimc::SharedLeaf;
    use crate::worlds::BeliefSource;
    use solver::FnLeaf;
    use std::sync::Arc;

    /// Fill assignment: position `k mod 8`, every world of a position taken by one thread only,
    /// the same on every run, and the supply never runs out.
    fn rec(id: u64, world: u64, pos: u8, nodes: u64, wall: f64, hash: &str) -> WorldRec {
        WorldRec {
            id,
            position: pos,
            world,
            threads: 1,
            tt_mb: 0,
            thread: Some(0),
            start_s: 0.0,
            wall_s: wall,
            finished: true,
            nodes,
            hash: Some(hash.to_string()),
        }
    }

    fn win(tt_mb: usize, wall_s: f64, worlds: Vec<WorldRec>) -> WindowRun {
        WindowRun {
            mode: Mode::Fill,
            window_s: wall_s,
            wall_s,
            threads: 4,
            usable_cores: 4.0,
            tt_mb: Some(tt_mb),
            plan: None,
            worlds_per_decision: None,
            stopped: None,
            cpu_s: None,
            peak_rss_mb: None,
            samples: Vec::new(),
            thread_rows: Vec::new(),
            worlds,
        }
    }

    #[test]
    fn sweep_pairs_finished_worlds() {
        let base = win(
            512,
            240.0,
            vec![
                rec(1, 0, 5, 1000, 10.0, "aa"),
                rec(1, 1, 6, 2000, 20.0, "bb"),
                rec(2, 0, 7, 3000, 30.0, "cc"),
            ],
        );
        // A small table: the shared worlds need 1.5x the nodes and 1.4x the time; the world only
        // it finished (3, 0) is not paired; one value differs.
        let small = win(
            64,
            90.0,
            vec![
                rec(1, 0, 5, 1500, 14.0, "aa"),
                rec(1, 1, 6, 3000, 28.0, "XX"),
                rec(3, 0, 8, 10, 1.0, "dd"),
            ],
        );
        let r = analyse_sweep(&base, &[small]);
        assert_eq!(r["reference_tt_mb"], 512);
        let e = &r["runs"][0];
        assert_eq!(e["tt_mb"], 64);
        assert_eq!(e["paired_worlds"], 2);
        assert_eq!(e["nodes_vs_reference"], 1.5);
        assert_eq!(e["wall_vs_reference"], 1.4);
        assert_eq!(e["paired_positions"], json!([6, 7])); // 1-based; stored 0-based
        assert_eq!(e["worlds_finished"], 3);
        assert_eq!(r["mismatches"].as_array().map(Vec::len), Some(1));
        let a = json!({ "machine": {}, "ok": true, "identity": {}, "golden": {},
                        "position_nodes": {}, "sweep": r });
        let text = summary_text(&a);
        assert!(
            text.contains("64 MB per thread") && text.contains("1.50x the nodes"),
            "{text}"
        );
    }

    #[test]
    fn fill_assignment() {
        for t in [1usize, 4, 8, 13, 64, 245, 384] {
            let mut seen: BTreeMap<(usize, usize, u64), usize> = BTreeMap::new();
            for k in 0..t {
                for n in 0..50u64 {
                    let (d, w) = fill_task(k, n, t, 4);
                    assert!(d < 4);
                    let prev = seen.insert((k % POSITIONS, d, w), k);
                    assert!(
                        prev.is_none(),
                        "t {t}: thread {k} and {prev:?} share a world"
                    );
                }
            }
        }
        // Thread 0's worlds of the leader on 64 threads: j = 0, 8, 16, …: decisions 0, 0, 0 …
        // worlds 0, 2, 4 with 4 decisions; thread 8 starts at j = 1.
        assert_eq!(fill_task(0, 0, 64, 4), (0, 0));
        assert_eq!(fill_task(0, 1, 64, 4), (0, 2));
        assert_eq!(fill_task(8, 0, 64, 4), (1, 0));
        assert_eq!(fill_task(9, 0, 64, 4), (1, 0), "position 2's own sequence");
        assert_eq!(fill_task(3, 7, 4, 4), (3, 1));
    }

    fn sample(t: f64, busy: u64, cpu_s: f64) -> Sample {
        Sample {
            t,
            busy,
            held: busy,
            nodes: 0,
            finished: 0,
            cpu_s,
        }
    }

    /// The guard: nothing during the ramp; fewer than 95% of the `T` threads working for more
    /// than the streak stops; a short dip does not; low CPU over the interval stops.
    #[test]
    fn guard_rules() {
        let g = Guard {
            ramp_s: 20.0,
            min_cpu_pct: 80.0,
            cpu_interval_s: 30.0,
            busy_streak_s: 10.0,
            min_busy_pct: 95.0,
            usable_cores: 4.0,
        };
        let run = |busy: &dyn Fn(f64) -> u64, cores_used: f64| -> Option<(f64, String)> {
            let mut st = GuardState::default();
            let mut v = vec![sample(0.0, 0, 0.0)];
            for i in 1..=120 {
                let t = f64::from(i);
                v.push(sample(t, busy(t), cores_used * t));
                if let Some(r) = g.check(&mut st, &v, Mode::Fill, 4) {
                    return Some((t, r));
                }
            }
            None
        };
        assert_eq!(run(&|_| 4, 4.0), None, "full: never");
        // Idle from the start: caught 10 s after the ramp ends.
        let (t, r) = run(&|_| 2, 4.0).expect("idle threads");
        assert_eq!(t, 31.0, "{r}");
        assert!(r.starts_with("idle threads: only 2 of 4"), "{r}");
        // A 5-second dip is fine; a 12-second one is not.
        assert_eq!(
            run(&|t| if (40.0..45.0).contains(&t) { 3 } else { 4 }, 4.0),
            None
        );
        let (t, _) = run(&|t| if t >= 50.0 { 3 } else { 4 }, 4.0).expect("dip");
        assert_eq!(t, 61.0);
        // Half the CPU: caught at the end of the ramp, measured from 10 s.
        let (t, r) = run(&|_| 4, 2.0).expect("low CPU");
        assert_eq!(t, 20.0);
        assert_eq!(r, "low CPU: 50% in fill at 20 s");
        // No CPU check when switched off.
        let off = Guard {
            min_cpu_pct: 0.0,
            ..g.clone()
        };
        let mut st = GuardState::default();
        let v = vec![sample(0.0, 0, 0.0), sample(25.0, 4, 0.0)];
        assert_eq!(off.check(&mut st, &v, Mode::Prod, 4), None);
        // 384 threads: the production plan's 9 worlds × 42 threads (378 working) pass; 364 (just
        // under 95%) for over the streak do not.
        let wide = Guard {
            min_cpu_pct: 0.0,
            ..g.clone()
        };
        let streak = |busy: u64| {
            let mut st = GuardState::default();
            let mut v = vec![sample(0.0, 0, 0.0)];
            (1..=60).find_map(|i| {
                v.push(sample(f64::from(i), busy, 0.0));
                wide.check(&mut st, &v, Mode::Prod, 384)
            })
        };
        assert_eq!(streak(378), None);
        assert_eq!(streak(365), None);
        assert!(streak(364).is_some_and(|r| r.starts_with("idle threads: only 364 of 384")));
    }

    fn leaf() -> Arc<SharedLeaf> {
        Arc::new(FnLeaf(|s: &equity::MatchState| {
            MoneyChance::from_weights(core::array::from_fn(|t| {
                1 + u128::from(s.scores[t]) + 40 * u128::from(s.games[t])
            }))
        }))
    }

    /// Every sample after the ramp has all `T` threads working (ratio of samples).
    fn all_busy_share(w: &WindowRun, after: f64) -> f64 {
        let s: Vec<&Sample> = w.samples.iter().filter(|s| s.t > after).collect();
        assert!(!s.is_empty());
        s.iter().filter(|s| s.busy as usize == w.threads).count() as f64 / s.len() as f64
    }

    /// Both windows with many more threads than cores (64 on the test machine's few) and tiny
    /// tables: every thread searches in at least 99% of the samples after the ramp, nodes and
    /// worlds are counted, every solve stops at the window's end, and the windows agree on any
    /// world both finished. A plan that cannot fill the machine is caught by the guard.
    #[test]
    fn windows_keep_every_thread_busy() {
        let starts = collect_starts(400, 5);
        let ds = build_decisions(1, 32, 7, &starts);
        let l = leaf();
        let src = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: engine::game::Rules::HOUSE,
        };
        let t = 64;
        // Starting 64 worlds on a few cores takes about a second in a test build (each thread
        // samples and sets up its first world); samples before `ramp` are left out, as the
        // bench leaves out its first 5 s.
        let ramp = 1.5;
        let mut fill_p = WindowParams::new(Mode::Fill, 4.0, t);
        fill_p.tt_mb = 1;
        fill_p.sample_ms = 100;
        let mut prod_p = WindowParams::new(Mode::Prod, 4.0, t);
        prod_p.budget_mb = 64;
        prod_p.worlds = 20;
        prod_p.sample_ms = 100;
        // Small table needs and a short target time: leader worlds on 2 threads each, 32 of
        // them at once (many parallel solves), the rest on one thread each.
        prod_p.auto.t1_table_mb = Some([1.0; 8]);
        prod_p.auto.target_s = Some(2500.0);
        let mut n_cb = 0;
        let fill = run_window(&ds, &ctx, &fill_p, &mut |_| {
            n_cb += 1;
            None
        })
        .expect("fill");
        assert!(n_cb >= 25, "{n_cb} samples");
        assert_eq!(fill.stopped, None);
        assert_eq!(fill.thread_rows.len(), t);
        assert!(fill.wall_s >= 4.0 && fill.wall_s < 7.0, "{}", fill.wall_s);
        let share = all_busy_share(&fill, ramp);
        assert!(
            share >= 0.99,
            "fill: all {t} threads busy in {share:.3} of samples: {:?}",
            fill.samples
                .iter()
                .map(|s| (s.t, s.busy, s.finished))
                .collect::<Vec<_>>()
        );
        assert!(fill.nodes() > 0);
        // Every thread's world was stopped (or finished) and counted; positions 1..8 by k mod 8.
        for r in &fill.thread_rows {
            assert_eq!(usize::from(r.position), r.k % POSITIONS);
            assert!(r.nodes_finished + r.nodes_partial > 0, "thread {}", r.k);
        }
        // Each thread's last world was stopped at the window's end (unless it had just finished).
        let stopped = fill.worlds.iter().filter(|w| !w.finished).count();
        assert!(stopped >= t - 2 && stopped <= t, "{stopped}");
        let prod = run_window(&ds, &ctx, &prod_p, &mut |_| None).expect("prod");
        assert_eq!(prod.stopped, None);
        let share = all_busy_share(&prod, ramp);
        assert!(
            share >= 0.99,
            "prod: all {t} threads busy in {share:.3} of samples: {:?}",
            prod.samples
                .iter()
                .map(|s| (s.t, s.busy, s.held))
                .collect::<Vec<_>>()
        );
        assert!(prod.nodes() > 0 && !prod.worlds.is_empty());
        let a = analyse(
            Some(&fill),
            Some(&prod),
            &AnalyseParams {
                threads: t,
                usable_cores: 4.0,
                dph: 1.0,
                ..AnalyseParams::default()
            },
        );
        assert_eq!(a["ok"], json!(true), "{}", a["identity"]);
        assert!(a["score"]["nodes_per_s"].as_f64().expect("score") > 0.0);
        assert!(summary_text(&a).contains("Machine score"));
        // The callback stops a window at once with its reason.
        let mut q = fill_p.clone();
        q.window_s = 30.0;
        let t0 = Instant::now();
        let r = run_window(&ds, &ctx, &q, &mut |s| {
            (s.t >= 1.0).then(|| "asked to stop".to_string())
        })
        .expect("stopped");
        assert_eq!(r.stopped.as_deref(), Some("asked to stop"));
        assert!(t0.elapsed() < Duration::from_secs(10));
        // A plan that leaves threads idle (worlds of at most 3 threads on 8: two at once, 2
        // threads idle) is stopped by the guard soon after its ramp.
        let mut idle = WindowParams::new(Mode::Prod, 30.0, 8);
        idle.budget_mb = 64;
        idle.worlds = 20;
        idle.sample_ms = 100;
        idle.auto.max_threads = 3;
        idle.guard = Some(Guard {
            ramp_s: 1.0,
            min_cpu_pct: 0.0,
            busy_streak_s: 1.0,
            min_busy_pct: 95.0,
            ..Guard::default()
        });
        let t0 = Instant::now();
        let r = run_window(&ds, &ctx, &idle, &mut |_| None).expect("idle");
        let why = r.stopped.expect("the guard stopped it");
        assert!(
            why.starts_with("idle threads: only ") && why.contains(" of 8 working for over 1 s"),
            "{why}"
        );
        assert!(
            r.samples.iter().all(|s| s.held <= 6),
            "two 3-thread worlds at most"
        );
        assert!(t0.elapsed() < Duration::from_secs(10));
    }

    /// The score weights positions by their share of the work (a harmonic mean of the rates),
    /// and the projection is the golden set's nodes at that speed.
    #[test]
    fn score_and_projection() {
        let rows = (0..16)
            .map(|k| ThreadRow {
                k,
                position: (k % 8) as u8,
                worlds_finished: 0,
                // Positions 1–4 twice as fast as 5–8.
                nodes_finished: if k % 8 < 4 { 200 } else { 100 },
                nodes_partial: 0,
            })
            .collect();
        let w = WindowRun {
            mode: Mode::Fill,
            window_s: 10.0,
            wall_s: 10.0,
            threads: 16,
            usable_cores: 16.0,
            tt_mb: Some(1),
            plan: None,
            worlds_per_decision: None,
            stopped: None,
            cpu_s: Some(160.0),
            peak_rss_mb: Some(100),
            samples: vec![],
            thread_rows: rows,
            worlds: vec![],
        };
        // Equal shares: 1 / (0.5/20 + 0.5/10) = 13.33 per thread.
        let s = machine_score(&w, &[1.0; 8]).expect("score");
        assert!((s - 16.0 * 40.0 / 3.0).abs() < 1e-9, "{s}");
        // All the work at positions 1–4: their rate.
        let s = machine_score(&w, &[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]).expect("score");
        assert!((s - 16.0 * 20.0).abs() < 1e-9, "{s}");
        let p = AnalyseParams {
            threads: 16,
            dph: 3.6,
            golden_decisions: 8,
            golden_worlds: 10,
            position_nodes: [1000.0; 8],
            ..AnalyseParams::default()
        };
        let a = analyse(Some(&w), None, &p);
        // 8 decisions x 10 worlds x 1000 nodes at 16 x 40/3 nodes per second.
        let hours = 80_000.0 / (16.0 * 40.0 / 3.0) / 3600.0;
        assert!((a["golden"]["hours"].as_f64().expect("h") - r3(hours)).abs() < 1e-9);
        assert_eq!(a["fill"]["cpu_pct"], json!(100.0));
        assert_eq!(
            a["fill"]["positions"][0]["nodes_per_s_per_thread"],
            json!(20.0)
        );
        // $3.6/hour = $0.001/s at 213.3 nodes/s.
        assert_eq!(
            a["score"]["dollars_per_tera_nodes"],
            json!(r3(0.001 / (16.0 * 40.0 / 3.0) * 1e12))
        );
        let text = summary_text(&a);
        assert!(text.contains("Production window: not run"), "{text}");
        // A disagreement between the windows fails loudly.
        let rec = |h: &str| WorldRec {
            id: 1,
            position: 7,
            world: 0,
            threads: 1,
            tt_mb: 1,
            thread: None,
            start_s: 0.0,
            wall_s: 1.0,
            finished: true,
            nodes: 10,
            hash: Some(h.into()),
        };
        let mut f = w.clone();
        f.worlds = vec![rec("aa")];
        let mut g = w.clone();
        g.mode = Mode::Prod;
        g.worlds = vec![rec("aa")];
        assert_eq!(
            analyse(Some(&f), Some(&g), &p)["identity"]["worlds_compared"],
            json!(1)
        );
        g.worlds = vec![rec("bb")];
        let a = analyse(Some(&f), Some(&g), &p);
        assert_eq!(a["ok"], json!(false));
        assert!(summary_text(&a).contains("FAILURE"));
    }
}
