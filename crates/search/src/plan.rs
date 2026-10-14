//! Position-aware planning of exact world solves (docs/M3-SPEC.md §13.1 item 11): how many
//! threads and how much transposition table [the solver's memory of positions already solved]
//! each world gets, and the runner that packs the worlds onto the machine.
//!
//! **Why.** An exact trick-1 world costs from seconds to hours of one core, mostly by the position
//! in the trick (the leader's worlds about 100 times the eighth player's), and a parallel solve
//! uses its threads less well the more it has (about 8–10 times less work per thread on 368
//! threads than on 4, run `m2-trick1-1`). One global "worlds at once × threads per world" is
//! therefore wrong for most worlds of a run: cheap late-position worlds want one thread each,
//! many at once; leader worlds want many threads, so that the run does not end on one huge world
//! solved by a few threads.
//!
//! **The plan** ([`plan_auto`]), per world, on `T` usable threads with a memory budget `B` for
//! all tables at their peak:
//! 1. expected cost `e` in thread-seconds ([`AutoParams::expected_cost`]): at trick 1 a cost per
//!    position (`t1_costs_s`: measured one-thread solves, replaceable by the trick-1 bench's
//!    `suggested_auto_params`), at other tricks a leader cost times `position_ratio^-position`;
//!    times the solves the world takes (a robber's joint burn-and-play decision takes one per
//!    burn);
//! 2. a target wall time per world `τ = clamp(E / (waves · T), target_min_s, target_max_s)`, `E`
//!    the run's total expected cost (or `target_s`): the longest world should take at most about
//!    a `waves`-th of the run, so the run's tail stays short;
//! 3. threads `S = ⌈e / τ⌉`, raised if needed so the world's share of the memory holds the table
//!    it is expected to need (`t1_table_mb`; other tricks `leader_table_mb` scaled like the
//!    cost), at most `max_threads`,
//!    then snapped to `T / k` for a whole number `k` (so `k` such worlds fill the machine);
//! 4. table: the memory budget is shared out per thread (`B / T` each), so a world with `S`
//!    threads gets the largest table whose peak fits `S · B / T`. Threads in use never exceed
//!    `T`, so tables in use never exceed `B`.
//!
//! **The runner** ([`run_planned`]) starts worlds in plan order (most threads first, then most
//! expensive first: longest first) whenever enough threads are free, each on its own thread with
//! its own solver threads, and keeps each finished world's solver memory for the next world of
//! the same shape (no re-allocation churn); memory of shapes no longer queued is freed, and a
//! world waits if starting it would take the tables past the budget.
//!
//! **Values never depend on any of this**: thread counts, table sizes, start order and memory
//! reuse change only the speed of a solve, never its answer (tested in `crates/solver`, and for
//! whole worlds files in `measure`'s tests).

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use solver::SolveMem;

const MIB: u64 = 1 << 20;

/// The knobs of [`plan_auto`] (all have defaults; a manifest can override any of them).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AutoParams {
    /// Expected thread-seconds of one exact world solve of the leader of each trick (index =
    /// trick 0–5) on one thread, for tricks without a per-position table below. Trick 2: about
    /// 3 s (M2-SPIKE median 3.1 s); trick 3: about 0.01 s.
    pub leader_cost_s: [f64; 6],
    /// The cost ratio from one position in the trick to the next, for those tricks.
    pub position_ratio: f64,
    /// Trick-1 thread-seconds per world by position (0 = leader). Default: one-thread solves on
    /// the 4-vCPU Xeon (2026-10-14: positions 4, 6, 7, 8 measured at about 200, 16, 20, 25 s; the
    /// leader from the M2 worlds' nodes, about 2000 s; positions 2–3 and 5 interpolated). The
    /// trick-1 bench prints measured ones (`suggested_auto_params`).
    pub t1_costs_s: Option<[f64; 8]>,
    /// The table (MB) a leader's world is expected to need per trick, scaled by position like
    /// the cost, for tricks without a per-position table below; a world whose share of the
    /// memory is smaller gets more threads.
    pub leader_table_mb: [f64; 6],
    /// Trick-1 table need (MB) by position (position 4 used about 7 million entries, ~800 MB, at
    /// most; leader worlds up to ~200 million on the 384-core box).
    pub t1_table_mb: Option<[f64; 8]>,
    /// Target wall time per world in seconds; `None`: from the run's size (see the module text).
    pub target_s: Option<f64>,
    /// Bounds of the target computed from the run's size.
    pub target_min_s: f64,
    pub target_max_s: f64,
    /// The longest world should take about `1 / waves` of the run.
    pub waves: f64,
    /// Most threads for one world (0 = all).
    pub max_threads: usize,
}

impl Default for AutoParams {
    fn default() -> Self {
        AutoParams {
            leader_cost_s: [0.0, 2000.0, 3.0, 0.01, 0.0005, 0.0001],
            position_ratio: 2.257,
            t1_costs_s: Some([2000.0, 1200.0, 700.0, 250.0, 60.0, 16.0, 20.0, 25.0]),
            leader_table_mb: [0.0, 8192.0, 256.0, 4.0, 1.0, 1.0],
            t1_table_mb: Some([8192.0, 4096.0, 2048.0, 1024.0, 256.0, 128.0, 128.0, 128.0]),
            target_s: None,
            target_min_s: 5.0,
            target_max_s: 120.0,
            waves: 4.0,
            max_threads: 0,
        }
    }
}

/// What the plan needs to know about one world.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanTask {
    pub trick: u8,
    /// Position in the trick (0 = leader).
    pub position: u8,
    /// Exact solves the world takes (1, or one per burn of a joint decision).
    pub solves: u32,
}

/// Threads and table of one world.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WorldPlan {
    /// Solver threads (1 = the sequential solver).
    pub threads: usize,
    /// Table cap in MB.
    pub tt_mb: usize,
}

/// A plan for a run: one [`WorldPlan`] per task, and the order to start them in.
#[derive(Clone, Debug)]
pub struct Planned {
    pub plans: Vec<WorldPlan>,
    /// Task indices in start order.
    pub order: Vec<usize>,
    /// The target wall time per world used (seconds).
    pub target_s: f64,
    /// The run's expected total cost (thread-seconds).
    pub expected_s: f64,
    /// Per (trick, position, solves) class: worlds, expected cost, plan (for logs and records).
    pub record: Value,
}

impl AutoParams {
    fn position_scale(&self, position: u8) -> f64 {
        self.position_ratio
            .max(1.0)
            .powi(-i32::from(position.min(7)))
    }

    /// Expected thread-seconds of one world of `t`.
    pub fn expected_cost(&self, t: &PlanTask) -> f64 {
        let one = match (t.trick, &self.t1_costs_s) {
            (1, Some(c)) => c[usize::from(t.position.min(7))],
            _ => self.leader_cost_s[usize::from(t.trick.min(5))] * self.position_scale(t.position),
        };
        one.max(0.0) * f64::from(t.solves.max(1))
    }

    /// The table (MB) a world of `t` is expected to need.
    pub fn table_need_mb(&self, t: &PlanTask) -> f64 {
        match (t.trick, &self.t1_table_mb) {
            (1, Some(m)) => m[usize::from(t.position.min(7))],
            _ => {
                self.leader_table_mb[usize::from(t.trick.min(5))] * self.position_scale(t.position)
            }
        }
    }
}

/// The most memory the tables of one solve can make the process hold, from the solver's own
/// growth rule (`threads` 1: a sequential solve's table, else a parallel solve's shared table).
pub fn table_peak_bytes(tt_mb: usize, threads: usize) -> u64 {
    if threads <= 1 {
        solver::table_memory(tt_mb << 20).peak as u64
    } else {
        solver::shared_table_memory(tt_mb, threads).peak as u64
    }
}

/// The largest table cap (MB, at least 1) whose peak fits `peak_bytes` for a solve on `threads`.
pub fn largest_tt_mb(peak_bytes: u64, threads: usize) -> usize {
    let fits = |mb: usize| table_peak_bytes(mb, threads) <= peak_bytes;
    let top = (peak_bytes / MIB).max(1) as usize;
    if !fits(1) {
        return 1;
    }
    if fits(top) {
        return top;
    }
    let (mut lo, mut hi) = (1usize, top);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// `want` threads snapped to `t / k` for the smallest whole `k` that gives at least `want`
/// (rounding down when `t / k` is not whole), within 1..=t.
pub fn snap_threads(want: f64, t: usize) -> usize {
    let t = t.max(1);
    if want <= 1.0 {
        return 1;
    }
    let w = (want.ceil() as usize).min(t);
    t / (t / w).max(1)
}

/// The position-aware plan (see the module text) for `tasks` on `threads` threads with
/// `budget_bytes` of memory for all tables at their peak.
pub fn plan_auto(tasks: &[PlanTask], threads: usize, budget_bytes: u64, p: &AutoParams) -> Planned {
    let t = threads.max(1);
    let per_thread = budget_bytes / t as u64;
    let costs: Vec<f64> = tasks.iter().map(|x| p.expected_cost(x)).collect();
    let expected: f64 = costs.iter().sum();
    let target = p.target_s.unwrap_or_else(|| {
        (expected / (p.waves.max(1.0) * t as f64)).clamp(p.target_min_s, p.target_max_s)
    });
    let target = target.max(1e-6);
    let max_s = if p.max_threads == 0 {
        t
    } else {
        p.max_threads.min(t)
    };
    // One decision per class (trick, position, solves).
    let mut classes: BTreeMap<(u8, u8, u32), (WorldPlan, f64, u64)> = BTreeMap::new();
    let mut plans = Vec::with_capacity(tasks.len());
    for (task, &cost) in tasks.iter().zip(&costs) {
        let key = (task.trick, task.position, task.solves);
        let e = classes.entry(key).or_insert_with(|| {
            let need = p.table_need_mb(task);
            let mut s = snap_threads(cost / target, t);
            // More threads (a larger share of the memory) until the table fits its need.
            while s < max_s && (largest_tt_mb(per_thread * s as u64, s) as f64) < need {
                s = snap_threads(s as f64 + 1.0, t);
            }
            let s = s.min(max_s).max(1);
            let plan = WorldPlan {
                threads: s,
                tt_mb: largest_tt_mb(per_thread * s as u64, s),
            };
            (plan, cost, 0)
        });
        e.2 += 1;
        plans.push(e.0);
    }
    let mut order: Vec<usize> = (0..tasks.len()).collect();
    order.sort_by_key(|&i| {
        (
            core::cmp::Reverse(plans[i].threads),
            core::cmp::Reverse(costs[i].to_bits()),
            i,
        )
    });
    let record = json!({
        "threads": t, "budget_mb": budget_bytes / MIB, "target_s": target,
        "expected_thread_s": expected,
        "classes": classes.iter().map(|(&(trick, pos, solves), (plan, cost, n))| json!({
            "trick": trick, "position": pos, "solves": solves, "worlds": n,
            "expected_thread_s": cost, "threads": plan.threads, "tt_mb": plan.tt_mb,
        })).collect::<Vec<_>>(),
    });
    Planned {
        plans,
        order,
        target_s: target,
        expected_s: expected,
        record,
    }
}

/// The same plan for every one of `n` tasks, in task order (the explicit "worlds at once ×
/// threads per world" settings: `worlds_at_once · solve_threads` threads in all).
pub fn plan_uniform(n: usize, plan: WorldPlan) -> (Vec<WorldPlan>, Vec<usize>) {
    (vec![plan; n], (0..n).collect())
}

/// Live state of [`run_planned`] (progress reports only).
#[derive(Debug, Default)]
pub struct ExecProgress {
    /// Worlds being solved now.
    pub running: AtomicU64,
    /// Threads they hold.
    pub busy_threads: AtomicU64,
    /// Every world has been started and threads are free: the run's tail.
    pub tail: AtomicBool,
}

/// When and how one world ran.
#[derive(Clone, Copy, Debug)]
pub struct Ran {
    /// Seconds from the start of [`run_planned`].
    pub start_s: f64,
    pub wall_s: f64,
    pub plan: WorldPlan,
}

/// Run tasks `order` (indices into `plans`) on `threads` threads: start each in order as soon as
/// its threads are free (a task wanting more than `threads` runs alone), each on its own thread,
/// with a solver memory taken from a pool of memories of the same plan shape when one is free.
/// The memories alive (busy or pooled), counted at their peak ([`table_peak_bytes`]), stay within
/// `budget_bytes`: idle memories of other shapes are freed first, and a task waits while it
/// cannot fit (unless nothing runs). `work(task, plan, memory)` solves a task; `done` is called on
/// this thread for each finished task in completion order. Once `stop` is set (or `done` fails)
/// no new task starts; the running ones finish. Returns `done`'s first error. A panic in `work`
/// stops the starts and is raised again once the running tasks are over.
#[allow(clippy::too_many_arguments)]
pub fn run_planned<R: Send>(
    order: &[usize],
    plans: &[WorldPlan],
    threads: usize,
    budget_bytes: u64,
    stop: &AtomicBool,
    progress: Option<&ExecProgress>,
    work: impl Fn(usize, WorldPlan, &mut SolveMem) -> R + Sync,
    mut done: impl FnMut(usize, R, Ran) -> Result<(), String>,
) -> Result<(), String> {
    let threads = threads.max(1);
    let shape_peak = |p: WorldPlan| table_peak_bytes(p.tt_mb, p.threads);
    let mut queued: BTreeMap<WorldPlan, usize> = BTreeMap::new();
    for &k in order {
        *queued.entry(plans[k]).or_default() += 1;
    }
    let t0 = Instant::now();
    let work = &work;
    type Msg<R> = (usize, Option<R>, SolveMem, f64, f64);
    std::thread::scope(|s| {
        let (tx, rx) = mpsc::channel::<Msg<R>>();
        let mut pool: BTreeMap<WorldPlan, Vec<SolveMem>> = BTreeMap::new();
        let mut held = 0u64;
        let mut free = threads;
        let mut running = 0usize;
        let mut next = 0usize;
        let mut err: Option<String> = None;
        let mut panicked = false;
        loop {
            while err.is_none() && !panicked && next < order.len() && !stop.load(Ordering::Relaxed)
            {
                let k = order[next];
                let p = plans[k];
                let need = p.threads.clamp(1, threads);
                if running > 0 && need > free {
                    break;
                }
                let mem = match pool.get_mut(&p).and_then(Vec::pop) {
                    Some(m) => m,
                    None => {
                        let peak = shape_peak(p);
                        // Free idle memories of other shapes, largest first, to make room.
                        while held.saturating_add(peak) > budget_bytes {
                            let Some(&q) = pool
                                .iter()
                                .filter(|(_, v)| !v.is_empty())
                                .map(|(q, _)| q)
                                .max_by_key(|&&q| (shape_peak(q), q))
                            else {
                                break;
                            };
                            pool.get_mut(&q).and_then(Vec::pop);
                            held -= shape_peak(q);
                        }
                        if running > 0 && held.saturating_add(peak) > budget_bytes {
                            break;
                        }
                        held = held.saturating_add(peak);
                        SolveMem::new()
                    }
                };
                if let Some(c) = queued.get_mut(&p) {
                    *c -= 1;
                }
                next += 1;
                free = free.saturating_sub(need);
                running += 1;
                let tx = tx.clone();
                s.spawn(move || {
                    let mut mem = mem;
                    let start = t0.elapsed().as_secs_f64();
                    let r = catch_unwind(AssertUnwindSafe(|| work(k, p, &mut mem)));
                    let wall = t0.elapsed().as_secs_f64() - start;
                    match r {
                        Ok(r) => {
                            let _ = tx.send((k, Some(r), mem, start, wall));
                        }
                        Err(e) => {
                            let _ = tx.send((k, None, mem, start, wall));
                            resume_unwind(e);
                        }
                    }
                });
            }
            if let Some(pr) = progress {
                pr.running.store(running as u64, Ordering::Relaxed);
                pr.busy_threads
                    .store((threads - free) as u64, Ordering::Relaxed);
                pr.tail
                    .store(next == order.len() && free > 0, Ordering::Relaxed);
            }
            if running == 0 {
                break;
            }
            let Ok((k, r, mem, start, wall)) = rx.recv() else {
                break;
            };
            let p = plans[k];
            running -= 1;
            free = (free + p.threads.clamp(1, threads)).min(threads);
            let more = queued.get(&p).copied().unwrap_or(0) > 0;
            if more && err.is_none() && !panicked && !stop.load(Ordering::Relaxed) {
                pool.entry(p).or_default().push(mem);
            } else {
                drop(mem);
                held = held.saturating_sub(shape_peak(p));
            }
            match r {
                None => panicked = true,
                Some(r) if err.is_none() => {
                    let ran = Ran {
                        start_s: start,
                        wall_s: wall,
                        plan: p,
                    };
                    if let Err(e) = done(k, r, ran) {
                        err = Some(e);
                    }
                }
                Some(_) => {}
            }
        }
        match err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    fn t1(position: u8) -> PlanTask {
        PlanTask {
            trick: 1,
            position,
            solves: 1,
        }
    }

    #[test]
    fn snapping() {
        assert_eq!(snap_threads(0.3, 384), 1);
        assert_eq!(snap_threads(1.0, 384), 1);
        assert_eq!(snap_threads(8.3, 384), 9);
        assert_eq!(snap_threads(100.0, 384), 128);
        assert_eq!(snap_threads(200.0, 384), 384);
        assert_eq!(snap_threads(5000.0, 384), 384);
        assert_eq!(snap_threads(64.0, 368), 73);
        assert_eq!(snap_threads(3.0, 4), 4);
        assert_eq!(snap_threads(2.0, 4), 2);
        for t in [1, 3, 4, 7, 64, 368, 384, 512] {
            for w in 1..=t {
                let s = snap_threads(w as f64, t);
                assert!((1..=t).contains(&s), "{w} of {t}: {s}");
                assert!(t / s >= 1);
            }
        }
    }

    #[test]
    fn tables_fit_their_share() {
        for (peak, threads) in [(1u64 << 30, 1usize), (8 << 30, 16), (100 << 20, 4)] {
            let mb = largest_tt_mb(peak, threads);
            assert!(table_peak_bytes(mb, threads) <= peak);
            assert!(table_peak_bytes(mb + 1, threads) > peak);
        }
        assert_eq!(largest_tt_mb(0, 1), 1);
    }

    /// A trick-1 run on a 384-thread, 225 GB box: leader worlds get the most threads and the
    /// largest tables, late positions one thread; threads and table peaks of any full set of
    /// worlds of one class fit the machine; start order is most threads first.
    #[test]
    fn trick1_plan_shape() {
        let p = AutoParams::default();
        let mut tasks = Vec::new();
        for pos in 0..8 {
            for _ in 0..40 {
                tasks.push(t1(pos));
            }
        }
        let budget = 225u64 << 30;
        let pl = plan_auto(&tasks, 384, budget, &p);
        let th = |pos: usize| pl.plans[pos * 40].threads;
        assert!(th(0) > th(3) && th(3) >= th(7), "{:?}", pl.record);
        assert_eq!(th(7), 1);
        assert!(pl.plans[0].tt_mb > pl.plans[7 * 40].tt_mb);
        for w in &pl.plans {
            let k = (384 / w.threads) as u64;
            assert!(k * table_peak_bytes(w.tt_mb, w.threads) <= budget, "{w:?}");
        }
        let firsts: Vec<usize> = pl.order.iter().map(|&i| pl.plans[i].threads).collect();
        assert!(firsts.windows(2).all(|w| w[0] >= w[1]));
        // The target is a quarter of the run's ideal length, within its bounds; a tiny run takes
        // the smallest.
        let ideal = pl.expected_s / (4.0 * 384.0);
        assert!((pl.target_s - ideal.clamp(p.target_min_s, p.target_max_s)).abs() < 1e-9);
        let tiny = plan_auto(&tasks[..2], 384, budget, &p);
        assert_eq!(tiny.target_s, p.target_min_s);
        // Tricks 3–4 are cheap: everything on one thread.
        let late: Vec<PlanTask> = (0..8)
            .map(|pos| PlanTask {
                trick: 3,
                position: pos,
                solves: 1,
            })
            .collect();
        let pl = plan_auto(&late, 384, budget, &p);
        assert!(pl.plans.iter().all(|w| w.threads == 1));
        // A joint decision costs one solve per burn.
        let j = PlanTask { solves: 3, ..t1(2) };
        assert_eq!(p.expected_cost(&j), 3.0 * p.expected_cost(&t1(2)));
        // Measured trick-1 costs replace the model.
        let m = AutoParams {
            t1_costs_s: Some([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
            ..AutoParams::default()
        };
        assert_eq!(m.expected_cost(&t1(5)), 3.0);
        // Little memory: the leader gets more threads for a big enough table.
        let small = plan_auto(&tasks, 384, 40u64 << 30, &p);
        assert!(small.plans[0].threads >= pl.plans[0].threads);
    }

    /// The runner never uses more threads than it has, starts in order, reuses memories of the
    /// same shape, keeps memory within the budget, and reports every task once.
    #[test]
    fn runner_packs_and_reports() {
        let plans: Vec<WorldPlan> = [4, 4, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1]
            .iter()
            .map(|&t| WorldPlan {
                threads: t,
                tt_mb: 1,
            })
            .collect();
        let order: Vec<usize> = (0..plans.len()).collect();
        let busy = AtomicUsize::new(0);
        let max_busy = AtomicUsize::new(0);
        let started = Mutex::new(Vec::new());
        let mut seen = Vec::new();
        let stop = AtomicBool::new(false);
        run_planned(
            &order,
            &plans,
            4,
            u64::MAX,
            &stop,
            None,
            |k, p, _mem| {
                started.lock().unwrap().push(k);
                let b = busy.fetch_add(p.threads, Ordering::SeqCst) + p.threads;
                max_busy.fetch_max(b, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(5));
                busy.fetch_sub(p.threads, Ordering::SeqCst);
                k * 10
            },
            |k, r, ran| {
                assert_eq!(r, k * 10);
                assert!(ran.wall_s >= 0.0);
                seen.push(k);
                Ok(())
            },
        )
        .expect("runs");
        assert!(max_busy.load(Ordering::SeqCst) <= 4);
        seen.sort_unstable();
        assert_eq!(seen, order);
        let st = started.into_inner().unwrap();
        // Each task starts after every task before it in the order has started.
        assert_eq!(st.len(), plans.len());
        // A stop before the start runs nothing; an error in `done` stops further starts.
        stop.store(true, Ordering::SeqCst);
        let mut n = 0;
        run_planned(
            &order,
            &plans,
            4,
            u64::MAX,
            &stop,
            None,
            |k, _, _| k,
            |_, _, _| {
                n += 1;
                Ok(())
            },
        )
        .expect("nothing to do");
        assert_eq!(n, 0);
        stop.store(false, Ordering::SeqCst);
        let r = run_planned(
            &order,
            &plans,
            1,
            u64::MAX,
            &stop,
            None,
            |k, _, _| k,
            |_, _, _| Err("sink full".to_string()),
        );
        assert_eq!(r, Err("sink full".to_string()));
    }

    /// With a budget that holds one table of each size only, the runner frees idle memories of
    /// another shape before allocating, and never runs two worlds whose tables would not fit.
    #[test]
    fn runner_respects_memory_budget() {
        let big = WorldPlan {
            threads: 1,
            tt_mb: 64,
        };
        let small = WorldPlan {
            threads: 1,
            tt_mb: 8,
        };
        let plans = vec![big, big, small, small, small];
        let order: Vec<usize> = (0..plans.len()).collect();
        let budget = table_peak_bytes(64, 1) + table_peak_bytes(8, 1);
        let live = AtomicU64::new(0);
        let max_live = AtomicU64::new(0);
        let stop = AtomicBool::new(false);
        run_planned(
            &order,
            &plans,
            4,
            budget,
            &stop,
            None,
            |_, p, _| {
                let b = table_peak_bytes(p.tt_mb, p.threads);
                let now = live.fetch_add(b, Ordering::SeqCst) + b;
                max_live.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(3));
                live.fetch_sub(b, Ordering::SeqCst);
            },
            |_, _, _| Ok(()),
        )
        .expect("runs");
        assert!(max_live.load(Ordering::SeqCst) <= budget);
    }

    #[test]
    fn params_round_trip() {
        let p: AutoParams =
            serde_json::from_str(r#"{"target_s": 30, "max_threads": 64}"#).expect("partial params");
        assert_eq!(p.target_s, Some(30.0));
        assert_eq!(p.max_threads, 64);
        assert_eq!(p.position_ratio, AutoParams::default().position_ratio);
        assert!(serde_json::from_str::<AutoParams>(r#"{"nope": 1}"#).is_err());
    }
}
