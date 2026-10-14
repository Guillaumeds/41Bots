//! Honest progress for a run of tasks of known kinds (docs/M3-SPEC.md §13.1 item 11): how much
//! of the work is done, weighted by what each task is expected to cost rather than counted, and
//! when the run should end. Reports only: nothing here ever changes a value or the order of work.
//!
//! Tasks fall into classes (for a worlds run: the trick and the position in the trick, named like
//! `t1p1` for trick 1's leader). Each task has a prior cost in thread-seconds from the
//! scheduler's own model ([`crate::plan::AutoParams::expected_cost`], the numbers that also put
//! the queue in longest-first order). As tasks finish, the class's measured cost (wall time ×
//! threads) per unit of prior replaces the prior's scale for that class; a class with no task
//! finished yet takes the scale measured over the whole run so far (1 before anything finished).
//! A running task counts as done for its elapsed thread-time, up to its expected cost.
//!
//! - `done_frac` = expected work done / all expected work;
//! - remaining time = work left / this session's rate (work done since this process started /
//!   its elapsed time), so work restored from a checkpoint does not make the rate look faster;
//! - `basis` is `"measured"` once at least 10% of the work is done and every class with work left
//!   has at least 2 tasks finished and timed in this session; before that it is `"rough: …"`,
//!   saying which of the two is missing (and which classes are unmeasured).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// The smallest prior cost of one task (seconds), so a class the model calls free still weighs.
pub const MIN_PRIOR_S: f64 = 1e-6;

/// The share of the work that must be done before the estimate can be called measured.
pub const MEASURED_SHARE: f64 = 0.10;

/// Tasks of one class that must be finished and timed before the class counts as measured.
pub const MEASURED_TASKS: u64 = 2;

/// One class's counts (prior costs in thread-seconds).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClassCounts {
    /// The class's name in reports (e.g. `t1p1`).
    pub name: String,
    /// Tasks in the run, including those done before this session.
    pub tasks: u64,
    /// Their summed prior cost.
    pub prior_total: f64,
    /// Tasks finished (before or in this session).
    pub done: u64,
    /// Their summed prior cost.
    pub prior_done: f64,
    /// Tasks finished and timed in this session.
    pub timed: u64,
    /// Their summed prior cost.
    pub prior_timed: f64,
    /// Their measured thread-seconds (wall time × threads).
    pub measured: f64,
    /// Summed prior cost of the tasks done before this session (restored from a checkpoint).
    pub prior_pre: f64,
}

/// A task being worked on now.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunningTask {
    /// Index of its class.
    pub class: usize,
    /// Its prior cost (thread-seconds).
    pub prior: f64,
    /// Threads it holds.
    pub threads: f64,
    /// Seconds since it started.
    pub elapsed_s: f64,
}

/// The estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct Eta {
    /// Expected work done / all expected work, 0 to 1.
    pub done_frac: f64,
    /// Seconds until the run ends at this session's rate (`None` before anything is done).
    pub remaining_s: Option<f64>,
    /// Whether the estimate rests on measured costs (see the module text).
    pub measured: bool,
    /// `"measured"`, or `"rough: …"` saying why not yet.
    pub basis: String,
    /// All expected work, thread-seconds.
    pub work_s: f64,
    /// Expected work done, thread-seconds.
    pub done_s: f64,
}

/// The estimate from the classes' counts, the running tasks and this session's elapsed seconds.
pub fn estimate(classes: &[ClassCounts], running: &[RunningTask], elapsed_s: f64) -> Eta {
    let (tm, tp) = classes
        .iter()
        .fold((0.0, 0.0), |(m, p), c| (m + c.measured, p + c.prior_timed));
    let global = if tp > 0.0 && tm > 0.0 { tm / tp } else { 1.0 };
    let scale: Vec<f64> = classes
        .iter()
        .map(|c| {
            if c.timed > 0 && c.prior_timed > 0.0 && c.measured > 0.0 {
                c.measured / c.prior_timed
            } else {
                global
            }
        })
        .collect();
    let sum = |f: &dyn Fn(&ClassCounts) -> f64| -> f64 {
        classes.iter().zip(&scale).map(|(c, s)| f(c) * s).sum()
    };
    let work = sum(&|c| c.prior_total);
    let pre = sum(&|c| c.prior_pre);
    // A class with every task done counts whole (its sums are kept in whole nanoseconds).
    let mut done = sum(&|c| {
        if c.done >= c.tasks {
            c.prior_total
        } else {
            c.prior_done.min(c.prior_total)
        }
    });
    for r in running {
        let s = scale.get(r.class).copied().unwrap_or(global);
        done += (r.elapsed_s.max(0.0) * r.threads.max(1.0)).min(r.prior.max(0.0) * s);
    }
    let done = done.min(work);
    let done_frac = if work > 0.0 { done / work } else { 1.0 };
    let session = done - pre;
    let remaining_s = if work - done <= 0.0 {
        Some(0.0)
    } else if session > 0.0 && elapsed_s > 0.0 {
        Some((work - done) * elapsed_s / session)
    } else {
        None
    };
    let unmeasured: Vec<&str> = classes
        .iter()
        .filter(|c| c.done < c.tasks && c.timed < MEASURED_TASKS)
        .map(|c| c.name.as_str())
        .collect();
    let measured = done_frac >= MEASURED_SHARE && unmeasured.is_empty();
    let basis = if measured {
        "measured".to_string()
    } else {
        let mut why = Vec::new();
        if done_frac < MEASURED_SHARE {
            why.push(format!(
                "{:.1}% of the work done (under {:.0}%)",
                done_frac * 100.0,
                MEASURED_SHARE * 100.0
            ));
        }
        if !unmeasured.is_empty() {
            why.push(format!(
                "fewer than {MEASURED_TASKS} tasks timed in {}",
                unmeasured.join(", ")
            ));
        }
        format!("rough: {}", why.join("; "))
    };
    Eta {
        done_frac,
        remaining_s,
        measured,
        basis,
        work_s: work,
        done_s: done,
    }
}

/// One class as the tracker is told it at the start.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EtaClass {
    pub name: String,
    /// Tasks in the run, including those done before this session.
    pub tasks: u64,
    /// Their summed prior cost (thread-seconds).
    pub prior_total_s: f64,
    /// Tasks done before this session.
    pub pre_done: u64,
    /// Their summed prior cost.
    pub prior_pre_s: f64,
}

#[derive(Debug, Default)]
struct Slot {
    busy: AtomicBool,
    /// Microseconds since the tracker's start, plus 1 (0: no task).
    start_us: AtomicU64,
    class: AtomicU64,
    prior_ns: AtomicU64,
    threads: AtomicU64,
}

fn ns(s: f64) -> u64 {
    (s.max(0.0) * 1e9) as u64
}

/// The live counters behind [`estimate`], updated by worker threads without locks.
#[derive(Debug)]
pub struct EtaTracker {
    t0: Instant,
    classes: Vec<EtaClass>,
    done: Vec<AtomicU64>,
    prior_done_ns: Vec<AtomicU64>,
    timed: Vec<AtomicU64>,
    prior_timed_ns: Vec<AtomicU64>,
    measured_ns: Vec<AtomicU64>,
    slots: Vec<Slot>,
}

impl EtaTracker {
    /// A tracker for `classes`, with room for `slots` tasks running at once; its clock starts now.
    pub fn new(classes: Vec<EtaClass>, slots: usize) -> EtaTracker {
        let atomics = |f: &dyn Fn(&EtaClass) -> u64| -> Vec<AtomicU64> {
            classes.iter().map(|c| AtomicU64::new(f(c))).collect()
        };
        EtaTracker {
            t0: Instant::now(),
            done: atomics(&|c| c.pre_done),
            prior_done_ns: atomics(&|c| ns(c.prior_pre_s)),
            timed: atomics(&|_| 0),
            prior_timed_ns: atomics(&|_| 0),
            measured_ns: atomics(&|_| 0),
            slots: (0..slots.max(1)).map(|_| Slot::default()).collect(),
            classes,
        }
    }

    /// A task of `class` (prior `prior_s`, on `threads` threads) starts; returns its slot (`hint`
    /// is tried first: a worker's own index keeps workers off each other's slots).
    pub fn begin(&self, hint: usize, class: usize, prior_s: f64, threads: usize) -> Option<usize> {
        let n = self.slots.len();
        let i = (0..n).map(|d| (hint + d) % n).find(|&i| {
            self.slots[i]
                .busy
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        })?;
        let s = &self.slots[i];
        s.class.store(class as u64, Ordering::Relaxed);
        s.prior_ns.store(ns(prior_s), Ordering::Relaxed);
        s.threads.store(threads.max(1) as u64, Ordering::Relaxed);
        let now = self.t0.elapsed().as_micros() as u64 + 1;
        s.start_us.store(now, Ordering::Release);
        Some(i)
    }

    /// The task started in `slot` finished after `wall_s` seconds.
    pub fn end(
        &self,
        slot: Option<usize>,
        class: usize,
        prior_s: f64,
        threads: usize,
        wall_s: f64,
    ) {
        if let Some(s) = slot.and_then(|i| self.slots.get(i)) {
            s.start_us.store(0, Ordering::Release);
            s.busy.store(false, Ordering::Release);
        }
        if class >= self.classes.len() {
            return;
        }
        let p = ns(prior_s);
        self.done[class].fetch_add(1, Ordering::Relaxed);
        self.prior_done_ns[class].fetch_add(p, Ordering::Relaxed);
        self.timed[class].fetch_add(1, Ordering::Relaxed);
        self.prior_timed_ns[class].fetch_add(p, Ordering::Relaxed);
        self.measured_ns[class].fetch_add(ns(wall_s * threads.max(1) as f64), Ordering::Relaxed);
    }

    /// The estimate now.
    pub fn snapshot(&self) -> Eta {
        let now_us = self.t0.elapsed().as_micros() as u64 + 1;
        let s = |x: &AtomicU64| x.load(Ordering::Relaxed) as f64 / 1e9;
        let classes: Vec<ClassCounts> = self
            .classes
            .iter()
            .enumerate()
            .map(|(i, c)| ClassCounts {
                name: c.name.clone(),
                tasks: c.tasks,
                prior_total: c.prior_total_s,
                done: self.done[i].load(Ordering::Relaxed),
                prior_done: s(&self.prior_done_ns[i]),
                timed: self.timed[i].load(Ordering::Relaxed),
                prior_timed: s(&self.prior_timed_ns[i]),
                measured: s(&self.measured_ns[i]),
                prior_pre: c.prior_pre_s,
            })
            .collect();
        let running: Vec<RunningTask> = self
            .slots
            .iter()
            .filter_map(|sl| {
                let start = sl.start_us.load(Ordering::Acquire);
                (start != 0).then(|| RunningTask {
                    class: sl.class.load(Ordering::Relaxed) as usize,
                    prior: s(&sl.prior_ns),
                    threads: sl.threads.load(Ordering::Relaxed) as f64,
                    elapsed_s: now_us.saturating_sub(start) as f64 / 1e6,
                })
            })
            .collect();
        estimate(&classes, &running, self.t0.elapsed().as_secs_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simulated run: `tasks` (class, true cost) taken in order by `threads` one-thread
    /// workers. Returns the end time and, at every `step` seconds, the time and the estimate.
    fn simulate(
        tasks: &[(usize, f64)],
        threads: usize,
        priors: &[f64],
        names: &[String],
        step: f64,
    ) -> (f64, Vec<(f64, Eta)>) {
        // Exact schedule first: each task starts when a worker frees up, in order.
        let mut free = vec![0.0f64; threads];
        let mut sched = Vec::new();
        for &(c, cost) in tasks {
            let w = (0..threads)
                .min_by(|&a, &b| free[a].total_cmp(&free[b]))
                .expect("a worker");
            let start = free[w];
            free[w] = start + cost;
            sched.push((c, start, start + cost));
        }
        let end = free.iter().copied().fold(0.0, f64::max);
        let mut out = Vec::new();
        let mut t = step;
        while t < end {
            let mut classes: Vec<ClassCounts> = names
                .iter()
                .map(|n| ClassCounts {
                    name: n.clone(),
                    ..ClassCounts::default()
                })
                .collect();
            let mut running = Vec::new();
            for &(c, s, e) in &sched {
                let k = &mut classes[c];
                k.tasks += 1;
                k.prior_total += priors[c];
                if e <= t {
                    k.done += 1;
                    k.prior_done += priors[c];
                    k.timed += 1;
                    k.prior_timed += priors[c];
                    k.measured += e - s;
                } else if s <= t {
                    running.push(RunningTask {
                        class: c,
                        prior: priors[c],
                        threads: 1.0,
                        elapsed_s: t - s,
                    });
                }
            }
            out.push((t, estimate(&classes, &running, t)));
            t += step;
        }
        (end, out)
    }

    /// Eight classes whose true costs fall from 20 to 1 (the first tasks 20 times the last), with
    /// priors that are wrong (all equal): once the basis is "measured", the predicted end is
    /// within 15% of the true one, both in longest-first order (the classes finish one after
    /// another) and with the classes interleaved (measured early); before that it says why.
    #[test]
    fn eta_within_15_percent_once_measured() {
        let names: Vec<String> = (0..8).map(|p| format!("t1p{}", p + 1)).collect();
        let cost = |c: usize| 20.0 * (1.0f64 / 20.0).powf(c as f64 / 7.0);
        let priors = vec![1.0; 8];
        let longest_first: Vec<(usize, f64)> = (0..8)
            .flat_map(|c| std::iter::repeat_n((c, cost(c)), 30))
            .collect();
        let interleaved: Vec<(usize, f64)> =
            (0..30).flat_map(|_| (0..8).map(|c| (c, cost(c)))).collect();
        for (name, tasks) in [
            ("longest first", longest_first),
            ("interleaved", interleaved),
        ] {
            let (end, ests) = simulate(&tasks, 6, &priors, &names, 0.5);
            let mut measured = 0;
            for (t, e) in &ests {
                assert!((0.0..=1.0).contains(&e.done_frac), "{name} at {t}: {e:?}");
                if e.measured {
                    measured += 1;
                    let predicted = t + e.remaining_s.expect("a rate once measured");
                    let err = (predicted - end).abs() / end;
                    assert!(
                        err <= 0.15,
                        "{name} at {t:.1}: predicted end {predicted:.1}, true {end:.1}"
                    );
                } else {
                    assert!(e.basis.starts_with("rough: "), "{e:?}");
                }
            }
            assert!(measured > 0, "{name}: the basis became measured");
        }
        // Before anything is timed: rough, naming every unmeasured class.
        let e = estimate(
            &[ClassCounts {
                name: "t1p1".into(),
                tasks: 4,
                prior_total: 4.0,
                ..ClassCounts::default()
            }],
            &[],
            10.0,
        );
        assert!(!e.measured && e.basis.contains("t1p1") && e.remaining_s.is_none());
    }

    /// Work restored from a checkpoint counts as done but not towards this session's rate; a
    /// running task is credited by its elapsed thread-time up to its expected cost.
    #[test]
    fn resumed_work_and_running_credit() {
        let c = ClassCounts {
            name: "t2p1".into(),
            tasks: 10,
            prior_total: 10.0,
            done: 6,
            prior_done: 6.0,
            timed: 2,
            prior_timed: 2.0,
            measured: 4.0,
            prior_pre: 4.0,
        };
        // Scale 2: all work 20, done 12 (8 of them restored), so this session did 4 in 4 s.
        let e = estimate(std::slice::from_ref(&c), &[], 4.0);
        assert!((e.done_frac - 0.6).abs() < 1e-9, "{e:?}");
        assert!((e.remaining_s.expect("rate") - 8.0).abs() < 1e-9, "{e:?}");
        assert!(e.measured);
        let r = |elapsed_s| RunningTask {
            class: 0,
            prior: 1.0,
            threads: 2.0,
            elapsed_s,
        };
        let e = estimate(std::slice::from_ref(&c), &[r(0.5)], 4.0);
        assert!((e.done_s - 13.0).abs() < 1e-9, "{e:?}");
        let e = estimate(std::slice::from_ref(&c), &[r(50.0)], 4.0);
        assert!(
            (e.done_s - 14.0).abs() < 1e-9,
            "capped at its expected cost: {e:?}"
        );
    }

    /// The lock-free tracker: begin/end from threads, snapshot sees counts and running tasks.
    #[test]
    fn tracker_counts() {
        let t = EtaTracker::new(
            vec![EtaClass {
                name: "t3p1".into(),
                tasks: 4,
                prior_total_s: 4.0,
                pre_done: 1,
                prior_pre_s: 1.0,
            }],
            2,
        );
        let a = t.begin(0, 0, 1.0, 1);
        let b = t.begin(0, 0, 1.0, 1);
        assert!(a.is_some() && b.is_some() && a != b);
        assert_eq!(t.begin(0, 0, 1.0, 1), None, "two slots only");
        t.end(a, 0, 1.0, 1, 0.5);
        let e = t.snapshot();
        assert!(e.done_frac > 0.49 && e.done_frac <= 1.0, "{e:?}");
        assert!(!e.measured, "one task timed: {e:?}");
        t.end(b, 0, 1.0, 1, 0.5);
        assert!(t.snapshot().done_frac > 0.74);
    }
}
