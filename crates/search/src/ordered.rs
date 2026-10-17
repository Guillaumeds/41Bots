//! Work units made on many threads and taken in order: the pipeline under the label shards
//! (`crate::oracle_labels`) and the start states (`crate::measure::collect_starts_with`).
//!
//! Units 0, 1, 2, … are claimed one at a time by worker threads (each unit as soon as a worker is
//! free, so no thread is left holding a long tail of work) and handed to the caller's `take` on
//! the calling thread in increasing order, whatever order they finish in. What a unit gives
//! depends only on its index, so the sequence `take` sees is the same for any number of threads.
//! The workers run at most a window of units ahead of the next unit `take` needs, so the memory
//! held is bounded however slowly `take` consumes (a slow disk, say), and `take` overlaps the
//! making of later units (writing and hashing a shard while its later rows are still being made).
//!
//! [`pooled`] is the same pipeline with two kinds of work on one pool of threads: *plans* 0, 1,
//! 2, … (a harvest's decisions: a search and a sample) taken in order turn into *items* (the
//! sampled positions' exact labels) appended to one ordered list, and every thread takes the next
//! item of that list, whichever plan it came from. A plan's items are therefore never left to the
//! one thread that made the plan, so the run ends one item's time after the last item is
//! claimed, not one plan's (`crate::oracle_labels`, the harvests).

use std::collections::{BTreeMap, VecDeque};
use std::sync::mpsc;
use std::sync::{Condvar, Mutex, MutexGuard};

struct Claims {
    /// The next unit to claim.
    next: u64,
    /// Units handed to `take` so far (`0..taken`).
    taken: u64,
    /// No more units are claimed.
    stop: bool,
}

/// The window [`ordered`] uses for `threads` workers: enough units for every thread to keep
/// working while one slow unit holds up the order.
pub fn window_for(threads: usize) -> u64 {
    4 * threads.max(1) as u64 + 16
}

fn lock(m: &Mutex<Claims>) -> MutexGuard<'_, Claims> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Units `0..units` (or 0, 1, 2, … without end when `units` is `None`) made by `make` on
/// `threads` worker threads, each worker with its own state from `init`, and handed to `take`
/// on the calling thread in increasing order of index. A worker claims unit `c` only while
/// `c < taken + window` (`taken`: units handed to `take` so far), so at most `window` units are
/// made but not yet taken. `take` returns `Ok(true)` to go on, `Ok(false)` to stop (no unit is
/// claimed after that; units still being made are dropped), or an error, which stops the same
/// way and is returned.
///
/// # Panics
/// If `make` panics (on any worker), the other workers stop claiming and the panic is passed on
/// once they have finished; `take` sees no unit after the gap.
pub fn ordered<S, T: Send>(
    threads: usize,
    units: Option<u64>,
    window: u64,
    init: impl Fn() -> S + Sync,
    make: impl Fn(&mut S, u64) -> T + Sync,
    mut take: impl FnMut(u64, T) -> Result<bool, String>,
) -> Result<(), String> {
    let window = window.max(1);
    let claims = Mutex::new(Claims {
        next: 0,
        taken: 0,
        stop: units == Some(0),
    });
    let moved = Condvar::new();
    let (tx, rx) = mpsc::channel::<(u64, T)>();
    std::thread::scope(|s| {
        let mut workers = Vec::with_capacity(threads.max(1));
        for _ in 0..threads.max(1) {
            let tx = tx.clone();
            let (claims, moved, init, make) = (&claims, &moved, &init, &make);
            workers.push(s.spawn(move || {
                // A panicking worker stops the others (they may be waiting for the window to
                // move past its unit, which never comes), so the panic reaches the caller.
                struct StopOnPanic<'a>(&'a Mutex<Claims>, &'a Condvar);
                impl Drop for StopOnPanic<'_> {
                    fn drop(&mut self) {
                        if std::thread::panicking() {
                            lock(self.0).stop = true;
                            self.1.notify_all();
                        }
                    }
                }
                let _stop = StopOnPanic(claims, moved);
                let mut state = init();
                loop {
                    let c = {
                        let mut g = lock(claims);
                        loop {
                            if g.stop || units.is_some_and(|n| g.next >= n) {
                                return;
                            }
                            if g.next < g.taken + window {
                                break;
                            }
                            g = moved.wait(g).unwrap_or_else(|e| e.into_inner());
                        }
                        g.next += 1;
                        g.next - 1
                    };
                    let t = make(&mut state, c);
                    if tx.send((c, t)).is_err() {
                        return;
                    }
                }
            }));
        }
        drop(tx);
        let mut pending: BTreeMap<u64, T> = BTreeMap::new();
        let mut taken = 0u64;
        let mut res = Ok(());
        'recv: while let Ok((c, t)) = rx.recv() {
            pending.insert(c, t);
            while let Some(t) = pending.remove(&taken) {
                let go = take(taken, t);
                taken += 1;
                let mut g = lock(&claims);
                g.taken = taken;
                match go {
                    Ok(true) => {}
                    Ok(false) => g.stop = true,
                    Err(e) => {
                        g.stop = true;
                        res = Err(e);
                    }
                }
                let done = g.stop || units.is_some_and(|n| taken >= n);
                drop(g);
                if done {
                    break 'recv;
                }
                // The window moved by one unit: one waiting worker may claim it.
                moved.notify_one();
            }
        }
        lock(&claims).stop = true;
        moved.notify_all();
        // Workers still making a unit find the channel closed and stop.
        drop(rx);
        // The first panic, passed on with its own message.
        let mut panicked = None;
        for w in workers {
            if let Err(p) = w.join() {
                panicked.get_or_insert(p);
            }
        }
        if let Some(p) = panicked {
            std::panic::resume_unwind(p);
        }
        res
    })
}

/// [`pooled`]'s shared state.
struct Pool<I> {
    /// The next plan to claim.
    next_plan: u64,
    /// Plans handed to `expand` so far.
    plans_taken: u64,
    /// No plan is claimed any more (`expand` said stop, or the run stops).
    plans_done: bool,
    /// Items not yet claimed, in list order: (index in the list, the item, the list's total
    /// weight through it).
    queue: VecDeque<(u64, I, u64)>,
    /// Items in the list so far.
    items_queued: u64,
    /// Items handed to `take` so far (`0..items_taken`).
    items_taken: u64,
    /// The list's total weight through the last item handed to `take`.
    weight_taken: u64,
    /// The run stops (an error, a panic, or the end): no work is claimed any more.
    stop: bool,
}

fn lock_pool<I>(m: &Mutex<Pool<I>>) -> MutexGuard<'_, Pool<I>> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A finished unit of [`pooled`]'s work, on its way to the calling thread.
enum Made<P, R> {
    Plan(u64, P),
    Item(u64, u64, R),
}

/// What a worker of [`pooled`] does next.
enum Claimed<I> {
    Plan(u64),
    Item(u64, u64, I),
}

/// Plans and items on one pool of `threads` workers (module notes), each worker with its own
/// state from `init`:
/// - plans 0, 1, 2, … are made by `plan` (claimed in order, at most `plan_window` ahead of the
///   next one `expand` needs) and handed to `expand` on the calling thread in increasing order;
///   `expand` returns the plan's items, each with a weight, appended in that order to one list,
///   and whether more plans are wanted (`false`: none is claimed after that; plans still being
///   made are dropped);
/// - the list's items are made by `make` (claimed in list order, an item before any plan, while
///   the list's weight through the item is within `item_window` of the weight already handed to
///   `take`; the next item `take` needs is always claimable, however heavy) and handed to `take`
///   on the calling thread in list order.
///
/// What a plan gives depends only on its index and what an item gives only on the item, so
/// `expand` and `take` see the same sequences for any number of threads. The run ends once
/// `expand` has said stop and every item of the list has been handed to `take`. An error from
/// `expand` or `take` stops the run the same way (work being made is finished and dropped) and is
/// returned.
///
/// # Panics
/// If `plan` or `make` panics (on any worker), the other workers stop claiming and the panic is
/// passed on once they have finished.
#[allow(clippy::too_many_arguments)]
pub fn pooled<S, P: Send, I: Send, R: Send>(
    threads: usize,
    plan_window: u64,
    item_window: u64,
    init: impl Fn() -> S + Sync,
    plan: impl Fn(&mut S, u64) -> P + Sync,
    mut expand: impl FnMut(u64, P) -> Result<(Vec<(I, u64)>, bool), String>,
    make: impl Fn(&mut S, I) -> R + Sync,
    mut take: impl FnMut(u64, R) -> Result<(), String>,
) -> Result<(), String> {
    let plan_window = plan_window.max(1);
    let pool = Mutex::new(Pool {
        next_plan: 0,
        plans_taken: 0,
        plans_done: false,
        queue: VecDeque::new(),
        items_queued: 0,
        items_taken: 0,
        weight_taken: 0,
        stop: false,
    });
    let moved = Condvar::new();
    let (tx, rx) = mpsc::channel::<Made<P, R>>();
    std::thread::scope(|s| {
        let mut workers = Vec::with_capacity(threads.max(1));
        for _ in 0..threads.max(1) {
            let tx = tx.clone();
            let (pool, moved, init, plan, make) = (&pool, &moved, &init, &plan, &make);
            workers.push(s.spawn(move || {
                // A panicking worker stops the others (they may be waiting for work that
                // never comes), so the panic reaches the caller.
                struct StopOnPanic<'a, I>(&'a Mutex<Pool<I>>, &'a Condvar);
                impl<I> Drop for StopOnPanic<'_, I> {
                    fn drop(&mut self) {
                        if std::thread::panicking() {
                            lock_pool(self.0).stop = true;
                            self.1.notify_all();
                        }
                    }
                }
                let _stop = StopOnPanic(pool, moved);
                let mut state = init();
                loop {
                    let job = {
                        let mut g = lock_pool(pool);
                        loop {
                            if g.stop {
                                return;
                            }
                            if let Some(&(i, _, through)) = g.queue.front()
                                && (i == g.items_taken
                                    || through <= g.weight_taken.saturating_add(item_window))
                            {
                                let (i, item, through) = g.queue.pop_front().expect("a front");
                                break Claimed::Item(i, through, item);
                            }
                            if !g.plans_done && g.next_plan < g.plans_taken + plan_window {
                                g.next_plan += 1;
                                break Claimed::Plan(g.next_plan - 1);
                            }
                            if g.plans_done && g.queue.is_empty() {
                                return;
                            }
                            g = moved.wait(g).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    let made = match job {
                        Claimed::Plan(j) => Made::Plan(j, plan(&mut state, j)),
                        Claimed::Item(i, through, item) => {
                            Made::Item(i, through, make(&mut state, item))
                        }
                    };
                    if tx.send(made).is_err() {
                        return;
                    }
                }
            }));
        }
        drop(tx);
        let mut plans: BTreeMap<u64, P> = BTreeMap::new();
        let mut items: BTreeMap<u64, (u64, R)> = BTreeMap::new();
        let (mut plans_taken, mut items_taken) = (0u64, 0u64);
        // The list's total weight so far.
        let mut weight_queued = 0u64;
        let mut plans_done = false;
        let mut res = Ok(());
        'recv: while let Ok(m) = rx.recv() {
            match m {
                Made::Plan(j, p) => {
                    if !plans_done {
                        plans.insert(j, p);
                    }
                    while !plans_done && let Some(p) = plans.remove(&plans_taken) {
                        let (new, go) = match expand(plans_taken, p) {
                            Ok(x) => x,
                            Err(e) => {
                                res = Err(e);
                                break 'recv;
                            }
                        };
                        plans_taken += 1;
                        plans_done = !go;
                        let mut g = lock_pool(&pool);
                        for (item, w) in new {
                            weight_queued = weight_queued.saturating_add(w);
                            let i = g.items_queued;
                            g.queue.push_back((i, item, weight_queued));
                            g.items_queued += 1;
                        }
                        g.plans_taken = plans_taken;
                        g.plans_done = plans_done;
                        drop(g);
                        moved.notify_all();
                    }
                    if plans_done {
                        // Plans made past the stop are not wanted.
                        plans.clear();
                    }
                }
                Made::Item(i, through, r) => {
                    items.insert(i, (through, r));
                    while let Some((through, r)) = items.remove(&items_taken) {
                        if let Err(e) = take(items_taken, r) {
                            res = Err(e);
                            break 'recv;
                        }
                        items_taken += 1;
                        let mut g = lock_pool(&pool);
                        g.items_taken = items_taken;
                        g.weight_taken = through;
                        drop(g);
                        moved.notify_all();
                    }
                }
            }
            let g = lock_pool(&pool);
            if g.plans_done && g.items_taken == g.items_queued {
                break;
            }
        }
        {
            let mut g = lock_pool(&pool);
            g.stop = true;
            g.plans_done = true;
        }
        moved.notify_all();
        // Workers still making something find the channel closed and stop.
        drop(rx);
        // The first panic, passed on with its own message.
        let mut panicked = None;
        for w in workers {
            if let Err(p) = w.join() {
                panicked.get_or_insert(p);
            }
        }
        if let Some(p) = panicked {
            std::panic::resume_unwind(p);
        }
        res
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A unit's value from its index alone, after an uneven amount of work.
    fn unit(c: u64) -> u64 {
        let mut x = c.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        for _ in 0..(c * 7919 % 5000) {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
        }
        x
    }

    #[test]
    fn units_arrive_in_order_for_any_thread_count() {
        let want: Vec<u64> = (0..500).map(unit).collect();
        for threads in [1, 2, 4, 7] {
            for window in [1, 3, window_for(threads)] {
                let mut got = Vec::new();
                ordered(
                    threads,
                    Some(500),
                    window,
                    || (),
                    |_, c| (c, unit(c)),
                    |c, (i, x)| {
                        assert_eq!(c, i);
                        got.push(x);
                        Ok(true)
                    },
                )
                .expect("runs");
                assert_eq!(got, want, "{threads} threads, window {window}");
            }
        }
        let mut none = 0;
        ordered(
            4,
            Some(0),
            8,
            || (),
            |_, c| c,
            |_, _| {
                none += 1;
                Ok(true)
            },
        )
        .expect("runs");
        assert_eq!(none, 0);
    }

    /// The workers never run more than the window ahead of `take`, and each keeps its own state.
    #[test]
    fn the_window_bounds_the_units_in_flight() {
        let made = AtomicU64::new(0);
        let mut taken = 0u64;
        let inits = AtomicU64::new(0);
        ordered(
            4,
            Some(300),
            5,
            || inits.fetch_add(1, Ordering::Relaxed),
            |_, c| {
                made.fetch_add(1, Ordering::SeqCst);
                c
            },
            |c, _| {
                taken = c + 1;
                assert!(
                    made.load(Ordering::SeqCst) <= taken - 1 + 5,
                    "at most the window ahead"
                );
                Ok(true)
            },
        )
        .expect("runs");
        assert_eq!(taken, 300);
        assert_eq!(inits.load(Ordering::Relaxed), 4, "one state per worker");
    }

    /// Without a count, units go on until `take` stops them; an error stops them too and is
    /// returned.
    #[test]
    fn take_stops_the_units() {
        for threads in [1, 3] {
            let mut got = Vec::new();
            ordered(
                threads,
                None,
                6,
                || (),
                |_, c| c * c,
                |c, x| {
                    got.push(x);
                    Ok(c < 40)
                },
            )
            .expect("runs");
            assert_eq!(got, (0..=40).map(|c| c * c).collect::<Vec<_>>());
            let e = ordered(
                threads,
                Some(1000),
                6,
                || (),
                |_, c| c,
                |c, _| {
                    if c == 17 {
                        Err("disk full".to_string())
                    } else {
                        Ok(true)
                    }
                },
            );
            assert_eq!(e, Err("disk full".to_string()));
        }
    }

    /// A plan's items: from 0 to 5 of them (plans 4, 10, 16, … have none), each with an uneven
    /// weight.
    fn plan_items(j: u64) -> Vec<(u64, u64)> {
        (0..(j * 7 + 2) % 6)
            .map(|i| (j * 100 + i, 1 + (j + i) % 3))
            .collect()
    }

    /// One run of [`pooled`]: the plans `expand` saw, the items `take` saw (in order), and the
    /// most weight any made item ran ahead of what `take` had taken when it was claimed.
    fn run_pooled(threads: usize, window: u64, stop_after: u64) -> (Vec<u64>, Vec<u64>, u64) {
        let ahead = std::sync::atomic::AtomicU64::new(0);
        let taken_w = std::sync::atomic::AtomicU64::new(0);
        let (mut plans, mut items) = (Vec::new(), Vec::new());
        // The list's total weight through each item, by item value.
        let through = Mutex::new(BTreeMap::new());
        let mut queued_w = 0u64;
        pooled(
            threads,
            3,
            window,
            || (),
            |_, j| (j, unit(j) % 1000),
            |j, (pj, _)| {
                assert_eq!(j, pj, "plans in order");
                plans.push(j);
                let its = plan_items(j);
                let mut t = through.lock().expect("lock");
                for &(x, w) in &its {
                    queued_w += w;
                    t.insert(x, (queued_w, w));
                }
                Ok((its, j + 1 < stop_after))
            },
            |_, x| {
                let (thr, w) = through.lock().expect("lock")[&x];
                let over = thr.saturating_sub(taken_w.load(Ordering::SeqCst) + w);
                ahead.fetch_max(over, Ordering::SeqCst);
                (x, unit(x) % 7)
            },
            |_, (x, _)| {
                items.push(x);
                let (thr, _) = through.lock().expect("lock")[&x];
                taken_w.store(thr, Ordering::SeqCst);
                Ok(())
            },
        )
        .expect("runs");
        (plans, items, ahead.load(Ordering::SeqCst))
    }

    /// `expand` sees plans 0, 1, 2, … and `take` the items of the list in order, the same for
    /// any thread count and window; no plan is expanded after `expand` says stop; and no item is
    /// claimed more than the window ahead of what `take` has (the next one excepted).
    #[test]
    fn pooled_plans_and_items_arrive_in_order() {
        let want_items: Vec<u64> = (0..40)
            .flat_map(|j| plan_items(j).into_iter().map(|x| x.0))
            .collect();
        for threads in [1, 2, 4, 7] {
            for window in [1, 4, 1000] {
                let (plans, items, ahead) = run_pooled(threads, window, 40);
                assert_eq!(plans, (0..40).collect::<Vec<_>>(), "{threads} threads");
                assert_eq!(items, want_items, "{threads} threads, window {window}");
                assert!(ahead <= window, "{ahead} past a window of {window}");
            }
        }
        // A stop at a plan with no items (plan 4).
        let (plans, items, _) = run_pooled(4, 8, 5);
        assert_eq!(plans, vec![0, 1, 2, 3, 4]);
        assert_eq!(
            items.len(),
            (0..5).map(|j| plan_items(j).len()).sum::<usize>()
        );
    }

    /// An error from `expand` or `take` stops the run and is returned; a panic in a plan or an
    /// item is passed on.
    #[test]
    fn pooled_errors_stop_the_run() {
        for threads in [1, 3] {
            let e = pooled(
                threads,
                4,
                10,
                || (),
                |_, j| j,
                |j, _| {
                    if j == 9 {
                        Err("plan 9".to_string())
                    } else {
                        Ok((vec![(j, 1), (j, 1)], true))
                    }
                },
                |_, x| x,
                |_, _| Ok(()),
            );
            assert_eq!(e, Err("plan 9".to_string()));
            let mut n = 0;
            let e = pooled(
                threads,
                4,
                10,
                || (),
                |_, j| j,
                |j, _| Ok((vec![(j, 1)], true)),
                |_, x| x,
                |i, _| {
                    n += 1;
                    if i == 17 {
                        Err("disk full".to_string())
                    } else {
                        Ok(())
                    }
                },
            );
            assert_eq!(e, Err("disk full".to_string()));
            assert_eq!(n, 18);
        }
        for (threads, in_plan) in [(1, true), (4, true), (1, false), (4, false)] {
            let r = std::panic::catch_unwind(|| {
                pooled(
                    threads,
                    4,
                    10,
                    || (),
                    |_, j| {
                        assert!(!(in_plan && j == 5), "plan 5");
                        j
                    },
                    |j, _| Ok((vec![(j, 1)], j < 100)),
                    |_, x| {
                        assert!(in_plan || x != 7, "item 7");
                        x
                    },
                    |_, _| Ok(()),
                )
            });
            let msg = r.expect_err("the panic is passed on");
            let msg = msg
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| msg.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            assert!(
                msg.contains(if in_plan { "plan 5" } else { "item 7" }),
                "{msg}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "unit 23")]
    fn a_panic_in_a_unit_is_passed_on() {
        let _ = ordered(
            4,
            Some(1000),
            3,
            || (),
            |_, c| {
                assert_ne!(c, 23, "unit 23");
                c
            },
            |_, _| Ok(true),
        );
    }
}
