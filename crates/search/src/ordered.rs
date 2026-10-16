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

use std::collections::BTreeMap;
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
