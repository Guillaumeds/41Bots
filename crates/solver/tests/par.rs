//! The parallel single-world solve (`solve_fast_par`) must give exactly `solve_fast`'s root
//! value, every root card's value and chosen card, for any thread count, both tie orders,
//! several switch combinations, two leaf tables and a tiny shared table (forces replacement), and
//! with its memory kept from earlier solves.
//! Trick 2 is `#[ignore]`d (slow in debug); `solver-identity --par-threads T` runs it at scale.

mod common;

use common::{L, leaf, position_mid, tie_leaf};
use solver::{
    FastOptions, Solution, SolveMem, TieOrder, from_match, solve_fast, solve_fast_par,
    solve_fast_par_with,
};

fn same(a: &Solution, b: &Solution, what: &str) {
    assert!(!a.timed_out && !b.timed_out, "{what}: timed out");
    assert_eq!(a.value, b.value, "{what}: value");
    assert_eq!(a.best, b.best, "{what}: chosen card");
    assert_eq!(a.children, b.children, "{what}: root card values");
    assert_eq!(a.root_tied, b.root_tied, "{what}: root ties");
}

fn check(seed: u64, k: u8, extra: u8, l: &L, combos: &[u8], tt_mbs: &[usize]) {
    let m = position_mid(seed, k, extra);
    let (pos, ctx, pending) = from_match(&m).expect("position");
    assert!(pending.is_none());
    for order in [TieOrder::Ascending, TieOrder::Descending] {
        for &bits in combos {
            for &tt_mb in tt_mbs {
                let opts = FastOptions {
                    order,
                    tt_mb,
                    ..FastOptions::default()
                }
                .combo(bits);
                let seq = solve_fast(&pos, ctx, l, opts);
                for threads in [1, 2, 4, 7] {
                    let par = solve_fast_par(&pos, ctx, l, opts, threads, None);
                    same(
                        &seq,
                        &par,
                        &format!(
                            "seed {seed} trick {k}+{extra} {order:?} combo {bits} tt {tt_mb} threads {threads}"
                        ),
                    );
                }
            }
        }
    }
}

/// One [`SolveMem`] kept across parallel solves (other positions, thread counts and table sizes,
/// so the shared table is sometimes reset and sometimes rebuilt) still gives exactly
/// `solve_fast`'s answers.
#[test]
fn par_identity_with_reused_memory() {
    let mut mem = SolveMem::new();
    for seed in 0..8 {
        for (k, extra) in [(4, 0), (3, 2), (5, 3)] {
            let m = position_mid(seed, k, extra);
            let (pos, ctx, pending) = from_match(&m).expect("position");
            assert!(pending.is_none());
            for (threads, tt_mb) in [(3, 1), (4, 64), (4, 64), (2, 64), (7, 1)] {
                let opts = FastOptions {
                    tt_mb,
                    ..FastOptions::default()
                };
                let seq = solve_fast(&pos, ctx, &leaf(), opts);
                let par = solve_fast_par_with(&pos, ctx, &leaf(), opts, threads, None, &mut mem);
                same(
                    &seq,
                    &par,
                    &format!("seed {seed} trick {k}+{extra} tt {tt_mb} threads {threads}"),
                );
            }
        }
    }
}

#[test]
fn par_identity_tricks_4_and_5() {
    for seed in 0..30 {
        for k in [4, 5] {
            for extra in [0, 3] {
                check(seed, k, extra, &leaf(), &[11, 15, 0, 9], &[1, 64]);
                check(seed, k, extra, &tie_leaf(), &[11, 15], &[1]);
            }
        }
    }
}

#[test]
fn par_identity_trick_3() {
    for seed in 0..6 {
        check(seed, 3, 0, &leaf(), &[11], &[1, 64]);
        check(seed, 3, 2, &tie_leaf(), &[11], &[64]);
    }
}

#[test]
#[ignore = "slow in debug builds; run with --release -- --ignored"]
fn par_identity_trick_2() {
    for seed in 0..2 {
        check(seed, 2, 0, &leaf(), &[11], &[256]);
    }
}
