//! The solvers on positions with turned partnerships (rules 34 and 39-41; the owner's decision of
//! 2026-10-17, docs/APP-ENGINE-CONTRACT.md §7): 6 or 4 seats still in play.
//!
//! The reference is the engine-driven brute force of `common` (clones of the real engine match,
//! which applies the turn itself): the naive solver, the optimised solver with every switch
//! combination and the parallel solver must all give its values. A turned position is never
//! valued by a frontier evaluator (the networks have never seen one): it is searched exactly
//! whatever frontier the solve is given.

mod common;

use common::{brute, leaf};
use engine::game::{Action, Match, Phase, Rules};
use engine::rng::Rng;
use equity::MoneyChance;
use solver::{
    CanonPos, FastOptions, FnFrontier, FrontierCut, MatchCtx, Options, SolveFamily, SolveMem,
    canonical, from_match, solve, solve_cut, solve_fast, solve_fast_cut, solve_fast_par,
    solve_fast_par_cut,
};

/// A random match at trick `k` (after `extra` more cards of it) with `n` partnerships turned at
/// random moments of the hand so far (each turn at the root or before; refused turns are
/// retried with other partnerships), or `None` when this seed's hand ended first.
fn turned_match(seed: u64, k: u8, extra: u8, n: usize) -> Option<Match> {
    let mut rng = Rng::derive(seed, &[u64::from(k), u64::from(extra), n as u64]);
    let mut m = Match::new(rng.next_u64(), Rules::HOUSE);
    // A few whole hands first, so the scores vary.
    let skip = rng.bounded(3) as u32;
    while m.hand_index() < skip {
        if m.phase() == Phase::MatchOver {
            return None;
        }
        let acts = m.legal_actions();
        m.apply(acts[rng.bounded(acts.len() as u64) as usize])
            .expect("legal");
    }
    let h = m.hand_index();
    let mut turned = 0;
    loop {
        if m.phase() == Phase::MatchOver || m.hand_index() != h {
            return None;
        }
        let p = m.public();
        let at_root = p.trick_no == k && p.trick_cards.len() == usize::from(extra);
        if let Phase::AwaitPlay(_) = m.phase() {
            // Turn when at the root (if still short), or now and then before it.
            let due = at_root || (p.trick_no >= 2 && rng.bounded(10) == 0);
            if turned < n && due {
                let team = rng.bounded(4) as u8;
                if m.turn_team(team).is_ok() {
                    turned += 1;
                }
                continue;
            }
            if at_root && turned == n {
                return Some(m);
            }
        }
        if p.trick_no > k || (p.trick_no == k && p.trick_cards.len() > usize::from(extra)) {
            return None;
        }
        let acts = m.legal_actions();
        m.apply(acts[rng.bounded(acts.len() as u64) as usize])
            .expect("legal");
    }
}

/// Positions with `n` partnerships turned at trick `k`: (seed, match).
fn positions(k: u8, n: usize, want: usize, seeds: core::ops::Range<u64>) -> Vec<(u64, Match)> {
    let mut out = Vec::new();
    for seed in seeds {
        let extra = (seed % 3) as u8;
        if let Some(m) = turned_match(seed, k, extra, n) {
            out.push((seed, m));
            if out.len() == want {
                break;
            }
        }
    }
    assert_eq!(
        out.len(),
        want,
        "not enough positions at trick {k} with {n} turned"
    );
    out
}

fn check_against_brute(k: u8, n: usize, want: usize) {
    let l = leaf();
    for (seed, m) in positions(k, n, want, 0..4000) {
        let (pos, ctx, pending) = from_match(&m).expect("position");
        assert!(pending.is_none());
        assert_eq!(pos.turned.count_ones() as usize, n);
        assert_eq!(pos.seats_in_play(), 8 - 2 * n as u8);
        let want = brute(&m, m.hand_index(), &l);
        let naive = solve(&pos, ctx, &l, Options::default());
        assert_eq!(naive.value, Some(want), "seed {seed} trick {k} turned {n}");
        for &(c, v) in &naive.children {
            let mut next = m.clone();
            next.apply(Action::Play(c)).expect("legal");
            assert_eq!(v, brute(&next, m.hand_index(), &l), "seed {seed} child {c}");
        }
        let with_table = solve(
            &pos,
            ctx,
            &l,
            Options {
                tt: true,
                ..Options::default()
            },
        );
        assert_eq!(
            with_table.children, naive.children,
            "seed {seed} naive table"
        );
        for bits in 0..16 {
            let fast = solve_fast(&pos, ctx, &l, FastOptions::default().combo(bits));
            assert_eq!(fast.value, naive.value, "seed {seed} combo {bits}");
            assert_eq!(fast.best, naive.best, "seed {seed} combo {bits}");
            assert_eq!(fast.children, naive.children, "seed {seed} combo {bits}");
        }
        let par = solve_fast_par(&pos, ctx, &l, FastOptions::default(), 3, None);
        assert_eq!(par.children, naive.children, "seed {seed} parallel");
    }
}

#[test]
fn one_card_endgames_with_turned_pairs_match_brute_force() {
    check_against_brute(5, 1, 80);
    check_against_brute(5, 2, 40);
}

#[test]
fn two_card_endgames_with_turned_pairs_match_brute_force() {
    check_against_brute(4, 1, 50);
    check_against_brute(4, 2, 25);
}

#[test]
fn three_card_endgames_with_turned_pairs_match_brute_force() {
    check_against_brute(3, 1, 12);
    check_against_brute(3, 2, 20);
}

#[test]
fn trick_two_with_two_pairs_turned_matches_brute_force() {
    check_against_brute(2, 2, 4);
}

/// The fast solver agrees with the naive solver on bigger turned positions (no brute force):
/// trick 2 with one partnership turned (six seats).
#[test]
fn fast_equals_naive_at_trick_two_with_six_seats() {
    let l = leaf();
    for (seed, m) in positions(2, 1, 2, 0..4000) {
        let (pos, ctx, _) = from_match(&m).expect("position");
        let naive = solve(
            &pos,
            ctx,
            &l,
            Options {
                tt: true,
                ..Options::default()
            },
        );
        let fast = solve_fast(&pos, ctx, &l, FastOptions::default());
        assert_eq!(fast.children, naive.children, "seed {seed}");
        assert_eq!(fast.best, naive.best, "seed {seed}");
    }
}

/// A frontier evaluator that must never be asked.
fn never(_: &CanonPos, _: &MatchCtx) -> MoneyChance {
    panic!("a turned position reached the frontier evaluator (a network)")
}

/// The networks are never used for a turned position: with any frontier, every solver searches
/// it exactly to the end of the hand (the evaluator is never asked, the values are the exact
/// solve's). `FrontierCut::applies` is the one gate (`solver::frontier`).
#[test]
fn turned_positions_never_reach_a_frontier_evaluator() {
    let l = leaf();
    let f = FnFrontier(never);
    let mut checked = 0;
    for (k, n) in [(2u8, 2usize), (3, 1), (3, 2), (4, 1)] {
        for (seed, m) in positions(k, n, 3, 0..4000) {
            let (pos, ctx, _) = from_match(&m).expect("position");
            let exact = solve_fast(&pos, ctx, &l, FastOptions::default());
            for cut_at in (k + 1)..=5 {
                let cut = FrontierCut::new(cut_at, &f);
                let a = solve_cut(&pos, ctx, &l, Options::default(), Some(cut));
                let b = solve_fast_cut(
                    &pos,
                    ctx,
                    &l,
                    FastOptions::default(),
                    Some(cut),
                    &mut SolveMem::new(),
                    None,
                );
                let c = solve_fast_par_cut(
                    &pos,
                    ctx,
                    &l,
                    FastOptions::default(),
                    2,
                    None,
                    &mut SolveMem::new(),
                    Some(cut),
                );
                for s in [&a, &b, &c] {
                    assert_eq!(s.children, exact.children, "seed {seed} cut {cut_at}");
                    assert_eq!(s.stats.frontier, 0, "seed {seed} cut {cut_at}");
                }
                checked += 1;
            }
        }
    }
    assert!(checked >= 20);
}

/// The canonical form (what a network reads) refuses a turned position outright.
#[test]
#[should_panic(expected = "never shown to a network")]
fn the_canonical_form_refuses_a_turned_position() {
    let (_, m) = positions(3, 1, 1, 0..4000).remove(0);
    let (pos, _, _) = from_match(&m).expect("position");
    let _ = canonical(&pos);
}

/// A family of turned roots shares one table exactly as solves from empty tables would, and a
/// root with other turned partnerships is not admitted.
#[test]
fn solve_families_of_turned_roots_are_exact() {
    let l = leaf();
    for (seed, m) in positions(3, 1, 4, 0..4000) {
        let fam = SolveFamily::of_match(&m).expect("a family");
        let (root, ctx, _) = from_match(&m).expect("position");
        assert!(fam.admits(&root, ctx), "seed {seed}");
        let mut other = root;
        let t = (0..4)
            .find(|&t| root.turned & (1 << t) == 0)
            .expect("a pair in play");
        other.turned |= 1 << t;
        assert!(!fam.admits(&other, ctx), "seed {seed}: other turned pairs");
        let mut mem = SolveMem::new();
        mem.share(fam);
        let mut q = m.clone();
        // Walk the hand with the family's memory, comparing each solve with a fresh one.
        for _ in 0..12 {
            let Ok((p, c, _)) = from_match(&q) else { break };
            if q.hand_index() != m.hand_index() {
                break;
            }
            let shared = solver::solve_fast_with(&p, c, &l, FastOptions::default(), &mut mem);
            let fresh = solve_fast(&p, c, &l, FastOptions::default());
            assert_eq!(shared.children, fresh.children, "seed {seed}");
            let a = q.legal_actions()[0];
            q.apply(a).expect("legal");
            if q.phase() == Phase::MatchOver {
                break;
            }
        }
    }
}
