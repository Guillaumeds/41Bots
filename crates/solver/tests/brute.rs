//! Correctness tests for the naive max^n solver (PLAN A2, validation layers 1–3).
//!
//! The brute-force reference below is independent of the solver's code: it drives clones of the
//! real engine [`Match`] through every legal play to the end of the hand, and values each
//! finished line from the engine's own scores, games won and dealer (so the engine's scoring and
//! game-end checks are the reference, not `equity::hand`). It uses the same backup rule and tie
//! order (the mover's team entry is maximised; the lowest card index wins ties).

mod common;

use common::{brute, leaf, position_at, tie_leaf};
use engine::game::Action;
use engine::team_of;
use equity::{MatchState, MoneyChance, ONE};
use solver::{FastOptions, FnLeaf, Options, TieOrder, from_match, solve, solve_fast};

fn check_against_brute(k: u8, seeds: core::ops::Range<u64>) {
    // The tie-heavy table only on the cheap endgames (the engine brute force is slow at trick 3).
    let leaves = if k >= 4 {
        vec![leaf(), tie_leaf()]
    } else {
        vec![leaf()]
    };
    for (li, l) in leaves.iter().enumerate() {
        check_against_brute_with(k, seeds.clone(), l, li);
    }
}

fn check_against_brute_with(k: u8, seeds: core::ops::Range<u64>, l: &common::L, li: usize) {
    for seed in seeds {
        let m = position_at(seed, k);
        let (pos, ctx, pending) = from_match(&m).expect("position");
        assert!(pending.is_none());
        let want = brute(&m, m.hand_index(), l);
        for tt in [false, true] {
            let got = solve(
                &pos,
                ctx,
                l,
                Options {
                    tt,
                    ..Options::default()
                },
            );
            assert_eq!(
                got.value,
                Some(want),
                "seed {seed} trick {k} tt {tt} leaf {li}"
            );
            // Each root child's value equals the brute-force value after that card.
            for &(c, v) in &got.children {
                let mut n = m.clone();
                n.apply(Action::Play(c)).expect("legal");
                assert_eq!(v, brute(&n, m.hand_index(), l), "seed {seed} child {c}");
            }
        }
        // The optimised solver, every switch combination, also matches the brute force.
        for bits in 0..16 {
            let got = solve_fast(&pos, ctx, l, FastOptions::default().combo(bits));
            assert_eq!(
                got.value,
                Some(want),
                "seed {seed} trick {k} combo {bits} leaf {li}"
            );
        }
    }
}

#[test]
fn one_card_endgames_match_brute_force() {
    check_against_brute(5, 0..60);
}

#[test]
fn two_card_endgames_match_brute_force() {
    check_against_brute(4, 100..130);
}

#[test]
fn three_card_endgames_match_brute_force() {
    check_against_brute(3, 200..203);
}

#[test]
fn values_sum_to_exactly_one_and_are_deterministic() {
    let l = leaf();
    for seed in 300..340 {
        let m = position_at(seed, 3 + (seed % 3) as u8);
        let (pos, ctx, _) = from_match(&m).expect("position");
        let a = solve(&pos, ctx, &l, Options::default());
        let b = solve(&pos, ctx, &l, Options::default());
        assert_eq!(a, b, "same input, same answer");
        let v = a.value.expect("no cap");
        assert_eq!(v.0.iter().sum::<u64>(), ONE);
        for (_, c) in &a.children {
            assert_eq!(c.0.iter().sum::<u64>(), ONE);
        }
        // The chosen card is the first (lowest index) of the tied best for the mover's team.
        let team = team_of(pos.to_act());
        let top = v.get(team);
        let first = a.children.iter().find(|(_, c)| c.get(team) == top);
        assert_eq!(first.map(|x| x.0), a.best);
    }
}

#[test]
fn transposition_table_gives_identical_answers() {
    let l = leaf();
    for seed in 400..430 {
        let m = position_at(seed, 3 + (seed % 3) as u8);
        let (pos, ctx, _) = from_match(&m).expect("position");
        for order in [TieOrder::Ascending, TieOrder::Descending] {
            let off = solve(
                &pos,
                ctx,
                &l,
                Options {
                    order,
                    ..Options::default()
                },
            );
            let on = solve(
                &pos,
                ctx,
                &l,
                Options {
                    tt: true,
                    order,
                    ..Options::default()
                },
            );
            assert_eq!(off.value, on.value);
            assert_eq!(off.best, on.best);
            assert_eq!(off.children, on.children);
            assert!(on.stats.nodes <= off.stats.nodes);
        }
    }
}

#[test]
fn a_finished_match_is_certain_money() {
    // A team with one game won that wins this game wins the match: the solver must return the
    // certain vector, not a table lookup. Search for a world where that happens at the root.
    let l = FnLeaf(|_: &MatchState| MoneyChance::even());
    for seed in 500..800 {
        let m = position_at(seed, 5);
        let (mut pos, mut ctx, _) = from_match(&m).expect("position");
        // Every team on 40 with one game: whoever takes the last trick wins the match.
        ctx.games = [1; 4];
        pos.hand.scores = [40; 4];
        pos.hand.flags = [false; 4];
        let s = solve(&pos, ctx, &l, Options::default());
        let v = s.value.expect("solved");
        assert!(v.0.contains(&ONE), "a certain winner: {v:?}");
    }
}
