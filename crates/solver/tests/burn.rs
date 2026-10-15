//! A pending robber's burn searched inside the solve (`Position::rob`): rules (H3/H4), the
//! naive solver against brute-force decomposition, and the fast and parallel solvers against the
//! naive one under every switch combination, both tie orders and two leaf tables.
//!
//! The deals are small (tricks 3–5 of random play, with a seat still to act in the current
//! trick given a pending rob of a card out of play), so the whole tree, burns included, is cheap.

mod common;

use common::{L, leaf, position_mid, tie_leaf};
use engine::cards::{Bits, Card, JOKER, Suit, bit, has, parse};
use engine::game::{Match, Phase, Rules};
use engine::rng::Rng;
use engine::rules::ace_of_trumps;
use engine::team_of;
use solver::{
    FastOptions, Options, Position, Rob, Solution, Step, TieOrder, from_match, solve, solve_fast,
    solve_fast_par,
};

fn same(a: &Solution, b: &Solution, what: &str) {
    assert!(!a.timed_out && !b.timed_out, "{what}: timed out");
    assert_eq!(a.value, b.value, "{what}: value");
    assert_eq!(a.best, b.best, "{what}: chosen card");
    assert_eq!(a.children, b.children, "{what}: root card values");
}

/// A random-play position with a pending rob: the robber is a seat still to act in the current
/// trick after the root mover (`offset` seats later), taking a card out of play. With
/// `ace = true` the robber is also given the ace of trumps when it is out of play (as a real
/// robber holds it), so it can never be burned.
fn with_rob(seed: u64, k: u8, extra: u8, ace: bool) -> Option<(Position, solver::MatchCtx)> {
    let m = position_mid(seed, k, extra);
    let (mut pos, ctx, pending) = from_match(&m).ok()?;
    assert!(pending.is_none());
    let room = 8 - u64::from(pos.n_played);
    if room < 2 {
        return None;
    }
    let mut rng = Rng::derive(seed, &[u64::from(k), u64::from(extra)]);
    let offset = 1 + rng.bounded(room - 1) as u8;
    let seat = (pos.to_act() + offset) % 8;
    let live = pos.live();
    let five = engine::rules::five_of(pos.trump);
    let dead: Vec<Card> = (0..=JOKER)
        .filter(|&c| !has(live, c) && c != five)
        .collect();
    let table = dead[rng.bounded(dead.len() as u64) as usize];
    let a = ace_of_trumps(pos.trump);
    if ace && a != table && !has(live, a) {
        // Swap one of the robber's cards for the ace of trumps (the robber's hand stays
        // the same size).
        let h = pos.hands[usize::from(seat)];
        let c = h.trailing_zeros() as Card;
        pos.hands[usize::from(seat)] = (h & !bit(c)) | bit(a);
    }
    pos.rob = Some(Rob { seat, card: table });
    Some((pos, ctx))
}

/// The naive solver's value of a position whose robber is to act: the robber's team's best over
/// every legal burn of the root solve after it (first strictly best in the tie order), with the
/// root mover's value read from that solve. Independent of the burn node.
fn decomposed(p: &Position, ctx: solver::MatchCtx, l: &L, order: TieOrder) -> equity::MoneyChance {
    let (_, burns) = p.burn_choice().expect("robber to act");
    let team = team_of(p.to_act());
    let mut best: Option<equity::MoneyChance> = None;
    let mut bs: Vec<Card> = engine::cards::cards(burns).collect();
    if order == TieOrder::Descending {
        bs.reverse();
    }
    for b in bs {
        let s = solve(
            &p.burnt(b),
            ctx,
            l,
            Options {
                order,
                ..Options::default()
            },
        );
        let v = s.value.expect("no cap");
        if best.is_none_or(|x| v.get(team) > x.get(team)) {
            best = Some(v);
        }
    }
    best.expect("a burn")
}

/// The naive solver with burn nodes equals the decomposition when the robber acts right after
/// the root mover: each root card's value is the robber's best burn's root solve.
#[test]
fn naive_burn_node_equals_decomposition() {
    let mut checked = 0;
    for seed in 0..60 {
        for k in [4, 5] {
            let m = position_mid(seed, k, (seed % 4) as u8);
            let Ok((mut pos, ctx, _)) = from_match(&m) else {
                continue;
            };
            if pos.n_played >= 7 {
                continue;
            }
            let seat = (pos.to_act() + 1) % 8;
            let five = engine::rules::five_of(pos.trump);
            let dead = (0..=JOKER)
                .find(|&c| !has(pos.live(), c) && c != five)
                .expect("dead card");
            pos.rob = Some(Rob { seat, card: dead });
            for (li, l) in [leaf(), tie_leaf()].iter().enumerate() {
                for order in [TieOrder::Ascending, TieOrder::Descending] {
                    for tt in [false, true] {
                        let s = solve(
                            &pos,
                            ctx,
                            l,
                            Options {
                                tt,
                                order,
                                ..Options::default()
                            },
                        );
                        for &(c, v) in &s.children {
                            let Step::Next(q) = pos.play(c) else {
                                unreachable!("a mid-trick play goes on")
                            };
                            assert_eq!(
                                v,
                                decomposed(&q, ctx, l, order),
                                "seed {seed} trick {k} leaf {li} {order:?} tt {tt} card {c}"
                            );
                        }
                    }
                }
            }
            checked += 1;
        }
    }
    assert!(checked >= 60, "{checked}");
}

const ALL: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

fn identity(pos: &Position, ctx: solver::MatchCtx, l: &L, combos: &[u8], label: &str) {
    for order in [TieOrder::Ascending, TieOrder::Descending] {
        let naive = solve(
            pos,
            ctx,
            l,
            Options {
                tt: true,
                order,
                ..Options::default()
            },
        );
        for &bits in combos {
            let opts = FastOptions {
                order,
                ..FastOptions::default()
            }
            .combo(bits);
            let f = solve_fast(pos, ctx, l, opts);
            same(&naive, &f, &format!("{label} {order:?} combo {bits}"));
            if bits == 15 {
                for threads in [2, 4] {
                    let p = solve_fast_par(pos, ctx, l, opts, threads, None);
                    same(&naive, &p, &format!("{label} {order:?} par {threads}"));
                }
            }
        }
    }
}

/// Fast (every switch combination) and parallel solves equal the naive solver with burn nodes.
#[test]
fn identity_with_pending_burn_tricks_4_and_5() {
    let mut n = 0;
    for seed in 0..80 {
        for k in [4, 5] {
            for extra in [0, 2] {
                let Some((pos, ctx)) = with_rob(seed, k, extra, seed % 2 == 0) else {
                    continue;
                };
                identity(
                    &pos,
                    ctx,
                    &leaf(),
                    &ALL,
                    &format!("seed {seed} t{k}+{extra}"),
                );
                identity(
                    &pos,
                    ctx,
                    &tie_leaf(),
                    &ALL,
                    &format!("tie {seed} t{k}+{extra}"),
                );
                n += 1;
            }
        }
    }
    assert!(n >= 200, "{n}");
}

#[test]
fn identity_with_pending_burn_trick_3_sample() {
    for seed in 0..3 {
        let (pos, ctx) = with_rob(seed, 3, 1, true).expect("room in trick 3");
        identity(
            &pos,
            ctx,
            &leaf(),
            &[15, 1, 2, 4, 8, 14, 13, 11, 7],
            &format!("seed {seed}"),
        );
    }
}

/// H3: any card but the ace of trumps may be burned, including plain aces, the A♥ when hearts
/// are not trumps, the Joker and the card just taken; the A♥ cannot be burned when hearts are
/// trumps. The burn choice holds the taken card, and a burn leaves the robber's hand size.
#[test]
fn burn_choice_follows_h3() {
    let m = position_mid(5, 4, 1);
    let (base, _, _) = from_match(&m).expect("position");
    let seat = (base.to_act() + 1) % 8;
    let set = |s: &str| -> Bits { s.split(' ').map(|x| bit(parse(x).expect("card"))).sum() };
    for (trump, hand, taken, want) in [
        (Suit::Clubs, "AC AD AH JK", "5C", "AD AH JK 5C"),
        (Suit::Hearts, "AC AD AH JK", "5C", "AC AD JK 5C"),
        (Suit::Spades, "AH 2S", "AS", "AH 2S"),
        (Suit::Diamonds, "JK 3D", "AC", "JK 3D AC"),
    ] {
        let mut p = base;
        p.trump = trump;
        p.hands[usize::from(seat)] = set(hand);
        p.rob = Some(Rob {
            seat,
            card: parse(taken).expect("card"),
        });
        assert_eq!(p.burn_choice(), None, "not the robber's turn yet");
        let Step::Next(q) = p.play(p.legal().trailing_zeros() as Card) else {
            unreachable!()
        };
        let (h, burns) = q.burn_choice().expect("robber to act");
        assert_eq!(h, set(hand) | set(taken), "{trump:?}: takes the card");
        assert_eq!(burns, set(want), "{trump:?} {hand} + {taken}");
        assert!(!has(burns, ace_of_trumps(trump)));
        for b in engine::cards::cards(burns) {
            let r = q.burnt(b);
            assert_eq!(r.rob, None);
            assert_eq!(r.hands[usize::from(seat)], h & !bit(b));
        }
    }
}

/// H4/H2: from a real trick-1 match where a seat after the leader holds the ace of trumps of an
/// ordinary turned card (including one turned after the dealer's Joker take, whose earlier burn
/// stays out of play), the pending rob is that seat's, of the table card, and its burns exclude
/// only the ace of trumps.
#[test]
fn pending_rob_from_real_matches() {
    let mut found = 0;
    let mut after_joker = 0;
    for seed in 0..4000u64 {
        let mut m = Match::new(seed, Rules::HOUSE);
        // Resolve the dealer's at-once burns (turned ace or Joker) with the first legal burn.
        while let Phase::AwaitBurn(_) = m.phase() {
            let a = m.legal_actions()[0];
            m.apply(a).expect("legal");
        }
        let Ok((mut pos, _, Some(s))) = from_match(&m) else {
            continue;
        };
        assert_ne!(
            s,
            pos.to_act(),
            "the leader's rob is a burn phase, not pending"
        );
        let p = m.public();
        let table = p.table_card.expect("turned card");
        let ace = ace_of_trumps(pos.trump);
        assert!(has(pos.hands[usize::from(s)], ace));
        pos.rob = Some(Rob {
            seat: s,
            card: table,
        });
        // Play the leader's and later seats' first cards until the robber is to act.
        while pos.to_act() != s {
            let Step::Next(q) = pos.play(pos.legal().trailing_zeros() as Card) else {
                unreachable!()
            };
            pos = q;
        }
        let (h, burns) = pos.burn_choice().expect("robber to act");
        assert_eq!(burns, h & !bit(ace), "seed {seed}");
        if p.burn_counts.iter().any(|&c| c > 0) {
            after_joker += 1;
        }
        found += 1;
    }
    assert!(found >= 50, "{found}");
    assert!(after_joker >= 1, "no Joker-then-ordinary case met");
}
