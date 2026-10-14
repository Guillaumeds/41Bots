//! The M2 gate for the optimised solver (PLAN A2 and §13 M2): the four validation layers.
//! (a) engine parity of the solver's own play-and-score step on random play-outs;
//! (b) search vs brute force on 1-, 2-, 3-card endgames (in `brute.rs`, every switch combination);
//! (c) every speed-up on/off gives identical root values, root card values and chosen card,
//!     against the naive solver, under both tie orders and with two leaf tables;
//! (d) exhaustive best response at the root from mid-hand engine states.
//! Plus equivalent-card decks, a tiny table that forces replacement, ties and root flips, and the
//! forced-denial check. Long sweeps are `#[ignore]`d; the release binary `solver-identity` runs
//! them at scale.

mod common;

use common::{engine_leaf_value, leaf, position_at, position_mid, tie_leaf};
use engine::cards::{Bits, Card, Suit, bit, parse};
use engine::game::{Action, Match, Phase};
use engine::rng::Rng;
use engine::team_of;
use equity::{MatchState, MoneyChance};
use solver::denial::forced_denial;
use solver::equiv::Classes;
use solver::{
    FastOptions, Leaf, Options, Position, Solution, Step, TieOrder, from_match, solve, solve_fast,
};

fn same(a: &Solution, b: &Solution, what: &str) {
    assert_eq!(a.value, b.value, "{what}: value");
    assert_eq!(a.best, b.best, "{what}: chosen card");
    assert_eq!(a.children, b.children, "{what}: root card values");
}

/// Naive (table on; table on = off is tested in brute.rs) vs every combination of switches.
fn identity(m: &Match, l: &common::L, combos: &[u8], label: &str) {
    let (pos, ctx, pending) = from_match(m).expect("position");
    assert!(pending.is_none());
    for order in [TieOrder::Ascending, TieOrder::Descending] {
        let naive = solve(
            &pos,
            ctx,
            l,
            Options {
                tt: true,
                order,
                ..Options::default()
            },
        );
        for &bits in combos {
            let f = solve_fast(
                &pos,
                ctx,
                l,
                FastOptions {
                    order,
                    ..FastOptions::default()
                }
                .combo(bits),
            );
            same(&naive, &f, &format!("{label} order {order:?} combo {bits}"));
        }
    }
}

const ALL: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

#[test]
fn identity_tricks_4_and_5_every_combination() {
    for seed in 0..120 {
        for k in [4, 5] {
            for extra in [0, 3] {
                let m = position_mid(seed, k, extra);
                identity(&m, &leaf(), &ALL, &format!("seed {seed} trick {k}+{extra}"));
                identity(
                    &m,
                    &tie_leaf(),
                    &ALL,
                    &format!("tie seed {seed} trick {k}+{extra}"),
                );
            }
        }
    }
}

#[test]
fn identity_trick_3_sample() {
    for seed in 0..3 {
        let m = position_mid(seed, 3, (seed % 2) as u8 * 2);
        // All on, each switch alone, each switch off.
        identity(
            &m,
            &leaf(),
            &[15, 1, 2, 4, 8, 14, 13, 11, 7],
            &format!("seed {seed}"),
        );
    }
}

#[test]
#[ignore = "long: run with --release -- --ignored, or use the solver-identity binary"]
fn identity_trick_3_every_combination_long() {
    for seed in 1000..1040 {
        let m = position_mid(seed, 3, (seed % 4) as u8);
        identity(&m, &leaf(), &ALL, &format!("seed {seed}"));
        identity(&m, &tie_leaf(), &[15, 0], &format!("tie seed {seed}"));
    }
}

#[test]
fn tiny_table_forces_replacement_without_changing_answers() {
    for seed in 0..6 {
        let m = position_at(seed, 3);
        let (pos, ctx, _) = from_match(&m).expect("position");
        let naive = solve(&pos, ctx, &leaf(), Options::default());
        let tiny = solve_fast(
            &pos,
            ctx,
            &leaf(),
            FastOptions {
                tt_mb: 0,
                ..FastOptions::default()
            },
        );
        same(&naive, &tiny, &format!("tiny table seed {seed}"));
        assert!(tiny.stats.tt_replaced > 0 || tiny.stats.tt_entries <= 4);
    }
}

// ---------- (a) engine parity on random play-outs ----------

fn same_position(a: &Position, b: &Position) -> bool {
    let n = usize::from(a.n_played);
    a.hands == b.hands
        && a.trump == b.trump
        && a.leader == b.leader
        && a.n_played == b.n_played
        && a.trick[..n] == b.trick[..n]
        && a.hand == b.hand
        && a.best == b.best
}

#[test]
fn engine_parity_on_random_playouts() {
    let mut rng = Rng::new(77);
    let mut checked = [0u32; 3];
    for _ in 0..40 {
        let mut m = Match::new(rng.next_u64(), engine::game::Rules::HOUSE);
        while m.phase() != Phase::MatchOver {
            let acts = m.legal_actions();
            let a = acts[rng.bounded(acts.len() as u64) as usize];
            let before = match (m.phase(), from_match(&m)) {
                (Phase::AwaitPlay(_), Ok((p, ctx, None))) => Some((p, ctx)),
                _ => None,
            };
            let hand0 = m.hand_index();
            m.apply(a).expect("legal");
            let Some((p, ctx)) = before else { continue };
            match p.play(a.card()) {
                Step::Next(q) => {
                    assert_eq!(m.hand_index(), hand0);
                    if let Ok((e, _, None)) = from_match(&m) {
                        assert!(same_position(&q, &e), "{q:?} vs {e:?}");
                    }
                    checked[0] += 1;
                }
                Step::GameWon(t) => {
                    let mut g = ctx.games;
                    g[usize::from(t)] += 1;
                    assert_eq!(m.games_won(), g);
                    checked[1] += 1;
                }
                Step::HandOver(s) => {
                    assert_ne!(m.hand_index(), hand0);
                    assert_eq!(m.games_won(), ctx.games);
                    assert_eq!(m.scores(), s);
                    checked[2] += 1;
                }
            }
        }
    }
    assert!(checked.iter().all(|&c| c > 20), "{checked:?}");
}

// ---------- (d) best response at the root ----------

/// Play on from `m` with every seat following the optimised solver until the hand ends; the
/// engine's own end state is then valued.
fn follow<Lf: Leaf>(mut m: Match, hand0: u32, l: &Lf) -> MoneyChance {
    while m.phase() != Phase::MatchOver && m.hand_index() == hand0 {
        let (pos, ctx, _) = from_match(&m).expect("position");
        let s = solve_fast(&pos, ctx, l, FastOptions::default());
        m.apply(Action::Play(s.best.expect("solved")))
            .expect("legal");
    }
    engine_leaf_value(&m, l)
}

#[test]
fn best_response_at_the_root() {
    for (seed, k, extra) in [
        (0, 4, 0),
        (1, 4, 2),
        (2, 4, 5),
        (3, 3, 0),
        (4, 3, 1),
        (5, 4, 7),
    ]
    .into_iter()
    .chain((10..40).map(|s| (s, 4 + (s % 2) as u8, (s % 8) as u8)))
    {
        let m = position_mid(seed, k, extra);
        let (pos, ctx, _) = from_match(&m).expect("position");
        let l = leaf();
        let root = solve_fast(&pos, ctx, &l, FastOptions::default());
        let team = team_of(pos.to_act());
        let v = root.value.expect("solved");
        for &(c, cv) in &root.children {
            let mut n = m.clone();
            n.apply(Action::Play(c)).expect("legal");
            let real = follow(n, m.hand_index(), &l);
            // The solver's value of each card is what that card really earns when everybody
            // then follows the solver ...
            assert_eq!(cv, real, "seed {seed} card {c}");
            // ... and no alternative card does better for the root team.
            assert!(v.get(team) >= real.get(team), "seed {seed} card {c}");
        }
    }
}

// ---------- ties and root flips (A2) ----------

#[test]
fn ties_are_common_and_reversing_the_order_keeps_the_mover_best_set() {
    let l = leaf();
    let (mut roots, mut tied, mut flips) = (0, 0, 0);
    for seed in 0..200 {
        let m = position_mid(seed, 4, (seed % 8) as u8);
        let (pos, ctx, _) = from_match(&m).expect("position");
        if pos.legal().count_ones() < 2 {
            continue;
        }
        let a = solve_fast(&pos, ctx, &l, FastOptions::default());
        let d = solve_fast(
            &pos,
            ctx,
            &l,
            FastOptions {
                order: TieOrder::Descending,
                ..FastOptions::default()
            },
        );
        roots += 1;
        tied += u32::from(a.root_tied >= 2);
        flips += u32::from(a.best != d.best);
        // A flip can only happen at a tied root.
        if a.best != d.best {
            assert!(a.root_tied >= 2);
        }
    }
    // The spike measured ~75% of trick-4 roots tied with the real table; the synthetic table must
    // show plenty of ties too, or the tie handling is not being exercised.
    assert!(tied * 4 >= roots, "tied {tied} of {roots}");
    assert!(flips > 0 && flips <= tied, "flips {flips}, tied {tied}");
}

// ---------- equivalent cards: positive and negative decks ----------

fn set(s: &str) -> Bits {
    s.split_whitespace()
        .map(|n| bit(parse(n).unwrap_or_else(|| panic!("card {n}"))))
        .fold(0, |a, b| a | b)
}

/// Are `a` and `b` (both in `hand`) in one group, given the other live cards?
fn merged(trump: Suit, hand: &str, others: &str, a: &str, b: &str) -> bool {
    let h = set(hand);
    let live = h | set(others);
    let mut g = [0; 16];
    let n = Classes::new(trump).groups(h, live, &mut g);
    // Every card of the hand is in exactly one group.
    assert_eq!(g[..n].iter().fold(0, |x, &y| x | y), h);
    assert_eq!(
        g[..n].iter().map(|x| x.count_ones()).sum::<u32>(),
        h.count_ones()
    );
    let (a, b) = (bit(parse(a).expect("a")), bit(parse(b).expect("b")));
    g[..n].iter().any(|&x| x & a != 0 && x & b != 0)
}

#[test]
fn equivalent_cards_positive_decks() {
    use Suit::*;
    // Adjacent ordinary trumps (red: Q then 10).
    assert!(merged(Diamonds, "QD 10D 3S", "9D KD", "QD", "10D"));
    // Top trumps next to each other: J and Joker; Joker and A♥.
    assert!(merged(Diamonds, "JD JK", "AD 2D", "JD", "JK"));
    assert!(merged(Diamonds, "JK AH", "5D KD", "JK", "AH"));
    // Black plain order: J A 2 (the ace sits between the jack and the 2).
    assert!(merged(Diamonds, "AC 2C", "JC 3C", "AC", "2C"));
    // Red plain order: 2 then A (ace lowest), hearts trumps aside: diamonds when clubs are trumps.
    assert!(merged(Clubs, "2D AD", "3D", "2D", "AD"));
    // Cards gone from play make cards adjacent: Q♣ already played.
    assert!(merged(Diamonds, "KC JC", "10C", "KC", "JC"));
    // Hearts trumps: Joker and A♥ (both top trumps).
    assert!(merged(Hearts, "JK AH", "KH", "JK", "AH"));
    // Black trumps: 2 then 3 (black numbers run 2 up to 10).
    assert!(merged(Spades, "2S 3S", "4S", "2S", "3S"));
}

#[test]
fn equivalent_cards_negative_decks() {
    use Suit::*;
    // The 5 of trumps is never merged (its bonus is immediate), not even with the J.
    assert!(!merged(Diamonds, "5D JD", "", "5D", "JD"));
    // Top trump vs ordinary trump: the A♥ may be held back against the ace of trumps' lead, the
    // ace of trumps may not.
    assert!(!merged(Diamonds, "AH AD", "", "AH", "AD"));
    // Hearts trumps: the A♥ (top) and the K♥ (highest ordinary trump) are not merged.
    assert!(!merged(Hearts, "AH KH", "", "AH", "KH"));
    // A live card between them, in another hand ...
    assert!(!merged(Diamonds, "QD 9D", "10D", "QD", "9D"));
    // ... or on the table in the current trick (the Joker between the J and the A♥).
    assert!(!merged(Diamonds, "JD AH", "JK", "JD", "AH"));
    // Different suits.
    assert!(!merged(Diamonds, "KC KS", "", "KC", "KS"));
    // Black plain: the ace is above the 2, so A and 10 are not adjacent while 2..9 are live.
    assert!(!merged(Diamonds, "AS 10S", "5S", "AS", "10S"));
    // Red vs black number order: in red the 10 is high; 10♥ and 2♥ are not adjacent with 5♥ live.
    assert!(!merged(Diamonds, "10H 2H", "5H", "10H", "2H"));
    // The A♥ is never plain: with spades trumps, it is not merged with the 2♥.
    assert!(!merged(Spades, "AH 2H", "", "AH", "2H"));
}

// ---------- forced denial (test-only scenario check) ----------

/// Brute force by engine clones: can the others deny `team` every remaining trick?
fn deny_brute(m: &Match, team: u8, hand0: u32, tricks0: [u8; 4]) -> bool {
    if m.phase() == Phase::MatchOver || m.hand_index() != hand0 {
        unreachable!("only used with game ends excluded");
    }
    let p = m.public();
    if p.tricks_won[usize::from(team)] > tricks0[usize::from(team)] {
        return false;
    }
    let Phase::AwaitPlay(seat) = m.phase() else {
        unreachable!()
    };
    if p.trick_winners.len() == 5 {
        return true;
    }
    let cards: Vec<Card> = m.legal_actions().iter().map(|a| a.card()).collect();
    let results = cards.iter().map(|&c| {
        let mut n = m.clone();
        n.apply(Action::Play(c)).expect("legal");
        if n.hand_index() != hand0 || n.phase() == Phase::MatchOver {
            // Hand over: decided by the last trick.
            let won = n.public();
            let _ = won;
            return last_trick_not_won(m, c, team);
        }
        deny_brute(&n, team, hand0, tricks0)
    });
    let r: Vec<bool> = results.collect();
    if team_of(seat) == team {
        r.iter().all(|&x| x)
    } else {
        r.iter().any(|&x| x)
    }
}

fn last_trick_not_won(m: &Match, c: Card, team: u8) -> bool {
    let p = m.public();
    let mut plays: Vec<Card> = p.trick_cards.iter().map(|&(_, x)| x).collect();
    plays.push(c);
    let w = engine::rules::trick_winner(&plays, p.trump.expect("trumps"));
    team_of(
        p.trick_cards
            .first()
            .map_or(0, |&(s, _)| s)
            .wrapping_add(w as u8)
            % 8,
    ) != team
}

#[test]
fn forced_denial_matches_engine_brute_force() {
    let mut yes = 0;
    let mut no = 0;
    for seed in 0..80 {
        let k = 4 + (seed % 2) as u8;
        let m = position_mid(seed, k, (seed % 5) as u8);
        let (pos, _, _) = from_match(&m).expect("position");
        let p = m.public();
        for team in 0..4u8 {
            // The engine brute force needs no game end mid-hand: only compare when nobody can
            // reach 41 (all scores low) — otherwise skip.
            if p.scores.iter().any(|&s| s >= 15) {
                continue;
            }
            let want = deny_brute(&m, team, m.hand_index(), p.tricks_won);
            assert_eq!(forced_denial(&pos, team), want, "seed {seed} team {team}");
            if want {
                yes += 1;
            } else {
                no += 1;
            }
        }
    }
    assert!(yes > 5 && no > 5, "yes {yes} no {no}");
}

#[test]
fn the_holder_of_the_five_of_trumps_cannot_be_denied() {
    for seed in 0..60 {
        let m = position_at(seed, 4);
        let (pos, _, _) = from_match(&m).expect("position");
        let five = engine::rules::five_of(pos.trump);
        if let Some(s) = (0..8u8).find(|&s| pos.hands[usize::from(s)] & bit(five) != 0) {
            assert!(!forced_denial(&pos, team_of(s)), "seed {seed}");
        }
    }
}

#[test]
fn match_state_helper_is_consistent() {
    // Guards the helper used by the best-response test.
    let m = position_at(5, 5);
    let v = engine_leaf_value(&m, &leaf());
    let s = MatchState {
        scores: m.scores(),
        games: m.games_won(),
        dealer_team: m.dealer() % 4,
    };
    assert_eq!(v, leaf().value(&s));
}
