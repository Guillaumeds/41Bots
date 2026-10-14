//! Whole matches, driven directly through the engine the way the table (docs/M1-SPEC.md §3.1)
//! will drive them: each bot gets its seat seed by the rule of §1, then only its own `SeatView`
//! and the legal actions.
//!
//! - determinism: the same seeds give identical action sequences;
//! - legality: every action either bot returns is one of the legal actions;
//! - the 1,000-match diagnostic: `rules-v1` in one team's two seats against six `random` seats.
//!   Run with `cargo test -p bots -- --nocapture` to see the money figure.

use bot::Bot;
use engine::game::{Action, Match, Rules};
use engine::rng::Rng;
use engine::{SEATS, TEAMS, team_of};

/// The seat seed rule of docs/M1-SPEC.md §1.
fn seat_seed(match_seed: u64, seat: u8) -> u64 {
    Rng::derive(match_seed, &[0xB07, u64::from(seat)]).next_u64()
}

fn build(names: [&str; SEATS]) -> Vec<Box<dyn Bot>> {
    names
        .iter()
        .map(|n| bots::factory(n).expect("a registered bot").build())
        .collect()
}

struct Outcome {
    actions: Vec<Action>,
    money: [i32; TEAMS],
}

/// Play one match. `bot_seed` gives each seat's `start_match` seed.
fn play(seed: u64, seats: &mut [Box<dyn Bot>], bot_seed: impl Fn(u8) -> u64) -> Outcome {
    let rules = Rules::HOUSE;
    let mut m = Match::new(seed, rules);
    for (s, b) in (0u8..).zip(seats.iter_mut()) {
        b.start_match(s, rules, bot_seed(s));
    }
    while let Some(seat) = m.to_act() {
        let view = m.view(seat);
        let legal = m.legal_actions();
        let bot = &mut seats[usize::from(seat)];
        let a = bot.act(&view, &legal);
        assert!(
            legal.contains(&a),
            "{} at seat {seat} chose {a:?}, not in {legal:?} (match seed {seed})",
            bot.name()
        );
        m.apply(a).expect("a legal action is accepted");
    }
    Outcome {
        actions: m.actions().to_vec(),
        money: m.money().expect("the match is over"),
    }
}

/// A whole number of thousandths as a decimal, such as `-1.250` (no floating point involved).
fn thousandths(x: i64) -> String {
    let sign = if x < 0 { "-" } else { "" };
    format!("{sign}{}.{:03}", x.abs() / 1000, x.abs() % 1000)
}

/// A mix of both bots that changes with the match.
fn mixed(k: u64) -> [&'static str; SEATS] {
    core::array::from_fn(|s| {
        if (s as u64 + k).is_multiple_of(3) {
            bots::RULES_V1
        } else {
            bots::RANDOM
        }
    })
}

#[test]
fn same_seed_gives_identical_actions_over_50_matches() {
    for k in 0..50u64 {
        let seed = 0x41_0000 + k;
        let names = mixed(k);
        let first = play(seed, &mut build(names), |s| seat_seed(seed, s));
        let second = play(seed, &mut build(names), |s| seat_seed(seed, s));
        assert_eq!(first.actions, second.actions, "match seed {seed}");
        assert_eq!(first.money, second.money, "match seed {seed}");
    }
}

#[test]
fn the_same_bot_objects_replay_identically_after_start_match() {
    // start_match resets each bot, so reusing the objects for a second match changes nothing.
    let mut seats = build(mixed(7));
    let a = play(99, &mut seats, |s| seat_seed(99, s));
    let _ = play(100, &mut seats, |s| seat_seed(100, s));
    let b = play(99, &mut seats, |s| seat_seed(99, s));
    assert_eq!(a.actions, b.actions);
}

#[test]
fn the_rules_bot_ignores_its_seed() {
    for seed in 0..10u64 {
        let names = [bots::RULES_V1; SEATS];
        let a = play(seed, &mut build(names), |s| seat_seed(seed, s));
        let b = play(seed, &mut build(names), |s| {
            u64::from(s).wrapping_mul(7919) ^ 0xFFFF
        });
        assert_eq!(a.actions, b.actions, "match seed {seed}");
    }
}

#[test]
fn the_random_bot_depends_on_its_seed() {
    let names = [bots::RANDOM; SEATS];
    let differ = (0..10u64)
        .filter(|&seed| {
            let a = play(seed, &mut build(names), |s| seat_seed(seed, s));
            let b = play(seed, &mut build(names), |s| seat_seed(seed ^ 1, s));
            a.actions != b.actions
        })
        .count();
    assert_eq!(differ, 10);
}

#[test]
fn the_random_bot_is_always_legal() {
    // `play` asserts legality of every action; 300 all-random matches cover many positions,
    // including the rare robbing and turned-Joker branches.
    let mut decisions = 0usize;
    for seed in 0..300u64 {
        let out = play(seed, &mut build([bots::RANDOM; SEATS]), |s| {
            seat_seed(seed, s)
        });
        decisions += out.actions.len();
        assert_eq!(out.money.iter().sum::<i32>(), 0);
    }
    assert!(decisions > 10_000);
}

#[test]
fn the_random_bot_is_uniform_over_the_legal_actions() {
    // Every legal action of a fixed decision is chosen about equally often over many draws.
    let m = Match::new(3, Rules::HOUSE);
    let seat = m.to_act().expect("a seat to act");
    let view = m.view(seat);
    let legal = m.legal_actions();
    assert!(legal.len() >= 2);
    let mut bot = bots::RandomBot::new();
    bot.start_match(seat, Rules::HOUSE, 12345);
    let draws = 1000 * legal.len();
    let mut counts = vec![0usize; legal.len()];
    for _ in 0..draws {
        let a = bot.act(&view, &legal);
        let i = legal.iter().position(|&x| x == a).expect("a legal action");
        counts[i] += 1;
    }
    for &n in &counts {
        assert!((850..=1150).contains(&n), "counts {counts:?}");
    }
}

#[test]
fn rules_v1_team_against_six_random_seats_over_1000_matches() {
    // rules-v1 fills one team's two seats (the team rotates with the match so every seat
    // position is used), random the other six. The money is in euros at the house stake (5):
    // +30 to the winning team, -10 to each other team, so a random team averages 0.
    let n: i64 = 1000;
    let stake = i64::from(Rules::HOUSE.stake);
    let mut total: i64 = 0;
    let mut matches_won: i64 = 0;
    for k in 0..n as u64 {
        let seed = 0x5EED_0000 + k;
        let team = (k % TEAMS as u64) as u8;
        let names: [&str; SEATS] = core::array::from_fn(|s| {
            if team_of(s as u8) == team {
                bots::RULES_V1
            } else {
                bots::RANDOM
            }
        });
        let out = play(seed, &mut build(names), |s| seat_seed(seed, s));
        let won = out.money[usize::from(team)];
        total += i64::from(won);
        if won > 0 {
            matches_won += 1;
        }
    }
    println!(
        "rules-v1 team vs 6 random seats, {n} matches: money {total} euros in total, \
         {} euros per match = {} stake units per match; matches won {matches_won} \
         ({}%, a random team wins 25%)",
        thousandths(total * 1000 / n),
        thousandths(total * 1000 / (n * stake)),
        thousandths(matches_won * 100_000 / n),
    );
    assert!(
        total > 0,
        "rules-v1 must win money against random: {total} euros over {n} matches"
    );
}
