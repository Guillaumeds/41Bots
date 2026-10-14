//! World-generator tests (docs/M3-SPEC.md §4.6): counting against enumeration and brute force,
//! the uniform sampler against uniformity and the rejection oracle (Pearson χ²), pruning,
//! thread independence and golden counts.

use engine::cards::{Bits, Card, bit, cards, count};
use engine::game::{Match, Phase, Rules};
use engine::rng::Rng;
use engine::view::BurnKind;

use super::{check_truth, env_size, random_match};
use crate::space::log2_q8;
use crate::tracker::Loc;
use crate::{Tracker, World, WorldSpace, labels};

// ----- helpers -----

/// Decisions of random matches where `want(m, tracker, count)` holds: (match, tracker), up to
/// `n` of them, drawn from matches `seed0..`.
fn positions(
    n: usize,
    seed0: u64,
    mut want: impl FnMut(&Match, &Tracker, u128) -> bool,
) -> Vec<(Match, Tracker)> {
    let mut out = Vec::new();
    let mut seed = seed0;
    while out.len() < n {
        random_match(Match::new(seed, Rules::HOUSE), seed, |m| {
            if out.len() >= n {
                return;
            }
            let seat = m.to_act().expect("to act");
            let t = Tracker::from_view(&m.view(seat));
            let ws = WorldSpace::new(&t).expect("feasible");
            if want(m, &t, ws.count()) {
                out.push((m.clone(), t));
            }
        });
        seed += 1;
    }
    out
}

/// Late positions: trick 5 with at least four cards played to it, `lo ≤ N ≤ hi`.
fn late_positions(n: usize, seed0: u64, lo: u128, hi: u128) -> Vec<Tracker> {
    positions(n, seed0, |m, _, count| {
        let p = m.public();
        p.trick_no == 5 && p.trick_cards.len() >= 4 && (lo..=hi).contains(&count)
    })
    .into_iter()
    .map(|(_, t)| t)
    .collect()
}

/// A hand-made tracker with no rule behind it: a few unknown cards in two to four hands, the
/// undealt pack and perhaps a burn, with classes of cards sharing random masks that always allow
/// one true placement.
fn synthetic(rng: &mut Rng) -> Tracker {
    let n_cards = 6 + rng.bounded(6) as usize;
    let mut pool: Vec<Card> = (0..53).collect();
    rng.shuffle(&mut pool);
    let unknown_cards: Vec<Card> = pool[..n_cards].to_vec();
    let unknown = unknown_cards.iter().fold(0, |a, &c| a | bit(c));
    // Locations with capacities summing to n_cards.
    let mut seats: Vec<u8> = (1..8).collect();
    rng.shuffle(&mut seats);
    let n_hands = 2 + rng.bounded(3) as usize;
    let mut hand_seats = seats[..n_hands].to_vec();
    hand_seats.sort();
    let with_burn = rng.bounded(2) == 1;
    let mut caps: Vec<u8> = vec![1; n_hands + 1 + usize::from(with_burn)];
    let burn_slot = with_burn.then_some(caps.len() - 1);
    let mut left = n_cards - caps.len();
    while left > 0 {
        let i = rng.bounded((n_hands + 1) as u64) as usize; // hands and undealt grow
        caps[i] += 1;
        left -= 1;
    }
    let mut locs: Vec<(Loc, u8)> = hand_seats
        .iter()
        .zip(&caps)
        .map(|(&s, &c)| (Loc::Hand(s), c))
        .collect();
    locs.push((Loc::Undealt, caps[n_hands]));
    if let Some(b) = burn_slot {
        locs.push((Loc::Burn(seats[n_hands], BurnKind::RobAtTurn), caps[b]));
    }
    locs.sort();
    // A true placement: consecutive cards to consecutive locations.
    let mut truth_loc = [0usize; 53];
    let mut it = unknown_cards.iter();
    for (i, &(_, cap)) in locs.iter().enumerate() {
        for _ in 0..cap {
            truth_loc[usize::from(*it.next().expect("enough"))] = i;
        }
    }
    // Classes: 1 to 4 groups of cards, each with one mask covering its members' true places.
    let n_groups = 1 + rng.bounded(4) as usize;
    let group: Vec<usize> = unknown_cards
        .iter()
        .map(|_| rng.bounded(n_groups as u64) as usize)
        .collect();
    let mut masks = vec![0u32; n_groups];
    for (i, &c) in unknown_cards.iter().enumerate() {
        masks[group[i]] |= 1 << truth_loc[usize::from(c)];
    }
    for m in &mut masks {
        for l in 0..locs.len() {
            if rng.bounded(3) == 0 {
                *m |= 1 << l;
            }
        }
    }
    let mut allowed = [0u32; 53];
    for (i, &c) in unknown_cards.iter().enumerate() {
        allowed[usize::from(c)] = masks[group[i]];
    }
    let hand = cards(engine::cards::ALL_CARDS & !unknown)
        .take(5)
        .fold(0, |a, c| a | bit(c));
    Tracker::from_parts(0, hand, unknown, locs, allowed)
}

/// Synthetic trackers with `lo ≤ N ≤ hi`.
fn synthetic_positions(n: usize, seed: u64, lo: u128, hi: u128) -> Vec<Tracker> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    while out.len() < n {
        let t = synthetic(&mut rng);
        let ws = WorldSpace::new(&t).expect("a true placement exists");
        if (lo..=hi).contains(&ws.count()) {
            out.push(t);
        }
    }
    out
}

/// A placement as a world: the cards of `parts[i]` in `t.locs[i]`.
fn world_of(t: &Tracker, parts: &[Bits]) -> World {
    let mut hands = [0 as Bits; 8];
    hands[usize::from(t.seat)] = t.hand;
    let mut burns = Vec::new();
    let mut undealt = 0;
    for (&(loc, _), &set) in t.locs.iter().zip(parts) {
        match loc {
            Loc::Hand(s) => hands[usize::from(s)] = set,
            Loc::Undealt => undealt = set,
            Loc::Burn(..) => burns.extend(cards(set).map(|c| (loc, c))),
        }
    }
    burns.sort();
    World {
        hands,
        burns,
        undealt,
    }
}

fn accepts(t: &Tracker, w: &World) -> bool {
    t.consistent(&w.hands, &w.burns_by_loc(), w.undealt)
}

/// The number of ways to deal the unknown cards into the locations by capacity alone.
fn multinomial(t: &Tracker) -> u128 {
    let mut left = count(t.unknown);
    let mut n: u128 = 1;
    for &(_, cap) in &t.locs {
        n *= crate::space::binom(left, cap);
        left -= cap;
    }
    n
}

/// Every placement of the unknown cards by capacity, kept if the tracker accepts it.
fn brute_force(t: &Tracker) -> Vec<World> {
    fn subsets(pool: &[Card], k: usize, from: usize, acc: Bits, f: &mut dyn FnMut(Bits)) {
        if k == 0 {
            f(acc);
            return;
        }
        for i in from..pool.len() {
            if pool.len() - i < k {
                break;
            }
            subsets(pool, k - 1, i + 1, acc | bit(pool[i]), f);
        }
    }
    fn rec(t: &Tracker, i: usize, left: Bits, parts: &mut Vec<Bits>, out: &mut Vec<World>) {
        if i == t.locs.len() {
            let w = world_of(t, parts);
            if accepts(t, &w) {
                out.push(w);
            }
            return;
        }
        let pool: Vec<Card> = cards(left).collect();
        subsets(&pool, usize::from(t.locs[i].1), 0, 0, &mut |set| {
            parts.push(set);
            rec(t, i + 1, left & !set, parts, out);
            parts.pop();
        });
    }
    let mut out = Vec::new();
    rec(t, 0, t.unknown, &mut Vec::new(), &mut out);
    out.sort();
    out
}

/// One rejection-oracle draw: the unknown cards dealt uniformly by capacity, ignoring the facts;
/// `None` if the tracker refuses the deal.
fn rejection_draw(t: &Tracker, rng: &mut Rng) -> Option<World> {
    let mut pool: Vec<Card> = cards(t.unknown).collect();
    rng.shuffle(&mut pool);
    let mut it = pool.into_iter();
    let parts: Vec<Bits> = t
        .locs
        .iter()
        .map(|&(_, cap)| (0..cap).fold(0, |a, _| a | bit(it.next().expect("enough"))))
        .collect();
    let w = world_of(t, &parts);
    accepts(t, &w).then_some(w)
}

// ----- χ² (floats: test statistics only) -----

fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + 7.5;
    let mut a = G[0];
    for (i, &g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// The upper regularised incomplete gamma function Q(a, x).
fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let front = (-x + a * x.ln() - ln_gamma(a)).exp();
    if x < a + 1.0 {
        // Series for P(a, x).
        let (mut sum, mut del, mut ap) = (1.0 / a, 1.0 / a, a);
        for _ in 0..100_000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        1.0 - sum * front
    } else {
        // Continued fraction (Lentz) for Q(a, x).
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..100_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        front * h
    }
}

/// P(χ²_k ≥ x).
fn chi2_sf(x: f64, k: f64) -> f64 {
    gamma_q(k / 2.0, x / 2.0)
}

/// Pearson χ² of counts against uniform: the p-value.
fn p_uniform(counts: &[u64]) -> f64 {
    let total: u64 = counts.iter().sum();
    let e = total as f64 / counts.len() as f64;
    let x: f64 = counts.iter().map(|&o| (o as f64 - e).powi(2) / e).sum();
    chi2_sf(x, counts.len() as f64 - 1.0)
}

/// χ² test that two histograms come from the same distribution: the p-value.
fn p_same(a: &[u64], b: &[u64]) -> f64 {
    let (ta, tb) = (a.iter().sum::<u64>() as f64, b.iter().sum::<u64>() as f64);
    let mut x = 0.0;
    let mut cells = 0.0;
    for (&oa, &ob) in a.iter().zip(b) {
        let s = (oa + ob) as f64;
        if s == 0.0 {
            continue;
        }
        cells += 1.0;
        let ea = s * ta / (ta + tb);
        let eb = s * tb / (ta + tb);
        x += (oa as f64 - ea).powi(2) / ea + (ob as f64 - eb).powi(2) / eb;
    }
    chi2_sf(x, cells - 1.0)
}

#[test]
fn chi2_tail_matches_tables() {
    for (x, k, p) in [
        (3.841, 1.0, 0.05),
        (18.307, 10.0, 0.05),
        (124.342, 100.0, 0.05),
        (6.635, 1.0, 0.01),
        (5104.0, 4999.0, 0.147),
    ] {
        let got = chi2_sf(x, k);
        assert!(
            (got - p).abs() < 0.01 * p.max(0.1),
            "χ²({x}, {k}) = {got}, want {p}"
        );
    }
}

// ----- the tests -----

/// The sampler is uniform and agrees with the rejection oracle on `positions`.
fn uniform_vs_rejection(trackers: &[Tracker], draws_per_world: u64, seed: u64) {
    for (i, t) in trackers.iter().enumerate() {
        let ws = WorldSpace::new(t).expect("feasible");
        let all = ws.enumerate(1_000_000).expect("small");
        let n = all.len() as u64;
        let index = |w: &World| all.binary_search(w).expect("a consistent world");
        let mut sampled = vec![0u64; all.len()];
        for k in 0..draws_per_world * n {
            let w = ws.sample(&mut Rng::derive(seed, &[labels::WORLD, i as u64, k]));
            sampled[index(&w)] += 1;
        }
        let mut oracle = vec![0u64; all.len()];
        let mut rng = Rng::derive(seed, &[0x0AC1E, i as u64]);
        let mut got = 0;
        while got < draws_per_world * n {
            if let Some(w) = rejection_draw(t, &mut rng) {
                oracle[index(&w)] += 1;
                got += 1;
            }
        }
        let pu = p_uniform(&sampled);
        let ps = p_same(&sampled, &oracle);
        assert!(
            pu >= 0.001 && ps >= 0.001,
            "position {i} (N = {n}): uniform p = {pu}, same-as-oracle p = {ps}"
        );
        // The oracle itself is uniform (a check of the check).
        assert!(p_uniform(&oracle) >= 0.001, "position {i}: the oracle");
    }
}

#[test]
fn uniform_vs_rejection_oracle() {
    // 20 real late-hand positions and 20 hand-made ones with several classes and multi-card
    // places, small enough for a unit test.
    let mut ts = late_positions(20, 100, 2, 60);
    ts.extend(synthetic_positions(20, 7, 2, 60));
    uniform_vs_rejection(&ts, 200, 1);
}

#[test]
#[ignore = "full size (§4.6): 40 positions with N ≤ 5,000, 200·N draws each; run in release"]
fn uniform_vs_rejection_oracle_full() {
    let mut ts = late_positions(20, 1_000, 2, 5_000);
    ts.extend(synthetic_positions(20, 70, 2, 5_000));
    uniform_vs_rejection(&ts, 200, 2);
}

/// `count()` equals the enumeration, every enumerated world is consistent and distinct; on the
/// first `brute` positions the brute-force count of every placement agrees too.
fn count_equals_enumeration_on(trackers: &[Tracker], brute: usize) -> usize {
    let mut bruted = 0;
    for (i, t) in trackers.iter().enumerate() {
        let ws = WorldSpace::new(t).expect("feasible");
        let all = ws.enumerate(10_000).expect("N ≤ 10^4");
        assert_eq!(all.len() as u128, ws.count(), "position {i}");
        assert!(all.windows(2).all(|w| w[0] < w[1]), "sorted, no duplicates");
        assert!(
            all.iter().all(|w| accepts(t, w)),
            "every world is consistent"
        );
        if bruted < brute && multinomial(t) <= 60_000 {
            assert_eq!(brute_force(t), all, "position {i}: brute force");
            bruted += 1;
        }
    }
    bruted
}

#[test]
fn count_equals_enumeration() {
    let mut ts = late_positions(150, 300, 1, 10_000);
    ts.extend(synthetic_positions(100, 9, 1, 10_000));
    let bruted = count_equals_enumeration_on(&ts, 60);
    assert_eq!(bruted, 60);
}

#[test]
#[ignore = "full size (§4.6): 500 late positions, 200 brute-forced"]
fn count_equals_enumeration_full() {
    let mut ts = late_positions(500, 3_000, 1, 10_000);
    ts.extend(synthetic_positions(200, 90, 1, 10_000));
    assert_eq!(count_equals_enumeration_on(&ts, 200), 200);
}

#[test]
fn count_prune_identity() {
    let ps = positions(300, 500, |m, _, _| m.public().trick_no >= 1);
    let mut tricks = [0u32; 6];
    for (m, t) in &ps {
        let a = WorldSpace::new(t).expect("feasible");
        let b = WorldSpace::new_unpruned(t).expect("feasible");
        assert_eq!(a.count(), b.count());
        tricks[usize::from(m.public().trick_no)] += 1;
    }
    assert!(tricks[1..].iter().all(|&n| n > 0), "{tricks:?}");
    for t in synthetic_positions(200, 11, 1, u128::MAX) {
        let a = WorldSpace::new(&t).expect("feasible");
        let b = WorldSpace::new_unpruned(&t).expect("feasible");
        assert_eq!(a.count(), b.count());
    }
}

#[test]
fn sampled_worlds_are_consistent_and_contain_the_truth_often_enough() {
    // At decisions of every trick, sampled worlds satisfy the tracker; at trick 5 the true
    // deal is among a large enough sample of a small space.
    let ps = positions(200, 700, |_, _, _| true);
    for (i, (m, t)) in ps.iter().enumerate() {
        let ws = WorldSpace::new(t).expect("feasible");
        for k in 0..20 {
            let w = ws.sample(&mut Rng::derive(i as u64, &[labels::WORLD, k]));
            assert!(accepts(t, &w));
            assert_eq!(w.hands[usize::from(t.seat)], t.hand);
        }
        check_truth(m);
    }
}

#[test]
fn sample_thread_independent() {
    let (_, t) = positions(1, 40, |m, _, _| m.public().trick_no == 2)
        .pop()
        .expect("a trick-2 decision");
    let ws = WorldSpace::new(&t).expect("feasible");
    let dseed = 0x5EED_1234;
    let draw = |w: u64| ws.sample(&mut Rng::derive(dseed, &[labels::WORLD, w]));
    let forward: Vec<World> = (0..1000).map(draw).collect();
    let mut backward: Vec<World> = (0..1000).rev().map(draw).collect();
    backward.reverse();
    assert_eq!(forward, backward);
    for threads in [1usize, 2, 4] {
        let mut slots: Vec<Option<World>> = vec![None; 1000];
        std::thread::scope(|s| {
            let chunks: Vec<_> = slots.chunks_mut(1000 / threads + 1).collect();
            let mut start = 0;
            for chunk in chunks {
                let from = start;
                start += chunk.len();
                let draw = &draw;
                s.spawn(move || {
                    for (j, slot) in chunk.iter_mut().enumerate() {
                        *slot = Some(draw((from + j) as u64));
                    }
                });
            }
        });
        let got: Vec<World> = slots.into_iter().map(|w| w.expect("drawn")).collect();
        assert_eq!(got, forward, "{threads} threads");
    }
    // Different worlds do differ.
    assert!(forward.windows(2).any(|w| w[0] != w[1]));
}

#[test]
fn next_turned_is_uniform_over_the_undealt_cards() {
    // The dealer's pre-trump burn after a turned Joker: worlds draw the card turned next.
    let m = super::hand_made(3, [""; 8], "JK 9D");
    let t = Tracker::from_view(&m.view(3));
    let ws = WorldSpace::new(&t).expect("feasible");
    let w = ws.sample(&mut Rng::new(5));
    assert_eq!(count(w.undealt), 12);
    let mut hits = [0u64; 53];
    for k in 0..12_000u64 {
        let c = w.draw_next_turned(&mut Rng::derive(
            9,
            &[labels::WORLD, k, labels::NEXT_TURNED],
        ));
        assert!(engine::cards::has(w.undealt, c));
        hits[usize::from(c)] += 1;
    }
    let counts: Vec<u64> = cards(w.undealt).map(|c| hits[usize::from(c)]).collect();
    assert!(p_uniform(&counts) >= 0.001, "{counts:?}");
}

/// N at the start of tricks 1, 4 and 5 (the leader's view at the trick's first decision) in
/// matches played by a seeded random player.
fn trick_start_counts(seed: u64) -> [u128; 3] {
    let mut out = [0u128; 3];
    let mut found = [false; 3];
    random_match(Match::new(seed, Rules::HOUSE), seed, |m| {
        let p = m.public();
        if !p.trick_cards.is_empty() || m.phase() == Phase::MatchOver {
            return;
        }
        let slot = match p.trick_no {
            1 => 0,
            4 => 1,
            5 => 2,
            _ => return,
        };
        if found[slot] {
            return;
        }
        let t = Tracker::from_view(&m.view(m.to_act().expect("to act")));
        out[slot] = WorldSpace::new(&t).expect("feasible").count();
        found[slot] = true;
    });
    assert_eq!(found, [true; 3]);
    out
}

#[test]
fn golden_counts() {
    // Stored test data: the exact counts must never change for these seeds.
    // Seed 3's trick-1 count is the plain multinomial 47! / (5!^7 · 12!): no fact yet narrows
    // any card for the leader.
    const GOLDEN: [(u64, [u128; 3]); 3] = [
        (
            1,
            [
                2_308_322_595_696_190_185_964_115_414_637_527_040,
                264_573_400_291_200,
                360_951_552,
            ],
        ),
        (
            2,
            [
                7_534_108_472_063_954_079_188_432_256_108_595_200,
                719_302_632_048_000,
                348_894_000,
            ],
        ),
        (
            3,
            [
                1_506_821_694_412_790_815_837_686_451_221_719_040,
                73_001_602_674_000,
                124_199_712,
            ],
        ),
    ];
    for (seed, want) in GOLDEN {
        let got = trick_start_counts(seed);
        assert_eq!(got, want, "seed {seed}");
        // Orders of magnitude as in A4: about 10^36 at trick 1, about 10^8 at trick 5.
        let l10 = |n: u128| f64::from(log2_q8(n)) / 256.0 * std::f64::consts::LOG10_2;
        assert!((30.0..37.0).contains(&l10(got[0])), "{}", l10(got[0]));
        assert!(l10(got[1]) > l10(got[2]) && l10(got[2]) > 4.0);
    }
}

#[test]
#[ignore = "A6 property with world counts on many matches; BELIEF_MATCHES_WS (default 2,000)"]
fn world_space_feasible_on_many_matches() {
    for seed in 0..env_size("BELIEF_MATCHES_WS", 2_000) {
        random_match(Match::new(seed ^ 0xF00D, Rules::HOUSE), seed, |m| {
            let t = Tracker::from_view(&m.view(m.to_act().expect("to act")));
            let ws = WorldSpace::new(&t).expect("feasible");
            let w = ws.sample(&mut Rng::new(seed));
            assert!(accepts(&t, &w));
        });
    }
}
