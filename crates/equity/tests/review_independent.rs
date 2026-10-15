//! Independent checks of the match-equity table (review B, 2026-10-15).
//!
//! Nothing here uses `equity::hand`, `MatchState::rotate`/`canonical` or `value_via`: the hand
//! scoring is re-written from RULES 27-33 and the "End checks" note, the canonical rotation and
//! index from docs/M1-SPEC.md §6.2, and a state's value is recomputed by plain recursion over
//! every trick sequence in floating point (test-only) from the model's probabilities and the
//! table's successor entries. Agreement checks the sweep's dynamic program, its scoring, its
//! dealer and game transitions and the rotation bookkeeping all at once.

use engine::rng::Rng;
use equity::hand::{AfterHand, AfterTrick, InHand, Scoring};
use equity::model::{COMPS, CompiledModel, HandModel, N_BEFORE_TRICK};
use equity::state::SCORE_VALUES;
use equity::sweep::{Scope, SweepOptions, sweep};
use equity::{EquityTable, MatchState, MoneyChance, ONE};

/// The table index of a score, by search in the documented list.
fn idx(score: u8) -> usize {
    SCORE_VALUES
        .iter()
        .position(|&s| s == score)
        .unwrap_or_else(|| panic!("unreachable score {score}"))
}

/// docs/M1-SPEC.md §6.2 by hand: rotate so the dealer's team is team 0 (canonical team i is real
/// team (i + d) mod 4), index, and give real team t the canonical entry (t − d) mod 4.
fn lookup_by_hand(table: &EquityTable, scores: [u8; 4], games: [u8; 4], dealer: u8) -> [f64; 4] {
    if let Some(w) = (0..4).find(|&t| games[t] >= 2) {
        let mut v = [0.0; 4];
        v[w] = 1.0;
        return v;
    }
    let d = usize::from(dealer);
    let mut bits = 0usize;
    let mut sidx = 0usize;
    for i in 0..4 {
        let real = (i + d) % 4;
        bits |= usize::from(games[real]) << i;
        sidx = sidx * 15 + idx(scores[real]);
    }
    let raw = table
        .get(bits * 50_625 + sidx)
        .unwrap_or_else(|| panic!("not solved: {scores:?} {games:?} {dealer}"));
    core::array::from_fn(|t| raw.0[(t + 4 - d) % 4] as f64 / ONE as f64)
}

/// RULES 27-33 by hand. State inside a hand, real team order.
#[derive(Clone, Copy)]
struct Hand {
    scores: [u8; 4],
    tricks: [u8; 4],
    five_played: bool,
    done: u8,
}

enum Out {
    Game(usize),
    Over([u8; 4]),
    Next(Hand),
}

/// Rule 28: first best-trump bonus of a team in a game is 6, later ones 5. A team has had its
/// first exactly when its score is 5k+6 (scores 11, 16, ..., 36), i.e. score mod 5 == 1.
fn bonus(score: u8) -> u8 {
    if score % 5 == 1 { 5 } else { 6 }
}

fn trick(h: Hand, w: usize, five_here: bool) -> Out {
    let mut h = h;
    h.scores[w] += 5; // rule 27
    h.tricks[w] += 1;
    h.done += 1;
    if five_here {
        // rule 29: the 5's trick takes the bonus at once (its flag is read before the +5 trick
        // points could change it: 5k+6 + 5 is still 5k'+6).
        h.scores[w] += bonus(h.scores[w] - 5);
        h.five_played = true;
    }
    if h.scores[w] >= 41 {
        return Out::Game(w); // rule 32
    }
    if h.done == 5 && h.tricks[w] == 5 {
        return Out::Game(w); // rule 33
    }
    Out::Next(h)
}

fn end_bonus(h: Hand, b: usize) -> Out {
    let mut s = h.scores;
    if !h.five_played {
        s[b] += bonus(s[b]); // rule 30
        if s[b] >= 41 {
            return Out::Game(b);
        }
    }
    Out::Over(s)
}

fn comp(rel: [u8; 4]) -> usize {
    COMPS.iter().position(|&c| c == rel).expect("composition")
}

/// The value of a hand position by recursion over every trick sequence (real team order).
fn value_rec(m: &CompiledModel, t: &EquityTable, h: Hand, games: [u8; 4], dealer: u8) -> [f64; 4] {
    let d = usize::from(dealer);
    let next_dealer = (dealer + 1) % 4;
    let game_end = |w: usize| {
        let mut g = games;
        g[w] += 1;
        lookup_by_hand(t, [0; 4], g, next_dealer)
    };
    let rel: [u8; 4] = core::array::from_fn(|o| h.tricks[(o + d) % 4]);
    let k = comp(rel);
    let mut v = [0.0; 4];
    let mut add = |p: f64, x: [f64; 4]| {
        for i in 0..4 {
            v[i] += p * x[i];
        }
    };
    if h.done == 5 {
        if h.five_played {
            return lookup_by_hand(t, h.scores, games, next_dealer);
        }
        for (o, &p) in m.end_bonus[k - N_BEFORE_TRICK].iter().enumerate() {
            if p == 0 {
                continue;
            }
            let b = (o + d) % 4;
            let x = match end_bonus(h, b) {
                Out::Game(w) => game_end(w),
                Out::Over(s) => {
                    for &sc in &s {
                        idx(sc); // every hand ends on a reachable score
                    }
                    lookup_by_hand(t, s, games, next_dealer)
                }
                Out::Next(_) => unreachable!(),
            };
            add(p as f64 / ONE as f64, x);
        }
        return v;
    }
    for (o, &p) in m.winner[k].iter().enumerate() {
        if p == 0 {
            continue;
        }
        let w = (o + d) % 4;
        let pf = if h.five_played {
            [ONE, 0]
        } else {
            m.five[usize::from(h.done)][o]
        };
        for (here, &q) in pf.iter().enumerate() {
            if q == 0 {
                continue;
            }
            let x = match trick(h, w, here == 1) {
                Out::Game(w) => game_end(w),
                Out::Next(n) => value_rec(m, t, n, games, dealer),
                Out::Over(_) => unreachable!(),
            };
            add(p as f64 / ONE as f64 * (q as f64 / ONE as f64), x);
        }
    }
    v
}

fn check_states(model: &CompiledModel, table: &EquityTable, n: usize, seed: u64) -> usize {
    let mut rng = Rng::derive(seed, &[0x0B]);
    let mut checked = 0;
    let mut tries = 0;
    while checked < n {
        tries += 1;
        assert!(tries < 1_000_000, "too few solved states");
        let i = rng.bounded(810_000) as usize;
        if table.get(i).is_none() {
            continue;
        }
        // Any dealer team, real team order, built by hand (not from the library's rotation).
        let dealer = rng.bounded(4) as u8;
        let rest = i % 50_625;
        let bits = i / 50_625;
        let canon_scores: [u8; 4] =
            core::array::from_fn(|c| SCORE_VALUES[(rest / 15usize.pow(3 - c as u32)) % 15]);
        let mut scores = [0u8; 4];
        let mut games = [0u8; 4];
        for (c, &cs) in canon_scores.iter().enumerate() {
            let real = (c + usize::from(dealer)) % 4;
            scores[real] = cs;
            games[real] = ((bits >> c) & 1) as u8;
        }
        let h = Hand {
            scores,
            tricks: [0; 4],
            five_played: false,
            done: 0,
        };
        let want = value_rec(model, table, h, games, dealer);
        let state = MatchState::new(scores, games, dealer).expect("valid");
        let got = table.lookup(&state).to_f64_for_display();
        let by_hand = lookup_by_hand(table, scores, games, dealer);
        for t in 0..4 {
            assert!(
                (got[t] - want[t]).abs() < 1e-7,
                "{state}: table {got:?}, recomputed {want:?}"
            );
            assert_eq!(got[t], by_hand[t], "{state}: lookup rotation");
        }
        checked += 1;
    }
    checked
}

#[test]
fn table_equals_independent_recursion_synthetic_model() {
    let model = HandModel::synthetic().compile();
    // Every state reachable from one team on 1 game and 36 points (all games patterns).
    let mut roots = Vec::new();
    for team in 0..4 {
        for d in 0..4u8 {
            let mut s = [0u8; 4];
            let mut g = [0u8; 4];
            s[team] = 36;
            g[team] = 1;
            roots.push(MatchState::new(s, g, d).expect("valid"));
        }
    }
    let (table, _) = sweep(
        &model,
        Scoring::HOUSE,
        &SweepOptions {
            threads: 2,
            scope: Scope::ReachableFrom(roots),
        },
    );
    assert_eq!(check_states(&model, &table, 300, 1), 300);
}

/// The real table v0: `EQUITY_TABLE=<equity-v0.npy> EQUITY_MODEL=<model json> cargo test
/// --release -p equity --test review_independent -- --ignored`.
#[test]
#[ignore]
fn table_equals_independent_recursion_real_table() {
    let t = std::env::var("EQUITY_TABLE").expect("EQUITY_TABLE");
    let m = std::env::var("EQUITY_MODEL").expect("EQUITY_MODEL");
    let table = EquityTable::from_npy(&std::fs::read(t).expect("read table")).expect("table");
    let model = HandModel::from_json(&std::fs::read_to_string(m).expect("read model"))
        .expect("model")
        .compile();
    assert_eq!(check_states(&model, &table, 3_000, 7), 3_000);
}

/// RULES 27-33 by hand against `InHand`, for many start scores and every trick sequence (every
/// winner of each trick, the 5 of trumps in any trick or none, the end bonus to any team that won
/// a trick).
#[test]
fn in_hand_scoring_equals_rules_by_hand() {
    let mut rng = Rng::derive(0xB, &[1]);
    for _ in 0..120 {
        let start: [u8; 4] = core::array::from_fn(|_| SCORE_VALUES[rng.bounded(15) as usize]);
        for seq in 0..1024u32 {
            let winners: [usize; 5] = core::array::from_fn(|i| ((seq >> (2 * i)) & 3) as usize);
            for five in 0..=5usize {
                let mut mine = Out::Next(Hand {
                    scores: start,
                    tricks: [0; 4],
                    five_played: false,
                    done: 0,
                });
                let mut lib = Some(InHand::start(start));
                let mut lib_done: Option<AfterHand> = None;
                for (i, &w) in winners.iter().enumerate() {
                    let here = five == i + 1;
                    if let Out::Next(h) = mine {
                        mine = trick(h, w, here);
                    }
                    if let Some(p) = lib.take() {
                        match p.play_trick(w as u8, here, Scoring::HOUSE) {
                            AfterTrick::GameWon(t, _) => {
                                lib_done = Some(AfterHand::GameWon(t, p_reason()))
                            }
                            AfterTrick::Next(q) | AfterTrick::TricksDone(q) => lib = Some(q),
                        }
                    }
                }
                for b in 0..4usize {
                    let (mine_b, lib_b) = match (&mine, &lib, &lib_done) {
                        (Out::Game(w), None, Some(AfterHand::GameWon(t, _))) => {
                            assert_eq!(*w, usize::from(*t), "{start:?} {winners:?} 5@{five}");
                            continue;
                        }
                        (Out::Next(h), Some(p), None) => {
                            if h.tricks[b] == 0 {
                                continue; // the best trump wins its trick
                            }
                            (end_bonus(*h, b), p.finish(Some(b as u8), Scoring::HOUSE))
                        }
                        _ => panic!("game end disagrees: {start:?} {winners:?} 5@{five}"),
                    };
                    match (mine_b, lib_b) {
                        (Out::Game(w), AfterHand::GameWon(t, _)) => assert_eq!(w, usize::from(t)),
                        (Out::Over(s), AfterHand::HandOver(u)) => assert_eq!(s, u),
                        _ => panic!("end of hand disagrees: {start:?} {winners:?} 5@{five} b{b}"),
                    }
                }
            }
        }
    }
}

fn p_reason() -> engine::game::GameEndReason {
    engine::game::GameEndReason::Jink
}

/// Hand-computed cases (RULES 27-33, "Reachable scores", "End checks").
#[test]
fn hand_computed_scoring_cases() {
    let sc = Scoring::HOUSE;
    let go = |start: [u8; 4], ws: &[u8], five: Option<usize>, best: Option<u8>| {
        let mut p = InHand::start(start);
        for (i, &w) in ws.iter().enumerate() {
            match p.play_trick(w, five == Some(i + 1), sc) {
                AfterTrick::GameWon(t, r) => return Err((t, r)),
                AfterTrick::Next(q) | AfterTrick::TricksDone(q) => p = q,
            }
        }
        match p.finish(best, sc) {
            AfterHand::GameWon(t, r) => Err((t, r)),
            AfterHand::HandOver(s) => Ok(s),
        }
    };
    use engine::game::GameEndReason::*;
    // 11 (first bonus taken) wins the 5's trick: 11 + 5 + 5 = 21, a later bonus.
    assert_eq!(
        go([11, 0, 0, 0], &[0, 1, 2, 3, 1], Some(1), None),
        Ok([21, 10, 5, 5])
    );
    // 35 (no bonus yet) wins the 5's trick: 35 + 5 = 40, + 6 = 46: 41 by the immediate bonus.
    assert_eq!(
        go([35, 0, 0, 0], &[0], Some(1), None),
        Err((0, FortyOneByImmediateBonus))
    );
    // 31 (bonus taken) wins the 5's trick: 36 + 5 = 41 by the immediate bonus.
    assert_eq!(
        go([31, 0, 0, 0], &[1, 0], Some(2), None),
        Err((0, FortyOneByImmediateBonus))
    );
    // 30 wins a plain trick: 35, no game; the end bonus of a team on 35 (no flag) is 6: 41.
    assert_eq!(
        go([30, 0, 0, 0], &[0, 1, 2, 3, 1], None, Some(0)),
        Err((0, FortyOneByEndBonus))
    );
    // 25 + one trick + end bonus = 36: no game; flag set (36 mod 5 == 1).
    assert_eq!(
        go([25, 0, 0, 0], &[0, 1, 2, 3, 1], None, Some(0)),
        Ok([36, 10, 5, 5])
    );
    // 26 (flag) + one trick + later end bonus = 36.
    assert_eq!(
        go([26, 0, 0, 0], &[0, 1, 2, 3, 1], None, Some(0)),
        Ok([36, 10, 5, 5])
    );
    // Jink from nought with the 5 in trick 1: 11, 16, 21, 26, 31: a jink, not 41.
    assert_eq!(go([0; 4], &[2, 2, 2, 2, 2], Some(1), None), Err((2, Jink)));
    // Jink from nought without the 5: 25 after five tricks: jink before the end bonus.
    assert_eq!(go([0; 4], &[3, 3, 3, 3, 3], None, Some(3)), Err((3, Jink)));
    // Team on 40 takes trick 5 while team 1 on 36 would get the end bonus: trick points first.
    // (25 + 5 + 5 + 5 = 40 after trick 3, 45 at trick 5.)
    assert_eq!(
        go([25, 36, 0, 0], &[0, 0, 0, 2, 0], None, Some(1)),
        Err((0, FortyOneByTrick))
    );
    // From 20 the same tricks only reach 40, so team 1's end bonus (36 + 5) wins instead.
    assert_eq!(
        go([20, 36, 0, 0], &[0, 0, 0, 2, 0], None, Some(1)),
        Err((1, FortyOneByEndBonus))
    );
    // Team 1 on 31 takes trick 4 (36); team 0 on 35 takes trick 5 (40, no game); the end bonus
    // to team 1 (36 + 5, a later bonus) = 41: team 1 wins.
    assert_eq!(
        go([35, 31, 0, 0], &[2, 2, 3, 1, 0], None, Some(1)),
        Err((1, FortyOneByEndBonus))
    );
    // Rule 31: no trump played, no bonus.
    assert_eq!(go([0; 4], &[0, 1, 2, 3, 0], None, None), Ok([10, 5, 5, 5]));
}

/// Fixed-point invariants on random vectors: every constructor and every chain of mixes sums to
/// exactly ONE, and each mixed share is the floor or ceiling of the exact share.
#[test]
fn money_chance_sum_invariant_property() {
    let mut rng = Rng::derive(0xB, &[2]);
    let rand_mc = |rng: &mut Rng| {
        let mut w: [u128; 4] = core::array::from_fn(|_| match rng.bounded(4) {
            0 => 0,
            1 => u128::from(rng.bounded(3)),
            _ => u128::from(rng.next_u64()) + 1,
        });
        if w.iter().all(|&x| x == 0) {
            w[rng.bounded(4) as usize] = 1;
        }
        MoneyChance::from_weights(w)
    };
    for _ in 0..3_000 {
        let n = 1 + rng.bounded(8) as usize;
        let mut terms: Vec<(u128, MoneyChance)> = Vec::new();
        for _ in 0..n {
            let c = rand_mc(&mut rng);
            terms.push((u128::from(rng.next_u64()) << rng.bounded(30), c));
        }
        if terms.iter().all(|&(w, _)| w == 0) {
            continue;
        }
        let m = MoneyChance::mix_wide(&terms);
        assert_eq!(
            m.0.iter().map(|&x| u128::from(x)).sum::<u128>(),
            u128::from(ONE)
        );
        let den: u128 = terms.iter().map(|&(w, _)| w).sum();
        for t in 0..4 {
            // Σ w·c fits u128: den < 2^94, c ≤ 2^32.
            let num: u128 = terms.iter().map(|&(w, c)| w * u128::from(c.0[t])).sum();
            let fl = (num / den) as u64;
            assert!(m.0[t] == fl || (!num.is_multiple_of(den) && m.0[t] == fl + 1));
        }
        for by in 0..8 {
            let r = m.rotate(by);
            assert_eq!(r.0.iter().sum::<u64>(), ONE);
            assert_eq!(r.rotate(4 - by % 4), m);
        }
        // A chain of mixes keeps the invariant.
        let chained = MoneyChance::mix(&[(3, m), (5, terms[0].1), (1u64 << 40, rand_mc(&mut rng))]);
        assert_eq!(chained.0.iter().sum::<u64>(), ONE);
    }
}
