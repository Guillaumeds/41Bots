//! Correctness tests for the naive max^n solver (PLAN A2, validation layers 1–3).
//!
//! The brute-force reference below is independent of the solver's code: it drives clones of the
//! real engine [`Match`] through every legal play to the end of the hand, and values each
//! finished line from the engine's own scores, games won and dealer (so the engine's scoring and
//! game-end checks are the reference, not `equity::hand`). It uses the same backup rule and tie
//! order (the mover's team entry is maximised; the lowest card index wins ties).

use engine::cards::Card;
use engine::game::{Action, Match, Phase, Rules};
use engine::rng::Rng;
use engine::team_of;
use equity::{MatchState, MoneyChance, ONE};
use solver::{FnLeaf, Leaf, Options, TieOrder, from_match, solve};

/// A deterministic, arbitrary leaf table: weights from a hash of the state (not rotation-symmetric
/// on purpose, so any mix-up of teams or dealers shows).
fn fake_value(s: &MatchState) -> MoneyChance {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s
        .scores
        .iter()
        .chain(&s.games)
        .chain(core::iter::once(&s.dealer_team))
    {
        h = (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01B3);
    }
    let w: [u128; 4] = core::array::from_fn(|t| u128::from((h >> (16 * t)) & 0xFFFF) + 1);
    MoneyChance::from_weights(w)
}

fn leaf() -> FnLeaf<fn(&MatchState) -> MoneyChance> {
    FnLeaf(fake_value)
}

/// Play random legal actions from a seeded match until the start of trick `k` (k >= 2).
fn position_at(seed: u64, k: u8) -> Match {
    let mut rng = Rng::new(seed);
    let mut m = Match::new(rng.next_u64(), Rules::HOUSE);
    let skip = rng.bounded(3) as u32;
    loop {
        if m.phase() == Phase::MatchOver {
            m = Match::new(rng.next_u64(), Rules::HOUSE);
            continue;
        }
        let p = m.public();
        if p.trick_no == k
            && p.trick_cards.is_empty()
            && m.hand_index() >= skip
            && matches!(m.phase(), Phase::AwaitPlay(_))
        {
            return m;
        }
        let acts = m.legal_actions();
        m.apply(acts[rng.bounded(acts.len() as u64) as usize])
            .expect("legal");
    }
}

/// The brute-force max^n value by engine clones.
fn brute<L: Leaf>(m: &Match, hand0: u32, leaf: &L) -> MoneyChance {
    if m.phase() == Phase::MatchOver {
        let w = (0..4u8)
            .find(|&t| m.games_won()[usize::from(t)] >= 2)
            .expect("a winner");
        return MoneyChance::certain(w);
    }
    if m.hand_index() != hand0 {
        return leaf.value(&MatchState {
            scores: m.scores(),
            games: m.games_won(),
            dealer_team: m.dealer() % 4,
        });
    }
    let Phase::AwaitPlay(seat) = m.phase() else {
        panic!("no burns after trick 1")
    };
    let team = team_of(seat);
    let mut cards: Vec<Card> = m.legal_actions().iter().map(|a| a.card()).collect();
    cards.sort_unstable();
    let mut best: Option<MoneyChance> = None;
    for c in cards {
        let mut n = m.clone();
        n.apply(Action::Play(c)).expect("legal");
        let v = brute(&n, hand0, leaf);
        if best.is_none_or(|b| v.get(team) > b.get(team)) {
            best = Some(v);
        }
    }
    best.expect("legal set is never empty")
}

fn check_against_brute(k: u8, seeds: core::ops::Range<u64>) {
    let l = leaf();
    for seed in seeds {
        let m = position_at(seed, k);
        let (pos, ctx, pending) = from_match(&m).expect("position");
        assert!(pending.is_none());
        let want = brute(&m, m.hand_index(), &l);
        for tt in [false, true] {
            let got = solve(
                &pos,
                ctx,
                &l,
                Options {
                    tt,
                    ..Options::default()
                },
            );
            assert_eq!(got.value, Some(want), "seed {seed} trick {k} tt {tt}");
            // Each root child's value equals the brute-force value after that card.
            for &(c, v) in &got.children {
                let mut n = m.clone();
                n.apply(Action::Play(c)).expect("legal");
                assert_eq!(v, brute(&n, m.hand_index(), &l), "seed {seed} child {c}");
            }
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
