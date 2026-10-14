//! Shared helpers for the solver tests: synthetic leaf tables, random mid-hand positions and the
//! engine-driven brute force.
#![allow(dead_code)]

use engine::cards::Card;
use engine::game::{Action, Match, Phase, Rules};
use engine::rng::Rng;
use engine::team_of;
use equity::{MatchState, MoneyChance};
use solver::{FnLeaf, Leaf};

/// A deterministic, arbitrary leaf table: weights from a hash of the state (not rotation-symmetric
/// on purpose, so any mix-up of teams or dealers shows).
pub fn fake_value(s: &MatchState) -> MoneyChance {
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

/// A coarse leaf with very many exact ties (only "has 25 or more", games and the dealer's team
/// matter), to stress the tie rule.
pub fn tie_heavy_value(s: &MatchState) -> MoneyChance {
    let w: [u128; 4] = core::array::from_fn(|t| {
        1 + u128::from(s.scores[t] >= 25) * 3
            + u128::from(s.games[t]) * 5
            + u128::from(s.dealer_team == t as u8)
    });
    MoneyChance::from_weights(w)
}

pub type L = FnLeaf<fn(&MatchState) -> MoneyChance>;

pub fn leaf() -> L {
    FnLeaf(fake_value)
}

pub fn tie_leaf() -> L {
    FnLeaf(tie_heavy_value)
}

/// Play random legal actions from a seeded match until the start of trick `k` (k >= 2).
pub fn position_at(seed: u64, k: u8) -> Match {
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

/// Like [`position_at`], then `extra` more random plays (so the root can be mid-trick).
pub fn position_mid(seed: u64, k: u8, extra: u8) -> Match {
    let mut m = position_at(seed, k);
    let mut rng = Rng::derive(seed, &[u64::from(extra)]);
    for _ in 0..extra {
        let acts = m.legal_actions();
        m.apply(acts[rng.bounded(acts.len() as u64) as usize])
            .expect("legal");
    }
    m
}

/// The value of a finished line from the engine's own state.
pub fn engine_leaf_value<Lf: Leaf>(m: &Match, leaf: &Lf) -> MoneyChance {
    if m.phase() == Phase::MatchOver {
        let w = (0..4u8)
            .find(|&t| m.games_won()[usize::from(t)] >= 2)
            .expect("a winner");
        return MoneyChance::certain(w);
    }
    leaf.value(&MatchState {
        scores: m.scores(),
        games: m.games_won(),
        dealer_team: m.dealer() % 4,
    })
}

/// The brute-force max^n value by engine clones.
pub fn brute<Lf: Leaf>(m: &Match, hand0: u32, leaf: &Lf) -> MoneyChance {
    if m.phase() == Phase::MatchOver || m.hand_index() != hand0 {
        return engine_leaf_value(m, leaf);
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
