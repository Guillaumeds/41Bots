//! Random solver worlds for measurements and identity sweeps (the `solver-spike` and
//! `solver-identity` binaries).

use engine::cards::{Bits, Card, bit, cards};
use engine::game::{Action, Match, Phase, Rules};
use engine::rng::Rng;
use engine::rules::legal_burns;

use crate::{MatchCtx, Position, Step, from_match};

fn pick(rng: &mut Rng, set: Bits) -> Card {
    let all: Vec<Card> = cards(set).collect();
    all[rng.bounded(all.len() as u64) as usize]
}

/// World number `world` of a run with this `seed`, at the start of trick `k`, then `extra`
/// more random legal plays (stopping early if the hand would end): random deals from the engine
/// RNG (`Rng::derive(seed, [world])`), 0–3 whole hands skipped at random so the match scores
/// vary, then random legal play. At trick 1 an outstanding rob is resolved with a random legal
/// burn. Returns the position, the match context and the root leader's seat relative to the
/// dealer. With `extra = 0` these are exactly the worlds of the M2 measurement spike.
pub fn sample_world(seed: u64, k: u8, world: u64, extra: u8) -> (Position, MatchCtx, u8) {
    let mut rng = Rng::derive(seed, &[world]);
    let mut m = Match::new(rng.next_u64(), Rules::HOUSE);
    let skip = rng.bounded(4) as u32;
    loop {
        if m.phase() == Phase::MatchOver {
            m = Match::new(rng.next_u64(), Rules::HOUSE);
            continue;
        }
        let p = m.public();
        let at_start = p.trick_no == k && p.trick_cards.is_empty() && m.hand_index() >= skip;
        if at_start {
            match m.phase() {
                // The trick-1 leader robs before leading: burn at random, then solve.
                Phase::AwaitBurn(_) if k == 1 => {}
                Phase::AwaitPlay(_) => {
                    let (mut pos, ctx, pending) = from_match(&m).expect("awaiting a play");
                    if let Some(s) = pending {
                        let table = p.table_card.expect("turned card on the table");
                        let h = pos.hands[usize::from(s)] | bit(table);
                        let burn = pick(&mut rng, legal_burns(h, p.trump));
                        pos.hands[usize::from(s)] = h & !bit(burn);
                    }
                    let rel = (pos.leader + 8 - m.dealer()) % 8;
                    let mut r2 = Rng::derive(seed, &[world, 1]);
                    for _ in 0..extra {
                        let c = pick(&mut r2, pos.legal());
                        match pos.play(c) {
                            Step::Next(q) if q.hands.iter().any(|&h| h != 0) => pos = q,
                            _ => break,
                        }
                    }
                    return (pos, ctx, rel);
                }
                _ => {}
            }
        }
        let acts = m.legal_actions();
        let a: Action = acts[rng.bounded(acts.len() as u64) as usize];
        m.apply(a).expect("legal");
    }
}
