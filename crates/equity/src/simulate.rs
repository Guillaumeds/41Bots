//! A Monte-Carlo simulator that plays whole matches from any state using the hand-outcome model
//! (docs/M1-SPEC.md §6.4 validation: a direct estimate that must agree with the table). It works
//! in the real team order and converts to the model's dealer-relative offsets itself, so it also
//! checks the sweep's canonical-rotation bookkeeping independently.

use engine::rng::Rng;

use crate::hand::{AfterHand, AfterTrick, InHand, Scoring};
use crate::model::{CompiledModel, N_BEFORE_TRICK, comp_index};
use crate::state::MatchState;

/// Draw an index from a fixed-point distribution (summing to 2^32), exactly.
pub fn sample_index(dist: &[u64], rng: &mut Rng) -> usize {
    let r = rng.next_u64() >> 32;
    let mut acc = 0u64;
    for (i, &p) in dist.iter().enumerate() {
        acc += p;
        if r < acc {
            return i;
        }
    }
    unreachable!("a fixed-point distribution sums to 2^32")
}

/// Play one hand from a state (the dealer of the hand is in `state.dealer_team`); returns the
/// state the match moves to.
pub fn simulate_hand(
    model: &CompiledModel,
    sc: Scoring,
    state: &MatchState,
    rng: &mut Rng,
) -> MatchState {
    let d = usize::from(state.dealer_team);
    let abs = |o: usize| ((o + d) % 4) as u8;
    let mut pos = InHand::start(state.scores);
    let next_dealer = (state.dealer_team + 1) % 4;
    let game_won = |t: u8| {
        let mut games = state.games;
        games[usize::from(t)] += 1;
        MatchState {
            scores: [0; 4],
            games,
            dealer_team: next_dealer,
        }
    };
    loop {
        let rel: [u8; 4] = core::array::from_fn(|o| pos.tricks[usize::from(abs(o))]);
        let k = comp_index(rel);
        if pos.tricks_done == engine::game::TRICKS_PER_HAND {
            let best = if pos.five_played {
                None
            } else {
                Some(abs(sample_index(&model.end_bonus[k - N_BEFORE_TRICK], rng)))
            };
            return match pos.finish(best, sc) {
                AfterHand::GameWon(t, _) => game_won(t),
                AfterHand::HandOver(scores) => MatchState {
                    scores,
                    games: state.games,
                    dealer_team: next_dealer,
                },
            };
        }
        let o = sample_index(&model.winner[k], rng);
        let here = !pos.five_played
            && sample_index(&model.five[usize::from(pos.tricks_done)][o], rng) == 1;
        match pos.play_trick(abs(o), here, sc) {
            AfterTrick::GameWon(t, _) => return game_won(t),
            AfterTrick::Next(p) | AfterTrick::TricksDone(p) => pos = p,
        }
    }
}

/// Play a whole match from a state; returns the team that wins the money.
pub fn simulate_match(model: &CompiledModel, sc: Scoring, start: &MatchState, rng: &mut Rng) -> u8 {
    let mut s = *start;
    loop {
        if let Some(w) = s.winner() {
            return w;
        }
        s = simulate_hand(model, sc, &s, rng);
    }
}
