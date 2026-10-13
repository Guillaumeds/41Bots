//! `random`: a uniformly random legal action.

use bot::Bot;
use engine::game::{Action, Rules};
use engine::rng::Rng;
use engine::view::SeatView;

use crate::RANDOM;

/// Picks one of the legal actions uniformly at random.
///
/// Decision number `k` of a match (counting this bot's own decisions from 0, reset by
/// [`Bot::start_match`]) uses the generator `Rng::derive(seed, &[k])`, where `seed` is the seed
/// the table gave at the start of the match, so the same seed always gives the same choices.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RandomBot {
    seed: u64,
    decision_index: u64,
}

impl RandomBot {
    /// A random bot; the seed comes with [`Bot::start_match`].
    pub fn new() -> Self {
        Self::default()
    }
}

impl Bot for RandomBot {
    fn name(&self) -> &str {
        RANDOM
    }

    fn start_match(&mut self, _seat: u8, _rules: Rules, seed: u64) {
        self.seed = seed;
        self.decision_index = 0;
    }

    fn act(&mut self, _view: &SeatView, legal: &[Action]) -> Action {
        assert!(
            !legal.is_empty(),
            "the random bot was asked to act with no legal action"
        );
        let mut rng = Rng::derive(self.seed, &[self.decision_index]);
        self.decision_index += 1;
        let i = rng.bounded(legal.len() as u64) as usize;
        legal[i]
    }
}
