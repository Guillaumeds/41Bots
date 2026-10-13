//! A random-legal bot for testing the model estimator (and `equity estimate-model --bot
//! random-test`) until the real bots of `crates/bots` are available. It is not a baseline for
//! tournaments: use `bots::RandomBot` (`random`) for that.

use bot::{Bot, BotFactory};
use engine::game::{Action, Rules};
use engine::rng::Rng;
use engine::view::SeatView;

/// Plays a uniformly random legal action, from `Rng::derive(seed, &[decision index])`.
#[derive(Clone, Debug, Default)]
pub struct RandomTestBot {
    seed: u64,
    decisions: u64,
}

impl Bot for RandomTestBot {
    fn name(&self) -> &str {
        RANDOM_TEST
    }

    fn start_match(&mut self, _seat: u8, _rules: Rules, seed: u64) {
        self.seed = seed;
        self.decisions = 0;
    }

    fn act(&mut self, _view: &SeatView, legal: &[Action]) -> Action {
        let i = Rng::derive(self.seed, &[self.decisions]).bounded(legal.len() as u64);
        self.decisions += 1;
        legal[i as usize]
    }
}

/// The registry name of [`RandomTestBot`].
pub const RANDOM_TEST: &str = "random-test";

/// Builds [`RandomTestBot`]s.
#[derive(Clone, Copy, Debug, Default)]
pub struct RandomTestFactory;

impl BotFactory for RandomTestFactory {
    fn name(&self) -> &str {
        RANDOM_TEST
    }

    fn build(&self) -> Box<dyn Bot> {
        Box::new(RandomTestBot::default())
    }
}
