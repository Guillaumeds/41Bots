//! Test-only bots (the real ones live in crates/bots, which tablemgr must not depend on).
#![allow(dead_code)]

use bot::Bot;
use engine::game::{Action, Rules};
use engine::rng::Rng;
use engine::view::SeatView;

/// Uniform over the legal actions, drawn from `Rng::derive(seed, &[decision index])`:
/// deterministic given its seed and the views it receives.
#[derive(Default)]
pub struct RandomLegal {
    seed: u64,
    decisions: u64,
}

impl Bot for RandomLegal {
    fn name(&self) -> &str {
        "test-random"
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

/// Always the lowest legal card.
pub struct Lowest;

impl Bot for Lowest {
    fn name(&self) -> &str {
        "test-lowest"
    }
    fn start_match(&mut self, _seat: u8, _rules: Rules, _seed: u64) {}
    fn act(&mut self, _view: &SeatView, legal: &[Action]) -> Action {
        legal[0]
    }
}

/// The test bot for a seat: random on even seats (teams 0 and 2), lowest on odd seats.
pub fn bot_for_seat(seat: usize) -> Box<dyn Bot> {
    if seat.is_multiple_of(2) {
        Box::new(RandomLegal::default())
    } else {
        Box::new(Lowest)
    }
}

/// A full table of [`bot_for_seat`] bots.
pub fn mixed_bots() -> [Box<dyn Bot>; 8] {
    core::array::from_fn(bot_for_seat)
}

/// Eight random bots.
pub fn random_bots() -> [Box<dyn Bot>; 8] {
    core::array::from_fn(|_| Box::new(RandomLegal::default()) as Box<dyn Bot>)
}

/// Plays like `RandomLegal`, but at its `bad_at`th decision returns an action that is not legal.
pub struct Cheater {
    inner: RandomLegal,
    decisions: u64,
    bad_at: u64,
}

impl Cheater {
    pub fn new(bad_at: u64) -> Cheater {
        Cheater {
            inner: RandomLegal::default(),
            decisions: 0,
            bad_at,
        }
    }
}

/// An action that is not in `legal`.
pub fn not_legal(legal: &[Action]) -> Action {
    (0..53u8)
        .map(Action::Play)
        .chain((0..53u8).map(Action::Burn))
        .find(|a| !legal.contains(a))
        .expect("some action is always illegal")
}

impl Bot for Cheater {
    fn name(&self) -> &str {
        "test-cheater"
    }
    fn start_match(&mut self, seat: u8, rules: Rules, seed: u64) {
        self.inner.start_match(seat, rules, seed);
        self.decisions = 0;
    }
    fn act(&mut self, view: &SeatView, legal: &[Action]) -> Action {
        self.decisions += 1;
        if self.decisions == self.bad_at {
            return not_legal(legal);
        }
        self.inner.act(view, legal)
    }
}
