//! The baseline bots of checkpoint M1 (docs/M1-SPEC.md §4).
//!
//! - [`RandomBot`] (`random`): a uniformly random legal action, drawn from the engine's pinned
//!   random-number generator.
//! - [`RulesBot`] (`rules-v1`): the published 25/45-family heuristics, written out as an exact,
//!   deterministic decision list (see [`rules_v1`]).
//!
//! Both bots read nothing but their own [`engine::view::SeatView`] and the legal actions, and
//! both are deterministic given their seed and the views they receive, so duplicate matches
//! replay exactly. Bots are chosen by name through [`factory`]; [`names`] lists the names.

pub mod random;
pub mod rules_v1;
pub mod tracker;

pub use random::RandomBot;
pub use rules_v1::{Rule, RulesBot, decide};
pub use tracker::Tracker;

use bot::{Bot, BotFactory};

/// The registry name of [`RandomBot`].
pub const RANDOM: &str = "random";

/// The registry name of [`RulesBot`].
pub const RULES_V1: &str = "rules-v1";

static NAMES: [&str; 2] = [RANDOM, RULES_V1];

/// Every bot name the registry knows, in a fixed order.
pub fn names() -> &'static [&'static str] {
    &NAMES
}

/// The factory for the bot with this name, or `None` if no bot has that name.
pub fn factory(name: &str) -> Option<Box<dyn BotFactory>> {
    match name {
        RANDOM => Some(Box::new(RandomFactory)),
        RULES_V1 => Some(Box::new(RulesFactory)),
        _ => None,
    }
}

/// Builds [`RandomBot`]s.
struct RandomFactory;

impl BotFactory for RandomFactory {
    fn name(&self) -> &str {
        RANDOM
    }
    fn build(&self) -> Box<dyn Bot> {
        Box::new(RandomBot::new())
    }
}

/// Builds [`RulesBot`]s.
struct RulesFactory;

impl BotFactory for RulesFactory {
    fn name(&self) -> &str {
        RULES_V1
    }
    fn build(&self) -> Box<dyn Bot> {
        Box::new(RulesBot::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_knows_both_bots() {
        assert_eq!(names(), &["random", "rules-v1"]);
        for &n in names() {
            let f = factory(n).expect("a listed name has a factory");
            assert_eq!(f.name(), n);
            assert_eq!(f.build().name(), n);
        }
        assert!(factory("rules-v2").is_none());
        assert!(factory("").is_none());
    }
}
