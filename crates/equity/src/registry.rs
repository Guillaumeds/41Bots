//! Bot lookup by name for `equity estimate-model --bot <name>`.
//!
//! The production model v0 is estimated from `rules-v1` (docs/M1-SPEC.md §6.3), which lives in
//! `crates/bots`. This crate does not depend on `crates/bots` yet, so only the test bot
//! `random-test` is available here.
//!
//! **Where the registry call goes.** `crates/bots` exports
//! `bots::factory(name: &str) -> Option<Box<dyn BotFactory>>` (names `random`, `rules-v1`). Its
//! factories are not `Sync`, while the estimator shares one factory between threads, so wrap the
//! lookup in [`ByName`], which looks the bot up again on every build:
//!
//! 1. add `bots = { path = "../bots" }` to this crate's `[dependencies]`;
//! 2. in [`bot_factory`], replace the `_ => None` arm with
//!    `other => ByName::new(other, bots::factory).map(|f| Box::new(f) as Box<dyn BotFactory + Sync>),`
//! 3. add `bots::names()` to [`known_bots`].
//!
//! Nothing else in `equity` changes.

use bot::{Bot, BotFactory};

use crate::testbots::{RANDOM_TEST, RandomTestFactory};

/// A by-name registry lookup, with the signature of `bots::factory`.
pub type Lookup = fn(&str) -> Option<Box<dyn BotFactory>>;

/// A `Sync` factory over a registry whose factories are not `Sync`: it keeps only the name and
/// the lookup function, and looks the bot up again for every bot it builds.
#[derive(Clone, Debug)]
pub struct ByName {
    /// The key it is looked up by.
    key: String,
    /// The name the registry's factory reports.
    name: String,
    lookup: Lookup,
}

impl ByName {
    /// The adapter for `name`, or `None` if the registry does not know it.
    pub fn new(name: &str, lookup: Lookup) -> Option<Self> {
        lookup(name).map(|f| ByName {
            key: name.to_string(),
            name: f.name().to_string(),
            lookup,
        })
    }
}

impl BotFactory for ByName {
    fn name(&self) -> &str {
        &self.name
    }

    fn build(&self) -> Box<dyn Bot> {
        (self.lookup)(&self.key)
            .unwrap_or_else(|| panic!("bot `{}` disappeared from the registry", self.key))
            .build()
    }
}

/// A factory for the named bot, or `None` if the name is unknown.
pub fn bot_factory(name: &str) -> Option<Box<dyn BotFactory + Sync>> {
    match name {
        RANDOM_TEST => Some(Box::new(RandomTestFactory)),
        // REGISTRY HOOK (see the module docs):
        other => {
            ByName::new(other, bots::factory).map(|f| Box::new(f) as Box<dyn BotFactory + Sync>)
        }
    }
}

/// The names [`bot_factory`] knows today.
pub fn known_bots() -> Vec<&'static str> {
    let mut v = vec![RANDOM_TEST];
    v.extend(bots::names().iter().copied());
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in with exactly the signature of `bots::factory`.
    fn fake_registry(name: &str) -> Option<Box<dyn BotFactory>> {
        (name == "stand-in").then(|| Box::new(RandomTestFactory) as Box<dyn BotFactory>)
    }

    #[test]
    fn by_name_adapter_is_sync_and_builds() {
        fn needs_sync<T: Sync>(_: &T) {}
        let f = ByName::new("stand-in", fake_registry).expect("known");
        needs_sync(&f);
        assert_eq!(f.name(), "random-test", "the registry's own bot name");
        assert_eq!(f.build().name(), "random-test");
        assert!(ByName::new("nobody", fake_registry).is_none());
        // The adapter works in the threaded estimator and gives the same model.
        let direct = crate::estimate::estimate_model(&RandomTestFactory, 300, 4).expect("legal");
        let adapted = crate::estimate::estimate_model_parallel(&f, 300, 4, 2).expect("legal");
        assert_eq!(adapted, direct);
    }

    #[test]
    fn registry_knows_the_test_bot_and_the_bots_crate() {
        let f = bot_factory("random-test").expect("known");
        assert_eq!(f.name(), "random-test");
        assert_eq!(bot_factory("rules-v1").expect("known").name(), "rules-v1");
        assert!(bot_factory("nobody").is_none());
        assert_eq!(known_bots(), vec!["random-test", "random", "rules-v1"]);
    }
}
