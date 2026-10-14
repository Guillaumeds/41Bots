//! The PIMC teacher of checkpoint M3 (docs/M3-SPEC.md §5, §8, §13).
//!
//! - [`worlds`]: worlds [complete guesses of where every hidden card is], the [`WorldSource`]
//!   trait with the counted belief sampler (default) and the rejection oracle, and rebuilding a
//!   world as a match.
//! - [`pimc`]: rollout-PIMC (`rules-v1` inner player) and exact PIMC (`solve_fast`), scored in
//!   money chance, with common worlds for every candidate and thread-independent results.
//! - [`bot`]: the `pimc-v0` / `pimc-teacher` bots, the robber's joint (burn, first card)
//!   decision, and the bot registry used by `arena`.
//! - [`measure`]: stratified decision sets and the early-trick yardstick measurements (§13.1).
//! - [`teacher`]: teacher hands and their output files (§5.10).
//!
//! Everything that scores is fixed point ([`equity::MoneyChance`]); floats appear in reports
//! only.

pub mod bot;
pub mod measure;
pub mod pimc;
pub mod teacher;
pub mod worlds;

pub use bot::{PimcBot, PimcFactory, PimcSpec, factory};
pub use pimc::{DecisionValues, SearchConfig, evaluate};
pub use worlds::{BeliefSource, RejectionSource, World, WorldSource, Worlds, world_count};

/// Labels for seed derivation (docs/M3-SPEC.md §1).
pub mod labels {
    /// World number `w` of a decision: `Rng::derive(dseed, &[WORLD, w])`.
    pub const WORLD: u64 = 0x3071D;
    /// Randomness inside a rollout.
    pub const ROLLOUT: u64 = 0x2011;
    /// The decision seed.
    pub const DECISION: u64 = 0xDEC1;
    /// A teacher hand's deck.
    pub const TEACHER_HAND: u64 = 0x7EAC;
    /// Stratified start states.
    pub const STRATUM: u64 = 0x5742;
    /// Measurement decision sets.
    pub const MEASURE: u64 = 0x3EA5;
    /// Agreement-curve and yardstick worlds.
    pub const AGREE: u64 = 0xA62E;
}

#[cfg(test)]
mod belief_tests;
#[cfg(test)]
mod tests;
