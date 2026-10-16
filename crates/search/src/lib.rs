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
//! - [`curves`]: N-curves from saved per-world values: money loss against N, power-law fits,
//!   held-out checks and the N each scenario needs (§13.1 item 10).
//! - [`eta`]: honest progress for a worlds run: work done weighted by expected cost per task,
//!   refined by measured times, and the time and money to the end.
//! - [`plan`]: position-aware threads and tables per exact world, and the runner that packs
//!   them onto the machine (§13.1 item 11).
//! - [`t1bench`]: the trick-1 machine bench: exact-solve work per second and per dollar with
//!   every thread busy, and a check that the production schedule fills the machine (§13.1
//!   item 11).
//! - [`teacher`]: teacher hands and their output files (§5.10).
//! - [`hybrid`]: the hybrid yardstick check (docs/M3-HYBRID-SPEC.md §3): exact against H_k on the
//!   same worlds; decision loss with a bootstrap bound, value error, agreement.
//! - [`oracle_labels`]: training labels for the oracle networks of the hybrid yardstick
//!   (docs/M3-HYBRID-SPEC.md §5): positions at the start of a trick, encoded, valued exactly.
//!
//! Everything that scores is fixed point ([`equity::MoneyChance`]); floats appear in reports
//! only.

pub mod bot;
pub mod curves;
pub mod eta;
pub mod hybrid;
pub mod measure;
pub mod oracle_labels;
pub mod ordered;
pub mod pimc;
pub mod plan;
pub mod saved;
pub mod t1bench;
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
#[cfg(test)]
mod yardstick_probes;
