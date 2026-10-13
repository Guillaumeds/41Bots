//! The money-chance type and the exact match-equity table (docs/M1-SPEC.md §6; PLAN §5.3, §6.6
//! and amendment A1).
//!
//! - [`money`]: [`MoneyChance`], four whole numbers in units of 2^-32 summing to exactly 1, the
//!   only score anything in the project produces.
//! - [`state`]: [`MatchState`] (scores, games, dealer team) and the canonical table index.
//! - [`model`]: the hand-outcome model v0 (three count tables relative to the dealer's team), its
//!   JSON file and its fixed-point form; [`estimate`] measures it from self-play.
//! - [`hand`]: scoring inside a hand in the engine's exact order.
//! - [`sweep`]: exact backward induction over all 810,000 canonical states, multi-threaded with
//!   byte-identical results; [`table`] stores it as `.npy` plus a JSON side file.
//! - [`simulate`]: a Monte-Carlo match simulator from the model; [`validate`]: the table checks
//!   of docs/M1-SPEC.md §6.4 (sums, rotation symmetry, sanity, monotonicity, Monte Carlo, and the
//!   reachable-score and bonus-flag properties).
//! - [`npy`] and [`sha256`]: the `.npy` writer/reader and content hashing (PLAN §12.7).
//! - [`registry`] and [`testbots`]: bot lookup by name for `equity estimate-model`, with a
//!   random-legal test bot until `crates/bots` is wired in (see [`registry`]).
//!
//! Everything that enters the table is whole-number arithmetic; floating point appears only when
//! printing.

pub mod estimate;
pub mod hand;
pub mod model;
pub mod money;
pub mod npy;
pub mod registry;
pub mod sha256;
pub mod simulate;
pub mod state;
pub mod sweep;
pub mod table;
pub mod testbots;
pub mod validate;

pub use money::{MoneyChance, ONE};
pub use state::MatchState;
pub use table::EquityTable;

#[cfg(test)]
mod tests;
