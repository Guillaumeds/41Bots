//! The card tracker and the world generator of checkpoint M3 (docs/M3-SPEC.md §3–4).
//!
//! - [`tracker`]: the hard facts [certainties from the rules and the cards seen] about where each
//!   hidden card can be, as one mask of allowed places per card (facts F1–F12).
//! - [`space`]: the worlds [complete guesses of where every hidden card is] those facts allow:
//!   counted exactly (`u128`), checked by max-flow, sampled exactly uniformly and enumerated.
//! - [`truth`]: the true deal of a match in the tracker's terms, for tests and measurements.
//!
//! Integer logic only; randomness only from `engine::rng::Rng`; no threads and no file I/O, so
//! the crate builds for wasm32.

pub mod space;
pub mod tracker;
pub mod truth;

pub use space::{Infeasible, World, WorldSpace, log2_q8};
pub use tracker::{Loc, Tracker};

/// Labels for seed derivation (docs/M3-SPEC.md §1, §4.4–4.5).
pub mod labels {
    /// World number `w` of a decision: `Rng::derive(dseed, &[WORLD, w])`.
    pub const WORLD: u64 = 0x3071D;
    /// The next turned card during the dealer's burn after a turned Joker:
    /// `Rng::derive(dseed, &[WORLD, w, NEXT_TURNED])` (deck position 41).
    pub const NEXT_TURNED: u64 = 41;
}

/// The tracker and world space of a view in one call: [`Tracker::from_view`] then
/// [`WorldSpace::new`].
pub fn world_space(view: &engine::view::SeatView) -> Result<(Tracker, WorldSpace), Infeasible> {
    let t = Tracker::from_view(view);
    let ws = WorldSpace::new(&t)?;
    Ok((t, ws))
}

#[cfg(test)]
mod tests;
