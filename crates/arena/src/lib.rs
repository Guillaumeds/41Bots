//! The arena (docs/M1-SPEC.md §5): duplicate blocks, the generalised sequential test, offline
//! Plackett-Luce ratings and the identical-bots canary.
//!
//! Determinism: every block score is an exact whole number (see [`block::SCORE_DEN`]); results
//! from worker threads are merged in block order; floats appear only in reports (SPRT, ratings).

pub mod block;
pub mod rating;
pub mod run;
pub mod sprt;

pub use block::{Assignment, BlockKind, BlockResult, block_seed, play_block};
pub use run::{RunConfig, Summary, run_blocks};
pub use sprt::{Decision, SprtParams, llr};
