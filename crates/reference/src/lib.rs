//! Reference implementation of the rules of 41 (docs/RULES.md), for checkpoint M0.
//!
//! This crate exists to catch misreadings in the main engine. It is written
//! independently and naively, straight from docs/RULES.md and
//! docs/M0-SPEC.md sections 1, 2, 4 and 5: `Vec` hands, enums for suit and
//! rank, card orders written out as explicit lists, and the hand flow as
//! explicit steps. It has no dependencies and never looks at the engine.
//!
//! It does not shuffle: the differential test hands it each deck.

pub mod card;
pub mod game;
pub mod order;

#[cfg(test)]
pub(crate) mod test_support;

pub use card::{Card, Rank, Suit, index_of_name, name_of_index};
pub use game::{Action, Lapse, Match, Phase, Public, Rules};
