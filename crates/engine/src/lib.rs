//! 41 rules engine.
//!
//! The single source of truth for the rules in `docs/RULES.md`. It is built natively for Windows
//! and Linux, and for `wasm32-unknown-unknown`. It uses whole-number logic only, plus a pinned,
//! deterministic RNG, so any game can be replayed exactly. See `docs/STATUS.md` for the M0 task list.

/// Number of cards in a 41 pack: 52 ordinary cards plus one Joker.
pub const DECK_SIZE: usize = 53;

/// Number of seats at a 41 table.
pub const SEATS: usize = 8;

/// Number of partnerships (teams of two, partners opposite).
pub const TEAMS: usize = 4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_constants() {
        assert_eq!(DECK_SIZE, 53);
        assert_eq!(SEATS, 2 * TEAMS);
    }
}
