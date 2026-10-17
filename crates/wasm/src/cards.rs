//! The app's card, suit and action numbering, and the engine's (docs/WASM-API.md §3; contract
//! §6.1).
//!
//! The app numbers a card `suit * 13 + (rank - 1)` with hearts 0, diamonds 1, clubs 2, spades 3
//! and the Joker 52 (53 is the second Joker of a variant the engine refuses); the engine the same
//! way with clubs 0, diamonds 1, hearts 2, spades 3. So the blocks 0–12 and 26–38 swap, and the
//! diamonds (13–25), the spades (39–51) and the Joker (52) keep their numbers. The mapping is its
//! own inverse.

use engine::cards::{Card, JOKER, Suit};

/// The app's number of the first Joker.
pub const APP_JOKER: i64 = 52;

/// The engine's card for an app card: `None` for anything that is not one of the 53 cards (the
/// second Joker, 53, included).
pub fn engine_card(app: i64) -> Option<Card> {
    match app {
        0..=12 => Some(app as Card + 26),
        26..=38 => Some(app as Card - 26),
        13..=25 | 39..=52 => Some(app as Card),
        _ => None,
    }
}

/// The app's number for an engine card.
///
/// # Panics
/// If `c` is not a card (0..=52).
pub fn app_card(c: Card) -> u8 {
    match c {
        0..=12 => c + 26,
        26..=38 => c - 26,
        13..=25 | 39..=JOKER => c,
        _ => panic!("no such card: {c}"),
    }
}

/// The engine's suit for an app suit (hearts 0, diamonds 1, clubs 2, spades 3).
pub fn engine_suit(app: i64) -> Option<Suit> {
    match app {
        0 => Some(Suit::Hearts),
        1 => Some(Suit::Diamonds),
        2 => Some(Suit::Clubs),
        3 => Some(Suit::Spades),
        _ => None,
    }
}

/// The app's number for an engine suit.
pub fn app_suit(s: Suit) -> u8 {
    match s {
        Suit::Hearts => 0,
        Suit::Diamonds => 1,
        Suit::Clubs => 2,
        Suit::Spades => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::cards::{ACE, DECK_SIZE, card, rank_of, suit_of};

    #[test]
    fn every_card_maps_both_ways() {
        let mut seen = [false; DECK_SIZE];
        for app in 0..=52i64 {
            let c = engine_card(app).expect("a card");
            assert!(
                !seen[usize::from(c)],
                "two app cards map to one engine card"
            );
            seen[usize::from(c)] = true;
            assert_eq!(
                i64::from(app_card(c)),
                app,
                "app {app} -> engine {c} -> app"
            );
        }
        for c in 0..DECK_SIZE as Card {
            assert_eq!(engine_card(i64::from(app_card(c))), Some(c));
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn suits_and_ranks_agree() {
        // The app's card is suit*13 + rank-1 in its own suit order.
        for app in 0..52i64 {
            let c = engine_card(app).expect("a card");
            let s = suit_of(c).expect("not the Joker");
            assert_eq!(i64::from(app_suit(s)), app / 13, "suit of app card {app}");
            assert_eq!(
                i64::from(rank_of(c)),
                app % 13 + 1,
                "rank of app card {app}"
            );
            assert_eq!(engine_suit(app / 13), Some(s));
        }
        // The contract's examples: the app's A of hearts (0) is the engine's 26, and back.
        assert_eq!(engine_card(0), Some(card(Suit::Hearts, ACE)));
        assert_eq!(card(Suit::Hearts, ACE), 26);
        assert_eq!(app_card(26), 0);
        assert_eq!(engine_card(APP_JOKER), Some(JOKER));
        assert_eq!(app_card(JOKER), 52);
    }

    #[test]
    fn non_cards_are_refused() {
        for bad in [-1, 53, 54, 1000, i64::MIN, i64::MAX] {
            assert_eq!(engine_card(bad), None, "{bad}");
        }
        for bad in [-1, 4, 99] {
            assert_eq!(engine_suit(bad), None);
        }
    }
}
