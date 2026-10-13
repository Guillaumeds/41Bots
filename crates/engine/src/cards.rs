//! Cards and bitboards.
//!
//! A card is a `u8` index: `suit * 13 + (rank - 1)` for the 52 ordinary cards, where suits are
//! clubs 0, diamonds 1, hearts 2, spades 3 and ranks run ace 1, 2..10, jack 11, queen 12, king 13.
//! The Joker is 52. A set of cards is a `Bits` (`u64`) with one bit per card.

/// A set of cards, one bit per card index.
pub type Bits = u64;

/// A card index (0..=52).
pub type Card = u8;

/// The Joker's card index.
pub const JOKER: Card = 52;

/// Number of cards in the pack.
pub const DECK_SIZE: usize = 53;

/// Every card in the pack.
pub const ALL_CARDS: Bits = (1u64 << 53) - 1;

/// Rank of an ace.
pub const ACE: u8 = 1;
/// Rank of a jack.
pub const JACK: u8 = 11;
/// Rank of a queen.
pub const QUEEN: u8 = 12;
/// Rank of a king.
pub const KING: u8 = 13;

/// The four suits.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Suit {
    /// Clubs (black).
    Clubs = 0,
    /// Diamonds (red).
    Diamonds = 1,
    /// Hearts (red).
    Hearts = 2,
    /// Spades (black).
    Spades = 3,
}

impl Suit {
    /// All four suits in index order.
    pub const ALL: [Suit; 4] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];

    /// Suit from its index (0..4).
    pub const fn from_index(i: u8) -> Suit {
        match i {
            0 => Suit::Clubs,
            1 => Suit::Diamonds,
            2 => Suit::Hearts,
            _ => Suit::Spades,
        }
    }

    /// Red suits are diamonds and hearts.
    pub const fn is_red(self) -> bool {
        matches!(self, Suit::Diamonds | Suit::Hearts)
    }

    /// The 13 cards of this suit.
    pub const fn mask(self) -> Bits {
        0x1FFF << (13 * (self as u64))
    }

    /// One-letter name.
    pub const fn letter(self) -> char {
        match self {
            Suit::Clubs => 'C',
            Suit::Diamonds => 'D',
            Suit::Hearts => 'H',
            Suit::Spades => 'S',
        }
    }
}

/// Build a card index from suit and rank (1..=13).
pub const fn card(suit: Suit, rank: u8) -> Card {
    debug_assert!(rank >= 1 && rank <= 13);
    (suit as u8) * 13 + (rank - 1)
}

/// The suit of a card, or `None` for the Joker.
pub const fn suit_of(c: Card) -> Option<Suit> {
    if c >= JOKER {
        None
    } else {
        Some(Suit::from_index(c / 13))
    }
}

/// The rank of a card (1..=13), or 0 for the Joker.
pub const fn rank_of(c: Card) -> u8 {
    if c >= JOKER { 0 } else { c % 13 + 1 }
}

/// The bit for one card.
pub const fn bit(c: Card) -> Bits {
    1u64 << c
}

/// Is the card in the set?
pub const fn has(set: Bits, c: Card) -> bool {
    set & bit(c) != 0
}

/// Number of cards in a set.
pub const fn count(set: Bits) -> u8 {
    set.count_ones() as u8
}

/// Iterate the cards of a set in ascending index order.
pub fn cards(set: Bits) -> impl Iterator<Item = Card> {
    let mut s = set;
    core::iter::from_fn(move || {
        if s == 0 {
            None
        } else {
            let c = s.trailing_zeros() as Card;
            s &= s - 1;
            Some(c)
        }
    })
}

/// Short name such as `5H`, `10S`, `AH`, `JK`.
pub fn name(c: Card) -> String {
    match suit_of(c) {
        None => "JK".to_string(),
        Some(s) => {
            let r = match rank_of(c) {
                ACE => "A".to_string(),
                JACK => "J".to_string(),
                QUEEN => "Q".to_string(),
                KING => "K".to_string(),
                n => n.to_string(),
            };
            format!("{r}{}", s.letter())
        }
    }
}

/// Parse a short name (`5H`, `10S`, `AH`, `JK`); case-insensitive.
pub fn parse(s: &str) -> Option<Card> {
    let s = s.trim().to_ascii_uppercase();
    if s == "JK" || s == "JOKER" {
        return Some(JOKER);
    }
    let (rank_str, suit_ch) = s.split_at(s.len().checked_sub(1)?);
    let suit = match suit_ch {
        "C" => Suit::Clubs,
        "D" => Suit::Diamonds,
        "H" => Suit::Hearts,
        "S" => Suit::Spades,
        _ => return None,
    };
    let rank = match rank_str {
        "A" => ACE,
        "J" => JACK,
        "Q" => QUEEN,
        "K" => KING,
        n => {
            let v: u8 = n.parse().ok()?;
            if (2..=10).contains(&v) {
                v
            } else {
                return None;
            }
        }
    };
    Some(card(suit, rank))
}

/// Names of all cards in a set, in index order, space-separated.
pub fn names(set: Bits) -> String {
    cards(set).map(name).collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_round_trip() {
        for s in Suit::ALL {
            for r in 1..=13u8 {
                let c = card(s, r);
                assert_eq!(suit_of(c), Some(s));
                assert_eq!(rank_of(c), r);
                assert_eq!(parse(&name(c)), Some(c));
            }
        }
        assert_eq!(suit_of(JOKER), None);
        assert_eq!(rank_of(JOKER), 0);
        assert_eq!(parse("jk"), Some(JOKER));
        assert_eq!(parse("10S"), Some(card(Suit::Spades, 10)));
        assert_eq!(parse("1S"), None);
        assert_eq!(parse("11S"), None);
    }

    #[test]
    fn masks_and_iteration() {
        assert_eq!(count(ALL_CARDS), 53);
        assert_eq!(count(Suit::Hearts.mask()), 13);
        let set = bit(card(Suit::Clubs, 2)) | bit(JOKER);
        assert_eq!(
            cards(set).collect::<Vec<_>>(),
            vec![card(Suit::Clubs, 2), JOKER]
        );
        assert_eq!(names(set), "2C JK");
    }
}
