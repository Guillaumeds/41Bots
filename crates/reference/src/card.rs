//! Cards: suits, ranks and the Joker, plus the conversion to and from the
//! shared card index of docs/M0-SPEC.md section 1:
//!
//! `index = suit * 13 + (rank - 1)`, suits clubs 0, diamonds 1, hearts 2,
//! spades 3, ranks ace 1, 2..10, jack 11, queen 12, king 13; the Joker is 52.

/// The four suits, in the order of the shared card index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Suit {
    Clubs,
    Diamonds,
    Hearts,
    Spades,
}

/// Every suit, in index order.
pub const ALL_SUITS: [Suit; 4] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];

impl Suit {
    /// The suit's number in the shared card index.
    pub fn index(self) -> u8 {
        match self {
            Suit::Clubs => 0,
            Suit::Diamonds => 1,
            Suit::Hearts => 2,
            Suit::Spades => 3,
        }
    }

    pub fn from_index(i: u8) -> Option<Suit> {
        match i {
            0 => Some(Suit::Clubs),
            1 => Some(Suit::Diamonds),
            2 => Some(Suit::Hearts),
            3 => Some(Suit::Spades),
            _ => None,
        }
    }

    /// The letter used in canonical card names.
    pub fn letter(self) -> &'static str {
        match self {
            Suit::Clubs => "C",
            Suit::Diamonds => "D",
            Suit::Hearts => "H",
            Suit::Spades => "S",
        }
    }

    /// Diamonds and hearts are red; clubs and spades are black.
    pub fn is_red(self) -> bool {
        match self {
            Suit::Diamonds | Suit::Hearts => true,
            Suit::Clubs | Suit::Spades => false,
        }
    }
}

/// The thirteen ranks of an ordinary card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rank {
    Ace,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
}

/// Every rank, in index order (ace first, king last).
pub const ALL_RANKS: [Rank; 13] = [
    Rank::Ace,
    Rank::Two,
    Rank::Three,
    Rank::Four,
    Rank::Five,
    Rank::Six,
    Rank::Seven,
    Rank::Eight,
    Rank::Nine,
    Rank::Ten,
    Rank::Jack,
    Rank::Queen,
    Rank::King,
];

impl Rank {
    /// The rank's number: ace 1, 2..10, jack 11, queen 12, king 13.
    pub fn number(self) -> u8 {
        match self {
            Rank::Ace => 1,
            Rank::Two => 2,
            Rank::Three => 3,
            Rank::Four => 4,
            Rank::Five => 5,
            Rank::Six => 6,
            Rank::Seven => 7,
            Rank::Eight => 8,
            Rank::Nine => 9,
            Rank::Ten => 10,
            Rank::Jack => 11,
            Rank::Queen => 12,
            Rank::King => 13,
        }
    }

    pub fn from_number(n: u8) -> Option<Rank> {
        ALL_RANKS.iter().copied().find(|r| r.number() == n)
    }

    /// The rank part of a canonical card name.
    pub fn name(self) -> &'static str {
        match self {
            Rank::Ace => "A",
            Rank::Two => "2",
            Rank::Three => "3",
            Rank::Four => "4",
            Rank::Five => "5",
            Rank::Six => "6",
            Rank::Seven => "7",
            Rank::Eight => "8",
            Rank::Nine => "9",
            Rank::Ten => "10",
            Rank::Jack => "J",
            Rank::Queen => "Q",
            Rank::King => "K",
        }
    }
}

/// A card: the Joker or an ordinary card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Card {
    Joker,
    Normal(Rank, Suit),
}

/// The Joker's card index.
pub const JOKER_INDEX: u8 = 52;

impl Card {
    pub const fn new(rank: Rank, suit: Suit) -> Card {
        Card::Normal(rank, suit)
    }

    /// The shared card index (M0-SPEC section 1).
    pub fn index(self) -> u8 {
        match self {
            Card::Joker => JOKER_INDEX,
            Card::Normal(rank, suit) => suit.index() * 13 + (rank.number() - 1),
        }
    }

    /// The card with this shared index, or `None` if the index is not 0..=52.
    pub fn from_index(i: u8) -> Option<Card> {
        if i == JOKER_INDEX {
            return Some(Card::Joker);
        }
        if i > JOKER_INDEX {
            return None;
        }
        let suit = Suit::from_index(i / 13)?;
        let rank = Rank::from_number(i % 13 + 1)?;
        Some(Card::Normal(rank, suit))
    }

    /// The canonical name: rank then suit letter (`5H`, `10S`, `AH`), or `JK`.
    pub fn name(self) -> String {
        match self {
            Card::Joker => "JK".to_string(),
            Card::Normal(rank, suit) => format!("{}{}", rank.name(), suit.letter()),
        }
    }

    /// Parses a canonical name such as `5H`, `10S` or `JK`.
    pub fn from_name(name: &str) -> Option<Card> {
        if name == "JK" {
            return Some(Card::Joker);
        }
        all_cards().into_iter().find(|c| c.name() == name)
    }

    /// The suit, or `None` for the Joker.
    pub fn suit(self) -> Option<Suit> {
        match self {
            Card::Joker => None,
            Card::Normal(_, suit) => Some(suit),
        }
    }

    /// The rank, or `None` for the Joker.
    pub fn rank(self) -> Option<Rank> {
        match self {
            Card::Joker => None,
            Card::Normal(rank, _) => Some(rank),
        }
    }

    /// True for the ace of any suit (the Joker is not an ace).
    pub fn is_ace(self) -> bool {
        self.rank() == Some(Rank::Ace)
    }
}

/// All 53 cards, in index order.
pub fn all_cards() -> Vec<Card> {
    let mut cards = Vec::new();
    for suit in ALL_SUITS {
        for rank in ALL_RANKS {
            cards.push(Card::Normal(rank, suit));
        }
    }
    cards.push(Card::Joker);
    cards
}

/// The canonical name of a card index. Panics on an index above 52.
pub fn name_of_index(i: u8) -> String {
    match Card::from_index(i) {
        Some(card) => card.name(),
        None => panic!("card index {i} is out of range"),
    }
}

/// The card index of a canonical name.
pub fn index_of_name(name: &str) -> Option<u8> {
    Card::from_name(name).map(Card::index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_formula_matches_spec_section_1() {
        // A few hand-checked values of suit * 13 + (rank - 1).
        assert_eq!(Card::new(Rank::Ace, Suit::Clubs).index(), 0);
        assert_eq!(Card::new(Rank::Ten, Suit::Clubs).index(), 9);
        assert_eq!(Card::new(Rank::King, Suit::Clubs).index(), 12);
        assert_eq!(Card::new(Rank::Ace, Suit::Diamonds).index(), 13);
        assert_eq!(Card::new(Rank::Five, Suit::Diamonds).index(), 17);
        assert_eq!(Card::new(Rank::Ace, Suit::Hearts).index(), 26);
        assert_eq!(Card::new(Rank::Five, Suit::Hearts).index(), 30);
        assert_eq!(Card::new(Rank::Jack, Suit::Hearts).index(), 36);
        assert_eq!(Card::new(Rank::Ace, Suit::Spades).index(), 39);
        assert_eq!(Card::new(Rank::King, Suit::Spades).index(), 51);
        assert_eq!(Card::Joker.index(), 52);
    }

    #[test]
    fn every_index_round_trips() {
        let cards = all_cards();
        assert_eq!(cards.len(), 53);
        for (i, card) in cards.iter().enumerate() {
            let i = i as u8;
            assert_eq!(card.index(), i);
            assert_eq!(Card::from_index(i), Some(*card));
            assert_eq!(Card::from_name(&card.name()), Some(*card));
            assert_eq!(index_of_name(&name_of_index(i)), Some(i));
        }
        assert_eq!(Card::from_index(53), None);
        assert_eq!(Card::from_index(255), None);
    }

    #[test]
    fn canonical_names() {
        assert_eq!(name_of_index(0), "AC");
        assert_eq!(name_of_index(9), "10C");
        assert_eq!(name_of_index(10), "JC");
        assert_eq!(name_of_index(11), "QC");
        assert_eq!(name_of_index(12), "KC");
        assert_eq!(name_of_index(26), "AH");
        assert_eq!(name_of_index(30), "5H");
        assert_eq!(name_of_index(48), "10S");
        assert_eq!(name_of_index(52), "JK");
        assert_eq!(Card::from_name("1C"), None);
        assert_eq!(Card::from_name("11C"), None);
        assert_eq!(Card::from_name("KX"), None);
        assert_eq!(Card::from_name(""), None);
    }

    #[test]
    fn aces_and_colours() {
        let aces: Vec<u8> = all_cards()
            .into_iter()
            .filter(|c| c.is_ace())
            .map(Card::index)
            .collect();
        assert_eq!(aces, vec![0, 13, 26, 39]);
        assert!(!Card::Joker.is_ace());
        assert!(Suit::Diamonds.is_red() && Suit::Hearts.is_red());
        assert!(!Suit::Clubs.is_red() && !Suit::Spades.is_red());
    }
}
