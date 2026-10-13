//! The rules bot's card tracker (docs/M1-SPEC.md §4).
//!
//! It is rebuilt from the bot's own [`SeatView`] at every decision, so it holds nothing the view
//! does not: the bot's hand and burns, the cards played this hand, the turned cards and who took
//! them, the seat that robbed, which seats have proved they lack the ace of trumps, whether the
//! ace of trumps is known to be undealt, and every seat's hand size.
//!
//! From those facts it answers two questions: which cards are **known out of play** (they can
//! never be played again this hand), and **could a given seat hold a given card**.

use engine::SEATS;
use engine::cards::{ALL_CARDS, Bits, Card, JOKER, Suit, bit, has};
use engine::rules::{ACE_HEARTS, ace_of_trumps, trump_mask};
use engine::view::SeatView;

/// What one seat knows about where the cards are, built from its own view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tracker {
    /// The seat this tracker belongs to.
    pub seat: u8,
    /// The trump suit; `None` only during the dealer's burn after taking a turned Joker.
    pub trump: Option<Suit>,
    /// The seat's current hand.
    pub hand: Bits,
    /// The cards this seat has burned this hand (nobody else's burns are visible).
    pub own_burns: Bits,
    /// Every card played to a trick this hand.
    pub played: Bits,
    /// The turned card face up on the table and not taken.
    pub table_card: Option<Card>,
    /// Every card turned this hand, in order, and the seat that took it (if any).
    pub turned: Vec<(Card, Option<u8>)>,
    /// The seat that robbed, which therefore holds the ace of trumps until it plays it.
    pub robber: Option<u8>,
    /// Seats that have shown they do not hold the ace of trumps.
    pub proved_no_ace: [bool; SEATS],
    /// Whether the ace of trumps has been announced as undealt (after trick 1, nobody robbed).
    pub ace_undealt: bool,
    /// Cards in each seat's hand.
    pub hand_sizes: [u8; SEATS],
}

impl Tracker {
    /// Build the tracker from the seat's view.
    pub fn from_view(view: &SeatView) -> Tracker {
        let p = &view.public;
        Tracker {
            seat: view.seat,
            trump: p.trump,
            hand: view.hand,
            own_burns: view.own_burns,
            played: p.played,
            table_card: p.table_card,
            turned: p.turned.clone(),
            robber: p.robber,
            proved_no_ace: p.proved_no_ace,
            ace_undealt: p.ace_undealt,
            hand_sizes: p.hand_sizes,
        }
    }

    /// Every trump. While the trump suit is unknown (the dealer's burn after a turned Joker),
    /// only the Joker and the ace of hearts are certain to be trumps.
    pub fn trumps(&self) -> Bits {
        match self.trump {
            Some(t) => trump_mask(t),
            None => bit(JOKER) | bit(ACE_HEARTS),
        }
    }

    /// The seat that took this turned card, if it was turned and taken this hand.
    pub fn taker(&self, c: Card) -> Option<u8> {
        self.turned
            .iter()
            .find(|&&(t, _)| t == c)
            .and_then(|&(_, s)| s)
    }

    /// The turned cards this seat took this hand that are trumps (the robbed card, and the
    /// Joker for a dealer who took a turned Joker).
    pub fn my_taken_trumps(&self) -> Bits {
        let trumps = self.trumps();
        self.turned
            .iter()
            .filter(|&&(c, s)| s == Some(self.seat) && has(trumps, c))
            .fold(0, |a, &(c, _)| a | bit(c))
    }

    /// Could `seat` hold the ace of trumps now? For this seat, whether it is in the hand. For
    /// another seat: never once it has been played or is known undealt; only the robber after a
    /// rob; otherwise any seat that has not proved it lacks it (a seat still to play its first
    /// card of trick 1). The ace can never be burned (house rule H3).
    pub fn could_hold_ace(&self, seat: u8) -> bool {
        let Some(t) = self.trump else {
            return false;
        };
        let ace = ace_of_trumps(t);
        if seat == self.seat {
            return has(self.hand, ace);
        }
        if has(self.played | self.hand, ace) || self.ace_undealt {
            return false;
        }
        match self.robber {
            Some(r) => r == seat,
            None => !self.proved_no_ace[usize::from(seat)],
        }
    }

    /// Could some seat other than this one still hold the ace of trumps?
    fn ace_open_elsewhere(&self) -> bool {
        (0..SEATS as u8).any(|s| s != self.seat && self.could_hold_ace(s))
    }

    /// The cards known to be out of play for the rest of the hand:
    /// - every card played to a trick;
    /// - this seat's own burns;
    /// - the turned card on the table, once nobody can rob it any more (no other seat can hold
    ///   the ace of trumps): before that, a robber still to play in trick 1 may take it;
    /// - the ace of trumps, once it is proved undealt: announced after trick 1, or deduced
    ///   earlier when nobody robbed and every other seat has proved it lacks the ace.
    pub fn known_out(&self) -> Bits {
        let mut out = self.played | self.own_burns;
        if let Some(t) = self.trump {
            let open = self.ace_open_elsewhere();
            let ace = ace_of_trumps(t);
            if !open && self.robber.is_none() && !has(self.hand | self.played, ace) {
                out |= bit(ace);
            }
            if let Some(tc) = self.table_card
                && !open
            {
                out |= bit(tc);
            }
        }
        out
    }

    /// The cards this seat cannot see and does not know to be out of play: each is in another
    /// seat's hand, in another seat's burn, in the undealt pack, or (in trick 1) the turned card
    /// a robber may still take.
    pub fn unseen(&self) -> Bits {
        ALL_CARDS & !self.hand & !self.known_out()
    }

    /// Could `seat` hold card `c` in its hand now, given everything this seat knows?
    ///
    /// For this seat the answer is exact. For another seat it is "no" for cards in this seat's
    /// hand or known out of play, and for any seat with an empty hand; the ace of trumps and a
    /// turned card still on the table can only be with a seat that could hold the ace (the turned
    /// card through a rob still to come); a turned card that was taken can only be with the seat
    /// that took it (unless that seat burned it); any other card could be with anyone.
    pub fn could_hold(&self, seat: u8, c: Card) -> bool {
        if seat == self.seat {
            return has(self.hand, c);
        }
        if has(self.hand | self.known_out(), c) || self.hand_sizes[usize::from(seat)] == 0 {
            return false;
        }
        if self.table_card == Some(c) {
            return self.could_hold_ace(seat);
        }
        if let Some(t) = self.trump
            && c == ace_of_trumps(t)
        {
            return self.could_hold_ace(seat);
        }
        match self.taker(c) {
            Some(r) => r == seat,
            None => true,
        }
    }
}
