//! Power-equivalent cards (PLAN §6.7 and amendment A2).
//!
//! Two cards in the **same hand** are *equivalent* at a position when playing one or the other
//! leads to exactly the same values for all four teams, now and in every later position. The
//! solver then searches only one of them (the first in the tie order, so the reported card is
//! the one the naive solver would report).
//!
//! # The rule used here
//! Every card belongs to exactly one *class*; within a class the cards are ranked:
//! - **the 5 of trumps** is a class of its own (its trick scores the bonus at once, rule 29, so
//!   it is never equivalent to the J or anything else);
//! - **the other top trumps** (J of trumps, Joker, A♥), ranked by trump order (rule 6). They may
//!   be held back against a lower trump lead (rule 16) and each forces a different set when led
//!   (rule 17); the class keeps these differences because it only merges cards with nothing
//!   between them (see below);
//! - **the ordinary trumps** (the ace of trumps when hearts are not trumps, K, Q and the numbers,
//!   red or black order), ranked by trump order. They can never be held back, so a top trump and
//!   an ordinary trump are never equivalent: holding the A♥ (may be held back) is not the same as
//!   holding the ace of trumps (must follow). When hearts are trumps the A♥ *is* the ace of
//!   trumps and is a top trump; there is no separate ace of trumps and the K♥ is the highest
//!   ordinary trump;
//! - **each plain suit**, ranked by the plain order of rules 9–10 (red: K Q J 10 … 2 A; black:
//!   K Q J A 2 … 10); the A♥ never belongs to plain hearts (rule 11).
//!
//! Two cards of the mover's hand are equivalent when they are in the same class and **no other
//! live card** (a card still in any hand, or already on the table in the current trick) of that
//! class ranks between them. Cards played in earlier tricks are gone, so they make the remaining
//! cards adjacent.
//!
//! # Why this is exact
//! Swap the two cards `a` and `b` (same hand, same class, adjacent among live cards) everywhere
//! in the rest of the game. Every rule that the rest of the hand consults gives the same answer:
//! - *Legality* (rules 13–18) depends on a card only through its effective suit, whether it is a
//!   top trump, and (for a top trump held back) whether it ranks above the led trump. `a` and `b`
//!   share the first two by class; the led card is another live card or one of `a`, `b`
//!   themselves (same hand, so never led against each other), and no live card ranks between
//!   them, so the comparison agrees. When `a` or `b` is led, the set it forces from another hand
//!   is decided by comparing ranks with that hand's cards, which again never fall between them.
//! - *Trick winner* (rule 19) compares `a` or `b` with live cards only (cards in this trick or
//!   later tricks), with the same result for both.
//! - *Scoring*: tricks count 5 whoever wins them; the immediate bonus belongs to the 5 alone
//!   (excluded). The end-of-hand bonus goes to the team that played the highest trump of the
//!   hand. If the hand reaches its end, both `a` and `b` have been played by the same team, so
//!   the bonus team is the same; if the game ends earlier, no end-of-hand bonus is paid at all.
//!   (A trump played in an earlier trick may rank between `a` and `b`; that changes which of
//!   them becomes "best so far" first, but not which team holds the best trump when the bonus is
//!   paid.)
//!
//! The tests check this against the naive solver on many random worlds, plus hand-made decks
//! that must (positive) and must not (negative) be merged.

use engine::cards::{Bits, Card, JOKER, Suit, bit, card};
use engine::rules::{ACE_HEARTS, five_of, jack_of, plain_mask, plain_rank, trump_mask, trump_rank};

use crate::TieOrder;

/// The ranked classes for one trump suit, each listed best card first (fixed-size arrays, so
/// building them costs no allocation).
#[derive(Clone, Debug)]
pub struct Classes {
    lists: [[Card; 15]; 6],
    lens: [u8; 6],
}

impl Classes {
    /// The classes for a trump suit.
    pub fn new(trump: Suit) -> Classes {
        let mut c = Classes {
            lists: [[0; 15]; 6],
            lens: [0; 6],
        };
        let push = |c: &mut Classes, i: usize, x: Card| {
            c.lists[i][usize::from(c.lens[i])] = x;
            c.lens[i] += 1;
        };
        push(&mut c, 0, five_of(trump));
        // Top trumps other than the 5: J, Joker, A♥ (rule 6 order).
        for x in [jack_of(trump), JOKER, ACE_HEARTS] {
            push(&mut c, 1, x);
        }
        // Ordinary trumps by trump order (ranks 11 down to 1).
        for r in (1..=11u8).rev() {
            for x in 0..52u8 {
                if trump_mask(trump) & bit(x) != 0 && x != ACE_HEARTS && trump_rank(x, trump) == r {
                    push(&mut c, 2, x);
                }
            }
        }
        // Plain suits by plain order (ranks 13 down to 1).
        let mut i = 3;
        for s in Suit::ALL {
            if s == trump {
                continue;
            }
            for pr in (1..=13u8).rev() {
                for r in 1..=13u8 {
                    let x = card(s, r);
                    if plain_mask(s, trump) & bit(x) != 0 && plain_rank(x) == pr {
                        push(&mut c, i, x);
                    }
                }
            }
            i += 1;
        }
        debug_assert_eq!(c.lens.iter().map(|&l| usize::from(l)).sum::<usize>(), 53);
        c
    }

    /// The groups of equivalent cards in `hand`, given the `live` cards (every card in a hand or
    /// in the current trick; `hand` ⊆ `live`). Each group is a set of one or more cards; every
    /// card of `hand` is in exactly one group. Groups are written to `out`; returns how many.
    pub fn groups(&self, hand: Bits, live: Bits, out: &mut [Bits; 16]) -> usize {
        let mut n = 0;
        for (list, &len) in self.lists.iter().zip(&self.lens) {
            let mut cur: Bits = 0;
            for &c in &list[..usize::from(len)] {
                let b = bit(c);
                if hand & b != 0 {
                    cur |= b;
                } else if live & b != 0 && cur != 0 {
                    out[n] = cur;
                    n += 1;
                    cur = 0;
                }
            }
            if cur != 0 {
                out[n] = cur;
                n += 1;
            }
        }
        n
    }

    /// One representative per group of equivalent cards among the `legal` cards of `hand`: the
    /// first card of the group in the tie order. (A group is either wholly legal or wholly not.)
    pub fn representatives(&self, hand: Bits, legal: Bits, live: Bits, order: TieOrder) -> Bits {
        let mut g = [0; 16];
        let n = self.groups(hand, live, &mut g);
        let mut reps = 0;
        for &grp in &g[..n] {
            let l = grp & legal;
            debug_assert!(l == 0 || l == grp, "equivalent cards share legality");
            if l != 0 {
                reps |= first_in_order(l, order);
            }
        }
        reps
    }
}

/// The bit of the first card of a non-empty set in the tie order.
pub fn first_in_order(set: Bits, order: TieOrder) -> Bits {
    match order {
        TieOrder::Ascending => set.isolate_lowest_one(),
        TieOrder::Descending => 1 << (63 - set.leading_zeros()),
    }
}
