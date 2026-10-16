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
//! Two cards of the mover's hand are equivalent when they are in the same class, **no other
//! live card** (a card still in any hand, or already on the table in the current trick) of that
//! class ranks between them, and **no other card of the mover's hand lies between them in the
//! tie order** (card index). Cards played in earlier tricks are gone, so they make the remaining
//! cards adjacent. A group is a chain of such pairs (consecutive in rank).
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
//! - *Tie order*: every seat breaks ties between equally good cards by card index (the naive
//!   solver's position-only rule), and a tie for the mover's team can hide different values for
//!   the other teams. So the swap must also keep the tie order: whenever the holder later
//!   chooses between the one of `a`, `b` it still holds and another card `x` of its hand, `x`
//!   must come before both or after both. That holds exactly when no card of the hand lies
//!   between `a` and `b` in card index (the hand only loses cards from here on; a pending
//!   robber's hand is taken with the turned card at its burn). Without this condition merging
//!   gave wrong values: with clubs trumps, a hand 4♣ 5♣ 7♣ with the 6♣ gone merged the 4♣ and the
//!   7♣, but keeping the 4♣ puts it before the 5♣ in the tie order and keeping the 7♣ puts it
//!   after (test `merged_cards_have_identical_naive_values` in `tests/probe.rs`).
//!
//! The tests check this against the naive solver on many random worlds, plus hand-made decks
//! that must (positive) and must not (negative) be merged.

use std::sync::OnceLock;

use engine::cards::{Bits, Card, JOKER, Suit, bit, card};
use engine::rules::{ACE_HEARTS, five_of, jack_of, plain_mask, plain_rank, trump_mask, trump_rank};

use crate::TieOrder;

/// The ranked classes for one trump suit, as the bit tables that [`Classes::groups`] uses
/// (the 5 of trumps and the J of trumps by card; the other classes as [`Field`]s; the top
/// trumps J, Joker, A♥ are handled by `tops_to_rank`). Built from [`lists`].
#[derive(Clone, Debug)]
pub struct Classes {
    five: Card,
    jack: Card,
    /// The ordinary trumps and the three plain suits (classes 2–5), each inside one suit's
    /// 13-bit field of the card word.
    fields: [Field; 4],
}

/// A class whose cards all lie in one suit's 13-bit field of the card word (bits `shift ..
/// shift + 13`): the permutation between that field (card-index order) and *rank space*,
/// where bit `j` is the class's `j`-th best card (bit 0 the best, bit `len − 1` the worst);
/// cards of the field outside the class map to nothing. Stored as byte tables, so a field is
/// permuted by two lookups and an OR (every source bit lands independently, so the low byte's
/// and the high five bits' contributions can be looked up apart).
#[derive(Clone, Debug)]
struct Field {
    shift: u8,
    len: u8,
    /// Field bits 0..8 → rank bits.
    lo: [u16; 256],
    /// Field bits 8..13 → rank bits.
    hi: [u16; 32],
    /// Rank bits 0..8 → field bits.
    back_lo: [u16; 256],
    /// Rank bits 8..15 → field bits.
    back_hi: [u16; 128],
}

impl Field {
    /// The field of `cards` (best first), all within the suit field at `shift`.
    fn new(shift: u8, cards: &[Card]) -> Field {
        let mut f = Field {
            shift,
            len: cards.len() as u8,
            lo: [0; 256],
            hi: [0; 32],
            back_lo: [0; 256],
            back_hi: [0; 128],
        };
        // Where each field bit lands in rank space and back (single bits), then every byte
        // value as the OR of its bits' images (built from the value with its lowest bit off).
        let mut to_rank = [0u16; 13];
        let mut to_field = [0u16; 15];
        for (j, &c) in cards.iter().enumerate() {
            let b = usize::from(c - shift);
            debug_assert!(b < 13);
            to_rank[b] = 1 << j;
            to_field[j] = 1 << b;
        }
        let fill = |t: &mut [u16], image: &dyn Fn(usize) -> u16, base: usize| {
            for v in 1..t.len() {
                let low = v.trailing_zeros() as usize;
                t[v] = t[v & (v - 1)] | image(base + low);
            }
        };
        fill(&mut f.lo, &|b| to_rank[b], 0);
        fill(&mut f.hi, &|b| to_rank[b], 8);
        fill(&mut f.back_lo, &|j| if j < 15 { to_field[j] } else { 0 }, 0);
        fill(&mut f.back_hi, &|j| if j < 15 { to_field[j] } else { 0 }, 8);
        f
    }

    /// The class's cards among `x`, in rank space.
    fn to_rank(&self, x: Bits) -> u32 {
        let f = ((x >> self.shift) & 0x1FFF) as usize;
        u32::from(self.lo[f & 0xFF] | self.hi[f >> 8])
    }

    /// A rank-space set back as cards.
    fn to_cards(&self, r: u32) -> Bits {
        let r = r as usize;
        Bits::from(self.back_lo[r & 0xFF] | self.back_hi[r >> 8]) << self.shift
    }
}

/// The six classes of a trump suit as lists of cards, each best card first, with their
/// lengths: the 5 of trumps; the other top trumps; the ordinary trumps; the three plain suits.
pub(crate) fn lists(trump: Suit) -> ([[Card; 15]; 6], [u8; 6]) {
    let mut lists = [[0 as Card; 15]; 6];
    let mut lens = [0u8; 6];
    let mut push = |i: usize, x: Card| {
        lists[i][usize::from(lens[i])] = x;
        lens[i] += 1;
    };
    push(0, five_of(trump));
    // Top trumps other than the 5: J, Joker, A♥ (rule 6 order).
    for x in [jack_of(trump), JOKER, ACE_HEARTS] {
        push(1, x);
    }
    // Ordinary trumps by trump order (ranks 11 down to 1).
    for r in (1..=11u8).rev() {
        for x in 0..52u8 {
            if trump_mask(trump) & bit(x) != 0 && x != ACE_HEARTS && trump_rank(x, trump) == r {
                push(2, x);
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
                    push(i, x);
                }
            }
        }
        i += 1;
    }
    debug_assert_eq!(lens.iter().map(|&l| usize::from(l)).sum::<usize>(), 53);
    (lists, lens)
}

impl Classes {
    /// The classes for a trump suit, built once per process and shared: a pure function of
    /// the trump suit, so a solve need not rebuild the tables.
    pub fn shared(trump: Suit) -> &'static Classes {
        static ALL: OnceLock<[Classes; 4]> = OnceLock::new();
        &ALL.get_or_init(|| Suit::ALL.map(Classes::new))[trump as usize]
    }

    /// The classes for a trump suit.
    pub fn new(trump: Suit) -> Classes {
        let (lists, lens) = lists(trump);
        let field = |i: usize| {
            let cards = &lists[i][..usize::from(lens[i])];
            Field::new(cards[0] / 13 * 13, cards)
        };
        Classes {
            five: five_of(trump),
            jack: jack_of(trump),
            fields: [field(2), field(3), field(4), field(5)],
        }
    }

    /// The top trumps other than the 5 (J, Joker, A♥) among `x`, in rank space.
    fn tops_to_rank(&self, x: Bits) -> u32 {
        (((x >> self.jack) & 1) | ((x >> JOKER) & 1) << 1 | ((x >> ACE_HEARTS) & 1) << 2) as u32
    }

    fn tops_to_cards(&self, r: u32) -> Bits {
        let r = Bits::from(r);
        (r & 1) << self.jack | ((r >> 1) & 1) << JOKER | ((r >> 2) & 1) << ACE_HEARTS
    }

    /// The groups of equivalent cards in `hand`, given the `live` cards (every card in a hand or
    /// in the current trick; `hand` ⊆ `live`). Each group is a set of one or more cards; every
    /// card of `hand` is in exactly one group. Groups are written to `out` in class order, best
    /// group first within a class; returns how many.
    ///
    /// Bit-parallel: each class is taken to rank space, where a group is a run of hand cards
    /// with no live non-hand card between them (dead cards, played in earlier tricks, are
    /// transparent). With `h` the hand cards, `o` the live cards and `dead` the class's other
    /// positions, `(dead | h) + h` sends a carry up from every hand bit through the dead bits
    /// above it to the next live bit; that bit is set in the sum when it is a hand bit (its own
    /// bit plus the carry: it then carries on upwards), so `sum & h` marks every hand card whose
    /// next live card below it in rank is also in the hand, that is, every hand card that
    /// continues a group. The other hand bits start the groups. A run is then cut wherever
    /// another hand card lies between two rank-consecutive members in card index (the tie-order
    /// condition of the module notes). `hand` holds at most 16 cards (the solver's at most 6).
    pub fn groups(&self, hand: Bits, live: Bits, out: &mut [Bits; 16]) -> usize {
        debug_assert_eq!(hand & !live, 0, "the hand is live");
        debug_assert!(hand.count_ones() <= 16, "at most one group per card");
        let mut n = 0;
        let five = bit(self.five);
        if hand & five != 0 {
            out[n] = five;
            n += 1;
        }
        let h = self.tops_to_rank(hand);
        if h != 0 {
            n = emit(
                h,
                self.tops_to_rank(live),
                3,
                |r| self.tops_to_cards(r),
                hand,
                out,
                n,
            );
        }
        for f in &self.fields {
            let h = f.to_rank(hand);
            if h != 0 {
                n = emit(h, f.to_rank(live), f.len, |r| f.to_cards(r), hand, out, n);
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

/// [`Classes::groups`] as a walk over every card of every class in rank order (the original
/// definition, kept as the oracle of the bit-parallel one).
#[cfg(test)]
fn groups_walk(trump: Suit, hand: Bits, live: Bits, out: &mut [Bits; 16]) -> usize {
    let (lists, lens) = lists(trump);
    {
        let mut n = 0;
        for (list, &len) in lists.iter().zip(&lens) {
            let mut cur: Bits = 0;
            let mut prev: Card = 0;
            for &c in &list[..usize::from(len)] {
                let b = bit(c);
                if hand & b != 0 {
                    // Another hand card between this one and the previous member in card
                    // index ends the group (the tie-order condition).
                    let (lo, hi) = (prev.min(c), prev.max(c));
                    if cur != 0 && (lo + 1..hi).any(|x| hand & bit(x) != 0) {
                        out[n] = cur;
                        n += 1;
                        cur = 0;
                    }
                    cur |= b;
                    prev = c;
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
}

/// The groups of one class in rank space (see [`Classes::groups`]): `h` the hand cards (not
/// empty), `o` the live cards, `len` the class size, `hand` the whole hand as cards; each group
/// goes through `back` into `out[n..]`, best group first. A run of rank-adjacent hand cards is
/// cut wherever another card of `hand` lies between two consecutive members in card index (the
/// tie-order condition of the module notes). Returns the new count.
fn emit(
    h: u32,
    o: u32,
    len: u8,
    back: impl Fn(u32) -> Bits,
    hand: Bits,
    out: &mut [Bits; 16],
    n: usize,
) -> usize {
    debug_assert!(h & !o == 0 && len <= 15);
    let span = (1u32 << len) - 1;
    let dead = span & !o;
    let link = ((dead | h) + h) & h;
    let mut starts = h & !link;
    let mut n = n;
    while starts != 0 {
        let s = starts.isolate_lowest_one();
        starts ^= s;
        // The run: the hand bits from this start up to the next.
        let below_next = if starts == 0 {
            span
        } else {
            starts.isolate_lowest_one() - 1
        };
        let run = h & below_next & !(s - 1);
        if run == s {
            out[n] = back(run);
            n += 1;
            continue;
        }
        // Walk the run best first, cutting it where the tie order separates two members.
        let mut rest = run;
        let mut cur: Bits = 0;
        let mut prev: Bits = 0;
        while rest != 0 {
            let j = rest.isolate_lowest_one();
            rest ^= j;
            let cb = back(j);
            if cur != 0 && hand & strictly_between(prev, cb) != 0 {
                out[n] = cur;
                n += 1;
                cur = 0;
            }
            cur |= cb;
            prev = cb;
        }
        out[n] = cur;
        n += 1;
    }
    n
}

/// The card bits strictly between two distinct single-card bits.
fn strictly_between(x: Bits, y: Bits) -> Bits {
    let (lo, hi) = if x < y { (x, y) } else { (y, x) };
    (hi - 1) & !(lo | (lo - 1))
}

/// The bit of the first card of a non-empty set in the tie order.
pub fn first_in_order(set: Bits, order: TieOrder) -> Bits {
    match order {
        TieOrder::Ascending => set.isolate_lowest_one(),
        TieOrder::Descending => 1 << (63 - set.leading_zeros()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::cards::ALL_CARDS;
    use engine::rng::Rng;

    /// The bit-parallel [`Classes::groups`] against the card walk it replaced, on random
    /// positions of every density for every trump suit: the same groups in the same order.
    #[test]
    fn groups_match_the_card_walk() {
        let mut rng = Rng::new(7);
        let mut checked = 0u64;
        for trump in Suit::ALL {
            let c = Classes::new(trump);
            for i in 0..60_000 {
                // Live cards: a random subset, thinned 0–3 times; the hand: a random subset of
                // them, thinned so it holds a few cards (sometimes many).
                let mut live = rng.next_u64() & ALL_CARDS;
                for _ in 0..(i % 4) {
                    live &= rng.next_u64();
                }
                let mut hand = live & rng.next_u64();
                for _ in 0..(rng.next_u64() % 3) {
                    hand &= rng.next_u64();
                }
                if i % 7 == 0 {
                    // Whole-suit and whole-live hands too.
                    hand = live
                        & (if i % 14 == 0 {
                            ALL_CARDS
                        } else {
                            0x1FFF << (13 * (i % 4))
                        });
                }
                // At most 16 cards, so at most 16 groups (the solver's hands hold at most 6).
                while hand.count_ones() > 16 {
                    hand &= rng.next_u64() | rng.next_u64();
                }
                let mut a = [0; 16];
                let mut b = [0; 16];
                let na = c.groups(hand, live, &mut a);
                let nb = groups_walk(trump, hand, live, &mut b);
                assert_eq!(
                    (na, &a[..na]),
                    (nb, &b[..nb]),
                    "trump {trump:?} hand {hand:#x} live {live:#x}"
                );
                // Every hand card in exactly one group, no other card anywhere.
                let all = a[..na].iter().fold(0, |x, &g| x | g);
                let sum: u32 = a[..na].iter().map(|g| g.count_ones()).sum();
                assert_eq!((all, sum), (hand, hand.count_ones()));
                checked += 1;
            }
        }
        assert_eq!(checked, 240_000);
    }

    /// Every class's cards, from [`Classes::new`]'s lists, and the field tables agree with them.
    #[test]
    fn fields_permute_every_class_card_and_nothing_else() {
        for trump in Suit::ALL {
            let c = Classes::new(trump);
            let (lists, lens) = lists(trump);
            for (k, f) in c.fields.iter().enumerate() {
                let list = &lists[k + 2][..usize::from(lens[k + 2])];
                for (j, &card) in list.iter().enumerate() {
                    assert_eq!(f.to_rank(bit(card)), 1 << j);
                    assert_eq!(f.to_cards(1 << j), bit(card));
                }
                let class: Bits = list.iter().fold(0, |x, &card| x | bit(card));
                assert_eq!(f.to_rank(ALL_CARDS), (1 << f.len) - 1);
                assert_eq!(f.to_cards((1 << f.len) - 1), class);
                assert_eq!(f.to_rank(ALL_CARDS & !class), 0);
            }
            assert_eq!(c.tops_to_rank(ALL_CARDS), 7);
            assert_eq!(
                c.tops_to_cards(7),
                bit(jack_of(trump)) | bit(JOKER) | bit(ACE_HEARTS)
            );
        }
    }
}
