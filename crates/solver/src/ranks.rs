//! The engine's card orders as lookup tables, for the solvers' hot path ([`crate::Position::play`]):
//! a trump's rank and a trick's winner read from small per-trump tables instead of being
//! recomputed card by card. Nothing here is a rule of its own: every entry is a value of
//! `engine::rules` (`trump_rank`, `plain_rank`, `effective_suit`), tabulated once per process,
//! and the winner is `engine::rules::trick_winner`'s by construction (proof at
//! [`trick_winner8`]; tested against the engine on every lead and every card).

use std::sync::OnceLock;

use engine::SEATS;
use engine::cards::{Card, Suit};
use engine::rules::{EffSuit, effective_suit, plain_rank, trump_rank};

/// Per trump suit: every card's trump rank (0 for a plain card), its effective suit's index
/// (0 a trump, 1 + suit for a plain card), and per effective suit led, every card's strength in
/// a trick of that lead: `32 + trump rank` for a trump, the plain rank for a card of the suit
/// led, 0 for any other card.
struct Tables {
    trump_rank: [[u8; 64]; 4],
    led: [[u8; 64]; 4],
    strength: [[[u8; 64]; 5]; 4],
}

fn suit_index(e: EffSuit) -> usize {
    match e {
        EffSuit::Trump => 0,
        EffSuit::Plain(s) => 1 + s as usize,
    }
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Tables {
            trump_rank: [[0; 64]; 4],
            led: [[0; 64]; 4],
            strength: [[[0; 64]; 5]; 4],
        };
        for trump in Suit::ALL {
            let ti = trump as usize;
            for c in 0..engine::cards::DECK_SIZE as Card {
                let ci = usize::from(c);
                let r = trump_rank(c, trump);
                let e = suit_index(effective_suit(c, trump));
                t.trump_rank[ti][ci] = r;
                t.led[ti][ci] = e as u8;
                for (l, row) in t.strength[ti].iter_mut().enumerate() {
                    row[ci] = if r > 0 {
                        32 + r
                    } else if e == l {
                        plain_rank(c)
                    } else {
                        0
                    };
                }
            }
        }
        t
    })
}

/// `engine::rules::trump_rank(c, trump)`, from the table.
#[inline]
pub(crate) fn trump_rank_of(c: Card, trump: Suit) -> u8 {
    tables().trump_rank[trump as usize][usize::from(c)]
}

/// The index (in play order) of the card that wins a full trick of eight cards:
/// `engine::rules::trick_winner(trick, trump)`.
///
/// Why it is the same. The engine starts from the led card and replaces its best only by a card
/// strictly stronger: a trump of higher trump rank than the best trump so far (any trump beats a
/// best that is not a trump), or, while no trump is played, a card of the suit led with a higher
/// plain rank. Under the strength here (`32 + trump rank` for a trump, at least 33, above every
/// plain rank, at most 13; the plain rank for a card of the suit led; 0 for the rest), each such
/// replacement is exactly "strictly greater strength": a trump beats any non-trump best, between
/// trumps the trump ranks decide, a card of the suit led beats a best of the suit led by plain
/// rank and never a trump, and a card of another suit (strength 0) never beats the led card
/// (strength at least 1: a trump, or a plain card of its own suit, plain ranks being 1..=13). So
/// both keep the first card of the greatest strength, the same index (tested on every lead and
/// every card for every trump).
#[inline]
pub(crate) fn trick_winner8(trick: &[Card; SEATS], trump: Suit) -> usize {
    let t = tables();
    let ti = trump as usize;
    let row = &t.strength[ti][usize::from(t.led[ti][usize::from(trick[0])])];
    let mut best = 0;
    let mut bs = row[usize::from(trick[0])];
    for (i, &c) in trick.iter().enumerate().skip(1) {
        let s = row[usize::from(c)];
        if s > bs {
            bs = s;
            best = i;
        }
    }
    best
}

/// The index (in play order) of the card that wins a trick of any length (1..=8 cards; a
/// trick of the seats still in play when partnerships are turned):
/// `engine::rules::trick_winner(trick, trump)`, by the same strengths as [`trick_winner8`].
#[inline]
pub(crate) fn trick_winner_of(trick: &[Card], trump: Suit) -> usize {
    let t = tables();
    let ti = trump as usize;
    let row = &t.strength[ti][usize::from(t.led[ti][usize::from(trick[0])])];
    let mut best = 0;
    let mut bs = row[usize::from(trick[0])];
    for (i, &c) in trick.iter().enumerate().skip(1) {
        let s = row[usize::from(c)];
        if s > bs {
            bs = s;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::rng::Rng;
    use engine::rules::trick_winner;

    #[test]
    fn ranks_are_the_engines() {
        for trump in Suit::ALL {
            for c in 0..engine::cards::DECK_SIZE as Card {
                assert_eq!(trump_rank_of(c, trump), trump_rank(c, trump));
                assert!(plain_rank(c) <= 13 && trump_rank(c, trump) <= 15);
            }
        }
    }

    /// Every lead against every card in every later place (the rest from one fixed filler), and
    /// random full tricks: the same winner as the engine.
    #[test]
    fn trick_winner_is_the_engines() {
        let n = engine::cards::DECK_SIZE as Card;
        for trump in Suit::ALL {
            for led in 0..n {
                for c in 0..n {
                    if c == led {
                        continue;
                    }
                    for place in 1..SEATS {
                        // Fill with distinct cards other than `led` and `c`.
                        let mut trick = [0 as Card; SEATS];
                        let mut next = 0;
                        for (i, x) in trick.iter_mut().enumerate() {
                            *x = if i == 0 {
                                led
                            } else if i == place {
                                c
                            } else {
                                while next == led || next == c {
                                    next += 1;
                                }
                                next += 1;
                                next - 1
                            };
                        }
                        assert_eq!(
                            trick_winner8(&trick, trump),
                            trick_winner(&trick, trump),
                            "{trump:?} {trick:?}"
                        );
                    }
                }
            }
        }
        let mut rng = Rng::new(43);
        for _ in 0..50_000 {
            let mut deck: [Card; engine::cards::DECK_SIZE] = core::array::from_fn(|c| c as Card);
            rng.shuffle(&mut deck);
            let trump = Suit::ALL[rng.bounded(4) as usize];
            let n = 1 + rng.bounded(SEATS as u64) as usize;
            assert_eq!(
                trick_winner_of(&deck[..n], trump),
                trick_winner(&deck[..n], trump)
            );
        }
        let mut rng = Rng::new(41);
        for _ in 0..200_000 {
            let mut deck: [Card; engine::cards::DECK_SIZE] = core::array::from_fn(|c| c as Card);
            rng.shuffle(&mut deck);
            let trump = Suit::ALL[rng.bounded(4) as usize];
            let trick: [Card; SEATS] = core::array::from_fn(|i| deck[i]);
            assert_eq!(trick_winner8(&trick, trump), trick_winner(&trick, trump));
        }
    }
}
