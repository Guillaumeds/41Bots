//! Card orders and the play rules of 41 (RULES.md rules 6–19), plus the burn rule (H3).
//!
//! Everything here is a pure function of bitboards and the trump suit.

use crate::cards::{
    ACE, Bits, Card, JACK, JOKER, KING, QUEEN, Suit, bit, card, has, rank_of, suit_of,
};

/// The ace of hearts, a trump in every hand (rule 7).
pub const ACE_HEARTS: Card = card(Suit::Hearts, ACE);

/// All four aces.
pub const ACES: Bits = bit(card(Suit::Clubs, ACE))
    | bit(card(Suit::Diamonds, ACE))
    | bit(card(Suit::Hearts, ACE))
    | bit(card(Suit::Spades, ACE));

/// The 5 of the trump suit.
pub const fn five_of(trump: Suit) -> Card {
    card(trump, 5)
}

/// The jack of the trump suit.
pub const fn jack_of(trump: Suit) -> Card {
    card(trump, JACK)
}

/// The ace of trumps: the ace of the trump suit (the A♥ when hearts are trumps).
pub const fn ace_of_trumps(trump: Suit) -> Card {
    card(trump, ACE)
}

/// Is the card a trump in a hand with this trump suit? (rules 6–8)
pub const fn is_trump(c: Card, trump: Suit) -> bool {
    if c == JOKER || c == ACE_HEARTS {
        return true;
    }
    match suit_of(c) {
        Some(s) => s as u8 == trump as u8,
        None => false,
    }
}

/// The set of all trumps for this trump suit (15 cards, or 14 when hearts are trumps).
pub const fn trump_mask(trump: Suit) -> Bits {
    trump.mask() | bit(JOKER) | bit(ACE_HEARTS)
}

/// Is the card one of the four top trumps that may be held back (rule 16)?
pub const fn is_top_trump(c: Card, trump: Suit) -> bool {
    c == JOKER || c == ACE_HEARTS || c == five_of(trump) || c == jack_of(trump)
}

/// Strength of a trump: higher wins. 0 for a non-trump. (rule 6)
///
/// 5 = 15, J = 14, Joker = 13, A♥ = 12, ace of trumps = 11, K = 10, Q = 9, then the numbers:
/// red trumps 10 down to 2 (8..1), black trumps 2 up to 10 (8..1).
pub const fn trump_rank(c: Card, trump: Suit) -> u8 {
    if c == JOKER {
        return 13;
    }
    if c == ACE_HEARTS {
        return 12;
    }
    let Some(s) = suit_of(c) else { return 0 };
    if s as u8 != trump as u8 {
        return 0;
    }
    match rank_of(c) {
        5 => 15,
        JACK => 14,
        ACE => 11,
        KING => 10,
        QUEEN => 9,
        r => {
            // r in {2, 3, 4, 6, 7, 8, 9, 10}
            if trump.is_red() {
                if r <= 4 { r - 1 } else { r - 2 }
            } else if r <= 4 {
                10 - r
            } else {
                11 - r
            }
        }
    }
}

/// Strength of a plain (non-trump) card within its own suit: higher wins. (rules 9–10)
///
/// Red plain suit: K Q J 10 … 2 A (ace lowest). Black plain suit: K Q J A 2 … 10 (10 lowest).
pub const fn plain_rank(c: Card) -> u8 {
    let Some(s) = suit_of(c) else { return 0 };
    let r = rank_of(c);
    match r {
        KING => 13,
        QUEEN => 12,
        JACK => 11,
        ACE => {
            if s.is_red() {
                1
            } else {
                10
            }
        }
        n => {
            if s.is_red() {
                n
            } else {
                11 - n
            }
        }
    }
}

/// The suit a card counts as when following (rule 11: the A♥ and Joker are never plain).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EffSuit {
    /// A trump.
    Trump,
    /// A plain card of this suit.
    Plain(Suit),
}

/// Effective suit of a card for this trump suit.
pub const fn effective_suit(c: Card, trump: Suit) -> EffSuit {
    if is_trump(c, trump) {
        EffSuit::Trump
    } else {
        match suit_of(c) {
            Some(s) => EffSuit::Plain(s),
            None => EffSuit::Trump,
        }
    }
}

/// The plain cards of suit `s` (its 13 cards minus any that are trumps).
pub const fn plain_mask(s: Suit, trump: Suit) -> Bits {
    s.mask() & !trump_mask(trump)
}

/// Legal cards to play from `hand` when `led` is the card that led the trick (`None` = leading).
/// (rules 13–18 and the interpretation notes)
pub fn legal_plays(hand: Bits, led: Option<Card>, trump: Suit) -> Bits {
    let Some(led) = led else { return hand };
    match effective_suit(led, trump) {
        EffSuit::Plain(s) => {
            let following = hand & plain_mask(s, trump);
            if following == 0 {
                hand
            } else {
                following | (hand & trump_mask(trump))
            }
        }
        EffSuit::Trump => {
            let led_rank = trump_rank(led, trump);
            let my_trumps = hand & trump_mask(trump);
            // Trumps that may NOT be held back: not a top trump, or a top trump not above the lead.
            let mut forced = 0;
            let mut t = my_trumps;
            while t != 0 {
                let c = t.trailing_zeros() as Card;
                t &= t - 1;
                if !(is_top_trump(c, trump) && trump_rank(c, trump) > led_rank) {
                    forced |= bit(c);
                }
            }
            if forced == 0 { hand } else { my_trumps }
        }
    }
}

/// Legal cards to burn from a hand: any card except the ace of trumps (rules 21 and 24, H3).
/// Plain aces, the A♥ when hearts are not trumps, the Joker and the card just taken may all be
/// burned. While trumps are unknown (the dealer's burn after a turned Joker) nothing is
/// forbidden.
pub const fn legal_burns(hand: Bits, trump: Option<Suit>) -> Bits {
    match trump {
        Some(t) => hand & !bit(ace_of_trumps(t)),
        None => hand,
    }
}

/// Which of the played cards wins the trick (rule 19). `plays` are in play order; the first led.
/// Returns the index into `plays`.
pub fn trick_winner(plays: &[Card], trump: Suit) -> usize {
    debug_assert!(!plays.is_empty());
    let led = plays[0];
    let mut best = 0usize;
    let mut best_trump = trump_rank(led, trump);
    let led_suit = effective_suit(led, trump);
    let mut best_plain = if best_trump == 0 { plain_rank(led) } else { 0 };
    for (i, &c) in plays.iter().enumerate().skip(1) {
        let tr = trump_rank(c, trump);
        if tr > 0 {
            if tr > best_trump {
                best_trump = tr;
                best = i;
            }
        } else if best_trump == 0 && effective_suit(c, trump) == led_suit {
            let pr = plain_rank(c);
            if pr > best_plain {
                best_plain = pr;
                best = i;
            }
        }
    }
    best
}

/// Does `a` beat `b` when `b` is currently winning a trick whose lead has effective suit `led`?
pub fn beats(a: Card, b: Card, led: EffSuit, trump: Suit) -> bool {
    let ta = trump_rank(a, trump);
    let tb = trump_rank(b, trump);
    if ta > 0 || tb > 0 {
        return ta > tb;
    }
    effective_suit(a, trump) == led && plain_rank(a) > plain_rank(b)
}

/// Is the card in the set (re-exported helper for tests).
pub const fn contains(set: Bits, c: Card) -> bool {
    has(set, c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::{names, parse};

    fn set(s: &str) -> Bits {
        s.split_whitespace()
            .map(|n| bit(parse(n).unwrap()))
            .fold(0, |a, b| a | b)
    }

    #[test]
    fn trump_order_red_and_black() {
        let t = Suit::Diamonds;
        let order = [
            "5D", "JD", "JK", "AH", "AD", "KD", "QD", "10D", "9D", "8D", "7D", "6D", "4D", "3D",
            "2D",
        ];
        let ranks: Vec<u8> = order
            .iter()
            .map(|n| trump_rank(parse(n).unwrap(), t))
            .collect();
        assert_eq!(ranks, (1..=15).rev().collect::<Vec<u8>>());
        let t = Suit::Clubs;
        let order = [
            "5C", "JC", "JK", "AH", "AC", "KC", "QC", "2C", "3C", "4C", "6C", "7C", "8C", "9C",
            "10C",
        ];
        let ranks: Vec<u8> = order
            .iter()
            .map(|n| trump_rank(parse(n).unwrap(), t))
            .collect();
        assert_eq!(ranks, (1..=15).rev().collect::<Vec<u8>>());
        assert_eq!(trump_rank(parse("AS").unwrap(), t), 0);
    }

    #[test]
    fn hearts_trumps_has_fourteen() {
        assert_eq!(crate::cards::count(trump_mask(Suit::Hearts)), 14);
        assert_eq!(crate::cards::count(trump_mask(Suit::Spades)), 15);
        assert_eq!(trump_rank(ACE_HEARTS, Suit::Hearts), 12);
        assert_eq!(ace_of_trumps(Suit::Hearts), ACE_HEARTS);
    }

    #[test]
    fn plain_orders() {
        let red = [
            "KD", "QD", "JD", "10D", "9D", "8D", "7D", "6D", "5D", "4D", "3D", "2D", "AD",
        ];
        let r: Vec<u8> = red.iter().map(|n| plain_rank(parse(n).unwrap())).collect();
        assert_eq!(r, (1..=13).rev().collect::<Vec<u8>>());
        let black = [
            "KS", "QS", "JS", "AS", "2S", "3S", "4S", "5S", "6S", "7S", "8S", "9S", "10S",
        ];
        let r: Vec<u8> = black
            .iter()
            .map(|n| plain_rank(parse(n).unwrap()))
            .collect();
        assert_eq!(r, (1..=13).rev().collect::<Vec<u8>>());
    }

    #[test]
    fn plain_lead_must_follow_or_trump() {
        let t = Suit::Spades;
        let hand = set("KD 3D 2C 5S JK");
        let legal = legal_plays(hand, parse("9D"), t);
        assert_eq!(names(legal), "3D KD 5S JK");
        // Void in the led suit: anything.
        let hand = set("2C 7H 5S JK QS");
        assert_eq!(legal_plays(hand, parse("9D"), t), hand);
        // Plain hearts led: the A♥ is not a heart (rule 11); holding only A♥ among hearts means void.
        let hand = set("AH 2C 3C");
        assert_eq!(legal_plays(hand, parse("9H"), t), hand);
        let hand = set("AH 4H 2C");
        assert_eq!(names(legal_plays(hand, parse("9H"), t)), "AH 4H");
    }

    #[test]
    fn reneging() {
        let t = Suit::Clubs;
        // Low trump led; holding only top trumps (above the lead) plus plain cards: anything.
        let hand = set("5C JK 9D 8D 7D");
        assert_eq!(legal_plays(hand, parse("2C"), t), hand);
        // Holding a lower trump too: must play a trump (any trump).
        let hand = set("5C JK 10C 8D 7D");
        assert_eq!(names(legal_plays(hand, parse("2C"), t)), "5C 10C JK");
        // The 5 led forces everything (rule 17).
        let hand = set("JC JK AH 8D 7D");
        assert_eq!(names(legal_plays(hand, parse("5C"), t)), "JC AH JK");
        // The J led forces the Joker and the A♥ but not the 5.
        let hand = set("5C JK 8D 7D 6D");
        assert_eq!(names(legal_plays(hand, parse("JC"), t)), "5C JK");
        // The Joker led forces the A♥.
        let hand = set("AH 8D 7D 6D 4D");
        assert_eq!(names(legal_plays(hand, parse("JK"), t)), "AH");
        let hand = set("5C 8D 7D 6D 4D");
        assert_eq!(legal_plays(hand, parse("JK"), t), hand);
        // A♥ led with hearts not trumps is a trump lead: forces the ace of trumps and lower.
        let hand = set("AC 8D 7D 6D 4D");
        assert_eq!(names(legal_plays(hand, parse("AH"), t)), "AC");
        let hand = set("5C 8D 7D 6D 4D");
        assert_eq!(legal_plays(hand, parse("AH"), t), hand);
    }

    #[test]
    fn any_card_may_be_burned() {
        let hand = set("AC AD AH AS JK 2C");
        assert_eq!(legal_burns(hand, None), hand);
        assert_eq!(
            names(legal_burns(hand, Some(Suit::Spades))),
            "AC 2C AD AH JK"
        );
        assert_eq!(
            names(legal_burns(hand, Some(Suit::Hearts))),
            "AC 2C AD AS JK"
        );
    }

    fn win(plays: &str, t: Suit) -> usize {
        let v: Vec<Card> = plays.split(' ').map(|n| parse(n).unwrap()).collect();
        trick_winner(&v, t)
    }

    #[test]
    fn plain_heart_led_ace_of_hearts_wins_only_without_a_higher_trump() {
        let t = Suit::Spades;
        assert_eq!(win("9H AH KH QH 2C 3C 4C 6C", t), 1);
        assert_eq!(win("9H AH KH JK 2C 3C 4C 6C", t), 3);
        assert_eq!(win("9H AH KH JS 2C 3C 4C 6C", t), 3);
        assert_eq!(win("9H AH KH 5S 2C 3C 4C 6C", t), 3);
    }

    #[test]
    fn three_of_diamonds_wins_only_as_the_led_plain_suit_without_trumps() {
        let t = Suit::Clubs;
        assert_eq!(win("3D 2D KS AD KH QS JH QH", t), 0);
        assert_eq!(win("3D 2D KS AD KH QS JH 2C", t), 7);
    }

    #[test]
    fn winners() {
        let t = Suit::Hearts;
        let plays = [
            parse("9D").unwrap(),
            parse("KD").unwrap(),
            parse("2H").unwrap(),
            parse("AD").unwrap(),
        ];
        assert_eq!(trick_winner(&plays, t), 2);
        let plays = [
            parse("9D").unwrap(),
            parse("KD").unwrap(),
            parse("AD").unwrap(),
            parse("KS").unwrap(),
        ];
        assert_eq!(trick_winner(&plays, t), 1);
        let plays = [
            parse("10S").unwrap(),
            parse("AS").unwrap(),
            parse("2S").unwrap(),
        ];
        assert_eq!(trick_winner(&plays, t), 1);
        let plays = [
            parse("JK").unwrap(),
            parse("AH").unwrap(),
            parse("5H").unwrap(),
        ];
        assert_eq!(trick_winner(&plays, t), 2);
        let plays = [
            parse("3C").unwrap(),
            parse("AH").unwrap(),
            parse("KC").unwrap(),
        ];
        assert_eq!(trick_winner(&plays, Suit::Spades), 1);
    }
}
