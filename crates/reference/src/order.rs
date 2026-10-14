//! Which cards beat which (RULES 6-11), what may be played (RULES 13-18),
//! what may be burned (H3) and who wins a trick (RULE 19).
//!
//! Every order is written out by hand as a list, best card first, straight
//! from the rules text. Nothing is computed by arithmetic on ranks.

use crate::card::Rank::{
    Ace, Eight, Five, Four, Jack, King, Nine, Queen, Seven, Six, Ten, Three, Two,
};
use crate::card::Suit::{Clubs, Diamonds, Hearts, Spades};
use crate::card::{Card, Rank, Suit};

const fn c(rank: Rank, suit: Suit) -> Card {
    Card::Normal(rank, suit)
}

const JK: Card = Card::Joker;

/// Clubs trumps (black, rule 6): 5, J, Joker, A of hearts, A, K, Q, then 2 up to 10.
const CLUBS_TRUMPS: [Card; 15] = [
    c(Five, Clubs),
    c(Jack, Clubs),
    JK,
    c(Ace, Hearts),
    c(Ace, Clubs),
    c(King, Clubs),
    c(Queen, Clubs),
    c(Two, Clubs),
    c(Three, Clubs),
    c(Four, Clubs),
    c(Six, Clubs),
    c(Seven, Clubs),
    c(Eight, Clubs),
    c(Nine, Clubs),
    c(Ten, Clubs),
];

/// Spades trumps (black, rule 6): 5, J, Joker, A of hearts, A, K, Q, then 2 up to 10.
const SPADES_TRUMPS: [Card; 15] = [
    c(Five, Spades),
    c(Jack, Spades),
    JK,
    c(Ace, Hearts),
    c(Ace, Spades),
    c(King, Spades),
    c(Queen, Spades),
    c(Two, Spades),
    c(Three, Spades),
    c(Four, Spades),
    c(Six, Spades),
    c(Seven, Spades),
    c(Eight, Spades),
    c(Nine, Spades),
    c(Ten, Spades),
];

/// Diamonds trumps (red, rule 6): 5, J, Joker, A of hearts, A, K, Q, then 10 down to 2.
const DIAMONDS_TRUMPS: [Card; 15] = [
    c(Five, Diamonds),
    c(Jack, Diamonds),
    JK,
    c(Ace, Hearts),
    c(Ace, Diamonds),
    c(King, Diamonds),
    c(Queen, Diamonds),
    c(Ten, Diamonds),
    c(Nine, Diamonds),
    c(Eight, Diamonds),
    c(Seven, Diamonds),
    c(Six, Diamonds),
    c(Four, Diamonds),
    c(Three, Diamonds),
    c(Two, Diamonds),
];

/// Hearts trumps (red, rules 6 and 8): 5, J, Joker, A of hearts (fourth), K, Q,
/// then 10 down to 2. Only 14 trumps.
const HEARTS_TRUMPS: [Card; 14] = [
    c(Five, Hearts),
    c(Jack, Hearts),
    JK,
    c(Ace, Hearts),
    c(King, Hearts),
    c(Queen, Hearts),
    c(Ten, Hearts),
    c(Nine, Hearts),
    c(Eight, Hearts),
    c(Seven, Hearts),
    c(Six, Hearts),
    c(Four, Hearts),
    c(Three, Hearts),
    c(Two, Hearts),
];

/// A red suit that is not trumps (rule 9): K, Q, J, 10 down to 2, then the ace (lowest).
const RED_PLAIN_RANKS: [Rank; 13] = [
    King, Queen, Jack, Ten, Nine, Eight, Seven, Six, Five, Four, Three, Two, Ace,
];

/// A black suit that is not trumps (rule 10): K, Q, J, A, then 2 up to 10 (lowest).
const BLACK_PLAIN_RANKS: [Rank; 13] = [
    King, Queen, Jack, Ace, Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten,
];

/// The trumps for this trump suit, best first.
pub fn trump_order(trump: Suit) -> &'static [Card] {
    match trump {
        Clubs => &CLUBS_TRUMPS,
        Diamonds => &DIAMONDS_TRUMPS,
        Hearts => &HEARTS_TRUMPS,
        Spades => &SPADES_TRUMPS,
    }
}

/// The plain (non-trump) cards of `suit`, best first, when `trump` is trumps.
/// Empty when `suit` is the trump suit. Plain hearts never contain the ace of
/// hearts, which is always a trump (rule 11).
pub fn plain_order(suit: Suit, trump: Suit) -> Vec<Card> {
    if suit == trump {
        return Vec::new();
    }
    let ranks = if suit.is_red() {
        RED_PLAIN_RANKS
    } else {
        BLACK_PLAIN_RANKS
    };
    let mut order = Vec::new();
    for rank in ranks {
        let card = c(rank, suit);
        if card == c(Ace, Hearts) {
            continue;
        }
        order.push(card);
    }
    order
}

/// True if the card is a trump this hand (rule 7: the Joker and the ace of
/// hearts always are).
pub fn is_trump(card: Card, trump: Suit) -> bool {
    trump_order(trump).contains(&card)
}

/// Position in the trump order (0 is the best trump), or `None` for a plain card.
pub fn trump_strength(card: Card, trump: Suit) -> Option<usize> {
    trump_order(trump).iter().position(|t| *t == card)
}

/// True if trump `a` beats trump `b`. Both must be trumps.
pub fn trump_beats(a: Card, b: Card, trump: Suit) -> bool {
    let a = trump_strength(a, trump).expect("trump_beats: first card is not a trump");
    let b = trump_strength(b, trump).expect("trump_beats: second card is not a trump");
    a < b
}

/// The ace of the trump suit (the ace of hearts when hearts are trumps).
pub fn ace_of_trumps(trump: Suit) -> Card {
    c(Ace, trump)
}

/// The 5 of the trump suit, the best card in the hand.
pub fn five_of_trumps(trump: Suit) -> Card {
    c(Five, trump)
}

/// The four top trumps of rule 16: the 5, the J, the Joker and the ace of hearts.
pub fn top_four(trump: Suit) -> [Card; 4] {
    [c(Five, trump), c(Jack, trump), JK, c(Ace, Hearts)]
}

/// The plain suit a card belongs to, or `None` if it is a trump.
pub fn plain_suit(card: Card, trump: Suit) -> Option<Suit> {
    if is_trump(card, trump) {
        None
    } else {
        card.suit()
    }
}

/// The cards of `hand` that may be played, in the hand's order.
/// `led` is the first card of the trick so far, or `None` when leading.
pub fn legal_plays(hand: &[Card], led: Option<Card>, trump: Suit) -> Vec<Card> {
    // Rule 13: any card may be led.
    let Some(led) = led else {
        return hand.to_vec();
    };

    let trumps_held: Vec<Card> = hand
        .iter()
        .copied()
        .filter(|card| is_trump(*card, trump))
        .collect();

    if is_trump(led, trump) {
        // Rules 16-18: a trump was led.
        let mut must_play_a_trump = false;
        for t in &trumps_held {
            let is_top = top_four(trump).contains(t);
            let higher_than_led = trump_beats(*t, led, trump);
            let may_hold_back = is_top && higher_than_led;
            if !may_hold_back {
                must_play_a_trump = true;
            }
        }
        if must_play_a_trump {
            // Any trump will do, the held-back ones included.
            trumps_held
        } else {
            // No trumps, or only trumps that may be held back (rule 18).
            hand.to_vec()
        }
    } else {
        // Rules 14-15: a plain suit was led.
        let led_suit = plain_suit(led, trump).expect("a plain card has a suit");
        let mut has_led_suit = false;
        for card in hand {
            if plain_suit(*card, trump) == Some(led_suit) {
                has_led_suit = true;
            }
        }
        if !has_led_suit {
            return hand.to_vec();
        }
        hand.iter()
            .copied()
            .filter(|card| plain_suit(*card, trump) == Some(led_suit) || is_trump(*card, trump))
            .collect()
    }
}

/// The cards of `hand` that may be burned: every card except the ace of trumps (rules 21, 24,
/// H3). Before trumps are known (the Joker burn) every card may go.
pub fn legal_burns(hand: &[Card], trump: Option<Suit>) -> Vec<Card> {
    hand.iter()
        .copied()
        .filter(|&card| trump.is_none_or(|t| card != ace_of_trumps(t)))
        .collect()
}

/// The seat that wins a trick given as `(seat, card)` in play order (rule 19):
/// the highest trump, or if there is none the highest card of the suit led.
pub fn trick_winner(trick: &[(u8, Card)], trump: Suit) -> u8 {
    assert!(!trick.is_empty(), "trick_winner: empty trick");

    let mut best_trump: Option<(u8, Card)> = None;
    for &(seat, card) in trick {
        if !is_trump(card, trump) {
            continue;
        }
        match best_trump {
            None => best_trump = Some((seat, card)),
            Some((_, best)) => {
                if trump_beats(card, best, trump) {
                    best_trump = Some((seat, card));
                }
            }
        }
    }
    if let Some((seat, _)) = best_trump {
        return seat;
    }

    let led_suit =
        plain_suit(trick[0].1, trump).expect("no trump in the trick, so the lead is plain");
    let order = plain_order(led_suit, trump);
    let mut best: Option<(u8, usize)> = None;
    for &(seat, card) in trick {
        let Some(pos) = order.iter().position(|o| *o == card) else {
            continue; // another suit never wins
        };
        match best {
            None => best = Some((seat, pos)),
            Some((_, best_pos)) => {
                if pos < best_pos {
                    best = Some((seat, pos));
                }
            }
        }
    }
    best.expect("the led card is of the led suit").0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{card, cards, names};

    fn trump_names(trump: Suit) -> String {
        names(trump_order(trump))
    }

    fn plain_names(suit: Suit, trump: Suit) -> String {
        names(&plain_order(suit, trump))
    }

    #[test]
    fn trump_order_black_suits() {
        // Rule 6: black numbers run 2 up to 10, so the 2 beats the 10.
        assert_eq!(
            trump_names(Clubs),
            "5C JC JK AH AC KC QC 2C 3C 4C 6C 7C 8C 9C 10C"
        );
        assert_eq!(
            trump_names(Spades),
            "5S JS JK AH AS KS QS 2S 3S 4S 6S 7S 8S 9S 10S"
        );
        assert!(trump_beats(card("2S"), card("10S"), Spades));
        assert!(trump_beats(card("AH"), card("AS"), Spades)); // rule 8
    }

    #[test]
    fn trump_order_red_suits() {
        // Rule 6: red numbers run 10 down to 2.
        assert_eq!(
            trump_names(Diamonds),
            "5D JD JK AH AD KD QD 10D 9D 8D 7D 6D 4D 3D 2D"
        );
        assert!(trump_beats(card("10D"), card("2D"), Diamonds));
        assert!(trump_beats(card("AH"), card("AD"), Diamonds)); // rule 8
    }

    #[test]
    fn hearts_as_trumps_has_14_trumps() {
        // Rule 8: the ace of hearts is fourth and there is no separate ace of trumps.
        assert_eq!(
            trump_names(Hearts),
            "5H JH JK AH KH QH 10H 9H 8H 7H 6H 4H 3H 2H"
        );
        assert_eq!(trump_order(Hearts).len(), 14);
        assert_eq!(trump_strength(card("AH"), Hearts), Some(3));
        assert_eq!(ace_of_trumps(Hearts), card("AH"));
        for trump in [Clubs, Diamonds, Spades] {
            assert_eq!(trump_order(trump).len(), 15);
        }
    }

    #[test]
    fn joker_and_ace_of_hearts_are_always_trumps() {
        // Rule 7 and rule 11.
        for trump in [Clubs, Diamonds, Hearts, Spades] {
            assert!(is_trump(Card::Joker, trump));
            assert!(is_trump(card("AH"), trump));
            assert_eq!(plain_suit(card("AH"), trump), None);
            assert!(!plain_order(Hearts, trump).contains(&card("AH")));
            assert_eq!(top_four(trump)[2], Card::Joker);
            assert_eq!(&top_four(trump)[..], &trump_order(trump)[..4]);
        }
        // Each card of the pack is in exactly one order.
        for trump in [Clubs, Diamonds, Hearts, Spades] {
            let mut total = trump_order(trump).len();
            for suit in [Clubs, Diamonds, Hearts, Spades] {
                total += plain_order(suit, trump).len();
            }
            assert_eq!(total, 53);
        }
    }

    #[test]
    fn plain_orders() {
        // Rule 9: red plain suit, the ace is the lowest red card.
        assert_eq!(
            plain_names(Diamonds, Clubs),
            "KD QD JD 10D 9D 8D 7D 6D 5D 4D 3D 2D AD"
        );
        // Rule 11: plain hearts have no ace.
        assert_eq!(
            plain_names(Hearts, Spades),
            "KH QH JH 10H 9H 8H 7H 6H 5H 4H 3H 2H"
        );
        // Rule 10: black plain suit, the ace beats the 2, the 10 is lowest.
        assert_eq!(
            plain_names(Clubs, Hearts),
            "KC QC JC AC 2C 3C 4C 5C 6C 7C 8C 9C 10C"
        );
        assert_eq!(
            plain_names(Spades, Diamonds),
            "KS QS JS AS 2S 3S 4S 5S 6S 7S 8S 9S 10S"
        );
        // The trump suit has no plain order.
        assert!(plain_order(Clubs, Clubs).is_empty());
    }

    fn legal(hand: &str, led: &str, trump: Suit) -> String {
        let led = if led.is_empty() {
            None
        } else {
            Some(card(led))
        };
        names(&legal_plays(&cards(hand), led, trump))
    }

    #[test]
    fn leading_any_card() {
        // Rule 13.
        assert_eq!(legal("5C 2D AH KS JK", "", Clubs), "5C 2D AH KS JK");
    }

    #[test]
    fn following_a_plain_suit() {
        // Rule 14: follow suit or trump; other suits only when void.
        assert_eq!(legal("2D 9D KS 3C JK", "KD", Clubs), "2D 9D 3C JK");
        // Rule 15: void in the suit led, so any card at all.
        assert_eq!(legal("KS 3C JK 4H 7H", "KD", Clubs), "KS 3C JK 4H 7H");
        // No trumps but holding the suit: only the suit.
        assert_eq!(legal("2D 9D KS 4H 7H", "KD", Clubs), "2D 9D");
        // Rule 11: the ace of hearts is not a heart, so holding only AH is a void in hearts.
        assert_eq!(legal("AH 2C 3S 4D 5S", "KH", Diamonds), "AH 2C 3S 4D 5S");
        // ...but it is a trump and may always be played on a plain lead.
        assert_eq!(legal("AH 2H 3S 4D 5S", "KH", Diamonds), "AH 2H 4D");
    }

    #[test]
    fn trumping_is_always_allowed_on_a_plain_lead() {
        // Rule 14: holding the suit led you may still trump, even with a top trump.
        assert_eq!(legal("5C JC 2D 3S 4S", "KD", Clubs), "5C JC 2D");
    }

    #[test]
    fn reneging_against_a_lower_trump() {
        // Rule 16: the four top trumps may be held back when a lower trump is led.
        // Only top trumps held: any card (rule 18).
        assert_eq!(legal("5C JK 7D 8H 9S", "KC", Clubs), "5C JK 7D 8H 9S");
        assert_eq!(legal("5C JC JK AH 9S", "2C", Clubs), "5C JC JK AH 9S");
        // A non-top trump must be played, and then any trump will do.
        assert_eq!(legal("5C 3C 7D 8H 9S", "KC", Clubs), "5C 3C");
        assert_eq!(legal("AC 7D 8H 9S 10S", "KC", Clubs), "AC");
        // No trumps at all: any card.
        assert_eq!(legal("7D 8H 9S 10S 2D", "KC", Clubs), "7D 8H 9S 10S 2D");
    }

    #[test]
    fn rule_17_forcing_cases() {
        // The 5 led forces out every trump.
        assert_eq!(legal("JC 7D 8H 9S 10S", "5C", Clubs), "JC");
        assert_eq!(legal("JK AH 7D 8H 9S", "5C", Clubs), "JK AH");
        // The J led forces out the Joker and the ace of hearts, but the 5 may be held back.
        assert_eq!(legal("5C JK 7D 8H 9S", "JC", Clubs), "5C JK");
        assert_eq!(legal("5C AH 7D 8H 9S", "JC", Clubs), "5C AH");
        assert_eq!(legal("5C 7D 8H 9S 10S", "JC", Clubs), "5C 7D 8H 9S 10S");
        // The Joker led forces out the ace of hearts; the 5 and J may be held back.
        assert_eq!(legal("AH 7D 8H 9C 10C", "JK", Spades), "AH");
        assert_eq!(legal("5S JS 7D 8H 9H", "JK", Spades), "5S JS 7D 8H 9H");
        // The ace of hearts led (hearts not trumps) is a trump lead: it forces the ace
        // of trumps and lower trumps, while the 5, J and Joker may be held back.
        assert_eq!(legal("AD 7C 8C 9S 10S", "AH", Diamonds), "AD");
        assert_eq!(legal("5D JD JK 9S 10S", "AH", Diamonds), "5D JD JK 9S 10S");
        // Every case also forces out every lower trump (rule 16).
        assert_eq!(legal("3C 7D 8H 9S 10S", "5C", Clubs), "3C");
        assert_eq!(legal("QC 7D 8H 9S 10S", "JC", Clubs), "QC");
        assert_eq!(legal("KS 7D 8H 9H 10H", "JK", Spades), "KS");
        assert_eq!(legal("2D 7C 8C 9S 10S", "AH", Diamonds), "2D");
        // Hearts trumps: AH is a top trump that may be held back against a lower heart.
        assert_eq!(legal("AH 7D 8C 9S 10S", "KH", Hearts), "AH 7D 8C 9S 10S");
        // Hearts trumps, AH led: the 2H must be played, any trump will do.
        assert_eq!(legal("JK 2H 7D 8C 9S", "AH", Hearts), "JK 2H");
    }

    #[test]
    fn rule_18_only_holdable_trumps_means_any_card() {
        assert_eq!(legal("JC 2D 3D 4S 5S", "QC", Clubs), "JC 2D 3D 4S 5S");
        assert_eq!(legal("5C JC JK AH 6D", "10C", Clubs), "5C JC JK AH 6D");
        // A higher-ranked lead removes the right to hold back.
        assert_eq!(legal("JC 2D 3D 4S 5S", "5C", Clubs), "JC");
    }

    #[test]
    fn legal_burns_refuse_only_the_ace_of_trumps() {
        let hand = cards("AC AD AH AS 5C JK");
        assert_eq!(names(&legal_burns(&hand, None)), "AC AD AH AS 5C JK");
        assert_eq!(names(&legal_burns(&hand, Some(Clubs))), "AD AH AS 5C JK");
        assert_eq!(names(&legal_burns(&hand, Some(Hearts))), "AC AD AS 5C JK");
        // Exhaustive over every card: only the ace of trumps is refused.
        let all = crate::card::all_cards();
        let burnable = legal_burns(&all, Some(Spades));
        assert_eq!(burnable.len(), 52);
        assert!(burnable.contains(&Card::Joker));
    }

    fn winner(trick: &str, trump: Suit) -> u8 {
        let played: Vec<(u8, Card)> = cards(trick)
            .into_iter()
            .enumerate()
            .map(|(i, c)| (i as u8, c))
            .collect();
        trick_winner(&played, trump)
    }

    #[test]
    fn plain_heart_led_ace_of_hearts_wins_only_without_a_higher_trump() {
        // Spades trumps, a plain heart led: the A♥ wins as a trump unless a higher trump
        // (5, J or Joker) is also played.
        assert_eq!(winner("9H AH KH QH 2C 3C 4C 6C", Spades), 1);
        assert_eq!(winner("9H AH KH JK 2C 3C 4C 6C", Spades), 3);
        assert_eq!(winner("9H AH KH JS 2C 3C 4C 6C", Spades), 3);
    }

    #[test]
    fn three_of_diamonds_wins_only_as_the_led_plain_suit_without_trumps() {
        // Clubs trumps, diamonds led, no trump: the 3D beats the 2D and AD (low in a red plain
        // suit) and kings of other suits, which do not follow.
        assert_eq!(winner("3D 2D KS AD KH QS JH QH", Clubs), 0);
        // The same trick with a trump played: the trump wins.
        assert_eq!(winner("3D 2D KS AD KH QS JH 2C", Clubs), 7);
    }

    #[test]
    fn trick_winner_rule_19() {
        // No trump: highest card of the suit led; other suits never win.
        assert_eq!(winner("7D KD KS QD 2D 9H 3D 4D", Clubs), 1);
        // Red plain ace is the lowest card of the suit.
        assert_eq!(winner("AD 2D 3S 4S 6S 7S 8S 9S", Clubs), 1);
        // Black plain ace beats the 2, and the 10 is lowest.
        assert_eq!(winner("10S AS 2S 7D 8D 9D 2H 3H", Clubs), 1);
        assert_eq!(winner("10S 9S 7D 8D 9D 2H 3H 4H", Clubs), 1);
        // Any trump beats the suit led.
        assert_eq!(winner("KD QD 10C JD 9D 8D 7D 6D", Clubs), 2);
        // Highest trump wins: black 2 beats black 10, the Joker beats the AH, the AH beats the ace of trumps.
        assert_eq!(winner("10C 2C 3D 4D 6D 7D 8D 9D", Clubs), 1);
        assert_eq!(winner("KD AH AD JK 2D 3D 4D 6D", Diamonds), 3);
        assert_eq!(winner("KS AS AH 2S 3S 4S 6S 7S", Spades), 2);
        // The 5 of trumps beats everything.
        assert_eq!(winner("JH JK AH 5H KH QH 10H 9H", Hearts), 3);
        // The AH led when diamonds are trumps is a trump lead and wins unless beaten.
        assert_eq!(winner("AH AD KD 2C 3C 4C 6C 7C", Diamonds), 0);
        // The 5 of a plain suit is just a plain card.
        assert_eq!(winner("5D KD 2C 3C 4C 6C 7C 8C", Spades), 1);
    }
}
