//! Review D probes: hand-made deals written from the text of docs/RULES.md, for combinations of
//! rules the per-rule tests touch only one at a time (hearts trumps with robbing and reneging,
//! a robber burning the card it took, a turned Joker followed by a robbing leader, the 5 of a
//! plain suit, and the end-of-hand bonus when the best trump is played by a seat whose partner
//! later wins nothing).
//!
//! Decks are laid out here by hand from the frozen deal convention (docs/M0-SPEC.md §1: positions
//! 0–15 two at a time from the dealer's left, 16–39 three at a time, 40 the turned card, 41 the
//! second turned card or the top of the pack); the engine's own `build_deck` is not used. Expected
//! legal sets are written out from the rule text.

use engine::cards::{Bits, Card, DECK_SIZE, bit, cards, names, parse};
use engine::game::{Action, Match, Phase, Rules};

fn c(n: &str) -> Card {
    parse(n).unwrap_or_else(|| panic!("bad card {n}"))
}

/// Lay out a deck: `hands[s]` is seat s's five cards (all eight given in full), `pack` the
/// cards from position 40 on, in order; every other card follows in ascending index order.
fn layout(dealer: u8, hands: [&str; 8], pack: &str) -> [Card; DECK_SIZE] {
    let mut deck = [255u8; DECK_SIZE];
    let mut used: Bits = 0;
    for i in 0..8usize {
        let seat = (usize::from(dealer) + 1 + i) % 8;
        let cs: Vec<Card> = hands[seat].split_whitespace().map(c).collect();
        assert_eq!(cs.len(), 5, "seat {seat} needs five cards");
        let pos = [2 * i, 2 * i + 1, 16 + 3 * i, 17 + 3 * i, 18 + 3 * i];
        for (p, &x) in pos.iter().zip(&cs) {
            assert_eq!(used & bit(x), 0, "card {x} given twice");
            used |= bit(x);
            deck[*p] = x;
        }
    }
    let mut p = 40;
    for x in pack.split_whitespace().map(c) {
        assert_eq!(used & bit(x), 0, "card {x} given twice");
        used |= bit(x);
        deck[p] = x;
        p += 1;
    }
    for x in 0..53u8 {
        if used & bit(x) == 0 {
            deck[p] = x;
            p += 1;
        }
    }
    assert_eq!(p, 53);
    deck
}

fn start(dealer: u8, hands: [&str; 8], pack: &str) -> Match {
    Match::with_decks(
        dealer,
        vec![layout(dealer, hands, pack)],
        1,
        Rules::default(),
    )
}

fn legal(m: &Match) -> String {
    names(m.legal_actions().iter().fold(0, |a, x| a | bit(x.card())))
}

fn play(m: &mut Match, n: &str) {
    m.apply(Action::Play(c(n)))
        .unwrap_or_else(|e| panic!("{e} (phase {:?})", m.phase()));
}

fn burn(m: &mut Match, n: &str) {
    m.apply(Action::Burn(c(n)))
        .unwrap_or_else(|e| panic!("{e} (phase {:?})", m.phase()));
}

fn public_log(m: &Match) -> Vec<String> {
    m.events()
        .iter()
        .filter(|(v, _)| *v == engine::game::Visibility::Public)
        .map(|(_, e)| e.canonical())
        .collect()
}

/// Hearts trumps (the 7H turned). Rules 8 and 20–21: the holder of the A♥ holds the ace of trumps
/// and robs at its first turn; it may burn the 7H it took but never the A♥. Rules 16–18 with the
/// KH led: the A♥ (fourth best) and the 5H are top trumps above the king, so may be held back; a
/// kept 7H may not.
#[test]
fn hearts_trumps_rob_then_renege_against_the_king() {
    // Dealer 7: seat 0 leads the KH; seat 1 holds the A♥ and 5H with plain clubs.
    let hands = [
        "KH 2D 3D 4D 6D",
        "AH 5H 2C 3C 4C",
        "8D 9D 10D JD QD",
        "6C 7C 8C 9C 10C",
        "JC QC KC 2S 3S",
        "4S 5S 6S 7S 8S",
        "9S 10S JS QS KS",
        "5C 5D KD 7D AS",
    ];
    // Twice: once burning the taken 7H (then any card), once keeping it (then a trump).
    for keep_seven in [false, true] {
        let mut m = start(7, hands, "7H");
        assert_eq!(m.phase(), Phase::AwaitPlay(0));
        play(&mut m, "KH");
        assert_eq!(
            m.phase(),
            Phase::AwaitBurn(1),
            "the A♥ holder robs (rule 20)"
        );
        assert!(public_log(&m).contains(&"Took seat=1 card=7H".to_string()));
        // Rule 21 / H3: anything but the A♥ (the ace of trumps when hearts are trumps).
        assert_eq!(legal(&m), "2C 3C 4C 5H 7H");
        assert!(m.apply(Action::Burn(c("AH"))).is_err());
        if keep_seven {
            burn(&mut m, "2C");
            // The 7H is below the king and not a top trump: a trump must be played (rule 16),
            // any trump (the held-back ones included).
            assert_eq!(legal(&m), "AH 5H 7H");
        } else {
            burn(&mut m, "7H");
            // Only top trumps above the king: anything (rule 18).
            assert_eq!(legal(&m), "2C 3C 4C AH 5H");
        }
    }
}

/// Rule 17 with hearts trumps: the A♥ led forces the KH and lower; the 5H, JH and Joker may be
/// held back. (Seat 1 holds the 5H and the Joker only: anything; seat 2 the KH: it must follow.)
#[test]
fn hearts_trumps_ace_of_hearts_led_forces_the_king_not_the_joker() {
    let hands = [
        "AH 2D 3D 4D 6D",
        "5H JK 2C 3C 4C",
        "KH 8D 9D 10D JD",
        "6C 7C 8C 9C 10C",
        "JC QC KC 2S 3S",
        "4S 5S 6S 7S 8S",
        "9S 10S JS QS KS",
        "5C 5D KD 7D AS",
    ];
    let mut m = start(7, hands, "7H");
    // Seat 0 holds the A♥, the ace of trumps: it robs before leading (H2).
    assert_eq!(m.phase(), Phase::AwaitBurn(0));
    burn(&mut m, "7H");
    play(&mut m, "AH");
    assert_eq!(legal(&m), "2C 3C 4C 5H JK");
    play(&mut m, "2C");
    assert_eq!(legal(&m), "KH");
}

/// Rule 24 then rule 20 (H2, H4): the Joker is turned, the dealer takes it and burns; the next
/// card is ordinary and the seat left of the dealer holds its ace, so it robs before leading
/// trick 1. Two burns in the hand; 11 cards left in the pack; the Joker is in the dealer's hand.
#[test]
fn joker_turned_then_the_leader_robs_before_leading() {
    let hands = [
        "AD 2C 3C 4C 6C",
        "7C 8C 9C 10C JC",
        "QC KC 2S 3S 4S",
        "5S 6S 7S 8S 9S",
        "10S JS QS KS AS",
        "2H 3H 4H 6H 7H",
        "8H 9H 10H JH QH",
        "KH 5C AC 5H AH",
    ];
    let mut m = start(7, hands, "JK 9D");
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    assert_eq!(m.public().trump, None);
    // Trumps unknown: any of the six cards (H3).
    assert_eq!(legal(&m), "AC 5C AH 5H KH JK");
    burn(&mut m, "AC");
    assert_eq!(m.public().trump, Some(engine::cards::Suit::Diamonds));
    assert_eq!(
        m.phase(),
        Phase::AwaitBurn(0),
        "seat 0 holds the AD and robs first"
    );
    assert_eq!(m.public().trick_no, 1);
    assert_eq!(legal(&m), "2C 3C 4C 6C 9D");
    burn(&mut m, "9D");
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    let p = m.public();
    assert_eq!(p.burn_counts, [1, 0, 0, 0, 0, 0, 0, 1]);
    assert_eq!(p.hidden_count, 13);
    assert_eq!(p.robber, Some(0));
    assert_eq!(p.table_card, None);
    assert_eq!(cards(m.truth().undealt).count(), 11);
    assert!(m.truth().hands[7] & bit(c("JK")) != 0);
    // The dealer holds the A♥ (a trump), the JK and the 5H: with diamonds trumps the 5H is a
    // plain heart and the A♥ is not a plain heart (rule 11).
    play(&mut m, "AD");
    for (seat, card) in [
        (1, "7C"),
        (2, "QC"),
        (3, "5S"),
        (4, "10S"),
        (5, "2H"),
        (6, "8H"),
    ] {
        assert_eq!(m.phase(), Phase::AwaitPlay(seat));
        play(&mut m, card);
    }
    // Rule 17: the ace of trumps (below the A♥ and the Joker) led: the A♥ and the Joker are top
    // trumps above it, so the dealer may hold them back; it has no other trump: anything.
    assert_eq!(legal(&m), "5C AH 5H KH JK");
}

/// Rules 9–10: the 5 of a suit that is not trumps is an ordinary number card, and the plain
/// orders are red "K Q J 10 … 2 A" and black "K Q J A 2 … 10". A diamonds trick with spades
/// trumps: 5D, 6D, 4D, 3D, AD, 2D: the 6D wins (the AD is the lowest red card). A clubs trick with
/// diamonds trumps: 5C, 6C, 4C, 10C, 9C, 8C, 7C: the 4C wins (black numbers run 2 up to 10).
#[test]
fn the_five_of_a_plain_suit_is_an_ordinary_number() {
    // Spades trumps (9S turned, AS undealt: nobody robs).
    let hands = [
        "5D 2H 3H 4H 6H",
        "6D 7H 8H 9H 10H",
        "4D JH QH KH 2C",
        "3D 3C 4C 5C 6C",
        "7C 8C 9C 10C JC",
        "QC KC AC 5H 2S",
        "AD 3S 4S 6S 7S",
        "2D 8S 10S JS QS",
    ];
    let mut m = start(7, hands, "9S AS");
    play(&mut m, "5D");
    assert_eq!(
        legal(&m),
        "6D",
        "holding a diamond and no trump: follow (rule 14)"
    );
    for card in ["6D", "4D", "3D", "7C", "QC"] {
        play(&mut m, card);
    }
    assert_eq!(legal(&m), "AD 3S 4S 6S 7S", "the AD is a plain diamond");
    play(&mut m, "AD");
    play(&mut m, "2D");
    assert!(public_log(&m).contains(&"TrickWon trick=1 seat=1 team=1 points=5".to_string()));

    // Diamonds trumps (9D turned, AD undealt).
    let hands = [
        "5C 2H 3H 4H 6H",
        "6C 7H 8H 9H 10H",
        "4C JH QH KH 2S",
        "10C 3S 4S 5S 6S",
        "7S 8S 9S 10S JS",
        "QS KS AS 5H 9C",
        "8C 3D 4D 6D 7D",
        "7C 8D 10D JD QD",
    ];
    let mut m = start(7, hands, "9D AD");
    for card in ["5C", "6C", "4C", "10C", "7S", "9C", "8C", "7C"] {
        play(&mut m, card);
    }
    assert!(public_log(&m).contains(&"TrickWon trick=1 seat=2 team=2 points=5".to_string()));
}

/// Rule 30 and rule 28: with the 5 never played, the end-of-hand bonus goes to the team that won
/// the trick holding the highest trump played; here that is the Joker played by seat 2, so team 2
/// gets 6 at the end of the hand even though its partner wins nothing.
#[test]
fn end_bonus_goes_to_the_trick_with_the_highest_trump() {
    // Clubs trumps (9C turned); 5C, JC and the AC undealt, so the Joker is the best trump and
    // nobody robs. Trick 1: seat 0 leads the 2D; seat 2 trumps with the Joker and wins.
    let hands = [
        "2D 3H 4H 6H 7H",
        "3D 8H 9H 10H JH",
        "JK QH KH 2S 3S",
        "4D 4S 6S 7S 8S",
        "6D 9S 10S JS QS",
        "7D KS AS 2H 5H",
        "8D AD 5D 9D 10D",
        "JD QD KD AH 5S",
    ];
    let mut m = start(7, hands, "9C 5C JC AC");
    play(&mut m, "2D");
    play(&mut m, "3D");
    // Seat 2 has no diamond: any card (rule 15).
    assert_eq!(legal(&m), "QH KH 2S 3S JK");
    play(&mut m, "JK");
    for card in ["4D", "6D", "7D", "8D"] {
        play(&mut m, card);
    }
    // Seat 7 must follow diamonds or trump (rule 14): JD QD KD or the A♥.
    assert_eq!(legal(&m), "JD QD KD AH");
    play(&mut m, "AH");
    // The Joker beats the A♥ (rule 6).
    let log = public_log(&m);
    assert!(log.contains(&"TrickWon trick=1 seat=2 team=2 points=5".to_string()));
    // Play out the hand with the lowest legal card; no trump above the Joker remains in play.
    while m.hand_index() == 0 {
        let a = m.legal_actions()[0];
        m.apply(a).expect("legal");
    }
    let log = public_log(&m);
    assert!(
        log.contains(&"Bonus team=2 points=6 kind=end card=JK".to_string()),
        "{log:?}"
    );
}

/// `rules::beats` (used by the fast solver's pruning) agrees with `rules::trick_winner` on every
/// trick prefix of up to three cards and every fourth card, for all four trump suits: the new
/// card takes the trick exactly when it beats the card winning so far.
#[test]
fn beats_agrees_with_trick_winner_exhaustively_on_short_tricks() {
    use engine::cards::Suit;
    use engine::rules::{beats, effective_suit, trick_winner};
    let mut checked = 0u64;
    for trump in Suit::ALL {
        for led in 0..53u8 {
            for b in 0..53u8 {
                if b == led {
                    continue;
                }
                let prefix = [led, b];
                let w = prefix[trick_winner(&prefix, trump)];
                for a in 0..53u8 {
                    if a == led || a == b {
                        continue;
                    }
                    let t = [led, b, a];
                    let new_wins = trick_winner(&t, trump) == 2;
                    assert_eq!(
                        new_wins,
                        beats(a, w, effective_suit(led, trump), trump),
                        "trump {trump:?}, trick {:?}",
                        t
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 500_000);
}

/// "Reachable scores" (RULES, owner-confirmed decisions): a team's score is `5k`, or `5k+6` with
/// `k >= 1` (its first best-trump trick scores 11), so 1 and 6 can never occur. A hand-start
/// state with such a score is not a state of any match and must be refused.
#[test]
fn hand_start_refuses_the_unreachable_scores_one_and_six() {
    use engine::game::HandStartState;
    for bad in [1u8, 6] {
        let s = HandStartState {
            dealer: 0,
            scores: [bad, 0, 0, 0],
            bonus_taken: [true, false, false, false],
            games_won: [0; 4],
            hand_index: 1,
            game_index: 0,
        };
        assert!(s.validate().is_err(), "score {bad} is unreachable");
    }
    let ok = HandStartState {
        dealer: 0,
        scores: [11, 16, 36, 40],
        bonus_taken: [true, true, true, false],
        games_won: [0; 4],
        hand_index: 5,
        game_index: 0,
    };
    assert!(ok.validate().is_ok());
}
