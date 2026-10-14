//! Unit tests of the hand, game and match state machine (docs/M0-SPEC.md §2, §7).
//!
//! The deals are hand-made with `build_deck`; play is either written out card by card or made by
//! the "lowest legal card" policy, whose effect was traced by hand for each deal.

use super::*;
use crate::cards::parse;
use crate::rules::trump_mask;

// ----- helpers -----

fn c(n: &str) -> Card {
    parse(n).unwrap_or_else(|| panic!("bad card name {n}"))
}

fn set(s: &str) -> Bits {
    s.split_whitespace().fold(0, |a, n| a | bit(c(n)))
}

fn deal(dealer: u8, hands: [&str; 8], pack: &str) -> [Card; DECK_SIZE] {
    let hb: [Bits; SEATS] = core::array::from_fn(|s| set(hands[s]));
    let pack: Vec<Card> = pack.split_whitespace().map(c).collect();
    build_deck(dealer, &hb, &pack).unwrap()
}

fn game_with(dealer: u8, hands: [&str; 8], pack: &str, rules: Rules) -> Match {
    Match::with_decks(dealer, vec![deal(dealer, hands, pack)], 1, rules)
}

fn game(dealer: u8, hands: [&str; 8], pack: &str) -> Match {
    game_with(dealer, hands, pack, Rules::default())
}

fn play(m: &mut Match, card: &str) {
    let a = Action::Play(c(card));
    if let Err(e) = m.apply(a) {
        panic!("{e} (phase {:?})", m.phase());
    }
}

fn burn(m: &mut Match, card: &str) {
    let a = Action::Burn(c(card));
    if let Err(e) = m.apply(a) {
        panic!("{e} (phase {:?})", m.phase());
    }
}

/// Play these cards in turn (seat order from whoever is to act).
fn plays(m: &mut Match, cards: &str) {
    for n in cards.split_whitespace() {
        play(m, n);
    }
}

/// Apply the lowest legal action.
fn lowest(m: &mut Match) {
    let a = m.legal_actions()[0];
    m.apply(a).unwrap();
}

/// Apply the lowest legal action until the current hand is over (or the match is).
fn finish_hand(m: &mut Match) {
    let h = m.hand_index();
    while m.phase() != Phase::MatchOver && m.hand_index() == h {
        lowest(m);
    }
}

/// Canonical strings of every event (public and private) from index `from` on.
fn log_from(m: &Match, from: usize) -> Vec<String> {
    m.events()[from..]
        .iter()
        .map(|(_, e)| e.canonical())
        .collect()
}

fn log(m: &Match) -> Vec<String> {
    log_from(m, 0)
}

/// Canonical strings of the public events only.
fn public_log(m: &Match) -> Vec<String> {
    m.events()
        .iter()
        .filter(|(v, _)| *v == Visibility::Public)
        .map(|(_, e)| e.canonical())
        .collect()
}

fn count_matching(m: &Match, prefix: &str) -> usize {
    log(m).iter().filter(|l| l.starts_with(prefix)).count()
}

/// The seats of the `ProvedNoAce` events, in order.
fn proved_seats(m: &Match) -> Vec<u8> {
    m.events()
        .iter()
        .filter_map(|(_, e)| match *e {
            Event::ProvedNoAce { seat } => Some(seat),
            _ => None,
        })
        .collect()
}

/// Index of the first event whose canonical string equals `line`.
fn find(m: &Match, line: &str) -> usize {
    log(m)
        .iter()
        .position(|l| l == line)
        .unwrap_or_else(|| panic!("no event {line}"))
}

fn hidden_cards(m: &Match) -> Bits {
    m.truth()
        .burns
        .iter()
        .fold(m.truth().undealt, |a, &b| a | b)
}

// ----- the deal (step 1) and the constructors -----

/// The identity deck with dealer 0, played out by the lowest-legal-card policy and traced by
/// hand: the deal convention, the dealer robbing as the eighth card of trick 1 (H2), the
/// ProvedNoAce timing, and an end-of-hand bonus won by the A♥.
#[test]
fn identity_deck_hand_traced_by_hand() {
    let mut m = Match::with_decks(0, vec![identity_deck()], 9, Rules::default());
    let deal_lines = [
        "MatchStart first_dealer=0",
        "HandStart hand=0 game=0 dealer=0",
        "Dealt seat=1 cards=AC,2C,4D,5D,6D",
        "Dealt seat=2 cards=3C,4C,7D,8D,9D",
        "Dealt seat=3 cards=5C,6C,10D,JD,QD",
        "Dealt seat=4 cards=7C,8C,KD,AH,2H",
        "Dealt seat=5 cards=9C,10C,3H,4H,5H",
        "Dealt seat=6 cards=JC,QC,6H,7H,8H",
        "Dealt seat=7 cards=KC,AD,9H,10H,JH",
        "Dealt seat=0 cards=2D,3D,QH,KH,AS",
        "Turned card=2S",
    ];
    assert_eq!(log(&m), deal_lines);
    // Positions 41..=52 are the undealt pack.
    assert_eq!(
        m.truth().undealt,
        set("3S 4S 5S 6S 7S 8S 9S 10S JS QS KS JK")
    );
    assert_eq!(m.phase(), Phase::AwaitPlay(1));
    let p = m.public();
    assert_eq!(p.trump, Some(Suit::Spades));
    assert_eq!(p.table_card, Some(c("2S")));
    assert_eq!(p.trick_no, 1);
    assert_eq!(p.leader, 1);
    assert_eq!(p.hidden_count, 12);

    // Trick 1: the dealer (seat 0) holds the ace of spades and robs as the eighth card.
    for _ in 0..7 {
        lowest(&mut m);
    }
    assert_eq!(m.phase(), Phase::AwaitBurn(0));
    assert_eq!(m.public().hand_sizes[0], 6);
    assert_eq!(m.legal_mask(), set("2D 3D QH KH 2S"));
    lowest(&mut m); // burns the 2D
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    lowest(&mut m); // plays the 3D
    let trick1 = [
        "ProvedNoAce seat=1",
        "Played seat=1 card=AC",
        "ProvedNoAce seat=2",
        "Played seat=2 card=3C",
        "ProvedNoAce seat=3",
        "Played seat=3 card=5C",
        "ProvedNoAce seat=4",
        "Played seat=4 card=7C",
        "ProvedNoAce seat=5",
        "Played seat=5 card=9C",
        "ProvedNoAce seat=6",
        "Played seat=6 card=JC",
        "ProvedNoAce seat=7",
        "Played seat=7 card=KC",
        "Took seat=0 card=2S",
        "Burned seat=0 card=2D",
        "BurnDone seat=0",
        "Played seat=0 card=3D",
        "TrickWon trick=1 seat=7 team=3 points=5",
    ];
    assert_eq!(log_from(&m, deal_lines.len()), trick1);
    assert_eq!(m.public().robber, Some(0));
    assert_eq!(m.public().hidden_count, 13);

    finish_hand(&mut m);
    let rest = [
        "Played seat=7 card=AD",
        "Played seat=0 card=QH",
        "Played seat=1 card=4D",
        "Played seat=2 card=7D",
        "Played seat=3 card=10D",
        "Played seat=4 card=KD",
        "Played seat=5 card=10C",
        "Played seat=6 card=QC",
        "TrickWon trick=2 seat=4 team=0 points=5",
        "Played seat=4 card=8C",
        "Played seat=5 card=3H",
        "Played seat=6 card=6H",
        "Played seat=7 card=9H",
        "Played seat=0 card=KH",
        "Played seat=1 card=2C",
        "Played seat=2 card=4C",
        "Played seat=3 card=6C",
        "TrickWon trick=3 seat=1 team=1 points=5",
        "Played seat=1 card=5D",
        "Played seat=2 card=8D",
        "Played seat=3 card=JD",
        "Played seat=4 card=AH",
        "Played seat=5 card=4H",
        "Played seat=6 card=7H",
        "Played seat=7 card=10H",
        "Played seat=0 card=AS",
        "TrickWon trick=4 seat=4 team=0 points=5",
        "Played seat=4 card=2H",
        "Played seat=5 card=5H",
        "Played seat=6 card=8H",
        "Played seat=7 card=JH",
        "Played seat=0 card=2S",
        "Played seat=1 card=6D",
        "Played seat=2 card=9D",
        "Played seat=3 card=QD",
        "TrickWon trick=5 seat=0 team=0 points=5",
        "Bonus team=0 points=6 kind=end card=AH",
        "HandEnd hand=0 aborted=false",
        "HandStart hand=1 game=0 dealer=1",
    ];
    let from = deal_lines.len() + trick1.len();
    assert_eq!(log_from(&m, from)[..rest.len()], rest);
    assert_eq!(m.scores(), [21, 5, 0, 5]);
    assert_eq!(m.bonus_taken(), [true, false, false, false]);
    assert_eq!(m.dealer(), 1);
    assert_eq!(m.hand_index(), 1);
    assert_eq!(m.game_index(), 0);
}

/// Rule 31 on a partial hand: the identity deal plays no trump in tricks 1 to 3, so when team B
/// (on 40) wins trick 3 the game ends with no best trump and no bonus. (A completed hand always
/// has a trump played at 8 players; the property tests assert that.)
#[test]
fn no_trump_played_means_no_bonus() {
    let mut m = Match::with_decks(0, vec![identity_deck()], 9, Rules::default());
    m.scores = [0, 40, 0, 0];
    for _ in 0..(8 + 1 + 8 + 8) {
        assert_eq!(m.public().best_trump, None);
        lowest(&mut m);
    }
    let l = log(&m);
    let i = find(&m, "TrickWon trick=3 seat=1 team=1 points=5");
    assert_eq!(
        l[i + 1..i + 3],
        [
            "GameWon team=1 reason=trick scores=5,45,0,5",
            "HandEnd hand=0 aborted=true",
        ]
    );
    assert_eq!(count_matching(&m, "Bonus"), 0);
}

#[test]
fn new_draws_the_dealer_then_shuffles_every_hand() {
    let seed = 0xDEAD_BEEF;
    let mut m = Match::new(seed, Rules::default());
    let mut rng = Rng::new(seed);
    let dealer = rng.bounded(8) as u8;
    let mut deck = identity_deck();
    rng.shuffle(&mut deck);
    assert_eq!(m.dealer(), dealer);
    assert_eq!(m.truth().deck, deck);
    assert_eq!(
        m.events()[0].1,
        Event::MatchStart {
            first_dealer: dealer
        }
    );
    finish_hand(&mut m);
    let mut deck2 = identity_deck();
    rng.shuffle(&mut deck2);
    assert_eq!(m.truth().deck, deck2, "each hand reshuffles all 53 cards");
    assert_eq!(m.dealer(), (dealer + 1) % 8);
}

#[test]
fn with_decks_uses_given_decks_then_seeded_shuffles() {
    let d0 = identity_deck();
    let mut m = Match::with_decks(3, vec![d0], 77, Rules::default());
    assert_eq!(m.dealer(), 3);
    assert_eq!(m.truth().deck, d0);
    finish_hand(&mut m);
    let mut rng = Rng::new(77);
    let mut d1 = identity_deck();
    rng.shuffle(&mut d1);
    assert_eq!(m.truth().deck, d1);
}

#[test]
#[should_panic(expected = "53 cards")]
fn with_decks_rejects_a_bad_deck() {
    let mut d = identity_deck();
    d[5] = d[6];
    let _ = Match::with_decks(0, vec![d], 1, Rules::default());
}

#[test]
fn build_deck_follows_the_deal_convention() {
    let d = deal(5, ["5H", "", "", "", "", "", "JK", ""], "7S AS");
    let m = Match::with_decks(5, vec![d], 1, Rules::default());
    assert!(has(m.truth().hands[0], c("5H")));
    assert!(has(m.truth().hands[6], JOKER));
    assert_eq!(d[40], c("7S"));
    assert_eq!(d[41], c("AS"));
    for s in 0..SEATS {
        assert_eq!(count(m.truth().hands[s]), 5);
    }
    assert!(build_deck(8, &[0; 8], &[]).is_err());
    assert!(build_deck(0, &[set("2C"), set("2C"), 0, 0, 0, 0, 0, 0], &[]).is_err());
    assert!(build_deck(0, &[0; 8], &[c("2C"), c("2C")]).is_err());
    assert!(build_deck(0, &[0; 8], &[60]).is_err());
}

// ----- robbing an ordinary turned card (steps 4 and 6, H2) -----

#[test]
fn robber_takes_at_their_own_first_turn_after_seeing_earlier_cards() {
    // Spades trumps; seat 2 holds all four aces and may burn any of its six cards (H3, amended
    // 2026-10-14).
    let mut m = game(7, ["", "", "AS AC AD AH 9C", "", "", "", "", ""], "7S");
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    let before = m.events().len();
    lowest(&mut m);
    lowest(&mut m);
    assert_eq!(m.phase(), Phase::AwaitBurn(2));
    let tail = log_from(&m, before);
    assert_eq!(tail[0], "ProvedNoAce seat=0");
    assert!(tail[1].starts_with("Played seat=0"));
    assert_eq!(tail[2], "ProvedNoAce seat=1");
    assert!(tail[3].starts_with("Played seat=1"));
    assert_eq!(tail[4], "Took seat=2 card=7S");
    assert_eq!(tail.len(), 5);
    let p = m.public();
    assert_eq!(p.hand_sizes[2], 6);
    assert_eq!(p.table_card, None);
    assert_eq!(p.robber, Some(2));
    assert_eq!(p.turned, vec![(c("7S"), Some(2))]);
    // Any card but the ace of trumps (AS) may be burned: plain aces and the card just taken may.
    assert_eq!(m.legal_mask(), set("AC AD AH 9C 7S"));
    assert_eq!(
        m.apply(Action::Burn(c("AS"))).unwrap_err().reason,
        "the ace of trumps may never be burned (rules 21 and 24)"
    );
    assert_eq!(
        m.apply(Action::Burn(c("2C"))).unwrap_err().reason,
        "that card is not in the burning player's hand"
    );
    assert_eq!(
        m.apply(Action::Play(c("9C"))).unwrap_err().reason,
        "a burn is due: the player who took the turned card must burn first"
    );
    burn(&mut m, "7S");
    assert_eq!(m.phase(), Phase::AwaitPlay(2));
    assert_eq!(m.public().hand_sizes[2], 5);
    assert_eq!(m.view(2).own_burns, set("7S"));
    // Seats after the robber never get a ProvedNoAce: the robber is known.
    finish_hand(&mut m);
    let first_hand: Vec<u8> = proved_seats(&m);
    assert_eq!(first_hand, vec![0, 1]);
}

#[test]
fn robber_leading_trick_one_robs_before_leading() {
    let mut m = game(7, ["AS", "", "", "", "", "", "", ""], "7S");
    assert_eq!(m.phase(), Phase::AwaitBurn(0));
    let tail = log(&m);
    assert_eq!(
        tail[tail.len() - 2..],
        ["Turned card=7S", "Took seat=0 card=7S"]
    );
    let p = m.public();
    assert_eq!(p.trick_no, 1);
    assert_eq!(p.leader, 0);
    assert_eq!(p.trick_cards, vec![]);
    lowest(&mut m);
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    finish_hand(&mut m);
    assert!(proved_seats(&m).is_empty());
    assert_eq!(count_matching(&m, "AceUndealt"), 0);
}

#[test]
fn ace_of_hearts_holder_does_not_rob_when_hearts_are_not_trumps() {
    // Spades trumps, the AS undealt, seat 0 holds the A♥.
    let mut m = game(7, ["AH", "", "", "", "", "", "", ""], "7S AS");
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    play(&mut m, "AH");
    let tail = log(&m);
    assert_eq!(
        tail[tail.len() - 2..],
        ["ProvedNoAce seat=0", "Played seat=0 card=AH"]
    );
}

#[test]
fn ace_undealt_is_proved_after_trick_one() {
    // Spades trumps with the AS undealt: all eight prove they lack it, in play order.
    let mut m = game(7, ["", "", "", "", "", "", "", ""], "7S AS");
    for _ in 0..8 {
        assert!(!m.public().ace_undealt);
        lowest(&mut m);
    }
    assert_eq!(proved_seats(&m), vec![0, 1, 2, 3, 4, 5, 6, 7]);
    let tail = log(&m);
    assert!(tail[tail.len() - 2].starts_with("TrickWon trick=1"));
    assert_eq!(tail[tail.len() - 1], "AceUndealt");
    let p = m.public();
    assert!(p.ace_undealt);
    assert_eq!(p.proved_no_ace, [true; 8]);
    assert_eq!(
        p.table_card,
        Some(c("7S")),
        "an unrobbed turned card stays on the table"
    );
    assert_eq!(p.hidden_count, 12);
    finish_hand(&mut m);
    assert_eq!(count_matching(&m, "AceUndealt"), 1);
    assert_eq!(count_matching(&m, "Took"), 0);
}

// ----- the dealer robbing at once (steps 2 and 3, H4) -----

#[test]
fn turned_plain_ace_is_robbed_by_the_dealer_at_once() {
    let mut m = game(7, ["", "", "", "", "", "", "", "2D 3D 4D 5D 6D"], "AC");
    let tail = log(&m);
    assert_eq!(
        tail[tail.len() - 2..],
        ["Turned card=AC", "Took seat=7 card=AC"]
    );
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    let p = m.public();
    assert_eq!(p.trick_no, 0, "the burn comes before the first lead");
    assert_eq!(p.trump, Some(Suit::Clubs));
    assert_eq!(p.robber, Some(7));
    assert_eq!(p.hand_sizes[7], 6);
    // The turned ace (the ace of trumps) may not be burned (rule 21).
    assert_eq!(m.legal_mask(), set("2D 3D 4D 5D 6D"));
    burn(&mut m, "6D");
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    assert_eq!(m.public().trick_no, 1);
    assert_eq!(m.public().hidden_count, 13);
    finish_hand(&mut m);
    assert!(
        proved_seats(&m).is_empty(),
        "nobody proves anything: the robber is known"
    );
    assert_eq!(count_matching(&m, "AceUndealt"), 0);
}

#[test]
fn turned_ace_of_hearts_makes_hearts_trumps_and_is_robbed() {
    let mut m = game(7, ["", "", "", "", "", "", "", "2D 3D 4D 5D 6D"], "AH");
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    assert_eq!(m.public().trump, Some(Suit::Hearts));
    assert_eq!(count(trump_mask(Suit::Hearts)), 14);
    assert!(
        !has(m.legal_mask(), c("AH")),
        "the A♥ is the ace of trumps here"
    );
    burn(&mut m, "2D");
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    assert_eq!(m.public().robber, Some(7));
}

// ----- the turned Joker (step 2, rule 24, H4) -----

#[test]
fn turned_joker_then_ordinary_card_with_the_ace_undealt() {
    let mut m = game(
        7,
        ["", "", "", "", "", "", "", "AC AD 2C 3C 4C"],
        "JK 7S AS",
    );
    let tail = log(&m);
    assert_eq!(
        tail[tail.len() - 2..],
        ["Turned card=JK", "Took seat=7 card=JK"]
    );
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    let p = m.public();
    assert_eq!(p.trump, None, "trumps are not known during the Joker burn");
    assert_eq!(p.trick_no, 0);
    assert!(p.joker_turned);
    assert_eq!(p.table_card, None);
    assert_eq!(p.robber, None, "taking the Joker is not a rob");
    assert_eq!(p.hand_sizes[7], 6);
    assert_eq!(p.hidden_count, 12);
    assert_eq!(count(m.truth().undealt), 12);
    // Any card may be burned before trumps are known, aces and the Joker included.
    assert_eq!(m.legal_mask(), set("AC AD 2C 3C 4C JK"));
    let before = m.events().len();
    burn(&mut m, "JK");
    assert_eq!(
        log_from(&m, before),
        ["Burned seat=7 card=JK", "BurnDone seat=7", "Turned card=7S",]
    );
    // H4 (amended 2026-10-14): the dealer proves nothing until it plays its first card.
    assert!(!m.public().proved_no_ace[7]);
    let p = m.public();
    assert_eq!(p.trump, Some(Suit::Spades));
    assert_eq!(p.table_card, Some(c("7S")));
    assert_eq!(p.turned, vec![(JOKER, Some(7)), (c("7S"), None)]);
    assert_eq!(p.trick_no, 1);
    assert_eq!(p.hidden_count, 12);
    assert_eq!(count(m.truth().undealt), 11);
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    for _ in 0..8 {
        lowest(&mut m);
    }
    // Every seat proves it lacks the ace as it plays, the dealer last.
    assert_eq!(proved_seats(&m), vec![0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(log(&m).last().unwrap(), "AceUndealt");
}

#[test]
fn turned_joker_then_ordinary_card_robbed_by_a_non_dealer() {
    let mut m = game(7, ["", "", "AS", "", "", "", "", "2C 3C 4C 5C 6C"], "JK 7S");
    burn(&mut m, "2C");
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    lowest(&mut m);
    lowest(&mut m);
    assert_eq!(m.phase(), Phase::AwaitBurn(2));
    assert_eq!(log(&m).last().unwrap(), "Took seat=2 card=7S");
    lowest(&mut m);
    let p = m.public();
    assert_eq!(
        p.burn_counts,
        [0, 0, 1, 0, 0, 0, 0, 1],
        "two burns, two places"
    );
    assert_eq!(p.hidden_count, 13);
    assert_eq!(p.turned, vec![(JOKER, Some(7)), (c("7S"), Some(2))]);
    assert_eq!(p.robber, Some(2));
    finish_hand(&mut m);
    assert_eq!(proved_seats(&m), vec![0, 1]);
}

#[test]
fn turned_joker_then_ace_dealer_takes_both() {
    let mut m = game(7, ["", "", "", "", "", "", "", "2C 3C 4C 5C 6C"], "JK AD");
    burn(&mut m, "2C");
    let tail = log(&m);
    assert_eq!(
        tail[tail.len() - 2..],
        ["Turned card=AD", "Took seat=7 card=AD"]
    );
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    assert_eq!(m.public().trick_no, 0);
    assert!(!has(m.legal_mask(), c("AD")));
    burn(&mut m, "3C");
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    let p = m.public();
    assert_eq!(p.burn_counts[7], 2);
    assert_eq!(count(m.truth().undealt), 11);
    assert_eq!(p.hidden_count, 13);
    assert_eq!(p.robber, Some(7));
    assert_eq!(p.trump, Some(Suit::Diamonds));
    assert_eq!(m.truth().burns[7], set("2C 3C"));
    finish_hand(&mut m);
    assert!(proved_seats(&m).is_empty());
}

#[test]
fn turned_joker_then_ace_of_hearts() {
    let mut m = game(7, ["", "", "", "", "", "", "", "2C 3C 4C 5C 6C"], "JK AH");
    burn(&mut m, "JK");
    assert_eq!(m.public().trump, Some(Suit::Hearts));
    assert_eq!(log(&m).last().unwrap(), "Took seat=7 card=AH");
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
}

#[test]
fn turned_joker_with_the_dealer_holding_the_new_ace() {
    // H4 (amended 2026-10-14): the dealer robs the ordinary card at its own first turn in
    // trick 1, as the eighth card, after seeing the other seven.
    let mut m = game(7, ["", "", "", "", "", "", "", "AS 2C 3C 4C 5C"], "JK 7S");
    burn(&mut m, "2C");
    assert_eq!(log(&m).last().unwrap(), "Turned card=7S");
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    for _ in 0..7 {
        lowest(&mut m);
    }
    assert_eq!(log(&m).last().unwrap(), "Took seat=7 card=7S");
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    assert_eq!(m.legal_mask(), set("3C 4C 5C 7S JK"));
    burn(&mut m, "JK");
    assert_eq!(m.phase(), Phase::AwaitPlay(7));
    assert_eq!(m.public().robber, Some(7));
    assert_eq!(m.truth().burns[7], set("2C JK"));
    finish_hand(&mut m);
    assert_eq!(proved_seats(&m), vec![0, 1, 2, 3, 4, 5, 6]);
}

#[test]
fn turned_joker_dealer_burns_the_future_ace_of_trumps() {
    // H3: the dealer burns the AS before trumps are known; the 7S then makes
    // it the ace of trumps. Nobody holds it, so nobody robs, and after trick 1 the ace is
    // announced out of play although it is in the dealer's burn, not the undealt pack.
    let mut m = game(7, ["", "", "", "", "", "", "", "AS 2C 3C 4C 5C"], "JK 7S");
    burn(&mut m, "AS");
    assert_eq!(log(&m).last().unwrap(), "Turned card=7S");
    assert_eq!(m.public().table_card, Some(c("7S")));
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    for _ in 0..8 {
        lowest(&mut m);
    }
    assert_eq!(m.public().robber, None);
    assert!(m.public().ace_undealt);
    assert_eq!(log(&m).last().unwrap(), "AceUndealt");
    assert!(!has(m.truth().undealt, c("AS")));
    assert_eq!(m.truth().burns[7], set("AS"));
}

// ----- scoring: the D2 deal, written out card by card -----
//
// Diamonds are trumps (2D turned, AD and 5D undealt). Team A takes trick 1 with the KD and
// trick 4 with the JD (the best trump; the 5 is never played); team B takes tricks 2 and 5;
// team C takes trick 3.

const D2_HANDS: [&str; 8] = [
    "KD 10C 4S 5H 3C",
    "QH KC 10S 4H AS",
    "JH 9C KS 10H 2S",
    "3H 8C 9S 9H 3S",
    "QC 7C 8S JD 2H",
    "JC 6C 7S 8H KH",
    "QS 5C 6S 7H AC",
    "JS 4C 5S 6H 2C",
];
const D2_PACK: &str = "2D 5D JK AH AD QD 10D 9D 8D 7D 6D 4D 3D";
const D2_TRICKS: [&str; 5] = [
    "KD QH JH 3H QC JC QS JS",
    "10C KC 9C 8C 7C 6C 5C 4C",
    "10S KS 9S 8S 7S 6S 5S 4S",
    "10H 9H JD 8H 7H 6H 5H 4H",
    "2H KH AC 2C 3C AS 2S 3S",
];

fn d2(scores: [u8; 4], bonus: [bool; 4]) -> Match {
    let mut m = game(7, D2_HANDS, D2_PACK);
    m.scores = scores;
    m.bonus_taken = bonus;
    m
}

#[test]
fn d2_end_of_hand_bonus_goes_to_the_highest_trump() {
    let mut m = d2([0; 4], [false; 4]);
    for t in D2_TRICKS {
        plays(&mut m, t);
    }
    let l = log(&m);
    let won: Vec<&String> = l.iter().filter(|s| s.starts_with("TrickWon")).collect();
    assert_eq!(
        won,
        [
            "TrickWon trick=1 seat=0 team=0 points=5",
            "TrickWon trick=2 seat=1 team=1 points=5",
            "TrickWon trick=3 seat=2 team=2 points=5",
            "TrickWon trick=4 seat=4 team=0 points=5",
            "TrickWon trick=5 seat=5 team=1 points=5",
        ]
    );
    let i = find(&m, "TrickWon trick=5 seat=5 team=1 points=5");
    assert_eq!(
        l[i + 1..i + 4],
        [
            "Bonus team=0 points=6 kind=end card=JD",
            "HandEnd hand=0 aborted=false",
            "HandStart hand=1 game=0 dealer=0",
        ]
    );
    let first = find(&m, "TrickWon trick=1 seat=0 team=0 points=5");
    assert_eq!(l[first + 1], "AceUndealt");
    assert_eq!(m.scores(), [16, 10, 5, 0]);
    assert_eq!(m.bonus_taken(), [true, false, false, false]);
}

#[test]
fn d2_end_bonus_alone_reaches_41() {
    // Team A on 25 makes 35 from its tricks; the +6 alone takes it to 41.
    let mut m = d2([25, 0, 0, 0], [false; 4]);
    for t in D2_TRICKS {
        plays(&mut m, t);
    }
    let l = log(&m);
    let i = find(&m, "TrickWon trick=5 seat=5 team=1 points=5");
    assert_eq!(
        l[i + 1..i + 5],
        [
            "Bonus team=0 points=6 kind=end card=JD",
            "GameWon team=0 reason=end_bonus scores=41,10,5,0",
            "HandEnd hand=0 aborted=false",
            "HandStart hand=1 game=1 dealer=0",
        ]
    );
    assert_eq!(m.games_won(), [1, 0, 0, 0]);
    assert_eq!(m.scores(), [0; 4]);
    assert_eq!(m.bonus_taken(), [false; 4]);
}

#[test]
fn d2_two_teams_over_41_after_trick_five_the_trick_winner_wins() {
    // Team B reaches 45 with trick 5 before team A's end bonus (which would make 41) is awarded.
    let mut m = d2([25, 35, 0, 0], [false; 4]);
    for t in D2_TRICKS {
        plays(&mut m, t);
    }
    let l = log(&m);
    let i = find(&m, "TrickWon trick=5 seat=5 team=1 points=5");
    assert_eq!(
        l[i + 1..i + 3],
        [
            "GameWon team=1 reason=trick scores=35,45,5,0",
            "HandEnd hand=0 aborted=false",
        ]
    );
    assert_eq!(count_matching(&m, "Bonus"), 0, "the pending bonus is lost");
}

#[test]
fn d2_game_ending_at_trick_three_loses_the_pending_bonus() {
    // Team C on 40 wins trick 3; team A's KD (best trump so far) never gets its bonus.
    let mut m = d2([0, 0, 40, 0], [false; 4]);
    for t in &D2_TRICKS[..3] {
        plays(&mut m, t);
    }
    let l = log(&m);
    let i = find(&m, "TrickWon trick=3 seat=2 team=2 points=5");
    assert_eq!(
        l[i + 1..i + 4],
        [
            "GameWon team=2 reason=trick scores=5,5,45,0",
            "HandEnd hand=0 aborted=true",
            "HandStart hand=1 game=1 dealer=0",
        ]
    );
    assert_eq!(count_matching(&m, "Bonus"), 0);
    assert_eq!(
        m.dealer(),
        0,
        "the deal moves left after a game ends mid-hand (H5)"
    );
    assert_eq!(m.hand_index(), 1);
    assert_eq!(m.game_index(), 1);
}

#[test]
fn d2_illegal_plays_are_rejected_with_a_reason() {
    let mut m = d2([0; 4], [false; 4]);
    plays(&mut m, D2_TRICKS[0]);
    // Trick 2: seat 0 leads the 10C; seat 1 holds the KC so may not discard the 10S.
    play(&mut m, "10C");
    let err = m.apply(Action::Play(c("10S"))).unwrap_err();
    assert_eq!(
        err.reason,
        "you hold the suit led: follow suit or play a trump (rule 14)"
    );
    let err = m.apply(Action::Play(c("9C"))).unwrap_err();
    assert_eq!(
        err.reason,
        "that card is not in the hand of the player to act"
    );
    let err = m.apply(Action::Burn(c("KC"))).unwrap_err();
    assert_eq!(
        err.reason,
        "a card must be played: nobody has a burn to make"
    );
    let err = m.apply(Action::Play(53)).unwrap_err();
    assert_eq!(err.reason, "no such card: card indices run from 0 to 52");
    assert_eq!(
        err.to_string(),
        "illegal play of card #53: no such card: card indices run from 0 to 52"
    );
}

// ----- the 5 of trumps (rule 29): the D3 deal -----
//
// Diamonds trumps (2D turned, everything but the 5D and KD undealt); seat 0 leads the 5D.

fn d3(scores: [u8; 4], bonus: [bool; 4], rules: Rules) -> Match {
    let mut m = game_with(
        7,
        ["5D KD", "", "", "", "", "", "", ""],
        "2D JD JK AH AD QD 10D 9D 8D 7D 6D 4D 3D",
        rules,
    );
    m.scores = scores;
    m.bonus_taken = bonus;
    play(&mut m, "5D");
    for _ in 0..7 {
        lowest(&mut m);
    }
    m
}

fn after_trick_one(m: &Match) -> Vec<String> {
    let i = find(m, "TrickWon trick=1 seat=0 team=0 points=5");
    log_from(m, i)
}

#[test]
fn immediate_bonus_when_the_five_wins_its_trick() {
    let mut m = d3([0; 4], [false; 4], Rules::default());
    assert_eq!(
        after_trick_one(&m),
        [
            "TrickWon trick=1 seat=0 team=0 points=5",
            "Bonus team=0 points=6 kind=immediate card=5D",
            "AceUndealt",
        ]
    );
    assert_eq!(m.scores(), [11, 0, 0, 0]);
    assert!(m.public().five_played);
    finish_hand(&mut m);
    assert_eq!(
        count_matching(&m, "Bonus"),
        1,
        "no second bonus at the end of the hand"
    );
}

#[test]
fn immediate_bonus_reaching_41_ends_the_game_mid_hand() {
    let m = d3([30, 0, 0, 0], [false; 4], Rules::default());
    assert_eq!(
        after_trick_one(&m)[..5],
        [
            "TrickWon trick=1 seat=0 team=0 points=5",
            "Bonus team=0 points=6 kind=immediate card=5D",
            "GameWon team=0 reason=immediate_bonus scores=41,0,0,0",
            "HandEnd hand=0 aborted=true",
            "HandStart hand=1 game=1 dealer=0",
        ]
    );
    // The new game: scores and bonus flags reset, games kept, deal moved left.
    assert_eq!(m.scores(), [0; 4]);
    assert_eq!(m.bonus_taken(), [false; 4]);
    assert_eq!(m.games_won(), [1, 0, 0, 0]);
    assert_eq!(m.game_index(), 1);
    assert_eq!(m.dealer(), 0);
    let p = m.public();
    assert_eq!(p.trick_winners, vec![]);
    assert!(!p.five_played);
}

#[test]
fn trick_reaching_41_wins_even_with_a_bonus_in_the_same_trick() {
    // On 36 (bonus already taken): the trick makes 41, then the later bonus of 5 makes 46.
    let m = d3([36, 0, 0, 0], [true, false, false, false], Rules::default());
    assert_eq!(
        after_trick_one(&m)[1..3],
        [
            "Bonus team=0 points=5 kind=immediate card=5D",
            "GameWon team=0 reason=trick scores=46,0,0,0",
        ]
    );
}

#[test]
fn plain_trick_from_40_makes_45() {
    let m = d3([40, 0, 0, 0], [false; 4], Rules::default());
    assert_eq!(
        after_trick_one(&m)[1..3],
        [
            "Bonus team=0 points=6 kind=immediate card=5D",
            "GameWon team=0 reason=trick scores=51,0,0,0",
        ]
    );
    let mut m = d2([40, 0, 0, 0], [false; 4]);
    plays(&mut m, D2_TRICKS[0]);
    let i = find(&m, "TrickWon trick=1 seat=0 team=0 points=5");
    assert_eq!(
        log(&m)[i + 1],
        "GameWon team=0 reason=trick scores=45,0,0,0"
    );
}

// ----- the match and the money (rules 35-36, H6) -----

#[test]
fn second_game_win_ends_the_match_with_the_money() {
    let mut m = game(
        7,
        ["5D KD", "", "", "", "", "", "", ""],
        "2D JD JK AH AD QD 10D 9D 8D 7D 6D 4D 3D",
    );
    m.scores = [30, 0, 0, 0];
    m.games_won = [1, 1, 1, 1];
    m.game_index = 4;
    play(&mut m, "5D");
    for _ in 0..7 {
        lowest(&mut m);
    }
    let l = log(&m);
    assert_eq!(
        l[l.len() - 3..],
        [
            "GameWon team=0 reason=immediate_bonus scores=41,0,0,0",
            "HandEnd hand=0 aborted=true",
            "MatchWon team=0 money=30,-10,-10,-10",
        ]
    );
    assert_eq!(m.phase(), Phase::MatchOver);
    assert_eq!(m.to_act(), None);
    assert_eq!(m.games_won(), [2, 1, 1, 1]);
    assert_eq!(m.money(), Some([30, -10, -10, -10]));
    assert!(m.legal_actions().is_empty());
    assert_eq!(m.legal_mask(), 0);
    let snapshot = m.clone();
    for a in [
        Action::Play(c("KD")),
        Action::Burn(c("2C")),
        Action::Play(200),
    ] {
        assert!(m.apply(a).is_err());
    }
    assert_eq!(m, snapshot);
    assert_eq!(
        m.apply(Action::Play(c("KD"))).unwrap_err().reason,
        "the match is over"
    );
}

#[test]
fn money_with_a_non_default_stake() {
    let rules = Rules { stake: 10 };
    let mut m = game_with(
        7,
        ["", "5D KD", "", "", "", "", "", ""],
        "2D JD JK AH AD QD 10D 9D 8D 7D 6D 4D 3D",
        rules,
    );
    m.scores = [0, 40, 0, 0];
    m.games_won = [0, 1, 0, 0];
    // Seat 0 leads its lowest card; seat 1 trumps with the 5D and wins.
    lowest(&mut m);
    play(&mut m, "5D");
    for _ in 0..6 {
        lowest(&mut m);
    }
    assert_eq!(m.money(), Some([-20, 60, -20, -20]));
    assert_eq!(m.money().unwrap().iter().sum::<i32>(), 0);
    assert_eq!(
        log(&m).last().unwrap(),
        "MatchWon team=1 money=-20,60,-20,-20"
    );
}

#[test]
fn a_match_lasts_at_most_five_games_and_the_deal_always_moves_left() {
    // Play whole matches with the lowest-card policy and check the dealer and game sequence.
    for seed in 0..6u64 {
        let mut m = Match::new(seed, Rules::default());
        while m.phase() != Phase::MatchOver {
            lowest(&mut m);
        }
        let mut dealers = Vec::new();
        let mut games = Vec::new();
        for (_, e) in m.events() {
            if let Event::HandStart {
                dealer, game_index, ..
            } = *e
            {
                dealers.push(dealer);
                games.push(game_index);
            }
        }
        for w in dealers.windows(2) {
            assert_eq!(w[1], (w[0] + 1) % 8);
        }
        assert!(games.last().copied().unwrap_or(0) <= 4);
        assert_eq!(m.games_won().iter().filter(|&&g| g == 2).count(), 1);
        assert_eq!(
            m.games_won().iter().map(|&g| u32::from(g)).sum::<u32>(),
            u32::from(games.last().copied().unwrap_or(0)) + 1
        );
    }
}

// ----- the jink (rule 33) -----
//
// Spades trumps; team A (seats 0 and 4) holds ten trumps and nobody else holds any. Seat 0 holds
// the AS, so robs the turned 9S before leading.

fn jink_deal(seat4: &str, pack: &str, scores: [u8; 4]) -> Match {
    let mut m = game(7, ["JS AS KS QS 2S", "", "", "", seat4, "", "", ""], pack);
    m.scores = scores;
    finish_hand(&mut m);
    m
}

#[test]
fn jink_wins_the_game_before_any_end_bonus() {
    let m = jink_deal("3S 4S 6S 7S 8S", "9S 5S 10S JK AH", [0; 4]);
    let l = log(&m);
    let i = find(&m, "TrickWon trick=5 seat=0 team=0 points=5");
    assert_eq!(
        l[i + 1..i + 4],
        [
            "GameWon team=0 reason=jink scores=25,0,0,0",
            "HandEnd hand=0 aborted=false",
            "HandStart hand=1 game=1 dealer=0",
        ]
    );
    assert_eq!(count_matching(&m, "Bonus"), 0);
    assert_eq!(m.games_won(), [1, 0, 0, 0]);
}

#[test]
fn jink_after_an_immediate_bonus() {
    let m = jink_deal("3S 4S 5S 6S 7S", "9S 8S 10S JK AH", [0; 4]);
    assert_eq!(
        count_matching(&m, "Bonus team=0 points=6 kind=immediate card=5S"),
        1
    );
    assert_eq!(
        count_matching(&m, "GameWon team=0 reason=jink scores=31,0,0,0"),
        1
    );
}

#[test]
fn jink_and_41_in_one_hand_count_as_one_game() {
    // On 20, the fifth trick makes 45: the game is won by the trick, once.
    let m = jink_deal("3S 4S 6S 7S 8S", "9S 5S 10S JK AH", [20, 0, 0, 0]);
    assert_eq!(count_matching(&m, "GameWon"), 1);
    assert_eq!(
        count_matching(&m, "GameWon team=0 reason=trick scores=45,0,0,0"),
        1
    );
    assert_eq!(m.games_won(), [1, 0, 0, 0]);
}

// ----- illegal actions never panic and never change anything (H1) -----

/// Every one of the 512 possible actions is accepted exactly when offered, and a refused one
/// leaves the match unchanged.
fn assert_only_legal_accepted(m: &Match) {
    let legal = m.legal_actions();
    for card in 0..=255u8 {
        for a in [Action::Burn(card), Action::Play(card)] {
            let mut trial = m.clone();
            let ok = trial.apply(a).is_ok();
            assert_eq!(ok, legal.contains(&a), "{a:?} in phase {:?}", m.phase());
            if !ok {
                assert_eq!(&trial, m, "a refused action changed the match");
            }
        }
    }
}

#[test]
fn only_offered_actions_are_accepted() {
    // A Joker burn before trumps are known.
    let m = game(
        7,
        ["", "", "", "", "", "", "", "AC AD 2C 3C 4C"],
        "JK 7S AS",
    );
    assert_only_legal_accepted(&m);
    // A rob burn, a plain lead, a trump lead, mid-trick.
    let mut m = game(7, ["", "", "AS AC AD AH 9C", "", "", "", "", ""], "7S");
    assert_only_legal_accepted(&m);
    lowest(&mut m);
    assert_only_legal_accepted(&m);
    lowest(&mut m);
    assert_only_legal_accepted(&m);
    let mut m = d2([0; 4], [false; 4]);
    play(&mut m, "KD");
    assert_only_legal_accepted(&m);
    // The match over.
    let mut m = Match::new(5, Rules::default());
    while m.phase() != Phase::MatchOver {
        lowest(&mut m);
    }
    assert_only_legal_accepted(&m);
}

// ----- views and visibility -----

#[test]
fn views_show_only_what_each_seat_may_see() {
    let mut m = game(7, ["", "", "AS", "", "", "", "", "2C 3C 4C 5C 6C"], "JK 7S");
    burn(&mut m, "2C");
    lowest(&mut m);
    lowest(&mut m);
    lowest(&mut m); // seat 2 burns after robbing
    for seat in 0..8u8 {
        let v = m.view(seat);
        assert_eq!(v.hand, m.truth().hands[usize::from(seat)]);
        assert_eq!(v.own_burns, m.truth().burns[usize::from(seat)]);
        let dealt: Vec<u8> = v
            .events
            .iter()
            .filter_map(|e| match *e {
                Event::Dealt { seat, .. } => Some(seat),
                _ => None,
            })
            .collect();
        assert_eq!(dealt, vec![seat], "a seat sees only its own deal");
        let burned: Vec<u8> = v
            .events
            .iter()
            .filter_map(|e| match *e {
                Event::Burned { seat, .. } => Some(seat),
                _ => None,
            })
            .collect();
        let expected: Vec<u8> = [7u8, 2].into_iter().filter(|&s| s == seat).collect();
        assert_eq!(burned, expected, "a seat sees only its own burns");
        assert_eq!(
            v.events
                .iter()
                .filter(|e| matches!(e, Event::BurnDone { .. }))
                .count(),
            2
        );
        assert_eq!(v.public, m.public());
        assert_eq!(v.canonical_events().len(), v.events.len());
    }
    // The burned cards are hidden.
    let burned = m.truth().burns.iter().fold(0, |a, &b| a | b);
    assert_eq!(hidden_cards(&m) & burned, burned);
}

#[test]
fn transcripts_render_one_line_per_event() {
    let mut m = d2([0; 4], [false; 4]);
    for t in D2_TRICKS {
        plays(&mut m, t);
    }
    let public_lines = public_log(&m).len();
    let t = transcript(&m);
    assert_eq!(t.lines().count(), public_lines);
    assert!(t.contains("Seat 0 plays the king of diamonds."));
    assert!(t.contains(
        "Best-trump bonus: 6 points to team A at the end of the hand, for the trick with the jack of diamonds. Scores: A 16, B 10, C 5, D 0."
    ));
    assert!(t.contains("The 2 of diamonds is turned: diamonds are trumps."));
    assert!(!t.contains("is dealt"));
    let full = transcript_full(&m);
    assert_eq!(full.lines().count(), m.events().len());
    assert!(full.contains("Seat 0 is dealt 3C 10C KD 5H 4S."));
}
