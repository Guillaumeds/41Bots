//! Each rule of `rules-v1` (docs/M1-SPEC.md §4) on a hand-made position.
//!
//! Every position is real: the deal is built with `build_deck`, the match with
//! `Match::with_decks`, and the cards are played through the engine up to the decision, so the
//! bot sees exactly the `SeatView` a table would give it. Dealer 7 deals unless stated, so seat 0
//! leads trick 1 and seats then play in order 0, 1, …, 7.

use bot::Bot;
use bots::{Rule, RulesBot, Tracker, decide, rules_v1::score_if_team_wins_trick};
use engine::SEATS;
use engine::cards::{
    ALL_CARDS, Bits, Card, DECK_SIZE, bit, cards, count, has, names, parse, suit_of,
};
use engine::game::{Action, Match, Phase, Rules, build_deck};
use engine::rules::{trump_mask, trump_rank};

// ===== helpers =====

fn c(n: &str) -> Card {
    parse(n).unwrap_or_else(|| panic!("bad card name {n}"))
}

fn set(s: &str) -> Bits {
    s.split_whitespace().fold(0, |a, n| a | bit(c(n)))
}

/// A deck for one hand. `hands[s]` names up to five cards for seat `s`; `pack` names the cards
/// from deck position 40 on (the turned card first; after a turned Joker, the second turned card
/// next). Each hand is then filled up to five cards, in deal order from the dealer's left, with
/// the lowest-index cards that are not trumps (the suit of the first ordinary card of `pack`,
/// plus the Joker and the A♥), and only if those run out with the lowest number trumps. The rest
/// is the undealt pack. So a seat holds a trump only if the test names it (or the plain cards run
/// out), and the ace of trumps is undealt unless a hand names it.
fn deck(dealer: u8, hands: [&str; 8], pack: &str) -> [Card; DECK_SIZE] {
    let pack: Vec<Card> = pack.split_whitespace().map(c).collect();
    let trump = pack
        .iter()
        .find_map(|&x| suit_of(x))
        .expect("an ordinary card in the pack");
    let trumps = trump_mask(trump);
    let mut full: [Bits; SEATS] = core::array::from_fn(|s| set(hands[s]));
    let mut used = full.iter().fold(0, |a, &h| a | h) | pack.iter().fold(0, |a, &x| a | bit(x));
    let numbers = cards(trumps)
        .filter(|&x| (1..=8).contains(&trump_rank(x, trump)))
        .fold(0, |a, x| a | bit(x));
    let mut fill = cards(ALL_CARDS & !trumps).chain(cards(numbers));
    for i in 0..SEATS {
        let s = (usize::from(dealer) + 1 + i) % SEATS;
        while count(full[s]) < 5 {
            let x = fill
                .by_ref()
                .find(|&x| !has(used, x))
                .expect("enough cards to fill the hands");
            full[s] |= bit(x);
            used |= bit(x);
        }
    }
    build_deck(dealer, &full, &pack).expect("a valid hand-made deal")
}

fn one_hand(dealer: u8, hands: [&str; 8], pack: &str) -> Match {
    Match::with_decks(dealer, vec![deck(dealer, hands, pack)], 1, Rules::HOUSE)
}

/// Play the named cards in order; `-` plays the lowest legal card (by card index) of the seat
/// to act.
fn play(m: &mut Match, plays: &str) {
    for p in plays.split_whitespace() {
        let a = if p == "-" {
            m.legal_actions()[0]
        } else {
            Action::Play(c(p))
        };
        if let Err(e) = m.apply(a) {
            panic!("{e} (phase {:?})", m.phase());
        }
    }
}

/// Ask `rules-v1` for the seat to act. The decision through `decide` and through the `Bot` trait
/// must agree, and it must be legal.
fn ask(m: &Match) -> (Action, Rule) {
    let seat = m.to_act().expect("a seat to act");
    let view = m.view(seat);
    let legal = m.legal_actions();
    let (a, rule) = decide(&view, &Rules::HOUSE, &legal);
    let mut bot = RulesBot::new();
    bot.start_match(seat, Rules::HOUSE, 0xDEAD_BEEF);
    assert_eq!(
        bot.act(&view, &legal),
        a,
        "the Bot trait and decide disagree"
    );
    assert!(legal.contains(&a), "rules-v1 chose an illegal {a:?}");
    (a, rule)
}

/// Ask the bot, check its choice and rule, and apply it.
fn expect(m: &mut Match, action: Action, rule: Rule) {
    let seat = m.to_act();
    let hand = seat.map(|s| names(m.truth().hands[usize::from(s)]));
    assert_eq!(
        ask(m),
        (action, rule),
        "seat {seat:?} holding {hand:?}, trick {:?}",
        m.public().trick_cards
    );
    m.apply(action).expect("the chosen action is legal");
}

fn expect_play(m: &mut Match, card: &str, rule: Rule) {
    expect(m, Action::Play(c(card)), rule);
}

fn expect_burn(m: &mut Match, card: &str) {
    expect(m, Action::Burn(c(card)), Rule::Burn);
}

fn tracker(m: &Match, seat: u8) -> Tracker {
    Tracker::from_view(&m.view(seat))
}

// ===== rule 1: the burn =====

#[test]
fn burn_robber_never_burns_the_taken_trump() {
    // Spades are trumps; seat 0 holds the ace and robs the 10S (the lowest trump) at its first
    // turn. Its other burnable cards are trumps too (KS, 4S): the lowest of them goes, never the
    // card it took.
    let mut m = one_hand(7, ["AS KS 4S AD AC", "", "", "", "", "", "", ""], "10S");
    assert_eq!(m.phase(), Phase::AwaitBurn(0));
    assert_eq!(names(m.truth().hands[0]), names(set("AS KS 4S AD AC 10S")));
    expect_burn(&mut m, "4S");
}

#[test]
fn burn_plain_rank_first_then_the_shorter_suit() {
    // A turned ace of clubs: the dealer robs at once. 2D and 9S share the lowest plain rank (2):
    // the dealer holds one spade but three diamonds, so the 9S goes.
    let mut m = one_hand(7, ["", "", "", "", "", "", "", "2D 3D 4D 9S 5C"], "AC");
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    expect_burn(&mut m, "9S");
}

#[test]
fn burn_after_a_turned_joker_lowest_rank_beats_a_short_suit_and_keeps_the_joker() {
    // The Joker is turned: the dealer takes it and burns before trumps are known. The 10S has the
    // lowest plain rank (black 10), so it goes although spades is the longest suit; the Joker,
    // a trump in every hand, is never burned.
    let mut m = one_hand(7, ["", "", "", "", "", "", "", "10S 9S 8S 2D KC"], "JK 7H");
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    assert_eq!(m.public().trump, None);
    expect_burn(&mut m, "10S");
    assert_eq!(m.public().trump, Some(engine::cards::Suit::Hearts));
}

#[test]
fn burn_twice_after_a_turned_joker_then_a_turned_ace() {
    // Joker turned, then the ace of diamonds: the dealer burns twice. First the lowest plain card
    // (3D, rank 3, against 2C, black 2 = rank 9); then only the 2C and the Joker may go, and the
    // Joker the dealer took is kept.
    let mut m = one_hand(7, ["", "", "", "", "", "", "", "AC AH AS 3D 2C"], "JK AD");
    expect_burn(&mut m, "3D");
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    expect_burn(&mut m, "2C");
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
}

#[test]
fn burn_a_taken_trump_only_when_nothing_else_may_go() {
    // The dealer holds all four aces. After the Joker it burns the 2C; then it must rob the 7S
    // (it holds the ace of spades) and may burn only the Joker or the 7S, both taken trumps: the
    // lower trump goes.
    let mut m = one_hand(7, ["", "", "", "", "", "", "", "AC AD AH AS 2C"], "JK 7S");
    expect_burn(&mut m, "2C");
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    expect_burn(&mut m, "7S");
}

// ===== rule 2: clinch defence =====

/// Hand 1 (dealer 7, spades): team B (seats 1 and 5) takes four tricks with the 5, J, Joker and
/// A♥ of trumps (5 + 6 + 5 + 5 + 5 = 26 points, its first bonus taken); team A takes trick 5.
fn clinch_hand_one() -> [Card; DECK_SIZE] {
    deck(
        7,
        [
            "2C 3C 4C 5C 6C",
            "5S JS JK AH 7C",
            "2D 3D 4D 5D 6D",
            "2H 3H 4H 5H 6H",
            "8C 9C 10C JC QC",
            "7D 8D 9D 10D JD",
            "7H 8H 9H 10H JH",
            "KC QD KD QH KH",
        ],
        "9S",
    )
}

/// Hand 2 (dealer 0, spades): seat 1 leads a club and seat 5 (team B) trumps it with the 5:
/// 5 points plus team B's later bonus of 5. Seat 5 then leads the KD and seat 6 (team C) holds
/// the Joker and three low diamonds.
fn clinch_hand_two() -> [Card; DECK_SIZE] {
    deck(
        0,
        [
            "QC KC AC JD AD",
            "2C 7H 8H 9H 10H",
            "3C 2H 3H 4H 5H",
            "4C 6H JH QH KH",
            "5C 2D 4D 5D 6D",
            "5S KD QD 7D 8D",
            "6C JK 3D 9D 10D",
            "7C 8C 9C 10C JC",
        ],
        "9S",
    )
}

#[test]
fn clinch_defence_spends_a_top_trump_to_stop_41() {
    let mut m = Match::with_decks(
        7,
        vec![clinch_hand_one(), clinch_hand_two()],
        1,
        Rules::HOUSE,
    );
    play(&mut m, "2C 5S 2D 2H 8C 7D 7H KC");
    play(&mut m, "JS 3D 3H 9C 8D 8H QD 3C");
    play(&mut m, "JK 4D 4H 10C 9D 9H KD 4C");
    play(&mut m, "AH 5D 5H JC 10D 10H QH 5C");
    play(&mut m, "7C 6D 6H QC JD JH KH 6C");
    assert_eq!(m.scores(), [5, 26, 0, 0]);
    assert_eq!(m.bonus_taken(), [false, true, false, false]);
    assert_eq!(m.dealer(), 0);

    // Hand 2, trick 1: seat 6 to play after seat 5's 5 of trumps. Winning this trick would give
    // team B 26 + 5 + 5 (its later bonus) = 36 and team C 0 + 5 + 6 = 11: no threat yet.
    play(&mut m, "2C 3C 4C 5C 5S");
    let v = m.view(6);
    assert_eq!(score_if_team_wins_trick(&v, &Rules::HOUSE, 1), 36);
    assert_eq!(score_if_team_wins_trick(&v, &Rules::HOUSE, 2), 11);
    play(&mut m, "6C 7C QC");
    assert_eq!(m.scores(), [5, 36, 0, 0]);

    // Trick 2: seat 5 leads the KD. Team B on 36 reaches 41 with this trick, so seat 6 spends
    // the Joker (a top trump, normally kept until trick 4) to take it.
    play(&mut m, "KD");
    assert_eq!(score_if_team_wins_trick(&m.view(6), &Rules::HOUSE, 1), 41);
    expect_play(&mut m, "JK", Rule::ClinchDefence);
}

#[test]
fn without_a_threat_the_same_trick_keeps_the_top_trump() {
    // The same hand as the first of a match: team B is on 11 after trick 1, so seat 6 keeps the
    // Joker and plays its lowest diamond (rule 5 spares top trumps before trick 4, rule 6).
    let mut m = Match::with_decks(0, vec![clinch_hand_two()], 1, Rules::HOUSE);
    play(&mut m, "2C 3C 4C 5C 5S 6C 7C QC");
    assert_eq!(m.scores(), [0, 11, 0, 0]);
    play(&mut m, "KD");
    expect_play(&mut m, "3D", Rule::PlayLow);
}

// ===== rule 3: the partner is winning =====

#[test]
fn partner_winning_and_last_to_play_does_not_overtake() {
    // Seat 3 (seat 7's partner) wins with the KD; seat 7 plays last, so nobody can beat it.
    // Rule 5 would trump with the 10S; rule 3 plays the lowest diamond instead.
    let mut m = one_hand(
        7,
        [
            "2D 2S",
            "3D",
            "4D",
            "KD",
            "5D",
            "6D",
            "7D",
            "QD 8D 10S 2H 3H",
        ],
        "9S",
    );
    play(&mut m, "2D 3D 4D KD 5D 6D 7D");
    expect_play(&mut m, "8D", Rule::PartnerWinning);
}

#[test]
fn partner_winning_but_beatable_is_covered_cheaply() {
    // Seat 0 leads the KD; seat 4 (its partner) plays fourth and three seats follow, any of
    // which could hold a trump. Rule 3 does not apply, and rule 5 covers with the lowest trump.
    let mut m = one_hand(
        7,
        ["KD 2S", "2D", "3D", "4D", "8D 3S 10S 2H 3H", "", "", ""],
        "9S",
    );
    play(&mut m, "KD 2D 3D 4D");
    expect_play(&mut m, "10S", Rule::WinCheaply);
}

/// Spades are trumps and the 5S is turned; the ace of spades is undealt, so the 5S stays on the
/// table. In trick 1 seat 0 leads the JS, which forces out the Joker (seat 1) and, when `ah` is
/// set, the A♥ (seat 2); seat 4 must follow with a trump (9S). In trick 2 seat 0 leads the QS
/// and seat 4, its partner, must play the KS or the 10S.
fn safe_partner_position(ah: bool) -> Match {
    let seat2 = if ah {
        "AH 9C 10C JC QC"
    } else {
        "9C 10C JC QC KC"
    };
    let mut m = one_hand(
        7,
        [
            "JS QS 2C 3C 4C",
            "JK 5C 6C 7C 8C",
            seat2,
            "",
            "KS 10S 9S 2D 3D",
            "",
            "",
            "",
        ],
        "5S",
    );
    play(&mut m, "JS JK");
    play(&mut m, if ah { "AH" } else { "-" });
    play(&mut m, "- 9S - - -");
    assert!(m.public().ace_undealt);
    assert_eq!(m.public().table_card, Some(c("5S")));
    play(&mut m, "QS - - -");
    assert_eq!(m.to_act(), Some(4));
    assert_eq!(
        m.legal_actions(),
        vec![Action::Play(c("10S")), Action::Play(c("KS"))]
    );
    m
}

#[test]
fn partner_winning_safely_by_the_public_facts() {
    // Only the KS (mine), the ace of trumps (undealt), the 5S (on the table for good), and the
    // JS, Joker and A♥ (played) beat the QS: nobody after seat 4 can, so it keeps the KS.
    let mut m = safe_partner_position(true);
    let t = tracker(&m, 4);
    let out = t.known_out();
    for n in ["5S", "AS", "JS", "JK", "AH"] {
        assert!(has(out, c(n)), "{n} must be known out of play");
    }
    expect_play(&mut m, "10S", Rule::PartnerWinning);
}

#[test]
fn partner_beatable_when_the_ace_of_hearts_is_unseen() {
    // The same, but the A♥ was never played: a seat after seat 4 could hold it, so the partner's
    // QS is not safe and seat 4 covers with the KS (not a top trump).
    let mut m = safe_partner_position(false);
    expect_play(&mut m, "KS", Rule::WinCheaply);
}

// ===== rule 4: leading =====

#[test]
fn lead_low_then_the_best_trump_once_the_five_and_jack_are_gone() {
    // The 5S is turned and stays on the table (the ace of spades is undealt). In trick 1 a
    // robber could still take it, and seat 0 holds the J itself, so seat 0 leads low from its
    // longest plain suit. After it wins trick 1 with the J, both are gone and it holds the best
    // remaining trump, the Joker: it leads it.
    let mut m = one_hand(7, ["JS JK 2D 3D 4D", "", "", "", "", "", "", ""], "5S");
    assert!(!has(tracker(&m, 0).known_out(), c("5S")));
    assert_eq!(ask(&m), (Action::Play(c("2D")), Rule::LeadLow));
    play(&mut m, "JS - - - - - - -");
    assert_eq!(m.to_act(), Some(0));
    expect_play(&mut m, "JK", Rule::LeadBestTrump);
}

#[test]
fn lead_low_from_the_longest_plain_suit() {
    // Three diamonds: lead the lowest, the ace (the lowest red card), although the 10C is just as
    // low.
    let mut m = one_hand(7, ["AD 2D KD 10C KH", "2S", "3S", "", "", "", "", ""], "9S");
    expect_play(&mut m, "AD", Rule::LeadLow);
}

#[test]
fn lead_low_equal_length_goes_to_the_suit_with_the_lower_card() {
    // Two diamonds and two hearts: hearts hold the lower card (2H against 3D), so the 2H is led
    // although diamonds come first in suit order.
    let mut m = one_hand(7, ["3D 9D 2H 7H KC", "2S", "3S", "", "", "", "", ""], "9S");
    expect_play(&mut m, "2H", Rule::LeadLow);
}

#[test]
fn lead_low_with_only_trumps_leads_the_lowest_trump() {
    // Black trumps run 2 up to 10, so the 6S is the lowest of 2S 3S 4S 6S KS.
    let mut m = one_hand(7, ["2S 3S 4S 6S KS", "", "", "", "", "", "", ""], "9S");
    expect_play(&mut m, "6S", Rule::LeadLow);
}

#[test]
fn lead_the_strongest_card_on_tricks_four_and_five() {
    // Hearts are trumps and nobody else holds one. Seat 0 wins tricks 1–3 with its kings, then
    // leads its strongest card on trick 4 (the 2H, a trump, over the QC) and its last on trick 5.
    let mut m = one_hand(7, ["KC KD KS 2H QC", "", "", "", "", "", "", ""], "9H");
    play(&mut m, "KC - - - - - - -");
    play(&mut m, "KD - - - - - - -");
    play(&mut m, "KS - - - - - - -");
    assert_eq!(m.public().trick_no, 4);
    expect_play(&mut m, "2H", Rule::LeadStrongest);
    play(&mut m, "- - - - - - -");
    assert_eq!(m.public().trick_no, 5);
    expect_play(&mut m, "QC", Rule::LeadStrongest);
}

// ===== rule 5: win cheaply =====

#[test]
fn win_cheaply_with_a_plain_card_before_a_trump() {
    // The 9D is led; seat 1 can win with the QD or the 10S: the plain card is cheaper.
    let mut m = one_hand(7, ["9D", "QD 3D 10S KC 2H", "2S", "", "", "", "", ""], "9S");
    play(&mut m, "9D");
    expect_play(&mut m, "QD", Rule::WinCheaply);
}

#[test]
fn a_top_trump_wins_when_every_legal_card_is_one() {
    // The Joker is led: seat 1 must play a trump (its A♥ cannot be held back), so its only legal
    // cards are the 5S and the A♥, both top trumps. Rule 5 may then use the 5S on trick 1.
    let mut m = one_hand(7, ["JK", "5S AH 2D 3D 4D", "", "", "", "", "", ""], "9S");
    play(&mut m, "JK");
    assert_eq!(
        m.legal_actions(),
        vec![Action::Play(c("AH")), Action::Play(c("5S"))]
    );
    expect_play(&mut m, "5S", Rule::WinCheaply);
}

#[test]
fn top_trumps_are_spared_until_trick_four() {
    // Hearts are trumps. Seat 0 leads the KC, KD and KS and wins; seat 1 could take each with
    // the Joker but plays low on tricks 1–3, then takes trick 4 (the QD led) with it.
    let mut m = one_hand(
        7,
        ["KC KD KS QD 3C", "JK 3D 4D 2C 4C", "", "", "", "", "", ""],
        "9H",
    );
    play(&mut m, "KC");
    expect_play(&mut m, "4C", Rule::PlayLow); // 4C is lower than 2C in a black suit
    play(&mut m, "- - - - - -");
    play(&mut m, "KD");
    expect_play(&mut m, "3D", Rule::PlayLow);
    play(&mut m, "- - - - - -");
    play(&mut m, "KS");
    expect_play(&mut m, "4D", Rule::PlayLow);
    play(&mut m, "- - - - - -");
    play(&mut m, "QD");
    assert_eq!(m.public().trick_no, 4);
    expect_play(&mut m, "JK", Rule::WinCheaply);
}

// ===== rule 6: play low =====

#[test]
fn play_low_when_the_trick_cannot_be_won() {
    // The 9D is led and trumped with the KS. Seat 2 cannot beat it and plays its lowest legal
    // card, a diamond; seat 3, void in diamonds, throws its lowest card (3H, plain rank 3).
    let mut m = one_hand(
        7,
        [
            "9D",
            "KS 2C 3C 4C 5C",
            "3D QD 10S 2H KH",
            "4S 6C QH 3H 4H",
            "",
            "",
            "",
            "",
        ],
        "9S",
    );
    play(&mut m, "9D KS");
    expect_play(&mut m, "3D", Rule::PlayLow);
    expect_play(&mut m, "3H", Rule::PlayLow);
}

#[test]
fn play_low_keeps_the_saver() {
    // The Joker is led; seat 1 must play a trump and cannot win, so it plays its lowest trump
    // (10S) and keeps its highest (KS) for later.
    let mut m = one_hand(7, ["JK", "KS QS 10S 2D 3D", "", "", "", "", "", ""], "9S");
    play(&mut m, "JK");
    expect_play(&mut m, "10S", Rule::PlayLow);
}

// ===== the tracker =====

#[test]
fn tracker_follows_the_robbing_facts() {
    // Spades are trumps (9S turned) and seat 3 holds the ace of spades: it robs at its turn.
    let mut m = one_hand(7, ["", "", "", "AS 2D 3D 4D 5D", "", "", "", ""], "9S");
    let t = tracker(&m, 0);
    assert!(!has(t.known_out(), c("9S")), "a robber may still take it");
    assert!((1..8).all(|s| t.could_hold(s, c("9S")) && t.could_hold_ace(s)));
    play(&mut m, "- -");
    let t = tracker(&m, 2);
    assert!(!t.could_hold(0, c("9S")) && !t.could_hold_ace(1));
    assert!(t.could_hold(3, c("9S")) && t.could_hold_ace(3));
    play(&mut m, "-");
    assert_eq!(m.phase(), Phase::AwaitBurn(3));
    let t = tracker(&m, 3);
    assert_eq!(t.my_taken_trumps(), bit(c("9S")));
    expect_burn(&mut m, "2D");
    play(&mut m, "-");
    let t = tracker(&m, 4);
    assert_eq!(t.robber, Some(3));
    assert!(t.could_hold_ace(3) && !t.could_hold_ace(5));
    assert!(t.could_hold(3, c("9S")) && !t.could_hold(5, c("9S")));
}

#[test]
fn tracker_deduces_the_undealt_ace_before_it_is_announced() {
    // Nobody holds the ace of spades. When seat 7 plays last to trick 1, the other seven have all
    // played without robbing, so seat 7 knows the ace is undealt and the 9S stays on the table,
    // before the engine announces it at the end of the trick.
    let mut m = one_hand(7, ["2S", "3S", "", "", "", "", "", ""], "9S");
    play(&mut m, "- - - - - - -");
    assert!(!m.public().ace_undealt);
    let out = tracker(&m, 7).known_out();
    assert!(has(out, c("AS")) && has(out, c("9S")));
    let out = tracker(&m, 6).known_out();
    assert!(
        !has(out, c("AS")) && !has(out, c("9S")),
        "seat 6 only knows seat 7 has not robbed yet"
    );
}
