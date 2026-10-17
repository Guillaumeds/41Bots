//! Turned partnerships (rules 34 and 37–41; the owner's decisions of 2026-10-17 in
//! docs/APP-ENGINE-CONTRACT.md §7): the app turns a pair (`Match::turn_team`), turns one right
//! after a card (`Match::play_then_turn`), or lets a renege through (`Match::let_through`) and the
//! engine catches it when the app would; the hand goes on exactly as the app plays it. The
//! engine's own players never act illegally (H1): none of this is among `legal_actions`.
//!
//! One test per rule (named after it), one per ruling of 2026-10-17 (Q1 to Q3, "play, then
//! turn", a turn while a rob is due), the catches, and properties over random hands with random
//! turns and lapses: records replay exactly, every seat's view and the hand-start state stay
//! right, and a refusal changes nothing.

use engine::cards::{Bits, Card, DECK_SIZE, bit, parse};
use engine::game::{
    Action, Event, Extra, GameEndReason, HandStartState, Lapse, LapseRefusal, Match, Phase,
    RenegeKind, Rules, TurnCause, TurnRefusal, build_deck, current_hand_start_of,
};
use engine::rng::Rng;
use engine::{SEATS, team_of};

// ===== helpers =====

fn c(n: &str) -> Card {
    parse(n).unwrap_or_else(|| panic!("bad card name {n}"))
}

fn set(s: &str) -> Bits {
    s.split_whitespace().fold(0, |a, n| a | bit(c(n)))
}

fn deck(dealer: u8, hands: [&str; 8], pack: &str) -> [Card; DECK_SIZE] {
    let hb: [Bits; SEATS] = core::array::from_fn(|s| set(hands[s]));
    let pack: Vec<Card> = pack.split_whitespace().map(c).collect();
    build_deck(dealer, &hb, &pack).expect("a valid hand-made deal")
}

fn play(m: &mut Match, n: &str) {
    let seat = m.to_act();
    if let Err(e) = m.apply(Action::Play(c(n))) {
        panic!("{e} (seat {seat:?}, phase {:?})", m.phase());
    }
}

fn plays(m: &mut Match, cards: &str) {
    for n in cards.split_whitespace() {
        play(m, n);
    }
}

fn turn(m: &mut Match, team: u8) {
    if let Err(e) = m.turn_team(team) {
        panic!("{e}");
    }
}

fn log_from(m: &Match, from: usize) -> Vec<String> {
    m.events()[from..]
        .iter()
        .filter(|(v, _)| *v == engine::game::Visibility::Public)
        .map(|(_, e)| e.canonical())
        .collect()
}

/// The hand of the rule tests, spades trumps, the ace of trumps undealt (nobody robs), dealer
/// 7 so seat 0 leads. Trick 1: seat 0's K♣ wins a round of clubs (the strongest club of the
/// others is seat 2's 2♣). Trick 2: seat 0 leads the 5♠ and wins it with its bonus; seat 3 must
/// play the 9♠ and seat 6 the K♠. Tricks 3–5: seat 1 leads K♥, Q♥ and K♦ and everyone follows.
const HANDS: [&str; 8] = [
    "KC 5S AD 2D 3D",
    "4C QC KH QH KD",
    "2C JC JH 10H QD",
    "5C 9S 9H 8H JD",
    "3C AC 4D 5D 6D",
    "6C 9C 7H 6H 10D",
    "7C KS 5H 4H 9D",
    "8C 10C 3H 2H 8D",
];
const TRICK1: &str = "KC 4C 2C 5C 3C 6C 7C 8C";
const TRICK2: &str = "5S QC JC 9S AC 9C KS 10C";

fn rule_hand() -> Match {
    Match::with_decks(7, vec![deck(7, HANDS, "7S AS")], 1, Rules::HOUSE)
}

/// The rule hand with a chosen pack (the turned card first).
fn rule_hand_pack(pack: &str) -> Match {
    Match::with_decks(7, vec![deck(7, HANDS, pack)], 1, Rules::HOUSE)
}

/// The rule hand with seat 5 holding the ace of trumps (A♠ for its 9♣): at its turn in trick 1
/// the engine takes the turned 7♠ for it and waits for its burn.
const ROB_HANDS: [&str; 8] = [
    "KC 5S AD 2D 3D",
    "4C QC KH QH KD",
    "2C JC JH 10H QD",
    "5C 9S 9H 8H JD",
    "3C AC 4D 5D 6D",
    "6C AS 7H 6H 10D",
    "7C KS 5H 4H 9D",
    "8C 10C 3H 2H 8D",
];

fn rob_hand() -> Match {
    Match::with_decks(7, vec![deck(7, ROB_HANDS, "7S 9C")], 1, Rules::HOUSE)
}

fn lapse(m: &mut Match, l: Lapse) {
    if let Err(e) = m.let_through(l) {
        panic!("{e}");
    }
}

fn play_turn(m: &mut Match, card: &str, team: u8) {
    if let Err(e) = m.play_then_turn(c(card), team) {
        panic!("{e}");
    }
}

/// The rule hand from a chosen start of hand (scores, games, bonus flags).
fn rule_hand_from(scores: [u8; 4], hands: [&str; 8]) -> Match {
    let s = HandStartState {
        dealer: 7,
        scores,
        bonus_taken: scores.map(|x| x % 5 == 1),
        games_won: [0; 4],
        hand_index: 3,
        game_index: 0,
    };
    Match::from_hand_start(s, deck(7, hands, "7S AS"), 1, Rules::HOUSE).expect("valid state")
}

// ===== one test per rule =====

/// Rule 41: "The turned partnership scores nothing for that hand."
///
/// Team 0 takes trick 1 (5) and trick 2 with the 5 of trumps (5 + 6): 16 points and its first
/// bonus. Turned before trick 3, it goes back to its score at the start of the hand (0), its
/// bonus flag with it (its 11 is not spent), and it scores nothing more: no trick, no bonus.
#[test]
fn rule_41_turned_partnership_scores_nothing_for_the_hand() {
    let mut m = rule_hand();
    plays(&mut m, TRICK1);
    plays(&mut m, TRICK2);
    assert_eq!(m.scores(), [16, 0, 0, 0]);
    assert_eq!(m.bonus_taken(), [true, false, false, false]);
    turn(&mut m, 0);
    assert_eq!(m.scores()[0], 0);
    assert_eq!(m.bonus_taken(), [false; 4]);
    assert_eq!(m.public().tricks_won[0], 0);
    assert_eq!(m.turned_teams(), [true, false, false, false]);
    // The rest of the hand: team 0 never plays and never scores.
    plays(&mut m, "KH JH 9H 7H 5H 3H");
    plays(&mut m, "QH 10H 8H 6H 4H 2H");
    plays(&mut m, "KD QD JD 10D 9D 8D");
    assert_eq!(m.hand_index(), 1, "the hand is over");
    assert_eq!(m.scores()[0], 0);
    assert_eq!(m.scores(), [0, 15, 16, 0]);
}

/// Rule 39: "Once a partnership is turned, none of its cards counts any more. Every trick it had
/// won in the hand passes, with its points, to the partnership still in play that played the
/// strongest card in that trick. If the highest trump of the hand was one of the turned cards,
/// the bonus goes to the highest trump played in the hand by a partnership that is not turned,
/// and counts for that partnership as rule 28 says."
///
/// Trick 1 (led K♣ by seat 0) passes to seat 2's 2♣, the highest club left; trick 2 (led 5♠ by
/// seat 0) to seat 6's K♠, the highest trump left: team 2 takes both, 10 points. The 5♠ no
/// longer counts, so the bonus waits for the end of the hand (rule 30) and goes to the K♠:
/// team 2's first bonus, 6 (its 11).
#[test]
fn rule_39_turned_tricks_pass_and_the_bonus_follows_the_best_trump_in_play() {
    let mut m = rule_hand();
    plays(&mut m, TRICK1);
    plays(&mut m, TRICK2);
    let from = m.events().len();
    turn(&mut m, 0);
    assert_eq!(
        log_from(&m, from),
        [
            "PairTurned team=0 score=0 cause=before_play",
            "TrickPassed trick=1 team=2 points=5",
            "TrickPassed trick=2 team=2 points=5",
        ]
    );
    let p = m.public();
    assert_eq!(p.scores, [0, 0, 10, 0]);
    assert_eq!(p.tricks_won, [0, 0, 2, 0]);
    assert_eq!(
        p.trick_winners,
        [0, 0],
        "who won each trick at the table is kept"
    );
    assert_eq!(
        p.best_trump,
        Some((c("KS"), 6)),
        "the best trump still in play"
    );
    assert!(!p.five_played, "the turned 5 of trumps no longer counts");
    plays(&mut m, "KH JH 9H 7H 5H 3H");
    plays(&mut m, "QH 10H 8H 6H 4H 2H");
    let from = m.events().len();
    plays(&mut m, "KD QD JD 10D 9D 8D");
    assert_eq!(
        log_from(&m, from)[..10],
        [
            "Played seat=1 card=KD",
            "Played seat=2 card=QD",
            "Played seat=3 card=JD",
            "Played seat=5 card=10D",
            "Played seat=6 card=9D",
            "Played seat=7 card=8D",
            "TrickWon trick=5 seat=1 team=1 points=5",
            "Bonus team=2 points=6 kind=end card=KS",
            "HandEnd hand=0 aborted=false",
            "HandStart hand=1 game=0 dealer=0",
        ]
    );
    assert_eq!(m.scores(), [0, 15, 16, 0]);
    // The same hand with team 2's 11 already spent: the bonus counts 5 for it (its 10).
    let mut m = rule_hand_from([0, 0, 11, 0], HANDS);
    plays(&mut m, TRICK1);
    plays(&mut m, TRICK2);
    turn(&mut m, 0);
    plays(
        &mut m,
        "KH JH 9H 7H 5H 3H QH 10H 8H 6H 4H 2H KD QD JD 10D 9D 8D",
    );
    assert_eq!(m.scores(), [0, 15, 11 + 10 + 5, 0]);
}

/// Rule 40: "When a partnership is turned in the middle of a trick, the trick goes on among the
/// other players without the turned cards. If the player who was to lead is turned, the lead
/// passes to the next player on the left who is still in."
///
/// Mid-trick: team 1 is turned after seats 0, 1 and 2 have played to trick 1; seat 1's 4♣ leaves
/// the trick, seat 5 never plays, and the trick is won among the six others. Before a lead:
/// team 0 (seat 0 to lead trick 3) is turned; the lead passes to seat 1.
#[test]
fn rule_40_turned_cards_leave_the_trick_and_the_lead_passes_on() {
    let mut m = rule_hand();
    plays(&mut m, "KC 4C 2C");
    turn(&mut m, 1);
    let p = m.public();
    assert_eq!(p.trick_cards, [(0, c("KC")), (2, c("2C"))]);
    assert_eq!(m.phase(), Phase::AwaitPlay(3));
    plays(&mut m, "5C 3C");
    assert_eq!(m.phase(), Phase::AwaitPlay(6), "seat 5 is skipped");
    let from = m.events().len();
    plays(&mut m, "7C 8C");
    assert_eq!(
        log_from(&m, from),
        [
            "ProvedNoAce seat=6",
            "Played seat=6 card=7C",
            "ProvedNoAce seat=7",
            "Played seat=7 card=8C",
            "TrickWon trick=1 seat=0 team=0 points=5",
        ],
        "no AceUndealt: seat 5 never showed it lacks the ace"
    );
    assert!(!m.public().ace_undealt);
    // Trick 2 among six: the 5 of trumps wins with its bonus.
    plays(&mut m, "5S JC 9S AC KS 10C");
    assert_eq!(m.scores(), [16, 0, 0, 0]);
    // The seat to lead (seat 0) is turned: the lead passes to seat 1, then (seat 1 turned too)
    // to seat 2.
    let mut n = rule_hand();
    plays(&mut n, TRICK1);
    plays(&mut n, TRICK2);
    turn(&mut n, 0);
    assert_eq!(n.phase(), Phase::AwaitPlay(1));
    assert_eq!(n.public().leader, 1);
    turn(&mut n, 1);
    assert_eq!(n.phase(), Phase::AwaitPlay(2));
    assert_eq!(n.public().leader, 2);
    // Four players left: seats 2, 3, 6, 7.
    plays(&mut n, "JH 9H 5H 3H");
    assert_eq!(n.public().trick_winners.len(), 3);
    assert_eq!(n.public().trick_winners[2], 2);
}

/// Rule 34: "If every other partnership has been turned, the partnership still in play takes
/// every trick of the hand, the turned partnerships' tricks included under rule 39, and that is
/// a jink: it wins the game."
///
/// As the app plays it: the last partnership in play plays the hand out alone, takes every trick
/// (tricks 1 and 2 passed to it, tricks 3 to 5 won), takes the bonus (the K♠, its 11) and wins
/// the game with a jink at the end of the hand.
#[test]
fn rule_34_the_last_partnership_in_play_takes_every_trick_and_wins_the_game() {
    let mut m = rule_hand();
    plays(&mut m, TRICK1);
    plays(&mut m, TRICK2);
    let from = m.events().len();
    turn(&mut m, 0);
    turn(&mut m, 1);
    turn(&mut m, 3);
    assert_eq!(
        log_from(&m, from),
        [
            "PairTurned team=0 score=0 cause=before_play",
            "TrickPassed trick=1 team=2 points=5",
            "TrickPassed trick=2 team=2 points=5",
            "PairTurned team=1 score=0 cause=before_play",
            "PairTurned team=3 score=0 cause=before_play",
        ]
    );
    assert_eq!(
        m.phase(),
        Phase::AwaitPlay(2),
        "seat 2 leads, alone with seat 6"
    );
    plays(&mut m, "JH 5H 10H 4H QD");
    let from = m.events().len();
    play(&mut m, "9D");
    assert_eq!(
        log_from(&m, from)[..5],
        [
            "Played seat=6 card=9D",
            "TrickWon trick=5 seat=2 team=2 points=5",
            "Bonus team=2 points=6 kind=end card=KS",
            "GameWon team=2 reason=jink scores=0,0,31,0",
            "HandEnd hand=0 aborted=false",
        ]
    );
    assert_eq!(m.games_won(), [0, 0, 1, 0]);
    assert_eq!(m.scores(), [0; 4], "a new game");
    // Team 1 wins trick 3 among six and is turned: trick 3 passes (rule 39) to seat 2's J♥, the
    // highest heart left; then team 3 is turned and team 2 plays tricks 4 and 5 alone.
    let mut m = rule_hand();
    plays(&mut m, TRICK1);
    plays(&mut m, TRICK2);
    turn(&mut m, 0);
    plays(&mut m, "KH JH 9H 7H 5H 3H");
    let from = m.events().len();
    turn(&mut m, 1);
    turn(&mut m, 3);
    assert_eq!(
        log_from(&m, from),
        [
            "PairTurned team=1 score=0 cause=before_play",
            "TrickPassed trick=3 team=2 points=5",
            "PairTurned team=3 score=0 cause=before_play",
        ]
    );
    plays(&mut m, "10H 4H QD 9D");
    assert_eq!(m.games_won(), [0, 0, 1, 0]);
}

// ===== the owner's rulings of 2026-10-17 =====

/// Q1, "ask the pros first": a partnership turned while the trick it led is still in progress
/// is refused (the app keeps the turned card as the card led), whatever turns it: a turn, a play
/// then a turn, or a card that proves its own renege. The match is unchanged.
#[test]
fn q1_leader_turned_mid_trick_is_refused_as_an_open_question() {
    let mut m = rule_hand();
    plays(&mut m, "KC 4C");
    let before = m.clone();
    let e = m.turn_team(0).expect_err("refused");
    assert_eq!(e.why, TurnRefusal::LedTrickInProgress);
    assert_eq!(m, before);
    // Its partner's seat (4) is not the leader, but its card would be the trick's lead card all
    // the same: still refused.
    plays(&mut m, "2C 5C");
    assert_eq!(
        m.turn_team(0).map_err(|e| e.why),
        Err(TurnRefusal::LedTrickInProgress)
    );
    // Even when the card that turns it is the trick's last: against which card is it won?
    plays(&mut m, "3C 6C 7C");
    let before = m.clone();
    assert_eq!(
        m.play_then_turn(c("8C"), 0).map_err(|e| e.why),
        Err(TurnRefusal::LedTrickInProgress)
    );
    assert_eq!(m, before);
    // Another partnership mid-trick is fine.
    assert!(m.turn_team(3).is_ok());
    // A legal card that proves a renege of the partnership that led the trick: the same
    // question, refused. Seat 3 holds the J♦ and plays the 9♥ to seat 0's 2♦ in trick 3 (let
    // through); teams 1 and 2 are turned, so seat 3 leads trick 4, and its J♦ lead proves the
    // renege while seats 4 and 0 still have to play.
    let mut m = rule_hand();
    plays(&mut m, TRICK1);
    plays(&mut m, TRICK2);
    plays(&mut m, "2D KD QD");
    lapse(&mut m, Lapse::Play(c("9H")));
    plays(&mut m, "4D 10D 9D 8D");
    turn(&mut m, 1);
    turn(&mut m, 2);
    assert_eq!(m.to_act(), Some(3));
    let before = m.clone();
    let e = m.apply(Action::Play(c("JD"))).expect_err("refused");
    assert_eq!(e.reason, TurnRefusal::LedTrickInProgress.reason());
    assert_eq!(m, before);
}

/// Q2, "first card played takes it" (as the app's `standingWinnerIndex`): a trick to pass in
/// which no card of a partnership still in play is a trump or of the suit led goes to the first
/// of their cards played. Seat 0 leads the K♣ and only its partner follows; turned, its trick
/// goes to seat 1's 2♦ (the first card of the others), then, seat 1 turned, to seat 2's 6♦.
#[test]
fn q2_a_trick_with_no_card_that_can_win_goes_to_the_first_card_played() {
    let hands = [
        "KC QC JC AC 2C",
        "2D 3D 4D 5D 2S",
        "6D 7D 8D 9D 3S",
        "10D JD QD KD 4S",
        "3C 4C 5C 6C 7C",
        "AD 2H 3H 4H 6S",
        "5H 6H 7H 8H 8S",
        "9H 10H JH QH KH",
    ];
    let mut m = Match::with_decks(7, vec![deck(7, hands, "7S AS")], 1, Rules::HOUSE);
    plays(&mut m, "KC 2D 6D 10D 3C AD 5H 9H");
    assert_eq!(m.public().trick_winners, [0]);
    let from = m.events().len();
    turn(&mut m, 0);
    turn(&mut m, 1);
    assert_eq!(
        log_from(&m, from),
        [
            "PairTurned team=0 score=0 cause=before_play",
            "TrickPassed trick=1 team=1 points=5",
            "PairTurned team=1 score=0 cause=before_play",
            "TrickPassed trick=1 team=2 points=5",
        ]
    );
    assert_eq!(m.scores(), [0, 0, 5, 0]);
    // Every partnership turned: nobody played a card that still counts, the trick stays with
    // nobody and the hand ends with nobody scoring.
    turn(&mut m, 2);
    let from = m.events().len();
    turn(&mut m, 3);
    assert_eq!(
        log_from(&m, from)[..3],
        [
            "PairTurned team=3 score=0 cause=before_play",
            "HandEnd hand=0 aborted=true",
            "HandStart hand=1 game=0 dealer=0",
        ]
    );
    assert_eq!(m.scores(), [0; 4]);
}

/// Q3, "same as the app": passed tricks are credited one at a time in trick order, and the game
/// ends at the first partnership to reach 41 (rule 32); later passes are not credited.
#[test]
fn q3_passed_tricks_are_credited_in_order_and_the_first_to_41_wins() {
    let mut m = rule_hand_from([10, 5, 36, 0], HANDS);
    plays(&mut m, TRICK1);
    plays(&mut m, TRICK2);
    let from = m.events().len();
    turn(&mut m, 0);
    assert_eq!(
        log_from(&m, from)[..4],
        [
            "PairTurned team=0 score=10 cause=before_play",
            "TrickPassed trick=1 team=2 points=5",
            "GameWon team=2 reason=passed_trick scores=10,5,41,0",
            "HandEnd hand=3 aborted=true",
        ]
    );
    assert_eq!(m.games_won(), [0, 0, 1, 0]);
    // Trick 2 passes to team 3 when seat 3 holds the K♠ and seat 6 the 9♠: both teams on 36
    // would reach 41, and team 2's trick 1 comes first.
    let mut swapped = HANDS;
    swapped[3] = "5C KS 9H 8H JD";
    swapped[6] = "7C 9S 5H 4H 9D";
    let mut m = rule_hand_from([10, 5, 36, 36], swapped);
    plays(&mut m, TRICK1);
    plays(&mut m, "5S QC JC KS AC 9C 9S 10C");
    let from = m.events().len();
    turn(&mut m, 0);
    assert_eq!(
        log_from(&m, from)[..3],
        [
            "PairTurned team=0 score=10 cause=before_play",
            "TrickPassed trick=1 team=2 points=5",
            "GameWon team=2 reason=passed_trick scores=10,5,41,36",
        ]
    );
    // Team 2 on 31 reaches only 36 with trick 1; team 3 then reaches 41 with trick 2.
    let mut m = rule_hand_from([10, 5, 31, 36], swapped);
    plays(&mut m, TRICK1);
    plays(&mut m, "5S QC JC KS AC 9C 9S 10C");
    let from = m.events().len();
    turn(&mut m, 0);
    assert_eq!(
        log_from(&m, from)[..4],
        [
            "PairTurned team=0 score=10 cause=before_play",
            "TrickPassed trick=1 team=2 points=5",
            "TrickPassed trick=2 team=3 points=5",
            "GameWon team=3 reason=passed_trick scores=10,5,36,41",
        ]
    );
}

/// "Play, then turn": the card is placed, the partnership is turned, then the trick goes on or
/// is resolved. Seat 7's 8♦ completes trick 3 and team 1 is turned with it: seat 1's K♦ and
/// seat 5's 10♦ no longer count, and seat 2's Q♦ wins the trick.
#[test]
fn play_then_turn_resolves_the_trick_without_the_turned_cards() {
    let mut m = rule_hand();
    plays(&mut m, TRICK1);
    plays(&mut m, TRICK2);
    plays(&mut m, "2D KD QD JD 4D 10D 9D");
    // An illegal card is refused (seat 7 holds the 8♦).
    let before = m.clone();
    assert_eq!(
        m.play_then_turn(c("3H"), 1).map_err(|e| e.why),
        Err(TurnRefusal::IllegalCard)
    );
    assert_eq!(m, before);
    let from = m.events().len();
    play_turn(&mut m, "8D", 1);
    assert_eq!(
        log_from(&m, from),
        [
            "Played seat=7 card=8D",
            "PairTurned team=1 score=0 cause=with_play",
            "TrickWon trick=3 seat=2 team=2 points=5",
        ]
    );
    assert_eq!(m.phase(), Phase::AwaitPlay(2));
    assert_eq!(m.extras(), [(23, Extra::PlayThenTurn(c("8D"), 1))]);
    // Mid-trick, the seat's own partnership: its card leaves the trick at once.
    let mut m = rule_hand();
    plays(&mut m, "KC 4C");
    play_turn(&mut m, "2C", 2);
    assert_eq!(m.public().trick_cards, [(0, c("KC")), (1, c("4C"))]);
    assert_eq!(m.phase(), Phase::AwaitPlay(3));
}

/// A turn while a rob is due (the app's Guide: the robber plays without robbing and is turned
/// before its card): the engine's automatic take is undone, the turned card stays on the table,
/// and the play goes on from the next seat in play.
#[test]
fn turn_while_a_rob_is_due_undoes_the_take_no_rob() {
    let mut m = rob_hand();
    plays(&mut m, "KC 4C 2C 5C");
    let from = m.events().len();
    play(&mut m, "3C");
    assert_eq!(m.phase(), Phase::AwaitBurn(5));
    turn(&mut m, 1);
    assert_eq!(
        log_from(&m, from),
        [
            "ProvedNoAce seat=4",
            "Played seat=4 card=3C",
            "PairTurned team=1 score=0 cause=no_rob"
        ]
    );
    assert!(
        !m.events()
            .iter()
            .any(|(_, e)| matches!(e, Event::Took { .. })),
        "the take never happened"
    );
    let p = m.public();
    assert_eq!(p.table_card, Some(c("7S")));
    assert_eq!(p.robber, None);
    assert_eq!(p.hand_sizes[5], 5);
    assert_eq!(m.phase(), Phase::AwaitPlay(6));
    plays(&mut m, "7C 8C");
    assert_eq!(m.public().trick_winners, [0]);
}

/// The dealer refuses a turned ace (the app's `declineRob`): its partnership is turned, the ace
/// stays on the table, out of play, and trick 1 starts as usual.
#[test]
fn turn_while_a_rob_is_due_declined_rob() {
    let mut m = rule_hand_pack("AS 7S");
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    turn(&mut m, 3);
    let l = log_from(&m, 0);
    assert_eq!(
        l[l.len() - 2..],
        [
            "Turned card=AS",
            "PairTurned team=3 score=0 cause=declined_rob"
        ]
    );
    let p = m.public();
    assert_eq!(p.table_card, Some(c("AS")));
    assert_eq!(p.robber, None);
    assert_eq!(p.hand_sizes[7], 5);
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    plays(&mut m, "KC 4C 2C 3C 6C 7C");
    assert_eq!(m.public().trick_winners, [0]);
    assert!(!m.public().ace_undealt);
}

/// The dealer leaves a turned Joker (the app's `leaveJoker`): its partnership is turned, the
/// Joker stays face up on the deck and the next card is turned; an ace turned next stays on the
/// pack unrobbed (the app's `afterTurn`).
#[test]
fn turn_while_a_rob_is_due_left_joker() {
    let mut m = rule_hand_pack("JK 7S AS");
    assert_eq!(m.phase(), Phase::AwaitBurn(7));
    turn(&mut m, 3);
    let l = log_from(&m, 0);
    assert_eq!(
        l[l.len() - 3..],
        [
            "Turned card=JK",
            "PairTurned team=3 score=0 cause=left_joker",
            "Turned card=7S"
        ]
    );
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    let p = m.public();
    assert_eq!(p.trump, Some(engine::cards::Suit::Spades));
    assert_eq!(p.turned, [(c("JK"), None), (c("7S"), None)]);
    assert_eq!(p.hand_sizes[7], 5);
    // An ace turned after the left Joker stays on the pack: the dealer's partnership is out.
    let mut m = rule_hand_pack("JK AS 7S");
    turn(&mut m, 3);
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    let p = m.public();
    assert_eq!(p.table_card, Some(c("AS")));
    assert_eq!(p.robber, None);
    assert_eq!(p.turned, [(c("JK"), None), (c("AS"), None)]);
    // Another partnership cannot be turned while the dealer's burn is due.
    let mut m = rule_hand_pack("JK 7S AS");
    assert_eq!(
        m.turn_team(0).map_err(|e| e.why),
        Err(TurnRefusal::NotBeforeAPlay)
    );
}

// ===== the referee: reneges let through and caught when the app catches them =====

/// Rule 37, a card proves the renege: seat 1 plays the K♥ to seat 0's club lead holding two
/// clubs (let through); its Q♣ in trick 2 proves it, and its partnership is turned with that
/// card, which leaves the trick.
#[test]
fn rule_37_a_later_card_proves_the_renege() {
    let mut m = rule_hand();
    play(&mut m, "KC");
    let before = m.clone();
    assert_eq!(
        m.let_through(Lapse::Play(c("4C"))).map_err(|e| e.why),
        Err(LapseRefusal::Legal)
    );
    assert_eq!(
        m.let_through(Lapse::Play(c("JH"))).map_err(|e| e.why),
        Err(LapseRefusal::NotInHand)
    );
    assert_eq!(m, before);
    lapse(&mut m, Lapse::Play(c("KH")));
    assert_eq!(m.reneges().len(), 1);
    assert_eq!(
        m.reneges()[0].kind,
        RenegeKind::Follow(engine::cards::Suit::Clubs)
    );
    plays(&mut m, "2C 5C 3C 6C 7C 8C");
    assert_eq!(m.turned_teams(), [false; 4], "nobody has seen it yet");
    play(&mut m, "5S");
    let from = m.events().len();
    play(&mut m, "QC");
    assert_eq!(
        log_from(&m, from),
        [
            "Played seat=1 card=QC",
            "PairTurned team=1 score=0 cause=proved"
        ]
    );
    assert!(m.reneges()[0].caught);
    assert_eq!(m.public().trick_cards, [(0, c("5S"))]);
    plays(&mut m, "JC 9S AC KS 10C");
    assert_eq!(m.scores(), [16, 0, 0, 0]);
    // Seat 5, the robber, plays without robbing (let through): playing the ace proves it.
    let mut m = rob_hand();
    plays(&mut m, "KC 4C 2C 5C 3C");
    lapse(&mut m, Lapse::PlayWithoutRob(c("6C")));
    assert_eq!(m.public().table_card, Some(c("7S")));
    plays(&mut m, "7C 8C 5S QC JC 9S AC");
    let from = m.events().len();
    play(&mut m, "AS");
    assert_eq!(
        log_from(&m, from),
        [
            "Played seat=5 card=AS",
            "PairTurned team=1 score=0 cause=proved"
        ]
    );
}

/// Rule 38, the ace shows from another hand: seat 2 takes the turned 7♠ without the ace (let
/// through); seat 5 plays the A♠ in trick 2 and seat 2's partnership is turned with that card.
#[test]
fn rule_38_the_ace_from_another_hand_catches_a_rob_without_it() {
    let mut m = rob_hand();
    plays(&mut m, "KC 4C");
    let before = m.clone();
    assert_eq!(
        m.let_through(Lapse::BurnAce).map_err(|e| e.why),
        Err(LapseRefusal::NotNow)
    );
    assert_eq!(m, before);
    lapse(&mut m, Lapse::FalseRob(c("JH")));
    assert_eq!(m.phase(), Phase::AwaitPlay(2));
    assert_eq!(m.public().robber, Some(2));
    plays(&mut m, "2C 5C 3C");
    assert_eq!(m.phase(), Phase::AwaitPlay(5), "seat 5 robs no more");
    plays(&mut m, "6C 7C 8C 5S QC 7S 9S AC");
    let from = m.events().len();
    play(&mut m, "AS");
    assert_eq!(
        log_from(&m, from),
        [
            "Played seat=5 card=AS",
            "PairTurned team=2 score=0 cause=ace_shown"
        ]
    );
    assert_eq!(m.phase(), Phase::AwaitPlay(7), "seat 6 is skipped");
}

/// Rule 38, the pair tries to claim the game: team 2 on 36 robbed without the ace and wins
/// trick 1 with the 7♠ it took: instead of 41 it is turned, the trick passes to seat 0's K♣,
/// and the next trick is led by the seat on the winner's left still in play.
#[test]
fn rule_38_a_claim_catches_a_rob_without_the_ace() {
    let mut m = rule_hand_from([0, 0, 36, 0], HANDS);
    plays(&mut m, "KC 4C");
    lapse(&mut m, Lapse::FalseRob(c("JH")));
    let from = m.events().len();
    plays(&mut m, "7S 5C 3C 6C 7C 8C");
    let l = log_from(&m, from);
    assert_eq!(
        l[l.len() - 4..],
        [
            "Played seat=7 card=8C",
            "TrickWon trick=1 seat=2 team=2 points=0",
            "PairTurned team=2 score=36 cause=claim",
            "TrickPassed trick=1 team=0 points=5",
        ]
    );
    assert_eq!(m.scores(), [5, 0, 36, 0]);
    assert_eq!(m.phase(), Phase::AwaitPlay(3));
}

/// Rule 38, the hand ends without the ace: seat 5 burns the A♠ when it robs (let through); at
/// the end of the hand its partnership is turned and its three tricks pass to team 2.
#[test]
fn rule_38_the_hand_ending_without_the_ace_catches_a_burned_ace() {
    let mut m = rob_hand();
    plays(&mut m, "KC 4C 2C 5C 3C");
    lapse(&mut m, Lapse::BurnAce);
    plays(&mut m, "6C 7C 8C");
    plays(&mut m, "5S QC JC 9S AC 7S KS 10C");
    plays(&mut m, "AD KD QD JD 4D 10D 9D 8D");
    plays(&mut m, "KH JH 9H 5D 7H 5H 3H 2D");
    let from = m.events().len();
    plays(&mut m, "QH 10H 8H 6D 6H 4H 2H 3D");
    let l = log_from(&m, from);
    assert_eq!(
        l[8..14],
        [
            "TrickWon trick=5 seat=1 team=1 points=5",
            "PairTurned team=1 score=0 cause=hand_end",
            "TrickPassed trick=3 team=2 points=5",
            "TrickPassed trick=4 team=2 points=5",
            "TrickPassed trick=5 team=2 points=5",
            "HandEnd hand=0 aborted=false",
        ]
    );
    assert_eq!(m.scores(), [16, 0, 15, 0]);
}

/// Every partnership turned in the middle of a trick: the trick goes to its best card for no
/// points (as the app's `resolveTrick`), the hand ends, and nobody scores it.
#[test]
fn every_partnership_turned_ends_the_hand_with_nobody_scoring() {
    let mut m = rule_hand();
    plays(&mut m, TRICK1);
    plays(&mut m, "5S QC");
    turn(&mut m, 2);
    turn(&mut m, 3);
    turn(&mut m, 1);
    let from = m.events().len();
    turn(&mut m, 0);
    assert_eq!(
        log_from(&m, from)[..3],
        [
            "PairTurned team=0 score=0 cause=before_play",
            "TrickWon trick=2 seat=0 team=0 points=0",
            "HandEnd hand=0 aborted=true",
        ]
    );
    assert_eq!(m.scores(), [0; 4]);
    assert_eq!(m.hand_index(), 1);
}

/// The turned pair played the 5 of trumps in the trick in progress (not as its lead): the 5
/// leaves the trick, so no immediate bonus is paid, and the bonus waits for the end of the hand.
#[test]
fn turned_five_in_the_trick_in_progress_pays_no_bonus() {
    // Seat 4 (team 0) holds the 5♠ and plays it to a trick seat 3 leads.
    let hands = [
        "KC 2S AD 2D 3D",
        "4C QC KH QH KD",
        "2C JC JH 10H QD",
        "5C 9S 9H 8H JD",
        "3C 5S 4D 5D 6D",
        "6C 9C 7H 6H 10D",
        "7C KS 5H 4H 9D",
        "8C 10C 3H 2H 8D",
    ];
    let mut m = Match::with_decks(2, vec![deck(2, hands, "7S AS")], 1, Rules::HOUSE);
    // Dealer 2: seat 3 leads trick 1 with the 9♠ (a trump lead): seat 4 may hold back its 5♠
    // (rule 16) but plays it; seats 6 and 0 must play their trumps.
    plays(&mut m, "9S 5S 9C KS 10C 2S");
    assert!(m.public().five_played);
    turn(&mut m, 0);
    let p = m.public();
    assert!(!p.five_played);
    assert_eq!(p.best_trump, Some((c("KS"), 6)));
    assert_eq!(
        p.trick_cards,
        [(3, c("9S")), (5, c("9C")), (6, c("KS")), (7, c("10C"))]
    );
    // Seat 1 and seat 2 still play; seat 4's 5♠ and seat 0's 2♠ do not count.
    let from = m.events().len();
    plays(&mut m, "QC JC");
    let l = log_from(&m, from);
    assert!(l.contains(&"TrickWon trick=1 seat=6 team=2 points=5".to_string()));
    assert!(
        !l.iter().any(|e| e.starts_with("Bonus")),
        "no immediate bonus"
    );
    assert_eq!(m.scores(), [0, 0, 5, 0]);
}

/// Two partnerships turned, the hand played out by the four players left.
#[test]
fn two_partnerships_turned_the_hand_goes_on_with_four() {
    let mut m = rule_hand();
    plays(&mut m, TRICK1);
    plays(&mut m, TRICK2);
    turn(&mut m, 0);
    turn(&mut m, 3);
    // Trick 2's K♠ (seat 6) is still the best trump in play; seat 3's 9♠ no longer counts.
    assert_eq!(m.public().best_trump, Some((c("KS"), 6)));
    assert_eq!(m.public().turned_teams, [true, false, false, true]);
    // Seats 1, 2, 5 and 6 play on.
    plays(&mut m, "KH JH 7H 5H");
    assert_eq!(m.phase(), Phase::AwaitPlay(1));
    plays(&mut m, "QH 10H");
    assert_eq!(m.to_act(), Some(5), "seats 3 and 4 are skipped");
    plays(&mut m, "6H 4H");
    plays(&mut m, "KD QD 10D 9D");
    assert_eq!(m.hand_index(), 1);
    assert_eq!(m.scores(), [0, 15, 16, 0]);
}

/// The record (actions and extras) replays to exactly the same match, and every seat's view
/// shows the turn events. The bots' actions never include a turn or a lapse.
#[test]
fn a_record_with_turns_and_lapses_replays_exactly() {
    let mut m = rob_hand();
    plays(&mut m, "KC 4C");
    lapse(&mut m, Lapse::FalseRob(c("JH")));
    plays(&mut m, "2C");
    turn(&mut m, 3);
    plays(&mut m, "3C 6C 7C");
    play_turn(&mut m, "5S", 1);
    let deck0 = deck(7, ROB_HANDS, "7S 9C");
    let mut r = Match::with_decks(7, vec![deck0], 1, Rules::HOUSE);
    r.apply_record(m.actions(), m.extras()).expect("replays");
    assert_eq!(r, m);
    assert_eq!(
        m.extras(),
        [
            (2, Extra::Lapse(Lapse::FalseRob(c("JH")))),
            (3, Extra::Turn(3)),
            (6, Extra::PlayThenTurn(c("5S"), 1)),
        ]
    );
    for seat in 0..8 {
        let v = m.view(seat);
        assert!(
            v.events
                .iter()
                .any(|e| matches!(e, Event::PairTurned { team: 3, .. }))
        );
    }
    // Out of order or past the end: refused.
    let mut r = rob_hand();
    assert!(r.apply_record(&m.actions()[..1], m.extras()).is_err());
}

/// For rebuilding a world from a seat's view (the search), never for a record: `observe_play`
/// places a card as the table saw it, legal or not, with nothing recorded as a renege, undoing a
/// rob the engine would have made, and turns the partnerships named with it (a proof before the
/// engine's own catches). It is not among the bots' actions and adds nothing to `extras()`.
#[test]
fn observe_play_places_a_card_as_the_table_saw_it() {
    // An illegal card: seat 1 holds two clubs and plays the K♥ to a club lead.
    let mut m = rule_hand();
    play(&mut m, "KC");
    assert!(!m.legal_actions().contains(&Action::Play(c("KH"))));
    m.observe_play(c("KH"), &[]).expect("placed");
    assert!(m.reneges().is_empty() && m.extras().is_empty());
    assert_eq!(m.phase(), Phase::AwaitPlay(2));
    // With a proof named: seat 1's Q♣ in trick 2 turns its pair there and then.
    plays(&mut m, "2C 5C 3C 6C 7C 8C 5S");
    let from = m.events().len();
    m.observe_play(c("QC"), &[(1, TurnCause::Proved)])
        .expect("placed");
    assert_eq!(
        log_from(&m, from),
        [
            "Played seat=1 card=QC",
            "PairTurned team=1 score=0 cause=proved"
        ]
    );
    // A rob the engine would make is undone: seat 5 holds the ace and plays without robbing.
    let mut m = rob_hand();
    plays(&mut m, "KC 4C 2C 5C 3C");
    assert_eq!(m.phase(), Phase::AwaitBurn(5));
    assert_eq!(
        m.observe_play(c("7S"), &[]),
        Err(TurnRefusal::IllegalCard),
        "the card it was given cannot be played so"
    );
    m.observe_play(c("6C"), &[]).expect("placed");
    assert_eq!(m.public().table_card, Some(c("7S")));
    assert_eq!(m.public().robber, None);
    assert!(m.reneges().is_empty());
    // The open question is refused here too: seat 0's partnership led the trick.
    let mut m = rule_hand();
    plays(&mut m, "KC 4C");
    let before = m.clone();
    assert_eq!(
        m.observe_play(c("2C"), &[(0, TurnCause::WithPlay)]),
        Err(TurnRefusal::LedTrickInProgress)
    );
    assert_eq!(m, before);
}

// ===== properties over random hands with random turns and lapses =====

/// Plays random matches, turning random partnerships now and then (before a play, with a play,
/// while a rob is due) and letting random reneges through, and checks after every step: the
/// record replays exactly; a turned partnership's score is its hand-start score and it never
/// acts; the tricks held add up; the hand-start state is recovered from any seat's view; a
/// refusal changes nothing.
#[test]
fn random_turns_and_lapses_keep_every_invariant() {
    let mut causes = [0u32; 9];
    let mut refusals = 0u32;
    let mut games_by_pass = 0u32;
    let mut jinks_by_turns = 0u32;
    let mut stuck = 0u32;
    for seed in 0..160u64 {
        let mut rng = Rng::new(seed ^ 0x7E57);
        let mut m = Match::new(seed, Rules::HOUSE);
        let mut start = m.current_hand_start();
        let mut steps = 0;
        let mut turned_this_hand = false;
        while m.phase() != Phase::MatchOver && steps < 4000 {
            steps += 1;
            let h = m.hand_index();
            // A turn or lapse while a rob is due undoes the take (the last event).
            let events_before =
                m.events().len() - usize::from(matches!(m.phase(), Phase::AwaitBurn(_)));
            let before = m.clone();
            let done = match random_extra(&mut m, &mut rng) {
                Some(Ok(())) => true,
                Some(Err(())) => {
                    refusals += 1;
                    assert_eq!(m, before, "a refusal changes nothing");
                    false
                }
                None => false,
            };
            if !done && !play_some_legal_action(&mut m, &mut rng) {
                // Every card the seat holds proves a renege of the partnership that led the
                // trick (the open question): the engine stops here.
                stuck += 1;
                break;
            }
            for (_, e) in &m.events()[events_before..] {
                match e {
                    Event::PairTurned { cause, .. } => {
                        causes[*cause as usize] += 1;
                        turned_this_hand = true;
                    }
                    Event::GameWon { reason, .. } => match reason {
                        GameEndReason::FortyOneByPassedTrick => games_by_pass += 1,
                        GameEndReason::Jink if turned_this_hand => jinks_by_turns += 1,
                        _ => {}
                    },
                    Event::HandStart { .. } => turned_this_hand = false,
                    _ => {}
                }
            }
            if m.phase() == Phase::MatchOver {
                break;
            }
            if m.hand_index() != h {
                start = m.current_hand_start();
            }
            check_hand(&m, &start);
        }
        let r = Match::replay_record(seed, Rules::HOUSE, m.actions(), m.extras()).expect("replay");
        assert_eq!(r, m, "seed {seed}");
    }
    println!(
        "causes {causes:?} refusals {refusals} by pass {games_by_pass} jinks {jinks_by_turns} \
         stuck {stuck}"
    );
    // before_play, with_play, proved, ace_shown, claim, hand_end, no_rob, declined_rob,
    // left_joker: every one is seen.
    assert!(causes.iter().all(|&n| n > 0), "{causes:?}");
    assert!(refusals > 0 && games_by_pass > 0 && jinks_by_turns > 0);
    assert!(stuck < 80, "most matches are played to the end");
}

/// One random turn or lapse now and then: `None` when none was tried.
fn random_extra(m: &mut Match, rng: &mut Rng) -> Option<Result<(), ()>> {
    let team = rng.bounded(4) as u8;
    match m.phase() {
        Phase::AwaitBurn(s) => {
            let hand = m.truth().hands[usize::from(s)];
            match rng.bounded(10) {
                0 | 1 => Some(m.turn_team(engine::team_of(s)).map_err(|_| ())),
                2 => Some(m.let_through(Lapse::BurnAce).map_err(|_| ())),
                3 => {
                    let cards: Vec<Card> = engine::cards::cards(hand).collect();
                    let card = cards[rng.bounded(cards.len() as u64) as usize];
                    Some(m.let_through(Lapse::PlayWithoutRob(card)).map_err(|_| ()))
                }
                4 => Some(m.turn_team(team).map_err(|_| ())),
                _ => None,
            }
        }
        Phase::AwaitPlay(s) => {
            let hand = m.truth().hands[usize::from(s)];
            let cards: Vec<Card> = engine::cards::cards(hand).collect();
            let card = cards[rng.bounded(cards.len() as u64) as usize];
            match rng.bounded(40) {
                0 => Some(m.turn_team(team).map_err(|_| ())),
                1 => Some(m.play_then_turn(card, team).map_err(|_| ())),
                2..=5 if !engine::cards::has(m.legal_mask(), card) => {
                    Some(m.let_through(Lapse::Play(card)).map_err(|_| ()))
                }
                6 | 7 => Some(m.let_through(Lapse::FalseRob(card)).map_err(|_| ())),
                _ => None,
            }
        }
        Phase::MatchOver => None,
    }
}

/// Apply a random legal action, trying the others when one is refused (a legal card that
/// proves a renege of the partnership that led the trick: the open question); when every one
/// is refused, the app catches the seat's partnership before its card, or the seat plays an
/// illegal card. `false` when nothing can be done (the engine stops: the app's own bot plays on).
fn play_some_legal_action(m: &mut Match, rng: &mut Rng) -> bool {
    let acts = m.legal_actions();
    let first = rng.bounded(acts.len() as u64) as usize;
    if (0..acts.len()).any(|i| m.apply(acts[(first + i) % acts.len()]).is_ok()) {
        return true;
    }
    let Some(seat) = m.to_act() else { return false };
    if m.turn_team(team_of(seat)).is_ok() {
        return true;
    }
    let hand = m.truth().hands[usize::from(seat)];
    engine::cards::cards(hand).any(|c| m.let_through(Lapse::Play(c)).is_ok())
}

fn check_hand(m: &Match, start: &HandStartState) {
    let p = m.public();
    let turned = m.turned_teams();
    assert_eq!(p.turned_teams, turned);
    for (t, &out) in turned.iter().enumerate() {
        if out {
            assert_eq!(p.scores[t], start.scores[t], "rule 41");
            assert_eq!(p.tricks_won[t], 0, "rule 39");
        }
    }
    if let Some(s) = m.to_act() {
        assert!(!turned[usize::from(team_of(s))], "a turned seat never acts");
    }
    assert!(
        p.trick_cards
            .iter()
            .all(|&(s, _)| !turned[usize::from(team_of(s))])
    );
    let held = p.tricks_won.iter().map(|&x| usize::from(x)).sum::<usize>();
    if turned.iter().any(|&t| t) {
        assert!(held <= p.trick_winners.len());
    } else {
        assert_eq!(held, p.trick_winners.len());
    }
    if let Some((_, seat)) = p.best_trump {
        assert!(!turned[usize::from(team_of(seat))]);
    }
    for seat in 0..8 {
        assert_eq!(current_hand_start_of(&m.view(seat)), *start);
    }
}
