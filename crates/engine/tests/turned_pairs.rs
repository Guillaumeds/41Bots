//! Turned partnerships (rules 34, 39, 40 and 41; the owner's decision of 2026-10-17 in
//! docs/APP-ENGINE-CONTRACT.md §7): the app tells the engine which pair it has caught and when
//! (`Match::turn_team`), and the hand goes on exactly as the rules say. The engine's own players
//! never act illegally (H1), so nothing here is ever caused by a play.
//!
//! One test per rule (named after it), the edge cases the owner's decision lists, the three
//! cases the rules leave open (refused, not guessed), and properties over random hands with
//! random turns: records replay exactly, every seat's view and the hand-start state stay right,
//! and a hand where nobody is turned is exactly what it always was.

use engine::cards::{Bits, Card, DECK_SIZE, bit, parse};
use engine::game::{
    Action, Event, GameEndReason, HandStartState, Match, Phase, Rules, TurnRefusal, build_deck,
    current_hand_start_of,
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
            "PairTurned team=0 score=0",
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
    // The seat to lead (seat 0) is turned: the lead passes to seat 1's left, seat 2 (seat 1 is
    // turned too).
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
/// After two tricks, teams 0, 1 and 3 are turned: team 2 alone is in play and wins the game at
/// once with a jink, with tricks 1 and 2 passed to it; the rest of the hand is not played.
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
        log_from(&m, from)[..7],
        [
            "PairTurned team=0 score=0",
            "TrickPassed trick=1 team=2 points=5",
            "TrickPassed trick=2 team=2 points=5",
            "PairTurned team=1 score=0",
            "PairTurned team=3 score=0",
            "GameWon team=2 reason=jink scores=0,0,10,0",
            "HandEnd hand=0 aborted=true",
        ]
    );
    assert_eq!(m.games_won(), [0, 0, 1, 0]);
    assert_eq!(m.scores(), [0; 4], "a new game");
    // Team 1 wins trick 3 among six and is turned: trick 3 passes (rule 39) to seat 2's J♥, the
    // highest heart left; then team 3 is turned and team 2, alone in play, wins the game.
    let mut m = rule_hand();
    plays(&mut m, TRICK1);
    plays(&mut m, TRICK2);
    turn(&mut m, 0);
    plays(&mut m, "KH JH 9H 7H 5H 3H");
    let from = m.events().len();
    turn(&mut m, 1);
    turn(&mut m, 3);
    assert_eq!(
        log_from(&m, from)[..4],
        [
            "PairTurned team=1 score=0",
            "TrickPassed trick=3 team=2 points=5",
            "PairTurned team=3 score=0",
            "GameWon team=2 reason=jink scores=0,0,15,0",
        ]
    );
}

// ===== edge cases =====

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

/// The leader's partnership turned after the lead (the trick in progress was led by a turned
/// card) is a case the rules leave open: refused, and the match is unchanged.
#[test]
fn leader_turned_mid_trick_is_refused_as_an_open_question() {
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
    // Another partnership mid-trick is fine.
    assert!(m.turn_team(3).is_ok());
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

/// Passed tricks take a partnership to 41: the game ends at once (rule 32, a passed trick
/// included). With two partnerships reaching 41 at the same moment the rules do not say who wins:
/// refused.
#[test]
fn passed_tricks_can_win_the_game_and_two_at_41_is_refused() {
    let mut m = rule_hand_from([10, 5, 36, 0], HANDS);
    plays(&mut m, TRICK1);
    plays(&mut m, TRICK2);
    let from = m.events().len();
    turn(&mut m, 0);
    assert_eq!(
        log_from(&m, from)[..5],
        [
            "PairTurned team=0 score=10",
            "TrickPassed trick=1 team=2 points=5",
            "TrickPassed trick=2 team=2 points=5",
            "GameWon team=2 reason=passed_trick scores=10,5,46,0",
            "HandEnd hand=3 aborted=true",
        ]
    );
    assert_eq!(m.games_won(), [0, 0, 1, 0]);
    // Trick 2 passes to team 3 instead when seat 3 holds the K♠ and seat 6 the 9♠.
    let mut swapped = HANDS;
    swapped[3] = "5C KS 9H 8H JD";
    swapped[6] = "7C 9S 5H 4H 9D";
    let mut m = rule_hand_from([10, 5, 36, 36], swapped);
    plays(&mut m, TRICK1);
    plays(&mut m, "5S QC JC KS AC 9C 9S 10C");
    let before = m.clone();
    let e = m.turn_team(0).expect_err("two partnerships at 41 at once");
    assert_eq!(e.why, TurnRefusal::SeveralReachFortyOne);
    assert_eq!(m, before);
    // One of them reaching 41 is fine.
    let mut m = rule_hand_from([10, 5, 36, 30], swapped);
    plays(&mut m, TRICK1);
    plays(&mut m, "5S QC JC KS AC 9C 9S 10C");
    turn(&mut m, 0);
    assert_eq!(m.games_won(), [0, 0, 1, 0]);
}

/// A trick the turned partnership holds in which no card of the partnerships still in play is a
/// trump or of the suit led, with two or more of them in play: "the strongest card in that trick"
/// is not defined by the rules: refused. With one partnership left in play, rule 34 decides.
#[test]
fn a_trick_with_no_card_that_can_win_is_refused_unless_one_pair_is_left() {
    // Seat 0 leads the K♣; seats 1, 2, 3, 5, 6 and 7 hold no club and discard a plain card.
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
    let before = m.clone();
    let e = m.turn_team(0).expect_err("no card in play can win trick 1");
    assert_eq!(e.why, TurnRefusal::NoStandingWinner { trick: 1 });
    assert_eq!(m, before);
    // With teams 1 and 3 turned first, team 2 is the only one left when team 0 is turned: it
    // takes trick 1 (rule 34) and wins the game with a jink.
    turn(&mut m, 1);
    turn(&mut m, 3);
    let from = m.events().len();
    turn(&mut m, 0);
    assert_eq!(
        log_from(&m, from)[..3],
        [
            "PairTurned team=0 score=0",
            "TrickPassed trick=1 team=2 points=5",
            "GameWon team=2 reason=jink scores=0,0,5,0",
        ]
    );
}

/// The record (actions and turns) replays to exactly the same match, and every seat's view
/// shows the turn events.
#[test]
fn a_record_with_turns_replays_exactly() {
    let mut m = rule_hand();
    plays(&mut m, "KC 4C 2C");
    turn(&mut m, 1);
    plays(&mut m, "5C 3C 7C 8C");
    turn(&mut m, 3);
    let deck0 = deck(7, HANDS, "7S AS");
    let mut r = Match::with_decks(7, vec![deck0], 1, Rules::HOUSE);
    r.apply_record(m.actions(), m.turns()).expect("replays");
    assert_eq!(r, m);
    assert_eq!(m.turns(), [(3, 1), (7, 3)]);
    for seat in 0..8 {
        let v = m.view(seat);
        assert!(
            v.events
                .iter()
                .any(|e| matches!(e, Event::PairTurned { team: 3, .. }))
        );
    }
}

// ===== properties over random hands with random turns =====

/// Plays random matches, turning a random partnership now and then (sometimes two or three in
/// a hand), and checks after every step: the record replays exactly; a turned partnership's
/// score is its hand-start score and it never acts; the tricks held add up; the hand-start state
/// is recovered from any seat's view; a refusal changes nothing.
#[test]
fn random_turns_keep_every_invariant() {
    let mut turned_seen = [0u32; 4];
    let mut refusals = 0u32;
    let mut games_by_pass = 0u32;
    let mut jinks_by_turns = 0u32;
    for seed in 0..120u64 {
        let mut rng = Rng::new(seed ^ 0x7E57);
        let mut m = Match::new(seed, Rules::HOUSE);
        let mut start = m.current_hand_start();
        let mut steps = 0;
        while m.phase() != Phase::MatchOver && steps < 3000 {
            steps += 1;
            let h = m.hand_index();
            if matches!(m.phase(), Phase::AwaitPlay(_)) && rng.bounded(14) == 0 {
                let team = rng.bounded(4) as u8;
                let before = m.clone();
                let n_turned = m.turned_teams().iter().filter(|&&t| t).count();
                match m.turn_team(team) {
                    Ok(()) => {
                        turned_seen[n_turned] += 1;
                        if let Some((_, Event::GameWon { reason, .. })) = m
                            .events()
                            .iter()
                            .rev()
                            .find(|(_, e)| matches!(e, Event::GameWon { .. }))
                            && m.hand_index() != h
                        {
                            match reason {
                                GameEndReason::FortyOneByPassedTrick => games_by_pass += 1,
                                GameEndReason::Jink => jinks_by_turns += 1,
                                _ => {}
                            }
                        }
                    }
                    Err(_) => {
                        refusals += 1;
                        assert_eq!(m, before, "a refusal changes nothing");
                    }
                }
            } else {
                let acts = m.legal_actions();
                let a = acts[rng.bounded(acts.len() as u64) as usize];
                m.apply(a).expect("legal");
            }
            if m.phase() == Phase::MatchOver {
                break;
            }
            if m.hand_index() != h {
                start = m.current_hand_start();
            }
            check_hand(&m, &start);
        }
        let r = Match::replay_record(seed, Rules::HOUSE, m.actions(), m.turns()).expect("replay");
        assert_eq!(r, m, "seed {seed}");
    }
    assert!(
        turned_seen.iter().take(3).all(|&n| n > 0),
        "{turned_seen:?}"
    );
    assert!(refusals > 0 && games_by_pass > 0 && jinks_by_turns > 0);
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
    assert_eq!(
        p.tricks_won.iter().map(|&x| usize::from(x)).sum::<usize>(),
        p.trick_winners.len()
    );
    if let Some((_, seat)) = p.best_trump {
        assert!(!turned[usize::from(team_of(seat))]);
    }
    for seat in 0..8 {
        assert_eq!(current_hand_start_of(&m.view(seat)), *start);
    }
}
