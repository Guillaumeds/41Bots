//! Tests of the hand, game and match flow. These live inside `game` so a test
//! can set a score directly instead of playing several hands to reach it.

use super::*;
use crate::test_support::*;

fn public_events(m: &Match) -> Vec<String> {
    m.events
        .iter()
        .filter(|(v, _)| v.is_none())
        .map(|(_, t)| t.clone())
        .collect()
}

fn count_events(m: &Match, prefix: &str) -> usize {
    m.events
        .iter()
        .filter(|(_, t)| t.starts_with(prefix))
        .count()
}

/// Asserts that lines starting with each of `expected` appear in `lines`, in
/// that order (full lines match themselves).
fn assert_in_order(lines: &[String], expected: &[&str]) {
    let mut i = 0;
    for line in lines {
        if i < expected.len() && line.starts_with(expected[i]) {
            i += 1;
        }
    }
    assert_eq!(
        i,
        expected.len(),
        "missing {:?} (in order) in:\n{}",
        expected.get(i),
        lines.join("\n")
    );
}

/// Plays the lowest legal action until the phase is `target`.
fn play_until(m: &mut Match, target: Phase) {
    for _ in 0..1000 {
        if m.phase() == target {
            return;
        }
        act_first(m);
    }
    panic!("never reached {target:?}");
}

fn identity_deck() -> [u8; 53] {
    std::array::from_fn(|i| i as u8)
}

// --------------------------------------------------------------------------
// The deal (rules 3-5, deal convention)
// --------------------------------------------------------------------------

#[test]
fn deal_convention_two_then_three_from_left_of_dealer() {
    let mut m = Match::new(3, Rules::default());
    assert!(m.needs_deck());
    assert_eq!(m.phase(), Phase::NeedsDeck);
    assert!(m.legal_actions().is_empty());
    m.start_hand(&identity_deck());

    // Seat (dealer + 1 + i) gets positions 2i, 2i+1 and 16+3i..16+3i+3.
    assert_eq!(m.hand(4), vec![0, 1, 16, 17, 18]);
    assert_eq!(m.hand(5), vec![2, 3, 19, 20, 21]);
    assert_eq!(m.hand(6), vec![4, 5, 22, 23, 24]);
    assert_eq!(m.hand(7), vec![6, 7, 25, 26, 27]);
    assert_eq!(m.hand(0), vec![8, 9, 28, 29, 30]);
    assert_eq!(m.hand(1), vec![10, 11, 31, 32, 33]);
    assert_eq!(m.hand(2), vec![12, 13, 34, 35, 36]);
    assert_eq!(m.hand(3), vec![14, 15, 37, 38, 39]);
    // Position 40 is turned; 41..52 stay undealt (40 dealt, 1 turned, 12 undealt).
    assert_eq!(m.undealt(), (41..53).collect::<Vec<u8>>());

    let p = m.public();
    assert_eq!(p.trump, Some(3)); // 40 is the 2 of spades
    assert_eq!(p.table_card, Some(idx("2S")));
    assert_eq!(p.turned, vec![(idx("2S"), None)]);
    assert!(!p.joker_turned);
    assert_eq!(p.hidden_count, 12);
    assert_eq!(p.hand_sizes, [5; 8]);
    assert_eq!(p.trick_no, 1);
    assert_eq!(p.leader, 4); // rule 12: left of the dealer leads
    assert_eq!(p.phase, Phase::AwaitPlay(4));
    assert_eq!(m.to_act(), Some(4));

    assert_eq!(
        m.events_for(4),
        vec![
            "MatchStart first_dealer=3",
            "HandStart hand=0 game=0 dealer=3",
            "Dealt seat=4 cards=AC,2C,4D,5D,6D",
            "Turned card=2S",
        ]
    );
    assert_eq!(m.events_for(3)[2], "Dealt seat=3 cards=2D,3D,QH,KH,AS");
    // Dealt events come seat by seat in deal order, from left of the dealer.
    let dealt_order: Vec<String> = m
        .events
        .iter()
        .filter(|(_, t)| t.starts_with("Dealt"))
        .map(|(v, _)| format!("{}", v.unwrap()))
        .collect();
    assert_eq!(dealt_order.join(""), "45670123");
}

#[test]
fn dealer_holding_the_ace_robs_as_the_eighth_card_of_trick_1() {
    // Identity deck, dealer 3: the dealer holds the AS and the 2S is turned.
    let mut m = Match::new(3, Rules::default());
    m.start_hand(&identity_deck());
    play_until(&mut m, Phase::AwaitBurn(3));
    assert_eq!(m.public().trick_cards.len(), 7);
    for seat in [4, 5, 6, 7, 0, 1, 2] {
        assert!(m.public().proved_no_ace[seat]);
    }
    assert!(!m.public().proved_no_ace[3]);
    assert_eq!(public_events(&m).last().unwrap(), "Took seat=3 card=2S");
    assert_eq!(m.public().hand_sizes[3], 6);
    // Burns: everything but the ace of spades (H3).
    assert_eq!(legal_burn_names(&m), "2D 3D QH KH 2S");
    m.apply(Action::Burn(idx("2S"))).unwrap();
    assert_eq!(m.phase(), Phase::AwaitPlay(3));
    assert_eq!(m.public().robber, Some(3));
    assert_eq!(m.public().hidden_count, 13);
    assert_eq!(m.burns(3), vec![idx("2S")]);
    act_first(&mut m);
    assert_eq!(count_events(&m, "ProvedNoAce"), 7);
    assert_eq!(count_events(&m, "AceUndealt"), 0);
}

// --------------------------------------------------------------------------
// Robbing (rules 20-23, H2, H3)
// --------------------------------------------------------------------------

#[test]
fn robber_robs_at_own_first_turn_in_trick_1() {
    // Hearts trumps: the AH is the ace of trumps, held by seat 2.
    let d = deck(7, ["", "", "AH AD", "", "", "", "", ""], "9H", "");
    let mut m = Match::new(7, Rules::default());
    m.start_hand(&d);
    assert_eq!(m.public().trump, Some(2));
    assert_eq!(m.phase(), Phase::AwaitPlay(0));

    act_first(&mut m); // seat 0
    act_first(&mut m); // seat 1
    assert_in_order(
        &public_events(&m),
        &[
            "ProvedNoAce seat=0",
            "Played seat=0 card=2C",
            "ProvedNoAce seat=1",
            "Took seat=2 card=9H",
        ],
    );
    // H2: the robber chooses the burn after seeing the trick-1 cards already played.
    assert_eq!(m.phase(), Phase::AwaitBurn(2));
    assert_eq!(m.public().trick_cards.len(), 2);
    assert_eq!(m.public().hand_sizes[2], 6);
    assert_eq!(m.public().robber, Some(2));
    assert_eq!(m.public().table_card, None);
    assert_eq!(m.public().turned, vec![(idx("9H"), Some(2))]);
    // H3: neither the ace of trumps nor any other ace may be burned.
    assert_eq!(legal_burn_names(&m), "QC KC 2D 9H");
    assert!(m.apply(Action::Burn(idx("AH"))).is_err());
    assert!(m.apply(Action::Burn(idx("AD"))).is_err());
    assert!(m.apply(Action::Play(idx("QC"))).is_err());

    m.apply(Action::Burn(idx("9H"))).unwrap();
    assert_eq!(m.phase(), Phase::AwaitPlay(2));
    assert_eq!(m.public().hand_sizes[2], 5);
    assert_eq!(m.burns(2), vec![idx("9H")]);
    assert_eq!(m.public().burn_counts, [0, 0, 1, 0, 0, 0, 0, 0]);
    assert_eq!(m.public().hidden_count, 13);
    // The burn is private to the burner; everyone sees that a burn happened.
    assert!(
        m.events_for(2)
            .contains(&"Burned seat=2 card=9H".to_string())
    );
    assert!(
        !m.events_for(3)
            .contains(&"Burned seat=2 card=9H".to_string())
    );
    assert!(m.events_for(3).contains(&"BurnDone seat=2".to_string()));

    // Nobody after the robber proves anything.
    for _ in 0..6 {
        act_first(&mut m);
    }
    assert_eq!(m.public().trick_winners.len(), 1);
    assert_eq!(count_events(&m, "ProvedNoAce"), 2);
    assert_eq!(count_events(&m, "AceUndealt"), 0);
}

#[test]
fn robber_who_leads_robs_before_leading() {
    let d = deck(7, ["AC", "", "", "", "", "", "", ""], "9C", "");
    let mut m = Match::new(7, Rules::default());
    m.start_hand(&d);
    assert_eq!(m.phase(), Phase::AwaitBurn(0));
    assert_eq!(m.public().trick_no, 1);
    assert_eq!(public_events(&m).last().unwrap(), "Took seat=0 card=9C");
    assert_eq!(count_events(&m, "ProvedNoAce"), 0);
    m.apply(Action::Burn(idx("9C"))).unwrap();
    assert_eq!(m.phase(), Phase::AwaitPlay(0));
    finish_hand_preferring(&mut m, &[]);
    assert_eq!(count_events(&m, "ProvedNoAce"), 0);
}

#[test]
fn nobody_robs_when_the_ace_of_trumps_is_undealt() {
    // Rule 23: AC undealt; the 9C stays on the table for the whole hand.
    let d = deck(7, ["", "", "", "", "", "", "", ""], "9C", "AC");
    let mut m = Match::new(7, Rules::default());
    m.start_hand(&d);
    for _ in 0..8 {
        act_first(&mut m);
    }
    assert_eq!(count_events(&m, "ProvedNoAce"), 8);
    assert_eq!(count_events(&m, "AceUndealt"), 1);
    assert_in_order(&public_events(&m), &["TrickWon trick=1", "AceUndealt"]);
    let p = m.public();
    assert!(p.ace_undealt);
    assert_eq!(p.robber, None);
    assert_eq!(p.table_card, Some(idx("9C")));
    assert_eq!(p.hidden_count, 12);
}

#[test]
fn dealer_takes_a_turned_ace_at_once() {
    // Rule 22 and H4.
    let d = deck(5, ["", "", "", "", "", "AC", "", ""], "AD", "");
    let mut m = Match::new(5, Rules::default());
    m.start_hand(&d);
    assert_in_order(
        &public_events(&m),
        &["Turned card=AD", "Took seat=5 card=AD"],
    );
    let p = m.public();
    assert_eq!(p.phase, Phase::AwaitBurn(5));
    assert_eq!(p.trump, Some(1));
    assert_eq!(p.trick_no, 0);
    assert_eq!(p.robber, Some(5));
    assert_eq!(p.table_card, None);
    assert_eq!(p.turned, vec![(idx("AD"), Some(5))]);
    assert_eq!(p.hand_sizes[5], 6);
    // Neither the turned ace nor the dealer's other ace may be burned.
    let burns = legal_burn_names(&m);
    assert!(!burns.contains("AD") && !burns.contains("AC"));
    assert_eq!(burns.split(' ').count(), 4);

    act_first(&mut m);
    let p = m.public();
    assert_eq!(p.phase, Phase::AwaitPlay(6));
    assert_eq!(p.trick_no, 1);
    assert_eq!(p.leader, 6);
    assert_eq!(p.hidden_count, 13);
    for _ in 0..8 {
        act_first(&mut m);
    }
    assert_eq!(count_events(&m, "ProvedNoAce"), 0);
    assert_eq!(count_events(&m, "AceUndealt"), 0);
}

#[test]
fn turned_ace_of_hearts_makes_hearts_trumps_and_is_robbed() {
    let d = deck(0, ["", "", "", "", "", "", "", ""], "AH", "");
    let mut m = Match::new(0, Rules::default());
    m.start_hand(&d);
    assert_eq!(m.public().trump, Some(2));
    assert_eq!(m.phase(), Phase::AwaitBurn(0));
    assert!(m.hand(0).contains(&idx("AH")));
    assert!(!legal_burn_names(&m).contains("AH"));
}

// --------------------------------------------------------------------------
// The turned Joker (rule 24, H4)
// --------------------------------------------------------------------------

#[test]
fn turned_joker_then_ordinary_card_with_ace_undealt() {
    let d = deck(2, ["", "", "AD", "", "", "", "", ""], "JK 7C", "");
    let mut m = Match::new(2, Rules::default());
    m.start_hand(&d);
    assert_in_order(
        &public_events(&m),
        &["Turned card=JK", "Took seat=2 card=JK"],
    );
    let p = m.public();
    assert_eq!(p.phase, Phase::AwaitBurn(2));
    assert_eq!(p.trump, None); // the trump suit is not known yet
    assert_eq!(p.trick_no, 0);
    assert!(p.joker_turned);
    assert_eq!(p.robber, None);
    assert_eq!(p.table_card, None);
    assert_eq!(p.hand_sizes[2], 6);
    assert_eq!(p.hidden_count, 12); // the 7C is still face down in the pack
    // H3: aces of every suit are refused even before trumps are known;
    // the Joker itself may be burned.
    let burns = legal_burn_names(&m);
    assert!(!burns.contains("AD"));
    assert!(burns.ends_with("JK"));
    assert!(m.apply(Action::Burn(idx("AD"))).is_err());

    m.apply(Action::Burn(idx("JK"))).unwrap();
    assert_in_order(
        &m.events_for(2),
        &[
            "Burned seat=2 card=JK",
            "BurnDone seat=2",
            "Turned card=7C",
            "ProvedNoAce seat=2",
        ],
    );
    let p = m.public();
    assert_eq!(p.phase, Phase::AwaitPlay(3));
    assert_eq!(p.trump, Some(0));
    assert_eq!(p.table_card, Some(idx("7C")));
    assert_eq!(p.turned, vec![(52, Some(2)), (idx("7C"), None)]);
    assert_eq!(m.undealt().len(), 11);
    let mut rest: Vec<u8> = d[42..].to_vec();
    rest.sort();
    assert_eq!(m.undealt(), rest);
    assert_eq!(p.hidden_count, 12);

    for _ in 0..8 {
        act_first(&mut m);
    }
    // Every other seat proves it lacks the AC; the dealer is not repeated.
    assert_eq!(count_events(&m, "ProvedNoAce seat=2"), 1);
    assert_eq!(count_events(&m, "ProvedNoAce"), 8);
    assert_eq!(count_events(&m, "AceUndealt"), 1);
    assert!(m.public().ace_undealt);
}

#[test]
fn turned_joker_then_ace_dealer_robs_twice() {
    let d = deck(2, ["", "", "", "", "", "", "", ""], "JK AS", "");
    let mut m = Match::new(2, Rules::default());
    m.start_hand(&d);
    act_first(&mut m); // first burn
    assert_in_order(
        &public_events(&m),
        &["BurnDone seat=2", "Turned card=AS", "Took seat=2 card=AS"],
    );
    let p = m.public();
    assert_eq!(p.phase, Phase::AwaitBurn(2));
    assert_eq!(p.trump, Some(3));
    assert_eq!(p.robber, Some(2));
    assert_eq!(p.turned, vec![(52, Some(2)), (idx("AS"), Some(2))]);
    assert!(!legal_burn_names(&m).contains("AS"));
    act_first(&mut m); // second burn
    let p = m.public();
    assert_eq!(p.phase, Phase::AwaitPlay(3));
    assert_eq!(p.burn_counts[2], 2);
    assert_eq!(p.hidden_count, 13);
    assert_eq!(p.hand_sizes, [5; 8]);
    for _ in 0..8 {
        act_first(&mut m);
    }
    assert_eq!(count_events(&m, "ProvedNoAce"), 0);
}

#[test]
fn turned_joker_dealer_holding_new_ace_of_trumps_robs_at_once() {
    let d = deck(2, ["", "", "AC", "", "", "", "", ""], "JK 7C", "");
    let mut m = Match::new(2, Rules::default());
    m.start_hand(&d);
    let first_burn = m.legal_actions()[0];
    m.apply(first_burn).unwrap();
    assert_in_order(
        &public_events(&m),
        &["Turned card=7C", "Took seat=2 card=7C"],
    );
    assert_eq!(m.phase(), Phase::AwaitBurn(2));
    assert_eq!(m.public().robber, Some(2));
    assert!(!legal_burn_names(&m).contains("AC"));
    assert!(legal_burn_names(&m).contains("7C"));
    m.apply(Action::Burn(idx("7C"))).unwrap();
    assert_eq!(m.phase(), Phase::AwaitPlay(3));
    assert_eq!(m.burns(2).len(), 2);
    assert_eq!(count_events(&m, "ProvedNoAce"), 0);
}

#[test]
fn turned_joker_another_seat_holds_ace_robs_at_its_turn() {
    // Two takes in one hand: the dealer's Joker and seat 5's rob; two burns.
    let d = deck(2, ["", "", "", "", "", "AC", "", ""], "JK 7C", "");
    let mut m = Match::new(2, Rules::default());
    m.start_hand(&d);
    act_first(&mut m); // the dealer's Joker burn
    assert_eq!(public_events(&m).last().unwrap(), "ProvedNoAce seat=2");
    assert_eq!(m.phase(), Phase::AwaitPlay(3));
    play_until(&mut m, Phase::AwaitBurn(5));
    assert_eq!(public_events(&m).last().unwrap(), "Took seat=5 card=7C");
    act_first(&mut m);
    for _ in 0..6 {
        act_first(&mut m);
    }
    assert_eq!(m.public().trick_winners.len(), 1);
    let proved: Vec<bool> = m.public().proved_no_ace.to_vec();
    assert_eq!(
        proved,
        vec![false, false, true, true, true, false, false, false]
    );
    assert_eq!(m.burns(2).len(), 1);
    assert_eq!(m.burns(5).len(), 1);
    assert_eq!(m.public().hidden_count, 13);
    assert_eq!(count_events(&m, "AceUndealt"), 0);
}

#[test]
fn turned_joker_then_ace_of_hearts() {
    let d = deck(6, ["", "", "", "", "", "", "", ""], "JK AH", "");
    let mut m = Match::new(6, Rules::default());
    m.start_hand(&d);
    act_first(&mut m);
    assert_eq!(m.public().trump, Some(2));
    assert_eq!(m.phase(), Phase::AwaitBurn(6));
    assert_eq!(public_events(&m).last().unwrap(), "Took seat=6 card=AH");
}

// --------------------------------------------------------------------------
// Illegal actions are impossible (H1)
// --------------------------------------------------------------------------

#[test]
fn illegal_actions_are_rejected_and_change_nothing() {
    let mut m = Match::new(7, Rules::default());
    assert!(m.apply(Action::Play(0)).is_err()); // no deck yet

    // Seat 0 must burn (it holds the AC); seat 1 holds the AD.
    let d = deck(7, ["AC", "AD", "", "", "", "", "", ""], "9C", "");
    m.start_hand(&d);
    let snapshot = |m: &Match| (m.public(), m.events.len(), m.hand(0), m.burns(0));
    let before = snapshot(&m);
    assert!(m.apply(Action::Play(idx("2C"))).is_err()); // a burn is due
    assert!(m.apply(Action::Burn(idx("AC"))).is_err()); // H3
    assert!(m.apply(Action::Burn(idx("AD"))).is_err()); // not seat 0's card
    assert!(m.apply(Action::Burn(53)).is_err()); // not a card
    assert_eq!(snapshot(&m), before);

    m.apply(Action::Burn(idx("9C"))).unwrap();
    let before = snapshot(&m);
    assert!(m.apply(Action::Burn(idx("2C"))).is_err()); // no burn is due
    assert!(m.apply(Action::Play(idx("AD"))).is_err()); // seat 1's card
    assert_eq!(snapshot(&m), before);
}

#[test]
fn not_following_suit_is_rejected() {
    // Seat 1 holds a diamond and a spade; diamonds are led.
    let d = deck(
        7,
        ["KD", "2D 3S 4S 6S 7S", "", "", "", "", "", ""],
        "9C",
        "AC",
    );
    let mut m = Match::new(7, Rules::default());
    m.start_hand(&d);
    m.apply(Action::Play(idx("KD"))).unwrap();
    assert_eq!(legal_names(&m), "2D");
    let events_before = m.events.len();
    let err = m.apply(Action::Play(idx("3S"))).unwrap_err();
    assert!(err.contains("rules 14-18"), "{err}");
    assert_eq!(m.events.len(), events_before);
    assert_eq!(m.phase(), Phase::AwaitPlay(1));
}

#[test]
#[should_panic(expected = "no deck is needed")]
fn start_hand_twice_panics() {
    let mut m = Match::new(0, Rules::default());
    m.start_hand(&identity_deck());
    m.start_hand(&identity_deck());
}

#[test]
#[should_panic(expected = "twice")]
fn start_hand_rejects_a_bad_deck() {
    let mut m = Match::new(0, Rules::default());
    let mut d = identity_deck();
    d[5] = 6;
    m.start_hand(&d);
}

// --------------------------------------------------------------------------
// Scoring (rules 27-31) and the bonus switch
// --------------------------------------------------------------------------

/// Three hands of one game, each with the 5 of trumps led or played in trick 1
/// by a known team: team 0, team 0 again, then team 1. Returns the bonus lines.
fn three_bonus_hands(rules: Rules) -> (Match, Vec<String>) {
    let prefer = ["5C", "5D", "5S"];
    let mut m = Match::new(7, rules);

    // Team 0 holds only low plain cards besides the 5, so the game goes on.
    // Hand 0, dealer 7: seat 0 (team 0) leads the 5C.
    let hands = ["5C 2D 3D 4D 6D", "", "", "", "2H 3H 4H 6H 7H", "", "", ""];
    m.start_hand(&deck(7, hands, "9C", ""));
    for _ in 0..8 {
        act_preferring(&mut m, &prefer);
    }
    assert_eq!(m.public().trick_winners, vec![0]);
    assert_eq!(m.scores()[0], 5 + rules.bonus_first);
    assert!(m.bonus_taken()[0]);
    finish_hand_preferring(&mut m, &prefer);
    assert_eq!(m.game_index(), 0, "the game must still be on");
    let after_hand_0 = m.scores();

    // Hand 1, dealer 0: seat 4 (team 0) plays the 5D in trick 1.
    let hands = ["10S 9S 8S 7S 6S", "", "", "", "5D 2H 3H 4H 6H", "", "", ""];
    m.start_hand(&deck(0, hands, "9D", ""));
    for _ in 0..8 {
        act_preferring(&mut m, &prefer);
    }
    assert_eq!(m.public().trick_winners, vec![4]);
    assert_eq!(m.scores()[0], after_hand_0[0] + 5 + rules.bonus_later);
    finish_hand_preferring(&mut m, &prefer);
    assert_eq!(m.game_index(), 0, "the game must still be on");
    assert!(!m.bonus_taken()[1]);

    // Hand 2, dealer 1: seat 5 (team 1) plays the 5S; team 1's first bonus.
    m.start_hand(&deck(1, ["", "", "", "", "", "5S", "", ""], "9S", ""));
    for _ in 0..8 {
        act_preferring(&mut m, &prefer);
    }
    assert_eq!(m.public().trick_winners, vec![5]);

    let bonuses: Vec<String> = public_events(&m)
        .into_iter()
        .filter(|t| t.starts_with("Bonus"))
        .collect();
    (m, bonuses)
}

#[test]
fn bonus_is_11_then_10_under_the_house_rule() {
    let (m, bonuses) = three_bonus_hands(Rules::default());
    assert_eq!(
        bonuses,
        vec![
            "Bonus team=0 points=6 kind=immediate card=5C",
            "Bonus team=0 points=5 kind=immediate card=5D",
            "Bonus team=1 points=6 kind=immediate card=5S",
        ]
    );
    // A team has taken its bonus exactly when its score is 1 more than a multiple of 5.
    for t in 0..4 {
        assert_eq!(m.bonus_taken()[t], m.scores()[t] % 5 == 1);
    }
}

#[test]
fn flat_11_switch_gives_6_every_time() {
    let rules = Rules {
        bonus_later: 6,
        ..Rules::default()
    };
    let (_, bonuses) = three_bonus_hands(rules);
    assert_eq!(
        bonuses,
        vec![
            "Bonus team=0 points=6 kind=immediate card=5C",
            "Bonus team=0 points=6 kind=immediate card=5D",
            "Bonus team=1 points=6 kind=immediate card=5S",
        ]
    );
}

#[test]
fn end_of_hand_bonus_goes_to_the_highest_trump_played() {
    // Rule 30: the 5C is undealt, so the JC (seat 3) is the best trump played.
    let d = deck(7, ["", "", "", "JC", "", "", "", ""], "9C", "5C");
    let mut m = Match::new(7, Rules::default());
    m.start_hand(&d);
    finish_hand_preferring(&mut m, &[]);
    let events = public_events(&m);
    assert_in_order(
        &events,
        &[
            "TrickWon trick=5",
            "Bonus team=3 points=6 kind=end card=JC",
            "HandEnd hand=0 aborted=false",
        ],
    );
    assert_eq!(count_events(&m, "Bonus"), 1);
    assert!(m.bonus_taken()[3]);
    assert_eq!(m.public().best_trump, Some((idx("JC"), 3)));
    assert!(!m.public().five_played);
}

// --------------------------------------------------------------------------
// Ending a game (rules 32-33)
// --------------------------------------------------------------------------

#[test]
fn game_ends_the_instant_a_team_reaches_41_mid_hand() {
    // Team 2 sits on 40 and seat 2 wins trick 1 with the JC (5C undealt).
    let d = deck(7, ["", "", "JC", "", "", "", "", ""], "9C", "5C");
    let mut m = Match::new(7, Rules::default());
    m.scores[2] = 40;
    m.start_hand(&d);
    for _ in 0..8 {
        act_preferring(&mut m, &["JC"]);
    }
    let events = public_events(&m);
    let tail: Vec<&str> = events
        .iter()
        .rev()
        .take(3)
        .rev()
        .map(|s| s.as_str())
        .collect();
    assert_eq!(
        tail,
        vec![
            "TrickWon trick=1 seat=2 team=2 points=5",
            "GameWon team=2 reason=trick scores=0,0,45,0",
            "HandEnd hand=0 aborted=true",
        ]
    );
    // The rest of the hand is not played; no AceUndealt, no pending bonus.
    assert_eq!(count_events(&m, "AceUndealt"), 0);
    assert_eq!(count_events(&m, "Bonus"), 0);
    assert!(m.needs_deck());
    assert!(m.legal_actions().is_empty());
    assert!(m.apply(Action::Play(m.hand(3)[0])).is_err());
    assert_eq!(m.games_won(), [0, 0, 1, 0]);
    assert_eq!(m.scores(), [0, 0, 0, 0]);
    assert_eq!(m.bonus_taken(), [false; 4]);
    assert_eq!(m.game_index(), 1);
    assert_eq!(m.hand_index(), 1);
    assert_eq!(m.dealer(), 0); // H5: the deal moves one seat left

    m.start_hand(&identity_deck());
    assert_eq!(
        public_events(&m)[events.len()],
        "HandStart hand=1 game=1 dealer=0"
    );
}

#[test]
fn game_ends_by_the_immediate_bonus() {
    // Team 2 on 30: the 5C trick makes 35, its bonus makes 41.
    let d = deck(7, ["", "", "5C", "", "", "", "", ""], "9C", "");
    let mut m = Match::new(7, Rules::default());
    m.scores[2] = 30;
    m.start_hand(&d);
    for _ in 0..8 {
        act_preferring(&mut m, &["5C"]);
    }
    assert_in_order(
        &public_events(&m),
        &[
            "TrickWon trick=1 seat=2 team=2 points=5",
            "Bonus team=2 points=6 kind=immediate card=5C",
            "GameWon team=2 reason=immediate_bonus scores=0,0,41,0",
            "HandEnd hand=0 aborted=true",
        ],
    );

    // On 40 the trick alone reaches 41: the bonus is still scored, reason trick.
    let mut m = Match::new(7, Rules::default());
    m.scores[2] = 40;
    m.start_hand(&d);
    for _ in 0..8 {
        act_preferring(&mut m, &["5C"]);
    }
    assert_in_order(
        &public_events(&m),
        &[
            "TrickWon trick=1 seat=2 team=2 points=5",
            "Bonus team=2 points=6 kind=immediate card=5C",
            "GameWon team=2 reason=trick scores=0,0,51,0",
        ],
    );
}

#[test]
fn game_ends_by_the_end_of_hand_bonus() {
    // A dry run tells us how many tricks team 3 (holding the JC, the best
    // trump dealt) takes; we then seat them on 35 minus those points.
    let d = deck(7, ["", "", "", "JC", "", "", "", ""], "9C", "5C");
    let mut dry = Match::new(7, Rules::default());
    dry.start_hand(&d);
    finish_hand_preferring(&mut dry, &[]);
    let t = dry.tricks_won[3];
    assert!((1..5).contains(&t));

    let mut m = Match::new(7, Rules::default());
    m.scores[3] = 35 - 5 * t;
    m.start_hand(&d);
    finish_hand_preferring(&mut m, &[]);
    let events = public_events(&m);
    let tail: Vec<&str> = events
        .iter()
        .rev()
        .take(3)
        .rev()
        .map(|s| s.as_str())
        .collect();
    assert_eq!(tail[0], "Bonus team=3 points=6 kind=end card=JC");
    assert_eq!(tail[2], "HandEnd hand=0 aborted=false");
    assert!(tail[1].starts_with("GameWon team=3 reason=end_bonus scores="));
    assert!(tail[1].ends_with(",41"));
    assert_eq!(m.scores(), [0; 4]);
    assert_eq!(m.games_won(), [0, 0, 0, 1]);
}

#[test]
fn trick_winner_at_41_beats_a_pending_end_bonus() {
    // Owner default: after trick 5 the trick's points come first; a pending
    // end-of-hand bonus for another team is lost.
    let d = deck(7, ["", "", "", "JC", "", "", "", ""], "9C", "5C");
    let mut dry = Match::new(7, Rules::default());
    dry.start_hand(&d);
    finish_hand_preferring(&mut dry, &[]);
    let winners: Vec<u8> = dry.finished_tricks.iter().map(|(_, w)| *w).collect();
    let last_team = team_of(winners[4]);
    assert_ne!(last_team, 3, "pick a deck where another team wins trick 5");
    let before_last = winners[..4]
        .iter()
        .filter(|w| team_of(**w) == last_team)
        .count() as u8;
    let t3 = dry.tricks_won[3];

    let mut m = Match::new(7, Rules::default());
    m.scores[last_team] = 40 - 5 * before_last;
    m.scores[3] = 35 - 5 * t3;
    m.start_hand(&d);
    finish_hand_preferring(&mut m, &[]);
    let events = public_events(&m);
    let game_won = events.iter().find(|t| t.starts_with("GameWon")).unwrap();
    assert!(game_won.starts_with(&format!("GameWon team={last_team} reason=trick")));
    assert_eq!(count_events(&m, "Bonus"), 0);
}

/// A deck in which `jinker` holds the five best clubs (clubs trumps, the 3C turned).
fn jink_deck(dealer: u8, jinker: u8) -> [u8; 53] {
    let mut hands = [""; 8];
    hands[jinker as usize] = "5C JC JK AH AC";
    deck(dealer, hands, "3C", "")
}

/// The jinker burns the turned 3C and plays its best trump; others play low.
fn jink_policy(m: &mut Match) {
    if let Phase::AwaitBurn(_) = m.phase() {
        m.apply(Action::Burn(idx("3C"))).unwrap();
    } else {
        act_preferring(m, &["5C", "JC", "JK", "AH", "AC"]);
    }
}

#[test]
fn jink_wins_the_game_whatever_the_score() {
    let mut m = Match::new(7, Rules::default());
    m.start_hand(&jink_deck(7, 0));
    assert_eq!(m.phase(), Phase::AwaitBurn(0));
    assert_eq!(legal_burn_names(&m), "3C 5C JC JK");
    while !m.needs_deck() {
        jink_policy(&mut m);
    }
    assert_in_order(
        &public_events(&m),
        &[
            "Played seat=0 card=5C",
            "TrickWon trick=1 seat=0 team=0 points=5",
            "Bonus team=0 points=6 kind=immediate card=5C",
            "TrickWon trick=2 seat=0 team=0 points=5",
            "TrickWon trick=3 seat=0 team=0 points=5",
            "TrickWon trick=4 seat=0 team=0 points=5",
            "Played seat=0 card=AC",
            "TrickWon trick=5 seat=0 team=0 points=5",
            "GameWon team=0 reason=jink scores=31,0,0,0",
            "HandEnd hand=0 aborted=false",
        ],
    );
    assert_eq!(m.games_won(), [1, 0, 0, 0]);
    assert_eq!(m.game_index(), 1);
    assert_eq!(m.dealer(), 0);
}

#[test]
fn jink_by_a_seat_that_does_not_lead() {
    // Seat 6 is fourth to play in trick 1, robs there, wins with the 5C and leads on.
    let mut m = Match::new(1, Rules::default());
    m.start_hand(&jink_deck(1, 6));
    while !m.needs_deck() {
        jink_policy(&mut m);
    }
    assert_eq!(m.public().trick_winners, vec![6; 5]);
    assert_eq!(m.games_won(), [0, 0, 1, 0]);
    assert!(public_events(&m).contains(&"GameWon team=2 reason=jink scores=0,0,31,0".to_string()));
}

// --------------------------------------------------------------------------
// The match and the money (rules 35-36, H6)
// --------------------------------------------------------------------------

#[test]
fn five_game_match_and_money() {
    // Each game is a jink by the seat left of the dealer: teams 0, 1, 2, 3, then 0.
    let mut m = Match::new(7, Rules::default());
    for game in 0..5u8 {
        assert!(m.needs_deck());
        assert_eq!(m.game_index(), game);
        assert_eq!(m.hand_index(), game as u32);
        let dealer = m.dealer();
        assert_eq!(dealer, (7 + game) % 8);
        m.start_hand(&jink_deck(dealer, (dealer + 1) % 8));
        assert_eq!(m.scores(), [0; 4]); // rule 36
        while !m.needs_deck() && !m.match_over() {
            jink_policy(&mut m);
        }
        assert_eq!(m.money().is_some(), game == 4);
    }
    assert!(m.match_over());
    assert!(!m.needs_deck());
    assert_eq!(m.phase(), Phase::MatchOver);
    assert_eq!(m.to_act(), None);
    assert!(m.legal_actions().is_empty());
    assert!(m.apply(Action::Play(0)).is_err());
    assert_eq!(m.games_won(), [2, 1, 1, 1]);
    // Each winner receives three stakes, each loser pays one: per team +6 and -2 stakes.
    assert_eq!(m.money(), Some([30, -10, -10, -10]));
    let events = public_events(&m);
    assert_eq!(
        events.last().unwrap(),
        "MatchWon team=0 money=30,-10,-10,-10"
    );
    assert_eq!(count_events(&m, "GameWon"), 5);
    assert_eq!(count_events(&m, "MatchStart"), 1);
    // The match ends where it stands: no dealer move, no new hand.
    assert_eq!(m.dealer(), 3);
    assert_eq!(m.hand_index(), 4);
    assert_eq!(m.game_index(), 4);
    assert_eq!(m.scores(), [31, 0, 0, 0]);
}

#[test]
fn two_nil_match_with_another_stake() {
    let rules = Rules {
        stake: 2,
        ..Rules::default()
    };
    let mut m = Match::new(1, rules);
    m.start_hand(&jink_deck(1, 6));
    while !m.needs_deck() {
        jink_policy(&mut m);
    }
    assert_eq!(m.dealer(), 2);
    m.start_hand(&jink_deck(2, 2));
    while !m.match_over() {
        jink_policy(&mut m);
    }
    assert_eq!(m.games_won(), [0, 0, 2, 0]);
    assert_eq!(m.money(), Some([-4, -4, 12, -4]));
    let total: i32 = m.money().unwrap().iter().sum();
    assert_eq!(total, 0);
    assert_in_order(
        &public_events(&m),
        &[
            "GameWon team=2 reason=jink scores=0,0,31,0",
            "HandEnd hand=1 aborted=false",
            "MatchWon team=2 money=-4,-4,12,-4",
        ],
    );
}

// --------------------------------------------------------------------------
// Random matches: invariants that must hold after every action
// --------------------------------------------------------------------------

fn check_invariants(m: &Match) {
    let p = m.public();
    // Every card is in exactly one place.
    let mut seen = [0u8; 53];
    for seat in 0..8 {
        for c in m.hand(seat) {
            seen[c as usize] += 1;
        }
        for c in m.burns(seat) {
            seen[c as usize] += 1;
            assert!(!Card::from_index(c).unwrap().is_ace(), "an ace was burned");
        }
    }
    for c in m.undealt() {
        seen[c as usize] += 1;
    }
    for &c in &p.played {
        seen[c as usize] += 1;
    }
    if let Some(c) = p.table_card {
        seen[c as usize] += 1;
    }
    assert!(seen.iter().all(|n| *n == 1), "card conservation broken");

    assert!(p.hidden_count == 12 || p.hidden_count == 13);
    assert!(p.burn_counts.iter().sum::<u8>() <= 2);
    if let Phase::AwaitBurn(_) | Phase::AwaitPlay(_) = p.phase {
        // Only the dealer can burn twice in one hand.
        for seat in 0..8u8 {
            if seat != m.dealer() {
                assert!(p.burn_counts[seat as usize] <= 1);
            }
        }
    }
    let legal = m.legal_actions();
    assert!(legal.windows(2).all(|w| w[0].card() < w[1].card()));
    match p.phase {
        Phase::AwaitBurn(s) => {
            assert_eq!(p.hand_sizes[s as usize], 6);
            assert!(!legal.is_empty());
        }
        Phase::AwaitPlay(_) => assert!(!legal.is_empty()),
        Phase::NeedsDeck | Phase::MatchOver => assert!(legal.is_empty()),
    }
    if m.rules.bonus_later == 5 {
        for t in 0..4 {
            assert_eq!(m.bonus_taken()[t], m.scores()[t] % 5 == 1);
        }
    }
    for t in 0..4 {
        assert!(m.scores()[t] <= 51);
    }
}

fn run_random_matches(rules: Rules, matches: u64) -> [usize; 8] {
    // Coverage counters: Joker turned, ace turned (first card), non-dealer
    // robbed, ace undealt, end bonus, immediate bonus, game ended mid-hand, jink.
    let mut seen = [0usize; 8];
    for seed in 0..matches {
        let mut rng = TestRng::new(seed);
        let first = rng.below(8) as u8;
        let mut m = Match::new(first, rules);
        let mut guard = 0;
        while !m.match_over() {
            guard += 1;
            assert!(guard < 100_000);
            if m.needs_deck() {
                assert_eq!(m.dealer() as u32, (first as u32 + m.hand_index()) % 8);
                let d = rng.deck();
                m.start_hand(&d);
            } else {
                let legal = m.legal_actions();
                let a = legal[rng.below(legal.len())];
                m.apply(a).unwrap();
            }
            check_invariants(&m);
        }
        let money = m.money().unwrap();
        assert_eq!(money.iter().sum::<i32>(), 0);
        let winner = (0..4).find(|t| m.games_won()[*t] == 2).unwrap();
        for (t, paid) in money.iter().enumerate() {
            let want = if t == winner {
                6 * rules.stake
            } else {
                -2 * rules.stake
            };
            assert_eq!(*paid, want);
            assert!(m.games_won()[t] <= 2);
        }
        // Private events reach only their owner.
        for seat in 0..8u8 {
            for line in m.events_for(seat) {
                if line.starts_with("Dealt") || line.starts_with("Burned") {
                    assert!(
                        line.starts_with(&format!("Dealt seat={seat} "))
                            || line.starts_with(&format!("Burned seat={seat} "))
                    );
                }
            }
        }
        let all = public_events(&m);
        for (i, line) in all.iter().enumerate() {
            if line == "Turned card=JK" {
                seen[0] += 1;
            }
            if line.starts_with("Turned card=A") && !all[i - 1].starts_with("BurnDone") {
                seen[1] += 1;
            }
            if line.starts_with("Took") && !line.contains("card=JK") {
                let seat: u8 = line["Took seat=".len()..]
                    .split(' ')
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
                // The most recent HandStart names the dealer.
                let hs = all[..i]
                    .iter()
                    .rev()
                    .find(|l| l.starts_with("HandStart"))
                    .unwrap();
                let dealer: u8 = hs.rsplit('=').next().unwrap().parse().unwrap();
                if seat != dealer {
                    seen[2] += 1;
                }
            }
            if line == "AceUndealt" {
                seen[3] += 1;
            }
            if line.contains("kind=end") {
                seen[4] += 1;
            }
            if line.contains("kind=immediate") {
                seen[5] += 1;
            }
            if line.ends_with("aborted=true") {
                seen[6] += 1;
            }
            if line.contains("reason=jink") {
                seen[7] += 1;
            }
        }
    }
    seen
}

#[test]
fn random_matches_keep_every_invariant_house_rule() {
    let seen = run_random_matches(Rules::default(), 150);
    for (i, n) in seen.iter().enumerate() {
        assert!(*n > 0, "branch {i} never reached in random play: {seen:?}");
    }
}

#[test]
fn random_matches_keep_every_invariant_flat_rule() {
    let rules = Rules {
        bonus_later: 6,
        stake: 3,
        ..Rules::default()
    };
    run_random_matches(rules, 40);
}

#[test]
fn every_completed_hand_has_a_trump_played() {
    // With 8 players only 13 cards are ever out of play, while there are 14 or 15
    // trumps, so rule 31 (no trump played, no bonus) cannot happen in a full hand.
    for seed in 0..60u64 {
        let mut rng = TestRng::new(1000 + seed);
        let mut m = Match::new(rng.below(8) as u8, Rules::default());
        m.start_hand(&rng.deck());
        while !m.needs_deck() && !m.match_over() {
            let legal = m.legal_actions();
            m.apply(legal[rng.below(legal.len())]).unwrap();
        }
        assert!(m.public().best_trump.is_some());
        let jink = count_events(&m, "GameWon") == 1;
        if !jink {
            assert_eq!(count_events(&m, "Bonus"), 1);
        }
    }
}
