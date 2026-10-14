//! One named test per tracker fact F2–F12 on hand-made deals (docs/M3-SPEC.md §3.3), each
//! asserting the exact places of the cards concerned.

use engine::cards::{ALL_CARDS, cards};
use engine::game::{Match, Phase};
use engine::view::BurnKind::{AceRobAtOnce, JokerTake, RobAtTurn};

use super::{act, c, check_truth, cs, describe, hand_made};
use crate::Loc::{self, Burn, Hand, Undealt};
use crate::{Tracker, WorldSpace};

fn tracker(m: &Match, seat: u8) -> Tracker {
    let t = Tracker::from_view(&m.view(seat));
    check_truth(m);
    t
}

fn places(t: &Tracker, card: &str) -> Vec<Loc> {
    t.places(c(card))
}

/// Every hand but these seats', plus the undealt pack, in `Loc` order.
fn hands_but(seats: &[u8]) -> Vec<Loc> {
    let mut v: Vec<Loc> = (0..8).filter(|s| !seats.contains(s)).map(Hand).collect();
    v.push(Undealt);
    v
}

/// Deal B: dealer 0, spades trumps (the 9 turned), the ace of spades undealt, so nobody robs.
/// Seat 2 is void in clubs and holds the 5 of trumps.
fn deal_b() -> Match {
    hand_made(
        0,
        [
            "QH KH 2H 3H 4H",
            "2C 3C 4C 5D 6D",
            "7H 8H 9H 10H 5S",
            "7C 8C 2S 7D 8D",
            "9C 10C 3S 9D 10D",
            "JC QC 4S JD QD",
            "KC AC 6S KD AD",
            "5C 6C 7S 2D 3D",
        ],
        "9S AS",
    )
}

#[test]
fn f2_void_in_the_led_plain_suit() {
    let mut m = deal_b();
    act(&mut m, "P 2C, P 7H");
    let t = tracker(&m, 3);
    // Seat 2 discarded a heart on a club lead: it holds no club.
    for club in ["3C", "4C", "AC", "KC"] {
        assert_eq!(places(&t, club), hands_but(&[2, 3]), "{club}");
    }
    // Nothing else is learnt: hearts, diamonds and trumps may still be anywhere.
    assert_eq!(places(&t, "8H"), hands_but(&[3]));
    assert_eq!(places(&t, "4D"), hands_but(&[3]));
    assert_eq!(places(&t, "5S"), hands_but(&[3]));
    assert_eq!(
        t.hand_forbid(2),
        cs("AC 2C 3C 4C 5C 6C 7C 8C 9C 10C JC QC KC")
    );
}

#[test]
fn f2_trumping_or_following_reveals_nothing() {
    let mut m = deal_b();
    act(&mut m, "P 2C, P 5S, P 7C");
    let t = tracker(&m, 4);
    assert_eq!(t.hand_forbid(2), 0, "trumping a plain lead proves no void");
    assert_eq!(t.hand_forbid(3), 0, "following suit proves nothing");
    assert_eq!(places(&t, "AC"), hands_but(&[4]));
}

/// Deal C: dealer 0, spades trumps, the ace of spades undealt. Seat 1 leads the A♥ (a trump
/// lead when hearts are not trumps, rule 17). Seat 2 has no trump; seat 4's only trump is the
/// Joker, which ranks above the A♥, so it may hold it back (reneging, rule 16).
fn deal_c() -> Match {
    hand_made(
        0,
        [
            "QH KH 2H 3H 4H",
            "AH 2C 3C 4C 5D",
            "7H 8H 9H 10H 6D",
            "10S 8C 7C 7D 8D",
            "JK 10C 9C 9D 10D",
            "JC QC 4S JD QD",
            "KC AC 6S KD AD",
            "5C 6C 7S 2D 3D",
        ],
        "9S AS 8S JS QS KS 2S 3S 5S JH 6H 5H 4D",
    )
}

#[test]
fn f3_trump_limit() {
    let mut m = deal_c();
    act(&mut m, "P AH, P 7H, P 10S, P 9D");
    let t = tracker(&m, 5);
    // Seats 2 and 4 played a non-trump on the A♥ lead: they hold no trump below it.
    for low in ["8S", "QS", "KS", "2S", "3S", "6S", "7S"] {
        assert_eq!(places(&t, low), hands_but(&[2, 4, 5]), "{low}");
    }
    let limit = t.hand_forbid(2);
    assert_eq!(limit, t.hand_forbid(4));
    assert_eq!(
        limit,
        cs("AS 2S 3S 4S 6S 7S 8S 9S 10S QS KS AH"),
        "every trump but the 5, J and Joker (the A♥ led included)"
    );
}

#[test]
fn reneging_top_trumps_allowed() {
    let mut m = deal_c();
    act(&mut m, "P AH, P 7H, P 10S, P 9D");
    let t = tracker(&m, 5);
    // The 5, J and Joker rank above the A♥: a seat that played a plain card may still hold them.
    for top in ["5S", "JS", "JK"] {
        assert_eq!(places(&t, top), hands_but(&[5]), "{top}");
    }
}

#[test]
fn a_hearts_led_not_trumps() {
    let mut m = deal_c();
    act(&mut m, "P AH, P 7H");
    let t = tracker(&m, 3);
    // The A♥ led is a trump lead: seat 2's heart is not a follow, it proves a trump limit,
    // and nothing about plain hearts.
    assert_eq!(places(&t, "JH"), hands_but(&[3]));
    assert_eq!(places(&t, "KS"), hands_but(&[2, 3]));
    assert_eq!(places(&t, "JK"), hands_but(&[3]));
}

#[test]
fn f4_proved_no_ace_and_f9_pending_robbers() {
    let mut m = deal_c();
    act(&mut m, "P AH, P 7H, P 10S, P 9D");
    // Seat 5, about to play: seats 1-4 have played to trick 1 without robbing (F4); seats 6, 7
    // and the dealer may still hold the ace and rob at their turn (F9), or it is undealt.
    let t = tracker(&m, 5);
    assert_eq!(places(&t, "AS"), vec![Hand(0), Hand(6), Hand(7), Undealt]);
    // Seat 1's view while seat 5 is asked to play: seat 5 does not hold the ace either (the
    // engine would have had it rob first).
    let t1 = tracker(&m, 1);
    assert_eq!(places(&t1, "AS"), vec![Hand(0), Hand(6), Hand(7), Undealt]);
}

#[test]
fn f11_ace_undealt_after_trick_one() {
    let mut m = deal_c();
    act(&mut m, "P AH, P 7H, P 10S, P 9D, P 4S, P 6S, P 7S, P QH");
    assert_eq!(m.public().trick_no, 2);
    assert!(m.public().ace_undealt);
    for seat in 0..8 {
        let t = tracker(&m, seat);
        assert_eq!(places(&t, "AS"), vec![Undealt], "seat {seat}");
    }
}

/// Deal R: dealer 0, spades trumps, seat 3 holds the ace of spades and robs at its turn.
fn deal_r() -> Match {
    hand_made(
        0,
        [
            "QH KH 2H 3H 4H",
            "2C 3C 4C 5D 6D",
            "7H 8H 9H 10H 6H",
            "AS 7C AC 7D 8D",
            "9C 10C 3S 9D 10D",
            "JC QC 4S JD QD",
            "KC 8C 6S KD AD",
            "5C 6C 7S 2D 3D",
        ],
        "9S",
    )
}

#[test]
fn f6_rob_at_turn() {
    let mut m = deal_r();
    act(&mut m, "P 2C, P 7H");
    assert_eq!(m.phase(), Phase::AwaitBurn(3));
    // Already at its turn, before it burns, seat 3 is known to hold the ace and the 9.
    let t = tracker(&m, 4);
    assert_eq!(places(&t, "AS"), vec![Hand(3)]);
    assert_eq!(places(&t, "9S"), vec![Hand(3)]);
    act(&mut m, "B AC, P 7C");
    let t = tracker(&m, 4);
    assert_eq!(places(&t, "AS"), vec![Hand(3)]);
    assert_eq!(places(&t, "9S"), vec![Hand(3), Burn(3, RobAtTurn)]);
}

#[test]
fn robbed_ace_never_in_burn() {
    // F6: the robber's burn is never the ace of trumps.
    let mut m = deal_r();
    act(&mut m, "P 2C, P 7H, B AC, P 7C");
    let t = tracker(&m, 4);
    assert_eq!(t.may_hold(Burn(3, RobAtTurn)), t.unknown & !cs("AS"));
    // F5, F8: a turned ace robbed at once is in the dealer's hand; the dealer's burn is never it.
    let mut m = hand_made(0, [""; 8], "AD");
    assert_eq!(m.phase(), Phase::AwaitBurn(0));
    let burn = cards(m.truth().hands[0] & !cs("AD"))
        .next()
        .expect("a card");
    m.apply(engine::game::Action::Burn(burn)).expect("legal");
    let t = tracker(&m, 1);
    assert_eq!(places(&t, "AD"), vec![Hand(0)]);
    assert_eq!(t.may_hold(Burn(0, AceRobAtOnce)), t.unknown & !cs("AD"));
    // A turned Joker, then a turned ace: the ace is in the dealer's hand, the Joker in its hand
    // or either burn.
    let mut m = hand_made(0, ["JC 2C 3C 4C 5C", "", "", "", "", "", "", ""], "JK AD");
    act(&mut m, "B 2C, B 3C");
    let t = tracker(&m, 1);
    assert_eq!(places(&t, "AD"), vec![Hand(0)]);
    assert_eq!(
        places(&t, "JK"),
        vec![Hand(0), Burn(0, JokerTake), Burn(0, AceRobAtOnce)]
    );
    assert_eq!(t.may_hold(Burn(0, JokerTake)), t.unknown & !cs("AD"));
    assert_eq!(t.may_hold(Burn(0, AceRobAtOnce)), t.unknown & !cs("AD"));
}

#[test]
fn plain_ace_may_be_burned() {
    let mut m = deal_r();
    act(&mut m, "P 2C, P 7H, B AC, P 7C");
    let t = tracker(&m, 4);
    // The burned plain ace of clubs: anywhere a club may be (not seat 2, void in clubs),
    // including the robber's burn.
    assert_eq!(
        places(&t, "AC"),
        vec![
            Hand(0),
            Hand(1),
            Hand(3),
            Hand(5),
            Hand(6),
            Hand(7),
            Undealt,
            Burn(3, RobAtTurn)
        ]
    );
    // The A♥ (not the ace of trumps here) may be burned too.
    assert!(places(&t, "AH").contains(&Burn(3, RobAtTurn)));
}

#[test]
fn joker_burn_may_be_future_ace() {
    // F7: the dealer burns the ace of diamonds before trumps are known; the 9 of diamonds is
    // turned next. Nobody robs, and after trick 1 the ace is undealt or in that burn.
    let mut m = hand_made(3, ["", "", "", "AD", "", "", "", ""], "JK 9D");
    assert_eq!(m.phase(), Phase::AwaitBurn(3));
    // The dealer's own view while burning: trumps unknown, 12 undealt cards (one turned next).
    let t = tracker(&m, 3);
    assert_eq!(t.trump, None);
    assert_eq!(t.loc_index(Undealt).map(|i| t.locs[i].1), Some(12));
    for card in cards(t.unknown) {
        assert_eq!(t.places(card), hands_but(&[3]));
    }
    act(&mut m, "B AD");
    let t = tracker(&m, 4);
    assert_eq!(
        places(&t, "AD"),
        vec![
            Hand(0),
            Hand(1),
            Hand(2),
            Hand(3),
            Hand(5),
            Hand(6),
            Hand(7),
            Undealt,
            Burn(3, JokerTake)
        ]
    );
    assert_eq!(places(&t, "JK"), vec![Hand(3), Burn(3, JokerTake)]);
    // Trick 1 with no rob.
    while m.public().trick_no == 1 {
        let a = m.legal_actions()[0];
        m.apply(a).expect("legal");
    }
    for seat in [0, 4, 7] {
        let t = tracker(&m, seat);
        assert_eq!(
            places(&t, "AD"),
            vec![Undealt, Burn(3, JokerTake)],
            "seat {seat}"
        );
    }
    // The dealer itself knows where the ace is.
    assert_eq!(tracker(&m, 3).places(c("AD")), Vec::<Loc>::new());
}

#[test]
fn dealer_after_joker_robs_eighth() {
    // F10: a turned Joker, then the 9 of diamonds; the dealer (seat 3) holds the ace of
    // diamonds. Nothing is proved at once: it robs as the eighth card of trick 1.
    let mut m = hand_made(3, ["", "", "", "AD", "", "", "", ""], "JK 9D");
    let burn = cards(m.truth().hands[3] & !cs("AD JK"))
        .next()
        .expect("a card");
    m.apply(engine::game::Action::Burn(burn)).expect("legal");
    for _ in 0..7 {
        let seat = m.to_act().expect("to act");
        let t = tracker(&m, seat);
        // The dealer is not excluded until its own turn.
        assert!(places(&t, "AD").contains(&Hand(3)), "seat {seat}");
        assert!(places(&t, "AD").contains(&Burn(3, JokerTake)));
        let a = m.legal_actions()[0];
        m.apply(a).expect("legal");
    }
    assert_eq!(
        m.phase(),
        Phase::AwaitBurn(3),
        "the dealer robs as the eighth card"
    );
    let t = tracker(&m, 0);
    assert_eq!(places(&t, "AD"), vec![Hand(3)]);
    assert_eq!(places(&t, "9D"), vec![Hand(3)]);
    let b = m.legal_actions()[0];
    m.apply(b).expect("legal");
    let t = tracker(&m, 0);
    assert_eq!(places(&t, "AD"), vec![Hand(3)]);
    assert_eq!(places(&t, "9D"), vec![Hand(3), Burn(3, RobAtTurn)]);
    assert_eq!(
        places(&t, "JK"),
        vec![Hand(3), Burn(3, JokerTake), Burn(3, RobAtTurn)]
    );
}

#[test]
fn f12_hidden_out_of_play_counts() {
    // No rob: 12 undealt. A rob: 12 + 1 burn. Joker, no rob: 11 + 1. Joker then a rob: 11 + 2.
    let mut m = deal_c();
    let t = tracker(&m, 1);
    assert_eq!(
        (t.hidden_count, t.locs.last().copied()),
        (12, Some((Undealt, 12)))
    );
    let mut r = deal_r();
    act(&mut r, "P 2C, P 7H, B AC");
    let t = tracker(&r, 4);
    assert_eq!(t.hidden_count, 13);
    assert_eq!(
        &t.locs[t.locs.len() - 2..],
        &[(Undealt, 12), (Burn(3, RobAtTurn), 1)]
    );
    let mut j = hand_made(3, ["", "", "", "", "", "", "", ""], "JK 9D");
    let burn = cards(j.truth().hands[3] & !cs("JK"))
        .next()
        .expect("a card");
    j.apply(engine::game::Action::Burn(burn)).expect("legal");
    let t = tracker(&j, 4);
    assert_eq!(t.hidden_count, 12);
    assert_eq!(
        &t.locs[t.locs.len() - 2..],
        &[(Undealt, 11), (Burn(3, JokerTake), 1)]
    );
    // The capacities always hold exactly the unknown cards.
    for t in [tracker(&m, 2), tracker(&r, 2), tracker(&j, 2)] {
        assert_eq!(t.capacity_sum(), u32::from(engine::cards::count(t.unknown)));
        assert_eq!(t.unknown | t.known, ALL_CARDS);
    }
    act(&mut m, "P AH");
    assert_eq!(tracker(&m, 2).hidden_count, 12);
}

#[test]
fn counts_on_hand_made_deals_are_exact() {
    // Seat 3's view after "P 2C, P 7H, B AC, P 7C" in deal R, counted by hand: seat 3 holds the
    // ace and the 9 of spades or burned the 9; the club void of seat 2 limits the clubs.
    let mut m = deal_r();
    act(&mut m, "P 2C, P 7H, B AC, P 7C");
    let t = tracker(&m, 4);
    let ws = WorldSpace::new(&t).expect("feasible");
    let unpruned = WorldSpace::new_unpruned(&t).expect("feasible");
    assert_eq!(ws.count(), unpruned.count());
    assert!(ws.count() > 1u128 << 90, "{}", ws.count());
    // At the dealer's pre-trump burn every unknown card may be anywhere: the count is the
    // multinomial 47! / (5!^7 · 12!).
    let j = hand_made(3, [""; 8], "JK 9D");
    let t = tracker(&j, 3);
    let ws = WorldSpace::new(&t).expect("feasible");
    let mut n: u128 = 1;
    let mut left = 47u8;
    for cap in [5u8, 5, 5, 5, 5, 5, 5, 12] {
        n *= crate::space::binom(left, cap);
        left -= cap;
    }
    assert_eq!(ws.count(), n, "{}", describe(&j, &t));
}
