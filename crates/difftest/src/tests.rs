//! Differential runs at test size, the corruption checks and forced rule branches.

use super::*;
use engine::cards::{ACE, Bits, JOKER, Suit, bit, card};
use engine::game::build_deck;

const HOUSE: Rules = Rules::HOUSE;

type Events = [Vec<String>; snapshot::SEATS];

fn assert_consistent(r: &Report, matches: u64) {
    assert_eq!(r.matches, matches);
    assert!(r.hands >= 2 * matches && r.games >= 2 * matches);
    assert_eq!(
        r.dealer_robbed_at_once + r.dealer_robbed_in_turn + r.non_dealer_robbed + r.nobody_robbed,
        r.hands,
        "every hand has exactly one rob outcome"
    );
    assert_eq!(
        r.won_by_trick + r.won_by_immediate_bonus + r.won_by_end_bonus + r.jink,
        r.games,
        "every game has exactly one end reason"
    );
    assert_eq!(r.match_wins.iter().sum::<u64>(), matches);
    assert_eq!(r.money.iter().sum::<i64>(), 0, "money sums to zero");
    assert_eq!(r.no_trump_hand, 0, "rule 31 is unreachable with 8 players");
    assert!(r.coverage_problems(false).is_empty());
}

#[test]
fn house_rule_200_matches() {
    let r = run_threads(1, 200, HOUSE, 4).unwrap_or_else(|d| panic!("{d}"));
    assert_consistent(&r, 200);
    let stake = i64::from(HOUSE.stake);
    for t in 0..4 {
        let w = r.match_wins[t] as i64;
        assert_eq!(r.money[t], 6 * stake * w - 2 * stake * (200 - w));
    }
}

#[test]
fn threads_give_the_same_report_as_one_thread() {
    let one = run(7, 12, HOUSE).unwrap_or_else(|d| panic!("{d}"));
    let three = run_threads(7, 12, HOUSE, 3).unwrap_or_else(|d| panic!("{d}"));
    assert_eq!(one, three);
    let alone = run_match(7, 5, HOUSE).unwrap_or_else(|d| panic!("{d}"));
    assert_eq!(alone.matches, 1);
}

#[test]
fn every_action_check_agrees_with_the_boundary_check() {
    for i in 0..6 {
        let ms = match_seed(8, i);
        let make = move || EMatch::new(ms, HOUSE);
        let fast = seeded_ctx(8, i, HOUSE, &make, None);
        let every = Ctx {
            check: Check::EveryAction,
            ..fast
        };
        let a = attempt(&fast).unwrap_or_else(|d| panic!("{d}"));
        let b = attempt(&every).unwrap_or_else(|d| panic!("{d}"));
        assert_eq!(a, b);
    }
}

/// Runs match 0 of seed 3 with the reference's side edited from action 25 on (a lasting
/// corruption, as a real bug would be), and checks where it is reported.
fn tampered(edit: &dyn Fn(&mut State, Option<&mut Events>)) -> Disagreement {
    let hook = |applied: u64, st: &mut State, ev: Option<&mut Events>| {
        if applied >= 25 {
            edit(st, ev);
        }
    };
    let mut report = Report::default();
    let err = run_seeded(3, 0, HOUSE, Some(&hook), &mut report)
        .expect_err("a corrupted snapshot must be reported");
    assert_eq!(report.matches, 0, "a failed match is not counted");
    assert_eq!((err.seed, err.match_index), (3, 0));
    assert_eq!(err.match_seed, match_seed(3, 0));
    assert_eq!(
        err.action_index, 25,
        "located at the first corrupted action"
    );
    assert!(err.last_action.is_some());
    assert!(
        err.reason.contains("located by rerunning"),
        "{}",
        err.reason
    );
    assert!(err.engine.contains("public.scores") && err.reference.contains("public.scores"));
    err
}

#[test]
fn corrupted_snapshot_is_reported() {
    // The same match untouched agrees.
    run_match(3, 0, HOUSE).unwrap_or_else(|d| panic!("{d}"));

    let d = tampered(&|st, _| st.scores[2] += 1);
    assert!(d.reason.contains("scores: engine"), "{}", d.reason);
    let text = d.to_string();
    assert!(text.starts_with("DISAGREEMENT") && text.contains("--- reference snapshot ---"));

    let d = tampered(&|st, _| st.hands[5].swap(0, 1));
    assert!(d.reason.contains("hand[5]: engine"), "{}", d.reason);

    let d = tampered(&|st, _| st.legal.truncate(1));
    assert!(d.reason.contains("legal: engine"), "{}", d.reason);

    let d = tampered(&|_, ev| {
        if let Some(ev) = ev {
            ev[6].insert(30, "AceUndealt".into());
        }
    });
    assert!(d.reason.contains("events seat 6"), "{}", d.reason);

    let d = tampered(&|_, ev| {
        if let Some(ev) = ev {
            ev[1][3] = "Turned card=JK".into();
        }
    });
    assert!(d.reason.contains("first difference at #3"), "{}", d.reason);
}

#[test]
fn event_corruption_is_caught_at_the_next_hand_boundary() {
    let hook = |applied: u64, _: &mut State, ev: Option<&mut Events>| {
        if let (true, Some(ev)) = (applied >= 25, ev) {
            ev[2].push("AceUndealt".into());
        }
    };
    let ms = match_seed(3, 0);
    let make = move || EMatch::new(ms, HOUSE);
    let fast = seeded_ctx(3, 0, HOUSE, &make, Some(&hook));
    let d = attempt(&fast).expect_err("caught by the full check at the next deal");
    assert!(d.action_index > 25, "{}", d.action_index);
    assert!(d.reason.contains("events seat 2"), "{}", d.reason);
    // At that point the reference has just been dealt a new hand.
    assert!(d.reference.contains("HandStart"));
}

#[test]
fn snapshot_differences_name_the_field() {
    let e = EMatch::new(11, HOUSE);
    let mut r = reference::Match::new(e.dealer(), reference_rules(HOUSE));
    r.start_hand(&e.truth().deck);
    let (a, mut b) = (Snapshot::from_engine(&e), Snapshot::from_reference(&r));
    assert_eq!(a, b);
    assert_eq!(State::from_engine_truth(&e), b.state);
    assert!(a.differences(&b, "engine", "reference").is_empty());
    b.state.public.hidden_count += 1;
    let d = a.differences(&b, "engine", "reference");
    assert_eq!(d.len(), 1);
    assert!(d[0].starts_with("public.hidden_count: engine"), "{}", d[0]);
}

#[test]
fn a_panic_becomes_a_located_disagreement() {
    let hook = |applied: u64, _: &mut State, _: Option<&mut Events>| {
        assert!(applied != 30, "deliberate test panic");
    };
    let mut report = Report::default();
    let d = run_seeded(4, 2, HOUSE, Some(&hook), &mut report).expect_err("must be reported");
    assert_eq!((d.seed, d.match_index, d.action_index), (4, 2, 30));
    assert!(
        d.reason
            .contains("a panic while building or comparing the snapshots")
            && d.reason.contains("deliberate test panic"),
        "{}",
        d.reason
    );
    // The rerun stopped just before the panic, so both states are shown.
    assert!(d.engine.contains("public.scores") && d.reference.contains("public.scores"));
}

#[test]
#[ignore = "prints a sample report; run with --ignored --nocapture"]
fn print_sample_disagreement() {
    let hook = |applied: u64, st: &mut State, ev: Option<&mut Events>| {
        if applied >= 40 {
            st.scores[1] += 5;
            if let Some(ev) = ev {
                ev[2].push("AceUndealt".into());
            }
        }
    };
    let mut report = Report::default();
    let d = run_seeded(3, 0, HOUSE, Some(&hook), &mut report).expect_err("must be reported");
    println!("{d}");
}

fn hands_with(seat: u8, c: u8) -> [Bits; 8] {
    let mut h = [0; 8];
    h[usize::from(seat)] = bit(c);
    h
}

fn forced(dealer: u8, hands: [Bits; 8], pack: &[u8]) -> Report {
    let deck = build_deck(dealer, &hands, pack).expect("a valid hand-made deal");
    run_decks(dealer, vec![deck], 99, HOUSE).unwrap_or_else(|d| panic!("{d}"))
}

#[test]
fn forced_branches_agree_and_are_counted() {
    let ad = card(Suit::Diamonds, ACE);
    let ah = card(Suit::Hearts, ACE);
    let as_ = card(Suit::Spades, ACE);
    let nine_d = card(Suit::Diamonds, 9);
    let nine_s = card(Suit::Spades, 9);
    let five_c = card(Suit::Clubs, 5);

    let r = forced(0, [0; 8], &[JOKER, ad]);
    assert!(r.turned_joker >= 1 && r.joker_then_ace >= 1 && r.dealer_robbed_at_once >= 1);

    let r = forced(3, hands_with(3, ad), &[JOKER, nine_d]);
    assert!(r.joker_dealer_holds_ace >= 1 && r.dealer_robbed_in_turn >= 1);

    let r = forced(6, [0; 8], &[JOKER, nine_d]);
    assert!(r.turned_joker >= 1);

    let r = forced(2, [0; 8], &[ah]);
    assert!(r.turned_ace_hearts >= 1 && r.dealer_robbed_at_once >= 1);

    let r = forced(5, hands_with(5, as_), &[nine_s]);
    assert!(r.dealer_robbed_in_turn >= 1);

    let r = forced(1, hands_with(4, card(Suit::Clubs, ACE)), &[five_c]);
    assert!(r.turned_five_of_trumps >= 1 && r.non_dealer_robbed >= 1);
}

#[test]
fn coverage_minimums_are_enforced() {
    let r = Report {
        matches: 100_000,
        hands: 10_000,
        turned_joker: 149, // 1.49%: below 1.5%
        joker_then_ace: 10,
        joker_dealer_holds_ace: 10,
        turned_ace_hearts: 150,
        turned_five_of_trumps: 600,
        ..Report::default()
    };
    assert!(r.coverage_problems(false).is_empty());
    let p = r.coverage_problems(true);
    assert_eq!(p.len(), 1, "{p:?}");
    assert!(p[0].starts_with("turned Joker"), "{p:?}");
    let impossible = Report {
        no_trump_hand: 1,
        ..Report::default()
    };
    assert_eq!(impossible.coverage_problems(false).len(), 1);
}
