//! The leakage tests of the M1 gate (docs/M1-SPEC.md §3.3).
//!
//! The deterministic bots here are the test-only ones in `common` (a seeded random-legal bot and
//! a lowest-card bot); the property does not depend on how a bot plays, only on it being a
//! deterministic function of its seed and views.

mod common;

use bot::Bot;
use common::{bot_for_seat, mixed_bots, random_bots};
use engine::cards::{Bits, Card, bit};
use engine::game::{Event, Match, Phase, Rules, block_deck, block_first_dealer};
use engine::rng::Rng;
use engine::view::PublicState;
use serde_json::Value;
use std::cell::RefCell;
use std::rc::Rc;
use std::thread;
use tablemgr::leak::{
    check_event_windows, current_hand_view, hidden_from, json_cards, truncate_json_events,
    view_cards,
};
use tablemgr::protocol::encode_table;
use tablemgr::{
    BlockSpec, MemLink, SeatLink, TableConfig, TableMsg, build_match, mem_link_pair, run_seat,
    seat_seed, serve_match,
};

// ----- 1. The undealt pack never influences anything -----

/// The block's deck for hand `k` with the undealt pack beyond the second turned card (deck
/// positions 42..=52) permuted, and nothing else changed.
fn permuted_deck(block_seed: u64, k: u32) -> [Card; 53] {
    let mut d = block_deck(block_seed, k);
    let original = d;
    Rng::derive(block_seed, &[0x9E2, u64::from(k)]).shuffle(&mut d[42..]);
    if d == original {
        d[42..].rotate_left(1);
    }
    d
}

fn start_all(bots: &mut [Box<dyn Bot>; 8], match_seed: u64, rules: Rules) {
    for (s, b) in (0u8..).zip(bots.iter_mut()) {
        b.start_match(s, rules, seat_seed(match_seed, s));
    }
}

#[test]
fn permuting_the_undealt_pack_changes_no_action_and_no_view() {
    for i in 0..200u64 {
        let block_seed = Rng::derive(0x1EA4, &[i]).next_u64();
        let match_seed = Rng::derive(0x1EA5, &[i]).next_u64();
        let rules = if i.is_multiple_of(3) {
            Rules::FLAT
        } else {
            Rules::HOUSE
        };
        let dealer = block_first_dealer(block_seed);
        let mut a = Match::new_block(
            block_seed,
            move |k| block_deck(block_seed, k),
            dealer,
            rules,
        );
        let mut b = Match::new_block(
            block_seed,
            move |k| permuted_deck(block_seed, k),
            dealer,
            rules,
        );
        let (mut bots_a, mut bots_b) = if i.is_multiple_of(2) {
            (mixed_bots(), mixed_bots())
        } else {
            (random_bots(), random_bots())
        };
        start_all(&mut bots_a, match_seed, rules);
        start_all(&mut bots_b, match_seed, rules);
        let mut hands_checked = 0u32;
        let mut last_hand = None;
        loop {
            assert_eq!(a.phase(), b.phase(), "match {i}");
            // The decks differ only in the undealt pack beyond the second turned card.
            let (da, db) = (a.truth().deck, b.truth().deck);
            assert_eq!(da[..42], db[..42]);
            assert_ne!(
                da[42..],
                db[42..],
                "match {i}: the permutation must change the deck"
            );
            // Everything any seat can see is identical: the public state, every hand and burn,
            // the undealt set, and the whole event log with its visibility tags (each seat's
            // view is a function of these).
            assert_eq!(a.public(), b.public(), "match {i}");
            assert_eq!(a.truth().hands, b.truth().hands);
            assert_eq!(a.truth().burns, b.truth().burns);
            assert_eq!(a.truth().undealt, b.truth().undealt);
            assert_eq!(a.events(), b.events(), "match {i}");
            if last_hand != Some(a.hand_index()) {
                last_hand = Some(a.hand_index());
                hands_checked += 1;
                for s in 0..8 {
                    assert_eq!(a.view(s), b.view(s), "match {i}, seat {s}");
                }
            }
            let Some(seat) = a.to_act() else { break };
            let (va, vb) = (a.view(seat), b.view(seat));
            assert_eq!(va, vb, "match {i}: the acting seat's view");
            let (la, lb) = (a.legal_actions(), b.legal_actions());
            assert_eq!(la, lb);
            let s = usize::from(seat);
            let xa = bots_a[s].act(&va, &la);
            let xb = bots_b[s].act(&vb, &lb);
            assert_eq!(xa, xb, "match {i}: seat {seat} acted differently");
            a.apply(xa).unwrap();
            b.apply(xb).unwrap();
        }
        assert_eq!(a.phase(), Phase::MatchOver);
        assert_eq!(a.actions(), b.actions());
        assert_eq!(a.money(), b.money());
        assert!(hands_checked >= 2);
    }
}

// ----- 2. A seat's view never names a hidden card -----

/// Check one seat's view of the match as it stands: the hand in progress must name no card hidden
/// from that seat, by set arithmetic on the bitboards and, if `json` is set, also by an
/// independent walk of the view's JSON (which must agree with the struct walk).
fn check_view(m: &Match, public: &PublicState, seat: u8, json: bool, label: &str) {
    let view = m.view(seat);
    let current = current_hand_view(&view);
    let hidden = hidden_from(seat, m.truth(), public);
    let named = view_cards(&current, 0);
    assert_eq!(
        named & hidden,
        0,
        "{label}: seat {seat}'s view names hidden cards"
    );
    // A control: the seat does see its own hand.
    assert_eq!(view.hand & !named, 0);
    if json {
        let text = serde_json::to_string(&current).unwrap();
        let value: Value = serde_json::from_str(&text).unwrap();
        let in_json = json_cards(&value).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(
            in_json, named,
            "{label}: the JSON and struct walks disagree"
        );
        assert_eq!(
            in_json & hidden,
            0,
            "{label}: seat {seat}'s JSON names hidden cards"
        );
    }
}

/// Play match `i` with test bots, checking views at every decision: every seat's view by set
/// arithmetic, and the acting seat's view (the one the table sends) also through its JSON. At
/// the end, every seat's final view both ways, and its whole event list hand by hand. Returns
/// the cards that were hidden from some seat at some point.
fn check_match_views(i: u64) -> Bits {
    let seed = Rng::derive(0x50E, &[i]).next_u64();
    let cfg = if i % 4 == 3 {
        TableConfig::in_block(seed, Rules::HOUSE, BlockSpec::new(seed ^ 0xB10C))
    } else {
        TableConfig::plain(
            seed,
            if i.is_multiple_of(2) {
                Rules::HOUSE
            } else {
                Rules::FLAT
            },
        )
    };
    let mut m = cfg.new_match();
    let mut bots = if i.is_multiple_of(2) {
        random_bots()
    } else {
        mixed_bots()
    };
    start_all(&mut bots, cfg.seed, cfg.rules);
    let mut hidden_seen: Bits = 0;
    let mut decision = 0;
    while let Some(to_act) = m.to_act() {
        let label = format!("match {i}, decision {decision}");
        let public = m.public();
        for seat in 0..8u8 {
            check_view(&m, &public, seat, seat == to_act, &label);
            hidden_seen |= hidden_from(seat, m.truth(), &public);
        }
        let view = m.view(to_act);
        let legal = m.legal_actions();
        let action = bots[usize::from(to_act)].act(&view, &legal);
        m.apply(action).unwrap();
        decision += 1;
    }
    let public = m.public();
    for seat in 0..8u8 {
        check_view(&m, &public, seat, true, &format!("match {i}, end"));
        check_event_windows(seat, &m.view(seat).events)
            .unwrap_or_else(|e| panic!("match {i}: {e}"));
    }
    hidden_seen
}

#[test]
fn views_never_name_another_seats_cards_or_the_undealt_pack() {
    // 100 matches, shared out over 4 threads (each match is checked on its own).
    const MATCHES: u64 = 100;
    const WORKERS: u64 = 4;
    let hidden_seen = thread::scope(|scope| {
        let workers: Vec<_> = (0..WORKERS)
            .map(|w| {
                scope.spawn(move || {
                    (w..MATCHES)
                        .step_by(WORKERS as usize)
                        .fold(0, |seen, i| seen | check_match_views(i))
                })
            })
            .collect();
        workers
            .into_iter()
            .fold(0, |seen, w| seen | w.join().unwrap())
    });
    // The check is not vacuous: every card was hidden from some seat at some point.
    assert_eq!(hidden_seen, engine::cards::ALL_CARDS);
}

#[test]
fn the_checks_catch_a_planted_leak() {
    let m = TableConfig::plain(42, Rules::HOUSE).new_match();
    let seat = 0u8;
    let other = m.truth().hands[3];
    let hidden = hidden_from(seat, m.truth(), &m.public());
    assert_ne!(other & hidden, 0);
    let one = bit(other.trailing_zeros() as Card);
    // Another seat's card slipped into the hand, or another seat's deal in the events.
    let mut leaky = current_hand_view(&m.view(seat));
    leaky.hand |= one;
    let value = serde_json::to_value(&leaky).unwrap();
    assert_ne!(json_cards(&value).unwrap() & hidden, 0);
    assert_ne!(view_cards(&leaky, 0) & hidden, 0);
    let mut leaky = m.view(seat);
    leaky.events.push(Event::Dealt {
        seat: 3,
        cards: other,
    });
    assert!(check_event_windows(seat, &leaky.events).is_err());
    // A card name in a string, and an unclassified field, are caught too.
    let named = serde_json::json!({ "v": 1, "type": "abort", "reason": engine::cards::name(one.trailing_zeros() as Card) });
    assert_eq!(json_cards(&named).unwrap(), one);
    let unknown = serde_json::json!({ "v": 1, "secret": 17 });
    assert!(json_cards(&unknown).is_err());
}

// ----- 3. The bytes sent to a seat are its view and the legal actions, nothing more -----

/// The table's end of a link, keeping a copy of every line sent to the seat.
struct Tap {
    inner: MemLink,
    sent: Rc<RefCell<Vec<String>>>,
}

impl SeatLink for Tap {
    fn send(&mut self, line: &str) {
        self.sent.borrow_mut().push(line.to_string());
        self.inner.send(line);
    }
    fn recv(&mut self) -> Option<String> {
        self.inner.recv()
    }
}

fn keys(v: &Value) -> Vec<String> {
    v.as_object().unwrap().keys().cloned().collect()
}

#[test]
fn every_message_to_a_seat_is_exactly_its_view_and_the_legal_actions() {
    let configs = [
        TableConfig::plain(11, Rules::HOUSE),
        TableConfig::plain(12, Rules::FLAT),
        TableConfig::plain(13, Rules::HOUSE),
        TableConfig::in_block(14, Rules::HOUSE, BlockSpec::new(140)),
        TableConfig::in_block(15, Rules::HOUSE, BlockSpec::new(150)),
        TableConfig::in_block(16, Rules::FLAT, BlockSpec::new(160)),
    ];
    for cfg in configs {
        let sent: [Rc<RefCell<Vec<String>>>; 8] = core::array::from_fn(|_| Rc::default());
        let mut table_ends: Vec<Box<dyn SeatLink>> = Vec::new();
        let mut clients = Vec::new();
        for (s, sent) in sent.iter().enumerate() {
            let (table_end, mut seat_end) = mem_link_pair();
            table_ends.push(Box::new(Tap {
                inner: table_end,
                sent: Rc::clone(sent),
            }));
            clients.push(thread::spawn(move || {
                run_seat(&mut seat_end, bot_for_seat(s).as_mut())
            }));
        }
        let links: [Box<dyn SeatLink>; 8] = table_ends.try_into().ok().unwrap();
        let record = serve_match(&cfg, links).unwrap();
        for c in clients {
            c.join().unwrap().unwrap();
        }
        let sent: Vec<Vec<String>> = sent.iter().map(|s| s.borrow().clone()).collect();

        // hello: the seat, the rules and the seat seed; no card.
        for (s, lines) in sent.iter().enumerate() {
            let hello: Value = serde_json::from_str(&lines[0]).unwrap();
            assert_eq!(keys(&hello), ["rules", "seat", "seed", "type", "v"]);
            assert_eq!(hello["seed"], seat_seed(cfg.seed, s as u8));
            assert_eq!(json_cards(&hello).unwrap(), 0);
        }
        // act: rebuild the match from the record and compare every message, byte for byte.
        let mut m = build_match(cfg.seed, cfg.block, cfg.rules);
        let mut next = [1usize; 8];
        for &action in &record.actions {
            let seat = m.to_act().unwrap();
            let s = usize::from(seat);
            let line = &sent[s][next[s]];
            next[s] += 1;
            let view = m.view(seat);
            let legal = m.legal_actions();
            let expected = encode_table(&TableMsg::Act {
                view: Box::new(view.clone()),
                legal: legal.clone(),
            });
            assert_eq!(
                line, &expected,
                "seat {seat}: the message is not exactly its view"
            );
            let mut value: Value = serde_json::from_str(line).unwrap();
            assert_eq!(keys(&value), ["legal", "type", "v", "view"]);
            assert_eq!(value["view"], serde_json::to_value(&view).unwrap());
            assert_eq!(value["legal"], serde_json::to_value(&legal).unwrap());
            truncate_json_events(&mut value);
            let named = json_cards(&value).unwrap();
            assert_eq!(named & hidden_from(seat, m.truth(), &m.public()), 0);
            check_event_windows(seat, &view.events).unwrap();
            m.apply(action).unwrap();
        }
        assert_eq!(m.phase(), Phase::MatchOver);
        // end: the record, once the match is over; nothing else was sent.
        for (s, lines) in sent.iter().enumerate() {
            assert_eq!(
                lines.len(),
                next[s] + 1,
                "seat {s}: unexpected extra messages"
            );
            let end = &lines[next[s]];
            assert_eq!(
                end,
                &encode_table(&TableMsg::End {
                    record: Box::new(record.clone())
                })
            );
        }
    }
}
