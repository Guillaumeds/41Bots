//! The in-process table (docs/M1-SPEC.md §3.1) and the seed rules (§1).

mod common;

use bot::Bot;
use common::{Cheater, Lowest, RandomLegal, bot_for_seat, mixed_bots, not_legal};
use engine::cards::Bits;
use engine::game::{Action, Event, Rules, block_first_dealer};
use engine::view::SeatView;
use std::cell::RefCell;
use std::rc::Rc;
use tablemgr::{BlockSpec, TableConfig, TableError, play_match, seat_seed};

#[test]
fn seat_seeds_follow_the_rule() {
    // Computed independently in Python: Rng::derive(123, &[0xB07, s]).next_u64().
    const WANT: [u64; 8] = [
        16066758440918305308,
        3304646943393591322,
        8699200146071074695,
        16672933277956425669,
        8970299552862313468,
        11443118306351516124,
        458877298928148108,
        8021428075215322391,
    ];
    for s in 0..8u8 {
        assert_eq!(seat_seed(123, s), WANT[usize::from(s)], "seat {s}");
    }
}

/// What a watching bot saw.
#[derive(Default)]
struct Seen {
    starts: Vec<(u8, Rules, u64)>,
    decisions: usize,
    last_events: Vec<Event>,
}

/// Plays like the seat's test bot and checks that it is only ever asked about its own seat.
struct Watcher {
    inner: Box<dyn Bot>,
    seat: Option<u8>,
    seen: Rc<RefCell<Seen>>,
}

impl Bot for Watcher {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn start_match(&mut self, seat: u8, rules: Rules, seed: u64) {
        self.seat = Some(seat);
        self.seen.borrow_mut().starts.push((seat, rules, seed));
        self.inner.start_match(seat, rules, seed);
    }
    fn act(&mut self, view: &SeatView, legal: &[Action]) -> Action {
        assert_eq!(Some(view.seat), self.seat, "a view of another seat");
        assert_eq!(view.public.to_act, Some(view.seat), "asked when not to act");
        assert!(!legal.is_empty());
        let mut seen = self.seen.borrow_mut();
        seen.decisions += 1;
        seen.last_events = view.events.clone();
        self.inner.act(view, legal)
    }
}

/// A table of watched bots, and what each one saw.
type Watched = ([Box<dyn Bot>; 8], [Rc<RefCell<Seen>>; 8]);

fn watched_table(make: impl Fn(usize) -> Box<dyn Bot>) -> Watched {
    let seen: [Rc<RefCell<Seen>>; 8] = core::array::from_fn(|_| Rc::default());
    let bots = core::array::from_fn(|s| {
        Box::new(Watcher {
            inner: make(s),
            seat: None,
            seen: Rc::clone(&seen[s]),
        }) as Box<dyn Bot>
    });
    (bots, seen)
}

#[test]
fn bots_are_started_once_with_their_seat_seed_and_asked_only_for_their_seat() {
    for cfg in [
        TableConfig::plain(77, Rules::HOUSE),
        TableConfig::in_block(77, Rules::HOUSE, BlockSpec::new(5)),
    ] {
        let (mut bots, seen) = watched_table(bot_for_seat);
        let record = play_match(&cfg, &mut bots).unwrap();
        let mut decisions = 0;
        for (s, seen) in seen.iter().enumerate() {
            let seen = seen.borrow();
            assert_eq!(
                seen.starts,
                vec![(s as u8, cfg.rules, seat_seed(cfg.seed, s as u8))]
            );
            decisions += seen.decisions;
        }
        assert_eq!(decisions, record.actions.len(), "one action per decision");
        record.verify().unwrap();
        assert_eq!(record.bots[0], "test-random");
        assert_eq!(record.bots[1], "test-lowest");
        assert_eq!(record.rules, cfg.rules);
        assert_eq!(record.block, cfg.block);
    }
}

#[test]
fn the_same_config_and_bots_give_the_same_record() {
    for seed in 0..5u64 {
        let cfg = TableConfig::plain(seed, Rules::HOUSE);
        let a = play_match(&cfg, &mut mixed_bots()).unwrap();
        let b = play_match(&cfg, &mut mixed_bots()).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.to_json(), b.to_json());
    }
}

/// The cards dealt to each seat in each hand, read from the seat's own events.
fn deals_by_hand(events: &[Event]) -> Vec<Bits> {
    let mut out = Vec::new();
    for e in events {
        match *e {
            Event::HandStart { hand_index, .. } => {
                assert_eq!(hand_index as usize, out.len());
                out.push(0);
            }
            Event::Dealt { cards, .. } => *out.last_mut().unwrap() = cards,
            _ => {}
        }
    }
    out
}

#[test]
fn every_match_of_a_block_deals_each_seat_the_same_cards_in_hand_k() {
    let block = BlockSpec::new(2024);
    assert_eq!(block.first_dealer, block_first_dealer(2024));
    let cfg = TableConfig::in_block(9, Rules::HOUSE, block);
    // Different bots, so the matches differ in length and result.
    let (mut a_bots, a_seen) = watched_table(|_| Box::new(Lowest) as Box<dyn Bot>);
    let (mut b_bots, b_seen) = watched_table(|_| Box::new(RandomLegal::default()) as Box<dyn Bot>);
    let a = play_match(&cfg, &mut a_bots).unwrap();
    let b = play_match(&cfg, &mut b_bots).unwrap();
    assert_eq!(a.first_dealer, block.first_dealer);
    assert_eq!(b.first_dealer, block.first_dealer);
    assert_ne!(a.actions, b.actions);
    let mut compared = 0;
    for s in 0..8 {
        let da = deals_by_hand(&a_seen[s].borrow().last_events);
        let db = deals_by_hand(&b_seen[s].borrow().last_events);
        for (k, (x, y)) in da.iter().zip(&db).enumerate() {
            assert_eq!(x, y, "seat {s}, hand {k}");
            compared += 1;
        }
    }
    assert!(compared >= 16, "at least two hands compared for every seat");
}

#[test]
fn an_illegal_action_stops_the_match_and_is_never_applied() {
    let cfg = TableConfig::plain(3, Rules::HOUSE);
    let mut bots = mixed_bots();
    bots[5] = Box::new(Cheater::new(4));
    let err = play_match(&cfg, &mut bots).unwrap_err();
    match &err {
        TableError::IllegalAction {
            seat,
            bot,
            action,
            legal,
        } => {
            assert_eq!(*seat, 5);
            assert_eq!(bot, "test-cheater");
            assert!(!legal.contains(action));
            assert_eq!(*action, not_legal(legal));
        }
        other => panic!("expected IllegalAction, got {other:?}"),
    }
    assert_eq!(err.seat(), 5);
    let text = err.to_string();
    assert!(text.contains("seat 5 (test-cheater)") && text.contains("not legal"));
    assert!(!err.public_reason().contains("Play") && !err.public_reason().contains("Burn"));
}
