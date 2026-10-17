//! What a seat may see: the card-set checks behind the leakage tests (docs/M1-SPEC.md §3.3).
//!
//! A seat may see its own hand and burns, and the public cards of the hand in progress (played
//! cards and turned cards, on the table or taken). It must never see another seat's hand or burn,
//! or the undealt pack. [`hidden_from`] gives the forbidden set from the engine's hidden truth;
//! [`view_cards`] lists the cards a view mentions by walking the Rust value, and [`json_cards`]
//! does the same independently on its JSON, so a test can check both against the forbidden set.

use engine::SEATS;
use engine::cards::{ALL_CARDS, Bits, Card, DECK_SIZE, bit, parse};
use engine::game::{Event, Truth};
use engine::view::{PublicState, SeatView};
use serde_json::Value;

/// The cards everyone has seen face up in the hand in progress: the played cards and every
/// turned card (still on the table or taken).
pub fn public_cards(p: &PublicState) -> Bits {
    let mut b = p.played;
    if let Some(c) = p.table_card {
        b |= bit(c);
    }
    for &(c, _) in &p.turned {
        b |= bit(c);
    }
    b
}

/// The cards `seat` must not see now: every other seat's hand and burns, and the undealt pack,
/// less the public cards.
pub fn hidden_from(seat: u8, truth: &Truth, public: &PublicState) -> Bits {
    let mut h = truth.undealt;
    for t in 0..SEATS {
        if t != usize::from(seat) {
            h |= truth.hands[t] | truth.burns[t];
        }
    }
    h & !public_cards(public)
}

/// The index of the latest `HandStart` event (0 if there is none): where the hand in progress
/// starts in an event list.
pub fn current_hand_start(events: &[Event]) -> usize {
    events
        .iter()
        .rposition(|e| matches!(e, Event::HandStart { .. }))
        .unwrap_or(0)
}

/// The cards an event names.
pub fn event_cards(e: &Event) -> Bits {
    match *e {
        Event::Dealt { cards, .. } => cards,
        Event::Turned { card }
        | Event::Took { card, .. }
        | Event::Burned { card, .. }
        | Event::Played { card, .. }
        | Event::Bonus { card, .. } => bit(card),
        Event::MatchStart { .. }
        | Event::HandStart { .. }
        | Event::BurnDone { .. }
        | Event::ProvedNoAce { .. }
        | Event::AceUndealt
        | Event::TrickWon { .. }
        | Event::GameWon { .. }
        | Event::HandEnd { .. }
        | Event::MatchWon { .. }
        | Event::PairTurned { .. }
        | Event::TrickPassed { .. } => 0,
    }
}

/// Every card a view names: its hand and burns, the public state's cards, and the cards of its
/// events from index `from` on.
pub fn view_cards(view: &SeatView, from: usize) -> Bits {
    let p = &view.public;
    let mut b = view.hand | view.own_burns | p.played;
    if let Some(c) = p.table_card {
        b |= bit(c);
    }
    for &(c, _) in &p.turned {
        b |= bit(c);
    }
    for &(_, c) in &p.trick_cards {
        b |= bit(c);
    }
    if let Some((c, _)) = p.best_trump {
        b |= bit(c);
    }
    for e in view.events.iter().skip(from) {
        b |= event_cards(e);
    }
    b
}

/// The view with only the events of the hand in progress (from its `HandStart` on). Earlier
/// hands' events name cards that were dealt again since, so they are checked hand by hand with
/// [`check_event_windows`] instead.
pub fn current_hand_view(view: &SeatView) -> SeatView {
    let mut v = view.clone();
    let start = current_hand_start(&v.events);
    v.events.drain(..start);
    v
}

/// Check a seat's event list hand by hand: in each hand, the seat sees only its own deal and
/// burns, and every other card it is told of was turned or played in that hand.
pub fn check_event_windows(seat: u8, events: &[Event]) -> Result<(), String> {
    let mut starts: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| matches!(e, Event::HandStart { .. }))
        .map(|(i, _)| i)
        .collect();
    starts.push(events.len());
    for w in starts.windows(2) {
        let window = &events[w[0]..w[1]];
        let mut allowed: Bits = 0;
        for e in window {
            match *e {
                Event::Dealt { seat: s, .. } | Event::Burned { seat: s, .. } if s != seat => {
                    return Err(format!(
                        "seat {seat} was shown seat {s}'s private event {e:?}"
                    ));
                }
                Event::Dealt { cards, .. } => allowed |= cards,
                Event::Burned { card, .. }
                | Event::Turned { card }
                | Event::Took { card, .. }
                | Event::Played { card, .. } => allowed |= bit(card),
                _ => {}
            }
        }
        for e in window {
            let extra = event_cards(e) & !allowed;
            if extra != 0 {
                return Err(format!(
                    "seat {seat} was told of a card never shown to it in that hand: {e:?}"
                ));
            }
        }
    }
    Ok(())
}

/// Keys whose value is one card (a number, or null).
const CARD_KEYS: [&str; 4] = ["card", "table_card", "Play", "Burn"];
/// Keys whose value is a set of cards (a bitboard number).
const BITS_KEYS: [&str; 4] = ["hand", "own_burns", "played", "cards"];
/// Keys that hold no card directly (their value is searched further).
const PLAIN_KEYS: [&str; 63] = [
    // SeatView and PublicState
    "seat",
    "public",
    "events",
    "phase",
    "to_act",
    "dealer",
    "hand_index",
    "game_index",
    "scores",
    "games_won",
    "bonus_taken",
    "trump",
    "joker_turned",
    "hand_sizes",
    "burn_counts",
    "burn_order",
    "hidden_count",
    "trick_no",
    "leader",
    "trick_winners",
    "tricks_won",
    "five_played",
    "robber",
    "proved_no_ace",
    "ace_undealt",
    "turned_teams",
    // Phase
    "AwaitBurn",
    "AwaitPlay",
    // Events and their fields
    "MatchStart",
    "first_dealer",
    "HandStart",
    "Dealt",
    "Turned",
    "Took",
    "Burned",
    "BurnDone",
    "Played",
    "ProvedNoAce",
    "TrickWon",
    "trick",
    "team",
    "points",
    "Bonus",
    "kind",
    "GameWon",
    "reason",
    "HandEnd",
    "aborted",
    "MatchWon",
    "money",
    "PairTurned",
    "score",
    "cause",
    "TrickPassed",
    // Messages (hello, act, ready, action, abort) and the rules
    "v",
    "type",
    "view",
    "legal",
    "rules",
    "stake",
    "seed",
    "name",
    "action",
];

fn card_value(key: &str, v: &Value) -> Result<Bits, String> {
    match v {
        Value::Null => Ok(0),
        Value::Number(n) => match n.as_u64() {
            Some(c) if c < DECK_SIZE as u64 => Ok(bit(c as Card)),
            _ => Err(format!("\"{key}\" holds {n}, which is not a card")),
        },
        _ => Err(format!("\"{key}\" holds {v}, which is not a card")),
    }
}

fn bits_value(key: &str, v: &Value) -> Result<Bits, String> {
    match v.as_u64() {
        Some(b) if b & !ALL_CARDS == 0 => Ok(b),
        _ => Err(format!("\"{key}\" holds {v}, which is not a set of cards")),
    }
}

/// The `i`th element of each pair in `v` (an array of pairs, or for `single` one pair or null).
fn pair_cards(key: &str, v: &Value, i: usize, single: bool) -> Result<Bits, String> {
    let pair = |p: &Value| match p.as_array() {
        Some(xs) if xs.len() == 2 => card_value(key, &xs[i]),
        _ => Err(format!("\"{key}\" holds {p}, which is not a pair")),
    };
    if single {
        return if v.is_null() { Ok(0) } else { pair(v) };
    }
    match v.as_array() {
        Some(xs) => xs.iter().try_fold(0, |acc, p| Ok(acc | pair(p)?)),
        None => Err(format!("\"{key}\" holds {v}, which is not a list")),
    }
}

fn walk(v: &Value, out: &mut Bits) -> Result<(), String> {
    match v {
        Value::Object(map) => {
            for (k, x) in map {
                let k = k.as_str();
                if CARD_KEYS.contains(&k) {
                    *out |= card_value(k, x)?;
                } else if BITS_KEYS.contains(&k) {
                    *out |= bits_value(k, x)?;
                } else if k == "turned" {
                    *out |= pair_cards(k, x, 0, false)?;
                } else if k == "trick_cards" {
                    *out |= pair_cards(k, x, 1, false)?;
                } else if k == "best_trump" {
                    *out |= pair_cards(k, x, 0, true)?;
                } else if PLAIN_KEYS.contains(&k) {
                    walk(x, out)?;
                } else {
                    return Err(format!(
                        "unclassified key \"{k}\": add it to the card-bearing or plain keys"
                    ));
                }
            }
            Ok(())
        }
        Value::Array(xs) => xs.iter().try_for_each(|x| walk(x, out)),
        Value::String(s) => {
            // A card name inside any string would be a channel too.
            if let Some(c) = parse(s) {
                *out |= bit(c);
            }
            Ok(())
        }
        Value::Number(_) | Value::Bool(_) | Value::Null => Ok(()),
    }
}

/// Every card named in a JSON seat view or table-to-seat message (`hello`, `act`, `abort`), read
/// from the JSON alone: numbers under card keys, bitboards under card-set keys, and any string
/// that is a card name. Fails on a key it does not know, so a new field must be classified
/// before it can pass a leakage test.
pub fn json_cards(v: &Value) -> Result<Bits, String> {
    let mut out = 0;
    walk(v, &mut out)?;
    Ok(out)
}

/// In a JSON seat view, or a message holding one under `"view"`, keep only the events of the
/// hand in progress (from the latest `HandStart` on), as [`current_hand_view`] does.
pub fn truncate_json_events(v: &mut Value) {
    let view = match v.get_mut("view") {
        Some(inner) => inner,
        None => v,
    };
    if let Some(Value::Array(events)) = view.get_mut("events") {
        let start = events
            .iter()
            .rposition(|e| e.get("HandStart").is_some())
            .unwrap_or(0);
        events.drain(..start);
    }
}
