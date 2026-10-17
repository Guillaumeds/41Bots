//! An engine position written as the app's `MatchState` (fivefingers `src/engine/types.ts`), for
//! the tests, the fixture requests and the measurements: what the app would send the engine at
//! that moment (docs/WASM-API.md §3).
//!
//! It follows the app's own bookkeeping where the two differ: a rob the engine has made but whose
//! burn is still due has not happened yet in the app (the turned card is still on the table, the
//! robber holds five cards and must rob before it plays); a turned pair's cards leave its hands
//! for `turnedDown`; a renege is shown once it is caught, as the app marks it (`caughtAt`).
//! Fields the engine never reads (trick points, the best trump, …) are filled plainly.

use engine::cards::{Bits, Card, JOKER, cards, count};
use engine::game::{Event, Match, Phase, TurnCause, Visibility};
use engine::game::{RenegeKind, current_hand_start_of};
use engine::rules::ace_of_trumps;
use engine::team_of;
use serde_json::{Value, json};

use crate::cards::{app_card, app_suit};

/// What of the match the app shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mask {
    /// `maskMatch(m, seat)`: the seat's own cards and the public table; every discard face down.
    Seat,
    /// As [`Mask::Seat`], with the seat's own discards face up (what a bot knows).
    SeatOwnBurns,
    /// The whole match, every card face up (the engine must read only what the seat sees).
    Open,
}

fn cards_json(set: Bits) -> Vec<Value> {
    cards(set).map(|c| json!(app_card(c))).collect()
}

/// The app's `MatchState` of `m` as `seat` sees it (`mask`).
///
/// # Panics
/// If the match is over.
pub fn app_match(m: &Match, seat: u8, mask: Mask) -> Value {
    let p = m.public();
    let view = m.view(seat);
    let start = current_hand_start_of(&view);
    let hs = view
        .events
        .iter()
        .rposition(|e| matches!(e, Event::HandStart { .. }))
        .expect("a hand started");
    let hand_events = &view.events[hs..];
    // Every event of the hand, private ones included (for the open mask and the burns' order).
    let all = m.events();
    let hs_all = all
        .iter()
        .rposition(|(_, e)| matches!(e, Event::HandStart { .. }))
        .expect("a hand started");
    let all_hand = &all[hs_all..];
    let to_act = m.to_act().expect("the match is not over");

    // The rob whose burn is still due has not happened in the app yet.
    let pending: Option<(u8, Card)> = match p.phase {
        Phase::AwaitBurn(s) => p
            .turned
            .last()
            .filter(|&&(_, by)| by == Some(s))
            .map(|&(c, _)| (s, c)),
        _ => None,
    };
    let (last_card, last_by) = *p.turned.last().expect("a card is turned");
    let phase = match p.phase {
        Phase::AwaitBurn(_) if p.trick_no == 0 && p.trump.is_none() => "jokerSteal",
        Phase::AwaitBurn(_) if p.trick_no == 0 => "dealerRob",
        _ => "play",
    };
    let on_table = match (pending, last_by) {
        (Some((_, c)), _) => Some(c),
        (None, None) => Some(last_card),
        (None, Some(_)) => None,
    };
    let taken_by: i64 = match (pending, last_by) {
        (None, Some(s)) if last_card != JOKER => i64::from(s),
        _ => -1,
    };
    let first_joker = p.turned.first().filter(|&&(c, _)| c == JOKER);
    let joker_stolen =
        matches!(first_joker, Some(&(_, Some(_)))) && !(pending.is_some_and(|(_, c)| c == JOKER));
    let joker_left = matches!(first_joker, Some(&(_, None))) && p.turned.len() > 1;
    let played: Bits = p.played;

    // Hands.
    let turned_seat = |s: u8| p.turned_teams[usize::from(team_of(s))];
    let mut hands = Vec::new();
    let mut turned_down = Vec::new();
    for s in 0..8u8 {
        let truth = m.truth().hands[usize::from(s)];
        let mut held = truth;
        if let Some((ps, c)) = pending
            && ps == s
        {
            held &= !engine::cards::bit(c);
        }
        if turned_seat(s) {
            hands.push(json!([]));
            for c in cards(held) {
                turned_down.push(if mask == Mask::Open {
                    json!(app_card(c))
                } else {
                    json!(-1)
                });
            }
            continue;
        }
        if s == seat || mask == Mask::Open {
            hands.push(Value::Array(cards_json(held)));
            continue;
        }
        // Face down, but for the cards everyone saw it take: the robbed card, a stolen Joker.
        let mut public: Bits = 0;
        if taken_by == i64::from(s) && !engine::cards::has(played, last_card) {
            public |= engine::cards::bit(last_card);
        }
        if joker_stolen && s == p.dealer && !engine::cards::has(played, JOKER) {
            public |= engine::cards::bit(JOKER);
        }
        let mut h: Vec<Value> = (0..count(held & !public)).map(|_| json!(-1)).collect();
        h.extend(cards_json(held & public));
        hands.push(Value::Array(h));
    }

    // Burns in order, face up where the mask shows them.
    let mut discards = Vec::new();
    for (vis, e) in all_hand {
        if let (Visibility::Private(s), Event::Burned { card, .. }) = (vis, e) {
            let show = mask == Mask::Open || (*s == seat && mask == Mask::SeatOwnBurns);
            discards.push(if show {
                json!(app_card(*card))
            } else {
                json!(-1)
            });
        }
    }

    // Tricks from the plays, split where tricks are won.
    let mut tricks = Vec::new();
    let mut cur: Vec<Value> = Vec::new();
    let mut cur_leader: Option<u8> = None;
    for e in hand_events {
        match *e {
            Event::Played { seat: s, card } => {
                cur_leader.get_or_insert(s);
                cur.push(json!({"seat": s, "card": app_card(card)}));
            }
            Event::TrickWon {
                seat: w, points, ..
            } => {
                tricks.push(json!({
                    "leader": cur_leader.unwrap_or(w), "plays": std::mem::take(&mut cur),
                    "winner": w, "points": points, "bonus": false, "bonusTo": -1
                }));
                cur_leader = None;
            }
            _ => {}
        }
    }
    let current = if p.trick_no == 0 {
        Value::Null
    } else {
        json!({"leader": p.leader, "plays": cur, "winner": -1, "points": 0, "bonus": false, "bonusTo": -1})
    };

    // Caught reneges, as the app marks them.
    let ace = p.trump.map(ace_of_trumps);
    let mut proved_seats = [false; 8];
    let mut claim_or_end = [false; 4];
    let mut last_player: Option<u8> = None;
    // The ace of trumps played: by whom, and which seats had taken the turned card before it
    // showed (the app catches their false robs then; a false rob after it is caught only at a
    // claim or at the end of the hand).
    let mut ace_by: Option<u8> = None;
    let mut took_before_ace = [false; 8];
    let mut reneges = Vec::new();
    for e in hand_events {
        match *e {
            Event::Took { seat: s, card } if card != JOKER && ace_by.is_none() => {
                took_before_ace[usize::from(s)] = true;
            }
            Event::Played { seat: s, card } => {
                last_player = Some(s);
                if Some(card) == ace {
                    ace_by = Some(s);
                }
            }
            Event::PairTurned { team, cause, .. } => match cause {
                TurnCause::Proved => {
                    if let Some(s) = last_player {
                        proved_seats[usize::from(s)] = true;
                    }
                }
                TurnCause::Claim | TurnCause::HandEnd => claim_or_end[usize::from(team)] = true,
                TurnCause::LeftJoker => reneges.push(json!({
                    "seat": p.dealer, "trickIndex": 0, "card": 52, "kind": "joker",
                    "suit": -1, "ledRank": 0, "caughtAt": 0
                })),
                _ => {}
            },
            _ => {}
        }
    }
    for r in m.reneges() {
        let s = usize::from(r.seat);
        let (kind, caught) = match r.kind {
            RenegeKind::Follow(_) => ("follow", proved_seats[s]),
            RenegeKind::TrumpLead(_) => ("trumpLead", proved_seats[s]),
            RenegeKind::NoRob => ("rob", proved_seats[s]),
            RenegeKind::FalseRob => (
                "falseRob",
                proved_seats[s]
                    || claim_or_end[usize::from(team_of(r.seat))]
                    || (took_before_ace[s] && ace_by.is_some_and(|a| a != r.seat)),
            ),
            RenegeKind::BurnedAce => (
                "burnedAce",
                proved_seats[s] || claim_or_end[usize::from(team_of(r.seat))],
            ),
        };
        if !caught && mask != Mask::Open && r.seat != seat {
            continue;
        }
        reneges.push(json!({
            "seat": r.seat, "trickIndex": 0, "card": ace.map_or(-1, |a| i64::from(app_card(a))),
            "kind": kind, "suit": -1, "ledRank": 0, "caughtAt": if caught { 0 } else { -1 }
        }));
    }

    let penalised: Vec<bool> = (0..8u8).map(turned_seat).collect();
    let robbed: Vec<bool> = (0..8u8).map(|s| taken_by == i64::from(s)).collect();
    let hand_points: Vec<i64> = (0..4)
        .map(|t| {
            if p.turned_teams[t] {
                0
            } else {
                i64::from(p.scores[t]) - i64::from(start.scores[t])
            }
        })
        .collect();
    let undealt = p.hidden_count - p.burn_counts.iter().sum::<u8>();
    json!({
        "config": {
            "n": 8, "target": 41, "gamesToWin": 2, "trickPoints": 5, "bonusExtra": 6,
            "jinkWins": true, "allowDiscardTrumpAce": false, "partnerPlaysOn": false,
            "jokers": 1, "jokerMode": "firstWins", "mustTrumpWhenVoid": false
        },
        "seed": 0, "pairs": 4,
        "gameScores": start.scores, "gamesWon": start.games_won,
        "gameIndex": start.game_index, "handIndex": start.hand_index, "dealer": start.dealer,
        "hand": {
            "index": start.hand_index, "seed": 0, "n": 8, "dealer": start.dealer,
            "trump": p.trump.map_or(-1, |t| i64::from(app_suit(t))),
            "turned": on_table.map_or(-1, |c| i64::from(app_card(c))),
            "turnedCard": app_card(last_card),
            "jokerTurned": usize::from(first_joker.is_some()),
            "jokerStolen": if joker_stolen { json!([52]) } else { json!([]) },
            "jokerLeft": if joker_left { json!([52]) } else { json!([]) },
            "jokerMode": "firstWins",
            "turnedTakenBy": taken_by,
            "hands": hands,
            "pack": vec![-1; usize::from(undealt)],
            "discards": discards,
            "turnedDown": turned_down,
            "tricks": tricks,
            "current": current,
            "penalised": penalised,
            "robbed": robbed,
            "voidedPairs": p.turned_teams,
            "reneges": reneges,
            "handPoints": hand_points,
            "bestTrump": null, "bonusSettled": false, "jink": -1,
            "phase": phase, "toAct": to_act
        },
        "status": "playing", "lastGameWinner": -1, "matchWinner": -1
    })
}

/// A request for the seat to act in `m` (docs/WASM-API.md §3).
pub fn app_request(m: &Match, deals: u32, seed: u64, mask: Mask) -> Value {
    let seat = m.to_act().expect("the match is not over");
    json!({
        "v": 1, "seat": seat, "deals": deals, "seed": seed.to_string(),
        "match": app_match(m, seat, mask)
    })
}
