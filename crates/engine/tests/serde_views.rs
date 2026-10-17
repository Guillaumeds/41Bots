//! JSON round trips of the serialisable engine types (docs/M1-SPEC.md §2). Built only with
//! `--features serde`.
#![cfg(feature = "serde")]

use engine::cards::Suit;
use engine::game::{Action, Event, Extra, Lapse, Match, Phase, Rules, TurnCause, Visibility};
use engine::rng::Rng;
use engine::view::{PublicState, SeatView};

fn round_trip<T>(x: &T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let json = serde_json::to_string(x).unwrap();
    serde_json::from_str(&json).unwrap()
}

#[test]
fn seat_views_round_trip_through_json_at_every_decision() {
    for seed in 0..3u64 {
        let mut m = Match::new(seed, Rules::HOUSE);
        let mut n = 0u64;
        loop {
            for seat in 0..8u8 {
                let v = m.view(seat);
                let back: SeatView = round_trip(&v);
                assert_eq!(back, v, "seed {seed}, decision {n}, seat {seat}");
            }
            let p = m.public();
            assert_eq!(round_trip::<PublicState>(&p), p);
            if m.phase() == Phase::MatchOver {
                break;
            }
            let legal = m.legal_actions();
            assert_eq!(round_trip::<Vec<Action>>(&legal), legal);
            let i = Rng::derive(seed, &[n]).bounded(legal.len() as u64) as usize;
            m.apply(legal[i]).unwrap();
            n += 1;
        }
        let log: Vec<(Visibility, Event)> = m.events().to_vec();
        assert_eq!(round_trip(&log), log);
    }
}

#[test]
fn the_json_shapes_are_the_protocol_shapes() {
    // docs/M1-SPEC.md §3.4: actions look like {"Play":17}.
    assert_eq!(
        serde_json::to_string(&Action::Play(17)).unwrap(),
        r#"{"Play":17}"#
    );
    assert_eq!(
        serde_json::to_string(&Action::Burn(3)).unwrap(),
        r#"{"Burn":3}"#
    );
    assert_eq!(
        serde_json::to_string(&Rules::HOUSE).unwrap(),
        r#"{"stake":5}"#
    );
    assert_eq!(
        serde_json::to_string(&Phase::AwaitPlay(4)).unwrap(),
        r#"{"AwaitPlay":4}"#
    );
    assert_eq!(serde_json::to_string(&Suit::Hearts).unwrap(), r#""Hearts""#);
    assert_eq!(
        serde_json::to_string(&Visibility::Private(2)).unwrap(),
        r#"{"Private":2}"#
    );
    let e = Event::Played { seat: 1, card: 52 };
    assert_eq!(
        serde_json::to_string(&e).unwrap(),
        r#"{"Played":{"seat":1,"card":52}}"#
    );
}

/// Turned partnerships (docs/APP-ENGINE-CONTRACT.md §7): views with turn events round trip, and
/// `turned_teams` appears in the JSON only in a hand where a partnership is turned, so the JSON
/// of every other view is unchanged.
#[test]
fn turned_views_round_trip_and_normal_json_is_unchanged() {
    let mut turned_json = 0;
    for seed in 0..4u64 {
        let mut m = Match::new(seed, Rules::HOUSE);
        let mut rng = Rng::new(seed ^ 0xBEEF);
        let mut n = 0u64;
        while m.phase() != Phase::MatchOver && n < 2000 {
            let p = m.public();
            let json = serde_json::to_string(&p).unwrap();
            assert_eq!(
                json.contains("turned_teams"),
                p.turned_teams.iter().any(|&t| t),
                "{json}"
            );
            turned_json += usize::from(json.contains("turned_teams"));
            for seat in 0..8u8 {
                let v = m.view(seat);
                assert_eq!(round_trip::<SeatView>(&v), v);
            }
            if matches!(m.phase(), Phase::AwaitPlay(_)) && rng.bounded(12) == 0 {
                let _ = m.turn_team(rng.bounded(4) as u8);
            } else {
                let legal = m.legal_actions();
                m.apply(legal[rng.bounded(legal.len() as u64) as usize])
                    .unwrap();
            }
            n += 1;
        }
        let log: Vec<(Visibility, Event)> = m.events().to_vec();
        assert_eq!(round_trip(&log), log);
    }
    assert!(turned_json > 0);
    assert_eq!(
        serde_json::to_string(&Event::PairTurned {
            team: 1,
            score: 10,
            cause: TurnCause::Claim
        })
        .unwrap(),
        r#"{"PairTurned":{"team":1,"score":10,"cause":"Claim"}}"#
    );
    // The match record's extras (turns, plays with a turn, lapses) round trip too.
    let extras = [
        (3u32, Extra::Turn(2)),
        (5, Extra::PlayThenTurn(7, 1)),
        (9, Extra::Lapse(Lapse::FalseRob(12))),
        (11, Extra::Lapse(Lapse::BurnAce)),
    ];
    assert_eq!(round_trip(&extras.to_vec()), extras.to_vec());
    assert_eq!(
        serde_json::to_string(&extras[2].1).unwrap(),
        r#"{"Lapse":{"FalseRob":12}}"#
    );
}
