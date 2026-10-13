//! JSON round trips of the serialisable engine types (docs/M1-SPEC.md §2). Built only with
//! `--features serde`.
#![cfg(feature = "serde")]

use engine::cards::Suit;
use engine::game::{Action, Event, Match, Phase, Rules, Visibility};
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
        r#"{"bonus_first":6,"bonus_later":5,"stake":5}"#
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
