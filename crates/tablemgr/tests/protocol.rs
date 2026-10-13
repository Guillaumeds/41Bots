//! The line protocol and the networked table (docs/M1-SPEC.md §3.4).

mod common;

use bot::Bot;
use common::{Cheater, bot_for_seat, mixed_bots};
use engine::game::{Action, Rules};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::Duration;
use tablemgr::protocol::{decode_seat, decode_table, encode_seat, encode_table};
use tablemgr::{
    BlockSpec, MatchRecord, SeatError, SeatLink, SeatMsg, TableConfig, TableError, TableMsg,
    TcpLink, mem_link_pair, play_match, run_seat, seat_seed, serve_match,
};

#[test]
fn messages_have_the_spec_shapes() {
    let hello = TableMsg::Hello {
        seat: 3,
        rules: Rules::HOUSE,
        seed: 123,
    };
    assert_eq!(
        encode_table(&hello),
        r#"{"v":1,"type":"hello","seat":3,"rules":{"bonus_first":6,"bonus_later":5,"stake":5},"seed":123}"#
    );
    let action = SeatMsg::Action {
        action: Action::Play(17),
    };
    assert_eq!(
        encode_seat(&action),
        r#"{"v":1,"type":"action","action":{"Play":17}}"#
    );
    assert_eq!(
        encode_seat(&SeatMsg::Ready {
            name: "rules-v1".to_string()
        }),
        r#"{"v":1,"type":"ready","name":"rules-v1"}"#
    );
    let abort = TableMsg::Abort {
        reason: "stop".to_string(),
    };
    assert_eq!(
        encode_table(&abort),
        r#"{"v":1,"type":"abort","reason":"stop"}"#
    );
    let m = TableConfig::plain(5, Rules::HOUSE).new_match();
    let seat = m.to_act().unwrap();
    let act = TableMsg::Act {
        view: Box::new(m.view(seat)),
        legal: m.legal_actions(),
    };
    let line = encode_table(&act);
    assert!(line.starts_with(r#"{"v":1,"type":"act","view":{"seat":"#));
    assert!(line.contains(r#"]},"legal":[{""#) && line.ends_with("}]}"));
    assert!(!line.contains('\n'));
    for msg in [hello, abort, act] {
        assert_eq!(decode_table(&encode_table(&msg)).unwrap(), msg);
    }
    assert_eq!(decode_seat(&encode_seat(&action)).unwrap(), action);
}

#[test]
fn other_versions_and_garbage_are_refused() {
    let err = decode_seat(r#"{"v":2,"type":"action","action":{"Play":17}}"#).unwrap_err();
    assert!(err.contains("version 2"), "{err}");
    let err = decode_seat(r#"{"v":2,"type":"teleport"}"#).unwrap_err();
    assert!(err.contains("version 2"), "{err}");
    assert!(decode_seat(r#"{"type":"action","action":{"Play":17}}"#).is_err());
    assert!(decode_seat(r#"{"v":1,"type":"action","action":{"Play":"17"}}"#).is_err());
    assert!(decode_seat("hello").is_err());
    assert!(decode_table(r#"{"v":1,"type":"act"}"#).is_err());
}

fn configs() -> Vec<TableConfig> {
    vec![
        TableConfig::plain(1, Rules::HOUSE),
        TableConfig::plain(2, Rules::FLAT),
        TableConfig::in_block(3, Rules::HOUSE, BlockSpec::new(30)),
        TableConfig::in_block(4, Rules::HOUSE, BlockSpec::new(31)),
    ]
}

/// Play `cfg` over in-memory links, each seat's client on its own thread with its test bot.
fn play_in_memory(
    cfg: &TableConfig,
) -> (
    Result<MatchRecord, TableError>,
    Vec<Result<MatchRecord, SeatError>>,
) {
    let mut table_ends: Vec<Box<dyn SeatLink>> = Vec::new();
    let mut clients = Vec::new();
    for s in 0..8 {
        let (table_end, mut seat_end) = mem_link_pair();
        table_ends.push(Box::new(table_end));
        clients.push(thread::spawn(move || {
            let mut bot = bot_for_seat(s);
            run_seat(&mut seat_end, bot.as_mut())
        }));
    }
    let links: [Box<dyn SeatLink>; 8] = table_ends.try_into().ok().unwrap();
    let result = serve_match(cfg, links);
    let seats = clients.into_iter().map(|c| c.join().unwrap()).collect();
    (result, seats)
}

#[test]
fn in_process_and_in_memory_transport_give_identical_records() {
    for cfg in configs() {
        let local = play_match(&cfg, &mut mixed_bots()).unwrap();
        let (remote, seats) = play_in_memory(&cfg);
        let remote = remote.unwrap();
        assert_eq!(remote, local);
        assert_eq!(remote.to_json(), local.to_json());
        remote.verify().unwrap();
        for s in seats {
            assert_eq!(
                s.unwrap(),
                local,
                "every seat receives the record at the end"
            );
        }
    }
}

#[test]
fn tcp_on_localhost_with_eight_client_threads_gives_the_in_process_record() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    for cfg in configs() {
        let local = play_match(&cfg, &mut mixed_bots()).unwrap();
        // Connect one client at a time and accept it, so client s is seat s.
        let mut table_ends: Vec<Box<dyn SeatLink>> = Vec::new();
        let mut clients = Vec::new();
        for s in 0..8 {
            let stream = TcpStream::connect(addr).unwrap();
            let (accepted, _) = listener.accept().unwrap();
            accepted
                .set_read_timeout(Some(Duration::from_secs(60)))
                .unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(60)))
                .unwrap();
            table_ends.push(Box::new(TcpLink::new(accepted).unwrap()));
            clients.push(thread::spawn(move || {
                let mut link = TcpLink::new(stream).unwrap();
                let mut bot = bot_for_seat(s);
                run_seat(&mut link, bot.as_mut())
            }));
        }
        let links: [Box<dyn SeatLink>; 8] = table_ends.try_into().ok().unwrap();
        let remote = serve_match(&cfg, links).unwrap();
        for c in clients {
            assert_eq!(c.join().unwrap().unwrap(), local);
        }
        assert_eq!(remote, local);
        assert_eq!(remote.to_json(), local.to_json());
    }
}

#[test]
fn an_illegal_action_over_the_wire_aborts_every_seat_without_naming_cards() {
    let cfg = TableConfig::plain(8, Rules::HOUSE);
    let mut table_ends: Vec<Box<dyn SeatLink>> = Vec::new();
    let mut clients = Vec::new();
    for s in 0..8 {
        let (table_end, mut seat_end) = mem_link_pair();
        table_ends.push(Box::new(table_end));
        clients.push(thread::spawn(move || {
            let mut bot: Box<dyn Bot> = if s == 2 {
                Box::new(Cheater::new(3))
            } else {
                bot_for_seat(s)
            };
            run_seat(&mut seat_end, bot.as_mut())
        }));
    }
    let links: [Box<dyn SeatLink>; 8] = table_ends.try_into().ok().unwrap();
    let err = serve_match(&cfg, links).unwrap_err();
    assert!(
        matches!(err, TableError::IllegalAction { seat: 2, ref bot, .. } if bot == "test-cheater")
    );
    for c in clients {
        match c.join().unwrap() {
            Err(SeatError::Aborted(reason)) => {
                assert_eq!(reason, err.public_reason());
                assert!(!reason.contains("Play") && !reason.contains("Burn"));
            }
            other => panic!("expected an abort, got {other:?}"),
        }
    }
}

#[test]
fn a_seat_that_disconnects_stops_the_match() {
    let cfg = TableConfig::plain(9, Rules::HOUSE);
    let mut table_ends: Vec<Box<dyn SeatLink>> = Vec::new();
    let mut clients = Vec::new();
    let mut quitter = None;
    for s in 0..8 {
        let (table_end, mut seat_end) = mem_link_pair();
        table_ends.push(Box::new(table_end));
        if s == 6 {
            // Seat 6 answers hello with ready, then hangs up.
            quitter = Some(thread::spawn(move || {
                let hello = decode_table(&seat_end.recv().unwrap()).unwrap();
                seat_end.send(&encode_seat(&SeatMsg::Ready {
                    name: "quitter".to_string(),
                }));
                hello
            }));
            continue;
        }
        clients.push(thread::spawn(move || {
            let mut bot = bot_for_seat(s);
            run_seat(&mut seat_end, bot.as_mut())
        }));
    }
    let links: [Box<dyn SeatLink>; 8] = table_ends.try_into().ok().unwrap();
    let err = serve_match(&cfg, links).unwrap_err();
    assert_eq!(err, TableError::Disconnected { seat: 6 });
    assert_eq!(
        quitter.unwrap().join().unwrap(),
        TableMsg::Hello {
            seat: 6,
            rules: cfg.rules,
            seed: seat_seed(cfg.seed, 6)
        }
    );
    for c in clients {
        assert!(matches!(c.join().unwrap(), Err(SeatError::Aborted(_))));
    }
}

#[test]
fn a_seat_client_refuses_an_act_for_another_seat() {
    let (mut table_end, mut seat_end) = mem_link_pair();
    let client = thread::spawn(move || run_seat(&mut seat_end, bot_for_seat(1).as_mut()));
    table_end.send(&encode_table(&TableMsg::Hello {
        seat: 1,
        rules: Rules::HOUSE,
        seed: 0,
    }));
    assert!(matches!(
        decode_seat(&table_end.recv().unwrap()).unwrap(),
        SeatMsg::Ready { .. }
    ));
    let m = TableConfig::plain(5, Rules::HOUSE).new_match();
    let other = (m.to_act().unwrap() + 1) % 8;
    let other = if other == 1 { 2 } else { other };
    table_end.send(&encode_table(&TableMsg::Act {
        view: Box::new(m.view(other)),
        legal: m.legal_actions(),
    }));
    assert!(matches!(
        client.join().unwrap(),
        Err(SeatError::Protocol(_))
    ));
}
