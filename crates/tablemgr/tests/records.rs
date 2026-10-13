//! Match records (docs/M1-SPEC.md §3.2): JSON Lines round trip, replay and verification.

mod common;

use common::{mixed_bots, random_bots};
use engine::game::{Action, Event, Phase, RULES_VERSION, Rules, Visibility};
use std::io::Cursor;
use tablemgr::{
    BlockSpec, FORMAT_VERSION, MatchRecord, RecordError, RecordReader, RecordWriter, TableConfig,
    play_match,
};

fn some_records() -> Vec<MatchRecord> {
    let mut out = Vec::new();
    for i in 0..6u64 {
        let cfg = if i.is_multiple_of(2) {
            TableConfig::plain(100 + i, Rules::HOUSE)
        } else {
            TableConfig::in_block(100 + i, Rules::FLAT, BlockSpec::new(i))
        };
        let mut bots = if i < 3 { mixed_bots() } else { random_bots() };
        out.push(play_match(&cfg, &mut bots).unwrap());
    }
    out
}

#[test]
fn records_round_trip_through_a_jsonl_file_and_replay_exactly() {
    let records = some_records();
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let path = dir.join(format!("records-round-trip-{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&path);
    // Two appends to the same file: the second adds to the first.
    for half in records.chunks(3) {
        let mut w = RecordWriter::append(&path).unwrap();
        for r in half {
            w.write(r).unwrap();
        }
        w.flush().unwrap();
    }
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(text.lines().count(), records.len());
    assert!(text.ends_with('\n') && !text.contains('\r'));
    let back: Vec<MatchRecord> = RecordReader::open(&path)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(back, records);
    for r in &back {
        r.verify().unwrap();
        let m = r.replay().unwrap();
        assert_eq!(m.phase(), Phase::MatchOver);
        assert_eq!(m.money(), Some(r.money));
        assert_eq!(m.actions(), &r.actions[..]);
    }
}

#[test]
fn writer_and_reader_work_on_any_stream_and_skip_blank_lines() {
    let records = some_records();
    let mut w = RecordWriter::new(Vec::new());
    for r in &records {
        w.write(r).unwrap();
    }
    let mut bytes = w.into_inner();
    bytes.extend_from_slice(b"\n\n");
    let back: Vec<MatchRecord> = RecordReader::new(Cursor::new(bytes))
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(back, records);
}

#[test]
fn a_bad_line_is_reported_with_its_number() {
    let r = &some_records()[0];
    let text = format!("{}\nnot json\n", r.to_json());
    let mut reader = RecordReader::new(Cursor::new(text.into_bytes()));
    assert!(reader.next().unwrap().is_ok());
    match reader.next().unwrap() {
        Err(RecordError::Json { line, .. }) => assert_eq!(line, 2),
        other => panic!("expected a JSON error, got {other:?}"),
    }
    assert!(reader.next().is_none());
}

/// The keys of the outermost JSON object, in order.
fn top_level_keys(json: &str) -> Vec<String> {
    let (mut depth, mut in_str, mut escaped) = (0, false, false);
    let mut keys = Vec::new();
    let mut current = String::new();
    let mut prev_significant = ' ';
    for ch in json.chars() {
        if in_str {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_str = false;
            } else {
                current.push(ch);
            }
            continue;
        }
        match ch {
            '"' => {
                in_str = true;
                current.clear();
            }
            '{' | '[' => depth += 1,
            '}' | ']' => depth -= 1,
            ':' if depth == 1 && prev_significant == '"' => keys.push(current.clone()),
            _ => {}
        }
        prev_significant = ch;
    }
    keys
}

#[test]
fn the_record_carries_versions_and_the_spec_fields_in_order() {
    let r = &some_records()[1];
    assert_eq!(r.format_version, FORMAT_VERSION);
    assert_eq!(FORMAT_VERSION, 1);
    assert_eq!(r.rules_version, RULES_VERSION);
    let json = r.to_json();
    assert!(json.starts_with(r#"{"format_version":1,"rules_version":1,"rules":{"#));
    assert_eq!(
        top_level_keys(&json),
        [
            "format_version",
            "rules_version",
            "rules",
            "seed",
            "block",
            "bots",
            "first_dealer",
            "actions",
            "games",
            "money",
            "winner",
            "public_timeline",
        ]
    );
    assert!(!json.contains('\n'));
}

#[test]
fn games_money_winner_and_timeline_agree_with_the_match() {
    for r in some_records() {
        let m = r.replay().unwrap();
        // Games: one per GameWon, the last one won by the match winner, two for the winner.
        assert!(r.games.len() >= 2 && r.games.len() <= 5);
        assert_eq!(r.games.last().unwrap().winner, r.winner);
        assert_eq!(r.games.iter().filter(|g| g.winner == r.winner).count(), 2);
        assert_eq!(m.games_won()[usize::from(r.winner)], 2);
        let hands: u32 = r.games.iter().map(|g| g.hands).sum();
        assert_eq!(hands, m.hand_index() + 1);
        for (i, g) in r.games.iter().enumerate() {
            assert_eq!(usize::from(g.game_index), i);
            assert!(
                g.scores[usize::from(g.winner)] >= 41
                    || g.reason == engine::game::GameEndReason::Jink
            );
        }
        // Money: the winner +6 stakes, the others -2 each.
        let stake = r.rules.stake;
        for t in 0..4 {
            let want = if t == usize::from(r.winner) {
                6 * stake
            } else {
                -2 * stake
            };
            assert_eq!(r.money[t], want);
        }
        // Timeline: one point per completed trick, in order.
        let tricks = m
            .events()
            .iter()
            .filter(|(v, e)| *v == Visibility::Public && matches!(e, Event::TrickWon { .. }))
            .count();
        assert_eq!(r.public_timeline.len(), tricks);
        let mut ended = 0;
        for (i, p) in r.public_timeline.iter().enumerate() {
            assert!((1..=5).contains(&p.trick));
            let next = r.public_timeline.get(i + 1);
            let game_ends = p.games_won.iter().sum::<u8>() as usize > ended;
            if game_ends {
                // The point at a game's end holds that game's final scores.
                assert_eq!(p.scores, r.games[ended].scores);
                ended += 1;
            } else if let Some(n) = next {
                // Scores never go down within a game.
                assert!((0..4).all(|t| n.scores[t] >= p.scores[t]));
            }
        }
        assert_eq!(ended, r.games.len());
        assert_eq!(r.public_timeline.last().unwrap().games_won, m.games_won());
    }
}

#[test]
fn verify_catches_a_tampered_record() {
    let good = &some_records()[0];
    good.verify().unwrap();

    let mut r = good.clone();
    r.money[0] += 1;
    assert!(matches!(
        r.verify(),
        Err(RecordError::Mismatch { field: "money" })
    ));

    let mut r = good.clone();
    r.winner = (r.winner + 1) % 4;
    assert!(matches!(
        r.verify(),
        Err(RecordError::Mismatch { field: "winner" })
    ));

    let mut r = good.clone();
    r.games[0].hands += 1;
    assert!(matches!(
        r.verify(),
        Err(RecordError::Mismatch { field: "games" })
    ));

    let mut r = good.clone();
    r.public_timeline[0].scores[0] += 1;
    assert!(matches!(
        r.verify(),
        Err(RecordError::Mismatch {
            field: "public_timeline"
        })
    ));

    let mut r = good.clone();
    r.actions.pop();
    assert!(matches!(r.verify(), Err(RecordError::Mismatch { .. })));

    let mut r = good.clone();
    r.actions[0] = Action::Burn(0); // the ace of clubs: never burnable
    assert!(matches!(
        r.verify(),
        Err(RecordError::Illegal { index: 0, .. })
    ));

    let mut r = good.clone();
    r.rules_version += 1;
    assert!(matches!(r.replay(), Err(RecordError::RulesVersion(_))));

    let mut r = good.clone();
    r.format_version = 2;
    assert!(matches!(r.replay(), Err(RecordError::FormatVersion(2))));

    let mut r = good.clone();
    r.block = Some(BlockSpec {
        block_seed: 1,
        first_dealer: 9,
    });
    assert!(matches!(r.replay(), Err(RecordError::BadFirstDealer(9))));

    // A record of one match does not verify as another's.
    let mut r = good.clone();
    r.seed += 1;
    assert!(r.verify().is_err());
}
