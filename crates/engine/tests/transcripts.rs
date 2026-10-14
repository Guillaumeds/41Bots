//! Replays the hand-made transcripts in `tests/transcripts/` against the engine
//! (docs/M0-SPEC.md §8) and checks every expectation they state.
//!
//! Format, one directive per line; `#` starts a comment anywhere on a line, and blank lines and
//! trailing whitespace are ignored:
//! - header: `title:`, `story:`, `rules:`, `dealer:`, `hand <seat>:`, `turned:`, `second:` and
//!   `undealt:` describe the first deal. The same deal directives prefixed by `then ` describe
//!   the deal of each later hand, one contiguous block per hand; hand k is dealt by seat
//!   `(dealer + k) % 8`, because the deal moves one seat left every hand (H5);
//! - `> Burn <card>` or `> Play <card>`: an action by the seat to act;
//! - expectations about the state right after the latest action (or right after the deal, before
//!   the first action): `= <canonical event>` lines form an in-order subsequence of the public
//!   events produced by that action; `scores:`, `phase:`, `legal:` (the exact legal cards, in
//!   card-index order) and `result:` (`game team=<n> reason=<r>` or
//!   `match team=<n> money=<a>,<b>,<c>,<d>`), and `burns:` (the public burn order,
//!   `none` or `<seat> <kind>, …`).
//!
//! A disagreement is a finding for the reviewer: the engine or the transcript may be the one
//! that is wrong, and RULES.md decides.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use engine::cards::{Card, DECK_SIZE, JOKER, name, parse};
use engine::game::{Action, Event, Match, Phase, Rules, Visibility, transcript, transcript_full};
use engine::view::BurnKind;

const SEATS: usize = 8;
const HAND_SIZE: usize = 5;
/// Deck position of the turned card (docs/M0-SPEC.md §1).
const TURNED_POS: usize = 40;
/// How many problems per failing file the final panic message shows.
const SHOWN_PER_FILE: usize = 3;

// ----- files -----

fn transcript_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("transcripts")
}

/// Every `*.txt` transcript, sorted by file name.
fn transcript_files() -> Vec<PathBuf> {
    let dir = transcript_dir();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|entry| entry.expect("readable directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    files.sort();
    files
}

fn file_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

// ----- parsed form -----

/// A problem found in a transcript; `line` 0 means the file as a whole.
#[derive(Debug)]
struct Failure {
    line: usize,
    message: String,
}

impl Failure {
    fn new(line: usize, message: impl Into<String>) -> Failure {
        Failure {
            line,
            message: message.into(),
        }
    }

    fn render(&self) -> String {
        if self.line == 0 {
            format!("file: {}", self.message)
        } else {
            format!("line {}: {}", self.line, self.message)
        }
    }
}

/// A value and the line it was written on.
type Placed<T> = Option<(usize, T)>;

/// One deal as written in the header: the first deal, or one `then` block.
#[derive(Default)]
struct DealSpec {
    /// Line of the block's first directive.
    line: usize,
    hands: [Placed<Vec<Card>>; SEATS],
    turned: Placed<Card>,
    second: Placed<Card>,
    undealt: Placed<Vec<Card>>,
}

enum Expect {
    Event(String),
    Scores([u8; 4]),
    Phase(Phase),
    Legal(Vec<Card>),
    Burns(Vec<(u8, BurnKind)>),
    Game { team: u8, reason: String },
    MatchWon { team: u8, money: [i32; 4] },
}

/// An action and the expectations that follow it (the first step has no action: the deal).
struct Step {
    action: Option<(usize, Action)>,
    expects: Vec<(usize, Expect)>,
}

struct Transcript {
    title: String,
    rules: Rules,
    dealer: Option<u8>,
    deals: Vec<DealSpec>,
    steps: Vec<Step>,
}

// ----- parser -----

fn parse_transcript(text: &str) -> Result<Transcript, Vec<Failure>> {
    let mut t = Transcript {
        title: String::new(),
        rules: Rules::default(),
        dealer: None,
        deals: vec![DealSpec::default()],
        steps: vec![Step {
            action: None,
            expects: Vec::new(),
        }],
    };
    let mut errors = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line_no = i + 1;
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if let Err(message) = parse_line(&mut t, line_no, line) {
            errors.push(Failure::new(
                line_no,
                format!("{message}: `{}`", raw.trim()),
            ));
        }
    }
    if t.dealer.is_none() {
        errors.push(Failure::new(0, "no `dealer:` line"));
    }
    if errors.is_empty() {
        Ok(t)
    } else {
        Err(errors)
    }
}

fn parse_line(t: &mut Transcript, line_no: usize, line: &str) -> Result<(), String> {
    if let Some(rest) = line.strip_prefix('>') {
        let action = parse_action(rest)?;
        t.steps.push(Step {
            action: Some((line_no, action)),
            expects: Vec::new(),
        });
        return Ok(());
    }
    if let Some(rest) = line.strip_prefix('=') {
        let event = rest.split_whitespace().collect::<Vec<_>>().join(" ");
        if event.is_empty() {
            return Err("an `=` line names no event".into());
        }
        return expect(t, line_no, Expect::Event(event));
    }
    let (key, value) = line
        .split_once(':')
        .ok_or("not a directive (expected `key: value`, `> action` or `= event`)")?;
    let value = value.trim();
    let words: Vec<&str> = key.split_whitespace().collect();
    let (then, key) = match words.split_first() {
        Some((&"then", rest)) => (true, rest),
        _ => (false, &words[..]),
    };
    match key {
        ["title"] if !then => t.title = value.to_string(),
        ["story"] if !then => {}
        ["rules"] if !then => t.rules = parse_rules(value)?,
        ["dealer"] if !then => {
            if t.dealer.is_some() {
                return Err("a second `dealer:` line".into());
            }
            t.dealer = Some(parse_seat(value)?);
        }
        ["hand", seat] => {
            let seat = usize::from(parse_seat(seat)?);
            let cards = parse_cards(value)?;
            deal_for(t, then, line_no, |d| d.hands[seat].is_some())?.hands[seat] =
                Some((line_no, cards));
        }
        ["turned"] => {
            let card = parse_one_card(value)?;
            deal_for(t, then, line_no, |d| d.turned.is_some())?.turned = Some((line_no, card));
        }
        ["second"] => {
            let card = parse_one_card(value)?;
            deal_for(t, then, line_no, |d| d.second.is_some())?.second = Some((line_no, card));
        }
        ["undealt"] => {
            let cards = parse_cards(value)?;
            deal_for(t, then, line_no, |d| d.undealt.is_some())?.undealt = Some((line_no, cards));
        }
        ["scores"] if !then => expect(t, line_no, Expect::Scores(parse_scores(value)?))?,
        ["phase"] if !then => expect(t, line_no, Expect::Phase(parse_phase(value)?))?,
        ["legal"] if !then => expect(t, line_no, Expect::Legal(parse_cards(value)?))?,
        ["burns"] if !then => expect(t, line_no, Expect::Burns(parse_burns(value)?))?,
        ["result"] if !then => expect(t, line_no, parse_result(value)?)?,
        _ => return Err("unknown directive".into()),
    }
    Ok(())
}

fn expect(t: &mut Transcript, line_no: usize, e: Expect) -> Result<(), String> {
    t.steps
        .last_mut()
        .ok_or("no step to attach the expectation to")?
        .expects
        .push((line_no, e));
    Ok(())
}

/// The deal a header directive belongs to. Plain directives go to the first deal. A `then`
/// directive goes to the latest `then` block unless that block already has this field or is
/// closed by its `undealt:` line, in which case it opens the next block.
fn deal_for(
    t: &mut Transcript,
    then: bool,
    line_no: usize,
    already_set: impl Fn(&DealSpec) -> bool,
) -> Result<&mut DealSpec, String> {
    if !then {
        let first = &mut t.deals[0];
        if already_set(first) {
            return Err("given twice for the first deal".into());
        }
        if first.line == 0 {
            first.line = line_no;
        }
        return Ok(first);
    }
    let open_new = match t.deals.last() {
        Some(last) => t.deals.len() == 1 || already_set(last) || last.undealt.is_some(),
        None => true,
    };
    if open_new {
        t.deals.push(DealSpec {
            line: line_no,
            ..DealSpec::default()
        });
    }
    t.deals.last_mut().ok_or_else(|| "no deal".to_string())
}

fn parse_action(rest: &str) -> Result<Action, String> {
    let words: Vec<&str> = rest.split_whitespace().collect();
    let [verb, card] = words[..] else {
        return Err("an action is `> Burn <card>` or `> Play <card>`".into());
    };
    let card = parse_one_card(card)?;
    if verb.eq_ignore_ascii_case("burn") {
        Ok(Action::Burn(card))
    } else if verb.eq_ignore_ascii_case("play") {
        Ok(Action::Play(card))
    } else {
        Err(format!("unknown action `{verb}` (Burn or Play)"))
    }
}

fn parse_seat(s: &str) -> Result<u8, String> {
    match s.trim().parse::<u8>() {
        Ok(seat) if usize::from(seat) < SEATS => Ok(seat),
        _ => Err(format!("`{s}` is not a seat 0-7")),
    }
}

fn parse_one_card(s: &str) -> Result<Card, String> {
    let cards = parse_cards(s)?;
    match cards[..] {
        [c] => Ok(c),
        _ => Err(format!("expected exactly one card, found {}", cards.len())),
    }
}

/// Card names separated by spaces (commas are tolerated too).
fn parse_cards(s: &str) -> Result<Vec<Card>, String> {
    s.split(|c: char| c.is_whitespace() || c == ',')
        .filter(|w| !w.is_empty())
        .map(|w| parse(w).ok_or_else(|| format!("`{w}` is not a card name")))
        .collect()
}

fn parse_rules(s: &str) -> Result<Rules, String> {
    let mut rules = Rules::default();
    for word in s.split_whitespace() {
        let (key, value) = word
            .split_once('=')
            .ok_or_else(|| format!("`{word}` is not key=value"))?;
        let bad = || format!("bad value in `{word}`");
        match key {
            "stake" => rules.stake = value.parse().map_err(|_| bad())?,
            _ => return Err(format!("unknown rule `{key}`")),
        }
    }
    Ok(rules)
}

fn parse_scores(s: &str) -> Result<[u8; 4], String> {
    let parts: Vec<u8> = s
        .split(',')
        .map(|p| p.trim().parse::<u8>())
        .collect::<Result<_, _>>()
        .map_err(|_| format!("`{s}` is not four scores a,b,c,d"))?;
    parts
        .try_into()
        .map_err(|_| format!("`{s}` is not four scores a,b,c,d"))
}

fn parse_money(s: &str) -> Result<[i32; 4], String> {
    let parts: Vec<i32> = s
        .split(',')
        .map(|p| p.trim().parse::<i32>())
        .collect::<Result<_, _>>()
        .map_err(|_| format!("`{s}` is not four amounts a,b,c,d"))?;
    parts
        .try_into()
        .map_err(|_| format!("`{s}` is not four amounts a,b,c,d"))
}

/// `burns: none` or `burns: <seat> <kind>, <seat> <kind>` with kinds `JokerTake`,
/// `AceRobAtOnce` and `RobAtTurn`: the public `burn_order` (docs/M3-SPEC.md §2.2).
fn parse_burns(s: &str) -> Result<Vec<(u8, BurnKind)>, String> {
    if s.trim() == "none" {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|item| {
            let words: Vec<&str> = item.split_whitespace().collect();
            match words[..] {
                [seat, kind] => {
                    let kind = match kind {
                        "JokerTake" => BurnKind::JokerTake,
                        "AceRobAtOnce" => BurnKind::AceRobAtOnce,
                        "RobAtTurn" => BurnKind::RobAtTurn,
                        _ => return Err(format!("unknown burn kind `{kind}`")),
                    };
                    Ok((parse_seat(seat)?, kind))
                }
                _ => Err("a burn is `<seat> <kind>`".into()),
            }
        })
        .collect()
}

fn parse_phase(s: &str) -> Result<Phase, String> {
    let words: Vec<&str> = s
        .split(|c: char| c.is_whitespace() || c == '(' || c == ')')
        .filter(|w| !w.is_empty())
        .collect();
    match words[..] {
        ["MatchOver"] => Ok(Phase::MatchOver),
        ["AwaitBurn", seat] => Ok(Phase::AwaitBurn(parse_seat(seat)?)),
        ["AwaitPlay", seat] => Ok(Phase::AwaitPlay(parse_seat(seat)?)),
        _ => Err("a phase is `AwaitBurn <seat>`, `AwaitPlay <seat>` or `MatchOver`".into()),
    }
}

fn parse_result(s: &str) -> Result<Expect, String> {
    let mut words = s.split_whitespace();
    let kind = words.next().ok_or("empty result")?;
    let mut team = None;
    let mut reason = None;
    let mut money = None;
    for word in words {
        let (key, value) = word
            .split_once('=')
            .ok_or_else(|| format!("`{word}` is not key=value"))?;
        match key {
            "team" => {
                team = Some(
                    value
                        .parse::<u8>()
                        .ok()
                        .filter(|&t| t < 4)
                        .ok_or_else(|| format!("`{value}` is not a team 0-3"))?,
                );
            }
            "reason" => {
                if !["trick", "immediate_bonus", "end_bonus", "jink"].contains(&value) {
                    return Err(format!("unknown game-end reason `{value}`"));
                }
                reason = Some(value.to_string());
            }
            "money" => money = Some(parse_money(value)?),
            _ => return Err(format!("unknown result field `{key}`")),
        }
    }
    let team = team.ok_or("the result names no team")?;
    match kind {
        "game" => Ok(Expect::Game {
            team,
            reason: reason.ok_or("a game result needs reason=")?,
        }),
        "match" => Ok(Expect::MatchWon {
            team,
            money: money.ok_or("a match result needs money=")?,
        }),
        _ => Err(format!("unknown result kind `{kind}` (game or match)")),
    }
}

// ----- deck builder -----

/// The deck, in deal order (docs/M0-SPEC.md §1), that deals `deal` with this dealer: seat
/// `(dealer + 1 + i) % 8` gets positions `2i, 2i + 1` (its first two listed cards) and
/// `16 + 3i ..= 18 + 3i` (its other three); position 40 is the turned card; position 41 is the
/// second turned card if given; the undealt cards follow in order. Checks that the 53 cards are a
/// permutation of the pack, naming every duplicate and missing card.
fn build_deck(dealer: u8, deal: &DealSpec) -> Result<[Card; DECK_SIZE], Vec<String>> {
    let mut errors = Vec::new();
    // Every placed card with where the file wrote it, by deck position (None = not given).
    let mut placed: Vec<Option<(Card, String)>> = vec![None; TURNED_POS];
    for i in 0..SEATS {
        let seat = (usize::from(dealer) + 1 + i) % SEATS;
        let Some((line, cards)) = &deal.hands[seat] else {
            errors.push(format!("no `hand {seat}:` line"));
            continue;
        };
        if cards.len() != HAND_SIZE {
            errors.push(format!(
                "hand {seat} (line {line}) lists {} cards, not {HAND_SIZE}",
                cards.len()
            ));
            continue;
        }
        let source = format!("hand {seat} (line {line})");
        let positions = [2 * i, 2 * i + 1, 16 + 3 * i, 17 + 3 * i, 18 + 3 * i];
        for (pos, &card) in positions.into_iter().zip(cards) {
            placed[pos] = Some((card, source.clone()));
        }
    }
    match deal.turned {
        Some((line, card)) => placed.push(Some((card, format!("turned (line {line})")))),
        None => errors.push("no `turned:` line".to_string()),
    }
    if let Some((line, card)) = deal.second {
        placed.push(Some((card, format!("second (line {line})"))));
    } else if matches!(deal.turned, Some((_, JOKER))) {
        errors.push("the Joker is turned, so a `second:` line is required".to_string());
    }
    match &deal.undealt {
        Some((line, cards)) => {
            for &card in cards {
                placed.push(Some((card, format!("undealt (line {line})"))));
            }
        }
        None => errors.push("no `undealt:` line".to_string()),
    }

    let mut first_seen: Vec<Option<&str>> = vec![None; DECK_SIZE];
    for (card, source) in placed.iter().flatten() {
        match first_seen[usize::from(*card)] {
            Some(earlier) => errors.push(format!(
                "card {} is given twice: in {earlier} and in {source}",
                name(*card)
            )),
            None => first_seen[usize::from(*card)] = Some(source.as_str()),
        }
    }
    for (card, seen) in (0..).zip(&first_seen) {
        if seen.is_none() {
            errors.push(format!(
                "card {} is missing (it is in no hand, not turned and not undealt)",
                name(card)
            ));
        }
    }
    if placed.len() != DECK_SIZE && errors.is_empty() {
        errors.push(format!("the deal places {} cards, not 53", placed.len()));
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    let mut deck = [0; DECK_SIZE];
    for (slot, entry) in deck.iter_mut().zip(&placed) {
        if let Some((card, _)) = entry {
            *slot = *card;
        }
    }
    Ok(deck)
}

/// Builds every deal of the file and starts the match. Returns the match and the number of
/// decks given.
fn start_match(t: &Transcript) -> Result<(Match, usize), Vec<Failure>> {
    let first_dealer = t.dealer.ok_or_else(|| vec![Failure::new(0, "no dealer")])?;
    let mut decks = Vec::new();
    let mut failures = Vec::new();
    for (k, deal) in t.deals.iter().enumerate() {
        let dealer = ((usize::from(first_dealer) + k) % SEATS) as u8;
        match build_deck(dealer, deal) {
            Ok(deck) => decks.push(deck),
            Err(errors) => failures.extend(errors.into_iter().map(|e| {
                Failure::new(
                    deal.line,
                    format!("deal for hand index {k} (dealer seat {dealer}): {e}"),
                )
            })),
        }
    }
    if !failures.is_empty() {
        return Err(failures);
    }
    let count = decks.len();
    Ok((Match::with_decks(first_dealer, decks, 0, t.rules), count))
}

// ----- driver -----

fn action_text(a: Action) -> String {
    match a {
        Action::Burn(c) => format!("> Burn {}", name(c)),
        Action::Play(c) => format!("> Play {}", name(c)),
    }
}

fn phase_text(p: Phase) -> String {
    match p {
        Phase::AwaitBurn(s) => format!("AwaitBurn {s}"),
        Phase::AwaitPlay(s) => format!("AwaitPlay {s}"),
        Phase::MatchOver => "MatchOver".to_string(),
    }
}

fn cards_text(cards: &[Card]) -> String {
    if cards.is_empty() {
        return "(none)".to_string();
    }
    cards.iter().map(|&c| name(c)).collect::<Vec<_>>().join(" ")
}

fn list_text<T: ToString>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn legal_cards(m: &Match) -> Vec<Card> {
    m.legal_actions().into_iter().map(Action::card).collect()
}

/// Replays one transcript and returns every problem found (empty when it passes).
fn run_transcript(text: &str) -> Vec<Failure> {
    let t = match parse_transcript(text) {
        Ok(t) => t,
        Err(errors) => return errors,
    };
    let (mut m, deck_count) = match start_match(&t) {
        Ok(started) => started,
        Err(errors) => return errors,
    };
    let first_dealer = t.dealer.unwrap_or(0);
    let mut failures = Vec::new();
    let mut seen = 0;
    for step in &t.steps {
        let context = match step.action {
            None => "after the deal".to_string(),
            Some((line, action)) => {
                let phase = m.phase();
                let legal = legal_cards(&m);
                if let Err(e) = m.apply(action) {
                    let mut message = format!(
                        "illegal action `{}`: the engine says \"{e}\"; phase {}, legal cards: {}",
                        action_text(action),
                        phase_text(phase),
                        cards_text(&legal)
                    );
                    if m.hand_index() as usize >= deck_count {
                        let _ = write!(
                            message,
                            " (the engine is in hand index {}, past the {deck_count} deal(s) the \
                             file gives, so it dealt a random deck)",
                            m.hand_index()
                        );
                    }
                    failures.push(Failure::new(line, message));
                    return failures;
                }
                format!("after `{}` (line {line})", action_text(action))
            }
        };
        let new_events = &m.events()[seen..];
        check_step(&m, new_events, step, &context, &mut failures);
        check_dealers(new_events, first_dealer, deck_count, step, &mut failures);
        seen = m.events().len();
    }
    let reached = m.hand_index() as usize + 1;
    if reached < deck_count {
        failures.push(Failure::new(
            t.deals[reached].line,
            format!(
                "the file gives {deck_count} deals but play ended in hand index {} (phase {}), \
                 so this deal and any later ones were never dealt",
                m.hand_index(),
                phase_text(m.phase())
            ),
        ));
    }
    failures
}

/// Checks one step's expectations against the state after it and the events it produced.
fn check_step(
    m: &Match,
    new_events: &[(Visibility, Event)],
    step: &Step,
    context: &str,
    failures: &mut Vec<Failure>,
) {
    let public: Vec<String> = new_events
        .iter()
        .filter(|(v, _)| *v == Visibility::Public)
        .map(|(_, e)| e.canonical())
        .collect();
    let mut cursor = 0;
    let mut listed = false;
    for (line, expectation) in &step.expects {
        let problem = match expectation {
            Expect::Event(want) => {
                if let Some(pos) = public[cursor..].iter().position(|e| e == want) {
                    cursor += pos + 1;
                    None
                } else {
                    let mut msg = format!("expected event `{want}` was not produced");
                    if public[..cursor].contains(want) {
                        msg.push_str(" in this order (it comes before an earlier expected event)");
                    } else if new_events
                        .iter()
                        .any(|(v, e)| *v != Visibility::Public && e.canonical() == *want)
                    {
                        msg.push_str(" as a public event (it is private)");
                    }
                    if listed {
                        msg.push_str("; this step's public events are listed above");
                    } else {
                        listed = true;
                        msg.push_str("; this step's public events were:");
                        if public.is_empty() {
                            msg.push_str(" (none)");
                        }
                        for e in &public {
                            let _ = write!(msg, "\n        {e}");
                        }
                    }
                    Some(msg)
                }
            }
            Expect::Scores(want) => (m.scores() != *want).then(|| {
                format!(
                    "scores are {}, expected {}",
                    list_text(&m.scores()),
                    list_text(want)
                )
            }),
            Expect::Phase(want) => (m.phase() != *want).then(|| {
                format!(
                    "phase is {}, expected {}",
                    phase_text(m.phase()),
                    phase_text(*want)
                )
            }),
            Expect::Legal(want) => {
                let actual = legal_cards(m);
                (actual != *want).then(|| {
                    let mut sorted = want.clone();
                    sorted.sort_unstable();
                    let note = if sorted == actual {
                        " (same cards, but the transcript does not list them in card-index order)"
                    } else {
                        ""
                    };
                    format!(
                        "legal cards are {}, expected {}{note}",
                        cards_text(&actual),
                        cards_text(want)
                    )
                })
            }
            Expect::Burns(want) => {
                let actual = m.public().burn_order;
                (actual != *want).then(|| format!("burn order is {actual:?}, expected {want:?}"))
            }
            Expect::Game { team, reason } => {
                let won = new_events.iter().rev().find_map(|(_, e)| match e {
                    Event::GameWon { team, reason, .. } => Some((*team, reason.canonical())),
                    _ => None,
                });
                match won {
                    None => Some(format!(
                        "expected game won by team {team} ({reason}), but no game ended here"
                    )),
                    Some((t, r)) if t == *team && r == reason.as_str() => None,
                    Some((t, r)) => Some(format!(
                        "game won by team {t} ({r}), expected team {team} ({reason})"
                    )),
                }
            }
            Expect::MatchWon { team, money } => {
                let won = new_events.iter().find_map(|(_, e)| match e {
                    Event::MatchWon { team, money } => Some((*team, *money)),
                    _ => None,
                });
                match won {
                    None => Some(format!(
                        "expected the match won by team {team}, but the match did not end here"
                    )),
                    Some((t, mo)) if t != *team || mo != *money => Some(format!(
                        "match won by team {t} with money {}, expected team {team} with money {}",
                        list_text(&mo),
                        list_text(money)
                    )),
                    Some(_) if m.money() != Some(*money) => Some(format!(
                        "money() is {:?}, expected {}",
                        m.money(),
                        list_text(money)
                    )),
                    Some(_) => None,
                }
            }
        };
        if let Some(message) = problem {
            failures.push(Failure::new(*line, format!("{context}: {message}")));
        }
    }
}

/// The harness deals hand k with dealer `(first dealer + k) % 8`; check the engine agrees, or
/// the decks were built for the wrong seats.
fn check_dealers(
    new_events: &[(Visibility, Event)],
    first_dealer: u8,
    deck_count: usize,
    step: &Step,
    failures: &mut Vec<Failure>,
) {
    for (_, e) in new_events {
        if let Event::HandStart {
            hand_index, dealer, ..
        } = *e
            && (hand_index as usize) < deck_count
        {
            let built_for = ((u32::from(first_dealer) + hand_index) % SEATS as u32) as u8;
            if dealer != built_for {
                failures.push(Failure::new(
                    step.action.map_or(0, |(line, _)| line),
                    format!(
                        "hand index {hand_index} is dealt by seat {dealer}, but the harness built \
                         its deck for dealer seat {built_for}"
                    ),
                ));
            }
        }
    }
}

/// Reads and replays one file, turning a panic into a failure so the other files still run.
fn check_file(path: &Path) -> Vec<Failure> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) => return vec![Failure::new(0, format!("cannot read the file: {e}"))],
    };
    match std::panic::catch_unwind(|| run_transcript(&text)) {
        Ok(failures) => failures,
        Err(payload) => {
            let what = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            vec![Failure::new(0, format!("panicked: {what}"))]
        }
    }
}

#[test]
fn all_transcripts() {
    let files = transcript_files();
    assert!(
        !files.is_empty(),
        "no transcripts in {}",
        transcript_dir().display()
    );
    let mut report = String::new();
    let (mut passed, mut failed) = (0, 0);
    for path in &files {
        let file = file_name(path);
        let failures = check_file(path);
        if failures.is_empty() {
            passed += 1;
            println!("PASS {file}");
            continue;
        }
        failed += 1;
        println!("FAIL {file} ({} problem(s))", failures.len());
        for f in &failures {
            println!("    {}", f.render());
        }
        let _ = writeln!(report, "{file}: {} problem(s)", failures.len());
        for f in failures.iter().take(SHOWN_PER_FILE) {
            let _ = writeln!(report, "    {}", f.render());
        }
        if failures.len() > SHOWN_PER_FILE {
            let _ = writeln!(
                report,
                "    ... and {} more",
                failures.len() - SHOWN_PER_FILE
            );
        }
    }
    let summary = format!("{passed} passed, {failed} failed");
    println!("{summary}");
    assert!(failed == 0, "transcript failures:\n{report}{summary}");
}

/// The file's lines with line `at` replaced by `with`, or `with` inserted after it.
fn edit<'a>(lines: &[&'a str], at: usize, replace: bool, with: &'a str) -> String {
    let mut out = lines.to_vec();
    if replace {
        out[at] = with;
    } else {
        out.insert(at + 1, with);
    }
    out.join("\n")
}

/// The harness must catch a wrong expectation of every kind, an illegal action and a bad deal:
/// each edit of the first transcript must be reported.
#[test]
fn harness_reports_disagreements() {
    let path = transcript_files().into_iter().next().expect("a transcript");
    let text = std::fs::read_to_string(&path).expect("readable transcript");
    assert!(
        run_transcript(&text).is_empty(),
        "{} must pass unedited",
        file_name(&path)
    );
    let lines: Vec<&str> = text.lines().collect();
    let first = |prefix: &str| {
        lines
            .iter()
            .position(|l| l.starts_with(prefix))
            .unwrap_or_else(|| panic!("no `{prefix}` line"))
    };
    let word = |prefix: &str, n: usize| lines[first(prefix)].split_whitespace().nth(n);
    let action = first(">");
    let illegal = format!("> Play {}", word("undealt:", 1).expect("an undealt card"));
    let duplicate = format!("turned: {}", word("hand 0:", 2).expect("a card of seat 0"));
    let cases = [
        (
            edit(&lines, first("scores:"), true, "scores: 99,99,99,99"),
            "scores are",
        ),
        (
            edit(&lines, action, false, "= NoSuchEvent"),
            "was not produced",
        ),
        (edit(&lines, action, false, "legal:"), "legal cards are"),
        (edit(&lines, action, false, "phase: MatchOver"), "phase is"),
        (
            edit(&lines, action, false, "result: game team=0 reason=jink"),
            "no game ended here",
        ),
        (edit(&lines, action, true, &illegal), "illegal action"),
        (
            edit(&lines, first("turned:"), true, &duplicate),
            "is given twice",
        ),
    ];
    for (edited, needle) in &cases {
        let failures = run_transcript(edited);
        assert!(
            failures.iter().any(|f| f.message.contains(needle)),
            "expected a problem containing `{needle}`, got {failures:?}"
        );
    }
}

// ----- plain-English rendering for the reviewer -----

/// The engine's plain-English transcript of one file's match, after applying the file's actions
/// up to the first illegal one. `full` adds every seat's dealt and burned cards.
fn render_file(path: &Path, full: bool) -> String {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) => return format!("cannot read the file: {e}\n"),
    };
    let problems = |failures: Vec<Failure>| {
        failures
            .iter()
            .map(|f| format!("{}\n", f.render()))
            .collect::<String>()
    };
    let t = match parse_transcript(&text) {
        Ok(t) => t,
        Err(failures) => return problems(failures),
    };
    let (mut m, _) = match start_match(&t) {
        Ok(started) => started,
        Err(failures) => return problems(failures),
    };
    let mut stopped = String::new();
    for (line, action) in t.steps.iter().filter_map(|s| s.action) {
        if let Err(e) = m.apply(action) {
            stopped = format!(
                "(stopped at line {line}: `{}` is illegal: {e})\n",
                action_text(action)
            );
            break;
        }
    }
    let body = if full {
        transcript_full(&m)
    } else {
        transcript(&m)
    };
    format!("{}\n{body}{stopped}", t.title)
}

/// Prints the engine's plain-English transcript of each file for review (not asserted).
/// `TRANSCRIPT=<part of a file name>` picks files; `TRANSCRIPT_FULL=1` adds the private events.
#[test]
#[ignore = "prints plain-English transcripts for review; run with -- --ignored --nocapture"]
fn transcript_render() {
    let filter = std::env::var("TRANSCRIPT").ok();
    let full = std::env::var_os("TRANSCRIPT_FULL").is_some();
    for path in transcript_files() {
        let file = file_name(&path);
        if filter.as_deref().is_some_and(|f| !file.contains(f)) {
            continue;
        }
        println!("===== {file} =====");
        println!("{}", render_file(&path, full));
    }
}
