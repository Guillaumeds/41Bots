//! Match records (docs/M1-SPEC.md §3.2): every action of a match plus what it led to, replayable
//! exactly, stored as JSON Lines (one record per line).

use engine::cards::{Card, DECK_SIZE};
use engine::game::{
    Action, Event, GameEndReason, HandStartState, IllegalAction, Match, Phase, RULES_VERSION,
    Rules, StateError, Visibility,
};
use engine::{SEATS, TEAMS};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

use crate::table::{BlockSpec, build_match};

/// The record format this crate writes. Version 2 (docs/M3-SPEC.md §2.1) adds the optional
/// mid-match start (`start`, `start_deck`); version-1 records still read and replay.
pub const FORMAT_VERSION: u32 = 2;

/// The oldest record format this crate still reads.
pub const OLDEST_FORMAT_VERSION: u32 = 1;

/// One game of a match.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameSummary {
    /// The game's index in the match, from 0.
    pub game_index: u8,
    /// The winning team.
    pub winner: u8,
    /// How it was won.
    pub reason: GameEndReason,
    /// All four scores when the game ended.
    pub scores: [u8; TEAMS],
    /// Hands dealt in this game, counting the one in which it ended.
    pub hands: u32,
}

/// The public state after one completed trick. The money-chance columns are added at M4.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrickPoint {
    /// The game's index in the match, from 0.
    pub game_index: u8,
    /// The hand's index in the match, from 0.
    pub hand_index: u32,
    /// The trick, 1..=5.
    pub trick: u8,
    /// The seat that won it.
    pub winner: u8,
    /// The scores after the trick and any bonus it brought (after trick 5 this includes the
    /// end-of-hand bonus). If the trick ended the game, the game's final scores.
    pub scores: [u8; TEAMS],
    /// Games won by each team after the trick.
    pub games_won: [u8; TEAMS],
}

/// A whole match: enough to replay it exactly (the actions), plus what it led to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchRecord {
    /// [`FORMAT_VERSION`].
    pub format_version: u32,
    /// `engine::game::RULES_VERSION` of the engine that played it.
    pub rules_version: u32,
    /// The rule switches.
    pub rules: Rules,
    /// The match seed (the bots' seeds, and the deal when `block` is `None`).
    pub seed: u64,
    /// The duplicate block that dealt the cards, if any.
    pub block: Option<BlockSpec>,
    /// The bot (or remote agent) name at each seat.
    pub bots: [String; SEATS],
    /// The first dealer.
    pub first_dealer: u8,
    /// Every action, in order: the only thing needed, with the seed or block, to replay.
    pub actions: Vec<Action>,
    /// Each game's result.
    pub games: Vec<GameSummary>,
    /// The money per team, settled by the engine.
    pub money: [i32; TEAMS],
    /// The team that won the match.
    pub winner: u8,
    /// The public state after every completed trick.
    pub public_timeline: Vec<TrickPoint>,
    /// Format 2: the match started mid-way, at the start of a hand in this state
    /// (`Match::from_hand_start`); `None` for a match from its first deal. Absent in format 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<HandStartState>,
    /// Format 2: the deck of the starting hand when `start` is set (53 cards in deal order).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_deck: Option<Vec<Card>>,
}

/// A record that cannot be read, replayed or verified.
#[derive(Debug)]
pub enum RecordError {
    /// The match is not over, so it has no record yet.
    NotOver,
    /// The record's format version is not one this crate reads.
    FormatVersion(u32),
    /// The record was played under other rules than this engine's `RULES_VERSION`.
    RulesVersion(u32),
    /// The block's first dealer is not a seat.
    BadFirstDealer(u8),
    /// The mid-match start state or deck is not valid.
    BadStart(Option<StateError>),
    /// The engine refused a recorded action.
    Illegal {
        /// The action's position in `actions`.
        index: usize,
        /// The refusal.
        error: IllegalAction,
    },
    /// The replayed match does not give what the record says.
    Mismatch {
        /// The field that differs.
        field: &'static str,
    },
    /// A line of a records file is not a record.
    Json {
        /// The line number, from 1.
        line: usize,
        /// The parser's message.
        error: String,
    },
    /// Reading the file failed.
    Io(io::Error),
}

impl core::fmt::Display for RecordError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RecordError::NotOver => write!(f, "the match is not over"),
            RecordError::FormatVersion(v) => {
                write!(
                    f,
                    "record format version {v}, expected {OLDEST_FORMAT_VERSION} to {FORMAT_VERSION}"
                )
            }
            RecordError::RulesVersion(v) => write!(
                f,
                "record played under rules version {v}, this engine is rules version {RULES_VERSION}"
            ),
            RecordError::BadFirstDealer(d) => write!(f, "first dealer {d} is not a seat"),
            RecordError::BadStart(Some(e)) => write!(f, "bad mid-match start: {e}"),
            RecordError::BadStart(None) => {
                write!(
                    f,
                    "bad mid-match start: start and start_deck must come together"
                )
            }
            RecordError::Illegal { index, error } => {
                write!(f, "recorded action {index} is refused: {error}")
            }
            RecordError::Mismatch { field } => {
                write!(f, "the replayed match does not match the record's {field}")
            }
            RecordError::Json { line, error } => write!(f, "line {line}: {error}"),
            RecordError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for RecordError {}

impl From<io::Error> for RecordError {
    fn from(e: io::Error) -> Self {
        RecordError::Io(e)
    }
}

impl MatchRecord {
    /// The record of a finished match. `seed` and `block` must be the ones the match was built
    /// from (see [`build_match`]); `bots` are the names at each seat.
    pub fn from_match(
        m: &Match,
        seed: u64,
        block: Option<BlockSpec>,
        bots: [String; SEATS],
    ) -> Result<MatchRecord, RecordError> {
        Self::build(m, seed, block, bots, None)
    }

    /// The record of a finished match started mid-way by `Match::from_hand_start(start, deck,
    /// seed, rules)` (format 2, docs/M3-SPEC.md §2.1). The first dealer is the starting hand's
    /// dealer, and the timeline's scores and games continue from `start`.
    pub fn from_match_started(
        m: &Match,
        seed: u64,
        start: HandStartState,
        deck: [Card; DECK_SIZE],
        bots: [String; SEATS],
    ) -> Result<MatchRecord, RecordError> {
        Self::build(m, seed, None, bots, Some((start, deck)))
    }

    fn build(
        m: &Match,
        seed: u64,
        block: Option<BlockSpec>,
        bots: [String; SEATS],
        start: Option<(HandStartState, [Card; DECK_SIZE])>,
    ) -> Result<MatchRecord, RecordError> {
        let money = m.money().ok_or(RecordError::NotOver)?;
        if !m.extras().is_empty() {
            // The table's bots never act illegally (H1), so nobody is ever turned and nothing is
            // let through at a table; a record holds actions only and could not replay either.
            return Err(RecordError::Mismatch {
                field: "extras (a table record holds no turn and no renege let through)",
            });
        }
        let mut first_dealer = 0;
        let mut winner = 0;
        let mut games = Vec::new();
        let mut timeline: Vec<TrickPoint> = Vec::new();
        let mut scores = start.map_or([0u8; TEAMS], |(s, _)| s.scores);
        let mut games_won = start.map_or([0u8; TEAMS], |(s, _)| s.games_won);
        let (mut hand_index, mut game_index, mut hands_in_game) = (0u32, 0u8, 0u32);
        for (vis, e) in m.events() {
            if *vis != Visibility::Public {
                continue;
            }
            match *e {
                Event::MatchStart { first_dealer: d } => first_dealer = d,
                Event::HandStart {
                    hand_index: h,
                    game_index: g,
                    ..
                } => {
                    hand_index = h;
                    game_index = g;
                    hands_in_game += 1;
                }
                Event::TrickWon {
                    trick,
                    seat,
                    team,
                    points,
                } => {
                    let t = usize::from(team % 4);
                    scores[t] = scores[t].saturating_add(points);
                    timeline.push(TrickPoint {
                        game_index,
                        hand_index,
                        trick,
                        winner: seat,
                        scores,
                        games_won,
                    });
                }
                Event::Bonus { team, points, .. } => {
                    let t = usize::from(team % 4);
                    scores[t] = scores[t].saturating_add(points);
                    if let Some(last) = timeline.last_mut() {
                        last.scores = scores;
                    }
                }
                Event::GameWon {
                    team,
                    reason,
                    scores: final_scores,
                } => {
                    let t = usize::from(team % 4);
                    games_won[t] = games_won[t].saturating_add(1);
                    if let Some(last) = timeline.last_mut() {
                        last.scores = final_scores;
                        last.games_won = games_won;
                    }
                    games.push(GameSummary {
                        game_index,
                        winner: team,
                        reason,
                        scores: final_scores,
                        hands: hands_in_game,
                    });
                    hands_in_game = 0;
                    scores = [0; TEAMS];
                }
                Event::MatchWon { team, .. } => winner = team,
                Event::Dealt { .. }
                | Event::Turned { .. }
                | Event::Took { .. }
                | Event::Burned { .. }
                | Event::BurnDone { .. }
                | Event::Played { .. }
                | Event::ProvedNoAce { .. }
                | Event::AceUndealt
                | Event::HandEnd { .. } => {}
                // Never in a table's match: refused above.
                Event::PairTurned { .. } | Event::TrickPassed { .. } => {}
            }
        }
        Ok(MatchRecord {
            format_version: FORMAT_VERSION,
            rules_version: RULES_VERSION,
            rules: m.rules(),
            seed,
            block,
            bots,
            first_dealer,
            actions: m.actions().to_vec(),
            games,
            money,
            winner,
            public_timeline: timeline,
            start: start.map(|(s, _)| s),
            start_deck: start.map(|(_, d)| d.to_vec()),
        })
    }

    /// Rebuild the match: `Match::new(seed)` or the block's `Match::new_block`, then every
    /// recorded action in order. Refuses another format or rules version.
    pub fn replay(&self) -> Result<Match, RecordError> {
        if !(OLDEST_FORMAT_VERSION..=FORMAT_VERSION).contains(&self.format_version) {
            return Err(RecordError::FormatVersion(self.format_version));
        }
        if self.rules_version != RULES_VERSION {
            return Err(RecordError::RulesVersion(self.rules_version));
        }
        if let Some(b) = self.block
            && usize::from(b.first_dealer) >= SEATS
        {
            return Err(RecordError::BadFirstDealer(b.first_dealer));
        }
        let mut m = match (self.start, &self.start_deck) {
            (None, None) => build_match(self.seed, self.block, self.rules),
            (Some(start), Some(deck)) => {
                let deck: [Card; DECK_SIZE] = deck.as_slice().try_into().map_err(|_| {
                    RecordError::BadStart(Some(StateError("the start deck is not 53 cards")))
                })?;
                Match::from_hand_start(start, deck, self.seed, self.rules)
                    .map_err(|e| RecordError::BadStart(Some(e)))?
            }
            _ => return Err(RecordError::BadStart(None)),
        };
        for (index, &a) in self.actions.iter().enumerate() {
            m.apply(a)
                .map_err(|error| RecordError::Illegal { index, error })?;
        }
        Ok(m)
    }

    /// Replay the match and check that it ends where the record says: the first dealer, every
    /// game, the money, the winner and the timeline must all be equal.
    pub fn verify(&self) -> Result<(), RecordError> {
        let m = self.replay()?;
        if m.phase() != Phase::MatchOver {
            return Err(RecordError::Mismatch {
                field: "actions (the match is not over after the last one)",
            });
        }
        let again = match (self.start, &self.start_deck) {
            (Some(start), Some(deck)) => {
                let deck = <[Card; DECK_SIZE]>::try_from(deck.as_slice()).map_err(|_| {
                    RecordError::BadStart(Some(StateError("the start deck is not 53 cards")))
                })?;
                MatchRecord::from_match_started(&m, self.seed, start, deck, self.bots.clone())?
            }
            _ => MatchRecord::from_match(&m, self.seed, self.block, self.bots.clone())?,
        };
        let checks: [(&'static str, bool); 6] = [
            ("rules", again.rules == self.rules),
            ("first_dealer", again.first_dealer == self.first_dealer),
            ("games", again.games == self.games),
            ("money", again.money == self.money),
            ("winner", again.winner == self.winner),
            (
                "public_timeline",
                again.public_timeline == self.public_timeline,
            ),
        ];
        match checks.iter().find(|(_, ok)| !ok) {
            Some(&(field, _)) => Err(RecordError::Mismatch { field }),
            None => Ok(()),
        }
    }

    /// The record as one line of JSON (no newline).
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("a record always serialises")
    }

    /// Parse one line of JSON.
    pub fn from_json(line: &str) -> Result<MatchRecord, serde_json::Error> {
        serde_json::from_str(line)
    }
}

/// Writes records as JSON Lines: one record per line, UTF-8, `\n` endings. Each record is written
/// with a single `write_all`, so whole lines are appended.
pub struct RecordWriter<W: Write> {
    out: W,
}

impl RecordWriter<File> {
    /// Open a records file for appending, creating it if needed.
    pub fn append(path: impl AsRef<Path>) -> io::Result<RecordWriter<File>> {
        let f = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(RecordWriter { out: f })
    }
}

impl<W: Write> RecordWriter<W> {
    /// Write records to any byte sink.
    pub fn new(out: W) -> RecordWriter<W> {
        RecordWriter { out }
    }

    /// Append one record as one line.
    pub fn write(&mut self, record: &MatchRecord) -> io::Result<()> {
        let mut line = record.to_json();
        line.push('\n');
        self.out.write_all(line.as_bytes())
    }

    /// Flush the sink.
    pub fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }

    /// The sink.
    pub fn into_inner(self) -> W {
        self.out
    }
}

/// Reads records from JSON Lines, one per line; blank lines are skipped.
pub struct RecordReader<R: BufRead> {
    lines: io::Lines<R>,
    line_no: usize,
}

impl RecordReader<BufReader<File>> {
    /// Open a records file.
    pub fn open(path: impl AsRef<Path>) -> io::Result<RecordReader<BufReader<File>>> {
        Ok(RecordReader::new(BufReader::new(File::open(path)?)))
    }
}

impl<R: BufRead> RecordReader<R> {
    /// Read records from any buffered byte source.
    pub fn new(input: R) -> RecordReader<R> {
        RecordReader {
            lines: input.lines(),
            line_no: 0,
        }
    }
}

impl<R: BufRead> Iterator for RecordReader<R> {
    type Item = Result<MatchRecord, RecordError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let line = match self.lines.next()? {
                Ok(l) => l,
                Err(e) => return Some(Err(RecordError::Io(e))),
            };
            self.line_no += 1;
            if line.trim().is_empty() {
                continue;
            }
            return Some(
                MatchRecord::from_json(&line).map_err(|e| RecordError::Json {
                    line: self.line_no,
                    error: e.to_string(),
                }),
            );
        }
    }
}
