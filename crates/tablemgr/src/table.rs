//! The in-process table (docs/M1-SPEC.md §3.1) and the seed rules (§1).

use bot::Bot;
use engine::SEATS;
use engine::game::{Action, IllegalAction, Match, Rules, block_deck, block_first_dealer};
use engine::rng::Rng;
use engine::view::SeatView;
use serde::{Deserialize, Serialize};

use crate::record::MatchRecord;

/// The label of the seat-seed derivation (docs/M1-SPEC.md §1).
pub const SEAT_SEED_LABEL: u64 = 0xB07;

/// The seed the bot at `seat` receives in `start_match` for a match with seed `match_seed`:
/// `Rng::derive(match_seed, &[0xB07, seat]).next_u64()` (docs/M1-SPEC.md §1).
pub fn seat_seed(match_seed: u64, seat: u8) -> u64 {
    Rng::derive(match_seed, &[SEAT_SEED_LABEL, u64::from(seat)]).next_u64()
}

/// A duplicate block's deal: the deck for hand `k` is `engine::game::block_deck(block_seed, k)`
/// and the first dealer is given (normally `block_first_dealer(block_seed)`, see
/// [`BlockSpec::new`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockSpec {
    /// The block seed, shared by every match of the block.
    pub block_seed: u64,
    /// The first dealer of every match of the block.
    pub first_dealer: u8,
}

impl BlockSpec {
    /// The standard block for this seed: first dealer `block_first_dealer(block_seed)`.
    pub fn new(block_seed: u64) -> BlockSpec {
        BlockSpec {
            block_seed,
            first_dealer: block_first_dealer(block_seed),
        }
    }
}

/// What the table needs to run one match.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableConfig {
    /// The match seed. It always gives the bots' seeds (see [`seat_seed`]); without a block it
    /// also gives the first dealer and every deck (`Match::new(seed, rules)`).
    pub seed: u64,
    /// The rule switches.
    pub rules: Rules,
    /// A duplicate block's deal, which replaces the seed's dealer and decks.
    pub block: Option<BlockSpec>,
}

impl TableConfig {
    /// A match dealt from its own seed.
    pub fn plain(seed: u64, rules: Rules) -> TableConfig {
        TableConfig {
            seed,
            rules,
            block: None,
        }
    }

    /// A match of a duplicate block; `seed` still gives the bots' seeds.
    pub fn in_block(seed: u64, rules: Rules, block: BlockSpec) -> TableConfig {
        TableConfig {
            seed,
            rules,
            block: Some(block),
        }
    }

    /// The match this configuration describes, at its first decision.
    ///
    /// # Panics
    /// If a block's first dealer is not a seat (0..=7).
    pub fn new_match(&self) -> Match {
        build_match(self.seed, self.block, self.rules)
    }
}

/// The match for a seed and an optional block: `Match::new(seed, rules)`, or
/// `Match::new_block(block_seed, |k| block_deck(block_seed, k), first_dealer, rules)`.
///
/// # Panics
/// If a block's first dealer is not a seat (0..=7).
pub fn build_match(seed: u64, block: Option<BlockSpec>, rules: Rules) -> Match {
    match block {
        None => Match::new(seed, rules),
        Some(b) => {
            let block_seed = b.block_seed;
            Match::new_block(
                block_seed,
                move |k| block_deck(block_seed, k),
                b.first_dealer,
                rules,
            )
        }
    }
}

/// Why the table stopped a match before its end.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TableError {
    /// A seat chose an action that is not among the legal ones. It was not applied.
    IllegalAction {
        /// The seat that chose it.
        seat: u8,
        /// The name of the bot (or remote agent) at that seat.
        bot: String,
        /// The refused action.
        action: Action,
        /// The legal actions it should have chosen from.
        legal: Vec<Action>,
    },
    /// The engine refused an action the table had checked as legal (an engine bug).
    EngineRefused {
        /// The seat to act.
        seat: u8,
        /// The engine's refusal.
        error: IllegalAction,
    },
    /// A networked seat closed its link, or the link failed.
    Disconnected {
        /// The seat.
        seat: u8,
    },
    /// A networked seat sent a message the protocol does not allow at that point.
    Protocol {
        /// The seat.
        seat: u8,
        /// What was wrong, in plain English.
        detail: String,
    },
}

impl TableError {
    /// The seat at fault.
    pub fn seat(&self) -> u8 {
        match *self {
            TableError::IllegalAction { seat, .. }
            | TableError::EngineRefused { seat, .. }
            | TableError::Disconnected { seat }
            | TableError::Protocol { seat, .. } => seat,
        }
    }

    /// A reason that names no card, safe to send to every seat (an abort must not reveal the
    /// faulty seat's hand, which other matches of a duplicate block deal again).
    pub fn public_reason(&self) -> String {
        match self {
            TableError::IllegalAction { seat, .. } => {
                format!("seat {seat} chose an illegal action; the match is stopped")
            }
            TableError::EngineRefused { .. } => {
                "the engine refused an action the table accepted (a table bug); the match is stopped"
                    .to_string()
            }
            TableError::Disconnected { seat } => {
                format!("seat {seat} disconnected; the match is stopped")
            }
            TableError::Protocol { seat, .. } => {
                format!("seat {seat} broke the protocol; the match is stopped")
            }
        }
    }
}

impl core::fmt::Display for TableError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TableError::IllegalAction {
                seat,
                bot,
                action,
                legal,
            } => write!(
                f,
                "seat {seat} ({bot}) chose {action:?}, which is not legal (legal: {legal:?}); \
                 it was not applied"
            ),
            TableError::EngineRefused { seat, error } => write!(
                f,
                "the engine refused seat {seat}'s action that the table had checked as legal: {error}"
            ),
            TableError::Disconnected { seat } => write!(f, "seat {seat} disconnected"),
            TableError::Protocol { seat, detail } => {
                write!(f, "seat {seat} broke the protocol: {detail}")
            }
        }
    }
}

impl std::error::Error for TableError {}

/// Run the match to its end. For each decision, `decide(seat, view, legal)` is given only the
/// acting seat's view and the legal actions; an action not in `legal` stops the match with
/// [`TableError::IllegalAction`] and is never applied. `names` are the agents' names, for errors.
pub fn drive_match<F>(
    m: &mut Match,
    names: &[String; SEATS],
    mut decide: F,
) -> Result<(), TableError>
where
    F: FnMut(u8, &SeatView, &[Action]) -> Result<Action, TableError>,
{
    while let Some(seat) = m.to_act() {
        let view = m.view(seat);
        let legal = m.legal_actions();
        let action = decide(seat, &view, &legal)?;
        if !legal.contains(&action) {
            return Err(TableError::IllegalAction {
                seat,
                bot: names[usize::from(seat)].clone(),
                action,
                legal,
            });
        }
        m.apply(action)
            .map_err(|error| TableError::EngineRefused { seat, error })?;
    }
    Ok(())
}

/// Play one match in this program (docs/M1-SPEC.md §3.1): build the match from `cfg`, start
/// every bot with its seat seed, then ask the seat to act for each decision until the money.
/// Each bot sees only its own [`SeatView`] and the legal actions.
pub fn play_match(
    cfg: &TableConfig,
    seats: &mut [Box<dyn Bot>; SEATS],
) -> Result<MatchRecord, TableError> {
    let names: [String; SEATS] = core::array::from_fn(|s| seats[s].name().to_string());
    let mut m = cfg.new_match();
    for (s, bot) in (0u8..).zip(seats.iter_mut()) {
        bot.start_match(s, cfg.rules, seat_seed(cfg.seed, s));
    }
    drive_match(&mut m, &names, |seat, view, legal| {
        Ok(seats[usize::from(seat)].act(view, legal))
    })?;
    Ok(MatchRecord::from_match(&m, cfg.seed, cfg.block, names)
        .expect("drive_match returns Ok only once the match is over"))
}
