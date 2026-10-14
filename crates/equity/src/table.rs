//! The match-equity table: one [`MoneyChance`] per canonical match state, stored as
//! `equity-v<N>.npy` (`uint64`, shape `[16, 15, 15, 15, 15, 4]`, little-endian, C order) with a
//! JSON side file `equity-v<N>.json` (docs/M1-SPEC.md §6.2).

use core::fmt;
use serde::{Deserialize, Serialize};

use crate::model::{HandModel, ModelSource, RulesRecord};
use crate::money::MoneyChance;
use crate::npy::{from_npy_u64, to_npy_u64};
use crate::sha256::sha256_hex;
use crate::state::{MatchState, SCORE_VALUES, StateError, TABLE_SHAPE, TABLE_STATES, value_via};
use crate::sweep::SweepStats;

/// Why a table could not be used or read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TableError {
    /// The state is not valid.
    State(StateError),
    /// The state's entry was not computed (a reduced test sweep).
    NotSolved(MatchState),
    /// The file is not a valid table.
    File(String),
}

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TableError::State(e) => write!(f, "{e}"),
            TableError::NotSolved(s) => write!(f, "state {s} is not in this (reduced) table"),
            TableError::File(e) => write!(f, "bad table file: {e}"),
        }
    }
}

impl std::error::Error for TableError {}

/// One money chance per canonical state (dealer team 0), indexed by
/// [`MatchState::canonical_index`]. An entry of all zeros marks a state a reduced sweep did not
/// compute; a complete table has none.
#[derive(Clone, PartialEq, Eq)]
pub struct EquityTable {
    values: Vec<[u64; 4]>,
}

impl fmt::Debug for EquityTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EquityTable({} of {} states solved)",
            self.solved_count(),
            TABLE_STATES
        )
    }
}

impl EquityTable {
    /// A table with nothing solved yet.
    pub fn unsolved() -> Self {
        EquityTable {
            values: vec![[0; 4]; TABLE_STATES],
        }
    }

    /// The value at a canonical index, or `None` if it was not computed.
    #[inline]
    pub fn get(&self, index: usize) -> Option<MoneyChance> {
        MoneyChance::from_raw(self.values[index])
    }

    pub(crate) fn set(&mut self, index: usize, v: MoneyChance) {
        self.values[index] = v.0;
    }

    /// The raw entries (team order of the canonical state: team 0 is the dealer's team).
    pub fn raw(&self) -> &[[u64; 4]] {
        &self.values
    }

    /// Number of computed entries.
    pub fn solved_count(&self) -> usize {
        self.values.iter().filter(|v| **v != [0; 4]).count()
    }

    /// Is every entry computed?
    pub fn is_complete(&self) -> bool {
        self.values.iter().all(|v| *v != [0; 4])
    }

    /// The value of any state, in its own team order: rotate to canonical form, look up, rotate
    /// back. A finished match is certain for its winner.
    pub fn try_lookup(&self, state: &MatchState) -> Result<MoneyChance, TableError> {
        state.validate().map_err(TableError::State)?;
        if let Some(w) = state.winner() {
            return Ok(MoneyChance::certain(w));
        }
        let (c, _) = state.canonical();
        let i = c.canonical_index().ok_or(TableError::NotSolved(*state))?;
        if self.get(i).is_none() {
            return Err(TableError::NotSolved(*state));
        }
        Ok(value_via(state, |j| {
            self.get(j).unwrap_or_else(|| unreachable!("checked above"))
        }))
    }

    /// Like [`EquityTable::try_lookup`].
    ///
    /// # Panics
    /// If the state is invalid or was not computed.
    pub fn lookup(&self, state: &MatchState) -> MoneyChance {
        self.try_lookup(state)
            .unwrap_or_else(|e| panic!("equity lookup failed: {e}"))
    }

    /// The `.npy` bytes (complete tables only).
    pub fn to_npy(&self) -> Result<Vec<u8>, TableError> {
        if !self.is_complete() {
            return Err(TableError::File(format!(
                "only {} of {TABLE_STATES} states are solved",
                self.solved_count()
            )));
        }
        let flat: Vec<u64> = self.values.iter().flatten().copied().collect();
        Ok(to_npy_u64(&TABLE_SHAPE, &flat))
    }

    /// Read a table from `.npy` bytes, checking the shape and that every entry sums to exactly
    /// `ONE`.
    pub fn from_npy(bytes: &[u8]) -> Result<Self, TableError> {
        let (shape, data) = from_npy_u64(bytes).map_err(|e| TableError::File(e.to_string()))?;
        if shape != TABLE_SHAPE {
            return Err(TableError::File(format!(
                "shape {shape:?} is not {TABLE_SHAPE:?}"
            )));
        }
        let mut values = Vec::with_capacity(TABLE_STATES);
        for (i, c) in data.as_chunks::<4>().0.iter().enumerate() {
            let v = *c;
            if MoneyChance::from_raw(v).is_none() {
                return Err(TableError::File(format!(
                    "entry {i} ({}) does not sum to exactly 2^32",
                    MatchState::from_canonical_index(i)
                )));
            }
            values.push(v);
        }
        Ok(EquityTable { values })
    }
}

/// The JSON side file of a table (`equity-v<N>.json`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableInfo {
    /// `41bots-match-equity`.
    pub format: String,
    /// 1.
    pub format_version: u32,
    /// The table version `N` of `equity-v<N>`.
    pub table_version: u32,
    /// The `.npy` file name.
    pub file: String,
    /// SHA-256 of the `.npy` bytes.
    pub sha256: String,
    /// The first 16 hex characters of `sha256` (the content-addressed id).
    pub content_id: String,
    /// `uint64`.
    pub dtype: String,
    /// `[16, 15, 15, 15, 15, 4]`.
    pub shape: Vec<usize>,
    /// Plain-English description of the layout and units.
    pub layout: Vec<String>,
    /// The score of each score index.
    pub score_values: Vec<u8>,
    /// The engine's `RULES_VERSION`.
    pub rules_version: u32,
    /// The rules the table is for.
    pub rules: RulesRecord,
    /// The hand-outcome model used.
    pub model: ModelInfo,
    /// What the sweep did.
    pub sweep: SweepStats,
}

/// The model a table was built from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// The model's id.
    pub model_id: String,
    /// SHA-256 of the model's JSON file.
    pub sha256: String,
    /// Where its counts came from (bot, seed, matches, hands, …).
    pub source: ModelSource,
    /// Sample sizes of the three tables.
    pub samples: ModelSamples,
    /// Distributions with no samples, replaced by a coarser one.
    pub backoffs: u32,
}

/// Total samples in each table of the model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSamples {
    /// Tricks counted in `trick_winner`.
    pub trick_winner: u64,
    /// Tricks counted in `five_in_trick`.
    pub five_in_trick: u64,
    /// Hands counted in `end_bonus`.
    pub end_bonus: u64,
    /// The smallest non-zero `trick_winner` row.
    pub smallest_trick_winner_row: u64,
    /// The smallest non-zero `end_bonus` row.
    pub smallest_end_bonus_row: u64,
}

impl ModelSamples {
    /// Count a model's samples.
    pub fn of(m: &HandModel) -> Self {
        let rows_w: Vec<u64> = m.trick_winner.iter().map(|r| r.iter().sum()).collect();
        let rows_b: Vec<u64> = m.end_bonus.iter().map(|r| r.iter().sum()).collect();
        ModelSamples {
            trick_winner: rows_w.iter().sum(),
            five_in_trick: m.five_in_trick.iter().flatten().flatten().sum(),
            end_bonus: rows_b.iter().sum(),
            smallest_trick_winner_row: rows_w.iter().copied().filter(|&x| x > 0).min().unwrap_or(0),
            smallest_end_bonus_row: rows_b.iter().copied().filter(|&x| x > 0).min().unwrap_or(0),
        }
    }
}

/// The plain-English layout lines written to every side file.
pub fn layout_notes() -> Vec<String> {
    [
        "Each entry is four uint64 chances of winning the match (the money), one per team, in units of 2^-32; the four sum to exactly 2^32.",
        "States are canonical: teams rotated so the dealer's team is team 0 (team 1 leads trick 1, then 2 and 3 to the left). Look up any state by rotating it so its dealer team is 0, indexing, and rotating the four values back.",
        "Axis 0: games bits, bit t = games won by team t (0 or 1) after rotation. Axes 1-4: score index of teams 0-3 (see score_values). Axis 5: team.",
        "A team has already taken its first best-trump bonus this game exactly when its score mod 5 == 1 (house rule, bonus 6 then 5).",
        "Values are for the state between hands, with the next hand dealt by a player of team 0.",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

impl TableInfo {
    /// Describe a table file.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        table_version: u32,
        file: &str,
        npy: &[u8],
        rules: engine::game::Rules,
        model: &HandModel,
        model_sha256: &str,
        backoffs: u32,
        sweep: SweepStats,
    ) -> Self {
        let sha = sha256_hex(npy);
        TableInfo {
            format: "41bots-match-equity".into(),
            format_version: 1,
            table_version,
            file: file.to_string(),
            content_id: sha[..16].to_string(),
            sha256: sha,
            dtype: "uint64".into(),
            shape: TABLE_SHAPE.to_vec(),
            layout: layout_notes(),
            score_values: SCORE_VALUES.to_vec(),
            rules_version: engine::game::RULES_VERSION,
            rules: rules.into(),
            model: ModelInfo {
                model_id: model.model_id.clone(),
                sha256: model_sha256.to_string(),
                source: model.source.clone(),
                samples: ModelSamples::of(model),
                backoffs,
            },
            sweep,
        }
    }
}
