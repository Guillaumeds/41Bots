//! The hand-outcome model v0 (docs/M1-SPEC.md §6.3): how a hand unfolds trick by trick, as three
//! tables of whole-number counts measured from self-play, expressed relative to the dealer's team
//! (offset 0 = the dealer's team, 1 = the team that leads trick 1, 2 and 3 further left).
//!
//! - `trick_winner`: who wins trick `t`, given the tricks each team has won so far this hand;
//! - `five_in_trick`: whether the 5 of trumps is in trick `t`, given it has not been played yet and
//!   who wins trick `t`;
//! - `end_bonus`: which team holds the best trump at the end of a hand in which the 5 of trumps was
//!   never played, given the final tricks won.
//!
//! The model is *stationary*: it ignores the scores, so the same counts serve every match state.
//! The counts are written to JSON with their sample sizes so the model is inspectable, and turned
//! into fixed-point probabilities (units of 2^-32, each distribution summing to exactly 2^32) for
//! the sweep.

use core::fmt;
use serde::{Deserialize, Serialize};

use crate::money::{ONE, normalise};

/// Number of ways to split `n ≤ 5` tricks among 4 teams, summed over `n = 0..=5`.
pub const N_COMPS: usize = 126;

/// First composition index for each number of tricks played (`COMP_START[n]..COMP_START[n + 1]`).
pub const COMP_START: [usize; 7] = [0, 1, 5, 15, 35, 70, 126];

/// Compositions with at most 4 tricks played (the states before tricks 1 to 5).
pub const N_BEFORE_TRICK: usize = 70;

/// Compositions of exactly 5 tricks (the states after trick 5).
pub const N_FINAL: usize = 56;

const fn build_comps() -> [[u8; 4]; N_COMPS] {
    let mut out = [[0u8; 4]; N_COMPS];
    let mut i = 0;
    let mut n = 0u8;
    while n <= 5 {
        let mut a = 0u8;
        while a <= n {
            let mut b = 0u8;
            while b <= n - a {
                let mut c = 0u8;
                while c <= n - a - b {
                    out[i] = [a, b, c, n - a - b - c];
                    i += 1;
                    c += 1;
                }
                b += 1;
            }
            a += 1;
        }
        n += 1;
    }
    out
}

/// Every split of up to 5 tricks among the 4 teams, ordered by tricks played, then
/// lexicographically.
pub const COMPS: [[u8; 4]; N_COMPS] = build_comps();

const fn pack(tw: [u8; 4]) -> usize {
    ((tw[0] as usize * 6 + tw[1] as usize) * 6 + tw[2] as usize) * 6 + tw[3] as usize
}

const fn build_comp_lookup() -> [u8; 1296] {
    let mut out = [u8::MAX; 1296];
    let mut i = 0;
    while i < N_COMPS {
        out[pack(COMPS[i])] = i as u8;
        i += 1;
    }
    out
}

const COMP_LOOKUP: [u8; 1296] = build_comp_lookup();

/// The index in [`COMPS`] of a split of tricks (at most 5 in total).
///
/// # Panics
/// If more than 5 tricks are given.
#[inline]
pub fn comp_index(tw: [u8; 4]) -> usize {
    assert!(tw.iter().all(|&x| x <= 5), "at most 5 tricks per team");
    let i = COMP_LOOKUP[pack(tw)];
    assert!(i != u8::MAX, "at most 5 tricks in a hand");
    usize::from(i)
}

/// Rule switches recorded with a model (the stake does not affect the model).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RulesRecord {
    /// First best-trump bonus in a game.
    pub bonus_first: u8,
    /// Later best-trump bonuses in the same game.
    pub bonus_later: u8,
    /// Stake per losing player.
    pub stake: i32,
}

impl From<engine::game::Rules> for RulesRecord {
    fn from(r: engine::game::Rules) -> Self {
        RulesRecord {
            bonus_first: r.bonus_first,
            bonus_later: r.bonus_later,
            stake: r.stake,
        }
    }
}

/// Where a model's counts came from (written to its JSON file).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSource {
    /// `self-play` or `synthetic`.
    pub kind: String,
    /// The bot that played every seat (registry name), or `synthetic`.
    pub bot: String,
    /// The estimation seed.
    pub seed: u64,
    /// How each match seed is derived from the estimation seed.
    pub match_seed_rule: String,
    /// How each bot's `start_match` seed is derived from the match seed.
    pub seat_seed_rule: String,
    /// Matches started (the last may be cut off once enough hands are recorded).
    pub matches: u64,
    /// Hands recorded.
    pub hands: u64,
    /// Hands played through all five tricks.
    pub hands_completed: u64,
    /// Hands cut short because a game ended before trick 5.
    pub hands_cut_short: u64,
    /// Tricks recorded.
    pub tricks: u64,
    /// The engine's `RULES_VERSION`.
    pub rules_version: u32,
    /// The rules the matches were played under.
    pub rules: RulesRecord,
}

/// The hand-outcome model: three tables of counts, relative to the dealer's team.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HandModel {
    /// A short name, for example `outcome-v0`.
    pub model_id: String,
    /// Where the counts came from.
    pub source: ModelSource,
    /// `trick_winner[k][o]`: tricks won by offset `o` when the split of earlier tricks this hand
    /// was `COMPS[k]` (so the trick is number `sum + 1`), `k < 70`.
    pub trick_winner: Vec<[u64; 4]>,
    /// `five_in_trick[t − 1][o] = [not in trick, in trick]`: for trick `t` won by offset `o`
    /// while the 5 of trumps had not yet been played this hand.
    pub five_in_trick: [[[u64; 2]; 4]; 5],
    /// `end_bonus[k − 70][o]`: hands completed without the 5 of trumps being played, with final
    /// split `COMPS[k]`, in which offset `o` held the best trump played.
    pub end_bonus: Vec<[u64; 4]>,
}

/// Why a model file could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelError(pub String);

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bad hand-outcome model: {}", self.0)
    }
}

impl std::error::Error for ModelError {}

/// The JSON file format name.
pub const MODEL_FORMAT: &str = "41bots-hand-outcome";

#[derive(Serialize, Deserialize)]
struct ModelFile {
    format: String,
    format_version: u32,
    model_id: String,
    about: Vec<String>,
    source: ModelSource,
    trick_winner: Vec<WinnerRow>,
    five_in_trick: Vec<FiveRow>,
    end_bonus: Vec<BonusRow>,
}

#[derive(Serialize, Deserialize)]
struct WinnerRow {
    trick: u8,
    tricks_won: [u8; 4],
    counts: [u64; 4],
    samples: u64,
}

#[derive(Serialize, Deserialize)]
struct FiveRow {
    trick: u8,
    winner: u8,
    in_trick: u64,
    not_in_trick: u64,
    samples: u64,
}

#[derive(Serialize, Deserialize)]
struct BonusRow {
    tricks_won: [u8; 4],
    counts: [u64; 4],
    samples: u64,
}

fn json<T: Serialize>(v: &T) -> String {
    serde_json::to_string(v).expect("plain data always serialises")
}

impl HandModel {
    /// A model with every count zero.
    pub fn empty(model_id: &str, source: ModelSource) -> Self {
        HandModel {
            model_id: model_id.to_string(),
            source,
            trick_winner: vec![[0; 4]; N_BEFORE_TRICK],
            five_in_trick: [[[0; 2]; 4]; 5],
            end_bonus: vec![[0; 4]; N_FINAL],
        }
    }

    /// A small hand-made model for tests and timing (not measured from play): a mild lead and
    /// dealer advantage at trick 1, "momentum" (teams that have won tricks hold more strength),
    /// the 5 of trumps in about 3 tricks in 4 hands, and the best trump with the teams that won
    /// most tricks.
    pub fn synthetic() -> Self {
        let source = ModelSource {
            kind: "synthetic".into(),
            bot: "synthetic".into(),
            seed: 0,
            match_seed_rule: "none (hand-made counts)".into(),
            seat_seed_rule: "none (hand-made counts)".into(),
            matches: 0,
            hands: 0,
            hands_completed: 0,
            hands_cut_short: 0,
            tricks: 0,
            rules_version: engine::game::RULES_VERSION,
            rules: engine::game::Rules::HOUSE.into(),
        };
        let mut m = HandModel::empty("synthetic-v0", source);
        for (k, row) in m.trick_winner.iter_mut().enumerate() {
            let tw = COMPS[k];
            let t: u8 = tw.iter().sum::<u8>() + 1;
            for (o, c) in row.iter_mut().enumerate() {
                let mut w = 24 + 5 * u64::from(tw[o]);
                if t == 1 && o == 1 {
                    w += 6;
                }
                if t == 1 && o == 0 {
                    w += 3;
                }
                *c = w;
            }
        }
        for t in 1..=5u64 {
            for o in 0..4 {
                let lead_bonus = u64::from(t == 1 && o == 1);
                m.five_in_trick[(t - 1) as usize][o] = [13 - 2 * t, 3 + lead_bonus];
            }
        }
        for (i, row) in m.end_bonus.iter_mut().enumerate() {
            let tw = COMPS[N_BEFORE_TRICK + i];
            for (o, c) in row.iter_mut().enumerate() {
                *c = if tw[o] == 0 {
                    0
                } else {
                    3 * u64::from(tw[o]) + 1
                };
            }
        }
        m
    }

    /// Add one recorded hand (offsets relative to the dealer's team): the winners of the tricks
    /// played, the trick that held the 5 of trumps (1..=5) if it was played, and the offset that
    /// held the best trump played in the hand.
    ///
    /// Every trick played counts toward `trick_winner` and `five_in_trick`, including the tricks
    /// of a hand cut short by the end of a game; `end_bonus` counts only hands that completed all
    /// five tricks without the 5 of trumps.
    pub fn add_hand(&mut self, winners: &[u8], five_trick: Option<u8>, best_trump: Option<u8>) {
        assert!(winners.len() <= 5, "at most five tricks");
        let mut tw = [0u8; 4];
        let mut five_played = false;
        for (i, &w) in winners.iter().enumerate() {
            let t = (i + 1) as u8;
            let o = usize::from(w % 4);
            self.trick_winner[comp_index(tw)][o] += 1;
            if !five_played {
                let here = five_trick == Some(t);
                self.five_in_trick[i][o][usize::from(here)] += 1;
                five_played = here;
            }
            tw[o] += 1;
        }
        if winners.len() == 5
            && five_trick.is_none()
            && let Some(b) = best_trump
        {
            assert!(
                tw[usize::from(b % 4)] > 0,
                "the best trump's team must have won its trick"
            );
            self.end_bonus[comp_index(tw) - N_BEFORE_TRICK][usize::from(b % 4)] += 1;
        }
    }

    /// Add every count of another model (same shape).
    pub fn merge_counts(&mut self, other: &HandModel) {
        for (a, b) in self.trick_winner.iter_mut().zip(&other.trick_winner) {
            for o in 0..4 {
                a[o] += b[o];
            }
        }
        for t in 0..5 {
            for o in 0..4 {
                for f in 0..2 {
                    self.five_in_trick[t][o][f] += other.five_in_trick[t][o][f];
                }
            }
        }
        for (a, b) in self.end_bonus.iter_mut().zip(&other.end_bonus) {
            for o in 0..4 {
                a[o] += b[o];
            }
        }
    }

    /// The JSON text (one table row per line, so the counts are easy to read and diff).
    pub fn to_json(&self) -> String {
        let mut s = String::new();
        s.push_str("{\n");
        s.push_str(&format!("  \"format\": {},\n", json(&MODEL_FORMAT)));
        s.push_str("  \"format_version\": 1,\n");
        s.push_str(&format!("  \"model_id\": {},\n", json(&self.model_id)));
        let about = [
            "Hand-outcome model for the match-equity table (docs/M1-SPEC.md 6.3).",
            "Offsets are relative to the dealer's team: 0 = dealer's team, 1 = the team that leads trick 1, then 2 and 3 further left.",
            "trick_winner: who won trick `trick` given tricks_won (per offset) earlier in the hand; every trick played counts, including tricks of hands cut short by the end of a game.",
            "five_in_trick: for trick `trick` won by `winner` while the 5 of trumps had not yet been played, whether the 5 was in that trick.",
            "end_bonus: hands that completed all five tricks without the 5 of trumps: the offset holding the best trump played, given the final tricks_won.",
            "samples = the sum of the row's counts; probabilities are counts / samples, in fixed point (units of 2^-32) for the sweep.",
        ];
        s.push_str(&format!("  \"about\": {},\n", json(&about)));
        s.push_str(&format!("  \"source\": {},\n", json(&self.source)));
        s.push_str("  \"trick_winner\": [\n");
        let rows: Vec<String> = self
            .trick_winner
            .iter()
            .enumerate()
            .map(|(k, c)| {
                let tw = COMPS[k];
                json(&WinnerRow {
                    trick: tw.iter().sum::<u8>() + 1,
                    tricks_won: tw,
                    counts: *c,
                    samples: c.iter().sum(),
                })
            })
            .collect();
        s.push_str(&format!("    {}\n  ],\n", rows.join(",\n    ")));
        s.push_str("  \"five_in_trick\": [\n");
        let mut rows = Vec::new();
        for t in 0..5 {
            for o in 0..4 {
                let [no, yes] = self.five_in_trick[t][o];
                rows.push(json(&FiveRow {
                    trick: (t + 1) as u8,
                    winner: o as u8,
                    in_trick: yes,
                    not_in_trick: no,
                    samples: yes + no,
                }));
            }
        }
        s.push_str(&format!("    {}\n  ],\n", rows.join(",\n    ")));
        s.push_str("  \"end_bonus\": [\n");
        let rows: Vec<String> = self
            .end_bonus
            .iter()
            .enumerate()
            .map(|(i, c)| {
                json(&BonusRow {
                    tricks_won: COMPS[N_BEFORE_TRICK + i],
                    counts: *c,
                    samples: c.iter().sum(),
                })
            })
            .collect();
        s.push_str(&format!("    {}\n  ]\n", rows.join(",\n    ")));
        s.push_str("}\n");
        s
    }

    /// Read a model from its JSON text, checking that every row is present exactly once and
    /// that every `samples` equals the sum of its counts.
    pub fn from_json(text: &str) -> Result<Self, ModelError> {
        let f: ModelFile =
            serde_json::from_str(text).map_err(|e| ModelError(format!("not valid JSON: {e}")))?;
        if f.format != MODEL_FORMAT || f.format_version != 1 {
            return Err(ModelError(format!(
                "format {} v{} is not {MODEL_FORMAT} v1",
                f.format, f.format_version
            )));
        }
        let mut m = HandModel::empty(&f.model_id, f.source);
        let mut seen_w = [false; N_BEFORE_TRICK];
        for r in &f.trick_winner {
            let n: u8 = r.tricks_won.iter().sum();
            if n > 4 || r.trick != n + 1 {
                return Err(ModelError(format!(
                    "trick_winner row: trick {} with tricks_won {:?}",
                    r.trick, r.tricks_won
                )));
            }
            if r.samples != r.counts.iter().sum::<u64>() {
                return Err(ModelError(format!(
                    "trick_winner row {:?}: samples is not the sum of counts",
                    r.tricks_won
                )));
            }
            let k = comp_index(r.tricks_won);
            if seen_w[k] {
                return Err(ModelError(format!("duplicate row {:?}", r.tricks_won)));
            }
            seen_w[k] = true;
            m.trick_winner[k] = r.counts;
        }
        if seen_w.iter().any(|&b| !b) {
            return Err(ModelError("trick_winner is missing rows".into()));
        }
        let mut seen_f = [[false; 4]; 5];
        for r in &f.five_in_trick {
            if !(1..=5).contains(&r.trick)
                || r.winner > 3
                || r.samples != r.in_trick + r.not_in_trick
            {
                return Err(ModelError(format!(
                    "bad five_in_trick row: trick {} winner {}",
                    r.trick, r.winner
                )));
            }
            let (t, o) = (usize::from(r.trick - 1), usize::from(r.winner));
            if seen_f[t][o] {
                return Err(ModelError("duplicate five_in_trick row".into()));
            }
            seen_f[t][o] = true;
            m.five_in_trick[t][o] = [r.not_in_trick, r.in_trick];
        }
        if seen_f.iter().flatten().any(|&b| !b) {
            return Err(ModelError("five_in_trick is missing rows".into()));
        }
        let mut seen_b = [false; N_FINAL];
        for r in &f.end_bonus {
            let n: u8 = r.tricks_won.iter().sum();
            if n != 5 || r.samples != r.counts.iter().sum::<u64>() {
                return Err(ModelError(format!("bad end_bonus row {:?}", r.tricks_won)));
            }
            if (0..4).any(|o| r.tricks_won[o] == 0 && r.counts[o] > 0) {
                return Err(ModelError(format!(
                    "end_bonus row {:?}: a team that won no trick cannot hold the best-trump trick",
                    r.tricks_won
                )));
            }
            let i = comp_index(r.tricks_won) - N_BEFORE_TRICK;
            if seen_b[i] {
                return Err(ModelError("duplicate end_bonus row".into()));
            }
            seen_b[i] = true;
            m.end_bonus[i] = r.counts;
        }
        if seen_b.iter().any(|&b| !b) {
            return Err(ModelError("end_bonus is missing rows".into()));
        }
        Ok(m)
    }

    /// Turn the counts into fixed-point probabilities (units of 2^-32, each distribution summing
    /// to exactly 2^32, largest-remainder rounding).
    ///
    /// A distribution with no samples at all is replaced by a coarser one (and counted in
    /// [`CompiledModel::backoffs`]): an unseen `trick_winner` row by the pooled row of the same
    /// trick (then uniform); an unseen `five_in_trick` row by the pooled row of the same trick
    /// (then even); an unseen `end_bonus` row by weights proportional to the tricks won (only a
    /// team that won a trick can hold the best-trump trick).
    ///
    /// # Panics
    /// If an `end_bonus` row gives the bonus to a team that won no trick (impossible in play;
    /// [`HandModel::from_json`] refuses such files).
    pub fn compile(&self) -> CompiledModel {
        for (i, row) in self.end_bonus.iter().enumerate() {
            let tw = COMPS[N_BEFORE_TRICK + i];
            assert!(
                (0..4).all(|o| tw[o] > 0 || row[o] == 0),
                "end_bonus row {tw:?} gives the bonus to a team without a trick"
            );
        }
        let mut backoffs = 0u32;
        let mut winner = [[0u64; 4]; N_BEFORE_TRICK];
        for (k, out) in winner.iter_mut().enumerate() {
            let row = self.trick_winner[k];
            let w = if row.iter().any(|&c| c > 0) {
                row
            } else {
                backoffs += 1;
                let n: u8 = COMPS[k].iter().sum();
                let mut pooled = [0u64; 4];
                for row in
                    &self.trick_winner[COMP_START[usize::from(n)]..COMP_START[usize::from(n) + 1]]
                {
                    for (p, &c) in pooled.iter_mut().zip(row) {
                        *p += c;
                    }
                }
                if pooled.iter().any(|&c| c > 0) {
                    pooled
                } else {
                    [1; 4]
                }
            };
            *out = normalise(w.map(u128::from));
        }
        let mut five = [[[0u64; 2]; 4]; 5];
        for (t, five_t) in five.iter_mut().enumerate() {
            for (o, out) in five_t.iter_mut().enumerate() {
                let row = self.five_in_trick[t][o];
                let w = if row.iter().any(|&c| c > 0) {
                    row
                } else {
                    backoffs += 1;
                    let mut pooled = [0u64; 2];
                    for oo in 0..4 {
                        pooled[0] += self.five_in_trick[t][oo][0];
                        pooled[1] += self.five_in_trick[t][oo][1];
                    }
                    if pooled.iter().any(|&c| c > 0) {
                        pooled
                    } else {
                        [1; 2]
                    }
                };
                *out = normalise(w.map(u128::from));
            }
        }
        let mut end_bonus = [[0u64; 4]; N_FINAL];
        for (i, out) in end_bonus.iter_mut().enumerate() {
            let row = self.end_bonus[i];
            let w = if row.iter().any(|&c| c > 0) {
                row
            } else {
                backoffs += 1;
                COMPS[N_BEFORE_TRICK + i].map(u64::from)
            };
            *out = normalise(w.map(u128::from));
        }
        CompiledModel {
            winner,
            five,
            end_bonus,
            backoffs,
        }
    }
}

/// The model in fixed point, ready for the sweep and the simulator. Every distribution sums to
/// exactly `ONE` (2^32).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledModel {
    /// `winner[k][o]`: chance that offset `o` wins the next trick after the split `COMPS[k]`.
    pub winner: [[u64; 4]; N_BEFORE_TRICK],
    /// `five[t − 1][o] = [not in trick, in trick]`, given the 5 is not yet played and offset `o`
    /// wins trick `t`.
    pub five: [[[u64; 2]; 4]; 5],
    /// `end_bonus[k − 70][o]`: chance that offset `o` holds the best trump after the final split
    /// `COMPS[k]`, the 5 never played.
    pub end_bonus: [[u64; 4]; N_FINAL],
    /// Distributions with no samples that were replaced by a coarser one.
    pub backoffs: u32,
}

impl CompiledModel {
    /// Check that every distribution sums to exactly `ONE`.
    pub fn check(&self) -> bool {
        self.winner.iter().all(|r| r.iter().sum::<u64>() == ONE)
            && self
                .five
                .iter()
                .flatten()
                .all(|r| r.iter().sum::<u64>() == ONE)
            && self.end_bonus.iter().all(|r| r.iter().sum::<u64>() == ONE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compositions_are_complete_and_ordered() {
        for n in 0..=5usize {
            let expect = (n + 1) * (n + 2) * (n + 3) / 6;
            assert_eq!(COMP_START[n + 1] - COMP_START[n], expect);
            for k in COMP_START[n]..COMP_START[n + 1] {
                assert_eq!(COMPS[k].iter().map(|&x| usize::from(x)).sum::<usize>(), n);
                assert_eq!(comp_index(COMPS[k]), k);
                if k > COMP_START[n] {
                    assert!(COMPS[k - 1] < COMPS[k]);
                }
            }
        }
        assert_eq!(COMP_START[5], N_BEFORE_TRICK);
        assert_eq!(N_COMPS - N_BEFORE_TRICK, N_FINAL);
    }

    #[test]
    fn json_round_trip_and_validation() {
        let m = HandModel::synthetic();
        let text = m.to_json();
        let back = HandModel::from_json(&text).expect("read back");
        assert_eq!(back, m);
        assert_eq!(back.to_json(), text, "byte-stable");
        // A wrong sample size is refused.
        let bad = text.replacen("\"samples\":", "\"samples\":1", 1);
        assert!(HandModel::from_json(&bad).is_err());
        // A missing row is refused.
        let mut f: serde_json::Value = serde_json::from_str(&text).expect("json");
        f["end_bonus"].as_array_mut().expect("array").pop();
        assert!(HandModel::from_json(&f.to_string()).is_err());
    }

    #[test]
    fn add_hand_counts_and_compile() {
        let mut m = HandModel::empty("t", HandModel::synthetic().source);
        // Offsets 1,1,0,2,1; the 5 in trick 2; offset 2 held the best trump (ignored: 5 played).
        m.add_hand(&[1, 1, 0, 2, 1], Some(2), Some(2));
        assert_eq!(m.trick_winner[comp_index([0, 0, 0, 0])], [0, 1, 0, 0]);
        assert_eq!(m.trick_winner[comp_index([0, 1, 0, 0])], [0, 1, 0, 0]);
        assert_eq!(m.trick_winner[comp_index([1, 2, 1, 0])], [0, 1, 0, 0]);
        assert_eq!(m.five_in_trick[0][1], [1, 0]);
        assert_eq!(m.five_in_trick[1][1], [0, 1]);
        assert_eq!(
            m.five_in_trick[2][0],
            [0, 0],
            "after the 5 is played nothing counts"
        );
        assert!(m.end_bonus.iter().all(|r| r == &[0; 4]));
        // A completed hand without the 5: the best-trump team counts.
        m.add_hand(&[0, 0, 3, 3, 3], None, Some(3));
        assert_eq!(
            m.end_bonus[comp_index([2, 0, 0, 3]) - N_BEFORE_TRICK],
            [0, 0, 0, 1]
        );
        // A hand cut short after two tricks: its tricks count, no end bonus.
        m.add_hand(&[2, 2], None, Some(2));
        assert_eq!(m.trick_winner[comp_index([0, 0, 1, 0])], [0, 0, 1, 0]);
        let c = m.compile();
        assert!(c.check());
        assert!(c.backoffs > 0);
        assert_eq!(
            c.winner[0],
            [ONE / 3 + 1, ONE / 3, ONE / 3, 0],
            "1,1,1 of 3 hands"
        );
        // Backed-off end bonus: proportional to tricks won.
        let k = comp_index([5, 0, 0, 0]) - N_BEFORE_TRICK;
        assert_eq!(c.end_bonus[k], [ONE, 0, 0, 0]);
    }

    #[test]
    fn synthetic_model_compiles_without_backoff() {
        let c = HandModel::synthetic().compile();
        assert!(c.check());
        assert_eq!(c.backoffs, 0);
    }
}
