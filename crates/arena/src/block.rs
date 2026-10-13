//! Duplicate blocks (docs/M1-SPEC.md §5.1) and their exact scores.
//!
//! Slots are the four team positions `{0,4}, {1,5}, {2,6}, {3,7}` (slot = team = seat mod 4). An
//! [`Assignment`] is a 4-bit mask: bit `t` set means candidate A plays slot `t`, clear means
//! incumbent B does.
//!
//! **Seeds (arena choices, documented here).**
//! - Block `i` of a run with master seed `m` and stage `g` has block seed
//!   `Rng::derive(m, &[0xA4E, g, i]).next_u64()`; the decks and first dealer come from it through
//!   `tablemgr::BlockSpec::new` (that is `engine::game::block_deck` / `block_first_dealer`).
//! - Every match of a block uses the same match seed
//!   `Rng::derive(block_seed, &[0x3A7C]).next_u64()`, so the bot at seat `s` gets the same seed in
//!   every assignment (true duplicate play: identical bots make identical decisions).
//! - With `split_seeds`, B's bots get `Rng::derive(seat_seed, &[0x5B]).next_u64()` instead, so two
//!   copies of the same *random* bot are different players (used to measure σ_b under "no
//!   difference" for a bot whose play depends on its seed).
//!
//! **Score units.** A block score is kept as a whole number in units of `2^-32 / 168` of a money
//! chance (and `1/168` of a stake for the money), because 168 is divisible by every
//! `matches × team size` the two block kinds need (14·1, 14·2, 14·3, 4·1, 4·3). Sums are exact, so
//! the canary can demand exactly zero.

use bot::{Bot, BotFactory};
use engine::SEATS;
use engine::game::Rules;
use engine::rng::Rng;
use equity::MoneyChance;
use tablemgr::{BlockSpec, MatchRecord, TableConfig, TableError, play_match};

/// Common denominator of a block score (see the module notes).
pub const SCORE_DEN: i64 = 168;
/// Label of the block-seed derivation.
pub const BLOCK_SEED_LABEL: u64 = 0xA4E;
/// Label of the match-seed derivation inside a block.
pub const MATCH_SEED_LABEL: u64 = 0x3A7C;
/// Label of the B-side seed split.
pub const SPLIT_LABEL: u64 = 0x5B;

/// A 4-bit mask of the slots candidate A plays.
pub type Assignment = u8;

/// Which assignments a block plays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BlockKind {
    /// Every mix but AAAA and BBBB: 4 × 1v3, 6 × 2v2, 4 × 3v1.
    Block14,
    /// A in one slot, B in the other three, rotated (the cheap pre-screen).
    Rot4,
}

impl BlockKind {
    /// The assignments, in a fixed order: by number of A slots, then by mask.
    pub fn assignments(self) -> Vec<Assignment> {
        let mut v: Vec<Assignment> = (1u8..15)
            .filter(|m| self == BlockKind::Block14 || m.count_ones() == 1)
            .collect();
        v.sort_by_key(|m| (m.count_ones(), *m));
        v
    }
}

/// The block seed of block `index` in stage `stage` of a run with master seed `master`.
pub fn block_seed(master: u64, stage: u64, index: u64) -> u64 {
    Rng::derive(master, &[BLOCK_SEED_LABEL, stage, index]).next_u64()
}

/// The match seed shared by every match of a block.
pub fn match_seed(block_seed: u64) -> u64 {
    Rng::derive(block_seed, &[MATCH_SEED_LABEL]).next_u64()
}

/// Passes a different seed to the wrapped bot (see `split_seeds`).
struct SplitSeed(Box<dyn Bot>);

impl Bot for SplitSeed {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn start_match(&mut self, seat: u8, rules: Rules, seed: u64) {
        self.0
            .start_match(seat, rules, Rng::derive(seed, &[SPLIT_LABEL]).next_u64());
    }
    fn act(
        &mut self,
        view: &engine::view::SeatView,
        legal: &[engine::game::Action],
    ) -> engine::game::Action {
        self.0.act(view, legal)
    }
}

/// Totals for the diagnostics (never scored): summed over the slot-matches of one side.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SideTotals {
    /// Slot-matches played (one slot in one match).
    pub slot_matches: u64,
    /// Games won.
    pub games: u64,
    /// Points at the end of every game (all games of the match).
    pub points: u64,
    /// Tricks won.
    pub tricks: u64,
    /// Matches won.
    pub matches: u64,
}

impl SideTotals {
    /// Add another total.
    pub fn add(&mut self, o: &SideTotals) {
        self.slot_matches += o.slot_matches;
        self.games += o.games;
        self.points += o.points;
        self.tricks += o.tricks;
        self.matches += o.matches;
    }
}

/// One played block.
#[derive(Clone, Debug)]
pub struct BlockResult {
    /// The block seed.
    pub block_seed: u64,
    /// Block score `s`, in units of `2^-32 / SCORE_DEN` of a money chance.
    pub score: i64,
    /// The same score in money: units of `stake / SCORE_DEN`.
    pub stake_score: i64,
    /// Per mix (index = number of A slots − 1): the sum of per-match scores in units of
    /// `2^-32 / 6`, and the number of matches.
    pub mix: [(i64, u64); 3],
    /// Diagnostics for A's slots and B's slots.
    pub diag: [SideTotals; 2],
    /// The match records, in assignment order.
    pub records: Vec<MatchRecord>,
}

/// Play one block: the same deal for every assignment of `kind`.
pub fn play_block(
    kind: BlockKind,
    block_seed: u64,
    a: &dyn BotFactory,
    b: &dyn BotFactory,
    rules: Rules,
    split_seeds: bool,
) -> Result<BlockResult, TableError> {
    let assigns = kind.assignments();
    let n = assigns.len() as i64;
    let cfg = TableConfig::in_block(match_seed(block_seed), rules, BlockSpec::new(block_seed));
    let mut out = BlockResult {
        block_seed,
        score: 0,
        stake_score: 0,
        mix: [(0, 0); 3],
        diag: [SideTotals::default(); 2],
        records: Vec::with_capacity(assigns.len()),
    };
    for &mask in &assigns {
        let mut seats: [Box<dyn Bot>; SEATS] = core::array::from_fn(|s| {
            if mask >> (s % 4) & 1 == 1 {
                a.build()
            } else if split_seeds {
                Box::new(SplitSeed(b.build())) as Box<dyn Bot>
            } else {
                b.build()
            }
        });
        let rec = play_match(&cfg, &mut seats)?;
        let chance = MoneyChance::certain(rec.winner);
        let na = i64::from(mask.count_ones() as u8);
        let nb = 4 - na;
        let mut match_score6: i64 = 0; // units 2^-32 / 6
        for t in 0..4usize {
            let is_a = mask >> t & 1 == 1;
            let c = chance.0[t] as i64; // 0 or 2^32
            let m = i64::from(rec.money[t]);
            let stake = i64::from(rules.stake);
            assert!(m % stake == 0, "money is a whole number of stakes");
            let (w, side) = if is_a {
                (SCORE_DEN / (n * na), 0)
            } else {
                (-SCORE_DEN / (n * nb), 1)
            };
            out.score += w * c;
            out.stake_score += w * (m / stake);
            match_score6 += if is_a { 6 / na * c } else { -6 / nb * c };
            let d = &mut out.diag[side];
            d.slot_matches += 1;
            d.matches += u64::from(rec.winner as usize == t);
            d.games += rec.games.iter().filter(|g| g.winner as usize == t).count() as u64;
            d.points += rec
                .games
                .iter()
                .map(|g| u64::from(g.scores[t]))
                .sum::<u64>();
            d.tricks += rec
                .public_timeline
                .iter()
                .filter(|p| p.winner as usize % 4 == t)
                .count() as u64;
        }
        let k = (na - 1) as usize;
        out.mix[k].0 += match_score6;
        out.mix[k].1 += 1;
        out.records.push(rec);
    }
    Ok(out)
}

/// A block score as a money chance (a float, for reports only).
pub fn score_f64(units: i64) -> f64 {
    units as f64 / (SCORE_DEN as f64 * 4_294_967_296.0)
}

/// A block stake score in stakes (a float, for reports only).
pub fn stake_f64(units: i64) -> f64 {
    units as f64 / SCORE_DEN as f64
}
