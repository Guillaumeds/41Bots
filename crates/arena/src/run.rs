//! Running blocks on several threads with results merged in block order, and the summary of a
//! sequence of blocks.
//!
//! Blocks are played in chunks of [`CHUNK`] (a constant, not the thread count); inside a chunk
//! worker threads take block indices from a shared counter and store each result in its slot;
//! the caller then sees the results strictly in block order and may stop after any block. Blocks
//! after the stop point are discarded, so what the caller sees (and the records written) is the
//! same for any number of threads.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use bot::BotFactory;
use engine::game::Rules;
use serde::{Deserialize, Serialize};
use tablemgr::TableError;

use crate::block::{BlockKind, BlockResult, SCORE_DEN, SideTotals, block_seed, play_block};
use crate::sprt::Moments;

/// Blocks per chunk.
pub const CHUNK: u64 = 32;

/// One sequence of blocks.
#[derive(Clone, Debug)]
pub struct RunConfig {
    /// Block kind.
    pub kind: BlockKind,
    /// Candidate A's registry name.
    pub a: String,
    /// Incumbent B's registry name.
    pub b: String,
    /// Rule switches.
    pub rules: Rules,
    /// Master seed of the run.
    pub master_seed: u64,
    /// Stage number (part of the block-seed derivation, so stages use fresh deals).
    pub stage: u64,
    /// Worker threads (does not change any result).
    pub threads: usize,
    /// Give B's bots different seeds (see `block`).
    pub split_seeds: bool,
}

fn factory(name: &str) -> Result<Box<dyn BotFactory + Send + Sync>, String> {
    // The registry (M3-SPEC §8.1): PIMC bots by name, falling back to `bots`.
    search::factory(name).ok_or_else(|| search::bot::factory_error(name))
}

/// Play blocks `0, 1, …` of `cfg` until `max_blocks` or until `visit` returns `false`, calling
/// `visit(index, &result)` in block order. Returns the number of blocks visited.
pub fn run_blocks<F>(cfg: &RunConfig, max_blocks: u64, mut visit: F) -> Result<u64, String>
where
    F: FnMut(u64, &BlockResult) -> bool,
{
    let a = factory(&cfg.a)?;
    let b = factory(&cfg.b)?;
    let threads = cfg.threads.max(1);
    let mut start = 0u64;
    while start < max_blocks {
        let end = (start + CHUNK).min(max_blocks);
        let slots: Mutex<Vec<Option<Result<BlockResult, TableError>>>> =
            Mutex::new((start..end).map(|_| None).collect());
        let next = AtomicU64::new(start);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= end {
                            break;
                        }
                        let seed = block_seed(cfg.master_seed, cfg.stage, i);
                        let r = play_block(cfg.kind, seed, &*a, &*b, cfg.rules, cfg.split_seeds);
                        slots.lock().expect("no panics while holding the lock")
                            [(i - start) as usize] = Some(r);
                    }
                });
            }
        });
        let results = slots.into_inner().expect("threads joined");
        for (i, r) in (start..).zip(results) {
            let r = r
                .expect("every slot filled")
                .map_err(|e| format!("block {i}: {e}"))?;
            if !visit(i, &r) {
                return Ok(i + 1);
            }
        }
        start = end;
    }
    Ok(max_blocks)
}

/// Exact totals over a sequence of blocks, plus float views for the report.
#[derive(Clone, Debug, Default)]
pub struct Summary {
    /// Money-chance block scores.
    pub chance: Moments,
    /// Stake-unit block scores.
    pub stake: Moments,
    /// Per mix: Σ per-match score (units 2^-32/6) and match count.
    pub mix: [(i128, u64); 3],
    /// Diagnostics for A and B.
    pub diag: [SideTotals; 2],
    /// Every block score (units of `block::SCORE_DEN`), in order.
    pub scores: Vec<i64>,
}

impl Summary {
    /// Add one block.
    pub fn push(&mut self, r: &BlockResult) {
        self.chance.push(r.score);
        self.stake.push(r.stake_score);
        for k in 0..3 {
            self.mix[k].0 += i128::from(r.mix[k].0);
            self.mix[k].1 += r.mix[k].1;
        }
        self.diag[0].add(&r.diag[0]);
        self.diag[1].add(&r.diag[1]);
        self.scores.push(r.score);
    }

    /// Mean and standard deviation of the block score in money-chance units.
    pub fn chance_mean_sd(&self) -> (f64, f64) {
        let (m, v) = self.chance.mean_var(SCORE_DEN as f64 * 4_294_967_296.0);
        (m, v.sqrt())
    }

    /// Mean and standard deviation of the block score in stakes.
    pub fn stake_mean_sd(&self) -> (f64, f64) {
        let (m, v) = self.stake.mean_var(SCORE_DEN as f64);
        (m, v.sqrt())
    }

    /// The report form.
    pub fn report(&self) -> SummaryReport {
        let (mu, sd) = self.chance_mean_sd();
        let (smu, ssd) = self.stake_mean_sd();
        let mix = core::array::from_fn(|k| {
            let (s, n) = self.mix[k];
            MixReport {
                mix: ["1v3", "2v2", "3v1"][k].to_string(),
                matches: n,
                mean: if n == 0 {
                    0.0
                } else {
                    s as f64 / n as f64 / (6.0 * 4_294_967_296.0)
                },
            }
        });
        let side = |d: &SideTotals| {
            let n = d.slot_matches.max(1) as f64;
            SideReport {
                slot_matches: d.slot_matches,
                match_win_rate: d.matches as f64 / n,
                games_per_match: d.games as f64 / n,
                points_per_match: d.points as f64 / n,
                tricks_per_match: d.tricks as f64 / n,
            }
        };
        SummaryReport {
            blocks: self.chance.n,
            mean: mu,
            sd,
            stake_mean: smu,
            stake_sd: ssd,
            score_sum_units: self.chance.sum.to_string(),
            mix,
            diagnostics: Diagnostics {
                a: side(&self.diag[0]),
                b: side(&self.diag[1]),
            },
        }
    }
}

/// Mean per-match score for one mix.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MixReport {
    /// `1v3`, `2v2` or `3v1` (A slots v B slots).
    pub mix: String,
    /// Matches of this mix.
    pub matches: u64,
    /// Mean per-match score (money chance).
    pub mean: f64,
}

/// Diagnostics for one side, per slot-match. Never scored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SideReport {
    /// Slot-matches.
    pub slot_matches: u64,
    /// Share of slot-matches won.
    pub match_win_rate: f64,
    /// Games won per slot-match.
    pub games_per_match: f64,
    /// Points per slot-match (sum of final game scores).
    pub points_per_match: f64,
    /// Tricks won per slot-match.
    pub tricks_per_match: f64,
}

/// Diagnostics of both sides.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Diagnostics {
    /// Candidate A.
    pub a: SideReport,
    /// Incumbent B.
    pub b: SideReport,
}

/// The report form of a [`Summary`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SummaryReport {
    /// Blocks.
    pub blocks: u64,
    /// μ, money chance.
    pub mean: f64,
    /// σ (block standard deviation, σ_b), money chance.
    pub sd: f64,
    /// μ in stakes.
    pub stake_mean: f64,
    /// σ_b in stakes.
    pub stake_sd: f64,
    /// Exact Σ block scores in units of 2^-32/168 (decimal string; the canary checks it is 0).
    pub score_sum_units: String,
    /// Per-mix breakdown.
    pub mix: [MixReport; 3],
    /// Games, points, tricks (diagnostics only).
    pub diagnostics: Diagnostics,
}

/// The result of one sequential-test stage.
#[derive(Clone, Debug)]
pub struct StageOutcome {
    /// Totals over the blocks played.
    pub summary: Summary,
    /// The decision.
    pub decision: crate::sprt::Decision,
    /// LLR after the last block.
    pub llr: f64,
}

/// Run one SPRT stage: blocks in order until the test decides or `max_blocks` is reached.
/// `reject_only` makes it a stage-1 pre-screen. Every match record of every block counted is
/// passed to `record` in block order.
pub fn sprt_stage<R>(
    cfg: &RunConfig,
    params: &crate::sprt::SprtParams,
    reject_only: bool,
    max_blocks: u64,
    mut record: R,
) -> Result<StageOutcome, String>
where
    R: FnMut(&tablemgr::MatchRecord) -> Result<(), String>,
{
    use crate::sprt::{Decision, decide, llr};
    let mut summary = Summary::default();
    let mut decision = Decision::Continue;
    let mut last = 0.0;
    let mut err = None;
    run_blocks(cfg, max_blocks, |_, r| {
        summary.push(r);
        for rec in &r.records {
            if let Err(e) = record(rec) {
                err = Some(e);
                return false;
            }
        }
        let (mu, sd) = summary.chance_mean_sd();
        last = llr(summary.chance.n, mu, sd * sd, params.s0, params.s1);
        decision = decide(summary.chance.n, last, params, reject_only, max_blocks);
        decision == Decision::Continue
    })?;
    if let Some(e) = err {
        return Err(e);
    }
    if decision == Decision::Continue {
        decision = Decision::Inconclusive;
    }
    Ok(StageOutcome {
        summary,
        decision,
        llr: last,
    })
}

/// The canary (docs/M1-SPEC.md §5.4): `cfg.a` must equal `cfg.b`; every block score (money chance
/// and stakes) must be exactly 0, so the mean is exactly 0. Returns the summary, or which block
/// broke it.
pub fn canary(cfg: &RunConfig, blocks: u64) -> Result<Summary, String> {
    if cfg.a != cfg.b || cfg.split_seeds {
        return Err("the canary needs identical bots with identical seeds".into());
    }
    let mut s = Summary::default();
    let mut bad = None;
    run_blocks(cfg, blocks, |i, r| {
        s.push(r);
        if r.score != 0 || r.stake_score != 0 {
            bad = Some(format!(
                "canary failed: block {i} (seed {}) scored {} / {} units, not 0",
                r.block_seed, r.score, r.stake_score
            ));
            return false;
        }
        true
    })?;
    if let Some(e) = bad {
        return Err(e);
    }
    if s.chance.sum != 0 || s.stake.sum != 0 {
        return Err("canary failed: the block scores do not sum to exactly 0".into());
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sprt::{Decision, SprtParams};

    fn cfg(kind: BlockKind, a: &str, b: &str, threads: usize) -> RunConfig {
        RunConfig {
            kind,
            a: a.into(),
            b: b.into(),
            rules: Rules::HOUSE,
            master_seed: 7,
            stage: 0,
            threads,
            split_seeds: false,
        }
    }

    #[test]
    fn assignments() {
        let a = BlockKind::Block14.assignments();
        assert_eq!(a.len(), 14);
        let by = |k| a.iter().filter(|m| m.count_ones() == k).count();
        assert_eq!((by(1), by(2), by(3)), (4, 6, 4));
        assert_eq!(BlockKind::Rot4.assignments(), vec![1, 2, 4, 8]);
    }

    #[test]
    fn canary_scores_exactly_zero() {
        for bot in ["rules-v1", "random"] {
            let s = canary(&cfg(BlockKind::Block14, bot, bot, 2), 5).expect("canary");
            assert_eq!(s.chance.n, 5);
            assert!(s.scores.iter().all(|&x| x == 0));
            assert_eq!(s.chance_mean_sd(), (0.0, 0.0));
        }
        let s = canary(&cfg(BlockKind::Rot4, "random", "random", 1), 5).expect("rot4 canary");
        assert_eq!(s.chance.sum, 0);
    }

    #[test]
    fn canary_catches_different_bots() {
        assert!(canary(&cfg(BlockKind::Block14, "rules-v1", "random", 1), 1).is_err());
    }

    #[test]
    fn same_deals_across_assignments_and_replay() {
        let mut first = None;
        run_blocks(
            &cfg(BlockKind::Block14, "rules-v1", "random", 1),
            1,
            |_, r| {
                first = Some(r.clone());
                false
            },
        )
        .unwrap();
        let r = first.unwrap();
        for rec in &r.records {
            assert_eq!(rec.block, r.records[0].block);
            assert_eq!(rec.seed, r.records[0].seed);
            rec.verify().expect("record replays exactly");
        }
    }

    #[test]
    fn results_independent_of_thread_count() {
        let run = |threads| {
            let mut v = Vec::new();
            run_blocks(
                &cfg(BlockKind::Rot4, "rules-v1", "random", threads),
                40,
                |_, r| {
                    v.push((r.score, r.stake_score, r.records.clone()));
                    true
                },
            )
            .unwrap();
            v
        };
        let one = run(1);
        assert_eq!(one, run(3));
        assert_eq!(one, run(4));
        assert!(one.iter().any(|x| x.0 != 0));
    }

    #[test]
    fn sprt_stage_deterministic_and_rejects_random() {
        // Random as candidate against rules-v1 should fail the gainer test quickly.
        let go = |threads| {
            let mut n = 0u64;
            let o = sprt_stage(
                &cfg(BlockKind::Rot4, "random", "rules-v1", threads),
                &SprtParams::GAINER,
                true,
                200,
                |_| {
                    n += 1;
                    Ok(())
                },
            )
            .unwrap();
            (o.decision, o.summary.chance.n, o.summary.scores, n)
        };
        let a = go(1);
        assert_eq!(a.0, Decision::Reject);
        assert!(a.1 >= crate::sprt::MIN_BLOCKS);
        assert_eq!(a.3, a.1 * 4);
        assert_eq!(a, go(4));
    }

    #[test]
    fn ratings_order_rules_above_random() {
        let mut recs = Vec::new();
        run_blocks(
            &cfg(BlockKind::Block14, "rules-v1", "random", 4),
            4,
            |_, r| {
                recs.extend(r.records.iter().cloned());
                true
            },
        )
        .unwrap();
        let rt = crate::rating::rate(&recs);
        assert_eq!(rt.len(), 2);
        assert_eq!(rt[0].program, "random");
        assert_eq!(rt[0].r, Some(0.0));
        let rules = &rt[1];
        assert!(rules.r.unwrap() > 0.0, "{rt:?}");
        assert!(rules.se.unwrap() > 0.0);
    }
}
