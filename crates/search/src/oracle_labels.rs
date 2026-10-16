//! Training labels for the oracle networks O_k (docs/M3-HYBRID-SPEC.md §5): positions at the
//! start of trick `k`, each encoded by `crates/nn` and valued exactly by the current solver.
//!
//! **Positions**, by source (recorded per row):
//! - `random` ([`nn::shard::SOURCE_RANDOM`]): a yardstick-style decision root (a `rules-v1`
//!   self-play hand from a stratified match state, a uniformly chosen decision node at a trick
//!   before `k`, every position in the trick and stratum S1–S4 equally often:
//!   [`crate::measure::build_decision`]), a world drawn from the decider's belief (the counted
//!   uniform sampler), then random legal play by every seat to the start of trick `k`. These are
//!   the kind of lines the hybrid's search explores, sensible or not.
//! - `rules` ([`nn::shard::SOURCE_RULES`]): `rules-v1` self-play from a stratified match state to
//!   the start of trick `k`.
//!
//! A line on which the game ends before trick `k` is replayed with fresh randomness.
//!
//! **Target**: the exact value of the position (`solver::solve_fast`, the documented tie rule:
//! the first strictly best card in increasing card index), the same value the solver would give
//! that position inside any search, in the encoder's rotated team order.
//!
//! **Determinism**: row `i` depends only on (seed, source, trick, `i`) through
//! `Rng::derive(seed, &[NN_LABEL, source, trick, i])`; rows are computed on any number of
//! threads into their slots, so the shard is byte-identical for any thread count.
//!
//! **Disjoint from the golden sets**: the golden decision sets use seed 1; seed 1 is refused.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use engine::game::{HandStartState, Match, Phase, Rules};
use engine::rng::Rng;
use nn::encode::encode;
use nn::shard::{Row, SOURCE_RANDOM, SOURCE_RULES};
use solver::{FastOptions, MatchCtx, Position, SolveMem, canonical, from_match, solve_fast_with};

use crate::measure::build_decision;
use crate::pimc::SharedLeaf;
use crate::worlds::{BeliefSource, WorldSource, rebuild};

/// Seed label of label rows.
pub const NN_LABEL: u64 = 0x0A1C;

/// The seed the golden decision sets use; label runs must use another.
pub const GOLDEN_SEED: u64 = 1;

/// Where label positions come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Random legal play from yardstick decision roots inside belief-sampled worlds.
    Random,
    /// `rules-v1` self-play.
    Rules,
}

impl Source {
    /// The shard's source tag.
    pub fn tag(self) -> u8 {
        match self {
            Source::Random => SOURCE_RANDOM,
            Source::Rules => SOURCE_RULES,
        }
    }

    /// Its name on the command line.
    pub fn name(self) -> &'static str {
        match self {
            Source::Random => "random",
            Source::Rules => "rules",
        }
    }

    /// Parse a command-line name.
    pub fn parse(s: &str) -> Result<Source, String> {
        match s {
            "random" => Ok(Source::Random),
            "rules" => Ok(Source::Rules),
            _ => Err(format!("unknown --source {s} (random or rules)")),
        }
    }
}

/// One label run.
#[derive(Clone, Copy, Debug)]
pub struct LabelJob {
    /// The trick whose start the positions are (2..=5).
    pub trick: u8,
    /// Rows.
    pub count: u64,
    /// Master seed (never [`GOLDEN_SEED`]).
    pub seed: u64,
    /// Where positions come from.
    pub source: Source,
    /// Worker threads (never changes a row).
    pub threads: usize,
}

/// Facts about a row besides the row itself (for the sidecar's counts).
#[derive(Clone, Copy, Debug, Default)]
pub struct RowInfo {
    /// Stratum of the match state (1–4).
    pub stratum: u8,
    /// The trick of the decision root (`random` only; 0 for `rules`).
    pub root_trick: u8,
    /// Lines replayed because the game ended before trick `k`.
    pub retries: u32,
    /// Solver nodes of the exact value.
    pub nodes: u64,
}

/// Play random legal actions from `m` to the start of trick `k`; `None` if the hand (or game, or
/// match) ends first.
fn random_to(m: &Match, k: u8, rng: &mut Rng) -> Option<Match> {
    let mut m = m.clone();
    let h = m.hand_index();
    loop {
        if m.phase() == Phase::MatchOver || m.hand_index() != h {
            return None;
        }
        let p = m.public();
        if p.trick_no == k && p.trick_cards.is_empty() && matches!(m.phase(), Phase::AwaitPlay(_)) {
            return Some(m);
        }
        let acts = m.legal_actions();
        m.apply(acts[rng.bounded(acts.len() as u64) as usize])
            .expect("legal");
    }
}

/// Play `rules-v1` at every seat from `m` to the start of trick `k`; `None` if the hand ends
/// first.
fn rules_to(m: &Match, k: u8) -> Option<Match> {
    let mut m = m.clone();
    let h = m.hand_index();
    loop {
        if m.phase() == Phase::MatchOver || m.hand_index() != h {
            return None;
        }
        let p = m.public();
        if p.trick_no == k && p.trick_cards.is_empty() && matches!(m.phase(), Phase::AwaitPlay(_)) {
            return Some(m);
        }
        let seat = m.to_act().expect("someone acts");
        let legal = m.legal_actions();
        let a = bots::decide(&m.view(seat), &Rules::HOUSE, &legal).0;
        m.apply(a).expect("legal");
    }
}

/// The position of row `i` (before valuing).
pub fn row_position(
    job: &LabelJob,
    i: u64,
    starts: &[Vec<HandStartState>; 4],
    source: &dyn WorldSource,
) -> (Position, MatchCtx, RowInfo) {
    let k = job.trick;
    let st = (i % 4) as usize;
    let mut rng = Rng::derive(
        job.seed,
        &[NN_LABEL, u64::from(job.source.tag()), u64::from(k), i],
    );
    let mut info = RowInfo {
        stratum: st as u8 + 1,
        ..RowInfo::default()
    };
    loop {
        let m = match job.source {
            Source::Random => {
                let t = 1 + rng.bounded(u64::from(k) - 1) as u8;
                let pos = ((i / 4) % 8) as usize;
                let d = build_decision(t, i, st, pos, &mut rng, starts);
                info.root_trick = t;
                let (_, view, _) = d.replay();
                let worlds = source.prepare(&view, Rules::HOUSE);
                let world = worlds.sample(&mut rng);
                let m = rebuild(&view, &world, Rules::HOUSE, rng.next_u64())
                    .expect("sampled worlds rebuild");
                random_to(&m, k, &mut rng)
            }
            Source::Rules => {
                let start = starts[st][rng.bounded(starts[st].len() as u64) as usize];
                let mut deck: [u8; engine::cards::DECK_SIZE] = core::array::from_fn(|c| c as u8);
                rng.shuffle(&mut deck);
                let m = Match::from_hand_start(start, deck, 0, Rules::HOUSE).expect("valid");
                rules_to(&m, k)
            }
        };
        match m {
            Some(m) => {
                let (pos, ctx, pending) = from_match(&m).expect("a play is awaited");
                debug_assert!(pending.is_none(), "no rob after trick 1");
                return (pos, ctx, info);
            }
            None => info.retries += 1,
        }
    }
}

/// Transposition-table megabytes for exact solves from the start of trick `k`.
pub fn tt_mb_for(k: u8) -> usize {
    match k {
        2 => 256,
        3 => 64,
        _ => 8,
    }
}

/// Every row of a job, in row order, and each row's facts. Byte-identical for any thread count.
///
/// # Panics
/// If `job.seed` is the golden-set seed or `job.trick` is not 2..=5.
pub fn label_rows(
    job: &LabelJob,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    progress: Option<&AtomicUsize>,
) -> Vec<(Row, RowInfo)> {
    assert_ne!(
        job.seed, GOLDEN_SEED,
        "seed 1 is the golden sets' seed: labels must use another"
    );
    assert!((2..=5).contains(&job.trick), "frontier tricks are 2..=5");
    let n = job.count as usize;
    let slots: Mutex<Vec<Option<(Row, RowInfo)>>> = Mutex::new(vec![None; n]);
    let next = AtomicUsize::new(0);
    let source = BeliefSource::new();
    let opts = FastOptions {
        tt_mb: tt_mb_for(job.trick),
        ..FastOptions::default()
    };
    let work = || {
        let mut mem = SolveMem::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            let (pos, ctx, mut info) = row_position(job, i as u64, starts, &source);
            let sol = solve_fast_with(&pos, ctx, &crate::pimc::DynLeaf(leaf), opts, &mut mem);
            info.nodes = sol.stats.nodes;
            let v = sol.value.expect("no time cap");
            let (features, r) = encode(&canonical(&pos), &ctx, leaf);
            let row = Row {
                features,
                value: v.rotate(r % 4),
                source: job.source.tag(),
                trick: job.trick,
            };
            slots.lock().unwrap_or_else(|e| e.into_inner())[i] = Some((row, info));
            if let Some(p) = progress {
                p.fetch_add(1, Ordering::Relaxed);
            }
        }
    };
    let threads = job.threads.max(1).min(n.max(1));
    if threads == 1 {
        work();
    } else {
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(work);
            }
        });
    }
    slots
        .into_inner()
        .unwrap_or_else(|e| e.into_inner())
        .into_iter()
        .map(|r| r.expect("every row made"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use equity::{MatchState, MoneyChance};
    use nn::encode::{decode, rotate_canon};
    use nn::synthetic::SyntheticLeaf;
    use solver::{FnLeaf, solve_fast};

    use crate::measure::collect_starts;

    fn leaf() -> Arc<SharedLeaf> {
        Arc::new(SyntheticLeaf)
    }

    #[test]
    fn rows_are_thread_independent_and_exact() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        for (trick, source) in [(4, Source::Random), (5, Source::Rules), (3, Source::Random)] {
            let job = |threads| LabelJob {
                trick,
                count: 24,
                seed: 7,
                source,
                threads,
            };
            let a = label_rows(&job(1), &starts, &*l, None);
            let b = label_rows(&job(3), &starts, &*l, None);
            let ra: Vec<Row> = a.iter().map(|x| x.0).collect();
            let rb: Vec<Row> = b.iter().map(|x| x.0).collect();
            assert_eq!(ra, rb, "trick {trick} {source:?}");
            for (i, (row, info)) in a.iter().enumerate() {
                assert_eq!(row.trick, trick);
                assert_eq!(info.stratum, (i % 4) as u8 + 1);
                // The row's target is the exact value of its position, rotated as encoded.
                let (pos, ctx, _) = row_position(&job(1), i as u64, &starts, &BeliefSource::new());
                assert_eq!(usize::from(pos.hand.tricks_done) + 1, usize::from(trick));
                let exact = solve_fast(&pos, ctx, &SyntheticLeaf, FastOptions::default())
                    .value
                    .expect("solved");
                let c = canonical(&pos);
                let r = nn::encode::rotation(&c);
                assert_eq!(row.value, exact.rotate(r % 4));
                let (dc, dctx) = decode(&row.features).expect("decodes");
                assert_eq!((dc, dctx), rotate_canon(&c, &ctx, r));
            }
        }
    }

    #[test]
    #[should_panic(expected = "golden")]
    fn golden_seed_is_refused() {
        let starts = collect_starts(100, 9);
        let l: Arc<SharedLeaf> = Arc::new(FnLeaf(|_: &MatchState| MoneyChance::even()));
        label_rows(
            &LabelJob {
                trick: 4,
                count: 1,
                seed: GOLDEN_SEED,
                source: Source::Rules,
                threads: 1,
            },
            &starts,
            &*l,
            None,
        );
    }
}
