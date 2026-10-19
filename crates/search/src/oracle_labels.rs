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
//! - `harvest` ([`nn::shard::SOURCE_HARVEST`], spec §5 source (c); DAgger, Ross et al. 2011: train
//!   on the positions the search itself visits): see [`harvest_rows`]. A training decision drawn
//!   by the yardstick's generator, a world drawn from the decider's belief, then every frontier
//!   position (start of trick `k`) that the hybrid's search from the decision reaches for every
//!   candidate, recorded through the solver's frontier hook, deduplicated by canonical form, and a
//!   seeded sample of at most a cap per decision. The search's frontier values come from a
//!   [`Driver`]: exact values (every reached position solved; the default, cheap at trick 4) or a
//!   network's (the previous round's O_k, as the hybrid really runs; only the sampled positions
//!   are solved, which is what makes harvesting at trick 3 affordable). A harvest may draw its
//!   decisions from some strata only ([`HarvestJob::strata`]) and keep the positions of chosen
//!   race classes (`nn::outcome::race_class`) up to a cap of their own ([`RaceKeep`]: the race
//!   harvest, docs/m3-hybrid/rung1.md "Why seats 3-4 fail"), or the positions inside any group of
//!   a predicate file up to a cap of their own ([`PredKeep`]: the audit loop's targeted harvest,
//!   docs/m3-hybrid/audit-loop.md §4 step 5).
//!
//! A line on which the game ends before trick `k` is replayed with fresh randomness.
//!
//! **Target**: the exact value of the position (`solver::solve_fast`, the documented tie rule:
//! the first strictly best card in increasing card index), the same value the solver would give
//! that position inside any search, in the encoder's rotated team order; and for the outcome
//! head (spec §6a, `nn::outcome`) the position's possible outcomes, their exact money vectors and
//! the label, the first possible outcome whose vector is the exact value. A position with no such
//! outcome would disprove the vocabulary's completeness: the run stops on it.
//!
//! **Determinism**: row `i` depends only on (seed, source, trick, `i`) through
//! `Rng::derive(seed, &[NN_LABEL, source, trick, i])`; rows are computed on any number of
//! threads and put in row order, so the shard is byte-identical for any thread count. Harvested
//! rows come by decision: decision `j`'s rows depend only on (seed, trick, `j`), the harvest's
//! settings and the driver (exact, or the network's weights), and the shard is decisions 0, 1,
//! 2, … in order, cut at the row count. Each decision's search runs on one thread (the
//! sequential solver), so the positions it reaches, and the order it reaches them in, never
//! depend on the thread count; a network's per-thread caches never change a value
//! (`nn::oracle`). Each label is a pure function of its position (the exact value; the solver's
//! table never changes it), so where and with what a label is solved never changes a row
//! ([`LabelExec`]: the table's size and the [`HarvestSchedule`]; tested).
//!
//! **Streaming** ([`write_label_shard`], [`write_harvest_shard`]): a shard of millions of rows is
//! never held in memory. Workers make chunks of consecutive rows, each claimed as a worker comes
//! free (`crate::ordered`), already as the shard's bytes; the chunks are hashed (SHA-256, for the
//! sidecar) in row order as they arrive and written in that order by a writer thread, so hashing
//! and writing overlap the making of later rows instead of following it. A harvest's decisions
//! are searched and sampled one per worker, in order, and their sampled positions pooled into one
//! list whose labels every worker takes in turn (`crate::ordered::pooled`, the default
//! [`HarvestSchedule::Pooled`]): at trick 2, where a label takes seconds to minutes and a
//! decision has 16 to 270 of them, no thread is left with one decision's labels while the others
//! wait (measured 2026-10-17 on 245 threads: the per-decision schedule ran at 7% of the plain
//! labels' speed), and no label is made for a decision past the row count. The bytes and the hash
//! are those of [`nn::shard::to_npy`] of [`label_rows`] / [`harvest_rows`] (tested), for any
//! thread count, chunk size and schedule.
//!
//! **Disjoint from the golden sets**: the golden decision sets use seed 1; seed 1 is refused.

use std::collections::BTreeMap;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use engine::game::{HandStartState, Match, Phase, Rules};
use engine::rng::Rng;
use equity::MoneyChance;
use nn::encode::encode;
use nn::outcome::{N_RACE_CLASSES, outcomes};
use nn::predicate::{Evaluator, Predicates};
use nn::shard::{Row, SOURCE_HARVEST, SOURCE_RANDOM, SOURCE_RULES, row_bytes, row_bytes_for};
use serde::Serialize;
use solver::frontier::{CanonPos, Frontier};
use solver::{
    FastOptions, FrontierCut, MatchCtx, Position, SolveFamily, SolveMem, canonical, from_match,
    solve_fast_cut, solve_fast_with,
};

use crate::measure::{DecisionSpec, build_decision};
use crate::ordered::{ordered, window_for};
use crate::pimc::{
    Candidate, Ctx, DynLeaf, SearchConfig, SharedLeaf, WorldSolve, candidates, exact_world,
};
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
    /// Frontier positions the hybrid's search reaches from training decisions ([`harvest_rows`]).
    Harvest,
}

impl Source {
    /// The shard's source tag.
    pub fn tag(self) -> u8 {
        match self {
            Source::Random => SOURCE_RANDOM,
            Source::Rules => SOURCE_RULES,
            Source::Harvest => SOURCE_HARVEST,
        }
    }

    /// Its name on the command line.
    pub fn name(self) -> &'static str {
        match self {
            Source::Random => "random",
            Source::Rules => "rules",
            Source::Harvest => "harvest",
        }
    }

    /// Parse a command-line name.
    pub fn parse(s: &str) -> Result<Source, String> {
        match s {
            "random" => Ok(Source::Random),
            "rules" => Ok(Source::Rules),
            "harvest" => Ok(Source::Harvest),
            _ => Err(format!("unknown --source {s} (random, rules or harvest)")),
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
    /// The trick of the decision root (`random` and `harvest`; 0 for `rules`).
    pub root_trick: u8,
    /// The decider's position in the trick, 1 (leader) to 8 (`random` and `harvest`; 0 for
    /// `rules`).
    pub position: u8,
    /// Lines replayed because the game ended before trick `k`.
    pub retries: u32,
    /// Solver nodes of the exact value. For a harvested row: the nodes of its solve on the
    /// table its decision's solves share (`solver::SolveFamily`), fewer than a solve from an
    /// empty table when an earlier position of the decision already solved part of its future.
    pub nodes: u64,
    /// The position's race class (`nn::outcome::race_class`, from the row: [`Row::race_class`]).
    pub class: u8,
    /// Valued by the hybrid ([`Labeller::Hybrid`]) instead of exactly; `nodes` are then the
    /// hybrid search's.
    pub hybrid: bool,
    /// Frontier (network) evaluations of a hybrid label's search (0 for an exact label).
    pub evals: u64,
}

/// Rows' facts totalled (the sidecar's counts); the totals do not depend on the order rows are
/// added in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowCounts {
    /// Rows.
    pub rows: u64,
    /// Rows by stratum 1–4 (index stratum − 1).
    pub by_stratum: [u64; 4],
    /// Rows by the trick of the decision root (index 0: `rules`, no root).
    pub by_root_trick: [u64; 6],
    /// Rows by the decider's position 1–8 (index 0: `rules`, no decider).
    pub by_position: [u64; 9],
    /// Lines replayed because the game ended before the trick.
    pub retries: u64,
    /// Solver nodes of the exact values.
    pub nodes: u64,
    /// Rows by race class 0–4 (`nn::outcome::race_class`).
    pub by_class: [u64; N_RACE_CLASSES],
    /// Rows valued by the hybrid ([`Labeller::Hybrid`]; 0 with exact labels).
    pub hybrid: u64,
    /// Frontier (network) evaluations of the hybrid labels' searches.
    pub evals: u64,
}

impl RowCounts {
    /// Count one row.
    pub fn add(&mut self, i: &RowInfo) {
        self.rows += 1;
        self.by_stratum[usize::from(i.stratum - 1)] += 1;
        self.by_root_trick[usize::from(i.root_trick)] += 1;
        self.by_position[usize::from(i.position)] += 1;
        self.retries += u64::from(i.retries);
        self.nodes += i.nodes;
        self.by_class[usize::from(i.class)] += 1;
        self.hybrid += u64::from(i.hybrid);
        self.evals += i.evals;
    }

    /// Add another total.
    pub fn merge(&mut self, o: &RowCounts) {
        self.rows += o.rows;
        for (a, b) in self.by_stratum.iter_mut().zip(o.by_stratum) {
            *a += b;
        }
        for (a, b) in self.by_root_trick.iter_mut().zip(o.by_root_trick) {
            *a += b;
        }
        for (a, b) in self.by_position.iter_mut().zip(o.by_position) {
            *a += b;
        }
        self.retries += o.retries;
        self.nodes += o.nodes;
        for (a, b) in self.by_class.iter_mut().zip(o.by_class) {
            *a += b;
        }
        self.hybrid += o.hybrid;
        self.evals += o.evals;
    }
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
                info.position = pos as u8 + 1;
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
            Source::Harvest => panic!("harvested rows come by decision: harvest_rows"),
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

/// The exact value of a position at the start of a trick (`solver::solve_fast_with`, the
/// documented tie rule) and the solver nodes it took.
pub(crate) fn exact_value(
    pos: &Position,
    ctx: MatchCtx,
    leaf: &SharedLeaf,
    opts: FastOptions,
    mem: &mut SolveMem,
) -> (MoneyChance, u64) {
    let sol = solve_fast_with(pos, ctx, &DynLeaf(leaf), opts, mem);
    (sol.value.expect("no time cap"), sol.stats.nodes)
}

/// Seed label of the exact anchor's draw ([`HybridLabels::exact_ppm`]).
pub const ANCHOR: u64 = 0xA2C4;

/// How a label's value is made (docs/m3-hybrid/audit-loop.md §12, "Trick-k labels from the
/// trick-(k+1) network"). Only training shards are ever labelled by the hybrid; the validation
/// shard, the audit shards and every golden set stay exact (the runner refuses otherwise).
#[derive(Clone, Copy, Default)]
pub enum Labeller<'a> {
    /// The exact value (`solver::solve_fast_with` to the end of the hand): every label before
    /// 2026-10-19, byte for byte.
    #[default]
    Exact,
    /// The hybrid H_{k+1} of a position at the start of trick `k` ([`hybrid_value`]).
    Hybrid(HybridLabels<'a>),
}

/// The hybrid labels' settings ([`Labeller::Hybrid`]).
#[derive(Clone, Copy)]
pub struct HybridLabels<'a> {
    /// The network at the start of the next trick (normally the integer O_{k+1}, `nn::Oracle`);
    /// the caller checks that it was trained for that trick (`search labels --label-net` does).
    pub net: &'a dyn Frontier,
    /// The exact anchor: rows labelled exactly instead, in parts per million (0: none). Row
    /// `a`, place `b` of a shard with seed `s` and source tag `t` at trick `k` is exact when
    /// `Rng::derive(s, [ANCHOR, t, k, a, b]).bounded(10⁶) < exact_ppm` ([`anchored`]): `a` is the
    /// row index and `b` 0 for random-play and rules-v1 rows, `a` the decision and `b` the
    /// position's place in the decision's sample for harvested rows (whose row index depends on
    /// the decisions before). A pure function of the row's key, for any thread count and
    /// schedule.
    pub exact_ppm: u32,
}

impl Labeller<'_> {
    /// Whether the row with key (`a`, `b`) of a shard (seed, source tag, trick) is labelled
    /// exactly ([`HybridLabels::exact_ppm`]); always with [`Labeller::Exact`].
    pub fn exact_row(&self, seed: u64, source: u8, trick: u8, a: u64, b: u64) -> bool {
        match self {
            Labeller::Exact => true,
            Labeller::Hybrid(h) => anchored(h.exact_ppm, seed, source, trick, a, b),
        }
    }

    /// The name a sidecar records (`label_source`).
    pub fn name(&self) -> &'static str {
        match self {
            Labeller::Exact => "exact",
            Labeller::Hybrid(_) => "hybrid",
        }
    }
}

/// The exact anchor's draw ([`HybridLabels::exact_ppm`]).
pub fn anchored(ppm: u32, seed: u64, source: u8, trick: u8, a: u64, b: u64) -> bool {
    ppm >= 1_000_000
        || (ppm > 0
            && Rng::derive(seed, &[ANCHOR, u64::from(source), u64::from(trick), a, b])
                .bounded(1_000_000)
                < u64::from(ppm))
}

/// The share of `--exact-share` (a decimal fraction from 0 to 1, at most six decimals) in parts
/// per million, parsed from its digits (no float rounding).
pub fn parse_exact_share(s: &str) -> Result<u32, String> {
    let t = s.trim();
    let bad = || format!("{s:?}: a share from 0 to 1 with at most six decimals, e.g. 0.1");
    let (int, frac) = t.split_once('.').unwrap_or((t, ""));
    if int.is_empty() && frac.is_empty()
        || frac.len() > 6
        || !int.bytes().all(|b| b.is_ascii_digit())
        || !frac.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(bad());
    }
    let i: u64 = if int.is_empty() {
        0
    } else {
        int.parse().map_err(|_| bad())?
    };
    let f: u64 = format!("{frac:0<6}").parse().map_err(|_| bad())?;
    let ppm = i
        .checked_mul(1_000_000)
        .and_then(|x| x.checked_add(f))
        .ok_or_else(bad)?;
    if ppm > 1_000_000 {
        return Err(bad());
    }
    Ok(ppm as u32)
}

/// The solver table (megabytes) of a hybrid label's search (one trick to the frontier: small,
/// and it never changes a value).
pub const HYBRID_TT_MB: usize = 8;

/// The hybrid value H_{k+1} of a position at the start of trick `k`: the exact solver
/// (`solver::solve_fast_cut`, the same max^n, tie rule and leaf rule at the end of the hand or
/// the game) through trick `k`, `net`'s value of every position reached at the start of trick
/// `k + 1` (its canonical form, as everywhere the hybrid runs: `search hybrid-check`, `search
/// worlds --method hybrid`, the student's teacher), and the solver nodes and frontier
/// evaluations it took.
pub fn hybrid_value(
    pos: &Position,
    ctx: MatchCtx,
    leaf: &SharedLeaf,
    net: &dyn Frontier,
    mem: &mut SolveMem,
) -> (MoneyChance, u64, u64) {
    let cut = FrontierCut::new(pos.hand.tricks_done + 2, net);
    let sol = solve_fast_cut(
        pos,
        ctx,
        &DynLeaf(leaf),
        label_opts(HYBRID_TT_MB),
        Some(cut),
        mem,
        None,
    );
    (
        sol.value.expect("no time cap"),
        sol.stats.nodes,
        sol.stats.frontier,
    )
}

/// A position's label value as `labeller` makes it for the row with key (`a`, `b`) of a shard
/// (seed, source tag): the exact value (`exact`, the search's own when it has one, else solved
/// now with `opts`), or the hybrid's. Returns the value and the row's facts (nodes, `hybrid`,
/// `evals`).
#[allow(clippy::too_many_arguments)]
fn label_value(
    labeller: Labeller<'_>,
    key: (u64, u8, u64, u64),
    pos: &Position,
    ctx: MatchCtx,
    exact: Option<(MoneyChance, u64)>,
    leaf: &SharedLeaf,
    opts: FastOptions,
    mem: &mut SolveMem,
) -> (MoneyChance, RowInfo) {
    let (seed, source, a, b) = key;
    let trick = pos.hand.tricks_done + 1;
    match labeller {
        Labeller::Hybrid(h) if !labeller.exact_row(seed, source, trick, a, b) => {
            let (v, nodes, evals) = hybrid_value(pos, ctx, leaf, h.net, mem);
            let info = RowInfo {
                nodes,
                hybrid: true,
                evals,
                ..RowInfo::default()
            };
            (v, info)
        }
        _ => {
            let (v, nodes) = exact.unwrap_or_else(|| exact_value(pos, ctx, leaf, opts, mem));
            let info = RowInfo {
                nodes,
                ..RowInfo::default()
            };
            (v, info)
        }
    }
}

/// The label row of a position at the start of trick `pos.hand.tricks_done + 1` with value `v`:
/// the encoding of its canonical form, the value in the encoder's rotated team order, and the
/// outcome head's label, mask and vectors (the same for every source). For an exact value the
/// label is the first possible outcome with exactly that vector.
///
/// With `hybrid`, the row of a hybrid value: the same encoding, value,
/// mask and outcome vectors, and as the outcome label the first possible outcome whose money
/// vector is nearest the value ([`nn::outcome::Outcomes::nearest`]: an exact value's own outcome,
/// at distance 0, so the exact rule's label whenever one has the value). A hybrid value is in
/// general a mixture of endings (the network's at the start of the next trick), so no outcome
/// need have it exactly; the trainer reads the row unchanged (its mixture term fits the value).
///
/// # Panics
/// If no possible outcome has an exact value (the outcome vocabulary would be incomplete).
fn labelled_row_by(
    pos: &Position,
    ctx: &MatchCtx,
    v: MoneyChance,
    source: u8,
    leaf: &SharedLeaf,
    hybrid: bool,
) -> Row {
    let c = canonical(pos);
    let (features, r) = encode(&c, ctx, leaf);
    let value = v.rotate(r % 4);
    let o = outcomes(&c, ctx, leaf);
    let label = if hybrid {
        Some(o.nearest(&value))
    } else {
        o.label(&value)
    };
    let label = label.unwrap_or_else(|| {
        panic!(
            "no possible outcome has the exact value (the outcome vocabulary is incomplete): \
             {pos:?}"
        )
    });
    Row {
        features,
        value,
        source,
        trick: pos.hand.tricks_done + 1,
        label,
        mask: o.mask,
        outcomes: o.vecs,
    }
}

/// Row `i` of a job, labelled as `labeller` says (an exact solve with a table of `tt_mb`
/// megabytes, or the hybrid), and its facts.
#[allow(clippy::too_many_arguments)]
fn make_row(
    job: &LabelJob,
    i: u64,
    starts: &[Vec<HandStartState>; 4],
    source: &dyn WorldSource,
    leaf: &SharedLeaf,
    labeller: Labeller<'_>,
    tt_mb: usize,
    mem: &mut SolveMem,
) -> (Row, RowInfo) {
    let opts = label_opts(tt_mb);
    let (pos, ctx, mut info) = row_position(job, i, starts, source);
    let key = (job.seed, job.source.tag(), i, 0);
    let (v, l) = label_value(labeller, key, &pos, ctx, None, leaf, opts, mem);
    (info.nodes, info.hybrid, info.evals) = (l.nodes, l.hybrid, l.evals);
    let row = labelled_row_by(&pos, &ctx, v, job.source.tag(), leaf, l.hybrid);
    info.class = row.race_class();
    (row, info)
}

fn check_label_job(job: &LabelJob) {
    assert_ne!(
        job.seed, GOLDEN_SEED,
        "seed 1 is the golden sets' seed: labels must use another"
    );
    assert!((2..=5).contains(&job.trick), "frontier tricks are 2..=5");
    assert_ne!(
        job.source,
        Source::Harvest,
        "harvested rows come by decision: harvest_rows"
    );
}

/// Every row of a job, in row order, and each row's facts. Byte-identical for any thread count.
///
/// # Panics
/// If `job.seed` is the golden-set seed, `job.trick` is not 2..=5, or the source is
/// [`Source::Harvest`] (whose rows come by decision: [`harvest_rows`]).
pub fn label_rows(
    job: &LabelJob,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    progress: Option<&AtomicUsize>,
) -> Vec<(Row, RowInfo)> {
    label_rows_by(job, starts, leaf, Labeller::Exact, progress)
}

/// [`label_rows`] labelled as `labeller` says.
///
/// # Panics
/// As [`label_rows`].
pub fn label_rows_by(
    job: &LabelJob,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    labeller: Labeller<'_>,
    progress: Option<&AtomicUsize>,
) -> Vec<(Row, RowInfo)> {
    check_label_job(job);
    let source = BeliefSource::new();
    let mut rows = Vec::with_capacity(job.count as usize);
    ordered(
        job.threads,
        Some(job.count),
        label_window(job.trick, 1, job.threads),
        SolveMem::new,
        |mem, i| {
            let r = make_row(
                job,
                i,
                starts,
                &source,
                leaf,
                labeller,
                tt_mb_for(job.trick),
                mem,
            );
            if let Some(p) = progress {
                p.fetch_add(1, Ordering::Relaxed);
            }
            r
        },
        |_, r| {
            rows.push(r);
            Ok(true)
        },
    )
    .expect("collecting rows cannot fail");
    rows
}

/// Rows a unit of [`write_label_shard`]'s work makes by default: small enough that the last
/// units leave no long tail on hundreds of threads (32 rows are about 10 ms of work at trick 4),
/// large enough that handing them over costs nothing. Trick 2 takes 1 ([`chunk_rows_for`]).
pub const CHUNK_ROWS: u64 = 32;

/// The bytes of finished rows a label stream (a shard of any source) holds at most beyond the next
/// row it writes: a row that takes long holds the writing up while every other thread goes on
/// with the rows after it, up to this much ([`label_window`]).
pub const LABEL_WINDOW_BYTES: u64 = 1 << 30;

/// How many units of `rows_per_unit` rows at frontier trick `trick` a label stream on `threads`
/// threads makes ahead of the next unit it writes: as many as hold [`LABEL_WINDOW_BYTES`] of
/// finished rows, at least [`window_for`]`(threads)`. A fixed count of units would make the
/// look-ahead depend on the chunk size: with one-row chunks at trick 2, `4 × 368 + 16` = 1,488
/// rows are half a minute's work for 368 threads, after which one row of minutes stops every
/// thread (the trick-2 mechanism test's random labels, run 6: the CPU swung between 3% and 96%).
/// At trick 2 the budget is about 570,000 rows. The window never changes a row.
pub fn label_window(trick: u8, rows_per_unit: u64, threads: usize) -> u64 {
    let unit = row_bytes_for(nn::outcome::vocab(trick)) as u64 * rows_per_unit.max(1);
    (LABEL_WINDOW_BYTES / unit).max(window_for(threads))
}

/// The default rows a unit of [`write_label_shard`]'s work at frontier trick `k` (`search labels
/// --chunk-rows`): [`CHUNK_ROWS`], but 1 at trick 2, where one exact label takes seconds (32 of
/// them are minutes on one thread, the tail of a short shard on hundreds of threads). One-row
/// chunks are safe because the stream's look-ahead is a byte budget ([`label_window`]), not a
/// count of chunks. The shard is the same bytes for any chunk size (tested).
pub fn chunk_rows_for(k: u8) -> u64 {
    if k <= 2 { 1 } else { CHUNK_ROWS }
}

/// Chunks queued for the writer thread before the hashing waits for the disk.
const WRITE_QUEUE: usize = 64;

/// A written shard: its rows, the SHA-256 of its bytes (lowercase hex), its rows' facts, and
/// for a harvest the facts of the decisions its rows came from.
#[derive(Clone, Debug)]
pub struct ShardSummary {
    /// Rows written.
    pub rows: u64,
    /// SHA-256 of every byte written (header and records), lowercase hex.
    pub sha256: String,
    /// The rows' facts.
    pub counts: RowCounts,
    /// Harvest decisions 0, 1, 2, … whose rows the shard holds (empty for other sources).
    pub decisions: Vec<HarvestDecision>,
    /// Time spent hashing (on the thread that puts the chunks in order), for the log.
    pub hash_time: Duration,
    /// Time spent writing (on the writer thread), for the log.
    pub write_time: Duration,
    /// A harvest's time in its decisions' searches, summed over threads (the log only; zero for
    /// other sources).
    pub search_time: Duration,
    /// A harvest's time labelling the sampled positions (their exact solves when a network drove
    /// the search), summed over threads (the log only).
    pub label_time: Duration,
    /// A targeted harvest's rows and reached positions per group (`None` for other runs).
    pub groups: Option<GroupTally>,
}

/// A targeted harvest's counts per group of its predicate file ([`PredKeep`]); a position in
/// several groups counts in each.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct GroupTally {
    /// Rows of the shard in each group.
    pub rows: Vec<u64>,
    /// Rows in at least one group.
    pub rows_in_any: u64,
    /// Distinct reached positions (per decision, over the shard's decisions) in each group.
    pub reached: Vec<u64>,
    /// Distinct reached positions in at least one group.
    pub reached_in_any: u64,
    /// Distinct reached positions in all.
    pub reached_all: u64,
}

impl GroupTally {
    /// An empty tally of `g` groups.
    pub fn new(g: usize) -> GroupTally {
        GroupTally {
            rows: vec![0; g],
            reached: vec![0; g],
            ..GroupTally::default()
        }
    }

    /// Count one decision: its reached positions' tallies, and the groups of the rows kept.
    fn add(&mut self, d: &HarvestDecision, reached: &[u32], rows: &[Vec<u32>]) {
        for (a, &b) in self.reached.iter_mut().zip(reached) {
            *a += u64::from(b);
        }
        self.reached_in_any += u64::from(d.keep_distinct);
        self.reached_all += u64::from(d.distinct);
        for r in rows {
            for &g in r {
                self.rows[g as usize] += 1;
            }
            self.rows_in_any += u64::from(!r.is_empty());
        }
    }
}

/// Where a shard's bytes go, in order: into the SHA-256 here, and to the writer thread.
struct ShardSink {
    hash: Sha256,
    tx: mpsc::SyncSender<Vec<u8>>,
    bytes: u64,
    hash_time: Duration,
}

impl ShardSink {
    /// The next bytes of the shard.
    fn put(&mut self, b: Vec<u8>) -> Result<(), String> {
        let t = Instant::now();
        self.hash.update(&b);
        self.hash_time += t.elapsed();
        self.bytes += b.len() as u64;
        self.tx
            .send(b)
            .map_err(|_| "the shard's writer stopped".to_string())
    }
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// Runs `fill` with a sink for the records of a shard of `rows` rows with `v` outcomes: the
/// header, then what `fill` puts, goes in order to `out` on a writer thread (so a slow disk
/// overlaps the hashing and the making of later rows) and into a SHA-256. Returns `fill`'s
/// result, the SHA-256 (lowercase hex) of every byte written, and the time spent hashing and
/// writing; an error writing comes first.
fn stream_shard<W: Write + Send, R>(
    out: &mut W,
    rows: u64,
    v: usize,
    fill: impl FnOnce(&mut ShardSink) -> Result<R, String>,
) -> Result<(R, String, Duration, Duration), String> {
    let want = nn::shard::header(rows as usize, v).len() as u64 + rows * row_bytes_for(v) as u64;
    let (r, sha, hash_time, write_time, bytes) = stream_shard_upto(out, rows, v, fill)?;
    if bytes != want {
        return Err(format!(
            "the shard has {bytes} bytes, but {rows} rows make {want}"
        ));
    }
    Ok((r, sha, hash_time, write_time))
}

/// [`stream_shard`] whose `fill` may put fewer rows than the header's `rows` (a bounded
/// harvest, [`write_harvest_shard_upto`]); also returns the bytes written.
fn stream_shard_upto<W: Write + Send, R>(
    out: &mut W,
    rows: u64,
    v: usize,
    fill: impl FnOnce(&mut ShardSink) -> Result<R, String>,
) -> Result<(R, String, Duration, Duration, u64), String> {
    let head = nn::shard::header(rows as usize, v);
    std::thread::scope(|s| {
        let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(WRITE_QUEUE);
        let writer = s.spawn(move || -> std::io::Result<Duration> {
            let mut busy = Duration::ZERO;
            for b in rx {
                let t = Instant::now();
                out.write_all(&b)?;
                busy += t.elapsed();
            }
            let t = Instant::now();
            out.flush()?;
            Ok(busy + t.elapsed())
        });
        let mut sink = ShardSink {
            hash: Sha256::new(),
            tx,
            bytes: 0,
            hash_time: Duration::ZERO,
        };
        let r = sink.put(head).and_then(|()| fill(&mut sink));
        let ShardSink {
            hash,
            tx,
            bytes,
            hash_time,
        } = sink;
        drop(tx);
        let w = writer
            .join()
            .unwrap_or_else(|p| std::panic::resume_unwind(p));
        let write_time = w.map_err(|e| format!("writing the shard: {e}"))?;
        let r = r?;
        Ok((r, hex(&hash.finalize()), hash_time, write_time, bytes))
    })
}

/// A job's shard ([`nn::shard`]'s `.npy` bytes of [`label_rows`]) streamed to `out`: chunks of
/// `chunk_rows` consecutive rows made on `job.threads` threads, hashed and written in row order
/// as they come (module notes), so no more than a window of chunks is ever held. The same bytes,
/// hash and counts for any thread count and chunk size.
///
/// # Panics
/// As [`label_rows`].
pub fn write_label_shard<W: Write + Send>(
    job: &LabelJob,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    progress: Option<&AtomicUsize>,
    chunk_rows: u64,
    out: &mut W,
) -> Result<ShardSummary, String> {
    write_label_shard_with(
        job,
        LabelExec::for_trick(job.trick),
        starts,
        leaf,
        progress,
        chunk_rows,
        out,
    )
}

/// [`write_label_shard`] with the exact solves' table of `exec.tt_mb` megabytes: the same bytes,
/// hash and counts apart from the solves' node counts (the table never changes a value).
///
/// # Panics
/// As [`label_rows`].
pub fn write_label_shard_with<W: Write + Send>(
    job: &LabelJob,
    exec: LabelExec,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    progress: Option<&AtomicUsize>,
    chunk_rows: u64,
    out: &mut W,
) -> Result<ShardSummary, String> {
    write_label_shard_by(
        job,
        exec,
        starts,
        leaf,
        Labeller::Exact,
        progress,
        chunk_rows,
        out,
    )
}

/// [`write_label_shard_with`] labelled as `labeller` says: the bytes of [`label_rows_by`], for
/// any thread count and chunk size.
///
/// # Panics
/// As [`label_rows`].
#[allow(clippy::too_many_arguments)]
pub fn write_label_shard_by<W: Write + Send>(
    job: &LabelJob,
    exec: LabelExec,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    labeller: Labeller<'_>,
    progress: Option<&AtomicUsize>,
    chunk_rows: u64,
    out: &mut W,
) -> Result<ShardSummary, String> {
    check_label_job(job);
    let source = BeliefSource::new();
    stream_label_rows(
        job.trick,
        job.count,
        job.threads,
        chunk_rows,
        |mem, i| make_row(job, i, starts, &source, leaf, labeller, exec.tt_mb, mem),
        progress,
        out,
    )
}

/// The shard of rows `0..n` at frontier trick `trick` made by `make` (row `i` from `i` alone, with
/// a worker's solver memory), streamed to `out`: chunks of `chunk_rows` consecutive rows made on
/// `threads` threads, claimed in order up to [`label_window`] chunks ahead of the next one
/// written (a byte budget, so no chunk size lets one slow row stop the other threads), hashed and
/// written in row order as they come. The same bytes, hash and counts for any thread count and
/// chunk size.
fn stream_label_rows<W: Write + Send>(
    trick: u8,
    n: u64,
    threads: usize,
    chunk_rows: u64,
    make: impl Fn(&mut SolveMem, u64) -> (Row, RowInfo) + Sync,
    progress: Option<&AtomicUsize>,
    out: &mut W,
) -> Result<ShardSummary, String> {
    let v = nn::outcome::vocab(trick);
    let chunk = chunk_rows.max(1);
    let mut counts = RowCounts::default();
    let ((), sha256, hash_time, write_time) = stream_shard(out, n, v, |sink| {
        ordered(
            threads,
            Some(n.div_ceil(chunk)),
            label_window(trick, chunk, threads),
            SolveMem::new,
            |mem, c| {
                let (lo, hi) = (c * chunk, ((c + 1) * chunk).min(n));
                let mut bytes = Vec::with_capacity((hi - lo) as usize * row_bytes_for(v));
                let mut cnt = RowCounts::default();
                for i in lo..hi {
                    let (row, info) = make(mem, i);
                    assert_eq!(row.outcomes.len(), v, "every row has the trick's outcomes");
                    row_bytes(&row, &mut bytes);
                    cnt.add(&info);
                }
                if let Some(p) = progress {
                    p.fetch_add((hi - lo) as usize, Ordering::Relaxed);
                }
                (bytes, cnt)
            },
            |_, (bytes, cnt)| {
                counts.merge(&cnt);
                sink.put(bytes)?;
                Ok(true)
            },
        )
    })?;
    Ok(ShardSummary {
        rows: n,
        sha256,
        counts,
        decisions: Vec::new(),
        hash_time,
        write_time,
        search_time: Duration::ZERO,
        label_time: Duration::ZERO,
        groups: None,
    })
}

// ---------------------------------------------------------------------------------------------
// Harvested frontier positions (spec §5 source (c)).
// ---------------------------------------------------------------------------------------------

/// The solver table (megabytes) of the search from a harvest decision: fixed, because the
/// positions a search reaches may depend on its table's replacements, so the harvest never
/// depends on the machine (the hybrid check's default too).
pub const HARVEST_TT_MB: usize = 64;

/// The default cap of harvested positions per decision (spec §5 (c): one decision must not
/// dominate; a trick-3 decision reaches about two hundred trick-4 positions).
pub const HARVEST_CAP: u32 = 16;

/// The default shares of decisions by the decider's position band (1–2, 3–4, 5–8): uniform
/// over the eight positions, as the yardstick.
pub const HARVEST_SHARES: &str = "0.25,0.25,0.5";

/// The position bands of the yardstick (spec §3): positions 1–2, 3–4, 5–8 (0-based ranges).
pub const BANDS: [(usize, usize); 3] = [(0, 2), (2, 4), (4, 8)];

/// The loss weights of a harvest's rows by their decision's seat band (`--seat-weights
/// w12,w34,w58`; docs/m3-hybrid/audit-loop.md §10, the leader focus): three numbers, for the
/// decider's positions 1–2, 3–4 and 5–8, each a multiple of 1/256 from 1/256 to 65535/256 (the
/// per-row weights' fixed point, `nn::rowweights`). Returned in that fixed point.
///
/// # Errors
/// If it is not three such numbers.
pub fn parse_seat_weights(s: &str) -> Result<[u16; 3], String> {
    let parts: Vec<&str> = s.split(',').map(str::trim).collect();
    if parts.len() != 3 {
        return Err(format!(
            "seat weights {s:?}: three numbers, for the decider's positions 1–2, 3–4 and 5–8 \
             (e.g. 2,1,1)"
        ));
    }
    let mut out = [0u16; 3];
    for (o, p) in out.iter_mut().zip(&parts) {
        *o = p
            .parse::<f64>()
            .ok()
            .filter(|x| x.is_finite())
            .and_then(nn::rowweights::fixed)
            .ok_or_else(|| {
                format!(
                    "seat weights {s:?}: {p:?} is not a multiple of 1/256 from 1/256 to 65535/256"
                )
            })?;
    }
    Ok(out)
}

/// The per-row weights of a harvest shard by its decisions' seat bands: each decision's rows (in
/// the order written, `ShardSummary::decisions`) weigh its decider's band's weight (`weights`, in
/// the fixed point of [`parse_seat_weights`]).
pub fn seat_row_weights(decisions: &[HarvestDecision], weights: [u16; 3]) -> Vec<u16> {
    let band = |position: u8| {
        let p = usize::from(position.max(1)) - 1;
        BANDS
            .iter()
            .position(|&(lo, hi)| (lo..hi).contains(&p))
            .unwrap_or(2)
    };
    decisions
        .iter()
        .flat_map(|d| std::iter::repeat_n(weights[band(d.position)], d.rows as usize))
        .collect()
}

/// One harvest run (rows at the start of trick `trick`, from decisions at `tricks`).
#[derive(Clone, Debug)]
pub struct HarvestJob {
    /// The frontier trick whose start the positions are (2..=5).
    pub trick: u8,
    /// Rows.
    pub count: u64,
    /// Master seed (never [`GOLDEN_SEED`]).
    pub seed: u64,
    /// Worker threads (never changes a row).
    pub threads: usize,
    /// At most this many positions per decision (at least 1).
    pub cap: u32,
    /// The decision tricks (each 1..`trick`), drawn uniformly per decision.
    pub tricks: Vec<u8>,
    /// The decider position of decision slot `q` (decisions `sq .. sq + s − 1`, one per drawn
    /// stratum, `s = strata.len()`) is `schedule[q mod len]` (0-based; [`position_schedule`]).
    pub schedule: Vec<u8>,
    /// The strata (1–4, ascending, distinct) the decisions' match states are drawn from:
    /// decision `j` is from stratum `strata[j mod s]` (all four, [`HARVEST_STRATA`], is the
    /// yardstick's `j mod 4 + 1`).
    pub strata: Vec<u8>,
    /// The race harvest (`--harvest-race-classes`): positions of these race classes are kept
    /// up to their own cap, the others up to `cap` (which may then be 0). `None`: every
    /// position up to `cap`.
    pub race: Option<RaceKeep>,
    /// The targeted harvest (`--harvest-keep`): positions in any group of a predicate file are
    /// kept up to their own cap, the others up to `cap` (which may then be 0). Not with `race`.
    pub keep: Option<PredKeep>,
}

/// The default cap of a targeted harvest's in-group positions per decision.
pub const KEEP_CAP: u32 = 256;

/// A targeted harvest's selection (docs/m3-hybrid/audit-loop.md §4 step 5): the frontier
/// positions inside any group of `preds` (`nn::predicate`, decided on the position's encoded
/// features) are kept, up to `cap` per decision.
#[derive(Clone, Debug)]
pub struct PredKeep {
    /// The groups.
    pub preds: Arc<Predicates>,
    /// At most this many in-group positions per decision (at least 1).
    pub cap: u32,
}

/// Every stratum: the yardstick's decisions.
pub const HARVEST_STRATA: [u8; 4] = [1, 2, 3, 4];

/// The default cap of a race harvest's race-class positions per decision.
pub const RACE_CAP: u32 = 256;

/// The default race classes of a race harvest: the races for the game (2) and for the money
/// (4), where the network errs most (docs/m3-hybrid/rung1.md, "Why seats 3-4 fail").
pub const RACE_CLASSES: &str = "2,4";

/// A race harvest's selection: the frontier positions whose race class
/// (`nn::outcome::race_class`) is in `classes` (bit `k` for class `k`) are kept, up to `cap` per
/// decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaceKeep {
    /// Bit `k`: class `k` (0–4).
    pub classes: u8,
    /// At most this many of them per decision (at least 1).
    pub cap: u32,
}

impl RaceKeep {
    /// Is class `k` one of the race classes?
    pub fn has(self, k: u8) -> bool {
        self.classes >> k & 1 == 1
    }

    /// The classes, ascending.
    pub fn list(self) -> Vec<u8> {
        (0..N_RACE_CLASSES as u8).filter(|&k| self.has(k)).collect()
    }
}

/// Parse race classes `"2,4"`: distinct whole numbers 0–4, at least one, as a bit set.
pub fn parse_race_classes(s: &str) -> Result<u8, String> {
    let mut bits = 0u8;
    for p in s.split(',').map(str::trim) {
        let k: u8 = p
            .parse()
            .ok()
            .filter(|&k| usize::from(k) < N_RACE_CLASSES)
            .ok_or_else(|| format!("race classes {s:?}: {p:?} is not a race class 0-4"))?;
        if bits >> k & 1 == 1 {
            return Err(format!("race classes {s:?}: {k} is given twice"));
        }
        bits |= 1 << k;
    }
    Ok(bits)
}

/// Parse strata `"4"` or `"2,4"`: distinct whole numbers 1–4, at least one, ascending.
pub fn parse_strata(s: &str) -> Result<Vec<u8>, String> {
    let mut v = Vec::new();
    for p in s.split(',').map(str::trim) {
        let k: u8 = p
            .parse()
            .ok()
            .filter(|k| (1..=4).contains(k))
            .ok_or_else(|| format!("strata {s:?}: {p:?} is not a stratum 1-4"))?;
        if v.contains(&k) {
            return Err(format!("strata {s:?}: {k} is given twice"));
        }
        v.push(k);
    }
    v.sort_unstable();
    Ok(v)
}

/// Parse band shares `"s12,s34,s58"` (three numbers, 0 or more, at most 6 decimals, not all 0)
/// into per-position integer weights (each band's share split evenly over its positions),
/// reduced by their greatest common divisor. `"0.25,0.25,0.5"` gives eight equal weights.
pub fn parse_band_shares(s: &str) -> Result<[u64; 8], String> {
    let parts: Vec<&str> = s.split(',').map(str::trim).collect();
    if parts.len() != 3 {
        return Err(format!(
            "position shares {s:?}: three numbers, for positions 1–2, 3–4 and 5–8 (e.g. 0.25,0.25,0.5)"
        ));
    }
    let mut micro = [0u64; 3];
    for (m, p) in micro.iter_mut().zip(&parts) {
        let (int, frac) = p.split_once('.').unwrap_or((p, ""));
        let ok = |x: &str| x.bytes().all(|b| b.is_ascii_digit());
        if (int.is_empty() && frac.is_empty())
            || !ok(int)
            || !ok(frac)
            || frac.len() > 6
            || int.len() > 6
        {
            return Err(format!(
                "position shares {s:?}: {p:?} is not a number 0 or more with at most 6 decimals"
            ));
        }
        let i: u64 = if int.is_empty() {
            0
        } else {
            int.parse().expect("digits")
        };
        let f: u64 = format!("{frac:0<6}").parse().expect("digits");
        *m = i * 1_000_000 + f;
    }
    if micro.iter().all(|&m| m == 0) {
        return Err(format!("position shares {s:?}: all 0"));
    }
    // A band of n positions gets share/n per position: in units of 1/4 micro, 4/n each.
    let mut w = [0u64; 8];
    for (b, &(lo, hi)) in BANDS.iter().enumerate() {
        for x in &mut w[lo..hi] {
            *x = micro[b] * 4 / (hi - lo) as u64;
        }
    }
    let g = w.iter().fold(0, |g, &x| gcd(g, x));
    Ok(w.map(|x| x / g))
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The deterministic order of decider positions for integer weights: one cycle of smooth
/// weighted round-robin (each step every position gains its weight, the largest total is
/// chosen, lowest position first on ties, and loses the sum of the weights), so every prefix
/// is as close to the weights as whole decisions allow. Equal weights give 0, 1, …, 7: the
/// yardstick's order.
///
/// # Panics
/// If every weight is 0, or the cycle would be longer than 2^24 positions.
pub fn position_schedule(w: [u64; 8]) -> Vec<u8> {
    let total: u64 = w.iter().sum();
    assert!(
        (1..=1 << 24).contains(&total),
        "position weights sum to {total}"
    );
    let mut cur = [0i64; 8];
    let mut out = Vec::with_capacity(total as usize);
    for _ in 0..total {
        for (c, &x) in cur.iter_mut().zip(&w) {
            *c += x as i64;
        }
        let mut best = 0;
        for p in 1..8 {
            if cur[p] > cur[best] {
                best = p;
            }
        }
        cur[best] -= total as i64;
        out.push(best as u8);
    }
    out
}

/// The values a harvest's search uses at the frontier.
#[derive(Clone, Copy)]
pub enum Driver<'a> {
    /// The exact value of every reached position: the search H_k makes when its network is right
    /// (the exact solver's own search down to trick `k`). Every reached position is solved, so
    /// the sampled ones' labels come with the search. The default; right for trick 4, where an
    /// exact solve costs about 0.04 ms.
    Exact,
    /// A network's values (normally the previous round's integer O_k, `nn::Oracle`, with its
    /// memo): the search H_k as it really runs with that network. Only the sampled positions are
    /// solved exactly, for their labels. For trick 3 and earlier, where an exact solve of every
    /// reached position is too dear (about 9 ms each at trick 3, and a trick-2 leader decision
    /// reaches about 2,200 positions on average: docs/m3-hybrid/rung2.md).
    Net(&'a dyn Frontier),
}

/// One frontier position reached by a harvest search.
#[derive(Clone, Copy, Debug)]
pub struct Reached {
    /// The position as the search first reached it.
    pub pos: Position,
    /// The match around it.
    pub ctx: MatchCtx,
    /// Its canonical form.
    pub canon: CanonPos,
    /// The value the search used for it (absolute team order): its exact value
    /// ([`Driver::Exact`]) or the network's ([`Driver::Net`]).
    pub value: MoneyChance,
    /// Solver nodes of the exact value, on the table the decision's solves share (0 when a
    /// network drove the search: the sampled positions' own solves are counted in their rows'
    /// facts).
    pub nodes: u64,
}

struct RecState {
    mem: SolveMem,
    seen: BTreeMap<CanonPos, usize>,
    list: Vec<Reached>,
    evals: u64,
}

/// The harvest's frontier evaluator: records every position the solver reaches at the frontier
/// (first visits, in the order the sequential solver reaches them) and returns the driver's
/// value of it: the exact value ([`Driver::Exact`]; the search is then exactly the hybrid H_k
/// with a perfect network at the frontier) or the network's ([`Driver::Net`]; the search is H_k
/// with that network). A canonical form met again (another line, or the same position after a
/// table replacement) gets its recorded value. Either value depends only on the canonical form
/// and the context (`solver::frontier`), as the solvers require.
pub(crate) struct Recorder<'a> {
    leaf: &'a SharedLeaf,
    opts: FastOptions,
    driver: Driver<'a>,
    state: Mutex<RecState>,
}

impl<'a> Recorder<'a> {
    /// A recorder valuing the frontier as `driver` says (exact solves with `opts`).
    pub(crate) fn new(leaf: &'a SharedLeaf, opts: FastOptions, driver: Driver<'a>) -> Self {
        Recorder {
            leaf,
            opts,
            driver,
            state: Mutex::new(RecState {
                mem: SolveMem::new(),
                seen: BTreeMap::new(),
                list: Vec::new(),
                evals: 0,
            }),
        }
    }

    /// The distinct positions reached, in first-visit order, and the frontier evaluations.
    pub(crate) fn into_reached(self) -> (Vec<Reached>, u64) {
        let st = self.state.into_inner().unwrap_or_else(|e| e.into_inner());
        (st.list, st.evals)
    }
}

impl Frontier for Recorder<'_> {
    fn value(&self, canon: &CanonPos, _ctx: &MatchCtx) -> MoneyChance {
        let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let i = *st
            .seen
            .get(canon)
            .expect("the recorder values the positions the solver gives it (value_at)");
        st.list[i].value
    }

    fn value_at(&self, pos: &Position, canon: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let st = &mut *guard;
        st.evals += 1;
        if let Some(&i) = st.seen.get(canon) {
            return st.list[i].value;
        }
        let (value, nodes) = match self.driver {
            Driver::Exact => exact_value(pos, *ctx, self.leaf, self.opts, &mut st.mem),
            Driver::Net(net) => (net.value_at(pos, canon, ctx), 0),
        };
        st.seen.insert(*canon, st.list.len());
        st.list.push(Reached {
            pos: *pos,
            ctx: *ctx,
            canon: *canon,
            value,
            nodes,
        });
        value
    }

    fn prefetch_depth(&self) -> u8 {
        match self.driver {
            Driver::Exact => 0,
            Driver::Net(net) => net.prefetch_depth(),
        }
    }

    /// Passed on to the network; nothing is recorded (the search records what it reaches
    /// through `value_at`, as without the batch).
    fn prefetch(&self, pos: &[CanonPos], ctx: &MatchCtx) {
        if let Driver::Net(net) = self.driver {
            net.prefetch(pos, ctx);
        }
    }
}

/// One harvest decision's facts (the sidecar's counts).
#[derive(Clone, Debug, Default, Serialize)]
pub struct HarvestDecision {
    /// Decision index `j` in the run.
    pub id: u64,
    /// The decision's trick.
    pub trick: u8,
    /// The decider's position in the trick, 1 (leader) to 8.
    pub position: u8,
    /// Stratum of the match state (1–4).
    pub stratum: u8,
    /// Candidates of the decision.
    pub cands: usize,
    /// Frontier evaluations of the search (table misses at the frontier).
    pub evals: u64,
    /// Distinct canonical forms among them.
    pub distinct: u32,
    /// Rows taken from the decision (at most the cap; the last decision of a run may give fewer).
    pub rows: u32,
    /// Solver nodes of the search from the decision.
    pub search_nodes: u64,
    /// A race harvest: distinct positions of the race classes among the reached ones (0
    /// otherwise).
    pub race_distinct: u32,
    /// A targeted harvest: distinct reached positions inside any group (0 otherwise; not written
    /// then, so other runs' facts are as before).
    #[serde(skip_serializing_if = "is_zero")]
    pub keep_distinct: u32,
}

fn is_zero(x: &u32) -> bool {
    *x == 0
}

/// Everything one harvest decision gives: its facts, its rows (already labelled) in the order
/// the search first reached them, and their positions.
pub struct DecisionHarvest {
    /// The facts.
    pub info: HarvestDecision,
    /// The rows, at most the cap.
    pub rows: Vec<(Row, RowInfo)>,
    /// Each row's position and context.
    pub positions: Vec<(Position, MatchCtx)>,
    /// Every distinct reached position, in first-visit order (before sampling).
    pub reached: Vec<Reached>,
    /// A targeted harvest: each row's groups (ascending; empty for a row in none), in row order
    /// (empty for other runs).
    pub row_groups: Vec<Vec<u32>>,
    /// A targeted harvest: distinct reached positions in each group (empty for other runs).
    pub reached_groups: Vec<u32>,
    /// Time in the search (reports only).
    pub search_time: Duration,
    /// Time labelling the sample, its exact solves included when a network drove the search
    /// (reports only).
    pub label_time: Duration,
}

/// Harvest decision `j` before its search: the decision (the yardstick's generator,
/// [`crate::measure::build_decision`], at a decision trick drawn uniformly from `job.tricks`, the
/// stratum `strata[j mod s]` and the position `schedule[(j div s) mod len]`, `s` the number of
/// strata: with all four, the stratum `j mod 4 + 1` of the yardstick), its candidates, a world
/// drawn from the decider's belief rebuilt as a match, and the decision's random stream (which
/// then draws the sample). A pure function of (seed, trick, `j`) and the job's settings.
pub fn harvest_root(
    job: &HarvestJob,
    j: u64,
    starts: &[Vec<HandStartState>; 4],
    source: &dyn WorldSource,
) -> (DecisionSpec, Vec<Candidate>, Match, Rng) {
    let k = job.trick;
    let mut rng = Rng::derive(
        job.seed,
        &[NN_LABEL, u64::from(SOURCE_HARVEST), u64::from(k), j],
    );
    let ns = job.strata.len() as u64;
    let st = usize::from(job.strata[(j % ns) as usize] - 1);
    let pos = usize::from(job.schedule[((j / ns) % job.schedule.len() as u64) as usize]);
    let t = job.tricks[rng.bounded(job.tricks.len() as u64) as usize];
    let d = build_decision(t, j, st, pos, &mut rng, starts);
    let (_, view, legal) = d.replay();
    let (_, cands) = candidates(&view, &legal);
    let worlds = source.prepare(&view, Rules::HOUSE);
    let world = worlds.sample(&mut rng);
    let m = rebuild(&view, &world, Rules::HOUSE, rng.next_u64()).expect("sampled worlds rebuild");
    (d, cands, m, rng)
}

/// How many of a decision's sampled positions make one unit of label work at frontier trick `k`
/// (the harvests' pooled schedule, [`HarvestSchedule::Pooled`]): the unit's positions are solved
/// in reach order on one table and leaf memo shared by the decision's family
/// (`solver::SolveFamily`), each unit from empty tables.
///
/// - Trick 2: **1**. A trick-2 solve takes seconds (36 M nodes on average, heavy-tailed to
///   minutes), so each label is its own unit and every thread takes the next label of the pooled
///   list, whichever decision it came from: the run ends about one label's time after the last
///   is claimed, however many labels one decision has (16 to 270). Sharing a table across a
///   decision's labels saved nothing measurable at trick 2 (docs/ENGINEERING-2026-10-17.md item
///   A: 2% for every lever together at trick 2, against 8% for the random labels' levers alone;
///   the sampled positions of one decision share little of their futures).
/// - Tricks 3–5: **all of the decision's**. A solve takes milliseconds or less, so a decision's
///   labels are one short unit, and sharing the table saves a quarter to three quarters of the
///   nodes (item A): exactly the solves, and the node counts, of the per-decision schedule.
///
/// The value of a label never depends on this (the shared table holds exact values only; tested).
pub fn label_group_for(k: u8) -> usize {
    if k <= 2 { 1 } else { usize::MAX }
}

/// The options of the exact label solves with a table of `tt_mb` megabytes (by default
/// [`tt_mb_for`] the trick; the table never changes a value).
fn label_opts(tt_mb: usize) -> FastOptions {
    FastOptions {
        tt_mb: tt_mb.max(1),
        ..FastOptions::default()
    }
}

/// How a label run is carried out over its threads and memory; never changes a row, a harvest
/// decision's facts or their order (tested), only the time and the solves' node counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LabelExec {
    /// The exact label solves' table, megabytes a thread ([`tt_mb_for`] the trick unless lowered
    /// to fit the machine's memory: `search labels --tt-mb`). A harvest's own search keeps its
    /// fixed table ([`HARVEST_TT_MB`]: what it reaches may depend on it).
    pub tt_mb: usize,
    /// A harvest's schedule.
    pub schedule: HarvestSchedule,
    /// A pooled harvest's positions a unit of label work ([`label_group_for`]).
    pub group: usize,
}

impl LabelExec {
    /// The defaults at frontier trick `k`.
    pub fn for_trick(k: u8) -> LabelExec {
        LabelExec {
            tt_mb: tt_mb_for(k),
            schedule: HarvestSchedule::Pooled,
            group: label_group_for(k),
        }
    }
}

/// How a harvest's work is shared out over its threads; never changes a row, a decision's facts
/// or the order of either (tested), only the time, and the label solves' node counts at trick 2
/// ([`label_group_for`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HarvestSchedule {
    /// The default. Each decision's search and sample on one thread (decisions claimed in
    /// order, at most a window ahead); the sampled positions of the decisions taken, in order,
    /// form one list whose labels ([`label_group_for`] positions a unit) are made by every
    /// thread, a label before any further decision (`crate::ordered::pooled`). No label is made
    /// for a decision past the row count, and no thread waits on one long decision.
    #[default]
    Pooled,
    /// Before 2026-10-17: each decision's search, sample and every one of its labels on one
    /// thread, decisions claimed in order up to a window of `4 × threads + 16` ahead
    /// (`crate::ordered::ordered`), the decisions being made when the row count is reached
    /// finished and dropped. At trick 2 a decision's labels take minutes to hours on one
    /// thread, so the run waits on its slowest decisions while the others make decisions that
    /// are thrown away (measured on a 245-thread EPYC 7742: 7% of the plain labels' speed).
    /// Kept for the identity tests and timing comparisons (`search labels --harvest-schedule
    /// decision`).
    PerDecision,
}

impl HarvestSchedule {
    /// Its name on the command line (`--harvest-schedule`).
    pub fn name(self) -> &'static str {
        match self {
            HarvestSchedule::Pooled => "pooled",
            HarvestSchedule::PerDecision => "decision",
        }
    }

    /// Parse a command-line name.
    pub fn parse(s: &str) -> Result<HarvestSchedule, String> {
        match s {
            "pooled" => Ok(HarvestSchedule::Pooled),
            "decision" => Ok(HarvestSchedule::PerDecision),
            _ => Err(format!("unknown harvest schedule {s} (pooled or decision)")),
        }
    }
}

/// One sampled frontier position of a harvest decision, its label still to make.
#[derive(Clone, Copy, Debug)]
pub struct Sampled {
    /// The position as the search first reached it.
    pub pos: Position,
    /// The match around it.
    pub ctx: MatchCtx,
    /// Its exact value and the nodes of its solve when the search already solved it
    /// ([`Driver::Exact`]); `None` with a network driver (solved for the label).
    pub exact: Option<(MoneyChance, u64)>,
}

/// A harvest decision searched and sampled ([`harvest_plan`]), its labels still to make.
pub struct DecisionPlan {
    /// The decision's facts (`rows`: the positions sampled).
    pub info: HarvestDecision,
    /// The sampled positions, in reach order.
    pub sample: Vec<Sampled>,
    /// A targeted harvest: each sampled position's groups (ascending; empty for one in none),
    /// in sample order (empty for other runs).
    pub row_groups: Vec<Vec<u32>>,
    /// A targeted harvest: distinct reached positions in each group (empty for other runs).
    pub reached_groups: Vec<u32>,
    /// The family whose solves may share a table: every frontier position the decision's
    /// search reaches in its world descends from it (`None`: each solve from empty tables).
    pub family: Option<SolveFamily>,
    /// Time in the search and sample (reports only).
    pub search_time: Duration,
}

impl DecisionPlan {
    /// The facts of the rows `RowInfo` shares (stratum, decision trick, decider position).
    fn row_base(&self) -> RowInfo {
        RowInfo {
            stratum: self.info.stratum,
            root_trick: self.info.trick,
            position: self.info.position,
            ..RowInfo::default()
        }
    }
}

/// One unit of label work: consecutive sampled positions of one decision ([`label_group_for`]).
pub struct LabelItem {
    /// The positions, in reach order.
    pub sample: Vec<Sampled>,
    /// Their decision's family ([`DecisionPlan::family`]).
    pub family: Option<SolveFamily>,
    /// Their rows' shared facts.
    base: RowInfo,
    /// Their decision's index `j` and the first position's place in its sample (the rows' keys
    /// for the exact anchor, [`HybridLabels::exact_ppm`]).
    key: (u64, u64),
    /// The run's seed (the exact anchor's draw).
    seed: u64,
}

/// Harvest decision `j` ([`harvest_root`]) searched and sampled: the hybrid's search from it for
/// every candidate (`crate::pimc::exact_world` with a frontier at the start of `job.trick` and
/// the [`Recorder`] there, valuing the frontier as `driver` says), the distinct reached positions
/// (returned too, in first-visit order), and a seeded sample of at most `job.cap` of them (kept in
/// reach order; with [`Driver::Exact`] each already valued exactly by the search). `search_mem`
/// is the search's memory; with the exact driver, the reached positions are solved on
/// `label_mem` with a table of `label_tt_mb`, shared by the decision's family
/// (`solver::SolveFamily`: the same values as solves from empty tables, fewer nodes). A pure function of (seed, trick, `j`), the job's settings and
/// the driver; the memories never change a value or a count.
#[allow(clippy::too_many_arguments)]
pub fn harvest_plan(
    job: &HarvestJob,
    j: u64,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    source: &dyn WorldSource,
    driver: Driver<'_>,
    label_tt_mb: usize,
    search_mem: &mut SolveMem,
    label_mem: &mut SolveMem,
) -> (DecisionPlan, Vec<Reached>) {
    let k = job.trick;
    let t0 = Instant::now();
    let (d, cands, m, mut rng) = harvest_root(job, j, starts, source);
    let (t, st, pos) = (d.trick, usize::from(d.stratum - 1), usize::from(d.position));
    // The exact solves of the decision's frontier positions (every reached one with the exact
    // driver, the sampled ones with a network) may share one table and leaf memo: they all
    // descend from the decision in this one world (`solver::SolveFamily`: the values are those
    // of solves from empty tables; only the node counts in the rows' facts change).
    let family = SolveFamily::of_match(&m);
    let mut lm = std::mem::take(label_mem);
    match family {
        Some(f) => lm.share(f),
        None => lm.unshare(),
    }
    let rec = Recorder {
        leaf,
        opts: label_opts(label_tt_mb),
        driver,
        state: Mutex::new(RecState {
            mem: lm,
            seen: BTreeMap::new(),
            list: Vec::new(),
            evals: 0,
        }),
    };
    let ctx = Ctx {
        leaf,
        source,
        rules: Rules::HOUSE,
    };
    let cfg = SearchConfig {
        tt_mb: HARVEST_TT_MB,
        ..SearchConfig::default()
    };
    let mut run = WorldSolve::default();
    exact_world(
        &m,
        &cands,
        &ctx,
        &cfg,
        None,
        search_mem,
        &mut run,
        Some(FrontierCut::new(k, &rec)),
    )
    .expect("nothing stops a harvest search");
    let RecState {
        mem: lm,
        seen: _,
        list,
        evals,
    } = rec.state.into_inner().unwrap_or_else(|e| e.into_inner());
    *label_mem = lm;
    let n = list.len();
    // A seeded sample of `take` of `idx` (a partial Fisher–Yates shuffle on the decision's
    // stream).
    let sample = |mut idx: Vec<usize>, take: usize, rng: &mut Rng| -> Vec<usize> {
        let take = take.min(idx.len());
        for i in 0..take {
            let r = i + rng.bounded((idx.len() - i) as u64) as usize;
            idx.swap(i, r);
        }
        idx.truncate(take);
        idx
    };
    let mut race_distinct = 0;
    let mut keep_distinct = 0;
    // A targeted harvest's groups of every reached position.
    let mut members: Vec<Vec<u32>> = Vec::new();
    let mut reached_groups: Vec<u32> = Vec::new();
    let mut chosen = match (job.race, &job.keep) {
        (None, Some(keep)) => {
            // The in-group positions first, up to their cap, then the others up to `cap`.
            let mut ev = Evaluator::new(&keep.preds, leaf);
            members = list.iter().map(|r| ev.members(&r.canon, &r.ctx)).collect();
            reached_groups = vec![0; keep.preds.len()];
            for m in &members {
                for &g in m {
                    reached_groups[g as usize] += 1;
                }
            }
            let (hit, other): (Vec<usize>, Vec<usize>) =
                (0..n).partition(|&i| !members[i].is_empty());
            keep_distinct = hit.len() as u32;
            let mut c = sample(hit, keep.cap as usize, &mut rng);
            c.extend(sample(other, job.cap as usize, &mut rng));
            c
        }
        (None, None) => sample((0..n).collect(), job.cap as usize, &mut rng),
        (Some(race), _) => {
            // The race classes' positions first, up to their cap, then the others up to `cap`.
            let (hit, other): (Vec<usize>, Vec<usize>) = (0..n).partition(|&i| {
                race.has(nn::outcome::race_class_of(
                    &list[i].canon,
                    &list[i].ctx,
                    leaf,
                ))
            });
            race_distinct = hit.len() as u32;
            let mut c = sample(hit, race.cap as usize, &mut rng);
            c.extend(sample(other, job.cap as usize, &mut rng));
            c
        }
    };
    chosen.sort_unstable();
    let samples: Vec<Sampled> = chosen
        .iter()
        .map(|&i| {
            let r = &list[i];
            Sampled {
                pos: r.pos,
                ctx: r.ctx,
                exact: match driver {
                    Driver::Exact => Some((r.value, r.nodes)),
                    Driver::Net(_) => None,
                },
            }
        })
        .collect();
    let row_groups = if members.is_empty() {
        Vec::new()
    } else {
        chosen.iter().map(|&i| members[i].clone()).collect()
    };
    let plan = DecisionPlan {
        info: HarvestDecision {
            id: j,
            trick: t,
            position: pos as u8 + 1,
            stratum: st as u8 + 1,
            cands: cands.len(),
            evals,
            distinct: n as u32,
            rows: samples.len() as u32,
            search_nodes: run.stats.nodes,
            race_distinct,
            keep_distinct,
        },
        sample: samples,
        row_groups,
        reached_groups,
        family,
        search_time: t0.elapsed(),
    };
    (plan, list)
}

/// The rows of a unit of label work, in order: each position's exact value (the search's with
/// [`Driver::Exact`], else solved now on `mem` with a table of `tt_mb`, the unit's solves sharing
/// one table by its decision's family, the first from empty tables), encoded with its outcomes
/// ([`labelled_row_by`]). A pure function of the item; `mem` never changes a value or a count.
pub fn label_item(
    item: &LabelItem,
    leaf: &SharedLeaf,
    tt_mb: usize,
    mem: &mut SolveMem,
) -> Vec<(Row, RowInfo)> {
    label_item_by(item, leaf, Labeller::Exact, tt_mb, mem)
}

/// [`label_item`] labelled as `labeller` says: a hybrid row's value is [`hybrid_value`]'s (an
/// exact value the search found is not used for it), an anchor row's the exact one.
pub fn label_item_by(
    item: &LabelItem,
    leaf: &SharedLeaf,
    labeller: Labeller<'_>,
    tt_mb: usize,
    mem: &mut SolveMem,
) -> Vec<(Row, RowInfo)> {
    let opts = label_opts(tt_mb);
    match item.family {
        Some(f) => mem.share(f),
        None => mem.unshare(),
    }
    let (j, first) = item.key;
    item.sample
        .iter()
        .enumerate()
        .map(|(p, s)| {
            let key = (item.seed, SOURCE_HARVEST, j, first + p as u64);
            let (value, l) = label_value(labeller, key, &s.pos, s.ctx, s.exact, leaf, opts, mem);
            let row = labelled_row_by(&s.pos, &s.ctx, value, SOURCE_HARVEST, leaf, l.hybrid);
            let class = row.race_class();
            (
                row,
                RowInfo {
                    nodes: l.nodes,
                    class,
                    hybrid: l.hybrid,
                    evals: l.evals,
                    ..item.base
                },
            )
        })
        .collect()
}

/// The units of label work of a decision's first `keep` sampled positions, each with its
/// weight (its rows): `group` positions a unit ([`label_group_for`]).
fn label_items(plan: &DecisionPlan, keep: usize, group: usize, seed: u64) -> Vec<(LabelItem, u64)> {
    let base = plan.row_base();
    let g = group.max(1);
    plan.sample[..keep]
        .chunks(g)
        .enumerate()
        .map(|(n, c)| {
            (
                LabelItem {
                    sample: c.to_vec(),
                    family: plan.family,
                    base,
                    key: (plan.info.id, (n * g) as u64),
                    seed,
                },
                c.len() as u64,
            )
        })
        .collect()
}

/// Harvest decision `j` whole on one thread: [`harvest_plan`], then every sampled position
/// labelled in reach order on one table shared by the decision's family (fresh memory for the
/// labels, as [`HarvestSchedule::PerDecision`] makes them; the same rows as any schedule, and at
/// tricks 3–5 the same node counts). With [`Driver::Exact`] the labels' values come from the
/// search; with [`Driver::Net`] only the sampled positions are solved exactly.
pub fn harvest_decision(
    job: &HarvestJob,
    j: u64,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    source: &dyn WorldSource,
    driver: Driver<'_>,
    mem: &mut SolveMem,
) -> DecisionHarvest {
    harvest_decision_with(
        job,
        j,
        starts,
        leaf,
        source,
        driver,
        Labeller::Exact,
        tt_mb_for(job.trick),
        mem,
    )
}

/// [`harvest_decision`] with the labels' table of `tt_mb` megabytes (the same rows).
#[allow(clippy::too_many_arguments)]
fn harvest_decision_with(
    job: &HarvestJob,
    j: u64,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    source: &dyn WorldSource,
    driver: Driver<'_>,
    labeller: Labeller<'_>,
    tt_mb: usize,
    mem: &mut SolveMem,
) -> DecisionHarvest {
    let mut label_mem = SolveMem::new();
    let (plan, reached) = harvest_plan(
        job,
        j,
        starts,
        leaf,
        source,
        driver,
        tt_mb,
        mem,
        &mut label_mem,
    );
    let t1 = Instant::now();
    let item = LabelItem {
        sample: plan.sample.clone(),
        family: plan.family,
        base: plan.row_base(),
        key: (plan.info.id, 0),
        seed: job.seed,
    };
    let rows = label_item_by(&item, leaf, labeller, tt_mb, &mut label_mem);
    DecisionHarvest {
        positions: plan.sample.iter().map(|s| (s.pos, s.ctx)).collect(),
        info: plan.info,
        rows,
        reached,
        row_groups: plan.row_groups,
        reached_groups: plan.reached_groups,
        search_time: plan.search_time,
        label_time: t1.elapsed(),
    }
}

fn check_harvest_job(job: &HarvestJob) {
    assert_ne!(
        job.seed, GOLDEN_SEED,
        "seed 1 is the golden sets' seed: labels must use another"
    );
    assert!((2..=5).contains(&job.trick), "frontier tricks are 2..=5");
    assert!(
        !job.tricks.is_empty() && job.tricks.iter().all(|&t| (1..job.trick).contains(&t)),
        "harvest decisions are at tricks 1..{} (before the frontier)",
        job.trick
    );
    assert!(
        job.cap >= 1 || job.race.is_some() || job.keep.is_some(),
        "the cap per decision is at least 1 (0 only in a race or targeted harvest)"
    );
    assert!(
        job.race.is_none() || job.keep.is_none(),
        "a harvest keeps race classes or a predicate file's groups, not both"
    );
    if let Some(k) = &job.keep {
        assert!(
            k.cap >= 1 && !k.preds.is_empty(),
            "a targeted harvest keeps at least one group, at least 1 a decision"
        );
    }
    if let Some(r) = job.race {
        assert!(
            r.cap >= 1 && r.classes != 0 && usize::from(r.classes) < 1 << N_RACE_CLASSES,
            "a race harvest keeps at least one race class, at least 1 a decision"
        );
    }
    assert!(
        !job.strata.is_empty()
            && job.strata.windows(2).all(|w| w[0] < w[1])
            && job.strata.iter().all(|s| (1..=4).contains(s)),
        "the strata are 1-4, ascending and distinct"
    );
    assert!(!job.schedule.is_empty() && job.schedule.iter().all(|&p| p < 8));
}

/// A harvest's run, whatever its schedule: decisions 0, 1, 2, … in order, each giving its sampled
/// positions' rows, until `job.count` rows (the last decision cut short) or `max_decisions`
/// decisions. `decision` gets each decision taken, in order, before any of its rows: its facts
/// (`rows`: the rows it gives), its reached positions' groups and its rows' groups (a targeted
/// harvest; empty otherwise). The rows are made in batches turned into `T` by `finish` (on the
/// workers with [`HarvestSchedule::Pooled`]) and handed to `rows` in row order. Returns the time
/// in the taken decisions' searches and in their labels, summed over threads.
#[allow(clippy::too_many_arguments)]
fn run_harvest<T: Send>(
    job: &HarvestJob,
    exec: LabelExec,
    max_decisions: Option<u64>,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    driver: Driver<'_>,
    labeller: Labeller<'_>,
    progress: Option<&AtomicUsize>,
    finish: impl Fn(Vec<(Row, RowInfo)>) -> T + Sync,
    mut decision: impl FnMut(&HarvestDecision, &[u32], &[Vec<u32>]),
    mut rows: impl FnMut(T) -> Result<(), String>,
) -> Result<(Duration, Duration), String> {
    check_harvest_job(job);
    assert!(max_decisions != Some(0), "a bound of at least one decision");
    let (mut search_time, mut label_time) = (Duration::ZERO, Duration::ZERO);
    if job.count == 0 {
        return Ok((search_time, label_time));
    }
    let source = BeliefSource::new();
    let mut left = job.count;
    let mut taken = 0u64;
    // The rows a decision gives (the last one cut at the count), and whether to go on.
    let mut keep = |n: usize| -> (usize, bool) {
        let k = left.min(n as u64) as usize;
        left -= k as u64;
        taken += 1;
        (k, left > 0 && max_decisions.is_none_or(|m| taken < m))
    };
    let groups_of = |g: &[Vec<u32>], k: usize| -> Vec<Vec<u32>> {
        if g.is_empty() {
            Vec::new()
        } else {
            g[..k].to_vec()
        }
    };
    match exec.schedule {
        HarvestSchedule::Pooled => {
            crate::ordered::pooled(
                job.threads,
                window_for(job.threads),
                label_window(job.trick, 1, job.threads),
                || (SolveMem::new(), SolveMem::new()),
                |(search_mem, label_mem), j| {
                    let (plan, _) = harvest_plan(
                        job, j, starts, leaf, &source, driver, exec.tt_mb, search_mem, label_mem,
                    );
                    plan
                },
                |_, plan| {
                    let (k, go) = keep(plan.sample.len());
                    let mut info = plan.info.clone();
                    info.rows = k as u32;
                    decision(&info, &plan.reached_groups, &groups_of(&plan.row_groups, k));
                    search_time += plan.search_time;
                    Ok((label_items(&plan, k, exec.group, job.seed), go))
                },
                |(_, label_mem), item| {
                    let t = Instant::now();
                    let r = label_item_by(&item, leaf, labeller, exec.tt_mb, label_mem);
                    if let Some(p) = progress {
                        p.fetch_add(r.len(), Ordering::Relaxed);
                    }
                    (finish(r), t.elapsed())
                },
                |_, (out, t)| {
                    label_time += t;
                    rows(out)
                },
            )?;
        }
        HarvestSchedule::PerDecision => {
            ordered(
                job.threads,
                None,
                window_for(job.threads),
                SolveMem::new,
                |mem, j| {
                    let out = harvest_decision_with(
                        job, j, starts, leaf, &source, driver, labeller, exec.tt_mb, mem,
                    );
                    if let Some(p) = progress {
                        p.fetch_add(out.rows.len(), Ordering::Relaxed);
                    }
                    out
                },
                |_, mut out| {
                    let (k, go) = keep(out.rows.len());
                    out.rows.truncate(k);
                    out.info.rows = k as u32;
                    decision(
                        &out.info,
                        &out.reached_groups,
                        &groups_of(&out.row_groups, k),
                    );
                    search_time += out.search_time;
                    label_time += out.label_time;
                    rows(finish(out.rows))?;
                    Ok(go)
                },
            )?;
        }
    }
    Ok((search_time, label_time))
}

/// The rows of a harvest run and the facts of the decisions they came from: decisions 0, 1, 2, …
/// in order, each giving its rows ([`harvest_decision`]'s, the frontier valued as `driver` says),
/// until `job.count` rows (the last decision cut short). Made on `job.threads` threads as
/// [`HarvestSchedule::Pooled`] shares them out; the result is byte-identical for any thread
/// count and schedule ([`harvest_rows_with`]).
///
/// # Panics
/// If `job.seed` is the golden-set seed, `job.trick` is not 2..=5, a decision trick is not
/// 1..`job.trick`, the cap is 0 or the schedule is empty.
pub fn harvest_rows(
    job: &HarvestJob,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    driver: Driver<'_>,
    progress: Option<&AtomicUsize>,
) -> (Vec<(Row, RowInfo)>, Vec<HarvestDecision>) {
    harvest_rows_with(
        job,
        LabelExec::for_trick(job.trick),
        starts,
        leaf,
        driver,
        progress,
    )
}

/// [`harvest_rows`] carried out as `exec` says (the same rows and facts; only the label solves'
/// node counts depend on it).
///
/// # Panics
/// As [`harvest_rows`].
pub fn harvest_rows_with(
    job: &HarvestJob,
    exec: LabelExec,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    driver: Driver<'_>,
    progress: Option<&AtomicUsize>,
) -> (Vec<(Row, RowInfo)>, Vec<HarvestDecision>) {
    harvest_rows_by(job, exec, starts, leaf, driver, Labeller::Exact, progress)
}

/// [`harvest_rows_with`] labelled as `labeller` says (the same positions and decisions; only
/// the labels' values, and with them the outcome labels, differ).
///
/// # Panics
/// As [`harvest_rows`].
pub fn harvest_rows_by(
    job: &HarvestJob,
    exec: LabelExec,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    driver: Driver<'_>,
    labeller: Labeller<'_>,
    progress: Option<&AtomicUsize>,
) -> (Vec<(Row, RowInfo)>, Vec<HarvestDecision>) {
    let mut out = Vec::with_capacity(job.count as usize);
    let mut decisions = Vec::new();
    run_harvest(
        job,
        exec,
        None,
        starts,
        leaf,
        driver,
        labeller,
        progress,
        |r| r,
        |d, _, _| decisions.push(d.clone()),
        |r| {
            out.extend(r);
            Ok(())
        },
    )
    .expect("collecting rows cannot fail");
    debug_assert_eq!(out.len() as u64, job.count);
    (out, decisions)
}

/// A harvest run's shard ([`nn::shard`]'s `.npy` bytes of [`harvest_rows`]) streamed to `out`:
/// the rows hashed and written in row order as they come (module notes). The same bytes, hash,
/// counts and decisions as [`harvest_rows`] for any thread count.
///
/// # Panics
/// As [`harvest_rows`].
pub fn write_harvest_shard<W: Write + Send>(
    job: &HarvestJob,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    driver: Driver<'_>,
    progress: Option<&AtomicUsize>,
    out: &mut W,
) -> Result<ShardSummary, String> {
    write_harvest_shard_upto(job, None, starts, leaf, driver, progress, out)
}

/// The default bound on a targeted harvest's decisions (`search labels --harvest-keep` without
/// `--max-decisions`): `max(1000, ⌈50 × count / keep_cap⌉)`. A decision keeps at most `keep_cap`
/// in-group positions, and reaches about 112 distinct frontier positions at trick 4 (mean;
/// median 24: docs/m3-hybrid/rung1.md), so groups holding a fair share of what the search
/// reaches fill `count` rows within the bound; with `keep_cap` 256 the bound is about count / 5
/// decisions, three times the decisions an ordinary harvest of `count` rows plays at its cap of
/// [`HARVEST_CAP`] (count / 16). It is reached only when the groups yield under `keep_cap` / 50
/// (about 5) rows a decision, i.e. they hold under about 4.5% of the reached positions; at the
/// rare groups of a late audit round (0.1–0.5% of positions, 0.1–0.6 rows a decision) `count`
/// rows would take millions of network-driven decisions (hours).
#[must_use]
pub fn default_max_decisions(count: u64, keep_cap: u32) -> u64 {
    (50 * count).div_ceil(u64::from(keep_cap.max(1))).max(1000)
}

/// [`write_harvest_shard`], playing at most `max_decisions` decisions (`None`: no bound, exactly
/// [`write_harvest_shard`]). The decisions are 0, 1, 2, … in order whatever the thread count, so a
/// bounded shard is byte-identical for any thread count too; when the bound stops it short of
/// `job.count` rows its summary's `rows` says how many it holds, and the header written to `out`
/// still says `job.count`: [`reheader_shard`] sets it right (the file's caller does).
///
/// # Panics
/// As [`harvest_rows`], and if `max_decisions` is `Some(0)`.
pub fn write_harvest_shard_upto<W: Write + Send>(
    job: &HarvestJob,
    max_decisions: Option<u64>,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    driver: Driver<'_>,
    progress: Option<&AtomicUsize>,
    out: &mut W,
) -> Result<ShardSummary, String> {
    write_harvest_shard_with(
        job,
        LabelExec::for_trick(job.trick),
        max_decisions,
        starts,
        leaf,
        driver,
        progress,
        out,
    )
}

/// [`write_harvest_shard_upto`] carried out as `exec` says: the same bytes, hash, decisions and
/// counts for any schedule, table size and thread count, apart from the label solves' node counts
/// (the rows' `nodes`); only the time differs.
///
/// # Panics
/// As [`write_harvest_shard_upto`].
#[allow(clippy::too_many_arguments)]
pub fn write_harvest_shard_with<W: Write + Send>(
    job: &HarvestJob,
    exec: LabelExec,
    max_decisions: Option<u64>,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    driver: Driver<'_>,
    progress: Option<&AtomicUsize>,
    out: &mut W,
) -> Result<ShardSummary, String> {
    write_harvest_shard_by(
        job,
        exec,
        max_decisions,
        starts,
        leaf,
        driver,
        Labeller::Exact,
        progress,
        out,
    )
}

/// [`write_harvest_shard_with`] labelled as `labeller` says: the bytes of [`harvest_rows_by`]
/// (cut as `max_decisions` says), for any schedule, table size and thread count.
///
/// # Panics
/// As [`write_harvest_shard_upto`].
#[allow(clippy::too_many_arguments)]
pub fn write_harvest_shard_by<W: Write + Send>(
    job: &HarvestJob,
    exec: LabelExec,
    max_decisions: Option<u64>,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    driver: Driver<'_>,
    labeller: Labeller<'_>,
    progress: Option<&AtomicUsize>,
    out: &mut W,
) -> Result<ShardSummary, String> {
    check_harvest_job(job);
    assert!(max_decisions != Some(0), "a bound of at least one decision");
    let v = nn::outcome::vocab(job.trick);
    let rb = row_bytes_for(v);
    let mut counts = RowCounts::default();
    let mut decisions = Vec::new();
    let mut groups = job.keep.as_ref().map(|k| GroupTally::new(k.preds.len()));
    let mut times = (Duration::ZERO, Duration::ZERO);
    let fill = |sink: &mut ShardSink| {
        times = run_harvest(
            job,
            exec,
            max_decisions,
            starts,
            leaf,
            driver,
            labeller,
            progress,
            |rows| {
                let mut bytes = Vec::with_capacity(rows.len() * rb);
                let mut infos = Vec::with_capacity(rows.len());
                for (row, info) in &rows {
                    assert_eq!(row.outcomes.len(), v, "every row has the trick's outcomes");
                    row_bytes(row, &mut bytes);
                    infos.push(*info);
                }
                (bytes, infos)
            },
            |info, reached_groups, row_groups| {
                if let Some(t) = &mut groups {
                    t.add(info, reached_groups, row_groups);
                }
                decisions.push(info.clone());
            },
            |(bytes, infos)| {
                for i in &infos {
                    counts.add(i);
                }
                sink.put(bytes)
            },
        )?;
        Ok(())
    };
    let ((), sha256, hash_time, write_time, bytes) = stream_shard_upto(out, job.count, v, fill)?;
    let rows = counts.rows;
    let want = nn::shard::header(job.count as usize, v).len() as u64 + rows * rb as u64;
    let given: u64 = decisions.iter().map(|d| u64::from(d.rows)).sum();
    if bytes != want || given != rows || (rows < job.count && max_decisions.is_none()) {
        return Err(format!(
            "the shard has {bytes} bytes, but {rows} rows make {want} (of {} asked)",
            job.count
        ));
    }
    let (search_time, label_time) = times;
    Ok(ShardSummary {
        rows,
        sha256,
        counts,
        decisions,
        hash_time,
        write_time,
        search_time,
        label_time,
        groups,
    })
}

/// A shard file whose header says more rows than it holds (a bounded harvest,
/// [`write_harvest_shard_upto`]) set right: its header rewritten in place for `rows` rows,
/// padded with spaces to the length it had (a header for fewer rows is never longer; NumPy and
/// [`nn::shard::parse_header`] skip the spaces, and it stays a multiple of 64 bytes), and the
/// file's SHA-256 computed again from its start. Returns that SHA-256 (lowercase hex).
///
/// # Errors
/// If the file is not a label shard, its records are not `rows` whole records, or reading or
/// writing fails.
pub fn reheader_shard<F: std::io::Read + Write + std::io::Seek>(
    f: &mut F,
    rows: u64,
) -> Result<String, String> {
    use std::io::{Read, SeekFrom};
    let io = |e: std::io::Error| format!("rewriting the shard's header: {e}");
    f.seek(SeekFrom::Start(0)).map_err(io)?;
    let mut head = Vec::new();
    Read::take(&mut *f, 1 << 16)
        .read_to_end(&mut head)
        .map_err(io)?;
    let (_, v, start) = nn::shard::parse_header(&head)?;
    let mut new = nn::shard::header(rows as usize, v);
    if new.len() > start || head[6..8] != [1, 0] {
        return Err(format!(
            "a header for {rows} rows ({} bytes) does not fit the shard's ({start} bytes)",
            new.len()
        ));
    }
    let pad = start - new.len();
    let nl = new.pop().expect("the header ends with a newline");
    new.extend(std::iter::repeat_n(b' ', pad));
    new.push(nl);
    let hlen = u16::try_from(start - 10).map_err(|_| "header too long")?;
    new[8..10].copy_from_slice(&hlen.to_le_bytes());
    let size = f.seek(SeekFrom::End(0)).map_err(io)?;
    let want = start as u64 + rows * row_bytes_for(v) as u64;
    if size != want {
        return Err(format!(
            "the shard has {size} bytes, but {rows} rows make {want}"
        ));
    }
    f.seek(SeekFrom::Start(0)).map_err(io)?;
    f.write_all(&new).map_err(io)?;
    f.flush().map_err(io)?;
    f.seek(SeekFrom::Start(0)).map_err(io)?;
    let mut hash = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(io)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(hex(&hash.finalize()))
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
            let ra: Vec<Row> = a.iter().map(|x| x.0.clone()).collect();
            let rb: Vec<Row> = b.iter().map(|x| x.0.clone()).collect();
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
                // The label's outcome is possible and carries exactly the target.
                assert_eq!(row.mask >> row.label & 1, 1);
                assert_eq!(row.outcomes[usize::from(row.label)], row.value);
                assert_eq!(row.outcomes.len(), nn::outcome::vocab(trick));
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

    fn harvest_job(trick: u8, tricks: Vec<u8>, weights: [u64; 8], cap: u32) -> HarvestJob {
        HarvestJob {
            trick,
            count: 0,
            seed: 7,
            threads: 1,
            cap,
            tricks,
            schedule: position_schedule(weights),
            strata: HARVEST_STRATA.to_vec(),
            race: None,
            keep: None,
        }
    }

    /// Every frontier position (start of trick `k`) any line from `p` reaches, by canonical form.
    fn reachable(p: &Position, k: u8, out: &mut std::collections::BTreeSet<CanonPos>) {
        if p.n_played == 0 && p.hand.tricks_done + 1 == k && p.rob.is_none() {
            out.insert(canonical(p));
            return;
        }
        for c in engine::cards::cards(p.legal()) {
            if let solver::Step::Next(q) = p.play(c) {
                reachable(&q, k, out);
            }
        }
    }

    /// Harvested positions are start-of-trick-k positions the decision's search really reaches
    /// (each in the set of every frontier position reachable from the decision's world),
    /// deduplicated by canonical form; the rows are a sample of at most the cap, in reach order,
    /// each labelled with the exact solver's value of its position (rotated as encoded), and the
    /// same on a second run.
    #[test]
    fn harvested_positions_are_reached_deduplicated_and_exact() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        let source = BeliefSource::new();
        let mut mem = SolveMem::new();
        let (mut decisions, mut repeats, mut capped, mut rows) = (0, 0, 0, 0);
        let every = [1; 8];
        let last = [0, 0, 0, 0, 0, 0, 0, 1];
        let cases: [(u8, Vec<u8>, [u64; 8], u64); 4] = [
            (4, vec![3], every, 24),
            (5, vec![4], every, 8),
            (4, vec![2], last, 2),
            (3, vec![2], last, 2),
        ];
        for (k, tricks, w, n) in cases {
            let job = harvest_job(k, tricks.clone(), w, 6);
            for j in 0..n {
                let out = harvest_decision(&job, j, &starts, &*l, &source, Driver::Exact, &mut mem);
                let (d, _, m, _) = harvest_root(&job, j, &starts, &source);
                assert!(tricks.contains(&d.trick));
                assert_eq!(d.stratum, (j % 4) as u8 + 1);
                assert_eq!(
                    usize::from(d.position),
                    usize::from(job.schedule[(j / 4) as usize % job.schedule.len()])
                );
                assert_eq!(
                    (out.info.trick, out.info.position, out.info.stratum),
                    (d.trick, d.position + 1, d.stratum)
                );
                let (root, _, _) = from_match(&m).expect("a play is awaited");
                let mut all = std::collections::BTreeSet::new();
                reachable(&root, k, &mut all);
                let mut seen = std::collections::BTreeSet::new();
                for r in &out.reached {
                    let p = &r.pos;
                    assert_eq!(
                        (p.n_played, p.hand.tricks_done + 1),
                        (0, k),
                        "start of trick {k}"
                    );
                    assert!(p.rob.is_none());
                    assert!(p.hands.iter().all(|h| h.count_ones() == 6 - u32::from(k)));
                    assert_eq!(canonical(p), r.canon);
                    assert!(all.contains(&r.canon), "reached positions are reachable");
                    assert!(seen.insert(r.canon), "one position per canonical form");
                }
                assert_eq!(out.info.distinct as usize, out.reached.len());
                assert!(out.info.evals >= u64::from(out.info.distinct));
                repeats += out.info.evals - u64::from(out.info.distinct);
                let take = out.reached.len().min(6);
                assert_eq!(out.rows.len(), take);
                capped += usize::from(out.reached.len() > take);
                // The sample keeps reach order.
                let at: Vec<usize> = out
                    .positions
                    .iter()
                    .map(|(p, _)| {
                        out.reached
                            .iter()
                            .position(|r| r.canon == canonical(p))
                            .expect("a sampled position was reached")
                    })
                    .collect();
                assert!(at.windows(2).all(|w| w[0] < w[1]));
                for ((row, info), (p, ctx)) in out.rows.iter().zip(&out.positions) {
                    let exact = solve_fast(p, *ctx, &SyntheticLeaf, FastOptions::default())
                        .value
                        .expect("solved");
                    let c = canonical(p);
                    let r = nn::encode::rotation(&c);
                    assert_eq!(
                        row.value,
                        exact.rotate(r % 4),
                        "the label is the exact value"
                    );
                    assert_eq!((row.source, row.trick), (SOURCE_HARVEST, k));
                    assert_eq!(
                        decode(&row.features).expect("decodes"),
                        rotate_canon(&c, ctx, r)
                    );
                    assert_eq!(row.outcomes[usize::from(row.label)], row.value);
                    assert_eq!((info.root_trick, info.position), (d.trick, d.position + 1));
                }
                let again = harvest_decision(
                    &job,
                    j,
                    &starts,
                    &*l,
                    &source,
                    Driver::Exact,
                    &mut SolveMem::new(),
                );
                assert_eq!(
                    again.rows.iter().map(|r| &r.0).collect::<Vec<_>>(),
                    out.rows.iter().map(|r| &r.0).collect::<Vec<_>>()
                );
                decisions += 1;
                rows += out.rows.len();
            }
        }
        assert_eq!(decisions, 36);
        assert!(
            repeats > 0,
            "some positions are met twice: the dedupe is exercised"
        );
        assert!(
            capped > 0 && rows > 100,
            "{capped} capped decisions, {rows} rows"
        );
    }

    /// With exact values at the frontier the hybrid's search is the exact search: the recorder
    /// gives every candidate its exact value (the hook passes the reached positions through).
    #[test]
    fn the_harvest_search_is_the_exact_search() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        let source = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &source,
            rules: Rules::HOUSE,
        };
        let cfg = SearchConfig::default();
        let mut mem = SolveMem::new();
        let (t4, t5) = (
            harvest_job(4, vec![3], [1; 8], 4),
            harvest_job(5, vec![4], [1; 8], 4),
        );
        for (job, j) in (0..16).map(|j| (&t4, j)).chain((0..8).map(|j| (&t5, j))) {
            let k = job.trick;
            let (_, cands, m, _) = harvest_root(job, j, &starts, &source);
            let exact = exact_world(
                &m,
                &cands,
                &ctx,
                &cfg,
                None,
                &mut mem,
                &mut WorldSolve::default(),
                None,
            )
            .expect("solved");
            let rec = Recorder {
                leaf: &*l,
                opts: FastOptions::default(),
                driver: Driver::Exact,
                state: Mutex::new(RecState {
                    mem: SolveMem::new(),
                    seen: BTreeMap::new(),
                    list: Vec::new(),
                    evals: 0,
                }),
            };
            let hybrid = exact_world(
                &m,
                &cands,
                &ctx,
                &cfg,
                None,
                &mut mem,
                &mut WorldSolve::default(),
                Some(FrontierCut::new(k, &rec)),
            )
            .expect("solved");
            assert_eq!(hybrid, exact, "trick {k} decision {j}");
            let st = rec.state.into_inner().expect("unpoisoned");
            assert!(st.evals >= st.list.len() as u64);
            // Asked again by canonical form, the recorder gives the recorded value.
            for r in &st.list {
                let rec2 = Recorder {
                    leaf: &*l,
                    opts: FastOptions::default(),
                    driver: Driver::Exact,
                    state: Mutex::new(RecState {
                        mem: SolveMem::new(),
                        seen: BTreeMap::from([(r.canon, 0)]),
                        list: vec![*r],
                        evals: 0,
                    }),
                };
                assert_eq!(rec2.value(&r.canon, &r.ctx), r.value);
            }
        }
    }

    /// A harvest run is decisions 0, 1, 2, … in order, cut at the row count, byte-identical on 1
    /// and 4 threads; the facts count every row once.
    #[test]
    fn harvest_rows_are_thread_independent() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        let job = |threads| HarvestJob {
            count: 77,
            threads,
            ..harvest_job(
                4,
                vec![3],
                parse_band_shares("0.4,0.4,0.2").expect("shares"),
                5,
            )
        };
        let (a, da) = harvest_rows(&job(1), &starts, &*l, Driver::Exact, None);
        let (b, db) = harvest_rows(&job(4), &starts, &*l, Driver::Exact, None);
        assert_eq!(a.len(), 77);
        let ra: Vec<Row> = a.iter().map(|x| x.0.clone()).collect();
        let rb: Vec<Row> = b.iter().map(|x| x.0.clone()).collect();
        assert_eq!(
            nn::shard::to_npy(&ra, nn::outcome::vocab(4)),
            nn::shard::to_npy(&rb, nn::outcome::vocab(4))
        );
        let facts = |d: &[HarvestDecision]| serde_json::to_string(d).expect("json");
        assert_eq!(facts(&da), facts(&db));
        assert_eq!(da.iter().map(|d| u64::from(d.rows)).sum::<u64>(), 77);
        assert!(da.iter().enumerate().all(|(j, d)| d.id == j as u64));
        // The rows are the decisions' own rows, in order.
        let src = BeliefSource::new();
        let mut mem = SolveMem::new();
        let mut want = Vec::new();
        for d in &da {
            let out = harvest_decision(&job(1), d.id, &starts, &*l, &src, Driver::Exact, &mut mem);
            want.extend(out.rows.into_iter().take(d.rows as usize).map(|r| r.0));
        }
        assert_eq!(want, ra);
        let empty = HarvestJob { count: 0, ..job(2) };
        assert!(
            harvest_rows(&empty, &starts, &*l, Driver::Exact, None)
                .0
                .is_empty()
        );
    }

    fn counts_of(rows: &[(Row, RowInfo)]) -> RowCounts {
        let mut c = RowCounts::default();
        for (_, i) in rows {
            c.add(i);
        }
        c
    }

    /// The streamed shard is exactly `to_npy` of the rows, whatever the thread count and chunk
    /// size (dividing the row count or not, larger than it, one row), with the SHA-256 of its
    /// bytes and the rows' counts; written to a file, the file hashes to the summary's SHA-256.
    #[test]
    fn streamed_shards_are_the_rows_bytes() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        for (trick, source, count) in [(4, Source::Random, 23), (5, Source::Rules, 16)] {
            let job = |threads| LabelJob {
                trick,
                count,
                seed: 11,
                source,
                threads,
            };
            let rows = label_rows(&job(1), &starts, &*l, None);
            let want = nn::shard::to_npy(
                &rows.iter().map(|r| r.0.clone()).collect::<Vec<_>>(),
                nn::outcome::vocab(trick),
            );
            let sha = equity::sha256::sha256_hex(&want);
            for threads in [1, 2, 4] {
                for chunk in [1, 4, 8, 16, 23, 50] {
                    let mut out = Vec::new();
                    let done = AtomicUsize::new(0);
                    let sum = write_label_shard(
                        &job(threads),
                        &starts,
                        &*l,
                        Some(&done),
                        chunk,
                        &mut out,
                    )
                    .expect("written");
                    assert!(
                        out == want,
                        "{source:?} {threads} threads, chunks of {chunk}"
                    );
                    assert_eq!(sum.sha256, sha);
                    assert_eq!(sum.rows, count);
                    assert_eq!(sum.counts, counts_of(&rows));
                    assert!(sum.decisions.is_empty());
                    assert_eq!(done.load(Ordering::Relaxed) as u64, count);
                }
            }
            let path = std::env::temp_dir()
                .join(format!("oracle-labels-{}-{trick}.npy", std::process::id()));
            let mut f = std::fs::File::create(&path).expect("create");
            let sum = write_label_shard(&job(3), &starts, &*l, None, 5, &mut f).expect("written");
            drop(f);
            let bytes = std::fs::read(&path).expect("read");
            let _ = std::fs::remove_file(&path);
            assert!(bytes == want);
            assert_eq!(sum.sha256, equity::sha256::sha256_hex(&bytes));
        }
        // No rows: the header alone.
        let empty = LabelJob {
            trick: 4,
            count: 0,
            seed: 11,
            source: Source::Random,
            threads: 4,
        };
        let mut out = Vec::new();
        let sum = write_label_shard(&empty, &starts, &*l, None, 8, &mut out).expect("written");
        assert_eq!(out, nn::shard::to_npy(&[], nn::outcome::vocab(4)));
        assert_eq!((sum.rows, sum.counts), (0, RowCounts::default()));
    }

    /// A writer that fails after some bytes.
    struct Failing(usize);

    impl Write for Failing {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            if self.0 < b.len() {
                return Err(std::io::Error::other("disk full"));
            }
            self.0 -= b.len();
            Ok(b.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// A failing disk ends the run with its error (no hang, whatever the thread count).
    #[test]
    fn a_write_error_stops_the_run() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        for threads in [1, 4] {
            let job = LabelJob {
                trick: 5,
                count: 40,
                seed: 11,
                source: Source::Rules,
                threads,
            };
            let e = write_label_shard(&job, &starts, &*l, None, 2, &mut Failing(5000))
                .expect_err("the disk fails");
            assert!(e.contains("disk full"), "{e}");
            let h = HarvestJob {
                count: 30,
                threads,
                ..harvest_job(5, vec![4], [1; 8], 4)
            };
            let e = write_harvest_shard(&h, &starts, &*l, Driver::Exact, None, &mut Failing(3000))
                .expect_err("the disk fails");
            assert!(e.contains("disk full"), "{e}");
        }
    }

    /// The streamed harvest shard is exactly `to_npy` of [`harvest_rows`], with the same
    /// decisions' facts, on 1 and 4 threads.
    #[test]
    fn streamed_harvest_shards_are_the_rows_bytes() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        let job = |threads, count| HarvestJob {
            count,
            threads,
            ..harvest_job(4, vec![3], [1; 8], 5)
        };
        let (rows, ds) = harvest_rows(&job(1, 41), &starts, &*l, Driver::Exact, None);
        let want = nn::shard::to_npy(
            &rows.iter().map(|r| r.0.clone()).collect::<Vec<_>>(),
            nn::outcome::vocab(4),
        );
        let facts = |d: &[HarvestDecision]| serde_json::to_string(d).expect("json");
        for threads in [1, 4] {
            let mut out = Vec::new();
            let sum = write_harvest_shard(
                &job(threads, 41),
                &starts,
                &*l,
                Driver::Exact,
                None,
                &mut out,
            )
            .expect("written");
            assert!(out == want, "{threads} threads");
            assert_eq!(sum.sha256, equity::sha256::sha256_hex(&want));
            assert_eq!(facts(&sum.decisions), facts(&ds));
            assert_eq!(sum.counts, counts_of(&rows));
            assert_eq!(sum.rows, 41);
        }
        let mut out = Vec::new();
        let sum = write_harvest_shard(&job(4, 0), &starts, &*l, Driver::Exact, None, &mut out)
            .expect("written");
        assert_eq!(out, nn::shard::to_npy(&[], nn::outcome::vocab(4)));
        assert!(sum.decisions.is_empty());
    }

    /// A harvest bounded at `m` decisions (`write_harvest_shard_upto`): when the bound stops it
    /// short, the shard is the rows of decisions 0..m (the unbounded run's first rows), on any
    /// thread count, its summary says how many, and once [`reheader_shard`] sets its header right
    /// it reads back as those rows and hashes to what it returns; a bound not reached is exactly
    /// the unbounded shard. A header that a smaller row count shortens is padded to its length.
    #[test]
    fn bounded_harvest_is_the_first_decisions_rows() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        let job = |threads| HarvestJob {
            count: 41,
            threads,
            ..harvest_job(4, vec![3], [1; 8], 5)
        };
        let (rows, ds) = harvest_rows(&job(1), &starts, &*l, Driver::Exact, None);
        let full = nn::shard::to_npy(
            &rows.iter().map(|r| r.0.clone()).collect::<Vec<_>>(),
            nn::outcome::vocab(4),
        );
        assert!(ds.len() >= 3, "{} decisions", ds.len());
        let m = ds.len() as u64 - 2;
        let r: usize = ds[..m as usize].iter().map(|d| d.rows as usize).sum();
        assert!(r < 41);
        let want: Vec<Row> = rows[..r].iter().map(|x| x.0.clone()).collect();
        let path = std::env::temp_dir().join(format!("oracle-bounded-{}.npy", std::process::id()));
        let mut shas = Vec::new();
        for threads in [1, 4] {
            let mut f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)
                .expect("open");
            let sum = write_harvest_shard_upto(
                &job(threads),
                Some(m),
                &starts,
                &*l,
                Driver::Exact,
                None,
                &mut f,
            )
            .expect("written");
            assert_eq!((sum.rows, sum.decisions.len() as u64), (r as u64, m));
            let sha = reheader_shard(&mut f, sum.rows).expect("reheadered");
            drop(f);
            let bytes = std::fs::read(&path).expect("read");
            assert_eq!(sha, equity::sha256::sha256_hex(&bytes));
            assert!(nn::shard::from_npy(&bytes).expect("a shard") == want);
            shas.push(sha);
            // The bound not reached: the unbounded shard, byte for byte.
            let mut out = Vec::new();
            let sum = write_harvest_shard_upto(
                &job(threads),
                Some(ds.len() as u64),
                &starts,
                &*l,
                Driver::Exact,
                None,
                &mut out,
            )
            .expect("written");
            assert!(out == full);
            assert_eq!(sum.rows, 41);
        }
        assert_eq!(shas[0], shas[1], "the same bytes on 1 and 4 threads");
        let _ = std::fs::remove_file(&path);
        // A header for many more rows than written, which the real count shortens.
        let v = nn::outcome::vocab(4);
        let short = nn::shard::header(r, v).len();
        let count = (1..19)
            .map(|k| 10usize.pow(k))
            .find(|&c| nn::shard::header(c, v).len() > short)
            .expect("a longer header");
        let body = &nn::shard::to_npy(&want, v)[short..];
        let mut c = std::io::Cursor::new([nn::shard::header(count, v), body.to_vec()].concat());
        let sha = reheader_shard(&mut c, r as u64).expect("padded");
        let bytes = c.into_inner();
        assert_eq!(bytes.len(), nn::shard::header(count, v).len() + body.len());
        assert_eq!(bytes.len() % 64, body.len() % 64);
        assert_eq!(sha, equity::sha256::sha256_hex(&bytes));
        assert!(nn::shard::from_npy(&bytes).expect("a shard") == want);
        // The records must be whole.
        let mut c = std::io::Cursor::new(nn::shard::to_npy(&want, v));
        assert!(reheader_shard(&mut c, r as u64 + 1).is_err());
        assert_eq!(default_max_decisions(300_000, 256), 58_594);
        assert_eq!(default_max_decisions(1_000, 256), 1_000);
    }

    #[test]
    #[should_panic(expected = "golden")]
    fn harvest_golden_seed_is_refused() {
        let starts = collect_starts(100, 9);
        let l = leaf();
        let job = HarvestJob {
            seed: GOLDEN_SEED,
            count: 1,
            ..harvest_job(4, vec![3], [1; 8], 4)
        };
        harvest_rows(&job, &starts, &*l, Driver::Exact, None);
    }

    #[test]
    fn band_shares_make_the_position_schedule() {
        assert_eq!(parse_band_shares("0.25,0.25,0.5"), Ok([1; 8]));
        assert_eq!(position_schedule([1; 8]), (0..8).collect::<Vec<u8>>());
        let w = parse_band_shares("0.4, 0.4, 0.2").expect("shares");
        assert_eq!(w, [4, 4, 4, 4, 1, 1, 1, 1]);
        let s = position_schedule(w);
        assert_eq!(s.len(), 20);
        for p in 0..8u8 {
            assert_eq!(
                s.iter().filter(|&&x| x == p).count() as u64,
                w[usize::from(p)]
            );
        }
        // Spread out: positions 1–4 take 4 of every 5 slots in each half cycle.
        assert_eq!(s[..10].iter().filter(|&&x| x < 4).count(), 8);
        assert_eq!(parse_band_shares("1,0,0"), Ok([1, 1, 0, 0, 0, 0, 0, 0]));
        assert_eq!(parse_band_shares("0,0,3"), Ok([0, 0, 0, 0, 1, 1, 1, 1]));
        assert_eq!(parse_band_shares("2,1,1"), Ok([4, 4, 2, 2, 1, 1, 1, 1]));
        for bad in [
            "0,0,0",
            "0.1,0.2",
            "-1,1,1",
            "0.1234567,1,1",
            "a,b,c",
            "1,,1",
            ".,1,1",
            "1,2,3,4",
        ] {
            assert!(parse_band_shares(bad).is_err(), "{bad}");
        }
    }

    /// `--seat-weights`: three multiples of 1/256 in the per-row fixed point; each decision's rows
    /// weigh its decider's band's weight, in the order written.
    #[test]
    fn seat_weights() {
        assert_eq!(parse_seat_weights("2,1,1"), Ok([512, 256, 256]));
        assert_eq!(parse_seat_weights(" 1.5, 0.5 ,255"), Ok([384, 128, 65280]));
        for bad in [
            "2,1",
            "2,1,1,1",
            "0,1,1",
            "1.001,1,1",
            "256,1,1",
            "-1,1,1",
            "x,1,1",
            "inf,1,1",
        ] {
            assert!(parse_seat_weights(bad).is_err(), "{bad}");
        }
        let d = |position: u8, rows: u32| HarvestDecision {
            position,
            rows,
            ..HarvestDecision::default()
        };
        let w = seat_row_weights(
            &[d(1, 2), d(4, 1), d(8, 0), d(5, 2), d(2, 1)],
            [512, 384, 256],
        );
        assert_eq!(w, vec![512, 512, 384, 256, 256, 512]);
        assert!(seat_row_weights(&[], [512, 256, 256]).is_empty());
    }

    /// A random integer network (outcome head) for frontier trick `k`, valued with the synthetic
    /// leaf: a stand-in for the previous round's O_k (tests only).
    /// A race harvest: decisions from the given strata only (decision `j` from stratum
    /// `strata[j mod s]` at the scheduled position `(j div s) mod len`); per decision every
    /// position of the race classes kept up to the race cap and the others up to the cap (0:
    /// none), in reach order; each row's class is its position's race class
    /// (`nn::outcome::race_class_of`) and the row's own (`Row::race_class`); the same rows and
    /// facts on 1 and 4 threads; and with a network driving the search (trick 3) the rows are
    /// the reached race positions, labelled exactly.
    #[test]
    fn race_harvest_keeps_the_race_classes() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        let source = BeliefSource::new();
        let mut mem = SolveMem::new();
        let race = RaceKeep {
            classes: parse_race_classes("2,4").expect("classes"),
            cap: 5,
        };
        assert_eq!(race.list(), [2, 4]);
        let base = harvest_job(4, vec![3], [1; 8], 2);
        for (strata, other) in [(vec![4], 0u32), (vec![2, 4], 2)] {
            let job = HarvestJob {
                strata: strata.clone(),
                race: Some(race),
                cap: other,
                ..base.clone()
            };
            check_harvest_job(&job);
            let ns = strata.len() as u64;
            let mut race_rows = 0;
            for j in 0..24u64 {
                let (d, _, _, _) = harvest_root(&job, j, &starts, &source);
                assert_eq!(d.stratum, strata[(j % ns) as usize]);
                assert_eq!(
                    usize::from(d.position),
                    usize::from(job.schedule[((j / ns) % job.schedule.len() as u64) as usize])
                );
                let out = harvest_decision(&job, j, &starts, &*l, &source, Driver::Exact, &mut mem);
                let classes: Vec<u8> = out
                    .reached
                    .iter()
                    .map(|r| nn::outcome::race_class_of(&r.canon, &r.ctx, &*l))
                    .collect();
                let hit = classes.iter().filter(|&&k| race.has(k)).count();
                assert_eq!(out.info.race_distinct as usize, hit);
                let kept_hit = out.rows.iter().filter(|(_, i)| race.has(i.class)).count();
                assert_eq!(kept_hit, hit.min(5), "decision {j}");
                assert_eq!(
                    out.rows.len() - kept_hit,
                    (classes.len() - hit).min(other as usize)
                );
                // In reach order: the rows' positions are a subsequence of the reached ones.
                let at: Vec<usize> = out
                    .positions
                    .iter()
                    .map(|(p, _)| {
                        out.reached
                            .iter()
                            .position(|r| r.pos == *p)
                            .expect("reached")
                    })
                    .collect();
                assert!(at.windows(2).all(|w| w[0] < w[1]));
                for ((row, info), (pos, ctx)) in out.rows.iter().zip(&out.positions) {
                    let k = nn::outcome::race_class_of(&canonical(pos), ctx, &*l);
                    assert_eq!((info.class, row.race_class()), (k, k));
                    assert_eq!(info.stratum, d.stratum);
                }
                race_rows += kept_hit;
            }
            assert!(race_rows > 0, "strata {strata:?} give race positions");
            let j1 = HarvestJob {
                count: 60,
                threads: 1,
                ..job.clone()
            };
            let (a, da) = harvest_rows(&j1, &starts, &*l, Driver::Exact, None);
            let (b, db) = harvest_rows(
                &HarvestJob { threads: 4, ..j1 },
                &starts,
                &*l,
                Driver::Exact,
                None,
            );
            let bytes = |r: &[(Row, RowInfo)]| {
                let rows: Vec<Row> = r.iter().map(|x| x.0.clone()).collect();
                nn::shard::to_npy(&rows, nn::outcome::vocab(4))
            };
            assert_eq!(bytes(&a), bytes(&b));
            let facts = |d: &[HarvestDecision]| serde_json::to_string(d).expect("json");
            assert_eq!(facts(&da), facts(&db));
            assert_eq!(counts_of(&a), counts_of(&b));
            let c = counts_of(&a);
            assert_eq!(c.by_class.iter().sum::<u64>(), 60);
            if other == 0 {
                assert_eq!(c.by_class[2] + c.by_class[4], 60, "{c:?}");
            }
        }
        // Driven by a network at trick 3: the rows are reached race positions, labelled exactly.
        let net = random_oracle(3, 29);
        let job = HarvestJob {
            strata: vec![4],
            race: Some(RaceKeep {
                classes: 0b10100,
                cap: 2,
            }),
            cap: 0,
            ..harvest_job(3, vec![2], [0, 0, 0, 0, 0, 0, 0, 1], 0)
        };
        let mut rows = 0;
        for j in 0..3 {
            let out = harvest_decision(&job, j, &starts, &*l, &source, Driver::Net(&net), &mut mem);
            for ((row, info), (pos, ctx)) in out.rows.iter().zip(&out.positions) {
                assert!(race.has(info.class));
                let v = solve_fast(pos, *ctx, &SyntheticLeaf, FastOptions::default())
                    .value
                    .expect("solved");
                assert_eq!(
                    row.value,
                    v.rotate(nn::encode::rotation(&canonical(pos)) % 4)
                );
                rows += 1;
            }
        }
        assert!(rows > 0);
    }

    /// A predicate file for the targeted-harvest tests: the next team (after the leader's) without
    /// a trick this hand (binary), and the leader's team's chance if the hand ended now above a
    /// threshold (dense; the median of the parity fixture's rows, `nn::predicate`).
    fn test_predicates() -> Arc<Predicates> {
        let text = format!(
            r#"[{{"name": "next team no trick", "conditions": [{{"feature": {}, "op": "==", "value": 1}}]}},
                {{"name": "leader team ahead", "conditions": [{{"feature": {}, "op": ">", "value": {}}}]}}]"#,
            nn::encode::OFF_TRICKS + 5,
            nn::encode::N_BINARY + nn::encode::DENSE_NOW,
            -983_597
        );
        Arc::new(Predicates::parse(text.as_bytes()).expect("parses"))
    }

    /// The targeted harvest (`--harvest-keep`): every reached position inside a group is kept up
    /// to the keep cap, sampled first on the decision's stream, the others up to `cap`; the rows
    /// keep reach order, each row's groups are its features' (as a shard reader finds them), the
    /// per-group counts are the reached positions', and the shard, its counts and its group
    /// tallies are byte-identical on 1 and 4 threads. Driven by a network at trick 3 too.
    #[test]
    fn targeted_harvest_keeps_the_groups() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        let source = BeliefSource::new();
        let mut mem = SolveMem::new();
        let preds = test_predicates();
        let keep = PredKeep {
            preds: preds.clone(),
            cap: 5,
        };
        let base = harvest_job(4, vec![3], [1; 8], 2);
        let mut kept_any = 0;
        for other in [0u32, 2] {
            let job = HarvestJob {
                keep: Some(keep.clone()),
                cap: other,
                ..base.clone()
            };
            check_harvest_job(&job);
            for j in 0..16u64 {
                let out = harvest_decision(&job, j, &starts, &*l, &source, Driver::Exact, &mut mem);
                let members: Vec<Vec<u32>> = out
                    .reached
                    .iter()
                    .map(|r| preds.members_of(&r.canon, &r.ctx, &*l))
                    .collect();
                let hit = members.iter().filter(|m| !m.is_empty()).count();
                assert_eq!(out.info.keep_distinct as usize, hit, "decision {j}");
                let per_group: Vec<u32> = (0..preds.len() as u32)
                    .map(|g| members.iter().filter(|m| m.contains(&g)).count() as u32)
                    .collect();
                assert_eq!(out.reached_groups, per_group);
                assert_eq!(out.row_groups.len(), out.rows.len());
                let kept_hit = out.row_groups.iter().filter(|g| !g.is_empty()).count();
                assert_eq!(kept_hit, hit.min(5), "decision {j}");
                assert_eq!(
                    out.rows.len() - kept_hit,
                    (members.len() - hit).min(other as usize)
                );
                for ((row, _), g) in out.rows.iter().zip(&out.row_groups) {
                    assert_eq!(&preds.members(&row.features), g, "the row's own features");
                }
                let at: Vec<usize> = out
                    .positions
                    .iter()
                    .map(|(p, _)| {
                        out.reached
                            .iter()
                            .position(|r| r.pos == *p)
                            .expect("reached")
                    })
                    .collect();
                assert!(at.windows(2).all(|w| w[0] < w[1]), "reach order");
                kept_any += kept_hit;
            }
            // Byte-identical for any thread count, with the same group tallies.
            let mk = |threads| HarvestJob {
                count: 50,
                threads,
                ..job.clone()
            };
            let (a, _) = harvest_rows(&mk(1), &starts, &*l, Driver::Exact, None);
            let want = nn::shard::to_npy(
                &a.iter().map(|x| x.0.clone()).collect::<Vec<_>>(),
                nn::outcome::vocab(4),
            );
            let mut tallies = Vec::new();
            for threads in [1, 4] {
                let mut bytes = Vec::new();
                let sum = write_harvest_shard(
                    &mk(threads),
                    &starts,
                    &*l,
                    Driver::Exact,
                    None,
                    &mut bytes,
                )
                .expect("written");
                assert!(bytes == want, "{threads} threads");
                let t = sum.groups.expect("a targeted harvest's tally");
                let rows_in: Vec<u64> = (0..preds.len())
                    .map(|g| a.iter().filter(|r| preds.holds(g, &r.0.features)).count() as u64)
                    .collect();
                assert_eq!(t.rows, rows_in);
                assert_eq!(
                    t.rows_in_any,
                    a.iter().filter(|r| preds.any(&r.0.features)).count() as u64
                );
                if other == 0 {
                    assert_eq!(t.rows_in_any, 50, "in-group rows only");
                }
                assert_eq!(
                    t.reached_all,
                    sum.decisions
                        .iter()
                        .map(|d| u64::from(d.distinct))
                        .sum::<u64>()
                );
                tallies.push(t);
            }
            assert_eq!(tallies[0], tallies[1]);
        }
        assert!(kept_any > 0, "the groups are reached");
        // Driven by a network at trick 3: the kept rows are in a group and labelled exactly.
        let net = random_oracle(3, 29);
        let job = HarvestJob {
            keep: Some(PredKeep {
                preds: preds.clone(),
                cap: 3,
            }),
            cap: 0,
            ..harvest_job(3, vec![2], [0, 0, 0, 0, 0, 0, 0, 1], 0)
        };
        let mut rows = 0;
        for j in 0..2 {
            let out = harvest_decision(&job, j, &starts, &*l, &source, Driver::Net(&net), &mut mem);
            for (row, (pos, ctx)) in out.rows.iter().map(|r| &r.0).zip(&out.positions) {
                assert!(preds.any(&row.features));
                let v = solve_fast(pos, *ctx, &SyntheticLeaf, FastOptions::default())
                    .value
                    .expect("solved");
                assert_eq!(
                    row.value,
                    v.rotate(nn::encode::rotation(&canonical(pos)) % 4)
                );
                rows += 1;
            }
        }
        assert!(rows > 0);
    }

    #[test]
    #[should_panic(expected = "not both")]
    fn race_and_keep_together_are_refused() {
        let job = HarvestJob {
            race: Some(RaceKeep {
                classes: 0b10000,
                cap: 1,
            }),
            keep: Some(PredKeep {
                preds: test_predicates(),
                cap: 1,
            }),
            ..harvest_job(4, vec![3], [1; 8], 1)
        };
        check_harvest_job(&job);
    }

    #[test]
    fn race_settings_parse() {
        assert_eq!(parse_race_classes("2,4"), Ok(0b10100));
        assert_eq!(parse_race_classes(" 4 , 0"), Ok(0b10001));
        for bad in ["", "5", "2,2", "x", "2,,4", "-1"] {
            assert!(parse_race_classes(bad).is_err(), "{bad:?}");
        }
        assert_eq!(parse_strata("4"), Ok(vec![4]));
        assert_eq!(parse_strata("4,2"), Ok(vec![2, 4]));
        for bad in ["", "0", "5", "1,1", "s4"] {
            assert!(parse_strata(bad).is_err(), "{bad:?}");
        }
        assert_eq!(
            RaceKeep {
                classes: parse_race_classes(RACE_CLASSES).expect("default"),
                cap: RACE_CAP
            }
            .list(),
            [2, 4]
        );
    }

    fn random_oracle(k: u8, seed: u64) -> nn::Oracle {
        let net = nn::synthetic::random_net(seed, &[32, 16], nn::outcome::vocab(k), 100);
        let meta = BTreeMap::from([("head".to_string(), "outcome-v1".to_string())]);
        nn::Oracle::from_bytes(&nn::oracle::write_net(&net, k, &meta), leaf()).expect("loads")
    }

    /// Driven by a network, the harvest's search is the hybrid H_k with that network (the same
    /// candidate values as a plain frontier of the network), it reaches only real frontier
    /// positions (each reachable from the decision's world, one per canonical form, valued by
    /// the network), and the rows are a sample of at most the cap in reach order, each labelled
    /// with the exact solver's value of its position (solved for the label only); the same again
    /// on a second run with fresh memory.
    #[test]
    fn net_driven_harvest_is_the_hybrid_search_and_labels_exactly() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        let source = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &source,
            rules: Rules::HOUSE,
        };
        let cfg = SearchConfig {
            tt_mb: HARVEST_TT_MB,
            ..SearchConfig::default()
        };
        let mut mem = SolveMem::new();
        let (mut decisions, mut repeats, mut capped, mut rows, mut net_differs) = (0, 0, 0, 0, 0);
        let every = [1; 8];
        let last = [0, 0, 0, 0, 0, 0, 0, 1];
        let cases: [(u8, Vec<u8>, [u64; 8], u64); 3] = [
            (4, vec![3], every, 16),
            (5, vec![4], every, 8),
            (3, vec![2], last, 2),
        ];
        for (k, tricks, w, n) in cases {
            let net = random_oracle(k, 23 + u64::from(k));
            let job = harvest_job(k, tricks.clone(), w, 6);
            for j in 0..n {
                let out =
                    harvest_decision(&job, j, &starts, &*l, &source, Driver::Net(&net), &mut mem);
                let (d, cands, m, _) = harvest_root(&job, j, &starts, &source);
                assert!(tricks.contains(&d.trick));
                assert_eq!(
                    (out.info.trick, out.info.position, out.info.stratum),
                    (d.trick, d.position + 1, d.stratum)
                );
                // The recording search is the hybrid with the network at the frontier.
                let rec = Recorder {
                    leaf: &*l,
                    opts: FastOptions::default(),
                    driver: Driver::Net(&net),
                    state: Mutex::new(RecState {
                        mem: SolveMem::new(),
                        seen: BTreeMap::new(),
                        list: Vec::new(),
                        evals: 0,
                    }),
                };
                let search = |cut| {
                    exact_world(
                        &m,
                        &cands,
                        &ctx,
                        &cfg,
                        None,
                        &mut SolveMem::new(),
                        &mut WorldSolve::default(),
                        Some(cut),
                    )
                    .expect("solved")
                };
                assert_eq!(
                    search(FrontierCut::new(k, &rec)),
                    search(FrontierCut::new(k, &net)),
                    "trick {k} decision {j}: the harvest search is H_{k}"
                );
                let st = rec.state.into_inner().expect("unpoisoned");
                assert_eq!(
                    st.list.iter().map(|r| r.canon).collect::<Vec<_>>(),
                    out.reached.iter().map(|r| r.canon).collect::<Vec<_>>(),
                    "the same positions in the same order"
                );
                let (root, _, _) = from_match(&m).expect("a play is awaited");
                let mut all = std::collections::BTreeSet::new();
                reachable(&root, k, &mut all);
                let mut seen = std::collections::BTreeSet::new();
                for r in &out.reached {
                    let p = &r.pos;
                    assert_eq!((p.n_played, p.hand.tricks_done + 1), (0, k));
                    assert!(p.rob.is_none());
                    assert_eq!(canonical(p), r.canon);
                    assert!(all.contains(&r.canon), "reached positions are reachable");
                    assert!(seen.insert(r.canon), "one position per canonical form");
                    assert_eq!(r.value, net.eval(&r.canon, &r.ctx), "the network's value");
                    assert_eq!(r.nodes, 0, "nothing solved during the search");
                }
                assert_eq!(out.info.distinct as usize, out.reached.len());
                repeats += out.info.evals - u64::from(out.info.distinct);
                let take = out.reached.len().min(6);
                assert_eq!(out.rows.len(), take);
                capped += usize::from(out.reached.len() > take);
                let at: Vec<usize> = out
                    .positions
                    .iter()
                    .map(|(p, _)| {
                        out.reached
                            .iter()
                            .position(|r| r.canon == canonical(p))
                            .expect("a sampled position was reached")
                    })
                    .collect();
                assert!(at.windows(2).all(|w| w[0] < w[1]), "reach order");
                // The labels' solves share one table, the decision's (`SolveFamily`): their
                // nodes are those of solves in row order on such a table.
                let mut fam = SolveMem::new();
                fam.share(SolveFamily::of_match(&m).expect("a family"));
                let fam_opts = FastOptions {
                    tt_mb: tt_mb_for(k),
                    ..FastOptions::default()
                };
                for (((row, info), (p, pctx)), &i) in out.rows.iter().zip(&out.positions).zip(&at) {
                    let sol = solve_fast(p, *pctx, &SyntheticLeaf, FastOptions::default());
                    let exact = sol.value.expect("solved");
                    let shared = solve_fast_with(p, *pctx, &SyntheticLeaf, fam_opts, &mut fam);
                    assert_eq!(shared.value, sol.value, "the shared table's value");
                    let c = canonical(p);
                    let r = nn::encode::rotation(&c);
                    assert_eq!(
                        row.value,
                        exact.rotate(r % 4),
                        "the label is the exact value"
                    );
                    assert_eq!((row.source, row.trick), (SOURCE_HARVEST, k));
                    assert_eq!(
                        decode(&row.features).expect("decodes"),
                        rotate_canon(&c, pctx, r)
                    );
                    assert_eq!(row.outcomes[usize::from(row.label)], row.value);
                    assert_eq!((info.root_trick, info.position), (d.trick, d.position + 1));
                    assert_eq!(info.nodes, shared.stats.nodes, "the label's own solve");
                    net_differs += usize::from(out.reached[i].value != exact);
                }
                let again = harvest_decision(
                    &job,
                    j,
                    &starts,
                    &*l,
                    &source,
                    Driver::Net(&random_oracle(k, 23 + u64::from(k))),
                    &mut SolveMem::new(),
                );
                assert_eq!(
                    again.rows.iter().map(|r| &r.0).collect::<Vec<_>>(),
                    out.rows.iter().map(|r| &r.0).collect::<Vec<_>>()
                );
                assert_eq!(
                    serde_json::to_string(&again.info).expect("json"),
                    serde_json::to_string(&out.info).expect("json")
                );
                decisions += 1;
                rows += out.rows.len();
            }
        }
        assert_eq!(decisions, 26);
        assert!(repeats > 0, "the dedupe is exercised");
        assert!(
            capped > 0 && rows > 60,
            "{capped} capped decisions, {rows} rows"
        );
        assert!(net_differs > 0, "the network's values are not the labels");
    }

    /// A net-driven harvest run is byte-identical on 1 and 4 threads (rows, streamed shard,
    /// decisions' facts), its streamed shard is `to_npy` of its rows, and where the network's
    /// search reaches what the exact search reaches its rows are the exact driver's rows.
    #[test]
    fn net_driven_harvest_is_thread_independent() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        for (k, shares, count, cap) in [(4, "0.4,0.4,0.2", 61, 5), (3, "0,0,1", 10, 3)] {
            let net = random_oracle(k, 31);
            let job = |threads, count| HarvestJob {
                count,
                threads,
                ..harvest_job(
                    k,
                    vec![k - 1],
                    parse_band_shares(shares).expect("shares"),
                    cap,
                )
            };
            let (a, da) = harvest_rows(&job(1, count), &starts, &*l, Driver::Net(&net), None);
            let (b, db) = harvest_rows(&job(4, count), &starts, &*l, Driver::Net(&net), None);
            let ra: Vec<Row> = a.iter().map(|x| x.0.clone()).collect();
            let rb: Vec<Row> = b.iter().map(|x| x.0.clone()).collect();
            let want = nn::shard::to_npy(&ra, nn::outcome::vocab(k));
            assert!(
                want == nn::shard::to_npy(&rb, nn::outcome::vocab(k)),
                "trick {k}"
            );
            let facts = |d: &[HarvestDecision]| serde_json::to_string(d).expect("json");
            assert_eq!(facts(&da), facts(&db));
            assert_eq!(da.iter().map(|d| u64::from(d.rows)).sum::<u64>(), count);
            assert!(
                a.iter()
                    .all(|r| r.0.trick == k && r.0.source == SOURCE_HARVEST)
            );
            for threads in [1, 4] {
                let mut out = Vec::new();
                let sum = write_harvest_shard(
                    &job(threads, count),
                    &starts,
                    &*l,
                    Driver::Net(&net),
                    None,
                    &mut out,
                )
                .expect("written");
                assert!(out == want, "trick {k}, {threads} threads");
                assert_eq!(sum.sha256, equity::sha256::sha256_hex(&want));
                assert_eq!(facts(&sum.decisions), facts(&da));
                assert_eq!(sum.counts, counts_of(&a));
            }
            if k == 3 {
                continue;
            }
            // Against the exact driver on the same decisions: where the two searches reach the same
            // positions in the same order (max^n prunes little, so the values steer the reached set
            // little: docs/m3-hybrid/rung1.md), the sample and the rows are the same bytes, since
            // the labels are exact either way.
            let src = BeliefSource::new();
            let mut mem = SolveMem::new();
            let mut same = 0;
            for d in &da {
                let mut run = |driver| {
                    harvest_decision(&job(1, 0), d.id, &starts, &*l, &src, driver, &mut mem)
                };
                let (e, n) = (run(Driver::Exact), run(Driver::Net(&net)));
                if e.reached
                    .iter()
                    .map(|r| r.canon)
                    .eq(n.reached.iter().map(|r| r.canon))
                {
                    same += 1;
                    assert_eq!(
                        e.rows.iter().map(|r| &r.0).collect::<Vec<_>>(),
                        n.rows.iter().map(|r| &r.0).collect::<Vec<_>>(),
                        "decision {}",
                        d.id
                    );
                }
            }
            assert!(same > 0, "some decisions compared");
        }
    }

    /// The facts of a summary that no schedule, table or thread count may change: the shard's
    /// SHA-256 and rows, the decisions' facts, the group tallies, and the rows' counts apart from
    /// the solves' nodes.
    fn identity(sum: &ShardSummary) -> (String, u64, String, Option<GroupTally>, RowCounts) {
        let mut counts = sum.counts;
        counts.nodes = 0;
        (
            sum.sha256.clone(),
            sum.rows,
            serde_json::to_string(&sum.decisions).expect("json"),
            sum.groups.clone(),
            counts,
        )
    }

    /// The identity of the schedules (`HarvestSchedule`): a harvest's shard, its SHA-256, its
    /// decisions' facts, its group tallies and its rows' counts are byte-identical between the
    /// per-decision schedule (before 2026-10-17) and the pooled one, on 1, 2 and 4 threads, with a
    /// decision's labels as one unit or each its own, and with a 1 MB label table: exact driver
    /// and network driver, tricks 3, 4 and 5, the race and targeted harvests and a harvest bounded
    /// by its decisions. The node counts are the per-decision schedule's when a decision's labels
    /// are one unit on the trick's table (tricks 3–5 by default), and with one position a unit
    /// each row's nodes are those of its solve from empty tables.
    #[test]
    fn harvest_schedules_give_the_same_shard() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        let net3 = random_oracle(3, 31);
        let net4 = random_oracle(4, 37);
        let shares = parse_band_shares("0.4,0.4,0.2").expect("shares");
        let base = |k: u8, cap: u32, count: u64| HarvestJob {
            count,
            ..harvest_job(k, vec![k - 1], shares, cap)
        };
        let race = RaceKeep {
            classes: parse_race_classes("2,4").expect("classes"),
            cap: 4,
        };
        let keep = PredKeep {
            preds: test_predicates(),
            cap: 4,
        };
        let cases: Vec<(&str, HarvestJob, Driver<'_>, Option<u64>)> = vec![
            ("trick 4, exact", base(4, 5, 41), Driver::Exact, None),
            ("trick 4, network", base(4, 4, 30), Driver::Net(&net4), None),
            (
                "trick 3, network",
                HarvestJob {
                    schedule: position_schedule(parse_band_shares("0,0,1").expect("shares")),
                    ..base(3, 3, 10)
                },
                Driver::Net(&net3),
                None,
            ),
            ("trick 5, exact", base(5, 4, 23), Driver::Exact, None),
            (
                "trick 4, race",
                HarvestJob {
                    race: Some(race),
                    cap: 1,
                    ..base(4, 1, 30)
                },
                Driver::Exact,
                None,
            ),
            (
                "trick 4, targeted",
                HarvestJob {
                    keep: Some(keep.clone()),
                    cap: 1,
                    ..base(4, 1, 30)
                },
                Driver::Exact,
                None,
            ),
            (
                "trick 4, targeted, network",
                HarvestJob {
                    keep: Some(keep.clone()),
                    cap: 1,
                    ..base(4, 1, 30)
                },
                Driver::Net(&net4),
                None,
            ),
            (
                "trick 3, targeted, network",
                HarvestJob {
                    keep: Some(keep.clone()),
                    cap: 1,
                    schedule: position_schedule(parse_band_shares("0,0,1").expect("shares")),
                    ..base(3, 1, 8)
                },
                Driver::Net(&net3),
                None,
            ),
            ("trick 4, bounded", base(4, 5, 41), Driver::Exact, Some(4)),
        ];
        for (what, job, driver, max) in &cases {
            let k = job.trick;
            let run = |threads: usize, exec: LabelExec| {
                let job = HarvestJob {
                    threads,
                    ..job.clone()
                };
                let mut out = Vec::new();
                let sum = write_harvest_shard_with(
                    &job, exec, *max, &starts, &*l, *driver, None, &mut out,
                )
                .expect("written");
                assert_eq!(sum.sha256, equity::sha256::sha256_hex(&out), "{what}");
                (out, sum)
            };
            let before = LabelExec {
                schedule: HarvestSchedule::PerDecision,
                ..LabelExec::for_trick(k)
            };
            let (want, want_sum) = run(1, before);
            assert!(want_sum.rows > 0, "{what}");
            let one_each = LabelExec {
                group: 1,
                ..LabelExec::for_trick(k)
            };
            let small = LabelExec {
                tt_mb: 1,
                ..LabelExec::for_trick(k)
            };
            for threads in [1, 2, 4] {
                for (exec, same_nodes) in [
                    (before, true),
                    (LabelExec::for_trick(k), true),
                    (one_each, false),
                    (small, false),
                ] {
                    let (got, sum) = run(threads, exec);
                    assert!(got == want, "{what}: {exec:?} on {threads} threads");
                    assert_eq!(identity(&sum), identity(&want_sum), "{what}: {exec:?}");
                    if same_nodes {
                        assert_eq!(sum.counts, want_sum.counts, "{what}: {exec:?}");
                    }
                }
            }
            // In memory too ([`harvest_rows_with`]), the same rows.
            if max.is_none() {
                let rows = |exec| {
                    let job = HarvestJob {
                        threads: 3,
                        ..job.clone()
                    };
                    harvest_rows_with(&job, exec, &starts, &*l, *driver, None)
                };
                let (a, da) = rows(before);
                let (b, db) = rows(one_each);
                let ra: Vec<Row> = a.iter().map(|x| x.0.clone()).collect();
                let rb: Vec<Row> = b.iter().map(|x| x.0.clone()).collect();
                assert!(
                    nn::shard::to_npy(&ra, nn::outcome::vocab(k)) == want,
                    "{what}"
                );
                assert!(ra == rb, "{what}");
                assert_eq!(
                    serde_json::to_string(&da).expect("json"),
                    serde_json::to_string(&db).expect("json")
                );
                // One position a unit: each row's nodes are its solve's from empty tables.
                if matches!(driver, Driver::Net(_)) {
                    let src = BeliefSource::new();
                    let mut mem = SolveMem::new();
                    let mut at = 0;
                    for d in &db {
                        let out =
                            harvest_decision(job, d.id, &starts, &*l, &src, *driver, &mut mem);
                        for (p, c) in out.positions.iter().take(d.rows as usize) {
                            let fresh = solve_fast_with(
                                p,
                                *c,
                                &SyntheticLeaf,
                                label_opts(tt_mb_for(k)),
                                &mut SolveMem::new(),
                            );
                            assert_eq!(b[at].1.nodes, fresh.stats.nodes, "{what}: row {at}");
                            at += 1;
                        }
                    }
                    assert_eq!(at, b.len());
                }
            }
        }
    }

    /// The label solves' table never changes a label: the plain sources' shards are
    /// byte-identical with a 1 MB table and the trick's (only the solves' node counts may differ).
    #[test]
    fn label_table_size_never_changes_a_row() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        for (trick, source, count) in [(3, Source::Random, 12), (4, Source::Rules, 30)] {
            let job = LabelJob {
                trick,
                count,
                seed: 13,
                source,
                threads: 2,
            };
            let shard = |tt_mb| {
                let mut out = Vec::new();
                let exec = LabelExec {
                    tt_mb,
                    ..LabelExec::for_trick(trick)
                };
                let sum = write_label_shard_with(&job, exec, &starts, &*l, None, 4, &mut out)
                    .expect("written");
                (out, sum)
            };
            let (a, sa) = shard(tt_mb_for(trick));
            let (b, sb) = shard(1);
            assert!(a == b, "trick {trick}");
            assert_eq!(identity(&sa), identity(&sb));
        }
    }

    /// The pooled schedule's units: one position each at trick 2, a decision's at tricks 3-5.
    #[test]
    fn label_units_by_trick() {
        assert_eq!(label_group_for(2), 1);
        for k in 3..=5 {
            assert_eq!(label_group_for(k), usize::MAX);
        }
        assert_eq!(LabelExec::for_trick(2).tt_mb, 256);
        assert_eq!(LabelExec::for_trick(2).schedule, HarvestSchedule::Pooled);
        for s in [HarvestSchedule::Pooled, HarvestSchedule::PerDecision] {
            assert_eq!(HarvestSchedule::parse(s.name()), Ok(s));
        }
        assert!(HarvestSchedule::parse("fast").is_err());
    }

    /// One slow row never stops the other threads, whatever the chunk size: while row 0 of a
    /// shard in one-row chunks is held, the other workers make every other row (399, against
    /// the 32 one-row chunks a window of `4 × threads + 16` units would have let them make), and
    /// the shard is the same bytes as made without the hold. The window is a byte budget
    /// ([`label_window`]): at trick 2 about 570,000 rows.
    #[test]
    fn a_slow_row_never_stops_the_others() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        let job = LabelJob {
            trick: 4,
            count: 400,
            seed: 11,
            source: Source::Rules,
            threads: 4,
        };
        let rows = label_rows(&job, &starts, &*l, None);
        let want = nn::shard::to_npy(
            &rows.iter().map(|r| r.0.clone()).collect::<Vec<_>>(),
            nn::outcome::vocab(4),
        );
        assert!(window_for(4) < 399, "the old window would stop the others");
        let row_bytes = row_bytes_for(nn::outcome::vocab(2)) as u64;
        assert_eq!(label_window(2, 1, 368), LABEL_WINDOW_BYTES / row_bytes);
        assert!(label_window(2, 1, 368) > 500_000);
        assert_eq!(label_window(2, 1 << 40, 368), window_for(368));
        let made = AtomicUsize::new(0);
        let mut out = Vec::new();
        let sum = stream_label_rows(
            4,
            400,
            4,
            1,
            |_, i| {
                if i == 0 {
                    // Held until every other row is made (at most 20 s: then the test fails).
                    let t = Instant::now();
                    while made.load(Ordering::SeqCst) < 399 && t.elapsed() < Duration::from_secs(20)
                    {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    assert_eq!(
                        made.load(Ordering::SeqCst),
                        399,
                        "the other rows are made while row 0 is held"
                    );
                }
                made.fetch_add(1, Ordering::SeqCst);
                rows[i as usize].clone()
            },
            None,
            &mut out,
        )
        .expect("written");
        assert!(out == want);
        assert_eq!(sum.counts, counts_of(&rows));
    }

    // ------------------------------------------------------------ hybrid labels

    /// The hybrid value of `pos` computed independently: the naive solver (`solver::solve_cut`,
    /// no transposition table, no equivalence merging) with the network at the next trick.
    fn naive_hybrid(pos: &Position, ctx: MatchCtx, net: &dyn Frontier) -> MoneyChance {
        let cut = FrontierCut::new(pos.hand.tricks_done + 2, net);
        solver::solve_cut(
            pos,
            ctx,
            &SyntheticLeaf,
            solver::Options::default(),
            Some(cut),
        )
        .value
        .expect("no time cap")
    }

    /// Hybrid labels (`Labeller::Hybrid`): the same positions as the exact labels of the same
    /// job (encodings, source, trick, mask and outcome vectors), each valued by H_{k+1} as the
    /// naive solver computes it independently (rotated as encoded), the outcome label the
    /// nearest possible outcome; byte-identical for any thread count and chunk size; and the
    /// rows' facts say hybrid with the network's evaluations.
    #[test]
    fn hybrid_labels_are_h_next_on_the_exact_rows_positions() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        for (trick, source, count) in [
            (4, Source::Random, 24),
            (3, Source::Rules, 12),
            (2, Source::Random, 4),
        ] {
            let net = random_oracle(trick + 1, 41 + u64::from(trick));
            let hy = Labeller::Hybrid(HybridLabels {
                net: &net,
                exact_ppm: 0,
            });
            let job = |threads| LabelJob {
                trick,
                count,
                seed: 7,
                source,
                threads,
            };
            let rows = label_rows_by(&job(1), &starts, &*l, hy, None);
            for (i, (row, info)) in rows.iter().enumerate() {
                let (pos, ctx, base) =
                    row_position(&job(1), i as u64, &starts, &BeliefSource::new());
                let h = naive_hybrid(&pos, ctx, &net);
                let c = canonical(&pos);
                let r = nn::encode::rotation(&c);
                assert_eq!(row.value, h.rotate(r % 4), "trick {trick} row {i}");
                assert_eq!(row.value.0.iter().sum::<u64>(), 1 << 32);
                let o = outcomes(&c, &ctx, &*l);
                assert_eq!(row.label, o.nearest(&row.value));
                assert_eq!((row.mask, &row.outcomes), (o.mask, &o.vecs));
                assert_eq!(row.mask >> row.label & 1, 1);
                assert!(info.hybrid, "{info:?}");
                assert_eq!((info.stratum, info.position), (base.stratum, base.position));
                assert_eq!(info.class, row.race_class());
            }
            let streamed = |threads, chunk| {
                let mut out = Vec::new();
                let exec = LabelExec::for_trick(trick);
                let sum = write_label_shard_by(
                    &job(threads),
                    exec,
                    &starts,
                    &*l,
                    hy,
                    None,
                    chunk,
                    &mut out,
                )
                .expect("written");
                (out, sum)
            };
            let want = nn::shard::to_npy(
                &rows.iter().map(|r| r.0.clone()).collect::<Vec<_>>(),
                nn::outcome::vocab(trick),
            );
            let (a, sa) = streamed(1, 1);
            let (b, sb) = streamed(3, 5);
            assert!(a == want && b == want, "trick {trick}");
            assert_eq!(sa.counts.hybrid, count);
            assert!(sa.counts.evals > 0, "the network values the frontier");
            assert_eq!(identity(&sa), identity(&sb));
            // The positions are the exact labels' (at trick 2 their exact solves are seconds
            // each: the positions are compared, not solved).
            if trick >= 3 {
                let exact = label_rows(&job(1), &starts, &*l, None);
                for ((e, _), (h, _)) in exact.iter().zip(&rows) {
                    assert_eq!(
                        (&e.features, e.source, e.trick, e.mask, &e.outcomes),
                        (&h.features, h.source, h.trick, h.mask, &h.outcomes)
                    );
                }
                assert!(
                    exact.iter().zip(&rows).any(|(e, h)| e.0.value != h.0.value),
                    "a random network is not the exact solver"
                );
            }
        }
    }

    /// The nearest outcome is the exact rule's label for every exact value, so an exact row's
    /// label never depends on which rule made it.
    #[test]
    fn the_nearest_outcome_of_an_exact_value_is_its_label() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        let rows = label_rows(
            &LabelJob {
                trick: 4,
                count: 40,
                seed: 19,
                source: Source::Random,
                threads: 2,
            },
            &starts,
            &*l,
            None,
        );
        for (row, _) in &rows {
            let o = nn::outcome::Outcomes {
                n: row.outcomes.len(),
                mask: row.mask,
                vecs: row.outcomes.clone(),
            };
            assert_eq!(o.nearest(&row.value), row.label);
            assert_eq!(o.label(&row.value), Some(row.label));
        }
    }

    /// The exact anchor: a share of the rows labelled exactly, drawn by each row's key. All of
    /// them (10⁶ per million) is the exact shard byte for byte; a half gives exact rows exactly
    /// where `anchored` says (the exact shard's rows there, the hybrid shard's elsewhere).
    #[test]
    fn the_exact_anchor_is_drawn_by_the_rows_key() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        let net = random_oracle(4, 43);
        let job = LabelJob {
            trick: 3,
            count: 16,
            seed: 23,
            source: Source::Random,
            threads: 2,
        };
        let by = |ppm| {
            label_rows_by(
                &job,
                &starts,
                &*l,
                Labeller::Hybrid(HybridLabels {
                    net: &net,
                    exact_ppm: ppm,
                }),
                None,
            )
        };
        let exact = label_rows(&job, &starts, &*l, None);
        let hybrid = by(0);
        let rows_of = |r: &[(Row, RowInfo)]| r.iter().map(|x| x.0.clone()).collect::<Vec<_>>();
        assert_eq!(rows_of(&by(1_000_000)), rows_of(&exact));
        let half = by(500_000);
        let mut n_exact = 0;
        for (i, (row, info)) in half.iter().enumerate() {
            let a = anchored(500_000, 23, SOURCE_RANDOM, 3, i as u64, 0);
            assert_eq!(info.hybrid, !a);
            let want = if a { &exact[i].0 } else { &hybrid[i].0 };
            assert_eq!(row, want, "row {i}");
            n_exact += usize::from(a);
        }
        assert!(
            n_exact > 0 && n_exact < half.len(),
            "{n_exact} of {}",
            half.len()
        );
        assert!(!anchored(0, 23, SOURCE_RANDOM, 3, 0, 0));
        assert!(anchored(1_000_000, 23, SOURCE_RANDOM, 3, 0, 0));
        assert_eq!(parse_exact_share("0.1"), Ok(100_000));
        assert_eq!(parse_exact_share("1"), Ok(1_000_000));
        assert_eq!(parse_exact_share(".25"), Ok(250_000));
        assert_eq!(parse_exact_share("0.000001"), Ok(1));
        for bad in ["", ".", "1.5", "0.1234567", "-0.1", "x", "2"] {
            assert!(parse_exact_share(bad).is_err(), "{bad}");
        }
    }

    /// Hybrid labels on a harvest: the same decisions and positions as the exact harvest (each
    /// row's encoding, mask and outcomes), each valued by H_{k+1} (the naive solver's, rotated as
    /// encoded), whatever the search's own exact values; the exact anchor by (decision, place);
    /// and the shard byte-identical over schedules and thread counts.
    #[test]
    fn hybrid_labels_on_a_harvest() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        let net5 = random_oracle(5, 47);
        let shares = parse_band_shares("0.4,0.4,0.2").expect("shares");
        let job = |threads| HarvestJob {
            count: 30,
            threads,
            ..harvest_job(4, vec![3], shares, 4)
        };
        let hy = |ppm| {
            Labeller::Hybrid(HybridLabels {
                net: &net5,
                exact_ppm: ppm,
            })
        };
        let exec = LabelExec::for_trick(4);
        let (exact, de) = harvest_rows(&job(1), &starts, &*l, Driver::Exact, None);
        let (rows, dh) = harvest_rows_by(&job(1), exec, &starts, &*l, Driver::Exact, hy(0), None);
        assert_eq!(
            serde_json::to_string(&de).expect("json"),
            serde_json::to_string(&dh).expect("json")
        );
        assert_eq!(rows.len(), exact.len());
        let mut keys = Vec::new();
        for d in &dh {
            for p in 0..u64::from(d.rows) {
                keys.push((d.id, p));
            }
        }
        for ((e, _), (h, info)) in exact.iter().zip(&rows) {
            assert_eq!(
                (&e.features, e.source, e.mask, &e.outcomes),
                (&h.features, h.source, h.mask, &h.outcomes)
            );
            assert!(info.hybrid);
        }
        // Each row's value is H_5 of its position: the harvest's own positions, re-reached.
        let mut mem = SolveMem::new();
        let source = BeliefSource::new();
        let mut i = 0;
        for d in &dh {
            let out = harvest_decision(
                &job(1),
                d.id,
                &starts,
                &*l,
                &source,
                Driver::Exact,
                &mut mem,
            );
            for (pos, ctx) in out.positions.iter().take(d.rows as usize) {
                let h = naive_hybrid(pos, *ctx, &net5);
                let r = nn::encode::rotation(&canonical(pos));
                assert_eq!(rows[i].0.value, h.rotate(r % 4), "row {i}");
                i += 1;
            }
        }
        assert_eq!(i, rows.len());
        // The anchor: by (decision, place in its sample).
        let (half, _) = harvest_rows_by(
            &job(1),
            exec,
            &starts,
            &*l,
            Driver::Exact,
            hy(500_000),
            None,
        );
        for (k, ((row, info), &(j, p))) in half.iter().zip(&keys).enumerate() {
            let a = anchored(500_000, 7, SOURCE_HARVEST, 4, j, p);
            assert_eq!(info.hybrid, !a);
            assert_eq!(row, if a { &exact[k].0 } else { &rows[k].0 }, "row {k}");
        }
        // Byte-identical over schedules and threads.
        let shard = |threads, schedule| {
            let mut out = Vec::new();
            let e = LabelExec { schedule, ..exec };
            let sum = write_harvest_shard_by(
                &job(threads),
                e,
                None,
                &starts,
                &*l,
                Driver::Exact,
                hy(500_000),
                None,
                &mut out,
            )
            .expect("written");
            (out, sum)
        };
        let (a, sa) = shard(1, HarvestSchedule::PerDecision);
        let (b, sb) = shard(3, HarvestSchedule::Pooled);
        let want = nn::shard::to_npy(
            &half.iter().map(|r| r.0.clone()).collect::<Vec<_>>(),
            nn::outcome::vocab(4),
        );
        assert!(a == want && b == want);
        let (mut ia, mut ib) = (identity(&sa), identity(&sb));
        ia.4.evals = 0;
        ib.4.evals = 0;
        assert_eq!(ia, ib);
        assert_eq!(
            sa.counts.hybrid,
            half.iter().filter(|r| r.1.hybrid).count() as u64
        );
    }

    /// Exact labels are byte-identical to those of the code before hybrid labels (2026-10-19,
    /// commit b40222c): the SHA-256 of a random-play shard, a rules-v1 shard and a harvested
    /// shard made by the old code (the same job, the synthetic leaf), here through the default
    /// labeller and through `Labeller::Exact` given explicitly.
    #[test]
    fn exact_labels_are_byte_identical_to_before() {
        let starts = collect_starts(400, 9);
        let l = leaf();
        let plain = |trick, source, count| {
            let job = LabelJob {
                trick,
                count,
                seed: 29,
                source,
                threads: 2,
            };
            let mut a = Vec::new();
            let sa = write_label_shard(&job, &starts, &*l, None, 3, &mut a).expect("written");
            let mut b = Vec::new();
            let sb = write_label_shard_by(
                &job,
                LabelExec::for_trick(trick),
                &starts,
                &*l,
                Labeller::Exact,
                None,
                3,
                &mut b,
            )
            .expect("written");
            assert!(a == b);
            assert_eq!(identity(&sa), identity(&sb));
            assert_eq!(sa.counts.hybrid, 0);
            sa.sha256
        };
        let harvest = {
            let job = HarvestJob {
                count: 40,
                threads: 2,
                ..harvest_job(4, vec![3], parse_band_shares("0.4,0.4,0.2").expect("s"), 4)
            };
            let mut a = Vec::new();
            write_harvest_shard(&job, &starts, &*l, Driver::Exact, None, &mut a)
                .expect("written")
                .sha256
        };
        let got = [
            plain(4, Source::Random, 40),
            plain(3, Source::Rules, 16),
            harvest,
        ];
        assert_eq!(got, GOLDEN_EXACT_SHARDS, "{got:?}");
    }

    /// The SHA-256 of [`exact_labels_are_byte_identical_to_before`]'s shards, made by the code
    /// of commit b40222c (before hybrid labels).
    const GOLDEN_EXACT_SHARDS: [&str; 3] = [
        "31cf6a5ceb558872be506ff8283830dc92bc08648e340fdc3a6c7ea29699dac6",
        "bdb81dd4c9799f6a493f7ef17fccbeed5bdf192b438e7235eb91f946babf58e6",
        "969fc5c5ad1a00eeec11f795262d94754ac0462395a70483ecef54657728f660",
    ];
}
