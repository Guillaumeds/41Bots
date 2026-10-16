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
//! (`nn::oracle`).
//!
//! **Streaming** ([`write_label_shard`], [`write_harvest_shard`]): a shard of millions of rows is
//! never held in memory. Workers make chunks of consecutive rows (or harvest decisions), each
//! claimed as a worker comes free (`crate::ordered`), already as the shard's bytes; the chunks are
//! hashed (SHA-256, for the sidecar) in row order as they arrive and written in that order by a
//! writer thread, so hashing and writing overlap the making of later rows instead of following
//! it. The bytes and the hash are those of [`nn::shard::to_npy`] of [`label_rows`] /
//! [`harvest_rows`] (tested), for any thread count and chunk size.
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
    solve_fast_with,
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

/// The label row of a position at the start of trick `pos.hand.tricks_done + 1` with exact value
/// `v`: the encoding of its canonical form, the value in the encoder's rotated team order, and
/// the outcome head's label, mask and vectors (the same for every source).
///
/// # Panics
/// If no possible outcome has the exact value (the outcome vocabulary would be incomplete).
fn labelled_row(
    pos: &Position,
    ctx: &MatchCtx,
    v: MoneyChance,
    source: u8,
    leaf: &SharedLeaf,
) -> Row {
    let c = canonical(pos);
    let (features, r) = encode(&c, ctx, leaf);
    let value = v.rotate(r % 4);
    let o = outcomes(&c, ctx, leaf);
    let label = o.label(&value).unwrap_or_else(|| {
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

/// Row `i` of a job, labelled, and its facts.
fn make_row(
    job: &LabelJob,
    i: u64,
    starts: &[Vec<HandStartState>; 4],
    source: &dyn WorldSource,
    leaf: &SharedLeaf,
    mem: &mut SolveMem,
) -> (Row, RowInfo) {
    let opts = FastOptions {
        tt_mb: tt_mb_for(job.trick),
        ..FastOptions::default()
    };
    let (pos, ctx, mut info) = row_position(job, i, starts, source);
    let (v, nodes) = exact_value(&pos, ctx, leaf, opts, mem);
    info.nodes = nodes;
    let row = labelled_row(&pos, &ctx, v, job.source.tag(), leaf);
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
    check_label_job(job);
    let source = BeliefSource::new();
    let mut rows = Vec::with_capacity(job.count as usize);
    ordered(
        job.threads,
        Some(job.count),
        window_for(job.threads),
        SolveMem::new,
        |mem, i| {
            let r = make_row(job, i, starts, &source, leaf, mem);
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
/// units leave no long tail on hundreds of threads (32 rows are about 10 ms of work), large
/// enough that handing them over costs nothing.
pub const CHUNK_ROWS: u64 = 32;

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
    check_label_job(job);
    let (n, v) = (job.count, nn::outcome::vocab(job.trick));
    let chunk = chunk_rows.max(1);
    let source = BeliefSource::new();
    let mut counts = RowCounts::default();
    let ((), sha256, hash_time, write_time) = stream_shard(out, n, v, |sink| {
        ordered(
            job.threads,
            Some(n.div_ceil(chunk)),
            window_for(job.threads),
            SolveMem::new,
            |mem, c| {
                let (lo, hi) = (c * chunk, ((c + 1) * chunk).min(n));
                let mut bytes = Vec::with_capacity((hi - lo) as usize * row_bytes_for(v));
                let mut cnt = RowCounts::default();
                for i in lo..hi {
                    let (row, info) = make_row(job, i, starts, &source, leaf, mem);
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

/// Harvest decision `j` ([`harvest_root`]): the hybrid's search from it for every candidate
/// (`crate::pimc::exact_world` with a frontier at the start of `job.trick` and the [`Recorder`]
/// there, valuing the frontier as `driver` says), the distinct reached positions, a seeded sample
/// of at most `job.cap` of them (kept in reach order), each labelled with its exact value (from
/// the search with [`Driver::Exact`]; solved now, and only these, with [`Driver::Net`]). The
/// decision's exact solves share one table (`solver::SolveFamily`): the same values as solves
/// from empty tables, fewer nodes.
pub fn harvest_decision(
    job: &HarvestJob,
    j: u64,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    source: &dyn WorldSource,
    driver: Driver<'_>,
    mem: &mut SolveMem,
) -> DecisionHarvest {
    let k = job.trick;
    let (d, cands, m, mut rng) = harvest_root(job, j, starts, source);
    let (t, st, pos) = (d.trick, usize::from(d.stratum - 1), usize::from(d.position));
    let opts = FastOptions {
        tt_mb: tt_mb_for(k),
        ..FastOptions::default()
    };
    // The exact solves of the decision's frontier positions (every reached one with the exact
    // driver, the sampled ones with a network) share one table and leaf memo: they all descend
    // from the decision in this one world (`solver::SolveFamily`: the values are those of
    // solves from empty tables; only the node counts in the rows' facts change).
    let mut label_mem = SolveMem::new();
    if let Some(f) = SolveFamily::of_match(&m) {
        label_mem.share(f);
    }
    let rec = Recorder {
        leaf,
        opts,
        driver,
        state: Mutex::new(RecState {
            mem: label_mem,
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
    let t0 = Instant::now();
    exact_world(
        &m,
        &cands,
        &ctx,
        &cfg,
        None,
        mem,
        &mut run,
        Some(FrontierCut::new(k, &rec)),
    )
    .expect("nothing stops a harvest search");
    let search_time = t0.elapsed();
    let t1 = Instant::now();
    let RecState {
        mem: mut label_mem,
        seen: _,
        list,
        evals,
    } = rec.state.into_inner().unwrap_or_else(|e| e.into_inner());
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
    let take = chosen.len();
    let mut rows = Vec::with_capacity(take);
    let mut positions = Vec::with_capacity(take);
    for &i in &chosen {
        let r = &list[i];
        let (value, nodes) = match driver {
            Driver::Exact => (r.value, r.nodes),
            Driver::Net(_) => exact_value(&r.pos, r.ctx, leaf, opts, &mut label_mem),
        };
        let row = labelled_row(&r.pos, &r.ctx, value, SOURCE_HARVEST, leaf);
        let class = row.race_class();
        rows.push((
            row,
            RowInfo {
                stratum: st as u8 + 1,
                root_trick: t,
                position: pos as u8 + 1,
                retries: 0,
                nodes,
                class,
            },
        ));
        positions.push((r.pos, r.ctx));
    }
    let row_groups = if members.is_empty() {
        Vec::new()
    } else {
        chosen.iter().map(|&i| members[i].clone()).collect()
    };
    DecisionHarvest {
        info: HarvestDecision {
            id: j,
            trick: t,
            position: pos as u8 + 1,
            stratum: st as u8 + 1,
            cands: cands.len(),
            evals,
            distinct: n as u32,
            rows: take as u32,
            search_nodes: run.stats.nodes,
            race_distinct,
            keep_distinct,
        },
        rows,
        positions,
        reached: list,
        row_groups,
        reached_groups,
        search_time,
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

/// Harvest decisions 0, 1, 2, … made on `job.threads` threads ([`harvest_decision`]) and handed
/// to `take` in order with their rows (each worker with its own solver memory) until `take`
/// returns `false` (decisions are claimed as workers come free, at most a window ahead of
/// `take`: `crate::ordered`).
#[allow(clippy::too_many_arguments)]
fn harvest_decisions<T: Send>(
    job: &HarvestJob,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    driver: Driver<'_>,
    progress: Option<&AtomicUsize>,
    make: impl Fn(DecisionHarvest) -> T + Sync,
    take: impl FnMut(u64, T) -> Result<bool, String>,
) -> Result<(), String> {
    let source = BeliefSource::new();
    ordered(
        job.threads,
        None,
        window_for(job.threads),
        SolveMem::new,
        |mem, j| {
            let out = harvest_decision(job, j, starts, leaf, &source, driver, mem);
            if let Some(p) = progress {
                p.fetch_add(out.rows.len(), Ordering::Relaxed);
            }
            make(out)
        },
        take,
    )
}

/// The rows of a harvest run and the facts of the decisions they came from: decisions 0, 1, 2, …
/// in order, each giving its rows ([`harvest_decision`], the frontier valued as `driver` says),
/// until `job.count` rows (the last decision cut short). Decisions run on `job.threads` threads;
/// the result is byte-identical for any thread count.
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
    check_harvest_job(job);
    let mut rows = Vec::with_capacity(job.count as usize);
    let mut decisions = Vec::new();
    if job.count > 0 {
        harvest_decisions(
            job,
            starts,
            leaf,
            driver,
            progress,
            |out| (out.info, out.rows),
            |_, (mut info, mut got)| {
                let keep = ((job.count - rows.len() as u64) as usize).min(got.len());
                got.truncate(keep);
                info.rows = keep as u32;
                rows.extend(got);
                decisions.push(info);
                Ok((rows.len() as u64) < job.count)
            },
        )
        .expect("collecting rows cannot fail");
    }
    debug_assert_eq!(rows.len() as u64, job.count);
    (rows, decisions)
}

/// A harvest run's shard ([`nn::shard`]'s `.npy` bytes of [`harvest_rows`]) streamed to `out`:
/// each decision's rows hashed and written in decision order as they come (module notes). The
/// same bytes, hash, counts and decisions as [`harvest_rows`] for any thread count.
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
    check_harvest_job(job);
    assert!(max_decisions != Some(0), "a bound of at least one decision");
    let v = nn::outcome::vocab(job.trick);
    let rb = row_bytes_for(v);
    let mut counts = RowCounts::default();
    let mut decisions = Vec::new();
    let mut left = job.count;
    let (mut search_time, mut label_time) = (Duration::ZERO, Duration::ZERO);
    let mut groups = job.keep.as_ref().map(|k| GroupTally::new(k.preds.len()));
    let fill = |sink: &mut ShardSink| {
        if left == 0 {
            return Ok(());
        }
        harvest_decisions(
            job,
            starts,
            leaf,
            driver,
            progress,
            |out| {
                let mut bytes = Vec::with_capacity(out.rows.len() * rb);
                let mut infos = Vec::with_capacity(out.rows.len());
                for (row, info) in &out.rows {
                    assert_eq!(row.outcomes.len(), v, "every row has the trick's outcomes");
                    row_bytes(row, &mut bytes);
                    infos.push(*info);
                }
                (
                    out.info,
                    bytes,
                    infos,
                    (out.search_time, out.label_time),
                    (out.row_groups, out.reached_groups),
                )
            },
            |_, (mut info, mut bytes, infos, (ts, tl), (row_groups, reached_groups))| {
                search_time += ts;
                label_time += tl;
                let keep = left.min(infos.len() as u64) as usize;
                bytes.truncate(keep * rb);
                for i in &infos[..keep] {
                    counts.add(i);
                }
                if let Some(t) = &mut groups {
                    t.add(&info, &reached_groups, &row_groups[..keep]);
                }
                info.rows = keep as u32;
                decisions.push(info);
                left -= keep as u64;
                sink.put(bytes)?;
                Ok(left > 0 && max_decisions.is_none_or(|m| (decisions.len() as u64) < m))
            },
        )
    };
    let ((), sha256, hash_time, write_time, bytes) = stream_shard_upto(out, job.count, v, fill)?;
    let rows = job.count - left;
    let want = nn::shard::header(job.count as usize, v).len() as u64 + rows * rb as u64;
    if bytes != want || (left > 0 && max_decisions.is_none()) {
        return Err(format!(
            "the shard has {bytes} bytes, but {rows} rows make {want} (of {} asked)",
            job.count
        ));
    }
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
            for threads in [1, 4] {
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
}
