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
//!   seeded sample of at most a cap per decision.
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
//! threads into their slots, so the shard is byte-identical for any thread count. Harvested rows
//! come by decision: decision `j`'s rows depend only on (seed, trick, `j`) and the harvest's
//! settings, and the shard is decisions 0, 1, 2, … in order, cut at the row count.
//!
//! **Disjoint from the golden sets**: the golden decision sets use seed 1; seed 1 is refused.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use engine::game::{HandStartState, Match, Phase, Rules};
use engine::rng::Rng;
use equity::MoneyChance;
use nn::encode::encode;
use nn::outcome::outcomes;
use nn::shard::{Row, SOURCE_HARVEST, SOURCE_RANDOM, SOURCE_RULES};
use serde::Serialize;
use solver::frontier::{CanonPos, Frontier};
use solver::{
    FastOptions, FrontierCut, MatchCtx, Position, SolveMem, canonical, from_match, solve_fast_with,
};

use crate::measure::{DecisionSpec, build_decision};
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
fn exact_value(
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
            let (v, nodes) = exact_value(&pos, ctx, leaf, opts, &mut mem);
            info.nodes = nodes;
            let row = labelled_row(&pos, &ctx, v, job.source.tag(), leaf);
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
    /// The decider position of decision slot `q` (decisions `4q .. 4q + 3`, one per stratum) is
    /// `schedule[q mod len]` (0-based; [`position_schedule`]).
    pub schedule: Vec<u8>,
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

/// One frontier position reached by a harvest search.
#[derive(Clone, Copy, Debug)]
pub struct Reached {
    /// The position as the search first reached it.
    pub pos: Position,
    /// The match around it.
    pub ctx: MatchCtx,
    /// Its canonical form.
    pub canon: CanonPos,
    /// Its exact value (absolute team order).
    pub value: MoneyChance,
    /// Solver nodes of that exact value.
    pub nodes: u64,
}

struct RecState {
    mem: SolveMem,
    seen: BTreeMap<CanonPos, usize>,
    list: Vec<Reached>,
    evals: u64,
}

/// The harvest's frontier evaluator: records every position the solver reaches at the frontier
/// (first visits, in the order the sequential solver reaches them) and returns its exact value,
/// so the search is exactly the hybrid H_k with a perfect network at the frontier (the exact
/// solver's own search down to trick `k`). A canonical form met again (another line, or the same
/// position after a table replacement) gets its recorded value. The exact value depends only on
/// the canonical form and the context (`solver::frontier`), as the solvers require.
struct Recorder<'a> {
    leaf: &'a SharedLeaf,
    opts: FastOptions,
    state: Mutex<RecState>,
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
        let (value, nodes) = exact_value(pos, *ctx, self.leaf, self.opts, &mut st.mem);
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
}

/// Harvest decision `j` before its search: the decision (the yardstick's generator,
/// [`crate::measure::build_decision`], at a decision trick drawn uniformly from `job.tricks`, the
/// stratum `j mod 4 + 1` and the scheduled position), its candidates, a world drawn from the
/// decider's belief rebuilt as a match, and the decision's random stream (which then draws the
/// sample).
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
    let st = (j % 4) as usize;
    let pos = usize::from(job.schedule[((j / 4) % job.schedule.len() as u64) as usize]);
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
/// there), the distinct reached positions, a seeded sample of at most `job.cap` of them (kept in
/// reach order), each labelled.
pub fn harvest_decision(
    job: &HarvestJob,
    j: u64,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    source: &dyn WorldSource,
    mem: &mut SolveMem,
) -> DecisionHarvest {
    let k = job.trick;
    let (d, cands, m, mut rng) = harvest_root(job, j, starts, source);
    let (t, st, pos) = (d.trick, usize::from(d.stratum - 1), usize::from(d.position));
    let rec = Recorder {
        leaf,
        opts: FastOptions {
            tt_mb: tt_mb_for(k),
            ..FastOptions::default()
        },
        state: Mutex::new(RecState {
            mem: SolveMem::new(),
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
        mem,
        &mut run,
        Some(FrontierCut::new(k, &rec)),
    )
    .expect("nothing stops a harvest search");
    let st_rec = rec.state.into_inner().unwrap_or_else(|e| e.into_inner());
    let n = st_rec.list.len();
    let take = n.min(job.cap as usize);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..take {
        let r = i + rng.bounded((n - i) as u64) as usize;
        idx.swap(i, r);
    }
    let mut chosen = idx[..take].to_vec();
    chosen.sort_unstable();
    let mut rows = Vec::with_capacity(take);
    let mut positions = Vec::with_capacity(take);
    for &i in &chosen {
        let r = &st_rec.list[i];
        rows.push((
            labelled_row(&r.pos, &r.ctx, r.value, SOURCE_HARVEST, leaf),
            RowInfo {
                stratum: st as u8 + 1,
                root_trick: t,
                position: pos as u8 + 1,
                retries: 0,
                nodes: r.nodes,
            },
        ));
        positions.push((r.pos, r.ctx));
    }
    DecisionHarvest {
        info: HarvestDecision {
            id: j,
            trick: t,
            position: pos as u8 + 1,
            stratum: st as u8 + 1,
            cands: cands.len(),
            evals: st_rec.evals,
            distinct: n as u32,
            rows: take as u32,
            search_nodes: run.stats.nodes,
        },
        rows,
        positions,
        reached: st_rec.list,
    }
}

/// The rows of a harvest run and the facts of the decisions they came from: decisions 0, 1, 2, …
/// in order, each giving its rows ([`harvest_decision`]), until `job.count` rows (the last
/// decision cut short). Decisions run on `job.threads` threads; the result is byte-identical for
/// any thread count.
///
/// # Panics
/// If `job.seed` is the golden-set seed, `job.trick` is not 2..=5, a decision trick is not
/// 1..`job.trick`, the cap is 0 or the schedule is empty.
pub fn harvest_rows(
    job: &HarvestJob,
    starts: &[Vec<HandStartState>; 4],
    leaf: &SharedLeaf,
    progress: Option<&AtomicUsize>,
) -> (Vec<(Row, RowInfo)>, Vec<HarvestDecision>) {
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
    assert!(job.cap >= 1, "the cap per decision is at least 1");
    assert!(!job.schedule.is_empty() && job.schedule.iter().all(|&p| p < 8));
    struct Done {
        /// Finished decisions' facts and rows (not their positions: a run holds only its rows).
        outs: BTreeMap<u64, (HarvestDecision, Vec<(Row, RowInfo)>)>,
        /// Decisions `0..prefix` are done, giving `prefix_rows` rows.
        prefix: u64,
        prefix_rows: u64,
    }
    let state = Mutex::new(Done {
        outs: BTreeMap::new(),
        prefix: 0,
        prefix_rows: 0,
    });
    let next = std::sync::atomic::AtomicU64::new(0);
    let source = BeliefSource::new();
    let work = || {
        let mut mem = SolveMem::new();
        loop {
            // Decisions are taken in increasing order: once the finished prefix has the rows,
            // every decision the shard needs is done (a later one may still be running).
            if state.lock().unwrap_or_else(|e| e.into_inner()).prefix_rows >= job.count {
                break;
            }
            let j = next.fetch_add(1, Ordering::Relaxed);
            let out = harvest_decision(job, j, starts, leaf, &source, &mut mem);
            let got = out.rows.len();
            let mut d = state.lock().unwrap_or_else(|e| e.into_inner());
            d.outs.insert(j, (out.info, out.rows));
            loop {
                let p = d.prefix;
                let Some(r) = d.outs.get(&p).map(|o| o.1.len() as u64) else {
                    break;
                };
                d.prefix += 1;
                d.prefix_rows += r;
            }
            drop(d);
            if let Some(p) = progress {
                p.fetch_add(got, Ordering::Relaxed);
            }
        }
    };
    let threads = job.threads.max(1);
    if job.count > 0 {
        if threads == 1 {
            work();
        } else {
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(work);
                }
            });
        }
    }
    let d = state.into_inner().unwrap_or_else(|e| e.into_inner());
    let mut rows = Vec::with_capacity(job.count as usize);
    let mut decisions = Vec::new();
    for (j, (mut info, mut got)) in d.outs {
        if rows.len() as u64 >= job.count || j >= d.prefix {
            break;
        }
        let keep = ((job.count - rows.len() as u64) as usize).min(got.len());
        got.truncate(keep);
        info.rows = keep as u32;
        rows.extend(got);
        decisions.push(info);
    }
    debug_assert_eq!(rows.len() as u64, job.count);
    (rows, decisions)
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
                let out = harvest_decision(&job, j, &starts, &*l, &source, &mut mem);
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
                let again = harvest_decision(&job, j, &starts, &*l, &source, &mut SolveMem::new());
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
        let (a, da) = harvest_rows(&job(1), &starts, &*l, None);
        let (b, db) = harvest_rows(&job(4), &starts, &*l, None);
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
            let out = harvest_decision(&job(1), d.id, &starts, &*l, &src, &mut mem);
            want.extend(out.rows.into_iter().take(d.rows as usize).map(|r| r.0));
        }
        assert_eq!(want, ra);
        let empty = HarvestJob { count: 0, ..job(2) };
        assert!(harvest_rows(&empty, &starts, &*l, None).0.is_empty());
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
        harvest_rows(&job, &starts, &*l, None);
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
}
