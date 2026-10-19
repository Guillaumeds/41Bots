//! The hybrid yardstick check (docs/M3-HYBRID-SPEC.md §3, "proven 0.01"): on a decision set of
//! the trick before the cut, every candidate is valued exactly and by the hybrid H_k (exact
//! search to the start of trick `k`, the oracle network O_k there) on the *same* worlds, and the
//! three measures of §3 are reported:
//!
//! 1. **Decision loss** (the standard): per decision, the hybrid's choice is the candidate with
//!    the best H_k mean over the decision's worlds (ties to the first candidate, as PIMC); its
//!    loss is the exact mean of the exact-best candidate minus the exact mean of the hybrid's
//!    choice, in the decider's team's money chance. Reported as the mean over decisions and its
//!    95% upper bound: the upper end of the 95% bootstrap interval, the 97.5th percentile of
//!    1,000 bootstrap means (decisions resampled with replacement, `Rng::derive(boot_seed, …)`,
//!    seed 1; the convention of `crate::curves`, and the stricter reading), overall and per
//!    position band (1–2, 3–4, 5–8; each band resampled on its own). Pass: upper bound ≤ 0.01%
//!    everywhere.
//! 2. **Value error** (diagnostic): per candidate, world and team, H_k minus exact: for each
//!    team relative to the decider (its own, the next, the opposite, the previous), the mean
//!    (bias), and over all four teams together the SD, the 99.9th percentile of the absolute
//!    error and the largest (other seats choose by their own team's entry inside the search, so
//!    every team's error matters); overall, per band and per position. Every team's bias must be
//!    within ±0.01%.
//! 3. **Agreement** (diagnostic): the share of decisions whose hybrid choice is the exact choice
//!    or ties it exactly (zero loss).
//!
//! **Worlds** are regenerated exactly as the yardstick draws them (`crate::measure`: decision seed
//! from (seed, label, id), world `w` from it), never stored twice. **Exact values** are computed
//! by the current solver, or read from an existing worlds file (`--worlds`), in which case a
//! sample of worlds (`verify_pct`, default 1%) is recomputed and must be byte-identical first
//! (spec §4); any mismatch stops the check.
//!
//! **Slices** ([`check_worlds_with`] with a predicate file, [`slices`]; docs/m3-hybrid/
//! audit-loop.md §3 item 2): the hybrid's own search in each world records the distinct frontier
//! positions it reaches and their groups (`crate::audit::GroupRecorder`, which never changes a
//! value); a decision is in a group when at least a share (default a half) of the positions its
//! worlds' searches reach are in it, and the §3 measures are reported per group and for the
//! decisions in no group.
//!
//! **Chosen worlds** ([`check_worlds`]): the same measures on any worlds of each decision, with
//! stored values where given, e.g. the exact trick-1 worlds earlier runs saved (`crate::saved`,
//! spec §3 "Trick-1 checks"), each set with the burn rule that valued it
//! ([`CheckJob::rules_robber_burn`]).
//!
//! **A silver reference** ([`check_worlds_ref`] with a [`Reference`]; docs/m3-hybrid/silver.md):
//! where no exact golden set is affordable (trick 1), the reference values are those of a deeper
//! hybrid H_ref (exact search to the start of trick `ref` > `cut`, its own network there), stored
//! in a worlds file of hybrid records (`crate::measure::HYBRID`) or recomputed here, on the same
//! worlds; everything else (the choices, the loss, the bootstrap, the bands, the verification of
//! stored values, byte for byte) is the golden-set check's. In the report the `exact_*` fields
//! of a decision hold its reference's values (`silver` says which). A decision may keep an exact
//! reference when one is stored for it and the caller allowed it (`DecisionWorlds::silver`
//! false): the anchor decisions that also measure the silver's own error.
//!
//! **Score and race groups** ([`strata`], [`race_groups`]): the §3 measures per stratum of the
//! hand's start state (S1–S4, the yardstick's strata) and per race class of the positions the
//! hybrid's own search hands its network (`nn::outcome::race_class_of`, the classes the label
//! shards count: [`RaceRecorder`], which never changes a value); a decision is in a race class
//! when at least a share (a half) of the distinct frontier positions its worlds' searches reach
//! are in it, else in "mixed".
//!
//! Every value is computed in fixed point; the result is byte-identical for any thread count.
//! Timings (milliseconds per world, nanoseconds per network evaluation) are reports only.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use engine::rng::Rng;
use equity::{MoneyChance, ONE};
use nn::predicate::Predicates;
use serde::Serialize;
use solver::frontier::{CanonPos, Frontier};
use solver::{FrontierCut, MatchCtx, SolveMem};

use crate::audit::{GroupRecorder, Reach};
use crate::measure::{DecisionSpec, WorldRows, measure_seed, stratum_of};
use crate::pimc::{
    Candidate, Ctx, SearchConfig, WorldSolve, aggregate, best_index, candidates, exact_world,
    world_match,
};

/// Seed label of the verification sample.
const VERIFY: u64 = 0x5E21;
/// Seed label of the bootstrap.
const BOOT: u64 = 0xB007;
/// The standard: 0.01% money chance (a fraction), the default of `hybrid-check --standard`.
pub const STANDARD: f64 = 1e-4;

/// The standard of `--standard PCT`, a percentage of money chance ("0.01", "0.005"), as a
/// fraction: the decimal text shifted two places and parsed once (correctly rounded), so
/// "0.01" is exactly [`STANDARD`].
pub fn parse_standard(pct: &str) -> Result<f64, String> {
    let t = pct.trim();
    let ok = !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit() || b == b'.');
    let x: f64 = if ok {
        format!("{t}e-2").parse().unwrap_or(f64::NAN)
    } else {
        f64::NAN
    };
    if x.is_finite() && x > 0.0 && x <= 0.01 {
        Ok(x)
    } else {
        Err(format!(
            "--standard {pct:?}: a percentage of money chance above 0 and at most 1, e.g. 0.01"
        ))
    }
}

/// Judges a summary by `standard` (a fraction): the decision-loss bound and every team's bias.
pub fn judge(s: &mut Summary, standard: f64) {
    s.loss_pass = s.loss_upper95 <= standard;
    s.bias_pass = s.bias.iter().all(|b| b.abs() <= standard);
}

/// One check run.
#[derive(Clone, Debug)]
pub struct CheckJob {
    /// The frontier trick.
    pub cut: u8,
    /// Master seed of the worlds (as the yardstick: 1).
    pub seed: u64,
    /// World label (1 = the yardstick's).
    pub label: u64,
    /// Worlds per decision.
    pub to: u64,
    /// Worker threads (never changes a value).
    pub threads: usize,
    /// Transposition-table megabytes per hybrid solve.
    pub tt_mb: usize,
    /// Percentage of stored worlds recomputed and compared (with stored exact values).
    pub verify_pct: f64,
    /// Recompute every exact value even when stored ones are given.
    pub recompute: bool,
    /// Another seat's rob burn later in trick 1 is the `rules-v1` burn, in the exact solves and
    /// the hybrid's alike (`SearchConfig::rules_robber_burn`): the rule of worlds valued before
    /// values version 2 (`crate::saved`).
    pub rules_robber_burn: bool,
    /// Threads inside each exact solve this run makes (`solve_fast_par`, the same answer as one
    /// thread); with more than one, the exact solves run first, `threads / exact_solve_threads`
    /// at a time. Trick-1 worlds need this and big tables to be recomputed in hours, not days.
    pub exact_solve_threads: usize,
    /// Transposition-table megabytes per exact solve.
    pub exact_tt_mb: usize,
    /// The optimizer's-curse test's noise at every leaf of the hybrid's search (`crate::curse`;
    /// `None`: none, the ordinary check). The noise never touches the reference's values.
    pub noise: Option<crate::curse::NoiseSpec>,
    /// Exact leaves (`crate::exact_leaves`): the hybrid's frontier positions this policy picks
    /// are solved exactly instead of being valued by the network (`None`: none, the hybrid as
    /// before, byte for byte). The reference's values are never touched.
    pub exact_leaves: Option<crate::exact_leaves::LeafPolicy>,
}

impl CheckJob {
    /// Whether world `w` of decision `id`, with stored exact values, is in the verification
    /// sample (`verify_pct` percent, drawn by `(seed, id, w)`).
    pub fn verify_sampled(&self, id: u64, w: u64) -> bool {
        Rng::derive(self.seed, &[VERIFY, id, w]).bounded(1_000_000)
            < (self.verify_pct * 10_000.0) as u64
    }

    /// The settings of the exact solves (`exact`) or of the hybrid's, with the job's burn rule.
    pub fn config(&self, exact: bool) -> SearchConfig {
        SearchConfig {
            tt_mb: if exact { self.exact_tt_mb } else { self.tt_mb },
            solve_threads: if exact {
                self.exact_solve_threads.max(1)
            } else {
                1
            },
            rules_robber_burn: self.rules_robber_burn,
            ..SearchConfig::default()
        }
    }
}

/// A frontier evaluator that times its evaluations (reports only).
struct Timed<'a> {
    inner: &'a dyn Frontier,
    ns: AtomicU64,
    n: AtomicU64,
}

impl Frontier for Timed<'_> {
    fn value(&self, pos: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        let t = Instant::now();
        let v = self.inner.value(pos, ctx);
        self.ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.n.fetch_add(1, Ordering::Relaxed);
        v
    }

    fn prefetch_depth(&self) -> u8 {
        self.inner.prefetch_depth()
    }

    /// The batch's time counts as evaluation time; the positions are counted when asked for.
    fn prefetch(&self, pos: &[CanonPos], ctx: &MatchCtx) {
        let t = Instant::now();
        self.inner.prefetch(pos, ctx);
        self.ns
            .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// One world to check: its number, the decision seed it is drawn from
/// ([`measure_seed`]`(seed, label, id)`) and its stored exact values, if any.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldTask {
    /// World number.
    pub w: u64,
    /// The decision seed the world is drawn from.
    pub dseed: u64,
    /// Stored exact values (candidate order).
    pub stored: Option<Vec<MoneyChance>>,
    /// How long the world is expected to take, in any unit common to the check's worlds (for
    /// example the solver nodes its saved exact solve took; 0 = unknown): the longest go first.
    /// It never changes a value.
    pub cost: u64,
}

/// The worlds of one decision to check (any world numbers, in the order given).
#[derive(Clone, Debug)]
pub struct DecisionWorlds<'a> {
    /// The decision.
    pub d: &'a DecisionSpec,
    /// Its worlds.
    pub worlds: Vec<WorldTask>,
    /// Its reference is the check's silver [`Reference`] (its stored values are a silver
    /// reference's, and recomputed ones are the reference hybrid's); `false`: exact.
    pub silver: bool,
}

/// A silver reference (module notes; docs/m3-hybrid/silver.md): the deeper hybrid H_`cut` with
/// network `net` at the start of trick `cut`, whose values stand in for exact ones. With the
/// same cut as the check's own it is the optimizer's-curse test's truth (`crate::curse`): the
/// same search with the network's noise-free values, against the player's noisy ones.
#[derive(Clone, Copy)]
pub struct Reference<'a> {
    /// The reference's frontier trick (after the check's own).
    pub cut: u8,
    /// Its network.
    pub net: &'a dyn Frontier,
}

/// A frontier evaluator that values every position exactly as `inner` does and records the
/// distinct canonical forms the search reaches there by race class (`nn::outcome::race_class_of`:
/// a [`Reach`] of [`nn::outcome::N_RACE_CLASSES`] groups, class `c` in `in_group[c]`). One per
/// search, as [`GroupRecorder`].
pub struct RaceRecorder<'a> {
    inner: &'a dyn Frontier,
    leaf: &'a (dyn solver::Leaf + Sync),
    state: Mutex<(std::collections::BTreeSet<CanonPos>, Reach)>,
}

impl<'a> RaceRecorder<'a> {
    /// A recorder of race classes around `inner`; `leaf` (the equity table) gives the outcomes'
    /// money and so which teams can still win the game or the match.
    pub fn new(inner: &'a dyn Frontier, leaf: &'a (dyn solver::Leaf + Sync)) -> Self {
        RaceRecorder {
            inner,
            leaf,
            state: Mutex::new((
                std::collections::BTreeSet::new(),
                Reach::new(nn::outcome::N_RACE_CLASSES),
            )),
        }
    }

    /// What the search reached, by race class.
    pub fn into_reach(self) -> Reach {
        self.state.into_inner().unwrap_or_else(|e| e.into_inner()).1
    }

    fn note(&self, canon: &CanonPos, ctx: &MatchCtx) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if st.0.insert(*canon) {
            let c = nn::outcome::race_class_of(canon, ctx, self.leaf);
            st.1.reached += 1;
            st.1.in_group[usize::from(c)] += 1;
        }
    }
}

impl Frontier for RaceRecorder<'_> {
    fn value(&self, canon: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        self.note(canon, ctx);
        self.inner.value(canon, ctx)
    }

    fn value_at(&self, pos: &solver::Position, canon: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        self.note(canon, ctx);
        self.inner.value_at(pos, canon, ctx)
    }

    fn prefetch_depth(&self) -> u8 {
        self.inner.prefetch_depth()
    }

    /// Passed on; nothing is recorded (the search records what it reaches through `value_at`).
    fn prefetch(&self, pos: &[CanonPos], ctx: &MatchCtx) {
        self.inner.prefetch(pos, ctx);
    }
}

/// One world of one decision: exact and hybrid values of every candidate.
#[derive(Clone, Debug)]
pub struct WorldCheck {
    /// World number.
    pub w: u64,
    /// Exact values (stored or computed here), candidate order.
    pub exact: Vec<MoneyChance>,
    /// H_k's values.
    pub hybrid: Vec<MoneyChance>,
    /// The exact values were computed by this run (no stored values, `recompute`, or the
    /// verification sample).
    pub exact_solved: bool,
    /// Stored exact values were compared with recomputed ones in this world: `Some(equal)`.
    pub verified: Option<bool>,
    /// Exact solve time (µs; 0 unless solved here).
    pub exact_us: u64,
    /// Exact solver nodes (0 unless solved here).
    pub exact_nodes: u64,
    /// Hybrid solve time (µs).
    pub hybrid_us: u64,
    /// Hybrid solver nodes.
    pub hybrid_nodes: u64,
    /// The hybrid's solver nodes by trick (1–5).
    pub hybrid_nodes_by_trick: [u64; 5],
    /// The hybrid's network evaluations.
    pub frontier: u64,
    /// The largest transposition table the hybrid's solves used (bytes).
    pub hybrid_tt_bytes: u64,
    /// The seat other than the decider that robs later in trick 1 in this world (the burn the
    /// burn rule decides), if any.
    pub robber: Option<u8>,
    /// With slices: the distinct frontier positions the hybrid's search reached, by group.
    pub reach: Option<Reach>,
    /// With race groups: the distinct frontier positions the hybrid's search reached, by race
    /// class ([`RaceRecorder`]).
    pub race: Option<Reach>,
    /// With exact leaves: the search's distinct leaves by race class, those solved exactly, and
    /// the leaf solves' nodes and time.
    pub leaves: Option<crate::exact_leaves::LeafStats>,
}

/// The check of one decision.
#[derive(Clone, Debug, Serialize)]
pub struct DecisionCheck {
    /// Decision id.
    pub id: u64,
    /// Position in the trick (1 = leader … 8).
    pub position: u8,
    /// Stratum (1–4).
    pub stratum: u8,
    /// The decider's team.
    pub team: u8,
    /// Candidates.
    pub cands: usize,
    /// Worlds.
    pub worlds: u64,
    /// Exact means' choice and the hybrid means' choice (candidate indices).
    pub exact_choice: usize,
    /// The hybrid's choice.
    pub hybrid_choice: usize,
    /// Loss of the hybrid's choice in exact values, decider's team, units of 2^-32.
    pub loss: u64,
    /// The hybrid's choice is exact's or ties it exactly.
    pub agree: bool,
    /// The decider's own team's errors (hybrid minus exact, every world and candidate) summed,
    /// units of 2^-32: the own-team bias is this over `cands × worlds` (the paired comparison
    /// of two checks, [`paired`], reads it).
    pub own_error_sum: i64,
    /// Per world, candidate and team relative to the decider (`(team − decider) mod 4`, the
    /// innermost index): hybrid minus exact, units of 2^-32.
    #[serde(skip)]
    pub errors: Vec<i64>,
    /// Worlds whose exact values this run computed (all of them unless stored values were
    /// given; then the verification sample).
    pub exact_worlds: u64,
    /// Exact solve time over those worlds (µs).
    pub exact_us: u64,
    /// Hybrid solve time over the decision's worlds (µs).
    pub hybrid_us: u64,
    /// Solver nodes, exact and hybrid.
    pub exact_nodes: u64,
    /// Solver nodes of the hybrid solves.
    pub hybrid_nodes: u64,
    /// Network evaluations.
    pub frontier: u64,
    /// With slices: the distinct frontier positions the hybrid's searches reached (per world,
    /// summed), by group.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reach: Option<Reach>,
    /// With race groups: the distinct frontier positions the hybrid's searches reached (per
    /// world, summed), by race class ([`RaceRecorder`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub race: Option<Reach>,
    /// With exact leaves: its worlds' leaves by race class, those solved exactly, the leaf
    /// solves' nodes and time (summed over worlds).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exact_leaves: Option<crate::exact_leaves::LeafStats>,
    /// The reference values (the `exact_*` fields) are a silver reference's ([`Reference`]),
    /// not exact.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub silver: bool,
    /// Every world's values and counters, in the order given.
    #[serde(skip)]
    pub world_checks: Vec<WorldCheck>,
}

/// The result of a check.
#[derive(Clone, Debug)]
pub struct CheckRun {
    /// Per decision, in the order given.
    pub decisions: Vec<DecisionCheck>,
    /// Stored worlds recomputed for verification, and how many differed.
    pub verified: (u64, u64),
    /// Time and count of network evaluations (ns, n).
    pub net: (u64, u64),
    /// Recomputed worlds whose stored exact values differ (decision, world). When there is any,
    /// the hybrid was not run and `decisions` is empty: the stored values are not this solver's.
    pub mismatched: Vec<(u64, u64)>,
    /// With [`CheckJob::noise`]: the realised noise (`crate::curse::NoiseStats`).
    pub noise: Option<crate::curse::NoiseStats>,
    /// With [`CheckJob::exact_leaves`]: the exact leaves over every world
    /// (`crate::exact_leaves::LeafStats`).
    pub exact_leaves: Option<crate::exact_leaves::LeafStats>,
}

/// Live counters of a running check, for progress reports (they never change a value).
#[derive(Debug, Default)]
pub struct CheckProgress {
    /// Exact solves this run makes, and those finished.
    pub exact_total: AtomicU64,
    /// See `exact_total`.
    pub exact_done: AtomicU64,
    /// Hybrid worlds, and those finished.
    pub hybrid_total: AtomicU64,
    /// See `hybrid_total`.
    pub hybrid_done: AtomicU64,
    /// Each hybrid world as it finishes, in finishing order: (decision, world, µs, solver
    /// nodes, network evaluations). Progress reports drain it; a stopped run keeps what finished.
    pub finished: Mutex<Vec<[u64; 5]>>,
}

/// The stored reference records a check takes from worlds files (`rows`, already joined by
/// `crate::measure::read_worlds`; `files` names them in messages), by decision id: those at
/// `trick` of the `selected` decisions. Against exact values (`reference` `None`) only exact
/// records are taken, and a silver reference's (hybrid) record is refused: a silver file is never
/// an exact golden set. Against a silver reference (its fingerprint), only hybrid records of
/// exactly that frontier and network are taken; another network's are refused, and so are exact
/// records unless `allow_exact` (they then stand in for the silver values of their decisions, an
/// exact record being the better reference: the anchor decisions). Other methods (rollout) are
/// ignored, as before.
pub fn stored_reference(
    rows: Vec<WorldRows>,
    trick: u8,
    selected: &dyn Fn(u64) -> bool,
    reference: Option<&crate::measure::FrontierFp>,
    allow_exact: bool,
    files: &str,
) -> Result<BTreeMap<u64, WorldRows>, String> {
    use crate::measure::HYBRID;
    let mut exact: BTreeMap<u64, WorldRows> = BTreeMap::new();
    let mut hybrid: BTreeMap<u64, WorldRows> = BTreeMap::new();
    for r in rows
        .into_iter()
        .filter(|r| r.trick == trick && selected(r.id))
    {
        if r.method == "exact" {
            exact.insert(r.id, r);
        } else if r.method == HYBRID {
            hybrid.insert(r.id, r);
        }
    }
    let frontier = |r: &WorldRows| r.fp.as_ref().and_then(|f| f.frontier.clone());
    let what = |r: &WorldRows| {
        frontier(r).map_or("a hybrid record without a frontier".into(), |f| {
            format!("a silver reference, {}", f.describe())
        })
    };
    match reference {
        None => {
            if let Some(r) = hybrid.values().next() {
                return Err(format!(
                    "{files}: decision {} holds {} (`{HYBRID}` records), not exact values: a \
                     silver file is never an exact golden set. Check against it with --reference \
                     silver --reference-net <its network>",
                    r.id,
                    what(r)
                ));
            }
            Ok(exact)
        }
        Some(fp) => {
            if let Some(r) = hybrid.values().find(|r| frontier(r).as_ref() != Some(fp)) {
                return Err(format!(
                    "{files}: decision {} holds {}, not this check's reference {}",
                    r.id,
                    what(r),
                    fp.describe()
                ));
            }
            if !allow_exact && let Some(id) = exact.keys().next() {
                return Err(format!(
                    "{files}: decision {id} holds exact values, not the silver reference {}: pass \
                     --allow-exact to let exact records stand in for it (the anchor decisions; \
                     the report counts them), or check against exact values (--reference exact)",
                    fp.describe()
                ));
            }
            let mut all = hybrid;
            all.extend(exact);
            Ok(all)
        }
    }
}

/// Run the check on `ds` (decisions before the cut's trick), worlds `0..job.to` of each.
/// `stored`: exact worlds records by decision id (from an existing worlds file), used instead of
/// recomputing (after the verification sample) unless `job.recompute`.
pub fn run_check(
    ds: &[DecisionSpec],
    stored: &BTreeMap<u64, WorldRows>,
    job: &CheckJob,
    ctx: &Ctx,
    net: &dyn Frontier,
) -> Result<CheckRun, String> {
    run_check_slices(ds, stored, job, ctx, net, None)
}

/// [`run_check`], recording each world's reach in the groups of `slices` when given (module
/// notes, [`slices`]).
pub fn run_check_slices(
    ds: &[DecisionSpec],
    stored: &BTreeMap<u64, WorldRows>,
    job: &CheckJob,
    ctx: &Ctx,
    net: &dyn Frontier,
    slices: Option<&Predicates>,
) -> Result<CheckRun, String> {
    run_check_ref(ds, stored, job, ctx, net, slices, None, false, None)
}

/// [`run_check_slices`] against a silver [`Reference`] when given (module notes): the stored
/// records are then the reference's own hybrid records (`crate::measure::HYBRID` with its
/// frontier and network: checked here), or exact ones standing in for it (each such decision keeps
/// an exact reference: [`DecisionWorlds::silver`]); without stored values the reference's values
/// are recomputed. `race`: record the race classes the hybrid's searches reach ([`race_groups`]).
/// `progress` receives live counts.
#[allow(clippy::too_many_arguments)]
pub fn run_check_ref(
    ds: &[DecisionSpec],
    stored: &BTreeMap<u64, WorldRows>,
    job: &CheckJob,
    ctx: &Ctx,
    net: &dyn Frontier,
    slices: Option<&Predicates>,
    reference: Option<(Reference, &crate::measure::FrontierFp)>,
    race: bool,
    progress: Option<&CheckProgress>,
) -> Result<CheckRun, String> {
    let mut plan = Vec::new();
    for d in ds {
        let rec = stored.get(&d.id);
        // Which reference the decision's values are (and, when stored, that they are its).
        let silver = match (rec, &reference) {
            (Some(r), _) if r.method == "exact" => {
                if r.fp.as_ref().is_some_and(|f| f.frontier.is_some()) {
                    return Err(format!(
                        "decision {}: an exact record with a frontier in its fingerprint",
                        d.id
                    ));
                }
                false
            }
            (Some(r), Some((_, want))) if r.method == crate::measure::HYBRID => {
                let have = r.fp.as_ref().and_then(|f| f.frontier.as_ref());
                if have != Some(*want) {
                    return Err(format!(
                        "decision {}: its stored silver values are {}, not the reference {} of \
                         this check",
                        d.id,
                        have.map_or("without a frontier".into(), |f| f.describe()),
                        want.describe()
                    ));
                }
                true
            }
            (Some(r), None) => {
                return Err(format!(
                    "decision {}: the stored record is `{}`, not exact: a check against exact \
                     values takes exact records only{}",
                    d.id,
                    r.method,
                    if r.method == crate::measure::HYBRID {
                        " (a hybrid record is a silver reference: check against it as one)"
                    } else {
                        ""
                    }
                ));
            }
            (Some(r), Some(_)) => {
                return Err(format!(
                    "decision {}: the stored record is `{}`, neither exact nor the silver \
                     reference's",
                    d.id, r.method
                ));
            }
            (None, r) => r.is_some(),
        };
        let seed = rec
            .and_then(|r| r.fp.as_ref())
            .map_or(job.seed, |fp| fp.seed);
        let label = rec.map_or(job.label, |r| r.label);
        if let Some(r) = rec {
            if d.trick >= 1 && d.trick < job.cut {
                let (_, view, legal) = d.replay();
                if r.cands != candidates(&view, &legal).1 {
                    return Err(format!("decision {}: stored candidates differ", d.id));
                }
            }
            if r.from != 0 || (r.rows.len() as u64) < job.to {
                return Err(format!(
                    "decision {}: the stored record has worlds {}..{}, not 0..{}",
                    d.id,
                    r.from,
                    r.from + r.rows.len() as u64,
                    job.to
                ));
            }
        }
        let dseed = measure_seed(seed, label, d.id);
        let worlds = (0..job.to)
            .map(|w| {
                Ok(WorldTask {
                    w,
                    dseed,
                    cost: 0,
                    stored: rec
                        .map(|r| {
                            r.rows[w as usize]
                                .iter()
                                .map(|&v| {
                                    MoneyChance::from_raw(v).ok_or_else(|| {
                                        format!(
                                            "decision {} world {w}: stored values do not sum \
                                             to one",
                                            d.id
                                        )
                                    })
                                })
                                .collect::<Result<Vec<_>, String>>()
                        })
                        .transpose()?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        plan.push(DecisionWorlds { d, worlds, silver });
    }
    check_worlds_ref(
        &plan,
        job,
        ctx,
        net,
        progress,
        slices,
        reference.map(|(r, _)| r),
        race,
    )
}

/// Runs `f(i, mem)` for every `i` of `order` on `threads` threads (a shared work queue in that
/// order, each thread with its own solver memory); the results by index (`None` for an index not
/// in `order`), `n` of them.
fn pool<T: Send + Clone>(
    threads: usize,
    order: &[usize],
    f: &(dyn Fn(usize, &mut SolveMem) -> T + Sync),
    n: usize,
) -> Vec<Option<T>> {
    let slots: Mutex<Vec<Option<T>>> = Mutex::new(vec![None; n]);
    let next = AtomicUsize::new(0);
    let work = || {
        let mut mem = SolveMem::new();
        loop {
            let q = next.fetch_add(1, Ordering::Relaxed);
            let Some(&i) = order.get(q) else { break };
            let out = f(i, &mut mem);
            slots.lock().unwrap_or_else(|e| e.into_inner())[i] = Some(out);
        }
    };
    let threads = threads.max(1).min(order.len().max(1));
    if threads == 1 {
        work();
    } else {
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(work);
            }
        });
    }
    slots.into_inner().unwrap_or_else(|e| e.into_inner())
}

/// Run the check on chosen worlds of each decision (any world numbers; stored exact values where
/// given). The same measures as [`run_check`], which is this with worlds `0..to`. Exact values
/// are computed for worlds without stored ones, for every world with `job.recompute`, and for
/// the verification sample (`job.verify_pct` of the stored worlds, drawn by `(seed, id, w)`);
/// those solves run first, then the hybrid's, one world per thread, the longest expected first
/// (the worlds' cost hints, then the earliest positions in the trick). If any recomputed world's
/// stored values differ, the hybrid is not run: the result lists those worlds (`mismatched`) and
/// has no decisions. `progress` receives live counts.
pub fn check_worlds(
    plan: &[DecisionWorlds],
    job: &CheckJob,
    ctx: &Ctx,
    net: &dyn Frontier,
    progress: Option<&CheckProgress>,
) -> Result<CheckRun, String> {
    check_worlds_with(plan, job, ctx, net, progress, None)
}

/// [`check_worlds`], recording each hybrid search's reach in the groups of `slices` when given
/// (`crate::audit::GroupRecorder` around the network: the same values).
pub fn check_worlds_with(
    plan: &[DecisionWorlds],
    job: &CheckJob,
    ctx: &Ctx,
    net: &dyn Frontier,
    progress: Option<&CheckProgress>,
    slices: Option<&Predicates>,
) -> Result<CheckRun, String> {
    check_worlds_ref(plan, job, ctx, net, progress, slices, None, false)
}

/// [`check_worlds_with`] against a silver [`Reference`] for the decisions marked so
/// ([`DecisionWorlds::silver`]: their stored values are the reference's, and recomputed ones are
/// the reference hybrid's, verification sample included), and recording the race classes the
/// hybrid's searches reach when `race` ([`RaceRecorder`] around the network: the same values).
#[allow(clippy::too_many_arguments)]
pub fn check_worlds_ref(
    plan: &[DecisionWorlds],
    job: &CheckJob,
    ctx: &Ctx,
    net: &dyn Frontier,
    progress: Option<&CheckProgress>,
    slices: Option<&Predicates>,
    reference: Option<Reference>,
    race: bool,
) -> Result<CheckRun, String> {
    struct Prep {
        view: engine::view::SeatView,
        cands: Vec<Candidate>,
    }
    // A silver reference looks deeper than the hybrid it checks; the curse test's truth (a
    // check with noise) is the same search without the noise, at the same cut.
    if let Some(r) = &reference
        && !((job.cut < r.cut || (job.cut == r.cut && job.noise.is_some()))
            && r.cut <= engine::game::TRICKS_PER_HAND)
    {
        return Err(format!(
            "a silver reference H_{} must look deeper than the hybrid it checks, H_{} (its \
             frontier after the check's, at most trick 5; the same cut only as the truth of the \
             optimizer's-curse test, a check with noise)",
            r.cut, job.cut
        ));
    }
    let mut preps = Vec::new();
    for p in plan {
        let d = p.d;
        if d.trick >= job.cut || d.trick == 0 {
            return Err(format!(
                "decision {} is at trick {}: the hybrid with a frontier at trick {} is defined for \
                 decisions at tricks 1..{}",
                d.id, d.trick, job.cut, job.cut
            ));
        }
        if p.silver && reference.is_none() {
            return Err(format!(
                "decision {} is to be checked against a silver reference, but none was given",
                d.id
            ));
        }
        let (_, view, legal) = d.replay();
        let (_, cands) = candidates(&view, &legal);
        for t in &p.worlds {
            if t.stored.as_ref().is_some_and(|s| s.len() != cands.len()) {
                return Err(format!(
                    "decision {} world {}: {} stored values for {} candidates",
                    d.id,
                    t.w,
                    t.stored.as_ref().map_or(0, Vec::len),
                    cands.len()
                ));
            }
        }
        preps.push(Prep { view, cands });
    }
    let worlds: Vec<_> = preps
        .iter()
        .map(|p| ctx.source.prepare(&p.view, ctx.rules))
        .collect();
    let timed = Timed {
        inner: net,
        ns: AtomicU64::new(0),
        n: AtomicU64::new(0),
    };
    let cut = FrontierCut::new(job.cut, &timed);
    // Every (decision, world), and the queue order: the longest expected first (by the worlds'
    // cost hints, then the earliest trick and position).
    let tasks: Vec<(usize, usize)> = plan
        .iter()
        .enumerate()
        .flat_map(|(k, p)| (0..p.worlds.len()).map(move |j| (k, j)))
        .collect();
    let mut order: Vec<usize> = (0..tasks.len()).collect();
    order.sort_by_key(|&i| {
        let (k, j) = tasks[i];
        let d = plan[k].d;
        (
            std::cmp::Reverse(plan[k].worlds[j].cost),
            d.trick,
            d.position,
            i,
        )
    });
    let verify_of = |i: usize| -> bool {
        let (k, j) = tasks[i];
        let t = &plan[k].worlds[j];
        t.stored.is_some() && !job.recompute && job.verify_sampled(plan[k].d.id, t.w)
    };
    let exact_needed = |i: usize| -> bool {
        let (k, j) = tasks[i];
        plan[k].worlds[j].stored.is_none() || job.recompute || verify_of(i)
    };
    let world_of = |i: usize| {
        let (k, j) = tasks[i];
        world_match(
            &preps[k].view,
            &*worlds[k],
            plan[k].worlds[j].w,
            plan[k].worlds[j].dseed,
            ctx.rules,
        )
    };
    // The exact solves.
    let ex_order: Vec<usize> = order.iter().copied().filter(|&i| exact_needed(i)).collect();
    if let Some(p) = progress {
        p.exact_total
            .store(ex_order.len() as u64, Ordering::Relaxed);
        p.hybrid_total.store(tasks.len() as u64, Ordering::Relaxed);
    }
    let cfg_ex = job.config(true);
    let solve_exact = |i: usize, mem: &mut SolveMem| -> (Vec<MoneyChance>, u64, u64) {
        let m = world_of(i);
        let mut run = WorldSolve::default();
        let t = Instant::now();
        // The decision's reference: exact, or the silver reference's hybrid.
        let ref_cut = reference
            .filter(|_| plan[tasks[i].0].silver)
            .map(|r| FrontierCut::new(r.cut, r.net));
        let e = exact_world(
            &m,
            &preps[tasks[i].0].cands,
            ctx,
            &cfg_ex,
            None,
            mem,
            &mut run,
            ref_cut,
        )
        .expect("nothing stops a solve here");
        let us = t.elapsed().as_micros() as u64;
        if let Some(p) = progress {
            p.exact_done.fetch_add(1, Ordering::Relaxed);
        }
        (e, us, run.stats.nodes)
    };
    let ex_threads = (job.threads / job.exact_solve_threads.max(1)).max(1);
    let exact = pool(ex_threads, &ex_order, &solve_exact, tasks.len());
    // Stored values must be this solver's before anything is compared with them.
    let mut verified = (0u64, 0u64);
    let mut mismatched = Vec::new();
    for &i in &ex_order {
        let (k, j) = tasks[i];
        let t = &plan[k].worlds[j];
        if let (Some(s), Some((e, _, _)), false) = (&t.stored, &exact[i], job.recompute) {
            verified.0 += 1;
            if s != e {
                verified.1 += 1;
                mismatched.push((plan[k].d.id, t.w));
            }
        }
    }
    if !mismatched.is_empty() {
        return Ok(CheckRun {
            decisions: Vec::new(),
            verified,
            net: (0, 0),
            mismatched,
            noise: None,
            exact_leaves: None,
        });
    }
    // The hybrid's solves.
    let cfg_h = job.config(false);
    type HybridOut = (
        Vec<MoneyChance>,
        u64,
        WorldSolve,
        Option<Reach>,
        Option<Reach>,
        Option<crate::exact_leaves::LeafStats>,
    );
    let noise_total: Mutex<crate::curse::NoiseStats> = Mutex::default();
    let solve_hybrid = |i: usize, mem: &mut SolveMem| -> HybridOut {
        let m = world_of(i);
        let mut run = WorldSolve::default();
        let t = Instant::now();
        // The curse test's noise sits right on the network, under every recorder.
        let noisy = job.noise.as_ref().map(|spec| {
            let (k, j) = tasks[i];
            crate::curse::NoisyLeaf::new(&timed, ctx.leaf, spec, plan[k].d.id, plan[k].worlds[j].w)
        });
        let noised: &dyn Frontier = match &noisy {
            Some(n) => n,
            None => &timed,
        };
        // Exact leaves replace the network's value where the policy says, under every recorder
        // (which see the values the search uses).
        let exacts = job.exact_leaves.as_ref().map(|policy| {
            let k = tasks[i].0;
            crate::exact_leaves::ExactLeaves::new(
                noised,
                ctx.leaf,
                policy,
                engine::team_of(plan[k].d.seat),
                solver::SolveFamily::of_match(&m),
            )
        });
        let base: &dyn Frontier = match &exacts {
            Some(x) => x,
            None => noised,
        };
        let rec = slices.map(|p| GroupRecorder::new(base, p, ctx.leaf));
        let inner: &dyn Frontier = match &rec {
            Some(r) => r,
            None => base,
        };
        let races = race.then(|| RaceRecorder::new(inner, ctx.leaf));
        let h = exact_world(
            &m,
            &preps[tasks[i].0].cands,
            ctx,
            &cfg_h,
            None,
            mem,
            &mut run,
            Some(match &races {
                Some(r) => FrontierCut::new(job.cut, r),
                None if rec.is_some() || noisy.is_some() || exacts.is_some() => {
                    FrontierCut::new(job.cut, inner)
                }
                None => cut,
            }),
        )
        .expect("nothing stops a solve here");
        let race_reach = races.map(RaceRecorder::into_reach);
        let reach = rec.map(GroupRecorder::into_reach);
        let leaves = exacts.as_ref().map(crate::exact_leaves::ExactLeaves::stats);
        drop(exacts);
        if let Some(n) = noisy {
            noise_total
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .add(&n.into_stats());
        }
        let us = t.elapsed().as_micros() as u64;
        if let Some(p) = progress {
            let (k, j) = tasks[i];
            p.finished.lock().unwrap_or_else(|e| e.into_inner()).push([
                plan[k].d.id,
                plan[k].worlds[j].w,
                us,
                run.stats.nodes,
                run.stats.frontier,
            ]);
            p.hybrid_done.fetch_add(1, Ordering::Relaxed);
        }
        (h, us, run, reach, race_reach, leaves)
    };
    let hybrid = pool(job.threads, &order, &solve_hybrid, tasks.len());
    let mut outs: Vec<WorldCheck> = Vec::with_capacity(tasks.len());
    for (i, (hy, ex)) in hybrid.into_iter().zip(exact).enumerate() {
        let (h, hus, run, reach, race_reach, leaves) =
            hy.expect("every world valued by the hybrid");
        let (k, j) = tasks[i];
        let t = &plan[k].worlds[j];
        let (exact, exact_us, exact_nodes, verified) = match (ex, &t.stored) {
            (Some((e, us, nodes)), s) => {
                let ok = s.as_ref().filter(|_| !job.recompute).map(|s| *s == e);
                (e, us, nodes, ok)
            }
            (None, Some(s)) => (s.clone(), 0, 0, None),
            (None, None) => unreachable!("worlds without stored values are solved"),
        };
        outs.push(WorldCheck {
            w: t.w,
            exact,
            hybrid: h,
            exact_solved: exact_needed(i),
            verified,
            exact_us,
            exact_nodes,
            hybrid_us: hus,
            hybrid_nodes: run.stats.nodes,
            hybrid_nodes_by_trick: run.stats.nodes_by_trick,
            frontier: run.stats.frontier,
            hybrid_tt_bytes: run.stats.tt_bytes,
            robber: run.robber,
            reach,
            race: race_reach,
            leaves,
        });
    }
    let mut decisions = Vec::new();
    let mut at = 0;
    for (k, p) in plan.iter().enumerate() {
        let ws = &outs[at..at + p.worlds.len()];
        at += p.worlds.len();
        let cands = &preps[k].cands;
        let team = engine::team_of(p.d.seat);
        let ex: Vec<Vec<MoneyChance>> = ws.iter().map(|o| o.exact.clone()).collect();
        let hy: Vec<Vec<MoneyChance>> = ws.iter().map(|o| o.hybrid.clone()).collect();
        let (em, ec, _) = aggregate(&ex, cands.len(), team);
        let (hm, _, _) = aggregate(&hy, cands.len(), team);
        let hc = best_index(&hm, team);
        let loss = em[ec].get(team) - em[hc].get(team);
        let mut errors = Vec::with_capacity(4 * ws.len() * cands.len());
        let mut own_error_sum = 0i64;
        for o in ws {
            for (e, h) in o.exact.iter().zip(&o.hybrid) {
                for rel in 0..4u8 {
                    let t = (team + rel) % 4;
                    errors.push(h.get(t) as i64 - e.get(t) as i64);
                }
                own_error_sum += h.get(team) as i64 - e.get(team) as i64;
            }
        }
        decisions.push(DecisionCheck {
            id: p.d.id,
            position: p.d.position + 1,
            stratum: p.d.stratum,
            team,
            cands: cands.len(),
            worlds: ws.len() as u64,
            exact_choice: ec,
            hybrid_choice: hc,
            loss,
            agree: loss == 0,
            own_error_sum,
            errors,
            exact_worlds: ws.iter().filter(|o| o.exact_solved).count() as u64,
            exact_us: ws.iter().map(|o| o.exact_us).sum(),
            hybrid_us: ws.iter().map(|o| o.hybrid_us).sum(),
            exact_nodes: ws.iter().map(|o| o.exact_nodes).sum(),
            hybrid_nodes: ws.iter().map(|o| o.hybrid_nodes).sum(),
            frontier: ws.iter().map(|o| o.frontier).sum(),
            reach: slices.map(|p| {
                let mut r = Reach::new(p.len());
                for o in ws {
                    r.add(o.reach.as_ref().expect("every world recorded"));
                }
                r
            }),
            race: race.then(|| {
                let mut r = Reach::new(nn::outcome::N_RACE_CLASSES);
                for o in ws {
                    r.add(o.race.as_ref().expect("every world recorded"));
                }
                r
            }),
            exact_leaves: job.exact_leaves.as_ref().map(|_| {
                let mut st = crate::exact_leaves::LeafStats::default();
                for o in ws {
                    st.add(o.leaves.as_ref().expect("every world counted"));
                }
                st
            }),
            silver: p.silver && reference.is_some(),
            world_checks: ws.to_vec(),
        });
    }
    let exact_leaves = job.exact_leaves.as_ref().map(|_| {
        let mut st = crate::exact_leaves::LeafStats::default();
        for d in &decisions {
            st.add(d.exact_leaves.as_ref().expect("every decision counted"));
        }
        st
    });
    Ok(CheckRun {
        decisions,
        verified,
        net: (
            timed.ns.load(Ordering::Relaxed),
            timed.n.load(Ordering::Relaxed),
        ),
        mismatched,
        noise: job
            .noise
            .as_ref()
            .map(|_| noise_total.into_inner().unwrap_or_else(|e| e.into_inner())),
        exact_leaves,
    })
}

/// The §3 measures of a group of decisions.
#[derive(Clone, Debug, Serialize)]
pub struct Summary {
    /// Decisions.
    pub decisions: usize,
    /// Mean decision loss (money chance, a fraction).
    pub mean_loss: f64,
    /// Its 95% upper bound (bootstrap 97.5th percentile).
    pub loss_upper95: f64,
    /// Agreement: share of decisions with zero loss.
    pub agreement: f64,
    /// Values compared (candidate × world × team).
    pub values: usize,
    /// Value error: mean (bias) of each team, relative to the decider (its own team, the next,
    /// the opposite, the previous), as fractions.
    pub bias: [f64; 4],
    /// Value error over all four teams: standard deviation.
    pub sd: f64,
    /// Value error over all four teams: 99.9th percentile of the absolute error.
    pub p999_abs: f64,
    /// Value error over all four teams: the largest absolute error.
    pub max_abs: f64,
    /// The decider's team alone: SD and 99.9th percentile of the absolute error.
    pub decider_sd: f64,
    /// See `decider_sd`.
    pub decider_p999_abs: f64,
    /// Decision loss within the standard at the 95% upper bound.
    pub loss_pass: bool,
    /// Every team's bias within ±0.01% (or the check's `--standard`, [`judge`]).
    pub bias_pass: bool,
}

/// The §3 measures of `ds`; the bootstrap uses `Rng::derive(boot_seed, &[BOOT, tag, b])` for
/// replicate `b`.
pub fn summarise(ds: &[&DecisionCheck], boots: u64, boot_seed: u64, tag: u64) -> Summary {
    let n = ds.len();
    let unit = ONE as f64;
    let losses: Vec<u64> = ds.iter().map(|d| d.loss).collect();
    let mean_loss = losses.iter().map(|&l| l as f64).sum::<f64>() / n.max(1) as f64 / unit;
    let mut reps: Vec<f64> = (0..boots)
        .map(|b| {
            let mut rng = Rng::derive(boot_seed, &[BOOT, tag, b]);
            let s: u128 = (0..n)
                .map(|_| u128::from(losses[rng.bounded(n as u64) as usize]))
                .sum();
            s as f64 / n.max(1) as f64 / unit
        })
        .collect();
    reps.sort_by(f64::total_cmp);
    let upper = if reps.is_empty() {
        mean_loss
    } else {
        reps[((reps.len() as f64 * 0.975).ceil() as usize).clamp(1, reps.len()) - 1]
    };
    let errs: Vec<i64> = ds.iter().flat_map(|d| d.errors.iter().copied()).collect();
    let stats = |e: &[i64]| -> (f64, f64, f64, f64) {
        let m = e.len().max(1) as f64;
        let mean = e.iter().map(|&x| x as f64).sum::<f64>() / m;
        let var = e.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / m;
        let mut abs: Vec<u64> = e.iter().map(|x| x.unsigned_abs()).collect();
        abs.sort_unstable();
        let p999 = if abs.is_empty() {
            0
        } else {
            abs[((abs.len() as f64 * 0.999).ceil() as usize).clamp(1, abs.len()) - 1]
        };
        let max = abs.last().copied().unwrap_or(0);
        (
            mean / unit,
            var.sqrt() / unit,
            p999 as f64 / unit,
            max as f64 / unit,
        )
    };
    let rel: Vec<Vec<i64>> = (0..4)
        .map(|k| errs.iter().skip(k).step_by(4).copied().collect())
        .collect();
    let bias: [f64; 4] = core::array::from_fn(|k| stats(&rel[k]).0);
    let (_, sd, p999_abs, max_abs) = stats(&errs);
    let (_, decider_sd, decider_p999_abs, _) = stats(&rel[0]);
    Summary {
        decisions: n,
        mean_loss,
        loss_upper95: upper,
        agreement: ds.iter().filter(|d| d.agree).count() as f64 / n.max(1) as f64,
        values: errs.len(),
        bias,
        sd,
        p999_abs,
        max_abs,
        decider_sd,
        decider_p999_abs,
        loss_pass: upper <= STANDARD,
        bias_pass: bias.iter().all(|b| b.abs() <= STANDARD),
    }
}

/// The position bands of §3: 1–2, 3–4, 5–8.
pub const BANDS: [(&str, u8, u8); 3] = [("pos 1-2", 1, 2), ("pos 3-4", 3, 4), ("pos 5-8", 5, 8)];

/// Overall, per band and per position.
pub fn report(run: &CheckRun, boots: u64, boot_seed: u64) -> Vec<(String, Summary)> {
    let all: Vec<&DecisionCheck> = run.decisions.iter().collect();
    let mut out = vec![("all".to_string(), summarise(&all, boots, boot_seed, 0))];
    for (i, &(name, lo, hi)) in BANDS.iter().enumerate() {
        let g: Vec<&DecisionCheck> = all
            .iter()
            .copied()
            .filter(|d| (lo..=hi).contains(&d.position))
            .collect();
        out.push((
            name.to_string(),
            summarise(&g, boots, boot_seed, 1 + i as u64),
        ));
    }
    for p in 1..=8u8 {
        let g: Vec<&DecisionCheck> = all.iter().copied().filter(|d| d.position == p).collect();
        if !g.is_empty() {
            out.push((
                format!("pos {p}"),
                summarise(&g, boots, boot_seed, 10 + u64::from(p)),
            ));
        }
    }
    out
}

/// The §3 measures per stratum of the hand's start state (`crate::measure::stratum_of`, S1–S4:
/// the score groups the yardstick's decisions are drawn by), the bootstrap tagged `100 + s`. Every
/// stratum is listed, with no decisions when the check has none.
pub fn strata(
    run: &CheckRun,
    ds: &[DecisionSpec],
    boots: u64,
    boot_seed: u64,
) -> Vec<(String, Summary)> {
    let stratum: BTreeMap<u64, u8> = ds.iter().map(|d| (d.id, stratum_of(&d.start))).collect();
    (1..=4u8)
        .map(|s| {
            let g: Vec<&DecisionCheck> = run
                .decisions
                .iter()
                .filter(|d| stratum.get(&d.id) == Some(&s))
                .collect();
            (
                format!("S{s}"),
                summarise(&g, boots, boot_seed, 100 + u64::from(s)),
            )
        })
        .collect()
}

/// The name of race-group `c` of [`race_groups`] (`nn::outcome::RACE_CLASSES` 0–4, then 5 for
/// the decisions with no class at the share).
pub fn race_group_name(c: usize) -> String {
    match nn::outcome::RACE_CLASS_NAMES.get(c) {
        Some(n) => format!("race {c}: {n}"),
        None => "race mixed: no class at the share".to_string(),
    }
}

/// The §3 measures per race class (module notes): a decision is in class `c` when at least the
/// share `micro` (millionths) of the distinct frontier positions its hybrid searches reached are
/// of that class, else in "mixed" (the last entry); the bootstrap tagged `200 + c`.
///
/// # Panics
/// If the run did not record race classes.
pub fn race_groups(
    run: &CheckRun,
    micro: u64,
    boots: u64,
    boot_seed: u64,
) -> Vec<(String, Summary)> {
    let class: Vec<Option<usize>> = run
        .decisions
        .iter()
        .map(|d| {
            let r = d
                .race
                .as_ref()
                .expect("a check with race groups records them");
            (0..nn::outcome::N_RACE_CLASSES)
                .find(|&c| nn::predicate::at_share(r.in_group[c], r.reached, micro))
        })
        .collect();
    (0..=nn::outcome::N_RACE_CLASSES)
        .map(|c| {
            let want = (c < nn::outcome::N_RACE_CLASSES).then_some(c);
            let g: Vec<&DecisionCheck> = run
                .decisions
                .iter()
                .zip(&class)
                .filter(|(_, k)| **k == want)
                .map(|(d, _)| d)
                .collect();
            (
                race_group_name(c),
                summarise(&g, boots, boot_seed, 200 + c as u64),
            )
        })
        .collect()
}

/// One slice of a check (docs/m3-hybrid/audit-loop.md §3 item 2): the decisions in a group of a
/// predicate file, or in none.
#[derive(Clone, Debug, Serialize)]
pub struct Slice {
    /// The group's index in the file (`None`: the decisions in no group).
    pub group: Option<usize>,
    /// The group's name (`no group`).
    pub name: String,
    /// The group's conditions, as text.
    pub conditions: String,
    /// The decisions in the slice (ids).
    pub ids: Vec<u64>,
    /// Over the check's decisions, the group's share of all the reached frontier positions
    /// (the group's weight as the hybrid reaches it; `None` for "no group").
    pub reached_share: Option<f64>,
    /// The §3 measures of the slice's decisions.
    pub summary: Summary,
    /// The decision-loss bound passes (`None` with no decisions: nothing to measure, a mini
    /// golden set is needed).
    pub pass: Option<bool>,
}

/// The slices of a check run with a predicate file ([`check_worlds_with`]): for each group, the
/// decisions in it (at least the share `micro`, in millionths, of their reached frontier
/// positions in the group: `crate::audit::Reach::groups_at`), then the decisions in no group;
/// each with the §3 measures (the bootstrap tagged `1000 + g`, and `999` for no group).
///
/// # Panics
/// If the run did not record reaches.
pub fn slices(
    run: &CheckRun,
    preds: &Predicates,
    micro: u64,
    boots: u64,
    boot_seed: u64,
) -> Vec<Slice> {
    let groups: Vec<Vec<u32>> = run
        .decisions
        .iter()
        .map(|d| {
            d.reach
                .as_ref()
                .expect("a check with slices records reaches")
                .groups_at(micro)
        })
        .collect();
    let mut total = Reach::new(preds.len());
    for d in &run.decisions {
        total.add(d.reach.as_ref().expect("recorded"));
    }
    let make = |g: Option<usize>, pick: &dyn Fn(&[u32]) -> bool| {
        let ds: Vec<&DecisionCheck> = run
            .decisions
            .iter()
            .zip(&groups)
            .filter(|(_, gs)| pick(gs))
            .map(|(d, _)| d)
            .collect();
        let summary = summarise(&ds, boots, boot_seed, g.map_or(999, |g| 1000 + g as u64));
        Slice {
            group: g,
            name: g.map_or("no group".to_string(), |g| preds.groups[g].name.clone()),
            conditions: g.map_or(String::new(), |g| preds.groups[g].conditions_text()),
            ids: ds.iter().map(|d| d.id).collect(),
            reached_share: g.map(|g| total.share(g)),
            pass: (!ds.is_empty()).then_some(summary.loss_pass),
            summary,
        }
    };
    let mut out: Vec<Slice> = (0..preds.len())
        .map(|g| make(Some(g), &|gs: &[u32]| gs.contains(&(g as u32))))
        .collect();
    out.push(make(None, &|gs: &[u32]| gs.is_empty()));
    out
}

/// Seed label of the paired bootstrap.
const PAIRED: u64 = 0xBA12;

/// One decision of a check as the paired comparison reads it (a `hybrid-check` JSON's
/// `decisions` entry, [`DecisionCheck`]).
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct PairedRow {
    /// Decision id.
    pub id: u64,
    /// Position in the trick (1 = leader … 8).
    pub position: u8,
    /// Candidates.
    pub cands: usize,
    /// Worlds.
    pub worlds: u64,
    /// Decision loss (units of 2^-32).
    pub loss: u64,
    /// The decider's own team's errors summed (units of 2^-32).
    pub own_error_sum: i64,
}

/// One measure of the paired comparison: both players' values, their difference (B − A) and
/// its bootstrap distribution (fractions of money chance).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PairedMeasure {
    /// What is measured: `loss` (mean decision loss) or `own bias` (the decider's own team's
    /// mean value error), and over which decisions (`all`, `pos 1-2`, `pos 1`, …).
    pub measure: String,
    /// Decisions in the group.
    pub decisions: usize,
    /// Player A's value.
    pub a: f64,
    /// Player B's value.
    pub b: f64,
    /// B − A.
    pub diff: f64,
    /// The bootstrap's 2.5th and 97.5th percentiles of B − A (the 95% interval).
    pub diff_lo95: f64,
    /// See `diff_lo95`.
    pub diff_hi95: f64,
    /// The share of replicates with B − A below 0, and above 0.
    pub share_below: f64,
    /// See `share_below`.
    pub share_above: f64,
}

/// The groups of the paired comparison: (name, lowest position, highest position).
pub const PAIRED_GROUPS: [(&str, u8, u8); 6] = [
    ("all", 1, 8),
    ("pos 1-2", 1, 2),
    ("pos 3-4", 3, 4),
    ("pos 5-8", 5, 8),
    ("pos 1", 1, 1),
    ("pos 2", 2, 2),
];

/// The paired comparison of two players' checks on the same decisions and worlds (e.g. one
/// network against an ensemble): for each group of [`PAIRED_GROUPS`], the mean decision loss and
/// the own-team bias of each, B − A, and its bootstrap distribution: replicate `r` resamples the
/// group's decisions with replacement (`Rng::derive(boot_seed, &[PAIRED, group, r])`, the same
/// decisions for both players, so what they share cancels) and takes B − A on the resample.
/// The decisions must be the same, in the same order, with the same candidates and worlds.
pub fn paired(
    a: &[PairedRow],
    b: &[PairedRow],
    boots: u64,
    boot_seed: u64,
) -> Result<Vec<PairedMeasure>, String> {
    if a.len() != b.len() || a.is_empty() {
        return Err(format!(
            "the two checks hold {} and {} decisions: a paired comparison needs the same ones",
            a.len(),
            b.len()
        ));
    }
    for (x, y) in a.iter().zip(b) {
        if (x.id, x.position, x.cands, x.worlds) != (y.id, y.position, y.cands, y.worlds) {
            return Err(format!(
                "decision {} (position {}, {} candidates, {} worlds) is paired with decision {} \
                 (position {}, {} candidates, {} worlds): not the same decisions",
                x.id, x.position, x.cands, x.worlds, y.id, y.position, y.cands, y.worlds
            ));
        }
    }
    let unit = ONE as f64;
    let loss = |rows: &[PairedRow], ix: &[usize]| -> f64 {
        let s: u128 = ix.iter().map(|&i| u128::from(rows[i].loss)).sum();
        s as f64 / ix.len().max(1) as f64 / unit
    };
    let bias = |rows: &[PairedRow], ix: &[usize]| -> f64 {
        let s: i128 = ix.iter().map(|&i| i128::from(rows[i].own_error_sum)).sum();
        let n: u128 = ix
            .iter()
            .map(|&i| rows[i].cands as u128 * u128::from(rows[i].worlds))
            .sum();
        s as f64 / n.max(1) as f64 / unit
    };
    let mut out = Vec::new();
    for (kind, f) in [
        ("loss", &loss as &dyn Fn(&[PairedRow], &[usize]) -> f64),
        ("own bias", &bias),
    ] {
        for (g, &(name, lo, hi)) in PAIRED_GROUPS.iter().enumerate() {
            let members: Vec<usize> = (0..a.len())
                .filter(|&i| (lo..=hi).contains(&a[i].position))
                .collect();
            if members.is_empty() {
                continue;
            }
            let (va, vb) = (f(a, &members), f(b, &members));
            let tag = g as u64 + if kind == "loss" { 0 } else { 100 };
            let n = members.len() as u64;
            let mut diffs: Vec<f64> = (0..boots)
                .map(|r| {
                    let mut rng = Rng::derive(boot_seed, &[PAIRED, tag, r]);
                    let ix: Vec<usize> = (0..n).map(|_| members[rng.bounded(n) as usize]).collect();
                    f(b, &ix) - f(a, &ix)
                })
                .collect();
            diffs.sort_by(f64::total_cmp);
            let pick = |q: f64| -> f64 {
                if diffs.is_empty() {
                    vb - va
                } else {
                    diffs[((diffs.len() as f64 * q).ceil() as usize).clamp(1, diffs.len()) - 1]
                }
            };
            let m = diffs.len().max(1) as f64;
            out.push(PairedMeasure {
                measure: format!("{kind}, {name}"),
                decisions: members.len(),
                a: va,
                b: vb,
                diff: vb - va,
                diff_lo95: pick(0.025),
                diff_hi95: pick(0.975),
                share_below: diffs.iter().filter(|&&d| d < 0.0).count() as f64 / m,
                share_above: diffs.iter().filter(|&&d| d > 0.0).count() as f64 / m,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// `--standard`: percent text to the fraction ("0.01" is exactly `STANDARD`), refusals, and
    /// the judgement of a summary by it (decision-loss bound and every team's bias).
    #[test]
    fn the_standard_is_a_setting() {
        assert_eq!(parse_standard("0.01"), Ok(STANDARD));
        assert_eq!(parse_standard("0.005"), Ok(5e-5));
        assert_eq!(parse_standard(" 0.0075 "), Ok(7.5e-5));
        assert_eq!(parse_standard("1"), Ok(0.01));
        for bad in ["0", "", "-0.01", "1.5", "abc", "1e-2", "0.01%"] {
            assert!(parse_standard(bad).is_err(), "{bad}");
        }
        let d = |loss: u64, e: i64| DecisionCheck {
            id: 0,
            position: 1,
            stratum: 1,
            team: 0,
            cands: 2,
            worlds: 1,
            exact_choice: 0,
            hybrid_choice: 1,
            loss,
            agree: loss == 0,
            own_error_sum: e,
            errors: vec![e, -e, 0, 0],
            exact_worlds: 0,
            exact_us: 0,
            hybrid_us: 0,
            exact_nodes: 0,
            hybrid_nodes: 0,
            frontier: 0,
            reach: None,
            race: None,
            exact_leaves: None,
            silver: false,
            world_checks: Vec::new(),
        };
        // Losses of about 0.006%, a bias of +-0.003%.
        let ds: Vec<DecisionCheck> = (0..20).map(|i| d(250_000 + 1000 * i, 130_000)).collect();
        let refs: Vec<&DecisionCheck> = ds.iter().collect();
        let mut s = summarise(&refs, 200, 1, 0);
        assert!(s.loss_pass && s.bias_pass, "{s:?}");
        judge(&mut s, 5e-5);
        assert!(!s.loss_pass && s.bias_pass, "{s:?}");
        judge(&mut s, 2e-5);
        assert!(!s.loss_pass && !s.bias_pass, "{s:?}");
        let u = s.loss_upper95;
        judge(&mut s, u);
        assert!(s.loss_pass && s.bias_pass);
        judge(&mut s, STANDARD);
        assert!(s.loss_pass && s.bias_pass);
    }

    use equity::MatchState;
    use solver::FnFrontier;

    use crate::measure::{build_decisions, collect_starts};
    use crate::pimc::SharedLeaf;
    use crate::worlds::BeliefSource;

    fn leaf() -> Arc<SharedLeaf> {
        Arc::new(solver::FnLeaf(|s: &MatchState| {
            MoneyChance::from_weights(core::array::from_fn(|t| {
                1 + u128::from(s.scores[t]) + 40 * u128::from(s.games[t])
            }))
        }))
    }

    fn even(_: &CanonPos, _: &MatchCtx) -> MoneyChance {
        MoneyChance::even()
    }

    /// The check is thread-independent, and with stored exact values (made by the yardstick's
    /// own worlds code) it verifies them and gives the same result.
    #[test]
    fn check_is_deterministic_and_reads_stored_values() {
        let starts = collect_starts(300, 3);
        let ds = build_decisions(4, 8, 5, &starts);
        let l = leaf();
        let src = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: engine::game::Rules::HOUSE,
        };
        let f = FnFrontier(even as fn(&CanonPos, &MatchCtx) -> MoneyChance);
        let job = |threads| CheckJob {
            cut: 5,
            seed: 5,
            label: 1,
            to: 6,
            threads,
            tt_mb: 4,
            verify_pct: 100.0,
            recompute: false,
            rules_robber_burn: false,
            exact_solve_threads: 1,
            exact_tt_mb: 4,
            noise: None,
            exact_leaves: None,
        };
        let a = run_check(&ds, &BTreeMap::new(), &job(1), &ctx, &f).expect("runs");
        let b = run_check(&ds, &BTreeMap::new(), &job(3), &ctx, &f).expect("runs");
        let key = |r: &CheckRun| -> Vec<(u64, usize, usize, u64, Vec<i64>)> {
            r.decisions
                .iter()
                .map(|d| {
                    (
                        d.id,
                        d.exact_choice,
                        d.hybrid_choice,
                        d.loss,
                        d.errors.clone(),
                    )
                })
                .collect()
        };
        assert_eq!(key(&a), key(&b));
        assert!(a.decisions.iter().any(|d| d.errors.iter().any(|&e| e != 0)));
        // Stored exact values from a yardstick-style worlds run.
        let cfg = SearchConfig {
            tt_mb: 4,
            ..SearchConfig::default()
        };
        let stored: BTreeMap<u64, WorldRows> = ds
            .iter()
            .map(|d| {
                let r = crate::measure::world_rows(
                    d,
                    crate::pimc::Method::ExactPimc,
                    1,
                    5,
                    0,
                    6,
                    "t",
                    &ctx,
                    &cfg,
                );
                (d.id, r)
            })
            .collect();
        let c = run_check(&ds, &stored, &job(2), &ctx, &f).expect("runs");
        assert_eq!(key(&a), key(&c));
        assert_eq!(c.verified, (8 * 6, 0));
        let s = report(&a, 50, 1);
        assert_eq!(s[0].1.decisions, 8);
        assert!(s[0].1.mean_loss <= s[0].1.loss_upper95 + 1e-12);
    }

    /// With slices, the check's values are unchanged, each decision's reach is its hybrid
    /// searches' (`crate::audit::decision_reach` with the same evaluator and table), the same on
    /// 1 and 3 threads, and the slices hold the decisions at the share, then those in no group.
    #[test]
    fn slices_record_the_hybrids_own_reach() {
        let starts = collect_starts(300, 3);
        let ds = build_decisions(3, 8, 5, &starts);
        let l = leaf();
        let src = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: engine::game::Rules::HOUSE,
        };
        let f = FnFrontier(even as fn(&CanonPos, &MatchCtx) -> MoneyChance);
        let job = |threads| CheckJob {
            cut: 4,
            seed: 5,
            label: 1,
            to: 3,
            threads,
            tt_mb: 4,
            verify_pct: 0.0,
            recompute: false,
            rules_robber_burn: false,
            exact_solve_threads: 1,
            exact_tt_mb: 4,
            noise: None,
            exact_leaves: None,
        };
        let text = format!(
            r#"[{{"name": "next team no trick", "conditions": [{{"feature": {}, "op": "==", "value": 1}}]}},
                {{"name": "never", "conditions": [{{"feature": {}, "op": "==", "value": 1}},
                                                   {{"feature": {}, "op": "==", "value": 1}}]}}]"#,
            nn::encode::OFF_TRICKS + 5,
            nn::encode::OFF_TRUMP,
            nn::encode::OFF_TRUMP + 1
        );
        let p = Predicates::parse(text.as_bytes()).expect("parses");
        let plain = run_check(&ds, &BTreeMap::new(), &job(1), &ctx, &f).expect("runs");
        let a = run_check_slices(&ds, &BTreeMap::new(), &job(1), &ctx, &f, Some(&p)).expect("runs");
        let b = run_check_slices(&ds, &BTreeMap::new(), &job(3), &ctx, &f, Some(&p)).expect("runs");
        let vals = |r: &CheckRun| -> Vec<(u64, u64, Vec<i64>)> {
            r.decisions
                .iter()
                .map(|d| (d.id, d.loss, d.errors.clone()))
                .collect()
        };
        assert_eq!(vals(&plain), vals(&a), "slices never change a value");
        assert!(plain.decisions.iter().all(|d| d.reach.is_none()));
        let reaches = |r: &CheckRun| -> Vec<Reach> {
            r.decisions
                .iter()
                .map(|d| d.reach.clone().expect("recorded"))
                .collect()
        };
        assert_eq!(reaches(&a), reaches(&b), "thread-independent");
        let mut mem = SolveMem::new();
        for (d, x) in ds.iter().zip(&a.decisions) {
            let want = crate::audit::decision_reach(
                d,
                &p,
                &ctx,
                4,
                4,
                measure_seed(5, 1, d.id),
                3,
                Some(&f),
                &mut mem,
            );
            assert_eq!(x.reach.as_ref(), Some(&want), "decision {}", d.id);
            assert_eq!(want.in_group[1], 0);
        }
        let sl = slices(&a, &p, 500_000, 50, 1);
        assert_eq!(sl.len(), 3);
        assert_eq!(
            sl.iter().map(|x| x.name.as_str()).collect::<Vec<_>>(),
            ["next team no trick", "never", "no group"]
        );
        let half: Vec<u64> = a
            .decisions
            .iter()
            .filter(|d| {
                let r = d.reach.as_ref().expect("recorded");
                2 * r.in_group[0] >= r.reached
            })
            .map(|d| d.id)
            .collect();
        assert_eq!(sl[0].ids, half);
        assert!(sl[1].ids.is_empty() && sl[1].pass.is_none());
        assert_eq!(sl[0].ids.len() + sl[2].ids.len(), 8, "a partition here");
        assert_eq!(sl[0].summary.decisions, sl[0].ids.len());
        assert!(!sl[0].ids.is_empty() && !sl[2].ids.is_empty(), "{sl:?}");
        assert_eq!(
            sl[0].conditions,
            format!("f{} == 1", nn::encode::OFF_TRICKS + 5)
        );
    }

    // ------------------------------------------------------------ silver references

    use crate::measure::{FrontierFp, HYBRID, WorldsJob, build_decision_at, world_rows_net};
    use crate::pimc::{Method, world_match};

    fn syn_leaf() -> Arc<SharedLeaf> {
        Arc::new(nn::synthetic::SyntheticLeaf)
    }

    /// A random integer network for frontier trick `k`.
    fn oracle(k: u8, seed: u64) -> nn::Oracle {
        let net = nn::synthetic::random_net(seed, &[32, 16], nn::outcome::vocab(k), 100);
        let meta = BTreeMap::from([("head".to_string(), "outcome-v1".to_string())]);
        nn::Oracle::from_bytes(&nn::oracle::write_net(&net, k, &meta), syn_leaf()).expect("loads")
    }

    fn fp(o: &nn::Oracle) -> FrontierFp {
        FrontierFp {
            cut: o.trick,
            net_sha256: o.sha256.clone(),
            encoder: nn::encode::ENCODER.to_string(),
        }
    }

    /// Decisions `ids` of the yardstick's generator at `trick` (seed 3), ids kept.
    fn decisions_at(trick: u8, ids: &[u64]) -> Vec<DecisionSpec> {
        let starts = collect_starts(400, 5);
        ids.iter()
            .map(|&i| build_decision_at(trick, i, 3, &starts))
            .collect()
    }

    fn check_job(cut: u8, threads: usize, to: u64) -> CheckJob {
        CheckJob {
            cut,
            seed: 7,
            label: 1,
            to,
            threads,
            tt_mb: 4,
            verify_pct: 100.0,
            recompute: false,
            rules_robber_burn: false,
            exact_solve_threads: 1,
            exact_tt_mb: 4,
            noise: None,
            exact_leaves: None,
        }
    }

    /// The silver reference's records of `ds` (worlds `0..to`, seed 7, label 1): a silver file's
    /// contents, made by the worlds runner.
    fn silver_rows(ds: &[DecisionSpec], o: &nn::Oracle, ctx: &Ctx, to: u64) -> Vec<WorldRows> {
        let job = WorldsJob {
            method: Method::ExactPimc,
            label: 1,
            seed: 7,
            from: 0,
            to,
            table: "t".into(),
            frontier: Some(fp(o)),
        };
        let cfg = SearchConfig {
            tt_mb: 4,
            ..SearchConfig::default()
        };
        world_rows_net(ds, &job, ctx, &cfg, Some(o)).expect("valued")
    }

    type Key = Vec<(u64, usize, usize, u64, Vec<i64>)>;

    fn key(r: &CheckRun) -> Key {
        r.decisions
            .iter()
            .map(|d| {
                (
                    d.id,
                    d.exact_choice,
                    d.hybrid_choice,
                    d.loss,
                    d.errors.clone(),
                )
            })
            .collect()
    }

    /// `hybrid-check` against a silver reference reproduces the plain H_2-vs-H_3 comparison on a
    /// tiny trick-1 case: per world, every candidate valued by H_3 (exact search through tricks 1
    /// and 2, O_3 at the start of trick 3) and by H_2 (O_2 at the start of trick 2) on the worlds
    /// the yardstick draws; the choices, the loss of H_2's choice under H_3's means and every
    /// value error are the check's, whether the reference values are recomputed by the check or
    /// read from a silver file made by the worlds runner (each then recomputed and found
    /// byte-identical), on 1 and 3 threads, and with the race groups recorded or not.
    #[test]
    fn silver_check_reproduces_the_plain_comparison() {
        // Trick-1 decisions at seats 7-8 (cheap), every stratum twice.
        let ds = decisions_at(1, &[24, 25, 26, 27, 28, 29, 30, 31]);
        assert!(ds.iter().all(|d| d.trick == 1 && d.position >= 6));
        let l = syn_leaf();
        let src = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: engine::game::Rules::HOUSE,
        };
        let (o2, o3) = (oracle(2, 21), oracle(3, 22));
        let to = 3;
        // The plain comparison.
        let cfg = SearchConfig {
            tt_mb: 4,
            ..SearchConfig::default()
        };
        let mut mem = SolveMem::new();
        let mut plain: Key = Vec::new();
        for d in &ds {
            let (_, view, legal) = d.replay();
            let (_, cands) = candidates(&view, &legal);
            let worlds = ctx.source.prepare(&view, ctx.rules);
            let dseed = measure_seed(7, 1, d.id);
            let (mut h3, mut h2) = (Vec::new(), Vec::new());
            for w in 0..to {
                let m = world_match(&view, &*worlds, w, dseed, ctx.rules);
                for (cut, net, out) in [
                    (3, &o3 as &dyn Frontier, &mut h3),
                    (2, &o2 as &dyn Frontier, &mut h2),
                ] {
                    let mut run = WorldSolve::default();
                    out.push(
                        exact_world(
                            &m,
                            &cands,
                            &ctx,
                            &cfg,
                            None,
                            &mut mem,
                            &mut run,
                            Some(FrontierCut::new(cut, net)),
                        )
                        .expect("solved"),
                    );
                }
            }
            let team = engine::team_of(d.seat);
            let (m3, c3, _) = aggregate(&h3, cands.len(), team);
            let (m2, _, _) = aggregate(&h2, cands.len(), team);
            let c2 = best_index(&m2, team);
            let mut errors = Vec::new();
            for (a, b) in h3.iter().zip(&h2) {
                for (e, h) in a.iter().zip(b) {
                    for rel in 0..4u8 {
                        let t = (team + rel) % 4;
                        errors.push(h.get(t) as i64 - e.get(t) as i64);
                    }
                }
            }
            plain.push((d.id, c3, c2, m3[c3].get(team) - m3[c2].get(team), errors));
        }
        assert!(
            plain.iter().any(|x| x.4.iter().any(|&e| e != 0)),
            "H_2 is not H_3"
        );
        let f3 = fp(&o3);
        let reference = Some((
            Reference {
                cut: 3,
                net: &o3 as &dyn Frontier,
            },
            &f3,
        ));
        // Recomputed by the check.
        let none = BTreeMap::new();
        let job = check_job(2, 1, to);
        let a =
            run_check_ref(&ds, &none, &job, &ctx, &o2, None, reference, true, None).expect("runs");
        assert_eq!(key(&a), plain, "the check recomputes the plain comparison");
        assert!(a.decisions.iter().all(|d| d.silver));
        assert_eq!(a.verified, (0, 0));
        // From a silver file (its every world recomputed and checked byte for byte).
        let rows = silver_rows(&ds, &o3, &ctx, to);
        assert!(rows.iter().all(|r| r.method == HYBRID));
        let stored =
            stored_reference(rows, 1, &|_| true, Some(&f3), false, "silver").expect("taken");
        let b = run_check_ref(
            &ds,
            &stored,
            &check_job(2, 3, to),
            &ctx,
            &o2,
            None,
            reference,
            true,
            None,
        )
        .expect("runs");
        assert_eq!(
            key(&b),
            plain,
            "the stored silver file gives the same check"
        );
        assert_eq!(b.verified, (ds.len() as u64 * to, 0));
        // Race groups and strata: recorded without changing a value; each a partition.
        let c = run_check_ref(
            &ds,
            &none,
            &check_job(2, 3, to),
            &ctx,
            &o2,
            None,
            reference,
            false,
            None,
        )
        .expect("runs");
        assert_eq!(key(&c), plain);
        assert!(c.decisions.iter().all(|d| d.race.is_none()));
        for d in &a.decisions {
            let r = d.race.as_ref().expect("recorded");
            assert!(
                r.reached > 0 && r.in_group.iter().sum::<u64>() == r.reached,
                "{r:?}"
            );
        }
        assert_eq!(
            a.decisions
                .iter()
                .map(|d| d.race.clone())
                .collect::<Vec<_>>(),
            b.decisions
                .iter()
                .map(|d| d.race.clone())
                .collect::<Vec<_>>(),
            "thread-independent"
        );
        let rg = race_groups(&a, 500_000, 50, 1);
        assert_eq!(rg.len(), nn::outcome::N_RACE_CLASSES + 1);
        assert_eq!(rg.iter().map(|(_, s)| s.decisions).sum::<usize>(), ds.len());
        let st = strata(&a, &ds, 50, 1);
        assert_eq!(st.len(), 4);
        assert_eq!(st.iter().map(|(_, s)| s.decisions).sum::<usize>(), ds.len());
        assert!(st.iter().all(|(_, s)| s.decisions == 2), "{st:?}");
        let r = report(&a, 50, 1);
        assert_eq!(r[0].1.decisions, ds.len());
    }

    /// A silver file is refused where exact values are required, and exact values where a
    /// silver reference is, unless allowed (then they stand in for it: the check is then exact
    /// for those decisions, exactly as the exact check); another network's silver file is
    /// refused; the reference must look deeper than the hybrid; and the saved trick-1 worlds
    /// refuse a silver line.
    #[test]
    fn silver_and_exact_references_are_never_confused() {
        // Trick-2 decisions late in the trick: H_3 checked against H_4, exact solves cheap.
        let ds = decisions_at(2, &[20, 21, 26, 31]);
        let l = syn_leaf();
        let src = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: engine::game::Rules::HOUSE,
        };
        let (o3, o4, o4b) = (oracle(3, 31), oracle(4, 32), oracle(4, 33));
        let (f4, f4b) = (fp(&o4), fp(&o4b));
        let to = 3;
        let silver = silver_rows(&ds, &o4, &ctx, to);
        let cfg = SearchConfig {
            tt_mb: 4,
            ..SearchConfig::default()
        };
        let exact: Vec<WorldRows> = ds
            .iter()
            .map(|d| crate::measure::world_rows(d, Method::ExactPimc, 1, 7, 0, to, "t", &ctx, &cfg))
            .collect();
        let all = |_: u64| true;
        // Exact required: a silver file is refused.
        let e = stored_reference(silver.clone(), 2, &all, None, false, "s.jsonl")
            .expect_err("silver as exact");
        assert!(
            e.contains("never an exact golden set") && e.contains("--reference silver"),
            "{e}"
        );
        let e = stored_reference(silver.clone(), 2, &all, None, true, "s.jsonl")
            .expect_err("allowing exact does not allow silver");
        assert!(e.contains("never an exact golden set"), "{e}");
        assert_eq!(
            stored_reference(exact.clone(), 2, &all, None, false, "e.jsonl")
                .expect("exact")
                .len(),
            ds.len()
        );
        // Silver required: exact values are refused unless allowed; another network's silver
        // file is refused.
        let e = stored_reference(exact.clone(), 2, &all, Some(&f4), false, "e.jsonl")
            .expect_err("exact as silver");
        assert!(e.contains("--allow-exact"), "{e}");
        let e = stored_reference(silver.clone(), 2, &all, Some(&f4b), true, "s.jsonl")
            .expect_err("another network");
        assert!(e.contains("not this check's reference"), "{e}");
        // Allowed: exact records stand in (the anchor decisions) and the silver ones serve the
        // rest; each decision's reference is the one stored for it.
        let mixed: Vec<WorldRows> = silver
            .iter()
            .filter(|r| r.id != ds[0].id)
            .cloned()
            .chain(exact.iter().filter(|r| r.id == ds[0].id).cloned())
            .collect();
        let stored = stored_reference(mixed, 2, &all, Some(&f4), true, "m").expect("allowed");
        let reference = Some((
            Reference {
                cut: 4,
                net: &o4 as &dyn Frontier,
            },
            &f4,
        ));
        let job = check_job(3, 2, to);
        let m = run_check_ref(&ds, &stored, &job, &ctx, &o3, None, reference, false, None)
            .expect("runs");
        assert_eq!(
            m.verified,
            (ds.len() as u64 * to, 0),
            "each against its own kind"
        );
        assert_eq!(
            m.decisions.iter().map(|d| d.silver).collect::<Vec<_>>(),
            [false, true, true, true]
        );
        let exact_map: BTreeMap<u64, WorldRows> = exact.iter().map(|r| (r.id, r.clone())).collect();
        let ex = run_check(&ds, &exact_map, &job, &ctx, &o3).expect("exact check");
        assert_eq!(
            key(&m)[0],
            key(&ex)[0],
            "the anchor decision is checked against exact"
        );
        let sv = run_check_ref(
            &ds,
            &BTreeMap::new(),
            &job,
            &ctx,
            &o3,
            None,
            reference,
            false,
            None,
        )
        .expect("silver check");
        assert_eq!(
            key(&m)[1..],
            key(&sv)[1..],
            "the others against the silver values"
        );
        // The library refuses a stored record of the wrong kind too.
        let silver_map: BTreeMap<u64, WorldRows> =
            silver.iter().map(|r| (r.id, r.clone())).collect();
        let e = run_check(&ds, &silver_map, &job, &ctx, &o3).expect_err("silver as exact");
        assert!(e.contains("not exact"), "{e}");
        let wrong = Some((
            Reference {
                cut: 4,
                net: &o4b as &dyn Frontier,
            },
            &f4b,
        ));
        let e = run_check_ref(&ds, &silver_map, &job, &ctx, &o3, None, wrong, false, None)
            .expect_err("another network");
        assert!(e.contains("not the reference"), "{e}");
        // The reference looks deeper than the hybrid it checks.
        let o3b = oracle(3, 34);
        let f3b = fp(&o3b);
        let shallow = Some((
            Reference {
                cut: 3,
                net: &o3b as &dyn Frontier,
            },
            &f3b,
        ));
        let e = run_check_ref(
            &ds,
            &BTreeMap::new(),
            &job,
            &ctx,
            &o3,
            None,
            shallow,
            false,
            None,
        )
        .expect_err("not deeper");
        assert!(e.contains("look deeper"), "{e}");
        // The saved trick-1 worlds (search::saved) refuse a silver record and a silver
        // checkpoint line.
        let line = serde_json::to_string(&silver[0]).expect("json");
        let e = crate::saved::parse_saved(&line, "x").expect_err("a silver record");
        assert!(e.contains("silver"), "{e}");
        let ck = crate::measure::CkptRow {
            id: 1,
            fp: 2,
            method: HYBRID.into(),
            label: 1,
            seed: 7,
            source: "belief-uniform".into(),
            table: "t".into(),
            w: 0,
            v: silver[0].rows[0].clone(),
            us: 1,
            nodes: 1,
            evals: 1,
            values: Some(crate::measure::VALUES_VERSION),
            frontier: Some(f4.clone()),
        };
        let e = crate::saved::parse_saved(&serde_json::to_string(&ck).expect("json"), "x")
            .expect_err("a silver checkpoint line");
        assert!(e.contains("silver"), "{e}");
    }

    // ------------------------------------------------------------ the curse test and ensembles

    /// Trick-3 decisions at seats 5-8 (cheap), checked with a frontier at trick 4.
    fn curse_setup() -> Vec<DecisionSpec> {
        let ds = decisions_at(3, &[16, 17, 20, 21, 24, 25, 28, 29]);
        assert!(ds.iter().all(|d| d.trick == 3));
        ds
    }

    fn noise_job(threads: usize, sd: &str, seed: u64) -> CheckJob {
        CheckJob {
            verify_pct: 0.0,
            noise: Some(crate::curse::NoiseSpec::parse(sd, seed).expect("valid")),
            ..check_job(4, threads, 3)
        }
    }

    /// The optimizer's-curse test: with the network as the truth and zero noise, the loss and
    /// every error are exactly 0; with noise, the result (values, choices, the realised noise) is
    /// the same on 1 and 3 threads, changes with the noise's seed, and the noise is really there.
    /// A reference at the check's own cut is refused without noise.
    #[test]
    fn curse_test_is_exact_at_zero_noise_and_thread_independent() {
        let ds = curse_setup();
        let l = syn_leaf();
        let src = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: engine::game::Rules::HOUSE,
        };
        let o4 = oracle(4, 41);
        let f4 = fp(&o4);
        let truth = || {
            Some((
                Reference {
                    cut: 4,
                    net: &o4 as &dyn Frontier,
                },
                &f4,
            ))
        };
        let none = BTreeMap::new();
        let run = |job: &CheckJob| {
            run_check_ref(&ds, &none, job, &ctx, &o4, None, truth(), false, None).expect("runs")
        };
        let zero = run(&noise_job(2, "0,0,0,0,0", 1));
        assert_eq!(zero.decisions.len(), 8);
        for d in &zero.decisions {
            assert_eq!(d.loss, 0, "decision {}", d.id);
            assert!(d.errors.iter().all(|&e| e == 0) && d.own_error_sum == 0);
            assert_eq!(d.exact_choice, d.hybrid_choice);
        }
        let g = report(&zero, 50, 1);
        assert_eq!((g[0].1.mean_loss, g[0].1.loss_upper95), (0.0, 0.0));
        assert_eq!(g[0].1.bias, [0.0; 4]);
        let st = zero.noise.as_ref().expect("recorded");
        assert!(st.leaves.iter().sum::<u64>() > 0 && st.sum_sq.iter().all(|&x| x == 0));
        // Noise: thread-independent, seeded.
        let sd = "3,3,5,5,8";
        let a = run(&noise_job(1, sd, 7));
        let b = run(&noise_job(3, sd, 7));
        assert_eq!(key(&a), key(&b));
        assert_eq!(a.noise, b.noise);
        let own = |r: &CheckRun| {
            r.decisions
                .iter()
                .map(|d| d.own_error_sum)
                .collect::<Vec<_>>()
        };
        assert_eq!(own(&a), own(&b));
        assert!(a.decisions.iter().any(|d| d.errors.iter().any(|&e| e != 0)));
        let c = run(&noise_job(2, sd, 8));
        assert_ne!(key(&a), key(&c), "another seed, other noise");
        let rep = a
            .noise
            .as_ref()
            .expect("recorded")
            .report(&crate::curse::NoiseSpec::parse(sd, 7).expect("valid"));
        assert!(rep.realised_sd_pct > 1.0, "{rep:?}");
        // The same cut as a reference is the curse test's truth only.
        let plain = check_job(4, 1, 3);
        let e = run_check_ref(&ds, &none, &plain, &ctx, &o4, None, truth(), false, None)
            .expect_err("refused");
        assert!(e.contains("deeper"), "{e}");
    }

    /// An ensemble of one network is that network, byte for byte (every value, choice and error
    /// of the check); the integer mean of identical networks is the network; and an ensemble of
    /// different networks differs from each member and stays thread-independent.
    #[test]
    fn ensembles_of_one_or_identical_networks_are_the_network() {
        let ds = curse_setup();
        let l = syn_leaf();
        let src = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: engine::game::Rules::HOUSE,
        };
        let none = BTreeMap::new();
        let job = |t| CheckJob {
            verify_pct: 0.0,
            ..check_job(4, t, 3)
        };
        let single = oracle(4, 51);
        let base = run_check(&ds, &none, &job(2), &ctx, &single).expect("runs");
        let one = nn::Ensemble::new(vec![oracle(4, 51)]).expect("one");
        assert_eq!(one.id(), single.sha256);
        let r1 = run_check(&ds, &none, &job(3), &ctx, &one).expect("runs");
        assert_eq!(key(&base), key(&r1));
        let world_values = |r: &CheckRun| -> Vec<Vec<MoneyChance>> {
            r.decisions
                .iter()
                .flat_map(|d| d.world_checks.iter().map(|w| w.hybrid.clone()))
                .collect()
        };
        assert_eq!(world_values(&base), world_values(&r1));
        let three =
            nn::Ensemble::new(vec![oracle(4, 51), oracle(4, 51), oracle(4, 51)]).expect("three");
        let r3 = run_check(&ds, &none, &job(1), &ctx, &three).expect("runs");
        assert_eq!(world_values(&base), world_values(&r3));
        let mixed =
            nn::Ensemble::new(vec![oracle(4, 51), oracle(4, 52), oracle(4, 53)]).expect("mixed");
        assert!(mixed.id().starts_with("ens3:"));
        let m1 = run_check(&ds, &none, &job(1), &ctx, &mixed).expect("runs");
        let m3 = run_check(&ds, &none, &job(3), &ctx, &mixed).expect("runs");
        assert_eq!(key(&m1), key(&m3));
        assert_eq!(world_values(&m1), world_values(&m3));
        assert_ne!(world_values(&m1), world_values(&base));
        assert!(nn::Ensemble::new(vec![oracle(4, 51), oracle(3, 52)]).is_err());
    }

    /// The paired bootstrap on toy checks: identical players differ by exactly 0; a player whose
    /// every loss is larger by a fixed amount differs by exactly that in every replicate; the
    /// own-team bias is the ratio of sums; different decisions are refused.
    #[test]
    fn paired_bootstrap_on_a_toy_case() {
        let row = |id: u64, position: u8, loss: u64, own: i64| PairedRow {
            id,
            position,
            cands: 2,
            worlds: 5,
            loss,
            own_error_sum: own,
        };
        let a: Vec<PairedRow> = (0..40)
            .map(|i| row(i, 1 + (i % 8) as u8, 1000 * (i % 7), -(i as i64) * 10))
            .collect();
        let same = paired(&a, &a, 200, 1).expect("paired");
        assert_eq!(same.len(), 12);
        for m in &same {
            assert_eq!((m.diff, m.diff_lo95, m.diff_hi95), (0.0, 0.0, 0.0), "{m:?}");
            assert_eq!((m.share_below, m.share_above), (0.0, 0.0));
        }
        let b: Vec<PairedRow> = a
            .iter()
            .map(|r| PairedRow {
                loss: r.loss + 4096,
                own_error_sum: r.own_error_sum + 10 * 30,
                ..r.clone()
            })
            .collect();
        let up = paired(&a, &b, 200, 1).expect("paired");
        let unit = ONE as f64;
        for m in &up {
            let want = if m.measure.starts_with("loss") {
                4096.0 / unit
            } else {
                30.0 / unit
            };
            assert!((m.diff - want).abs() < 1e-15, "{m:?}");
            assert!((m.diff_lo95 - want).abs() < 1e-15 && (m.diff_hi95 - want).abs() < 1e-15);
            assert_eq!((m.share_below, m.share_above), (0.0, 1.0));
        }
        let all = &up[0];
        assert_eq!((all.measure.as_str(), all.decisions), ("loss, all", 40));
        let seat1 = up
            .iter()
            .find(|m| m.measure == "own bias, pos 1")
            .expect("seat 1");
        assert_eq!(seat1.decisions, 5);
        let want_a: f64 =
            [0i64, 8, 16, 24, 32].iter().map(|i| -i * 10).sum::<i64>() as f64 / 50.0 / unit;
        assert!((seat1.a - want_a).abs() < 1e-18);
        // A noisy difference: the interval holds the observed difference.
        let c: Vec<PairedRow> = a
            .iter()
            .map(|r| PairedRow {
                loss: r.loss + 37 * (r.id % 5),
                ..r.clone()
            })
            .collect();
        let n = paired(&a, &c, 500, 3).expect("paired");
        assert!(n[0].diff_lo95 <= n[0].diff && n[0].diff <= n[0].diff_hi95);
        assert_eq!(n, paired(&a, &c, 500, 3).expect("paired"), "seeded");
        let mut d = a.clone();
        d[3].id = 99;
        assert!(paired(&a, &d, 10, 1).is_err());
        assert!(paired(&a, &a[..10], 10, 1).is_err());
    }

    // ------------------------------------------------------------ exact leaves

    use crate::exact_leaves::{ExactLeaves, LeafPolicy, LeafStats};
    use crate::worlds::WorldSource;

    fn leaf_job(threads: usize, policy: Option<LeafPolicy>) -> CheckJob {
        CheckJob {
            verify_pct: 0.0,
            exact_leaves: policy,
            ..check_job(4, threads, 3)
        }
    }

    fn world_values(r: &CheckRun) -> Vec<Vec<MoneyChance>> {
        r.decisions
            .iter()
            .flat_map(|d| d.world_checks.iter().map(|w| w.hybrid.clone()))
            .collect()
    }

    /// No policy is the check as before, and a policy that picks no leaf changes no value, choice
    /// or error either, byte for byte (it only counts the leaves); with every leaf exact, H_k is
    /// exact search: its values are the exact values in every world (no loss, no error); and the
    /// leaves of chosen race classes are solved, the others not, the same on 1 and 3 threads.
    #[test]
    fn exact_leaves_none_all_and_race_classes() {
        let ds = curse_setup();
        let l = syn_leaf();
        let src = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: engine::game::Rules::HOUSE,
        };
        let none = BTreeMap::new();
        let net = oracle(4, 61);
        let run = |job: &CheckJob| run_check(&ds, &none, job, &ctx, &net).expect("runs");
        let base = run(&leaf_job(2, None));
        assert!(base.exact_leaves.is_none());
        assert!(base.decisions.iter().all(|d| d.exact_leaves.is_none()));
        assert!(
            base.decisions
                .iter()
                .any(|d| d.loss > 0 || d.errors.iter().any(|&e| e != 0))
        );
        // Nothing picked: the same, byte for byte.
        let nothing = run(&leaf_job(3, Some(LeafPolicy::race(0, 4))));
        assert_eq!(key(&base), key(&nothing));
        assert_eq!(world_values(&base), world_values(&nothing));
        let st0 = nothing.exact_leaves.clone().expect("counted");
        assert_eq!(st0.exact, [0; 5]);
        assert_eq!(st0.nodes, 0);
        assert!(st0.distinct.iter().sum::<u64>() > 0);
        // Every leaf exact: exact search.
        let all = run(&leaf_job(2, Some(LeafPolicy::all(4))));
        for d in &all.decisions {
            assert_eq!(d.loss, 0, "decision {}", d.id);
            assert!(d.errors.iter().all(|&e| e == 0) && d.own_error_sum == 0);
            for w in &d.world_checks {
                assert_eq!(w.hybrid, w.exact, "decision {} world {}", d.id, w.w);
            }
        }
        let sa = all.exact_leaves.clone().expect("counted");
        assert_eq!(sa.exact, sa.distinct);
        assert!(sa.nodes > 0);
        // One race class: the most common (not every leaf), exact there only; thread-independent.
        let c = (0..5)
            .max_by_key(|&k| (st0.distinct[k], std::cmp::Reverse(k)))
            .expect("a class");
        assert!(
            st0.distinct[c] < st0.distinct.iter().sum::<u64>(),
            "{st0:?}"
        );
        let p = LeafPolicy::race(1 << c, 4);
        let r1 = run(&leaf_job(1, Some(p.clone())));
        let r3 = run(&leaf_job(3, Some(p)));
        assert_eq!(key(&r1), key(&r3));
        assert_eq!(world_values(&r1), world_values(&r3));
        let counts = |r: &CheckRun| -> Vec<_> {
            r.decisions
                .iter()
                .map(|d| d.exact_leaves.as_ref().expect("counted").counts())
                .collect()
        };
        assert_eq!(counts(&r1), counts(&r3));
        let s1: LeafStats = r1.exact_leaves.clone().expect("counted");
        for k in 0..5 {
            assert_eq!(s1.exact[k], if k == c { s1.distinct[k] } else { 0 });
        }
        assert!(s1.exact[c] > 0);
        assert_ne!(
            world_values(&r1),
            world_values(&base),
            "the exact leaves change values"
        );
    }

    /// In one world's search, each leaf the policy picks holds exactly the exact solver's value
    /// of its position (the value its label would carry: `oracle_labels::exact_value`, from
    /// empty tables), and every other leaf the network's; a band picks the leaves whose network
    /// value for the decider's team is inside it, and a band of 0-100% picks every leaf.
    #[test]
    fn an_exact_leaf_is_the_exact_solvers_value() {
        let ds = curse_setup();
        let l = syn_leaf();
        let src = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: engine::game::Rules::HOUSE,
        };
        let net = oracle(4, 62);
        let cfg = SearchConfig {
            tt_mb: 4,
            ..SearchConfig::default()
        };
        let mut checked = (0, 0, 0);
        for d in ds.iter().take(4) {
            let (_, view, legal) = d.replay();
            let (_, cands) = candidates(&view, &legal);
            let worlds = src.prepare(&view, ctx.rules);
            let team = engine::team_of(d.seat);
            for w in 0..2 {
                let m = world_match(&view, &*worlds, w, measure_seed(7, 1, d.id), ctx.rules);
                for text in ["race:0,2,4", "race:1+band:30-70", "band:0-100"] {
                    let policy = LeafPolicy::parse(text, 4).expect("valid").expect("some");
                    let x = ExactLeaves::new(
                        &net,
                        &*l,
                        &policy,
                        team,
                        solver::SolveFamily::of_match(&m),
                    );
                    let mut run = WorldSolve::default();
                    exact_world(
                        &m,
                        &cands,
                        &ctx,
                        &cfg,
                        None,
                        &mut SolveMem::new(),
                        &mut run,
                        Some(FrontierCut::new(4, &x)),
                    )
                    .expect("solves");
                    let leaves = x.leaves();
                    assert!(!leaves.is_empty());
                    for (pos, mctx, canon, v, exact) in leaves {
                        let class = nn::outcome::race_class_of(&canon, &mctx, &*l);
                        let nv = net.value(&canon, &mctx);
                        let in_band = policy
                            .band
                            .is_some_and(|(lo, hi)| (lo..=hi).contains(&nv.get(team)));
                        assert_eq!(exact, policy.has_class(class) || in_band, "{text}");
                        if exact {
                            let (e, _) = crate::oracle_labels::exact_value(
                                &pos,
                                mctx,
                                &*l,
                                solver::FastOptions::default(),
                                &mut SolveMem::new(),
                            );
                            assert_eq!(v, e, "{text}: an exact leaf is its exact value");
                            checked.0 += 1;
                        } else {
                            assert_eq!(v, nv, "{text}: the network's value");
                            checked.1 += 1;
                        }
                        if text.starts_with("band:0-100") {
                            assert!(exact);
                            checked.2 += 1;
                        }
                    }
                }
            }
        }
        assert!(
            checked.0 > 0 && checked.1 > 0 && checked.2 > 0,
            "{checked:?}"
        );
    }

    // ------------------------------------------------------------ learned exact leaves

    /// A leaf-risk model of one split: `feature` ≤ `threshold` scores `lo`, otherwise `hi`
    /// (`nn::leafrisk`'s format).
    fn risk_model(
        feature: usize,
        threshold: i64,
        lo: i64,
        hi: i64,
    ) -> Arc<nn::leafrisk::RiskModel> {
        let tree = if lo == hi {
            serde_json::json!({"f": [-1], "t": [0], "l": [0], "r": [0], "v": [lo]})
        } else {
            serde_json::json!({"f": [feature, -1, -1], "t": [threshold, 0, 0], "l": [1, 0, 0],
                               "r": [2, 0, 0], "v": [0, lo, hi]})
        };
        let v = serde_json::json!({
            "format": nn::leafrisk::FORMAT, "encoder": nn::encode::ENCODER,
            "n_inputs": nn::leafrisk::N_INPUTS, "extras": nn::leafrisk::EXTRA_NAMES,
            "net_sha256": "synthetic", "trick": 4, "base": 0, "trees": [tree],
            "budget": {"quantiles": [hi, lo]},
        });
        Arc::new(nn::leafrisk::RiskModel::parse(v.to_string().as_bytes()).expect("a valid model"))
    }

    /// The learned policy: an infinite threshold is the check without exact leaves, byte for
    /// byte (values, choices, errors); a threshold of 0 is every leaf exact, byte for byte the
    /// same as `all`; a split model solves exactly the leaves it scores at or above the
    /// threshold, the same on 1 and 3 threads; and the predictor's scores are the same in every
    /// world and thread (a pure function of the position and the network's value).
    #[test]
    fn learned_exact_leaves_inf_zero_and_split() {
        let ds = curse_setup();
        let l = syn_leaf();
        let src = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: engine::game::Rules::HOUSE,
        };
        let none = BTreeMap::new();
        let net = oracle(4, 61);
        let run = |job: &CheckJob| run_check(&ds, &none, job, &ctx, &net).expect("runs");
        let base = run(&leaf_job(2, None));
        let m = risk_model(nn::encode::N_FEATURES + 9, 0, 5, 50);
        // t = ∞: nothing solved, nothing scored, the check as before.
        let inf = run(&leaf_job(3, Some(LeafPolicy::learned(m.clone(), None, 4))));
        assert_eq!(key(&base), key(&inf));
        assert_eq!(world_values(&base), world_values(&inf));
        let si = inf.exact_leaves.clone().expect("counted");
        assert_eq!((si.exact, si.nodes, si.scored), ([0; 5], 0, 0));
        // t = 0: every leaf solved: exactly `all`.
        let all = run(&leaf_job(2, Some(LeafPolicy::all(4))));
        let zero = run(&leaf_job(
            3,
            Some(LeafPolicy::learned(m.clone(), Some(0), 4)),
        ));
        assert_eq!(key(&all), key(&zero));
        assert_eq!(world_values(&all), world_values(&zero));
        let sz = zero.exact_leaves.clone().expect("counted");
        assert_eq!(sz.exact, sz.distinct);
        assert_eq!(sz.exact, all.exact_leaves.clone().expect("counted").exact);
        // A threshold above every score: nothing solved, but every leaf scored.
        let above = run(&leaf_job(
            2,
            Some(LeafPolicy::learned(m.clone(), Some(51), 4)),
        ));
        assert_eq!(world_values(&base), world_values(&above));
        let sa = above.exact_leaves.clone().expect("counted");
        assert_eq!(sa.exact, [0; 5]);
        assert_eq!(sa.scored, sa.distinct.iter().sum::<u64>());
        // The split (L1 to the farthest possible outcome above 0: some leaves, not all).
        let p = LeafPolicy::learned(m.clone(), Some(50), 4);
        assert_eq!(p.describe(), format!("learned:{}:t=50", &m.sha256[..12]));
        let r1 = run(&leaf_job(1, Some(p.clone())));
        let r3 = run(&leaf_job(3, Some(p.clone())));
        assert_eq!(key(&r1), key(&r3));
        assert_eq!(world_values(&r1), world_values(&r3));
        let (s1, s3) = (
            r1.exact_leaves.clone().expect("counted"),
            r3.exact_leaves.clone().expect("counted"),
        );
        assert_eq!(s1.counts(), s3.counts());
        let solved: u64 = s1.exact.iter().sum();
        assert!(
            solved > 0 && solved < s1.distinct.iter().sum::<u64>(),
            "{s1:?}"
        );
        // Leaf by leaf: exact exactly where the score reaches the threshold.
        let cfg = SearchConfig {
            tt_mb: 4,
            ..SearchConfig::default()
        };
        let mut seen = (0, 0);
        for (d, w) in ds.iter().flat_map(|d| [(d, 0), (d, 1)]) {
            let (_, view, legal) = d.replay();
            let (_, cands) = candidates(&view, &legal);
            let worlds = src.prepare(&view, ctx.rules);
            let m0 = world_match(&view, &*worlds, w, measure_seed(7, 1, d.id), ctx.rules);
            let x = ExactLeaves::new(
                &net,
                &*l,
                &p,
                engine::team_of(d.seat),
                solver::SolveFamily::of_match(&m0),
            );
            let mut run = WorldSolve::default();
            exact_world(
                &m0,
                &cands,
                &ctx,
                &cfg,
                None,
                &mut SolveMem::new(),
                &mut run,
                Some(FrontierCut::new(4, &x)),
            )
            .expect("solves");
            for (_, mctx, canon, _, exact) in x.leaves() {
                let nv = net.value(&canon, &mctx);
                let score = m.score(&canon, &mctx, &*l, &nv);
                assert_eq!(score, m.score(&canon, &mctx, &*l, &nv), "a pure function");
                assert!(score == 5 || score == 50);
                assert_eq!(exact, score >= 50);
                if exact {
                    seen.0 += 1;
                } else {
                    seen.1 += 1;
                }
            }
        }
        assert!(seen.0 > 0 && seen.1 > 0, "{seen:?}");
    }
}
