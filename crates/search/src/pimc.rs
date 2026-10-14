//! PIMC [Perfect-Information Monte Carlo: score each legal action in many worlds as if all cards
//! were face up, average, play the best] scored in money chance (docs/M3-SPEC.md §5.3–5.7).
//!
//! Two evaluators:
//! - **rollout-PIMC**: in each world, play the candidate, then every seat (the decider included)
//!   is played by `rules-v1` from its own view of the rebuilt world to the end of the hand; the end
//!   is valued by the leaf rule (§5.4). No seat sees another's cards inside a rollout.
//! - **exact PIMC**: in each world, the open-hand solver (`solve_fast`, or `solve_fast_par` with
//!   `solve_threads > 1`, identical answers) values every root card.
//!
//! Every candidate is valued over the **same** worlds (common random numbers). Worlds are worked
//! on in parallel into pre-sized slots and averaged in world order with `MoneyChance::mix`, so
//! the result is byte-identical for any thread count. Each worker thread keeps one
//! [`SolveMem`] (the solver's transposition tables) for all its worlds instead of allocating a
//! table per solve; that changes no value (see `SolveMem`).

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use bots::decide;
use engine::cards::{Card, bit, cards};
use engine::game::{Action, Match, Phase, Rules};
use engine::rules::{legal_burns, legal_plays};
use engine::team_of;
use engine::view::SeatView;
use equity::{MatchState, MoneyChance};
use serde::{Deserialize, Serialize};
use solver::{
    FastOptions, Leaf, ParProgress, SolveMem, Stats, from_match, solve_fast_par_with,
    solve_fast_watched,
};

use crate::worlds::{WorldSource, Worlds, rebuild, world_rng};

/// A leaf evaluator shared between threads.
pub type SharedLeaf = dyn Leaf + Send + Sync;

/// `&dyn Leaf` as a sized leaf for the solver's generic functions.
struct DynLeaf<'a>(&'a SharedLeaf);

impl Leaf for DynLeaf<'_> {
    fn value(&self, state: &MatchState) -> MoneyChance {
        self.0.value(state)
    }
}

/// What kind of decision is being made.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionKind {
    /// Play a card.
    Play,
    /// A burn with nothing played straight after (none at present: kept for the format).
    Burn,
    /// A robber's burn at its turn in trick 1 together with the card it then plays.
    JointBurnPlay,
    /// The dealer's burn after taking a turned ace or Joker (trick 0).
    DealerBurn,
}

/// How the values were computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Method {
    /// Rollouts by `rules-v1`.
    RolloutPimc,
    /// Exact open-hand solves.
    ExactPimc,
    /// A single legal action; nothing computed.
    Forced,
}

/// Which worlds are believed (docs/M3-SPEC.md §5.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeliefMode {
    /// Uniform over the worlds consistent with the hard facts.
    Uniform,
}

/// One candidate: an action, the card played straight after it (joint decisions only).
pub type Candidate = (Action, Option<Card>);

/// The recorded result of one decision (docs/M3-SPEC.md §5.7).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionValues {
    /// The decision's kind.
    pub kind: DecisionKind,
    /// How it was valued.
    pub method: Method,
    /// Worlds used.
    pub n_worlds: u32,
    /// The exact number of consistent worlds, when the world source counts them.
    pub world_count: Option<u128>,
    /// The belief.
    pub belief: BeliefMode,
    /// Every candidate with its money chance (average over the worlds), in candidate order.
    pub candidates: Vec<(Action, Option<Card>, MoneyChance)>,
    /// The chosen candidate's index.
    pub chosen: usize,
    /// Per candidate: the sample standard deviation over worlds of the decider's team entry, in
    /// units of 2^-32 (integer square root).
    pub own_sd: Vec<u64>,
}

/// The method for an early trick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Early {
    /// Rollout-PIMC.
    Rollout,
    /// Exact PIMC (the few-worlds arm at trick 2; the yardstick method at trick 1).
    Exact,
}

/// Search budgets and switches. Budgets are world counts, never time (A4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchConfig {
    /// Rollout worlds (tricks 0–2).
    pub w12: u32,
    /// Exact worlds (tricks 3–4).
    pub w34: u32,
    /// Trick-1 method, and its world count when exact.
    pub t1: Early,
    /// Worlds for trick 1 when exact.
    pub t1w: u32,
    /// Trick-2 method.
    pub t2: Early,
    /// Worlds for trick 2 when exact (the few-worlds arm).
    pub t2w: u32,
    /// Threads over worlds (never changes a result).
    pub threads: usize,
    /// Threads inside one exact solve (`solve_fast_par`; never changes a result).
    pub solve_threads: usize,
    /// Transposition-table megabytes per exact solve.
    pub tt_mb: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            w12: 64,
            w34: 32,
            t1: Early::Rollout,
            t1w: 20,
            t2: Early::Rollout,
            t2w: 8,
            threads: 1,
            solve_threads: 1,
            tt_mb: 64,
        }
    }
}

/// The candidates of the decision the view is waiting for, in candidate order (actions ascending
/// by card; pairs by (burn, play)), and the decision kind.
pub fn candidates(view: &SeatView, legal: &[Action]) -> (DecisionKind, Vec<Candidate>) {
    let p = &view.public;
    match p.phase {
        Phase::AwaitBurn(_) if p.trick_no == 1 => {
            let trump = p.trump.expect("trumps are known at a rob");
            let led = p.trick_cards.first().map(|&(_, c)| c);
            let mut out = Vec::new();
            for &a in legal {
                let b = a.card();
                let rest = view.hand & !bit(b);
                for c in cards(legal_plays(rest, led, trump)) {
                    out.push((a, Some(c)));
                }
            }
            (DecisionKind::JointBurnPlay, out)
        }
        Phase::AwaitBurn(_) => (
            DecisionKind::DealerBurn,
            legal.iter().map(|&a| (a, None)).collect(),
        ),
        _ => (
            DecisionKind::Play,
            legal.iter().map(|&a| (a, None)).collect(),
        ),
    }
}

/// The method and world count the configuration uses for this view's decision.
pub fn method_for(view: &SeatView, n_cands: usize, cfg: &SearchConfig) -> (Method, u32) {
    if n_cands <= 1 {
        return (Method::Forced, 0);
    }
    let p = &view.public;
    let dealer_burn = matches!(p.phase, Phase::AwaitBurn(_)) && p.trick_no == 0;
    match p.trick_no {
        _ if dealer_burn => (Method::RolloutPimc, cfg.w12),
        0 | 1 => match cfg.t1 {
            Early::Rollout => (Method::RolloutPimc, cfg.w12),
            Early::Exact => (Method::ExactPimc, cfg.t1w),
        },
        2 => match cfg.t2 {
            Early::Rollout => (Method::RolloutPimc, cfg.w12),
            Early::Exact => (Method::ExactPimc, cfg.t2w),
        },
        _ => (Method::ExactPimc, cfg.w34),
    }
}

/// What the evaluators need besides the view.
#[derive(Clone, Copy)]
pub struct Ctx<'a> {
    /// The leaf (normally the match-equity table).
    pub leaf: &'a SharedLeaf,
    /// The world generator.
    pub source: &'a dyn WorldSource,
    /// The rule switches.
    pub rules: Rules,
}

/// The value of a match once the current hand is over (docs/M3-SPEC.md §5.4).
fn leaf_now(m: &Match, leaf: &SharedLeaf) -> MoneyChance {
    if let Some(money) = m.money() {
        let t = (0..4u8)
            .max_by_key(|&t| (money[usize::from(t)], core::cmp::Reverse(t)))
            .expect("four teams");
        return MoneyChance::certain(t);
    }
    leaf.value(&MatchState::from_public(&m.public()))
}

/// Play the hand out with `rules-v1` at every seat, each from its own view, and value the end.
pub fn rollout(m: &mut Match, leaf: &SharedLeaf, rules: Rules) -> MoneyChance {
    let h = m.hand_index();
    while m.phase() != Phase::MatchOver && m.hand_index() == h {
        let seat = m.to_act().expect("someone acts");
        let view = m.view(seat);
        let legal = m.legal_actions();
        let a = if legal.len() == 1 {
            legal[0]
        } else {
            decide(&view, &rules, &legal).0
        };
        m.apply(a).expect("rules-v1 plays legal actions");
    }
    leaf_now(m, leaf)
}

fn apply_candidate(m: &mut Match, cand: Candidate) {
    m.apply(cand.0).expect("candidate is legal");
    if let Some(c) = cand.1 {
        m.apply(Action::Play(c)).expect("joint play is legal");
    }
}

/// The rollout value of every candidate in one rebuilt world.
fn rollout_world(m: &Match, cands: &[Candidate], ctx: &Ctx) -> Vec<MoneyChance> {
    cands
        .iter()
        .map(|&cand| {
            let mut mm = m.clone();
            apply_candidate(&mut mm, cand);
            rollout(&mut mm, ctx.leaf, ctx.rules)
        })
        .collect()
}

/// The solver position of a match awaiting a play, with a future robber's rob resolved: the
/// robber's burn is the inner player's (`rules-v1`) burn, which depends only on the robber's own
/// cards and the trump suit, so it is found by playing the hand on with `rules-v1` up to the rob.
fn solver_position(m: &Match, rules: Rules) -> (solver::Position, solver::MatchCtx) {
    let (mut pos, mctx, pending) = from_match(m).expect("awaiting a play with trumps known");
    if let Some(s) = pending {
        let table = m.public().table_card.expect("turned card on the table");
        let mut mm = m.clone();
        let burn = loop {
            match mm.phase() {
                Phase::AwaitBurn(r) if r == s => {
                    let view = mm.view(r);
                    let legal = mm.legal_actions();
                    break decide(&view, &rules, &legal).0.card();
                }
                Phase::AwaitPlay(r) | Phase::AwaitBurn(r) => {
                    let view = mm.view(r);
                    let legal = mm.legal_actions();
                    let a = decide(&view, &rules, &legal).0;
                    mm.apply(a).expect("legal");
                }
                Phase::MatchOver => unreachable!("trick 1 cannot end the match before the rob"),
            }
        };
        let h = pos.hands[usize::from(s)] | bit(table);
        debug_assert!(engine::cards::has(legal_burns(h, Some(pos.trump)), burn));
        pos.hands[usize::from(s)] = h & !bit(burn);
    }
    (pos, mctx)
}

/// Counters of the exact solves of one world (read by the trick-1 bench; they never change a
/// value).
#[derive(Clone, Copy, Debug, Default)]
pub struct WorldSolve {
    /// Counters summed over the world's solves (one solve, or one per burn of a joint decision);
    /// `tt_entries` and `tt_bytes` are the largest of any one solve.
    pub stats: Stats,
    /// Solves made.
    pub solves: u32,
}

impl WorldSolve {
    fn add(&mut self, b: &Stats) {
        let a = &mut self.stats;
        a.nodes += b.nodes;
        a.leaves += b.leaves;
        a.decision_nodes += b.decision_nodes;
        a.tied_nodes += b.tied_nodes;
        a.tt_probes += b.tt_probes;
        a.tt_hits += b.tt_hits;
        a.tt_entries = a.tt_entries.max(b.tt_entries);
        a.tt_replaced += b.tt_replaced;
        a.cutoffs += b.cutoffs;
        a.equiv_skipped += b.equiv_skipped;
        a.tt_bytes = a.tt_bytes.max(b.tt_bytes);
        self.solves += 1;
    }
}

/// One exact solve; `None` if it was stopped through `par`'s `abort` flag before it finished
/// (its counters, the nodes searched so far included, still go into `run`).
fn solve(
    pos: &solver::Position,
    mctx: solver::MatchCtx,
    ctx: &Ctx,
    cfg: &SearchConfig,
    par: Option<&ParProgress>,
    mem: &mut SolveMem,
    run: &mut WorldSolve,
) -> Option<Vec<(Card, MoneyChance)>> {
    let opts = FastOptions {
        tt_mb: cfg.tt_mb,
        ..FastOptions::default()
    };
    let leaf = DynLeaf(ctx.leaf);
    let sol = if cfg.solve_threads > 1 {
        // The progress counters describe the solve now running (read by progress reports only).
        if let Some(p) = par {
            p.tasks_total.store(0, Ordering::Relaxed);
            p.tasks_done.store(0, Ordering::Relaxed);
        }
        solve_fast_par_with(pos, mctx, &leaf, opts, cfg.solve_threads, par, mem)
    } else {
        // `par` here only carries a stop flag and live counters (planned runs); the answer is
        // `solve_fast_with`'s.
        solve_fast_watched(pos, mctx, &leaf, opts, mem, par)
    };
    run.add(&sol.stats);
    (!sol.timed_out).then_some(sol.children)
}

fn pick(children: &[(Card, MoneyChance)], c: Card) -> MoneyChance {
    children
        .iter()
        .find(|&&(x, _)| x == c)
        .map(|&(_, v)| v)
        .expect("every legal root card is valued")
}

/// The exact value of every candidate in one rebuilt world (`None` if stopped through `par`).
fn exact_world(
    m: &Match,
    cands: &[Candidate],
    ctx: &Ctx,
    cfg: &SearchConfig,
    par: Option<&ParProgress>,
    mem: &mut SolveMem,
    run: &mut WorldSolve,
) -> Option<Vec<MoneyChance>> {
    let mut out = vec![MoneyChance::even(); cands.len()];
    let mut i = 0;
    while i < cands.len() {
        match cands[i] {
            (Action::Play(_), None) => {
                // One solve values every root card.
                let (pos, mctx) = solver_position(m, ctx.rules);
                let ch = solve(&pos, mctx, ctx, cfg, par, mem, run)?;
                for (j, &(a, _)) in cands.iter().enumerate() {
                    out[j] = pick(&ch, a.card());
                }
                return Some(out);
            }
            (Action::Burn(b), Some(_)) => {
                // One solve per burn values every play after it.
                let mut mm = m.clone();
                mm.apply(Action::Burn(b)).expect("legal burn");
                let (pos, mctx) = solver_position(&mm, ctx.rules);
                let ch = solve(&pos, mctx, ctx, cfg, par, mem, run)?;
                while i < cands.len() && cands[i].0 == Action::Burn(b) {
                    out[i] = pick(&ch, cands[i].1.expect("joint"));
                    i += 1;
                }
            }
            (a, _) => {
                // A dealer's burn: not an exact-mode decision; valued by rollout.
                let mut mm = m.clone();
                mm.apply(a).expect("legal");
                out[i] = rollout(&mut mm, ctx.leaf, ctx.rules);
                i += 1;
            }
        }
    }
    Some(out)
}

/// The exact solves one world of a decision takes: one, or one per burn of a robber's joint
/// (burn, first card) decision (the scheduler's cost estimate).
pub fn solves_per_world(cands: &[Candidate]) -> u32 {
    let mut burns: Vec<Action> = cands
        .iter()
        .filter(|c| c.1.is_some())
        .map(|c| c.0)
        .collect();
    burns.dedup();
    (burns.len() as u32).max(1)
}

/// [`world_row`] for exact PIMC, with a stop flag (`par`'s `abort`) and the solves' counters
/// (`run`): the values of every candidate in world `w`, exactly [`world_row`]'s, or `None` if it
/// was stopped first.
#[allow(clippy::too_many_arguments)]
pub fn exact_world_row(
    view: &SeatView,
    worlds: &dyn Worlds,
    cands: &[Candidate],
    w: u64,
    dseed: u64,
    ctx: &Ctx,
    cfg: &SearchConfig,
    par: Option<&ParProgress>,
    mem: &mut SolveMem,
    run: &mut WorldSolve,
) -> Option<Vec<MoneyChance>> {
    let world = worlds.sample(&mut world_rng(dseed, w));
    let m = rebuild(view, &world, ctx.rules, dseed ^ w).expect("sampled worlds rebuild");
    exact_world(&m, cands, ctx, cfg, par, mem, run)
}

/// The values of every candidate in world `w` of the decision with seed `dseed` (`worlds`
/// prepared from `view`). `par` receives the progress of a parallel exact solve
/// (`solve_threads > 1`); it never changes a value. `mem` is the calling thread's solver memory,
/// kept from world to world (what it held before never changes a value).
#[allow(clippy::too_many_arguments)]
pub fn world_row(
    view: &SeatView,
    worlds: &dyn Worlds,
    cands: &[Candidate],
    method: Method,
    w: u64,
    dseed: u64,
    ctx: &Ctx,
    cfg: &SearchConfig,
    par: Option<&ParProgress>,
    mem: &mut SolveMem,
) -> Vec<MoneyChance> {
    let world = worlds.sample(&mut world_rng(dseed, w));
    let m = rebuild(view, &world, ctx.rules, dseed ^ w).expect("sampled worlds rebuild");
    match method {
        Method::RolloutPimc => rollout_world(&m, cands, ctx),
        Method::ExactPimc => exact_world(&m, cands, ctx, cfg, par, mem, &mut WorldSolve::default())
            .expect("nothing stops a solve here"),
        Method::Forced => vec![MoneyChance::even(); cands.len()],
    }
}

/// The values of every candidate in worlds `from..to` of the decision with seed `dseed`, one row
/// per world in world order. Thread count never changes the result.
#[allow(clippy::too_many_arguments)]
pub fn world_values(
    view: &SeatView,
    cands: &[Candidate],
    method: Method,
    from: u64,
    to: u64,
    dseed: u64,
    ctx: &Ctx,
    cfg: &SearchConfig,
) -> Vec<Vec<MoneyChance>> {
    let worlds = ctx.source.prepare(view, ctx.rules);
    let n = (to - from) as usize;
    let slots: Mutex<Vec<Option<Vec<MoneyChance>>>> = Mutex::new(vec![None; n]);
    let next = AtomicUsize::new(0);
    let work = || {
        // One solver memory per worker thread, kept for all its worlds.
        let mut mem = SolveMem::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            let w = from + i as u64;
            let v = world_row(
                view, &*worlds, cands, method, w, dseed, ctx, cfg, None, &mut mem,
            );
            slots.lock().unwrap_or_else(|e| e.into_inner())[i] = Some(v);
        }
    };
    let threads = cfg.threads.max(1).min(n.max(1));
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
        .map(|v| v.expect("every world valued"))
        .collect()
}

/// Integer square root (floor).
pub fn isqrt(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    // Newton's method from above, in integers only.
    let mut x = 1u128 << (128 - n.leading_zeros()).div_ceil(2);
    loop {
        let y = (x + n / x) / 2;
        if y >= x {
            return x;
        }
        x = y;
    }
}

/// Average rows (worlds) of candidate values in world order; the choice; own-team SDs.
pub fn aggregate(
    rows: &[Vec<MoneyChance>],
    n_cands: usize,
    team: u8,
) -> (Vec<MoneyChance>, usize, Vec<u64>) {
    let mut means = Vec::with_capacity(n_cands);
    let mut sds = Vec::with_capacity(n_cands);
    let t = usize::from(team);
    for j in 0..n_cands {
        let terms: Vec<(u64, MoneyChance)> = rows.iter().map(|r| (1, r[j])).collect();
        means.push(MoneyChance::mix(&terms));
        let n = rows.len() as u128;
        let (s, s2) = rows.iter().fold((0u128, 0u128), |(a, b), r| {
            let x = u128::from(r[j].0[t]);
            (a + x, b + x * x)
        });
        let sd = if n >= 2 {
            isqrt((n * s2 - s * s) / (n * (n - 1)))
        } else {
            0
        };
        sds.push(sd as u64);
    }
    (means.clone(), best_index(&means, team), sds)
}

/// The candidate with the largest entry for `team`; ties to the first.
pub fn best_index(values: &[MoneyChance], team: u8) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if v.get(team) > values[best].get(team) {
            best = i;
        }
    }
    best
}

/// Decide with PIMC: the candidates, their values over `n` worlds of the configured method, and
/// the choice. A single candidate is returned at once with no world drawn (§6.8).
pub fn evaluate(
    view: &SeatView,
    legal: &[Action],
    dseed: u64,
    ctx: &Ctx,
    cfg: &SearchConfig,
) -> DecisionValues {
    let (kind, cands) = candidates(view, legal);
    let (method, n) = method_for(view, cands.len(), cfg);
    if method == Method::Forced {
        return DecisionValues {
            kind,
            method,
            n_worlds: 0,
            world_count: None,
            belief: BeliefMode::Uniform,
            candidates: cands
                .iter()
                .map(|&(a, p)| (a, p, MoneyChance::even()))
                .collect(),
            chosen: 0,
            own_sd: vec![0; cands.len()],
        };
    }
    // The exact number of consistent worlds, when the source counts them (for records and the
    // world-count bands; preparing again costs a fraction of a millisecond).
    let world_count = ctx.source.prepare(view, ctx.rules).count();
    let rows = world_values(view, &cands, method, 0, u64::from(n), dseed, ctx, cfg);
    let (means, chosen, own_sd) = aggregate(&rows, cands.len(), team_of(view.seat));
    DecisionValues {
        kind,
        method,
        n_worlds: n,
        world_count,
        belief: BeliefMode::Uniform,
        candidates: cands
            .iter()
            .zip(means)
            .map(|(&(a, p), v)| (a, p, v))
            .collect(),
        chosen,
        own_sd,
    }
}
