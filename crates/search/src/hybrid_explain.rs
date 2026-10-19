//! Why the hybrid H_k and exact search choose differently on a decision (`search hybrid-check
//! --explain IDS`, docs/M3-HYBRID-SPEC.md §3). A diagnostic only: it changes no value and no
//! verdict of the check.
//!
//! For each explained decision and each of its worlds (regenerated exactly as the check draws
//! them), every candidate is searched four ways on the same world:
//! - **exact** (the plain exact solve, the check's exact values);
//! - **exact to the frontier** (the hybrid's search with the exact value at every frontier
//!   position, `crate::oracle_labels::Driver::Exact`): it must give exactly the plain exact
//!   values (otherwise the explanation stops with an error), and it names, per candidate, the
//!   frontier position(s) whose value the exact search copied up to the root;
//! - **hybrid** (the network at the frontier, `Driver::Net`): the check's hybrid values, and per
//!   candidate the frontier position(s) whose network value the hybrid copied up to the root;
//! - **exact, reverse tie order** (`solver::TieOrder::Descending`: a seat choosing between
//!   equally good cards takes the last by card index instead of the first). The documented tie
//!   rule decides which card an *indifferent* seat plays, so where the two orders differ the
//!   exact value rests on that convention, not on anyone's interest (a tie-decided value).
//!
//! A max^n backup only copies child vectors, so a candidate's value in a world is the value of
//! one frontier position on its principal line (or of a leaf before the frontier, when the game
//! ends earlier). "Principal positions" are the reached frontier positions whose value equals
//! the candidate's value exactly, preferring those reached after the candidate's own card (an
//! equivalent card may have been searched in its place); several may share the vector.
//!
//! Every reached frontier position of the hybrid's search is also valued exactly (both tie
//! orders), so the network's error is measured where the hybrid asks it (frontier error), split
//! into tie-decided positions and the others. Per world, the hybrid's **misranking** is
//! `(H[h] − H[e]) − (E[h] − E[e])` in the decider's team's chance, with `h` the hybrid's choice
//! and `e` exact's over all the decision's worlds: the worlds whose misranking is largest are
//! the ones that mislead the hybrid.
//!
//! Values are fixed point; floats appear only in the report (percentages, the network's outcome
//! probabilities). The result is the same for any thread count.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use engine::cards::{Card, bit, name, names};
use engine::game::{Action, Match};
use equity::{MoneyChance, ONE};
use nn::Oracle;
use nn::encode::encode;
use nn::outcome::{GAME_WON, compositions, outcomes};
use serde::Serialize;
use solver::frontier::{CanonPos, Frontier};
use solver::{
    FastOptions, FrontierCut, MatchCtx, Position, SolveMem, TieOrder, from_match, solve_fast_with,
};

use crate::measure::{DecisionSpec, measure_seed};
use crate::oracle_labels::{Driver, Reached, Recorder, exact_value, tt_mb_for};
use crate::pimc::{
    Candidate, Ctx, DynLeaf, SearchConfig, SharedLeaf, WorldSolve, aggregate, best_index,
    candidates, exact_world, world_match,
};

/// One explanation run.
#[derive(Clone, Debug)]
pub struct ExplainJob {
    /// The frontier trick.
    pub cut: u8,
    /// Master seed of the worlds (as the yardstick: 1).
    pub seed: u64,
    /// World label (1 = the yardstick's).
    pub label: u64,
    /// Worlds per decision (`0..to`).
    pub to: u64,
    /// Worker threads (never changes a value).
    pub threads: usize,
    /// Transposition-table megabytes per root solve.
    pub tt_mb: usize,
}

/// A frontier position on a candidate's principal line in one world.
#[derive(Clone, Debug, Serialize)]
pub struct LineEnd {
    /// The position, rendered (seats absolute; `*` marks the decider).
    pub position: String,
    /// The network's value of it, absolute team order, units of 2^-32 (the value the hybrid
    /// copied, on a hybrid line).
    pub net: [u64; 4],
    /// Its exact value (the value exact search copied, on an exact line).
    pub exact: [u64; 4],
    /// Its exact value under the reverse tie order.
    pub exact_reverse: [u64; 4],
    /// Its race class ([`race_class`]).
    pub class: u8,
    /// Reached positions of the world with this same vector (more than one: ambiguous).
    pub ties: usize,
    /// Whether it was reached after the candidate's own card (else after an equivalent card).
    pub under: bool,
    /// The network's most likely outcomes there with their probabilities (empty without a
    /// network).
    pub outcomes: Vec<(String, f64)>,
    /// The outcome the exact value is (the label the network is trained on).
    pub label: Option<String>,
}

/// One world of an explained decision.
#[derive(Clone, Debug, Serialize)]
pub struct WorldExplain {
    /// World index.
    pub world: u64,
    /// Exact values per candidate (absolute team order).
    pub exact: Vec<[u64; 4]>,
    /// Exact values per candidate under the reverse tie order.
    pub exact_reverse: Vec<[u64; 4]>,
    /// Hybrid values per candidate.
    pub hybrid: Vec<[u64; 4]>,
    /// The hybrid's misranking here, decider's team, units of 2^-32 (module notes).
    pub misrank: i64,
    /// Per candidate: the hybrid's principal frontier position (`None`: the value is not a
    /// frontier position's, the line ends before the frontier).
    pub hybrid_line: Vec<Option<LineEnd>>,
    /// Per candidate: the exact search's principal frontier position.
    pub exact_line: Vec<Option<LineEnd>>,
    /// Distinct frontier positions the hybrid's search reached.
    pub reached: usize,
    /// Of those, how many the network values wrongly for the decider's team by more than 1%.
    pub reached_wrong_1pct: usize,
    /// Of those, how many are tie-decided (exact value differs between the tie orders).
    pub reached_tie_decided: usize,
    /// Of those, how many in each race class ([`race_class`]).
    pub reached_by_class: [usize; N_CLASSES],
    /// The race class of the most contested of the four principal positions (the hybrid's and
    /// exact's lines of the hybrid's and exact's choices); `None` if none is a frontier
    /// position.
    pub line_class: Option<u8>,
    /// Per reached position of the hybrid: its race class, whether it is tie-decided, and the
    /// network's error per team relative to the decider (units of 2^-32).
    #[serde(skip)]
    errors: Vec<(u8, bool, [i64; 4], [bool; 4])>,
}

/// Error statistics (fractions of one money chance).
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ErrStats {
    /// Values.
    pub n: usize,
    /// Mean (bias).
    pub mean: f64,
    /// Standard deviation.
    pub sd: f64,
    /// Mean absolute error.
    pub mean_abs: f64,
    /// 99.9th percentile of the absolute error.
    pub p999_abs: f64,
    /// Largest absolute error.
    pub max_abs: f64,
}

/// Statistics of errors in units of 2^-32.
pub fn err_stats(e: &[i64]) -> ErrStats {
    if e.is_empty() {
        return ErrStats::default();
    }
    let unit = ONE as f64;
    let m = e.len() as f64;
    let mean = e.iter().map(|&x| x as f64).sum::<f64>() / m;
    let var = e.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / m;
    let mut abs: Vec<u64> = e.iter().map(|x| x.unsigned_abs()).collect();
    abs.sort_unstable();
    let p = abs[((abs.len() as f64 * 0.999).ceil() as usize).clamp(1, abs.len()) - 1];
    ErrStats {
        n: e.len(),
        mean: mean / unit,
        sd: var.sqrt() / unit,
        mean_abs: abs.iter().map(|&x| x as f64).sum::<f64>() / m / unit,
        p999_abs: p as f64 / unit,
        max_abs: *abs.last().expect("not empty") as f64 / unit,
    }
}

/// Race classes ([`race_class`]).
pub const N_CLASSES: usize = nn::outcome::N_RACE_CLASSES;

/// The race class of a frontier position ([`nn::outcome::race_class`]): what can be decided in
/// this hand, from nobody being able to win the game (0) to a team able to win the match while
/// another can win the game (4).
pub fn race_class(c: &CanonPos, ctx: &MatchCtx, leaf: &SharedLeaf) -> u8 {
    race_info(c, ctx, leaf).0
}

/// [`race_class`] and the teams (absolute order) that can win the match in this hand.
pub fn race_info(c: &CanonPos, ctx: &MatchCtx, leaf: &SharedLeaf) -> (u8, [bool; 4]) {
    let mask = outcomes(c, ctx, leaf).mask;
    let r = usize::from(nn::encode::rotation(c) % 4);
    // The outcomes are in the encoder's rotated order: rotated team u is real team u − r.
    let rotated_games: [bool; 4] = core::array::from_fn(|u| ctx.games[(u + 4 - r) % 4] >= 1);
    let money: [bool; 4] = core::array::from_fn(|t| {
        let u = (t + r) % 4;
        mask >> u & 1 == 1 && ctx.games[t] >= 1
    });
    (nn::outcome::race_class(mask, rotated_games), money)
}

/// Names of the race classes.
pub const RACE_CLASSES: [&str; N_CLASSES] = nn::outcome::RACE_CLASS_NAMES;

/// Network error at frontier positions: all four teams, and the decider's team alone; over every
/// position, over the tie-decided ones and over the others, and per race class (all teams).
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct FrontierError {
    /// Every position, all teams.
    pub all: ErrStats,
    /// Every position, the decider's team.
    pub decider: ErrStats,
    /// Tie-decided positions, all teams.
    pub tie_decided: ErrStats,
    /// The other positions, all teams.
    pub plain: ErrStats,
    /// Per race class ([`race_class`]), all teams.
    pub by_class: [ErrStats; N_CLASSES],
    /// Per race class, the entries of the teams that can win the match in this hand (the public
    /// enemy): a positive mean means the network overrates them.
    pub enemy: [ErrStats; N_CLASSES],
}

impl FrontierError {
    fn of<'a>(ws: impl Iterator<Item = &'a WorldExplain>) -> FrontierError {
        let (mut all, mut own, mut tie, mut plain) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut by: [Vec<i64>; N_CLASSES] = Default::default();
        let mut enemy: [Vec<i64>; N_CLASSES] = Default::default();
        for &(class, tied, e, m) in ws.flat_map(|w| w.errors.iter()) {
            all.extend(e);
            own.push(e[0]);
            if tied { &mut tie } else { &mut plain }.extend(e);
            by[usize::from(class)].extend(e);
            enemy[usize::from(class)].extend((0..4).filter(|&k| m[k]).map(|k| e[k]));
        }
        FrontierError {
            all: err_stats(&all),
            decider: err_stats(&own),
            tie_decided: err_stats(&tie),
            plain: err_stats(&plain),
            by_class: core::array::from_fn(|k| err_stats(&by[k])),
            enemy: core::array::from_fn(|k| err_stats(&enemy[k])),
        }
    }
}

/// An explained decision.
#[derive(Clone, Debug, Serialize)]
pub struct DecisionExplain {
    /// Decision id.
    pub id: u64,
    /// The decision, rendered: match state, the hand so far, the decider's cards.
    pub header: Vec<String>,
    /// The decider's seat.
    pub seat: u8,
    /// The decider's team.
    pub team: u8,
    /// Position in the trick (1 = leader … 8).
    pub position: u8,
    /// Candidate names.
    pub cands: Vec<String>,
    /// Exact means per candidate (absolute team order).
    pub exact_mean: Vec<[u64; 4]>,
    /// Exact means per candidate under the reverse tie order.
    pub exact_mean_reverse: Vec<[u64; 4]>,
    /// Hybrid means per candidate.
    pub hybrid_mean: Vec<[u64; 4]>,
    /// Exact's choice.
    pub exact_choice: usize,
    /// Exact's choice under the reverse tie order.
    pub exact_choice_reverse: usize,
    /// The hybrid's choice.
    pub hybrid_choice: usize,
    /// Loss of the hybrid's choice (decider's team, units of 2^-32).
    pub loss: u64,
    /// Loss of the hybrid's choice measured with the reverse tie order's exact means.
    pub loss_reverse: u64,
    /// Sum of the worlds' misrankings (units of 2^-32).
    pub misrank_total: i64,
    /// The part of it from worlds where a principal position or a root value of the hybrid's or
    /// exact's choice is tie-decided.
    pub misrank_tie_worlds: i64,
    /// The misranking by the worlds' line class ([`WorldExplain::line_class`]; the last index:
    /// no frontier position on those lines).
    pub misrank_by_class: [i64; N_CLASSES + 1],
    /// Worlds pushing towards the hybrid's choice, and against it.
    pub misrank_worlds: (usize, usize),
    /// Network error at every distinct frontier position the hybrid reached, per world.
    pub frontier_error: FrontierError,
    /// Every world, in world order.
    pub worlds: Vec<WorldExplain>,
}

fn arr(v: &MoneyChance) -> [u64; 4] {
    core::array::from_fn(|t| v.get(t as u8))
}

/// A frontier position (or any trick-boundary position) as one line of text.
pub fn render_position(p: &Position, ctx: &MatchCtx, decider: u8) -> String {
    let hands: Vec<String> = (0..8u8)
        .map(|s| {
            format!(
                "s{s}{}(t{}): {}",
                if s == decider { "*" } else { "" },
                s % 4,
                names(p.hands[usize::from(s)])
            )
        })
        .collect();
    let best = p
        .best
        .map_or("none".to_string(), |(r, t)| format!("rank {r} by team {t}"));
    format!(
        "trick {} led by s{}; trumps {:?}; scores {:?}; tricks this hand {:?}; bonus taken {:?}; \
         5 played {}; best trump so far {best}; games {:?}; dealer team {} | {}",
        p.hand.tricks_done + 1,
        p.leader,
        p.trump,
        p.hand.scores,
        p.hand.tricks,
        p.hand.flags,
        p.hand.five_played,
        ctx.games,
        ctx.dealer_team,
        hands.join(" | ")
    )
}

/// The name of outcome `o` of a canonical position (absolute teams).
fn outcome_name(c: &CanonPos, o: usize) -> String {
    let r = usize::from(nn::encode::rotation(c) % 4);
    if o < GAME_WON {
        format!("team {} wins the game", (o + 4 - r) % 4)
    } else {
        let n4 = compositions(engine::game::TRICKS_PER_HAND - c.tricks_done)[o - GAME_WON];
        let n: [u8; 4] = core::array::from_fn(|t| n4[(t + r) % 4]);
        format!("hand ends, tricks left by team {n:?}")
    }
}

/// The network's outcome probabilities at `c` (most likely first, at most `top`) and the label
/// of the exact value `v`.
fn outcome_report(
    net: &Oracle,
    c: &CanonPos,
    ctx: &MatchCtx,
    leaf: &SharedLeaf,
    v: &MoneyChance,
    top: usize,
) -> (Vec<(String, f64)>, Option<String>) {
    let (f, r) = encode(c, ctx, leaf);
    let o = outcomes(c, ctx, leaf);
    let label = o
        .label(&v.rotate(r % 4))
        .map(|l| outcome_name(c, usize::from(l)));
    let z = net.net().forward(&f);
    let live: Vec<usize> = (0..o.n).filter(|&i| o.mask >> i & 1 == 1).collect();
    let m = live
        .iter()
        .map(|&i| z[i])
        .max()
        .expect("a possible outcome");
    let w: Vec<(usize, u64)> = live
        .iter()
        .map(|&i| (i, nn::money::exp_weight((m - z[i]) as u64)))
        .collect();
    let total: f64 = w.iter().map(|&(_, x)| x as f64).sum();
    let mut p: Vec<(String, f64)> = w
        .iter()
        .map(|&(i, x)| (outcome_name(c, i), x as f64 / total))
        .collect();
    p.sort_by(|a, b| b.1.total_cmp(&a.1));
    p.truncate(top);
    (p, label)
}

/// A principal position: its index in the reached list, how many reached positions share its
/// vector, and whether it was reached after the candidate's own card.
type Principal = (usize, usize, bool);

/// The principal frontier position of a candidate valued `v`: the first reached position with
/// exactly that value, preferring those after the candidate's own card.
fn principal(list: &[Reached], v: &MoneyChance, seat: u8, card: Card) -> Option<Principal> {
    let hits: Vec<usize> = (0..list.len()).filter(|&i| list[i].value == *v).collect();
    let under = |i: &usize| list[*i].pos.hands[usize::from(seat)] & bit(card) == 0;
    let first_under = hits.iter().find(|i| under(i)).copied();
    first_under
        .map(|i| (i, hits.len(), true))
        .or_else(|| hits.first().map(|&i| (i, hits.len(), false)))
}

/// The decision rendered: match state, the hand so far, the decider's cards and candidates.
pub fn render_decision(d: &DecisionSpec, cands: &[Candidate]) -> Vec<String> {
    let mut deck = [0u8; engine::cards::DECK_SIZE];
    deck.copy_from_slice(&d.deck);
    let mut m =
        Match::from_hand_start(d.start, deck, 0, engine::game::Rules::HOUSE).expect("valid");
    let mut out = vec![format!(
        "decision {}: trick {}, seat {} (team {}), position {} in the trick, stratum {}; \
         dealer seat {} (team {}); start scores {:?}, bonus taken {:?}, games won {:?}",
        d.id,
        d.trick,
        d.seat,
        d.seat % 4,
        d.position + 1,
        d.stratum,
        d.start.dealer,
        d.start.dealer % 4,
        d.start.scores,
        d.start.bonus_taken,
        d.start.games_won
    )];
    let p0 = m.public();
    out.push(format!(
        "trumps {:?}; turned {}",
        p0.trump,
        p0.turned
            .iter()
            .map(|&(c, s)| match s {
                Some(s) => format!("{} (taken by s{s})", name(c)),
                None => format!("{} (not taken)", name(c)),
            })
            .collect::<Vec<_>>()
            .join(", ")
    ));
    let mut line = String::new();
    for &a in &d.actions {
        let s = m.to_act().expect("someone acts");
        let p = m.public();
        // A robber leading trick 1 burns, then plays: one trick.
        if line.is_empty() || (p.trick_cards.is_empty() && !line.ends_with("burns)")) {
            if !line.is_empty() {
                out.push(line);
            }
            line = format!("trick {}:", p.trick_no);
        }
        match a {
            Action::Play(c) => line.push_str(&format!(" s{s} {}", name(c))),
            Action::Burn(_) => line.push_str(&format!(" (s{s} burns)")),
        }
        m.apply(a).expect("recorded actions are legal");
        let q = m.public();
        if q.trick_winners.len() > p.trick_winners.len() {
            line.push_str(&format!(
                " -> won by s{}; scores {:?}",
                q.trick_winners.last().expect("a winner"),
                q.scores
            ));
        }
    }
    if !line.is_empty() {
        out.push(line);
    }
    let v = m.view(d.seat);
    out.push(format!(
        "decider s{} holds {}; candidates {}",
        d.seat,
        names(v.hand),
        cands
            .iter()
            .map(|c| name(c.0.card()))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    out
}

struct Prep<'a> {
    d: &'a DecisionSpec,
    view: engine::view::SeatView,
    cands: Vec<Candidate>,
    dseed: u64,
}

/// One world's searches (the parallel part of [`explain`]).
struct WorldRaw {
    exact: Vec<MoneyChance>,
    exact_reverse: Vec<MoneyChance>,
    hybrid: Vec<MoneyChance>,
    /// The exact search's reached positions (exact values).
    re: Vec<Reached>,
    /// The hybrid's reached positions (network values), their exact values in both tie orders.
    rh: Vec<Reached>,
    rh_exact: Vec<MoneyChance>,
    rh_reverse: Vec<MoneyChance>,
    rh_class: Vec<(u8, [bool; 4])>,
    /// Per candidate, the exact search's principal position and its value under the reverse
    /// tie order.
    re_principal: Vec<Option<(Principal, MoneyChance)>>,
}

fn world_raw(
    p: &Prep,
    m: &Match,
    job: &ExplainJob,
    ctx: &Ctx,
    net: &dyn Frontier,
    mem: &mut SolveMem,
) -> Result<WorldRaw, String> {
    let cfg = SearchConfig {
        tt_mb: job.tt_mb,
        ..SearchConfig::default()
    };
    let opts = FastOptions {
        tt_mb: tt_mb_for(job.cut),
        ..FastOptions::default()
    };
    let reverse = FastOptions {
        order: TieOrder::Descending,
        ..opts
    };
    let mut run = WorldSolve::default();
    let plain = exact_world(m, &p.cands, ctx, &cfg, None, mem, &mut run, None)
        .expect("nothing stops a solve here");
    let rec_e = Recorder::new(ctx.leaf, opts, Driver::Exact);
    let cut_e = FrontierCut::new(job.cut, &rec_e);
    let exact = exact_world(m, &p.cands, ctx, &cfg, None, mem, &mut run, Some(cut_e))
        .expect("nothing stops a solve here");
    if exact != plain {
        return Err(format!(
            "decision {}: the search with exact frontier values differs from the plain exact \
             solve",
            p.d.id
        ));
    }
    let rec_h = Recorder::new(ctx.leaf, opts, Driver::Net(net));
    let cut_h = FrontierCut::new(job.cut, &rec_h);
    let hybrid = exact_world(m, &p.cands, ctx, &cfg, None, mem, &mut run, Some(cut_h))
        .expect("nothing stops a solve here");
    // The root under the reverse tie order (no rob is pending after trick 1).
    let (pos, mctx, pending) = from_match(m).expect("a play is awaited");
    debug_assert!(pending.is_none());
    let root = solve_fast_with(
        &pos,
        mctx,
        &DynLeaf(ctx.leaf),
        FastOptions {
            tt_mb: job.tt_mb,
            order: TieOrder::Descending,
            ..FastOptions::default()
        },
        mem,
    );
    let exact_reverse = p
        .cands
        .iter()
        .map(|c| {
            root.children
                .iter()
                .find(|&&(x, _)| x == c.0.card())
                .map(|&(_, v)| v)
                .expect("every candidate is a root card")
        })
        .collect();
    let re = rec_e.into_reached().0;
    let rh = rec_h.into_reached().0;
    let known: BTreeMap<CanonPos, MoneyChance> = re.iter().map(|x| (x.canon, x.value)).collect();
    let rh_exact = rh
        .iter()
        .map(|r| {
            known
                .get(&r.canon)
                .copied()
                .unwrap_or_else(|| exact_value(&r.pos, r.ctx, ctx.leaf, opts, mem).0)
        })
        .collect();
    let rh_reverse = rh
        .iter()
        .map(|r| exact_value(&r.pos, r.ctx, ctx.leaf, reverse, mem).0)
        .collect();
    let rh_class = rh
        .iter()
        .map(|r| race_info(&r.canon, &r.ctx, ctx.leaf))
        .collect();
    let re_principal = p
        .cands
        .iter()
        .zip(&exact)
        .map(|(c, v)| {
            principal(&re, v, p.d.seat, c.0.card()).map(|k| {
                let r = &re[k.0];
                (k, exact_value(&r.pos, r.ctx, ctx.leaf, reverse, mem).0)
            })
        })
        .collect();
    Ok(WorldRaw {
        exact,
        exact_reverse,
        hybrid,
        re,
        rh,
        rh_exact,
        rh_reverse,
        rh_class,
        re_principal,
    })
}

/// Explain decisions `ds` (each before the cut's trick, a card play). `net` is the frontier
/// evaluator; with `oracle` the report adds the network's outcome probabilities at the principal
/// positions.
///
/// # Errors
/// If a decision is not before the cut or not a card play, or the search with exact frontier
/// values does not give the plain exact values.
pub fn explain(
    ds: &[DecisionSpec],
    job: &ExplainJob,
    ctx: &Ctx,
    net: &dyn Frontier,
    oracle: Option<&Oracle>,
) -> Result<Vec<DecisionExplain>, String> {
    let mut preps = Vec::new();
    for d in ds {
        if d.trick <= 1 || d.trick >= job.cut {
            return Err(format!(
                "decision {} is at trick {}: explained decisions are at tricks 2..{} (after \
                 trick 1, before the frontier)",
                d.id, d.trick, job.cut
            ));
        }
        let (_, view, legal) = d.replay();
        let (_, cands) = candidates(&view, &legal);
        preps.push(Prep {
            d,
            view,
            cands,
            dseed: measure_seed(job.seed, job.label, d.id),
        });
    }
    let worlds: Vec<_> = preps
        .iter()
        .map(|p| ctx.source.prepare(&p.view, ctx.rules))
        .collect();
    let per = job.to as usize;
    let n_tasks = preps.len() * per;
    let slots: Mutex<Vec<Option<Result<WorldRaw, String>>>> =
        Mutex::new((0..n_tasks).map(|_| None).collect());
    let next = AtomicUsize::new(0);
    let work = || {
        let mut mem = SolveMem::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n_tasks {
                break;
            }
            let (k, w) = (i / per, (i % per) as u64);
            let p = &preps[k];
            let m = world_match(&p.view, &*worlds[k], w, p.dseed, ctx.rules);
            let out =
                world_raw(p, &m, job, ctx, net, &mut mem).map_err(|e| format!("{e} (world {w})"));
            slots.lock().unwrap_or_else(|e| e.into_inner())[i] = Some(out);
        }
    };
    let threads = job.threads.max(1).min(n_tasks.max(1));
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(work);
        }
    });
    let outs: Vec<WorldRaw> = slots
        .into_inner()
        .unwrap_or_else(|e| e.into_inner())
        .into_iter()
        .map(|o| o.expect("every world valued"))
        .collect::<Result<_, _>>()?;
    let mut res = Vec::new();
    for (k, p) in preps.iter().enumerate() {
        let team = engine::team_of(p.d.seat);
        let ws = &outs[k * per..(k + 1) * per];
        let ex: Vec<Vec<MoneyChance>> = ws.iter().map(|o| o.exact.clone()).collect();
        let er: Vec<Vec<MoneyChance>> = ws.iter().map(|o| o.exact_reverse.clone()).collect();
        let hy: Vec<Vec<MoneyChance>> = ws.iter().map(|o| o.hybrid.clone()).collect();
        let (em, ec, _) = aggregate(&ex, p.cands.len(), team);
        let (rm, rc, _) = aggregate(&er, p.cands.len(), team);
        let (hm, _, _) = aggregate(&hy, p.cands.len(), team);
        let hc = best_index(&hm, team);
        let loss = em[ec].get(team) - em[hc].get(team);
        let loss_reverse = rm[rc].get(team) - rm[hc].get(team);
        let mut worlds_out = Vec::with_capacity(per);
        let mut misrank_tie_worlds = 0;
        let mut misrank_by_class = [0i64; N_CLASSES + 1];
        for (w, o) in ws.iter().enumerate() {
            let g = |v: &MoneyChance| i64::try_from(v.get(team)).expect("fits");
            let (e, h) = (&o.exact, &o.hybrid);
            let misrank = (g(&h[hc]) - g(&h[ec])) - (g(&e[hc]) - g(&e[ec]));
            let mut errors = Vec::with_capacity(o.rh.len());
            let (mut wrong, mut tied) = (0, 0);
            let mut by_class = [0usize; N_CLASSES];
            for (((r, x), y), &(class, money)) in
                o.rh.iter()
                    .zip(&o.rh_exact)
                    .zip(&o.rh_reverse)
                    .zip(&o.rh_class)
            {
                let e: [i64; 4] = core::array::from_fn(|k| {
                    let tt = (team + k as u8) % 4;
                    r.value.get(tt) as i64 - x.get(tt) as i64
                });
                tied += usize::from(x != y);
                by_class[usize::from(class)] += 1;
                if e[0].unsigned_abs() > ONE / 100 {
                    wrong += 1;
                }
                let rel: [bool; 4] =
                    core::array::from_fn(|k| money[usize::from((team + k as u8) % 4)]);
                errors.push((class, x != y, e, rel));
            }
            // On a hybrid line (`hybrid`), the list is the hybrid's with its exact values; on
            // an exact line, the exact search's (exact values; the reverse order solved now).
            let line = |hybrid: bool, j: usize| -> Option<LineEnd> {
                let found = if hybrid {
                    principal(&o.rh, &h[j], p.d.seat, p.cands[j].0.card())
                        .map(|k| (k, o.rh_exact[k.0], o.rh_reverse[k.0]))
                } else {
                    o.re_principal[j].map(|(k, y)| (k, o.re[k.0].value, y))
                };
                found.map(|((i, ties, under), xv, yv)| {
                    let r = if hybrid { &o.rh[i] } else { &o.re[i] };
                    let class = if hybrid {
                        o.rh_class[i].0
                    } else {
                        race_class(&r.canon, &r.ctx, ctx.leaf)
                    };
                    let nv = if hybrid {
                        r.value
                    } else {
                        net.value(&r.canon, &r.ctx)
                    };
                    let (outcomes, label) = match oracle {
                        Some(o) => outcome_report(o, &r.canon, &r.ctx, ctx.leaf, &xv, 4),
                        None => (Vec::new(), None),
                    };
                    LineEnd {
                        position: render_position(&r.pos, &r.ctx, p.d.seat),
                        net: arr(&nv),
                        exact: arr(&xv),
                        exact_reverse: arr(&yv),
                        class,
                        ties,
                        under,
                        outcomes,
                        label,
                    }
                })
            };
            let hybrid_line: Vec<Option<LineEnd>> =
                (0..p.cands.len()).map(|j| line(true, j)).collect();
            let exact_line: Vec<Option<LineEnd>> =
                (0..p.cands.len()).map(|j| line(false, j)).collect();
            let tie_world = [hc, ec].iter().any(|&j| {
                o.exact[j] != o.exact_reverse[j]
                    || [&hybrid_line[j], &exact_line[j]]
                        .iter()
                        .any(|l| l.as_ref().is_some_and(|l| l.exact != l.exact_reverse))
            });
            if tie_world {
                misrank_tie_worlds += misrank;
            }
            let line_class = [hc, ec]
                .iter()
                .flat_map(|&j| [&hybrid_line[j], &exact_line[j]])
                .filter_map(|l| l.as_ref().map(|l| l.class))
                .max();
            misrank_by_class[line_class.map_or(N_CLASSES, usize::from)] += misrank;
            worlds_out.push(WorldExplain {
                world: w as u64,
                exact: e.iter().map(arr).collect(),
                exact_reverse: o.exact_reverse.iter().map(arr).collect(),
                hybrid: h.iter().map(arr).collect(),
                misrank,
                hybrid_line,
                exact_line,
                reached: o.rh.len(),
                reached_wrong_1pct: wrong,
                reached_tie_decided: tied,
                reached_by_class: by_class,
                line_class,
                errors,
            });
        }
        res.push(DecisionExplain {
            id: p.d.id,
            header: render_decision(p.d, &p.cands),
            seat: p.d.seat,
            team,
            position: p.d.position + 1,
            cands: p.cands.iter().map(|c| name(c.0.card())).collect(),
            exact_mean: em.iter().map(arr).collect(),
            exact_mean_reverse: rm.iter().map(arr).collect(),
            hybrid_mean: hm.iter().map(arr).collect(),
            exact_choice: ec,
            exact_choice_reverse: rc,
            hybrid_choice: hc,
            loss,
            loss_reverse,
            misrank_total: worlds_out.iter().map(|w| w.misrank).sum(),
            misrank_tie_worlds,
            misrank_by_class,
            misrank_worlds: (
                worlds_out.iter().filter(|w| w.misrank > 0).count(),
                worlds_out.iter().filter(|w| w.misrank < 0).count(),
            ),
            frontier_error: FrontierError::of(worlds_out.iter()),
            worlds: worlds_out,
        });
    }
    Ok(res)
}

impl DecisionExplain {
    /// Drop the principal positions of every world but the `top` most misleading ones (to keep
    /// a JSON report of many decisions small).
    pub fn keep_lines(&mut self, top: usize) {
        let mut order: Vec<usize> = (0..self.worlds.len()).collect();
        order.sort_by(|&a, &b| {
            self.worlds[b]
                .misrank
                .cmp(&self.worlds[a].misrank)
                .then(a.cmp(&b))
        });
        for &i in &order[top.min(order.len())..] {
            self.worlds[i].hybrid_line.clear();
            self.worlds[i].exact_line.clear();
        }
    }
}

/// The network's error at the hybrid's frontier positions pooled over explained decisions.
pub fn pooled_frontier_error(xs: &[DecisionExplain]) -> FrontierError {
    FrontierError::of(xs.iter().flat_map(|x| x.worlds.iter()))
}

/// One line of frontier error statistics.
pub fn render_frontier_error(f: &FrontierError) -> String {
    let s = |e: &ErrStats| {
        format!(
            "{} values, SD {:.3}%, mean |e| {:.3}%, p99.9 {:.2}%, max {:.2}%",
            e.n,
            100.0 * e.sd,
            100.0 * e.mean_abs,
            100.0 * e.p999_abs,
            100.0 * e.max_abs
        )
    };
    let classes: Vec<String> = (0..N_CLASSES)
        .map(|k| {
            let e = &f.enemy[k];
            format!(
                "class {k} ({}): {}{}",
                RACE_CLASSES[k],
                s(&f.by_class[k]),
                if e.n > 0 {
                    format!(
                        " [the match-winning teams' entries: {} values, bias {:+.4}%, SD {:.3}%]",
                        e.n,
                        100.0 * e.mean,
                        100.0 * e.sd
                    )
                } else {
                    String::new()
                }
            )
        })
        .collect();
    format!(
        "all teams {}; decider's team bias {:+.4}%, SD {:.3}%; tie-decided positions {}; other \
         positions {}; {}",
        s(&f.all),
        100.0 * f.decider.mean,
        100.0 * f.decider.sd,
        s(&f.tie_decided),
        s(&f.plain),
        classes.join("; ")
    )
}

/// The explanation as text: per decision the header, candidates with exact (both tie orders)
/// and hybrid means, the misranking's spread over worlds, the frontier error, and the `top`
/// most misleading worlds with their principal positions.
pub fn render(x: &DecisionExplain, top: usize) -> Vec<String> {
    let pct = |v: u64| format!("{:.4}%", 100.0 * v as f64 / ONE as f64);
    let spct = |v: i64| format!("{:+.4}%", 100.0 * v as f64 / ONE as f64);
    let t = usize::from(x.team);
    let mut out = x.header.clone();
    for (j, c) in x.cands.iter().enumerate() {
        out.push(format!(
            "  {c:>4}: exact mean {} (reverse tie order {}) hybrid mean {}{}{}{}",
            pct(x.exact_mean[j][t]),
            pct(x.exact_mean_reverse[j][t]),
            pct(x.hybrid_mean[j][t]),
            if j == x.exact_choice {
                "  <- exact"
            } else {
                ""
            },
            if j == x.exact_choice_reverse {
                "  <- exact, reverse ties"
            } else {
                ""
            },
            if j == x.hybrid_choice {
                "  <- hybrid"
            } else {
                ""
            },
        ));
    }
    out.push(format!(
        "  loss {} (with the reverse tie order's exact means {})",
        pct(x.loss),
        pct(x.loss_reverse)
    ));
    if x.exact_choice != x.hybrid_choice {
        let mut ms: Vec<i64> = x.worlds.iter().map(|w| w.misrank).collect();
        ms.sort_unstable_by(|a, b| b.cmp(a));
        let share = |n: usize| {
            let s: i64 = ms.iter().take(n).sum();
            format!("{:.0}%", 100.0 * s as f64 / x.misrank_total.max(1) as f64)
        };
        out.push(format!(
            "  misranking: total over worlds {} ({} a world); worlds pushing to the hybrid's \
             choice {}, against {}; the top 1/3/10/32 worlds carry {}/{}/{}/{} of the total; \
             worlds with a tie-decided value on the two choices' lines carry {}; by the lines' \
             race class 0/1/2/3/4/none {}",
            spct(x.misrank_total),
            spct(x.misrank_total / x.worlds.len().max(1) as i64),
            x.misrank_worlds.0,
            x.misrank_worlds.1,
            share(1),
            share(3),
            share(10),
            share(32),
            spct(x.misrank_tie_worlds),
            x.misrank_by_class
                .iter()
                .map(|&m| spct(m))
                .collect::<Vec<_>>()
                .join("/")
        ));
    }
    out.push(format!(
        "  network error at the hybrid's frontier positions: {}",
        render_frontier_error(&x.frontier_error)
    ));
    let mut order: Vec<&WorldExplain> = x.worlds.iter().collect();
    order.sort_by(|a, b| b.misrank.cmp(&a.misrank).then(a.world.cmp(&b.world)));
    let (ec, hc) = (x.exact_choice, x.hybrid_choice);
    for w in order.iter().take(top) {
        out.push(format!(
            "  world {}: misranking {}; exact {} {} / {} {} (reverse tie order {} / {}); hybrid \
             {} / {}; {} frontier positions, {} off by more than 1% for the decider, {} \
             tie-decided",
            w.world,
            spct(w.misrank),
            x.cands[hc],
            pct(w.exact[hc][t]),
            x.cands[ec],
            pct(w.exact[ec][t]),
            pct(w.exact_reverse[hc][t]),
            pct(w.exact_reverse[ec][t]),
            pct(w.hybrid[hc][t]),
            pct(w.hybrid[ec][t]),
            w.reached,
            w.reached_wrong_1pct,
            w.reached_tie_decided
        ));
        for (what, lines) in [("hybrid", &w.hybrid_line), ("exact", &w.exact_line)] {
            for j in [hc, ec] {
                match &lines[j] {
                    None => out.push(format!(
                        "    {what} line of {}: no frontier position (the line ends before it)",
                        x.cands[j]
                    )),
                    Some(l) => {
                        out.push(format!(
                            "    {what} line of {}: net {} exact {} (reverse tie order {}) \
                             (decider's team; race class {}; {} with this vector{})",
                            x.cands[j],
                            pct(l.net[t]),
                            pct(l.exact[t]),
                            pct(l.exact_reverse[t]),
                            l.class,
                            l.ties,
                            if l.under {
                                ""
                            } else {
                                ", after an equivalent card"
                            }
                        ));
                        out.push(format!("      {}", l.position));
                        if !l.outcomes.is_empty() {
                            out.push(format!(
                                "      net: {}; exact outcome: {}",
                                l.outcomes
                                    .iter()
                                    .map(|(n, p)| format!("{n} {:.1}%", 100.0 * p))
                                    .collect::<Vec<_>>()
                                    .join(", "),
                                l.label.as_deref().unwrap_or("?")
                            ));
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use equity::MatchState;
    use solver::{FnFrontier, canonical};

    use crate::measure::{build_decisions, collect_starts};
    use crate::worlds::BeliefSource;

    fn leaf() -> Arc<SharedLeaf> {
        Arc::new(solver::FnLeaf(|s: &MatchState| {
            MoneyChance::from_weights(core::array::from_fn(|t| {
                1 + u128::from(s.scores[t]) + 40 * u128::from(s.games[t])
            }))
        }))
    }

    /// A deliberately wrong evaluator: favours the leader's team.
    fn skewed(c: &CanonPos, _: &MatchCtx) -> MoneyChance {
        MoneyChance::from_weights(core::array::from_fn(|t| {
            if t as u8 == c.leader % 4 { 5 } else { 1 }
        }))
    }

    /// The explanation agrees with the check (same choices, loss and values), every candidate's
    /// principal position carries its value, the exact frontier values give back the exact
    /// solve (checked inside), and the result does not depend on the thread count.
    #[test]
    fn explain_matches_the_check_and_is_thread_independent() {
        let starts = collect_starts(300, 3);
        let ds = build_decisions(3, 8, 5, &starts);
        let l = leaf();
        let src = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: engine::game::Rules::HOUSE,
        };
        let f = FnFrontier(skewed as fn(&CanonPos, &MatchCtx) -> MoneyChance);
        let job = |threads| ExplainJob {
            cut: 4,
            seed: 5,
            label: 1,
            to: 4,
            threads,
            tt_mb: 4,
        };
        let a = explain(&ds, &job(1), &ctx, &f, None).expect("runs");
        let b = explain(&ds, &job(3), &ctx, &f, None).expect("runs");
        let check = crate::hybrid::run_check(
            &ds,
            &std::collections::BTreeMap::new(),
            &crate::hybrid::CheckJob {
                cut: 4,
                seed: 5,
                label: 1,
                to: 4,
                threads: 2,
                tt_mb: 4,
                verify_pct: 0.0,
                recompute: false,
                rules_robber_burn: false,
                exact_solve_threads: 1,
                exact_tt_mb: 4,
                noise: None,
            },
            &ctx,
            &f,
        )
        .expect("runs");
        assert_eq!(
            serde_json::to_string(&a).expect("json"),
            serde_json::to_string(&b).expect("json")
        );
        let mut principal_found = 0;
        for (x, c) in a.iter().zip(&check.decisions) {
            assert_eq!(
                (x.exact_choice, x.hybrid_choice, x.loss),
                (c.exact_choice, c.hybrid_choice, c.loss)
            );
            for w in &x.worlds {
                for j in 0..x.cands.len() {
                    if let Some(l) = &w.hybrid_line[j] {
                        assert_eq!(l.net, w.hybrid[j]);
                        principal_found += 1;
                    }
                    if let Some(l) = &w.exact_line[j] {
                        assert_eq!(l.exact, w.exact[j]);
                    }
                }
            }
            assert!(!render(x, 2).is_empty());
        }
        assert!(principal_found > 0);
        // The skewed evaluator is wrong somewhere, and the frontier error sees it.
        assert!(a.iter().any(|x| x.frontier_error.all.max_abs > 0.0));
        let pooled = pooled_frontier_error(&a);
        assert_eq!(
            pooled.all.n,
            a.iter().map(|x| x.frontier_error.all.n).sum::<usize>()
        );
        assert_eq!(pooled.all.n, pooled.plain.n + pooled.tie_decided.n);
    }

    fn bits(s: &str) -> u64 {
        s.split_whitespace()
            .map(|c| bit(engine::cards::parse(c).expect("a card")))
            .fold(0, |a, b| a | b)
    }

    /// Golden decision 335 (trick-3 golden set, seed 1), world 216, the hybrid's principal
    /// position after JD: the start of trick 4, clubs trumps, the 5 of trumps out of play, the
    /// Joker (the best trump, team 1's) played in trick 1. Teams 0 and 1 each have a game and
    /// stand on 36 and 40, team 2 on 36, team 3 on 26. Seat 7 leads.
    fn position_335() -> (Position, MatchCtx) {
        let hands = [
            "AD 7D", "3S 4S", "5H JH", "10D KS", "7S JS", "8H AS", "4D 10S", "7C 6D",
        ]
        .map(bits);
        let pos = Position {
            hands,
            trump: engine::cards::Suit::Clubs,
            leader: 7,
            trick: [0; 8],
            n_played: 0,
            hand: equity::hand::InHand {
                scores: [36, 40, 36, 26],
                flags: [true, false, true, true],
                tricks: [0, 2, 0, 1],
                five_played: false,
                tricks_done: 3,
            },
            best: Some((
                engine::rules::trump_rank(engine::cards::JOKER, engine::cards::Suit::Clubs),
                1,
            )),
            rob: None,
            turned: 0,
        };
        (
            pos,
            MatchCtx {
                games: [1, 1, 0, 0],
                dealer_team: 0,
            },
        )
    }

    /// The race classes, and the tie-decided match of golden decision 335: team 3 gets nothing
    /// whoever wins (team 0 or team 1 would win the match), so its seats are indifferent and the
    /// tie rule decides the match. With the documented order (first card by index) seat 3
    /// discards 10D under seat 7's 7C and seat 0's 7D wins the last trick: team 0 wins. With
    /// the reverse order seat 3 keeps 10D, team 3 takes both tricks and team 1 wins at the end
    /// of the hand with its first best-trump bonus (40 + 6).
    #[test]
    fn race_classes_and_a_tie_decided_match() {
        let l = leaf();
        let (p, ctx) = position_335();
        let c = canonical(&p);
        assert_eq!(race_class(&c, &ctx, &*l), 4);
        let solve = |p: &Position, order| {
            solve_fast_with(
                p,
                ctx,
                &DynLeaf(&*l),
                FastOptions {
                    order,
                    ..FastOptions::default()
                },
                &mut SolveMem::new(),
            )
            .value
            .expect("no time cap")
        };
        assert_eq!(solve(&p, TieOrder::Ascending), MoneyChance::certain(0));
        assert_eq!(solve(&p, TieOrder::Descending), MoneyChance::certain(1));
        // No game won yet: three teams can win the game, nobody the match.
        let none = MatchCtx {
            games: [0; 4],
            ..ctx
        };
        assert_eq!(race_class(&c, &none, &*l), 2);
        // Only team 1 (40) can reach 41: with its game won it can win the match, else the game.
        let mut q = p;
        q.hand.scores = [5, 40, 0, 10];
        q.hand.flags = [false; 4];
        assert_eq!(
            race_info(&canonical(&q), &ctx, &*l),
            (3, [false, true, false, false])
        );
        assert_eq!(race_class(&canonical(&q), &none, &*l), 1);
        // Nobody can reach 41 in two tricks, nor make a jink.
        q.hand.scores = [5, 10, 0, 5];
        assert_eq!(race_class(&canonical(&q), &ctx, &*l), 0);
    }

    #[test]
    fn error_statistics() {
        let s = err_stats(&[0, 2 * ONE as i64 / 100, -2 * ONE as i64 / 100, 0]);
        assert_eq!(s.n, 4);
        assert!(s.mean.abs() < 1e-12);
        assert!((s.max_abs - 0.02).abs() < 1e-9);
        assert!((s.mean_abs - 0.01).abs() < 1e-9);
        assert!((s.sd - 0.02f64 / 2f64.sqrt()).abs() < 1e-9);
        assert_eq!(err_stats(&[]).n, 0);
    }
}
