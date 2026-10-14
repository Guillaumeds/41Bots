//! Stratified decision sets and the early-trick yardstick measurements (docs/M3-SPEC.md §5.8,
//! §5.10 strata, §13.1).
//!
//! The pipeline, every step deterministic and resumable in chunks:
//! 1. [`build_decisions`]: decisions at one trick, sampled from `rules-v1` self-play hands that
//!    start from stratified match states (S1–S4, a quarter each). Stored as (start state, deck,
//!    actions so far, seat) so they replay exactly.
//! 2. [`world_rows_batch`] / [`worlds_file`]: per decision, the value of every candidate in worlds
//!    `from..to` by one method (exact PIMC or rollout-PIMC). World `w` of decision `d` under label
//!    `L` is drawn from `Rng::derive(dseed, &[WORLD, w])` with
//!    `dseed = Rng::derive(seed, &[AGREE, L, d])`, so world sets are nested (N worlds are the first
//!    N of 2N) and a chunk can be redone identically. All (decision, world) tasks of a run share
//!    one work queue (see [`world_rows_batch`]), so no thread waits for another decision's slow
//!    world; the output file is byte-identical for any thread count.
//! 3. [`agreement_curve`]: the plateau rule of §13.1 on nested world sets.
//! 4. [`score`]: a method against the yardstick (exact PIMC at the plateau N): agreement, mean
//!    money-chance loss, loss distribution, and paired comparisons with confidence intervals.

use std::cmp::Reverse;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use engine::cards::{Bits, DECK_SIZE, JOKER, Suit, bit, count};
use engine::game::{Action, HandStartState, Match, Phase, Rules};
use engine::rng::Rng;
use engine::view::SeatView;
use equity::{MoneyChance, ONE};
use serde::{Deserialize, Serialize};
use solver::{ParProgress, SolveMem};

use crate::labels;
use crate::pimc::{
    Candidate, Ctx, Method, SearchConfig, aggregate, candidates, solves_per_world, world_row,
};
use crate::plan::{AutoParams, ExecProgress, PlanTask, WorldPlan, plan_auto, run_planned};
use crate::worlds::Worlds;

/// The four strata of start states (§5.10).
pub fn stratum_of(s: &HandStartState) -> u8 {
    let max = s.scores.iter().copied().max().unwrap_or(0);
    let at1 = |t: usize| s.games_won[t] == 1;
    if (0..4).any(|t| at1(t) && s.scores[t] >= 26) || max >= 31 {
        4
    } else if (0..4).any(at1) && max <= 25 {
        3
    } else if s.games_won.iter().all(|&g| g == 0) && max <= 20 {
        1
    } else {
        2
    }
}

/// Hand-start states visited by `rules-v1` self-play, by stratum (index 0 = S1), in visit order,
/// with repeats (the empirical distribution). Plays until `hands` hand starts are collected.
pub fn collect_starts(hands: u64, seed: u64) -> [Vec<HandStartState>; 4] {
    let mut out: [Vec<HandStartState>; 4] = Default::default();
    let mut n = 0u64;
    let mut k = 0u64;
    while n < hands {
        let mut m = Match::new(
            Rng::derive(seed, &[labels::STRATUM, k]).next_u64(),
            Rules::HOUSE,
        );
        k += 1;
        let mut last = u32::MAX;
        while m.phase() != Phase::MatchOver && n < hands {
            if m.hand_index() != last {
                last = m.hand_index();
                let s = m.current_hand_start();
                out[usize::from(stratum_of(&s) - 1)].push(s);
                n += 1;
            }
            let seat = m.to_act().expect("to act");
            let legal = m.legal_actions();
            let a = bots::decide(&m.view(seat), &Rules::HOUSE, &legal).0;
            m.apply(a).expect("legal");
        }
    }
    out
}

/// One decision of a measurement set.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionSpec {
    /// Index in the set.
    pub id: u64,
    /// The trick (0 = the dealer's burns).
    pub trick: u8,
    /// The stratum of the start state (1–4).
    pub stratum: u8,
    /// The hand's start state.
    pub start: HandStartState,
    /// The hand's deck in deal order.
    pub deck: Vec<u8>,
    /// The hand's actions before the decision.
    pub actions: Vec<Action>,
    /// The deciding seat.
    pub seat: u8,
    /// Position in the trick: cards already played to it (0 = leader … 7 = eighth to play).
    pub position: u8,
    /// The deciding seat's hand type at the decision (§13.1 item 9); `None` before trumps are
    /// known, and in sets made before the field existed.
    #[serde(default)]
    pub hand_type: Option<HandType>,
}

/// A hand's type, from what its holder can see (docs/M3-SPEC.md §13.1 item 9).
///
/// The last three fields were added on 2026-10-14 (owner). They are `None` in records made before
/// they existed; [`relabel`] recomputes them from the decision set, which stores each decision's
/// deck and actions, so old measurements need no new solves. They are never written as `null`, so
/// old records keep their exact bytes when a file is rewritten.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandType {
    /// Trumps held (the trump suit, the Joker and the A♥).
    pub trumps: u8,
    /// Top trumps held: the 5 and J of trumps, the Joker, the A♥ and the ace of trumps (four
    /// cards when hearts are trumps).
    pub top_trumps: u8,
    /// Cards in the longest plain (non-trump) suit.
    pub longest_plain: u8,
    /// Holds the 5 of trumps (the bonus card).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub has_five: Option<bool>,
    /// Has taken a turned card this hand: robbed it at their own turn in trick 1 (rule 20, H2), or
    /// is the dealer who robbed a turned ace or took a turned Joker (rules 22 and 24, H4). Public
    /// knowledge, from the hand's burn order or the burn now due from this seat ([`view_robber`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robber: Option<bool>,
    /// Plain (non-trump) suits the hand holds no card of, 0–3 (the A♥ is a trump, not a heart,
    /// when another suit is trumps).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voids: Option<u8>,
}

impl HandType {
    /// The bands reported by `search agree`: trumps 0, 1, 2, 3+; top trumps 0, 1, 2+; longest
    /// plain suit 0–1, 2, 3+ (each band named by its smallest value).
    pub fn bands(&self) -> [u8; 3] {
        [
            self.trumps.min(3),
            self.top_trumps.min(2),
            self.longest_plain.clamp(1, 3),
        ]
    }

    /// The bands of the 2026-10-14 features: holds the 5 of trumps (0 no, 1 yes), robber (0 no,
    /// 1 yes), void plain suits 0, 1, 2+; `None` in records made before they existed.
    pub fn feature_bands(&self) -> Option<[u8; 3]> {
        Some([
            u8::from(self.has_five?),
            u8::from(self.robber?),
            self.voids?.min(2),
        ])
    }
}

/// The name of band `band` of feature `feature` of [`HandType::feature_bands`] (0 = the 5 of
/// trumps, 1 = robber, 2 = voids), as `search agree` and `search curves` report it.
pub fn feature_band_name(feature: usize, band: u8) -> String {
    let yes = if band == 1 { "yes" } else { "no" };
    match feature {
        0 => format!("has5-{yes}"),
        1 => format!("robber-{yes}"),
        _ => format!("voids{band}{}", if band >= 2 { "+" } else { "" }),
    }
}

/// The hand type of the cards `hand` with trump suit `trump`; `robber`: whether the holder has
/// taken a turned card this hand.
pub fn hand_type(hand: Bits, trump: Suit, robber: bool) -> HandType {
    use engine::rules::{ACE_HEARTS, ace_of_trumps, five_of, jack_of, plain_mask, trump_mask};
    let top = bit(five_of(trump))
        | bit(jack_of(trump))
        | bit(JOKER)
        | bit(ACE_HEARTS)
        | bit(ace_of_trumps(trump));
    let plain: Vec<u8> = Suit::ALL
        .iter()
        .filter(|&&s| s != trump)
        .map(|&s| count(hand & plain_mask(s, trump)))
        .collect();
    HandType {
        trumps: count(hand & trump_mask(trump)),
        top_trumps: count(hand & top),
        longest_plain: plain.iter().copied().max().unwrap_or(0),
        has_five: Some(hand & bit(five_of(trump)) != 0),
        robber: Some(robber),
        voids: Some(plain.iter().filter(|&&n| n == 0).count() as u8),
    }
}

/// Whether the view's seat has taken a turned card this hand, from public facts only: a burn by
/// this seat in the hand's burn order, or the burn this seat must make now (every take is followed
/// at once by the taker's burn, so this is exactly "took a turned card").
pub fn view_robber(v: &SeatView) -> bool {
    v.public.burn_order.iter().any(|&(s, _)| s == v.seat)
        || v.public.phase == Phase::AwaitBurn(v.seat)
}

/// The deciding seat's hand type in a view, once trumps are known.
pub fn view_hand_type(v: &SeatView) -> Option<HandType> {
    v.public.trump.map(|t| hand_type(v.hand, t, view_robber(v)))
}

/// Recompute the hand type of worlds records from their decision set, matched by (trick, id): each
/// decision is replayed from its start state, deck and actions, and labelled from the deciding
/// seat's view, exactly as a new run would label it. This adds the 2026-10-14 features to records
/// made before they existed, with no new solves. A record whose decision is not in `ds` is left
/// as it is; one whose decision disagrees with it (position, stratum, team, candidates, or the
/// labels it already has) is an error: the wrong decision file. Returns the records relabelled.
pub fn relabel(rows: &mut [WorldRows], ds: &[DecisionSpec]) -> Result<usize, String> {
    let by: std::collections::BTreeMap<(u8, u64), &DecisionSpec> =
        ds.iter().map(|d| ((d.trick, d.id), d)).collect();
    let mut n = 0;
    for r in rows.iter_mut() {
        let Some(d) = by.get(&(r.trick, r.id)) else {
            continue;
        };
        let (_, view, legal) = d.replay();
        let (_, cands) = candidates(&view, &legal);
        let h = view_hand_type(&view);
        let old = r
            .hand_type
            .map(|x| (x.trumps, x.top_trumps, x.longest_plain));
        if d.position != r.position
            || d.stratum != r.stratum
            || engine::team_of(d.seat) != r.team
            || cands != r.cands
            || old.is_some_and(|o| Some(o) != h.map(|x| (x.trumps, x.top_trumps, x.longest_plain)))
        {
            return Err(format!(
                "trick {} decision {}: the decision file does not match the worlds record \
                 (another decision set?)",
                r.trick, r.id
            ));
        }
        r.hand_type = h;
        n += 1;
    }
    Ok(n)
}

impl DecisionSpec {
    /// Replay the decision: the deciding seat's view and its legal actions.
    pub fn replay(&self) -> (Match, SeatView, Vec<Action>) {
        let mut deck = [0u8; DECK_SIZE];
        deck.copy_from_slice(&self.deck);
        let mut m = Match::from_hand_start(self.start, deck, 0, Rules::HOUSE).expect("valid");
        for &a in &self.actions {
            m.apply(a).expect("recorded actions are legal");
        }
        let v = m.view(self.seat);
        let legal = m.legal_actions();
        (m, v, legal)
    }
}

/// `count` decisions at trick `trick`, equal numbers per stratum and per position in the trick
/// (§13.1 item 6): decision `i` is from stratum `i mod 4 + 1` at position `(i / 4) mod 8`; each a uniformly chosen decision node (two or more candidates) of a `rules-v1`
/// self-play hand from a start state drawn from that stratum's visited states.
pub fn build_decisions(
    trick: u8,
    count: u64,
    seed: u64,
    starts: &[Vec<HandStartState>; 4],
) -> Vec<DecisionSpec> {
    let mut out = Vec::new();
    for i in 0..count {
        let st = (i % 4) as usize;
        let pos = ((i / 4) % 8) as usize;
        assert!(!starts[st].is_empty(), "stratum S{} has no states", st + 1);
        let mut rng = Rng::derive(seed, &[labels::MEASURE, u64::from(trick), i]);
        loop {
            let start = starts[st][rng.bounded(starts[st].len() as u64) as usize];
            let mut deck: [u8; DECK_SIZE] = core::array::from_fn(|k| k as u8);
            rng.shuffle(&mut deck);
            let mut m = Match::from_hand_start(start, deck, 0, Rules::HOUSE).expect("valid");
            let h = m.hand_index();
            let mut nodes: Vec<(Vec<Action>, u8, Option<HandType>)> = Vec::new();
            while m.phase() != Phase::MatchOver && m.hand_index() == h {
                let seat = m.to_act().expect("to act");
                let v = m.view(seat);
                let legal = m.legal_actions();
                if v.public.trick_no == trick
                    && v.public.trick_cards.len() == pos
                    && candidates(&v, &legal).1.len() >= 2
                {
                    nodes.push((m.actions().to_vec(), seat, view_hand_type(&v)));
                }
                if v.public.trick_no > trick {
                    break;
                }
                let a = bots::decide(&v, &Rules::HOUSE, &legal).0;
                m.apply(a).expect("legal");
            }
            if nodes.is_empty() {
                continue;
            }
            let (actions, seat, hand_type) =
                nodes.swap_remove(rng.bounded(nodes.len() as u64) as usize);
            out.push(DecisionSpec {
                id: i,
                trick,
                stratum: st as u8 + 1,
                start,
                deck: deck.to_vec(),
                actions,
                seat,
                position: pos as u8,
                hand_type,
            });
            break;
        }
    }
    out
}

/// The decision seed of decision `id` under world label `label` (1 = the reference / yardstick
/// worlds, others for methods and repeatability).
pub fn measure_seed(seed: u64, label: u64, id: u64) -> u64 {
    Rng::derive(seed, &[labels::AGREE, label, id]).next_u64()
}

/// Per-world candidate values of one decision. Everything in it is a pure function of the
/// decision and the job, so files of these are byte-identical for any thread count (timings go
/// to a separate [`DecisionTiming`] file).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldRows {
    /// The decision's id.
    pub id: u64,
    /// The trick.
    pub trick: u8,
    /// The stratum.
    pub stratum: u8,
    /// Position in the trick (0 = leader).
    pub position: u8,
    /// The deciding team.
    pub team: u8,
    /// `exact` or `rollout`.
    pub method: String,
    /// World label.
    pub label: u64,
    /// First world.
    pub from: u64,
    /// The candidates, in candidate order.
    pub cands: Vec<Candidate>,
    /// One row per world (from `from`), each the candidates' values (four u64 per value).
    pub rows: Vec<Vec<[u64; 4]>>,
    /// floor(log2) of the exact number of worlds consistent with the decider's view, when the
    /// world source can count them (the belief source does; `None` with the rejection sampler).
    #[serde(default)]
    pub log2_worlds: Option<u32>,
    /// The deciding seat's hand type (§13.1 item 9), once trumps are known.
    #[serde(default)]
    pub hand_type: Option<HandType>,
}

/// How long one decision's worlds took (reports only; never part of a result).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionTiming {
    /// The decision's id.
    pub id: u64,
    /// The trick.
    pub trick: u8,
    /// Position in the trick (0 = leader).
    pub position: u8,
    /// `exact` or `rollout`.
    pub method: String,
    /// World label.
    pub label: u64,
    /// Worlds valued.
    pub worlds: u64,
    /// Sum over its worlds of the time to value one world, in microseconds (one worker thread;
    /// with `solve_threads > 1`, the wall time of the parallel solves).
    pub world_us: u64,
    /// The slowest world, in microseconds.
    pub max_world_us: u64,
    /// Solver threads per world, in a position-aware run ([`Schedule::Auto`]); absent otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<u64>,
}

/// How a worlds run is spread over the machine. Values never depend on it.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Schedule {
    /// The explicit settings: `cfg.threads` workers, each solving one world at a time on
    /// `cfg.solve_threads` threads with `cfg.tt_mb`-MB tables.
    #[default]
    Fixed,
    /// Position-aware ([`crate::plan`]): each exact world gets its own threads and table, on
    /// `threads` threads in all, with `budget_mb` of memory for all tables at their peak.
    Auto {
        threads: usize,
        budget_mb: u64,
        params: Box<AutoParams>,
    },
}

/// One worlds job: which worlds of every decision, valued by which method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorldsJob {
    /// Exact or rollout PIMC.
    pub method: Method,
    /// World label (1 = yardstick).
    pub label: u64,
    /// Master seed.
    pub seed: u64,
    /// First world.
    pub from: u64,
    /// One past the last world.
    pub to: u64,
}

/// The name a method is recorded under.
pub fn method_name(m: Method) -> &'static str {
    match m {
        Method::ExactPimc => "exact",
        Method::RolloutPimc => "rollout",
        Method::Forced => "forced",
    }
}

/// Live counters of a [`world_rows_batch`] run, for progress reports only.
#[derive(Debug, Default)]
pub struct BatchProgress {
    /// Decisions in the run (including ones already done before a resume).
    pub decisions_total: AtomicU64,
    /// Decisions finished and handed to the sink.
    pub decisions_done: AtomicU64,
    /// (decision, world) tasks in the run (including ones already done before a resume).
    pub worlds_total: AtomicU64,
    /// Tasks finished.
    pub worlds_done: AtomicU64,
    /// Worker threads in use (a position-aware run: all its threads).
    pub threads: AtomicU64,
    /// The parallel exact solve now running, when one worker runs `solve_fast_par`
    /// (`threads == 1`, `solve_threads > 1`): its subproblems done and total.
    pub par: ParProgress,
    /// Set by a position-aware run ([`Schedule::Auto`] with more than one plan shape), whose
    /// worlds run at once and tail are in `exec`.
    pub planned: AtomicBool,
    /// The position-aware runner's state: worlds running, threads busy, tail.
    pub exec: ExecProgress,
}

/// A decision made ready for valuing.
struct Prepared<'d> {
    d: &'d DecisionSpec,
    view: SeatView,
    cands: Vec<Candidate>,
    dseed: u64,
}

/// The order a batch works in and writes in: longest expected first, so the run ends on cheap
/// worlds and its tail is short. Earlier tricks first, then earlier positions in the trick (the
/// leader sees the fewest cards: trick-2 exact worlds take seconds at position 1 and hundredths
/// of a second at position 8), then more candidates first, then decision id.
fn batch_order(p: &Prepared) -> (u8, u8, Reverse<usize>, u64) {
    batch_key(p.d, p.cands.len())
}

/// [`batch_order`]'s key for decision `d` with `n_cands` candidates (also the order of the
/// trick-1 bench's production window, `crate::t1bench`).
pub fn batch_key(d: &DecisionSpec, n_cands: usize) -> (u8, u8, Reverse<usize>, u64) {
    (d.trick, d.position, Reverse(n_cands), d.id)
}

/// Value every candidate of every decision in `ds` in worlds `job.from..job.to`, handing each
/// finished decision to `sink` with its timing.
///
/// Scheduling: one flat work queue holds every (decision, world) task of the whole run, ordered
/// by [`batch_order`] then world; `cfg.threads` workers take the next task from an atomic index,
/// so a slow world never holds other threads up (there is no barrier per decision). Each result
/// goes into a pre-sized slot by (decision, world); a decision is handed to `sink`, in batch
/// order, once it and every decision before it are complete. Values, rows and the order of the
/// `sink` calls are therefore the same for any thread count; only the timings differ.
pub fn world_rows_batch(
    ds: &[DecisionSpec],
    job: &WorldsJob,
    ctx: &Ctx,
    cfg: &SearchConfig,
    progress: Option<&BatchProgress>,
    mut sink: impl FnMut(WorldRows, DecisionTiming) -> Result<(), String>,
) -> Result<(), String> {
    assert!(job.to >= job.from, "worlds from..to");
    let preps = prepare_batch(ds, job);
    let worlds: Vec<_> = preps
        .iter()
        .map(|p| ctx.source.prepare(&p.view, ctx.rules))
        .collect();
    let nw = (job.to - job.from) as usize;
    let total = preps.len() * nw;
    let slots: Vec<OnceLock<Vec<MoneyChance>>> = (0..total).map(|_| OnceLock::new()).collect();
    let left: Vec<AtomicUsize> = preps.iter().map(|_| AtomicUsize::new(nw)).collect();
    let us: Vec<AtomicU64> = preps.iter().map(|_| AtomicU64::new(0)).collect();
    let max_us: Vec<AtomicU64> = preps.iter().map(|_| AtomicU64::new(0)).collect();
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let threads = cfg.threads.max(1).min(total.max(1));
    if let Some(p) = progress {
        p.threads.store(threads as u64, Ordering::Relaxed);
    }
    // Solve progress is only meaningful with one parallel solve at a time.
    let par = progress
        .filter(|_| threads == 1 && cfg.solve_threads > 1)
        .map(|p| &p.par);
    let work = |tx: mpsc::Sender<()>| {
        // One solver memory per worker for the whole run: its transposition table is allocated
        // once and emptied in O(1) between worlds, not allocated, first-touched and freed per
        // world (with hundreds of threads on short worlds that churn queued the threads on the
        // kernel's memory map). Values are unchanged (see `SolveMem`).
        let mut mem = SolveMem::new();
        while !stop.load(Ordering::Relaxed) {
            let k = next.fetch_add(1, Ordering::Relaxed);
            if k >= total {
                break;
            }
            let (j, i) = (k / nw, k % nw);
            let p = &preps[j];
            let t0 = Instant::now();
            let w = job.from + i as u64;
            let v = world_row(
                &p.view,
                &*worlds[j],
                &p.cands,
                job.method,
                w,
                p.dseed,
                ctx,
                cfg,
                par,
                &mut mem,
            );
            let t = t0.elapsed().as_micros() as u64;
            us[j].fetch_add(t, Ordering::Relaxed);
            max_us[j].fetch_max(t, Ordering::Relaxed);
            assert!(slots[k].set(v).is_ok(), "each task is taken once");
            if let Some(p) = progress {
                p.worlds_done.fetch_add(1, Ordering::Relaxed);
            }
            if left[j].fetch_sub(1, Ordering::AcqRel) == 1 {
                // The writer may have stopped (a sink error); nothing to do then.
                let _ = tx.send(());
            }
        }
    };
    std::thread::scope(|s| {
        let (tx, rx) = mpsc::channel::<()>();
        for _ in 0..threads.min(total) {
            let tx = tx.clone();
            s.spawn(move || work(tx));
        }
        drop(tx);
        // The writer (this thread): hand finished decisions out in batch order.
        let mut j = 0;
        let result = 'write: loop {
            while j < preps.len() && left[j].load(Ordering::Acquire) == 0 {
                let rows: Vec<Vec<[u64; 4]>> = slots[j * nw..(j + 1) * nw]
                    .iter()
                    .map(|c| {
                        c.get()
                            .expect("a finished decision has every world")
                            .iter()
                            .map(|v| v.0)
                            .collect()
                    })
                    .collect();
                let (r, t) = records(
                    &preps[j],
                    job,
                    &*worlds[j],
                    rows,
                    (
                        us[j].load(Ordering::Relaxed),
                        max_us[j].load(Ordering::Relaxed),
                    ),
                    None,
                );
                if let Err(e) = sink(r, t) {
                    break 'write Err(e);
                }
                if let Some(pr) = progress {
                    pr.decisions_done.fetch_add(1, Ordering::Relaxed);
                }
                j += 1;
            }
            if j == preps.len() {
                break Ok(());
            }
            if rx.recv().is_err() {
                // Every worker has stopped (one panicked); the scope re-raises the panic.
                break Ok(());
            }
        };
        stop.store(true, Ordering::Relaxed);
        result
    })
}

/// The decisions of a batch made ready, in [`batch_order`] (the order of the output file).
fn prepare_batch<'d>(ds: &'d [DecisionSpec], job: &WorldsJob) -> Vec<Prepared<'d>> {
    let mut preps: Vec<Prepared> = ds
        .iter()
        .map(|d| {
            let (_, view, legal) = d.replay();
            let (_, cands) = candidates(&view, &legal);
            Prepared {
                d,
                view,
                cands,
                dseed: measure_seed(job.seed, job.label, d.id),
            }
        })
        .collect();
    preps.sort_by_key(batch_order);
    preps
}

/// The worlds record and timing of a finished decision (`us`: total and slowest world in µs).
fn records(
    p: &Prepared,
    job: &WorldsJob,
    worlds: &dyn Worlds,
    rows: Vec<Vec<[u64; 4]>>,
    us: (u64, u64),
    threads: Option<u64>,
) -> (WorldRows, DecisionTiming) {
    let nw = rows.len() as u64;
    let r = WorldRows {
        id: p.d.id,
        trick: p.d.trick,
        stratum: p.d.stratum,
        position: p.d.position,
        team: engine::team_of(p.d.seat),
        method: method_name(job.method).to_string(),
        label: job.label,
        from: job.from,
        cands: p.cands.clone(),
        rows,
        log2_worlds: worlds.count().map(|c| c.max(1).ilog2()),
        hand_type: view_hand_type(&p.view),
    };
    let t = DecisionTiming {
        id: p.d.id,
        trick: p.d.trick,
        position: p.d.position,
        method: r.method.clone(),
        label: job.label,
        worlds: nw,
        world_us: us.0,
        max_world_us: us.1,
        threads,
    };
    (r, t)
}

/// [`world_rows_batch`] under a [`Schedule`]: `Fixed` is [`world_rows_batch`] itself. `Auto`
/// plans every exact world's threads and table by position ([`crate::plan::plan_auto`]) and
/// runs them with [`crate::plan::run_planned`]; when every world gets one thread and the same
/// table (cheap tricks), that is `world_rows_batch` on all threads with that table, and rollout
/// methods (no tables) also run that way. Rows, their order and the `sink` calls are exactly
/// `world_rows_batch`'s under any schedule; only the timings differ.
pub fn world_rows_sched(
    ds: &[DecisionSpec],
    job: &WorldsJob,
    ctx: &Ctx,
    cfg: &SearchConfig,
    sched: &Schedule,
    progress: Option<&BatchProgress>,
    mut sink: impl FnMut(WorldRows, DecisionTiming) -> Result<(), String>,
) -> Result<(), String> {
    let Schedule::Auto {
        threads,
        budget_mb,
        params,
    } = sched
    else {
        return world_rows_batch(ds, job, ctx, cfg, progress, sink);
    };
    let threads = (*threads).max(1);
    let budget = budget_mb.saturating_mul(1 << 20);
    let flat = |tt_mb: usize| SearchConfig {
        threads,
        solve_threads: 1,
        tt_mb,
        ..*cfg
    };
    if job.method != Method::ExactPimc || job.to <= job.from {
        return world_rows_batch(ds, job, ctx, &flat(cfg.tt_mb), progress, sink);
    }
    assert!(job.to >= job.from, "worlds from..to");
    let preps = prepare_batch(ds, job);
    let nw = (job.to - job.from) as usize;
    let tasks: Vec<PlanTask> = preps
        .iter()
        .flat_map(|p| {
            let t = PlanTask {
                trick: p.d.trick,
                position: p.d.position,
                solves: solves_per_world(&p.cands),
            };
            std::iter::repeat_n(t, nw)
        })
        .collect();
    let planned = plan_auto(&tasks, threads, budget, params);
    let one = planned.plans.first().copied();
    if let Some(one) = one
        && one.threads == 1
        && planned.plans.iter().all(|w| *w == one)
    {
        return world_rows_batch(ds, job, ctx, &flat(one.tt_mb), progress, sink);
    }
    eprintln!("worlds: position-aware plan {}", planned.record);
    let worlds: Vec<_> = preps
        .iter()
        .map(|p| ctx.source.prepare(&p.view, ctx.rules))
        .collect();
    if let Some(p) = progress {
        p.threads.store(threads as u64, Ordering::Relaxed);
        p.planned.store(true, Ordering::Relaxed);
    }
    let mut rows: Vec<Vec<Option<Vec<[u64; 4]>>>> = vec![vec![None; nw]; preps.len()];
    let mut left = vec![nw; preps.len()];
    let mut us = vec![(0u64, 0u64); preps.len()];
    let mut j_out = 0usize;
    let stop = AtomicBool::new(false);
    let work = |k: usize, plan: WorldPlan, mem: &mut SolveMem| {
        let (j, i) = (k / nw, k % nw);
        let p = &preps[j];
        let c = SearchConfig {
            threads: 1,
            solve_threads: plan.threads,
            tt_mb: plan.tt_mb,
            ..*cfg
        };
        let w = job.from + i as u64;
        let v = world_row(
            &p.view,
            &*worlds[j],
            &p.cands,
            job.method,
            w,
            p.dseed,
            ctx,
            &c,
            None,
            mem,
        );
        v.into_iter().map(|x| x.0).collect::<Vec<[u64; 4]>>()
    };
    run_planned(
        &planned.order,
        &planned.plans,
        threads,
        budget,
        &stop,
        progress.map(|p| &p.exec),
        work,
        |k, v, ran| {
            let (j, i) = (k / nw, k % nw);
            rows[j][i] = Some(v);
            left[j] -= 1;
            let t = (ran.wall_s * 1e6) as u64;
            us[j] = (us[j].0 + t, us[j].1.max(t));
            if let Some(p) = progress {
                p.worlds_done.fetch_add(1, Ordering::Relaxed);
            }
            // Hand finished decisions out in batch order.
            while j_out < preps.len() && left[j_out] == 0 {
                let r: Vec<Vec<[u64; 4]>> = core::mem::take(&mut rows[j_out])
                    .into_iter()
                    .map(|x| x.expect("a finished decision has every world"))
                    .collect();
                let threads = planned.plans[j_out * nw].threads as u64;
                let (r, t) = records(
                    &preps[j_out],
                    job,
                    &*worlds[j_out],
                    r,
                    us[j_out],
                    Some(threads),
                );
                sink(r, t)?;
                if let Some(pr) = progress {
                    pr.decisions_done.fetch_add(1, Ordering::Relaxed);
                }
                j_out += 1;
            }
            Ok(())
        },
    )
}

/// Value every candidate of `d` in worlds `from..to` by `method` (a batch of one decision).
#[allow(clippy::too_many_arguments)]
pub fn world_rows(
    d: &DecisionSpec,
    method: Method,
    label: u64,
    seed: u64,
    from: u64,
    to: u64,
    ctx: &Ctx,
    cfg: &SearchConfig,
) -> WorldRows {
    let job = WorldsJob {
        method,
        label,
        seed,
        from,
        to,
    };
    let mut out = None;
    world_rows_batch(std::slice::from_ref(d), &job, ctx, cfg, None, |r, _| {
        out = Some(r);
        Ok(())
    })
    .expect("the sink never fails");
    out.expect("one decision")
}

/// Read a JSON-lines file; a torn line (a run killed mid-write) is skipped.
pub fn read_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for l in std::io::BufReader::new(f).lines() {
        let l = l.map_err(|e| e.to_string())?;
        if l.trim().is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str(&l) {
            out.push(v);
        }
    }
    Ok(out)
}

/// The timing file that goes with a worlds file: `<out>.times.jsonl`.
pub fn timing_path(out: &Path) -> PathBuf {
    let mut s = out.as_os_str().to_owned();
    s.push(".times.jsonl");
    PathBuf::from(s)
}

/// The `search worlds` step: value decisions `ds` by `job` into the JSON-lines file `out`, one
/// line per decision in batch order, resuming from what `out` already holds (decisions already
/// there with this job's worlds are skipped; a torn last line is dropped and redone). The file is
/// byte-identical for any thread count, and whether or not the run was interrupted and resumed.
/// Timings are appended to [`timing_path`]`(out)`. `on_done` sees each decision as it is written
/// (for logs). Returns the number of decisions valued by this call.
///
/// Extension (§13.1 item 10): records of the same method and label for other world ranges may
/// already be in `out` (or in other files). A job with `from` at the end of an earlier range
/// appends its worlds as new records, which [`merge_rows`] joins to the earlier ones, so the
/// result is exactly what one longer run would have written (world `w` of a decision depends
/// only on the seed, label, decision and `w`).
pub fn worlds_file(
    out: &Path,
    ds: &[DecisionSpec],
    job: &WorldsJob,
    ctx: &Ctx,
    cfg: &SearchConfig,
    progress: Option<&BatchProgress>,
    on_done: impl FnMut(&WorldRows, &DecisionTiming),
) -> Result<usize, String> {
    worlds_file_sched(out, ds, job, ctx, cfg, &Schedule::Fixed, progress, on_done)
}

/// [`worlds_file`] under a [`Schedule`] ([`world_rows_sched`]): the file is the same bytes under
/// any schedule.
#[allow(clippy::too_many_arguments)]
pub fn worlds_file_sched(
    out: &Path,
    ds: &[DecisionSpec],
    job: &WorldsJob,
    ctx: &Ctx,
    cfg: &SearchConfig,
    sched: &Schedule,
    progress: Option<&BatchProgress>,
    mut on_done: impl FnMut(&WorldRows, &DecisionTiming),
) -> Result<usize, String> {
    let done: Vec<WorldRows> = if out.exists() {
        read_jsonl(out)?
    } else {
        Vec::new()
    };
    let nw = job.to - job.from;
    let mine = |r: &WorldRows| r.method == method_name(job.method) && r.label == job.label;
    if let Some(r) = done.iter().find(|r| !mine(r)) {
        return Err(format!(
            "{}: decision {} was made with other settings ({} label {}); use another output file",
            out.display(),
            r.id,
            r.method,
            r.label,
        ));
    }
    if let Some(r) = done
        .iter()
        .find(|r| r.from == job.from && r.rows.len() as u64 != nw)
    {
        let end = r.from + r.rows.len() as u64;
        return Err(format!(
            "{}: decision {} already has worlds {}..{end}; to add worlds, run with \
             --from-world {end} --to {} (the new worlds are appended as records that \
             agree/curves merge), or use another output file",
            out.display(),
            r.id,
            r.from,
            job.to
        ));
    }
    // Rewrite the file without a torn line, then append.
    let mut text = String::new();
    for r in &done {
        text.push_str(&serde_json::to_string(r).map_err(|e| e.to_string())?);
        text.push('\n');
    }
    std::fs::write(out, text).map_err(|e| format!("{}: {e}", out.display()))?;
    let open = |p: &Path| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .map_err(|e| format!("{}: {e}", p.display()))
    };
    let mut f = open(out)?;
    let mut ft = open(&timing_path(out))?;
    let todo: Vec<DecisionSpec> = ds
        .iter()
        .filter(|d| !done.iter().any(|r| r.id == d.id && r.from == job.from))
        .cloned()
        .collect();
    if let Some(p) = progress {
        let before = (ds.len() - todo.len()) as u64;
        p.decisions_total.store(ds.len() as u64, Ordering::Relaxed);
        p.decisions_done.store(before, Ordering::Relaxed);
        p.worlds_total
            .store(ds.len() as u64 * nw, Ordering::Relaxed);
        p.worlds_done.store(before * nw, Ordering::Relaxed);
    }
    let mut n = 0;
    world_rows_sched(&todo, job, ctx, cfg, sched, progress, |r, t| {
        let line = serde_json::to_string(&r).map_err(|e| e.to_string())?;
        writeln!(f, "{line}").map_err(|e| e.to_string())?;
        f.flush().map_err(|e| e.to_string())?;
        let tl = serde_json::to_string(&t).map_err(|e| e.to_string())?;
        writeln!(ft, "{tl}").map_err(|e| e.to_string())?;
        on_done(&r, &t);
        n += 1;
        Ok(())
    })?;
    Ok(n)
}

/// Join the records of one or more worlds files (§13.1 item 10): records of the same decision,
/// method and label whose world ranges continue or overlap each other become one record holding
/// every world from the first on, so N worlds are always the first N. Decisions keep the order in
/// which they first appear. Overlapping worlds must hold identical values (the same worlds valued
/// twice); otherwise the files were made with different seeds and this is an error. A range that
/// leaves a gap is left out, with a warning.
pub fn merge_rows(recs: Vec<WorldRows>) -> Result<(Vec<WorldRows>, Vec<String>), String> {
    type Key = (u64, String, u64);
    let mut order: Vec<Key> = Vec::new();
    let mut parts: std::collections::BTreeMap<Key, Vec<WorldRows>> = Default::default();
    for r in recs {
        let k = (r.id, r.method.clone(), r.label);
        let e = parts.entry(k.clone()).or_default();
        if e.is_empty() {
            order.push(k);
        }
        e.push(r);
    }
    let mut out = Vec::new();
    let mut warnings = Vec::new();
    for k in order {
        let mut segs = parts.remove(&k).expect("key listed");
        segs.sort_by_key(|r| (r.from, Reverse(r.rows.len())));
        let mut it = segs.into_iter();
        let mut cur = it.next().expect("at least one record");
        for s in it {
            let end = cur.from + cur.rows.len() as u64;
            if s.from > end {
                warnings.push(format!(
                    "decision {} ({} label {}): worlds {}..{} left out (worlds {end}..{} are missing)",
                    s.id,
                    s.method,
                    s.label,
                    s.from,
                    s.from + s.rows.len() as u64,
                    s.from
                ));
                continue;
            }
            if s.cands != cur.cands || s.team != cur.team || s.trick != cur.trick {
                return Err(format!(
                    "decision {}: records disagree on the decision itself (other decision set?)",
                    s.id
                ));
            }
            let skip = (end - s.from) as usize;
            let overlap = skip.min(s.rows.len());
            let at = (s.from - cur.from) as usize;
            if s.rows[..overlap] != cur.rows[at..at + overlap] {
                return Err(format!(
                    "decision {} ({} label {}): worlds {}..{} were valued twice with different \
                     values (files made with different seeds?)",
                    s.id,
                    s.method,
                    s.label,
                    s.from,
                    s.from + overlap as u64
                ));
            }
            cur.rows.extend_from_slice(&s.rows[overlap..]);
        }
        out.push(cur);
    }
    Ok((out, warnings))
}

/// Read and merge one or more worlds files ([`merge_rows`]).
pub fn read_worlds(paths: &[&Path]) -> Result<(Vec<WorldRows>, Vec<String>), String> {
    let mut all = Vec::new();
    for p in paths {
        all.extend(read_jsonl::<WorldRows>(p)?);
    }
    merge_rows(all)
}

/// Decision ids from `--ids`: a list (`3,17,42`) or the name of a file holding ids separated by
/// commas, spaces or new lines (`#` starts a comment).
pub fn parse_ids(arg: &str) -> Result<Vec<u64>, String> {
    let list = |s: &str| -> Result<Vec<u64>, String> {
        s.split(|c: char| c == ',' || c.is_whitespace())
            .filter(|t| !t.is_empty())
            .map(|t| t.parse().map_err(|_| format!("--ids: not an id: {t}")))
            .collect()
    };
    if !arg.is_empty()
        && arg
            .chars()
            .all(|c| c.is_ascii_digit() || c == ',' || c == ' ')
    {
        return list(arg);
    }
    let text = std::fs::read_to_string(arg).map_err(|e| format!("--ids {arg}: {e}"))?;
    let body: String = text
        .lines()
        .map(|l| l.split('#').next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");
    list(&body)
}

fn mc(v: [u64; 4]) -> MoneyChance {
    MoneyChance::from_raw(v).expect("stored values sum to one")
}

/// Mean values over the first `n` worlds, and the chosen index.
pub fn mean_first(r: &WorldRows, n: usize) -> (Vec<MoneyChance>, usize) {
    let rows: Vec<Vec<MoneyChance>> = r.rows[..n]
        .iter()
        .map(|row| row.iter().map(|&v| mc(v)).collect())
        .collect();
    let (m, c, _) = aggregate(&rows, r.cands.len(), r.team);
    (m, c)
}

/// Is candidate `i` among the tied-best for `team`?
pub fn tied_best(v: &[MoneyChance], team: u8, i: usize) -> bool {
    let best = v.iter().map(|x| x.get(team)).max().unwrap_or(0);
    v[i].get(team) == best
}

/// Money chance as a fraction (reports only).
pub fn frac(x: u64) -> f64 {
    x as f64 / ONE as f64
}

/// One step of the agreement curve: N against 2N.
#[derive(Clone, Debug, Serialize)]
pub struct CurveStep {
    /// Worlds.
    pub n: usize,
    /// Doubled.
    pub n2: usize,
    /// Decisions compared.
    pub decisions: usize,
    /// Share of decisions whose N-world choice is not among the 2N-world tied-best cards.
    pub change_rate: f64,
    /// |mean over decisions of V_2N(choice_2N) − mean of V_N(choice_N)| (§13.1 literal).
    pub mean_value_change: f64,
    /// Mean over decisions of V_2N(choice_2N) − V_2N(choice_N) (the N choice's loss under 2N).
    pub mean_loss: f64,
    /// Standard error of `mean_loss`.
    pub loss_se: f64,
    /// Both §13.1 conditions hold (change < 1%, value change and loss < 0.001).
    pub plateau_ok: bool,
}

/// The steps of the agreement curve on a grid: every N whose double is also in the grid, paired
/// with it (a doubling grid such as 10,20,40,… and a fine one such as 10,20,30,… both work);
/// consecutive grid values when no N has its double in the grid.
pub fn curve_steps(grid: &[usize]) -> Vec<(usize, usize)> {
    let mut g = grid.to_vec();
    g.sort_unstable();
    g.dedup();
    let doubling: Vec<(usize, usize)> = g
        .iter()
        .filter(|&&n| g.binary_search(&(2 * n)).is_ok())
        .map(|&n| (n, 2 * n))
        .collect();
    if doubling.is_empty() {
        g.windows(2).map(|w| (w[0], w[1])).collect()
    } else {
        doubling
    }
}

/// The agreement curve over nested world sets on `grid` ([`curve_steps`]: N against 2N), and the
/// plateau N (§13.1): the smallest N where doubling changes the chosen card in under 1% of
/// decisions and changes the mean chosen-card value by under 0.001. The value condition is
/// applied to both readings, the literal one (difference of the means) and the loss of the
/// N-world choice under the 2N-world values, so cancellation cannot hide a change. A step uses
/// the decisions with at least 2N worlds.
pub fn agreement_curve(rows: &[WorldRows], grid: &[usize]) -> (Vec<CurveStep>, Option<usize>) {
    let mut steps = Vec::new();
    for (n, n2) in curve_steps(grid) {
        let use_: Vec<&WorldRows> = rows.iter().filter(|r| r.rows.len() >= n2).collect();
        if use_.is_empty() {
            continue;
        }
        let mut changes = 0usize;
        let (mut s_a, mut s_b) = (0f64, 0f64);
        let mut losses = Vec::new();
        for r in &use_ {
            let (v1, c1) = mean_first(r, n);
            let (v2, c2) = mean_first(r, n2);
            if !tied_best(&v2, r.team, c1) {
                changes += 1;
            }
            s_a += frac(v2[c2].get(r.team));
            s_b += frac(v1[c1].get(r.team));
            losses.push(frac(v2[c2].get(r.team) - v2[c1].get(r.team)));
        }
        let k = use_.len() as f64;
        let (ml, se) = mean_se(&losses);
        let change_rate = changes as f64 / k;
        let mvc = ((s_a - s_b) / k).abs();
        steps.push(CurveStep {
            n,
            n2,
            decisions: use_.len(),
            change_rate,
            mean_value_change: mvc,
            mean_loss: ml,
            loss_se: se,
            plateau_ok: change_rate < 0.01 && mvc < 0.001 && ml < 0.001,
        });
    }
    let plateau = steps.iter().find(|s| s.plateau_ok).map(|s| s.n);
    (steps, plateau)
}

/// Mean and standard error.
pub fn mean_se(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    if x.is_empty() {
        return (0.0, 0.0);
    }
    let m = x.iter().sum::<f64>() / n;
    if x.len() < 2 {
        return (m, 0.0);
    }
    let v = x.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / (n - 1.0);
    (m, (v / n).sqrt())
}

/// A method's per-decision result against the yardstick.
#[derive(Clone, Debug, Serialize)]
pub struct Scored {
    /// Decision id.
    pub id: u64,
    /// Agrees with the yardstick (its choice is among the yardstick's tied-best).
    pub agree: bool,
    /// Loss of its choice under the yardstick values, decider's team, units of 2^-32.
    pub loss: u64,
}

/// Score a method's rows (its own worlds, `n_method` of them) against yardstick rows (the first
/// `n_ref` worlds), on the decisions both have.
pub fn score(
    yard: &[WorldRows],
    n_ref: usize,
    method: &[WorldRows],
    n_method: usize,
) -> Vec<Scored> {
    let mut out = Vec::new();
    for m in method {
        let Some(y) = yard.iter().find(|y| y.id == m.id) else {
            continue;
        };
        if y.rows.len() < n_ref || m.rows.len() < n_method {
            continue;
        }
        assert_eq!(y.cands, m.cands, "same decision, same candidates");
        let (vy, cy) = mean_first(y, n_ref);
        let (_, cm) = mean_first(m, n_method);
        out.push(Scored {
            id: m.id,
            agree: tied_best(&vy, y.team, cm),
            loss: vy[cy].get(y.team) - vy[cm].get(y.team),
        });
    }
    out
}

/// Summary of a scored method.
#[derive(Clone, Debug, Serialize)]
pub struct ScoreSummary {
    /// Decisions.
    pub decisions: usize,
    /// Agreement rate and its standard error.
    pub agreement: (f64, f64),
    /// Mean loss in money chance, and its standard error.
    pub mean_loss: (f64, f64),
    /// Mean loss in units of 2^-32.
    pub mean_loss_units: f64,
    /// Mean loss in stakes per player (8 × money chance: a match is worth +6 or −2 stakes).
    pub mean_loss_stakes: f64,
    /// Loss distribution: share with zero loss, then quantiles p50, p90, p99 and the maximum, in
    /// money chance.
    pub zero_share: f64,
    /// Quantiles (p50, p90, p99, max).
    pub quantiles: [f64; 4],
    /// Histogram of losses: < 0.001, < 0.003, < 0.01, < 0.03, ≥ 0.03 (counts).
    pub histogram: [usize; 5],
}

/// Summarise scored decisions.
pub fn summarise(s: &[Scored]) -> ScoreSummary {
    let agree: Vec<f64> = s.iter().map(|x| f64::from(u8::from(x.agree))).collect();
    let loss: Vec<f64> = s.iter().map(|x| frac(x.loss)).collect();
    let mut sorted = loss.clone();
    sorted.sort_by(f64::total_cmp);
    let q = |p: f64| {
        if sorted.is_empty() {
            0.0
        } else {
            sorted[((sorted.len() - 1) as f64 * p).round() as usize]
        }
    };
    let mut hist = [0usize; 5];
    for &l in &loss {
        let b = [0.001, 0.003, 0.01, 0.03]
            .iter()
            .position(|&e| l < e)
            .unwrap_or(4);
        hist[b] += 1;
    }
    let ml = mean_se(&loss);
    ScoreSummary {
        decisions: s.len(),
        agreement: mean_se(&agree),
        mean_loss: ml,
        mean_loss_units: ml.0 * ONE as f64,
        mean_loss_stakes: ml.0 * 8.0,
        zero_share: s.iter().filter(|x| x.loss == 0).count() as f64 / s.len().max(1) as f64,
        quantiles: [q(0.5), q(0.9), q(0.99), q(1.0)],
        histogram: hist,
    }
}

/// Paired comparison of two scored methods on their common decisions: mean difference (A − B)
/// of agreement and of loss, each with a 95% confidence interval.
#[derive(Clone, Debug, Serialize)]
pub struct Paired {
    /// Common decisions.
    pub decisions: usize,
    /// Agreement difference A − B, and its 95% interval.
    pub agreement_diff: (f64, f64, f64),
    /// Loss difference A − B (money chance), and its 95% interval.
    pub loss_diff: (f64, f64, f64),
    /// Share of decisions where exactly one of the two agrees (the disagreement rate that sizes
    /// the set, §13.1 point 2).
    pub discordant: f64,
}

/// Compare two scored methods, paired by decision.
pub fn paired(a: &[Scored], b: &[Scored]) -> Paired {
    let mut da = Vec::new();
    let mut dl = Vec::new();
    let mut disc = 0usize;
    for x in a {
        if let Some(y) = b.iter().find(|y| y.id == x.id) {
            da.push(f64::from(u8::from(x.agree)) - f64::from(u8::from(y.agree)));
            dl.push(frac(x.loss) - frac(y.loss));
            if x.agree != y.agree {
                disc += 1;
            }
        }
    }
    let (ma, sa) = mean_se(&da);
    let (ml, sl) = mean_se(&dl);
    Paired {
        decisions: da.len(),
        agreement_diff: (ma, ma - 1.96 * sa, ma + 1.96 * sa),
        loss_diff: (ml, ml - 1.96 * sl, ml + 1.96 * sl),
        discordant: disc as f64 / da.len().max(1) as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use engine::cards::{Card, card};
    use equity::MatchState;
    use solver::FnLeaf;

    use crate::pimc::{SharedLeaf, world_row, world_values};
    use crate::worlds::RejectionSource;

    fn leaf() -> Arc<SharedLeaf> {
        Arc::new(FnLeaf(|s: &MatchState| {
            MoneyChance::from_weights(core::array::from_fn(|t| {
                1 + u128::from(s.scores[t]) + 40 * u128::from(s.games[t])
            }))
        }))
    }

    /// A small mixed set: exact-cheap trick-4 decisions and trick-3 ones (ids kept distinct).
    fn small_set() -> Vec<DecisionSpec> {
        let starts = collect_starts(400, 5);
        let mut ds = build_decisions(4, 12, 3, &starts);
        for mut d in build_decisions(3, 4, 4, &starts) {
            d.id += 100;
            ds.push(d);
        }
        ds
    }

    fn tmp(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("search-measure-{}-{tag}.jsonl", std::process::id()))
    }

    fn run(
        path: &Path,
        ds: &[DecisionSpec],
        job: &WorldsJob,
        threads: usize,
        tt_mb: usize,
    ) -> Vec<u8> {
        let l = leaf();
        let src = RejectionSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: Rules::HOUSE,
        };
        let cfg = SearchConfig {
            threads,
            tt_mb,
            ..SearchConfig::default()
        };
        worlds_file(path, ds, job, &ctx, &cfg, None, |_, _| {}).expect("runs");
        std::fs::read(path).expect("written")
    }

    fn clean(p: &Path) {
        let _ = std::fs::remove_file(p);
        let _ = std::fs::remove_file(timing_path(p));
    }

    /// The worlds file is byte-identical on 1, 3 and 8 threads, for exact and rollout PIMC; it
    /// is in batch order (earlier positions first); an interrupted run (torn last line) resumes to
    /// the same bytes; the rows equal the per-decision `world_values` the bots use, and the values
    /// of a new solver memory per world (each worker keeps one for all its worlds; with a 1 MB
    /// table the trick-3 solves also fill it and replace entries).
    #[test]
    fn worlds_files_thread_independent_and_resumable() {
        let ds = small_set();
        for (k, (method, tt_mb)) in [
            (Method::ExactPimc, 4),
            (Method::ExactPimc, 1),
            (Method::RolloutPimc, 4),
        ]
        .into_iter()
        .enumerate()
        {
            let job = WorldsJob {
                method,
                label: 1 + k as u64,
                seed: 7,
                from: 1,
                to: 5,
            };
            let mut files = Vec::new();
            for threads in [1, 3, 8] {
                let p = tmp(&format!("m{k}-t{threads}"));
                clean(&p);
                files.push(run(&p, &ds, &job, threads, tt_mb));
                let times: Vec<DecisionTiming> = read_jsonl(&timing_path(&p)).expect("timings");
                assert_eq!(times.len(), ds.len());
                clean(&p);
            }
            assert_eq!(files[0], files[1], "1 vs 3 threads");
            assert_eq!(files[0], files[2], "1 vs 8 threads");
            let text = String::from_utf8(files[0].clone()).expect("utf8");
            let rows: Vec<WorldRows> = text
                .lines()
                .map(|l| serde_json::from_str(l).expect("row"))
                .collect();
            assert_eq!(rows.len(), ds.len());
            let order: Vec<(u8, u8)> = rows.iter().map(|r| (r.trick, r.position)).collect();
            assert!(order.windows(2).all(|w| w[0] <= w[1]), "{order:?}");
            assert!(
                rows.iter()
                    .all(|r| r.rows.len() == 4 && r.hand_type.is_some())
            );
            // Interrupted after 5 decisions, mid-way through writing the 6th: resume.
            let lines: Vec<&str> = text.lines().collect();
            let p = tmp(&format!("m{k}-resume"));
            clean(&p);
            let mut part = lines[..5].join("\n");
            part.push('\n');
            part.push_str(&lines[5][..lines[5].len() / 2]);
            std::fs::write(&p, part).expect("write");
            assert_eq!(run(&p, &ds, &job, 3, tt_mb), files[0], "resumed run");
            // A file made with other settings is refused, not mixed.
            let l = leaf();
            let src = RejectionSource::new();
            let ctx = Ctx {
                leaf: &*l,
                source: &src,
                rules: Rules::HOUSE,
            };
            let other = WorldsJob { label: 9, ..job };
            let cfg = SearchConfig::default();
            assert!(worlds_file(&p, &ds, &other, &ctx, &cfg, None, |_, _| {}).is_err());
            clean(&p);
            // Same values as the per-decision path, and as a new solver memory for every world
            // (the table allocated per solve, as before tables were kept per worker).
            for (i, r) in rows.iter().enumerate() {
                let d = ds.iter().find(|d| d.id == r.id).expect("decision");
                let (_, v, legal) = d.replay();
                let (_, cands) = candidates(&v, &legal);
                let cfg = SearchConfig {
                    tt_mb,
                    ..SearchConfig::default()
                };
                let dseed = measure_seed(job.seed, job.label, d.id);
                let worlds = ctx.source.prepare(&v, ctx.rules);
                let fresh: Vec<Vec<[u64; 4]>> = (job.from..job.to)
                    .map(|w| {
                        let mut mem = SolveMem::new();
                        world_row(
                            &v, &*worlds, &cands, method, w, dseed, &ctx, &cfg, None, &mut mem,
                        )
                        .into_iter()
                        .map(|x| x.0)
                        .collect()
                    })
                    .collect();
                assert_eq!(r.rows, fresh, "decision {}: new memory per world", d.id);
                if i >= 3 {
                    continue;
                }
                let want: Vec<Vec<[u64; 4]>> =
                    world_values(&v, &cands, method, job.from, job.to, dseed, &ctx, &cfg)
                        .into_iter()
                        .map(|row| row.into_iter().map(|x| x.0).collect())
                        .collect();
                assert_eq!(r.rows, want);
                assert_eq!(r.cands, cands);
            }
        }
    }

    /// The position-aware schedule writes the same bytes as the explicit settings, on 1, 3 and 8
    /// threads, under plans that give worlds different thread counts and tables (costs and table
    /// needs raised so trick-3/4 worlds get 1–8 threads; budgets down to 1 MB tables, which forces
    /// replacement), including a resumed run and the all-one-thread plan that falls back to the
    /// plain workers; its timings record the threads each decision's worlds had.
    #[test]
    fn planned_schedules_write_identical_files() {
        let ds = small_set();
        let job = WorldsJob {
            method: Method::ExactPimc,
            label: 1,
            seed: 11,
            from: 0,
            to: 3,
        };
        let base = tmp("plan-base");
        clean(&base);
        let want = run(&base, &ds, &job, 2, 4);
        clean(&base);
        let l = leaf();
        let src = RejectionSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: Rules::HOUSE,
        };
        let spread = AutoParams {
            leader_cost_s: [0.0, 0.0, 0.0, 400.0, 100.0, 1.0],
            position_ratio: 1.5,
            leader_table_mb: [0.0, 0.0, 0.0, 64.0, 8.0, 1.0],
            target_s: Some(20.0),
            ..AutoParams::default()
        };
        let mut planned_seen = false;
        for (k, (threads, budget_mb, params)) in [
            (1, 64, spread.clone()),
            (3, 8, spread.clone()),
            (8, 512, spread.clone()),
            (8, 1, spread.clone()),
            (3, 64, AutoParams::default()),
        ]
        .into_iter()
        .enumerate()
        {
            let sched = Schedule::Auto {
                threads,
                budget_mb,
                params: Box::new(params.clone()),
            };
            let p = tmp(&format!("plan-{k}"));
            clean(&p);
            let cfg = SearchConfig::default();
            let prog = BatchProgress::default();
            worlds_file_sched(&p, &ds, &job, &ctx, &cfg, &sched, Some(&prog), |_, _| {})
                .expect("runs");
            assert_eq!(std::fs::read(&p).expect("written"), want, "plan {k}");
            let times: Vec<DecisionTiming> = read_jsonl(&timing_path(&p)).expect("timings");
            assert_eq!(times.len(), ds.len());
            let planned = prog.planned.load(Ordering::Relaxed);
            planned_seen |= planned;
            if planned {
                assert!(times.iter().all(|t| t.threads.is_some()), "plan {k}");
                if threads == 8 && budget_mb > 1 {
                    let spreads: std::collections::BTreeSet<u64> =
                        times.iter().filter_map(|t| t.threads).collect();
                    assert!(spreads.len() >= 2, "plan {k}: {spreads:?}");
                }
            } else {
                assert!(times.iter().all(|t| t.threads.is_none()), "plan {k}");
            }
            assert_eq!(
                prog.worlds_done.load(Ordering::Relaxed),
                (ds.len() * 3) as u64
            );
            // Interrupted after 4 decisions with a torn 5th line: resumed under the plan.
            let text = String::from_utf8(want.clone()).expect("utf8");
            let lines: Vec<&str> = text.lines().collect();
            let mut part = lines[..4].join("\n");
            part.push('\n');
            part.push_str(&lines[4][..lines[4].len() / 3]);
            std::fs::write(&p, part).expect("write");
            worlds_file_sched(&p, &ds, &job, &ctx, &cfg, &sched, None, |_, _| {}).expect("resumes");
            assert_eq!(
                std::fs::read(&p).expect("written"),
                want,
                "plan {k} resumed"
            );
            clean(&p);
        }
        assert!(planned_seen, "some plan used the position-aware runner");
    }

    /// Extending chosen decisions (`--ids` with `--from-world`) appends records that
    /// [`merge_rows`] joins into exactly the rows one longer run writes, whether the extension
    /// goes into the same file or another; re-running it adds nothing; a longer job over the first
    /// range is refused; overlapping records must agree and gaps are reported.
    #[test]
    fn extending_worlds_equals_one_longer_run() {
        let ds: Vec<DecisionSpec> = small_set()
            .into_iter()
            .filter(|d| d.trick == 4)
            .take(6)
            .collect();
        let job = |from, to| WorldsJob {
            method: Method::ExactPimc,
            label: 1,
            seed: 7,
            from,
            to,
        };
        let long = tmp("ext-long");
        clean(&long);
        run(&long, &ds, &job(0, 8), 2, 4);
        let (want, warn) = read_worlds(&[&long]).expect("reads");
        assert!(warn.is_empty() && want.len() == ds.len());
        let pick: Vec<DecisionSpec> = vec![ds[1].clone(), ds[4].clone()];
        let picked = |id: u64| pick.iter().any(|d| d.id == id);
        // Into the same file.
        let base = tmp("ext-base");
        clean(&base);
        run(&base, &ds, &job(0, 4), 2, 4);
        let once = run(&base, &pick, &job(4, 8), 3, 4);
        assert_eq!(
            run(&base, &pick, &job(4, 8), 1, 4),
            once,
            "re-running adds nothing"
        );
        let (got, warn) = read_worlds(&[&base]).expect("reads");
        assert!(warn.is_empty());
        assert_eq!(got.len(), ds.len());
        for r in &got {
            let w = want.iter().find(|w| w.id == r.id).expect("same decisions");
            if picked(r.id) {
                assert_eq!(r, w, "decision {}: extended = one longer run", r.id);
            } else {
                assert_eq!(r.rows[..], w.rows[..4]);
            }
        }
        // Into another file.
        let base2 = tmp("ext-base2");
        let ext = tmp("ext-more");
        clean(&base2);
        clean(&ext);
        run(&base2, &ds, &job(0, 4), 1, 4);
        run(&ext, &pick, &job(4, 8), 2, 4);
        assert_eq!(read_worlds(&[&base2, &ext]).expect("reads").0, got);
        assert_eq!(
            read_worlds(&[&ext, &base2]).expect("reads").0.len(),
            ds.len()
        );
        // `agree` sees the extended rows: the 4→8 step uses exactly the extended decisions.
        let (steps, _) = agreement_curve(&got, &[4, 8]);
        assert_eq!(steps[0].decisions, pick.len());
        // A longer job over the saved range is refused (extend with --from-world instead).
        let l = leaf();
        let src = RejectionSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: Rules::HOUSE,
        };
        let e = worlds_file(
            &base2,
            &ds,
            &job(0, 8),
            &ctx,
            &SearchConfig::default(),
            None,
            |_, _| {},
        )
        .expect_err("refused");
        assert!(e.contains("--from-world 4"), "{e}");
        // Overlapping records join when they agree; a conflict or a gap is reported.
        let w = &want[0];
        let seg = |from: usize, to: usize| WorldRows {
            from: from as u64,
            rows: w.rows[from..to].to_vec(),
            ..w.clone()
        };
        let (m, _) = merge_rows(vec![seg(3, 8), seg(0, 5)]).expect("joins");
        assert_eq!(m, vec![w.clone()]);
        let mut bad = seg(3, 8);
        bad.rows[0] = bad.rows[1].clone();
        if bad.rows[0] != w.rows[3] {
            assert!(merge_rows(vec![seg(0, 5), bad]).is_err());
        }
        let (m, warn) = merge_rows(vec![seg(0, 3), seg(5, 8)]).expect("gap");
        assert_eq!(m[0].rows.len(), 3);
        assert_eq!(warn.len(), 1);
        for p in [&long, &base, &base2, &ext] {
            clean(p);
        }
    }

    #[test]
    fn ids_and_steps() {
        assert_eq!(parse_ids("3,17, 42").expect("list"), vec![3, 17, 42]);
        let p = tmp("ids");
        std::fs::write(&p, "# chosen\n3 17\n42,7 # hard\n").expect("write");
        assert_eq!(
            parse_ids(p.to_str().expect("utf8")).expect("file"),
            vec![3, 17, 42, 7]
        );
        let _ = std::fs::remove_file(&p);
        assert!(parse_ids("no-such-file").is_err());
        let doubling = [10, 20, 40, 80];
        assert_eq!(curve_steps(&doubling), vec![(10, 20), (20, 40), (40, 80)]);
        let fine: Vec<usize> = (1..=8).map(|i| 10 * i).collect();
        assert_eq!(
            curve_steps(&fine),
            vec![(10, 20), (20, 40), (30, 60), (40, 80)]
        );
        assert_eq!(curve_steps(&[10, 30, 50]), vec![(10, 30), (30, 50)]);
    }

    #[test]
    fn hand_types() {
        let h = |cs: &[(Suit, u8)]| cs.iter().fold(0, |a, &(s, r)| a | bit(card(s, r)));
        use Suit::{Clubs, Diamonds, Hearts, Spades};
        let a = h(&[(Hearts, 5), (Hearts, 11), (Hearts, 1), (Spades, 13)]) | bit(JOKER);
        let t = hand_type(a, Hearts, false);
        assert_eq!((t.trumps, t.top_trumps, t.longest_plain), (4, 4, 1));
        assert_eq!(t.bands(), [3, 2, 1]);
        // Holds the 5 of hearts; void in clubs and diamonds.
        assert_eq!(
            (t.has_five, t.robber, t.voids),
            (Some(true), Some(false), Some(2))
        );
        assert_eq!(t.feature_bands(), Some([1, 0, 2]));
        let b = h(&[
            (Hearts, 1),
            (Spades, 1),
            (Spades, 2),
            (Clubs, 3),
            (Clubs, 4),
            (Clubs, 5),
        ]);
        let t = hand_type(b, Spades, true);
        assert_eq!((t.trumps, t.top_trumps, t.longest_plain), (3, 2, 3));
        // The 5 of clubs is not the 5 of trumps; the A♥ is a trump, so hearts is void too.
        assert_eq!(
            (t.has_five, t.robber, t.voids),
            (Some(false), Some(true), Some(2))
        );
        assert_eq!(t.feature_bands(), Some([0, 1, 2]));
        let t = hand_type(h(&[(Clubs, 2), (Spades, 3)]), Hearts, false);
        assert_eq!((t.trumps, t.top_trumps, t.longest_plain), (0, 0, 1));
        assert_eq!(t.bands(), [0, 0, 1]);
        assert_eq!(t.voids, Some(1), "diamonds");
        // Voids counting: none, the A♥ leaving hearts void, and all three plain suits (band 2+).
        let none = h(&[
            (Clubs, 2),
            (Diamonds, 3),
            (Spades, 4),
            (Hearts, 5),
            (Hearts, 6),
        ]);
        assert_eq!(hand_type(none, Hearts, false).voids, Some(0));
        let ah = h(&[
            (Hearts, 1),
            (Clubs, 2),
            (Spades, 3),
            (Diamonds, 4),
            (Diamonds, 5),
        ]);
        let t = hand_type(ah, Diamonds, false);
        assert_eq!((t.has_five, t.voids), (Some(true), Some(1)));
        let all = h(&[(Hearts, 5), (Hearts, 11), (Hearts, 1), (Hearts, 13)]) | bit(JOKER);
        let t = hand_type(all, Hearts, false);
        assert_eq!(t.voids, Some(3));
        assert_eq!(t.feature_bands(), Some([1, 0, 2]));
        assert_eq!(
            [0, 1].map(|b| feature_band_name(0, b)),
            ["has5-no", "has5-yes"]
        );
        assert_eq!(feature_band_name(1, 1), "robber-yes");
        assert_eq!(
            [0, 1, 2].map(|b| feature_band_name(2, b)),
            ["voids0", "voids1", "voids2+"]
        );
    }

    /// Records made before the 2026-10-14 features read with them unknown (no feature bands), and
    /// are written back byte for byte; new records carry them.
    #[test]
    fn hand_types_read_old_records() {
        let old = r#"{"trumps":2,"top_trumps":1,"longest_plain":3}"#;
        let t: HandType = serde_json::from_str(old).expect("old record");
        assert_eq!((t.has_five, t.robber, t.voids), (None, None, None));
        assert_eq!(t.feature_bands(), None);
        assert_eq!(t.bands(), [2, 1, 3]);
        assert_eq!(serde_json::to_string(&t).expect("json"), old);
        let new = HandType {
            has_five: Some(true),
            robber: Some(false),
            voids: Some(1),
            ..t
        };
        let s = serde_json::to_string(&new).expect("json");
        assert_eq!(
            s,
            r#"{"trumps":2,"top_trumps":1,"longest_plain":3,"has_five":true,"robber":false,"voids":1}"#
        );
        assert_eq!(serde_json::from_str::<HandType>(&s).expect("new"), new);
    }

    fn deal(dealer: u8, holds: &[(u8, Card)], pack: &[Card]) -> Match {
        let mut hands = [0 as Bits; 8];
        for &(s, c) in holds {
            hands[usize::from(s)] |= bit(c);
        }
        let deck = engine::game::build_deck(dealer, &hands, pack).expect("valid deal");
        Match::with_decks(dealer, vec![deck], 1, Rules::HOUSE)
    }

    fn apply_lowest(m: &mut Match) {
        let a = m.legal_actions()[0];
        m.apply(a).expect("legal");
    }

    /// Robber after the dealer takes a turned Joker, and the ace holder robbing at their own turn;
    /// the 5 of trumps known only once trumps are; all from the seat's own view.
    #[test]
    fn robber_and_five_from_the_view() {
        let ad = card(Suit::Diamonds, 1);
        let five_d = card(Suit::Diamonds, 5);
        // Dealer 0; seat 3 holds the A♦; the Joker is turned, then the 5♦.
        let mut m = deal(0, &[(3, ad)], &[JOKER, five_d]);
        assert_eq!(m.phase(), Phase::AwaitBurn(0));
        let v0 = m.view(0);
        assert!(view_robber(&v0), "the dealer took the Joker and must burn");
        assert_eq!(
            view_hand_type(&v0),
            None,
            "trumps unknown: no hand type, no 5 of trumps"
        );
        assert!((1..8).all(|s| view_hand_type(&m.view(s)).is_none()));
        apply_lowest(&mut m); // the dealer's burn; the 5♦ is turned: diamonds are trumps
        assert_eq!(m.view(0).public.trump, Some(Suit::Diamonds));
        let ht = |m: &Match, s: u8| view_hand_type(&m.view(s)).expect("trumps known");
        assert_eq!(ht(&m, 0).robber, Some(true), "robber after the Joker take");
        for s in 1..8 {
            let t = ht(&m, s);
            assert_eq!(t.robber, Some(false), "seat {s}");
            assert_eq!(t.has_five, Some(false), "the 5♦ is on the table");
        }
        // Seats 1 and 2 play; then seat 3 takes the 5♦ at its own turn and must burn.
        apply_lowest(&mut m);
        apply_lowest(&mut m);
        assert_eq!(m.phase(), Phase::AwaitBurn(3));
        let t = ht(&m, 3);
        assert_eq!(
            (t.robber, t.has_five),
            (Some(true), Some(true)),
            "robber at own turn"
        );
        assert!(
            !m.view(3).public.burn_order.iter().any(|&(s, _)| s == 3),
            "not yet in the burn order"
        );
        apply_lowest(&mut m); // seat 3's burn (any card but the A♦)
        assert_eq!(m.phase(), Phase::AwaitPlay(3));
        assert_eq!(ht(&m, 3).robber, Some(true));
        // Everyone else is unchanged, and stays so to the end of the hand.
        while m.view(0).public.trick_no <= 3 {
            for s in [1, 2, 4, 5, 6, 7] {
                assert_eq!(ht(&m, s).robber, Some(false), "seat {s}");
            }
            assert_eq!(ht(&m, 0).robber, Some(true));
            assert_eq!(ht(&m, 3).robber, Some(true));
            apply_lowest(&mut m);
        }
    }

    /// A turned ace is robbed by the dealer at once: trumps are known, so the dealer's burn
    /// decision already has a hand type, with robber set; the 5 of trumps is seen in hand.
    #[test]
    fn dealer_robs_a_turned_ace() {
        let as_ = card(Suit::Spades, 1);
        let five_s = card(Suit::Spades, 5);
        let mut m = deal(5, &[(2, five_s)], &[as_]);
        assert_eq!(m.phase(), Phase::AwaitBurn(5));
        let t = view_hand_type(&m.view(5)).expect("trumps known");
        assert_eq!(t.robber, Some(true));
        assert_eq!(
            view_hand_type(&m.view(2)).expect("known").has_five,
            Some(true)
        );
        assert_eq!(
            view_hand_type(&m.view(1)).expect("known").has_five,
            Some(false)
        );
        apply_lowest(&mut m);
        assert_eq!(
            view_hand_type(&m.view(5)).expect("known").robber,
            Some(true)
        );
        assert!(
            (0..8)
                .filter(|&s| s != 5)
                .all(|s| { view_hand_type(&m.view(s)).expect("known").robber == Some(false) })
        );
        // No rob at all: a plain turned card with the ace of trumps undealt (it is the next
        // card of the pack).
        let m = deal(5, &[], &[card(Suit::Clubs, 9), card(Suit::Clubs, 1)]);
        assert!((0..8).all(|s| view_hand_type(&m.view(s)).expect("known").robber == Some(false)));
    }

    /// `relabel` recomputes the hand types of records from their decision set (old records gain
    /// the features), leaves records of other decisions alone, and refuses a mismatched set.
    #[test]
    fn relabel_from_decisions() {
        let starts = collect_starts(400, 5);
        let ds = build_decisions(3, 16, 2, &starts);
        assert!(
            ds.iter()
                .all(|d| d.hand_type.is_some_and(|h| h.feature_bands().is_some()))
        );
        let rec = |d: &DecisionSpec| {
            let (_, v, legal) = d.replay();
            let (_, cands) = candidates(&v, &legal);
            let h = view_hand_type(&v).expect("trumps known at trick 3");
            assert_eq!(Some(h), d.hand_type);
            WorldRows {
                id: d.id,
                trick: d.trick,
                stratum: d.stratum,
                position: d.position,
                team: engine::team_of(d.seat),
                method: "exact".into(),
                label: 1,
                from: 0,
                cands,
                rows: Vec::new(),
                log2_worlds: None,
                // As written before the features existed.
                hand_type: Some(HandType {
                    has_five: None,
                    robber: None,
                    voids: None,
                    ..h
                }),
            }
        };
        let mut rows: Vec<WorldRows> = ds.iter().map(rec).collect();
        let mut other = rec(&ds[0]);
        other.id = 999;
        rows.push(other.clone());
        assert_eq!(relabel(&mut rows, &ds), Ok(ds.len()));
        for (r, d) in rows.iter().zip(&ds) {
            assert_eq!(r.hand_type, d.hand_type, "decision {}", d.id);
        }
        assert_eq!(rows[ds.len()], other, "not in the set: unchanged");
        // A decision set that does not match is refused.
        let mut bad = vec![rec(&ds[1])];
        bad[0].position = (bad[0].position + 1) % 8;
        assert!(relabel(&mut bad, &ds).is_err());
        let mut bad = vec![rec(&ds[1])];
        bad[0].id = ds[2].id;
        assert!(relabel(&mut bad, &ds).is_err());
        let mut wrong_trick = vec![rec(&ds[1])];
        wrong_trick[0].trick = 4;
        assert_eq!(relabel(&mut wrong_trick, &ds), Ok(0));
    }

    #[test]
    fn strata_classify() {
        let s = |scores: [u8; 4], games: [u8; 4]| HandStartState {
            dealer: 0,
            scores,
            bonus_taken: core::array::from_fn(|t| scores[t] % 5 == 1),
            games_won: games,
            hand_index: 0,
            game_index: games.iter().sum(),
        };
        assert_eq!(stratum_of(&s([0, 0, 0, 0], [0; 4])), 1);
        assert_eq!(stratum_of(&s([20, 0, 0, 0], [0; 4])), 1);
        assert_eq!(stratum_of(&s([25, 0, 0, 0], [0; 4])), 2);
        assert_eq!(stratum_of(&s([10, 0, 0, 0], [0, 1, 0, 0])), 3);
        assert_eq!(stratum_of(&s([0, 26, 0, 0], [0, 1, 0, 0])), 4);
        assert_eq!(stratum_of(&s([31, 0, 0, 0], [0; 4])), 4);
        assert_eq!(stratum_of(&s([28, 10, 0, 0], [0, 1, 0, 0])), 2);
    }

    #[test]
    fn decisions_are_deterministic_and_replay() {
        let starts = collect_starts(400, 5);
        assert!(starts.iter().all(|s| !s.is_empty()));
        let a = build_decisions(2, 16, 9, &starts);
        let b = build_decisions(2, 16, 9, &starts);
        assert_eq!(a, b);
        for d in &a {
            let (_, v, legal) = d.replay();
            assert_eq!(v.public.trick_cards.len(), usize::from(d.position));
            assert_eq!(v.public.trick_no, 2);
            assert!(candidates(&v, &legal).1.len() >= 2);
            assert_eq!(v.seat, d.seat);
        }
        let t1 = build_decisions(1, 16, 9, &starts);
        assert!(t1.iter().all(|d| d.replay().1.public.trick_no == 1));
    }

    #[test]
    fn isqrt_exact() {
        for n in [
            0u128,
            1,
            2,
            3,
            4,
            15,
            16,
            17,
            1 << 64,
            (1 << 66) + 12345,
            u128::MAX >> 2,
        ] {
            let r = crate::pimc::isqrt(n);
            assert!(r * r <= n && (r + 1) * (r + 1) > n, "{n}");
        }
    }
}
