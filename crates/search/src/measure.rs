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
use crate::pimc::{Candidate, Ctx, Method, SearchConfig, aggregate, candidates, world_row};

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

/// A hand's type, from the cards its holder can see (docs/M3-SPEC.md §13.1 item 9).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HandType {
    /// Trumps held (the trump suit, the Joker and the A♥).
    pub trumps: u8,
    /// Top trumps held: the 5 and J of trumps, the Joker, the A♥ and the ace of trumps (four
    /// cards when hearts are trumps).
    pub top_trumps: u8,
    /// Cards in the longest plain (non-trump) suit.
    pub longest_plain: u8,
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
}

/// The hand type of the cards `hand` with trump suit `trump`.
pub fn hand_type(hand: Bits, trump: Suit) -> HandType {
    use engine::rules::{ACE_HEARTS, ace_of_trumps, five_of, jack_of, plain_mask, trump_mask};
    let top = bit(five_of(trump))
        | bit(jack_of(trump))
        | bit(JOKER)
        | bit(ACE_HEARTS)
        | bit(ace_of_trumps(trump));
    HandType {
        trumps: count(hand & trump_mask(trump)),
        top_trumps: count(hand & top),
        longest_plain: Suit::ALL
            .iter()
            .filter(|&&s| s != trump)
            .map(|&s| count(hand & plain_mask(s, trump)))
            .max()
            .unwrap_or(0),
    }
}

/// The deciding seat's hand type in a view, once trumps are known.
pub fn view_hand_type(v: &SeatView) -> Option<HandType> {
    v.public.trump.map(|t| hand_type(v.hand, t))
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
    /// Worker threads in use.
    pub threads: AtomicU64,
    /// The parallel exact solve now running, when one worker runs `solve_fast_par`
    /// (`threads == 1`, `solve_threads > 1`): its subproblems done and total.
    pub par: ParProgress,
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
    (p.d.trick, p.d.position, Reverse(p.cands.len()), p.d.id)
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
                let p = &preps[j];
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
                    log2_worlds: worlds[j].count().map(|c| c.max(1).ilog2()),
                    hand_type: view_hand_type(&p.view),
                };
                let t = DecisionTiming {
                    id: p.d.id,
                    trick: p.d.trick,
                    position: p.d.position,
                    method: r.method.clone(),
                    label: job.label,
                    worlds: nw as u64,
                    world_us: us[j].load(Ordering::Relaxed),
                    max_world_us: max_us[j].load(Ordering::Relaxed),
                };
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
/// there are skipped; a torn last line is dropped and redone). The file is byte-identical for any
/// thread count, and whether or not the run was interrupted and resumed. Timings are appended to
/// [`timing_path`]`(out)`. `on_done` sees each decision as it is written (for logs). Returns the
/// number of decisions valued by this call.
pub fn worlds_file(
    out: &Path,
    ds: &[DecisionSpec],
    job: &WorldsJob,
    ctx: &Ctx,
    cfg: &SearchConfig,
    progress: Option<&BatchProgress>,
    mut on_done: impl FnMut(&WorldRows, &DecisionTiming),
) -> Result<usize, String> {
    let done: Vec<WorldRows> = if out.exists() {
        read_jsonl(out)?
    } else {
        Vec::new()
    };
    let nw = job.to - job.from;
    if let Some(r) = done.iter().find(|r| {
        r.method != method_name(job.method)
            || r.label != job.label
            || r.from != job.from
            || r.rows.len() as u64 != nw
    }) {
        return Err(format!(
            "{}: decision {} was made with other settings ({} label {} worlds {}..{}); \
             use another output file",
            out.display(),
            r.id,
            r.method,
            r.label,
            r.from,
            r.from + r.rows.len() as u64
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
        .filter(|d| !done.iter().any(|r| r.id == d.id))
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
    world_rows_batch(&todo, job, ctx, cfg, progress, |r, t| {
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

/// The agreement curve over nested world sets `grid` (each decision needs `max(grid)` worlds),
/// and the plateau N (§13.1): the smallest N where doubling changes the chosen card in under 1%
/// of decisions and changes the mean chosen-card value by under 0.001. The value condition is
/// applied to both readings, the literal one (difference of the means) and the loss of the
/// N-world choice under the 2N-world values, so cancellation cannot hide a change.
pub fn agreement_curve(rows: &[WorldRows], grid: &[usize]) -> (Vec<CurveStep>, Option<usize>) {
    let mut steps = Vec::new();
    for w in grid.windows(2) {
        let (n, n2) = (w[0], w[1]);
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

    use engine::cards::card;
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

    #[test]
    fn hand_types() {
        let h = |cs: &[(Suit, u8)]| cs.iter().fold(0, |a, &(s, r)| a | bit(card(s, r)));
        use Suit::{Clubs, Hearts, Spades};
        let a = h(&[(Hearts, 5), (Hearts, 11), (Hearts, 1), (Spades, 13)]) | bit(JOKER);
        let t = hand_type(a, Hearts);
        assert_eq!((t.trumps, t.top_trumps, t.longest_plain), (4, 4, 1));
        assert_eq!(t.bands(), [3, 2, 1]);
        let b = h(&[
            (Hearts, 1),
            (Spades, 1),
            (Spades, 2),
            (Clubs, 3),
            (Clubs, 4),
            (Clubs, 5),
        ]);
        let t = hand_type(b, Spades);
        assert_eq!((t.trumps, t.top_trumps, t.longest_plain), (3, 2, 3));
        let t = hand_type(h(&[(Clubs, 2), (Spades, 3)]), Hearts);
        assert_eq!((t.trumps, t.top_trumps, t.longest_plain), (0, 0, 1));
        assert_eq!(t.bands(), [0, 0, 1]);
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
