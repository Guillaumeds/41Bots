//! The student's teacher records (docs/STUDENT.md §3–§4): decisions at trick 1 or 2, each valued
//! by PIMC over D deals with the hybrid H_k (exact search to the start of trick k, the oracle O_k
//! there), every candidate's money chance in every deal stored with its stamps
//! (`student::record`).
//!
//! **Positions** (`student::record::SOURCE_*`): `rules`: the yardstick's generator
//! (`crate::measure::build_decision`'s way: a `rules-v1` self-play hand from a start state of the
//! decision's stratum, a uniformly chosen decision node at the trick and place with two or more
//! candidates); `mixed`: the same with every action before the decision a uniformly random legal
//! one with probability 1/4; `random`: every action before the decision random. Decision `i` is
//! from stratum `i mod 4 + 1` at place `(i div 4) mod 8`. Decision nodes are those of the app's
//! candidates ([`app_candidates`]), never the play straight after a seat's own rob (that card is
//! chosen with the rob: `crate::measure::play_after_own_rob`).
//!
//! **Deals**: deal `w` of decision `i` is `crate::pimc::world_match(view, worlds, w, dseed_i)`
//! (the belief sampler, the WASM API's and the yardstick's draw), with `dseed_i` from (seed,
//! source, trick, `i`); each candidate in each deal is valued by `crate::pimc::exact_world` with
//! the teacher's frontier (or exactly, for tests and small runs).
//!
//! **Determinism**: decision `i` depends only on (seed, source, trick, `i`), the deal range and the
//! teacher; decisions are made on any number of threads and written in index order
//! (`crate::ordered`), so a shard is byte-identical for any thread count (tested), and equal to
//! the plain PIMC values of the same view, seed and deals (tested against `crate::pimc`).

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use engine::game::{Action, HandStartState, Match, Phase, RULES_VERSION, Rules};
use engine::rng::Rng;
use engine::view::SeatView;
use equity::MoneyChance;
use serde_json::{Value, json};
use solver::frontier::Frontier;
use solver::{FrontierCut, SolveMem};
use student::position::{Cand, RawPosition};
use student::record::{
    KIND_JOINT, KIND_PLAY, SOURCE_MIXED, SOURCE_RANDOM, SOURCE_RULES, SOURCE_STUDENT, ShardInfo,
    ShardWriter, TeacherRecord, source_name,
};

use crate::measure::{VALUES_VERSION, play_after_own_rob, stratum_of};
use crate::ordered::{ordered, window_for};
use crate::pimc::{
    Candidate, Ctx, DecisionKind, SearchConfig, SharedLeaf, Students, WorldSolve, apply_candidate,
    candidates, exact_world, playout_world, student_candidate, world_match,
};
use crate::worlds::WorldSource;

/// Seed label of the positions.
pub const POSITION: u64 = 0x57D7;
/// Seed label of the decision seeds.
pub const DEALS: u64 = 0x57DE;

/// The candidate set's name in the stamps: the WASM API's.
pub const CANDIDATES: &str = "app-v1";

/// The sampler's name in the stamps.
pub const SAMPLER: &str = "belief-uniform";

/// The candidates of the decision the view awaits, as the app's API lists them
/// (`docs/WASM-API.md` §3): `crate::pimc::candidates` without a burn of the card the seat has
/// just taken (the app's rob burns from the hand before the take, so that burn has no app action).
pub fn app_candidates(view: &SeatView, legal: &[Action]) -> (DecisionKind, Vec<Candidate>) {
    let (kind, all) = candidates(view, legal);
    let taken = view
        .public
        .turned
        .iter()
        .rev()
        .find(|&&(_, by)| by == Some(view.seat))
        .map(|&(c, _)| c)
        .filter(|_| matches!(view.public.phase, Phase::AwaitBurn(_)));
    let cands = all
        .into_iter()
        .filter(|&(a, _)| !(matches!(a, Action::Burn(_)) && Some(a.card()) == taken))
        .collect();
    (kind, cands)
}

/// How often (per mille) an action before the decision is a uniformly random legal one.
fn policy_eps(source: u8) -> u64 {
    match source {
        SOURCE_RULES | SOURCE_STUDENT => 0,
        SOURCE_MIXED => 250,
        SOURCE_RANDOM => 1000,
        _ => panic!("source {source} is not generated here"),
    }
}

/// Who plays the actions before a decision: `rules-v1` (the sources `rules`, `mixed` and
/// `random`, the last two with their random share), or the students of the playout teacher at
/// tricks 1-2 with `rules-v1` for the dealer's burn (the source `student`: the positions the
/// students themselves reach, the DAgger step of docs/STUDENT.md §6).
#[derive(Clone, Copy)]
pub enum Inner<'a> {
    /// `rules-v1` throughout.
    Rules,
    /// The students at tricks 1-2 from each seat's own view.
    Students(Students<'a>, &'a SharedLeaf),
}

impl Inner<'_> {
    /// The players of `source` under `teacher`: the students for the source `student` (the playout
    /// teacher has them), `rules-v1` otherwise.
    pub fn of<'a>(source: u8, teacher: &Teacher<'a>, ctx: &Ctx<'a>) -> Result<Inner<'a>, String> {
        if source != SOURCE_STUDENT {
            return Ok(Inner::Rules);
        }
        match teacher {
            Teacher::Playout { students, .. } => Ok(Inner::Students(*students, ctx.leaf)),
            _ => Err(
                "the source `student` plays the positions with the playout teacher's \
                      students: it needs that teacher"
                    .to_string(),
            ),
        }
    }
}

/// One generator run (one shard).
#[derive(Clone, Copy, Debug)]
pub struct TeacherJob {
    /// The decisions' trick (1 or 2 for the student; 3 and 4 work too, exactly valued, for tests).
    pub trick: u8,
    /// Decisions in the shard.
    pub count: u64,
    /// The first decision's index (a shard is decisions `first..first + count`).
    pub first: u64,
    /// Deals per decision.
    pub deals: u32,
    /// The first deal's index (more deals for stored positions: the same seed, later deals).
    pub deal_from: u32,
    /// The generator's seed.
    pub seed: u64,
    /// Where positions come from (`SOURCE_RULES`, `SOURCE_MIXED`, `SOURCE_RANDOM`).
    pub source: u8,
    /// Worker threads (never changes a byte).
    pub threads: usize,
    /// Transposition-table megabytes per solve (never changes a value).
    pub tt_mb: usize,
    /// Hand starts collected for the strata (`crate::measure::collect_starts_with` with the job's
    /// seed): part of what a shard is, so it is in the sidecar.
    pub start_hands: u64,
}

/// The teacher's search in each deal.
#[derive(Clone, Copy)]
pub enum Teacher<'a> {
    /// Exact search to the end of the hand.
    Exact,
    /// The hybrid H_cut: exact to the start of trick `cut`, the network there.
    Hybrid {
        /// The frontier evaluator (an oracle network O_cut).
        net: &'a dyn Frontier,
        /// The frontier trick.
        cut: u8,
        /// The network file's SHA-256.
        sha256: &'a str,
        /// Exact leaves (`crate::exact_leaves`; `None`: none): in each deal the frontier
        /// positions this policy picks are valued by its expert (the exact solver, or a deeper
        /// hybrid: learned deferral, docs/m3-hybrid/learned-leaves.md).
        leaves: Option<&'a crate::exact_leaves::LeafPolicy>,
    },
    /// The playout teacher (docs/STUDENT.md, "The playout teacher"; PLAN §6.8): in each deal the
    /// candidate, then every seat played by the students S1 and S2 from its own view until the
    /// start of trick `cut`, exact search from there (`crate::pimc::playout_value`).
    Playout {
        /// The hidden-card players.
        students: Students<'a>,
        /// The trick at whose start the exact solve takes over (3: tricks 1-2 played out).
        cut: u8,
        /// S1's file SHA-256.
        sha1: &'a str,
        /// S2's file SHA-256.
        sha2: &'a str,
        /// S3's file SHA-256, when trick 3 is played out by S3 (a cut at 4 or later).
        sha3: Option<&'a str>,
        /// S4's file SHA-256, when trick 4 is played out by S4 too (a cut at 5 or later).
        sha4: Option<&'a str>,
    },
}

impl Teacher<'_> {
    /// The teacher's stamp: `exact`, or `hybrid:<sha256>:cut<k>`, with exact leaves
    /// `…+exact:<policy><its SHA-256s>` (`LeafPolicy::describe`, `LeafPolicy::values_suffix`).
    pub fn stamp(&self) -> String {
        match self {
            Teacher::Exact => "exact".into(),
            Teacher::Hybrid {
                cut,
                sha256,
                leaves,
                ..
            } => {
                let mut s = format!("hybrid:{sha256}:cut{cut}");
                if let Some(p) = leaves {
                    s.push_str(&format!("+exact:{}{}", p.describe(), p.values_suffix()));
                }
                s
            }
            // Without S3 the stamp is the one of before (shards made then keep their stamp).
            Teacher::Playout {
                cut,
                sha1,
                sha2,
                sha3: None,
                ..
            } => format!("playout:{sha1}:{sha2}:cut{cut}"),
            Teacher::Playout {
                cut,
                sha1,
                sha2,
                sha3: Some(sha3),
                sha4,
                ..
            } => match sha4 {
                None => format!("playout:{sha1}:{sha2}:{sha3}:cut{cut}"),
                Some(sha4) => format!("playout:{sha1}:{sha2}:{sha3}:{sha4}:cut{cut}"),
            },
        }
    }
}

/// Decision `i`'s seed: deal `w` is `world_match(view, worlds, w, seed)`.
pub fn decision_seed(job: &TeacherJob, i: u64) -> u64 {
    Rng::derive(
        job.seed,
        &[DEALS, u64::from(job.source), u64::from(job.trick), i],
    )
    .next_u64()
}

/// Decision `i`'s position, its stratum (1–4) and place in the trick (0–7); `inner` plays the
/// actions before it.
pub fn decision_position(
    job: &TeacherJob,
    i: u64,
    starts: &[Vec<HandStartState>; 4],
    inner: Inner<'_>,
) -> (RawPosition, u8, u8) {
    let st = (i % 4) as usize;
    let place = ((i / 4) % 8) as usize;
    let eps = policy_eps(job.source);
    let mut rng = Rng::derive(
        job.seed,
        &[POSITION, u64::from(job.source), u64::from(job.trick), i],
    );
    assert!(!starts[st].is_empty(), "stratum S{} has no states", st + 1);
    loop {
        let start = starts[st][rng.bounded(starts[st].len() as u64) as usize];
        let mut deck: [u8; engine::cards::DECK_SIZE] = core::array::from_fn(|k| k as u8);
        rng.shuffle(&mut deck);
        let mut m = Match::from_hand_start(start, deck, 0, Rules::HOUSE).expect("valid");
        let h = m.hand_index();
        let mut nodes: Vec<(Vec<Action>, u8)> = Vec::new();
        while m.phase() != Phase::MatchOver && m.hand_index() == h {
            let seat = m.to_act().expect("to act");
            let v = m.view(seat);
            if v.public.trick_no > job.trick {
                break;
            }
            let legal = m.legal_actions();
            if v.public.trick_no == job.trick
                && v.public.trick_cards.len() == place
                && app_candidates(&v, &legal).1.len() >= 2
                && !play_after_own_rob(&v)
            {
                nodes.push((m.actions().to_vec(), seat));
            }
            if eps > 0 && rng.bounded(1000) < eps {
                let a = legal[rng.bounded(legal.len() as u64) as usize];
                m.apply(a).expect("legal");
                continue;
            }
            let student = match inner {
                Inner::Students(students, leaf) => {
                    students.for_trick(v.public.trick_no).map(|net| (net, leaf))
                }
                Inner::Rules => None,
            };
            match student {
                Some((net, leaf)) => {
                    // A robber's burn and first card are one choice of the student's, as the
                    // app's (the play after its own rob is then never a node, as above).
                    let cand = student_candidate(&v, &legal, net, leaf)
                        .expect("a generated position is consistent with eight seats in play");
                    apply_candidate(&mut m, cand);
                }
                None => {
                    let a = bots::decide(&v, &Rules::HOUSE, &legal).0;
                    m.apply(a).expect("legal");
                }
            }
        }
        if nodes.is_empty() {
            continue;
        }
        let (all, seat) = nodes.swap_remove(rng.bounded(nodes.len() as u64) as usize);
        // The hand's actions only (a match started at the hand's start has no earlier ones).
        let pos = RawPosition {
            start,
            deck,
            actions: all,
            seat,
        };
        debug_assert_eq!(stratum_of(&start), st as u8 + 1);
        return (pos, st as u8 + 1, place as u8);
    }
}

/// Every candidate's value in deals `from..from + n` of the decision seen in `view`, deal by deal
/// (`[deal][candidate]`): exactly what `crate::pimc` and the WASM API compute for the same view,
/// seed and deals. `mem` is the calling thread's solver memory (it never changes a value).
#[allow(clippy::too_many_arguments)]
pub fn deal_values(
    view: &SeatView,
    cands: &[Candidate],
    dseed: u64,
    from: u32,
    n: u32,
    ctx: &Ctx,
    cfg: &SearchConfig,
    teacher: Teacher,
    mem: &mut SolveMem,
) -> (Vec<Vec<MoneyChance>>, Option<u128>) {
    let worlds = ctx.source.prepare(view, ctx.rules);
    let count = worlds.count();
    let rows = (from..from + n)
        .map(|w| {
            let m = world_match(view, &*worlds, u64::from(w), dseed, ctx.rules);
            let mut run = WorldSolve::default();
            if let Teacher::Playout { students, cut, .. } = teacher {
                return playout_world(&m, cands, &students, cut, ctx, cfg, mem, &mut run);
            }
            // Exact leaves: one wrapper per deal around the network.
            let x = match teacher {
                Teacher::Hybrid {
                    net,
                    leaves: Some(policy),
                    ..
                } => Some(crate::exact_leaves::ExactLeaves::new(
                    net,
                    ctx.leaf,
                    policy,
                    engine::team_of(view.seat),
                    solver::SolveFamily::of_match(&m),
                )),
                _ => None,
            };
            let cut = match teacher {
                Teacher::Exact | Teacher::Playout { .. } => None,
                Teacher::Hybrid { net, cut, .. } => Some(match &x {
                    Some(x) => FrontierCut::new(cut, x),
                    None => FrontierCut::new(cut, net),
                }),
            };
            exact_world(&m, cands, ctx, cfg, None, mem, &mut run, cut)
                .expect("nothing stops a solve here")
        })
        .collect();
    (rows, count)
}

/// The search settings of the teacher's solves.
pub fn teacher_config(tt_mb: usize) -> SearchConfig {
    SearchConfig {
        tt_mb,
        threads: 1,
        solve_threads: 1,
        ..SearchConfig::default()
    }
}

/// Decision `i`'s record.
pub fn make_record(
    job: &TeacherJob,
    i: u64,
    starts: &[Vec<HandStartState>; 4],
    ctx: &Ctx,
    teacher: Teacher,
    mem: &mut SolveMem,
) -> TeacherRecord {
    make_record_part(job, i, job.deal_from, job.deals, starts, ctx, teacher, mem)
}

/// Decision `i`'s record over deals `from..from + n` only (a block of its deals: the deals are
/// independent, so blocks put side by side in deal order are the whole record).
#[allow(clippy::too_many_arguments)]
pub fn make_record_part(
    job: &TeacherJob,
    i: u64,
    from: u32,
    n: u32,
    starts: &[Vec<HandStartState>; 4],
    ctx: &Ctx,
    teacher: Teacher,
    mem: &mut SolveMem,
) -> TeacherRecord {
    let inner = Inner::of(job.source, &teacher, ctx).expect("check_job allowed the source");
    let (pos, stratum, place) = decision_position(job, i, starts, inner);
    let r = pos.replay().expect("a generated position replays");
    let (kind, cands) = app_candidates(&r.view, &r.legal);
    let dseed = decision_seed(job, i);
    let (values, count) = deal_values(
        &r.view,
        &cands,
        dseed,
        from,
        n,
        ctx,
        &teacher_config(job.tt_mb),
        teacher,
        mem,
    );
    TeacherRecord {
        pos,
        source: job.source,
        gen_seed: job.seed,
        index: i,
        trick: job.trick,
        place,
        stratum,
        kind: match kind {
            DecisionKind::JointBurnPlay => KIND_JOINT,
            DecisionKind::Play => KIND_PLAY,
            k => panic!("a {k:?} decision is not a student decision"),
        },
        cands: cands.iter().map(|&(a, p)| -> Cand { (a, p) }).collect(),
        dseed,
        deal_from: from,
        log2_worlds_q8: count.map_or(0, belief::log2_q8),
        values,
    }
}

/// Deals per unit of work in [`teacher_records`]: a decision's deals are cut into blocks this big
/// so that the few heavy decisions (a G2 teacher's leader seats take minutes on one thread) are
/// spread over every thread at the end of a part instead of leaving the machine idle behind them.
pub const DEAL_BLOCK: u32 = 8;

/// A stored golden decision with its exact worlds as a student check reads a teacher decision
/// (`search student-check --golden-decisions D --golden-worlds W`): the position, the app's
/// candidates ([`app_candidates`], in their order), and per world each candidate's exact value
/// taken from the golden record by candidate (the golden set lists `crate::pimc::candidates`,
/// a superset). `None` for a decision the student never meets: fewer than two app candidates,
/// or the play straight after the seat's own rob. Refused: a record of another decision
/// (id, trick, seat's team), not exact, or missing a candidate.
pub fn golden_merged(
    d: &crate::measure::DecisionSpec,
    w: &crate::measure::WorldRows,
) -> Result<Option<student::data::Merged>, String> {
    let what = format!("golden decision {}", d.id);
    if w.id != d.id || w.trick != d.trick || w.position != d.position {
        return Err(format!("{what}: its worlds record is another decision's"));
    }
    if w.method != "exact" {
        return Err(format!(
            "{what}: its worlds are {}, not exact values",
            w.method
        ));
    }
    let deck: [u8; engine::cards::DECK_SIZE] = d
        .deck
        .as_slice()
        .try_into()
        .map_err(|_| format!("{what}: a deck of {} cards", d.deck.len()))?;
    let pos = RawPosition {
        start: d.start,
        deck,
        actions: d.actions.clone(),
        seat: d.seat,
    };
    let r = pos.replay()?;
    if engine::team_of(d.seat) != w.team {
        return Err(format!("{what}: its worlds record another team's decision"));
    }
    if play_after_own_rob(&r.view) {
        return Ok(None);
    }
    let (kind, cands) = app_candidates(&r.view, &r.legal);
    if cands.len() < 2 {
        return Ok(None);
    }
    let kind = match kind {
        DecisionKind::JointBurnPlay => KIND_JOINT,
        DecisionKind::Play => KIND_PLAY,
        k => {
            return Err(format!(
                "{what}: a {k:?} decision is not a student decision"
            ));
        }
    };
    let at: Vec<usize> = cands
        .iter()
        .map(|c| {
            w.cands
                .iter()
                .position(|x| x == c)
                .ok_or_else(|| format!("{what}: candidate {c:?} has no golden value"))
        })
        .collect::<Result<_, _>>()?;
    let values: Vec<Vec<MoneyChance>> = w
        .rows
        .iter()
        .map(|row| {
            at.iter()
                .map(|&i| {
                    row.get(i)
                        .map(|&v| MoneyChance(v))
                        .ok_or_else(|| format!("{what}: a world row is short"))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<_, _>>()?;
    if values.is_empty() {
        return Err(format!("{what}: no worlds"));
    }
    let n = values.len() as u32;
    let from = u32::try_from(w.from).map_err(|_| format!("{what}: first world too large"))?;
    let means = student::record::means_of(&values, cands.len());
    let head = student::record::RecordHead {
        pos_sha256: pos.sha256(),
        pos,
        source: 0,
        gen_seed: crate::oracle_labels::GOLDEN_SEED,
        index: d.id,
        trick: d.trick,
        place: d.position,
        stratum: d.stratum,
        kind,
        cands: cands.iter().map(|&(a, p)| -> Cand { (a, p) }).collect(),
        dseed: crate::measure::measure_seed(crate::oracle_labels::GOLDEN_SEED, w.label, d.id),
        deal_from: from,
        n_deals: n,
        log2_worlds_q8: w.log2_worlds.map_or(0, |x| (x.min(255) as u16) << 8),
        means,
        own_sd: vec![0; cands.len()],
        values_at: 0,
        values_sha256: [0; 32],
    };
    Ok(Some(student::data::Merged {
        head,
        deals: (from..from + n).collect(),
        values,
    }))
}

/// Counts of a shard's decisions, for its sidecar.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TeacherCounts {
    /// Decisions by place in the trick (0 = leader).
    pub by_place: [u64; 8],
    /// Decisions by stratum S1–S4.
    pub by_stratum: [u64; 4],
    /// Joint (rob) decisions.
    pub joint: u64,
    /// Candidates over all decisions.
    pub candidates: u64,
}

impl TeacherCounts {
    fn add(&mut self, r: &TeacherRecord) {
        self.by_place[usize::from(r.place)] += 1;
        self.by_stratum[usize::from(r.stratum - 1)] += 1;
        self.joint += u64::from(r.kind == KIND_JOINT);
        self.candidates += r.cands.len() as u64;
    }
}

/// Check a job (the seeds kept apart, the trick, the source, the teacher's frontier).
pub fn check_job(job: &TeacherJob, teacher: &Teacher, test_role: bool) -> Result<(), String> {
    if !(1..=4).contains(&job.trick) {
        return Err(format!("--trick {}: decisions at tricks 1..=4", job.trick));
    }
    if job.seed == crate::oracle_labels::GOLDEN_SEED {
        return Err("seed 1 is the golden sets' seed".into());
    }
    if test_role != (job.seed == student::STUDENT_TEST_SEED) {
        return Err(format!(
            "the test set (and only the test set) uses seed {} (student::STUDENT_TEST_SEED)",
            student::STUDENT_TEST_SEED
        ));
    }
    if job.deals == 0 || job.count == 0 {
        return Err("--deals and --count must be at least 1".into());
    }
    policy_eps_checked(job.source)?;
    if job.source == SOURCE_STUDENT && !matches!(teacher, Teacher::Playout { .. }) {
        return Err(
            "the source `student` plays the positions with the playout teacher's \
                    students: it needs --playout"
                .to_string(),
        );
    }
    if let Teacher::Hybrid { cut, .. } = teacher
        && (*cut <= job.trick || *cut > 5)
    {
        return Err(format!(
            "a trick-{} decision needs a network at the start of a later trick, not {cut}",
            job.trick
        ));
    }
    if let Teacher::Hybrid {
        cut,
        leaves: Some(p),
        ..
    } = teacher
    {
        p.check_for(*cut)?;
    }
    if let Teacher::Playout { students, sha3, .. } = teacher
        && students.s3.is_some() != sha3.is_some()
    {
        return Err("a playout teacher's S3 and its SHA-256 come together".into());
    }
    if let Teacher::Playout { students, sha4, .. } = teacher {
        if students.s4.is_some() != sha4.is_some() {
            return Err("a playout teacher's S4 and its SHA-256 come together".into());
        }
        if students.s4.is_some() && students.s3.is_none() {
            return Err("S4 plays trick 4 after S3's trick 3: give S3 too".into());
        }
    }
    if let Teacher::Playout { cut, .. } = teacher
        && !(1..=6).contains(cut)
    {
        return Err(format!(
            "a playout teacher's cut is the trick at whose start the exact solve takes over, \
             1 to 5, or 6 to play every trick out; not {cut}"
        ));
    }
    Ok(())
}

fn policy_eps_checked(source: u8) -> Result<u64, String> {
    match source {
        SOURCE_RULES | SOURCE_MIXED | SOURCE_RANDOM | SOURCE_STUDENT => Ok(policy_eps(source)),
        s => Err(format!(
            "source {} is not generated by this command",
            source_name(s)
        )),
    }
}

/// The records of decisions `first..first + count`, in order, on `job.threads` threads (the same
/// records for any thread count), each handed to `take`.
pub fn teacher_records(
    job: &TeacherJob,
    starts: &[Vec<HandStartState>; 4],
    ctx: &Ctx,
    teacher: Teacher,
    progress: Option<&AtomicUsize>,
    mut take: impl FnMut(TeacherRecord) -> Result<(), String>,
) -> Result<(), String> {
    // A unit is one block of a decision's deals (`DEAL_BLOCK`), handed on in index order, so a
    // decision's blocks arrive one after the other and are joined into its record.
    let blocks = u64::from(job.deals.div_ceil(DEAL_BLOCK).max(1));
    let units = job.count * blocks;
    let mut joined: Option<TeacherRecord> = None;
    ordered(
        job.threads,
        Some(units),
        // A window of the whole part: a few decisions cost minutes (G2 teachers), and a window of
        // 4 x threads + 16 let every worker run dry behind one of them (CPU use fell to a few
        // percent for minutes on end, run student-t2-gv2-250k-4). A unit is a few kilobytes, so
        // holding a part's worth of finished ones costs nothing; the order and the bytes are the
        // same for any window.
        window_for(job.threads).max(units),
        SolveMem::new,
        |mem, u| {
            let (k, b) = (u / blocks, (u % blocks) as u32);
            let from = job.deal_from + b * DEAL_BLOCK;
            let n = DEAL_BLOCK.min(job.deals - b * DEAL_BLOCK);
            let r = make_record_part(job, job.first + k, from, n, starts, ctx, teacher, mem);
            if u64::from(b) + 1 == blocks
                && let Some(p) = progress
            {
                p.fetch_add(1, Ordering::Relaxed);
            }
            r
        },
        |u, part| {
            let mut record = match joined.take() {
                None => part,
                Some(mut r) => {
                    r.values.extend(part.values);
                    r
                }
            };
            if u % blocks + 1 == blocks {
                // The first block holds the first deal's index; the record is the whole decision.
                record.deal_from = job.deal_from;
                take(record)?;
            } else {
                joined = Some(record);
            }
            Ok(true)
        },
    )
}

/// The stamps and settings of a shard, for its sidecar (`binary_sha256` and `equity_sha256` from
/// the caller; no timing or thread count, so the sidecar is the same for any machine).
pub fn sidecar(
    job: &TeacherJob,
    teacher: &Teacher,
    equity_sha256: &str,
    binary_sha256: &str,
    role: &str,
    counts: &TeacherCounts,
) -> Value {
    json!({
        "role": role,
        "teacher": teacher.stamp(),
        "equity_sha256": equity_sha256,
        "rules_version": RULES_VERSION,
        "values_version": VALUES_VERSION,
        "binary_sha256": binary_sha256,
        "sampler": SAMPLER,
        "candidates": CANDIDATES,
        "tie_rule": "first strictly best card in increasing card index",
        "trick": job.trick,
        "deals": job.deals,
        "deal_from": job.deal_from,
        "gen_seed": job.seed,
        "source": source_name(job.source),
        "first": job.first,
        "count": job.count,
        "start_hands": job.start_hands,
        "by_place": counts.by_place,
        "by_stratum": counts.by_stratum,
        "joint": counts.joint,
        "candidates_total": counts.candidates,
    })
}

/// Write a shard of `job`'s decisions to `path` (`NAME.npy`, with `NAME.values.npy` and
/// `NAME.npy.json`): the records streamed in order as they are made.
#[allow(clippy::too_many_arguments)]
pub fn write_teacher_shard(
    job: &TeacherJob,
    starts: &[Vec<HandStartState>; 4],
    ctx: &Ctx,
    teacher: Teacher,
    equity_sha256: &str,
    binary_sha256: &str,
    role: &str,
    path: &Path,
    progress: Option<&AtomicUsize>,
) -> Result<ShardInfo, String> {
    check_job(job, &teacher, role == "test")?;
    let mut w = ShardWriter::create(path)?;
    let mut counts = TeacherCounts::default();
    teacher_records(job, starts, ctx, teacher, progress, |r| {
        counts.add(&r);
        w.push(&r)
    })?;
    w.finish(sidecar(
        job,
        &teacher,
        equity_sha256,
        binary_sha256,
        role,
        &counts,
    ))
}

/// The view's world source (the belief sampler) wrapped for a context, for callers that have
/// none at hand.
pub fn belief_ctx<'a>(leaf: &'a crate::pimc::SharedLeaf, source: &'a dyn WorldSource) -> Ctx<'a> {
    Ctx {
        leaf,
        source,
        rules: Rules::HOUSE,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use super::*;
    use crate::measure::collect_starts;
    use crate::pimc::{Method, aggregate, world_values};
    use crate::worlds::BeliefSource;
    use nn::synthetic::{SyntheticLeaf, random_net};
    use student::record::ShardReader;

    fn leaf() -> Arc<dyn solver::Leaf + Send + Sync> {
        Arc::new(SyntheticLeaf)
    }

    fn oracle(k: u8) -> nn::Oracle {
        let net = random_net(u64::from(k) * 7 + 1, &[32, 16], nn::outcome::vocab(k), 100);
        let meta = BTreeMap::from([("head".to_string(), "outcome-v1".to_string())]);
        nn::Oracle::from_bytes(&nn::oracle::write_net(&net, k, &meta), leaf()).expect("loads")
    }

    fn job(trick: u8, count: u64, deals: u32, source: u8, threads: usize) -> TeacherJob {
        TeacherJob {
            trick,
            count,
            first: 0,
            deals,
            deal_from: 0,
            seed: 11,
            source,
            threads,
            tt_mb: 8,
            start_hands: 400,
        }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("student-teacher-{}", std::process::id()));
        std::fs::create_dir_all(&d).expect("dir");
        d.join(name)
    }

    /// The deal values are exactly `crate::pimc`'s for the same view, seed and deals (exact
    /// search at trick 3, where it is cheap), and the record's means are `aggregate`'s.
    #[test]
    fn deal_values_are_pimcs() {
        let starts = collect_starts(400, 5);
        let l = leaf();
        let source = BeliefSource::new();
        let ctx = belief_ctx(&*l, &source);
        let j = job(3, 6, 5, SOURCE_RULES, 1);
        let mut mem = SolveMem::new();
        for i in 0..j.count {
            let r = make_record(&j, i, &starts, &ctx, Teacher::Exact, &mut mem);
            let rp = r.pos.replay().expect("replays");
            let (_, cands) = app_candidates(&rp.view, &rp.legal);
            assert_eq!(r.cands, cands);
            let cfg = teacher_config(j.tt_mb);
            let rows = world_values(
                &rp.view,
                &cands,
                Method::ExactPimc,
                0,
                u64::from(j.deals),
                r.dseed,
                &ctx,
                &cfg,
            );
            assert_eq!(r.values, rows, "decision {i}");
            let (means, _, sds) = aggregate(&rows, cands.len(), engine::team_of(rp.view.seat));
            assert_eq!(r.means(), means);
            assert_eq!(r.own_sd(), sds);
            assert_eq!(r.trick, 3);
            assert_eq!(rp.view.public.trick_cards.len(), usize::from(r.place));
        }
    }

    /// A decision's deals are worked in blocks of `DEAL_BLOCK` on any thread and joined: the records
    /// are exactly the whole-decision ones, for deals that are not a multiple of the block, from a
    /// first deal other than 0, and for any thread count.
    #[test]
    fn blocks_of_deals_join_into_the_whole_decisions_records() {
        let starts = collect_starts(400, 5);
        let l = leaf();
        let source = BeliefSource::new();
        let ctx = belief_ctx(&*l, &source);
        for (deals, from, threads) in [(19, 0, 3), (19, 5, 1), (8, 0, 2), (3, 2, 4), (16, 0, 3)] {
            let mut j = job(3, 5, deals, SOURCE_RULES, threads);
            j.deal_from = from;
            let mut mem = SolveMem::new();
            let whole: Vec<_> = (0..j.count)
                .map(|i| make_record(&j, i, &starts, &ctx, Teacher::Exact, &mut mem))
                .collect();
            let mut got = Vec::new();
            teacher_records(&j, &starts, &ctx, Teacher::Exact, None, |r| {
                got.push(r);
                Ok(())
            })
            .expect("records");
            assert_eq!(got, whole, "{deals} deals from {from} on {threads} threads");
            assert!(
                got.iter()
                    .all(|r| r.deal_from == from && r.values.len() == deals as usize)
            );
        }
    }

    /// With a network, every deal's values are the hybrid's (`exact_world` with the frontier, as
    /// the WASM API calls it), and more deals later continue the same deals by index.
    #[test]
    fn hybrid_values_are_the_wasm_apis_and_deals_extend_by_index() {
        let starts = collect_starts(400, 6);
        let l = leaf();
        let source = BeliefSource::new();
        let ctx = belief_ctx(&*l, &source);
        let o = oracle(3);
        let teacher = Teacher::Hybrid {
            net: &o,
            cut: 3,
            sha256: &o.sha256,
            leaves: None,
        };
        let j = job(2, 3, 4, SOURCE_MIXED, 1);
        let mut more = j;
        more.deal_from = 4;
        more.deals = 2;
        let mut mem = SolveMem::new();
        for i in 0..j.count {
            let r = make_record(&j, i, &starts, &ctx, teacher, &mut mem);
            let rp = r.pos.replay().expect("replays");
            let worlds = ctx.source.prepare(&rp.view, Rules::HOUSE);
            let cfg = teacher_config(8);
            for (w, row) in r.values.iter().enumerate() {
                let m = world_match(&rp.view, &*worlds, w as u64, r.dseed, Rules::HOUSE);
                let mut run = WorldSolve::default();
                let want = exact_world(
                    &m,
                    &r.cands,
                    &ctx,
                    &cfg,
                    None,
                    &mut SolveMem::new(),
                    &mut run,
                    Some(FrontierCut::new(3, &o)),
                )
                .expect("solved");
                assert_eq!(row, &want, "decision {i} deal {w}");
            }
            let r2 = make_record(&more, i, &starts, &ctx, teacher, &mut mem);
            assert_eq!((r2.pos.clone(), r2.dseed), (r.pos.clone(), r.dseed));
            let mut wide = j;
            wide.deals = 6;
            let r6 = make_record(&wide, i, &starts, &ctx, teacher, &mut mem);
            let joined: Vec<_> = r.values.iter().chain(&r2.values).cloned().collect();
            assert_eq!(r6.values, joined);
        }
    }

    /// Exact leaves on the teacher: `all` (every frontier leaf solved exactly) gives the exact
    /// teacher's values, deal by deal, and its stamp names the policy; no policy is the hybrid
    /// as before.
    #[test]
    fn teacher_exact_leaves_all_is_the_exact_teacher() {
        let starts = collect_starts(400, 6);
        let l = leaf();
        let source = BeliefSource::new();
        let ctx = belief_ctx(&*l, &source);
        let o = oracle(4);
        let all = crate::exact_leaves::LeafPolicy::all(8);
        let with = Teacher::Hybrid {
            net: &o,
            cut: 4,
            sha256: &o.sha256,
            leaves: Some(&all),
        };
        let plain = Teacher::Hybrid {
            net: &o,
            cut: 4,
            sha256: &o.sha256,
            leaves: None,
        };
        assert_eq!(plain.stamp(), format!("hybrid:{}:cut4", o.sha256));
        assert_eq!(with.stamp(), format!("hybrid:{}:cut4+exact:all", o.sha256));
        let j = job(3, 4, 3, SOURCE_RULES, 1);
        check_job(&j, &with, false).expect("checks");
        let mut mem = SolveMem::new();
        for i in 0..j.count {
            let a = make_record(&j, i, &starts, &ctx, with, &mut mem);
            let b = make_record(&j, i, &starts, &ctx, Teacher::Exact, &mut mem);
            assert_eq!(a.values, b.values, "decision {i}");
        }
    }

    /// A shard is the same bytes for 1 and 3 threads, and reads back as the records made one by
    /// one.
    #[test]
    fn shards_are_byte_identical_for_any_thread_count() {
        let starts = collect_starts(400, 7);
        let l = leaf();
        let source = BeliefSource::new();
        let ctx = belief_ctx(&*l, &source);
        let o = oracle(2);
        let teacher = Teacher::Hybrid {
            net: &o,
            cut: 2,
            sha256: &o.sha256,
            leaves: None,
        };
        let mut keys = Vec::new();
        for threads in [1, 3] {
            let j = job(1, 10, 2, SOURCE_RULES, threads);
            let p = tmp(&format!("t1-{threads}.npy"));
            let info =
                write_teacher_shard(&j, &starts, &ctx, teacher, "eq", "bin", "train", &p, None)
                    .expect("written");
            let values = std::fs::read(student::record::values_path(&p)).expect("values");
            let side = std::fs::read(student::record::sidecar_path(&p)).expect("sidecar");
            keys.push((info.clone(), values, side));
            let mut rd = ShardReader::open(&p).expect("opens");
            rd.verify_values_file().expect("values file");
            let mut mem = SolveMem::new();
            for k in 0..j.count {
                let want = make_record(&j, k, &starts, &ctx, teacher, &mut mem);
                assert_eq!(rd.record(k as usize).expect("record"), want);
                assert_eq!(want.trick, 1);
            }
        }
        assert_eq!(keys[0], keys[1]);
        // Joint rob decisions appear at trick 1 (decisions at the robber's place).
        let j = job(1, 64, 1, SOURCE_RULES, 2);
        let mut joint = 0;
        let mut mem = SolveMem::new();
        for k in 0..j.count {
            let r = make_record(&j, k, &starts, &ctx, teacher, &mut mem);
            joint += u32::from(r.kind == KIND_JOINT);
            let rp = r.pos.replay().expect("replays");
            assert!(!crate::measure::play_after_own_rob(&rp.view));
            if r.kind == KIND_JOINT {
                // Never a burn of the card just taken.
                let taken = rp
                    .view
                    .public
                    .turned
                    .iter()
                    .find(|t| t.1 == Some(rp.view.seat));
                let taken = taken.expect("the robber took the turned card").0;
                assert!(r.cands.iter().all(|c| c.0 != Action::Burn(taken)));
            }
        }
        assert!(joint > 0, "no rob in 64 trick-1 decisions");
    }

    fn students() -> (student::net::StudentNet, student::net::StudentNet) {
        (
            student::net::random_net(4, &[16, 8], 12),
            student::net::random_net(5, &[16, 8], 12),
        )
    }

    /// With the cut at (or below) the decision's own trick nothing is played out: the playout
    /// teacher is the exact teacher, byte for byte, joint rob decisions included.
    #[test]
    fn playout_cut_at_the_decisions_trick_is_the_exact_teacher() {
        let starts = collect_starts(400, 5);
        let l = leaf();
        let source = BeliefSource::new();
        let ctx = belief_ctx(&*l, &source);
        let (s1, s2) = students();
        for (trick, cut) in [(3, 3), (4, 4), (4, 2)] {
            let teacher = Teacher::Playout {
                students: Students::two(&s1, &s2),
                cut,
                sha1: "a",
                sha2: "b",
                sha3: None,
                sha4: None,
            };
            let j = job(trick, 5, 4, SOURCE_RULES, 1);
            let mut mem = SolveMem::new();
            for i in 0..j.count {
                let a = make_record(&j, i, &starts, &ctx, Teacher::Exact, &mut mem);
                let b = make_record(&j, i, &starts, &ctx, teacher, &mut mem);
                assert_eq!(a, b, "trick {trick} cut {cut} decision {i}");
            }
        }
        assert_eq!(
            Teacher::Playout {
                students: Students::two(&s1, &s2),
                cut: 3,
                sha1: "a",
                sha2: "b",
                sha3: None,
                sha4: None,
            }
            .stamp(),
            "playout:a:b:cut3"
        );
        // With S3 the stamp names it; S3 and its SHA-256 come together.
        let s3 = student::net::random_net(6, &[16, 8], 12);
        let with3 = Students {
            s3: Some(&s3),
            ..Students::two(&s1, &s2)
        };
        assert_eq!(
            Teacher::Playout {
                students: with3,
                cut: 4,
                sha1: "a",
                sha2: "b",
                sha3: Some("c"),
                sha4: None,
            }
            .stamp(),
            "playout:a:b:c:cut4"
        );
        // S4 too: the stamp names both; S4 without S3, or without its SHA-256, is refused.
        let s4 = student::net::random_net(9, &[16, 8], 12);
        let with4 = Students {
            s4: Some(&s4),
            ..with3
        };
        assert_eq!(
            Teacher::Playout {
                students: with4,
                cut: 5,
                sha1: "a",
                sha2: "b",
                sha3: Some("c"),
                sha4: Some("d"),
            }
            .stamp(),
            "playout:a:b:c:d:cut5"
        );
        let j3 = job(3, 5, 4, SOURCE_RULES, 1);
        let only4 = Students {
            s4: Some(&s4),
            ..Students::two(&s1, &s2)
        };
        for (st, sha3, sha4, word) in [
            (only4, None, Some("d"), "give S3 too"),
            (with4, Some("c"), None, "S4 and its SHA-256"),
        ] {
            let t = Teacher::Playout {
                students: st,
                cut: 5,
                sha1: "a",
                sha2: "b",
                sha3,
                sha4,
            };
            assert!(
                check_job(&j3, &t, false).unwrap_err().contains(word),
                "{word}"
            );
        }
        let j = job(3, 5, 4, SOURCE_RULES, 1);
        for (st, sha3) in [(with3, None), (Students::two(&s1, &s2), Some("c"))] {
            let t = Teacher::Playout {
                students: st,
                cut: 4,
                sha1: "a",
                sha2: "b",
                sha3,
                sha4: None,
            };
            assert!(
                check_job(&j, &t, false)
                    .unwrap_err()
                    .contains("come together")
            );
        }
    }

    /// Trick-2 decisions played out by the students to the start of trick 3 and solved exactly
    /// from there: legal all the way (the playout would panic otherwise), the same bytes on 1 and
    /// 3 threads, every shard record `make_record`'s, values that are money chances, and not the
    /// hybrid teacher's values at the same cut (the students' play counts, where the hybrid
    /// searches the trick open-handed). The exact teacher at trick 2 is a four-trick solve, too
    /// slow for a unit test; the cut-equals-trick test above proves the exact tail.
    #[test]
    fn playout_shards_are_byte_identical_for_any_thread_count() {
        let starts = collect_starts(400, 7);
        let l = leaf();
        let source = BeliefSource::new();
        let ctx = belief_ctx(&*l, &source);
        let (s1, s2) = students();
        let teacher = Teacher::Playout {
            students: Students::two(&s1, &s2),
            cut: 3,
            sha1: "a",
            sha2: "b",
            sha3: None,
            sha4: None,
        };
        let mut keys = Vec::new();
        for threads in [1, 3] {
            let j = job(2, 4, 3, SOURCE_RULES, threads);
            let p = tmp(&format!("t2-playout-{threads}.npy"));
            let info =
                write_teacher_shard(&j, &starts, &ctx, teacher, "eq", "bin", "train", &p, None)
                    .expect("written");
            let values = std::fs::read(student::record::values_path(&p)).expect("values");
            let side = std::fs::read(student::record::sidecar_path(&p)).expect("sidecar");
            assert!(
                String::from_utf8_lossy(&side).contains("playout:a:b:cut3"),
                "the sidecar names the teacher"
            );
            keys.push((info.clone(), values, side));
            let mut rd = ShardReader::open(&p).expect("opens");
            rd.verify_values_file().expect("values file");
            let mut mem = SolveMem::new();
            for k in 0..j.count {
                let want = make_record(&j, k, &starts, &ctx, teacher, &mut mem);
                assert_eq!(rd.record(k as usize).expect("record"), want);
                assert_eq!(want.trick, 2);
                for row in &want.values {
                    for v in row {
                        assert_eq!(
                            v.0.iter().map(|&x| u128::from(x)).sum::<u128>(),
                            1u128 << 32
                        );
                    }
                }
            }
        }
        assert_eq!(keys[0], keys[1]);
        let o = oracle(3);
        let hybrid = Teacher::Hybrid {
            net: &o,
            cut: 3,
            sha256: &o.sha256,
            leaves: None,
        };
        let j = job(2, 4, 3, SOURCE_RULES, 1);
        let mut mem = SolveMem::new();
        let mut differs = false;
        for i in 0..j.count {
            let a = make_record(&j, i, &starts, &ctx, hybrid, &mut mem);
            let b = make_record(&j, i, &starts, &ctx, teacher, &mut mem);
            assert_eq!((a.cands.clone(), a.dseed), (b.cands.clone(), b.dseed));
            differs |= a.values != b.values;
        }
        assert!(
            differs,
            "the students' play changed no value in 4 trick-2 decisions"
        );
    }

    /// Trick-1 decisions: the students play the rest of trick 1 (a later rob in the world is a
    /// joint decision of theirs) and trick 2, exact from trick 3; a cut of 6 plays every trick
    /// out and values the end by the leaf rule.
    #[test]
    fn playout_from_trick_one_plays_the_robs_and_both_tricks() {
        let starts = collect_starts(400, 7);
        let l = leaf();
        let source = BeliefSource::new();
        let ctx = belief_ctx(&*l, &source);
        let (s1, s2) = students();
        let mut mem = SolveMem::new();
        for cut in [3u8, 6] {
            let teacher = Teacher::Playout {
                students: Students::two(&s1, &s2),
                cut,
                sha1: "a",
                sha2: "b",
                sha3: None,
                sha4: None,
            };
            let j = job(1, 12, 2, SOURCE_RULES, 1);
            check_job(&j, &teacher, false).expect("a valid job");
            let mut robs = 0;
            for i in 0..j.count {
                let r = make_record(&j, i, &starts, &ctx, teacher, &mut mem);
                robs += u32::from(r.kind == KIND_JOINT);
                assert_eq!(r.values.len(), 2);
                assert!(r.values.iter().all(|row| row.len() == r.cands.len()));
            }
            assert!(robs > 0, "no joint rob decision in 12 trick-1 decisions");
        }
        let bad = Teacher::Playout {
            students: Students::two(&s1, &s2),
            cut: 7,
            sha1: "a",
            sha2: "b",
            sha3: None,
            sha4: None,
        };
        assert!(check_job(&job(1, 1, 1, SOURCE_RULES, 1), &bad, false).is_err());
    }

    /// The source `student`: the students play the actions before the decision (the positions
    /// they reach themselves), the dealer's burn by `rules-v1`; the positions differ from the
    /// rules source's at the same seed (the students choose otherwise somewhere), every action of
    /// a position replays, a robber's play after its own rob is never a node, and the source needs
    /// the playout teacher (the hybrid has no students to play with).
    #[test]
    fn the_student_source_plays_the_positions_with_the_students() {
        let starts = collect_starts(400, 7);
        let l = leaf();
        let source = BeliefSource::new();
        let ctx = belief_ctx(&*l, &source);
        let (s1, s2) = students();
        let teacher = Teacher::Playout {
            students: Students::two(&s1, &s2),
            cut: 3,
            sha1: "a",
            sha2: "b",
            sha3: None,
            sha4: None,
        };
        let mut differ = 0;
        for trick in [1u8, 2] {
            let st = job(trick, 16, 1, SOURCE_STUDENT, 1);
            check_job(&st, &teacher, false).expect("a valid job");
            let ru = job(trick, 16, 1, SOURCE_RULES, 1);
            let inner = Inner::of(SOURCE_STUDENT, &teacher, &ctx).expect("students");
            for i in 0..st.count {
                let (a, stratum, place) = decision_position(&st, i, &starts, inner);
                let (b, _, _) = decision_position(&ru, i, &starts, Inner::Rules);
                assert_eq!((stratum, place), ((i % 4) as u8 + 1, ((i / 4) % 8) as u8));
                let r = a.replay().expect("replays");
                assert_eq!(r.view.public.trick_no, trick);
                assert_eq!(r.view.public.trick_cards.len(), usize::from(place));
                assert!(app_candidates(&r.view, &r.legal).1.len() >= 2);
                assert!(!play_after_own_rob(&r.view));
                differ += u32::from(a.actions != b.actions || a.deck != b.deck);
            }
        }
        assert!(
            differ > 0,
            "the students never played otherwise than rules-v1"
        );
        // A record of the student source is one record like any other.
        let mut mem = SolveMem::new();
        let st = job(2, 2, 2, SOURCE_STUDENT, 1);
        for i in 0..st.count {
            let r = make_record(&st, i, &starts, &ctx, teacher, &mut mem);
            assert_eq!(r.source, SOURCE_STUDENT);
            assert_eq!(r.values.len(), 2);
        }
        let o = oracle(3);
        let hybrid = Teacher::Hybrid {
            net: &o,
            cut: 3,
            sha256: &o.sha256,
            leaves: None,
        };
        assert!(check_job(&st, &hybrid, false).is_err());
        assert!(Inner::of(SOURCE_STUDENT, &hybrid, &ctx).is_err());
        assert!(matches!(
            Inner::of(SOURCE_MIXED, &hybrid, &ctx).expect("rules"),
            Inner::Rules
        ));
    }

    #[test]
    fn jobs_keep_the_seeds_apart() {
        let o = oracle(3);
        let h = Teacher::Hybrid {
            net: &o,
            cut: 3,
            sha256: "x",
            leaves: None,
        };
        let mut j = job(2, 1, 1, SOURCE_RULES, 1);
        assert!(check_job(&j, &h, false).is_ok());
        assert!(check_job(&j, &h, true).is_err());
        j.seed = 1;
        assert!(check_job(&j, &h, false).is_err());
        j.seed = student::STUDENT_TEST_SEED;
        assert!(check_job(&j, &h, false).is_err());
        assert!(check_job(&j, &h, true).is_ok());
        j.trick = 3;
        assert!(check_job(&j, &h, true).is_err(), "cut 3 at trick 3");
        assert!(check_job(&j, &Teacher::Exact, true).is_ok());
    }

    /// More deals added later merge exactly: shards of deals 0..2 and 2..5 of the same decisions
    /// read as one decision each, with the values, means and feature rows of one shard of deals
    /// 0..5; overlapping deals are refused; the check and the encoding run end to end on them.
    #[test]
    fn added_deals_merge_exactly_and_the_check_runs() {
        use student::data::{Shards, Uniform, feature_row};
        let starts = collect_starts(400, 8);
        let l = leaf();
        let source = BeliefSource::new();
        let ctx = belief_ctx(&*l, &source);
        let o = oracle(3);
        let teacher = Teacher::Hybrid {
            net: &o,
            cut: 3,
            sha256: &o.sha256,
            leaves: None,
        };
        let mut a = job(2, 6, 2, SOURCE_RULES, 2);
        let mut b = a;
        b.deal_from = 2;
        b.deals = 3;
        let mut all = a;
        all.deals = 5;
        let pa = tmp("m-a.npy");
        let pb = tmp("m-b.npy");
        let pall = tmp("m-all.npy");
        for (j, p) in [(&a, &pa), (&b, &pb), (&all, &pall)] {
            write_teacher_shard(j, &starts, &ctx, teacher, "eq", "bin", "train", p, None)
                .expect("written");
        }
        let mut two = Shards::open(&[&pb, &pa], &[]).expect("opens");
        let mut one = Shards::open(&[&pall], &[]).expect("opens");
        assert_eq!((two.groups.len(), one.groups.len()), (6, 6));
        for g in 0..6 {
            let m2 = two.merged(g).expect("merged");
            let m1 = one.merged(g).expect("merged");
            assert_eq!(m2.deals, vec![0, 1, 2, 3, 4]);
            assert_eq!(m2.values, m1.values);
            assert_eq!(m2.means(), m1.means());
            assert_eq!(
                feature_row(&m2, &*l, &Uniform).expect("encodes"),
                feature_row(&m1, &*l, &Uniform).expect("encodes")
            );
        }
        // Deal weights (docs/STUDENT.md §3 item 2): weight 1 everywhere gives the plain means;
        // weight 0 on deals 0..2 gives the means of deals 2..5 alone.
        let wfile = |w: &dyn Fn(u32) -> u64| -> std::path::PathBuf {
            let mut body = Vec::new();
            let mut n = 0;
            for g in 0..one.groups.len() {
                let (sh, i) = one.groups[g][0];
                let h = &one.readers[sh].records[i];
                for d in h.deal_from..h.deal_from + h.n_deals {
                    body.extend_from_slice(&h.pos_sha256);
                    body.extend_from_slice(&h.dseed.to_le_bytes());
                    body.extend_from_slice(&d.to_le_bytes());
                    body.extend_from_slice(&w(d).to_le_bytes());
                    n += 1;
                }
            }
            let mut hd = format!(
                "{{'descr': {}, 'fortran_order': False, 'shape': ({n},), }}",
                student::data::WEIGHTS_DESCR
            );
            let pad = (64 - (11 + hd.len()) % 64) % 64;
            hd.push_str(&" ".repeat(pad));
            hd.push('\n');
            let mut b = b"\x93NUMPY\x01\x00".to_vec();
            b.extend_from_slice(&(hd.len() as u16).to_le_bytes());
            b.extend_from_slice(hd.as_bytes());
            b.extend_from_slice(&body);
            let p = tmp(&format!("w-{}.npy", w(0)));
            std::fs::write(&p, b).expect("written");
            p
        };
        let ones = student::data::WeightsFile::read(&wfile(&|_| 1)).expect("reads");
        let late = student::data::WeightsFile::read(&wfile(&|d| u64::from(d >= 2))).expect("reads");
        for g in 0..6 {
            let m = one.merged(g).expect("merged");
            let plain = feature_row(&m, &*l, &Uniform).expect("encodes");
            assert_eq!(feature_row(&m, &*l, &ones).expect("encodes"), plain);
            let w = feature_row(&m, &*l, &late).expect("encodes");
            let mb = two.merged(g).expect("merged");
            let tail = student::record::means_of(&mb.values[2..], mb.head.cands.len());
            let team = engine::team_of(m.head.pos.seat);
            let rotated: Vec<_> = tail
                .iter()
                .map(|&x| student::encode::rotate(x, team))
                .collect();
            assert_eq!(w.targets, rotated);
        }
        // Overlapping deals are refused.
        assert!(Shards::open(&[&pa, &pall], &[]).is_err());
        // A test shard cannot be read with training shards.
        a.seed = student::STUDENT_TEST_SEED;
        let pt = tmp("m-t.npy");
        write_teacher_shard(&a, &starts, &ctx, teacher, "eq", "bin", "test", &pt, None)
            .expect("written");
        assert!(Shards::open(&[&pa, &pt], &[]).is_err());
        // The check: a random student against the teacher's values.
        let net = student::net::random_net(4, &[16, 8], 12);
        let judged = student::check::judge(&net, &mut one, &*l).expect("judged");
        assert_eq!(judged.len(), 6);
        for (g, d) in judged.iter().enumerate() {
            let m = one.merged(g).expect("merged");
            let means = m.means();
            let t = engine::team_of(m.head.pos.seat);
            let best = means.iter().map(|v| v.get(t)).max().expect("candidates");
            assert_eq!(d.loss, best - means[usize::from(d.chosen)].get(t));
        }
        // The random-card baseline: seeded picks among the candidates (the same twice), each
        // judged against the means; the expected loss is the mean over candidates and decisions.
        let (r, expected) = student::check::judge_random(&mut one, 1).expect("judged");
        assert_eq!(
            student::check::judge_random(&mut one, 1).expect("judged").0,
            r
        );
        let mut want = 0.0;
        for (g, d) in r.iter().enumerate() {
            let m = one.merged(g).expect("merged");
            let means = m.means();
            let t = engine::team_of(m.head.pos.seat);
            let best = means.iter().map(|v| v.get(t)).max().expect("candidates");
            assert!(usize::from(d.chosen) < means.len());
            assert_eq!(d.loss, best - means[usize::from(d.chosen)].get(t));
            want += means
                .iter()
                .map(|v| (best - v.get(t)) as f64 / equity::ONE as f64)
                .sum::<f64>()
                / means.len() as f64;
        }
        assert!((expected - want / r.len() as f64).abs() < 1e-12);
    }
    /// The stored golden set as a student check reads it (a tiny fixture: 16 trick-3 decisions
    /// x 3 exact worlds, cheap at trick 3): each judged decision's values are the golden
    /// record's for the app's candidates, world by world; decisions the student never meets are
    /// skipped; the loss is the exact best mean less the student's choice's; a record of another
    /// decision or method is refused.
    #[test]
    fn golden_decisions_are_judged_on_their_exact_worlds() {
        use crate::measure::{build_decisions, world_rows};
        use crate::pimc::Method;
        let starts = collect_starts(400, 9);
        let l = leaf();
        let source = BeliefSource::new();
        let ctx = belief_ctx(&*l, &source);
        let cfg = teacher_config(8);
        let ds = build_decisions(3, 16, 5, &starts);
        let mut set = Vec::new();
        let mut skipped = 0;
        for d in &ds {
            let w = world_rows(d, Method::ExactPimc, 1, 5, 0, 3, "t", &ctx, &cfg);
            match golden_merged(d, &w).expect("reads") {
                None => skipped += 1,
                Some(m) => {
                    assert_eq!((m.deals.clone(), m.values.len()), (vec![0, 1, 2], 3));
                    for (j, row) in m.values.iter().enumerate() {
                        for (c, v) in row.iter().enumerate() {
                            let i = w
                                .cands
                                .iter()
                                .position(|x| *x == m.head.cands[c])
                                .expect("a golden candidate");
                            assert_eq!(v.0, w.rows[j][i]);
                        }
                    }
                    assert_eq!(m.head.place, d.position);
                    set.push(m);
                }
            }
            let mut bad = w.clone();
            bad.method = "rollout".into();
            assert!(golden_merged(d, &bad).is_err());
            let mut other = w.clone();
            other.id += 1;
            assert!(golden_merged(d, &other).is_err());
        }
        assert!(set.len() >= 8, "{} judged, {skipped} skipped", set.len());
        let net = student::net::random_net(4, &[16, 8], 12);
        let judged = student::check::judge(&net, &mut set, &*l).expect("judged");
        assert_eq!(judged.len(), set.len());
        for (m, d) in set.iter().zip(&judged) {
            let means = m.means();
            let t = engine::team_of(m.head.pos.seat);
            let best = means.iter().map(|v| v.get(t)).max().expect("candidates");
            assert_eq!(d.loss, best - means[usize::from(d.chosen)].get(t));
        }
        let rows = student::check::report(&judged, 50, 1);
        assert!(!rows.is_empty());
    }

    /// `student encode --max-deals N` (docs/STUDENT.md §5): a record of 5 deals cut to its first
    /// 2 gives exactly the merged values, means and feature rows of a record generated with
    /// `--deals 2` for the same seed; cut to all its deals it gives byte-identical feature rows;
    /// more deals than stored, and any cut of a test set, are refused.
    #[test]
    fn records_cut_to_their_first_deals_are_the_smaller_teacher_runs() {
        use student::data::{Shards, Uniform, feature_rows};
        let starts = collect_starts(400, 8);
        let l = leaf();
        let source = BeliefSource::new();
        let ctx = belief_ctx(&*l, &source);
        let o = oracle(3);
        let teacher = Teacher::Hybrid {
            net: &o,
            cut: 3,
            sha256: &o.sha256,
            leaves: None,
        };
        let wide = job(2, 6, 5, SOURCE_MIXED, 2);
        let mut narrow = wide;
        narrow.deals = 2;
        let pw = tmp("cut-wide.npy");
        let pn = tmp("cut-narrow.npy");
        for (j, p) in [(&wide, &pw), (&narrow, &pn)] {
            write_teacher_shard(j, &starts, &ctx, teacher, "eq", "bin", "train", p, None)
                .expect("written");
        }
        let mut w = Shards::open(&[&pw], &[]).expect("opens");
        let mut n = Shards::open(&[&pn], &[]).expect("opens");
        for g in 0..6 {
            let cut = w.merged(g).expect("merged").first_deals(2).expect("cut");
            let small = n.merged(g).expect("merged");
            assert_eq!((&cut.deals, &cut.values), (&small.deals, &small.values));
            assert_eq!(cut.means(), small.means());
            assert!(w.merged(g).expect("merged").first_deals(6).is_err());
            assert!(w.merged(g).expect("merged").first_deals(0).is_err());
        }
        let bytes = |rows: &[student::features::FeatureRow]| {
            let mut b = student::features::header(rows.len() as u64);
            for r in rows {
                student::features::row_bytes(r, &mut b);
            }
            b
        };
        let cut = feature_rows(&mut w, &*l, &Uniform, Some(2)).expect("encodes");
        let small = feature_rows(&mut n, &*l, &Uniform, None).expect("encodes");
        assert_eq!(cut, small);
        assert_eq!(bytes(&cut), bytes(&small));
        let all = feature_rows(&mut w, &*l, &Uniform, None).expect("encodes");
        let full = feature_rows(&mut w, &*l, &Uniform, Some(5)).expect("encodes");
        assert_eq!(bytes(&all), bytes(&full));
        assert_ne!(bytes(&all), bytes(&cut));
        assert!(feature_rows(&mut w, &*l, &Uniform, Some(6)).is_err());
        // A test set is never cut.
        let mut t = wide;
        t.seed = student::STUDENT_TEST_SEED;
        let pt = tmp("cut-test.npy");
        write_teacher_shard(&t, &starts, &ctx, teacher, "eq", "bin", "test", &pt, None)
            .expect("written");
        let mut ts = Shards::open(&[&pt], &[]).expect("opens");
        assert!(feature_rows(&mut ts, &*l, &Uniform, Some(2)).is_err());
        assert!(feature_rows(&mut ts, &*l, &Uniform, Some(5)).is_err());
        assert_eq!(
            feature_rows(&mut ts, &*l, &Uniform, None)
                .expect("encodes")
                .len(),
            6
        );
    }
}
