//! Differential test of the engine against the independent reference (docs/M0-SPEC.md §6).
//!
//! For each match the engine is created from a seed; the reference is created with the engine's
//! first dealer and is handed each of the engine's decks as it needs them. The two are compared
//! through a canonical [`Snapshot`]:
//!
//! - after every action, the whole [`State`]: phase, seat to act, legal actions, every field of
//!   the public state, scores, games, bonus flags, money, every seat's hand and burns, and the
//!   undealt pack;
//! - in addition, after every deal (each hand boundary, including the first) and at the end of
//!   the match, the full snapshot: the state built from the engine's eight seat views, and every
//!   seat's complete canonical event list.
//!
//! Event lists are compared in full at those points only because the reference can give them
//! only as a fresh copy of its whole history (`events_for`), which made the every-action check
//! quadratic in the match length; both logs only grow, so every event is still compared. When
//! anything differs, the match is rerun comparing the full snapshot after every action, so the
//! report names the first action at which the two differ.
//!
//! The action is the k-th legal action, with `k` drawn by [`action_rng`]. Each finished engine
//! match must then replay exactly from its seed and actions. The first difference stops the run
//! with a [`Disagreement`].
//!
//! Seeds: match `i` of a run with seed `s` uses the engine seed [`match_seed`]`(s, i)` and the
//! action generator [`action_rng`]`(s, i)`, so any match can be rerun alone with [`run_match`],
//! and a run split over threads gives exactly the same report as a single thread.
//!
//! **Turned partnerships and the app's referee** ([`run_turns`], docs/APP-ENGINE-CONTRACT.md §7):
//! the same, and now and then (before a play, one chance in `turn_every`; while a rob is due, one
//! in three; drawn from [`turn_rng`]) both implementations are asked to do the same random step
//! the app can ask for: turn a random partnership, play a random legal card then turn one, let a
//! random card through (an illegal card, a play without robbing, a rob without the ace, a burned
//! ace). Both must accept, or both refuse with the same reason code, and the full snapshot (the
//! hidden reneges included) is compared after every step. Several steps can follow one another
//! at the same moment. A legal card may then be refused by both (a renege it proves would turn
//! the partnership that led the trick: the open question); another is tried, and when every one
//! is refused the app's catch of the seat's partnership or an illegal card is tried, and failing
//! those the match stops there (counted as stuck). The match record (actions and extras) must
//! replay exactly.

pub mod report;
pub mod snapshot;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};

use engine::cards::Card;
use engine::game::{
    Action as EAction, Event, Extra, Lapse, Match as EMatch, Phase as EPhase, Rules, TurnRefusal,
    Visibility,
};
use engine::rng::Rng;

pub use report::{COVERAGE_MIN_MATCHES, Line, Per, Report};
pub use snapshot::{SnapAction, SnapPhase, Snapshot, State};

/// A match that has not ended after this many actions is reported (a real one needs under 1,500).
const MAX_ACTIONS: u64 = 20_000;

/// The engine seed of match `match_index` in a run with seed `seed`: the first output of
/// `Rng::derive(seed, &[match_index, 1])`.
pub fn match_seed(seed: u64, match_index: u64) -> u64 {
    Rng::derive(seed, &[match_index, 1]).next_u64()
}

/// The generator that picks the actions of match `match_index`: `Rng::derive(seed, &[match_index])`.
pub fn action_rng(seed: u64, match_index: u64) -> Rng {
    Rng::derive(seed, &[match_index])
}

/// The generator that decides the turns of match `match_index` in a run with turns:
/// `Rng::derive(seed, &[match_index, 2])`.
pub fn turn_rng(seed: u64, match_index: u64) -> Rng {
    Rng::derive(seed, &[match_index, 2])
}

/// The default chance of a turn attempt before a play in [`run_turns`]: one in this many.
pub const TURN_EVERY: u64 = 12;

/// The engine's rule switches in the reference's own type.
pub fn reference_rules(r: Rules) -> reference::Rules {
    reference::Rules { stake: r.stake }
}

fn reference_action(a: EAction) -> reference::Action {
    match a {
        EAction::Burn(c) => reference::Action::Burn(c),
        EAction::Play(c) => reference::Action::Play(c),
    }
}

fn reference_lapse(l: Lapse) -> reference::Lapse {
    match l {
        Lapse::Play(c) => reference::Lapse::Play(c),
        Lapse::PlayWithoutRob(c) => reference::Lapse::PlayWithoutRob(c),
        Lapse::FalseRob(c) => reference::Lapse::FalseRob(c),
        Lapse::BurnAce => reference::Lapse::BurnAce,
    }
}

/// Asks both implementations for the same step: `Ok(true)` when both applied it, `Ok(false)`
/// when both refused it with the same code (counted in `tally`), else the two answers.
fn both_extra(
    step: Extra,
    engine: &mut EMatch,
    reference: &mut reference::Match,
    tally: &mut TurnTally,
) -> Result<bool, String> {
    let (e, r): (Result<(), &'static str>, Result<(), String>) = match step {
        Extra::Turn(t) => (
            engine.turn_team(t).map_err(|x| x.why.code()),
            reference.turn_team(t),
        ),
        Extra::PlayThenTurn(c, t) => (
            engine.play_then_turn(c, t).map_err(|x| x.why.code()),
            reference.play_then_turn(c, t),
        ),
        Extra::Lapse(l) => (
            engine.let_through(l).map_err(|x| x.why.code()),
            reference.let_through(reference_lapse(l)),
        ),
    };
    let lapse = matches!(step, Extra::Lapse(_));
    if lapse {
        tally.lapse_attempts += 1;
    } else {
        tally.attempts += 1;
    }
    match (&e, &r) {
        (Ok(()), Ok(())) => {
            if let Extra::Lapse(l) = step {
                tally.lapses[report::lapse_index(l)] += 1;
            }
            Ok(true)
        }
        (Err(code), Err(text)) if text.split(':').next() == Some(code) => {
            let (codes, counts): (&[&str], &mut [u64]) = if lapse {
                (&report::LAPSE_REFUSALS, &mut tally.lapse_refused)
            } else {
                (&report::TURN_REFUSALS, &mut tally.refused)
            };
            if let Some(i) = codes.iter().position(|c| c == code) {
                counts[i] += 1;
            }
            Ok(false)
        }
        _ => Err(format!(
            "{step:?}: the engine says {e:?}, the reference says {r:?}"
        )),
    }
}

/// A random step the app can ask for now (see the module notes).
fn random_extra(turns: &mut Rng, engine: &EMatch) -> Option<Extra> {
    let seat = engine.to_act()?;
    let hand: Vec<Card> = engine::cards::cards(engine.truth().hands[usize::from(seat)]).collect();
    let card = |turns: &mut Rng| hand[turns.bounded(hand.len() as u64) as usize];
    let team = |turns: &mut Rng| turns.bounded(4) as u8;
    match engine.phase() {
        EPhase::AwaitBurn(_) => Some(match turns.bounded(4) {
            0 => Extra::Turn(engine::team_of(seat)),
            1 => Extra::Turn(team(turns)),
            2 => Extra::Lapse(Lapse::PlayWithoutRob(card(turns))),
            _ => Extra::Lapse(Lapse::BurnAce),
        }),
        EPhase::AwaitPlay(_) => Some(match turns.bounded(10) {
            0..=3 => Extra::Turn(team(turns)),
            4 | 5 => {
                let legal = engine.legal_actions();
                let c = legal[turns.bounded(legal.len() as u64) as usize].card();
                Extra::PlayThenTurn(c, team(turns))
            }
            6..=8 => Extra::Lapse(Lapse::Play(card(turns))),
            // A rob without the ace burns a card it holds or the card it takes (H3).
            _ => match engine.public().table_card {
                Some(t) if turns.bounded(4) == 0 => Extra::Lapse(Lapse::FalseRob(t)),
                _ => Extra::Lapse(Lapse::FalseRob(card(turns))),
            },
        }),
        EPhase::MatchOver => None,
    }
}

/// The first difference found, with everything needed to reproduce and read it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Disagreement {
    /// The run's seed.
    pub seed: u64,
    /// The match within the run (rerun it alone with [`run_match`]).
    pub match_index: u64,
    /// The engine seed of that match ([`match_seed`]).
    pub match_seed: u64,
    /// The engine's hand index when the difference was found.
    pub hand_index: u32,
    /// Actions applied in this match so far (replaying that many from the seed rebuilds the
    /// engine's side).
    pub action_index: u64,
    /// The last action applied, if any.
    pub last_action: Option<EAction>,
    /// What went wrong, then one line per differing field.
    pub reason: String,
    /// The engine's snapshot, as text.
    pub engine: String,
    /// The reference's snapshot, as text (for a replay failure: the engine's replay).
    pub reference: String,
}

impl core::fmt::Display for Disagreement {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let last = match self.last_action {
            Some(a) => snapshot::action_text(a.into()),
            None => "none (at the first deal)".into(),
        };
        writeln!(f, "DISAGREEMENT")?;
        writeln!(f, "  run seed      {}", self.seed)?;
        writeln!(
            f,
            "  match index   {} (engine match seed {})",
            self.match_index, self.match_seed
        )?;
        writeln!(f, "  hand index    {}", self.hand_index)?;
        writeln!(f, "  action index  {} actions applied", self.action_index)?;
        writeln!(f, "  last action   {last}")?;
        writeln!(
            f,
            "  rerun alone   difftest::run_match({}, {}, rules)",
            self.seed, self.match_index
        )?;
        writeln!(f, "{}", self.reason)?;
        writeln!(f, "--- engine snapshot ---")?;
        write!(f, "{}", self.engine)?;
        writeln!(f, "--- reference snapshot ---")?;
        write!(f, "{}", self.reference)
    }
}

impl std::error::Error for Disagreement {}

/// Where the driver is in a match (kept outside the match so a panic can be located).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Deal,
    Compare,
    Turn,
    Apply,
    Replay,
}

#[derive(Clone, Copy, Debug)]
struct Progress {
    hand_index: u32,
    applied: u64,
    stage: Stage,
    last_action: Option<EAction>,
}

impl Progress {
    fn new() -> Self {
        Progress {
            hand_index: 0,
            applied: 0,
            stage: Stage::Deal,
            last_action: None,
        }
    }
}

/// A test hook that edits the reference's side (given the action count) before comparison: its
/// state, and its event lists when those are being compared.
type Tamper<'a> = &'a dyn Fn(u64, &mut State, Option<&mut [Vec<String>; snapshot::SEATS]>);

/// How much is compared after each action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Check {
    /// The state after every action; the full snapshot after every deal and at the match end.
    Boundaries,
    /// The full snapshot after every action (used to locate a difference).
    EveryAction,
}

/// One match to drive.
#[derive(Clone, Copy)]
struct Ctx<'a> {
    seed: u64,
    match_index: u64,
    match_seed: u64,
    rules: Rules,
    /// Builds the engine match afresh (the same match every time).
    make: &'a dyn Fn() -> EMatch,
    /// Built by `Match::new(match_seed, rules)`, so `Match::replay` applies.
    seeded: bool,
    tamper: Option<Tamper<'a>>,
    check: Check,
    /// Stop just before this stage at this action count (to show the state before a panic).
    stop: Option<(u64, Stage)>,
    /// Turn partnerships at random, one chance in this many before each play (`None`: never).
    turn_every: Option<u64>,
}

/// The app's steps tried in one match: turn attempts and refusals by code
/// ([`report::TURN_REFUSALS`]), lapses tried, applied by kind and refused by code
/// ([`report::LAPSE_REFUSALS`]), legal cards both refused, and whether the match got stuck.
#[derive(Clone, Copy, Debug, Default)]
struct TurnTally {
    attempts: u64,
    refused: [u64; 5],
    lapse_attempts: u64,
    lapses: [u64; 4],
    lapse_refused: [u64; 4],
    refused_plays: u64,
    stuck: u64,
}

enum Outcome {
    Finished(Box<EMatch>, TurnTally),
    Stopped { engine: String, reference: String },
}

impl Ctx<'_> {
    fn disagreement(&self, pr: &Progress, reason: String, e: String, r: String) -> Disagreement {
        Disagreement {
            seed: self.seed,
            match_index: self.match_index,
            match_seed: self.match_seed,
            hand_index: pr.hand_index,
            action_index: pr.applied,
            last_action: pr.last_action,
            reason,
            engine: e,
            reference: r,
        }
    }

    fn stops(&self, pr: &Progress, stage: Stage) -> bool {
        self.stop == Some((pr.applied, stage))
    }
}

/// The engine's event log with each event's canonical string, extended incrementally: after
/// `update(log)`, `entries == log` and `strings[i] == log[i].1.canonical()` for every `i`,
/// exactly as if rebuilt from scratch. One string per event, shared by all seats.
#[derive(Default)]
struct EngineLog {
    entries: Vec<(Visibility, Event)>,
    strings: Vec<String>,
}

impl EngineLog {
    fn update(&mut self, now: &[(Visibility, Event)]) {
        let keep = self
            .entries
            .iter()
            .zip(now)
            .take_while(|(a, b)| a == b)
            .count();
        self.entries.truncate(keep);
        self.strings.truncate(keep);
        for entry in &now[keep..] {
            self.strings.push(entry.1.canonical());
            self.entries.push(entry.clone());
        }
    }

    /// Exactly `view_events.map(canonical) == reference_events`, where `view_events` is seat
    /// `seat`'s view of the engine, using the cached strings when the view lists the same events
    /// as the log's entries visible to that seat (it always should).
    fn seat_matches(&self, seat: u8, view_events: &[Event], reference_events: &[String]) -> bool {
        let mut i = 0;
        for ((vis, event), text) in self.entries.iter().zip(&self.strings) {
            if !vis.visible_to(seat) {
                continue;
            }
            if view_events.get(i) != Some(event) {
                return canonical_matches(view_events, reference_events);
            }
            if reference_events.get(i) != Some(text) {
                return false;
            }
            i += 1;
        }
        if i != view_events.len() {
            return canonical_matches(view_events, reference_events);
        }
        i == reference_events.len()
    }
}

fn canonical_matches(events: &[Event], texts: &[String]) -> bool {
    events.len() == texts.len() && events.iter().zip(texts).all(|(e, t)| e.canonical() == *t)
}

fn render_guarded(f: impl FnOnce() -> String) -> String {
    catch_unwind(AssertUnwindSafe(f))
        .unwrap_or_else(|_| "  (unavailable: building this snapshot panicked)\n".into())
}

fn stopped(engine: &EMatch, reference: &reference::Match) -> Outcome {
    Outcome::Stopped {
        engine: render_guarded(|| Snapshot::from_engine(engine).render()),
        reference: render_guarded(|| Snapshot::from_reference(reference).render()),
    }
}

fn differences_text(head: &str, lines: &[String]) -> String {
    let mut s = head.to_string();
    for l in lines {
        s.push_str("\n  ");
        s.push_str(l);
    }
    s
}

/// Is the list strictly ascending by card index?
fn ascending(cards: impl Iterator<Item = Card>) -> bool {
    let v: Vec<Card> = cards.collect();
    v.windows(2).all(|w| w[0] < w[1])
}

/// Compares the two implementations now: the state only, or (`full`) the whole snapshot. The
/// full comparison is exactly `Snapshot::from_engine(engine) == Snapshot::from_reference(..)`,
/// with the engine's canonical event strings cached in `log`.
fn compare(
    ctx: &Ctx,
    pr: &Progress,
    engine: &EMatch,
    reference: &reference::Match,
    log: &mut EngineLog,
    full: bool,
) -> Result<(), Disagreement> {
    let same = if full {
        let views = snapshot::engine_views(engine);
        let e_state = State::from_engine(engine, &views);
        let mut r_snap = Snapshot::from_reference(reference);
        if let Some(tamper) = ctx.tamper {
            tamper(pr.applied, &mut r_snap.state, Some(&mut r_snap.events));
        }
        log.update(engine.events());
        e_state == r_snap.state
            && (0..snapshot::SEATS)
                .all(|s| log.seat_matches(s as u8, &views[s].events, &r_snap.events[s]))
    } else {
        let e_state = State::from_engine_truth(engine);
        let mut r_state = State::from_reference(reference);
        if let Some(tamper) = ctx.tamper {
            tamper(pr.applied, &mut r_state, None);
        }
        e_state == r_state
    };
    if same {
        return Ok(());
    }
    let e_snap = Snapshot::from_engine(engine);
    let mut r_snap = Snapshot::from_reference(reference);
    if let Some(tamper) = ctx.tamper {
        tamper(pr.applied, &mut r_snap.state, Some(&mut r_snap.events));
    }
    let diffs = e_snap.differences(&r_snap, "engine", "reference");
    Err(ctx.disagreement(
        pr,
        differences_text("the engine and the reference differ:", &diffs),
        e_snap.render(),
        r_snap.render(),
    ))
}

/// Drives one match to the end (or to `ctx.stop`).
fn drive(ctx: &Ctx, pr: &mut Progress) -> Result<Outcome, Disagreement> {
    let mut engine = (ctx.make)();
    let mut reference = reference::Match::new(engine.dealer(), reference_rules(ctx.rules));
    let mut rng = action_rng(ctx.seed, ctx.match_index);
    let mut turns = turn_rng(ctx.seed, ctx.match_index);
    let mut tally = TurnTally::default();
    // A further attempt at the same moment is drawn after each one (two or three partnerships
    // turned before one play).
    let mut again = false;
    let mut log = EngineLog::default();
    let fail = |pr: &Progress, engine: &EMatch, reference: &reference::Match, why: String| {
        ctx.disagreement(
            pr,
            why,
            Snapshot::from_engine(engine).render(),
            Snapshot::from_reference(reference).render(),
        )
    };
    loop {
        pr.hand_index = engine.hand_index();
        if ctx.stops(pr, Stage::Deal) {
            return Ok(stopped(&engine, &reference));
        }
        pr.stage = Stage::Deal;
        let dealt = reference.needs_deck();
        if dealt {
            reference.start_hand(&engine.truth().deck);
        }
        if ctx.stops(pr, Stage::Compare) {
            return Ok(stopped(&engine, &reference));
        }
        pr.stage = Stage::Compare;
        let over = engine.phase() == EPhase::MatchOver;
        let full = ctx.check == Check::EveryAction || dealt || over;
        compare(ctx, pr, &engine, &reference, &mut log, full)?;
        if over {
            break;
        }
        if let Some(every) = ctx.turn_every {
            let burn = matches!(engine.phase(), EPhase::AwaitBurn(_));
            if (again || turns.bounded(if burn { 3 } else { every }) == 0)
                && let Some(step) = random_extra(&mut turns, &engine)
            {
                again = turns.bounded(3) == 0;
                if ctx.stops(pr, Stage::Turn) {
                    return Ok(stopped(&engine, &reference));
                }
                pr.stage = Stage::Turn;
                let applied = both_extra(step, &mut engine, &mut reference, &mut tally)
                    .map_err(|why| fail(pr, &engine, &reference, why))?;
                if applied {
                    // The full snapshot after every step (the hand may have ended: deal first).
                    if reference.needs_deck() {
                        reference.start_hand(&engine.truth().deck);
                    }
                    compare(ctx, pr, &engine, &reference, &mut log, true)?;
                }
                continue;
            }
        }
        again = false;
        let legal = engine.legal_actions();
        if legal.is_empty() {
            let why = "no legal action although the match is not over".to_string();
            return Err(fail(pr, &engine, &reference, why));
        }
        if !ascending(legal.iter().map(|a| a.card())) {
            let why = "the engine's legal_actions() is not sorted by card index".to_string();
            return Err(fail(pr, &engine, &reference, why));
        }
        if !ascending(reference.legal_actions().iter().map(|a| a.card())) {
            let why = "the reference's legal_actions() is not sorted by card index".to_string();
            return Err(fail(pr, &engine, &reference, why));
        }
        let first = rng.bounded(legal.len() as u64) as usize;
        if ctx.stops(pr, Stage::Apply) {
            return Ok(stopped(&engine, &reference));
        }
        pr.stage = Stage::Apply;
        let mut played = false;
        for k in 0..legal.len() {
            let action = legal[(first + k) % legal.len()];
            pr.last_action = Some(action);
            match engine.apply(action) {
                Ok(()) => {}
                Err(e) if e.reason == TurnRefusal::LedTrickInProgress.reason() => {
                    // A renege let through would be proved by this card and turn the
                    // partnership that led the trick (the open question): both refuse it.
                    match reference.apply(reference_action(action)) {
                        Err(r) if r.contains("led_trick_in_progress") => {
                            tally.refused_plays += 1;
                            continue;
                        }
                        r => {
                            let why = format!(
                                "the engine refused a legal card as the open question, the \
                                 reference says {r:?}"
                            );
                            return Err(fail(pr, &engine, &reference, why));
                        }
                    }
                }
                Err(e) => {
                    let why = format!("the engine refused its own legal action: {e}");
                    return Err(fail(pr, &engine, &reference, why));
                }
            }
            if let Err(e) = reference.apply(reference_action(action)) {
                let why = format!(
                    "the reference refused an action legal in both (the engine has applied \
                     it): {e}"
                );
                return Err(fail(pr, &engine, &reference, why));
            }
            played = true;
            break;
        }
        if !played {
            // Every legal card is refused: the app catches the seat's partnership before its
            // card, or the seat plays an illegal card; failing those the engine stops here.
            let seat = engine.to_act().expect("a play is awaited");
            let mut steps = vec![Extra::Turn(engine::team_of(seat))];
            steps.extend(
                engine::cards::cards(engine.truth().hands[usize::from(seat)])
                    .map(|c| Extra::Lapse(Lapse::Play(c))),
            );
            let mut moved = false;
            for step in steps {
                if both_extra(step, &mut engine, &mut reference, &mut tally)
                    .map_err(|why| fail(pr, &engine, &reference, why))?
                {
                    moved = true;
                    break;
                }
            }
            if moved {
                if reference.needs_deck() {
                    reference.start_hand(&engine.truth().deck);
                }
                compare(ctx, pr, &engine, &reference, &mut log, true)?;
                continue;
            }
            tally.stuck = 1;
            compare(ctx, pr, &engine, &reference, &mut log, true)?;
            break;
        }
        pr.applied += 1;
        if pr.applied > MAX_ACTIONS {
            let why = format!("the match has not ended after {MAX_ACTIONS} actions");
            return Err(fail(pr, &engine, &reference, why));
        }
    }
    if ctx.stops(pr, Stage::Replay) {
        return Ok(stopped(&engine, &reference));
    }
    pr.stage = Stage::Replay;
    check_replay(ctx, pr, &engine)?;
    Ok(Outcome::Finished(Box::new(engine), tally))
}

/// The finished engine match must rebuild exactly from its start and its record (its actions,
/// and its turns when partnerships were turned).
fn check_replay(ctx: &Ctx, pr: &Progress, engine: &EMatch) -> Result<(), Disagreement> {
    let rebuilt = if ctx.seeded && engine.extras().is_empty() {
        EMatch::replay(ctx.match_seed, ctx.rules, engine.actions()).map_err(|e| e.to_string())
    } else if ctx.seeded {
        EMatch::replay_record(ctx.match_seed, ctx.rules, engine.actions(), engine.extras())
            .map_err(|e| e.to_string())
    } else {
        let mut m = (ctx.make)();
        m.apply_record(engine.actions(), engine.extras())
            .map(|()| m)
            .map_err(|e| e.to_string())
    };
    let original = || Snapshot::from_engine(engine).render();
    match rebuilt {
        Err(e) => Err(ctx.disagreement(
            pr,
            format!("the engine's replay from seed and record refused an action or turn: {e}"),
            original(),
            "  (no replay: it stopped at the refused action)\n".into(),
        )),
        Ok(r) if r.events() != engine.events() || r != *engine => {
            let (a, b) = (Snapshot::from_engine(engine), Snapshot::from_engine(&r));
            let mut diffs = a.differences(&b, "played", "replayed");
            if r.events() != engine.events() {
                diffs.push(format!(
                    "event logs differ: played {} events, replayed {}",
                    engine.events().len(),
                    r.events().len()
                ));
            }
            Err(ctx.disagreement(
                pr,
                differences_text(
                    "the engine's replay from seed and actions differs (the second snapshot is \
                     the replay):",
                    &diffs,
                ),
                a.render(),
                b.render(),
            ))
        }
        Ok(_) => Ok(()),
    }
}

fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "(no message)".to_string()
    }
}

/// Drives one match and returns it finished, or the disagreement; a panic in either
/// implementation becomes a [`Disagreement`] that shows both states just before it.
fn attempt(ctx: &Ctx) -> Result<(Box<EMatch>, TurnTally), Disagreement> {
    let mut pr = Progress::new();
    match catch_unwind(AssertUnwindSafe(|| drive(ctx, &mut pr))) {
        Ok(Ok(Outcome::Finished(m, t))) => Ok((m, t)),
        Ok(Ok(Outcome::Stopped { engine, reference })) => Err(ctx.disagreement(
            &pr,
            "internal: the driver stopped early".into(),
            engine,
            reference,
        )),
        Ok(Err(d)) => Err(d),
        Err(payload) => {
            let msg = panic_text(payload.as_ref());
            // Rerun the same match the same way and stop just before the stage that panicked.
            let again = Ctx {
                stop: Some((pr.applied, pr.stage)),
                ..*ctx
            };
            let mut pr2 = Progress::new();
            let (e, r) = match catch_unwind(AssertUnwindSafe(|| drive(&again, &mut pr2))) {
                Ok(Ok(Outcome::Stopped { engine, reference })) => (engine, reference),
                _ => {
                    let na = "  (unavailable: the rerun did not stop cleanly)\n".to_string();
                    (na.clone(), na)
                }
            };
            let stage = match pr.stage {
                Stage::Deal => "dealing the reference's hand",
                Stage::Compare => "building or comparing the snapshots",
                Stage::Turn => "turning a partnership",
                Stage::Apply => "applying the last action",
                Stage::Replay => "replaying the engine match",
            };
            let why = format!(
                "a panic while {stage}: {msg}\n  (the snapshots show the state just before it)"
            );
            Err(ctx.disagreement(&pr, why, e, r))
        }
    }
}

/// Drives one match and counts it into `report` when it agrees. On a difference found with the
/// state-per-action check, the match is rerun comparing the full snapshot after every action,
/// and the earliest difference that rerun finds is reported.
fn run_ctx(ctx: &Ctx, report: &mut Report) -> Result<(), Disagreement> {
    match attempt(ctx) {
        Ok((m, t)) => {
            report.count_match(&m);
            report.turn_attempts += t.attempts;
            for (a, b) in report.turn_refusals.iter_mut().zip(&t.refused) {
                *a += b;
            }
            report.lapse_attempts += t.lapse_attempts;
            for (a, b) in report.lapses.iter_mut().zip(&t.lapses) {
                *a += b;
            }
            for (a, b) in report.lapse_refusals.iter_mut().zip(&t.lapse_refused) {
                *a += b;
            }
            report.refused_plays += t.refused_plays;
            report.stuck_matches += t.stuck;
            Ok(())
        }
        Err(first) if ctx.check == Check::Boundaries => {
            let every = Ctx {
                check: Check::EveryAction,
                ..*ctx
            };
            match attempt(&every) {
                Err(mut located) => {
                    located.reason.push_str(
                        "\n  (located by rerunning the match with the full snapshot compared \
                         after every action)",
                    );
                    Err(located)
                }
                Ok(_) => {
                    let mut d = first;
                    d.reason.push_str(
                        "\n  (a rerun comparing the full snapshot after every action did not \
                         reproduce this)",
                    );
                    Err(d)
                }
            }
        }
        Err(d) => Err(d),
    }
}

/// Plays match `match_index` of a run with seed `seed` alone, and returns its counts.
pub fn run_match(seed: u64, match_index: u64, rules: Rules) -> Result<Report, Disagreement> {
    let mut report = Report::default();
    run_seeded(seed, match_index, rules, None, None, &mut report)?;
    Ok(report)
}

/// Plays match `match_index` of a run with turns ([`run_turns`]) alone, and returns its counts.
pub fn run_match_turns(
    seed: u64,
    match_index: u64,
    rules: Rules,
    turn_every: u64,
) -> Result<Report, Disagreement> {
    let mut report = Report::default();
    run_seeded(
        seed,
        match_index,
        rules,
        None,
        Some(turn_every.max(1)),
        &mut report,
    )?;
    Ok(report)
}

fn run_seeded(
    seed: u64,
    match_index: u64,
    rules: Rules,
    tamper: Option<Tamper<'_>>,
    turn_every: Option<u64>,
    report: &mut Report,
) -> Result<(), Disagreement> {
    let ms = match_seed(seed, match_index);
    let make = move || EMatch::new(ms, rules);
    let ctx = Ctx {
        turn_every,
        ..seeded_ctx(seed, match_index, rules, &make, tamper)
    };
    run_ctx(&ctx, report)
}

fn seeded_ctx<'a>(
    seed: u64,
    match_index: u64,
    rules: Rules,
    make: &'a dyn Fn() -> EMatch,
    tamper: Option<Tamper<'a>>,
) -> Ctx<'a> {
    Ctx {
        seed,
        match_index,
        match_seed: match_seed(seed, match_index),
        rules,
        make,
        seeded: true,
        tamper,
        check: Check::Boundaries,
        stop: None,
        turn_every: None,
    }
}

/// Plays one match whose first hands use the given decks (`Match::with_decks`; later hands are
/// shuffled from `seed`), with actions from `action_rng(seed, 0)`, comparing as [`run`] does.
/// For hand-made deals and deals built to force a rule branch.
pub fn run_decks(
    first_dealer: u8,
    decks: Vec<[Card; 53]>,
    seed: u64,
    rules: Rules,
) -> Result<Report, Disagreement> {
    let make = move || EMatch::with_decks(first_dealer, decks.clone(), seed, rules);
    let ctx = Ctx {
        seed,
        match_index: 0,
        match_seed: seed,
        rules,
        make: &make,
        seeded: false,
        tamper: None,
        check: Check::Boundaries,
        stop: None,
        turn_every: None,
    };
    let mut report = Report::default();
    run_ctx(&ctx, &mut report)?;
    Ok(report)
}

fn run_range(
    seed: u64,
    start: u64,
    end: u64,
    rules: Rules,
    turn_every: Option<u64>,
    first_bad: Option<&AtomicU64>,
) -> Result<Report, Disagreement> {
    let mut report = Report::default();
    for i in start..end {
        if first_bad.is_some_and(|f| f.load(Ordering::Relaxed) < i) {
            break; // an earlier match has already failed; it is the one reported
        }
        if let Err(d) = run_seeded(seed, i, rules, None, turn_every, &mut report) {
            if let Some(f) = first_bad {
                f.fetch_min(i, Ordering::Relaxed);
            }
            return Err(d);
        }
    }
    Ok(report)
}

/// Plays matches `0..matches` of the run with seed `seed` on one thread.
pub fn run(seed: u64, matches: u64, rules: Rules) -> Result<Report, Disagreement> {
    run_range(seed, 0, matches, rules, None, None)
}

/// Plays matches `0..matches` of the run with seed `seed` on one thread, turning partnerships at
/// random (one chance in `turn_every` before each play; see the module notes).
pub fn run_turns(
    seed: u64,
    matches: u64,
    rules: Rules,
    turn_every: u64,
) -> Result<Report, Disagreement> {
    run_range(seed, 0, matches, rules, Some(turn_every.max(1)), None)
}

/// As [`run`], split over `threads` threads (each takes a contiguous block of matches). The
/// report is the same as [`run`]'s; on failure, the disagreement of the lowest match index.
pub fn run_threads(
    seed: u64,
    matches: u64,
    rules: Rules,
    threads: usize,
) -> Result<Report, Disagreement> {
    run_threads_with(seed, matches, rules, threads, None)
}

/// As [`run_turns`], split over `threads` threads (the same report as one thread).
pub fn run_turns_threads(
    seed: u64,
    matches: u64,
    rules: Rules,
    turn_every: u64,
    threads: usize,
) -> Result<Report, Disagreement> {
    run_threads_with(seed, matches, rules, threads, Some(turn_every.max(1)))
}

fn run_threads_with(
    seed: u64,
    matches: u64,
    rules: Rules,
    threads: usize,
    turn_every: Option<u64>,
) -> Result<Report, Disagreement> {
    let n = (threads.max(1) as u64).min(matches.max(1));
    let first_bad = AtomicU64::new(u64::MAX);
    let bound = |t: u64| (u128::from(matches) * u128::from(t) / u128::from(n)) as u64;
    let results: Vec<Result<Report, Disagreement>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n)
            .map(|t| {
                let (a, b, fb) = (bound(t), bound(t + 1), &first_bad);
                scope.spawn(move || run_range(seed, a, b, rules, turn_every, Some(fb)))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
            .collect()
    });
    let mut total = Report::default();
    let mut first: Option<Disagreement> = None;
    for r in results {
        match r {
            Ok(rep) => total.add(&rep),
            Err(d) => {
                if first.as_ref().is_none_or(|f| d.match_index < f.match_index) {
                    first = Some(d);
                }
            }
        }
    }
    match first {
        Some(d) => Err(d),
        None => Ok(total),
    }
}

#[cfg(test)]
mod tests;
