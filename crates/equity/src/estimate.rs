//! Estimating the hand-outcome model from self-play (docs/M1-SPEC.md §6.3).
//!
//! The estimator drives `engine::game::Match` directly: it builds eight bots from one factory,
//! starts each with the seat-seed rule of docs/M1-SPEC.md §1, and loops `view(to_act)` →
//! `legal_actions()` → the bot's choice → `apply`. From the public events it records, for every
//! hand, the trick winners as team offsets relative to the dealer's team, the trick holding the
//! 5 of trumps (if played) and the offset holding the best trump played. Matches are played in
//! index order and hands are taken in order until the requested number is reached, so the result
//! is identical for any thread count.

use core::fmt;

use bot::{Bot, BotFactory};
use engine::cards::{JOKER, Suit, suit_of};
use engine::game::{Action, Event, GameEndReason, Match, RULES_VERSION, Rules};
use engine::rng::Rng;
use engine::rules::{five_of, trump_rank};
use engine::team_of;

use crate::model::{HandModel, ModelSource};

/// Label of the match-seed derivation: match `i` of an estimate with seed `s` has seed
/// `Rng::derive(s, &[0xE57, i]).next_u64()`.
pub const MATCH_SEED_LABEL: u64 = 0xE57;

/// Label of the seat-seed derivation (docs/M1-SPEC.md §1): the bot at seat `s` of a match with
/// seed `m` starts with `Rng::derive(m, &[0xB07, s]).next_u64()`.
pub const SEAT_SEED_LABEL: u64 = 0xB07;

/// The seed of match `match_index` of an estimate.
pub fn match_seed(seed: u64, match_index: u64) -> u64 {
    Rng::derive(seed, &[MATCH_SEED_LABEL, match_index]).next_u64()
}

/// The `start_match` seed of the bot at `seat` in a match (docs/M1-SPEC.md §1).
pub fn seat_seed(match_seed: u64, seat: u8) -> u64 {
    Rng::derive(match_seed, &[SEAT_SEED_LABEL, u64::from(seat)]).next_u64()
}

/// How a recorded hand ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandOutcome {
    /// A team won the game during or at the end of this hand.
    GameWon {
        /// The winning team (real team number).
        team: u8,
        /// The engine's reason.
        reason: GameEndReason,
    },
    /// The hand ended normally with these scores (real team order).
    HandOver {
        /// Scores after the hand.
        scores: [u8; 4],
    },
}

/// One hand as the estimator saw it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HandRecord {
    /// The match, in estimate order.
    pub match_index: u64,
    /// The hand's index in its match.
    pub hand_index: u32,
    /// The dealer's team.
    pub dealer_team: u8,
    /// Scores when the hand was dealt (real team order).
    pub start_scores: [u8; 4],
    /// The engine's bonus flags when the hand was dealt.
    pub start_flags: [bool; 4],
    /// Games won when the hand was dealt.
    pub games: [u8; 4],
    /// The winner of each trick played, as an offset from the dealer's team.
    pub winners: Vec<u8>,
    /// The trick (1..=5) that held the 5 of trumps, if it was played.
    pub five_trick: Option<u8>,
    /// The offset of the team that played the best trump of the hand, if any trump was played.
    pub best_trump: Option<u8>,
    /// How the hand ended.
    pub outcome: HandOutcome,
}

/// Why an estimate stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EstimateError {
    /// A bot chose an action that was not legal (a bot bug); it was never applied.
    IllegalAction {
        /// The match.
        match_index: u64,
        /// The seat.
        seat: u8,
        /// The bot's name.
        bot: String,
        /// The refused action.
        action: Action,
        /// What was legal.
        legal: Vec<Action>,
    },
}

impl fmt::Display for EstimateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EstimateError::IllegalAction {
                match_index,
                seat,
                bot,
                action,
                legal,
            } => write!(
                f,
                "match {match_index}: bot {bot} at seat {seat} chose {action:?}, legal were {legal:?}"
            ),
        }
    }
}

impl std::error::Error for EstimateError {}

struct Current {
    hand_index: u32,
    dealer_team: u8,
    start_scores: [u8; 4],
    start_flags: [bool; 4],
    games: [u8; 4],
    trump: Option<Suit>,
    winners: Vec<u8>,
    five_trick: Option<u8>,
    best: Option<(u8, u8)>,
    outcome: Option<HandOutcome>,
}

struct Tracker {
    match_index: u64,
    current: Option<Current>,
    done: Vec<HandRecord>,
}

impl Tracker {
    /// Read the events from `from` on. Called right after each `apply`, so when a hand ends or a
    /// new hand starts the match's scores are those between the two hands (nothing is scored
    /// between a hand's start and its first decision).
    fn process(&mut self, m: &Match, from: usize) {
        for (_, ev) in &m.events()[from..] {
            match *ev {
                Event::HandStart {
                    hand_index, dealer, ..
                } => {
                    self.current = Some(Current {
                        hand_index,
                        dealer_team: team_of(dealer),
                        start_scores: m.scores(),
                        start_flags: m.bonus_taken(),
                        games: m.games_won(),
                        trump: None,
                        winners: Vec::with_capacity(5),
                        five_trick: None,
                        best: None,
                        outcome: None,
                    });
                }
                Event::Turned { card } => {
                    if let Some(c) = self.current.as_mut()
                        && card != JOKER
                    {
                        c.trump = suit_of(card);
                    }
                }
                Event::Played { seat, card } => {
                    if let Some(c) = self.current.as_mut()
                        && let Some(trump) = c.trump
                    {
                        if card == five_of(trump) {
                            c.five_trick = Some((c.winners.len() + 1) as u8);
                        }
                        let rank = trump_rank(card, trump);
                        if rank > 0 && c.best.is_none_or(|(r, _)| rank > r) {
                            c.best = Some((rank, team_of(seat)));
                        }
                    }
                }
                Event::TrickWon { team, .. } => {
                    if let Some(c) = self.current.as_mut() {
                        c.winners.push((team + 4 - c.dealer_team) % 4);
                    }
                }
                Event::GameWon { team, reason, .. } => {
                    if let Some(c) = self.current.as_mut() {
                        c.outcome = Some(HandOutcome::GameWon { team, reason });
                    }
                }
                Event::HandEnd { .. } => {
                    if let Some(c) = self.current.take() {
                        let outcome = c
                            .outcome
                            .unwrap_or(HandOutcome::HandOver { scores: m.scores() });
                        self.done.push(HandRecord {
                            match_index: self.match_index,
                            hand_index: c.hand_index,
                            dealer_team: c.dealer_team,
                            start_scores: c.start_scores,
                            start_flags: c.start_flags,
                            games: c.games,
                            five_trick: c.five_trick,
                            best_trump: c.best.map(|(_, t)| (t + 4 - c.dealer_team) % 4),
                            winners: c.winners,
                            outcome,
                        });
                    }
                }
                _ => {}
            }
        }
    }
}

/// Play match `match_index` of an estimate and return its hands in order, stopping once
/// `max_hands` hands are recorded.
pub fn play_match_hands(
    factory: &dyn BotFactory,
    seed: u64,
    match_index: u64,
    rules: Rules,
    max_hands: u64,
) -> Result<Vec<HandRecord>, EstimateError> {
    let ms = match_seed(seed, match_index);
    let mut m = Match::new(ms, rules);
    let mut bots: Vec<Box<dyn Bot>> = (0..8).map(|_| factory.build()).collect();
    for (s, b) in bots.iter_mut().enumerate() {
        b.start_match(s as u8, rules, seat_seed(ms, s as u8));
    }
    let mut tracker = Tracker {
        match_index,
        current: None,
        done: Vec::new(),
    };
    tracker.process(&m, 0);
    let mut seen = m.events().len();
    while let Some(seat) = m.to_act() {
        if tracker.done.len() as u64 >= max_hands {
            break;
        }
        let view = m.view(seat);
        let legal = m.legal_actions();
        let bot = &mut bots[usize::from(seat)];
        let action = bot.act(&view, &legal);
        if !legal.contains(&action) || m.apply(action).is_err() {
            return Err(EstimateError::IllegalAction {
                match_index,
                seat,
                bot: bot.name().to_string(),
                action,
                legal,
            });
        }
        tracker.process(&m, seen);
        seen = m.events().len();
    }
    let keep = usize::try_from(max_hands).unwrap_or(usize::MAX);
    tracker.done.truncate(keep);
    Ok(tracker.done)
}

/// Play matches 0, 1, 2, … in order and return the first `hands` hands (for tests and checks;
/// the estimator itself counts without keeping them).
pub fn collect_hands(
    factory: &dyn BotFactory,
    hands: u64,
    seed: u64,
) -> Result<Vec<HandRecord>, EstimateError> {
    let mut out = Vec::new();
    let mut i = 0;
    while (out.len() as u64) < hands {
        let more = play_match_hands(factory, seed, i, Rules::HOUSE, hands - out.len() as u64)?;
        out.extend(more);
        i += 1;
    }
    Ok(out)
}

struct Counter {
    model: HandModel,
    hands: u64,
    completed: u64,
    cut_short: u64,
    tricks: u64,
    matches: u64,
}

impl Counter {
    fn new(bot: &str, seed: u64) -> Self {
        let source = ModelSource {
            kind: "self-play".into(),
            bot: bot.to_string(),
            seed,
            match_seed_rule: "match i (from 0): Rng::derive(seed, &[0xE57, i]).next_u64(); \
                              Match::new(match seed, Rules::HOUSE); hands taken in match order"
                .into(),
            seat_seed_rule: "seat s: Rng::derive(match seed, &[0xB07, s]).next_u64() \
                             (docs/M1-SPEC.md section 1)"
                .into(),
            matches: 0,
            hands: 0,
            hands_completed: 0,
            hands_cut_short: 0,
            tricks: 0,
            rules_version: RULES_VERSION,
            rules: Rules::HOUSE.into(),
        };
        Counter {
            model: HandModel::empty("outcome-v0", source),
            hands: 0,
            completed: 0,
            cut_short: 0,
            tricks: 0,
            matches: 0,
        }
    }

    fn add_match(&mut self, match_index: u64, records: &[HandRecord], hands: u64) {
        for r in records {
            if self.hands >= hands {
                return;
            }
            self.model.add_hand(&r.winners, r.five_trick, r.best_trump);
            self.hands += 1;
            self.tricks += r.winners.len() as u64;
            if r.winners.len() == 5 {
                self.completed += 1;
            } else {
                self.cut_short += 1;
            }
            self.matches = match_index + 1;
        }
    }

    fn finish(mut self) -> HandModel {
        let s = &mut self.model.source;
        s.matches = self.matches;
        s.hands = self.hands;
        s.hands_completed = self.completed;
        s.hands_cut_short = self.cut_short;
        s.tricks = self.tricks;
        self.model
    }
}

/// Estimate the hand-outcome model from `hands` self-play hands of the factory's bot (all eight
/// seats), on one thread. The model id is `outcome-v0`; change `model_id` to rename it.
pub fn estimate_model(
    bot_factory: &dyn BotFactory,
    hands: u64,
    seed: u64,
) -> Result<HandModel, EstimateError> {
    let mut c = Counter::new(bot_factory.name(), seed);
    let mut i = 0;
    while c.hands < hands {
        let recs = play_match_hands(bot_factory, seed, i, Rules::HOUSE, hands - c.hands)?;
        c.add_match(i, &recs, hands);
        i += 1;
    }
    Ok(c.finish())
}

/// Like [`estimate_model`], playing matches on several threads. The result is identical to the
/// single-threaded one for any thread count.
pub fn estimate_model_parallel(
    bot_factory: &(dyn BotFactory + Sync),
    hands: u64,
    seed: u64,
    threads: usize,
) -> Result<HandModel, EstimateError> {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Condvar, Mutex, mpsc};

    let threads = threads.max(1);
    if threads == 1 {
        return estimate_model(bot_factory, hands, seed);
    }
    // Work stealing over fixed-size blocks of consecutive matches: each worker claims the next
    // block from a shared counter as soon as it finishes one, so no thread waits for a slow
    // round. The merge consumes blocks strictly in match order, so the result does not depend
    // on the thread count or on timing. Workers stay at most `WINDOW` blocks ahead of the merge
    // so memory stays bounded.
    const BLOCK: u64 = 4;
    let window = 8 * threads as u64;
    type Block = Vec<Result<Vec<HandRecord>, EstimateError>>;
    let next = AtomicU64::new(0);
    let stop = AtomicBool::new(false);
    let merged = (Mutex::new(0u64), Condvar::new());
    let (tx, rx) = mpsc::channel::<(u64, Block)>();
    let mut c = Counter::new(bot_factory.name(), seed);
    let mut outcome = Ok(());
    std::thread::scope(|s| {
        for _ in 0..threads {
            let tx = tx.clone();
            let (next, stop, merged) = (&next, &stop, &merged);
            s.spawn(move || {
                loop {
                    let b = next.fetch_add(1, Ordering::Relaxed);
                    {
                        let (lock, cv) = merged;
                        let mut done = lock.lock().expect("merge lock");
                        while b >= *done + window && !stop.load(Ordering::Relaxed) {
                            done = cv.wait(done).expect("merge lock");
                        }
                    }
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    // `hands` caps any one match; a match's records are a prefix-stable
                    // sequence, so the merge's own cap reproduces the serial result.
                    let block: Block = (b * BLOCK..(b + 1) * BLOCK)
                        .map(|i| play_match_hands(bot_factory, seed, i, Rules::HOUSE, hands))
                        .collect();
                    if tx.send((b, block)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        let mut pending: BTreeMap<u64, Block> = BTreeMap::new();
        let mut want = 0u64;
        'merge: while c.hands < hands {
            let Ok((b, block)) = rx.recv() else { break };
            pending.insert(b, block);
            while let Some(block) = pending.remove(&want) {
                for (i, r) in (want * BLOCK..).zip(block) {
                    if c.hands >= hands {
                        break 'merge;
                    }
                    let recs = match r {
                        Ok(recs) => recs,
                        // The serial estimate plays this match capped at the hands still
                        // needed; replay it that way so an error past the cut does not count.
                        Err(_) => match play_match_hands(
                            bot_factory,
                            seed,
                            i,
                            Rules::HOUSE,
                            hands - c.hands,
                        ) {
                            Ok(recs) => recs,
                            Err(e) => {
                                outcome = Err(e);
                                break 'merge;
                            }
                        },
                    };
                    c.add_match(i, &recs, hands);
                }
                want += 1;
                let (lock, cv) = &merged;
                *lock.lock().expect("merge lock") = want;
                cv.notify_all();
            }
        }
        stop.store(true, Ordering::Relaxed);
        let (lock, cv) = &merged;
        drop(lock.lock().expect("merge lock"));
        cv.notify_all();
        drop(rx);
    });
    outcome?;
    Ok(c.finish())
}

#[cfg(test)]
mod parallel_tests {
    use super::*;
    use crate::testbots::RandomTestFactory;

    /// The work-stealing merge gives the serial result for any thread count, including counts
    /// far above the block count and hand totals that end mid-match and mid-block.
    #[test]
    fn work_stealing_is_thread_count_independent() {
        for hands in [1, 37, 2_001] {
            let m = estimate_model(&RandomTestFactory, hands, 9).expect("legal play");
            for threads in [2, 4, 7, 64] {
                let p = estimate_model_parallel(&RandomTestFactory, hands, 9, threads)
                    .expect("legal play");
                assert_eq!(p, m, "{hands} hands, {threads} threads");
            }
        }
    }
}
