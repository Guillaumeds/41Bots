//! Exact open-hand ("all cards face up") max^n solver for 41, scored in money chance
//! (PLAN §6.7 and amendment A2).
//!
//! This is the **M2 measurement spike**: a deliberately naive, obviously-correct solver with no
//! pruning, no move ordering and no equivalent cards. It exists to *measure* costs (nodes, time,
//! memory, ties) before any optimisation is designed. An optional plain transposition table [a
//! memory of positions already solved] keyed at trick boundaries can be switched on; it must give
//! identical values.
//!
//! **What it solves.** One fully known world (all hands, trumps, robbing already resolved), from a
//! position inside the current hand, with the match state around it. Every finished line is
//! valued by a [`Leaf`] evaluator (normally the match-equity table, [`equity::EquityTable`]):
//! - the game ends mid-hand (41 or jink): the winner's games go up by one, scores reset, the deal
//!   moves left; a second game won is certain money for that team;
//! - the hand ends without a game end: the next hand's state (scores after the end-of-hand bonus,
//!   the deal moves left) is looked up.
//!
//! Scoring inside the hand uses [`equity::hand::InHand`], which follows the engine's exact order
//! (trick points → immediate 5-of-trumps bonus → 41 → hand over; after trick 5: jink → end bonus →
//! 41) and is parity-tested against the engine. Legal cards and trick winners come from
//! [`engine::rules`]; no rule is re-implemented here.
//!
//! **Backup rule (pre-M4, PLAN A2).** At a seat's node the solver picks the child that maximises
//! that seat's *team* entry of the money chance. **Tie order:** children are tried in a fixed
//! position-only order, by default increasing card index (engine card numbering), and a later
//! child replaces the current best only if it is *strictly* better for the mover's team; so among
//! tied best cards the first in that order is kept. [`TieOrder::Descending`] reverses the order,
//! for the root-flip measurement.
//!
//! Everything is whole-number: values are [`MoneyChance`] (units of 2^-32, summing to exactly 1).

use std::collections::HashMap;
use std::time::Instant;

use engine::cards::{Bits, Card, Suit, bit, has};
use engine::game::{Match, Phase};
use engine::rules::{ace_of_trumps, five_of, legal_plays, trick_winner, trump_rank};
use engine::{SEATS, team_of};
use equity::hand::{AfterHand, AfterTrick, InHand, Scoring};
use equity::{EquityTable, MatchState, MoneyChance};

/// Values the end of a line (the match state the line leads to).
pub trait Leaf {
    /// The money chance of a match state that is not finished.
    fn value(&self, state: &MatchState) -> MoneyChance;
}

impl Leaf for EquityTable {
    fn value(&self, state: &MatchState) -> MoneyChance {
        self.lookup(state)
    }
}

/// A closure as a leaf evaluator (for tests and synthetic tables).
pub struct FnLeaf<F>(pub F);

impl<F: Fn(&MatchState) -> MoneyChance> Leaf for FnLeaf<F> {
    fn value(&self, state: &MatchState) -> MoneyChance {
        (self.0)(state)
    }
}

/// The match around the hand: games won and the team of this hand's dealer. (Scores live in
/// [`Position::hand`].)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatchCtx {
    /// Games each team has won so far (each 0 or 1 while the match goes on).
    pub games: [u8; 4],
    /// The team of the dealer of the current hand.
    pub dealer_team: u8,
}

/// A fully known position inside a hand. Robbing must already be resolved (every hand holds
/// exactly the cards it will play).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Position {
    /// Each seat's remaining cards.
    pub hands: [Bits; SEATS],
    /// The trump suit.
    pub trump: Suit,
    /// The seat that led (or will lead) the current trick.
    pub leader: u8,
    /// Cards played to the current trick, in play order (the first `n_played` are valid).
    pub trick: [Card; SEATS],
    /// How many cards are in the current trick (0..8).
    pub n_played: u8,
    /// Scores, bonus flags, tricks per team, whether the 5 was played, tricks done.
    pub hand: InHand,
    /// The best trump played so far this hand: (its trump rank, the team that played it).
    pub best: Option<(u8, u8)>,
}

/// The order children are tried in (the tie rule: the first strictly best is kept).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TieOrder {
    /// Increasing card index (the documented default).
    #[default]
    Ascending,
    /// Decreasing card index (only for measuring how often the root choice flips).
    Descending,
}

/// Solver switches.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// Use the plain transposition table at trick boundaries.
    pub tt: bool,
    /// The tie order.
    pub order: TieOrder,
    /// Give up after this many milliseconds (checked every 4096 nodes); `None` = no limit.
    pub time_cap_ms: Option<u64>,
}

/// Counters from one solve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Nodes expanded (one per position in which a card is chosen).
    pub nodes: u64,
    /// Leaves valued (finished lines).
    pub leaves: u64,
    /// Nodes with at least two legal cards.
    pub decision_nodes: u64,
    /// Decision nodes where two or more cards tie for the mover's best value.
    pub tied_nodes: u64,
    /// Transposition-table probes and hits (trick boundaries only).
    pub tt_probes: u64,
    /// Probes that found the position.
    pub tt_hits: u64,
    /// Positions stored in the table at the end.
    pub tt_entries: u64,
}

/// The result of a root solve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solution {
    /// The root value (`None` if the time cap was hit).
    pub value: Option<MoneyChance>,
    /// The chosen card at the root (`None` on timeout).
    pub best: Option<Card>,
    /// Every legal root card with its value, in the tie order (empty on timeout).
    pub children: Vec<(Card, MoneyChance)>,
    /// How many root cards tie for the root mover's best value.
    pub root_tied: u32,
    /// Counters.
    pub stats: Stats,
    /// Whether the time cap was hit.
    pub timed_out: bool,
}

/// Why a match position cannot be turned into a solver position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FromMatchError {
    /// The match is over.
    MatchOver,
    /// A burn is awaited.
    AwaitingBurn,
    /// Trumps are not known yet.
    NoTrump,
}

/// A match position (awaiting a play) as a solver position. Returns the position, the match
/// context, and the seat that still has to rob in trick 1, if any: the caller must resolve that
/// rob (add the turned card to that hand and remove a burn) before solving.
pub fn from_match(m: &Match) -> Result<(Position, MatchCtx, Option<u8>), FromMatchError> {
    match m.phase() {
        Phase::MatchOver => return Err(FromMatchError::MatchOver),
        Phase::AwaitBurn(_) => return Err(FromMatchError::AwaitingBurn),
        Phase::AwaitPlay(_) => {}
    }
    let p = m.public();
    let trump = p.trump.ok_or(FromMatchError::NoTrump)?;
    let hands = m.truth().hands;
    let mut trick = [0 as Card; SEATS];
    for (slot, &(_, c)) in trick.iter_mut().zip(&p.trick_cards) {
        *slot = c;
    }
    let mut hand = InHand::start_with_flags(p.scores, p.bonus_taken);
    hand.tricks = p.tricks_won;
    hand.five_played = p.five_played;
    hand.tricks_done = p.trick_winners.len() as u8;
    let pending = if p.trick_no == 1 && p.robber.is_none() && p.table_card.is_some() {
        let ace = ace_of_trumps(trump);
        (0..SEATS as u8).find(|&s| has(hands[usize::from(s)], ace))
    } else {
        None
    };
    Ok((
        Position {
            hands,
            trump,
            leader: p.leader,
            trick,
            n_played: p.trick_cards.len() as u8,
            hand,
            best: p
                .best_trump
                .map(|(c, s)| (trump_rank(c, trump), team_of(s))),
        },
        MatchCtx {
            games: p.games_won,
            dealer_team: p.dealer % 4,
        },
        pending,
    ))
}

impl Position {
    /// The seat to play next.
    pub fn to_act(&self) -> u8 {
        (self.leader + self.n_played) % SEATS as u8
    }

    /// The legal cards of the seat to act (engine rules).
    pub fn legal(&self) -> Bits {
        let led = (self.n_played > 0).then(|| self.trick[0]);
        legal_plays(self.hands[usize::from(self.to_act())], led, self.trump)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    hands: [Bits; SEATS],
    leader: u8,
    scores: [u8; 4],
    flags: u8,
    tricks: [u8; 4],
    tricks_done: u8,
    five_played: bool,
    best: Option<(u8, u8)>,
}

impl Key {
    fn of(p: &Position) -> Key {
        let mut flags = 0u8;
        for t in 0..4 {
            flags |= u8::from(p.hand.flags[t]) << t;
        }
        Key {
            hands: p.hands,
            leader: p.leader,
            scores: p.hand.scores,
            flags,
            tricks: p.hand.tricks,
            tricks_done: p.hand.tricks_done,
            five_played: p.hand.five_played,
            best: p.best,
        }
    }
}

/// Bytes per stored transposition entry (key + value), for memory estimates.
pub const TT_ENTRY_BYTES: usize = core::mem::size_of::<Key>() + core::mem::size_of::<MoneyChance>();

struct Search<'a, L: Leaf> {
    ctx: MatchCtx,
    leaf: &'a L,
    opts: Options,
    tt: HashMap<Key, MoneyChance>,
    stats: Stats,
    start: Instant,
    aborted: bool,
}

/// The cards of a set in the tie order.
fn ordered(set: Bits, order: TieOrder) -> impl Iterator<Item = Card> {
    let mut s = set;
    core::iter::from_fn(move || {
        if s == 0 {
            return None;
        }
        let c = match order {
            TieOrder::Ascending => s.trailing_zeros(),
            TieOrder::Descending => 63 - s.leading_zeros(),
        } as Card;
        s &= !bit(c);
        Some(c)
    })
}

impl<L: Leaf> Search<'_, L> {
    fn game_won(&mut self, team: u8) -> MoneyChance {
        self.stats.leaves += 1;
        let mut games = self.ctx.games;
        games[usize::from(team)] += 1;
        if games[usize::from(team)] >= engine::game::GAMES_TO_WIN {
            return MoneyChance::certain(team);
        }
        self.leaf.value(&MatchState {
            scores: [0; 4],
            games,
            dealer_team: (self.ctx.dealer_team + 1) % 4,
        })
    }

    fn hand_over(&mut self, scores: [u8; 4]) -> MoneyChance {
        self.stats.leaves += 1;
        self.leaf.value(&MatchState {
            scores,
            games: self.ctx.games,
            dealer_team: (self.ctx.dealer_team + 1) % 4,
        })
    }

    /// The value after `seat` plays `c` in `p`.
    fn child(&mut self, p: &Position, seat: u8, c: Card) -> Option<MoneyChance> {
        let mut q = *p;
        q.hands[usize::from(seat)] &= !bit(c);
        q.trick[usize::from(q.n_played)] = c;
        q.n_played += 1;
        let r = trump_rank(c, q.trump);
        if r > 0 && q.best.is_none_or(|(b, _)| r > b) {
            q.best = Some((r, team_of(seat)));
        }
        if usize::from(q.n_played) < SEATS {
            return self.node(&q);
        }
        let w = trick_winner(&q.trick, q.trump);
        let winner = (q.leader + w as u8) % SEATS as u8;
        let five_here = q.trick.contains(&five_of(q.trump));
        match q
            .hand
            .play_trick(team_of(winner), five_here, Scoring::HOUSE)
        {
            AfterTrick::GameWon(t, _) => Some(self.game_won(t)),
            AfterTrick::Next(h) => {
                q.hand = h;
                q.leader = winner;
                q.n_played = 0;
                self.node(&q)
            }
            AfterTrick::TricksDone(h) => match h.finish(q.best.map(|(_, t)| t), Scoring::HOUSE) {
                AfterHand::GameWon(t, _) => Some(self.game_won(t)),
                AfterHand::HandOver(s) => Some(self.hand_over(s)),
            },
        }
    }

    fn node(&mut self, p: &Position) -> Option<MoneyChance> {
        self.stats.nodes += 1;
        if self.stats.nodes & 0xFFF == 0
            && let Some(cap) = self.opts.time_cap_ms
            && self.start.elapsed().as_millis() as u64 >= cap
        {
            self.aborted = true;
        }
        if self.aborted {
            return None;
        }
        let key = (self.opts.tt && p.n_played == 0).then(|| Key::of(p));
        if let Some(k) = &key {
            self.stats.tt_probes += 1;
            if let Some(v) = self.tt.get(k) {
                self.stats.tt_hits += 1;
                return Some(*v);
            }
        }
        let seat = p.to_act();
        let team = team_of(seat);
        let legal = p.legal();
        debug_assert!(legal != 0, "the legal set is never empty");
        let mut best: Option<MoneyChance> = None;
        let mut tied = 1u32;
        for c in ordered(legal, self.opts.order) {
            let v = self.child(p, seat, c)?;
            match best {
                None => best = Some(v),
                Some(b) if v.get(team) > b.get(team) => {
                    best = Some(v);
                    tied = 1;
                }
                Some(b) if v.get(team) == b.get(team) => tied += 1,
                Some(_) => {}
            }
        }
        if legal.count_ones() >= 2 {
            self.stats.decision_nodes += 1;
            if tied >= 2 {
                self.stats.tied_nodes += 1;
            }
        }
        let v = best?;
        if let Some(k) = key {
            self.tt.insert(k, v);
        }
        Some(v)
    }
}

/// Solve a position exactly: every legal root card is valued and the mover's team's best is
/// chosen under the tie order.
pub fn solve<L: Leaf>(pos: &Position, ctx: MatchCtx, leaf: &L, opts: Options) -> Solution {
    let mut s = Search {
        ctx,
        leaf,
        opts,
        tt: HashMap::new(),
        stats: Stats::default(),
        start: Instant::now(),
        aborted: false,
    };
    s.stats.nodes += 1;
    let seat = pos.to_act();
    let team = team_of(seat);
    let legal = pos.legal();
    let mut children = Vec::new();
    for c in ordered(legal, opts.order) {
        match s.child(pos, seat, c) {
            Some(v) => children.push((c, v)),
            None => {
                s.stats.tt_entries = s.tt.len() as u64;
                return Solution {
                    value: None,
                    best: None,
                    children: Vec::new(),
                    root_tied: 0,
                    stats: s.stats,
                    timed_out: true,
                };
            }
        }
    }
    let mut best = 0usize;
    for (i, (_, v)) in children.iter().enumerate() {
        if v.get(team) > children[best].1.get(team) {
            best = i;
        }
    }
    let top = children[best].1.get(team);
    let root_tied = children.iter().filter(|(_, v)| v.get(team) == top).count() as u32;
    if legal.count_ones() >= 2 {
        s.stats.decision_nodes += 1;
        if root_tied >= 2 {
            s.stats.tied_nodes += 1;
        }
    }
    s.stats.tt_entries = s.tt.len() as u64;
    Solution {
        value: Some(children[best].1),
        best: Some(children[best].0),
        children,
        root_tied,
        stats: s.stats,
        timed_out: false,
    }
}
