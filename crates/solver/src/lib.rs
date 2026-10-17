//! Exact open-hand ("all cards face up") max^n solver for 41, scored in money chance
//! (PLAN §6.7 and amendment A2).
//!
//! Two solvers live here. [`solve`] is the **M2 measurement spike**: a deliberately naive,
//! obviously-correct solver with no pruning, no move ordering and no equivalent cards; it stays
//! as the reference. [`solve_fast`] (module [`fast`]) is the optimised solver, gated to give
//! identical answers (see that module). The rest of this text describes the naive solver. It exists to *measure* costs (nodes, time,
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
//! **Hybrid frontier (docs/M3-HYBRID-SPEC.md).** [`solve_cut`], [`solve_fast_cut`] and
//! [`solve_fast_par_cut`] take an optional [`FrontierCut`]: positions reached at the start of
//! trick `k` are valued by an evaluator (the oracle network O_k) instead of being searched. The
//! evaluator sees the canonical form of the position ([`frontier`]), which keeps equivalent-card
//! merging and the transposition table exact, so the three solvers return identical hybrid
//! values. Without a frontier every solve is exactly what it was.
//!
//! Everything is whole-number: values are [`MoneyChance`] (units of 2^-32, summing to exactly 1).

mod clock;
pub mod denial;
pub mod equiv;
pub mod fast;
pub mod frontier;
mod ranks;
mod tt;
pub mod worlds;

pub use fast::{
    FastOptions, ParProgress, SolveFamily, SolveMem, solve_fast, solve_fast_cut, solve_fast_par,
    solve_fast_par_cut, solve_fast_par_split, solve_fast_par_with, solve_fast_watched,
    solve_fast_with,
};
pub use frontier::{CanonPos, FnFrontier, Frontier, FrontierCut, canonical};
pub use tt::{TableMemory, shared_table_memory, table_memory};
pub use worlds::sample_world;

use crate::clock::Stopwatch;
use std::collections::HashMap;

use engine::cards::{Bits, Card, Suit, bit, has};
use engine::game::{Match, Phase};
use engine::rules::{ace_of_trumps, five_of, legal_burns, legal_plays, trick_winner, trump_rank};
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

/// A rob still to be made in trick 1 by a seat other than the root mover (rule 20, H2): at its
/// first turn the seat takes `card` (the turned card) and burns any card but the ace of trumps
/// (rule 21, H3), then plays. The burn is a decision node of the solve (see
/// [`Position::burn_choice`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rob {
    /// The robbing seat.
    pub seat: u8,
    /// The turned card it takes.
    pub card: Card,
}

/// A fully known position inside a hand. Robbing must already be resolved (every hand holds
/// exactly the cards it will play), except a pending rob recorded in [`Position::rob`].
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
    /// A rob still to be made in the current trick (`None` once made, or when there is none).
    /// [`from_match`] leaves it `None`; a caller sets it from the pending seat it returns.
    pub rob: Option<Rob>,
    /// Partnerships turned this hand (bit `t` for team `t`; rules 39-41 and 34,
    /// docs/APP-ENGINE-CONTRACT.md §7): their seats hold no cards here and are skipped in the
    /// play order, a trick is complete when every seat still in play has played, and `hand` and
    /// `best` already count only the partnerships still in play (the engine passed the turned
    /// pair's tricks and took its points away when it was turned). 0 when nobody is turned;
    /// such a position is never valued by a frontier evaluator (see [`frontier`]).
    pub turned: u8,
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
    /// Optimised solver only: table entries overwritten by the replacement policy.
    pub tt_replaced: u64,
    /// Optimised solver only: nodes stopped early by shallow pruning (plus nodes stopped because
    /// the mover's team already had a certain win).
    pub cutoffs: u64,
    /// Optimised solver only: legal cards not searched because an equivalent card was.
    pub equiv_skipped: u64,
    /// Optimised solver only: bytes allocated by the transposition table at the end.
    pub tt_bytes: u64,
    /// Optimised solver only: nodes by trick (index = tricks done this hand, 0..5), so the
    /// weight of each trick in a solve is visible.
    pub nodes_by_trick: [u64; 5],
    /// Optimised solver only: cards played by the forced-play shortcut (a seat with one legal
    /// card, and the whole last trick), each in place of a node (see `fast::Search::forced`).
    pub forced: u64,
    /// Positions valued by the frontier evaluator (hybrid solves only; 0 without a frontier).
    pub frontier: u64,
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
    let mut turned = 0u8;
    for (t, &out) in p.turned_teams.iter().enumerate() {
        turned |= u8::from(out) << t;
    }
    // A turned seat's cards are face down and out of play (rule 40).
    let hands: [Bits; SEATS] = core::array::from_fn(|s| {
        if turned & (1 << team_of(s as u8)) == 0 {
            m.truth().hands[s]
        } else {
            0
        }
    });
    let mut trick = [0 as Card; SEATS];
    for (slot, &(_, c)) in trick.iter_mut().zip(&p.trick_cards) {
        *slot = c;
    }
    let mut hand = InHand::start_with_flags(p.scores, p.bonus_taken);
    hand.tricks = p.tricks_won;
    // `InHand::five_played` means "the 5 was in a *completed* trick" (its bonus is scored when
    // its trick is won); the engine's public flag is set as soon as the 5 hits the table.
    hand.five_played = p.five_played && !p.trick_cards.iter().any(|&(_, c)| c == five_of(trump));
    hand.tricks_done = p.trick_winners.len() as u8;
    // The ace holder robs at its first turn in trick 1; one that already played its first card
    // without robbing (a play the app's referee let through) robs no more.
    let pending = if p.trick_no == 1 && p.robber.is_none() && p.table_card.is_some() {
        let ace = ace_of_trumps(trump);
        (0..SEATS as u8)
            .find(|&s| has(hands[usize::from(s)], ace))
            .filter(|&s| !p.proved_no_ace[usize::from(s)])
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
            rob: None,
            turned,
        },
        MatchCtx {
            games: p.games_won,
            dealer_team: p.dealer % 4,
        },
        pending,
    ))
}

/// What one card leads to (see [`Position::play`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// The hand goes on from this position.
    Next(Position),
    /// The game ends at once: this team won it.
    GameWon(u8),
    /// The hand is over without a game end, with these scores (after the end-of-hand bonus).
    HandOver([u8; 4]),
}

impl Position {
    /// The position after the seat to act plays `c` (which must be legal): the trick is
    /// completed and scored when it is the eighth card. Used by the optimised solver; the naive
    /// solver keeps its own copy of these few lines, and both are parity-tested against the
    /// engine.
    pub fn play(&self, c: Card) -> Step {
        if self.turned == 0 {
            self.play_plain(c)
        } else {
            self.play_turned(c)
        }
    }

    /// [`Position::play`] when nobody is turned: exactly the code the solver always ran.
    pub(crate) fn play_plain(&self, c: Card) -> Step {
        let seat = (self.leader + self.n_played) % SEATS as u8;
        let mut q = *self;
        q.hands[usize::from(seat)] &= !bit(c);
        q.trick[usize::from(q.n_played)] = c;
        q.n_played += 1;
        // The engine's ranks and trick winner, read from tables (`ranks`: the same values).
        let r = ranks::trump_rank_of(c, q.trump);
        if r > 0 && q.best.is_none_or(|(b, _)| r > b) {
            q.best = Some((r, team_of(seat)));
        }
        if usize::from(q.n_played) < SEATS {
            return Step::Next(q);
        }
        let w = ranks::trick_winner8(&q.trick, q.trump);
        let winner = (q.leader + w as u8) % SEATS as u8;
        // The 5 of trumps is the highest card of all, so it is in the trick exactly when it
        // wins it.
        let five_here = q.trick[w] == five_of(q.trump);
        match q
            .hand
            .play_trick(team_of(winner), five_here, Scoring::HOUSE)
        {
            AfterTrick::GameWon(t, _) => Step::GameWon(t),
            AfterTrick::Next(h) => {
                q.hand = h;
                q.leader = winner;
                q.n_played = 0;
                Step::Next(q)
            }
            AfterTrick::TricksDone(h) => match h.finish(q.best.map(|(_, t)| t), Scoring::HOUSE) {
                AfterHand::GameWon(t, _) => Step::GameWon(t),
                AfterHand::HandOver(s) => Step::HandOver(s),
            },
        }
    }

    /// [`Position::play`] when a partnership is turned (rule 40: the trick is played by the seats
    /// still in play only), kept out of the hot path of hands where nobody is turned.
    #[cold]
    #[inline(never)]
    pub(crate) fn play_turned(&self, c: Card) -> Step {
        let seat = self.seat_at_turned(self.n_played);
        let mut q = *self;
        q.hands[usize::from(seat)] &= !bit(c);
        q.trick[usize::from(q.n_played)] = c;
        q.n_played += 1;
        let r = ranks::trump_rank_of(c, q.trump);
        if r > 0 && q.best.is_none_or(|(b, _)| r > b) {
            q.best = Some((r, team_of(seat)));
        }
        if q.n_played < q.seats_in_play() {
            return Step::Next(q);
        }
        let w = ranks::trick_winner_of(&q.trick[..usize::from(q.n_played)], q.trump);
        let winner = q.seat_at_turned(w as u8);
        // As in `play_plain`: the trick's points, the 5's bonus, then the next trick or the end.
        let five_here = q.trick[w] == five_of(q.trump);
        match q
            .hand
            .play_trick(team_of(winner), five_here, Scoring::HOUSE)
        {
            AfterTrick::GameWon(t, _) => Step::GameWon(t),
            AfterTrick::Next(h) => {
                q.hand = h;
                q.leader = winner;
                q.n_played = 0;
                Step::Next(q)
            }
            AfterTrick::TricksDone(h) => match h.finish(q.best.map(|(_, t)| t), Scoring::HOUSE) {
                AfterHand::GameWon(t, _) => Step::GameWon(t),
                AfterHand::HandOver(s) => Step::HandOver(s),
            },
        }
    }

    /// Every card still in a hand or on the table in the current trick.
    pub fn live(&self) -> Bits {
        let mut live = self.hands.iter().fold(0, |a, &h| a | h);
        for &c in &self.trick[..usize::from(self.n_played)] {
            live |= bit(c);
        }
        // A pending robber's turned card is live until the burn: it may stay in play, so no
        // card may be merged across it (the burnt card leaves play in [`Position::burnt`]).
        if let Some(r) = self.rob {
            live |= bit(r.card);
        }
        live
    }

    /// The seat to play next.
    #[inline]
    pub fn to_act(&self) -> u8 {
        if self.turned == 0 {
            (self.leader + self.n_played) % SEATS as u8
        } else {
            self.seat_at_turned(self.n_played)
        }
    }

    /// Is the seat still in play (its partnership not turned)?
    #[inline]
    pub fn in_play(&self, seat: u8) -> bool {
        self.turned & (1 << team_of(seat)) == 0
    }

    /// How many seats play to each trick: 8, or 6 or 4 with turned partnerships.
    #[inline]
    pub fn seats_in_play(&self) -> u8 {
        if self.turned == 0 {
            SEATS as u8
        } else {
            SEATS as u8 - 2 * self.turned.count_ones() as u8
        }
    }

    /// The seat that plays the `i`-th card (from 0) of the current trick: the `i`-th seat still
    /// in play from the leader, going left (the leader is always in play).
    #[inline]
    pub fn seat_at(&self, i: u8) -> u8 {
        if self.turned == 0 {
            (self.leader + i) % SEATS as u8
        } else {
            self.seat_at_turned(i)
        }
    }

    /// [`Position::seat_at`] when a partnership is turned (kept out of the hot path).
    #[cold]
    #[inline(never)]
    pub(crate) fn seat_at_turned(&self, i: u8) -> u8 {
        let mut s = self.leader;
        let mut k = i;
        loop {
            if self.in_play(s) {
                if k == 0 {
                    return s;
                }
                k -= 1;
            }
            s = (s + 1) % SEATS as u8;
        }
    }

    /// When the seat to act must rob first: its hand with the turned card taken, and the cards it
    /// may burn (any but the ace of trumps: rule 21, H3). `None` otherwise. At such a position
    /// the seat's choice is its burn (a node of the solve); [`Position::legal`] and
    /// [`Position::play`] apply only after [`Position::burnt`].
    pub fn burn_choice(&self) -> Option<(Bits, Bits)> {
        let r = self.rob?;
        if r.seat != self.to_act() {
            return None;
        }
        let h = self.hands[usize::from(r.seat)] | bit(r.card);
        Some((h, legal_burns(h, Some(self.trump))))
    }

    /// The position after the pending robber (the seat to act) takes the turned card and burns
    /// `b` (which must be in [`Position::burn_choice`]'s burnable set).
    pub fn burnt(&self, b: Card) -> Position {
        let r = self.rob.expect("a rob is pending");
        debug_assert_eq!(r.seat, self.to_act());
        let mut q = *self;
        q.hands[usize::from(r.seat)] = (q.hands[usize::from(r.seat)] | bit(r.card)) & !bit(b);
        q.rob = None;
        q
    }

    /// The legal cards of the seat to act (engine rules).
    pub fn legal(&self) -> Bits {
        let led = (self.n_played > 0).then(|| self.trick[0]);
        legal_plays(self.hands[usize::from(self.to_act())], led, self.trump)
    }

    // The optimised solver's hot path, for a solve known to have no turned partnership
    // (`T = false`: exactly the code it always ran) or known to have one (`T = true`). Within a
    // solve the turned partnerships never change, so the solver picks `T` once, at the root.

    /// [`Position::to_act`] (the seat to play) for a solve with (`T`) or without turned
    /// partnerships.
    #[inline(always)]
    pub(crate) fn mover_as<const T: bool>(&self) -> u8 {
        debug_assert_eq!(T, self.turned != 0);
        if T {
            self.seat_at_turned(self.n_played)
        } else {
            (self.leader + self.n_played) % SEATS as u8
        }
    }

    /// [`Position::play`] for a solve with (`T`) or without turned partnerships.
    #[inline(always)]
    pub(crate) fn play_as<const T: bool>(&self, c: Card) -> Step {
        if T {
            self.play_turned(c)
        } else {
            self.play_plain(c)
        }
    }

    /// [`Position::burn_choice`] for a solve with (`T`) or without turned partnerships.
    #[inline(always)]
    pub(crate) fn burn_choice_as<const T: bool>(&self) -> Option<(Bits, Bits)> {
        let r = self.rob?;
        if r.seat != self.mover_as::<T>() {
            return None;
        }
        let h = self.hands[usize::from(r.seat)] | bit(r.card);
        Some((h, legal_burns(h, Some(self.trump))))
    }

    /// [`Position::legal`] for a solve with (`T`) or without turned partnerships.
    #[inline(always)]
    pub(crate) fn legal_as<const T: bool>(&self) -> Bits {
        let led = (self.n_played > 0).then(|| self.trick[0]);
        legal_plays(
            self.hands[usize::from(self.mover_as::<T>())],
            led,
            self.trump,
        )
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
    turned: u8,
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
            turned: p.turned,
        }
    }
}

/// Bytes per stored transposition entry (key + value), for memory estimates.
pub const TT_ENTRY_BYTES: usize = core::mem::size_of::<Key>() + core::mem::size_of::<MoneyChance>();

struct Search<'a, L: Leaf> {
    ctx: MatchCtx,
    leaf: &'a L,
    opts: Options,
    cut: Option<FrontierCut<'a>>,
    tt: HashMap<Key, MoneyChance>,
    stats: Stats,
    start: Stopwatch,
    aborted: bool,
}

/// The cards of a set in the tie order.
pub(crate) fn ordered(set: Bits, order: TieOrder) -> impl Iterator<Item = Card> {
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
        // The seats that play to this trick, in order from the leader: all eight, or those of
        // the partnerships not turned (rule 40).
        let mut order = [0u8; SEATS];
        let mut n = 0;
        for i in 0..SEATS as u8 {
            let s = (q.leader + i) % SEATS as u8;
            if q.turned >> team_of(s) & 1 == 0 {
                order[n] = s;
                n += 1;
            }
        }
        if usize::from(q.n_played) < n {
            return self.node(&q);
        }
        let w = trick_winner(&q.trick[..n], q.trump);
        let winner = order[w];
        let five_here = q.trick[..n].contains(&five_of(q.trump));
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
            && self.start.elapsed_ms() >= cap
        {
            self.aborted = true;
        }
        if self.aborted {
            return None;
        }
        if let Some(cut) = self.cut
            && cut.applies(p)
        {
            self.stats.frontier += 1;
            return Some(cut.value(p, &self.ctx));
        }
        if let Some((_, burns)) = p.burn_choice() {
            return self.burn_node(p, burns);
        }
        let key = (self.opts.tt && p.n_played == 0 && p.rob.is_none()).then(|| Key::of(p));
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

    /// A pending robber's burn: every legal burn (any card but the ace of trumps) in the tie
    /// order, the robber's team's best kept, the first strictly best among equals (the same
    /// backup rule as a card).
    fn burn_node(&mut self, p: &Position, burns: Bits) -> Option<MoneyChance> {
        let team = team_of(p.to_act());
        let mut best: Option<MoneyChance> = None;
        let mut tied = 1u32;
        for b in ordered(burns, self.opts.order) {
            let v = self.node(&p.burnt(b))?;
            match best {
                None => best = Some(v),
                Some(x) if v.get(team) > x.get(team) => {
                    best = Some(v);
                    tied = 1;
                }
                Some(x) if v.get(team) == x.get(team) => tied += 1,
                Some(_) => {}
            }
        }
        if burns.count_ones() >= 2 {
            self.stats.decision_nodes += 1;
            if tied >= 2 {
                self.stats.tied_nodes += 1;
            }
        }
        best
    }
}

/// Solve a position exactly: every legal root card is valued and the mover's team's best is
/// chosen under the tie order.
pub fn solve<L: Leaf>(pos: &Position, ctx: MatchCtx, leaf: &L, opts: Options) -> Solution {
    solve_cut(pos, ctx, leaf, opts, None)
}

/// [`solve`] with an optional frontier (the hybrid H_k, docs/M3-HYBRID-SPEC.md §2): every
/// position reached at the start of the frontier's trick is valued by its evaluator instead of
/// being searched; everything else is [`solve`]'s search. With `None` this is exactly [`solve`].
pub fn solve_cut<L: Leaf>(
    pos: &Position,
    ctx: MatchCtx,
    leaf: &L,
    opts: Options,
    cut: Option<FrontierCut>,
) -> Solution {
    let mut s = Search {
        ctx,
        leaf,
        opts,
        cut,
        tt: HashMap::new(),
        stats: Stats::default(),
        start: Stopwatch::start(),
        aborted: false,
    };
    s.stats.nodes += 1;
    assert!(
        pos.burn_choice().is_none(),
        "the root mover's own burn is decided outside the solve"
    );
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

/// The index (in play order) of the card currently winning a partial trick (rule 19).
pub(crate) fn trick_winner_index(plays: &[Card], trump: Suit) -> usize {
    trick_winner(plays, trump)
}
