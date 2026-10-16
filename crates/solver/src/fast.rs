//! The optimised exact solver (M2). It must return **exactly** what the naive solver
//! ([`crate::solve`]) returns: the same root value, the same value for every root card, and the
//! same chosen card under the same tie order. Each speed-up has its own switch in
//! [`FastOptions`] so that "on" can be tested against "off" (and against the naive solver).
//!
//! # The speed-ups
//! 1. **Transposition table** at trick boundaries, bounded memory, key per PLAN A2 (see the
//!    `tt` module for the key, the hash and why replacement cannot change an answer).
//! 2. **Equivalent cards** (see [`crate::equiv`]): of a group of equivalent cards only the first
//!    in the tie order is searched; the others get its value. Because the naive solver keeps the
//!    first strictly best card in the tie order, and equivalent cards have equal values, it never
//!    reports a card that is not first in its group, so the reported card is unchanged.
//! 3. **Move ordering**: cards are *tried* in a heuristic order (likely good cards first, so
//!    pruning happens sooner), but the *choice* is still made by the tie rule: the best value for
//!    the mover's team, and among equal values the card first in the tie order. So the order of
//!    trying cannot change which card or which value is chosen.
//! 4. **Shallow pruning** (Sturtevant & Korf 2000, for max^n with a known bound on the sum of
//!    the values), proof below.
//!
//! With a hybrid frontier ([`crate::frontier`]) every speed-up stays exact: a frontier value is
//! a pure function of the position's canonical form, which equivalent cards share and the table
//! key determines; pruning only ever compares values, wherever they come from.
//!
//! # Shallow pruning: why it is safe here
//! Every value is four non-negative whole numbers summing to exactly `ONE`. Take a node `N`
//! whose mover's team is `t`, which has already found a best value `b` for `t` (from card `cb`),
//! and is now searching card `c`, which leads to a node `C` whose mover's team is `u ≠ t`.
//! While `C` searches its own cards, its running best for `u`, call it `r`, can only go up, so
//! `C`'s final value `v` has `v[u] ≥ r`, hence `v[t] ≤ ONE − v[u] ≤ ONE − r`.
//! - `N` replaces its best only by a value strictly better for `t`, or by an equal value from a
//!   card earlier in the tie order. If `c` comes **after** `cb` in the tie order, `N` cannot pick
//!   `c` once `v[t] ≤ b`, which is guaranteed as soon as `r ≥ ONE − b`.
//! - If `c` comes **before** `cb` in the tie order (possible only when move ordering tried `cb`
//!   first), an equal value would be picked, so `C` may stop only once `v[t] < b` is
//!   guaranteed: `r > ONE − b`.
//!
//! When the condition holds, `C` stops and reports "not chosen"; `N`'s own value is unchanged,
//! so it is still exact. A stopped node's partial value is never stored in the table. Only the
//! node directly below `N` uses `N`'s bound (*shallow* pruning); the bound is not passed deeper.
//! When `u = t` (a trick won by the mover's partner or the mover leading again) no bound is
//! passed.
//!
//! In addition, a node stops as soon as its mover's team has a certain win (`ONE`): no card can
//! be strictly better, and every value with `ONE` for team `t` is the same vector (all other
//! entries are 0), so whichever tied card the tie rule would pick, the node's value is the same.
//! The root never stops early: it values every legal card exactly (the decision-time search
//! needs every card's value).
//!
//! # Deep and speculative pruning: not used
//! *Deep* pruning (a bound from a grandparent or higher) is unsound in max^n (Korf 1991): an
//! intermediate player's choice can change which value reaches the ancestor even when the
//! pruned branch could not be chosen by the ancestor itself. Sturtevant's *speculative* pruning
//! (2003; last-branch pruning is its special case) prunes speculatively and re-searches when the
//! speculation is proved wrong; its correctness proof assumes a fixed sum and a strict
//! best-reply rule, and depends on how ties between equal values are broken. With four teams,
//! pervasive ties (82–85% of decision nodes in the spike) and our position-only tie rule, the
//! re-search condition would also have to track which tied card each intermediate player keeps
//! (tie order versus the order the cards were tried). We have no published proof for that
//! setting, so per the plan ("prune only where provably safe") it is left out; shallow pruning
//! plus the certain-win stop is what is used.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use engine::cards::{Bits, Card, bit};
use engine::rules::{EffSuit, beats, effective_suit, plain_rank, trump_rank};
use engine::team_of;
use equity::{MatchState, MoneyChance, ONE};

use crate::equiv::Classes;
use crate::frontier::{CanonPos, FrontierCut, canonical};
use crate::tt::{SLOT_BYTES, SharedTable, Table, TtKey};
use crate::{Leaf, MatchCtx, Position, Solution, Stats, Step, TieOrder, ordered};

/// Switches of the optimised solver. Every combination gives identical answers (tested). On by
/// default: table, equivalent cards, pruning. Move ordering is off by default because it was
/// measured to cost time without saving nodes (docs/M2-SPIKE.md): shallow pruning rarely fires
/// in 4-team max^n, so trying good cards first has little to gain.
#[derive(Clone, Copy, Debug)]
pub struct FastOptions {
    /// Transposition table at trick boundaries.
    pub tt: bool,
    /// Memory cap of the table in megabytes.
    pub tt_mb: usize,
    /// Equivalent cards.
    pub equiv: bool,
    /// Move ordering (try likely good cards first).
    pub ordering: bool,
    /// Shallow pruning and the certain-win stop.
    pub prune: bool,
    /// The tie order (which of several equally good cards is reported).
    pub order: TieOrder,
    /// Give up after this many milliseconds (checked every 4096 nodes); `None` = no limit.
    pub time_cap_ms: Option<u64>,
}

impl Default for FastOptions {
    fn default() -> Self {
        FastOptions {
            tt: true,
            tt_mb: 256,
            equiv: true,
            ordering: false,
            prune: true,
            order: TieOrder::Ascending,
            time_cap_ms: None,
        }
    }
}

impl FastOptions {
    /// Every speed-up off (the search is then the naive one).
    pub fn none() -> Self {
        FastOptions {
            tt: false,
            equiv: false,
            ordering: false,
            prune: false,
            ..FastOptions::default()
        }
    }

    /// The 16 combinations of the four switches, `bits` 0..16 (bit 0 table, 1 equivalent cards,
    /// 2 ordering, 3 pruning), other settings from `self`.
    pub fn combo(self, bits: u8) -> Self {
        FastOptions {
            tt: bits & 1 != 0,
            equiv: bits & 2 != 0,
            ordering: bits & 4 != 0,
            prune: bits & 8 != 0,
            ..self
        }
    }
}

/// The bound a parent passes to the node directly below it (shallow pruning).
#[derive(Clone, Copy)]
struct Cut {
    /// The parent mover's team.
    team: u8,
    /// `ONE − b`, where `b` is the parent's best so far for its team.
    bound: u64,
    /// Whether the card leading here comes before the parent's best card in the tie order.
    strict: bool,
}

enum Res {
    Exact(MoneyChance),
    /// Stopped by the parent's bound: cannot be chosen by the parent.
    Cut,
    /// Time cap hit.
    Abort,
}

/// The transposition table a search uses: none, its own, or one shared by the threads of a
/// parallel solve.
enum Tt<'a> {
    Off,
    Own(&'a mut Table),
    Shared(&'a SharedTable),
}

/// Memory kept between exact solves, normally one per thread for a whole run: the
/// transposition tables of [`solve_fast_with`] and [`solve_fast_par_with`].
///
/// Why: a solve's table grows to as much as `tt_mb` megabytes. Allocating it for every solve
/// (and the operating system zeroing each new page on first touch, then unmapping it on free)
/// is cheap with a few threads, but with hundreds of threads and short solves the threads queue
/// on the process's memory map and sit idle. With a `SolveMem` each table is allocated once per
/// thread and emptied in O(1) between solves (see the `tt` module).
///
/// Reuse never changes a result: every solve starts from an empty table of the same starting
/// size, grown by the same rule, with slot-for-slot the contents a new table would have, so the
/// answer and every counter equal those of [`solve_fast`] / [`solve_fast_par`] (tested).
#[derive(Default)]
pub struct SolveMem {
    own: Option<Table>,
    shared: Option<SharedTable>,
}

impl SolveMem {
    /// Empty memory (tables are allocated by the first solve that needs them).
    pub fn new() -> SolveMem {
        SolveMem::default()
    }

    /// The sequential solve's table, emptied, capped at `mb` megabytes.
    fn own(&mut self, mb: usize) -> &mut Table {
        let t = self.own.get_or_insert_with(|| Table::new(mb));
        t.reset(mb << 20);
        t
    }

    /// The parallel solve's shared table, emptied, of `mb` megabytes for `threads` threads.
    fn shared(&mut self, mb: usize, threads: usize) -> &SharedTable {
        let t = self
            .shared
            .get_or_insert_with(|| SharedTable::new(mb, threads));
        t.reset(mb, threads);
        t
    }
}

impl Tt<'_> {
    fn on(&self) -> bool {
        !matches!(self, Tt::Off)
    }

    fn get(&self, k: &TtKey) -> Option<MoneyChance> {
        match self {
            Tt::Off => None,
            Tt::Own(t) => t.get(k),
            Tt::Shared(t) => t.get(k),
        }
    }

    fn put(&mut self, k: TtKey, v: MoneyChance) {
        match self {
            Tt::Off => {}
            Tt::Own(t) => t.put(k, v),
            Tt::Shared(t) => t.put(k, v),
        }
    }
}

/// Live counters of a parallel solve, readable from another thread while it runs (for progress
/// lines). They are diagnostics only and never affect an answer.
#[derive(Debug, Default)]
pub struct ParProgress {
    /// Nodes searched so far (updated every 4096 nodes per thread).
    pub nodes: AtomicU64,
    /// Entries in the shared table.
    pub tt_entries: AtomicU64,
    /// Subproblems finished, and their total (set once the split is made).
    pub tasks_done: AtomicU64,
    /// Total subproblems.
    pub tasks_total: AtomicU64,
    /// Set to stop the solve early (it then reports a timeout); also set by the time cap.
    pub abort: AtomicBool,
    /// Threads searching for this progress's solves right now: a sequential solve counts 1 while
    /// it runs; a parallel solve counts its splitting thread while it splits and combines, and
    /// each worker while it still takes subproblems (a worker that found none left no longer
    /// counts, so a parallel solve's tail shows here). Read by the trick-1 bench's busy-thread
    /// samples.
    pub active: AtomicU64,
}

/// Counts one thread in [`ParProgress::active`] while alive.
struct Active<'a>(Option<&'a ParProgress>);

impl<'a> Active<'a> {
    fn enter(p: Option<&'a ParProgress>) -> Self {
        if let Some(p) = p {
            p.active.fetch_add(1, Ordering::Relaxed);
        }
        Active(p)
    }
}

impl Drop for Active<'_> {
    fn drop(&mut self) {
        if let Some(p) = self.0 {
            p.active.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// The leaf values of one solve, remembered. The match state around the hand (games won, the
/// dealer's team) is fixed during a solve, so a game-won leaf depends only on the winning team
/// and a hand-end leaf only on the four scores: each is taken from the leaf evaluator (the
/// equity table: validate, rotate to canonical form, index, rotate back) once, then read from
/// here. The hand-end memo is direct-mapped with [`LEAF_SLOTS`] entries keyed by the packed
/// scores (bounded: 40 KB, allocated on the first hand-end leaf; a slot holding other scores is
/// overwritten), and the whole key is compared on every read, so a hit is exactly the
/// evaluator's value for those scores.
struct LeafCache {
    game: [Option<MoneyChance>; 4],
    hand: Option<Box<[(u64, MoneyChance)]>>,
}

/// Hand-end memo entries (a power of two).
const LEAF_SLOTS: usize = 1024;

/// The most frontier positions one batch gathers ([`Search::gather`]).
const GATHER_CAP: usize = 256;

/// Seats at the table.
const SEATS: usize = engine::SEATS;

impl LeafCache {
    fn new() -> LeafCache {
        LeafCache {
            game: [None; 4],
            hand: None,
        }
    }
}

struct Search<'a, L: Leaf> {
    ctx: MatchCtx,
    leaf: &'a L,
    opts: FastOptions,
    /// The hybrid frontier, if any (`crate::frontier`).
    cut: Option<FrontierCut<'a>>,
    classes: &'static Classes,
    leaves: LeafCache,
    tt: Tt<'a>,
    stats: Stats,
    start: Instant,
    aborted: bool,
    /// Live counters and the stop flag (parallel solves, and sequential solves given one).
    progress: Option<&'a ParProgress>,
}

/// The position of a card in the tie order (smaller = earlier).
fn canon(c: Card, order: TieOrder) -> u8 {
    match order {
        TieOrder::Ascending => c,
        TieOrder::Descending => 63 - c,
    }
}

/// A card's strength for ordering: trumps above plain cards.
fn strength(c: Card, p: &Position) -> u8 {
    let t = trump_rank(c, p.trump);
    if t > 0 { 20 + t } else { plain_rank(c) }
}

/// Move-ordering score (smaller is tried first). Leading: strongest card first. Following:
/// if the mover's partner is winning the trick, lowest card first; otherwise the cheapest card
/// that takes the lead in the trick, then the rest lowest first.
fn order_key(c: Card, p: &Position, winner: Option<(Card, u8, EffSuit)>, team: u8) -> u16 {
    let s = u16::from(strength(c, p));
    match winner {
        None => 64 - s,
        Some((w, wteam, led)) => {
            if wteam == team || beats(c, w, led, p.trump) {
                s
            } else {
                64 + s
            }
        }
    }
}

impl<L: Leaf> Search<'_, L> {
    fn game_won(&mut self, team: u8) -> MoneyChance {
        self.stats.leaves += 1;
        if let Some(v) = self.leaves.game[usize::from(team)] {
            return v;
        }
        let mut games = self.ctx.games;
        games[usize::from(team)] += 1;
        let v = if games[usize::from(team)] >= engine::game::GAMES_TO_WIN {
            MoneyChance::certain(team)
        } else {
            self.leaf.value(&MatchState {
                scores: [0; 4],
                games,
                dealer_team: (self.ctx.dealer_team + 1) % 4,
            })
        };
        self.leaves.game[usize::from(team)] = Some(v);
        v
    }

    fn hand_over(&mut self, scores: [u8; 4]) -> MoneyChance {
        self.stats.leaves += 1;
        // The key: the four scores, with a bit above them so that no key is zero (an empty
        // slot); the slot from a fixed mixing of the key.
        let key = u64::from(u32::from_le_bytes(scores)) | 1 << 32;
        let slot = ((key as u32).wrapping_mul(0x9E37_79B1) >> 22) as usize % LEAF_SLOTS;
        let memo = self.leaves.hand.get_or_insert_with(|| {
            vec![(0, MoneyChance::certain(0)); LEAF_SLOTS].into_boxed_slice()
        });
        if memo[slot].0 == key {
            return memo[slot].1;
        }
        let v = self.leaf.value(&MatchState {
            scores,
            games: self.ctx.games,
            dealer_team: (self.ctx.dealer_team + 1) % 4,
        });
        memo[slot] = (key, v);
        v
    }

    fn child(&mut self, p: &Position, c: Card, cut: Option<Cut>) -> Res {
        match p.play(c) {
            Step::Next(q) => self.node(&q, cut),
            Step::GameWon(t) => Res::Exact(self.game_won(t)),
            Step::HandOver(s) => Res::Exact(self.hand_over(s)),
        }
    }

    /// The cards to search at `p`, in the order to try them (at most 16).
    fn candidates(&mut self, p: &Position, out: &mut [Card; 16]) -> usize {
        self.candidates_of(p, p.legal(), out)
    }

    /// [`Search::candidates`] with the legal set already computed.
    fn candidates_of(&mut self, p: &Position, legal: Bits, out: &mut [Card; 16]) -> usize {
        let seat = p.to_act();
        let hand = p.hands[usize::from(seat)];
        debug_assert_eq!(legal, p.legal());
        debug_assert!(legal != 0, "the legal set is never empty");
        let set: Bits = if self.opts.equiv && legal.count_ones() >= 2 {
            let r = self
                .classes
                .representatives(hand, legal, p.live(), self.opts.order);
            self.stats.equiv_skipped += u64::from(legal.count_ones() - r.count_ones());
            r
        } else {
            legal
        };
        let mut n = 0;
        for c in ordered(set, self.opts.order) {
            out[n] = c;
            n += 1;
        }
        if self.opts.ordering && n >= 2 {
            let winner = (p.n_played > 0).then(|| {
                let w = crate::trick_winner_index(&p.trick[..usize::from(p.n_played)], p.trump);
                let wseat = (p.leader + w as u8) % 8;
                (
                    p.trick[w],
                    team_of(wseat),
                    effective_suit(p.trick[0], p.trump),
                )
            });
            let team = team_of(seat);
            let order = self.opts.order;
            out[..n].sort_by_key(|&c| (order_key(c, p, winner, team), canon(c, order)));
        }
        n
    }

    /// Forced play inside a trick: from `p`, play every card that is the only legal one straight
    /// away, without a node. Returns the first position with a choice, or the next trick
    /// boundary (where the table is consulted), or the leaf value when the forced plays end
    /// the hand.
    ///
    /// Why this changes no answer. A node with one legal card has exactly the value of its one
    /// child (the backup rule picks the only card), so a chain of such nodes has the value of
    /// the position at its end, and the search reports that value either way. The parent's
    /// pruning bound is checked at the end of the chain instead of at its start: shallow
    /// pruning needs only that the node checking the bound has the value the parent compares,
    /// which the chain's end has; and a node reporting `Cut` or `Exact` leads the parent to the
    /// same choice (a value that meets the cut condition is never chosen). Node counts change;
    /// values do not.
    fn forced(&mut self, p: &Position, legal: Bits) -> Result<(Position, Bits), MoneyChance> {
        debug_assert!(p.n_played > 0 && legal.count_ones() == 1);
        let mut q = *p;
        let mut legal = legal;
        loop {
            self.stats.forced += 1;
            match q.play(legal.trailing_zeros() as Card) {
                Step::Next(r) => q = r,
                Step::GameWon(t) => return Err(self.game_won(t)),
                Step::HandOver(s) => return Err(self.hand_over(s)),
            }
            if q.burn_choice().is_some() {
                // A pending robber's burn is a choice: stop here (its legal set is unused).
                return Ok((q, 0));
            }
            legal = q.legal();
            if q.n_played == 0 || legal.count_ones() != 1 {
                return Ok((q, legal));
            }
        }
    }

    /// The value of a last-trick boundary `p` (every hand holds one card, so the whole trick is
    /// forced): the trick's winner, the bonuses and the leaf, computed directly, with no node,
    /// table probe or table store per card. Called after the boundary's own table probe missed;
    /// the caller stores the value as it would the searched value, which this equals: the node's
    /// only line is this one, so the search would have reached exactly this leaf (and the table
    /// holds only exact values, so a hit could only have returned it too). Node counts change;
    /// values do not.
    fn last_trick(&mut self, p: &Position) -> MoneyChance {
        debug_assert_eq!(p.n_played, 0);
        debug_assert_eq!(p.hand.tricks_done, engine::game::TRICKS_PER_HAND - 1);
        let mut q = *p;
        loop {
            let only = q.hands[usize::from(q.to_act())];
            debug_assert_eq!(only.count_ones(), 1, "one card per hand in the last trick");
            self.stats.forced += 1;
            match q.play(only.trailing_zeros() as Card) {
                Step::Next(r) => q = r,
                Step::GameWon(t) => return self.game_won(t),
                Step::HandOver(s) => return self.hand_over(s),
            }
        }
    }

    /// The hybrid's batch (docs/ENGINEERING-2026-10-17.md, item B): at the first node of a line
    /// within `d` plays of the frontier (`d` = the evaluator's [`FrontierCut::prefetch_depth`]),
    /// hand the evaluator every frontier position reachable from it ([`Search::gather`]) so it
    /// can value them as one batch. `arrival` is the position the line reached the node with,
    /// `end` the node's position after its forced plays (the root: both the root). The node is the
    /// first within `d` when `end` is within `d` plays and `arrival` is at least `d` away (the
    /// parent's own position, one play before `arrival`, was then more than `d` away): exactly
    /// one node of each line gathers, and its gathering covers every frontier position below it.
    ///
    /// It changes nothing the search does: no counter, no table entry, no order of visiting;
    /// every frontier position the search reaches is still asked for through the evaluator, at
    /// the same point and in the same order as without the batch (the evaluator answers from
    /// what it kept), with the same value, the evaluator being a pure function of the canonical
    /// form.
    fn prefetch_ahead(&self, arrival: &Position, end: &Position, root: bool) {
        let Some(cut) = self.cut else { return };
        let d = i32::from(cut.prefetch_depth().min(SEATS as u8 - 1));
        if d == 0 || arrival.rob.is_some() {
            return;
        }
        // Plays from a position to the frontier (0 at the frontier).
        let left = |p: &Position| {
            SEATS as i32 * (i32::from(cut.trick()) - 1 - i32::from(p.hand.tricks_done))
                - i32::from(p.n_played)
        };
        let (a, e) = (left(arrival), left(end));
        if e > d || a < 1 || (a < d && !root) {
            return;
        }
        let mut out = Vec::new();
        self.gather(arrival, cut, &mut out);
        if !out.is_empty() {
            cut.prefetch(&out, &self.ctx);
        }
    }

    /// The canonical forms of the frontier positions the search reaches from `p` (inside the
    /// trick before the frontier's) when nothing stops it early: depth first through the same
    /// cards the search plays (each group of equivalent cards through its first card, as
    /// [`Search::candidates_of`]; forced cards included), at most [`GATHER_CAP`]. Counts nothing.
    fn gather(&self, p: &Position, cut: FrontierCut, out: &mut Vec<CanonPos>) {
        let legal = p.legal();
        let set = if self.opts.equiv && legal.count_ones() >= 2 {
            let hand = p.hands[usize::from(p.to_act())];
            self.classes
                .representatives(hand, legal, p.live(), self.opts.order)
        } else {
            legal
        };
        for c in ordered(set, self.opts.order) {
            if out.len() >= GATHER_CAP {
                return;
            }
            if let Step::Next(q) = p.play(c) {
                if q.n_played == 0 {
                    if cut.applies(&q) {
                        out.push(canonical(&q));
                    }
                } else if q.burn_choice().is_none() {
                    self.gather(&q, cut, out);
                }
            }
        }
    }

    fn node(&mut self, p: &Position, cut: Option<Cut>) -> Res {
        self.stats.nodes += 1;
        self.stats.nodes_by_trick[usize::from(p.hand.tricks_done.min(4))] += 1;
        if self.stats.nodes & 0xFFF == 0 {
            if let Some(cap) = self.opts.time_cap_ms
                && self.start.elapsed().as_millis() as u64 >= cap
            {
                self.aborted = true;
            }
            if let Some(pr) = self.progress {
                pr.nodes.fetch_add(0x1000, Ordering::Relaxed);
                let entries = match &self.tt {
                    Tt::Off => 0,
                    Tt::Own(t) => t.entries,
                    Tt::Shared(t) => t.entries(),
                };
                pr.tt_entries.store(entries, Ordering::Relaxed);
                if self.aborted {
                    pr.abort.store(true, Ordering::Relaxed);
                }
                if pr.abort.load(Ordering::Relaxed) {
                    self.aborted = true;
                }
            }
        }
        if self.aborted {
            return Res::Abort;
        }
        if p.burn_choice().is_some() {
            return self.burn_node(p);
        }
        let mut legal = p.legal();
        let chained;
        let arrival = p;
        let p = if p.n_played > 0 && legal.count_ones() == 1 {
            match self.forced(p, legal) {
                Ok((q, l)) => {
                    chained = q;
                    legal = l;
                    &chained
                }
                Err(v) => return Res::Exact(v),
            }
        } else {
            p
        };
        if p.burn_choice().is_some() {
            return self.burn_node(p);
        }
        self.prefetch_ahead(arrival, p, false);
        let key = (self.tt.on() && p.n_played == 0 && p.rob.is_none()).then(|| TtKey::of(p));
        if let Some(k) = &key {
            self.stats.tt_probes += 1;
            if let Some(v) = self.tt.get(k) {
                self.stats.tt_hits += 1;
                return Res::Exact(v);
            }
        }
        if let Some(cut) = self.cut
            && cut.applies(p)
        {
            // A frontier position: the evaluator's value, stored in the table like a searched
            // value (it is a function of the key within this solve: see `crate::frontier`).
            self.stats.frontier += 1;
            let v = cut.value(p, &self.ctx);
            if let Some(k) = key {
                self.tt.put(k, v);
            }
            return Res::Exact(v);
        }
        if p.n_played == 0 && p.hand.tricks_done == engine::game::TRICKS_PER_HAND - 1 {
            let v = self.last_trick(p);
            if let Some(k) = key {
                self.tt.put(k, v);
            }
            return Res::Exact(v);
        }
        let team = team_of(p.to_act());
        // The parent's bound applies only when the parent's team differs from ours.
        let cut = cut.filter(|c| c.team != team);
        let mut cards = [0 as Card; 16];
        let n = self.candidates_of(p, legal, &mut cards);
        let mut best: Option<(MoneyChance, u8)> = None;
        for &c in &cards[..n] {
            let cc = canon(c, self.opts.order);
            let child_cut = match best {
                Some((b, bc)) if self.opts.prune => Some(Cut {
                    team,
                    bound: ONE - b.get(team),
                    strict: cc < bc,
                }),
                _ => None,
            };
            let v = match self.child(p, c, child_cut) {
                Res::Exact(v) => v,
                Res::Cut => {
                    self.stats.cutoffs += 1;
                    continue;
                }
                Res::Abort => return Res::Abort,
            };
            let better = match best {
                None => true,
                Some((b, bc)) => {
                    v.get(team) > b.get(team) || (v.get(team) == b.get(team) && cc < bc)
                }
            };
            if better {
                best = Some((v, cc));
            }
            if self.opts.prune {
                let r = best.map_or(0, |(b, _)| b.get(team));
                if let Some(k) = cut
                    && (r > k.bound || (!k.strict && r == k.bound))
                {
                    return Res::Cut;
                }
                if r == ONE {
                    break;
                }
            }
        }
        let Some((v, _)) = best else {
            // Every searched card was cut: impossible, the first card is searched without a
            // bound.
            unreachable!("the first card is always searched exactly");
        };
        if let Some(k) = key {
            self.tt.put(k, v);
        }
        Res::Exact(v)
    }

    /// A pending robber's burn (`p.burn_choice()` is `Some`): the naive solver's burn node with
    /// equivalent burns merged. Each burn group is searched exactly (no bound) through its first
    /// card in the tie order, and the robber's team's best is kept, the first strictly best in
    /// the tie order among equals, exactly as the naive solver. Merging is sound: two cards of
    /// one equivalence group of the robber's hand (no live card between them in rank, no card
    /// of that hand between them in the tie order) map onto each other by swapping them, so
    /// burning either leaves positions of the same value (vector included); the group's first
    /// card is the one the naive order would keep. The
    /// ace of trumps may share a group with cards that may be burned; it is simply left out.
    /// Values are exact, so the parent's bound is not needed (an exact value is always a valid
    /// answer to a bounded call).
    fn burn_node(&mut self, p: &Position) -> Res {
        let (h, burns) = p.burn_choice().expect("a burn is due");
        let team = team_of(p.to_act());
        let order = self.opts.order;
        let set = if self.opts.equiv {
            let mut q = *p;
            q.hands[usize::from(p.to_act())] = h;
            let mut g = [0 as Bits; 16];
            let n = self.classes.groups(h, q.live(), &mut g);
            let mut reps: Bits = 0;
            for &grp in &g[..n] {
                let l = grp & burns;
                if l != 0 {
                    reps |= crate::equiv::first_in_order(l, order);
                }
            }
            self.stats.equiv_skipped += u64::from(burns.count_ones() - reps.count_ones());
            reps
        } else {
            burns
        };
        let mut best: Option<MoneyChance> = None;
        for b in ordered(set, order) {
            let v = match self.node(&p.burnt(b), None) {
                Res::Exact(v) => v,
                Res::Cut => unreachable!("a burn's child gets no bound"),
                Res::Abort => return Res::Abort,
            };
            if best.is_none_or(|x| v.get(team) > x.get(team)) {
                best = Some(v);
            }
        }
        if burns.count_ones() >= 2 {
            self.stats.decision_nodes += 1;
        }
        Res::Exact(best.expect("a robber can always burn"))
    }
}

/// Solve a position exactly with the speed-ups switched on in `opts`. The result equals
/// [`crate::solve`]'s (value, chosen card, every root card's value) under the same tie order;
/// the counters differ. The table is allocated for this solve; to solve many positions, keep a
/// [`SolveMem`] per thread and use [`solve_fast_with`] (identical results).
pub fn solve_fast<L: Leaf>(pos: &Position, ctx: MatchCtx, leaf: &L, opts: FastOptions) -> Solution {
    solve_fast_with(pos, ctx, leaf, opts, &mut SolveMem::new())
}

/// [`solve_fast`] with its table taken from `mem` (kept for the next solve). The result, every
/// counter included, is exactly [`solve_fast`]'s, whatever `mem` was used for before.
pub fn solve_fast_with<L: Leaf>(
    pos: &Position,
    ctx: MatchCtx,
    leaf: &L,
    opts: FastOptions,
    mem: &mut SolveMem,
) -> Solution {
    solve_fast_watched(pos, ctx, leaf, opts, mem, None)
}

/// [`solve_fast_with`] that also keeps `progress`'s node and table counters up to date and stops
/// (reporting a timeout) once its `abort` flag is set, as [`solve_fast_par`] does. Everything
/// else, the answer and every counter included, is exactly [`solve_fast_with`]'s.
pub fn solve_fast_watched<L: Leaf>(
    pos: &Position,
    ctx: MatchCtx,
    leaf: &L,
    opts: FastOptions,
    mem: &mut SolveMem,
    progress: Option<&ParProgress>,
) -> Solution {
    solve_fast_cut(pos, ctx, leaf, opts, None, mem, progress)
}

/// [`solve_fast_watched`] with an optional frontier (the hybrid H_k, docs/M3-HYBRID-SPEC.md
/// §2): every position reached at the start of the frontier's trick is valued by its evaluator
/// instead of being searched. The result equals [`crate::solve_cut`]'s with the same frontier
/// (value, chosen card, every root card's value) under the same tie order, for every switch
/// combination; with `None` it is exactly [`solve_fast_watched`]'s.
pub fn solve_fast_cut<L: Leaf>(
    pos: &Position,
    ctx: MatchCtx,
    leaf: &L,
    opts: FastOptions,
    cut: Option<FrontierCut>,
    mem: &mut SolveMem,
    progress: Option<&ParProgress>,
) -> Solution {
    let _active = Active::enter(progress);
    let mut s = Search {
        ctx,
        leaf,
        opts,
        cut,
        classes: Classes::shared(pos.trump),
        leaves: LeafCache::new(),
        tt: if opts.tt {
            Tt::Own(mem.own(opts.tt_mb))
        } else {
            Tt::Off
        },
        stats: Stats::default(),
        start: Instant::now(),
        aborted: false,
        progress,
    };
    s.stats.nodes += 1;
    assert!(
        pos.burn_choice().is_none(),
        "the root mover's own burn is decided outside the solve"
    );
    let mut values: Vec<(Card, MoneyChance)> = Vec::new();
    s.prefetch_ahead(pos, pos, true);
    for l in root_groups(pos, s.classes, opts) {
        let rep = ordered(l, opts.order).next().expect("non-empty");
        match s.child(pos, rep, None) {
            Res::Exact(v) => {
                for c in ordered(l, opts.order) {
                    values.push((c, v));
                }
                s.stats.equiv_skipped += u64::from(l.count_ones() - 1);
            }
            Res::Cut => unreachable!("the root passes no bound"),
            Res::Abort => {
                if let Tt::Own(t) = &s.tt {
                    s.stats.tt_entries = t.entries;
                }
                return timed_out(s.stats);
            }
        }
    }
    if let Tt::Own(t) = &s.tt {
        s.stats.tt_entries = t.entries;
        s.stats.tt_replaced = t.replaced;
        s.stats.tt_bytes = t.bytes() as u64;
    }
    finish(values, team_of(pos.to_act()), opts.order, s.stats)
}

fn timed_out(stats: Stats) -> Solution {
    Solution {
        value: None,
        best: None,
        children: Vec::new(),
        root_tied: 0,
        stats,
        timed_out: true,
    }
}

/// The root's legal cards in groups of equivalent cards (one card per group without the
/// equivalent-card switch), each group searched once through its first card in the tie order.
fn root_groups(pos: &Position, classes: &Classes, opts: FastOptions) -> Vec<Bits> {
    let seat = pos.to_act();
    let hand = pos.hands[usize::from(seat)];
    let legal = pos.legal();
    let mut groups = [0 as Bits; 16];
    let ng = if opts.equiv {
        classes.groups(hand, pos.live(), &mut groups)
    } else {
        let mut n = 0;
        for c in ordered(hand, opts.order) {
            groups[n] = bit(c);
            n += 1;
        }
        n
    };
    groups[..ng]
        .iter()
        .map(|&g| g & legal)
        .filter(|&l| l != 0)
        .collect()
}

/// The root decision: the best value for the root mover's team, the first card in the tie order
/// among equals.
fn finish(
    mut values: Vec<(Card, MoneyChance)>,
    team: u8,
    order: TieOrder,
    stats: Stats,
) -> Solution {
    values.sort_by_key(|&(c, _)| canon(c, order));
    let mut best = 0usize;
    for (i, (_, v)) in values.iter().enumerate() {
        if v.get(team) > values[best].1.get(team) {
            best = i;
        }
    }
    let top = values[best].1.get(team);
    let root_tied = values.iter().filter(|(_, v)| v.get(team) == top).count() as u32;
    Solution {
        value: Some(values[best].1),
        best: Some(values[best].0),
        children: values,
        root_tied,
        stats,
        timed_out: false,
    }
}

/// A node of the top of the tree that a parallel solve splits off.
enum PNode {
    /// A finished line (the game or hand ended).
    Done(MoneyChance),
    /// A subproblem, solved by one thread.
    Task(Position),
    /// Expanded here: the mover's team and each searched card (tie-order position, node).
    Inner { team: u8, kids: Vec<(u8, usize)> },
}

/// Subproblems per thread the split aims for (more than one so that threads that finish early
/// take more work: subtrees differ in size by orders of magnitude).
pub const PAR_TASKS_PER_THREAD: usize = 16;

/// Expand the open position `q` by one card `c`: a finished line is valued now, an open
/// position joins the frontier. Returns the new node's index.
fn par_push<L: Leaf>(
    nodes: &mut Vec<PNode>,
    frontier: &mut Vec<usize>,
    exp: &mut Search<'_, L>,
    q: &Position,
    c: Card,
) -> usize {
    let n = match q.play(c) {
        Step::Next(r) => match exp.cut {
            // A hybrid frontier position is valued here, never split or searched.
            Some(cut) if cut.applies(&r) => {
                exp.stats.frontier += 1;
                PNode::Done(cut.value(&r, &exp.ctx))
            }
            _ => {
                frontier.push(nodes.len());
                PNode::Task(r)
            }
        },
        Step::GameWon(t) => PNode::Done(exp.game_won(t)),
        Step::HandOver(sc) => PNode::Done(exp.hand_over(sc)),
    };
    nodes.push(n);
    nodes.len() - 1
}

fn unpoison<T>(r: std::sync::LockResult<T>) -> T {
    r.unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// [`solve_fast`] on `threads` threads, for one position. The result (value, chosen card, every
/// root card's value) is exactly [`solve_fast`]'s; only the counters and the time differ.
///
/// How: the top of the tree is expanded ply by ply, with the same cards per node as the
/// sequential search (equivalent cards and all), until there are at least
/// [`PAR_TASKS_PER_THREAD`] × `threads` open positions (or 16 plies). The open positions are
/// numbered in that expansion order; each thread repeatedly takes the next unsolved one and
/// solves it exactly with the sequential search (no bound from above, so it returns the exact
/// value). All threads share one transposition table ([`crate::tt::SharedTable`]), which only
/// ever holds exact values, so what it holds when changes the speed, never a value. The expanded
/// top is then valued bottom-up with the sequential rule: the best value for the mover's team,
/// the first card in the tie order among equals. Which cards are searched at a node does not
/// change its value (equivalent cards have equal values; pruning only skips cards that cannot be
/// chosen), so the top's values equal the sequential search's, and the root decides exactly as
/// [`solve_fast`] does.
///
/// `opts.tt_mb` is the shared table's total size. `progress`, if given, is updated while the
/// solve runs; setting its `abort` flag stops the solve, which then reports a timeout.
pub fn solve_fast_par<L: Leaf + Sync>(
    pos: &Position,
    ctx: MatchCtx,
    leaf: &L,
    opts: FastOptions,
    threads: usize,
    progress: Option<&ParProgress>,
) -> Solution {
    solve_fast_par_with(
        pos,
        ctx,
        leaf,
        opts,
        threads,
        progress,
        &mut SolveMem::new(),
    )
}

/// [`solve_fast_par`] with its shared table taken from `mem` (kept for the next solve; the
/// answer is the same whatever `mem` was used for before).
pub fn solve_fast_par_with<L: Leaf + Sync>(
    pos: &Position,
    ctx: MatchCtx,
    leaf: &L,
    opts: FastOptions,
    threads: usize,
    progress: Option<&ParProgress>,
    mem: &mut SolveMem,
) -> Solution {
    par_solve(
        pos,
        ctx,
        leaf,
        opts,
        threads,
        PAR_TASKS_PER_THREAD,
        progress,
        mem,
        None,
    )
}

/// [`solve_fast_par_with`] with an optional frontier (see [`solve_fast_cut`]): exactly
/// [`solve_fast_cut`]'s result (and so [`crate::solve_cut`]'s) for any thread count.
#[allow(clippy::too_many_arguments)]
pub fn solve_fast_par_cut<L: Leaf + Sync>(
    pos: &Position,
    ctx: MatchCtx,
    leaf: &L,
    opts: FastOptions,
    threads: usize,
    progress: Option<&ParProgress>,
    mem: &mut SolveMem,
    cut: Option<FrontierCut>,
) -> Solution {
    par_solve(
        pos,
        ctx,
        leaf,
        opts,
        threads,
        PAR_TASKS_PER_THREAD,
        progress,
        mem,
        cut,
    )
}

/// [`solve_fast_par`] aiming for `tasks_per_thread` subproblems per thread instead of
/// [`PAR_TASKS_PER_THREAD`] (a finer split balances the load better; the answer is the same).
pub fn solve_fast_par_split<L: Leaf + Sync>(
    pos: &Position,
    ctx: MatchCtx,
    leaf: &L,
    opts: FastOptions,
    threads: usize,
    tasks_per_thread: usize,
    progress: Option<&ParProgress>,
) -> Solution {
    par_solve(
        pos,
        ctx,
        leaf,
        opts,
        threads,
        tasks_per_thread,
        progress,
        &mut SolveMem::new(),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn par_solve<L: Leaf + Sync>(
    pos: &Position,
    ctx: MatchCtx,
    leaf: &L,
    opts: FastOptions,
    threads: usize,
    tasks_per_thread: usize,
    progress: Option<&ParProgress>,
    mem: &mut SolveMem,
    cut: Option<FrontierCut>,
) -> Solution {
    let threads = threads.max(1);
    let own = ParProgress::default();
    let pr = progress.unwrap_or(&own);
    // This (splitting) thread counts while it splits and combines, not while the workers run.
    let splitter = Active::enter(Some(pr));
    let start = Instant::now();
    let shared = mem.shared(if opts.tt { opts.tt_mb } else { 1 }, threads);
    let mut exp = Search {
        ctx,
        leaf,
        opts,
        cut,
        classes: Classes::shared(pos.trump),
        leaves: LeafCache::new(),
        tt: Tt::Off,
        stats: Stats::default(),
        start,
        aborted: false,
        progress: None,
    };
    exp.stats.nodes += 1;
    assert!(
        pos.burn_choice().is_none(),
        "the root mover's own burn is decided outside the solve"
    );
    let mut nodes: Vec<PNode> = Vec::new();
    let mut frontier: Vec<usize> = Vec::new();
    // The root's groups of equivalent legal cards, each searched through its first card.
    let mut root: Vec<(Bits, usize)> = Vec::new();
    for l in root_groups(pos, exp.classes, opts) {
        let rep = ordered(l, opts.order).next().expect("non-empty");
        let i = par_push(&mut nodes, &mut frontier, &mut exp, pos, rep);
        root.push((l, i));
        exp.stats.equiv_skipped += u64::from(l.count_ones() - 1);
    }
    let target = tasks_per_thread.max(1) * threads;
    let mut depth = 1;
    while !frontier.is_empty() && frontier.len() < target && depth < 16 {
        depth += 1;
        for i in core::mem::take(&mut frontier) {
            let PNode::Task(q) = nodes[i] else {
                unreachable!("the frontier holds open positions only")
            };
            if q.burn_choice().is_some() {
                // A pending robber's burn stays one task (the workers' search decides it).
                frontier.push(i);
                continue;
            }
            exp.stats.nodes += 1;
            let team = team_of(q.to_act());
            let mut cards = [0 as Card; 16];
            let n = exp.candidates(&q, &mut cards);
            let mut kids = Vec::with_capacity(n);
            for &c in &cards[..n] {
                let k = par_push(&mut nodes, &mut frontier, &mut exp, &q, c);
                kids.push((canon(c, opts.order), k));
            }
            nodes[i] = PNode::Inner { team, kids };
        }
    }
    let tasks: Vec<(usize, Position)> = frontier
        .iter()
        .map(|&i| match nodes[i] {
            PNode::Task(q) => (i, q),
            _ => unreachable!("the frontier holds open positions only"),
        })
        .collect();
    pr.tasks_total.store(tasks.len() as u64, Ordering::Relaxed);
    let next = AtomicUsize::new(0);
    let results: Vec<Mutex<Option<MoneyChance>>> =
        (0..tasks.len()).map(|_| Mutex::new(None)).collect();
    let workers = threads.min(tasks.len()).max(1);
    let mut stats = exp.stats;
    drop(splitter);
    let worker_stats: Vec<Stats> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..workers)
            .map(|_| {
                let (tasks, next, results) = (&tasks, &next, &results);
                sc.spawn(move || {
                    let _active = Active::enter(Some(pr));
                    let mut s = Search {
                        ctx,
                        leaf,
                        opts,
                        cut,
                        classes: Classes::shared(pos.trump),
                        leaves: LeafCache::new(),
                        tt: if opts.tt { Tt::Shared(shared) } else { Tt::Off },
                        stats: Stats::default(),
                        start,
                        aborted: false,
                        progress: Some(pr),
                    };
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        if k >= tasks.len() || pr.abort.load(Ordering::Relaxed) {
                            break;
                        }
                        match s.node(&tasks[k].1, None) {
                            Res::Exact(v) => {
                                *unpoison(results[k].lock()) = Some(v);
                                pr.tasks_done.fetch_add(1, Ordering::Relaxed);
                            }
                            Res::Cut => unreachable!("a subproblem gets no bound"),
                            Res::Abort => {
                                pr.abort.store(true, Ordering::Relaxed);
                                break;
                            }
                        }
                    }
                    s.stats
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().expect("solver thread panicked"))
            .collect()
    });
    let _combiner = Active::enter(Some(pr));
    for w in &worker_stats {
        stats.nodes += w.nodes;
        stats.leaves += w.leaves;
        stats.decision_nodes += w.decision_nodes;
        stats.tied_nodes += w.tied_nodes;
        stats.tt_probes += w.tt_probes;
        stats.tt_hits += w.tt_hits;
        stats.cutoffs += w.cutoffs;
        stats.equiv_skipped += w.equiv_skipped;
        stats.forced += w.forced;
        stats.frontier += w.frontier;
        for (a, b) in stats.nodes_by_trick.iter_mut().zip(&w.nodes_by_trick) {
            *a += b;
        }
    }
    if opts.tt {
        stats.tt_entries = shared.entries();
        stats.tt_replaced = shared.replaced();
        stats.tt_bytes = shared.bytes() as u64;
    }
    let mut val: Vec<Option<MoneyChance>> = nodes
        .iter()
        .map(|n| match n {
            PNode::Done(v) => Some(*v),
            _ => None,
        })
        .collect();
    for ((i, _), r) in tasks.iter().zip(results) {
        val[*i] = unpoison(r.into_inner());
        if val[*i].is_none() {
            return timed_out(stats);
        }
    }
    // Children always come after their parent in the expansion order.
    for i in (0..nodes.len()).rev() {
        if let PNode::Inner { team, kids } = &nodes[i] {
            let t = *team;
            let mut best: Option<(MoneyChance, u8)> = None;
            for &(cc, k) in kids {
                let v = val[k].expect("children are valued before parents");
                let better = match best {
                    None => true,
                    Some((b, bc)) => v.get(t) > b.get(t) || (v.get(t) == b.get(t) && cc < bc),
                };
                if better {
                    best = Some((v, cc));
                }
            }
            val[i] = best.map(|(v, _)| v);
        }
    }
    let mut values: Vec<(Card, MoneyChance)> = Vec::new();
    for (l, i) in root {
        let v = val[i].expect("every root card is valued");
        for c in ordered(l, opts.order) {
            values.push((c, v));
        }
    }
    finish(values, team_of(pos.to_act()), opts.order, stats)
}

/// Bytes per transposition slot of the optimised solver.
pub const FAST_TT_SLOT_BYTES: usize = SLOT_BYTES;
