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

use std::time::Instant;

use engine::cards::{Bits, Card, bit};
use engine::rules::{EffSuit, beats, effective_suit, plain_rank, trump_rank};
use engine::team_of;
use equity::{MatchState, MoneyChance, ONE};

use crate::equiv::Classes;
use crate::tt::{SLOT_BYTES, Table, TtKey};
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

struct Search<'a, L: Leaf> {
    ctx: MatchCtx,
    leaf: &'a L,
    opts: FastOptions,
    classes: Classes,
    tt: Option<Table>,
    stats: Stats,
    start: Instant,
    aborted: bool,
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

    fn child(&mut self, p: &Position, c: Card, cut: Option<Cut>) -> Res {
        match p.play(c) {
            Step::Next(q) => self.node(&q, cut),
            Step::GameWon(t) => Res::Exact(self.game_won(t)),
            Step::HandOver(s) => Res::Exact(self.hand_over(s)),
        }
    }

    /// The cards to search at `p`, in the order to try them (at most 16).
    fn candidates(&mut self, p: &Position, out: &mut [Card; 16]) -> usize {
        let seat = p.to_act();
        let hand = p.hands[usize::from(seat)];
        let legal = p.legal();
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

    fn node(&mut self, p: &Position, cut: Option<Cut>) -> Res {
        self.stats.nodes += 1;
        if self.stats.nodes & 0xFFF == 0
            && let Some(cap) = self.opts.time_cap_ms
            && self.start.elapsed().as_millis() as u64 >= cap
        {
            self.aborted = true;
        }
        if self.aborted {
            return Res::Abort;
        }
        let key = (self.tt.is_some() && p.n_played == 0).then(|| TtKey::of(p));
        if let (Some(k), Some(tt)) = (&key, &self.tt) {
            self.stats.tt_probes += 1;
            if let Some(v) = tt.get(k) {
                self.stats.tt_hits += 1;
                return Res::Exact(v);
            }
        }
        let team = team_of(p.to_act());
        // The parent's bound applies only when the parent's team differs from ours.
        let cut = cut.filter(|c| c.team != team);
        let mut cards = [0 as Card; 16];
        let n = self.candidates(p, &mut cards);
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
        if let (Some(k), Some(tt)) = (key, &mut self.tt) {
            tt.put(k, v);
        }
        Res::Exact(v)
    }
}

/// Solve a position exactly with the speed-ups switched on in `opts`. The result equals
/// [`crate::solve`]'s (value, chosen card, every root card's value) under the same tie order;
/// the counters differ.
pub fn solve_fast<L: Leaf>(pos: &Position, ctx: MatchCtx, leaf: &L, opts: FastOptions) -> Solution {
    let mut s = Search {
        ctx,
        leaf,
        opts,
        classes: Classes::new(pos.trump),
        tt: opts.tt.then(|| Table::new(opts.tt_mb)),
        stats: Stats::default(),
        start: Instant::now(),
        aborted: false,
    };
    s.stats.nodes += 1;
    let seat = pos.to_act();
    let team = team_of(seat);
    let hand = pos.hands[usize::from(seat)];
    let legal = pos.legal();
    // Root: every legal card gets its exact value; equivalent cards share one search.
    let mut groups = [0 as Bits; 16];
    let ng = if opts.equiv {
        s.classes.groups(hand, pos.live(), &mut groups)
    } else {
        let mut n = 0;
        for c in ordered(hand, opts.order) {
            groups[n] = bit(c);
            n += 1;
        }
        n
    };
    let mut values: Vec<(Card, MoneyChance)> = Vec::new();
    for &g in &groups[..ng] {
        let l = g & legal;
        if l == 0 {
            continue;
        }
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
                s.stats.tt_entries = s.tt.as_ref().map_or(0, |t| t.entries);
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
    values.sort_by_key(|&(c, _)| canon(c, opts.order));
    let mut best = 0usize;
    for (i, (_, v)) in values.iter().enumerate() {
        if v.get(team) > values[best].1.get(team) {
            best = i;
        }
    }
    let top = values[best].1.get(team);
    let root_tied = values.iter().filter(|(_, v)| v.get(team) == top).count() as u32;
    if let Some(t) = &s.tt {
        s.stats.tt_entries = t.entries;
        s.stats.tt_replaced = t.replaced;
        s.stats.tt_bytes = t.bytes() as u64;
    }
    Solution {
        value: Some(values[best].1),
        best: Some(values[best].0),
        children: values,
        root_tied,
        stats: s.stats,
        timed_out: false,
    }
}

/// Bytes per transposition slot of the optimised solver.
pub const FAST_TT_SLOT_BYTES: usize = SLOT_BYTES;
