//! The hybrid cut (docs/M3-HYBRID-SPEC.md §2, §6): an optional *frontier* at the start of trick
//! `k` (no card of trick `k` played yet) where the solver stops searching and takes an
//! evaluator's money chance instead, normally the oracle network O_k. The search above the
//! frontier is unchanged (same max^n backup, same tie rule, same leaf rule when the hand or the
//! game ends before trick `k`). With no frontier configured nothing changes at all.
//!
//! # The evaluator sees the canonical form
//! An evaluator is never shown the raw [`Position`]. It is shown [`CanonPos`]: the position at a
//! trick boundary reduced to exactly what its exact value depends on. Two reasons:
//!
//! 1. **The optimised solver must stay exact.** It searches only one card of a group of
//!    equivalent cards (`crate::equiv`: same hand, same class, no live card between them in
//!    rank, no card of the hand between them in card index) and gives the others its value.
//!    Under exact search that is sound. With a frontier, playing `a` or `b` from such a group
//!    reaches frontier positions that differ: one line still holds `b` (and `a` is gone), the
//!    other holds `a` (and `b` is gone), and the best trump so far may differ too. An evaluator
//!    reading card identities could value the two differently, and the optimised solver would
//!    no longer return the naive solver's values. Both have the same canonical form (proof
//!    below), so for *any* evaluator of the canonical form merging stays exact, and the
//!    optimised and parallel solvers return exactly the naive solver's hybrid values (tested).
//! 2. **The transposition table stays exact.** Within one solve every card's owner is fixed by
//!    the root hands and the scores follow from the key (see `crate::tt`), so the table key
//!    determines the canonical form: positions with equal keys get equal frontier values. A
//!    frontier value is stored in the table like a searched one (it is the solve's own value of
//!    that position), and the frontier is fixed for the whole solve, so the table never mixes
//!    frontier values with values searched past the frontier: positions of trick `k` are always
//!    frontier values, positions of earlier tricks always searched-to-the-frontier values, and
//!    no position after the frontier is ever reached. Every solve starts from an empty table.
//!
//! # What the canonical form keeps (and why it is complete)
//! At a trick boundary the hand's future depends on the cards only through rules that compare
//! *live* cards (cards still in a hand): legality (effective suit, whether a card is one of the
//! top trumps that may be held back, and the rank of a held-back top trump against the led
//! trump, rules 13–18), the trick winner (rule 19), the immediate bonus of the 5 of trumps
//! (rule 29) and the tie rule (each seat breaks ties between equally good cards by card index,
//! so the order of each hand's own cards matters). Dead cards (played earlier, undealt, burnt)
//! are never compared again. So the canonical form keeps, per class of `crate::equiv` (the 5 of
//! trumps; the top trumps J, Joker, A♥; the ordinary trumps; each plain suit), the live cards in
//! rank order, each as its holder and its place in its holder's hand in card-index order, and
//! drops which dead cards lie between them.
//!
//! The best trump so far matters only for the end-of-hand bonus (rule 30): `InHand::finish`
//! reads only its team, and only when the 5 of trumps was never played. If the hand reaches its
//! end every live card has been played, so the bonus goes to the team of the highest of {the best
//! trump so far, every live trump}; if the game ends earlier no end bonus is paid. If the 5 is
//! already played, or still live (it will then be played before the hand ends), no end bonus is
//! ever paid. So the canonical form keeps only that team ([`CanonPos::end_bonus`]), or none.
//! Everything else (trumps, leader, scores, bonus flags, tricks per team, whether the 5 was
//! played, tricks done) is kept as it is; games won and the dealer's team are in
//! [`MatchCtx`], which the evaluator also gets.
//!
//! # Why a merge gives equal canonical forms
//! Let `a`, `b` be equivalent cards of seat `s` (team `T`) at a node `N` before the frontier,
//! and follow the line where `a` is played and the swapped line where `b` is played (every
//! later play of `b` becomes a play of `a`; equiv's proof shows every trick goes to the same
//! seat, so scores, tricks and flags agree). At the frontier either both are gone (the two
//! positions are identical: the same cards were played by the same seats, so even the best trump
//! agrees), or one of them is still in `s`'s hand: `b` in one line, `a` in the other. They are
//! still adjacent among the live cards of their class (cards only die), and no card of `s`'s
//! hand lies between them in card index (the hand only shrinks), so the remaining card has the
//! same rank among live cards and the same place in `s`'s hand: equal slots. The end-bonus
//! team: if `a`, `b` are trumps, the highest of {played trumps, live trumps} other than `a`,
//! `b` is the same card in both lines with the same holder or player, and `max(a, b)` was played
//! by or is held by `s` in both lines, so the team of the overall highest is the same. A burn
//! merge (trick 1) is the same: no played card can lie between the two cards (every card played
//! before the rob is in trick 1, hence live at the burn).

use std::sync::OnceLock;

use engine::cards::{Bits, Card, Suit, bit, cards, has};
use engine::rules::{five_of, trump_rank};
use engine::team_of;
use equity::MoneyChance;

use crate::{MatchCtx, Position};

/// Slots of [`CanonPos::slots`]: the 5 of trumps (1), the other top trumps J, Joker, A♥ (3), the
/// ordinary trumps (up to 11), then the three plain suits in suit order (13 each).
pub const SLOTS: usize = 54;

/// First slot of each class (5, top trumps, ordinary trumps, plain suits in suit order).
pub const CLASS_BASE: [usize; 6] = [0, 1, 4, 15, 28, 41];

/// Slots of each class.
pub const CLASS_SLOTS: [usize; 6] = [1, 3, 11, 13, 13, 13];

/// A position at a trick boundary in canonical form: everything its exact value depends on, and
/// nothing else (see the module notes). Equal canonical forms (with equal [`MatchCtx`]) have
/// equal exact values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CanonPos {
    /// The trump suit.
    pub trump: Suit,
    /// The seat to lead.
    pub leader: u8,
    /// Tricks completed this hand (`k − 1` at the start of trick `k`).
    pub tricks_done: u8,
    /// Each team's score now.
    pub scores: [u8; 4],
    /// Whether each team has taken a best-trump bonus in this game.
    pub flags: [bool; 4],
    /// Tricks each team has won this hand.
    pub tricks: [u8; 4],
    /// Whether the 5 of trumps has been played this hand.
    pub five_played: bool,
    /// The team that takes the end-of-hand bonus if the hand is played to its end: the team of
    /// the highest of the best trump so far and every live trump. `None` when no end bonus can
    /// be paid (the 5 of trumps played or still live) or no trump is played or live.
    pub end_bonus: Option<u8>,
    /// The live cards by class and rank (see [`CLASS_BASE`]): within a class, the best live card
    /// in the class's first slot, the next in the next, and so on. `0` = no card; otherwise
    /// `1 + seat + 8 × place`, where `place` is the card's place (0 first) among its holder's
    /// cards in card-index order.
    pub slots: [u8; SLOTS],
}

impl CanonPos {
    /// The holder and place of a slot's card, if any.
    pub fn slot(&self, i: usize) -> Option<(u8, u8)> {
        match self.slots[i] {
            0 => None,
            x => Some(((x - 1) % 8, (x - 1) / 8)),
        }
    }
}

/// Every card in canonical order (class by class as in [`CLASS_BASE`], best first within a
/// class) with its class (0..6), for a trump suit.
pub fn canonical_order(trump: Suit) -> &'static [(Card, u8); 53] {
    static ALL: OnceLock<[[(Card, u8); 53]; 4]> = OnceLock::new();
    &ALL.get_or_init(|| {
        Suit::ALL.map(|t| {
            let (lists, lens) = crate::equiv::lists(t);
            let mut out = [(0 as Card, 0u8); 53];
            let mut n = 0;
            for (cls, (list, &len)) in lists.iter().zip(&lens).enumerate() {
                debug_assert!(usize::from(len) <= CLASS_SLOTS[cls]);
                for &c in &list[..usize::from(len)] {
                    out[n] = (c, cls as u8);
                    n += 1;
                }
            }
            debug_assert_eq!(n, 53);
            out
        })
    })[trump as usize]
}

/// The canonical form of a position at a trick boundary (no card of the current trick played,
/// no rob pending).
pub fn canonical(p: &Position) -> CanonPos {
    debug_assert_eq!(
        p.n_played, 0,
        "the canonical form is taken at a trick boundary"
    );
    debug_assert!(
        p.rob.is_none(),
        "no rob is pending at a trick boundary after trick 1"
    );
    let mut owner = [0u8; 64];
    let mut live: Bits = 0;
    for (s, &h) in p.hands.iter().enumerate() {
        live |= h;
        for c in cards(h) {
            owner[usize::from(c)] = s as u8;
        }
    }
    let mut slots = [0u8; SLOTS];
    let mut next = CLASS_BASE;
    let mut top_trump: Option<Card> = None;
    for &(c, cls) in canonical_order(p.trump) {
        if !has(live, c) {
            continue;
        }
        let s = owner[usize::from(c)];
        let place = (p.hands[usize::from(s)] & (bit(c) - 1)).count_ones() as u8;
        let k = usize::from(cls);
        slots[next[k]] = 1 + s + 8 * place;
        next[k] += 1;
        // Classes 0–2 are the trumps, best first: the first live one met is the highest.
        if cls <= 2 && top_trump.is_none() {
            top_trump = Some(c);
        }
    }
    let end_bonus = if p.hand.five_played || has(live, five_of(p.trump)) {
        None
    } else {
        let live_top = top_trump.map(|c| (trump_rank(c, p.trump), team_of(owner[usize::from(c)])));
        match (p.best, live_top) {
            (Some((b, bt)), Some((r, t))) => Some(if r > b { t } else { bt }),
            (Some((_, bt)), None) => Some(bt),
            (None, Some((_, t))) => Some(t),
            (None, None) => None,
        }
    };
    CanonPos {
        trump: p.trump,
        leader: p.leader,
        tricks_done: p.hand.tricks_done,
        scores: p.hand.scores,
        flags: p.hand.flags,
        tricks: p.hand.tricks,
        five_played: p.hand.five_played,
        end_bonus,
        slots,
    }
}

/// Values positions at the frontier (normally the oracle network). It must be a pure function
/// of its arguments: the solvers call it from several threads and in any order, and take each
/// value as the value of every position with that canonical form.
pub trait Frontier: Sync {
    /// The money chance of a frontier position.
    fn value(&self, pos: &CanonPos, ctx: &MatchCtx) -> MoneyChance;

    /// The money chance of a frontier position as the solver reached it. Every evaluator but a
    /// recording one keeps the default, [`Frontier::value`] of the canonical form. The label
    /// harvester (`search::oracle_labels`) overrides it to record the positions a search reaches
    /// and to value them with the exact solver, which needs the position itself; whatever an
    /// override does with `pos`, the value it returns must still depend only on `canon` and
    /// `ctx` (the module notes: otherwise the optimised solvers would no longer be exact).
    fn value_at(&self, pos: &Position, canon: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        let _ = pos;
        self.value(canon, ctx)
    }
}

/// A closure as a frontier evaluator (tests and synthetic evaluators).
pub struct FnFrontier<F>(pub F);

impl<F: Fn(&CanonPos, &MatchCtx) -> MoneyChance + Sync> Frontier for FnFrontier<F> {
    fn value(&self, pos: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        (self.0)(pos, ctx)
    }
}

/// A frontier at the start of trick `trick` (2..=5): every position reached there (below the
/// root) is valued by the evaluator instead of being searched. The root itself is never cut, so
/// the hybrid is meant for roots before trick `trick` (a root at or after it is solved exactly).
#[derive(Clone, Copy)]
pub struct FrontierCut<'a> {
    trick: u8,
    eval: &'a dyn Frontier,
}

impl<'a> FrontierCut<'a> {
    /// A frontier at the start of trick `trick`.
    ///
    /// # Panics
    /// If `trick` is not 2..=5 (a rob may be pending in trick 1, and there are five tricks).
    pub fn new(trick: u8, eval: &'a dyn Frontier) -> Self {
        assert!(
            (2..=engine::game::TRICKS_PER_HAND).contains(&trick),
            "a frontier is at the start of trick 2..=5"
        );
        FrontierCut { trick, eval }
    }

    /// The trick at whose start the frontier lies.
    pub fn trick(&self) -> u8 {
        self.trick
    }

    /// Is `p` on the frontier (the start of the frontier's trick)?
    #[inline]
    pub fn applies(&self, p: &Position) -> bool {
        p.n_played == 0 && p.hand.tricks_done + 1 == self.trick && p.rob.is_none()
    }

    /// The evaluator's value of a frontier position.
    pub fn value(&self, p: &Position, ctx: &MatchCtx) -> MoneyChance {
        self.eval.value_at(p, &canonical(p), ctx)
    }
}
