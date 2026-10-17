//! The card tracker (docs/M3-SPEC.md §3, PLAN A6): for every card the viewer cannot see, the set
//! of places it may be in, from hard facts [certainties from the rules and the cards seen] only.
//!
//! The facts are kept as small per-hand records updated one event at a time ([`Tracker::update`]);
//! the per-card masks are rebuilt from those records and the current view after each update.
//! Every rule fact comes from the engine's own functions, never from re-derived copies (A6):
//! - voids and trump limits (F2, F3) from `legal_plays`: a play `p` to a trick led by `L` rules
//!   out exactly the cards `c` for which `p` is not legal from the two-card hand `{p, c}` (legality
//!   is decided card by card, so this is the whole fact; the test `tracker_matches_engine_legality`
//!   checks it against the rule text);
//! - the ace of trumps from `ace_of_trumps`, `legal_burns` and the `ProvedNoAce` / `AceUndealt`
//!   events (F4–F11);
//! - capacities from the view's hand sizes, burn counts, burn order and hidden count (F12).
//!
//! A partnership the app has turned (docs/APP-ENGINE-CONTRACT.md §7) may have broken a rule
//! unseen: its seats' show-outs and "no ace" proofs prove nothing, and a turned robber may have
//! taken the turned card without the ace of trumps or burned it. Its face-down cards are still
//! cards in its hand (capacities as before), out of play but not anywhere else. What the turn
//! itself makes public is kept: a robber's partnership caught at a claim or at the end of the
//! hand does not hold the ace of trumps; a robber turned for not robbing holds it. A robber still
//! in play whose own plays show it cannot hold the ace of trumps (a plain card to a trump lead)
//! took the turned card without it or burned it, uncaught yet: its rob proves nothing either.

use engine::cards::{ALL_CARDS, Bits, Card, JOKER, Suit, bit, cards, count, has, suit_of};
use engine::game::{Event, Phase, TurnCause};
use engine::rules::{ace_of_trumps, legal_burns, legal_plays};
use engine::view::{BurnKind, SeatView};

/// A place a hidden card can be in (docs/M3-SPEC.md §3.1). The derived order is the order of
/// locations everywhere (`Hand` < `Undealt` < `Burn`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Loc {
    /// The hand of this seat.
    Hand(u8),
    /// The undealt pack (face down; during the dealer's burn after a turned Joker it includes the
    /// card that will be turned next).
    Undealt,
    /// The single card this seat burned with this kind of burn.
    Burn(u8, BurnKind),
}

/// Facts gathered from the hand's events (independent of the current view).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Facts {
    dealer: u8,
    trump: Option<Suit>,
    /// Cards proved absent from each seat's hand by its plays (F2, F3).
    hand_forbid: [Bits; 8],
    /// Taken turned cards: (seat, card, number of burns made before the take) (F5–F7).
    taken: Vec<(u8, Card, usize)>,
    /// Burns made so far (counted from public `BurnDone` events).
    burns_done: usize,
    /// The seat that robbed (took a turned card other than the Joker): it holds the ace of
    /// trumps until it plays it (F5, F6).
    robber: Option<u8>,
    /// `ProvedNoAce` per seat (F4).
    no_ace: [bool; 8],
    /// `AceUndealt` (F11).
    ace_undealt: bool,
    /// The card that led the current trick.
    led: Option<Card>,
    /// The seat that played the last card of the current trick.
    last_played: Option<u8>,
    /// Partnerships turned this hand (their seats' plays prove nothing).
    turned: [bool; 4],
    /// The seat turned for not robbing: it holds the ace of trumps.
    ace_holder: Option<u8>,
    /// The robber's partnership was caught at a claim or at the end of the hand: the robber does
    /// not hold the ace of trumps.
    robber_lacks_ace: bool,
    /// Seats the engine itself shows without the ace of trumps whatever they did (F9 at a past
    /// moment): asked to play in trick 1, with nobody robbed, when the app turned a partnership
    /// before that play or with it.
    no_ace_certain: [bool; 8],
    /// Tricks completed this hand.
    tricks_done: u8,
    /// The seat that played the ace of trumps, once played.
    ace_played_by: Option<u8>,
}

impl Facts {
    fn new(dealer: u8) -> Facts {
        Facts {
            dealer,
            trump: None,
            hand_forbid: [0; 8],
            taken: Vec::new(),
            burns_done: 0,
            robber: None,
            no_ace: [false; 8],
            ace_undealt: false,
            led: None,
            last_played: None,
            turned: [false; 4],
            ace_holder: None,
            robber_lacks_ace: false,
            no_ace_certain: [false; 8],
            tricks_done: 0,
            ace_played_by: None,
        }
    }

    /// The seat to act in trick 1 now: the first seat still in play after the last card played,
    /// or from the dealer's left.
    fn to_act_in_trick_one(&self) -> u8 {
        let mut s = self
            .last_played
            .map_or((self.dealer + 1) % 8, |p| (p + 1) % 8);
        while self.turned[usize::from(s % 4)] {
            s = (s + 1) % 8;
        }
        s
    }

    fn apply(&mut self, ev: &Event) {
        match *ev {
            Event::HandStart { dealer, .. } => *self = Facts::new(dealer),
            Event::Turned { card } => {
                if card != JOKER && self.trump.is_none() {
                    self.trump = suit_of(card);
                }
            }
            Event::Took { seat, card } => {
                self.taken.push((seat, card, self.burns_done));
                if card != JOKER {
                    self.robber = Some(seat);
                }
            }
            Event::BurnDone { .. } => self.burns_done += 1,
            Event::ProvedNoAce { seat } => self.no_ace[usize::from(seat)] = true,
            Event::AceUndealt => self.ace_undealt = true,
            Event::TrickWon { .. } => {
                self.led = None;
                self.last_played = None;
                self.tricks_done += 1;
            }
            Event::Played { seat, card } => {
                self.last_played = Some(seat);
                if self.trump.is_some_and(|t| card == ace_of_trumps(t)) {
                    self.ace_played_by = Some(seat);
                }
                match self.led {
                    None => self.led = Some(card),
                    Some(led) => {
                        let trump = self.trump.expect("tricks are played with trumps known");
                        self.hand_forbid[usize::from(seat)] |= play_forbids(card, led, trump);
                    }
                }
            }
            Event::PairTurned { team, cause, .. } => {
                let t = usize::from(team);
                let rob_open = self.tricks_done == 0 && self.robber.is_none();
                match cause {
                    TurnCause::NoRob => {
                        // The seat whose rob was due: the seat to act in trick 1.
                        let s = self.to_act_in_trick_one();
                        debug_assert_eq!(usize::from(s % 4), t);
                        self.ace_holder = Some(s);
                    }
                    // The engine robs by itself: a seat asked to play in trick 1 with nobody
                    // robbed lacks the ace (the app turned a partnership before its card, or right
                    // after it).
                    TurnCause::BeforePlay if rob_open && self.trump.is_some() => {
                        self.no_ace_certain[usize::from(self.to_act_in_trick_one())] = true;
                    }
                    TurnCause::WithPlay if rob_open => {
                        if let Some(s) = self.last_played {
                            self.no_ace_certain[usize::from(s)] = true;
                        }
                    }
                    TurnCause::Claim | TurnCause::HandEnd
                        if self.robber.is_some_and(|r| usize::from(r % 4) == t) =>
                    {
                        self.robber_lacks_ace = true;
                    }
                    _ => {}
                }
                self.turned[t] = true;
            }
            _ => {}
        }
    }
}

/// F2 and F3: the cards a seat cannot hold after playing `card` to a trick led by `led`. These
/// are exactly the cards `c` for which `card` is not a legal play from the hand `{card, c}`
/// (`legal_plays`): a plain card of the led plain suit when another plain suit is played (rule
/// 14), every trump that may not be held back when a non-trump follows a trump lead (rules
/// 16–18). Trumping a plain lead, following suit and following a trump lead with a trump prove
/// nothing.
pub fn play_forbids(card: Card, led: Card, trump: Suit) -> Bits {
    let mut forbid = 0;
    for c in cards(ALL_CARDS & !bit(card)) {
        if !has(legal_plays(bit(card) | bit(c), Some(led), trump), card) {
            forbid |= bit(c);
        }
    }
    forbid
}

/// What one seat can know about where the hidden cards are (docs/M3-SPEC.md §3.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tracker {
    /// The viewer.
    pub seat: u8,
    /// The viewer's own hand.
    pub hand: Bits,
    /// The trump suit, `None` during the dealer's burn after a turned Joker.
    pub trump: Option<Suit>,
    /// Cards the viewer sees: own hand, own burns, played cards, turned cards left face up.
    pub known: Bits,
    /// The other cards of the 53.
    pub unknown: Bits,
    /// Every location and its capacity, sorted by `Loc` (locations of capacity 0 are left out).
    pub locs: Vec<(Loc, u8)>,
    /// Per card: bit `i` set iff the card may be in `locs[i]`; 0 for known cards.
    pub allowed: [u32; 53],
    /// The hidden out-of-play count of the view (12 or 13, F12; 11 after a Joker left on the
    /// deck).
    pub hidden_count: u8,
    /// The robber (if any) may have taken the turned card without the ace of trumps or burned it
    /// and not been caught: its partnership is turned, or the cards the viewer sees show it.
    pub robber_may_have_lapsed: bool,
    /// Seats (never the viewer) whose show-outs and "no ace" proofs this tracker does not trust:
    /// a turned partnership's, or, when the cards seen admit no placement otherwise, every seat's.
    pub lapsed: [bool; 8],
    facts: Facts,
}

/// The events of the current hand (from its `HandStart`) in a view.
pub fn hand_events(view: &SeatView) -> &[Event] {
    let i = view
        .events
        .iter()
        .rposition(|e| matches!(e, Event::HandStart { .. }))
        .unwrap_or(0);
    &view.events[i..]
}

impl Tracker {
    /// A tracker from the hand's start events (`HandStart` and the deal) of this view. Call
    /// [`Tracker::update`] with every later event of the hand, or use [`Tracker::from_view`].
    pub fn new(view: &SeatView) -> Tracker {
        let evs = hand_events(view);
        Self::from_events(view, &evs[..Self::start_len(evs)])
    }

    /// A tracker with the facts of these events of the hand, its masks built against `view`.
    fn from_events(view: &SeatView, evs: &[Event]) -> Tracker {
        let mut t = Tracker {
            seat: view.seat,
            hand: view.hand,
            trump: None,
            known: 0,
            unknown: 0,
            locs: Vec::new(),
            allowed: [0; 53],
            hidden_count: view.public.hidden_count,
            robber_may_have_lapsed: false,
            lapsed: [false; 8],
            facts: Facts::new(view.public.dealer),
        };
        for e in evs {
            t.facts.apply(e);
        }
        t.rebuild(view);
        t
    }

    /// The number of the hand's start events (`HandStart` then `Dealt`s) that [`Tracker::new`]
    /// reads.
    pub fn start_len(evs: &[Event]) -> usize {
        evs.iter()
            .take_while(|e| matches!(e, Event::HandStart { .. } | Event::Dealt { .. }))
            .count()
    }

    /// Add one event of the hand (in order) and refresh the masks against `view`, the view after
    /// it (or after the action that produced it).
    pub fn update(&mut self, ev: &Event, view: &SeatView) {
        self.facts.apply(ev);
        self.rebuild(view);
    }

    /// [`Tracker::new`] then every later event of the current hand: the tracker of this view.
    /// (The masks are built once, from all the facts: exactly what the masks built after the
    /// start events and rebuilt after the rest would be.)
    pub fn from_view(view: &SeatView) -> Tracker {
        Self::from_events(view, hand_events(view))
    }

    /// A tracker built directly from its masks, with no facts behind them: for hand-made
    /// positions in tests and oracles. `locs` must be sorted by `Loc` with capacities summing to
    /// `count(unknown)`, `allowed` must be 0 outside `unknown`, and `seat` has no location.
    pub fn from_parts(
        seat: u8,
        hand: Bits,
        unknown: Bits,
        locs: Vec<(Loc, u8)>,
        allowed: [u32; 53],
    ) -> Tracker {
        assert!(locs.windows(2).all(|w| w[0].0 < w[1].0), "sorted locations");
        assert_eq!(
            locs.iter().map(|&(_, c)| u32::from(c)).sum::<u32>(),
            u32::from(count(unknown)),
            "the capacities hold exactly the unknown cards"
        );
        assert!(locs.len() <= 32 && locs.iter().all(|&(l, _)| l != Loc::Hand(seat)));
        assert!((0..53u8).all(|c| has(unknown, c) || allowed[usize::from(c)] == 0));
        let hidden_count = locs
            .iter()
            .filter(|(l, _)| !matches!(l, Loc::Hand(_)))
            .map(|&(_, c)| c)
            .sum();
        Tracker {
            seat,
            hand,
            trump: None,
            known: ALL_CARDS & !unknown,
            unknown,
            locs,
            allowed,
            hidden_count,
            robber_may_have_lapsed: false,
            lapsed: [false; 8],
            facts: Facts::new(0),
        }
    }

    /// The seat that robbed this hand, if any.
    pub fn robber(&self) -> Option<u8> {
        self.facts.robber
    }

    /// The dealer of the hand.
    pub fn dealer(&self) -> u8 {
        self.facts.dealer
    }

    /// Cards proved absent from a seat's hand by its plays (F2, F3).
    pub fn hand_forbid(&self, seat: u8) -> Bits {
        self.facts.hand_forbid[usize::from(seat)]
    }

    /// Rebuild `known`, `unknown`, `locs` and `allowed` from the facts and the view. When the
    /// facts admit no placement at all (only possible after the app's referee let a renege through
    /// that the cards seen give away but nobody has caught yet), every other seat is taken as
    /// possibly having broken the rules, as a turned seat is.
    fn rebuild(&mut self, view: &SeatView) {
        self.fill(view, false);
        if !self.placeable() {
            self.fill(view, true);
        }
    }

    /// Can the unknown cards be placed at all (max-flow over the card classes)?
    fn placeable(&self) -> bool {
        let mut by_mask: Vec<(u32, u8)> = Vec::new();
        for c in cards(self.unknown) {
            let m = self.allowed[usize::from(c)];
            match by_mask.iter_mut().find(|(mm, _)| *mm == m) {
                Some((_, n)) => *n += 1,
                None => by_mask.push((m, 1)),
            }
        }
        let rem: Vec<u8> = by_mask.iter().map(|&(_, n)| n).collect();
        let allow: Vec<u32> = by_mask.iter().map(|&(m, _)| m).collect();
        let caps: Vec<u8> = self.locs.iter().map(|&(_, c)| c).collect();
        crate::space::feasible(&rem, &allow, &caps, 0)
    }

    /// [`Tracker::rebuild`]'s work, with every seat but the viewer taken as possibly having broken
    /// the rules (`relax_all`) or only those the app turned and those the cards seen give away.
    fn fill(&mut self, view: &SeatView, relax_all: bool) {
        let p = &view.public;
        let me = view.seat;
        let f = &self.facts;
        self.hand = view.hand;
        self.trump = f.trump;
        self.hidden_count = p.hidden_count;
        // F1: own hand and burns, played cards, turned cards nobody took.
        let mut known = view.hand | view.own_burns | p.played;
        if let Some(c) = p.table_card {
            known |= bit(c);
        }
        for &(c, taker) in &p.turned {
            if taker.is_none() {
                known |= bit(c);
            }
        }
        self.known = known;
        self.unknown = ALL_CARDS & !known;
        // Locations and capacities (§3.1, F12).
        let mut locs: Vec<(Loc, u8)> = Vec::new();
        for s in 0..8u8 {
            let n = p.hand_sizes[usize::from(s)];
            if s != me && n > 0 {
                locs.push((Loc::Hand(s), n));
            }
        }
        // 11 only when the dealer left a turned Joker on the deck (the app's referee: the Joker
        // and the next turned card are both face up, nothing burned).
        assert!(
            (12..=13).contains(&p.hidden_count) || (p.hidden_count == 11 && p.joker_turned),
            "F12: 12 or 13 hidden out-of-play cards, not {}",
            p.hidden_count
        );
        let burned: u8 = p.burn_counts.iter().sum();
        assert_eq!(
            usize::from(burned),
            p.burn_order.len(),
            "one burn-order entry per burn"
        );
        let undealt = p.hidden_count - burned;
        if undealt > 0 {
            locs.push((Loc::Undealt, undealt));
        }
        // Each burn's position in the hand's burn order (for "burned after the take").
        let mut burn_pos: Vec<(Loc, usize)> = Vec::new();
        for (i, &(s, k)) in p.burn_order.iter().enumerate() {
            if s != me {
                locs.push((Loc::Burn(s, k), 1));
                burn_pos.push((Loc::Burn(s, k), i));
            }
        }
        locs.sort();
        assert!(
            locs.windows(2).all(|w| w[0].0 != w[1].0),
            "one burn per seat and kind"
        );
        assert_eq!(
            locs.iter().map(|&(_, c)| u32::from(c)).sum::<u32>(),
            u32::from(count(self.unknown)),
            "F12: the capacities hold exactly the unknown cards"
        );
        let ace = f.trump.map(ace_of_trumps);
        let out = |s: u8| relax_all || f.turned[usize::from(s % 4)];
        // The robber holds the ace of trumps until it plays it (F5, F6), unless its partnership
        // is turned, or its own plays show it cannot (a plain card to a trump lead, or every card
        // played without it), or it is the viewer without it, or the engine showed it without the
        // ace before its rob, or another seat was turned for not robbing with it: then it took the
        // turned card without it, or burned it, and the app has not caught it yet.
        // A robber that took the ace of trumps itself from the table at its turn (a dealer had
        // refused it) did not hold it before: it took the turned card without it.
        let took_ace_at_turn = f.robber.is_some_and(|r| {
            ace.is_some_and(|a| f.taken.iter().any(|&(t, c, _)| t == r && c == a))
                && p.burn_order
                    .iter()
                    .any(|&(s, k)| s == r && k == BurnKind::RobAtTurn)
        });
        let robber_in_play = f.robber.filter(|&r| {
            r != me
                && !out(r)
                && !took_ace_at_turn
                && f.ace_holder.is_none()
                && !f.no_ace_certain[usize::from(r)]
                && p.hand_sizes[usize::from(r)] > 0
                && !ace.is_some_and(|a| has(f.hand_forbid[usize::from(r)], a))
        });
        let lapsed_robber = f.robber.filter(|&r| robber_in_play != Some(r));
        // Seen elsewhere: in the viewer's hand or burn (the viewer not the robber), or played by
        // another seat.
        let ace_elsewhere = ace.is_some_and(|a| {
            (has(view.hand | view.own_burns, a) && f.robber != Some(me))
                || f.ace_played_by.is_some_and(|s| Some(s) != f.robber)
        });
        self.robber_may_have_lapsed =
            f.robber.is_some() && (ace_elsewhere || lapsed_robber.is_some_and(|r| r != me));
        // F9 (the engine robs automatically): a seat asked to play its first card of trick 1
        // while nobody has robbed does not hold the ace of trumps.
        let to_play_without_ace = match p.phase {
            Phase::AwaitPlay(s) if p.trick_no == 1 && f.robber.is_none() => Some(s),
            _ => None,
        };
        let mut allowed = [0u32; 53];
        for c in cards(self.unknown) {
            let taken = f.taken.iter().find(|&&(_, tc, _)| tc == c).copied();
            let is_ace = ace == Some(c);
            let mut mask = 0u32;
            for (i, &(loc, _)) in locs.iter().enumerate() {
                let ok = match loc {
                    Loc::Hand(s) => {
                        let si = usize::from(s);
                        // F2, F3 (a turned seat's show-outs prove nothing).
                        let mut ok = out(s) || !has(f.hand_forbid[si], c);
                        // F5–F7: a taken card stays with its taker (hand or a later burn).
                        if let Some((t, _, _)) = taken {
                            ok &= t == s;
                        }
                        if is_ace {
                            if let Some(h) = f.ace_holder {
                                ok &= h == s;
                            }
                            if let Some(r) = robber_in_play {
                                ok &= r == s; // F5, F6.
                            }
                            if f.robber_lacks_ace {
                                ok &= f.robber != Some(s);
                            }
                            ok &= out(s) || !f.no_ace[si]; // F4.
                            ok &= !f.no_ace_certain[si]; // F9 at a turn.
                            ok &= out(s) || !f.ace_undealt; // F11.
                            ok &= to_play_without_ace != Some(s); // F9.
                        }
                        ok
                    }
                    // A taken card is never back in the pack; nor is a robbed ace (F5, F6).
                    Loc::Undealt => {
                        taken.is_none()
                            && !(is_ace && (robber_in_play.is_some() || f.ace_holder.is_some()))
                    }
                    Loc::Burn(s, k) => {
                        let pos = burn_pos
                            .iter()
                            .find(|&&(l, _)| l == loc)
                            .map(|&(_, i)| i)
                            .expect("every burn location has a position");
                        let mut ok = true;
                        if let Some((t, _, before)) = taken {
                            // A taken card can only be burned by its taker, after the take.
                            ok &= t == s && pos >= before;
                        }
                        if is_ace {
                            // F6, F8: a burn made with trumps known never holds the ace of
                            // trumps (`legal_burns`), unless a robber whose partnership is turned
                            // made it. F7: the dealer's burn after a turned Joker was made before
                            // trumps were known and may hold it, unless someone still in play
                            // robbed (then the robber holds it).
                            let trump_at_burn = match k {
                                BurnKind::JokerTake => None,
                                BurnKind::AceRobAtOnce | BurnKind::RobAtTurn => f.trump,
                            };
                            let lapsed = k == BurnKind::RobAtTurn
                                && lapsed_robber == Some(s)
                                && !f.no_ace_certain[usize::from(s)]
                                && !took_ace_at_turn;
                            ok &= lapsed || legal_burns(bit(c), trump_at_burn) != 0;
                            ok &= robber_in_play.is_none() && f.ace_holder.is_none();
                        }
                        ok
                    }
                };
                if ok {
                    mask |= 1 << i;
                }
            }
            allowed[usize::from(c)] = mask;
        }
        self.locs = locs;
        self.allowed = allowed;
        self.lapsed = core::array::from_fn(|s| s as u8 != me && out(s as u8));
    }

    /// The sum of all capacities (equal to `count(unknown)`, F12).
    pub fn capacity_sum(&self) -> u32 {
        self.locs.iter().map(|&(_, c)| u32::from(c)).sum()
    }

    /// The index of a location in `locs`.
    pub fn loc_index(&self, loc: Loc) -> Option<usize> {
        self.locs.iter().position(|&(l, _)| l == loc)
    }

    /// The cards that may be in `loc` (0 if it is not a location).
    pub fn may_hold(&self, loc: Loc) -> Bits {
        let Some(i) = self.loc_index(loc) else {
            return 0;
        };
        cards(self.unknown)
            .filter(|&c| self.allowed[usize::from(c)] & (1 << i) != 0)
            .fold(0, |a, c| a | bit(c))
    }

    /// The locations a card may be in.
    pub fn places(&self, c: Card) -> Vec<Loc> {
        self.locs
            .iter()
            .enumerate()
            .filter(|&(i, _)| self.allowed[usize::from(c)] & (1 << i) != 0)
            .map(|(_, &(l, _))| l)
            .collect()
    }

    /// Does this placement of the cards agree with every fact? `hands[viewer]` is ignored;
    /// `burns_by_loc` may also list the viewer's own burns (ignored). Every location must be
    /// filled to its capacity with unknown cards it allows, and every unknown card placed once.
    pub fn consistent(
        &self,
        hands: &[Bits; 8],
        burns_by_loc: &[(Loc, Bits)],
        undealt: Bits,
    ) -> bool {
        let mut seen: Bits = 0;
        for (i, &(loc, cap)) in self.locs.iter().enumerate() {
            let set = match loc {
                Loc::Hand(s) => hands[usize::from(s)],
                Loc::Undealt => undealt,
                Loc::Burn(..) => match burns_by_loc.iter().find(|&&(l, _)| l == loc) {
                    Some(&(_, b)) => b,
                    None => return false,
                },
            };
            if count(set) != cap || set & !self.unknown != 0 || set & seen != 0 {
                return false;
            }
            if cards(set).any(|c| self.allowed[usize::from(c)] & (1 << i) == 0) {
                return false;
            }
            seen |= set;
        }
        // Seats without a location (the viewer, empty hands) hold no unknown card.
        for s in 0..8u8 {
            if s != self.seat
                && self.loc_index(Loc::Hand(s)).is_none()
                && hands[usize::from(s)] & self.unknown != 0
            {
                return false;
            }
        }
        seen == self.unknown
    }
}
