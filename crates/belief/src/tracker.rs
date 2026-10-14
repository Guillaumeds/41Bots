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

use engine::cards::{ALL_CARDS, Bits, Card, JOKER, Suit, bit, cards, count, has, suit_of};
use engine::game::{Event, Phase};
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
        }
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
            Event::TrickWon { .. } => self.led = None,
            Event::Played { seat, card } => match self.led {
                None => self.led = Some(card),
                Some(led) => {
                    let trump = self.trump.expect("tricks are played with trumps known");
                    self.hand_forbid[usize::from(seat)] |= play_forbids(card, led, trump);
                }
            },
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
    /// The hidden out-of-play count of the view (12 or 13, F12).
    pub hidden_count: u8,
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
        let mut t = Tracker {
            seat: view.seat,
            hand: view.hand,
            trump: None,
            known: 0,
            unknown: 0,
            locs: Vec::new(),
            allowed: [0; 53],
            hidden_count: view.public.hidden_count,
            facts: Facts::new(view.public.dealer),
        };
        for e in &evs[..Self::start_len(evs)] {
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
    pub fn from_view(view: &SeatView) -> Tracker {
        let mut t = Tracker::new(view);
        let evs = hand_events(view);
        for e in &evs[Self::start_len(evs)..] {
            t.facts.apply(e);
        }
        t.rebuild(view);
        t
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

    /// Rebuild `known`, `unknown`, `locs` and `allowed` from the facts and the view.
    fn rebuild(&mut self, view: &SeatView) {
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
        assert!(
            p.hidden_count == 12 || p.hidden_count == 13,
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
                        // F2, F3.
                        let mut ok = !has(f.hand_forbid[si], c);
                        // F5–F7: a taken card stays with its taker (hand or a later burn).
                        if let Some((t, _, _)) = taken {
                            ok &= t == s;
                        }
                        if is_ace {
                            if let Some(r) = f.robber {
                                ok &= r == s; // F5, F6.
                            }
                            ok &= !f.no_ace[si]; // F4.
                            ok &= !f.ace_undealt; // F11.
                            ok &= to_play_without_ace != Some(s); // F9.
                        }
                        ok
                    }
                    // A taken card is never back in the pack; nor is a robbed ace (F5, F6).
                    Loc::Undealt => taken.is_none() && !(is_ace && f.robber.is_some()),
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
                            // trumps (`legal_burns`). F7: the dealer's burn after a turned
                            // Joker was made before trumps were known and may hold it, unless
                            // someone robbed (then the robber holds it).
                            let trump_at_burn = match k {
                                BurnKind::JokerTake => None,
                                BurnKind::AceRobAtOnce | BurnKind::RobAtTurn => f.trump,
                            };
                            ok &= legal_burns(bit(c), trump_at_burn) != 0;
                            ok &= f.robber.is_none();
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
