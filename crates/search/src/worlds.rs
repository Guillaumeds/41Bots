//! Worlds [complete guesses of where every hidden card is] and rebuilding a world as a match
//! (docs/M3-SPEC.md §4, §5.1).
//!
//! The search draws worlds through the small [`WorldSource`] trait, so the world generator can be
//! swapped without touching the search:
//! - [`BeliefSource`] (the default, M3-SPEC §5.2 `belief = uniform`): `crates/belief`'s card
//!   tracker and counted world space. Exactly uniform over the worlds consistent with the hard
//!   facts, and it reports the exact number of those worlds (for records and the §13.1
//!   world-count bands).
//! - [`RejectionSource`]: deal the unknown cards uniformly into the hidden places by size, rebuild
//!   the match and keep the deal only if the deciding seat's view comes out identical. It is
//!   exactly uniform over the worlds consistent with the view (every consistent deal is kept,
//!   every inconsistent one refused) and needs no card tracker, but it cannot count worlds and
//!   gets slow when the facts are tight. It is the test oracle for the belief source.
//!
//! World `w` of a decision is drawn from `Rng::derive(dseed, &[WORLD, w])`, so it is the same
//! however many worlds are drawn and whichever thread draws it.

use engine::cards::{ALL_CARDS, Bits, Card, DECK_SIZE, JOKER, bit, cards, count, has};
use engine::game::{Action, Event, Match, Phase, Rules, build_deck, current_hand_start_of};
use engine::rng::Rng;
use engine::view::SeatView;

use crate::labels;

/// One complete guess of the hidden cards, consistent with a seat's view.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct World {
    /// Every seat's current hand (the viewer's is its real hand).
    pub hands: [Bits; 8],
    /// The hidden burns of the *other* seats this hand, one per `BurnDone` of another seat, in
    /// the order they were made.
    pub burns: Vec<Card>,
    /// The undealt pack (as a set; its order never matters).
    pub undealt: Bits,
    /// Only during the dealer's burn after a turned Joker: the card that will be turned next
    /// (one of `undealt`'s cards, which then excludes it).
    pub next_turned: Option<Card>,
}

/// Prepared worlds for one decision.
pub trait Worlds: Sync {
    /// The exact number of consistent worlds, when the source can count them.
    fn count(&self) -> Option<u128>;
    /// Draw one world, uniformly (or by the source's belief) from the consistent worlds.
    fn sample(&self, rng: &mut Rng) -> World;
}

/// A world generator.
pub trait WorldSource: Send + Sync {
    /// A short stable name for records.
    fn name(&self) -> &str;
    /// Prepare the worlds consistent with this view (and these rule switches).
    fn prepare<'a>(&'a self, view: &'a SeatView, rules: Rules) -> Box<dyn Worlds + 'a>;
}

/// Draw world number `w` of the decision with seed `dseed`.
pub fn world_rng(dseed: u64, w: u64) -> Rng {
    Rng::derive(dseed, &[labels::WORLD, w])
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

/// Why a world could not be rebuilt as a match (the world is inconsistent with the view).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RebuildError(pub String);

/// The hidden places of a view: the other seats' hand sizes, the number of other seats' hidden
/// burns, the undealt count, and the unknown cards.
#[derive(Clone, Debug)]
pub struct Hidden {
    /// Cards whose place the viewer does not know.
    pub unknown: Bits,
    /// Hand size per seat (the viewer's is not a hidden place, so 0 here).
    pub hand_sizes: [u8; 8],
    /// Hidden burns of other seats, by seat, in burn order.
    pub burn_seats: Vec<u8>,
    /// Cards in the undealt pack (including the next turned card in the Joker-burn case).
    pub undealt: u8,
    /// The dealer is burning after a turned Joker: one undealt card will be turned next.
    pub joker_burn: bool,
}

impl Hidden {
    /// The hidden places of this view.
    pub fn of(view: &SeatView) -> Hidden {
        let p = &view.public;
        let me = view.seat;
        let mut known = view.hand | view.own_burns | p.played;
        if let Some(c) = p.table_card {
            known |= bit(c);
        }
        let mut burn_seats = Vec::new();
        for e in hand_events(view) {
            if let Event::BurnDone { seat } = *e
                && seat != me
            {
                burn_seats.push(seat);
            }
        }
        let mut hand_sizes = p.hand_sizes;
        hand_sizes[usize::from(me)] = 0;
        let burns_total: u8 = p.burn_counts.iter().sum();
        let undealt = p.hidden_count - burns_total;
        let joker_burn = p.trump.is_none();
        Hidden {
            unknown: ALL_CARDS & !known,
            hand_sizes,
            burn_seats,
            undealt,
            joker_burn,
        }
    }
}

/// Rebuild a world as a match (docs/M3-SPEC.md §5.1): build the hand's deck from the world, start
/// the match at the hand's start state, and re-apply the hand's actions. The viewer's own actions
/// come from its events; another seat's burn comes from the world.
pub fn rebuild(
    view: &SeatView,
    world: &World,
    rules: Rules,
    seed: u64,
) -> Result<Match, RebuildError> {
    let err = |s: &str| Err(RebuildError(s.to_string()));
    let me = view.seat;
    let start = current_hand_start_of(view);
    let evs = hand_events(view);
    // Each seat's original five cards: current hand + its plays + its burns − what it took.
    let mut orig = world.hands;
    orig[usize::from(me)] = view.hand;
    let mut k = 0usize;
    let mut actions = Vec::new();
    for e in evs {
        match *e {
            Event::Played { seat, card } => {
                orig[usize::from(seat)] |= bit(card);
                actions.push(Action::Play(card));
            }
            Event::Burned { seat, card } if seat == me => {
                orig[usize::from(seat)] |= bit(card);
                actions.push(Action::Burn(card));
            }
            Event::BurnDone { seat } if seat != me => {
                let Some(&c) = world.burns.get(k) else {
                    return err("too few burns in the world");
                };
                k += 1;
                orig[usize::from(seat)] |= bit(c);
                actions.push(Action::Burn(c));
            }
            _ => {}
        }
    }
    if k != world.burns.len() {
        return err("too many burns in the world");
    }
    for e in evs {
        if let Event::Took { seat, card } = *e {
            if !has(orig[usize::from(seat)], card) {
                return err("a taken card is missing from the taker's cards");
            }
            orig[usize::from(seat)] &= !bit(card);
        }
    }
    if orig.iter().any(|&h| count(h) != 5) {
        return err("a seat would have been dealt other than five cards");
    }
    let mut pack: Vec<Card> = view.public.turned.iter().map(|&(c, _)| c).collect();
    let mut undealt = world.undealt;
    if let Some(n) = world.next_turned {
        if !has(undealt, n) || pack.len() != 1 || pack[0] != JOKER {
            return err("the next turned card is only drawn during the Joker burn");
        }
        undealt &= !bit(n);
        pack.push(n);
    }
    pack.extend(cards(undealt));
    let used = orig.iter().fold(0, |a, &h| a | h);
    if pack.len() != DECK_SIZE - 40
        || pack.iter().any(|&c| has(used, c))
        || pack.iter().fold(used, |a, &c| a | bit(c)) != ALL_CARDS
    {
        return err("the world's cards are not one pack");
    }
    let deck = match build_deck(start.dealer, &orig, &pack) {
        Ok(d) => d,
        Err(e) => return err(e),
    };
    let mut m = match Match::from_hand_start(start, deck, seed, rules) {
        Ok(m) => m,
        Err(e) => return err(e.0),
    };
    for a in actions {
        if m.apply(a).is_err() {
            return err("a recorded action is illegal in this world");
        }
    }
    Ok(m)
}

/// Does the rebuilt match show the viewer exactly the view it was built from (the strongest
/// consistency check, docs/M3-SPEC.md §4.6)?
pub fn same_view(m: &Match, view: &SeatView) -> bool {
    let v = m.view(view.seat);
    v.hand == view.hand
        && v.own_burns == view.own_burns
        && v.public == view.public
        && hand_events(&v) == hand_events(view)
}

/// Exactly uniform worlds by rejection, built on the engine (see the module docs).
#[derive(Clone, Copy, Debug, Default)]
pub struct RejectionSource {
    /// Give up after this many tries per world (a panic: the view must be consistent).
    pub max_tries: u64,
}

impl RejectionSource {
    /// The default source (up to 10 million tries per world).
    pub fn new() -> Self {
        RejectionSource {
            max_tries: 10_000_000,
        }
    }
}

struct RejectionWorlds<'a> {
    view: &'a SeatView,
    rules: Rules,
    hidden: Hidden,
    max_tries: u64,
    /// Facts placed directly (the ace of trumps when a seat robbed at its turn, the Joker after
    /// the dealer took it): each is a certainty, so fixing it keeps the draw uniform over the
    /// consistent worlds; the rejection test still checks everything.
    fixed: Vec<(u8, Card)>,
}

impl WorldSource for RejectionSource {
    fn name(&self) -> &str {
        "rejection"
    }

    fn prepare<'a>(&'a self, view: &'a SeatView, rules: Rules) -> Box<dyn Worlds + 'a> {
        let hidden = Hidden::of(view);
        let p = &view.public;
        let mut fixed = Vec::new();
        // A rob at turn: the robber holds the ace of trumps until it plays it (rules 20-21).
        if let (Some(r), Some(trump)) = (p.robber, p.trump) {
            let ace = engine::rules::ace_of_trumps(trump);
            if r != view.seat && has(hidden.unknown, ace) {
                // The robber either robbed with the ace (then holds it) or took a turned ace
                // (then the ace is the taken card, also in its hand: it can never burn it).
                fixed.push((r, ace));
            }
        }
        Box::new(RejectionWorlds {
            view,
            rules,
            hidden,
            max_tries: if self.max_tries == 0 {
                10_000_000
            } else {
                self.max_tries
            },
            fixed,
        })
    }
}

impl RejectionWorlds<'_> {
    fn deal(&self, rng: &mut Rng) -> World {
        let h = &self.hidden;
        let mut pool: Vec<Card> = cards(h.unknown).collect();
        let mut hands = [0 as Bits; 8];
        hands[usize::from(self.view.seat)] = self.view.hand;
        let mut sizes = h.hand_sizes;
        for &(s, c) in &self.fixed {
            hands[usize::from(s)] |= bit(c);
            sizes[usize::from(s)] -= 1;
            pool.retain(|&x| x != c);
        }
        rng.shuffle(&mut pool);
        let mut it = pool.into_iter();
        for s in 0..8 {
            for _ in 0..sizes[s] {
                hands[s] |= bit(it.next().expect("enough cards"));
            }
        }
        let burns: Vec<Card> = h
            .burn_seats
            .iter()
            .map(|_| it.next().expect("enough cards"))
            .collect();
        let rest: Vec<Card> = it.collect();
        let undealt = rest.iter().fold(0, |a, &c| a | bit(c));
        // The next turned card is a uniform card of the pack (deck position 41).
        let next_turned = h.joker_burn.then(|| rest[0]);
        World {
            hands,
            burns,
            undealt,
            next_turned,
        }
    }
}

impl Worlds for RejectionWorlds<'_> {
    fn count(&self) -> Option<u128> {
        None
    }

    fn sample(&self, rng: &mut Rng) -> World {
        for _ in 0..self.max_tries {
            let w = self.deal(rng);
            if let Ok(m) = rebuild(self.view, &w, self.rules, 0)
                && same_view(&m, self.view)
            {
                return w;
            }
        }
        panic!("no consistent world found by rejection: the view is inconsistent or too tight");
    }
}

/// The counted, exactly uniform world generator of `crates/belief` (docs/M3-SPEC.md §4): the
/// default source.
#[derive(Clone, Copy, Debug, Default)]
pub struct BeliefSource;

impl BeliefSource {
    /// The belief source.
    pub fn new() -> Self {
        BeliefSource
    }
}

struct BeliefWorlds {
    space: belief::WorldSpace,
    /// The other seats' burn locations, in the order the burns were made (`World::burns`).
    burn_locs: Vec<belief::Loc>,
    /// The dealer is burning after a turned Joker: each world also draws the card turned next.
    joker_burn: bool,
}

impl WorldSource for BeliefSource {
    fn name(&self) -> &str {
        "belief-uniform"
    }

    fn prepare<'a>(&'a self, view: &'a SeatView, _rules: Rules) -> Box<dyn Worlds + 'a> {
        Box::new(BeliefWorlds::of(view))
    }
}

impl BeliefWorlds {
    fn of(view: &SeatView) -> BeliefWorlds {
        let t = belief::Tracker::from_view(view);
        let space = belief::WorldSpace::new(&t)
            .expect("the hard facts never exclude the true deal (A6), so some world exists");
        let burn_locs = view
            .public
            .burn_order
            .iter()
            .filter(|&&(s, _)| s != view.seat)
            .map(|&(s, k)| belief::Loc::Burn(s, k))
            .collect();
        BeliefWorlds {
            space,
            burn_locs,
            joker_burn: view.public.trump.is_none(),
        }
    }
}

impl Worlds for BeliefWorlds {
    fn count(&self) -> Option<u128> {
        Some(self.space.count())
    }

    /// One world from the counted space; during the dealer's burn after a turned Joker, the card
    /// turned next is then drawn uniformly from the world's undealt cards with the same
    /// generator (§4.5: deck position 41 is a uniform position of the pack). Both depend only on
    /// the world's own generator, so world `w` is the same on any thread.
    fn sample(&self, rng: &mut Rng) -> World {
        let w = self.space.sample(rng);
        let burns = self
            .burn_locs
            .iter()
            .map(|l| {
                w.burns
                    .iter()
                    .find(|(x, _)| x == l)
                    .map(|&(_, c)| c)
                    .expect("one card per burn location")
            })
            .collect();
        let next_turned = self.joker_burn.then(|| w.draw_next_turned(rng));
        World {
            hands: w.hands,
            burns,
            undealt: w.undealt,
            next_turned,
        }
    }
}

/// The exact number of worlds consistent with a view's hard facts (`crates/belief`), for records
/// and the §13.1 world-count bands.
pub fn world_count(view: &SeatView) -> u128 {
    BeliefWorlds::of(view).space.count()
}

/// Is the phase a decision of the viewer?
pub fn is_my_turn(view: &SeatView) -> bool {
    match view.public.phase {
        Phase::AwaitBurn(s) | Phase::AwaitPlay(s) => s == view.seat,
        Phase::MatchOver => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bot::Bot;
    use bots::RulesBot;

    /// Play rules-v1 matches and check, at many decisions, that the true deal rebuilds to the
    /// same view, and that sampled worlds do too.
    #[test]
    fn true_world_rebuilds_and_samples_are_consistent() {
        let src = RejectionSource::new();
        let mut checked = 0;
        for seed in 0..12u64 {
            let mut m = Match::new(seed, Rules::HOUSE);
            let mut bot = RulesBot::new();
            bot.start_match(0, Rules::HOUSE, 0);
            let mut n = 0;
            while m.phase() != Phase::MatchOver && n < 200 {
                let seat = m.to_act().expect("to act");
                let view = m.view(seat);
                // The truth as a world.
                let t = m.truth();
                let hidden = Hidden::of(&view);
                let mut burns = Vec::new();
                let mut used = [0 as Bits; 8];
                for e in hand_events(&view) {
                    if let Event::BurnDone { seat: s } = *e
                        && s != seat
                    {
                        // Burn order per seat: the engine's burns are a set per seat; a seat
                        // burns twice only as the dealer after a Joker, in which case the
                        // first burn was made before the second turned card.
                        let left = t.burns[usize::from(s)] & !used[usize::from(s)];
                        let c = cards(left).next().expect("a burn");
                        used[usize::from(s)] |= bit(c);
                        burns.push(c);
                    }
                }
                let next_turned = hidden.joker_burn.then(|| t.deck[41]);
                let truth = World {
                    hands: t.hands,
                    burns,
                    undealt: t.undealt,
                    next_turned,
                };
                let dealer_twice = count(t.burns[usize::from(m.dealer())]) == 2;
                if !dealer_twice {
                    let r = rebuild(&view, &truth, Rules::HOUSE, 1).expect("truth rebuilds");
                    assert!(same_view(&r, &view));
                }
                if n % 17 == 0 {
                    let ws = src.prepare(&view, Rules::HOUSE);
                    let w = ws.sample(&mut world_rng(seed, n));
                    let r = rebuild(&view, &w, Rules::HOUSE, 1).expect("sample rebuilds");
                    assert!(same_view(&r, &view));
                    checked += 1;
                }
                let legal = m.legal_actions();
                let a = bot.act(&view, &legal);
                m.apply(a).expect("legal");
                n += 1;
            }
        }
        assert!(checked > 50);
    }
}
