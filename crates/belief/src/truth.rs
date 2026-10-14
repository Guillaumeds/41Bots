//! The true deal of a match in the tracker's terms (for tests and measurements; a bot never sees
//! it).

use engine::cards::{Bits, bit};
use engine::game::{Event, Match};

use crate::space::World;
use crate::tracker::Loc;

/// The true placement of the current hand: every seat's hand, every burn by location (all seats,
/// in burn order) and the undealt pack (which, during the dealer's burn after a turned Joker,
/// still holds the card turned next).
pub fn truth_placement(m: &Match) -> ([Bits; 8], Vec<(Loc, Bits)>, Bits) {
    let t = m.truth();
    let order = m.public().burn_order;
    let evs = m.events();
    let start = evs
        .iter()
        .rposition(|(_, e)| matches!(e, Event::HandStart { .. }))
        .unwrap_or(0);
    let burned: Vec<_> = evs[start..]
        .iter()
        .filter_map(|(_, e)| match *e {
            Event::Burned { seat, card } => Some((seat, card)),
            _ => None,
        })
        .collect();
    assert_eq!(burned.len(), order.len(), "one Burned event per burn");
    let burns = burned
        .iter()
        .zip(&order)
        .map(|(&(s, c), &(s2, k))| {
            assert_eq!(s, s2, "burn order agrees with the Burned events");
            (Loc::Burn(s, k), bit(c))
        })
        .collect();
    (t.hands, burns, t.undealt)
}

/// The true deal as a [`World`] for `viewer` (its own burns left out).
pub fn truth_world(m: &Match, viewer: u8) -> World {
    let (hands, burns, undealt) = truth_placement(m);
    let mut burns: Vec<_> = burns
        .into_iter()
        .filter(|&(l, _)| !matches!(l, Loc::Burn(s, _) if s == viewer))
        .map(|(l, b)| (l, b.trailing_zeros() as u8))
        .collect();
    burns.sort();
    World {
        hands,
        burns,
        undealt,
    }
}
