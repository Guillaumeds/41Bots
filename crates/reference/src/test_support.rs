//! Helpers shared by the unit tests: card names, deck building, simple
//! deterministic players and a tiny test-only random-number generator.

use crate::card::{Card, all_cards};
use crate::game::{Action, Match};

/// The card with this canonical name.
pub fn card(name: &str) -> Card {
    Card::from_name(name).unwrap_or_else(|| panic!("bad card name {name:?}"))
}

/// The card index with this canonical name.
pub fn idx(name: &str) -> u8 {
    card(name).index()
}

/// Space-separated names to cards (order kept).
pub fn cards(names: &str) -> Vec<Card> {
    names.split_whitespace().map(card).collect()
}

/// Space-separated names to indices (order kept).
pub fn idxs(names: &str) -> Vec<u8> {
    names.split_whitespace().map(idx).collect()
}

/// Cards to space-separated names (order kept).
pub fn names(cards: &[Card]) -> String {
    let v: Vec<String> = cards.iter().map(|c| c.name()).collect();
    v.join(" ")
}

/// Builds a deck that follows the frozen deal convention (M0-SPEC section 1).
///
/// `hands[seat]` lists up to five cards for that seat (the first two go in the
/// two-card round). `turned` is position 40 (and 41 after a Joker). `pack`
/// starts the undealt pack. Missing hand cards are filled, in seat order 0..7,
/// with the lowest-indexed unused cards that are neither aces nor the Joker,
/// so an ace or the Joker is only ever dealt when a test names it.
pub fn deck(dealer: u8, hands: [&str; 8], turned: &str, pack: &str) -> [u8; 53] {
    let mut used = [false; 53];
    let mut mark = |i: u8| {
        assert!(
            !used[i as usize],
            "card {} named twice",
            crate::name_of_index(i)
        );
        used[i as usize] = true;
    };
    let mut hand_cards: Vec<Vec<u8>> = Vec::new();
    for h in hands {
        let v = idxs(h);
        assert!(v.len() <= 5, "too many cards for one hand: {h}");
        for &i in &v {
            mark(i);
        }
        hand_cards.push(v);
    }
    let turned = idxs(turned);
    assert!(!turned.is_empty() && turned.len() <= 2);
    for &i in &turned {
        mark(i);
    }
    let pack = idxs(pack);
    for &i in &pack {
        mark(i);
    }
    let mut filler: Vec<u8> = all_cards()
        .into_iter()
        .filter(|c| !used[c.index() as usize] && !c.is_ace() && *c != Card::Joker)
        .map(|c| c.index())
        .collect();
    for hand in hand_cards.iter_mut() {
        while hand.len() < 5 {
            let i = filler.remove(0);
            used[i as usize] = true;
            hand.push(i);
        }
    }

    let mut out = Vec::new();
    let mut slots = [0u8; 40];
    for i in 0..8u8 {
        let seat = ((dealer + 1 + i) % 8) as usize;
        let h = &hand_cards[seat];
        let i = i as usize;
        slots[2 * i] = h[0];
        slots[2 * i + 1] = h[1];
        slots[16 + 3 * i] = h[2];
        slots[16 + 3 * i + 1] = h[3];
        slots[16 + 3 * i + 2] = h[4];
    }
    out.extend_from_slice(&slots);
    out.extend_from_slice(&turned);
    out.extend_from_slice(&pack);
    for i in 0..53u8 {
        if !used[i as usize] {
            out.push(i);
        }
    }
    let deck: [u8; 53] = out.try_into().expect("53 cards");
    deck
}

/// Plays the first legal action whose card is in `prefer` (tried in order),
/// else the lowest legal action.
pub fn act_preferring(m: &mut Match, prefer: &[&str]) {
    let legal = m.legal_actions();
    assert!(!legal.is_empty(), "no legal action");
    for name in prefer {
        let i = idx(name);
        if let Some(a) = legal.iter().find(|a| a.card() == i) {
            m.apply(*a).unwrap();
            return;
        }
    }
    m.apply(legal[0]).unwrap();
}

/// Plays the lowest legal action.
pub fn act_first(m: &mut Match) {
    act_preferring(m, &[]);
}

/// Plays with `act_preferring` until a deck is needed or the match is over.
pub fn finish_hand_preferring(m: &mut Match, prefer: &[&str]) {
    while !m.needs_deck() && !m.match_over() {
        act_preferring(m, prefer);
    }
}

/// The legal actions' cards as names.
pub fn legal_names(m: &Match) -> String {
    let v: Vec<String> = m
        .legal_actions()
        .iter()
        .map(|a| crate::name_of_index(a.card()))
        .collect();
    v.join(" ")
}

/// The legal actions' kinds and cards, for asserting that a burn is due.
pub fn legal_burn_names(m: &Match) -> String {
    let mut v = Vec::new();
    for a in m.legal_actions() {
        match a {
            Action::Burn(c) => v.push(crate::name_of_index(c)),
            Action::Play(_) => panic!("expected only burns, got {a:?}"),
        }
    }
    v.join(" ")
}

/// A tiny xorshift64* generator, for tests only (integer arithmetic).
pub struct TestRng(u64);

impl TestRng {
    pub fn new(seed: u64) -> TestRng {
        TestRng((seed ^ 0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    pub fn deck(&mut self) -> [u8; 53] {
        let mut d: [u8; 53] = std::array::from_fn(|i| i as u8);
        for i in (1..53).rev() {
            let j = self.below(i + 1);
            d.swap(i, j);
        }
        d
    }
}
