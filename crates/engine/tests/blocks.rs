//! Duplicate-block matches (docs/M1-SPEC.md §1–2): `Match::new_block`, `block_deck` and
//! `block_first_dealer`.

use engine::DECK_SIZE;
use engine::cards::Card;
use engine::game::{Match, Phase, Rules, block_deck, block_first_dealer};
use engine::rng::Rng;
use std::sync::{Arc, Mutex};

/// The block's match with the standard deck closure and first dealer.
fn block_match(block_seed: u64, rules: Rules) -> Match {
    Match::new_block(
        block_seed,
        move |k| block_deck(block_seed, k),
        block_first_dealer(block_seed),
        rules,
    )
}

/// Play to the end with `pick(legal_count, decision_index)` choosing the action, and return the
/// deck dealt in each hand, by hand index.
fn play_collecting_decks(m: &mut Match, pick: impl Fn(usize, u64) -> usize) -> Vec<[Card; 53]> {
    let mut decks = vec![m.truth().deck];
    let mut n = 0u64;
    while m.phase() != Phase::MatchOver {
        let legal = m.legal_actions();
        m.apply(legal[pick(legal.len(), n)]).unwrap();
        n += 1;
        let k = m.hand_index() as usize;
        if k == decks.len() {
            decks.push(m.truth().deck);
        }
        assert_eq!(decks.len(), k + 1, "hand indices count up one at a time");
    }
    decks
}

#[test]
fn matches_of_one_block_deal_identical_decks_for_every_hand_index() {
    for block_seed in [1u64, 2, 99, 0xDEAD_BEEF, u64::MAX] {
        let mut a = block_match(block_seed, Rules::HOUSE);
        let mut b = block_match(block_seed, Rules::HOUSE);
        assert_eq!(a.dealer(), block_first_dealer(block_seed));
        assert_eq!(b.dealer(), a.dealer());
        assert_eq!(a.seed(), block_seed);
        // Different play, so the matches last different numbers of hands and games.
        let da = play_collecting_decks(&mut a, |_, _| 0);
        let db = play_collecting_decks(&mut b, |len, n| {
            let mut r = Rng::derive(block_seed, &[n]);
            r.bounded(len as u64) as usize
        });
        for (k, (x, y)) in da.iter().zip(&db).enumerate() {
            assert_eq!(x, y, "block {block_seed}: hand {k} dealt different decks");
            assert_eq!(*x, block_deck(block_seed, k as u32));
        }
        for (k, d) in da.iter().chain(&db).enumerate() {
            let mut sorted = *d;
            sorted.sort_unstable();
            assert_eq!(sorted.to_vec(), (0..53u8).collect::<Vec<_>>(), "hand {k}");
        }
    }
}

#[test]
fn the_dealer_of_hand_k_is_the_same_in_every_match_of_a_block() {
    let block_seed = 7;
    let first = block_first_dealer(block_seed);
    let mut m = block_match(block_seed, Rules::FLAT);
    while m.phase() != Phase::MatchOver {
        assert_eq!(
            u32::from(m.dealer()),
            (u32::from(first) + m.hand_index()) % 8
        );
        let legal = m.legal_actions();
        m.apply(legal[legal.len() - 1]).unwrap();
    }
}

#[test]
fn block_deck_golden_values_for_block_seed_1() {
    // Computed independently in Python from the algorithms in tools/rng_golden.py plus
    // Rng::derive (SplitMix64 chaining of the labels), not by this engine.
    const HAND_0: [Card; DECK_SIZE] = [
        38, 19, 36, 16, 1, 26, 21, 27, 47, 32, 29, 9, 48, 8, 20, 3, 40, 7, 34, 33, 5, 35, 13, 49,
        14, 12, 11, 22, 43, 25, 24, 42, 39, 50, 41, 23, 15, 4, 44, 6, 18, 31, 46, 10, 51, 45, 37,
        17, 28, 2, 52, 30, 0,
    ];
    const HAND_1: [Card; DECK_SIZE] = [
        26, 22, 40, 38, 20, 10, 24, 39, 41, 30, 49, 18, 1, 47, 19, 25, 45, 48, 28, 8, 27, 34, 31,
        11, 21, 9, 36, 16, 33, 7, 2, 13, 29, 43, 46, 37, 32, 52, 17, 14, 5, 3, 50, 6, 4, 15, 42,
        44, 12, 0, 23, 35, 51,
    ];
    const HAND_2: [Card; DECK_SIZE] = [
        22, 8, 41, 49, 28, 14, 31, 21, 50, 5, 52, 7, 16, 20, 9, 25, 3, 4, 27, 51, 39, 23, 19, 12,
        15, 13, 38, 37, 17, 42, 33, 11, 26, 48, 35, 18, 2, 10, 30, 45, 24, 36, 29, 6, 40, 0, 34,
        47, 46, 43, 32, 44, 1,
    ];
    assert_eq!(block_deck(1, 0), HAND_0);
    assert_eq!(block_deck(1, 1), HAND_1);
    assert_eq!(block_deck(1, 2), HAND_2);
    assert_eq!(block_first_dealer(1), 1);
    // And the match deals them.
    let m = block_match(1, Rules::HOUSE);
    assert_eq!(m.dealer(), 1);
    assert_eq!(m.truth().deck, HAND_0);
}

#[test]
fn the_closure_is_asked_once_per_hand_in_hand_index_order() {
    let asked = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&asked);
    let mut m = Match::new_block(
        5,
        move |k| {
            log.lock().unwrap().push(k);
            block_deck(5, k)
        },
        3,
        Rules::HOUSE,
    );
    while m.phase() != Phase::MatchOver {
        let legal = m.legal_actions();
        m.apply(legal[0]).unwrap();
    }
    let asked = asked.lock().unwrap().clone();
    let hands = m.hand_index() + 1;
    assert_eq!(asked, (0..hands).collect::<Vec<u32>>());
    assert!(
        hands >= 2,
        "a match has at least two games, so at least two hands"
    );
}

#[test]
fn a_block_match_replays_from_its_actions_and_clones_compare_equal() {
    let block_seed = 11;
    let mut m = block_match(block_seed, Rules::HOUSE);
    let mut n = 0u64;
    while m.phase() != Phase::MatchOver {
        let legal = m.legal_actions();
        let i = Rng::derive(3, &[n]).bounded(legal.len() as u64) as usize;
        m.apply(legal[i]).unwrap();
        n += 1;
    }
    let mut again = block_match(block_seed, Rules::HOUSE);
    for &a in m.actions() {
        again.apply(a).unwrap();
    }
    assert_eq!(again.events(), m.events());
    assert_eq!(again.money(), m.money());
    // A clone shares the deck closure, so it is equal; an independent rebuild is not (closures
    // cannot be compared), though it holds the same state.
    let copy = m.clone();
    assert_eq!(copy, m);
    assert_ne!(again, m);
}

#[test]
#[should_panic(expected = "must hold each of the 53 cards exactly once")]
fn a_bad_deck_from_the_closure_is_refused() {
    let _ = Match::new_block(1, |_| [0; DECK_SIZE], 0, Rules::HOUSE);
}

#[test]
#[should_panic(expected = "no such dealer seat")]
fn a_bad_first_dealer_is_refused() {
    let _ = Match::new_block(1, |k| block_deck(1, k), 8, Rules::HOUSE);
}
