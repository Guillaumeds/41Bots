//! Tests of the card tracker and the world generator (docs/M3-SPEC.md §3.3, §4.6).
//!
//! - this file: shared helpers, the A6 property `tracker_never_excludes_truth` (random players;
//!   the `rules-v1` mixture runs in `crates/search`, which may depend on `bots`), the transcript
//!   deals and the forced rare branches, `tracker_incremental_equals_batch`,
//!   `tracker_capacity_sums` and `tracker_matches_engine_legality`;
//! - [`facts`]: one named test per fact F2–F12 on hand-made deals, with exact masks;
//! - [`worlds`]: counting, enumeration, the uniform sampler against the rejection oracle (χ²),
//!   pruning, thread independence and golden counts.
//!
//! Sizes: the unit tests stay at a few seconds each; the full spec sizes run with `--ignored`
//! (release mode recommended: `cargo test --release -p belief -- --ignored`).

mod facts;
mod worlds;

use engine::cards::{Bits, Card, bit, cards, count, has, name, parse};
use engine::game::{Action, Event, Match, Phase, Rules, build_deck};
use engine::rng::Rng;
use engine::rules::{
    EffSuit, effective_suit, is_top_trump, legal_plays, plain_mask, trump_mask, trump_rank,
};
use proptest::prelude::*;
use proptest::test_runner::RngSeed;

use crate::tracker::{hand_events, play_forbids};
use crate::truth::{truth_placement, truth_world};
use crate::{Tracker, WorldSpace};

/// A card by name.
pub(crate) fn c(s: &str) -> Card {
    parse(s).unwrap_or_else(|| panic!("no card {s}"))
}

/// A set of cards by names.
pub(crate) fn cs(s: &str) -> Bits {
    s.split_whitespace().fold(0, |a, x| a | bit(c(x)))
}

/// A match whose first hand deals `hands` (by seat) and puts `pack` from deck position 40 (the
/// turned card first); any card not named fills the hands and the pack (`build_deck`).
pub(crate) fn hand_made(dealer: u8, hands: [&str; 8], pack: &str) -> Match {
    let h: [Bits; 8] = core::array::from_fn(|s| cs(hands[s]));
    let p: Vec<Card> = pack.split_whitespace().map(c).collect();
    let deck = build_deck(dealer, &h, &p).expect("a valid hand-made deal");
    Match::with_decks(dealer, vec![deck], 7, Rules::HOUSE)
}

/// Apply actions written `P 2C` (play) or `B 3S` (burn), separated by commas.
pub(crate) fn act(m: &mut Match, acts: &str) {
    for a in acts.split(',').map(str::trim).filter(|a| !a.is_empty()) {
        let (k, card) = a.split_once(' ').expect("`P card` or `B card`");
        let action = match k {
            "P" => Action::Play(c(card)),
            "B" => Action::Burn(c(card)),
            _ => panic!("unknown action kind in `{a}`"),
        };
        m.apply(action)
            .unwrap_or_else(|e| panic!("`{a}` refused: {e:?}"));
    }
}

/// Play `m` to the end with uniformly random legal actions (from `seed`), calling `f` before
/// every action.
pub(crate) fn random_match(mut m: Match, seed: u64, mut f: impl FnMut(&Match)) {
    let mut rng = Rng::derive(seed, &[0xA11]);
    while m.phase() != Phase::MatchOver {
        f(&m);
        let legal = m.legal_actions();
        let a = legal[rng.bounded(legal.len() as u64) as usize];
        m.apply(a).expect("legal");
    }
}

/// Play the current hand of `m` to its end (or the match's) with uniformly random legal actions,
/// calling `f` before every action.
pub(crate) fn random_hand(mut m: Match, seed: u64, mut f: impl FnMut(&Match)) {
    let mut rng = Rng::derive(seed, &[0xA12]);
    let h = m.hand_index();
    while m.phase() != Phase::MatchOver && m.hand_index() == h {
        f(&m);
        let legal = m.legal_actions();
        let a = legal[rng.bounded(legal.len() as u64) as usize];
        m.apply(a).expect("legal");
    }
}

/// A readable account of a tracker against the truth, for failure messages.
pub(crate) fn describe(m: &Match, t: &Tracker) -> String {
    let (hands, burns, undealt) = truth_placement(m);
    let mut s = format!(
        "seat {} trump {:?} locs {:?}\nphase {:?}\n",
        t.seat,
        t.trump,
        t.locs,
        m.phase()
    );
    for card in cards(t.unknown) {
        let truly = (0..8u8)
            .find(|&x| has(hands[usize::from(x)], card))
            .map(|x| format!("Hand({x})"))
            .or_else(|| {
                burns
                    .iter()
                    .find(|(_, b)| has(*b, card))
                    .map(|(l, _)| format!("{l:?}"))
            })
            .unwrap_or_else(|| {
                if has(undealt, card) {
                    "Undealt".into()
                } else {
                    "?".into()
                }
            });
        s += &format!("  {}: {:?} (truly {truly})\n", name(card), t.places(card));
    }
    s
}

/// The A6 check at one moment of a match: every seat's tracker accepts the true deal, its
/// capacities hold exactly its unknown cards, and (for the seat to act) the world space is
/// feasible, counts at least one world and, when small, enumerates the true deal.
pub(crate) fn check_truth(m: &Match) {
    let (hands, burns, undealt) = truth_placement(m);
    for s in 0..8u8 {
        let t = Tracker::from_view(&m.view(s));
        assert!(
            t.consistent(&hands, &burns, undealt),
            "the tracker excludes the true deal\n{}",
            describe(m, &t)
        );
        assert_eq!(t.capacity_sum(), u32::from(count(t.unknown)));
        assert!(t.hidden_count == 12 || t.hidden_count == 13);
    }
    if let Some(s) = m.to_act() {
        let t = Tracker::from_view(&m.view(s));
        let ws = WorldSpace::new(&t).unwrap_or_else(|e| panic!("{e}\n{}", describe(m, &t)));
        assert!(ws.count() >= 1);
        // true_deal_has_positive_mass.
        if let Some(all) = ws.enumerate(10_000) {
            assert!(
                all.binary_search(&truth_world(m, s)).is_ok(),
                "the true deal is not among the enumerated worlds\n{}",
                describe(m, &t)
            );
        }
    }
}

/// The deals that force the rare rule branches (as in the difftest), each with a prefix of
/// actions where the branch needs a particular choice: (dealer, hands, pack, prefix).
pub(crate) fn rare_branches() -> Vec<(u8, [&'static str; 8], &'static str, &'static str)> {
    let none = [""; 8];
    let seat_holds = |seat: usize, card: &'static str| {
        let mut h = none;
        h[seat] = card;
        h
    };
    vec![
        // A turned Joker, then a turned ace: two at-once dealer burns.
        (0, none, "JK AD", ""),
        // A turned Joker, then an ordinary card; the dealer holds the new ace and robs as the
        // eighth card of trick 1.
        (3, seat_holds(3, "AD"), "JK 9D", ""),
        // The same, but the dealer burns the future ace of trumps at once (legal, H3): nobody
        // robs.
        (3, seat_holds(3, "AD"), "JK 9D", "B AD"),
        // A turned Joker, then an ordinary card, ace held by another seat or undealt.
        (6, none, "JK 9D", ""),
        (6, seat_holds(1, "AD"), "JK 9D", ""),
        // A turned ace of hearts: hearts are trumps, the dealer robs at once.
        (2, none, "AH", ""),
        // A turned plain ace.
        (4, none, "AC", ""),
        // The dealer holds the ace of an ordinary turned card.
        (5, seat_holds(5, "AS"), "9S", ""),
        // A turned 5 of trumps robbed by another seat.
        (1, seat_holds(4, "AC"), "5C", ""),
        // A turned Joker taken and burned at once by the dealer.
        (7, none, "JK 4H", "B JK"),
    ]
}

/// One rare-branch match: the forced deal, the prefix, then random play from `seed`.
pub(crate) fn rare_match((dealer, hands, pack, prefix): (u8, [&str; 8], &str, &str)) -> Match {
    let mut m = hand_made(dealer, hands, pack);
    act(&mut m, prefix);
    m
}

/// A transcript deal: file name, first dealer, decks, actions.
pub(crate) type TranscriptDeal = (String, u8, Vec<[Card; 53]>, Vec<Action>);

/// The transcript deals of `crates/engine/tests/transcripts` with their recorded actions.
pub(crate) fn transcripts() -> Vec<TranscriptDeal> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../engine/tests/transcripts");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("the engine's transcripts")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    files.sort();
    let mut out = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f).expect("readable");
        let mut dealer = None;
        // Deals in order: hands, then the pack (turned, second, undealt).
        let mut deals: Vec<([Option<Bits>; 8], Vec<Card>)> = Vec::new();
        let mut actions = Vec::new();
        for raw in text.lines() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if let Some(a) = line.strip_prefix('>') {
                let mut w = a.split_whitespace();
                let kind = w.next().expect("action kind");
                let card = c(w.next().expect("action card"));
                actions.push(match kind {
                    "Play" => Action::Play(card),
                    "Burn" => Action::Burn(card),
                    _ => panic!("unknown action {kind}"),
                });
                continue;
            }
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let key: Vec<&str> = key.split_whitespace().collect();
            let key: Vec<&str> = key.into_iter().filter(|&k| k != "then").collect();
            match key[..] {
                ["dealer"] => dealer = Some(value.trim().parse::<u8>().expect("seat")),
                ["hand", seat] => {
                    let seat: usize = seat.parse().expect("seat");
                    if deals.last().is_none_or(|d| d.0[seat].is_some()) {
                        deals.push(([None; 8], Vec::new()));
                    }
                    deals.last_mut().expect("a deal").0[seat] = Some(cs(value));
                }
                ["turned"] | ["second"] | ["undealt"] => {
                    let d = deals.last_mut().expect("hands come first");
                    d.1.extend(value.split_whitespace().map(c));
                }
                _ => {}
            }
        }
        let dealer = dealer.expect("a dealer line");
        let decks = deals
            .iter()
            .enumerate()
            .map(|(k, (h, pack))| {
                let hands = h.map(|x| x.unwrap_or(0));
                build_deck((dealer + k as u8) % 8, &hands, pack).expect("a valid transcript deal")
            })
            .collect();
        let file = f.file_name().expect("name").to_string_lossy().into_owned();
        out.push((file, dealer, decks, actions));
    }
    out
}

/// Run `f(i)` for `i` in `0..n` on all cores (test sweeps only; the crate itself has no threads).
pub(crate) fn on_threads(n: u64, f: impl Fn(u64) + Sync) {
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()) as u64;
    std::thread::scope(|s| {
        for k in 0..threads {
            let f = &f;
            s.spawn(move || {
                let mut i = k;
                while i < n {
                    f(i);
                    i += threads;
                }
            });
        }
    });
}

/// Number of matches for a full-size run, from an environment variable (for `--ignored` runs).
pub(crate) fn env_size(var: &str, default: u64) -> u64 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

// ----- tracker_never_excludes_truth (A6) -----

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 12,
        rng_seed: RngSeed::Fixed(0x41),
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    /// The A6 property on random matches: at every decision, every seat's tracker accepts the
    /// true deal (and the deciding seat's world space contains it).
    #[test]
    fn tracker_never_excludes_truth(seed in any::<u64>()) {
        random_match(Match::new(seed, Rules::HOUSE), seed, check_truth);
    }
}

/// The true deal has positive mass: the deciding seat's world space counts at least one world
/// and, whenever it holds at most 10^4, its enumeration contains the true deal (`check_truth`).
#[test]
fn true_deal_has_positive_mass() {
    let mut enumerated = 0;
    for seed in 100..106u64 {
        random_match(Match::new(seed, Rules::HOUSE), seed, |m| {
            check_truth(m);
            let t = Tracker::from_view(&m.view(m.to_act().expect("to act")));
            let ws = WorldSpace::new(&t).expect("feasible");
            if ws.count() <= 10_000 {
                enumerated += 1;
            }
        });
    }
    assert!(enumerated >= 50, "small spaces occur ({enumerated})");
}

#[test]
fn tracker_never_excludes_truth_on_transcripts_and_rare_branches() {
    for (file, dealer, decks, actions) in transcripts() {
        let mut m = Match::with_decks(dealer, decks, 3, Rules::HOUSE);
        for a in actions {
            if m.phase() == Phase::MatchOver {
                break;
            }
            check_truth(&m);
            m.apply(a)
                .unwrap_or_else(|e| panic!("{file}: {a:?} refused: {e:?}"));
        }
        // Then on to the end of that hand with random play.
        random_hand(m, 5, check_truth);
    }
    for (i, b) in rare_branches().into_iter().enumerate() {
        for seed in 0..8u64 {
            random_hand(rare_match(b), (i as u64) << 8 | seed, check_truth);
        }
    }
}

#[test]
#[ignore = "full size: BELIEF_MATCHES (default 20,000) random matches; run in release"]
fn tracker_never_excludes_truth_full() {
    let n = env_size("BELIEF_MATCHES", 20_000);
    on_threads(n, |seed| {
        random_match(Match::new(seed, Rules::HOUSE), seed, check_truth);
    });
    for (i, b) in rare_branches().into_iter().enumerate() {
        for seed in 0..500u64 {
            random_hand(rare_match(b), (i as u64) << 16 | seed, check_truth);
        }
    }
}

// ----- tracker_incremental_equals_batch -----

/// At every decision of a match, every seat's tracker kept with `new` + `update` per event
/// equals `from_view`.
fn incremental_equals_batch(m: Match, seed: u64) {
    // Per seat: the tracker, its hand index and how many hand events it has read.
    let mut kept: Vec<Option<(Tracker, u32, usize)>> = vec![None; 8];
    random_match(m, seed, |m| {
        for s in 0..8u8 {
            let v = m.view(s);
            let evs = hand_events(&v);
            let slot = &mut kept[usize::from(s)];
            if slot.as_ref().is_none_or(|k| k.1 != v.public.hand_index) {
                let t = Tracker::new(&v);
                *slot = Some((t, v.public.hand_index, Tracker::start_len(evs)));
            }
            let (t, _, read) = slot.as_mut().expect("set");
            for e in &evs[*read..] {
                t.update(e, &v);
            }
            *read = evs.len();
            assert_eq!(*t, Tracker::from_view(&v), "seat {s}");
        }
    });
}

#[test]
fn tracker_incremental_equals_batch() {
    for seed in 0..20u64 {
        incremental_equals_batch(Match::new(seed, Rules::HOUSE), seed);
    }
    for (i, b) in rare_branches().into_iter().enumerate() {
        incremental_equals_batch(rare_match(b), i as u64);
    }
}

#[test]
#[ignore = "full size: BELIEF_MATCHES_INC (default 2,000) matches"]
fn tracker_incremental_equals_batch_full() {
    on_threads(env_size("BELIEF_MATCHES_INC", 2_000), |seed| {
        incremental_equals_batch(Match::new(seed, Rules::HOUSE), seed);
    });
}

// ----- tracker_capacity_sums -----

#[test]
fn tracker_capacity_sums() {
    let mut seen = [false; 2];
    for seed in 0..30u64 {
        random_match(Match::new(seed, Rules::HOUSE), seed, |m| {
            for s in 0..8u8 {
                let v = m.view(s);
                let t = Tracker::from_view(&v);
                assert_eq!(t.capacity_sum(), u32::from(count(t.unknown)));
                assert!(t.hidden_count == 12 || t.hidden_count == 13);
                seen[usize::from(t.hidden_count - 12)] = true;
                // Undealt plus the other seats' burns plus the viewer's own burns.
                let out_of_play: u32 = t
                    .locs
                    .iter()
                    .filter(|(l, _)| !matches!(l, crate::Loc::Hand(_)))
                    .map(|&(_, c)| u32::from(c))
                    .sum::<u32>()
                    + u32::from(count(v.own_burns));
                assert_eq!(out_of_play, u32::from(t.hidden_count));
                // The known cards are what the view shows.
                assert_eq!(t.known & t.unknown, 0);
                assert_eq!(t.known | t.unknown, engine::cards::ALL_CARDS);
                assert_eq!(t.known & v.hand, v.hand);
            }
        });
    }
    assert_eq!(seen, [true, true], "both 12 and 13 hidden cards occur");
}

// ----- tracker_matches_engine_legality -----

/// F2 and F3 by the rule text (rules 14–18), written independently of `legal_plays`.
fn forbids_by_rule_text(card: Card, led: Card, trump: engine::cards::Suit) -> Bits {
    match effective_suit(led, trump) {
        EffSuit::Plain(s) => match effective_suit(card, trump) {
            EffSuit::Plain(t) if t != s => plain_mask(s, trump),
            _ => 0,
        },
        EffSuit::Trump => match effective_suit(card, trump) {
            EffSuit::Trump => 0,
            EffSuit::Plain(_) => {
                let l = trump_rank(led, trump);
                cards(trump_mask(trump))
                    .filter(|&t| !(is_top_trump(t, trump) && trump_rank(t, trump) > l))
                    .fold(0, |a, t| a | bit(t))
            }
        },
    }
}

#[test]
fn tracker_matches_engine_legality() {
    let mut facts_seen = 0;
    for seed in 0..40u64 {
        let mut m = Match::new(seed, Rules::HOUSE);
        let mut rng = Rng::new(seed);
        while m.phase() != Phase::MatchOver {
            let seat = m.to_act().expect("to act");
            let p = m.public();
            let before = m.truth().hands[usize::from(seat)];
            let led = p.trick_cards.first().map(|&(_, c)| c);
            let legal = m.legal_actions();
            let a = legal[rng.bounded(legal.len() as u64) as usize];
            m.apply(a).expect("legal");
            let (Action::Play(card), Some(led), Some(trump)) = (a, led, p.trump) else {
                continue;
            };
            // The seat's actual hand made its play legal under the engine's function.
            assert!(has(legal_plays(before, Some(led), trump), card));
            // The fact equals the rule text, and the seat's remaining hand never breaks it.
            let f = play_forbids(card, led, trump);
            assert_eq!(f, forbids_by_rule_text(card, led, trump));
            assert_eq!(f & (before & !bit(card)), 0);
            // A hand avoiding the fact makes the play legal: add any one allowed card.
            for x in cards(engine::cards::ALL_CARDS & !f & !bit(card) & !bit(led)) {
                assert!(has(legal_plays(bit(card) | bit(x), Some(led), trump), card));
            }
            if f != 0 {
                facts_seen += 1;
                // Other seats' trackers carry the fact (unless the hand moved on).
                if m.public().hand_index == p.hand_index {
                    let other = (seat + 1) % 8;
                    let t = Tracker::from_view(&m.view(other));
                    assert_eq!(t.hand_forbid(seat) & f, f);
                }
            }
        }
    }
    assert!(facts_seen > 100, "voids and trump limits occur");
}

/// The `AceUndealt` and `ProvedNoAce` events the tracker reads appear where the rules say.
#[test]
fn ace_events_follow_trick_one() {
    for seed in 0..20u64 {
        random_match(Match::new(seed, Rules::HOUSE), seed, |m| {
            let v = m.view(0);
            let evs = hand_events(&v);
            let undealt = evs.iter().any(|e| matches!(e, Event::AceUndealt));
            if undealt {
                assert!(v.public.robber.is_none());
                assert!(v.public.trick_no >= 2 || v.public.phase == Phase::MatchOver);
            }
        });
    }
}
