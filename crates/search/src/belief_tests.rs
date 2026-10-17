//! The belief world source inside the search (docs/M3-SPEC.md §3.3, §4.6, §5.1):
//! - `world_rebuild_view_identity`: every sampled world, rebuilt as a match, shows the deciding
//!   seat exactly its real view (the strongest consistency check);
//! - `tracker_never_excludes_truth_rules_v1`: the A6 property with `rules-v1` and random players
//!   mixed at the table (the belief crate cannot use `bots`);
//! - `belief_equals_engine_oracle`: on small late positions, the worlds the tracker allows are
//!   exactly the placements the engine rebuilds into the same view (no fact missing, none wrong);
//! - the exact world count is exposed to records and the §13.1 bands.

use belief::truth::truth_placement;
use belief::{Loc, Tracker, WorldSpace};
use engine::cards::{Bits, Card, bit, cards};
use engine::game::{Action, Lapse, Match, Phase, Rules, build_deck};
use engine::rng::Rng;

use crate::pimc::{Ctx, SearchConfig, evaluate};
use crate::worlds::{
    BeliefSource, RejectionSource, World, WorldSource, rebuild, same_view, world_count, world_rng,
};

/// Seats played by `rules-v1` in a mixed match: the bits of `mask`.
fn mixed_action(m: &Match, mask: u8, rng: &mut Rng) -> Action {
    let seat = m.to_act().expect("to act");
    let legal = m.legal_actions();
    if mask & (1 << seat) != 0 {
        bots::decide(&m.view(seat), &Rules::HOUSE, &legal).0
    } else {
        legal[rng.bounded(legal.len() as u64) as usize]
    }
}

/// Play `m` to the end with a mixture of `rules-v1` and random seats, calling `f` before every
/// action.
fn mixed_match(mut m: Match, seed: u64, mut f: impl FnMut(&Match)) {
    let mut rng = Rng::derive(seed, &[0x313]);
    let mask = (seed % 256) as u8;
    while m.phase() != Phase::MatchOver {
        f(&m);
        let a = mixed_action(&m, mask, &mut rng);
        m.apply(a).expect("legal");
    }
}

/// Hand-made first deals that force the rare branches (turned Joker, Joker then ace, the dealer
/// holding the new ace, the dealer burning the future ace, a turned A♥).
fn rare_matches() -> Vec<Match> {
    let ad = engine::cards::card(engine::cards::Suit::Diamonds, 1);
    let nine_d = engine::cards::card(engine::cards::Suit::Diamonds, 9);
    let ah = engine::cards::card(engine::cards::Suit::Hearts, 1);
    let joker = engine::cards::JOKER;
    let with = |seat: usize, c: Card| {
        let mut h = [0 as Bits; 8];
        h[seat] = bit(c);
        h
    };
    let deal = |dealer: u8, hands: [Bits; 8], pack: &[Card]| {
        let deck = build_deck(dealer, &hands, pack).expect("valid deal");
        Match::with_decks(dealer, vec![deck], 11, Rules::HOUSE)
    };
    let mut burned_ace = deal(3, with(3, ad), &[joker, nine_d]);
    burned_ace
        .apply(Action::Burn(ad))
        .expect("legal: trumps unknown");
    vec![
        deal(0, [0; 8], &[joker, ad]),
        deal(3, with(3, ad), &[joker, nine_d]),
        burned_ace,
        deal(6, with(1, ad), &[joker, nine_d]),
        deal(2, [0; 8], &[ah]),
    ]
}

/// Sample `k` belief worlds of the deciding seat's view and check each rebuilds to that view.
fn check_rebuild(m: &Match, k: u64, dseed: u64) {
    let seat = m.to_act().expect("to act");
    let view = m.view(seat);
    let src = BeliefSource::new();
    let worlds = src.prepare(&view, Rules::HOUSE);
    assert_eq!(worlds.count(), Some(world_count(&view)));
    for w in 0..k {
        let world = worlds.sample(&mut world_rng(dseed, w));
        let r = rebuild(&view, &world, Rules::HOUSE, dseed ^ w)
            .unwrap_or_else(|e| panic!("world {w} does not rebuild: {e:?}"));
        assert!(same_view(&r, &view), "world {w} rebuilds to another view");
    }
}

#[test]
fn world_rebuild_view_identity() {
    let mut decisions = 0;
    let mut seed = 0;
    while decisions < 2_000 {
        mixed_match(Match::new(seed, Rules::HOUSE), seed, |m| {
            if decisions < 2_000 {
                check_rebuild(m, 2, seed);
                decisions += 1;
            }
        });
        seed += 1;
    }
    // Every decision of the rare branches' first hands, the dealer's burns included.
    let mut joker_burns = 0;
    for (i, m) in rare_matches().into_iter().enumerate() {
        let h = m.hand_index();
        let mut m = m;
        let mut rng = Rng::new(i as u64);
        while m.phase() != Phase::MatchOver && m.hand_index() == h {
            if m.public().trump.is_none() {
                joker_burns += 1;
            }
            check_rebuild(&m, 4, i as u64);
            let a = mixed_action(&m, 0x0F, &mut rng);
            m.apply(a).expect("legal");
        }
    }
    assert!(
        joker_burns >= 3,
        "the dealer's burn after a turned Joker is covered"
    );
}

#[test]
fn tracker_never_excludes_truth_rules_v1() {
    let check = |m: &Match| {
        let (hands, burns, undealt) = truth_placement(m);
        for s in 0..8 {
            let t = Tracker::from_view(&m.view(s));
            assert!(t.consistent(&hands, &burns, undealt), "seat {s}");
        }
    };
    for seed in 0..24u64 {
        mixed_match(Match::new(seed * 7 + 1, Rules::HOUSE), seed, check);
    }
    for m in rare_matches() {
        mixed_match(m, 0xFF, check);
    }
}

#[test]
#[ignore = "full size: SEARCH_MATCHES_A6 (default 20,000) mixed matches; run in release"]
fn tracker_never_excludes_truth_rules_v1_full() {
    let n: u64 = std::env::var("SEARCH_MATCHES_A6")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()) as u64;
    std::thread::scope(|sc| {
        for k in 0..threads {
            sc.spawn(move || {
                for seed in (k..n).step_by(threads as usize) {
                    mixed_match(Match::new(seed, Rules::HOUSE), seed, |m| {
                        let (hands, burns, undealt) = truth_placement(m);
                        for s in 0..8 {
                            let t = Tracker::from_view(&m.view(s));
                            assert!(
                                t.consistent(&hands, &burns, undealt),
                                "match {seed} seat {s}"
                            );
                        }
                    });
                }
            });
        }
    });
}

/// Every placement of the unknown cards by capacity, as a search world (burns in burn order).
fn all_placements(t: &Tracker, burn_locs: &[Loc]) -> Vec<World> {
    fn rec(
        t: &Tracker,
        burn_locs: &[Loc],
        i: usize,
        left: Bits,
        parts: &mut Vec<Bits>,
        out: &mut Vec<World>,
    ) {
        if i == t.locs.len() {
            let mut hands = [0 as Bits; 8];
            hands[usize::from(t.seat)] = t.hand;
            let mut undealt = 0;
            for (&(loc, _), &set) in t.locs.iter().zip(parts.iter()) {
                match loc {
                    Loc::Hand(s) => hands[usize::from(s)] = set,
                    Loc::Undealt => undealt = set,
                    Loc::Burn(..) => {}
                }
            }
            let burns = burn_locs
                .iter()
                .map(|l| {
                    let i = t.loc_index(*l).expect("a burn location");
                    cards(parts[i]).next().expect("one card")
                })
                .collect();
            out.push(World {
                hands,
                burns,
                undealt,
                next_turned: None,
            });
            return;
        }
        fn subsets(pool: &[Card], k: usize, from: usize, acc: Bits, f: &mut dyn FnMut(Bits)) {
            if k == 0 {
                f(acc);
                return;
            }
            for i in from..pool.len() {
                if pool.len() - i < k {
                    break;
                }
                subsets(pool, k - 1, i + 1, acc | bit(pool[i]), f);
            }
        }
        let pool: Vec<Card> = cards(left).collect();
        subsets(&pool, usize::from(t.locs[i].1), 0, 0, &mut |set| {
            parts.push(set);
            rec(t, burn_locs, i + 1, left & !set, parts, out);
            parts.pop();
        });
    }
    let mut out = Vec::new();
    rec(t, burn_locs, 0, t.unknown, &mut Vec::new(), &mut out);
    out
}

/// Is this position small enough for the oracle (at most `max_placements` placements by
/// capacity)? Then: the tracker's worlds are exactly the placements the engine rebuilds into the
/// seat to act's view. Returns whether the position was checked and whether it had hidden burns.
fn oracle_check(m: &Match, max_placements: u128, what: &str) -> Option<bool> {
    let p = m.public();
    let seat = m.to_act()?;
    let view = m.view(seat);
    let t = Tracker::from_view(&view);
    let mut left = engine::cards::count(t.unknown);
    let mut placements: u128 = 1;
    for &(_, cap) in &t.locs {
        placements *= belief::space::binom(left, cap);
        left -= cap;
    }
    if placements > max_placements {
        return None;
    }
    let burn_locs: Vec<Loc> = p
        .burn_order
        .iter()
        .filter(|&&(s, _)| s != seat)
        .map(|&(s, k)| Loc::Burn(s, k))
        .collect();
    let mut engine_ok: Vec<World> = all_placements(&t, &burn_locs)
        .into_iter()
        .filter(|w| rebuild(&view, w, Rules::HOUSE, 1).is_ok_and(|r| same_view(&r, &view)))
        .collect();
    engine_ok.sort();
    let ws = WorldSpace::new(&t).expect("feasible");
    let mut belief_ok: Vec<World> = ws
        .enumerate(1_000_000)
        .expect("small")
        .into_iter()
        .map(|w| World {
            hands: w.hands,
            burns: burn_locs
                .iter()
                .map(|l| w.burns.iter().find(|(x, _)| x == l).expect("burn").1)
                .collect(),
            undealt: w.undealt,
            next_turned: None,
        })
        .collect();
    belief_ok.sort();
    assert_eq!(
        belief_ok.len(),
        engine_ok.len(),
        "{what}: tracker {} worlds, engine {}",
        belief_ok.len(),
        engine_ok.len()
    );
    assert_eq!(belief_ok, engine_ok, "{what}");
    Some(!burn_locs.is_empty())
}

/// On small late positions: the tracker's worlds are exactly the placements the engine rebuilds
/// into the same view.
fn belief_equals_engine_oracle_on(n_positions: usize, max_placements: u128, seed0: u64) {
    let mut done = 0;
    let mut seed = seed0;
    let mut with_burns = 0;
    while done < n_positions {
        mixed_match(Match::new(seed, Rules::HOUSE), seed, |m| {
            let p = m.public();
            if done >= n_positions || p.trick_no < 5 || p.trick_cards.len() < 4 {
                return;
            }
            if let Some(burns) = oracle_check(m, max_placements, &format!("match {seed}")) {
                with_burns += usize::from(burns);
                done += 1;
            }
        });
        seed += 1;
    }
    assert!(with_burns > 0, "positions with hidden burns are covered");
}

#[test]
fn belief_equals_engine_oracle() {
    belief_equals_engine_oracle_on(40, 3_000, 0);
}

#[test]
#[ignore = "full size: 500 positions with up to 50,000 placements each"]
fn belief_equals_engine_oracle_full() {
    belief_equals_engine_oracle_on(500, 50_000, 10_000);
}

#[test]
fn rejection_and_belief_sources_agree_on_the_world_set() {
    // Both sources draw only consistent worlds; the belief one also counts them.
    let mut m = Match::new(5, Rules::HOUSE);
    let mut rng = Rng::new(5);
    let mut checked = 0;
    while m.phase() != Phase::MatchOver && checked < 30 {
        let seat = m.to_act().expect("to act");
        let view = m.view(seat);
        if view.public.trick_no >= 3 {
            let (bs, rs) = (BeliefSource::new(), RejectionSource::new());
            let b = bs.prepare(&view, Rules::HOUSE);
            let r = rs.prepare(&view, Rules::HOUSE);
            assert!(b.count().is_some_and(|n| n >= 1));
            assert_eq!(r.count(), None);
            let t = Tracker::from_view(&view);
            for w in 0..3 {
                for world in [
                    b.sample(&mut world_rng(9, w)),
                    r.sample(&mut world_rng(9, w)),
                ] {
                    let rb = rebuild(&view, &world, Rules::HOUSE, 0).expect("rebuilds");
                    assert!(same_view(&rb, &view));
                    let (hands, burns, undealt) = truth_placement(&rb);
                    assert!(t.consistent(&hands, &burns, undealt));
                }
            }
            checked += 1;
        }
        let a = mixed_action(&m, 0xAA, &mut rng);
        m.apply(a).expect("legal");
    }
    assert_eq!(checked, 30);
}

#[test]
fn decisions_record_the_exact_world_count() {
    use std::sync::Arc;
    let leaf: Arc<crate::pimc::SharedLeaf> = Arc::new(solver::FnLeaf(|_: &equity::MatchState| {
        equity::MoneyChance::even()
    }));
    let src = BeliefSource::new();
    let ctx = Ctx {
        leaf: &*leaf,
        source: &src,
        rules: Rules::HOUSE,
    };
    let cfg = SearchConfig {
        w12: 2,
        w34: 2,
        threads: 1,
        tt_mb: 4,
        ..SearchConfig::default()
    };
    let mut m = Match::new(8, Rules::HOUSE);
    let mut rng = Rng::new(8);
    let mut seen = 0;
    while m.phase() != Phase::MatchOver && seen < 6 {
        let seat = m.to_act().expect("to act");
        let view = m.view(seat);
        let legal = m.legal_actions();
        if legal.len() >= 2 && view.public.trick_no >= 3 {
            let dv = evaluate(&view, &legal, 1, &ctx, &cfg);
            assert_eq!(dv.world_count, Some(world_count(&view)));
            assert!(belief::log2_q8(dv.world_count.expect("counted")) > 0);
            seen += 1;
        }
        let a = mixed_action(&m, 0, &mut rng);
        m.apply(a).expect("legal");
    }
    assert_eq!(seen, 6);
}

/// M3-SPEC §2.1 with `rules-v1` players (the engine's own test uses a fixed pick rule, since the
/// engine cannot depend on `bots`): for 500 seeds, play to the start of some hand, rebuild the
/// match there with `Match::from_hand_start` from the hand's public start state and deck, play
/// both on: every event and every seat's view agree.
#[test]
fn from_hand_start_equals_played_match_rules_v1() {
    use engine::game::current_hand_start_of;
    for seed in 0..500u64 {
        let mut m = Match::new(seed, Rules::HOUSE);
        let mut rng = Rng::new(seed);
        let mask = (seed * 37 % 256) as u8;
        let target = (seed % 7) as u32;
        while m.hand_index() < target && m.phase() != Phase::MatchOver {
            let a = mixed_action(&m, mask, &mut rng);
            m.apply(a).expect("legal");
        }
        if m.phase() == Phase::MatchOver {
            continue;
        }
        let start = current_hand_start_of(&m.view(0));
        let mut r = Match::from_hand_start(start, m.truth().deck, seed ^ 5, Rules::HOUSE)
            .expect("a reachable start");
        let h = m.hand_index();
        let tail = |v: &engine::view::SeatView| crate::worlds::hand_events(v).to_vec();
        while m.hand_index() == h && m.phase() != Phase::MatchOver {
            for s in 0..8 {
                let (a, b) = (m.view(s), r.view(s));
                assert_eq!(tail(&a), tail(&b), "seed {seed} seat {s}");
                assert_eq!(
                    (a.public, a.hand, a.own_burns),
                    (b.public, b.hand, b.own_burns)
                );
            }
            let a = mixed_action(&m, mask, &mut rng);
            m.apply(a).expect("legal");
            r.apply(a).expect("legal in the rebuilt match");
        }
        assert_eq!(m.scores(), r.scores());
        assert_eq!(m.games_won(), r.games_won());
    }
}

// ----- the app's referee: turned partnerships and reneges let through (APP-ENGINE-CONTRACT §7) -----

/// Play match `seed` with random legal cards and, now and then, the app's referee steps (turns
/// before a play, with a play and while a rob is due; reneges let through, caught later),
/// calling `f` before every step. Stops early when every card of the seat to act is refused
/// (a renege it proves would turn the partnership that led the trick: the open question).
fn referee_match(seed: u64, mut f: impl FnMut(&Match)) {
    let mut rng = Rng::derive(seed, &[0x7E5]);
    let mut m = Match::new(seed, Rules::HOUSE);
    while m.phase() != Phase::MatchOver {
        f(&m);
        let seat = m.to_act().expect("to act");
        let hand: Vec<Card> = cards(m.truth().hands[usize::from(seat)]).collect();
        let card = hand[rng.bounded(hand.len() as u64) as usize];
        let team = rng.bounded(4) as u8;
        let legal = m.legal_actions();
        let pick = legal[rng.bounded(legal.len() as u64) as usize].card();
        let stepped = match (m.phase(), rng.bounded(16)) {
            (Phase::AwaitBurn(_), 0) => m.turn_team(engine::team_of(seat)).is_ok(),
            (Phase::AwaitBurn(_), 1) => m.let_through(Lapse::BurnAce).is_ok(),
            (Phase::AwaitBurn(_), 2) => m.let_through(Lapse::PlayWithoutRob(card)).is_ok(),
            (Phase::AwaitPlay(_), 0) => m.turn_team(team).is_ok(),
            (Phase::AwaitPlay(_), 1) => m.play_then_turn(pick, team).is_ok(),
            (Phase::AwaitPlay(_), 2 | 3) => m.let_through(Lapse::Play(card)).is_ok(),
            (Phase::AwaitPlay(_), 4) => m.let_through(Lapse::FalseRob(card)).is_ok(),
            _ => false,
        };
        if stepped {
            continue;
        }
        let first = rng.bounded(legal.len() as u64) as usize;
        let moved = (0..legal.len()).any(|i| m.apply(legal[(first + i) % legal.len()]).is_ok())
            || m.turn_team(engine::team_of(seat)).is_ok()
            || hand.iter().any(|&c| m.let_through(Lapse::Play(c)).is_ok());
        if !moved {
            return;
        }
    }
}

/// Where two views of the same seat first differ (for test messages).
fn view_difference(a: &engine::view::SeatView, b: &engine::view::SeatView) -> String {
    let ea: Vec<String> = crate::worlds::hand_events(a)
        .iter()
        .map(|e| e.canonical())
        .collect();
    let eb: Vec<String> = crate::worlds::hand_events(b)
        .iter()
        .map(|e| e.canonical())
        .collect();
    let i = ea.iter().zip(&eb).take_while(|(x, y)| x == y).count();
    format!(
        "events: rebuilt {:?} | real {:?} at #{i}\nreal events:\n{}\nhand {} {}, burns {} {}\npublic equal: {}\nrebuilt public {:?}\nreal public {:?}",
        ea.get(i),
        eb.get(i),
        eb.join("\n"),
        engine::cards::names(a.hand),
        engine::cards::names(b.hand),
        engine::cards::names(a.own_burns),
        engine::cards::names(b.own_burns),
        a.public == b.public,
        a.public,
        b.public
    )
}

/// The true deal of the current hand as a search world for the seat to act.
fn truth_as_world(m: &Match, viewer: u8) -> World {
    let (hands, burns, undealt) = truth_placement(m);
    let burns = burns
        .into_iter()
        .filter(|&(l, _)| !matches!(l, Loc::Burn(s, _) if s == viewer))
        .map(|(_, b)| cards(b).next().expect("one card a burn"))
        .collect();
    let joker_burn = m.public().trump.is_none();
    World {
        hands,
        burns,
        undealt,
        next_turned: joker_burn.then(|| m.truth().deck[41]),
    }
}

/// Views after the app's referee (docs/APP-ENGINE-CONTRACT.md §7): a turned seat's show-outs and
/// "no ace" proofs prove nothing, so a sampled world gives each turned seat exactly its number
/// of face-down cards, of any suit; every sampled world rebuilds as a match that shows the seat
/// its own view; the tracker never excludes the true deal once every renege's partnership is
/// turned (a renege still hidden is, rightly, invisible to it), and the true deal rebuilds too.
#[test]
fn referee_views_rebuild_and_turned_seats_have_no_voids() {
    let src = BeliefSource::new();
    let (mut checked, mut truths, mut void_cards, mut turned_cards) = (0, 0, 0u32, 0u32);
    let seeds: u64 = std::env::var("REFEREE_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    for seed in 0..seeds {
        let mut n = 0u64;
        referee_match(seed, |m| {
            let p = m.public();
            let turned = p.turned_teams;
            if !turned.iter().any(|&t| t) && m.reneges().is_empty() {
                return;
            }
            n += 1;
            if !n.is_multiple_of(3) {
                return;
            }
            let seat = m.to_act().expect("to act");
            let view = m.view(seat);
            let t = Tracker::from_view(&view);
            if m.reneges()
                .iter()
                .all(|r| turned[usize::from(engine::team_of(r.seat))])
            {
                let (hands, burns, undealt) = truth_placement(m);
                assert!(
                    t.consistent(&hands, &burns, undealt),
                    "match {seed}: the tracker excludes the true deal"
                );
                let r =
                    rebuild(&view, &truth_as_world(m, seat), Rules::HOUSE, 1).unwrap_or_else(|e| {
                        panic!("match {seed}: the true deal does not rebuild: {e:?}")
                    });
                assert!(
                    same_view(&r, &view),
                    "match {seed}: the true deal rebuilds to another view"
                );
                truths += 1;
            }
            let ws = src.prepare(&view, Rules::HOUSE);
            for w in 0..2 {
                let world = ws.sample(&mut world_rng(seed ^ n, w));
                let r = rebuild(&view, &world, Rules::HOUSE, 1).unwrap_or_else(|e| {
                    let evs: Vec<String> = crate::worlds::hand_events(&view)
                        .iter()
                        .map(|e| e.canonical())
                        .collect();
                    panic!(
                        "match {seed}: world {w} does not rebuild: {e:?}\nviewer {seat}\n{}\n{world:?}\nreneges {:?}\ntruth {:?}",
                        evs.join("\n"),
                        m.reneges(),
                        m.truth().hands
                    )
                });
                assert!(
                    same_view(&r, &view),
                    "match {seed}: world {w} rebuilds to another view\n{}",
                    view_difference(&r.view(seat), &view)
                );
                for s in (0..8u8).filter(|&s| s != seat && turned[usize::from(engine::team_of(s))])
                {
                    let h = world.hands[usize::from(s)];
                    assert_eq!(engine::cards::count(h), p.hand_sizes[usize::from(s)]);
                    turned_cards += u32::from(engine::cards::count(h));
                    void_cards += engine::cards::count(h & t.hand_forbid(s)) as u32;
                }
            }
            checked += 1;
        });
    }
    assert!(
        checked > 300 && truths > 100,
        "checked {checked}, truths {truths}"
    );
    assert!(
        turned_cards > 1000 && void_cards > 20,
        "turned seats' face-down cards {turned_cards}, of a suit they showed out of {void_cards}"
    );
}

/// The rule hand of the engine's turned-pair tests: seat 1 plays the K♥ to a club lead holding
/// two clubs (let through), proves it with its Q♣ and is turned. To the others its show-out
/// proves nothing now: its three face-down cards may include the 4♣ it really holds.
#[test]
fn a_turned_seats_show_out_proves_nothing() {
    let hands = [
        "KC 5S AD 2D 3D",
        "4C QC KH QH KD",
        "2C JC JH 10H QD",
        "5C 9S 9H 8H JD",
        "3C AC 4D 5D 6D",
        "6C 9C 7H 6H 10D",
        "7C KS 5H 4H 9D",
        "8C 10C 3H 2H 8D",
    ];
    let set = |s: &str| {
        s.split_whitespace()
            .map(|n| engine::cards::parse(n).expect("card"))
            .fold(0, |a, c| a | bit(c))
    };
    let pack: Vec<Card> = ["7S", "AS"]
        .iter()
        .map(|n| engine::cards::parse(n).expect("card"))
        .collect();
    let deck = build_deck(7, &hands.map(set), &pack).expect("deal");
    let mut m = Match::with_decks(7, vec![deck], 1, Rules::HOUSE);
    let c = |n: &str| engine::cards::parse(n).expect("card");
    m.apply(Action::Play(c("KC"))).expect("legal");
    m.let_through(Lapse::Play(c("KH"))).expect("let through");
    for n in ["2C", "5C", "3C", "6C", "7C", "8C", "5S", "QC"] {
        m.apply(Action::Play(c(n))).expect("legal");
    }
    assert_eq!(m.turned_teams(), [false, true, false, false]);
    let view = m.view(2);
    let t = Tracker::from_view(&view);
    assert!(
        has_bit(t.hand_forbid(1), c("4C")),
        "the fact is still recorded"
    );
    assert!(
        t.places(c("4C")).contains(&Loc::Hand(1)),
        "but proves nothing now"
    );
    let cap = t
        .locs
        .iter()
        .find(|(l, _)| *l == Loc::Hand(1))
        .map(|&(_, n)| n);
    assert_eq!(cap, Some(3), "its three face-down cards");
    let (hands, burns, undealt) = truth_placement(&m);
    assert!(t.consistent(&hands, &burns, undealt));
    let src = BeliefSource::new();
    let ws = src.prepare(&view, Rules::HOUSE);
    let mut with_club = 0;
    for w in 0..64 {
        let world = ws.sample(&mut world_rng(5, w));
        assert_eq!(engine::cards::count(world.hands[1]), 3);
        let r = rebuild(&view, &world, Rules::HOUSE, 1).expect("rebuilds");
        assert!(same_view(&r, &view));
        with_club += usize::from(world.hands[1] & t.hand_forbid(1) != 0);
    }
    assert!(with_club > 0, "some world gives seat 1 a club");
}

fn has_bit(set: Bits, c: Card) -> bool {
    engine::cards::has(set, c)
}

/// On small late positions after the app's referee: the tracker's worlds are exactly the
/// placements the engine rebuilds into the same view (no fact missing, none wrong, the turned
/// seats' relaxed facts included).
#[test]
fn belief_equals_engine_oracle_after_the_referee() {
    let (mut done, mut with_burns, mut seed) = (0, 0, 0u64);
    let want: usize = std::env::var("REFEREE_SEEDS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map_or(25, |n| n / 2);
    while done < want && seed < 40_000 {
        referee_match(seed, |m| {
            let p = m.public();
            let turned = p.turned_teams.iter().any(|&t| t);
            let hidden_ok = m
                .reneges()
                .iter()
                .all(|r| p.turned_teams[usize::from(engine::team_of(r.seat))]);
            if done >= want || !turned || !hidden_ok || p.trick_no < 4 {
                return;
            }
            if let Some(burns) = oracle_check(m, 3_000, &format!("referee match {seed}")) {
                with_burns += usize::from(burns);
                done += 1;
            }
        });
        seed += 1;
    }
    assert!(
        done >= want && with_burns > 0,
        "positions {done}, with burns {with_burns}"
    );
}

/// Exact PIMC from views after the app's referee (turned partnerships, reneges let through and
/// caught) at tricks 3 and 4: it searches every sampled world exactly (never a network), and each
/// card's value is the mean of naive solves of those worlds.
#[test]
fn exact_pimc_on_referee_views_equals_mean_of_solves() {
    use std::sync::Arc;
    let leaf: Arc<crate::pimc::SharedLeaf> = Arc::new(solver::FnLeaf(|_: &equity::MatchState| {
        equity::MoneyChance::even()
    }));
    let src = BeliefSource::new();
    let ctx = Ctx {
        leaf: &*leaf,
        source: &src,
        rules: Rules::HOUSE,
    };
    let cfg = SearchConfig {
        w12: 2,
        w34: 2,
        threads: 1,
        tt_mb: 4,
        ..SearchConfig::default()
    };
    let mut checked = 0;
    let mut seed = 0u64;
    while checked < 12 && seed < 400 {
        let mut views = Vec::new();
        referee_match(seed, |m| {
            let p = m.public();
            let legal = m.legal_actions();
            if views.len() < 2
                && (p.trick_no == 3 || p.trick_no == 4)
                && p.turned_teams.iter().any(|&t| t)
                && legal.len() >= 2
                && matches!(m.phase(), Phase::AwaitPlay(_))
            {
                let seat = m.to_act().expect("to act");
                views.push((m.view(seat), legal));
            }
        });
        for (i, (v, legal)) in views.iter().enumerate() {
            let d = evaluate(v, legal, i as u64, &ctx, &cfg);
            assert_eq!(d.method, crate::pimc::Method::ExactPimc);
            let worlds = src.prepare(v, Rules::HOUSE);
            let mut per: Vec<Vec<(u64, equity::MoneyChance)>> = vec![Vec::new(); legal.len()];
            for w in 0..u64::from(cfg.w34) {
                let world = worlds.sample(&mut world_rng(i as u64, w));
                let m = rebuild(v, &world, Rules::HOUSE, 0).expect("rebuild");
                let (pos, mctx, _) = solver::from_match(&m).expect("play");
                assert_ne!(pos.turned, 0);
                let sol = solver::solve(
                    &pos,
                    mctx,
                    &solver::FnLeaf(|s: &equity::MatchState| leaf.value(s)),
                    solver::Options::default(),
                );
                for (j, a) in legal.iter().enumerate() {
                    let val = sol
                        .children
                        .iter()
                        .find(|x| x.0 == a.card())
                        .expect("valued")
                        .1;
                    per[j].push((1, val));
                }
            }
            for (j, terms) in per.iter().enumerate() {
                assert_eq!(d.candidates[j].2, equity::MoneyChance::mix(terms));
            }
            checked += 1;
        }
        seed += 1;
    }
    assert!(checked >= 12, "checked {checked}");
}
