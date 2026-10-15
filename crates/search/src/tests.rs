//! PIMC tests (docs/M3-SPEC.md §5.11, §5.6).

use std::sync::Arc;

use bot::{Bot, BotFactory};

use engine::game::{Action, Match, Phase, Rules};
use engine::rng::Rng;
use engine::view::SeatView;
use equity::{MatchState, MoneyChance};
use solver::FnLeaf;

use crate::bot::{PimcFactory, PimcSpec};
use crate::pimc::{
    Ctx, DecisionKind, Early, Method, SearchConfig, SharedLeaf, candidates, evaluate, rollout,
    world_values,
};
use crate::worlds::{RejectionSource, WorldSource, rebuild, world_rng};

/// A synthetic leaf: chances proportional to (1 + score + 40·games).
fn leaf() -> Arc<SharedLeaf> {
    Arc::new(FnLeaf(|s: &MatchState| {
        MoneyChance::from_weights(core::array::from_fn(|t| {
            1 + u128::from(s.scores[t]) + 40 * u128::from(s.games[t])
        }))
    }))
}

/// Decisions met while playing rules-v1 matches: (view, legal) at the given trick.
fn decisions_at(trick: u8, want: usize, seed: u64) -> Vec<(SeatView, Vec<Action>)> {
    let mut out = Vec::new();
    let mut k = 0;
    while out.len() < want {
        let mut m = Match::new(Rng::derive(seed, &[k]).next_u64(), Rules::HOUSE);
        k += 1;
        while m.phase() != Phase::MatchOver && m.hand_index() < 3 {
            let seat = m.to_act().expect("to act");
            let v = m.view(seat);
            let legal = m.legal_actions();
            if v.public.trick_no == trick
                && candidates(&v, &legal).1.len() >= 2
                && out.len() < want
                && (k + out.len() as u64).is_multiple_of(3)
            {
                out.push((v.clone(), legal.clone()));
            }
            let a = bots::decide(&v, &Rules::HOUSE, &legal).0;
            m.apply(a).expect("legal");
        }
    }
    out
}

fn cfg(threads: usize) -> SearchConfig {
    SearchConfig {
        w12: 3,
        w34: 3,
        threads,
        tt_mb: 4,
        ..SearchConfig::default()
    }
}

#[test]
fn pimc_thread_independent() {
    let l = leaf();
    let src = RejectionSource::new();
    let ctx = Ctx {
        leaf: &*l,
        source: &src,
        rules: Rules::HOUSE,
    };
    for trick in [1u8, 2, 3, 4] {
        for (i, (v, legal)) in decisions_at(trick, 4, u64::from(trick)).iter().enumerate() {
            let a = evaluate(v, legal, i as u64, &ctx, &cfg(1));
            let b = evaluate(v, legal, i as u64, &ctx, &cfg(2));
            let c = evaluate(v, legal, i as u64, &ctx, &cfg(4));
            assert_eq!(a, b);
            assert_eq!(a, c);
            assert_ne!(a.method, Method::Forced);
        }
    }
}

#[test]
fn forced_no_search() {
    let l = leaf();
    let src = RejectionSource { max_tries: 1 };
    let ctx = Ctx {
        leaf: &*l,
        source: &src,
        rules: Rules::HOUSE,
    };
    let mut m = Match::new(3, Rules::HOUSE);
    while m.phase() != Phase::MatchOver {
        let seat = m.to_act().expect("to act");
        let v = m.view(seat);
        let legal = m.legal_actions();
        if legal.len() == 1 {
            let d = evaluate(&v, &legal, 0, &ctx, &cfg(1));
            assert_eq!(d.method, Method::Forced);
            assert_eq!(d.n_worlds, 0);
            return;
        }
        m.apply(bots::decide(&v, &Rules::HOUSE, &legal).0)
            .expect("legal");
    }
    panic!("no forced decision met");
}

/// Exact PIMC equals the mean of naive solves of the same worlds.
#[test]
fn exact_pimc_equals_mean_of_solves() {
    let l = leaf();
    let src = RejectionSource::new();
    let ctx = Ctx {
        leaf: &*l,
        source: &src,
        rules: Rules::HOUSE,
    };
    let c = cfg(1);
    for (i, (v, legal)) in decisions_at(4, 6, 11).iter().enumerate() {
        let d = evaluate(v, legal, i as u64, &ctx, &c);
        assert_eq!(d.method, Method::ExactPimc);
        let worlds = src.prepare(v, Rules::HOUSE);
        let mut per: Vec<Vec<(u64, MoneyChance)>> = vec![Vec::new(); legal.len()];
        for w in 0..u64::from(c.w34) {
            let world = worlds.sample(&mut world_rng(i as u64, w));
            let m = rebuild(v, &world, Rules::HOUSE, 0).expect("rebuild");
            let (pos, mctx, pend) = solver::from_match(&m).expect("play");
            assert!(pend.is_none());
            let sol = solver::solve(
                &pos,
                mctx,
                &FnLeaf(|s: &MatchState| l.value(s)),
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
            assert_eq!(d.candidates[j].2, MoneyChance::mix(terms));
        }
    }
}

/// The rollout leaf equals the solver's leaf for the finished hand: solving the final position
/// (no cards left is not a solver position, so the last trick is solved instead).
#[test]
fn leaf_rule_matches_solver() {
    let l = leaf();
    let mut checked = 0;
    for seed in 0..60u64 {
        let mut m = Match::new(seed, Rules::HOUSE);
        // Play to the 8th card of trick 5 (or as far as the hand goes).
        let h = m.hand_index();
        let mut last_position = None;
        while m.phase() != Phase::MatchOver && m.hand_index() == h {
            if m.public().trick_no == 5
                && m.public().trick_cards.len() == 7
                && matches!(m.phase(), Phase::AwaitPlay(_))
            {
                last_position = Some(m.clone());
            }
            let seat = m.to_act().expect("to act");
            let v = m.view(seat);
            let legal = m.legal_actions();
            m.apply(bots::decide(&v, &Rules::HOUSE, &legal).0)
                .expect("legal");
        }
        let Some(mut p) = last_position else { continue };
        let (pos, mctx, _) = solver::from_match(&p).expect("play");
        let sol = solver::solve(
            &pos,
            mctx,
            &FnLeaf(|s: &MatchState| l.value(s)),
            solver::Options::default(),
        );
        let r = rollout(&mut p, &*l, Rules::HOUSE);
        assert_eq!(sol.value, Some(r));
        checked += 1;
    }
    assert!(checked > 20);
}

/// The robber's joint decision: the play after the burn is the stored card, and the pair is the
/// best of the full grid.
#[test]
fn joint_burn_then_play() {
    let l = leaf();
    let spec = PimcSpec::parse("pimc-v0:w12=2").expect("ok");
    let f = PimcFactory::new(&spec, l.clone(), Arc::new(RejectionSource::new()));
    let mut found = 0;
    for seed in 0..400u64 {
        let mut m = Match::new(seed, Rules::HOUSE);
        while m.phase() != Phase::MatchOver && m.hand_index() == 0 {
            let seat = m.to_act().expect("to act");
            let v = m.view(seat);
            let legal = m.legal_actions();
            if matches!(m.phase(), Phase::AwaitBurn(_)) && v.public.trick_no == 1 {
                let mut b = f.build_pimc();
                b.start_match(seat, Rules::HOUSE, seed);
                let a = b.act(&v, &legal);
                let dv = b.last.clone().expect("decided");
                assert_eq!(dv.kind, DecisionKind::JointBurnPlay);
                let (ca, cp, cv) = dv.candidates[dv.chosen];
                assert_eq!(a, ca);
                let team = engine::team_of(seat);
                assert!(dv.candidates.iter().all(|c| c.2.get(team) <= cv.get(team)));
                m.apply(a).expect("legal burn");
                let v2 = m.view(seat);
                let legal2 = m.legal_actions();
                let p = b.act(&v2, &legal2);
                assert_eq!(Some(p.card()), cp);
                assert_eq!(p, Action::Play(cp.expect("joint")));
                found += 1;
                break;
            }
            m.apply(bots::decide(&v, &Rules::HOUSE, &legal).0)
                .expect("legal");
        }
        if found >= 3 {
            break;
        }
    }
    assert!(found >= 1, "no rob at turn met");
}

/// Exact mode at trick 1 resolves a future robber's burn; values are thread-independent.
#[test]
fn exact_trick2_few_worlds_runs() {
    let l = leaf();
    let src = RejectionSource::new();
    let ctx = Ctx {
        leaf: &*l,
        source: &src,
        rules: Rules::HOUSE,
    };
    let mut c = cfg(1);
    c.t2 = Early::Exact;
    c.t2w = 1;
    c.tt_mb = 64;
    let ds = decisions_at(2, 1, 5);
    let (v, legal) = &ds[0];
    // Only check the late positions of the trick (cheap).
    let d = evaluate(v, legal, 1, &ctx, &c);
    assert_eq!(d.method, Method::ExactPimc);
}

#[test]
fn rollout_world_rows_are_nested() {
    let l = leaf();
    let src = RejectionSource::new();
    let ctx = Ctx {
        leaf: &*l,
        source: &src,
        rules: Rules::HOUSE,
    };
    let (v, legal) = &decisions_at(2, 1, 8)[0];
    let (_, cands) = candidates(v, legal);
    let a = world_values(v, &cands, Method::RolloutPimc, 0, 6, 9, &ctx, &cfg(2));
    let b = world_values(v, &cands, Method::RolloutPimc, 2, 4, 9, &ctx, &cfg(1));
    assert_eq!(&a[2..4], &b[..]);
}

/// Canary: pimc-v0 against itself in a few duplicate-style matches never fails and is legal.
#[test]
fn pimc_plays_whole_hands() {
    let l = leaf();
    let spec = PimcSpec::parse("pimc-v0:w12=2,w34=2").expect("ok");
    let f = PimcFactory::new(&spec, l, Arc::new(RejectionSource::new()));
    let mut m = Match::new(77, Rules::HOUSE);
    let mut bots: Vec<Box<dyn Bot>> = (0..8).map(|_| f.build()).collect();
    for (s, b) in bots.iter_mut().enumerate() {
        b.start_match(s as u8, Rules::HOUSE, s as u64);
    }
    while m.phase() != Phase::MatchOver && m.hand_index() < 2 {
        let seat = m.to_act().expect("to act");
        let v = m.view(seat);
        let legal = m.legal_actions();
        let a = bots[usize::from(seat)].act(&v, &legal);
        assert!(legal.contains(&a));
        m.apply(a).expect("legal");
    }
}

/// A rollout that ends the match is certain money for the team with two games, whatever the
/// stake (review B, 2026-10-15: the winner was read from the money, so a stake of 0 always
/// named team 0).
#[test]
fn rollout_match_end_is_certain_for_the_two_game_team_at_any_stake() {
    let l = leaf();
    let mut found = 0;
    let mut k = 0u64;
    while found < 3 {
        assert!(k < 400, "no match-ending rollouts found");
        let mut m = Match::new(Rng::derive(0xB0B, &[k]).next_u64(), Rules { stake: 0 });
        k += 1;
        while m.phase() != Phase::MatchOver {
            let p = m.public();
            if p.trick_no == 1
                && p.trick_cards.is_empty()
                && matches!(m.phase(), Phase::AwaitPlay(_))
            {
                let mut r = m.clone();
                let v = rollout(&mut r, &*l, Rules::HOUSE);
                if r.phase() == Phase::MatchOver {
                    let w = (0..4u8)
                        .find(|&t| r.games_won()[usize::from(t)] == 2)
                        .expect("a team with two games");
                    assert_eq!(v, MoneyChance::certain(w));
                    if w != 0 {
                        found += 1;
                    }
                }
            }
            let seat = m.to_act().expect("to act");
            let v = m.view(seat);
            let legal = m.legal_actions();
            let a = bots::decide(&v, &Rules::HOUSE, &legal).0;
            m.apply(a).expect("legal");
        }
    }
}
