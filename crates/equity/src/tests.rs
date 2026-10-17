//! Validation of the sweep, the estimator and engine parity (docs/M1-SPEC.md §6.4).
//!
//! The full 810,000-state sweep runs in the `equity` binary (and in the ignored test
//! `full_synthetic_table_golden`); here the sweep is limited to closed sets of states so
//! `cargo test` stays fast: every state with each team on one game (50,625 states, where any game
//! end is the match), and everything reachable from the "36 points and one game" sanity states
//! (55,092 states over every games pattern). A closed set gives exactly the full table's values.

use std::sync::OnceLock;

use bot::{Bot, BotFactory};
use engine::game::{Action, GameEndReason, Rules};
use engine::view::SeatView;

use crate::estimate::{
    EstimateError, HandOutcome, collect_hands, estimate_model, estimate_model_parallel,
};
use crate::hand::{AfterHand, AfterTrick, InHand, Scoring};
use crate::model::{CompiledModel, HandModel, N_BEFORE_TRICK, comp_index};
use crate::money::{MoneyChance, ONE};
use crate::sha256::sha256_hex;
use crate::state::{MatchState, SCORE_COMBOS, TABLE_STATES, has_first_bonus};
use crate::sweep::{Scope, Solver, SweepOptions, SweepStats, structural_successors, sweep};
use crate::table::{EquityTable, TableError};
use crate::testbots::RandomTestFactory;
use crate::validate;

const SC: Scoring = Scoring::HOUSE;

fn synthetic() -> &'static CompiledModel {
    static M: OnceLock<CompiledModel> = OnceLock::new();
    M.get_or_init(|| HandModel::synthetic().compile())
}

/// Every canonical state in which each team has won one game.
fn all_one_game_states() -> Vec<MatchState> {
    (0..SCORE_COMBOS)
        .map(|i| MatchState::from_canonical_index(15 * SCORE_COMBOS + i))
        .collect()
}

fn run(threads: usize, scope: Scope) -> (EquityTable, SweepStats) {
    sweep(synthetic(), SC, &SweepOptions { threads, scope })
}

/// The one-game-each table, built on 4 threads (shared by several tests).
fn one_game_table() -> &'static (EquityTable, SweepStats) {
    static T: OnceLock<(EquityTable, SweepStats)> = OnceLock::new();
    T.get_or_init(|| run(4, Scope::ReachableFrom(all_one_game_states())))
}

/// Everything reachable from the sanity states, built on 4 threads.
fn sanity_table() -> &'static (EquityTable, SweepStats) {
    static T: OnceLock<(EquityTable, SweepStats)> = OnceLock::new();
    T.get_or_init(|| run(4, Scope::ReachableFrom(validate::sanity_states())))
}

fn raw_hash(t: &EquityTable) -> String {
    let mut bytes = Vec::new();
    for (i, v) in t.raw().iter().enumerate() {
        if *v != [0; 4] {
            bytes.extend_from_slice(&(i as u32).to_le_bytes());
            for x in v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    sha256_hex(&bytes)
}

#[test]
fn sweep_is_byte_identical_for_1_2_and_4_threads() {
    let (t4, s4) = one_game_table();
    for threads in [1, 2] {
        let (t, s) = run(threads, Scope::ReachableFrom(all_one_game_states()));
        assert!(t.raw() == t4.raw(), "{threads} threads differ from 4");
        assert_eq!(&s, s4);
    }
    assert_eq!(s4.states, SCORE_COMBOS as u64);
    // The multi-game scope too (2 vs 4 threads).
    let (t2, _) = run(2, Scope::ReachableFrom(validate::sanity_states()));
    assert!(t2.raw() == sanity_table().0.raw());
}

#[test]
fn reduced_tables_are_byte_stable() {
    // Golden hashes of the synthetic-model tables: any change to the model, the scoring, the
    // rounding or the sweep order shows up here, and CI checks them on Windows and Linux.
    assert_eq!(
        raw_hash(&one_game_table().0),
        "365364e812b8d729a08e4242e7d131bf75c63d0aa07fdd79f6d574b4a81cfbfa",
        "one-game-each table"
    );
    assert_eq!(
        raw_hash(&sanity_table().0),
        "86e3d0eaa36c4aeec12a18614f6a683f2c38ae2669878aeca38856382ac84b42",
        "sanity closure table"
    );
    assert_eq!(sanity_table().1.states, 55_092);
}

#[test]
fn every_entry_sums_to_exactly_one() {
    assert_eq!(validate::check_sums(&one_game_table().0), Ok(SCORE_COMBOS));
    assert_eq!(validate::check_sums(&sanity_table().0), Ok(55_092));
}

#[test]
fn team_rotation_symmetry() {
    for (t, _) in [one_game_table(), sanity_table()] {
        let pairs = validate::check_rotation_symmetry(t, 1).expect("symmetric");
        assert_eq!(pairs, 4 * t.solved_count() as u64);
    }
}

#[test]
fn one_game_and_36_points_is_above_one_half() {
    let lowest = validate::check_sanity(&sanity_table().0).expect("sanity");
    assert!(lowest > ONE / 2);
}

#[test]
fn monotone_in_own_points() {
    let rep = validate::check_monotonicity(&one_game_table().0);
    // Every state, every team, every score but the top one: 50,625 × 4 × 14/15.
    assert_eq!(rep.pairs, 189_000);
    assert_eq!(rep.violations, 0, "{:?}", rep.examples);
}

#[test]
fn monte_carlo_agrees_within_three_standard_errors() {
    let (table, _) = sanity_table();
    let states = validate::sample_states(table, 20, 2026);
    assert_eq!(states.len(), 20);
    // The sample covers several games patterns, so game transitions are exercised.
    let patterns: Vec<[u8; 4]> = states.iter().map(|s| s.canonical().0.games).collect();
    assert!(patterns.iter().any(|g| g.iter().sum::<u8>() == 1));
    let rows = validate::monte_carlo_check(table, synthetic(), SC, &states, 10_000, 2026);
    for r in &rows {
        assert!(
            r.within,
            "{}: table {} vs wins {:?} of {} (z {:?})",
            r.state, r.table, r.wins, r.matches, r.z
        );
    }
}

#[test]
fn stored_values_equal_a_fresh_solve() {
    let (table, _) = sanity_table();
    let mut solver = Solver::new(synthetic(), SC);
    let value = |j: usize| table.get(j).expect("closed scope");
    for (k, i) in (0..TABLE_STATES)
        .filter(|&i| table.get(i).is_some())
        .enumerate()
    {
        if k % 97 != 0 {
            continue;
        }
        let s = MatchState::from_canonical_index(i);
        assert_eq!(
            solver.solve(&s, &value),
            table.get(i).expect("solved"),
            "{s}"
        );
    }
}

#[test]
fn closed_form_values() {
    let (table, _) = one_game_table();
    // Every team at one game and 40 (or 36) points: whoever wins the first trick wins the money,
    // so the value is exactly the model's first-trick distribution.
    let first_trick = MoneyChance(synthetic().winner[0]);
    for score in [40u8, 36] {
        let s = MatchState::new([score; 4], [1; 4], 0).expect("valid");
        assert_eq!(table.lookup(&s), first_trick);
    }
    // Rotated to another dealer: the same chances move with the dealer's team.
    let s = MatchState::new([40; 4], [1; 4], 3).expect("valid");
    assert_eq!(table.lookup(&s), first_trick.rotate(3));
    // Exact ties are bit-identical: 36 (flag) and 40 both reach 41 with any trick or bonus.
    let a = MatchState::new([36, 10, 5, 0], [1; 4], 0).expect("valid");
    let b = MatchState::new([40, 10, 5, 0], [1; 4], 0).expect("valid");
    assert_eq!(table.lookup(&a), table.lookup(&b));
}

#[test]
fn hand_positions_have_reachable_scores_and_matching_flags() {
    // Every 23rd state of the whole table: every position of its hand, under any trick
    // sequence, keeps flag == (score mod 5 == 1) and ends on reachable scores; every successor
    // is a valid state with a table entry (or a finished match).
    let mut positions = 0;
    let mut succ = Vec::new();
    for i in (0..TABLE_STATES).step_by(23) {
        let s = MatchState::from_canonical_index(i);
        positions += validate::check_hand_positions(&s, SC).expect("property holds");
        succ.clear();
        structural_successors(&s, SC, &mut succ);
        for n in &succ {
            assert!(n.validate().is_ok(), "{s} -> {n}");
            assert!(n.winner().is_some() || n.canonical_index().is_some());
        }
    }
    assert!(positions > 1_000_000);
}

#[test]
fn lookup_terminal_unsolved_and_npy_round_trip() {
    let (table, _) = one_game_table();
    let done = MatchState::new([45, 10, 0, 0], [2, 1, 1, 0], 2).expect("finished");
    assert_eq!(table.lookup(&done), MoneyChance::certain(0));
    let elsewhere = MatchState::new([0; 4], [0; 4], 0).expect("valid");
    assert_eq!(
        table.try_lookup(&elsewhere),
        Err(TableError::NotSolved(elsewhere))
    );
    assert!(
        table.to_npy().is_err(),
        "an incomplete table is never written"
    );
    // A complete table survives the .npy round trip.
    let mut full = EquityTable::unsolved();
    for i in 0..TABLE_STATES {
        let v = match i % 3 {
            0 => MoneyChance::even(),
            1 => MoneyChance::certain((i % 4) as u8),
            _ => MoneyChance::from_weights([i as u128, 1, 2, 3]),
        };
        full.set(i, v);
    }
    let bytes = full.to_npy().expect("complete");
    let back = EquityTable::from_npy(&bytes).expect("read");
    assert!(back == full);
    // A corrupted entry (sum not 2^32) is refused.
    let mut bad = bytes.clone();
    let last = bad.len() - 1;
    bad[last] ^= 1;
    assert!(EquityTable::from_npy(&bad).is_err());
}

/// The engine's hand, replayed through the table's scoring functions, must end the same way.
fn replay_record(r: &crate::estimate::HandRecord) -> Result<HandOutcome, String> {
    let abs = |o: u8| (o + r.dealer_team) % 4;
    let mut pos = InHand::start_with_flags(r.start_scores, r.start_flags);
    for (i, &o) in r.winners.iter().enumerate() {
        let here = r.five_trick == Some((i + 1) as u8);
        match pos.play_trick(abs(o), here, SC) {
            AfterTrick::GameWon(team, reason) => {
                if i + 1 != r.winners.len() {
                    return Err(format!(
                        "game ends at trick {} but the engine played on",
                        i + 1
                    ));
                }
                return Ok(HandOutcome::GameWon { team, reason });
            }
            AfterTrick::Next(p) | AfterTrick::TricksDone(p) => pos = p,
        }
    }
    if r.winners.len() != 5 {
        return Err("the engine stopped early but nobody reached 41".into());
    }
    Ok(match pos.finish(r.best_trump.map(abs), SC) {
        AfterHand::GameWon(team, reason) => HandOutcome::GameWon { team, reason },
        AfterHand::HandOver(scores) => HandOutcome::HandOver { scores },
    })
}

#[test]
fn engine_parity_on_self_play_hands() {
    let records = collect_hands(&RandomTestFactory, 3_000, 41).expect("legal play");
    assert_eq!(records.len(), 3_000);
    let mut reasons = [0u32; 4];
    for r in &records {
        // Between hands the engine only ever shows reachable scores, and its explicit bonus
        // flags equal score mod 5 == 1 (house rule).
        let state = MatchState::new(r.start_scores, r.games, r.dealer_team);
        assert!(state.is_ok(), "{r:?}");
        assert_eq!(r.start_flags, r.start_scores.map(has_first_bonus), "{r:?}");
        // Rule 31 never fires in a completed hand: some trump is always played.
        if r.winners.len() == 5 {
            assert!(r.best_trump.is_some());
        }
        let ours = replay_record(r).unwrap_or_else(|e| panic!("{e}: {r:?}"));
        assert_eq!(ours, r.outcome, "{r:?}");
        if let HandOutcome::GameWon { reason, .. } = r.outcome {
            reasons[match reason {
                GameEndReason::FortyOneByTrick => 0,
                GameEndReason::FortyOneByImmediateBonus => 1,
                GameEndReason::FortyOneByEndBonus => 2,
                GameEndReason::Jink => 3,
                GameEndReason::FortyOneByPassedTrick => panic!("nobody is turned in legal play"),
            }] += 1;
        }
    }
    // Random play reaches every way of winning a game except (rarely) a jink.
    assert!(
        reasons[0] > 0 && reasons[1] > 0 && reasons[2] > 0,
        "{reasons:?}"
    );
}

#[test]
fn estimator_counts_are_consistent_and_thread_independent() {
    let hands = 1_200;
    let m = estimate_model(&RandomTestFactory, hands, 5).expect("legal play");
    for threads in [1, 3] {
        let p = estimate_model_parallel(&RandomTestFactory, hands, 5, threads).expect("legal");
        assert_eq!(p, m, "{threads} threads");
    }
    let s = &m.source;
    assert_eq!(s.hands, hands);
    assert_eq!(s.hands_completed + s.hands_cut_short, hands);
    assert_eq!(s.bot, "random-test");
    // Every hand has a first trick; every trick is counted once.
    assert_eq!(m.trick_winner[0].iter().sum::<u64>(), hands);
    let tricks: u64 = m.trick_winner.iter().flatten().sum();
    assert_eq!(tricks, s.tricks);
    let five_rows: u64 = m.five_in_trick.iter().flatten().flatten().sum();
    assert!(five_rows <= tricks);
    let five_seen: u64 = m.five_in_trick.iter().flatten().map(|r| r[1]).sum();
    let end: u64 = m.end_bonus.iter().flatten().sum();
    assert!(end <= s.hands_completed);
    // A completed hand either played the 5 or counts once in end_bonus; a hand cut short may
    // also have played the 5.
    assert!(five_seen + end >= s.hands_completed && five_seen + end <= hands);
    // Never a bonus count for a team that won no trick.
    for (i, row) in m.end_bonus.iter().enumerate() {
        let tw = crate::model::COMPS[N_BEFORE_TRICK + i];
        for o in 0..4 {
            assert!(tw[o] > 0 || row[o] == 0);
        }
    }
    // Round trip through JSON, and a usable compiled model.
    assert_eq!(HandModel::from_json(&m.to_json()).expect("json"), m);
    assert!(m.compile().check());
    // The leading team (offset 1) and the others all win first tricks.
    assert!(m.trick_winner[comp_index([0; 4])].iter().all(|&c| c > 0));
}

#[test]
fn estimated_model_builds_a_valid_table() {
    let m = estimate_model(&RandomTestFactory, 2_000, 9).expect("legal play");
    let compiled = m.compile();
    let (table, stats) = sweep(
        &compiled,
        SC,
        &SweepOptions {
            threads: 4,
            scope: Scope::ReachableFrom(all_one_game_states()),
        },
    );
    assert_eq!(stats.states, SCORE_COMBOS as u64);
    assert_eq!(validate::check_sums(&table), Ok(SCORE_COMBOS));
}

struct CheatBot;

impl Bot for CheatBot {
    fn name(&self) -> &str {
        "cheat"
    }
    fn start_match(&mut self, _seat: u8, _rules: Rules, _seed: u64) {}
    fn act(&mut self, view: &SeatView, legal: &[Action]) -> Action {
        // Play a card it does not hold.
        let not_mine = (0..53u8)
            .find(|&c| view.hand & (1u64 << c) == 0)
            .unwrap_or(0);
        match legal[0] {
            Action::Burn(_) => Action::Burn(not_mine),
            Action::Play(_) => Action::Play(not_mine),
        }
    }
}

struct CheatFactory;

impl BotFactory for CheatFactory {
    fn name(&self) -> &str {
        "cheat"
    }
    fn build(&self) -> Box<dyn Bot> {
        Box::new(CheatBot)
    }
}

#[test]
fn estimator_refuses_illegal_actions() {
    match estimate_model(&CheatFactory, 10, 1) {
        Err(EstimateError::IllegalAction { bot, .. }) => assert_eq!(bot, "cheat"),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// The full 810,000-state sweep with the synthetic model (about 7 s in release on 4 cores; run
/// with `cargo test -p equity --release -- --ignored`).
#[test]
#[ignore]
fn full_synthetic_table_golden() {
    let (table, stats) = run(4, Scope::Full);
    assert!(table.is_complete());
    assert_eq!(stats.states, TABLE_STATES as u64);
    let npy = table.to_npy().expect("complete");
    assert_eq!(
        sha256_hex(&npy),
        "4d452d8ed75a9a93bb1faccd90d472bc1b1bef0f166620ec9d6c67913b606e08"
    );
    assert_eq!(validate::check_monotonicity(&table).violations, 0);
    validate::check_sanity(&table).expect("sanity");
}
