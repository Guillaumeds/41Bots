//! Validation of a built table (docs/M1-SPEC.md §6.4): sums, team-rotation symmetry, the
//! "36 points and one game" sanity check, monotonicity in a team's own points, and agreement with
//! a direct Monte-Carlo estimate. Used by the tests on reduced tables and by `equity check` on the
//! full table. Floating point appears here only in reports, never in the table.

use engine::rng::Rng;

use crate::hand::Scoring;
use crate::model::CompiledModel;
use crate::money::{MoneyChance, ONE};
use crate::simulate::simulate_match;
use crate::state::{MatchState, N_SCORES, SCORE_COMBOS, TABLE_STATES};
use crate::table::EquityTable;

/// Every computed entry sums to exactly `ONE`; returns the number of computed entries.
pub fn check_sums(table: &EquityTable) -> Result<usize, String> {
    let mut n = 0;
    for (i, v) in table.raw().iter().enumerate() {
        if *v == [0; 4] {
            continue;
        }
        let sum: u128 = v.iter().map(|&x| u128::from(x)).sum();
        if sum != u128::from(ONE) {
            return Err(format!(
                "entry {i} ({}) sums to {sum}",
                MatchState::from_canonical_index(i)
            ));
        }
        n += 1;
    }
    Ok(n)
}

/// For every computed state and every rotation, looking up the rotated state gives the rotated
/// value. Returns the number of (state, rotation) pairs checked.
pub fn check_rotation_symmetry(table: &EquityTable, stride: usize) -> Result<u64, String> {
    let mut n = 0;
    for i in (0..TABLE_STATES).step_by(stride.max(1)) {
        let Some(v) = table.get(i) else { continue };
        let s = MatchState::from_canonical_index(i);
        for by in 0..4u8 {
            let r = s.rotate(by);
            let got = table.lookup(&r);
            if got != v.rotate(by) {
                return Err(format!("{r}: {got:?} is not the rotation of {v:?}"));
            }
            n += 1;
        }
    }
    Ok(n)
}

/// A violation of monotonicity: team `team` has a lower chance at the higher score.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MonotonicityViolation {
    /// The state with the lower score for `team`.
    pub lower: MatchState,
    /// The same state with `team` on the next reachable score.
    pub higher: MatchState,
    /// The team.
    pub team: u8,
    /// Its chance at the lower score (units of 2^-32).
    pub at_lower: u64,
    /// Its chance at the higher score.
    pub at_higher: u64,
}

/// The result of the monotonicity check.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MonotonicityReport {
    /// Pairs (state, team, next score) with both entries computed.
    pub pairs: u64,
    /// Pairs where the chance went down.
    pub violations: u64,
    /// The first few violations, by index order.
    pub examples: Vec<MonotonicityViolation>,
    /// The largest drop seen (units of 2^-32).
    pub largest_drop: u64,
}

/// Check that each team's chance never falls when its own score rises to the next reachable
/// value, everything else fixed (canonical states, so the dealer team is fixed too).
pub fn check_monotonicity(table: &EquityTable) -> MonotonicityReport {
    let mut rep = MonotonicityReport::default();
    let place = [
        N_SCORES * N_SCORES * N_SCORES,
        N_SCORES * N_SCORES,
        N_SCORES,
        1,
    ];
    for i in 0..TABLE_STATES {
        let Some(v) = table.get(i) else { continue };
        let rest = i % SCORE_COMBOS;
        for (t, &step) in place.iter().enumerate() {
            let si = (rest / step) % N_SCORES;
            if si + 1 >= N_SCORES {
                continue;
            }
            let j = i + step;
            let Some(w) = table.get(j) else { continue };
            rep.pairs += 1;
            if w.0[t] < v.0[t] {
                rep.violations += 1;
                rep.largest_drop = rep.largest_drop.max(v.0[t] - w.0[t]);
                if rep.examples.len() < 10 {
                    rep.examples.push(MonotonicityViolation {
                        lower: MatchState::from_canonical_index(i),
                        higher: MatchState::from_canonical_index(j),
                        team: t as u8,
                        at_lower: v.0[t],
                        at_higher: w.0[t],
                    });
                }
            }
        }
    }
    rep
}

/// One state of the Monte-Carlo comparison.
#[derive(Clone, Debug)]
pub struct McRow {
    /// The state (real team order, any dealer team).
    pub state: MatchState,
    /// The table's value.
    pub table: MoneyChance,
    /// Simulated match wins per team.
    pub wins: [u64; 4],
    /// Simulated matches.
    pub matches: u64,
    /// `(wins − n·p) / sqrt(n·p·(1 − p))` per team (0 when p is 0 or 1), for reports.
    pub z: [f64; 4],
    /// Whether every team is within 3 standard errors, plus one match of continuity correction.
    pub within: bool,
}

/// Compare the table with direct simulation from each state.
pub fn monte_carlo_check(
    table: &EquityTable,
    model: &CompiledModel,
    sc: Scoring,
    states: &[MatchState],
    matches: u64,
    seed: u64,
) -> Vec<McRow> {
    states
        .iter()
        .enumerate()
        .map(|(k, s)| {
            let mut rng = Rng::derive(seed, &[0x3C, k as u64]);
            let mut wins = [0u64; 4];
            for _ in 0..matches {
                wins[usize::from(simulate_match(model, sc, s, &mut rng))] += 1;
            }
            let v = table.lookup(s);
            let p = v.to_f64_for_display();
            let n = matches as f64;
            let mut z = [0.0; 4];
            let mut within = true;
            for t in 0..4 {
                let sd = (n * p[t] * (1.0 - p[t])).sqrt();
                let dev = wins[t] as f64 - n * p[t];
                z[t] = if sd > 0.0 { dev / sd } else { 0.0 };
                if dev.abs() > 3.0 * sd + 1.0 {
                    within = false;
                }
            }
            McRow {
                state: *s,
                table: v,
                wins,
                matches,
                z,
                within,
            }
        })
        .collect()
}

/// `count` distinct computed states chosen by the seed, each given a random dealer team (so
/// lookups exercise the rotation).
pub fn sample_states(table: &EquityTable, count: usize, seed: u64) -> Vec<MatchState> {
    let solved: Vec<usize> = (0..TABLE_STATES)
        .filter(|&i| table.get(i).is_some())
        .collect();
    let mut rng = Rng::derive(seed, &[0x5A]);
    let mut out: Vec<MatchState> = Vec::new();
    let mut picked: Vec<usize> = Vec::new();
    while out.len() < count.min(solved.len()) {
        let i = solved[rng.bounded(solved.len() as u64) as usize];
        if picked.contains(&i) {
            continue;
        }
        picked.push(i);
        let by = rng.bounded(4) as u8;
        out.push(MatchState::from_canonical_index(i).rotate(by));
    }
    out
}

/// The sanity states: one team on 1 game and 36 points, the others on 0 games and 0 points, for
/// each team and each dealer team.
pub fn sanity_states() -> Vec<MatchState> {
    let mut out = Vec::new();
    for team in 0..4usize {
        for dealer in 0..4u8 {
            let mut scores = [0u8; 4];
            let mut games = [0u8; 4];
            scores[team] = 36;
            games[team] = 1;
            out.push(MatchState {
                scores,
                games,
                dealer_team: dealer,
            });
        }
    }
    out
}

/// Check that the leader's chance is above one half in every sanity state; returns the lowest.
pub fn check_sanity(table: &EquityTable) -> Result<u64, String> {
    let mut lowest = ONE;
    for s in sanity_states() {
        let team = (0..4u8)
            .find(|&t| s.games[usize::from(t)] == 1)
            .unwrap_or(0);
        let v = table.lookup(&s).get(team);
        if v <= ONE / 2 {
            return Err(format!("{s}: team {team} has only {v} / 2^32"));
        }
        lowest = lowest.min(v);
    }
    Ok(lowest)
}

/// Walk every position of a hand from a state under any trick sequence (every winner, the 5 of
/// trumps in any trick, the end bonus to any team that won a trick) and check the M1 property
/// tests: the bonus flag equals `score mod 5 == 1` at every position and after the end bonus, and
/// every score a hand can end on (without ending the game) is one of the 15 reachable values.
/// Returns the number of positions visited.
pub fn check_hand_positions(state: &MatchState, sc: Scoring) -> Result<u64, String> {
    use crate::hand::{AfterHand, AfterTrick, InHand};
    use crate::model::comp_index;
    use crate::state::score_index;

    fn walk(
        pos: InHand,
        five: u8,
        sc: Scoring,
        seen: &mut [bool; crate::sweep::NODE_SLOTS],
        count: &mut u64,
    ) -> Result<(), String> {
        let key = comp_index(pos.tricks) * 5 + usize::from(five);
        if seen[key] {
            return Ok(());
        }
        seen[key] = true;
        *count += 1;
        if !pos.flags_match_scores() {
            return Err(format!(
                "flags {:?} disagree with scores {:?}",
                pos.flags, pos.scores
            ));
        }
        if pos.tricks_done == engine::game::TRICKS_PER_HAND {
            let teams: Vec<Option<u8>> = if pos.five_played {
                vec![None]
            } else {
                (0..4u8)
                    .filter(|&b| pos.tricks[usize::from(b)] > 0)
                    .map(Some)
                    .collect()
            };
            for b in teams {
                if let AfterHand::HandOver(s) = pos.finish(b, sc) {
                    let after = InHand {
                        scores: s,
                        flags: pos.flags_after_bonus(b),
                        ..pos
                    };
                    if !after.flags_match_scores() {
                        return Err(format!("flags after the end bonus disagree with {s:?}"));
                    }
                    if let Some(&bad) = s.iter().find(|&&x| score_index(x).is_none()) {
                        return Err(format!("hand ends on unreachable score {bad} ({s:?})"));
                    }
                }
            }
            return Ok(());
        }
        for w in 0..4u8 {
            for here in [false, true] {
                if here && pos.five_played {
                    continue;
                }
                match pos.play_trick(w, here, sc) {
                    AfterTrick::GameWon(..) => {}
                    AfterTrick::Next(p) | AfterTrick::TricksDone(p) => {
                        walk(p, if here { 1 + w } else { five }, sc, seen, count)?;
                    }
                }
            }
        }
        Ok(())
    }

    let mut seen = [false; crate::sweep::NODE_SLOTS];
    let mut count = 0;
    walk(InHand::start(state.scores), 0, sc, &mut seen, &mut count)
        .map_err(|e| format!("{state}: {e}"))?;
    Ok(count)
}
