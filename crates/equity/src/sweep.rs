//! The exact backward-induction sweep (docs/M1-SPEC.md §6.4, PLAN amendment A1).
//!
//! States are solved in the order (games won, most first) then (total points, most first). Every
//! hand either ends a game (one more game won) or adds at least 30 points (five tricks of 5 and
//! exactly one best-trump bonus, since a completed 8-player hand always contains a trump), so
//! every successor of a state is solved before the state itself, and one pass is exact given the
//! hand-outcome model. States with the same (games, points) never depend on each other, so each
//! such *level* is split over threads; every state's value depends only on finished entries and
//! is computed by the same integer steps, so the table is byte-identical for any thread count.
//!
//! Inside a hand, a dynamic program [a calculation that solves each distinct sub-position once and
//! reuses it] expands trick by trick, collapsing equal positions (tricks won per team, whether
//! and by whom the 5 of trumps was played; the scores and flags follow from these), and applies
//! the engine's scoring and end checks ([`crate::hand`]). A game end moves to the next game's
//! first state (scores and flags reset, games + 1, dealer team + 1) or, at 2 games, to certainty;
//! a plain hand end moves to (new scores, dealer team + 1). Every average is
//! [`MoneyChance::mix_wide`] with exact integer weights.

use std::sync::atomic::{AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};

use crate::hand::{AfterHand, AfterTrick, InHand, Scoring};
use crate::model::{COMPS, CompiledModel, N_BEFORE_TRICK, N_COMPS, comp_index};
use crate::money::{MoneyChance, ONE};
use crate::state::{MatchState, TABLE_STATES, value_via};
use crate::table::EquityTable;

/// Which states to solve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Scope {
    /// All 810,000 canonical states (the production table).
    Full,
    /// Only these states and every state reachable from them by any trick sequence (a closed set,
    /// so the values are exactly those of the full table). For fast tests.
    ReachableFrom(Vec<MatchState>),
}

/// How to run a sweep.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SweepOptions {
    /// Worker threads (1 or more). The result does not depend on it.
    pub threads: usize,
    /// Which states.
    pub scope: Scope,
}

/// What a sweep did (independent of the thread count).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SweepStats {
    /// States solved.
    pub states: u64,
    /// Levels (groups of states with the same games won and total points).
    pub levels: u64,
    /// States in the largest level.
    pub largest_level: u64,
    /// In-hand positions evaluated, summed over all states.
    pub in_hand_positions: u64,
}

/// Positions inside a hand: a split of tricks (126) × the 5 of trumps (not played, or played and
/// won by team 0..=3).
pub(crate) const NODE_SLOTS: usize = N_COMPS * 5;

/// Per-thread scratch for solving states one at a time.
pub struct Solver<'m> {
    model: &'m CompiledModel,
    sc: Scoring,
    memo: Vec<MoneyChance>,
    stamp: Vec<u32>,
    current: u32,
    /// In-hand positions evaluated so far.
    pub positions: u64,
}

struct Ctx<'a> {
    games: [u8; 4],
    game_end: [MoneyChance; 4],
    value: &'a dyn Fn(usize) -> MoneyChance,
}

impl Ctx<'_> {
    /// The hand is over with these scores: the next hand is dealt by team 1 of this frame.
    fn hand_over(&self, scores: [u8; 4]) -> MoneyChance {
        let next = MatchState {
            scores,
            games: self.games,
            dealer_team: 1,
        };
        value_via(&next, self.value)
    }
}

impl<'m> Solver<'m> {
    /// Scratch for this model and scoring.
    pub fn new(model: &'m CompiledModel, sc: Scoring) -> Self {
        Solver {
            model,
            sc,
            memo: vec![MoneyChance::even(); NODE_SLOTS],
            stamp: vec![0; NODE_SLOTS],
            current: 0,
            positions: 0,
        }
    }

    /// Solve one canonical state (dealer team 0) from the values of its successors, given as a
    /// function from canonical index to value.
    ///
    /// # Panics
    /// If the state is not canonical or finished, or a successor has no value.
    pub fn solve(
        &mut self,
        state: &MatchState,
        value: &dyn Fn(usize) -> MoneyChance,
    ) -> MoneyChance {
        assert_eq!(state.dealer_team, 0, "solve canonical states only");
        assert!(
            state.winner().is_none(),
            "a finished match needs no solving"
        );
        self.current = self.current.wrapping_add(1);
        if self.current == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.current = 1;
        }
        let game_end: [MoneyChance; 4] = core::array::from_fn(|w| {
            let mut games = state.games;
            games[w] += 1;
            if games[w] >= engine::game::GAMES_TO_WIN {
                MoneyChance::certain(w as u8)
            } else {
                let next = MatchState {
                    scores: [0; 4],
                    games,
                    dealer_team: 1,
                };
                value_via(&next, value)
            }
        });
        let ctx = Ctx {
            games: state.games,
            game_end,
            value,
        };
        self.node(&ctx, 0, 0, InHand::start(state.scores))
    }

    /// The value of an in-hand position: split of tricks `COMPS[k]`, `five` = 0 if the 5 of
    /// trumps is not played yet, else 1 + the team that won its trick.
    fn node(&mut self, ctx: &Ctx<'_>, k: usize, five: u8, pos: InHand) -> MoneyChance {
        let key = k * 5 + usize::from(five);
        if self.stamp[key] == self.current {
            return self.memo[key];
        }
        debug_assert_eq!(COMPS[k], pos.tricks);
        debug_assert!(
            pos.flags_match_scores(),
            "bonus flag must equal score mod 5 == 1"
        );
        self.positions += 1;
        let mut terms = [(0u128, MoneyChance::even()); 8];
        let mut n = 0;
        if pos.tricks_done == engine::game::TRICKS_PER_HAND {
            if pos.five_played {
                let v = ctx.hand_over(pos.scores);
                return self.remember(key, v);
            }
            // End-of-hand bonus to the team holding the best trump (weights sum to 2^32).
            let pb = self.model.end_bonus[k - N_BEFORE_TRICK];
            for (b, &p) in pb.iter().enumerate() {
                if p == 0 {
                    continue;
                }
                let v = match pos.finish(Some(b as u8), self.sc) {
                    AfterHand::GameWon(t, _) => ctx.game_end[usize::from(t)],
                    AfterHand::HandOver(s) => {
                        debug_assert!(
                            InHand {
                                scores: s,
                                flags: pos.flags_after_bonus(Some(b as u8)),
                                ..pos
                            }
                            .flags_match_scores()
                        );
                        ctx.hand_over(s)
                    }
                };
                terms[n] = (u128::from(p), v);
                n += 1;
            }
        } else {
            // Next trick: winner w and whether the 5 of trumps is in it. Weights are exact
            // products of two fixed-point probabilities and sum to exactly 2^64.
            let t = usize::from(pos.tricks_done);
            let pw = self.model.winner[k];
            for (w, &p) in pw.iter().enumerate() {
                if p == 0 {
                    continue;
                }
                let options = if pos.five_played {
                    [(ONE, false), (0, true)]
                } else {
                    let f = self.model.five[t][w];
                    [(f[0], false), (f[1], true)]
                };
                for (pf, here) in options {
                    if pf == 0 {
                        continue;
                    }
                    let v = match pos.play_trick(w as u8, here, self.sc) {
                        AfterTrick::GameWon(team, _) => ctx.game_end[usize::from(team)],
                        AfterTrick::Next(next) | AfterTrick::TricksDone(next) => {
                            let five_next = if here { 1 + w as u8 } else { five };
                            self.node(ctx, comp_index(next.tricks), five_next, next)
                        }
                    };
                    terms[n] = (u128::from(p) * u128::from(pf), v);
                    n += 1;
                }
            }
        }
        let v = MoneyChance::mix_wide(&terms[..n]);
        self.remember(key, v)
    }

    fn remember(&mut self, key: usize, v: MoneyChance) -> MoneyChance {
        self.stamp[key] = self.current;
        self.memo[key] = v;
        v
    }
}

/// Every successor of a canonical state under any trick sequence (ignoring the model's
/// probabilities), in the state's own frame (the next dealer is team 1); finished matches are
/// left out.
pub fn structural_successors(state: &MatchState, sc: Scoring, out: &mut Vec<MatchState>) {
    fn visit(
        pos: InHand,
        five: u8,
        games: [u8; 4],
        sc: Scoring,
        seen: &mut [bool; NODE_SLOTS],
        out: &mut Vec<MatchState>,
    ) {
        let key = comp_index(pos.tricks) * 5 + usize::from(five);
        if seen[key] {
            return;
        }
        seen[key] = true;
        let game_end = |t: u8, out: &mut Vec<MatchState>| {
            let mut g = games;
            g[usize::from(t)] += 1;
            if g[usize::from(t)] < engine::game::GAMES_TO_WIN {
                out.push(MatchState {
                    scores: [0; 4],
                    games: g,
                    dealer_team: 1,
                });
            }
        };
        if pos.tricks_done == engine::game::TRICKS_PER_HAND {
            // Without the 5, the bonus goes to a team that won a trick (the trick holding the
            // best trump); with it, there is no end-of-hand bonus.
            let bonus_teams: Vec<Option<u8>> = if pos.five_played {
                vec![None]
            } else {
                (0..4u8)
                    .filter(|&b| pos.tricks[usize::from(b)] > 0)
                    .map(Some)
                    .collect()
            };
            for b in bonus_teams {
                match pos.finish(b, sc) {
                    AfterHand::GameWon(t, _) => game_end(t, out),
                    AfterHand::HandOver(s) => out.push(MatchState {
                        scores: s,
                        games,
                        dealer_team: 1,
                    }),
                }
            }
            return;
        }
        for w in 0..4u8 {
            for here in [false, true] {
                if here && pos.five_played {
                    continue;
                }
                match pos.play_trick(w, here, sc) {
                    AfterTrick::GameWon(t, _) => game_end(t, out),
                    AfterTrick::Next(p) | AfterTrick::TricksDone(p) => {
                        let f = if here { 1 + w } else { five };
                        visit(p, f, games, sc, seen, out);
                    }
                }
            }
        }
    }
    let mut seen = [false; NODE_SLOTS];
    visit(
        InHand::start(state.scores),
        0,
        state.games,
        sc,
        &mut seen,
        out,
    );
}

/// The canonical indices of the given states and of every state reachable from them, sorted.
pub fn reachable_closure(roots: &[MatchState], sc: Scoring) -> Vec<usize> {
    let mut seen = vec![false; TABLE_STATES];
    let mut stack = Vec::new();
    for r in roots {
        if let Some(i) = r.canonical_index()
            && !seen[i]
        {
            seen[i] = true;
            stack.push(i);
        }
    }
    let mut succ = Vec::new();
    while let Some(i) = stack.pop() {
        succ.clear();
        structural_successors(&MatchState::from_canonical_index(i), sc, &mut succ);
        for s in &succ {
            if let Some(j) = s.canonical_index()
                && !seen[j]
            {
                seen[j] = true;
                stack.push(j);
            }
        }
    }
    (0..TABLE_STATES).filter(|&i| seen[i]).collect()
}

/// The sweep order: (games won, most first), then (total points, most first), then index.
/// Returns the states grouped into levels of equal (games, points).
pub fn levels(indices: &[usize]) -> Vec<Vec<usize>> {
    let mut keyed: Vec<(u32, u32, usize)> = indices
        .iter()
        .map(|&i| {
            let s = MatchState::from_canonical_index(i);
            (s.total_games(), s.total_points(), i)
        })
        .collect();
    keyed.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut last = None;
    for (g, p, i) in keyed {
        if last != Some((g, p)) {
            out.push(Vec::new());
            last = Some((g, p));
        }
        if let Some(level) = out.last_mut() {
            level.push(i);
        }
    }
    out
}

/// Solve one state from the table being built.
fn solve_from_table(solver: &mut Solver<'_>, table: &EquityTable, i: usize) -> MoneyChance {
    let value = |j: usize| {
        table.get(j).unwrap_or_else(|| {
            panic!(
                "successor {} is not solved yet (sweep order or scope bug)",
                MatchState::from_canonical_index(j)
            )
        })
    };
    solver.solve(&MatchState::from_canonical_index(i), &value)
}

/// Run the sweep.
///
/// # Panics
/// If the compiled model is not normalised.
pub fn sweep(model: &CompiledModel, sc: Scoring, opts: &SweepOptions) -> (EquityTable, SweepStats) {
    assert!(
        model.check(),
        "every model distribution must sum to exactly 2^32"
    );
    let indices: Vec<usize> = match &opts.scope {
        Scope::Full => (0..TABLE_STATES).collect(),
        Scope::ReachableFrom(roots) => reachable_closure(roots, sc),
    };
    let levels = levels(&indices);
    let threads = opts.threads.max(1);
    let mut table = EquityTable::unsolved();
    let mut stats = SweepStats {
        states: indices.len() as u64,
        levels: levels.len() as u64,
        largest_level: levels.iter().map(|l| l.len() as u64).max().unwrap_or(0),
        in_hand_positions: 0,
    };
    for level in &levels {
        let (values, positions) = solve_level(model, sc, &table, level, threads);
        for (&i, v) in level.iter().zip(values) {
            table.set(i, v);
        }
        stats.in_hand_positions += positions;
    }
    (table, stats)
}

/// Solve one level. Threads take blocks of states from a shared counter (so the load balances),
/// each block's values are tagged with its block number, and the blocks are put back in order:
/// the values do not depend on which thread computed them, so the result is the same for any
/// thread count.
fn solve_level(
    model: &CompiledModel,
    sc: Scoring,
    table: &EquityTable,
    level: &[usize],
    threads: usize,
) -> (Vec<MoneyChance>, u64) {
    const BLOCK: usize = 64;
    let run_block = |solver: &mut Solver<'_>, block: &[usize]| -> Vec<MoneyChance> {
        block
            .iter()
            .map(|&i| solve_from_table(solver, table, i))
            .collect()
    };
    if threads == 1 || level.len() <= BLOCK {
        let mut solver = Solver::new(model, sc);
        let v = run_block(&mut solver, level);
        return (v, solver.positions);
    }
    let blocks: Vec<&[usize]> = level.chunks(BLOCK).collect();
    let next = AtomicUsize::new(0);
    let workers = threads.min(blocks.len());
    let mut done: Vec<(usize, Vec<MoneyChance>)> = Vec::with_capacity(blocks.len());
    let mut positions = 0;
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                s.spawn(|| {
                    let mut solver = Solver::new(model, sc);
                    let mut mine = Vec::new();
                    loop {
                        let b = next.fetch_add(1, Ordering::Relaxed);
                        if b >= blocks.len() {
                            break;
                        }
                        mine.push((b, run_block(&mut solver, blocks[b])));
                    }
                    (mine, solver.positions)
                })
            })
            .collect();
        for h in handles {
            let (mine, p) = h.join().expect("a sweep thread panicked");
            done.extend(mine);
            positions += p;
        }
    });
    done.sort_by_key(|(b, _)| *b);
    let values = done.into_iter().flat_map(|(_, v)| v).collect();
    (values, positions)
}
