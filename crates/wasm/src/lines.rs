//! Each deal's principal line, for the coach's fields (docs/WASM-API.md §4, Stage B; contract §1
//! and §2): who wins the current trick, each pair's points at the end of the hand, whether a game
//! ends in the hand, and in which trick the acting seat plays each of its cards.
//!
//! The principal line of a candidate in a deal is the line the search's value comes from: after
//! the candidate, every seat in turn plays the card the solver chooses there (its team's best,
//! ties to the first card in the tie order: `Solution::best`), a robber burns its team's best card
//! (ties to the lowest). It is walked by solving again at each step with the same evaluator and
//! frontier as the candidate's own value, so it is the line behind that value (a max^n backup only
//! copies a child's vector). It ends at the end of the hand (exact search), or at the network's
//! frontier (the hybrid): there the network's outcome head gives each way the hand can end its
//! probability, and the points and the game's end are their mean (contract §6.3: a game won by a
//! pair credits it the points that reach 41 and every other pair its points at the frontier, a
//! lower bound). Everything is in whole numbers: probabilities in units of 2^-32 from the network's
//! integer softmax weights, so merged sums are exact and the bytes identical everywhere.

use std::collections::BTreeMap;

use engine::cards::Card;
use engine::game::{Action, Event, GameEndReason, Match, Phase, WINNING_SCORE};
use engine::team_of;
use equity::{MatchState, MoneyChance, ONE};
use nn::Oracle;
use nn::encode::encode;
use nn::oracle::Head;
use nn::outcome::{GAME_WON, compositions, outcomes};
use search::pimc::{Candidate, DynLeaf, SharedLeaf};
use solver::{FastOptions, FrontierCut, Rob, SolveMem, canonical, from_match, solve_fast_cut};

/// What one deal's principal line says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Line {
    /// The seat that wins the decision's trick (`None`: the game ended first).
    pub winner: Option<u8>,
    /// Each pair's points in the hand at its end (the mean over the network's endings beyond the
    /// frontier), in units of 2^-32 points.
    pub points_fp: [u128; 4],
    /// The chance the game ends in this hand (2^32: certain).
    pub game_end_fp: u128,
    /// The acting seat's cards played on the line, with the trick (0-based).
    pub plays: Vec<(Card, u8)>,
    /// The money chance where the line ends (the leaf after the hand, or the network's value at
    /// the frontier): the candidate's value in this deal, which the tests check.
    pub end_value: MoneyChance,
}

/// The line's sums over deals, for one candidate.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LineSums {
    /// Deals in which each seat wins the decision's trick.
    pub winner: [u64; 8],
    /// Σ of each pair's points (units of 2^-32 points).
    pub points_fp: [u128; 4],
    /// Σ of the chance the game ends in the hand (units of 2^-32).
    pub game_end_fp: u128,
    /// The acting seat's card → deals in which the line plays it in trick 0..5.
    pub plan: BTreeMap<Card, [u64; 5]>,
}

impl LineSums {
    /// Add one deal's line.
    pub fn add(&mut self, l: &Line) {
        if let Some(w) = l.winner {
            self.winner[usize::from(w)] += 1;
        }
        for (s, x) in self.points_fp.iter_mut().zip(l.points_fp) {
            *s += x;
        }
        self.game_end_fp += l.game_end_fp;
        for &(c, t) in &l.plays {
            self.plan.entry(c).or_default()[usize::from(t)] += 1;
        }
    }

    /// Add another candidate's sums over other deals (merging shards).
    pub fn merge(&mut self, o: &LineSums) {
        for (s, x) in self.winner.iter_mut().zip(o.winner) {
            *s += x;
        }
        for (s, x) in self.points_fp.iter_mut().zip(o.points_fp) {
            *s += x;
        }
        self.game_end_fp += o.game_end_fp;
        for (c, v) in &o.plan {
            let e = self.plan.entry(*c).or_default();
            for (s, x) in e.iter_mut().zip(v) {
                *s += x;
            }
        }
    }
}

/// How a line is searched: the leaf, the frontier (the trick it lies at and its network), the
/// memory table's size.
pub struct Walker<'a> {
    /// The solver's leaf (the equity table).
    pub leaf: &'a SharedLeaf,
    /// The hybrid's frontier, if any.
    pub cut: Option<(u8, &'a Oracle)>,
    /// The transposition table's megabytes.
    pub tt_mb: usize,
}

/// The solver's choice for the seat to play in `m`, and its value.
fn solve_best(m: &Match, w: &Walker, mem: &mut SolveMem) -> Card {
    let (mut pos, mctx, pending) = from_match(m).expect("a play with trumps known");
    if let Some(s) = pending {
        let card = m.public().table_card.expect("the turned card on the table");
        pos.rob = Some(Rob { seat: s, card });
    }
    let opts = FastOptions {
        tt_mb: w.tt_mb,
        ..FastOptions::default()
    };
    let cut = w.cut.map(|(k, o)| FrontierCut::new(k, o));
    let sol = solve_fast_cut(&pos, mctx, &DynLeaf(w.leaf), opts, cut, mem, None);
    sol.best.expect("nothing stops a solve here")
}

/// The value the robber's team gets from burning `b` (then playing its best), by the solver.
fn burn_value(m: &Match, b: Card, w: &Walker, mem: &mut SolveMem) -> u64 {
    let mut mm = m.clone();
    mm.apply(Action::Burn(b)).expect("a legal burn");
    let seat = mm.to_act().expect("the robber plays");
    let (mut pos, mctx, pending) = from_match(&mm).expect("a play with trumps known");
    if let Some(s) = pending {
        let card = mm
            .public()
            .table_card
            .expect("the turned card on the table");
        pos.rob = Some(Rob { seat: s, card });
    }
    let opts = FastOptions {
        tt_mb: w.tt_mb,
        ..FastOptions::default()
    };
    let cut = w.cut.map(|(k, o)| FrontierCut::new(k, o));
    let sol = solve_fast_cut(&pos, mctx, &DynLeaf(w.leaf), opts, cut, mem, None);
    sol.value
        .expect("nothing stops a solve here")
        .get(team_of(seat))
}

/// The principal line of candidate `cand` in the rebuilt deal `world` (the seat to act is the
/// decider).
pub fn principal_line(world: &Match, cand: Candidate, w: &Walker, mem: &mut SolveMem) -> Line {
    let seat = world.to_act().expect("a decision");
    let hand = world.hand_index();
    let start = world.current_hand_start().scores;
    let trick0 = world.public().trick_no.max(1);
    let mut m = world.clone();
    let from = m.events().len();
    let mut plays = Vec::new();
    let record = |m: &Match, c: Card, plays: &mut Vec<(Card, u8)>| {
        let t = m.public().trick_no.max(1) - 1;
        plays.push((c, t));
    };
    m.apply(cand.0).expect("a candidate is legal");
    if let Action::Play(c) = cand.0 {
        record(world, c, &mut plays);
    }
    if let Some(c) = cand.1 {
        let before = m.clone();
        m.apply(Action::Play(c)).expect("its play is legal");
        record(&before, c, &mut plays);
    }
    let mut frontier: Option<([u128; 4], u128)> = None;
    let mut end_value: Option<MoneyChance> = None;
    loop {
        if m.phase() == Phase::MatchOver || m.hand_index() != hand {
            break;
        }
        let p = m.public();
        if let Some((k, o)) = w.cut
            && p.trick_no == k
            && p.trick_cards.is_empty()
            && !p.turned_teams.iter().any(|&x| x)
            && matches!(p.phase, Phase::AwaitPlay(_))
        {
            frontier = Some(beyond(&m, o, w.leaf, &start));
            let (pos, mctx, _) = from_match(&m).expect("a frontier position");
            end_value = Some(FrontierCut::new(k, o).value(&pos, &mctx));
            break;
        }
        match p.phase {
            Phase::AwaitPlay(s) => {
                let c = solve_best(&m, w, mem);
                if s == seat {
                    record(&m, c, &mut plays);
                }
                m.apply(Action::Play(c))
                    .expect("the solver plays legal cards");
            }
            Phase::AwaitBurn(_) => {
                let legal = m.legal_actions();
                let mut best: Option<(u64, Card)> = None;
                for a in legal {
                    let v = burn_value(&m, a.card(), w, mem);
                    if best.is_none_or(|(bv, _)| v > bv) {
                        best = Some((v, a.card()));
                    }
                }
                let (_, b) = best.expect("a burn");
                m.apply(Action::Burn(b)).expect("a legal burn");
            }
            Phase::MatchOver => unreachable!("checked above"),
        }
    }
    // What happened on the line.
    let mut winner = None;
    let mut ended: Option<[u128; 4]> = None;
    for e in &m.events()[from..] {
        match *e {
            (_, Event::TrickWon { trick, seat, .. }) if trick == trick0 && winner.is_none() => {
                winner = Some(seat);
            }
            (
                _,
                Event::GameWon {
                    team,
                    reason,
                    scores,
                },
            ) if ended.is_none() => {
                let mut pts: [u128; 4] = core::array::from_fn(|t| {
                    u128::from(scores[t].saturating_sub(start[t])) * u128::from(ONE)
                });
                if reason == GameEndReason::Jink {
                    // The app credits a jink with what the pair still needs, at least 10.
                    let t = usize::from(team);
                    let p = u128::from(scores[t].saturating_sub(start[t]));
                    let need = u128::from(WINNING_SCORE.saturating_sub(start[t]));
                    pts[t] = (p + 10).max(need) * u128::from(ONE);
                }
                ended = Some(pts);
            }
            _ => {}
        }
    }
    let (points_fp, game_end_fp) = match (ended, frontier) {
        (Some(pts), _) => (pts, u128::from(ONE)),
        (None, Some(f)) => f,
        (None, None) => {
            // The hand ended and the game goes on: the scores now are the hand's end.
            let now = m.scores();
            (
                core::array::from_fn(|t| {
                    u128::from(now[t].saturating_sub(start[t])) * u128::from(ONE)
                }),
                0,
            )
        }
    };
    let end_value = end_value.unwrap_or_else(|| {
        // The solver's leaf after the hand: the match's money if it is over, else the equity of
        // the state the hand leads to.
        let state = MatchState::from_public(&m.public());
        match state.winner() {
            Some(t) => MoneyChance::certain(t),
            None => w.leaf.value(&state),
        }
    });
    Line {
        winner,
        points_fp,
        game_end_fp,
        plays,
        end_value,
    }
}

/// Beyond the frontier: each pair's expected points in the hand and the chance the game ends,
/// from the network's outcome head (units of 2^-32). A network with the plain money head gives
/// no endings: the points at the frontier and no game's end.
fn beyond(m: &Match, o: &Oracle, leaf: &SharedLeaf, start: &[u8; 4]) -> ([u128; 4], u128) {
    let (pos, mctx, _) = from_match(m).expect("a frontier position");
    let c = canonical(&pos);
    let at: [u128; 4] = core::array::from_fn(|t| u128::from(c.scores[t].saturating_sub(start[t])));
    if o.head != Head::Outcome {
        return (at.map(|x| x * u128::from(ONE)), 0);
    }
    let (f, r) = encode(&c, &mctx, leaf);
    let z = o.net().forward(&f);
    let out = outcomes(&c, &mctx, leaf);
    let live: Vec<usize> = (0..out.n).filter(|&i| out.mask >> i & 1 == 1).collect();
    let top = live
        .iter()
        .map(|&i| z[i])
        .max()
        .expect("a possible outcome");
    let weights: Vec<(usize, u128)> = live
        .iter()
        .map(|&i| (i, u128::from(nn::money::exp_weight((top - z[i]) as u64))))
        .collect();
    let total: u128 = weights.iter().map(|&(_, x)| x).sum();
    // The bonus still to come (contract §6.3; `nn::outcome`): the live 5's holder's pair, else
    // the end-of-hand bonus pair; none once the 5 is played.
    let bonus_team = if c.five_played {
        None
    } else {
        c.slot(0).map(|(s, _)| team_of(s)).or(c.end_bonus)
    };
    let bonus = |t: usize| -> u128 {
        if bonus_team == Some(t as u8) {
            u128::from(if c.flags[t] {
                engine::game::BONUS_LATER
            } else {
                engine::game::BONUS_FIRST
            })
        } else {
            0
        }
    };
    let comps = compositions(engine::game::TRICKS_PER_HAND - c.tricks_done);
    let r = usize::from(r % 4);
    let mut num = [0u128; 4];
    let mut ends = 0u128;
    for &(i, wt) in &weights {
        let pts: [u128; 4] = if i < GAME_WON {
            let winner = (i + 4 - r) % 4;
            ends += wt;
            core::array::from_fn(|t| {
                if t == winner {
                    u128::from(WINNING_SCORE.saturating_sub(start[t]))
                } else {
                    at[t]
                }
            })
        } else {
            let n4 = comps[i - GAME_WON];
            core::array::from_fn(|t| at[t] + 5 * u128::from(n4[(t + r) % 4]) + bonus(t))
        };
        for (s, p) in num.iter_mut().zip(pts) {
            *s += wt * p;
        }
    }
    let one = u128::from(ONE);
    (num.map(|x| x * one / total), ends * one / total)
}
