//! Stratified decision sets and the early-trick yardstick measurements (docs/M3-SPEC.md §5.8,
//! §5.10 strata, §13.1).
//!
//! The pipeline, every step deterministic and resumable in chunks:
//! 1. [`build_decisions`]: decisions at one trick, sampled from `rules-v1` self-play hands that
//!    start from stratified match states (S1–S4, a quarter each). Stored as (start state, deck,
//!    actions so far, seat) so they replay exactly.
//! 2. [`world_rows`]: per decision, the value of every candidate in worlds `from..to` by one method
//!    (exact PIMC or rollout-PIMC). World `w` of decision `d` under label `L` is drawn from
//!    `Rng::derive(dseed, &[WORLD, w])` with `dseed = Rng::derive(seed, &[AGREE, L, d])`, so world
//!    sets are nested (N worlds are the first N of 2N) and a chunk can be redone identically.
//! 3. [`agreement_curve`]: the plateau rule of §13.1 on nested world sets.
//! 4. [`score`]: a method against the yardstick (exact PIMC at the plateau N): agreement, mean
//!    money-chance loss, loss distribution, and paired comparisons with confidence intervals.

use engine::cards::DECK_SIZE;
use engine::game::{Action, HandStartState, Match, Phase, Rules};
use engine::rng::Rng;
use engine::view::SeatView;
use equity::{MoneyChance, ONE};
use serde::{Deserialize, Serialize};

use crate::labels;
use crate::pimc::{Candidate, Ctx, Method, SearchConfig, aggregate, candidates, world_values};

/// The four strata of start states (§5.10).
pub fn stratum_of(s: &HandStartState) -> u8 {
    let max = s.scores.iter().copied().max().unwrap_or(0);
    let at1 = |t: usize| s.games_won[t] == 1;
    if (0..4).any(|t| at1(t) && s.scores[t] >= 26) || max >= 31 {
        4
    } else if (0..4).any(at1) && max <= 25 {
        3
    } else if s.games_won.iter().all(|&g| g == 0) && max <= 20 {
        1
    } else {
        2
    }
}

/// Hand-start states visited by `rules-v1` self-play, by stratum (index 0 = S1), in visit order,
/// with repeats (the empirical distribution). Plays until `hands` hand starts are collected.
pub fn collect_starts(hands: u64, seed: u64) -> [Vec<HandStartState>; 4] {
    let mut out: [Vec<HandStartState>; 4] = Default::default();
    let mut n = 0u64;
    let mut k = 0u64;
    while n < hands {
        let mut m = Match::new(
            Rng::derive(seed, &[labels::STRATUM, k]).next_u64(),
            Rules::HOUSE,
        );
        k += 1;
        let mut last = u32::MAX;
        while m.phase() != Phase::MatchOver && n < hands {
            if m.hand_index() != last {
                last = m.hand_index();
                let s = m.current_hand_start();
                out[usize::from(stratum_of(&s) - 1)].push(s);
                n += 1;
            }
            let seat = m.to_act().expect("to act");
            let legal = m.legal_actions();
            let a = bots::decide(&m.view(seat), &Rules::HOUSE, &legal).0;
            m.apply(a).expect("legal");
        }
    }
    out
}

/// One decision of a measurement set.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionSpec {
    /// Index in the set.
    pub id: u64,
    /// The trick (0 = the dealer's burns).
    pub trick: u8,
    /// The stratum of the start state (1–4).
    pub stratum: u8,
    /// The hand's start state.
    pub start: HandStartState,
    /// The hand's deck in deal order.
    pub deck: Vec<u8>,
    /// The hand's actions before the decision.
    pub actions: Vec<Action>,
    /// The deciding seat.
    pub seat: u8,
    /// Position in the trick: cards already played to it (0 = leader … 7 = eighth to play).
    pub position: u8,
}

impl DecisionSpec {
    /// Replay the decision: the deciding seat's view and its legal actions.
    pub fn replay(&self) -> (Match, SeatView, Vec<Action>) {
        let mut deck = [0u8; DECK_SIZE];
        deck.copy_from_slice(&self.deck);
        let mut m = Match::from_hand_start(self.start, deck, 0, Rules::HOUSE).expect("valid");
        for &a in &self.actions {
            m.apply(a).expect("recorded actions are legal");
        }
        let v = m.view(self.seat);
        let legal = m.legal_actions();
        (m, v, legal)
    }
}

/// `count` decisions at trick `trick`, equal numbers per stratum and per position in the trick
/// (§13.1 item 6): decision `i` is from stratum `i mod 4 + 1` at position `(i / 4) mod 8`; each a uniformly chosen decision node (two or more candidates) of a `rules-v1`
/// self-play hand from a start state drawn from that stratum's visited states.
pub fn build_decisions(
    trick: u8,
    count: u64,
    seed: u64,
    starts: &[Vec<HandStartState>; 4],
) -> Vec<DecisionSpec> {
    let mut out = Vec::new();
    for i in 0..count {
        let st = (i % 4) as usize;
        let pos = ((i / 4) % 8) as usize;
        assert!(!starts[st].is_empty(), "stratum S{} has no states", st + 1);
        let mut rng = Rng::derive(seed, &[labels::MEASURE, u64::from(trick), i]);
        loop {
            let start = starts[st][rng.bounded(starts[st].len() as u64) as usize];
            let mut deck: [u8; DECK_SIZE] = core::array::from_fn(|k| k as u8);
            rng.shuffle(&mut deck);
            let mut m = Match::from_hand_start(start, deck, 0, Rules::HOUSE).expect("valid");
            let h = m.hand_index();
            let mut nodes: Vec<(Vec<Action>, u8)> = Vec::new();
            while m.phase() != Phase::MatchOver && m.hand_index() == h {
                let seat = m.to_act().expect("to act");
                let v = m.view(seat);
                let legal = m.legal_actions();
                if v.public.trick_no == trick
                    && v.public.trick_cards.len() == pos
                    && candidates(&v, &legal).1.len() >= 2
                {
                    nodes.push((m.actions().to_vec(), seat));
                }
                if v.public.trick_no > trick {
                    break;
                }
                let a = bots::decide(&v, &Rules::HOUSE, &legal).0;
                m.apply(a).expect("legal");
            }
            if nodes.is_empty() {
                continue;
            }
            let (actions, seat) = nodes.swap_remove(rng.bounded(nodes.len() as u64) as usize);
            out.push(DecisionSpec {
                id: i,
                trick,
                stratum: st as u8 + 1,
                start,
                deck: deck.to_vec(),
                actions,
                seat,
                position: pos as u8,
            });
            break;
        }
    }
    out
}

/// The decision seed of decision `id` under world label `label` (1 = the reference / yardstick
/// worlds, others for methods and repeatability).
pub fn measure_seed(seed: u64, label: u64, id: u64) -> u64 {
    Rng::derive(seed, &[labels::AGREE, label, id]).next_u64()
}

/// Per-world candidate values of one decision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldRows {
    /// The decision's id.
    pub id: u64,
    /// The trick.
    pub trick: u8,
    /// The stratum.
    pub stratum: u8,
    /// Position in the trick (0 = leader).
    pub position: u8,
    /// The deciding team.
    pub team: u8,
    /// `exact` or `rollout`.
    pub method: String,
    /// World label.
    pub label: u64,
    /// First world.
    pub from: u64,
    /// The candidates, in candidate order.
    pub cands: Vec<Candidate>,
    /// One row per world (from `from`), each the candidates' values (four u64 per value).
    pub rows: Vec<Vec<[u64; 4]>>,
    /// Wall time of the whole decision, in milliseconds (reports only).
    pub ms: u64,
}

/// Value every candidate of `d` in worlds `from..to` by `method`.
#[allow(clippy::too_many_arguments)]
pub fn world_rows(
    d: &DecisionSpec,
    method: Method,
    label: u64,
    seed: u64,
    from: u64,
    to: u64,
    ctx: &Ctx,
    cfg: &SearchConfig,
) -> WorldRows {
    let t0 = std::time::Instant::now();
    let (_, view, legal) = d.replay();
    let (_, cands) = candidates(&view, &legal);
    let dseed = measure_seed(seed, label, d.id);
    let rows = world_values(&view, &cands, method, from, to, dseed, ctx, cfg);
    WorldRows {
        id: d.id,
        trick: d.trick,
        stratum: d.stratum,
        position: d.position,
        team: engine::team_of(d.seat),
        method: match method {
            Method::ExactPimc => "exact",
            Method::RolloutPimc => "rollout",
            Method::Forced => "forced",
        }
        .to_string(),
        label,
        from,
        cands,
        rows: rows
            .into_iter()
            .map(|r| r.into_iter().map(|v| v.0).collect())
            .collect(),
        ms: t0.elapsed().as_millis() as u64,
    }
}

fn mc(v: [u64; 4]) -> MoneyChance {
    MoneyChance::from_raw(v).expect("stored values sum to one")
}

/// Mean values over the first `n` worlds, and the chosen index.
pub fn mean_first(r: &WorldRows, n: usize) -> (Vec<MoneyChance>, usize) {
    let rows: Vec<Vec<MoneyChance>> = r.rows[..n]
        .iter()
        .map(|row| row.iter().map(|&v| mc(v)).collect())
        .collect();
    let (m, c, _) = aggregate(&rows, r.cands.len(), r.team);
    (m, c)
}

/// Is candidate `i` among the tied-best for `team`?
pub fn tied_best(v: &[MoneyChance], team: u8, i: usize) -> bool {
    let best = v.iter().map(|x| x.get(team)).max().unwrap_or(0);
    v[i].get(team) == best
}

/// Money chance as a fraction (reports only).
pub fn frac(x: u64) -> f64 {
    x as f64 / ONE as f64
}

/// One step of the agreement curve: N against 2N.
#[derive(Clone, Debug, Serialize)]
pub struct CurveStep {
    /// Worlds.
    pub n: usize,
    /// Doubled.
    pub n2: usize,
    /// Decisions compared.
    pub decisions: usize,
    /// Share of decisions whose N-world choice is not among the 2N-world tied-best cards.
    pub change_rate: f64,
    /// |mean over decisions of V_2N(choice_2N) − mean of V_N(choice_N)| (§13.1 literal).
    pub mean_value_change: f64,
    /// Mean over decisions of V_2N(choice_2N) − V_2N(choice_N) (the N choice's loss under 2N).
    pub mean_loss: f64,
    /// Standard error of `mean_loss`.
    pub loss_se: f64,
    /// Both §13.1 conditions hold (change < 1%, value change and loss < 0.001).
    pub plateau_ok: bool,
}

/// The agreement curve over nested world sets `grid` (each decision needs `max(grid)` worlds),
/// and the plateau N (§13.1): the smallest N where doubling changes the chosen card in under 1%
/// of decisions and changes the mean chosen-card value by under 0.001. The value condition is
/// applied to both readings, the literal one (difference of the means) and the loss of the
/// N-world choice under the 2N-world values, so cancellation cannot hide a change.
pub fn agreement_curve(rows: &[WorldRows], grid: &[usize]) -> (Vec<CurveStep>, Option<usize>) {
    let mut steps = Vec::new();
    for w in grid.windows(2) {
        let (n, n2) = (w[0], w[1]);
        let use_: Vec<&WorldRows> = rows.iter().filter(|r| r.rows.len() >= n2).collect();
        if use_.is_empty() {
            continue;
        }
        let mut changes = 0usize;
        let (mut s_a, mut s_b) = (0f64, 0f64);
        let mut losses = Vec::new();
        for r in &use_ {
            let (v1, c1) = mean_first(r, n);
            let (v2, c2) = mean_first(r, n2);
            if !tied_best(&v2, r.team, c1) {
                changes += 1;
            }
            s_a += frac(v2[c2].get(r.team));
            s_b += frac(v1[c1].get(r.team));
            losses.push(frac(v2[c2].get(r.team) - v2[c1].get(r.team)));
        }
        let k = use_.len() as f64;
        let (ml, se) = mean_se(&losses);
        let change_rate = changes as f64 / k;
        let mvc = ((s_a - s_b) / k).abs();
        steps.push(CurveStep {
            n,
            n2,
            decisions: use_.len(),
            change_rate,
            mean_value_change: mvc,
            mean_loss: ml,
            loss_se: se,
            plateau_ok: change_rate < 0.01 && mvc < 0.001 && ml < 0.001,
        });
    }
    let plateau = steps.iter().find(|s| s.plateau_ok).map(|s| s.n);
    (steps, plateau)
}

/// Mean and standard error.
pub fn mean_se(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    if x.is_empty() {
        return (0.0, 0.0);
    }
    let m = x.iter().sum::<f64>() / n;
    if x.len() < 2 {
        return (m, 0.0);
    }
    let v = x.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / (n - 1.0);
    (m, (v / n).sqrt())
}

/// A method's per-decision result against the yardstick.
#[derive(Clone, Debug, Serialize)]
pub struct Scored {
    /// Decision id.
    pub id: u64,
    /// Agrees with the yardstick (its choice is among the yardstick's tied-best).
    pub agree: bool,
    /// Loss of its choice under the yardstick values, decider's team, units of 2^-32.
    pub loss: u64,
}

/// Score a method's rows (its own worlds, `n_method` of them) against yardstick rows (the first
/// `n_ref` worlds), on the decisions both have.
pub fn score(
    yard: &[WorldRows],
    n_ref: usize,
    method: &[WorldRows],
    n_method: usize,
) -> Vec<Scored> {
    let mut out = Vec::new();
    for m in method {
        let Some(y) = yard.iter().find(|y| y.id == m.id) else {
            continue;
        };
        if y.rows.len() < n_ref || m.rows.len() < n_method {
            continue;
        }
        assert_eq!(y.cands, m.cands, "same decision, same candidates");
        let (vy, cy) = mean_first(y, n_ref);
        let (_, cm) = mean_first(m, n_method);
        out.push(Scored {
            id: m.id,
            agree: tied_best(&vy, y.team, cm),
            loss: vy[cy].get(y.team) - vy[cm].get(y.team),
        });
    }
    out
}

/// Summary of a scored method.
#[derive(Clone, Debug, Serialize)]
pub struct ScoreSummary {
    /// Decisions.
    pub decisions: usize,
    /// Agreement rate and its standard error.
    pub agreement: (f64, f64),
    /// Mean loss in money chance, and its standard error.
    pub mean_loss: (f64, f64),
    /// Mean loss in units of 2^-32.
    pub mean_loss_units: f64,
    /// Mean loss in stakes per player (8 × money chance: a match is worth +6 or −2 stakes).
    pub mean_loss_stakes: f64,
    /// Loss distribution: share with zero loss, then quantiles p50, p90, p99 and the maximum, in
    /// money chance.
    pub zero_share: f64,
    /// Quantiles (p50, p90, p99, max).
    pub quantiles: [f64; 4],
    /// Histogram of losses: < 0.001, < 0.003, < 0.01, < 0.03, ≥ 0.03 (counts).
    pub histogram: [usize; 5],
}

/// Summarise scored decisions.
pub fn summarise(s: &[Scored]) -> ScoreSummary {
    let agree: Vec<f64> = s.iter().map(|x| f64::from(u8::from(x.agree))).collect();
    let loss: Vec<f64> = s.iter().map(|x| frac(x.loss)).collect();
    let mut sorted = loss.clone();
    sorted.sort_by(f64::total_cmp);
    let q = |p: f64| {
        if sorted.is_empty() {
            0.0
        } else {
            sorted[((sorted.len() - 1) as f64 * p).round() as usize]
        }
    };
    let mut hist = [0usize; 5];
    for &l in &loss {
        let b = [0.001, 0.003, 0.01, 0.03]
            .iter()
            .position(|&e| l < e)
            .unwrap_or(4);
        hist[b] += 1;
    }
    let ml = mean_se(&loss);
    ScoreSummary {
        decisions: s.len(),
        agreement: mean_se(&agree),
        mean_loss: ml,
        mean_loss_units: ml.0 * ONE as f64,
        mean_loss_stakes: ml.0 * 8.0,
        zero_share: s.iter().filter(|x| x.loss == 0).count() as f64 / s.len().max(1) as f64,
        quantiles: [q(0.5), q(0.9), q(0.99), q(1.0)],
        histogram: hist,
    }
}

/// Paired comparison of two scored methods on their common decisions: mean difference (A − B)
/// of agreement and of loss, each with a 95% confidence interval.
#[derive(Clone, Debug, Serialize)]
pub struct Paired {
    /// Common decisions.
    pub decisions: usize,
    /// Agreement difference A − B, and its 95% interval.
    pub agreement_diff: (f64, f64, f64),
    /// Loss difference A − B (money chance), and its 95% interval.
    pub loss_diff: (f64, f64, f64),
    /// Share of decisions where exactly one of the two agrees (the disagreement rate that sizes
    /// the set, §13.1 point 2).
    pub discordant: f64,
}

/// Compare two scored methods, paired by decision.
pub fn paired(a: &[Scored], b: &[Scored]) -> Paired {
    let mut da = Vec::new();
    let mut dl = Vec::new();
    let mut disc = 0usize;
    for x in a {
        if let Some(y) = b.iter().find(|y| y.id == x.id) {
            da.push(f64::from(u8::from(x.agree)) - f64::from(u8::from(y.agree)));
            dl.push(frac(x.loss) - frac(y.loss));
            if x.agree != y.agree {
                disc += 1;
            }
        }
    }
    let (ma, sa) = mean_se(&da);
    let (ml, sl) = mean_se(&dl);
    Paired {
        decisions: da.len(),
        agreement_diff: (ma, ma - 1.96 * sa, ma + 1.96 * sa),
        loss_diff: (ml, ml - 1.96 * sl, ml + 1.96 * sl),
        discordant: disc as f64 / da.len().max(1) as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strata_classify() {
        let s = |scores: [u8; 4], games: [u8; 4]| HandStartState {
            dealer: 0,
            scores,
            bonus_taken: core::array::from_fn(|t| scores[t] % 5 == 1),
            games_won: games,
            hand_index: 0,
            game_index: games.iter().sum(),
        };
        assert_eq!(stratum_of(&s([0, 0, 0, 0], [0; 4])), 1);
        assert_eq!(stratum_of(&s([20, 0, 0, 0], [0; 4])), 1);
        assert_eq!(stratum_of(&s([25, 0, 0, 0], [0; 4])), 2);
        assert_eq!(stratum_of(&s([10, 0, 0, 0], [0, 1, 0, 0])), 3);
        assert_eq!(stratum_of(&s([0, 26, 0, 0], [0, 1, 0, 0])), 4);
        assert_eq!(stratum_of(&s([31, 0, 0, 0], [0; 4])), 4);
        assert_eq!(stratum_of(&s([28, 10, 0, 0], [0, 1, 0, 0])), 2);
    }

    #[test]
    fn decisions_are_deterministic_and_replay() {
        let starts = collect_starts(400, 5);
        assert!(starts.iter().all(|s| !s.is_empty()));
        let a = build_decisions(2, 16, 9, &starts);
        let b = build_decisions(2, 16, 9, &starts);
        assert_eq!(a, b);
        for d in &a {
            let (_, v, legal) = d.replay();
            assert_eq!(v.public.trick_cards.len(), usize::from(d.position));
            assert_eq!(v.public.trick_no, 2);
            assert!(candidates(&v, &legal).1.len() >= 2);
            assert_eq!(v.seat, d.seat);
        }
        let t1 = build_decisions(1, 16, 9, &starts);
        assert!(t1.iter().all(|d| d.replay().1.public.trick_no == 1));
    }

    #[test]
    fn isqrt_exact() {
        for n in [
            0u128,
            1,
            2,
            3,
            4,
            15,
            16,
            17,
            1 << 64,
            (1 << 66) + 12345,
            u128::MAX >> 2,
        ] {
            let r = crate::pimc::isqrt(n);
            assert!(r * r <= n && (r + 1) * (r + 1) > n, "{n}");
        }
    }
}
