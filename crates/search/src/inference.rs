//! Policy-based world weighting (PLAN §6.4 point 3; docs/RESEARCH.md §6, §15): a sampled world
//! is weighted by how likely the other seats' actual plays were, had the world been true, under a
//! model of their play. The model here is the students (docs/STUDENT.md): at a trick-1 or trick-2
//! decision of another seat with all eight seats in play, the seat's student values every
//! candidate from the view it would have had in that world, and the probability of the play it
//! made is a softmax over its own team's values at temperature `tau` with a floor
//! (`(1 − floor) · softmax + floor / n`), so no world's weight is ever zero (PLAN §6.4: a
//! surprising play, a human's or a bot's, never breaks inference). The dealer's burn before
//! trick 1 and every decision from trick 3 on carry no model and weigh 1. A robber's burn and
//! first card are one choice (the app's candidates), read from the two actions together.
//!
//! Everything is integer: the softmax by `nn::money::exp_weight` (2^62·e^(−d/2^24), the oracle
//! head's rule) over `d = (v_max − v) · 2^24 / tau`, probabilities in units of 2^-32, a world's
//! weight as the product of its decisions' probabilities held as a mantissa and a binary
//! exponent, and the worlds' weights scaled so the likeliest is 2^60 ([`WEIGHT_TOP`]), ready for
//! `MoneyChance::mix`. The same bytes on any machine and thread count.

use engine::game::{Action, Match, Phase};
use engine::team_of;
use engine::view::SeatView;
use equity::MoneyChance;
use nn::money::exp_weight;
use student::encode::{encode_cand, encode_view};

use crate::pimc::{Candidate, SharedLeaf, Students, world_match};
use crate::student_teacher::app_candidates;
use crate::worlds::Worlds;

/// The likeliest world's weight; the others fall off from it (a world 2^60 times less likely than
/// the likeliest weighs 0 and is dropped: nothing a 64-bit weight could carry anyway).
pub const WEIGHT_TOP: u64 = 1 << 60;

/// The weighting's settings.
#[derive(Clone, Copy)]
pub struct Inference<'a> {
    /// The play model: the students of tricks 1 and 2.
    pub students: Students<'a>,
    /// The softmax temperature in money units of 2^-32 (`0.01 · 2^32` = 1% of the money).
    pub tau_q32: u64,
    /// The floor in units of 2^-32: the share of the probability spread evenly over the
    /// candidates (`0.05 · 2^32` = 5%).
    pub floor_q32: u64,
}

impl Inference<'_> {
    /// The defaults of the first arena arm: tau 1% of the money, floor 5% (`floor(x · 2^32 / 1000)`
    /// of 10 and 50 permille, as the manifest's unit reads them).
    pub const TAU_DEFAULT_Q32: u64 = 42_949_672;
    pub const FLOOR_DEFAULT_Q32: u64 = 214_748_364;

    /// A name part for records and bots: `inf=tau<permille>/floor<permille>`.
    pub fn name(&self) -> String {
        format!(
            "inf=tau{}/floor{}",
            permille(self.tau_q32),
            permille(self.floor_q32)
        )
    }

    /// The probabilities (units of 2^-32) of each candidate for `team`, from the student's
    /// values: `(1 − floor) · softmax(v / tau) + floor / n`. Every entry is at least `floor / n`
    /// (minus rounding) and the entries sum to about 2^32.
    pub fn probabilities(&self, values: &[MoneyChance], team: u8) -> Vec<u64> {
        let n = values.len() as u128;
        assert!(n > 0, "probabilities of no candidate");
        let own: Vec<u64> = values.iter().map(|v| v.get(team)).collect();
        let vmax = own.iter().copied().max().expect("some");
        let tau = u128::from(self.tau_q32.max(1));
        let weights: Vec<u64> = own
            .iter()
            .map(|&v| {
                let d = (u128::from(vmax - v) << 24) / tau;
                exp_weight(d.min(u128::from(u64::MAX)) as u64)
            })
            .collect();
        let total: u128 = weights.iter().map(|&w| u128::from(w)).sum();
        let one = 1u128 << 32;
        let keep = one - u128::from(self.floor_q32.min(1 << 32));
        let spread = u128::from(self.floor_q32.min(1 << 32)) / n;
        weights
            .iter()
            .map(|&w| {
                let p = (u128::from(w) << 32) / total;
                ((keep * p) >> 32) as u64 + spread as u64
            })
            .collect()
    }
}

fn permille(q32: u64) -> u64 {
    (u128::from(q32) * 1000 + (1 << 31)) as u64 >> 32
}

/// A weight as `mant · 2^(exp − 63)` with `mant` in `[2^63, 2^64)` (or 0 only for a zero weight,
/// which the floor rules out).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogWeight {
    mant: u64,
    exp: i64,
}

impl LogWeight {
    /// Weight 1.
    pub const ONE: LogWeight = LogWeight {
        mant: 1 << 63,
        exp: 0,
    };

    /// The weight's mantissa and exponent (`mant · 2^(exp − 63)`), to carry it whole between
    /// processes (the WASM API's shard results).
    pub fn parts(self) -> (u64, i64) {
        (self.mant, self.exp)
    }

    /// The weight from [`LogWeight::parts`]; `None` unless `mant` is in `[2^63, 2^64)`.
    pub fn from_parts(mant: u64, exp: i64) -> Option<LogWeight> {
        (mant >> 63 == 1).then_some(LogWeight { mant, exp })
    }

    /// Multiplied by a probability in units of 2^-32.
    fn times_q32(self, p: u64) -> LogWeight {
        let p = p.max(1);
        let prod = u128::from(self.mant) * u128::from(p); // < 2^96
        let shift = 128 - prod.leading_zeros() as i64 - 64; // bits above 64
        let mant = (prod >> shift) as u64;
        LogWeight {
            mant,
            exp: self.exp + shift - 32,
        }
    }

    /// `other` in units where `self` (the largest) is exactly `WEIGHT_TOP`.
    fn scale(self, other: LogWeight) -> u64 {
        debug_assert!(other.exp <= self.exp);
        let down = self.exp - other.exp; // >= 0
        if down >= 64 {
            return 0;
        }
        // other.mant / self.mant in [2^-1, 1], times 2^60, then down by the exponent gap.
        let rel = (u128::from(other.mant) << 60) / u128::from(self.mant);
        (rel >> down) as u64
    }
}

/// The weight of one world: the product of the probabilities, under the students, of every play
/// another seat made in this hand at tricks 1-2 with all eight seats in play (the decider's own
/// plays and every other decision weigh 1). `world` is a rebuilt world at the decision
/// (`world_match`); `decider` the seat deciding.
pub fn world_weight(
    world: &Match,
    decider: u8,
    inf: &Inference,
    leaf: &SharedLeaf,
) -> Result<LogWeight, String> {
    let start = world.current_hand_start();
    let deck = world.truth().deck;
    let mut m = Match::from_hand_start(start, deck, 0, world.rules()).map_err(|e| e.0)?;
    let actions = world.actions();
    let mut w = LogWeight::ONE;
    let mut k = 0;
    while k < actions.len() {
        let a = actions[k];
        let Some(seat) = m.to_act() else {
            return Err("the world's history runs past the hand".into());
        };
        let view = m.view(seat);
        let p = &view.public;
        let modelled =
            seat != decider && (1..=2).contains(&p.trick_no) && p.turned_teams.iter().all(|&t| !t);
        let legal = m.legal_actions();
        // A robber's burn and first card are one choice: read both actions.
        let joint = matches!(p.phase, Phase::AwaitBurn(_)) && p.trick_no == 1;
        let played: Candidate = if joint {
            let next = actions.get(k + 1).copied();
            (a, next.map(Action::card))
        } else {
            (a, None)
        };
        if modelled && let Some(net) = inf.students.for_trick(p.trick_no) {
            let (_, cands) = app_candidates(&view, &legal);
            if cands.len() >= 2 {
                let f = encode_view(&view, leaf)?;
                let cf: Vec<Vec<u16>> = cands.iter().map(|&c| encode_cand(&view, c)).collect();
                let values = net.evaluate(&f, &cf);
                let probs = inf.probabilities(&values, team_of(seat));
                let p_played = match cands.iter().position(|&c| c == played) {
                    Some(i) => probs[i],
                    // Not among the app's candidates (a robber burning the card it took): the
                    // floor's share alone.
                    None => (inf.floor_q32 / cands.len() as u64).max(1),
                };
                w = w.times_q32(p_played);
            }
        }
        m.apply(a).map_err(|e| format!("{e:?}"))?;
        k += 1;
        if joint && let Some(c) = played.1 {
            m.apply(Action::Play(c)).map_err(|e| format!("{e:?}"))?;
            k += 1;
        }
    }
    Ok(w)
}

/// The weights of worlds `from..to` of the decision in `view` (the same deals as
/// `crate::pimc::world_values` draws for `dseed`), scaled so the likeliest is [`WEIGHT_TOP`];
/// index order, one thread, so the result is the same for any search thread count.
#[allow(clippy::too_many_arguments)]
pub fn world_weights(
    view: &SeatView,
    worlds: &dyn Worlds,
    from: u64,
    to: u64,
    dseed: u64,
    rules: engine::game::Rules,
    inf: &Inference,
    leaf: &SharedLeaf,
) -> Vec<u64> {
    let logs: Vec<LogWeight> = (from..to)
        .map(|w| {
            let m = world_match(view, worlds, w, dseed, rules);
            world_weight(&m, view.seat, inf, leaf)
                .expect("a sampled world replays and its views are consistent")
        })
        .collect();
    scale_weights(&logs)
}

/// [`LogWeight`]s as `u64` weights with the largest at [`WEIGHT_TOP`].
pub fn scale_weights(logs: &[LogWeight]) -> Vec<u64> {
    let Some(top) = logs
        .iter()
        .copied()
        .max_by(|a, b| a.exp.cmp(&b.exp).then(a.mant.cmp(&b.mant)))
    else {
        return Vec::new();
    };
    logs.iter().map(|&l| top.scale(l)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use nn::synthetic::SyntheticLeaf;
    use student::net::random_net;

    fn students() -> (student::net::StudentNet, student::net::StudentNet) {
        (random_net(4, &[16, 8], 12), random_net(5, &[16, 8], 12))
    }

    /// A student that prefers the higher card (by the encoder's canonical rank) whatever the
    /// position: layer 0's unit 0 reads the candidate's play feature with weight 54 − rank,
    /// layer 1 passes it to the deciding pair's logit, so adjacent ranks differ by 1/16 of a
    /// natural logit and the values spread from about 25% to 90% of the money. The random
    /// networks of `random_net` saturate and value every candidate alike, so they cannot test
    /// a weighting; this one tells worlds apart by which cards a seat could have chosen from.
    fn ranked_student() -> student::net::StudentNet {
        use nn::mlp::Layer;
        use student::encode::{N_CANON, N_FEATURES, OFF_C_PLAY};
        let n_out = 2;
        let mut w0 = vec![0i32; N_FEATURES * n_out];
        for k in 0..N_CANON {
            w0[(OFF_C_PLAY + k) * n_out] = (N_CANON - k) as i32;
        }
        let l0 = Layer {
            n_in: N_FEATURES,
            n_out,
            w: w0,
            b: vec![0; n_out],
            shift: 4, // 2^24 inputs to 2^20 activations
        };
        let mut w1 = vec![0i32; n_out * 4];
        w1[0] = 1; // unit 0 to logit 0 (the deciding pair): 1/16 of a natural logit a rank
        let l1 = Layer {
            n_in: n_out,
            n_out: 4,
            w: w1,
            b: vec![0; 4],
            shift: 0,
        };
        student::net::StudentNet::new(vec![l0, l1]).expect("a small valid network")
    }

    fn chance(own: u64) -> MoneyChance {
        let rest = ((1u64 << 32) - own) / 3;
        let mut v = [rest; 4];
        v[0] = own;
        v[3] = (1u64 << 32) - own - 2 * rest;
        MoneyChance(v)
    }

    #[test]
    fn probabilities_are_a_floored_softmax_over_the_teams_values() {
        let (s1, s2) = students();
        let inf = Inference {
            students: Students::two(&s1, &s2),
            tau_q32: Inference::TAU_DEFAULT_Q32,
            floor_q32: Inference::FLOOR_DEFAULT_Q32,
        };
        // Equal values: equal shares.
        let p = inf.probabilities(&[chance(1 << 30); 4], 0);
        assert!(p.iter().all(|&x| x.abs_diff(1 << 30) <= 4), "{p:?}");
        // One candidate 1% of the money better: e^1 more likely before the floor.
        let a = chance(1 << 30);
        let b = chance((1 << 30) + Inference::TAU_DEFAULT_Q32);
        let p = inf.probabilities(&[a, b], 0);
        let e = std::f64::consts::E;
        let soft_b = e / (1.0 + e);
        let want_b = 0.95 * soft_b + 0.025;
        let got_b = p[1] as f64 / (1u64 << 32) as f64;
        assert!((got_b - want_b).abs() < 1e-6, "{got_b} vs {want_b}");
        assert!((p[0] + p[1]).abs_diff(1 << 32) <= 8);
        // A hopeless candidate keeps the floor's share.
        let c = chance((1 << 30) + 40 * Inference::TAU_DEFAULT_Q32);
        let p = inf.probabilities(&[a, c], 0);
        let floor_share = Inference::FLOOR_DEFAULT_Q32 / 2;
        assert!(p[0] >= floor_share && p[0] <= floor_share + 16, "{p:?}");
        // The floor at 100% makes every play equally likely whatever the values.
        let flat = Inference {
            floor_q32: 1 << 32,
            ..inf
        };
        let p = flat.probabilities(&[a, c, b], 0);
        assert!(
            p.iter().all(|&x| x.abs_diff((1u64 << 32) / 3) <= 4),
            "{p:?}"
        );
        assert_eq!(inf.name(), "inf=tau10/floor50");
    }

    #[test]
    fn log_weights_multiply_and_scale_exactly() {
        let one = LogWeight::ONE;
        let half = one.times_q32(1 << 31);
        assert_eq!(
            half,
            LogWeight {
                mant: 1 << 63,
                exp: -1
            }
        );
        let quarter = half.times_q32(1 << 31);
        assert_eq!(
            scale_weights(&[one, half, quarter]),
            [1 << 60, 1 << 59, 1 << 58]
        );
        // Three quarters: a mantissa with bits below the top.
        let tq = one.times_q32(3 << 30);
        assert_eq!(scale_weights(&[one, tq]), [1 << 60, 3 << 58]);
        // Sixty-four halvings and more vanish.
        let mut tiny = one;
        for _ in 0..70 {
            tiny = tiny.times_q32(1 << 31);
        }
        assert_eq!(scale_weights(&[one, tiny]), [1 << 60, 0]);
        // Scaling is relative: the largest is always the top.
        assert_eq!(scale_weights(&[tiny, tiny]), [1 << 60, 1 << 60]);
        assert!(scale_weights(&[]).is_empty());
    }

    /// A match the students play to a trick-3 decision: the true deal's weight is at least the
    /// median sampled world's (the plays were the students' own), the floor at 100% weighs every
    /// world the same, and a sharper temperature spreads the weights more.
    #[test]
    fn the_true_world_of_a_students_hand_weighs_at_least_the_median_sample() {
        use crate::bot::{AppPlay, PimcBot};
        use crate::pimc::SearchConfig;
        use crate::worlds::{BeliefSource, WorldSource};
        use bot::Bot;
        use engine::game::Rules;
        use engine::rng::Rng;

        let (s1, s2) = (Arc::new(ranked_student()), Arc::new(ranked_student()));
        let leaf: Arc<SharedLeaf> = Arc::new(SyntheticLeaf);
        let source = crate::bot::default_source();
        let app = Arc::new(AppPlay {
            s1: s1.clone(),
            s2: s2.clone(),
            deals3: [1; 8],
            deals45: 1,
            inference: None,
            s3: None,
            s4: None,
        });
        let cfg = SearchConfig {
            w12: 1,
            w34: 1,
            t1w: 1,
            t2w: 1,
            tt_mb: 1,
            ..SearchConfig::default()
        };
        let inf = Inference {
            students: Students::two(&s1, &s2),
            tau_q32: Inference::TAU_DEFAULT_Q32,
            floor_q32: Inference::FLOOR_DEFAULT_Q32,
        };
        let belief = BeliefSource::new();
        let (mut at_least_median, mut hands) = (0, 0);
        for seed in 1..9u64 {
            let mut m = Match::new(seed, Rules::HOUSE);
            let mut bots: Vec<PimcBot> = (0..8)
                .map(|_| {
                    PimcBot::new("app".into(), cfg, leaf.clone(), source.clone())
                        .with_app(app.clone())
                })
                .collect();
            for (s, b) in (0u8..).zip(bots.iter_mut()) {
                b.start_match(
                    s,
                    Rules::HOUSE,
                    Rng::derive(seed, &[7, u64::from(s)]).next_u64(),
                );
            }
            // Play to the first trick-3 decision of the hand.
            loop {
                let seat = m.to_act().expect("acts");
                let view = m.view(seat);
                if view.public.trick_no >= 3 {
                    break;
                }
                let legal = m.legal_actions();
                let a = bots[usize::from(seat)].act(&view, &legal);
                m.apply(a).expect("legal");
            }
            let seat = m.to_act().expect("acts");
            let view = m.view(seat);
            if view.public.turned_teams.iter().any(|&t| t) {
                continue;
            }
            hands += 1;
            let truth = world_weight(&m, seat, &inf, &*leaf).expect("the true world");
            let worlds = belief.prepare(&view, Rules::HOUSE);
            let logs: Vec<LogWeight> = (0..24u64)
                .map(|w| {
                    let wm = world_match(&view, &*worlds, w, 99, Rules::HOUSE);
                    world_weight(&wm, seat, &inf, &*leaf).expect("a world")
                })
                .collect();
            let mut all = logs.clone();
            all.push(truth);
            let scaled = scale_weights(&all);
            let truth_w = scaled[24];
            let mut sampled = scaled[..24].to_vec();
            sampled.sort_unstable();
            at_least_median += usize::from(truth_w >= sampled[12]);
            // The flat model (floor 100%) reads only how many cards each seat could choose from
            // in each world (1/n a play): no world is dropped, and the spread is far smaller
            // than the students' model's.
            let flat = Inference {
                floor_q32: 1 << 32,
                ..inf
            };
            let fw = world_weights(&view, &*worlds, 0, 8, 99, Rules::HOUSE, &flat, &*leaf);
            assert!(fw.iter().all(|&w| w > WEIGHT_TOP >> 20), "{fw:?}");
            assert_eq!(*fw.iter().max().expect("some"), WEIGHT_TOP);
            // Weights in index order, the same for a second call.
            let w1 = world_weights(&view, &*worlds, 0, 8, 99, Rules::HOUSE, &inf, &*leaf);
            let w2 = world_weights(&view, &*worlds, 0, 8, 99, Rules::HOUSE, &inf, &*leaf);
            assert_eq!(w1, w2);
            assert_eq!(*w1.iter().max().expect("some"), WEIGHT_TOP);
        }
        assert!(
            hands >= 4,
            "{hands} hands reached trick 3 with eight seats in play"
        );
        assert!(
            at_least_median * 3 >= hands * 2,
            "the true world weighed at least the median sample in {at_least_median} of {hands}"
        );
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;
    use nn::synthetic::SyntheticLeaf;
    use student::net::random_net;

    #[test]
    fn debug_nets() {
        use engine::game::{Match, Rules};
        let leaf = SyntheticLeaf;
        let m = Match::new(3, Rules::HOUSE);
        let mut mm = m.clone();
        // play to trick 1 place 2 by rules
        for _ in 0..2 {
            let s = mm.to_act().unwrap();
            let v = mm.view(s);
            let l = mm.legal_actions();
            let a = bots::decide(&v, &Rules::HOUSE, &l).0;
            mm.apply(a).unwrap();
        }
        let s = mm.to_act().unwrap();
        let v = mm.view(s);
        let l = mm.legal_actions();
        let (_, cands) = app_candidates(&v, &l);
        let f = encode_view(&v, &leaf).unwrap();
        let cf: Vec<Vec<u16>> = cands.iter().map(|&c| encode_cand(&v, c)).collect();
        for bits in [4u32, 6, 8, 10, 12] {
            for seed in [4u64, 5, 9, 11] {
                let net = random_net(seed, &[16, 8], bits);
                let vals = net.evaluate(&f, &cf);
                let own: Vec<u64> = vals.iter().map(|x| x.get(engine::team_of(s))).collect();
                eprintln!("bits {bits} seed {seed}: {:?}", own);
            }
        }
    }
}
