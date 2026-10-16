//! Synthetic inputs for tests and fixtures only (never for a yardstick value): a leaf that
//! commutes with team rotation like the equity table, a seeded integer network, and seeded
//! feature vectors. The network and features are generated from SplitMix64 by a fixed recipe
//! that `python/nn` repeats, so the two integer forward passes can be checked against the same
//! golden outputs (`tests/data/intnet-golden.json`) without either side running the other.

use engine::rng::SplitMix64;
use equity::{MatchState, MoneyChance};
use solver::Leaf;

use crate::encode::{Features, MAX_ACTIVE, N_BINARY, N_DENSE, N_FEATURES, PAD};
use crate::mlp::{IntMlp, Layer};

/// A deterministic leaf that commutes with team rotation (weights from a hash of the state with
/// the dealer's team rotated to 0, then rotated back), as the equity table does.
pub struct SyntheticLeaf;

impl Leaf for SyntheticLeaf {
    fn value(&self, s: &MatchState) -> MoneyChance {
        if let Some(w) = s.winner() {
            return MoneyChance::certain(w);
        }
        let (c, d) = s.canonical();
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in c.scores.iter().chain(&c.games) {
            h = (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01B3);
        }
        let w: [u128; 4] = core::array::from_fn(|t| {
            u128::from((h >> (16 * t)) & 0xFFFF) + 1 + 2000 * u128::from(c.scores[t])
        });
        MoneyChance::from_weights(w).rotate(d)
    }
}

/// The golden network: widths `[N_FEATURES, 32, 16, 4]` (see [`golden_net_out`]).
pub fn golden_net(seed: u64) -> IntMlp {
    golden_net_out(seed, 4)
}

/// The golden network with `n_out` outputs: widths `[N_FEATURES, 32, 16, n_out]`, weights
/// uniform in `[−2^20, 2^20)`,
/// biases uniform in `[−2^39, 2^39)`, shifts `[27, 23, 16]`, drawn in order (layer by layer,
/// weights input-major then biases) from SplitMix64 seeded with `seed`.
pub fn golden_net_out(seed: u64, n_out: usize) -> IntMlp {
    let mut sm = SplitMix64::new(seed);
    let dims = [N_FEATURES, 32, 16, n_out];
    let shifts = [27, 23, 16];
    let layers = (0..3)
        .map(|k| {
            let (n_in, n_out) = (dims[k], dims[k + 1]);
            let w = (0..n_in * n_out)
                .map(|_| ((sm.next_u64() % (1 << 21)) as i64 - (1 << 20)) as i32)
                .collect();
            let b = (0..n_out)
                .map(|_| (sm.next_u64() % (1 << 40)) as i64 - (1 << 39))
                .collect();
            Layer {
                n_in,
                n_out,
                w,
                b,
                shift: shifts[k],
            }
        })
        .collect();
    IntMlp::new(layers).expect("the golden network is valid")
}

/// The golden inputs: `count` feature vectors; for each, every binary index in order is active
/// when the next draw is 0 mod 10 (up to [`MAX_ACTIVE`]), then each dense value is minus the
/// next draw mod (2^24 + 1).
pub fn golden_features(seed: u64, count: usize) -> Vec<Features> {
    let mut sm = SplitMix64::new(seed);
    (0..count)
        .map(|_| {
            let mut f = Features {
                active: [PAD; MAX_ACTIVE],
                n_active: 0,
                dense: [0; N_DENSE],
            };
            for i in 0..N_BINARY {
                if sm.next_u64().is_multiple_of(10) && usize::from(f.n_active) < MAX_ACTIVE {
                    f.active[usize::from(f.n_active)] = i as u16;
                    f.n_active += 1;
                }
            }
            for d in f.dense.iter_mut() {
                *d = -((sm.next_u64() % ((1 << 24) + 1)) as i32);
            }
            f
        })
        .collect()
}

/// Golden outcome tables for the outcome head: `count` tables of `v` outcomes; for each, the
/// mask is the next draw mod 2^v with the bit (next draw mod v) set, then for each outcome three
/// entries are the next draws mod 2^30 and the fourth makes the sum 2^32.
pub fn golden_outcomes(seed: u64, count: usize, v: usize) -> Vec<crate::outcome::Outcomes> {
    let mut sm = SplitMix64::new(seed);
    (0..count)
        .map(|_| {
            let mut mask = sm.next_u64() % (1 << v);
            mask |= 1 << (sm.next_u64() % v as u64);
            let vecs = (0..v)
                .map(|_| {
                    let a = sm.next_u64() % (1 << 30);
                    let b = sm.next_u64() % (1 << 30);
                    let c = sm.next_u64() % (1 << 30);
                    MoneyChance([a, b, c, equity::ONE - a - b - c])
                })
                .collect();
            crate::outcome::Outcomes { n: v, mask, vecs }
        })
        .collect()
}

/// Frontier positions as a hybrid solve reaches them: for each of `worlds` random roots at the
/// start of trick `trick − 1` ([`solver::worlds::sample_world`] with `seed`), every position the
/// optimised solver (default options) values at a frontier at the start of `trick`, in the order
/// it asks for them (repeats included), up to `max` in all. The frontier evaluator used while
/// collecting is a cheap hash of the canonical form, so the search prunes as with a real
/// network; the positions are real reached frontier positions. For benchmarks and tests.
pub fn reached_frontier(
    seed: u64,
    trick: u8,
    worlds: u64,
    max: usize,
    leaf: &(dyn Leaf + Sync),
) -> Vec<(solver::CanonPos, solver::MatchCtx)> {
    use std::sync::Mutex;
    let out = Mutex::new(Vec::new());
    for w in 0..worlds {
        if out.lock().expect("no panics").len() >= max {
            break;
        }
        let (pos, ctx, _) = solver::worlds::sample_world(seed, trick - 1, w, 0);
        let rec = solver::FnFrontier(|c: &solver::CanonPos, x: &solver::MatchCtx| {
            let mut v = out.lock().expect("no panics");
            if v.len() < max {
                v.push((*c, *x));
            }
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for b in c.slots.iter().chain(&c.scores).chain(&c.tricks) {
                h = (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01B3);
            }
            let w: [u128; 4] = core::array::from_fn(|t| u128::from((h >> (16 * t)) & 0xFFFF) + 1);
            MoneyChance::from_weights(w)
        });
        let cut = solver::FrontierCut::new(trick, &rec);
        let mut mem = solver::SolveMem::new();
        let _ = solver::solve_fast_cut(
            &pos,
            ctx,
            &DynLeaf(leaf),
            solver::FastOptions::default(),
            Some(cut),
            &mut mem,
            None,
        );
    }
    out.into_inner().expect("no panics")
}

struct DynLeaf<'a>(&'a (dyn Leaf + Sync));

impl Leaf for DynLeaf<'_> {
    fn value(&self, s: &MatchState) -> MoneyChance {
        self.0.value(s)
    }
}

/// A random integer network for benchmarks and differential tests: hidden widths `widths`,
/// `n_out` outputs, the scheme of a trained network quantised by `python/nn` (every layer's
/// weights `round(w × 2^24)`, shifts `[28, 24, …, 24, 20]`), with weights uniform in
/// `±(pct / 100) / sqrt(fan-in)` and biases in `±(pct / 100) / 8` (fan-in of the first layer:
/// 64, about the active inputs of a position). Integer only; drawn from SplitMix64 seeded with
/// `seed`.
pub fn random_net(seed: u64, widths: &[usize], n_out: usize, pct: u64) -> IntMlp {
    let mut sm = SplitMix64::new(seed);
    let mut dims = vec![N_FEATURES];
    dims.extend_from_slice(widths);
    dims.push(n_out);
    let last = dims.len() - 2;
    let layers = (0..dims.len() - 1)
        .map(|k| {
            let (n_in, n_out) = (dims[k], dims[k + 1]);
            let fan = if k == 0 { 64 } else { n_in as u64 };
            let a = ((1u64 << 48) / fan).isqrt() * pct / 100;
            let w = (0..n_in * n_out)
                .map(|_| ((sm.next_u64() % (2 * a + 1)) as i64 - a as i64) as i32)
                .collect();
            let s_in = if k == 0 { 24 } else { 20 };
            let ab = (1u64 << 21) * pct / 100;
            let b = (0..n_out)
                .map(|_| ((sm.next_u64() % (2 * ab + 1)) as i64 - ab as i64) << s_in)
                .collect();
            let shift = 24 + s_in - if k == last { 24 } else { 20 };
            Layer {
                n_in,
                n_out,
                w,
                b,
                shift,
            }
        })
        .collect();
    IntMlp::new(layers).expect("a random network within the bounds")
}
