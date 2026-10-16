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

/// The golden network: widths `[N_FEATURES, 32, 16, 4]`, weights uniform in `[−2^20, 2^20)`,
/// biases uniform in `[−2^39, 2^39)`, shifts `[18, 23, 16]`, drawn in order (layer by layer,
/// weights input-major then biases) from SplitMix64 seeded with `seed`.
pub fn golden_net(seed: u64) -> IntMlp {
    let mut sm = SplitMix64::new(seed);
    let dims = [N_FEATURES, 32, 16, 4];
    let shifts = [18, 23, 16];
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
/// when the next draw is 0 mod 10 (up to [`MAX_ACTIVE`]), then each dense value is the next
/// draw mod 32769.
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
                *d = (sm.next_u64() % 32769) as u16;
            }
            f
        })
        .collect()
}
