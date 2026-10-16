//! Integer-only MLP inference (docs/M3-HYBRID-SPEC.md §6, "Inference").
//!
//! The scheme (format `41bots-int-mlp-v1`; the Python quantiser `python/nn` writes it and its
//! integer forward pass mirrors this one bit for bit):
//! - **Inputs**: a binary feature that is on is `2^15`; a dense value is `0..=2^15`
//!   ([`INPUT_SHIFT`]).
//! - **Weights**: `i32`, layer `l` holding `round(w × 2^s_l)`; **biases** `i64` holding
//!   `round(b × 2^(s_l + in_l))`, where `in_l` is the layer's input scale (15 for the first
//!   layer, [`ACT_SHIFT`] after). Stored input-major: `w[i × n_out + j]`.
//! - **Layer**: `acc_j = b_j + Σ_i w_ij · x_i` in `i64` (exact: the file is refused unless
//!   `Σ_i |w_ij| · max x_i + |b_j| ≤ 2^62` for every `j`, so no sum can overflow for any input),
//!   then `y_j = (acc_j + 2^(k−1)) >> k` with the layer's `shift` `k` (round half up; arithmetic
//!   shift), so `y` is in units of `2^-ACT_SHIFT` for a hidden layer and `2^-LOGIT_SHIFT` for
//!   the output layer.
//! - **Activation** (hidden layers): ReLU clipped at 64, `min(max(y, 0), 64 · 2^ACT_SHIFT)`
//!   (training uses the same clip, so the float network and the integer one compute the same
//!   function up to rounding).
//! - **Output**: four logits in units of `2^-LOGIT_SHIFT`, turned into a money chance by
//!   [`crate::money::money_from_logits`].
//!
//! Integer sums are exact and do not depend on the order of the terms, so the result is
//! byte-identical on every machine and target.

use crate::encode::{Features, INPUT_SHIFT, N_BINARY, N_FEATURES};

/// Activations are in units of 2^-20.
pub const ACT_SHIFT: u32 = 20;
/// Logits are in units of 2^-24.
pub const LOGIT_SHIFT: u32 = 24;
/// The ReLU clip: 64.
pub const CLIP: i64 = 64 << ACT_SHIFT;
/// The widest layer supported.
pub const MAX_WIDTH: usize = 1024;
/// The bound every layer's worst-case sum must stay within.
pub const SUM_BOUND: i128 = 1 << 62;

/// One layer: `w` input-major (`n_in × n_out`), `b` per output, and the right shift after the
/// sum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layer {
    /// Inputs.
    pub n_in: usize,
    /// Outputs.
    pub n_out: usize,
    /// Weights, `w[i * n_out + j]`.
    pub w: Vec<i32>,
    /// Biases.
    pub b: Vec<i64>,
    /// Right shift (with rounding) after the sum.
    pub shift: u32,
}

/// An integer MLP: the encoder's features in, four logits out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntMlp {
    layers: Vec<Layer>,
}

/// `(x + 2^(k−1)) >> k`: division by 2^k rounding halves up (`k = 0`: unchanged).
#[inline]
pub fn round_shift(x: i64, k: u32) -> i64 {
    if k == 0 { x } else { (x + (1 << (k - 1))) >> k }
}

impl IntMlp {
    /// A checked network: the first layer takes [`N_FEATURES`] inputs, the last gives 4, the
    /// layers chain, no layer is wider than [`MAX_WIDTH`], shifts are below 63, and no sum can
    /// leave `±2^62` for any input (so `i64` arithmetic is exact).
    pub fn new(layers: Vec<Layer>) -> Result<IntMlp, String> {
        if layers.len() < 2 {
            return Err("at least one hidden layer".into());
        }
        if layers[0].n_in != N_FEATURES {
            return Err(format!(
                "the first layer takes {} inputs; the encoder gives {N_FEATURES}",
                layers[0].n_in
            ));
        }
        if layers[layers.len() - 1].n_out != 4 {
            return Err("the last layer must give 4 logits".into());
        }
        for (k, l) in layers.iter().enumerate() {
            if l.w.len() != l.n_in * l.n_out || l.b.len() != l.n_out {
                return Err(format!("layer {k}: weight or bias size"));
            }
            if l.n_out > MAX_WIDTH {
                return Err(format!("layer {k}: wider than {MAX_WIDTH}"));
            }
            if k > 0 && l.n_in != layers[k - 1].n_out {
                return Err(format!("layer {k}: takes {} inputs", l.n_in));
            }
            if l.shift >= 63 {
                return Err(format!("layer {k}: shift {}", l.shift));
            }
            let max_in: i128 = if k == 0 {
                1 << INPUT_SHIFT
            } else {
                CLIP.into()
            };
            for j in 0..l.n_out {
                let mut s: i128 = i128::from(l.b[j]).abs();
                for i in 0..l.n_in {
                    s += i128::from(l.w[i * l.n_out + j]).abs() * max_in;
                }
                if s > SUM_BOUND {
                    return Err(format!(
                        "layer {k} output {j}: worst-case sum {s} exceeds 2^62"
                    ));
                }
            }
        }
        Ok(IntMlp { layers })
    }

    /// The layers.
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Hidden widths and the output, for reports (e.g. `[256, 64, 4]`).
    pub fn widths(&self) -> Vec<usize> {
        self.layers.iter().map(|l| l.n_out).collect()
    }

    /// The four logits (units of 2^-[`LOGIT_SHIFT`]) of an encoded position.
    pub fn forward(&self, f: &Features) -> [i64; 4] {
        let l0 = &self.layers[0];
        let n = l0.n_out;
        // Binary features: their rows summed once, then scaled by the input's 2^15.
        let mut bin = [0i64; MAX_WIDTH];
        for &i in f.active() {
            let row = &l0.w[usize::from(i) * n..(usize::from(i) + 1) * n];
            for (a, &w) in bin[..n].iter_mut().zip(row) {
                *a += i64::from(w);
            }
        }
        let mut acc = [0i64; MAX_WIDTH];
        for j in 0..n {
            acc[j] = l0.b[j] + (bin[j] << INPUT_SHIFT);
        }
        for (d, &x) in f.dense.iter().enumerate() {
            if x == 0 {
                continue;
            }
            let i = N_BINARY + d;
            let row = &l0.w[i * n..(i + 1) * n];
            for (a, &w) in acc[..n].iter_mut().zip(row) {
                *a += i64::from(w) * i64::from(x);
            }
        }
        let mut h = [0i64; MAX_WIDTH];
        for j in 0..n {
            h[j] = round_shift(acc[j], l0.shift).clamp(0, CLIP);
        }
        let mut width = n;
        let last = self.layers.len() - 1;
        for (k, l) in self.layers.iter().enumerate().skip(1) {
            let m = l.n_out;
            acc[..m].copy_from_slice(&l.b);
            for (i, &x) in h[..width].iter().enumerate() {
                if x == 0 {
                    continue;
                }
                let row = &l.w[i * m..(i + 1) * m];
                for (a, &w) in acc[..m].iter_mut().zip(row) {
                    *a += i64::from(w) * x;
                }
            }
            if k == last {
                return core::array::from_fn(|j| round_shift(acc[j], l.shift));
            }
            for j in 0..m {
                h[j] = round_shift(acc[j], l.shift).clamp(0, CLIP);
            }
            width = m;
        }
        unreachable!("the last layer returns")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{MAX_ACTIVE, N_DENSE, PAD};

    fn tiny(w0: i32, shift: u32) -> Vec<Layer> {
        vec![
            Layer {
                n_in: N_FEATURES,
                n_out: 2,
                w: vec![w0; N_FEATURES * 2],
                b: vec![0, -5],
                shift,
            },
            Layer {
                n_in: 2,
                n_out: 4,
                w: vec![1, 2, 3, 4, -1, -2, -3, -4],
                b: vec![0; 4],
                shift: 0,
            },
        ]
    }

    #[test]
    fn rounding_shift_rounds_halves_up() {
        assert_eq!(round_shift(5, 1), 3);
        assert_eq!(round_shift(-5, 1), -2);
        assert_eq!(round_shift(-6, 1), -3);
        assert_eq!(round_shift(7, 0), 7);
        assert_eq!(round_shift(1 << 20, 20), 1);
        assert_eq!(round_shift((1 << 19) - 1, 20), 0);
    }

    #[test]
    fn forward_by_hand() {
        let net = IntMlp::new(tiny(3, 15)).expect("valid");
        let mut f = Features {
            active: [PAD; MAX_ACTIVE],
            n_active: 2,
            dense: [0; N_DENSE],
        };
        f.active[0] = 0;
        f.active[1] = 7;
        f.dense[0] = 1 << 14;
        // Layer 0: 3·2^15·2 + 3·2^14 = 3·2^14·5; >> 15 rounds 7.5 to 8 (and −5 + … to 7).
        let h0 = round_shift(3 * (1 << 15) * 2 + 3 * (1 << 14), 15);
        let h1 = round_shift(-5 + 3 * (1 << 15) * 2 + 3 * (1 << 14), 15);
        assert_eq!((h0, h1), (8, 7));
        assert_eq!(
            net.forward(&f),
            [h0 - h1, 2 * h0 - 2 * h1, 3 * h0 - 3 * h1, 4 * h0 - 4 * h1]
        );
    }

    /// A first layer of `width` outputs, then 4 logits with every weight `w1`.
    fn wide(width: usize, w1: i32) -> Vec<Layer> {
        vec![
            Layer {
                n_in: N_FEATURES,
                n_out: width,
                w: vec![1; N_FEATURES * width],
                b: vec![0; width],
                shift: 15,
            },
            Layer {
                n_in: width,
                n_out: 4,
                w: vec![w1; width * 4],
                b: vec![0; 4],
                shift: 0,
            },
        ]
    }

    #[test]
    fn overflowing_networks_are_refused() {
        // 1024 inputs × 2^31 × 64·2^20 = 2^67 > 2^62.
        assert!(IntMlp::new(wide(1024, i32::MAX)).is_err());
        // 1024 × 2^20 × 2^26 = 2^56.
        assert!(IntMlp::new(wide(1024, 1 << 20)).is_ok());
        assert!(IntMlp::new(wide(MAX_WIDTH + 1, 1)).is_err());
        let mut bad = tiny(1, 15);
        bad[1].n_out = 3;
        assert!(IntMlp::new(bad).is_err());
        let mut bad = tiny(1, 15);
        bad[0].shift = 63;
        assert!(IntMlp::new(bad).is_err());
    }
}
