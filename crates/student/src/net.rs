//! The student's integer network (docs/STUDENT.md §1, §5): the oracle's integer scheme
//! (`41bots-int-mlp-v1`, `nn::mlp`: `i32` weights, `i64` sums, ReLU clipped at 64, round-half-up
//! shifts, four logits in units of 2^-24, the integer softmax of `nn::money`) over the student's
//! inputs (`crate::encode`: the view's and the candidate's binary features, then the dense block).
//!
//! **One scorer for every candidate.** The first layer is linear, so its sum for a candidate is the
//! view's part (bias, the view's active rows, the dense rows) plus the candidate's active rows:
//! [`StudentNet::evaluate`] computes the view's part once and adds each candidate's rows to it.
//! Integer sums do not depend on the order of their terms, and every partial sum is a sub-sum of
//! the terms the per-network bound covers (`nn::mlp`, "Widths and the overflow bound"), so this is
//! exactly the plain forward pass of each (view, candidate) input ([`StudentNet::logits_plain`],
//! tested), on every machine and in WASM.
//!
//! Weights file: safetensors with tensors `w0, b0, w1, b1, …` (as the oracle's) and metadata
//! `format` (`41bots-int-mlp-v1`), `encoder` (`student-v1`), `encoder_sha256` (must be this build's
//! [`crate::encode::encoder_sha256`]), `n_features`, `head` (`student-money4`), `shifts`, plus what
//! training recorded. The file's SHA-256 names the network.

use std::collections::BTreeMap;

use equity::MoneyChance;
use equity::sha256::sha256_hex;
use nn::encode::INPUT_SHIFT;
use nn::mlp::{CLIP, Layer, MAX_WIDTH, SUM_BOUND, round_shift, width_error};
use nn::money::money_from_logits;
use nn::safetensors;

use crate::encode::{ENCODER, N_BINARY, N_FEATURES, ViewFeatures, encoder_sha256, unrotate};

/// The weights format.
pub const FORMAT: &str = "41bots-int-mlp-v1";
/// The head's name.
pub const HEAD: &str = "student-money4";

/// A loaded student network.
pub struct StudentNet {
    layers: Vec<Layer>,
    /// SHA-256 of the weights file (hex).
    pub sha256: String,
    /// The file's metadata.
    pub meta: BTreeMap<String, String>,
}

/// Check integer layers: the first takes [`N_FEATURES`] inputs, the last gives 4 logits, the layers
/// chain, none is wider than `nn::MAX_WIDTH`, shifts are below 63, and no sum can leave `±2^62`
/// for any input (so `i64` arithmetic is exact).
pub fn check_layers(layers: &[Layer]) -> Result<(), String> {
    if layers.len() < 2 {
        return Err("at least one hidden layer".into());
    }
    if layers[0].n_in != N_FEATURES {
        return Err(format!(
            "the first layer takes {} inputs; the student encoder gives {N_FEATURES}",
            layers[0].n_in
        ));
    }
    if layers[layers.len() - 1].n_out != 4 {
        return Err("the last layer gives 4 money logits".into());
    }
    for (k, l) in layers.iter().enumerate() {
        if l.w.len() != l.n_in * l.n_out || l.b.len() != l.n_out {
            return Err(format!("layer {k}: weight or bias size"));
        }
        if l.n_out > MAX_WIDTH {
            return Err(width_error(k, l.n_out));
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
    Ok(())
}

/// A weights file for integer layers (the format [`StudentNet::from_bytes`] reads), with any
/// extra metadata.
pub fn write_net(layers: &[Layer], extra: &BTreeMap<String, String>) -> Vec<u8> {
    let mut f = safetensors::File::default();
    for (k, l) in layers.iter().enumerate() {
        f.tensors.insert(
            format!("w{k}"),
            safetensors::Tensor::i32(vec![l.n_in, l.n_out], &l.w),
        );
        f.tensors.insert(
            format!("b{k}"),
            safetensors::Tensor::i64(vec![l.n_out], &l.b),
        );
    }
    f.meta = extra.clone();
    f.meta.insert("format".into(), FORMAT.into());
    f.meta.insert("encoder".into(), ENCODER.into());
    f.meta
        .insert("encoder_sha256".into(), encoder_sha256().to_string());
    f.meta.insert("n_features".into(), N_FEATURES.to_string());
    f.meta.insert("head".into(), HEAD.into());
    let shifts: Vec<u32> = layers.iter().map(|l| l.shift).collect();
    f.meta.insert(
        "shifts".into(),
        serde_json::to_string(&shifts).expect("a list of numbers"),
    );
    safetensors::write(&f)
}

impl StudentNet {
    /// A network from checked layers (tests and tools; its `sha256` is that of [`write_net`]'s
    /// file with no extra metadata).
    pub fn new(layers: Vec<Layer>) -> Result<StudentNet, String> {
        check_layers(&layers)?;
        let sha256 = sha256_hex(&write_net(&layers, &BTreeMap::new()));
        Ok(StudentNet {
            layers,
            sha256,
            meta: BTreeMap::new(),
        })
    }

    /// Load a weights file: refused unless it is a student network for this build's encoder (its
    /// name and hash) with checked layers.
    pub fn from_bytes(bytes: &[u8]) -> Result<StudentNet, String> {
        let f = safetensors::read(bytes)?;
        if f.meta("format")? != FORMAT {
            return Err(format!("format {} is not {FORMAT}", f.meta("format")?));
        }
        if f.meta("encoder")? != ENCODER {
            return Err(format!(
                "network made for encoder {}; this build encodes {ENCODER}",
                f.meta("encoder")?
            ));
        }
        if f.meta("encoder_sha256")? != encoder_sha256() {
            return Err(format!(
                "network made for encoder hash {}; this build's is {}",
                f.meta("encoder_sha256")?,
                encoder_sha256()
            ));
        }
        if f.meta("head")? != HEAD {
            return Err(format!("head {} is not {HEAD}", f.meta("head")?));
        }
        if f.meta("n_features")? != N_FEATURES.to_string() {
            return Err(format!("network takes {} features", f.meta("n_features")?));
        }
        let shifts: Vec<u32> =
            serde_json::from_str(f.meta("shifts")?).map_err(|e| format!("shifts: {e}"))?;
        let mut layers = Vec::new();
        for (k, &shift) in shifts.iter().enumerate() {
            let w = f.get(&format!("w{k}"))?;
            let b = f.get(&format!("b{k}"))?;
            if w.shape.len() != 2 || b.shape.len() != 1 || b.shape[0] != w.shape[1] {
                return Err(format!("layer {k}: shapes {:?} {:?}", w.shape, b.shape));
            }
            layers.push(Layer {
                n_in: w.shape[0],
                n_out: w.shape[1],
                w: w.as_i32()?,
                b: b.as_i64()?,
                shift,
            });
        }
        if f.tensors.len() != 2 * layers.len() {
            return Err("tensors other than the layers".into());
        }
        check_layers(&layers)?;
        Ok(StudentNet {
            layers,
            sha256: sha256_hex(bytes),
            meta: f.meta,
        })
    }

    /// The layers.
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Hidden widths (e.g. `[256, 128]`).
    pub fn hidden(&self) -> Vec<usize> {
        self.layers[..self.layers.len() - 1]
            .iter()
            .map(|l| l.n_out)
            .collect()
    }

    /// The first layer's sum for the view alone: the bias, the view's active rows, the dense rows.
    pub fn view_sum(&self, v: &ViewFeatures) -> Vec<i64> {
        let l0 = &self.layers[0];
        let n = l0.n_out;
        let mut bin = vec![0i64; n];
        for &i in &v.active {
            let row = &l0.w[usize::from(i) * n..(usize::from(i) + 1) * n];
            for (a, &w) in bin.iter_mut().zip(row) {
                *a += i64::from(w);
            }
        }
        let mut acc: Vec<i64> = (0..n).map(|j| l0.b[j] + (bin[j] << INPUT_SHIFT)).collect();
        for (d, &x) in v.dense.iter().enumerate() {
            if x == 0 {
                continue;
            }
            let i = N_BINARY + d;
            let row = &l0.w[i * n..(i + 1) * n];
            for (a, &w) in acc.iter_mut().zip(row) {
                *a += i64::from(w) * i64::from(x);
            }
        }
        acc
    }

    /// The logits of one candidate from the view's first-layer sum.
    pub fn cand_logits(&self, view_sum: &[i64], cand: &[u16]) -> [i64; 4] {
        let l0 = &self.layers[0];
        let n = l0.n_out;
        let mut bin = vec![0i64; n];
        for &i in cand {
            let row = &l0.w[usize::from(i) * n..(usize::from(i) + 1) * n];
            for (a, &w) in bin.iter_mut().zip(row) {
                *a += i64::from(w);
            }
        }
        let h: Vec<i64> = (0..n)
            .map(|j| round_shift(view_sum[j] + (bin[j] << INPUT_SHIFT), l0.shift).clamp(0, CLIP))
            .collect();
        self.rest(h)
    }

    /// The layers after the first, from the first layer's activations.
    fn rest(&self, mut h: Vec<i64>) -> [i64; 4] {
        let last = self.layers.len() - 1;
        for (k, l) in self.layers.iter().enumerate().skip(1) {
            let m = l.n_out;
            let mut acc = l.b.clone();
            for (i, &x) in h.iter().enumerate() {
                if x == 0 {
                    continue;
                }
                let row = &l.w[i * m..(i + 1) * m];
                for (a, &w) in acc.iter_mut().zip(row) {
                    *a += i64::from(w) * x;
                }
            }
            if k == last {
                return core::array::from_fn(|j| round_shift(acc[j], l.shift));
            }
            h = acc
                .iter()
                .map(|&a| round_shift(a, l.shift).clamp(0, CLIP))
                .collect();
        }
        unreachable!("the last layer returns")
    }

    /// The plain forward pass of one (view, candidate) input: every input in one sum (the
    /// reference [`StudentNet::evaluate`] is tested against).
    pub fn logits_plain(&self, v: &ViewFeatures, cand: &[u16]) -> [i64; 4] {
        let l0 = &self.layers[0];
        let n = l0.n_out;
        let mut x = vec![0i64; N_FEATURES];
        for &i in v.active.iter().chain(cand) {
            x[usize::from(i)] = 1 << INPUT_SHIFT;
        }
        for (d, &y) in v.dense.iter().enumerate() {
            x[N_BINARY + d] = i64::from(y);
        }
        let h: Vec<i64> = (0..n)
            .map(|j| {
                let mut a = l0.b[j];
                for (i, &xi) in x.iter().enumerate() {
                    a += i64::from(l0.w[i * n + j]) * xi;
                }
                round_shift(a, l0.shift).clamp(0, CLIP)
            })
            .collect();
        self.rest(h)
    }

    /// Every candidate's logits (rotated pair order: the deciding pair first).
    pub fn logits(&self, v: &ViewFeatures, cands: &[Vec<u16>]) -> Vec<[i64; 4]> {
        let s = self.view_sum(v);
        cands.iter().map(|c| self.cand_logits(&s, c)).collect()
    }

    /// Every candidate's money chance, in real pair order.
    pub fn evaluate(&self, v: &ViewFeatures, cands: &[Vec<u16>]) -> Vec<MoneyChance> {
        self.logits(v, cands)
            .iter()
            .map(|z| unrotate(money_from_logits(z), v.team))
            .collect()
    }
}

/// The bot's choice: the candidate with the highest value for `team`; ties to the first.
pub fn choose(values: &[MoneyChance], team: u8) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if v.get(team) > values[best].get(team) {
            best = i;
        }
    }
    best
}

/// A seeded random student network with the given hidden widths (tests and tools only; never a
/// bot): weights uniform in `±2^(bits−1)` for `pct` percent of them (the rest 0), biases small.
pub fn random_net(seed: u64, hidden: &[usize], bits: u32) -> StudentNet {
    let mut sm = engine::rng::SplitMix64::new(seed);
    let mut dims = vec![N_FEATURES];
    dims.extend_from_slice(hidden);
    dims.push(4);
    let half = 1i64 << (bits - 1);
    let layers: Vec<Layer> = (0..dims.len() - 1)
        .map(|k| {
            let (n_in, n_out) = (dims[k], dims[k + 1]);
            let w = (0..n_in * n_out)
                .map(|_| ((sm.next_u64() % (2 * half as u64)) as i64 - half) as i32)
                .collect();
            let b = (0..n_out)
                .map(|_| (sm.next_u64() % (1 << 30)) as i64 - (1 << 29))
                .collect();
            // Layer 0 sums 2^24-scaled inputs: shift back to activations of about 2^20.
            let shift = if k == 0 { 24 + bits - 18 } else { 14 + bits };
            Layer {
                n_in,
                n_out,
                w,
                b,
                shift,
            }
        })
        .collect();
    StudentNet::new(layers).expect("a random network within the bounds")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{encode_cand, encode_view};
    use engine::game::{Match, Phase, Rules};
    use nn::synthetic::SyntheticLeaf;

    /// The incremental evaluation is exactly the plain forward pass, on real encoded positions;
    /// a written network loads back to the same network; a file for another encoder is refused.
    #[test]
    fn incremental_equals_plain_and_files_round_trip() {
        let net = random_net(3, &[24, 8], 12);
        let bytes = write_net(net.layers(), &BTreeMap::new());
        let back = StudentNet::from_bytes(&bytes).expect("loads");
        assert_eq!(back.layers(), net.layers());
        assert_eq!(back.sha256, net.sha256);
        let leaf = SyntheticLeaf;
        let mut checked = 0;
        for s in 0..6u64 {
            let mut m = Match::new(900 + s, Rules::HOUSE);
            while let Some(seat) = m.to_act() {
                let v = m.view(seat);
                let legal = m.legal_actions();
                if v.public.trump.is_some()
                    && v.public.trick_no >= 1
                    && matches!(v.public.phase, Phase::AwaitPlay(_))
                {
                    let f = encode_view(&v, &leaf).expect("encodes");
                    let cands: Vec<Vec<u16>> =
                        legal.iter().map(|&a| encode_cand(&v, (a, None))).collect();
                    let z = net.logits(&f, &cands);
                    for (c, zc) in cands.iter().zip(&z) {
                        assert_eq!(*zc, net.logits_plain(&f, c));
                        checked += 1;
                    }
                    let vals = net.evaluate(&f, &cands);
                    assert!(choose(&vals, f.team) < vals.len());
                }
                let a = bots::decide(&v, &Rules::HOUSE, &legal).0;
                m.apply(a).expect("legal");
            }
        }
        assert!(checked > 200);
        // Another encoder's file is refused.
        let mut f = safetensors::read(&bytes).expect("reads");
        f.meta.insert("encoder_sha256".into(), "0".repeat(64));
        assert!(StudentNet::from_bytes(&safetensors::write(&f)).is_err());
        f.meta
            .insert("encoder_sha256".into(), encoder_sha256().to_string());
        f.meta.insert("encoder".into(), "oracle-v2".into());
        assert!(StudentNet::from_bytes(&safetensors::write(&f)).is_err());
    }

    #[test]
    fn overflowing_networks_are_refused() {
        let mut l = random_net(1, &[4], 12).layers().to_vec();
        assert!(check_layers(&l).is_ok());
        // Every input of output 0 at the largest weight: 3,292 × 2^31 × 2^24 > 2^62.
        let (n_in, n_out) = (l[0].n_in, l[0].n_out);
        for i in 0..n_in {
            l[0].w[i * n_out] = i32::MAX;
        }
        assert!(check_layers(&l).is_err());
        let mut l = random_net(1, &[4], 12).layers().to_vec();
        l[1].n_out = 3;
        assert!(check_layers(&l).is_err());
    }

    #[test]
    fn ties_go_to_the_first_candidate() {
        let a = MoneyChance([1 << 30; 4]);
        let b = MoneyChance([(1 << 30) + 1, (1 << 30) - 1, 1 << 30, 1 << 30]);
        assert_eq!(choose(&[a, a], 0), 0);
        assert_eq!(choose(&[a, b], 0), 1);
        assert_eq!(choose(&[b, a], 1), 1);
    }
}
