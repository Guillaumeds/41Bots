//! A quantised oracle network as a solver frontier evaluator: canonical position → encoder →
//! integer MLP → integer softmax → money chance (rotated back to the real teams).
//!
//! Weights file (`41bots-int-mlp-v1`, safetensors): tensors `w0, b0, w1, b1, …` (`I32`
//! `[n_in, n_out]` input-major, `I64` `[n_out]`), and metadata `format`, `encoder` (must be
//! [`ENCODER`]), `n_features`, `shifts` (a JSON list, one right shift per layer), `trick` (the
//! frontier trick the network was trained for), plus whatever training recorded (data hashes,
//! sizes). The file's SHA-256 names the network in hybrid values (spec §6: the values version
//! names the net).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use equity::MoneyChance;
use equity::sha256::sha256_hex;
use solver::frontier::{CanonPos, Frontier};
use solver::{Leaf, MatchCtx};

use crate::encode::{ENCODER, N_FEATURES, encode, unrotate};
use crate::mlp::{IntMlp, Layer};
use crate::money::money_from_logits;
use crate::safetensors;

/// The weights format this crate reads.
pub const FORMAT: &str = "41bots-int-mlp-v1";

/// A loaded oracle network.
pub struct Oracle {
    net: IntMlp,
    leaf: Arc<dyn Leaf + Send + Sync>,
    /// The frontier trick the network was trained for.
    pub trick: u8,
    /// SHA-256 of the weights file (hex).
    pub sha256: String,
    /// The file's metadata.
    pub meta: BTreeMap<String, String>,
}

/// Parse the integer network of a weights file.
pub fn parse_net(f: &safetensors::File) -> Result<IntMlp, String> {
    if f.meta("format")? != FORMAT {
        return Err(format!("format {} is not {FORMAT}", f.meta("format")?));
    }
    if f.meta("encoder")? != ENCODER {
        return Err(format!(
            "network made for encoder {}; this build encodes {ENCODER}",
            f.meta("encoder")?
        ));
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
    IntMlp::new(layers)
}

/// A weights file for an integer network (the format [`parse_net`] reads), with the frontier
/// trick and any extra metadata.
pub fn write_net(net: &IntMlp, trick: u8, extra: &BTreeMap<String, String>) -> Vec<u8> {
    let mut f = safetensors::File::default();
    for (k, l) in net.layers().iter().enumerate() {
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
    f.meta.insert("n_features".into(), N_FEATURES.to_string());
    f.meta.insert("trick".into(), trick.to_string());
    let shifts: Vec<u32> = net.layers().iter().map(|l| l.shift).collect();
    f.meta.insert(
        "shifts".into(),
        serde_json::to_string(&shifts).expect("a list of numbers"),
    );
    safetensors::write(&f)
}

impl Oracle {
    /// Load a weights file; `leaf` (the equity table) gives the encoder's dense features.
    pub fn load(path: &Path, leaf: Arc<dyn Leaf + Send + Sync>) -> Result<Oracle, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Oracle::from_bytes(&bytes, leaf).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// [`Oracle::load`] from the file's bytes.
    pub fn from_bytes(bytes: &[u8], leaf: Arc<dyn Leaf + Send + Sync>) -> Result<Oracle, String> {
        let f = safetensors::read(bytes)?;
        let net = parse_net(&f)?;
        let trick: u8 = f
            .meta("trick")?
            .parse()
            .map_err(|_| "trick: not a number".to_string())?;
        Ok(Oracle {
            net,
            leaf,
            trick,
            sha256: sha256_hex(bytes),
            meta: f.meta,
        })
    }

    /// The integer network.
    pub fn net(&self) -> &IntMlp {
        &self.net
    }

    /// The network's money chance of a canonical position (real team order).
    pub fn eval(&self, c: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        let (f, r) = encode(c, ctx, &*self.leaf);
        unrotate(money_from_logits(&self.net.forward(&f)), r)
    }
}

impl Frontier for Oracle {
    fn value(&self, pos: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        self.eval(pos, ctx)
    }
}
