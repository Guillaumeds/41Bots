//! The leaf-risk predictor (docs/m3-hybrid/rung2.md "Learned exact leaves"): a small model that
//! predicts how wrong the oracle network O_k is at a frontier position, so that the hybrid H_k
//! solves exactly the positions where an exact value is worth most (learning to defer, Geifman
//! & El-Yaniv 2017, Mozannar & Sontag 2020; value of computation, Russell & Wefald 1991).
//!
//! **What it predicts.** The expected squared error of the network's money chance at the
//! position, summed over the four teams: `E[Σ_t (net_t − exact_t)^2 | position]`, in units of
//! 2^-40 of a squared money chance (whole numbers; a squared error of 1 is 2^40). The sum over
//! the teams is the same in any team order, so it is the same for the leader-relative rows of a
//! label shard and for the decider-relative view of the search.
//!
//! **The model** is an ensemble of regression trees fitted by gradient boosting (Friedman 2001,
//! squared loss, shrinkage, row subsampling: Friedman 2002), trained in Python
//! (`python/nn/fortyone_nn/leafrisk.py`) and stored as JSON ([`FORMAT`]): every split is "input
//! `f` ≤ whole number `t`" (left) or not (right), every leaf a whole number, and the prediction
//! the base plus the leaves reached, summed in `i128` and clamped to `0..=i64::MAX` (a squared
//! error is never negative). So a prediction is the same on every machine, thread count and
//! WASM, bit for bit with the Python trainer's integer path (tested on a shared fixture).
//!
//! **Its inputs** ([`RiskInputs`], in the encoder's rotated team order, the leader's team first):
//! - `0..N_BINARY`: the encoder's binary features (`crate::encode`), 0 or 1;
//! - `N_BINARY..N_FEATURES`: its dense features (whole numbers, units of 2^-24);
//! - `N_FEATURES..N_INPUTS`: [`EXTRA_NAMES`], cheap whole numbers from the network's own value
//!   and the position's possible outcomes (`crate::outcome`): the race class, the network's four
//!   shares, its top share and top-two margin, its Gini impurity (an entropy measure without
//!   logarithms), the L1 distance from the network's vector to the nearest and to the farthest
//!   possible outcome's exact vector (the exact value is one of those vectors, so the farthest
//!   is a hard bound on the network's L1 error), the number of possible outcomes and of teams
//!   that can win the game, and the widest range of one team's money over the possible outcomes.

use std::path::Path;

use equity::MoneyChance;
use serde_json::Value;
use solver::frontier::CanonPos;
use solver::{Leaf, MatchCtx};

use crate::encode::{Features, N_BINARY, N_DENSE, N_FEATURES, OFF_GAMES, encode};
use crate::outcome::{GAME_WON, Outcomes, outcomes, race_class};

/// The model file's format.
pub const FORMAT: &str = "41bots-leafrisk-v1";

/// The extra inputs after the encoder's features.
pub const N_EXTRAS: usize = 13;

/// Every input: the encoder's features, then the extras.
pub const N_INPUTS: usize = N_FEATURES + N_EXTRAS;

/// The extras' names, in order (shared with Python's `leafrisk.EXTRA_NAMES`).
pub const EXTRA_NAMES: [&str; N_EXTRAS] = [
    "race class",
    "network: leader's team",
    "network: next team",
    "network: opposite team",
    "network: previous team",
    "network: top share",
    "network: top-two margin",
    "network: Gini impurity",
    "L1 to the nearest possible outcome",
    "L1 to the farthest possible outcome",
    "possible outcomes",
    "teams that can win the game",
    "widest money range of one team",
];

/// The unit of a prediction: 2^-40 of a squared money chance.
pub const SCORE_SHIFT: u32 = 40;

/// A position's inputs (module notes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RiskInputs {
    /// The binary features as a bit set.
    bits: [u64; N_BINARY.div_ceil(64)],
    /// The dense features.
    dense: [i32; N_DENSE],
    /// The extras ([`EXTRA_NAMES`]).
    pub extras: [i64; N_EXTRAS],
}

impl RiskInputs {
    /// The inputs of an encoded position with its possible outcomes and the network's value, all
    /// in the encoder's rotated team order (as a label shard's row holds them).
    pub fn new(f: &Features, o: &Outcomes, net_rotated: &MoneyChance) -> RiskInputs {
        let mut bits = [0u64; N_BINARY.div_ceil(64)];
        for &a in f.active() {
            bits[usize::from(a) / 64] |= 1 << (a % 64);
        }
        let on = |i: usize| bits[i / 64] >> (i % 64) & 1 == 1;
        let has_game = core::array::from_fn(|t| on(OFF_GAMES + t));
        RiskInputs {
            bits,
            dense: f.dense,
            extras: extras(o, net_rotated, has_game),
        }
    }

    /// The inputs of a frontier position (`net`: the network's value in real team order).
    pub fn of(c: &CanonPos, ctx: &MatchCtx, leaf: &dyn Leaf, net: &MoneyChance) -> RiskInputs {
        let (f, r) = encode(c, ctx, leaf);
        let o = outcomes(c, ctx, leaf);
        RiskInputs::new(&f, &o, &net.rotate(r % 4))
    }

    /// Input `i` as a whole number.
    ///
    /// # Panics
    /// If `i >= N_INPUTS`.
    pub fn get(&self, i: usize) -> i64 {
        if i < N_BINARY {
            i64::from(self.bits[i / 64] >> (i % 64) & 1 == 1)
        } else if i < N_FEATURES {
            i64::from(self.dense[i - N_BINARY])
        } else {
            self.extras[i - N_FEATURES]
        }
    }
}

/// The extras ([`EXTRA_NAMES`]) of a position: its possible outcomes `o`, the network's value `m`
/// and the teams with a game won, all in one team order (the encoder's rotated one).
pub fn extras(o: &Outcomes, m: &MoneyChance, has_game: [bool; 4]) -> [i64; N_EXTRAS] {
    let v = m.0;
    let mut sorted = v;
    sorted.sort_unstable();
    // Σ_{t<u} m_t m_u = (1 − Σ m_t^2) / 2: below 2^63 since the shares sum to 2^32.
    let mut pairs: u64 = 0;
    for t in 0..4 {
        for u in t + 1..4 {
            pairs += v[t] * v[u];
        }
    }
    let l1 = |w: &MoneyChance| -> i64 { (0..4).map(|t| (w.0[t] as i64 - v[t] as i64).abs()).sum() };
    let possible: Vec<&MoneyChance> = (0..o.n)
        .filter(|&i| o.mask >> i & 1 == 1)
        .map(|i| &o.vecs[i])
        .collect();
    let near = possible.iter().map(|w| l1(w)).min().unwrap_or(0);
    let far = possible.iter().map(|w| l1(w)).max().unwrap_or(0);
    let range = (0..4)
        .map(|t| {
            let hi = possible.iter().map(|w| w.0[t]).max().unwrap_or(0);
            let lo = possible.iter().map(|w| w.0[t]).min().unwrap_or(0);
            (hi - lo) as i64
        })
        .max()
        .unwrap_or(0);
    let game_mask = o.mask & ((1 << GAME_WON) - 1);
    [
        i64::from(race_class(o.mask, has_game)),
        v[0] as i64,
        v[1] as i64,
        v[2] as i64,
        v[3] as i64,
        sorted[3] as i64,
        (sorted[3] - sorted[2]) as i64,
        (pairs >> 32) as i64,
        near,
        far,
        i64::from(o.mask.count_ones()),
        i64::from(game_mask.count_ones()),
        range,
    ]
}

/// One regression tree: nodes in an order where every child comes after its parent; node `i` is
/// a leaf when `feature[i] < 0`, else it sends a position left when input `feature[i]` ≤
/// `threshold[i]`, right otherwise.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Tree {
    feature: Vec<i32>,
    threshold: Vec<i64>,
    left: Vec<u32>,
    right: Vec<u32>,
    value: Vec<i64>,
}

impl Tree {
    fn leaf_value(&self, x: &RiskInputs) -> i64 {
        let mut i = 0usize;
        loop {
            let f = self.feature[i];
            if f < 0 {
                return self.value[i];
            }
            i = if x.get(f as usize) <= self.threshold[i] {
                self.left[i]
            } else {
                self.right[i]
            } as usize;
        }
    }
}

/// A loaded leaf-risk model (module notes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RiskModel {
    base: i64,
    trees: Vec<Tree>,
    /// `quantiles[k - 1]` for `k` in `1..=K`: the score at or above which a share `k / K` of the
    /// model's test rows lie (the budget table, [`RiskModel::threshold_for_share`]).
    quantiles: Vec<i64>,
    /// SHA-256 of the file (lowercase hex).
    pub sha256: String,
    /// The SHA-256 of the network whose errors it predicts.
    pub net_sha256: String,
    /// The frontier trick.
    pub trick: u8,
}

fn whole(v: &Value) -> Option<i64> {
    v.as_i64()
}

fn ints(v: &Value, what: &str) -> Result<Vec<i64>, String> {
    v.as_array()
        .ok_or_else(|| format!("{what}: not a list"))?
        .iter()
        .map(|x| whole(x).ok_or_else(|| format!("{what}: {x} is not a whole number")))
        .collect()
}

impl RiskModel {
    /// Parse a model file's bytes ([`FORMAT`]).
    pub fn parse(bytes: &[u8]) -> Result<RiskModel, String> {
        let v: Value =
            serde_json::from_slice(bytes).map_err(|e| format!("not a leaf-risk model: {e}"))?;
        if v["format"] != FORMAT {
            return Err(format!("not a {FORMAT} file (format {})", v["format"]));
        }
        if v["encoder"] != crate::encode::ENCODER {
            return Err(format!(
                "encoder {} (this build's is {})",
                v["encoder"],
                crate::encode::ENCODER
            ));
        }
        if v["n_inputs"].as_u64() != Some(N_INPUTS as u64) {
            return Err(format!(
                "{} inputs (this build has {N_INPUTS})",
                v["n_inputs"]
            ));
        }
        let names: Vec<&str> = v["extras"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if names != EXTRA_NAMES {
            return Err("the extras differ from this build's".into());
        }
        let base = whole(&v["base"]).ok_or("base: not a whole number")?;
        let mut trees = Vec::new();
        for (k, t) in v["trees"]
            .as_array()
            .ok_or("trees: not a list")?
            .iter()
            .enumerate()
        {
            let at = |s: &str| format!("tree {k}: {s}");
            let feature = ints(&t["f"], &at("f"))?;
            let threshold = ints(&t["t"], &at("t"))?;
            let left = ints(&t["l"], &at("l"))?;
            let right = ints(&t["r"], &at("r"))?;
            let value = ints(&t["v"], &at("v"))?;
            let n = feature.len();
            if n == 0 || [threshold.len(), left.len(), right.len(), value.len()] != [n; 4] {
                return Err(at("lists of unequal or zero length"));
            }
            for i in 0..n {
                if feature[i] >= 0 {
                    let ok_f = (feature[i] as usize) < N_INPUTS;
                    let ok_c = |c: i64| c > i as i64 && (c as usize) < n;
                    if !ok_f || !ok_c(left[i]) || !ok_c(right[i]) {
                        return Err(at(&format!("node {i}: a bad feature or child")));
                    }
                } else if feature[i] != -1 {
                    return Err(at(&format!("node {i}: feature {}", feature[i])));
                }
            }
            trees.push(Tree {
                feature: feature.iter().map(|&f| f as i32).collect(),
                threshold,
                left: left.iter().map(|&c| c as u32).collect(),
                right: right.iter().map(|&c| c as u32).collect(),
                value,
            });
        }
        let quantiles = match v.get("budget") {
            Some(b) if !b.is_null() => ints(&b["quantiles"], "budget.quantiles")?,
            _ => Vec::new(),
        };
        if quantiles.windows(2).any(|w| w[0] < w[1]) {
            return Err("budget.quantiles must not increase".into());
        }
        Ok(RiskModel {
            base,
            trees,
            quantiles,
            sha256: equity::sha256::sha256_hex(bytes),
            net_sha256: v["net_sha256"].as_str().unwrap_or("").to_string(),
            trick: v["trick"].as_u64().unwrap_or(0) as u8,
        })
    }

    /// Load a model file.
    pub fn load(path: &Path) -> Result<RiskModel, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        RiskModel::parse(&b).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The predicted squared error (units of 2^-40; never negative).
    pub fn predict(&self, x: &RiskInputs) -> i64 {
        let s: i128 = i128::from(self.base)
            + self
                .trees
                .iter()
                .map(|t| i128::from(t.leaf_value(x)))
                .sum::<i128>();
        s.clamp(0, i128::from(i64::MAX)) as i64
    }

    /// The predicted squared error of a frontier position (`net`: the network's value there, in
    /// real team order).
    pub fn score(&self, c: &CanonPos, ctx: &MatchCtx, leaf: &dyn Leaf, net: &MoneyChance) -> i64 {
        self.predict(&RiskInputs::of(c, ctx, leaf, net))
    }

    /// Trees in the model.
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Steps of the budget table (0: the model has none).
    pub fn budget_steps(&self) -> usize {
        self.quantiles.len()
    }

    /// The threshold that picks about a share `k / K` of the model's test rows (`K` the table's
    /// steps): `None` (pick nothing) for `k = 0`, otherwise the table's score at `k`.
    pub fn threshold_for_share(&self, k: usize) -> Option<i64> {
        (k > 0).then(|| self.quantiles[k.min(self.quantiles.len()) - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use equity::ONE;

    fn model_json(trees: &str, quantiles: &str) -> String {
        let names: Vec<String> = EXTRA_NAMES.iter().map(|n| format!("{n:?}")).collect();
        format!(
            "{{\"format\": \"{FORMAT}\", \"encoder\": \"oracle-v2\", \"n_inputs\": {N_INPUTS}, \
             \"extras\": [{}], \"net_sha256\": \"ab\", \"trick\": 3, \"base\": 100, \
             \"trees\": {trees}, \"budget\": {{\"quantiles\": {quantiles}}}}}",
            names.join(", ")
        )
    }

    fn inputs(active: &[u16], dense0: i32, net: [u64; 4], mask: u64) -> RiskInputs {
        let mut f = Features {
            active: [crate::encode::PAD; crate::encode::MAX_ACTIVE],
            n_active: 0,
            dense: [0; N_DENSE],
        };
        for &a in active {
            f.active[usize::from(f.n_active)] = a;
            f.n_active += 1;
        }
        f.dense[0] = dense0;
        let o = Outcomes {
            n: 6,
            mask,
            vecs: vec![
                MoneyChance([ONE, 0, 0, 0]),
                MoneyChance([0, ONE, 0, 0]),
                MoneyChance([0, 0, ONE, 0]),
                MoneyChance([0, 0, 0, ONE]),
                MoneyChance([ONE / 2, ONE / 2, 0, 0]),
                MoneyChance([ONE / 4; 4]),
            ],
        };
        RiskInputs::new(&f, &o, &MoneyChance(net))
    }

    /// Splits go left at ≤ the threshold, binary inputs are 0/1, dense and extras whole numbers;
    /// the prediction is the base plus the leaves, never negative; the budget table; bad files
    /// are refused.
    #[test]
    fn trees_extras_and_budget() {
        let trees = format!(
            "[{{\"f\": [5, -1, {d}], \"t\": [0, 0, -7], \"l\": [1, 0, 3], \"r\": [2, 0, 4], \
               \"v\": [0, 10, 0]}}]",
            d = N_BINARY
        );
        // Node 2 has children 3 and 4 that do not exist: refused.
        assert!(RiskModel::parse(model_json(&trees, "[]").as_bytes()).is_err());
        let trees = format!(
            "[{{\"f\": [5, -1, {d}, -1, -1], \"t\": [0, 0, -7, 0, 0], \"l\": [1, 0, 3, 0, 0], \
               \"r\": [2, 0, 4, 0, 0], \"v\": [0, 10, 0, 1000, -500]}}, \
              {{\"f\": [{e}, -1, -1], \"t\": [2, 0, 0], \"l\": [1, 0, 0], \"r\": [2, 0, 0], \
               \"v\": [0, 1, -1000]}}]",
            d = N_BINARY,
            e = N_FEATURES + 10
        );
        let m = RiskModel::parse(model_json(&trees, "[900, 50, 50, 3]").as_bytes()).expect("ok");
        assert_eq!(m.n_trees(), 2);
        let even = [ONE / 4; 4];
        // Binary 5 off: tree 1 gives 10; possible outcomes 2 (≤ 2): tree 2 gives 1.
        let x = inputs(&[3], 0, even, 0b11);
        assert_eq!(x.get(5), 0);
        assert_eq!(x.extras[10], 2);
        assert_eq!(m.predict(&x), 100 + 10 + 1);
        // Binary 5 on, dense 0 = −7 (≤ −7): 1000; three possible outcomes: −1000.
        let x = inputs(&[5, 9], -7, even, 0b111);
        assert_eq!((x.get(5), x.get(9), x.get(N_BINARY)), (1, 1, -7));
        assert_eq!(m.predict(&x), 100 + 1000 - 1000);
        // Dense −6 > −7: −500, clamped at 0.
        let x = inputs(&[5], -6, even, 0b111);
        assert_eq!(m.predict(&x), 0);
        assert_eq!(m.threshold_for_share(0), None);
        assert_eq!(m.threshold_for_share(1), Some(900));
        assert_eq!(m.threshold_for_share(4), Some(3));
        assert_eq!(m.threshold_for_share(9), Some(3));
        assert!(RiskModel::parse(model_json(&trees, "[1, 2]").as_bytes()).is_err());
        let bad = model_json(&trees, "[]").replace("oracle-v2", "oracle-v1");
        assert!(RiskModel::parse(bad.as_bytes()).is_err());
        let bad = model_json(&trees, "[]").replace("\"base\": 100", "\"base\": 1.5");
        assert!(RiskModel::parse(bad.as_bytes()).is_err());
    }

    /// The extras of a hand-made case.
    #[test]
    fn extras_by_hand() {
        let q = ONE / 4;
        // Network: 1/2, 1/4, 1/4, 0; possible: team 0 wins, team 1 wins, the even split.
        let x = inputs(&[], 0, [2 * q, q, q, 0], 0b100011);
        let e = x.extras;
        // Teams 0 and 1 can win the game, nobody has a game: class 2.
        assert_eq!(e[0], 2);
        assert_eq!(e[1..5], [2 * q as i64, q as i64, q as i64, 0]);
        assert_eq!((e[5], e[6]), (2 * q as i64, q as i64));
        // Σ_{t<u} m_t m_u = (2·1 + 2·1 + 1·1)/16 = 5/16 of 2^64, in units of 2^-32.
        assert_eq!(e[7], 5 * (1i64 << 28));
        // L1 to [1,0,0,0] = 1/2+1/4+1/4 = 1; to [0,1,0,0] = 1/2+3/4+1/4 = 1.5; to even: 1/4+0+0+1/4.
        assert_eq!((e[8], e[9]), (2 * q as i64, 6 * q as i64));
        assert_eq!((e[10], e[11]), (3, 2));
        // Team 0 ranges 0..1.
        assert_eq!(e[12], ONE as i64);
        // A team with a game that can win: class 4 when another can win too.
        let mut f = Features {
            active: [crate::encode::PAD; crate::encode::MAX_ACTIVE],
            n_active: 1,
            dense: [0; N_DENSE],
        };
        f.active[0] = OFF_GAMES as u16;
        let o = Outcomes {
            n: 2,
            mask: 0b11,
            vecs: vec![MoneyChance([ONE, 0, 0, 0]), MoneyChance([0, ONE, 0, 0])],
        };
        assert_eq!(RiskInputs::new(&f, &o, &MoneyChance([q; 4])).extras[0], 4);
    }

    /// The parity fixture shared with Python (`python/nn/tests/fixtures/leafrisk-fixture.json`:
    /// real trick-3 harvest rows with O_3's value, a model's first trees, and the extras and
    /// scores Python computed): the same extras and the same scores, bit for bit.
    #[test]
    fn python_fixture() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/nn/tests/fixtures/leafrisk-fixture.json");
        let v: Value =
            serde_json::from_slice(&std::fs::read(&path).expect("fixture")).expect("json");
        let m = RiskModel::parse(v["model"].to_string().as_bytes()).expect("model");
        assert!(m.n_trees() > 0);
        let u = |x: &Value| x.as_u64().expect("whole number");
        let rows = v["rows"].as_array().expect("rows");
        assert!(rows.len() >= 16);
        let mut scores = std::collections::BTreeSet::new();
        for r in rows {
            let mut f = Features {
                active: [crate::encode::PAD; crate::encode::MAX_ACTIVE],
                n_active: 0,
                dense: [0; N_DENSE],
            };
            for a in r["active"].as_array().expect("active") {
                f.active[usize::from(f.n_active)] = u(a) as u16;
                f.n_active += 1;
            }
            for (k, d) in r["dense"].as_array().expect("dense").iter().enumerate() {
                f.dense[k] = d.as_i64().expect("dense") as i32;
            }
            let vecs: Vec<MoneyChance> = r["outcomes"]
                .as_array()
                .expect("outcomes")
                .iter()
                .map(|o| MoneyChance(core::array::from_fn(|t| u(&o[t]))))
                .collect();
            let o = Outcomes {
                n: vecs.len(),
                mask: u(&r["mask"]),
                vecs,
            };
            let net = MoneyChance(core::array::from_fn(|t| u(&r["net"][t])));
            let x = RiskInputs::new(&f, &o, &net);
            let want: Vec<i64> = r["extras"]
                .as_array()
                .expect("extras")
                .iter()
                .map(|e| e.as_i64().expect("extra"))
                .collect();
            assert_eq!(x.extras.to_vec(), want);
            let s = m.predict(&x);
            assert_eq!(s, r["score"].as_i64().expect("score"));
            scores.insert(s);
        }
        assert!(scores.len() > 4, "the fixture's scores differ: {scores:?}");
    }
}
