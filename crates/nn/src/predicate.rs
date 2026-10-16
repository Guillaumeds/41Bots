//! Predicates over the encoder's features (docs/m3-hybrid/audit-loop.md §6): the groups of
//! positions the audit loop finds weak, shared by the auditor (Python, `fortyone_nn.audit`) and
//! every Rust piece that selects positions by them (the targeted harvest, the mini golden sets,
//! the golden slices).
//!
//! **Format** (a JSON file; its SHA-256 identifies it in every sidecar and report):
//!
//! ```json
//! [ { "name": "race-4 seats 3-4",
//!     "conditions": [ { "feature": 801, "op": "==", "value": 1 },
//!                     { "feature": 845, "op": "<=", "value": -3145728 } ],
//!     "weight": 0.012, "bias": 0.006, "sd": 0.027 } ]
//! ```
//!
//! or the same list as `{"groups": [...], ...}` (other top-level keys are kept by the writer and
//! ignored here). A group is the conjunction of its conditions (no conditions: every position).
//! Each condition names a feature by its index in the encoder's feature vector
//! ([`crate::encode`], encoder `oracle-v2`): indices `0..N_BINARY` (809) are the binary features,
//! `N_BINARY..N_FEATURES` (809..909) the dense ones, dense feature `j` at index `N_BINARY + j`.
//!
//! - A **binary** feature takes `"op": "=="` with `"value"` 0 or 1: the feature is off or on (on
//!   when its index is in the row's `active` list).
//! - A **dense** feature takes `"op": "<="` or `">"` with an integer `"value"` in the dense
//!   feature's own units, exactly as the encoder stores it and the label shards hold it
//!   (`dense` column): `log2(p) / 32` in units of 2^-24, an integer in `[−2^24, 0]` (a
//!   threshold of `-3145728` is `log2(p) / 32 = −0.1875`, `p = 2^-6`). Comparisons are of whole
//!   numbers, so Rust and Python agree exactly. A threshold between two whole numbers (a tree's
//!   midpoint) is written as its floor: for a whole `x`, `x <= t` exactly when `x <= floor(t)`
//!   and `x > t` exactly when `x > floor(t)`. A value written as a number with a fraction
//!   (`-3.5`) is refused; one with a zero fraction (`1.0`) is read as the whole number.
//!
//! Anything else (another op, a binary value other than 0 or 1, an index out of range, a group
//! without a name) is refused with the reason. Names need not be unique (the auditor names a group
//! by its conditions in words, rounded); a group is identified by its place in the file. Every other key of a
//! group (`weight`, `bias`, `sd`, `example`, …: the auditor's measurements) is kept as it is, for
//! reports only.
//!
//! **Membership** is a function of the encoded features alone ([`Predicates::holds`]), so a
//! position's groups are the same in a shard row and in a search: [`Predicates::features_of`]
//! encodes a canonical position for them (the dense features only when a condition needs one).
//!
//! The parity fixture `python/nn/tests/fixtures/{predicate-rows,predicates,predicate-members}.json`
//! (real rows, groups covering every op and boundary, and each group's members; written by the
//! Python auditor's tests) is checked here too; `nn-tool groups` writes the membership of any
//! shard.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{Map, Value};
use solver::frontier::CanonPos;
use solver::{Leaf, MatchCtx};

use crate::encode::{
    DenseKey, Features, MAX_ACTIVE, N_BINARY, N_DENSE, N_FEATURES, PAD, binary_features,
    dense_features,
};

/// A condition's comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// A binary feature is off (value 0) or on (value 1).
    Eq,
    /// A dense feature is at most the value.
    Le,
    /// A dense feature is above the value.
    Gt,
}

impl Op {
    /// As written in the file.
    pub fn text(self) -> &'static str {
        match self {
            Op::Eq => "==",
            Op::Le => "<=",
            Op::Gt => ">",
        }
    }
}

/// One condition on one feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Condition {
    /// The feature's index in the encoder's vector (binary, then dense).
    pub feature: u16,
    /// The comparison.
    pub op: Op,
    /// 0 or 1 for a binary feature; the threshold in units of 2^-24 for a dense one.
    pub value: i64,
}

impl Condition {
    /// Does the condition hold for `f`?
    #[inline]
    pub fn holds(&self, f: &Features) -> bool {
        let i = usize::from(self.feature);
        match self.op {
            Op::Eq => {
                let on = f.active().binary_search(&self.feature).is_ok();
                on == (self.value == 1)
            }
            Op::Le => i64::from(f.dense[i - N_BINARY]) <= self.value,
            Op::Gt => i64::from(f.dense[i - N_BINARY]) > self.value,
        }
    }

    /// The condition as text, e.g. `f801 == 1`, `d36 <= -3145728` (dense features by their
    /// dense index).
    pub fn text(&self) -> String {
        let i = usize::from(self.feature);
        if i < N_BINARY {
            format!("f{i} {} {}", self.op.text(), self.value)
        } else {
            format!("d{} {} {}", i - N_BINARY, self.op.text(), self.value)
        }
    }
}

/// One group: a name and a conjunction of conditions.
#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    /// Its name (for reports; not necessarily unique).
    pub name: String,
    /// All of these hold.
    pub conditions: Vec<Condition>,
    /// The group's other keys (the auditor's `weight`, `bias`, `sd`, …), for reports.
    pub meta: Map<String, Value>,
}

impl Group {
    /// Does every condition hold for `f`?
    pub fn holds(&self, f: &Features) -> bool {
        self.conditions.iter().all(|c| c.holds(f))
    }

    /// The conditions as text, e.g. `f801 == 1 and d36 <= -3145728` (`every position` with
    /// none).
    pub fn conditions_text(&self) -> String {
        if self.conditions.is_empty() {
            return "every position".to_string();
        }
        let c: Vec<String> = self.conditions.iter().map(Condition::text).collect();
        c.join(" and ")
    }
}

/// A predicate file: its groups, in file order, and the file's SHA-256.
#[derive(Clone, Debug, PartialEq)]
pub struct Predicates {
    /// The groups.
    pub groups: Vec<Group>,
    /// SHA-256 of the file's bytes (lowercase hex).
    pub sha256: String,
    dense: bool,
}

/// A whole number from a JSON number (an integral float is read as the whole number).
fn whole(v: &Value) -> Option<i64> {
    if let Some(x) = v.as_i64() {
        return Some(x);
    }
    let x = v.as_f64()?;
    (x.is_finite() && x.fract() == 0.0 && x.abs() <= 9.0e15).then_some(x as i64)
}

fn condition(g: &str, k: usize, c: &Value) -> Result<Condition, String> {
    let at = || format!("group {g:?}, condition {}", k + 1);
    let o = c
        .as_object()
        .ok_or_else(|| format!("{}: not an object", at()))?;
    if let Some(x) = o
        .keys()
        .find(|x| !["feature", "op", "value"].contains(&x.as_str()))
    {
        return Err(format!(
            "{}: unknown key {x:?} (a condition is feature, op, value)",
            at()
        ));
    }
    let feature = o
        .get("feature")
        .and_then(whole)
        .filter(|&i| (0..N_FEATURES as i64).contains(&i))
        .ok_or_else(|| {
            format!(
                "{}: feature must be a whole number 0..{N_FEATURES} (binary 0..{N_BINARY}, \
                 dense {N_BINARY}..{N_FEATURES})",
                at()
            )
        })? as u16;
    let op = match o.get("op").and_then(Value::as_str) {
        Some("==") => Op::Eq,
        Some("<=") => Op::Le,
        Some(">") => Op::Gt,
        _ => return Err(format!("{}: op must be \"==\", \"<=\" or \">\"", at())),
    };
    let raw = o
        .get("value")
        .ok_or_else(|| format!("{}: no value", at()))?;
    let value = whole(raw).ok_or_else(|| {
        format!(
            "{}: value {raw} is not a whole number (dense thresholds are whole numbers in units \
             of 2^-24: write floor(t) for a threshold t between two)",
            at()
        )
    })?;
    let binary = usize::from(feature) < N_BINARY;
    match (binary, op) {
        (true, Op::Eq) if value == 0 || value == 1 => {}
        (true, Op::Eq) => {
            return Err(format!(
                "{}: a binary feature's value is 0 or 1, not {value}",
                at()
            ));
        }
        (true, _) => {
            return Err(format!(
                "{}: feature {feature} is binary: its op is \"==\" (0 or 1)",
                at()
            ));
        }
        (false, Op::Eq) => {
            return Err(format!(
                "{}: feature {feature} is dense (dense index {}): its op is \"<=\" or \">\"",
                at(),
                usize::from(feature) - N_BINARY
            ));
        }
        (false, _) => {}
    }
    Ok(Condition { feature, op, value })
}

impl Predicates {
    /// Parse a predicate file's bytes (module notes).
    pub fn parse(bytes: &[u8]) -> Result<Predicates, String> {
        let v: Value =
            serde_json::from_slice(bytes).map_err(|e| format!("not a predicate file: {e}"))?;
        let list = match &v {
            Value::Array(a) => a,
            Value::Object(o) => o.get("groups").and_then(Value::as_array).ok_or(
                "a predicate file is a list of groups, or an object with a \"groups\" list",
            )?,
            _ => return Err("a predicate file is a list of groups".into()),
        };
        let mut groups: Vec<Group> = Vec::with_capacity(list.len());
        for (k, g) in list.iter().enumerate() {
            let o = g
                .as_object()
                .ok_or_else(|| format!("group {}: not an object", k + 1))?;
            let name = o
                .get("name")
                .and_then(Value::as_str)
                .filter(|n| !n.trim().is_empty())
                .ok_or_else(|| format!("group {}: no name", k + 1))?
                .to_string();
            let conds = o
                .get("conditions")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("group {name:?}: no conditions list"))?;
            let conditions = conds
                .iter()
                .enumerate()
                .map(|(i, c)| condition(&name, i, c))
                .collect::<Result<Vec<_>, _>>()?;
            let meta = o
                .iter()
                .filter(|(k, _)| !["name", "conditions"].contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            groups.push(Group {
                name,
                conditions,
                meta,
            });
        }
        let dense = groups
            .iter()
            .flat_map(|g| &g.conditions)
            .any(|c| usize::from(c.feature) >= N_BINARY);
        Ok(Predicates {
            groups,
            sha256: equity::sha256::sha256_hex(bytes),
            dense,
        })
    }

    /// Read and parse a predicate file.
    pub fn load(path: &Path) -> Result<Predicates, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Predicates::parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The number of groups.
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// No groups at all.
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Whether any condition is on a dense feature.
    pub fn uses_dense(&self) -> bool {
        self.dense
    }

    /// Is `f` in group `g`?
    pub fn holds(&self, g: usize, f: &Features) -> bool {
        self.groups[g].holds(f)
    }

    /// The groups `f` is in, ascending.
    pub fn members(&self, f: &Features) -> Vec<u32> {
        (0..self.groups.len())
            .filter(|&g| self.groups[g].holds(f))
            .map(|g| g as u32)
            .collect()
    }

    /// Is `f` in any group?
    pub fn any(&self, f: &Features) -> bool {
        self.groups.iter().any(|g| g.holds(f))
    }

    /// The encoded features a canonical position's membership is decided on: the encoder's
    /// ([`crate::encode::encode`]), with the dense ones only when a condition needs them (else
    /// left at 0, which no condition reads).
    pub fn features_of(&self, c: &CanonPos, ctx: &MatchCtx, leaf: &dyn Leaf) -> Features {
        let mut f = binary_features(c, ctx);
        if self.dense {
            f.dense = dense_features(&DenseKey::of(c, ctx), leaf);
        }
        f
    }

    /// The groups of a canonical position ([`Predicates::features_of`], [`Predicates::members`]).
    pub fn members_of(&self, c: &CanonPos, ctx: &MatchCtx, leaf: &dyn Leaf) -> Vec<u32> {
        self.members(&self.features_of(c, ctx, leaf))
    }

    /// The groups' names, in order.
    pub fn names(&self) -> Vec<&str> {
        self.groups.iter().map(|g| g.name.as_str()).collect()
    }

    /// One line per group: its name and conditions (for logs and reports).
    pub fn describe(&self) -> Vec<String> {
        self.groups
            .iter()
            .map(|g| format!("{}: {}", g.name, g.conditions_text()))
            .collect()
    }
}

/// Memberships of many positions (a search's reached positions), each position's dense
/// features computed once per [`DenseKey`] (positions of one search share few keys); the same
/// answers as [`Predicates::members_of`].
pub struct Evaluator<'a> {
    preds: &'a Predicates,
    leaf: &'a (dyn Leaf + Sync),
    dense: BTreeMap<DenseKey, [i32; N_DENSE]>,
}

impl<'a> Evaluator<'a> {
    /// An evaluator of `preds`, with `leaf` for the dense features.
    pub fn new(preds: &'a Predicates, leaf: &'a (dyn Leaf + Sync)) -> Self {
        Evaluator {
            preds,
            leaf,
            dense: BTreeMap::new(),
        }
    }

    /// The groups of a canonical position, ascending.
    pub fn members(&mut self, c: &CanonPos, ctx: &MatchCtx) -> Vec<u32> {
        let mut f = binary_features(c, ctx);
        if self.preds.dense {
            let leaf = self.leaf;
            f.dense = *self
                .dense
                .entry(DenseKey::of(c, ctx))
                .or_insert_with_key(|k| dense_features(k, leaf));
        }
        self.preds.members(&f)
    }
}

/// A row's features from a shard's `active` (padded with [`PAD`]) and `dense` columns, for
/// membership tests on stored rows.
pub fn features_from(active: &[u16; MAX_ACTIVE], dense: &[i32; N_DENSE]) -> Features {
    Features {
        active: *active,
        n_active: active.iter().take_while(|&&a| a != PAD).count() as u8,
        dense: *dense,
    }
}

/// Parse a share `"0.5"`: a number 0 to 1 with at most 6 decimals, as millionths (so share
/// comparisons are of whole numbers: `in_group × 1,000,000 ≥ share × reached`).
pub fn parse_share(s: &str) -> Result<u64, String> {
    let t = s.trim();
    let (int, frac) = t.split_once('.').unwrap_or((t, ""));
    let digits = |x: &str| x.bytes().all(|b| b.is_ascii_digit());
    if (int.is_empty() && frac.is_empty()) || !digits(int) || !digits(frac) || frac.len() > 6 {
        return Err(format!(
            "share {s:?}: a number from 0 to 1 with at most 6 decimals"
        ));
    }
    let i: u64 = if int.is_empty() {
        0
    } else {
        int.parse().map_err(|_| format!("share {s:?}: too large"))?
    };
    let f: u64 = format!("{frac:0<6}").parse().expect("digits");
    let m = i
        .checked_mul(1_000_000)
        .and_then(|x| x.checked_add(f))
        .filter(|&m| m <= 1_000_000)
        .ok_or_else(|| format!("share {s:?}: at most 1"))?;
    Ok(m)
}

/// Is `part` of `whole` at least the share `micro` (millionths)? `whole` 0 never is.
pub fn at_share(part: u64, whole: u64, micro: u64) -> bool {
    whole > 0 && u128::from(part) * 1_000_000 >= u128::from(micro) * u128::from(whole)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feat(active: &[u16], dense: [i32; N_DENSE]) -> Features {
        let mut a = [PAD; MAX_ACTIVE];
        a[..active.len()].copy_from_slice(active);
        features_from(&a, &dense)
    }

    #[test]
    fn conditions_hold_as_documented() {
        let p = Predicates::parse(
            br#"{"groups": [
                {"name": "on", "conditions": [{"feature": 801, "op": "==", "value": 1}],
                 "weight": 0.01, "bias": 0.002, "sd": 0.02},
                {"name": "off", "conditions": [{"feature": 801, "op": "==", "value": 0.0}]},
                {"name": "low", "conditions": [{"feature": 809, "op": "<=", "value": -100}]},
                {"name": "high", "conditions": [{"feature": 908, "op": ">", "value": -100}]},
                {"name": "all", "conditions": []},
                {"name": "both", "conditions": [{"feature": 3, "op": "==", "value": 1},
                                                {"feature": 845, "op": ">", "value": -5}]}
            ], "made_by": "test"}"#,
        )
        .expect("parses");
        assert_eq!(p.len(), 6);
        assert!(p.uses_dense());
        assert_eq!(p.groups[0].meta["sd"], serde_json::json!(0.02));
        let mut d = [0i32; N_DENSE];
        d[0] = -100;
        d[99] = -100;
        d[36] = -5;
        let f = feat(&[3, 801], d);
        assert_eq!(p.members(&f), [0, 2, 4]);
        d[0] = -99;
        d[99] = -99;
        d[36] = -4;
        let f = feat(&[3, 800], d);
        assert_eq!(p.members(&f), [1, 3, 4, 5]);
        assert!(p.any(&f));
        assert_eq!(
            p.describe()[5],
            "both: f3 == 1 and d36 > -5",
            "dense features by their dense index"
        );
        let q = Predicates::parse(
            br#"[{"name": "x", "conditions": [{"feature": 5, "op": "==", "value": 1}]}]"#,
        )
        .expect("a bare list");
        assert!(!q.uses_dense());
        assert_eq!(q.sha256.len(), 64);
    }

    #[test]
    fn bad_files_are_refused() {
        let bad = |s: &str, says: &str| {
            let e = Predicates::parse(s.as_bytes()).expect_err(s);
            assert!(e.contains(says), "{s}: {e}");
        };
        let one = |c: &str| format!(r#"[{{"name": "g", "conditions": [{c}]}}]"#);
        bad("{", "not a predicate file");
        bad("7", "a list of groups");
        bad(r#"{"x": []}"#, "\"groups\" list");
        bad(r#"[{"conditions": []}]"#, "no name");
        bad(r#"[{"name": " ", "conditions": []}]"#, "no name");
        bad(r#"[{"name": "a"}]"#, "no conditions list");
        bad(
            &one(r#"{"feature": 909, "op": ">", "value": 0}"#),
            "feature must be",
        );
        bad(
            &one(r#"{"feature": -1, "op": "==", "value": 0}"#),
            "feature must be",
        );
        bad(
            &one(r#"{"feature": 3, "op": "<=", "value": 0}"#),
            "is binary",
        );
        bad(&one(r#"{"feature": 3, "op": "==", "value": 2}"#), "0 or 1");
        bad(
            &one(r#"{"feature": 809, "op": "==", "value": 0}"#),
            "is dense",
        );
        bad(
            &one(r#"{"feature": 809, "op": "<", "value": 0}"#),
            "op must be",
        );
        bad(
            &one(r#"{"feature": 809, "op": "<=", "value": -3.5}"#),
            "not a whole number",
        );
        bad(
            &one(r#"{"feature": 809, "op": "<=", "value": "x"}"#),
            "not a whole number",
        );
        bad(&one(r#"{"feature": 809, "op": "<="}"#), "no value");
        bad(
            &one(r#"{"feature": 809, "op": "<=", "value": 1, "x": 1}"#),
            "unknown key",
        );
    }

    #[test]
    fn shares_parse_and_compare_in_whole_numbers() {
        assert_eq!(parse_share("0.5"), Ok(500_000));
        assert_eq!(parse_share("1"), Ok(1_000_000));
        assert_eq!(parse_share(".25"), Ok(250_000));
        assert_eq!(parse_share("0.000001"), Ok(1));
        for b in ["", ".", "1.5", "2", "-0.1", "0.1234567", "x"] {
            assert!(parse_share(b).is_err(), "{b:?}");
        }
        assert!(at_share(1, 2, 500_000));
        assert!(!at_share(1, 3, 500_000));
        assert!(at_share(0, 3, 0));
        assert!(!at_share(0, 0, 0), "no reached positions: in no group");
    }

    /// The parity fixture shared with the Python auditor (`python/nn/tests/fixtures`: 48 real
    /// trick-4 rows, 10 groups covering every op and boundary, and each group's members, written
    /// by `fortyone_nn.audit`'s tests): this evaluator gives the same members.
    #[test]
    fn python_parity_fixture_holds() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/nn/tests/fixtures");
        let read = |f: &str| -> Value {
            serde_json::from_slice(&std::fs::read(dir.join(f)).expect("fixture")).expect("json")
        };
        let rows = read("predicate-rows.json");
        let want = read("predicate-members.json");
        let p = Predicates::load(&dir.join("predicates.json")).expect("predicates");
        let rows = rows["rows"].as_array().expect("rows");
        assert_eq!(rows.len() as u64, want["rows"].as_u64().expect("rows"));
        let feats: Vec<Features> = rows
            .iter()
            .map(|r| {
                let mut active = [PAD; MAX_ACTIVE];
                let list: Vec<u16> = r["active"]
                    .as_array()
                    .expect("active")
                    .iter()
                    .map(|x| x.as_u64().expect("index") as u16)
                    .filter(|&x| x != PAD)
                    .collect();
                active[..list.len()].copy_from_slice(&list);
                let mut dense = [0i32; N_DENSE];
                for (d, x) in dense.iter_mut().zip(r["dense"].as_array().expect("dense")) {
                    *d = x.as_i64().expect("whole") as i32;
                }
                features_from(&active, &dense)
            })
            .collect();
        let members = want["members"].as_array().expect("members");
        assert_eq!(members.len(), p.len());
        for (g, m) in members.iter().enumerate() {
            let want: Vec<u64> = m
                .as_array()
                .expect("list")
                .iter()
                .map(|x| x.as_u64().expect("row"))
                .collect();
            let got: Vec<u64> = (0..feats.len())
                .filter(|&i| p.holds(g, &feats[i]))
                .map(|i| i as u64)
                .collect();
            assert_eq!(got, want, "group {g}: {}", p.groups[g].name);
        }
        assert!(p.groups.iter().any(|g| g.conditions.is_empty()));
        assert!(p.uses_dense());
    }
}
