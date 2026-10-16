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
//! without a name, two groups with one name) is refused with the reason. Every other key of a
//! group (`weight`, `bias`, `sd`, `example`, …: the auditor's measurements) is kept as it is, for
//! reports only.
//!
//! **Membership** is a function of the encoded features alone ([`Predicates::holds`]), so a
//! position's groups are the same in a shard row and in a search: [`Predicates::features_of`]
//! encodes a canonical position for them (the dense features only when a condition needs one).
//!
//! The parity fixture `crates/nn/testdata/predicate-fixture.{npy,json,expected.json}` (a small
//! shard, a predicate file, and every row's membership) is checked here and by the Python side;
//! `nn-tool groups` writes the membership of any shard.

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
    /// Its name (unique in its file).
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
            if groups.iter().any(|x| x.name == name) {
                return Err(format!("two groups are named {name:?}"));
            }
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
    use crate::shard::from_npy;

    fn testdata(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata")
            .join(name)
    }

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
            r#"[{"name": "a", "conditions": []}, {"name": "a", "conditions": []}]"#,
            "two groups are named",
        );
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

    /// The parity fixture: the committed shard, predicate file and memberships agree with this
    /// evaluator, and are what [`fixture`] makes (so they cannot drift from the encoder).
    #[test]
    fn parity_fixture_holds() {
        let shard = std::fs::read(testdata("predicate-fixture.npy")).expect("shard");
        let rows = from_npy(&shard).expect("a shard");
        let p = Predicates::load(&testdata("predicate-fixture.json")).expect("predicates");
        let want: Value = serde_json::from_slice(
            &std::fs::read(testdata("predicate-fixture.expected.json")).expect("expected"),
        )
        .expect("json");
        assert_eq!(want["predicates_sha256"], p.sha256.as_str());
        assert_eq!(
            want["shard_sha256"],
            equity::sha256::sha256_hex(&shard).as_str()
        );
        let m = want["membership"].as_array().expect("membership");
        assert_eq!(m.len(), rows.len());
        let mut hits = vec![0usize; p.len()];
        for (r, w) in rows.iter().zip(m) {
            let got: Vec<u64> = (0..p.len())
                .map(|g| u64::from(p.holds(g, &r.features)))
                .collect();
            let w: Vec<u64> = w
                .as_array()
                .expect("row")
                .iter()
                .map(|x| x.as_u64().expect("0 or 1"))
                .collect();
            assert_eq!(got, w);
            for (h, g) in hits.iter_mut().zip(&got) {
                *h += *g as usize;
            }
        }
        // Every group holds for some rows and not for others, so the fixture tests something.
        assert!(
            hits.iter().all(|&h| h > 0 && h < rows.len()),
            "{hits:?} of {}",
            rows.len()
        );
        let (s, j, e) = fixture();
        assert!(s == shard, "the committed shard is the fixture's");
        assert_eq!(
            j,
            std::fs::read(testdata("predicate-fixture.json")).expect("json")
        );
        assert_eq!(
            e,
            std::fs::read(testdata("predicate-fixture.expected.json")).expect("expected")
        );
    }

    /// Writes the parity fixture's files (run by hand after an encoder change:
    /// `cargo test -p nn write_parity_fixture -- --ignored`).
    #[test]
    #[ignore = "writes crates/nn/testdata"]
    fn write_parity_fixture() {
        let (s, j, e) = fixture();
        let dir = testdata("");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(testdata("predicate-fixture.npy"), s).expect("write");
        std::fs::write(testdata("predicate-fixture.json"), j).expect("write");
        std::fs::write(testdata("predicate-fixture.expected.json"), e).expect("write");
    }

    /// The parity fixture: 48 random trick-4 positions (as `nn-tool fixture --trick 4 --count 48
    /// --seed 7`), a predicate file whose dense thresholds sit exactly on values the rows hold
    /// (so `<=` and `>` are tested at equality), and every row's membership.
    fn fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        use crate::encode::{OFF_GAMES, OFF_SCORE, OFF_TRICKS_DONE, OFF_TRUMP, encode};
        use crate::outcome::{outcomes, vocab};
        use crate::shard::{Row, SOURCE_FIXTURE, to_npy};
        use crate::synthetic::SyntheticLeaf;
        let leaf = SyntheticLeaf;
        let rows: Vec<Row> = (0..48)
            .map(|i| {
                let (pos, ctx, _) = solver::worlds::sample_world(7, 4, i, 0);
                let v = solver::solve_fast(&pos, ctx, &leaf, solver::FastOptions::default())
                    .value
                    .expect("solved");
                let c = solver::canonical(&pos);
                let (features, r) = encode(&c, &ctx, &leaf);
                let value = v.rotate(r % 4);
                let o = outcomes(&c, &ctx, &leaf);
                let label = o.label(&value).expect("an outcome");
                Row {
                    features,
                    value,
                    source: SOURCE_FIXTURE,
                    trick: 4,
                    label,
                    mask: o.mask,
                    outcomes: o.vecs,
                }
            })
            .collect();
        let shard = to_npy(&rows, vocab(4));
        // Thresholds on values the rows hold: the median of dense features 0 (the leader's team's
        // chance if the hand ended now) and 36 (after the leader's team wins the next two tricks).
        let median = |j: usize| {
            let mut v: Vec<i32> = rows.iter().map(|r| r.features.dense[j]).collect();
            v.sort_unstable();
            v[v.len() / 2]
        };
        let (m0, m36) = (median(0), median(36));
        // A binary feature on in some rows and off in others: the most balanced of the trump
        // suits, tricks done, team 0's scores and the games won.
        let candidates: Vec<usize> = (OFF_TRUMP..OFF_TRUMP + 4)
            .chain(OFF_GAMES..OFF_GAMES + 4)
            .chain(OFF_SCORE..OFF_SCORE + 15)
            .chain(OFF_TRICKS_DONE..OFF_TRICKS_DONE + 5)
            .collect();
        let on = |i: usize| {
            rows.iter()
                .filter(|r| r.features.active().contains(&(i as u16)))
                .count()
        };
        let b = *candidates
            .iter()
            .min_by_key(|&&i| (on(i) as i64 - 24).abs())
            .expect("a feature");
        let d = |j: usize| N_BINARY + j;
        let text = serde_json::to_string_pretty(&serde_json::json!({
            "about": "parity fixture of crates/nn/src/predicate.rs: the groups of \
                      predicate-fixture.expected.json, evaluated on predicate-fixture.npy",
            "groups": [
                { "name": "binary on", "conditions": [{ "feature": b, "op": "==", "value": 1 }],
                  "weight": 0.5, "bias": 0.001, "sd": 0.01 },
                { "name": "binary off", "conditions": [{ "feature": b, "op": "==", "value": 0 }] },
                { "name": "dense at most", "conditions": [{ "feature": d(0), "op": "<=", "value": m0 }] },
                { "name": "dense above", "conditions": [{ "feature": d(0), "op": ">", "value": m0 }] },
                { "name": "conjunction", "conditions": [
                    { "feature": b, "op": "==", "value": 1 },
                    { "feature": d(36), "op": ">", "value": m36 }] },
                { "name": "below the median", "conditions": [
                    { "feature": d(36), "op": "<=", "value": m36 - 1 }] },
            ]
        }))
        .expect("json")
            + "\n";
        let p = Predicates::parse(text.as_bytes()).expect("the fixture's predicates parse");
        let membership: Vec<Vec<u8>> = rows
            .iter()
            .map(|r| {
                (0..p.len())
                    .map(|g| u8::from(p.holds(g, &r.features)))
                    .collect()
            })
            .collect();
        let expected = serde_json::to_string(&serde_json::json!({
            "about": "row r, group g: 1 when row r of predicate-fixture.npy is in group g of \
                      predicate-fixture.json",
            "shard_sha256": equity::sha256::sha256_hex(&shard),
            "predicates_sha256": p.sha256,
            "groups": p.names(),
            "membership": membership,
        }))
        .expect("json")
            + "\n";
        (shard, text.into_bytes(), expected.into_bytes())
    }
}
