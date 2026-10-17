//! Stored label shards reused by a new run (`label_inputs`; docs/m3-hybrid/audit-loop.md §10,
//! owner 2026-10-17: "labels are never made twice"), instead of generated again.
//!
//! ```json
//! "label_inputs": [
//!   { "set": "random", "seed": 321, "rows": 25600000,
//!     "shard": "inputs/sha256/<hex>", "sidecar": "inputs/sha256/<hex>" },
//!   { "set": "val", "seed": 401, "shard": "…", "sidecar": "…" },
//!   { "set": "extra", "seed": 33151, "weight": 2, "shard": "…", "sidecar": "…",
//!     "rowweights": "inputs/sha256/<hex>" } ]
//! ```
//! Each entry names a stored shard and its sidecar (the `NAME.npy.json` `search labels` wrote) by
//! key, the set it stands for (`random`, `rules`, `harvest`, `race`, `val`, or `extra`: any
//! other training shard, such as an earlier run's targeted harvest) and its seed; optionally its
//! rows, its loss weight (`weight`, a training shard's `NAME.npy.weight`; not for `val`) and its
//! per-row weights (`rowweights`, the key of its `NAME.npy.rowweights`). A run with
//! `upload_shards` writes these entries, ready to paste, in `out/hybrid/shards.json`
//! (`label_inputs`), and in the audit loop (additive mode, or trick 2 where the targeted shards are
//! shared anyway) also shares every targeted harvest (set `extra`, with its sidecar, weight and row
//! weights) and lists it in `out/hybrid/audit/label_inputs.json`: an additive run trains on old and
//! new data together, so the old targeted data is kept too.
//!
//! **Skipped generation.** A set named here is not generated: its `labels_<set>` and
//! `labels_<set>_seed` (`val_count` and `val_seed` for `val`) are refused, not ignored; the other
//! sets are generated as before. The reused training shards come first in the training set, in
//! the order given, then the generated ones; the validation shard is the reused one when given.
//! Not with `label_parts` above 1.
//!
//! **Checked before use** (step "label inputs", at the start of the labels step): each file's
//! SHA-256 is its key (`remote::fetch_input`); the sidecar records that SHA-256 for the shard
//! (`shard_sha256`), the format `41bots-label-shard-v1`, this binary's encoder and feature count,
//! the rung's trick (`cut`) and its outcomes, this binary's values version, the entry's seed, a
//! source that fits the set, and the rows the shard's header holds (and the entry's `rows`, when
//! given); the row weights, when given, are one per row. Anything else stops the chunk before any
//! training.
//!
//! **Seeds.** A reused shard's seed counts like a generated one's: training, validation, audit,
//! targeted-harvest and golden seeds must all differ (spec §5).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use serde_json::{Map, Value, json};

use super::{GOLDEN_SEED, Rung, read_json, secs, write_json};
use crate::remote::{fetch_input, input_hex, input_key};

/// The settings of this module.
pub(crate) const KEYS: [&str; 1] = ["label_inputs"];

/// The sets an entry can stand for.
const SETS: [&str; 6] = ["random", "rules", "harvest", "race", "val", "extra"];

/// The keys of an entry.
const ENTRY_KEYS: [&str; 7] = [
    "set",
    "seed",
    "rows",
    "shard",
    "sidecar",
    "weight",
    "rowweights",
];

/// One stored shard to reuse.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LabelInput {
    pub set: &'static str,
    pub seed: u64,
    pub rows: Option<u64>,
    /// SHA-256 of the shard, of its sidecar, of its row weights.
    pub shard: String,
    pub sidecar: String,
    pub rowweights: Option<String>,
    /// The shard's loss weight (1: no weight file).
    pub weight: f64,
}

impl LabelInput {
    /// The file the shard is put in: `t<cut>-<set>-s<seed>.npy`, as a generated one would be
    /// named.
    pub fn file(&self, cut: u64) -> String {
        format!("t{cut}-{}-s{}.npy", self.set, self.seed)
    }

    pub fn to_json(&self) -> Value {
        let mut v = json!({ "set": self.set, "seed": self.seed, "shard": input_key(&self.shard),
                            "sidecar": input_key(&self.sidecar) });
        if let Some(r) = self.rows {
            v["rows"] = json!(r);
        }
        if self.weight != 1.0 {
            v["weight"] = json!(self.weight);
        }
        if let Some(w) = &self.rowweights {
            v["rowweights"] = json!(input_key(w));
        }
        v
    }

    /// The sidecar's `source` this set's shards have (`None`: any).
    fn source(&self) -> Option<&'static str> {
        match self.set {
            "random" | "val" => Some("random"),
            "rules" => Some("rules"),
            "harvest" | "race" => Some("harvest"),
            _ => None,
        }
    }
}

/// Reads `label_inputs` (empty when absent); `label_parts`: the rung's.
pub(crate) fn parse(o: &Map<String, Value>, label_parts: u64) -> Result<Vec<LabelInput>, String> {
    let list = match o.get("label_inputs") {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Array(a)) if !a.is_empty() => a,
        Some(_) => {
            return Err(
                "hybrid_rung: label_inputs must be a list of stored shards, each { \"set\", \
                 \"seed\", \"shard\", \"sidecar\" } (leave it out to generate every set)"
                    .into(),
            );
        }
    };
    if label_parts > 1 {
        return Err(
            "hybrid_rung: label_inputs with label_parts above 1: the reused shards are all there \
             at once, so nothing streams; use label_parts 1"
                .into(),
        );
    }
    let mut out: Vec<LabelInput> = Vec::new();
    for (i, e) in list.iter().enumerate() {
        let n = i + 1;
        let e = e
            .as_object()
            .ok_or_else(|| format!("hybrid_rung: label_inputs entry {n} is not an object"))?;
        if let Some(k) = e.keys().find(|k| !ENTRY_KEYS.contains(&k.as_str())) {
            return Err(format!(
                "hybrid_rung: label_inputs entry {n}: unknown key {k:?} (keys: {})",
                ENTRY_KEYS.join(", ")
            ));
        }
        let set = e
            .get("set")
            .and_then(Value::as_str)
            .and_then(|s| SETS.iter().find(|x| **x == s).copied())
            .ok_or_else(|| {
                format!(
                    "hybrid_rung: label_inputs entry {n}: set must be one of {}",
                    SETS.join(", ")
                )
            })?;
        let seed = e.get("seed").and_then(Value::as_u64).ok_or_else(|| {
            format!("hybrid_rung: label_inputs entry {n}: seed (the shard's, a whole number)")
        })?;
        if seed == GOLDEN_SEED {
            return Err(format!(
                "hybrid_rung: label_inputs entry {n}: seed {GOLDEN_SEED} is the golden sets'"
            ));
        }
        let key = |k: &str, need: bool| -> Result<Option<String>, String> {
            match e.get(k) {
                None | Some(Value::Null) if !need => Ok(None),
                Some(Value::String(v)) => input_hex(v).map(Some).ok_or_else(|| {
                    format!(
                        "hybrid_rung: label_inputs entry {n}: {k} {v:?} must be a stored input's \
                         key, \"inputs/sha256/<64 hex digits>\""
                    )
                }),
                _ => Err(format!(
                    "hybrid_rung: label_inputs entry {n}: {k} must be a stored input's key, \
                     \"inputs/sha256/<64 hex digits>\""
                )),
            }
        };
        let shard = key("shard", true)?.expect("needed");
        let sidecar = key("sidecar", true)?.expect("needed");
        let rowweights = key("rowweights", false)?;
        let rows = match e.get("rows") {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.as_u64().filter(|r| *r > 0).ok_or_else(|| {
                format!("hybrid_rung: label_inputs entry {n}: rows must be a whole number above 0")
            })?),
        };
        let weight = match e.get("weight") {
            None | Some(Value::Null) => 1.0,
            Some(v) => v
                .as_f64()
                .filter(|w| w.is_finite() && *w > 0.0 && *w <= 100.0)
                .ok_or_else(|| {
                    format!(
                        "hybrid_rung: label_inputs entry {n}: weight must be a number above 0 (at \
                         most 100)"
                    )
                })?,
        };
        if set == "val" && (weight != 1.0 || rowweights.is_some()) {
            return Err(format!(
                "hybrid_rung: label_inputs entry {n}: the validation shard takes no weight (its \
                 loss is the plain mean)"
            ));
        }
        if set == "val" && out.iter().any(|x| x.set == "val") {
            return Err("hybrid_rung: label_inputs: two validation shards".into());
        }
        if out.iter().any(|x| x.shard == shard) {
            return Err(format!(
                "hybrid_rung: label_inputs entry {n}: the same shard twice"
            ));
        }
        if out.iter().any(|x| x.set == set && x.seed == seed) {
            return Err(format!(
                "hybrid_rung: label_inputs entry {n}: two {set} shards with seed {seed}"
            ));
        }
        out.push(LabelInput {
            set,
            seed,
            rows,
            shard,
            sidecar,
            rowweights,
            weight,
        });
    }
    Ok(out)
}

/// The settings that make the set `set`, refused when `label_inputs` gives it (they would be
/// ignored).
pub(crate) fn generating_keys(set: &str) -> &'static [&'static str] {
    match set {
        "random" => &["labels_random", "labels_random_seed"],
        "rules" => &["labels_rules", "labels_rules_seed"],
        "harvest" => &["labels_harvest", "labels_harvest_seed"],
        "race" => &["labels_race", "labels_race_seed"],
        "val" => &["val_count", "val_seed"],
        _ => &[],
    }
}

/// A sidecar's checks against its entry ([`LabelInput`]) and the rung (`cut`); `rows`: the rows
/// the shard's header holds. The reason, if it does not fit.
pub(crate) fn check_sidecar(
    x: &LabelInput,
    side: &Value,
    cut: u64,
    rows: u64,
) -> Result<(), String> {
    let want = |k: &str, v: Value| -> Result<(), String> {
        if side[k] == v {
            Ok(())
        } else {
            Err(format!("its sidecar's {k} is {}, not {v}", side[k]))
        }
    };
    want("shard_sha256", json!(x.shard))?;
    want("format", json!("41bots-label-shard-v1"))?;
    want("encoder", json!(nn::encode::ENCODER))?;
    want("n_features", json!(nn::encode::N_FEATURES))?;
    want("trick", json!(cut))?;
    want("outcomes", json!(nn::outcome::vocab(cut as u8)))?;
    want("values_version", json!(search::measure::VALUES_VERSION))?;
    want("seed", json!(x.seed))?;
    if let Some(s) = x.source() {
        want("source", json!(s))?;
    }
    let side_rows = side["rows"].as_u64().or_else(|| side["count"].as_u64());
    if side_rows != Some(rows) {
        return Err(format!(
            "the shard holds {rows} rows, its sidecar says {side_rows:?}"
        ));
    }
    if x.rows.is_some_and(|r| r != rows) {
        return Err(format!(
            "the shard holds {rows} rows, the entry says {}",
            x.rows.unwrap_or(0)
        ));
    }
    Ok(())
}

/// The reused shards ([`Rung::reused_labels`]): the training ones in order, the validation one,
/// and their records for `rung.json`'s `labels`.
pub(super) type Reused = (Vec<PathBuf>, Option<PathBuf>, Vec<Value>);

impl Rung<'_> {
    /// The stored shards of `label_inputs`, fetched into `dir` and checked (module notes); nothing
    /// when there are none. Records `rung.json`'s `label_inputs`.
    pub(super) fn reused_labels(&mut self, dir: &Path) -> Result<Reused, String> {
        if self.p.label_inputs.is_empty() {
            return Ok((Vec::new(), None, Vec::new()));
        }
        let t = Instant::now();
        let list = self.p.label_inputs.clone();
        let progress = self.progress("label inputs", json!({ "shards": list.len() }));
        self.ctx.set_phase("chunk", progress);
        let remote = self.tools.remote.clone().ok_or(
            "hybrid_rung: label_inputs come from the remote store (R2, or RUNNER_REMOTE_DIR), and \
             there is none",
        )?;
        let (mut train, mut val, mut recs) = (Vec::new(), None, Vec::new());
        for x in &list {
            let t1 = Instant::now();
            let file = x.file(self.p.cut);
            let path = dir.join(&file);
            let with = |suffix: &str| PathBuf::from(format!("{}{suffix}", path.display()));
            let mut fetched = Vec::new();
            for (what, hex, local) in [
                ("shard", Some(x.shard.clone()), path.clone()),
                ("sidecar", Some(x.sidecar.clone()), with(".json")),
                ("rowweights", x.rowweights.clone(), with(".rowweights")),
            ] {
                let Some(hex) = hex else { continue };
                let (r, h, l) = (Arc::clone(&remote), hex.clone(), local.clone());
                let downloaded = self
                    .ctx
                    .blocking(move || fetch_input(&*r, &h, &l))
                    .map_err(|e| format!("hybrid_rung: label_inputs {file} ({what}): {e}"))?;
                fetched.push(
                    json!({ "what": what, "key": input_key(&hex), "downloaded": downloaded }),
                );
            }
            let bad = |e: String| {
                format!(
                    "hybrid_rung: label_inputs {} ({}, seed {}) does not fit this run: {e}",
                    input_key(&x.shard),
                    x.set,
                    x.seed
                )
            };
            let side =
                read_json(&with(".json")).ok_or_else(|| bad("its sidecar is not JSON".into()))?;
            let rows = nn::rowweights::shard_rows(&path).map_err(bad)? as u64;
            check_sidecar(x, &side, self.p.cut, rows).map_err(bad)?;
            if x.rowweights.is_some() {
                let w = nn::rowweights::read(&path)
                    .map_err(bad)?
                    .unwrap_or_default();
                if w.len() as u64 != rows {
                    return Err(bad(format!("{} row weights for {rows} rows", w.len())));
                }
            }
            let weight_file = with(".weight");
            if x.weight == 1.0 {
                let _ = fs::remove_file(&weight_file);
            } else {
                fs::write(&weight_file, format!("{}\n", x.weight))
                    .map_err(|e| format!("write {}: {e}", weight_file.display()))?;
            }
            write_json(&self.out.join("labels").join(format!("{file}.json")), &side)?;
            let mut rec = json!({ "name": x.set, "source": side["source"], "count": rows,
                                  "seed": x.seed, "file": file,
                                  "bytes": fs::metadata(&path).map_or(0, |m| m.len()),
                                  "shard_sha256": x.shard, "reused": x.to_json(),
                                  "fetched": fetched, "seconds": secs(t1) });
            if let Some(c) = side.get("rows_by_race_class") {
                rec["rows_by_race_class"] = c.clone();
            }
            if x.weight != 1.0 {
                rec["weight"] = json!(x.weight);
            }
            recs.push(rec);
            if x.set == "val" {
                val = Some(path);
            } else {
                train.push(path);
            }
        }
        println!(
            "runner: hybrid_rung: label inputs: {} stored shards checked ({} training{})",
            list.len(),
            train.len(),
            if val.is_some() {
                ", the validation shard"
            } else {
                ""
            }
        );
        self.rec.insert(
            "label_inputs".into(),
            json!({ "entries": list.iter().map(LabelInput::to_json).collect::<Vec<_>>(),
                    "seconds": secs(t) }),
        );
        self.rec.insert("labels".into(), json!(recs));
        self.done("label inputs", t)?;
        Ok((train, val, recs))
    }

    /// `upload_shards`: what a later run's `label_inputs` names this shard by (its set and seed
    /// from `rung.json`'s labels record `rec`, its row count from its sidecar), with its sidecar,
    /// its row weights and its weight file's number; the sidecar and the row weights are shared
    /// content-addressed here (the shard itself by the caller, as `sha`).
    pub(super) fn label_input_entry(
        &mut self,
        path: &Path,
        rec: &Value,
        sha: &str,
    ) -> Result<Value, String> {
        let remote = self
            .tools
            .remote
            .clone()
            .ok_or("hybrid_rung: no remote store")?;
        let with = |suffix: &str| PathBuf::from(format!("{}{suffix}", path.display()));
        let mut put = |p: PathBuf| -> Result<String, String> {
            let r = Arc::clone(&remote);
            self.ctx
                .blocking(move || crate::remote::put_input(&*r, &p, None))
                .map(|(h, _, _)| input_key(&h))
        };
        let sidecar = put(with(".json"))?;
        let rowweights = if with(".rowweights").is_file() {
            Some(put(with(".rowweights"))?)
        } else {
            None
        };
        let side = read_json(&with(".json")).unwrap_or(Value::Null);
        let mut e = json!({ "set": rec["name"], "seed": rec["seed"], "shard": input_key(sha),
                            "sidecar": sidecar });
        if let Some(r) = side["rows"].as_u64().or_else(|| side["count"].as_u64()) {
            e["rows"] = json!(r);
        }
        if let Some(w) = fs::read_to_string(with(".weight"))
            .ok()
            .and_then(|t| t.trim().parse::<f64>().ok())
            .filter(|w| *w != 1.0)
        {
            e["weight"] = json!(w);
        }
        if let Some(k) = rowweights {
            e["rowweights"] = json!(k);
        }
        Ok(e)
    }
}

impl Rung<'_> {
    /// `upload_shards` in the audit loop: a targeted harvest's shard (`path`, seed `seed`) shared
    /// content-addressed with its sidecar, weight and row weights, and its `label_inputs` entry
    /// (set `extra`) added to `dir/label_inputs.json`, so that a later run (an additive one)
    /// trains on it again without making it. A failure is only a warning.
    pub(super) fn share_targeted(&mut self, dir: &Path, path: &Path, seed: u64) {
        let Some(remote) = self.tools.remote.clone() else {
            return;
        };
        let side = read_json(&PathBuf::from(format!("{}.json", path.display())));
        let want = side
            .as_ref()
            .and_then(|s| s["shard_sha256"].as_str())
            .map(str::to_string);
        let p2 = path.to_path_buf();
        let shared = self
            .ctx
            .blocking(move || crate::remote::put_input(&*remote, &p2, want.as_deref()))
            .and_then(|(sha, _, _)| {
                self.label_input_entry(path, &json!({ "name": "extra", "seed": seed }), &sha)
            });
        match shared {
            Ok(e) => {
                let list_path = dir.join("label_inputs.json");
                let mut list = read_json(&list_path)
                    .and_then(|v| v.as_array().cloned())
                    .unwrap_or_default();
                list.push(e);
                if let Err(e) = write_json(&list_path, &Value::Array(list)) {
                    self.warn(format!("audit/label_inputs.json: {e}"));
                }
            }
            Err(e) => self.warn(format!(
                "upload_shards: the targeted shard {} was not shared: {e}",
                path.display()
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(c: char) -> String {
        format!("inputs/sha256/{}", c.to_string().repeat(64))
    }

    fn o(v: Value) -> Map<String, Value> {
        v.as_object().cloned().expect("object")
    }

    #[test]
    fn entries_are_read_and_checked() {
        assert_eq!(parse(&o(json!({})), 1), Ok(vec![]));
        let v = parse(
            &o(json!({ "label_inputs": [
                { "set": "random", "seed": 321, "rows": 10, "shard": key('a'), "sidecar": key('b') },
                { "set": "val", "seed": 401, "shard": key('c'), "sidecar": key('d') },
                { "set": "extra", "seed": 33151, "weight": 2, "shard": key('e'), "sidecar": key('f'),
                  "rowweights": key('0') } ] })),
            1,
        )
        .expect("entries");
        assert_eq!(v.len(), 3);
        assert_eq!(
            (v[0].set, v[0].seed, v[0].rows, v[0].weight),
            ("random", 321, Some(10), 1.0)
        );
        assert_eq!(v[1].file(3), "t3-val-s401.npy");
        assert_eq!(
            (v[2].weight, v[2].rowweights.as_deref()),
            (2.0, Some("0".repeat(64).as_str()))
        );
        assert_eq!(v[2].to_json()["rowweights"], key('0'));
        assert_eq!(v[0].to_json()["shard"], key('a'));
        let one = |e: Value| parse(&o(json!({ "label_inputs": [e] })), 1);
        for (e, want) in [
            (
                json!({ "set": "x", "seed": 3, "shard": key('a'), "sidecar": key('b') }),
                "set must be",
            ),
            (
                json!({ "set": "random", "shard": key('a'), "sidecar": key('b') }),
                "seed",
            ),
            (
                json!({ "set": "random", "seed": 1, "shard": key('a'), "sidecar": key('b') }),
                "golden",
            ),
            (
                json!({ "set": "random", "seed": 3, "sidecar": key('b') }),
                "shard must be",
            ),
            (
                json!({ "set": "random", "seed": 3, "shard": key('a') }),
                "sidecar must be",
            ),
            (
                json!({ "set": "random", "seed": 3, "shard": "a", "sidecar": key('b') }),
                "stored input's key",
            ),
            (
                json!({ "set": "random", "seed": 3, "shard": key('a'), "sidecar": key('b'), "rows": 0 }),
                "rows",
            ),
            (
                json!({ "set": "random", "seed": 3, "shard": key('a'), "sidecar": key('b'), "weight": 0 }),
                "weight",
            ),
            (
                json!({ "set": "val", "seed": 3, "shard": key('a'), "sidecar": key('b'), "weight": 2 }),
                "no weight",
            ),
            (
                json!({ "set": "random", "seed": 3, "shard": key('a'), "sidecar": key('b'), "note": 1 }),
                "unknown key",
            ),
            (json!("x"), "not an object"),
        ] {
            let err = one(e.clone()).expect_err("refused");
            assert!(err.contains(want), "{e}: {err}");
        }
        for (v, want) in [
            (json!([]), "must be a list"),
            (json!({}), "must be a list"),
            (
                json!([{ "set": "val", "seed": 3, "shard": key('a'), "sidecar": key('b') },
                    { "set": "val", "seed": 4, "shard": key('c'), "sidecar": key('d') }]),
                "two validation",
            ),
            (
                json!([{ "set": "random", "seed": 3, "shard": key('a'), "sidecar": key('b') },
                    { "set": "rules", "seed": 4, "shard": key('a'), "sidecar": key('d') }]),
                "the same shard",
            ),
            (
                json!([{ "set": "extra", "seed": 3, "shard": key('a'), "sidecar": key('b') },
                    { "set": "extra", "seed": 3, "shard": key('c'), "sidecar": key('d') }]),
                "two extra shards",
            ),
        ] {
            let err = parse(&o(json!({ "label_inputs": v.clone() })), 1).expect_err("refused");
            assert!(err.contains(want), "{v}: {err}");
        }
        let err = parse(
            &o(json!({ "label_inputs": [{ "set": "random", "seed": 3, "shard": key('a'), "sidecar": key('b') }] })),
            2,
        )
        .expect_err("parts");
        assert!(err.contains("label_parts"), "{err}");
    }

    #[test]
    fn sidecars_are_checked() {
        let x = LabelInput {
            set: "random",
            seed: 321,
            rows: Some(7),
            shard: "a".repeat(64),
            sidecar: "b".repeat(64),
            rowweights: None,
            weight: 1.0,
        };
        let good = json!({ "shard_sha256": "a".repeat(64), "format": "41bots-label-shard-v1",
                           "encoder": nn::encode::ENCODER, "n_features": nn::encode::N_FEATURES,
                           "trick": 3, "outcomes": nn::outcome::vocab(3),
                           "values_version": search::measure::VALUES_VERSION, "seed": 321,
                           "source": "random", "count": 7 });
        assert_eq!(check_sidecar(&x, &good, 3, 7), Ok(()));
        for (k, v, want) in [
            ("shard_sha256", json!("c".repeat(64)), "shard_sha256"),
            ("format", json!("other"), "format"),
            ("encoder", json!("oracle-v1"), "encoder"),
            ("trick", json!(4), "trick"),
            ("values_version", json!(1), "values_version"),
            ("seed", json!(322), "seed"),
            ("source", json!("rules"), "source"),
            ("count", json!(8), "sidecar says"),
        ] {
            let mut s = good.clone();
            s[k] = v;
            let e = check_sidecar(&x, &s, 3, 7).expect_err(k);
            assert!(e.contains(want), "{k}: {e}");
        }
        assert!(check_sidecar(&x, &good, 4, 7).is_err(), "another cut");
        // Trick 2 (the trick-2 loop's stored shards): its own trick and vocabulary.
        let mut t2 = good.clone();
        t2["trick"] = json!(2);
        t2["outcomes"] = json!(nn::outcome::vocab(2));
        assert_eq!(check_sidecar(&x, &t2, 2, 7), Ok(()));
        assert!(check_sidecar(&x, &t2, 3, 7).is_err());
        assert!(check_sidecar(&x, &good, 3, 6).is_err(), "the header's rows");
        let extra = LabelInput {
            set: "extra",
            rows: None,
            ..x.clone()
        };
        let mut s = good.clone();
        s["source"] = json!("harvest");
        s["rows"] = json!(7);
        assert_eq!(
            check_sidecar(&extra, &s, 3, 7),
            Ok(()),
            "any source; rows over count"
        );
        for set in ["random", "rules", "harvest", "race", "val"] {
            assert_eq!(generating_keys(set).len(), 2);
        }
        assert!(generating_keys("extra").is_empty());
    }
}
