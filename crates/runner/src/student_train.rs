//! The `student_train` chunk kind (docs/STUDENT.md §5, §7, §12 item 2): the student's learning
//! curve on a rented machine, from teacher shards already stored content-addressed
//! (`inputs/sha256/<hex>`, as the `student_teacher` chunks share them).
//!
//! ```json
//! { "name": "curve", "priority": 1, "kind": "student_train", "chunks": 1,
//!   "params": { "threads": "all",
//!     "train_inputs": [ { "run": "student-t2-base-1", "stream": "rules", "parts": 40 },
//!                       { "run": "student-t2-base-1", "stream": "mixed", "parts": 10 } ],
//!     "test_inputs": [ { "run": "student-t2-base-1", "stream": "test", "parts": 2 } ],
//!     "sizes": [50000, 100000, 250000], "widths": ["256,128", "512,256"],
//!     "device": "cuda" } }
//! ```
//!
//! **Inputs.** `train_inputs` and `test_inputs` are lists of entries of two forms:
//! - `{ "run": R, "stream": S, "parts": N }`: every part a finished `student_teacher` stream made,
//!   in chunk order. The chunk reads `runs/R/merged.json` (the run must be complete, the stream
//!   wholly done, and hold `parts` chunks when given), fetches each chunk's bundle listed there
//!   (SHA-256 and size checked) and takes from it the part's `student/<file>.inputs.json`: the
//!   records' and values' keys and sizes and the teacher's sidecar. A part whose shard was not
//!   shared (its upload failed, so its values are only in its bundle) stops the chunk. With
//!   `"bundles": [ { "path", "sha256", "bytes" }, … ]` the parts are those bundles, in the order
//!   listed, and `merged.json` is not read: a run stopped before its merged report (the stored
//!   bundles under `runs/R/out/S/`, each checked by SHA-256 and size as from a merged report).
//! - a stored shard named directly, as a part's `.inputs.json` gives it:
//!   `{ "records": "sha256:<hex>", "records_bytes": B, "values": "sha256:<hex>",
//!   "values_bytes": B, "sidecar": { … } }` (`part` and `file` may stay; nothing else).
//!
//! **Order and sizes.** The training parts are put in one order: `order` `"spread"` (the
//! default) interleaves the entries in proportion (part `i` of an entry of `n` parts at
//! `(i + ½) / n`, ties to the earlier entry: with 40 rules parts and 10 mixed, every five parts hold
//! four rules ones and one mixed one), `"given"` keeps them entry by entry. An entry may carry
//! `"group": k` (default 0): the groups come in ascending order, the order applied within each, so
//! a later run's parts can follow an earlier run's without changing the earlier prefix. Each size
//! trains on the first that many decisions in that order (the fewest parts that hold them, and the
//! trainer's `--limit`): a deterministic prefix, so the learning curve's data only grows. Only the
//! parts the largest size needs are fetched.
//!
//! **Points.** Either `sizes` (decisions, ascending, distinct) × `widths`, every target from all
//! the stored deals, or `points`: `[ { "size": N, "widths": "512,256", "deals": D }, … ]`, trained
//! in the order listed, `deals` optional. A point with `deals` trains on targets built from each
//! decision's first D deals only (`student encode --max-deals D`: exactly what a teacher run with
//! D deals would have stored), so the number of teacher deals a student needs is measured on data
//! already paid for; the test set always uses all its deals. Each deal count is encoded once into
//! its own feature shard set (`features/d<D>/`), over the parts its largest size needs; a point's
//! directory and resume key gain `-d<D>`.
//!
//! **Steps** (the heartbeat names each; `partial_cpu_ok`, as `hybrid_rung`: the GPU steps leave the
//! CPU idle by design):
//! 1. Equity table v0 (`spike::build_table_from`; the teacher's: `student encode` checks it).
//! 2. Inputs: entries resolved, every shard's role (`train` / `test`), trick and stamps checked
//!    against the first (`student::record::STAMPS`, but `allow_stamps`), no shard twice; written
//!    to `out/student/inputs.json`.
//! 3. Fetch and encode: each needed shard's two files fetched (`fetch_jobs` at a time; SHA-256 and
//!    size checked) and encoded once per deal count that needs it (`student encode`, encoder
//!    `student-v1`, `--max-deals` for a cut) into a feature shard of its own; the test parts into
//!    one test feature shard, never cut. A training part's raw files are
//!    deleted once encoded unless `keep_raw` (the test set's stay: `search student-check` reads
//!    them).
//! 4. The Python environment and the GPU check (`hybrid_rung`'s steps 2 and 3, the same groups,
//!    limits and records): `out/student/device.json`.
//! 5. For each point (sizes × widths: size outer, width inner): `python -m fortyone_nn.student` (the settings
//!    below, `--limit` the size, `--test` the test feature shard), then `search student-check` of
//!    the integer network on the test set with `--baseline` each of `baselines`, then the
//!    networks shared content-addressed (`upload`; a failure is a warning). Into
//!    `out/student/n<size>-w<w1>x<w2>[-d<deals>]/`: `float.safetensors`, `int.safetensors`, `report.json`
//!    (the trainer's), `check.json` and `check.txt` (the measure), and `done.json` last (the
//!    files' SHA-256, the keys, the parts used, the deals).
//! 6. `out/student/curve.json` and `curve.txt` (rewritten after every pair): per (size, width,
//!    deals; `null` / "all" for every deal) the test set's decision loss, its bootstrap 95% upper
//!    bound and the agreement, overall, by place in the trick and by score class (stratum S1–S4),
//!    beside the baselines.
//!
//! **Resume** at the point level (size, widths, deals): the runner restores a `student_train`
//! chunk's live files (`runs/<run>/partial/<stream>/<index>/out/`, uploaded every
//! `upload_every_s`) on the machine that continues it, and a point whose `done.json` is there
//! with every file it lists (by SHA-256) is not trained again; the data are fetched and encoded again (minutes). Float training on a GPU
//! is not bit-reproducible, so a pair redone gives another network; its integer file's SHA-256 is
//! what is recorded.
//!
//! Settings: `train_inputs`, `test_inputs`, and `sizes` with `widths` or `points` (required);
//! `order` (`spread`);
//! the trainer's `epochs` (40), `batch` (256), `lr` (0.001), `patience` (4), `seed` (1),
//! `val_permille` (50), `max_minutes` (0: none) as `fortyone_nn.student`'s defaults; `device`
//! (`auto`, `cuda` or `cpu`; default `auto`), `train_threads` (8 beside a GPU, else all);
//! `train_stall_minutes` (30: the trainer is killed when its progress file stops changing for
//! the larger of this and 3 epochs); `baselines` (`["rules", "random"]`), `boot` (1000);
//! `keep_raw` (false), `fetch_jobs` (4), `upload` (true), `allow_stamps` ([]); `table` and
//! `equity_input` as other kinds; any other key is refused ([`KEYS`]). One chunk a stream.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};

use crate::bundle::{read_member, sha256_file};
use crate::hybrid::{Device, Tools, choose_group, gpu_check, gpus_listed, python_command, uv_sync};
use crate::remote::{Remote, fetch_input, input_hex, input_key, put_input, retry};
use crate::spike::build_table_from;
use crate::{Ctx, sibling};

/// The settings a `student_train` chunk takes; any other key is refused.
pub(crate) const KEYS: [&str; 27] = [
    "train_inputs",
    "test_inputs",
    "order",
    "sizes",
    "widths",
    "points",
    "epochs",
    "batch",
    "lr",
    "patience",
    "seed",
    "val_permille",
    "max_minutes",
    "device",
    "train_threads",
    "train_stall_minutes",
    "baselines",
    "boot",
    "keep_raw",
    "fetch_jobs",
    "upload",
    "allow_stamps",
    "table",
    "equity_input",
    "threads",
    "est_seconds",
    "chunk_sizes",
];

/// The keys of a run entry and of a stored shard's entry.
const RUN_KEYS: [&str; 5] = ["run", "stream", "parts", "bundles", "group"];
const SHARD_KEYS: [&str; 8] = [
    "records",
    "records_bytes",
    "values",
    "values_bytes",
    "sidecar",
    "part",
    "file",
    "group",
];

/// The keys of a curve point and of a listed bundle.
const POINT_KEYS: [&str; 7] = [
    "size",
    "widths",
    "deals",
    "seed",
    "rank_weight",
    "rank_temperature",
    "stop_on",
];
const BUNDLE_KEYS: [&str; 3] = ["path", "sha256", "bytes"];

/// The baselines `search student-check` knows.
const BASELINES: [&str; 2] = ["rules", "random"];

/// One entry of `train_inputs` / `test_inputs`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Entry {
    /// Every part of a finished `student_teacher` stream (from its `merged.json`), or the parts
    /// whose bundles are listed (`bundles`: a run stopped before its merged report).
    Run {
        run: String,
        stream: String,
        parts: Option<u64>,
        bundles: Option<Vec<Bundle>>,
    },
    /// One stored shard.
    Shard(Shard),
}

/// A stored part bundle of a `student_teacher` stream, as a merged report lists it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Bundle {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

/// One point of the curve: a training size, the hidden widths and the teacher deals per
/// decision its targets are built from (`None`: every stored deal), and the trainer's variant
/// settings (all default: the run's own seed, the value loss alone, stopping on its validation
/// value).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Point {
    pub size: u64,
    pub widths: String,
    pub deals: Option<u32>,
    /// A seed of its own (`--seed`), instead of the run's.
    pub seed: Option<u64>,
    /// `--rank-weight` in thousandths (0: no decision-aware term).
    pub rank_milli: u64,
    /// `--rank-temperature` in millionths.
    pub rank_temp_micro: u64,
    /// `--stop-on decision`.
    pub stop_decision: bool,
}

/// The default `--rank-temperature`, in millionths (0.01).
pub(crate) const RANK_TEMP_MICRO: u64 = 10_000;

impl Point {
    /// A point with the run's own training settings.
    pub fn plain(size: u64, widths: String, deals: Option<u32>) -> Point {
        Point {
            size,
            widths,
            deals,
            seed: None,
            rank_milli: 0,
            rank_temp_micro: RANK_TEMP_MICRO,
            stop_decision: false,
        }
    }

    /// Its directory name, the resume key: `n<size>-w<w1>x<w2>…`, then `-d<deals>` when cut, `-s<seed>`
    /// for a seed of its own, `-r<weight>t<temperature>` with a rank term (thousandths, millionths),
    /// `-sd` when it stops on the decision loss.
    pub fn name(&self) -> String {
        let mut n = pair_name(self.size, &self.widths, self.deals);
        if let Some(s) = self.seed {
            n.push_str(&format!("-s{s}"));
        }
        if self.rank_milli > 0 {
            n.push_str(&format!("-r{}t{}", self.rank_milli, self.rank_temp_micro));
        }
        if self.stop_decision {
            n.push_str("-sd");
        }
        n
    }
}

/// A stored teacher shard: its two files by SHA-256 and size, and its sidecar.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Shard {
    pub records: String,
    pub records_bytes: u64,
    pub values: String,
    pub values_bytes: u64,
    pub sidecar: Value,
    /// Where it came from (`run/stream/part N`, or `entry N`), for messages and the record.
    pub label: String,
}

impl Shard {
    /// Decisions (records) in the shard, from its sidecar.
    pub fn decisions(&self) -> u64 {
        self.sidecar["records"].as_u64().unwrap_or(0)
    }

    fn to_json(&self) -> Value {
        json!({ "label": self.label, "records": format!("sha256:{}", self.records),
                "records_bytes": self.records_bytes, "values": format!("sha256:{}", self.values),
                "values_bytes": self.values_bytes, "decisions": self.decisions(),
                "role": self.sidecar["role"], "source": self.sidecar["source"],
                "gen_seed": self.sidecar["gen_seed"], "first": self.sidecar["first"],
                "deals": self.sidecar["deals"] })
    }
}

/// The order of the training parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Order {
    Spread,
    Given,
}

/// The checked settings.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Params {
    pub train: Vec<Entry>,
    /// The group of each training entry (0 unless given): groups in ascending order, the order
    /// applied within each.
    pub train_groups: Vec<u64>,
    pub test: Vec<Entry>,
    pub order: Order,
    /// The curve's points, in training order.
    pub points: Vec<Point>,
    pub epochs: u64,
    pub batch: u64,
    pub lr: f64,
    pub patience: u64,
    pub seed: u64,
    pub val_permille: u64,
    pub max_minutes: f64,
    pub device: Device,
    pub train_threads: Option<u64>,
    pub train_stall_minutes: f64,
    pub baselines: Vec<String>,
    pub boot: u64,
    pub keep_raw: bool,
    pub fetch_jobs: u64,
    pub upload: bool,
    pub allow_stamps: Vec<String>,
}

fn err(e: impl std::fmt::Display) -> String {
    format!("student_train: {e}")
}

fn parse_entries(p: &Value, key: &str) -> Result<(Vec<Entry>, Vec<u64>), String> {
    let list = match p.get(key) {
        Some(Value::Array(a)) if !a.is_empty() => a,
        _ => {
            return Err(err(format!(
                "{key} must be a list of entries: {{ \"run\", \"stream\" }} (every part of a \
                 finished student_teacher stream) or a stored shard {{ \"records\", \
                 \"records_bytes\", \"values\", \"values_bytes\", \"sidecar\" }}"
            )));
        }
    };
    let mut out = Vec::new();
    let mut groups = Vec::new();
    for (i, e) in list.iter().enumerate() {
        let n = i + 1;
        let o = e
            .as_object()
            .ok_or_else(|| err(format!("{key} entry {n} is not an object")))?;
        groups.push(match o.get("group") {
            None | Some(Value::Null) => 0,
            Some(v) => v
                .as_u64()
                .ok_or_else(|| err(format!("{key} entry {n}: group must be a whole number")))?,
        });
        if o.contains_key("run") {
            if let Some(k) = o.keys().find(|k| !RUN_KEYS.contains(&k.as_str())) {
                return Err(err(format!(
                    "{key} entry {n}: unknown key {k:?} (a run entry takes {})",
                    RUN_KEYS.join(", ")
                )));
            }
            let text = |k: &str| -> Result<String, String> {
                o.get(k)
                    .and_then(Value::as_str)
                    .filter(|s| {
                        !s.is_empty()
                            && s.bytes()
                                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
                    })
                    .map(str::to_string)
                    .ok_or_else(|| {
                        err(format!(
                            "{key} entry {n}: {k} must be a name (letters, digits, - _ .)"
                        ))
                    })
            };
            let parts = match o.get("parts") {
                None | Some(Value::Null) => None,
                Some(v) => Some(v.as_u64().filter(|x| *x > 0).ok_or_else(|| {
                    err(format!(
                        "{key} entry {n}: parts must be a whole number above 0"
                    ))
                })?),
            };
            let (run, stream) = (text("run")?, text("stream")?);
            let bundles = match o.get("bundles") {
                None | Some(Value::Null) => None,
                Some(v) => Some(parse_bundles(
                    v,
                    &run,
                    &stream,
                    &format!("{key} entry {n}"),
                )?),
            };
            if let (Some(b), Some(np)) = (&bundles, parts)
                && b.len() as u64 != np
            {
                return Err(err(format!(
                    "{key} entry {n}: {} bundles listed, parts says {np}",
                    b.len()
                )));
            }
            out.push(Entry::Run {
                run,
                stream,
                parts,
                bundles,
            });
        } else {
            if let Some(k) = o.keys().find(|k| !SHARD_KEYS.contains(&k.as_str())) {
                return Err(err(format!(
                    "{key} entry {n}: unknown key {k:?} (a stored shard takes {})",
                    SHARD_KEYS.join(", ")
                )));
            }
            let mut o = o.clone();
            o.remove("group");
            out.push(Entry::Shard(shard_of(&o, &format!("{key} entry {n}"))?));
        }
    }
    Ok((out, groups))
}

/// A run entry's `bundles`: a non-empty list of `{ "path", "sha256", "bytes" }`, each path one
/// of the run's stream's part bundles (`runs/<run>/out/<stream>/…`), no bundle twice.
fn parse_bundles(v: &Value, run: &str, stream: &str, label: &str) -> Result<Vec<Bundle>, String> {
    let list = v.as_array().filter(|a| !a.is_empty()).ok_or_else(|| {
        err(format!(
            "{label}: bundles must be a list of {{ \"path\", \"sha256\", \"bytes\" }}"
        ))
    })?;
    let prefix = format!("runs/{run}/out/{stream}/");
    let mut out: Vec<Bundle> = Vec::new();
    for (i, b) in list.iter().enumerate() {
        let what = format!("{label}: bundle {}", i + 1);
        let o = b
            .as_object()
            .ok_or_else(|| err(format!("{what} is not an object")))?;
        if let Some(k) = o.keys().find(|k| !BUNDLE_KEYS.contains(&k.as_str())) {
            return Err(err(format!(
                "{what}: unknown key {k:?} (a bundle takes {})",
                BUNDLE_KEYS.join(", ")
            )));
        }
        let path = o
            .get("path")
            .and_then(Value::as_str)
            .filter(|p| {
                p.strip_prefix(&prefix).is_some_and(|f| {
                    f.ends_with(".tar.zst")
                        && f.bytes()
                            .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
                })
            })
            .ok_or_else(|| err(format!("{what}: path must be {prefix}<file>.tar.zst")))?;
        let sha256 = o
            .get("sha256")
            .and_then(Value::as_str)
            .filter(|h| h.len() == 64 && h.bytes().all(|c| c.is_ascii_hexdigit()))
            .ok_or_else(|| err(format!("{what}: sha256 must be 64 hex digits")))?;
        let bytes = o
            .get("bytes")
            .and_then(Value::as_u64)
            .filter(|x| *x > 0)
            .ok_or_else(|| err(format!("{what}: bytes must be a whole number above 0")))?;
        if out.iter().any(|x| x.path == path || x.sha256 == sha256) {
            return Err(err(format!("{what}: the same bundle twice")));
        }
        out.push(Bundle {
            path: path.to_string(),
            sha256: sha256.to_ascii_lowercase(),
            bytes,
        });
    }
    Ok(out)
}

/// Checks one widths string ("512,256"); returns it without spaces.
fn check_widths(w: &str) -> Result<String, String> {
    let hidden: Vec<usize> = w
        .split(',')
        .map(|x| x.trim().parse::<usize>().ok().filter(|n| *n > 0))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| err(format!("widths {w:?}: hidden-layer sizes like \"256,128\"")))?;
    nn::mlp::check_hidden_widths(&hidden).map_err(|e| err(format!("widths {w:?}: {e}")))?;
    Ok(w.replace(' ', ""))
}

/// `points`: a non-empty list of `{ "size", "widths", "deals" }` (`deals` optional: every stored
/// deal), no point twice.
fn parse_points(v: &Value) -> Result<Vec<Point>, String> {
    let list = v.as_array().filter(|a| !a.is_empty()).ok_or_else(|| {
        err("points must be a list of { \"size\", \"widths\", \"deals\" (optional) }")
    })?;
    let mut out: Vec<Point> = Vec::new();
    for (i, x) in list.iter().enumerate() {
        let what = format!("points entry {}", i + 1);
        let o = x
            .as_object()
            .ok_or_else(|| err(format!("{what} is not an object")))?;
        if let Some(k) = o.keys().find(|k| !POINT_KEYS.contains(&k.as_str())) {
            return Err(err(format!(
                "{what}: unknown key {k:?} (a point takes {})",
                POINT_KEYS.join(", ")
            )));
        }
        let size = o
            .get("size")
            .and_then(Value::as_u64)
            .filter(|n| *n > 0)
            .ok_or_else(|| err(format!("{what}: size must be a whole number above 0")))?;
        let widths = check_widths(
            o.get("widths")
                .and_then(Value::as_str)
                .ok_or_else(|| err(format!("{what}: widths such as \"512,256\"")))?,
        )?;
        let deals = match o.get("deals") {
            None | Some(Value::Null) => None,
            Some(d) => Some(
                d.as_u64()
                    .filter(|n| (1..=u64::from(u32::MAX)).contains(n))
                    .ok_or_else(|| err(format!("{what}: deals must be a whole number above 0")))?
                    as u32,
            ),
        };
        let seed = match o.get("seed") {
            None | Some(Value::Null) => None,
            Some(d) => Some(
                d.as_u64()
                    .ok_or_else(|| err(format!("{what}: seed must be a whole number")))?,
            ),
        };
        // A decimal setting as a whole number of `unit`ths, refused when it is not one.
        let scaled = |k: &str, unit: f64, d: Option<f64>| -> Result<u64, String> {
            let Some(x) = o.get(k).filter(|v| !v.is_null()) else {
                return d
                    .map(|d| (d * unit).round() as u64)
                    .ok_or_else(|| err(format!("{what}: {k} is needed")));
            };
            let x = x
                .as_f64()
                .filter(|x| x.is_finite() && *x >= 0.0)
                .ok_or_else(|| err(format!("{what}: {k} must be a number, 0 or more")))?;
            let n = (x * unit).round();
            if (n / unit - x).abs() > 1e-9 {
                return Err(err(format!(
                    "{what}: {k} {x} is finer than 1/{unit} (it names the point's directory)"
                )));
            }
            Ok(n as u64)
        };
        let rank_milli = scaled("rank_weight", 1e3, Some(0.0))?;
        let rank_temp_micro = scaled("rank_temperature", 1e6, Some(0.01))?;
        if rank_temp_micro == 0 {
            return Err(err(format!("{what}: rank_temperature must be above 0")));
        }
        let stop_decision = match o.get("stop_on").and_then(Value::as_str) {
            None | Some("val_loss") => false,
            Some("decision") => true,
            Some(x) => {
                return Err(err(format!("{what}: stop_on {x:?} (val_loss or decision)")));
            }
        };
        let p = Point {
            size,
            widths,
            deals,
            seed,
            rank_milli,
            rank_temp_micro,
            stop_decision,
        };
        if out.contains(&p) {
            return Err(err(format!("{what}: the same point twice")));
        }
        out.push(p);
    }
    Ok(out)
}

/// A stored shard from a part's `.inputs.json` (or a manifest entry of that form), checked: its
/// keys, sizes and sidecar agree.
fn shard_of(o: &Map<String, Value>, label: &str) -> Result<Shard, String> {
    let key = |k: &str| -> Result<String, String> {
        o.get(k)
            .and_then(Value::as_str)
            .and_then(input_hex)
            .ok_or_else(|| err(format!("{label}: {k} must be \"sha256:<64 hex digits>\"")))
    };
    let bytes = |k: &str| -> Result<u64, String> {
        o.get(k)
            .and_then(Value::as_u64)
            .filter(|b| *b > 0)
            .ok_or_else(|| err(format!("{label}: {k} must be a whole number above 0")))
    };
    let s = Shard {
        records: key("records")?,
        records_bytes: bytes("records_bytes")?,
        values: key("values")?,
        values_bytes: bytes("values_bytes")?,
        sidecar: o
            .get("sidecar")
            .filter(|v| v.is_object())
            .cloned()
            .ok_or_else(|| {
                err(format!(
                    "{label}: sidecar (the shard's NAME.npy.json, as its .inputs.json gives it)"
                ))
            })?,
        label: label.to_string(),
    };
    let side = &s.sidecar;
    if side["format"] != "41bots-student-teacher-v1" {
        return Err(err(format!(
            "{label}: the sidecar is not a student teacher shard's ({})",
            side["format"]
        )));
    }
    if side["key"] != s.records.as_str()
        || side["values_sha256"] != s.values.as_str()
        || side["records_bytes"] != s.records_bytes
        || side["values_bytes"] != s.values_bytes
    {
        return Err(err(format!(
            "{label}: the keys and sizes do not match the sidecar's"
        )));
    }
    if s.decisions() == 0 {
        return Err(err(format!("{label}: the sidecar records no decisions")));
    }
    Ok(s)
}

/// Checks the settings (also when the manifest is read).
pub(crate) fn parse(p: &Value) -> Result<Params, String> {
    let o = p
        .as_object()
        .ok_or_else(|| err("params must be an object"))?;
    for k in o.keys() {
        if !KEYS.contains(&k.as_str()) {
            return Err(err(format!("unknown setting `{k}`")));
        }
    }
    let u = |k: &str, d: u64| -> Result<u64, String> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(v) => v
                .as_u64()
                .ok_or_else(|| err(format!("{k} must be a whole number"))),
        }
    };
    let f = |k: &str, d: f64| -> Result<f64, String> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(v) => v
                .as_f64()
                .filter(|x| x.is_finite() && *x >= 0.0)
                .ok_or_else(|| err(format!("{k} must be a number, at least 0"))),
        }
    };
    let b = |k: &str, d: bool| -> Result<bool, String> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(Value::Bool(x)) => Ok(*x),
            Some(_) => Err(err(format!("{k} must be true or false"))),
        }
    };
    let strings = |k: &str, d: &[&str]| -> Result<Vec<String>, String> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(d.iter().map(|s| s.to_string()).collect()),
            Some(Value::Array(a)) => a
                .iter()
                .map(|x| {
                    x.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| err(format!("{k} must be a list of strings")))
                })
                .collect(),
            Some(_) => Err(err(format!("{k} must be a list of strings"))),
        }
    };
    let (train, train_groups) = parse_entries(p, "train_inputs")?;
    let (test, test_groups) = parse_entries(p, "test_inputs")?;
    if test_groups.iter().any(|g| *g != 0) {
        return Err(err("test_inputs take no group"));
    }
    let order = match p.get("order").and_then(Value::as_str).unwrap_or("spread") {
        "spread" => Order::Spread,
        "given" => Order::Given,
        x => return Err(err(format!("order {x:?}: spread or given"))),
    };
    let points = match p.get("points") {
        Some(v) => {
            if p.get("sizes").is_some() || p.get("widths").is_some() {
                return Err(err(
                    "points or sizes and widths: the points list every (size, widths, deals)",
                ));
            }
            parse_points(v)?
        }
        None => {
            let sizes: Vec<u64> = match p.get("sizes") {
                Some(Value::Array(a)) if !a.is_empty() => a
                    .iter()
                    .map(|x| x.as_u64().filter(|n| *n > 0))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| err("sizes must be whole numbers above 0"))?,
                _ => {
                    return Err(err(
                        "sizes must be a list of decision counts (or give points)",
                    ));
                }
            };
            if sizes.windows(2).any(|w| w[0] >= w[1]) {
                return Err(err("sizes must be ascending and distinct"));
            }
            let widths = strings("widths", &[])?;
            if widths.is_empty() {
                return Err(err(
                    "widths must be a list such as [\"256,128\", \"512,256\"]",
                ));
            }
            let normal = widths
                .iter()
                .map(|w| check_widths(w))
                .collect::<Result<Vec<_>, _>>()?;
            if (1..normal.len()).any(|i| normal[..i].contains(&normal[i])) {
                return Err(err("widths: the same widths twice"));
            }
            sizes
                .iter()
                .flat_map(|&size| {
                    normal
                        .iter()
                        .map(move |w| Point::plain(size, w.clone(), None))
                })
                .collect()
        }
    };
    let device = match p.get("device").and_then(Value::as_str).unwrap_or("auto") {
        "auto" => Device::Auto,
        "cuda" => Device::Cuda,
        "cpu" => Device::Cpu,
        d => return Err(err(format!("device {d:?}: auto, cuda or cpu"))),
    };
    let baselines = strings("baselines", &BASELINES)?;
    if let Some(x) = baselines.iter().find(|x| !BASELINES.contains(&x.as_str())) {
        return Err(err(format!("baselines: {x:?} (rules, random)")));
    }
    let epochs = u("epochs", 40)?;
    let batch = u("batch", 256)?;
    if epochs == 0 || batch == 0 {
        return Err(err("epochs and batch must be at least 1"));
    }
    let lr = f("lr", 0.001)?;
    if lr <= 0.0 {
        return Err(err("lr must be above 0"));
    }
    let val_permille = u("val_permille", 50)?;
    if !(1..1000).contains(&val_permille) {
        return Err(err("val_permille must be 1 to 999"));
    }
    let train_threads = match p.get("train_threads") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_u64()
                .filter(|t| *t > 0)
                .ok_or_else(|| err("train_threads must be a whole number above 0"))?,
        ),
    };
    Ok(Params {
        train,
        train_groups,
        test,
        order,
        points,
        epochs,
        batch,
        lr,
        patience: u("patience", 4)?.max(1),
        seed: u("seed", 1)?,
        val_permille,
        max_minutes: f("max_minutes", 0.0)?,
        device,
        train_threads,
        train_stall_minutes: f("train_stall_minutes", 30.0)?.max(1.0),
        baselines,
        boot: u("boot", 1000)?,
        keep_raw: b("keep_raw", false)?,
        fetch_jobs: u("fetch_jobs", 4)?.clamp(1, 32),
        upload: b("upload", true)?,
        allow_stamps: strings("allow_stamps", &[])?,
    })
}

/// A point's directory name: `n<size>-w<w1>x<w2>…`, then `-d<deals>` when its targets come from
/// the first `deals` deals only (a run without `deals` keeps its old names).
pub(crate) fn pair_name(size: u64, widths: &str, deals: Option<u32>) -> String {
    let d = deals.map_or(String::new(), |d| format!("-d{d}"));
    format!("n{size}-w{}{d}", widths.replace(',', "x"))
}

/// The training parts in one order (`entries[k]`: the parts of entry `k`, in their order).
pub(crate) fn ordered(entries: Vec<Vec<Shard>>, order: Order) -> Vec<Shard> {
    match order {
        Order::Given => entries.into_iter().flatten().collect(),
        Order::Spread => {
            // Part i of an entry of n parts sits at (2i + 1) / 2n; compared exactly as
            // (2i + 1) · n' against (2i' + 1) · n, ties to the earlier entry.
            let mut keyed: Vec<(u64, u64, usize, Shard)> = Vec::new();
            for (k, parts) in entries.into_iter().enumerate() {
                let n = parts.len() as u64;
                for (i, s) in parts.into_iter().enumerate() {
                    keyed.push((2 * i as u64 + 1, n, k, s));
                }
            }
            keyed.sort_by(|a, b| {
                (u128::from(a.0) * u128::from(b.1))
                    .cmp(&(u128::from(b.0) * u128::from(a.1)))
                    .then(a.2.cmp(&b.2))
            });
            keyed.into_iter().map(|x| x.3).collect()
        }
    }
}

/// The training parts in one order by group: groups ascending (`groups[k]` is entry `k`'s), the
/// order within each group ([`ordered`]).
pub(crate) fn ordered_groups(entries: Vec<Vec<Shard>>, groups: &[u64], order: Order) -> Vec<Shard> {
    let group = |k: usize| groups.get(k).copied().unwrap_or(0);
    let mut keys: Vec<u64> = (0..entries.len()).map(group).collect();
    keys.sort_unstable();
    keys.dedup();
    let mut slots: Vec<Option<Vec<Shard>>> = entries.into_iter().map(Some).collect();
    let mut out = Vec::new();
    for g in keys {
        let members: Vec<Vec<Shard>> = (0..slots.len())
            .filter(|&k| group(k) == g)
            .filter_map(|k| slots[k].take())
            .collect();
        out.extend(ordered(members, order));
    }
    out
}

/// How many of the parts (in order, `decisions` each) the first `size` decisions need.
pub(crate) fn parts_for(decisions: &[u64], size: u64) -> Option<usize> {
    let mut total = 0;
    for (i, d) in decisions.iter().enumerate() {
        total += d;
        if total >= size {
            return Some(i + 1);
        }
    }
    None
}

/// Per deal count of the points (`None`: every deal), the encoded training parts (feature shard,
/// decisions), in training order.
type DealSets = BTreeMap<Option<u32>, Vec<(PathBuf, u64)>>;

/// Where the feature shards of one deal count go: `features/` for every deal (the old layout),
/// `features/d<N>/` for the first N.
fn deal_dir(feat: &Path, deals: Option<u32>) -> PathBuf {
    deals.map_or(feat.to_path_buf(), |d| feat.join(format!("d{d}")))
}

/// The programs a chunk runs (the release bundle's; fakes in the tests).
pub(crate) struct Programs {
    pub tools: Tools,
    pub student: PathBuf,
}

/// Run one `student_train` chunk.
pub(crate) fn chunk(ctx: &mut Ctx, p: &Value) -> Result<Value, String> {
    parse(p)?;
    let programs = Programs {
        tools: Tools::find()?,
        student: sibling("student"),
    };
    ctx.set_phase(
        "chunk",
        json!({ "kind": "student_train", "step": "setup: table v0", "partial_cpu_ok": true }),
    );
    let (table, table_rec) = build_table_from(ctx, p.get("table").unwrap_or(&json!({})), "auto")?;
    chunk_with(ctx, p, &programs, &table, table_rec)
}

/// The chunk's state as it runs.
struct Train<'a> {
    ctx: &'a mut Ctx,
    q: Params,
    progs: &'a Programs,
    remote: Arc<dyn Remote>,
    table: PathBuf,
    /// `out/student` (bundled; its files are uploaded live, so a resumed chunk finds its pairs).
    out: PathBuf,
    /// Outside `out/`: bundles, raw and feature shards, the Python environment.
    work: PathBuf,
    rec: Map<String, Value>,
    warnings: Vec<String>,
    t0: Instant,
}

/// [`chunk`] after the equity table, with the given programs.
pub(crate) fn chunk_with(
    ctx: &mut Ctx,
    p: &Value,
    progs: &Programs,
    table: &Path,
    table_rec: Value,
) -> Result<Value, String> {
    let q = parse(p)?;
    let remote = progs.tools.remote.clone().ok_or_else(|| {
        err("the teacher shards come from the remote store (R2, or RUNNER_REMOTE_DIR), and there is none")
    })?;
    let out = ctx.root.join("out").join("student");
    let work = ctx.root.join("student-train-work");
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    fs::create_dir_all(&work).map_err(|e| e.to_string())?;
    let mut rec = Map::new();
    rec.insert("table".into(), table_rec);
    let mut t = Train {
        ctx,
        q,
        progs,
        remote,
        table: table.to_path_buf(),
        out,
        work,
        rec,
        warnings: Vec::new(),
        t0: Instant::now(),
    };
    t.run()
}

fn write_json(p: &Path, v: &Value) -> Result<(), String> {
    if let Some(d) = p.parent() {
        fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(v).map_err(|e| e.to_string())? + "\n";
    let tmp = PathBuf::from(format!("{}.tmp", p.display()));
    fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    fs::rename(&tmp, p).map_err(|e| format!("{}: {e}", p.display()))
}

fn read_json(p: &Path) -> Result<Value, String> {
    serde_json::from_str(&fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?)
        .map_err(|e| format!("{}: {e}", p.display()))
}

fn secs(t: Instant) -> f64 {
    (t.elapsed().as_secs_f64() * 10.0).round() / 10.0
}

/// The local names of a shard's files: `<dir>/<records hex>.npy` (+ `.values.npy`, `.npy.json`).
fn raw_path(dir: &Path, s: &Shard) -> PathBuf {
    dir.join(format!("{}.npy", s.records))
}

/// Fetch a shard's files into `dir` (SHA-256 and sizes checked) and write its sidecar beside.
fn fetch_shard(r: &dyn Remote, s: &Shard, dir: &Path) -> Result<PathBuf, String> {
    let rp = raw_path(dir, s);
    let vp = student::record::values_path(&rp);
    for (hex, bytes, path) in [
        (&s.records, s.records_bytes, &rp),
        (&s.values, s.values_bytes, &vp),
    ] {
        fetch_input(r, hex, path).map_err(|e| format!("{}: {e}", s.label))?;
        let n = fs::metadata(path).map_err(|e| e.to_string())?.len();
        if n != bytes {
            let _ = fs::remove_file(path);
            return Err(format!(
                "{}: {} has {n} bytes, the record says {bytes}",
                s.label,
                input_key(hex)
            ));
        }
    }
    let text = serde_json::to_string_pretty(&s.sidecar).map_err(|e| e.to_string())? + "\n";
    fs::write(student::record::sidecar_path(&rp), text).map_err(|e| e.to_string())?;
    Ok(rp)
}

/// `student encode` of `teacher` shards into the feature shard `out` (its output to `log`).
fn encode(
    student: &Path,
    teacher: &[PathBuf],
    table: &Path,
    out: &Path,
    allow: &[String],
    max_deals: Option<u32>,
    log: &Path,
) -> Result<(), String> {
    let mut c = Command::new(student);
    c.arg("encode");
    for t in teacher {
        c.arg("--teacher").arg(t);
    }
    for a in allow {
        c.args(["--allow-stamp", a]);
    }
    if let Some(n) = max_deals {
        c.args(["--max-deals", &n.to_string()]);
    }
    c.arg("--equity").arg(table).arg("--out").arg(out);
    let o = c
        .output()
        .map_err(|e| format!("cannot run {}: {e}", student.display()))?;
    let mut text = o.stdout.clone();
    text.extend_from_slice(&o.stderr);
    if let Some(d) = log.parent() {
        let _ = fs::create_dir_all(d);
    }
    let _ = fs::write(log, &text);
    if !o.status.success() {
        return Err(format!(
            "student encode {}: {}",
            out.display(),
            String::from_utf8_lossy(&o.stderr).trim()
        ));
    }
    Ok(())
}

impl Train<'_> {
    fn progress(&self, step: &str, extra: Value) -> Value {
        let mut v = json!({ "kind": "student_train", "step": step, "partial_cpu_ok": true,
                            "warnings": self.warnings });
        if let (Some(o), Value::Object(e)) = (v.as_object_mut(), extra) {
            o.extend(e);
        }
        v
    }

    fn phase(&mut self, step: &str, extra: Value) {
        let p = self.progress(step, extra);
        self.ctx.set_phase("chunk", p);
    }

    fn warn(&mut self, w: String) {
        eprintln!("runner: WARNING: student_train: {w}");
        self.warnings.push(w);
    }

    fn save(&mut self) -> Result<(), String> {
        self.rec.insert("warnings".into(), json!(self.warnings));
        self.rec.insert("seconds".into(), json!(secs(self.t0)));
        write_json(
            &self.out.join("train.json"),
            &Value::Object(self.rec.clone()),
        )
    }

    /// A remote call beside the heartbeat.
    fn remote_call<T: Send>(
        &mut self,
        f: impl FnOnce(&dyn Remote) -> Result<T, String> + Send,
    ) -> Result<T, String> {
        let r = Arc::clone(&self.remote);
        self.ctx.blocking(move || f(&*r))
    }

    /// The parts of every entry (resolving runs through their merged reports and bundles).
    fn resolve(&mut self, entries: &[Entry], what: &str) -> Result<Vec<Vec<Shard>>, String> {
        let mut out = Vec::new();
        for (k, e) in entries.iter().enumerate() {
            match e {
                Entry::Shard(s) => out.push(vec![s.clone()]),
                Entry::Run {
                    run,
                    stream,
                    parts,
                    bundles,
                } => {
                    self.phase(
                        "inputs",
                        json!({ "set": what, "entry": k + 1, "run": run, "stream": stream }),
                    );
                    out.push(match bundles {
                        None => self.resolve_run(run, stream, *parts)?,
                        Some(b) => {
                            let dir = self.work.join("runs").join(run);
                            fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                            let list = b
                                .iter()
                                .enumerate()
                                .map(|(i, x)| (i as u64, x.clone()))
                                .collect::<Vec<_>>();
                            self.parts_of(run, stream, &dir, &list)?
                        }
                    });
                }
            }
        }
        Ok(out)
    }

    fn resolve_run(
        &mut self,
        run: &str,
        stream: &str,
        parts: Option<u64>,
    ) -> Result<Vec<Shard>, String> {
        let dir = self.work.join("runs").join(run);
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let merged_path = dir.join("merged.json");
        let key = format!("runs/{run}/merged.json");
        let (k2, m2) = (key.clone(), merged_path.clone());
        if !self.remote_call(move |r| retry(3, || r.get(&k2, &m2)))? {
            return Err(err(format!(
                "{key} not found: is {run} a finished run of this store?"
            )));
        }
        let merged = read_json(&merged_path)?;
        if merged["complete"] != true {
            return Err(err(format!("{key}: the run {run} is not complete")));
        }
        let s = merged["streams"]
            .as_array()
            .and_then(|a| a.iter().find(|s| s["name"] == stream))
            .ok_or_else(|| err(format!("{key}: no stream {stream:?}")))?;
        if s["kind"] != "student_teacher" {
            return Err(err(format!(
                "{run}/{stream} is a {} stream, not student_teacher",
                s["kind"]
            )));
        }
        if s["chunks_done"] != s["chunks_total"] {
            return Err(err(format!(
                "{run}/{stream}: {} of {} chunks done",
                s["chunks_done"], s["chunks_total"]
            )));
        }
        let mut chunks: Vec<Value> = s["chunks"].as_array().cloned().unwrap_or_default();
        chunks.sort_by_key(|c| c["index"].as_u64().unwrap_or(u64::MAX));
        if let Some(n) = parts
            && chunks.len() as u64 != n
        {
            return Err(err(format!(
                "{run}/{stream} holds {} parts, the manifest says {n}",
                chunks.len()
            )));
        }
        let mut list = Vec::new();
        for c in &chunks {
            let index = c["index"].as_u64().unwrap_or(0);
            let o = c["outputs"]
                .as_array()
                .and_then(|o| o.first())
                .ok_or_else(|| err(format!("{run}/{stream}/{index}: no bundle listed")))?;
            list.push((
                index,
                Bundle {
                    path: o["path"].as_str().unwrap_or_default().to_string(),
                    sha256: o["sha256"].as_str().unwrap_or_default().to_string(),
                    bytes: o["bytes"].as_u64().unwrap_or(0),
                },
            ));
        }
        self.parts_of(run, stream, &dir, &list)
    }

    /// The stored shards of a stream's part bundles (`(chunk index, bundle)`, in order): each
    /// bundle fetched and checked (SHA-256 and size), its part's `.inputs.json` read.
    fn parts_of(
        &mut self,
        run: &str,
        stream: &str,
        dir: &Path,
        list: &[(u64, Bundle)],
    ) -> Result<Vec<Shard>, String> {
        let mut out = Vec::new();
        for (index, b) in list {
            let (index, path, sha, bytes) = (*index, b.path.clone(), b.sha256.clone(), b.bytes);
            let local = dir.join(format!("{stream}-{index:06}.tar.zst"));
            let (p2, l2) = (path.clone(), local.clone());
            if !self.remote_call(move |r| retry(3, || r.get(&p2, &l2)))? {
                return Err(err(format!("{path} not found")));
            }
            let (h, n) = sha256_file(&local)?;
            if h != sha || n != bytes {
                return Err(err(format!(
                    "{path}: SHA-256 {h} and {n} bytes, its listing says {sha} and {bytes}"
                )));
            }
            let data = fs::read(&local).map_err(|e| e.to_string())?;
            let result = read_member(&data, "result.json")?
                .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
                .unwrap_or(Value::Null);
            let file = result["file"].as_str().unwrap_or_default();
            let name = format!("student/{file}.inputs.json");
            let inputs = read_member(&data, &name)?.ok_or_else(|| {
                err(format!(
                    "{run}/{stream} part {}: the bundle has no {name}: its shard was not shared \
                     content-addressed ({})",
                    result["part"], result["warning"]
                ))
            })?;
            let v: Value =
                serde_json::from_slice(&inputs).map_err(|e| err(format!("{name}: {e}")))?;
            let label = format!("{run}/{stream}/part {}", v["part"]);
            let obj = v
                .as_object()
                .ok_or_else(|| err(format!("{name}: not an object")))?;
            out.push(shard_of(obj, &label)?);
            let _ = fs::remove_file(&local);
        }
        Ok(out)
    }

    /// Steps 2: the inputs, resolved and checked; returns (training parts in order, test parts).
    fn inputs(&mut self) -> Result<(Vec<Shard>, Vec<Shard>), String> {
        let t = Instant::now();
        let train_entries = self.q.train.clone();
        let test_entries = self.q.test.clone();
        let groups = self.q.train_groups.clone();
        let train = ordered_groups(
            self.resolve(&train_entries, "train")?,
            &groups,
            self.q.order,
        );
        let test: Vec<Shard> = self
            .resolve(&test_entries, "test")?
            .into_iter()
            .flatten()
            .collect();
        let first = train
            .first()
            .ok_or_else(|| err("train_inputs give no parts"))?
            .clone();
        let mut seen = std::collections::BTreeSet::new();
        for (s, role) in train
            .iter()
            .map(|s| (s, "train"))
            .chain(test.iter().map(|s| (s, "test")))
        {
            if s.sidecar["role"] != role {
                return Err(err(format!(
                    "{}: role {}, but it is in the {role} inputs",
                    s.label, s.sidecar["role"]
                )));
            }
            if !seen.insert(s.records.clone()) {
                return Err(err(format!("{}: the same shard twice", s.label)));
            }
            for k in student::record::STAMPS {
                if self.q.allow_stamps.iter().any(|a| a == k) {
                    continue;
                }
                let (a, b) = (
                    student::record::stamp(&first.sidecar, k),
                    student::record::stamp(&s.sidecar, k),
                );
                if a != b {
                    return Err(err(format!(
                        "stamp {k}: {} has {a}, {} has {b} (allow it with allow_stamps)",
                        first.label, s.label
                    )));
                }
            }
        }
        let total: u64 = train.iter().map(Shard::decisions).sum();
        let largest = self.q.points.iter().map(|p| p.size).max().expect("points");
        if largest > total {
            return Err(err(format!(
                "size {largest}: the training inputs hold {total} decisions"
            )));
        }
        self.rec.insert(
            "inputs".into(),
            json!({ "train_decisions": total,
                    "test_decisions": test.iter().map(Shard::decisions).sum::<u64>(),
                    "train_parts": train.len(), "test_parts": test.len(),
                    "order": if self.q.order == Order::Spread { "spread" } else { "given" },
                    "teacher": first.sidecar["teacher"], "trick": first.sidecar["trick"],
                    "equity_sha256": first.sidecar["equity_sha256"], "seconds": secs(t) }),
        );
        write_json(
            &self.out.join("inputs.json"),
            &json!({ "train": train.iter().map(Shard::to_json).collect::<Vec<_>>(),
                     "test": test.iter().map(Shard::to_json).collect::<Vec<_>>() }),
        )?;
        self.save()?;
        Ok((train, test))
    }

    /// Step 3: fetch and encode; returns (per deal count of the points, the feature shards of the
    /// training parts its largest size needs, with their decisions; the test feature shard; the
    /// test set's raw records files). A part is fetched once and encoded once per deal count that
    /// needs it (`student encode --max-deals`, `None`: every deal), then its raw files deleted.
    #[allow(clippy::type_complexity)]
    fn fetch_encode(
        &mut self,
        train: &[Shard],
        test: &[Shard],
    ) -> Result<(DealSets, PathBuf, Vec<PathBuf>), String> {
        let t = Instant::now();
        let raw = self.work.join("raw");
        let feat = self.work.join("features");
        let logs = self.ctx.logs.join("student-train");
        fs::create_dir_all(&raw).map_err(|e| e.to_string())?;
        fs::create_dir_all(&feat).map_err(|e| e.to_string())?;
        let decisions: Vec<u64> = train.iter().map(Shard::decisions).collect();
        // Per deal count, the parts its largest size needs.
        let mut needs: BTreeMap<Option<u32>, usize> = BTreeMap::new();
        for p in &self.q.points {
            let n = parts_for(&decisions, p.size).expect("checked in inputs");
            let e = needs.entry(p.deals).or_insert(0);
            *e = (*e).max(n);
        }
        let need = needs.values().copied().max().expect("points");
        let largest = self.q.points.iter().map(|p| p.size).max().expect("points");
        for d in needs.keys() {
            fs::create_dir_all(deal_dir(&feat, *d)).map_err(|e| e.to_string())?;
        }
        // The test set first (it stays raw for the check, and is never cut), then the training
        // parts in batches of `fetch_jobs`, each part fetched and encoded on its own thread.
        let mut test_raw = Vec::new();
        for (i, s) in test.iter().enumerate() {
            self.phase(
                "fetch test set",
                json!({ "part": i + 1, "parts": test.len() }),
            );
            let s2 = s.clone();
            let d2 = raw.clone();
            test_raw.push(self.remote_call(move |r| fetch_shard(r, &s2, &d2))?);
        }
        let test_feat = feat.join("test.npy");
        self.phase("encode test set", json!({ "parts": test.len() }));
        {
            let (st, tr, tb, tf, al, lg) = (
                self.progs.student.clone(),
                test_raw.clone(),
                self.table.clone(),
                test_feat.clone(),
                self.q.allow_stamps.clone(),
                logs.join("encode-test.log"),
            );
            self.ctx
                .blocking(move || encode(&st, &tr, &tb, &tf, &al, None, &lg))
                .map_err(err)?;
        }
        let mut sets: DealSets = needs.keys().map(|d| (*d, Vec::new())).collect();
        let jobs = self.q.fetch_jobs as usize;
        let mut downloaded = 0u64;
        let mut fetched = 0usize;
        for (b, batch) in train[..need].chunks(jobs).enumerate() {
            self.phase(
                "fetch and encode",
                json!({ "parts_done": fetched, "parts": need,
                        "decisions_needed": largest, "deal_sets": needs.len() }),
            );
            let r = Arc::clone(&self.remote);
            let (st, tb, al, keep) = (
                self.progs.student.clone(),
                self.table.clone(),
                self.q.allow_stamps.clone(),
                self.q.keep_raw,
            );
            let (raw2, feat2, logs2) = (raw.clone(), feat.clone(), logs.clone());
            let batch: Vec<(usize, Shard)> = batch
                .iter()
                .enumerate()
                .map(|(i, s)| (b * jobs + i, s.clone()))
                .collect();
            let needs2 = needs.clone();
            let bytes: u64 = batch
                .iter()
                .map(|(_, s)| s.records_bytes + s.values_bytes)
                .sum();
            type Encoded = Vec<(Option<u32>, PathBuf)>;
            let done: Vec<Result<Encoded, String>> = self.ctx.blocking(move || {
                std::thread::scope(|sc| {
                    let hs: Vec<_> = batch
                        .iter()
                        .map(|(k, s)| {
                            let (r, st, tb, al) = (Arc::clone(&r), &st, &tb, &al);
                            let (raw2, feat2, logs2, needs2) = (&raw2, &feat2, &logs2, &needs2);
                            sc.spawn(move || -> Result<Encoded, String> {
                                let rp = fetch_shard(&*r, s, raw2)?;
                                let mut out = Vec::new();
                                for (&d, &n) in needs2 {
                                    if *k >= n {
                                        continue;
                                    }
                                    let f = deal_dir(feat2, d).join(format!("{}.npy", s.records));
                                    let tag = d.map_or(String::new(), |d| format!("-d{d}"));
                                    let log =
                                        logs2.join(format!("encode-{}{tag}.log", &s.records[..16]));
                                    encode(st, std::slice::from_ref(&rp), tb, &f, al, d, &log)?;
                                    out.push((d, f));
                                }
                                if !keep {
                                    let _ = fs::remove_file(&rp);
                                    let _ = fs::remove_file(student::record::values_path(&rp));
                                    let _ = fs::remove_file(student::record::sidecar_path(&rp));
                                }
                                Ok(out)
                            })
                        })
                        .collect();
                    hs.into_iter()
                        .map(|h| {
                            h.join()
                                .unwrap_or_else(|_| Err("a fetch thread panicked".into()))
                        })
                        .collect()
                })
            });
            for d in done {
                for (deals, f) in d.map_err(err)? {
                    let side = read_json(&PathBuf::from(format!("{}.json", f.display())))?;
                    if side["role"] != "train" {
                        return Err(err(format!(
                            "{}: a training part encoded with role {}",
                            f.display(),
                            side["role"]
                        )));
                    }
                    if side["max_deals"] != json!(deals) {
                        return Err(err(format!(
                            "{}: encoded with max_deals {}, the point asks {deals:?}",
                            f.display(),
                            side["max_deals"]
                        )));
                    }
                    let n = side["records"].as_u64().unwrap_or(0);
                    sets.get_mut(&deals).expect("a deal set").push((f, n));
                }
                fetched += 1;
            }
            downloaded += bytes;
        }
        let tside = read_json(&PathBuf::from(format!("{}.json", test_feat.display())))?;
        if tside["role"] != "test" {
            return Err(err("the test set's feature shard is not role test"));
        }
        self.rec.insert(
            "data".into(),
            json!({ "parts_used": need, "training_bytes_fetched": downloaded,
                    "test_decisions": tside["records"], "encoder": tside["encoder"],
                    "encoder_sha256": tside["encoder_sha256"],
                    "deal_sets": sets.iter().map(|(d, f)| json!({
                        "deals": d, "parts": f.len(),
                        "train_feature_decisions": f.iter().map(|x| x.1).sum::<u64>() }))
                        .collect::<Vec<_>>(),
                    "keep_raw": self.q.keep_raw, "seconds": secs(t) }),
        );
        self.save()?;
        Ok((sets, test_feat, test_raw))
    }

    /// Step 4: the Python environment and the GPU check; returns (Python, device).
    fn python(&mut self) -> Result<(PathBuf, String), String> {
        let t = Instant::now();
        let gpus = gpus_listed(&self.progs.tools.nvidia_smi);
        let choice = choose_group(self.q.device, &gpus)
            .map_err(|e| e.replace("hybrid_rung", "student_train"))?;
        for w in choice.warnings.clone() {
            self.warn(w);
        }
        let env_dir = self.work.join(format!("pyenv-{}", choice.group));
        self.phase(
            "python environment",
            json!({ "group": choice.group, "compute_cap": choice.compute_cap, "gpus": gpus }),
        );
        let (python, uv_version, attempts) = uv_sync(
            self.ctx,
            &self.progs.tools,
            choice.group,
            &env_dir,
            "student-train/uv-sync.log",
        )
        .map_err(err)?;
        let lock = sha256_file(&self.progs.tools.python_dir.join("uv.lock"))?.0;
        self.rec.insert(
            "python_env".into(),
            json!({ "uv": uv_version, "group": choice.group, "compute_cap": choice.compute_cap,
                    "uv_lock_sha256": lock, "attempts": attempts, "nvidia_smi": gpus,
                    "seconds": secs(t) }),
        );
        self.phase("GPU check", json!({}));
        let c = python_command(&python, &self.progs.tools.python_dir, "fortyone_nn.device");
        let facts = gpu_check(
            self.ctx,
            c,
            self.q.device,
            &self.out.join("device.json"),
            "student-train/device.log",
        )
        .map_err(err)?;
        let device = facts["device"].as_str().unwrap_or("cpu").to_string();
        if device != "cuda" && self.q.device == Device::Auto {
            self.warn(format!(
                "no usable GPU ({}): training on the CPU",
                facts["reason"].as_str().unwrap_or("no reason recorded")
            ));
        }
        self.rec.insert("device".into(), facts);
        self.save()?;
        Ok((python, device))
    }

    fn train_threads(&self, device: &str) -> u64 {
        let all = self.ctx.threads as u64;
        self.q
            .train_threads
            .unwrap_or(if device == "cuda" {
                all.min(crate::hybrid::GPU_TRAIN_THREADS)
            } else {
                all
            })
            .max(1)
    }

    /// Whether `dir` holds a finished pair: its `done.json` and every file it lists, by SHA-256.
    fn finished(dir: &Path) -> Option<Value> {
        let done = read_json(&dir.join("done.json")).ok()?;
        let files = done["files"].as_object()?;
        for (name, sha) in files {
            let (h, _) = sha256_file(&dir.join(name)).ok()?;
            if sha.as_str() != Some(h.as_str()) {
                return None;
            }
        }
        Some(done)
    }

    /// Shares a network content-addressed; its key, or `None` with a warning.
    fn share(&mut self, path: &Path, what: &str) -> Option<String> {
        if !self.q.upload {
            return None;
        }
        let p2 = path.to_path_buf();
        match self.remote_call(move |r| put_input(r, &p2, None)) {
            Ok((sha, bytes, status)) => {
                println!(
                    "runner: student_train: {what}: {} ({bytes} bytes, {status})",
                    input_key(&sha)
                );
                Some(input_key(&sha))
            }
            Err(e) => {
                self.warn(format!(
                    "{what} was not shared as an input: {e}; it is in this run's bundle"
                ));
                None
            }
        }
    }

    /// Step 5 for one point: train, export, check, share; returns its `done.json`.
    #[allow(clippy::too_many_arguments)]
    fn pair(
        &mut self,
        point: &Point,
        feats: &[(PathBuf, u64)],
        test_feat: &Path,
        test_raw: &[PathBuf],
        python: &Path,
        device: &str,
    ) -> Result<Value, String> {
        let t = Instant::now();
        let (size, widths, deals) = (point.size, point.widths.as_str(), point.deals);
        let name = point.name();
        let dest = self.out.join(&name);
        let decisions: Vec<u64> = feats.iter().map(|f| f.1).collect();
        let n = parts_for(&decisions, size).ok_or_else(|| {
            err(format!(
                "size {size}: the encoded parts hold {} decisions",
                decisions.iter().sum::<u64>()
            ))
        })?;
        let dir = self.work.join("train").join(&name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let prog = dir.join("progress.json");
        let threads = self.train_threads(device);
        let mut c = python_command(python, &self.progs.tools.python_dir, "fortyone_nn.student");
        c.arg("--train");
        for (f, _) in &feats[..n] {
            c.arg(f);
        }
        c.args(["--limit", &size.to_string()])
            .arg("--test")
            .arg(test_feat)
            .arg("--out")
            .arg(&dir)
            .args(["--widths", widths])
            .args(["--epochs", &self.q.epochs.to_string()])
            .args(["--batch", &self.q.batch.to_string()])
            .args(["--lr", &self.q.lr.to_string()])
            .args(["--patience", &self.q.patience.to_string()])
            .args(["--seed", &point.seed.unwrap_or(self.q.seed).to_string()])
            .args(["--val-permille", &self.q.val_permille.to_string()])
            .args(["--threads", &threads.to_string()])
            .args(["--device", device])
            .args(["--max-minutes", &self.q.max_minutes.to_string()])
            .arg("--progress")
            .arg(&prog);
        if point.rank_milli > 0 {
            c.args([
                "--rank-weight",
                &(point.rank_milli as f64 / 1e3).to_string(),
            ])
            .args([
                "--rank-temperature",
                &(point.rank_temp_micro as f64 / 1e6).to_string(),
            ]);
        }
        if point.stop_decision {
            c.args(["--stop-on", "decision"]);
        }
        let base = self.progress(
            "train",
            json!({ "size": size, "widths": widths, "deals": deals, "parts": n,
                    "device": device, "train_threads": threads }),
        );
        self.ctx.set_phase("chunk", base.clone());
        let stall = crate::Stall::new(
            prog.clone(),
            Duration::from_secs_f64(self.q.train_stall_minutes * 60.0),
            3,
        );
        self.ctx.run_child_stall(
            c,
            &format!("student-train/{name}.train.log"),
            stall,
            &mut |ctx| {
                let mut p = base.clone();
                if let Ok(text) = fs::read_to_string(&prog)
                    && let Ok(v) = serde_json::from_str::<Value>(&text)
                {
                    p["train"] = v;
                }
                ctx.progress = p;
            },
        )?;
        let train_secs = secs(t);
        // The measure: `search student-check` of the integer network on the test set.
        self.phase(
            "check",
            json!({ "size": size, "widths": widths, "deals": deals }),
        );
        let int = dir.join("int.safetensors");
        let mut c = Command::new(&self.progs.tools.search);
        c.arg("student-check").arg("--net").arg(&int);
        for t in test_raw {
            c.arg("--teacher").arg(t);
        }
        for a in &self.q.allow_stamps {
            c.args(["--allow-stamp", a]);
        }
        c.arg("--equity").arg(&self.table);
        for b in &self.q.baselines {
            c.args(["--baseline", b]);
        }
        c.args(["--boot", &self.q.boot.to_string()])
            .arg("--json")
            .arg(dir.join("check.json"));
        self.ctx.run_child_io(
            c,
            &format!("student-train/{name}.check.log"),
            Some(&dir.join("check.txt")),
            None,
            &mut |_| {},
        )?;
        let check = read_json(&dir.join("check.json"))?;
        let report = read_json(&dir.join("report.json"))?;
        // The trainer's own integer number on the test set must be the check's (docs/STUDENT.md
        // §11: both ways the same); a difference is said, not hidden.
        let ours = report["test"]["int"]["all"]["decision_loss_pct"].as_f64();
        let theirs = check["rows"][0]["mean_loss_pct"].as_f64();
        if let (Some(a), Some(b)) = (ours, theirs)
            && (a - b).abs() > 1e-6
        {
            self.warn(format!(
                "{name}: the trainer's integer test loss {a:.6}% differs from the check's {b:.6}%"
            ));
        }
        let int_key = self.share(&int, &format!("{name} integer network"));
        let float_key = self.share(
            &dir.join("float.safetensors"),
            &format!("{name} float weights"),
        );
        // Into out/: the files, then done.json last.
        let _ = fs::remove_dir_all(&dest);
        fs::create_dir_all(&dest).map_err(|e| e.to_string())?;
        let mut files = Map::new();
        for f in [
            "float.safetensors",
            "int.safetensors",
            "report.json",
            "check.json",
            "check.txt",
        ] {
            fs::copy(dir.join(f), dest.join(f)).map_err(|e| format!("{f}: {e}"))?;
            files.insert(f.into(), json!(sha256_file(&dest.join(f))?.0));
        }
        let done = json!({
            "size": size, "widths": widths, "deals": deals, "parts": n, "variant": name,
            "seed": point.seed, "rank_weight": point.rank_milli as f64 / 1e3,
            "rank_temperature": point.rank_temp_micro as f64 / 1e6,
            "stop_on": if point.stop_decision { "decision" } else { "val_loss" },
            "part_files": feats[..n].iter().map(|(f, d)| json!({
                "features": f.file_name().map(|x| x.to_string_lossy().into_owned()),
                "decisions": d })).collect::<Vec<_>>(),
            "device": device, "train_threads": threads,
            "int_sha256": files["int.safetensors"], "float_sha256": files["float.safetensors"],
            "int_key": int_key, "float_key": float_key,
            "train_seconds": train_secs, "seconds": secs(t),
            "files": files,
        });
        write_json(&dest.join("done.json"), &done)?;
        Ok(done)
    }

    fn run(&mut self) -> Result<Value, String> {
        let (train, test) = self.inputs()?;
        let (sets, test_feat, test_raw) = self.fetch_encode(&train, &test)?;
        let (python, device) = self.python()?;
        let mut pairs = Vec::new();
        for point in self.q.points.clone() {
            let dir = self.out.join(point.name());
            let done = match Self::finished(&dir) {
                Some(d) => {
                    println!(
                        "runner: student_train: {} was finished before (resumed): not trained again",
                        point.name()
                    );
                    d
                }
                None => {
                    let feats = &sets[&point.deals];
                    self.pair(&point, feats, &test_feat, &test_raw, &python, &device)?
                }
            };
            pairs.push(done);
            self.rec.insert("pairs".into(), json!(pairs));
            self.save()?;
            write_curve(&self.out)?;
        }
        let curve = write_curve(&self.out)?;
        self.save()?;
        Ok(
            json!({ "kind": "student_train", "pairs": pairs.len(), "device": device,
                   "curve": curve["rows"], "baselines": curve["baselines"],
                   "warnings": self.warnings, "seconds": secs(self.t0) }),
        )
    }
}

/// A row of a `student-check` table, by group name (`all`, `place 3`, `stratum S2`).
fn group_row<'a>(rows: &'a Value, group: &str) -> Option<&'a Value> {
    rows.as_array()?.iter().find(|r| r["group"] == group)
}

fn brief(r: Option<&Value>) -> Value {
    r.map_or(Value::Null, |r| {
        json!({ "n": r["n"], "loss_pct": r["mean_loss_pct"], "upper95_pct": r["upper95_pct"],
                "agreement": r["agreement"] })
    })
}

/// The groups of the curve's breakdown: every place in the trick, every score class.
fn groups() -> Vec<String> {
    let mut g: Vec<String> = (1..=8).map(|p| format!("place {p}")).collect();
    g.extend((1..=4).map(|s| format!("stratum S{s}")));
    g
}

/// Writes `curve.json` and `curve.txt` from the finished pairs under `out` (every
/// `n<size>-w<…>/done.json` with its `check.json` and `report.json`), in size then width order.
pub(crate) fn write_curve(out: &Path) -> Result<Value, String> {
    let mut found: Vec<(u64, String, u64, PathBuf)> = Vec::new();
    for e in fs::read_dir(out).map_err(|e| e.to_string())? {
        let p = e.map_err(|e| e.to_string())?.path();
        if let Some(d) = Train::finished(&p) {
            found.push((
                d["size"].as_u64().unwrap_or(0),
                d["widths"].as_str().unwrap_or_default().to_string(),
                d["deals"].as_u64().unwrap_or(u64::MAX),
                p,
            ));
        }
    }
    found.sort_by(|a, b| (a.0, &a.1, a.2, &a.3).cmp(&(b.0, &b.1, b.2, &b.3)));
    let mut rows = Vec::new();
    let mut baselines = Value::Null;
    for (size, widths, deals, dir) in &found {
        let check = read_json(&dir.join("check.json"))?;
        let report = read_json(&dir.join("report.json"))?;
        let done = read_json(&dir.join("done.json"))?;
        if baselines.is_null() {
            baselines = json!(
                check["baselines"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|b| {
                        let mut by = Map::new();
                        for g in groups() {
                            by.insert(g.clone(), brief(group_row(&b["rows"], &g)));
                        }
                        json!({ "bot": b["bot"], "all": brief(group_row(&b["rows"], "all")),
                            "expected_loss_pct": b["expected_loss_pct"], "by_group": by })
                    })
                    .collect::<Vec<_>>()
            );
        }
        let mut by = Map::new();
        for g in groups() {
            by.insert(g.clone(), brief(group_row(&check["rows"], &g)));
        }
        rows.push(json!({
            "size": size, "widths": widths, "variant": done["variant"],
            "deals": if *deals == u64::MAX { Value::Null } else { json!(deals) },
            "train_decisions": report["data"]["train_rows"], "val_decisions": report["data"]["val_rows"],
            "epochs_run": report["history"].as_array().map_or(0, Vec::len),
            "best_epoch": report["best_epoch"],
            "train_minutes": report["seconds"].as_f64().map(|s| s / 60.0),
            "device": done["device"],
            "val": brief_report(&report["val"]["int"]["all"]),
            "test": brief(group_row(&check["rows"], "all")),
            "test_value_mae_pct": group_row(&check["rows"], "all").map(|r| r["value_mae_pct"].clone()),
            "by_group": by,
            "int_sha256": done["int_sha256"], "int_key": done["int_key"],
            "float_sha256": done["float_sha256"], "float_key": done["float_key"],
            "dir": dir.file_name().map(|x| x.to_string_lossy().into_owned()),
        }));
    }
    let curve = json!({ "rows": rows, "baselines": baselines });
    write_json(&out.join("curve.json"), &curve)?;
    fs::write(out.join("curve.txt"), curve_text(&curve)).map_err(|e| e.to_string())?;
    Ok(curve)
}

fn brief_report(r: &Value) -> Value {
    json!({ "n": r["decisions"], "loss_pct": r["decision_loss_pct"], "agreement": r["agreement"] })
}

/// What a point's directory name adds to `n<size>-w<…>[-d<deals>]` (a seed, a rank term, the stop
/// rule), for the curve's player column; empty for a plain point.
fn variant_tag(r: &Value) -> String {
    let v = r["variant"].as_str().unwrap_or_default();
    let tail: Vec<&str> = v
        .split('-')
        .filter(|x| {
            (x.starts_with('s') && x[1..].chars().all(|c| c.is_ascii_digit()) && x.len() > 1)
                || x.starts_with('r') && x.contains('t')
                || *x == "sd"
        })
        .collect();
    if tail.is_empty() {
        String::new()
    } else {
        format!(" {}", tail.join(" "))
    }
}

/// The curve as text: the overall table, then the loss by place and by score class.
pub(crate) fn curve_text(curve: &Value) -> String {
    let num = |v: &Value| v.as_f64().map_or("-".to_string(), |x| format!("{x:.4}"));
    let pct = |v: &Value| {
        v.as_f64()
            .map_or("-".to_string(), |x| format!("{:.1}", 100.0 * x))
    };
    let mut s = String::from(
        "Student learning curve on the fixed test set (decision loss: the teacher's best mean less \
         the mean of the player's choice, % of the deciding pair's money chance; upper: bootstrap \
         95% bound; agree: % of decisions on the teacher's best)\n\n",
    );
    let deals = |r: &Value| {
        r["deals"]
            .as_u64()
            .map_or("all".to_string(), |d| d.to_string())
    };
    s.push_str(&format!(
        "{:<22} {:>9} {:>6} {:>9} {:>9} {:>7} {:>7} {:>8}\n",
        "player", "train", "deals", "loss %", "upper %", "agree", "epochs", "minutes"
    ));
    let rows = curve["rows"].as_array().cloned().unwrap_or_default();
    let bases = curve["baselines"].as_array().cloned().unwrap_or_default();
    for b in &bases {
        s.push_str(&format!(
            "{:<22} {:>9} {:>6} {:>9} {:>9} {:>7} {:>7} {:>8}\n",
            format!("baseline {}", b["bot"].as_str().unwrap_or("?")),
            "-",
            "-",
            num(&b["all"]["loss_pct"]),
            num(&b["all"]["upper95_pct"]),
            pct(&b["all"]["agreement"]),
            "-",
            "-"
        ));
    }
    for r in &rows {
        s.push_str(&format!(
            "{:<22} {:>9} {:>6} {:>9} {:>9} {:>7} {:>7} {:>8}\n",
            format!(
                "student {}{}",
                r["widths"].as_str().unwrap_or("?"),
                variant_tag(r)
            ),
            r["size"].to_string(),
            deals(r),
            num(&r["test"]["loss_pct"]),
            num(&r["test"]["upper95_pct"]),
            pct(&r["test"]["agreement"]),
            r["epochs_run"].to_string(),
            r["train_minutes"]
                .as_f64()
                .map_or("-".into(), |m| format!("{m:.1}")),
        ));
    }
    s.push_str(
        "\nDecision loss % by place in the trick (1 = the leader) and by score class (S1-S4)\n\n",
    );
    let gs = groups();
    let short: Vec<String> = gs
        .iter()
        .map(|g| g.replace("place ", "P").replace("stratum ", ""))
        .collect();
    s.push_str(&format!("{:<22} {:>9} {:>6}", "player", "train", "deals"));
    for g in &short {
        s.push_str(&format!(" {g:>7}"));
    }
    s.push('\n');
    let line = |label: String, size: String, d: String, by: &Value| {
        let mut l = format!("{label:<22} {size:>9} {d:>6}");
        for g in &gs {
            let v = &by[g.as_str()]["loss_pct"];
            l.push_str(&format!(
                " {:>7}",
                v.as_f64().map_or("-".to_string(), |x| format!("{x:.3}"))
            ));
        }
        l.push('\n');
        l
    };
    for b in &bases {
        s.push_str(&line(
            format!("baseline {}", b["bot"].as_str().unwrap_or("?")),
            "-".into(),
            "-".into(),
            &b["by_group"],
        ));
    }
    for r in &rows {
        s.push_str(&line(
            format!("student {}", r["widths"].as_str().unwrap_or("?")),
            r["size"].to_string(),
            deals(r),
            &r["by_group"],
        ));
    }
    s
}

#[cfg(test)]
mod tests;
