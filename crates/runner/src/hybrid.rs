//! The `hybrid_rung` chunk kind (docs/M3-HYBRID-SPEC.md §3, §5, §6, §6a; docs/m3-hybrid/vast.md):
//! one whole rung of the hybrid yardstick on one machine, the job `.github/workflows/hybrid-rung1.yml`
//! does on a GitHub runner, with every usable thread and, when there is one, the GPU.
//!
//! ```json
//! { "hybrid_rung": { "cut": 4,
//!     "labels_random": 3200000, "labels_random_seed": 101,
//!     "labels_rules": 800000, "labels_rules_seed": 102,
//!     "val_count": 100000, "val_seed": 201,
//!     "golden_trick": 3, "golden_count": 400, "golden_seed": 1, "golden_worlds": 320,
//!     "widths": "1024,512,128", "head": "outcome", "epochs": 60, "batch": 1024, "lr": 0.001,
//!     "patience": 5, "train_seed": 1, "max_minutes": 200, "device": "auto" } }
//! ```
//! Every setting is optional (the defaults are the ones above, those of the rung-1 workflow); any
//! other key is refused ([`KEYS`]). Harvested labels (spec §5 source (c), `search labels --source
//! harvest`: the frontier positions the hybrid's own search reaches from training decisions):
//! `labels_harvest` (rows, default 0: none), `labels_harvest_seed` (103),
//! `harvest_cap_per_decision` (16) and `harvest_positions` (the shares of decisions for the
//! decider's positions 1–2, 3–4 and 5–8, default `"0.25,0.25,0.5"`, uniform over the eight
//! positions); the shard joins the training set after the random-play and rules-v1 shards.
//! `harvest_net` (none): a stored integer network (`int.safetensors` of an earlier run, the
//! previous round's O_cut) whose values drive the harvest's search instead of exact ones (`search
//! labels --harvest-net`: only the sampled positions are solved exactly, the way to harvest at
//! trick 3), given by its key `inputs/sha256/<hex>` (or `sha256:<hex>`); it drives the race
//! harvest's search too.
//! The race harvest (docs/m3-hybrid/rung1.md, "Why seats 3-4 fail"; `search labels --source
//! harvest --harvest-race-classes …`): `labels_race` (rows, default 0: none), `labels_race_seed`
//! (105), `race_classes` (`"2,4"`: the race classes whose reached positions are kept, up to
//! `race_cap`, 256, per decision), `race_strata` (`"4"`: the strata its decisions are drawn
//! from), `race_positions` (the shares of decisions by the decider's position band, default the
//! harvest's `harvest_positions`) and `race_other_cap` (0: the other classes' positions kept per
//! decision, so by default the shard holds race positions only); the shard joins the training set
//! after the harvest shard, and its settings are refused when `labels_race` is 0. `golden_decisions_input`
//! and `golden_worlds_input` (none; both or neither): a stored golden set (a decisions file and its
//! exact worlds file, by key) reused instead of generated (step 5); `verify_pct` must then be at
//! least 1 (spec §4: reuse after a byte-identical recheck).
//! `check` (true): `false` trains only, for a rung whose network is checked elsewhere (O_2: the
//! trick-1 check on the saved exact worlds, `hybrid_t1check`): no golden set (step 5) and no
//! hybrid check (step 7), and the verdict is [`TRAIN_ONLY`]; the golden set's and the check's
//! settings (`golden_*`, `verify_pct`, `tt_mb`) are then refused, not ignored.
//! The GPU need not wait for the CPU (owner, 2026-10-16), answer-neutral in the golden set and
//! the check: `overlap` (true): on a GPU, a golden set to generate (step 5) is made while the
//! network trains (step 6), on the threads the trainer leaves (all usable threads less its 8);
//! either failing stops the other at once and the chunk fails with the reason; `rung.json`'s
//! `overlap` records both times and how long they ran together. `label_parts` (1): the
//! random-play and rules-v1 labels are made in that many shards each ([`Params::shards`]: part `k`
//! of a set with seed `s` has its share of the rows and the seed `s × 100 + k`, so the rows are a
//! pure function of the manifest, whatever the machine), the validation shard first; on a GPU
//! the training starts once part 1 is made, watching the parts' directory (`train.py
//! --watch-dir --expect`: each later shard joins between epochs; patience, the learning-rate
//! scheduler and the best network count only epochs on every shard), and the later parts are
//! made on the threads it leaves; its time limit counts from the last shard. `partial_epochs`
//! (0: no limit) caps the epochs on each partial set (`--partial-epochs`). One part is the former
//! behaviour, step for step. Epochs on part of the data warm-start the full training, which can
//! cost a little accuracy (docs/m3-hybrid/overlap.md); the validation loss says.
//! Also: `mix_weight` (1.0), `start_hands` (20000, for the label roots and the golden
//! decisions), `tt_mb` (64, the solver table per thread of the golden worlds
//! and the check), `verify_pct` (1, the check's recomputed sample of stored exact worlds),
//! `train_threads` (CPU threads of the trainer: all usable threads on the CPU, at most 8 beside
//! a GPU), `compare_cpu` (false; after a GPU training, train once more on the CPU with the same
//! settings and record both, for small checks), `upload_shards` (false; see step 4), `table`
//! (the equity table, as `search_measure`'s), and the scheduler's `threads`, `est_seconds`,
//! `equity_input`.
//!
//! Steps (each a child process with its own log; the heartbeat names the step and its progress):
//! 0. **Inputs** (only when the settings name stored inputs): `harvest_net` and the stored golden
//!    set fetched from the remote store (`inputs/sha256/<hex>`, the path a format-2 manifest's
//!    `inputs` take: `remote::fetch_input`) into `hybrid-work/inputs/`, each SHA-256 checked,
//!    and a stored decisions file must hold `golden_count` decisions at `golden_trick` (a missing
//!    key, other bytes or another set stop the chunk before anything is installed); recorded in
//!    `rung.json`'s `inputs`.
//! 1. **Equity table v0** (`spike::build_table_from`: a manifest input, else R2's copy under its
//!    recorded hash, else built from the model in the repository; hash-checked).
//! 2. **Python environment**: `uv sync --locked` of `python/nn` (the release bundle carries the
//!    source, its `pyproject.toml` and `uv.lock`, and `uv` itself, all hash-checked at boot) into
//!    a fresh environment, with the dependency group `cu126` (PyTorch 2.14.1+cu126, pinned with
//!    hashes in the lock) when `nvidia-smi` lists a GPU and `device` is not `cpu`, else `cpu`.
//!    Up to 3 attempts. Records uv's version, the group, and the lock's SHA-256.
//! 3. **GPU check** (`python -m fortyone_nn.device`; PLAN §12.3 step 5): `nvidia-smi`,
//!    `torch.cuda.is_available()`, the compute capability against the build's architectures, one
//!    tiny training step. With `device` `auto` a failed check means training on the CPU with all
//!    usable threads, said in the heartbeat's warnings, on stderr and in `rung.json` (never in
//!    silence); with `cuda` the chunk fails. Writes `out/hybrid/device.json` (GPU name, driver,
//!    PyTorch, CUDA and cuDNN versions).
//! 4. **Labels** (`search labels`, all usable threads): the random-play, rules-v1, harvested and
//!    race-harvested training shards (a count of 0 skips one) and the validation shard (random
//!    play), at the start of trick `cut`; `rung.json` records each shard's rows by race class. The shards (about 1 KB a row: 4 GB for 4M) stay outside `out/`; their sidecars
//!    (with the shard's SHA-256) go to `out/hybrid/labels/`. With `upload_shards`, each shard is
//!    hashed (it must match its sidecar) and uploaded content-addressed to `inputs/sha256/<hex>`
//!    (skipped when already there; checked by size), and `out/hybrid/shards.json` lists them in
//!    the form a format-2 manifest's `inputs` takes.
//! 5. **Golden set** (`search decisions --allow-uneven`, its start states on all usable threads,
//!    then `search worlds --method exact`):
//!    `golden_count` decisions at trick `golden_trick` (seed `golden_seed`), `golden_worlds`
//!    worlds each, valued exactly; `out/hybrid/decisions-t<G>.jsonl`, `exact-t<G>-n<W>.jsonl`.
//!    Both files are then shared content-addressed to `inputs/sha256/<hex>` (as `upload_shards`;
//!    a failure is a warning, the files being in the bundle too), and `rung.json`'s
//!    `golden.reuse_params` gives the two settings that reuse them. With a stored golden set
//!    (step 0) nothing is generated, and the check recomputes `verify_pct`% of the stored worlds
//!    with this binary and stops on any byte difference.
//! 6. **Training** (`python -m fortyone_nn.train`, which also quantises the network and measures
//!    the integer network): `out/hybrid/net/` holds `float.safetensors`, `int.safetensors` and
//!    `report.json` (history, errors, device facts). The heartbeat shows the epoch and batches.
//!    The integer network is then shared content-addressed (`put_input`, checked against its
//!    SHA-256; `rung.json` `net.int_key` and `result.json` `net_int_key` give the key a later
//!    run names it by, as `harvest_net` or a check's network); a failure is only a warning.
//! 7. **Hybrid check** (`search hybrid-check`, all usable threads): H_cut against exact on the
//!    golden set; `out/hybrid/hc-t<G>-o<K>.json` and its printed tables in `.txt`. Skipped, with
//!    step 5, when `check` is false.
//! 8. **Summary**: `out/hybrid/rung.json` (rewritten after every step): the resolved settings,
//!    each step's seconds, the environment and device facts, the label sidecars, the networks'
//!    SHA-256, the check's verdict and its overall row, and every warning.
//!
//! CPU use: every step sets `partial_cpu_ok`, so the launcher's low-CPU rule (85% for 5 minutes)
//! does not apply to this kind. The long stretches leave the CPU idle by design (the download of
//! the environment, and the training on the GPU, where the CPU only feeds it), and on hundreds of
//! threads the CPU steps finish in seconds to a minute, partly in serial parts (loading the
//! table, the last rows of a shard; the shards are hashed and written while their rows are made,
//! and the start states are played on every thread), so their heartbeats can read low too: the
//! rule would stop a healthy run. Instead the steps that could hang are time-limited (killed, and the chunk fails):
//! each `uv sync` attempt 30 minutes, the GPU check 10 minutes, the training `max_minutes` plus
//! 30 minutes; the run's dollar cap bounds everything.
//!
//! Reproducibility: the labels, the golden set and the check are byte-identical on every machine
//! and thread count (`search`). Float training on a GPU is not bit-reproducible across machines;
//! the integer network is what the check validates, and its SHA-256 is recorded
//! (docs/M3-HYBRID-SPEC.md §6).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};

use search::oracle_labels::{
    HARVEST_CAP, HARVEST_SHARES, RACE_CAP, RACE_CLASSES, RaceKeep, parse_band_shares,
    parse_race_classes, parse_strata,
};

use crate::bundle::sha256_file;
use crate::remote::{Remote, fetch_input, input_hex, input_key, put_input};
use crate::spike::build_table_from;
use crate::{Ctx, sibling};

/// The settings a `hybrid_rung` chunk takes; any other key is refused.
pub(crate) const KEYS: [&str; 49] = [
    "cut",
    "check",
    "label_parts",
    "overlap",
    "partial_epochs",
    "labels_random",
    "labels_random_seed",
    "labels_rules",
    "labels_rules_seed",
    "labels_harvest",
    "labels_harvest_seed",
    "harvest_cap_per_decision",
    "harvest_positions",
    "harvest_net",
    "labels_race",
    "labels_race_seed",
    "race_classes",
    "race_cap",
    "race_strata",
    "race_positions",
    "race_other_cap",
    "val_count",
    "val_seed",
    "golden_trick",
    "golden_count",
    "golden_seed",
    "golden_worlds",
    "golden_decisions_input",
    "golden_worlds_input",
    "start_hands",
    "tt_mb",
    "verify_pct",
    "widths",
    "head",
    "epochs",
    "batch",
    "lr",
    "patience",
    "train_seed",
    "max_minutes",
    "mix_weight",
    "device",
    "train_threads",
    "compare_cpu",
    "upload_shards",
    "table",
    // Read by the format-2 scheduler and worker.
    "threads",
    "est_seconds",
    "equity_input",
];

/// The verdict of a chunk that trains only (`"check": false`).
pub(crate) const TRAIN_ONLY: &str = "not checked (train only)";

/// The golden sets' seed, which `search labels` refuses (`search::oracle_labels::GOLDEN_SEED`).
const GOLDEN_SEED: u64 = 1;

/// CPU threads of the trainer beside a GPU (it only gathers batches and launches GPU work).
const GPU_TRAIN_THREADS: u64 = 8;

/// Attempts at `uv sync` (it downloads about 3.7 GB for the GPU build), and each one's limit.
const SYNC_ATTEMPTS: u32 = 3;
const SYNC_LIMIT: Duration = Duration::from_secs(30 * 60);

/// The GPU check's limit (a CUDA start-up that hangs on a faulty GPU).
const CHECK_LIMIT: Duration = Duration::from_secs(10 * 60);

/// The training's limit beyond its own `max_minutes`.
const TRAIN_EXTRA: Duration = Duration::from_secs(30 * 60);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Device {
    Auto,
    Cuda,
    Cpu,
}

impl Device {
    fn name(self) -> &'static str {
        match self {
            Device::Auto => "auto",
            Device::Cuda => "cuda",
            Device::Cpu => "cpu",
        }
    }
}

/// One label shard to make.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LabelSet {
    /// `random`, `rules`, `harvest`, `race` or `val` (the file is `t<cut>-<name>-s<seed>.npy`).
    pub name: &'static str,
    /// `search labels --source`.
    pub source: &'static str,
    pub count: u64,
    pub seed: u64,
    /// More `search labels` arguments (the harvest's settings).
    pub args: Vec<String>,
}

/// The race harvest's settings (`labels_race`; docs/m3-hybrid/rung1.md, "Why seats 3-4 fail"),
/// as given to `search labels --source harvest`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RaceSettings {
    /// `race_classes`: the race classes kept up to `cap` (ascending).
    pub classes: Vec<u8>,
    /// `race_cap`: at most this many positions of the race classes per decision.
    pub cap: u64,
    /// `race_strata`: the strata the decisions are drawn from (ascending).
    pub strata: Vec<u8>,
    /// `race_positions`: the shares of decisions by the decider's position band.
    pub positions: String,
    /// `race_other_cap`: at most this many positions of the other classes per decision (0:
    /// race positions only).
    pub other_cap: u64,
}

/// A `hybrid_rung` chunk's settings, checked.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Params {
    pub cut: u64,
    /// Make the golden set and run the hybrid check (steps 5 and 7); `false` (`"check": false`)
    /// trains only: the network is checked elsewhere (for O_2, on the saved exact trick-1 worlds,
    /// `hybrid_t1check`).
    pub check: bool,
    /// The training shards (counts above 0), random first.
    pub train: Vec<LabelSet>,
    pub val: LabelSet,
    pub golden_trick: u64,
    pub golden_count: u64,
    pub golden_seed: u64,
    pub golden_worlds: u64,
    pub start_hands: u64,
    pub tt_mb: u64,
    pub verify_pct: f64,
    pub widths: Vec<u64>,
    pub head: String,
    pub epochs: u64,
    pub batch: u64,
    pub lr: f64,
    pub patience: u64,
    pub train_seed: u64,
    pub max_minutes: f64,
    pub mix_weight: f64,
    pub device: Device,
    pub train_threads: Option<u64>,
    pub compare_cpu: bool,
    pub upload_shards: bool,
    /// `harvest_net`: the SHA-256 of the integer network (`int.safetensors` of an earlier run,
    /// stored at `inputs/sha256/<hex>`) whose values drive the harvest's search (and the race
    /// harvest's); `None`: exact values at the frontier.
    pub harvest_net: Option<String>,
    /// The race harvest's settings, when `labels_race` is above 0.
    pub race: Option<RaceSettings>,
    /// `golden_decisions_input` and `golden_worlds_input`: the SHA-256 of a stored golden set's
    /// decisions and exact worlds files, reused instead of generated; `None`: generated.
    pub golden_inputs: Option<(String, String)>,
    /// `label_parts`: the random-play and rules-v1 labels are made in this many shards each
    /// ([`Params::shards`]), and on a GPU the training starts after the first ones (1: one shard
    /// each, the training after all of them, as before).
    pub label_parts: u64,
    /// `overlap` (true): on a GPU, a generated golden set is made while the network trains.
    pub overlap: bool,
    /// `partial_epochs` (0: no limit): with `label_parts` above 1, at most this many epochs on
    /// each partial set of shards (the trainer's `--partial-epochs`).
    pub partial_epochs: u64,
}

/// One label shard to make: its set, and its part (`(k, of)`, 1-based) when the set is made in
/// parts.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ShardJob {
    pub set: LabelSet,
    pub part: Option<(u64, u64)>,
}

impl ShardJob {
    /// `t<cut>-<name>-s<seed>.npy`, or `t<cut>-<name>-p<k>of<n>-s<seed>.npy` for a part (its
    /// own seed).
    pub fn file(&self, cut: u64) -> String {
        match self.part {
            None => format!("t{cut}-{}-s{}.npy", self.set.name, self.set.seed),
            Some((k, n)) => format!("t{cut}-{}-p{k}of{n}-s{}.npy", self.set.name, self.set.seed),
        }
    }
}

/// The seed of part `k` (1-based) of a set with seed `seed` made in parts: `seed × 100 + k`
/// (so part 2 of seed 101 is 10102), `None` on overflow. A pure function of the manifest.
pub(crate) fn part_seed(seed: u64, k: u64) -> Option<u64> {
    seed.checked_mul(100)?.checked_add(k)
}

/// Rows of part `k` (1-based) of `count` rows in `n` parts: equal, the first `count mod n`
/// parts one more.
pub(crate) fn part_count(count: u64, k: u64, n: u64) -> u64 {
    count / n + u64::from(k <= count % n)
}

impl Params {
    /// The settings as recorded in `rung.json`.
    fn to_json(&self) -> Value {
        let set = |s: &LabelSet| {
            let mut v = json!({ "source": s.source, "count": s.count, "seed": s.seed });
            if !s.args.is_empty() {
                v["args"] = json!(s.args);
            }
            v
        };
        let golden = self.check.then(|| {
            json!({ "trick": self.golden_trick, "count": self.golden_count,
                    "seed": self.golden_seed, "worlds": self.golden_worlds,
                    "decisions_input": self.golden_inputs.as_ref().map(|(d, _)| input_key(d)),
                    "worlds_input": self.golden_inputs.as_ref().map(|(_, w)| input_key(w)) })
        });
        json!({
            "cut": self.cut,
            "check": self.check,
            "train": self.train.iter().map(|s| (s.name.to_string(), set(s))).collect::<Map<_, _>>(),
            "val": set(&self.val),
            "golden": golden,
            "harvest_net": self.harvest_net.as_deref().map(input_key),
            "race": self.race.as_ref().map(|r| json!({
                "classes": r.classes, "cap": r.cap, "strata": r.strata,
                "positions": r.positions, "other_cap": r.other_cap,
            })),
            "start_hands": self.start_hands, "tt_mb": self.tt_mb, "verify_pct": self.verify_pct,
            "net": { "widths": self.widths, "head": self.head, "epochs": self.epochs,
                     "batch": self.batch, "lr": self.lr, "patience": self.patience,
                     "seed": self.train_seed, "max_minutes": self.max_minutes,
                     "mix_weight": self.mix_weight },
            "device": self.device.name(), "train_threads": self.train_threads,
            "compare_cpu": self.compare_cpu, "upload_shards": self.upload_shards,
            "label_parts": self.label_parts, "overlap": self.overlap,
            "partial_epochs": self.partial_epochs,
        })
    }

    /// The label shards in the order they are made. One part: the training sets, then the
    /// validation shard (as before). More: the validation shard, then part 1 of the random-play
    /// and rules-v1 sets, part 2, …, each part `part_count` rows with its own seed
    /// (`part_seed`), then the harvest (one shard).
    pub(crate) fn shards(&self) -> Vec<ShardJob> {
        let one = |set: &LabelSet| ShardJob {
            set: set.clone(),
            part: None,
        };
        if self.label_parts <= 1 {
            return self
                .train
                .iter()
                .chain(std::iter::once(&self.val))
                .map(one)
                .collect();
        }
        let n = self.label_parts;
        let mut out = vec![one(&self.val)];
        for k in 1..=n {
            for s in self.train.iter().filter(|s| s.source != "harvest") {
                out.push(ShardJob {
                    set: LabelSet {
                        count: part_count(s.count, k, n),
                        seed: part_seed(s.seed, k).expect("checked when read"),
                        ..s.clone()
                    },
                    part: Some((k, n)),
                });
            }
        }
        out.extend(self.train.iter().filter(|s| s.source == "harvest").map(one));
        out
    }

    fn widths_arg(&self) -> String {
        self.widths
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Reads and checks a `hybrid_rung` chunk's settings (module notes).
pub(crate) fn parse(p: &Value) -> Result<Params, String> {
    let empty = Map::new();
    let o = match p {
        Value::Null => &empty,
        Value::Object(o) => o,
        _ => return Err("hybrid_rung: the settings must be an object".into()),
    };
    if let Some(k) = o.keys().find(|k| !KEYS.contains(&k.as_str())) {
        return Err(format!(
            "hybrid_rung: unknown setting {k:?} (settings: {})",
            KEYS.join(", ")
        ));
    }
    let u = |k: &str, d: u64| -> Result<u64, String> {
        match o.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(v) => v
                .as_u64()
                .ok_or_else(|| format!("hybrid_rung: {k} must be a whole number")),
        }
    };
    let f = |k: &str, d: f64| -> Result<f64, String> {
        match o.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(v) => v
                .as_f64()
                .filter(|x| x.is_finite() && *x >= 0.0)
                .ok_or_else(|| format!("hybrid_rung: {k} must be a number, 0 or more")),
        }
    };
    let b = |k: &str| -> Result<bool, String> {
        match o.get(k) {
            None | Some(Value::Null) => Ok(false),
            Some(v) => v
                .as_bool()
                .ok_or_else(|| format!("hybrid_rung: {k} must be true or false")),
        }
    };
    let s = |k: &str, d: &str| -> Result<String, String> {
        match o.get(k) {
            None | Some(Value::Null) => Ok(d.to_string()),
            Some(v) => v
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("hybrid_rung: {k} must be a string")),
        }
    };
    let cut = u("cut", 4)?;
    if !(2..=5).contains(&cut) {
        return Err(format!(
            "hybrid_rung: cut {cut} must be 2..=5 (the frontier at the start of that trick)"
        ));
    }
    // Train only (`"check": false`): the golden set's and the check's settings would be
    // ignored, so they are refused rather than silently dropped.
    let check = match o.get("check") {
        None | Some(Value::Null) => true,
        Some(v) => v
            .as_bool()
            .ok_or("hybrid_rung: check must be true or false")?,
    };
    if !check {
        let golden_keys = [
            "golden_trick",
            "golden_count",
            "golden_seed",
            "golden_worlds",
            "golden_decisions_input",
            "golden_worlds_input",
            "verify_pct",
            "tt_mb",
        ];
        if let Some(k) = golden_keys.iter().find(|k| o.contains_key(**k)) {
            return Err(format!(
                "hybrid_rung: check false trains only (no golden set, no hybrid check), but                  {k} is set: it belongs to the golden set or the check; remove it"
            ));
        }
    }
    let set = |name: &'static str, source: &'static str, count: &str, seed: &str, d: (u64, u64)| {
        Ok::<_, String>(LabelSet {
            name,
            source,
            count: u(count, d.0)?,
            seed: u(seed, d.1)?,
            args: Vec::new(),
        })
    };
    let random = set(
        "random",
        "random",
        "labels_random",
        "labels_random_seed",
        (3_200_000, 101),
    )?;
    let rules = set(
        "rules",
        "rules",
        "labels_rules",
        "labels_rules_seed",
        (800_000, 102),
    )?;
    let mut harvest = set(
        "harvest",
        "harvest",
        "labels_harvest",
        "labels_harvest_seed",
        (0, 103),
    )?;
    let cap = u("harvest_cap_per_decision", u64::from(HARVEST_CAP))?;
    if !(1..=u64::from(u32::MAX)).contains(&cap) {
        return Err("hybrid_rung: harvest_cap_per_decision must be at least 1".into());
    }
    // Stored inputs, by content: `inputs/sha256/<hex>` (or `sha256:<hex>`, as a manifest's
    // `inputs` and `shards.json` write them).
    let input = |k: &str| -> Result<Option<String>, String> {
        match o.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(v)) => input_hex(v).map(Some).ok_or_else(|| {
                format!(
                    "hybrid_rung: {k} {v:?} must be a stored input's key, \
                     \"inputs/sha256/<64 hex digits>\" (or \"sha256:<64 hex digits>\")"
                )
            }),
            Some(_) => Err(format!(
                "hybrid_rung: {k} must be a string, \"inputs/sha256/<64 hex digits>\""
            )),
        }
    };
    let mut race = set(
        "race",
        "harvest",
        "labels_race",
        "labels_race_seed",
        (0, 105),
    )?;
    let harvest_net = input("harvest_net")?;
    if harvest_net.is_some() && harvest.count == 0 && race.count == 0 {
        return Err(
            "hybrid_rung: harvest_net drives the harvest's search: it needs labels_harvest or \
             labels_race above 0"
                .into(),
        );
    }
    let golden_inputs = match (
        input("golden_decisions_input")?,
        input("golden_worlds_input")?,
    ) {
        (Some(d), Some(w)) => Some((d, w)),
        (None, None) => None,
        _ => {
            return Err(
                "hybrid_rung: golden_decisions_input and golden_worlds_input go together (a \
                 stored golden set is its decisions and their exact worlds)"
                    .into(),
            );
        }
    };
    let shares = s("harvest_positions", HARVEST_SHARES)?;
    parse_band_shares(&shares)
        .map_err(|e| format!("hybrid_rung: harvest_positions: {e}"))
        .and_then(|w| {
            if w.iter().sum::<u64>() > 1 << 24 {
                Err(format!(
                    "hybrid_rung: harvest_positions {shares:?}: use fewer decimals"
                ))
            } else {
                Ok(())
            }
        })?;
    harvest.args = vec![
        "--harvest-cap".into(),
        cap.to_string(),
        "--harvest-positions".into(),
        shares.clone(),
    ];
    // The race harvest (`labels_race`): its settings are refused without it, not ignored.
    let race_keys = [
        "labels_race_seed",
        "race_classes",
        "race_cap",
        "race_strata",
        "race_positions",
        "race_other_cap",
    ];
    let race_settings = if race.count == 0 {
        if let Some(k) = race_keys.iter().find(|k| o.contains_key(**k)) {
            return Err(format!(
                "hybrid_rung: {k} is set but labels_race is 0: the race harvest's settings need \
                 labels_race above 0"
            ));
        }
        None
    } else {
        let classes = s("race_classes", RACE_CLASSES)?;
        let classes = RaceKeep {
            classes: parse_race_classes(&classes)
                .map_err(|e| format!("hybrid_rung: race_classes: {e}"))?,
            cap: 1,
        }
        .list();
        let race_cap = u("race_cap", u64::from(RACE_CAP))?;
        if !(1..=u64::from(u32::MAX)).contains(&race_cap) {
            return Err("hybrid_rung: race_cap must be at least 1".into());
        }
        let strata = parse_strata(&s("race_strata", "4")?)
            .map_err(|e| format!("hybrid_rung: race_strata: {e}"))?;
        let positions = s("race_positions", &shares)?;
        parse_band_shares(&positions)
            .map_err(|e| format!("hybrid_rung: race_positions: {e}"))
            .and_then(|w| {
                if w.iter().sum::<u64>() > 1 << 24 {
                    Err(format!(
                        "hybrid_rung: race_positions {positions:?}: use fewer decimals"
                    ))
                } else {
                    Ok(())
                }
            })?;
        let other_cap = u("race_other_cap", 0)?;
        if other_cap > u64::from(u32::MAX) {
            return Err("hybrid_rung: race_other_cap is too large".into());
        }
        let list = |v: &[u8]| v.iter().map(u8::to_string).collect::<Vec<_>>().join(",");
        race.args = vec![
            "--harvest-cap".into(),
            other_cap.to_string(),
            "--harvest-positions".into(),
            positions.clone(),
            "--harvest-strata".into(),
            list(&strata),
            "--harvest-race-classes".into(),
            list(&classes),
            "--harvest-race-cap".into(),
            race_cap.to_string(),
        ];
        Some(RaceSettings {
            classes,
            cap: race_cap,
            strata,
            positions,
            other_cap,
        })
    };
    let val = set("val", "random", "val_count", "val_seed", (100_000, 201))?;
    let train: Vec<LabelSet> = [random, rules, harvest, race]
        .into_iter()
        .filter(|s| s.count > 0)
        .collect();
    if train.is_empty() {
        return Err(
            "hybrid_rung: labels_random, labels_rules, labels_harvest and labels_race are all 0: \
             no training labels"
                .into(),
        );
    }
    if val.count == 0 {
        return Err("hybrid_rung: val_count must be at least 1".into());
    }
    let label_parts = u("label_parts", 1)?;
    if !(1..=99).contains(&label_parts) {
        return Err("hybrid_rung: label_parts must be 1 to 99".into());
    }
    let overlap = match o.get("overlap") {
        None | Some(Value::Null) => true,
        Some(v) => v
            .as_bool()
            .ok_or("hybrid_rung: overlap must be true or false")?,
    };
    let partial_epochs = u("partial_epochs", 0)?;
    if partial_epochs > 0 && label_parts == 1 {
        return Err(
            "hybrid_rung: partial_epochs limits the epochs on part of the labels: it needs \
             label_parts above 1"
                .into(),
        );
    }
    // The seeds the shards are made with: a set's own, or with parts each part's.
    let mut effective: Vec<LabelSet> = Vec::new();
    for x in train.iter().chain(std::iter::once(&val)) {
        if label_parts > 1 && x.source != "harvest" && x.name != "val" {
            for k in 1..=label_parts {
                let seed = part_seed(x.seed, k).ok_or_else(|| {
                    format!(
                        "hybrid_rung: the {} labels' seed {} is too large for parts",
                        x.name, x.seed
                    )
                })?;
                effective.push(LabelSet { seed, ..x.clone() });
            }
        } else {
            effective.push(x.clone());
        }
    }
    // Training, validation and golden-set positions use disjoint seeds (spec §5).
    let mut seen: Vec<(u64, &str)> = Vec::new();
    for x in &effective {
        if x.seed == GOLDEN_SEED {
            return Err(format!(
                "hybrid_rung: the {} labels' seed is {GOLDEN_SEED}, the golden sets' seed; use another",
                x.name
            ));
        }
        if let Some((_, other)) = seen.iter().find(|(s, _)| *s == x.seed) {
            return Err(format!(
                "hybrid_rung: the {} and {other} labels share seed {}; training and validation \
                 positions need disjoint seeds",
                x.name, x.seed
            ));
        }
        seen.push((x.seed, x.name));
    }
    let golden_trick = u("golden_trick", cut - 1)?;
    if golden_trick == 0 || golden_trick >= cut {
        return Err(format!(
            "hybrid_rung: golden_trick {golden_trick} must be 1..{cut} (a trick before the cut)"
        ));
    }
    let golden_count = u("golden_count", 400)?;
    let golden_worlds = u("golden_worlds", 320)?;
    if golden_count == 0 || golden_worlds == 0 {
        return Err(
            "hybrid_rung: golden_count and golden_worlds must be at least 1 (to train without a \
             golden set and a check, set \"check\": false)"
                .into(),
        );
    }
    let widths_text = s("widths", "1024,512,128")?;
    let widths: Vec<u64> = widths_text
        .split(',')
        .map(|w| w.trim().parse::<u64>().ok().filter(|&w| (1..=65_536).contains(&w)))
        .collect::<Option<Vec<_>>>()
        .filter(|w| !w.is_empty())
        .ok_or_else(|| {
            format!("hybrid_rung: widths {widths_text:?} must be hidden-layer sizes like \"1024,512,128\"")
        })?;
    let head = s("head", "outcome")?;
    if !["outcome", "money4"].contains(&head.as_str()) {
        return Err(format!("hybrid_rung: head {head:?}: outcome or money4"));
    }
    let device = match s("device", "auto")?.as_str() {
        "auto" => Device::Auto,
        "cuda" => Device::Cuda,
        "cpu" => Device::Cpu,
        d => return Err(format!("hybrid_rung: device {d:?}: auto, cuda or cpu")),
    };
    let epochs = u("epochs", 60)?;
    let batch = u("batch", 1024)?;
    if epochs == 0 || batch == 0 {
        return Err("hybrid_rung: epochs and batch must be at least 1".into());
    }
    let lr = f("lr", 0.001)?;
    if lr <= 0.0 {
        return Err("hybrid_rung: lr must be above 0".into());
    }
    let train_threads = match o.get("train_threads") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_u64()
                .filter(|&t| t > 0)
                .ok_or("hybrid_rung: train_threads must be a whole number, at least 1")?,
        ),
    };
    let verify_pct = f("verify_pct", 1.0)?;
    if verify_pct > 100.0 {
        return Err("hybrid_rung: verify_pct is a percentage (0 to 100)".into());
    }
    if golden_inputs.is_some() && verify_pct < 1.0 {
        return Err(
            "hybrid_rung: a stored golden set is reused only after at least 1% of its worlds are \
             recomputed and found byte-identical (spec §4): verify_pct must be at least 1"
                .into(),
        );
    }
    Ok(Params {
        cut,
        check,
        train,
        val,
        golden_trick,
        golden_count,
        golden_seed: u("golden_seed", GOLDEN_SEED)?,
        golden_worlds,
        start_hands: u("start_hands", 20_000)?,
        tt_mb: u("tt_mb", 64)?.max(1),
        verify_pct,
        widths,
        head,
        epochs,
        batch,
        lr,
        patience: u("patience", 5)?,
        train_seed: u("train_seed", 1)?,
        max_minutes: f("max_minutes", 200.0)?,
        mix_weight: f("mix_weight", 1.0)?,
        device,
        train_threads,
        compare_cpu: b("compare_cpu")?,
        upload_shards: b("upload_shards")?,
        harvest_net,
        race: race_settings,
        golden_inputs,
        label_parts,
        overlap,
        partial_epochs,
    })
}

/// The programs a chunk runs (the release bundle's in production; fakes in the tests).
pub(crate) struct Tools {
    pub search: PathBuf,
    pub uv: PathBuf,
    /// `python/nn`: `pyproject.toml`, `uv.lock`, `fortyone_nn/`.
    pub python_dir: PathBuf,
    pub nvidia_smi: PathBuf,
    /// The remote store (R2 in production, `RUNNER_REMOTE_DIR` on a laptop, a directory in the
    /// tests): stored inputs come from it, the shards and the golden set go to it.
    pub remote: Option<Arc<dyn Remote>>,
}

impl Tools {
    /// `search` and `uv` next to this runner (else on the PATH), and the training code:
    /// `RUNNER_PYTHON_DIR`, else `python/nn` in this executable's directory (the release bundle)
    /// or one of its parents (a repository checkout's `target/release`).
    pub(crate) fn find() -> Result<Tools, String> {
        Ok(Tools {
            search: sibling("search"),
            uv: sibling("uv"),
            python_dir: python_dir()?,
            nvidia_smi: PathBuf::from("nvidia-smi"),
            remote: crate::remote::from_env().map(Arc::from),
        })
    }
}

fn is_python_dir(d: &Path) -> bool {
    d.join("pyproject.toml").is_file()
        && d.join("uv.lock").is_file()
        && d.join("fortyone_nn").is_dir()
}

fn python_dir() -> Result<PathBuf, String> {
    if let Some(d) = std::env::var_os("RUNNER_PYTHON_DIR").filter(|v| !v.is_empty()) {
        let d = PathBuf::from(d);
        return if is_python_dir(&d) {
            Ok(std::path::absolute(&d).unwrap_or(d))
        } else {
            Err(format!(
                "RUNNER_PYTHON_DIR {} has no pyproject.toml, uv.lock and fortyone_nn/",
                d.display()
            ))
        };
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    exe.parent()
        .into_iter()
        .flat_map(Path::ancestors)
        .map(|d| d.join("python").join("nn"))
        .find(|d| is_python_dir(d))
        .ok_or_else(|| {
            "hybrid_rung: the training code (python/nn with pyproject.toml, uv.lock and \
             fortyone_nn/) is neither in the release bundle next to the runner nor in a parent \
             directory; set RUNNER_PYTHON_DIR"
                .to_string()
        })
}

/// The GPUs `nvidia-smi` lists (one line each), or `None` when it is missing, fails or lists
/// none.
fn gpus_listed(nvidia_smi: &Path) -> Option<Vec<String>> {
    let o = Command::new(nvidia_smi)
        .args([
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let lines: Vec<String> = String::from_utf8_lossy(&o.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    (!lines.is_empty()).then_some(lines)
}

/// The decisions at trick `trick` in a decisions file (`search decisions`' JSON lines).
fn decisions_at(path: &Path, trick: u64) -> Result<u64, String> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("hybrid_rung: {}: {e}", path.display()))?;
    let mut n = 0;
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| {
            format!(
                "hybrid_rung: golden_decisions_input, line {}: not a decision ({e})",
                i + 1
            )
        })?;
        n += u64::from(v["trick"].as_u64() == Some(trick));
    }
    Ok(n)
}

/// The last `labels: <done> of <total> rows` line `search labels` wrote to its log.
fn label_progress(log: &Path) -> Option<(u64, u64)> {
    let t = fs::read_to_string(log).ok()?;
    t.lines().rev().find_map(|l| {
        let rest = l.trim().strip_prefix("labels: ")?;
        let (done, rest) = rest.split_once(" of ")?;
        let (total, _) = rest.split_once(" rows")?;
        Some((done.parse().ok()?, total.parse().ok()?))
    })
}

fn read_json(p: &Path) -> Option<Value> {
    fs::read_to_string(p)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
}

/// A step's progress file (`prog`, a JSON object the child rewrites) into another step's
/// heartbeat progress `p`: its fields as `<prefix>_<field>` (its `device` left out), and its name
/// `name` added to the step's (`"golden worlds + train"`), so the run's heartbeat shows both
/// steps (`progress.chunk_steps` keeps fields that end in `_done` or `_total` and the trainer's
/// `train_` fields). A missing or half-written file adds only the name.
fn merge_step(p: &mut Value, prefix: &str, name: &str, prog: Option<&Path>) {
    if let Some(Value::Object(x)) = prog.and_then(read_json) {
        for (k, v) in x {
            if k != "device" {
                p[format!("{prefix}_{k}")] = v;
            }
        }
    }
    let joined = format!(" + {name}");
    match p["step"].as_str() {
        Some(step)
            if step == name || step.ends_with(&joined) || step.contains(&format!("{name} + ")) => {}
        Some(step) => p["step"] = json!(format!("{step}{joined}")),
        None => p["step"] = json!(name),
    }
}

/// The trainer's progress (`prog`) into another step's heartbeat progress ([`merge_step`]).
fn merge_train(p: &mut Value, prog: &Path) {
    merge_step(p, "train", "train", Some(prog));
}

/// Steps running beside the current one, for the heartbeat: (prefix, step name, progress file).
type Beside = Vec<(String, String, Option<PathBuf>)>;

fn merge_beside(p: &mut Value, beside: &Beside) {
    for (prefix, name, prog) in beside {
        merge_step(p, prefix, name, prog.as_deref());
    }
}

/// One step of [`Rung::concurrent`]: a child process run at the same time as others, with its
/// own share of the threads (`threads`; its command is given that many), its log, the JSON
/// progress file it rewrites, if any (its fields join the heartbeat as `<prefix>_<field>`), and
/// a time limit.
// For the audit loop (docs/m3-hybrid/audit-loop.md §4: its CPU steps beside the GPU); tested here.
#[allow(dead_code)]
pub(crate) struct ConcurrentStep {
    pub step: String,
    pub cmd: Command,
    pub threads: u64,
    pub log: String,
    pub progress: Option<PathBuf>,
    pub prefix: String,
    pub limit: Option<Duration>,
}

/// What a trainer reads: given shards (the training ones, then the validation one), or the
/// parts' directory it watches (`--watch-dir`, `expect` shards) and the validation shard.
enum Feed {
    Files(Vec<PathBuf>),
    Watch {
        dir: PathBuf,
        expect: usize,
        val: PathBuf,
    },
}

/// A training running in the background ([`Rung::start_train`]).
struct BgTrain {
    progress: PathBuf,
    dir: PathBuf,
    device: String,
    threads: u64,
    t: Instant,
}

fn write_json(p: &Path, v: &Value) -> Result<(), String> {
    if let Some(d) = p.parent() {
        fs::create_dir_all(d).map_err(|e| format!("mkdir {}: {e}", d.display()))?;
    }
    let text = serde_json::to_string_pretty(v).map_err(|e| e.to_string())? + "\n";
    fs::write(p, text).map_err(|e| format!("write {}: {e}", p.display()))
}

fn path_str(p: &Path) -> String {
    p.display().to_string()
}

/// To a tenth.
fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// Rounded seconds.
fn secs(t: Instant) -> f64 {
    (t.elapsed().as_secs_f64() * 10.0).round() / 10.0
}

/// Run one `hybrid_rung` chunk.
pub(crate) fn chunk(ctx: &mut Ctx, p: &Value) -> Result<Value, String> {
    let params = parse(p)?;
    let tools = Tools::find()?;
    ctx.set_phase(
        "chunk",
        json!({ "kind": "hybrid_rung", "step": "setup: table v0", "partial_cpu_ok": true }),
    );
    let (table, table_rec) = build_table_from(ctx, p.get("table").unwrap_or(&json!({})), "auto")?;
    chunk_steps(ctx, &params, &tools, &table, table_rec)
}

/// [`chunk_with`] that never leaves a child running beside the steps when it fails.
fn chunk_steps(
    ctx: &mut Ctx,
    params: &Params,
    tools: &Tools,
    table: &Path,
    table_rec: Value,
) -> Result<Value, String> {
    let r = chunk_with(ctx, params, tools, table, table_rec);
    if r.is_err() {
        ctx.kill_bg();
    }
    r
}

/// The steps of a chunk after the equity table (module notes), with the given programs.
pub(crate) fn chunk_with(
    ctx: &mut Ctx,
    params: &Params,
    tools: &Tools,
    table: &Path,
    table_rec: Value,
) -> Result<Value, String> {
    // Absolute: the Python steps run from `python/nn`.
    let abs = |p: PathBuf| std::path::absolute(&p).unwrap_or(p);
    let out = abs(ctx.root.join("out").join("hybrid"));
    let work = abs(ctx.root.join("hybrid-work"));
    for d in [&out, &work] {
        fs::create_dir_all(d).map_err(|e| format!("mkdir {}: {e}", d.display()))?;
    }
    let mut r = Rung {
        ctx,
        p: params,
        tools,
        table: abs(table.to_path_buf()),
        out,
        work,
        rec: Map::new(),
        warnings: Vec::new(),
        steps: Vec::new(),
        t0: Instant::now(),
        python: PathBuf::new(),
        device: String::new(),
        inputs: BTreeMap::new(),
        bg_train: None,
        beside: Vec::new(),
    };
    r.rec.insert("params".into(), params.to_json());
    r.rec.insert("table".into(), table_rec);
    r.rec.insert("threads".into(), json!(r.ctx.threads));
    r.fetch_inputs()?;
    r.python_env()?;
    r.device_check()?;
    // The GPU need not wait for the CPU (owner, 2026-10-16): with label parts it starts on the
    // first ones while the rest are made, and a golden set to generate is made while it trains.
    // Both only on a GPU: on the CPU they would share the same threads.
    let gpu = r.device == "cuda";
    let shards = r.labels(gpu && params.label_parts > 1)?;
    if params.upload_shards {
        r.upload_shards(&shards)?;
    }
    let generate = params.check && params.golden_inputs.is_none();
    let overlap = gpu && params.overlap && generate;
    if overlap && r.bg_train.is_none() {
        r.start_train(&Feed::Files(shards.clone()))?;
    }
    let tg = Instant::now();
    let golden = if params.check {
        Some(r.golden()?)
    } else {
        None
    };
    let golden_ran = (tg, Instant::now());
    let net = r.out.join("net");
    if r.bg_train.is_some() {
        let (ts, te) = r.finish_train()?;
        let (gs, ge) = golden_ran;
        let span = |a: Instant, b: Instant| b.saturating_duration_since(a).as_secs_f64();
        let both = span(ts.max(gs), te.min(ge));
        r.rec.insert(
            "overlap".into(),
            json!({ "golden_with_training": overlap,
                    "train_threads": r.train_threads(),
                    "golden_threads": if overlap { json!(r.threads().saturating_sub(r.train_threads()).max(1)) } else { Value::Null },
                    "train_s": round1(span(ts, te)),
                    "golden_s": if generate { json!(round1(span(gs, ge))) } else { Value::Null },
                    "both_s": if overlap { json!(round1(both)) } else { Value::Null } }),
        );
        r.save()?;
    } else {
        if params.check && generate && params.overlap {
            r.rec.insert(
                "overlap".into(),
                json!({ "golden_with_training": false,
                        "why": "training on the CPU: the golden set and the training would share \
                                its threads" }),
            );
        }
        r.train(&shards, &net, &r.device.clone(), "train", "net")?;
    }
    if params.compare_cpu {
        if r.device == "cuda" {
            r.train(
                &shards,
                &r.out.join("net-cpu"),
                "cpu",
                "train (CPU comparison)",
                "net_cpu",
            )?;
            r.compare();
        } else {
            r.warn("compare_cpu: the network was trained on the CPU already; no comparison".into());
        }
    }
    let verdict = match golden {
        Some((decisions, worlds)) => r.check(&net, &decisions, &worlds)?,
        None => {
            r.rec.insert(
                "check".into(),
                json!({ "verdict": TRAIN_ONLY,
                        "note": "check false: no golden set was made and no hybrid check was \
                                 run; the network (net.int_key) is checked elsewhere" }),
            );
            TRAIN_ONLY.to_string()
        }
    };
    r.rec.insert("seconds".into(), json!(secs(r.t0)));
    r.save()?;
    r.ctx.set_phase(
        "chunk",
        json!({ "kind": "hybrid_rung", "step": "done", "partial_cpu_ok": true,
                "verdict": verdict, "warnings": r.warnings }),
    );
    Ok(json!({
        "kind": "hybrid_rung",
        "verdict": verdict,
        "device": r.device,
        "gpu": r.rec.get("device").and_then(|d| d.get("gpu_name")).cloned(),
        "net_int_sha256": r.rec.get("net").and_then(|n| n.get("int_sha256")).cloned(),
        "net_int_key": r.rec.get("net").and_then(|n| n.get("int_key")).cloned(),
        "steps": r.steps,
        "warnings": r.warnings,
    }))
}

struct Rung<'a> {
    ctx: &'a mut Ctx,
    p: &'a Params,
    tools: &'a Tools,
    table: PathBuf,
    /// `out/hybrid` (bundled and uploaded).
    out: PathBuf,
    /// Work files outside `out/`: the label shards, the Python environment, progress files.
    work: PathBuf,
    /// `rung.json` as it grows.
    rec: Map<String, Value>,
    warnings: Vec<String>,
    /// `{step, seconds}` of every finished step.
    steps: Vec<Value>,
    t0: Instant,
    /// The environment's Python.
    python: PathBuf,
    /// Where the network trains: `cuda` or `cpu` (empty before the GPU check).
    device: String,
    /// Stored inputs fetched and checked (`harvest_net`, `golden_decisions`, `golden_worlds`).
    inputs: BTreeMap<&'static str, PathBuf>,
    /// The training, while it runs in the background.
    bg_train: Option<BgTrain>,
    /// Steps running beside the current one ([`Rung::concurrent`], the background training):
    /// their thread shares and progress, for the heartbeat.
    beside: Vec<(String, String, Option<PathBuf>, u64)>,
}

impl Rung<'_> {
    fn threads(&self) -> u64 {
        self.ctx.threads as u64
    }

    /// The heartbeat's progress for a step: the kind, the step, `partial_cpu_ok` (always: module
    /// notes), the warnings so far, and `extra`.
    fn progress(&self, step: &str, extra: Value) -> Value {
        let mut v = json!({ "kind": "hybrid_rung", "step": step, "partial_cpu_ok": true,
                            "steps_done": self.steps.len(), "warnings": self.warnings });
        if !self.device.is_empty() {
            v["device"] = json!(self.device);
        }
        if let (Some(o), Value::Object(e)) = (v.as_object_mut(), extra) {
            o.extend(e);
        }
        v
    }

    fn warn(&mut self, w: String) {
        eprintln!("runner: WARNING: hybrid_rung: {w}");
        self.warnings.push(w);
        self.rec.insert("warnings".into(), json!(self.warnings));
    }

    /// Records a finished step and rewrites `rung.json`.
    fn done(&mut self, step: &str, t: Instant) -> Result<(), String> {
        self.steps.push(json!({ "step": step, "seconds": secs(t) }));
        self.save()
    }

    fn save(&mut self) -> Result<(), String> {
        self.rec.insert("steps".into(), json!(self.steps));
        self.rec.insert("warnings".into(), json!(self.warnings));
        write_json(
            &self.out.join("rung.json"),
            &Value::Object(self.rec.clone()),
        )
    }

    /// A Python command in the environment, run from `python/nn` (so `-m fortyone_nn.…` finds the
    /// package).
    fn python_cmd(&self, module: &str) -> Command {
        let mut c = Command::new(&self.python);
        c.current_dir(&self.tools.python_dir)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .args(["-m", module]);
        c
    }

    /// Step 0 (only when the settings name stored inputs): the harvest's network and a stored
    /// golden set, each fetched from `inputs/sha256/<hex>` into the work directory
    /// (`remote::fetch_input`, the path a format-2 run's manifest inputs take) and its SHA-256
    /// checked; first, so a wrong key stops the chunk before anything is installed.
    fn fetch_inputs(&mut self) -> Result<(), String> {
        let mut want: Vec<(&'static str, String)> = Vec::new();
        if let Some(h) = &self.p.harvest_net {
            want.push(("harvest_net", h.clone()));
        }
        if let Some((d, w)) = &self.p.golden_inputs {
            want.push(("golden_decisions", d.clone()));
            want.push(("golden_worlds", w.clone()));
        }
        if want.is_empty() {
            return Ok(());
        }
        let t = Instant::now();
        let progress = self.progress("inputs", json!({ "inputs": want.len() }));
        self.ctx.set_phase("chunk", progress);
        let remote = self.tools.remote.clone().ok_or(
            "hybrid_rung: stored inputs (harvest_net, golden_decisions_input, \
             golden_worlds_input) come from the remote store (R2, or RUNNER_REMOTE_DIR), and \
             there is none",
        )?;
        let dir = self.work.join("inputs");
        let mut recs = Map::new();
        for (name, hex) in want {
            let t1 = Instant::now();
            let local = dir.join(&hex);
            let (r, h, l) = (Arc::clone(&remote), hex.clone(), local.clone());
            let downloaded = self
                .ctx
                .blocking(move || fetch_input(&*r, &h, &l))
                .map_err(|e| format!("hybrid_rung: {name}: {e}"))?;
            let bytes = fs::metadata(&local).map_or(0, |m| m.len());
            println!(
                "runner: hybrid_rung: {name}: {} ({bytes} bytes, SHA-256 checked)",
                input_key(&hex)
            );
            recs.insert(
                name.into(),
                json!({ "key": input_key(&hex), "sha256": hex, "bytes": bytes,
                        "downloaded": downloaded, "seconds": secs(t1) }),
            );
            self.inputs.insert(name, local);
        }
        // A stored golden set must be the one the settings describe; checked now, before the
        // labels and the training.
        if let Some(d) = self.inputs.get("golden_decisions") {
            let (g, want) = (self.p.golden_trick, self.p.golden_count);
            let n = decisions_at(d, g)?;
            if n != want {
                return Err(format!(
                    "hybrid_rung: the stored golden decisions hold {n} decisions at trick {g}, \
                     but golden_trick {g} and golden_count {want} ask for {want}"
                ));
            }
        }
        self.rec.insert("inputs".into(), Value::Object(recs));
        self.done("inputs", t)
    }

    /// Step 2: the Python environment.
    fn python_env(&mut self) -> Result<(), String> {
        let t = Instant::now();
        let gpus = gpus_listed(&self.tools.nvidia_smi);
        let group = match (self.p.device, &gpus) {
            (Device::Cpu, _) => "cpu",
            (_, Some(_)) => "cu126",
            (Device::Cuda, None) => {
                return Err(
                    "hybrid_rung: device \"cuda\" was asked for, but nvidia-smi is \
                            missing or lists no GPU"
                        .into(),
                );
            }
            (Device::Auto, None) => "cpu",
        };
        let env_dir = self.work.join(format!("pyenv-{group}"));
        let progress = self.progress(
            "python environment",
            json!({ "group": group, "gpus": gpus }),
        );
        self.ctx.set_phase("chunk", progress);
        let version = Command::new(&self.tools.uv)
            .arg("--version")
            .output()
            .map_err(|e| {
                format!(
                    "hybrid_rung: cannot run uv ({}): {e}",
                    self.tools.uv.display()
                )
            })?;
        let uv_version = String::from_utf8_lossy(&version.stdout).trim().to_string();
        let mut attempts = 0;
        loop {
            attempts += 1;
            let mut c = Command::new(&self.tools.uv);
            c.args(["sync", "--locked", "--no-default-groups", "--group", group])
                .args(["--python", "3.14", "--no-progress", "--project"])
                .arg(&self.tools.python_dir)
                .env("UV_PROJECT_ENVIRONMENT", &env_dir);
            match self.ctx.run_child_io(
                c,
                "hybrid/uv-sync.log",
                None,
                Some(SYNC_LIMIT),
                &mut |_| {},
            ) {
                Ok(()) => break,
                Err(e) if attempts < SYNC_ATTEMPTS && crate::stop::requested().is_none() => {
                    eprintln!(
                        "runner: hybrid_rung: uv sync attempt {attempts} failed ({e}); trying again in 15 s"
                    );
                    self.ctx
                        .blocking(|| std::thread::sleep(Duration::from_secs(15)));
                }
                Err(e) => return Err(format!("hybrid_rung: uv sync ({group}): {e}")),
            }
        }
        self.python = if cfg!(windows) {
            env_dir.join("Scripts").join("python.exe")
        } else {
            env_dir.join("bin").join("python")
        };
        let lock = sha256_file(&self.tools.python_dir.join("uv.lock"))?.0;
        self.rec.insert(
            "python_env".into(),
            json!({ "uv": uv_version, "group": group, "uv_lock_sha256": lock,
                    "attempts": attempts, "nvidia_smi": gpus, "seconds": secs(t) }),
        );
        self.done("python environment", t)
    }

    /// Step 3: the GPU check; decides `self.device`.
    fn device_check(&mut self) -> Result<(), String> {
        let t = Instant::now();
        let progress = self.progress("GPU check", json!({}));
        self.ctx.set_phase("chunk", progress);
        let path = self.out.join("device.json");
        let mut c = self.python_cmd("fortyone_nn.device");
        c.args(["--want", self.p.device.name(), "--json"])
            .arg(&path);
        let res =
            self.ctx
                .run_child_io(c, "hybrid/device.log", None, Some(CHECK_LIMIT), &mut |_| {});
        let facts = read_json(&path);
        let reason = facts
            .as_ref()
            .and_then(|f| f["reason"].as_str())
            .unwrap_or("no reason recorded")
            .to_string();
        if let Err(e) = res {
            return Err(match self.p.device {
                Device::Cuda => {
                    format!(
                        "hybrid_rung: device \"cuda\" was asked for, but there is no usable GPU: {reason}"
                    )
                }
                _ => format!("hybrid_rung: GPU check: {e}"),
            });
        }
        let facts = facts.ok_or("hybrid_rung: the GPU check wrote no device.json")?;
        self.device = facts["device"].as_str().unwrap_or("cpu").to_string();
        if self.device != "cuda" && self.p.device == Device::Auto {
            let threads = self.train_threads();
            self.warn(format!(
                "no usable GPU ({reason}): training on the CPU with {threads} threads"
            ));
        }
        self.rec.insert("device".into(), facts);
        self.done("GPU check", t)
    }

    fn train_threads(&self) -> u64 {
        self.train_threads_on(&self.device)
    }

    fn train_threads_on(&self, device: &str) -> u64 {
        self.p.train_threads.unwrap_or(if device == "cuda" {
            self.threads().min(GPU_TRAIN_THREADS)
        } else {
            self.threads()
        })
    }

    /// Threads for a step while the network may be training beside it: all usable threads,
    /// less the trainer's when it runs ([`Rung::train_threads`]; at least 1).
    fn step_threads(&self) -> u64 {
        let taken: u64 = self.beside.iter().map(|b| b.3).sum();
        self.threads().saturating_sub(taken).max(1)
    }

    /// The steps beside the current one, for [`merge_beside`].
    fn beside_progress(&self) -> Beside {
        self.beside
            .iter()
            .map(|(prefix, name, prog, _)| (prefix.clone(), name.clone(), prog.clone()))
            .collect()
    }

    /// Runs `steps` at the same time, each a child with its own share of the threads (their
    /// shares, and those of any step already running beside, may not exceed the machine's), the
    /// heartbeat showing every one (names joined by `+`, each one's progress fields prefixed);
    /// when one fails, the others are stopped at once and that is the error (nothing is left
    /// running). Records each as a finished step, in the order given, with its share and when it
    /// started and ended (`rung.json`'s `concurrent`). Returns when each ran. For the audit loop's
    /// CPU steps beside the GPU, and any other answer-neutral overlap (the results never depend
    /// on what runs beside: each child gets a fixed thread count).
    #[allow(dead_code)]
    pub(crate) fn concurrent(
        &mut self,
        steps: Vec<ConcurrentStep>,
    ) -> Result<Vec<(Instant, Instant)>, String> {
        let want: u64 = steps.iter().map(|s| s.threads).sum::<u64>()
            + self.beside.iter().map(|b| b.3).sum::<u64>();
        if want > self.threads() {
            return Err(format!(
                "hybrid_rung: steps {} ask for {want} threads at once, more than the machine's {}",
                steps
                    .iter()
                    .map(|s| s.step.as_str())
                    .collect::<Vec<_>>()
                    .join(" + "),
                self.threads()
            ));
        }
        let t0 = Instant::now();
        let names: Vec<String> = steps.iter().map(|s| s.step.clone()).collect();
        let shares: Vec<u64> = steps.iter().map(|s| s.threads).collect();
        for s in steps {
            if let Err(e) = self.ctx.spawn_bg(s.cmd, &s.log, &s.step, s.limit) {
                self.ctx.kill_bg();
                self.beside.clear();
                return Err(e);
            }
            self.beside.push((
                s.prefix.clone(),
                s.step.clone(),
                s.progress.clone(),
                s.threads,
            ));
        }
        let mut ran = Vec::new();
        for name in &names {
            let base = json!({ "kind": "hybrid_rung", "step": names.join(" + "),
                               "partial_cpu_ok": true, "steps_done": self.steps.len(),
                               "warnings": self.warnings });
            self.ctx.set_phase("chunk", base.clone());
            let beside = self.beside_progress();
            let r = self.ctx.wait_bg(name, &mut |ctx| {
                let mut p = base.clone();
                merge_beside(&mut p, &beside);
                ctx.progress = p;
            });
            match r {
                // The group's steps stay in the heartbeat (with their last progress) until all
                // have finished.
                Ok(Some(x)) => ran.push(x),
                Ok(None) => {
                    self.beside.clear();
                    return Err(format!("hybrid_rung: step {name} was not running"));
                }
                Err(e) => {
                    self.ctx.kill_bg();
                    self.beside.clear();
                    return Err(format!("hybrid_rung: {e}"));
                }
            }
        }
        self.beside.retain(|b| !names.contains(&b.1));
        let at = |t: Instant| round1(t.saturating_duration_since(self.t0).as_secs_f64());
        let mut group = Vec::new();
        for ((name, (a, b)), share) in names.iter().zip(&ran).zip(&shares) {
            self.steps.push(json!({ "step": name, "seconds": round1(b.saturating_duration_since(*a).as_secs_f64()),
                                    "threads": share, "started_s": at(*a), "ended_s": at(*b) }));
            group.push(
                json!({ "step": name, "threads": share, "started_s": at(*a), "ended_s": at(*b) }),
            );
        }
        let mut all = self
            .rec
            .get("concurrent")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        all.push(json!({ "steps": group, "wall_s": round1(t0.elapsed().as_secs_f64()) }));
        self.rec.insert("concurrent".into(), Value::Array(all));
        self.save()?;
        Ok(ran)
    }

    /// Step 4: the label shards ([`Params::shards`]). Returns the training shards' paths, then
    /// the validation shard's. With `stream` (parts, on a GPU), the training starts in the
    /// background once the validation shard and every shard of part 1 are made
    /// ([`Rung::start_train`], watching the parts' directory), and the later parts are made on
    /// the threads it leaves.
    fn labels(&mut self, stream: bool) -> Result<Vec<PathBuf>, String> {
        let dir = self.work.join("labels");
        // Training shards made in parts go to a directory of their own, which a streamed
        // trainer watches; the validation shard stays outside it.
        let parts_dir = dir.join("train");
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let parted = self.p.label_parts > 1;
        if parted {
            fs::create_dir_all(&parts_dir).map_err(|e| e.to_string())?;
        }
        let jobs = self.p.shards();
        let mut train = Vec::new();
        let mut val = None;
        let mut recs = Vec::new();
        for (i, job) in jobs.iter().enumerate() {
            let s = &job.set;
            let t = Instant::now();
            let file = job.file(self.p.cut);
            let path = if parted && s.name != "val" {
                parts_dir.join(&file)
            } else {
                dir.join(&file)
            };
            let log = format!("hybrid/labels-{}.log", s.name);
            let part = job
                .part
                .map_or(String::new(), |(k, n)| format!(" p{k}/{n}"));
            let step = format!("labels {}{part} ({} of {})", s.name, i + 1, jobs.len());
            let threads = self.step_threads();
            let mut c = Command::new(&self.tools.search);
            c.args(["labels", "--trick", &self.p.cut.to_string()])
                .args([
                    "--count",
                    &s.count.to_string(),
                    "--seed",
                    &s.seed.to_string(),
                ])
                .args(["--source", s.source, "--out"])
                .arg(&path)
                .arg("--equity")
                .arg(&self.table)
                .args(["--threads", &threads.to_string()])
                .args(["--start-hands", &self.p.start_hands.to_string()])
                .args(&s.args);
            // The network that drives the harvest's search (`harvest_net`), fetched and checked.
            let net = self
                .inputs
                .get("harvest_net")
                .filter(|_| s.source == "harvest");
            if let Some(net) = net {
                c.arg("--harvest-net").arg(net);
            }
            let base = self.progress(&step, json!({ "rows_total": s.count }));
            self.ctx.set_phase("chunk", base.clone());
            let log_path = self.ctx.logs.join(&log);
            let beside = self.beside_progress();
            self.ctx.run_child_with(c, &log, &mut |ctx| {
                let mut p = base.clone();
                if let Some((done, total)) = label_progress(&log_path) {
                    p["rows_done"] = json!(done);
                    p["rows_total"] = json!(total);
                }
                merge_beside(&mut p, &beside);
                ctx.progress = p;
            })?;
            let side_src = PathBuf::from(format!("{}.json", path.display()));
            let side = read_json(&side_src)
                .ok_or_else(|| format!("hybrid_rung: {} wrote no sidecar", path.display()))?;
            let side_dst = self.out.join("labels").join(format!("{file}.json"));
            write_json(&side_dst, &side)?;
            let bytes = fs::metadata(&path).map_or(0, |m| m.len());
            let mut rec = json!({ "name": s.name, "source": s.source, "count": s.count,
                                  "seed": s.seed, "file": file, "bytes": bytes,
                                  "shard_sha256": side["shard_sha256"],
                                  "threads": threads, "seconds": secs(t) });
            if let Some((k, n)) = job.part {
                rec["part"] = json!([k, n]);
            }
            if net.is_some() {
                rec["harvest_net_sha256"] = json!(self.p.harvest_net);
                rec["harvest_driver"] = side["harvest"]["frontier_driver"].clone();
            }
            if let Some(c) = side.get("rows_by_race_class") {
                rec["rows_by_race_class"] = c.clone();
            }
            recs.push(rec);
            self.rec.insert("labels".into(), json!(recs));
            self.done(&step, t)?;
            if s.name == "val" {
                val = Some(path);
            } else {
                train.push(path);
            }
            // Part 1 made (and the validation shard): the training starts beside the rest.
            let next_part = jobs.get(i + 1).and_then(|j| j.part).map(|(k, _)| k);
            if stream
                && self.bg_train.is_none()
                && job.part.is_some_and(|(k, _)| k == 1)
                && next_part != Some(1)
            {
                let expect = jobs.iter().filter(|j| j.set.name != "val").count();
                let v = val
                    .clone()
                    .ok_or("hybrid_rung: the validation shard comes first")?;
                self.start_train(&Feed::Watch {
                    dir: parts_dir.clone(),
                    expect,
                    val: v,
                })?;
                self.rec.insert(
                    "stream".into(),
                    json!({ "label_parts": self.p.label_parts, "shards": expect,
                            "train_started_after": format!("{} of {} shards (part 1)", train.len(), expect),
                            "train_started_s": secs(self.t0),
                            "label_threads_while_training": self.step_threads(),
                            "partial_epochs": self.p.partial_epochs }),
                );
            }
        }
        if self.bg_train.is_some() {
            // The trainer's time limit counts from when its last shard is made.
            self.ctx.bg_limit_from_now("train");
        }
        train.push(val.ok_or("hybrid_rung: no validation shard")?);
        Ok(train)
    }

    /// Step 4b (`upload_shards`): each shard to `inputs/sha256/<hex>`.
    fn upload_shards(&mut self, shards: &[PathBuf]) -> Result<(), String> {
        let t = Instant::now();
        let progress = self.progress("upload shards", json!({ "shards": shards.len() }));
        self.ctx.set_phase("chunk", progress);
        let Some(remote) = self.tools.remote.clone() else {
            self.warn(
                "upload_shards: no remote store (R2, or RUNNER_REMOTE_DIR): the label shards \
                 were not uploaded"
                    .into(),
            );
            return self.done("upload shards", t);
        };
        let labels = self.rec.get("labels").cloned().unwrap_or(Value::Null);
        let mut list = Vec::new();
        let mut inputs = Map::new();
        for path in shards {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let rec = labels
                .as_array()
                .into_iter()
                .flatten()
                .find(|r| r["file"].as_str() == Some(name.as_str()))
                .cloned()
                .unwrap_or(Value::Null);
            let want = rec["shard_sha256"].as_str().unwrap_or_default().to_string();
            let file = name;
            let (p2, r) = (path.clone(), Arc::clone(&remote));
            let (sha, bytes, status) = self
                .ctx
                .blocking(move || put_input(&*r, &p2, Some(&want)))
                .map_err(|e| format!("hybrid_rung: upload_shards: {e}"))?;
            println!("runner: hybrid_rung: {file}: inputs/sha256/{sha} ({status})");
            inputs.insert(
                file.trim_end_matches(".npy").to_string(),
                json!(format!("sha256:{sha}")),
            );
            list.push(json!({ "file": file, "sha256": sha, "bytes": bytes,
                              "key": input_key(&sha), "status": status }));
        }
        let doc = json!({ "shards": list, "manifest_inputs": inputs });
        write_json(&self.out.join("shards.json"), &doc)?;
        self.rec.insert("shards".into(), doc);
        self.done("upload shards", t)
    }

    /// Step 5: the golden set, generated (then shared: [`Rung::share_golden`]) or stored
    /// ([`Rung::golden_stored`]). Returns (decisions file, exact worlds file).
    fn golden(&mut self) -> Result<(PathBuf, PathBuf), String> {
        if let (Some(d), Some(w)) = (
            self.inputs.get("golden_decisions").cloned(),
            self.inputs.get("golden_worlds").cloned(),
        ) {
            return self.golden_stored(d, w);
        }
        let (g, w, seed) = (
            self.p.golden_trick,
            self.p.golden_worlds,
            self.p.golden_seed,
        );
        let t = Instant::now();
        let decisions = self.out.join(format!("decisions-t{g}.jsonl"));
        let progress = self.progress("golden decisions", json!({}));
        self.ctx.set_phase("chunk", progress);
        let mut c = Command::new(&self.tools.search);
        c.args(["decisions", "--trick", &g.to_string()])
            .args(["--count", &self.p.golden_count.to_string()])
            .args(["--seed", &seed.to_string(), "--allow-uneven"])
            .args(["--threads", &self.step_threads().to_string()])
            .args(["--start-hands", &self.p.start_hands.to_string(), "--out"])
            .arg(&decisions);
        self.ctx.run_child(c, "hybrid/golden.log")?;
        self.done("golden decisions", t)?;

        let t = Instant::now();
        let worlds = self.out.join(format!("exact-t{g}-n{w}.jsonl"));
        let prog = self.work.join("worlds.progress.json");
        let _ = fs::remove_file(&prog);
        let mut c = Command::new(&self.tools.search);
        c.args(["worlds", "--decisions"])
            .arg(&decisions)
            .args(["--method", "exact", "--to", &w.to_string()])
            .args(["--seed", &seed.to_string(), "--out"])
            .arg(&worlds)
            .arg("--equity")
            .arg(&self.table)
            .args(["--threads", &self.step_threads().to_string()])
            .args(["--tt-mb", &self.p.tt_mb.to_string(), "--progress"])
            .arg(&prog);
        let base = self.progress("golden worlds", json!({}));
        self.ctx.set_phase("chunk", base.clone());
        let beside = self.beside_progress();
        self.ctx
            .run_child_with(c, "hybrid/golden.log", &mut |ctx| {
                let mut p = base.clone();
                merge_beside(&mut p, &beside);
                if let Some(x) = read_json(&prog) {
                    let g = |k: &str| x.get(k).and_then(Value::as_u64).unwrap_or(0);
                    for k in [
                        "decisions_done",
                        "decisions_total",
                        "worlds_done",
                        "worlds_total",
                    ] {
                        p[k] = json!(g(k));
                    }
                }
                ctx.progress = p;
            })?;
        let sha = |p: &Path| sha256_file(p).map(|x| x.0);
        self.rec.insert(
            "golden".into(),
            json!({ "trick": g, "count": self.p.golden_count, "seed": seed, "worlds": w,
                    "decisions_file": format!("decisions-t{g}.jsonl"),
                    "decisions_sha256": sha(&decisions)?,
                    "worlds_file": format!("exact-t{g}-n{w}.jsonl"),
                    "worlds_sha256": sha(&worlds)?,
                    "seconds": secs(t) }),
        );
        self.done("golden worlds", t)?;
        self.share_golden(&decisions, &worlds)?;
        Ok((decisions, worlds))
    }

    /// Step 5 with a stored golden set (`golden_decisions_input` and `golden_worlds_input`,
    /// fetched and checked at step 0, where its decisions file was found to hold `golden_count`
    /// decisions at `golden_trick`): nothing is generated. The check (step 7) then recomputes
    /// `verify_pct` (at least 1) percent of the stored worlds with this binary and stops on any
    /// byte difference (docs/M3-HYBRID-SPEC.md §4), and refuses a decision whose stored record
    /// does not hold worlds `0..golden_worlds` or whose candidates differ.
    fn golden_stored(
        &mut self,
        decisions: PathBuf,
        worlds: PathBuf,
    ) -> Result<(PathBuf, PathBuf), String> {
        let t = Instant::now();
        let progress = self.progress("golden set (stored)", json!({}));
        self.ctx.set_phase("chunk", progress);
        let g = self.p.golden_trick;
        let (dh, wh) = self
            .p
            .golden_inputs
            .clone()
            .ok_or("hybrid_rung: no stored golden set was asked for")?;
        self.rec.insert(
            "golden".into(),
            json!({ "trick": g, "count": self.p.golden_count, "seed": self.p.golden_seed,
                    "worlds": self.p.golden_worlds, "stored": true,
                    "decisions_key": input_key(&dh), "decisions_sha256": dh,
                    "worlds_key": input_key(&wh), "worlds_sha256": wh,
                    "verify_pct": self.p.verify_pct,
                    "note": "a stored golden set, reused: the hybrid check recomputes verify_pct \
                             percent of its worlds with this binary and stops on any byte \
                             difference (docs/M3-HYBRID-SPEC.md §4)",
                    "seconds": secs(t) }),
        );
        self.done("golden set (stored)", t)?;
        Ok((decisions, worlds))
    }

    /// Step 5b, after generating the golden set: its decisions and exact worlds files go to
    /// `inputs/sha256/<hex>` (content-addressed, as the label shards), so a later run can reuse
    /// them (`golden_decisions_input`, `golden_worlds_input`: `rung.json`'s
    /// `golden.reuse_params`). A failure is only a warning: the files are in this run's bundle
    /// too.
    fn share_golden(&mut self, decisions: &Path, worlds: &Path) -> Result<(), String> {
        let t = Instant::now();
        let progress = self.progress("share golden set", json!({}));
        self.ctx.set_phase("chunk", progress);
        let Some(remote) = self.tools.remote.clone() else {
            self.warn(
                "the golden set was not shared for reuse: no remote store (R2, or \
                 RUNNER_REMOTE_DIR); it is in this run's bundle"
                    .into(),
            );
            return self.done("share golden set", t);
        };
        let mut reuse = Map::new();
        for (name, path, param) in [
            ("decisions", decisions, "golden_decisions_input"),
            ("worlds", worlds, "golden_worlds_input"),
        ] {
            let want = self.rec["golden"][format!("{name}_sha256").as_str()]
                .as_str()
                .map(str::to_string);
            let (p2, r) = (path.to_path_buf(), Arc::clone(&remote));
            match self
                .ctx
                .blocking(move || put_input(&*r, &p2, want.as_deref()))
            {
                Ok((sha, bytes, status)) => {
                    println!(
                        "runner: hybrid_rung: golden {name}: {} ({bytes} bytes, {status})",
                        input_key(&sha)
                    );
                    if let Some(g) = self.rec.get_mut("golden") {
                        g[format!("{name}_key").as_str()] = json!(input_key(&sha));
                    }
                    reuse.insert(param.into(), json!(input_key(&sha)));
                }
                Err(e) => self.warn(format!(
                    "the golden {name} were not shared for reuse: {e}; they are in this run's \
                     bundle"
                )),
            }
        }
        if reuse.len() == 2
            && let Some(g) = self.rec.get_mut("golden")
        {
            g["reuse_params"] = Value::Object(reuse);
        }
        self.done("share golden set", t)
    }

    /// The trainer's command: on `feed`, into `dir`, on `device` with `threads` threads, its
    /// progress to `prog`.
    fn train_cmd(
        &self,
        feed: &Feed,
        dir: &Path,
        device: &str,
        threads: u64,
        prog: &Path,
    ) -> Command {
        let mut c = self.python_cmd("fortyone_nn.train");
        match feed {
            Feed::Files(shards) => {
                let (train, val) = shards.split_at(shards.len() - 1);
                c.arg("--train").args(train).arg("--val").arg(&val[0]);
            }
            Feed::Watch { dir, expect, val } => {
                c.arg("--watch-dir")
                    .arg(dir)
                    .args(["--expect", &expect.to_string()])
                    .arg("--val")
                    .arg(val)
                    // Loading a new part must not take the threads the labels are made on.
                    .args(["--load-threads", &threads.to_string()]);
                if self.p.partial_epochs > 0 {
                    c.args(["--partial-epochs", &self.p.partial_epochs.to_string()]);
                }
            }
        }
        c.arg("--out")
            .arg(dir)
            .args(["--head", &self.p.head, "--widths", &self.p.widths_arg()])
            .args(["--epochs", &self.p.epochs.to_string()])
            .args(["--batch", &self.p.batch.to_string()])
            .args(["--lr", &self.p.lr.to_string()])
            .args(["--patience", &self.p.patience.to_string()])
            .args(["--seed", &self.p.train_seed.to_string()])
            .args(["--threads", &threads.to_string()])
            .args(["--max-minutes", &self.p.max_minutes.to_string()])
            .args(["--mix-weight", &self.p.mix_weight.to_string()])
            .args(["--device", device, "--progress"])
            .arg(prog);
        c
    }

    /// The trainer's time limit: it stops itself after `max_minutes` (between epochs); loading,
    /// the last epoch, quantising and the integer check get 30 minutes more.
    fn train_limit(&self) -> Option<Duration> {
        (self.p.max_minutes > 0.0)
            .then(|| Duration::from_secs_f64(self.p.max_minutes * 60.0) + TRAIN_EXTRA)
    }

    /// Step 6: one training run on `device` into `dir`, recorded under `key`.
    fn train(
        &mut self,
        shards: &[PathBuf],
        dir: &Path,
        device: &str,
        step: &str,
        key: &str,
    ) -> Result<(), String> {
        let t = Instant::now();
        let prog = self.work.join(format!("{key}.progress.json"));
        let _ = fs::remove_file(&prog);
        let threads = self.train_threads_on(device);
        let c = self.train_cmd(&Feed::Files(shards.to_vec()), dir, device, threads, &prog);
        let base = self.progress(
            step,
            json!({ "train_device": device, "train_threads": threads }),
        );
        self.ctx.set_phase("chunk", base.clone());
        let limit = self.train_limit();
        self.ctx
            .run_child_io(c, &format!("hybrid/{key}.log"), None, limit, &mut |ctx| {
                let mut p = base.clone();
                merge_train(&mut p, &prog);
                ctx.progress = p;
            })?;
        self.train_record(dir, device, threads, key, step, t)
    }

    /// Step 6 in the background (on the GPU, beside the later label parts or the golden set):
    /// the trainer of the network `net` started on `feed`; [`Rung::finish_train`] waits for it.
    fn start_train(&mut self, feed: &Feed) -> Result<(), String> {
        let dir = self.out.join("net");
        let prog = self.work.join("net.progress.json");
        let _ = fs::remove_file(&prog);
        let device = self.device.clone();
        let threads = self.train_threads();
        let c = self.train_cmd(feed, &dir, &device, threads, &prog);
        let limit = self.train_limit();
        self.ctx.spawn_bg(c, "hybrid/net.log", "train", limit)?;
        self.beside
            .push(("train".into(), "train".into(), Some(prog.clone()), threads));
        println!(
            "runner: hybrid_rung: training started beside the next steps ({threads} threads on {device})"
        );
        self.bg_train = Some(BgTrain {
            progress: prog,
            dir,
            device,
            threads,
            t: Instant::now(),
        });
        Ok(())
    }

    /// Waits for the background training ([`Rung::start_train`]) and records it as the step
    /// `train`. Returns when it ran.
    fn finish_train(&mut self) -> Result<(Instant, Instant), String> {
        let b = self
            .bg_train
            .take()
            .ok_or("hybrid_rung: no training is running")?;
        let base = self.progress(
            "train",
            json!({ "train_device": b.device, "train_threads": b.threads }),
        );
        self.ctx.set_phase("chunk", base.clone());
        let prog = b.progress.clone();
        let ran = self.ctx.wait_bg("train", &mut |ctx| {
            let mut p = base.clone();
            merge_train(&mut p, &prog);
            ctx.progress = p;
        });
        self.beside.retain(|x| x.1 != "train");
        let ran = ran?.ok_or("hybrid_rung: the training was not running")?;
        self.train_record(&b.dir, &b.device, b.threads, "net", "train", b.t)?;
        Ok(ran)
    }

    /// Records a finished training (its report, the networks' SHA-256, the shared key) under
    /// `key` and the step `step`, begun at `t`.
    fn train_record(
        &mut self,
        dir: &Path,
        device: &str,
        threads: u64,
        key: &str,
        step: &str,
        t: Instant,
    ) -> Result<(), String> {
        let report = read_json(&dir.join("report.json")).ok_or_else(|| {
            format!(
                "hybrid_rung: training wrote no {}/report.json",
                dir.display()
            )
        })?;
        let sha = |f: &str| sha256_file(&dir.join(f)).map(|x| x.0);
        let rel = dir
            .strip_prefix(&self.out)
            .map_or_else(|_| path_str(dir), path_str);
        let int_sha = sha("int.safetensors")?;
        // The integer network is shared content-addressed (as the golden set), so a later run
        // names it as an input (`harvest_net`, a check's network); a failure is only a warning.
        let int_key = if key == "net" {
            self.share_net(&dir.join("int.safetensors"), &int_sha)
        } else {
            None
        };
        let mut rec = json!({ "dir": rel, "device": device, "threads": threads,
                "int_sha256": int_sha, "int_key": int_key,
                "float_sha256": sha("float.safetensors")?,
                "epochs_run": report["history"].as_array().map_or(0, Vec::len),
                "best_epoch": report["best_epoch"],
                "float": report["float"],
                "int": report["int"],
                "val_loss_by_epoch": report["history"].as_array().map(|h| {
                    h.iter().map(|e| e["val_loss"].clone()).collect::<Vec<_>>()
                }),
                "minutes": report["minutes"],
                "device_facts": report["device"],
                "seconds": secs(t) });
        if !report["stream"].is_null() {
            rec["stream"] = report["stream"].clone();
        }
        self.rec.insert(key.into(), rec);
        self.done(step, t)
    }

    /// Uploads the trained integer network to `inputs/sha256/<hex>` (`put_input`, checked
    /// against `sha`); its key, or `None` with a warning when there is no remote store or the
    /// upload fails (the network is in this run's bundle either way).
    fn share_net(&mut self, path: &Path, sha: &str) -> Option<String> {
        let Some(remote) = self.tools.remote.clone() else {
            self.warn(
                "the network was not shared as an input: no remote store (R2, or \
                 RUNNER_REMOTE_DIR); it is in this run's bundle"
                    .into(),
            );
            return None;
        };
        let (p2, want) = (path.to_path_buf(), sha.to_string());
        match self
            .ctx
            .blocking(move || put_input(&*remote, &p2, Some(&want)))
        {
            Ok((sha, bytes, status)) => {
                println!(
                    "runner: hybrid_rung: network: {} ({bytes} bytes, {status})",
                    input_key(&sha)
                );
                Some(input_key(&sha))
            }
            Err(e) => {
                self.warn(format!(
                    "the network was not shared as an input: {e}; it is in this run's bundle"
                ));
                None
            }
        }
    }

    /// `compare_cpu`: the GPU and CPU trainings side by side.
    fn compare(&mut self) {
        let get = |k: &str| self.rec.get(k).cloned().unwrap_or(Value::Null);
        let (g, c) = (&get("net"), &get("net_cpu"));
        let curve = |v: &Value| -> Vec<f64> {
            v["val_loss_by_epoch"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_f64).collect())
                .unwrap_or_default()
        };
        let (cg, cc) = (curve(g), curve(c));
        let max_diff = cg
            .iter()
            .zip(&cc)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let doc = json!({
            "same_int_net": g["int_sha256"] == c["int_sha256"],
            "epochs": [cg.len(), cc.len()],
            "max_abs_val_loss_difference_per_epoch": max_diff,
            "gpu_int_error": g["int"]["error"], "cpu_int_error": c["int"]["error"],
            "note": "GPU float training is not bit-identical to the CPU's (kernels add in other \
                     orders); the curves should agree closely",
        });
        self.rec.insert("compare_cpu".into(), doc);
    }

    /// Step 7: the hybrid check. Returns `PASS` or `FAIL`.
    fn check(&mut self, net: &Path, decisions: &Path, worlds: &Path) -> Result<String, String> {
        let t = Instant::now();
        let (g, k) = (self.p.golden_trick, self.p.cut);
        let json_path = self.out.join(format!("hc-t{g}-o{k}.json"));
        let txt_path = self.out.join(format!("hc-t{g}-o{k}.txt"));
        let mut c = Command::new(&self.tools.search);
        c.args([
            "hybrid-check",
            "--trick",
            &g.to_string(),
            "--cut",
            &k.to_string(),
        ])
        .arg("--net")
        .arg(net.join("int.safetensors"))
        .arg("--equity")
        .arg(&self.table)
        .arg("--decisions")
        .arg(decisions)
        .arg("--worlds")
        .arg(worlds)
        .args(["--to", &self.p.golden_worlds.to_string()])
        .args(["--seed", &self.p.golden_seed.to_string()])
        .args(["--threads", &self.threads().to_string()])
        .args(["--tt-mb", &self.p.tt_mb.to_string()])
        .args(["--verify-pct", &self.p.verify_pct.to_string(), "--json"])
        .arg(&json_path);
        let progress = self.progress("hybrid check", json!({}));
        self.ctx.set_phase("chunk", progress);
        self.ctx
            .run_child_io(c, "hybrid/check.log", Some(&txt_path), None, &mut |_| {})?;
        let hc = read_json(&json_path)
            .ok_or_else(|| format!("hybrid_rung: hybrid-check wrote no {}", json_path.display()))?;
        let pass = hc["pass"].as_bool().unwrap_or(false);
        let verdict = if pass { "PASS" } else { "FAIL" };
        self.rec.insert(
            "check".into(),
            json!({ "verdict": verdict, "json": json_path.file_name().map(|n| n.to_string_lossy().into_owned()),
                    "text": txt_path.file_name().map(|n| n.to_string_lossy().into_owned()),
                    "values": hc["values"], "net_sha256": hc["net"]["sha256"],
                    "overall": hc["groups"].get(0).cloned().unwrap_or(Value::Null),
                    "time": hc["time"], "seconds": secs(t) }),
        );
        self.done("hybrid check", t)?;
        Ok(verdict.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_rung1_workflow() {
        let p = parse(&json!({})).expect("defaults");
        assert_eq!(p.cut, 4);
        assert_eq!(
            p.train,
            vec![
                LabelSet {
                    name: "random",
                    source: "random",
                    count: 3_200_000,
                    seed: 101,
                    args: vec![],
                },
                LabelSet {
                    name: "rules",
                    source: "rules",
                    count: 800_000,
                    seed: 102,
                    args: vec![],
                },
            ],
            "no harvested labels by default"
        );
        assert_eq!(
            p.val,
            LabelSet {
                name: "val",
                source: "random",
                count: 100_000,
                seed: 201,
                args: vec![],
            }
        );
        assert_eq!(
            (
                p.golden_trick,
                p.golden_count,
                p.golden_seed,
                p.golden_worlds
            ),
            (3, 400, 1, 320)
        );
        assert_eq!(p.widths, vec![1024, 512, 128]);
        assert_eq!(p.head, "outcome");
        assert_eq!(
            (p.epochs, p.batch, p.patience, p.train_seed),
            (60, 1024, 5, 1)
        );
        assert_eq!((p.lr, p.max_minutes, p.mix_weight), (0.001, 200.0, 1.0));
        assert_eq!(p.device, Device::Auto);
        assert!(!p.compare_cpu && !p.upload_shards);
        assert_eq!(
            (p.harvest_net.as_ref(), p.golden_inputs.as_ref()),
            (None, None)
        );
        assert_eq!((p.label_parts, p.overlap, p.partial_epochs), (1, true, 0));
        assert_eq!(
            p.shards()
                .iter()
                .map(|j| (j.set.name, j.part, j.file(4)))
                .collect::<Vec<_>>(),
            [
                ("random", None, "t4-random-s101.npy".to_string()),
                ("rules", None, "t4-rules-s102.npy".to_string()),
                ("val", None, "t4-val-s201.npy".to_string()),
            ],
            "one part: the training sets, then the validation shard, as before"
        );
        assert_eq!(p.to_json()["harvest_net"], Value::Null);
        assert_eq!(p.to_json()["golden"]["worlds_input"], Value::Null);
        assert_eq!(parse(&Value::Null).expect("null"), p);
        // The scheduler's keys pass.
        parse(
            &json!({ "threads": "all", "est_seconds": 1800, "equity_input": "equity",
                       "table": { "hands": 1000000, "seed": 1 } }),
        )
        .expect("scheduler keys");
    }

    /// `"check": false` trains only; the golden set's and the check's settings are refused with
    /// it, and `golden_count: 0` points to it.
    #[test]
    fn train_only_settings() {
        let p = parse(&json!({ "cut": 2, "check": false, "labels_random": 10,
                               "labels_random_seed": 511, "labels_rules": 0,
                               "val_count": 5, "val_seed": 601 }))
        .expect("train only");
        assert!(!p.check);
        assert_eq!(p.cut, 2);
        let j = p.to_json();
        assert_eq!(
            (j["check"].clone(), j["golden"].clone()),
            (json!(false), Value::Null)
        );
        assert!(parse(&json!({})).expect("defaults").check);
        let key = format!("inputs/sha256/{}", "a".repeat(64));
        for (k, v) in [
            ("golden_trick", json!(1)),
            ("golden_count", json!(32)),
            ("golden_seed", json!(1)),
            ("golden_worlds", json!(4)),
            ("golden_decisions_input", json!(key)),
            ("golden_worlds_input", json!(key)),
            ("verify_pct", json!(1)),
            ("tt_mb", json!(64)),
        ] {
            let mut x = json!({ "cut": 2, "check": false });
            x[k] = v;
            let e = parse(&x).expect_err(k);
            assert!(e.contains(&format!("{k} is set")), "{e}");
        }
        let e = parse(&json!({ "golden_count": 0 })).expect_err("no golden set");
        assert!(e.contains("\"check\": false"), "{e}");
        assert!(parse(&json!({ "check": "no" })).is_err());
    }

    /// Label parts: the validation shard first, then part 1 of each set, part 2, …, each part
    /// its share of the rows with its own seed (seed × 100 + part), the harvest last; seeds that
    /// would collide, and settings that contradict each other, are refused.
    #[test]
    fn label_parts_settings() {
        let p = parse(
            &json!({ "label_parts": 4, "labels_random": 10, "labels_rules": 3,
                               "labels_harvest": 5, "val_count": 2, "partial_epochs": 2,
                               "overlap": false }),
        )
        .expect("parts");
        assert_eq!((p.label_parts, p.overlap, p.partial_epochs), (4, false, 2));
        let jobs = p.shards();
        type Got<'a> = (&'a str, Option<(u64, u64)>, u64, u64);
        let got: Vec<Got> = jobs
            .iter()
            .map(|j| (j.set.name, j.part, j.set.count, j.set.seed))
            .collect();
        assert_eq!(
            got,
            [
                ("val", None, 2, 201),
                ("random", Some((1, 4)), 3, 10101),
                ("rules", Some((1, 4)), 1, 10201),
                ("random", Some((2, 4)), 3, 10102),
                ("rules", Some((2, 4)), 1, 10202),
                ("random", Some((3, 4)), 2, 10103),
                ("rules", Some((3, 4)), 1, 10203),
                ("random", Some((4, 4)), 2, 10104),
                ("rules", Some((4, 4)), 0, 10204),
                ("harvest", None, 5, 103),
            ]
        );
        assert_eq!(jobs[1].file(4), "t4-random-p1of4-s10101.npy");
        assert_eq!((part_count(10, 1, 4), part_count(10, 4, 4)), (3, 2));
        assert_eq!(part_seed(u64::MAX, 1), None);
        let j = p.to_json();
        assert_eq!(
            (j["label_parts"].clone(), j["overlap"].clone()),
            (json!(4), json!(false))
        );
        for (bad, why) in [
            (json!({ "label_parts": 0 }), "1 to 99"),
            (json!({ "label_parts": 100 }), "1 to 99"),
            (json!({ "partial_epochs": 1 }), "needs label_parts above 1"),
            (
                json!({ "label_parts": 2, "labels_random_seed": 2 }),
                "share seed 201",
            ),
            (json!({ "overlap": "yes" }), "overlap must be"),
            (
                json!({ "label_parts": 2, "labels_random_seed": u64::MAX }),
                "too large",
            ),
        ] {
            let e = parse(&bad).expect_err(why);
            assert!(e.contains(why), "{bad}: {e}");
        }
    }

    #[test]
    fn settings_are_read() {
        let p = parse(&json!({
            "cut": 3, "labels_random": 1000, "labels_random_seed": 301, "labels_rules": 0,
            "val_count": 50, "val_seed": 302, "golden_trick": 2, "golden_count": 8,
            "golden_seed": 1, "golden_worlds": 4, "widths": "64, 32", "head": "money4",
            "epochs": 2, "batch": 256, "lr": 0.0005, "patience": 1, "train_seed": 7,
            "max_minutes": 0, "device": "cpu", "train_threads": 3, "compare_cpu": true,
            "upload_shards": true, "tt_mb": 16, "verify_pct": 0, "start_hands": 500,
        }))
        .expect("settings");
        assert_eq!(p.cut, 3);
        assert_eq!(p.train.len(), 1, "a count of 0 skips the rules shard");
        assert_eq!((p.train[0].count, p.train[0].seed), (1000, 301));
        assert_eq!((p.val.count, p.val.seed), (50, 302));
        assert_eq!(p.widths_arg(), "64,32");
        assert_eq!((p.device, p.train_threads), (Device::Cpu, Some(3)));
        assert!(p.compare_cpu && p.upload_shards);
        assert_eq!((p.tt_mb, p.verify_pct, p.start_hands), (16, 0.0, 500));
        let j = p.to_json();
        assert_eq!(j["train"]["random"]["seed"], 301);
        assert_eq!(j["net"]["widths"], json!([64, 32]));

        // Harvested labels: after the random-play and rules-v1 shards, with their settings.
        let p = parse(&json!({
            "labels_harvest": 4_000_000, "labels_harvest_seed": 104,
            "harvest_cap_per_decision": 24, "harvest_positions": "0.4,0.4,0.2",
        }))
        .expect("harvest settings");
        let names: Vec<&str> = p.train.iter().map(|s| s.name).collect();
        assert_eq!(names, ["random", "rules", "harvest"]);
        assert_eq!(
            p.train[2],
            LabelSet {
                name: "harvest",
                source: "harvest",
                count: 4_000_000,
                seed: 104,
                args: vec![
                    "--harvest-cap".into(),
                    "24".into(),
                    "--harvest-positions".into(),
                    "0.4,0.4,0.2".into()
                ],
            }
        );
        let j = p.to_json();
        assert_eq!(j["train"]["harvest"]["count"], 4_000_000);
        assert_eq!(j["train"]["harvest"]["args"][3], "0.4,0.4,0.2");
        assert!(j["train"]["random"].get("args").is_none());
        // Stored inputs: the harvest's network and a golden set, as keys or manifest values.
        let (n, d, w) = ("a1".repeat(32), "b2".repeat(32), "C3".repeat(32));
        let p = parse(&json!({
            "cut": 3, "labels_harvest": 1000, "harvest_net": format!("inputs/sha256/{n}"),
            "golden_decisions_input": format!("sha256:{d}"),
            "golden_worlds_input": format!("inputs/sha256/{w}"), "verify_pct": 2,
        }))
        .expect("stored inputs");
        let w = w.to_lowercase();
        assert_eq!(p.harvest_net.as_deref(), Some(n.as_str()));
        assert_eq!(p.golden_inputs, Some((d.clone(), w.clone())));
        assert_eq!(p.golden_trick, 2);
        let j = p.to_json();
        assert_eq!(j["harvest_net"], format!("inputs/sha256/{n}"));
        assert_eq!(j["golden"]["decisions_input"], format!("inputs/sha256/{d}"));
        assert_eq!(j["golden"]["worlds_input"], format!("inputs/sha256/{w}"));
        // The race harvest: after the harvest shard, with its settings (defaults: classes 2 and
        // 4 up to 256 a decision, stratum 4, the harvest's position shares, no other classes).
        let p = parse(&json!({
            "labels_harvest": 10, "harvest_positions": "0.3,0.55,0.15",
            "labels_race": 20, "labels_race_seed": 155,
        }))
        .expect("race settings");
        let names: Vec<&str> = p.train.iter().map(|s| s.name).collect();
        assert_eq!(names, ["random", "rules", "harvest", "race"]);
        assert_eq!(
            p.train[3],
            LabelSet {
                name: "race",
                source: "harvest",
                count: 20,
                seed: 155,
                args: [
                    "--harvest-cap",
                    "0",
                    "--harvest-positions",
                    "0.3,0.55,0.15",
                    "--harvest-strata",
                    "4",
                    "--harvest-race-classes",
                    "2,4",
                    "--harvest-race-cap",
                    "256"
                ]
                .map(String::from)
                .to_vec(),
            }
        );
        let j = p.to_json();
        assert_eq!(
            j["race"],
            json!({ "classes": [2, 4], "cap": 256, "strata": [4],
                    "positions": "0.3,0.55,0.15", "other_cap": 0 })
        );
        assert_eq!(j["train"]["race"]["seed"], 155);
        let p = parse(&json!({
            "labels_race": 20, "race_classes": "4, 2,3", "race_cap": 64, "race_strata": "4,2",
            "race_positions": "0.5,0.5,0", "race_other_cap": 8,
        }))
        .expect("race settings given");
        let r = p.race.as_ref().expect("race");
        assert_eq!(
            (
                r.classes.as_slice(),
                r.cap,
                r.strata.as_slice(),
                r.other_cap
            ),
            ([2, 3, 4].as_slice(), 64, [2, 4].as_slice(), 8)
        );
        assert_eq!(
            p.train.last().expect("race").args,
            [
                "--harvest-cap",
                "8",
                "--harvest-positions",
                "0.5,0.5,0",
                "--harvest-strata",
                "2,4",
                "--harvest-race-classes",
                "2,3,4",
                "--harvest-race-cap",
                "64"
            ]
        );
        assert_eq!(p.train.last().expect("race").seed, 105);
        assert_eq!(parse(&json!({})).expect("defaults").race, None);
        assert_eq!(
            parse(&json!({})).expect("defaults").to_json()["race"],
            Value::Null
        );
        // A race harvest alone is a training set, and takes the harvest's network.
        let n = "d4".repeat(32);
        let p = parse(
            &json!({ "labels_random": 0, "labels_rules": 0, "labels_race": 5,
                               "harvest_net": format!("inputs/sha256/{n}") }),
        )
        .expect("race only");
        assert_eq!(
            (p.train.len(), p.harvest_net.as_deref()),
            (1, Some(n.as_str()))
        );
        // Defaults of the harvest's settings; a harvest alone is a training set.
        let p = parse(&json!({ "labels_random": 0, "labels_rules": 0, "labels_harvest": 10 }))
            .expect("harvest only");
        assert_eq!(p.train.len(), 1);
        assert_eq!(p.train[0].seed, 103);
        assert_eq!(
            p.train[0].args,
            [
                "--harvest-cap",
                "16",
                "--harvest-positions",
                "0.25,0.25,0.5"
            ]
        );
    }

    #[test]
    fn bad_settings_are_refused() {
        let bad = |v: Value, says: &str| {
            let e = parse(&v).expect_err(&v.to_string());
            assert!(e.contains(says), "{v}: {e}");
        };
        bad(json!({ "cutt": 4 }), "unknown setting \"cutt\"");
        bad(
            json!({ "golden": { "count": 400 } }),
            "unknown setting \"golden\"",
        );
        bad(json!({ "cut": 6 }), "cut 6 must be 2..=5");
        bad(json!({ "cut": 1 }), "cut 1 must be 2..=5");
        bad(
            json!({ "cut": 3, "golden_trick": 3 }),
            "golden_trick 3 must be 1..3",
        );
        bad(json!({ "golden_trick": 0 }), "golden_trick 0");
        bad(json!({ "labels_random_seed": 1 }), "golden sets' seed");
        bad(json!({ "val_seed": 101 }), "share seed 101");
        bad(json!({ "labels_rules_seed": 101 }), "share seed 101");
        bad(
            json!({ "labels_random": 0, "labels_rules": 0 }),
            "no training labels",
        );
        bad(
            json!({ "labels_harvest": 5, "labels_harvest_seed": 1 }),
            "golden sets' seed",
        );
        bad(
            json!({ "labels_harvest": 5, "labels_harvest_seed": 102 }),
            "share seed 102",
        );
        bad(
            json!({ "labels_harvest": 5, "val_seed": 103 }),
            "share seed 103",
        );
        bad(
            json!({ "harvest_cap_per_decision": 0 }),
            "harvest_cap_per_decision must be at least 1",
        );
        bad(json!({ "harvest_positions": "0,0,0" }), "harvest_positions");
        bad(json!({ "harvest_positions": "0.5,0.5" }), "three numbers");
        bad(
            json!({ "harvest_positions": [0.4, 0.4, 0.2] }),
            "harvest_positions must be a string",
        );
        bad(
            json!({ "labels_harvest": "lots" }),
            "labels_harvest must be a whole number",
        );
        bad(
            json!({ "harvest_cap": 16 }),
            "unknown setting \"harvest_cap\"",
        );
        bad(json!({ "val_count": 0 }), "val_count");
        bad(json!({ "widths": "1024,,128" }), "widths");
        bad(json!({ "widths": "" }), "widths");
        bad(json!({ "widths": 1024 }), "widths must be a string");
        bad(json!({ "head": "values" }), "head \"values\"");
        bad(json!({ "device": "gpu" }), "device \"gpu\"");
        bad(json!({ "epochs": 0 }), "epochs and batch");
        bad(json!({ "lr": 0 }), "lr must be above 0");
        bad(json!({ "lr": "fast" }), "lr must be a number");
        bad(json!({ "max_minutes": -1 }), "max_minutes");
        bad(json!({ "train_threads": 0 }), "train_threads");
        bad(
            json!({ "compare_cpu": "yes" }),
            "compare_cpu must be true or false",
        );
        bad(json!({ "golden_count": 0 }), "golden_count");
        bad(
            json!({ "labels_random": -5 }),
            "labels_random must be a whole number",
        );
        bad(json!({ "verify_pct": 150 }), "verify_pct");
        bad(json!([1, 2]), "must be an object");
        let key = format!("inputs/sha256/{}", "ab".repeat(32));
        bad(
            json!({ "harvest_net": key }),
            "harvest_net drives the harvest's search: it needs labels_harvest or labels_race above 0",
        );
        bad(
            json!({ "labels_harvest": 5, "harvest_net": "o3/int.safetensors" }),
            "harvest_net \"o3/int.safetensors\" must be a stored input's key",
        );
        bad(
            json!({ "labels_harvest": 5,
                    "harvest_net": format!("inputs/sha256/{}", "ab".repeat(31)) }),
            "must be a stored input's key",
        );
        bad(
            json!({ "labels_harvest": 5, "harvest_net": 7 }),
            "harvest_net must be a string",
        );
        bad(
            json!({ "race_classes": "2,4" }),
            "race_classes is set but labels_race is 0",
        );
        bad(
            json!({ "labels_race_seed": 155 }),
            "labels_race_seed is set but labels_race is 0",
        );
        bad(
            json!({ "labels_race": 5, "race_classes": "2,5" }),
            "race_classes",
        );
        bad(
            json!({ "labels_race": 5, "race_classes": "2,2" }),
            "given twice",
        );
        bad(
            json!({ "labels_race": 5, "race_classes": 4 }),
            "race_classes must be a string",
        );
        bad(
            json!({ "labels_race": 5, "race_cap": 0 }),
            "race_cap must be at least 1",
        );
        bad(
            json!({ "labels_race": 5, "race_strata": "0" }),
            "race_strata",
        );
        bad(
            json!({ "labels_race": 5, "race_strata": "4,4" }),
            "given twice",
        );
        bad(
            json!({ "labels_race": 5, "race_positions": "0,0,0" }),
            "race_positions",
        );
        bad(
            json!({ "labels_race": 5, "race_other_cap": -1 }),
            "race_other_cap must be a whole number",
        );
        bad(
            json!({ "labels_race": 5, "labels_race_seed": 103, "labels_harvest": 5 }),
            "share seed 103",
        );
        bad(
            json!({ "labels_race": 5, "labels_race_seed": 1 }),
            "golden sets' seed",
        );
        bad(
            json!({ "labels_random": 0, "labels_rules": 0, "labels_race": 0 }),
            "no training labels",
        );
        bad(
            json!({ "golden_decisions_input": key }),
            "golden_decisions_input and golden_worlds_input go together",
        );
        bad(
            json!({ "golden_worlds_input": key }),
            "golden_decisions_input and golden_worlds_input go together",
        );
        bad(
            json!({ "golden_decisions_input": key, "golden_worlds_input": key, "verify_pct": 0.5 }),
            "verify_pct must be at least 1",
        );
        bad(
            json!({ "golden_decisions_input": key, "golden_worlds_input": "worlds.jsonl" }),
            "golden_worlds_input \"worlds.jsonl\" must be a stored input's key",
        );
    }

    /// The committed manifests (docs/vast/m3-hybrid-*.manifest.json) are runnable: the runner
    /// accepts them (their hybrid settings included) and they ask the launcher for one GPU.
    #[test]
    fn committed_manifests_parse() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/vast");
        for (name, cut, labels, golden) in [
            ("m3-hybrid-rung1b", 4, 4_000_000, (3, 400, 320)),
            ("m3-hybrid-smoke", 4, 500_000, (3, 32, 32)),
            ("m3-hybrid-harvest", 4, 12_000_000, (3, 400, 320)),
        ] {
            let path = dir.join(format!("{name}.manifest.json"));
            let v: Value = serde_json::from_str(&fs::read_to_string(&path).expect("read"))
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let m = crate::sched::parse_v2(&v).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(m.streams.len(), 1, "{name}");
            let s = &m.streams[0];
            assert_eq!(
                (s.kind.as_str(), s.threads_all),
                ("hybrid_rung", true),
                "{name}"
            );
            let p = parse(&s.params).expect("settings");
            assert_eq!(p.cut, cut, "{name}");
            assert_eq!(
                p.train.iter().map(|s| s.count).sum::<u64>(),
                labels,
                "{name}"
            );
            assert_eq!(
                (p.golden_trick, p.golden_count, p.golden_worlds),
                golden,
                "{name}"
            );
            assert_eq!(p.golden_seed, 1, "{name}");
            assert_eq!(
                (p.widths_arg().as_str(), p.head.as_str()),
                ("1024,512,128", "outcome")
            );
            assert_eq!(v["machine"]["gpus"], 1, "{name}");
            assert!(
                v.get("bin_sha").is_none(),
                "{name}: the launcher sets bin_sha"
            );
            if name == "m3-hybrid-harvest" {
                let h = p
                    .train
                    .iter()
                    .find(|s| s.name == "harvest")
                    .expect("harvest");
                assert_eq!((h.count, h.seed), (4_000_000, 103));
                assert_eq!(
                    h.args,
                    ["--harvest-cap", "16", "--harvest-positions", "0.4,0.4,0.2"]
                );
            }
        }
        // The lead-4 proposal (docs/m3-hybrid/rung1.md, "Why seats 3-4 fail"): the lead-3c mix
        // with half of the harvest replaced by a race harvest, generated golden set.
        let path = dir.join("m3-hybrid-lead-4.manifest.json");
        let v: Value =
            serde_json::from_str(&fs::read_to_string(&path).expect("read")).expect("json");
        let m = crate::sched::parse_v2(&v).expect("runnable");
        let s = &m.streams[0];
        assert_eq!((s.kind.as_str(), s.threads_all), ("hybrid_rung", true));
        let p = parse(&s.params).expect("settings");
        assert_eq!((p.cut, p.check, p.patience), (4, true, 6));
        assert_eq!(
            p.train
                .iter()
                .map(|s| (s.name, s.count, s.seed))
                .collect::<Vec<_>>(),
            [
                ("random", 16_000_000, 151),
                ("rules", 4_000_000, 152),
                ("harvest", 10_000_000, 154),
                ("race", 10_000_000, 155)
            ]
        );
        assert_eq!(
            p.train[2].args,
            [
                "--harvest-cap",
                "64",
                "--harvest-positions",
                "0.3,0.55,0.15"
            ]
        );
        assert_eq!(
            p.train[3].args,
            [
                "--harvest-cap",
                "0",
                "--harvest-positions",
                "0.3,0.55,0.15",
                "--harvest-strata",
                "4",
                "--harvest-race-classes",
                "2,4",
                "--harvest-race-cap",
                "256"
            ]
        );
        assert_eq!(
            (
                p.golden_trick,
                p.golden_count,
                p.golden_worlds,
                p.golden_seed,
                p.golden_inputs.is_none()
            ),
            (3, 400, 320, 1, true)
        );
        assert_eq!((p.val.count, p.val.seed), (100_000, 201));
        assert_eq!(
            (p.widths_arg().as_str(), p.head.as_str()),
            ("1024,512,128", "outcome")
        );
        assert_eq!(v["machine"]["gpus"], 1);
        assert!(v.get("bin_sha").is_none());
        // The trick-2 (O_2) round-1 proposal: train only (its check is the trick-1 check on the
        // saved worlds, `hybrid_t1check`), runnable as it is.
        let path = dir.join("m3-hybrid-t2-round1.manifest.json");
        let v: Value =
            serde_json::from_str(&fs::read_to_string(&path).expect("read")).expect("json");
        let m = crate::sched::parse_v2(&v).expect("runnable");
        let s = &m.streams[0];
        assert_eq!((s.kind.as_str(), s.threads_all), ("hybrid_rung", true));
        let p = parse(&s.params).expect("settings");
        assert_eq!((p.cut, p.check), (2, false));
        assert_eq!(
            p.train
                .iter()
                .map(|s| (s.name, s.count, s.seed))
                .collect::<Vec<_>>(),
            [("random", 1_000_000, 511), ("rules", 250_000, 512)]
        );
        assert_eq!((p.val.count, p.val.seed), (50_000, 601));
        assert_eq!(
            (p.widths_arg().as_str(), p.head.as_str()),
            ("1024,512,128", "outcome")
        );
        assert_eq!(v["machine"]["gpus"], 1);
        assert!(v.get("bin_sha").is_none());
        // The rung-2 (O_3) round-2 proposal: its three stored inputs are placeholders until
        // round 1 has run and the golden files are uploaded, so it is refused as it is; filled
        // in, it is a runnable trick-3 rung reusing the stored trick-2 golden set.
        let path = dir.join("m3-hybrid-t3-round2.manifest.json");
        let text = fs::read_to_string(&path).expect("read");
        let v: Value = serde_json::from_str(&text).expect("json");
        let e = crate::sched::parse_v2(&v).expect_err("placeholders");
        assert!(
            e.contains("harvest_net \"inputs/sha256/REPLACE_WITH"),
            "{e}"
        );
        let filled = [
            ("REPLACE_WITH_THE_ROUND_1_O3_INT_SAFETENSORS_SHA256", "a"),
            ("REPLACE_WITH_THE_TRICK_2_GOLDEN_DECISIONS_SHA256", "b"),
            ("REPLACE_WITH_THE_TRICK_2_GOLDEN_WORLDS_SHA256", "c"),
        ]
        .iter()
        .fold(text.clone(), |t, (from, x)| {
            assert_eq!(t.matches(from).count(), 1, "{from}");
            t.replace(from, &x.repeat(64))
        });
        assert!(!filled.contains("REPLACE"));
        let v: Value = serde_json::from_str(&filled).expect("json");
        let m = crate::sched::parse_v2(&v).expect("filled in");
        let p = parse(&m.streams[0].params).expect("settings");
        assert_eq!(p.cut, 3);
        assert_eq!(
            (
                p.golden_trick,
                p.golden_count,
                p.golden_worlds,
                p.golden_seed
            ),
            (2, 400, 320, 1)
        );
        assert_eq!(p.train.iter().map(|s| s.count).sum::<u64>(), 4_000_000);
        assert_eq!(p.harvest_net, Some("a".repeat(64)));
        assert_eq!(p.golden_inputs, Some(("b".repeat(64), "c".repeat(64))));
        assert_eq!(p.verify_pct, 1.0);
        assert_eq!(
            (p.widths_arg().as_str(), p.head.as_str()),
            ("1024,512,128", "outcome")
        );
        assert_eq!(v["machine"]["gpus"], 1);
        assert!(v.get("bin_sha").is_none());

        // A typo in a hybrid stream stops the manifest at once.
        let bad = json!({ "format": 2, "run_id": "x", "run_seed": 1, "streams": [
            { "name": "rung", "priority": 1, "kind": "hybrid_rung", "chunks": 1,
              "params": { "threads": "all", "epoch": 3 } }] });
        let e = crate::sched::parse_v2(&bad).expect_err("typo");
        assert!(
            e.contains("stream `rung`: hybrid_rung: unknown setting \"epoch\""),
            "{e}"
        );
    }

    /// A `Rung` on fake programs, for the steps' own tests.
    #[cfg(unix)]
    fn test_rung<'a>(ctx: &'a mut Ctx, p: &'a Params, tools: &'a Tools, root: &Path) -> Rung<'a> {
        let out = root.join("out").join("hybrid");
        let work = root.join("hybrid-work");
        fs::create_dir_all(&out).expect("mkdir");
        fs::create_dir_all(&work).expect("mkdir");
        Rung {
            ctx,
            p,
            tools,
            table: root.join("equity-v0.npy"),
            out,
            work,
            rec: Map::new(),
            warnings: Vec::new(),
            steps: Vec::new(),
            t0: Instant::now(),
            python: PathBuf::new(),
            device: String::new(),
            inputs: BTreeMap::new(),
            bg_train: None,
            beside: Vec::new(),
        }
    }

    /// `concurrent`: steps at once within the thread budget, each recorded with its share and
    /// times, the heartbeat naming them all with each one's progress; one failing stops the
    /// others at once; a budget over the machine's threads is refused.
    #[cfg(unix)]
    #[test]
    fn concurrent_steps() {
        let base = std::env::temp_dir().join(format!("hybrid-conc-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).expect("mkdir");
        let p = parse(&json!({})).expect("defaults");
        let tools = Tools {
            search: PathBuf::from("search"),
            uv: PathBuf::from("uv"),
            python_dir: base.clone(),
            nvidia_smi: PathBuf::from("nvidia-smi"),
            remote: None,
        };
        let sh = |script: String| {
            let mut c = Command::new("sh");
            c.args(["-c", &script]);
            c
        };
        let step =
            |name: &str, script: String, threads: u64, prog: Option<PathBuf>| ConcurrentStep {
                step: name.to_string(),
                cmd: sh(script),
                threads,
                log: format!("hybrid/{name}.log"),
                progress: prog,
                prefix: name.replace(' ', "_"),
                limit: None,
            };
        let mut ctx = Ctx::local("hybrid-test", &base, 4);
        let mut r = test_rung(&mut ctx, &p, &tools, &base);
        let prog = base.join("audit.progress.json");
        let ran = r
            .concurrent(vec![
                step(
                    "audit shard",
                    format!(
                        "echo '{{\"rows_done\": 5, \"rows_total\": 9}}' > '{}'; sleep 0.6",
                        prog.display()
                    ),
                    3,
                    Some(prog.clone()),
                ),
                step("golden slices", "sleep 0.3".into(), 1, None),
            ])
            .expect("both run");
        assert_eq!(ran.len(), 2);
        assert!(ran.iter().all(|(a, b)| b > a));
        let steps: Vec<(&str, u64)> = r
            .steps
            .iter()
            .map(|s| {
                (
                    s["step"].as_str().unwrap_or(""),
                    s["threads"].as_u64().unwrap_or(0),
                )
            })
            .collect();
        assert_eq!(steps, [("audit shard", 3), ("golden slices", 1)]);
        assert_eq!(
            r.rec["concurrent"][0]["steps"].as_array().map(Vec::len),
            Some(2)
        );
        let hb = &r.ctx.progress;
        assert_eq!(hb["step"], "audit shard + golden slices", "{hb}");
        assert_eq!(
            (
                hb["audit_shard_rows_done"].clone(),
                hb["audit_shard_rows_total"].clone()
            ),
            (json!(5), json!(9)),
            "{hb}"
        );
        assert!(r.beside.is_empty() && r.ctx.bg.is_empty());
        // One fails: the other is stopped at once.
        let marker = base.join("finished");
        let t0 = Instant::now();
        let e = r
            .concurrent(vec![
                step(
                    "long",
                    format!("sleep 5; touch '{}'", marker.display()),
                    2,
                    None,
                ),
                step("broken", "sleep 0.2; exit 4".into(), 2, None),
            ])
            .expect_err("one failed");
        assert!(e.contains("step broken (run beside others) exited"), "{e}");
        assert!(t0.elapsed() < Duration::from_secs(5));
        assert!(r.beside.is_empty() && r.ctx.bg.is_empty());
        std::thread::sleep(Duration::from_millis(200));
        assert!(!marker.exists());
        // Over the machine's threads: refused before anything starts.
        let e = r
            .concurrent(vec![
                step("a", "true".into(), 3, None),
                step("b", "true".into(), 2, None),
            ])
            .expect_err("5 threads of 4");
        assert!(
            e.contains("ask for 5 threads at once, more than the machine's 4"),
            "{e}"
        );
        let _ = fs::remove_dir_all(&base);
    }

    /// The trainer's progress joins another step's heartbeat (`chunk_steps` shows both).
    #[test]
    fn training_progress_joins_another_steps() {
        let d = std::env::temp_dir().join(format!("hybrid-merge-{}", std::process::id()));
        fs::create_dir_all(&d).expect("mkdir");
        let prog = d.join("p.json");
        let mut p = json!({ "step": "golden worlds", "worlds_done": 3, "worlds_total": 9 });
        merge_train(&mut p, &prog);
        assert_eq!(
            p["step"], "golden worlds + train",
            "no file yet: the step is still named"
        );
        fs::write(
            &prog,
            r#"{"step": "train", "epoch": 2, "batches_done": 5, "device": "cuda"}"#,
        )
        .expect("write");
        merge_train(&mut p, &prog);
        assert_eq!(p["step"], "golden worlds + train");
        assert_eq!(
            (p["train_epoch"].clone(), p["train_batches_done"].clone()),
            (json!(2), json!(5))
        );
        assert_eq!(
            (p["worlds_done"].clone(), p.get("train_device")),
            (json!(3), None)
        );
        let mut t = json!({ "step": "train" });
        merge_train(&mut t, &prog);
        assert_eq!(t["step"], "train");
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn label_progress_reads_the_log() {
        let d = std::env::temp_dir().join(format!("hybrid-lp-{}", std::process::id()));
        fs::create_dir_all(&d).expect("mkdir");
        let log = d.join("labels.log");
        assert_eq!(label_progress(&log), None);
        fs::write(
            &log,
            "labels: trick 4, 3200000 rows from random, seed 101, 384 threads (table and start states in 2.8 s)\n\
             labels: 120000 of 3200000 rows, 20 s\nlabels: 250000 of 3200000 rows, 40 s\n",
        )
        .expect("write");
        assert_eq!(label_progress(&log), Some((250_000, 3_200_000)));
        let _ = fs::remove_dir_all(&d);
    }

    /// A fake program (a shell script) `name` in `bin`.
    #[cfg(unix)]
    fn fake_script(bin: &Path, name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let p = bin.join(name);
        fs::write(&p, format!("#!/bin/sh\n{body}")).expect("write");
        fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).expect("chmod");
        p
    }

    /// The fake programs of the step tests, in a fresh directory `hybrid-<tag>-<pid>`.
    #[cfg(unix)]
    struct Fakes {
        base: PathBuf,
        bin: PathBuf,
        /// Every call of a fake program, one line each.
        calls: PathBuf,
        search: PathBuf,
        uv: PathBuf,
        py: PathBuf,
        fake_py: PathBuf,
        gpu: PathBuf,
        table: PathBuf,
        /// The SHA-256 the fake label sidecars record.
        sha: String,
    }

    /// A fake `search` (writes each output and a sidecar), a fake `uv` (creates the environment
    /// with a fake `python`), a fake `python` (the device check and the trainer), a fake
    /// `nvidia-smi` and a fake equity table.
    #[cfg(unix)]
    fn fakes(tag: &str) -> Fakes {
        let base = std::env::temp_dir().join(format!("hybrid-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let bin = base.join("bin");
        let py = base.join("python-nn");
        fs::create_dir_all(&bin).expect("mkdir");
        fs::create_dir_all(py.join("fortyone_nn")).expect("mkdir");
        fs::write(py.join("pyproject.toml"), "[project]\n").expect("write");
        fs::write(py.join("uv.lock"), "version = 1\n").expect("write");
        let script = |name: &str, body: &str| fake_script(&bin, name, body);
        let calls = base.join("calls.log");
        let sha = crate::bundle::sha256_hex(b"npy\n");
        // `arg NAME "$@"` prints the value after NAME.
        let arg = r#"arg() { n=$1; shift; while [ $# -gt 0 ]; do [ "$1" = "$n" ] && { echo "$2"; return; }; shift; done; }"#;
        let search = script(
            "search",
            &format!(
                r#"{arg}
echo "search $*" >> '{calls}'
out=$(arg --out "$@")
case $1 in
  labels) echo "labels: 3 of 7 rows, 1 s" >&2; echo npy > "$out"
          echo '{{"shard_sha256": "{sha}", "count": 7}}' > "$out.json";;
  decisions) echo '{{"id": 0}}' > "$out";;
  worlds) echo '{{"id": 0, "method": "exact"}}' > "$out"
          echo '{{"worlds_done": 1, "worlds_total": 1}}' > "$(arg --progress "$@")";;
  hybrid-check) echo '{{"pass": false, "groups": [{{"group": "all", "summary": {{"mean_loss": 0.0002}}}}], "net": {{"sha256": "abc"}}, "values": {{"hybrid": "2+O4:abc"}}}}' > "$(arg --json "$@")"
          echo "standard: FAIL";;
esac
"#,
                calls = calls.display()
            ),
        );
        let fake_py = script(
            "fake-python",
            &format!(
                r#"{arg}
echo "python $*" >> '{calls}'
case "$2" in
  fortyone_nn.device)
    if [ -n "$FAKE_GPU" ]; then echo '{{"device": "cuda", "gpu_name": "RTX A2000", "reason": "every GPU check passed"}}' > "$(arg --json "$@")"
    else echo '{{"device": "cpu", "reason": "nvidia-smi is missing or failed: no GPU in this container"}}' > "$(arg --json "$@")"; fi;;
  fortyone_nn.train)
    out=$(arg --out "$@"); mkdir -p "$out"
    echo "int $(arg --device "$@")" > "$out/int.safetensors"; echo float > "$out/float.safetensors"
    echo '{{"history": [{{"val_loss": 1.5}}, {{"val_loss": 1.25}}], "best_epoch": 1, "float": {{}}, "int": {{"error": {{"sd": 0.01}}}}, "minutes": 0.1, "device": {{}}}}' > "$out/report.json"
    echo '{{"step": "done", "epoch": 1}}' > "$(arg --progress "$@")";;
esac
"#,
                calls = calls.display()
            ),
        );
        let uv = script(
            "uv",
            &format!(
                r#"echo "uv $*" >> '{calls}'
case $1 in
  --version) echo "uv 0.13.0";;
  sync) mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"; cp '{py}' "$UV_PROJECT_ENVIRONMENT/bin/python";;
esac
"#,
                calls = calls.display(),
                py = fake_py.display()
            ),
        );
        let gpu = script(
            "nvidia-smi",
            "echo 'NVIDIA RTX A2000, 6138 MiB, 570.86, 8.6'\n",
        );
        let table = base.join("equity-v0.npy");
        fs::write(&table, "table").expect("write");
        Fakes {
            base,
            bin,
            calls,
            search,
            uv,
            py,
            fake_py,
            gpu,
            table,
            sha,
        }
    }

    /// Step sequencing with fake programs ([`fakes`]). Without a GPU (`nvidia-smi` missing) the
    /// chunk installs the `cpu` group, warns, and trains on the CPU with every thread; with a fake
    /// GPU it installs `cu126` and trains on it with at most 8 threads, then (compare_cpu) once
    /// more on the CPU. The generated golden set is shared to the (fake) remote store.
    #[cfg(unix)]
    #[test]
    fn steps_run_in_order_with_fake_programs() {
        let Fakes {
            base,
            bin,
            calls,
            search,
            uv,
            py,
            fake_py,
            gpu,
            table,
            sha,
        } = fakes("fake");
        let script = |name: &str, body: &str| fake_script(&bin, name, body);
        let remote: Arc<dyn Remote> = Arc::new(crate::remote::DirRemote {
            root: base.join("remote"),
        });
        let params = parse(&json!({
            "labels_random": 7, "labels_rules": 5, "val_count": 3, "golden_count": 2,
            "golden_worlds": 4, "widths": "16,8", "epochs": 2, "compare_cpu": true,
            "labels_harvest": 9, "harvest_positions": "0.4,0.4,0.2",
        }))
        .expect("params");

        // 1. No GPU.
        let root = base.join("nogpu");
        let tools = Tools {
            search: search.clone(),
            uv: uv.clone(),
            python_dir: py.clone(),
            nvidia_smi: base.join("no-such-nvidia-smi"),
            remote: Some(Arc::clone(&remote)),
        };
        let mut ctx = Ctx::local("hybrid-test", &root, 4);
        let r = chunk_with(
            &mut ctx,
            &params,
            &tools,
            &table,
            json!({ "source": "test" }),
        )
        .expect("the chunk runs");
        assert_eq!(r["verdict"], "FAIL");
        assert_eq!(r["device"], "cpu");
        let w = r["warnings"].as_array().expect("warnings");
        assert!(
            w.iter().any(|w| w.as_str().is_some_and(|w| w.contains(
                "no usable GPU (nvidia-smi is missing or failed: no GPU in this container): \
                 training on the CPU with 4 threads"
            ))),
            "{r}"
        );
        assert!(
            w.iter()
                .any(|w| w.as_str().is_some_and(|w| w.contains("compare_cpu"))),
            "{r}"
        );
        let log = fs::read_to_string(&calls).expect("calls");
        let lines: Vec<&str> = log.lines().collect();
        let first = |p: &str| {
            lines
                .iter()
                .position(|l| l.starts_with(p))
                .unwrap_or_else(|| panic!("no call {p:?} in\n{log}"))
        };
        let order = [
            "uv --version",
            "uv sync --locked --no-default-groups --group cpu --python 3.14 --no-progress --project",
            "python -m fortyone_nn.device --want auto --json",
            "search labels --trick 4 --count 7 --seed 101 --source random",
            "search labels --trick 4 --count 5 --seed 102 --source rules",
            "search labels --trick 4 --count 9 --seed 103 --source harvest",
            "search labels --trick 4 --count 3 --seed 201 --source random",
            "search decisions --trick 3 --count 2 --seed 1 --allow-uneven",
            "search worlds --decisions",
            "python -m fortyone_nn.train --train",
            "search hybrid-check --trick 3 --cut 4 --net",
        ];
        let at: Vec<usize> = order.iter().map(|p| first(p)).collect();
        assert!(at.windows(2).all(|w| w[0] < w[1]), "order {at:?} in\n{log}");
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.starts_with("python -m fortyone_nn.train"))
                .count(),
            1,
            "a CPU training is not repeated for compare_cpu"
        );
        let train = lines[first("python -m fortyone_nn.train")];
        for want in [
            "t4-random-s101.npy",
            "t4-rules-s102.npy",
            "t4-harvest-s103.npy --val",
            "t4-val-s201.npy --out",
            "--head outcome --widths 16,8 --epochs 2 --batch 1024 --lr 0.001 --patience 5 --seed 1 \
             --threads 4 --max-minutes 200 --mix-weight 1 --device cpu --progress",
        ] {
            assert!(train.contains(want), "{want:?} in {train}");
        }
        let labels = lines[first("search labels --trick 4 --count 7")];
        assert!(
            labels.ends_with("--threads 4 --start-hands 20000"),
            "{labels}"
        );
        let harvest = lines[first("search labels --trick 4 --count 9")];
        assert!(
            harvest.ends_with(
                "--threads 4 --start-hands 20000 --harvest-cap 16 --harvest-positions 0.4,0.4,0.2"
            ),
            "{harvest}"
        );
        let worlds = lines[first("search worlds")];
        assert!(
            worlds.contains("--method exact --to 4 --seed 1 --out")
                && worlds.contains("exact-t3-n4.jsonl")
                && worlds.contains("--threads 4 --tt-mb 64 --progress"),
            "{worlds}"
        );
        let check = lines[first("search hybrid-check")];
        assert!(
            check.contains("net/int.safetensors")
                && check.contains("--to 4 --seed 1 --threads 4 --tt-mb 64 --verify-pct 1 --json")
                && check.contains("hc-t3-o4.json"),
            "{check}"
        );
        let out = root.join("out").join("hybrid");
        for f in [
            "rung.json",
            "device.json",
            "labels/t4-random-s101.npy.json",
            "labels/t4-rules-s102.npy.json",
            "labels/t4-harvest-s103.npy.json",
            "labels/t4-val-s201.npy.json",
            "decisions-t3.jsonl",
            "exact-t3-n4.jsonl",
            "net/int.safetensors",
            "net/report.json",
            "hc-t3-o4.json",
            "hc-t3-o4.txt",
        ] {
            assert!(out.join(f).is_file(), "out/hybrid/{f} missing");
        }
        // The shards stay outside out/.
        assert!(!out.join("labels/t4-random-s101.npy").exists());
        assert!(root.join("hybrid-work/labels/t4-random-s101.npy").is_file());
        assert_eq!(
            fs::read_to_string(out.join("hc-t3-o4.txt"))
                .expect("txt")
                .trim(),
            "standard: FAIL"
        );
        let rung = read_json(&out.join("rung.json")).expect("rung.json");
        assert_eq!(rung["check"]["verdict"], "FAIL");
        assert_eq!(rung["python_env"]["group"], "cpu");
        assert_eq!(rung["python_env"]["uv"], "uv 0.13.0");
        assert_eq!(rung["labels"][0]["shard_sha256"], sha.as_str());
        assert_eq!(rung["net"]["device"], "cpu");
        assert_eq!(rung["net"]["epochs_run"], 2);
        assert!(
            rung["net"]["int_key"]
                .as_str()
                .is_some_and(|k| k.starts_with("inputs/sha256/")),
            "{rung}"
        );
        assert_eq!(rung["params"]["golden"]["worlds"], 4);
        let steps: Vec<&str> = rung["steps"]
            .as_array()
            .expect("steps")
            .iter()
            .filter_map(|s| s["step"].as_str())
            .collect();
        assert_eq!(
            steps,
            [
                "python environment",
                "GPU check",
                "labels random (1 of 4)",
                "labels rules (2 of 4)",
                "labels harvest (3 of 4)",
                "labels val (4 of 4)",
                "golden decisions",
                "golden worlds",
                "share golden set",
                "train",
                "hybrid check"
            ]
        );
        // The generated golden set is in the remote store under its SHA-256, for reuse.
        let g = &rung["golden"];
        for (name, file) in [
            ("decisions", "decisions-t3.jsonl"),
            ("worlds", "exact-t3-n4.jsonl"),
        ] {
            let hex = crate::bundle::sha256_file(&out.join(file)).expect("sha").0;
            assert_eq!(g[format!("{name}_sha256").as_str()], hex.as_str());
            let key = format!("inputs/sha256/{hex}");
            assert_eq!(g[format!("{name}_key").as_str()], key.as_str());
            assert_eq!(
                g["reuse_params"][format!("golden_{name}_input").as_str()],
                key.as_str()
            );
            assert_eq!(
                remote.sha256(&key).expect("listed"),
                Some((hex, fs::metadata(out.join(file)).expect("file").len()))
            );
        }
        // The last heartbeat says done, with the warning.
        let hb = read_json(&root.join("heartbeat.json")).expect("heartbeat");
        assert_eq!(hb["progress"]["step"], "done");
        assert!(
            hb["progress"]["warnings"]
                .as_array()
                .is_some_and(|w| !w.is_empty())
        );

        // 2. A (fake) GPU: the cu126 group, training on it with at most 8 threads, then the CPU
        //    comparison with every thread.
        fs::write(&calls, "").expect("truncate");
        let root = base.join("gpu");
        let tools = Tools {
            nvidia_smi: gpu,
            ..tools
        };
        // The GPU machine's fake python says the device check passed (FAKE_GPU).
        let mut ctx = Ctx::local("hybrid-test", &root, 16);
        let fake_gpu_py = script(
            "fake-python-gpu",
            &format!("FAKE_GPU=1 exec '{}' \"$@\"\n", fake_py.display()),
        );
        let uv_gpu = script(
            "uv-gpu",
            &format!(
                r#"echo "uv $*" >> '{calls}'
case $1 in
  --version) echo "uv 0.13.0";;
  sync) mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"; cp '{py}' "$UV_PROJECT_ENVIRONMENT/bin/python";;
esac
"#,
                calls = calls.display(),
                py = fake_gpu_py.display()
            ),
        );
        let tools = Tools {
            uv: uv_gpu,
            ..tools
        };
        let r = chunk_with(&mut ctx, &params, &tools, &table, json!({})).expect("gpu chunk");
        assert_eq!(r["device"], "cuda");
        assert_eq!(r["gpu"], "RTX A2000");
        assert_eq!(r["warnings"], json!([]));
        let log = fs::read_to_string(&calls).expect("calls");
        assert!(
            log.contains("uv sync --locked --no-default-groups --group cu126"),
            "{log}"
        );
        let trains: Vec<&str> = log
            .lines()
            .filter(|l| l.starts_with("python -m fortyone_nn.train"))
            .collect();
        assert_eq!(trains.len(), 2, "{log}");
        // On the GPU the golden set is made while the network trains, on the threads it leaves.
        let at = |p: &str| log.find(p).unwrap_or_else(|| panic!("no {p:?} in\n{log}"));
        assert!(
            at("python -m fortyone_nn.train") < at("search decisions"),
            "{log}"
        );
        for l in log
            .lines()
            .filter(|l| l.starts_with("search decisions") || l.starts_with("search worlds"))
        {
            assert!(l.contains("--threads 8 "), "{l}");
        }
        assert!(
            log.lines()
                .find(|l| l.starts_with("search hybrid-check"))
                .is_some_and(|l| l.contains("--threads 16 ")),
            "{log}"
        );
        assert!(trains[0].contains("--threads 8 ") && trains[0].contains("--device cuda"));
        assert!(trains[1].contains("--threads 16 ") && trains[1].contains("--device cpu"));
        assert!(trains[1].contains("net-cpu"));
        let rung = read_json(&root.join("out/hybrid/rung.json")).expect("rung.json");
        let o = &rung["overlap"];
        assert_eq!(
            (
                o["golden_with_training"].clone(),
                o["train_threads"].clone(),
                o["golden_threads"].clone()
            ),
            (json!(true), json!(8), json!(8)),
            "{rung}"
        );
        assert!(o["both_s"].as_f64().is_some_and(|x| x >= 0.0));
        let steps: Vec<&str> = rung["steps"]
            .as_array()
            .expect("steps")
            .iter()
            .filter_map(|s| s["step"].as_str())
            .collect();
        assert_eq!(
            &steps[6..],
            [
                "golden decisions",
                "golden worlds",
                "share golden set",
                "train",
                "train (CPU comparison)",
                "hybrid check"
            ],
            "the same steps in the same order as without the overlap"
        );
        assert_eq!(rung["compare_cpu"]["same_int_net"], false);
        assert_eq!(
            rung["compare_cpu"]["max_abs_val_loss_difference_per_epoch"],
            0.0
        );
        assert_eq!(rung["python_env"]["group"], "cu126");

        // 3. "cuda" asked for with no GPU: refused before anything is installed.
        let tools = Tools {
            nvidia_smi: base.join("no-such-nvidia-smi"),
            ..tools
        };
        let p = parse(
            &json!({ "device": "cuda", "labels_random": 7, "labels_rules": 0,
                               "val_count": 3 }),
        )
        .expect("params");
        let mut ctx = Ctx::local("hybrid-test", &base.join("cuda"), 4);
        let e = chunk_with(&mut ctx, &p, &tools, &table, json!({})).expect_err("no GPU");
        assert!(e.contains("\"cuda\" was asked for"), "{e}");
        let _ = fs::remove_dir_all(&base);
    }

    /// Stored inputs in a dry run with fake programs ([`fakes`]) and a fake remote store: the
    /// harvest's network and a stored golden set are fetched from `inputs/sha256/<hex>` first,
    /// their SHA-256 checked, the network passed to the harvest's labels only, nothing of the
    /// golden set generated, and the stored files given to the check with its 1% recheck. A
    /// missing key, an object whose bytes are not its key's, a decision count that is not
    /// `golden_count` and no remote store each stop the chunk before anything is installed.
    /// Tools for a machine with a (fake) GPU: the fake python, preceded by `prefix` (a shell
    /// line, e.g. a slow or failing trainer), behind a fake `uv` that installs it, and `search`.
    #[cfg(unix)]
    fn gpu_tools(f: &Fakes, tag: &str, prefix: &str, search: PathBuf) -> Tools {
        let py = fake_script(
            &f.bin,
            &format!("fake-python-{tag}"),
            &format!(
                "{prefix}\nFAKE_GPU=1 exec '{}' \"$@\"\n",
                f.fake_py.display()
            ),
        );
        let uv = fake_script(
            &f.bin,
            &format!("uv-{tag}"),
            &format!(
                r#"echo "uv $*" >> '{calls}'
case $1 in
  --version) echo "uv 0.13.0";;
  sync) mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"; cp '{py}' "$UV_PROJECT_ENVIRONMENT/bin/python";;
esac
"#,
                calls = f.calls.display(),
                py = py.display()
            ),
        );
        Tools {
            search,
            uv,
            python_dir: f.py.clone(),
            nvidia_smi: f.gpu.clone(),
            remote: None,
        }
    }

    /// The golden set made beside the training: when either fails the other is stopped at once
    /// and the chunk fails with the reason (nothing is left running).
    #[cfg(unix)]
    #[test]
    fn golden_beside_training_fails_cleanly() {
        let f = fakes("overlapfail");
        let params = parse(
            &json!({ "labels_random": 7, "labels_rules": 0, "val_count": 3,
                                    "golden_count": 2, "golden_worlds": 4, "widths": "16,8",
                                    "epochs": 2 }),
        )
        .expect("params");
        let wrap = |name: &str, line: &str| {
            fake_script(
                &f.bin,
                name,
                &format!("{line}\nexec '{}' \"$@\"\n", f.search.display()),
            )
        };
        // 1. The training fails while the golden worlds are being made (5 s).
        let marker = f.base.join("worlds-finished");
        let slow = wrap(
            "search-slow-worlds",
            &format!(
                "[ \"$1\" = worlds ] && {{ sleep 5; touch '{}'; }}",
                marker.display()
            ),
        );
        let tools = gpu_tools(
            &f,
            "failing-train",
            r#"case "$2" in fortyone_nn.train) sleep 1; echo "no memory" >&2; exit 1;; esac"#,
            slow,
        );
        let mut ctx = Ctx::local("hybrid-test", &f.base.join("a"), 16);
        let t = Instant::now();
        let e = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({}))
            .expect_err("the training failed");
        assert!(e.contains("step train (run beside others) exited"), "{e}");
        assert!(
            t.elapsed() < Duration::from_secs(5),
            "the golden worlds were stopped"
        );
        assert!(!marker.exists());
        assert!(ctx.bg.is_empty());
        let log = fs::read_to_string(f.base.join("a/logs/hybrid/net.log")).expect("log");
        assert!(log.contains("no memory"), "{log}");
        // 2. The golden worlds fail while the network trains (5 s).
        let marker = f.base.join("training-finished");
        let failing = wrap(
            "search-failing-worlds",
            r#"[ "$1" = worlds ] && { echo "worlds broke" >&2; exit 1; }"#,
        );
        let tools = gpu_tools(
            &f,
            "slow-train",
            &format!(
                "case \"$2\" in fortyone_nn.train) sleep 5; touch '{}';; esac",
                marker.display()
            ),
            failing,
        );
        let mut ctx = Ctx::local("hybrid-test", &f.base.join("b"), 16);
        let t = Instant::now();
        let e = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({}))
            .expect_err("the golden worlds failed");
        assert!(e.contains("exited with") && e.contains("golden.log"), "{e}");
        assert!(
            t.elapsed() < Duration::from_secs(5),
            "the training was stopped"
        );
        assert!(ctx.bg.is_empty());
        std::thread::sleep(Duration::from_millis(300));
        assert!(!marker.exists());
        let _ = fs::remove_dir_all(&f.base);
    }

    /// Label parts on a GPU: the validation shard and part 1 first, then the training starts
    /// (watching the parts' directory, expecting every training shard) while the later parts are
    /// made on the threads it leaves.
    #[cfg(unix)]
    #[test]
    fn label_parts_stream_into_the_training() {
        let f = fakes("stream");
        let params = parse(&json!({ "cut": 2, "check": false, "label_parts": 2,
                                    "labels_random": 8, "labels_random_seed": 511,
                                    "labels_rules": 4, "labels_rules_seed": 512, "val_count": 3,
                                    "val_seed": 601, "widths": "16,8", "epochs": 2,
                                    "partial_epochs": 3 }))
        .expect("params");
        let tools = gpu_tools(&f, "stream", "", f.search.clone());
        let root = f.base.join("run");
        let mut ctx = Ctx::local("hybrid-test", &root, 16);
        let r = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("runs");
        assert_eq!(
            (r["device"].clone(), r["verdict"].clone()),
            (json!("cuda"), json!(TRAIN_ONLY))
        );
        let log = fs::read_to_string(&f.calls).expect("calls");
        let lines: Vec<&str> = log
            .lines()
            .filter(|l| l.starts_with("search") || l.starts_with("python -m fortyone_nn.train"))
            .collect();
        let want = [
            (
                "search labels --trick 2 --count 3 --seed 601 --source random",
                "--threads 16 ",
            ),
            (
                "search labels --trick 2 --count 4 --seed 51101 --source random",
                "--threads 16 ",
            ),
            (
                "search labels --trick 2 --count 2 --seed 51201 --source rules",
                "--threads 16 ",
            ),
            ("python -m fortyone_nn.train --watch-dir", "--expect 4 "),
            (
                "search labels --trick 2 --count 4 --seed 51102 --source random",
                "--threads 8 ",
            ),
            (
                "search labels --trick 2 --count 2 --seed 51202 --source rules",
                "--threads 8 ",
            ),
        ];
        assert_eq!(lines.len(), want.len(), "{log}");
        for (l, (start, has)) in lines.iter().zip(want) {
            assert!(
                l.starts_with(start) && l.contains(has),
                "{l} (want {start} … {has})"
            );
        }
        let train = lines[3];
        for has in [
            "labels/train ",
            "t2-val-s601.npy",
            "--load-threads 8 ",
            "--partial-epochs 3 ",
            "--threads 8 ",
            "--device cuda ",
        ] {
            assert!(train.contains(has), "{has} in {train}");
        }
        assert!(!train.contains("--train "), "{train}");
        let work = root.join("hybrid-work/labels");
        for p in [
            "train/t2-random-p1of2-s51101.npy",
            "train/t2-rules-p2of2-s51202.npy",
            "t2-val-s601.npy",
        ] {
            assert!(work.join(p).is_file(), "{p}");
        }
        let rung = read_json(&root.join("out/hybrid/rung.json")).expect("rung.json");
        let steps: Vec<&str> = rung["steps"]
            .as_array()
            .expect("steps")
            .iter()
            .filter_map(|s| s["step"].as_str())
            .collect();
        assert_eq!(
            steps,
            [
                "python environment",
                "GPU check",
                "labels val (1 of 5)",
                "labels random p1/2 (2 of 5)",
                "labels rules p1/2 (3 of 5)",
                "labels random p2/2 (4 of 5)",
                "labels rules p2/2 (5 of 5)",
                "train"
            ]
        );
        assert_eq!(
            rung["stream"]["train_started_after"],
            "2 of 4 shards (part 1)"
        );
        assert_eq!(rung["stream"]["label_threads_while_training"], 8);
        assert_eq!(rung["labels"][4]["part"], json!([2, 2]));
        assert_eq!(rung["labels"][4]["threads"], 8);
        assert!(rung["net"]["int_sha256"].is_string());
        // Without a GPU the same parts are made (the data never depends on the machine), then
        // the training on all of them, as before.
        fs::write(&f.calls, "").expect("truncate");
        let tools = Tools {
            nvidia_smi: f.base.join("no-such-nvidia-smi"),
            uv: f.uv.clone(),
            ..tools
        };
        let mut ctx = Ctx::local("hybrid-test", &f.base.join("cpu"), 4);
        chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("runs");
        let log = fs::read_to_string(&f.calls).expect("calls");
        let train = log
            .lines()
            .find(|l| l.starts_with("python -m fortyone_nn.train"))
            .expect("trained");
        assert!(
            train.contains("--train ")
                && train.contains("t2-random-p2of2-s51102.npy")
                && !train.contains("--watch-dir"),
            "{train}"
        );
        assert_eq!(
            log.lines()
                .filter(|l| l.starts_with("search labels"))
                .map(|l| l.split(" --out").next().unwrap_or(l))
                .collect::<Vec<_>>()
                .len(),
            5
        );
        let _ = fs::remove_dir_all(&f.base);
    }

    /// A train-only rung (`"check": false`, here at cut 2): labels, training, the network
    /// shared content-addressed, and no golden set and no hybrid check (no `search decisions`,
    /// `worlds` or `hybrid-check` call); the verdict says it was not checked. Without a remote
    /// store the network stays in the bundle, with a warning.
    #[cfg(unix)]
    #[test]
    fn train_only_runs_no_golden_set_and_no_check() {
        let f = fakes("trainonly");
        let remote: Arc<dyn Remote> = Arc::new(crate::remote::DirRemote {
            root: f.base.join("remote"),
        });
        let params = parse(&json!({
            "cut": 2, "check": false, "labels_random": 7, "labels_random_seed": 511,
            "labels_rules": 5, "labels_rules_seed": 512, "val_count": 3, "val_seed": 601,
            "widths": "16,8", "epochs": 2,
        }))
        .expect("params");
        let tools = Tools {
            search: f.search.clone(),
            uv: f.uv.clone(),
            python_dir: f.py.clone(),
            nvidia_smi: f.base.join("no-such-nvidia-smi"),
            remote: Some(Arc::clone(&remote)),
        };
        let root = f.base.join("run");
        let mut ctx = Ctx::local("hybrid-test", &root, 4);
        let r = chunk_with(&mut ctx, &params, &tools, &f.table, json!({})).expect("the chunk runs");
        assert_eq!(r["verdict"], TRAIN_ONLY);
        let log = fs::read_to_string(&f.calls).expect("calls");
        for gone in ["search decisions", "search worlds", "search hybrid-check"] {
            assert!(!log.contains(gone), "{gone} in\n{log}");
        }
        for want in [
            "search labels --trick 2 --count 7 --seed 511 --source random",
            "search labels --trick 2 --count 5 --seed 512 --source rules",
            "search labels --trick 2 --count 3 --seed 601 --source random",
            "python -m fortyone_nn.train --train",
        ] {
            assert!(log.contains(want), "{want} in\n{log}");
        }
        let out = root.join("out").join("hybrid");
        let rung = read_json(&out.join("rung.json")).expect("rung.json");
        let steps: Vec<&str> = rung["steps"]
            .as_array()
            .expect("steps")
            .iter()
            .filter_map(|s| s["step"].as_str())
            .collect();
        assert_eq!(
            steps,
            [
                "python environment",
                "GPU check",
                "labels random (1 of 3)",
                "labels rules (2 of 3)",
                "labels val (3 of 3)",
                "train"
            ]
        );
        assert_eq!(rung["check"]["verdict"], TRAIN_ONLY);
        assert_eq!(rung["params"]["golden"], Value::Null);
        assert!(out.join("labels/t2-random-s511.npy.json").is_file());
        assert!(!out.join("decisions-t1.jsonl").exists());
        // The integer network, shared under its SHA-256.
        let int = out.join("net").join("int.safetensors");
        let hex = crate::bundle::sha256_file(&int).expect("sha").0;
        let key = format!("inputs/sha256/{hex}");
        assert_eq!(rung["net"]["int_sha256"], hex.as_str());
        assert_eq!(rung["net"]["int_key"], key.as_str());
        assert_eq!(r["net_int_key"], key.as_str());
        assert_eq!(
            remote.sha256(&key).expect("listed"),
            Some((hex, fs::metadata(&int).expect("file").len()))
        );
        // No remote store: the network stays in the bundle, with a warning.
        let tools = Tools {
            remote: None,
            ..tools
        };
        let mut ctx = Ctx::local("hybrid-test", &f.base.join("local"), 4);
        let r = chunk_with(&mut ctx, &params, &tools, &f.table, json!({})).expect("runs");
        assert_eq!(r["net_int_key"], Value::Null);
        assert!(
            r["warnings"].as_array().is_some_and(|w| w.iter().any(|w| w
                .as_str()
                .is_some_and(|w| w.contains("the network was not shared")))),
            "{r}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn stored_inputs_are_fetched_checked_and_used() {
        let f = fakes("stored");
        let remote: Arc<dyn Remote> = Arc::new(crate::remote::DirRemote {
            root: f.base.join("remote"),
        });
        let store = |name: &str, bytes: &[u8]| {
            let p = f.base.join(name);
            fs::write(&p, bytes).expect("write");
            put_input(&*remote, &p, None).expect("stored").0
        };
        let net = store("int.safetensors", b"an integer O_3");
        let dec = store(
            "decisions-t2.jsonl",
            b"{\"id\":0,\"trick\":2}\n{\"id\":1,\"trick\":2}\n",
        );
        let wor = store(
            "exact-t2-n320.jsonl",
            b"{\"id\":0,\"method\":\"exact\"}\n{\"id\":1,\"method\":\"exact\"}\n",
        );
        let settings = |net: &str, golden_count: u64| {
            json!({
                "cut": 3, "labels_random": 7, "labels_rules": 0, "labels_harvest": 9,
                "harvest_net": format!("inputs/sha256/{net}"), "val_count": 3,
                "golden_count": golden_count, "golden_worlds": 4,
                "golden_decisions_input": format!("sha256:{dec}"),
                "golden_worlds_input": format!("inputs/sha256/{wor}"),
                "widths": "16,8", "epochs": 1,
            })
        };
        let tools = Tools {
            search: f.search.clone(),
            uv: f.uv.clone(),
            python_dir: f.py.clone(),
            nvidia_smi: f.base.join("no-such-nvidia-smi"),
            remote: Some(Arc::clone(&remote)),
        };
        let run = |root: &str, params: &Params, tools: &Tools| {
            fs::write(&f.calls, "").expect("truncate");
            let mut ctx = Ctx::local("hybrid-test", &f.base.join(root), 4);
            chunk_with(&mut ctx, params, tools, &f.table, json!({}))
        };

        let params = parse(&settings(&net, 2)).expect("params");
        let r = run("ok", &params, &tools).expect("the chunk runs");
        assert_eq!(r["verdict"], "FAIL");
        let log = fs::read_to_string(&f.calls).expect("calls");
        assert!(
            !log.contains("search decisions") && !log.contains("search worlds"),
            "nothing of the golden set is generated:\n{log}"
        );
        let root = f.base.join("ok");
        let inputs = std::path::absolute(root.join("hybrid-work").join("inputs")).expect("abs");
        let line = |p: &str| {
            log.lines()
                .find(|l| l.starts_with(p))
                .unwrap_or_else(|| panic!("no call {p:?} in\n{log}"))
        };
        let harvest = line("search labels --trick 3 --count 9 --seed 103 --source harvest");
        assert!(
            harvest.ends_with(&format!(
                "--harvest-cap 16 --harvest-positions 0.25,0.25,0.5 --harvest-net {}",
                inputs.join(&net).display()
            )),
            "{harvest}"
        );
        for other in [
            "search labels --trick 3 --count 7 --seed 101",
            "search labels --trick 3 --count 3 --seed 201",
        ] {
            assert!(!line(other).contains("--harvest-net"), "{other}");
        }
        let check = line("search hybrid-check --trick 2 --cut 3");
        for want in [
            format!("--decisions {}", inputs.join(&dec).display()),
            format!("--worlds {}", inputs.join(&wor).display()),
            "--to 4 --seed 1 --threads 4 --tt-mb 64 --verify-pct 1 --json".to_string(),
        ] {
            assert!(check.contains(&want), "{want:?} in {check}");
        }
        for hex in [&net, &dec, &wor] {
            assert!(inputs.join(hex).is_file());
        }
        let rung = read_json(&root.join("out/hybrid/rung.json")).expect("rung.json");
        let steps: Vec<&str> = rung["steps"]
            .as_array()
            .expect("steps")
            .iter()
            .filter_map(|s| s["step"].as_str())
            .collect();
        assert_eq!(
            steps,
            [
                "inputs",
                "python environment",
                "GPU check",
                "labels random (1 of 3)",
                "labels harvest (2 of 3)",
                "labels val (3 of 3)",
                "golden set (stored)",
                "train",
                "hybrid check"
            ]
        );
        for (name, hex) in [
            ("harvest_net", &net),
            ("golden_decisions", &dec),
            ("golden_worlds", &wor),
        ] {
            let x = &rung["inputs"][name];
            assert_eq!(x["sha256"], hex.as_str(), "{name}");
            assert_eq!(x["key"], format!("inputs/sha256/{hex}"), "{name}");
            assert_eq!(x["downloaded"], true, "{name}");
        }
        assert_eq!(rung["labels"][1]["name"], "harvest");
        assert_eq!(rung["labels"][1]["harvest_net_sha256"], net.as_str());
        assert!(rung["labels"][0].get("harvest_net_sha256").is_none());
        let g = &rung["golden"];
        assert_eq!(g["stored"], true);
        assert_eq!(
            (g["decisions_sha256"].as_str(), g["worlds_sha256"].as_str()),
            (Some(dec.as_str()), Some(wor.as_str()))
        );
        assert_eq!(g["verify_pct"], 1.0);
        assert_eq!(
            rung["params"]["harvest_net"],
            format!("inputs/sha256/{net}")
        );

        // A key that is not in the store: stopped before anything is installed.
        let missing = "0".repeat(64);
        let p = parse(&settings(&missing, 2)).expect("params");
        let e = run("missing", &p, &tools).expect_err("missing input");
        assert_eq!(
            e,
            format!("hybrid_rung: harvest_net: inputs/sha256/{missing} not found")
        );
        assert_eq!(
            fs::read_to_string(&f.calls).expect("calls"),
            "",
            "nothing ran"
        );
        // An object whose bytes are not its key's.
        let bad = "1".repeat(64);
        fs::write(
            f.base.join("remote/inputs/sha256").join(&bad),
            b"not these bytes",
        )
        .expect("write");
        let p = parse(&settings(&bad, 2)).expect("params");
        let e = run("bad", &p, &tools).expect_err("wrong bytes");
        assert!(e.contains("downloaded sha256"), "{e}");
        assert!(!f.base.join("bad/hybrid-work/inputs").join(&bad).exists());
        assert_eq!(
            fs::read_to_string(&f.calls).expect("calls"),
            "",
            "nothing ran"
        );
        // The stored decisions are not the set the settings describe.
        let p = parse(&settings(&net, 3)).expect("params");
        let e = run("count", &p, &tools).expect_err("count");
        assert!(
            e.contains("the stored golden decisions hold 2 decisions at trick 2"),
            "{e}"
        );
        assert_eq!(
            fs::read_to_string(&f.calls).expect("calls"),
            "",
            "nothing ran"
        );
        // No remote store.
        let none = Tools {
            remote: None,
            search: f.search.clone(),
            uv: f.uv.clone(),
            python_dir: f.py.clone(),
            nvidia_smi: f.base.join("no-such-nvidia-smi"),
        };
        let e = run("none", &params, &none).expect_err("no store");
        assert!(e.contains("and there is none"), "{e}");
        assert_eq!(
            fs::read_to_string(&f.calls).expect("calls"),
            "",
            "nothing ran"
        );
        let _ = fs::remove_dir_all(&f.base);
    }

    /// The race harvest with fake programs ([`fakes`]): its labels come after the harvest's and
    /// before the validation shard, with its settings and the harvest's network, the shard joins
    /// the training set after the harvest shard, and `rung.json` records its settings and both
    /// harvests' rows by race class (from their sidecars).
    #[cfg(unix)]
    #[test]
    fn race_labels_join_the_training_set() {
        let f = fakes("race");
        let remote: Arc<dyn Remote> = Arc::new(crate::remote::DirRemote {
            root: f.base.join("remote"),
        });
        let p = f.base.join("int.safetensors");
        fs::write(&p, b"an integer O_4").expect("write");
        let net = put_input(&*remote, &p, None).expect("stored").0;
        // The fake `search labels` writes no class counts; this one adds them to its sidecars.
        let search = fake_script(
            &f.bin,
            "search-race",
            &format!(
                r#"'{search}' "$@"
if [ "$1" = labels ]; then
  out=""; prev=""; for a in "$@"; do [ "$prev" = --out ] && out=$a; prev=$a; done
  echo '{{"shard_sha256": "{sha}", "count": 7, "rows_by_race_class": [0, 0, 3, 0, 4]}}' > "$out.json"
fi
"#,
                search = f.search.display(),
                sha = f.sha
            ),
        );
        let params = parse(&json!({
            "labels_random": 7, "labels_rules": 0, "labels_harvest": 9,
            "harvest_cap_per_decision": 64, "harvest_positions": "0.3,0.55,0.15",
            "labels_race": 6, "labels_race_seed": 155, "race_strata": "2,4",
            "harvest_net": format!("inputs/sha256/{net}"),
            "val_count": 3, "golden_count": 2, "golden_worlds": 4, "widths": "16,8", "epochs": 1,
        }))
        .expect("params");
        let tools = Tools {
            search,
            uv: f.uv.clone(),
            python_dir: f.py.clone(),
            nvidia_smi: f.base.join("no-such-nvidia-smi"),
            remote: Some(Arc::clone(&remote)),
        };
        let root = f.base.join("run");
        let mut ctx = Ctx::local("hybrid-test", &root, 4);
        let r = chunk_with(&mut ctx, &params, &tools, &f.table, json!({})).expect("the chunk runs");
        assert_eq!(r["verdict"], "FAIL");
        let log = fs::read_to_string(&f.calls).expect("calls");
        let lines: Vec<&str> = log.lines().collect();
        let first = |p: &str| {
            lines
                .iter()
                .position(|l| l.starts_with(p))
                .unwrap_or_else(|| panic!("no call {p:?} in\n{log}"))
        };
        let order = [
            "search labels --trick 4 --count 7 --seed 101 --source random",
            "search labels --trick 4 --count 9 --seed 103 --source harvest",
            "search labels --trick 4 --count 6 --seed 155 --source harvest",
            "search labels --trick 4 --count 3 --seed 201 --source random",
            "python -m fortyone_nn.train --train",
        ];
        let at: Vec<usize> = order.iter().map(|p| first(p)).collect();
        assert!(at.windows(2).all(|w| w[0] < w[1]), "order {at:?} in\n{log}");
        let inputs = std::path::absolute(root.join("hybrid-work").join("inputs")).expect("abs");
        let race = lines[first("search labels --trick 4 --count 6 --seed 155")];
        assert!(
            race.ends_with(&format!(
                "--threads 4 --start-hands 20000 --harvest-cap 0 --harvest-positions 0.3,0.55,0.15 \
                 --harvest-strata 2,4 --harvest-race-classes 2,4 --harvest-race-cap 256 \
                 --harvest-net {}",
                inputs.join(&net).display()
            )),
            "{race}"
        );
        let harvest = lines[first("search labels --trick 4 --count 9 --seed 103")];
        assert!(harvest.contains("--harvest-net"), "{harvest}");
        assert!(!lines[first("search labels --trick 4 --count 7")].contains("--harvest-net"));
        let train = lines[first("python -m fortyone_nn.train")];
        assert!(
            train.contains("t4-harvest-s103.npy") && train.contains("t4-race-s155.npy --val"),
            "{train}"
        );
        let out = root.join("out").join("hybrid");
        assert!(out.join("labels/t4-race-s155.npy.json").is_file());
        let rung = read_json(&out.join("rung.json")).expect("rung.json");
        let names: Vec<&str> = rung["labels"]
            .as_array()
            .expect("labels")
            .iter()
            .filter_map(|l| l["name"].as_str())
            .collect();
        assert_eq!(names, ["random", "harvest", "race", "val"]);
        assert_eq!(
            rung["labels"][2]["rows_by_race_class"],
            json!([0, 0, 3, 0, 4])
        );
        assert_eq!(rung["labels"][2]["harvest_net_sha256"], net.as_str());
        assert_eq!(rung["params"]["race"]["strata"], json!([2, 4]));
        assert_eq!(rung["params"]["train"]["race"]["seed"], 155);
        let steps: Vec<&str> = rung["steps"]
            .as_array()
            .expect("steps")
            .iter()
            .filter_map(|s| s["step"].as_str())
            .collect();
        assert!(steps.contains(&"labels race (3 of 4)"), "{steps:?}");
        let _ = fs::remove_dir_all(&f.base);
    }
}
