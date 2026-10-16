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
//! other key is refused ([`KEYS`]). Also: `mix_weight` (1.0), `start_hands` (20000, for the label
//! roots and the golden decisions), `tt_mb` (64, the solver table per thread of the golden worlds
//! and the check), `verify_pct` (1, the check's recomputed sample of stored exact worlds),
//! `train_threads` (CPU threads of the trainer: all usable threads on the CPU, at most 8 beside
//! a GPU), `compare_cpu` (false; after a GPU training, train once more on the CPU with the same
//! settings and record both, for small checks), `upload_shards` (false; see step 4), `table`
//! (the equity table, as `search_measure`'s), and the scheduler's `threads`, `est_seconds`,
//! `equity_input`.
//!
//! Steps (each a child process with its own log; the heartbeat names the step and its progress):
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
//! 4. **Labels** (`search labels`, all usable threads): the random-play and rules-v1 training
//!    shards (a count of 0 skips one) and the validation shard (random play), at the start of
//!    trick `cut`. The shards (about 1 KB a row: 4 GB for 4M) stay outside `out/`; their sidecars
//!    (with the shard's SHA-256) go to `out/hybrid/labels/`. With `upload_shards`, each shard is
//!    hashed (it must match its sidecar) and uploaded content-addressed to `inputs/sha256/<hex>`
//!    (skipped when already there; checked by size), and `out/hybrid/shards.json` lists them in
//!    the form a format-2 manifest's `inputs` takes.
//! 5. **Golden set** (`search decisions --allow-uneven`, then `search worlds --method exact`):
//!    `golden_count` decisions at trick `golden_trick` (seed `golden_seed`), `golden_worlds`
//!    worlds each, valued exactly; `out/hybrid/decisions-t<G>.jsonl`, `exact-t<G>-n<W>.jsonl`.
//! 6. **Training** (`python -m fortyone_nn.train`, which also quantises the network and measures
//!    the integer network): `out/hybrid/net/` holds `float.safetensors`, `int.safetensors` and
//!    `report.json` (history, errors, device facts). The heartbeat shows the epoch and batches.
//! 7. **Hybrid check** (`search hybrid-check`, all usable threads): H_cut against exact on the
//!    golden set; `out/hybrid/hc-t<G>-o<K>.json` and its printed tables in `.txt`.
//! 8. **Summary**: `out/hybrid/rung.json` (rewritten after every step): the resolved settings,
//!    each step's seconds, the environment and device facts, the label sidecars, the networks'
//!    SHA-256, the check's verdict and its overall row, and every warning.
//!
//! CPU use: every step sets `partial_cpu_ok`, so the launcher's low-CPU rule (85% for 5 minutes)
//! does not apply to this kind. The long stretches leave the CPU idle by design (the download of
//! the environment, and the training on the GPU, where the CPU only feeds it), and on hundreds of
//! threads the CPU steps finish in seconds to a minute, mostly in their serial parts (start
//! states, writing gigabytes of shards), so their heartbeats read low too: the rule would stop a
//! healthy run. Instead the steps that could hang are time-limited (killed, and the chunk fails):
//! each `uv sync` attempt 30 minutes, the GPU check 10 minutes, the training `max_minutes` plus
//! 30 minutes; the run's dollar cap bounds everything.
//!
//! Reproducibility: the labels, the golden set and the check are byte-identical on every machine
//! and thread count (`search`). Float training on a GPU is not bit-reproducible across machines;
//! the integer network is what the check validates, and its SHA-256 is recorded
//! (docs/M3-HYBRID-SPEC.md §6).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};

use crate::bundle::sha256_file;
use crate::spike::build_table_from;
use crate::{Ctx, sibling};

/// The settings a `hybrid_rung` chunk takes; any other key is refused.
pub(crate) const KEYS: [&str; 31] = [
    "cut",
    "labels_random",
    "labels_random_seed",
    "labels_rules",
    "labels_rules_seed",
    "val_count",
    "val_seed",
    "golden_trick",
    "golden_count",
    "golden_seed",
    "golden_worlds",
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
    /// `random`, `rules` or `val` (the file is `t<cut>-<name>-s<seed>.npy`).
    pub name: &'static str,
    /// `search labels --source`.
    pub source: &'static str,
    pub count: u64,
    pub seed: u64,
}

/// A `hybrid_rung` chunk's settings, checked.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Params {
    pub cut: u64,
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
}

impl Params {
    /// The settings as recorded in `rung.json`.
    fn to_json(&self) -> Value {
        let set = |s: &LabelSet| json!({ "source": s.source, "count": s.count, "seed": s.seed });
        json!({
            "cut": self.cut,
            "train": self.train.iter().map(|s| (s.name.to_string(), set(s))).collect::<Map<_, _>>(),
            "val": set(&self.val),
            "golden": { "trick": self.golden_trick, "count": self.golden_count,
                        "seed": self.golden_seed, "worlds": self.golden_worlds },
            "start_hands": self.start_hands, "tt_mb": self.tt_mb, "verify_pct": self.verify_pct,
            "net": { "widths": self.widths, "head": self.head, "epochs": self.epochs,
                     "batch": self.batch, "lr": self.lr, "patience": self.patience,
                     "seed": self.train_seed, "max_minutes": self.max_minutes,
                     "mix_weight": self.mix_weight },
            "device": self.device.name(), "train_threads": self.train_threads,
            "compare_cpu": self.compare_cpu, "upload_shards": self.upload_shards,
        })
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
    let set = |name: &'static str, source: &'static str, count: &str, seed: &str, d: (u64, u64)| {
        Ok::<_, String>(LabelSet {
            name,
            source,
            count: u(count, d.0)?,
            seed: u(seed, d.1)?,
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
    let val = set("val", "random", "val_count", "val_seed", (100_000, 201))?;
    let train: Vec<LabelSet> = [random, rules]
        .into_iter()
        .filter(|s| s.count > 0)
        .collect();
    if train.is_empty() {
        return Err(
            "hybrid_rung: labels_random and labels_rules are both 0: no training labels".into(),
        );
    }
    if val.count == 0 {
        return Err("hybrid_rung: val_count must be at least 1".into());
    }
    // Training, validation and golden-set positions use disjoint seeds (spec §5).
    let mut seen: Vec<(u64, &str)> = Vec::new();
    for x in train.iter().chain(std::iter::once(&val)) {
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
        return Err("hybrid_rung: golden_count and golden_worlds must be at least 1".into());
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
    Ok(Params {
        cut,
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
    })
}

/// The programs a chunk runs (the release bundle's in production; fakes in the tests).
pub(crate) struct Tools {
    pub search: PathBuf,
    pub uv: PathBuf,
    /// `python/nn`: `pyproject.toml`, `uv.lock`, `fortyone_nn/`.
    pub python_dir: PathBuf,
    pub nvidia_smi: PathBuf,
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
    chunk_with(ctx, &params, &tools, &table, table_rec)
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
    };
    r.rec.insert("params".into(), params.to_json());
    r.rec.insert("table".into(), table_rec);
    r.rec.insert("threads".into(), json!(r.ctx.threads));
    r.python_env()?;
    r.device_check()?;
    let shards = r.labels()?;
    if params.upload_shards {
        r.upload_shards(&shards)?;
    }
    let (decisions, worlds) = r.golden()?;
    let net = r.out.join("net");
    r.train(&shards, &net, &r.device.clone(), "train", "net")?;
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
    let verdict = r.check(&net, &decisions, &worlds)?;
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

    /// Step 4: the label shards (training sets, then validation). Returns their paths in that
    /// order.
    fn labels(&mut self) -> Result<Vec<PathBuf>, String> {
        let dir = self.work.join("labels");
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let sets: Vec<LabelSet> = self
            .p
            .train
            .iter()
            .chain(std::iter::once(&self.p.val))
            .cloned()
            .collect();
        let mut paths = Vec::new();
        let mut recs = Vec::new();
        for (i, s) in sets.iter().enumerate() {
            let t = Instant::now();
            let file = format!("t{}-{}-s{}.npy", self.p.cut, s.name, s.seed);
            let path = dir.join(&file);
            let log = format!("hybrid/labels-{}.log", s.name);
            let step = format!("labels {} ({} of {})", s.name, i + 1, sets.len());
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
                .args(["--threads", &self.threads().to_string()])
                .args(["--start-hands", &self.p.start_hands.to_string()]);
            let base = self.progress(&step, json!({ "rows_total": s.count }));
            self.ctx.set_phase("chunk", base.clone());
            let log_path = self.ctx.logs.join(&log);
            self.ctx.run_child_with(c, &log, &mut |ctx| {
                let mut p = base.clone();
                if let Some((done, total)) = label_progress(&log_path) {
                    p["rows_done"] = json!(done);
                    p["rows_total"] = json!(total);
                }
                ctx.progress = p;
            })?;
            let side_src = PathBuf::from(format!("{}.json", path.display()));
            let side = read_json(&side_src)
                .ok_or_else(|| format!("hybrid_rung: {} wrote no sidecar", path.display()))?;
            let side_dst = self.out.join("labels").join(format!("{file}.json"));
            write_json(&side_dst, &side)?;
            let bytes = fs::metadata(&path).map_or(0, |m| m.len());
            recs.push(
                json!({ "name": s.name, "source": s.source, "count": s.count,
                              "seed": s.seed, "file": file, "bytes": bytes,
                              "shard_sha256": side["shard_sha256"],
                              "seconds": secs(t) }),
            );
            self.rec.insert("labels".into(), json!(recs));
            self.done(&step, t)?;
            paths.push(path);
        }
        Ok(paths)
    }

    /// Step 4b (`upload_shards`): each shard to `inputs/sha256/<hex>`.
    fn upload_shards(&mut self, shards: &[PathBuf]) -> Result<(), String> {
        let t = Instant::now();
        let progress = self.progress("upload shards", json!({ "shards": shards.len() }));
        self.ctx.set_phase("chunk", progress);
        let Some(remote) =
            crate::remote::from_env().map(std::sync::Arc::<dyn crate::remote::Remote>::from)
        else {
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
        for (k, path) in shards.iter().enumerate() {
            let want = labels[k]["shard_sha256"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            let file = labels[k]["file"].as_str().unwrap_or_default().to_string();
            let (p2, r) = (path.clone(), std::sync::Arc::clone(&remote));
            let res = self
                .ctx
                .blocking(move || -> Result<(String, u64, &'static str), String> {
                    let (sha, bytes) = sha256_file(&p2)?;
                    if sha != want {
                        return Err(format!(
                            "{}: SHA-256 {sha}, but its sidecar says {want}",
                            p2.display()
                        ));
                    }
                    let key = format!("inputs/sha256/{sha}");
                    let size = |r: &dyn crate::remote::Remote| -> Result<Option<u64>, String> {
                        Ok(r.list_sized("inputs/sha256/")?
                            .into_iter()
                            .find(|(k, _)| *k == key)
                            .map(|(_, n)| n))
                    };
                    if size(&*r)? == Some(bytes) {
                        return Ok((sha, bytes, "already there"));
                    }
                    crate::remote::retry(3, || r.put(&p2, &key, true))?;
                    match size(&*r)? {
                        Some(n) if n == bytes => Ok((sha, bytes, "uploaded")),
                        other => Err(format!(
                            "{key}: {other:?} bytes after the upload, want {bytes}"
                        )),
                    }
                });
            let (sha, bytes, status) =
                res.map_err(|e| format!("hybrid_rung: upload_shards: {e}"))?;
            println!("runner: hybrid_rung: {file}: inputs/sha256/{sha} ({status})");
            inputs.insert(
                file.trim_end_matches(".npy").to_string(),
                json!(format!("sha256:{sha}")),
            );
            list.push(json!({ "file": file, "sha256": sha, "bytes": bytes,
                              "key": format!("inputs/sha256/{sha}"), "status": status }));
        }
        let doc = json!({ "shards": list, "manifest_inputs": inputs });
        write_json(&self.out.join("shards.json"), &doc)?;
        self.rec.insert("shards".into(), doc);
        self.done("upload shards", t)
    }

    /// Step 5: the golden set. Returns (decisions file, exact worlds file).
    fn golden(&mut self) -> Result<(PathBuf, PathBuf), String> {
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
            .args(["--threads", &self.threads().to_string()])
            .args(["--tt-mb", &self.p.tt_mb.to_string(), "--progress"])
            .arg(&prog);
        let base = self.progress("golden worlds", json!({}));
        self.ctx.set_phase("chunk", base.clone());
        self.ctx
            .run_child_with(c, "hybrid/golden.log", &mut |ctx| {
                let mut p = base.clone();
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
        Ok((decisions, worlds))
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
        let (train, val) = shards.split_at(shards.len() - 1);
        let prog = self.work.join(format!("{key}.progress.json"));
        let _ = fs::remove_file(&prog);
        let threads = self.train_threads_on(device);
        let mut c = self.python_cmd("fortyone_nn.train");
        c.arg("--train")
            .args(train)
            .arg("--val")
            .arg(&val[0])
            .arg("--out")
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
            .arg(&prog);
        let base = self.progress(
            step,
            json!({ "train_device": device, "train_threads": threads }),
        );
        self.ctx.set_phase("chunk", base.clone());
        // The trainer stops itself after `max_minutes` (between epochs); loading, the last epoch,
        // quantising and the integer check get 30 minutes more.
        let limit = (self.p.max_minutes > 0.0)
            .then(|| Duration::from_secs_f64(self.p.max_minutes * 60.0) + TRAIN_EXTRA);
        self.ctx
            .run_child_io(c, &format!("hybrid/{key}.log"), None, limit, &mut |ctx| {
                let mut p = base.clone();
                if let Some(Value::Object(x)) = read_json(&prog) {
                    for (k, v) in x {
                        if k != "device" {
                            p[format!("train_{k}")] = v;
                        }
                    }
                }
                ctx.progress = p;
            })?;
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
        self.rec.insert(
            key.into(),
            json!({ "dir": rel, "device": device, "threads": threads,
                    "int_sha256": sha("int.safetensors")?,
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
                    "seconds": secs(t) }),
        );
        self.done(step, t)
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
                    seed: 101
                },
                LabelSet {
                    name: "rules",
                    source: "rules",
                    count: 800_000,
                    seed: 102
                },
            ]
        );
        assert_eq!(
            p.val,
            LabelSet {
                name: "val",
                source: "random",
                count: 100_000,
                seed: 201
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
        assert_eq!(parse(&Value::Null).expect("null"), p);
        // The scheduler's keys pass.
        parse(
            &json!({ "threads": "all", "est_seconds": 1800, "equity_input": "equity",
                       "table": { "hands": 1000000, "seed": 1 } }),
        )
        .expect("scheduler keys");
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
    }

    /// The committed manifests (docs/vast/m3-hybrid-*.manifest.json) are runnable: the runner
    /// accepts them (their hybrid settings included) and they ask the launcher for one GPU.
    #[test]
    fn committed_manifests_parse() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/vast");
        for (name, cut, labels, golden) in [
            ("m3-hybrid-rung1b", 4, 4_000_000, (3, 400, 320)),
            ("m3-hybrid-smoke", 4, 500_000, (3, 32, 32)),
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
        }
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

    /// Step sequencing with fake programs: a fake `search` (writes each output and a sidecar), a
    /// fake `uv` (creates the environment with a fake `python`) and a fake `python` (the device
    /// check and the trainer). Without a GPU (`nvidia-smi` missing) the chunk installs the `cpu`
    /// group, warns, and trains on the CPU with every thread; with a fake GPU it installs `cu126`
    /// and trains on it with at most 8 threads, then (compare_cpu) once more on the CPU.
    #[cfg(unix)]
    #[test]
    fn steps_run_in_order_with_fake_programs() {
        use std::os::unix::fs::PermissionsExt;
        let base = std::env::temp_dir().join(format!("hybrid-fake-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let bin = base.join("bin");
        let py = base.join("python-nn");
        fs::create_dir_all(&bin).expect("mkdir");
        fs::create_dir_all(py.join("fortyone_nn")).expect("mkdir");
        fs::write(py.join("pyproject.toml"), "[project]\n").expect("write");
        fs::write(py.join("uv.lock"), "version = 1\n").expect("write");
        let script = |name: &str, body: &str| {
            let p = bin.join(name);
            fs::write(&p, format!("#!/bin/sh\n{body}")).expect("write");
            fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).expect("chmod");
            p
        };
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
  worlds) echo '{{"id": 0}}' > "$out"
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
        let params = parse(&json!({
            "labels_random": 7, "labels_rules": 5, "val_count": 3, "golden_count": 2,
            "golden_worlds": 4, "widths": "16,8", "epochs": 2, "compare_cpu": true,
        }))
        .expect("params");
        let table = base.join("equity-v0.npy");
        fs::write(&table, "table").expect("write");

        // 1. No GPU.
        let root = base.join("nogpu");
        let tools = Tools {
            search: search.clone(),
            uv: uv.clone(),
            python_dir: py.clone(),
            nvidia_smi: base.join("no-such-nvidia-smi"),
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
            "t4-rules-s102.npy --val",
            "t4-val-s201.npy --out",
            "--head outcome --widths 16,8 --epochs 2 --batch 1024 --lr 0.001 --patience 5 --seed 1 \
             --threads 4 --max-minutes 200 --mix-weight 1 --device cpu --progress",
        ] {
            assert!(train.contains(want), "{want:?} in {train}");
        }
        let labels = lines[first("search labels --trick 4 --count 7")];
        assert!(
            labels.contains("--threads 4 --start-hands 20000"),
            "{labels}"
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
                "labels random (1 of 3)",
                "labels rules (2 of 3)",
                "labels val (3 of 3)",
                "golden decisions",
                "golden worlds",
                "train",
                "hybrid check"
            ]
        );
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
        assert!(trains[0].contains("--threads 8 ") && trains[0].contains("--device cuda"));
        assert!(trains[1].contains("--threads 16 ") && trains[1].contains("--device cpu"));
        assert!(trains[1].contains("net-cpu"));
        let rung = read_json(&root.join("out/hybrid/rung.json")).expect("rung.json");
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
}
