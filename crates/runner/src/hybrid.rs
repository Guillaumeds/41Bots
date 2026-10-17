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
//! harvest's search too. Its trick must be the cut's (checked when it is fetched); it is required
//! by `labels_harvest` and `labels_race` at cut 3 or 2 and by the audit loop at cut 3.
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
//! made on the threads it leaves (its stall watch never counts a wait for a shard). `partial_epochs`
//! (0: no limit) caps the epochs on each partial set (`--partial-epochs`). One part is the former
//! behaviour, step for step. Epochs on part of the data warm-start the full training, which can
//! cost a little accuracy (docs/m3-hybrid/overlap.md); the validation loss says.
//! The audit loop (docs/m3-hybrid/audit-loop.md §4; module [`audit`]): `audit_rounds` (0: none,
//! the chunk exactly as before) above 0 replaces steps 6–7 by rounds of training, an audit shard,
//! the auditor (`python -m fortyone_nn.audit`), the golden check with a slice per weak group and
//! mini golden sets, and a targeted harvest that joins the next round's training, until the
//! network is flat or `audit_rounds` is reached; with its settings `audit_rows`, `audit_seed`,
//! `audit_gamma`, `audit_standard`, `audit_push_standard`, `audit_mini_candidates`,
//! `audit_mini_min`,
//! `audit_keep_cap`, `audit_weight`, `audit_mini_decisions`, `audit_mini_sets`,
//! `audit_shapes`, `audit_min_gain`, `audit_target_rows` and `audit_predicates_input` (refused
//! without it); the network grows up the `audit_shapes` ladder when a round's targeted harvest
//! stops helping. On a GPU with `overlap` each round's training
//! runs beside the CPU steps that do not need it. At cut 2 (no trick-1 golden set is built: owner,
//! 2026-10-16) the loop needs `t1_decisions_input` and `t1_saved_inputs` (the saved exact trick-1
//! worlds, its rough test, reported only; `t1_verify_*` its round-1 verification sample), refuses
//! the golden set's and the mini sets' settings, and is flat when a round's audit confirms no
//! group; a cut-2 rung with a check but no loop (a trick-1 golden set) is refused.
//! Also: `mix_weight` (1.0), `start_hands` (20000, for the label roots and the golden
//! decisions), `tt_mb` (64, the solver table per thread of the golden worlds
//! and the check), `verify_pct` (1, the check's recomputed sample of stored exact worlds),
//! `train_threads` (CPU threads of the trainer: all usable threads on the CPU, at most 8 beside
//! a GPU), `compare_cpu` (false; after a GPU training, train once more on the CPU with the same
//! settings and record both, for small checks), `upload_shards` (false; see step 4), `table`
//! (the equity table, as `search_measure`'s), and the scheduler's `threads`, `est_seconds`,
//! `equity_input`.
//!
//! Training modes, reused labels and the leader focus (docs/m3-hybrid/audit-loop.md §10; owner,
//! 2026-10-17): `train_mode` (`"scratch"`, the default: everything above, unchanged; or
//! `"additive"`: the audit loop continues a stored network, `init_net` and `init_net_float`,
//! keeping a round only if the golden set is no worse), `additive_epochs`, `additive_lr`,
//! `shrink_perturb`, `adopt_tolerance` and `harvest_seat_weights` ([`audit::additive`]);
//! `label_inputs`, stored shards reused instead of generated ([`reuse`]).
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
//!    run names it by, as `harvest_net` or a check's network); a failure is only a warning. Its
//!    float weights are shared the same way (`net.float_key`: `init_net_float` of an additive run).
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
//! each `uv sync` attempt 30 minutes, the GPU check 10 minutes; the training stops itself after
//! `max_minutes` and is killed only when its progress file stops changing for the larger of
//! `train_stall_minutes` (30) and 3 times its longest epoch while it is not waiting for data
//! ([`crate::Stall`]); the run's dollar cap bounds everything.
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

mod audit;
mod reuse;

use audit::additive;

/// The settings a `hybrid_rung` chunk takes; any other key is refused.
pub(crate) const KEYS: [&str; 72] = [
    "cut",
    "check",
    "audit_rounds",
    "audit_rows",
    "audit_gamma",
    "audit_mini_candidates",
    "audit_mini_min",
    "audit_standard",
    "audit_push_standard",
    "audit_seed",
    "audit_keep_cap",
    "audit_weight",
    "audit_mini_decisions",
    "audit_mini_sets",
    "audit_shapes",
    "audit_min_gain",
    "audit_target_rows",
    "audit_predicates_input",
    "t1_decisions_input",
    "t1_saved_inputs",
    "t1_verify_pct",
    "t1_verify_rule",
    "t1_verify_solve_threads",
    "t1_verify_tt_mb",
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
    "train_stall_minutes",
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

/// The trainer is watched through its progress file, not a time limit ([`crate::Stall`]): it
/// stops itself after `max_minutes` between epochs, but an epoch of a wide network on 60M+ rows,
/// the loading of a shard that joins and the waits for the run's own shards (the audit loop's
/// targeted harvest, mini golden sets and slices) can each take longer than any fixed extra.
/// It is killed only when its progress file (rewritten about every 20 s within an epoch, and at
/// every step) stays unchanged for the larger of `train_stall_minutes` (default this) and
/// `TRAIN_STALL_EPOCHS` times its longest epoch, never while it waits for data the run is still
/// making.
const TRAIN_STALL_MINUTES: f64 = 30.0;
const TRAIN_STALL_EPOCHS: u32 = 3;

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
    /// `train_stall_minutes` (30): the floor of the trainer's stall watch ([`TRAIN_STALL_EPOCHS`]).
    pub train_stall_minutes: f64,
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
    /// `partial_epochs` (0: no limit): with `label_parts` above 1, or in the audit loop (each
    /// round's training on a GPU with `overlap` starts before the round before's targeted shard
    /// lands), at most this many epochs on each partial set of shards (the trainer's
    /// `--partial-epochs`); then the trainer waits for the rest.
    pub partial_epochs: u64,
    /// The audit loop's settings (`audit_rounds` above 0; [`audit`]); `None`: one training and
    /// one check, as before.
    pub audit: Option<audit::AuditSettings>,
    /// `train_mode` `"additive"`'s settings ([`additive`]); `None`: scratch, as before.
    pub additive: Option<additive::Additive>,
    /// `label_inputs`: stored shards reused instead of generated ([`reuse`]).
    pub label_inputs: Vec<reuse::LabelInput>,
    /// `harvest_seat_weights`: the harvested rows' loss weights by seat band ([`additive`]).
    pub seat_weights: Option<String>,
    /// The harvest's settings (`harvest_cap_per_decision`, `harvest_positions`) when its shard is
    /// reused, for the audit loop's harvests (else they come from the generated harvest set).
    pub reused_harvest_args: Option<Vec<String>>,
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
                     "train_stall_minutes": self.train_stall_minutes,
                     "mix_weight": self.mix_weight },
            "device": self.device.name(), "train_threads": self.train_threads,
            "compare_cpu": self.compare_cpu, "upload_shards": self.upload_shards,
            "label_parts": self.label_parts, "overlap": self.overlap,
            "partial_epochs": self.partial_epochs,
            "audit": self.audit.as_ref().map(audit::AuditSettings::to_json),
            "train_mode": self.additive.as_ref().map_or(json!({ "train_mode": "scratch" }), additive::Additive::to_json),
            "label_inputs": self.label_inputs.iter().map(reuse::LabelInput::to_json).collect::<Vec<_>>(),
            "harvest_seat_weights": self.seat_weights,
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
                // A reused validation shard (`label_inputs`) has no rows to make.
                .chain(std::iter::once(&self.val).filter(|v| v.count > 0))
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
    let known =
        |k: &str| KEYS.contains(&k) || additive::KEYS.contains(&k) || reuse::KEYS.contains(&k);
    if let Some(k) = o.keys().find(|k| !known(k.as_str())) {
        return Err(format!(
            "hybrid_rung: unknown setting {k:?} (settings: {}, {}, {})",
            KEYS.join(", "),
            additive::KEYS.join(", "),
            reuse::KEYS.join(", ")
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
    let mut random = set(
        "random",
        "random",
        "labels_random",
        "labels_random_seed",
        (3_200_000, 101),
    )?;
    let mut rules = set(
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
    // The audit loop's harvests (its audit shards and targeted harvests) are harvests too: at
    // trick 3, round 1's audit shard is driven by `harvest_net` (no network of the loop's own
    // exists yet), later rounds by the latest network. At trick 2 round 1's audit shard waits
    // for round 1's own network instead (`audit.rs`, "Trick 2"), so the loop needs none there.
    let audit_rounds = o.get("audit_rounds").and_then(Value::as_u64).unwrap_or(0);
    if harvest_net.is_some() && harvest.count == 0 && race.count == 0 && audit_rounds == 0 {
        return Err(
            "hybrid_rung: harvest_net drives the harvest's search: it needs labels_harvest, \
             labels_race or audit_rounds above 0"
                .into(),
        );
    }
    // Additive mode: round 1's network is `init_net` itself, which drives its harvests.
    let additive_mode = o.get("train_mode").and_then(Value::as_str) == Some("additive");
    if audit_rounds > 0 && cut == 3 && harvest_net.is_none() && !additive_mode {
        return Err(
            "hybrid_rung: the audit loop at cut 3 needs harvest_net: round 1's audit shard is \
             harvested beside round 1's training, before the loop has a network of its own, and \
             exact values at the frontier of trick 3 (a search of every reached position to the \
             end of the hand) would not finish"
                .into(),
        );
    }
    // The same reason for the base labels' harvests at trick 3 and earlier: they are made
    // before any network of the run exists, so without `harvest_net` every reached frontier
    // position would be solved exactly (thousands a decision, seconds each at trick 2). The
    // driver must be a network of the cut's own trick (`search labels` refuses another).
    if (harvest.count > 0 || race.count > 0) && cut <= 3 && harvest_net.is_none() {
        return Err(format!(
            "hybrid_rung: labels_harvest and labels_race at cut {cut} need harvest_net (an \
             integer O_{cut}, trained for trick-{cut} positions): exact values at the frontier of \
             trick {cut} for every reached position would not finish"
        ));
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
    let mut val = set("val", "random", "val_count", "val_seed", (100_000, 201))?;
    // Stored shards reused instead of generated (`label_inputs`): the sets they give are not
    // made, and their own settings are refused rather than ignored.
    let label_inputs = reuse::parse(o, u("label_parts", 1)?)?;
    let mut reused_harvest_args = None;
    for x in &label_inputs {
        if let Some(k) = reuse::generating_keys(x.set)
            .iter()
            .find(|k| o.contains_key(**k))
        {
            return Err(format!(
                "hybrid_rung: {k} is set, but label_inputs gives the {} shards: remove it",
                x.set
            ));
        }
        match x.set {
            "random" => random.count = 0,
            "rules" => rules.count = 0,
            "harvest" => {
                reused_harvest_args = Some(harvest.args.clone());
                harvest.count = 0;
            }
            "race" => race.count = 0,
            "val" => {
                val.count = 0;
                val.seed = x.seed;
            }
            _ => {}
        }
    }
    let train: Vec<LabelSet> = [random, rules, harvest, race]
        .into_iter()
        .filter(|s| s.count > 0)
        .collect();
    let reused_train = label_inputs.iter().filter(|x| x.set != "val").count();
    let reused_val = label_inputs.iter().any(|x| x.set == "val");
    if train.is_empty() && reused_train == 0 {
        return Err(
            "hybrid_rung: labels_random, labels_rules, labels_harvest and labels_race are all 0: \
             no training labels"
                .into(),
        );
    }
    if val.count == 0 && !reused_val {
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
    // The audit loop's trainings on a GPU with `overlap` are streamed too (each round's starts
    // before the round before's targeted shard lands: `audit.rs`), so the cap applies there
    // with any `label_parts`: at trick 2 a targeted harvest takes hours, and uncapped the
    // trainer would make dozens of epochs on the earlier shards (docs/m3-hybrid/overlap.md: 10
    // partial epochs a part cost 7% of validation loss, 2 about nothing).
    if partial_epochs > 0 && label_parts == 1 && audit_rounds == 0 {
        return Err(
            "hybrid_rung: partial_epochs limits the epochs on part of the labels: it needs \
             label_parts above 1 (or the audit loop)"
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
    // A reused shard's seed counts like a generated one's (the validation shard's is in `val`).
    effective.retain(|x| x.count > 0 || x.name != "val" || !reused_val);
    for x in label_inputs.iter() {
        effective.push(LabelSet {
            name: x.set,
            source: "stored",
            count: x.rows.unwrap_or(0),
            seed: x.seed,
            args: Vec::new(),
        });
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
    // The integer inference cannot load a layer wider than nn::MAX_WIDTH, so a network with one is
    // refused here, before the machine installs or trains anything (a trick-3 run once trained
    // 2048-1024-256 for 27 minutes and then failed at the check, when the limit was 1024).
    let hidden: Vec<usize> = widths.iter().map(|&w| w as usize).collect();
    nn::mlp::check_hidden_widths(&hidden)
        .map_err(|e| format!("hybrid_rung: widths {widths_text:?}: {e}"))?;
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
    let audit = audit::parse_settings(
        o,
        check,
        cut,
        &widths,
        &seen.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
    )?;
    // No trick-1 golden set is built (owner, 2026-10-16: 10-80×10⁹ solver nodes a world): at
    // cut 2 a rung trains only, or runs the audit loop with the saved trick-1 worlds as its
    // rough test.
    if cut == 2 && check && audit.is_none() {
        return Err(
            "hybrid_rung: cut 2 with a check would build a trick-1 golden set, which is never \
             built (owner, 2026-10-16): train only (\"check\": false, then hybrid_t1check), or \
             run the audit loop (audit_rounds with t1_decisions_input and t1_saved_inputs)"
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
        train_stall_minutes: match f("train_stall_minutes", TRAIN_STALL_MINUTES)? {
            m if m > 0.0 => m,
            _ => return Err("hybrid_rung: train_stall_minutes must be above 0".into()),
        },
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
        audit,
        additive: additive::parse(o, audit_rounds, label_parts)?,
        label_inputs,
        seat_weights: additive::parse_seat_weights(o)?,
        reused_harvest_args,
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
        audit_hb: Map::new(),
        shape: None,
        additive: None,
    };
    r.rec.insert("params".into(), params.to_json());
    r.rec.insert("table".into(), table_rec);
    r.rec.insert("threads".into(), json!(r.ctx.threads));
    r.fetch_inputs()?;
    r.python_env()?;
    r.device_check()?;
    // Additive mode: the starting network proven before any label is made.
    r.additive_prove_init()?;
    // The GPU need not wait for the CPU (owner, 2026-10-16): with label parts it starts on the
    // first ones while the rest are made, and a golden set to generate is made while it trains.
    // Both only on a GPU: on the CPU they would share the same threads.
    let gpu = r.device == "cuda";
    let shards = r.labels(gpu && params.label_parts > 1)?;
    if params.upload_shards {
        r.upload_shards(&shards)?;
    }
    if params.audit.is_some() {
        return r.audit_chunk(shards);
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
                    // When each began (seconds into the chunk): the trainer is started before
                    // the golden set, in the runner's own order.
                    "train_started_s": round1(span(r.t0, ts)),
                    "golden_started_s": if generate { json!(round1(span(r.t0, gs))) } else { Value::Null },
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
    /// Stored inputs fetched and checked (`harvest_net`, `golden_decisions`, `golden_worlds`,
    /// `audit_predicates`, `t1_decisions`, `t1_saved:<run>`).
    inputs: BTreeMap<String, PathBuf>,
    /// The training, while it runs in the background.
    bg_train: Option<BgTrain>,
    /// Steps running beside the current one ([`Rung::concurrent`], the background training):
    /// their thread shares and progress, for the heartbeat.
    beside: Vec<(String, String, Option<PathBuf>, u64)>,
    /// The audit loop's heartbeat fields ([`audit`]: `round`, `groups_found`, …), in every
    /// step's progress while the loop runs.
    audit_hb: Map<String, Value>,
    /// The hidden widths the next training uses when the audit loop's capacity ladder has moved
    /// (`None`: `widths`).
    shape: Option<String>,
    /// `train_mode` `"additive"`: the starting network, the best so far and each round's
    /// adoption ([`additive`]); `None` in scratch mode.
    additive: Option<additive::Run>,
}

impl Rung<'_> {
    fn threads(&self) -> u64 {
        self.ctx.threads as u64
    }

    /// The hidden widths of the next training: the audit loop's current shape, else `widths`.
    fn widths_now(&self) -> String {
        self.shape.clone().unwrap_or_else(|| self.p.widths_arg())
    }

    /// The heartbeat's progress for a step: the kind, the step, `partial_cpu_ok` (always: module
    /// notes), the warnings so far, and `extra`.
    fn progress(&self, step: &str, extra: Value) -> Value {
        let mut v = json!({ "kind": "hybrid_rung", "step": step, "partial_cpu_ok": true,
                            "steps_done": self.steps.len(), "warnings": self.warnings });
        if !self.device.is_empty() {
            v["device"] = json!(self.device);
        }
        if let Some(o) = v.as_object_mut() {
            o.extend(self.audit_hb.clone());
            if let Value::Object(e) = extra {
                o.extend(e);
            }
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
        let mut want: Vec<(String, String)> = Vec::new();
        if let Some(h) = &self.p.harvest_net {
            want.push(("harvest_net".into(), h.clone()));
        }
        if let Some((d, w)) = &self.p.golden_inputs {
            want.push(("golden_decisions".into(), d.clone()));
            want.push(("golden_worlds".into(), w.clone()));
        }
        if let Some(h) = self
            .p
            .audit
            .as_ref()
            .and_then(|a| a.predicates_input.as_ref())
        {
            want.push(("audit_predicates".into(), h.clone()));
        }
        if let Some(rt) = self.p.audit.as_ref().and_then(|a| a.rough.as_ref()) {
            want.push(("t1_decisions".into(), rt.decisions.clone()));
            for (run, h) in &rt.saved {
                want.push((audit::RoughTest::input_name(run), h.clone()));
            }
        }
        // Additive mode's starting network (`init_net`) and its float weights.
        if let Some(a) = &self.p.additive {
            want.push(("init_net".into(), a.init_net.clone()));
            want.push(("init_net_float".into(), a.init_float.clone()));
        }
        if want.is_empty() {
            return Ok(());
        }
        let t = Instant::now();
        let progress = self.progress("inputs", json!({ "inputs": want.len() }));
        self.ctx.set_phase("chunk", progress);
        let remote = self.tools.remote.clone().ok_or(
            "hybrid_rung: stored inputs (harvest_net, golden_decisions_input, \
             golden_worlds_input, audit_predicates_input, t1_decisions_input, t1_saved_inputs) \
             come from the remote store (R2, or RUNNER_REMOTE_DIR), and there is none",
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
                name.clone(),
                json!({ "key": input_key(&hex), "sha256": hex, "bytes": bytes,
                        "downloaded": downloaded, "seconds": secs(t1) }),
            );
            self.inputs.insert(name, local);
        }
        // The network that drives the harvests must value the cut's own frontier: `search labels`
        // refuses another (`--harvest-net … was trained for trick 3 positions, not trick 2`),
        // and it would do so only when its first harvest starts, hours into a trick-2 run.
        if let Some(h) = self.inputs.get("harvest_net") {
            let cut = self.p.cut;
            let trick = fs::read(h)
                .map_err(|e| e.to_string())
                .and_then(|b| {
                    let f = nn::safetensors::read(&b)?;
                    f.meta("trick")?
                        .parse::<u64>()
                        .map_err(|_| "its trick is not a number".to_string())
                })
                .map_err(|e| format!("hybrid_rung: harvest_net is not an integer network ({e})"))?;
            if trick != cut {
                return Err(format!(
                    "hybrid_rung: harvest_net is a trick-{trick} network (O_{trick}), but the \
                     harvests at cut {cut} reach trick-{cut} positions: they need an O_{cut} \
                     (`search labels --harvest-net` refuses another)"
                ));
            }
            if let Some(x) = recs.get_mut("harvest_net") {
                x["trick"] = json!(trick);
            }
        }
        // The rough test's decisions file must hold trick-1 decisions (its worlds are checked
        // against it by `search hybrid-check --saved`).
        if let Some(d) = self.inputs.get("t1_decisions") {
            let n = decisions_at(d, 1)?;
            if n == 0 {
                return Err(
                    "hybrid_rung: t1_decisions_input holds no trick-1 decision: it must be the \
                     decisions file the saved trick-1 worlds were valued for"
                        .into(),
                );
            }
            if let Some(x) = recs.get_mut("t1_decisions") {
                x["trick1_decisions"] = json!(n);
            }
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
        self.additive_check_init()?;
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
        // Stored shards reused (`label_inputs`): fetched and checked, first in the training set.
        let (mut train, mut val, mut recs) = self.reused_labels(&dir)?;
        for (i, job) in jobs.iter().enumerate() {
            let s = &job.set;
            let t = Instant::now();
            let started_s = secs(self.t0);
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
            // The harvested rows' loss weights by seat band (`harvest_seat_weights`).
            if let (Some(w), "harvest") = (&self.p.seat_weights, s.source) {
                c.args(["--seat-weights", w]);
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
                                  "threads": threads, "started_s": started_s,
                                  "seconds": secs(t) });
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
        let mut label_inputs = Vec::new();
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
            // What a later run's `label_inputs` names it by (its sidecar shared too). A failure
            // is only a warning: the sidecar is in the bundle, and a paid run never stops for it.
            let entry = match self.label_input_entry(path, &rec, &sha) {
                Ok(e) => Some(e),
                Err(e) => {
                    self.warn(format!(
                        "upload_shards: {file}: its sidecar was not shared ({e}); it is in this \
                         run's bundle (out/hybrid/labels/)"
                    ));
                    None
                }
            };
            list.push(json!({ "file": file, "sha256": sha, "bytes": bytes,
                              "key": input_key(&sha), "status": status,
                              "sidecar_key": entry.as_ref().map(|e| e["sidecar"].clone()) }));
            label_inputs.extend(entry);
        }
        let doc = json!({ "shards": list, "manifest_inputs": inputs,
                          "label_inputs": label_inputs });
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
        // Additive mode: the next training's epochs and rate, and its start (`--init`).
        let (epochs, lr) = self.epochs_lr();
        c.arg("--out")
            .arg(dir)
            .args(["--head", &self.p.head, "--widths", &self.widths_now()])
            .args(["--epochs", &epochs.to_string()])
            .args(["--batch", &self.p.batch.to_string()])
            .args(["--lr", &lr.to_string()])
            .args(["--patience", &self.p.patience.to_string()])
            .args(["--seed", &self.p.train_seed.to_string()])
            .args(["--threads", &threads.to_string()])
            .args(["--max-minutes", &self.p.max_minutes.to_string()])
            .args(["--mix-weight", &self.p.mix_weight.to_string()])
            .args(["--device", device, "--progress"])
            .arg(prog)
            .args(self.init_args());
        c
    }

    /// The trainer's watch ([`TRAIN_STALL_MINUTES`]) on its progress file `prog`. Its own
    /// `max_minutes` still ends its training between epochs.
    fn train_stall(&self, prog: &Path) -> crate::Stall {
        crate::Stall::new(
            prog.to_path_buf(),
            Duration::from_secs_f64(self.p.train_stall_minutes * 60.0),
            TRAIN_STALL_EPOCHS,
        )
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
        let stall = self.train_stall(&prog);
        self.ctx
            .run_child_stall(c, &format!("hybrid/{key}.log"), stall, &mut |ctx| {
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
        let threads = self.train_threads();
        let c = self.train_cmd(feed, &dir, &self.device.clone(), threads, &prog);
        self.start_train_with(c, dir, prog, "hybrid/net.log")
    }

    /// [`Rung::start_train`] with a trainer command already made (`train_cmd` and more
    /// arguments: the audit loop's rounds), its network into `dir`, its progress to `prog`.
    fn start_train_with(
        &mut self,
        c: Command,
        dir: PathBuf,
        prog: PathBuf,
        log: &str,
    ) -> Result<(), String> {
        let device = self.device.clone();
        let threads = self.train_threads();
        let stall = self.train_stall(&prog);
        self.ctx.spawn_bg(c, log, "train", None)?;
        self.ctx.watch_bg("train", stall);
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
        self.finish_train_as("net", "train")
    }

    /// [`Rung::finish_train`], recorded under `key` as the step `step`.
    fn finish_train_as(&mut self, key: &str, step: &str) -> Result<(Instant, Instant), String> {
        let b = self
            .bg_train
            .take()
            .ok_or("hybrid_rung: no training is running")?;
        let base = self.progress(
            step,
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
        self.train_record(&b.dir, &b.device, b.threads, key, step, b.t)?;
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
        let int_key = if key == "net" || key.starts_with("net_r") {
            self.share_net(&dir.join("int.safetensors"), &int_sha)
        } else {
            None
        };
        // Its float weights too, so that a later run can continue it (`init_net_float`).
        let float_sha = sha("float.safetensors")?;
        let float_key = if int_key.is_some() {
            self.share_float(&dir.join("float.safetensors"), &float_sha)
        } else {
            None
        };
        let mut rec = json!({ "dir": rel, "device": device, "threads": threads,
                "int_sha256": int_sha, "int_key": int_key,
                "float_sha256": float_sha, "float_key": float_key,
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

    /// Uploads a trained network's float weights to `inputs/sha256/<hex>` beside its integer
    /// network ([`Rung::share_net`]), so that a later run can continue it (`train_mode`
    /// `"additive"`'s `init_net_float`); its key, or `None` with a warning when the upload fails.
    fn share_float(&mut self, path: &Path, sha: &str) -> Option<String> {
        let remote = self.tools.remote.clone()?;
        let (p2, want) = (path.to_path_buf(), sha.to_string());
        match self
            .ctx
            .blocking(move || put_input(&*remote, &p2, Some(&want)))
        {
            Ok((sha, _, _)) => Some(input_key(&sha)),
            Err(e) => {
                self.warn(format!(
                    "the float network was not shared as an input: {e}; it is in this run's bundle"
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

    /// The additive mode and stored label shards reused (`hybrid/tests/additive.rs`).
    mod additive;

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
        // A layer over the integer inference's limit (nn::MAX_WIDTH, 4096) is refused with the layer and the limit.
        bad(
            json!({ "widths": "8192,1024,256" }),
            "widths \"8192,1024,256\": layer 0: 8192 units, wider than 4096",
        );
        bad(
            json!({ "widths": "512, 4097" }),
            "layer 1: 4097 units, wider than 4096",
        );
        bad(json!({ "widths": "4097" }), "MAX_WIDTH");
        // 2048 and 4096 wide are within it.
        for w in ["2048,1024,256", "2048,2048,1024,256", "4096,2048,256"] {
            assert!(parse(&json!({ "widths": w })).is_ok(), "{w}");
        }
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
            "harvest_net drives the harvest's search: it needs labels_harvest, labels_race or \
             audit_rounds above 0",
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

    /// Every committed manifest (docs/vast/*.manifest.json, found by listing the directory, so
    /// one added later is checked too, before anyone rents for it) is one the runner accepts at
    /// boot (`sched::parse_v2`: every stream's settings through its kind's parse, a hybrid
    /// rung's audit settings included), its placeholders (`REPLACE_WITH_…`, stored inputs still
    /// to upload) filled with stored-input keys; and each hybrid rung's settings parse again
    /// here. None is skipped.
    #[test]
    fn every_committed_manifest_parses() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/vast");
        let mut paths: Vec<PathBuf> = fs::read_dir(&dir)
            .expect("docs/vast")
            .map(|e| e.expect("entry").path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with(".manifest.json"))
            })
            .collect();
        paths.sort();
        assert!(paths.len() >= 8, "{paths:?}");
        let mut rungs = 0;
        for path in &paths {
            let mut text = fs::read_to_string(path).expect("read");
            // Each placeholder its own 64-hex key.
            let mut k = 0u64;
            while let Some(at) = text.find("REPLACE_WITH_") {
                let end = text[at..]
                    .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
                    .map_or(text.len(), |n| at + n);
                let token = text[at..end].to_string();
                k += 1;
                text = text.replace(&token, &format!("{k:064x}"));
            }
            let v: Value =
                serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let m = crate::sched::parse_v2(&v)
                .unwrap_or_else(|e| panic!("{}: the runner refuses it: {e}", path.display()));
            for s in m.streams.iter().filter(|s| s.kind == "hybrid_rung") {
                parse(&s.params).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                rungs += 1;
            }
        }
        assert!(rungs >= 6, "{rungs} hybrid rungs");
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
            ("m3-hybrid-t4-audit-mech", 4, 9_000_000, (3, 400, 320)),
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

        // The trick-2 audit loop (scratch, as the overnight trick-3 and trick-4 loops): cut 2,
        // random and rules-v1 labels only, 2048-1024-256, four rounds of 200,000 audit rows and
        // 250,000 targeted rows, every shard stored, partial epochs capped beside the hours-long
        // harvests, and the rough test on the saved trick-1 worlds named by the very hashes and
        // run labels of the t1check manifest; no driver network (none of trick 2 exists yet:
        // round 1's audit shard waits for network 1), and no GPU model named (CPU-bound).
        let path = dir.join("m3-hybrid-t2-loop.manifest.json");
        let v: Value =
            serde_json::from_str(&fs::read_to_string(&path).expect("read")).expect("json");
        let m = crate::sched::parse_v2(&v).expect("runnable");
        assert_eq!(m.streams.len(), 1);
        let s = &m.streams[0];
        assert_eq!((s.kind.as_str(), s.threads_all), ("hybrid_rung", true));
        let p = parse(&s.params).expect("settings");
        assert_eq!((p.cut, p.check, p.upload_shards), (2, true, true));
        assert_eq!(
            p.train
                .iter()
                .map(|s| (s.name, s.count, s.seed))
                .collect::<Vec<_>>(),
            [("random", 1_200_000, 511), ("rules", 300_000, 512)]
        );
        assert_eq!((p.val.count, p.val.seed), (50_000, 601));
        assert_eq!(
            (p.widths_arg().as_str(), p.head.as_str()),
            ("2048,1024,256", "outcome")
        );
        assert!(p.harvest_net.is_none() && p.golden_inputs.is_none());
        assert!(p.overlap && p.partial_epochs == 2 && p.max_minutes == 0.0);
        assert_eq!((p.epochs, p.patience), (40, 6));
        let a = p.audit.as_ref().expect("the audit loop");
        assert_eq!(
            (a.rounds, a.rows, a.target_rows, a.gamma, a.seed),
            (4, 200_000, 250_000, 0.001, 221)
        );
        assert_eq!(
            a.shapes,
            vec![
                vec![2048, 1024, 256],
                vec![2048, 2048, 1024, 256],
                vec![4096, 2048, 1024, 256]
            ]
        );
        let rt = a.rough.as_ref().expect("the rough test");
        let t1: Value = serde_json::from_str(
            &fs::read_to_string(dir.join("m3-hybrid-t1check.manifest.json")).expect("read"),
        )
        .expect("json");
        let hex = |name: &str| {
            t1["inputs"][name]
                .as_str()
                .and_then(|h| h.strip_prefix("sha256:"))
                .expect("a t1check input")
                .to_string()
        };
        assert_eq!(rt.decisions, hex("t1-decisions"));
        let want: Vec<(String, String)> = t1["streams"][0]["params"]["saved"]
            .as_array()
            .expect("saved")
            .iter()
            .map(|e| {
                (
                    e["run"].as_str().expect("run").to_string(),
                    hex(e["input"].as_str().expect("input")),
                )
            })
            .collect();
        assert_eq!(rt.saved, want);
        assert_eq!((rt.verify_pct, rt.verify_rule.as_str()), (3.0, "rules-v1"));
        let mach = &v["machine"];
        assert_eq!(
            (&mach["min_cores"], &mach["gpus"], &mach["min_ram_gb"]),
            (&json!(300), &json!(1), &json!(192))
        );
        assert_eq!(mach["price_steps"], json!([1.4, 1.7, 2.0]));
        assert!(mach.get("gpu_names").is_none());
        assert!(
            v["note"]
                .as_str()
                .is_some_and(|n| !n.contains("PLACEHOLDER") && n.contains("--max-dollars 35"))
        );
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
            audit_hb: Map::new(),
            shape: None,
            additive: None,
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
                    format!("sleep 30; touch '{}'", marker.display()),
                    2,
                    None,
                ),
                step("broken", "sleep 0.2; exit 4".into(), 2, None),
            ])
            .expect_err("one failed");
        assert!(e.contains("step broken (run beside others) exited"), "{e}");
        assert!(t0.elapsed() < Duration::from_secs(25));
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
        // On the GPU the golden set is made while the network trains, on the threads it leaves
        // (which began first is the runner's own record, below: the children's log lines may
        // come in either order).
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
        assert!(
            o["train_started_s"].as_f64().is_some()
                && o["train_started_s"].as_f64() <= o["golden_started_s"].as_f64(),
            "the trainer starts before the golden set: {o}"
        );
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
    /// The trainer is watched through its progress file (`train_stall_minutes`, here 0.02 min):
    /// a trainer whose file stops changing while it trains is killed with the reason and the
    /// chunk fails at once; one that waits for data beside the steps making it is not (the
    /// stream test's trainers, and `crate::tests::stall_watch_kills_only_a_quiet_child`).
    #[cfg(unix)]
    #[test]
    fn a_quiet_trainer_is_killed_by_its_stall_watch() {
        let f = fakes("stall");
        let params = parse(
            &json!({ "labels_random": 7, "labels_rules": 0, "val_count": 3,
                                    "golden_count": 2, "golden_worlds": 4, "widths": "16,8",
                                    "epochs": 2, "train_stall_minutes": 0.02 }),
        )
        .expect("params");
        assert_eq!(params.to_json()["net"]["train_stall_minutes"], 0.02);
        let tools = gpu_tools(
            &f,
            "quiet-train",
            r#"case "$2" in fortyone_nn.train)
  p=""; prev=""; for a in "$@"; do [ "$prev" = --progress ] && p="$a"; prev="$a"; done
  echo '{"step": "train", "epoch": 0}' > "$p"; sleep 30;; esac"#,
            f.search.clone(),
        );
        let mut ctx = Ctx::local("hybrid-test", &f.base.join("a"), 16);
        let t = Instant::now();
        let e = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({}))
            .expect_err("the trainer was quiet");
        assert!(
            e.contains("step train: no progress in") && e.contains("net.progress.json"),
            "{e}"
        );
        assert!(t.elapsed() < Duration::from_secs(25), "{:?}", t.elapsed());
        assert!(ctx.bg.is_empty());
        for bad in [json!(0), json!(-1)] {
            let Err(e) = parse(&json!({ "train_stall_minutes": bad })) else {
                panic!("refused")
            };
            assert!(e.contains("train_stall_minutes"), "{e}");
        }
    }

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
                "[ \"$1\" = worlds ] && {{ sleep 30; touch '{}'; }}",
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
            t.elapsed() < Duration::from_secs(25),
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
                "case \"$2\" in fortyone_nn.train) sleep 30; touch '{}';; esac",
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
            t.elapsed() < Duration::from_secs(25),
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
        // The label steps run one after another (their calls in order); the trainer runs beside
        // them, so its line may come anywhere: when it started is the runner's own record below.
        let lines: Vec<&str> = log.lines().filter(|l| l.starts_with("search")).collect();
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
        let trains: Vec<&str> = log
            .lines()
            .filter(|l| l.starts_with("python -m fortyone_nn.train"))
            .collect();
        assert_eq!(trains.len(), 1, "{log}");
        let train = trains[0];
        assert!(
            train.contains("--watch-dir") && train.contains("--expect 4 "),
            "{train}"
        );
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
        // The trainer started after part 1 was made and before part 2 began.
        let started = |k: usize| rung["labels"][k]["started_s"].as_f64().expect("started_s");
        let train_at = rung["stream"]["train_started_s"]
            .as_f64()
            .expect("train_started_s");
        assert!(started(2) <= train_at && train_at <= started(3), "{rung}");
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

    /// A small integer network's weights file for frontier trick `trick` (the format `search`
    /// loads), for the tests that store a `harvest_net`.
    fn tiny_net(trick: u8) -> Vec<u8> {
        nn::oracle::write_net(
            &nn::synthetic::random_net(7, &[8], 4, 50),
            trick,
            &BTreeMap::new(),
        )
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
        let net = store("int.safetensors", &tiny_net(3));
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
        // A driver of another trick (an O_2 at cut 3; the same for an O_3 at cut 2): refused at
        // once, not when the first harvest starts. Not an integer network at all: refused too.
        for (name, bytes, says) in [
            (
                "o2",
                tiny_net(2),
                "harvest_net is a trick-2 network (O_2), but the harvests at cut 3 reach \
                 trick-3 positions: they need an O_3",
            ),
            (
                "junk",
                b"an integer O_3".to_vec(),
                "harvest_net is not an integer network",
            ),
        ] {
            let other = store(&format!("{name}.safetensors"), &bytes);
            let p = parse(&settings(&other, 2)).expect("params");
            let e = run(name, &p, &tools).expect_err("wrong driver");
            assert!(e.contains(says), "{e}");
            assert_eq!(
                fs::read_to_string(&f.calls).expect("calls"),
                "",
                "nothing ran"
            );
        }
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
        fs::write(&p, tiny_net(4)).expect("write");
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

    // ---------------------------------------------------------------------------------------
    // The audit loop (`audit`; docs/m3-hybrid/audit-loop.md §4).

    /// The audit loop's settings: refused without `audit_rounds`, with `"check": false` or
    /// `compare_cpu`, or when a round's seed meets another; defaults otherwise.
    #[test]
    fn audit_settings() {
        assert_eq!(parse(&json!({})).expect("defaults").audit, None);
        assert_eq!(
            parse(&json!({ "audit_rounds": 0 })).expect("0 is off"),
            parse(&json!({})).expect("defaults")
        );
        let p = parse(&json!({ "audit_rounds": 4 })).expect("audit");
        let a = p.audit.as_ref().expect("on");
        assert_eq!(
            (
                a.rounds,
                a.rows,
                a.seed,
                a.keep_cap,
                a.weight,
                a.mini_decisions,
                a.mini_sets,
                a.target_rows
            ),
            (4, 1_000_000, 301, 256, 2.0, 50, 4, 10_000_000)
        );
        assert_eq!((a.audit_seed(1), a.target_seed(4)), (30101, 30154));
        // The default ladder: the rung's widths, then the larger shapes (all within nn::MAX_WIDTH,
        // 4096 since 2026-10-16).
        assert_eq!(
            a.shapes,
            vec![
                vec![1024, 512, 128],
                vec![2048, 1024, 256],
                vec![2048, 2048, 1024, 256],
                vec![4096, 2048, 1024, 256]
            ]
        );
        assert_eq!(a.min_gain, 0.15);
        let net = format!("inputs/sha256/{}", "ab".repeat(32));
        let at3 =
            parse(&json!({ "cut": 3, "audit_rounds": 1, "harvest_net": net })).expect("trick 3");
        assert_eq!(at3.audit.expect("on").target_rows, 4_000_000);
        // harvest_net drives round 1's audit shard at trick 3: accepted with the audit loop
        // alone (no harvest or race labels), and required there.
        assert_eq!(at3.harvest_net, Some("ab".repeat(32)));
        assert!(
            at3.train
                .iter()
                .all(|s| s.name != "harvest" && s.name != "race")
        );
        let Err(e) = parse(&json!({ "cut": 3, "audit_rounds": 2 })) else {
            panic!("refused")
        };
        assert!(
            e.contains("the audit loop at cut 3 needs harvest_net"),
            "{e}"
        );
        // At trick 2 the loop needs none: round 1's audit shard waits for round 1's own
        // network (the trick-2 design); the base labels' harvests still do, at trick 3 and 2.
        let t1 = |o: &mut Value| {
            o["t1_decisions_input"] = json!(format!("inputs/sha256/{}", "d".repeat(64)));
            o["t1_saved_inputs"] =
                json!([{ "run": "r", "input": format!("inputs/sha256/{}", "e".repeat(64)) }]);
        };
        let mut o = json!({ "cut": 2, "audit_rounds": 2 });
        t1(&mut o);
        let at2 = parse(&o).expect("trick 2 without harvest_net");
        assert!(at2.harvest_net.is_none() && at2.audit.is_some());
        for (cut, k) in [
            (2, "labels_harvest"),
            (3, "labels_race"),
            (2, "labels_race"),
        ] {
            let mut o = json!({ "cut": cut, "check": false, k: 1000 });
            let Err(e) = parse(&o) else {
                panic!("refused at cut {cut}")
            };
            assert!(
                e.contains(&format!(
                    "labels_harvest and labels_race at cut {cut} need harvest_net (an integer \
                     O_{cut}"
                )),
                "{e}"
            );
            o["harvest_net"] = json!(net);
            parse(&o).expect("with harvest_net");
        }
        let at4 = parse(&json!({ "audit_rounds": 2, "harvest_net": net })).expect("trick 4");
        assert!(at4.audit.is_some() && at4.harvest_net.is_some());
        let sh = parse(&json!({ "widths": "16,8", "audit_rounds": 2,
                                "audit_shapes": "<widths>; 32,16 ;64,32,8", "audit_min_gain": 0.3 }))
        .expect("shapes");
        let a = sh.audit.clone().expect("on");
        assert_eq!(a.shapes, vec![vec![16, 8], vec![32, 16], vec![64, 32, 8]]);
        assert_eq!(a.min_gain, 0.3);
        assert_eq!(
            sh.to_json()["audit"]["shapes"],
            json!(["16,8", "32,16", "64,32,8"])
        );
        let j = p.to_json();
        assert_eq!(
            j["audit"]["audit_seeds"],
            json!([30101, 30102, 30103, 30104])
        );
        assert_eq!(j["audit"]["predicates_input"], Value::Null);
        let key = format!("inputs/sha256/{}", "c".repeat(64));
        let p = parse(
            &json!({ "audit_rounds": 2, "audit_rows": 5000, "audit_seed": 7,
                               "audit_keep_cap": 64, "audit_weight": 1.5,
                               "audit_mini_decisions": 60, "audit_target_rows": 9000,
                               "audit_predicates_input": key }),
        )
        .expect("given");
        let a = p.audit.expect("on");
        assert_eq!(
            (
                a.rows,
                a.seed,
                a.keep_cap,
                a.weight,
                a.mini_decisions,
                a.target_rows
            ),
            (5000, 7, 64, 1.5, 60, 9000)
        );
        assert_eq!(a.predicates_input, Some("c".repeat(64)));
        // Gamma, the shard's size with it, the standards (decisions of 2026-10-16 night).
        let a = parse(&json!({ "audit_rounds": 2 }))
            .expect("defaults")
            .audit
            .expect("on");
        assert_eq!(
            (
                a.gamma,
                a.rows,
                a.mini_candidates,
                a.mini_min,
                a.standard,
                a.push_standard
            ),
            (0.001, 1_000_000, 10_000, 30, 0.01, None)
        );
        // `audit_mini_sets` has no upper limit (tonight's trick-4 loop measures every waiting
        // group each round).
        let a = parse(&json!({ "audit_rounds": 2, "audit_mini_sets": 400, "audit_mini_min": 40 }))
            .expect("many mini sets")
            .audit
            .expect("on");
        assert_eq!((a.mini_sets, a.mini_min), (400, 40));
        for (cut, rows) in [(4, 20_000_000), (3, 10_000_000), (2, 200_000)] {
            // An audit loop at cut 3 needs `harvest_net` (round 1's audit shard); at cut 2 the
            // saved trick-1 worlds, and its audit shard stays at 200,000 rows at any gamma.
            let mut o = json!({ "cut": cut, "audit_rounds": 2, "audit_gamma": 0.0001 });
            if cut == 3 {
                o["harvest_net"] = json!(format!("inputs/sha256/{}", "ab".repeat(32)));
            }
            if cut == 2 {
                o["t1_decisions_input"] = json!(format!("inputs/sha256/{}", "d".repeat(64)));
                o["t1_saved_inputs"] =
                    json!([{ "run": "r", "input": format!("inputs/sha256/{}", "e".repeat(64)) }]);
            }
            let a = parse(&o).expect("fine gamma").audit.expect("on");
            assert_eq!((a.gamma, a.rows), (0.0001, rows), "cut {cut}");
        }
        let a = parse(
            &json!({ "audit_rounds": 2, "audit_gamma": 0.0001, "audit_rows": 7000,
                               "audit_push_standard": 0.005, "audit_mini_candidates": 600 }),
        )
        .expect("given")
        .audit
        .expect("on");
        assert_eq!(
            (a.rows, a.push_standard, a.mini_candidates),
            (7000, Some(0.005), 600)
        );
        let j = a.to_json();
        assert_eq!(
            (&j["gamma"], &j["standard_pct"], &j["push_standard_pct"]),
            (&json!(0.0001), &json!(0.01), &json!(0.005))
        );
        // The default ladder holds only shapes larger than the rung's, each once.
        for (w, want) in [
            (
                "2048,1024,256",
                vec![
                    vec![2048, 1024, 256],
                    vec![2048, 2048, 1024, 256],
                    vec![4096, 2048, 1024, 256],
                ],
            ),
            (
                "2048,2048,1024,256",
                vec![vec![2048, 2048, 1024, 256], vec![4096, 2048, 1024, 256]],
            ),
            ("4096,4096,2048", vec![vec![4096, 4096, 2048]]),
        ] {
            let a = parse(&json!({ "audit_rounds": 2, "widths": w }))
                .expect("ladder")
                .audit
                .expect("on");
            assert_eq!(a.shapes, want, "{w}");
        }
        // The mini sets' keep share: a half from a weight of 0.1%, 500 times the weight below,
        // at least 5%.
        for (w, share) in [
            (0.05, 500_000),
            (0.001, 500_000),
            (0.0005, 250_000),
            (0.0001, 50_000),
            (0.00002, 50_000),
            (f64::NAN, 50_000),
        ] {
            assert_eq!(audit::keep_share(w), share, "{w}");
        }
        assert_eq!(audit::std_text(0.0075), "0.0075%");
        let bad = |v: Value, says: &str| {
            let e = parse(&v).expect_err(&v.to_string());
            assert!(e.contains(says), "{v}: {e}");
        };
        bad(
            json!({ "audit_rows": 10 }),
            "audit_rows is set but audit_rounds is 0",
        );
        bad(
            json!({ "audit_rounds": 0, "audit_weight": 2 }),
            "audit_rounds is 0",
        );
        bad(json!({ "audit_rounds": 50 }), "at most 49");
        bad(
            json!({ "audit_rounds": 2, "check": false }),
            "it needs \"check\": true",
        );
        bad(
            json!({ "audit_rounds": 2, "compare_cpu": true }),
            "compare_cpu",
        );
        bad(json!({ "audit_rounds": 2, "audit_rows": 0 }), "audit_rows");
        bad(json!({ "audit_rounds": 2, "audit_seed": 0 }), "audit_seed");
        bad(
            json!({ "audit_rounds": 2, "audit_keep_cap": 0 }),
            "audit_keep_cap",
        );
        bad(
            json!({ "audit_rounds": 2, "audit_weight": 0 }),
            "audit_weight",
        );
        bad(
            json!({ "audit_rounds": 2, "audit_mini_decisions": 0 }),
            "audit_mini_decisions",
        );
        bad(
            json!({ "audit_rounds": 2, "audit_target_rows": 0 }),
            "audit_target_rows",
        );
        bad(
            json!({ "audit_rounds": 2, "audit_shapes": "<widths>;8192,1024,256" }),
            "MAX_WIDTH",
        );
        bad(
            json!({ "audit_rounds": 2, "widths": "16,8", "audit_shapes": "32,16" }),
            "must start with the rung's widths (16,8",
        );
        bad(
            json!({ "audit_rounds": 2, "widths": "16,8", "audit_shapes": "16,8;x" }),
            "is not a list of hidden widths",
        );
        bad(
            json!({ "audit_rounds": 2, "widths": "16,8", "audit_shapes": "16,8;32,16;32,16" }),
            "given twice",
        );
        // The ladder only grows: a shape not larger than the rung's is refused.
        bad(
            json!({ "audit_rounds": 2, "widths": "16,8", "audit_shapes": "16,8;8,8" }),
            "is not larger than the rung's widths",
        );
        bad(
            json!({ "audit_rounds": 2, "audit_gamma": 0 }),
            "audit_gamma",
        );
        bad(
            json!({ "audit_rounds": 2, "audit_gamma": 0.5 }),
            "audit_gamma",
        );
        bad(
            json!({ "audit_rounds": 2, "audit_mini_decisions": 50, "audit_mini_candidates": 49 }),
            "audit_mini_candidates",
        );
        bad(
            json!({ "audit_rounds": 2, "audit_mini_min": 0 }),
            "audit_mini_min",
        );
        bad(
            json!({ "audit_rounds": 2, "audit_mini_min": 51 }),
            "audit_mini_min",
        );
        bad(
            json!({ "audit_rounds": 2, "audit_standard": 0 }),
            "audit_standard",
        );
        bad(
            json!({ "audit_rounds": 2, "audit_standard": 2 }),
            "audit_standard",
        );
        bad(
            json!({ "audit_rounds": 2, "audit_push_standard": 0.01 }),
            "audit_push_standard must be tighter",
        );
        bad(
            json!({ "audit_rounds": 2, "audit_standard": 0.005, "audit_push_standard": 0.0075 }),
            "audit_push_standard must be tighter",
        );
        bad(json!({ "audit_gamma": 0.0001 }), "audit_rounds is 0");
        bad(json!({ "audit_push_standard": 0.005 }), "audit_rounds is 0");
        bad(
            json!({ "audit_rounds": 2, "audit_min_gain": 1 }),
            "audit_min_gain",
        );
        bad(
            json!({ "audit_rounds": 2, "audit_predicates_input": "p.json" }),
            "must be a stored input's key",
        );
        // Round 1's audit shard would be 101·100 + 1 = 10101: label part 1 of seed 101.
        bad(
            json!({ "audit_rounds": 2, "audit_seed": 101, "label_parts": 2 }),
            "round 1's audit shard would have seed 10101",
        );
        bad(
            json!({ "audit_rounds": 1, "audit_seed": 1, "labels_random_seed": 111,
                    "labels_harvest": 5, "labels_harvest_seed": 151 }),
            "round 1's targeted harvest would have seed 151",
        );
    }

    /// The trick-2 loop's settings (cut 2; owner decision of 2026-10-16: no trick-1 golden set):
    /// the saved trick-1 worlds as its rough test are required, the golden set's and the mini
    /// sets' settings refused; `t1_*` refused elsewhere and without the loop; a cut-2 check
    /// without the loop (a trick-1 golden set) refused; the per-cut defaults (200,000 audit rows
    /// and 1,000,000 targeted rows at trick 2).
    #[test]
    fn audit_settings_at_trick_2() {
        let (d, a1, a2) = ("d".repeat(64), "a".repeat(64), "b".repeat(64));
        let base = || {
            json!({ "cut": 2, "audit_rounds": 4,
                    "t1_decisions_input": format!("inputs/sha256/{d}"),
                    "t1_saved_inputs": [ { "run": "m3-t1-tables-1", "input": format!("sha256:{a1}") },
                                         { "run": "m3-agree-t1-ab1", "input": format!("inputs/sha256/{a2}") } ] })
        };
        let p = parse(&base()).expect("trick 2");
        let a = p.audit.as_ref().expect("on");
        assert_eq!((a.rows, a.target_rows), (200_000, 1_000_000));
        let rt = a.rough.as_ref().expect("the rough test");
        assert_eq!(rt.decisions, d);
        assert_eq!(
            rt.saved,
            vec![
                ("m3-t1-tables-1".to_string(), a1.clone()),
                ("m3-agree-t1-ab1".to_string(), a2.clone())
            ]
        );
        assert_eq!(
            (
                rt.verify_pct,
                rt.verify_rule.as_str(),
                rt.verify_solve_threads,
                rt.verify_tt_mb
            ),
            (3.0, "rules-v1", 46, 8192)
        );
        let j = p.to_json();
        assert_eq!(
            j["audit"]["rough_test"]["saved"][1]["input"],
            format!("inputs/sha256/{a2}")
        );
        assert!(
            j["audit"]["rough_test"]["gates"]
                .as_str()
                .is_some_and(|g| g.starts_with("never"))
        );
        let mut given = base();
        for (k, v) in [
            ("t1_verify_pct", json!(0)),
            ("t1_verify_rule", json!("all")),
            ("t1_verify_solve_threads", json!(8)),
            ("t1_verify_tt_mb", json!(512)),
            ("audit_rows", json!(5000)),
        ] {
            given[k] = v;
        }
        let a = parse(&given).expect("given").audit.expect("on");
        let rt = a.rough.expect("rough");
        assert_eq!(
            (
                rt.verify_pct,
                rt.verify_rule.as_str(),
                rt.verify_solve_threads,
                rt.verify_tt_mb,
                a.rows
            ),
            (0.0, "all", 8, 512, 5000)
        );
        // At trick 3 the defaults are as before, and the rough test is absent.
        let net = format!("inputs/sha256/{}", "ab".repeat(32));
        let a3 = parse(&json!({ "cut": 3, "audit_rounds": 1, "harvest_net": net }))
            .expect("trick 3")
            .audit
            .expect("on");
        assert_eq!((a3.rows, a3.rough.is_none()), (1_000_000, true));
        // The auditor's gamma applies at trick 2 too (its standard is the position-level audit);
        // the shard stays at 200,000 rows unless set.
        let mut v = base();
        v["audit_gamma"] = json!(0.0001);
        let a = parse(&v).expect("gamma at trick 2").audit.expect("on");
        assert_eq!((a.gamma, a.rows), (0.0001, 200_000));
        let bad = |v: Value, says: &str| {
            let e = parse(&v).expect_err(&v.to_string());
            assert!(e.contains(says), "{v}: {e}");
        };
        let without = |k: &str| {
            let mut v = base();
            v.as_object_mut().expect("object").remove(k);
            v
        };
        bad(without("t1_decisions_input"), "needs t1_decisions_input");
        bad(without("t1_saved_inputs"), "t1_saved_inputs must be a list");
        for (k, v) in [
            ("golden_count", json!(32)),
            ("golden_worlds", json!(320)),
            ("verify_pct", json!(1)),
            ("audit_mini_decisions", json!(50)),
            ("audit_mini_sets", json!(2)),
            // The mini sets' and the decision-level standards' settings of the overnight loops
            // (eng-auditscale) have nothing to act on at trick 2.
            ("audit_mini_candidates", json!(600)),
            ("audit_mini_min", json!(10)),
            ("audit_standard", json!(0.01)),
            ("audit_push_standard", json!(0.0075)),
        ] {
            let mut v2 = base();
            v2[k] = v;
            bad(
                v2,
                &format!("{k} is set, but at cut 2 there is no golden set"),
            );
        }
        let mut v = base();
        v["t1_saved_inputs"][1]["run"] = json!("m3-t1-tables-1");
        bad(v, "is given twice");
        let mut v = base();
        v["t1_saved_inputs"][0]["path"] = json!("/x");
        bad(v, "unknown key \"path\"");
        let mut v = base();
        v["t1_saved_inputs"][0]["input"] = json!("REPLACE_ME");
        bad(v, "must be a stored input's key");
        let mut v = base();
        v["t1_verify_rule"] = json!("some");
        bad(v, "t1_verify_rule");
        let mut v = base();
        v["cut"] = json!(3);
        v["harvest_net"] = json!(net);
        bad(v, "t1_decisions_input is for the trick-2 loop (cut 2)");
        let mut v = base();
        v["audit_rounds"] = json!(0);
        bad(v, "t1_decisions_input is set but audit_rounds is 0");
        bad(
            json!({ "cut": 2 }),
            "cut 2 with a check would build a trick-1 golden set",
        );
        // Train only at cut 2 stays as before.
        assert!(parse(&json!({ "cut": 2, "check": false })).is_ok());
    }

    /// Fake programs for the audit loop on top of [`fakes`]: a `search` whose targeted harvest
    /// writes its group counts, whose `decisions --keep` keeps two decisions, and whose
    /// `hybrid-check` passes with one decision in each named slice; a `python` (on a GPU when
    /// `gpu`) whose trainer waits, with `--watch-dir`, for its `--expect` shards and records the
    /// ones it saw, and whose auditor confirms the groups of `audit-<n>.json` in the fakes'
    /// directory at its `n`th call (none without the file), touching `stop` at the call that
    /// `stop-at-<n>` names.
    #[cfg(unix)]
    fn audit_tools(f: &Fakes, gpu: bool, stop: Option<&Path>) -> Tools {
        let arg = r#"arg() { n=$1; shift; while [ $# -gt 0 ]; do [ "$1" = "$n" ] && { echo "$2"; return; }; shift; done; }"#;
        let base = f.base.display().to_string();
        let calls = f.calls.display().to_string();
        let search = fake_script(
            &f.bin,
            "search-audit",
            &format!(
                r#"{arg}
case $1 in
  labels)
    '{search}' "$@" || exit 1
    out=$(arg --out "$@")
    if [ -n "$(arg --harvest-keep "$@")" ]; then
      echo '{{"shard_sha256": "{sha}", "count": 7, "harvest": {{"keep": {{"groups": [{{"name": "g-one", "rows": 7}}], "rows_in_any_group": 7}}}}}}' > "$out.json"
    fi;;
  decisions)
    echo "search $*" >> '{calls}'
    out=$(arg --out "$@")
    if [ -n "$(arg --keep "$@")" ] && [ -f '{base}/mini-unfilled' ]; then
      printf '{{"id": 400, "trick": 3}}\n' > "$out"
      echo '{{"kept": 1, "candidates_drawn": 30, "complete": false}}' > "$out.json"
    elif [ -n "$(arg --keep "$@")" ]; then
      printf '{{"id": 400, "trick": 3}}\n{{"id": 403, "trick": 3}}\n' > "$out"
      echo '{{"kept": 2, "candidates_drawn": 9, "complete": true}}' > "$out.json"
    else
      printf '{{"id": 0, "trick": 3}}\n{{"id": 1, "trick": 3}}\n' > "$out"
    fi;;
  hybrid-check)
    echo "search $*" >> '{calls}'
    sl=$(arg --slices "$@"); s=""; i=0
    if [ -n "$(arg --saved "$@")" ]; then
      # The trick-2 rough test (`--saved`): two sets as the saved trick-1 worlds give them (8
      # decisions under the searched burn, 5 under the rules-v1 burn, seats 1-2), every group's
      # slice failing; the overall rows fail with round k's network when `rough-fail-<k>` exists.
      net=$(arg --net "$@"); k=1
      case "$net" in */audit/round-*) k=$(echo "$net" | sed 's|.*/audit/round-\([0-9]*\)/net/.*|\1|');; esac
      ok=true; up=0.00008; [ -f '{base}/rough-fail-'$k ] && {{ ok=false; up=0.0004; }}
      if [ -n "$sl" ]; then
        for n in $(grep -o '"name": "[^"]*"' "$sl" | sed 's/"name": //'); do
          [ -n "$s" ] && s="$s, "
          s="$s{{\"group\": $i, \"name\": $n, \"ids\": [0, 1], \"summary\": {{\"loss_upper95\": 0.0003}}, \"pass\": false, \"reached_share\": 0.2}}"
          i=$((i+1))
        done
      fi
      d1=""; for x in 0:1 1:1 2:1 3:1 4:2 5:2 6:2 7:2; do [ -n "$d1" ] && d1="$d1, "; d1="$d1{{\"id\": ${{x%:*}}, \"position\": ${{x#*:}}, \"worlds\": 4}}"; done
      d2=""; for x in 0:1 1:1 3:1 4:2 6:2; do [ -n "$d2" ] && d2="$d2, "; d2="$d2{{\"id\": ${{x%:*}}, \"position\": ${{x#*:}}, \"worlds\": 71}}"; done
      set1="{{\"set\": 1, \"burn_rule\": \"searched burn\", \"values_version\": 2, \"plan\": {{\"decisions\": 8, \"worlds\": 31}}, \"exact_values\": \"stored, NOT re-verified on this binary (--verify-pct 0)\", \"verification\": {{\"done\": false}}, \"groups\": [{{\"group\": \"all\", \"summary\": {{\"decisions\": 8, \"mean_loss\": 0.00003, \"loss_upper95\": 0.00007, \"loss_pass\": true, \"bias\": [0.00001, 0.0, 0.0, -0.00001], \"bias_pass\": true, \"agreement\": 0.75}}}}], \"pass\": true, \"decisions\": [$d1], \"slices\": {{\"groups\": [$s]}}}}"
      set2="{{\"set\": 2, \"burn_rule\": \"rules-v1 burn\", \"values_version\": 1, \"plan\": {{\"decisions\": 5, \"worlds\": 355}}, \"exact_values\": \"stored; 8 worlds recomputed by this binary and byte-identical\", \"verification\": {{\"done\": true}}, \"groups\": [{{\"group\": \"all\", \"summary\": {{\"decisions\": 5, \"mean_loss\": 0.00004, \"loss_upper95\": $up, \"loss_pass\": $ok, \"bias\": [0.00002, 0.0, 0.0, -0.00002], \"bias_pass\": true, \"agreement\": 0.8}}}}], \"pass\": $ok, \"decisions\": [$d2], \"slices\": {{\"groups\": [$s]}}}}"
      echo "{{\"sets\": [$set1, $set2], \"hard\": [], \"both_rules\": []}}" > "$(arg --json "$@")"
      echo "rough test: report"
      exit 0
    fi
    if [ -n "$sl" ]; then
      for n in $(grep -o '"name": "[^"]*"' "$sl" | sed 's/"name": //'); do
        [ -n "$s" ] && s="$s, "
        s="$s{{\"group\": $i, \"name\": $n, \"ids\": [0], \"summary\": {{\"loss_upper95\": 0.00004}}, \"pass\": true}}"
        i=$((i+1))
      done
    fi
    # The bound of round k's network (k 1: out/hybrid/net, else audit/round-k/net), judged by
    # `--standard` like the real check: a mini golden set's is 0.00004 (`mini-bounds`: the k-th
    # number), 0.0002 with `mini-fail-all` or while k is below the number in `mini-pass-from`;
    # the golden check's overall row's 0.00005 (`overall-bounds`: the k-th number), 0.0002 with
    # `overall-fail`.
    std=$(arg --standard "$@"); [ -z "$std" ] && std=0.01
    net=$(arg --net "$@"); k=1
    case "$net" in */audit/round-*) k=$(echo "$net" | sed 's|.*/audit/round-\([0-9]*\)/net/.*|\1|');; esac
    case "$sl" in */mini/*)
      up=0.00004; [ -f '{base}/mini-bounds' ] && up=$(cut -d' ' -f$k '{base}/mini-bounds')
      [ -f '{base}/mini-fail-all' ] && up=0.0002
      if [ -f '{base}/mini-pass-from' ] && [ "$k" -lt "$(cat '{base}/mini-pass-from')" ]; then up=0.0002; fi;;
      *) up=0.00005; [ -f '{base}/overall-bounds' ] && up=$(cut -d' ' -f$k '{base}/overall-bounds')
      [ -f '{base}/overall-fail' ] && up=0.0002;;
    esac
    ok=$(awk -v u="$up" -v s="$std" 'BEGIN {{ print (u <= s / 100) ? "true" : "false" }}')
    echo "{{\"pass\": $ok, \"pass_without_slices\": $ok, \"groups\": [{{\"group\": \"all\", \"summary\": {{\"decisions\": 2, \"mean_loss\": 0.00002, \"loss_upper95\": $up, \"loss_pass\": $ok, \"bias_pass\": true}}}}], \"net\": {{\"sha256\": \"abc\"}}, \"values\": {{}}, \"slices\": {{\"groups\": [$s]}}}}" > "$(arg --json "$@")"
    echo "standard: PASS";;
  *) exec '{search}' "$@";;
esac
"#,
                search = f.search.display(),
                sha = f.sha,
            ),
        );
        let device = if gpu {
            r#"{"device": "cuda", "gpu_name": "RTX A2000", "reason": "every GPU check passed"}"#
        } else {
            r#"{"device": "cpu", "reason": "nvidia-smi is missing or failed: no GPU in this container"}"#
        };
        let stop_line = stop.map_or(String::new(), |p| {
            format!(
                "if [ -f '{base}/stop-at-'$n ]; then touch '{}'; fi",
                p.display()
            )
        });
        let py = fake_script(
            &f.bin,
            "fake-python-audit",
            &format!(
                r#"{arg}
echo "python $*" >> '{calls}'
case "$2" in
  fortyone_nn.device) echo '{device}' > "$(arg --json "$@")";;
  fortyone_nn.train)
    out=$(arg --out "$@"); mkdir -p "$out"
    w=$(arg --watch-dir "$@")
    if [ -n "$w" ]; then
      want=$(arg --expect "$@"); ntrain=0; on=0
      for a in "$@"; do case "$a" in --train) on=1;; --*) on=0;; *) [ $on = 1 ] && ntrain=$((ntrain+1));; esac; done
      i=0
      while [ $i -lt 300 ]; do
        n=$(ls "$w" | grep -c '\.npy$'); [ $((n + ntrain)) -ge "$want" ] && break; sleep 0.1; i=$((i+1))
      done
      ls "$w" > "$out/seen.txt"
    fi
    echo "int $(arg --device "$@") $out" > "$out/int.safetensors"; echo float > "$out/float.safetensors"
    echo '{{"history": [{{"val_loss": 1.5}}], "best_epoch": 0, "float": {{}}, "int": {{"error": {{"sd": 0.01}}}}, "minutes": 0.1, "device": {{}}}}' > "$out/report.json"
    echo '{{"step": "done", "epoch": 1}}' > "$(arg --progress "$@")";;
  fortyone_nn.audit)
    out=$(arg --out "$@"); mkdir -p "$out"
    n=$(cat '{base}/audit-count' 2>/dev/null || echo 0); n=$((n+1)); echo $n > '{base}/audit-count'
    if [ -f '{base}/audit-'$n'.json' ]; then cp '{base}/audit-'$n'.json' "$out/predicates.json"; else echo '[]' > "$out/predicates.json"; fi
    ov=""; [ -f '{base}/audit-sd-'$n ] && ov=", \"overall\": {{\"sd\": $(cat '{base}/audit-sd-'$n)}}"
    echo '{{"thresholds": {{"gamma": 0.001, "alpha_bias": 0.0025, "alpha_sd": 0.01, "fdr": 0.05}}, "audit": {{"candidates": 5, "confirmed": 1}}'"$ov"'}}' > "$out/report.json"; echo "the audit" > "$out/report.txt"
    {stop_line};;
esac
"#,
            ),
        );
        let uv = fake_script(
            &f.bin,
            "uv-audit",
            &format!(
                r#"echo "uv $*" >> '{calls}'
case $1 in
  --version) echo "uv 0.13.0";;
  sync) mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"; cp '{py}' "$UV_PROJECT_ENVIRONMENT/bin/python";;
esac
"#,
                py = py.display()
            ),
        );
        Tools {
            search,
            uv,
            python_dir: f.py.clone(),
            nvidia_smi: if gpu {
                f.gpu.clone()
            } else {
                f.base.join("no-such-nvidia-smi")
            },
            remote: Some(Arc::new(crate::remote::DirRemote {
                root: f.base.join("remote"),
            })),
        }
    }

    /// Two confirmed groups, as the auditor writes them.
    const GROUPS_TWO: &str = r#"[{"name": "g-one", "conditions": [{"feature": 801, "op": "==", "value": 1}], "weight": 0.05, "bias": [0.004, -0.001, 0.0, -0.003], "sd": 0.03},
        {"name": "g-two", "conditions": [{"feature": 845, "op": "<=", "value": -3145728}], "weight": 0.01, "bias": [0.001, 0.0, 0.0, -0.001], "sd": 0.02}]"#;

    /// One confirmed group, as the auditor writes it.
    const GROUP_ONE: &str = r#"[{"name": "g-one", "conditions": [{"feature": 801, "op": "==", "value": 1}], "weight": 0.05, "bias": [0.004, -0.001, 0.0, -0.003], "sd": 0.03}]"#;

    #[cfg(unix)]
    fn audit_params(extra: Value) -> Params {
        let mut v = json!({ "labels_random": 7, "labels_rules": 0, "labels_harvest": 9,
                            "val_count": 3, "golden_count": 2, "golden_worlds": 4,
                            "widths": "16,8", "epochs": 1, "audit_rows": 50,
                            "audit_mini_decisions": 2, "audit_target_rows": 1000 });
        if let (Some(o), Value::Object(e)) = (v.as_object_mut(), extra) {
            o.extend(e);
        }
        parse(&v).expect("params")
    }

    #[cfg(unix)]
    fn steps_of(rung: &Value) -> Vec<String> {
        rung["steps"]
            .as_array()
            .expect("steps")
            .iter()
            .filter_map(|s| s["step"].as_str().map(str::to_string))
            .collect()
    }

    /// A loop that goes flat in two rounds, on a (fake) GPU: round 1 confirms a group whose
    /// golden slice is too small, so it gets a mini golden set, which fails with round 1's
    /// network (the group is remaining) and passes with round 2's; round 2 confirms nothing. Round
    /// 2's training starts beside round 1's CPU steps, watching the directory its targeted shard
    /// (with its weight) lands in, the mini set first (the golden check left flatness open); the
    /// heartbeat carries the loop's fields; the reports, the summary and the stop reason are
    /// written; the last network is the chunk's.
    #[cfg(unix)]
    #[test]
    fn audit_loop_goes_flat_in_two_rounds() {
        let f = fakes("auditflat");
        fs::write(f.base.join("audit-1.json"), GROUP_ONE).expect("write");
        fs::write(f.base.join("mini-pass-from"), "2").expect("write");
        let tools = audit_tools(&f, true, None);
        let params = audit_params(json!({ "audit_rounds": 3 }));
        let root = f.base.join("run");
        let mut ctx = Ctx::local("hybrid-test", &root, 16);
        let r = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("the loop runs");
        assert_eq!(r["verdict"], "PASS", "{r}");
        assert_eq!(
            r["audit"]["stop"],
            "flat after round 2 (1 groups confirmed by the auditor so far, every one within 0.01% \
             at the decision level)"
        );
        assert_eq!(r["audit"]["rounds"], 2);
        let out = root.join("out").join("hybrid");
        let rung = read_json(&out.join("rung.json")).expect("rung.json");
        let steps = steps_of(&rung);
        let at = |s: &str| {
            steps
                .iter()
                .position(|x| x == s)
                .unwrap_or_else(|| panic!("no step {s:?} in {steps:?}"))
        };
        for (a, b) in [
            ("golden worlds", "train (round 1)"),
            ("audit shard (round 1)", "train (round 1)"),
            ("train (round 1)", "audit (round 1)"),
            ("audit (round 1)", "golden slices (round 1)"),
            (
                "golden slices (round 1)",
                "mini golden set m1 decisions (round 1)",
            ),
            (
                "mini golden set m1 decisions (round 1)",
                "mini golden set m1 worlds (round 1)",
            ),
            (
                "mini golden set m1 worlds (round 1)",
                "mini golden set m1 (round 1)",
            ),
            ("mini golden set m1 (round 1)", "targeted harvest (round 1)"),
            ("targeted harvest (round 1)", "audit shard (round 2)"),
            ("audit shard (round 2)", "train (round 2)"),
            ("train (round 2)", "audit (round 2)"),
            ("audit (round 2)", "golden slices (round 2)"),
            ("golden slices (round 2)", "mini golden set m1 (round 2)"),
        ] {
            assert!(at(a) < at(b), "{a} before {b}: {steps:?}");
        }
        assert!(!steps.iter().any(|s| s == "targeted harvest (round 2)"));
        // Round 2 confirmed nothing but g-one was still failing, so round 3's training started
        // beside round 2's checks; g-one passed, so it was stopped unused.
        let r3: Vec<&String> = steps.iter().filter(|s| s.contains("round 3")).collect();
        assert_eq!(
            r3,
            ["train (round 3): not needed, round 2 is flat"],
            "{steps:?}"
        );
        let log = fs::read_to_string(&f.calls).expect("calls");
        let line = |p: &str| -> Vec<&str> { log.lines().filter(|l| l.starts_with(p)).collect() };
        // The audit shards: seeds 30101 and 30102, exact at trick 4, the harvest's settings.
        let audits = line("search labels --trick 4 --count 50");
        assert_eq!(audits.len(), 2, "{log}");
        assert!(
            audits[0].contains("--seed 30101 --source harvest")
                && audits[1].contains("--seed 30102")
        );
        assert!(audits.iter().all(|l| {
            l.contains("--harvest-cap 16 --harvest-positions 0.25,0.25,0.5")
                && !l.contains("--harvest-net")
        }));
        // The targeted harvest: round 1's groups, in-group positions only, into round 2's
        // watched directory, with its weight file.
        let target = line("search labels --trick 4 --count 1000 --seed 30151 --source harvest");
        assert_eq!(target.len(), 1, "{log}");
        let watch = std::path::absolute(root.join("hybrid-work/audit/train-r2")).expect("abs");
        assert!(
            target[0].contains(&format!(
                "--out {}",
                watch.join("t4-target-r1-s30151.npy").display()
            )),
            "{}",
            target[0]
        );
        assert!(
            target[0].contains("--harvest-cap 0 --harvest-positions 0.25,0.25,0.5 --harvest-keep")
        );
        assert!(target[0].ends_with("--keep-cap 256"));
        assert_eq!(
            fs::read_to_string(watch.join("t4-target-r1-s30151.npy.weight")).expect("weight"),
            "2\n"
        );
        // Round 2's training: the base shards as files, the targeted one through the directory.
        // Round 3's started (g-one still failing after round 2's audit) and was stopped unused.
        let trains = line("python -m fortyone_nn.train");
        assert_eq!(trains.len(), 3, "{log}");
        assert!(!trains[0].contains("--watch-dir"));
        assert!(
            trains[2].contains("hybrid-work/audit/train-r3 --expect 4"),
            "{}",
            trains[2]
        );
        assert!(!out.join("audit/round-3").exists());
        assert!(
            trains[1].contains(&format!(
                "--watch-dir {} --expect 3 --load-threads 8",
                watch.display()
            )),
            "{}",
            trains[1]
        );
        assert!(trains[1].contains("--widths 16,8") && trains[1].contains("--device cuda"));
        assert_eq!(
            fs::read_to_string(out.join("audit/round-2/net/seen.txt")).expect("seen"),
            "t4-target-r1-s30151.npy\nt4-target-r1-s30151.npy.json\nt4-target-r1-s30151.npy.weight\n"
        );
        // The auditor on each round's network and audit shard; the golden slices with the union.
        let aud = line("python -m fortyone_nn.audit");
        assert_eq!(aud.len(), 2);
        assert!(
            aud[0].contains("out/hybrid/net/int.safetensors --shard")
                && aud[0].contains("--seed 30101 --gamma 0.001 --workers 16 --work ")
                && aud[0].ends_with("hybrid-work/audit/auditor-r1"),
            "{}",
            aud[0]
        );
        assert!(aud[1].contains("audit/round-2/net/int.safetensors"));
        let checks = line("search hybrid-check");
        assert!(
            checks[0].contains("--verify-pct 1 --standard 0.01 --slices")
                && checks[0].contains("round-1/union.json --slice-share 0.5"),
            "{}",
            checks[0]
        );
        assert!(checks.iter().skip(1).all(|c| c.contains("--verify-pct 0")));
        let keep = line("search decisions --keep");
        assert_eq!(keep.len(), 1, "the mini set is made once");
        assert!(keep[0].contains("--keep-share 0.5 --max-candidates 400 --keep-worlds 8 --trick 3 --cut 4 --count 2 --seed 1 --first-candidate 2 --net"), "{}", keep[0]);
        // Each audit shard is deleted once its auditor has run; its sidecar stays in round-<r>/.
        for r in 1..=2 {
            let shard = root.join(format!("hybrid-work/audit/t4-audit-r{r}-s3010{r}.npy"));
            assert!(!shard.exists(), "{} kept", shard.display());
            assert!(
                out.join(format!("audit/round-{r}/t4-audit-r{r}-s3010{r}.npy.json"))
                    .is_file()
            );
        }
        // Reports.
        let audit = out.join("audit");
        for x in [
            "round-1.txt",
            "round-1.json",
            "round-2.txt",
            "round-2.json",
            "audit-summary.txt",
            "round-1/predicates.json",
            "round-1/union.json",
            "round-1/auditor/report.txt",
            "round-2/union.json",
            "mini/m1/group.json",
            "mini/m1/decisions.jsonl",
            "mini/m1/exact-t3-n4.jsonl",
            "round-1/mini-m1.json",
            "round-2/mini-m1.json",
            "round-1/t4-target-r1-s30151.npy.json",
            "round-2/t4-audit-r2-s30102.npy.json",
        ] {
            assert!(audit.join(x).is_file(), "audit/{x} missing");
        }
        let summary = fs::read_to_string(audit.join("audit-summary.txt")).expect("summary");
        let lines: Vec<&str> = summary.lines().collect();
        assert!(lines[0].starts_with("round 1: 1 groups confirmed (5 found) -> 1 remaining, 1 failing; golden 95% upper 0.0050% (PASS)"), "{summary}");
        assert!(
            lines[1].starts_with("round 2: 0 groups confirmed (5 found) -> 0 remaining, 0 failing"),
            "{summary}"
        );
        assert!(
            lines[2].starts_with("best: round 2 (flat at 0.01%; golden 95% upper 0.0050%)"),
            "{summary}"
        );
        assert!(
            lines[3].starts_with("stop: flat after round 2"),
            "{summary}"
        );
        assert!(
            lines[4].starts_with("final network: audit/round-2/net (sha256 "),
            "{summary}"
        );
        let r1 = read_json(&audit.join("round-1.json")).expect("round 1");
        assert_eq!(r1["groups_confirmed"][0]["name"], "g-one");
        assert_eq!(r1["golden"]["slices"][0]["decisions"], 1);
        assert_eq!(r1["mini_sets"][0]["decisions"], 2);
        assert_eq!(r1["mini_sets"][0]["pass"], false);
        assert_eq!(r1["decision_level"][0]["basis"], "mini golden set");
        assert_eq!(r1["groups_remaining"], json!(["g-one"]));
        assert_eq!(
            r1["groups_failing"],
            json!([{ "group": "g-one", "failing_since": 1 }])
        );
        assert_eq!(r1["targeted"]["why"], "1 groups confirmed this round");
        assert_eq!(r1["targeted"]["targets"]["confirmed"], json!(["g-one"]));
        assert_eq!(r1["calibration"]["pairs"][0]["sd"], 0.03);
        assert_eq!(r1["targeted"]["rows"], 1000);
        assert_eq!(r1["targeted"]["weight"], 2.0);
        assert_eq!(
            (r1["shape"].as_str(), r1["shape_step"].as_u64()),
            (Some("16,8"), Some(1))
        );
        assert!(
            r1["shared"]["report_json"]
                .as_str()
                .is_some_and(|k| k.starts_with("inputs/sha256/"))
        );
        assert!(
            r1["shared"]["predicates"]
                .as_str()
                .is_some_and(|k| k.starts_with("inputs/sha256/"))
        );
        let r2 = read_json(&audit.join("round-2.json")).expect("round 2");
        assert_eq!(r2["decision_level"][0]["pass"], true);
        assert_eq!(r2["flat"], true);
        let txt = fs::read_to_string(audit.join("round-1.txt")).expect("txt");
        assert!(
            txt.contains("g-one: weight 5.00%, bias +0.40%, SD 3.00%"),
            "{txt}"
        );
        assert!(
            txt.contains("Harvested for the next round: 1000 positions"),
            "{txt}"
        );
        assert!(
            txt.contains("g-one: SD 3.00% on the audit shard; mini golden set -> 95% upper"),
            "{txt}"
        );
        // The last network is the chunk's; every round's network is shared.
        assert_eq!(rung["net"]["dir"], "audit/round-2/net");
        assert!(
            r["net_int_key"]
                .as_str()
                .is_some_and(|k| k.starts_with("inputs/sha256/"))
        );
        assert!(rung["net_r2"]["int_key"].as_str().is_some());
        assert!(
            rung["audit"]["stop"]
                .as_str()
                .is_some_and(|s| s.starts_with("flat after round 2"))
        );
        assert_eq!(rung["check"]["verdict"], "PASS");
        assert!(out.join("hc-t3-o4.json").is_file());
        // The heartbeat carries the loop's fields.
        let hb = read_json(&root.join("heartbeat.json")).expect("heartbeat");
        let p = &hb["progress"];
        assert_eq!(p["round"], "2 of 3");
        assert_eq!(
            (
                p["groups_found"].as_u64(),
                p["groups_confirmed"].as_u64(),
                p["groups_remaining"].as_u64()
            ),
            (Some(5), Some(0), Some(0))
        );
        assert_eq!(p["groups_failing"], 0);
        assert_eq!(p["worst_group"], "none");
        assert_eq!(p["slices_failing"], "0 of 1");
        assert_eq!(
            p["last_check"],
            "round 2: mean loss 0.0020%, 95% upper 0.0050% (PASS)"
        );
        assert_eq!(
            (p["shape"].as_str(), p["shape_step"].as_str()),
            (Some("16,8"), Some("1 of 4"))
        );
        assert_eq!(
            p["thresholds"],
            "gamma 0.10%, alpha_bias 0.25%, alpha_sd 1.00%, fdr 5%"
        );
        assert!(
            p["audit_stop"]
                .as_str()
                .is_some_and(|s| s.starts_with("flat after round 2"))
        );
        let _ = fs::remove_dir_all(&f.base);
    }

    /// A round whose confirmed group is within 0.01% at the decision level is flat: on a GPU the
    /// next round's training, started beside its checks, is stopped unused and nothing is
    /// harvested.
    #[cfg(unix)]
    #[test]
    fn audit_loop_flat_with_a_confirmed_group() {
        let f = fakes("auditflat1");
        fs::write(f.base.join("audit-1.json"), GROUP_ONE).expect("write");
        let tools = audit_tools(&f, true, None);
        let params = audit_params(json!({ "audit_rounds": 3 }));
        let root = f.base.join("run");
        let mut ctx = Ctx::local("hybrid-test", &root, 16);
        let r = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("the loop runs");
        assert_eq!(r["verdict"], "PASS", "{r}");
        assert!(
            r["audit"]["stop"]
                .as_str()
                .is_some_and(|s| s.starts_with("flat after round 1 (1 groups")),
            "{r}"
        );
        let rung = read_json(&root.join("out/hybrid/rung.json")).expect("rung.json");
        let steps = steps_of(&rung);
        assert!(
            steps
                .iter()
                .any(|s| s == "train (round 2): not needed, round 1 is flat"),
            "{steps:?}"
        );
        assert!(
            !steps.iter().any(|s| s.starts_with("targeted harvest")),
            "{steps:?}"
        );
        assert_eq!(rung["net"]["dir"], "net");
        let log = fs::read_to_string(&f.calls).expect("calls");
        assert!(!log.contains("--harvest-keep"), "{log}");
        let _ = fs::remove_dir_all(&f.base);
    }

    /// A loop stopped by its round limit, on the CPU: every round confirms two groups, the
    /// targeted shard of round 1 (with its weight) joins round 2's training as a file, round 2
    /// harvests nothing (no round 3) but still checks its slices and mini sets; one new mini set
    /// a round (`audit_mini_sets` 1), the most severe group's first.
    #[cfg(unix)]
    #[test]
    fn audit_loop_stops_at_the_round_limit() {
        let f = fakes("auditlimit");
        for n in 1..=2 {
            fs::write(f.base.join(format!("audit-{n}.json")), GROUPS_TWO).expect("write");
        }
        fs::write(f.base.join("mini-fail-all"), "").expect("write");
        let tools = audit_tools(&f, false, None);
        let params = audit_params(json!({ "audit_rounds": 2, "audit_target_rows": 4000,
                                           "audit_weight": 3, "audit_mini_sets": 1 }));
        let root = f.base.join("run");
        let mut ctx = Ctx::local("hybrid-test", &root, 4);
        let r = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("the loop runs");
        assert_eq!(r["verdict"], "FAIL");
        assert_eq!(
            r["audit"]["stop"],
            "round limit (2 rounds; 2 groups still failing at the decision level: the next \
             lever is network capacity)"
        );
        let log = fs::read_to_string(&f.calls).expect("calls");
        let line = |p: &str| -> Vec<&str> { log.lines().filter(|l| l.starts_with(p)).collect() };
        assert_eq!(
            line("search labels --trick 4 --count 4000").len(),
            1,
            "{log}"
        );
        let trains = line("python -m fortyone_nn.train");
        assert_eq!(trains.len(), 2);
        let target =
            std::path::absolute(root.join("hybrid-work/audit/targets/t4-target-r1-s30151.npy"))
                .expect("abs");
        assert!(
            trains[1].contains(&format!("{} --val", target.display())),
            "{}",
            trains[1]
        );
        assert!(!trains[1].contains("--watch-dir") && trains[1].contains("--device cpu"));
        assert_eq!(
            fs::read_to_string(format!("{}.weight", target.display())).expect("weight"),
            "3\n"
        );
        let rung = read_json(&root.join("out/hybrid/rung.json")).expect("rung.json");
        let steps = steps_of(&rung);
        let i = |s: &str| {
            steps
                .iter()
                .position(|x| x == s)
                .unwrap_or_else(|| panic!("{s} in {steps:?}"))
        };
        assert!(i("targeted harvest (round 1)") < i("train (round 2)"));
        assert!(i("audit (round 2)") < i("golden slices (round 2)"));
        assert!(i("golden slices (round 2)") < i("mini golden set m1 (round 2)"));
        assert!(!steps.iter().any(|s| s == "targeted harvest (round 2)"));
        // One new mini set a round: g-one's in round 1 (g-two waits), g-two's in round 2.
        assert!(i("mini golden set m1 decisions (round 1)") < i("audit shard (round 2)"));
        assert!(
            !steps
                .iter()
                .any(|s| s == "mini golden set m2 decisions (round 1)")
        );
        // Round 2: the earlier mini set is re-measured before the new one is made.
        assert!(i("mini golden set m1 (round 2)") < i("mini golden set m2 decisions (round 2)"));
        let r1 = read_json(&root.join("out/hybrid/audit/round-1.json")).expect("round 1");
        assert_eq!(r1["mini_sets"][1]["waiting"], json!(["g-two"]));
        let r1t = fs::read_to_string(root.join("out/hybrid/audit/round-1.txt")).expect("txt");
        assert!(r1t.contains("No mini golden set yet for 1 groups"), "{r1t}");
        let summary =
            fs::read_to_string(root.join("out/hybrid/audit/audit-summary.txt")).expect("summary");
        assert!(
            summary.contains("round 2: 2 groups confirmed (5 found) -> 2 remaining, 2 failing"),
            "{summary}"
        );
        assert!(summary.contains("stop: round limit"), "{summary}");
        let hb = read_json(&root.join("heartbeat.json")).expect("heartbeat");
        assert_eq!(hb["progress"]["round"], "2 of 2");
        assert_eq!(hb["progress"]["groups_remaining"], 2);
        assert_eq!(hb["progress"]["groups_failing"], 2);
        assert!(
            hb["progress"]["worst_group"]
                .as_str()
                .is_some_and(|w| w.starts_with("g-one: weight 5.00%"))
        );
        let _ = fs::remove_dir_all(&f.base);
    }

    /// A failing group stays a target until its bound passes: round 1 confirms g-one, whose mini
    /// golden set fails; round 2 confirms nothing, but g-one still fails (re-measured with round
    /// 2's network), so round 2 re-targets it; round 3's network passes it: flat after round 3.
    #[cfg(unix)]
    #[test]
    fn audit_loop_re_targets_failing_groups() {
        let f = fakes("auditretarget");
        fs::write(f.base.join("audit-1.json"), GROUP_ONE).expect("write");
        fs::write(f.base.join("mini-pass-from"), "3").expect("write");
        let tools = audit_tools(&f, false, None);
        let params = audit_params(json!({ "audit_rounds": 4 }));
        let root = f.base.join("run");
        let mut ctx = Ctx::local("hybrid-test", &root, 4);
        let r = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("the loop runs");
        assert_eq!(r["verdict"], "PASS", "{r}");
        assert!(
            r["audit"]["stop"]
                .as_str()
                .is_some_and(|s| s.starts_with("flat after round 3 (1 groups")),
            "{r}"
        );
        let log = fs::read_to_string(&f.calls).expect("calls");
        let line = |p: &str| -> Vec<&str> { log.lines().filter(|l| l.starts_with(p)).collect() };
        // Two targeted harvests (rounds 1 and 2), both on g-one; round 3 trains on both.
        let targets = line("search labels --trick 4 --count 1000");
        assert_eq!(targets.len(), 2, "{log}");
        assert!(targets[0].contains("--seed 30151") && targets[0].contains("round-1/targets.json"));
        assert!(targets[1].contains("--seed 30152") && targets[1].contains("round-2/targets.json"));
        let trains = line("python -m fortyone_nn.train");
        assert_eq!(trains.len(), 3, "{log}");
        let tdir = std::path::absolute(root.join("hybrid-work/audit/targets")).expect("abs");
        for x in ["t4-target-r1-s30151.npy", "t4-target-r2-s30152.npy"] {
            assert!(
                trains[2].contains(&tdir.join(x).display().to_string()),
                "{}",
                trains[2]
            );
        }
        assert!(!trains[1].contains("t4-target-r2"));
        assert_eq!(
            line("search decisions --keep").len(),
            1,
            "one mini set, re-measured"
        );
        let out = root.join("out/hybrid/audit");
        let t2 = read_json(&out.join("round-2/targets.json")).expect("targets");
        assert_eq!(t2[0]["name"], "g-one");
        let r2 = read_json(&out.join("round-2.json")).expect("round 2");
        assert_eq!(r2["groups_confirmed"], json!([]));
        assert_eq!(r2["decision_level"][0]["basis"], "mini golden set");
        assert_eq!(r2["decision_level"][0]["pass"], false);
        assert_eq!(
            r2["groups_failing"],
            json!([{ "group": "g-one", "failing_since": 1 }])
        );
        assert_eq!(r2["targeted"]["why"], "re-targeting 1 groups still failing");
        assert_eq!(
            r2["targeted"]["targets"]["failing"],
            json!([{ "group": "g-one", "failing_since": 1 }])
        );
        let t2 = fs::read_to_string(out.join("round-2.txt")).expect("txt");
        assert!(
            t2.contains("(re-targeting 1 groups still failing; seed 30152"),
            "{t2}"
        );
        assert!(
            t2.contains("Failing groups (targets until their bound passes): g-one (since round 1)"),
            "{t2}"
        );
        let r3 = read_json(&out.join("round-3.json")).expect("round 3");
        assert_eq!(r3["decision_level"][0]["pass"], true);
        assert_eq!(r3["groups_failing"], json!([]));
        assert_eq!(r3["flat"], true);
        assert!(r3["targeted"].is_null());
        assert!(!out.join("round-4.json").exists());
        let summary = fs::read_to_string(out.join("audit-summary.txt")).expect("summary");
        let lines: Vec<&str> = summary.lines().collect();
        assert!(
            lines[0].starts_with("round 1: 1 groups confirmed (5 found) -> 1 remaining, 1 failing"),
            "{summary}"
        );
        assert!(
            lines[1].starts_with("round 2: 0 groups confirmed (5 found) -> 1 remaining, 1 failing"),
            "{summary}"
        );
        assert!(
            lines[2].starts_with("round 3: 0 groups confirmed (5 found) -> 0 remaining, 0 failing"),
            "{summary}"
        );
        assert!(
            lines[3].starts_with("best: round 3 (flat at 0.01%"),
            "{summary}"
        );
        assert!(
            lines[4].starts_with("stop: flat after round 3"),
            "{summary}"
        );
        let rung = read_json(&root.join("out/hybrid/rung.json")).expect("rung.json");
        assert_eq!(rung["net"]["dir"], "audit/round-3/net");
        assert!(!steps_of(&rung).iter().any(|s| s.contains("round 4")));
        let hb = read_json(&root.join("heartbeat.json")).expect("heartbeat");
        assert_eq!(hb["progress"]["round"], "4 of 4");
        assert_eq!(
            (
                hb["progress"]["groups_remaining"].as_u64(),
                hb["progress"]["groups_failing"].as_u64()
            ),
            (Some(0), Some(0))
        );
        let _ = fs::remove_dir_all(&f.base);
    }

    /// A round that confirms nothing, with nothing failing, while the golden set's overall row
    /// fails: nothing to target, decided before another training starts.
    #[cfg(unix)]
    #[test]
    fn audit_loop_nothing_to_target() {
        let f = fakes("auditnothing");
        fs::write(f.base.join("overall-fail"), "").expect("write");
        let tools = audit_tools(&f, true, None);
        let params = audit_params(json!({ "audit_rounds": 3 }));
        let root = f.base.join("run");
        let mut ctx = Ctx::local("hybrid-test", &root, 16);
        let r = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("the loop runs");
        assert_eq!(r["verdict"], "FAIL");
        assert_eq!(
            r["audit"]["stop"],
            "nothing to target after round 1: no group confirmed and none still failing, but the \
             golden set's overall row fails (the next lever is network capacity)"
        );
        let rung = read_json(&root.join("out/hybrid/rung.json")).expect("rung.json");
        let steps = steps_of(&rung);
        assert!(!steps.iter().any(|s| s.contains("round 2")), "{steps:?}");
        let log = fs::read_to_string(&f.calls).expect("calls");
        assert_eq!(
            log.lines()
                .filter(|l| l.starts_with("python -m fortyone_nn.train"))
                .count(),
            1
        );
        let _ = fs::remove_dir_all(&f.base);
    }

    /// The trick-2 loop (cut 2; owner decision of 2026-10-16), on a fake GPU and on the CPU:
    /// round 1 has no earlier network, so its audit shard is made after its training, driven by
    /// it; its audit confirms g-one, which is targeted (the harvest driven by network 1), and
    /// its rough test on the saved trick-1 worlds fails overall, with g-one's slice failing too.
    /// Round 2 confirms nothing: flat after round 2, although g-one's slice still fails (the
    /// rough test never gates and never makes a group failing). Every round's rough test is
    /// `search hybrid-check --trick 1 --cut 2 --saved …` with the slices (the verification
    /// sample in round 1 only); no golden set and no mini golden set is made; the reports say
    /// plainly how thin the trick-1 evidence is (32 decisions in the file, 8 with saved worlds).
    #[cfg(unix)]
    #[test]
    fn audit_loop_at_trick_2() {
        for gpu in [true, false] {
            let f = fakes(if gpu { "audit2gpu" } else { "audit2cpu" });
            fs::write(f.base.join("audit-1.json"), GROUP_ONE).expect("write");
            fs::write(f.base.join("rough-fail-1"), "").expect("write");
            let tools = audit_tools(&f, gpu, None);
            let remote = tools.remote.clone().expect("a remote store");
            let store = |name: &str, bytes: &[u8]| {
                let p = f.base.join(name);
                fs::write(&p, bytes).expect("write");
                put_input(&*remote, &p, None).expect("stored").0
            };
            let decisions: String = (0..32)
                .map(|i| format!("{{\"id\":{i},\"trick\":1}}\n"))
                .collect();
            let dec = store("decisions-t1.jsonl", decisions.as_bytes());
            let tables = store("tables.ckpt", b"{\"id\":0,\"w\":3}\n");
            let ab1 = store("ab1.ckpt", b"{\"id\":0,\"w\":5}\n");
            let params = parse(&json!({
                "cut": 2, "labels_random": 7, "labels_rules": 0, "val_count": 3,
                "widths": "16,8", "epochs": 1, "audit_rounds": 3, "audit_rows": 50,
                "audit_target_rows": 1000, "upload_shards": true, "partial_epochs": 2,
                "t1_decisions_input": format!("inputs/sha256/{dec}"),
                "t1_saved_inputs": [ { "run": "m3-t1-tables-1", "input": format!("sha256:{tables}") },
                                     { "run": "m3-agree-t1-ab1", "input": format!("sha256:{ab1}") } ] }))
            .expect("params");
            let root = f.base.join("run");
            let mut ctx = Ctx::local("hybrid-test", &root, 16);
            let r =
                chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("the loop runs");
            assert_eq!(r["verdict"], "PASS", "{r}");
            let stop = r["audit"]["stop"].as_str().expect("stop");
            assert!(
                stop.starts_with(
                    "flat after round 2: no weak group confirmed at the position level on the \
                     fresh trick-2 audit shard (1 groups confirmed in earlier rounds"
                ),
                "{stop}"
            );
            let log = fs::read_to_string(&f.calls).expect("calls");
            let line =
                |p: &str| -> Vec<&str> { log.lines().filter(|l| l.starts_with(p)).collect() };
            // No golden set and no mini golden set at trick 2.
            assert!(
                line("search decisions").is_empty() && line("search worlds").is_empty(),
                "{log}"
            );
            // On the GPU round 2 trains beside round 1's targeted harvest (hours at trick 2):
            // at most `partial_epochs` epochs before that shard lands, then it waits.
            let streamed: Vec<&str> = line("python -m fortyone_nn.train")
                .into_iter()
                .filter(|l| l.contains("--watch-dir"))
                .collect();
            assert_eq!(streamed.len(), usize::from(gpu), "{log}");
            assert!(
                streamed.iter().all(|l| l.contains("--partial-epochs 2")),
                "{log}"
            );
            // The rough tests: the t1check's command with the slices; the sample in round 1 only.
            let checks = line("search hybrid-check");
            assert_eq!(checks.len(), 2, "{log}");
            let inputs = std::path::absolute(root.join("hybrid-work/inputs")).expect("abs");
            for (k, c) in checks.iter().enumerate() {
                assert!(
                    c.starts_with("search hybrid-check --trick 1 --cut 2 --net "),
                    "{c}"
                );
                for want in [
                    format!("--decisions {}", inputs.join(&dec).display()),
                    format!(
                        "--saved m3-t1-tables-1={} --saved m3-agree-t1-ab1={}",
                        inputs.join(&tables).display(),
                        inputs.join(&ab1).display()
                    ),
                    format!("round-{}/union.json --slice-share 0.5", k + 1),
                    format!(
                        "--json {}",
                        root.join("out/hybrid/audit")
                            .join(format!("round-{}", k + 1))
                            .join("t1check.json")
                            .display()
                    ),
                ] {
                    assert!(c.contains(&want), "{want} in {c}");
                }
                let verify = if k == 0 {
                    "--verify-pct 3 --verify-rule rules-v1 --verify-solve-threads 46"
                } else {
                    "--verify-pct 0 --verify-rule rules-v1"
                };
                assert!(c.contains(verify), "{c}");
            }
            assert!(checks[0].contains("out/hybrid/net/int.safetensors"));
            assert!(checks[1].contains("audit/round-2/net/int.safetensors"));
            // Round 1's audit shard after its training, driven by network 1; round 2's driven by
            // network 1 as well (beside or before round 2's training).
            let net1 =
                std::path::absolute(root.join("out/hybrid/net/int.safetensors")).expect("abs");
            let shards = line("search labels --trick 2 --count 50");
            assert_eq!(shards.len(), 2, "{log}");
            for s in &shards {
                assert!(
                    s.contains("--harvest-cap 16 --harvest-positions 0.25,0.25,0.5")
                        && s.ends_with(&format!("--harvest-net {}", net1.display())),
                    "{s}"
                );
            }
            // One targeted harvest (round 1's g-one), driven by network 1.
            let target = line("search labels --trick 2 --count 1000 --seed 30151 --source harvest");
            assert_eq!(target.len(), 1, "{log}");
            assert!(
                target[0].contains("--harvest-keep")
                    && target[0]
                        .ends_with(&format!("--keep-cap 256 --harvest-net {}", net1.display())),
                "{}",
                target[0]
            );
            let rung = read_json(&root.join("out/hybrid/rung.json")).expect("rung.json");
            let steps = steps_of(&rung);
            let at = |s: &str| {
                steps
                    .iter()
                    .position(|x| x == s)
                    .unwrap_or_else(|| panic!("no step {s:?} in {steps:?}"))
            };
            for (a, b) in [
                ("train (round 1)", "audit shard (round 1)"),
                ("audit shard (round 1)", "audit (round 1)"),
                ("targeted harvest (round 1)", "rough test (round 1)"),
                ("rough test (round 1)", "audit shard (round 2)"),
                ("audit (round 2)", "rough test (round 2)"),
            ] {
                assert!(at(a) < at(b), "{a} before {b}: {steps:?}");
            }
            assert!(!steps.iter().any(|s| s.starts_with("golden")), "{steps:?}");
            assert!(!steps.iter().any(|s| s == "targeted harvest (round 2)"));
            assert_eq!(rung["inputs"]["t1_decisions"]["trick1_decisions"], 32);
            assert!(rung["inputs"]["t1_saved:m3-agree-t1-ab1"]["sha256"] == ab1.as_str());
            // Reports.
            let out = root.join("out/hybrid/audit");
            let r1 = read_json(&out.join("round-1.json")).expect("round 1");
            assert_eq!(r1["rough_test"]["overall_pass"], false);
            assert_eq!(r1["rough_test"]["file_decisions"], 32);
            assert_eq!(r1["rough_test"]["decisions_with_worlds"], 8);
            assert_eq!(r1["rough_test"]["seats"], json!([1, 2]));
            assert_eq!(r1["rough_test"]["slices"][0]["label"], "thin: 4 decisions");
            assert_eq!(r1["position_level"][0]["pass"], false);
            assert_eq!(r1["groups_remaining"], json!(["g-one"]));
            assert_eq!(r1["groups_failing"], json!([]));
            assert_eq!(r1["targeted"]["targets"]["confirmed"], json!(["g-one"]));
            assert!(r1.get("golden").is_none() && r1.get("decision_level").is_none());
            assert_eq!(r1["audit_shard"]["driver"], net1.display().to_string());
            assert!(r1["audit_shard"]["driver_note"].is_string());
            // With upload_shards the loop's shards (hours of exact labels at trick 2) are shared.
            for k in [&r1["audit_shard"]["key"], &r1["targeted"]["key"]] {
                assert!(
                    k.as_str().is_some_and(|k| k.starts_with("inputs/sha256/")),
                    "{r1}"
                );
            }
            let r2 = read_json(&out.join("round-2.json")).expect("round 2");
            assert_eq!(r2["flat"], true);
            assert_eq!(r2["rough_test"]["overall_pass"], true);
            assert_eq!(r2["position_level"][0]["pass"], true);
            assert_eq!(
                r2["position_level"][0]["rough"]["pass"], false,
                "still failing"
            );
            assert_eq!(r2["groups_failing"], json!([]));
            assert!(r2["targeted"].is_null());
            let t1 = fs::read_to_string(out.join("round-1.txt")).expect("txt");
            for want in [
                "32 decisions in the saved trick-1 decisions file, 8 of them with saved exact \
                 worlds (searched burn: 8 decisions, 31 worlds; rules-v1 burn: 5 decisions, 355 \
                 worlds), at seats 1-2 only (seats 3-8 have no saved decision)",
                "far weaker evidence than the 400 decisions x 320 worlds of the golden sets",
                "it never decides the stop and never makes a group failing",
                "rules-v1 burn (values version 1): 5 decisions, 355 worlds; decision loss mean \
                 0.0040%, 95% upper 0.0400% (over the 0.01% line)",
                "g-one: SD 3.00% on the audit shard; confirmed this round (OUT OF SPEC); rough \
                 test: thin: 4 decisions, 95% upper 0.0300% (over; report only)",
                "Mini golden sets: none at trick 2",
            ] {
                assert!(t1.contains(want), "{want:?} in\n{t1}");
            }
            let summary = fs::read_to_string(out.join("audit-summary.txt")).expect("summary");
            let lines: Vec<&str> = summary.lines().collect();
            assert!(
                lines[0].starts_with(
                    "round 1: 1 groups confirmed (5 found) at the position level; rough test \
                     (report only): searched burn 95% upper 0.0070% (8 decisions, would pass), \
                     rules-v1 burn 95% upper 0.0400% (5 decisions, would fail); 8 of 32 trick-1 \
                     decisions have saved worlds"
                ),
                "{summary}"
            );
            assert!(
                lines[1].starts_with(
                    "round 2: 0 groups confirmed (5 found) at the position level (flat)"
                ),
                "{summary}"
            );
            // The kept round (no golden set at trick 2): flat at the position level first.
            assert!(
                lines[2].starts_with(
                    "best: round 2 (flat at the position level; overall SD 1.000% on its own \
                     fresh audit shard)"
                ),
                "{summary}"
            );
            assert!(lines[3].starts_with("trick 2: the standard is the position-level audit"));
            assert!(
                lines[4].starts_with("stop: flat after round 2"),
                "{summary}"
            );
            assert_eq!(rung["audit"]["best_round"], 2);
            assert_eq!(rung["net"]["dir"], "audit/round-2/net");
            // Each audit shard was shared when it was made (exact trick-2 labels, kept for
            // later runs), and only then deleted after its auditor.
            for r in 1..=2 {
                let shard = root.join(format!("hybrid-work/audit/t2-audit-r{r}-s3010{r}.npy"));
                assert!(!shard.exists(), "{} kept", shard.display());
            }
            assert_eq!(
                r1["audit_shard"]["local_file"],
                "deleted after the auditor, once shared"
            );
            assert!(rung["audit"]["settings"]["rough_test"].is_object());
            assert_eq!(rung["check"]["verdict"], "PASS");
            assert_eq!(rung["check"]["rough_test"]["file_decisions"], 32);
            assert!(root.join("out/hybrid/hc-t1-o2.json").is_file());
            let hb = read_json(&root.join("heartbeat.json")).expect("heartbeat");
            let p = &hb["progress"];
            assert_eq!(p["groups_failing"], 0);
            assert_eq!(p["groups_remaining"], 0);
            assert!(
                p["last_check"].as_str().is_some_and(|c| c.starts_with(
                    "round 2: 0 groups confirmed at the position level (flat); rough test (report only)"
                )),
                "{p}"
            );
            assert!(
                p["slices_failing"]
                    .as_str()
                    .is_some_and(|s| s.contains("rough test, report only")),
                "{p}"
            );
            let _ = fs::remove_dir_all(&f.base);
        }
    }

    /// The trick-2 loop without a flat round, on one shape (fake GPU and CPU): every round's
    /// audit confirms g-one at the same SD, so rounds 2 and 3 (both on the largest, only, shape)
    /// lower the worst confirmed group's SD by less than `audit_min_gain`: diminishing returns at
    /// the position level after round 3, before the round limit, with no targeted harvest in
    /// round 3. The kept round is the one with the lowest overall SD on its own audit shard
    /// (1.2%, 0.9%, 1.0%: round 2), never the rough test's. Without `upload_shards` the audit
    /// shards (exact trick-2 labels) are kept on the disk, with a warning, not deleted.
    #[cfg(unix)]
    #[test]
    fn audit_loop_at_trick_2_keeps_the_best_round_and_stops_on_diminishing_returns() {
        for gpu in [true, false] {
            let f = fakes(if gpu { "audit2dimgpu" } else { "audit2dimcpu" });
            for (n, sd) in [(1, "0.012"), (2, "0.009"), (3, "0.010"), (4, "0.011")] {
                fs::write(f.base.join(format!("audit-{n}.json")), GROUP_ONE).expect("write");
                fs::write(f.base.join(format!("audit-sd-{n}")), sd).expect("write");
            }
            let tools = audit_tools(&f, gpu, None);
            let remote = tools.remote.clone().expect("a remote store");
            let store = |name: &str, bytes: &[u8]| {
                let p = f.base.join(name);
                fs::write(&p, bytes).expect("write");
                put_input(&*remote, &p, None).expect("stored").0
            };
            let decisions: String = (0..32)
                .map(|i| format!("{{\"id\":{i},\"trick\":1}}\n"))
                .collect();
            let dec = store("decisions-t1.jsonl", decisions.as_bytes());
            let tables = store("tables.ckpt", b"{\"id\":0,\"w\":3}\n");
            let params = parse(&json!({
                "cut": 2, "labels_random": 7, "labels_rules": 0, "val_count": 3,
                "widths": "16,8", "epochs": 1, "audit_rounds": 4, "audit_rows": 50,
                "audit_target_rows": 1000, "audit_shapes": "<widths>",
                "t1_decisions_input": format!("inputs/sha256/{dec}"),
                "t1_saved_inputs": [ { "run": "m3-t1-tables-1", "input": format!("sha256:{tables}") } ] }))
            .expect("params");
            let root = f.base.join("run");
            let mut ctx = Ctx::local("hybrid-test", &root, 16);
            let r =
                chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("the loop runs");
            assert_eq!(r["verdict"], "FAIL", "{r}");
            let stop = r["audit"]["stop"].as_str().expect("stop");
            assert!(
                stop.starts_with(
                    "diminishing returns (rounds 2 and 3 on the largest shape, 16,8, each lowered \
                     the worst confirmed group's SD on its fresh audit shard by less than 15%"
                ),
                "{stop}"
            );
            let log = fs::read_to_string(&f.calls).expect("calls");
            let line =
                |p: &str| -> Vec<&str> { log.lines().filter(|l| l.starts_with(p)).collect() };
            // Three rounds trained, audited and rough-tested; targeted harvests after rounds 1
            // and 2 only (round 3's is saved by the stop).
            assert_eq!(line("python -m fortyone_nn.audit").len(), 3, "{log}");
            assert_eq!(line("search hybrid-check").len(), 3, "{log}");
            assert_eq!(
                line("search labels --trick 2 --count 1000").len(),
                2,
                "{log}"
            );
            let rung = read_json(&root.join("out/hybrid/rung.json")).expect("rung.json");
            assert_eq!(rung["audit"]["best_round"], 2);
            assert_eq!(rung["net"]["dir"], "audit/round-2/net");
            assert_eq!(rung["check"]["best_round"], 2);
            assert_eq!(rung["check"]["verdict"], "FAIL");
            assert!(rung["check"]["rough_test"]["sets"].is_array());
            assert_eq!(r["audit"]["flat_at_pct"], Value::Null);
            let summary = fs::read_to_string(root.join("out/hybrid/audit/audit-summary.txt"))
                .expect("summary");
            for want in [
                "still limiting at round 3: g-one: weight 5.0000%, SD 3.00% on the audit shard",
                "best: round 2 (not flat; overall SD 0.900% on its own fresh audit shard)",
                "final network: audit/round-2/net (sha256",
            ] {
                assert!(summary.contains(want), "{want:?} in\n{summary}");
            }
            // Not shared (no upload_shards): every audit shard stays on the disk.
            for r in 1..=3 {
                let shard = root.join(format!("hybrid-work/audit/t2-audit-r{r}-s3010{r}.npy"));
                assert!(shard.is_file(), "{} deleted", shard.display());
            }
            let r1 = read_json(&root.join("out/hybrid/audit/round-1.json")).expect("round 1");
            assert_eq!(
                r1["audit_shard"]["local_file"],
                "kept on the disk (not shared)"
            );
            assert!(
                r["warnings"].as_array().is_some_and(|w| w.iter().any(|x| x
                    .as_str()
                    .is_some_and(|s| s.contains("kept on this machine's disk")))),
                "{}",
                r["warnings"]
            );
            let _ = fs::remove_dir_all(&f.base);
        }
    }

    /// The capacity ladder: round 2's targeted data lowers the worst group's SD by only 3% (less
    /// than `audit_min_gain`), so round 3 retrains on everything with the next shape; with a
    /// ladder of one shape the same round ends the loop ("capacity ladder exhausted").
    #[cfg(unix)]
    #[test]
    fn audit_loop_climbs_the_capacity_ladder() {
        let group = |sd: f64| {
            format!(
                r#"[{{"name": "g-one", "conditions": [{{"feature": 801, "op": "==", "value": 1}}], "weight": 0.05, "bias": [0.004, 0.0, 0.0, -0.004], "sd": {sd}}}]"#
            )
        };
        for (tag, shapes, stop) in [
            ("auditshape", "16,8;32,16;64,32", "round limit"),
            ("auditshape1", "<widths>", "diminishing returns"),
        ] {
            let f = fakes(tag);
            for (n, sd) in [(1, 0.03), (2, 0.029), (3, 0.02)] {
                fs::write(f.base.join(format!("audit-{n}.json")), group(sd)).expect("write");
            }
            fs::write(f.base.join("mini-fail-all"), "").expect("write");
            let tools = audit_tools(&f, false, None);
            let params = audit_params(json!({ "audit_rounds": 3, "audit_shapes": shapes }));
            let root = f.base.join("run");
            let mut ctx = Ctx::local("hybrid-test", &root, 4);
            let r = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("runs");
            let why = r["audit"]["stop"].as_str().expect("stop");
            assert!(why.starts_with(stop), "{tag}: {why}");
            let log = fs::read_to_string(&f.calls).expect("calls");
            let trains: Vec<&str> = log
                .lines()
                .filter(|l| l.starts_with("python -m fortyone_nn.train"))
                .collect();
            let out = root.join("out/hybrid/audit");
            let r2 = read_json(&out.join("round-2.json")).expect("round 2");
            let c = &r2["capacity"]["compared"];
            assert_eq!(
                (c["sd_before"].as_f64(), c["sd_now"].as_f64()),
                (Some(0.03), Some(0.029))
            );
            assert_eq!(c["no_gain"], true);
            if stop == "round limit" {
                assert_eq!(trains.len(), 3, "{log}");
                assert!(
                    trains[0].contains("--widths 16,8 ") && trains[1].contains("--widths 16,8 ")
                );
                assert!(trains[2].contains("--widths 32,16 "), "{}", trains[2]);
                assert_eq!(r2["capacity"]["next_shape"], "32,16");
                let r3 = read_json(&out.join("round-3.json")).expect("round 3");
                assert_eq!(
                    (r3["shape"].as_str(), r3["shape_step"].as_u64()),
                    (Some("32,16"), Some(2))
                );
                // Round 3 compares with round 2: 0.029 -> 0.02 is a gain of 31%.
                assert_eq!(r3["capacity"]["compared"]["no_gain"], false);
                let t2 = fs::read_to_string(out.join("round-2.txt")).expect("txt");
                assert!(
                    t2.contains("the next round trains the larger shape 32,16"),
                    "{t2}"
                );
                let hb = read_json(&root.join("heartbeat.json")).expect("heartbeat");
                assert_eq!(
                    (
                        hb["progress"]["shape"].as_str(),
                        hb["progress"]["shape_step"].as_str()
                    ),
                    (Some("32,16"), Some("2 of 3"))
                );
                let summary = fs::read_to_string(out.join("audit-summary.txt")).expect("summary");
                assert!(summary.contains("shape 32,16"), "{summary}");
            } else {
                // One shape only: the SD's small fall changes nothing, and rounds 2 and 3 (both
                // on the largest shape) lower neither decision-level bound: diminishing returns.
                assert_eq!(trains.len(), 3, "{log}");
                assert!(out.join("round-3.json").exists());
                assert!(
                    why.contains("rounds 2 and 3 on the largest shape, 16,8"),
                    "{why}"
                );
                let summary = fs::read_to_string(out.join("audit-summary.txt")).expect("summary");
                assert!(
                    summary.contains("still limiting at round 3: g-one: weight 5.0000%, SD 2.00%"),
                    "{summary}"
                );
            }
            let _ = fs::remove_dir_all(&f.base);
        }
    }

    /// The push to a tighter standard (owner, 2026-10-16 night): round 1 is flat at 0.01% (its
    /// group's mini set at 0.007%), so the same round is judged again by 0.005%, where the group
    /// fails and is targeted; round 2's network brings it to 0.004%: flat at 0.005%. The
    /// auditor gets `audit_gamma`; every check after the push runs with `--standard 0.005`.
    #[cfg(unix)]
    #[test]
    fn audit_loop_pushes_to_the_tighter_standard() {
        let f = fakes("auditpush");
        fs::write(f.base.join("audit-1.json"), GROUP_ONE).expect("write");
        fs::write(f.base.join("mini-bounds"), "0.00007 0.00004 0.00004").expect("write");
        let tools = audit_tools(&f, false, None);
        let params = audit_params(json!({ "audit_rounds": 3, "audit_push_standard": 0.005,
                                           "audit_gamma": 0.0001 }));
        let root = f.base.join("run");
        let mut ctx = Ctx::local("hybrid-test", &root, 4);
        let r = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("the loop runs");
        assert_eq!(r["verdict"], "PASS", "{r}");
        let stop = r["audit"]["stop"].as_str().expect("stop");
        assert!(
            stop.starts_with(
                "flat at 0.005% after round 2 (flat at 0.01% after round 1, then pushed) (1 \
                 groups confirmed by the auditor so far, every one within 0.005%"
            ),
            "{stop}"
        );
        let log = fs::read_to_string(&f.calls).expect("calls");
        let line = |p: &str| -> Vec<&str> { log.lines().filter(|l| l.starts_with(p)).collect() };
        assert!(
            line("python -m fortyone_nn.audit")
                .iter()
                .all(|l| l.contains("--gamma 0.0001 ")),
            "{log}"
        );
        let checks = line("search hybrid-check");
        // Round 1 by 0.01% (golden, mini), again by 0.005% (golden, mini), round 2 by 0.005%.
        assert_eq!(checks.len(), 6, "{log}");
        assert!(checks[..2].iter().all(|c| c.contains("--standard 0.01 ")));
        assert!(checks[2..].iter().all(|c| c.contains("--standard 0.005 ")));
        let out = root.join("out/hybrid/audit");
        for x in [
            "round-1.json",
            "round-1-push.json",
            "round-1-push.txt",
            "round-1/hc-t3-o4-push.json",
            "round-2.json",
        ] {
            assert!(out.join(x).is_file(), "{x}");
        }
        let r1p = read_json(&out.join("round-1-push.json")).expect("push");
        assert_eq!(r1p["standard_pct"], 0.005);
        assert_eq!(r1p["decision_level"][0]["pass"], false);
        assert_eq!(r1p["targeted"]["why"], "1 groups confirmed this round");
        let summary = fs::read_to_string(out.join("audit-summary.txt")).expect("summary");
        assert!(
            summary.contains("\nflat at 0.01% after round 1; pushing to 0.005%\n"),
            "{summary}"
        );
        assert!(
            summary.contains("round 1 (judged again by the push standard): "),
            "{summary}"
        );
        assert!(
            summary.contains("best: round 2 (flat at 0.005%"),
            "{summary}"
        );
        let rung = read_json(&root.join("out/hybrid/rung.json")).expect("rung.json");
        assert_eq!(rung["net"]["dir"], "audit/round-2/net");
        assert_eq!(rung["audit"]["push"]["flat_at_push"], true);
        assert_eq!(rung["audit"]["best_round"], 2);
        let hb = read_json(&root.join("heartbeat.json")).expect("heartbeat");
        assert_eq!(
            hb["progress"]["standard"],
            "0.005% (pushing: flat at 0.01% after round 1)"
        );
        let _ = fs::remove_dir_all(&f.base);
    }

    /// A mini golden set with fewer than `audit_mini_min` decisions (1 kept of 2 when the
    /// candidate cap is reached) gives no verdict: its group is reported "unmeasurable at the
    /// decision level" in every round and the summary, neither failing nor blocking "flat" (the
    /// overall row covers it, as for a set that found no decision).
    #[cfg(unix)]
    #[test]
    fn audit_loop_unmeasurable_group_is_reported_not_judged() {
        let f = fakes("auditunfilled");
        fs::write(f.base.join("audit-1.json"), GROUP_ONE).expect("write");
        fs::write(f.base.join("mini-unfilled"), "").expect("write");
        let tools = audit_tools(&f, false, None);
        let params = audit_params(json!({ "audit_rounds": 2, "audit_mini_candidates": 30 }));
        let root = f.base.join("run");
        let mut ctx = Ctx::local("hybrid-test", &root, 4);
        let r = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("the loop runs");
        assert_eq!(r["verdict"], "PASS", "{r}");
        let stop = r["audit"]["stop"].as_str().expect("stop");
        assert!(stop.starts_with("flat after round 1"), "{stop}");
        let log = fs::read_to_string(&f.calls).expect("calls");
        assert!(log.contains("--max-candidates 30 "), "{log}");
        assert!(!log.contains("--harvest-keep"), "nothing targeted: {log}");
        let out = root.join("out/hybrid/audit");
        let x = read_json(&out.join("round-1.json")).expect("round 1");
        assert_eq!(
            x["decision_level"][0]["basis"],
            "unmeasurable at the decision level"
        );
        assert_eq!(x["decision_level"][0]["pass"], Value::Null);
        assert_eq!(x["flat"], true);
        assert_eq!(x["groups_failing"], json!([]));
        assert_eq!(x["groups_unmeasurable"], json!(["g-one"]));
        let m = &x["mini_sets"][0];
        assert_eq!(
            (m["unfilled"].as_bool(), m["pass"].is_null()),
            (Some(true), true)
        );
        assert!(
            m["fill"].as_str().is_some_and(|t| t.starts_with(
                "1 decisions kept of 2 wanted after drawing 30 candidates (the cap: 30"
            )),
            "{m}"
        );
        let t = fs::read_to_string(out.join("round-1.txt")).expect("txt");
        assert!(
            t.contains("unmeasurable at the decision level: 1 decisions kept of 2 wanted"),
            "{t}"
        );
        assert!(
            t.contains("Unmeasurable at the decision level (mini golden sets of fewer than 2 decisions; reported, not judged): 1 groups (g-one)."),
            "{t}"
        );
        let summary = fs::read_to_string(out.join("audit-summary.txt")).expect("summary");
        assert!(summary.contains("; unmeasurable: 1 groups\n"), "{summary}");
        assert!(
            summary.contains("\nunmeasurable: 1 groups (mini golden sets of fewer than 2 decisions; reported, not judged): g-one\n"),
            "{summary}"
        );
        let _ = fs::remove_dir_all(&f.base);
    }

    /// Diminishing returns never stop the loop while the shape can still step up: round 2's SD
    /// does not fall, so round 3 trains the larger (top) shape; rounds 3 and 4, both on it,
    /// lower neither bound: the loop stops after round 4, before its round limit, naming the
    /// groups still limiting it.
    #[cfg(unix)]
    #[test]
    fn audit_loop_stops_on_diminishing_returns() {
        let f = fakes("auditdim");
        for n in 1..=5 {
            fs::write(f.base.join(format!("audit-{n}.json")), GROUP_ONE).expect("write");
        }
        fs::write(f.base.join("mini-fail-all"), "").expect("write");
        let tools = audit_tools(&f, false, None);
        let params = audit_params(json!({ "audit_rounds": 5, "audit_shapes": "16,8;32,16" }));
        let root = f.base.join("run");
        let mut ctx = Ctx::local("hybrid-test", &root, 4);
        let r = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("the loop runs");
        let stop = r["audit"]["stop"].as_str().expect("stop");
        assert!(
            stop.starts_with("diminishing returns (rounds 3 and 4 on the largest shape, 32,16"),
            "{stop}"
        );
        let log = fs::read_to_string(&f.calls).expect("calls");
        let trains: Vec<&str> = log
            .lines()
            .filter(|l| l.starts_with("python -m fortyone_nn.train"))
            .collect();
        assert_eq!(trains.len(), 4, "{log}");
        assert!(trains[2].contains("--widths 32,16 ") && trains[1].contains("--widths 16,8 "));
        let summary =
            fs::read_to_string(root.join("out/hybrid/audit/audit-summary.txt")).expect("summary");
        assert!(
            summary.contains("still limiting at round 4: g-one: weight 5.0000%, SD 3.00%"),
            "{summary}"
        );
        let _ = fs::remove_dir_all(&f.base);
    }

    /// The chunk keeps the best round's network and check, not the last: by the golden set's
    /// overall bound when no round is flat (0.006%, 0.004%, 0.005%: round 2).
    #[cfg(unix)]
    #[test]
    fn audit_loop_keeps_the_best_round() {
        let f = fakes("auditbest");
        for n in 1..=3 {
            fs::write(f.base.join(format!("audit-{n}.json")), GROUP_ONE).expect("write");
        }
        fs::write(f.base.join("mini-fail-all"), "").expect("write");
        fs::write(f.base.join("overall-bounds"), "0.00006 0.00004 0.00005").expect("write");
        let tools = audit_tools(&f, false, None);
        let params = audit_params(json!({ "audit_rounds": 3 }));
        let root = f.base.join("run");
        let mut ctx = Ctx::local("hybrid-test", &root, 4);
        let r = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect("the loop runs");
        assert_eq!(r["verdict"], "FAIL");
        let rung = read_json(&root.join("out/hybrid/rung.json")).expect("rung.json");
        assert_eq!(rung["net"]["dir"], "audit/round-2/net");
        assert_eq!(rung["audit"]["best_round"], 2);
        assert_eq!(rung["check"]["best_round"], 2);
        let summary =
            fs::read_to_string(root.join("out/hybrid/audit/audit-summary.txt")).expect("summary");
        assert!(
            summary.contains("best: round 2 (not flat; golden 95% upper 0.0040%)"),
            "{summary}"
        );
        assert!(
            summary.contains("final network: audit/round-2/net (sha256"),
            "{summary}"
        );
        let _ = fs::remove_dir_all(&f.base);
    }

    /// A stop the launcher asks for over the dollar cap is reported as one.
    #[test]
    fn audit_stop_reason_names_the_dollar_cap() {
        for why in [
            "the launcher asked the run to stop (runs/x/STOP) dollar cap reached",
            "the launcher asked the run to stop (runs/x/STOP) spent $1.71 of $1.70",
            "the launcher asked the run to stop (runs/x/STOP) over budget",
        ] {
            assert!(audit::stop_reason(why).starts_with("dollar cap ("), "{why}");
        }
        assert!(audit::stop_reason("STOP file /x/STOP by hand").starts_with("stopped ("));
    }

    /// A stop file that appears during round 2's audit: the step that follows is not started,
    /// the chunk fails with the stop, and the summary (round 1's line, then the stop reason) and
    /// `rung.json` are written first. Run in a child process: a stop request is process-wide.
    #[cfg(unix)]
    #[test]
    fn audit_loop_stops_on_the_stop_file() {
        if std::env::var_os("HYBRID_AUDIT_STOP_CHILD").is_none() {
            let out = Command::new(std::env::current_exe().expect("exe"))
                .args([
                    "--exact",
                    "hybrid::tests::audit_loop_stops_on_the_stop_file",
                    "--nocapture",
                ])
                .env("HYBRID_AUDIT_STOP_CHILD", "1")
                .output()
                .expect("runs");
            assert!(
                out.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            assert!(String::from_utf8_lossy(&out.stdout).contains("1 passed"));
            return;
        }
        let f = fakes("auditstop");
        for n in 1..=3 {
            fs::write(f.base.join(format!("audit-{n}.json")), GROUP_ONE).expect("write");
        }
        fs::write(f.base.join("mini-fail-all"), "").expect("write");
        fs::write(f.base.join("stop-at-2"), "").expect("write");
        let root = f.base.join("run");
        let tools = audit_tools(&f, false, Some(&root.join("STOP")));
        let params = audit_params(json!({ "audit_rounds": 3 }));
        let mut ctx = Ctx::local("hybrid-test", &root, 4);
        let e = chunk_steps(&mut ctx, &params, &tools, &f.table, json!({})).expect_err("stopped");
        assert!(e.contains("stopped"), "{e}");
        let log = fs::read_to_string(&f.calls).expect("calls");
        assert_eq!(
            log.lines()
                .filter(|l| l.starts_with("python -m fortyone_nn.audit"))
                .count(),
            2
        );
        assert!(
            !log.contains("--seed 30152"),
            "round 2's targeted harvest never started:\n{log}"
        );
        let out = root.join("out").join("hybrid");
        let summary = fs::read_to_string(out.join("audit/audit-summary.txt")).expect("summary");
        let lines: Vec<&str> = summary.lines().collect();
        assert!(
            lines[0].starts_with("round 1: 1 groups confirmed"),
            "{summary}"
        );
        // A stop keeps the best round judged so far (round 1; round 2 was trained and audited
        // but never judged).
        assert!(lines[1].starts_with("best: round 1 (not flat"), "{summary}");
        assert!(
            lines[2].starts_with("stop: stopped (STOP file"),
            "{summary}"
        );
        assert!(
            lines[3].starts_with("final network: net (sha256"),
            "{summary}"
        );
        let rung = read_json(&out.join("rung.json")).expect("rung.json");
        assert!(
            rung["audit"]["stop"]
                .as_str()
                .is_some_and(|s| s.starts_with("stopped"))
        );
        assert_eq!(rung["audit"]["rounds"].as_array().map(Vec::len), Some(1));
        let _ = fs::remove_dir_all(&f.base);
    }

    /// Without the audit settings (or with `audit_rounds` 0) a chunk is exactly as before: the
    /// same calls in the same order and the same steps.
    #[cfg(unix)]
    #[test]
    fn no_audit_settings_is_the_former_chunk() {
        let f = fakes("auditnone");
        let tools = Tools {
            search: f.search.clone(),
            uv: f.uv.clone(),
            python_dir: f.py.clone(),
            nvidia_smi: f.base.join("no-such-nvidia-smi"),
            remote: None,
        };
        let base = json!({ "labels_random": 7, "labels_rules": 5, "val_count": 3,
                           "golden_count": 2, "golden_worlds": 4, "widths": "16,8", "epochs": 2 });
        let mut zero = base.clone();
        zero["audit_rounds"] = json!(0);
        let mut runs = Vec::new();
        for (k, v) in [("absent", base), ("zero", zero)] {
            fs::write(&f.calls, "").expect("truncate");
            let root = f.base.join(k);
            let mut ctx = Ctx::local("hybrid-test", &root, 4);
            let p = parse(&v).expect("params");
            assert_eq!(p.audit, None);
            let r = chunk_steps(&mut ctx, &p, &tools, &f.table, json!({})).expect("runs");
            assert!(r.get("audit").is_none());
            let rung = read_json(&root.join("out/hybrid/rung.json")).expect("rung.json");
            assert!(rung.get("audit").is_none());
            let calls = fs::read_to_string(&f.calls)
                .expect("calls")
                .replace(&root.display().to_string(), "ROOT");
            runs.push((steps_of(&rung), calls));
        }
        assert_eq!(runs[0], runs[1]);
        assert_eq!(
            runs[0].0,
            [
                "python environment",
                "GPU check",
                "labels random (1 of 3)",
                "labels rules (2 of 3)",
                "labels val (3 of 3)",
                "golden decisions",
                "golden worlds",
                "share golden set",
                "train",
                "hybrid check"
            ]
        );
        assert!(!runs[0].1.contains("fortyone_nn.audit") && !runs[0].1.contains("--slices"));
        let _ = fs::remove_dir_all(&f.base);
    }
}
