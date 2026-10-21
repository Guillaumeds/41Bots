//! Format-2 runs (docs/M3-SPEC.md section 7): several streams of chunks on one machine, with
//! priorities, a deadline, a versioned done-list and resume.
//!
//! The orchestrator (`runner run` on a format-2 manifest) starts each chunk as a worker child
//! process (`runner chunk <spec.json>`), so a chunk can be killed or lost without harming the
//! others. When a worker finishes, the orchestrator bundles its `out/` directory into a
//! deterministic `.tar.zst`, uploads it to `runs/<run>/out/<stream>/<index:06>-<sha16>.tar.zst`
//! (immutable), reads it back to verify the SHA-256, then appends a done-list entry. Every
//! `upload_every_s` the whole done-list is written as a new key `done/<run>/<seq:08>-<utc>.json`.
//! A chunk's logs (timings, so never in the bundle) go to `runs/<run>/logs/<stream>/<index:06>.tar.zst`.
//!
//! Resume: on start the highest done-list is read; listed chunks whose remote bundles still match
//! are skipped, everything else is redone from its seed (local leftovers are deleted first). A
//! chunk's bundle depends only on its seed and parameters, so a chunk redone from scratch
//! reproduces the same bytes and the same key. A `search_measure` chunk that continues from
//! its live files (below) is the exception: its worlds file is byte-identical, but its timing
//! file and the `resumed` counts in its result differ, so its bundle and key do too.
//!
//! Live files: every `upload_every_s` the running chunks' changed `out/` files go, in the
//! background, to `runs/<run>/partial/<stream>/<index:06>/out/` (mutable keys, checkpoints
//! first; `net.rs`); a stopped chunk uploads them once more before its partial bundle. A
//! `search_measure` chunk started on a machine that resumes the run first downloads them back, so
//! it continues world by world from its worlds file and checkpoint (`search::measure`): only the
//! worlds being solved, or finished after the last upload, are redone, and the worlds file is the
//! same bytes. Other kinds are redone from their seed.
//!
//! Network: the orchestrator writes its local `heartbeat.json` every `heartbeat_every_s` whatever
//! the network does; heartbeat uploads and STOP checks run in the background, and every other
//! remote call runs beside the heartbeat (`Run::net_call`), each with hard limits (`remote.rs`).
//! Heartbeats carry `upload_failures`, `other_net_failures` and `last_upload_ok_s`, and the
//! running chunks' progress estimates (`progress.chunk_eta`; with one chunk running also
//! `progress.eta`, priced over the whole run) and, for kinds that name their steps, each running
//! chunk's step and counts (`progress.chunk_steps`, e.g. a `hybrid_rung`'s label rows or epoch).
//!
//! Warnings a running chunk puts in its own heartbeat (`progress.warnings`, e.g. a transposition
//! table size lowered to fit memory) appear in the run's heartbeat as `<stream>/<index>: …`
//! while it runs, on stderr once, and in `final-manifest.json` (`warnings`).
//!
//! End: every listed output is checked in the remote, `merged.json` (the deterministic merged
//! report) is uploaded, then `final-manifest.json` (git commit, image, build, seeds, threads,
//! versions), then `DONE` (or `FAILED`), then the instance destroys itself (M1 behaviour).
//!
//! Chunk kinds and what a worker runs (`<seed>` is the chunk seed, `out/` the bundled directory):
//! - `teacher`: `search teacher --hands <params.hands_per_chunk> --seed <seed> --config
//!   <params.config> --equity <input> [--start-hands H] [params.args…] --out out/`;
//! - `arena_blocks`: in-process, `params.a` vs `params.b`, block `j` of the chunk seeded
//!   `arena::block_seed(<seed>, 0, j)`, writes `out/blocks.json`; the scores feed the stream's
//!   sequential test (`params.sprt`);
//! - `suitability`, `agree`, `belief_bench`: `search <kind with - for _> --seed <seed>
//!   [--hands <params.hands_per_chunk>] [--equity <input>] [params.args…] --out out/`;
//! - `difftest`, `equity_sweep`: as in format 1 with `seed` = the chunk seed;
//! - `benchmark`, `solver_spike`, `search_measure`, `t1bench`, `hybrid_rung`: as in format 1,
//!   params unchanged, except that `search_measure`'s `params.ids_input` names an input whose
//!   path becomes `params.ids`; `hybrid_rung` (`hybrid.rs`) trains in Python and keeps its label
//!   shards outside `out/` (uploaded content-addressed when `params.upload_shards`, each as soon as it is made);
//! - `hybrid_t1check` (`t1check.rs`, format 2 only): its `net_input`, `decisions_input` and
//!   `saved[].input` name inputs whose paths are added to its params (`net`, `decisions`,
//!   `saved[].path`);
//! - `hybrid_checks` (`checks.rs`, format 2 only): its `decisions_input`, `worlds_input` and
//!   every check's `nets` name inputs whose paths are added to its params (`decisions`,
//!   `worlds_path`, `net_paths`);
//! - `search_measure`'s silver reference (`measure.rs`): `reference.net_input`,
//!   `check.net_input` and `worlds_input` name inputs whose paths are added to its params
//!   (`reference.net`, `check.net`, `worlds`);
//! - `student_teacher` (`student.rs`, format 2 only): part `first_part + index` of a stream of
//!   the student's teacher records; its `net_input` names an input whose path is added (`net`);
//! - `student_train` (`student_train.rs`, format 2 only, one chunk): the student's learning curve
//!   from stored teacher shards; like `search_measure` its live files are restored on the machine
//!   that continues it, so its finished (size, width) pairs are not trained again.
//!
//! The equity input is `inputs.equity` (or the input named by `params.equity_input`).
//!
//! Remote: `RUNNER_REMOTE_DIR` (a local directory, used by tests and the laptop), else R2 through
//! rclone when `RCLONE_CONFIG_R2_TYPE` and `R2_BUCKET` are set, else `<work>/remote`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::bundle::bundle_dir;
use crate::net::{self, Net};
use crate::remote::{DirRemote, Remote, UPLOAD_ENTRY, kill_point, retry, retry_patient};
use crate::sched::{
    Chunks, Manifest2, StreamState, chunk_seed, finished_median, latest_done, parse_v2, pick,
    size_note, sprt_prefix,
};
use crate::stop;
use crate::{
    CpuSampler, Ctx, default_threads, hardware, run_kind, self_destroy, sibling, unix_now,
    utc_string,
};

// ---------------------------------------------------------------- worker side

static ORPHANED: AtomicBool = AtomicBool::new(false);

/// True once the orchestrator that started this worker has gone (its stdin pipe closed).
pub fn orphaned() -> bool {
    ORPHANED.load(Ordering::Relaxed)
}

/// Watches stdin: the orchestrator holds the other end, so end-of-file means it died (or wants
/// this chunk stopped). The worker then stops its own child and exits, so a killed runner never
/// leaves chunk processes writing into a directory a restarted runner may reuse.
fn watch_parent() {
    std::thread::spawn(|| {
        let mut sink = [0u8; 64];
        let mut stdin = std::io::stdin();
        while matches!(stdin.read(&mut sink), Ok(n) if n > 0) {}
        ORPHANED.store(true, Ordering::Relaxed);
        std::thread::sleep(Duration::from_secs(2));
        std::process::exit(3);
    });
}

/// `runner chunk <spec.json>`: run one format-2 chunk into `spec.dir/out`.
pub fn worker(spec_path: &Path) -> ExitCode {
    watch_parent();
    match worker_inner(spec_path) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("runner chunk: {e}");
            ExitCode::FAILURE
        }
    }
}

fn s<'a>(v: &'a Value, k: &str) -> Result<&'a str, String> {
    v.get(k)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("chunk spec needs `{k}`"))
}

fn worker_inner(spec_path: &Path) -> Result<(), String> {
    let spec: Value = serde_json::from_str(
        &fs::read_to_string(spec_path).map_err(|e| format!("{}: {e}", spec_path.display()))?,
    )
    .map_err(|e| e.to_string())?;
    let dir = PathBuf::from(s(&spec, "dir")?);
    let kind = s(&spec, "kind")?.to_string();
    let params = spec.get("params").cloned().unwrap_or_else(|| json!({}));
    let seed = spec.get("seed").and_then(Value::as_u64).ok_or("no seed")?;
    let index = spec
        .get("index")
        .and_then(Value::as_u64)
        .ok_or("no index")?;
    let threads = spec.get("threads").and_then(Value::as_u64).unwrap_or(1) as usize;
    let inputs: BTreeMap<String, String> = spec
        .get("inputs")
        .and_then(Value::as_object)
        .map(|o| {
            o.iter()
                .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string())))
                .collect()
        })
        .unwrap_or_default();
    let out = dir.join("out");
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut ctx = Ctx::local(s(&spec, "stream")?, &dir, threads);
    let equity = || {
        let name = params
            .get("equity_input")
            .and_then(Value::as_str)
            .unwrap_or("equity");
        inputs
            .get(name)
            .cloned()
            .ok_or_else(|| format!("kind {kind} needs input `{name}` in the manifest"))
    };
    let result = match kind.as_str() {
        "teacher" => {
            let hands = params
                .get("hands_per_chunk")
                .and_then(Value::as_u64)
                .ok_or("teacher needs params.hands_per_chunk")?;
            let mut c = Command::new(sibling("search"));
            c.args(["teacher", "--hands", &hands.to_string()])
                .args(["--seed", &seed.to_string()])
                .args([
                    "--config",
                    params
                        .get("config")
                        .and_then(Value::as_str)
                        .unwrap_or("pimc-teacher"),
                ])
                .arg("--equity")
                .arg(equity()?)
                .arg("--out")
                .arg(&out);
            if let Some(h) = params.get("start_hands").and_then(Value::as_u64) {
                c.args(["--start-hands", &h.to_string()]);
            }
            extra_args(&mut c, &params)?;
            ctx.run_child(c, "search.log")?;
            json!({ "kind": kind, "seed": seed, "hands": hands })
        }
        "arena_blocks" => arena_blocks(&params, seed, index, &spec, &out, equity().ok())?,
        "student_teacher" => {
            // The network's path joins the settings; the chunk's index is its part of the stream
            // (the generator's own seed is in the settings: a part is the same shard on any run).
            let p = chunk_params(&kind, &params, seed, &inputs)?;
            crate::student::chunk(&mut ctx, &p, index)?
        }
        "student_train" => {
            // A prebuilt equity table among the inputs replaces building it (hash-checked).
            let mut p = params.clone();
            if let Ok(path) = equity()
                && let Some(o) = p.as_object_mut()
            {
                let t = o.entry("table").or_insert_with(|| json!({}));
                if let Some(t) = t.as_object_mut() {
                    t.entry("path").or_insert(json!(path));
                }
            }
            crate::student_train::chunk(&mut ctx, &p)?
        }
        "suitability" | "agree" | "belief_bench" => {
            // The `search` subcommand of the same name (`belief_bench` → `belief-bench`), with
            // `--seed`, `--out`, the equity table when given, and `params.args` verbatim.
            let sub = kind.replace('_', "-");
            let mut c = Command::new(sibling("search"));
            c.arg(&sub).args(["--seed", &seed.to_string()]);
            if let Some(h) = params.get("hands_per_chunk").and_then(Value::as_u64) {
                c.args(["--hands", &h.to_string()]);
            }
            if let Ok(e) = equity() {
                c.arg("--equity").arg(e);
            }
            extra_args(&mut c, &params)?;
            c.arg("--out").arg(&out);
            ctx.run_child(c, "search.log")?;
            json!({ "kind": kind, "seed": seed })
        }
        _ => {
            // Existing kinds: their settings with the inputs' paths ([`chunk_params`]).
            let p = chunk_params(&kind, &params, seed, &inputs)?;
            let mut r = run_kind(&mut ctx, &kind, &p)?;
            if let Some(o) = r.as_object_mut() {
                o.insert("kind".into(), json!(kind));
            }
            r
        }
    };
    let text = serde_json::to_string_pretty(&result).map_err(|e| e.to_string())? + "\n";
    fs::write(out.join("result.json"), text).map_err(|e| e.to_string())
}

/// The settings a chunk of kind `kind` runs with: the manifest's (as the scheduler passed them)
/// with the chunk seed where it is the chunk's own randomness and the local paths of the inputs
/// they name (`inputs`: name → path) added, as each kind's own `resolve` does. The chunk then
/// reads these settings itself, so every key added here must be one it accepts (tested end to
/// end on the committed manifests).
pub(crate) fn chunk_params(
    kind: &str,
    params: &Value,
    seed: u64,
    inputs: &BTreeMap<String, String>,
) -> Result<Value, String> {
    let equity = || {
        let name = params
            .get("equity_input")
            .and_then(Value::as_str)
            .unwrap_or("equity");
        inputs
            .get(name)
            .cloned()
            .ok_or_else(|| format!("kind {kind} needs input `{name}` in the manifest"))
    };
    let with_table = |p: &mut Value| {
        if let Ok(path) = equity()
            && let Some(o) = p.as_object_mut()
        {
            let t = o.entry("table").or_insert_with(|| json!({}));
            if let Some(t) = t.as_object_mut() {
                t.entry("path").or_insert(json!(path));
            }
        }
    };
    if kind == "student_teacher" {
        // The network's path joins the settings; the chunk's index is its part of the stream
        // (the generator's own seed is in the settings: a part is the same shard on any run).
        let mut p = crate::student::resolve(params, inputs)?;
        with_table(&mut p);
        return Ok(p);
    }
    // Existing kinds. Where `seed` is the chunk's own randomness the chunk seed replaces
    // it; the whole-machine measurements keep their fixed measurement seeds.
    let mut p = params.clone();
    if SEEDED_KINDS.contains(&kind)
        && let Some(o) = p.as_object_mut()
    {
        o.insert("seed".into(), json!(seed));
    }
    // `search_measure` may take its decision-id list as a manifest input.
    if kind == "search_measure"
        && let Some(name) = params.get("ids_input").and_then(Value::as_str)
        && let Some(o) = p.as_object_mut()
    {
        let path = inputs
            .get(name)
            .ok_or_else(|| format!("search_measure needs input `{name}` (ids_input)"))?;
        o.insert("ids".into(), json!(path));
    }
    // `hybrid_t1check` names its files among the inputs: their paths join its params.
    if kind == "hybrid_t1check" {
        p = crate::t1check::resolve(&p, inputs)?;
    }
    // `hybrid_checks` too (`checks.rs`), and `leafrisk`'s network (`leafrisk.rs`).
    if kind == "hybrid_checks" {
        p = crate::checks::resolve(&p, inputs)?;
    }
    if kind == "leafrisk" {
        p = crate::leafrisk::resolve(&p, inputs)?;
    }
    if kind == "specialist" {
        p = crate::specialist::resolve(&p, inputs)?;
    }
    // A `search_measure` chunk's silver reference, check network and stored worlds
    // (`measure.rs`, "A silver reference") are inputs too.
    if kind == "search_measure" {
        p = crate::measure::resolve(&p, inputs)?;
    }
    // A prebuilt equity table among the inputs (`params.equity_input`, default
    // `equity`) replaces building it; its hash is still checked against the record.
    if [
        "search_measure",
        "t1bench",
        "hybrid_rung",
        "hybrid_t1check",
        "hybrid_checks",
        "leafrisk",
        "specialist",
    ]
    .contains(&kind)
    {
        with_table(&mut p);
    }
    Ok(p)
}

/// Existing kinds whose `seed` parameter is the chunk's own randomness, so the chunk seed
/// `Rng::derive(run_seed, &[k, i])` replaces it. The others (`benchmark`, `solver_spike`,
/// `search_measure`) are whole-machine measurements whose `seed` is a fixed constant of the
/// measurement (the equity table's, the decision set's); their params pass through unchanged.
const SEEDED_KINDS: [&str; 2] = ["difftest", "equity_sweep"];

/// Appends `params.args` (a list of strings) to a `search` command line verbatim.
fn extra_args(c: &mut Command, params: &Value) -> Result<(), String> {
    for a in params
        .get("args")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        c.arg(a.as_str().ok_or("params.args must be strings")?);
    }
    Ok(())
}

/// `arena_blocks`: plays this chunk's blocks in order (block `j` of chunk `i` uses
/// `block_seed(chunk_seed, 0, j)`) and writes `blocks.json` with each block's score.
fn arena_blocks(
    params: &Value,
    seed: u64,
    index: u64,
    spec: &Value,
    out: &Path,
    equity: Option<String>,
) -> Result<Value, String> {
    if let Some(e) = &equity {
        search::bot::set_equity_path(e);
    }
    let name = |k: &str| {
        params
            .get(k)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("arena_blocks needs params.{k}"))
    };
    let (an, bn) = (name("a")?, name("b")?);
    let a = search::factory(an).ok_or_else(|| search::bot::factory_error(an))?;
    let b = search::factory(bn).ok_or_else(|| search::bot::factory_error(bn))?;
    let kind = if params
        .get("block14")
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        arena::BlockKind::Block14
    } else {
        arena::BlockKind::Rot4
    };
    let split = params
        .get("split_seeds")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let n = spec.get("blocks").and_then(Value::as_u64).unwrap_or(1);
    let mut rows = Vec::new();
    for j in 0..n {
        if orphaned() {
            std::process::exit(3);
        }
        let bs = arena::block_seed(seed, 0, j);
        let r = arena::play_block(kind, bs, &*a, &*b, engine::game::Rules::HOUSE, split)
            .map_err(|e| format!("block {j}: {e:?}"))?;
        rows.push(json!({
            "block": j, "block_seed": bs, "score": r.score, "stake_score": r.stake_score,
            "mix": r.mix.iter().map(|&(s, m)| json!([s, m])).collect::<Vec<_>>(),
        }));
    }
    let scores: Vec<i64> = rows.iter().filter_map(|r| r["score"].as_i64()).collect();
    // Inputs are stored under their sha256, so the file name records the table (M3-SPEC §8.1).
    let table = equity.as_deref().and_then(|e| {
        Path::new(e)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
    });
    let doc = json!({ "a": an, "b": bn, "kind": format!("{kind:?}"), "chunk": index,
                      "seed": seed, "equity_sha256": table, "blocks": rows });
    fs::write(
        out.join("blocks.json"),
        serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| e.to_string())?;
    Ok(json!({ "kind": "arena_blocks", "seed": seed, "a": an, "b": bn, "scores": scores }))
}

// ---------------------------------------------------------------- orchestrator side

struct Running {
    child: Child,
    stream: usize,
    index: u64,
    seed: u64,
    dir: PathBuf,
    started: Instant,
    /// The memory it was admitted with (`mem::chunk_need`).
    mem: u64,
}

struct Run {
    m: Manifest2,
    /// The remote store (counting its failures) and the background network work (`net.rs`).
    remote: Arc<dyn Remote>,
    net: Net,
    root: PathBuf,
    threads: usize,
    inputs: BTreeMap<String, String>,
    states: Vec<StreamState>,
    /// The done-list, in completion order.
    entries: Vec<Value>,
    done_seq: u64,
    dirty: bool,
    last_done_write: Instant,
    hb_seq: u64,
    last_hb: Option<Instant>,
    cpu: CpuSampler,
    start: Instant,
    dph: f64,
    /// Partial CPU use is expected now (idle tail, or a chunk says so): the launcher's low-CPU
    /// rule skips such heartbeats.
    partial_ok: bool,
    /// Kinds whose chunk-size note has been printed.
    noted: BTreeSet<String>,
    /// Warnings of the chunks running now (`<stream>/<index>: …`), for the heartbeat.
    warnings: Vec<String>,
    /// Standing warnings about delivering results (a chunk bundle that could not be uploaded
    /// after all retries); they stay in every heartbeat from then on.
    delivery_warnings: Vec<String>,
    /// Every warning seen this session, for the final manifest.
    warned: BTreeSet<String>,
    /// Unfinished outputs of chunks stopped mid-way (deadline or stop request), uploaded under
    /// `runs/<run>/partial/` as bundles, for reading.
    partial: Vec<Value>,
    /// The phase of the last heartbeat (repeated by heartbeats during network calls).
    phase: String,
    /// When the running chunks' files were last uploaded to their live prefixes.
    last_partial: Instant,
    /// The progress estimates of the running chunks (`<stream>/<index>`: their `progress.eta`).
    chunk_etas: BTreeMap<String, Value>,
    /// The steps of the running chunks that name one (`<stream>/<index>`: `chunk_step`).
    chunk_steps: BTreeMap<String, Value>,
    /// The container's memory, for the check that chunks running together fit in it.
    limit: crate::mem::Limit,
    /// Chunks whose wait for memory has been logged.
    mem_waits: BTreeSet<(usize, u64)>,
}

fn compact_utc() -> String {
    utc_string(unix_now()).replace(['-', ':'], "")
}

fn choose_remote(work: &Path) -> Box<dyn Remote> {
    crate::remote::from_env().unwrap_or_else(|| {
        Box::new(DirRemote {
            root: work.join("remote"),
        })
    })
}

fn env_nonempty(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

/// `runner run` on a format-2 manifest.
pub fn run(manifest: &Value, work: &Path) -> ExitCode {
    let mut m = match parse_v2(manifest) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("runner: manifest: {e}");
            return ExitCode::from(2);
        }
    };
    // Absolute, so chunk workers and their children get paths that do not depend on a cwd.
    let work = std::path::absolute(work).unwrap_or_else(|_| work.to_path_buf());
    let net = Net::new(choose_remote(&work), &format!("runs/{}", m.run_id));
    let remote = Arc::clone(&net.remote);
    let root = work.join(&m.run_id);
    let threads = m.threads.unwrap_or_else(default_threads);
    m.resolve_threads(threads);
    println!(
        "runner: format-2 run {}, {threads} threads, {} streams, remote {}",
        m.run_id,
        m.streams.len(),
        remote.describe()
    );
    let mut r = Run {
        states: vec![StreamState::default(); m.streams.len()],
        m,
        remote,
        net,
        root,
        threads,
        inputs: BTreeMap::new(),
        entries: Vec::new(),
        done_seq: 0,
        dirty: false,
        last_done_write: Instant::now(),
        hb_seq: 0,
        last_hb: None,
        cpu: CpuSampler::new(),
        start: Instant::now(),
        dph: std::env::var("DPH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0),
        partial_ok: false,
        noted: BTreeSet::new(),
        warnings: Vec::new(),
        delivery_warnings: Vec::new(),
        warned: BTreeSet::new(),
        partial: Vec::new(),
        phase: "start".into(),
        last_partial: Instant::now(),
        chunk_etas: BTreeMap::new(),
        chunk_steps: BTreeMap::new(),
        limit: crate::mem::Sys::host().limit(),
        mem_waits: BTreeSet::new(),
    };
    let outcome = r.go();
    let ok = outcome.is_ok();
    let marker = if ok { "DONE" } else { "FAILED" };
    let body = json!({ "utc": utc_string(unix_now()), "elapsed_s": r.start.elapsed().as_secs(),
                       "error": outcome.as_ref().err(), "stopped": stop::requested(),
                       "partial": r.partial });
    r.heartbeat(true, if ok { "uploading" } else { "failed" });
    // The marker goes last: a DONE means every output, the merged report and the final manifest
    // are already in the remote.
    let uploaded = r.put_json(marker, &format!("{}/{marker}", r.prefix()), &body);
    if let Err(e) = &uploaded {
        eprintln!("runner: upload {marker}: {e}");
    }
    r.net.drain(Duration::from_secs(10));
    match &outcome {
        Ok(()) => println!("runner: {marker}"),
        Err(e) => eprintln!("runner: FAILED: {e}"),
    }
    if std::env::var("RUNNER_NO_DESTROY").is_err() {
        self_destroy();
    }
    if ok && uploaded.is_ok() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

impl Run {
    fn prefix(&self) -> String {
        format!("runs/{}", self.m.run_id)
    }

    fn put_json(&mut self, local_name: &str, key: &str, v: &Value) -> Result<(), String> {
        let p = self.root.join("tmp").join(local_name);
        fs::create_dir_all(p.parent().unwrap_or(&self.root)).map_err(|e| e.to_string())?;
        fs::write(
            &p,
            serde_json::to_string_pretty(v).map_err(|e| e.to_string())? + "\n",
        )
        .map_err(|e| e.to_string())?;
        let key = key.to_string();
        self.net_call(move |r| retry(3, || r.put(&p, &key, false)))
    }

    /// Runs `f` on the remote on a helper thread while this one keeps heartbeating, so no
    /// network call (each with hard limits, `remote.rs`) stops the heartbeat.
    fn net_call<T: Send>(&mut self, f: impl FnOnce(&dyn Remote) -> T + Send) -> T {
        let r = Arc::clone(&self.remote);
        std::thread::scope(|s| {
            let h = s.spawn(move || f(&*r));
            while !h.is_finished() {
                let phase = self.phase.clone();
                self.heartbeat(false, &phase);
                std::thread::sleep(Duration::from_millis(50));
            }
            h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))
        })
    }

    /// The peak memory a chunk of stream `k` will hold (`mem::chunk_need`).
    fn chunk_need(&self, k: usize) -> u64 {
        let s = &self.m.streams[k];
        let name = s
            .params
            .get("equity_input")
            .and_then(Value::as_str)
            .unwrap_or("equity");
        let equity = self
            .inputs
            .get(name)
            .and_then(|p| fs::metadata(p).ok())
            .map_or(128 << 20, |m| m.len());
        crate::mem::chunk_need(&s.kind, &s.params, s.threads as u64, equity, &self.limit)
    }

    /// The live prefix of chunk `index` of stream `k`, where its files are uploaded as it runs
    /// (`runs/<run>/partial/<stream>/<index:06>/out`).
    fn live_prefix(&self, k: usize, index: u64) -> String {
        format!(
            "{}/partial/{}/{index:06}/out",
            self.prefix(),
            self.m.streams[k].name
        )
    }

    fn go(&mut self) -> Result<(), String> {
        fs::create_dir_all(&self.root).map_err(|e| e.to_string())?;
        // The binaries must be the ones the manifest names.
        if let (Some(want), Some(have)) = (
            self.m.bin_sha.clone().filter(|s| !s.is_empty()),
            env_nonempty("BIN_SHA"),
        ) && want != have
        {
            return Err(format!(
                "manifest bin_sha {want} but these binaries are {have}"
            ));
        }
        // Heartbeat numbering continues across machines.
        let hb_prefix = format!("{}/heartbeat/", self.prefix());
        self.hb_seq = self
            .net_call(move |r| retry(3, || r.list(&hb_prefix)))?
            .iter()
            .filter_map(|k| {
                k.rsplit('/')
                    .next()?
                    .strip_suffix(".json")?
                    .parse::<u64>()
                    .ok()
            })
            .max()
            .map_or(0, |n| n + 1);
        self.heartbeat(true, "start");
        let stamp = compact_utc();
        self.put_json(
            "hardware.json",
            &format!("{}/hardware/{stamp}.json", self.prefix()),
            &hardware(),
        )?;
        self.fetch_inputs()?;
        self.resume()?;
        if self.m.benchmark_seconds > 0 {
            self.heartbeat(true, "benchmark");
            let mut ctx = Ctx::local(&self.m.run_id, &self.root.join("bench"), self.threads);
            let b = crate::benchmark(&mut ctx, self.m.benchmark_seconds)?;
            self.put_json(
                "benchmark.json",
                &format!("{}/benchmark/{stamp}.json", self.prefix()),
                &b,
            )?;
        }
        let res = self.schedule();
        // Whatever happened, keep what finished.
        if self.dirty {
            self.write_done()?;
        }
        res?;
        self.heartbeat(true, "checking");
        self.final_check()?;
        let merged = self.merged();
        self.put_json(
            "merged.json",
            &format!("{}/merged.json", self.prefix()),
            &merged,
        )?;
        // The run manifest goes last before the marker (PLAN §12.5).
        let fm = self.final_manifest();
        self.put_json(
            "final-manifest.json",
            &format!("{}/final-manifest.json", self.prefix()),
            &fm,
        )
    }

    fn fetch_inputs(&mut self) -> Result<(), String> {
        let dir = self.root.parent().unwrap_or(&self.root).join("inputs");
        for (name, hex) in self.m.inputs.clone() {
            let local = dir.join(&hex);
            let (h2, l2) = (hex.clone(), local.clone());
            self.net_call(move |r| crate::remote::fetch_input(r, &h2, &l2))
                .map_err(|e| format!("input `{name}`: {e}"))?;
            let abs = fs::canonicalize(&local).unwrap_or(local);
            self.inputs.insert(name, abs.display().to_string());
        }
        Ok(())
    }

    fn stream_index(&self, name: &str) -> Option<usize> {
        self.m.streams.iter().position(|s| s.name == name)
    }

    /// Reads the highest done-list; keeps the entries whose remote bundles still match (each read
    /// back in full and hashed), so everything else is redone from its seed.
    fn resume(&mut self) -> Result<(), String> {
        let done_prefix = format!("done/{}/", self.m.run_id);
        let keys = self.net_call(move |r| retry(3, || r.list(&done_prefix)))?;
        // Local leftovers of unfinished chunks are never reused.
        let _ = fs::remove_dir_all(self.root.join("chunks"));
        let _ = fs::remove_dir_all(self.root.join("bundles"));
        let Some((seq, key)) = latest_done(&keys) else {
            return Ok(());
        };
        self.done_seq = seq + 1;
        let local = self.root.join("tmp").join("done-latest.json");
        let (k2, l2) = (key.clone(), local.clone());
        if !self.net_call(move |r| retry(3, || r.get(&k2, &l2)))? {
            return Err(format!("{key} vanished"));
        }
        let doc: Value =
            serde_json::from_str(&fs::read_to_string(&local).map_err(|e| e.to_string())?)
                .map_err(|e| format!("{key}: {e}"))?;
        let mut seen = BTreeSet::new();
        let mut redo = 0;
        for e in doc["entries"].as_array().cloned().unwrap_or_default() {
            self.heartbeat(false, "resume");
            let (Some(stream), Some(index)) = (e["stream"].as_str(), e["index"].as_u64()) else {
                continue;
            };
            let Some(k) = self.stream_index(stream) else {
                continue;
            };
            if !seen.insert((k, index)) {
                continue;
            }
            let mut ok = e["outputs"].as_array().is_some_and(|o| !o.is_empty());
            for o in e["outputs"].as_array().into_iter().flatten() {
                let key = o["path"].as_str().unwrap_or_default().to_string();
                let want = o["sha256"].as_str().unwrap_or_default().to_string();
                ok &= matches!(self.net_call(move |r| retry(3, || r.sha256(&key)))?,
                               Some((h, _)) if h == want);
            }
            if !ok {
                redo += 1;
                self.dirty = true;
                continue;
            }
            self.record(k, index, &e);
            self.entries.push(e);
        }
        println!(
            "runner: resuming from {key}: {} chunks done, {redo} listed chunks to redo",
            self.entries.len()
        );
        Ok(())
    }

    fn record(&mut self, k: usize, index: u64, e: &Value) {
        let st = &mut self.states[k];
        st.done.insert(index, e["seconds"].as_f64().unwrap_or(0.0));
        if let Some(b) = e.get("blocks").and_then(Value::as_array) {
            st.scores
                .insert(index, b.iter().filter_map(Value::as_i64).collect());
        }
    }

    fn write_done(&mut self) -> Result<(), String> {
        let key = format!(
            "done/{}/{:08}-{}.json",
            self.m.run_id,
            self.done_seq,
            compact_utc()
        );
        let doc = json!({ "run_id": self.m.run_id, "seq": self.done_seq,
                          "utc": utc_string(unix_now()), "entries": self.entries });
        let p = self
            .root
            .join("tmp")
            .join(format!("done-{}.json", self.done_seq));
        fs::create_dir_all(p.parent().unwrap_or(&self.root)).map_err(|e| e.to_string())?;
        fs::write(
            &p,
            serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())? + "\n",
        )
        .map_err(|e| e.to_string())?;
        let p2 = p.clone();
        self.net_call(move |r| retry(3, || r.put(&p2, &key, true)))?;
        let _ = fs::remove_file(&p);
        self.done_seq += 1;
        self.dirty = false;
        self.last_done_write = Instant::now();
        kill_point("after_done", self.entries.len() as u64);
        Ok(())
    }

    fn spawn(&mut self, k: usize, index: u64) -> Result<Running, String> {
        let s = &self.m.streams[k];
        let seed = chunk_seed(self.m.run_seed, k as u64, index);
        let dir = self
            .root
            .join("chunks")
            .join(&s.name)
            .join(format!("{index:06}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let logs = dir.join("logs");
        fs::create_dir_all(&logs).map_err(|e| e.to_string())?;
        if s.kind == "search_measure" || s.kind == "student_train" {
            // Continue where an earlier machine stopped (a `student_train` chunk: its finished
            // (size, width) pairs, which it then does not train again): its files as last uploaded, with the
            // checkpoint of the worlds finished since its last complete decision. A store that
            // cannot be read is retried three times; if it still fails, the chunk starts from
            // scratch (the same bytes, the saved worlds valued again) and the log says so loudly.
            let (name, live, out) = (s.name.clone(), self.live_prefix(k, index), dir.join("out"));
            let l2 = live.clone();
            match self
                .net_call(move |r| retry(3, || net::restore(r, std::slice::from_ref(&l2), &out)))
            {
                Ok(got) if got.listed == 0 => {}
                Ok(got) => println!(
                    "runner: chunk {name}/{index}: {} files listed, {} files ({} bytes) restored \
                     from {live}",
                    got.listed, got.files, got.bytes
                ),
                Err(e) => eprintln!(
                    "runner: WARNING: chunk {name}/{index}: could not restore {live} after 3 \
                     tries ({e}); the chunk starts from scratch and values the saved worlds again"
                ),
            }
        }
        let s = &self.m.streams[k];
        let spec = json!({
            "stream": s.name, "index": index, "seed": seed, "kind": s.kind,
            "params": s.params, "threads": s.threads, "inputs": self.inputs,
            "blocks": s.blocks_in(index), "dir": dir.display().to_string(),
        });
        let sp = dir.join("spec.json");
        fs::write(&sp, spec.to_string()).map_err(|e| e.to_string())?;
        let log = fs::File::create(logs.join("worker.log")).map_err(|e| e.to_string())?;
        let log2 = log.try_clone().map_err(|e| e.to_string())?;
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        // stdin is a pipe the worker watches: it ends when this process dies or stops the chunk.
        let child = Command::new(exe)
            .arg("chunk")
            .arg(&sp)
            .stdin(Stdio::piped())
            .stdout(log)
            .stderr(log2)
            .env("RAYON_NUM_THREADS", s.threads.to_string())
            .spawn()
            .map_err(|e| format!("cannot start chunk worker: {e}"))?;
        self.states[k].running.insert(index);
        Ok(Running {
            child,
            stream: k,
            index,
            seed,
            dir,
            started: Instant::now(),
            mem: 0,
        })
    }

    /// Uploads a chunk's logs (worker and child-process output: diagnostics with timings, so
    /// outside the deterministic bundle) to `runs/<run>/logs/<stream>/<index:06>.tar.zst`,
    /// replacing any earlier attempt's. Never fatal.
    fn upload_logs(&mut self, r: &Running) {
        let name = self.m.streams[r.stream].name.clone();
        let local = self
            .root
            .join("tmp")
            .join(format!("logs-{name}-{:06}.tar.zst", r.index));
        let key = format!("{}/logs/{name}/{:06}.tar.zst", self.prefix(), r.index);
        let res = bundle_dir(&r.dir.join("logs"), &local).and_then(|_| {
            let l2 = local.clone();
            self.net_call(move |rem| retry(3, || rem.put(&l2, &key, false)))
        });
        if let Err(e) = res {
            eprintln!("runner: logs of {name}/{}: {e}", r.index);
        }
        let _ = fs::remove_file(&local);
    }

    /// Keeps what a chunk stopped mid-way had finished: its `out/` as it stands, bundled and
    /// uploaded to `runs/<run>/partial/<stream>/<index:06>-<utc>.tar.zst` (outside the done-list,
    /// so a resumed run still redoes the chunk from its seed), plus its logs. Never fatal.
    fn keep_partial(&mut self, r: &Running) {
        self.states[r.stream].running.remove(&r.index);
        let name = self.m.streams[r.stream].name.clone();
        // First the chunk's files to its live prefix, checkpoints first: what a resumed run
        // continues from (no older background copy may land after them).
        self.quiesce_partial();
        let (dir, live, snap, done) = (
            r.dir.join("out"),
            self.live_prefix(r.stream, r.index),
            self.root.join("tmp").join("partial-stop"),
            self.net.sent(),
        );
        let (sent, failed) =
            self.net_call(move |rem| net::upload_changed(rem, &dir, &live, &snap, &done, None));
        println!(
            "runner: chunk {name}/{}: {sent} files uploaded to its live prefix ({failed} failed)",
            r.index
        );
        let local = self
            .root
            .join("tmp")
            .join(format!("partial-{name}-{:06}.tar.zst", r.index));
        let key = format!(
            "{}/partial/{name}/{:06}-{}.tar.zst",
            self.prefix(),
            r.index,
            compact_utc()
        );
        let res = bundle_dir(&r.dir.join("out"), &local).and_then(|(sha, bytes)| {
            let (l2, k2) = (local.clone(), key.clone());
            self.net_call(move |rem| retry(3, || rem.put(&l2, &k2, false)))?;
            Ok((sha, bytes))
        });
        match res {
            Ok((sha, bytes)) => {
                println!(
                    "runner: chunk {name}/{} stopped; kept its partial outputs in {key}",
                    r.index
                );
                self.partial
                    .push(json!({ "stream": name, "index": r.index, "seed": r.seed,
                                          "path": key, "sha256": sha, "bytes": bytes }));
            }
            Err(e) => eprintln!("runner: partial outputs of {name}/{}: {e}", r.index),
        }
        let _ = fs::remove_file(&local);
        self.upload_logs(r);
    }

    /// Bundle, upload, verify, list one finished chunk.
    fn finish(&mut self, r: &Running) -> Result<(), String> {
        let s = &self.m.streams[r.stream];
        let name = s.name.clone();
        let arena = s.kind == "arena_blocks";
        let local = self
            .root
            .join("bundles")
            .join(&name)
            .join(format!("{:06}.tar.zst", r.index));
        let (sha, bytes) = bundle_dir(&r.dir.join("out"), &local)?;
        let key = format!(
            "{}/out/{name}/{:06}-{}.tar.zst",
            self.prefix(),
            r.index,
            &sha[..16]
        );
        let n = self.entries.len() as u64 + 1;
        UPLOAD_ENTRY.store(n, Ordering::Relaxed);
        // A failure here ends the run (no new chunks, then FAILED), so a network failure of a
        // few minutes is waited out (`retry_patient`; the heartbeat goes on meanwhile).
        let (l2, k2) = (local.clone(), key.clone());
        if let Err(e) = self.net_call(move |rem| retry_patient(8, || rem.put(&l2, &k2, true))) {
            let w = result_upload_warning(&name, r.index, &key, bytes, 8, &e);
            eprintln!("runner: WARNING: {w}");
            self.warned.insert(w.clone());
            self.delivery_warnings.push(w);
            // Say so in a heartbeat now, not at the next interval.
            self.heartbeat(true, "result_upload_failed");
            return Err(format!("upload {key}: {e}"));
        }
        // Read back in full and hash before listing it.
        let k2 = key.clone();
        match self.net_call(move |rem| retry_patient(8, || rem.sha256(&k2)))? {
            Some((h, b)) if h == sha && b == bytes => {}
            other => return Err(format!("upload check {key}: got {other:?}, want {sha}")),
        }
        kill_point("after_upload", n);
        let seconds = (r.started.elapsed().as_secs_f64() * 10.0).round() / 10.0;
        let mut e = json!({ "stream": name, "index": r.index, "seed": r.seed,
            "outputs": [{ "path": key, "sha256": sha, "bytes": bytes }], "seconds": seconds });
        if arena {
            let res: Value = serde_json::from_str(
                &fs::read_to_string(r.dir.join("out").join("result.json"))
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            e["blocks"] = res["scores"].clone();
        }
        self.states[r.stream].running.remove(&r.index);
        self.record(r.stream, r.index, &e);
        self.entries.push(e);
        self.dirty = true;
        self.upload_logs(r);
        let _ = fs::remove_file(&local);
        let _ = fs::remove_dir_all(&r.dir);
        let kind = self.m.streams[r.stream].kind.clone();
        if let Some(note) =
            finished_median(&self.m.streams, &self.states, &self.m.streams[r.stream])
                .and_then(size_note)
            && self.noted.insert(kind.clone())
        {
            println!("runner: {kind}: {note}");
        }
        if self.last_done_write.elapsed().as_secs() >= self.m.upload_every_s {
            self.write_done_periodic();
        }
        Ok(())
    }

    /// The periodic done-list write. A failure (the network down past `retry`'s three tries) is
    /// logged and tried again at the next interval: the chunks' bundles are already uploaded and
    /// verified, and the run's last write (`go`) must still succeed. Before review E
    /// (2026-10-16) it ended the run: from the scheduler's loop it returned at once, dropping the
    /// running chunks (their workers stop when the orchestrator lets go of them) and writing
    /// FAILED, so a one-minute network failure lost every running chunk's unsaved work and ended
    /// the run with no replacement.
    fn write_done_periodic(&mut self) {
        if let Err(e) = self.write_done() {
            eprintln!(
                "runner: WARNING: the done-list could not be written ({e}); trying again in {} s",
                self.m.upload_every_s
            );
            self.last_done_write = Instant::now();
        }
    }

    fn schedule(&mut self) -> Result<(), String> {
        let mut running: Vec<Running> = Vec::new();
        let mut error: Option<String> = None;
        let mut stopped = false;
        loop {
            let now = unix_now();
            if self.m.deadline.is_some_and(|d| now >= d) {
                stop::request("the manifest's deadline_utc was reached");
            }
            if let Some(why) = stop::requested()
                && !stopped
            {
                eprintln!("runner: {why}; stopping running chunks and keeping what they finished");
                stopped = true;
                for r in &mut running {
                    stop_child(&mut r.child);
                }
                // Chunks that finished just before the stop are kept as usual.
                for mut r in running.drain(..) {
                    if matches!(r.child.try_wait(), Ok(Some(st)) if st.success()) {
                        if let Err(e) = self.finish(&r) {
                            error.get_or_insert(e);
                        }
                    } else {
                        self.keep_partial(&r);
                    }
                }
            }
            // Start chunks while threads are free and their memory fits beside the running ones.
            let mut used: usize = running
                .iter()
                .map(|r| self.m.streams[r.stream].threads)
                .sum();
            let mut held: u64 = running.iter().map(|r| r.mem).sum();
            let mem_budget = crate::mem::budget(0, &self.limit).0;
            if error.is_none() && !stopped {
                loop {
                    // A chunk needing more threads than the machine has still runs, alone.
                    let free = if used == 0 {
                        usize::MAX
                    } else {
                        self.threads.saturating_sub(used)
                    };
                    let Some((k, i)) =
                        pick(&self.m.streams, &self.states, free, now, self.m.deadline)
                    else {
                        break;
                    };
                    // Lower-priority chunks never jump ahead of one that waits for memory.
                    let need = self.chunk_need(k);
                    if !fits_memory(held, need, running.len(), mem_budget) {
                        if self.mem_waits.insert((k, i)) {
                            println!(
                                "runner: chunk {}/{i} waits for memory: it needs {} MB, the \
                                 running chunks hold {} MB of the {} MB budget",
                                self.m.streams[k].name,
                                need >> 20,
                                held >> 20,
                                mem_budget >> 20
                            );
                        }
                        break;
                    }
                    let mut r = self.spawn(k, i)?;
                    r.mem = need;
                    held += need;
                    used += self.m.streams[k].threads;
                    running.push(r);
                }
            }
            if running.is_empty() {
                break;
            }
            // Free threads with nothing to start (the tail, or a deadline cut-off), or a chunk
            // that says so itself: partial CPU use is expected.
            self.partial_ok = used < self.threads || running.iter().any(|r| chunk_partial(&r.dir));
            let warnings: Vec<String> = running
                .iter()
                .flat_map(|r| {
                    let name = &self.m.streams[r.stream].name;
                    chunk_warnings(&r.dir)
                        .into_iter()
                        .map(move |w| format!("{name}/{}: {w}", r.index))
                })
                .collect();
            for w in &warnings {
                if self.warned.insert(w.clone()) {
                    eprintln!("runner: WARNING: chunk {w}");
                }
            }
            self.warnings = warnings;
            self.chunk_etas = running
                .iter()
                .filter_map(|r| {
                    let name = &self.m.streams[r.stream].name;
                    chunk_eta(&r.dir).map(|e| (format!("{name}/{}", r.index), e))
                })
                .collect();
            self.chunk_steps = running
                .iter()
                .filter_map(|r| {
                    let name = &self.m.streams[r.stream].name;
                    chunk_step(&r.dir).map(|e| (format!("{name}/{}", r.index), e))
                })
                .collect();
            // Every upload_every_s, the running chunks' changed files go to their live prefixes
            // in the background (checkpoints first), so a replacement machine can continue them.
            if !stopped
                && self.last_partial.elapsed().as_secs() >= self.m.upload_every_s
                && !self.net.partial_busy()
            {
                let sets = running
                    .iter()
                    .map(|r| (r.dir.join("out"), self.live_prefix(r.stream, r.index)))
                    .collect();
                self.net
                    .partial(sets, self.root.join("tmp").join("partial"));
                self.last_partial = Instant::now();
            }
            // Reap finished workers in a fixed order.
            let mut i = 0;
            while i < running.len() {
                match running[i].child.try_wait() {
                    Ok(None) => i += 1,
                    Ok(Some(st)) => {
                        let r = running.remove(i);
                        if st.success() {
                            // Chunks in flight when another failed are still kept.
                            if let Err(e) = self.finish(&r) {
                                error.get_or_insert(e);
                            }
                        } else if stop::requested().is_some() {
                            // Ended by the stop (e.g. Ctrl-C reached its processes first).
                            self.keep_partial(&r);
                        } else {
                            self.states[r.stream].running.remove(&r.index);
                            let name = self.m.streams[r.stream].name.clone();
                            eprintln!(
                                "runner: chunk {name}/{} failed ({st}); last lines of its log:\n{}",
                                r.index,
                                log_tail(&r.dir.join("logs").join("worker.log"), 20)
                            );
                            self.upload_logs(&r);
                            // No new chunks after a failure: the run ends FAILED once the
                            // chunks in flight are done.
                            if error.is_none() {
                                error = Some(format!(
                                    "chunk {name}/{} failed ({st}); logs in {}/logs/{name}/{:06}.tar.zst",
                                    r.index,
                                    self.prefix(),
                                    r.index
                                ));
                            }
                        }
                    }
                    Err(e) => return Err(e.to_string()),
                }
            }
            if self.dirty && self.last_done_write.elapsed().as_secs() >= self.m.upload_every_s {
                self.write_done_periodic();
            }
            self.heartbeat(false, "chunks");
            std::thread::sleep(Duration::from_millis(100));
        }
        self.partial_ok = false;
        self.warnings.clear();
        self.chunk_etas.clear();
        self.chunk_steps.clear();
        match error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// The end check (PLAN §12.5's `rclone check`): every output in the done-list is in the
    /// remote with its recorded size. Each was also read back in full and hashed when it was
    /// uploaded (this session) or when the run resumed (earlier sessions).
    fn final_check(&mut self) -> Result<(), String> {
        let out_prefix = format!("{}/out/", self.prefix());
        let listed: BTreeMap<String, u64> = self
            .net_call(move |r| retry(3, || r.list_sized(&out_prefix)))?
            .into_iter()
            .collect();
        for e in &self.entries {
            for o in e["outputs"].as_array().into_iter().flatten() {
                let key = o["path"].as_str().unwrap_or_default();
                let want = o["bytes"].as_u64();
                match listed.get(key) {
                    Some(&n) if Some(n) == want => {}
                    other => {
                        return Err(format!(
                            "final check: {key} has {other:?} bytes, want {want:?}"
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn queue(&self) -> Value {
        let v: Vec<Value> = self
            .m
            .streams
            .iter()
            .zip(&self.states)
            .map(|(s, st)| {
                let med = finished_median(&self.m.streams, &self.states, s);
                let mut q = json!({ "stream": s.name, "done": st.done.len(),
                        "running": st.running.len(), "left": st.left(s), "decided": st.decided(s),
                        "median_s": med.map(|x| (x * 10.0).round() / 10.0) });
                if let Some(note) = med.and_then(size_note) {
                    q["size_note"] = json!(note);
                }
                q
            })
            .collect();
        json!(v)
    }

    /// Ends the background uploads of the chunks' live files (between files), heartbeating
    /// meanwhile, so an older copy never lands after a stop's own upload.
    fn quiesce_partial(&mut self) {
        self.net.cancel_partial();
        let t0 = Instant::now();
        while self.net.partial_busy() && t0.elapsed() < Duration::from_secs(300) {
            let phase = self.phase.clone();
            self.heartbeat(false, &phase);
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Writes `heartbeat.json` (every `heartbeat_every_s`, or now with `force`) and hands its
    /// upload to the background (`net.rs`); also checks the local STOP file and starts a
    /// background check of the remote one.
    fn heartbeat(&mut self, force: bool, phase: &str) {
        self.phase = phase.to_string();
        let every = Duration::from_secs(self.m.heartbeat_every_s);
        if !force && self.last_hb.is_some_and(|t| t.elapsed() < every) {
            return;
        }
        self.last_hb = Some(Instant::now());
        // The launcher's STOP object (checked in the background) or a local STOP file.
        stop::poll_local(&self.root);
        self.net.poll_stop(&self.root);
        let cpu = self.cpu.sample(None);
        let round = |x: Option<f64>| x.map(|x| (x * 10.0).round() / 10.0);
        let elapsed = self.start.elapsed().as_secs_f64();
        let mut progress = json!({ "queue": self.queue(), "entries": self.entries.len(),
                                   "partial_cpu_ok": self.partial_ok, "warnings": heartbeat_warnings(&self.warnings, &self.delivery_warnings) });
        if !self.chunk_etas.is_empty() {
            progress["chunk_eta"] = json!(self.chunk_etas);
        }
        if !self.chunk_steps.is_empty() {
            progress["chunk_steps"] = json!(self.chunk_steps);
        }
        if let [(_, e)] = self.chunk_etas.iter().collect::<Vec<_>>()[..] {
            // One chunk running: its estimate is the run's, priced over the whole run.
            let mut e = e.clone();
            e["projected_usd"] = json!(
                e["remaining_s"]
                    .as_f64()
                    .filter(|_| self.dph > 0.0)
                    .map(|r| ((elapsed + r) / 3600.0 * self.dph * 100.0).round() / 100.0)
            );
            progress["eta"] = e;
        }
        let mut hb = json!({
            "run_id": self.m.run_id, "seq": self.hb_seq, "utc": utc_string(unix_now()),
            "phase": phase,
            "progress": progress,
            "elapsed_s": elapsed.round(), "threads": self.threads,
            "usable_cores": (self.cpu.cores * 100.0).round() / 100.0,
            "cpu_util_pct": round(cpu.cgroup), "cpu_util_source": "cgroup",
            "host_cpu_busy_pct": round(cpu.host),
            "est_cost_usd": (elapsed / 3600.0 * self.dph * 10_000.0).round() / 10_000.0,
        });
        self.net.add_health(&mut hb);
        let text = serde_json::to_string_pretty(&hb).unwrap_or_default() + "\n";
        let local = self.root.join("heartbeat.json");
        if let Err(e) = fs::create_dir_all(&self.root).and_then(|()| fs::write(&local, &text)) {
            eprintln!("runner: heartbeat {}: {e}", local.display());
        }
        let key = format!("{}/heartbeat/{:06}.json", self.prefix(), self.hb_seq);
        self.net.heartbeat(
            text,
            key,
            self.root.join("tmp").join("heartbeat-upload.json"),
        );
        self.hb_seq += 1;
    }

    /// The merged report: depends only on which chunks are done and their bundles, never on
    /// timing or completion order. A decided `until_decided` stream lists only the chunks of the
    /// decided prefix: chunks still in flight at the decision finish and stay in the done-list,
    /// but how many there were depends on timing, so they are left out here.
    fn merged(&self) -> Value {
        let mut by: Vec<BTreeMap<u64, &Value>> = vec![BTreeMap::new(); self.m.streams.len()];
        for e in &self.entries {
            if let (Some(k), Some(i)) = (
                e["stream"].as_str().and_then(|s| self.stream_index(s)),
                e["index"].as_u64(),
            ) {
                by[k].insert(i, e);
            }
        }
        let streams: Vec<Value> = self
            .m
            .streams
            .iter()
            .zip(&self.states)
            .zip(&by)
            .map(|((s, st), es)| {
                let prefix = s.sprt.as_ref().map(|sp| sprt_prefix(&st.scores, sp));
                let cut = match &prefix {
                    Some(p) if s.chunks == Chunks::UntilDecided && st.decided(s) => p.chunks,
                    _ => u64::MAX,
                };
                let chunks: Vec<Value> = es
                    .range(..cut)
                    .map(|(_, e)| json!({ "index": e["index"], "seed": e["seed"], "outputs": e["outputs"] }))
                    .collect();
                let mut v = json!({ "name": s.name, "kind": s.kind, "priority": s.priority,
                    "chunks_total": s.total(), "chunks_done": chunks.len(), "left": st.left(s),
                    "chunks": chunks });
                if let Some(p) = prefix {
                    v["sprt"] = json!({ "blocks": p.blocks, "chunks": p.chunks, "llr": p.llr,
                        "decision": format!("{:?}", p.decision), "mean": p.mean, "sd": p.sd,
                        "until_decided": s.chunks == Chunks::UntilDecided });
                }
                v
            })
            .collect();
        let complete = self
            .m
            .streams
            .iter()
            .zip(&self.states)
            .all(|(s, st)| st.left(s) == 0);
        json!({ "run_id": self.m.run_id, "run_seed": self.m.run_seed, "complete": complete,
                "streams": streams })
    }

    fn final_manifest(&self) -> Value {
        let seeds: Vec<Value> = self
            .m
            .streams
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let n = self.states[k].done.keys().max().map_or(0, |m| m + 1);
                json!({ "stream": s.name, "seeds": (0..n).map(|i| chunk_seed(self.m.run_seed, k as u64, i)).collect::<Vec<_>>() })
            })
            .collect();
        json!({
            "manifest": self.m.raw,
            // Release binaries are named by their commit (BIN_SHA); a local build may set GIT_COMMIT.
            "git_commit": option_env!("GIT_COMMIT").map(str::to_string)
                .or_else(|| env_nonempty("GIT_COMMIT")).or_else(|| env_nonempty("BIN_SHA")),
            "bin_sha": env_nonempty("BIN_SHA"),
            "bin_build": env_nonempty("BIN_BUILD"),
            "bin_rustc": env_nonempty("BIN_RUSTC"),
            "image": env_nonempty("IMAGE"),
            "image_digest": env_nonempty("IMAGE_DIGEST"),
            "runner_version": env!("CARGO_PKG_VERSION"),
            "threads": self.threads,
            "stream_threads": self.m.streams.iter().map(|s| json!({ "stream": s.name, "threads": s.threads })).collect::<Vec<_>>(),
            "inputs": self.m.inputs.iter().map(|(n, h)| json!({ "name": n, "sha256": h })).collect::<Vec<_>>(),
            "chunk_seeds": seeds,
            "done_list_seq": self.done_seq.saturating_sub(1),
            "warnings": self.warned,
            "stopped": stop::requested(),
            "partial": self.partial,
            "utc": utc_string(unix_now()),
        })
    }
}

/// Whether a chunk needing `need` bytes may start beside running chunks that hold `held`:
/// always when none runs (a chunk sizes its own tables to the machine), else only within
/// `budget` (the container's memory less the safety margin, `mem::budget`).
fn fits_memory(held: u64, need: u64, running: usize, budget: u64) -> bool {
    running == 0 || held.saturating_add(need) <= budget
}

/// Whether a running chunk's own heartbeat (written by its worker into the chunk directory)
/// declares a stretch of deliberately partial CPU use.
fn chunk_partial(dir: &Path) -> bool {
    fs::read_to_string(dir.join("heartbeat.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v["progress"]["partial_cpu_ok"].as_bool())
        .unwrap_or(false)
}

/// The progress estimate in a running chunk's own heartbeat (`progress.eta`), if it has one.
fn chunk_eta(dir: &Path) -> Option<Value> {
    fs::read_to_string(dir.join("heartbeat.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .map(|v| v["progress"]["eta"].clone())
        .filter(Value::is_object)
}

/// The step a running chunk's own heartbeat names (`progress.step`, e.g. a `hybrid_rung`'s
/// `labels random (1 of 3)`), with its counts and its training progress, for the run's heartbeat
/// (`progress.chunk_steps`); `None` for kinds that name no step.
fn chunk_step(dir: &Path) -> Option<Value> {
    let v: Value =
        serde_json::from_str(&fs::read_to_string(dir.join("heartbeat.json")).ok()?).ok()?;
    let p = &v["progress"];
    let step = p["step"].as_str()?;
    let mut out = json!({ "step": step });
    let keep = |k: &str| {
        k.ends_with("_done")
            || k.ends_with("_total")
            || k.starts_with("train_")
            || [
                "device",
                "rows_done",
                "rows_total",
                "verdict",
                // The audit loop's (hybrid_rung, docs/m3-hybrid/audit-loop.md §4 "Reporting").
                "round",
                "groups_found",
                "groups_confirmed",
                "groups_remaining",
                "groups_failing",
                "worst_group",
                "shape",
                "shape_step",
                "thresholds",
                "slices_failing",
                "last_check",
                "audit_stop",
            ]
            .contains(&k)
    };
    if let Some(o) = p.as_object() {
        for (k, x) in o {
            if keep(k) && k != "steps_done" && !x.is_object() {
                out[k.as_str()] = x.clone();
            }
        }
    }
    Some(out)
}

/// The warning for a chunk result that could not be uploaded after all retries.
fn result_upload_warning(
    stream: &str,
    index: u64,
    key: &str,
    bytes: u64,
    attempts: u32,
    err: &str,
) -> String {
    format!(
        "RESULT NOT DELIVERED: upload of {stream}/{index} ({:.1} MiB, {key}) failed after \
         {attempts} attempts: {err}",
        bytes as f64 / 1_048_576.0
    )
}

/// The heartbeat's warnings: the running chunks' own, then the standing delivery warnings.
fn heartbeat_warnings(chunk: &[String], delivery: &[String]) -> Vec<String> {
    chunk.iter().chain(delivery).cloned().collect()
}

/// The warnings a running chunk's own heartbeat lists (`progress.warnings`).
fn chunk_warnings(dir: &Path) -> Vec<String> {
    fs::read_to_string(dir.join("heartbeat.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| {
            v["progress"]["warnings"].as_array().map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
        })
        .unwrap_or_default()
}

/// The last `n` lines of a log file.
fn log_tail(p: &Path, n: usize) -> String {
    let t = fs::read_to_string(p).unwrap_or_default();
    let lines: Vec<&str> = t.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// Stops a chunk worker: closing its stdin makes it stop its own child and exit; kill it if it
/// has not gone after a few seconds.
fn stop_child(c: &mut Child) {
    drop(c.stdin.take());
    for _ in 0..30 {
        if matches!(c.try_wait(), Ok(Some(_))) {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let _ = c.kill();
    let _ = c.wait();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed manifests as a rented run takes them: the scheduler's reading (`parse_v2`,
    /// which turns stored keys into input names), each stream's settings with the inputs' local
    /// paths added as the worker adds them ([`chunk_params`]), then the chunk's own reading of
    /// those settings and, for the student's teacher, its `search` command. A key added by the
    /// resolution that a chunk refused stopped student-t2-gv2-1 in 5 s (2026-10-20).
    #[test]
    fn manifests_resolve_into_settings_their_chunks_accept() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/vast");
        for (file, kinds) in [
            ("student-t2-gv2.manifest.json", 2),
            ("student-t2-base.manifest.json", 3),
            ("m3-hybrid-t2-hlabels.manifest.json", 1),
            ("o3-learnedleaf.manifest.json", 1),
            ("o2-leafrisk.manifest.json", 1),
            ("o2-gv2-labels.manifest.json", 1),
            ("o2-labeltest.manifest.json", 2),
            ("o3-round-1.manifest.json", 1),
        ] {
            let v: Value = serde_json::from_str(&fs::read_to_string(root.join(file)).expect(file))
                .expect(file);
            let m = crate::sched::parse_v2(&v).unwrap_or_else(|e| panic!("{file}: {e}"));
            let inputs: BTreeMap<String, String> = m
                .inputs
                .iter()
                .map(|(n, _)| (n.clone(), format!("/in/{n}")))
                .chain([("equity".to_string(), "/in/equity".to_string())])
                .collect();
            let mut seen = 0;
            for st in &m.streams {
                let p = chunk_params(&st.kind, &st.params, 7, &inputs)
                    .unwrap_or_else(|e| panic!("{file}/{}: {e}", st.name));
                let what = format!("{file}/{}", st.name);
                match st.kind.as_str() {
                    "student_teacher" => {
                        let q = crate::student::parse(&p).unwrap_or_else(|e| panic!("{what}: {e}"));
                        let leaves = p.get("exact_leaves_path").and_then(Value::as_str);
                        assert_eq!(leaves.is_some(), q.exact_leaves.is_some(), "{what}");
                        if let Some(l) = leaves {
                            assert!(l.starts_with("learned:/in/"), "{what}: {l}");
                        }
                        let c = crate::student::command(
                            Path::new("search"),
                            &q,
                            0,
                            p["net"].as_str().expect("net"),
                            leaves.map(|l| (l, p.get("defer_expert_path").and_then(Value::as_str))),
                            Path::new("/in/equity"),
                            Path::new("o.npy"),
                            4,
                            Path::new("p.json"),
                        );
                        let args: Vec<String> = c
                            .get_args()
                            .map(|a| a.to_string_lossy().into_owned())
                            .collect();
                        assert_eq!(
                            args.iter().any(|a| a == "--exact-leaves"),
                            leaves.is_some(),
                            "{what}"
                        );
                        assert!(!args.iter().any(|a| a.contains("inputs/sha256/")), "{what}");
                    }
                    "hybrid_rung" => {
                        let q = crate::hybrid::parse(&p).unwrap_or_else(|e| panic!("{what}: {e}"));
                        if file.contains("hlabels") {
                            let h = q.hybrid_labels.as_ref().expect("hybrid labels");
                            assert!(h.leafrisk.is_some(), "{what}: a learned model to fetch");
                        }
                    }
                    "hybrid_checks" => {
                        crate::checks::parse(&p).unwrap_or_else(|e| panic!("{what}: {e}"));
                    }
                    "leafrisk" => {
                        crate::leafrisk::parse(&p).unwrap_or_else(|e| panic!("{what}: {e}"));
                    }
                    k => panic!("{what}: kind {k} not covered"),
                }
                seen += 1;
            }
            assert_eq!(seen, kinds, "{file}");
        }
    }

    #[test]
    fn failed_result_upload_is_a_clear_heartbeat_warning() {
        let w = result_upload_warning(
            "rung",
            7,
            "runs/x/out/rung/000007-ab.tar.zst",
            31_457_280,
            8,
            "killed after 59 s (its hard limit)",
        );
        assert!(w.starts_with("RESULT NOT DELIVERED"), "{w}");
        assert!(
            w.contains("rung/7") && w.contains("30.0 MiB") && w.contains("8 attempts"),
            "{w}"
        );
        assert!(w.contains("hard limit"), "{w}");
        let chunk = vec!["rung/8: slow".to_string()];
        let all = heartbeat_warnings(&chunk, std::slice::from_ref(&w));
        assert_eq!(all, vec!["rung/8: slow".to_string(), w]);
        assert!(heartbeat_warnings(&[], &[]).is_empty());
    }
    use crate::mem::Limit;

    /// Review E (2026-10-16): the table measurement's three `search_measure` streams of 32
    /// one-thread worlds at 12, 3 and 1 GB tables on a 500 GB machine. Their threads fit
    /// together (96 of 190), but their tables (384 + 96 + 32 GB) do not: the 12 GB stream runs
    /// first, alone, then the other two together, instead of all three at once into the kernel's
    /// out-of-memory killer.
    #[test]
    fn streams_whose_tables_do_not_fit_together_run_one_after_another() {
        let stream = |name: &str, priority: u64, tt: u64| {
            json!({ "name": name, "priority": priority, "kind": "search_measure", "chunks": 1,
                    "params": { "threads": 32, "trick": 1, "worlds_at_once": 32,
                                "solve_threads": 1, "tt_mb": tt } })
        };
        let m = crate::sched::parse_v2(&json!({ "format": 2, "run_id": "t", "run_seed": 1,
            "streams": [stream("tt12g", 1, 12_288), stream("tt3g", 2, 3_072),
                        stream("tt1g", 3, 1_024)] }))
        .expect("manifest");
        let limit = Limit {
            bytes: 500_000_000_000,
            source: "test",
            cgroup: None,
            mem_total: Some(500_000_000_000),
            mem_available: None,
        };
        let budget = crate::mem::budget(0, &limit).0;
        let need = |k: usize| {
            let s = &m.streams[k];
            crate::mem::chunk_need(&s.kind, &s.params, s.threads as u64, 26 << 20, &limit)
        };
        let gib = |b: u64| b >> 30;
        assert_eq!((gib(need(0)), gib(need(1)), gib(need(2))), (385, 97, 33));
        // The scheduler's loop, with 190 threads: start chunks while threads and memory allow.
        let mut states = vec![StreamState::default(); 3];
        let wave = |states: &mut Vec<StreamState>, running: &mut Vec<usize>| {
            let mut started = Vec::new();
            loop {
                let used: usize = running.iter().map(|&k| m.streams[k].threads).sum();
                let held: u64 = running.iter().map(|&k| need(k)).sum();
                let free = if used == 0 { usize::MAX } else { 190 - used };
                let Some((k, i)) = crate::sched::pick(&m.streams, states, free, 0, None) else {
                    break;
                };
                if !fits_memory(held, need(k), running.len(), budget) {
                    break;
                }
                states[k].running.insert(i);
                running.push(k);
                started.push(k);
            }
            started
        };
        let mut running = Vec::new();
        assert_eq!(
            wave(&mut states, &mut running),
            vec![0],
            "the 12 GB stream alone"
        );
        // It finishes; the other two fit together.
        running.clear();
        states[0].running.clear();
        states[0].done.insert(0, 1.0);
        assert_eq!(wave(&mut states, &mut running), vec![1, 2]);
        // Small kinds and a chunk with nothing beside it are never held back.
        assert!(fits_memory(u64::MAX, 1, 0, 0));
        assert_eq!(
            crate::mem::chunk_need("teacher", &json!({}), 1, 0, &limit),
            0
        );
        assert_eq!(
            crate::mem::chunk_need("search_measure", &json!({ "tt_mb": "auto" }), 32, 0, &limit),
            budget,
            "auto tables take the machine"
        );
    }
}
