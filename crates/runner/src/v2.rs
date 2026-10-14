//! Format-2 runs (docs/M3-SPEC.md section 7): several streams of chunks on one machine, with
//! priorities, a deadline, a versioned done-list and resume.
//!
//! The orchestrator (`runner run` on a format-2 manifest) starts each chunk as a worker child
//! process (`runner chunk <spec.json>`), so a chunk can be killed or lost without harming the
//! others. When a worker finishes, the orchestrator bundles its `out/` directory into a
//! deterministic `.tar.zst`, uploads it to `runs/<run>/out/<stream>/<index:06>-<sha16>.tar.zst`
//! (immutable), reads it back to verify the SHA-256, then appends a done-list entry. Every
//! `upload_every_s` the whole done-list is written as a new key `done/<run>/<seq:08>-<utc>.json`.
//!
//! Resume: on start the highest done-list is read; listed chunks whose remote bundles still match
//! are skipped, everything else is redone from its seed (local leftovers are deleted first). A
//! chunk's bundle depends only on its seed and parameters, so a redone chunk reproduces the same
//! bytes and the same key.
//!
//! End: every bundle of this session is re-verified, `merged.json` (the deterministic merged
//! report) is uploaded, then `final-manifest.json` (git commit, image, build, seeds, threads,
//! versions), then `DONE` (or `FAILED`), then the instance destroys itself (M1 behaviour).
//!
//! Remote: `RUNNER_REMOTE_DIR` (a local directory, used by tests and the laptop), else R2 through
//! rclone when `RCLONE_CONFIG_R2_TYPE` and `R2_BUCKET` are set, else `<work>/remote`.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::bundle::{bundle_dir, sha256_file};
use crate::remote::{DirRemote, RcloneRemote, Remote, UPLOAD_ENTRY, kill_point};
use crate::sched::{
    Chunks, Manifest2, StreamState, chunk_seed, kind_median, latest_done, parse_v2, pick,
    sprt_prefix,
};
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
            ctx.run_child(c, "search.log")?;
            json!({ "kind": kind, "seed": seed, "hands": hands })
        }
        "arena_blocks" => arena_blocks(&params, seed, index, &spec, &out, equity().ok())?,
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
            for a in params
                .get("args")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                c.arg(a.as_str().ok_or("params.args must be strings")?);
            }
            c.arg("--out").arg(&out);
            ctx.run_child(c, "search.log")?;
            json!({ "kind": kind, "seed": seed })
        }
        _ => {
            // Existing kinds: their `seed` is the chunk seed, so chunks differ.
            let mut p = params.clone();
            if let Some(o) = p.as_object_mut() {
                o.insert("seed".into(), json!(seed));
            }
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
    let doc = json!({ "a": an, "b": bn, "kind": format!("{kind:?}"), "chunk": index,
                      "seed": seed, "blocks": rows });
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
}

struct Run {
    m: Manifest2,
    remote: Box<dyn Remote>,
    root: PathBuf,
    threads: usize,
    inputs: BTreeMap<String, String>,
    states: Vec<StreamState>,
    /// The done-list, in completion order.
    entries: Vec<Value>,
    /// Entries verified in this session (re-checked at the end).
    session: Vec<usize>,
    done_seq: u64,
    dirty: bool,
    last_done_write: Instant,
    hb_seq: u64,
    last_hb: Option<Instant>,
    cpu: CpuSampler,
    start: Instant,
    dph: f64,
}

fn compact_utc() -> String {
    utc_string(unix_now()).replace(['-', ':'], "")
}

fn choose_remote(work: &Path) -> Box<dyn Remote> {
    if let Ok(d) = std::env::var("RUNNER_REMOTE_DIR")
        && !d.is_empty()
    {
        return Box::new(DirRemote { root: d.into() });
    }
    match (
        std::env::var("RCLONE_CONFIG_R2_TYPE"),
        std::env::var("R2_BUCKET"),
    ) {
        (Ok(_), Ok(b)) if !b.is_empty() => Box::new(RcloneRemote {
            base: format!("r2:{b}"),
        }),
        _ => Box::new(DirRemote {
            root: work.join("remote"),
        }),
    }
}

/// `runner run` on a format-2 manifest.
pub fn run(manifest: &Value, work: &Path) -> ExitCode {
    let m = match parse_v2(manifest) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("runner: manifest: {e}");
            return ExitCode::from(2);
        }
    };
    let remote = choose_remote(work);
    let root = work.join(&m.run_id);
    let threads = m.threads.unwrap_or_else(default_threads);
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
        root,
        threads,
        inputs: BTreeMap::new(),
        entries: Vec::new(),
        session: Vec::new(),
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
    };
    let outcome = r.go();
    let ok = outcome.is_ok();
    let marker = if ok { "DONE" } else { "FAILED" };
    let body = json!({ "utc": utc_string(unix_now()), "elapsed_s": r.start.elapsed().as_secs(),
                       "error": outcome.as_ref().err() });
    r.heartbeat(true, if ok { "uploading" } else { "failed" });
    let uploaded = r.put_json(marker, &format!("{}/{marker}", r.prefix()), &body);
    if let Err(e) = &uploaded {
        eprintln!("runner: upload {marker}: {e}");
    }
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

    fn put_json(&self, local_name: &str, key: &str, v: &Value) -> Result<(), String> {
        let p = self.root.join("tmp").join(local_name);
        fs::create_dir_all(p.parent().unwrap_or(&self.root)).map_err(|e| e.to_string())?;
        fs::write(
            &p,
            serde_json::to_string_pretty(v).map_err(|e| e.to_string())? + "\n",
        )
        .map_err(|e| e.to_string())?;
        self.remote.put(&p, key, false)
    }

    fn go(&mut self) -> Result<(), String> {
        fs::create_dir_all(&self.root).map_err(|e| e.to_string())?;
        // Heartbeat numbering continues across machines.
        self.hb_seq = self
            .remote
            .list(&format!("{}/heartbeat/", self.prefix()))?
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
        for &i in &self.session {
            for o in self.entries[i]["outputs"].as_array().into_iter().flatten() {
                let key = o["path"].as_str().unwrap_or_default();
                let want = o["sha256"].as_str().unwrap_or_default();
                match self.remote.sha256(key)? {
                    Some((h, _)) if h == want => {}
                    other => return Err(format!("final check: {key} is {other:?}, want {want}")),
                }
            }
        }
        let merged = self.merged();
        self.put_json(
            "merged.json",
            &format!("{}/merged.json", self.prefix()),
            &merged,
        )?;
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
            let have = local.is_file() && sha256_file(&local).map(|x| x.0) == Ok(hex.clone());
            if !have {
                let key = format!("inputs/sha256/{hex}");
                if !self.remote.get(&key, &local)? {
                    return Err(format!("input `{name}`: {key} not found"));
                }
                let (h, _) = sha256_file(&local)?;
                if h != hex {
                    let _ = fs::remove_file(&local);
                    return Err(format!("input `{name}`: downloaded sha256 {h}, want {hex}"));
                }
            }
            let abs = fs::canonicalize(&local).unwrap_or(local);
            self.inputs.insert(name, abs.display().to_string());
        }
        Ok(())
    }

    fn stream_index(&self, name: &str) -> Option<usize> {
        self.m.streams.iter().position(|s| s.name == name)
    }

    fn resume(&mut self) -> Result<(), String> {
        let keys = self.remote.list(&format!("done/{}/", self.m.run_id))?;
        // Local leftovers of unfinished chunks are never reused.
        let _ = fs::remove_dir_all(self.root.join("chunks"));
        let _ = fs::remove_dir_all(self.root.join("bundles"));
        let Some((seq, key)) = latest_done(&keys) else {
            return Ok(());
        };
        self.done_seq = seq + 1;
        let local = self.root.join("tmp").join("done-latest.json");
        if !self.remote.get(&key, &local)? {
            return Err(format!("{key} vanished"));
        }
        let doc: Value =
            serde_json::from_str(&fs::read_to_string(&local).map_err(|e| e.to_string())?)
                .map_err(|e| format!("{key}: {e}"))?;
        let mut seen = std::collections::BTreeSet::new();
        let mut redo = 0;
        for e in doc["entries"].as_array().cloned().unwrap_or_default() {
            let (Some(stream), Some(index)) = (e["stream"].as_str(), e["index"].as_u64()) else {
                continue;
            };
            let Some(k) = self.stream_index(stream) else {
                continue;
            };
            if !seen.insert((k, index)) {
                continue;
            }
            let mut ok = true;
            for o in e["outputs"].as_array().into_iter().flatten() {
                let key = o["path"].as_str().unwrap_or_default();
                let want = o["sha256"].as_str().unwrap_or_default();
                ok &= matches!(self.remote.sha256(key)?, Some((h, _)) if h == want);
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
            "runner: resuming from {key}: {} chunks done, {redo} to redo",
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
            serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        self.remote.put(&p, &key, true)?;
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
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let spec = json!({
            "stream": s.name, "index": index, "seed": seed, "kind": s.kind,
            "params": s.params, "threads": s.threads, "inputs": self.inputs,
            "blocks": s.blocks_in(index), "dir": dir.display().to_string(),
        });
        let sp = dir.join("spec.json");
        fs::write(&sp, spec.to_string()).map_err(|e| e.to_string())?;
        let logs = self.root.join("logs");
        fs::create_dir_all(&logs).map_err(|e| e.to_string())?;
        let log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(logs.join(format!("{}-{index:06}.log", s.name)))
            .map_err(|e| e.to_string())?;
        let log2 = log.try_clone().map_err(|e| e.to_string())?;
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
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
        })
    }

    /// Bundle, upload, verify, list one finished chunk.
    fn finish(&mut self, r: &Running) -> Result<(), String> {
        let s = &self.m.streams[r.stream];
        let name = s.name.clone();
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
        self.remote.put(&local, &key, true)?;
        match self.remote.sha256(&key)? {
            Some((h, b)) if h == sha && b == bytes => {}
            other => return Err(format!("upload check {key}: got {other:?}, want {sha}")),
        }
        kill_point("after_upload", n);
        let seconds = (r.started.elapsed().as_secs_f64() * 10.0).round() / 10.0;
        let mut e = json!({ "stream": name, "index": r.index, "seed": r.seed,
            "outputs": [{ "path": key, "sha256": sha, "bytes": bytes }], "seconds": seconds });
        if s.kind == "arena_blocks" {
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
        self.session.push(self.entries.len() - 1);
        self.dirty = true;
        let _ = fs::remove_dir_all(&r.dir);
        if self.last_done_write.elapsed().as_secs() >= self.m.upload_every_s {
            self.write_done()?;
        }
        Ok(())
    }

    fn schedule(&mut self) -> Result<(), String> {
        let mut running: Vec<Running> = Vec::new();
        let mut error: Option<String> = None;
        let mut stopped = false;
        loop {
            let now = unix_now();
            if self.m.deadline.is_some_and(|d| now >= d) && !stopped {
                eprintln!("runner: deadline reached; stopping running chunks");
                stopped = true;
                for r in &mut running {
                    stop_child(&mut r.child);
                }
                for r in running.drain(..) {
                    self.states[r.stream].running.remove(&r.index);
                }
            }
            // Start chunks while threads are free.
            if error.is_none() && !stopped {
                loop {
                    let used: usize = running
                        .iter()
                        .map(|r| self.m.streams[r.stream].threads)
                        .sum();
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
                    let r = self.spawn(k, i)?;
                    running.push(r);
                }
            }
            if running.is_empty() {
                break;
            }
            // Reap finished workers in a fixed order.
            let mut i = 0;
            while i < running.len() {
                match running[i].child.try_wait() {
                    Ok(None) => i += 1,
                    Ok(Some(st)) => {
                        let r = running.remove(i);
                        if st.success() && error.is_none() {
                            if let Err(e) = self.finish(&r) {
                                error = Some(e);
                            }
                        } else {
                            self.states[r.stream].running.remove(&r.index);
                            if error.is_none() {
                                error = Some(format!(
                                    "chunk {}/{} failed ({st}); see logs/{}-{:06}.log",
                                    self.m.streams[r.stream].name,
                                    r.index,
                                    self.m.streams[r.stream].name,
                                    r.index
                                ));
                            }
                        }
                    }
                    Err(e) => return Err(e.to_string()),
                }
            }
            if self.dirty && self.last_done_write.elapsed().as_secs() >= self.m.upload_every_s {
                self.write_done()?;
            }
            self.heartbeat(false, "chunks");
            std::thread::sleep(Duration::from_millis(100));
        }
        match error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn queue(&self) -> Value {
        let v: Vec<Value> = self
            .m
            .streams
            .iter()
            .zip(&self.states)
            .map(|(s, st)| {
                json!({ "stream": s.name, "done": st.done.len(), "running": st.running.len(),
                        "left": st.left(s), "decided": st.decided(s),
                        "median_s": kind_median(&self.m.streams, &self.states, s) })
            })
            .collect();
        json!(v)
    }

    fn heartbeat(&mut self, force: bool, phase: &str) {
        let every = Duration::from_secs(self.m.heartbeat_every_s);
        if !force && self.last_hb.is_some_and(|t| t.elapsed() < every) {
            return;
        }
        self.last_hb = Some(Instant::now());
        let cpu = self.cpu.sample(None);
        let round = |x: Option<f64>| x.map(|x| (x * 10.0).round() / 10.0);
        let elapsed = self.start.elapsed().as_secs_f64();
        let hb = json!({
            "run_id": self.m.run_id, "seq": self.hb_seq, "utc": utc_string(unix_now()),
            "phase": phase, "progress": { "queue": self.queue(), "entries": self.entries.len() },
            "elapsed_s": elapsed.round(), "threads": self.threads,
            "cpu_util_pct": round(cpu.cgroup), "cpu_util_source": "cgroup",
            "host_cpu_busy_pct": round(cpu.host),
            "est_cost_usd": (elapsed / 3600.0 * self.dph * 10_000.0).round() / 10_000.0,
        });
        let key = format!("{}/heartbeat/{:06}.json", self.prefix(), self.hb_seq);
        if let Err(e) = self.put_json("heartbeat.json", &key, &hb) {
            eprintln!("runner: heartbeat: {e}");
        }
        self.hb_seq += 1;
    }

    /// The merged report: depends only on which chunks are done and their bundles, never on
    /// timing or completion order.
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
                let chunks: Vec<Value> = es
                    .values()
                    .map(|e| json!({ "index": e["index"], "seed": e["seed"], "outputs": e["outputs"] }))
                    .collect();
                let mut v = json!({ "name": s.name, "kind": s.kind, "priority": s.priority,
                    "chunks_total": s.total(), "chunks_done": es.len(), "left": st.left(s),
                    "chunks": chunks });
                if let Some(sp) = &s.sprt {
                    let p = sprt_prefix(&st.scores, sp);
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
            "git_commit": option_env!("GIT_COMMIT").map(str::to_string).or_else(|| std::env::var("GIT_COMMIT").ok()),
            "bin_sha": std::env::var("BIN_SHA").ok(),
            "bin_build": std::env::var("BIN_BUILD").ok(),
            "image_digest": std::env::var("IMAGE_DIGEST").ok(),
            "runner_version": env!("CARGO_PKG_VERSION"),
            "threads": self.threads,
            "inputs": self.m.inputs.iter().map(|(n, h)| json!({ "name": n, "sha256": h })).collect::<Vec<_>>(),
            "chunk_seeds": seeds,
            "done_list_seq": self.done_seq.saturating_sub(1),
            "utc": utc_string(unix_now()),
        })
    }
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
