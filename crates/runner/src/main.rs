//! `runner`: executes a run manifest on one machine.
//!
//! ```text
//! runner run <manifest.json> [--work-dir DIR]
//! runner hardware
//! ```
//!
//! **Format 1** (docs/M1-SPEC.md section 7; no `format` key): one chunk. Steps: record the
//! hardware, run the micro-benchmark, run the chunk while writing a heartbeat every
//! `heartbeat_every_s` (default 30), write outputs then `DONE` (or `FAILED`), upload everything to
//! R2 with rclone, and, on vast.ai (`CONTAINER_ID` set), ask vast.ai to destroy this instance last
//! (see [`self_destroy`]; the launcher destroys it too, and is the mechanism that does not depend
//! on this machine).
//! R2 layout under the manifest's `out` prefix (default `runs/<run_id>`):
//! `hardware.json`, `benchmark.json`, `heartbeat/<seq:06>.json` (versioned, never overwritten),
//! `partial/out/...` (mutable: the chunk's changed files every `upload_every_s`, default 300,
//! checkpoints first), `out/...` (immutable, at the end), `result.json`, and `DONE` or `FAILED`
//! uploaded last.
//! Uploads are skipped (with a notice) when `RCLONE_CONFIG_R2_TYPE` is unset, so the same
//! manifest runs on the laptop; `RUNNER_REMOTE_DIR` makes a local directory the remote (tests).
//! **The network never blocks the runner** (`net.rs`): the local `heartbeat.json` is written every
//! interval whatever the network does; its upload, the STOP check and the partial uploads run on
//! background threads, every rclone call has hard time limits (`remote.rs`), and the other
//! uploads run beside the heartbeat ([`Ctx::blocking`]). Heartbeats carry `upload_failures`,
//! `other_net_failures`, `last_upload_ok_s`, and, during a `search_measure` worlds step,
//! `progress.eta` (`measure.rs`). The machine's cost per solver node is followed in every
//! heartbeat and written to `machines/<run_id>.json` every 10 minutes and at the end
//! (`machine.rs`).
//! A stop request (SIGTERM/SIGINT/SIGHUP, the launcher's `<out>/STOP`, a local `STOP` file, or
//! the deadline) ends the chunk, uploads its checkpoints, then `out/` with whatever it finished,
//! and writes `FAILED` with `stopped` and the file list (`stop.rs`).
//! **Resume:** `"resume_from": "<run id>"` (in the manifest or the chunk's params; a
//! `search_measure` chunk only) downloads that run's latest `out/` (its final one, else its
//! `partial/out/`) before the chunk starts, so a replacement machine continues world by world;
//! give the new run its own `run_id`.
//!
//! **Format 2** (`"format": 2`, docs/M3-SPEC.md section 7): several prioritised streams of
//! chunks, a deadline, a versioned done-list and resume after a lost machine; see `v2.rs`.
//!
//! The chunk work itself is done by the sibling release binaries (`difftest`, `equity`,
//! `solver-spike`, `solver-identity`, `search`) found next to this executable; `arena_blocks`
//! chunks play in-process through `arena` and `search` (the bot registry).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

mod bundle;
mod machine;
mod measure;
mod mem;
mod net;
mod remote;
mod sched;
mod spike;
mod stop;
mod t1bench;
mod v2;

const USAGE: &str = "usage:
  runner run <manifest.json> [--work-dir DIR]
  runner chunk <spec.json>        (internal: one format-2 chunk, started by `runner run`)
  runner hardware";

/// The default heartbeat interval (format 1's `heartbeat_every_s`).
const HEARTBEAT_EVERY: Duration = Duration::from_secs(30);

/// Format 1's default interval between partial uploads of `out/` (`upload_every_s`).
const UPLOAD_EVERY: Duration = Duration::from_secs(300);

fn main() -> ExitCode {
    stop::install();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") if args.len() >= 2 => {
            let mut work = PathBuf::from("work");
            let mut it = args[2..].iter();
            while let Some(a) = it.next() {
                match (a.as_str(), it.next()) {
                    ("--work-dir", Some(v)) => work = PathBuf::from(v),
                    _ => {
                        eprintln!("{USAGE}");
                        return ExitCode::from(2);
                    }
                }
            }
            run(Path::new(&args[1]), &work)
        }
        Some("chunk") if args.len() == 2 => v2::worker(Path::new(&args[1])),
        Some("hardware") => {
            println!(
                "{}",
                serde_json::to_string_pretty(&hardware()).unwrap_or_default()
            );
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

// ---------------------------------------------------------------- time

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Days since 1970-01-01 → (year, month, day). Howard Hinnant's civil-from-days.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = i64::from((m + 9) % 12);
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn utc_string(secs: u64) -> String {
    let s = secs as i64;
    let (y, mo, d) = civil_from_days(s.div_euclid(86_400));
    let r = s.rem_euclid(86_400);
    format!(
        "{y:04}-{mo:02}-{d:02}T{:02}:{:02}:{:02}Z",
        r / 3600,
        r / 60 % 60,
        r % 60
    )
}

/// Parses `YYYY-MM-DDTHH:MM:SSZ` (UTC only).
fn parse_utc(s: &str) -> Option<u64> {
    let b = s.as_bytes();
    if b.len() != 20 || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[19] != b'Z' {
        return None;
    }
    let n = |r: std::ops::Range<usize>| s.get(r)?.parse::<u32>().ok();
    let (y, mo, d) = (n(0..4)?, n(5..7)?, n(8..10)?);
    let (h, mi, se) = (n(11..13)?, n(14..16)?, n(17..19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || se > 60 {
        return None;
    }
    let days = days_from_civil(i64::from(y), mo, d);
    u64::try_from(days * 86_400 + i64::from(h * 3600 + mi * 60 + se)).ok()
}

// ---------------------------------------------------------------- hardware and CPU use

fn read_trim(path: &str) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

fn hardware() -> Value {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let field = |name: &str| {
        cpuinfo
            .lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_once(':'))
            .map(|(_, v)| v.trim().to_string())
    };
    let flags = field("flags").unwrap_or_default();
    let logical = cpuinfo
        .lines()
        .filter(|l| l.starts_with("processor"))
        .count();
    let mem_total_kb = fs::read_to_string("/proc/meminfo").ok().and_then(|m| {
        m.lines()
            .find(|l| l.starts_with("MemTotal:"))
            .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
    });
    let gpu = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    json!({
        "utc": utc_string(unix_now()),
        "cpu_model": field("model name"),
        "cpu_logical_host": logical,
        "available_parallelism": std::thread::available_parallelism().map(|n| n.get()).ok(),
        "x86_64_v3": cpu_is_v3(&flags),
        "cpu_flags": flags,
        "cgroup_cpu_max": read_trim("/sys/fs/cgroup/cpu.max"),
        "cgroup_v1_cfs_quota_us": read_trim("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
        "cgroup_v1_cfs_period_us": read_trim("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
        "cgroup_quota_cores": cgroup_quota_cores(),
        "usable_cores": usable_cores(),
        "threads_default": default_threads(),
        "mem_total_kb": mem_total_kb,
        "nvidia_smi": gpu,
        "ncpu_env": std::env::var("NCPU").ok(),
        "dph_env": std::env::var("DPH").ok(),
        "vast_container_id": std::env::var("CONTAINER_ID").ok(),
        // Which destroy keys are present (never their values).
        "vast_destroy_keys": DESTROY_KEYS.iter()
            .filter(|k| std::env::var(k).is_ok_and(|v| !v.is_empty()))
            .collect::<Vec<_>>(),
        "bin_sha": std::env::var("BIN_SHA").ok(),
        "bin_build": std::env::var("BIN_BUILD").ok(),
    })
}

fn cpu_is_v3(flags: &str) -> bool {
    let have: Vec<&str> = flags.split_whitespace().collect();
    ["avx", "avx2", "bmi1", "bmi2", "fma", "f16c", "movbe", "abm"]
        .iter()
        .all(|f| have.contains(f))
}

/// Container CPU time in microseconds (cgroup v2, else v1).
fn cgroup_usage_us() -> Option<u64> {
    if let Ok(s) = fs::read_to_string("/sys/fs/cgroup/cpu.stat") {
        return s
            .lines()
            .find(|l| l.starts_with("usage_usec"))
            .and_then(|l| l.split_whitespace().nth(1)?.parse().ok());
    }
    read_trim("/sys/fs/cgroup/cpuacct/cpuacct.usage")?
        .parse::<u64>()
        .ok()
        .map(|ns| ns / 1000)
}

/// Host-wide (busy, total) jiffies from `/proc/stat`.
fn proc_stat() -> Option<(u64, u64)> {
    let s = fs::read_to_string("/proc/stat").ok()?;
    let line = s.lines().next()?;
    let v: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|x| x.parse().ok())
        .collect();
    let total: u64 = v.iter().take(8).sum();
    let idle = v.get(3).copied().unwrap_or(0) + v.get(4).copied().unwrap_or(0);
    Some((total - idle, total))
}

/// Parses a cgroup v2 `cpu.max` line ("max 100000" or "<quota> <period>") into cores.
fn parse_cpu_max(text: &str) -> Option<f64> {
    let mut it = text.split_whitespace();
    let quota = it.next()?;
    let period: f64 = it.next()?.parse().ok()?;
    if quota == "max" || period <= 0.0 {
        return None;
    }
    let q: f64 = quota.parse().ok()?;
    (q > 0.0).then(|| q / period)
}

/// The container's CPU quota in cores (cgroup v2 `cpu.max`, else v1 CFS quota), if limited.
/// vast.ai rents a fraction of a host this way, while `/proc/cpuinfo` and `nproc` show every
/// host core.
fn cgroup_quota_cores() -> Option<f64> {
    if let Some(t) = read_trim("/sys/fs/cgroup/cpu.max") {
        return parse_cpu_max(&t);
    }
    let q: f64 = read_trim("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")?
        .parse()
        .ok()?;
    let p: f64 = read_trim("/sys/fs/cgroup/cpu/cpu.cfs_period_us")?
        .parse()
        .ok()?;
    (q > 0.0 && p > 0.0).then(|| q / p)
}

/// Cores this container can keep busy: the CPUs it may run on (affinity), capped by the cgroup
/// quota. This is the denominator of `cpu_util_pct`.
fn usable_cores() -> f64 {
    let affinity = std::thread::available_parallelism().map_or(1, |n| n.get()) as f64;
    cgroup_quota_cores().map_or(affinity, |q| q.min(affinity))
}

/// Worker threads by default: the usable cores, rounded up so a fractional quota is used fully.
fn default_threads() -> usize {
    (usable_cores().ceil() as usize).max(1)
}

/// Clock ticks per second for `/proc/<pid>/stat` times.
fn clk_tck() -> f64 {
    Command::new("getconf")
        .arg("CLK_TCK")
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<f64>()
                .ok()
        })
        .filter(|&x| x > 0.0)
        .unwrap_or(100.0)
}

/// (ppid, utime + stime + cutime + cstime in ticks) from a `/proc/<pid>/stat` line.
fn parse_proc_stat(text: &str) -> Option<(u32, u64)> {
    let rest = &text[text.rfind(')')? + 1..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    // f[0] is field 3 (state), so field n is f[n - 3].
    let ppid = f.get(1)?.parse().ok()?;
    let mut t = 0u64;
    for i in 11..=14 {
        t += f.get(i)?.parse::<i64>().ok()?.max(0) as u64;
    }
    Some((ppid, t))
}

/// CPU ticks used by the process tree rooted at `root`: every live member's own time plus the
/// time of children it has already reaped (cutime/cstime), so exited helpers still count.
fn tree_ticks(root: u32) -> Option<u64> {
    let mut procs: Vec<(u32, u32, u64)> = Vec::new();
    for e in fs::read_dir("/proc").ok()?.flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|n| n.parse::<u32>().ok()) else {
            continue;
        };
        if let Some((ppid, t)) = fs::read_to_string(format!("/proc/{pid}/stat"))
            .ok()
            .as_deref()
            .and_then(parse_proc_stat)
        {
            procs.push((pid, ppid, t));
        }
    }
    procs.iter().find(|p| p.0 == root)?;
    let mut members = vec![root];
    let mut i = 0;
    while i < members.len() {
        let parent = members[i];
        members.extend(procs.iter().filter(|p| p.1 == parent).map(|p| p.0));
        i += 1;
    }
    Some(
        procs
            .iter()
            .filter(|p| members.contains(&p.0))
            .map(|p| p.2)
            .sum(),
    )
}

struct CpuSampler {
    at: Instant,
    cgroup: Option<u64>,
    stat: Option<(u64, u64)>,
    tree: Option<(u32, u64, Instant)>,
    tck: f64,
    cores: f64,
}

struct CpuSample {
    /// Chunk process tree CPU time over the interval / (interval x usable cores), percent.
    tree: Option<f64>,
    /// Same from the whole container's cgroup usage.
    cgroup: Option<f64>,
    /// Host-wide busy share from `/proc/stat` (all host cores; diagnostic only).
    host: Option<f64>,
}

impl CpuSampler {
    fn new() -> Self {
        CpuSampler {
            at: Instant::now(),
            cgroup: cgroup_usage_us(),
            stat: proc_stat(),
            tree: None,
            tck: clk_tck(),
            cores: usable_cores(),
        }
    }

    /// Starts the process-tree baseline for a newly started chunk, so its first heartbeat
    /// already measures the tree (the cgroup baseline is left alone).
    fn prime_tree(&mut self, pid: u32) {
        self.tree = tree_ticks(pid).map(|t| (pid, t, Instant::now()));
    }

    /// Cumulative-delta CPU use since the previous sample. `child` is the running chunk process.
    fn sample(&mut self, child: Option<u32>) -> CpuSample {
        let at = Instant::now();
        let cgroup = cgroup_usage_us();
        let stat = proc_stat();
        let tree = child.and_then(|pid| Some((pid, tree_ticks(pid)?, at)));
        let wall_s = at.duration_since(self.at).as_secs_f64();
        let cores = self.cores;
        let pct = move |used_s: f64, wall_s: f64| 100.0 * used_s / (wall_s * cores);
        let out = CpuSample {
            tree: match (self.tree, tree) {
                (Some((p0, t0, a0)), Some((p1, t1, a1))) if p0 == p1 && a1 > a0 => Some(pct(
                    t1.saturating_sub(t0) as f64 / self.tck,
                    a1.duration_since(a0).as_secs_f64(),
                )),
                _ => None,
            },
            cgroup: match (self.cgroup, cgroup) {
                (Some(a), Some(b)) if wall_s > 0.0 => {
                    Some(pct(b.saturating_sub(a) as f64 / 1e6, wall_s))
                }
                _ => None,
            },
            host: match (self.stat, stat) {
                (Some((b0, t0)), Some((b1, t1))) if t1 > t0 => {
                    Some(100.0 * (b1 - b0) as f64 / (t1 - t0) as f64)
                }
                _ => None,
            },
        };
        self.at = at;
        self.cgroup = cgroup;
        self.stat = stat;
        self.tree = tree;
        out
    }
}

// ---------------------------------------------------------------- run context

struct Ctx {
    run_id: String,
    root: PathBuf,
    /// Where child-process logs go (`out/` for format 1; outside the bundle for format 2).
    logs: PathBuf,
    /// Format 1's remote store (counting its failures) and its background network work, with
    /// the key prefix (`runs/<run_id>` unless the manifest says); `None` without a remote and in
    /// format-2 chunk workers (their orchestrator does their uploads).
    net: Option<net::Net>,
    prefix: String,
    threads: usize,
    dph: f64,
    start: Instant,
    deadline: Option<u64>,
    seq: u32,
    last_hb: Option<Instant>,
    cpu: CpuSampler,
    child: Option<u32>,
    phase: String,
    progress: Value,
    /// Print one compact line per heartbeat to stdout ([`hb_line`]; format 1, where stdout is
    /// the machine's log that `vastai logs` shows).
    hb_stdout: bool,
    /// Heartbeat interval (format 1's `heartbeat_every_s`, default 30 s).
    heartbeat_every: Duration,
    /// Interval of the background partial uploads of `out/` (format 1's `upload_every_s`,
    /// default 300 s), and when the last one started; off once the final uploads begin.
    upload_every: Duration,
    last_partial: Instant,
    partial_off: bool,
    /// This machine's solver work and cost per node (`machine.rs`).
    tally: machine::Tally,
}

impl Ctx {
    /// A context with no uploads and no deadline, rooted at `root` (format-2 chunk workers and
    /// the start-up benchmark); child logs go to `root/logs`, outside the bundled `root/out`.
    fn local(run_id: &str, root: &Path, threads: usize) -> Ctx {
        Ctx {
            run_id: run_id.to_string(),
            root: root.to_path_buf(),
            logs: root.join("logs"),
            net: None,
            prefix: format!("runs/{run_id}"),
            threads,
            // The machine's price, for a chunk's own cost projection (`progress.eta`).
            dph: std::env::var("DPH")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0),
            start: Instant::now(),
            deadline: None,
            seq: 0,
            last_hb: None,
            cpu: CpuSampler::new(),
            child: None,
            phase: "chunk".into(),
            progress: Value::Null,
            hb_stdout: false,
            heartbeat_every: HEARTBEAT_EVERY,
            upload_every: UPLOAD_EVERY,
            last_partial: Instant::now(),
            partial_off: true,
            tally: machine::Tally::default(),
        }
    }

    fn elapsed(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn write_json(&self, rel: &str, v: &Value) -> Result<PathBuf, String> {
        let p = self.root.join(rel);
        if let Some(d) = p.parent() {
            fs::create_dir_all(d).map_err(|e| format!("mkdir {}: {e}", d.display()))?;
        }
        let text = serde_json::to_string_pretty(v).map_err(|e| e.to_string())?;
        fs::write(&p, text + "\n").map_err(|e| format!("write {}: {e}", p.display()))?;
        Ok(p)
    }

    /// The remote store, if this run has one.
    fn remote(&self) -> Option<Arc<dyn remote::Remote>> {
        self.net.as_ref().map(|n| Arc::clone(&n.remote))
    }

    /// Runs `f` (a network call) on a helper thread while this one keeps heartbeating, so a slow
    /// upload or download never stops the heartbeat.
    fn blocking<T: Send>(&mut self, f: impl FnOnce() -> T + Send) -> T {
        std::thread::scope(|s| {
            let h = s.spawn(f);
            while !h.is_finished() {
                self.heartbeat(false);
                std::thread::sleep(Duration::from_millis(50));
            }
            h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))
        })
    }

    /// Uploads one file to `<prefix>/<rel>` (up to 3 tries, each with hard limits, heartbeating
    /// meanwhile); failures are logged, never fatal.
    fn upload_file(&mut self, local: &Path, rel: &str, immutable: bool) -> bool {
        let Some(r) = self.remote() else {
            return true;
        };
        let (local, key) = (local.to_path_buf(), format!("{}/{rel}", self.prefix));
        match self.blocking(move || remote::retry(3, || r.put(&local, &key, immutable))) {
            Ok(()) => true,
            Err(e) => {
                eprintln!("runner: upload {rel}: {e}");
                false
            }
        }
    }

    /// Every `upload_every`, uploads the changed files of `out/` to `<prefix>/partial/out/` in
    /// the background, checkpoints first (`net.rs`); skipped while the last one is still going.
    fn upload_partial(&mut self) {
        let Some(n) = &self.net else {
            return;
        };
        if self.partial_off || self.last_partial.elapsed() < self.upload_every || n.partial_busy() {
            return;
        }
        self.last_partial = Instant::now();
        n.partial(
            vec![(
                self.root.join("out"),
                format!("{}/partial/out", self.prefix),
            )],
            self.root.join("tmp").join("partial"),
        );
    }

    /// Ends the background partial uploads before the final ones: none starts again, and the
    /// one under way stops after its current file (it writes `partial/out/`, the final uploads
    /// `out/`, which a resumed run prefers, so they never need to wait for each other).
    fn stop_partial(&mut self) {
        self.partial_off = true;
        if let Some(n) = &self.net {
            n.cancel_partial();
        }
    }

    /// Writes `heartbeat.json` (every `heartbeat_every`, or now with `force`) and hands its
    /// upload to the background (`net.rs`): the network never holds this up. Also checks the
    /// local STOP file, starts a background check of the remote one, and starts a partial upload
    /// when one is due.
    fn heartbeat(&mut self, force: bool) {
        self.upload_partial();
        if !force
            && self
                .last_hb
                .is_some_and(|t| t.elapsed() < self.heartbeat_every)
        {
            return;
        }
        self.last_hb = Some(Instant::now());
        stop::poll_local(&self.root);
        if let Some(n) = &self.net {
            n.poll_stop(&self.root);
        }
        let cpu = self.cpu.sample(self.child);
        let round = |x: Option<f64>| x.map(|x| (x * 10.0).round() / 10.0);
        let elapsed = self.elapsed();
        // The machine's solver work so far, and its cost per node, added to the progress.
        self.tally.observe(&self.phase, &self.progress);
        let mut progress = self.progress.clone();
        if self.tally.active()
            && let Some(o) = progress.as_object_mut()
        {
            o.extend(self.tally.fields(elapsed, self.dph, self.threads));
        }
        let hb = json!({
            "run_id": self.run_id,
            "seq": self.seq,
            "utc": utc_string(unix_now()),
            "phase": self.phase,
            "progress": progress,
            "elapsed_s": elapsed.round(),
            "threads": self.threads,
            "usable_cores": (self.cpu.cores * 100.0).round() / 100.0,
            // Chunk process tree if one is running, else the container cgroup.
            "cpu_util_pct": round(cpu.tree.or(cpu.cgroup)),
            "cpu_util_source": if cpu.tree.is_some() { "process-tree" } else { "cgroup" },
            "cgroup_cpu_pct": round(cpu.cgroup),
            "host_cpu_busy_pct": round(cpu.host),
            "est_cost_usd": (elapsed / 3600.0 * self.dph * 10_000.0).round() / 10_000.0,
        });
        let mut hb = hb;
        if let Some(n) = &self.net {
            n.add_health(&mut hb);
        }
        if self.hb_stdout {
            println!(
                "{}",
                hb_line(
                    self.seq,
                    &self.phase,
                    &hb["progress"],
                    cpu.tree.or(cpu.cgroup)
                )
            );
            let _ = std::io::stdout().flush();
        }
        match self.write_json("heartbeat.json", &hb) {
            Ok(p) => {
                if let Some(n) = &self.net {
                    n.heartbeat(
                        fs::read_to_string(&p).unwrap_or_default(),
                        format!("{}/heartbeat/{:06}.json", self.prefix, self.seq),
                        self.root.join("tmp").join("heartbeat-upload.json"),
                    );
                }
            }
            Err(e) => eprintln!("runner: heartbeat: {e}"),
        }
        self.seq += 1;
        if self.tally.write_due() {
            self.write_machine("running", None, false);
        }
    }

    /// Writes the machine record (`machine.rs`) under the run root and to the remote store's
    /// `machines/` keys: in the background (`net.rs`) for the running record, or waiting for
    /// the upload (with retries, heartbeating meanwhile) for the final one.
    fn write_machine(&mut self, state: &str, outcome: Option<&str>, wait: bool) {
        let utc = utc_string(unix_now());
        let facts = machine::RunFacts {
            run_id: &self.run_id,
            utc: utc.clone(),
            elapsed_s: self.elapsed(),
            threads: self.threads,
            dph: self.dph,
        };
        let rec = machine::record(&self.tally, &facts, state, outcome);
        self.tally.written();
        let rel = format!("machines/{}.json", self.run_id);
        let local = match self.write_json(&rel, &rec) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("runner: machine record: {e}");
                return;
            }
        };
        let keys = machine::keys(&self.run_id, &utc);
        if !wait {
            if let Some(n) = &self.net {
                n.upload_text(
                    rec.to_string() + "\n",
                    keys.to_vec(),
                    self.root.join("tmp").join("machine-upload.json"),
                );
            }
            return;
        }
        let Some(r) = self.remote() else {
            return;
        };
        let failed = self.blocking(move || {
            keys.iter()
                .filter(|(k, immutable)| remote::retry(3, || r.put(&local, k, *immutable)).is_err())
                .map(|(k, _)| k.clone())
                .collect::<Vec<_>>()
        });
        for k in failed {
            eprintln!("runner: upload {k} failed");
        }
    }

    fn set_phase(&mut self, phase: &str, progress: Value) {
        self.phase = phase.to_string();
        self.progress = progress;
        self.heartbeat(true);
    }

    /// Runs a child process to completion, heartbeating meanwhile and killing it at the deadline.
    /// Its stdout and stderr are appended to `<logs>/<log>` (`out/` in format 1).
    fn run_child(&mut self, cmd: Command, log: &str) -> Result<(), String> {
        self.run_child_with(cmd, log, &mut |_| {})
    }

    /// [`Ctx::run_child`], calling `poll` every half second before the heartbeat check, so it can
    /// update `progress` from the child's own progress file.
    fn run_child_with(
        &mut self,
        mut cmd: Command,
        log: &str,
        poll: &mut dyn FnMut(&mut Ctx),
    ) -> Result<(), String> {
        if let Some(why) = stop::requested() {
            return Err(format!("stopped before {log}: {why}"));
        }
        let log_path = self.logs.join(log);
        if let Some(d) = log_path.parent() {
            fs::create_dir_all(d).map_err(|e| e.to_string())?;
        }
        let f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|e| format!("open {}: {e}", log_path.display()))?;
        let f2 = f.try_clone().map_err(|e| e.to_string())?;
        cmd.stdin(Stdio::null()).stdout(f).stderr(f2);
        cmd.env("RAYON_NUM_THREADS", self.threads.to_string());
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("cannot start {cmd:?}: {e}"))?;
        self.child = Some(child.id());
        self.cpu.prime_tree(child.id());
        let outcome = self.wait_child(&mut child, &cmd, log, poll);
        self.child = None;
        outcome
    }

    fn wait_child(
        &mut self,
        child: &mut std::process::Child,
        cmd: &Command,
        log: &str,
        poll: &mut dyn FnMut(&mut Ctx),
    ) -> Result<(), String> {
        loop {
            poll(self);
            if let Some(why) = stop::requested() {
                // Whatever the child finished is already in its files (they are written as each
                // piece finishes); the run uploads them next.
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("stopped: {why}"));
            }
            if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
                return if status.success() {
                    Ok(())
                } else {
                    Err(format!(
                        "{cmd:?} exited with {status} (see {})",
                        self.logs.join(log).display()
                    ))
                };
            }
            if self.deadline.is_some_and(|d| unix_now() >= d) {
                stop::request("the manifest's deadline_utc was reached");
                let _ = child.kill();
                let _ = child.wait();
                return Err("deadline reached; chunk killed".into());
            }
            if v2::orphaned() {
                let _ = child.kill();
                let _ = child.wait();
                std::process::exit(3);
            }
            self.heartbeat(false);
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}

/// The compact heartbeat line, e.g. `hb 17 fill 95s cpu 98.7% busy 245/245 nodes 3.1e11
/// rate 3.3e9/s` (a `t1bench` window), or `hb 3 chunk search_measure cpu 99.2%`.
fn hb_line(seq: u32, phase: &str, progress: &Value, cpu: Option<f64>) -> String {
    let mut s = format!("hb {seq}");
    match progress.get("window").and_then(Value::as_str) {
        Some(w) if progress["started"].as_bool() == Some(true) => {
            s.push_str(&format!(
                " {w} {:.0}s",
                progress["t_s"].as_f64().unwrap_or(0.0)
            ));
        }
        Some(w) => s.push_str(&format!(" {w} starting")),
        None => {
            s.push(' ');
            s.push_str(phase);
            for k in ["kind", "step"] {
                if let Some(x) = progress.get(k).and_then(Value::as_str) {
                    s.push(' ');
                    s.push_str(x);
                }
            }
        }
    }
    match cpu {
        Some(c) => s.push_str(&format!(" cpu {c:.1}%")),
        None => s.push_str(" cpu -"),
    }
    if let (Some(b), Some(t)) = (progress["busy"].as_u64(), progress["threads"].as_u64()) {
        s.push_str(&format!(" busy {b}/{t}"));
    }
    if let Some(n) = progress["nodes"].as_f64() {
        s.push_str(&format!(" nodes {n:.1e}"));
    }
    if let Some(r) = progress["rate"].as_f64() {
        s.push_str(&format!(" rate {r:.1e}/s"));
    }
    if let Some(c) = progress["usd_per_1e12_nodes"].as_f64() {
        s.push_str(&format!(" $/1e12 {c:.3}"));
        if progress["settling"].as_bool() == Some(true) {
            s.push_str(" (settling)");
        }
    }
    s
}

fn sibling(name: &str) -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| {
            p.parent()
                .map(|d| d.join(format!("{name}{}", std::env::consts::EXE_SUFFIX)))
        })
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from(name))
}

fn num(v: &Value, key: &str) -> Result<u64, String> {
    v.get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("chunk needs a whole-number `{key}`"))
}

// ---------------------------------------------------------------- the run

fn run(manifest_path: &Path, work: &Path) -> ExitCode {
    let manifest: Value = match fs::read_to_string(manifest_path)
        .map_err(|e| e.to_string())
        .and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string()))
    {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "runner: cannot read manifest {}: {e}",
                manifest_path.display()
            );
            return ExitCode::from(2);
        }
    };
    match sched::format_of(&manifest) {
        Ok(1) => {}
        Ok(_) => return v2::run(&manifest, work),
        Err(e) => {
            eprintln!("runner: {e}");
            return ExitCode::from(2);
        }
    }
    let Some(run_id) = manifest.get("run_id").and_then(Value::as_str) else {
        eprintln!("runner: manifest needs a `run_id`");
        return ExitCode::from(2);
    };
    if run_id.is_empty()
        || !run_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
    {
        eprintln!("runner: run_id may only use letters, digits, - _ .");
        return ExitCode::from(2);
    }
    // Threads: the manifest's explicit value, else the usable cores from the cgroup quota (not
    // NCPU or nproc, which can exceed the quota and oversubscribe).
    let threads = manifest
        .get("threads")
        .and_then(Value::as_u64)
        .map(|t| t as usize)
        .filter(|&t| t > 0)
        .unwrap_or_else(default_threads);
    let deadline = match manifest.get("deadline_utc").and_then(Value::as_str) {
        None => None,
        Some(s) => match parse_utc(s) {
            Some(t) => Some(t),
            None => {
                eprintln!("runner: deadline_utc must look like 2026-10-13T06:00:00Z");
                return ExitCode::from(2);
            }
        },
    };
    let out = manifest
        .get("out")
        .and_then(Value::as_str)
        .map(|s| s.trim_matches('/').to_string())
        .unwrap_or_else(|| format!("runs/{run_id}"));
    let secs = |key: &str, default: Duration| -> Result<Duration, String> {
        match manifest.get(key) {
            None | Some(Value::Null) => Ok(default),
            Some(v) => v
                .as_u64()
                .map(Duration::from_secs)
                .ok_or_else(|| format!("{key} must be a whole number of seconds")),
        }
    };
    let (heartbeat_every, upload_every) = match (
        secs("heartbeat_every_s", HEARTBEAT_EVERY),
        secs("upload_every_s", UPLOAD_EVERY),
    ) {
        (Ok(h), Ok(u)) => (h.max(Duration::from_secs(1)), u),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("runner: {e}");
            return ExitCode::from(2);
        }
    };
    let net = remote::from_env().map(|r| net::Net::new(r, &out));
    if net.is_none() {
        println!(
            "runner: RCLONE_CONFIG_R2_TYPE or R2_BUCKET unset (and no RUNNER_REMOTE_DIR): local \
             run, no uploads"
        );
    }
    let root = work.join(run_id);
    let mut ctx = Ctx {
        run_id: run_id.to_string(),
        logs: root.join("out"),
        root,
        net,
        prefix: out,
        threads,
        dph: std::env::var("DPH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0),
        start: Instant::now(),
        deadline,
        seq: 0,
        last_hb: None,
        cpu: CpuSampler::new(),
        child: None,
        phase: "start".into(),
        progress: Value::Null,
        hb_stdout: true,
        heartbeat_every,
        upload_every,
        last_partial: Instant::now(),
        partial_off: false,
        tally: machine::Tally::default(),
    };
    // Continue a heartbeat sequence if this run id was started before on this disk.
    if let Ok(t) = fs::read_to_string(ctx.root.join("heartbeat.json"))
        && let Ok(v) = serde_json::from_str::<Value>(&t)
        && let Some(s) = v.get("seq").and_then(Value::as_u64)
    {
        ctx.seq = s as u32 + 1;
    }
    println!(
        "runner: run {run_id}, {threads} threads ({:.2} usable cores), uploading to {}",
        ctx.cpu.cores,
        ctx.remote().map_or("(nowhere)".to_string(), |r| format!(
            "{}/{}",
            r.describe(),
            ctx.prefix
        ))
    );

    let outcome = steps(&mut ctx, &manifest);
    let ok = outcome.is_ok();
    let marker = if ok { "DONE" } else { "FAILED" };
    // The background partial uploads end here; the final uploads follow.
    ctx.stop_partial();
    // A stopped run (signal, the launcher's STOP, the deadline) keeps everything it finished:
    // its checkpoints and the whole out/ are uploaded below, and the marker lists them.
    let stopped = stop::requested();
    // The machine's final cost record (`machine.rs`), once it counted any solver work.
    ctx.set_phase("machine-record", Value::Null);
    if ctx.tally.active() {
        let outcome_word = if ok {
            "DONE"
        } else if stopped.is_some() {
            "STOPPED"
        } else {
            "FAILED"
        };
        ctx.write_machine("final", Some(outcome_word), true);
    }
    let outputs: Vec<Value> = bundle::list_files(&ctx.root.join("out"))
        .unwrap_or_default()
        .into_iter()
        .map(|rel| {
            let n = fs::metadata(ctx.root.join("out").join(&rel)).map_or(0, |m| m.len());
            json!({ "path": format!("out/{rel}"), "bytes": n })
        })
        .collect();
    let marker_body = json!({
        "utc": utc_string(unix_now()),
        "elapsed_s": ctx.elapsed().round(),
        "error": outcome.as_ref().err(),
        "stopped": stopped,
        "partial": !ok,
        "outputs": outputs,
    });
    ctx.set_phase(
        if ok {
            "uploading"
        } else if stopped.is_some() {
            "stopped"
        } else {
            "failed"
        },
        Value::Null,
    );
    let _ = ctx.write_json(marker, &marker_body);
    let mut uploaded = true;
    if let Some(r) = ctx.remote() {
        let (root, prefix) = (ctx.root.clone(), ctx.prefix.clone());
        uploaded = ctx.blocking(move || final_uploads(&*r, &root, &prefix, marker));
        ctx.heartbeat(true);
        if let Some(n) = &ctx.net {
            n.drain(Duration::from_secs(10));
        }
    }
    match &outcome {
        Ok(()) => println!("runner: {marker}"),
        Err(e) => eprintln!("runner: FAILED: {e}"),
    }
    if !uploaded {
        eprintln!("runner: some uploads failed");
    }
    self_destroy();
    if ok && uploaded {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Format 1's last uploads, most valuable first: the checkpoints of `out/` (a stopped
/// `search_measure` chunk's worlds finished since its last complete decision: small, and what a
/// resumed run needs), the run's records, the whole `out/` (immutable keys, checked), and the
/// marker last, so a `DONE` or `FAILED` in the remote means everything else is there. Each upload
/// is tried up to 3 times, each try with hard limits (`remote.rs`).
fn final_uploads(r: &dyn remote::Remote, root: &Path, prefix: &str, marker: &str) -> bool {
    let put = |local: &Path, key: String, immutable: bool| -> bool {
        match remote::retry(3, || r.put(local, &key, immutable)) {
            Ok(()) => true,
            Err(e) => {
                eprintln!("runner: upload {key}: {e}");
                false
            }
        }
    };
    let mut ok = true;
    let out = root.join("out");
    for rel in bundle::list_files(&out).unwrap_or_default() {
        if net::is_ckpt(&rel) {
            ok &= put(&out.join(&rel), format!("{prefix}/out/{rel}"), true);
        }
    }
    for f in ["hardware.json", "benchmark.json", "result.json"] {
        let p = root.join(f);
        if p.exists() {
            ok &= put(&p, format!("{prefix}/{f}"), false);
        }
    }
    if let Err(e) = remote::retry(3, || r.put_dir(&out, &format!("{prefix}/out"), true)) {
        eprintln!("runner: upload out/: {e}");
        ok = false;
    }
    ok & put(&root.join(marker), format!("{prefix}/{marker}"), false)
}

/// The keys tried to destroy this instance, in order: the per-instance key vast.ai injects
/// (documented as able to start, stop or destroy only this instance), then the destroy-only key
/// the launcher creates for the run and passes with `--env` (`python/launcher/launch.py`).
const DESTROY_KEYS: [&str; 2] = ["CONTAINER_API_KEY", "VAST_DESTROY_KEY"];

/// Asks vast.ai to destroy this instance, as the very last step. Nothing to do off vast.ai
/// (`CONTAINER_ID` unset). Each key goes to `vastai` through its `VAST_API_KEY` environment
/// variable, never on a command line, and only the API's error line is printed (`vastai`'s own
/// hints print a key's last characters). In the runs of 2026-10-14 vast.ai answered the injected
/// key with `401 Invalid user key`; the launcher's key, then the launcher's own destroy (on
/// `DONE`/`FAILED`, a stale heartbeat or its time limit) and `vast-cleanup.yml` cover that.
fn self_destroy() {
    let Some(id) = std::env::var("CONTAINER_ID").ok().filter(|v| !v.is_empty()) else {
        return;
    };
    println!("runner: destroying vast.ai instance {id}");
    let _ = std::io::stdout().flush();
    let mut tried = false;
    for name in DESTROY_KEYS {
        let Some(key) = std::env::var(name).ok().filter(|v| !v.is_empty()) else {
            continue;
        };
        tried = true;
        let out = Command::new("vastai")
            .args(["destroy", "instance", &id, "-y"])
            .env("VAST_API_KEY", &key)
            .env("VASTAI_NO_UPDATE_CHECK", "1")
            .stdin(Stdio::null())
            .output();
        let outcome = match out {
            Ok(o) => destroy_outcome(
                &format!(
                    "{}\n{}",
                    String::from_utf8_lossy(&o.stdout),
                    String::from_utf8_lossy(&o.stderr)
                ),
                &id,
                &key,
            ),
            Err(e) => Err(format!("cannot run vastai: {e}")),
        };
        match outcome {
            Ok(()) => {
                println!("runner: vast.ai accepted the destroy of instance {id} ({name})");
                let _ = std::io::stdout().flush();
                return;
            }
            Err(why) => eprintln!("runner: self-destroy with {name} refused: {why}"),
        }
    }
    if !tried {
        eprintln!("runner: no destroy key set ({})", DESTROY_KEYS.join(", "));
    }
    eprintln!(
        "runner: instance {id} could not destroy itself; the launcher destroys it when it sees \
         DONE or FAILED (python/launcher/launch.py), and vast-cleanup.yml destroys any 41bots- \
         instance left after 2 hours. With neither running, destroy it in the vast.ai console."
    );
}

/// Whether `vastai destroy instance <id>` succeeded, from what it printed: vastai 1.8.3 exits
/// with 0 even when the API refuses, and prints `destroying instance <id>.` only on success.
/// A refusal is reduced to the API's error line, with `key` masked should it appear.
fn destroy_outcome(text: &str, id: &str, key: &str) -> Result<(), String> {
    let ok = format!("destroying instance {id}.");
    if text.lines().any(|l| l.trim() == ok) {
        return Ok(());
    }
    let why = text
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("Failed with error"))
        .or_else(|| text.lines().map(str::trim).find(|l| !l.is_empty()))
        .unwrap_or("vastai printed nothing");
    let mut why = why.chars().take(200).collect::<String>();
    if !key.is_empty() {
        why = why.replace(key, "***");
    }
    // vastai's hint lines name a key's last characters; never pass one on.
    if why.contains("ends in") {
        why = "vastai refused (details withheld: they name the key)".into();
    }
    Err(why)
}

fn steps(ctx: &mut Ctx, manifest: &Value) -> Result<(), String> {
    fs::create_dir_all(ctx.root.join("out")).map_err(|e| e.to_string())?;
    // (1) hardware
    let hw_rec = hardware();
    ctx.tally.hardware(&hw_rec);
    let hw = ctx.write_json("hardware.json", &hw_rec)?;
    ctx.upload_file(&hw, "hardware.json", false);
    ctx.set_phase("benchmark", Value::Null);
    // (2) benchmark: difftest matches per second per core
    let secs = manifest
        .get("benchmark_seconds")
        .and_then(Value::as_u64)
        .unwrap_or(60);
    if secs > 0 {
        let b = benchmark(ctx, secs)?;
        let p = ctx.write_json("benchmark.json", &b)?;
        ctx.upload_file(&p, "benchmark.json", false);
    }
    // (3) the chunk
    let chunk = manifest.get("chunk").ok_or("manifest needs a `chunk`")?;
    let (kind, params) = chunk
        .as_object()
        .filter(|o| o.len() == 1)
        .and_then(|o| o.iter().next())
        .ok_or("`chunk` must be an object with exactly one kind, e.g. {\"difftest\": {...}}")?;
    let resumed = match resume_from(manifest, params)? {
        None => None,
        Some(_) if kind != "search_measure" => {
            return Err(format!(
                "resume_from works with search_measure chunks only (this chunk is {kind})"
            ));
        }
        Some(from) => Some(resume_out(ctx, &from)?),
    };
    ctx.set_phase("chunk", json!({ "kind": kind }));
    let t0 = Instant::now();
    let mut result = run_kind(ctx, kind, params)?;
    if let (Some(o), Some(r)) = (result.as_object_mut(), resumed) {
        o.insert("resumed_from".into(), r);
    }
    let secs = t0.elapsed().as_secs_f64();
    if let Some(o) = result.as_object_mut() {
        o.insert("kind".into(), json!(kind));
        o.insert("seconds".into(), json!((secs * 10.0).round() / 10.0));
        o.insert("threads".into(), json!(ctx.threads));
        o.insert("dph".into(), json!(ctx.dph));
    }
    ctx.write_json("result.json", &result)?;
    Ok(())
}

/// The earlier run a format-1 chunk continues: `resume_from` in the manifest or the chunk's
/// params, a run id (its files are under `runs/<id>`) or a key prefix containing `/`.
fn resume_from(manifest: &Value, params: &Value) -> Result<Option<String>, String> {
    let v = manifest
        .get("resume_from")
        .or_else(|| params.get("resume_from"));
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.contains('/') => Ok(Some(s.trim_matches('/').to_string())),
        Some(Value::String(s))
            if !s.is_empty()
                && s.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) =>
        {
            Ok(Some(format!("runs/{s}")))
        }
        Some(_) => Err("resume_from must be a run id (or a key prefix with /)".into()),
    }
}

/// `resume_from`: downloads the latest copy of an earlier run's `out/` into this run's `out/`
/// before the chunk starts: its final `out/` where it got that far (uploaded at its end or stop),
/// else its partial uploads (`partial/out/`, every `upload_every_s`). A `search_measure` chunk
/// then continues from its worlds file and checkpoint, redoing only the worlds that were being
/// solved (or finished after the last upload). Local files at least as large are kept. A store
/// that cannot be listed, or a run with nothing under either prefix, is an error: a resume that
/// restored nothing would redo the whole run in silence.
fn resume_out(ctx: &mut Ctx, from: &str) -> Result<Value, String> {
    let r = ctx
        .remote()
        .ok_or("resume_from needs a remote store (R2, or RUNNER_REMOTE_DIR)")?;
    ctx.set_phase("resume", json!({ "from": from }));
    let out = ctx.root.join("out");
    let prefixes = vec![format!("{from}/out"), format!("{from}/partial/out")];
    let got = ctx
        .blocking(move || net::restore(&*r, &prefixes, &out))
        .map_err(|e| format!("resume_from {from}: {e}"))?;
    if got.listed == 0 {
        return Err(format!(
            "resume_from {from}: nothing to resume under {from}/out/ or {from}/partial/out/ \
             (is the run id right? did it upload anything?)"
        ));
    }
    println!(
        "runner: resuming from {from}: {} files listed, {} files ({} bytes) restored into out/",
        got.listed, got.files, got.bytes
    );
    Ok(json!({ "from": from, "listed": got.listed, "files": got.files, "bytes": got.bytes }))
}

/// Runs one chunk of an existing (format-1) kind; also used by format-2 workers.
fn run_kind(ctx: &mut Ctx, kind: &str, params: &Value) -> Result<Value, String> {
    Ok(match kind {
        "difftest" => {
            let matches = num(params, "matches")?;
            let seed = num(params, "seed")?;
            let mut c = Command::new(sibling("difftest"));
            c.arg(matches.to_string())
                .arg(seed.to_string())
                .args(["--threads", &ctx.threads.to_string()]);
            ctx.run_child(c, "difftest.log")?;
            json!({ "matches": matches, "seed": seed })
        }
        "equity_sweep" => {
            // Smoke-testable form: estimate the hand-outcome model, then optionally build the table.
            let bot = params
                .get("bot")
                .and_then(Value::as_str)
                .unwrap_or("rules-v1");
            let hands = num(params, "hands")?;
            let seed = num(params, "seed")?;
            let model = ctx.root.join("out").join("outcome.json");
            let mut c = Command::new(sibling("equity"));
            c.args([
                "estimate-model",
                "--bot",
                bot,
                "--hands",
                &hands.to_string(),
            ])
            .args([
                "--seed",
                &seed.to_string(),
                "--threads",
                &ctx.threads.to_string(),
            ])
            .arg("--out")
            .arg(&model);
            ctx.run_child(c, "equity.log")?;
            if params
                .get("build")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                ctx.set_phase("chunk", json!({ "kind": kind, "step": "build" }));
                let table = ctx.root.join("out").join("equity-v0.npy");
                let mut c = Command::new(sibling("equity"));
                c.arg("build")
                    .arg("--model")
                    .arg(&model)
                    .arg("--out")
                    .arg(&table)
                    .args(["--threads", &ctx.threads.to_string()]);
                ctx.run_child(c, "equity.log")?;
            }
            json!({ "hands": hands, "seed": seed, "bot": bot })
        }
        "benchmark" => {
            let s = num(params, "seconds")?;
            benchmark(ctx, s)?
        }
        "solver_spike" => spike::chunk(ctx, params)?,
        "search_measure" => measure::chunk(ctx, params)?,
        "t1bench" => t1bench::chunk(ctx, params)?,
        "arena_block" => {
            return Err("arena_block chunks need the arena binary (not built yet)".into());
        }
        other => return Err(format!("unknown chunk kind `{other}`")),
    })
}

/// Runs growing `difftest` batches for about `secs` seconds on all threads.
fn benchmark(ctx: &mut Ctx, secs: u64) -> Result<Value, String> {
    let t0 = Instant::now();
    let mut batch: u64 = 64 * ctx.threads as u64;
    let mut total = 0u64;
    let mut seed = 1u64;
    while t0.elapsed().as_secs() < secs {
        let t = Instant::now();
        let mut c = Command::new(sibling("difftest"));
        c.arg(batch.to_string())
            .arg(seed.to_string())
            .args(["--threads", &ctx.threads.to_string()]);
        ctx.run_child(c, "benchmark.log")?;
        total += batch;
        seed += 1;
        if t.elapsed().as_secs_f64() < secs as f64 / 6.0 {
            batch *= 2;
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let per_sec = total as f64 / elapsed;
    let cpu_model = hardware()["cpu_model"].clone();
    Ok(json!({
        "utc": utc_string(unix_now()),
        "cpu_model": cpu_model,
        "threads": ctx.threads,
        "matches": total,
        "seconds": (elapsed * 10.0).round() / 10.0,
        "matches_per_sec": (per_sec * 10.0).round() / 10.0,
        "matches_per_sec_per_core": (per_sec / ctx.threads as f64 * 100.0).round() / 100.0,
        "dph": ctx.dph,
        "matches_per_dollar": if ctx.dph > 0.0 { Some((per_sec * 3600.0 / ctx.dph).round()) } else { None },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utc_round_trip() {
        for t in [0u64, 951_782_400, 1_760_313_600, 4_102_444_799] {
            assert_eq!(parse_utc(&utc_string(t)), Some(t));
        }
        assert_eq!(utc_string(1_760_313_600), "2025-10-13T00:00:00Z");
        assert_eq!(parse_utc("2026-10-13 06:00:00Z"), None);
        assert_eq!(parse_utc("2026-13-01T00:00:00Z"), None);
    }

    #[test]
    fn v3_flags() {
        assert!(cpu_is_v3(
            "fpu avx avx2 bmi1 bmi2 fma f16c movbe abm sse4_2"
        ));
        assert!(!cpu_is_v3("fpu avx sse4_2 popcnt"));
    }

    #[test]
    fn cgroup_quota_parses() {
        assert_eq!(parse_cpu_max("max 100000"), None);
        assert_eq!(parse_cpu_max("4800000 100000"), Some(48.0));
        assert_eq!(parse_cpu_max("150000 100000"), Some(1.5));
        assert_eq!(parse_cpu_max("garbage"), None);
    }

    #[test]
    fn destroy_outcome_reads_vastai() {
        let key = "0123456789abcdef0123456789abcdef";
        assert_eq!(
            destroy_outcome("destroying instance 4242.\n\n", "4242", key),
            Ok(())
        );
        // vastai 1.8.3 refusing the injected key (with exit status 0).
        let refused = format!(
            "\nFailed with error 401: Invalid user key\n  Sent key from $VAST_API_KEY (ends in ...{}). Update the env var with a new key.\n",
            &key[28..]
        );
        let e = destroy_outcome(&refused, "4242", key).expect_err("refused");
        assert_eq!(e, "Failed with error 401: Invalid user key");
        // Another instance's confirmation is not this one's.
        assert!(destroy_outcome("destroying instance 42.", "4242", key).is_err());
        // Unknown output: its first line, the key masked, never a key suffix.
        let e = destroy_outcome(&format!("odd {key} answer"), "4242", key).expect_err("odd");
        assert_eq!(e, "odd *** answer");
        let e = destroy_outcome("Sent key (ends in ...cdef)", "4242", key).expect_err("hint");
        assert!(!e.contains("cdef"), "{e}");
        assert_eq!(
            destroy_outcome("", "4242", key),
            Err("vastai printed nothing".into())
        );
    }

    #[test]
    fn heartbeat_lines() {
        let p = json!({ "kind": "t1bench", "window": "fill", "started": true, "t_s": 95.2,
                        "threads": 245, "busy": 245, "nodes": 3.1e11, "rate": 3.26e9 });
        assert_eq!(
            hb_line(17, "chunk", &p, Some(98.66)),
            "hb 17 fill 95s cpu 98.7% busy 245/245 nodes 3.1e11 rate 3.3e9/s"
        );
        let p = json!({ "kind": "t1bench", "window": "prod", "started": false, "threads": 4 });
        assert_eq!(hb_line(2, "chunk", &p, None), "hb 2 prod starting cpu -");
        let p = json!({ "kind": "search_measure", "step": "exact worlds" });
        assert_eq!(
            hb_line(3, "chunk", &p, Some(99.24)),
            "hb 3 chunk search_measure exact worlds cpu 99.2%"
        );
        assert_eq!(hb_line(0, "start", &Value::Null, None), "hb 0 start cpu -");
        let p = json!({ "kind": "search_measure", "step": "exact worlds",
                        "usd_per_1e12_nodes": 0.3124, "settling": true });
        assert_eq!(
            hb_line(4, "chunk", &p, Some(99.0)),
            "hb 4 chunk search_measure exact worlds cpu 99.0% $/1e12 0.312 (settling)"
        );
    }

    /// The network never blocks the runner (docs/M3-SPEC.md §7.3). With an rclone that hangs on
    /// every call (a fake that sleeps 300 s), while a chunk (`sleep 4`) runs: the local heartbeat
    /// is still written every interval, each hung call (heartbeat uploads, STOP checks, partial
    /// uploads) is killed at its 1 s limit, `upload_failures` rises, and the chunk ends on time.
    #[cfg(unix)]
    #[test]
    fn hung_network_never_blocks() {
        use std::os::unix::fs::PermissionsExt;
        let base = std::env::temp_dir().join(format!("runner-hang-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).expect("mkdir");
        let log = base.join("calls.log");
        let fake = base.join("rclone");
        fs::write(
            &fake,
            format!(
                "#!/bin/sh\necho \"$$ $1\" >> '{}'\nexec sleep 300\n",
                log.display()
            ),
        )
        .expect("write");
        fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).expect("chmod");
        let raw = remote::RcloneRemote {
            base: "r2:bucket".into(),
            bin: fake,
            limit: Duration::from_secs(1),
        };
        let root = base.join("work");
        fs::create_dir_all(root.join("out")).expect("mkdir");
        fs::write(root.join("out/w.jsonl.ckpt"), "1\n").expect("write");
        let mut ctx = Ctx::local("hang", &root, 1);
        ctx.net = Some(net::Net::new(Box::new(raw), "runs/hang"));
        ctx.heartbeat_every = Duration::from_millis(200);
        ctx.upload_every = Duration::from_millis(500);
        ctx.partial_off = false;
        let t0 = Instant::now();
        let mut c = Command::new("sleep");
        c.arg("4");
        ctx.run_child(c, "sleep.log").expect("the chunk completes");
        let took = t0.elapsed();
        assert!(took < Duration::from_secs(6), "the chunk took {took:?}");
        assert!(ctx.seq >= 8, "{} local heartbeats in {took:?}", ctx.seq);
        let hb: Value =
            serde_json::from_str(&fs::read_to_string(root.join("heartbeat.json")).expect("hb"))
                .expect("json");
        assert!(hb["upload_failures"].as_u64().expect("count") >= 2, "{hb}");
        assert!(
            hb["other_net_failures"].as_u64().expect("count") >= 1,
            "{hb}"
        );
        assert_eq!(hb["last_upload_ok_s"], Value::Null, "{hb}");
        // Every hung call was killed within its limit: none of those started so far is left
        // 1.5 s later (the background lanes go on starting new ones; they are not counted).
        let calls = fs::read_to_string(&log).expect("calls");
        std::thread::sleep(Duration::from_millis(1500));
        let kinds: std::collections::BTreeSet<&str> = calls
            .lines()
            .filter_map(|l| l.split_whitespace().nth(1))
            .collect();
        assert!(kinds.contains("copyto") && kinds.contains("lsf"), "{calls}");
        eprintln!(
            "hung network: chunk {took:?}, {} local heartbeats, {} hung rclone calls killed, \
             upload_failures {}, other_net_failures {}",
            ctx.seq,
            calls.lines().count(),
            hb["upload_failures"],
            hb["other_net_failures"]
        );
        for pid in calls.lines().filter_map(|l| l.split_whitespace().next()) {
            let cmd = fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
            assert!(
                !String::from_utf8_lossy(&cmd).contains("300"),
                "hung rclone {pid} is still running"
            );
        }
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn proc_stat_parses() {
        let line =
            "123 (my (odd) name) R 45 123 123 0 -1 4194304 1 0 0 0 700 50 9 1 20 0 5 0 1 2 3";
        assert_eq!(parse_proc_stat(line), Some((45, 760)));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn tree_ticks_reads_own_process() {
        assert!(tree_ticks(std::process::id()).is_some());
    }
}
