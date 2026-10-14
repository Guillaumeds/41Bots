//! `runner`: executes a run manifest on one machine.
//!
//! ```text
//! runner run <manifest.json> [--work-dir DIR]
//! runner hardware
//! ```
//!
//! **Format 1** (docs/M1-SPEC.md section 7; no `format` key): one chunk. Steps: record the
//! hardware, run the micro-benchmark, run the chunk while writing a heartbeat every 30 s, write
//! outputs then `DONE` (or `FAILED`), upload everything to R2 with rclone, and, when
//! `CONTAINER_ID` and `CONTAINER_API_KEY` are set, destroy this vast.ai instance last.
//! R2 layout under the manifest's `out` prefix (default `runs/<run_id>`):
//! `hardware.json`, `benchmark.json`, `heartbeat/<seq:06>.json` (versioned, never overwritten),
//! `out/...` (immutable), `result.json`, and `DONE` or `FAILED` uploaded last.
//! Uploads are skipped (with a notice) when `RCLONE_CONFIG_R2_TYPE` is unset, so the same
//! manifest runs on the laptop.
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
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

mod bundle;
mod measure;
mod remote;
mod sched;
mod spike;
mod v2;

const USAGE: &str = "usage:
  runner run <manifest.json> [--work-dir DIR]
  runner chunk <spec.json>        (internal: one format-2 chunk, started by `runner run`)
  runner hardware";

const HEARTBEAT_EVERY: Duration = Duration::from_secs(30);

fn main() -> ExitCode {
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
    remote: Option<String>,
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
}

impl Ctx {
    /// A context with no uploads and no deadline, rooted at `root` (format-2 chunk workers and
    /// the start-up benchmark); child logs go to `root/logs`, outside the bundled `root/out`.
    fn local(run_id: &str, root: &Path, threads: usize) -> Ctx {
        Ctx {
            run_id: run_id.to_string(),
            root: root.to_path_buf(),
            logs: root.join("logs"),
            remote: None,
            threads,
            dph: 0.0,
            start: Instant::now(),
            deadline: None,
            seq: 0,
            last_hb: None,
            cpu: CpuSampler::new(),
            child: None,
            phase: "chunk".into(),
            progress: Value::Null,
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

    /// `rclone copyto` one file to `<remote>/<rel>`; failures are logged, never fatal.
    fn upload_file(&self, local: &Path, rel: &str, immutable: bool) -> bool {
        let Some(remote) = &self.remote else {
            return true;
        };
        let mut c = Command::new("rclone");
        c.args(["copyto", "--s3-no-check-bucket", "--retries", "5"]);
        if immutable {
            c.arg("--immutable");
        }
        c.arg(local).arg(format!("{remote}/{rel}"));
        rclone(c)
    }

    fn heartbeat(&mut self, force: bool) {
        if !force && self.last_hb.is_some_and(|t| t.elapsed() < HEARTBEAT_EVERY) {
            return;
        }
        self.last_hb = Some(Instant::now());
        let cpu = self.cpu.sample(self.child);
        let round = |x: Option<f64>| x.map(|x| (x * 10.0).round() / 10.0);
        let elapsed = self.elapsed();
        let hb = json!({
            "run_id": self.run_id,
            "seq": self.seq,
            "utc": utc_string(unix_now()),
            "phase": self.phase,
            "progress": self.progress,
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
        match self.write_json("heartbeat.json", &hb) {
            Ok(p) => {
                let rel = format!("heartbeat/{:06}.json", self.seq);
                self.upload_file(&p, &rel, true);
            }
            Err(e) => eprintln!("runner: heartbeat: {e}"),
        }
        self.seq += 1;
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

fn rclone(mut c: Command) -> bool {
    match c.stdin(Stdio::null()).status() {
        Ok(s) if s.success() => true,
        Ok(s) => {
            eprintln!("runner: rclone failed ({s}): {c:?}");
            false
        }
        Err(e) => {
            eprintln!("runner: cannot run rclone: {e}");
            false
        }
    }
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
    let remote = match (
        std::env::var("RCLONE_CONFIG_R2_TYPE"),
        std::env::var("R2_BUCKET"),
    ) {
        (Ok(_), Ok(bucket)) if !bucket.is_empty() => Some(format!("r2:{bucket}/{out}")),
        _ => {
            println!("runner: RCLONE_CONFIG_R2_TYPE or R2_BUCKET unset: local run, no uploads");
            None
        }
    };
    let root = work.join(run_id);
    let mut ctx = Ctx {
        run_id: run_id.to_string(),
        logs: root.join("out"),
        root,
        remote,
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
        ctx.remote.as_deref().unwrap_or("(nowhere)")
    );

    let outcome = steps(&mut ctx, &manifest);
    let ok = outcome.is_ok();
    let marker = if ok { "DONE" } else { "FAILED" };
    let marker_body = json!({
        "utc": utc_string(unix_now()),
        "elapsed_s": ctx.elapsed().round(),
        "error": outcome.as_ref().err(),
    });
    ctx.set_phase(if ok { "uploading" } else { "failed" }, Value::Null);
    let _ = ctx.write_json(marker, &marker_body);
    let mut uploaded = true;
    if let Some(remote) = ctx.remote.clone() {
        for f in ["hardware.json", "benchmark.json", "result.json"] {
            let p = ctx.root.join(f);
            if p.exists() {
                uploaded &= ctx.upload_file(&p, f, false);
            }
        }
        let mut c = Command::new("rclone");
        c.args([
            "copy",
            "--s3-no-check-bucket",
            "--immutable",
            "--retries",
            "5",
        ])
        .arg(ctx.root.join("out"))
        .arg(format!("{remote}/out"));
        uploaded &= !ctx.root.join("out").exists() || rclone(c);
        let mut c = Command::new("rclone");
        c.args(["check", "--one-way"])
            .arg(ctx.root.join("out"))
            .arg(format!("{remote}/out"));
        uploaded &= !ctx.root.join("out").exists() || rclone(c);
        // The marker goes last: a DONE in R2 means every output is already there.
        let p = ctx.root.join(marker);
        uploaded &= ctx.upload_file(&p, marker, false);
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

fn self_destroy() {
    let (Ok(id), Ok(key)) = (
        std::env::var("CONTAINER_ID"),
        std::env::var("CONTAINER_API_KEY"),
    ) else {
        return;
    };
    if id.is_empty() || key.is_empty() {
        return;
    }
    println!("runner: destroying vast.ai instance {id}");
    let _ = std::io::stdout().flush();
    let status = Command::new("vastai")
        .args(["destroy", "instance", &id, "-y", "--api-key", &key])
        .stdin(Stdio::null())
        .status();
    if !matches!(status, Ok(s) if s.success()) {
        eprintln!("runner: self-destroy failed ({status:?}); the launcher will destroy it");
    }
}

fn steps(ctx: &mut Ctx, manifest: &Value) -> Result<(), String> {
    fs::create_dir_all(ctx.root.join("out")).map_err(|e| e.to_string())?;
    // (1) hardware
    let hw = ctx.write_json("hardware.json", &hardware())?;
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
    ctx.set_phase("chunk", json!({ "kind": kind }));
    let t0 = Instant::now();
    let mut result = run_kind(ctx, kind, params)?;
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
