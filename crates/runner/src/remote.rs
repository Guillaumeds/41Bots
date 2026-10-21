//! The remote store behind one trait: R2 through rclone on rented machines, a local directory on
//! the laptop and in the `resume_identity` test. Keys are bucket-relative `/` paths
//! (`runs/<run>/...`, `done/<run>/...`, `inputs/sha256/<hex>`).
//!
//! Every write is whole-object: S3/R2 objects appear atomically, and the directory backend writes
//! a temporary file then renames it, so a kill mid-upload never leaves a partial object.
//! Immutable writes accept an identical existing object (a redone chunk reproduces its bundle
//! byte for byte) and refuse a different one.
//!
//! **Hard limits.** No rclone call may hang the runner (a hung heartbeat upload once stopped a
//! run's heartbeats while its solver kept working, and the launcher destroyed the machine with its
//! results): rclone gets short timeouts of its own (`--contimeout 10s --timeout 30s
//! --low-level-retries 2 --retries 1`), and [`run_limited`] kills the call (its whole process
//! group) once it has run 45 s (`RUNNER_NET_LIMIT_S`) plus one second per MiB it moves. Uploads
//! that matter are tried again ([`retry`]); heartbeats and STOP checks simply go again at the next
//! heartbeat. [`Counted`] counts failures for the heartbeat ([`NetStats`]).

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use crate::bundle::{sha256_file, sha256_reader};

pub trait Remote: Send + Sync {
    /// Human-readable location.
    fn describe(&self) -> String;
    /// Uploads `local` to `key`.
    fn put(&self, local: &Path, key: &str, immutable: bool) -> Result<(), String>;
    /// Downloads `key` to `local`; `Ok(false)` when it does not exist.
    fn get(&self, key: &str, local: &Path) -> Result<bool, String>;
    /// Every key under `prefix` (which ends with `/`) with its size in bytes, sorted by key.
    fn list_sized(&self, prefix: &str) -> Result<Vec<(String, u64)>, String>;
    /// Every key under `prefix` (which ends with `/`), sorted.
    fn list(&self, prefix: &str) -> Result<Vec<String>, String> {
        Ok(self
            .list_sized(prefix)?
            .into_iter()
            .map(|(k, _)| k)
            .collect())
    }
    /// SHA-256 and size of the stored object, read back in full; `None` when missing.
    fn sha256(&self, key: &str) -> Result<Option<(String, u64)>, String>;
    /// Whether `key` exists (a cheap check: one listing of its directory level).
    fn exists(&self, key: &str) -> Result<bool, String>;
    /// Uploads every file under `dir` to `prefix/<relative path>` and checks each arrived with
    /// its size. Returns the keys uploaded.
    fn put_dir(&self, dir: &Path, prefix: &str, immutable: bool) -> Result<Vec<String>, String> {
        let mut keys = Vec::new();
        if !dir.is_dir() {
            return Ok(keys);
        }
        let prefix = prefix.trim_end_matches('/');
        for rel in crate::bundle::list_files(dir)? {
            let key = format!("{prefix}/{rel}");
            self.put(&dir.join(&rel), &key, immutable)?;
            keys.push(key);
        }
        let listed: std::collections::BTreeMap<String, u64> = self
            .list_sized(&format!("{prefix}/"))?
            .into_iter()
            .collect();
        for (rel, key) in crate::bundle::list_files(dir)?.iter().zip(&keys) {
            let n = fs::metadata(dir.join(rel))
                .map_err(|e| e.to_string())?
                .len();
            if listed.get(key) != Some(&n) {
                return Err(format!("upload check: {key} is not there with {n} bytes"));
            }
        }
        Ok(keys)
    }
}

/// The remote store given by the environment, if any: `RUNNER_REMOTE_DIR` (a local directory,
/// for tests and the laptop), else R2 through rclone when `RCLONE_CONFIG_R2_TYPE` and
/// `R2_BUCKET` are set.
pub fn from_env() -> Option<Box<dyn Remote>> {
    if let Ok(d) = std::env::var("RUNNER_REMOTE_DIR")
        && !d.is_empty()
    {
        return Some(Box::new(DirRemote { root: d.into() }));
    }
    match (
        std::env::var("RCLONE_CONFIG_R2_TYPE"),
        std::env::var("R2_BUCKET"),
    ) {
        (Ok(_), Ok(b)) if !b.is_empty() => Some(Box::new(RcloneRemote::new(&b))),
        _ => None,
    }
}

// ---------------------------------------------------------------- test kill hook

/// The `resume_identity` test hook: when `RUNNER_KILL_AT` equals `<point>:<n>`, the runner writes
/// `RUNNER_KILL_FLAG` (a file path) and waits forever, so the test can SIGKILL it at exactly
/// this point. Points: `after_done:N` (done-list holding N entries just written),
/// `mid_upload:N` (half of the bundle of the N-th entry written to the remote),
/// `after_upload:N` (that bundle uploaded and verified, not yet in a done-list).
pub fn kill_point(point: &str, n: u64) {
    let Ok(want) = std::env::var("RUNNER_KILL_AT") else {
        return;
    };
    if want != format!("{point}:{n}") {
        return;
    }
    if let Ok(flag) = std::env::var("RUNNER_KILL_FLAG") {
        let _ = fs::write(flag, format!("{point}:{n}\n"));
    }
    eprintln!("runner: test kill point {point}:{n} reached; waiting to be killed");
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

/// Which entry number an upload in progress belongs to (for the `mid_upload` kill point).
pub static UPLOAD_ENTRY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// ---------------------------------------------------------------- local directory

pub struct DirRemote {
    pub root: PathBuf,
}

impl DirRemote {
    fn path(&self, key: &str) -> PathBuf {
        key.split('/').fold(self.root.clone(), |p, s| p.join(s))
    }
}

impl Remote for DirRemote {
    fn describe(&self) -> String {
        format!("directory {}", self.root.display())
    }

    fn put(&self, local: &Path, key: &str, immutable: bool) -> Result<(), String> {
        let dest = self.path(key);
        let data = fs::read(local).map_err(|e| format!("{}: {e}", local.display()))?;
        if immutable && dest.exists() {
            let have = fs::read(&dest).map_err(|e| e.to_string())?;
            return if have == data {
                Ok(())
            } else {
                Err(format!("immutable key {key} exists with different content"))
            };
        }
        if let Some(d) = dest.parent() {
            fs::create_dir_all(d).map_err(|e| e.to_string())?;
        }
        let tmp = dest.with_extension(format!("partial-{}", std::process::id()));
        let mut f = fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        let half = data.len() / 2;
        f.write_all(&data[..half]).map_err(|e| e.to_string())?;
        f.flush().map_err(|e| e.to_string())?;
        if key.contains("/out/") && key.ends_with(".tar.zst") {
            kill_point(
                "mid_upload",
                UPLOAD_ENTRY.load(std::sync::atomic::Ordering::Relaxed),
            );
        }
        f.write_all(&data[half..]).map_err(|e| e.to_string())?;
        drop(f);
        fs::rename(&tmp, &dest).map_err(|e| format!("rename {}: {e}", dest.display()))
    }

    fn get(&self, key: &str, local: &Path) -> Result<bool, String> {
        let src = self.path(key);
        if !src.is_file() {
            return Ok(false);
        }
        if let Some(d) = local.parent() {
            fs::create_dir_all(d).map_err(|e| e.to_string())?;
        }
        fs::copy(&src, local).map_err(|e| format!("{}: {e}", src.display()))?;
        Ok(true)
    }

    fn list_sized(&self, prefix: &str) -> Result<Vec<(String, u64)>, String> {
        let dir = self.path(prefix.trim_end_matches('/'));
        if !dir.is_dir() {
            return Ok(vec![]);
        }
        let mut v = Vec::new();
        for r in crate::bundle::list_files(&dir)? {
            // A temporary file of an upload that was killed is not an object.
            if r.contains(".partial-") {
                continue;
            }
            let n = fs::metadata(dir.join(&r)).map_err(|e| e.to_string())?.len();
            v.push((format!("{}/{r}", prefix.trim_end_matches('/')), n));
        }
        v.sort();
        Ok(v)
    }

    fn sha256(&self, key: &str) -> Result<Option<(String, u64)>, String> {
        let p = self.path(key);
        if !p.is_file() {
            return Ok(None);
        }
        sha256_file(&p).map(Some)
    }

    fn exists(&self, key: &str) -> Result<bool, String> {
        Ok(self.path(key).is_file())
    }
}

// ---------------------------------------------------------------- rclone (R2)

/// rclone's own limits on every call: a connection must open within 10 s, a transfer that moves
/// nothing for 30 s fails, and one retry each at the low (request) and high (whole operation)
/// levels. The wall-clock kill ([`RcloneRemote::limit_for`]) is the backstop behind them.
const RCLONE_LIMITS: [&str; 8] = [
    "--contimeout",
    "10s",
    "--timeout",
    "30s",
    "--low-level-retries",
    "2",
    "--retries",
    "1",
];

/// Wall-clock limit of an rclone call that moves (almost) no data, unless `RUNNER_NET_LIMIT_S`
/// says otherwise: a heartbeat upload or a STOP check.
pub const NET_LIMIT: Duration = Duration::from_secs(45);

/// Seconds of wall-clock limit granted per MiB moved: a transfer may run as slowly as 0.5 MiB/s
/// and still finish (a host with a slow route to R2 was seen below 1 MiB/s).
pub const SECS_PER_MIB: u64 = 2;

/// The longest wall-clock limit a size-scaled call gets (never below the base limit itself). Three
/// hours: a 36 GB stored shard fetched by one stream from R2's European region to a host in
/// Thailand took longer than the former 30 minutes on every attempt (round o3r-1-1, 2026-10-21),
/// so the run never got past its inputs; rclone's own `--timeout` still ends a transfer that stalls.
pub const MAX_NET_LIMIT: Duration = Duration::from_secs(3 * 60 * 60);

/// Downloads larger than this are fetched as [`MULTI_THREAD_STREAMS`] parallel ranges (rclone's
/// multi-thread download): one stream to a far host is limited by the route's latency, not its
/// bandwidth.
pub const MULTI_THREAD_CUTOFF_MIB: u64 = 64;

/// Parallel ranges of a large download.
pub const MULTI_THREAD_STREAMS: u32 = 16;

/// The wall-clock limit of a call moving `bytes`: `base` plus [`SECS_PER_MIB`] per MiB, at most
/// [`MAX_NET_LIMIT`] (or `base` when that is larger). Used for uploads and downloads alike.
pub fn limit_for_bytes(base: Duration, bytes: u64) -> Duration {
    let extra = (bytes >> 20).saturating_mul(SECS_PER_MIB);
    base.saturating_add(Duration::from_secs(extra))
        .min(MAX_NET_LIMIT.max(base))
}

pub struct RcloneRemote {
    /// e.g. `r2:bucket`.
    pub base: String,
    /// The rclone executable (`RUNNER_RCLONE` overrides `rclone`; tests use a fake that hangs).
    pub bin: PathBuf,
    /// Wall-clock limit of a call moving no data ([`NET_LIMIT`]); [`SECS_PER_MIB`] per MiB moved is
    /// added ([`RcloneRemote::limit_for`]).
    pub limit: Duration,
}

impl RcloneRemote {
    /// R2 bucket `bucket`, with the executable and limit from the environment.
    pub fn new(bucket: &str) -> RcloneRemote {
        RcloneRemote {
            base: format!("r2:{bucket}"),
            bin: std::env::var_os("RUNNER_RCLONE")
                .filter(|v| !v.is_empty())
                .map_or_else(|| PathBuf::from("rclone"), PathBuf::from),
            limit: std::env::var("RUNNER_NET_LIMIT_S")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|&s| s > 0)
                .map_or(NET_LIMIT, Duration::from_secs),
        }
    }

    fn url(&self, key: &str) -> String {
        format!("{}/{key}", self.base)
    }

    /// `rclone <sub>` with the limits. On a rented machine (`CONTAINER_ID` set) with no
    /// `RCLONE_CONFIG`, its configuration is kept in memory (`RCLONE_CONFIG=/dev/null`, rclone's
    /// documented switch): the image has no config file (the `r2` remote comes from the
    /// `RCLONE_CONFIG_R2_*` variables), and otherwise rclone prints `NOTICE: Config file "…" not
    /// found - using defaults` on every call, that is with every heartbeat. Elsewhere the user's
    /// own config is kept.
    fn cmd(&self, sub: &str) -> Command {
        let mut c = Command::new(&self.bin);
        if std::env::var_os("RCLONE_CONFIG").is_none()
            && std::env::var("CONTAINER_ID").is_ok_and(|v| !v.is_empty())
        {
            c.env("RCLONE_CONFIG", "/dev/null");
        }
        c.arg(sub).args(RCLONE_LIMITS);
        c
    }

    /// The wall-clock limit of a call moving `bytes`: see [`limit_for_bytes`].
    pub fn limit_for(&self, bytes: u64) -> Duration {
        limit_for_bytes(self.limit, bytes)
    }

    fn run(&self, c: Command, limit: Duration) -> Result<(), String> {
        let what = describe_cmd(&c);
        // Its output is read to the end and dropped (never a closed pipe under it).
        let drain = |mut o: ChildStdout| {
            let _ = std::io::copy(&mut o, &mut std::io::sink());
        };
        match run_limited(c, limit, drain)? {
            (s, ()) if s.success() => Ok(()),
            (s, ()) => Err(format!("{what} failed ({s})")),
        }
    }

    /// Size of `key`, `None` when it does not exist (one listing of its directory level).
    fn size_of(&self, key: &str) -> Result<Option<u64>, String> {
        let (dir, _) = key.rsplit_once('/').unwrap_or(("", key));
        Ok(self
            .list_sized(&format!("{dir}/"))?
            .into_iter()
            .find(|(k, _)| k == key)
            .map(|(_, n)| n))
    }
}

/// `rclone <sub> … <last argument>` for messages (never the environment).
fn describe_cmd(c: &Command) -> String {
    let args: Vec<String> = c
        .get_args()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    format!(
        "{} {} … {}",
        Path::new(c.get_program()).display(),
        args.first().map_or("", String::as_str),
        args.last().map_or("", String::as_str)
    )
}

/// Runs `c` (stdin closed, stdout read by `read` on its own thread, stderr inherited) and kills
/// it, with everything it started, if it is still running after `limit`. Returns its exit
/// status and what `read` returned. Nothing here waits longer than `limit` (plus a moment): a
/// killed process is reaped on a background thread. On Unix the call gets its own process group,
/// so a Ctrl-C or SIGTERM aimed at the runner does not cut an upload short (the runner then
/// uploads what it finished, `stop.rs`), and the kill reaches the whole group.
pub fn run_limited<T: Send + 'static>(
    mut c: Command,
    limit: Duration,
    read: impl FnOnce(ChildStdout) -> T + Send + 'static,
) -> Result<(ExitStatus, T), String> {
    let what = describe_cmd(&c);
    c.stdin(Stdio::null()).stdout(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        c.process_group(0);
    }
    let mut child = c.spawn().map_err(|e| format!("cannot run {what}: {e}"))?;
    let out = child.stdout.take().ok_or("no stdout")?;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(read(out));
    });
    let t0 = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(st)) => {
                // Its output ends with it (unless something it started still holds it open).
                let left = limit
                    .saturating_sub(t0.elapsed())
                    .max(Duration::from_secs(1));
                return rx
                    .recv_timeout(left)
                    .map(|t| (st, t))
                    .map_err(|_| format!("{what}: its output stayed open after it ended"));
            }
            Ok(None) => {}
            Err(e) => {
                kill_tree(child);
                return Err(format!("{what}: {e}"));
            }
        }
        if t0.elapsed() >= limit {
            kill_tree(child);
            return Err(format!(
                "{what}: killed after {:.0} s (its hard limit)",
                limit.as_secs_f64()
            ));
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Kills a child and its process group, and reaps it on a background thread.
fn kill_tree(mut child: Child) {
    #[cfg(unix)]
    {
        let _ = Command::new("kill")
            .args(["-KILL", "--", &format!("-{}", child.id())])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill();
    std::thread::spawn(move || {
        let _ = child.wait();
    });
}

/// Reads everything a reader yields.
fn read_all(mut r: impl Read) -> Result<Vec<u8>, String> {
    let mut v = Vec::new();
    r.read_to_end(&mut v).map_err(|e| e.to_string())?;
    Ok(v)
}

impl Remote for RcloneRemote {
    fn describe(&self) -> String {
        self.base.clone()
    }

    fn put(&self, local: &Path, key: &str, immutable: bool) -> Result<(), String> {
        let bytes = fs::metadata(local).map_or(0, |m| m.len());
        let mut c = self.cmd("copyto");
        c.arg("--s3-no-check-bucket");
        if immutable {
            c.arg("--immutable");
        }
        c.arg(local).arg(self.url(key));
        self.run(c, self.limit_for(bytes))
    }

    fn get(&self, key: &str, local: &Path) -> Result<bool, String> {
        let Some(bytes) = self.size_of(key)? else {
            return Ok(false);
        };
        let mut c = self.cmd("copyto");
        if bytes > MULTI_THREAD_CUTOFF_MIB << 20 {
            c.args(["--multi-thread-streams", &MULTI_THREAD_STREAMS.to_string()])
                .args([
                    "--multi-thread-cutoff",
                    &format!("{MULTI_THREAD_CUTOFF_MIB}M"),
                ]);
        }
        c.arg(self.url(key)).arg(local);
        self.run(c, self.limit_for(bytes)).map(|()| true)
    }

    fn list_sized(&self, prefix: &str) -> Result<Vec<(String, u64)>, String> {
        let mut c = self.cmd("lsf");
        c.args(["-R", "--files-only", "--format", "sp", "--separator", "\t"])
            .arg(self.url(prefix.trim_end_matches('/')));
        let (st, out) = run_limited(c, self.limit, read_all)?;
        let out = out?;
        if !st.success() {
            // rclone's exit code 3 is "directory not found": a prefix nothing was ever written
            // under lists as empty. Any other failure (network, credentials, a killed call) is an
            // error: a listing that came back empty by mistake would make a resume restore
            // nothing and redo the work in silence.
            if st.code() == Some(3) {
                return Ok(vec![]);
            }
            return Err(format!("rclone lsf {prefix} failed ({st})"));
        }
        let mut v = Vec::new();
        for l in String::from_utf8_lossy(&out).lines() {
            let Some((size, path)) = l.split_once('\t') else {
                continue;
            };
            let n = size
                .trim()
                .parse::<u64>()
                .map_err(|_| format!("rclone lsf {prefix}: bad line {l:?}"))?;
            v.push((format!("{}/{path}", prefix.trim_end_matches('/')), n));
        }
        v.sort();
        Ok(v)
    }

    fn sha256(&self, key: &str) -> Result<Option<(String, u64)>, String> {
        let Some(bytes) = self.size_of(key)? else {
            return Ok(None);
        };
        let mut c = self.cmd("cat");
        c.arg(self.url(key));
        let (status, out) = run_limited(c, self.limit_for(bytes), sha256_reader)?;
        if !status.success() {
            return Err(format!("rclone cat {key} failed ({status})"));
        }
        out.map(Some)
    }

    fn exists(&self, key: &str) -> Result<bool, String> {
        // One level of the key's directory only (cheap on S3: a delimited listing).
        let (dir, name) = key.rsplit_once('/').unwrap_or(("", key));
        let mut c = self.cmd("lsf");
        c.args(["--files-only", "--max-depth", "1", "--include"])
            .arg(format!("/{name}"))
            .arg(self.url(dir));
        let (st, out) = run_limited(c, self.limit, read_all)?;
        if !st.success() {
            return Ok(false);
        }
        Ok(String::from_utf8_lossy(&out?)
            .lines()
            .any(|l| l.trim() == name))
    }

    fn put_dir(&self, dir: &Path, prefix: &str, immutable: bool) -> Result<Vec<String>, String> {
        if !dir.is_dir() {
            return Ok(Vec::new());
        }
        let files = crate::bundle::list_files(dir)?;
        let bytes: u64 = files
            .iter()
            .map(|r| fs::metadata(dir.join(r)).map_or(0, |m| m.len()))
            .sum();
        let limit = self
            .limit_for(bytes)
            .saturating_add(Duration::from_secs(files.len() as u64));
        let prefix = prefix.trim_end_matches('/');
        let mut c = self.cmd("copy");
        c.arg("--s3-no-check-bucket");
        if immutable {
            c.arg("--immutable");
        }
        c.arg(dir).arg(self.url(prefix));
        self.run(c, limit)?;
        let mut c = self.cmd("check");
        c.arg("--one-way").arg(dir).arg(self.url(prefix));
        self.run(c, limit)?;
        Ok(files.into_iter().map(|r| format!("{prefix}/{r}")).collect())
    }
}

// ---------------------------------------------------------------- counting wrapper

/// Network health for the heartbeat: failed uploads, other failed calls, the last good upload.
#[derive(Debug, Default)]
pub struct NetStats {
    /// Uploads (`put`, `put_dir`) that failed or were killed at their limit.
    pub upload_failures: AtomicU64,
    /// Other calls (downloads, listings, STOP checks) that failed or were killed.
    pub other_failures: AtomicU64,
    last_upload_ok: Mutex<Option<Instant>>,
}

impl NetStats {
    /// Seconds since the last successful upload (`None` before the first).
    pub fn last_upload_ok_s(&self) -> Option<f64> {
        self.last_upload_ok
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .map(|t| t.elapsed().as_secs_f64())
    }

    fn upload<T>(&self, r: Result<T, String>) -> Result<T, String> {
        match &r {
            Ok(_) => {
                *self
                    .last_upload_ok
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
            }
            Err(_) => {
                self.upload_failures.fetch_add(1, Ordering::Relaxed);
            }
        }
        r
    }

    fn other<T>(&self, r: Result<T, String>) -> Result<T, String> {
        if r.is_err() {
            self.other_failures.fetch_add(1, Ordering::Relaxed);
        }
        r
    }
}

/// A remote that counts its failures and successes in [`NetStats`].
pub struct Counted {
    pub inner: Box<dyn Remote>,
    pub stats: Arc<NetStats>,
}

impl Remote for Counted {
    fn describe(&self) -> String {
        self.inner.describe()
    }
    fn put(&self, local: &Path, key: &str, immutable: bool) -> Result<(), String> {
        self.stats.upload(self.inner.put(local, key, immutable))
    }
    fn get(&self, key: &str, local: &Path) -> Result<bool, String> {
        self.stats.other(self.inner.get(key, local))
    }
    fn list_sized(&self, prefix: &str) -> Result<Vec<(String, u64)>, String> {
        self.stats.other(self.inner.list_sized(prefix))
    }
    fn sha256(&self, key: &str) -> Result<Option<(String, u64)>, String> {
        self.stats.other(self.inner.sha256(key))
    }
    fn exists(&self, key: &str) -> Result<bool, String> {
        self.stats.other(self.inner.exists(key))
    }
    fn put_dir(&self, dir: &Path, prefix: &str, immutable: bool) -> Result<Vec<String>, String> {
        self.stats
            .upload(self.inner.put_dir(dir, prefix, immutable))
    }
}

/// Tries `f` up to `attempts` times, 2 s apart, until it succeeds (for uploads that matter:
/// markers, outputs, done-lists; heartbeats and STOP checks are simply tried again next time).
pub fn retry<T>(attempts: u32, mut f: impl FnMut() -> Result<T, String>) -> Result<T, String> {
    let mut last = Err("no attempt".to_string());
    for a in 0..attempts.max(1) {
        if a > 0 {
            std::thread::sleep(Duration::from_secs(2));
        }
        last = f();
        match &last {
            Ok(_) => break,
            Err(e) => eprintln!("runner: network call failed (attempt {}): {e}", a + 1),
        }
    }
    last
}

// ---------------------------------------------------------------- content-addressed inputs

/// The prefix of content-addressed inputs: `inputs/sha256/<hex>` holds the file whose SHA-256 is
/// `<hex>` (format-2 manifests' `inputs`, `launch.py put-input`, the label shards of
/// `hybrid_rung`'s `upload_shards`).
pub const INPUTS_PREFIX: &str = "inputs/sha256/";

/// The key of the content-addressed input with SHA-256 `hex`.
pub fn input_key(hex: &str) -> String {
    format!("{INPUTS_PREFIX}{hex}")
}

/// The SHA-256 (lowercase hex) named by an input reference: a key `inputs/sha256/<64 hex>` or a
/// manifest value `sha256:<64 hex>`; `None` for anything else.
pub fn input_hex(reference: &str) -> Option<String> {
    let hex = reference
        .strip_prefix(INPUTS_PREFIX)
        .or_else(|| reference.strip_prefix("sha256:"))?;
    (hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
        .then(|| hex.to_ascii_lowercase())
}

/// Fetches the content-addressed input `inputs/sha256/<hex>` to `local` (up to 3 attempts),
/// unless a file with that SHA-256 is there already, and checks the SHA-256 of what arrived: a
/// file that does not match is deleted and refused. Returns whether it was downloaded.
pub fn fetch_input(r: &dyn Remote, hex: &str, local: &Path) -> Result<bool, String> {
    if local.is_file() && sha256_file(local).is_ok_and(|(h, _)| h == hex) {
        return Ok(false);
    }
    let key = input_key(hex);
    if !retry(3, || r.get(&key, local))? {
        return Err(format!("{key} not found"));
    }
    let (h, _) = sha256_file(local)?;
    if h != hex {
        let _ = fs::remove_file(local);
        return Err(format!("{key}: downloaded sha256 {h}, want {hex}"));
    }
    Ok(true)
}

/// Uploads `local` content-addressed, to `inputs/sha256/<its SHA-256>` (up to 3 attempts), unless
/// an object of that key and size is there already; checks the size after the upload. With
/// `want`, the file's SHA-256 must be that (e.g. the one its sidecar recorded). Returns the
/// SHA-256, the size and what happened (`"uploaded"` or `"already there"`).
pub fn put_input(
    r: &dyn Remote,
    local: &Path,
    want: Option<&str>,
) -> Result<(String, u64, &'static str), String> {
    let (sha, bytes) = sha256_file(local)?;
    if let Some(w) = want
        && sha != w
    {
        return Err(format!(
            "{}: SHA-256 {sha}, but its record says {w}",
            local.display()
        ));
    }
    let key = input_key(&sha);
    let size = || -> Result<Option<u64>, String> {
        Ok(r.list_sized(INPUTS_PREFIX)?
            .into_iter()
            .find(|(k, _)| *k == key)
            .map(|(_, n)| n))
    };
    if size()? == Some(bytes) {
        return Ok((sha, bytes, "already there"));
    }
    retry(3, || r.put(local, &key, true))?;
    match size()? {
        Some(n) if n == bytes => Ok((sha, bytes, "uploaded")),
        other => Err(format!(
            "{key}: {other:?} bytes after the upload, want {bytes}"
        )),
    }
}

/// The pause before attempt `a` (from 1) of [`retry_patient`]: 2, 4, 8, 16, 32 s, then 60 s.
pub fn backoff(a: u32) -> Duration {
    Duration::from_secs((2u64 << a.saturating_sub(1).min(5)).min(60))
}

/// [`retry`] with growing pauses ([`backoff`]): for uploads whose failure would end the run (a
/// finished chunk's bundle), so a network failure of a few minutes is waited out. Eight attempts
/// wait about 3 minutes in all, besides the attempts themselves.
pub fn retry_patient<T>(
    attempts: u32,
    mut f: impl FnMut() -> Result<T, String>,
) -> Result<T, String> {
    let mut last = Err("no attempt".to_string());
    for a in 0..attempts.max(1) {
        if a > 0 {
            std::thread::sleep(backoff(a));
        }
        last = f();
        match &last {
            Ok(_) => break,
            Err(e) => eprintln!("runner: network call failed (attempt {}): {e}", a + 1),
        }
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_scales_with_size_and_is_capped() {
        let base = Duration::from_secs(60);
        let mib = 1u64 << 20;
        assert_eq!(limit_for_bytes(base, 0), base);
        assert_eq!(limit_for_bytes(base, mib - 1), base);
        assert_eq!(limit_for_bytes(base, mib), Duration::from_secs(62));
        // 100 MiB: 60 s + 200 s.
        assert_eq!(limit_for_bytes(base, 100 * mib), Duration::from_secs(260));
        // The 14 MiB file killed at ~56-59 s on the slow host now gets 88 s.
        assert_eq!(limit_for_bytes(base, 14 * mib), Duration::from_secs(88));
        // Capped at three hours, whatever the size (no overflow either).
        assert_eq!(limit_for_bytes(base, 10_000 * mib), MAX_NET_LIMIT);
        assert_eq!(limit_for_bytes(base, u64::MAX), MAX_NET_LIMIT);
        // A base above the cap is kept as it is.
        let big = Duration::from_secs(4 * 3600);
        assert_eq!(limit_for_bytes(big, 5 * mib), big);
        // The 36 GB base shard that one stream could not fetch in 30 minutes now gets three hours.
        assert_eq!(
            limit_for_bytes(base, 36_000 * mib),
            Duration::from_secs(3 * 3600)
        );
        // Monotonic in size.
        assert!(limit_for_bytes(base, 50 * mib) < limit_for_bytes(base, 51 * mib));
    }

    #[test]
    fn patient_backoff() {
        let s: Vec<u64> = (1..=8).map(|a| backoff(a).as_secs()).collect();
        assert_eq!(s, [2, 4, 8, 16, 32, 60, 60, 60]);
        assert_eq!(
            s[..7].iter().sum::<u64>(),
            182,
            "eight attempts wait about 3 minutes"
        );
        let mut n = 0;
        assert_eq!(
            retry_patient(3, || {
                n += 1;
                if n < 2 { Err("x".into()) } else { Ok(n) }
            }),
            Ok(2)
        );
    }

    #[test]
    fn dir_remote_immutable_and_list() {
        let base = std::env::temp_dir().join(format!("runner-remote-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let r = DirRemote {
            root: base.join("r"),
        };
        let a = base.join("a");
        fs::write(&a, b"one").unwrap();
        r.put(&a, "x/y/1.tar.zst", true).unwrap();
        r.put(&a, "x/y/1.tar.zst", true).unwrap(); // identical: fine
        let b = base.join("b");
        fs::write(&b, b"two").unwrap();
        assert!(r.put(&b, "x/y/1.tar.zst", true).is_err());
        r.put(&b, "x/z.json", false).unwrap();
        assert_eq!(r.list("x/").unwrap(), vec!["x/y/1.tar.zst", "x/z.json"]);
        assert_eq!(
            r.list_sized("x/y/").unwrap(),
            vec![("x/y/1.tar.zst".to_string(), 3)]
        );
        // A killed upload's temporary file is not listed.
        fs::write(base.join("r/x/y/2.partial-1"), b"half").unwrap();
        assert_eq!(r.list("x/y/").unwrap(), vec!["x/y/1.tar.zst"]);
        assert_eq!(r.list("nothing/").unwrap(), Vec::<String>::new());
        assert_eq!(r.sha256("x/y/1.tar.zst").unwrap().unwrap().1, 3);
        assert_eq!(r.sha256("x/none").unwrap(), None);
        let _ = fs::remove_dir_all(&base);
    }

    /// Content-addressed inputs: references, upload (once), download (verified, kept when the
    /// local copy already matches), a missing key and an object whose bytes are not its key's.
    #[test]
    fn inputs_by_sha256() {
        let h = "ab".repeat(32);
        assert_eq!(input_hex(&format!("inputs/sha256/{h}")), Some(h.clone()));
        assert_eq!(
            input_hex(&format!("sha256:{}", h.to_uppercase())),
            Some(h.clone())
        );
        for bad in [
            "inputs/sha256/abc".to_string(),
            format!("inputs/sha256/{}", "g".repeat(64)),
            format!("inputs/{h}"),
            h.clone(),
            format!("runs/x/{h}"),
            format!("inputs/sha256/{h}/x"),
        ] {
            assert_eq!(input_hex(&bad), None, "{bad}");
        }
        let base = std::env::temp_dir().join(format!("runner-inputs-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let r = DirRemote {
            root: base.join("r"),
        };
        let f = base.join("net.safetensors");
        fs::write(&f, b"weights").unwrap();
        let sha = crate::bundle::sha256_hex(b"weights");
        assert_eq!(
            put_input(&r, &f, Some(&sha)),
            Ok((sha.clone(), 7, "uploaded"))
        );
        assert_eq!(
            put_input(&r, &f, None),
            Ok((sha.clone(), 7, "already there"))
        );
        let e = put_input(&r, &f, Some(&h)).expect_err("not its record's hash");
        assert!(e.contains("but its record says"), "{e}");
        let got = base.join("in").join(&sha);
        assert_eq!(fetch_input(&r, &sha, &got), Ok(true));
        assert_eq!(fs::read(&got).unwrap(), b"weights");
        assert_eq!(fetch_input(&r, &sha, &got), Ok(false), "kept: it matches");
        let e = fetch_input(&r, &h, &base.join("in").join(&h)).expect_err("missing");
        assert_eq!(e, format!("inputs/sha256/{h} not found"));
        // An object whose bytes do not hash to its key is refused and not kept.
        fs::write(base.join("r/inputs/sha256").join(&h), b"other").unwrap();
        let local = base.join("in").join(&h);
        let e = fetch_input(&r, &h, &local).expect_err("wrong bytes");
        assert!(e.contains("downloaded sha256"), "{e}");
        assert!(!local.exists());
        let _ = fs::remove_dir_all(&base);
    }
}
