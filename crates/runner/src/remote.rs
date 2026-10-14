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

pub struct RcloneRemote {
    /// e.g. `r2:bucket`.
    pub base: String,
    /// The rclone executable (`RUNNER_RCLONE` overrides `rclone`; tests use a fake that hangs).
    pub bin: PathBuf,
    /// Wall-clock limit of a call moving no data ([`NET_LIMIT`]); one second per MiB moved is
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

    /// The wall-clock limit of a call moving `bytes`: [`RcloneRemote::limit`] plus one second
    /// per MiB (that is, the transfer may run at 1 MiB/s and still finish).
    pub fn limit_for(&self, bytes: u64) -> Duration {
        self.limit + Duration::from_secs(bytes >> 20)
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
            // A missing prefix is an empty listing on S3; other errors show in the log.
            eprintln!("runner: rclone lsf {prefix}: exit {st}");
            return Ok(vec![]);
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
        let limit = self.limit_for(bytes) + Duration::from_secs(files.len() as u64);
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
