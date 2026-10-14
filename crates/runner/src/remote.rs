//! The remote store behind one trait: R2 through rclone on rented machines, a local directory on
//! the laptop and in the `resume_identity` test. Keys are bucket-relative `/` paths
//! (`runs/<run>/...`, `done/<run>/...`, `inputs/sha256/<hex>`).
//!
//! Every write is whole-object: S3/R2 objects appear atomically, and the directory backend writes
//! a temporary file then renames it, so a kill mid-upload never leaves a partial object.
//! Immutable writes accept an identical existing object (a redone chunk reproduces its bundle
//! byte for byte) and refuse a different one.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::bundle::{sha256_file, sha256_reader};

pub trait Remote {
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
        (Ok(_), Ok(b)) if !b.is_empty() => Some(Box::new(RcloneRemote {
            base: format!("r2:{b}"),
        })),
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

/// An `rclone` command. On a rented machine (`CONTAINER_ID` set) with no `RCLONE_CONFIG`, its
/// configuration is kept in memory (`RCLONE_CONFIG=/dev/null`, rclone's documented switch): the
/// image has no config file (the `r2` remote comes from the `RCLONE_CONFIG_R2_*` variables), and
/// otherwise rclone prints `NOTICE: Config file "…" not found - using defaults` on every call,
/// that is with every heartbeat. Nothing else changes; elsewhere the user's own config is kept.
fn rclone() -> Command {
    let mut c = Command::new("rclone");
    if std::env::var_os("RCLONE_CONFIG").is_none()
        && std::env::var("CONTAINER_ID").is_ok_and(|v| !v.is_empty())
    {
        c.env("RCLONE_CONFIG", "/dev/null");
    }
    c
}

pub struct RcloneRemote {
    /// e.g. `r2:bucket`.
    pub base: String,
}

impl RcloneRemote {
    fn url(&self, key: &str) -> String {
        format!("{}/{key}", self.base)
    }

    fn run(&self, mut c: Command) -> Result<(), String> {
        match c.stdin(Stdio::null()).status() {
            Ok(s) if s.success() => Ok(()),
            Ok(s) => Err(format!("rclone failed ({s}): {c:?}")),
            Err(e) => Err(format!("cannot run rclone: {e}")),
        }
    }
}

impl Remote for RcloneRemote {
    fn describe(&self) -> String {
        self.base.clone()
    }

    fn put(&self, local: &Path, key: &str, immutable: bool) -> Result<(), String> {
        let mut c = rclone();
        c.args(["copyto", "--s3-no-check-bucket", "--retries", "5"]);
        if immutable {
            c.arg("--immutable");
        }
        c.arg(local).arg(self.url(key));
        self.run(c)
    }

    fn get(&self, key: &str, local: &Path) -> Result<bool, String> {
        if self.sha256_exists(key)?.is_none() {
            return Ok(false);
        }
        let mut c = rclone();
        c.args(["copyto", "--retries", "5"])
            .arg(self.url(key))
            .arg(local);
        self.run(c).map(|()| true)
    }

    fn list_sized(&self, prefix: &str) -> Result<Vec<(String, u64)>, String> {
        let o = rclone()
            .args([
                "lsf",
                "-R",
                "--files-only",
                "--format",
                "sp",
                "--separator",
                "\t",
            ])
            .arg(self.url(prefix.trim_end_matches('/')))
            .stdin(Stdio::null())
            .output()
            .map_err(|e| format!("cannot run rclone: {e}"))?;
        if !o.status.success() {
            // A missing prefix is an empty listing on S3; other errors show in the log.
            eprintln!(
                "runner: rclone lsf {prefix}: {}",
                String::from_utf8_lossy(&o.stderr).trim()
            );
            return Ok(vec![]);
        }
        let mut v = Vec::new();
        for l in String::from_utf8_lossy(&o.stdout).lines() {
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
        if self.sha256_exists(key)?.is_none() {
            return Ok(None);
        }
        let mut child = rclone()
            .args(["cat", "--retries", "5"])
            .arg(self.url(key))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot run rclone: {e}"))?;
        let out = sha256_reader(child.stdout.take().ok_or("no rclone stdout")?);
        let status = child.wait().map_err(|e| e.to_string())?;
        if !status.success() {
            return Err(format!("rclone cat {key} failed ({status})"));
        }
        out.map(Some)
    }

    fn exists(&self, key: &str) -> Result<bool, String> {
        // One level of the key's directory only (cheap on S3: a delimited listing).
        let (dir, name) = key.rsplit_once('/').unwrap_or(("", key));
        let o = rclone()
            .args(["lsf", "--files-only", "--max-depth", "1", "--include"])
            .arg(format!("/{name}"))
            .arg(self.url(dir))
            .stdin(Stdio::null())
            .output()
            .map_err(|e| format!("cannot run rclone: {e}"))?;
        if !o.status.success() {
            return Ok(false);
        }
        Ok(String::from_utf8_lossy(&o.stdout)
            .lines()
            .any(|l| l.trim() == name))
    }

    fn put_dir(&self, dir: &Path, prefix: &str, immutable: bool) -> Result<Vec<String>, String> {
        if !dir.is_dir() {
            return Ok(Vec::new());
        }
        let prefix = prefix.trim_end_matches('/');
        let mut c = rclone();
        c.args(["copy", "--s3-no-check-bucket", "--retries", "5"]);
        if immutable {
            c.arg("--immutable");
        }
        c.arg(dir).arg(self.url(prefix));
        self.run(c)?;
        let mut c = rclone();
        c.args(["check", "--one-way"])
            .arg(dir)
            .arg(self.url(prefix));
        self.run(c)?;
        Ok(crate::bundle::list_files(dir)?
            .into_iter()
            .map(|r| format!("{prefix}/{r}"))
            .collect())
    }
}

impl RcloneRemote {
    /// `Some(())` when the key exists (one `lsf` of its exact name).
    fn sha256_exists(&self, key: &str) -> Result<Option<()>, String> {
        let (dir, name) = key.rsplit_once('/').unwrap_or(("", key));
        let found = self.list(&format!("{dir}/"))?;
        let want = format!("{dir}/{name}");
        Ok(found.contains(&want).then_some(()))
    }
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
