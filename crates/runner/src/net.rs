//! Network work off the runner's main loop (docs/M3-SPEC.md §7.3, "the network never blocks the
//! runner").
//!
//! The runner's loop writes the local `heartbeat.json` every heartbeat interval whatever the
//! network does; uploading it, checking the launcher's `STOP` object and uploading the chunk's
//! partial outputs each run on their own background thread (a [`Lane`]). A lane runs one job at
//! a time; a job submitted while another waits replaces it, so the newest heartbeat wins and a
//! STOP check or partial upload is skipped while the previous one is still going. Every rclone
//! call has hard limits (`remote.rs`): rclone's own short timeouts, and a wall-clock kill (45 s
//! for a heartbeat or STOP check, more for larger files: 2 s per MiB, at most 30 min). Heartbeats report
//! `upload_failures`, `other_net_failures` and `last_upload_ok_s`.
//!
//! **Partial uploads** ([`Net::partial`], [`upload_changed`]): every `upload_every_s` the files
//! of the chunk's `out/` that changed since their last upload are copied aside (so a file still
//! being appended to goes up whole up to that moment) and uploaded to a mutable prefix:
//! `runs/<run>/partial/out/<path>` in format 1, `runs/<run>/partial/<stream>/<index:06>/out/<path>`
//! in format 2. Checkpoints (`*.ckpt`, the worlds finished since the last complete decision, see
//! `search::measure`) go first, then the other files, smallest first. A resumed run downloads
//! them back ([`restore`]) and continues at world granularity.

use std::collections::BTreeMap;
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime};

use serde_json::{Value, json};

use crate::remote::{Counted, NetStats, Remote};
use crate::stop;

type Job = Box<dyn FnOnce() + Send>;

#[derive(Default)]
struct LaneState {
    pending: Option<Job>,
    busy: bool,
}

/// A background thread that runs jobs one at a time; a job submitted while another is waiting
/// replaces it (the newest wins).
pub struct Lane {
    shared: Arc<(Mutex<LaneState>, Condvar)>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Lane {
    pub fn new(name: &str) -> Lane {
        let shared = Arc::new((Mutex::new(LaneState::default()), Condvar::new()));
        let s = Arc::clone(&shared);
        let spawned = std::thread::Builder::new()
            .name(format!("net-{name}"))
            .spawn(move || {
                loop {
                    let job = {
                        let (m, cv) = &*s;
                        let mut st = lock(m);
                        loop {
                            if let Some(j) = st.pending.take() {
                                st.busy = true;
                                break j;
                            }
                            st = cv.wait(st).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    // A job that panics must not end the lane.
                    if catch_unwind(AssertUnwindSafe(job)).is_err() {
                        eprintln!("runner: a background network job panicked");
                    }
                    let (m, cv) = &*s;
                    lock(m).busy = false;
                    cv.notify_all();
                }
            });
        if let Err(e) = spawned {
            eprintln!("runner: cannot start the {name} network thread: {e}");
        }
        Lane { shared }
    }

    /// Queues `job`, replacing a job still waiting.
    pub fn submit(&self, job: impl FnOnce() + Send + 'static) {
        let (m, cv) = &*self.shared;
        lock(m).pending = Some(Box::new(job));
        cv.notify_all();
    }

    /// Whether a job is running or waiting.
    pub fn busy(&self) -> bool {
        let st = lock(&self.shared.0);
        st.busy || st.pending.is_some()
    }

    /// Waits until no job is running or waiting, at most `limit`; whether it is idle.
    pub fn wait_idle(&self, limit: Duration) -> bool {
        let (m, cv) = &*self.shared;
        let t0 = Instant::now();
        let mut st = lock(m);
        while st.busy || st.pending.is_some() {
            let left = limit.saturating_sub(t0.elapsed());
            if left.is_zero() {
                return false;
            }
            st = cv
                .wait_timeout(st, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        true
    }
}

/// When each remote key was last uploaded by the partial uploads: (bytes, modification time) of
/// the local file then.
type Sent = Arc<Mutex<BTreeMap<String, (u64, Option<SystemTime>)>>>;

/// A run's remote store and its background network work.
pub struct Net {
    /// The store, counting its failures in `stats`.
    pub remote: Arc<dyn Remote>,
    pub stats: Arc<NetStats>,
    /// The run's key prefix (`runs/<run_id>` unless a format-1 manifest says otherwise).
    prefix: String,
    hb: Lane,
    stop: Lane,
    partial: Lane,
    /// Small records written now and then (the machine-cost record, `machine.rs`).
    misc: Lane,
    /// Asks a partial upload in progress to stop between files.
    cancel: Arc<AtomicBool>,
    sent: Sent,
}

impl Net {
    pub fn new(raw: Box<dyn Remote>, prefix: &str) -> Net {
        let stats = Arc::new(NetStats::default());
        Net {
            remote: Arc::new(Counted {
                inner: raw,
                stats: Arc::clone(&stats),
            }),
            stats,
            prefix: prefix.trim_end_matches('/').to_string(),
            hb: Lane::new("heartbeat"),
            stop: Lane::new("stop"),
            partial: Lane::new("partial"),
            misc: Lane::new("misc"),
            cancel: Arc::new(AtomicBool::new(false)),
            sent: Arc::default(),
        }
    }

    /// Uploads heartbeat `text` to `key` in the background (written to `tmp` first); a heartbeat
    /// still waiting is replaced by this newer one.
    pub fn heartbeat(&self, text: String, key: String, tmp: PathBuf) {
        let r = Arc::clone(&self.remote);
        self.hb.submit(move || {
            if let Some(d) = tmp.parent() {
                let _ = fs::create_dir_all(d);
            }
            let res = fs::write(&tmp, text)
                .map_err(|e| format!("{}: {e}", tmp.display()))
                .and_then(|()| r.put(&tmp, &key, false));
            if let Err(e) = res {
                eprintln!("runner: heartbeat upload {key}: {e}");
            }
        });
    }

    /// Uploads `text` to each of `keys` (key, immutable) in the background, written to `tmp`
    /// first; a job still waiting is replaced by this newer one. Failures are logged only.
    pub fn upload_text(&self, text: String, keys: Vec<(String, bool)>, tmp: PathBuf) {
        let r = Arc::clone(&self.remote);
        self.misc.submit(move || {
            if let Some(d) = tmp.parent() {
                let _ = fs::create_dir_all(d);
            }
            if let Err(e) = fs::write(&tmp, text) {
                eprintln!("runner: {}: {e}", tmp.display());
                return;
            }
            for (key, immutable) in keys {
                if let Err(e) = r.put(&tmp, &key, immutable) {
                    eprintln!("runner: upload {key}: {e}");
                }
            }
        });
    }

    /// Checks the launcher's `<prefix>/STOP` object in the background (skipped while a check is
    /// still running); a found object requests the stop (`stop.rs`).
    pub fn poll_stop(&self, local_dir: &Path) {
        if self.stop.busy() || stop::requested().is_some() {
            return;
        }
        let r = Arc::clone(&self.remote);
        let (prefix, dir) = (self.prefix.clone(), local_dir.to_path_buf());
        self.stop
            .submit(move || stop::global().poll_remote(&*r, &prefix, &dir));
    }

    /// Whether a partial upload is running or waiting.
    pub fn partial_busy(&self) -> bool {
        self.partial.busy()
    }

    /// Uploads, in the background, the changed files of each `(local dir, key prefix)` set
    /// ([`upload_changed`]), copying them aside under `snap` first. Skipped while one is running.
    pub fn partial(&self, sets: Vec<(PathBuf, String)>, snap: PathBuf) {
        if self.partial.busy() {
            return;
        }
        self.cancel.store(false, Ordering::Relaxed);
        let (r, sent, cancel) = (
            Arc::clone(&self.remote),
            Arc::clone(&self.sent),
            Arc::clone(&self.cancel),
        );
        self.partial.submit(move || {
            for (dir, prefix) in sets {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                upload_changed(&*r, &dir, &prefix, &snap, &sent, Some(&cancel));
            }
        });
    }

    /// Asks a background partial upload to stop after the file it is on (before a stop's or
    /// the end's own uploads, which must not be overwritten by an older copy).
    pub fn cancel_partial(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// What the partial uploads have sent (so a stop's own upload skips unchanged files).
    pub fn sent(&self) -> Sent {
        Arc::clone(&self.sent)
    }

    /// Waits up to `limit` for the last heartbeat upload (before the process ends), then as
    /// long again for a record upload still going.
    pub fn drain(&self, limit: Duration) {
        self.hb.wait_idle(limit);
        self.misc.wait_idle(limit);
    }

    /// The network fields of a heartbeat.
    pub fn health(&self) -> Value {
        json!({
            "upload_failures": self.stats.upload_failures.load(Ordering::Relaxed),
            "other_net_failures": self.stats.other_failures.load(Ordering::Relaxed),
            "last_upload_ok_s": self.stats.last_upload_ok_s().map(f64::round),
        })
    }

    /// Adds [`Net::health`] to a heartbeat object.
    pub fn add_health(&self, hb: &mut Value) {
        if let (Some(o), Value::Object(h)) = (hb.as_object_mut(), self.health()) {
            o.extend(h);
        }
    }
}

/// Whether `rel` is a checkpoint (`search worlds`' `<out>.ckpt`).
pub fn is_ckpt(rel: &str) -> bool {
    rel.ends_with(".ckpt")
}

/// Whether `rel` is a file still being written (`<file>.tmp`: a checkpoint or progress file
/// being rewritten before it is renamed into place; `<file>.restore-tmp`: a download in
/// progress). Never uploaded: a copy of it could be torn, and it is gone a moment later.
pub fn is_temp(rel: &str) -> bool {
    rel.ends_with(".tmp") || rel.ends_with(".restore-tmp")
}

/// The files of `dir` in upload order: checkpoints first, then the others smallest first;
/// temporary files ([`is_temp`]) are left out.
fn upload_order(dir: &Path) -> Vec<(String, u64, Option<SystemTime>)> {
    let mut v: Vec<(String, u64, Option<SystemTime>)> = crate::bundle::list_files(dir)
        .unwrap_or_default()
        .into_iter()
        .filter(|rel| !is_temp(rel))
        .filter_map(|rel| {
            let m = fs::metadata(dir.join(&rel)).ok()?;
            Some((rel, m.len(), m.modified().ok()))
        })
        .collect();
    v.sort_by_key(|(rel, n, _)| (!is_ckpt(rel), *n, rel.clone()));
    v
}

/// Uploads the files of `dir` that changed since their last upload (by size and modification
/// time) to `<prefix>/<relative path>` (mutable keys), checkpoints first, then the others
/// smallest first; each is copied under `snap` first, so a file still being appended to goes up
/// as it stood at that moment. `cancel` stops it between files. Returns (uploaded, failed).
pub fn upload_changed(
    r: &dyn Remote,
    dir: &Path,
    prefix: &str,
    snap: &Path,
    sent: &Mutex<BTreeMap<String, (u64, Option<SystemTime>)>>,
    cancel: Option<&AtomicBool>,
) -> (usize, usize) {
    let (mut ok, mut failed) = (0, 0);
    if !dir.is_dir() {
        return (ok, failed);
    }
    let prefix = prefix.trim_end_matches('/');
    for (rel, bytes, mtime) in upload_order(dir) {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            break;
        }
        let key = format!("{prefix}/{rel}");
        if lock(sent).get(&key) == Some(&(bytes, mtime)) {
            continue;
        }
        let copy = snap.join(&rel);
        let res = copy
            .parent()
            .map_or(Ok(()), fs::create_dir_all)
            .and_then(|()| fs::copy(dir.join(&rel), &copy))
            .map_err(|e| format!("{}: {e}", copy.display()))
            .and_then(|_| r.put(&copy, &key, false));
        let _ = fs::remove_file(&copy);
        match res {
            Ok(()) => {
                lock(sent).insert(key, (bytes, mtime));
                ok += 1;
            }
            Err(e) => {
                eprintln!("runner: partial upload {key}: {e}");
                failed += 1;
            }
        }
    }
    (ok, failed)
}

/// What [`restore`] found and did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Restored {
    /// Files listed under the prefixes (whether or not they were downloaded).
    pub listed: usize,
    /// Files downloaded.
    pub files: usize,
    /// Bytes downloaded.
    pub bytes: u64,
}

/// Downloads every file under `<prefix>/` into `dest/<relative path>`, unless the local copy
/// is at least as large (append-only files: longer is newer). `prefixes` are tried in order and
/// the first that holds a path wins. A listing that fails is an error (nothing is guessed to be
/// absent); a prefix nothing was ever written under lists as empty. Temporary files
/// ([`is_temp`]) are skipped.
pub fn restore(r: &dyn Remote, prefixes: &[String], dest: &Path) -> Result<Restored, String> {
    let mut chosen: BTreeMap<String, (String, u64)> = BTreeMap::new();
    for p in prefixes {
        let p = p.trim_end_matches('/');
        for (key, n) in r.list_sized(&format!("{p}/"))? {
            let Some(rel) = key.strip_prefix(&format!("{p}/")) else {
                continue;
            };
            if rel
                .split('/')
                .any(|s| s.is_empty() || s == "." || s == "..")
                || is_temp(rel)
            {
                continue;
            }
            chosen.entry(rel.to_string()).or_insert((key, n));
        }
    }
    let mut got = Restored {
        listed: chosen.len(),
        ..Restored::default()
    };
    for (rel, (key, n)) in chosen {
        let local = rel.split('/').fold(dest.to_path_buf(), |p, s| p.join(s));
        if fs::metadata(&local).is_ok_and(|m| m.len() >= n) {
            continue;
        }
        if let Some(d) = local.parent() {
            fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        let tmp = local.with_extension("restore-tmp");
        if !crate::remote::retry(3, || r.get(&key, &tmp))? {
            continue;
        }
        fs::rename(&tmp, &local).map_err(|e| format!("{}: {e}", local.display()))?;
        got.files += 1;
        got.bytes += n;
    }
    Ok(got)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::DirRemote;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("runner-net-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).expect("mkdir");
        d
    }

    /// The newest job wins, jobs run one at a time, and waiting works.
    #[test]
    fn lanes_run_newest() {
        let lane = Lane::new("test");
        let seen = Arc::new(Mutex::new(Vec::new()));
        let gate = Arc::new(AtomicBool::new(false));
        let (s, g) = (Arc::clone(&seen), Arc::clone(&gate));
        lane.submit(move || {
            while !g.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(5));
            }
            lock(&s).push(0);
        });
        std::thread::sleep(Duration::from_millis(50));
        for i in 1..=3 {
            let s = Arc::clone(&seen);
            lane.submit(move || lock(&s).push(i));
        }
        assert!(lane.busy());
        assert!(
            !lane.wait_idle(Duration::from_millis(50)),
            "the first job is held"
        );
        gate.store(true, Ordering::Relaxed);
        assert!(lane.wait_idle(Duration::from_secs(5)));
        assert_eq!(*lock(&seen), vec![0, 3], "jobs 1 and 2 were replaced by 3");
        lane.submit(|| panic!("a failing job"));
        assert!(lane.wait_idle(Duration::from_secs(5)));
        let s = Arc::clone(&seen);
        lane.submit(move || lock(&s).push(4));
        assert!(lane.wait_idle(Duration::from_secs(5)));
        assert_eq!(*lock(&seen), vec![0, 3, 4], "the lane survives a panic");
    }

    /// Changed files only, checkpoints first; restore prefers the first prefix and never
    /// shortens a local file.
    #[test]
    fn partial_upload_and_restore() {
        let d = tmp("partial");
        let out = d.join("out");
        fs::create_dir_all(out.join("measure")).expect("mkdir");
        fs::write(out.join("measure/w.jsonl"), "a\n").expect("write");
        fs::write(out.join("measure/w.jsonl.ckpt"), "c1\nc2\n").expect("write");
        fs::write(out.join("big.log"), "x".repeat(100)).expect("write");
        // Files still being written are never uploaded.
        fs::write(out.join("measure/w.jsonl.ckpt.tmp"), "c1\n").expect("write");
        fs::write(out.join("measure/w.restore-tmp"), "half").expect("write");
        let r = DirRemote {
            root: d.join("remote"),
        };
        let sent = Mutex::new(BTreeMap::new());
        assert_eq!(upload_order(&out)[0].0, "measure/w.jsonl.ckpt");
        assert_eq!(upload_order(&out).len(), 3, "{:?}", upload_order(&out));
        let snap = d.join("snap");
        assert_eq!(
            upload_changed(&r, &out, "runs/x/partial/out", &snap, &sent, None),
            (3, 0)
        );
        assert_eq!(
            upload_changed(&r, &out, "runs/x/partial/out", &snap, &sent, None),
            (0, 0),
            "nothing changed"
        );
        fs::write(out.join("measure/w.jsonl"), "a\nb\n").expect("write");
        assert_eq!(
            upload_changed(&r, &out, "runs/x/partial/out", &snap, &sent, None),
            (1, 0)
        );
        // A later, complete copy under out/ wins over the partial one.
        let fin = d.join("final");
        fs::create_dir_all(&fin).expect("mkdir");
        fs::write(fin.join("big.log"), "y".repeat(150)).expect("write");
        r.put(&fin.join("big.log"), "runs/x/out/big.log", true)
            .expect("put");
        let dest = d.join("dest");
        let prefixes = ["runs/x/out".to_string(), "runs/x/partial/out".to_string()];
        // A temporary file that reached the remote some other way is not restored either.
        r.put(
            &out.join("measure/w.restore-tmp"),
            "runs/x/out/stray.tmp",
            true,
        )
        .expect("put");
        assert_eq!(
            restore(&r, &prefixes, &dest).expect("restore"),
            Restored {
                listed: 3,
                files: 3,
                bytes: 160
            }
        );
        assert!(!dest.join("stray.tmp").exists());
        assert_eq!(
            fs::read_to_string(dest.join("big.log")).expect("read"),
            "y".repeat(150)
        );
        assert_eq!(
            fs::read_to_string(dest.join("measure/w.jsonl")).expect("read"),
            "a\nb\n"
        );
        // A longer local file is kept.
        fs::write(dest.join("measure/w.jsonl"), "a\nb\nc\n").expect("write");
        assert_eq!(
            restore(&r, &prefixes, &dest).expect("restore"),
            Restored {
                listed: 3,
                files: 0,
                bytes: 0
            }
        );
        // Nothing under a prefix: an empty answer, not an error; a failing store is an error.
        let none = ["runs/none/out".to_string()];
        assert_eq!(
            restore(&r, &none, &dest).expect("empty"),
            Restored::default()
        );
        struct Broken;
        impl Remote for Broken {
            fn describe(&self) -> String {
                "broken".into()
            }
            fn put(&self, _: &Path, _: &str, _: bool) -> Result<(), String> {
                Err("down".into())
            }
            fn get(&self, _: &str, _: &Path) -> Result<bool, String> {
                Err("down".into())
            }
            fn list_sized(&self, _: &str) -> Result<Vec<(String, u64)>, String> {
                Err("down".into())
            }
            fn sha256(&self, _: &str) -> Result<Option<(String, u64)>, String> {
                Err("down".into())
            }
            fn exists(&self, _: &str) -> Result<bool, String> {
                Err("down".into())
            }
        }
        assert!(restore(&Broken, &prefixes, &dest).is_err());
        let _ = fs::remove_dir_all(&d);
    }
}
