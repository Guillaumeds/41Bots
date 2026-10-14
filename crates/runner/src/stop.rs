//! Stopping a run cleanly (docs/M3-SPEC.md §13.1 item 11): whatever the reason (the machine's
//! deadline, the launcher's low-CPU rule or time limit, a cancelled workflow, a person pressing
//! Ctrl-C), the runner stops its work, uploads everything finished so far, then writes its end
//! marker, so the launcher can destroy the machine without losing finished results.
//!
//! A stop is requested by
//! - a signal: SIGTERM (`docker stop`, `kill`; the boot script passes its own SIGTERM on to the
//!   runner), SIGINT or SIGHUP. The handler only sets a flag (`signal_hook::flag::register`);
//!   the runner's loops read it every few hundred milliseconds;
//! - a `STOP` object in the run's remote prefix (`runs/<run_id>/STOP`), which the launcher writes
//!   before it destroys a machine that is still working; it is read at every heartbeat;
//! - a `STOP` file in the run's local work directory (the laptop, and tests).
//!
//! What a stopped run keeps: format 1 uploads its whole `out/` (a `search_measure` chunk's worlds
//! file holds every decision finished, one line each, written as it finishes; a line cut short is
//! skipped by every reader) and writes `FAILED` with `stopped` and the list of files. Format 2
//! stops its running chunks, uploads each one's unfinished `out/` as
//! `runs/<run>/partial/<stream>/<index>-<utc>.tar.zst` (not in the done-list: a resumed run
//! redoes the chunk from its seed), writes the done-list, then `DONE` with `stopped`.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::remote::Remote;

/// Stop requests: one flag per signal (so the reason can name it), and the first reason given.
pub struct StopState {
    signals: Vec<(&'static str, Arc<AtomicBool>)>,
    reason: Mutex<Option<String>>,
}

impl Default for StopState {
    fn default() -> Self {
        StopState {
            signals: ["SIGTERM", "SIGINT", "SIGHUP"]
                .iter()
                .map(|&n| (n, Arc::new(AtomicBool::new(false))))
                .collect(),
            reason: Mutex::new(None),
        }
    }
}

impl StopState {
    /// Asks the run to stop (the first reason given is kept).
    pub fn request(&self, reason: &str) {
        let mut r = self.reason.lock().unwrap_or_else(|e| e.into_inner());
        if r.is_none() {
            eprintln!("runner: stop requested: {reason}");
            *r = Some(reason.to_string());
        }
    }

    /// Why the run must stop, if it must.
    pub fn requested(&self) -> Option<String> {
        for (name, flag) in &self.signals {
            if flag.load(Ordering::Relaxed) {
                self.request(&format!("{name} received"));
            }
        }
        self.reason
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Checks the local `STOP` file in `local_dir` and the remote `<prefix>/STOP` object; either
    /// one requests a stop, with the reason written in it when it can be read.
    pub fn poll(&self, remote: Option<(&dyn Remote, &str)>, local_dir: &Path) {
        if self.requested().is_some() {
            return;
        }
        let local = local_dir.join("STOP");
        if local.is_file() {
            let why = std::fs::read_to_string(&local).unwrap_or_default();
            self.request(&format!("STOP file {} {}", local.display(), why.trim()));
            return;
        }
        if let Some((r, prefix)) = remote {
            let key = format!("{}/STOP", prefix.trim_end_matches('/'));
            if matches!(r.exists(&key), Ok(true)) {
                let tmp = local_dir.join("tmp").join("STOP.remote");
                let why = match r.get(&key, &tmp) {
                    Ok(true) => std::fs::read_to_string(&tmp).unwrap_or_default(),
                    _ => String::new(),
                };
                let why: String = why.trim().chars().take(300).collect();
                self.request(&format!("the launcher asked the run to stop ({key}) {why}"));
            }
        }
    }
}

/// The process's stop state.
pub fn global() -> &'static StopState {
    static S: OnceLock<StopState> = OnceLock::new();
    S.get_or_init(StopState::default)
}

/// Turns SIGTERM, SIGINT and SIGHUP into stop requests (Unix; elsewhere nothing to do).
pub fn install() {
    #[cfg(unix)]
    {
        use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};
        for ((_, f), sig) in global().signals.iter().zip([SIGTERM, SIGINT, SIGHUP]) {
            if let Err(e) = signal_hook::flag::register(sig, Arc::clone(f)) {
                eprintln!("runner: cannot watch signal {sig}: {e}");
            }
        }
    }
}

/// [`StopState::requested`] on the process's state.
pub fn requested() -> Option<String> {
    global().requested()
}

/// [`StopState::request`] on the process's state.
pub fn request(reason: &str) {
    global().request(reason);
}

/// [`StopState::poll`] on the process's state.
pub fn poll(remote: Option<(&dyn Remote, &str)>, local_dir: &Path) {
    global().poll(remote, local_dir);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::DirRemote;

    /// The launcher's remote request, a local STOP file and a signal each stop a run; the first
    /// reason is kept.
    #[test]
    fn stop_requests() {
        let base = std::env::temp_dir().join(format!("runner-stop-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let work = base.join("work");
        std::fs::create_dir_all(&work).expect("mkdir");
        let remote = DirRemote {
            root: base.join("remote"),
        };
        let st = StopState::default();
        st.poll(Some((&remote, "runs/x")), &work);
        assert_eq!(st.requested(), None);
        let f = base.join("why");
        std::fs::write(&f, "{\"reason\": \"low CPU\"}").expect("write");
        remote.put(&f, "runs/x/STOP", false).expect("put");
        st.poll(Some((&remote, "runs/x")), &work);
        let why = st.requested().expect("stop requested");
        assert!(
            why.contains("runs/x/STOP") && why.contains("low CPU"),
            "{why}"
        );
        st.request("later reason");
        assert_eq!(st.requested(), Some(why));
        let st = StopState::default();
        std::fs::write(work.join("STOP"), "by hand").expect("write");
        st.poll(None, &work);
        assert!(st.requested().expect("stop").contains("by hand"));
        let st = StopState::default();
        st.signals[0].1.store(true, Ordering::Relaxed);
        assert_eq!(st.requested(), Some("SIGTERM received".to_string()));
        let _ = std::fs::remove_dir_all(&base);
    }
}
