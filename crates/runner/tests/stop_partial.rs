//! `stop_partial` (docs/M3-SPEC.md §13.1 item 11): a run stopped mid-step keeps what it
//! finished. A `search_measure` chunk (exact trick-2 worlds: a few seconds each) is stopped once
//! its worlds file holds finished decisions:
//! - format 1, by SIGTERM to the runner (Unix): the runner stops its child, uploads `out/` (the
//!   finished decisions' lines included) and writes `FAILED` with `stopped` and the files;
//! - format 2, by the launcher's request (a `runs/<run>/STOP` object in the remote): the runner
//!   stops the chunk, uploads its unfinished `out/` to `runs/<run>/partial/…` and ends `DONE`
//!   with `stopped` and that bundle listed.
//!
//! Every uploaded worlds line must be a complete decision record. Like `resume_identity` it
//! needs release binaries, so it is `#[ignore]`d in the plain `cargo test --workspace`:
//! ```text
//! cargo build --release -p runner -p search -p equity -p difftest
//! RUNNER_TEST_BIN_DIR=target/release cargo test --release -p runner --test stop_partial -- --ignored
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const EXE: &str = std::env::consts::EXE_SUFFIX;

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn bin_dir() -> PathBuf {
    let d = std::env::var("RUNNER_TEST_BIN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/release"));
    let d = if d.is_absolute() {
        d
    } else {
        workspace().join(d)
    };
    for n in ["runner", "search", "equity"] {
        assert!(
            d.join(format!("{n}{EXE}")).is_file(),
            "{} lacks {n}: build release binaries first",
            d.display()
        );
    }
    d
}

/// A small exact trick-2 measurement: 32 decisions × 4 worlds, slow enough to be stopped
/// mid-way (leader worlds take seconds), on 2 threads.
fn measure_params() -> Value {
    json!({ "trick": 2, "count": 32, "seed": 1, "to": 4, "grid": "2,4",
            "start_hands": 2000, "threads": 2 })
}

fn start(bins: &Path, dir: &Path, manifest: &Value) -> Child {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("m.json"), manifest.to_string()).unwrap();
    Command::new(bins.join(format!("runner{EXE}")))
        .arg("run")
        .arg(dir.join("m.json"))
        .arg("--work-dir")
        .arg(dir.join("work"))
        .env("RUNNER_REMOTE_DIR", dir.join("remote"))
        .env("RUNNER_NO_DESTROY", "1")
        .env_remove("RCLONE_CONFIG_R2_TYPE")
        .env_remove("CONTAINER_ID")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
}

/// Complete lines in a worlds file (a line being written may be cut short).
fn complete_lines(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .collect()
}

/// Waits until `find()` gives a worlds file with at least `n` complete lines.
fn wait_for_lines(child: &mut Child, n: usize, find: impl Fn() -> Option<PathBuf>) {
    let t0 = Instant::now();
    loop {
        assert!(
            child.try_wait().unwrap().is_none(),
            "the run ended before it could be stopped"
        );
        if let Some(p) = find()
            && complete_lines(&fs::read_to_string(&p).unwrap_or_default()).len() >= n
        {
            return;
        }
        assert!(t0.elapsed() < Duration::from_secs(240), "no progress");
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn wait_exit(child: &mut Child) -> std::process::ExitStatus {
    let t0 = Instant::now();
    loop {
        if let Some(st) = child.try_wait().unwrap() {
            return st;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(120),
            "the stopped run did not end"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn check_rows(lines: &[Value]) {
    assert!(!lines.is_empty());
    for r in lines {
        assert_eq!(r["trick"], json!(2));
        assert_eq!(r["rows"].as_array().map(Vec::len), Some(4), "{r}");
    }
}

/// The members of an uncompressed ustar archive (name, bytes).
fn untar(data: &[u8]) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let mut at = 0;
    while at + 512 <= data.len() {
        let h = &data[at..at + 512];
        if h.iter().all(|&b| b == 0) {
            break;
        }
        let name = String::from_utf8_lossy(&h[..100])
            .trim_end_matches('\0')
            .to_string();
        let size_text = String::from_utf8_lossy(&h[124..136]);
        let size =
            usize::from_str_radix(size_text.trim_matches(|c: char| c == '\0' || c == ' '), 8)
                .unwrap();
        let body = data[at + 512..at + 512 + size].to_vec();
        out.push((name, body));
        at += 512 + size.div_ceil(512) * 512;
    }
    out
}

#[cfg(unix)]
#[test]
#[ignore = "needs release binaries (see the module docs)"]
fn format1_sigterm_uploads_finished_decisions() {
    let bins = bin_dir();
    let dir = std::env::temp_dir().join(format!("runner-stop1-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let run = "stop-f1";
    let mut p = measure_params();
    p.as_object_mut().unwrap().remove("threads");
    let manifest = json!({ "run_id": run, "benchmark_seconds": 0, "threads": 2,
                           "chunk": { "search_measure": p } });
    let mut child = start(&bins, &dir, &manifest);
    let local = dir
        .join("work")
        .join(run)
        .join("out/measure/exact-t2-n4.jsonl");
    wait_for_lines(&mut child, 2, || Some(local.clone()));
    let st = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(st.success());
    let st = wait_exit(&mut child);
    assert!(!st.success(), "a stopped run is not a success");
    let remote = dir.join("remote").join("runs").join(run);
    let marker: Value =
        serde_json::from_str(&fs::read_to_string(remote.join("FAILED")).unwrap()).unwrap();
    assert!(
        marker["stopped"]
            .as_str()
            .is_some_and(|s| s.contains("SIGTERM")),
        "{marker}"
    );
    assert_eq!(marker["partial"], json!(true));
    let listed: Vec<&str> = marker["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|o| o["path"].as_str())
        .collect();
    assert!(
        listed.contains(&"out/measure/exact-t2-n4.jsonl"),
        "{listed:?}"
    );
    let up = fs::read_to_string(remote.join("out/measure/exact-t2-n4.jsonl")).unwrap();
    let lines = complete_lines(&up);
    assert!(
        lines.len() >= 2,
        "{} finished decisions uploaded",
        lines.len()
    );
    check_rows(&lines);
    // What was uploaded is exactly what the machine had written.
    assert_eq!(up, fs::read_to_string(&local).unwrap());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "needs release binaries (see the module docs)"]
fn format2_stop_request_uploads_partial_chunk() {
    let bins = bin_dir();
    let dir = std::env::temp_dir().join(format!("runner-stop2-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let run = "stop-f2";
    let manifest = json!({
        "format": 2, "run_id": run, "run_seed": 7, "threads": 2,
        "upload_every_s": 0, "heartbeat_every_s": 1, "benchmark_seconds": 0,
        "streams": [ { "name": "m", "priority": 1, "kind": "search_measure",
                       "params": measure_params(), "chunks": 1 } ] });
    let mut child = start(&bins, &dir, &manifest);
    let chunks = dir.join("work").join(run).join("chunks").join("m");
    let find = || {
        let d = fs::read_dir(&chunks).ok()?.flatten().next()?.path();
        Some(d.join("out/measure/exact-t2-n4.jsonl"))
    };
    wait_for_lines(&mut child, 1, find);
    // The launcher's stop request.
    let remote = dir.join("remote").join("runs").join(run);
    fs::create_dir_all(&remote).unwrap();
    fs::write(remote.join("STOP"), r#"{"reason": "test"}"#).unwrap();
    let st = wait_exit(&mut child);
    assert!(st.success(), "a stopped format-2 run ends DONE: {st}");
    let done: Value =
        serde_json::from_str(&fs::read_to_string(remote.join("DONE")).unwrap()).unwrap();
    assert!(
        done["stopped"].as_str().is_some_and(|s| s.contains("STOP")),
        "{done}"
    );
    let partial = done["partial"].as_array().unwrap();
    assert_eq!(partial.len(), 1, "{done}");
    let key = partial[0]["path"].as_str().unwrap();
    assert!(
        key.starts_with(&format!("runs/{run}/partial/m/000000-")),
        "{key}"
    );
    let bundle = fs::read(dir.join("remote").join(key)).unwrap();
    let tar = zstd::stream::decode_all(&bundle[..]).unwrap();
    let members = untar(&tar);
    let (_, worlds) = members
        .iter()
        .find(|(n, _)| n.ends_with("measure/exact-t2-n4.jsonl"))
        .expect("the worlds file is in the partial bundle");
    let lines = complete_lines(&String::from_utf8_lossy(worlds));
    assert!(!lines.is_empty());
    check_rows(&lines);
    // Not in the done-list: a resumed run redoes the chunk from its seed.
    let merged: Value =
        serde_json::from_str(&fs::read_to_string(remote.join("merged.json")).unwrap()).unwrap();
    assert_eq!(merged["complete"], json!(false));
    assert_eq!(merged["streams"][0]["chunks_done"], json!(0));
    let _ = fs::remove_dir_all(&dir);
}
