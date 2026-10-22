//! `resume_worlds` (docs/M3-SPEC.md §7.3): a `search_measure` chunk killed with SIGKILL mid-way,
//! while some decisions are only part-done, continues on a "new machine" (an empty work
//! directory) world by world from what it had uploaded, and ends with a worlds file
//! byte-identical to an uninterrupted run's:
//! - format 1: the killed run uploads `partial/out/` every second; a second run with
//!   `resume_from` set to the first downloads it and continues;
//! - format 2: the killed run uploads its running chunk's files to the chunk's live prefix every
//!   second; the same manifest started again (the launcher's replacement) restores them.
//!
//! In both, the resumed run must value exactly the worlds that were not in the uploaded copy
//! (`resumed.worlds_valued` = all worlds - the saved ones), so only the worlds in flight (or
//! finished after the last upload) are redone. A resume under other settings is refused: another
//! seed, and an uploaded copy made with another equity table, both end FAILED with a message
//! naming the field, and nothing is valued. Like `resume_identity` it needs release binaries,
//! so it is `#[ignore]`d in the plain `cargo test --workspace`:
//! ```text
//! cargo build --release -p runner -p search -p equity -p difftest
//! RUNNER_TEST_BIN_DIR=target/release cargo test --release -p runner --test resume_worlds -- --ignored --nocapture
//! ```

// The tests below run on Linux only (they need release binaries and `kill -9`); elsewhere their
// helpers and imports are unused, which `-D warnings` in CI would turn into errors.
#![cfg_attr(not(target_os = "linux"), allow(dead_code, unused_imports))]

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const EXE: &str = std::env::consts::EXE_SUFFIX;
const WORLDS: &str = "measure/exact-t2-n4.jsonl";
const NW: u64 = 4;

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

/// Exact trick-2 worlds of decisions 4–15 (positions 2–4 of the trick, seconds per world), 4
/// worlds each, on 2 threads.
fn measure_params() -> Value {
    json!({ "trick": 2, "count": 32, "seed": 1, "to": NW, "grid": "2,4", "first": 4, "last": 15,
            "start_hands": 2000 })
}

fn start(bins: &Path, dir: &Path, work: &str, manifest: &Value) -> Child {
    fs::create_dir_all(dir).unwrap();
    let m = dir.join(format!("{work}.json"));
    fs::write(&m, manifest.to_string()).unwrap();
    let log = fs::File::create(dir.join(format!("{work}.log"))).unwrap();
    Command::new(bins.join(format!("runner{EXE}")))
        .arg("run")
        .arg(&m)
        .arg("--work-dir")
        .arg(dir.join(work))
        .env("RUNNER_REMOTE_DIR", dir.join("remote"))
        .env("RUNNER_NO_DESTROY", "1")
        .env_remove("RCLONE_CONFIG_R2_TYPE")
        .env_remove("CONTAINER_ID")
        .stdin(Stdio::null())
        .stdout(log.try_clone().unwrap())
        .stderr(log)
        .spawn()
        .unwrap()
}

fn wait_exit(child: &mut Child, limit_s: u64) -> std::process::ExitStatus {
    let t0 = Instant::now();
    loop {
        if let Some(st) = child.try_wait().unwrap() {
            return st;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(limit_s),
            "the run did not end"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// `pid` and every process below it (Linux `/proc`).
fn tree(pid: u32) -> Vec<u32> {
    let mut all = vec![pid];
    let mut i = 0;
    while i < all.len() {
        let p = all[i];
        let kids = fs::read_to_string(format!("/proc/{p}/task/{p}/children")).unwrap_or_default();
        all.extend(
            kids.split_whitespace()
                .filter_map(|k| k.parse::<u32>().ok()),
        );
        i += 1;
    }
    all
}

/// SIGKILL to the runner and everything it started (a lost machine).
fn kill9(child: &mut Child) {
    for p in tree(child.id()) {
        let _ = Command::new("kill")
            .args(["-KILL", &p.to_string()])
            .status();
    }
    let _ = child.wait();
}

fn lines(p: &Path) -> Vec<Value> {
    fs::read_to_string(p)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// Worlds saved in a worlds file and its checkpoint: (complete decisions × 4 + checkpointed
/// worlds of the other decisions, part-done decisions).
fn saved(worlds: &Path) -> (u64, usize) {
    let done: BTreeSet<u64> = lines(worlds)
        .iter()
        .filter_map(|r| r["id"].as_u64())
        .collect();
    let mut ck = PathBuf::from(worlds);
    ck.as_mut_os_string().push(".ckpt");
    let part: BTreeSet<(u64, u64)> = lines(&ck)
        .iter()
        .filter_map(|r| Some((r["id"].as_u64()?, r["w"].as_u64()?)))
        .filter(|(id, _)| !done.contains(id))
        .collect();
    let decisions: BTreeSet<u64> = part.iter().map(|p| p.0).collect();
    (done.len() as u64 * NW + part.len() as u64, decisions.len())
}

/// Waits until the uploaded copy at `remote_out` holds a complete decision and a part-done one.
fn wait_part_done(child: &mut Child, remote_out: &Path) {
    let t0 = Instant::now();
    loop {
        assert!(
            child.try_wait().unwrap().is_none(),
            "the run ended before it could be killed"
        );
        let w = remote_out.join(WORLDS);
        let (n, part) = saved(&w);
        if !lines(&w).is_empty() && part > 0 && n >= 2 * NW {
            return;
        }
        assert!(t0.elapsed() < Duration::from_secs(300), "no progress");
        std::thread::sleep(Duration::from_millis(100));
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
        out.push((name, data[at + 512..at + 512 + size].to_vec()));
        at += 512 + size.div_ceil(512) * 512;
    }
    out
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "needs release binaries (see the module docs)"]
fn kill9_then_resume_world_by_world() {
    let bins = bin_dir();
    let dir = std::env::temp_dir().join(format!("runner-rw-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let total = 12 * NW;
    let f1 = |run: &str, extra: Value| {
        let mut m = json!({ "run_id": run, "benchmark_seconds": 0, "threads": 2,
                            "heartbeat_every_s": 1, "upload_every_s": 1,
                            "chunk": { "search_measure": measure_params() } });
        if let (Some(o), Some(e)) = (m.as_object_mut(), extra.as_object()) {
            o.extend(e.clone());
        }
        m
    };

    // The reference: one uninterrupted run.
    let t0 = Instant::now();
    let mut a = start(&bins, &dir, "ref", &f1("rw-ref", json!({})));
    assert!(wait_exit(&mut a, 600).success());
    let want = fs::read(dir.join("ref/rw-ref/out").join(WORLDS)).unwrap();
    println!(
        "reference: {} decisions, {total} worlds in {:.0} s",
        lines(&dir.join("ref/rw-ref/out").join(WORLDS)).len(),
        t0.elapsed().as_secs_f64()
    );

    // Format 1: killed mid-way, then a new run with resume_from.
    let mut b = start(&bins, &dir, "b", &f1("rw-b", json!({})));
    let up = dir.join("remote/runs/rw-b/partial/out");
    wait_part_done(&mut b, &up);
    kill9(&mut b);
    let (uploaded, part) = saved(&up.join(WORLDS));
    let (local, _) = saved(&dir.join("b/rw-b/out").join(WORLDS));
    let mut c = start(
        &bins,
        &dir,
        "c",
        &f1("rw-c", json!({ "resume_from": "rw-b" })),
    );
    assert!(wait_exit(&mut c, 600).success());
    let got = fs::read(dir.join("c/rw-c/out").join(WORLDS)).unwrap();
    assert!(got == want, "format 1: the resumed worlds file differs");
    let res: Value =
        serde_json::from_str(&fs::read_to_string(dir.join("c/rw-c/result.json")).unwrap()).unwrap();
    let r = &res["resumed"];
    println!(
        "format 1: killed with {local} of {total} worlds done locally, {uploaded} in the uploaded \
         copy ({part} decisions part-done); the resumed run restored {}, valued {}; worlds file \
         byte-identical ({} bytes)",
        r["worlds"],
        r["worlds_valued"],
        got.len()
    );
    assert_eq!(r["worlds"], json!(uploaded), "{res}");
    assert_eq!(r["worlds_valued"], json!(total - uploaded), "{res}");

    // A resume under other settings is refused, by name, before anything is valued: another
    // seed, and an uploaded copy whose fingerprint names another equity table.
    let refused = |work: &str, run: &str, m: &Value, word: &str| {
        let mut x = start(&bins, &dir, work, m);
        assert!(!wait_exit(&mut x, 600).success(), "{run} should fail");
        let log = fs::read_to_string(dir.join(format!("{work}.log"))).unwrap();
        assert!(
            log.contains("resume refused") && log.contains(word),
            "{run}: the log does not name {word}:\n{log}"
        );
        assert!(
            dir.join(format!("remote/runs/{run}/FAILED")).is_file(),
            "{run}: no FAILED marker"
        );
        assert!(
            !dir.join(format!("{work}/{run}/out")).join(WORLDS).exists()
                || lines(&dir.join(format!("{work}/{run}/out")).join(WORLDS)).len()
                    == lines(&up.join(WORLDS)).len(),
            "{run}: valued worlds after refusing"
        );
    };
    let mut other_seed = f1("rw-c-seed", json!({ "resume_from": "rw-b" }));
    other_seed["chunk"]["search_measure"]["seed"] = json!(2);
    refused("c-seed", "rw-c-seed", &other_seed, "seed");
    let fp_path = up.join("measure/fingerprint.json");
    let mut fp: Value = serde_json::from_str(&fs::read_to_string(&fp_path).unwrap()).unwrap();
    let real_table = fp["table_sha256"].clone();
    fp["table_sha256"] = json!("0000000000000000000000000000000000000000000000000000000000000000");
    fs::write(&fp_path, fp.to_string()).unwrap();
    let m = f1("rw-c-table", json!({ "resume_from": "rw-b" }));
    refused("c-table", "rw-c-table", &m, "table_sha256");
    fp["table_sha256"] = real_table;
    fs::write(&fp_path, fp.to_string()).unwrap();
    println!(
        "format 1: a resume with another seed, and from a copy made with another table, were refused"
    );

    // Format 2: killed mid-way, then the same manifest on an empty work directory.
    let mut p = measure_params();
    p["threads"] = json!(2);
    let m2 = json!({
        "format": 2, "run_id": "rw-f2", "run_seed": 7, "threads": 2,
        "upload_every_s": 1, "heartbeat_every_s": 1, "benchmark_seconds": 0,
        "streams": [ { "name": "m", "priority": 1, "kind": "search_measure",
                       "params": p, "chunks": 1 } ] });
    let mut d = start(&bins, &dir, "d1", &m2);
    let live = dir.join("remote/runs/rw-f2/partial/m/000000/out");
    wait_part_done(&mut d, &live);
    kill9(&mut d);
    let (uploaded, part) = saved(&live.join(WORLDS));
    let mut e = start(&bins, &dir, "d2", &m2);
    assert!(wait_exit(&mut e, 600).success());
    let outs: Vec<PathBuf> = fs::read_dir(dir.join("remote/runs/rw-f2/out/m"))
        .unwrap()
        .flatten()
        .map(|x| x.path())
        .collect();
    assert_eq!(outs.len(), 1, "{outs:?}");
    let tar = zstd::stream::decode_all(&fs::read(&outs[0]).unwrap()[..]).unwrap();
    let members = untar(&tar);
    let file = |name: &str| {
        members
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, b)| b.clone())
            .unwrap_or_else(|| panic!("{name} is not in the bundle"))
    };
    let got = file(WORLDS);
    assert!(got == want, "format 2: the resumed worlds file differs");
    let res: Value = serde_json::from_slice(&file("result.json")).unwrap();
    let r = &res["resumed"];
    println!(
        "format 2: killed with {uploaded} of {total} worlds in the uploaded copy ({part} \
         decisions part-done); the replacement restored {}, valued {}; worlds file byte-identical",
        r["worlds"], r["worlds_valued"]
    );
    assert_eq!(r["worlds"], json!(uploaded), "{res}");
    assert_eq!(r["worlds_valued"], json!(total - uploaded), "{res}");
    assert!(
        !members.iter().any(|(n, _)| n.ends_with(".ckpt")),
        "no checkpoint in the finished bundle"
    );
    let _ = fs::remove_dir_all(&dir);
}
