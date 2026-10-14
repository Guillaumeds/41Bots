//! `resume_identity` (docs/M3-SPEC.md section 7.4): a format-2 run killed at named points and
//! restarted gives byte-identical outputs and merged report to an uninterrupted run, and lists
//! no chunk twice.
//!
//! The remote is a local directory (`RUNNER_REMOTE_DIR`, the runner's `DirRemote`, behind the
//! same `Remote` trait as the rclone backend). The kill is `Child::kill` (SIGKILL on Unix,
//! `TerminateProcess` on Windows) once the runner reports reaching the named point.
//!
//! It needs release binaries (`runner`, `search`, `equity`, `difftest`; the teacher is far too
//! slow in a debug build), so it is `#[ignore]`d in the plain `cargo test --workspace`. CI runs:
//! ```text
//! cargo build --release -p runner -p search -p equity -p difftest
//! RUNNER_TEST_BIN_DIR=target/release cargo test --release -p runner --test resume_identity -- --ignored
//! ```
//! (`RUNNER_TEST_BIN_DIR` may be relative to the workspace root; without it `target/release`.)
//!
//! The third stream is `difftest` standing in for `suitability` (2 chunks), because the
//! `search suitability` binary (section 6) does not exist yet; swap it in when it does.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const RUN: &str = "resume-identity";
const EXE: &str = std::env::consts::EXE_SUFFIX;

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn bin_dir() -> PathBuf {
    let names = ["runner", "search", "equity", "difftest"];
    let d = std::env::var("RUNNER_TEST_BIN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/release"));
    let d = if d.is_absolute() {
        d
    } else {
        workspace().join(d)
    };
    for n in names {
        assert!(
            d.join(format!("{n}{EXE}")).is_file(),
            "{} lacks {n}: build release binaries first \
             (cargo build --release -p runner -p search -p equity -p difftest)",
            d.display()
        );
    }
    d
}

fn fxhash(b: &[u8]) -> u128 {
    // Only used to make failure messages short; equality is checked on the bytes themselves.
    b.iter().fold(0u128, |h, &x| {
        (h.rotate_left(5) ^ u128::from(x)).wrapping_mul(0x100_0000_01b3)
    })
}

fn files(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(base: &Path, d: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        let Ok(rd) = fs::read_dir(d) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(base, &p, out);
            } else {
                let rel = p
                    .strip_prefix(base)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                if !rel.contains(".partial-") {
                    out.insert(rel, fs::read(&p).unwrap());
                }
            }
        }
    }
    let mut m = BTreeMap::new();
    walk(root, root, &mut m);
    m
}

fn manifest(eq_hex: &str) -> String {
    serde_json::json!({
        "format": 2, "run_id": RUN, "run_seed": 41, "threads": 4,
        "upload_every_s": 0, "heartbeat_every_s": 5, "benchmark_seconds": 0,
        "inputs": { "equity": format!("sha256:{eq_hex}") },
        "streams": [
            { "name": "teacher", "priority": 1, "kind": "teacher",
              "params": { "hands_per_chunk": 2, "config": "pimc-teacher" }, "chunks": 6 },
            { "name": "gate", "priority": 2, "kind": "arena_blocks",
              "params": { "a": "rules-v1", "b": "random", "blocks_per_chunk": 1,
                          "sprt": { "s0": 0.0, "s1": 0.02, "max_blocks": 3 } },
              "chunks": "until_decided" },
            // Stands in for `suitability` until `search suitability` exists (section 6).
            { "name": "diff", "priority": 3, "kind": "difftest",
              "params": { "matches": 30 }, "chunks": 2 }
        ]
    })
    .to_string()
}

/// Runs the runner; with `kill_at`, kills it once it reports reaching that point.
fn run(bins: &Path, dir: &Path, kill_at: Option<&str>) {
    let flag = dir.join("killflag");
    let _ = fs::remove_file(&flag);
    let mut c = Command::new(bins.join(format!("runner{EXE}")));
    c.arg("run")
        .arg(dir.join("m.json"))
        .arg("--work-dir")
        .arg(dir.join("work"))
        .env("RUNNER_REMOTE_DIR", dir.join("remote"))
        .env("RUNNER_NO_DESTROY", "1")
        .env_remove("RCLONE_CONFIG_R2_TYPE")
        .env_remove("CONTAINER_ID")
        .stdin(Stdio::null())
        .stdout(Stdio::null());
    if let Some(k) = kill_at {
        c.env("RUNNER_KILL_AT", k).env("RUNNER_KILL_FLAG", &flag);
    }
    let mut child = c.spawn().unwrap();
    let t0 = Instant::now();
    loop {
        if let Some(st) = child.try_wait().unwrap() {
            assert!(
                kill_at.is_none(),
                "runner exited ({st}) before kill point {kill_at:?}"
            );
            assert!(st.success(), "runner failed: {st}");
            return;
        }
        if kill_at.is_some() && flag.exists() {
            child.kill().unwrap();
            child.wait().unwrap();
            // Its chunk workers notice the closed pipe and exit within ~2 s.
            std::thread::sleep(Duration::from_secs(3));
            return;
        }
        assert!(
            t0.elapsed() < Duration::from_secs(300),
            "runner took too long"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn setup(base: &Path, name: &str, table: &Path, hex: &str) -> PathBuf {
    let d = base.join(name);
    let inp = d.join("remote").join("inputs").join("sha256");
    fs::create_dir_all(&inp).unwrap();
    fs::copy(table, inp.join(hex)).unwrap();
    fs::write(d.join("m.json"), manifest(hex)).unwrap();
    d
}

#[test]
#[ignore = "needs release binaries; CI runs it with --ignored (see the file's docs)"]
fn resume_identity() {
    let bins = bin_dir();
    let base = std::env::temp_dir().join(format!("runner-resume-{}", std::process::id()));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    // The production equity table, rebuilt from the committed model (about 8 s).
    let table = base.join("equity-v0.npy");
    let st = Command::new(bins.join(format!("equity{EXE}")))
        .arg("build")
        .arg("--model")
        .arg(workspace().join("data/equity/model-rules-v1-1M-seed1.json"))
        .arg("--out")
        .arg(&table)
        .args(["--threads", "4"])
        .stdout(Stdio::null())
        .status()
        .unwrap();
    assert!(st.success());
    let hex = fs::read_to_string(workspace().join("data/equity/equity-v0-rules-v1.sha256"))
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .to_string();

    // 1. Uninterrupted.
    let a = setup(&base, "a", &table, &hex);
    run(&bins, &a, None);
    let out_a = files(&a.join("remote").join("runs").join(RUN).join("out"));
    let merged_a = fs::read(a.join("remote/runs").join(RUN).join("merged.json")).unwrap();
    assert_eq!(out_a.len(), 6 + 3 + 2, "{:?}", out_a.keys());

    // 2. Killed after chunk 2, mid-upload, between upload and done-list, after chunk 5.
    let b = setup(&base, "b", &table, &hex);
    for k in [
        "after_done:2",
        "mid_upload:4",
        "after_upload:5",
        "after_done:5",
    ] {
        run(&bins, &b, Some(k));
    }
    // 3. Restart to the end.
    run(&bins, &b, None);
    let out_b = files(&b.join("remote").join("runs").join(RUN).join("out"));
    let merged_b = fs::read(b.join("remote/runs").join(RUN).join("merged.json")).unwrap();
    let short = |m: &BTreeMap<String, Vec<u8>>| {
        m.iter()
            .map(|(k, v)| (k.clone(), format!("{}:{:x}", v.len(), fxhash(v))))
            .collect::<Vec<_>>()
    };
    assert_eq!(short(&out_a), short(&out_b));
    assert_eq!(out_a, out_b, "bundles differ");
    assert_eq!(
        String::from_utf8_lossy(&merged_a),
        String::from_utf8_lossy(&merged_b)
    );

    // Both runs finished cleanly and the report covers every chunk.
    for d in [&a, &b] {
        assert!(d.join("remote/runs").join(RUN).join("DONE").is_file());
        assert!(
            d.join("remote/runs")
                .join(RUN)
                .join("final-manifest.json")
                .is_file()
        );
    }
    let merged: serde_json::Value = serde_json::from_slice(&merged_a).unwrap();
    assert_eq!(merged["complete"], true);
    let gate = &merged["streams"][1];
    assert_eq!(
        (gate["name"].as_str(), gate["chunks_done"].as_u64()),
        (Some("gate"), Some(3))
    );
    assert_eq!(gate["sprt"]["decision"], "Inconclusive"); // 3 blocks: the cap, below 20

    // Done-list versions have distinct, increasing sequence numbers (one per write, across
    // restarts), and no chunk is listed twice in the last one.
    let done = files(&b.join("remote").join("done").join(RUN));
    let seqs: Vec<u64> = done
        .keys()
        .map(|k| k[..8].parse::<u64>().unwrap())
        .collect();
    assert!(seqs.windows(2).all(|w| w[0] < w[1]), "{seqs:?}");
    let (_, last) = done.iter().next_back().unwrap();
    let doc: serde_json::Value = serde_json::from_slice(last).unwrap();
    let entries = doc["entries"].as_array().unwrap();
    let keys: BTreeSet<(String, u64)> = entries
        .iter()
        .map(|e| {
            (
                e["stream"].as_str().unwrap().to_string(),
                e["index"].as_u64().unwrap(),
            )
        })
        .collect();
    assert_eq!(keys.len(), entries.len(), "a chunk is listed twice");
    assert_eq!(keys.len(), 11);
    let _ = fs::remove_dir_all(&base);
}
