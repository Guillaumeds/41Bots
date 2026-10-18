//! `student_train` end to end (docs/STUDENT.md §5, §7): real programs on a local remote store.
//!
//! 1. A format-2 run of `student_teacher` streams (a random O_3 so it takes seconds: a test part
//!    and four training parts, three rules and one mixed) shares its shards content-addressed and
//!    leaves `runs/<run>/merged.json` and its bundles, as the rented runs do in R2.
//! 2. A format-2 run of one `student_train` chunk names those streams by run and stream, fetches
//!    and encodes the parts its sizes need (in the spread order), installs the CPU Python
//!    environment (`uv sync --locked`), trains two widths at two sizes, measures each integer
//!    network with `search student-check` (rules and random baselines), shares the networks
//!    content-addressed and writes the curve.
//! 3. The same training under a new run id whose live copy holds one finished pair (as a lost
//!    machine leaves it): that pair is not trained again, the others are.
//!
//! It needs release binaries and `uv` (with PyTorch in its cache, or the network), so it is
//! `#[ignore]`d in the plain `cargo test --workspace`:
//! ```text
//! cargo build --release -p runner -p search -p student -p equity -p nn
//! RUNNER_TEST_BIN_DIR=target/release cargo test --release -p runner --test student_train -- --ignored --nocapture
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

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
    for n in ["runner", "search", "student", "equity", "nn-tool"] {
        assert!(
            d.join(format!("{n}{EXE}")).is_file(),
            "{} lacks {n}: build release binaries first",
            d.display()
        );
    }
    d
}

fn sha256(p: &Path) -> String {
    let o = Command::new("sha256sum").arg(p).output().unwrap();
    String::from_utf8_lossy(&o.stdout)
        .split_whitespace()
        .next()
        .unwrap()
        .to_string()
}

/// Puts `file` into the store content-addressed; its manifest value.
fn put_input(remote: &Path, file: &Path) -> String {
    let hex = sha256(file);
    let d = remote.join("inputs/sha256");
    fs::create_dir_all(&d).unwrap();
    fs::copy(file, d.join(&hex)).unwrap();
    format!("sha256:{hex}")
}

fn run(bins: &Path, dir: &Path, work: &str, manifest: &Value) -> String {
    let m = dir.join(format!("{work}.json"));
    fs::write(&m, manifest.to_string()).unwrap();
    let t0 = Instant::now();
    let out = Command::new(bins.join(format!("runner{EXE}")))
        .arg("run")
        .arg(&m)
        .arg("--work-dir")
        .arg(dir.join(work))
        .env("RUNNER_REMOTE_DIR", dir.join("remote"))
        .env("RUNNER_NO_DESTROY", "1")
        .env("RUNNER_PYTHON_DIR", workspace().join("python/nn"))
        .env_remove("RCLONE_CONFIG_R2_TYPE")
        .env_remove("CONTAINER_ID")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    fs::write(dir.join(format!("{work}.log")), &log).unwrap();
    println!("{work}: {:.0} s", t0.elapsed().as_secs_f64());
    let run_id = manifest["run_id"].as_str().unwrap();
    assert!(
        out.status.success() && dir.join(format!("remote/runs/{run_id}/DONE")).is_file(),
        "{work} failed:\n{log}"
    );
    log
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

/// The members of the one bundle of `runs/<run>/out/<stream>/`.
fn bundle(remote: &Path, run: &str, stream: &str) -> Vec<(String, Vec<u8>)> {
    let outs: Vec<PathBuf> = fs::read_dir(remote.join(format!("runs/{run}/out/{stream}")))
        .unwrap()
        .flatten()
        .map(|x| x.path())
        .collect();
    assert_eq!(outs.len(), 1, "{outs:?}");
    untar(&zstd::stream::decode_all(&fs::read(&outs[0]).unwrap()[..]).unwrap())
}

fn member(m: &[(String, Vec<u8>)], name: &str) -> Vec<u8> {
    m.iter()
        .find(|(n, _)| n == name)
        .map(|(_, b)| b.clone())
        .unwrap_or_else(|| panic!("{name} is not in the bundle"))
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "needs release binaries and uv (see the module docs)"]
fn student_train_end_to_end() {
    let bins = bin_dir();
    let dir = std::env::temp_dir().join(format!("runner-st-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let remote = dir.join("remote");
    // The equity table v0 (from the repository's model: its recorded SHA-256) and a random O_3.
    let table = dir.join("equity.npy");
    let st = Command::new(bins.join(format!("equity{EXE}")))
        .arg("build")
        .arg("--model")
        .arg(workspace().join("data/equity/model-rules-v1-1M-seed1.json"))
        .arg("--out")
        .arg(&table)
        .args(["--threads", "4"])
        .output()
        .unwrap();
    assert!(st.status.success());
    let o3 = dir.join("o3.safetensors");
    let st = Command::new(bins.join(format!("nn-tool{EXE}")))
        .args(["random", "--trick", "3", "--widths", "32,16", "--out"])
        .arg(&o3)
        .output()
        .unwrap();
    assert!(st.status.success());
    let inputs = json!({ "equity": put_input(&remote, &table), "o3": put_input(&remote, &o3) });

    // 1. The teacher data.
    let teach = |name: &str,
                 prio: u64,
                 chunks: u64,
                 count: u64,
                 deals: u64,
                 seed: u64,
                 source: &str,
                 role: &str| {
        json!({ "name": name, "priority": prio, "kind": "student_teacher", "chunks": chunks,
                "params": { "threads": "all", "trick": 2, "count": count, "deals": deals,
                            "seed": seed, "source": source, "role": role, "net_input": "o3",
                            "start_hands": 400 } })
    };
    let m1 = json!({
        "format": 2, "run_id": "st-teach-1", "run_seed": 1, "threads": 4,
        "upload_every_s": 1, "heartbeat_every_s": 1, "benchmark_seconds": 0, "inputs": inputs,
        "streams": [ teach("test", 1, 1, 24, 8, 32343, "rules", "test"),
                     teach("rules", 2, 3, 30, 4, 9401, "rules", "train"),
                     teach("mixed", 3, 1, 30, 4, 9402, "mixed", "train") ] });
    run(&bins, &dir, "teach", &m1);

    // 2. The training: sizes 40 and 100 (two and four parts in the spread order), two widths.
    let train = |run_id: &str| {
        json!({
            "format": 2, "run_id": run_id, "run_seed": 1, "threads": 4,
            "upload_every_s": 1, "heartbeat_every_s": 1, "benchmark_seconds": 0,
            "inputs": inputs,
            "streams": [ { "name": "curve", "priority": 1, "kind": "student_train", "chunks": 1,
                "params": { "threads": "all",
                    "train_inputs": [ { "run": "st-teach-1", "stream": "rules", "parts": 3 },
                                      { "run": "st-teach-1", "stream": "mixed", "parts": 1 } ],
                    "test_inputs": [ { "run": "st-teach-1", "stream": "test", "parts": 1 } ],
                    "sizes": [40, 100], "widths": ["16,8", "8"], "epochs": 2, "batch": 16,
                    "val_permille": 200, "device": "cpu", "train_threads": 2,
                    "fetch_jobs": 2, "boot": 100 } } ] })
    };
    run(&bins, &dir, "train", &train("st-train-1"));
    let b = bundle(&remote, "st-train-1", "curve");
    let curve: Value = serde_json::from_slice(&member(&b, "student/curve.json")).unwrap();
    println!(
        "{}",
        String::from_utf8_lossy(&member(&b, "student/curve.txt"))
    );
    let rows = curve["rows"].as_array().unwrap();
    let pairs: Vec<(u64, &str)> = rows
        .iter()
        .map(|r| (r["size"].as_u64().unwrap(), r["widths"].as_str().unwrap()))
        .collect();
    assert_eq!(
        pairs,
        vec![(40, "16,8"), (40, "8"), (100, "16,8"), (100, "8")]
    );
    for r in rows {
        assert_eq!(r["test"]["n"], 24, "{r}");
        assert_eq!(
            r["train_decisions"].as_u64().unwrap() + r["val_decisions"].as_u64().unwrap(),
            r["size"].as_u64().unwrap()
        );
        assert!(r["by_group"]["place 1"]["n"].as_u64().unwrap() > 0, "{r}");
        assert!(r["by_group"]["stratum S4"]["n"].as_u64().is_some(), "{r}");
        // The integer network is shared content-addressed.
        let key = r["int_key"].as_str().unwrap();
        assert!(remote.join(key).is_file(), "{key}");
        assert_eq!(
            key,
            format!("inputs/sha256/{}", r["int_sha256"].as_str().unwrap())
        );
    }
    let bots: Vec<&str> = curve["baselines"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b["bot"].as_str().unwrap())
        .collect();
    assert_eq!(bots, vec!["rules-v1", "random"]);
    // The spread order: rules 0, rules 1, mixed 0, rules 2; size 40 used two parts.
    let used: Value = serde_json::from_slice(&member(&b, "student/inputs.json")).unwrap();
    let labels: Vec<&str> = used["train"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["label"].as_str().unwrap())
        .collect();
    assert_eq!(
        labels,
        vec![
            "st-teach-1/rules/part 0",
            "st-teach-1/rules/part 1",
            "st-teach-1/mixed/part 0",
            "st-teach-1/rules/part 2"
        ]
    );
    let done40: Value = serde_json::from_slice(&member(&b, "student/n40-w16x8/done.json")).unwrap();
    assert_eq!(done40["parts"], 2);
    let check: Value = serde_json::from_slice(&member(&b, "student/n40-w16x8/check.json")).unwrap();
    assert_eq!(check["role"], "test");
    assert_eq!(check["baselines"].as_array().unwrap().len(), 2);

    // 3. A resume: the live copy of a new run holds the finished pair n40-w16x8 (as a lost
    // machine leaves it); the replacement does not train it again.
    let live = remote.join("runs/st-train-2/partial/curve/000000/out/student/n40-w16x8");
    fs::create_dir_all(&live).unwrap();
    for (n, data) in &b {
        if let Some(f) = n.strip_prefix("student/n40-w16x8/") {
            fs::write(live.join(f), data).unwrap();
        }
    }
    let log = run(&bins, &dir, "train2", &train("st-train-2"));
    let b2 = bundle(&remote, "st-train-2", "curve");
    let curve2: Value = serde_json::from_slice(&member(&b2, "student/curve.json")).unwrap();
    assert_eq!(curve2["rows"].as_array().unwrap().len(), 4);
    assert_eq!(curve2["rows"][0]["int_sha256"], rows[0]["int_sha256"]);
    let worker_logs = dir.join("train2");
    let mut said = log.contains("n40-w16x8 was finished before");
    if !said {
        // The worker's own log (where the chunk prints) is kept beside the chunk.
        for e in walk(&worker_logs) {
            if e.ends_with("worker.log")
                && fs::read_to_string(&e)
                    .unwrap_or_default()
                    .contains("n40-w16x8 was finished before")
            {
                said = true;
            }
        }
    }
    let logs = remote.join("runs/st-train-2/logs/curve");
    said |= logs.is_dir()
        && walk(&logs).iter().any(|p| {
            zstd::stream::decode_all(&fs::read(p).unwrap()[..])
                .ok()
                .is_some_and(|t| {
                    untar(&t).iter().any(|(_, d)| {
                        String::from_utf8_lossy(d).contains("n40-w16x8 was finished before")
                    })
                })
        });
    assert!(said, "the resumed pair was trained again");
    let _ = fs::remove_dir_all(&dir);
}

fn walk(d: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(d) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(walk(&p));
            } else {
                out.push(p);
            }
        }
    }
    out
}
