//! Label shards shared as each is made (`upload_shards`, step 4 of the module notes), with the
//! fake programs of the step tests: the order of the fake `search`'s calls and of the uploads is
//! read from one log.

use super::*;
use crate::remote::DirRemote;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A remote that notes each label shard's upload (`.npy`) in the log of the fake programs, and
/// fails the first `fail` of them; everything else as `DirRemote`.
struct LogRemote {
    inner: DirRemote,
    log: PathBuf,
    fail: usize,
    seen: AtomicUsize,
}

impl Remote for LogRemote {
    fn describe(&self) -> String {
        self.inner.describe()
    }
    fn put(&self, local: &Path, key: &str, immutable: bool) -> Result<(), String> {
        let shard = local.extension().is_some_and(|e| e == "npy");
        let failed = shard && self.seen.fetch_add(1, Ordering::SeqCst) < self.fail;
        if shard {
            let name = local
                .file_name()
                .map_or(String::new(), |n| n.to_string_lossy().into_owned());
            let mut f = fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(&self.log)
                .expect("log");
            writeln!(f, "put {name} {}", if failed { "err" } else { "ok" }).expect("log");
        }
        if failed {
            return Err("the upload was killed at its time limit".into());
        }
        self.inner.put(local, key, immutable)
    }
    fn get(&self, key: &str, local: &Path) -> Result<bool, String> {
        self.inner.get(key, local)
    }
    fn list_sized(&self, prefix: &str) -> Result<Vec<(String, u64)>, String> {
        self.inner.list_sized(prefix)
    }
    fn sha256(&self, key: &str) -> Result<Option<(String, u64)>, String> {
        self.inner.sha256(key)
    }
    fn exists(&self, key: &str) -> Result<bool, String> {
        self.inner.exists(key)
    }
}

/// Runs a chunk making three shards (random, rules, validation) with sharing on; returns the
/// lines of the log that matter (`search labels` and `put`) and the run's root.
fn run(tag: &str, fail: usize) -> (Vec<String>, PathBuf, Fakes) {
    let f = fakes(tag);
    let remote: Arc<dyn Remote> = Arc::new(LogRemote {
        inner: DirRemote {
            root: f.base.join("remote"),
        },
        log: f.calls.clone(),
        fail,
        seen: AtomicUsize::new(0),
    });
    let params = parse(&json!({
        "check": false, "labels_random": 7, "labels_random_seed": 511, "labels_rules": 5,
        "labels_rules_seed": 512, "val_count": 3, "val_seed": 601, "widths": "16,8", "epochs": 2,
        "upload_shards": true }))
    .expect("params");
    // The fake `search` writes the same bytes for every shard, which would be stored once; this
    // one gives each shard its own bytes (its seed) and records their SHA-256 in the sidecar.
    let search = fake_script(
        &f.bin,
        "search-distinct",
        &format!(
            r#"'{real}' "$@"
if [ "$1" = labels ]; then
  while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; [ "$1" = --seed ] && seed=$2; shift; done
  echo "seed $seed" >> "$out"
  sha=$(sha256sum "$out" | cut -d' ' -f1)
  echo "{{\"shard_sha256\": \"$sha\", \"count\": 7}}" > "$out.json"
fi
"#,
            real = f.search.display()
        ),
    );
    let tools = Tools {
        search,
        uv: f.uv.clone(),
        python_dir: f.py.clone(),
        nvidia_smi: f.base.join("no-such-nvidia-smi"),
        remote: Some(remote),
    };
    let root = f.base.join("run");
    let mut ctx = Ctx::local("hybrid-test", &root, 4);
    chunk_with(&mut ctx, &params, &tools, &f.table, json!({})).expect("the chunk runs");
    let log = fs::read_to_string(&f.calls).expect("calls");
    let lines = log
        .lines()
        .filter(|l| l.starts_with("search labels") || l.starts_with("put "))
        .map(|l| {
            if let Some(i) = l.find("--source ") {
                format!("labels {}", l[i + 9..].split(' ').next().unwrap_or(""))
            } else {
                l.to_string()
            }
        })
        .collect();
    (lines, root, f)
}

/// Each shard is uploaded before the next one's labels begin, and `shards.json` ends as it
/// always did: training shards in order, the validation shard last.
#[cfg(unix)]
#[test]
fn a_shard_is_shared_before_the_next_is_made() {
    let (lines, root, f) = run("shareorder", 0);
    let want: Vec<String> = [
        "labels random",
        "put t4-random-s511.npy ok",
        "labels rules",
        "put t4-rules-s512.npy ok",
        "labels random",
        "put t4-val-s601.npy ok",
    ]
    .map(String::from)
    .to_vec();
    assert_eq!(lines, want);
    let doc = read_json(&root.join("out/hybrid/shards.json")).expect("shards.json");
    let files: Vec<&str> = doc["shards"]
        .as_array()
        .expect("shards")
        .iter()
        .map(|s| s["file"].as_str().expect("file"))
        .collect();
    assert_eq!(
        files,
        ["t4-random-s511.npy", "t4-rules-s512.npy", "t4-val-s601.npy"]
    );
    assert_eq!(doc["label_inputs"].as_array().map(Vec::len), Some(3));
    assert_eq!(doc["label_inputs_pending"], json!([]));
    assert_eq!(doc["manifest_inputs"].as_object().map(Map::len), Some(3));
    let _ = fs::remove_dir_all(&f.base);
}

/// A shard whose upload fails (all three attempts) does not stop the labels: the next shard is
/// made and shared, and the failed one is tried again at the chunk's end.
#[cfg(unix)]
#[test]
fn a_failed_upload_of_shard_1_does_not_block_shard_2() {
    let (lines, root, f) = run("sharefail", 3);
    let pos = |needle: &str| lines.iter().position(|l| l == needle).expect(needle);
    let failed = lines
        .iter()
        .filter(|l| *l == "put t4-random-s511.npy err")
        .count();
    assert_eq!(failed, 3, "{lines:?}");
    let rules = lines
        .iter()
        .position(|l| l == "labels rules")
        .expect("rules made");
    assert!(
        lines[..rules]
            .iter()
            .all(|l| l != "put t4-rules-s512.npy ok"),
        "{lines:?}"
    );
    assert!(pos("labels rules") < pos("put t4-rules-s512.npy ok"));
    // Shard 2 is shared before the validation shard's labels begin.
    let val_labels = lines
        .iter()
        .rposition(|l| l == "labels random")
        .expect("val made");
    assert!(pos("put t4-rules-s512.npy ok") < val_labels, "{lines:?}");
    // The retry at the end stores shard 1.
    assert_eq!(
        lines.last().map(String::as_str),
        Some("put t4-random-s511.npy ok")
    );
    let out = root.join("out/hybrid");
    let rung = read_json(&out.join("rung.json")).expect("rung.json");
    assert!(rung["warnings"].to_string().contains("was not uploaded ("));
    let doc = read_json(&out.join("shards.json")).expect("shards.json");
    assert_eq!(doc["shards"][0]["retried"], true, "{doc}");
    assert!(doc["shards"][1]["status"].is_string());
    assert_eq!(doc["label_inputs"].as_array().map(Vec::len), Some(3));
    assert_eq!(doc["label_inputs_pending"], json!([]));
    let _ = fs::remove_dir_all(&f.base);
}
