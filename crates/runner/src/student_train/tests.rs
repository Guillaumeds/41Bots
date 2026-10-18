use super::*;
use crate::bundle::{bundle_dir, sha256_hex};
use crate::remote::DirRemote;

fn run_entry(stream: &str, parts: u64) -> Value {
    json!({ "run": "teach-1", "stream": stream, "parts": parts })
}

fn base() -> Value {
    json!({ "threads": "all",
            "train_inputs": [run_entry("rules", 3), run_entry("mixed", 1)],
            "test_inputs": [run_entry("test", 1)],
            "sizes": [40, 100], "widths": ["16,8", "8"] })
}

fn sidecar(records: &[u8], values: &[u8], n: u64, role: &str, source: &str) -> Value {
    json!({ "format": "41bots-student-teacher-v1", "key": sha256_hex(records),
            "values_sha256": sha256_hex(values), "records_bytes": records.len(),
            "values_bytes": values.len(), "records": n, "role": role, "source": source,
            "teacher": "hybrid:fake:cut3", "equity_sha256": "eq", "rules_version": 2,
            "values_version": 2, "sampler": "belief-uniform", "candidates": "app-v1",
            "trick": 2 })
}

fn shard_entry(records: &[u8], values: &[u8], n: u64, role: &str) -> Value {
    json!({ "records": format!("sha256:{}", sha256_hex(records)),
            "records_bytes": records.len(),
            "values": format!("sha256:{}", sha256_hex(values)), "values_bytes": values.len(),
            "sidecar": sidecar(records, values, n, role, "rules") })
}

#[test]
fn settings_are_checked() {
    let q = parse(&base()).expect("parses");
    assert_eq!(q.sizes, vec![40, 100]);
    assert_eq!(q.widths, vec!["16,8".to_string(), "8".to_string()]);
    assert_eq!(
        (q.epochs, q.batch, q.patience, q.seed, q.val_permille),
        (40, 256, 4, 1, 50)
    );
    assert_eq!(q.lr, 0.001);
    assert_eq!((q.order, q.device), (Order::Spread, Device::Auto));
    assert_eq!(q.baselines, vec!["rules".to_string(), "random".to_string()]);
    assert!(!q.keep_raw && q.upload);
    assert_eq!(
        q.train[0],
        Entry::Run {
            run: "teach-1".into(),
            stream: "rules".into(),
            parts: Some(3)
        }
    );
    for (k, v) in [
        ("sizes", json!([100, 40])),
        ("sizes", json!([40, 40])),
        ("sizes", json!([])),
        ("sizes", json!([0])),
        ("widths", json!([])),
        ("widths", json!(["16,x"])),
        ("widths", json!(["8", "8"])),
        ("widths", json!(["100000"])),
        ("device", json!("gpu")),
        ("order", json!("random")),
        ("baselines", json!(["oracle"])),
        ("epochs", json!(0)),
        ("lr", json!(0)),
        ("val_permille", json!(1000)),
        ("keep_raw", json!("yes")),
        ("train_inputs", json!([])),
        ("train_inputs", json!([{ "run": "a/b", "stream": "rules" }])),
        (
            "train_inputs",
            json!([{ "run": "a", "stream": "rules", "parts": 0 }]),
        ),
        (
            "train_inputs",
            json!([{ "run": "a", "stream": "rules", "rows": 3 }]),
        ),
        ("test_inputs", json!([{ "records": "sha256:00" }])),
        ("nonsense", json!(1)),
    ] {
        let mut p = base();
        p[k] = v.clone();
        assert!(parse(&p).is_err(), "{k} = {v}");
    }
    // A stored shard named directly, as its .inputs.json gives it; its keys must match its
    // sidecar.
    let mut p = base();
    p["test_inputs"] = json!([shard_entry(b"rec", b"val", 3, "test")]);
    let q = parse(&p).expect("parses");
    let Entry::Shard(s) = &q.test[0] else {
        panic!("a shard")
    };
    assert_eq!((s.records_bytes, s.decisions()), (3, 3));
    let mut e = shard_entry(b"rec", b"val", 3, "test");
    e["values_bytes"] = json!(4);
    p["test_inputs"] = json!([e]);
    assert!(parse(&p).is_err());
    let mut e = shard_entry(b"rec", b"val", 3, "test");
    e["sidecar"]["key"] = json!("0".repeat(64));
    p["test_inputs"] = json!([e]);
    assert!(parse(&p).is_err());
}

#[test]
fn the_spread_order_interleaves_in_proportion_and_sizes_are_prefixes() {
    let s = |label: &str| Shard {
        records: label.into(),
        records_bytes: 1,
        values: label.into(),
        values_bytes: 1,
        sidecar: json!({ "records": 10 }),
        label: label.into(),
    };
    let rules: Vec<Shard> = (0..8).map(|i| s(&format!("r{i}"))).collect();
    let mixed: Vec<Shard> = (0..2).map(|i| s(&format!("m{i}"))).collect();
    let labels = |v: Vec<Shard>| v.into_iter().map(|x| x.label).collect::<Vec<_>>();
    assert_eq!(
        labels(ordered(vec![rules.clone(), mixed.clone()], Order::Spread)),
        ["r0", "r1", "m0", "r2", "r3", "r4", "r5", "m1", "r6", "r7"]
    );
    assert_eq!(
        labels(ordered(
            vec![rules[..3].to_vec(), mixed[..1].to_vec()],
            Order::Spread
        )),
        ["r0", "r1", "m0", "r2"]
    );
    assert_eq!(
        labels(ordered(
            vec![rules[..2].to_vec(), mixed.clone()],
            Order::Given
        )),
        ["r0", "r1", "m0", "m1"]
    );
    // 40 rules and 10 mixed parts: every five hold four rules ones and one mixed one.
    let r40: Vec<Shard> = (0..40).map(|i| s(&format!("r{i}"))).collect();
    let m10: Vec<Shard> = (0..10).map(|i| s(&format!("m{i}"))).collect();
    let o = labels(ordered(vec![r40, m10], Order::Spread));
    for w in o.chunks(5) {
        assert_eq!(w.iter().filter(|l| l.starts_with('m')).count(), 1, "{w:?}");
    }
    assert_eq!(parts_for(&[30, 30, 30], 1), Some(1));
    assert_eq!(parts_for(&[30, 30, 30], 30), Some(1));
    assert_eq!(parts_for(&[30, 30, 30], 31), Some(2));
    assert_eq!(parts_for(&[30, 30, 30], 90), Some(3));
    assert_eq!(parts_for(&[30, 30, 30], 91), None);
    assert_eq!(pair_name(50000, "512,256"), "n50000-w512x256");
}

/// The manifests in docs/vast parse, and a student_train stream is one chunk.
#[test]
fn the_manifests_parse() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/vast");
    for name in [
        "student-t2-curve.manifest.json",
        "student-t2-curve-mech.manifest.json",
    ] {
        let text = fs::read_to_string(root.join(name)).expect(name);
        let v: Value = serde_json::from_str(&text).expect(name);
        crate::sched::parse_v2(&v).unwrap_or_else(|e| panic!("{name}: {e}"));
        let p = &v["streams"][0]["params"];
        let q = parse(p).expect(name);
        assert!(!q.train.is_empty() && !q.test.is_empty(), "{name}");
    }
    let mut v: Value = serde_json::from_str(
        &fs::read_to_string(root.join("student-t2-curve.manifest.json")).expect("read"),
    )
    .expect("JSON");
    let q = parse(&v["streams"][0]["params"]).expect("parses");
    assert_eq!(q.sizes, vec![50_000, 100_000, 250_000]);
    assert_eq!(q.widths, vec!["256,128".to_string(), "512,256".to_string()]);
    assert_eq!(q.device, Device::Cuda);
    v["streams"][0]["chunks"] = json!(2);
    assert!(crate::sched::parse_v2(&v).is_err());
}

/// A fake program at `bin/<name>` running `body`.
#[cfg(unix)]
fn script(bin: &Path, name: &str, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let p = bin.join(name);
    fs::write(&p, format!("#!/bin/sh\n{body}")).expect("write");
    fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).expect("chmod");
    p
}

#[cfg(unix)]
const ARG: &str = r#"arg() { n=$1; shift; while [ $# -gt 0 ]; do [ "$1" = "$n" ] && { echo "$2"; return; }; shift; done; }"#;

/// A stored `student_teacher` run in `store`: per stream, its parts (decisions each), each part's
/// two files in `inputs/sha256/` and its bundle (result.json and the `.inputs.json`) listed in
/// `runs/<run>/merged.json`.
fn stored_run(store: &Path, run: &str, streams: &[(&str, &str, &[u64])]) {
    let remote = DirRemote {
        root: store.to_path_buf(),
    };
    let mut ms = Vec::new();
    for (stream, role, parts) in streams {
        let mut chunks = Vec::new();
        for (i, &n) in parts.iter().enumerate() {
            let records = format!("records {stream} {i}").into_bytes();
            let values = format!("values of {stream} part {i}").into_bytes();
            for b in [&records, &values] {
                let d = store.join("inputs/sha256");
                fs::create_dir_all(&d).expect("mkdir");
                fs::write(d.join(sha256_hex(b)), b).expect("write");
            }
            let file = format!("t2-{role}-{stream}-p{i:05}.npy");
            let mut inputs = shard_entry(&records, &values, n, role);
            inputs["part"] = json!(i);
            inputs["file"] = json!(file);
            inputs["sidecar"]["source"] = json!(stream);
            let out = store.join("tmp-bundle").join(format!("{stream}-{i}"));
            fs::create_dir_all(out.join("student")).expect("mkdir");
            fs::write(
                out.join("result.json"),
                json!({ "kind": "student_teacher", "part": i, "file": file }).to_string(),
            )
            .expect("write");
            fs::write(
                out.join(format!("student/{file}.inputs.json")),
                inputs.to_string(),
            )
            .expect("write");
            let local = store
                .join("tmp-bundle")
                .join(format!("{stream}-{i}.tar.zst"));
            let (sha, bytes) = bundle_dir(&out, &local).expect("bundle");
            let key = format!("runs/{run}/out/{stream}/{i:06}-{}.tar.zst", &sha[..16]);
            remote.put(&local, &key, true).expect("put");
            chunks.push(json!({ "index": i, "seed": 1,
                                "outputs": [ { "path": key, "sha256": sha, "bytes": bytes } ] }));
        }
        ms.push(
            json!({ "name": stream, "kind": "student_teacher", "chunks": chunks,
                        "chunks_done": parts.len(), "chunks_total": parts.len() }),
        );
    }
    let merged = store.join("tmp-bundle/merged.json");
    fs::write(
        &merged,
        json!({ "complete": true, "run_id": run, "streams": ms }).to_string(),
    )
    .expect("write");
    remote
        .put(&merged, &format!("runs/{run}/merged.json"), true)
        .expect("put");
    let _ = fs::remove_dir_all(store.join("tmp-bundle"));
}

/// The chunk with fake programs on a local store: the inputs resolved from a stored run in the
/// spread order, only the parts the largest size needs fetched and encoded (the test set apart),
/// one training and one check per (size, width) with `--limit` the size, the integer networks
/// shared content-addressed, the curve written; a second call on the same directory trains
/// nothing (every pair finished); a wrong part count, a test shard among the training inputs and
/// a size beyond the data stop the chunk.
#[cfg(unix)]
#[test]
fn a_chunk_trains_the_curve_with_fake_programs() {
    let tmp = std::env::temp_dir().join(format!("student-train-{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    let bin = tmp.join("bin");
    fs::create_dir_all(&bin).expect("mkdir");
    let calls = tmp.join("calls.log");
    let c = calls.display();
    let py = script(
        &bin,
        "fake-python",
        &format!(
            r#"{ARG}
echo "python $*" >> '{c}'
case "$2" in
fortyone_nn.device) echo '{{"device": "cpu", "reason": "fake"}}' > "$(arg --json "$@")" ;;
fortyone_nn.student)
  out=$(arg --out "$@"); n=$(arg --limit "$@"); w=$(arg --widths "$@")
  mkdir -p "$out"
  printf "float $w $n" > "$out/float.safetensors"
  printf "int $w $n" > "$out/int.safetensors"
  echo "{{\"data\": {{\"train_rows\": $((n - 2)), \"val_rows\": 2}}, \"history\": [{{}}], \"best_epoch\": 0, \"seconds\": 6, \"val\": {{\"int\": {{\"all\": {{\"decisions\": 2}}}}}}, \"test\": {{\"int\": {{\"all\": {{\"decision_loss_pct\": 0.25}}}}}}}}" > "$out/report.json"
  echo '{{"step": "done", "epoch": 0}}' > "$(arg --progress "$@")" ;;
esac
"#
        ),
    );
    let uv = script(
        &bin,
        "uv",
        &format!(
            r#"echo "uv $*" >> '{c}'
[ "$1" = "--version" ] && {{ echo "uv 0.0 (fake)"; exit 0; }}
mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"
cp '{py}' "$UV_PROJECT_ENVIRONMENT/bin/python"
"#,
            py = py.display()
        ),
    );
    let student = script(
        &bin,
        "student",
        &format!(
            r#"echo "student $*" >> '{c}'
out=""; n=0; role=""
while [ $# -gt 0 ]; do
  case "$1" in
  --out) out=$2; shift ;;
  --teacher) side="$2.json"
     k=$(grep -o '"records": *[0-9]*' "$side" | head -1 | cut -d: -f2); n=$((n + k))
     role=$(grep -o '"role": *"[a-z]*"' "$side" | head -1 | cut -d'"' -f4); shift ;;
  esac
  shift
done
printf features > "$out"
echo "{{\"records\": $n, \"role\": \"$role\", \"encoder\": \"student-v1\", \"encoder_sha256\": \"e\"}}" > "$out.json"
"#
        ),
    );
    let search = script(
        &bin,
        "search",
        &format!(
            r#"{ARG}
echo "search $*" >> '{c}'
row() {{ echo "{{\"group\": \"$1\", \"n\": $2, \"mean_loss_pct\": 0.25, \"upper95_pct\": 0.5, \"agreement\": 0.5, \"value_mae_pct\": 1}}"; }}
rows="[$(row all 24), $(row 'place 1' 3), $(row 'stratum S1' 6)]"
echo "{{\"role\": \"test\", \"rows\": $rows, \"baselines\": [{{\"bot\": \"rules-v1\", \"rows\": $rows}}, {{\"bot\": \"random\", \"expected_loss_pct\": 1, \"rows\": $rows}}]}}" > "$(arg --json "$@")"
echo "group n loss"
"#
        ),
    );
    let pydir = tmp.join("py");
    fs::create_dir_all(pydir.join("fortyone_nn")).expect("mkdir");
    fs::write(pydir.join("uv.lock"), "lock").expect("write");
    let store = tmp.join("store");
    stored_run(
        &store,
        "teach-1",
        &[
            ("test", "test", &[24]),
            ("rules", "train", &[30, 30, 30]),
            ("mixed", "train", &[30]),
        ],
    );
    let remote: Arc<dyn Remote> = Arc::new(DirRemote {
        root: store.clone(),
    });
    let progs = Programs {
        tools: Tools {
            search,
            uv,
            python_dir: pydir,
            nvidia_smi: tmp.join("no-nvidia-smi"),
            remote: Some(Arc::clone(&remote)),
        },
        student,
    };
    let table = tmp.join("equity.npy");
    fs::write(&table, "table").expect("write");
    let root = tmp.join("c1");
    let mut ctx = Ctx::local("student-train-test", &root, 4);
    let mut p = base();
    p["device"] = json!("cpu");
    p["fetch_jobs"] = json!(2);
    let r = chunk_with(&mut ctx, &p, &progs, &table, json!({})).expect("runs");
    assert_eq!(r["pairs"], 4, "{r}");
    let out = root.join("out/student");
    let curve = read_json(&out.join("curve.json")).expect("curve");
    let pairs: Vec<(u64, String)> = curve["rows"]
        .as_array()
        .expect("rows")
        .iter()
        .map(|r| (r["size"].as_u64().expect("size"), r["widths"].to_string()))
        .collect();
    assert_eq!(
        pairs,
        vec![
            (40, "\"16,8\"".into()),
            (40, "\"8\"".into()),
            (100, "\"16,8\"".into()),
            (100, "\"8\"".into())
        ]
    );
    assert_eq!(curve["baselines"][1]["bot"], "random");
    let text = fs::read_to_string(out.join("curve.txt")).expect("text");
    assert!(
        text.contains("baseline rules-v1") && text.contains("student 16,8"),
        "{text}"
    );
    // The integer networks in the store, by SHA-256.
    for row in curve["rows"].as_array().expect("rows") {
        let key = row["int_key"].as_str().expect("key");
        assert!(store.join(key).is_file(), "{key}");
    }
    let inputs = read_json(&out.join("inputs.json")).expect("inputs");
    let labels: Vec<&str> = inputs["train"]
        .as_array()
        .expect("train")
        .iter()
        .map(|s| s["label"].as_str().expect("label"))
        .collect();
    assert_eq!(
        labels,
        [
            "teach-1/rules/part 0",
            "teach-1/rules/part 1",
            "teach-1/mixed/part 0",
            "teach-1/rules/part 2"
        ]
    );
    let done = read_json(&out.join("n40-w16x8/done.json")).expect("done");
    assert_eq!(done["parts"], 2);
    let log = fs::read_to_string(&calls).expect("calls");
    // The test set encoded once by itself; each needed training part once.
    assert_eq!(log.matches("student encode").count(), 5, "{log}");
    let trainings: Vec<&str> = log
        .lines()
        .filter(|l| l.contains("fortyone_nn.student"))
        .collect();
    assert_eq!(trainings.len(), 4, "{log}");
    assert!(trainings[0].contains("--limit 40 ") && trainings[0].contains("--widths 16,8 "));
    assert!(trainings[0].contains("--device cpu ") && trainings[0].contains("--test "));
    assert_eq!(log.matches("search student-check").count(), 4, "{log}");
    assert!(log.contains("--baseline rules --baseline random"), "{log}");
    assert!(
        log.contains("uv sync --locked --no-default-groups --group cpu "),
        "{log}"
    );
    // The training parts' raw files are gone (keep_raw false); the test set's stay.
    let raw: Vec<String> = fs::read_dir(root.join("student-train-work/raw"))
        .expect("raw")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(raw.len(), 3, "{raw:?}");

    // Every pair finished: the same chunk again trains nothing.
    fs::write(&calls, "").expect("clear");
    let r = chunk_with(&mut ctx, &p, &progs, &table, json!({})).expect("runs");
    assert_eq!(r["pairs"], 4);
    let log = fs::read_to_string(&calls).expect("calls");
    assert!(!log.contains("fortyone_nn.student"), "{log}");
    // A pair whose files no longer match its done.json is trained again.
    fs::write(out.join("n100-w8/int.safetensors"), "other").expect("write");
    chunk_with(&mut ctx, &p, &progs, &table, json!({})).expect("runs");
    let log = fs::read_to_string(&calls).expect("calls");
    assert_eq!(log.matches("fortyone_nn.student").count(), 1, "{log}");

    // Only the parts the largest size needs are fetched.
    fs::write(&calls, "").expect("clear");
    let mut small = p.clone();
    small["sizes"] = json!([10]);
    small["widths"] = json!(["8"]);
    let mut ctx2 = Ctx::local("student-train-test", &tmp.join("c2"), 4);
    chunk_with(&mut ctx2, &small, &progs, &table, json!({})).expect("runs");
    let log = fs::read_to_string(&calls).expect("calls");
    assert_eq!(log.matches("student encode").count(), 2, "{log}");

    // Refusals.
    let fails = |q: Value, word: &str| {
        let mut ctx = Ctx::local("student-train-test", &tmp.join("c3"), 4);
        let e = chunk_with(&mut ctx, &q, &progs, &table, json!({})).expect_err("refused");
        assert!(e.contains(word), "{e}");
    };
    let mut q = p.clone();
    q["train_inputs"] = json!([run_entry("rules", 4)]);
    fails(q, "holds 3 parts");
    let mut q = p.clone();
    q["train_inputs"] = json!([run_entry("rules", 3), run_entry("test", 1)]);
    fails(q, "role");
    let mut q = p.clone();
    q["sizes"] = json!([121]);
    fails(q, "hold 120 decisions");
    let mut q = p.clone();
    q["test_inputs"] = json!([{ "run": "teach-2", "stream": "test" }]);
    fails(q, "not found");
    let _ = fs::remove_dir_all(&tmp);
}
