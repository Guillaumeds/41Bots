//! The `student_teacher` chunk kind (docs/STUDENT.md §3–§4, §8): one part of a stream of the
//! student's teacher records on one machine with every usable thread, the shard shared
//! content-addressed for later runs to train on.
//!
//! ```json
//! { "name": "base", "priority": 2, "kind": "student_teacher", "chunks": 50,
//!   "params": { "threads": "all", "trick": 2, "count": 5000, "deals": 320, "seed": 401,
//!               "source": "rules", "net_input": "o3", "tt_mb": 64 } }
//! ```
//! Chunk `i` of the stream makes part `first_part + i`: decisions `part × count` to
//! `(part + 1) × count` of the generator's stream (seed, source, trick), each valued over deals
//! `0..deals` with the hybrid H_k (`net_input`: a manifest input, an O_k with k after the trick),
//! by `search teacher --trick` (`search::student_teacher`): the same bytes on any machine and
//! thread count, so a part redone after a lost machine is the same shard. Earlier runs only ever
//! add to the data: a later run continues the stream with a larger `first_part`, and
//! `skip_parts` lists parts already stored (an interrupted run's), which are not made again.
//!
//! Outputs: the records file and its sidecar in `out/student/` (small; in the chunk's bundle), and
//! both files of the shard uploaded to `inputs/sha256/<hex>` (the records file's SHA-256 is the
//! shard's key), listed in `out/student/<name>.inputs.json` for a later run's training. When the
//! upload fails (or there is no remote store) the values file goes into `out/student/` too, so
//! the bundle still holds the whole shard.
//!
//! Settings: `trick` (1 or 2), `count` (decisions per part), `deals`, `seed` (the generator's:
//! never 1; `student::STUDENT_TEST_SEED` only with `role: "test"`), `source` (`rules`, `mixed`
//! or `random`), `role` (`train` or `test`), `net_input` (required), `tt_mb` (64), `start_hands`
//! (20,000), `first_part` (0), `skip_parts` ([]), `upload` (true), `table` (as `hybrid_t1check`:
//! table v0 built, or the `equity` input), `equity_input`; any other key is refused ([`KEYS`]).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde_json::{Value, json};

use crate::remote::{input_key, put_input};
use crate::spike::build_table_from;
use crate::{Ctx, sibling};

/// The settings a `student_teacher` chunk takes (after `v2.rs` added the network's path,
/// `net`); any other key is refused.
pub(crate) const KEYS: [&str; 17] = [
    "trick",
    "count",
    "deals",
    "seed",
    "source",
    "role",
    "net_input",
    "tt_mb",
    "start_hands",
    "first_part",
    "skip_parts",
    "upload",
    "table",
    "equity_input",
    "threads",
    "est_seconds",
    "chunk_sizes",
];

/// The checked settings.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Params {
    pub trick: u64,
    pub count: u64,
    pub deals: u64,
    pub seed: u64,
    pub source: String,
    pub role: String,
    pub net_input: String,
    pub tt_mb: u64,
    pub start_hands: u64,
    pub first_part: u64,
    pub skip_parts: Vec<u64>,
    pub upload: bool,
}

/// The test set's seed (`student::STUDENT_TEST_SEED`).
const TEST_SEED: u64 = 0x7E57;

/// Checks the settings (also when the manifest is read, so a typo stops the run before it
/// starts).
pub(crate) fn parse(p: &Value) -> Result<Params, String> {
    let o = p
        .as_object()
        .ok_or("student_teacher: params must be an object")?;
    for k in o.keys() {
        if !KEYS.contains(&k.as_str()) && k != "net" {
            return Err(format!("student_teacher: unknown setting `{k}`"));
        }
    }
    let n = |k: &str, d: Option<u64>| -> Result<u64, String> {
        match p.get(k) {
            None | Some(Value::Null) => d.ok_or_else(|| format!("student_teacher: needs {k}")),
            Some(v) => v
                .as_u64()
                .ok_or_else(|| format!("student_teacher: {k} must be a whole number")),
        }
    };
    let s = |k: &str, d: Option<&str>| -> Result<String, String> {
        match p.get(k) {
            None | Some(Value::Null) => d
                .map(str::to_string)
                .ok_or_else(|| format!("student_teacher: needs {k}")),
            Some(Value::String(x)) => Ok(x.clone()),
            Some(_) => Err(format!("student_teacher: {k} must be a string")),
        }
    };
    let trick = n("trick", None)?;
    if !(1..=2).contains(&trick) {
        return Err(format!(
            "student_teacher: trick {trick}: the student's decisions are at tricks 1 and 2"
        ));
    }
    let count = n("count", None)?;
    let deals = n("deals", None)?;
    if count == 0 || deals == 0 || deals > u64::from(u32::MAX) {
        return Err("student_teacher: count and deals must be at least 1".into());
    }
    let seed = n("seed", None)?;
    let role = s("role", Some("train"))?;
    if !["train", "test"].contains(&role.as_str()) {
        return Err(format!("student_teacher: role {role}: train or test"));
    }
    if seed == 1 {
        return Err("student_teacher: seed 1 is the golden sets'".into());
    }
    if (role == "test") != (seed == TEST_SEED) {
        return Err(format!(
            "student_teacher: the test set (and only it) uses seed {TEST_SEED}"
        ));
    }
    let source = s("source", Some("rules"))?;
    if !["rules", "mixed", "random"].contains(&source.as_str()) {
        return Err(format!(
            "student_teacher: source {source}: rules, mixed or random"
        ));
    }
    let skip_parts = match p.get("skip_parts") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| {
                x.as_u64()
                    .ok_or("student_teacher: skip_parts must be whole numbers")
            })
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => return Err("student_teacher: skip_parts must be a list".into()),
    };
    let upload = match p.get("upload") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err("student_teacher: upload must be true or false".into()),
    };
    Ok(Params {
        trick,
        count,
        deals,
        seed,
        source,
        role,
        net_input: s("net_input", None)?,
        tt_mb: n("tt_mb", Some(64))?.max(1),
        start_hands: n("start_hands", Some(20_000))?,
        first_part: n("first_part", Some(0))?,
        skip_parts,
        upload,
    })
}

/// The inputs a stream names (its network).
pub(crate) fn input_names(p: &Value) -> Result<Vec<String>, String> {
    Ok(vec![parse(p)?.net_input])
}

/// Adds the network's local path (`net`) from the worker's `inputs` (name → path).
pub(crate) fn resolve(
    p: &Value,
    inputs: &std::collections::BTreeMap<String, String>,
) -> Result<Value, String> {
    let q = parse(p)?;
    let path = inputs.get(&q.net_input).cloned().ok_or_else(|| {
        format!(
            "student_teacher needs input `{}` in the manifest's inputs",
            q.net_input
        )
    })?;
    let mut out = p.clone();
    out["net"] = json!(path);
    Ok(out)
}

/// The shard's file name for part `part`.
pub(crate) fn shard_name(q: &Params, part: u64) -> String {
    format!(
        "t{}-{}-{}-s{}-d{}-p{part:05}.npy",
        q.trick, q.role, q.source, q.seed, q.deals
    )
}

/// The `search teacher` command of part `part`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn command(
    search: &Path,
    q: &Params,
    part: u64,
    net: &str,
    table: &Path,
    out: &Path,
    threads: usize,
    progress: &Path,
) -> Command {
    let mut c = Command::new(search);
    c.arg("teacher")
        .args(["--trick", &q.trick.to_string()])
        .args(["--count", &q.count.to_string()])
        .args(["--first", &(part * q.count).to_string()])
        .args(["--deals", &q.deals.to_string()])
        .args(["--seed", &q.seed.to_string()])
        .args(["--source", &q.source])
        .args(["--role", &q.role])
        .args(["--net", net])
        .arg("--equity")
        .arg(table)
        .arg("--out")
        .arg(out)
        .args(["--threads", &threads.max(1).to_string()])
        .args(["--tt-mb", &q.tt_mb.to_string()])
        .args(["--start-hands", &q.start_hands.to_string()])
        .arg("--progress")
        .arg(progress);
    c
}

/// Run one `student_teacher` chunk: chunk `index` of its stream (the network's path resolved by
/// `v2.rs`).
pub(crate) fn chunk(ctx: &mut Ctx, p: &Value, index: u64) -> Result<Value, String> {
    let q = parse(p)?;
    let part = q.first_part + index;
    if q.skip_parts.contains(&part) {
        return Ok(json!({ "kind": "student_teacher", "part": part, "skipped": true }));
    }
    ctx.set_phase(
        "chunk",
        json!({ "kind": "student_teacher", "step": "setup: table v0", "part": part }),
    );
    let (table, table_rec) = build_table_from(ctx, p.get("table").unwrap_or(&json!({})), "auto")?;
    let remote = if q.upload {
        crate::remote::from_env().map(std::sync::Arc::from)
    } else {
        None
    };
    chunk_with(ctx, p, index, &sibling("search"), &table, table_rec, remote)
}

/// [`chunk`] after the equity table, with the given `search` program and remote store.
pub(crate) fn chunk_with(
    ctx: &mut Ctx,
    p: &Value,
    index: u64,
    search: &Path,
    table: &Path,
    table_rec: Value,
    remote: Option<std::sync::Arc<dyn crate::remote::Remote>>,
) -> Result<Value, String> {
    let q = parse(p)?;
    let part = q.first_part + index;
    if q.skip_parts.contains(&part) {
        return Ok(json!({ "kind": "student_teacher", "part": part, "skipped": true }));
    }
    let net = p
        .get("net")
        .and_then(Value::as_str)
        .ok_or("student_teacher: the network's path was not resolved")?
        .to_string();
    let work = ctx.root.join("student-work");
    let outdir = ctx.root.join("out").join("student");
    fs::create_dir_all(&work).map_err(|e| e.to_string())?;
    fs::create_dir_all(&outdir).map_err(|e| e.to_string())?;
    let name = shard_name(&q, part);
    let shard = work.join(&name);
    let progress = work.join("progress.json");
    let t = Instant::now();
    let cmd = command(
        search,
        &q,
        part,
        &net,
        table,
        &shard,
        ctx.threads,
        &progress,
    );
    let count = q.count;
    ctx.run_child_with(cmd, "search.log", &mut |c: &mut Ctx| {
        if let Ok(text) = fs::read_to_string(&progress)
            && let Ok(v) = serde_json::from_str::<Value>(&text)
        {
            c.progress = json!({ "kind": "student_teacher", "step": "teacher", "part": part,
                                 "done": v["done"], "count": count });
        }
    })?;
    let secs = t.elapsed().as_secs_f64();
    let values = PathBuf::from(format!(
        "{}.values.npy",
        shard.to_string_lossy().trim_end_matches(".npy")
    ));
    let sidecar = PathBuf::from(format!("{}.json", shard.display()));
    let side: Value = serde_json::from_str(
        &fs::read_to_string(&sidecar).map_err(|e| format!("{}: {e}", sidecar.display()))?,
    )
    .map_err(|e| e.to_string())?;
    let key = side["key"].as_str().unwrap_or_default().to_string();
    let values_sha = side["values_sha256"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    // The small files go into the bundle; both shard files to the content-addressed store.
    fs::copy(&shard, outdir.join(&name)).map_err(|e| e.to_string())?;
    fs::copy(&sidecar, outdir.join(format!("{name}.json"))).map_err(|e| e.to_string())?;
    let mut uploaded = false;
    let mut warning = Value::Null;
    if let Some(r) = remote {
        // Beside the heartbeat (the network never blocks the runner).
        let (s2, v2, k2, w2) = (
            shard.clone(),
            values.clone(),
            key.clone(),
            values_sha.clone(),
        );
        let both = ctx.blocking(move || {
            put_input(&*r, &s2, Some(&k2))
                .and_then(|a| put_input(&*r, &v2, Some(&w2)).map(|b| (a, b)))
        });
        match both {
            Ok(((rs, rb, rst), (vs, vb, vst))) => {
                uploaded = true;
                println!(
                    "runner: student_teacher: part {part}: {} ({rst}), {} ({vst})",
                    input_key(&rs),
                    input_key(&vs)
                );
                let inputs = json!({ "part": part, "file": name,
                    "records": format!("sha256:{rs}"), "records_bytes": rb,
                    "values": format!("sha256:{vs}"), "values_bytes": vb, "sidecar": side });
                fs::write(
                    outdir.join(format!("{name}.inputs.json")),
                    serde_json::to_string_pretty(&inputs).map_err(|e| e.to_string())? + "\n",
                )
                .map_err(|e| e.to_string())?;
            }
            Err(e) => {
                warning = json!(format!(
                    "the shard was not uploaded ({e}): its values file is in the bundle"
                ));
            }
        }
    }
    if !uploaded {
        let vname = values
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        fs::copy(&values, outdir.join(vname)).map_err(|e| e.to_string())?;
    }
    Ok(json!({
        "kind": "student_teacher",
        "part": part,
        "first": part * q.count,
        "count": q.count,
        "deals": q.deals,
        "file": name,
        "key": key,
        "values_sha256": values_sha,
        "uploaded": uploaded,
        "warning": warning,
        "seconds": secs,
        "decisions_per_hour": q.count as f64 * 3600.0 / secs.max(1e-9),
        "table": table_rec,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Value {
        json!({ "threads": "all", "trick": 2, "count": 5000, "deals": 320, "seed": 401,
                "source": "rules", "net_input": "o3" })
    }

    fn base_params() -> Value {
        base()
    }

    #[test]
    fn settings_are_checked() {
        let q = parse(&base()).expect("parses");
        assert_eq!((q.trick, q.count, q.deals, q.seed), (2, 5000, 320, 401));
        assert_eq!((q.role.as_str(), q.tt_mb, q.first_part), ("train", 64, 0));
        assert!(q.upload);
        for (k, v) in [
            ("trick", json!(3)),
            ("count", json!(0)),
            ("seed", json!(1)),
            ("seed", json!(TEST_SEED)),
            ("source", json!("harvest")),
            ("role", json!("val")),
            ("nonsense", json!(1)),
            ("skip_parts", json!("3")),
        ] {
            let mut p = base();
            p[k] = v.clone();
            assert!(parse(&p).is_err(), "{k} = {v}");
        }
        let mut p = base();
        p["seed"] = json!(TEST_SEED);
        p["role"] = json!("test");
        assert!(parse(&p).is_ok());
        let mut p = base();
        p.as_object_mut().expect("object").remove("net_input");
        assert!(parse(&p).is_err());
        assert_eq!(input_names(&base()).expect("names"), vec!["o3".to_string()]);
    }

    #[test]
    fn a_part_is_its_slice_of_the_stream() {
        let mut p = base();
        p["first_part"] = json!(10);
        p["skip_parts"] = json!([11]);
        let q = parse(&p).expect("parses");
        assert_eq!(shard_name(&q, 12), "t2-train-rules-s401-d320-p00012.npy");
        let c = command(
            Path::new("search"),
            &q,
            12,
            "o3.safetensors",
            Path::new("equity.npy"),
            Path::new("out.npy"),
            8,
            Path::new("p.json"),
        );
        let args: Vec<String> = c
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        let at = |k: &str| args[args.iter().position(|a| a == k).expect(k) + 1].clone();
        assert_eq!(at("--first"), "60000");
        assert_eq!(at("--count"), "5000");
        assert_eq!(at("--trick"), "2");
        assert_eq!(at("--threads"), "8");
        let inputs = std::collections::BTreeMap::from([("o3".to_string(), "/x/o3".to_string())]);
        assert_eq!(resolve(&p, &inputs).expect("resolves")["net"], "/x/o3");
        assert!(resolve(&p, &std::collections::BTreeMap::new()).is_err());
    }

    /// The chunk with a fake `search` writing a shard: the records file and its sidecar go into
    /// `out/student/`, both shard files to the store's `inputs/sha256/<hex>` (listed in
    /// `<name>.inputs.json`); with no store the values file goes into `out/student/` instead; a
    /// skipped part makes nothing.
    #[cfg(unix)]
    #[test]
    fn a_chunk_makes_and_shares_its_part() {
        use std::os::unix::fs::PermissionsExt;
        let base = std::env::temp_dir().join(format!("student-chunk-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).expect("mkdir");
        let calls = base.join("calls.log");
        let search = base.join("search");
        let arg = r#"arg() { n=$1; shift; while [ $# -gt 0 ]; do [ "$1" = "$n" ] && { echo "$2"; return; }; shift; done; }"#;
        fs::write(
            &search,
            format!(
                r#"#!/bin/sh
{arg}
echo "search $*" >> '{calls}'
out=$(arg --out "$@")
stem=${{out%.npy}}
printf records > "$out"
printf values > "$stem.values.npy"
r=$(sha256sum "$out" | cut -d' ' -f1)
v=$(sha256sum "$stem.values.npy" | cut -d' ' -f1)
echo "{{\"key\": \"$r\", \"values_sha256\": \"$v\"}}" > "$out.json"
echo '{{"done": 7}}' > "$(arg --progress "$@")"
"#,
                calls = calls.display()
            ),
        )
        .expect("write");
        fs::set_permissions(&search, fs::Permissions::from_mode(0o755)).expect("chmod");
        let mut p = base_params();
        p["net"] = json!("/x/o3");
        p["first_part"] = json!(3);
        let store = base.join("store");
        let remote: std::sync::Arc<dyn crate::remote::Remote> =
            std::sync::Arc::new(crate::remote::DirRemote {
                root: store.clone(),
            });
        let table = Path::new("/x/equity.npy");
        let mut ctx = Ctx::local("student-test", &base.join("c1"), 4);
        let r = chunk_with(&mut ctx, &p, 2, &search, table, json!({}), Some(remote)).expect("runs");
        let name = "t2-train-rules-s401-d320-p00005.npy";
        assert_eq!(
            (r["part"].as_u64(), r["file"].as_str()),
            (Some(5), Some(name))
        );
        assert_eq!(r["uploaded"], true);
        let out = base.join("c1/out/student");
        assert_eq!(
            fs::read_to_string(out.join(name)).expect("records"),
            "records"
        );
        assert!(out.join(format!("{name}.json")).exists());
        assert!(
            !out.join("t2-train-rules-s401-d320-p00005.values.npy")
                .exists()
        );
        let inputs: Value = serde_json::from_str(
            &fs::read_to_string(out.join(format!("{name}.inputs.json"))).expect("inputs"),
        )
        .expect("JSON");
        let key = inputs["records"]
            .as_str()
            .expect("key")
            .trim_start_matches("sha256:");
        assert_eq!(key, r["key"].as_str().expect("key"));
        assert!(store.join("inputs/sha256").join(key).exists());
        let vkey = inputs["values"]
            .as_str()
            .expect("key")
            .trim_start_matches("sha256:");
        assert_eq!(
            fs::read_to_string(store.join("inputs/sha256").join(vkey)).expect("values"),
            "values"
        );
        let call = fs::read_to_string(&calls).expect("calls");
        assert!(call.contains("--first 25000 "), "{call}");
        assert!(call.contains("--net /x/o3 "), "{call}");
        // No store: the values file stays in the bundle.
        let mut ctx = Ctx::local("student-test", &base.join("c2"), 4);
        let r = chunk_with(&mut ctx, &p, 0, &search, table, json!({}), None).expect("runs");
        assert_eq!(r["uploaded"], false);
        assert!(
            base.join("c2/out/student/t2-train-rules-s401-d320-p00003.values.npy")
                .exists()
        );
        // A skipped part.
        p["skip_parts"] = json!([4]);
        let mut ctx = Ctx::local("student-test", &base.join("c3"), 4);
        let r = chunk_with(&mut ctx, &p, 1, &search, table, json!({}), None).expect("runs");
        assert_eq!(r["skipped"], true);
        let _ = fs::remove_dir_all(&base);
    }
}
