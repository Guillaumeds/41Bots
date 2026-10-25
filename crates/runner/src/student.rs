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
//! table v0 built, or the `equity` input), `equity_input`, `exact_leaves` and `defer_expert`
//! (below); any other key is refused ([`KEYS`]).
//!
//! **Exact leaves** (`exact_leaves`, default none; docs/m3-hybrid/learned-leaves.md): the syntax
//! of `search teacher --exact-leaves` (`search::exact_leaves`): in each deal the hybrid's
//! frontier positions the policy picks are valued by their expert instead of the network, e.g.
//! the adopted rule for trick-2 labels `"learned:inputs/sha256/<hex>:t=T"` (H_3 with learned
//! exact leaves). `defer_expert` (default exact; needs `exact_leaves`) hands the picked leaves to
//! a deeper hybrid instead, `"h3:NET:SPEC"` (learned deferral, for trick-1 labels with H_2). A
//! learned model and an expert's network are manifest inputs, named by input name or by stored
//! key `inputs/sha256/<hex>` (the scheduler turns a key into the input listing that SHA-256, and
//! refuses one no input lists: fetched and hash-checked like every input); `search teacher`
//! checks that a model is one made for its network's leaves (trick and SHA-256). Without them
//! the command, the shard and its name are exactly as before; with them the shard's name ends
//! `-x<8 hex>` (of the settings as written) and its stamp names the policy and its SHA-256s.

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
pub(crate) const KEYS: [&str; 23] = [
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
    "exact_leaves",
    "defer_expert",
    "teacher",
    "student1",
    "student2",
    "playout_cut",
];

/// The keys [`resolve`] adds (local paths of the named inputs) before the chunk reads its
/// settings again.
const RESOLVED: [&str; 5] = [
    "net",
    "exact_leaves_path",
    "defer_expert_path",
    "student1_path",
    "student2_path",
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
    /// Exact leaves as written (learned models by input name; `None`: none).
    pub exact_leaves: Option<String>,
    /// Their expert as written (`None`: exact).
    pub defer_expert: Option<String>,
    /// Input names of the leaf-risk models (the expert's own included) and of the expert's
    /// networks.
    pub models: Vec<String>,
    pub expert_nets: Vec<String>,
    /// The playout teacher (`"teacher": "playout"`, docs/STUDENT.md): the input names of the
    /// students S1 and S2 that play the deals out, and the trick at whose start the exact solve
    /// takes over (`playout_cut`, default 3). `None`: the search teacher (`net_input`).
    pub playout: Option<(String, String, u64)>,
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
        if !KEYS.contains(&k.as_str()) && !RESOLVED.contains(&k.as_str()) {
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
    if !["rules", "mixed", "random", "student"].contains(&source.as_str()) {
        return Err(format!(
            "student_teacher: source {source}: rules, mixed, random or student"
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
    let opt = |k: &str| -> Result<Option<String>, String> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(x)) if x.trim() == "none" && k == "exact_leaves" => Ok(None),
            Some(Value::String(x)) => Ok(Some(x.clone())),
            Some(_) => Err(format!("student_teacher: {k} must be a string")),
        }
    };
    let (exact_leaves, defer_expert, models, expert_nets) = crate::checks::leaf_settings(
        opt("exact_leaves")?.as_deref(),
        opt("defer_expert")?.as_deref(),
        "student_teacher",
    )?;
    let upload = match p.get("upload") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err("student_teacher: upload must be true or false".into()),
    };
    let teacher = s("teacher", Some("search"))?;
    let playout = match teacher.as_str() {
        "search" => {
            for k in ["student1", "student2", "playout_cut"] {
                if p.get(k).is_some_and(|v| !v.is_null()) {
                    return Err(format!(
                        "student_teacher: {k} is for the playout teacher (\"teacher\": \"playout\")"
                    ));
                }
            }
            None
        }
        "playout" => {
            if exact_leaves.is_some() || defer_expert.is_some() {
                return Err(
                    "student_teacher: exact_leaves and defer_expert are for the search teacher"
                        .into(),
                );
            }
            let cut = n("playout_cut", Some(3))?;
            if !(1..=6).contains(&cut) {
                return Err(format!(
                    "student_teacher: playout_cut {cut}: the trick at whose start the exact solve \
                     takes over, 1 to 5, or 6 to play every trick out"
                ));
            }
            Some((s("student1", None)?, s("student2", None)?, cut))
        }
        t => {
            return Err(format!(
                "student_teacher: teacher {t}: \"search\" (a network at the cut, net_input) or \
                 \"playout\" (the students play the deals out)"
            ));
        }
    };
    // The source `student` plays the positions with the playout teacher's students (the
    // positions the students themselves reach, docs/STUDENT.md 6: DAgger).
    if source == "student" && playout.is_none() {
        return Err(
            "student_teacher: source student plays the positions with the playout teacher's \
             students: it needs \"teacher\": \"playout\""
                .into(),
        );
    }
    // The search teacher needs its network; the playout teacher none (`net_input` is left out
    // or empty there).
    let net_input = match (&playout, p.get("net_input")) {
        (Some(_), None | Some(Value::Null)) => String::new(),
        (Some(_), Some(Value::String(x))) if x.is_empty() => String::new(),
        (Some(_), Some(_)) => {
            return Err(
                "student_teacher: the playout teacher takes no net_input (its students play the deals out)"
                    .into(),
            );
        }
        (None, _) => s("net_input", None)?,
    };
    Ok(Params {
        trick,
        count,
        deals,
        seed,
        source,
        role,
        net_input,
        tt_mb: n("tt_mb", Some(64))?.max(1),
        start_hands: n("start_hands", Some(20_000))?,
        first_part: n("first_part", Some(0))?,
        skip_parts,
        upload,
        exact_leaves,
        defer_expert,
        models,
        expert_nets,
        playout,
    })
}

/// The inputs a stream names (its network, or the playout teacher's students).
pub(crate) fn input_names(p: &Value) -> Result<Vec<String>, String> {
    let q = parse(p)?;
    let mut v = match &q.playout {
        Some((s1, s2, _)) => vec![s1.clone(), s2.clone()],
        None => vec![q.net_input],
    };
    for n in q.models.into_iter().chain(q.expert_nets) {
        if !v.contains(&n) {
            v.push(n);
        }
    }
    v.dedup();
    Ok(v)
}

/// The settings with each learned model and expert network named by stored key
/// (`inputs/sha256/<hex>`) turned into the name of the manifest input listing that SHA-256
/// (`inputs`: name and SHA-256 in lowercase hex, as the scheduler reads them), so the scheduler fetches and checks it like any
/// input. A key no input lists is left as it is (and then refused as an unknown input).
pub(crate) fn name_keyed_inputs(p: &Value, inputs: &[(String, String)]) -> Value {
    let mut out = p.clone();
    for k in ["exact_leaves", "defer_expert"] {
        let Some(text) = p.get(k).and_then(Value::as_str) else {
            continue;
        };
        let mut t = text.to_string();
        for (name, hex) in inputs {
            t = t.replace(&input_key(hex), name);
        }
        out[k] = json!(t);
    }
    out
}

/// Adds the network's local path (`net`) from the worker's `inputs` (name → path).
pub(crate) fn resolve(
    p: &Value,
    inputs: &std::collections::BTreeMap<String, String>,
) -> Result<Value, String> {
    let q = parse(p)?;
    let path_of = |n: &str| -> Result<String, String> {
        inputs
            .get(n)
            .cloned()
            .ok_or_else(|| format!("student_teacher needs input `{n}` in the manifest's inputs"))
    };
    let mut out = p.clone();
    match &q.playout {
        Some((s1, s2, _)) => {
            out["student1_path"] = json!(path_of(s1)?);
            out["student2_path"] = json!(path_of(s2)?);
        }
        None => out["net"] = json!(path_of(&q.net_input)?),
    }
    if let Some(l) = &q.exact_leaves {
        out["exact_leaves_path"] = json!(crate::checks::leaves_with_paths(l, &q.models, &path_of)?);
    }
    if let Some(e) = &q.defer_expert {
        out["defer_expert_path"] = json!(crate::checks::expert_with_paths(e, &path_of, &path_of)?);
    }
    Ok(out)
}

/// The shard's file name for part `part`.
pub(crate) fn shard_name(q: &Params, part: u64) -> String {
    let x = match (&q.exact_leaves, &q.defer_expert) {
        (None, None) => String::new(),
        (l, e) => {
            let text = format!(
                "{}|{}",
                l.as_deref().unwrap_or("none"),
                e.as_deref().unwrap_or("exact")
            );
            format!("-x{}", &crate::bundle::sha256_hex(text.as_bytes())[..8])
        }
    };
    let pl = match &q.playout {
        Some((_, _, cut)) => format!("-pl{cut}"),
        None => String::new(),
    };
    format!(
        "t{}-{}-{}-s{}-d{}{x}{pl}-p{part:05}.npy",
        q.trick, q.role, q.source, q.seed, q.deals
    )
}

/// The teacher's networks on the command line: the search teacher's `net` (an O_k's path, or
/// `exact`), or the playout teacher's two students.
pub(crate) enum Nets<'a> {
    /// `--net PATH`.
    Net(&'a str),
    /// `--playout --student1 S1 --student2 S2 --playout-cut K`.
    Playout { s1: &'a str, s2: &'a str, cut: u64 },
}

/// The `search teacher` command of part `part`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn command(
    search: &Path,
    q: &Params,
    part: u64,
    nets: Nets,
    leaves: Option<(&str, Option<&str>)>,
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
        .args(["--role", &q.role]);
    match nets {
        Nets::Net(net) => {
            c.args(["--net", net]);
        }
        Nets::Playout { s1, s2, cut } => {
            c.arg("--playout")
                .args(["--student1", s1])
                .args(["--student2", s2])
                .args(["--playout-cut", &cut.to_string()]);
        }
    }
    c.arg("--equity")
        .arg(table)
        .arg("--out")
        .arg(out)
        .args(["--threads", &threads.max(1).to_string()])
        .args(["--tt-mb", &q.tt_mb.to_string()])
        .args(["--start-hands", &q.start_hands.to_string()])
        .arg("--progress")
        .arg(progress);
    if let Some((l, e)) = leaves {
        c.args(["--exact-leaves", l]);
        if let Some(e) = e {
            c.args(["--defer-expert", e]);
        }
    }
    c
}

/// The networks of a resolved chunk's settings (`resolve` added the paths).
pub(crate) fn nets_of<'a>(p: &'a Value, q: &Params) -> Result<Nets<'a>, String> {
    let path = |k: &str| -> Result<&str, String> {
        p.get(k)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("student_teacher: {k} is not resolved to a file"))
    };
    Ok(match &q.playout {
        Some((_, _, cut)) => Nets::Playout {
            s1: path("student1_path")?,
            s2: path("student2_path")?,
            cut: *cut,
        },
        None => Nets::Net(path("net")?),
    })
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
    let nets = nets_of(p, &q)?;
    let work = ctx.root.join("student-work");
    let outdir = ctx.root.join("out").join("student");
    fs::create_dir_all(&work).map_err(|e| e.to_string())?;
    fs::create_dir_all(&outdir).map_err(|e| e.to_string())?;
    let name = shard_name(&q, part);
    let shard = work.join(&name);
    let progress = work.join("progress.json");
    let t = Instant::now();
    let path = |k: &str| -> Result<String, String> {
        p.get(k)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("student_teacher: {k} was not resolved"))
    };
    let leaves = match (&q.exact_leaves, &q.defer_expert) {
        (None, _) => None,
        (Some(_), None) => Some((path("exact_leaves_path")?, None)),
        (Some(_), Some(_)) => Some((path("exact_leaves_path")?, Some(path("defer_expert_path")?))),
    };
    let cmd = command(
        search,
        &q,
        part,
        nets,
        leaves.as_ref().map(|(l, e)| (l.as_str(), e.as_deref())),
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
    let mut r = json!({
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
    });
    if let Some(l) = &q.exact_leaves {
        r["exact_leaves"] = json!(l);
        r["defer_expert"] = json!(q.defer_expert.as_deref().unwrap_or("exact"));
        r["teacher"] = side["teacher"].clone();
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Value {
        json!({ "threads": "all", "trick": 2, "count": 5000, "deals": 320, "seed": 401,
                "source": "rules", "net_input": "o3" })
    }

    #[cfg(unix)]
    fn base_params() -> Value {
        base()
    }

    #[test]
    fn the_playout_teacher_names_its_students_and_no_network() {
        let mut p = base();
        p.as_object_mut().expect("object").remove("net_input");
        p["teacher"] = json!("playout");
        p["student1"] = json!("s1");
        p["student2"] = json!("s2");
        let q = parse(&p).expect("parses");
        assert_eq!(q.playout, Some(("s1".into(), "s2".into(), 3)));
        assert_eq!(q.net_input, "");
        assert_eq!(input_names(&p).expect("names"), vec!["s1", "s2"]);
        assert_eq!(shard_name(&q, 3), "t2-train-rules-s401-d320-pl3-p00003.npy");
        let paths = std::collections::BTreeMap::from([
            ("s1".to_string(), "/x/s1".to_string()),
            ("s2".to_string(), "/x/s2".to_string()),
        ]);
        let r = resolve(&p, &paths).expect("resolves");
        assert_eq!(
            (r["student1_path"].as_str(), r["student2_path"].as_str()),
            (Some("/x/s1"), Some("/x/s2"))
        );
        assert!(r.get("net").is_none());
        let nets = nets_of(&r, &q).expect("nets");
        let c = command(
            Path::new("search"),
            &q,
            3,
            nets,
            None,
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
        assert!(args.iter().any(|a| a == "--playout"));
        assert!(!args.iter().any(|a| a == "--net"));
        assert_eq!(
            (at("--student1"), at("--student2"), at("--playout-cut")),
            ("/x/s1".into(), "/x/s2".into(), "3".into())
        );
        // The cut is checked; a network is refused; the search teacher refuses the students.
        p["playout_cut"] = json!(7);
        assert!(parse(&p).is_err());
        p["playout_cut"] = json!(3);
        p["net_input"] = json!("o3");
        assert!(parse(&p).is_err());
        let mut s = base();
        s["student1"] = json!("s1");
        assert!(parse(&s).is_err());
        // The source `student` (the students' own positions) needs the playout teacher, and names
        // its shards by it.
        p["net_input"] = Value::Null;
        p["source"] = json!("student");
        let q = parse(&p).expect("parses");
        assert_eq!(
            shard_name(&q, 0),
            "t2-train-student-s401-d320-pl3-p00000.npy"
        );
        let mut s = base();
        s["source"] = json!("student");
        let e = parse(&s).expect_err("no students to play with");
        assert!(e.contains("source student"), "{e}");
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
            Nets::Net("o3.safetensors"),
            None,
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

    /// Exact leaves: none leaves the command and the name as before; a learned model by stored
    /// key becomes the input listing it, its path goes to `--exact-leaves`, an expert's to
    /// `--defer-expert`; a key no input lists, or an expert without leaves, is refused.
    #[test]
    fn exact_leaves_are_inputs_and_args() {
        let hex = "20c41678c64681855003e5b8660e0a2e1fb9e8d552f7d898dd956055c767a9d5";
        let q0 = parse(&base()).expect("parses");
        let mut p = base();
        p["exact_leaves"] = json!("none");
        assert_eq!(parse(&p).expect("none"), q0);
        assert_eq!(shard_name(&q0, 1), "t2-train-rules-s401-d320-p00001.npy");
        let mut p = base();
        p["exact_leaves"] = json!(format!("learned:inputs/sha256/{hex}:t=277338260"));
        let inputs = vec![
            ("o3".to_string(), "f".repeat(64)),
            ("risk".to_string(), hex.to_string()),
        ];
        let named = name_keyed_inputs(&p, &inputs);
        assert_eq!(named["exact_leaves"], "learned:risk:t=277338260");
        assert_eq!(
            input_names(&named).expect("names"),
            vec!["o3".to_string(), "risk".to_string()]
        );
        // A key no input lists stays a key: an unknown input.
        let unlisted = name_keyed_inputs(&p, &inputs[..1]);
        assert!(input_names(&unlisted).expect("names")[1].starts_with("inputs/sha256/"));
        let q = parse(&named).expect("parses");
        assert!(shard_name(&q, 1).starts_with("t2-train-rules-s401-d320-x"));
        let paths = std::collections::BTreeMap::from([
            ("o3".to_string(), "/x/o3".to_string()),
            ("risk".to_string(), "/x/risk.safetensors".to_string()),
            ("o3b".to_string(), "/x/o3b".to_string()),
        ]);
        let r = resolve(&named, &paths).expect("resolves");
        assert_eq!(
            r["exact_leaves_path"],
            "learned:/x/risk.safetensors:t=277338260"
        );
        let c = command(
            Path::new("search"),
            &q,
            1,
            Nets::Net("/x/o3"),
            Some((
                r["exact_leaves_path"].as_str().expect("path"),
                Some("h3:/x/o3b"),
            )),
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
        assert_eq!(
            at("--exact-leaves"),
            "learned:/x/risk.safetensors:t=277338260"
        );
        assert_eq!(at("--defer-expert"), "h3:/x/o3b");
        // Trick 1 with a deeper expert: its network and model are inputs too.
        let mut p1 = base();
        p1["trick"] = json!(1);
        p1["exact_leaves"] = json!("learned:risk2:t=5");
        p1["defer_expert"] = json!("h3:o3b:learned:risk:t=277338260");
        let names = input_names(&p1).expect("names");
        for n in ["o3", "risk2", "risk", "o3b"] {
            assert!(names.contains(&n.to_string()), "{n}");
        }
        let mut bad = base();
        bad["defer_expert"] = json!("h3:o3b");
        assert!(parse(&bad).is_err());
        let mut bad = base();
        bad["exact_leaves"] = json!(3);
        assert!(parse(&bad).is_err());
    }

    /// The G v2 relabelling manifest parses: its learned model named by stored key becomes the
    /// `leafrisk_o3` input, and its streams are the stored data's (same seeds, sources, parts).
    #[test]
    fn the_gv2_manifest_parses() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/vast");
        let v: Value = serde_json::from_str(
            &fs::read_to_string(root.join("student-t2-gv2.manifest.json")).expect("read"),
        )
        .expect("JSON");
        let m = crate::sched::parse_v2(&v).expect("parses");
        let mut decisions = 0;
        for s in &m.streams {
            let q = parse(&s.params).expect("params");
            assert_eq!(
                q.exact_leaves.as_deref(),
                Some("learned:leafrisk_o3:t=277338260")
            );
            assert!(
                input_names(&s.params)
                    .expect("names")
                    .contains(&"leafrisk_o3".into())
            );
            if q.role == "train" {
                let crate::sched::Chunks::Fixed(n) = s.chunks else {
                    panic!("fixed chunks")
                };
                decisions += q.count * n;
            }
        }
        assert_eq!(decisions, 1_000_000);
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
