//! The `leafrisk` chunk kind: the rows, training and threshold of a leaf-risk model
//! (`nn::leafrisk`, docs/m3-hybrid/learned-leaves.md) for the oracle network O_k at the start of
//! trick `k`, on one CPU machine with every usable thread.
//!
//! ```json
//! { "name": "leafrisk", "priority": 1, "kind": "leafrisk", "chunks": 1,
//!   "params": { "threads": "all", "trick": 2, "net_input": "o2-505a0c",
//!     "splits": [ { "name": "train", "count": 60000, "seed": 78001 },
//!                 { "name": "cal", "count": 20000, "seed": 78002 },
//!                 { "name": "test", "count": 20000, "seed": 78003 } ],
//!     "harvest_cap": 32, "harvest_positions": "0.4,0.3,0.3", "start_hands": 20000,
//!     "train": { "trees": 150, "depth": 6, "rate": 0.05, "min_leaf": 400, "subsample": 0.5,
//!                "seed": 1, "holdout": 0.1 },
//!     "targets": "2.0,1.8,1.6,1.4,1.2,1.0,0.9,0.8,0.7,0.6", "delta": 0.05 } }
//! ```
//! The rows are what the hybrid H_k really meets: a **network-driven harvest** (`search labels
//! --trick k --source harvest --harvest-net <O_k>`, decisions at trick `k − 1` searched by H_k,
//! at most `harvest_cap` of the trick-`k` frontier positions each search reaches kept per
//! decision, each labelled exactly), one shard per split with its own seed (separate decisions;
//! never O_k's training seeds). At trick 2 an exact label is about 3.6 thread-seconds, which is
//! why the rows are made here and not on the build VM. Then, in `python/nn`'s CPU environment
//! (`uv sync --group cpu`): `fortyone_nn.leafrisk prepare` (O_k's integer value of every row, its
//! squared error and the inputs), `train` (the trees; the model records O_k's trick and SHA-256)
//! and `calibrate --write-budget` (Learn then Test thresholds on the `cal` split, checked on the
//! `test` split, and the budget table in the model). Splits must be named `train`, `cal` and
//! `test`.
//!
//! Outputs in `out/leafrisk/`: `model.json`, `calibration.json`, the shards' sidecars, the logs;
//! the shards themselves (and the model) go to the content-addressed store (`inputs/sha256/<hex>`,
//! a failure being a warning) and stay in the chunk's work directory, not the bundle. A chunk
//! resumes split by split (a finished shard with its sidecar is kept). Optional: `threads`,
//! `harvest_cap` (32), `harvest_positions` ("0.4,0.3,0.3"), `start_hands` (20,000), `tt_mb`
//! (the label solves' table, `search labels`' own default when absent), `train`, `targets`,
//! `delta` (0.05), `table`, `equity_input`, `est_seconds`; any other key is refused ([`KEYS`]).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde_json::{Value, json};

use crate::Ctx;
use crate::bundle::sha256_file;
use crate::hybrid::{Tools, python_command, uv_sync};
use crate::remote::{input_key, put_input};
use crate::spike::build_table_from;

/// The settings a `leafrisk` chunk takes (after `v2.rs` added the network's path, `net`); any
/// other key is refused.
pub(crate) const KEYS: [&str; 14] = [
    "trick",
    "net_input",
    "splits",
    "harvest_cap",
    "harvest_positions",
    "start_hands",
    "tt_mb",
    "train",
    "targets",
    "delta",
    "table",
    "equity_input",
    "threads",
    "est_seconds",
];

/// The training settings (`fortyone_nn.leafrisk train`'s flags).
const TRAIN_KEYS: [&str; 7] = [
    "trees",
    "depth",
    "rate",
    "min_leaf",
    "subsample",
    "seed",
    "holdout",
];

/// One split of the rows.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Split {
    pub name: String,
    pub count: u64,
    pub seed: u64,
}

/// The checked settings.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Params {
    pub trick: u64,
    pub net_input: String,
    pub splits: Vec<Split>,
    pub harvest_cap: u64,
    pub harvest_positions: String,
    pub start_hands: u64,
    pub tt_mb: Option<u64>,
    /// `--flag value` pairs for `train`.
    pub train: Vec<(String, String)>,
    pub targets: String,
    pub delta: f64,
}

/// Checks the settings (also when the manifest is read).
pub(crate) fn parse(p: &Value) -> Result<Params, String> {
    let o = p.as_object().ok_or("leafrisk: params must be an object")?;
    for k in o.keys() {
        if !KEYS.contains(&k.as_str()) && k != "net" {
            return Err(format!("leafrisk: unknown setting `{k}`"));
        }
    }
    let n = |k: &str, d: u64| -> Result<u64, String> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(x) => x
                .as_u64()
                .ok_or_else(|| format!("leafrisk: {k} must be a whole number")),
        }
    };
    let trick = n("trick", 2)?;
    if !(2..=5).contains(&trick) {
        return Err(format!(
            "leafrisk: trick {trick}: the leaves' trick (the network's frontier), 2 to 5"
        ));
    }
    let net_input = p
        .get("net_input")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or("leafrisk: needs net_input (the O_k whose errors it predicts)")?
        .to_string();
    let mut splits: Vec<Split> = Vec::new();
    for s in p
        .get("splits")
        .and_then(Value::as_array)
        .ok_or("leafrisk: needs splits: [{name, count, seed}] for train, cal and test")?
    {
        for k in s.as_object().into_iter().flat_map(|o| o.keys()) {
            if !["name", "count", "seed"].contains(&k.as_str()) {
                return Err(format!("leafrisk: a split: unknown key `{k}`"));
            }
        }
        let name = s["name"].as_str().unwrap_or_default().to_string();
        let count = s["count"].as_u64().filter(|&c| c > 0);
        let seed = s["seed"].as_u64().filter(|&x| x > 1);
        let (Some(count), Some(seed)) = (count, seed) else {
            return Err(format!(
                "leafrisk: split `{name}`: a count above 0 and a seed (never 1, the golden sets')"
            ));
        };
        if splits.iter().any(|x| x.name == name || x.seed == seed) {
            return Err(format!(
                "leafrisk: split `{name}`: names and seeds must differ between splits"
            ));
        }
        splits.push(Split { name, count, seed });
    }
    let mut names: Vec<&str> = splits.iter().map(|s| s.name.as_str()).collect();
    names.sort_unstable();
    if names != ["cal", "test", "train"] {
        return Err("leafrisk: splits must be exactly train, cal and test".into());
    }
    let harvest_positions = p
        .get("harvest_positions")
        .and_then(Value::as_str)
        .unwrap_or("0.4,0.3,0.3")
        .to_string();
    let shares: Vec<f64> = harvest_positions
        .split(',')
        .map(|x| x.trim().parse::<f64>().unwrap_or(-1.0))
        .collect();
    if shares.len() != 3 || shares.iter().any(|x| !(0.0..=1.0).contains(x)) {
        return Err(format!(
            "leafrisk: harvest_positions {harvest_positions:?}: three shares (seats 1-2, 3-4, 5-8)"
        ));
    }
    let mut train = Vec::new();
    if let Some(t) = p.get("train").filter(|v| !v.is_null()) {
        let o = t.as_object().ok_or("leafrisk: train must be an object")?;
        for (k, v) in o {
            if !TRAIN_KEYS.contains(&k.as_str()) || !v.is_number() {
                return Err(format!(
                    "leafrisk: train.{k}: one of {} with a number",
                    TRAIN_KEYS.join(", ")
                ));
            }
            train.push((format!("--{}", k.replace('_', "-")), v.to_string()));
        }
    }
    let targets = p
        .get("targets")
        .and_then(Value::as_str)
        .unwrap_or("2.0,1.8,1.6,1.4,1.2,1.0,0.9,0.8,0.7,0.6,0.5")
        .to_string();
    if targets
        .split(',')
        .any(|x| x.trim().parse::<f64>().map_or(true, |v| v <= 0.0))
    {
        return Err(format!(
            "leafrisk: targets {targets:?}: SDs left per team, in percent"
        ));
    }
    let delta = match p.get("delta") {
        None | Some(Value::Null) => 0.05,
        Some(v) => v
            .as_f64()
            .filter(|d| *d > 0.0 && *d < 1.0)
            .ok_or("leafrisk: delta must be between 0 and 1")?,
    };
    Ok(Params {
        trick,
        net_input,
        splits,
        harvest_cap: n("harvest_cap", 32)?.max(1),
        harvest_positions,
        start_hands: n("start_hands", 20_000)?,
        tt_mb: match p.get("tt_mb") {
            None | Some(Value::Null) => None,
            Some(_) => Some(n("tt_mb", 0)?.max(1)),
        },
        train,
        targets,
        delta,
    })
}

/// The manifest inputs a chunk's settings name.
pub(crate) fn input_names(p: &Value) -> Result<Vec<String>, String> {
    Ok(vec![parse(p)?.net_input])
}

/// Adds the network's local path (`net`) from the worker's inputs.
pub(crate) fn resolve(
    p: &Value,
    inputs: &std::collections::BTreeMap<String, String>,
) -> Result<Value, String> {
    let q = parse(p)?;
    let path = inputs.get(&q.net_input).ok_or_else(|| {
        format!(
            "leafrisk needs input `{}` in the manifest's inputs",
            q.net_input
        )
    })?;
    let mut out = p.clone();
    out["net"] = json!(path);
    Ok(out)
}

/// The `search labels` command of one split's harvest.
pub(crate) fn labels_command(
    search: &Path,
    q: &Params,
    s: &Split,
    net: &str,
    table: &Path,
    threads: usize,
    out: &Path,
) -> Command {
    let mut c = Command::new(search);
    c.args(["labels", "--trick", &q.trick.to_string()])
        .args([
            "--count",
            &s.count.to_string(),
            "--seed",
            &s.seed.to_string(),
        ])
        .args(["--source", "harvest", "--harvest-net", net])
        .args(["--harvest-cap", &q.harvest_cap.to_string()])
        .args(["--harvest-positions", &q.harvest_positions])
        .args(["--start-hands", &q.start_hands.to_string()])
        .args(["--threads", &threads.to_string()])
        .arg("--equity")
        .arg(table)
        .arg("--out")
        .arg(out);
    if let Some(t) = q.tt_mb {
        c.args(["--tt-mb", &t.to_string()]);
    }
    c
}

/// Run one `leafrisk` chunk (its network resolved to a path by `v2.rs`).
pub(crate) fn chunk(ctx: &mut Ctx, p: &Value) -> Result<Value, String> {
    let q = parse(p)?;
    let tools = Tools::find()?;
    ctx.set_phase(
        "chunk",
        json!({ "kind": "leafrisk", "step": "setup: table v0", "partial_cpu_ok": true }),
    );
    let (table, table_rec) = build_table_from(ctx, p.get("table").unwrap_or(&json!({})), "auto")?;
    let net = p
        .get("net")
        .and_then(Value::as_str)
        .ok_or("leafrisk: net_input is not resolved to a file")?
        .to_string();
    let work = ctx.root.join("leafrisk-work");
    let out = ctx.root.join("out").join("leafrisk");
    let mut r = build(ctx, &tools, &q, &net, &table, &work, &out, None)?;
    r["table"] = table_rec;
    Ok(r)
}

/// The rows, training and calibration of a leaf-risk model for the network `net` (module
/// notes): shards in `work`, the model, calibration and sidecars in `out`. `python`: an
/// installed environment with numpy (the round's, `hybrid::round`), else the CPU group is
/// installed in `work`. Returns the chunk's record (without the table's).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build(
    ctx: &mut Ctx,
    tools: &Tools,
    q: &Params,
    net: &str,
    table: &Path,
    work: &Path,
    out: &Path,
    python: Option<&Path>,
) -> Result<Value, String> {
    let t0 = Instant::now();
    fs::create_dir_all(work).map_err(|e| e.to_string())?;
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let search = tools.search.clone();
    let threads = ctx.threads.max(1);
    let mut splits = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    // 1. The rows: one network-driven harvest per split (kept when already finished).
    for s in &q.splits {
        let shard = work.join(format!("t{}-{}-s{}.npy", q.trick, s.name, s.seed));
        let side = PathBuf::from(format!("{}.json", shard.display()));
        let t = Instant::now();
        if !(shard.exists() && side.exists()) {
            ctx.set_phase(
                "chunk",
                json!({ "kind": "leafrisk", "step": format!("rows: {} ({} decisions' positions)", s.name, s.count),
                        "partial_cpu_ok": false }),
            );
            ctx.run_child(
                labels_command(&search, q, s, net, table, threads, &shard),
                &format!("leafrisk/labels-{}.log", s.name),
            )?;
        }
        fs::copy(&side, out.join(format!("{}.npy.json", s.name))).map_err(|e| e.to_string())?;
        let (sha, bytes) = sha256_file(&shard)?;
        let mut rec = json!({ "name": s.name, "count": s.count, "seed": s.seed,
                              "sha256": sha, "bytes": bytes,
                              "seconds": (t.elapsed().as_secs_f64() * 10.0).round() / 10.0 });
        if let Some(r) = ctx.remote() {
            let sh = shard.clone();
            match ctx.blocking(move || put_input(&*r, &sh, None)) {
                Ok((sha, _, how)) => {
                    rec["stored"] = json!(format!("{} ({how})", input_key(&sha)));
                }
                Err(e) => warnings.push(format!("the {} shard was not stored: {e}", s.name)),
            }
        }
        splits.push(rec);
    }
    // 2. The Python environment (CPU) and 3. prepare, train, calibrate.
    ctx.set_phase(
        "chunk",
        json!({ "kind": "leafrisk", "step": "python environment", "partial_cpu_ok": true }),
    );
    let (python, uv_version) = match python {
        Some(p) => (p.to_path_buf(), Value::Null),
        None => {
            let env_dir = work.join("pyenv-cpu");
            let (py, v, _) = uv_sync(ctx, tools, "cpu", &env_dir, "leafrisk/uv-sync.log")?;
            (py, json!(v))
        }
    };
    let net_abs = std::path::absolute(net).map_err(|e| e.to_string())?;
    let npz = |name: &str| work.join(format!("{name}.npz"));
    for s in &q.splits {
        if npz(&s.name).exists() {
            continue;
        }
        ctx.set_phase(
            "chunk",
            json!({ "kind": "leafrisk", "step": format!("prepare {}", s.name), "partial_cpu_ok": true }),
        );
        let shard = work.join(format!("t{}-{}-s{}.npy", q.trick, s.name, s.seed));
        let part = work.join(format!("{}.part.npz", s.name));
        let mut c = python_command(&python, &tools.python_dir, "fortyone_nn.leafrisk");
        c.arg("prepare")
            .arg("--net")
            .arg(&net_abs)
            .arg("--shard")
            .arg(std::path::absolute(&shard).map_err(|e| e.to_string())?)
            .arg("--out")
            .arg(std::path::absolute(&part).map_err(|e| e.to_string())?);
        ctx.run_child(c, &format!("leafrisk/prepare-{}.log", s.name))?;
        fs::rename(&part, npz(&s.name)).map_err(|e| e.to_string())?;
    }
    let model = std::path::absolute(out.join("model.json")).map_err(|e| e.to_string())?;
    ctx.set_phase(
        "chunk",
        json!({ "kind": "leafrisk", "step": "train", "partial_cpu_ok": true }),
    );
    let abs = |p: PathBuf| std::path::absolute(p).map_err(|e| e.to_string());
    let mut c = python_command(&python, &tools.python_dir, "fortyone_nn.leafrisk");
    c.arg("train")
        .arg("--data")
        .arg(abs(npz("train"))?)
        .arg("--out")
        .arg(&model);
    for (k, v) in &q.train {
        c.args([k, v]);
    }
    ctx.run_child(c, "leafrisk/train.log")?;
    ctx.set_phase(
        "chunk",
        json!({ "kind": "leafrisk", "step": "calibrate", "partial_cpu_ok": true }),
    );
    let calib = abs(out.join("calibration.json"))?;
    let mut c = python_command(&python, &tools.python_dir, "fortyone_nn.leafrisk");
    c.arg("calibrate")
        .arg("--model")
        .arg(&model)
        .arg("--cal")
        .arg(abs(npz("cal"))?)
        .arg("--test")
        .arg(abs(npz("test"))?)
        .arg("--out")
        .arg(&calib)
        .args(["--delta", &q.delta.to_string()])
        .args(["--targets", &q.targets])
        .arg("--write-budget");
    ctx.run_child(c, "leafrisk/calibrate.log")?;
    let (model_sha, model_bytes) = sha256_file(&model)?;
    let mut stored = Value::Null;
    if let Some(r) = ctx.remote() {
        let m = model.clone();
        match ctx.blocking(move || put_input(&*r, &m, None)) {
            Ok((sha, _, how)) => stored = json!(format!("{} ({how})", input_key(&sha))),
            Err(e) => warnings.push(format!("the model was not stored: {e}")),
        }
    }
    let net_sha = sha256_file(Path::new(net))?.0;
    ctx.set_phase(
        "chunk",
        json!({ "kind": "leafrisk", "step": "done", "partial_cpu_ok": true }),
    );
    Ok(json!({
        "kind": "leafrisk", "trick": q.trick, "net_sha256": net_sha,
        "splits": splits,
        "model": { "sha256": model_sha, "bytes": model_bytes, "stored": stored },
        "python": { "uv": uv_version },
        "threads": threads, "warnings": warnings,
        "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Value {
        json!({ "threads": "all", "trick": 2, "net_input": "o2",
                "splits": [ { "name": "train", "count": 600, "seed": 78001 },
                            { "name": "cal", "count": 200, "seed": 78002 },
                            { "name": "test", "count": 200, "seed": 78003 } ],
                "train": { "trees": 150, "min_leaf": 400 } })
    }

    /// The settings and their refusals; the harvest's command (a network-driven trick-2 harvest
    /// of O_2's frontier, the split's own seed).
    #[test]
    fn settings_and_command() {
        let q = parse(&params()).expect("valid");
        assert_eq!((q.trick, q.harvest_cap, q.delta), (2, 32, 0.05));
        assert_eq!(q.harvest_positions, "0.4,0.3,0.3");
        assert_eq!(input_names(&params()).expect("names"), ["o2"]);
        assert!(q.train.contains(&("--min-leaf".into(), "400".into())));
        let bad = |f: &dyn Fn(&mut Value)| {
            let mut p = params();
            f(&mut p);
            parse(&p).is_err()
        };
        assert!(bad(&|p| p["typo"] = json!(1)));
        assert!(bad(&|p| p["trick"] = json!(1)));
        assert!(bad(&|p| p["net_input"] = json!(null)));
        assert!(bad(&|p| p["splits"][1]["name"] = json!("val")));
        assert!(bad(&|p| p["splits"][1]["seed"] = json!(78001)));
        assert!(bad(&|p| p["splits"][1]["seed"] = json!(1)));
        assert!(bad(&|p| p["splits"][0]["count"] = json!(0)));
        assert!(bad(&|p| p["train"]["lr"] = json!(1)));
        assert!(bad(&|p| p["harvest_positions"] = json!("0.5,0.5")));
        assert!(bad(&|p| p["targets"] = json!("1,x")));
        let inputs = std::collections::BTreeMap::from([("o2".to_string(), "/in/o2".to_string())]);
        assert_eq!(
            resolve(&params(), &inputs).expect("resolves")["net"],
            "/in/o2"
        );
        assert!(resolve(&params(), &Default::default()).is_err());
        let c = labels_command(
            Path::new("search"),
            &q,
            &q.splits[1],
            "/in/o2",
            Path::new("eq.npy"),
            8,
            Path::new("cal.npy"),
        );
        let a: Vec<String> = c
            .get_args()
            .map(|x| x.to_string_lossy().into_owned())
            .collect();
        let s = a.join(" ");
        assert!(
            s.starts_with(
                "labels --trick 2 --count 200 --seed 78002 --source harvest --harvest-net /in/o2 \
                 --harvest-cap 32 --harvest-positions 0.4,0.3,0.3 --start-hands 20000 --threads 8"
            ),
            "{s}"
        );
    }
}
