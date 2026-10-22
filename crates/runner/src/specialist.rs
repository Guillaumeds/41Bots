//! The `specialist` chunk kind: the network S that is trained only on the trick-3 positions G2
//! (`nn::leafrisk`) flags as risky, for the routed play check (docs/m3-hybrid/specialist.md).
//!
//! ```json
//! { "name": "specialist", "priority": 1, "kind": "specialist", "chunks": 1,
//!   "params": { "trick": 3, "net_input": "o3-f5766a", "risk_input": "g2-o3-v1",
//!     "threshold": 277338260,
//!     "shards": [ { "input": "base-part-1", "role": "train" },
//!                 { "input": "val-s401", "role": "val" } ],
//!     "widths": "512,256,64", "epochs": 40, "batch": 512, "lr": 0.001, "patience": 4,
//!     "seed": 1, "threads": "all", "device": "auto" } }
//! ```
//! Each stored shard (a manifest input, `role` `train` or `val`) is filtered to the positions G2
//! flags at `threshold` (`search risk-filter`: the play rule's own cut, scored with O_3's value),
//! the flagged rows of each role are the training and the held-out set, and the trainer
//! (`fortyone_nn.train`) starts S from random weights on the training rows, validating on the
//! held-out ones. Nothing but the flagged rows is trained on. Exactly one `val` shard is needed.
//!
//! Outputs in `out/specialist/`: the flagged shards' sidecars, the trainer's files (`float` and
//! `int` safetensors, the network S), the logs and `record.json`. Every flagged shard and S go to
//! the content-addressed store (`inputs/sha256/<hex>`), a failure being a warning.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde_json::{Value, json};

use crate::Ctx;
use crate::bundle::sha256_file;
use crate::hybrid::{Device, Tools, choose_group, gpus_listed, python_command, uv_sync};
use crate::remote::{input_key, put_input};
use crate::spike::build_table_from;

/// The settings a `specialist` chunk takes; any other key is refused.
pub(crate) const KEYS: [&str; 14] = [
    "trick",
    "net_input",
    "risk_input",
    "threshold",
    "shards",
    "widths",
    "epochs",
    "batch",
    "lr",
    "patience",
    "seed",
    "threads",
    "device",
    "table",
];

/// One stored shard to filter, and its role.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Shard {
    pub input: String,
    pub role: String,
}

/// The checked settings.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Params {
    pub net_input: String,
    pub risk_input: String,
    pub threshold: i64,
    pub shards: Vec<Shard>,
    pub widths: String,
    pub epochs: u64,
    pub batch: u64,
    pub lr: f64,
    pub patience: u64,
    pub seed: u64,
    pub threads: Option<u64>,
    pub device: Device,
}

fn whole(p: &Value, k: &str, d: u64) -> Result<u64, String> {
    match p.get(k) {
        None | Some(Value::Null) => Ok(d),
        Some(x) => x
            .as_u64()
            .ok_or_else(|| format!("specialist: {k} must be a whole number")),
    }
}

/// Checks the settings (also when the manifest is read).
pub(crate) fn parse(p: &Value) -> Result<Params, String> {
    let o = p
        .as_object()
        .ok_or("specialist: params must be an object")?;
    for k in o.keys() {
        if !KEYS.contains(&k.as_str()) && k != "net" && k != "risk" && k != "shard_paths" {
            return Err(format!("specialist: unknown setting `{k}`"));
        }
    }
    if whole(p, "trick", 3)? != 3 {
        return Err("specialist: trick must be 3 (the O_3 leaves; the play rule's cut)".into());
    }
    let text = |k: &str| -> Result<String, String> {
        p.get(k)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("specialist: {k} is required"))
    };
    let threshold = p
        .get("threshold")
        .and_then(Value::as_i64)
        .ok_or("specialist: threshold (a whole number, the play rule's t) is required")?;
    let shards = p
        .get("shards")
        .and_then(Value::as_array)
        .ok_or("specialist: shards (a list of { input, role }) is required")?
        .iter()
        .map(|s| {
            let input = s
                .get("input")
                .and_then(Value::as_str)
                .ok_or("specialist: a shard needs `input`")?;
            let role = s.get("role").and_then(Value::as_str).unwrap_or("");
            if role != "train" && role != "val" {
                return Err(format!(
                    "specialist: shard {input} needs role \"train\" or \"val\""
                ));
            }
            Ok(Shard {
                input: input.to_string(),
                role: role.to_string(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    if shards.iter().filter(|s| s.role == "val").count() != 1 {
        return Err("specialist: exactly one shard has role \"val\"".into());
    }
    if !shards.iter().any(|s| s.role == "train") {
        return Err("specialist: at least one shard has role \"train\"".into());
    }
    let device = match p.get("device").and_then(Value::as_str).unwrap_or("auto") {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda,
        "auto" => Device::Auto,
        other => {
            return Err(format!(
                "specialist: device {other:?} is not cpu, cuda or auto"
            ));
        }
    };
    let lr = p.get("lr").and_then(Value::as_f64).unwrap_or(0.001);
    if lr.is_nan() || lr <= 0.0 {
        return Err("specialist: lr must be positive".into());
    }
    Ok(Params {
        net_input: text("net_input")?,
        risk_input: text("risk_input")?,
        threshold,
        shards,
        widths: p
            .get("widths")
            .and_then(Value::as_str)
            .unwrap_or("512,256,64")
            .to_string(),
        epochs: whole(p, "epochs", 40)?,
        batch: whole(p, "batch", 512)?,
        lr,
        patience: whole(p, "patience", 4)?,
        seed: whole(p, "seed", 1)?,
        threads: match p.get("threads") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) if s == "all" => None,
            Some(_) => Some(whole(p, "threads", 0)?),
        },
        device,
    })
}

/// The manifest inputs a specialist names: its network, its G2 model and every shard.
pub(crate) fn input_names(p: &Value) -> Result<Vec<String>, String> {
    let q = parse(p)?;
    let mut names = vec![q.net_input, q.risk_input];
    names.extend(q.shards.into_iter().map(|s| s.input));
    Ok(names)
}

/// Adds the local paths of the network, the G2 model and every shard (`shard_paths`) from the
/// worker's inputs.
pub(crate) fn resolve(
    p: &Value,
    inputs: &std::collections::BTreeMap<String, String>,
) -> Result<Value, String> {
    let q = parse(p)?;
    let find = |name: &str| {
        inputs
            .get(name)
            .cloned()
            .ok_or_else(|| format!("specialist needs input `{name}` in the manifest's inputs"))
    };
    let mut out = p.clone();
    out["net"] = json!(find(&q.net_input)?);
    out["risk"] = json!(find(&q.risk_input)?);
    let mut paths = serde_json::Map::new();
    for s in &q.shards {
        paths.insert(s.input.clone(), json!(find(&s.input)?));
    }
    out["shard_paths"] = Value::Object(paths);
    Ok(out)
}

/// Run one `specialist` chunk (its files resolved by `v2.rs`).
pub(crate) fn chunk(ctx: &mut Ctx, p: &Value) -> Result<Value, String> {
    let q = parse(p)?;
    let tools = Tools::find()?;
    ctx.set_phase(
        "chunk",
        json!({ "kind": "specialist", "step": "setup: table v0", "partial_cpu_ok": true }),
    );
    let (table, table_rec) = build_table_from(ctx, p.get("table").unwrap_or(&json!({})), "auto")?;
    let path = |k: &str| -> Result<PathBuf, String> {
        p.get(k)
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .ok_or_else(|| format!("specialist: {k} is not resolved to a file"))
    };
    let net = path("net")?;
    let risk = path("risk")?;
    let work = ctx.root.join("specialist-work");
    let out = ctx.root.join("out").join("specialist");
    let mut r = build(ctx, &tools, &q, &net, &risk, &table, p, &work, &out)?;
    r["table"] = table_rec;
    Ok(r)
}

/// Filters, stores, trains. Returns the chunk's record (without the table's).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build(
    ctx: &mut Ctx,
    tools: &Tools,
    q: &Params,
    net: &Path,
    risk: &Path,
    table: &Path,
    p: &Value,
    work: &Path,
    out: &Path,
) -> Result<Value, String> {
    let t0 = Instant::now();
    fs::create_dir_all(work).map_err(|e| e.to_string())?;
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let threads = q.threads.map_or(ctx.threads.max(1), |t| t as usize).max(1);
    let mut warnings: Vec<String> = Vec::new();
    let mut flagged: Vec<(Shard, PathBuf, Value)> = Vec::new();
    // 1. Filter every stored shard to the flagged rows (kept when already finished).
    let paths = p.get("shard_paths").cloned().unwrap_or(json!({}));
    for s in &q.shards {
        let src = PathBuf::from(
            paths[&s.input]
                .as_str()
                .ok_or_else(|| format!("specialist: no local file for `{}`", s.input))?,
        );
        let dst = work.join(format!("flagged-{}.npy", s.input));
        let t = Instant::now();
        if !dst.exists() {
            ctx.set_phase(
                "chunk",
                json!({ "kind": "specialist", "step": format!("filter {} ({})", s.input, s.role),
                        "partial_cpu_ok": false }),
            );
            let mut c = Command::new(&tools.search);
            c.args(["risk-filter", "--shard"])
                .arg(&src)
                .arg("--out")
                .arg(&dst)
                .arg("--equity")
                .arg(table)
                .arg("--net")
                .arg(net)
                .arg("--risk-model")
                .arg(risk)
                .args(["--threshold", &q.threshold.to_string()])
                .args(["--threads", &threads.to_string()]);
            ctx.run_child(c, &format!("specialist/filter-{}.log", s.input))?;
        }
        let (sha, bytes) = sha256_file(&dst)?;
        let mut rec = json!({ "input": s.input, "role": s.role, "sha256": sha, "bytes": bytes,
                              "seconds": (t.elapsed().as_secs_f64() * 10.0).round() / 10.0 });
        if let Some(r) = ctx.remote() {
            let d = dst.clone();
            match ctx.blocking(move || put_input(&*r, &d, None)) {
                Ok((sha, _, how)) => rec["stored"] = json!(format!("{} ({how})", input_key(&sha))),
                Err(e) => {
                    warnings.push(format!("the flagged {} shard was not stored: {e}", s.input))
                }
            }
        }
        // The rows of this shard's own sidecar (its count) are read back from the file header.
        let rows = nn::shard::from_npy(&fs::read(&dst).map_err(|e| e.to_string())?)?.len();
        rec["rows"] = json!(rows);
        flagged.push((s.clone(), dst, rec));
    }
    let train: Vec<PathBuf> = flagged
        .iter()
        .filter(|f| f.0.role == "train")
        .map(|f| f.1.clone())
        .collect();
    let val = flagged
        .iter()
        .find(|f| f.0.role == "val")
        .map(|f| f.1.clone())
        .ok_or("no val shard")?;
    // 2. The Python environment for the device, and the trainer from random weights.
    ctx.set_phase(
        "chunk",
        json!({ "kind": "specialist", "step": "python environment", "partial_cpu_ok": true }),
    );
    let gpus = gpus_listed(&tools.nvidia_smi);
    let choice = choose_group(q.device, &gpus)?;
    for w in &choice.warnings {
        warnings.push(w.clone());
    }
    let env_dir = work.join(format!("pyenv-{}", choice.group));
    let (python, uv_version, _) =
        uv_sync(ctx, tools, choice.group, &env_dir, "specialist/uv-sync.log")?;
    let abs = |p: &Path| std::path::absolute(p).map_err(|e| e.to_string());
    let dir = abs(&work.join("train-out"))?;
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    ctx.set_phase(
        "chunk",
        json!({ "kind": "specialist", "step": "train S (random start)", "partial_cpu_ok": true }),
    );
    let mut c = python_command(&python, &tools.python_dir, "fortyone_nn.train");
    c.arg("--train");
    for t in &train {
        c.arg(abs(t)?);
    }
    c.arg("--val")
        .arg(abs(&val)?)
        .arg("--out")
        .arg(&dir)
        .args(["--head", "outcome", "--widths", &q.widths])
        .args(["--epochs", &q.epochs.to_string()])
        .args(["--batch", &q.batch.to_string()])
        .args(["--lr", &q.lr.to_string()])
        .args(["--patience", &q.patience.to_string()])
        .args(["--seed", &q.seed.to_string()])
        .args(["--threads", &threads.to_string()])
        .args([
            "--device",
            match q.device {
                Device::Cuda => "cuda",
                Device::Cpu => "cpu",
                Device::Auto => "auto",
            },
        ])
        .arg("--rows-dir")
        .arg(abs(&work.join("train-rows"))?);
    ctx.run_child(c, "specialist/train.log")?;
    // 3. Store S (both files) and record it.
    let mut nets = serde_json::Map::new();
    for name in ["int.safetensors", "float.safetensors"] {
        let f = dir.join(name);
        let (sha, bytes) = sha256_file(&f)?;
        fs::copy(&f, out.join(name)).map_err(|e| e.to_string())?;
        let mut rec = json!({ "sha256": sha, "bytes": bytes });
        if let Some(r) = ctx.remote() {
            let g = f.clone();
            match ctx.blocking(move || put_input(&*r, &g, None)) {
                Ok((sha, _, how)) => rec["stored"] = json!(format!("{} ({how})", input_key(&sha))),
                Err(e) => warnings.push(format!("{name} was not stored: {e}")),
            }
        }
        nets.insert(name.into(), rec);
    }
    ctx.set_phase(
        "chunk",
        json!({ "kind": "specialist", "step": "done", "partial_cpu_ok": true }),
    );
    let shards: Vec<Value> = flagged.into_iter().map(|f| f.2).collect();
    let record = json!({
        "kind": "specialist", "trick": 3, "threshold": q.threshold,
        "widths": q.widths, "epochs": q.epochs, "lr": q.lr, "seed": q.seed,
        "flagged": shards, "networks": nets,
        "python": { "uv": uv_version, "group": choice.group },
        "threads": threads, "warnings": warnings,
        "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
    });
    fs::write(
        out.join("record.json"),
        serde_json::to_vec_pretty(&record).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Value {
        json!({ "trick": 3, "net_input": "o3", "risk_input": "g2", "threshold": 277338260,
                "shards": [ { "input": "base-1", "role": "train" },
                            { "input": "val", "role": "val" } ] })
    }

    #[test]
    fn settings_are_read_with_defaults() {
        let q = parse(&params()).unwrap();
        assert_eq!(q.threshold, 277338260);
        assert_eq!(q.widths, "512,256,64");
        assert_eq!(q.epochs, 40);
        assert_eq!(q.shards.len(), 2);
        assert_eq!(q.threads, None);
    }

    #[test]
    fn wrong_settings_are_refused() {
        let mut p = params();
        p["trick"] = json!(2);
        assert!(parse(&p).is_err(), "only trick 3");
        let mut p = params();
        p["shards"] = json!([ { "input": "a", "role": "train" } ]);
        assert!(
            parse(&p).unwrap_err().contains("exactly one"),
            "a val shard is needed"
        );
        let mut p = params();
        p["shards"] = json!([ { "input": "a", "role": "val" } ]);
        assert!(
            parse(&p).unwrap_err().contains("train"),
            "a train shard is needed"
        );
        let mut p = params();
        p["threshold"] = json!("high");
        assert!(parse(&p).is_err());
        let mut p = params();
        p["lr"] = json!(0);
        assert!(parse(&p).is_err());
        let mut p = params();
        p["bogus"] = json!(1);
        assert!(parse(&p).unwrap_err().contains("unknown"));
    }

    #[test]
    fn resolve_adds_the_local_paths() {
        let mut inputs = std::collections::BTreeMap::new();
        for (k, v) in [
            ("o3", "/w/o3"),
            ("g2", "/w/g2"),
            ("base-1", "/w/b1"),
            ("val", "/w/v"),
        ] {
            inputs.insert(k.to_string(), v.to_string());
        }
        let r = resolve(&params(), &inputs).unwrap();
        assert_eq!(r["net"], "/w/o3");
        assert_eq!(r["risk"], "/w/g2");
        assert_eq!(r["shard_paths"]["base-1"], "/w/b1");
        inputs.remove("g2");
        assert!(resolve(&params(), &inputs).is_err());
    }
}
