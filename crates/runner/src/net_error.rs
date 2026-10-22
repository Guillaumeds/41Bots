//! The `net_error` chunk kind: one network's value error against a stored shard's exact values, on
//! all its trick-3 rows and on the rows G2 flags at the play rule's threshold (`search net-error`,
//! `search::net_error`). The held-out comparison for the specialist S (docs/m3-hybrid/specialist.md).
//!
//! ```json
//! { "name": "net_error", "priority": 1, "kind": "net_error", "chunks": 1,
//!   "params": { "net_input": "o3-f5766a", "risk_input": "leafrisk-o3-v1", "shard_input": "val",
//!     "threshold": 277338260, "threads": "all", "table": { "hands": 1000000, "seed": 1 } } }
//! ```
//! Outputs in `out/net_error/`: `net-error.json` (the statistics of all rows and of the flagged
//! rows) and the log. The JSON is also stored in the content-addressed store when one is set.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use serde_json::{Value, json};

use crate::Ctx;
use crate::hybrid::Tools;
use crate::remote::{input_key, put_input};
use crate::spike::build_table_from;

/// The settings a `net_error` chunk takes; any other key is refused.
pub(crate) const KEYS: [&str; 6] = [
    "net_input",
    "risk_input",
    "shard_input",
    "threshold",
    "threads",
    "table",
];

/// The checked settings.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Params {
    pub net_input: String,
    pub risk_input: String,
    pub shard_input: String,
    pub threshold: i64,
    pub threads: Option<u64>,
}

/// Checks the settings (also when the manifest is read).
pub(crate) fn parse(p: &Value) -> Result<Params, String> {
    let o = p.as_object().ok_or("net_error: params must be an object")?;
    for k in o.keys() {
        if !KEYS.contains(&k.as_str()) && k != "net" && k != "risk" && k != "shard" {
            return Err(format!("net_error: unknown setting `{k}`"));
        }
    }
    let text = |k: &str| -> Result<String, String> {
        p.get(k)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("net_error: {k} is required"))
    };
    let threshold = p
        .get("threshold")
        .and_then(Value::as_i64)
        .ok_or("net_error: threshold (a whole number) is required")?;
    Ok(Params {
        net_input: text("net_input")?,
        risk_input: text("risk_input")?,
        shard_input: text("shard_input")?,
        threshold,
        threads: match p.get("threads") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) if s == "all" => None,
            Some(x) => Some(
                x.as_u64()
                    .ok_or("net_error: threads must be a whole number or \"all\"")?,
            ),
        },
    })
}

/// The manifest inputs a `net_error` chunk names.
pub(crate) fn input_names(p: &Value) -> Result<Vec<String>, String> {
    let q = parse(p)?;
    Ok(vec![q.net_input, q.risk_input, q.shard_input])
}

/// Adds the local paths of the network, the G2 model and the shard from the worker's inputs.
pub(crate) fn resolve(
    p: &Value,
    inputs: &std::collections::BTreeMap<String, String>,
) -> Result<Value, String> {
    let q = parse(p)?;
    let find = |name: &str| {
        inputs
            .get(name)
            .cloned()
            .ok_or_else(|| format!("net_error needs input `{name}` in the manifest's inputs"))
    };
    let mut out = p.clone();
    out["net"] = json!(find(&q.net_input)?);
    out["risk"] = json!(find(&q.risk_input)?);
    out["shard"] = json!(find(&q.shard_input)?);
    Ok(out)
}

/// Run one `net_error` chunk (its files resolved by `v2.rs`).
pub(crate) fn chunk(ctx: &mut Ctx, p: &Value) -> Result<Value, String> {
    let q = parse(p)?;
    let tools = Tools::find()?;
    ctx.set_phase(
        "chunk",
        json!({ "kind": "net_error", "step": "setup: table v0", "partial_cpu_ok": true }),
    );
    let (table, table_rec) = build_table_from(ctx, p.get("table").unwrap_or(&json!({})), "auto")?;
    let path = |k: &str| -> Result<PathBuf, String> {
        p.get(k)
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .ok_or_else(|| format!("net_error: {k} is not resolved to a file"))
    };
    let (net, risk, shard) = (path("net")?, path("risk")?, path("shard")?);
    let threads = q.threads.map_or(ctx.threads.max(1), |t| t as usize).max(1);
    let out = ctx.root.join("out").join("net_error");
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let json_path = out.join("net-error.json");
    let t0 = Instant::now();
    ctx.set_phase(
        "chunk",
        json!({ "kind": "net_error", "step": "errors on the shard's rows", "partial_cpu_ok": false }),
    );
    let mut c = std::process::Command::new(&tools.search);
    c.args(["net-error", "--shard"])
        .arg(&shard)
        .arg("--net")
        .arg(&net)
        .arg("--risk-model")
        .arg(&risk)
        .arg("--equity")
        .arg(&table)
        .args(["--threshold", &q.threshold.to_string()])
        .args(["--threads", &threads.to_string()])
        .arg("--out")
        .arg(&json_path);
    ctx.run_child(c, "net_error/net-error.log")?;
    let mut stored = Value::Null;
    let mut warnings: Vec<String> = Vec::new();
    if let Some(r) = ctx.remote() {
        let j = json_path.clone();
        match ctx.blocking(move || put_input(&*r, &j, None)) {
            Ok((sha, _, how)) => stored = json!(format!("{} ({how})", input_key(&sha))),
            Err(e) => warnings.push(format!("the result was not stored: {e}")),
        }
    }
    let report: Value = serde_json::from_slice(&fs::read(&json_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    Ok(json!({
        "kind": "net_error", "result": report, "stored": stored,
        "table": table_rec, "threads": threads, "warnings": warnings,
        "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_and_refusals() {
        let ok = json!({ "net_input": "o3", "risk_input": "g2", "shard_input": "val", "threshold": 277338260 });
        let q = parse(&ok).unwrap();
        assert_eq!(q.threshold, 277338260);
        assert_eq!(q.threads, None);
        let mut bad = ok.clone();
        bad["bogus"] = json!(1);
        assert!(parse(&bad).unwrap_err().contains("unknown"));
        let mut no_t = ok.clone();
        no_t.as_object_mut().unwrap().remove("threshold");
        assert!(parse(&no_t).is_err());
    }

    #[test]
    fn resolve_adds_paths_and_needs_all_inputs() {
        let p =
            json!({ "net_input": "o3", "risk_input": "g2", "shard_input": "val", "threshold": 1 });
        let mut inputs = std::collections::BTreeMap::new();
        for (k, v) in [("o3", "/w/o3"), ("g2", "/w/g2"), ("val", "/w/val")] {
            inputs.insert(k.to_string(), v.to_string());
        }
        let r = resolve(&p, &inputs).unwrap();
        assert_eq!(r["net"], "/w/o3");
        assert_eq!(r["shard"], "/w/val");
        inputs.remove("val");
        assert!(resolve(&p, &inputs).is_err());
    }
}
