//! The `search_measure` chunk kind (docs/M3-SPEC.md §13.1): the early-trick agreement curve,
//! the exact-PIMC yardstick and the method-vs-yardstick scorer, run by the sibling `search`
//! binary.
//!
//! ```json
//! { "search_measure": { "trick": 2, "count": 200, "seed": 1, "to": 320,
//!     "first": 0, "last": 199, "grid": "10,20,40,80,160,320",
//!     "solve_threads": 1, "tt_mb": 256, "start_hands": 20000,
//!     "methods": [ { "method": "rollout", "label": 2, "to": 64 } ],
//!     "table": { "hands": 1000000, "seed": 1 } } }
//! ```
//! Steps: build and verify equity table v0; `search decisions` (stratified per stratum and per
//! position in the trick); `search worlds --method exact` for decisions `first..=last` up to `to`
//! worlds (label 1, nested sets; resumable); `search agree` (per trick, per position and per
//! position pair); then each listed method's worlds and `search score` against the yardstick at
//! the largest grid N available. With `solve_threads > 1` each exact world is solved on that many
//! threads (`solve_fast_par`, trick 1) and worlds run one at a time; otherwise worlds run in
//! parallel on all threads.

use std::fs;
use std::process::Command;

use serde_json::{Value, json};

use crate::spike::build_table;
use crate::{Ctx, sibling};

fn n(p: &Value, k: &str, d: u64) -> u64 {
    p.get(k).and_then(Value::as_u64).unwrap_or(d)
}

/// Run one `search_measure` chunk.
pub(crate) fn chunk(ctx: &mut Ctx, p: &Value) -> Result<Value, String> {
    let (table, table_rec) = build_table(ctx, p.get("table").unwrap_or(&json!({})))?;
    let out = ctx.root.join("out").join("measure");
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let trick = n(p, "trick", 2);
    let count = n(p, "count", 200);
    let seed = n(p, "seed", 1);
    let to = n(p, "to", 320);
    let solve_threads = n(p, "solve_threads", 1).max(1);
    let threads = if solve_threads > 1 {
        1
    } else {
        ctx.threads as u64
    };
    let grid = p
        .get("grid")
        .and_then(Value::as_str)
        .unwrap_or("10,20,40,80,160,320")
        .to_string();
    let dec = out.join(format!("decisions-t{trick}.jsonl"));
    let mut c = Command::new(sibling("search"));
    c.args(["decisions", "--trick", &trick.to_string()])
        .args(["--count", &count.to_string(), "--seed", &seed.to_string()])
        .args(["--start-hands", &n(p, "start_hands", 20_000).to_string()])
        .arg("--out")
        .arg(&dec);
    ctx.set_phase(
        "chunk",
        json!({ "kind": "search_measure", "step": "decisions" }),
    );
    ctx.run_child(c, "measure/search.log")?;
    let worlds = |label: u64, method: &str, to: u64, file: &std::path::Path| {
        let mut c = Command::new(sibling("search"));
        c.args(["worlds", "--method", method, "--to", &to.to_string()])
            .args(["--label", &label.to_string(), "--seed", &seed.to_string()])
            .args(["--first", &n(p, "first", 0).to_string()])
            .args(["--last", &n(p, "last", count.saturating_sub(1)).to_string()])
            .args(["--threads", &threads.to_string()])
            .args(["--solve-threads", &solve_threads.to_string()])
            .args(["--tt-mb", &n(p, "tt_mb", 256).to_string()])
            .arg("--decisions")
            .arg(&dec)
            .arg("--equity")
            .arg(&table)
            .arg("--out")
            .arg(file);
        c
    };
    let yard = out.join(format!("exact-t{trick}-n{to}.jsonl"));
    ctx.set_phase(
        "chunk",
        json!({ "kind": "search_measure", "step": "exact worlds" }),
    );
    ctx.run_child(worlds(1, "exact", to, &yard), "measure/worlds.log")?;
    let mut c = Command::new(sibling("search"));
    c.arg("agree")
        .arg("--worlds")
        .arg(&yard)
        .args(["--grid", &grid])
        .arg("--json")
        .arg(out.join(format!("agree-t{trick}.json")));
    ctx.run_child(c, "measure/agree.log")?;
    let n_ref = grid
        .split(',')
        .filter_map(|x| x.parse::<u64>().ok())
        .filter(|&x| x <= to)
        .max()
        .unwrap_or(to);
    let mut methods = Vec::new();
    for (i, m) in p
        .get("methods")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        let method = m.get("method").and_then(Value::as_str).unwrap_or("rollout");
        let label = n(m, "label", 2 + i as u64);
        let mto = n(m, "to", 64);
        let f = out.join(format!("{method}-t{trick}-l{label}-n{mto}.jsonl"));
        ctx.set_phase(
            "chunk",
            json!({ "kind": "search_measure", "step": format!("method {i}") }),
        );
        ctx.run_child(worlds(label, method, mto, &f), "measure/worlds.log")?;
        methods.push(format!("{}:{mto}", f.display()));
    }
    if !methods.is_empty() {
        let mut c = Command::new(sibling("search"));
        c.arg("score")
            .arg("--yardstick")
            .arg(&yard)
            .args(["--n", &n_ref.to_string()]);
        for m in &methods {
            c.args(["--method", m]);
        }
        ctx.run_child(c, "measure/score.log")?;
    }
    Ok(
        json!({ "trick": trick, "count": count, "seed": seed, "to": to, "grid": grid,
               "solve_threads": solve_threads, "table": table_rec, "methods": methods }),
    )
}
