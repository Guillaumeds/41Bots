//! The `t1bench` chunk kind (docs/M3-SPEC.md §13.1 item 11): the fastest and cheapest way to
//! spread exact trick-1 world solves over this machine, measured on a fixed small set of real
//! yardstick worlds (identical work on every machine, so machines compare directly). The bench
//! itself is `search bench-t1` (one run per configuration, each its own process so its peak
//! memory is its own) and `search::t1bench` (sets, measures, analysis).
//!
//! ```json
//! { "t1bench": { "dph": 1.25, "configs": ["16x", "8x", "32x", "4x", "auto", "1xall"],
//!                "budget_s": 240, "cap_s": 120, "late_cap_s": 45, "lead_cap_s": 150,
//!                "stop_ratio": 1.5 } }
//! ```
//! Every field is optional; the measured solving takes at most `budget_s + 2 × late_cap_s +
//! lead_cap_s` (8 minutes by default), much less on big machines. Steps:
//! 1. Table v0 (`table.source` `"auto"`: a format-2 input, else `inputs/sha256/<recorded hash>`
//!    in R2, else rebuilt from the model in the repository in seconds; hash-checked), and the
//!    stratified trick-1 decision set (`count` 32, `seed` 1: the yardstick's own).
//! 2. The mid set (`search::t1bench::MID_SET`, 48 worlds at positions 4–6) under each
//!    configuration in `configs`, in that order: `"<W>x"` = W worlds at once on T/W threads each,
//!    `"1xall"` = one world at a time on all T threads, `"<W>x<S>"` explicit, `"auto"` = the
//!    position-aware plan. Each run's tables are sized from the memory per configuration
//!    (`mem.rs`; `tt_mb` may fix them). A run stops when it reaches `cap_s`, or once its
//!    occupancy (thread-seconds held) passes `stop_ratio` × the best complete run's: it cannot be
//!    the best. A run starts only while the mid runs' measured time is under `budget_s`.
//! 3. The late set (128 worlds at positions 7–8) with one thread per world (`"<T>x1"`), then
//!    with the best mid configuration, each capped at `late_cap_s`.
//! 4. The lead set (one world at each of positions 1–3) once with `lead_config` (default
//!    `"3x"`: all three at once on a third of the threads each, so a huge leader world does not
//!    hold up the others), capped at `lead_cap_s`: their node counts scale every configuration
//!    to the real position mix (a world stopped at the cap counts as a lower bound). These
//!    counts belong to the game, not the machine: `position_nodes` (8 mean node counts, e.g.
//!    from the yardstick run or an earlier bench) replaces them and skips this step.
//! 5. `out/t1bench/results.json` (`search::t1bench::analyse`) and `out/t1bench/summary.txt`
//!    (plain English): per run, worlds per hour for the real position mix, CPU use, peak memory,
//!    table fill and dollars per 1000 worlds at `dph` (default: the launcher's `DPH`); the best
//!    configuration and the best per position; a projection for a yardstick of `golden`
//!    (default `[192, 200]`) decisions × worlds. Every world solved by several runs must have
//!    byte-identical values; otherwise the chunk fails after writing its results.
//!
//! The heartbeat marks the whole bench `partial_cpu_ok` (the slow configurations are what it
//! measures).

use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use search::t1bench::{
    AnalyseParams, BenchRun, LATE_SET, LEAD_SET, MID_SET, analyse, summary_text, totals,
};
use serde_json::{Value, json};

use crate::mem::{self, Requested, Tables};
use crate::spike::build_table_from;
use crate::{Ctx, hardware, sibling};

/// The default mid-set configurations, roughly shortest first.
const CONFIGS: [&str; 6] = ["16x", "8x", "32x", "4x", "auto", "1xall"];

/// A configuration on `t` threads: `Some((worlds at once, threads per world))`, or `None` for
/// the position-aware plan.
pub(crate) fn parse_config(name: &str, t: u64) -> Result<Option<(u64, u64)>, String> {
    let bad = || format!("t1bench: unknown configuration {name:?} (e.g. 8x, 1xall, 4x96, auto)");
    if name == "auto" {
        return Ok(None);
    }
    let (w, s) = name.split_once('x').ok_or_else(bad)?;
    let w: u64 = w.parse().map_err(|_| bad())?;
    let s = match s {
        "all" => t,
        "" => t / w.max(1),
        n => n.parse().map_err(|_| bad())?,
    };
    if w == 0 || s == 0 || w * s > t.max(1) {
        return Err(format!(
            "t1bench: {name} needs {} threads; this machine has {t}",
            w * s.max(1)
        ));
    }
    Ok(Some((w, s)))
}

fn f(p: &Value, k: &str, d: f64) -> f64 {
    p.get(k).and_then(Value::as_f64).unwrap_or(d)
}

struct Bench<'a> {
    dec: &'a Path,
    table: &'a Path,
    out: &'a Path,
    p: &'a Value,
    runs: Vec<BenchRun>,
    measured_s: f64,
    skipped: Vec<Value>,
    warnings: Vec<String>,
}

impl Bench<'_> {
    /// One `search bench-t1` run; `None` if the configuration does not fit this machine.
    fn run(
        &mut self,
        ctx: &mut Ctx,
        name: &str,
        config: &str,
        set: &str,
        cap_s: f64,
        stop_occ: Option<f64>,
    ) -> Result<Option<BenchRun>, String> {
        let t = ctx.threads as u64;
        let k = self.runs.len();
        let file = self.out.join(format!("run-{k:02}-{name}.json"));
        let progress = ctx.root.join("t1bench-progress.json");
        let _ = fs::remove_file(&progress);
        let mut c = Command::new(sibling("search"));
        c.arg("bench-t1")
            .arg("--decisions")
            .arg(self.dec)
            .arg("--equity")
            .arg(self.table)
            .args(["--set", set, "--name", name])
            .args(["--threads", &t.to_string()])
            .args(["--cap-s", &format!("{cap_s:.1}")])
            .arg("--json")
            .arg(&file)
            .arg("--progress")
            .arg(&progress);
        if let Some(x) = stop_occ {
            c.args(["--stop-occ-s", &format!("{x:.1}")]);
        }
        let tt = match parse_config(config, t) {
            Err(e) => {
                self.skipped.push(json!({ "run": name, "why": e }));
                return Ok(None);
            }
            Ok(Some((w, s))) => {
                let tables = if s > 1 {
                    Tables::Shared {
                        count: w,
                        solve_threads: s,
                    }
                } else {
                    Tables::Own { count: w }
                };
                let tt = mem::plan_here(
                    Requested::from_param(self.p, "tt_mb")?,
                    tables,
                    self.table,
                    &format!("t1bench {name}"),
                );
                c.args(["--schedule", "fixed"])
                    .args(["--worlds-at-once", &w.to_string()])
                    .args(["--solve-threads", &s.to_string()])
                    .args(["--tt-mb", &tt.tt_mb.to_string()]);
                tt
            }
            Ok(None) => {
                let (tt, b) = mem::budget_here(self.table, t, &format!("t1bench {name}"));
                c.args([
                    "--schedule",
                    "auto",
                    "--tt-budget-mb",
                    &(b >> 20).to_string(),
                ]);
                if let Some(s) = self.p.get("schedule").filter(|v| v.is_object()) {
                    c.args(["--sched-json", &s.to_string()]);
                }
                tt
            }
        };
        if let Some(w) = &tt.warning {
            self.warnings.push(format!("{name}: {w}"));
        }
        let warnings = self.warnings.clone();
        let (runs_done, total_runs) = (
            k,
            self.p.get("configs").map_or(CONFIGS.len(), |c| {
                c.as_array().map_or(CONFIGS.len(), Vec::len)
            }) + 3,
        );
        let label = name.to_string();
        ctx.set_phase(
            "chunk",
            json!({ "kind": "t1bench", "run": label, "runs_done": runs_done,
                    "partial_cpu_ok": true, "warnings": warnings }),
        );
        let t0 = Instant::now();
        ctx.run_child_with(c, "t1bench/search.log", &mut |ctx| {
            let pr: Value = fs::read_to_string(&progress)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or(Value::Null);
            ctx.progress = json!({
                "kind": "t1bench", "run": label, "runs_done": runs_done,
                "runs_planned": total_runs,
                "worlds_done": pr["worlds_done"], "worlds_total": pr["worlds_total"],
                "running": pr["running"], "partial_cpu_ok": true, "warnings": warnings,
            });
        })?;
        self.measured_s += t0.elapsed().as_secs_f64();
        let mut r: BenchRun = serde_json::from_str(
            &fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?,
        )
        .map_err(|e| format!("{}: {e}", file.display()))?;
        r.schedule["config"] = json!(config);
        r.schedule["tt"] = tt.record;
        let tot = totals(&r);
        println!(
            "runner: t1bench {name} ({config}): {}/{} worlds, {:.0} s, occupancy {:.0} thread-s{}",
            tot.finished,
            tot.worlds,
            r.wall_s,
            tot.occupancy_s,
            r.stopped
                .as_deref()
                .map_or(String::new(), |s| format!(", stopped ({s})"))
        );
        self.runs.push(r.clone());
        Ok(Some(r))
    }
}

/// Run one `t1bench` chunk.
pub(crate) fn chunk(ctx: &mut Ctx, p: &Value) -> Result<Value, String> {
    let (table, table_rec) = build_table_from(ctx, p.get("table").unwrap_or(&json!({})), "auto")?;
    let out = ctx.root.join("out").join("t1bench");
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let count = p.get("count").and_then(Value::as_u64).unwrap_or(32);
    let seed = p.get("seed").and_then(Value::as_u64).unwrap_or(1);
    let dec = out.join("decisions-t1.jsonl");
    let mut c = Command::new(sibling("search"));
    c.args(["decisions", "--trick", "1"])
        .args(["--count", &count.to_string(), "--seed", &seed.to_string()])
        .args([
            "--start-hands",
            &p.get("start_hands")
                .and_then(Value::as_u64)
                .unwrap_or(20_000)
                .to_string(),
        ])
        .arg("--out")
        .arg(&dec);
    ctx.set_phase(
        "chunk",
        json!({ "kind": "t1bench", "step": "decisions", "partial_cpu_ok": true }),
    );
    ctx.run_child(c, "t1bench/search.log")?;
    let set = |k: &str, d: &str| p.get(k).and_then(Value::as_str).unwrap_or(d).to_string();
    let (mid, late, lead) = (
        set("mid_set", MID_SET),
        set("late_set", LATE_SET),
        set("lead_set", LEAD_SET),
    );
    let configs: Vec<String> = match p.get("configs") {
        None => CONFIGS.iter().map(|s| s.to_string()).collect(),
        Some(v) => v
            .as_array()
            .ok_or("t1bench: configs must be a list")?
            .iter()
            .map(|x| {
                x.as_str()
                    .map(str::to_string)
                    .ok_or("t1bench: configs are names like \"8x\"".to_string())
            })
            .collect::<Result<_, _>>()?,
    };
    for c in &configs {
        // Unknown names fail now; too big for this machine is only skipped later.
        parse_config(c, u64::MAX / 2)?;
    }
    let budget_s = f(p, "budget_s", 240.0);
    let cap_s = f(p, "cap_s", 120.0);
    let late_cap_s = f(p, "late_cap_s", 45.0);
    let stop_ratio = f(p, "stop_ratio", 1.5).max(1.0);
    let t = ctx.threads as u64;
    let mut b = Bench {
        dec: &dec,
        table: &table,
        out: &out,
        p,
        runs: Vec::new(),
        measured_s: 0.0,
        skipped: Vec::new(),
        warnings: Vec::new(),
    };
    // The ranking: every configuration on the mid set.
    let mut best: Option<(String, f64)> = None;
    for cfg in &configs {
        let left = budget_s - b.measured_s;
        if left < 10.0 {
            b.skipped
                .push(json!({ "run": cfg, "why": "bench time budget used" }));
            continue;
        }
        let stop_occ = best.as_ref().map(|(_, o)| o * stop_ratio);
        if let Some(r) = b.run(ctx, cfg, cfg, &mid, cap_s.min(left), stop_occ)? {
            let tot = totals(&r);
            if tot.finished == tot.worlds && best.as_ref().is_none_or(|(_, o)| tot.occupancy_s < *o)
            {
                best = Some((cfg.clone(), tot.occupancy_s));
            }
        }
    }
    // Late positions (their own caps, outside the mid budget): one thread per world, then the
    // best mid configuration.
    let one = format!("{t}x1");
    let mut late_occ = None;
    if let Some(r) = b.run(ctx, "late-1thread", &one, &late, late_cap_s, None)? {
        let tot = totals(&r);
        late_occ = (tot.finished == tot.worlds).then_some(tot.occupancy_s);
    }
    if let Some((cfg, _)) = &best
        && *cfg != one
    {
        b.run(
            ctx,
            &format!("late-{cfg}"),
            cfg,
            &late,
            late_cap_s,
            late_occ.map(|o| o * stop_ratio),
        )?;
    }
    // The big positions once, three at once (a huge leader world does not hold up the other
    // two), for their node counts; not needed when the node counts are given.
    let position_nodes: Option<[f64; 8]> = match p.get("position_nodes") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            serde_json::from_value(v.clone())
                .map_err(|_| "t1bench: position_nodes must be 8 numbers (positions 1-8)")?,
        ),
    };
    let lead_cfg = p
        .get("lead_config")
        .and_then(Value::as_str)
        .unwrap_or("3x")
        .to_string();
    if position_nodes.is_some() && p.get("lead_set").is_none() {
        b.skipped
            .push(json!({ "run": "lead", "why": "position_nodes given" }));
    } else {
        b.run(
            ctx,
            &format!("lead-{lead_cfg}"),
            &lead_cfg,
            &lead,
            f(p, "lead_cap_s", 150.0),
            None,
        )?;
    }
    // Analysis.
    let hw = hardware();
    let params = AnalyseParams {
        threads: ctx.threads,
        dph: p
            .get("dph")
            .and_then(Value::as_f64)
            .unwrap_or(ctx.dph)
            .max(0.0),
        golden_decisions: p
            .get("golden")
            .and_then(|g| g.get(0))
            .and_then(Value::as_u64)
            .unwrap_or(192),
        golden_worlds: p
            .get("golden")
            .and_then(|g| g.get(1))
            .and_then(Value::as_u64)
            .unwrap_or(200),
        stop_ratio,
        cpu_model: hw["cpu_model"].as_str().unwrap_or("").to_string(),
        mem_gb: mem::Sys::host().limit().mb() as f64 / 1024.0,
        mid_set: mid.clone(),
        position_nodes,
    };
    let mut a = analyse(&b.runs, &params);
    a["table"] = table_rec.clone();
    a["skipped"] = json!(b.skipped);
    a["warnings"] = json!(b.warnings);
    a["measured_s"] = json!((b.measured_s * 10.0).round() / 10.0);
    a["sets"] = json!({ "mid": mid, "late": late, "lead": lead });
    a["configs"] = json!(configs);
    let text = summary_text(&a, &params)
        + &format!(
            "\nMeasured solving took {:.0} s in all (table: {}).\n",
            b.measured_s,
            table_rec["source"].as_str().unwrap_or("?")
        );
    fs::write(
        out.join("results.json"),
        serde_json::to_string_pretty(&a).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| e.to_string())?;
    fs::write(out.join("summary.txt"), &text).map_err(|e| e.to_string())?;
    println!("{text}");
    if a["ok"].as_bool() != Some(true) {
        return Err(format!(
            "t1bench: configurations gave different values for {} world(s): this is a bug; \
             see out/t1bench/results.json (identity.mismatches)",
            a["identity"]["mismatches"].as_array().map_or(0, Vec::len)
        ));
    }
    Ok(json!({
        "kind": "t1bench", "threads": ctx.threads, "dph": params.dph,
        "best_config": a["best_config"], "combined": a["combined"], "golden": a["golden"],
        "identity": a["identity"]["worlds_compared"], "measured_s": a["measured_s"],
        "table": table_rec, "warnings": b.warnings,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configs_parse() {
        assert_eq!(parse_config("auto", 256), Ok(None));
        assert_eq!(parse_config("8x", 256), Ok(Some((8, 32))));
        assert_eq!(parse_config("1xall", 256), Ok(Some((1, 256))));
        assert_eq!(parse_config("256x1", 256), Ok(Some((256, 1))));
        assert_eq!(parse_config("4x96", 384), Ok(Some((4, 96))));
        assert_eq!(parse_config("32x", 368), Ok(Some((32, 11))));
        assert!(
            parse_config("8x", 4).is_err(),
            "too many worlds for 4 threads"
        );
        assert!(parse_config("4x2", 4).is_err());
        assert!(parse_config("fast", 64).is_err());
        assert!(parse_config("0x", 64).is_err());
    }
}
