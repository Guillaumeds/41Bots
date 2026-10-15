//! The `t1bench` chunk kind (docs/M3-SPEC.md §13.1 item 11): ranks machines by exact trick-1
//! solving work per dollar, with every thread busy the whole time. The windows are run by
//! `search bench-t1` (each its own process, so each has its own peak memory); the windows,
//! guard and analysis are `search::t1bench`.
//!
//! ```json
//! { "t1bench": { "dph": 1.25, "fill_s": 240, "prod_s": 120, "fill_tt_mb": 512,
//!                "sweep_mb": [64, 256, 1024, 2048], "sweep_s": 90,
//!                "min_cpu_pct": 80, "golden": [128, 200],
//!                "position_nodes": [1e10, 4.6e9, 2.2e9, 1e9, 2.8e8, 8e7, 1e8, 1.25e8] } }
//! ```
//! Every field is optional. Steps:
//! 1. **Setup** (the only stretch the heartbeat marks `partial_cpu_ok`; a warning if it takes
//!    over 90 s): table v0 (`table.source` `"auto"`: a format-2 input, else
//!    `inputs/sha256/<recorded hash>` in R2, else rebuilt from the model in the repository in
//!    seconds; hash-checked) and the stratified trick-1 decision set (`count` 32, `seed` 1: the
//!    yardstick's own).
//! 2. **Fill window** (`fill_s`, 240 s): `T` (this runner's usable threads) worlds at once, one
//!    solver thread each, thread `k` on yardstick worlds of position `k mod 8`, every table
//!    `fill_tt_mb` (lowered to fit the memory plan of `mem.rs`, with a warning).
//! 3. **Production window** (`prod_s`, 120 s): the position-aware schedule with the memory budget
//!    of the real runs (`schedule` overrides its settings) on every decision × `prod_worlds`
//!    worlds (default: as many world solves as the golden set, so the plan is the golden run's).
//! 4. **Table-size sweep** (only if `sweep_mb` is given; after the main windows): one more fill
//!    window of `sweep_s` (90) seconds per listed table size (MB per thread, lowered to fit the
//!    memory with a warning; a size already run is skipped). The worlds and their order are the
//!    fill window's, so `out/t1bench/fill-sweep-<mb>.json` can be compared world by world: the
//!    results' `sweep` section gives, per size, the nodes and wall time of the worlds finished in
//!    both it and the fill window, and any value that differs fails the chunk (table size changes
//!    speed only). Only short worlds (mostly positions 5-8) finish in the windows.
//! 5. **Results**: `out/t1bench/results.json` (`search::t1bench::analyse`, plus the table, setup
//!    time, warnings and any failure), `out/t1bench/summary.txt` (plain English), and each
//!    window's raw record `out/t1bench/fill.json`, `prod.json`. A world finished in both windows
//!    must have byte-identical values, or the chunk fails after writing them.
//!
//! **Self-protection.** During either window, from `ramp_s` (20 s) on: CPU over the last 30 s
//! below `min_cpu_pct` (80) of the usable cores, or fewer than `min_busy_pct` (95%) of the `T`
//! solver threads searching for more than `busy_streak_s` (10 s) in a row, stops the bench at once (the window's own guard;
//! the runner also measures the process tree's CPU over each 30-s heartbeat interval and asks the
//! window to stop through its stop file). The results so far are written and the chunk ends
//! FAILED with the reason, e.g. `low CPU: 41% in fill at 50 s`, so the normal end of the run
//! uploads them and the machine destroys itself.
//!
//! **Heartbeats** show the window, seconds into it, threads searching of `T`, nodes so far and
//! the node rate, with `partial_cpu_ok: false`; the runner prints one compact line per heartbeat
//! (`hb 17 fill 95s cpu 98.7% busy 245/245 nodes 3.1e11 rate 3.3e9/s`).

use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use search::t1bench::{
    AnalyseParams, DEFAULT_POSITION_NODES, Mode, POSITIONS, WindowRun, analyse, analyse_sweep,
    summary_text,
};
use serde_json::{Value, json};

use crate::mem::{self, Requested, Tables};
use crate::spike::build_table_from;
use crate::{Ctx, HEARTBEAT_EVERY, clk_tck, hardware, sibling, stop, tree_ticks};

/// The settings a `t1bench` chunk takes (anything else is refused: the configuration sweep's
/// settings were removed on 2026-10-14).
const KEYS: [&str; 18] = [
    "dph",
    "fill_s",
    "prod_s",
    "fill_tt_mb",
    "sweep_mb",
    "sweep_s",
    "min_cpu_pct",
    "ramp_s",
    "busy_streak_s",
    "min_busy_pct",
    "golden",
    "position_nodes",
    "schedule",
    "prod_worlds",
    "count",
    "seed",
    "start_hands",
    "table",
];

/// Setup longer than this is reported as a warning.
const SETUP_WARN_S: f64 = 90.0;

/// How long a window may take to stop after the runner asked it to, before the runner stops the
/// whole run (and with it the window's process).
const STOP_GRACE: Duration = Duration::from_secs(60);

fn f(p: &Value, k: &str, d: f64) -> Result<f64, String> {
    match p.get(k) {
        None | Some(Value::Null) => Ok(d),
        Some(v) => v
            .as_f64()
            .filter(|x| x.is_finite() && *x >= 0.0)
            .ok_or_else(|| format!("t1bench: {k} must be a number (at least 0)")),
    }
}

/// The CPU use (percent of `cores`) of `ticks` over `secs`, when it is below `min_pct`.
fn low_cpu(ticks: u64, tck: f64, secs: f64, cores: f64, min_pct: f64) -> Option<f64> {
    if secs <= 0.0 || cores <= 0.0 || tck <= 0.0 || min_pct <= 0.0 {
        return None;
    }
    let pct = 100.0 * ticks as f64 / tck / (secs * cores);
    (pct < min_pct).then_some(pct)
}

/// The runner's own check during a window: the process tree's CPU over each heartbeat interval
/// from the ramp's end on; when low, the window is asked to stop (its stop file), and if it has
/// not stopped `STOP_GRACE` later, the run is stopped.
struct CpuWatch {
    min_pct: f64,
    ramp_s: f64,
    cores: f64,
    tck: f64,
    base: Option<(u64, Instant)>,
    asked: Option<Instant>,
}

impl CpuWatch {
    fn poll(&mut self, child: Option<u32>, mode: Mode, started: bool, t_s: f64, stop_file: &Path) {
        if let Some(a) = self.asked {
            if a.elapsed() > STOP_GRACE {
                stop::request("the t1bench window did not stop when asked (low CPU)");
            }
            return;
        }
        if !started || t_s < self.ramp_s {
            return;
        }
        let Some(pid) = child else { return };
        let due = self
            .base
            .is_none_or(|(_, at)| at.elapsed() >= HEARTBEAT_EVERY);
        if !due {
            return;
        }
        let Some(ticks) = tree_ticks(pid) else { return };
        let now = Instant::now();
        if let Some((t0, a0)) = self.base {
            let secs = now.duration_since(a0).as_secs_f64();
            if let Some(pct) = low_cpu(
                ticks.saturating_sub(t0),
                self.tck,
                secs,
                self.cores,
                self.min_pct,
            ) {
                let why = format!("low CPU: {pct:.0}% in {} at {t_s:.0} s", mode.name());
                eprintln!("runner: t1bench: {why} (process tree over {secs:.0} s); stopping");
                // Whole or not at all: written aside, then renamed into place.
                let tmp = stop_file.with_extension("tmp");
                if fs::write(&tmp, &why).is_ok() && fs::rename(&tmp, stop_file).is_ok() {
                    self.asked = Some(now);
                } else {
                    stop::request(&why);
                }
            }
        }
        self.base = Some((ticks, now));
    }
}

/// One window's settings.
struct Win<'a> {
    mode: Mode,
    /// Added to the record's file name (`fill-sweep-64.json`), so extra windows keep their own.
    tag: String,
    window_s: f64,
    dec: &'a Path,
    table: &'a Path,
    out: &'a Path,
    /// Mode-specific `search bench-t1` arguments.
    extra: Vec<String>,
    min_cpu_pct: f64,
    ramp_s: f64,
    busy_streak_s: f64,
    min_busy_pct: f64,
}

/// The heartbeat's progress during a window, from the window's progress file.
fn window_progress(mode: Mode, threads: usize, pr: &Value, warnings: &[String]) -> Value {
    json!({
        "kind": "t1bench", "window": mode.name(),
        "started": pr["started"].as_bool().unwrap_or(false),
        "t_s": pr["t_s"], "window_s": pr["window_s"], "threads": threads,
        "busy": pr["busy"], "held": pr["held"], "nodes": pr["nodes"], "rate": pr["rate"],
        "finished_worlds": pr["finished_worlds"], "stopped": pr["stopped"],
        "partial_cpu_ok": false, "warnings": warnings,
    })
}

/// Runs one window (`search bench-t1`) and reads its record.
fn window(ctx: &mut Ctx, w: &Win, warnings: &[String]) -> Result<WindowRun, String> {
    let name = w.mode.name();
    let json_path = w.out.join(format!("{name}{}.json", w.tag));
    let progress = ctx.root.join("t1bench-progress.json");
    let stop_file = ctx.root.join("t1bench-stop");
    for p in [&json_path, &progress, &stop_file] {
        let _ = fs::remove_file(p);
    }
    let t = ctx.threads;
    let cores = ctx.cpu.cores;
    let mut c = Command::new(sibling("search"));
    c.args(["bench-t1", "--mode", name])
        .arg("--decisions")
        .arg(w.dec)
        .arg("--equity")
        .arg(w.table)
        .args(["--threads", &t.to_string()])
        .args(["--window-s", &format!("{}", w.window_s)])
        .args(["--usable-cores", &format!("{cores}")])
        .args(["--min-cpu-pct", &format!("{}", w.min_cpu_pct)])
        .args(["--ramp-s", &format!("{}", w.ramp_s)])
        .args(["--busy-streak-s", &format!("{}", w.busy_streak_s)])
        .args(["--min-busy-pct", &format!("{}", w.min_busy_pct)])
        .args([
            "--cpu-interval-s",
            &format!("{}", HEARTBEAT_EVERY.as_secs()),
        ])
        .arg("--json")
        .arg(&json_path)
        .arg("--progress")
        .arg(&progress)
        .arg("--stop-file")
        .arg(&stop_file)
        .args(&w.extra);
    println!(
        "runner: t1bench {name} window: {} s on {t} threads ({cores:.2} usable cores)",
        w.window_s
    );
    ctx.set_phase(
        "chunk",
        window_progress(w.mode, t, &json!({ "window_s": w.window_s }), warnings),
    );
    let mut watch = CpuWatch {
        min_pct: w.min_cpu_pct,
        ramp_s: w.ramp_s,
        cores,
        tck: clk_tck(),
        base: None,
        asked: None,
    };
    let res = ctx.run_child_with(c, "t1bench/search.log", &mut |ctx| {
        let pr: Value = fs::read_to_string(&progress)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(Value::Null);
        if pr.is_object() {
            ctx.progress = window_progress(w.mode, t, &pr, warnings);
        }
        watch.poll(
            ctx.child,
            w.mode,
            pr["started"].as_bool().unwrap_or(false),
            pr["t_s"].as_f64().unwrap_or(0.0),
            &stop_file,
        );
    });
    // The record is kept whenever the window wrote it, even if the process then failed.
    let run: Option<WindowRun> = fs::read_to_string(&json_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());
    match (res, run) {
        (Ok(()), Some(r)) => {
            println!(
                "runner: t1bench {name}: {:.3e} nodes in {:.0} s{}",
                r.nodes() as f64,
                r.wall_s,
                r.stopped
                    .as_deref()
                    .map_or(String::new(), |s| format!("; stopped early: {s}"))
            );
            Ok(r)
        }
        (Ok(()), None) => Err(format!(
            "t1bench: the {name} window wrote no record ({})",
            json_path.display()
        )),
        (Err(e), _) => Err(e),
    }
}

/// Run one `t1bench` chunk.
pub(crate) fn chunk(ctx: &mut Ctx, p: &Value) -> Result<Value, String> {
    if let Some(o) = p.as_object()
        && let Some(k) = o.keys().find(|k| !KEYS.contains(&k.as_str()))
    {
        return Err(format!(
            "t1bench: unknown setting {k:?} (settings: {}; the configuration sweep was removed)",
            KEYS.join(", ")
        ));
    }
    let setup0 = Instant::now();
    ctx.set_phase(
        "chunk",
        json!({ "kind": "t1bench", "step": "setup: table v0", "partial_cpu_ok": true }),
    );
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
        json!({ "kind": "t1bench", "step": "setup: decisions", "partial_cpu_ok": true }),
    );
    ctx.run_child(c, "t1bench/search.log")?;
    let setup_s = setup0.elapsed().as_secs_f64();
    let mut warnings: Vec<String> = Vec::new();
    if setup_s > SETUP_WARN_S {
        warnings.push(format!(
            "setup (table v0 and the decision set) took {setup_s:.0} s, over {SETUP_WARN_S:.0} s"
        ));
    }
    // Settings.
    let t = ctx.threads;
    let fill_s = f(p, "fill_s", 240.0)?;
    let prod_s = f(p, "prod_s", 120.0)?;
    let sweep_s = f(p, "sweep_s", 90.0)?;
    let sweep_mb: Vec<u64> = match p.get("sweep_mb") {
        None | Some(Value::Null) => Vec::new(),
        Some(v) => v
            .as_array()
            .and_then(|a| a.iter().map(|x| x.as_u64().filter(|n| *n > 0)).collect())
            .ok_or(
                "t1bench: sweep_mb must be a list of table sizes in MB (whole numbers, at least 1)",
            )?,
    };
    let min_cpu_pct = f(p, "min_cpu_pct", 80.0)?;
    let ramp_s = f(p, "ramp_s", 20.0)?;
    let busy_streak_s = f(p, "busy_streak_s", 10.0)?;
    let min_busy_pct = f(p, "min_busy_pct", 95.0)?;
    let golden = match p.get("golden") {
        None | Some(Value::Null) => (128, 200),
        Some(g) => match (
            g.get(0).and_then(Value::as_u64),
            g.get(1).and_then(Value::as_u64),
        ) {
            (Some(d), Some(w)) if d > 0 && w > 0 => (d, w),
            _ => return Err("t1bench: golden must be [decisions, worlds]".into()),
        },
    };
    let (position_nodes, pn_source) = match p.get("position_nodes") {
        None | Some(Value::Null) => (DEFAULT_POSITION_NODES, "default"),
        Some(v) => (
            serde_json::from_value::<[f64; POSITIONS]>(v.clone())
                .ok()
                .filter(|x| x.iter().all(|n| n.is_finite() && *n >= 0.0))
                .ok_or("t1bench: position_nodes must be 8 numbers (positions 1-8)")?,
            "given",
        ),
    };
    let prod_worlds = match p.get("prod_worlds").and_then(Value::as_u64) {
        Some(w) if w > 0 => w,
        _ => (golden.0 * golden.1).div_ceil(count.max(1)),
    };
    // Fill tables: `fill_tt_mb` each, lowered to fit the memory with a warning.
    let requested = match Requested::from_param(p, "fill_tt_mb")? {
        Requested::Auto => Requested::Mb(512),
        r => r,
    };
    let tt = mem::plan_here(
        requested,
        Tables::Own { count: t as u64 },
        &table,
        "t1bench fill",
    );
    if let Some(w) = &tt.warning {
        warnings.push(format!("fill window: {w}"));
    }
    // Production tables: the budget the real runs use.
    let (budget_rec, budget) = mem::budget_here(&table, t as u64, "t1bench prod");
    let mut prod_extra = vec![
        "--tt-budget-mb".to_string(),
        (budget >> 20).to_string(),
        "--worlds-per-decision".to_string(),
        prod_worlds.to_string(),
    ];
    if let Some(s) = p.get("schedule").filter(|v| !v.is_null()) {
        if !s.is_object() {
            return Err("t1bench: schedule must be an object of plan settings".into());
        }
        prod_extra.extend(["--sched-json".to_string(), s.to_string()]);
    }
    let win = |mode: Mode, tag: String, window_s: f64, extra: Vec<String>| Win {
        mode,
        tag,
        window_s,
        dec: &dec,
        table: &table,
        out: &out,
        extra,
        min_cpu_pct,
        ramp_s,
        busy_streak_s,
        min_busy_pct,
    };
    // The windows: production only if the fill window ran to its end.
    let fill = window(
        ctx,
        &win(
            Mode::Fill,
            String::new(),
            fill_s,
            vec!["--tt-mb".into(), tt.tt_mb.to_string()],
        ),
        &warnings,
    )?;
    let mut failure = fill.stopped.clone();
    let prod = if failure.is_none() {
        // A failed production window still leaves the fill window's results written.
        match window(
            ctx,
            &win(Mode::Prod, String::new(), prod_s, prod_extra),
            &warnings,
        ) {
            Ok(r) => {
                failure = r.stopped.clone();
                Some(r)
            }
            Err(e) => {
                failure = Some(e);
                None
            }
        }
    } else {
        None
    };
    // The table-size sweep: after the main windows, so it cannot disturb them. Each size runs
    // the same worlds as the fill window (see `search::t1bench::analyse_sweep`); a size the
    // memory plan lowers to one already run is skipped, and a failed sweep window ends the sweep
    // with a warning (the main results are already complete).
    let mut sweep_runs: Vec<WindowRun> = Vec::new();
    if failure.is_none() {
        let mut done = vec![tt.tt_mb];
        for &mb in &sweep_mb {
            let used = mem::plan_here(
                Requested::Mb(mb),
                Tables::Own { count: t as u64 },
                &table,
                "t1bench sweep",
            );
            if let Some(w) = &used.warning {
                warnings.push(format!("sweep {mb} MB: {w}"));
            }
            if done.contains(&used.tt_mb) {
                warnings.push(format!(
                    "sweep {mb} MB: skipped (table size {} MB was already run)",
                    used.tt_mb
                ));
                continue;
            }
            done.push(used.tt_mb);
            let w = win(
                Mode::Fill,
                format!("-sweep-{}", used.tt_mb),
                sweep_s,
                vec!["--tt-mb".into(), used.tt_mb.to_string()],
            );
            match window(ctx, &w, &warnings) {
                Ok(r) => {
                    let stopped = r.stopped.clone();
                    sweep_runs.push(r);
                    if let Some(why) = stopped {
                        warnings.push(format!("sweep {mb} MB stopped early: {why}; no more sizes"));
                        break;
                    }
                }
                Err(e) => {
                    warnings.push(format!("sweep {mb} MB failed: {e}; no more sizes"));
                    break;
                }
            }
        }
    }
    // Analysis.
    let hw = hardware();
    let dph = p
        .get("dph")
        .and_then(Value::as_f64)
        .unwrap_or(ctx.dph)
        .max(0.0);
    let params = AnalyseParams {
        threads: t,
        usable_cores: ctx.cpu.cores,
        dph,
        golden_decisions: golden.0,
        golden_worlds: golden.1,
        position_nodes,
        position_nodes_source: pn_source.to_string(),
        cpu_model: hw["cpu_model"].as_str().unwrap_or("").to_string(),
        mem_gb: mem::Sys::host().limit().mb() as f64 / 1024.0,
        busy_skip_s: 5.0,
    };
    let mut a = analyse(Some(&fill), prod.as_ref(), &params);
    if !sweep_runs.is_empty() {
        let sw = analyse_sweep(&fill, &sweep_runs);
        let bad: Vec<Value> = sw["mismatches"].as_array().cloned().unwrap_or_default();
        if !bad.is_empty() {
            // Table size must change speed only, never a value.
            a["ok"] = json!(false);
            if let Some(m) = a["identity"]["mismatches"].as_array_mut() {
                m.extend(bad);
            }
        }
        a["sweep"] = sw;
    }
    a["failure"] = json!(failure);
    a["setup_s"] = json!((setup_s * 10.0).round() / 10.0);
    a["table"] = table_rec.clone();
    a["fill_tables"] = tt.record.clone();
    a["prod_tables"] = budget_rec.record.clone();
    a["warnings"] = json!(warnings);
    a["settings"] = json!({
        "fill_s": fill_s, "prod_s": prod_s, "fill_tt_mb_requested": p.get("fill_tt_mb"),
        "sweep_mb": sweep_mb, "sweep_s": sweep_s,
        "min_cpu_pct": min_cpu_pct, "ramp_s": ramp_s, "busy_streak_s": busy_streak_s,
        "min_busy_pct": min_busy_pct, "prod_worlds": prod_worlds, "count": count, "seed": seed, "schedule": p.get("schedule"),
    });
    let text = summary_text(&a)
        + &format!(
            "\nSetup took {setup_s:.0} s (table v0: {}; then the decision set).\n",
            table_rec["source"].as_str().unwrap_or("?")
        );
    fs::write(
        out.join("results.json"),
        serde_json::to_string_pretty(&a).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| e.to_string())?;
    fs::write(out.join("summary.txt"), &text).map_err(|e| e.to_string())?;
    println!("{text}");
    if let Some(why) = failure {
        return Err(format!(
            "t1bench: {why} (results so far in out/t1bench/results.json)"
        ));
    }
    if a["ok"].as_bool() != Some(true) {
        return Err(format!(
            "t1bench: the windows gave different values for {} world(s): this is a bug; \
             see out/t1bench/results.json (identity.mismatches)",
            a["identity"]["mismatches"].as_array().map_or(0, Vec::len)
        ));
    }
    Ok(json!({
        "kind": "t1bench", "threads": t, "dph": dph,
        "score": a["score"], "golden": a["golden"],
        "fill": { "nodes_per_s": a["fill"]["nodes_per_s"], "cpu_pct": a["fill"]["cpu_pct"],
                  "busy_pct": a["fill"]["busy_pct"], "tt_mb": tt.tt_mb },
        "prod": { "nodes_per_s": a["prod"]["nodes_per_s"], "cpu_pct": a["prod"]["cpu_pct"],
                  "busy_pct": a["prod"]["busy_pct"] },
        "identity": a["identity"]["worlds_compared"], "setup_s": a["setup_s"],
        "table": table_rec, "warnings": warnings,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_cpu_rule() {
        // 4 cores for 30 s at 100 ticks a second: 12,000 ticks is 100%.
        assert_eq!(low_cpu(12_000, 100.0, 30.0, 4.0, 80.0), None);
        assert_eq!(low_cpu(9_600, 100.0, 30.0, 4.0, 80.0), None, "exactly 80%");
        assert_eq!(low_cpu(4_920, 100.0, 30.0, 4.0, 80.0), Some(41.0));
        assert_eq!(low_cpu(0, 100.0, 30.0, 4.0, 0.0), None, "check off");
        assert_eq!(low_cpu(0, 100.0, 0.0, 4.0, 80.0), None, "no interval yet");
    }

    #[test]
    fn progress_for_heartbeats() {
        let pr = json!({ "window": "fill", "started": true, "t_s": 95.0, "window_s": 240.0,
                         "threads": 245, "busy": 245, "nodes": 3.1e11, "rate": 3.3e9 });
        let v = window_progress(Mode::Fill, 245, &pr, &["w".to_string()]);
        assert_eq!(v["partial_cpu_ok"], json!(false));
        assert_eq!(v["window"], json!("fill"));
        assert_eq!(v["busy"], json!(245));
        assert_eq!(v["warnings"], json!(["w"]));
        assert_eq!(
            crate::hb_line(17, "chunk", &v, Some(98.7)),
            "hb 17 fill 95s cpu 98.7% busy 245/245 nodes 3.1e11 rate 3.3e9/s"
        );
    }
}
