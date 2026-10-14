//! The `search_measure` chunk kind (docs/M3-SPEC.md §13.1): the early-trick agreement curve,
//! the exact-PIMC yardstick and the method-vs-yardstick scorer, run by the sibling `search`
//! binary.
//!
//! ```json
//! { "search_measure": { "trick": 2, "count": 224, "seed": 1, "to": 320,
//!     "first": 0, "last": 223, "grid": "10,20,40,80,160,320",
//!     "worlds_at_once": "auto", "tt_mb": "auto", "start_hands": 20000,
//!     "methods": [ { "method": "rollout", "label": 2, "to": 64 } ],
//!     "table": { "hands": 1000000, "seed": 1 } } }
//! ```
//! Steps: equity table v0 (`table.source` `"auto"`: a manifest input, else the copy in R2 under
//! its recorded hash, else rebuilt from the model in the repository; always hash-checked; see
//! `spike.rs`); `search decisions` (stratified per stratum and per
//! position in the trick; `count` a multiple of 32 unless `"allow_uneven": true`);
//! `search worlds --method exact` for decisions `first..=last` up to `to` worlds (label 1, nested
//! sets; resumable); `search agree` (N against 2N on `grid`, per trick, position, position pair,
//! stratum, world-count band and hand-type band); `search curves` (money-loss curves on `grid`,
//! which may be fine, e.g. `"10:320:10"`; writes `curves-t<trick>.json`); then each listed
//! method's worlds and `search score` against the yardstick at the largest grid N available.
//!
//! Targeted extension (docs/M3-SPEC.md §13.1 item 10): `"ids"` (a list of decision ids, a string
//! `"3,17,42"`, or the path of a file of ids; in format 2, `"ids_input"` names a manifest input
//! holding the file) restricts the exact worlds to those decisions, and `"from"` (default 0) sets
//! the first world. With `from > 0` the step only values worlds `from..to` into
//! `exact-t<trick>-w<from>-<to>.jsonl` (same decision set and seed as the first run, so the rows
//! continue it exactly) and skips the analysis steps: run `search agree` / `search curves` with
//! `--worlds` on the first run's file and this one, which they join.
//!
//! Table size: `tt_mb` absent or `"auto"` sizes the solver's transposition tables from the
//! container's memory (see `mem.rs`): one table per world solved at once (`worlds_at_once` shared
//! tables with `solve_threads > 1`, else one per thread), each counted at its peak. An explicit
//! `tt_mb` whose tables could exhaust memory is lowered to the largest size that fits, with a
//! warning in the heartbeat (`progress.warnings`), in `result.json` (`warnings`, and the whole
//! calculation under `tt`) and on stderr. The size changes speed only, never a value.
//!
//! Scheduling: `search worlds` puts every (decision, world) task of the step in one work queue,
//! longest expected first, so no thread waits for another decision's slow world. With
//! `worlds_at_once` `"auto"` (the default when neither it nor `solve_threads` is a number) the
//! exact worlds are planned by position (docs/M3-SPEC.md §13.1 item 11, `search::plan`): each
//! world gets its own threads and table within the memory budget of `mem.rs` (`schedule` in the
//! params overrides the plan's settings); at tricks 2–4 that is one thread per world, as before.
//! Explicit numbers keep their meaning: with `solve_threads > 1` each exact world is solved by
//! `solve_fast_par` on that many threads, `worlds_at_once` (default 1) worlds at a time; with
//! `solve_threads` 1, one world per thread on every thread.
//!
//! Heartbeat progress during a worlds step: the step, its N, decisions and worlds done of total,
//! and the running parallel solve's subproblems done of total. `partial_cpu_ok` is set only for a
//! heartbeat interval that was at least [`TAIL_SHARE_OK`] in a genuine tail: fewer tasks left in
//! the queue than worker threads (the end of the step), or fewer subproblems left in the running
//! parallel solve than its threads, or (planned runs) every world started and threads free.

use std::fs;
use std::path::Path;
use std::process::Command;

use serde_json::{Value, json};

use crate::mem::{self, Requested, Tables};
use crate::spike::build_table_from;
use crate::{Ctx, sibling};

/// The share of a heartbeat interval spent in a tail from which that interval's CPU sample is
/// exempt from the launcher's low-CPU rule: the shortfall below its default 85% threshold.
const TAIL_SHARE_OK: f64 = 0.15;

fn n(p: &Value, k: &str, d: u64) -> u64 {
    p.get(k).and_then(Value::as_u64).unwrap_or(d)
}

/// `params.ids` as a `search --ids` argument: a list of numbers, or a string (a list or a path).
fn ids_arg(p: &Value) -> Result<Option<String>, String> {
    match p.get("ids") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| {
                x.as_u64()
                    .map(|v| v.to_string())
                    .ok_or_else(|| "params.ids: decision ids are whole numbers".to_string())
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|v| Some(v.join(","))),
        Some(_) => Err("params.ids: a list of ids, \"3,17,42\" or a file path".into()),
    }
}

/// The largest N of a grid (`10,20,40` or `start:end:step`) not above `to`.
fn grid_max(grid: &str, to: u64) -> u64 {
    let vals: Vec<u64> = if grid.contains(':') {
        let p: Vec<u64> = grid
            .split(':')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        match p[..] {
            [a, b, s] if s > 0 && a > 0 => (a..=b).step_by(s as usize).collect(),
            _ => Vec::new(),
        }
    } else {
        grid.split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect()
    };
    vals.into_iter().filter(|&x| x <= to).max().unwrap_or(to)
}

/// Run one `search worlds` step, with its progress in the heartbeat.
#[allow(clippy::too_many_arguments)]
fn run_worlds(
    ctx: &mut Ctx,
    mut c: Command,
    step: &str,
    n_worlds: u64,
    threads: u64,
    solve_threads: u64,
    tt: &mem::Plan,
    progress_file: &Path,
) -> Result<(), String> {
    let _ = fs::remove_file(progress_file);
    c.arg("--progress").arg(progress_file);
    let warnings: Vec<&String> = tt.warning.iter().collect();
    ctx.set_phase(
        "chunk",
        json!({ "kind": "search_measure", "step": step, "n": n_worlds,
                "tt_mb": tt.tt_mb, "warnings": warnings }),
    );
    let (mut seq, mut polls, mut tail_polls) = (ctx.seq, 0u64, 0u64);
    let step = step.to_string();
    ctx.run_child_with(c, "measure/worlds.log", &mut |ctx| {
        let p: Option<Value> = fs::read_to_string(progress_file)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok());
        let p = p.unwrap_or(Value::Null);
        let g = |k: &str| p.get(k).and_then(Value::as_u64).unwrap_or(0);
        if ctx.seq != seq {
            // A heartbeat went out: start a new interval.
            (seq, polls, tail_polls) = (ctx.seq, 0, 0);
        }
        polls += 1;
        tail_polls += u64::from(in_tail(&p, solve_threads));
        let share = tail_polls as f64 / polls as f64;
        ctx.progress = json!({
            "kind": "search_measure", "step": step, "n": n_worlds,
            "threads": threads, "solve_threads": solve_threads,
            "decisions_done": g("decisions_done"), "decisions_total": g("decisions_total"),
            "worlds_done": g("worlds_done"), "worlds_total": g("worlds_total"),
            "par_tasks_done": g("par_tasks_done"), "par_tasks_total": g("par_tasks_total"),
            "tail_share": (share * 100.0).round() / 100.0,
            "partial_cpu_ok": share >= TAIL_SHARE_OK,
            "tt_mb": tt.tt_mb, "warnings": warnings,
        });
    })
}

/// Is a `search worlds` progress record in a genuine tail? Either the end of the step (fewer
/// tasks left than worker threads, so some have nothing to take) or the last subproblems of the
/// running parallel solve (fewer left than its threads).
fn in_tail(p: &Value, solve_threads: u64) -> bool {
    // A position-aware run says itself when every world has started and threads are free.
    if p.get("planned").and_then(Value::as_bool) == Some(true) {
        return p.get("tail").and_then(Value::as_bool) == Some(true);
    }
    let g = |k: &str| p.get(k).and_then(Value::as_u64).unwrap_or(0);
    let left = g("worlds_total").saturating_sub(g("worlds_done"));
    let workers = g("threads").max(1);
    let queue_tail = workers > 1 && left > 0 && left < workers;
    let (pd, pt) = (g("par_tasks_done"), g("par_tasks_total"));
    let par_tail = solve_threads > 1 && pt > 0 && pt.saturating_sub(pd) < solve_threads;
    queue_tail || par_tail
}

/// How the exact worlds of a `search_measure` chunk are spread over the machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Spread {
    /// Position-aware: every world its own threads and table (`search worlds --schedule auto`).
    Auto,
    /// `workers` worlds at once, each on `solve_threads` threads.
    Fixed { workers: u64, solve_threads: u64 },
}

/// `worlds_at_once` and `solve_threads`: `"auto"` in either, or both absent, is the
/// position-aware schedule. Numbers keep the explicit settings exactly as before: with
/// `solve_threads` above 1, `worlds_at_once` (default 1) parallel solves at a time; with
/// `solve_threads` 1 (or absent), one world per thread on every thread.
pub(crate) fn spread(p: &Value, threads: u64) -> Result<Spread, String> {
    let get = |k: &str| match p.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s == "auto" => Ok(Some(None)),
        Some(v) => v
            .as_u64()
            .filter(|&n| n > 0)
            .map(|n| Some(Some(n)))
            .ok_or_else(|| format!("params.{k} must be a whole number (at least 1) or \"auto\"")),
    };
    let (w, s) = (get("worlds_at_once")?, get("solve_threads")?);
    if matches!(w, Some(None)) || matches!(s, Some(None)) || (w.is_none() && s.is_none()) {
        return Ok(Spread::Auto);
    }
    let s = s.flatten().unwrap_or(1);
    Ok(if s > 1 {
        Spread::Fixed {
            workers: w.flatten().unwrap_or(1),
            solve_threads: s,
        }
    } else {
        Spread::Fixed {
            workers: threads,
            solve_threads: 1,
        }
    })
}

/// Run one `search_measure` chunk.
pub(crate) fn chunk(ctx: &mut Ctx, p: &Value) -> Result<Value, String> {
    let (table, table_rec) = build_table_from(ctx, p.get("table").unwrap_or(&json!({})), "auto")?;
    let out = ctx.root.join("out").join("measure");
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let trick = n(p, "trick", 2);
    let count = n(p, "count", 224);
    let seed = n(p, "seed", 1);
    let to = n(p, "to", 320);
    let from = n(p, "from", 0);
    if from >= to {
        return Err(format!("search_measure: from {from} must be below to {to}"));
    }
    let ids = ids_arg(p)?;
    let sched_json = match p.get("schedule") {
        None | Some(Value::Null) => None,
        Some(v @ Value::Object(_)) => Some(v.to_string()),
        Some(_) => return Err("params.schedule must be an object of plan settings".into()),
    };
    let (threads, solve_threads, tt, budget_mb) = match spread(p, ctx.threads as u64)? {
        Spread::Fixed {
            workers,
            solve_threads,
        } => {
            // One table per worker of `search worlds`, kept between its worlds.
            let tables = if solve_threads > 1 {
                Tables::Shared {
                    count: workers,
                    solve_threads,
                }
            } else {
                Tables::Own { count: workers }
            };
            let tt = mem::plan_here(
                Requested::from_param(p, "tt_mb")?,
                tables,
                &table,
                "search_measure",
            );
            (workers, solve_threads, tt, None)
        }
        Spread::Auto => {
            // Every world its own threads and table: `search` shares this budget out.
            let (tt, budget) = mem::budget_here(&table, ctx.threads as u64, "search_measure");
            (ctx.threads as u64, 1, tt, Some(budget >> 20))
        }
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
    if p.get("allow_uneven")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        c.arg("--allow-uneven");
    }
    ctx.set_phase(
        "chunk",
        json!({ "kind": "search_measure", "step": "decisions" }),
    );
    ctx.run_child(c, "measure/search.log")?;
    let worlds = |label: u64, method: &str, from: u64, to: u64, file: &std::path::Path| {
        let mut c = Command::new(sibling("search"));
        c.args(["worlds", "--method", method, "--to", &to.to_string()])
            .args(["--from-world", &from.to_string()])
            .args(["--label", &label.to_string(), "--seed", &seed.to_string()])
            .args(["--first", &n(p, "first", 0).to_string()])
            .args(["--last", &n(p, "last", count.saturating_sub(1)).to_string()])
            .args(["--threads", &threads.to_string()])
            .args(["--solve-threads", &solve_threads.to_string()])
            .args(["--tt-mb", &tt.tt_mb.to_string()])
            .arg("--decisions")
            .arg(&dec)
            .arg("--equity")
            .arg(&table)
            .arg("--out")
            .arg(file);
        if let Some(ids) = &ids {
            c.args(["--ids", ids]);
        }
        if let Some(b) = budget_mb {
            c.args(["--schedule", "auto", "--tt-budget-mb", &b.to_string()]);
            if let Some(j) = &sched_json {
                c.args(["--sched-json", j]);
            }
        }
        c
    };
    let progress_file = ctx.root.join("worlds-progress.json");
    let yard = if from == 0 {
        out.join(format!("exact-t{trick}-n{to}.jsonl"))
    } else {
        out.join(format!("exact-t{trick}-w{from}-{to}.jsonl"))
    };
    run_worlds(
        ctx,
        worlds(1, "exact", from, to, &yard),
        "exact worlds",
        to,
        threads,
        solve_threads,
        &tt,
        &progress_file,
    )?;
    let base = json!({ "trick": trick, "count": count, "seed": seed, "from": from, "to": to,
                       "ids": ids, "grid": grid, "solve_threads": solve_threads,
                       "worlds_threads": threads, "table": table_rec,
                       "schedule": if budget_mb.is_some() { "auto" } else { "fixed" },
                       "tt_budget_mb": budget_mb,
                       "tt": tt.record, "warnings": tt.warning.iter().collect::<Vec<_>>() });
    if from > 0 {
        // An extension: the analysis needs the earlier worlds too, so it is run where both
        // files are (`search agree|curves --worlds <first run> --worlds <this>`).
        let mut r = base;
        r["methods"] = json!([]);
        r["extension"] = json!(true);
        return Ok(r);
    }
    let mut c = Command::new(sibling("search"));
    c.arg("agree")
        .arg("--worlds")
        .arg(&yard)
        .args(["--grid", &grid])
        .arg("--json")
        .arg(out.join(format!("agree-t{trick}.json")));
    ctx.set_phase(
        "chunk",
        json!({ "kind": "search_measure", "step": "agree" }),
    );
    ctx.run_child(c, "measure/agree.log")?;
    let mut c = Command::new(sibling("search"));
    c.arg("curves")
        .arg("--worlds")
        .arg(&yard)
        .args(["--grid", &grid])
        .args(["--threads", &ctx.threads.to_string()])
        .arg("--json")
        .arg(out.join(format!("curves-t{trick}.json")));
    ctx.set_phase(
        "chunk",
        json!({ "kind": "search_measure", "step": "curves" }),
    );
    ctx.run_child(c, "measure/curves.log")?;
    let n_ref = grid_max(&grid, to);
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
        run_worlds(
            ctx,
            worlds(label, method, 0, mto, &f),
            &format!("method {i} ({method}) worlds"),
            mto,
            threads,
            solve_threads,
            &tt,
            &progress_file,
        )?;
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
        ctx.set_phase(
            "chunk",
            json!({ "kind": "search_measure", "step": "score" }),
        );
        ctx.run_child(c, "measure/score.log")?;
    }
    let mut r = base;
    r["methods"] = json!(methods);
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_and_grids() {
        assert_eq!(ids_arg(&json!({})).expect("none"), None);
        assert_eq!(
            ids_arg(&json!({ "ids": [3, 17, 42] })).expect("list"),
            Some("3,17,42".into())
        );
        assert_eq!(
            ids_arg(&json!({ "ids": "ids.txt" })).expect("path"),
            Some("ids.txt".into())
        );
        assert!(ids_arg(&json!({ "ids": [1.5] })).is_err());
        assert!(ids_arg(&json!({ "ids": 7 })).is_err());
        assert_eq!(grid_max("10,20,40,80,160,320", 320), 320);
        assert_eq!(grid_max("10,20,40,80,160,320", 200), 160);
        assert_eq!(grid_max("10:320:10", 155), 150);
        assert_eq!(grid_max("bad", 64), 64);
    }

    #[test]
    fn tails() {
        let q = |done: u64, threads: u64| json!({ "worlds_done": done, "worlds_total": 64_000, "threads": threads });
        assert!(!in_tail(&q(0, 244), 1));
        assert!(!in_tail(&q(63_756, 244), 1), "244 left: every thread busy");
        assert!(in_tail(&q(63_757, 244), 1), "243 left on 244 threads");
        assert!(!in_tail(&q(64_000, 244), 1), "finished");
        assert!(!in_tail(&q(63_999, 1), 1), "one worker is never idle");
        let par = |done: u64, total: u64| {
            json!({ "worlds_done": 3, "worlds_total": 20, "threads": 1,
                    "par_tasks_done": done, "par_tasks_total": total })
        };
        assert!(!in_tail(&par(0, 0), 368), "split not made yet");
        assert!(!in_tail(&par(5_000, 5_888), 368));
        assert!(in_tail(&par(5_600, 5_888), 368));
        assert!(!in_tail(&par(5_600, 5_888), 1), "no parallel solve");
        assert!(!in_tail(&Value::Null, 368));
        // A position-aware run says so itself.
        let planned = |tail: bool| {
            json!({ "planned": true, "tail": tail, "threads": 368,
                                           "worlds_done": 10, "worlds_total": 12 })
        };
        assert!(in_tail(&planned(true), 1));
        assert!(
            !in_tail(&planned(false), 1),
            "2 worlds left but every thread busy"
        );
    }

    /// `"auto"` or nothing is the position-aware plan; numbers keep their old meaning.
    #[test]
    fn spreads() {
        let s = |v: Value| spread(&v, 64).expect("valid");
        assert_eq!(s(json!({})), Spread::Auto);
        assert_eq!(s(json!({ "worlds_at_once": "auto" })), Spread::Auto);
        assert_eq!(
            s(json!({ "worlds_at_once": "auto", "solve_threads": 8 })),
            Spread::Auto
        );
        assert_eq!(
            s(json!({ "solve_threads": 16, "worlds_at_once": 4 })),
            Spread::Fixed {
                workers: 4,
                solve_threads: 16
            }
        );
        assert_eq!(
            s(json!({ "solve_threads": 92 })),
            Spread::Fixed {
                workers: 1,
                solve_threads: 92
            }
        );
        assert_eq!(
            s(json!({ "solve_threads": 1 })),
            Spread::Fixed {
                workers: 64,
                solve_threads: 1
            }
        );
        assert_eq!(
            s(json!({ "worlds_at_once": 4 })),
            Spread::Fixed {
                workers: 64,
                solve_threads: 1
            },
            "as before: worlds_at_once only matters with solve_threads above 1"
        );
        assert!(spread(&json!({ "worlds_at_once": 0 }), 64).is_err());
        assert!(spread(&json!({ "solve_threads": "many" }), 64).is_err());
    }
}
