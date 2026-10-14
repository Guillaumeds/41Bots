//! The `search_measure` chunk kind (docs/M3-SPEC.md §13.1): the early-trick agreement curve,
//! the exact-PIMC yardstick and the method-vs-yardstick scorer, run by the sibling `search`
//! binary.
//!
//! ```json
//! { "search_measure": { "trick": 2, "count": 200, "seed": 1, "to": 320,
//!     "first": 0, "last": 199, "grid": "10,20,40,80,160,320",
//!     "solve_threads": 1, "worlds_at_once": 1, "tt_mb": 256, "start_hands": 20000,
//!     "methods": [ { "method": "rollout", "label": 2, "to": 64 } ],
//!     "table": { "hands": 1000000, "seed": 1 } } }
//! ```
//! Steps: build and verify equity table v0; `search decisions` (stratified per stratum and per
//! position in the trick); `search worlds --method exact` for decisions `first..=last` up to `to`
//! worlds (label 1, nested sets; resumable); `search agree` (per trick, position, position pair,
//! stratum, world-count band and hand-type band); then each listed method's worlds and
//! `search score` against the yardstick at the largest grid N available.
//!
//! Scheduling: `search worlds` puts every (decision, world) task of the step in one work queue,
//! longest expected first, on all threads, so no thread waits for another decision's slow world.
//! With `solve_threads > 1` (trick 1) each exact world is solved by `solve_fast_par` on that many
//! threads, `worlds_at_once` (default 1) worlds at a time.
//!
//! Heartbeat progress during a worlds step: the step, its N, decisions and worlds done of total,
//! and the running parallel solve's subproblems done of total. `partial_cpu_ok` is set only for a
//! heartbeat interval that was at least [`TAIL_SHARE_OK`] in a genuine tail: fewer tasks left in
//! the queue than worker threads (the end of the step), or fewer subproblems left in the running
//! parallel solve than its threads.

use std::fs;
use std::path::Path;
use std::process::Command;

use serde_json::{Value, json};

use crate::spike::build_table;
use crate::{Ctx, sibling};

/// The share of a heartbeat interval spent in a tail from which that interval's CPU sample is
/// exempt from the launcher's low-CPU rule: the shortfall below its default 85% threshold.
const TAIL_SHARE_OK: f64 = 0.15;

fn n(p: &Value, k: &str, d: u64) -> u64 {
    p.get(k).and_then(Value::as_u64).unwrap_or(d)
}

/// Run one `search worlds` step, with its progress in the heartbeat.
fn run_worlds(
    ctx: &mut Ctx,
    mut c: Command,
    step: &str,
    n_worlds: u64,
    threads: u64,
    solve_threads: u64,
    progress_file: &Path,
) -> Result<(), String> {
    let _ = fs::remove_file(progress_file);
    c.arg("--progress").arg(progress_file);
    ctx.set_phase(
        "chunk",
        json!({ "kind": "search_measure", "step": step, "n": n_worlds }),
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
        });
    })
}

/// Is a `search worlds` progress record in a genuine tail? Either the end of the step (fewer
/// tasks left than worker threads, so some have nothing to take) or the last subproblems of the
/// running parallel solve (fewer left than its threads).
fn in_tail(p: &Value, solve_threads: u64) -> bool {
    let g = |k: &str| p.get(k).and_then(Value::as_u64).unwrap_or(0);
    let left = g("worlds_total").saturating_sub(g("worlds_done"));
    let workers = g("threads").max(1);
    let queue_tail = workers > 1 && left > 0 && left < workers;
    let (pd, pt) = (g("par_tasks_done"), g("par_tasks_total"));
    let par_tail = solve_threads > 1 && pt > 0 && pt.saturating_sub(pd) < solve_threads;
    queue_tail || par_tail
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
        n(p, "worlds_at_once", 1).max(1)
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
    let progress_file = ctx.root.join("worlds-progress.json");
    let yard = out.join(format!("exact-t{trick}-n{to}.jsonl"));
    run_worlds(
        ctx,
        worlds(1, "exact", to, &yard),
        "exact worlds",
        to,
        threads,
        solve_threads,
        &progress_file,
    )?;
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
        run_worlds(
            ctx,
            worlds(label, method, mto, &f),
            &format!("method {i} ({method}) worlds"),
            mto,
            threads,
            solve_threads,
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
    Ok(
        json!({ "trick": trick, "count": count, "seed": seed, "to": to, "grid": grid,
               "solve_threads": solve_threads, "worlds_threads": threads, "table": table_rec,
               "methods": methods }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }
}
