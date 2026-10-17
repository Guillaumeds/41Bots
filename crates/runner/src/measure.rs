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
//! worlds restored from an earlier run, the running parallel solve's subproblems done of total,
//! and `eta` = `{done_frac, remaining_s, eta_utc, projected_usd, basis}`: the share of the step's
//! work done, weighted by each world's expected cost (the plan's model, refined by this run's
//! measured times per trick and position; `search::eta`), when the step should end, what the run
//! will then have cost at this machine's price (`DPH`), and `basis` `"measured"` once at least
//! 10% is done and every position with worlds left has at least 2 timed, else `"rough: …"` saying
//! what is not measured yet.
//!
//! Resume: `search worlds` checkpoints every world as it finishes (`<worlds file>.ckpt`), so a
//! killed chunk continues world by world: in format 1 from an earlier run's uploaded files
//! (`resume_from`, see `main.rs`), in format 2 from the chunk's live files (`v2.rs`). The chunk
//! writes `out/measure/fingerprint.json` (method, label, seed, trick, count, start_hands, ids,
//! first/last, from, to, the equity table's SHA-256) before any worlds step; a resumed chunk that
//! finds one with any field different stops with a plain message instead of building on files
//! made under other settings (`search worlds` checks every record and checkpoint line as well).
//! `result.json` records what was taken over (`resumed`: decisions, worlds, worlds valued here). `partial_cpu_ok` is set only for a
//! heartbeat interval that was at least [`TAIL_SHARE_OK`] in a genuine tail: fewer tasks left in
//! the queue than worker threads (the end of the step), or fewer subproblems left in the running
//! parallel solve than its threads, or (planned runs) every world started and threads free.
//!
//! **A silver reference** (docs/m3-hybrid/silver.md; [`Silver`]): where no exact golden set is
//! affordable (trick 1), the chunk values its decisions' worlds with a deeper hybrid instead of
//! exact search. Settings (format 2: each network is a manifest input, named):
//! ```json
//! "reference": { "cut": 3, "net_input": "o3" },
//! "check": { "cut": 2, "net_input": "o2", "verify_pct": 1, "tt_mb": 64, "standard": "0.01",
//!            "boot": 1000, "race_groups": true },
//! "worlds_input": "silver-t1", "share": true
//! ```
//! - `reference`: the worlds step is `search worlds --method hybrid --cut 3 --net <o3>` into
//!   `silver-t<trick>-c<cut>-n<to>.jsonl` (the same decisions, worlds, checkpoint and resume, world
//!   by world, as the exact step; the records say `hybrid` with the network's SHA-256, never
//!   exact). The fingerprint names the method `hybrid` and the frontier (cut, the network's
//!   SHA-256), so a resume never builds on exact files or another network's. `methods` is refused
//!   (`search score` scores against exact yardsticks); `agree` runs on the silver file and
//!   `curves` with `--method hybrid` (how many trick-1 worlds a decision needs, by this reference).
//! - `share` (default true with `reference`, else false): once the worlds are in, the decisions
//!   file and the worlds file go to `inputs/sha256/<hex>` (content-addressed; a failure is a
//!   warning, both files being in the bundle), and `result.json`'s `shared.reuse_params` gives
//!   the settings that reuse them (`worlds_input`, with the same `reference`).
//! - `worlds_input`: a stored worlds file (a manifest input; e.g. a shared silver set) is copied
//!   in as the worlds file before the step, which then values only what it lacks (nothing, for a
//!   whole set: a later network's check against the stored silver set costs no H_3 solve). Its
//!   records must be the step's own kind (`search worlds` refuses others).
//! - `check` (one object, or a list with distinct cuts: the exact anchor set checks H_3 and H_2
//!   against the same exact worlds): after the worlds (and the analysis), `search hybrid-check
//!   --trick <trick> --cut 2
//!   --net <o2> --worlds <the worlds file>`, with `--reference silver --reference-net <o3>` when
//!   the chunk's reference is silver, `--verify-pct` of the stored worlds recomputed first and
//!   required byte-identical, `--race-groups` (default true: the score and race groups), into
//!   `out/measure/hc-t<trick>-o<cut>-<silver|exact>.json` and the printed tables in `.txt`
//!   (`verify_solve_threads`: threads inside each recomputed reference solve, default 1);
//!   `result.json`'s `checks` give each verdict and the overall and band rows. A check's cut must
//!   be before the reference's. Not with an extension (`from` above 0). Without `reference` the
//!   worlds are exact and so is each check's reference.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::bundle::sha256_file;
use crate::mem::{self, Requested, Tables};
use crate::remote::{Remote, input_key, put_input};
use crate::spike::build_table_from;
use crate::{Ctx, sibling, unix_now, utc_string};

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

/// The heartbeat's `progress.eta` from `search worlds`' own estimate (`search::eta`): the share
/// of the step's work done (weighted by each world's expected cost, refined by measured times),
/// when the step should end at this rate, what the run will then have cost at this machine's
/// price, and whether that rests on measured times (`basis`).
fn eta_block(ctx: &Ctx, p: &Value) -> Value {
    let Some(e) = p.get("eta").filter(|e| e.is_object()) else {
        return Value::Null;
    };
    let remaining = e["remaining_s"]
        .as_f64()
        .filter(|r| r.is_finite() && *r >= 0.0);
    json!({
        "done_frac": e["done_frac"].as_f64().map(|x| (x * 10_000.0).round() / 10_000.0),
        "remaining_s": remaining.map(f64::round),
        "eta_utc": remaining.map(|r| utc_string(unix_now() + r.round() as u64)),
        "projected_usd": remaining.filter(|_| ctx.dph > 0.0)
            .map(|r| ((ctx.elapsed() + r) / 3600.0 * ctx.dph * 100.0).round() / 100.0),
        "basis": e["basis"],
        "of": "the worlds step",
    })
}

/// Run one `search worlds` step, with its progress in the heartbeat. Returns the step's last
/// progress record (counts done, and how many were restored from an earlier run).
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
    extra: &Value,
) -> Result<Value, String> {
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
    let read = || -> Value {
        fs::read_to_string(progress_file)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(Value::Null)
    };
    // The heartbeat's progress from the child's progress record `p` and this interval's tail
    // share.
    let progress = |ctx: &Ctx, p: &Value, share: f64| -> Value {
        let g = |k: &str| p.get(k).and_then(Value::as_u64).unwrap_or(0);
        json!({
            "kind": "search_measure", "step": step, "n": n_worlds,
            "threads": threads, "solve_threads": solve_threads,
            "decisions_done": g("decisions_done"), "decisions_total": g("decisions_total"),
            "worlds_done": g("worlds_done"), "worlds_total": g("worlds_total"),
            "worlds_resumed": g("worlds_resumed"), "eta": eta_block(ctx, p),
            // This machine's own work, for its cost per node (`machine.rs`).
            "worlds_valued": g("worlds_valued"), "nodes_valued": g("nodes_valued"),
            // A silver run's network evaluations (its other unit of work).
            "evals_valued": g("evals_valued"),
            "schedule": extra["schedule"], "trick": extra["trick"],
            "tt_budget_mb": extra["tt_budget_mb"],
            "par_tasks_done": g("par_tasks_done"), "par_tasks_total": g("par_tasks_total"),
            "tail_share": (share * 100.0).round() / 100.0,
            "partial_cpu_ok": share >= TAIL_SHARE_OK,
            "tt_mb": tt.tt_mb, "warnings": warnings,
        })
    };
    ctx.run_child_with(c, "measure/worlds.log", &mut |ctx| {
        let p = read();
        if ctx.seq != seq {
            // A heartbeat went out: start a new interval.
            (seq, polls, tail_polls) = (ctx.seq, 0, 0);
        }
        polls += 1;
        tail_polls += u64::from(in_tail(&p, solve_threads));
        ctx.progress = progress(ctx, &p, tail_polls as f64 / polls as f64);
    })?;
    // The step's final counts go out in a heartbeat of their own, so a step shorter than a
    // heartbeat interval still reaches the machine's node tally (`machine.rs`).
    let last = read();
    ctx.progress = progress(ctx, &last, 1.0);
    ctx.heartbeat(true);
    Ok(last)
}

/// What a worlds step took over from an earlier run (its last progress record `p`).
fn resumed(p: &Value) -> Value {
    json!({ "decisions": p["decisions_resumed"], "worlds": p["worlds_resumed"],
            "worlds_valued": p["worlds_valued"], "worlds_total": p["worlds_total"] })
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

/// A network a `search_measure` chunk names: its frontier trick, the manifest input that holds it
/// (format 2) and, once resolved, its local path.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NetRef {
    pub cut: u64,
    pub input: Option<String>,
    pub path: Option<String>,
}

/// The `check` of a `search_measure` chunk (module notes).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CheckSpec {
    pub net: NetRef,
    pub verify_pct: f64,
    /// Threads inside each recomputed reference solve (`--verify-solve-threads`: the parallel
    /// solver, the same values; a heavy trick-1 world on one thread can take hours).
    pub verify_solve_threads: u64,
    pub tt_mb: u64,
    pub standard: String,
    pub boot: u64,
    pub race_groups: bool,
}

/// A `search_measure` chunk's silver-reference settings (module notes): all absent, the chunk
/// exactly as before.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Silver {
    pub reference: Option<NetRef>,
    /// The checks, in order (`check`: one object, or a list of them with distinct cuts).
    pub checks: Vec<CheckSpec>,
    /// A stored worlds file to start from: its input name, and its local path once resolved.
    pub worlds_input: Option<String>,
    pub worlds_path: Option<String>,
    pub share: bool,
}

/// The keys of `reference` and of `check` (any other is refused).
const REFERENCE_KEYS: [&str; 3] = ["cut", "net_input", "net"];
const CHECK_KEYS: [&str; 9] = [
    "cut",
    "net_input",
    "net",
    "verify_pct",
    "verify_solve_threads",
    "tt_mb",
    "standard",
    "boot",
    "race_groups",
];

/// Reads a network block (`reference` or `check`) of `p`.
fn net_ref(v: &Value, what: &str, keys: &[&str]) -> Result<NetRef, String> {
    let o = v
        .as_object()
        .ok_or_else(|| format!("search_measure: {what} must be an object"))?;
    for k in o.keys() {
        if !keys.contains(&k.as_str()) {
            return Err(format!("search_measure: {what}: unknown setting `{k}`"));
        }
    }
    let cut = v
        .get("cut")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("search_measure: {what}.cut: the frontier trick, 2 to 5"))?;
    if !(2..=5).contains(&cut) {
        return Err(format!(
            "search_measure: {what}.cut {cut}: the frontier trick, 2 to 5"
        ));
    }
    let s = |k: &str| match v.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(x)) if !x.is_empty() => Ok(Some(x.clone())),
        Some(_) => Err(format!("search_measure: {what}.{k} must be a string")),
    };
    let (input, path) = (s("net_input")?, s("net")?);
    if input.is_none() && path.is_none() {
        return Err(format!(
            "search_measure: {what} needs net_input (a manifest input holding the network)"
        ));
    }
    Ok(NetRef { cut, input, path })
}

/// Checks the silver-reference settings of a `search_measure` chunk (also when the manifest is
/// read, so a typo stops the run before it starts).
pub(crate) fn parse_silver(p: &Value) -> Result<Silver, String> {
    let trick = n(p, "trick", 2);
    let reference = match p.get("reference") {
        None | Some(Value::Null) => None,
        Some(v) => Some(net_ref(v, "reference", &REFERENCE_KEYS)?),
    };
    if let Some(r) = &reference {
        if r.cut <= trick {
            return Err(format!(
                "search_measure: reference.cut {} must be after the decisions' trick {trick}",
                r.cut
            ));
        }
        if p.get("methods")
            .and_then(Value::as_array)
            .is_some_and(|a| !a.is_empty())
        {
            return Err(
                "search_measure: methods are scored against an exact yardstick (search score), \
                 not a silver reference"
                    .into(),
            );
        }
    }
    let check_values: Vec<Value> = match p.get("check") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) => a.clone(),
        Some(v) => vec![v.clone()],
    };
    let mut checks: Vec<CheckSpec> = Vec::new();
    for v in &check_values {
        let check = {
            let net = net_ref(v, "check", &CHECK_KEYS)?;
            if net.cut <= trick {
                return Err(format!(
                    "search_measure: check.cut {} must be after the decisions' trick {trick}",
                    net.cut
                ));
            }
            if let Some(r) = &reference
                && net.cut >= r.cut
            {
                return Err(format!(
                    "search_measure: check.cut {} must be before reference.cut {}: a silver \
                     reference looks deeper than the hybrid it checks",
                    net.cut, r.cut
                ));
            }
            if n(p, "from", 0) > 0 {
                return Err(
                    "search_measure: check needs the worlds from 0 (not an extension, from > 0)"
                        .into(),
                );
            }
            let verify_pct = match v.get("verify_pct") {
                None | Some(Value::Null) => 1.0,
                Some(x) => x
                    .as_f64()
                    .filter(|x| (0.0..=100.0).contains(x))
                    .ok_or("search_measure: check.verify_pct must be a number from 0 to 100")?,
            };
            let standard = match v.get("standard") {
                None | Some(Value::Null) => "0.01".to_string(),
                Some(Value::String(x)) => x.clone(),
                Some(_) => {
                    return Err(
                        "search_measure: check.standard must be a string, e.g. \"0.01\"".into(),
                    );
                }
            };
            let race_groups = match v.get("race_groups") {
                None | Some(Value::Null) => true,
                Some(b) => b
                    .as_bool()
                    .ok_or("search_measure: check.race_groups must be true or false")?,
            };
            CheckSpec {
                net,
                verify_pct,
                verify_solve_threads: v
                    .get("verify_solve_threads")
                    .and_then(Value::as_u64)
                    .unwrap_or(1)
                    .max(1),
                tt_mb: v.get("tt_mb").and_then(Value::as_u64).unwrap_or(64).max(1),
                standard,
                boot: v.get("boot").and_then(Value::as_u64).unwrap_or(1000),
                race_groups,
            }
        };
        if checks.iter().any(|c| c.net.cut == check.net.cut) {
            return Err(format!(
                "search_measure: two checks at cut {} (one a cut: their reports are named by it)",
                check.net.cut
            ));
        }
        checks.push(check);
    }
    let worlds_input = match p.get("worlds_input") {
        None | Some(Value::Null) => None,
        Some(Value::String(x)) if !x.is_empty() => Some(x.clone()),
        Some(_) => return Err("search_measure: worlds_input must be an input name".into()),
    };
    let worlds_path = p.get("worlds").and_then(Value::as_str).map(str::to_string);
    let share = match p.get("share") {
        None | Some(Value::Null) => reference.is_some(),
        Some(b) => b
            .as_bool()
            .ok_or("search_measure: share must be true or false")?,
    };
    Ok(Silver {
        reference,
        checks,
        worlds_input,
        worlds_path,
        share,
    })
}

/// The manifest inputs a chunk's silver settings name (for the manifest check, `sched.rs`).
pub(crate) fn input_names(p: &Value) -> Result<Vec<String>, String> {
    let q = parse_silver(p)?;
    Ok(q.reference
        .iter()
        .chain(q.checks.iter().map(|c| &c.net))
        .filter_map(|r| r.input.clone())
        .chain(q.worlds_input)
        .collect())
}

/// Adds the local path of each named input (`reference.net`, `check.net`, `worlds`) from the
/// worker's `inputs` (name → path); a name the manifest does not list is an error.
pub(crate) fn resolve(
    p: &Value,
    inputs: &std::collections::BTreeMap<String, String>,
) -> Result<Value, String> {
    let q = parse_silver(p)?;
    let path = |name: &str| {
        inputs
            .get(name)
            .cloned()
            .ok_or_else(|| format!("search_measure needs input `{name}` in the manifest's inputs"))
    };
    let mut out = p.clone();
    if let Some(i) = q.reference.as_ref().and_then(|r| r.input.as_deref()) {
        out["reference"]["net"] = json!(path(i)?);
    }
    match out.get_mut("check") {
        Some(Value::Array(a)) => {
            for (v, c) in a.iter_mut().zip(&q.checks) {
                if let Some(i) = c.net.input.as_deref() {
                    v["net"] = json!(path(i)?);
                }
            }
        }
        Some(v @ Value::Object(_)) => {
            if let Some(i) = q.checks.first().and_then(|c| c.net.input.as_deref()) {
                v["net"] = json!(path(i)?);
            }
        }
        _ => {}
    }
    if let Some(i) = &q.worlds_input {
        out["worlds"] = json!(path(i)?);
    }
    Ok(out)
}

/// The SHA-256 and frontier trick of a network file (`int.safetensors`): its trick must be `cut`.
fn net_facts(path: &str, cut: u64, what: &str) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("search_measure: {what} {path}: {e}"))?;
    let f = nn::safetensors::read(&bytes)
        .map_err(|e| format!("search_measure: {what} {path}: not a network ({e})"))?;
    let trick = f
        .meta("trick")
        .ok()
        .and_then(|t| t.parse::<u64>().ok())
        .ok_or_else(|| format!("search_measure: {what} {path}: no frontier trick recorded"))?;
    if trick != cut {
        return Err(format!(
            "search_measure: {what} is a trick-{trick} network (O_{trick}), not O_{cut}"
        ));
    }
    Ok(sha256_file(Path::new(path))?.0)
}

/// Run one `search_measure` chunk.
pub(crate) fn chunk(ctx: &mut Ctx, p: &Value) -> Result<Value, String> {
    let (table, table_rec) = build_table_from(ctx, p.get("table").unwrap_or(&json!({})), "auto")?;
    let remote = ctx
        .remote()
        .or_else(|| crate::remote::from_env().map(Arc::from));
    chunk_with(ctx, p, &sibling("search"), &table, table_rec, remote)
}

/// The chunk after the equity table, with the given `search` program and remote store (for
/// sharing a silver set).
pub(crate) fn chunk_with(
    ctx: &mut Ctx,
    p: &Value,
    search: &Path,
    table: &Path,
    table_rec: Value,
    remote: Option<Arc<dyn Remote>>,
) -> Result<Value, String> {
    let table = table.to_path_buf();
    let silver = parse_silver(p)?;
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
    let start_hands = n(p, "start_hands", 20_000);
    // The silver reference's network (its SHA-256 is part of the fingerprint).
    let reference = match &silver.reference {
        None => None,
        Some(r) => {
            let path = r.path.clone().ok_or(
                "search_measure: reference.net_input is not resolved to a file (a format-2 \
                 manifest's input)",
            )?;
            let sha = net_facts(&path, r.cut, "reference")?;
            Some((r.cut, path, sha))
        }
    };
    let mut fingerprint = json!({
        "method": if reference.is_some() { "hybrid" } else { "exact" },
        "label": 1, "seed": seed, "trick": trick, "count": count,
        "start_hands": start_hands, "ids": ids, "first": n(p, "first", 0),
        "last": n(p, "last", count.saturating_sub(1)), "from": from, "to": to,
        "table_sha256": crate::bundle::sha256_file(&table)?.0,
    });
    if let Some((cut, _, sha)) = &reference {
        fingerprint["frontier"] = json!({ "cut": cut, "net_sha256": sha });
    }
    check_fingerprint(&out.join("fingerprint.json"), &fingerprint)?;
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
    let mut c = Command::new(search);
    c.args(["decisions", "--trick", &trick.to_string()])
        .args(["--count", &count.to_string(), "--seed", &seed.to_string()])
        .args(["--start-hands", &start_hands.to_string()])
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
        let mut c = Command::new(search);
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
        // A silver reference: the hybrid with its network (the step's own method only).
        if let (Some((cut, path, _)), "hybrid") = (&reference, method) {
            c.args(["--cut", &cut.to_string(), "--net", path]);
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
    // How the worlds are spread, for the machine record: the position-aware plan, one thread
    // per world ("fill"), or each world split over several threads.
    let extra = json!({
        "schedule": if budget_mb.is_some() { "plan" } else if solve_threads > 1 { "split" } else { "fill" },
        "trick": trick, "tt_budget_mb": budget_mb,
    });
    // The worlds file: `exact-t<trick>-…` as before, `silver-t<trick>-c<cut>-…` for a silver
    // reference.
    let (method, kind) = match &reference {
        None => ("exact", format!("exact-t{trick}")),
        Some((cut, _, _)) => ("hybrid", format!("silver-t{trick}-c{cut}")),
    };
    let base_name = if from == 0 {
        format!("{kind}-n{to}.jsonl")
    } else {
        format!("{kind}-w{from}-{to}.jsonl")
    };
    let yard = out.join(&base_name);
    // A stored worlds file to start from (copied once; the step values only what it lacks).
    let mut stored_in = Value::Null;
    if let Some(src) = &silver.worlds_path {
        if !yard.exists() {
            fs::copy(src, &yard).map_err(|e| format!("search_measure: worlds_input {src}: {e}"))?;
        }
        let (sha, bytes) = sha256_file(Path::new(src))?;
        stored_in = json!({ "input": silver.worlds_input, "sha256": sha, "bytes": bytes });
    }
    let last = run_worlds(
        ctx,
        worlds(1, method, from, to, &yard),
        if reference.is_some() {
            "silver worlds"
        } else {
            "exact worlds"
        },
        to,
        threads,
        solve_threads,
        &tt,
        &progress_file,
        &extra,
    )?;
    let mut warnings: Vec<String> = tt.warning.iter().cloned().collect();
    let mut base = json!({ "resumed": resumed(&last), "trick": trick, "count": count, "seed": seed, "from": from, "to": to,
                       "ids": ids, "grid": grid, "solve_threads": solve_threads,
                       "worlds_threads": threads, "table": table_rec,
                       "schedule": if budget_mb.is_some() { "auto" } else { "fixed" },
                       "tt_budget_mb": budget_mb,
                       "tt": tt.record });
    base["worlds_file"] = json!(base_name);
    base["worlds_sha256"] = json!(sha256_file(&yard)?.0);
    if let Some((cut, path, sha)) = &reference {
        base["reference"] = json!({
            "kind": "silver", "cut": cut, "net_sha256": sha,
            "net": Path::new(path).file_name().map(|f| f.to_string_lossy().into_owned()),
            "what": "a silver reference (docs/m3-hybrid/silver.md): every world valued by the \
                     hybrid H_cut with this network, never exact values",
        });
    }
    if !stored_in.is_null() {
        base["worlds_input"] = stored_in;
    }
    if silver.share {
        let shared = share(ctx, remote.as_ref(), &dec, &yard, &mut warnings);
        base["shared"] = shared;
    }
    base["warnings"] = json!(warnings);
    if from > 0 {
        // An extension: the analysis needs the earlier worlds too, so it is run where both
        // files are (`search agree|curves --worlds <first run> --worlds <this>`).
        let mut r = base;
        r["methods"] = json!([]);
        r["extension"] = json!(true);
        return Ok(r);
    }
    let mut c = Command::new(search);
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
    let mut c = Command::new(search);
    c.arg("curves")
        .arg("--worlds")
        .arg(&yard)
        .args(["--grid", &grid])
        .args(["--method", method])
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
        let _ = run_worlds(
            ctx,
            worlds(label, method, 0, mto, &f),
            &format!("method {i} ({method}) worlds"),
            mto,
            threads,
            solve_threads,
            &tt,
            &progress_file,
            &extra,
        )?;
        methods.push(format!("{}:{mto}", f.display()));
    }
    if !methods.is_empty() {
        let mut c = Command::new(search);
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
    let mut checks = Vec::new();
    for cs in &silver.checks {
        let check = check_step(
            ctx,
            search,
            &CheckRun {
                spec: cs,
                reference: reference.as_ref().map(|(c, p, _)| (*c, p.as_str())),
                trick,
                seed,
                to,
                ids: ids.as_deref(),
                first: n(p, "first", 0),
                last: n(p, "last", count.saturating_sub(1)),
                table: &table,
                decisions: &dec,
                worlds: &yard,
                out: &out,
            },
        )?;
        checks.push(check);
    }
    if !checks.is_empty() {
        r["checks"] = json!(checks);
    }
    Ok(r)
}

/// Shares the decisions and worlds files content-addressed (`inputs/sha256/<hex>`), so a later
/// run names them as inputs (`worlds_input`); returns the record. A failure is a warning: both
/// files are in the bundle.
fn share(
    ctx: &mut Ctx,
    remote: Option<&Arc<dyn Remote>>,
    decisions: &Path,
    worlds: &Path,
    warnings: &mut Vec<String>,
) -> Value {
    let Some(remote) = remote else {
        let w = "the worlds were not shared for reuse: no remote store (R2, or \
                 RUNNER_REMOTE_DIR); they are in this run's bundle"
            .to_string();
        eprintln!("runner: WARNING: search_measure: {w}");
        warnings.push(w);
        return Value::Null;
    };
    ctx.set_phase(
        "chunk",
        json!({ "kind": "search_measure", "step": "share worlds", "partial_cpu_ok": true }),
    );
    let mut rec = serde_json::Map::new();
    for (name, path) in [("decisions", decisions), ("worlds", worlds)] {
        let (p2, r) = (path.to_path_buf(), Arc::clone(remote));
        match ctx.blocking(move || put_input(&*r, &p2, None)) {
            Ok((sha, bytes, status)) => {
                println!(
                    "runner: search_measure: {name}: {} ({bytes} bytes, {status})",
                    input_key(&sha)
                );
                rec.insert(
                    name.into(),
                    json!({ "key": input_key(&sha), "sha256": sha, "bytes": bytes }),
                );
            }
            Err(e) => {
                let w = format!(
                    "the {name} file was not shared for reuse: {e}; it is in this run's bundle"
                );
                eprintln!("runner: WARNING: search_measure: {w}");
                warnings.push(w);
            }
        }
    }
    if let Some(k) = rec.get("worlds").and_then(|w| w["key"].as_str()) {
        rec.insert(
            "reuse_params".into(),
            json!({ "worlds_input": "<a manifest input name for this key>", "key": k,
                    "note": "a format-2 manifest lists the key under its inputs, names it in \
                             worlds_input, and keeps reference, trick, count, seed and to as \
                             they were (the decisions are made again, byte-identical)" }),
        );
    }
    Value::Object(rec)
}

/// What the check step runs on.
struct CheckRun<'a> {
    spec: &'a CheckSpec,
    /// The silver reference's cut and network file (`None`: the worlds are exact).
    reference: Option<(u64, &'a str)>,
    trick: u64,
    seed: u64,
    to: u64,
    ids: Option<&'a str>,
    first: u64,
    last: u64,
    table: &'a Path,
    decisions: &'a Path,
    worlds: &'a Path,
    out: &'a Path,
}

/// The `search hybrid-check` command of the check step, and its JSON and text outputs.
fn check_command(
    search: &Path,
    r: &CheckRun,
    threads: usize,
    progress: &Path,
) -> Result<(Command, PathBuf, PathBuf), String> {
    let cs = r.spec;
    let net = cs.net.path.as_deref().ok_or(
        "search_measure: check.net_input is not resolved to a file (a format-2 manifest's input)",
    )?;
    let kind = if r.reference.is_some() {
        "silver"
    } else {
        "exact"
    };
    let stem = format!("hc-t{}-o{}-{kind}", r.trick, cs.net.cut);
    let (json_out, txt_out) = (
        r.out.join(format!("{stem}.json")),
        r.out.join(format!("{stem}.txt")),
    );
    let mut c = Command::new(search);
    c.args(["hybrid-check", "--trick", &r.trick.to_string()])
        .args(["--cut", &cs.net.cut.to_string()])
        .arg("--net")
        .arg(net)
        .arg("--equity")
        .arg(r.table)
        .arg("--decisions")
        .arg(r.decisions)
        .arg("--worlds")
        .arg(r.worlds)
        .args(["--to", &r.to.to_string(), "--seed", &r.seed.to_string()])
        .args([
            "--first",
            &r.first.to_string(),
            "--last",
            &r.last.to_string(),
        ]);
    if let Some(ids) = r.ids {
        c.args(["--ids", ids]);
    }
    if let Some((cut, path)) = r.reference {
        c.args(["--reference", "silver", "--reference-cut", &cut.to_string()])
            .arg("--reference-net")
            .arg(path);
    }
    if cs.race_groups {
        c.arg("--race-groups");
    }
    c.args(["--verify-pct", &cs.verify_pct.to_string()])
        .args([
            "--verify-solve-threads",
            &cs.verify_solve_threads.to_string(),
        ])
        .args(["--threads", &threads.to_string()])
        .args(["--tt-mb", &cs.tt_mb.to_string()])
        .args(["--boot", &cs.boot.to_string()])
        .args(["--standard", &cs.standard])
        .arg("--json")
        .arg(&json_out)
        .arg("--progress")
        .arg(progress);
    Ok((c, json_out, txt_out))
}

/// The check step: H_cut of the check's network against the chunk's worlds (module notes).
fn check_step(ctx: &mut Ctx, search: &Path, r: &CheckRun) -> Result<Value, String> {
    let threads = ctx.threads.max(1);
    let progress = ctx.root.join("check-progress.json");
    let _ = fs::remove_file(&progress);
    let (c, json_out, txt_out) = check_command(search, r, threads, &progress)?;
    let step = json!({ "kind": "search_measure", "step": "hybrid check",
                       "cut": r.spec.net.cut, "partial_cpu_ok": false });
    ctx.set_phase("chunk", step.clone());
    ctx.run_child_io(c, "measure/check.log", Some(&txt_out), None, &mut |ctx| {
        let mut s = step.clone();
        if let Some(x) = fs::read_to_string(&progress)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        {
            for k in ["worlds_done", "worlds_total", "exact_done", "exact_total"] {
                s[k] = x[k].clone();
            }
            s["partial_cpu_ok"] = json!(x["tail"].as_bool().unwrap_or(false));
        }
        ctx.progress = s;
    })?;
    let hc: Value = fs::read_to_string(&json_out)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .ok_or("search_measure: hybrid-check wrote no JSON")?;
    let rows = |g: &Value| -> Vec<Value> {
        g.as_array()
            .into_iter()
            .flatten()
            .take(4)
            .cloned()
            .collect()
    };
    Ok(json!({
        "file": json_out.file_name().map(|f| f.to_string_lossy().into_owned()),
        "cut": r.spec.net.cut,
        "net_sha256": hc["net"]["sha256"],
        "reference": hc["reference"],
        "pass": hc["pass"],
        "standard_pct": hc["standard_pct"],
        "verified_worlds": hc["verified_worlds"],
        "groups": rows(&hc["groups"]),
        "race_groups": hc["race_groups"]["groups"],
        "score_groups": hc["score_groups"]["groups"],
        "time": hc["time"],
    }))
}

/// Writes the chunk's fingerprint to `path`, or, when a restored one is there already (a
/// resumed chunk), checks it field by field: any difference is an error naming the fields, so a
/// run never builds on files made under other settings. A file that cannot be read is an error
/// too (nothing is guessed).
fn check_fingerprint(path: &Path, want: &Value) -> Result<(), String> {
    if path.is_file() {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let have: Value = serde_json::from_str(&text)
            .map_err(|e| format!("{}: not a fingerprint ({e})", path.display()))?;
        let differ: Vec<String> = want
            .as_object()
            .into_iter()
            .flatten()
            .filter(|(k, v)| have.get(*k) != Some(v))
            .map(|(k, v)| format!("{k}: {} now {v}", have.get(k).unwrap_or(&Value::Null)))
            .collect();
        if !differ.is_empty() {
            return Err(format!(
                "resume refused: the files being resumed were made with other settings ({}); \
                 resume with the same settings, or start a new run without resume_from",
                differ.join("; ")
            ));
        }
        return Ok(());
    }
    fs::write(path, want.to_string()).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fingerprint is written once and then checked: the same settings pass, a changed field
    /// (the seed, the table) is refused by name, and an unreadable file is an error.
    #[test]
    fn fingerprint_refuses_other_settings() {
        let d = std::env::temp_dir().join(format!("runner-fp-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).expect("mkdir");
        let p = d.join("fingerprint.json");
        let fp = json!({ "seed": 1, "trick": 2, "table_sha256": "abc" });
        check_fingerprint(&p, &fp).expect("written");
        check_fingerprint(&p, &fp).expect("same settings");
        let e = check_fingerprint(&p, &json!({ "seed": 2, "trick": 2, "table_sha256": "abc" }))
            .expect_err("seed");
        assert!(e.contains("seed: 1 now 2") && !e.contains("table"), "{e}");
        let e = check_fingerprint(&p, &json!({ "seed": 1, "trick": 2, "table_sha256": "def" }))
            .expect_err("table");
        assert!(e.contains("table_sha256: \"abc\" now \"def\""), "{e}");
        fs::write(&p, "{not json").expect("write");
        assert!(check_fingerprint(&p, &fp).is_err());
        let _ = fs::remove_dir_all(&d);
    }

    /// A silver reference's settings: checked (cuts, the reference deeper than the check, no
    /// methods, no extension with a check, known keys, a network named), the inputs they name, and
    /// those inputs resolved to paths; absent, nothing changes (and nothing is shared).
    #[test]
    fn silver_settings_are_checked() {
        let p = json!({ "trick": 1, "count": 32, "to": 8,
                        "reference": { "cut": 3, "net_input": "o3" },
                        "check": { "cut": 2, "net_input": "o2", "verify_pct": 2.5 },
                        "worlds_input": "stored" });
        let q = parse_silver(&p).expect("valid");
        assert_eq!(q.reference.as_ref().map(|r| r.cut), Some(3));
        let c = q.checks.first().expect("a check");
        assert_eq!(
            (c.net.cut, c.verify_pct, c.tt_mb, c.boot),
            (2, 2.5, 64, 1000)
        );
        assert!(c.race_groups && c.standard == "0.01" && q.share);
        assert_eq!(input_names(&p).expect("names"), ["o3", "o2", "stored"]);
        let plain = parse_silver(&json!({ "trick": 2 })).expect("plain");
        assert!(plain.reference.is_none() && plain.checks.is_empty() && !plain.share);
        assert!(input_names(&json!({})).expect("none").is_empty());
        let bad = |f: &dyn Fn(&mut Value), want: &str| {
            let mut x = p.clone();
            f(&mut x);
            let e = parse_silver(&x).expect_err(want);
            assert!(e.contains(want), "{e} (wanted {want})");
        };
        bad(&|x| x["reference"]["cut"] = json!(1), "reference.cut 1");
        bad(&|x| x["reference"]["cut"] = json!(6), "2 to 5");
        bad(
            &|x| x["check"]["cut"] = json!(3),
            "must be before reference.cut",
        );
        bad(&|x| x["trick"] = json!(2), "check.cut 2 must be after");
        bad(
            &|x| x["reference"]["typo"] = json!(1),
            "unknown setting `typo`",
        );
        bad(&|x| x["check"]["typo"] = json!(1), "unknown setting `typo`");
        bad(&|x| x["reference"] = json!({ "cut": 3 }), "needs net_input");
        bad(
            &|x| x["methods"] = json!([{ "method": "rollout" }]),
            "not a silver reference",
        );
        bad(&|x| x["from"] = json!(4), "not an extension");
        bad(&|x| x["check"]["verify_pct"] = json!(101), "verify_pct");
        bad(&|x| x["share"] = json!("yes"), "share must be");
        // Resolved: the paths join the settings, which stay valid.
        let mut inputs = std::collections::BTreeMap::new();
        inputs.insert("o3".to_string(), "/in/o3".to_string());
        inputs.insert("o2".to_string(), "/in/o2".to_string());
        let e = resolve(&p, &inputs).expect_err("stored is missing");
        assert!(e.contains("`stored`"), "{e}");
        inputs.insert("stored".to_string(), "/in/w".to_string());
        let r = resolve(&p, &inputs).expect("resolves");
        let q = parse_silver(&r).expect("still valid");
        assert_eq!(q.reference.and_then(|r| r.path).as_deref(), Some("/in/o3"));
        assert_eq!(q.checks[0].net.path.as_deref(), Some("/in/o2"));
        // Several checks (the exact anchor set: H_3 and H_2 against the same exact worlds), each
        // resolved; two at one cut are refused.
        let anchor = json!({ "trick": 1, "check": [{ "cut": 3, "net_input": "o3" },
                                                   { "cut": 2, "net_input": "o2" }] });
        assert_eq!(input_names(&anchor).expect("names"), ["o3", "o2"]);
        let ra = resolve(&anchor, &inputs).expect("resolves");
        let qa = parse_silver(&ra).expect("valid");
        assert_eq!(
            qa.checks
                .iter()
                .map(|c| (c.net.cut, c.net.path.clone()))
                .collect::<Vec<_>>(),
            [
                (3, Some("/in/o3".to_string())),
                (2, Some("/in/o2".to_string()))
            ]
        );
        assert!(!qa.share && qa.reference.is_none());
        let e = parse_silver(
            &json!({ "trick": 1, "check": [{ "cut": 2, "net_input": "a" },
                                                            { "cut": 2, "net_input": "b" }] }),
        )
        .expect_err("two at cut 2");
        assert!(e.contains("two checks at cut 2"), "{e}");
        assert_eq!(q.worlds_path.as_deref(), Some("/in/w"));
        // The manifest check: a named input must be listed.
        let m = json!({ "format": 2, "run_id": "silver-test", "run_seed": 1,
                        "inputs": { "o3": format!("sha256:{}", "a".repeat(64)) },
                        "streams": [{ "name": "silver", "priority": 1, "kind": "search_measure",
                                      "chunks": 1, "params": p }] });
        let e = crate::sched::parse_v2(&m).expect_err("o2 and stored are not listed");
        assert!(e.contains("`o2`"), "{e}");
    }

    /// The chunk with a silver reference, with a fake `search`: the worlds step is the hybrid
    /// with the reference's network into `silver-t1-c3-n4.jsonl` (a stored worlds file copied in
    /// first), the fingerprint names it, the analysis reads it as `hybrid`, the decisions and
    /// worlds are shared content-addressed, and the check runs H_2 against it as a silver
    /// reference, its verdict in `result.json`. A network of the wrong trick is refused, and a
    /// resume under another network is refused by the fingerprint.
    #[cfg(unix)]
    #[test]
    fn silver_chunk_runs_shares_and_checks() {
        use std::collections::BTreeMap;
        use std::os::unix::fs::PermissionsExt;
        let base = std::env::temp_dir().join(format!("measure-silver-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("in")).expect("mkdir");
        let net = |k: u8, seed: u64| {
            let n = nn::synthetic::random_net(seed, &[8], nn::outcome::vocab(k), 100);
            let meta = BTreeMap::from([("head".to_string(), "outcome-v1".to_string())]);
            let f = base.join("in").join(format!("o{k}-{seed}"));
            fs::write(&f, nn::oracle::write_net(&n, k, &meta)).expect("write");
            f.display().to_string()
        };
        let (o3, o2, o3b) = (net(3, 1), net(2, 2), net(3, 3));
        let stored = base.join("in").join("stored");
        fs::write(&stored, "{\"stored\": true}\n").expect("write");
        let table = base.join("equity-v0.npy");
        fs::write(&table, "table").expect("write");
        let calls = base.join("calls.log");
        let search = base.join("search");
        let arg = r#"arg() { n=$1; shift; while [ $# -gt 0 ]; do [ "$1" = "$n" ] && { echo "$2"; return; }; shift; done; }"#;
        fs::write(
            &search,
            format!(
                r#"#!/bin/sh
{arg}
echo "search $*" >> '{calls}'
case "$1" in
  decisions) echo '{{"id": 0}}' > "$(arg --out "$@")" ;;
  worlds) echo '{{"id": 0, "method": "hybrid"}}' >> "$(arg --out "$@")"
          echo '{{"worlds_done": 4, "worlds_total": 4}}' > "$(arg --progress "$@")" ;;
  agree|curves) echo '{{}}' > "$(arg --json "$@")" ;;
  hybrid-check)
    echo '{{"exact_done": 1, "exact_total": 1, "worlds_done": 4, "worlds_total": 4}}' > "$(arg --progress "$@")"
    echo '{{"net": {{"sha256": "o2sha"}}, "reference": {{"kind": "silver", "cut": 3}}, "pass": true,
      "standard_pct": "0.01", "verified_worlds": 1,
      "groups": [{{"group": "all"}}, {{"group": "pos 1-2"}}, {{"group": "pos 3-4"}}, {{"group": "pos 5-8"}}, {{"group": "pos 1"}}],
      "race_groups": {{"groups": [{{"group": "race 2"}}]}}, "score_groups": {{"groups": [{{"group": "S1"}}]}},
      "time": {{"wall_s": 1}}}}' > "$(arg --json "$@")"
    echo "the check's tables" ;;
esac
"#,
                calls = calls.display()
            ),
        )
        .expect("write");
        fs::set_permissions(&search, fs::Permissions::from_mode(0o755)).expect("chmod");
        let remote_dir = base.join("remote");
        let remote: Arc<dyn Remote> = Arc::new(crate::remote::DirRemote {
            root: remote_dir.clone(),
        });
        let params = |o3: &str| {
            json!({ "trick": 1, "count": 32, "to": 4, "grid": "2,4", "worlds_at_once": 1,
                    "solve_threads": 1, "tt_mb": 64, "allow_uneven": true,
                    "reference": { "cut": 3, "net": o3 },
                    "check": { "cut": 2, "net": o2.clone(), "verify_pct": 3 },
                    "worlds_input": "stored", "worlds": stored.display().to_string() })
        };
        let chunk_dir = base.join("chunk");
        let mut ctx = Ctx::local("silver-test", &chunk_dir, 4);
        let r = chunk_with(
            &mut ctx,
            &params(&o3),
            &search,
            &table,
            json!({"sha256": "t"}),
            Some(Arc::clone(&remote)),
        )
        .expect("runs");
        let out = chunk_dir.join("out/measure");
        assert_eq!(r["worlds_file"], "silver-t1-c3-n4.jsonl");
        assert_eq!(r["reference"]["kind"], "silver");
        assert_eq!(r["reference"]["cut"], 3);
        let o3_sha = sha256_file(Path::new(&o3)).expect("sha").0;
        assert_eq!(r["reference"]["net_sha256"], json!(o3_sha));
        assert_eq!(r["worlds_input"]["input"], "stored");
        // The stored file was copied in, and the step appended to it.
        let worlds = fs::read_to_string(out.join("silver-t1-c3-n4.jsonl")).expect("worlds");
        assert!(worlds.starts_with("{\"stored\": true}\n"), "{worlds}");
        // Shared, content-addressed.
        for what in ["decisions", "worlds"] {
            let key = r["shared"][what]["key"].as_str().expect("a key");
            assert!(remote_dir.join(key).is_file(), "{key}");
        }
        let fp: Value = serde_json::from_str(
            &fs::read_to_string(out.join("fingerprint.json")).expect("fingerprint"),
        )
        .expect("json");
        assert_eq!(fp["method"], "hybrid");
        assert_eq!(fp["frontier"], json!({ "cut": 3, "net_sha256": o3_sha }));
        // The check.
        assert_eq!(r["checks"][0]["pass"], true);
        assert_eq!(r["checks"][0]["file"], "hc-t1-o2-silver.json");
        assert_eq!(r["checks"][0]["groups"].as_array().map(Vec::len), Some(4));
        assert_eq!(r["checks"][0]["race_groups"][0]["group"], "race 2");
        assert_eq!(
            fs::read_to_string(out.join("hc-t1-o2-silver.txt")).expect("report"),
            "the check's tables\n"
        );
        let call = fs::read_to_string(&calls).expect("calls");
        let line = |sub: &str| {
            call.lines()
                .find(|l| l.starts_with(&format!("search {sub} ")))
                .unwrap_or_else(|| panic!("no {sub} call: {call}"))
                .to_string()
        };
        let w = line("worlds");
        assert!(
            w.contains("--method hybrid") && w.contains(&format!("--cut 3 --net {o3}")),
            "{w}"
        );
        assert!(line("curves").contains("--method hybrid"));
        let h = line("hybrid-check");
        assert!(
            h.contains(&format!("--cut 2 --net {o2}"))
                && h.contains(&format!(
                    "--reference silver --reference-cut 3 --reference-net {o3}"
                ))
                && h.contains("--race-groups --verify-pct 3 --verify-solve-threads 1")
                && h.contains("silver-t1-c3-n4.jsonl"),
            "{h}"
        );
        assert!(!call.contains("search score"), "no methods");
        // A resume under another network is refused (the fingerprint), as is a network of
        // another trick.
        let mut ctx = Ctx::local("silver-test", &chunk_dir, 4);
        let e = chunk_with(&mut ctx, &params(&o3b), &search, &table, json!({}), None)
            .expect_err("another network");
        assert!(e.contains("frontier"), "{e}");
        let mut ctx = Ctx::local("silver-test", &base.join("chunk2"), 4);
        let e = chunk_with(&mut ctx, &params(&o2), &search, &table, json!({}), None)
            .expect_err("an O_2 as the reference");
        assert!(e.contains("trick-2 network"), "{e}");
        let _ = fs::remove_dir_all(&base);
    }

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
