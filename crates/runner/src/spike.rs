//! The `solver_spike` chunk: the M2 solver cost measurement (docs/M2-SPIKE.md) on a second CPU
//! type, plus a solver-identity check.
//!
//! ```json
//! {"solver_spike": {"plan": "xeon", "identity": "moderate"}}
//! ```
//!
//! Steps:
//! 1. Build match-equity table v0 on the box (`equity estimate-model --bot rules-v1 --hands
//!    1000000 --seed 1`, then `equity build`) and check its SHA-256 against
//!    `data/equity/equity-v0-rules-v1.sha256`; a mismatch fails the chunk. (`hands` other than
//!    1,000,000 or seed other than 1 needs `"allow_unverified_table": true`; for smoke tests.)
//! 2. Run the `solver-spike` jobs of the plan: `xeon` repeats every run in docs/M2-SPIKE.md with
//!    the same seed, world counts, tricks, caps and per-job thread counts; `tiny` is a smoke test.
//!    Independent jobs run at the same time while their threads fit in the usable cores and
//!    their memory estimates fit in 80% of RAM; each job's record says how many threads were
//!    busy on the box while it ran (`busy_threads_max`), so per-world times can be read with
//!    the contention in view. `"concurrent": false` runs them one at a time.
//! 3. Run `solver-identity` (all 16 switch combinations, both tie orders, real table) on all
//!    threads; any difference fails the chunk.
//!
//! Plan `trick1` (`{"solver_spike": {"plan": "trick1"}}`) instead measures how long one exact
//! trick-1 solve takes with the whole box: after step 1 it solves worlds 0..`worlds` (default
//! 4; seed 1, the same worlds as the spike's trick-1 runs) **one at a time**, each with the
//! parallel single-world solve on all usable threads and a shared table of `tt_ram_pct`
//! (default 70) percent of RAM, with a per-world cap of `cap_secs` (default 3600; a world that
//! hits it is recorded as "> 60 min" and the next one starts). `trick` (default 1) and
//! `allow_unverified_table` allow a quick plumbing test at trick 3. While a solve runs the CPU
//! should be near 100%; only its last stretch, when fewer subproblems remain than threads, is
//! marked `partial_cpu_ok`. Outputs `out/trick1/world-<w>.txt` (the binary's progress and
//! result lines) and `out/trick1/results.json`.
//!
//! Outputs: `out/spike/<job>.txt` (the binaries' own output), `out/spike/jobs.json`,
//! `out/identity/<job>.txt`, `out/equity/outcome.json`, `out/equity/equity-v0.sha256`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::{Ctx, sibling, unix_now};

/// SHA-256 of the recorded table v0 (rules-v1 bot, 1,000,000 hands, seed 1).
const EQUITY_V0_SHA256: &str = include_str!("../../../data/equity/equity-v0-rules-v1.sha256");

struct Job {
    name: String,
    prog: &'static str,
    args: Vec<String>,
    threads: usize,
    mem_mb: u64,
}

#[allow(clippy::too_many_arguments)] // one flat row per docs/M2-SPIKE.md run reads best
fn spike(
    name: &str,
    table: &str,
    solver: &str,
    trick: u8,
    worlds: u64,
    extra: &[&str],
    threads: usize,
    mem_mb: u64,
) -> Job {
    let mut args: Vec<String> = [
        "--table",
        table,
        "--solver",
        solver,
        "--trick",
        &trick.to_string(),
        "--worlds",
        &worlds.to_string(),
        "--seed",
        "1",
        "--threads",
        &threads.to_string(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    args.extend(extra.iter().map(|s| s.to_string()));
    Job {
        name: name.into(),
        prog: "solver-spike",
        args,
        threads,
        mem_mb,
    }
}

/// The docs/M2-SPIKE.md runs, longest first so the short single-thread jobs fill around them.
/// Memory estimates include the 76 MB table each process loads.
fn plan_jobs(plan: &str, table: &str) -> Result<Vec<Job>, String> {
    let s = spike;
    let cap60 = ["--cap-ms", "60000"];
    Ok(match plan {
        "xeon" => vec![
            // Tricks 1-2: four worlds at once (one per thread), 60 s cap; trick 1 also 300 s.
            s(
                "fast-t1-cap300-2at-once",
                table,
                "fast",
                1,
                2,
                &["--cap-ms", "300000", "--tt-mb", "3072"],
                2,
                2 * 3400,
            ),
            s(
                "naive-t1-tt-cap60",
                table,
                "naive",
                1,
                16,
                &["--tt", "on", "--cap-ms", "60000"],
                4,
                3400,
            ),
            s(
                "fast-t1-cap60",
                table,
                "fast",
                1,
                8,
                &["--cap-ms", "60000", "--tt-mb", "2048"],
                4,
                4 * 2200,
            ),
            s(
                "naive-t2-tt-cap60",
                table,
                "naive",
                2,
                100,
                &["--tt", "on", "--cap-ms", "60000"],
                4,
                2600,
            ),
            s("naive-t2-cap60", table, "naive", 2, 8, &cap60, 4, 300),
            s(
                "fast-t2-cap60",
                table,
                "fast",
                2,
                100,
                &["--cap-ms", "60000", "--tt-mb", "512"],
                4,
                4 * 600,
            ),
            // Tricks 3-5: one thread each.
            s("naive-t3", table, "naive", 3, 1000, &[], 1, 300),
            s(
                "naive-t3-tt-flip",
                table,
                "naive",
                3,
                1000,
                &["--tt", "on", "--flip"],
                1,
                400,
            ),
            s("fast-t3-flip", table, "fast", 3, 1000, &["--flip"], 1, 400),
            s(
                "fast-t3-sw1",
                table,
                "fast",
                3,
                1000,
                &["--switches", "1"],
                1,
                400,
            ),
            s(
                "fast-t3-sw2",
                table,
                "fast",
                3,
                1000,
                &["--switches", "2"],
                1,
                400,
            ),
            s(
                "fast-t3-sw3",
                table,
                "fast",
                3,
                1000,
                &["--switches", "3"],
                1,
                400,
            ),
            s(
                "fast-t3-sw9",
                table,
                "fast",
                3,
                1000,
                &["--switches", "9"],
                1,
                400,
            ),
            s(
                "fast-t3-sw15",
                table,
                "fast",
                3,
                1000,
                &["--switches", "15"],
                1,
                400,
            ),
            s("naive-t4", table, "naive", 4, 2000, &[], 1, 300),
            s(
                "naive-t4-tt-flip",
                table,
                "naive",
                4,
                2000,
                &["--tt", "on", "--flip"],
                1,
                300,
            ),
            s("fast-t4", table, "fast", 4, 2000, &[], 1, 600),
            s("naive-t5", table, "naive", 5, 2000, &[], 1, 300),
            s("fast-t5", table, "fast", 5, 2000, &[], 1, 600),
        ],
        "tiny" => vec![
            s(
                "naive-t4",
                table,
                "naive",
                4,
                20,
                &["--tt", "on", "--flip"],
                1,
                300,
            ),
            s("fast-t4", table, "fast", 4, 20, &[], 1, 600),
            s("naive-t5", table, "naive", 5, 20, &[], 1, 300),
            s("fast-t5", table, "fast", 5, 20, &[], 1, 600),
            s(
                "fast-t3",
                table,
                "fast",
                3,
                4,
                &["--cap-ms", "60000"],
                2,
                2 * 400,
            ),
        ],
        other => return Err(format!("unknown solver_spike plan `{other}` (xeon|tiny)")),
    })
}

/// solver-identity runs: (trick, worlds) on all threads, every combination, real table.
fn identity_runs(size: &str) -> Result<Vec<(u8, u64)>, String> {
    Ok(match size {
        "none" => vec![],
        "tiny" => vec![(5, 200), (4, 100)],
        "moderate" => vec![(5, 50_000), (4, 50_000), (3, 1000)],
        other => {
            return Err(format!(
                "unknown identity size `{other}` (none|tiny|moderate)"
            ));
        }
    })
}

/// Memory available to this container in MB, times 4/5.
fn mem_budget_mb() -> u64 {
    mem_total_mb() * 4 / 5
}

/// Memory available to this container in MB: RAM, capped by a cgroup v2/v1 limit.
fn mem_total_mb() -> u64 {
    let total = fs::read_to_string("/proc/meminfo").ok().and_then(|m| {
        m.lines()
            .find(|l| l.starts_with("MemTotal:"))
            .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
            .map(|kb| kb / 1024)
    });
    let cg = crate::read_trim("/sys/fs/cgroup/memory.max")
        .or_else(|| crate::read_trim("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
        .and_then(|s| s.parse::<u64>().ok())
        .map(|b| b >> 20);
    match (total, cg) {
        (Some(t), Some(c)) => t.min(c),
        (Some(t), None) => t,
        (None, Some(c)) => c,
        (None, None) => 16 * 1024,
    }
}

fn sha256_file(p: &Path) -> Result<String, String> {
    let o = Command::new("sha256sum")
        .arg(p)
        .output()
        .map_err(|e| format!("cannot run sha256sum: {e}"))?;
    if !o.status.success() {
        return Err(format!("sha256sum {} failed", p.display()));
    }
    String::from_utf8_lossy(&o.stdout)
        .split_whitespace()
        .next()
        .map(str::to_string)
        .ok_or_else(|| "sha256sum printed nothing".into())
}

/// Step 1: build table v0 and verify it. Returns the table path and its record.
pub(crate) fn build_table(ctx: &mut Ctx, params: &Value) -> Result<(PathBuf, Value), String> {
    let hands = params
        .get("hands")
        .and_then(Value::as_u64)
        .unwrap_or(1_000_000);
    let seed = params.get("seed").and_then(Value::as_u64).unwrap_or(1);
    let allow = params
        .get("allow_unverified_table")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let recorded = (hands, seed) == (1_000_000, 1);
    if !recorded && !allow {
        return Err("table v0 needs hands=1000000 and seed=1 (or allow_unverified_table)".into());
    }
    let dir = ctx.root.join("out").join("equity");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let model = dir.join("outcome.json");
    // The 76 MB table stays outside out/ (no need to upload it; its hash is recorded).
    let table = ctx.root.join("equity-v0.npy");
    let threads = ctx.threads.to_string();
    let t0 = Instant::now();
    ctx.set_phase(
        "chunk",
        json!({ "kind": "solver_spike", "step": "equity estimate-model" }),
    );
    let mut c = Command::new(sibling("equity"));
    c.args(["estimate-model", "--bot", "rules-v1"])
        .args(["--hands", &hands.to_string(), "--seed", &seed.to_string()])
        .args(["--threads", &threads])
        .arg("--out")
        .arg(&model);
    ctx.run_child(c, "equity/equity.log")?;
    ctx.set_phase(
        "chunk",
        json!({ "kind": "solver_spike", "step": "equity build" }),
    );
    let mut c = Command::new(sibling("equity"));
    c.arg("build")
        .arg("--model")
        .arg(&model)
        .arg("--out")
        .arg(&table)
        .args(["--threads", &threads]);
    ctx.run_child(c, "equity/equity.log")?;
    let got = sha256_file(&table)?;
    let want = EQUITY_V0_SHA256.trim();
    fs::write(dir.join("equity-v0.sha256"), format!("{got}\n")).map_err(|e| e.to_string())?;
    let matches = got == want;
    if recorded && !matches {
        return Err(format!(
            "equity table v0 sha256 {got} differs from the recorded {want}"
        ));
    }
    Ok((
        table,
        json!({
            "hands": hands, "seed": seed, "sha256": got, "recorded_sha256": want,
            "matches_recorded": matches, "seconds": t0.elapsed().as_secs_f64().round(),
        }),
    ))
}

struct Running {
    idx: usize,
    child: Child,
    started: f64,
    busy_max: usize,
    jobs_max: usize,
}

/// Step 2: run the jobs, several at once when they fit.
fn run_pool(
    ctx: &mut Ctx,
    jobs: &[Job],
    dir: &str,
    concurrent: bool,
) -> Result<Vec<Value>, String> {
    let out = ctx.root.join("out").join(dir);
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let cores = ctx.threads;
    let mem = mem_budget_mb();
    let t0 = Instant::now();
    let mut pending: Vec<usize> = (0..jobs.len()).collect();
    let mut running: Vec<Running> = Vec::new();
    let mut records: Vec<Option<Value>> = vec![None; jobs.len()];
    let mut failure: Option<String> = None;
    loop {
        // Start what fits (in list order; a job larger than the box runs alone).
        let busy: usize = running.iter().map(|r| jobs[r.idx].threads).sum();
        let used: u64 = running.iter().map(|r| jobs[r.idx].mem_mb).sum();
        let (mut busy, mut used) = (busy, used);
        let mut k = 0;
        while k < pending.len() && failure.is_none() {
            let j = &jobs[pending[k]];
            let alone = running.is_empty();
            let fits = busy + j.threads <= cores && used + j.mem_mb <= mem;
            if alone || (concurrent && fits) {
                let mut c = Command::new(sibling(j.prog));
                c.args(&j.args);
                let f = fs::File::create(out.join(format!("{}.txt", j.name)))
                    .map_err(|e| e.to_string())?;
                let f2 = f.try_clone().map_err(|e| e.to_string())?;
                c.stdin(Stdio::null()).stdout(f).stderr(f2);
                let child = c
                    .spawn()
                    .map_err(|e| format!("cannot start {}: {e}", j.prog))?;
                busy += j.threads;
                used += j.mem_mb;
                running.push(Running {
                    idx: pending.remove(k),
                    child,
                    started: t0.elapsed().as_secs_f64(),
                    busy_max: 0,
                    jobs_max: 0,
                });
                if !concurrent {
                    break;
                }
            } else {
                k += 1;
            }
        }
        if running.is_empty() {
            break;
        }
        let n = running.len();
        for r in &mut running {
            r.busy_max = r.busy_max.max(busy);
            r.jobs_max = r.jobs_max.max(n);
        }
        ctx.progress = json!({
            "kind": "solver_spike", "step": dir, "done": records.iter().filter(|r| r.is_some()).count(),
            "running": running.iter().map(|r| jobs[r.idx].name.as_str()).collect::<Vec<_>>(),
            "pending": pending.len(), "busy_threads": busy,
            // Timing jobs use fixed thread counts, so the box is not always full.
            "partial_cpu_ok": true,
        });
        // Reap finished jobs.
        let mut i = 0;
        while i < running.len() {
            if let Some(st) = running[i].child.try_wait().map_err(|e| e.to_string())? {
                let r = running.remove(i);
                let j = &jobs[r.idx];
                if !st.success() && failure.is_none() {
                    failure = Some(format!(
                        "{} exited with {st} (see out/{dir}/{}.txt)",
                        j.prog, j.name
                    ));
                }
                records[r.idx] = Some(json!({
                    "name": j.name, "prog": j.prog, "args": j.args, "threads": j.threads,
                    "mem_estimate_mb": j.mem_mb, "start_s": (r.started * 10.0).round() / 10.0,
                    "wall_s": ((t0.elapsed().as_secs_f64() - r.started) * 10.0).round() / 10.0,
                    "exit_ok": st.success(), "busy_threads_max": r.busy_max,
                    "concurrent_jobs_max": r.jobs_max,
                }));
            } else {
                i += 1;
            }
        }
        if ctx.deadline.is_some_and(|d| unix_now() >= d) {
            for r in &mut running {
                let _ = r.child.kill();
                let _ = r.child.wait();
            }
            failure.get_or_insert_with(|| "deadline reached; chunk killed".into());
            running.clear();
        }
        if failure.is_some() && running.is_empty() {
            break;
        }
        ctx.heartbeat(false);
        std::thread::sleep(Duration::from_millis(250));
    }
    let recs: Vec<Value> = records.into_iter().flatten().collect();
    let summary = json!({ "usable_threads": cores, "mem_budget_mb": mem, "concurrent": concurrent,
        "wall_s": t0.elapsed().as_secs_f64().round(), "jobs": recs });
    ctx.write_json(&format!("out/{dir}/jobs.json"), &summary)?;
    match failure {
        Some(e) => Err(e),
        None => Ok(recs),
    }
}

/// The `key=value` fields of the last line of `text` starting with `tag` and a tab.
fn last_fields(text: &str, tag: &str) -> Option<Vec<(String, String)>> {
    let line = text
        .lines()
        .rev()
        .find(|l| l.starts_with(tag) && l.as_bytes().get(tag.len()) == Some(&b'\t'))?;
    Some(
        line.split('\t')
            .skip(1)
            .flat_map(|f| f.split(' '))
            .filter_map(|kv| kv.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    )
}

fn field<'a>(f: &'a [(String, String)], k: &str) -> Option<&'a str> {
    f.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str())
}

/// JSON of a `key=value` list: whole numbers and decimals as numbers, the rest as strings.
fn fields_json(f: &[(String, String)]) -> Value {
    let mut m = serde_json::Map::new();
    for (k, v) in f {
        let j = if let Ok(n) = v.parse::<u64>() {
            json!(n)
        } else if let Ok(x) = v.parse::<f64>() {
            json!(x)
        } else {
            json!(v)
        };
        m.insert(k.clone(), j);
    }
    Value::Object(m)
}

/// Plan `trick1`: one world at a time, each on every usable thread (see the module text).
fn trick1(ctx: &mut Ctx, params: &Value, table: &str) -> Result<Value, String> {
    let trick = params.get("trick").and_then(Value::as_u64).unwrap_or(1);
    let worlds = params.get("worlds").and_then(Value::as_u64).unwrap_or(4);
    let cap = params
        .get("cap_secs")
        .and_then(Value::as_u64)
        .unwrap_or(3600);
    let pct = params
        .get("tt_ram_pct")
        .and_then(Value::as_u64)
        .unwrap_or(70)
        .clamp(1, 85);
    let threads = ctx.threads;
    let ram = mem_total_mb();
    let tt_mb = (ram * pct / 100).max(64);
    let dir = ctx.root.join("out").join("trick1");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut recs = Vec::new();
    for w in 0..worlds {
        let name = format!("world-{w}.txt");
        let path = dir.join(&name);
        let mut c = Command::new(sibling("solver-spike"));
        c.args(["--table", table, "--solver", "fast"])
            .args(["--trick", &trick.to_string(), "--seed", "1"])
            .args(["--first", &w.to_string(), "--worlds", "1"])
            .args(["--solve-threads", &threads.to_string()])
            .args(["--tt-mb", &tt_mb.to_string()])
            .args(["--cap-secs", &cap.to_string(), "--progress-secs", "60"]);
        let f = fs::File::create(&path).map_err(|e| e.to_string())?;
        let f2 = f.try_clone().map_err(|e| e.to_string())?;
        c.stdin(Stdio::null()).stdout(f).stderr(f2);
        let mut child = c
            .spawn()
            .map_err(|e| format!("cannot start solver-spike: {e}"))?;
        ctx.child = Some(child.id());
        ctx.cpu.prime_tree(child.id());
        let t0 = Instant::now();
        // The solver stops itself at the cap; this hard stop is a safety net 5 minutes later.
        let hard = Duration::from_secs(cap + 300);
        let mut killed = None;
        let status = loop {
            if let Some(st) = child.try_wait().map_err(|e| e.to_string())? {
                break Some(st);
            }
            if cap > 0 && t0.elapsed() >= hard {
                killed = Some("hard wall cap");
            } else if ctx.deadline.is_some_and(|d| unix_now() >= d) {
                killed = Some("deadline");
            }
            if killed.is_some() {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            let text = fs::read_to_string(&path).unwrap_or_default();
            let prog = last_fields(&text, "progress");
            let num = |k: &str| {
                prog.as_deref().and_then(|p| field(p, k)).and_then(|v| {
                    v.split('/')
                        .map(str::parse::<u64>)
                        .collect::<Result<Vec<_>, _>>()
                        .ok()
                })
            };
            let tasks = num("tasks_done");
            // The last subproblems: fewer left than threads, so some threads are idle.
            let tail = tasks
                .as_deref()
                .is_some_and(|t| t.len() == 2 && t[1] > 0 && t[1] - t[0] < threads as u64);
            ctx.progress = json!({
                "kind": "solver_spike", "step": "trick1", "world": w, "worlds": worlds,
                "elapsed_s": t0.elapsed().as_secs(), "solve_threads": threads, "tt_mb": tt_mb,
                "last_progress": prog.as_deref().map(fields_json),
                "partial_cpu_ok": tail,
            });
            ctx.heartbeat(false);
            std::thread::sleep(Duration::from_millis(500));
        };
        ctx.child = None;
        let wall = t0.elapsed().as_secs_f64();
        let text = fs::read_to_string(&path).unwrap_or_default();
        let res = last_fields(&text, "par_world");
        let timed_out = killed.is_some()
            || res
                .as_deref()
                .and_then(|r| field(r, "timed_out"))
                .is_none_or(|v| v != "0");
        if killed.is_none() && !status.is_some_and(|s| s.success()) {
            return Err(format!(
                "solver-spike failed on world {w} (see out/trick1/{name})"
            ));
        }
        let limit = if cap.is_multiple_of(60) {
            format!("> {} min", cap / 60)
        } else {
            format!("> {cap} s")
        };
        recs.push(json!({
            "world": w, "wall_s": (wall * 10.0).round() / 10.0, "timed_out": timed_out,
            "result": if timed_out { limit.clone() } else { "solved".into() },
            "killed": killed, "solver": res.as_deref().map(fields_json),
        }));
        ctx.write_json(
            "out/trick1/results.json",
            &json!({ "trick": trick, "seed": 1, "solve_threads": threads, "ram_mb": ram,
                "tt_mb": tt_mb, "cap_secs": cap, "worlds": recs }),
        )?;
    }
    Ok(json!({ "trick": trick, "solve_threads": threads, "tt_mb": tt_mb, "worlds": recs }))
}

pub(crate) fn chunk(ctx: &mut Ctx, params: &Value) -> Result<Value, String> {
    let plan = params.get("plan").and_then(Value::as_str).unwrap_or("xeon");
    if plan == "trick1" {
        let (table, equity) = build_table(ctx, params)?;
        let r = trick1(ctx, params, &table.to_string_lossy())?;
        return Ok(json!({ "plan": plan, "equity": equity, "trick1": r }));
    }
    let identity = params
        .get("identity")
        .and_then(Value::as_str)
        .unwrap_or("moderate");
    let concurrent = params
        .get("concurrent")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let runs = identity_runs(identity)?;
    let (table, equity) = build_table(ctx, params)?;
    let table_s = table.to_string_lossy().to_string();
    let jobs = plan_jobs(plan, &table_s)?;
    let spike = run_pool(ctx, &jobs, "spike", concurrent)?;
    let threads = ctx.threads;
    let id_jobs: Vec<Job> = runs
        .iter()
        .map(|&(trick, worlds)| Job {
            name: format!("identity-t{trick}-{worlds}"),
            prog: "solver-identity",
            args: [
                "--leaf",
                &table_s,
                "--trick",
                &trick.to_string(),
                "--worlds",
                &worlds.to_string(),
                "--seed",
                "1",
                "--threads",
                &threads.to_string(),
                "--combos",
                "all",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            threads,
            mem_mb: 600,
        })
        .collect();
    let ident = run_pool(ctx, &id_jobs, "identity", false)?;
    Ok(json!({
        "plan": plan, "identity": identity, "concurrent": concurrent, "equity": equity,
        "spike_jobs": spike.len(), "identity_runs": ident,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_last_tagged_line() {
        let text = "par\ttrick=1\nprogress\tworld=0 nodes=5 tasks_done=3/9\n\
                    progress\tworld=0 nodes=9 tasks_done=7/9\n\
                    par_world\ttrick=1 secs=1.5 value=0.5,0.5 best=13\n";
        let p = last_fields(text, "progress").expect("progress line");
        assert_eq!(field(&p, "tasks_done"), Some("7/9"));
        let r = fields_json(&last_fields(text, "par_world").expect("result line"));
        assert_eq!(r["secs"], json!(1.5));
        assert_eq!(r["best"], json!(13));
        assert_eq!(r["value"], json!("0.5,0.5"));
        assert!(last_fields(text, "par_worlds").is_none());
    }
}
