//! `search`: the M3 PIMC measurement and teacher tools (docs/M3-SPEC.md §5.8–5.10, §13.1).
//!
//! ```text
//! search decisions --trick T --count N --out FILE [--seed S] [--start-hands H] [--allow-uneven]
//! search worlds    --decisions FILE --method exact|rollout --to N --out FILE
//!                  [--from-world A] [--label L] [--seed S] [--first I] [--last J]
//!                  [--ids 3,17,42 | --ids FILE] [--threads K] [--solve-threads K] [--tt-mb M]
//!                  [--schedule fixed|auto] [--tt-budget-mb B] [--sched-json JSON|FILE]
//!                  --equity TABLE [--progress FILE]
//! search bench-t1  --mode fill|prod --decisions FILE --equity TABLE --json OUT --threads T
//!                  [--window-s 240] [--tt-mb 512 (fill)] [--tt-budget-mb B]
//!                  [--worlds-per-decision 800] [--sched-json …] (prod) [--seed 1] [--label 1]
//!                  [--usable-cores C] [--min-cpu-pct 80] [--ramp-s 20] [--busy-streak-s 10] [--min-busy-pct 95]
//!                  [--cpu-interval-s 30] [--no-guard] [--sample-ms 1000]
//!                  [--progress FILE] [--stop-file FILE]
//! search agree     --worlds FILE [--worlds FILE …] [--grid 10,20,40,80,160,320] [--json FILE]
//!                  [--decisions FILE …]
//! search curves    --worlds FILE [--worlds FILE …] [--decisions FILE …]
//!                  [--grid 10,20,…|10:320:10] [--m M]
//!                  [--splits 400] [--boot 1000] [--seed 1] [--ids …]
//!                  [--thresholds 0.0001,0.00005,0.00002] [--min-cell 10] [--threads K]
//!                  [--method exact] [--label 1] [--json FILE]
//! search score     --yardstick FILE --n N --method FILE:N [--method FILE:N …] [--json FILE]
//!                  [--boot 1000] [--seed 1]
//! search teacher   --hands N --out DIR --equity TABLE [--seed S] [--config NAME] [--start-hands H]
//! ```
//! `decisions` stratifies equally over the 4 strata and 8 positions, so `--count` must be a
//! multiple of 32 (`--allow-uneven` overrides); it prints the position × stratum table it made.
//! `worlds` and `teacher` take `--source belief|rejection`: the world generator, by default
//! `crates/belief`'s counted uniform sampler (which also records each decision's exact world
//! count); `rejection` is the engine-based oracle (same distribution, no counts).
//! `worlds` is the yardstick builder and the method runner: every (decision, world) task of the
//! run goes through one work queue on `--threads` threads (longest expected first), and one JSON
//! line per decision is appended to `--out` in that order; decisions already there are skipped,
//! so an interrupted chunk resumes. The file is byte-identical for any thread count; timings go to
//! `<out>.times.jsonl`. `--progress FILE` is rewritten every 2 s with the counts done (for the
//! runner's heartbeat). Exact worlds are the yardstick (label 1); a method's own worlds use
//! another label. `--ids` restricts the run to chosen decisions; with `--from-world A` (A = the
//! worlds already saved) it extends them: the new worlds are appended as new records (to the same
//! file or another), and `agree`, `curves` and `score` join every record of a decision, so they
//! see exactly the rows one longer run would have written (§13.1 item 10).
//! `--schedule auto` gives every exact world its own threads and table by position on `--threads`
//! threads within `--tt-budget-mb` (`search::plan`, §13.1 item 11); the file is the same bytes.
//! `bench-t1` is one timed window of the trick-1 machine bench (`search::t1bench`): `fill` (one
//! world per thread on `--threads` threads, positions spread evenly, `--tt-mb` tables) or `prod`
//! (the position-aware schedule within `--tt-budget-mb`, on every decision × `--worlds-per-decision`
//! worlds). It stops every solve at the window's end, or earlier when the guard sees low CPU or
//! idle threads, or when `--stop-file` appears; it writes every world's nodes (stopped ones so
//! far) and value hash, the samples, CPU time and peak memory to `--json`.
//! `agree` reports N against 2N on the grid, per trick, per position in the trick (§13.1 item 6),
//! per match-state stratum and per log2 world-count band (item 8) and per hand-type band (item 9:
//! trumps, top trumps, longest plain suit; and the hand features added on 2026-10-14, holds the 5
//! of trumps, robber, void plain suits). Worlds records made before those features existed lack
//! them: `--decisions FILE` (the decision set, repeatable) replays each decision from its deck and
//! actions and labels it again, with no new solves; `agree` and `curves` both take it.
//! `curves` turns the saved rows into money-loss curves on a fine N grid, with power-law fits,
//! held-out checks and the N needed per money-loss threshold (item 10, `search::curves`);
//! `score` reports per position and stratum.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use search::bot::{PimcFactory, PimcSpec, load_equity};
use search::curves::{CurveParams, N_CAP, TrickReport, analyse, score_groups};
use search::measure::{
    BatchProgress, DecisionSpec, DecisionTiming, Schedule, WorldRows, WorldsJob, agreement_curve,
    build_decisions, collect_starts, feature_band_name, paired, parse_ids, read_jsonl, read_worlds,
    relabel, score, stratum_of, summarise, timing_path, worlds_file_sched,
};
use search::pimc::{Ctx, Method, SearchConfig, SharedLeaf};
use search::plan::AutoParams;
use search::t1bench::{Guard, Mode, Sample, WindowParams, run_window};
use search::teacher::{decisions_npy, teacher_deck, teacher_hand};
use search::worlds::{BeliefSource, RejectionSource, WorldSource};

struct Args(Vec<String>);

impl Args {
    fn get(&self, k: &str) -> Option<&str> {
        self.0
            .iter()
            .position(|a| a == k)
            .and_then(|i| self.0.get(i + 1))
            .map(String::as_str)
    }
    fn all(&self, k: &str) -> Vec<&str> {
        self.0
            .iter()
            .enumerate()
            .filter(|(_, a)| *a == k)
            .filter_map(|(i, _)| self.0.get(i + 1).map(String::as_str))
            .collect()
    }
    fn num(&self, k: &str, d: u64) -> Result<u64, String> {
        self.get(k).map_or(Ok(d), |v| {
            v.parse().map_err(|_| format!("{k}: not a number: {v}"))
        })
    }
    fn req(&self, k: &str) -> Result<&str, String> {
        self.get(k).ok_or_else(|| format!("missing {k}"))
    }
    fn has(&self, k: &str) -> bool {
        self.0.iter().any(|a| a == k)
    }
}

/// `--source belief|rejection` (default `belief`): the world generator.
fn world_source(a: &Args) -> Result<Arc<dyn WorldSource>, String> {
    match a.get("--source").unwrap_or("belief") {
        "belief" => Ok(Arc::new(BeliefSource::new())),
        "rejection" => Ok(Arc::new(RejectionSource::new())),
        s => Err(format!("unknown --source {s} (belief or rejection)")),
    }
}

fn read_lines<T: serde::de::DeserializeOwned>(path: &str) -> Result<Vec<T>, String> {
    read_jsonl(Path::new(path))
}

fn main() -> ExitCode {
    let a = Args(std::env::args().skip(1).collect());
    let r = match a.0.first().map(String::as_str) {
        Some("decisions") => decisions(&a),
        Some("worlds") => worlds(&a),
        Some("agree") => agree(&a),
        Some("curves") => curves(&a),
        Some("score") => score_cmd(&a),
        Some("teacher") => teacher(&a),
        Some("bench-t1") => bench_t1(&a),
        _ => Err(
            "usage: search decisions|worlds|agree|curves|score|teacher|bench-t1 … (see the source docs)"
                .into(),
        ),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("search: {e}");
            ExitCode::from(2)
        }
    }
}

/// Decisions per stratified block: 4 strata × 8 positions.
const BLOCK: u64 = 32;

fn decisions(a: &Args) -> Result<(), String> {
    let trick = a.num("--trick", 2)? as u8;
    let count = a.num("--count", 7 * BLOCK)?;
    let seed = a.num("--seed", 1)?;
    if count % BLOCK != 0 && !a.has("--allow-uneven") {
        let (lo, hi) = (count / BLOCK * BLOCK, (count / BLOCK + 1) * BLOCK);
        let use_ = if lo == 0 {
            hi.to_string()
        } else {
            format!("{lo} or {hi}")
        };
        return Err(format!(
            "--count {count} is not a multiple of {BLOCK}: decision i is stratum i mod 4 at \
             position (i / 4) mod 8, so positions and strata would not be covered equally \
             (use {use_}, or pass --allow-uneven)"
        ));
    }
    let starts = collect_starts(a.num("--start-hands", 20_000)?, seed);
    let ds = build_decisions(trick, count, seed, &starts);
    let mut cells = [[0usize; 4]; 8];
    for d in &ds {
        cells[usize::from(d.position)][usize::from(d.stratum - 1)] += 1;
    }
    println!(
        "trick {trick}: {} decisions by position (rows) and stratum (columns)",
        ds.len()
    );
    println!("        S1    S2    S3    S4  total");
    for (p, row) in cells.iter().enumerate() {
        println!(
            "pos{}  {:>4}  {:>4}  {:>4}  {:>4}  {:>5}",
            p + 1,
            row[0],
            row[1],
            row[2],
            row[3],
            row.iter().sum::<usize>()
        );
    }
    let mut out = String::new();
    for d in &ds {
        out.push_str(&serde_json::to_string(d).map_err(|e| e.to_string())?);
        out.push('\n');
    }
    std::fs::write(a.req("--out")?, out).map_err(|e| e.to_string())
}

/// Rewrite the progress file (write a temporary file, then rename it over the old one, so a
/// reader never sees half a file).
fn write_progress(
    path: &Path,
    p: &BatchProgress,
    job: &WorldsJob,
    cfg: &SearchConfig,
    t0: Instant,
    finished: bool,
) {
    let ld = |x: &std::sync::atomic::AtomicU64| x.load(Ordering::Relaxed);
    let v = serde_json::json!({
        "method": search::measure::method_name(job.method),
        "label": job.label,
        "from": job.from,
        "to": job.to,
        "threads": ld(&p.threads),
        "solve_threads": cfg.solve_threads,
        "decisions_done": ld(&p.decisions_done),
        "decisions_total": ld(&p.decisions_total),
        "worlds_done": ld(&p.worlds_done),
        "worlds_total": ld(&p.worlds_total),
        "par_tasks_done": ld(&p.par.tasks_done),
        "par_tasks_total": ld(&p.par.tasks_total),
        "planned": p.planned.load(Ordering::Relaxed),
        "running": ld(&p.exec.running),
        "busy_threads": ld(&p.exec.busy_threads),
        "tail": p.exec.tail.load(Ordering::Relaxed),
        "elapsed_s": t0.elapsed().as_secs(),
        "finished": finished,
    });
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    if std::fs::write(&tmp, v.to_string()).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

fn worlds(a: &Args) -> Result<(), String> {
    let ds: Vec<DecisionSpec> = read_lines(a.req("--decisions")?)?;
    let method = match a.req("--method")? {
        "exact" => Method::ExactPimc,
        "rollout" => Method::RolloutPimc,
        m => return Err(format!("unknown method {m}")),
    };
    let t0 = Instant::now();
    let table = load_equity(a.req("--equity")?)?;
    let leaf: Arc<SharedLeaf> = Arc::new(table);
    let src = world_source(a)?;
    let ctx = Ctx {
        leaf: &*leaf,
        source: &*src,
        rules: engine::game::Rules::HOUSE,
    };
    let cfg = SearchConfig {
        threads: a.num("--threads", 1)? as usize,
        solve_threads: a.num("--solve-threads", 1)? as usize,
        tt_mb: a.num("--tt-mb", 256)? as usize,
        ..SearchConfig::default()
    };
    let sched = schedule(a, cfg.threads)?;
    let (from, to) = (a.num("--from-world", 0)?, a.num("--to", 10)?);
    if to < from {
        return Err(format!("--to {to} is below --from-world {from}"));
    }
    let job = WorldsJob {
        method,
        label: a.num("--label", 1)?,
        seed: a.num("--seed", 1)?,
        from,
        to,
    };
    let (first, last) = (a.num("--first", 0)?, a.num("--last", u64::MAX)?);
    let ids = a.get("--ids").map(parse_ids).transpose()?;
    if let Some(ids) = &ids
        && let Some(x) = ids.iter().find(|&&i| !ds.iter().any(|d| d.id == i))
    {
        return Err(format!("--ids: decision {x} is not in the decisions file"));
    }
    let ds: Vec<DecisionSpec> = ds
        .into_iter()
        .filter(|d| d.id >= first && d.id <= last)
        .filter(|d| ids.as_ref().is_none_or(|ids| ids.contains(&d.id)))
        .collect();
    let out = Path::new(a.req("--out")?);
    let progress_path = a.get("--progress").map(PathBuf::from);
    let prog = BatchProgress::default();
    let finished = AtomicBool::new(false);
    match &sched {
        Schedule::Fixed => eprintln!(
            "worlds: {} decisions, worlds {from}..{to}, {} threads, {} solve threads (table loaded in {:.1} s)",
            ds.len(),
            cfg.threads,
            cfg.solve_threads,
            t0.elapsed().as_secs_f64()
        ),
        Schedule::Auto {
            threads, budget_mb, ..
        } => eprintln!(
            "worlds: {} decisions, worlds {from}..{to}, position-aware on {threads} threads with \
             {budget_mb} MB for tables (table loaded in {:.1} s)",
            ds.len(),
            t0.elapsed().as_secs_f64()
        ),
    }
    let n = std::thread::scope(|s| {
        if let Some(p) = &progress_path {
            let (prog, finished, job, cfg) = (&prog, &finished, &job, &cfg);
            s.spawn(move || {
                while !finished.load(Ordering::Relaxed) {
                    write_progress(p, prog, job, cfg, t0, false);
                    for _ in 0..20 {
                        if finished.load(Ordering::Relaxed) {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    }
                }
            });
        }
        let r = worlds_file_sched(out, &ds, &job, &ctx, &cfg, &sched, Some(&prog), |r, t| {
            eprintln!(
                "decision {} trick {} pos {} cands {}: {:.1} ms per world (slowest {:.1} ms); {}/{} decisions, {:.0} s",
                r.id,
                r.trick,
                r.position + 1,
                r.cands.len(),
                t.world_us as f64 / 1000.0 / t.worlds.max(1) as f64,
                t.max_world_us as f64 / 1000.0,
                prog.decisions_done.load(Ordering::Relaxed) + 1,
                prog.decisions_total.load(Ordering::Relaxed),
                t0.elapsed().as_secs_f64()
            );
        });
        finished.store(true, Ordering::Relaxed);
        r
    })?;
    if let Some(p) = &progress_path {
        write_progress(p, &prog, &job, &cfg, t0, true);
    }
    eprintln!(
        "worlds: {n} decisions valued in {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// `--schedule fixed|auto` (default fixed): with `auto`, every exact world gets its own threads
/// and table by position ([`search::plan`]) on `--threads` threads in all, with
/// `--tt-budget-mb` MB for all tables at their peak (default: 16384) and the plan's knobs from
/// `--sched-json` (a JSON object of `AutoParams` fields, or a file holding one).
fn schedule(a: &Args, threads: usize) -> Result<Schedule, String> {
    match a.get("--schedule").unwrap_or("fixed") {
        "fixed" => Ok(Schedule::Fixed),
        "auto" => Ok(Schedule::Auto {
            threads: threads.max(1),
            budget_mb: a.num("--tt-budget-mb", 16_384)?,
            params: Box::new(auto_params(a)?),
        }),
        s => Err(format!("unknown --schedule {s} (fixed or auto)")),
    }
}

/// `--sched-json`: the position-aware plan's knobs, inline JSON or a file.
fn auto_params(a: &Args) -> Result<AutoParams, String> {
    match a.get("--sched-json") {
        None => Ok(AutoParams::default()),
        Some(s) => {
            let text = if s.trim_start().starts_with('{') {
                s.to_string()
            } else {
                std::fs::read_to_string(s).map_err(|e| format!("--sched-json {s}: {e}"))?
            };
            serde_json::from_str(&text).map_err(|e| format!("--sched-json: {e}"))
        }
    }
}

/// Write `v` to `path` whole (a temporary file renamed over it, so a reader never sees half).
fn write_atomic(path: &Path, v: &serde_json::Value) {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    if std::fs::write(&tmp, v.to_string()).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

fn float(a: &Args, k: &str, d: f64) -> Result<f64, String> {
    a.get(k).map_or(Ok(d), |s| {
        s.parse::<f64>()
            .map_err(|_| format!("{k}: not a number: {s}"))
    })
}

/// `search bench-t1`: one window of the trick-1 machine bench (`search::t1bench`), written as
/// JSON to `--json`. `--progress FILE` is rewritten every sample (about every second) with the
/// window's state; a `--stop-file` appearing stops the window at once with the file's text as the
/// reason.
fn bench_t1(a: &Args) -> Result<(), String> {
    let mode = Mode::parse(a.req("--mode")?)?;
    let ds: Vec<DecisionSpec> = read_lines(a.req("--decisions")?)?;
    let table = load_equity(a.req("--equity")?)?;
    let leaf: Arc<SharedLeaf> = Arc::new(table);
    let src = world_source(a)?;
    let ctx = Ctx {
        leaf: &*leaf,
        source: &*src,
        rules: engine::game::Rules::HOUSE,
    };
    let threads = a.num("--threads", 1)?.max(1) as usize;
    let mut p = WindowParams::new(mode, float(a, "--window-s", 240.0)?, threads);
    p.seed = a.num("--seed", 1)?;
    p.label = a.num("--label", 1)?;
    p.tt_mb = a.num("--tt-mb", 512)? as usize;
    p.budget_mb = a.num("--tt-budget-mb", 16_384)?;
    p.auto = auto_params(a)?;
    p.worlds = a.num("--worlds-per-decision", 800)?;
    p.sample_ms = a.num("--sample-ms", 1000)?;
    let d = Guard::default();
    let guard = Guard {
        ramp_s: float(a, "--ramp-s", d.ramp_s)?,
        min_cpu_pct: float(a, "--min-cpu-pct", d.min_cpu_pct)?,
        cpu_interval_s: float(a, "--cpu-interval-s", d.cpu_interval_s)?,
        busy_streak_s: float(a, "--busy-streak-s", d.busy_streak_s)?,
        min_busy_pct: float(a, "--min-busy-pct", d.min_busy_pct)?,
        usable_cores: float(a, "--usable-cores", d.usable_cores)?,
    };
    // `--no-guard` keeps the usable cores (for the CPU share) but never stops the window.
    p.guard = Some(if a.has("--no-guard") {
        Guard {
            ramp_s: f64::INFINITY,
            ..guard
        }
    } else {
        guard
    });
    let progress_path = a.get("--progress").map(PathBuf::from);
    let stop_file = a.get("--stop-file").map(PathBuf::from);
    if let Some(f) = &stop_file {
        let _ = std::fs::remove_file(f);
    }
    let window_s = p.window_s;
    let state = |s: Option<&Sample>, started: bool, stopped: Option<&str>| {
        let (t, nodes) = s.map_or((0.0, 0), |s| (s.t, s.nodes));
        serde_json::json!({
            "window": mode.name(), "started": started, "window_s": window_s,
            "t_s": (t * 10.0).round() / 10.0, "threads": threads,
            "busy": s.map(|s| s.busy), "held": s.map(|s| s.held), "nodes": nodes,
            "finished_worlds": s.map(|s| s.finished),
            "rate": if t > 0.0 { (nodes as f64 / t).round() } else { 0.0 },
            "cpu_s": s.map(|s| (s.cpu_s * 10.0).round() / 10.0),
            "stopped": stopped,
        })
    };
    if let Some(path) = &progress_path {
        write_atomic(path, &state(None, false, None));
    }
    eprintln!(
        "bench-t1 {}: {} decisions, {threads} threads, {window_s} s",
        mode.name(),
        ds.len()
    );
    let mut on_sample = |s: &Sample| {
        if let Some(path) = &progress_path {
            write_atomic(path, &state(Some(s), true, None));
        }
        let f = stop_file.as_ref()?;
        let why = std::fs::read_to_string(f).ok()?;
        Some(match why.trim() {
            "" => "stopped by the runner".to_string(),
            w => w.to_string(),
        })
    };
    let run = run_window(&ds, &ctx, &p, &mut on_sample)?;
    if let Some(path) = &progress_path {
        write_atomic(
            path,
            &state(run.samples.last(), true, run.stopped.as_deref())
                .as_object()
                .cloned()
                .map(|mut o| {
                    o.insert("finished".into(), serde_json::json!(true));
                    serde_json::Value::Object(o)
                })
                .unwrap_or_default(),
        );
    }
    eprintln!(
        "bench-t1 {}: {} nodes in {:.1} s ({:.3e} a second), {} of {} worlds finished{}",
        mode.name(),
        run.nodes(),
        run.wall_s,
        run.nodes() as f64 / run.wall_s.max(1e-9),
        run.worlds.iter().filter(|w| w.finished).count(),
        run.worlds.len(),
        run.stopped
            .as_deref()
            .map_or(String::new(), |s| format!("; stopped early: {s}"))
    );
    let text = serde_json::to_string(&run).map_err(|e| e.to_string())?;
    std::fs::write(a.req("--json")?, text + "\n").map_err(|e| e.to_string())
}

/// `--grid`: a list (`10,20,40`) or a range `start:end:step` (`10:320:10`).
fn parse_grid(s: &str) -> Result<Vec<usize>, String> {
    let bad = || format!("bad grid {s}");
    if s.contains(':') {
        let p: Vec<usize> = s
            .split(':')
            .map(|x| x.trim().parse().map_err(|_| bad()))
            .collect::<Result<_, _>>()?;
        let [a, b, step] = p[..] else {
            return Err(bad());
        };
        if step == 0 || a == 0 {
            return Err(bad());
        }
        return Ok((a..=b).step_by(step).collect());
    }
    s.split(',')
        .map(|x| x.trim().parse().map_err(|_| bad()))
        .collect()
}

fn grid(a: &Args) -> Result<Vec<usize>, String> {
    parse_grid(a.get("--grid").unwrap_or("10,20,40,80,160,320"))
}

/// Every `--worlds` file, merged (extension records joined to the earlier worlds), restricted to
/// `--ids` when given, with hand types recomputed from the `--decisions` files when given.
fn worlds_args(a: &Args) -> Result<Vec<WorldRows>, String> {
    let paths: Vec<&Path> = a.all("--worlds").into_iter().map(Path::new).collect();
    if paths.is_empty() {
        return Err("missing --worlds".into());
    }
    let (rows, warnings) = read_worlds(&paths)?;
    for w in warnings {
        eprintln!("warning: {w}");
    }
    let ids = a.get("--ids").map(parse_ids).transpose()?;
    let mut rows: Vec<WorldRows> = rows
        .into_iter()
        .filter(|r| ids.as_ref().is_none_or(|ids| ids.contains(&r.id)))
        .collect();
    // Hand types recomputed from the decision sets (the 2026-10-14 features for old records).
    let decision_files = a.all("--decisions");
    if !decision_files.is_empty() {
        let mut ds: Vec<DecisionSpec> = Vec::new();
        for f in &decision_files {
            ds.extend(read_lines::<DecisionSpec>(f)?);
        }
        let n = relabel(&mut rows, &ds)?;
        eprintln!(
            "hand types recomputed for {n} of {} records from {}",
            rows.len(),
            decision_files.join(", ")
        );
    }
    if rows
        .iter()
        .any(|r| r.hand_type.is_some_and(|h| h.feature_bands().is_none()))
    {
        eprintln!(
            "note: some records predate the hand features (holds the 5 of trumps, robber, voids); \
             pass --decisions FILE (their decision set) to compute them"
        );
    }
    Ok(rows)
}

/// The fewest decisions a reported cell should have (§13.1 item 8).
const MIN_CELL: usize = 25;

fn agree(a: &Args) -> Result<(), String> {
    let rows = worlds_args(a)?;
    // Timings, when the timing files are there: per decision, summed over its records (an
    // extension adds a record).
    let mut times: BTreeMap<u64, DecisionTiming> = BTreeMap::new();
    for path in a.all("--worlds") {
        if let Ok(ts) = read_jsonl::<DecisionTiming>(&timing_path(Path::new(path))) {
            for t in ts {
                times
                    .entry(t.id)
                    .and_modify(|x| {
                        x.worlds += t.worlds;
                        x.world_us += t.world_us;
                        x.max_world_us = x.max_world_us.max(t.max_world_us);
                    })
                    .or_insert(t);
            }
        }
    }
    let g = grid(a)?;
    // Groups by (trick, kind, index): all, position, position pair, stratum, log2 world-count band.
    let mut groups: BTreeMap<(u8, u8, u32), Vec<&WorldRows>> = BTreeMap::new();
    for r in &rows {
        let p = u32::from(r.position);
        let mut keys = vec![
            (0, 0),
            (1, p + 1),
            (2, p / 2 * 2 + 1),
            (3, u32::from(r.stratum)),
        ];
        if let Some(b) = r.log2_worlds {
            keys.push((4, b));
        }
        if let Some(h) = r.hand_type {
            let [t, top, plain] = h.bands();
            keys.extend([
                (5, u32::from(t)),
                (6, u32::from(top)),
                (7, u32::from(plain)),
            ]);
            if let Some(f) = h.feature_bands() {
                keys.extend((0..3).map(|j| (8 + j as u8, u32::from(f[j]))));
            }
        }
        for (kind, i) in keys {
            groups.entry((r.trick, kind, i)).or_default().push(r);
        }
    }
    let name = |kind: u8, i: u32| match kind {
        0 => "all".to_string(),
        1 => format!("pos{i}"),
        2 => format!("pair{i}-{}", i + 1),
        3 => format!("S{i}"),
        4 => format!("log2worlds{i}"),
        5 => format!("trumps{i}{}", if i == 3 { "+" } else { "" }),
        6 => format!("top{i}{}", if i == 2 { "+" } else { "" }),
        7 => match i {
            1 => "plain0-1".to_string(),
            3 => "plain3+".to_string(),
            _ => format!("plain{i}"),
        },
        _ => feature_band_name(usize::from(kind - 8), i as u8),
    };
    let mut report = serde_json::Map::new();
    let mut last_kind = None;
    for (&(trick, kind, i), rs) in &groups {
        if last_kind != Some((trick, kind)) {
            last_kind = Some((trick, kind));
            println!(
                "== trick {trick}, by {} ==",
                [
                    "all",
                    "position",
                    "position pair",
                    "stratum",
                    "log2 world count",
                    "hand type: trumps held",
                    "hand type: top trumps held (5, J, Joker, A♥, ace of trumps)",
                    "hand type: longest plain suit",
                    "hand type: holds the 5 of trumps",
                    "hand type: robber (took a turned card this hand)",
                    "hand type: void plain suits",
                ][usize::from(kind)]
            );
        }
        let group = name(kind, i);
        let owned: Vec<WorldRows> = rs.iter().map(|r| (*r).clone()).collect();
        let (steps, plateau) = agreement_curve(&owned, &g);
        let timed: Vec<f64> = rs
            .iter()
            .filter_map(|r| times.get(&r.id))
            .map(|t| t.world_us as f64 / 1000.0 / t.worlds.max(1) as f64)
            .collect();
        let ms = (!timed.is_empty()).then(|| timed.iter().sum::<f64>() / timed.len() as f64);
        let enough = rs.len() >= MIN_CELL;
        println!(
            "trick {trick} {group}: {} decisions{}, {} per world, plateau N = {}",
            rs.len(),
            if enough {
                String::new()
            } else {
                format!(" (under {MIN_CELL}: too few to set N)")
            },
            ms.map_or("n/a ms".to_string(), |m| format!("{m:.1} ms")),
            plateau.map_or("none on this grid".to_string(), |n| n.to_string())
        );
        println!("   N→2N  change%  |Δmean|   loss(±SE)");
        for s in &steps {
            println!(
                "  {:>3}→{:<3} {:>6.2}  {:.5}  {:.5}±{:.5} {}",
                s.n,
                s.n2,
                100.0 * s.change_rate,
                s.mean_value_change,
                s.mean_loss,
                s.loss_se,
                if s.plateau_ok { "ok" } else { "" }
            );
        }
        report.insert(
            format!("trick{trick}-{group}"),
            serde_json::json!({ "decisions": rs.len(), "enough_decisions": enough,
                                "ms_per_world": ms, "plateau": plateau, "steps": steps }),
        );
    }
    let tricks: std::collections::BTreeSet<u8> = rows.iter().map(|r| r.trick).collect();
    for t in tricks {
        if !rows.iter().any(|r| r.trick == t && r.log2_worlds.is_some()) {
            println!(
                "trick {t} by log2 world count: n/a (the world source cannot count worlds; use --source belief)"
            );
            report.insert(format!("trick{t}-log2worlds"), serde_json::json!("n/a"));
        }
    }
    if let Some(p) = a.get("--json") {
        std::fs::write(
            p,
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// A money chance as a percentage for the text report.
fn pct(x: f64) -> String {
    if x.is_finite() {
        format!("{:.4}%", 100.0 * x)
    } else {
        "n/a".into()
    }
}

/// A predicted N for the text report.
fn n_text(x: f64) -> String {
    if !x.is_finite() {
        "n/a".into()
    } else if x >= N_CAP * 0.999 {
        "never".into()
    } else if x > 1e6 {
        format!(">{:.0e}", 1e6)
    } else {
        format!("{x:.0}")
    }
}

fn print_trick(t: &TrickReport) {
    println!(
        "== trick {}: {} decisions ({} with fewer than {} worlds left out), M = {} worlds, \
         reference K = {}, fit on N <= {}; {} splits, {} bootstrap replicates, seed {}",
        t.trick,
        t.decisions,
        t.dropped,
        t.worlds,
        t.worlds,
        t.k,
        t.fit_max,
        t.splits,
        t.boots,
        t.seed
    );
    match &t.b {
        Some(b) => println!(
            "shared exponent b = {:.3} [95%: {:.3}, {:.3}]; half-data fit b = {}",
            b.v,
            b.lo,
            b.hi,
            t.b_half.map_or("n/a".into(), |x| format!(
                "{:.3} [{:.3}, {:.3}]",
                x.v, x.lo, x.hi
            ))
        ),
        None => println!("shared exponent b: no fit (too few points)"),
    }
    println!(
        "b fitted on N up to: {}",
        t.b_by_range
            .iter()
            .map(|(n, b)| format!("{n}: {b:.3}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "held-out check ({} < N < {}, all + positions): {}/{} predictions inside the measured 95% \
         band; mean predicted/measured = {:.2}",
        t.fit_max,
        t.k,
        t.heldout_inside.0,
        t.heldout_inside.1,
        t.heldout_log_ratio.exp()
    );
    println!(
        "extrapolation check (first {} worlds predict the {}-world curve, N < {}): {}/{} inside; \
         mean predicted/measured = {:.2}",
        t.worlds / 2,
        t.worlds,
        t.k,
        t.half_inside.0,
        t.half_inside.1,
        t.half_log_ratio.exp()
    );
    let at = |g: &search::curves::GroupReport, n: usize| {
        t.grid
            .iter()
            .position(|&x| x == n)
            .and_then(|i| g.cv.mean.get(i).copied())
            .map_or("-".into(), pct)
    };
    let quarter = t.fit_max / 2;
    print!(
        "{:<22} {:>4} {:>10} {:>10} {:>10} {:>11} {:>22}",
        "group",
        "dec",
        format!("cv({})", t.grid.first().copied().unwrap_or(0)),
        format!("cv({quarter})"),
        format!("cv({})", t.fit_max),
        "a",
        "free b [95%]"
    );
    for th in &t.thresholds {
        print!(" {:>24}", format!("N({}) [95%]", pct(*th)));
    }
    println!();
    for g in &t.groups {
        print!(
            "{:<22} {:>4} {:>10} {:>10} {:>10} {:>11} {:>22}",
            format!("{}{}", g.name, if g.enough { "" } else { "*" }),
            g.decisions,
            at(g, t.grid.first().copied().unwrap_or(0)),
            at(g, quarter),
            at(g, t.fit_max),
            g.a.map_or("n/a".into(), |a| format!("{:.3e}", a.v)),
            g.free_b.map_or(String::new(), |b| format!(
                "{:.2} [{:.2}, {:.2}]",
                b.v, b.lo, b.hi
            ))
        );
        for e in &g.n_for {
            print!(
                " {:>24}",
                format!("{} [{}, {}]", n_text(e.v), n_text(e.lo), n_text(e.hi))
            );
        }
        println!();
    }
    println!(
        "(* fewer than {} decisions; cv(N) = measured L(N) - L({}), money chance)",
        search::curves::ENOUGH,
        t.k
    );
    if !t.contrasts.is_empty() {
        println!(
            "hand features, N({}) of one group over another [95%, paired bootstrap]: {}",
            pct(t.thresholds.first().copied().unwrap_or(f64::NAN)),
            t.contrasts
                .iter()
                .map(|c| format!(
                    "{}/{} {:.2} [{:.2}, {:.2}]",
                    c.group, c.vs, c.n_ratio.v, c.n_ratio.lo, c.n_ratio.hi
                ))
                .collect::<Vec<_>>()
                .join("; ")
        );
    }
}

fn curves(a: &Args) -> Result<(), String> {
    let t0 = Instant::now();
    let method = a.get("--method").unwrap_or("exact");
    let label = a.num("--label", 1)?;
    let rows: Vec<WorldRows> = worlds_args(a)?
        .into_iter()
        .filter(|r| r.method == method && r.label == label)
        .collect();
    if rows.is_empty() {
        return Err(format!(
            "no {method} label {label} records in the --worlds files"
        ));
    }
    let d = CurveParams::default();
    let p = CurveParams {
        grid: a
            .get("--grid")
            .map_or(Ok((1..=1000).map(|i| 10 * i).collect()), parse_grid)?,
        worlds: a
            .get("--m")
            .map(|_| a.num("--m", 0))
            .transpose()?
            .map(|m| m as usize),
        splits: a.num("--splits", d.splits)?,
        boots: a.num("--boot", d.boots)?,
        seed: a.num("--seed", d.seed)?,
        thresholds: match a.get("--thresholds") {
            Some(s) => s
                .split(',')
                .map(|x| x.trim().parse().map_err(|_| format!("bad threshold {x}")))
                .collect::<Result<_, _>>()?,
            None => d.thresholds,
        },
        min_cell: a.num("--min-cell", d.min_cell as u64)? as usize,
        threads: a.num(
            "--threads",
            std::thread::available_parallelism().map_or(1, |n| n.get()) as u64,
        )? as usize,
    };
    let reports = analyse(&rows, &p);
    for t in &reports {
        print_trick(t);
    }
    if let Some(path) = a.get("--json") {
        let v = serde_json::json!({
            "method": method, "label": label,
            "files": a.all("--worlds"),
            "tricks": reports,
        });
        std::fs::write(path, serde_json::to_string(&v).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    }
    eprintln!("curves: done in {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
}

fn score_cmd(a: &Args) -> Result<(), String> {
    let (yard, _) = read_worlds(&[Path::new(a.req("--yardstick")?)])?;
    let n = a.num("--n", 80)? as usize;
    let mut scored = Vec::new();
    let mut reports = Vec::new();
    for m in a.all("--method") {
        let (path, nm) = m.rsplit_once(':').ok_or("--method FILE:N")?;
        let nm: usize = nm.parse().map_err(|_| "--method FILE:N")?;
        let (rows, _) = read_worlds(&[Path::new(path)])?;
        let s = score(&yard, n, &rows, nm);
        let sum = summarise(&s);
        println!(
            "{m}: {}",
            serde_json::to_string(&sum).map_err(|e| e.to_string())?
        );
        // Per position in the trick, then per stratum.
        let mut cells: BTreeMap<(u8, u8), Vec<search::measure::Scored>> = BTreeMap::new();
        for x in &s {
            if let Some(y) = yard.iter().find(|y| y.id == x.id) {
                cells
                    .entry((0, y.position + 1))
                    .or_default()
                    .push(x.clone());
                cells.entry((1, y.stratum)).or_default().push(x.clone());
            }
        }
        for (&(kind, i), v) in &cells {
            let ps = summarise(v);
            println!(
                "  {} {i}: n={} agree={:.3} loss={:.5}±{:.5}",
                if kind == 0 { "pos" } else { "S" },
                ps.decisions,
                ps.agreement.0,
                ps.mean_loss.0,
                ps.mean_loss.1
            );
        }
        if a.get("--json").is_some() {
            reports.push(score_groups(
                m,
                &yard,
                n,
                &rows,
                nm,
                a.num("--boot", 1000)?,
                a.num("--seed", 1)?,
            ));
        }
        scored.push((m.to_string(), s));
    }
    let mut pairs = Vec::new();
    for i in 0..scored.len() {
        for j in i + 1..scored.len() {
            let p = paired(&scored[i].1, &scored[j].1);
            println!(
                "{} vs {}: {}",
                scored[i].0,
                scored[j].0,
                serde_json::to_string(&p).map_err(|e| e.to_string())?
            );
            pairs.push(serde_json::json!({ "a": scored[i].0, "b": scored[j].0, "paired": p }));
        }
    }
    if let Some(path) = a.get("--json") {
        let v = serde_json::json!({ "yardstick": a.req("--yardstick")?, "n_ref": n,
                                    "methods": reports, "paired": pairs });
        std::fs::write(path, serde_json::to_string(&v).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn teacher(a: &Args) -> Result<(), String> {
    let hands = a.num("--hands", 10)?;
    let seed = a.num("--seed", 1)?;
    let spec = PimcSpec::parse(a.get("--config").unwrap_or("pimc-teacher"))?;
    let table = load_equity(a.req("--equity")?)?;
    let leaf: Arc<SharedLeaf> = Arc::new(table);
    let f = PimcFactory::new(&spec, leaf, world_source(a)?);
    let starts = collect_starts(a.num("--start-hands", 20_000)?, seed);
    let dir = std::path::Path::new(a.req("--out")?);
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let commit = option_env!("GIT_COMMIT").unwrap_or("unknown");
    let mut lines = String::new();
    let mut rows = Vec::new();
    for i in 0..hands {
        let st = &starts[(i % 4) as usize];
        let mut rng = engine::rng::Rng::derive(seed, &[search::labels::STRATUM, i]);
        let start = st[rng.bounded(st.len() as u64) as usize];
        debug_assert_eq!(u64::from(stratum_of(&start)), i % 4 + 1);
        let (line, ds) = teacher_hand(i, start, teacher_deck(seed, i), &f, seed ^ i, commit);
        lines.push_str(&line.to_string());
        lines.push('\n');
        rows.extend(ds);
    }
    let npy = decisions_npy(&rows);
    let h = |b: &[u8]| equity::sha256::sha256_hex(b)[..16].to_string();
    std::fs::write(
        dir.join(format!("hands-{}.jsonl", h(lines.as_bytes()))),
        &lines,
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(dir.join(format!("decisions-{}.npy", h(&npy))), &npy)
        .map_err(|e| e.to_string())?;
    println!("{hands} hands, {} decisions", rows.len());
    Ok(())
}
