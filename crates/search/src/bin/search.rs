//! `search`: the M3 PIMC measurement and teacher tools (docs/M3-SPEC.md §5.8–5.10, §13.1).
//!
//! ```text
//! search decisions --trick T --count N --out FILE [--seed S] [--start-hands H] [--allow-uneven]
//!                  [--threads T]
//! search worlds    --decisions FILE --method exact|rollout --to N --out FILE
//!                  [--from-world A] [--label L] [--seed S] [--first I] [--last J]
//!                  [--ids 3,17,42 | --ids FILE] [--threads K] [--solve-threads K] [--tt-mb M]
//!                  [--schedule fixed|auto] [--tt-budget-mb B] [--sched-json JSON|FILE]
//!                  --equity TABLE [--progress FILE]
//! search robber-burn [--trick 1] [--count 32] [--seed 1] [--start-hands 20000] [--position 0]
//!                  [--share-worlds 1000] [--worlds 4] [--equity TABLE] [--tt-mb 512] [--label 1]
//!                  [--source belief|rejection]
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
//! search labels    --trick K --count N --seed S --source random|rules|harvest --out FILE.npy
//!                  --equity TABLE [--threads T] [--start-hands 20000] [--chunk-rows 32]
//!                  harvest only: [--harvest-cap 16] [--harvest-positions 0.25,0.25,0.5]
//!                  [--harvest-tricks K-1 | --harvest-tricks 2,3] [--harvest-net WEIGHTS]
//!                  [--harvest-strata 1,2,3,4] [--harvest-race-classes 2,4 [--harvest-race-cap 256]]
//! search hybrid-check --trick T --cut K --net WEIGHTS --equity TABLE
//!                  [--decisions FILE | --count N] [--seed 1] [--start-hands 20000]
//!                  [--first I] [--last J] [--ids …] [--to 320] [--label 1]
//!                  [--worlds FILE …] [--verify-pct 1] [--recompute] [--threads T] [--tt-mb 64]
//!                  [--boot 1000] [--boot-seed 1] [--json OUT] [--source belief|rejection]
//!                  [--verify-solve-threads 1] [--verify-tt-mb (--tt-mb)]
//!                  [--explain 143,335 [--explain-worlds 3] [--explain-json FILE]]
//! search hybrid-check --trick T --cut K --net WEIGHTS --equity TABLE --decisions FILE
//!                  --saved RUN=FILE [--saved RUN=FILE …] [--only D:W,…] [--hard D:W,…]
//!                  [--verify-pct 0] [--verify-rule all|rules-v1|searched]
//!                  [--verify-solve-threads 1] [--verify-tt-mb M] [--threads T] [--tt-mb 64]
//!                  [--boot 1000] [--boot-seed 1] [--json OUT] [--progress FILE] [--plan-only]
//! ```
//! `labels` writes training labels for the oracle network O_K (docs/M3-HYBRID-SPEC.md §5,
//! `search::oracle_labels`): a label shard (`nn::shard`, `.npy` records: encoded features, the exact
//! value as four u64, the source tag, the trick) and a JSON sidecar `FILE.npy.json` with the
//! encoder, values version, tie rule, this binary's SHA-256, the equity table's SHA-256, the seed,
//! counts and the shard's SHA-256. Byte-identical for any `--threads` and `--chunk-rows`; seed 1
//! (the golden sets') is refused. The shard is streamed
//! (`search::oracle_labels::write_label_shard`): chunks of `--chunk-rows` rows are made on every
//! thread, then hashed and written in row order as they come, under the name `FILE.npy.part`,
//! renamed to `FILE.npy` once whole. The start states (`--start-hands`; `decisions` and
//! `hybrid-check` too) are played on `--threads` threads, the same states for any count
//! (`search::measure::collect_starts_with`). `--source harvest` (spec §5 source (c),
//! `search::oracle_labels::harvest_rows`): training decisions drawn by the yardstick's generator at the tricks `--harvest-tricks`
//! (default K−1), a world from the decider's belief each, the frontier positions the hybrid's
//! search from the decision reaches (exact values at the frontier by default), deduplicated by
//! canonical form, at most `--harvest-cap` per decision (a seeded sample); `--harvest-positions` gives the
//! shares of decisions for the decider's positions 1–2, 3–4 and 5–8 (default uniform over the
//! eight positions). `--harvest-net WEIGHTS` (an integer O_K of an earlier round, as
//! `hybrid-check` takes it) drives the search with that network's values at the frontier instead
//! of exact ones, and only the sampled positions are solved exactly, for their labels: the way to
//! harvest at trick 3, where solving every reached position is too dear. The sidecar adds the
//! harvest's settings, the driver (and the network's SHA-256) and counts per position, stratum
//! and decision trick. `--harvest-strata 4` (or `2,4`) draws the decisions from those strata
//! only: decision `j` from stratum `strata[j mod s]`, its decider at `schedule[(j div s) mod
//! cycle]` (`s` strata; all four, the default, is the yardstick's `j mod 4 + 1`), so the rows stay
//! a pure function of (seed, `j`) and the settings. `--harvest-race-classes 2,4` makes a race
//! harvest (docs/m3-hybrid/rung1.md, "Why seats 3-4 fail"): the reached positions of those race
//! classes (`nn::outcome::race_class`: 0 no game can end, 1 one team can win the game, 2 several
//! can, 3 one team can win the match, 4 a team can win the match and another the game) are a
//! seeded sample of at most `--harvest-race-cap` (256) per decision, the other classes' of at most
//! `--harvest-cap` (which may then be 0: race positions only). Every shard's sidecar counts its
//! rows by race class (`rows_by_race_class`); a harvest's records its strata and race settings.
//! `hybrid-check` measures the hybrid H_K (exact to the start of trick K, the oracle network
//! there) against exact search on the same worlds (docs/M3-HYBRID-SPEC.md §3, `search::hybrid`):
//! decision loss with its bootstrap 95% upper bound, value error and agreement, overall, per
//! position band and per position. Decisions come from `--decisions` (those at trick T) or are
//! built like `search decisions` (`--count`, `--seed`); worlds are regenerated as the yardstick
//! draws them (`--seed`, `--label`, worlds `0..--to`). Exact values are recomputed, or read from
//! `--worlds` files (a `--verify-pct` sample recomputed first and required byte-identical;
//! `--recompute` ignores them). Worlds files made before the belief sampler became the default
//! (no world counts, e.g. the trick-4 golden set `m3-agree-t4c`) were drawn by the rejection
//! sampler: check them with `--source rejection`.
//! With `--saved` (docs/M3-HYBRID-SPEC.md §3 "Trick-1 checks", `search::saved`) the worlds are
//! those earlier runs saved exact values for (worlds files or their `.ckpt` checkpoints, each
//! labelled by its run), split by the burn rule that valued them (values version 1: the
//! `rules-v1` burn, checked with `rules_robber_burn`; the current version: searched); the sets run
//! at once on `--threads` shared out by worlds, one world per thread, the largest saved first;
//! each world's line goes to stderr as it finishes. `--verify-pct` (default 0 here) recomputes
//! that share of the saved worlds first (`--verify-rule` picks the sets), each solve on
//! `--verify-solve-threads` threads with a `--verify-tt-mb` table, and a set with any difference
//! is not compared. The report: per world, per decision, the §3 tables per set, the `--hard`
//! worlds with their saved exact times, the peak memory; `--plan-only` prints the sets and the
//! verification sample and solves nothing.
//! `--explain IDS` (not with `--saved`) runs, instead of the check, the explanation of those
//! decisions (`search::hybrid_explain`): exact means in both tie orders and hybrid means per
//! candidate, how the hybrid's misranking spreads over the worlds and over race classes, the
//! network's error at every frontier position the hybrid reached (per race class, tie-decided
//! or not), and for the `--explain-worlds` most misleading worlds the frontier positions each
//! search copied up to each candidate, with the network's outcome probabilities there;
//! `--explain-json FILE` writes every world (the principal positions of the
//! `--explain-worlds` most misleading ones only).
//! `decisions` stratifies equally over the 4 strata and 8 positions, so `--count` must be a
//! multiple of 32 (`--allow-uneven` overrides); it prints the position × stratum table it made.
//! `worlds` and `teacher` take `--source belief|rejection`: the world generator, by default
//! `crates/belief`'s counted uniform sampler (which also records each decision's exact world
//! count); `rejection` is the engine-based oracle (same distribution, no counts).
//! `worlds` is the yardstick builder and the method runner: every (decision, world) task of the
//! run goes through one work queue on `--threads` threads (longest expected first), and one JSON
//! line per decision is appended to `--out` in that order; each world is also appended to the
//! checkpoint `<out>.ckpt` as soon as it finishes. A rerun skips the decisions already in `--out`
//! and the worlds already in the checkpoint, so an interrupted chunk loses only the worlds that
//! were being solved; the checkpoint is deleted once every decision is in `--out`. The file is
//! byte-identical for any thread count and any interruptions; timings go to
//! `<out>.times.jsonl`. `--progress FILE` is rewritten every 2 s with the counts done, the worlds
//! restored at the start, and the progress estimate (`eta`: work done weighted by each world's
//! expected cost, refined by measured times, and the seconds left; `search::eta`) for the
//! runner's heartbeat. Exact worlds are the yardstick (label 1); a method's own worlds use
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

use search::bot::{PimcFactory, PimcSpec, load_equity, load_equity_sha};
use search::curves::{CurveParams, N_CAP, TrickReport, analyse, score_groups};
use search::measure::{
    BatchProgress, DecisionSpec, DecisionTiming, Schedule, WorldRows, WorldsJob, agreement_curve,
    build_decisions, collect_starts, collect_starts_with, feature_band_name, paired, parse_ids,
    read_jsonl, read_worlds, relabel, score, stratum_of, summarise, timing_path, worlds_file_sched,
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
        Some("robber-burn") => robber_burn(&a),
        Some("labels") => labels_cmd(&a),
        Some("hybrid-check") => hybrid_check(&a),
        _ => Err(
            "usage: search decisions|worlds|agree|curves|score|teacher|bench-t1|robber-burn|labels|hybrid-check … (see the source docs)"
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
    let threads = a.num("--threads", 1)?.max(1) as usize;
    let starts = collect_starts_with(a.num("--start-hands", 20_000)?, seed, threads);
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
        "decisions_resumed": ld(&p.decisions_resumed),
        "worlds_resumed": ld(&p.worlds_resumed),
        "worlds_valued": ld(&p.worlds_valued),
        "nodes_valued": ld(&p.nodes_valued),
        "eta": p.eta().map(|e| serde_json::json!({
            "done_frac": e.done_frac, "remaining_s": e.remaining_s, "measured": e.measured,
            "basis": e.basis, "work_thread_s": e.work_s, "done_thread_s": e.done_s,
        })),
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
    let (table, table_sha) = load_equity_sha(a.req("--equity")?)?;
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
        table: table_sha,
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

/// `search robber-burn`: the cost of searching another seat's rob burn inside the exact solve.
/// From the decision set (`--trick`, `--count`, `--seed`, `--start-hands`, as `search decisions`)
/// the decisions at `--position` (0 = leader) are taken. For each, `--share-worlds` yardstick
/// worlds (label `--label`) are sampled to give the share where another seat robs later in the
/// trick (no solve). With `--equity TABLE`, the first `--worlds` worlds of each decision are also
/// solved exactly twice, with the burn searched (the yardstick) and with the `rules-v1` burn,
/// and the node counts compared (the multiplier is searched nodes / rules-burn nodes).
fn robber_burn(a: &Args) -> Result<(), String> {
    use search::pimc::{WorldSolve, candidates, exact_world_row, world_robber};
    let trick = a.num("--trick", 1)? as u8;
    let seed = a.num("--seed", 1)?;
    let label = a.num("--label", 1)?;
    let position = a.num("--position", 0)? as u8;
    let starts = collect_starts(a.num("--start-hands", 20_000)?, seed);
    let ds: Vec<DecisionSpec> = build_decisions(trick, a.num("--count", 32)?, seed, &starts)
        .into_iter()
        .filter(|d| d.position == position)
        .collect();
    let share_worlds = a.num("--share-worlds", 1000)?;
    let solve_worlds = a.num("--worlds", 4)?;
    let table = a.get("--equity").map(load_equity).transpose()?;
    let src = world_source(a)?;
    let fallback: Arc<SharedLeaf> = Arc::new(solver::FnLeaf(|_: &equity::MatchState| {
        equity::MoneyChance::even()
    }));
    let leaf: Arc<SharedLeaf> = match table {
        Some(t) => Arc::new(t),
        None => fallback,
    };
    let ctx = Ctx {
        leaf: &*leaf,
        source: &*src,
        rules: engine::game::Rules::HOUSE,
    };
    let cfg = SearchConfig {
        tt_mb: a.num("--tt-mb", 512)? as usize,
        ..SearchConfig::default()
    };
    println!(
        "robber-burn: trick {trick}, position {position}: {} decisions",
        ds.len()
    );
    let (mut robbed, mut total) = (0u64, 0u64);
    let (mut n_search, mut n_rules) = (0u64, 0u64);
    for d in &ds {
        let (_, view, legal) = d.replay();
        let (_, cands) = candidates(&view, &legal);
        let worlds = ctx.source.prepare(&view, ctx.rules);
        let dseed = search::measure::measure_seed(seed, label, d.id);
        let mut r = 0u64;
        for w in 0..share_worlds {
            r += u64::from(world_robber(&view, &*worlds, w, dseed, &ctx).is_some());
        }
        robbed += r;
        total += share_worlds;
        println!(
            "decision {}: another seat robs in {r} of {share_worlds} worlds",
            d.id
        );
        if a.get("--equity").is_none() {
            continue;
        }
        for w in 0..solve_worlds {
            let mut nodes = [0u64; 2];
            for (k, rules_burn) in [false, true].into_iter().enumerate() {
                let c = SearchConfig {
                    rules_robber_burn: rules_burn,
                    ..cfg
                };
                let mut mem = solver::SolveMem::new();
                let mut run = WorldSolve::default();
                let t = Instant::now();
                exact_world_row(
                    &view, &*worlds, &cands, w, dseed, &ctx, &c, None, &mut mem, &mut run,
                )
                .expect("not stopped");
                nodes[k] = run.stats.nodes;
                println!(
                    "  world {w} {}: robber {:?}, {} nodes, {:.1} s",
                    if rules_burn {
                        "rules burn "
                    } else {
                        "burn search"
                    },
                    run.robber,
                    run.stats.nodes,
                    t.elapsed().as_secs_f64()
                );
            }
            n_search += nodes[0];
            n_rules += nodes[1];
        }
    }
    println!(
        "share of worlds where another seat robs: {robbed} / {total} = {:.2}%",
        100.0 * robbed as f64 / total.max(1) as f64
    );
    if n_rules > 0 {
        println!(
            "nodes: burn searched {n_search}, rules burn {n_rules}, multiplier {:.3}",
            n_search as f64 / n_rules as f64
        );
    }
    Ok(())
}

/// SHA-256 (hex) of the running binary, for records (empty if it cannot be read).
fn binary_sha256() -> String {
    std::env::current_exe()
        .and_then(std::fs::read)
        .map(|b| equity::sha256::sha256_hex(&b))
        .unwrap_or_default()
}

/// `search labels`: a label shard for the oracle network O_K and its sidecar.
fn labels_cmd(a: &Args) -> Result<(), String> {
    use search::oracle_labels::{
        CHUNK_ROWS, Driver, GOLDEN_SEED, HARVEST_CAP, HARVEST_SHARES, HARVEST_STRATA,
        HARVEST_TT_MB, HarvestJob, LabelJob, RACE_CAP, RaceKeep, Source, parse_band_shares,
        parse_race_classes, parse_strata, position_schedule, write_harvest_shard,
        write_label_shard,
    };
    let trick = a.num("--trick", 4)? as u8;
    if !(2..=5).contains(&trick) {
        return Err("--trick is 2..=5 (the start of that trick)".into());
    }
    let seed: u64 = a
        .req("--seed")?
        .parse()
        .map_err(|_| "--seed: not a number".to_string())?;
    if seed == GOLDEN_SEED {
        return Err("--seed 1 is the golden sets' seed: use another".into());
    }
    let job = LabelJob {
        trick,
        count: a.num("--count", 1000)?,
        seed,
        source: Source::parse(a.req("--source")?)?,
        threads: a.num("--threads", 1)?.max(1) as usize,
    };
    let harvest_flags = [
        "--harvest-cap",
        "--harvest-positions",
        "--harvest-tricks",
        "--harvest-net",
        "--harvest-strata",
        "--harvest-race-classes",
        "--harvest-race-cap",
    ];
    let harvest = if job.source == Source::Harvest {
        let race = match a.get("--harvest-race-classes") {
            None => {
                if a.has("--harvest-race-cap") {
                    return Err("--harvest-race-cap is for a race harvest: give --harvest-race-classes (e.g. 2,4)"
                        .into());
                }
                None
            }
            Some(c) => {
                let classes =
                    parse_race_classes(c).map_err(|e| format!("--harvest-race-classes: {e}"))?;
                let cap = a.num("--harvest-race-cap", u64::from(RACE_CAP))?;
                if !(1..=u64::from(u32::MAX)).contains(&cap) {
                    return Err("--harvest-race-cap is at least 1".into());
                }
                Some(RaceKeep {
                    classes,
                    cap: cap as u32,
                })
            }
        };
        let cap = a.num("--harvest-cap", u64::from(HARVEST_CAP))?;
        if cap > u64::from(u32::MAX) || (cap == 0 && race.is_none()) {
            return Err(
                "--harvest-cap is at least 1 (0, keeping only the race classes, in a race harvest)"
                    .into(),
            );
        }
        let strata = match a.get("--harvest-strata") {
            None => HARVEST_STRATA.to_vec(),
            Some(x) => parse_strata(x).map_err(|e| format!("--harvest-strata: {e}"))?,
        };
        let shares = a.get("--harvest-positions").unwrap_or(HARVEST_SHARES);
        let weights = parse_band_shares(shares).map_err(|e| format!("--harvest-positions: {e}"))?;
        let tricks: Vec<u8> = match a.get("--harvest-tricks") {
            None => vec![trick - 1],
            Some(t) => t
                .split(',')
                .map(|x| {
                    x.trim()
                        .parse::<u8>()
                        .ok()
                        .filter(|t| (1..trick).contains(t))
                })
                .collect::<Option<Vec<_>>>()
                .filter(|v| !v.is_empty())
                .ok_or_else(|| {
                    let last = trick - 1;
                    format!(
                        "--harvest-tricks {t:?}: decision tricks from 1 to {last} (before the \
                         frontier), e.g. {last}"
                    )
                })?,
        };
        let total: u64 = weights.iter().sum();
        if total > 1 << 24 {
            return Err(format!(
                "--harvest-positions {shares:?}: the shares make a cycle of {total} decisions; use \
                 fewer decimals"
            ));
        }
        Some((
            HarvestJob {
                trick,
                count: job.count,
                seed,
                threads: job.threads,
                cap: cap as u32,
                tricks,
                schedule: position_schedule(weights),
                strata,
                race,
            },
            shares.to_string(),
            weights,
        ))
    } else {
        if let Some(f) = harvest_flags.iter().find(|f| a.has(f)) {
            return Err(format!("{f} is for --source harvest only"));
        }
        None
    };
    let out = PathBuf::from(a.req("--out")?);
    let chunk_rows = a.num("--chunk-rows", CHUNK_ROWS)?.max(1);
    let t0 = Instant::now();
    let (table, table_sha) = load_equity_sha(a.req("--equity")?)?;
    let leaf: Arc<SharedLeaf> = Arc::new(table);
    let table_s = t0.elapsed().as_secs_f64();
    // The network that drives a harvest's search (`--harvest-net`): the integer O_k of an
    // earlier round, loaded as `search hybrid-check` loads it.
    let net = match a.get("--harvest-net") {
        Some(p) => {
            let o = nn::Oracle::load(Path::new(p), leaf.clone())?;
            if o.trick != trick {
                return Err(format!(
                    "--harvest-net {p} was trained for trick {} positions, not trick {trick}",
                    o.trick
                ));
            }
            eprintln!(
                "labels: harvest driven by the network {p} (sha256 {}, {} head, widths {:?}): \
                 only the sampled positions are solved exactly",
                o.sha256,
                o.head.name(),
                o.net().widths()
            );
            Some((o, p.to_string()))
        }
        None => None,
    };
    let driver = net.as_ref().map_or(Driver::Exact, |(o, _)| Driver::Net(o));
    let start_hands = a.num("--start-hands", 20_000)?;
    let starts = collect_starts_with(start_hands, seed, job.threads);
    eprintln!(
        "labels: trick {trick}, {} rows from {}, seed {seed}, {} threads (table and start states in {:.1} s: table {table_s:.1} s)",
        job.count,
        job.source.name(),
        job.threads,
        t0.elapsed().as_secs_f64()
    );
    let done = std::sync::atomic::AtomicUsize::new(0);
    let finished = AtomicBool::new(false);
    let t1 = Instant::now();
    let shard = std::thread::scope(|s| {
        let (done, finished) = (&done, &finished);
        s.spawn(move || {
            let mut last = Instant::now();
            while !finished.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(200));
                if last.elapsed() >= Duration::from_secs(20) {
                    last = Instant::now();
                    eprintln!(
                        "labels: {} of {} rows, {:.0} s",
                        (done.load(Ordering::Relaxed) as u64).min(job.count),
                        job.count,
                        t0.elapsed().as_secs_f64()
                    );
                }
            }
        });
        // Stops the progress thread however `label_rows` ends (a panic included).
        struct Finish<'a>(&'a AtomicBool);
        impl Drop for Finish<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        let _finish = Finish(finished);
        // The shard goes to a temporary name, renamed once whole: a stopped run leaves no
        // shard that looks finished.
        let mut part = out.as_os_str().to_owned();
        part.push(".part");
        let part = PathBuf::from(part);
        let mut file =
            std::fs::File::create(&part).map_err(|e| format!("{}: {e}", part.display()))?;
        let summary = match &harvest {
            Some((h, _, _)) => {
                write_harvest_shard(h, &starts, &*leaf, driver, Some(done), &mut file)
            }
            None => write_label_shard(&job, &starts, &*leaf, Some(done), chunk_rows, &mut file),
        }
        .map_err(|e| format!("{}: {e}", part.display()))?;
        drop(file);
        std::fs::rename(&part, &out).map_err(|e| format!("{}: {e}", out.display()))?;
        Ok::<_, String>(summary)
    })?;
    let rows_s = t1.elapsed().as_secs_f64();
    let n_rows = shard.rows;
    let c = shard.counts;
    let (by_stratum, by_root, by_position) = (c.by_stratum, c.by_root_trick, c.by_position);
    let (retries, nodes) = (c.retries, c.nodes);
    let (hash_s, write_s) = (
        shard.hash_time.as_secs_f64(),
        shard.write_time.as_secs_f64(),
    );
    let (search_s, label_s) = (
        shard.search_time.as_secs_f64(),
        shard.label_time.as_secs_f64(),
    );
    let decisions = shard.decisions;
    let secs = t0.elapsed().as_secs_f64();
    let mut side = serde_json::json!({
        "format": "41bots-label-shard-v1",
        "encoder": nn::encode::ENCODER,
        "n_features": nn::encode::N_FEATURES,
        "n_binary": nn::encode::N_BINARY,
        "n_dense": nn::encode::N_DENSE,
        "max_active": nn::encode::MAX_ACTIVE,
        "record": nn::shard::descr(nn::outcome::vocab(trick)),
        "outcomes": nn::outcome::vocab(trick),
        "value_order": "rotated: the leader is seat 0 (as the encoder's outputs)",
        "trick": trick,
        "source": job.source.name(),
        "count": job.count,
        "seed": seed,
        "start_hands": start_hands,
        "values_version": search::measure::VALUES_VERSION,
        "tie_rule": "first strictly best card in increasing card index (solver default)",
        "binary_sha256": binary_sha256(),
        "equity_sha256": table_sha,
        "shard_sha256": shard.sha256,
        "rows_by_stratum": by_stratum,
        "rows_by_race_class": c.by_class,
        "race_classes": nn::outcome::RACE_CLASS_NAMES,
        "rows_by_root_trick": &by_root[1..trick as usize],
        "replayed_lines": retries,
        "solver_nodes": nodes,
        "seconds": (secs * 10.0).round() / 10.0,
    });
    if job.source != Source::Rules {
        side["rows_by_position"] = serde_json::json!(&by_position[1..]);
    }
    if let Some((h, shares, weights)) = &harvest {
        side["harvest"] = harvest_record(
            h,
            shares,
            weights,
            &decisions,
            HARVEST_TT_MB,
            net.as_ref().map(|(o, p)| (o, p.as_str())),
        );
        let n = decisions.len().max(1) as f64;
        let distinct: u64 = decisions.iter().map(|d| u64::from(d.distinct)).sum();
        eprintln!(
            "labels: harvest: {} decisions (by position 1-8: {}), {:.1} distinct frontier \
             positions a decision ({:.1} frontier evaluations), {}; rows by race class 0-4: {:?}",
            decisions.len(),
            side["harvest"]["decisions_by_position"],
            distinct as f64 / n,
            decisions.iter().map(|d| d.evals).sum::<u64>() as f64 / n,
            match h.race {
                None => format!(
                    "{} decisions capped at {}",
                    decisions.iter().filter(|d| d.distinct > h.cap).count(),
                    h.cap
                ),
                Some(r) => format!(
                    "{:.1} positions of race classes {:?} a decision, {} decisions capped at {} \
                     of them (other classes: at most {} a decision)",
                    decisions
                        .iter()
                        .map(|d| u64::from(d.race_distinct))
                        .sum::<u64>() as f64
                        / n,
                    r.list(),
                    decisions.iter().filter(|d| d.race_distinct > r.cap).count(),
                    r.cap,
                    h.cap
                ),
            },
            c.by_class
        );
        eprintln!(
            "labels: harvest: frontier driven by {}; searches {search_s:.1} thread-s, labelling \
             the sample {label_s:.1} thread-s ({:.2} thread-s per 1,000 rows in all)",
            match &net {
                Some((o, _)) => format!("the network {}", &o.sha256[..12]),
                None => "exact values".to_string(),
            },
            (search_s + label_s) * 1000.0 / n_rows.max(1) as f64
        );
    }
    let mut side_path = out.as_os_str().to_owned();
    side_path.push(".json");
    std::fs::write(
        &side_path,
        serde_json::to_string_pretty(&side).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| e.to_string())?;
    eprintln!(
        "labels: {} rows made, hashed and written in {:.1} s ({:.3} ms a row on {} threads; \
         hashing {:.1} s and writing {:.1} s alongside; {secs:.1} s in all); shard sha256 {}",
        n_rows,
        rows_s,
        rows_s * 1000.0 / n_rows.max(1) as f64,
        job.threads,
        hash_s,
        write_s,
        side["shard_sha256"].as_str().unwrap_or("")
    );
    Ok(())
}

/// The sidecar's `harvest` record: the settings, the decisions' counts by position, stratum and
/// trick, and the distribution of the distinct frontier positions per decision.
fn harvest_record(
    h: &search::oracle_labels::HarvestJob,
    shares: &str,
    weights: &[u64; 8],
    ds: &[search::oracle_labels::HarvestDecision],
    tt_mb: usize,
    net: Option<(&nn::Oracle, &str)>,
) -> serde_json::Value {
    let mut by_pos = [0u64; 8];
    let mut rows_by_pos = [0u64; 8];
    let mut distinct_by_pos = [0u64; 8];
    let mut evals_by_pos = [0u64; 8];
    let mut by_stratum = [0u64; 4];
    let mut by_trick = [0u64; 6];
    let mut distinct: Vec<u32> = Vec::with_capacity(ds.len());
    for d in ds {
        let p = usize::from(d.position - 1);
        by_pos[p] += 1;
        rows_by_pos[p] += u64::from(d.rows);
        distinct_by_pos[p] += u64::from(d.distinct);
        evals_by_pos[p] += d.evals;
        by_stratum[usize::from(d.stratum - 1)] += 1;
        by_trick[usize::from(d.trick)] += 1;
        distinct.push(d.distinct);
    }
    let mean_by_pos = |tot: &[u64; 8]| -> Vec<f64> {
        tot.iter()
            .zip(&by_pos)
            .map(|(&t, &n)| (t as f64 / n.max(1) as f64 * 10.0).round() / 10.0)
            .collect()
    };
    distinct.sort_unstable();
    let q = |f: f64| {
        distinct
            .get(((distinct.len().max(1) - 1) as f64 * f).round() as usize)
            .copied()
            .unwrap_or(0)
    };
    let n = ds.len().max(1) as f64;
    let race = h.race.map(|r| {
        serde_json::json!({
            "classes": r.list(),
            "cap_per_decision": r.cap,
            "other_classes_cap_per_decision": h.cap,
            "what": "frontier positions of the race classes (nn::outcome::race_class) are a \
                     seeded sample of at most cap_per_decision of them, the other classes' of at \
                     most other_classes_cap_per_decision (drawn after them on the same stream)",
            "race_positions_reached": ds.iter().map(|d| u64::from(d.race_distinct)).sum::<u64>(),
            "decisions_race_capped": ds.iter().filter(|d| d.race_distinct > r.cap).count(),
        })
    });
    let shares_given: Vec<f64> = shares
        .split(',')
        .map(|x| x.trim().parse().unwrap_or(f64::NAN))
        .collect();
    let (what, frontier_values, driver) = match net {
        None => (
            "frontier positions (start of the trick) that the hybrid's search from training \
             decisions reaches for every candidate, with exact values at the frontier (H_k with a \
             perfect network), deduplicated by canonical form within a decision, then a seeded \
             sample of at most cap_per_decision of them, kept in the order reached",
            "exact (solver::solve_fast_with of the reached position, the label's own value)"
                .to_string(),
            serde_json::json!({ "kind": "exact" }),
        ),
        Some((o, path)) => (
            "frontier positions (start of the trick) that the hybrid's search from training \
             decisions reaches for every candidate, with the network's values at the frontier \
             (H_k with that network, as the hybrid runs), deduplicated by canonical form within \
             a decision, then a seeded sample of at most cap_per_decision of them, kept in the \
             order reached; only the sampled positions are solved exactly, for their labels",
            format!(
                "the network's (net sha256 {}: nn::Oracle, integer inference with its memo) during \
                 the search; each sampled position then solved exactly (solver::solve_fast_with, \
                 the label's value)",
                o.sha256
            ),
            serde_json::json!({
                "kind": "net",
                "drove_the_search": true,
                "net_sha256": o.sha256,
                "net_file": Path::new(path).file_name().map(|f| f.to_string_lossy().into_owned()),
                "net_trick": o.trick,
                "net_head": o.head.name(),
                "net_widths": o.net().widths(),
                "net_meta": o.meta,
            }),
        ),
    };
    serde_json::json!({
        "what": what,
        "decision_generator": "search::measure::build_decision (the yardstick's): stratum strata[j mod s] \
                               and decider position schedule[(j div s) mod position_cycle], s = the number \
                               of strata (all four: stratum j mod 4 + 1, as the yardstick), decision trick \
                               uniform over decision_tricks",
        "strata": h.strata,
        "race": race,
        "decision_seed": "Rng::derive(seed, [NN_LABEL, SOURCE_HARVEST, trick, j]); the same stream \
                          then draws the world (belief sampler) and the sample",
        "search": format!("pimc::exact_world, sequential solver, table {tt_mb} MB, frontier at the start of trick {}", h.trick),
        "frontier_values": frontier_values,
        "frontier_driver": driver,
        "cap_per_decision": h.cap,
        "decision_tricks": h.tricks,
        "position_shares": { "bands": ["1-2", "3-4", "5-8"], "given": shares, "shares": shares_given },
        "position_weights": weights,
        "position_cycle": h.schedule.len(),
        "decisions": ds.len(),
        "decisions_by_position": by_pos,
        "rows_by_position": rows_by_pos,
        "decisions_by_stratum": by_stratum,
        "decisions_by_trick": &by_trick[1..usize::from(h.trick)],
        "decisions_capped": ds.iter().filter(|d| d.distinct > h.cap && h.race.is_none()).count(),
        "frontier_evaluations": ds.iter().map(|d| d.evals).sum::<u64>(),
        "distinct_positions": distinct.iter().map(|&x| u64::from(x)).sum::<u64>(),
        "distinct_mean_by_position": mean_by_pos(&distinct_by_pos),
        "evaluations_mean_by_position": mean_by_pos(&evals_by_pos),
        "distinct_per_decision": {
            "mean": (distinct.iter().map(|&x| f64::from(x)).sum::<f64>() / n * 10.0).round() / 10.0,
            "min": q(0.0), "p10": q(0.1), "median": q(0.5), "p90": q(0.9), "max": q(1.0),
        },
        "search_nodes": ds.iter().map(|d| d.search_nodes).sum::<u64>(),
    })
}

/// `search hybrid-check`: H_K against exact on the same worlds (docs/M3-HYBRID-SPEC.md §3).
fn hybrid_check(a: &Args) -> Result<(), String> {
    use search::hybrid::{CheckJob, report, run_check};
    let trick = a.num("--trick", 3)? as u8;
    let cut = a.num("--cut", u64::from(trick) + 1)? as u8;
    if !(2..=5).contains(&cut) || trick == 0 || trick >= cut {
        return Err(format!(
            "--cut {cut} must be 2..=5 and after --trick {trick} (and --trick at least 1)"
        ));
    }
    let seed = a.num("--seed", 1)?;
    let t0 = Instant::now();
    let (table, table_sha) = load_equity_sha(a.req("--equity")?)?;
    let leaf: Arc<SharedLeaf> = Arc::new(table);
    let net_path = a.req("--net")?;
    let oracle = nn::Oracle::load(Path::new(net_path), leaf.clone())?;
    if oracle.trick != cut {
        return Err(format!(
            "{net_path} was trained for trick {} positions, not trick {cut}",
            oracle.trick
        ));
    }
    let mut ds: Vec<DecisionSpec> = match a.get("--decisions") {
        Some(f) => read_lines::<DecisionSpec>(f)?
            .into_iter()
            .filter(|d| d.trick == trick)
            .collect(),
        None => {
            let threads = a.num("--threads", 1)?.max(1) as usize;
            let starts = collect_starts_with(a.num("--start-hands", 20_000)?, seed, threads);
            build_decisions(trick, a.num("--count", 32)?, seed, &starts)
        }
    };
    let (first, last) = (a.num("--first", 0)?, a.num("--last", u64::MAX)?);
    let ids = a.get("--ids").map(parse_ids).transpose()?;
    let selected =
        |id: u64| id >= first && id <= last && ids.as_ref().is_none_or(|i| i.contains(&id));
    ds.retain(|d| selected(d.id));
    if ds.is_empty() {
        return Err(format!("no decisions at trick {trick}"));
    }
    let tt_mb = a.num("--tt-mb", 64)? as usize;
    let job = CheckJob {
        cut,
        seed,
        label: a.num("--label", 1)?,
        to: a.num("--to", 320)?,
        threads: a.num("--threads", 1)?.max(1) as usize,
        tt_mb,
        verify_pct: float(a, "--verify-pct", if a.has("--saved") { 0.0 } else { 1.0 })?,
        recompute: a.has("--recompute"),
        rules_robber_burn: false,
        exact_solve_threads: a.num("--verify-solve-threads", 1)?.max(1) as usize,
        exact_tt_mb: a.num("--verify-tt-mb", tt_mb as u64)? as usize,
    };
    let src = world_source(a)?;
    let ctx = Ctx {
        leaf: &*leaf,
        source: &*src,
        rules: engine::game::Rules::HOUSE,
    };
    let boots = a.num("--boot", 1000)?;
    let boot_seed = a.num("--boot-seed", 1)?;
    if a.has("--saved") {
        return hybrid_check_saved(
            a,
            SavedRun {
                trick,
                ds: &ds,
                selected: &selected,
                job,
                ctx: &ctx,
                oracle: &oracle,
                net_path,
                table_sha: &table_sha,
                boots,
                boot_seed,
                t0,
            },
        );
    }
    let stored: BTreeMap<u64, WorldRows> = if a.has("--worlds") {
        let paths: Vec<&Path> = a.all("--worlds").into_iter().map(Path::new).collect();
        let (rows, warnings) = read_worlds(&paths)?;
        for w in warnings {
            eprintln!("warning: {w}");
        }
        rows.into_iter()
            .filter(|r| r.method == "exact" && r.trick == trick)
            .map(|r| (r.id, r))
            .collect()
    } else {
        BTreeMap::new()
    };
    if a.has("--worlds")
        && let Some(d) = ds.iter().find(|d| !stored.contains_key(&d.id))
    {
        return Err(format!("decision {} has no stored exact worlds", d.id));
    }
    if src.name() != "rejection"
        && let Some(r) = stored.values().find(|r| r.log2_worlds.is_none())
    {
        return Err(format!(
            "decision {}: the stored worlds have no world count, so they were drawn by the \
             rejection sampler (the default before the belief sampler): pass --source rejection",
            r.id
        ));
    }
    if let Some(ids) = a.get("--explain") {
        return hybrid_explain(a, ids, &ds, &stored, &job, &ctx, &oracle);
    }
    let exact_from = if stored.is_empty() || job.recompute {
        "recomputed by this binary".to_string()
    } else {
        format!(
            "stored ({}), {}% sample recomputed",
            a.all("--worlds").join(", "),
            job.verify_pct
        )
    };
    eprintln!(
        "hybrid-check: {} decisions at trick {trick}, frontier at trick {cut}, net {} {:?}, {} worlds each (seed {seed}, label {}, {} source), {} threads; exact values {exact_from}",
        ds.len(),
        &oracle.sha256[..12],
        oracle.net().widths(),
        job.to,
        job.label,
        src.name(),
        job.threads
    );
    let t1 = Instant::now();
    let run = run_check(&ds, &stored, &job, &ctx, &oracle)?;
    let wall = t1.elapsed().as_secs_f64();
    if run.verified.1 > 0 {
        return Err(format!(
            "{} of {} recomputed worlds differ from the stored exact values: recompute them \
             (--recompute, or off this VM for trick 2)",
            run.verified.1, run.verified.0
        ));
    }
    let groups = report(&run, boots, boot_seed);
    println!(
        "H_{cut} vs exact, trick {trick}: {} decisions x {} worlds (seed {seed}, label {}, {} source); net {} {:?}; exact values {exact_from}{}",
        run.decisions.len(),
        job.to,
        job.label,
        src.name(),
        &oracle.sha256[..12],
        oracle.net().widths(),
        if run.verified.0 > 0 {
            format!(" ({} worlds verified byte-identical)", run.verified.0)
        } else {
            String::new()
        }
    );
    print_groups(&groups);
    let time = check_time(&run, job.threads, wall);
    print_time(&time);
    let (pass, line) = verdict(&groups);
    println!("{line}");
    if let Some(out) = a.get("--json") {
        let v = serde_json::json!({
            "trick": trick,
            "cut": cut,
            "seed": seed,
            "label": job.label,
            "worlds": job.to,
            "world_source": src.name(),
            "exact_values": exact_from,
            "verified_worlds": run.verified.0,
            "net": net_record(net_path, &oracle),
            "values": {
                "exact": search::measure::VALUES_VERSION,
                "hybrid": format!("{}+O{cut}:{}", search::measure::VALUES_VERSION, oracle.sha256),
            },
            "equity_sha256": table_sha,
            "binary_sha256": binary_sha256(),
            "boot": {"replicates": boots, "seed": boot_seed, "bound": "97.5th percentile of bootstrap means (upper end of the 95% interval)"},
            "groups": groups.iter().map(|(n, s)| serde_json::json!({"group": n, "summary": s})).collect::<Vec<_>>(),
            "pass": pass,
            "time": time_json(&time, t1.duration_since(t0).as_secs_f64()),
            "decisions": run.decisions,
        });
        write_json(out, &v)?;
    }
    Ok(())
}

fn write_json(out: &str, v: &serde_json::Value) -> Result<(), String> {
    std::fs::write(
        out,
        serde_json::to_string_pretty(v).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| format!("{out}: {e}"))
}

fn net_record(net_path: &str, oracle: &nn::Oracle) -> serde_json::Value {
    serde_json::json!({
        "path": net_path,
        "sha256": oracle.sha256,
        "widths": oracle.net().widths(),
        "trick": oracle.trick,
        "meta": oracle.meta,
    })
}

/// The §3 table of `hybrid-check` (`search::hybrid::report`'s groups).
fn print_groups(groups: &[(String, search::hybrid::Summary)]) {
    let spct = |x: f64| format!("{:+.4}%", 100.0 * x);
    println!(
        "{:<9} {:>5} {:>10} {:>10} {:>7} {:>7} {:>8} {:>38} {:>9} {:>10} {:>9}",
        "group",
        "dec",
        "mean loss",
        "95% upper",
        "<=0.01%",
        "agree",
        "values",
        "bias: own / next / opposite / previous",
        "SD",
        "p99.9|e|",
        "max|e|"
    );
    for (name, s) in groups.iter().filter(|(_, s)| s.decisions > 0) {
        println!(
            "{:<9} {:>5} {:>10} {:>10} {:>7} {:>6.1}% {:>8} {:>38} {:>9} {:>10} {:>9}",
            name,
            s.decisions,
            pct(s.mean_loss),
            pct(s.loss_upper95),
            if s.loss_pass { "yes" } else { "no" },
            100.0 * s.agreement,
            s.values,
            s.bias.map(spct).join(" "),
            pct(s.sd),
            pct(s.p999_abs),
            pct(s.max_abs)
        );
    }
    if let Some((_, s)) = groups.first() {
        println!(
            "(value error over all four teams' chances; bias per team relative to the decider; the decider's team alone: SD {}, p99.9|e| {})",
            pct(s.decider_sd),
            pct(s.decider_p999_abs)
        );
    }
}

/// The standard of §3 on `groups`: whether it passes, and the line that says so.
fn verdict(groups: &[(String, search::hybrid::Summary)]) -> (bool, String) {
    let loss_pass = groups.iter().take(4).all(|(_, s)| s.loss_pass);
    let bias_pass = groups.first().is_some_and(|(_, s)| s.bias_pass);
    let pass = loss_pass && bias_pass;
    let worst_bias = groups.first().map_or(0.0, |(_, s)| {
        s.bias.iter().fold(0.0f64, |m, b| m.max(b.abs()))
    });
    let line = format!(
        "standard: decision loss 95% upper bound <= 0.01% overall and in every band: {}; every team's |bias| <= 0.01%: {} (largest {}); {}",
        if loss_pass { "yes" } else { "no" },
        if bias_pass { "yes" } else { "no" },
        pct(worst_bias),
        if pass { "PASS" } else { "FAIL" }
    );
    (pass, line)
}

/// Times and counts of a check (reports only).
struct CheckTime {
    worlds: f64,
    exact_worlds: f64,
    exact_us: u64,
    hybrid_us: u64,
    exact_nodes: u64,
    hybrid_nodes: u64,
    evals: u64,
    ns_eval: f64,
    wall: f64,
    threads: usize,
}

fn check_time(run: &search::hybrid::CheckRun, threads: usize, wall: f64) -> CheckTime {
    let d = &run.decisions;
    CheckTime {
        worlds: d.iter().map(|d| d.worlds).sum::<u64>().max(1) as f64,
        exact_worlds: d.iter().map(|d| d.exact_worlds).sum::<u64>() as f64,
        exact_us: d.iter().map(|d| d.exact_us).sum(),
        hybrid_us: d.iter().map(|d| d.hybrid_us).sum(),
        exact_nodes: d.iter().map(|d| d.exact_nodes).sum(),
        hybrid_nodes: d.iter().map(|d| d.hybrid_nodes).sum(),
        evals: d.iter().map(|d| d.frontier).sum(),
        ns_eval: run.net.0 as f64 / run.net.1.max(1) as f64,
        wall,
        threads,
    }
}

fn print_time(t: &CheckTime) {
    println!(
        "time: exact {:.3} ms a world ({:.0} nodes; {} worlds solved exactly here), hybrid {:.3} ms a world ({:.0} nodes, {:.0} frontier evaluations); {:.0} ns per frontier evaluation (encode + integer forward + softmax, or the oracle's memo of recent positions); {:.1} s wall on {} threads",
        t.exact_us as f64 / 1000.0 / t.exact_worlds.max(1.0),
        t.exact_nodes as f64 / t.exact_worlds.max(1.0),
        t.exact_worlds,
        t.hybrid_us as f64 / 1000.0 / t.worlds,
        t.hybrid_nodes as f64 / t.worlds,
        t.evals as f64 / t.worlds,
        t.ns_eval,
        t.wall,
        t.threads
    );
}

fn time_json(t: &CheckTime, setup_s: f64) -> serde_json::Value {
    serde_json::json!({
        "exact_worlds_solved": t.exact_worlds,
        "exact_ms_per_world": t.exact_us as f64 / 1000.0 / t.exact_worlds.max(1.0),
        "hybrid_ms_per_world": t.hybrid_us as f64 / 1000.0 / t.worlds,
        "exact_nodes_per_world": t.exact_nodes as f64 / t.exact_worlds.max(1.0),
        "hybrid_nodes_per_world": t.hybrid_nodes as f64 / t.worlds,
        "net_evals_per_world": t.evals as f64 / t.worlds,
        "ns_per_net_eval": t.ns_eval,
        "wall_s": t.wall,
        "threads": t.threads,
        "setup_s": setup_s,
    })
}

/// What `hybrid_check_saved` needs from the common set-up.
struct SavedRun<'a> {
    trick: u8,
    ds: &'a [DecisionSpec],
    selected: &'a dyn Fn(u64) -> bool,
    job: search::hybrid::CheckJob,
    ctx: &'a Ctx<'a>,
    oracle: &'a nn::Oracle,
    net_path: &'a str,
    table_sha: &'a str,
    boots: u64,
    boot_seed: u64,
    t0: Instant,
}

/// `search hybrid-check --saved LABEL=FILE …`: H_K against the exact values earlier runs saved
/// (`search::saved`; docs/M3-HYBRID-SPEC.md §3, "Trick-1 checks"), one set per burn rule, each
/// checked with the matching `rules_robber_burn`; per world, per decision, the §3 tables per
/// set, and the `--hard` worlds on their own lines with the saved exact times.
fn hybrid_check_saved(a: &Args, r: SavedRun) -> Result<(), String> {
    use search::hybrid::{CheckProgress, CheckRun, check_worlds, report};
    use search::saved::{BurnRule, join_saved, parse_pairs, plan, read_saved, saved_sets};
    for k in ["--worlds", "--recompute", "--to"] {
        if a.has(k) {
            return Err(format!(
                "{k} does not go with --saved (the saved files give the worlds; --verify-pct 100 \
                 recomputes them all)"
            ));
        }
    }
    let SavedRun { trick, ds, job, .. } = r;
    let cut = job.cut;
    let mut all = Vec::new();
    let mut files = Vec::new();
    for spec in a.all("--saved") {
        let (label, path) = spec.split_once('=').unwrap_or((spec, spec));
        let (w, info) = read_saved(Path::new(path), label)?;
        eprintln!(
            "saved {label}: {} lines ({} checkpoint lines, {} worlds records, {} torn), {} exact worlds, by values version {:?}{}",
            info.lines,
            info.ckpt_lines,
            info.records,
            info.torn,
            info.worlds,
            info.by_version,
            if info.other_method > 0 {
                format!(", {} worlds of other methods skipped", info.other_method)
            } else {
                String::new()
            }
        );
        files.push(info);
        all.extend(w);
    }
    let only = a.get("--only").map(parse_pairs).transpose()?;
    let hard = a
        .get("--hard")
        .map(parse_pairs)
        .transpose()?
        .unwrap_or_default();
    let sets = saved_sets(join_saved(all)?)?;
    let keep =
        |id: u64, w: u64| (r.selected)(id) && only.as_ref().is_none_or(|o| o.contains(&(id, w)));
    let verify_rule = a.get("--verify-rule").unwrap_or("all");
    if !["all", "rules-v1", "searched"].contains(&verify_rule) {
        return Err(format!(
            "--verify-rule {verify_rule}: all, rules-v1 or searched"
        ));
    }
    let both: Vec<(u64, u64, bool)> = search::saved::both_rules(&sets)
        .into_iter()
        .filter(|&(d, w, _)| keep(d, w))
        .collect();
    let src_name = r.ctx.source.name().to_string();
    // Each set's plan and settings (its burn rule; the verification where asked).
    let mut work = Vec::new();
    for (si, set) in sets.iter().enumerate() {
        let (pl, notes) = plan(ds, set, job.seed, &src_name, r.table_sha, &keep)?;
        if pl.is_empty() {
            continue;
        }
        let verify_this = match verify_rule {
            "rules-v1" => set.rule == BurnRule::RulesV1,
            "searched" => set.rule == BurnRule::Searched,
            _ => true,
        };
        let sjob = search::hybrid::CheckJob {
            rules_robber_burn: set.rule.rules_robber_burn(),
            verify_pct: if verify_this { job.verify_pct } else { 0.0 },
            ..job.clone()
        };
        work.push((si, set, pl, notes, sjob));
    }
    // The sets run at once, the threads shared out by worlds (never more threads than worlds,
    // never more in all than --threads, so no world's time is inflated by another's).
    let counts: Vec<usize> = work
        .iter()
        .map(|w| w.2.iter().map(|d| d.worlds.len()).sum())
        .collect();
    let split = split_threads(job.threads, &counts);
    for (w, &t) in work.iter_mut().zip(&split) {
        w.4.threads = t;
    }
    println!(
        "H_{cut} vs saved exact worlds, trick {trick}: net {} {:?}; {} source; {} threads (one world each; {}), tables {} MB; saved files: {}",
        &r.oracle.sha256[..12],
        r.oracle.net().widths(),
        src_name,
        job.threads,
        work.iter()
            .map(|w| format!("set {}: {} threads", w.0 + 1, w.4.threads))
            .collect::<Vec<_>>()
            .join(", "),
        job.tt_mb,
        files
            .iter()
            .map(|f| format!("{} ({} exact worlds)", f.from, f.worlds))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if !both.is_empty() {
        let differ: Vec<String> = both
            .iter()
            .filter(|b| !b.2)
            .map(|(d, w, _)| format!("{d}:{w}"))
            .collect();
        println!(
            "worlds saved under both burn rules: {}; values identical in {}{}",
            both.len(),
            both.len() - differ.len(),
            if differ.is_empty() {
                String::new()
            } else {
                format!(", different in {} ({})", differ.join(","), differ.len())
            }
        );
    }
    if a.has("--plan-only") {
        // What the check would do, with nothing solved: to price a run before renting.
        for (si, set, pl, notes, sjob) in &work {
            let mut verify = Vec::new();
            for d in pl {
                for t in &d.worlds {
                    if sjob.verify_sampled(d.d.id, t.w) {
                        verify.push(format!("{}:{} (saved cost {})", d.d.id, t.w, t.cost));
                    }
                }
            }
            let cost: u64 = pl.iter().flat_map(|d| &d.worlds).map(|t| t.cost).sum();
            println!(
                "plan: set {} ({}): {} decisions, {} worlds, saved cost {cost} (nodes where recorded, else microseconds), {} threads; verification ({}%): {} worlds{}{}",
                si + 1,
                set.rule.name(),
                notes.decisions,
                notes.worlds,
                sjob.threads,
                sjob.verify_pct,
                verify.len(),
                if verify.is_empty() { "" } else { ": " },
                verify.join(", ")
            );
        }
        return Ok(());
    }
    let progress_path = a.get("--progress").map(PathBuf::from);
    let progs: Vec<CheckProgress> = work.iter().map(|_| CheckProgress::default()).collect();
    let finished = AtomicBool::new(false);
    let t1 = Instant::now();
    let runs: Vec<(Result<CheckRun, String>, f64)> = std::thread::scope(|s| {
        let (progs, finished) = (&progs, &finished);
        let sum = move |f: fn(&CheckProgress) -> &std::sync::atomic::AtomicU64| -> u64 {
            progs.iter().map(|p| f(p).load(Ordering::Relaxed)).sum()
        };
        if let Some(p) = &progress_path {
            let threads = job.threads as u64;
            let ex_workers: u64 = work
                .iter()
                .map(|w| (w.4.threads / w.4.exact_solve_threads.max(1)).max(1) as u64)
                .sum();
            s.spawn(move || {
                while !finished.load(Ordering::Relaxed) {
                    let (ed, et) = (sum(|p| &p.exact_done), sum(|p| &p.exact_total));
                    let (hd, ht) = (sum(|p| &p.hybrid_done), sum(|p| &p.hybrid_total));
                    write_atomic(
                        p,
                        &serde_json::json!({
                            "exact_done": ed, "exact_total": et,
                            "worlds_done": hd, "worlds_total": ht,
                            // Fewer solves left than threads: some threads are idle by design.
                            "tail": if ed < et { et - ed < ex_workers } else { ht - hd < threads },
                            "elapsed_s": t1.elapsed().as_secs_f64(),
                        }),
                    );
                    for _ in 0..20 {
                        if finished.load(Ordering::Relaxed) {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    }
                }
            });
        }
        // Each world on stderr as it finishes, so a stopped run still shows what finished.
        let set_of: Vec<usize> = work.iter().map(|w| w.0 + 1).collect();
        s.spawn(move || {
            let mut last = false;
            while !last {
                last = finished.load(Ordering::Relaxed);
                for (p, set) in progs.iter().zip(&set_of) {
                    let done: Vec<[u64; 5]> =
                        std::mem::take(&mut *p.finished.lock().unwrap_or_else(|e| e.into_inner()));
                    for [d, w, us, nodes, evals] in done {
                        eprintln!(
                            "set {set} d{d}:w{w}: H_{cut} {:.2} s, {:.3e} nodes, {evals} evaluations ({}/{} worlds, {:.0} s)",
                            us as f64 / 1e6,
                            nodes as f64,
                            sum(|p| &p.hybrid_done),
                            sum(|p| &p.hybrid_total),
                            t1.elapsed().as_secs_f64()
                        );
                    }
                }
                if !last {
                    std::thread::sleep(Duration::from_millis(200));
                }
            }
        });
        let handles: Vec<_> = work
            .iter()
            .zip(progs)
            .map(|(w, p)| {
                s.spawn(move || {
                    let t = Instant::now();
                    let run = check_worlds(&w.2, &w.4, r.ctx, r.oracle, Some(p));
                    (run, t.elapsed().as_secs_f64())
                })
            })
            .collect();
        let out = handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect();
        finished.store(true, Ordering::Relaxed);
        out
    });
    let mut set_docs = Vec::new();
    let mut hard_lines: Vec<serde_json::Value> = Vec::new();
    for ((si, set, _pl, notes, sjob), (run, wall)) in work.iter().zip(runs) {
        let run = run?;
        let n_worlds: usize = run.decisions.iter().map(|d| d.worlds as usize).sum();
        let found: usize = set
            .worlds
            .iter()
            .filter(|w| keep(w.id, w.w))
            .map(|w| w.found.len())
            .sum();
        println!();
        println!(
            "set {}: {} (values version {}, rules_robber_burn = {}): {} decisions, {} worlds (found {found} times in the files); {} threads",
            si + 1,
            set.rule.name(),
            set.values,
            sjob.rules_robber_burn,
            notes.decisions,
            if run.mismatched.is_empty() {
                n_worlds
            } else {
                notes.worlds
            },
            sjob.threads
        );
        if notes.table_unrecorded > 0 || notes.seed_unrecorded > 0 || notes.fp_unrecorded > 0 {
            println!(
                "  not recorded in the files (so not checked): the equity table of {} worlds, the seed of {} (seed {} assumed), the decision fingerprint of {}",
                notes.table_unrecorded, notes.seed_unrecorded, job.seed, notes.fp_unrecorded
            );
        }
        let verification = if run.verified.0 == 0 {
            if set.rule == BurnRule::RulesV1 {
                format!(
                    "stored, NOT re-verified on this binary (--verify-pct {}): these values were made by an older binary (values version 1); that the current solver with rules_robber_burn gives the same bytes is not confirmed",
                    sjob.verify_pct
                )
            } else {
                format!(
                    "stored, NOT re-verified on this binary (--verify-pct {})",
                    sjob.verify_pct
                )
            }
        } else if run.verified.1 == 0 {
            format!(
                "stored; {} worlds recomputed by this binary (--verify-pct {}, {} threads a solve) and byte-identical",
                run.verified.0, sjob.verify_pct, sjob.exact_solve_threads
            )
        } else {
            format!(
                "stored; {} of {} recomputed worlds DIFFER from this binary's (decision:world {}): this set's stored values are not the current solver's{}, so H_{cut} was not compared with them",
                run.verified.1,
                run.verified.0,
                run.mismatched
                    .iter()
                    .map(|(d, w)| format!("{d}:{w}"))
                    .collect::<Vec<_>>()
                    .join(","),
                if sjob.rules_robber_burn {
                    " with rules_robber_burn"
                } else {
                    ""
                }
            )
        };
        println!("  exact values: {verification}");
        let mut doc = serde_json::json!({
            "set": si + 1,
            "burn_rule": set.rule.name(),
            "values_version": set.values,
            "rules_robber_burn": sjob.rules_robber_burn,
            "threads": sjob.threads,
            "plan": notes,
            "exact_values": verification,
            "verification": {
                "verify_pct": sjob.verify_pct,
                "solve_threads": sjob.exact_solve_threads,
                "tt_mb": sjob.exact_tt_mb,
                "recomputed": run.verified.0,
                "differ": run.verified.1,
                "mismatched": run.mismatched,
                "done": run.verified.0 > 0,
            },
        });
        if !run.mismatched.is_empty() {
            set_docs.push(doc);
            continue;
        }
        let saved_of = |id: u64, w: u64| {
            set.worlds
                .iter()
                .find(|s| s.id == id && s.w == w)
                .map(|s| s.found.clone())
                .unwrap_or_default()
        };
        println!(
            "  per world: the decider's team's money chance for each candidate, exact | H_{cut} (%), the largest |H - exact| over the four teams; H_{cut}'s time on one thread, nodes and network evaluations; the saved exact time and nodes (as recorded by each run, with its own threads per world)"
        );
        let mut worlds_json = Vec::new();
        for d in &run.decisions {
            for wc in &d.world_checks {
                let saved = saved_of(d.id, wc.w);
                let ch = |v: &[equity::MoneyChance]| {
                    v.iter()
                        .map(|m| {
                            format!("{:.3}", 100.0 * m.get(d.team) as f64 / equity::ONE as f64)
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                let worst = wc
                    .exact
                    .iter()
                    .zip(&wc.hybrid)
                    .flat_map(|(e, h)| (0..4u8).map(move |t| h.get(t).abs_diff(e.get(t))))
                    .max()
                    .unwrap_or(0);
                let saved_txt = saved
                    .iter()
                    .map(|s| {
                        format!(
                            "{} {}{}",
                            s.from,
                            s.us.map_or("-".to_string(), |u| format!("{:.1} s", u as f64 / 1e6)),
                            s.nodes
                                .map_or(String::new(), |n| format!(" {:.3e} nodes", n as f64))
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ");
                println!(
                    "  d{} w{} pos {} S{}{}: exact {} | H {} | max|e| {}; H_{cut} {:.2} s, {:.3e} nodes (tricks 1-5: {:?}), {} evals{}; saved exact: {saved_txt}",
                    d.id,
                    wc.w,
                    d.position,
                    d.stratum,
                    wc.robber
                        .map_or(String::new(), |s| format!(" (seat {s} robs later)")),
                    ch(&wc.exact),
                    ch(&wc.hybrid),
                    pct(worst as f64 / equity::ONE as f64),
                    wc.hybrid_us as f64 / 1e6,
                    wc.hybrid_nodes as f64,
                    wc.hybrid_nodes_by_trick,
                    wc.frontier,
                    match wc.verified {
                        Some(true) => "; exact recomputed: identical",
                        Some(false) => "; exact recomputed: DIFFERS",
                        None => "",
                    }
                );
                let raw = |v: &[equity::MoneyChance]| v.iter().map(|m| m.0).collect::<Vec<_>>();
                let wj = serde_json::json!({
                    "id": d.id, "w": wc.w, "position": d.position, "stratum": d.stratum,
                    "team": d.team, "cands": d.cands, "robber": wc.robber,
                    "exact": raw(&wc.exact), "hybrid": raw(&wc.hybrid),
                    "max_abs_error": worst,
                    "exact_solved": wc.exact_solved, "verified": wc.verified,
                    "exact_us": wc.exact_us, "exact_nodes": wc.exact_nodes,
                    "hybrid_us": wc.hybrid_us, "hybrid_nodes": wc.hybrid_nodes,
                    "hybrid_nodes_by_trick": wc.hybrid_nodes_by_trick,
                    "net_evals": wc.frontier, "hybrid_tt_bytes": wc.hybrid_tt_bytes,
                    "saved": saved,
                });
                if hard.contains(&(d.id, wc.w)) {
                    hard_lines.push(serde_json::json!({ "set": si + 1, "burn_rule": set.rule.name(), "world": wj.clone() }));
                }
                worlds_json.push(wj);
            }
        }
        println!(
            "  per decision (choices are candidate indices; loss in the decider's team's money chance):"
        );
        for d in &run.decisions {
            println!(
                "  d{} pos {} S{}: {} worlds, {} candidates; exact choice {}, H_{cut} choice {}, loss {}{}",
                d.id,
                d.position,
                d.stratum,
                d.worlds,
                d.cands,
                d.exact_choice,
                d.hybrid_choice,
                pct(d.loss as f64 / equity::ONE as f64),
                if d.agree { " (agree)" } else { "" }
            );
        }
        let groups = report(&run, r.boots, r.boot_seed);
        print_groups(&groups);
        let time = check_time(&run, sjob.threads, wall);
        print_time(&time);
        let (pass, line) = verdict(&groups);
        println!("{line}");
        let o = doc.as_object_mut().expect("an object");
        o.insert(
            "values".into(),
            serde_json::json!({
                "exact": set.values,
                "hybrid": format!("{}+O{cut}:{}", search::measure::VALUES_VERSION, r.oracle.sha256),
            }),
        );
        o.insert(
            "groups".into(),
            groups
                .iter()
                .map(|(n, s)| serde_json::json!({"group": n, "summary": s}))
                .collect(),
        );
        o.insert("pass".into(), serde_json::json!(pass));
        o.insert("time".into(), time_json(&time, 0.0));
        o.insert("decisions".into(), serde_json::json!(run.decisions));
        o.insert("worlds".into(), serde_json::json!(worlds_json));
        set_docs.push(doc);
    }
    println!();
    for h in &hard {
        let lines: Vec<&serde_json::Value> = hard_lines
            .iter()
            .filter(|l| l["world"]["id"] == h.0 && l["world"]["w"] == h.1)
            .collect();
        if lines.is_empty() {
            println!("hard world d{}:w{}: not among the checked worlds", h.0, h.1);
        }
        for l in lines {
            let w = &l["world"];
            let hs = w["hybrid_us"].as_u64().unwrap_or(0) as f64 / 1e6;
            let saved = w["saved"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|s| {
                    let us = s["us"].as_u64();
                    format!(
                        "{}: {}{}{}",
                        s["from"].as_str().unwrap_or("?"),
                        us.map_or("time not recorded".to_string(), |u| format!(
                            "{:.0} s",
                            u as f64 / 1e6
                        )),
                        s["nodes"]
                            .as_u64()
                            .map_or(String::new(), |n| format!(", {:.3e} nodes", n as f64)),
                        us.filter(|_| hs > 0.0).map_or(String::new(), |u| format!(
                            " ({:.1} times H_{cut}'s time)",
                            u as f64 / 1e6 / hs
                        ))
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            println!(
                "hard world d{}:w{} ({}): H_{cut} {hs:.1} s on one thread, {:.3e} nodes, {} network evaluations, largest |H - exact| {}; saved exact: {saved}",
                h.0,
                h.1,
                l["burn_rule"].as_str().unwrap_or("?"),
                w["hybrid_nodes"].as_u64().unwrap_or(0) as f64,
                w["net_evals"],
                pct(w["max_abs_error"].as_u64().unwrap_or(0) as f64 / equity::ONE as f64)
            );
        }
    }
    let peak = search::t1bench::proc_peak_rss_mb();
    let max_tt = set_docs
        .iter()
        .flat_map(|d| d["worlds"].as_array().into_iter().flatten())
        .filter_map(|w| w["hybrid_tt_bytes"].as_u64())
        .max()
        .unwrap_or(0);
    println!(
        "memory: peak resident {} MB for {} threads (largest hybrid table {:.0} MB of {} MB allowed)",
        peak.map_or("?".to_string(), |m| m.to_string()),
        job.threads,
        max_tt as f64 / f64::from(1 << 20),
        job.tt_mb
    );
    if let Some(out) = a.get("--json") {
        let v = serde_json::json!({
            "trick": trick,
            "cut": cut,
            "seed": job.seed,
            "world_source": src_name,
            "saved_files": files,
            "only": only,
            "net": net_record(r.net_path, r.oracle),
            "equity_sha256": r.table_sha,
            "binary_sha256": binary_sha256(),
            "boot": {"replicates": r.boots, "seed": r.boot_seed, "bound": "97.5th percentile of bootstrap means (upper end of the 95% interval)"},
            "threads": job.threads,
            "tt_mb": job.tt_mb,
            "peak_rss_mb": peak,
            "wall_s": t1.elapsed().as_secs_f64(),
            "setup_s": t1.duration_since(r.t0).as_secs_f64(),
            "hard": hard_lines,
            "both_rules": both.iter().map(|(d, w, same)| serde_json::json!({"id": d, "w": w, "identical": same})).collect::<Vec<_>>(),
            "verify_rule": verify_rule,
            "sets": set_docs,
        });
        write_json(out, &v)?;
    }
    Ok(())
}

/// `threads` shared out among checks of `n[i]` worlds: one each first, then one at a time to
/// the check with the most worlds per thread so far (ties to the first), never more than a
/// check's worlds. Every check gets at least one thread, even when that makes more than
/// `threads`.
fn split_threads(threads: usize, n: &[usize]) -> Vec<usize> {
    let mut t = vec![1usize; n.len()];
    for _ in n.len()..threads {
        let best = (0..n.len()).filter(|&i| t[i] < n[i]).max_by(|&i, &j| {
            // n[i] / t[i] against n[j] / t[j], exactly; ties to the first.
            (n[i] * t[j]).cmp(&(n[j] * t[i])).then(j.cmp(&i))
        });
        match best {
            Some(i) => t[i] += 1,
            None => break,
        }
    }
    t
}

/// `hybrid-check --explain IDS`: why the hybrid and exact search choose differently on the
/// decisions `IDS` (`search::hybrid_explain`), instead of the check. Exact values are recomputed
/// (and must equal the stored ones when `--worlds` is given); `--explain-worlds N` (default 3)
/// details the N most misleading worlds of each decision; `--explain-json FILE` writes every
/// world (the principal positions of those N only).
fn hybrid_explain(
    a: &Args,
    ids: &str,
    ds: &[DecisionSpec],
    stored: &BTreeMap<u64, WorldRows>,
    job: &search::hybrid::CheckJob,
    ctx: &Ctx,
    oracle: &nn::Oracle,
) -> Result<(), String> {
    use search::hybrid_explain::{
        ExplainJob, explain, pooled_frontier_error, render, render_frontier_error,
    };
    let ids = parse_ids(ids)?;
    let chosen: Vec<DecisionSpec> = ds.iter().filter(|d| ids.contains(&d.id)).cloned().collect();
    if chosen.len() != ids.len() {
        return Err(format!(
            "--explain: {} of the {} ids are among the decisions given",
            chosen.len(),
            ids.len()
        ));
    }
    let xjob = ExplainJob {
        cut: job.cut,
        seed: job.seed,
        label: job.label,
        to: job.to,
        threads: job.threads,
        tt_mb: job.tt_mb,
    };
    let t = Instant::now();
    let mut xs = explain(&chosen, &xjob, ctx, oracle, Some(oracle))?;
    for x in &xs {
        if let Some(r) = stored.get(&x.id) {
            for w in &x.worlds {
                if r.rows.get(w.world as usize) != Some(&w.exact) {
                    return Err(format!(
                        "decision {} world {}: the stored exact values differ from this binary's",
                        x.id, w.world
                    ));
                }
            }
        }
    }
    let top = a.num("--explain-worlds", 3)? as usize;
    println!(
        "H_{} vs exact, explained: {} decisions x {} worlds (seed {}, label {}, {} source); net {} {:?}; {:.1} s on {} threads{}",
        job.cut,
        xs.len(),
        job.to,
        job.seed,
        job.label,
        ctx.source.name(),
        &oracle.sha256[..12],
        oracle.net().widths(),
        t.elapsed().as_secs_f64(),
        job.threads,
        if stored.is_empty() {
            ""
        } else {
            "; exact values equal to the stored ones"
        }
    );
    println!(
        "network error at the hybrid's frontier positions, all explained decisions: {}",
        render_frontier_error(&pooled_frontier_error(&xs))
    );
    for x in &xs {
        println!();
        for l in render(x, top) {
            println!("{l}");
        }
    }
    if let Some(out) = a.get("--explain-json") {
        for x in &mut xs {
            x.keep_lines(top);
        }
        let v = serde_json::json!({
            "cut": job.cut,
            "seed": job.seed,
            "label": job.label,
            "worlds": job.to,
            "net_sha256": oracle.sha256,
            "binary_sha256": binary_sha256(),
            "decisions": xs,
        });
        std::fs::write(
            out,
            serde_json::to_string(&v).map_err(|e| e.to_string())? + "\n",
        )
        .map_err(|e| format!("{out}: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use search::oracle_labels::{HarvestDecision, HarvestJob, position_schedule};

    /// Threads are shared out by worlds, never more than a set's worlds, at most `threads` in
    /// all (each set at least one).
    #[test]
    fn threads_are_shared_by_worlds() {
        assert_eq!(split_threads(184, &[31, 355]), vec![15, 169]);
        assert_eq!(split_threads(184, &[31]), vec![31]);
        assert_eq!(split_threads(4, &[1, 9]), vec![1, 3]);
        assert_eq!(split_threads(1, &[5, 5]), vec![1, 1]);
        assert_eq!(split_threads(8, &[2, 2]), vec![2, 2]);
        assert_eq!(split_threads(3, &[]), Vec::<usize>::new());
    }

    fn decision(position: u8, distinct: u32, evals: u64, rows: u32) -> HarvestDecision {
        HarvestDecision {
            id: 0,
            trick: 2,
            position,
            stratum: 1,
            cands: 3,
            evals,
            distinct,
            rows,
            search_nodes: 10,
            race_distinct: distinct / 4,
        }
    }

    /// The harvest sidecar names its driver: `exact`, or `net` with the network's SHA-256, file,
    /// trick, head and widths (the network drove the search); and gives the mean positions
    /// reached and evaluations per decision for each seat.
    #[test]
    fn harvest_record_names_the_driver() {
        let h = HarvestJob {
            trick: 3,
            count: 40,
            seed: 7,
            threads: 4,
            cap: 16,
            tricks: vec![2],
            schedule: position_schedule([1; 8]),
            strata: vec![1, 2, 3, 4],
            race: None,
        };
        let ds = [
            decision(1, 2000, 2500, 16),
            decision(1, 1000, 1100, 16),
            decision(2, 300, 310, 8),
        ];
        let net = nn::synthetic::random_net(5, &[16, 8], nn::outcome::vocab(3), 100);
        let meta = BTreeMap::from([("head".to_string(), "outcome-v1".to_string())]);
        let bytes = nn::oracle::write_net(&net, 3, &meta);
        let o =
            nn::Oracle::from_bytes(&bytes, Arc::new(nn::synthetic::SyntheticLeaf)).expect("loads");
        let shares = "0.25,0.25,0.5";
        let w = [1; 8];
        let r = harvest_record(
            &h,
            shares,
            &w,
            &ds,
            64,
            Some((&o, "/runs/o3/int.safetensors")),
        );
        let d = &r["frontier_driver"];
        assert_eq!(d["kind"], "net");
        assert_eq!(d["drove_the_search"], true);
        assert_eq!(d["net_sha256"], equity::sha256::sha256_hex(&bytes).as_str());
        assert_eq!(
            (
                d["net_file"].as_str(),
                d["net_trick"].as_u64(),
                d["net_head"].as_str()
            ),
            (Some("int.safetensors"), Some(3), Some("outcome-v1"))
        );
        assert_eq!(d["net_widths"], serde_json::json!([16, 8, 24]));
        assert!(
            r["frontier_values"]
                .as_str()
                .is_some_and(|v| v.contains(&o.sha256) && v.contains("solved exactly")),
            "{r}"
        );
        assert!(
            r["what"]
                .as_str()
                .is_some_and(|v| v.contains("only the sampled"))
        );
        assert_eq!(
            r["distinct_mean_by_position"],
            serde_json::json!([1500.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(
            r["evaluations_mean_by_position"],
            serde_json::json!([1800.0, 310.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(
            r["decisions_by_position"],
            serde_json::json!([2, 1, 0, 0, 0, 0, 0, 0])
        );
        assert_eq!(
            r["rows_by_position"],
            serde_json::json!([32, 8, 0, 0, 0, 0, 0, 0])
        );
        let e = harvest_record(&h, shares, &w, &ds, 64, None);
        assert_eq!(e["frontier_driver"], serde_json::json!({ "kind": "exact" }));
        assert!(
            e["frontier_values"]
                .as_str()
                .is_some_and(|v| v.starts_with("exact"))
        );
        assert_eq!(
            e["distinct_mean_by_position"],
            r["distinct_mean_by_position"]
        );
        assert_eq!(e["strata"], serde_json::json!([1, 2, 3, 4]));
        assert_eq!(e["race"], serde_json::Value::Null);
        assert_eq!(e["decisions_capped"], 3);
        // A race harvest from stratum 4: its settings and the race positions reached.
        let race = HarvestJob {
            strata: vec![4],
            race: Some(search::oracle_labels::RaceKeep {
                classes: 0b10100,
                cap: 256,
            }),
            cap: 0,
            ..h
        };
        let x = harvest_record(&race, shares, &w, &ds, 64, None);
        assert_eq!(x["strata"], serde_json::json!([4]));
        assert_eq!(x["race"]["classes"], serde_json::json!([2, 4]));
        assert_eq!(x["race"]["cap_per_decision"], 256);
        assert_eq!(x["race"]["other_classes_cap_per_decision"], 0);
        assert_eq!(x["race"]["race_positions_reached"], 500 + 250 + 75);
        assert_eq!(x["race"]["decisions_race_capped"], 1);
        assert_eq!(x["decisions_capped"], 0);
        assert!(
            x["decision_generator"]
                .as_str()
                .is_some_and(|g| g.contains("strata[j mod s]"))
        );
    }
}
