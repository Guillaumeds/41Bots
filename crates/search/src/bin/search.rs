//! `search`: the M3 PIMC measurement and teacher tools (docs/M3-SPEC.md §5.8–5.10, §13.1).
//!
//! ```text
//! search decisions --trick T --count N --out FILE [--seed S] [--start-hands H]
//! search worlds    --decisions FILE --method exact|rollout --to N --out FILE
//!                  [--from-world A] [--label L] [--seed S] [--first I] [--last J]
//!                  [--threads K] [--solve-threads K] [--tt-mb M] --equity TABLE
//!                  [--progress FILE]
//! search agree     --worlds FILE [--grid 10,20,40,80,160,320] [--json FILE]
//! search score     --yardstick FILE --n N --method FILE:N [--method FILE:N …]
//! search teacher   --hands N --out DIR --equity TABLE [--seed S] [--config NAME] [--start-hands H]
//! ```
//! `worlds` is the yardstick builder and the method runner: every (decision, world) task of the
//! run goes through one work queue on `--threads` threads (longest expected first), and one JSON
//! line per decision is appended to `--out` in that order; decisions already there are skipped,
//! so an interrupted chunk resumes. The file is byte-identical for any thread count; timings go to
//! `<out>.times.jsonl`. `--progress FILE` is rewritten every 2 s with the counts done (for the
//! runner's heartbeat). Exact worlds are the yardstick (label 1); a method's own worlds use
//! another label. `agree` reports per trick, per position in the trick (§13.1 item 6), per
//! match-state stratum and per log2 world-count band (item 8) and per hand-type band (item 9);
//! `score` per position and stratum.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use search::bot::{PimcFactory, PimcSpec, load_equity};
use search::measure::{
    BatchProgress, DecisionSpec, DecisionTiming, WorldRows, WorldsJob, agreement_curve,
    build_decisions, collect_starts, paired, read_jsonl, score, stratum_of, summarise, timing_path,
    worlds_file,
};
use search::pimc::{Ctx, Method, SearchConfig, SharedLeaf};
use search::teacher::{decisions_npy, teacher_deck, teacher_hand};
use search::worlds::RejectionSource;

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
        Some("score") => score_cmd(&a),
        Some("teacher") => teacher(&a),
        _ => {
            Err("usage: search decisions|worlds|agree|score|teacher … (see the source docs)".into())
        }
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("search: {e}");
            ExitCode::from(2)
        }
    }
}

fn decisions(a: &Args) -> Result<(), String> {
    let trick = a.num("--trick", 2)? as u8;
    let count = a.num("--count", 200)?;
    let seed = a.num("--seed", 1)?;
    let starts = collect_starts(a.num("--start-hands", 20_000)?, seed);
    let ds = build_decisions(trick, count, seed, &starts);
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
    let src = RejectionSource::new();
    let ctx = Ctx {
        leaf: &*leaf,
        source: &src,
        rules: engine::game::Rules::HOUSE,
    };
    let cfg = SearchConfig {
        threads: a.num("--threads", 1)? as usize,
        solve_threads: a.num("--solve-threads", 1)? as usize,
        tt_mb: a.num("--tt-mb", 256)? as usize,
        ..SearchConfig::default()
    };
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
    let ds: Vec<DecisionSpec> = ds
        .into_iter()
        .filter(|d| d.id >= first && d.id <= last)
        .collect();
    let out = Path::new(a.req("--out")?);
    let progress_path = a.get("--progress").map(PathBuf::from);
    let prog = BatchProgress::default();
    let finished = AtomicBool::new(false);
    eprintln!(
        "worlds: {} decisions, worlds {from}..{to}, {} threads, {} solve threads (table loaded in {:.1} s)",
        ds.len(),
        cfg.threads,
        cfg.solve_threads,
        t0.elapsed().as_secs_f64()
    );
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
        let r = worlds_file(out, &ds, &job, &ctx, &cfg, Some(&prog), |r, t| {
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

fn grid(a: &Args) -> Result<Vec<usize>, String> {
    a.get("--grid")
        .unwrap_or("10,20,40,80,160,320")
        .split(',')
        .map(|x| x.parse().map_err(|_| format!("bad grid {x}")))
        .collect()
}

/// The fewest decisions a reported cell should have (§13.1 item 8).
const MIN_CELL: usize = 25;

fn agree(a: &Args) -> Result<(), String> {
    let path = a.req("--worlds")?;
    let rows: Vec<WorldRows> = read_lines(path)?;
    // Timings (the last record per decision), when the timing file is there.
    let mut times: BTreeMap<u64, DecisionTiming> = BTreeMap::new();
    if let Ok(ts) = read_jsonl::<DecisionTiming>(&timing_path(Path::new(path))) {
        for t in ts {
            times.insert(t.id, t);
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
        _ => match i {
            1 => "plain0-1".to_string(),
            3 => "plain3+".to_string(),
            _ => format!("plain{i}"),
        },
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
                "trick {t} by log2 world count: n/a (the world source cannot count worlds; needs crates/belief)"
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

fn score_cmd(a: &Args) -> Result<(), String> {
    let yard: Vec<WorldRows> = read_lines(a.req("--yardstick")?)?;
    let n = a.num("--n", 80)? as usize;
    let mut scored = Vec::new();
    for m in a.all("--method") {
        let (path, nm) = m.rsplit_once(':').ok_or("--method FILE:N")?;
        let nm: usize = nm.parse().map_err(|_| "--method FILE:N")?;
        let rows: Vec<WorldRows> = read_lines(path)?;
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
        scored.push((m.to_string(), s));
    }
    for i in 0..scored.len() {
        for j in i + 1..scored.len() {
            let p = paired(&scored[i].1, &scored[j].1);
            println!(
                "{} vs {}: {}",
                scored[i].0,
                scored[j].0,
                serde_json::to_string(&p).map_err(|e| e.to_string())?
            );
        }
    }
    Ok(())
}

fn teacher(a: &Args) -> Result<(), String> {
    let hands = a.num("--hands", 10)?;
    let seed = a.num("--seed", 1)?;
    let spec = PimcSpec::parse(a.get("--config").unwrap_or("pimc-teacher"))?;
    let table = load_equity(a.req("--equity")?)?;
    let leaf: Arc<SharedLeaf> = Arc::new(table);
    let f = PimcFactory::new(&spec, leaf, Arc::new(RejectionSource::new()));
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
