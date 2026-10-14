//! `search`: the M3 PIMC measurement and teacher tools (docs/M3-SPEC.md §5.8–5.10, §13.1).
//!
//! ```text
//! search decisions --trick T --count N --out FILE [--seed S] [--start-hands H]
//! search worlds    --decisions FILE --method exact|rollout --to N --out FILE
//!                  [--from-world A] [--label L] [--seed S] [--first I] [--last J]
//!                  [--threads K] [--solve-threads K] [--tt-mb M] --equity TABLE
//! search agree     --worlds FILE [--grid 10,20,40,80,160,320]
//! search score     --yardstick FILE --n N --method FILE:N [--method FILE:N …]
//! search teacher   --hands N --out DIR --equity TABLE [--seed S] [--config NAME] [--start-hands H]
//! ```
//! `worlds` is the yardstick builder and the method runner: it appends one JSON line per decision
//! to `--out` and skips decisions already there, so an interrupted chunk resumes. Exact worlds are
//! the yardstick (label 1); a method's own worlds use another label. `agree` and `score` report
//! per trick and per position in the trick (§13.1 item 6).

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::process::ExitCode;
use std::sync::Arc;

use search::bot::{PimcFactory, PimcSpec, load_equity};
use search::measure::{
    DecisionSpec, WorldRows, agreement_curve, build_decisions, collect_starts, paired, score,
    stratum_of, summarise, world_rows,
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
    let f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut out = Vec::new();
    for l in std::io::BufReader::new(f).lines() {
        let l = l.map_err(|e| e.to_string())?;
        if l.trim().is_empty() {
            continue;
        }
        // A torn last line (killed mid-write) is skipped; the decision is redone.
        if let Ok(v) = serde_json::from_str(&l) {
            out.push(v);
        }
    }
    Ok(out)
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

fn worlds(a: &Args) -> Result<(), String> {
    let ds: Vec<DecisionSpec> = read_lines(a.req("--decisions")?)?;
    let method = match a.req("--method")? {
        "exact" => Method::ExactPimc,
        "rollout" => Method::RolloutPimc,
        m => return Err(format!("unknown method {m}")),
    };
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
    let label = a.num("--label", 1)?;
    let seed = a.num("--seed", 1)?;
    let (first, last) = (a.num("--first", 0)?, a.num("--last", u64::MAX)?);
    let out = a.req("--out")?;
    let done: Vec<WorldRows> = read_lines(out).unwrap_or_default();
    // Rewrite the file without a torn line, then append.
    let mut text = String::new();
    for r in &done {
        text.push_str(&serde_json::to_string(r).map_err(|e| e.to_string())?);
        text.push('\n');
    }
    std::fs::write(out, text).map_err(|e| e.to_string())?;
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(out)
        .map_err(|e| e.to_string())?;
    for d in ds.iter().filter(|d| d.id >= first && d.id <= last) {
        if done.iter().any(|r| r.id == d.id) {
            continue;
        }
        let r = world_rows(d, method, label, seed, from, to, &ctx, &cfg);
        eprintln!(
            "decision {} trick {} pos {} cands {}: {} ms ({:.1} ms/world)",
            d.id,
            d.trick,
            d.position + 1,
            r.cands.len(),
            r.ms,
            r.ms as f64 / (to - from) as f64
        );
        writeln!(
            f,
            "{}",
            serde_json::to_string(&r).map_err(|e| e.to_string())?
        )
        .map_err(|e| e.to_string())?;
        f.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn grid(a: &Args) -> Result<Vec<usize>, String> {
    a.get("--grid")
        .unwrap_or("10,20,40,80,160,320")
        .split(',')
        .map(|x| x.parse().map_err(|_| format!("bad grid {x}")))
        .collect()
}

fn agree(a: &Args) -> Result<(), String> {
    let rows: Vec<WorldRows> = read_lines(a.req("--worlds")?)?;
    let g = grid(a)?;
    let mut groups: BTreeMap<(u8, String), Vec<WorldRows>> = BTreeMap::new();
    for r in &rows {
        groups
            .entry((r.trick, "all".into()))
            .or_default()
            .push(r.clone());
        groups
            .entry((r.trick, format!("pos{}", r.position + 1)))
            .or_default()
            .push(r.clone());
        groups
            .entry((
                r.trick,
                format!("pair{}-{}", r.position / 2 * 2 + 1, r.position / 2 * 2 + 2),
            ))
            .or_default()
            .push(r.clone());
    }
    let mut report = serde_json::Map::new();
    for ((trick, group), rs) in &groups {
        let (steps, plateau) = agreement_curve(rs, &g);
        let ms: f64 = rs
            .iter()
            .map(|r| r.ms as f64 / r.rows.len() as f64)
            .sum::<f64>()
            / rs.len() as f64;
        println!(
            "trick {trick} {group}: {} decisions, {:.1} ms per world, plateau N = {}",
            rs.len(),
            ms,
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
            serde_json::json!({ "decisions": rs.len(), "ms_per_world": ms, "plateau": plateau, "steps": steps }),
        );
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
        let mut by_pos: BTreeMap<u8, Vec<search::measure::Scored>> = BTreeMap::new();
        for x in &s {
            if let Some(y) = yard.iter().find(|y| y.id == x.id) {
                by_pos.entry(y.position).or_default().push(x.clone());
            }
        }
        for (p, v) in &by_pos {
            let ps = summarise(v);
            println!(
                "  pos {}: n={} agree={:.3} loss={:.5}±{:.5}",
                p + 1,
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
