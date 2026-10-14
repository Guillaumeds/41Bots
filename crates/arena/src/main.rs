//! `arena`: duplicate-block tests between bots (docs/M1-SPEC.md §5.4).
//!
//! ```text
//! arena sprt --a <bot> --b <bot> [--block14|--rot4] [--s0 0 --s1 0.02 --alpha 0.05 --beta 0.05]
//!            [--max-blocks N | --sigma-b X] [--stage1-blocks N] [--threads N] [--seed S] --out <dir>
//! arena canary [--bot rules-v1] [--blocks 50] [--threads N] [--seed S]
//! arena measure-sigma --a <bot> [--b <bot>] [--blocks 200] [--rot4] [--split-seeds] [--threads N] [--seed S] [--out <dir>]
//! arena rate <records.jsonl>...
//! ```
//!
//! Two-stage design (an arena choice): with `--block14` (the default) stage 1 plays 4-rotation
//! blocks and can only reject; if it does not reject, stage 2 plays fresh 14-way blocks and
//! decides. With `--rot4` there is a single 4-rotation stage that decides. The cap of each stage
//! is `--max-blocks`, else 5 × 8.67 (σ_b/Δ)² from `--sigma-b` (default [`DEFAULT_SIGMA_B`]);
//! stage 1's cap defaults to a quarter of that (`--stage1-blocks`).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use arena::block::BlockKind;
use arena::rating::rate;
use arena::run::{RunConfig, StageOutcome, Summary, run_blocks, sprt_stage};
use arena::sprt::{Decision, SprtParams};
use engine::game::Rules;
use serde_json::json;
use tablemgr::{RecordReader, RecordWriter};

/// Default σ_b (money chance, 14-way block) for the cap when none is given: the measured
/// rules-v1 vs random value, rounded up (see the M1 status notes).
const DEFAULT_SIGMA_B: f64 = 0.15;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("arena: {e}");
            ExitCode::FAILURE
        }
    }
}

struct Flags {
    kv: Vec<(String, String)>,
    switches: Vec<String>,
    positional: Vec<String>,
}

const SWITCHES: [&str; 3] = ["block14", "rot4", "split-seeds"];

impl Flags {
    fn parse(args: &[String]) -> Result<Flags, String> {
        let mut f = Flags {
            kv: vec![],
            switches: vec![],
            positional: vec![],
        };
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if let Some(k) = a.strip_prefix("--") {
                if SWITCHES.contains(&k) {
                    f.switches.push(k.to_string());
                } else {
                    let v = it.next().ok_or_else(|| format!("--{k} needs a value"))?;
                    f.kv.push((k.to_string(), v.clone()));
                }
            } else {
                f.positional.push(a.clone());
            }
        }
        Ok(f)
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.kv
            .iter()
            .rev()
            .find(|(a, _)| a == k)
            .map(|(_, v)| v.as_str())
    }
    fn num<T: std::str::FromStr>(&self, k: &str, d: T) -> Result<T, String> {
        match self.get(k) {
            None => Ok(d),
            Some(v) => v.parse().map_err(|_| format!("--{k}: cannot read `{v}`")),
        }
    }
    fn has(&self, k: &str) -> bool {
        self.switches.iter().any(|s| s == k)
    }
    fn threads(&self) -> Result<usize, String> {
        let d = std::thread::available_parallelism().map_or(1, |n| n.get());
        self.num("threads", d)
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let (cmd, rest) = args
        .split_first()
        .ok_or("usage: arena sprt|canary|measure-sigma|rate …")?;
    let f = Flags::parse(rest)?;
    match cmd.as_str() {
        "sprt" => sprt(&f),
        "canary" => canary(&f),
        "measure-sigma" => measure_sigma(&f),
        "rate" => rate_cmd(&f),
        _ => Err(format!("unknown command `{cmd}`")),
    }
}

fn stage_json(name: &str, kind: BlockKind, o: &StageOutcome, p: &SprtParams) -> serde_json::Value {
    json!({
        "stage": name, "kind": kind, "decision": o.decision, "llr": o.llr,
        "bounds": [p.lower(), p.upper()], "summary": o.summary.report(),
    })
}

fn stage_text(name: &str, o: &StageOutcome) -> String {
    let r = o.summary.report();
    let mut s = format!(
        "{name}: {} blocks, mean {:+.4} (money chance), block sd {:.4}, LLR {:.3}, decision {:?}\n",
        r.blocks, r.mean, r.sd, o.llr, o.decision
    );
    s += &format!(
        "  in stakes: mean {:+.4}, block sd {:.4}\n",
        r.stake_mean, r.stake_sd
    );
    for m in &r.mix {
        s += &format!(
            "  {}: {} matches, mean per match {:+.4}\n",
            m.mix, m.matches, m.mean
        );
    }
    let (a, b) = (&r.diagnostics.a, &r.diagnostics.b);
    s += &format!(
        "  diagnostics per slot-match (A / B): match wins {:.3} / {:.3}, games {:.3} / {:.3}, points {:.1} / {:.1}, tricks {:.2} / {:.2}\n",
        a.match_win_rate,
        b.match_win_rate,
        a.games_per_match,
        b.games_per_match,
        a.points_per_match,
        b.points_per_match,
        a.tricks_per_match,
        b.tricks_per_match
    );
    s
}

fn writer(dir: &Path, name: &str) -> Result<RecordWriter<std::fs::File>, String> {
    let path = dir.join(name);
    let _ = std::fs::remove_file(&path);
    RecordWriter::append(&path).map_err(|e| format!("{}: {e}", path.display()))
}

fn sprt(f: &Flags) -> Result<(), String> {
    let a = f.get("a").ok_or("--a is required")?.to_string();
    let b = f.get("b").ok_or("--b is required")?.to_string();
    let out = PathBuf::from(f.get("out").ok_or("--out is required")?);
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let p = SprtParams {
        s0: f.num("s0", 0.0)?,
        s1: f.num("s1", 0.02)?,
        alpha: f.num("alpha", 0.05)?,
        beta: f.num("beta", 0.05)?,
    };
    if p.s1.partial_cmp(&p.s0) != Some(std::cmp::Ordering::Greater) {
        return Err("--s1 must be above --s0".into());
    }
    let sigma: f64 = f.num("sigma-b", DEFAULT_SIGMA_B)?;
    let max_blocks: u64 = f.num("max-blocks", p.default_max_blocks(sigma))?;
    let stage1_blocks: u64 = f.num("stage1-blocks", (max_blocks / 4).max(20))?;
    let rot4_only = f.has("rot4");
    let seed: u64 = f.num("seed", 1)?;
    let cfg = |kind, stage| RunConfig {
        kind,
        a: a.clone(),
        b: b.clone(),
        rules: Rules::HOUSE,
        master_seed: seed,
        stage,
        threads: 1,
        split_seeds: false,
    };
    let threads = f.threads()?;
    let t0 = Instant::now();
    let mut stages = Vec::new();
    let mut files = Vec::new();
    let mut matches = 0u64;
    let plan: Vec<(&str, BlockKind, bool, u64)> = if rot4_only {
        vec![("rot4", BlockKind::Rot4, false, max_blocks)]
    } else {
        vec![
            ("stage1-rot4", BlockKind::Rot4, true, stage1_blocks),
            ("stage2-block14", BlockKind::Block14, false, max_blocks),
        ]
    };
    let mut final_decision = Decision::Inconclusive;
    for (i, (name, kind, reject_only, cap)) in plan.into_iter().enumerate() {
        let file = format!("records-{name}.jsonl");
        let mut w = writer(&out, &file)?;
        let mut c = cfg(kind, i as u64);
        c.threads = threads;
        let o = sprt_stage(&c, &p, reject_only, cap, |r| {
            matches += 1;
            w.write(r).map_err(|e| e.to_string())
        })?;
        w.flush().map_err(|e| e.to_string())?;
        files.push(out.join(&file).display().to_string());
        final_decision = o.decision;
        let stop = reject_only && o.decision == Decision::Reject;
        if reject_only && o.decision == Decision::Inconclusive {
            final_decision = Decision::Inconclusive;
        }
        stages.push((name, kind, o));
        if stop {
            break;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let report = json!({
        "command": "sprt", "a": a, "b": b, "seed": seed, "params": p, "max_blocks": max_blocks,
        "stages": stages.iter().map(|(n, k, o)| stage_json(n, *k, o, &p)).collect::<Vec<_>>(),
        "decision": final_decision, "matches": matches,
        "timing": {"seconds": secs, "matches_per_second": matches as f64 / secs.max(1e-9)},
        "records": files,
    });
    let mut text = format!(
        "SPRT {a} (A) against {b} (B), hypotheses [{}, {}], alpha {}, beta {}, cap {max_blocks} blocks\n",
        p.s0, p.s1, p.alpha, p.beta
    );
    for (n, _, o) in &stages {
        text += &stage_text(n, o);
    }
    text += &format!(
        "Decision: {final_decision:?}. {matches} matches in {secs:.1} s ({:.1} matches/s).\nRecords: {}\n",
        matches as f64 / secs.max(1e-9),
        files.join(", ")
    );
    write_reports(&out, "sprt", &report, &text)
}

fn write_reports(
    out: &Path,
    stem: &str,
    report: &serde_json::Value,
    text: &str,
) -> Result<(), String> {
    let j = out.join(format!("{stem}-report.json"));
    std::fs::write(
        &j,
        serde_json::to_string_pretty(report).expect("json") + "\n",
    )
    .map_err(|e| format!("{}: {e}", j.display()))?;
    let t = out.join(format!("{stem}-report.txt"));
    std::fs::write(&t, text).map_err(|e| format!("{}: {e}", t.display()))?;
    print!("{text}");
    Ok(())
}

fn canary(f: &Flags) -> Result<(), String> {
    let bot = f.get("bot").unwrap_or(bots::RULES_V1).to_string();
    let blocks: u64 = f.num("blocks", 50)?;
    let cfg = RunConfig {
        kind: BlockKind::Block14,
        a: bot.clone(),
        b: bot.clone(),
        rules: Rules::HOUSE,
        master_seed: f.num("seed", 1)?,
        stage: 0,
        threads: f.threads()?,
        split_seeds: false,
    };
    let s = arena::run::canary(&cfg, blocks)?;
    println!(
        "canary: {bot} against itself, {} blocks in all 14 assignments: every block score is exactly 0 (mean {:.3}).",
        s.chance.n,
        s.chance_mean_sd().0
    );
    Ok(())
}

fn measure_sigma(f: &Flags) -> Result<(), String> {
    let a = f.get("a").ok_or("--a is required")?.to_string();
    let b = f.get("b").unwrap_or(&a).to_string();
    let blocks: u64 = f.num("blocks", 200)?;
    let kind = if f.has("rot4") {
        BlockKind::Rot4
    } else {
        BlockKind::Block14
    };
    let cfg = RunConfig {
        kind,
        a: a.clone(),
        b: b.clone(),
        rules: Rules::HOUSE,
        master_seed: f.num("seed", 1)?,
        stage: 0,
        threads: f.threads()?,
        split_seeds: f.has("split-seeds"),
    };
    let t0 = Instant::now();
    let mut s = Summary::default();
    let mut w = match f.get("out") {
        Some(d) => {
            std::fs::create_dir_all(d).map_err(|e| format!("{d}: {e}"))?;
            Some(writer(Path::new(d), "records-measure-sigma.jsonl")?)
        }
        None => None,
    };
    let mut werr = None;
    run_blocks(&cfg, blocks, |_, r| {
        s.push(r);
        if let Some(w) = w.as_mut() {
            for rec in &r.records {
                if let Err(e) = w.write(rec) {
                    werr = Some(e.to_string());
                    return false;
                }
            }
        }
        true
    })?;
    if let Some(e) = werr {
        return Err(e);
    }
    let secs = t0.elapsed().as_secs_f64();
    let r = s.report();
    let matches = r.blocks * kind.assignments().len() as u64;
    let text = format!(
        "sigma_b: {a} (A) against {b} (B), {kind:?}, {} blocks{}\n  mean {:+.4}, sigma_b {:.4} (money chance); in stakes mean {:+.4}, sigma_b {:.4}\n  standard error of the mean {:.4}; {matches} matches in {secs:.1} s ({:.1} matches/s)\n",
        r.blocks,
        if cfg.split_seeds {
            ", B seeds split"
        } else {
            ""
        },
        r.mean,
        r.sd,
        r.stake_mean,
        r.stake_sd,
        r.sd / (r.blocks as f64).sqrt(),
        matches as f64 / secs.max(1e-9),
    );
    let report = json!({"command": "measure-sigma", "a": a, "b": b, "kind": kind,
        "split_seeds": cfg.split_seeds, "summary": r,
        "timing": {"seconds": secs, "matches_per_second": matches as f64 / secs.max(1e-9)}});
    match f.get("out") {
        Some(d) => write_reports(Path::new(d), "measure-sigma", &report, &text),
        None => {
            print!("{text}");
            println!("{}", serde_json::to_string(&report).expect("json"));
            Ok(())
        }
    }
}

fn rate_cmd(f: &Flags) -> Result<(), String> {
    if f.positional.is_empty() {
        return Err("arena rate <records.jsonl>...".into());
    }
    let mut recs = Vec::new();
    for p in &f.positional {
        let rd = RecordReader::open(p).map_err(|e| format!("{p}: {e}"))?;
        for r in rd {
            recs.push(r.map_err(|e| format!("{p}: {e}"))?);
        }
    }
    let ratings = rate(&recs);
    println!(
        "Plackett-Luce ratings from {} records (r in natural-log units, anchor r = 0):",
        recs.len()
    );
    for r in &ratings {
        println!(
            "  {:<24} r = {:>8} ± {:<7} ({} wins of {} team entries)",
            r.program,
            r.r.map_or("n/a".into(), |x| format!("{x:+.3}")),
            r.se.map_or("-".into(), |x| format!("{x:.3}")),
            r.wins,
            r.entries
        );
    }
    println!("{}", serde_json::to_string(&ratings).expect("json"));
    Ok(())
}
