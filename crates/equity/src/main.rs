//! `equity`: build, estimate, check and query the match-equity table (docs/M1-SPEC.md §6.4).
//!
//! ```text
//! equity build --model outcome-v0.json --out equity-v0.npy [--threads N] [--table-version N]
//! equity estimate-model --bot rules-v1 --hands N --out outcome-v0.json [--seed S] [--threads N]
//! equity lookup --table equity-v0.npy --state "scores=36,0,0,0 games=1,0,0,0 dealer_team=0"
//! equity check --table equity-v0.npy --model outcome-v0.json [--mc-states 20] [--mc-matches 10000]
//! equity synthetic-model --out synthetic.json
//! ```

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use engine::game::Rules;
use equity::estimate::estimate_model_parallel;
use equity::hand::Scoring;
use equity::model::HandModel;
use equity::registry::{bot_factory, known_bots};
use equity::sha256::sha256_hex;
use equity::state::MatchState;
use equity::sweep::{Scope, SweepOptions, sweep};
use equity::table::{EquityTable, TableInfo};
use equity::validate;

const USAGE: &str = "usage:
  equity build --model <outcome.json> --out <equity-vN.npy> [--threads N] [--table-version N]
  equity estimate-model --bot <name> --hands N --out <outcome.json> [--seed S] [--threads N] [--model-id ID]
  equity lookup --table <equity-vN.npy> --state \"scores=36,0,0,0 games=1,0,0,0 dealer_team=0\" [--stake 5]
  equity check --table <equity-vN.npy> --model <outcome.json> [--mc-states 20] [--mc-matches 10000] [--seed S]
  equity synthetic-model --out <file.json>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("equity: {e}");
            ExitCode::FAILURE
        }
    }
}

struct Flags(Vec<(String, String)>);

impl Flags {
    fn parse(args: &[String]) -> Result<Flags, String> {
        let mut out = Vec::new();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            let key = a
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected argument `{a}`\n{USAGE}"))?;
            let value = it
                .next()
                .ok_or_else(|| format!("--{key} needs a value\n{USAGE}"))?;
            out.push((key.to_string(), value.clone()));
        }
        Ok(Flags(out))
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.0
            .iter()
            .rev()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    fn need(&self, key: &str) -> Result<&str, String> {
        self.get(key)
            .ok_or_else(|| format!("missing --{key}\n{USAGE}"))
    }

    fn number<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T, String> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v
                .replace('_', "")
                .parse()
                .map_err(|_| format!("--{key}: `{v}` is not a number")),
        }
    }

    fn check_known(&self, known: &[&str]) -> Result<(), String> {
        for (k, _) in &self.0 {
            if !known.contains(&k.as_str()) {
                return Err(format!("unknown option --{k}\n{USAGE}"));
            }
        }
        Ok(())
    }
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn run(args: &[String]) -> Result<(), String> {
    let Some((cmd, rest)) = args.split_first() else {
        return Err(USAGE.to_string());
    };
    let flags = Flags::parse(rest)?;
    match cmd.as_str() {
        "build" => build(&flags),
        "estimate-model" => estimate(&flags),
        "lookup" => lookup(&flags),
        "check" => check(&flags),
        "synthetic-model" => synthetic(&flags),
        "help" | "--help" | "-h" => {
            println!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command `{other}`\n{USAGE}")),
    }
}

fn read(path: &str) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

fn load_model(path: &str) -> Result<(HandModel, String), String> {
    let bytes = read(path)?;
    let text = String::from_utf8(bytes.clone()).map_err(|_| format!("{path} is not UTF-8"))?;
    let model = HandModel::from_json(&text).map_err(|e| format!("{path}: {e}"))?;
    Ok((model, sha256_hex(&bytes)))
}

/// `equity-v7.npy` → 7.
fn version_from_name(path: &Path) -> Option<u32> {
    let stem = path.file_stem()?.to_str()?;
    stem.strip_prefix("equity-v")?.parse().ok()
}

fn build(flags: &Flags) -> Result<(), String> {
    flags.check_known(&["model", "out", "threads", "table-version"])?;
    let model_path = flags.need("model")?;
    let out = PathBuf::from(flags.need("out")?);
    let threads = flags.number("threads", default_threads())?;
    let version = flags.number("table-version", version_from_name(&out).unwrap_or(0))?;
    let (model, model_sha) = load_model(model_path)?;
    let compiled = model.compile();
    let rules = Rules::HOUSE;
    let sc = Scoring::HOUSE;
    println!(
        "model {} ({}), sha256 {}…, {} empty distributions backed off",
        model.model_id,
        model.source.bot,
        &model_sha[..16],
        compiled.backoffs
    );
    let start = Instant::now();
    let (table, stats) = sweep(
        &compiled,
        sc,
        &SweepOptions {
            threads,
            scope: Scope::Full,
        },
    );
    let secs = start.elapsed().as_secs_f64();
    println!(
        "sweep: {} states in {} levels (largest {}), {} in-hand positions, {threads} threads, {secs:.2} s wall time",
        stats.states, stats.levels, stats.largest_level, stats.in_hand_positions
    );
    let npy = table.to_npy().map_err(|e| e.to_string())?;
    let file_name = out
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("equity.npy")
        .to_string();
    let info = TableInfo::new(
        version,
        &file_name,
        &npy,
        rules,
        &model,
        &model_sha,
        compiled.backoffs,
        stats,
    );
    write(&out, &npy)?;
    let json_path = out.with_extension("json");
    let json = serde_json::to_string_pretty(&info).map_err(|e| e.to_string())? + "\n";
    write(&json_path, json.as_bytes())?;
    println!(
        "wrote {} ({} bytes, content id {}) and {}",
        out.display(),
        npy.len(),
        info.content_id,
        json_path.display()
    );
    report_quick_checks(&table)
}

/// The cheap whole-table checks, printed after a build and by `check`.
fn report_quick_checks(table: &EquityTable) -> Result<(), String> {
    let n = validate::check_sums(table)?;
    println!("check: all {n} entries sum to exactly 2^32");
    let pairs = validate::check_rotation_symmetry(table, 7)?;
    println!("check: team-rotation symmetry holds ({pairs} state-rotation pairs)");
    match validate::check_sanity(table) {
        Ok(lowest) => println!(
            "check: one game and 36 points against 0-0: chance at least {:.6} (> 0.5)",
            lowest as f64 / equity::ONE as f64
        ),
        Err(e) => println!("CHECK FAILED: sanity: {e}"),
    }
    let mono = validate::check_monotonicity(table);
    if mono.violations == 0 {
        println!(
            "check: monotone in own points ({} adjacent-score pairs)",
            mono.pairs
        );
    } else {
        println!(
            "CHECK FAILED: monotonicity: {} of {} pairs fall, largest drop {} / 2^32 ({:.3e})",
            mono.violations,
            mono.pairs,
            mono.largest_drop,
            mono.largest_drop as f64 / equity::ONE as f64
        );
        for v in &mono.examples {
            println!(
                "  team {}: {} -> {:.9}, {} -> {:.9}",
                v.team,
                v.lower,
                v.at_lower as f64 / equity::ONE as f64,
                v.higher,
                v.at_higher as f64 / equity::ONE as f64
            );
        }
    }
    Ok(())
}

fn estimate(flags: &Flags) -> Result<(), String> {
    flags.check_known(&["bot", "hands", "out", "seed", "threads", "model-id"])?;
    let name = flags.need("bot")?;
    let hands: u64 = flags.number("hands", 0)?;
    if hands == 0 {
        return Err("--hands must be at least 1".into());
    }
    let out = PathBuf::from(flags.need("out")?);
    let seed: u64 = flags.number("seed", 1)?;
    let threads = flags.number("threads", default_threads())?;
    let factory = bot_factory(name).ok_or_else(|| {
        format!(
            "unknown bot `{name}` (known here: {}; `rules-v1` arrives with crates/bots, see src/registry.rs)",
            known_bots().join(", ")
        )
    })?;
    let start = Instant::now();
    let mut model = estimate_model_parallel(factory.as_ref(), hands, seed, threads)
        .map_err(|e| e.to_string())?;
    if let Some(id) = flags.get("model-id") {
        model.model_id = id.to_string();
    }
    let secs = start.elapsed().as_secs_f64();
    let json = model.to_json();
    write(&out, json.as_bytes())?;
    let s = &model.source;
    println!(
        "estimated {} from {} hands ({} completed, {} cut short by a game end) in {} matches of {} self-play, seed {seed}, {threads} threads, {secs:.1} s",
        model.model_id, s.hands, s.hands_completed, s.hands_cut_short, s.matches, s.bot
    );
    println!(
        "wrote {} (sha256 {})",
        out.display(),
        &sha256_hex(json.as_bytes())[..16]
    );
    Ok(())
}

fn load_table(path: &str) -> Result<EquityTable, String> {
    EquityTable::from_npy(&read(path)?).map_err(|e| format!("{path}: {e}"))
}

fn lookup(flags: &Flags) -> Result<(), String> {
    flags.check_known(&["table", "state", "stake"])?;
    let table = load_table(flags.need("table")?)?;
    let state = MatchState::parse(flags.need("state")?).map_err(|e| e.to_string())?;
    let stake: i32 = flags.number("stake", Rules::HOUSE.stake)?;
    let v = table.try_lookup(&state).map_err(|e| e.to_string())?;
    let p = v.to_f64_for_display();
    let money = v.expected_money(stake);
    let cents = v.expected_money(stake.saturating_mul(100));
    println!("{state}");
    for t in 0..4 {
        let sign = if cents[t] < 0 { '-' } else { '+' };
        let c = cents[t].unsigned_abs();
        println!(
            "team {} (seats {} and {}): chance of the money {:.9} ({} / 2^32); expected money per team {:+} at stake {stake} ({sign}{}.{:02} to the cent)",
            char::from(b'A' + t as u8),
            t,
            t + 4,
            p[t],
            v.0[t],
            money[t],
            c / 100,
            c % 100
        );
    }
    Ok(())
}

fn check(flags: &Flags) -> Result<(), String> {
    flags.check_known(&["table", "model", "mc-states", "mc-matches", "seed"])?;
    let table = load_table(flags.need("table")?)?;
    let (model, _) = load_model(flags.need("model")?)?;
    let compiled = model.compile();
    report_quick_checks(&table)?;
    let states: usize = flags.number("mc-states", 20)?;
    let matches: u64 = flags.number("mc-matches", 10_000)?;
    let seed: u64 = flags.number("seed", 1)?;
    let start = Instant::now();
    let mut positions = 0u64;
    for i in 0..equity::state::TABLE_STATES {
        let s = MatchState::from_canonical_index(i);
        positions += validate::check_hand_positions(&s, Scoring::HOUSE)?;
    }
    println!(
        "check: reachable scores and flag == (score mod 5 == 1) at all {positions} in-hand positions of all {} states ({:.1} s)",
        equity::state::TABLE_STATES,
        start.elapsed().as_secs_f64()
    );
    let sample = validate::sample_states(&table, states, seed);
    let rows =
        validate::monte_carlo_check(&table, &compiled, Scoring::HOUSE, &sample, matches, seed);
    let mut all = true;
    let mut max_z: f64 = 0.0;
    for r in &rows {
        all &= r.within;
        for z in r.z {
            max_z = max_z.max(z.abs());
        }
    }
    println!(
        "check: Monte Carlo, {} states x {matches} matches: {} (largest |z| {max_z:.2})",
        rows.len(),
        if all {
            "all within 3 standard errors"
        } else {
            "SOME OUTSIDE 3 standard errors"
        }
    );
    Ok(())
}

fn synthetic(flags: &Flags) -> Result<(), String> {
    flags.check_known(&["out"])?;
    let out = PathBuf::from(flags.need("out")?);
    let json = HandModel::synthetic().to_json();
    write(&out, json.as_bytes())?;
    println!("wrote the synthetic model to {}", out.display());
    Ok(())
}
