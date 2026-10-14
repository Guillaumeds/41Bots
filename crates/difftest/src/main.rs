//! `difftest <matches> [seed] [--bonus-later N] [--threads N]`
//!
//! Plays `matches` random matches through the engine and the reference, comparing them after
//! every action, then prints the rule-branch counts. Exits with status 1 on any disagreement, or
//! (for runs of 100,000 matches or more) when a rare branch is below its minimum rate.
//! Floating point is used here only to print rates and speed.

use std::process::ExitCode;
use std::time::Instant;

use difftest::{COVERAGE_MIN_MATCHES, Per, Report};
use engine::game::Rules;

const USAGE: &str = "usage: difftest <matches> [seed] [--bonus-later N] [--threads N]\n\
    \x20 matches        number of matches to play\n\
    \x20 seed           run seed (default 0)\n\
    \x20 --bonus-later  points for a team's later best-trump bonuses in a game (default 5; 6 is the flat-11 rule)\n\
    \x20 --threads      worker threads (default: all cores); the report does not depend on it";

struct Options {
    matches: u64,
    seed: u64,
    bonus_later: u8,
    threads: usize,
}

fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut positional = Vec::new();
    let mut bonus_later = Rules::HOUSE.bonus_later;
    let mut threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--bonus-later" => {
                let v = it.next().ok_or("--bonus-later needs a value")?;
                bonus_later = v
                    .parse()
                    .map_err(|_| format!("--bonus-later: not a number 0-255: {v}"))?;
            }
            "--threads" => {
                let v = it.next().ok_or("--threads needs a value")?;
                threads = v
                    .parse()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or(format!("--threads: not a positive number: {v}"))?;
            }
            "-h" | "--help" => return Err(String::new()),
            s if s.starts_with("--") => return Err(format!("unknown option {s}")),
            _ => positional.push(a),
        }
    }
    let (matches, seed) = match positional.as_slice() {
        [m] => (m, None),
        [m, s] => (m, Some(s)),
        _ => return Err("expected <matches> and optionally [seed]".into()),
    };
    let matches = matches
        .parse()
        .map_err(|_| format!("matches: not a number: {matches}"))?;
    let seed = match seed {
        None => 0,
        Some(s) => s.parse().map_err(|_| format!("seed: not a number: {s}"))?,
    };
    Ok(Options {
        matches,
        seed,
        bonus_later,
        threads,
    })
}

/// 1234567 -> "1,234,567".
fn grouped(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn percent(count: u64, out_of: u64) -> f64 {
    if out_of == 0 {
        0.0
    } else {
        100.0 * count as f64 / out_of as f64
    }
}

fn print_report(r: &Report) {
    println!(
        "totals: {} matches, {} games, {} hands, {} actions",
        grouped(r.matches),
        grouped(r.games),
        grouped(r.hands),
        grouped(r.actions)
    );
    println!();
    // Columns: label 44, count 11, rate 9 ("12.345%"), " of <base>" 11, minimum.
    println!(
        "{:<44} {:>11}  {:>9}{:11}  minimum",
        "branch", "count", "rate", ""
    );
    for l in r.lines() {
        let per = match l.per {
            Per::Hand => "hands",
            Per::Game => "games",
            Per::Match => "matches",
        };
        let min = match (l.minimum_per_mille, l.must_be_zero) {
            (Some(m), _) => format!("{}.{}%", m / 10, m % 10),
            (None, true) => "must be 0".into(),
            (None, false) => String::new(),
        };
        println!(
            "{:<44} {:>11}  {:>8.3}% of {:<7}  {}",
            l.label,
            grouped(l.count),
            percent(l.count, l.out_of),
            per,
            min
        );
    }
    println!(
        "(\"turned\" counts the card that fixed trumps: the turned card, or the second after a Joker)"
    );
    println!();
    let wins: Vec<String> = r.match_wins.iter().map(|&w| grouped(w)).collect();
    let money: Vec<String> = r.money.iter().map(|m| m.to_string()).collect();
    println!("matches won by team 0-3: {}", wins.join(" / "));
    println!("money by team 0-3:       {}", money.join(" / "));
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let o = match parse_args(&args) {
        Ok(o) => o,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("difftest: {e}");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let rules = Rules {
        bonus_later: o.bonus_later,
        ..Rules::HOUSE
    };
    println!(
        "difftest: {} matches, seed {}, rules bonus_first={} bonus_later={} stake={}, {} thread{}",
        grouped(o.matches),
        o.seed,
        rules.bonus_first,
        rules.bonus_later,
        rules.stake,
        o.threads,
        if o.threads == 1 { "" } else { "s" }
    );
    let start = Instant::now();
    let result = difftest::run_threads(o.seed, o.matches, rules, o.threads);
    let secs = start.elapsed().as_secs_f64();
    let report = match result {
        Ok(r) => r,
        Err(d) => {
            println!("{d}");
            println!("RESULT: FAIL: the engine and the reference disagree (details above)");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "no disagreements: the engine and the reference agreed after every action, and every \
         match replayed identically from its seed and actions"
    );
    println!(
        "time {secs:.1} s, {:.1} matches per second",
        if secs > 0.0 {
            o.matches as f64 / secs
        } else {
            0.0
        }
    );
    print_report(&report);
    let gated = o.matches >= COVERAGE_MIN_MATCHES;
    let problems = report.coverage_problems(gated);
    println!();
    if !gated {
        println!(
            "coverage minimums not checked: they apply to runs of {} matches or more",
            grouped(COVERAGE_MIN_MATCHES)
        );
    }
    if problems.is_empty() {
        println!("RESULT: PASS");
        ExitCode::SUCCESS
    } else {
        for p in &problems {
            println!("coverage problem: {p}");
        }
        println!("RESULT: FAIL: coverage");
        ExitCode::FAILURE
    }
}
