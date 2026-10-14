//! M2 measurement spike (PLAN amendment A2): sample random deals, play them forward with random
//! legal play to the start of trick k, solve the root exactly with the naive max^n solver, and
//! report costs per trick index and per root seat.
//!
//! ```text
//! solver-spike --table equity-v0.npy --trick K --worlds N [--seed S] [--tt on|off]
//!              [--cap-ms MS] [--threads T] [--flip] [--first W]
//!              [--solver naive|fast] [--switches BITS] [--tt-mb MB]
//!              [--solve-threads N --cap-secs S [--progress-secs P] [--tasks-per-thread K]]
//! ```
//!
//! `--solve-threads N` (fast solver only) solves the worlds **one at a time**, each on N threads
//! with the parallel single-world solve (`solve_fast_par`, identical answers) and one shared
//! table of `--tt-mb` megabytes; `--threads` is then ignored. `--cap-secs S` is the per-world
//! cap (0 = none; required in this mode so a cap is always chosen explicitly). Every
//! `--progress-secs` (default 60) a `progress` line shows nodes so far and table fill; each world
//! ends with a `par_world` line: seconds, nodes, peak RSS, table stats, the root value.
//!
//! `--solver fast` uses the optimised solver; `--switches` picks its speed-ups (bit 0 table,
//! 1 equivalent cards, 2 move ordering, 3 pruning; default 11 = the solver's defaults: all
//! but move ordering) and `--tt-mb` its table cap.
//! The same `--seed` gives the same worlds for both solvers.
//!
//! Worlds are numbered; world `i` uses `Rng::derive(seed, [i])`, so a run can be split into
//! chunks with `--first`. Random play-forward: skip 0–3 whole hands (to vary the match state),
//! then play on to the start of trick K of the next hand that reaches it. At trick 1 an
//! outstanding rob is resolved first with a random legal burn (the solver needs a fully known
//! world; searching the burn is M5 work, amendment A4).

use std::io::Write;
use std::sync::atomic::Ordering;
use std::time::Instant;

use equity::EquityTable;
use solver::fast::FAST_TT_SLOT_BYTES;
use solver::{
    FastOptions, Options, ParProgress, Solution, TT_ENTRY_BYTES, TieOrder, sample_world, solve,
    solve_fast, solve_fast_par_split,
};

struct Args {
    table: String,
    trick: u8,
    worlds: u64,
    first: u64,
    seed: u64,
    tt: bool,
    cap_ms: Option<u64>,
    threads: usize,
    flip: bool,
    fast: bool,
    switches: u8,
    tt_mb: usize,
    solve_threads: usize,
    cap_secs: Option<u64>,
    progress_secs: u64,
    tasks_per_thread: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        table: String::new(),
        trick: 5,
        worlds: 100,
        first: 0,
        seed: 1,
        tt: false,
        cap_ms: None,
        threads: 1,
        flip: false,
        fast: false,
        switches: 11,
        tt_mb: 256,
        solve_threads: 0,
        cap_secs: None,
        progress_secs: 60,
        tasks_per_thread: solver::fast::PAR_TASKS_PER_THREAD,
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let val = || v.get(i + 1).cloned().unwrap_or_default();
        match v[i].as_str() {
            "--table" => a.table = val(),
            "--trick" => a.trick = val().parse().expect("--trick"),
            "--worlds" => a.worlds = val().parse().expect("--worlds"),
            "--first" => a.first = val().parse().expect("--first"),
            "--seed" => a.seed = val().parse().expect("--seed"),
            "--tt" => a.tt = val() == "on",
            "--cap-ms" => a.cap_ms = Some(val().parse().expect("--cap-ms")),
            "--threads" => a.threads = val().parse().expect("--threads"),
            "--solver" => a.fast = val() == "fast",
            "--switches" => a.switches = val().parse().expect("--switches"),
            "--tt-mb" => a.tt_mb = val().parse().expect("--tt-mb"),
            "--solve-threads" => a.solve_threads = val().parse().expect("--solve-threads"),
            "--cap-secs" => a.cap_secs = Some(val().parse().expect("--cap-secs")),
            "--progress-secs" => a.progress_secs = val().parse().expect("--progress-secs"),
            "--tasks-per-thread" => a.tasks_per_thread = val().parse().expect("--tasks-per-thread"),
            "--flip" => {
                a.flip = true;
                i += 1;
                continue;
            }
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    assert!(!a.table.is_empty(), "--table is required");
    assert!((1..=5).contains(&a.trick), "--trick must be 1..=5");
    if a.solve_threads > 0 {
        assert!(a.fast, "--solve-threads needs --solver fast");
        assert!(
            a.cap_secs.is_some(),
            "--solve-threads needs --cap-secs (0 = no cap)"
        );
        assert!(a.progress_secs > 0, "--progress-secs must be positive");
    }
    a
}

/// `--solve-threads`: worlds one at a time, each solved on all the given threads.
fn par_main(a: &Args, table: &EquityTable) {
    let cap_ms = a.cap_secs.filter(|&s| s > 0).map(|s| s * 1000);
    let fopts = FastOptions {
        tt_mb: a.tt_mb,
        time_cap_ms: cap_ms,
        ..FastOptions::default().combo(a.switches)
    };
    println!(
        "par\ttrick={} worlds={} first={} seed={} solve_threads={} tt_mb={} cap_secs={} switches={} tasks_per_thread={}",
        a.trick,
        a.worlds,
        a.first,
        a.seed,
        a.solve_threads,
        a.tt_mb,
        a.cap_secs.unwrap_or(0),
        a.switches,
        a.tasks_per_thread
    );
    for w in a.first..a.first + a.worlds {
        let (pos, ctx, rel) = sample_world(a.seed, a.trick, w, 0);
        let pr = ParProgress::default();
        let t0 = Instant::now();
        let sol = std::thread::scope(|sc| {
            let h = sc.spawn(|| {
                solve_fast_par_split(
                    &pos,
                    ctx,
                    table,
                    fopts,
                    a.solve_threads,
                    a.tasks_per_thread,
                    Some(&pr),
                )
            });
            let mut last = Instant::now();
            while !h.is_finished() {
                std::thread::sleep(std::time::Duration::from_millis(10));
                if last.elapsed().as_secs() >= a.progress_secs {
                    last = Instant::now();
                    println!(
                        "progress\tworld={} elapsed_s={:.0} nodes={} tt_entries={} tt_fill={:.3} tasks_done={}/{} rss_mb={:.0}",
                        w,
                        t0.elapsed().as_secs_f64(),
                        pr.nodes.load(Ordering::Relaxed),
                        pr.tt_entries.load(Ordering::Relaxed),
                        pr.tt_entries.load(Ordering::Relaxed) as f64 * FAST_TT_SLOT_BYTES as f64
                            / (a.tt_mb as f64 * 1048576.0),
                        pr.tasks_done.load(Ordering::Relaxed),
                        pr.tasks_total.load(Ordering::Relaxed),
                        rss_mb("VmRSS:")
                    );
                    let _ = std::io::stdout().flush();
                }
            }
            h.join().expect("solve thread")
        });
        let secs = t0.elapsed().as_secs_f64();
        let value = sol.value.map_or("-".to_string(), |v| {
            (0..4u8)
                .map(|t| format!("{:.6}", v.get(t) as f64 / 4294967296.0))
                .collect::<Vec<_>>()
                .join(",")
        });
        println!(
            "par_world\ttrick={} world={} rel_seat={} root_legal={} secs={:.3} timed_out={} nodes={} leaves={} tt_probes={} tt_hits={} tt_entries={} tt_replaced={} tt_mb_used={:.1} peak_rss_mb={:.0} best={} value={} cutoffs={} equiv_skipped={}",
            a.trick,
            w,
            rel,
            pos.legal().count_ones(),
            secs,
            u8::from(sol.timed_out),
            sol.stats.nodes,
            sol.stats.leaves,
            sol.stats.tt_probes,
            sol.stats.tt_hits,
            sol.stats.tt_entries,
            sol.stats.tt_replaced,
            sol.stats.tt_bytes as f64 / 1048576.0,
            peak_rss_mb(),
            sol.best.map_or("-".to_string(), |c| c.to_string()),
            value,
            sol.stats.cutoffs,
            sol.stats.equiv_skipped
        );
        let _ = std::io::stdout().flush();
    }
}

#[derive(Clone, Copy, Default)]
struct Rec {
    rel_seat: u8,
    ms: f64,
    nodes: u64,
    leaves: u64,
    probes: u64,
    hits: u64,
    entries: u64,
    decision: u64,
    tied: u64,
    root_legal: u32,
    root_tied: u32,
    timed_out: bool,
    flipped: Option<bool>,
    value_changed: Option<bool>,
    cutoffs: u64,
    equiv_skipped: u64,
    replaced: u64,
    tt_bytes: u64,
}

fn run_world(a: &Args, table: &EquityTable, w: u64) -> Rec {
    let (pos, ctx, rel) = sample_world(a.seed, a.trick, w, 0);
    let opts = Options {
        tt: a.tt,
        order: TieOrder::Ascending,
        time_cap_ms: a.cap_ms,
    };
    let fopts = FastOptions {
        tt_mb: a.tt_mb,
        time_cap_ms: a.cap_ms,
        ..FastOptions::default().combo(a.switches)
    };
    let run = |order: TieOrder| -> Solution {
        if a.fast {
            solve_fast(&pos, ctx, table, FastOptions { order, ..fopts })
        } else {
            solve(&pos, ctx, table, Options { order, ..opts })
        }
    };
    let t0 = Instant::now();
    let sol = run(TieOrder::Ascending);
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let rev = (a.flip && !sol.timed_out).then(|| run(TieOrder::Descending));
    let flipped = rev.as_ref().map(|r| r.best != sol.best);
    let value_changed = rev.as_ref().map(|r| r.value != sol.value);
    Rec {
        rel_seat: rel,
        ms,
        nodes: sol.stats.nodes,
        leaves: sol.stats.leaves,
        probes: sol.stats.tt_probes,
        hits: sol.stats.tt_hits,
        entries: sol.stats.tt_entries,
        decision: sol.stats.decision_nodes,
        tied: sol.stats.tied_nodes,
        root_legal: pos.legal().count_ones(),
        root_tied: sol.root_tied,
        timed_out: sol.timed_out,
        flipped,
        value_changed,
        cutoffs: sol.stats.cutoffs,
        equiv_skipped: sol.stats.equiv_skipped,
        replaced: sol.stats.tt_replaced,
        tt_bytes: sol.stats.tt_bytes,
    }
}

fn peak_rss_mb() -> f64 {
    rss_mb("VmHWM:")
}

fn rss_mb(field: &str) -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(field))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        })
        .map_or(f64::NAN, |kb| kb / 1024.0)
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|x, y| x.total_cmp(y));
    let i = ((v.len() as f64 - 1.0) * p).round() as usize;
    v[i]
}

fn main() {
    let a = parse_args();
    let bytes = std::fs::read(&a.table).expect("read table");
    let table = EquityTable::from_npy(&bytes).expect("valid table");
    if a.solve_threads > 0 {
        par_main(&a, &table);
        return;
    }
    let rss_before = peak_rss_mb();
    let ids: Vec<u64> = (a.first..a.first + a.worlds).collect();
    let mut recs = vec![Rec::default(); ids.len()];
    let t_all = Instant::now();
    std::thread::scope(|sc| {
        for (chunk_ids, chunk_out) in ids
            .chunks(ids.len().div_ceil(a.threads.max(1)))
            .zip(recs.chunks_mut(ids.len().div_ceil(a.threads.max(1))))
        {
            let a = &a;
            let table = &table;
            sc.spawn(move || {
                for (w, out) in chunk_ids.iter().zip(chunk_out.iter_mut()) {
                    *out = run_world(a, table, *w);
                }
            });
        }
    });
    let wall = t_all.elapsed().as_secs_f64();
    // Per-world lines (machine-readable).
    for (w, r) in ids.iter().zip(&recs) {
        println!(
            "world\t{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            a.trick,
            w,
            r.rel_seat,
            r.ms,
            r.nodes,
            r.leaves,
            r.probes,
            r.hits,
            r.entries,
            r.decision,
            r.tied,
            r.root_legal,
            r.root_tied,
            u8::from(r.timed_out),
            r.flipped.map_or("-", |f| if f { "1" } else { "0" }),
            r.value_changed.map_or("-", |f| if f { "1" } else { "0" })
        );
    }
    let done: Vec<&Rec> = recs.iter().filter(|r| !r.timed_out).collect();
    let mut ms: Vec<f64> = done.iter().map(|r| r.ms).collect();
    let mut nodes: Vec<f64> = done.iter().map(|r| r.nodes as f64).collect();
    let probes: u64 = done.iter().map(|r| r.probes).sum();
    let hits: u64 = done.iter().map(|r| r.hits).sum();
    let max_entries = done.iter().map(|r| r.entries).max().unwrap_or(0);
    let dec: u64 = done.iter().map(|r| r.decision).sum();
    let tied: u64 = done.iter().map(|r| r.tied).sum();
    let multi: Vec<&&Rec> = done.iter().filter(|r| r.root_legal >= 2).collect();
    let root_tie_frac: f64 = multi
        .iter()
        .map(|r| f64::from(r.root_tied) / f64::from(r.root_legal))
        .sum::<f64>()
        / multi.len() as f64;
    let root_any_tie = multi.iter().filter(|r| r.root_tied >= 2).count();
    let flips: Vec<bool> = done.iter().filter_map(|r| r.flipped).collect();
    let vchg = done
        .iter()
        .filter(|r| r.value_changed == Some(true))
        .count();
    println!(
        "summary\ttrick={} worlds={} solver={} switches={} tt={} cap_ms={:?} threads={} wall_s={:.1}",
        a.trick,
        a.worlds,
        if a.fast { "fast" } else { "naive" },
        a.switches,
        a.tt,
        a.cap_ms,
        a.threads,
        wall
    );
    let sum = |f: fn(&&Rec) -> u64| done.iter().map(f).sum::<u64>();
    let max_bytes = done.iter().map(|r| r.tt_bytes).max().unwrap_or(0);
    println!(
        "summary\tcutoffs={} equiv_skipped={} tt_replaced={} fast_tt_max_mb={:.1} nodes_total={}",
        sum(|r| r.cutoffs),
        sum(|r| r.equiv_skipped),
        sum(|r| r.replaced),
        max_bytes as f64 / 1048576.0,
        sum(|r| r.nodes)
    );
    println!(
        "summary\tsolved={} timeouts={} ms_median={:.3} ms_p90={:.3} ms_max={:.3} nodes_median={:.0} nodes_p90={:.0} nodes_max={:.0}",
        done.len(),
        recs.len() - done.len(),
        pct(&mut ms.clone(), 0.5),
        pct(&mut ms.clone(), 0.9),
        pct(&mut ms, 1.0),
        pct(&mut nodes.clone(), 0.5),
        pct(&mut nodes.clone(), 0.9),
        pct(&mut nodes, 1.0),
    );
    println!(
        "summary\ttt_hit_rate={:.4} tt_max_entries={} tt_max_mb={:.1} peak_rss_mb={:.1} (before solving {:.1})",
        if probes > 0 {
            hits as f64 / probes as f64
        } else {
            f64::NAN
        },
        max_entries,
        (max_entries as f64 * TT_ENTRY_BYTES as f64) / 1048576.0,
        peak_rss_mb(),
        rss_before
    );
    println!(
        "summary\ttied_node_frac={:.4} root_tied_child_frac={:.4} root_any_tie={}/{} card_flips={}/{} root_value_changed={}",
        if dec > 0 {
            tied as f64 / dec as f64
        } else {
            f64::NAN
        },
        root_tie_frac,
        root_any_tie,
        multi.len(),
        flips.iter().filter(|&&f| f).count(),
        flips.len(),
        vchg
    );
    for s in 0..8u8 {
        let mut m: Vec<f64> = done
            .iter()
            .filter(|r| r.rel_seat == s)
            .map(|r| r.ms)
            .collect();
        let mut n: Vec<f64> = done
            .iter()
            .filter(|r| r.rel_seat == s)
            .map(|r| r.nodes as f64)
            .collect();
        let to = recs
            .iter()
            .filter(|r| r.rel_seat == s && r.timed_out)
            .count();
        println!(
            "seat\t{}\tcount={} timeouts={} ms_median={:.3} ms_p90={:.3} nodes_median={:.0}",
            s,
            m.len(),
            to,
            pct(&mut m.clone(), 0.5),
            pct(&mut m, 0.9),
            pct(&mut n, 0.5)
        );
    }
}
