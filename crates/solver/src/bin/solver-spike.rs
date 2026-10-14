//! M2 measurement spike (PLAN amendment A2): sample random deals, play them forward with random
//! legal play to the start of trick k, solve the root exactly with the naive max^n solver, and
//! report costs per trick index and per root seat.
//!
//! ```text
//! solver-spike --table equity-v0.npy --trick K --worlds N [--seed S] [--tt on|off]
//!              [--cap-ms MS] [--threads T] [--flip] [--first W]
//! ```
//!
//! Worlds are numbered; world `i` uses `Rng::derive(seed, [i])`, so a run can be split into
//! chunks with `--first`. Random play-forward: skip 0–3 whole hands (to vary the match state),
//! then play on to the start of trick K of the next hand that reaches it. At trick 1 an
//! outstanding rob is resolved first with a random legal burn (the solver needs a fully known
//! world; searching the burn is M5 work, amendment A4).

use std::time::Instant;

use engine::cards::{Bits, Card, bit, cards};
use engine::game::{Action, Match, Phase, Rules};
use engine::rng::Rng;
use engine::rules::legal_burns;
use equity::EquityTable;
use solver::{Options, Position, TT_ENTRY_BYTES, TieOrder, from_match, solve};

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
    a
}

fn pick(rng: &mut Rng, set: Bits) -> Card {
    let all: Vec<Card> = cards(set).collect();
    all[rng.bounded(all.len() as u64) as usize]
}

/// Play random legal actions until the start of trick `k` (robbing resolved); returns the solver
/// position, context and root seat relative to the dealer.
fn sample(seed: u64, k: u8, world: u64) -> (Position, solver::MatchCtx, u8) {
    let mut rng = Rng::derive(seed, &[world]);
    let mut m = Match::new(rng.next_u64(), Rules::HOUSE);
    let skip = rng.bounded(4) as u32;
    loop {
        if m.phase() == Phase::MatchOver {
            m = Match::new(rng.next_u64(), Rules::HOUSE);
            continue;
        }
        let p = m.public();
        let at_start = p.trick_no == k && p.trick_cards.is_empty() && m.hand_index() >= skip;
        if at_start {
            match m.phase() {
                // The trick-1 leader robs before leading: burn at random, then solve.
                Phase::AwaitBurn(_) if k == 1 => {}
                Phase::AwaitPlay(_) => {
                    let (mut pos, ctx, pending) = from_match(&m).expect("awaiting a play");
                    if let Some(s) = pending {
                        let table = p.table_card.expect("turned card on the table");
                        let h = pos.hands[usize::from(s)] | bit(table);
                        let burn = pick(&mut rng, legal_burns(h, p.trump));
                        pos.hands[usize::from(s)] = h & !bit(burn);
                    }
                    let rel = (pos.leader + 8 - m.dealer()) % 8;
                    return (pos, ctx, rel);
                }
                _ => {}
            }
        }
        let acts = m.legal_actions();
        let a: Action = acts[rng.bounded(acts.len() as u64) as usize];
        m.apply(a).expect("legal");
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
}

fn run_world(a: &Args, table: &EquityTable, w: u64) -> Rec {
    let (pos, ctx, rel) = sample(a.seed, a.trick, w);
    let opts = Options {
        tt: a.tt,
        order: TieOrder::Ascending,
        time_cap_ms: a.cap_ms,
    };
    let t0 = Instant::now();
    let sol = solve(&pos, ctx, table, opts);
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let rev = (a.flip && !sol.timed_out).then(|| {
        solve(
            &pos,
            ctx,
            table,
            Options {
                order: TieOrder::Descending,
                ..opts
            },
        )
    });
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
    }
}

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
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
        "summary\ttrick={} worlds={} tt={} cap_ms={:?} threads={} wall_s={:.1}",
        a.trick, a.worlds, a.tt, a.cap_ms, a.threads, wall
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
