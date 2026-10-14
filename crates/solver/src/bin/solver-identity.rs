//! The M2 identity gate at scale (PLAN A2, validation layer (c)): on many random worlds, the
//! optimised solver must give exactly the naive solver's root value, the value of every root
//! card and the same chosen card, for every combination of its switches and both tie orders.
//!
//! ```text
//! solver-identity --leaf <equity.npy | fake | tie> --trick K --worlds N [--first W] [--seed S]
//!                 [--threads T] [--combos all|lean|tt] [--extra-max E] [--cap-ms MS]
//! ```
//!
//! World `i` is `sample_world(seed, K, i, i % (E + 1))`, so roots are at the start of the trick
//! and also part-way through it. `lean` checks all on, each switch alone and each switch off
//! (9 combinations) instead of all 16; `tt` checks the 8 combinations with the table on (the
//! others are too slow at trick 2). Worlds whose naive solve exceeds `--cap-ms` are skipped
//! and counted. Exits with status 1 on any difference.

use std::sync::atomic::{AtomicU64, Ordering};

use equity::{EquityTable, MatchState, MoneyChance};
use solver::{FastOptions, FnLeaf, Leaf, Options, TieOrder, sample_world, solve, solve_fast};

fn fake_value(s: &MatchState) -> MoneyChance {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s
        .scores
        .iter()
        .chain(&s.games)
        .chain(core::iter::once(&s.dealer_team))
    {
        h = (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01B3);
    }
    let w: [u128; 4] = core::array::from_fn(|t| u128::from((h >> (16 * t)) & 0xFFFF) + 1);
    MoneyChance::from_weights(w)
}

fn tie_value(s: &MatchState) -> MoneyChance {
    let w: [u128; 4] = core::array::from_fn(|t| {
        1 + u128::from(s.scores[t] >= 25) * 3
            + u128::from(s.games[t]) * 5
            + u128::from(s.dealer_team == t as u8)
    });
    MoneyChance::from_weights(w)
}

enum AnyLeaf {
    Table(Box<EquityTable>),
    F(FnLeaf<fn(&MatchState) -> MoneyChance>),
}

impl Leaf for AnyLeaf {
    fn value(&self, s: &MatchState) -> MoneyChance {
        match self {
            AnyLeaf::Table(t) => t.value(s),
            AnyLeaf::F(f) => f.value(s),
        }
    }
}

fn main() {
    let v: Vec<String> = std::env::args().skip(1).collect();
    let get = |name: &str| {
        v.iter()
            .position(|x| x == name)
            .and_then(|i| v.get(i + 1).cloned())
    };
    let leaf_arg = get("--leaf").expect("--leaf <npy|fake|tie>");
    let trick: u8 = get("--trick").expect("--trick").parse().expect("trick");
    let worlds: u64 = get("--worlds").map_or(100, |s| s.parse().expect("worlds"));
    let first: u64 = get("--first").map_or(0, |s| s.parse().expect("first"));
    let seed: u64 = get("--seed").map_or(1, |s| s.parse().expect("seed"));
    let threads: usize = get("--threads").map_or(1, |s| s.parse().expect("threads"));
    let extra_max: u64 = get("--extra-max").map_or(7, |s| s.parse().expect("extra-max"));
    let cap: Option<u64> = get("--cap-ms").map(|s| s.parse().expect("cap"));
    let combos: Vec<u8> = match get("--combos").as_deref() {
        Some("lean") => vec![15, 1, 2, 4, 8, 14, 13, 11, 7],
        Some("tt") => vec![1, 3, 5, 7, 9, 11, 13, 15],
        _ => (0..16).collect(),
    };
    let leaf = match leaf_arg.as_str() {
        "fake" => AnyLeaf::F(FnLeaf(fake_value)),
        "tie" => AnyLeaf::F(FnLeaf(tie_value)),
        path => AnyLeaf::Table(Box::new(
            EquityTable::from_npy(&std::fs::read(path).expect("read table")).expect("table"),
        )),
    };
    let next = AtomicU64::new(first);
    let checked = AtomicU64::new(0);
    let skipped = AtomicU64::new(0);
    let bad = AtomicU64::new(0);
    let solves = AtomicU64::new(0);
    let t0 = std::time::Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| {
                loop {
                    let w = next.fetch_add(1, Ordering::Relaxed);
                    if w >= first + worlds {
                        break;
                    }
                    let extra = (w % (extra_max + 1)) as u8;
                    let (pos, ctx, _) = sample_world(seed, trick, w, extra);
                    for order in [TieOrder::Ascending, TieOrder::Descending] {
                        let naive = solve(
                            &pos,
                            ctx,
                            &leaf,
                            Options {
                                tt: true,
                                order,
                                time_cap_ms: cap,
                            },
                        );
                        if naive.timed_out {
                            skipped.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        for &bits in &combos {
                            let f = solve_fast(
                                &pos,
                                ctx,
                                &leaf,
                                FastOptions {
                                    order,
                                    tt_mb: 1024,
                                    ..FastOptions::default()
                                }
                                .combo(bits),
                            );
                            solves.fetch_add(1, Ordering::Relaxed);
                            if f.value != naive.value
                                || f.best != naive.best
                                || f.children != naive.children
                            {
                                bad.fetch_add(1, Ordering::Relaxed);
                                println!(
                                    "MISMATCH world {w} extra {extra} order {order:?} combo {bits}"
                                );
                            }
                        }
                        checked.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        }
    });
    let bad = bad.into_inner();
    println!(
        "identity\tleaf={leaf_arg} trick={trick} worlds={worlds} first={first} seed={seed} combos={} \
         root_solves_checked={} (worlds x 2 tie orders) fast_solves={} skipped_naive_timeouts={} \
         mismatches={bad} wall_s={:.1} {}",
        combos.len(),
        checked.into_inner(),
        solves.into_inner(),
        skipped.into_inner(),
        t0.elapsed().as_secs_f64(),
        if bad == 0 { "PASS" } else { "FAIL" }
    );
    if bad != 0 {
        std::process::exit(1);
    }
}
