//! What an exact solve costs once partnerships are turned (docs/APP-ENGINE-CONTRACT.md §7):
//! positions after a turn are searched exactly (never by a network), with 6 or 4 seats in play.
//!
//! `cargo run --release -p solver --example turned_cost -- <equity.npy> [worlds] [seed] [tricks]
//! [seats]`
//!
//! For each trick `k` (default 2, 3, 4; trick 1 is allowed, its pending rob searched inside the
//! solve as exact PIMC does) and each world: a random match (0–3 whole hands first,
//! so the scores vary) is played with random legal cards to the start of trick `k`; the same
//! position is then solved with nobody turned (8 seats), with one partnership turned (6 seats)
//! and with two (4 seats) (default all three; `seats` such as `6,4` picks), each turned
//! partnership chosen at random (another is tried when the
//! engine refuses one, and the world is left out when a turn ends the game). Every solve is the
//! optimised exact solver with its default switches, single-threaded, one `SolveMem` reused. It
//! prints, per trick and seats in play, the solves, the time in milliseconds (mean, median, 90th
//! percentile, largest) and the nodes (mean, median, largest). Timings are wall-clock and only
//! measure; nothing else depends on them.

use std::time::Instant;

use engine::game::{Match, Phase, Rules};
use engine::rng::Rng;
use equity::EquityTable;
use solver::{FastOptions, Rob, SolveMem, from_match, solve_fast_with};

/// The match at the start of trick `k` for world `w`, or `None` if the match ended first.
fn at_trick(seed: u64, k: u8, w: u64) -> Option<Match> {
    let mut rng = Rng::derive(seed, &[u64::from(k), w]);
    let mut m = Match::new(rng.next_u64(), Rules::HOUSE);
    let skip = rng.bounded(4) as u32;
    loop {
        if m.phase() == Phase::MatchOver {
            return None;
        }
        let p = m.public();
        if m.hand_index() >= skip
            && p.trick_no == k
            && p.trick_cards.is_empty()
            && matches!(m.phase(), Phase::AwaitPlay(_))
        {
            return Some(m);
        }
        let acts = m.legal_actions();
        m.apply(acts[rng.bounded(acts.len() as u64) as usize])
            .expect("legal");
    }
}

/// `m` with `n` partnerships turned at random (another is tried when one is refused), or `None`
/// when no such turn leaves the hand going at the same trick.
fn turned(m: &Match, n: u32, seed: u64, w: u64) -> Option<Match> {
    let mut rng = Rng::derive(seed, &[0x7u64, w, u64::from(n)]);
    let mut t = m.clone();
    let mut done = 0;
    let mut tries = 0;
    while done < n {
        tries += 1;
        if tries > 40 {
            return None;
        }
        let mut u = t.clone();
        if u.turn_team(rng.bounded(4) as u8).is_ok()
            && u.hand_index() == m.hand_index()
            && u.public().trick_no == m.public().trick_no
            && matches!(u.phase(), Phase::AwaitPlay(_))
        {
            t = u;
            done += 1;
        }
    }
    Some(t)
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: turned_cost <equity.npy> [worlds] [seed] [tricks: 2,3,4] [seats: 8,6,4]");
        std::process::exit(2);
    };
    let worlds: u64 = args.get(1).map_or(20, |s| s.parse().expect("worlds"));
    let seed: u64 = args.get(2).map_or(17, |s| s.parse().expect("seed"));
    let tricks: Vec<u8> = args.get(3).map_or(vec![2, 3, 4], |s| {
        s.split(',').map(|x| x.parse().expect("trick")).collect()
    });
    let seats: Vec<u32> = args.get(4).map_or(vec![8, 6, 4], |s| {
        s.split(',').map(|x| x.parse().expect("seats")).collect()
    });
    let table = EquityTable::from_npy(&std::fs::read(path).expect("read the table"))
        .expect("a valid equity table");
    let opts = FastOptions::default();
    let mut mem = SolveMem::new();
    println!(
        "exact solves (optimised solver, default switches, one thread), {worlds} worlds per \
         trick, seed {seed}"
    );
    println!(
        "{:>5} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>12} {:>12} {:>12}",
        "trick",
        "seats",
        "solves",
        "mean ms",
        "median ms",
        "p90 ms",
        "max ms",
        "mean nodes",
        "median nodes",
        "max nodes"
    );
    for &k in &tricks {
        for n in 0..=2u32 {
            if !seats.contains(&(8 - 2 * n)) {
                continue;
            }
            let mut ms = Vec::new();
            let mut nodes = Vec::new();
            for w in 0..worlds {
                let Some(m) = at_trick(seed, k, w) else {
                    continue;
                };
                let Some(t) = turned(&m, n, seed, w) else {
                    continue;
                };
                let (mut pos, ctx, pending) = from_match(&t).expect("a play is awaited");
                if let Some(seat) = pending {
                    // Trick 1: the robber's burn is a node of the solve.
                    let card = t.public().table_card.expect("the turned card");
                    pos.rob = Some(Rob { seat, card });
                }
                assert_eq!(pos.turned.count_ones(), n);
                let start = Instant::now();
                let s = solve_fast_with(&pos, ctx, &table, opts, &mut mem);
                ms.push(start.elapsed().as_secs_f64() * 1000.0);
                nodes.push(s.stats.nodes as f64);
            }
            if ms.is_empty() {
                continue;
            }
            let c = ms.len();
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let (mm, mn) = (mean(&ms), mean(&nodes));
            let (med, p90, mx) = (pct(&mut ms, 0.5), pct(&mut ms, 0.9), pct(&mut ms, 1.0));
            let (nmed, nmx) = (pct(&mut nodes, 0.5), pct(&mut nodes, 1.0));
            println!(
                "{:>5} {:>6} {:>6} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>12.0} {:>12.0} {:>12.0}",
                k,
                8 - 2 * n,
                c,
                mm,
                med,
                p90,
                mx,
                mn,
                nmed,
                nmx
            );
        }
    }
}
