//! Cost bench of the world generator (docs/M3-SPEC.md §4.6): `Tracker::from_view` +
//! `WorldSpace::new` + `count` + 100 `sample` per decision, median and p90 milliseconds per trick.
//! Positions come from matches played by uniformly random legal players.
//!
//! Usage: `belief-bench [positions_per_trick=2000] [seed=1]`

use std::time::Instant;

use belief::{Tracker, WorldSpace};
use engine::game::{Match, Phase, Rules};
use engine::rng::Rng;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let per_trick: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    // Index 0 = the dealer's burns before trick 1, then tricks 1..=5.
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); 6];
    let mut log_n: Vec<Vec<f64>> = vec![Vec::new(); 6];
    let mut classes: Vec<Vec<usize>> = vec![Vec::new(); 6];
    let mut match_seed = seed << 32;
    while times[1..].iter().any(|t| t.len() < per_trick) {
        let mut m = Match::new(match_seed, Rules::HOUSE);
        let mut rng = Rng::new(match_seed ^ 0x5eed);
        match_seed += 1;
        while m.phase() != Phase::MatchOver {
            let seat = m.to_act().expect("a seat acts");
            let view = m.view(seat);
            let tr = usize::from(view.public.trick_no);
            if times[tr].len() < per_trick {
                let t0 = Instant::now();
                let t = Tracker::from_view(&view);
                let ws = WorldSpace::new(&t).expect("feasible");
                let n = ws.count();
                let mut wr = Rng::new(n as u64);
                for _ in 0..100 {
                    std::hint::black_box(ws.sample(&mut wr));
                }
                times[tr].push(t0.elapsed().as_secs_f64() * 1e3);
                log_n[tr].push((n as f64).log10());
                classes[tr].push(ws.classes());
            }
            let legal = m.legal_actions();
            let a = legal[rng.bounded(legal.len() as u64) as usize];
            m.apply(a).expect("legal");
        }
    }
    println!("trick  positions  median_ms  p90_ms  max_ms  median_log10_N  max_classes");
    for tr in 0..6 {
        let mut t = times[tr].clone();
        t.sort_by(f64::total_cmp);
        let mut l = log_n[tr].clone();
        l.sort_by(f64::total_cmp);
        let q = |v: &[f64], p: f64| v[((v.len() - 1) as f64 * p) as usize];
        println!(
            "{tr:>5}  {:>9}  {:>9.3}  {:>6.3}  {:>6.3}  {:>14.1}  {:>11}",
            t.len(),
            q(&t, 0.5),
            q(&t, 0.9),
            t[t.len() - 1],
            q(&l, 0.5),
            classes[tr].iter().max().copied().unwrap_or(0)
        );
    }
}
