//! Cost bench of the world generator (docs/M3-SPEC.md §4.6): `Tracker::from_view` +
//! `WorldSpace::new` + `count` + 100 `sample` per decision, median and p90 milliseconds per trick.
//! Tricks 1–5: decisions of matches played by uniformly random legal players. Trick 0 (the
//! dealer's burns before trick 1, rare in play): first hands dealt with a turned Joker or a turned
//! ace forced to deck position 40.
//!
//! Usage: `belief-bench [positions_per_trick=2000] [seed=1]`

use std::time::Instant;

use belief::{Tracker, WorldSpace};
use engine::cards::{ACE, JOKER, Suit, card};
use engine::game::{Match, Phase, Rules};
use engine::rng::Rng;
use engine::view::SeatView;

/// Time one decision: tracker, world space, count and 100 samples. Returns (ms, log10 N,
/// classes).
fn measure(view: &SeatView) -> (f64, f64, usize) {
    let t0 = Instant::now();
    let t = Tracker::from_view(view);
    let ws = WorldSpace::new(&t).expect("feasible");
    let n = ws.count();
    let mut wr = Rng::new(n as u64);
    for _ in 0..100 {
        std::hint::black_box(ws.sample(&mut wr));
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    (
        ms,
        f64::from(belief::log2_q8(n)) / 256.0 * std::f64::consts::LOG10_2,
        ws.classes(),
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let per_trick: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    // Index 0 = the dealer's burns before trick 1, then tricks 1..=5.
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); 6];
    let mut log_n: Vec<Vec<f64>> = vec![Vec::new(); 6];
    let mut classes: Vec<Vec<usize>> = vec![Vec::new(); 6];
    type Stats<'a> = (&'a mut [Vec<f64>], &'a mut [Vec<f64>], &'a mut [Vec<usize>]);
    fn record(st: Stats<'_>, tr: usize, (ms, l, c): (f64, f64, usize)) {
        st.0[tr].push(ms);
        st.1[tr].push(l);
        st.2[tr].push(c);
    }
    // Trick 0: forced turned Jokers and turned aces.
    let mut rng = Rng::derive(seed, &[0xB0]);
    let mut k = 0usize;
    while k < per_trick {
        let mut deck: [u8; 53] = core::array::from_fn(|i| i as u8);
        rng.shuffle(&mut deck);
        let forced = if k.is_multiple_of(2) {
            JOKER
        } else {
            card(Suit::from_index(rng.bounded(4) as u8), ACE)
        };
        let at = deck.iter().position(|&c| c == forced).expect("in the deck");
        deck.swap(at, 40);
        let dealer = rng.bounded(8) as u8;
        let mut m = Match::with_decks(dealer, vec![deck], seed, Rules::HOUSE);
        // Every decision before trick 1 (the Joker burn, then an at-once rob burn if any).
        while m.phase() != Phase::MatchOver && m.public().trick_no == 0 && k < per_trick {
            let seat = m.to_act().expect("a seat acts");
            let r = measure(&m.view(seat));
            record((&mut times, &mut log_n, &mut classes), 0, r);
            k += 1;
            let legal = m.legal_actions();
            let a = legal[rng.bounded(legal.len() as u64) as usize];
            m.apply(a).expect("legal");
        }
    }
    // Tricks 1-5.
    let mut match_seed = seed << 32;
    while times[1..].iter().any(|t| t.len() < per_trick) {
        let mut m = Match::new(match_seed, Rules::HOUSE);
        let mut rng = Rng::new(match_seed ^ 0x5eed);
        match_seed += 1;
        while m.phase() != Phase::MatchOver {
            let seat = m.to_act().expect("a seat acts");
            let view = m.view(seat);
            let tr = usize::from(view.public.trick_no);
            if tr >= 1 && times[tr].len() < per_trick {
                let r = measure(&view);
                record((&mut times, &mut log_n, &mut classes), tr, r);
            }
            let legal = m.legal_actions();
            let a = legal[rng.bounded(legal.len() as u64) as usize];
            m.apply(a).expect("legal");
        }
    }
    println!("trick  positions  median_ms  p90_ms  max_ms  log10_N median (p10-p90)  max_classes");
    for tr in 0..6 {
        let mut t = times[tr].clone();
        t.sort_by(f64::total_cmp);
        let mut l = log_n[tr].clone();
        l.sort_by(f64::total_cmp);
        let q = |v: &[f64], p: f64| v[((v.len() - 1) as f64 * p) as usize];
        println!(
            "{tr:>5}  {:>9}  {:>9.3}  {:>6.3}  {:>6.3}  {:>8.1} ({:>4.1}-{:>4.1})        {:>11}",
            t.len(),
            q(&t, 0.5),
            q(&t, 0.9),
            t[t.len() - 1],
            q(&l, 0.5),
            q(&l, 0.1),
            q(&l, 0.9),
            classes[tr].iter().max().copied().unwrap_or(0)
        );
    }
}
