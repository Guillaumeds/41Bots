//! Golden-vector test: the Rust RNG must reproduce `tools/rng_golden.py` exactly.

use engine::rng::Rng;

const GOLDEN: &str = include_str!("data/rng_golden.txt");

#[test]
fn rng_matches_python_golden_vectors() {
    let mut rng: Option<Rng> = None;
    let mut seeds_checked = 0;
    for line in GOLDEN.lines() {
        let mut parts = line.split_whitespace();
        let Some(tag) = parts.next() else { continue };
        match tag {
            "seed" => {
                let seed = u64::from_str_radix(parts.next().unwrap(), 16).unwrap();
                rng = Some(Rng::new(seed));
                seeds_checked += 1;
            }
            "next" => {
                let r = rng.as_mut().unwrap();
                for want in parts {
                    let want = u64::from_str_radix(want, 16).unwrap();
                    assert_eq!(r.next_u64(), want, "next_u64 mismatch");
                }
            }
            "bounded6" | "bounded53" | "bounded8" => {
                let n: u64 = tag.trim_start_matches("bounded").parse().unwrap();
                let r = rng.as_mut().unwrap();
                for want in parts {
                    let want: u64 = want.parse().unwrap();
                    assert_eq!(r.bounded(n), want, "bounded({n}) mismatch");
                }
            }
            "boundedbig" => {
                let n: u64 = (1u64 << 63) + 12345;
                let r = rng.as_mut().unwrap();
                for want in parts {
                    let want: u64 = want.parse().unwrap();
                    assert_eq!(r.bounded(n), want, "bounded(big) mismatch");
                }
            }
            "shuffle" => {
                let r = rng.as_mut().unwrap();
                let mut deck: Vec<u8> = (0..53).collect();
                r.shuffle(&mut deck);
                let want: Vec<u8> = parts.map(|s| s.parse().unwrap()).collect();
                assert_eq!(deck, want, "shuffle mismatch");
            }
            other => panic!("unknown golden tag {other}"),
        }
    }
    assert_eq!(seeds_checked, 6);
}
