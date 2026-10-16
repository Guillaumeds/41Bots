//! Wide networks (`nn::MAX_WIDTH` = 4096; the proof is in `crates/nn/src/mlp.rs`):
//! - every network up to 1024 wide gives the outputs it gave before the limit was raised (hashes
//!   recorded with the code of `54cfb3f`, MAX_WIDTH 1024), by the plain and the fast pass;
//! - the fast pass equals the plain reference bit for bit at widths 1024, 2048 and 4096 (first
//!   layers of 2048 and 4096 units, and fan-ins of 2048 and 4096 into the next layer), on real
//!   frontier positions, at every instruction-set level the machine has, with random weights and
//!   with weights at the limits the format allows;
//! - golden: the outputs of seeded wide networks, as hashes (a change of any output is a change of
//!   a hash); `python/nn` (`test_intnet`) checks wide networks of the same recipe, made by
//!   `nn-tool random`, against its own integer forward pass through `nn-tool eval`.

use engine::rng::{Rng, SplitMix64};
use equity::sha256::sha256_hex;
use nn::encode::{Features, N_FEATURES, encode};
use nn::fast::{FastMlp, Scratch};
use nn::kernel::{Level, detected};
use nn::mlp::{IntMlp, Layer, MAX_WIDTH};
use nn::outcome::vocab;
use nn::synthetic::{SyntheticLeaf, random_net, reached_frontier};

fn levels() -> Vec<Level> {
    Level::ALL
        .into_iter()
        .filter(|&l| l <= detected())
        .collect()
}

/// Real frontier positions at the start of `trick`, encoded, as the solver asks for them.
fn features(trick: u8, max: usize) -> Vec<Features> {
    let p = reached_frontier(101 + u64::from(trick), trick, 400, max, &SyntheticLeaf);
    assert_eq!(p.len(), max, "enough positions at trick {trick}");
    p.iter()
        .map(|(c, x)| encode(c, x, &SyntheticLeaf).0)
        .collect()
}

/// The SHA-256 of the logits of `net` on `fs` (little-endian, every logit of every position).
fn logits_hash(
    fs: &[Features],
    mut z: impl FnMut(&Features) -> [i64; nn::outcome::MAX_OUT],
) -> String {
    let mut b = Vec::new();
    for f in fs {
        for x in z(f) {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    sha256_hex(&b)
}

/// The plain and the fast pass give `want` (the hash of the logits) on `fs`, the fast one at
/// every level; returns the plain hash for a report.
fn both_hash(net: &IntMlp, fs: &[Features]) -> String {
    let plain = logits_hash(fs, |f| net.forward(f));
    let fast = FastMlp::new(net.clone());
    for l in levels() {
        let mut s = Scratch::with_level(l);
        assert_eq!(
            logits_hash(fs, |f| fast.forward(f, &mut s)),
            plain,
            "level {l:?}"
        );
    }
    plain
}

#[test]
fn outputs_up_to_1024_wide_are_unchanged() {
    let fs = features(3, 300);
    // Recorded with the code of 54cfb3f (MAX_WIDTH 1024), plain and fast at every level.
    let want = [
        (
            vec![64, 32],
            "9a67722464a5b3c236b09bc6d5f259abce97928253a69577c03c94d1269b43f3",
        ),
        (
            vec![512, 256, 64],
            "9859913e3739cdae8772d4138d835943d2781fbb6fbbc3bbf68079a79ab0920b",
        ),
        (
            vec![1024, 512, 128],
            "8106d56dbe70b55794c42c4b5a71a1fbcf64ee4721b4357aa1ad6f6ec3ebd6b9",
        ),
        (
            vec![1024, 1024, 256],
            "76c707060a0d959c95b700c63f8c0102e087cce5ecbe52e744883ea6afa72edb",
        ),
        (
            vec![1024, 1024, 1024, 64],
            "f4986152c37d6981539ac8514096c0d8eb15d86f5a77e31f08bb7045506d9584",
        ),
    ];
    for (widths, h) in want {
        let net = random_net(31, &widths, vocab(3), 100);
        assert_eq!(both_hash(&net, &fs), h, "{widths:?}");
    }
}

/// The fast pass equals the reference on `fs` in order, then shuffled, at every level.
fn check_net(net: &IntMlp, fs: &[Features], what: &str) {
    let fast = FastMlp::new(net.clone());
    let want: Vec<_> = fs.iter().map(|f| net.forward(f)).collect();
    let mut order: Vec<usize> = (0..fs.len()).collect();
    let mut rng = Rng::derive(9, &[fs.len() as u64]);
    for i in (1..order.len()).rev() {
        order.swap(i, rng.bounded(i as u64 + 1) as usize);
    }
    for l in levels() {
        let mut s = Scratch::with_level(l);
        for (i, f) in fs.iter().enumerate() {
            assert_eq!(
                fast.forward(f, &mut s),
                want[i],
                "{what} level {l:?} row {i}"
            );
        }
        for &i in &order {
            assert_eq!(
                fast.forward(&fs[i], &mut s),
                want[i],
                "{what} level {l:?} row {i} shuffled"
            );
        }
        // In batches, shuffled, one past the largest batch.
        check_batches(
            &fast,
            fs,
            &want,
            &order,
            nn::kernel::MAX_BATCH + 1,
            &mut s,
            &format!("{what} level {l:?} shuffled"),
        );
    }
}

/// [`FastMlp::forward_batch`] equals the one-at-a-time logits `want` on `fs`, in batches of `b`
/// rows taken in the given order (every batch order at every level is checked against plain
/// sums by the kernel tests); returns how many comparisons were made.
fn check_batches(
    fast: &FastMlp,
    fs: &[Features],
    want: &[[i64; nn::outcome::MAX_OUT]],
    order: &[usize],
    b: usize,
    s: &mut Scratch,
    what: &str,
) -> usize {
    let mut n = 0;
    for chunk in order.chunks(b) {
        let bases: Vec<Vec<i64>> = chunk
            .iter()
            .map(|&i| {
                let mut v = vec![0; fast.base_len()];
                fast.base(&fs[i].dense, s, &mut v);
                v
            })
            .collect();
        let items: Vec<(&[u16], &[i64])> = chunk
            .iter()
            .zip(&bases)
            .map(|(&i, v)| (fs[i].active(), v.as_slice()))
            .collect();
        let mut got = vec![[7i64; nn::outcome::MAX_OUT]; chunk.len()];
        fast.forward_batch(&items, s, &mut got);
        for (k, &i) in chunk.iter().enumerate() {
            assert_eq!(got[k], want[i], "{what} batch {b} row {i}");
            n += 1;
        }
    }
    n
}

/// Widths 1024, 2048 and 4096: first layers that wide, and fan-ins that wide into the next
/// layer, with random weights of a trained network's scale (and three times it), on real
/// positions of tricks 3 and 4 (the networks of the next runs; trick 2's are made by the same
/// code, `fast_forward_equals_the_reference_on_random_networks_and_real_positions` in `fast.rs`).
#[test]
fn fast_equals_plain_at_1024_2048_and_4096() {
    assert_eq!(MAX_WIDTH, 4096);
    let nets: [(&[usize], u64, usize); 7] = [
        (&[1024, 512, 128], 100, 150),
        (&[2048, 1024, 256], 100, 120),
        (&[2048, 2048, 1024, 256], 100, 60),
        (&[2048, 1024, 256], 300, 60),
        (&[4096, 256], 100, 60),
        (&[512, 4096, 64], 100, 60),
        (&[4096, 2048, 64], 100, 40),
    ];
    for trick in [3u8, 4] {
        let fs = features(trick, 150);
        for (k, &(widths, pct, n)) in nets.iter().enumerate() {
            let net = random_net(40 + k as u64, widths, vocab(trick), pct);
            check_net(
                &net,
                &fs[..n],
                &format!("trick {trick} {widths:?} at {pct}%"),
            );
        }
    }
}

/// A network of the given widths with weights and biases at the format's limits: for each
/// output, a bias and then weights of `i32::MIN`, `i32::MAX` and other sizes while the bound
/// allows (most of a wide layer's weights stay small: the bound is shared by all its inputs).
fn extreme_net(seed: u64, widths: &[usize], n_out: usize) -> IntMlp {
    let mut sm = SplitMix64::new(seed);
    let mut dims = vec![N_FEATURES];
    dims.extend_from_slice(widths);
    dims.push(n_out);
    let layers = (0..dims.len() - 1)
        .map(|k| {
            let (n_in, n_out) = (dims[k], dims[k + 1]);
            let max_in: i128 = if k == 0 { 1 << 24 } else { 64 << 20 };
            let mut w = vec![0i32; n_in * n_out];
            let mut b = vec![0i64; n_out];
            for j in 0..n_out {
                let mut budget: i128 = 1 << 62;
                let bias = match sm.next_u64() % 3 {
                    0 => 0,
                    1 => (1i64 << 61) - (sm.next_u64() % 1000) as i64,
                    _ => -((sm.next_u64() >> 4) as i64),
                };
                budget -= i128::from(bias).abs();
                b[j] = bias;
                for _ in 0..2 * n_in {
                    let i = (sm.next_u64() % n_in as u64) as usize;
                    let v: i32 = match sm.next_u64() % 4 {
                        0 => i32::MIN,
                        1 => i32::MAX,
                        2 => (sm.next_u64() >> 40) as i32,
                        _ => -((sm.next_u64() >> 36) as i32),
                    };
                    let cost = (i128::from(v).abs() - i128::from(w[i * n_out + j]).abs()) * max_in;
                    if cost <= budget {
                        budget -= cost;
                        w[i * n_out + j] = v;
                    }
                }
            }
            let shift = match sm.next_u64() % 4 {
                0 => 0,
                1 => 62,
                _ => (sm.next_u64() % 63) as u32,
            };
            Layer {
                n_in,
                n_out,
                w,
                b,
                shift,
            }
        })
        .collect();
    IntMlp::new(layers).expect("within the format's bounds")
}

/// At the limits, wide: sums that reach the 2^62 bound over 2048 and 4096 inputs.
#[test]
fn fast_equals_plain_at_the_limits_when_wide() {
    let fs = features(3, 120);
    for (seed, widths) in [
        (1u64, &[2048usize, 64][..]),
        (2, &[64, 4096, 32]),
        (3, &[4096, 16]),
    ] {
        let net = extreme_net(seed, widths, vocab(3));
        check_net(&net, &fs, &format!("extreme {widths:?}"));
    }
    // Every input at the clip and every weight at the largest the bound allows at a fan-in of
    // 4096: each logit is exactly ±2^62 before its shift.
    let at_bound = IntMlp::new(vec![
        Layer {
            n_in: N_FEATURES,
            n_out: 4096,
            w: vec![0; N_FEATURES * 4096],
            b: vec![(64i64 << 20) << 24; 4096],
            shift: 24,
        },
        Layer {
            n_in: 4096,
            n_out: 4,
            w: [1 << 24, -(1 << 24), 1 << 24, -(1 << 24)].repeat(4096),
            b: vec![0; 4],
            shift: 40,
        },
    ])
    .expect("at the bound");
    check_net(&at_bound, &fs[..20], "at the bound");
}

/// Golden hashes of the logits of seeded wide networks on real positions (plain and fast at
/// every level give them; any change of an output changes a hash).
#[test]
fn wide_golden_outputs() {
    let fs = features(3, 150);
    let want = [
        (
            vec![2048, 1024, 256],
            "eb6a08dce36182e2e6474dd577df29be593508da9940a37e32e484793f8e001b",
        ),
        (
            vec![2048, 2048, 1024, 256],
            "bcbfe9c1295847a460a5a7df886852b43ade744b84a6f29062d60e8d8f1ec661",
        ),
        (
            vec![4096, 2048, 256],
            "e07938e9aa86cc350d2ee4241a0b42498e36e74511331cc661e56a6d20a57984",
        ),
    ];
    for (widths, h) in want {
        let net = random_net(77, &widths, vocab(3), 100);
        assert_eq!(both_hash(&net, &fs), h, "{widths:?}");
    }
}

/// Which networks the solver batches by default (`nn::oracle::default_prefetch`): those with a
/// hidden layer of at least 2048 × 1024 weights.
#[test]
fn wide_networks_batch_by_default() {
    use nn::oracle::{WIDE_PREFETCH, default_prefetch};
    let zero = |widths: &[usize]| -> IntMlp {
        let mut dims = vec![N_FEATURES];
        dims.extend_from_slice(widths);
        dims.push(4);
        IntMlp::new(
            dims.windows(2)
                .map(|d| Layer {
                    n_in: d[0],
                    n_out: d[1],
                    w: vec![0; d[0] * d[1]],
                    b: vec![0; d[1]],
                    shift: 0,
                })
                .collect(),
        )
        .expect("valid")
    };
    assert_eq!(default_prefetch(&zero(&[512, 256, 64])), 0);
    assert_eq!(default_prefetch(&zero(&[1024, 512, 128])), 0);
    assert_eq!(default_prefetch(&zero(&[1024, 1024, 256])), 0);
    assert_eq!(default_prefetch(&zero(&[2048, 1024, 256])), WIDE_PREFETCH);
    assert_eq!(
        default_prefetch(&zero(&[2048, 2048, 1024, 256])),
        WIDE_PREFETCH
    );
    assert_eq!(default_prefetch(&zero(&[1024, 2048, 256])), WIDE_PREFETCH);
    assert_eq!(WIDE_PREFETCH, 3);
}
