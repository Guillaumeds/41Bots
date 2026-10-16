//! The fast evaluation path gives exactly the plain one (PLAN §10: speed only through
//! engineering that gives identical answers):
//! - [`FastMlp::forward`] equals [`IntMlp::forward`] (the plain reference, unchanged) bit for
//!   bit, at every instruction-set level this machine has, on random networks of several sizes
//!   (512-256-64 and 1024-512-128 among them), on networks with weights and biases at the limits
//!   the format allows (including first layers whose binary sums need `i64`, and incremental
//!   updates that wrap `i32` on the way), and on thousands of real frontier positions at tricks
//!   2, 3 and 4 in the order a solve asks for them (so the incremental first layer runs as it
//!   does in a search), then shuffled, and with one scratch shared by two networks;
//! - [`Oracle::eval`] (caches and memo) equals [`Oracle::eval_reference`] for both heads, with
//!   the memo on and off, with several oracles interleaved in one thread, and from several
//!   threads;
//! - the fast forward pass gives the committed golden outputs that `python/nn` also gives.

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::rng::{Rng, SplitMix64};
use nn::encode::{Features, MAX_ACTIVE, N_BINARY, N_DENSE, N_FEATURES, PAD, encode};
use nn::fast::{FastMlp, Scratch};
use nn::kernel::{Level, detected};
use nn::mlp::{IntMlp, Layer};
use nn::money::money_from_logits;
use nn::oracle::{Oracle, write_net};
use nn::outcome::{money_from_outcomes, vocab};
use nn::synthetic::{
    SyntheticLeaf, golden_features, golden_net, golden_net_out, golden_outcomes, random_net,
    reached_frontier,
};
use solver::{CanonPos, MatchCtx};

fn levels() -> Vec<Level> {
    Level::ALL
        .into_iter()
        .filter(|&l| l <= detected())
        .collect()
}

/// Real frontier positions at the start of `trick`, as the solver asks for them.
fn positions(trick: u8, max: usize) -> Vec<(CanonPos, MatchCtx)> {
    let p = reached_frontier(101 + u64::from(trick), trick, 400, max, &SyntheticLeaf);
    assert_eq!(p.len(), max, "enough positions at trick {trick}");
    p
}

fn features(pos: &[(CanonPos, MatchCtx)]) -> Vec<Features> {
    pos.iter()
        .map(|(c, x)| encode(c, x, &SyntheticLeaf).0)
        .collect()
}

/// The fast forward pass equals the reference on `fs` in order, then in a shuffled order, at
/// every level; returns how many comparisons were made.
fn check_net(net: &IntMlp, fs: &[Features], what: &str) -> usize {
    let fast = FastMlp::new(net.clone());
    let want: Vec<_> = fs.iter().map(|f| net.forward(f)).collect();
    let mut order: Vec<usize> = (0..fs.len()).collect();
    let mut rng = Rng::derive(7, &[fs.len() as u64]);
    for i in (1..order.len()).rev() {
        order.swap(i, rng.bounded(i as u64 + 1) as usize);
    }
    let mut n = 0;
    for l in levels() {
        let mut s = Scratch::with_level(l);
        for (i, f) in fs.iter().enumerate() {
            assert_eq!(
                fast.forward(f, &mut s),
                want[i],
                "{what} level {l:?} row {i}"
            );
            n += 1;
        }
        for &i in &order {
            assert_eq!(
                fast.forward(&fs[i], &mut s),
                want[i],
                "{what} level {l:?} row {i} shuffled"
            );
            n += 1;
        }
        // In batches: in order (neighbours alike, as the solver gathers them), then shuffled,
        // one past the largest batch.
        let inorder: Vec<usize> = (0..fs.len()).collect();
        n += check_batches(
            &fast,
            fs,
            &want,
            &inorder,
            8,
            &mut s,
            &format!("{what} level {l:?}"),
        );
        n += check_batches(
            &fast,
            fs,
            &want,
            &order,
            nn::kernel::MAX_BATCH + 1,
            &mut s,
            &format!("{what} level {l:?} shuffled"),
        );
    }
    n
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

#[test]
fn fast_forward_equals_the_reference_on_random_networks_and_real_positions() {
    let mut n = 0;
    for trick in [2u8, 3, 4] {
        let fs = features(&positions(trick, 1500));
        let v = vocab(trick);
        for (k, widths) in [vec![512, 256, 64], vec![64, 32], vec![48, 40, 24, 8]]
            .into_iter()
            .enumerate()
        {
            let net = random_net(11 + k as u64, &widths, v, 100);
            n += check_net(&net, &fs, &format!("trick {trick} widths {widths:?}"));
        }
        // The large network on fewer positions (its reference pass is slow in tests).
        let net = random_net(5, &[1024, 512, 128], v, 100);
        n += check_net(&net, &fs[..300], &format!("trick {trick} 1024-512-128"));
        // Sparser and denser hidden layers.
        for pct in [30, 300] {
            let net = random_net(9, &[256, 128], v, pct);
            n += check_net(&net, &fs, &format!("trick {trick} scale {pct}%"));
        }
    }
    assert!(n > 50_000, "{n}");
}

/// A network with weights and biases at the format's limits (every partial sum within the
/// `2^62` bound, as [`IntMlp::new`] requires): columns of `i32::MIN` / `i32::MAX` weights,
/// biases taking up to the rest of the bound, and any shifts.
fn extreme_net(seed: u64, widths: &[usize], n_out: usize, bin_i32: bool) -> IntMlp {
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
                // The bias first, then weights while the bound allows.
                let bias = match sm.next_u64() % 3 {
                    0 => 0,
                    1 => (1i64 << 61) - (sm.next_u64() % 1000) as i64,
                    _ => -((sm.next_u64() >> 4) as i64),
                };
                budget -= i128::from(bias).abs();
                b[j] = bias;
                for _ in 0..3 * n_in {
                    let i = (sm.next_u64() % n_in as u64) as usize;
                    let v: i32 = if k == 0 && bin_i32 {
                        // Large, and the sum of any MAX_ACTIVE of them still fits i32.
                        let m = i32::MAX / MAX_ACTIVE as i32;
                        if sm.next_u64().is_multiple_of(2) {
                            m
                        } else {
                            -m
                        }
                    } else {
                        match sm.next_u64() % 4 {
                            0 => i32::MIN,
                            1 => i32::MAX,
                            2 => (sm.next_u64() >> 40) as i32,
                            _ => -((sm.next_u64() >> 36) as i32),
                        }
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

/// Feature vectors that drive extreme networks: random active binary inputs (up to
/// [`MAX_ACTIVE`], ascending) and dense values at and between the limits.
fn wild_features(seed: u64, count: usize) -> Vec<Features> {
    let mut sm = SplitMix64::new(seed);
    (0..count)
        .map(|_| {
            let mut f = Features {
                active: [PAD; MAX_ACTIVE],
                n_active: 0,
                dense: [0; N_DENSE],
            };
            let p = 1 + sm.next_u64() % 20;
            for i in 0..N_BINARY {
                if sm.next_u64() % 100 < p && usize::from(f.n_active) < MAX_ACTIVE {
                    f.active[usize::from(f.n_active)] = i as u16;
                    f.n_active += 1;
                }
            }
            for d in f.dense.iter_mut() {
                *d = match sm.next_u64() % 4 {
                    0 => 0,
                    1 => -(1 << 24),
                    _ => -((sm.next_u64() % ((1 << 24) + 1)) as i32),
                };
            }
            f
        })
        .collect()
}

#[test]
fn fast_forward_equals_the_reference_at_the_limits() {
    let mut n = 0;
    let real = features(&positions(3, 400));
    for seed in 0..6u64 {
        for bin_i32 in [true, false] {
            let net = extreme_net(seed, &[96, 48], vocab(2 + (seed % 3) as u8), bin_i32);
            let fast = FastMlp::new(net.clone());
            assert_eq!(fast.bin_i32(), bin_i32, "seed {seed}");
            n += check_net(
                &net,
                &wild_features(seed, 200),
                &format!("extreme {seed} {bin_i32}"),
            );
            n += check_net(&net, &real, &format!("extreme {seed} {bin_i32} real"));
        }
    }
    // A first layer whose rows all equal the largest weight that keeps the binary sums in i32:
    // removing rows after adding others wraps i32 on the way and must still give the sum.
    let m = i32::MAX / MAX_ACTIVE as i32;
    let mut net = random_net(3, &[64, 32], 14, 100).layers().to_vec();
    for w in net[0].w[..N_BINARY * 64].iter_mut() {
        *w = m;
    }
    let net = IntMlp::new(net).expect("valid");
    assert!(FastMlp::new(net.clone()).bin_i32());
    let mut fs = Vec::new();
    // Full sums at the limit: two disjoint sets of MAX_ACTIVE inputs in turn (each change is
    // too large for an update, so each sum is made afresh).
    for k in 0..100usize {
        let mut f = Features {
            active: [PAD; MAX_ACTIVE],
            n_active: MAX_ACTIVE as u8,
            dense: [-(1 << 20); N_DENSE],
        };
        for (t, a) in f.active.iter_mut().enumerate() {
            *a = (2 * t + k % 2) as u16;
        }
        fs.push(f);
    }
    // Updates that wrap: MAX_ACTIVE consecutive inputs moved by one to six places each time.
    // The sum of MAX_ACTIVE rows is within 31 of i32::MAX, so the rows added (before the rows
    // removed) take the running sum past it, and wrapping brings it back exactly.
    for k in 0..300usize {
        let mut f = fs[0];
        for (t, a) in f.active.iter_mut().enumerate() {
            *a = (t + k % 7) as u16;
        }
        fs.push(f);
    }
    n += check_net(&net, &fs, "wrapping i32 sums");
    assert!(n > 20_000, "{n}");
}

#[test]
fn a_scratch_shared_by_two_networks_starts_afresh() {
    let fs = features(&positions(4, 300));
    let a = random_net(1, &[64, 32], 14, 100);
    let b = random_net(2, &[64, 32], 14, 100);
    let (fa, fb) = (FastMlp::new(a.clone()), FastMlp::new(b.clone()));
    for l in levels() {
        let mut s = Scratch::with_level(l);
        for (i, f) in fs.iter().enumerate() {
            let (net, fast) = if i % 3 == 0 { (&b, &fb) } else { (&a, &fa) };
            assert_eq!(
                fast.forward(f, &mut s),
                net.forward(f),
                "row {i} level {l:?}"
            );
        }
    }
}

#[test]
fn fast_forward_gives_the_golden_outputs() {
    let text = include_str!("data/intnet-golden.json");
    let v: serde_json::Value = serde_json::from_str(text).expect("json");
    let seed = v["seed"].as_u64().expect("seed");
    let count = v["count"].as_u64().expect("count") as usize;
    let want: Vec<[u64; 4]> = serde_json::from_value(v["outputs"].clone()).expect("outputs");
    let fs = golden_features(seed + 1, count);
    let net = FastMlp::new(golden_net(seed));
    let n_out = v["outcome"]["n_out"].as_u64().expect("n_out") as usize;
    let owant: Vec<[u64; 4]> =
        serde_json::from_value(v["outcome"]["outputs"].clone()).expect("outputs");
    let onet = FastMlp::new(golden_net_out(seed, n_out));
    let tables = golden_outcomes(seed + 2, count, n_out);
    for l in levels() {
        let mut s = Scratch::with_level(l);
        let got: Vec<[u64; 4]> = fs
            .iter()
            .map(|f| money_from_logits(&net.forward(f, &mut s)).0)
            .collect();
        assert_eq!(got, want, "level {l:?}");
        let got: Vec<[u64; 4]> = fs
            .iter()
            .zip(&tables)
            .map(|(f, o)| money_from_outcomes(&onet.forward(f, &mut s), o).0)
            .collect();
        assert_eq!(got, owant, "level {l:?}");
    }
}

fn oracle(net: &IntMlp, trick: u8, outcome_head: bool) -> Oracle {
    let mut meta = BTreeMap::new();
    if outcome_head {
        meta.insert("head".to_string(), "outcome-v1".to_string());
    }
    Oracle::from_bytes(&write_net(net, trick, &meta), Arc::new(SyntheticLeaf)).expect("loads")
}

#[test]
fn oracle_eval_equals_the_plain_evaluation() {
    let mut n = 0;
    for trick in [2u8, 3, 4] {
        let pos = positions(trick, 1200);
        let outcome = oracle(&random_net(21, &[128, 64], vocab(trick), 100), trick, true);
        let money4 = oracle(&random_net(22, &[96, 32], 4, 100), trick, false);
        let mut nomemo = oracle(&random_net(21, &[128, 64], vocab(trick), 100), trick, true);
        nomemo.set_memo(false);
        let want: Vec<_> = pos
            .iter()
            .map(|(c, x)| (outcome.eval_reference(c, x), money4.eval_reference(c, x)))
            .collect();
        // Interleaved in one thread, each position asked for twice (the memo answers the
        // second time).
        for _ in 0..2 {
            for ((c, x), w) in pos.iter().zip(&want) {
                assert_eq!(outcome.eval(c, x), w.0, "trick {trick}");
                assert_eq!(money4.eval(c, x), w.1, "trick {trick}");
                assert_eq!(nomemo.eval(c, x), w.0, "trick {trick} no memo");
                n += 3;
            }
        }
        let st = outcome.thread_stats();
        assert_eq!(st.requests, 2 * pos.len() as u64);
        assert!(st.memo_hits >= pos.len() as u64 / 2, "{st:?}");
        assert_eq!(nomemo.thread_stats().memo_hits, 0);
        // From several threads at once, each in its own order.
        std::thread::scope(|s| {
            for t in 0..3usize {
                let (pos, want, outcome) = (&pos, &want, &outcome);
                s.spawn(move || {
                    for i in 0..pos.len() {
                        let k = (i * (2 * t + 1) + t) % pos.len();
                        let (c, x) = &pos[k];
                        assert_eq!(outcome.eval(c, x), want[k].0, "thread {t} position {k}");
                    }
                });
            }
        });
    }
    assert!(n > 20_000, "{n}");
}

/// [`Oracle::eval_batch`] equals [`Oracle::eval_reference`] position by position, for both
/// heads and a first layer that needs `i64` sums, in batches of several sizes (one past the
/// largest batch included), with positions repeated inside a batch and batches asked for again
/// (the memo answers), and with the memo off.
#[test]
fn oracle_batches_equal_the_plain_evaluation() {
    let mut n = 0;
    for trick in [2u8, 3, 4] {
        let pos = positions(trick, 700);
        // One match context per batch: group the positions by it, in order.
        let mut groups: BTreeMap<([u8; 4], u8), Vec<CanonPos>> = BTreeMap::new();
        for (c, x) in &pos {
            groups.entry((x.games, x.dealer_team)).or_default().push(*c);
        }
        let nets = [
            (random_net(21, &[128, 64], vocab(trick), 100), true),
            (random_net(22, &[96, 32], 4, 100), false),
            (random_net(23, &[64, 32], vocab(trick), 1500), true),
        ];
        assert!(
            !FastMlp::new(nets[2].0.clone()).bin_i32(),
            "a wide first layer"
        );
        for (k, (net, outcome_head)) in nets.iter().enumerate() {
            let o = oracle(net, trick, *outcome_head);
            let mut nomemo = oracle(net, trick, *outcome_head);
            nomemo.set_memo(false);
            for (&(games, dealer_team), cs) in &groups {
                let x = MatchCtx { games, dealer_team };
                let want: Vec<_> = cs.iter().map(|c| o.eval_reference(c, &x)).collect();
                for b in [1usize, 4, 64, 65] {
                    for (chunk, w) in cs.chunks(b).zip(want.chunks(b)) {
                        // Each position twice in a row, then the batch again.
                        let twice: Vec<CanonPos> = chunk.iter().flat_map(|c| [*c, *c]).collect();
                        let mut got = vec![equity::MoneyChance::certain(0); twice.len()];
                        for o in [&o, &nomemo] {
                            o.eval_batch(&twice, &x, &mut got);
                            for (i, v) in got.iter().enumerate() {
                                assert_eq!(*v, w[i / 2], "trick {trick} net {k} batch {b}");
                                n += 1;
                            }
                            o.eval_batch(chunk, &x, &mut got);
                            assert_eq!(&got[..chunk.len()], w, "trick {trick} net {k} again");
                        }
                    }
                }
            }
            let st = o.thread_stats();
            assert!(st.batched > 0 && st.batch_hits > st.batched / 2, "{st:?}");
        }
    }
    assert!(n > 20_000, "{n}");
}

/// The hybrid's search with the frontier batches (docs/ENGINEERING-2026-10-17.md, item B) is the
/// search without them: on trick-3 roots (frontier at trick 4) and trick-2 roots (frontier at
/// trick 3), at every place in the trick, the solve's answer and every counter, and the
/// sequence of frontier positions it asks the oracle for with their values, are the same for
/// batches gathered 1, 2, 3 or 5 plays before the frontier as with none; and every value is the
/// plain evaluation's.
#[test]
fn hybrid_with_batches_equals_the_hybrid_without() {
    use std::sync::Mutex;

    use solver::frontier::Frontier;
    use solver::{FastOptions, FrontierCut, Position, SolveMem, solve_fast_cut};

    /// The oracle, logging what the solver asks for, batching from `depth` plays.
    struct Logged {
        o: Oracle,
        depth: u8,
        log: Mutex<Vec<(CanonPos, equity::MoneyChance)>>,
    }
    impl Frontier for Logged {
        fn value(&self, pos: &CanonPos, ctx: &MatchCtx) -> equity::MoneyChance {
            self.o.eval(pos, ctx)
        }
        fn value_at(
            &self,
            _pos: &Position,
            canon: &CanonPos,
            ctx: &MatchCtx,
        ) -> equity::MoneyChance {
            let v = self.o.eval(canon, ctx);
            self.log.lock().expect("no panics").push((*canon, v));
            v
        }
        fn prefetch_depth(&self) -> u8 {
            self.depth
        }
        fn prefetch(&self, pos: &[CanonPos], ctx: &MatchCtx) {
            self.o.prefetch(pos, ctx);
        }
    }

    let mut asked = 0;
    let mut batched = 0;
    for (k, cut, worlds) in [(3u8, 4u8, 16u64), (2, 3, 3)] {
        let net = random_net(31, &[256, 128, 32], vocab(cut), 100);
        for w in 0..worlds {
            let (pos, ctx, _) = solver::worlds::sample_world(77, k, w, (w % 8) as u8);
            let mut want = None;
            for depth in [0u8, 1, 2, 3, 5] {
                let f = Logged {
                    o: oracle(&net, cut, true),
                    depth,
                    log: Mutex::new(Vec::new()),
                };
                let sol = solve_fast_cut(
                    &pos,
                    ctx,
                    &SyntheticLeaf,
                    FastOptions::default(),
                    Some(FrontierCut::new(cut, &f)),
                    &mut SolveMem::new(),
                    None,
                );
                let st = f.o.thread_stats();
                let log = f.log.into_inner().expect("no panics");
                if depth > 0 {
                    batched += st.batched;
                }
                match &want {
                    None => {
                        for (i, (c, v)) in log.iter().enumerate() {
                            if i % 5 == 0 {
                                assert_eq!(f.o.eval_reference(c, &ctx), *v, "world {w}");
                            }
                        }
                        asked += log.len();
                        want = Some((sol, log));
                    }
                    Some((s0, l0)) => {
                        let what = format!("trick {k} world {w} depth {depth}");
                        assert_eq!(s0.value, sol.value, "{what}");
                        assert_eq!(s0.best, sol.best, "{what}");
                        assert_eq!(s0.children, sol.children, "{what}");
                        assert_eq!(s0.root_tied, sol.root_tied, "{what}");
                        assert_eq!(s0.stats, sol.stats, "{what}: counters");
                        assert_eq!(*l0, log, "{what}: requests and values");
                    }
                }
            }
        }
    }
    assert!(asked > 3_000, "{asked}");
    assert!(batched > 3_000, "{batched}");
}
