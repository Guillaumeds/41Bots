//! The encoder, the integer network and the oracle as a frontier:
//! - the encoding is complete (it decodes back to the rotated canonical form and match context)
//!   and rotation-invariant (every rotation of a position gives the same features);
//! - the integer forward pass and softmax give the golden outputs that `python/nn`'s integer
//!   forward pass also gives (`tests/data/intnet-golden.json`);
//! - an oracle network loaded from a weights file is a frontier evaluator under which the
//!   optimised and parallel hybrid solves equal the naive one, and its values rotate with the
//!   teams.

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::rng::Rng;
use nn::encode::{MAX_ACTIVE, N_DENSE, decode, encode, rotate_canon, rotation};
use nn::money::money_from_logits;
use nn::oracle::{Oracle, write_net};
use nn::synthetic::{SyntheticLeaf, golden_features, golden_net};
use solver::worlds::sample_world;
use solver::{
    FastOptions, FrontierCut, MatchCtx, Options, Position, SolveMem, Step, TieOrder, canonical,
    solve_cut, solve_fast_cut, solve_fast_par_cut,
};

/// A random position at the start of trick `k` (random deals and play), with its context.
fn frontier_position(seed: u64, k: u8, i: u64) -> (Position, MatchCtx) {
    let (p, ctx, _) = sample_world(seed, k, i, 0);
    (p, ctx)
}

#[test]
fn encoding_is_complete_and_rotation_invariant() {
    let leaf = SyntheticLeaf;
    let mut n = 0;
    for k in 2..=5u8 {
        for i in 0..400 {
            let (p, ctx) = frontier_position(11, k, i);
            let c = canonical(&p);
            let (f, r) = encode(&c, &ctx, &leaf);
            assert_eq!(r, rotation(&c));
            assert!(usize::from(f.n_active) <= MAX_ACTIVE);
            assert!(f.dense.iter().all(|&d| d <= 1 << 15));
            let (dc, dctx) = decode(&f).expect("decodes");
            let (rc, rctx) = rotate_canon(&c, &ctx, r);
            assert_eq!((dc, dctx), (rc, rctx), "trick {k} position {i}");
            for s in 1..8 {
                let (c2, ctx2) = rotate_canon(&c, &ctx, s);
                let (f2, r2) = encode(&c2, &ctx2, &leaf);
                assert_eq!(f2, f, "trick {k} position {i} rotation {s}");
                assert_eq!((r2 + s) % 8, r);
            }
            n += 1;
        }
    }
    assert_eq!(n, 1600);
}

#[test]
fn integer_network_matches_the_golden_outputs() {
    let text = include_str!("data/intnet-golden.json");
    let v: serde_json::Value = serde_json::from_str(text).expect("json");
    let seed = v["seed"].as_u64().expect("seed");
    let count = v["count"].as_u64().expect("count") as usize;
    let want: Vec<[u64; 4]> = serde_json::from_value(v["outputs"].clone()).expect("outputs");
    let net = golden_net(seed);
    let fs = golden_features(seed + 1, count);
    let got: Vec<[u64; 4]> = fs
        .iter()
        .map(|f| money_from_logits(&net.forward(f)).0)
        .collect();
    assert_eq!(got, want);
    // The golden outputs are not degenerate: most are strictly between 0 and 1.
    let mid = got.iter().filter(|o| o.iter().all(|&x| x > 0)).count();
    assert!(mid > count / 4, "{mid} of {count}");
    let _ = N_DENSE;
}

fn golden_oracle(trick: u8) -> Oracle {
    let bytes = write_net(&golden_net(41), trick, &BTreeMap::new());
    Oracle::from_bytes(&bytes, Arc::new(SyntheticLeaf)).expect("loads")
}

#[test]
fn oracle_values_rotate_with_the_teams() {
    let o = golden_oracle(4);
    for i in 0..300 {
        let (p, ctx) = frontier_position(5, 3 + (i % 3) as u8, i);
        let c = canonical(&p);
        let v = o.eval(&c, &ctx);
        for s in 1..8 {
            let (c2, ctx2) = rotate_canon(&c, &ctx, s);
            assert_eq!(
                o.eval(&c2, &ctx2),
                v.rotate(s % 4),
                "position {i} rotation {s}"
            );
        }
    }
}

/// A position `extra` random plays after the start of trick `k`.
fn mid(seed: u64, k: u8, extra: u8) -> (Position, MatchCtx) {
    let (mut p, ctx) = frontier_position(seed, k, 0);
    let mut rng = Rng::derive(seed, &[0x41D]);
    for _ in 0..extra {
        let cs: Vec<u8> = engine::cards::cards(p.legal()).collect();
        match p.play(cs[rng.bounded(cs.len() as u64) as usize]) {
            Step::Next(q) => p = q,
            _ => break,
        }
    }
    (p, ctx)
}

#[test]
fn oracle_frontier_hybrid_identity() {
    let leaf = SyntheticLeaf;
    let o = golden_oracle(4);
    let mut evals = 0;
    for seed in 0..10 {
        for (k, extra, cut) in [(3, 0, 4), (3, 4, 4), (3, 2, 5), (4, 1, 5)] {
            let (p, ctx) = mid(seed, k, extra);
            for order in [TieOrder::Ascending, TieOrder::Descending] {
                let c = Some(FrontierCut::new(cut, &o));
                let naive = solve_cut(
                    &p,
                    ctx,
                    &leaf,
                    Options {
                        tt: true,
                        order,
                        ..Options::default()
                    },
                    c,
                );
                evals += naive.stats.frontier;
                for bits in [15, 0, 2, 13] {
                    let opts = FastOptions {
                        order,
                        ..FastOptions::default()
                    }
                    .combo(bits);
                    let f = solve_fast_cut(&p, ctx, &leaf, opts, c, &mut SolveMem::new(), None);
                    assert_eq!(
                        (f.value, f.best, &f.children),
                        (naive.value, naive.best, &naive.children),
                        "seed {seed} trick {k}+{extra} cut {cut} {order:?} combo {bits}"
                    );
                    let par =
                        solve_fast_par_cut(&p, ctx, &leaf, opts, 3, None, &mut SolveMem::new(), c);
                    assert_eq!(
                        (par.value, par.best, &par.children),
                        (naive.value, naive.best, &naive.children),
                        "seed {seed} trick {k}+{extra} cut {cut} {order:?} combo {bits} par"
                    );
                }
            }
        }
    }
    assert!(evals > 1000, "{evals}");
}
