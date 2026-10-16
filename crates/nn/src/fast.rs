//! The fast integer forward pass: the same function as [`IntMlp::forward`] (the plain reference,
//! kept unchanged), computed with less work. Every value is the same integer the reference
//! computes, bit for bit (tested against it on random and extreme networks and real positions,
//! at every instruction-set level the machine has: `tests/fast.rs`; and by `nn-tool eval`, which
//! the Python parity check runs).
//!
//! What changes (PLAN §10: engineering that gives identical answers):
//! - **Kernels** ([`crate::kernel`]): only the rows of non-zero inputs are read (as before);
//!   activations are held as `i32` (they lie in `0..=2^26`) and weights in the packed-pair layout
//!   ([`pack_pairs`]), so each product is one 32 × 32 → 64-bit signed multiply that the compiler
//!   vectorises; compiled for x86-64-v2/v3/v4 with runtime dispatch (`multiversion`).
//! - **First layer split** into its dense part and its binary part. The dense part
//!   (`b_j + Σ_d w_dj · x_d` over the 100 equity inputs) is computed once per dense input vector
//!   and can be reused by the caller ([`FastMlp::base`], cached by [`crate::oracle`]). The binary
//!   part (`Σ` of the weight rows of the active binary inputs) is kept between calls in the
//!   [`Scratch`] and updated by the rows that switched on and off (an incremental first layer).
//! - **Narrow first-layer sums.** When the network allows it ([`FastMlp::bin_i32`]), the binary
//!   part is summed in wrapping `i32`: for every output `j` the sum of the [`MAX_ACTIVE`] largest
//!   `|w_ij|` over all inputs is at most `2^31 − 1`, so the true sum of the rows of any position
//!   (at most [`MAX_ACTIVE`] rows) fits `i32`, and wrapping arithmetic (addition mod `2^32`)
//!   gives exactly it, whatever the order of the additions and removals that led there.
//!   Otherwise the binary part is summed in `i64` like the dense part.
//! - **No allocation per call**, and no branch per activation (branch-free compaction of the
//!   non-zero activations).
//!
//! **Overflow bounds.** [`IntMlp::new`] refuses a network unless, for every layer and output,
//! `|b_j| + Σ_i |w_ij| · max x_i ≤ 2^62` (max input `2^24` for the first layer, `2^26` after).
//! Every partial sum formed here is the bias plus a sub-sum of the reference's terms
//! `w_ij · x_i` (or a sub-sum without the bias), so its absolute value is at most that bound: no
//! `i64` sum can overflow, in any order. Each product `w_ij · x_i` is at most
//! `2^31 · 2^26 = 2^57`. Adding the rounding half (`2^(k−1) ≤ 2^61`) before the shift stays
//! below `2^63`. The binary sums use `i32` only under the bound above, and are widened before
//! they are scaled by `2^24`. Padded outputs (zero weights and biases) are never read.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::encode::{Features, INPUT_SHIFT, MAX_ACTIVE, N_BINARY, N_DENSE, N_FEATURES};
use crate::kernel::{
    Level, activate_at, activate0_at, compact, level, mac_pairs_at, mac_rows_at, pack_pairs,
    sum_rows_at,
};
use crate::mlp::{IntMlp, MAX_WIDTH, round_shift};
use crate::outcome::MAX_OUT;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Words of a bit set of the network's inputs.
const WORDS: usize = N_FEATURES.div_ceil(64);

/// A layer in the packed-pair layout ([`pack_pairs`]).
#[derive(Clone, Debug)]
struct Packed {
    /// Outputs.
    n: usize,
    /// Outputs padded to whole groups.
    np: usize,
    /// The weights, packed.
    wp: Vec<u64>,
    /// The biases, padded with zeros to `np`.
    b: Vec<i64>,
    /// The right shift after the sum.
    shift: u32,
}

impl Packed {
    fn of(w: &[i32], n_in: usize, n: usize, b: &[i64], shift: u32) -> Packed {
        let (wp, np) = pack_pairs(w, n_in, n);
        let mut b = b.to_vec();
        b.resize(np, 0);
        Packed {
            n,
            np,
            wp,
            b,
            shift,
        }
    }
}

/// A network prepared for the fast forward pass.
#[derive(Clone, Debug)]
pub struct FastMlp {
    net: IntMlp,
    /// The first layer's dense rows (inputs `N_BINARY..`), packed, with its biases.
    dense0: Packed,
    /// Layers 1.., packed.
    hidden: Vec<Packed>,
    bin_i32: bool,
    /// Identifies the network for the incremental state of a [`Scratch`].
    id: u64,
}

/// Reusable buffers and the incremental first-layer state of one thread's evaluations (no
/// allocation per evaluation). A scratch may be used with any network: it notices a change of
/// network and starts afresh.
#[derive(Clone, Debug)]
pub struct Scratch {
    /// The network whose binary sums `bin` holds (0: none).
    owner: u64,
    /// The active inputs that `bin` sums, as a bit set.
    prev_bits: [u64; WORDS],
    /// `Σ_{i ∈ prev} w_ij` in wrapping `i32`.
    bin: Vec<i32>,
    add: Vec<u16>,
    sub: Vec<u16>,
    acc: Vec<i64>,
    h: Vec<i32>,
    idx: Vec<u16>,
    val: Vec<i32>,
    base: Vec<i64>,
    ones: Vec<i32>,
    level: Option<Level>,
}

impl Default for Scratch {
    fn default() -> Self {
        Scratch {
            owner: 0,
            prev_bits: [0; WORDS],
            bin: Vec::with_capacity(MAX_WIDTH),
            add: Vec::with_capacity(MAX_ACTIVE),
            sub: Vec::with_capacity(MAX_ACTIVE),
            acc: vec![0; MAX_WIDTH],
            h: vec![0; MAX_WIDTH],
            idx: vec![0; MAX_WIDTH],
            val: vec![0; MAX_WIDTH],
            base: Vec::new(),
            ones: vec![1 << INPUT_SHIFT; MAX_ACTIVE],
            level: None,
        }
    }
}

impl Scratch {
    /// A new scratch.
    pub fn new() -> Self {
        Scratch::default()
    }

    /// Use the kernels of level `l` (at most the detected one) instead of the current level
    /// (tests and benchmarks; results never depend on it).
    pub fn with_level(l: Level) -> Self {
        Scratch {
            level: Some(l.min(crate::kernel::detected())),
            ..Scratch::default()
        }
    }

    fn level(&self) -> Level {
        self.level.unwrap_or_else(level)
    }
}

impl FastMlp {
    /// Prepare a checked network.
    pub fn new(net: IntMlp) -> FastMlp {
        let layers = net.layers();
        let l0 = &layers[0];
        let n = l0.n_out;
        // For each output, the sum of the MAX_ACTIVE largest |w_ij| over every input row.
        let mut col: Vec<u64> = Vec::with_capacity(l0.n_in);
        let mut bin_i32 = true;
        for j in 0..n {
            col.clear();
            col.extend((0..l0.n_in).map(|i| u64::from(l0.w[i * n + j].unsigned_abs())));
            let k = MAX_ACTIVE.min(col.len());
            if k < col.len() {
                col.select_nth_unstable_by(k - 1, |a, b| b.cmp(a));
            }
            let top: u64 = col[..k].iter().sum();
            if top > i32::MAX as u64 {
                bin_i32 = false;
                break;
            }
        }
        let dense0 = Packed::of(&l0.w[N_BINARY * n..], N_DENSE, n, &l0.b, l0.shift);
        let hidden = layers[1..]
            .iter()
            .map(|l| Packed::of(&l.w, l.n_in, l.n_out, &l.b, l.shift))
            .collect();
        FastMlp {
            dense0,
            hidden,
            bin_i32,
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            net,
        }
    }

    /// The network.
    pub fn net(&self) -> &IntMlp {
        &self.net
    }

    /// Whether the binary part of the first layer is summed in `i32` (see the module notes).
    pub fn bin_i32(&self) -> bool {
        self.bin_i32
    }

    /// The length of a [`FastMlp::base`] vector (the first layer's width padded to whole groups
    /// of [`GROUP`]).
    pub fn base_len(&self) -> usize {
        self.dense0.np
    }

    /// The first layer's bias plus its dense part for these dense inputs:
    /// `out[j] = b_j + Σ_d w_(N_BINARY + d) j · dense_d` for each output `j` (padding: 0).
    ///
    /// # Panics
    /// If `out` is shorter than [`FastMlp::base_len`].
    pub fn base(&self, dense: &[i32; N_DENSE], s: &mut Scratch, out: &mut [i64]) {
        let mut k = 0;
        for (d, &x) in dense.iter().enumerate() {
            s.idx[k] = d as u16;
            s.val[k] = x;
            k += usize::from(x != 0);
        }
        let p = &self.dense0;
        mac_pairs_at(s.level(), &p.wp, p.np, &s.idx[..k], &s.val[..k], &p.b, out);
    }

    /// The logits of an encoded position: exactly [`IntMlp::forward`].
    pub fn forward(&self, f: &Features, s: &mut Scratch) -> [i64; MAX_OUT] {
        let mut base = core::mem::take(&mut s.base);
        base.resize(self.base_len(), 0);
        self.base(&f.dense, s, &mut base);
        let z = self.forward_base(f.active(), &base, s);
        s.base = base;
        z
    }

    /// The logits of a position given its active binary inputs and its [`FastMlp::base`]
    /// vector: exactly [`IntMlp::forward`] of the features with those binary inputs and the
    /// dense inputs `base` was made from.
    ///
    /// # Panics
    /// If there are more than [`MAX_ACTIVE`] active inputs, an index is not below
    /// [`N_FEATURES`], or `base` is shorter than the first layer.
    pub fn forward_base(&self, active: &[u16], base: &[i64], s: &mut Scratch) -> [i64; MAX_OUT] {
        assert!(
            active.len() <= MAX_ACTIVE,
            "at most MAX_ACTIVE active inputs"
        );
        assert!(
            active.iter().all(|&i| usize::from(i) < N_FEATURES),
            "an input index"
        );
        let lv = s.level();
        let l0 = &self.net.layers()[0];
        let n0 = l0.n_out;
        if self.bin_i32 {
            self.binary_sums(active, s, lv);
            activate0_at(lv, base, &s.bin, l0.shift, &mut s.h[..n0]);
        } else {
            mac_rows_at(
                lv,
                &l0.w,
                n0,
                active,
                &s.ones[..active.len()],
                base,
                &mut s.acc,
            );
            activate_at(lv, &s.acc, l0.shift, &mut s.h[..n0]);
        }
        let mut k = compact(&s.h[..n0], &mut s.idx, &mut s.val);
        let last = self.hidden.len() - 1;
        for (i, p) in self.hidden.iter().enumerate() {
            mac_pairs_at(lv, &p.wp, p.np, &s.idx[..k], &s.val[..k], &p.b, &mut s.acc);
            if i == last {
                return core::array::from_fn(|j| {
                    if j < p.n {
                        round_shift(s.acc[j], p.shift)
                    } else {
                        0
                    }
                });
            }
            activate_at(lv, &s.acc, p.shift, &mut s.h[..p.n]);
            k = compact(&s.h[..p.n], &mut s.idx, &mut s.val);
        }
        unreachable!("the last layer returns")
    }
    /// `s.bin[j] = Σ_{i ∈ active} w_ij` (wrapping `i32`, exact by the bound), updated from the
    /// previous call's sums by the rows that switched on and off when that is less work. The
    /// previous call's inputs are kept as a bit set; an input list that is not strictly
    /// ascending (never one the encoder makes) is summed afresh and not kept.
    fn binary_sums(&self, active: &[u16], s: &mut Scratch, lv: Level) {
        let l0 = &self.net.layers()[0];
        let n0 = l0.n_out;
        let sorted = active.windows(2).all(|p| p[0] < p[1]);
        let mut cur = [0u64; WORDS];
        for &i in active {
            cur[usize::from(i) / 64] |= 1 << (i % 64);
        }
        // Unowned while it is being changed: if anything were to stop this call half-way, the
        // next one starts afresh.
        let owned = s.owner == self.id;
        s.owner = 0;
        let mut fresh = true;
        if owned && sorted {
            s.add.clear();
            s.sub.clear();
            for (w, (&c, &p)) in cur.iter().zip(&s.prev_bits).enumerate() {
                let mut on = c & !p;
                while on != 0 {
                    s.add.push((64 * w) as u16 + on.trailing_zeros() as u16);
                    on &= on - 1;
                }
                let mut off = p & !c;
                while off != 0 {
                    s.sub.push((64 * w) as u16 + off.trailing_zeros() as u16);
                    off &= off - 1;
                }
            }
            if s.add.len() + s.sub.len() < active.len() {
                sum_rows_at(lv, &l0.w, n0, &s.add, &s.sub, &mut s.bin);
                fresh = false;
            }
        }
        if fresh {
            s.bin.clear();
            s.bin.resize(n0, 0);
            sum_rows_at(lv, &l0.w, n0, active, &[], &mut s.bin);
        }
        // An unsorted list (or one with repeats) cannot be diffed: start afresh next time.
        if sorted {
            s.prev_bits = cur;
            s.owner = self.id;
        }
    }
}
