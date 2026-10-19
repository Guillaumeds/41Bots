//! The inner loops of the fast forward pass ([`crate::fast`]), written as plain safe Rust that the
//! compiler turns into SIMD instructions, and compiled several times by `multiversion` (PLAN §11:
//! runtime dispatch for hot kernels) so one binary uses the best instructions the machine has:
//! x86-64-v4 (AVX-512), v3 (AVX2), v2 (SSE4.1), or the plain build ([`Level`]). Every version
//! computes the same integers (integer addition and multiplication are exact, and the order of
//! the terms of an integer sum does not change it), so the choice never changes an answer; the
//! tests run every version the machine has against plain loops.
//!
//! - [`sum_rows_at`]: `acc[j] += Σ_{i ∈ add} w[i][j] − Σ_{i ∈ sub} w[i][j]` in **wrapping** `i32`
//!   arithmetic (addition mod 2^32). The caller proves the true result fits `i32`, so the
//!   wrapped result equals it whatever happens on the way.
//! - [`mac_pairs_at`] (and [`mac_rows_at`] on the plain layout): `out[j] = init[j] + Σ_k
//!   w[idx_k][j] · val_k` in `i64`, each product of an `i32` weight and an `i32` value (exact in
//!   `i64`: at most `2^31 · 2^31`); the caller proves no partial sum leaves `i64` (the network's
//!   `2^62` bound covers every partial sum: each is at most the sum of the absolute values of all
//!   its terms).
//! - [`activate_at`], [`activate0_at`]: the rounding shift and the clipped ReLU; [`compact`]: the
//!   non-zero activations as (index, value) lists, so the next layer skips the zeros.
//!
//! **Layouts and loop orders** (chosen by measurement on this project's build VM, an AVX-512
//! Xeon; docs/m3-hybrid/nn-speed.md). Weights are input-major (`w[i * n + j]`). The packed-pair
//! layout ([`pack_pairs`]) stores two outputs' weights in each 64-bit word, so one 32 × 32 →
//! 64-bit signed multiply (`pmuldq`) takes the low half directly and the high half after a
//! shift, with no unpacking. At v4 the outputs are taken in blocks of 64 held in registers while
//! the rows of the non-zero inputs stream past; at v3 and below, and at v4 for a matrix larger
//! than 1 MiB (it does not stay in the L2 cache), each group of four rows is streamed through
//! accumulators kept in the L1 cache (the register-blocked loop is slower there, and the compiler
//! turns the streaming loop into gathers when it may use AVX-512, so v4 borrows the v3 build).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use crate::encode::INPUT_SHIFT;
use crate::mlp::CLIP;

/// An instruction-set level of x86-64 (the plain build elsewhere).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// No SIMD beyond the target's baseline (on x86-64: SSE2).
    Base = 0,
    /// x86-64-v2 (SSE4.1: 32 × 32 → 64-bit signed multiplies).
    V2 = 1,
    /// x86-64-v3 (AVX2).
    V3 = 2,
    /// x86-64-v4 (AVX-512 F, BW, CD, DQ, VL).
    V4 = 3,
}

impl Level {
    /// Every level, lowest first.
    pub const ALL: [Level; 4] = [Level::Base, Level::V2, Level::V3, Level::V4];

    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Level::Base => "base",
            Level::V2 => "x86-64-v2",
            Level::V3 => "x86-64-v3",
            Level::V4 => "x86-64-v4",
        }
    }

    fn from_u8(x: u8) -> Level {
        Level::ALL[usize::from(x.min(3))]
    }
}

/// The best level this machine supports (detected once).
pub fn detected() -> Level {
    static D: OnceLock<Level> = OnceLock::new();
    *D.get_or_init(detect)
}

fn detect() -> Level {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as has;
        let v2 = has!("sse4.2") && has!("ssse3") && has!("popcnt");
        let v3 = v2
            && has!("avx2")
            && has!("avx")
            && has!("fma")
            && has!("bmi1")
            && has!("bmi2")
            && has!("lzcnt")
            && has!("movbe");
        let v4 = v3
            && has!("avx512f")
            && has!("avx512bw")
            && has!("avx512cd")
            && has!("avx512dq")
            && has!("avx512vl");
        if v4 {
            return Level::V4;
        }
        if v3 {
            return Level::V3;
        }
        if v2 {
            return Level::V2;
        }
    }
    Level::Base
}

/// No cap set.
const UNSET: u8 = u8::MAX;
static CAP: AtomicU8 = AtomicU8::new(UNSET);

/// The level in use: the detected one, or lower if [`set_cap`] asked for it.
pub fn level() -> Level {
    let c = CAP.load(Ordering::Relaxed);
    let d = detected();
    if c == UNSET {
        d
    } else {
        d.min(Level::from_u8(c))
    }
}

/// Use at most `cap` (benchmarks; `None` = the best the machine has). Results never depend on
/// it.
pub fn set_cap(cap: Option<Level>) {
    CAP.store(cap.map_or(UNSET, |l| l as u8), Ordering::Relaxed);
}

#[inline(always)]
fn sum_rows_body<const B: usize>(w: &[i32], n: usize, add: &[u16], sub: &[u16], acc: &mut [i32]) {
    let acc = &mut acc[..n];
    let mut j0 = 0;
    // Blocks of B, then of 8, then single outputs.
    while j0 + B <= n {
        sum_block::<B>(w, n, add, sub, j0, acc);
        j0 += B;
    }
    while j0 + 8 <= n {
        sum_block::<8>(w, n, add, sub, j0, acc);
        j0 += 8;
    }
    while j0 < n {
        sum_block::<1>(w, n, add, sub, j0, acc);
        j0 += 1;
    }
}

#[inline(always)]
fn sum_block<const B: usize>(
    w: &[i32],
    n: usize,
    add: &[u16],
    sub: &[u16],
    j0: usize,
    acc: &mut [i32],
) {
    let blk: &mut [i32; B] = (&mut acc[j0..j0 + B]).try_into().expect("a full block");
    let mut a = *blk;
    for &i in add {
        let o = usize::from(i) * n + j0;
        let row: &[i32; B] = w[o..o + B].try_into().expect("a full block");
        for t in 0..B {
            a[t] = a[t].wrapping_add(row[t]);
        }
    }
    for &i in sub {
        let o = usize::from(i) * n + j0;
        let row: &[i32; B] = w[o..o + B].try_into().expect("a full block");
        for t in 0..B {
            a[t] = a[t].wrapping_sub(row[t]);
        }
    }
    *blk = a;
}

#[inline(always)]
fn sum_rows64_body<const B: usize>(w: &[i32], n: usize, add: &[u16], sub: &[u16], acc: &mut [i64]) {
    let acc = &mut acc[..n];
    let mut j0 = 0;
    // Blocks of B, then of 8, then single outputs.
    while j0 + B <= n {
        sum_block64::<B>(w, n, add, sub, j0, acc);
        j0 += B;
    }
    while j0 + 8 <= n {
        sum_block64::<8>(w, n, add, sub, j0, acc);
        j0 += 8;
    }
    while j0 < n {
        sum_block64::<1>(w, n, add, sub, j0, acc);
        j0 += 1;
    }
}

#[inline(always)]
fn sum_block64<const B: usize>(
    w: &[i32],
    n: usize,
    add: &[u16],
    sub: &[u16],
    j0: usize,
    acc: &mut [i64],
) {
    let blk: &mut [i64; B] = (&mut acc[j0..j0 + B]).try_into().expect("a full block");
    let mut a = *blk;
    for &i in add {
        let o = usize::from(i) * n + j0;
        let row: &[i32; B] = w[o..o + B].try_into().expect("a full block");
        for t in 0..B {
            a[t] += i64::from(row[t]);
        }
    }
    for &i in sub {
        let o = usize::from(i) * n + j0;
        let row: &[i32; B] = w[o..o + B].try_into().expect("a full block");
        for t in 0..B {
            a[t] -= i64::from(row[t]);
        }
    }
    *blk = a;
}

#[inline(always)]
fn mac_rows_body<const B: usize>(
    w: &[i32],
    n: usize,
    idx: &[u16],
    val: &[i32],
    init: &[i64],
    out: &mut [i64],
) {
    let out = &mut out[..n];
    let init = &init[..n];
    assert_eq!(idx.len(), val.len(), "an index per value");
    let mut j0 = 0;
    // Blocks of B, then of 8, then single outputs.
    while j0 + B <= n {
        mac_block::<B>(w, n, idx, val, j0, init, out);
        j0 += B;
    }
    while j0 + 8 <= n {
        mac_block::<8>(w, n, idx, val, j0, init, out);
        j0 += 8;
    }
    while j0 < n {
        mac_block::<1>(w, n, idx, val, j0, init, out);
        j0 += 1;
    }
}

#[inline(always)]
fn mac_block<const B: usize>(
    w: &[i32],
    n: usize,
    idx: &[u16],
    val: &[i32],
    j0: usize,
    init: &[i64],
    out: &mut [i64],
) {
    let mut a: [i64; B] = init[j0..j0 + B].try_into().expect("a full block");
    for (&i, &x) in idx.iter().zip(val) {
        let o = usize::from(i) * n + j0;
        let row: &[i32; B] = w[o..o + B].try_into().expect("a full block");
        let x = i64::from(x);
        for t in 0..B {
            a[t] += i64::from(row[t]) * x;
        }
    }
    out[j0..j0 + B].copy_from_slice(&a);
}

/// Outputs per group of the packed-pair layout ([`pack_pairs`]).
pub const GROUP: usize = 16;

/// A matrix (`w[i * n + j]`, `n_in` rows) in the packed-pair layout that [`mac_pairs_at`] reads:
/// the outputs padded with zero weights to `np`, a multiple of [`GROUP`]; each row holds `np / 2`
/// words, and word `8g + t` of a row (`t < 8`) packs the weights of outputs `16g + t` (low half)
/// and `16g + 8 + t` (high half), each as its 32-bit two's complement. Returns the words and `np`.
pub fn pack_pairs(w: &[i32], n_in: usize, n: usize) -> (Vec<u64>, usize) {
    assert_eq!(w.len(), n_in * n, "a matrix of n_in × n");
    let np = n.div_ceil(GROUP) * GROUP;
    let get = |i: usize, j: usize| if j < n { w[i * n + j] } else { 0 };
    let mut out = Vec::with_capacity(n_in * np / 2);
    for i in 0..n_in {
        for g in 0..np / GROUP {
            for t in 0..8 {
                let lo = get(i, GROUP * g + t) as u32;
                let hi = get(i, GROUP * g + 8 + t) as u32;
                out.push(u64::from(lo) | (u64::from(hi) << 32));
            }
        }
    }
    (out, np)
}

/// The two weights of a packed word, as `i64`.
#[inline(always)]
fn halves(v: u64) -> (i64, i64) {
    (
        i64::from(v as u32 as i32),
        i64::from((v >> 32) as u32 as i32),
    )
}

/// Register-blocked: blocks of `2W` outputs held in registers while the listed rows stream past.
#[inline(always)]
fn mac_pairs_blocked<const W: usize>(
    wp: &[u64],
    np: usize,
    idx: &[u16],
    val: &[i32],
    init: &[i64],
    out: &mut [i64],
) {
    assert!(np.is_multiple_of(GROUP), "outputs padded to whole groups");
    assert_eq!(idx.len(), val.len(), "an index per value");
    let out = &mut out[..np];
    let init = &init[..np];
    let wpr = np / 2;
    let mut j0 = 0;
    // Blocks of W words (2W outputs), then single groups.
    while j0 + 2 * W <= np {
        mac_pair_block::<W>(wp, wpr, idx, val, j0, init, out);
        j0 += 2 * W;
    }
    while j0 < np {
        mac_pair_block::<8>(wp, wpr, idx, val, j0, init, out);
        j0 += GROUP;
    }
}

#[inline(always)]
fn mac_pair_block<const W: usize>(
    wp: &[u64],
    wpr: usize,
    idx: &[u16],
    val: &[i32],
    j0: usize,
    init: &[i64],
    out: &mut [i64],
) {
    let mut lo = [0i64; W];
    let mut hi = [0i64; W];
    for w in 0..W {
        let j = j0 + GROUP * (w / 8) + w % 8;
        lo[w] = init[j];
        hi[w] = init[j + 8];
    }
    for (&i, &x) in idx.iter().zip(val) {
        let o = usize::from(i) * wpr + j0 / 2;
        let row: &[u64; W] = wp[o..o + W].try_into().expect("a full block");
        let x = i64::from(x);
        for w in 0..W {
            let (a, b) = halves(row[w]);
            lo[w] += a * x;
            hi[w] += b * x;
        }
    }
    for w in 0..W {
        let j = j0 + GROUP * (w / 8) + w % 8;
        out[j] = lo[w];
        out[j + 8] = hi[w];
    }
}

/// Row-streaming: `R` listed rows at a time pass once over every group of outputs, whose sums
/// stay in `out` (in the L1 cache).
#[inline(always)]
fn mac_pairs_streamed<const R: usize>(
    wp: &[u64],
    np: usize,
    idx: &[u16],
    val: &[i32],
    init: &[i64],
    out: &mut [i64],
) {
    assert!(np.is_multiple_of(GROUP), "outputs padded to whole groups");
    assert_eq!(idx.len(), val.len(), "an index per value");
    let out = &mut out[..np];
    out.copy_from_slice(&init[..np]);
    let wpr = np / 2;
    let (mut k, n) = (0, idx.len());
    while k + R <= n {
        let rows: [&[u64]; R] = core::array::from_fn(|r| {
            let o = usize::from(idx[k + r]) * wpr;
            &wp[o..o + wpr]
        });
        let xs: [i64; R] = core::array::from_fn(|r| i64::from(val[k + r]));
        stream_group::<R>(rows, xs, out);
        k += R;
    }
    while k < n {
        let o = usize::from(idx[k]) * wpr;
        stream_group::<1>([&wp[o..o + wpr]], [i64::from(val[k])], out);
        k += 1;
    }
}

#[inline(always)]
fn stream_group<const R: usize>(rows: [&[u64]; R], xs: [i64; R], out: &mut [i64]) {
    for (g, a) in out.as_chunks_mut::<GROUP>().0.iter_mut().enumerate() {
        let (lo, hi) = a.split_at_mut(8);
        let lo: &mut [i64; 8] = lo.try_into().expect("8 outputs");
        let hi: &mut [i64; 8] = hi.try_into().expect("8 outputs");
        let (mut l, mut h) = (*lo, *hi);
        for r in 0..R {
            let row: &[u64; 8] = rows[r][8 * g..8 * g + 8].try_into().expect("8 words");
            for t in 0..8 {
                let (a, b) = halves(row[t]);
                l[t] += a * xs[r];
                h[t] += b * xs[r];
            }
        }
        *lo = l;
        *hi = h;
    }
}

/// The most positions one batched call takes ([`mac_pairs_batch_at`]).
pub const MAX_BATCH: usize = 256;

/// One position of a batch: its non-zero inputs (indices strictly ascending, as [`compact`]
/// gives them) and their values.
pub type BatchRows<'a> = (&'a [u16], &'a [i32]);

/// Batched, block-outer: for each block of `2W` outputs, every position in turn accumulates its
/// listed rows in registers ([`mac_pair_block`]), so the block's column strip of the matrix is
/// read from memory once per batch and from the cache for the other positions.
#[inline(always)]
fn mac_pairs_batch_blocked<const W: usize>(
    wp: &[u64],
    np: usize,
    rows: &[BatchRows],
    init: &[i64],
    out: &mut [i64],
) {
    assert!(np.is_multiple_of(GROUP), "outputs padded to whole groups");
    assert!(out.len() >= rows.len() * np, "an output row per position");
    let init = &init[..np];
    let wpr = np / 2;
    let mut j0 = 0;
    while j0 + 2 * W <= np {
        for (b, &(idx, val)) in rows.iter().enumerate() {
            assert_eq!(idx.len(), val.len(), "an index per value");
            mac_pair_block::<W>(wp, wpr, idx, val, j0, init, &mut out[b * np..(b + 1) * np]);
        }
        j0 += 2 * W;
    }
    while j0 < np {
        for (b, &(idx, val)) in rows.iter().enumerate() {
            assert_eq!(idx.len(), val.len(), "an index per value");
            mac_pair_block::<8>(wp, wpr, idx, val, j0, init, &mut out[b * np..(b + 1) * np]);
        }
        j0 += GROUP;
    }
}

/// Batched, row-chunk-outer: the matrix is taken `chunk` rows at a time, and every position
/// streams its listed rows of the chunk through its sums ([`stream_group`], `R` rows a pass), so
/// each chunk is read from memory once per batch and from the cache for the other positions.
#[inline(always)]
fn mac_pairs_batch_streamed<const R: usize>(
    wp: &[u64],
    np: usize,
    rows: &[BatchRows],
    init: &[i64],
    out: &mut [i64],
    chunk: usize,
) {
    assert!(np.is_multiple_of(GROUP), "outputs padded to whole groups");
    assert!(out.len() >= rows.len() * np, "an output row per position");
    assert!(rows.len() <= MAX_BATCH && chunk > 0, "a batch and a chunk");
    let wpr = np / 2;
    let n_in = wp.len() / wpr.max(1);
    for b in 0..rows.len() {
        assert_eq!(rows[b].0.len(), rows[b].1.len(), "an index per value");
        out[b * np..(b + 1) * np].copy_from_slice(&init[..np]);
    }
    let mut at = [0usize; MAX_BATCH];
    let mut r0 = 0;
    while r0 < n_in {
        let r1 = r0 + chunk;
        for (b, &(idx, val)) in rows.iter().enumerate() {
            let start = at[b];
            let mut end = start;
            while end < idx.len() && usize::from(idx[end]) < r1 {
                end += 1;
            }
            at[b] = end;
            let (idx, val) = (&idx[start..end], &val[start..end]);
            let o = &mut out[b * np..(b + 1) * np];
            let mut k = 0;
            while k + R <= idx.len() {
                let rs: [&[u64]; R] = core::array::from_fn(|r| {
                    let s = usize::from(idx[k + r]) * wpr;
                    &wp[s..s + wpr]
                });
                let xs: [i64; R] = core::array::from_fn(|r| i64::from(val[k + r]));
                stream_group::<R>(rs, xs, o);
                k += R;
            }
            while k < idx.len() {
                let s = usize::from(idx[k]) * wpr;
                stream_group::<1>([&wp[s..s + wpr]], [i64::from(val[k])], o);
                k += 1;
            }
        }
        r0 = r1;
    }
    // Rows past the matrix's last (none for a well-formed list) would be out of range above.
    for (b, &(idx, _)) in rows.iter().enumerate() {
        assert_eq!(
            at[b],
            idx.len(),
            "every listed row is below the matrix's rows"
        );
    }
}

#[inline(always)]
fn activate_body(acc: &[i64], shift: u32, h: &mut [i32]) {
    let half = if shift == 0 { 0 } else { 1i64 << (shift - 1) };
    for (y, &a) in h.iter_mut().zip(acc) {
        *y = ((a + half) >> shift).clamp(0, CLIP) as i32;
    }
}

#[inline(always)]
fn activate0_body(base: &[i64], bin: &[i32], shift: u32, h: &mut [i32]) {
    let half = if shift == 0 { 0 } else { 1i64 << (shift - 1) };
    for ((y, &a), &b) in h.iter_mut().zip(base).zip(bin) {
        let a = a + (i64::from(b) << INPUT_SHIFT);
        *y = ((a + half) >> shift).clamp(0, CLIP) as i32;
    }
}

#[inline(always)]
fn activate0_64_body(base: &[i64], bin: &[i64], shift: u32, h: &mut [i32]) {
    let half = if shift == 0 { 0 } else { 1i64 << (shift - 1) };
    for ((y, &a), &b) in h.iter_mut().zip(base).zip(bin) {
        let a = a + (b << INPUT_SHIFT);
        *y = ((a + half) >> shift).clamp(0, CLIP) as i32;
    }
}

/// One function per level, each compiled by `multiversion` for that level's features only and
/// calling `$body::<$c>` (the dispatch below picks the level, so a level never runs on a machine
/// without it).
macro_rules! at_level {
    (v4, $name:ident, $body:ident $(::<$c:tt>)?, ($($arg:ident: $ty:ty),*)) => {
        #[multiversion::multiversion(targets(
            "x86_64+avx512f+avx512bw+avx512cd+avx512dq+avx512vl+avx2+avx+fma+bmi1+bmi2+lzcnt+movbe+popcnt+sse4.2+sse4.1+ssse3"
        ))]
        fn $name($($arg: $ty),*) {
            $body$(::<{ $c }>)?($($arg),*)
        }
    };
    (v3, $name:ident, $body:ident $(::<$c:tt>)?, ($($arg:ident: $ty:ty),*)) => {
        #[multiversion::multiversion(targets(
            "x86_64+avx2+avx+fma+bmi1+bmi2+lzcnt+movbe+popcnt+sse4.2+sse4.1+ssse3"
        ))]
        fn $name($($arg: $ty),*) {
            $body$(::<{ $c }>)?($($arg),*)
        }
    };
    (v2, $name:ident, $body:ident $(::<$c:tt>)?, ($($arg:ident: $ty:ty),*)) => {
        #[multiversion::multiversion(targets("x86_64+popcnt+sse4.2+sse4.1+ssse3"))]
        fn $name($($arg: $ty),*) {
            $body$(::<{ $c }>)?($($arg),*)
        }
    };
}

at_level!(v4, sum_rows_v4, sum_rows_body::<128>,
    (w: &[i32], n: usize, add: &[u16], sub: &[u16], acc: &mut [i32]));
at_level!(v3, sum_rows_v3, sum_rows_body::<64>,
    (w: &[i32], n: usize, add: &[u16], sub: &[u16], acc: &mut [i32]));
at_level!(v2, sum_rows_v2, sum_rows_body::<32>,
    (w: &[i32], n: usize, add: &[u16], sub: &[u16], acc: &mut [i32]));
at_level!(v4, sum_rows64_v4, sum_rows64_body::<64>,
    (w: &[i32], n: usize, add: &[u16], sub: &[u16], acc: &mut [i64]));
at_level!(v3, sum_rows64_v3, sum_rows64_body::<32>,
    (w: &[i32], n: usize, add: &[u16], sub: &[u16], acc: &mut [i64]));
at_level!(v2, sum_rows64_v2, sum_rows64_body::<16>,
    (w: &[i32], n: usize, add: &[u16], sub: &[u16], acc: &mut [i64]));
at_level!(v4, activate0_64_v4, activate0_64_body,
    (base: &[i64], bin: &[i64], shift: u32, h: &mut [i32]));
at_level!(v3, activate0_64_v3, activate0_64_body,
    (base: &[i64], bin: &[i64], shift: u32, h: &mut [i32]));
at_level!(v2, activate0_64_v2, activate0_64_body,
    (base: &[i64], bin: &[i64], shift: u32, h: &mut [i32]));
at_level!(v4, mac_rows_v4, mac_rows_body::<64>,
    (w: &[i32], n: usize, idx: &[u16], val: &[i32], init: &[i64], out: &mut [i64]));
at_level!(v3, mac_rows_v3, mac_rows_body::<32>,
    (w: &[i32], n: usize, idx: &[u16], val: &[i32], init: &[i64], out: &mut [i64]));
at_level!(v2, mac_rows_v2, mac_rows_body::<16>,
    (w: &[i32], n: usize, idx: &[u16], val: &[i32], init: &[i64], out: &mut [i64]));
at_level!(v4, mac_pairs_v4, mac_pairs_blocked::<32>,
    (wp: &[u64], np: usize, idx: &[u16], val: &[i32], init: &[i64], out: &mut [i64]));
at_level!(v3, mac_pairs_v3, mac_pairs_streamed::<4>,
    (wp: &[u64], np: usize, idx: &[u16], val: &[i32], init: &[i64], out: &mut [i64]));
at_level!(v2, mac_pairs_v2, mac_pairs_streamed::<4>,
    (wp: &[u64], np: usize, idx: &[u16], val: &[i32], init: &[i64], out: &mut [i64]));
at_level!(v4, mac_pairs_batch_blocked_v4, mac_pairs_batch_blocked::<32>,
    (wp: &[u64], np: usize, rows: &[BatchRows<'_>], init: &[i64], out: &mut [i64]));
at_level!(v3, mac_pairs_batch_blocked_v3, mac_pairs_batch_blocked::<16>,
    (wp: &[u64], np: usize, rows: &[BatchRows<'_>], init: &[i64], out: &mut [i64]));
at_level!(v2, mac_pairs_batch_blocked_v2, mac_pairs_batch_blocked::<8>,
    (wp: &[u64], np: usize, rows: &[BatchRows<'_>], init: &[i64], out: &mut [i64]));
at_level!(v3, mac_pairs_batch_streamed_v3, mac_pairs_batch_streamed::<4>,
    (wp: &[u64], np: usize, rows: &[BatchRows<'_>], init: &[i64], out: &mut [i64], chunk: usize));
at_level!(v2, mac_pairs_batch_streamed_v2, mac_pairs_batch_streamed::<4>,
    (wp: &[u64], np: usize, rows: &[BatchRows<'_>], init: &[i64], out: &mut [i64], chunk: usize));
at_level!(v4, activate_v4, activate_body, (acc: &[i64], shift: u32, h: &mut [i32]));
at_level!(v3, activate_v3, activate_body, (acc: &[i64], shift: u32, h: &mut [i32]));
at_level!(v2, activate_v2, activate_body, (acc: &[i64], shift: u32, h: &mut [i32]));
at_level!(v4, activate0_v4, activate0_body,
    (base: &[i64], bin: &[i32], shift: u32, h: &mut [i32]));
at_level!(v3, activate0_v3, activate0_body,
    (base: &[i64], bin: &[i32], shift: u32, h: &mut [i32]));
at_level!(v2, activate0_v2, activate0_body,
    (base: &[i64], bin: &[i32], shift: u32, h: &mut [i32]));

/// `acc[j] += Σ_{i ∈ add} w[i·n + j] − Σ_{i ∈ sub} w[i·n + j]` for `j < n`, wrapping `i32`
/// arithmetic, at level `l` (at most [`detected`]).
///
/// # Panics
/// If a row index is out of range or `acc` is shorter than `n`.
pub fn sum_rows_at(l: Level, w: &[i32], n: usize, add: &[u16], sub: &[u16], acc: &mut [i32]) {
    match l {
        Level::V4 => sum_rows_v4(w, n, add, sub, acc),
        Level::V3 => sum_rows_v3(w, n, add, sub, acc),
        Level::V2 => sum_rows_v2(w, n, add, sub, acc),
        Level::Base => sum_rows_body::<32>(w, n, add, sub, acc),
    }
}

/// `acc[j] += Σ_{i ∈ add} w[i·n + j] − Σ_{i ∈ sub} w[i·n + j]` for `j < n` in `i64`, at level
/// `l` (at most [`detected`]). Each term is an `i32`, so no partial sum of fewer than `2^32`
/// terms can leave `i64` from a start below `2^62` in size.
///
/// # Panics
/// If a row index is out of range or `acc` is shorter than `n`.
pub fn sum_rows64_at(l: Level, w: &[i32], n: usize, add: &[u16], sub: &[u16], acc: &mut [i64]) {
    match l {
        Level::V4 => sum_rows64_v4(w, n, add, sub, acc),
        Level::V3 => sum_rows64_v3(w, n, add, sub, acc),
        Level::V2 => sum_rows64_v2(w, n, add, sub, acc),
        Level::Base => sum_rows64_body::<16>(w, n, add, sub, acc),
    }
}

/// `out[j] = init[j] + Σ_k w[idx_k·n + j] · val_k` for `j < n` in `i64` (no partial sum may
/// leave `i64`: the caller's bound), on the plain input-major layout, at level `l` (at most
/// [`detected`]).
///
/// # Panics
/// If a row index is out of range, or `init` or `out` is shorter than `n`.
pub fn mac_rows_at(
    l: Level,
    w: &[i32],
    n: usize,
    idx: &[u16],
    val: &[i32],
    init: &[i64],
    out: &mut [i64],
) {
    match l {
        Level::V4 => mac_rows_v4(w, n, idx, val, init, out),
        Level::V3 => mac_rows_v3(w, n, idx, val, init, out),
        Level::V2 => mac_rows_v2(w, n, idx, val, init, out),
        Level::Base => mac_rows_body::<16>(w, n, idx, val, init, out),
    }
}

/// At v4, matrices of more words than this (1 MiB) use the streaming loop of v3.
const STREAM_WORDS: usize = 1 << 17;

/// `out[j] = init[j] + Σ_k w[idx_k][j] · val_k` for `j < np`, the matrix in the packed-pair
/// layout of [`pack_pairs`] (`np` its padded width), in `i64` (no partial sum may leave `i64`:
/// the caller's bound), at level `l` (at most [`detected`]).
///
/// # Panics
/// If a row index is out of range, `np` is not a multiple of [`GROUP`], or `init` or `out` is
/// shorter than `np`.
pub fn mac_pairs_at(
    l: Level,
    wp: &[u64],
    np: usize,
    idx: &[u16],
    val: &[i32],
    init: &[i64],
    out: &mut [i64],
) {
    match l {
        // A matrix larger than about an L2 cache streams better row by row (measured).
        Level::V4 if wp.len() > STREAM_WORDS => mac_pairs_v3(wp, np, idx, val, init, out),
        Level::V4 => mac_pairs_v4(wp, np, idx, val, init, out),
        Level::V3 => mac_pairs_v3(wp, np, idx, val, init, out),
        Level::V2 => mac_pairs_v2(wp, np, idx, val, init, out),
        Level::Base => mac_pairs_streamed::<4>(wp, np, idx, val, init, out),
    }
}

/// How [`mac_pairs_batch_with`] orders a batch's work (benchmarks; every choice gives the same
/// integers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchOrder {
    /// Blocks of outputs outermost, positions inside, sums in registers.
    Blocked,
    /// Chunks of this many bytes of the matrix outermost, positions inside, sums streamed.
    Streamed(usize),
    /// Each position in turn through [`mac_pairs_at`] (one at a time).
    Each,
}

/// The batch order [`mac_pairs_batch_at`] uses at level `l` for a matrix of `words` packed words
/// (chosen by measurement: docs/ENGINEERING-2026-10-17.md, "Item B, done").
pub fn batch_order(l: Level, words: usize) -> BatchOrder {
    match BATCH_ORDER.load(Ordering::Relaxed) {
        // At v4 a matrix that stays in the second-level cache goes through the register-blocked
        // loop of `mac_pairs_at`, position by position (faster there than any batch order
        // measured); every other one streams by chunks, read once per batch.
        0 if l == Level::V4 && words <= STREAM_WORDS => BatchOrder::Each,
        0 => BatchOrder::Streamed(BATCH_CHUNK_BYTES),
        1 => BatchOrder::Blocked,
        2 => BatchOrder::Each,
        x => BatchOrder::Streamed(x as usize),
    }
}

/// The batch order set by [`set_batch_order`]: 0 none, 1 blocked, 2 each, otherwise streamed
/// chunks of that many bytes.
static BATCH_ORDER: AtomicU64 = AtomicU64::new(0);

/// Use this batch order instead of [`batch_order`]'s choice (benchmarks; `None` = the choice).
/// Results never depend on it.
pub fn set_batch_order(o: Option<BatchOrder>) {
    let x = match o {
        None => 0,
        Some(BatchOrder::Blocked) => 1,
        Some(BatchOrder::Each) => 2,
        Some(BatchOrder::Streamed(b)) => (b as u64).max(3),
    };
    BATCH_ORDER.store(x, Ordering::Relaxed);
}

/// The chunk of the streamed batch order: well inside the 512 KiB second-level cache of the
/// rented EPYC machines, with room for the positions' sums.
const BATCH_CHUNK_BYTES: usize = 128 << 10;

/// `out[b·np + j] = init[j] + Σ_k w[idx_bk][j] · val_bk` for every position `b` of `rows` and
/// `j < np`: [`mac_pairs_at`] of each position (the same integers: each output is the same sum of
/// the same terms, and an integer sum does not depend on the order of its terms; no partial sum
/// leaves the caller's bound), with the matrix read once per batch rather than once per
/// position. At level `l` (at most [`detected`]).
///
/// # Panics
/// As [`mac_pairs_at`]; also if `rows` holds more than [`MAX_BATCH`] positions, a position's
/// indices are not ascending, or `out` is shorter than `rows.len() · np`.
pub fn mac_pairs_batch_at(
    l: Level,
    wp: &[u64],
    np: usize,
    rows: &[BatchRows<'_>],
    init: &[i64],
    out: &mut [i64],
) {
    mac_pairs_batch_with(l, batch_order(l, wp.len()), wp, np, rows, init, out);
}

/// [`mac_pairs_batch_at`] in the given order (benchmarks and tests).
///
/// # Panics
/// As [`mac_pairs_batch_at`].
pub fn mac_pairs_batch_with(
    l: Level,
    order: BatchOrder,
    wp: &[u64],
    np: usize,
    rows: &[BatchRows<'_>],
    init: &[i64],
    out: &mut [i64],
) {
    assert!(rows.len() <= MAX_BATCH, "at most MAX_BATCH positions");
    assert!(
        rows.iter()
            .all(|(idx, _)| idx.windows(2).all(|p| p[0] < p[1])),
        "each position's indices ascending"
    );
    match order {
        BatchOrder::Blocked => match l {
            Level::V4 => mac_pairs_batch_blocked_v4(wp, np, rows, init, out),
            Level::V3 => mac_pairs_batch_blocked_v3(wp, np, rows, init, out),
            Level::V2 => mac_pairs_batch_blocked_v2(wp, np, rows, init, out),
            Level::Base => mac_pairs_batch_blocked::<8>(wp, np, rows, init, out),
        },
        BatchOrder::Each => {
            for (b, &(idx, val)) in rows.iter().enumerate() {
                mac_pairs_at(l, wp, np, idx, val, init, &mut out[b * np..(b + 1) * np]);
            }
        }
        BatchOrder::Streamed(bytes) => {
            let chunk = (bytes / (4 * np.max(GROUP))).max(1);
            match l {
                // The streaming loop at v4 borrows the v3 build, as in `mac_pairs_at`.
                Level::V4 | Level::V3 => {
                    mac_pairs_batch_streamed_v3(wp, np, rows, init, out, chunk)
                }
                Level::V2 => mac_pairs_batch_streamed_v2(wp, np, rows, init, out, chunk),
                Level::Base => mac_pairs_batch_streamed::<4>(wp, np, rows, init, out, chunk),
            }
        }
    }
}

/// `h[j] = min(max(round_shift(acc[j], shift), 0), CLIP)` for `j < h.len()` (a hidden layer's
/// activation, [`crate::mlp::round_shift`] and the clipped ReLU), at level `l`.
///
/// # Panics
/// If `acc` is shorter than `h` or `shift ≥ 63`.
pub fn activate_at(l: Level, acc: &[i64], shift: u32, h: &mut [i32]) {
    assert!(
        acc.len() >= h.len() && shift < 63,
        "activation sizes and shift"
    );
    match l {
        Level::V4 => activate_v4(acc, shift, h),
        Level::V3 => activate_v3(acc, shift, h),
        Level::V2 => activate_v2(acc, shift, h),
        Level::Base => activate_body(acc, shift, h),
    }
}

/// The first layer's activation: [`activate_at`] of `base[j] + bin[j] · 2^24`.
///
/// # Panics
/// If `base` or `bin` is shorter than `h` or `shift ≥ 63`.
pub fn activate0_at(l: Level, base: &[i64], bin: &[i32], shift: u32, h: &mut [i32]) {
    assert!(
        base.len() >= h.len() && bin.len() >= h.len() && shift < 63,
        "activation sizes and shift"
    );
    match l {
        Level::V4 => activate0_v4(base, bin, shift, h),
        Level::V3 => activate0_v3(base, bin, shift, h),
        Level::V2 => activate0_v2(base, bin, shift, h),
        Level::Base => activate0_body(base, bin, shift, h),
    }
}

/// The first layer's activation from `i64` binary sums: [`activate_at`] of
/// `base[j] + bin[j] · 2^24` (the caller's bound keeps that sum within `±2^62`).
///
/// # Panics
/// If `base` or `bin` is shorter than `h` or `shift ≥ 63`.
pub fn activate0_64_at(l: Level, base: &[i64], bin: &[i64], shift: u32, h: &mut [i32]) {
    assert!(
        base.len() >= h.len() && bin.len() >= h.len() && shift < 63,
        "activation sizes and shift"
    );
    match l {
        Level::V4 => activate0_64_v4(base, bin, shift, h),
        Level::V3 => activate0_64_v3(base, bin, shift, h),
        Level::V2 => activate0_64_v2(base, bin, shift, h),
        Level::Base => activate0_64_body(base, bin, shift, h),
    }
}

/// The non-zero entries of `h`: their indices and values, in order, into `idx` and `val`
/// (branch-free); returns how many.
///
/// # Panics
/// If `idx` or `val` is shorter than `h`, or `h` is longer than 65,536.
pub fn compact(h: &[i32], idx: &mut [u16], val: &mut [i32]) -> usize {
    assert!(idx.len() >= h.len() && val.len() >= h.len() && h.len() <= 1 << 16);
    let mut k = 0;
    for (j, &y) in h.iter().enumerate() {
        idx[k] = j as u16;
        val[k] = y;
        k += usize::from(y != 0);
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::rng::SplitMix64;

    fn levels() -> Vec<Level> {
        Level::ALL
            .into_iter()
            .filter(|&l| l <= detected())
            .collect()
    }

    /// Weights from every scale, the limits included.
    fn weight(sm: &mut SplitMix64) -> i32 {
        match sm.next_u64() % 8 {
            0 => i32::MIN,
            1 => i32::MAX,
            2 => 0,
            3 => (sm.next_u64() % 7) as i32 - 3,
            _ => (sm.next_u64() >> (sm.next_u64() % 33)) as i32,
        }
    }

    #[test]
    fn sum_rows_wraps_like_the_plain_sum() {
        let mut sm = SplitMix64::new(1);
        for n in [1usize, 7, 8, 9, 31, 64, 65, 127, 128, 129, 200, 512, 1000] {
            let n_in = 50;
            let w: Vec<i32> = (0..n_in * n).map(|_| weight(&mut sm)).collect();
            for _ in 0..4 {
                let add: Vec<u16> = (0..sm.next_u64() % 40)
                    .map(|_| (sm.next_u64() % n_in as u64) as u16)
                    .collect();
                let sub: Vec<u16> = (0..sm.next_u64() % 40)
                    .map(|_| (sm.next_u64() % n_in as u64) as u16)
                    .collect();
                let start: Vec<i32> = (0..n).map(|_| weight(&mut sm)).collect();
                let mut want = start.clone();
                for j in 0..n {
                    for &i in &add {
                        want[j] = want[j].wrapping_add(w[usize::from(i) * n + j]);
                    }
                    for &i in &sub {
                        want[j] = want[j].wrapping_sub(w[usize::from(i) * n + j]);
                    }
                }
                for l in levels() {
                    let mut got = start.clone();
                    sum_rows_at(l, &w, n, &add, &sub, &mut got);
                    assert_eq!(got, want, "n {n} level {l:?}");
                }
            }
        }
    }

    #[test]
    fn mac_kernels_equal_the_plain_sums() {
        let mut sm = SplitMix64::new(2);
        for n in [1usize, 4, 8, 15, 16, 17, 33, 64, 100, 128, 256, 300] {
            let n_in = 70;
            // Weights of every scale; values up to 2^26 (activations) or −2^24 (dense inputs),
            // and few enough terms that no sum leaves i64.
            let w: Vec<i32> = (0..n_in * n).map(|_| weight(&mut sm)).collect();
            let (wp, np) = pack_pairs(&w, n_in, n);
            assert_eq!(np % GROUP, 0);
            for _ in 0..4 {
                let k = (sm.next_u64() % 40) as usize;
                let idx: Vec<u16> = (0..k)
                    .map(|_| (sm.next_u64() % n_in as u64) as u16)
                    .collect();
                let val: Vec<i32> = (0..k)
                    .map(|_| match sm.next_u64() % 4 {
                        0 => CLIP as i32,
                        1 => -(1 << 24),
                        _ => (sm.next_u64() % (CLIP as u64 + 1)) as i32,
                    })
                    .collect();
                let init: Vec<i64> = (0..np)
                    .map(|_| (sm.next_u64() >> 3) as i64 - (1 << 60))
                    .collect();
                let want: Vec<i64> = (0..n)
                    .map(|j| {
                        idx.iter().zip(&val).fold(init[j], |a, (&i, &x)| {
                            a + i64::from(w[usize::from(i) * n + j]) * i64::from(x)
                        })
                    })
                    .collect();
                for l in levels() {
                    let mut got = vec![7i64; np];
                    mac_rows_at(l, &w, n, &idx, &val, &init, &mut got);
                    assert_eq!(&got[..n], &want[..], "rows n {n} level {l:?}");
                    let mut got = vec![7i64; np];
                    mac_pairs_at(l, &wp, np, &idx, &val, &init, &mut got);
                    assert_eq!(&got[..n], &want[..], "pairs n {n} level {l:?}");
                    // Padding outputs carry their initial values (zero weights).
                    assert_eq!(&got[n..], &init[n..np]);
                }
            }
        }
    }

    /// Outputs and inputs 1024, 2048 and 4096 wide (the widest layers `MAX_WIDTH` allows, and
    /// row indices up to 4095): every kernel at every level equals the plain sums and rules.
    #[test]
    fn kernels_at_widths_up_to_4096() {
        let mut sm = SplitMix64::new(4);
        for (n_in, n) in [(1024usize, 1024usize), (4096, 64), (2048, 2048), (64, 4096)] {
            // Weights of a trained network's scale (2^24 / sqrt(fan-in)), and a few at the
            // extremes; values at and below the clip.
            let a = ((1u64 << 48) / n_in as u64).isqrt();
            let w: Vec<i32> = (0..n_in * n)
                .map(|_| match sm.next_u64() % 64 {
                    0 => i32::MIN,
                    1 => i32::MAX,
                    _ => ((sm.next_u64() % (2 * a + 1)) as i64 - a as i64) as i32,
                })
                .collect();
            let (wp, np) = pack_pairs(&w, n_in, n);
            let k = n_in.min(600);
            let mut idx: Vec<u16> = (0..k)
                .map(|_| (sm.next_u64() % n_in as u64) as u16)
                .collect();
            idx[0] = (n_in - 1) as u16;
            let val: Vec<i32> = (0..k)
                .map(|_| match sm.next_u64() % 3 {
                    0 => CLIP as i32,
                    _ => (sm.next_u64() % (CLIP as u64 + 1)) as i32,
                })
                .collect();
            let init: Vec<i64> = (0..np)
                .map(|_| (sm.next_u64() >> 24) as i64 - (1 << 39))
                .collect();
            let want: Vec<i64> = (0..n)
                .map(|j| {
                    idx.iter().zip(&val).fold(init[j], |acc, (&i, &x)| {
                        acc + i64::from(w[usize::from(i) * n + j]) * i64::from(x)
                    })
                })
                .collect();
            let mut want_h = vec![0i32; n];
            for (y, &a) in want_h.iter_mut().zip(&want) {
                *y = crate::mlp::round_shift(a, 24).clamp(0, CLIP) as i32;
            }
            let add: Vec<u16> = idx[..k.min(96)].to_vec();
            let sub: Vec<u16> = idx[k.min(96) / 2..k.min(96)].to_vec();
            let mut want_sum = vec![0i32; n];
            for (j, s) in want_sum.iter_mut().enumerate() {
                for &i in &add {
                    *s = s.wrapping_add(w[usize::from(i) * n + j]);
                }
                for &i in &sub {
                    *s = s.wrapping_sub(w[usize::from(i) * n + j]);
                }
            }
            for l in levels() {
                let mut got = vec![7i64; np];
                mac_rows_at(l, &w, n, &idx, &val, &init, &mut got);
                assert_eq!(&got[..n], &want[..], "rows {n_in}x{n} level {l:?}");
                let mut got = vec![7i64; np];
                mac_pairs_at(l, &wp, np, &idx, &val, &init, &mut got);
                assert_eq!(&got[..n], &want[..], "pairs {n_in}x{n} level {l:?}");
                let mut h = vec![-1; n];
                activate_at(l, &got, 24, &mut h);
                assert_eq!(h, want_h, "activation {n} level {l:?}");
                let mut s = vec![0i32; n];
                sum_rows_at(l, &w, n, &add, &sub, &mut s);
                assert_eq!(s, want_sum, "binary sums {n_in}x{n} level {l:?}");
            }
            let mut ci = vec![0u16; n];
            let mut cv = vec![0i32; n];
            let c = compact(&want_h, &mut ci, &mut cv);
            let nz: Vec<(u16, i32)> = want_h
                .iter()
                .enumerate()
                .filter(|(_, y)| **y != 0)
                .map(|(j, &y)| (j as u16, y))
                .collect();
            assert_eq!(c, nz.len());
            assert!(ci[..c].iter().zip(&cv[..c]).map(|(&i, &v)| (i, v)).eq(nz));
        }
    }

    /// The batched product at every level and in every batch order equals the plain sums of
    /// each position: widths 1 to 4096 (row indices up to 4095), batches of 1 to [`MAX_BATCH`]
    /// positions (empty row lists, repeated positions, the matrix's last row), weights of every
    /// scale and values at the clip and at `−2^24`.
    #[test]
    fn batched_products_equal_the_plain_sums() {
        let mut sm = SplitMix64::new(5);
        let orders = [
            BatchOrder::Each,
            BatchOrder::Blocked,
            BatchOrder::Streamed(1),
            BatchOrder::Streamed(4096),
            BatchOrder::Streamed(BATCH_CHUNK_BYTES),
            BatchOrder::Streamed(1 << 30),
        ];
        for (n_in, n, extreme) in [
            (70usize, 17usize, true),
            (64, 100, true),
            (300, 256, true),
            (1024, 512, false),
            (2048, 1024, false),
            (4096, 64, false),
            (64, 4096, false),
        ] {
            let a = ((1u64 << 48) / n_in as u64).isqrt();
            let w: Vec<i32> = (0..n_in * n)
                .map(|_| {
                    if extreme {
                        weight(&mut sm)
                    } else {
                        match sm.next_u64() % 64 {
                            0 => i32::MIN,
                            1 => i32::MAX,
                            _ => ((sm.next_u64() % (2 * a + 1)) as i64 - a as i64) as i32,
                        }
                    }
                })
                .collect();
            let (wp, np) = pack_pairs(&w, n_in, n);
            let init: Vec<i64> = (0..np)
                .map(|_| (sm.next_u64() >> 24) as i64 - (1 << 39))
                .collect();
            for nb in [1usize, 2, 3, 7, MAX_BATCH] {
                // Each position: a sorted subset of the rows (few terms when the weights are
                // extreme, so no sum leaves i64), sometimes empty, sometimes the last row.
                let cap = if extreme { 40 } else { n_in.min(700) };
                let lists: Vec<(Vec<u16>, Vec<i32>)> = (0..nb)
                    .map(|b| {
                        if b % 5 == 4 {
                            return (Vec::new(), Vec::new());
                        }
                        let k = (sm.next_u64() % (cap as u64 + 1)) as usize;
                        let mut idx: Vec<u16> = (0..k)
                            .map(|_| (sm.next_u64() % n_in as u64) as u16)
                            .collect();
                        if b % 3 == 0 {
                            idx.push((n_in - 1) as u16);
                        }
                        idx.sort_unstable();
                        idx.dedup();
                        let val: Vec<i32> = idx
                            .iter()
                            .map(|_| match sm.next_u64() % 4 {
                                0 => CLIP as i32,
                                1 => -(1 << 24),
                                _ => (sm.next_u64() % (CLIP as u64 + 1)) as i32,
                            })
                            .collect();
                        (idx, val)
                    })
                    .collect();
                let rows: Vec<BatchRows<'_>> = lists
                    .iter()
                    .map(|(i, v)| (i.as_slice(), v.as_slice()))
                    .collect();
                let mut want: Vec<i64> = Vec::with_capacity(nb * np);
                for (idx, val) in &lists {
                    for j in 0..np {
                        want.push(idx.iter().zip(val).fold(init[j], |acc, (&i, &x)| {
                            acc + if j < n {
                                i64::from(w[usize::from(i) * n + j]) * i64::from(x)
                            } else {
                                0
                            }
                        }));
                    }
                }
                for l in levels() {
                    for o in orders {
                        let mut got = vec![7i64; nb * np];
                        mac_pairs_batch_with(l, o, &wp, np, &rows, &init, &mut got);
                        assert_eq!(got, want, "{n_in}x{n} batch {nb} level {l:?} {o:?}");
                    }
                    let mut got = vec![7i64; nb * np];
                    mac_pairs_batch_at(l, &wp, np, &rows, &init, &mut got);
                    assert_eq!(got, want, "{n_in}x{n} batch {nb} level {l:?}");
                    // One position at a time gives the same.
                    for (b, (idx, val)) in lists.iter().enumerate() {
                        let mut one = vec![7i64; np];
                        mac_pairs_at(l, &wp, np, idx, val, &init, &mut one);
                        assert_eq!(one, want[b * np..(b + 1) * np], "{n_in}x{n} one {b}");
                    }
                }
            }
        }
    }

    /// The `i64` binary sums and their activation equal the plain rule, from any start,
    /// with rows added and removed, weights at the limits included.
    #[test]
    fn wide_binary_sums_equal_the_plain_rule() {
        let mut sm = SplitMix64::new(6);
        for n in [1usize, 7, 8, 9, 31, 64, 65, 129, 1000, 2048] {
            let n_in = 120;
            let w: Vec<i32> = (0..n_in * n).map(|_| weight(&mut sm)).collect();
            for _ in 0..3 {
                let add: Vec<u16> = (0..sm.next_u64() % 97)
                    .map(|_| (sm.next_u64() % n_in as u64) as u16)
                    .collect();
                let sub: Vec<u16> = (0..sm.next_u64() % 97)
                    .map(|_| (sm.next_u64() % n_in as u64) as u16)
                    .collect();
                let start: Vec<i64> = (0..n).map(|_| i64::from(weight(&mut sm)) << 6).collect();
                let mut want = start.clone();
                for (j, s) in want.iter_mut().enumerate() {
                    for &i in &add {
                        *s += i64::from(w[usize::from(i) * n + j]);
                    }
                    for &i in &sub {
                        *s -= i64::from(w[usize::from(i) * n + j]);
                    }
                }
                let base: Vec<i64> = (0..n)
                    .map(|_| (sm.next_u64() >> 2) as i64 - (1 << 61))
                    .collect();
                for shift in [0u32, 20, 24, 28] {
                    let want_h: Vec<i32> = base
                        .iter()
                        .zip(&want)
                        .map(|(&a, &b)| {
                            crate::mlp::round_shift(a / 4 + (b << INPUT_SHIFT), shift)
                                .clamp(0, CLIP) as i32
                        })
                        .collect();
                    let quarter: Vec<i64> = base.iter().map(|&a| a / 4).collect();
                    for l in levels() {
                        let mut got = start.clone();
                        sum_rows64_at(l, &w, n, &add, &sub, &mut got);
                        assert_eq!(got, want, "n {n} level {l:?}");
                        let mut h = vec![-1; n];
                        activate0_64_at(l, &quarter, &got, shift, &mut h);
                        assert_eq!(h, want_h, "n {n} shift {shift} level {l:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn activation_and_compaction_equal_the_plain_rule() {
        let mut sm = SplitMix64::new(3);
        for n in [1usize, 5, 16, 63, 64, 65, 512, 1024] {
            for shift in [0u32, 1, 20, 24, 28, 62] {
                let acc: Vec<i64> = (0..n)
                    .map(|_| match sm.next_u64() % 5 {
                        0 => 0,
                        1 => (1 << 62) - (sm.next_u64() % 1000) as i64,
                        2 => -(1 << 62) + (sm.next_u64() % 1000) as i64,
                        _ => (sm.next_u64() >> (sm.next_u64() % 64)) as i64 >> 1,
                    })
                    .collect();
                let bin: Vec<i32> = (0..n).map(|_| weight(&mut sm)).collect();
                let base: Vec<i64> = acc.iter().map(|&a| a >> 2).collect();
                let want: Vec<i32> = acc
                    .iter()
                    .map(|&a| crate::mlp::round_shift(a, shift).clamp(0, CLIP) as i32)
                    .collect();
                let want0: Vec<i32> = base
                    .iter()
                    .zip(&bin)
                    .map(|(&a, &b)| {
                        crate::mlp::round_shift(a + (i64::from(b) << INPUT_SHIFT), shift)
                            .clamp(0, CLIP) as i32
                    })
                    .collect();
                for l in levels() {
                    let mut h = vec![-1; n];
                    activate_at(l, &acc, shift, &mut h);
                    assert_eq!(h, want, "n {n} shift {shift} level {l:?}");
                    activate0_at(l, &base, &bin, shift, &mut h);
                    assert_eq!(h, want0, "n {n} shift {shift} level {l:?}");
                }
                let mut idx = vec![0u16; n];
                let mut val = vec![0i32; n];
                let k = compact(&want, &mut idx, &mut val);
                let nz: Vec<(u16, i32)> = want
                    .iter()
                    .enumerate()
                    .filter(|(_, y)| **y != 0)
                    .map(|(j, &y)| (j as u16, y))
                    .collect();
                assert_eq!(k, nz.len());
                assert!(idx[..k].iter().zip(&val[..k]).map(|(&i, &v)| (i, v)).eq(nz));
            }
        }
    }
}
