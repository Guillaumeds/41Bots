//! The world space of a tracker (docs/M3-SPEC.md §4): classes of interchangeable cards, the exact
//! count of consistent worlds by dynamic programming over locations (`u128`, checked), max-flow
//! feasibility, exactly uniform sampling and enumeration.
//!
//! A **class** is a maximal set of unknown cards with identical masks of allowed locations. A
//! **class matrix** `M[c][l]` says how many cards of class `c` lie in location `l` (row sums =
//! class sizes, column sums = capacities, 0 where `l` is forbidden for `c`). The number of
//! consistent worlds is `N = Σ_M Π_c n_c! / Π_l M[c][l]!`, computed by a dynamic program over the
//! locations in `Loc` order whose state is the vector of cards of each class still unplaced.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use engine::cards::{Bits, Card, bit, cards, count};
use engine::rng::Rng;

use crate::tracker::{Loc, Tracker};

/// One complete placement of the hidden cards, consistent with a tracker. The derived order is the
/// canonical order of worlds.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct World {
    /// Every seat's hand (the viewer's is its real hand).
    pub hands: [Bits; 8],
    /// The other seats' hidden burns, sorted by location.
    pub burns: Vec<(Loc, Card)>,
    /// The undealt pack, as a set (during the dealer's burn after a turned Joker it includes the
    /// card turned next).
    pub undealt: Bits,
}

impl World {
    /// The burns as `(location, set)` pairs, for [`Tracker::consistent`].
    pub fn burns_by_loc(&self) -> Vec<(Loc, Bits)> {
        self.burns.iter().map(|&(l, c)| (l, bit(c))).collect()
    }

    /// During the dealer's burn after a turned Joker: the card turned next, uniform over the
    /// world's undealt cards (deck position 41 is a uniform position of the shuffled pack, §4.5).
    /// Draw it from `Rng::derive(dseed, &[labels::WORLD, w, labels::NEXT_TURNED])`.
    pub fn draw_next_turned(&self, rng: &mut Rng) -> Card {
        let n = u64::from(count(self.undealt));
        let i = rng.bounded(n) as usize;
        cards(self.undealt).nth(i).expect("i < n")
    }
}

/// The facts contradict each other (a bug: the true deal always satisfies them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Infeasible;

impl core::fmt::Display for Infeasible {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("no world satisfies the tracker's facts")
    }
}

impl std::error::Error for Infeasible {}

/// FNV-1a: a fixed hash for the memo, which is only looked up, never iterated (M3-SPEC §1).
#[derive(Default)]
struct Fnv(u64);

impl Hasher for Fnv {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        let mut h = if self.0 == 0 {
            0xcbf2_9ce4_8422_2325
        } else {
            self.0
        };
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.0 = h;
    }
}

/// Counting states: (location index, unplaced cards per class) packed into one `u128`.
type Memo = HashMap<u128, u128, BuildHasherDefault<Fnv>>;

/// At most this many classes (6 bits per class count, 16 classes, plus the location index).
pub const MAX_CLASSES: usize = 16;

fn key(l: usize, rem: &[u8]) -> u128 {
    let mut k = l as u128;
    for &r in rem {
        k = (k << 6) | u128::from(r);
    }
    k
}

/// Binomial coefficients up to 53 (exact in `u128`).
pub fn binom(n: u8, k: u8) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut r: u128 = 1;
    for i in 0..u128::from(k) {
        r = r * (u128::from(n) - i) / (i + 1);
    }
    r
}

/// `⌊256 · log2 n⌋` in integers (0 for `n` ≤ 1): the world count's size on the scale used by
/// teacher records and the §13.1 world-count bands.
pub fn log2_q8(n: u128) -> u16 {
    if n <= 1 {
        return 0;
    }
    let e = 127 - n.leading_zeros(); // ⌊log2 n⌋
    // The mantissa n / 2^e in [1, 2) as a Q62 fixed-point number, truncated.
    let mut x: u128 = if e >= 62 {
        n >> (e - 62)
    } else {
        n << (62 - e)
    };
    let mut frac = 0u32;
    for _ in 0..8 {
        // Square: x² / 2^62 stays below 2^64, so the product fits in u128.
        x = (x * x) >> 62;
        frac <<= 1;
        if x >= 1 << 63 {
            frac |= 1;
            x >>= 1;
        }
    }
    (256 * e + frac) as u16
}

/// Integer max-flow (Edmonds–Karp) on a dense matrix: can `rem[c]` cards of each class be placed
/// into locations `from..` with capacities `caps`, class `c` only where `allow[c]` has the bit?
pub fn feasible(rem: &[u8], allow: &[u32], caps: &[u8], from: usize) -> bool {
    let nc = rem.len();
    let nl = caps.len() - from;
    let need: u32 = rem.iter().map(|&r| u32::from(r)).sum();
    let have: u32 = caps[from..].iter().map(|&c| u32::from(c)).sum();
    if need != have {
        return false;
    }
    if need == 0 {
        return true;
    }
    // Nodes: 0 = source, 1..=nc classes, nc+1..=nc+nl locations, nc+nl+1 = sink.
    let n = nc + nl + 2;
    let sink = n - 1;
    let mut cap = vec![0i32; n * n];
    for (c, &r) in rem.iter().enumerate() {
        cap[c + 1] = i32::from(r);
        for l in 0..nl {
            if allow[c] & (1 << (from + l)) != 0 {
                cap[(c + 1) * n + nc + 1 + l] = 64; // "infinite": more than any class holds
            }
        }
    }
    for l in 0..nl {
        cap[(nc + 1 + l) * n + sink] = i32::from(caps[from + l]);
    }
    let mut flow = 0u32;
    let mut prev = vec![usize::MAX; n];
    let mut queue = Vec::with_capacity(n);
    loop {
        prev.iter_mut().for_each(|p| *p = usize::MAX);
        prev[0] = 0;
        queue.clear();
        queue.push(0);
        let mut qi = 0;
        while qi < queue.len() && prev[sink] == usize::MAX {
            let u = queue[qi];
            qi += 1;
            for v in 0..n {
                if prev[v] == usize::MAX && cap[u * n + v] > 0 {
                    prev[v] = u;
                    queue.push(v);
                }
            }
        }
        if prev[sink] == usize::MAX {
            break;
        }
        let mut aug = i32::MAX;
        let mut v = sink;
        while v != 0 {
            let u = prev[v];
            aug = aug.min(cap[u * n + v]);
            v = u;
        }
        let mut v = sink;
        while v != 0 {
            let u = prev[v];
            cap[u * n + v] -= aug;
            cap[v * n + u] += aug;
            v = u;
        }
        flow += aug as u32;
    }
    flow == need
}

/// The worlds consistent with a tracker (docs/M3-SPEC.md §4.2).
#[derive(Clone, Debug)]
pub struct WorldSpace {
    seat: u8,
    own_hand: Bits,
    /// The locations, in `Loc` order.
    locs: Vec<Loc>,
    caps: Vec<u8>,
    /// Cards of each class, and the locations each class allows (classes in ascending mask
    /// order).
    class_cards: Vec<Bits>,
    class_allow: Vec<u32>,
    prune: bool,
    memo: Memo,
    total: u128,
}

impl WorldSpace {
    /// The world space of `t`: a max-flow feasibility check (§4.3), then the exact count with
    /// max-flow pruning of the counting.
    pub fn new(t: &Tracker) -> Result<WorldSpace, Infeasible> {
        Self::build(t, true)
    }

    /// The same without max-flow pruning inside the counting (the root check still runs); for
    /// the identity test `count_prune_identity`.
    pub fn new_unpruned(t: &Tracker) -> Result<WorldSpace, Infeasible> {
        Self::build(t, false)
    }

    fn build(t: &Tracker, prune: bool) -> Result<WorldSpace, Infeasible> {
        // Classes: unknown cards grouped by identical masks, in ascending mask order.
        let mut by_mask: Vec<(u32, Bits)> = Vec::new();
        for c in cards(t.unknown) {
            let m = t.allowed[usize::from(c)];
            match by_mask.iter_mut().find(|(mm, _)| *mm == m) {
                Some((_, b)) => *b |= bit(c),
                None => by_mask.push((m, bit(c))),
            }
        }
        by_mask.sort();
        assert!(
            by_mask.len() <= MAX_CLASSES,
            "at most {MAX_CLASSES} card classes (found {})",
            by_mask.len()
        );
        assert!(t.locs.len() < 64, "location index fits the memo key");
        let mut ws = WorldSpace {
            seat: t.seat,
            own_hand: t.hand,
            locs: t.locs.iter().map(|&(l, _)| l).collect(),
            caps: t.locs.iter().map(|&(_, c)| c).collect(),
            class_cards: by_mask.iter().map(|&(_, b)| b).collect(),
            class_allow: by_mask.iter().map(|&(m, _)| m).collect(),
            prune,
            memo: Memo::default(),
            total: 0,
        };
        let rem = ws.class_sizes();
        if !feasible(&rem, &ws.class_allow, &ws.caps, 0) {
            return Err(Infeasible);
        }
        let mut memo = Memo::default();
        ws.total = ws.count_rec(0, &rem, &mut memo);
        ws.memo = memo;
        if ws.total == 0 {
            return Err(Infeasible);
        }
        Ok(ws)
    }

    fn class_sizes(&self) -> Vec<u8> {
        self.class_cards.iter().map(|&b| count(b)).collect()
    }

    /// Every transition `k` at location `l` from `rem` (a count vector with `Σ k = cap(l)`,
    /// `k_c ≤ rem_c`, `k_c = 0` where `l` is forbidden for `c`), in lexicographic order over the
    /// classes, with its weight `Π_c C(rem_c, k_c)`. Stops early when `f` returns `false`.
    fn transitions(&self, l: usize, rem: &[u8], mut f: impl FnMut(&[u8], u128) -> bool) {
        let nc = rem.len();
        let cap = self.caps[l];
        let mut k = vec![0u8; nc];
        let allowed: Vec<usize> = (0..nc)
            .filter(|&c| self.class_allow[c] & (1 << l) != 0 && rem[c] > 0)
            .collect();
        // Cards still placeable by the remaining allowed classes, to skip impossible prefixes.
        let mut suffix = vec![0u32; allowed.len() + 1];
        for i in (0..allowed.len()).rev() {
            suffix[i] = suffix[i + 1] + u32::from(rem[allowed[i]]);
        }
        if suffix[0] < u32::from(cap) {
            return;
        }
        #[allow(clippy::too_many_arguments)]
        fn rec(
            i: usize,
            left: u8,
            allowed: &[usize],
            suffix: &[u32],
            rem: &[u8],
            k: &mut [u8],
            w: u128,
            f: &mut dyn FnMut(&[u8], u128) -> bool,
        ) -> bool {
            if i == allowed.len() {
                return if left == 0 { f(k, w) } else { true };
            }
            let c = allowed[i];
            let lo = u32::from(left).saturating_sub(suffix[i + 1]) as u8;
            let hi = left.min(rem[c]);
            for x in lo..=hi {
                k[c] = x;
                let w2 = w
                    .checked_mul(binom(rem[c], x))
                    .expect("world count overflow");
                if !rec(i + 1, left - x, allowed, suffix, rem, k, w2, f) {
                    k[c] = 0;
                    return false;
                }
            }
            k[c] = 0;
            true
        }
        rec(0, cap, &allowed, &suffix, rem, &mut k, 1, &mut f);
    }

    fn count_rec(&self, l: usize, rem: &[u8], memo: &mut Memo) -> u128 {
        if l == self.locs.len() {
            return u128::from(rem.iter().all(|&r| r == 0));
        }
        let kk = key(l, rem);
        if let Some(&v) = memo.get(&kk) {
            return v;
        }
        if self.prune && !feasible(rem, &self.class_allow, &self.caps, l) {
            memo.insert(kk, 0);
            return 0;
        }
        let mut subs: Vec<(Vec<u8>, u128)> = Vec::new();
        self.transitions(l, rem, |k, w| {
            subs.push((rem.iter().zip(k).map(|(&r, &x)| r - x).collect(), w));
            true
        });
        let mut total: u128 = 0;
        for (nx, w) in subs {
            let sub = self.count_rec(l + 1, &nx, memo);
            if sub != 0 {
                total = total
                    .checked_add(w.checked_mul(sub).expect("world count overflow"))
                    .expect("world count overflow");
            }
        }
        memo.insert(kk, total);
        total
    }

    /// The count from a state already visited by the counting.
    fn lookup(&self, l: usize, rem: &[u8]) -> u128 {
        if l == self.locs.len() {
            return u128::from(rem.iter().all(|&r| r == 0));
        }
        self.memo.get(&key(l, rem)).copied().unwrap_or(0)
    }

    /// The exact number of consistent worlds (≥ 1).
    pub fn count(&self) -> u128 {
        self.total
    }

    /// The number of card classes.
    pub fn classes(&self) -> usize {
        self.class_cards.len()
    }

    /// The classes: their cards and their masks of allowed locations.
    pub fn class_list(&self) -> Vec<(Bits, u32)> {
        self.class_cards
            .iter()
            .copied()
            .zip(self.class_allow.iter().copied())
            .collect()
    }

    /// The number of memoised counting states.
    pub fn states(&self) -> usize {
        self.memo.len()
    }

    /// An exactly uniform integer in `0..n` (`n > 0`): two `u64` draws make a `u128`, and values
    /// at or above `⌊2^128 / n⌋ · n` are drawn again, so there is no modulo bias (§4.4).
    pub fn uniform_below(rng: &mut Rng, n: u128) -> u128 {
        assert!(n > 0, "a range of at least one value");
        // n divides 2^128 exactly when 2^128 mod n = 0, i.e. (2^128 − 1) mod n = n − 1.
        let divides = u128::MAX % n == n - 1;
        // ⌊(2^128 − 1) / n⌋ · n = ⌊2^128 / n⌋ · n unless n divides 2^128.
        let limit = (u128::MAX / n) * n;
        loop {
            let x = (u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64());
            if divides || x < limit {
                return x % n;
            }
        }
    }

    /// One world, exactly uniform over the consistent worlds (docs/M3-SPEC.md §4.4):
    /// 1. draw `r` uniform in `[0, N)`;
    /// 2. walk the locations; at each, go through its transitions in order, each with mass
    ///    `w(k) · count(next)`, and take the one `r` falls in (then `r` modulo `count(next)` is
    ///    uniform for the rest of the walk), so the class matrix M has probability
    ///    `Π_c (n_c! / Π_l M[c][l]!) / N`;
    /// 3. shuffle each class's cards and deal them to the locations in order by `M[c][·]`, which
    ///    gives each of the `Π_c n_c! / Π_l M[c][l]!` placements of M the same probability.
    ///
    /// So every world has probability exactly `1/N`.
    pub fn sample(&self, rng: &mut Rng) -> World {
        let mut r = Self::uniform_below(rng, self.total);
        let nc = self.class_cards.len();
        let mut rem = self.class_sizes();
        let mut matrix: Vec<Vec<u8>> = Vec::with_capacity(self.locs.len());
        for l in 0..self.locs.len() {
            let mut chosen: Option<Vec<u8>> = None;
            let mut next = vec![0u8; nc];
            self.transitions(l, &rem, |k, w| {
                for c in 0..nc {
                    next[c] = rem[c] - k[c];
                }
                let sub = self.lookup(l + 1, &next);
                let mass = w * sub; // ≤ N, so no overflow
                if r < mass {
                    // r is uniform in [0, w·sub); its residue is uniform in [0, sub) for the
                    // locations that follow (the factor w is the choice of cards, made by the
                    // shuffle below).
                    r %= sub;
                    chosen = Some(k.to_vec());
                    false
                } else {
                    r -= mass;
                    true
                }
            });
            let k = chosen.expect("the masses sum to the count");
            for c in 0..nc {
                rem[c] -= k[c];
            }
            matrix.push(k);
        }
        // Within each class: a uniform permutation, cut into consecutive groups by location.
        let mut world = self.empty_world();
        for c in 0..nc {
            let mut cs: Vec<Card> = cards(self.class_cards[c]).collect();
            rng.shuffle(&mut cs);
            let mut it = cs.into_iter();
            for (l, row) in matrix.iter().enumerate() {
                for _ in 0..row[c] {
                    self.place(&mut world, l, it.next().expect("class size"));
                }
            }
        }
        world.burns.sort();
        world
    }

    fn empty_world(&self) -> World {
        let mut hands = [0 as Bits; 8];
        hands[usize::from(self.seat)] = self.own_hand;
        World {
            hands,
            burns: Vec::new(),
            undealt: 0,
        }
    }

    fn place(&self, w: &mut World, l: usize, c: Card) {
        match self.locs[l] {
            Loc::Hand(s) => w.hands[usize::from(s)] |= bit(c),
            Loc::Undealt => w.undealt |= bit(c),
            loc @ Loc::Burn(..) => w.burns.push((loc, c)),
        }
    }

    /// Every consistent world in canonical (sorted) order, iff there are at most `limit`.
    pub fn enumerate(&self, limit: u64) -> Option<Vec<World>> {
        if self.total > u128::from(limit) {
            return None;
        }
        let rem = self.class_sizes();
        let mut out = Vec::with_capacity(self.total as usize);
        let mut matrix = Vec::new();
        self.enum_matrices(0, &rem, &mut matrix, &mut out);
        out.sort();
        assert_eq!(
            out.len() as u128,
            self.total,
            "enumeration matches the count"
        );
        Some(out)
    }

    fn enum_matrices(&self, l: usize, rem: &[u8], m: &mut Vec<Vec<u8>>, out: &mut Vec<World>) {
        if l == self.locs.len() {
            let w = self.empty_world();
            self.enum_cards(0, m, w, out);
            return;
        }
        let mut ks: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        self.transitions(l, rem, |k, _| {
            let nx: Vec<u8> = rem.iter().zip(k).map(|(&r, &x)| r - x).collect();
            if self.lookup(l + 1, &nx) > 0 {
                ks.push((k.to_vec(), nx));
            }
            true
        });
        for (k, nx) in ks {
            m.push(k);
            self.enum_matrices(l + 1, &nx, m, out);
            m.pop();
        }
    }

    /// Every way to split each class's cards among the locations by the matrix.
    fn enum_cards(&self, c: usize, m: &[Vec<u8>], w: World, out: &mut Vec<World>) {
        if c == self.class_cards.len() {
            let mut w = w;
            w.burns.sort();
            out.push(w);
            return;
        }
        let counts: Vec<u8> = m.iter().map(|row| row[c]).collect();
        self.split(c, 0, self.class_cards[c], &counts, m, w, out);
    }

    #[allow(clippy::too_many_arguments)]
    fn split(
        &self,
        c: usize,
        l: usize,
        left: Bits,
        counts: &[u8],
        m: &[Vec<u8>],
        w: World,
        out: &mut Vec<World>,
    ) {
        if l == counts.len() {
            self.enum_cards(c + 1, m, w, out);
            return;
        }
        let pool: Vec<Card> = cards(left).collect();
        let mut pick = Vec::new();
        subsets(&pool, usize::from(counts[l]), 0, &mut pick, &mut |sel| {
            let mut w2 = w.clone();
            let mut used: Bits = 0;
            for &x in sel {
                self.place(&mut w2, l, x);
                used |= bit(x);
            }
            self.split(c, l + 1, left & !used, counts, m, w2, out);
        });
    }
}

/// Every `k`-subset of `pool[from..]` added to `pick`, in lexicographic order.
fn subsets(pool: &[Card], k: usize, from: usize, pick: &mut Vec<Card>, f: &mut dyn FnMut(&[Card])) {
    if pick.len() == k {
        f(pick);
        return;
    }
    for i in from..pool.len() {
        if pool.len() - i < k - pick.len() {
            break;
        }
        pick.push(pool[i]);
        subsets(pool, k, i + 1, pick, f);
        pick.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binomials() {
        assert_eq!(binom(53, 26), 973_469_712_824_056);
        assert_eq!(binom(5, 7), 0);
        assert_eq!(binom(0, 0), 1);
        // Row sums are powers of two.
        for n in 0..=53u8 {
            let s: u128 = (0..=n).map(|k| binom(n, k)).sum();
            assert_eq!(s, 1u128 << n);
        }
    }

    #[test]
    fn flow_detects_infeasible() {
        // Two classes of 2 cards; locations of capacity 2 and 2.
        assert!(feasible(&[2, 2], &[0b01, 0b11], &[2, 2], 0));
        assert!(!feasible(&[2, 2], &[0b01, 0b01], &[2, 2], 0));
        assert!(feasible(&[1], &[0b10], &[2, 1], 1));
        assert!(!feasible(&[1, 1], &[0b10, 0b10], &[2, 1], 1));
        assert!(feasible(&[0, 0], &[0, 0], &[3, 0], 1));
    }

    #[test]
    fn uniform_below_is_in_range_and_unbiased_on_small_ranges() {
        let mut rng = Rng::new(1);
        for n in [1u128, 2, 3, 1 << 100, u128::MAX, (1u128 << 127) + 1] {
            for _ in 0..50 {
                assert!(WorldSpace::uniform_below(&mut rng, n) < n);
            }
        }
        let mut hits = [0u32; 3];
        for _ in 0..30_000 {
            hits[WorldSpace::uniform_below(&mut rng, 3) as usize] += 1;
        }
        assert!(
            hits.iter().all(|&h| (9_500..10_500).contains(&h)),
            "{hits:?}"
        );
    }

    #[test]
    fn log2_q8_is_the_floor() {
        assert_eq!(log2_q8(0), 0);
        assert_eq!(log2_q8(1), 0);
        assert_eq!(log2_q8(2), 256);
        assert_eq!(log2_q8(3), 405); // 256·1.58496 = 405.75
        assert_eq!(log2_q8(1000), 2551); // 256·9.96578 = 2551.24
        assert_eq!(log2_q8(1u128 << 120), 256 * 120);
        assert_eq!(log2_q8((1u128 << 120) - 1), 256 * 120 - 1);
        assert_eq!(log2_q8(u128::MAX), 256 * 128 - 1);
        // 10^36: 256·119.589 = 30614.9
        assert_eq!(log2_q8(10u128.pow(36)), 30614);
    }

    #[test]
    fn memo_keys_are_distinct() {
        assert_ne!(key(1, &[0, 1]), key(0, &[1, 1]));
        assert_ne!(key(0, &[1, 0]), key(0, &[0, 1]));
        assert_ne!(key(0, &[53, 0]), key(0, &[0, 53]));
    }
}
