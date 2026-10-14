//! The engine's own pinned random-number generator.
//!
//! SplitMix64 expands a 64-bit seed into the state of xoshiro256**, which generates the numbers.
//! Bounded integers use Lemire's unbiased method on `u64`/`u128` only (no `usize` inside), and the
//! shuffle is our own Fisher–Yates. The algorithms are fixed forever: `tools/rng_golden.py` is an
//! independent Python implementation and `tests/rng_golden.rs` checks that both agree on every
//! platform (Windows, Linux, wasm32).

/// SplitMix64, used only to expand a seed.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Start from a seed.
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// xoshiro256**, the engine's generator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Seed the generator: the four state words are consecutive SplitMix64 outputs of `seed`.
    pub fn new(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        let s = [sm.next_u64(), sm.next_u64(), sm.next_u64(), sm.next_u64()];
        Self { s }
    }

    /// Derive an independent generator for a labelled sub-stream (chunk, world, seat, ...).
    /// Each label is mixed through SplitMix64 so related labels give unrelated streams.
    pub fn derive(seed: u64, labels: &[u64]) -> Self {
        let mut sm = SplitMix64::new(seed);
        let mut acc = sm.next_u64();
        for &l in labels {
            let mut inner = SplitMix64::new(acc ^ l);
            acc = inner.next_u64();
        }
        Self::new(acc)
    }

    /// Next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// An unbiased integer in `0..n` (Lemire's nearly-divisionless method). Panics if `n == 0`.
    pub fn bounded(&mut self, n: u64) -> u64 {
        assert!(n > 0, "bounded(0) is undefined");
        let threshold = n.wrapping_neg() % n;
        loop {
            let x = self.next_u64();
            let m = u128::from(x) * u128::from(n);
            let low = m as u64;
            if low >= threshold {
                return (m >> 64) as u64;
            }
        }
    }

    /// Fisher–Yates shuffle, from the end of the slice.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        let len = items.len() as u64;
        if len < 2 {
            return;
        }
        let mut i = len - 1;
        while i >= 1 {
            let j = self.bounded(i + 1);
            items.swap(i as usize, j as usize);
            i -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_reference_value() {
        // First output of SplitMix64 from seed 0 (well known: 0xE220A8397B1DCDAF).
        let mut sm = SplitMix64::new(0);
        assert_eq!(sm.next_u64(), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn bounded_stays_in_range_and_covers() {
        let mut r = Rng::new(7);
        let mut seen = [false; 8];
        for _ in 0..1000 {
            let v = r.bounded(8);
            assert!(v < 8);
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut r = Rng::new(99);
        let mut deck: Vec<u8> = (0..53).collect();
        r.shuffle(&mut deck);
        let mut sorted = deck.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..53).collect::<Vec<u8>>());
        assert_ne!(deck, (0..53).collect::<Vec<u8>>());
    }

    #[test]
    fn derive_differs_by_label() {
        let a = Rng::derive(1, &[1, 2]);
        let b = Rng::derive(1, &[2, 1]);
        assert_ne!(a, b);
        assert_eq!(Rng::derive(1, &[1, 2]), a);
    }
}
