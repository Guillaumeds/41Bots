//! The 41 engine built for WebAssembly (docs/PLAN.md §6.1 and §7).
//!
//! For now it holds only the M0 determinism check (PLAN §13, M0 gate "WASM results equal native"):
//! [`random_matches_hash`] plays seeded random matches with the engine and hashes everything that
//! happened in them. `tools/wasm_check.sh` runs the same code natively (the `wasm-native` binary)
//! and as a `.wasm` module under Node, and fails if the two hashes differ.
//!
//! Lints: this crate denies `unsafe_code` instead of inheriting the workspace's `forbid`, because
//! the export [`result_hash`] needs the `no_mangle` attribute (see Cargo.toml). That export is the
//! only item allowed it. Everything else here is ordinary safe Rust.

use engine::game::{Match, Rules, Visibility};
use engine::rng::Rng;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a, 64-bit: a tiny, fixed, platform-independent hash of a byte stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fnv1a(u64);

impl Fnv1a {
    /// An empty hash (the offset basis).
    pub const fn new() -> Self {
        Self(FNV_OFFSET)
    }

    /// Feed bytes, in order.
    pub fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }

    /// The hash of everything written so far.
    pub const fn finish(self) -> u64 {
        self.0
    }
}

impl Default for Fnv1a {
    fn default() -> Self {
        Self::new()
    }
}

/// Plays random match `i` of the run that starts at `seed`, under the house rules, to the money.
///
/// The match is `Match::new(seed + i, Rules::HOUSE)` (wrapping addition). Every decision takes the
/// k-th legal action (legal actions are sorted by card index), with k drawn by
/// `bounded(number of legal actions)` from one generator per match, `Rng::derive(seed, &[i])`.
///
/// # Panics
/// If the engine refuses one of its own legal actions (an engine bug).
pub fn play_random_match(seed: u64, i: u32) -> Match {
    let mut m = Match::new(seed.wrapping_add(u64::from(i)), Rules::HOUSE);
    let mut chooser = Rng::derive(seed, &[u64::from(i)]);
    while m.to_act().is_some() {
        let legal = m.legal_actions();
        let k = chooser.bounded(legal.len() as u64) as usize;
        m.apply(legal[k])
            .expect("the engine refused one of its own legal actions");
    }
    m
}

/// Feeds one finished match into the hash.
///
/// Byte format, one line per event in log order: `public <canonical>\n` or
/// `private=<seat> <canonical>\n`, where `<canonical>` is `Event::canonical()`; then
/// `money=<a>,<b>,<c>,<d>\n` with the four teams' money.
///
/// # Panics
/// If the match is not over.
pub fn hash_match(h: &mut Fnv1a, m: &Match) {
    for (visibility, event) in m.events() {
        let line = match *visibility {
            Visibility::Public => format!("public {}\n", event.canonical()),
            Visibility::Private(seat) => format!("private={seat} {}\n", event.canonical()),
        };
        h.write(line.as_bytes());
    }
    let money = m
        .money()
        .expect("the match must be over before it is hashed");
    h.write(
        format!(
            "money={},{},{},{}\n",
            money[0], money[1], money[2], money[3]
        )
        .as_bytes(),
    );
}

/// Plays `matches` random matches from `seed` ([`play_random_match`] for `i` in `0..matches`) and
/// returns the FNV-1a hash of all of them, in order ([`hash_match`]).
pub fn random_matches_hash(seed: u64, matches: u32) -> u64 {
    let mut h = Fnv1a::new();
    for i in 0..matches {
        hash_match(&mut h, &play_random_match(seed, i));
    }
    h.finish()
}

/// The WebAssembly export: [`random_matches_hash`]. The native binary calls this same function.
///
/// From JavaScript: `result_hash(seed, matches)` with `seed` a BigInt and `matches` a number; the
/// u64 result arrives as a signed BigInt, so read it with `BigInt.asUintN(64, result)`.
#[allow(unsafe_code)] // The one exception in this crate: an export needs the `no_mangle` attribute.
#[unsafe(no_mangle)]
pub extern "C" fn result_hash(seed: u64, matches: u32) -> u64 {
    random_matches_hash(seed, matches)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fnv(bytes: &[u8]) -> u64 {
        let mut h = Fnv1a::new();
        h.write(bytes);
        h.finish()
    }

    #[test]
    fn fnv1a_published_vectors() {
        assert_eq!(fnv(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn no_matches_hash_to_the_offset_basis() {
        assert_eq!(result_hash(42, 0), FNV_OFFSET);
    }

    #[test]
    fn random_matches_finish_and_replay() {
        for i in 0..3 {
            let m = play_random_match(9, i);
            assert_eq!(m.to_act(), None);
            let money = m.money().expect("over");
            assert_eq!(money.iter().sum::<i32>(), 0);
            let again = Match::replay(9 + u64::from(i), Rules::HOUSE, m.actions()).expect("replay");
            assert_eq!(again, m);
        }
    }

    #[test]
    fn hash_is_deterministic_and_seed_sensitive() {
        let a = result_hash(5, 3);
        assert_eq!(a, result_hash(5, 3));
        assert_eq!(a, random_matches_hash(5, 3));
        assert_ne!(a, result_hash(6, 3));
        assert_ne!(a, result_hash(5, 2));
    }

    #[test]
    fn hash_streams_match_after_match() {
        let mut h = Fnv1a::new();
        for i in 0..2 {
            hash_match(&mut h, &play_random_match(u64::MAX, i));
        }
        assert_eq!(h.finish(), result_hash(u64::MAX, 2));
    }
}
