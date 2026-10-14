//! The optimised solver's transposition table [a memory of positions already solved]: a fixed
//! number of slots (bounded memory), keyed at trick boundaries exactly as PLAN amendment A2 says.
//!
//! - **Key**: the remaining hands, the leader, the tricks each team has won this hand, whether
//!   the 5 of trumps has been played, the best trump so far and its team, and the per-team bonus
//!   flags. The match state (games won, dealer's team) and the scores at the root are constant
//!   during one solve, and a new table is made for every solve. The current scores need not be
//!   in the key: they equal the root scores plus 5 per trick won since the root plus the
//!   immediate bonus of the 5 of trumps if it was played since the root, and that bonus went to
//!   the team that played the 5 (it wins its trick), which the key holds as the best trump's
//!   team; its size (6 or 5) follows from the flags at the root. Tricks done this hand follow
//!   from the hand sizes.
//! - **Hash**: our own fixed mixing function (the SplitMix64 finaliser over the key's words);
//!   no random seed, so the table behaves identically on every run and machine.
//! - **Exactness**: every slot stores the whole key and is only used on an exact key match, and
//!   only exact values (from a node that was searched to the end, not stopped by pruning) are
//!   stored. So a probe either returns the true value or nothing, and replacement can only cost
//!   time, never change an answer.
//! - **Replacement** (deterministic): a key hashes to one bucket of [`WAYS`] slots. A new entry
//!   takes an empty slot if there is one; otherwise it replaces the slot holding the position
//!   with the fewest cards left (the cheapest to solve again), the lowest slot index first among
//!   equals, and only if that position has no more cards left than the new one.

use engine::SEATS;
use engine::cards::Bits;
use equity::MoneyChance;

use crate::Position;

/// Slots per bucket.
pub const WAYS: usize = 8;

/// A trick-boundary key (A2).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct TtKey {
    hands: [Bits; SEATS],
    /// leader (3 bits) | tricks per team (4 × 3 bits) | five played (1) | best present (1) |
    /// best rank (4) | best team (2) | flags (4).
    meta: u32,
}

impl TtKey {
    /// The key of a position at a trick boundary.
    pub fn of(p: &Position) -> TtKey {
        debug_assert_eq!(p.n_played, 0);
        let mut m = u32::from(p.leader);
        for t in 0..4 {
            m |= u32::from(p.hand.tricks[t]) << (3 + 3 * t);
        }
        m |= u32::from(p.hand.five_played) << 15;
        if let Some((r, t)) = p.best {
            m |= 1 << 16;
            m |= u32::from(r) << 17;
            m |= u32::from(t) << 21;
        }
        for t in 0..4 {
            m |= u32::from(p.hand.flags[t]) << (23 + t);
        }
        TtKey {
            hands: p.hands,
            meta: m,
        }
    }

    fn hash(&self) -> u64 {
        let mut h: u64 = 0x9E37_79B9_7F4A_7C15 ^ u64::from(self.meta);
        for &w in &self.hands {
            h = mix(h ^ w);
        }
        mix(h)
    }

    fn cards_left(&self) -> u32 {
        self.hands.iter().map(|h| h.count_ones()).sum()
    }
}

/// The SplitMix64 finaliser.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[derive(Clone, Copy)]
struct Slot {
    key: TtKey,
    value: MoneyChance,
    used: bool,
}

/// Bytes per slot.
pub const SLOT_BYTES: usize = core::mem::size_of::<Slot>();

/// The bounded table. It starts small and doubles (re-inserting every entry in slot order, so
/// the result is deterministic) when it is half full, or two-fifths full and a bucket overflows, until the
/// memory cap; at the cap the replacement policy takes over.
pub struct Table {
    slots: Vec<Slot>,
    buckets: u64,
    max_buckets: u64,
    /// Entries in use.
    pub entries: u64,
    /// Entries overwritten.
    pub replaced: u64,
}

const EMPTY: Slot = Slot {
    key: TtKey {
        hands: [0; SEATS],
        meta: 0,
    },
    value: MoneyChance([0; 4]),
    used: false,
};

impl Table {
    /// A table using at most about `mb` megabytes (at least one bucket).
    pub fn new(mb: usize) -> Table {
        let max_buckets = (((mb << 20) / SLOT_BYTES / WAYS) as u64).max(1);
        let buckets = max_buckets.min(16);
        Table {
            slots: vec![EMPTY; buckets as usize * WAYS],
            buckets,
            max_buckets,
            entries: 0,
            replaced: 0,
        }
    }

    /// Bytes allocated now.
    pub fn bytes(&self) -> usize {
        self.slots.len() * SLOT_BYTES
    }

    fn grow(&mut self) {
        let nb = (self.buckets * 2).min(self.max_buckets);
        let old = core::mem::replace(&mut self.slots, vec![EMPTY; nb as usize * WAYS]);
        self.buckets = nb;
        self.entries = 0;
        for s in old.into_iter().filter(|s| s.used) {
            self.put(s.key, s.value);
        }
    }

    fn bucket(&self, k: &TtKey) -> usize {
        // Multiply-shift range reduction (no modulo bias concerns: placement only).
        (((u128::from(k.hash()) * u128::from(self.buckets)) >> 64) as usize) * WAYS
    }

    /// The stored value of a key, if any.
    pub fn get(&self, k: &TtKey) -> Option<MoneyChance> {
        let b = self.bucket(k);
        self.slots[b..b + WAYS]
            .iter()
            .find(|s| s.used && s.key == *k)
            .map(|s| s.value)
    }

    /// Store an exact value.
    pub fn put(&mut self, k: TtKey, v: MoneyChance) {
        if self.buckets < self.max_buckets && self.entries * 2 >= self.slots.len() as u64 {
            self.grow();
        }
        let b = self.bucket(&k);
        let new = Slot {
            key: k,
            value: v,
            used: true,
        };
        if let Some(i) = (b..b + WAYS).find(|&i| !self.slots[i].used) {
            self.slots[i] = new;
            self.entries += 1;
            return;
        }
        // A full bucket below the memory cap in a table at least two-fifths full: double the
        // table first.
        if self.buckets < self.max_buckets && self.entries * 5 >= self.slots.len() as u64 * 2 {
            self.grow();
            return self.put(k, v);
        }
        let left = k.cards_left();
        let mut victim = b;
        for i in b + 1..b + WAYS {
            if self.slots[i].key.cards_left() < self.slots[victim].key.cards_left() {
                victim = i;
            }
        }
        if self.slots[victim].key.cards_left() <= left {
            self.slots[victim] = new;
            self.replaced += 1;
        }
    }
}
