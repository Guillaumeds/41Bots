//! The optimised solver's transposition table [a memory of positions already solved]: a fixed
//! number of slots (bounded memory), keyed at trick boundaries exactly as PLAN amendment A2 says.
//!
//! - **Key**: the cards still in hands, the leader, the tricks each team has won this hand,
//!   whether the 5 of trumps has been played, the best trump so far and its team, and the
//!   per-team bonus flags. The hands themselves are not stored: within one solve every card's
//!   owner is fixed by the root hands (a card only ever leaves a hand for the table; the table
//!   is emptied for every solve, and in a parallel solve every subproblem descends from the same
//!   root), so seat `s` holds exactly `root_hand[s] ∩ cards`, and the set of cards left (one
//!   word, 53 bits) names all eight hands. The match state (games won, dealer's team) and the
//!   scores at the root are constant during one solve. The current scores need not be in the
//!   key: they equal the root scores plus 5 per trick won since the root plus the immediate
//!   bonus of the 5 of trumps if it was played since the root, and that bonus went to the team
//!   that played the 5 (it wins its trick), which the key holds as the best trump's team; its
//!   size (6 or 5) follows from the flags at the root. Tricks done this hand follow from the
//!   number of cards left.
//! - **Value**: four chances summing to exactly `ONE` = 2^32, stored losslessly as three `u32`
//!   plus the index of the omitted (largest) entry; see [`Packed`] for the proof.
//! - **Slot**: 32 bytes (key 12, value 13, generation 2); a bucket of [`WAYS`] slots is 256
//!   bytes, aligned to 64, so a probe touches exactly four cache lines.
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
//! - **Reuse** (see [`Table`]): a table is kept between solves and emptied in O(1) by moving to
//!   a new generation [a solve number stamped on every slot written; slots stamped with an
//!   older number count as empty]. Its contents after every operation are slot for slot those of
//!   a new table, so reuse cannot change an answer either.

use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use engine::cards::Bits;
use equity::{MoneyChance, ONE};

use crate::Position;

/// Slots per bucket.
pub const WAYS: usize = 8;

/// A trick-boundary key (A2).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct TtKey {
    /// Every card still in a hand (the trick is empty at a boundary).
    cards: Bits,
    /// leader (3 bits) | tricks per team (4 × 3 bits) | five played (1) | best present (1) |
    /// best rank (4) | best team (2) | flags (4): 27 bits.
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
            cards: p.hands.iter().fold(0, |a, &h| a | h),
            meta: m,
        }
    }

    pub(crate) fn hash(&self) -> u64 {
        // The meta word is spread over all 64 bits by an odd multiplier before it meets the
        // card set, so keys that differ only in it land apart.
        mix(self.cards ^ u64::from(self.meta).wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    fn cards_left(&self) -> u32 {
        self.cards.count_ones()
    }
}

/// The SplitMix64 finaliser.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A money chance packed into 13 bytes, losslessly.
///
/// **Proof that the packing is exact.** A value is four whole numbers `v[0..4] ≥ 0` summing to
/// exactly `ONE` = 2^32. Let `m` be the index of a largest entry (the lowest such index). For
/// every other index `i`, `v[i] ≤ v[m]` and `v[i] + v[m] ≤ ONE`, so `2·v[i] ≤ ONE`, that is
/// `v[i] ≤ 2^31 < 2^32`: it fits a `u32` exactly. The omitted entry is recovered as
/// `v[m] = ONE − Σ_{i≠m} v[i]`, a subtraction of whole numbers with no rounding (the sum is at
/// most `ONE`). So `unpack(pack(v)) = v` for every valid value, including the degenerate ones
/// (`v[m] = ONE`, the rest 0; two entries of 2^31; and so on). Only the largest entry can equal
/// 2^32, which is why it is the one omitted. Tested in `packed_values_round_trip`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct Packed {
    /// The entries other than the omitted one, in team order.
    rest: [u32; 3],
    /// The omitted team (its entry is the largest).
    omit: u8,
}

impl Packed {
    fn pack(v: MoneyChance) -> Packed {
        let mut m = 0usize;
        for i in 1..4 {
            if v.0[i] > v.0[m] {
                m = i;
            }
        }
        let mut rest = [0u32; 3];
        let mut j = 0;
        for i in 0..4 {
            if i != m {
                debug_assert!(v.0[i] <= ONE / 2);
                rest[j] = v.0[i] as u32;
                j += 1;
            }
        }
        Packed {
            rest,
            omit: m as u8,
        }
    }

    fn unpack(self) -> MoneyChance {
        let m = usize::from(self.omit);
        let mut v = [0u64; 4];
        let mut j = 0;
        let mut sum = 0u64;
        for (i, x) in v.iter_mut().enumerate() {
            if i != m {
                *x = u64::from(self.rest[j]);
                sum += *x;
                j += 1;
            }
        }
        v[m] = ONE - sum;
        MoneyChance(v)
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Slot {
    cards: Bits,
    meta: u32,
    value: Packed,
    /// The generation the slot was written in: it is filled only while that is the table's
    /// current generation (generation 0 is never current).
    generation: u16,
}

/// A bucket: [`WAYS`] slots on four whole cache lines.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Bucket {
    slots: [Slot; WAYS],
}

/// Bytes per bucket.
const BUCKET_BYTES: usize = core::mem::size_of::<Bucket>();

/// Bytes per slot.
pub const SLOT_BYTES: usize = BUCKET_BYTES / WAYS;

/// Buckets of a table at the start of every solve.
const START_BUCKETS: u64 = 16;

/// The bounded table. It starts small and doubles (with exactly the contents that re-inserting
/// every entry in slot order into a new, larger table gives, so the result is deterministic)
/// when it is half full, or two-fifths full and a bucket overflows, until the memory cap; at the
/// cap the replacement policy takes over.
///
/// A table is kept between solves (one per thread) and emptied by [`Table::reset`], which costs
/// O(1): it moves to a new generation, and a slot counts as filled only if it was written in
/// the current generation, so no entry of an earlier solve can ever be returned. After a reset
/// the table starts again at the same small size and grows by the same rule, inside the memory
/// it already holds: a run of solves allocates (and first touches) its memory once per thread,
/// not once per solve. Slot for slot, its contents after every operation are those of a new
/// table given the same operations (tested against the table that allocated per solve), so
/// reuse changes neither an answer nor a counter.
pub struct Table {
    /// The buckets held; the table in use is the first `buckets` of them (the rest are kept for
    /// later growth and hold only slots of earlier generations).
    slots: Vec<Bucket>,
    buckets: u64,
    max_buckets: u64,
    /// The current generation (≥ 1).
    generation: u16,
    /// Entries in use.
    pub entries: u64,
    /// Entries overwritten.
    pub replaced: u64,
}

const EMPTY: Slot = Slot {
    cards: 0,
    meta: 0,
    value: Packed {
        rest: [0; 3],
        omit: 0,
    },
    generation: 0,
};

const EMPTY_BUCKET: Bucket = Bucket {
    slots: [EMPTY; WAYS],
};

/// The bucket count of a table capped at about `bytes` bytes (at least one bucket).
fn max_buckets_of(bytes: usize) -> u64 {
    ((bytes / BUCKET_BYTES) as u64).max(1)
}

/// Memory of one table, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableMemory {
    /// What the table holds once grown to its cap, and keeps between solves.
    pub cap: usize,
    /// The most it can make the process hold: the same as `cap`. The bucket vector is reserved
    /// to the cap when the table is made ([`Table::with_bytes`], [`Table::reset`]), as virtual
    /// memory whose pages are touched only as the table grows into them, so growth never
    /// reallocates: no block but the final one ever exists, and the process holds at most the
    /// cap (plus the pages the kernel rounds up to). Earlier, growth reallocated at each
    /// doubling, and the old and new blocks could be resident together, up to just under
    /// 3 × cap; that is why `peak` is still a separate field (and `peak_factor` is kept for the
    /// runner's records, where it now reads 1).
    pub peak: usize,
}

impl TableMemory {
    /// `peak / cap` (1: see [`TableMemory::peak`]).
    pub fn peak_factor(&self) -> f64 {
        if self.cap == 0 {
            1.0
        } else {
            self.peak as f64 / self.cap as f64
        }
    }
}

/// The memory of a [`Table`] capped at about `bytes` bytes ([`Table::with_bytes`]).
pub fn table_memory(bytes: usize) -> TableMemory {
    let cap = max_buckets_of(bytes) as usize * BUCKET_BYTES;
    TableMemory { cap, peak: cap }
}

/// The memory of a [`SharedTable`] of `mb` megabytes for `threads` threads
/// ([`SharedTable::new`]): every shard is a [`Table`], each counted at its cap, plus the shard
/// locks.
pub fn shared_table_memory(mb: usize, threads: usize) -> TableMemory {
    let (n, per) = shard_layout(mb, threads);
    let shard = table_memory(per);
    let locks = n * core::mem::size_of::<Mutex<Table>>();
    TableMemory {
        cap: n * shard.cap + locks,
        peak: n * shard.peak + locks,
    }
}

/// The bucket of `hash` in a table of `buckets` buckets. Multiply-shift range reduction (no
/// modulo bias concerns: placement only); with more buckets a key's bucket index never
/// decreases.
fn bucket_in(hash: u64, buckets: u64) -> usize {
    ((u128::from(hash) * u128::from(buckets)) >> 64) as usize
}

impl Table {
    /// A table using at most about `mb` megabytes (at least one bucket).
    pub fn new(mb: usize) -> Table {
        Table::with_bytes(mb << 20)
    }

    /// A table using at most about `bytes` bytes (at least one bucket).
    pub fn with_bytes(bytes: usize) -> Table {
        let max_buckets = max_buckets_of(bytes);
        let buckets = max_buckets.min(START_BUCKETS);
        // The whole cap is reserved now (virtual memory: the kernel maps a page only when it is
        // first written, which happens as the table grows into it), so growth never
        // reallocates and the process never holds more than the cap for this table.
        let mut slots = Vec::with_capacity(max_buckets as usize);
        slots.resize(buckets as usize, EMPTY_BUCKET);
        Table {
            slots,
            buckets,
            max_buckets,
            generation: 1,
            entries: 0,
            replaced: 0,
        }
    }

    /// Empty the table for a new solve, now capped at about `bytes` bytes. It is then exactly a
    /// new table of that cap, keeping the memory it holds (memory above a lower cap is given
    /// back; a higher cap is reserved now, as [`Table::with_bytes`] does). O(1), except when
    /// the cap changes or the generation number wraps around (every 2^16 − 1 resets), when
    /// every slot is marked empty.
    pub fn reset(&mut self, bytes: usize) {
        self.max_buckets = max_buckets_of(bytes);
        self.buckets = self.max_buckets.min(START_BUCKETS);
        self.entries = 0;
        self.replaced = 0;
        let cap = self.max_buckets as usize;
        if self.slots.len() > cap {
            self.slots.truncate(cap);
        }
        if self.slots.capacity() > cap {
            self.slots.shrink_to(cap);
        } else if self.slots.capacity() < cap {
            self.slots.reserve_exact(cap - self.slots.len());
        }
        let need = self.buckets as usize;
        if self.slots.len() < need {
            self.slots.resize(need, EMPTY_BUCKET);
        }
        if self.generation == u16::MAX {
            for b in &mut self.slots {
                for s in &mut b.slots {
                    s.generation = 0;
                }
            }
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }

    /// Bytes of the table in use now (what a new table with the same contents would have
    /// allocated; memory kept from an earlier, larger solve is not counted).
    pub fn bytes(&self) -> usize {
        self.buckets as usize * BUCKET_BYTES
    }

    /// Slots of the table in use now.
    fn len(&self) -> usize {
        self.buckets as usize * WAYS
    }

    /// Double the table (up to the cap) in place. The result is, slot for slot, a new table of
    /// the larger size into which every entry is put again in slot order: a key's new bucket is
    /// never before its old one, so new buckets are filled from the last down, each from the (at
    /// most two) old buckets whose hash ranges meet it, which are at or before it and so not yet
    /// overwritten. Each new bucket receives its entries in old slot order through the same
    /// placement rule as [`Table::put`] (a full table at the cap never grows), and a bucket's
    /// contents depend only on the entries put into it and their order, so the order in which
    /// buckets are filled does not matter.
    fn grow(&mut self) {
        let ob = self.buckets;
        let nb = (ob * 2).min(self.max_buckets);
        let need = nb as usize;
        if self.slots.len() < need {
            // First growth to this size on this table: within the capacity reserved at
            // creation, so no reallocation (these pages are touched now, and kept after).
            debug_assert!(self.slots.capacity() >= need);
            self.slots.resize(need, EMPTY_BUCKET);
        }
        self.buckets = nb;
        self.entries = 0;
        let g = self.generation;
        let mut moving = [EMPTY; 2 * WAYS];
        for k in (0..nb).rev() {
            // Old buckets whose hash ranges meet new bucket k (consecutive, at most two: each
            // new bucket's range is narrower than an old one's; all ≤ k since nb > ob).
            let lo = (u128::from(k) * u128::from(ob) / u128::from(nb)) as usize;
            let hi = ((u128::from(k + 1) * u128::from(ob) / u128::from(nb)) as usize)
                .min(ob as usize - 1);
            let b = k as usize;
            let mut n = 0;
            for s in self.slots[lo..=hi].iter().flat_map(|bk| bk.slots.iter()) {
                if s.generation == g && bucket_in(s.key().hash(), nb) == b {
                    moving[n] = *s;
                    n += 1;
                }
            }
            for s in &mut self.slots[b].slots {
                s.generation = 0;
            }
            for s in &moving[..n] {
                if !self.fill(b, s.key(), s.value) {
                    self.replace(b, s.key(), s.value);
                }
            }
        }
    }

    fn bucket(&self, k: &TtKey) -> usize {
        bucket_in(k.hash(), self.buckets)
    }

    /// The stored value of a key, if any.
    pub fn get(&self, k: &TtKey) -> Option<MoneyChance> {
        let b = self.bucket(k);
        let g = self.generation;
        self.slots[b]
            .slots
            .iter()
            .find(|s| s.generation == g && s.cards == k.cards && s.meta == k.meta)
            .map(|s| s.value.unpack())
    }

    /// Store an exact value.
    pub fn put(&mut self, k: TtKey, v: MoneyChance) {
        if self.buckets < self.max_buckets && self.entries * 2 >= self.len() as u64 {
            self.grow();
        }
        let b = self.bucket(&k);
        let pv = Packed::pack(v);
        if self.fill(b, k, pv) {
            return;
        }
        // A full bucket below the memory cap in a table at least two-fifths full: double the
        // table first.
        if self.buckets < self.max_buckets && self.entries * 5 >= self.len() as u64 * 2 {
            self.grow();
            return self.put(k, v);
        }
        self.replace(b, k, pv);
    }

    /// Put the entry in the first empty slot of bucket `b`; `false` if the bucket is full.
    fn fill(&mut self, b: usize, k: TtKey, v: Packed) -> bool {
        let g = self.generation;
        match self.slots[b].slots.iter_mut().find(|s| s.generation != g) {
            Some(s) => {
                *s = Slot {
                    cards: k.cards,
                    meta: k.meta,
                    value: v,
                    generation: g,
                };
                self.entries += 1;
                true
            }
            None => false,
        }
    }

    /// The replacement policy on the full bucket `b`.
    fn replace(&mut self, b: usize, k: TtKey, v: Packed) {
        let left = k.cards_left();
        let slots = &mut self.slots[b].slots;
        let mut victim = 0;
        for i in 1..WAYS {
            if slots[i].cards.count_ones() < slots[victim].cards.count_ones() {
                victim = i;
            }
        }
        if slots[victim].cards.count_ones() <= left {
            slots[victim] = Slot {
                cards: k.cards,
                meta: k.meta,
                value: v,
                generation: self.generation,
            };
            self.replaced += 1;
        }
    }
}

impl Slot {
    fn key(&self) -> TtKey {
        TtKey {
            cards: self.cards,
            meta: self.meta,
        }
    }
}

/// A table shared by the threads of one parallel solve: [`Table`] shards, each behind its own
/// lock, the shard picked by the low bits of the key's hash (the bucket inside a shard uses the
/// high bits). Every property that makes [`Table`] exact holds per shard: only exact values are
/// stored and a probe matches the whole key, so a probe returns the true value or nothing. What
/// is cached when depends on thread timing, which changes only the speed, never an answer.
pub struct SharedTable {
    shards: Vec<Mutex<Table>>,
    mask: u64,
    entries: AtomicI64,
    replaced: AtomicU64,
}

/// The shard count and bytes per shard of a shared table of `mb` megabytes for `threads`
/// threads: a power-of-two number of shards, enough to keep the threads from waiting on each
/// other, each at least 1 MB.
fn shard_layout(mb: usize, threads: usize) -> (usize, usize) {
    let want = (threads.max(1) * 64).next_power_of_two();
    let n = want.min(mb.max(1).next_power_of_two() / 2).max(1);
    (n, (mb << 20) / n)
}

impl SharedTable {
    /// A table of about `mb` megabytes in all, split into shards by [`shard_layout`].
    pub fn new(mb: usize, threads: usize) -> SharedTable {
        let (n, per) = shard_layout(mb, threads);
        SharedTable {
            shards: (0..n).map(|_| Mutex::new(Table::with_bytes(per))).collect(),
            mask: n as u64 - 1,
            entries: AtomicI64::new(0),
            replaced: AtomicU64::new(0),
        }
    }

    /// Empty the table for a new solve of `mb` megabytes on `threads` threads: it is then
    /// exactly `SharedTable::new(mb, threads)`, keeping the shards' memory when the shard count
    /// is unchanged (every shard is reset, O(1) each; see [`Table::reset`]).
    pub fn reset(&mut self, mb: usize, threads: usize) {
        let (n, per) = shard_layout(mb, threads);
        if n != self.shards.len() {
            *self = SharedTable::new(mb, threads);
            return;
        }
        for s in &mut self.shards {
            s.get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .reset(per);
        }
        *self.entries.get_mut() = 0;
        *self.replaced.get_mut() = 0;
    }

    fn shard(&self, k: &TtKey) -> std::sync::MutexGuard<'_, Table> {
        // A poisoned lock means a worker panicked; the solve fails with that panic anyway.
        self.shards[(k.hash() & self.mask) as usize]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The stored value of a key, if any.
    pub fn get(&self, k: &TtKey) -> Option<MoneyChance> {
        self.shard(k).get(k)
    }

    /// Store an exact value.
    pub fn put(&self, k: TtKey, v: MoneyChance) {
        let mut t = self.shard(&k);
        let (e, r) = (t.entries, t.replaced);
        t.put(k, v);
        let (e2, r2) = (t.entries, t.replaced);
        drop(t);
        self.entries
            .fetch_add(e2 as i64 - e as i64, Ordering::Relaxed);
        self.replaced.fetch_add(r2 - r, Ordering::Relaxed);
    }

    /// Entries in use (cheap; read while a solve runs).
    pub fn entries(&self) -> u64 {
        self.entries.load(Ordering::Relaxed).max(0) as u64
    }

    /// Entries overwritten.
    pub fn replaced(&self) -> u64 {
        self.replaced.load(Ordering::Relaxed)
    }

    /// Bytes of the shards' tables in use now (locks every shard; call when the solve is over).
    pub fn bytes(&self) -> usize {
        self.shards
            .iter()
            .map(|s| {
                s.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .bytes()
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::rng::Rng;

    /// The table as it was before reuse: allocated for every solve, and a new, larger allocation
    /// filled by putting every entry again on each doubling. Kept as the oracle (with the
    /// compact slot, so the two can be compared slot for slot).
    mod fresh {
        use super::super::{Packed, SLOT_BYTES, TtKey, WAYS};
        use equity::MoneyChance;

        #[derive(Clone, Copy)]
        pub struct Slot {
            pub key: TtKey,
            pub value: Packed,
            pub used: bool,
        }

        pub struct Table {
            pub slots: Vec<Slot>,
            buckets: u64,
            max_buckets: u64,
            pub entries: u64,
            pub replaced: u64,
        }

        const EMPTY: Slot = Slot {
            key: TtKey { cards: 0, meta: 0 },
            value: Packed {
                rest: [0; 3],
                omit: 0,
            },
            used: false,
        };

        impl Table {
            pub fn with_bytes(bytes: usize) -> Table {
                let max_buckets = ((bytes / SLOT_BYTES / WAYS) as u64).max(1);
                let buckets = max_buckets.min(16);
                Table {
                    slots: vec![EMPTY; buckets as usize * WAYS],
                    buckets,
                    max_buckets,
                    entries: 0,
                    replaced: 0,
                }
            }

            pub fn bytes(&self) -> usize {
                self.slots.len() * SLOT_BYTES
            }

            fn grow(&mut self) {
                let nb = (self.buckets * 2).min(self.max_buckets);
                let old = core::mem::replace(&mut self.slots, vec![EMPTY; nb as usize * WAYS]);
                self.buckets = nb;
                self.entries = 0;
                for s in old.into_iter().filter(|s| s.used) {
                    self.put(s.key, s.value.unpack());
                }
            }

            fn bucket(&self, k: &TtKey) -> usize {
                (((u128::from(k.hash()) * u128::from(self.buckets)) >> 64) as usize) * WAYS
            }

            pub fn get(&self, k: &TtKey) -> Option<MoneyChance> {
                let b = self.bucket(k);
                self.slots[b..b + WAYS]
                    .iter()
                    .find(|s| s.used && s.key == *k)
                    .map(|s| s.value.unpack())
            }

            pub fn put(&mut self, k: TtKey, v: MoneyChance) {
                if self.buckets < self.max_buckets && self.entries * 2 >= self.slots.len() as u64 {
                    self.grow();
                }
                let b = self.bucket(&k);
                let new = Slot {
                    key: k,
                    value: Packed::pack(v),
                    used: true,
                };
                if let Some(i) = (b..b + WAYS).find(|&i| !self.slots[i].used) {
                    self.slots[i] = new;
                    self.entries += 1;
                    return;
                }
                if self.buckets < self.max_buckets
                    && self.entries * 5 >= self.slots.len() as u64 * 2
                {
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
    }

    type Layout = Vec<Option<(TtKey, MoneyChance)>>;

    fn layout(t: &Table) -> Layout {
        t.slots[..t.buckets as usize]
            .iter()
            .flat_map(|b| b.slots.iter())
            .map(|s| (s.generation == t.generation).then_some((s.key(), s.value.unpack())))
            .collect()
    }

    fn fresh_layout(t: &fresh::Table) -> Layout {
        t.slots
            .iter()
            .map(|s| s.used.then_some((s.key, s.value.unpack())))
            .collect()
    }

    /// A universe of keys: few enough that keys repeat, with varied card counts (the replacement
    /// rule compares them) and meta words.
    fn universe(rng: &mut Rng, n: usize) -> Vec<TtKey> {
        (0..n)
            .map(|_| {
                let mut x = rng.next_u64();
                // Thin out to a varying number of cards.
                for _ in 0..(rng.next_u64() % 4) {
                    x &= rng.next_u64();
                }
                TtKey {
                    cards: x & 0x001F_FFFF_FFFF_FFFF,
                    meta: rng.next_u64() as u32 & 0x07FF_FFFF,
                }
            })
            .collect()
    }

    /// A valid value from a random split and a serial number.
    fn value(rng: &mut Rng, i: usize) -> MoneyChance {
        let a = rng.next_u64() % (ONE / 2);
        let b = (i as u64 * 7919) % (ONE / 4);
        MoneyChance::from_raw([a, b, ONE - a - b, 0]).expect("valid")
    }

    fn same(t: &Table, f: &fresh::Table, what: &str) {
        assert_eq!(t.entries, f.entries, "{what}: entries");
        assert_eq!(t.replaced, f.replaced, "{what}: replaced");
        assert_eq!(t.bytes(), f.bytes(), "{what}: bytes");
    }

    /// One solve's worth of operations on the reused table `t` (just reset to `bytes`) and on a
    /// new oracle table: identical answers, counters and slot-for-slot contents throughout.
    fn round(t: &mut Table, bytes: usize, keys: &[TtKey], ops: usize, rng: &mut Rng, what: &str) {
        let mut f = fresh::Table::with_bytes(bytes);
        same(t, &f, what);
        assert_eq!(layout(t), fresh_layout(&f), "{what}: empty at start");
        for i in 0..ops {
            let k = keys[(rng.next_u64() % keys.len() as u64) as usize];
            if rng.next_u64() % 10 < 7 {
                let v = value(rng, i);
                t.put(k, v);
                f.put(k, v);
            } else {
                assert_eq!(t.get(&k), f.get(&k), "{what}: get at op {i}");
            }
            same(t, &f, &format!("{what} op {i}"));
            if i % 97 == 0 {
                assert_eq!(layout(t), fresh_layout(&f), "{what}: layout at op {i}");
            }
        }
        assert_eq!(layout(t), fresh_layout(&f), "{what}: layout at the end");
        for k in keys {
            assert_eq!(t.get(k), f.get(k), "{what}: final get");
        }
    }

    /// Caps (in buckets) that exercise: no growth (1), doubling only (16·2^k), and a last
    /// growth step short of doubling (17, 23, 40, 100, 333), plus caps below the start size.
    const CAPS: [usize; 12] = [1, 2, 3, 5, 7, 16, 17, 23, 40, 64, 100, 333];

    #[test]
    fn slot_and_bucket_sizes() {
        assert_eq!(SLOT_BYTES, 32);
        assert_eq!(BUCKET_BYTES, 256);
        assert_eq!(core::mem::size_of::<Slot>(), 32);
        assert_eq!(core::mem::align_of::<Bucket>(), 64);
    }

    /// Packing is exact for every valid value, degenerate ones included (see [`Packed`]).
    #[test]
    fn packed_values_round_trip() {
        let check = |v: MoneyChance| {
            let p = Packed::pack(v);
            assert_eq!(p.unpack(), v, "{v:?}");
            assert!(p.rest.iter().all(|&x| u64::from(x) <= ONE / 2));
        };
        // Certain wins (one entry 2^32: the only entry that can exceed a u32).
        for t in 0..4 {
            check(MoneyChance::certain(t));
        }
        // Every arrangement of boundary amounts that sums to ONE: entries drawn from a set
        // including 0, 1, 2^31 − 1, 2^31, 2^31 + 1, 2^32 − 1 and 2^32.
        let edge = [
            0u64,
            1,
            2,
            ONE / 4,
            ONE / 2 - 1,
            ONE / 2,
            ONE / 2 + 1,
            ONE - 1,
            ONE,
        ];
        let mut n = 0;
        for &a in &edge {
            for &b in &edge {
                for &c in &edge {
                    if a + b + c <= ONE {
                        let d = ONE - a - b - c;
                        for v in [[a, b, c, d], [d, a, b, c], [c, d, a, b], [b, c, d, a]] {
                            check(MoneyChance::from_raw(v).expect("sums to ONE"));
                            n += 1;
                        }
                    }
                }
            }
        }
        assert!(n > 100);
        // Random values, including ties for the largest entry.
        let mut rng = Rng::new(3);
        for _ in 0..20_000 {
            let w: [u128; 4] = core::array::from_fn(|_| u128::from(rng.next_u64() >> 40));
            if w.iter().any(|&x| x > 0) {
                check(MoneyChance::from_weights(w));
            }
            let a = rng.next_u64() % (ONE / 2 + 1);
            check(MoneyChance::from_raw([a, a, ONE - 2 * a, 0]).expect("valid"));
            check(MoneyChance::from_raw([0, ONE - 2 * a, a, a]).expect("valid"));
        }
    }

    #[test]
    fn reused_table_matches_a_new_table_slot_for_slot() {
        let mut rng = Rng::new(41);
        let keys = universe(&mut rng, 4000);
        let mut t = Table::with_bytes(0);
        for r in 0..80 {
            let cap = CAPS[(rng.next_u64() % CAPS.len() as u64) as usize];
            // A byte budget that is not a whole number of buckets, to check the rounding too.
            let bytes = cap * BUCKET_BYTES + (rng.next_u64() % BUCKET_BYTES as u64) as usize;
            let ops = (rng.next_u64() % 6000) as usize;
            t.reset(bytes);
            round(
                &mut t,
                bytes,
                &keys,
                ops,
                &mut rng,
                &format!("round {r} cap {cap}"),
            );
        }
    }

    /// Generation numbers wrap around after 2^16 − 1 resets: every slot is then marked empty,
    /// so no entry of an earlier solve ever comes back.
    #[test]
    fn generation_wrap_never_returns_stale_entries() {
        let mut rng = Rng::new(5);
        let keys = universe(&mut rng, 600);
        let bytes = 23 * BUCKET_BYTES;
        let mut t = Table::with_bytes(bytes);
        t.generation = u16::MAX - 2;
        for r in 0..6 {
            t.reset(bytes);
            assert!(t.generation >= 1);
            assert!(
                keys.iter().all(|k| t.get(k).is_none()),
                "round {r}: stale entry"
            );
            round(
                &mut t,
                bytes,
                &keys,
                900,
                &mut rng,
                &format!("wrap round {r}"),
            );
            assert!(t.entries > 0);
        }
    }

    /// [`table_memory`] against the table itself: the bucket vector holds exactly the cap from
    /// the start, never reallocates while the table is filled past its cap (so the only block
    /// that ever exists is the cap: `peak == cap`), and the same holds across resets to lower
    /// and higher caps.
    #[test]
    fn table_memory_matches_the_table() {
        let mut rng = Rng::new(17);
        let keys = universe(&mut rng, 20_000);
        let mut t = Table::with_bytes(0);
        for &cap in &CAPS {
            for extra in [0, BUCKET_BYTES / 2] {
                let bytes = cap * BUCKET_BYTES + extra;
                let m = table_memory(bytes);
                assert_eq!(m.peak, m.cap, "cap {cap}: peak is the cap");
                assert_eq!(m.peak_factor(), 1.0);
                for (what, t) in [("new", &mut Table::with_bytes(bytes)), ("reset", &mut t)] {
                    if what == "reset" {
                        t.reset(bytes);
                    }
                    let block = t.slots.as_ptr();
                    assert_eq!(t.slots.capacity() * BUCKET_BYTES, m.cap, "{what} cap {cap}");
                    for (i, k) in keys.iter().enumerate() {
                        t.put(*k, value(&mut rng, i));
                        assert_eq!(t.slots.capacity() * BUCKET_BYTES, m.cap, "{what} {cap}");
                        assert_eq!(t.slots.as_ptr(), block, "{what} cap {cap}: moved");
                    }
                    assert_eq!(t.bytes(), m.cap, "{what} cap {cap}: grown to the cap");
                }
            }
        }
    }

    /// [`shared_table_memory`] against a shared table filled until every shard is at its cap.
    #[test]
    fn shared_table_memory_matches_a_full_table() {
        let mut rng = Rng::new(23);
        let keys = universe(&mut rng, 60_000);
        for (mb, threads) in [(1, 1), (2, 3), (3, 2)] {
            let s = SharedTable::new(mb, threads);
            for (i, k) in keys.iter().enumerate() {
                s.put(*k, value(&mut rng, i));
            }
            let m = shared_table_memory(mb, threads);
            let (n, per) = shard_layout(mb, threads);
            let locks = n * core::mem::size_of::<Mutex<Table>>();
            assert_eq!(s.bytes() + locks, m.cap, "{mb} MB, {threads} threads");
            assert_eq!(m.peak, n * table_memory(per).peak + locks);
            assert!(m.cap <= (mb << 20) + locks && m.peak == m.cap);
        }
    }

    #[test]
    fn shared_table_reset_empties_it() {
        let mut rng = Rng::new(9);
        let keys = universe(&mut rng, 300);
        let mut s = SharedTable::new(2, 3);
        for (i, k) in keys.iter().enumerate() {
            s.put(*k, value(&mut rng, i));
        }
        assert!(s.entries() > 0);
        s.reset(2, 3);
        assert_eq!((s.entries(), s.replaced()), (0, 0));
        assert!(keys.iter().all(|k| s.get(k).is_none()));
        assert_eq!(s.bytes(), SharedTable::new(2, 3).bytes());
        // Another layout (more threads, more megabytes): rebuilt, also empty.
        s.reset(64, 8);
        assert!(keys.iter().all(|k| s.get(k).is_none()));
        assert_eq!(s.bytes(), SharedTable::new(64, 8).bytes());
    }
}
