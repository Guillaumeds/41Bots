//! The optimised solver's transposition table [a memory of positions already solved]: a fixed
//! number of slots (bounded memory), keyed at trick boundaries exactly as PLAN amendment A2 says.
//!
//! - **Key**: the remaining hands, the leader, the tricks each team has won this hand, whether
//!   the 5 of trumps has been played, the best trump so far and its team, and the per-team bonus
//!   flags. The match state (games won, dealer's team) and the scores at the root are constant
//!   during one solve, and the table is emptied for every solve. The current scores need not be
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
//! - **Reuse** (see [`Table`]): a table is kept between solves and emptied in O(1) by moving to
//!   a new generation [a solve number stamped on every slot written; slots stamped with an
//!   older number count as empty]. Its contents after every operation are slot for slot those of
//!   a new table, so reuse cannot change an answer either.

use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

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

    pub(crate) fn hash(&self) -> u64 {
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
    /// The generation the slot was written in: it is filled only while that is the table's
    /// current generation (generation 0 is never current).
    generation: u32,
}

/// Bytes per slot.
pub const SLOT_BYTES: usize = core::mem::size_of::<Slot>();

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
    /// The slots held; the table in use is the first `buckets * WAYS` of them (the rest are kept
    /// for later growth and hold only slots of earlier generations).
    slots: Vec<Slot>,
    buckets: u64,
    max_buckets: u64,
    /// The current generation (≥ 1).
    generation: u32,
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
    generation: 0,
};

/// The bucket count of a table capped at about `bytes` bytes (at least one bucket).
fn max_buckets_of(bytes: usize) -> u64 {
    ((bytes / SLOT_BYTES / WAYS) as u64).max(1)
}

/// Memory of one table, in bytes, from the growth rule ([`Table::put`], [`Table::grow`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableMemory {
    /// What the table holds once grown to its cap, and keeps between solves.
    pub cap: usize,
    /// The most it can make the process hold: every block of its growth chain at once. Each
    /// growth step reallocates the slot vector to the new size; unless the allocator extends the
    /// block in place, the old block and the new one are both held while the slots are copied,
    /// and the old block, though freed, may stay resident (glibc keeps freed blocks below its
    /// mmap threshold in its heap for reuse). The chain is the start size doubled while below
    /// the cap (the last of these, `old` buckets, is at least half the cap), then the cap, so
    /// the peak is `cap + old + old/2 + … + start` = `cap + 2·old − start`: about 2 × cap when
    /// the cap is an exact doubling of the start size, up to just under 3 × cap. It depends on
    /// the growth rule only, never on the allocator, so no allocator can exceed it; glibc in
    /// practice stays well below (blocks of 32 MiB and more are moved by `mremap`, not copied).
    pub peak: usize,
}

impl TableMemory {
    /// `peak / cap` (1 when the table never grows).
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
    let max = max_buckets_of(bytes);
    let mut b = max.min(START_BUCKETS);
    let mut peak = b;
    // The growth steps in order: b → min(2b, max) while below the cap; every block counted.
    while b < max {
        b = (b * 2).min(max);
        peak += b;
    }
    let slot = WAYS * SLOT_BYTES;
    TableMemory {
        cap: max as usize * slot,
        peak: peak as usize * slot,
    }
}

/// The memory of a [`SharedTable`] of `mb` megabytes for `threads` threads
/// ([`SharedTable::new`]): every shard is a [`Table`], each counted at its own cap and peak (a
/// shard's old blocks may stay resident after it grows, so all shards can be at their peak at
/// once), plus the shard locks.
pub fn shared_table_memory(mb: usize, threads: usize) -> TableMemory {
    let (n, per) = shard_layout(mb, threads);
    let shard = table_memory(per);
    let locks = n * core::mem::size_of::<Mutex<Table>>();
    TableMemory {
        cap: n * shard.cap + locks,
        peak: n * shard.peak + locks,
    }
}

/// The first slot of the bucket of `hash` in a table of `buckets` buckets. Multiply-shift range
/// reduction (no modulo bias concerns: placement only); with more buckets a key's bucket index
/// never decreases.
fn bucket_in(hash: u64, buckets: u64) -> usize {
    (((u128::from(hash) * u128::from(buckets)) >> 64) as usize) * WAYS
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
        Table {
            slots: vec![EMPTY; buckets as usize * WAYS],
            buckets,
            max_buckets,
            generation: 1,
            entries: 0,
            replaced: 0,
        }
    }

    /// Empty the table for a new solve, now capped at about `bytes` bytes. It is then exactly a
    /// new table of that cap, keeping the memory it holds (memory above a lower cap is given
    /// back). O(1), except when memory is given back or the generation number wraps around
    /// (every 2^32 − 1 resets), when every slot is marked empty.
    pub fn reset(&mut self, bytes: usize) {
        self.max_buckets = max_buckets_of(bytes);
        self.buckets = self.max_buckets.min(START_BUCKETS);
        self.entries = 0;
        self.replaced = 0;
        let cap = self.max_buckets as usize * WAYS;
        if self.slots.len() > cap {
            self.slots.truncate(cap);
            self.slots.shrink_to_fit();
        }
        let need = self.buckets as usize * WAYS;
        if self.slots.len() < need {
            self.slots.resize(need, EMPTY);
        }
        if self.generation == u32::MAX {
            for s in &mut self.slots {
                s.generation = 0;
            }
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }

    /// Bytes of the table in use now (what a new table with the same contents would have
    /// allocated; memory kept from an earlier, larger solve is not counted).
    pub fn bytes(&self) -> usize {
        self.len() * SLOT_BYTES
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
        let need = nb as usize * WAYS;
        if self.slots.len() < need {
            // First growth to this size on this table: take the memory (exactly; kept after).
            self.slots.reserve_exact(need - self.slots.len());
            self.slots.resize(need, EMPTY);
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
            let b = k as usize * WAYS;
            let mut n = 0;
            for s in &self.slots[lo * WAYS..(hi + 1) * WAYS] {
                if s.generation == g && bucket_in(s.key.hash(), nb) == b {
                    moving[n] = *s;
                    n += 1;
                }
            }
            for s in &mut self.slots[b..b + WAYS] {
                s.generation = 0;
            }
            for s in &moving[..n] {
                if !self.fill(b, s.key, s.value) {
                    self.replace(b, s.key, s.value);
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
        self.slots[b..b + WAYS]
            .iter()
            .find(|s| s.generation == g && s.key == *k)
            .map(|s| s.value)
    }

    /// Store an exact value.
    pub fn put(&mut self, k: TtKey, v: MoneyChance) {
        if self.buckets < self.max_buckets && self.entries * 2 >= self.len() as u64 {
            self.grow();
        }
        let b = self.bucket(&k);
        if self.fill(b, k, v) {
            return;
        }
        // A full bucket below the memory cap in a table at least two-fifths full: double the
        // table first.
        if self.buckets < self.max_buckets && self.entries * 5 >= self.len() as u64 * 2 {
            self.grow();
            return self.put(k, v);
        }
        self.replace(b, k, v);
    }

    /// Put the entry in the first empty slot of the bucket starting at slot `b`; `false` if the
    /// bucket is full.
    fn fill(&mut self, b: usize, k: TtKey, v: MoneyChance) -> bool {
        let g = self.generation;
        match self.slots[b..b + WAYS]
            .iter_mut()
            .find(|s| s.generation != g)
        {
            Some(s) => {
                *s = Slot {
                    key: k,
                    value: v,
                    generation: g,
                };
                self.entries += 1;
                true
            }
            None => false,
        }
    }

    /// The replacement policy on the full bucket starting at slot `b`.
    fn replace(&mut self, b: usize, k: TtKey, v: MoneyChance) {
        let left = k.cards_left();
        let mut victim = b;
        for i in b + 1..b + WAYS {
            if self.slots[i].key.cards_left() < self.slots[victim].key.cards_left() {
                victim = i;
            }
        }
        if self.slots[victim].key.cards_left() <= left {
            self.slots[victim] = Slot {
                key: k,
                value: v,
                generation: self.generation,
            };
            self.replaced += 1;
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
    /// filled by putting every entry again on each doubling. Kept verbatim as the oracle.
    mod fresh {
        use super::super::{SLOT_BYTES, TtKey, WAYS};
        use engine::SEATS;
        use equity::MoneyChance;

        #[derive(Clone, Copy)]
        pub struct Slot {
            pub key: TtKey,
            pub value: MoneyChance,
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
            key: TtKey {
                hands: [0; SEATS],
                meta: 0,
            },
            value: MoneyChance([0; 4]),
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
                    self.put(s.key, s.value);
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
                    .map(|s| s.value)
            }

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
        t.slots[..t.len()]
            .iter()
            .map(|s| (s.generation == t.generation).then_some((s.key, s.value)))
            .collect()
    }

    fn fresh_layout(t: &fresh::Table) -> Layout {
        t.slots
            .iter()
            .map(|s| s.used.then_some((s.key, s.value)))
            .collect()
    }

    /// A universe of keys: few enough that keys repeat, with varied card counts (the replacement
    /// rule compares them) and meta words.
    fn universe(rng: &mut Rng, n: usize) -> Vec<TtKey> {
        (0..n)
            .map(|_| {
                let mut hands = [0 as Bits; SEATS];
                for h in &mut hands {
                    let mut x = rng.next_u64();
                    // Thin out to a few cards per hand, the count varying by key.
                    for _ in 0..(rng.next_u64() % 4) {
                        x &= rng.next_u64();
                    }
                    *h = x & 0x003F_FFFF_FFFF_FFFF;
                }
                TtKey {
                    hands,
                    meta: rng.next_u64() as u32,
                }
            })
            .collect()
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
                let v = MoneyChance([rng.next_u64(), i as u64, 0, 0]);
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
    fn reused_table_matches_a_new_table_slot_for_slot() {
        let mut rng = Rng::new(41);
        let keys = universe(&mut rng, 4000);
        let mut t = Table::with_bytes(0);
        for r in 0..80 {
            let cap = CAPS[(rng.next_u64() % CAPS.len() as u64) as usize];
            // A byte budget that is not a whole number of buckets, to check the rounding too.
            let bytes = cap * WAYS * SLOT_BYTES + (rng.next_u64() % SLOT_BYTES as u64) as usize;
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

    /// Generation numbers wrap around after 2^32 − 1 resets: every slot is then marked empty,
    /// so no entry of an earlier solve ever comes back.
    #[test]
    fn generation_wrap_never_returns_stale_entries() {
        let mut rng = Rng::new(5);
        let keys = universe(&mut rng, 600);
        let bytes = 23 * WAYS * SLOT_BYTES;
        let mut t = Table::with_bytes(bytes);
        t.generation = u32::MAX - 2;
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

    /// [`table_memory`] against the table itself: filled past its cap, the blocks its slot
    /// vector actually held (each capacity it had) sum to `peak`, and it ends holding `cap`.
    #[test]
    fn table_memory_matches_the_growth_chain() {
        let mut rng = Rng::new(17);
        let keys = universe(&mut rng, 20_000);
        for &cap in &CAPS {
            for extra in [0, SLOT_BYTES / 2] {
                let bytes = cap * WAYS * SLOT_BYTES + extra;
                let m = table_memory(bytes);
                let mut t = Table::with_bytes(bytes);
                let mut blocks = vec![t.slots.capacity()];
                for (i, k) in keys.iter().enumerate() {
                    t.put(*k, MoneyChance([i as u64, 0, 0, 0]));
                    if t.slots.capacity() != *blocks.last().expect("one block") {
                        blocks.push(t.slots.capacity());
                    }
                }
                assert_eq!(t.bytes(), m.cap, "cap {cap}: grown to the cap");
                assert_eq!(
                    t.slots.capacity() * SLOT_BYTES,
                    m.cap,
                    "cap {cap}: no spare"
                );
                assert_eq!(
                    blocks.iter().sum::<usize>() * SLOT_BYTES,
                    m.peak,
                    "cap {cap}: blocks {blocks:?}"
                );
            }
        }
        // The factor: 1 without growth, about 2 at an exact doubling, under 3 just above one.
        assert_eq!(table_memory(16 * WAYS * SLOT_BYTES).peak_factor(), 1.0);
        let f = |b: usize| table_memory(b * WAYS * SLOT_BYTES).peak_factor();
        assert!((f(1024) - (1024.0 + 2.0 * 512.0 - 16.0) / 1024.0).abs() < 1e-12);
        assert!((f(1025) - (1025.0 + 2.0 * 1024.0 - 16.0) / 1025.0).abs() < 1e-12);
        assert!(f(1025) < 3.0 && f(1024) < 2.0);
    }

    /// [`shared_table_memory`] against a shared table filled until every shard is at its cap.
    #[test]
    fn shared_table_memory_matches_a_full_table() {
        let mut rng = Rng::new(23);
        let keys = universe(&mut rng, 60_000);
        for (mb, threads) in [(1, 1), (2, 3), (3, 2)] {
            let s = SharedTable::new(mb, threads);
            for (i, k) in keys.iter().enumerate() {
                s.put(*k, MoneyChance([i as u64, 0, 0, 0]));
            }
            let m = shared_table_memory(mb, threads);
            let (n, per) = shard_layout(mb, threads);
            let locks = n * core::mem::size_of::<Mutex<Table>>();
            assert_eq!(s.bytes() + locks, m.cap, "{mb} MB, {threads} threads");
            assert_eq!(m.peak, n * table_memory(per).peak + locks);
            assert!(m.cap <= (mb << 20) + locks && m.peak > m.cap);
        }
    }

    #[test]
    fn shared_table_reset_empties_it() {
        let mut rng = Rng::new(9);
        let keys = universe(&mut rng, 300);
        let mut s = SharedTable::new(2, 3);
        for (i, k) in keys.iter().enumerate() {
            s.put(*k, MoneyChance([i as u64, 0, 0, 0]));
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
