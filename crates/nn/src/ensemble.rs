//! An ensemble of oracle networks as one solver frontier evaluator (docs/M3-HYBRID-SPEC.md §6;
//! the optimizer's-curse and ensemble test of 2026-10-19, `search hybrid-check --net A,B,…`).
//!
//! The ensemble's value of a position is the **integer mean** of its members' money chances:
//! each member values the position exactly as it would alone ([`Oracle::eval`], its own caches
//! and batches), the four per-team totals are summed in `u128` over the members in the order
//! given (index order; the sum of whole numbers does not depend on it anyway), and the mean is
//! [`MoneyChance::mix`] with weight 1 each: every team's share is rounded down and the units
//! still missing to reach exactly 2^32 (at most three) go to the teams with the largest
//! remainders, ties to the lower team index (`equity::money`'s largest-remainder rule, the one
//! every money chance in the project uses). So the result is a whole-number function of the
//! members' values, byte-identical on every machine, thread count and WASM, and an ensemble of one
//! network is that network, byte for byte (the mean of one value is the value).
//!
//! **Speed** (2026-10-19): the members' weights together (about 70 MB for the four 2048-wide
//! O_3s) do not fit a processor's last-level cache as one network's do (16 MB per core complex on
//! an AMD EPYC 7B12), so every batch streams them from memory and the cost goes with the number
//! of passes over the weights, not with the positions valued: the first rented run made one pass
//! per member for every 13 positions or so (the solver's batches from 3 plays before the
//! frontier) and ran about 100 times slower than one network. An ensemble of several networks
//! therefore takes the solver's batches from as far back as they go ([`ENSEMBLE_PREFETCH`]: at
//! trick 2 every frontier position below the decision's first play, which the search reaches
//! almost all of anyway), values each such batch through every member in passes of up to
//! `nn::kernel::MAX_BATCH` positions, and keeps the batch's means in a table of this thread's
//! ([`Ensemble::prefetch`]) where the search's own requests find them. None of it changes a
//! value: a mean is a pure function of the position, kept or recomputed. The mean is taken from
//! per-team totals with no allocation ([`MoneyChance::mean_of_totals`]).
//!
//! The ensemble's name in values and reports is [`Ensemble::id`]: the member's SHA-256 for one
//! network, else `ens<N>:` and the members' SHA-256s joined by `+` in the order given.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use equity::MoneyChance;
use solver::MatchCtx;
use solver::frontier::{CanonPos, Frontier};

use crate::oracle::Oracle;

/// The batch depth an ensemble of several networks asks the solver for: 7 plays before the
/// frontier, the deepest a solver gathers from (module notes).
pub const ENSEMBLE_PREFETCH: u8 = 7;

static NEXT_ENSEMBLE: AtomicU64 = AtomicU64::new(1);

/// One thread's means of the positions of the latest batch an ensemble was handed
/// ([`Ensemble::prefetch`]): which ensemble, the match context, and the values (looked up, never
/// iterated, so the map's hashing cannot change anything).
#[derive(Default)]
struct Gathered {
    ensemble: u64,
    ctx: Option<MatchCtx>,
    values: HashMap<CanonPos, MoneyChance>,
}

thread_local! {
    static GATHERED: RefCell<Gathered> = RefCell::new(Gathered::default());
}

/// Several oracle networks of the same frontier trick, valued as their integer mean (module
/// notes).
pub struct Ensemble {
    members: Vec<Oracle>,
    /// Identifies this ensemble's batch table ([`Gathered`]).
    id: u64,
}

impl Ensemble {
    /// The ensemble of `members` (at least one, all trained for the same frontier trick).
    pub fn new(members: Vec<Oracle>) -> Result<Ensemble, String> {
        let first = members
            .first()
            .ok_or("an ensemble needs at least one network")?;
        if let Some(o) = members.iter().find(|o| o.trick != first.trick) {
            return Err(format!(
                "ensemble members of different tricks: {} (trick {}) and {} (trick {})",
                &first.sha256[..12],
                first.trick,
                &o.sha256[..12],
                o.trick
            ));
        }
        Ok(Ensemble {
            members,
            id: NEXT_ENSEMBLE.fetch_add(1, Ordering::Relaxed),
        })
    }

    /// The members, in the order given.
    pub fn members(&self) -> &[Oracle] {
        &self.members
    }

    /// The frontier trick.
    pub fn trick(&self) -> u8 {
        self.members[0].trick
    }

    /// The ensemble's name: the single member's SHA-256, or `ens<N>:<sha>+<sha>+…`.
    pub fn id(&self) -> String {
        if self.members.len() == 1 {
            return self.members[0].sha256.clone();
        }
        let shas: Vec<&str> = self.members.iter().map(|o| o.sha256.as_str()).collect();
        format!("ens{}:{}", shas.len(), shas.join("+"))
    }

    /// The integer mean of the members' values of one position (module notes).
    pub fn eval(&self, c: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        if self.members.len() == 1 {
            return self.members[0].eval(c, ctx);
        }
        let kept = GATHERED.with(|g| {
            let g = g.borrow();
            if g.ensemble == self.id && g.ctx == Some(*ctx) {
                g.values.get(c).copied()
            } else {
                None
            }
        });
        if let Some(v) = kept {
            return v;
        }
        let mut tot = [0u64; 4];
        for o in &self.members {
            let v = o.eval(c, ctx);
            add(&mut tot, &v);
        }
        MoneyChance::mean_of_totals(tot, self.members.len() as u64)
    }
}

/// Adds a money chance to per-team totals.
fn add(tot: &mut [u64; 4], v: &MoneyChance) {
    for (s, x) in tot.iter_mut().zip(v.0) {
        *s += x;
    }
}

/// The integer mean of money chances: totals per team, index-ordered, rounded by largest
/// remainder (ties to the lower team) so the result sums to exactly 2^32 ([`MoneyChance::mix`]
/// with weight 1 each, computed as [`MoneyChance::mean_of_totals`]: the same numbers). The mean
/// of one value is that value.
///
/// # Panics
/// If `vs` is empty.
pub fn mean(vs: &[MoneyChance]) -> MoneyChance {
    assert!(!vs.is_empty(), "the mean of no values");
    if vs.len() == 1 {
        return vs[0];
    }
    let mut tot = [0u64; 4];
    for v in vs {
        add(&mut tot, v);
    }
    MoneyChance::mean_of_totals(tot, vs.len() as u64)
}

impl Frontier for Ensemble {
    fn value(&self, pos: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        self.eval(pos, ctx)
    }

    /// Each member values the batch as one (its own batched kernel), then the means are taken
    /// position by position.
    fn value_batch(&self, pos: &[CanonPos], ctx: &MatchCtx, out: &mut [MoneyChance]) {
        if self.members.len() == 1 {
            self.members[0].eval_batch(pos, ctx, out);
            return;
        }
        let n = pos.len();
        let mut each = vec![MoneyChance::certain(0); n * self.members.len()];
        for (k, o) in self.members.iter().enumerate() {
            o.eval_batch(pos, ctx, &mut each[k * n..(k + 1) * n]);
        }
        for (i, slot) in out.iter_mut().enumerate().take(n) {
            let mut tot = [0u64; 4];
            for k in 0..self.members.len() {
                add(&mut tot, &each[k * n + i]);
            }
            *slot = MoneyChance::mean_of_totals(tot, self.members.len() as u64);
        }
    }

    /// One network's own batch depth; for several, [`ENSEMBLE_PREFETCH`] (module notes; values
    /// never depend on it).
    fn prefetch_depth(&self) -> u8 {
        if self.members.len() == 1 {
            self.members[0].prefetch_depth()
        } else {
            ENSEMBLE_PREFETCH
        }
    }

    /// One network: its own batch. Several: every member values the positions ahead as batches
    /// of up to `MAX_BATCH`, and their means replace this thread's table of the ensemble's latest
    /// batch, where [`Ensemble::eval`] finds them (module notes).
    fn prefetch(&self, pos: &[CanonPos], ctx: &MatchCtx) {
        if self.members.len() == 1 {
            self.members[0].prefetch(pos, ctx);
            return;
        }
        if pos.is_empty() {
            return;
        }
        let mut out = vec![MoneyChance::certain(0); pos.len()];
        self.value_batch(pos, ctx, &mut out);
        GATHERED.with(|g| {
            let mut g = g.borrow_mut();
            g.ensemble = self.id;
            g.ctx = Some(*ctx);
            g.values.clear();
            g.values.extend(pos.iter().copied().zip(out));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use equity::ONE;

    #[test]
    fn mean_sums_exactly_and_rounds_by_the_fixed_rule() {
        let a = MoneyChance::from_raw([ONE - 3, 1, 1, 1]).expect("valid");
        let b = MoneyChance::from_raw([0, 1, 1, ONE - 2]).expect("valid");
        let c = MoneyChance::from_raw([1, 0, ONE - 1, 0]).expect("valid");
        let m = mean(&[a, b, c]);
        assert_eq!(m.0.iter().sum::<u64>(), ONE);
        // Totals ONE-2, 2, ONE+1, ONE-1 (sum 3 ONE; 2^32 is 1 more than a multiple of 3): the
        // floors of a third leave remainders 2, 2, 2, 0 and sum to ONE-2, so the two missing
        // units go to the largest remainders, ties to the lower team: teams 0 and 1.
        assert_eq!(
            m.0,
            [(ONE - 2) / 3 + 1, 1, (ONE + 1) / 3, (ONE - 1) / 3],
            "{m:?}"
        );
        // The same values in another order: the same mean (sums of whole numbers).
        assert_eq!(mean(&[c, a, b]), m);
        // The same numbers as the weighted mix with weight 1 each.
        assert_eq!(m, MoneyChance::mix(&[(1, a), (1, b), (1, c)]));
        // The mean of one value, and of identical values, is that value.
        assert_eq!(mean(&[a]), a);
        assert_eq!(mean(&[b, b, b, b]), b);
        // Many random vectors: always exactly ONE.
        let mut sm = engine::rng::SplitMix64::new(7);
        for _ in 0..1000 {
            let vs: Vec<MoneyChance> = (0..1 + (sm.next_u64() % 5) as usize)
                .map(|_| {
                    let w: [u128; 4] =
                        core::array::from_fn(|_| u128::from(sm.next_u64() % 1_000_000));
                    if w.iter().all(|&x| x == 0) {
                        MoneyChance::even()
                    } else {
                        MoneyChance::from_weights(w)
                    }
                })
                .collect();
            let m = mean(&vs);
            assert_eq!(m.0.iter().sum::<u64>(), ONE);
            let terms: Vec<(u64, MoneyChance)> = vs.iter().map(|&v| (1, v)).collect();
            assert_eq!(m, MoneyChance::mix(&terms));
        }
    }

    fn oracle(seed: u64) -> Oracle {
        let net = crate::synthetic::random_net(seed, &[32, 16], crate::outcome::vocab(4), 100);
        let meta =
            std::collections::BTreeMap::from([("head".to_string(), "outcome-v1".to_string())]);
        let bytes = crate::oracle::write_net(&net, 4, &meta);
        Oracle::from_bytes(&bytes, std::sync::Arc::new(crate::synthetic::SyntheticLeaf))
            .expect("loads")
    }

    /// On real reached frontier positions: an ensemble of one network gives exactly the
    /// network's values (one by one and in batches); of identical networks too; of different
    /// networks, the integer mean of the members' values, the same one by one and in batches.
    #[test]
    fn ensemble_values_are_the_members_integer_mean() {
        let pos =
            crate::synthetic::reached_frontier(3, 4, 6, 400, &crate::synthetic::SyntheticLeaf);
        assert!(pos.len() > 50);
        let single = oracle(11);
        let one = Ensemble::new(vec![oracle(11)]).expect("one");
        let same =
            Ensemble::new(vec![oracle(11), oracle(11), oracle(11), oracle(11)]).expect("four");
        let mixed = Ensemble::new(vec![oracle(11), oracle(12), oracle(13)]).expect("three");
        assert_eq!(mixed.trick(), 4);
        for (c, x) in &pos {
            let v = single.eval_reference(c, x);
            assert_eq!(one.value(c, x), v);
            assert_eq!(same.value(c, x), v);
            let ms: Vec<MoneyChance> = mixed
                .members()
                .iter()
                .map(|o| o.eval_reference(c, x))
                .collect();
            let m = mixed.value(c, x);
            assert_eq!(m, mean(&ms));
            assert_eq!(m.0.iter().sum::<u64>(), ONE);
        }
        // Batches (positions of one match context at a time).
        let ctx = pos[0].1;
        let cs: Vec<CanonPos> = pos
            .iter()
            .filter(|(_, x)| *x == ctx)
            .map(|(c, _)| *c)
            .collect();
        for e in [&one, &same, &mixed] {
            let mut out = vec![MoneyChance::certain(0); cs.len()];
            e.value_batch(&cs, &ctx, &mut out);
            for (c, v) in cs.iter().zip(&out) {
                assert_eq!(*v, e.value(c, &ctx));
            }
        }
        assert_eq!(one.prefetch_depth(), single.prefetch_depth());
        // Several networks: a batch handed over first (the table of the latest batch) gives
        // the same values, for the positions in it and for others; another context is not
        // answered from it.
        assert_eq!(mixed.prefetch_depth(), ENSEMBLE_PREFETCH);
        let half = &cs[..cs.len() / 2];
        mixed.prefetch(half, &ctx);
        for c in &cs {
            let ms: Vec<MoneyChance> = mixed
                .members()
                .iter()
                .map(|o| o.eval_reference(c, &ctx))
                .collect();
            assert_eq!(mixed.value(c, &ctx), mean(&ms));
        }
        if let Some((c, x)) = pos.iter().find(|(_, x)| *x != ctx) {
            let ms: Vec<MoneyChance> = mixed
                .members()
                .iter()
                .map(|o| o.eval_reference(c, x))
                .collect();
            assert_eq!(mixed.value(c, x), mean(&ms));
        }
    }
}
