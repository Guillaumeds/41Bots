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
//! The ensemble's name in values and reports is [`Ensemble::id`]: the member's SHA-256 for one
//! network, else `ens<N>:` and the members' SHA-256s joined by `+` in the order given.

use equity::MoneyChance;
use solver::MatchCtx;
use solver::frontier::{CanonPos, Frontier};

use crate::oracle::Oracle;

/// Several oracle networks of the same frontier trick, valued as their integer mean (module
/// notes).
pub struct Ensemble {
    members: Vec<Oracle>,
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
        Ok(Ensemble { members })
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
        let vs: Vec<MoneyChance> = self.members.iter().map(|o| o.eval(c, ctx)).collect();
        mean(&vs)
    }
}

/// The integer mean of money chances: `u128` totals per team, index-ordered, rounded by largest
/// remainder (ties to the lower team) so the result sums to exactly 2^32 ([`MoneyChance::mix`]
/// with weight 1 each). The mean of one value is that value.
///
/// # Panics
/// If `vs` is empty.
pub fn mean(vs: &[MoneyChance]) -> MoneyChance {
    assert!(!vs.is_empty(), "the mean of no values");
    if vs.len() == 1 {
        return vs[0];
    }
    let terms: Vec<(u64, MoneyChance)> = vs.iter().map(|&v| (1, v)).collect();
    MoneyChance::mix(&terms)
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
        let mut vs = Vec::with_capacity(self.members.len());
        for (i, slot) in out.iter_mut().enumerate().take(n) {
            vs.clear();
            vs.extend((0..self.members.len()).map(|k| each[k * n + i]));
            *slot = mean(&vs);
        }
    }

    /// The deepest of the members' batch depths (values never depend on it).
    fn prefetch_depth(&self) -> u8 {
        self.members
            .iter()
            .map(Frontier::prefetch_depth)
            .max()
            .unwrap_or(0)
    }

    /// Every member values the positions ahead into its own memo, as one batch.
    fn prefetch(&self, pos: &[CanonPos], ctx: &MatchCtx) {
        for o in &self.members {
            o.prefetch(pos, ctx);
        }
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
    }
}
