//! Exact leaves (fix "G" of the 2026-10-19 diagnosis, docs/m3-hybrid/rung2.md "Exact leaves
//! (G)"): in the hybrid H_k (exact search to the start of trick `k`, the oracle network O_k
//! there), some frontier positions are valued by the **exact solver** instead of the network:
//! those where O_k is least reliable. An exact solve from trick 3 costs milliseconds; this spends
//! compute for quality and lowers no standard.
//!
//! **Which leaves** ([`LeafPolicy`], `hybrid-check --exact-leaves SPEC`): a union of terms joined
//! by `+`:
//! - `race:LIST`: every leaf of the race classes in `LIST` (`nn::outcome::race_class_of`: 0 no
//!   game can end this hand … 4 a team can win the match and another the game; `LIST` is classes
//!   and ranges, e.g. `4`, `2,4`, `1-4`);
//! - `band:LO-HI`: every leaf where the network gives the decider's team a money chance from
//!   `LO`% to `HI`% (both included; the bounds are turned into units of 2^-32 once, and the test
//!   is on whole numbers);
//! - `all`: every leaf (H_k is then exact search: a test, and a ceiling);
//! - `none`: no leaf (the default): the hybrid as before, byte for byte (the check does not wrap
//!   the network at all).
//!
//! Tie-decided leaves (`crate::hybrid_explain`: the exact value differs between the two tie
//! orders) are not a term: telling them apart needs two exact solves of every leaf, the cost of
//! `all`. The race classes hold almost all of them (docs/m3-hybrid/rung1.md §5).
//!
//! **The value of an exact leaf** is `crate::oracle_labels::exact_value` of the position as the
//! search first reached it: the exact solver with the documented tie rule and fixed-point money
//! chances, the very value a label of that position carries (tested). The policy is decided once
//! per distinct canonical form in a world and the value kept ([`ExactLeaves`] memoises per world,
//! as the network's frontier values are memoised), so the value depends only on the canonical
//! form and the context, as the solvers require (`solver::frontier`).
//!
//! **Determinism.** The policy reads only whole numbers (the race class from the outcome mask,
//! the network's integer money chance); each world's search is sequential, its leaf solves too,
//! and the leaf solves of one world share a table (`solver::SolveFamily`: the values of solves
//! from empty tables, fewer nodes). So every value is byte-identical for any thread count and
//! machine. The leaf solves' times are reports only.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Instant;

use equity::MoneyChance;
use serde::Serialize;
use solver::frontier::{CanonPos, Frontier};
use solver::{FastOptions, MatchCtx, Position, SolveFamily, SolveMem};

use crate::pimc::SharedLeaf;

/// The race classes (`nn::outcome::N_RACE_CLASSES`).
pub const CLASSES: usize = nn::outcome::N_RACE_CLASSES;
const ALL_CLASSES: u8 = (1 << CLASSES) - 1;

/// Which frontier positions are solved exactly (module notes). Never "none": no policy is
/// `Option::None`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafPolicy {
    /// Bit `k`: every leaf of race class `k`.
    pub classes: u8,
    /// Every leaf whose network value for the decider's team is in `lo..=hi` (units of 2^-32).
    pub band: Option<(u64, u64)>,
    /// The band's bounds as given (percentages), for the description.
    band_text: Option<(String, String)>,
    /// Transposition-table megabytes of each leaf solve (never changes a value).
    pub tt_mb: usize,
}

/// A percentage of money chance ("40", "52.5") in units of 2^-32: the decimal text parsed once
/// (correctly rounded) and rounded to a whole number, the same on every machine.
fn pct_units(t: &str) -> Result<u64, String> {
    let ok = !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit() || b == b'.');
    let x: f64 = if ok {
        t.parse().unwrap_or(f64::NAN)
    } else {
        f64::NAN
    };
    if !(x.is_finite() && (0.0..=100.0).contains(&x)) {
        return Err(format!("{t:?} is not a percentage from 0 to 100"));
    }
    Ok((x / 100.0 * equity::ONE as f64).round() as u64)
}

impl LeafPolicy {
    /// From `--exact-leaves` (module notes); `None` for `none`. `tt_mb`: each leaf solve's table.
    pub fn parse(text: &str, tt_mb: usize) -> Result<Option<LeafPolicy>, String> {
        let err = |m: String| format!("--exact-leaves {text:?}: {m}");
        let t = text.trim();
        if t == "none" {
            return Ok(None);
        }
        let mut p = LeafPolicy {
            classes: 0,
            band: None,
            band_text: None,
            tt_mb: tt_mb.max(1),
        };
        for term in t.split('+').map(str::trim) {
            if term == "all" {
                p.classes = ALL_CLASSES;
            } else if let Some(list) = term.strip_prefix("race:") {
                let mut any = false;
                for item in list.split(',').map(str::trim) {
                    let (a, b) = item.split_once('-').unwrap_or((item, item));
                    let k = |s: &str| -> Result<u8, String> {
                        s.trim()
                            .parse::<u8>()
                            .ok()
                            .filter(|&k| usize::from(k) < CLASSES)
                            .ok_or_else(|| {
                                err(format!(
                                    "race class {s:?}: a class from 0 to {}",
                                    CLASSES - 1
                                ))
                            })
                    };
                    let (a, b) = (k(a)?, k(b)?);
                    if a > b {
                        return Err(err(format!("race classes {item:?}: an empty range")));
                    }
                    for c in a..=b {
                        p.classes |= 1 << c;
                    }
                    any = true;
                }
                if !any {
                    return Err(err("race: needs classes, e.g. race:2,4".into()));
                }
            } else if let Some(b) = term.strip_prefix("band:") {
                if p.band.is_some() {
                    return Err(err("one band only".into()));
                }
                let (lo, hi) = b
                    .split_once('-')
                    .ok_or_else(|| err(format!("band {b:?}: LO-HI, e.g. band:40-60")))?;
                let (lo, hi) = (lo.trim(), hi.trim());
                let (ul, uh) = (pct_units(lo).map_err(err)?, pct_units(hi).map_err(err)?);
                if ul > uh {
                    return Err(err(format!("band {b:?}: LO above HI")));
                }
                p.band = Some((ul, uh));
                p.band_text = Some((lo.to_string(), hi.to_string()));
            } else {
                return Err(err(format!(
                    "term {term:?}: none, all, race:LIST (e.g. race:2,4 or race:1-4) or \
                     band:LO-HI, joined by +"
                )));
            }
        }
        Ok(Some(p))
    }

    /// Every leaf exact.
    pub fn all(tt_mb: usize) -> LeafPolicy {
        LeafPolicy {
            classes: ALL_CLASSES,
            band: None,
            band_text: None,
            tt_mb: tt_mb.max(1),
        }
    }

    /// The leaves of the race classes of the bit mask `classes` (bit `k`: class `k`; 0: none of
    /// them, a policy that selects nothing).
    pub fn race(classes: u8, tt_mb: usize) -> LeafPolicy {
        LeafPolicy {
            classes: classes & ALL_CLASSES,
            band: None,
            band_text: None,
            tt_mb: tt_mb.max(1),
        }
    }

    /// The policy in its normal form (`all`, `race:2,4`, `race:4+band:40-60`, …): the same for
    /// every way of writing the same policy, for reports and the values' fingerprint.
    pub fn describe(&self) -> String {
        if self.classes == ALL_CLASSES {
            return "all".into();
        }
        let mut terms = Vec::new();
        if self.classes != 0 {
            let cs: Vec<String> = (0..CLASSES)
                .filter(|&k| self.classes >> k & 1 == 1)
                .map(|k| k.to_string())
                .collect();
            terms.push(format!("race:{}", cs.join(",")));
        }
        if let Some((lo, hi)) = &self.band_text {
            terms.push(format!("band:{lo}-{hi}"));
        }
        if terms.is_empty() {
            "nothing".into()
        } else {
            terms.join("+")
        }
    }

    /// Whether race class `c` is solved exactly whatever the network says.
    pub fn has_class(&self, c: u8) -> bool {
        usize::from(c) < CLASSES && self.classes >> c & 1 == 1
    }
}

/// What the exact leaves of a search (or of many, summed) cost and covered: distinct frontier
/// positions per race class and how many of them were solved exactly, with the leaf solves'
/// nodes and time.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct LeafStats {
    /// Distinct frontier positions reached, per race class (summed over searches).
    pub distinct: [u64; CLASSES],
    /// Of those, solved exactly.
    pub exact: [u64; CLASSES],
    /// Solver nodes of the leaf solves.
    pub nodes: u64,
    /// Time of the leaf solves (µs; a report only, not compared by the tests).
    pub us: u64,
}

impl LeafStats {
    /// Adds another's counts.
    pub fn add(&mut self, o: &LeafStats) {
        for k in 0..CLASSES {
            self.distinct[k] += o.distinct[k];
            self.exact[k] += o.exact[k];
        }
        self.nodes += o.nodes;
        self.us += o.us;
    }

    /// The counts without the time (what is the same on every machine and thread count).
    pub fn counts(&self) -> ([u64; CLASSES], [u64; CLASSES], u64) {
        (self.distinct, self.exact, self.nodes)
    }
}

thread_local! {
    /// Each thread's memory for leaf solves, kept from world to world (`SolveMem`: reuse never
    /// changes a value), apart from the memory of the hybrid's own search.
    static LEAF_MEM: RefCell<Option<SolveMem>> = const { RefCell::new(None) };
}

struct LeafState {
    mem: Option<SolveMem>,
    /// Per canonical form: the position as first reached, its value and whether it was solved
    /// exactly.
    memo: BTreeMap<CanonPos, (Position, MatchCtx, MoneyChance, bool)>,
    stats: LeafStats,
}

/// A frontier evaluator: the policy's leaves solved exactly, the others valued by `inner` (the
/// network). One per search (one world of one decision), as the noise of the curse test.
pub struct ExactLeaves<'a> {
    inner: &'a dyn Frontier,
    leaf: &'a SharedLeaf,
    policy: &'a LeafPolicy,
    opts: FastOptions,
    team: u8,
    state: Mutex<LeafState>,
}

impl<'a> ExactLeaves<'a> {
    /// The exact leaves of one search around `inner`: `leaf` the equity table, `team` the
    /// decider's team (for a band), `family` the world's positions (`SolveFamily::of_match` of
    /// the world: the leaf solves share one table; `None`: each from empty tables).
    pub fn new(
        inner: &'a dyn Frontier,
        leaf: &'a SharedLeaf,
        policy: &'a LeafPolicy,
        team: u8,
        family: Option<SolveFamily>,
    ) -> Self {
        let mut mem = LEAF_MEM.with(|m| m.borrow_mut().take()).unwrap_or_default();
        match family {
            Some(f) => mem.share(f),
            None => mem.unshare(),
        }
        ExactLeaves {
            inner,
            leaf,
            policy,
            opts: FastOptions {
                tt_mb: policy.tt_mb,
                ..FastOptions::default()
            },
            team,
            state: Mutex::new(LeafState {
                mem: Some(mem),
                memo: BTreeMap::new(),
                stats: LeafStats::default(),
            }),
        }
    }

    /// What this search's leaves cost and covered.
    pub fn stats(&self) -> LeafStats {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stats
            .clone()
    }

    /// Every distinct leaf valued, by canonical form: the position as the search first reached
    /// it, the match around it, its value and whether it was solved exactly.
    pub fn leaves(&self) -> Vec<(Position, MatchCtx, CanonPos, MoneyChance, bool)> {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .memo
            .iter()
            .map(|(c, &(p, x, v, e))| (p, x, *c, v, e))
            .collect()
    }
}

impl Drop for ExactLeaves<'_> {
    /// The leaf memory goes back to the thread for its next search.
    fn drop(&mut self) {
        let st = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        if let Some(mut mem) = st.mem.take() {
            mem.unshare();
            LEAF_MEM.with(|m| *m.borrow_mut() = Some(mem));
        }
    }
}

impl Frontier for ExactLeaves<'_> {
    /// The value of a leaf already reached (the solver asks through `value_at`).
    fn value(&self, canon: &CanonPos, _ctx: &MatchCtx) -> MoneyChance {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .memo
            .get(canon)
            .expect("exact leaves value the positions the solver gives them (value_at)")
            .2
    }

    fn value_at(&self, pos: &Position, canon: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let st = &mut *guard;
        if let Some(&(_, _, v, _)) = st.memo.get(canon) {
            return v;
        }
        let class = nn::outcome::race_class_of(canon, ctx, self.leaf);
        let mut exact = self.policy.has_class(class);
        let mut net = None;
        if !exact && let Some((lo, hi)) = self.policy.band {
            let v = self.inner.value_at(pos, canon, ctx);
            exact = (lo..=hi).contains(&v.get(self.team));
            net = Some(v);
        }
        let k = usize::from(class);
        st.stats.distinct[k] += 1;
        let v = if exact {
            let mem = st.mem.get_or_insert_with(SolveMem::new);
            let t = Instant::now();
            let (v, nodes) =
                crate::oracle_labels::exact_value(pos, *ctx, self.leaf, self.opts, mem);
            st.stats.us += t.elapsed().as_micros() as u64;
            st.stats.nodes += nodes;
            st.stats.exact[k] += 1;
            v
        } else {
            net.unwrap_or_else(|| self.inner.value_at(pos, canon, ctx))
        };
        st.memo.insert(*canon, (*pos, *ctx, v, exact));
        v
    }

    fn prefetch_depth(&self) -> u8 {
        self.inner.prefetch_depth()
    }

    /// Passed on to the network (it values the batch into its memo; a leaf solved exactly is
    /// never asked of it, so its value there is unused). Nothing is recorded.
    fn prefetch(&self, pos: &[CanonPos], ctx: &MatchCtx) {
        self.inner.prefetch(pos, ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policies_parse_normalise_and_refuse() {
        assert_eq!(LeafPolicy::parse("none", 8), Ok(None));
        let p = |s: &str| LeafPolicy::parse(s, 8).expect("valid").expect("some");
        assert_eq!(p("race:4").classes, 0b10000);
        assert_eq!(p("race:4,2").describe(), "race:2,4");
        assert_eq!(p("race:2, 4").classes, 0b10100);
        assert_eq!(p("race:1-4").classes, 0b11110);
        assert_eq!(p("race:1-4").describe(), "race:1,2,3,4");
        assert_eq!(p("race:0-4").describe(), "all");
        assert_eq!(p("all").describe(), "all");
        assert_eq!(p("all"), LeafPolicy::all(8));
        assert_eq!(p("race:2,4"), LeafPolicy::race(0b10100, 8));
        let b = p("race:4+band:40-60");
        assert_eq!(b.describe(), "race:4+band:40-60");
        let (lo, hi) = b.band.expect("band");
        assert_eq!(lo, (0.4 * equity::ONE as f64).round() as u64);
        assert_eq!(hi, (0.6 * equity::ONE as f64).round() as u64);
        assert_eq!(p("band:0-100").band, Some((0, equity::ONE)));
        assert!(p("race:4").has_class(4) && !p("race:4").has_class(3));
        for bad in [
            "",
            "race:",
            "race:5",
            "race:3-1",
            "race:x",
            "band:60-40",
            "band:1-101",
            "band:50",
            "ties",
            "race:4+band:1-2+band:3-4",
            "nones",
        ] {
            assert!(LeafPolicy::parse(bad, 8).is_err(), "{bad:?}");
        }
        assert_eq!(LeafPolicy::race(0, 8).describe(), "nothing");
    }
}
