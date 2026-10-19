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
//! - `learned:MODEL:t=T` (or `t=inf`, or `budget=S`): every leaf whose predicted squared error
//!   of the network (`nn::leafrisk`, a learned leaf-risk model: gradient-boosted integer trees
//!   over the network's own inputs and the extras of its answer) is at least `T` (units of 2^-40
//!   of a squared money chance), the threshold chosen with a statistical guarantee on the error
//!   left (Learn then Test; docs/m3-hybrid/learned-leaves.md). `budget=S` takes the threshold
//!   that picks a share `S` of the model's test rows (its budget table). The score is the value
//!   of computing the leaf exactly with the decision's sensitivity taken as constant (Russell &
//!   Wefald 1991: an error `e` flips a comparison whose gap density near 0 is `f(0)` with expected
//!   loss about `f(0) e^2 / 2`); a position's score depends only on the position and the
//!   network's value there, so a leaf's value stays a function of its canonical form;
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
//! the network's integer money chance, the learned model's integer score); each world's search is sequential, its leaf solves too,
//! and the leaf solves of one world share a table (`solver::SolveFamily`: the values of solves
//! from empty tables, fewer nodes). So every value is byte-identical for any thread count and
//! machine. The leaf solves' times are reports only.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use equity::MoneyChance;
use nn::leafrisk::RiskModel;
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
    /// The learned term (`learned:MODEL:t=T` or `:budget=S`): every leaf whose predicted
    /// squared error of the network is at least the threshold.
    pub learned: Option<Learned>,
    /// Transposition-table megabytes of each leaf solve (never changes a value).
    pub tt_mb: usize,
}

/// The learned term of a policy (docs/m3-hybrid/rung2.md "Learned exact leaves"): the leaf-risk
/// model (`nn::leafrisk`) and the threshold on its score (units of 2^-40 of a squared money
/// chance; `None`: infinite, no leaf).
#[derive(Clone, Debug)]
pub struct Learned {
    /// The model.
    pub model: Arc<RiskModel>,
    /// A leaf is solved exactly when its score is at least this (`None`: never).
    pub t: Option<i64>,
    /// The budget it came from (`budget=S`), for the report.
    pub budget: Option<String>,
}

impl PartialEq for Learned {
    fn eq(&self, o: &Learned) -> bool {
        self.model.sha256 == o.model.sha256 && self.t == o.t
    }
}
impl Eq for Learned {}

impl Learned {
    /// Whether a leaf of this score is solved exactly.
    pub fn picks(&self, score: i64) -> bool {
        self.t.is_some_and(|t| score >= t)
    }

    /// Normal form: `learned:<first 12 hex of the model's SHA-256>:t=<T or inf>`.
    pub fn describe(&self) -> String {
        let t = self.t.map_or("inf".to_string(), |t| t.to_string());
        format!("learned:{}:t={t}", &self.model.sha256[..12])
    }
}

/// The parts of a learned term `learned:MODEL:t=T` / `learned:MODEL:budget=S` (the model's
/// path or manifest input name, and the threshold text), checked without reading the model:
/// `T` a whole number at least 0 or `inf`, `S` a share from 0 to 1. The model is the text up to
/// the last `:`.
pub fn learned_parts(term: &str) -> Result<(&str, LearnedThreshold), String> {
    let rest = term
        .strip_prefix("learned:")
        .ok_or_else(|| format!("{term:?} is not a learned term"))?;
    let (model, thr) = rest.rsplit_once(':').ok_or_else(|| {
        format!("{term:?}: learned:MODEL:t=T or learned:MODEL:budget=S (S a share 0-1)")
    })?;
    if model.is_empty() {
        return Err(format!("{term:?}: no model"));
    }
    let thr = if let Some(t) = thr.strip_prefix("t=") {
        if t == "inf" {
            LearnedThreshold::Never
        } else {
            LearnedThreshold::At(
                t.parse::<i64>()
                    .ok()
                    .filter(|&t| t >= 0)
                    .ok_or_else(|| format!("{term:?}: t={t} is not a whole number >= 0 or inf"))?,
            )
        }
    } else if let Some(b) = thr.strip_prefix("budget=") {
        let ok = !b.is_empty() && b.bytes().all(|c| c.is_ascii_digit() || c == b'.');
        let x: f64 = if ok {
            b.parse().unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };
        if !(x.is_finite() && (0.0..=1.0).contains(&x)) {
            return Err(format!("{term:?}: budget={b} is not a share from 0 to 1"));
        }
        LearnedThreshold::Budget(b.to_string(), x)
    } else {
        return Err(format!("{term:?}: the threshold is t=T or budget=S"));
    };
    Ok((model, thr))
}

/// A learned term's threshold as written.
#[derive(Clone, Debug, PartialEq)]
pub enum LearnedThreshold {
    /// `t=inf`.
    Never,
    /// `t=T`.
    At(i64),
    /// `budget=S` (as written, and its value).
    Budget(String, f64),
}

/// Checks a policy's text without reading a learned term's model (the runner, before the inputs
/// are fetched): every term as [`LeafPolicy::parse`] reads it. Returns the learned terms' model
/// names.
pub fn check_spec(text: &str) -> Result<Vec<String>, String> {
    let mut models = Vec::new();
    let mut plain = Vec::new();
    for term in text.trim().split('+').map(str::trim) {
        if term.starts_with("learned:") {
            let (m, _) =
                learned_parts(term).map_err(|e| format!("--exact-leaves {text:?}: {e}"))?;
            models.push(m.to_string());
        } else {
            plain.push(term);
        }
    }
    if models.len() > 1 {
        return Err(format!("--exact-leaves {text:?}: one learned term only"));
    }
    if !plain.is_empty() {
        LeafPolicy::parse(&plain.join("+"), 1)?;
    } else if models.is_empty() {
        return Err(format!("--exact-leaves {text:?}: empty"));
    }
    Ok(models)
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
            learned: None,
            tt_mb: tt_mb.max(1),
        };
        for term in t.split('+').map(str::trim) {
            if term.starts_with("learned:") {
                if p.learned.is_some() {
                    return Err(err("one learned term only".into()));
                }
                let (path, thr) = learned_parts(term).map_err(err)?;
                let model = Arc::new(RiskModel::load(Path::new(path)).map_err(err)?);
                let (t, budget) = match thr {
                    LearnedThreshold::Never => (None, None),
                    LearnedThreshold::At(t) => (Some(t), None),
                    LearnedThreshold::Budget(text, x) => {
                        let k = model.budget_steps();
                        if k == 0 {
                            return Err(err("the model has no budget table".into()));
                        }
                        let step = (x * k as f64).round() as usize;
                        (model.threshold_for_share(step), Some(text))
                    }
                };
                p.learned = Some(Learned { model, t, budget });
            } else if term == "all" {
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
                    "term {term:?}: none, all, race:LIST (e.g. race:2,4 or race:1-4), \
                     band:LO-HI or learned:MODEL:t=T (or :budget=S), joined by +"
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
            learned: None,
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
            learned: None,
            tt_mb: tt_mb.max(1),
        }
    }

    /// The leaves whose predicted squared error under `model` is at least `t` (`None`: none).
    pub fn learned(model: Arc<RiskModel>, t: Option<i64>, tt_mb: usize) -> LeafPolicy {
        LeafPolicy {
            classes: 0,
            band: None,
            band_text: None,
            learned: Some(Learned {
                model,
                t,
                budget: None,
            }),
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
        if let Some(l) = &self.learned {
            terms.push(l.describe());
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
    /// Leaves the learned term scored.
    pub scored: u64,
    /// Time of the learned term's scoring (µs; a report only).
    pub score_us: u64,
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
        self.scored += o.scored;
        self.score_us += o.score_us;
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
        if !exact
            && let Some(l) = &self.policy.learned
            && l.t.is_some()
        {
            let v = net.unwrap_or_else(|| self.inner.value_at(pos, canon, ctx));
            let t = Instant::now();
            exact = l.picks(l.model.score(canon, ctx, self.leaf, &v));
            st.stats.score_us += t.elapsed().as_micros() as u64;
            st.stats.scored += 1;
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

    /// Learned terms: the threshold forms, checked without the model; a model file read with
    /// `t=`, `t=inf` and `budget=` (from its table), its normal form; refusals.
    #[test]
    fn learned_terms_parse() {
        use LearnedThreshold::*;
        assert_eq!(learned_parts("learned:m.json:t=12"), Ok(("m.json", At(12))));
        assert_eq!(
            learned_parts("learned:C:/x/m.json:t=inf"),
            Ok(("C:/x/m.json", Never))
        );
        assert_eq!(
            learned_parts("learned:o3-risk:budget=0.25"),
            Ok(("o3-risk", Budget("0.25".into(), 0.25)))
        );
        for bad in [
            "learned:m.json",
            "learned::t=1",
            "learned:m.json:t=-1",
            "learned:m.json:t=1.5",
            "learned:m.json:budget=1.5",
            "learned:m.json:budget=x",
            "learned:m.json:q=1",
        ] {
            assert!(learned_parts(bad).is_err(), "{bad:?}");
        }
        assert_eq!(
            check_spec("race:4+learned:risk:t=9"),
            Ok(vec!["risk".to_string()])
        );
        assert_eq!(
            check_spec("learned:risk:budget=0.5"),
            Ok(vec!["risk".to_string()])
        );
        assert_eq!(check_spec("race:2,4"), Ok(vec![]));
        assert!(check_spec("learned:a:t=1+learned:b:t=2").is_err());
        assert!(check_spec("race:9+learned:a:t=1").is_err());
        // A model file.
        let dir = std::env::temp_dir().join(format!("leafrisk-parse-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("m.json");
        let v = serde_json::json!({
            "format": nn::leafrisk::FORMAT, "encoder": nn::encode::ENCODER,
            "n_inputs": nn::leafrisk::N_INPUTS, "extras": nn::leafrisk::EXTRA_NAMES,
            "net_sha256": "x", "trick": 3, "base": 7,
            "trees": [{"f": [-1], "t": [0], "l": [0], "r": [0], "v": [0]}],
            "budget": {"quantiles": [90, 80, 70, 60]},
        });
        std::fs::write(&path, v.to_string()).expect("write");
        let at = |thr: &str| {
            LeafPolicy::parse(&format!("learned:{}:{thr}", path.display()), 8)
                .expect("valid")
                .expect("some")
        };
        let p = at("t=70");
        let l = p.learned.clone().expect("learned");
        assert_eq!((l.t, l.picks(70), l.picks(69)), (Some(70), true, false));
        assert_eq!(
            p.describe(),
            format!("learned:{}:t=70", &l.model.sha256[..12])
        );
        assert_eq!(at("t=inf").learned.expect("l").t, None);
        assert!(!at("t=inf").learned.expect("l").picks(i64::MAX));
        // budget=S: the table's entry round(S * steps).
        assert_eq!(at("budget=0.5").learned.expect("l").t, Some(80));
        assert_eq!(at("budget=1").learned.expect("l").t, Some(60));
        assert_eq!(at("budget=0").learned.expect("l").t, None);
        assert_eq!(at("budget=0.5"), at("t=80"));
        let both = LeafPolicy::parse(&format!("race:4+learned:{}:t=1", path.display()), 8)
            .expect("valid")
            .expect("some");
        assert!(both.has_class(4) && both.learned.is_some());
        assert!(LeafPolicy::parse("learned:/no/such/file.json:t=1", 8).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
