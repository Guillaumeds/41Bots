//! The search side of the audit loop (docs/m3-hybrid/audit-loop.md §3 items 2–3, §6 B2): which
//! of the frontier positions a decision's search reaches lie in the groups of a predicate file
//! (`nn::predicate`), for the golden slices of `search hybrid-check --slices` and the mini golden
//! sets of `search decisions --keep`.
//!
//! **Reach** ([`Reach`]): for one search (one world of a decision: `crate::pimc::exact_world`
//! with a frontier at the start of trick `cut`), the distinct canonical forms the solver reaches
//! at the frontier, and how many of them are in each group; for a decision, the sums over its
//! worlds. A decision is **in group `g`** when `in_group[g] / reached ≥ share` (compared in whole
//! numbers: `in_group[g] × 1,000,000 ≥ share_millionths × reached`). The recording never changes
//! a value: [`GroupRecorder`] returns its inner evaluator's value for every position.
//!
//! **Mini golden sets** ([`keep_decisions`], spec §3 item 3: "keep drawing decisions, keep only
//! those whose search reaches the group"): candidates are decisions `first, first + 1, …` of the
//! yardstick's generator (`crate::measure::build_decision_at`: the same decision `i` as
//! `search decisions` with the same seed), each searched on its worlds `0..worlds` as the
//! yardstick draws them (decision seed `measure_seed(seed, label, id)`), with the network at the
//! frontier (the hybrid H_cut) or exact values there (H_cut with a perfect network: the right
//! choice at trick 4, where an exact solve is cheap); a candidate is kept when it is in at least
//! one group. Candidates run on any number of threads and are taken in order, so the kept set is
//! byte-identical for any thread count. Each world's search is sequential with a fixed table
//! (`tt_mb`), so the positions it reaches never depend on the machine.

use std::collections::BTreeSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use engine::game::HandStartState;
use equity::MoneyChance;
use nn::predicate::{Evaluator, Predicates, at_share};
use serde::{Deserialize, Serialize};
use solver::frontier::{CanonPos, Frontier};
use solver::{FastOptions, FrontierCut, Leaf, MatchCtx, Position, SolveMem};

use crate::measure::{DecisionSpec, build_decision_at, measure_seed};
use crate::oracle_labels::{Driver, Recorder, tt_mb_for};
use crate::ordered::{ordered, window_for};
use crate::pimc::{Ctx, SearchConfig, WorldSolve, candidates, exact_world, world_match};

/// Reached frontier positions of searches, and how many of them lie in each group.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reach {
    /// Distinct reached positions (per search, summed over searches).
    pub reached: u64,
    /// Of them, in each group of the predicate file (file order).
    pub in_group: Vec<u64>,
}

impl Reach {
    /// Nothing reached yet, for `groups` groups.
    pub fn new(groups: usize) -> Reach {
        Reach {
            reached: 0,
            in_group: vec![0; groups],
        }
    }

    /// Add another search's reach.
    pub fn add(&mut self, o: &Reach) {
        self.reached += o.reached;
        if self.in_group.len() < o.in_group.len() {
            self.in_group.resize(o.in_group.len(), 0);
        }
        for (a, b) in self.in_group.iter_mut().zip(&o.in_group) {
            *a += b;
        }
    }

    /// The groups holding at least the share `micro` (millionths) of the reached positions,
    /// ascending.
    pub fn groups_at(&self, micro: u64) -> Vec<u32> {
        (0..self.in_group.len())
            .filter(|&g| at_share(self.in_group[g], self.reached, micro))
            .map(|g| g as u32)
            .collect()
    }

    /// Group `g`'s share of the reached positions (reports only).
    pub fn share(&self, g: usize) -> f64 {
        self.in_group[g] as f64 / self.reached.max(1) as f64
    }
}

struct RecGroups<'a> {
    seen: BTreeSet<CanonPos>,
    ev: Evaluator<'a>,
    reach: Reach,
}

/// A frontier evaluator that values every position exactly as `inner` does and records the
/// distinct canonical forms the search reaches there, with their groups (decided on each
/// position's encoded features, `nn::predicate`). One per search: a search is sequential, so the
/// lock is never contended.
pub struct GroupRecorder<'a> {
    inner: &'a dyn Frontier,
    state: Mutex<RecGroups<'a>>,
}

impl<'a> GroupRecorder<'a> {
    /// A recorder of `preds`' groups around `inner`; `leaf` gives the dense features.
    pub fn new(
        inner: &'a dyn Frontier,
        preds: &'a Predicates,
        leaf: &'a (dyn Leaf + Sync),
    ) -> Self {
        GroupRecorder {
            inner,
            state: Mutex::new(RecGroups {
                seen: BTreeSet::new(),
                ev: Evaluator::new(preds, leaf),
                reach: Reach::new(preds.len()),
            }),
        }
    }

    /// What the search reached.
    pub fn into_reach(self) -> Reach {
        self.state
            .into_inner()
            .unwrap_or_else(|e| e.into_inner())
            .reach
    }

    fn note(&self, canon: &CanonPos, ctx: &MatchCtx) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if st.seen.insert(*canon) {
            let groups = st.ev.members(canon, ctx);
            st.reach.reached += 1;
            for g in groups {
                st.reach.in_group[g as usize] += 1;
            }
        }
    }
}

impl Frontier for GroupRecorder<'_> {
    fn value(&self, canon: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        self.note(canon, ctx);
        self.inner.value(canon, ctx)
    }

    fn value_at(&self, pos: &Position, canon: &CanonPos, ctx: &MatchCtx) -> MoneyChance {
        self.note(canon, ctx);
        self.inner.value_at(pos, canon, ctx)
    }

    fn prefetch_depth(&self) -> u8 {
        self.inner.prefetch_depth()
    }

    /// Passed on; nothing is recorded (the search records what it reaches through `value_at`).
    fn prefetch(&self, pos: &[CanonPos], ctx: &MatchCtx) {
        self.inner.prefetch(pos, ctx);
    }
}

/// The reach of a decision's searches on its worlds `0..worlds` (drawn as the yardstick draws
/// them, decision seed `dseed`): the hybrid H_cut with `net` at the frontier, or with exact
/// values there (`None`); sequential searches with `tt_mb`-megabyte tables.
#[allow(clippy::too_many_arguments)]
pub fn decision_reach(
    d: &DecisionSpec,
    preds: &Predicates,
    ctx: &Ctx,
    cut: u8,
    tt_mb: usize,
    dseed: u64,
    worlds: u64,
    net: Option<&dyn Frontier>,
    mem: &mut SolveMem,
) -> Reach {
    assert!(
        d.trick >= 1 && d.trick < cut,
        "decision {} is at trick {}: the hybrid with a frontier at trick {cut} is defined for \
         decisions at tricks 1..{cut}",
        d.id,
        d.trick
    );
    let (_, view, legal) = d.replay();
    let (_, cands) = candidates(&view, &legal);
    let ws = ctx.source.prepare(&view, ctx.rules);
    let cfg = SearchConfig {
        tt_mb,
        ..SearchConfig::default()
    };
    let opts = FastOptions {
        tt_mb: tt_mb_for(cut),
        ..FastOptions::default()
    };
    let mut total = Reach::new(preds.len());
    for w in 0..worlds {
        let m = world_match(&view, &*ws, w, dseed, ctx.rules);
        let exact;
        let inner: &dyn Frontier = match net {
            Some(n) => n,
            None => {
                exact = Recorder::new(ctx.leaf, opts, Driver::Exact);
                &exact
            }
        };
        let rec = GroupRecorder::new(inner, preds, ctx.leaf);
        exact_world(
            &m,
            &cands,
            ctx,
            &cfg,
            None,
            mem,
            &mut WorldSolve::default(),
            Some(FrontierCut::new(cut, &rec)),
        )
        .expect("nothing stops this search");
        total.add(&rec.into_reach());
    }
    total
}

/// One rejection-sampling run (`search decisions --keep`).
#[derive(Clone, Copy, Debug)]
pub struct KeepJob {
    /// The decisions' trick.
    pub trick: u8,
    /// The frontier trick (after `trick`).
    pub cut: u8,
    /// The decisions' seed, and the worlds' (the yardstick's: 1).
    pub seed: u64,
    /// The worlds' label (the yardstick's: 1).
    pub label: u64,
    /// Worlds searched per candidate (`0..worlds`).
    pub worlds: u64,
    /// A candidate is kept when a group holds at least this share (millionths) of its reached
    /// positions.
    pub share: u64,
    /// Decisions to keep.
    pub count: u64,
    /// The first candidate (decision index of the generator).
    pub first: u64,
    /// At most this many candidates are drawn.
    pub max_candidates: u64,
    /// Worker threads (never changes the result).
    pub threads: usize,
    /// Table megabytes of each search.
    pub tt_mb: usize,
}

/// A kept decision.
#[derive(Clone, Debug)]
pub struct KeptDecision {
    /// The decision (its id is its index in the generator).
    pub d: DecisionSpec,
    /// Its searches' reach over the worlds searched.
    pub reach: Reach,
    /// The groups it is in (share at least the job's).
    pub groups: Vec<u32>,
}

/// The result of [`keep_decisions`].
#[derive(Clone, Debug)]
pub struct KeepRun {
    /// The kept decisions, in generator order.
    pub kept: Vec<KeptDecision>,
    /// Candidates drawn (`first .. first + drawn`).
    pub drawn: u64,
    /// Of the candidates drawn, those in each group.
    pub drawn_in_group: Vec<u64>,
    /// The candidates' reach, summed.
    pub reach: Reach,
    /// Time in the searches, summed over threads (reports only).
    pub search_time: Duration,
}

/// Rejection sampling of decisions (module notes): candidates `job.first ..` until `job.count`
/// are kept or `job.max_candidates` are drawn. `net`: the network at the frontier (`None`:
/// exact values). `progress` counts candidates searched. Byte-identical for any thread count.
pub fn keep_decisions(
    job: &KeepJob,
    preds: &Predicates,
    starts: &[Vec<HandStartState>; 4],
    ctx: &Ctx,
    net: Option<&dyn Frontier>,
    progress: Option<&AtomicU64>,
) -> KeepRun {
    assert!(
        job.trick >= 1 && job.trick < job.cut && job.cut <= 5,
        "decisions at tricks 1..cut, cut at most 5"
    );
    assert!(job.worlds >= 1, "at least one world per candidate");
    let mut run = KeepRun {
        kept: Vec::new(),
        drawn: 0,
        drawn_in_group: vec![0; preds.len()],
        reach: Reach::new(preds.len()),
        search_time: Duration::ZERO,
    };
    if job.count == 0 || job.max_candidates == 0 {
        return run;
    }
    ordered(
        job.threads,
        Some(job.max_candidates),
        window_for(job.threads),
        SolveMem::new,
        |mem, j| {
            let t = Instant::now();
            let d = build_decision_at(job.trick, job.first + j, job.seed, starts);
            let dseed = measure_seed(job.seed, job.label, d.id);
            let reach = decision_reach(
                &d, preds, ctx, job.cut, job.tt_mb, dseed, job.worlds, net, mem,
            );
            if let Some(p) = progress {
                p.fetch_add(1, Ordering::Relaxed);
            }
            let groups = reach.groups_at(job.share);
            (KeptDecision { d, reach, groups }, t.elapsed())
        },
        |j, (k, dt)| {
            run.drawn = j + 1;
            run.search_time += dt;
            run.reach.add(&k.reach);
            for &g in &k.groups {
                run.drawn_in_group[g as usize] += 1;
            }
            if !k.groups.is_empty() {
                run.kept.push(k);
            }
            Ok((run.kept.len() as u64) < job.count)
        },
    )
    .expect("collecting decisions cannot fail");
    run
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use nn::synthetic::SyntheticLeaf;

    use crate::measure::{build_decisions, collect_starts};
    use crate::pimc::SharedLeaf;
    use crate::worlds::BeliefSource;

    fn preds() -> Predicates {
        let text = format!(
            r#"[{{"name": "next team no trick", "conditions": [{{"feature": {}, "op": "==", "value": 1}}]}},
                {{"name": "leader team ahead", "conditions": [{{"feature": {}, "op": ">", "value": {}}}]}}]"#,
            nn::encode::OFF_TRICKS + 5,
            nn::encode::N_BINARY + nn::encode::DENSE_NOW,
            -983_597
        );
        Predicates::parse(text.as_bytes()).expect("parses")
    }

    /// A decision's reach is the distinct frontier positions of each world's search, with exact
    /// values or a network's (the recorder never changes a value), and its groups by share.
    #[test]
    fn reach_counts_distinct_positions_and_groups() {
        let starts = collect_starts(300, 3);
        let l: Arc<SharedLeaf> = Arc::new(SyntheticLeaf);
        let src = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: engine::game::Rules::HOUSE,
        };
        let p = preds();
        let ds = build_decisions(3, 8, 5, &starts);
        let mut mem = SolveMem::new();
        let mut some_group = 0;
        for d in &ds {
            let dseed = measure_seed(5, 1, d.id);
            let r = decision_reach(d, &p, &ctx, 4, 4, dseed, 3, None, &mut mem);
            // The same positions, recorded by the harvest's own recorder, world by world.
            let (_, view, legal) = d.replay();
            let (_, cands) = candidates(&view, &legal);
            let ws = ctx.source.prepare(&view, ctx.rules);
            let mut want = Reach::new(p.len());
            for w in 0..3 {
                let m = world_match(&view, &*ws, w, dseed, ctx.rules);
                let rec = Recorder::new(&*l, FastOptions::default(), Driver::Exact);
                let cfg = SearchConfig {
                    tt_mb: 4,
                    ..SearchConfig::default()
                };
                let v = exact_world(
                    &m,
                    &cands,
                    &ctx,
                    &cfg,
                    None,
                    &mut SolveMem::new(),
                    &mut WorldSolve::default(),
                    Some(FrontierCut::new(4, &rec)),
                )
                .expect("solved");
                let exact = exact_world(
                    &m,
                    &cands,
                    &ctx,
                    &cfg,
                    None,
                    &mut SolveMem::new(),
                    &mut WorldSolve::default(),
                    None,
                )
                .expect("solved");
                assert_eq!(
                    v, exact,
                    "exact values at the frontier are the exact search"
                );
                let (list, _) = rec.into_reached();
                want.reached += list.len() as u64;
                for x in &list {
                    for g in p.members_of(&x.canon, &x.ctx, &*l) {
                        want.in_group[g as usize] += 1;
                    }
                }
            }
            assert_eq!(r, want, "decision {}", d.id);
            assert!(r.reached > 0);
            let half = r.groups_at(500_000);
            for g in 0..p.len() {
                assert_eq!(
                    half.contains(&(g as u32)),
                    2 * r.in_group[g] >= r.reached,
                    "share of group {g}"
                );
            }
            some_group += usize::from(!half.is_empty());
        }
        assert!(some_group > 0, "some decision is in a group");
    }

    /// Rejection sampling keeps exactly the candidates in a group, in generator order, the same
    /// decisions `search decisions` makes with that seed, and the same on 1 and 3 threads.
    #[test]
    fn kept_decisions_are_the_generators_and_thread_independent() {
        let starts = collect_starts(300, 3);
        let l: Arc<SharedLeaf> = Arc::new(SyntheticLeaf);
        let src = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: engine::game::Rules::HOUSE,
        };
        let p = preds();
        let job = |threads| KeepJob {
            trick: 3,
            cut: 4,
            seed: 5,
            label: 1,
            worlds: 2,
            share: 800_000,
            count: 3,
            first: 4,
            max_candidates: 40,
            threads,
            tt_mb: 4,
        };
        let a = keep_decisions(&job(1), &p, &starts, &ctx, None, None);
        let b = keep_decisions(&job(3), &p, &starts, &ctx, None, None);
        assert_eq!(a.kept.len(), 3, "{} drawn", a.drawn);
        let key = |r: &KeepRun| {
            (
                r.drawn,
                r.drawn_in_group.clone(),
                r.reach.clone(),
                r.kept
                    .iter()
                    .map(|k| (k.d.clone(), k.reach.clone(), k.groups.clone()))
                    .collect::<Vec<_>>(),
            )
        };
        assert_eq!(key(&a), key(&b));
        let all = build_decisions(3, 4 + a.drawn, 5, &starts);
        let mut mem = SolveMem::new();
        let mut kept = Vec::new();
        for d in &all[4..] {
            let r = decision_reach(
                d,
                &p,
                &ctx,
                4,
                4,
                measure_seed(5, 1, d.id),
                2,
                None,
                &mut mem,
            );
            if !r.groups_at(800_000).is_empty() {
                kept.push(d.clone());
            }
        }
        assert_eq!(
            a.kept.iter().map(|k| k.d.clone()).collect::<Vec<_>>(),
            kept,
            "the generator's decisions, those in a group"
        );
        assert!(
            a.drawn > 3,
            "some candidates were rejected ({} drawn)",
            a.drawn
        );
        // Too few candidates: what was found.
        let short = keep_decisions(
            &KeepJob {
                max_candidates: 2,
                ..job(2)
            },
            &p,
            &starts,
            &ctx,
            None,
            None,
        );
        assert_eq!(short.drawn, 2);
        assert!(short.kept.len() <= 2);
    }
}
