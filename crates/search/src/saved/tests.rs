//! Tests of the saved-worlds reader and of the hybrid check on saved worlds.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use equity::{MatchState, MoneyChance, ONE};
use solver::frontier::CanonPos;
use solver::{FnFrontier, FrontierCut, MatchCtx, SolveMem};

use super::*;
use crate::hybrid::{CheckJob, CheckRun, check_worlds, report, run_check};
use crate::measure::{
    CkptRow, DecisionSpec, WorldRows, build_decisions, collect_starts, decision_fingerprint,
    read_jsonl, world_rows,
};
use crate::pimc::{
    Ctx, Method, SearchConfig, SharedLeaf, WorldSolve, candidates, exact_world, world_match,
    world_robber,
};
use crate::worlds::{BeliefSource, WorldSource};

fn leaf() -> Arc<SharedLeaf> {
    Arc::new(solver::FnLeaf(|s: &MatchState| {
        MoneyChance::from_weights(core::array::from_fn(|t| {
            1 + u128::from(s.scores[t]) + 40 * u128::from(s.games[t])
        }))
    }))
}

fn even(_: &CanonPos, _: &MatchCtx) -> MoneyChance {
    MoneyChance::even()
}

/// A cheap frontier that depends on the scores, the tricks and who holds which live card.
fn by_cards(p: &CanonPos, _: &MatchCtx) -> MoneyChance {
    let mut w: [u128; 4] =
        core::array::from_fn(|t| 1 + 16 * u128::from(p.scores[t]) + 64 * u128::from(p.tricks[t]));
    for i in 0..p.slots.len() {
        if let Some((holder, place)) = p.slot(i) {
            w[usize::from(engine::team_of(holder))] += (i as u128 * 7 + u128::from(place)) % 11;
        }
    }
    MoneyChance::from_weights(w)
}

fn job(cut: u8, seed: u64, to: u64, threads: usize) -> CheckJob {
    CheckJob {
        cut,
        seed,
        label: 1,
        to,
        threads,
        tt_mb: 4,
        verify_pct: 0.0,
        recompute: false,
        rules_robber_burn: false,
        exact_solve_threads: 1,
        exact_tt_mb: 4,
    }
}

/// One world's values: (world, exact, hybrid).
type WorldKey = (u64, Vec<[u64; 4]>, Vec<[u64; 4]>);
/// Everything about a decision's check that is a value (no timings).
type DecisionKey = (u64, u64, usize, usize, u64, Vec<i64>, Vec<WorldKey>);

fn key(r: &CheckRun) -> Vec<DecisionKey> {
    let raw = |v: &[MoneyChance]| v.iter().map(|m| m.0).collect::<Vec<_>>();
    r.decisions
        .iter()
        .map(|d| {
            (
                d.id,
                d.worlds,
                d.exact_choice,
                d.hybrid_choice,
                d.loss,
                d.errors.clone(),
                d.world_checks
                    .iter()
                    .map(|w| (w.w, raw(&w.exact), raw(&w.hybrid)))
                    .collect(),
            )
        })
        .collect()
}

fn ckpt_line(d: &DecisionSpec, src: &str, seed: u64, w: u64, v: &[[u64; 4]]) -> String {
    serde_json::to_string(&CkptRow {
        id: d.id,
        fp: decision_fingerprint(d),
        method: "exact".into(),
        label: 1,
        seed,
        source: src.into(),
        table: "t".into(),
        w,
        v: v.to_vec(),
        us: 1000 + w,
        nodes: 10 + w,
        evals: 0,
        values: Some(VALUES_VERSION),
        frontier: None,
    })
    .expect("serialises")
}

/// The reader on synthetic records: checkpoint lines (current and pre-2026-10-15 forms), worlds
/// records, a torn line and another method; joining keeps a world found twice once, with both
/// places, and refuses different values; sets split by burn rule and refuse a version no switch
/// reproduces.
#[test]
fn reads_and_joins_synthetic_records() {
    let starts = collect_starts(300, 3);
    let ds = build_decisions(3, 2, 5, &starts);
    let l = leaf();
    let src = BeliefSource::new();
    let ctx = Ctx {
        leaf: &*l,
        source: &src,
        rules: engine::game::Rules::HOUSE,
    };
    let cfg = SearchConfig {
        tt_mb: 4,
        ..SearchConfig::default()
    };
    let rec = world_rows(&ds[0], Method::ExactPimc, 1, 5, 0, 3, "t", &ctx, &cfg);
    let mut text = String::new();
    for (w, v) in rec.rows.iter().enumerate() {
        text += &ckpt_line(&ds[0], src.name(), 5, w as u64, v);
        text.push('\n');
    }
    // An old checkpoint line: no table, no nodes, no values version.
    let quarter = [ONE / 4; 4];
    let old = serde_json::json!({"id": ds[1].id, "fp": decision_fingerprint(&ds[1]),
        "method": "exact", "label": 1, "seed": 5, "source": src.name(), "w": 7,
        "v": [quarter], "us": 55});
    text += &format!("{old}\n");
    // Another method, a torn line, and the worlds record of the same worlds.
    let mut other: CkptRow =
        serde_json::from_str(text.lines().next().expect("a line")).expect("a line");
    other.method = "rollout".into();
    text += &format!("{}\n", serde_json::to_string(&other).expect("json"));
    text += "{\"id\": 3, \"fp\": 1, \"meth\n";
    text += &format!("{}\n", serde_json::to_string(&rec).expect("json"));
    let (worlds, info) = parse_saved(&text, "run-a").expect("parses");
    assert_eq!(
        (
            info.lines,
            info.torn,
            info.ckpt_lines,
            info.records,
            info.worlds
        ),
        (7, 1, 5, 1, 7)
    );
    assert_eq!(info.other_method, 1);
    assert_eq!(
        info.by_version,
        BTreeMap::from([(1, 1), (VALUES_VERSION, 6)])
    );
    let w0 = &worlds[0];
    assert_eq!(
        (w0.id, w0.w, w0.values, w0.seed, w0.table.as_deref()),
        (ds[0].id, 0, VALUES_VERSION, Some(5), Some("t"))
    );
    assert_eq!(
        w0.found,
        vec![SavedTime {
            from: "run-a".into(),
            us: Some(1000),
            nodes: Some(10)
        }]
    );
    assert_eq!(w0.v, rec.rows[0]);
    let o = worlds.iter().find(|s| s.w == 7).expect("the old line");
    assert_eq!(
        (o.values, o.table.clone(), o.found[0].nodes),
        (1, None, None)
    );
    assert_eq!(o.burn_rule(), Some(BurnRule::RulesV1));
    // The record's worlds carry its candidates.
    let r1 = worlds.iter().rfind(|s| s.w == 1).expect("record world 1");
    assert_eq!(r1.cands.as_ref(), Some(&rec.cands));
    assert_eq!(r1.found[0].us, None);

    let joined = join_saved(worlds.clone()).expect("joins");
    assert_eq!(joined.len(), 4);
    let j1 = joined.iter().find(|s| s.w == 1).expect("world 1");
    assert_eq!(j1.found.len(), 2, "the checkpoint line and the record");
    assert!(j1.cands.is_some());
    let sets = saved_sets(joined).expect("sets");
    assert_eq!(
        sets.iter()
            .map(|s| (s.rule, s.values, s.worlds.len()))
            .collect::<Vec<_>>(),
        vec![
            (BurnRule::Searched, VALUES_VERSION, 3),
            (BurnRule::RulesV1, 1, 1)
        ]
    );
    assert!(BurnRule::RulesV1.rules_robber_burn() && !BurnRule::Searched.rules_robber_burn());

    // The same world with other values is refused; so is an unknown version.
    let mut bad = worlds.clone();
    bad[0].v[0] = [ONE, 0, 0, 0];
    bad.push(worlds[0].clone());
    assert!(join_saved(bad).is_err());
    let mut v9 = worlds[0].clone();
    v9.values = 99;
    assert!(saved_sets(vec![v9]).is_err());
    // A line that is neither form is refused.
    assert!(parse_saved("{\"id\": 1}\n", "x").is_err());
    assert_eq!(
        parse_pairs("0:3, 4:1").expect("pairs"),
        vec![(0, 3), (4, 1)]
    );
    assert!(parse_pairs("0-3").is_err());
}

/// The check on saved worlds is the hybrid check: at trick 3 with the frontier at trick 4 (where
/// exact is cheap), saved checkpoint lines give exactly what `run_check` gives on the same
/// worlds stored as a worlds file, or recomputed; 1 and 4 threads give the same bytes; a
/// verification sample of 100% finds every saved value identical, and a changed one is caught
/// before the hybrid runs; chosen worlds (not `0..N`) are regenerated as the yardstick draws
/// them.
#[test]
fn saved_check_is_the_hybrid_check() {
    let starts = collect_starts(300, 3);
    let ds = build_decisions(3, 8, 5, &starts);
    let l = leaf();
    let src = BeliefSource::new();
    let ctx = Ctx {
        leaf: &*l,
        source: &src,
        rules: engine::game::Rules::HOUSE,
    };
    let cfg = SearchConfig {
        tt_mb: 4,
        ..SearchConfig::default()
    };
    let f = FnFrontier(by_cards as fn(&CanonPos, &MatchCtx) -> MoneyChance);
    let rows: Vec<WorldRows> = ds
        .iter()
        .map(|d| world_rows(d, Method::ExactPimc, 1, 5, 0, 6, "t", &ctx, &cfg))
        .collect();
    let mut text = String::new();
    for (d, r) in ds.iter().zip(&rows) {
        for (w, v) in r.rows.iter().enumerate() {
            text += &ckpt_line(d, src.name(), 5, w as u64, v);
            text.push('\n');
        }
    }
    let (worlds, _) = parse_saved(&text, "t3").expect("parses");
    let sets = saved_sets(join_saved(worlds).expect("joins")).expect("sets");
    assert_eq!(sets.len(), 1);
    let (pl, notes) = plan(&ds, &sets[0], 99, src.name(), "t", &|_, _| true).expect("plan");
    assert_eq!((notes.decisions, notes.worlds), (8, 48));
    assert_eq!(notes.table_unrecorded + notes.seed_unrecorded, 0);

    let saved = check_worlds(&pl, &job(4, 5, 6, 1), &ctx, &f, None).expect("runs");
    let stored: BTreeMap<u64, WorldRows> = rows.iter().map(|r| (r.id, r.clone())).collect();
    let direct = run_check(&ds, &stored, &job(4, 5, 6, 2), &ctx, &f).expect("runs");
    let fresh = run_check(&ds, &BTreeMap::new(), &job(4, 5, 6, 3), &ctx, &f).expect("runs");
    assert_eq!(key(&saved), key(&direct));
    assert_eq!(key(&saved), key(&fresh));
    assert!(
        saved
            .decisions
            .iter()
            .any(|d| d.errors.iter().any(|&e| e != 0))
    );
    let text_of = |r: &CheckRun| serde_json::to_string(&report(r, 50, 1)).expect("json");
    assert_eq!(text_of(&saved), text_of(&direct));
    // Four threads, and every saved world recomputed.
    let four = check_worlds(&pl, &job(4, 5, 6, 4), &ctx, &f, None).expect("runs");
    assert_eq!(key(&four), key(&saved));
    let mut all = job(4, 5, 6, 4);
    all.verify_pct = 100.0;
    let verified = check_worlds(&pl, &all, &ctx, &f, None).expect("runs");
    assert_eq!(verified.verified, (48, 0));
    assert!(verified.mismatched.is_empty());
    assert_eq!(key(&verified), key(&saved));
    assert!(
        verified
            .decisions
            .iter()
            .flat_map(|d| &d.world_checks)
            .all(|w| w.verified == Some(true) && w.exact_solved)
    );
    // A changed saved value is caught by the verification, and the hybrid is not run.
    let mut wrong = sets[0].clone();
    wrong.worlds[5].v[0] = [ONE, 0, 0, 0];
    let (pw, _) = plan(&ds, &wrong, 5, src.name(), "t", &|_, _| true).expect("plan");
    let caught = check_worlds(&pw, &all, &ctx, &f, None).expect("runs");
    assert_eq!(
        caught.mismatched,
        vec![(wrong.worlds[5].id, wrong.worlds[5].w)]
    );
    assert!(caught.decisions.is_empty());

    // Chosen worlds: the odd ones, each the world the yardstick draws (the worlds file's exact
    // values; hybrid values as a direct hybrid solve of that world).
    let (odd, n) = plan(&ds, &sets[0], 5, src.name(), "t", &|_, w| w % 2 == 1).expect("plan");
    assert_eq!(n.worlds, 24);
    let r = check_worlds(&odd, &job(4, 5, 0, 2), &ctx, &f, None).expect("runs");
    let cut = FrontierCut::new(4, &f);
    for (d, dc) in ds.iter().zip(&r.decisions) {
        let (_, view, legal) = d.replay();
        let cands = candidates(&view, &legal).1;
        let worlds = src.prepare(&view, ctx.rules);
        let dseed = crate::measure::measure_seed(5, 1, d.id);
        assert_eq!(dc.worlds, 3);
        for wc in &dc.world_checks {
            assert_eq!(wc.w % 2, 1);
            let m = world_match(&view, &*worlds, wc.w, dseed, ctx.rules);
            let h = exact_world(
                &m,
                &cands,
                &ctx,
                &cfg,
                None,
                &mut SolveMem::new(),
                &mut WorldSolve::default(),
                Some(cut),
            )
            .expect("solves");
            assert_eq!(wc.hybrid, h);
            let raw: Vec<[u64; 4]> = wc.exact.iter().map(|m| m.0).collect();
            assert_eq!(raw, rows[dc.id as usize].rows[wc.w as usize]);
            assert!(!wc.exact_solved);
        }
    }
    // A decision missing from the decisions file, and other settings, are refused.
    assert!(plan(&ds[1..], &sets[0], 5, src.name(), "t", &|_, _| true).is_err());
    assert!(plan(&ds, &sets[0], 5, "rejection", "t", &|_, _| true).is_err());
    assert!(plan(&ds, &sets[0], 5, src.name(), "other", &|_, _| true).is_err());
    let mut moved = ds.clone();
    moved[2].seat = (moved[2].seat + 1) % 8;
    assert!(plan(&moved, &sets[0], 5, src.name(), "t", &|_, _| true).is_err());
}

/// Slices on saved worlds (the audit loop's decision-level stage at trick 2,
/// docs/m3-hybrid/audit-loop.md): the check of a saved set with a predicate file gives the
/// values of the plain check, the reaches and slices of the golden-set check with slices on the
/// same worlds, and the same on 1 and 3 threads.
#[test]
fn saved_check_slices_are_the_golden_checks() {
    use crate::hybrid::{check_worlds_with, run_check_slices, slices};
    let starts = collect_starts(300, 3);
    let ds = build_decisions(3, 8, 5, &starts);
    let l = leaf();
    let src = BeliefSource::new();
    let ctx = Ctx {
        leaf: &*l,
        source: &src,
        rules: engine::game::Rules::HOUSE,
    };
    let cfg = SearchConfig {
        tt_mb: 4,
        ..SearchConfig::default()
    };
    let f = FnFrontier(by_cards as fn(&CanonPos, &MatchCtx) -> MoneyChance);
    let rows: Vec<WorldRows> = ds
        .iter()
        .map(|d| world_rows(d, Method::ExactPimc, 1, 5, 0, 4, "t", &ctx, &cfg))
        .collect();
    let mut text = String::new();
    for (d, r) in ds.iter().zip(&rows) {
        for (w, v) in r.rows.iter().enumerate() {
            text += &ckpt_line(d, src.name(), 5, w as u64, v);
            text.push('\n');
        }
    }
    let (worlds, _) = parse_saved(&text, "t3").expect("parses");
    let sets = saved_sets(join_saved(worlds).expect("joins")).expect("sets");
    let (pl, _) = plan(&ds, &sets[0], 5, src.name(), "t", &|_, _| true).expect("plan");
    let preds = format!(
        r#"[{{"name": "next team no trick", "conditions": [{{"feature": {}, "op": "==", "value": 1}}]}}]"#,
        nn::encode::OFF_TRICKS + 5
    );
    let p = nn::predicate::Predicates::parse(preds.as_bytes()).expect("parses");
    let plain = check_worlds(&pl, &job(4, 5, 4, 2), &ctx, &f, None).expect("runs");
    let one = check_worlds_with(&pl, &job(4, 5, 4, 1), &ctx, &f, None, Some(&p)).expect("runs");
    let three = check_worlds_with(&pl, &job(4, 5, 4, 3), &ctx, &f, None, Some(&p)).expect("runs");
    let stored: BTreeMap<u64, WorldRows> = rows.iter().map(|r| (r.id, r.clone())).collect();
    let golden =
        run_check_slices(&ds, &stored, &job(4, 5, 4, 2), &ctx, &f, Some(&p)).expect("runs");
    assert_eq!(key(&one), key(&plain), "slices never change a value");
    assert_eq!(key(&one), key(&golden));
    let reach = |r: &CheckRun| -> Vec<_> { r.decisions.iter().map(|d| d.reach.clone()).collect() };
    assert_eq!(reach(&one), reach(&golden));
    assert_eq!(reach(&one), reach(&three));
    let sl = |r: &CheckRun| serde_json::to_string(&slices(r, &p, 500_000, 50, 1)).expect("json");
    assert_eq!(sl(&one), sl(&golden));
    let s = slices(&one, &p, 500_000, 50, 1);
    assert_eq!(s.len(), 2);
    assert_eq!(s[0].ids.len() + s[1].ids.len(), 8, "a partition here");
}

/// The burn rule reaches the solves: a set valued with the `rules-v1` burn is checked with
/// `rules_robber_burn` in the hybrid's solves (and in the exact ones, `CheckJob::config`). At
/// trick 1, with the frontier at trick 2 (cheap), in worlds where a seat after the decider robs,
/// the hybrid's values are those of a solve with the set's switch, and at least one world's
/// values differ between the two rules (so the test can tell them apart).
#[test]
fn burn_rule_switch_is_honoured() {
    let mut j = job(2, 9, 0, 1);
    for rules in [false, true] {
        j.rules_robber_burn = rules;
        assert_eq!(j.config(true).rules_robber_burn, rules);
        assert_eq!(j.config(false).rules_robber_burn, rules);
    }
    j.exact_solve_threads = 3;
    j.exact_tt_mb = 7;
    assert_eq!((j.config(true).solve_threads, j.config(true).tt_mb), (3, 7));
    assert_eq!(
        (j.config(false).solve_threads, j.config(false).tt_mb),
        (1, 4)
    );

    let starts = collect_starts(300, 3);
    let l = leaf();
    let src = BeliefSource::new();
    let ctx = Ctx {
        leaf: &*l,
        source: &src,
        rules: engine::game::Rules::HOUSE,
    };
    let f = FnFrontier(by_cards as fn(&CanonPos, &MatchCtx) -> MoneyChance);
    let cut = FrontierCut::new(2, &f);
    // Late positions of trick 1 (cheap), worlds where a later seat robs.
    let ds: Vec<DecisionSpec> = build_decisions(1, 64, 9, &starts)
        .into_iter()
        .filter(|d| d.position >= 4)
        .collect();
    let mut robbed: Vec<(usize, u64)> = Vec::new();
    for (k, d) in ds.iter().enumerate() {
        let (_, view, _) = d.replay();
        let worlds = src.prepare(&view, ctx.rules);
        let dseed = crate::measure::measure_seed(9, 1, d.id);
        for w in 0..12 {
            if world_robber(&view, &*worlds, w, dseed, &ctx).is_some() {
                robbed.push((k, w));
            }
        }
    }
    assert!(robbed.len() >= 4, "{} robbed worlds", robbed.len());
    robbed.truncate(12);
    let hybrid = |k: usize, w: u64, rules: bool| -> Vec<MoneyChance> {
        let d = &ds[k];
        let (_, view, legal) = d.replay();
        let cands = candidates(&view, &legal).1;
        let worlds = src.prepare(&view, ctx.rules);
        let m = world_match(
            &view,
            &*worlds,
            w,
            crate::measure::measure_seed(9, 1, d.id),
            ctx.rules,
        );
        let cfg = SearchConfig {
            tt_mb: 4,
            rules_robber_burn: rules,
            ..SearchConfig::default()
        };
        exact_world(
            &m,
            &cands,
            &ctx,
            &cfg,
            None,
            &mut SolveMem::new(),
            &mut WorldSolve::default(),
            Some(cut),
        )
        .expect("solves")
    };
    // Saved sets of those worlds under each rule (the stored values are placeholders: nothing is
    // verified here; the hybrid's values are what is tested).
    let saved = |values: u32| -> Vec<SavedWorld> {
        robbed
            .iter()
            .map(|&(k, w)| {
                let d = &ds[k];
                let (_, view, legal) = d.replay();
                SavedWorld {
                    id: d.id,
                    w,
                    values,
                    method: "exact".into(),
                    label: 1,
                    seed: Some(9),
                    source: Some(src.name().into()),
                    table: None,
                    decision_fp: Some(decision_fingerprint(d)),
                    cands: None,
                    v: vec![MoneyChance::even().0; candidates(&view, &legal).1.len()],
                    found: Vec::new(),
                }
            })
            .collect()
    };
    let mut differ = 0;
    for values in [1, VALUES_VERSION] {
        let sets = saved_sets(saved(values)).expect("sets");
        assert_eq!(sets.len(), 1);
        let set = &sets[0];
        let rules = values == 1;
        assert_eq!(set.rule.rules_robber_burn(), rules);
        let (pl, _) = plan(&ds, set, 9, src.name(), "t", &|_, _| true).expect("plan");
        let sjob = CheckJob {
            rules_robber_burn: set.rule.rules_robber_burn(),
            ..job(2, 9, 0, 2)
        };
        let run = check_worlds(&pl, &sjob, &ctx, &f, None).expect("runs");
        for dc in &run.decisions {
            let k = ds.iter().position(|d| d.id == dc.id).expect("decision");
            for wc in &dc.world_checks {
                assert!(wc.robber.is_some());
                assert_eq!(wc.hybrid, hybrid(k, wc.w, rules));
                if rules && wc.hybrid != hybrid(k, wc.w, false) {
                    differ += 1;
                }
            }
        }
    }
    assert!(differ > 0, "no world where the burn rule changes a value");
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/t1-saved")
}

/// Real saved trick-1 files (small copies in `testdata/t1-saved`): the 26 checkpoint lines of
/// run `m3-t1-tables-1`, the two worlds records of `m3-t1-leader-layouts-1` and nine lines of
/// `m3-agree-t1-ab1` (values version 1), against the trick-1 decisions file (count 32, seed 1).
#[test]
fn real_saved_trick1_files() {
    let dir = fixtures();
    let ds: Vec<DecisionSpec> = read_jsonl(&dir.join("decisions-t1.jsonl")).expect("decisions");
    assert_eq!(ds.len(), 32);
    let (t, ti) = read_saved(&dir.join("m3-t1-tables-1.ckpt"), "m3-t1-tables-1").expect("reads");
    assert_eq!(
        (ti.ckpt_lines, ti.records, ti.torn, ti.worlds),
        (26, 0, 0, 26)
    );
    assert_eq!(ti.by_version, BTreeMap::from([(2, 26)]));
    let (l, li) = read_saved(
        &dir.join("m3-t1-leader-layouts-1.jsonl"),
        "m3-t1-leader-layouts-1",
    )
    .expect("reads");
    assert_eq!((li.records, li.worlds), (2, 8));
    let (o, oi) =
        read_saved(&dir.join("m3-agree-t1-ab1.part.ckpt"), "m3-agree-t1-ab1").expect("reads");
    assert_eq!((oi.ckpt_lines, oi.by_version.get(&1)), (9, Some(&9)));
    assert!(
        o.iter()
            .all(|s| s.table.is_none() && s.found[0].nodes.is_none())
    );
    let joined = join_saved([t, l, o].concat()).expect("joins");
    // The known hard world: decision 0, world 3, in both runs' files with the same values.
    let hard = joined
        .iter()
        .find(|s| (s.id, s.w, s.values) == (0, 3, 2))
        .expect("the hard world");
    assert_eq!(
        hard.found,
        vec![
            SavedTime {
                from: "m3-t1-tables-1".into(),
                us: Some(13_056_957_069),
                nodes: Some(79_320_348_281)
            },
            SavedTime {
                from: "m3-t1-leader-layouts-1".into(),
                us: None,
                nodes: None
            }
        ]
    );
    let sets = saved_sets(joined).expect("sets");
    assert_eq!(
        sets.iter()
            .map(|s| (s.rule, s.worlds.len()))
            .collect::<Vec<_>>(),
        vec![(BurnRule::Searched, 30), (BurnRule::RulesV1, 9)]
    );
    let src = BeliefSource::new();
    let table = crate::bot::EQUITY_SHA256;
    let (p1, n1) = plan(&ds, &sets[0], 1, src.name(), table, &|_, _| true).expect("plan");
    assert_eq!(
        (
            n1.decisions,
            n1.worlds,
            n1.table_unrecorded,
            n1.fp_unrecorded
        ),
        (8, 30, 0, 0)
    );
    assert_eq!(
        p1[0].worlds.iter().map(|t| t.w).collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    let (_, n2) = plan(&ds, &sets[1], 1, src.name(), table, &|_, _| true).expect("plan");
    assert_eq!((n2.decisions, n2.worlds, n2.table_unrecorded), (3, 9, 9));
    assert!(plan(&ds, &sets[0], 1, src.name(), "other", &|_, _| true).is_err());
    // Worlds valued under both rules: where no later seat robs, the rules agree.
    assert_eq!(
        both_rules(&sets),
        vec![(3, 0, true), (3, 3, true), (4, 0, true), (6, 1, true)]
    );
}

/// With the production equity table (`FORTYONE_EQUITY`; skipped without it): world 25 of
/// decision 6, saved by run `m3-agree-t1-ab1` (values version 1), is byte-identical on this
/// solver with `rules_robber_burn`, and differs without it (seat 2 robs later in trick 1).
#[test]
fn real_rules_burn_world_with_the_table() {
    let Ok(path) = std::env::var("FORTYONE_EQUITY") else {
        return;
    };
    let (table, sha) = crate::bot::load_equity_sha(&path).expect("the production table");
    let l: Arc<SharedLeaf> = Arc::new(table);
    let src = BeliefSource::new();
    let ctx = Ctx {
        leaf: &*l,
        source: &src,
        rules: engine::game::Rules::HOUSE,
    };
    let dir = fixtures();
    let ds: Vec<DecisionSpec> = read_jsonl(&dir.join("decisions-t1.jsonl")).expect("decisions");
    let (o, _) =
        read_saved(&dir.join("m3-agree-t1-ab1.part.ckpt"), "m3-agree-t1-ab1").expect("reads");
    let sets = saved_sets(join_saved(o).expect("joins")).expect("sets");
    let (pl, _) = plan(&ds, &sets[0], 1, src.name(), &sha, &|d, w| {
        (d, w) == (6, 25)
    })
    .expect("plan");
    let f = FnFrontier(even as fn(&CanonPos, &MatchCtx) -> MoneyChance);
    let mut j = job(3, 1, 0, 1);
    j.verify_pct = 100.0;
    j.rules_robber_burn = sets[0].rule.rules_robber_burn();
    assert!(j.rules_robber_burn);
    let r = check_worlds(&pl, &j, &ctx, &f, None).expect("runs");
    assert_eq!(r.verified, (1, 0));
    assert_eq!(r.decisions[0].world_checks[0].robber, Some(2));
    j.rules_robber_burn = false;
    let r = check_worlds(&pl, &j, &ctx, &f, None).expect("runs");
    assert_eq!(r.mismatched, vec![(6, 25)]);
}

/// The whole `m3-agree-t1-ab1` checkpoint (355 worlds, values version 1, 117 KB: too big for a
/// fixture), read from `T1_SAVED_AB1` when set; skipped otherwise.
#[test]
fn full_ab1_file_when_present() {
    let Ok(path) = std::env::var("T1_SAVED_AB1") else {
        return;
    };
    let (o, info) = read_saved(Path::new(&path), "m3-agree-t1-ab1").expect("reads");
    assert_eq!((info.ckpt_lines, info.torn), (355, 0));
    assert_eq!(info.by_version, BTreeMap::from([(1, 355)]));
    let sets = saved_sets(join_saved(o).expect("joins")).expect("sets");
    assert_eq!((sets.len(), sets[0].rule), (1, BurnRule::RulesV1));
    let ds: Vec<DecisionSpec> =
        read_jsonl(&fixtures().join("decisions-t1.jsonl")).expect("decisions");
    let (_, n) = plan(
        &ds,
        &sets[0],
        1,
        "belief-uniform",
        crate::bot::EQUITY_SHA256,
        &|_, _| true,
    )
    .expect("plan");
    assert_eq!((n.decisions, n.worlds), (5, 355));
}
