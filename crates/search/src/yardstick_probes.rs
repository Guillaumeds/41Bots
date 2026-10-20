//! Review C probes (2026-10-15): the trick-1 world samplers against the engine, and the
//! yardstick's resume fingerprint.
//!
//! - `trick1_belief_marginals_match_engine_oracle`: at trick-1 decisions (pending robber, rob
//!   at turn, turned Joker, Joker then ace, turned ace), the exact probability that a card lies
//!   in each place, from the belief counter, against (a) the engine-based rejection oracle and
//!   (b) the belief sampler; every rejection world is also accepted by the tracker (no fact
//!   missing from the tracker at trick 1).

use belief::{Loc, Tracker, WorldSpace};
use engine::cards::{Bits, Card, DECK_SIZE, JOKER, bit, cards, has};
use engine::game::{Event, Match, Phase, Rules};
use engine::rng::Rng;
use engine::rules::ace_of_trumps;
use engine::view::SeatView;

use crate::pimc::candidates;
use crate::worlds::{BeliefSource, RejectionSource, World, WorldSource, hand_events, world_rng};

/// A random first deal of a match, with `pack` forced at deck positions 40.. (the turned card
/// and the next one).
fn deal_with(seed: u64, pack: &[Card]) -> Match {
    let mut rng = Rng::derive(seed, &[0xC1]);
    let mut deck: [Card; DECK_SIZE] = core::array::from_fn(|i| i as Card);
    rng.shuffle(&mut deck);
    for (k, &c) in pack.iter().enumerate() {
        let at = deck.iter().position(|&x| x == c).expect("card");
        deck.swap(at, 40 + k);
    }
    let dealer = (seed % 8) as u8;
    Match::with_decks(dealer, vec![deck], seed, Rules::HOUSE)
}

/// Trick-1 decision views (two or more candidates) of the first hand of `m`, every seat played
/// by `rules-v1`.
fn trick1_views(mut m: Match) -> Vec<SeatView> {
    let mut out = Vec::new();
    let h = m.hand_index();
    while m.phase() != Phase::MatchOver && m.hand_index() == h {
        let seat = m.to_act().expect("to act");
        let v = m.view(seat);
        let legal = m.legal_actions();
        if v.public.trick_no > 1 {
            break;
        }
        if v.public.trick_no == 1 && candidates(&v, &legal).1.len() >= 2 {
            out.push(v.clone());
        }
        let a = bots::decide(&v, &Rules::HOUSE, &legal).0;
        m.apply(a).expect("legal");
    }
    out
}

/// Where card `c` is in a search world, as a tracker location.
fn loc_of(w: &World, c: Card, burn_locs: &[Loc]) -> Option<Loc> {
    for s in 0..8u8 {
        if has(w.hands[usize::from(s)], c) {
            return Some(Loc::Hand(s));
        }
    }
    if has(w.undealt, c) {
        return Some(Loc::Undealt);
    }
    w.burns.iter().position(|&b| b == c).map(|i| burn_locs[i])
}

/// The exact number of worlds with card `c` in location index `i`.
fn count_with(t: &Tracker, c: Card, i: usize) -> u128 {
    let mut allowed = t.allowed;
    allowed[usize::from(c)] &= 1 << i;
    if allowed[usize::from(c)] == 0 {
        return 0;
    }
    let t2 = Tracker::from_parts(t.seat, t.hand, t.unknown, t.locs.clone(), allowed);
    WorldSpace::new(&t2).map_or(0, |ws| ws.count())
}

#[test]
fn trick1_belief_marginals_match_engine_oracle() {
    use engine::cards::{Suit, card};
    let mut views = Vec::new();
    for seed in 0..48u64 {
        let pack: Vec<Card> = match seed % 6 {
            // A turned Joker, then an ordinary card: the dealer may be the pending robber.
            0 => vec![JOKER, card(Suit::Clubs, 7)],
            // A turned Joker then an ace (the dealer robs it at once).
            1 => vec![JOKER, card(Suit::Spades, 1)],
            // A turned ace (the dealer robs it at once).
            2 => vec![card(Suit::Diamonds, 1)],
            // An ordinary turned card (rob at turn or pending robber).
            _ => vec![card(Suit::Hearts, 9)],
        };
        let vs = trick1_views(deal_with(seed, &pack));
        // A spread of positions in the trick.
        for (k, v) in vs.into_iter().enumerate() {
            if k % 3 == (seed % 3) as usize {
                views.push(v);
            }
        }
    }
    assert!(views.len() >= 40, "{} views", views.len());
    let (mut tests, mut worst) = (0usize, 0f64);
    let samples = 5000u64;
    for (vi, view) in views.iter().enumerate() {
        let t = Tracker::from_view(view);
        let ws = WorldSpace::new(&t).expect("feasible");
        let n = ws.count() as f64;
        let burn_locs: Vec<Loc> = view
            .public
            .burn_order
            .iter()
            .filter(|&&(s, _)| s != view.seat)
            .map(|&(s, k)| Loc::Burn(s, k))
            .collect();
        // Cards with facts of their own: the ace of trumps, the taken cards, the Joker; and two
        // plain unknown cards.
        let mut probe: Vec<Card> = Vec::new();
        if let Some(tr) = view.public.trump {
            probe.push(ace_of_trumps(tr));
        }
        for e in hand_events(view) {
            if let Event::Took { card, .. } = *e {
                probe.push(card);
            }
        }
        probe.push(JOKER);
        let unknown: Vec<Card> = cards(t.unknown).collect();
        probe.push(unknown[vi % unknown.len()]);
        probe.push(unknown[(vi * 7 + 3) % unknown.len()]);
        probe.retain(|&c| has(t.unknown, c));
        probe.sort_unstable();
        probe.dedup();
        let rej = RejectionSource::new();
        let rw = rej.prepare(view, Rules::HOUSE);
        let bel = BeliefSource::new();
        let bw = bel.prepare(view, Rules::HOUSE);
        let rworlds: Vec<World> = (0..samples)
            .map(|w| rw.sample(&mut world_rng(0x5EED ^ vi as u64, w)))
            .collect();
        let bworlds: Vec<World> = (0..samples)
            .map(|w| bw.sample(&mut world_rng(0xB0B ^ vi as u64, w)))
            .collect();
        // Support: every engine-accepted world satisfies every tracker fact.
        for w in &rworlds {
            let burns: Vec<(Loc, Bits)> = burn_locs
                .iter()
                .zip(&w.burns)
                .map(|(&l, &c)| (l, bit(c)))
                .collect();
            assert!(
                t.consistent(&w.hands, &burns, w.undealt),
                "view {vi}: an engine-consistent world is excluded by the tracker"
            );
        }
        for &c in &probe {
            for (i, &(loc, _)) in t.locs.iter().enumerate() {
                let p = count_with(&t, c, i) as f64 / n;
                for (name, ws) in [("rejection", &rworlds), ("belief", &bworlds)] {
                    let k = ws
                        .iter()
                        .filter(|w| loc_of(w, c, &burn_locs) == Some(loc))
                        .count() as f64;
                    let e = p * samples as f64;
                    if p == 0.0 {
                        assert_eq!(k, 0.0, "view {vi} {name}: card {c} in {loc:?} has no mass");
                        continue;
                    }
                    let z = (k - e) / (e * (1.0 - p)).sqrt().max(1e-9);
                    tests += 1;
                    worst = worst.max(z.abs());
                    assert!(
                        z.abs() < 5.5,
                        "view {vi} {name}: card {c} in {loc:?}: {k} of {samples}, exact p {p:.4} \
                         (z {z:.1})"
                    );
                }
            }
        }
    }
    assert!(tests > 500, "{tests} tests");
    eprintln!("trick-1 marginals: {tests} tests, worst |z| {worst:.2}");
}

// ---------------------------------------------------------------- resume across binaries

mod stale {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use engine::game::Rules;
    use equity::{MatchState, MoneyChance};
    use serde_json::json;
    use solver::FnLeaf;

    use crate::measure::{
        DecisionSpec, WorldRows, WorldsJob, build_decisions, ckpt_path, collect_starts,
        decision_fingerprint, merge_rows, read_jsonl, timing_path, worlds_file,
    };
    use crate::measure::{Schedule, worlds_file_sched};
    use crate::pimc::{Ctx, Method, SearchConfig, SharedLeaf};
    use crate::plan::AutoParams;
    use crate::worlds::{BeliefSource, RejectionSource};

    const TABLE: &str = "review-c-table";

    fn leaf() -> Arc<SharedLeaf> {
        Arc::new(FnLeaf(|s: &MatchState| {
            MoneyChance::from_weights(core::array::from_fn(|t| {
                1 + u128::from(s.scores[t]) + 40 * u128::from(s.games[t])
            }))
        }))
    }

    fn tmp(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "search-review-c-{}-{tag}.jsonl",
            std::process::id()
        ))
    }

    fn clean(p: &Path) {
        for q in [p.to_path_buf(), timing_path(p), ckpt_path(p)] {
            let _ = std::fs::remove_file(q);
        }
    }

    fn set() -> Vec<DecisionSpec> {
        let starts = collect_starts(400, 5);
        build_decisions(4, 3, 3, &starts)
    }

    fn job(from: u64, to: u64) -> WorldsJob {
        WorldsJob {
            method: Method::ExactPimc,
            label: 1,
            seed: 7,
            from,
            to,
            table: TABLE.to_string(),
            frontier: None,
            exact_leaves: None,
        }
    }

    fn run(p: &Path, ds: &[DecisionSpec], job: &WorldsJob) -> Result<Vec<u8>, String> {
        let l = leaf();
        let src = RejectionSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: Rules::HOUSE,
        };
        let cfg = SearchConfig {
            tt_mb: 4,
            ..SearchConfig::default()
        };
        worlds_file(p, ds, job, &ctx, &cfg, None, |_, _| {})?;
        Ok(std::fs::read(p).expect("written"))
    }

    /// A checkpoint line exactly as the binaries before this review wrote it (every fingerprint
    /// field this run checks, no values version), holding values the current code does not
    /// compute: what a checkpoint uploaded by a run on an older binary (another seat's rob burn
    /// by `rules-v1`, before 2026-10-15 22:29 UTC, or the solver before 7253b48) looks like to a
    /// resumed run on the current one.
    fn old_line(d: &DecisionSpec, job: &WorldsJob, w: u64, v: &[[u64; 4]]) -> String {
        json!({
            "id": d.id, "fp": decision_fingerprint(d), "method": "exact", "label": job.label,
            "seed": job.seed, "source": "rejection", "table": TABLE, "w": w, "v": v,
            "us": 1, "nodes": 1,
        })
        .to_string()
    }

    pub(super) fn belief_identity() {
        let starts = collect_starts(400, 5);
        let mut ds = build_decisions(3, 8, 11, &starts);
        ds.extend(build_decisions(4, 8, 12, &starts).into_iter().map(|mut d| {
            d.id += 100;
            d
        }));
        let j = job(0, 4);
        let l = leaf();
        let src = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*l,
            source: &src,
            rules: Rules::HOUSE,
        };
        let go = |tag: &str, ds: &[DecisionSpec], threads: usize, sched: &Schedule| {
            let p = tmp(tag);
            clean(&p);
            let cfg = SearchConfig {
                threads,
                tt_mb: 2,
                ..SearchConfig::default()
            };
            worlds_file_sched(&p, ds, &j, &ctx, &cfg, sched, None, |_, _| {}).expect("runs");
            let b = std::fs::read(&p).expect("written");
            clean(&p);
            b
        };
        let one = go("b1", &ds, 1, &Schedule::Fixed);
        assert_eq!(one, go("b4", &ds, 4, &Schedule::Fixed), "1 vs 4 threads");
        let auto = Schedule::Auto {
            threads: 4,
            budget_mb: 16,
            params: Box::new(AutoParams {
                leader_cost_s: [0.0, 0.0, 0.0, 400.0, 100.0, 1.0],
                position_ratio: 1.5,
                leader_table_mb: [0.0, 0.0, 0.0, 8.0, 2.0, 1.0],
                target_s: Some(20.0),
                ..AutoParams::default()
            }),
        };
        assert_eq!(one, go("ba", &ds, 4, &auto), "position-aware schedule");
        // Two chunks of decisions, joined: the same records.
        let (a, b): (Vec<DecisionSpec>, Vec<DecisionSpec>) =
            ds.iter().cloned().partition(|d| d.id % 2 == 0);
        let mut recs: Vec<WorldRows> = Vec::new();
        for (tag, part) in [("bc1", &a), ("bc2", &b)] {
            let bytes = go(tag, part, 2, &Schedule::Fixed);
            recs.extend(
                String::from_utf8(bytes)
                    .expect("utf8")
                    .lines()
                    .map(|l| serde_json::from_str::<WorldRows>(l).expect("row")),
            );
        }
        let whole: Vec<WorldRows> = String::from_utf8(one)
            .expect("utf8")
            .lines()
            .map(|l| serde_json::from_str(l).expect("row"))
            .collect();
        for r in &whole {
            let c = recs.iter().find(|x| x.id == r.id).expect("chunked record");
            assert_eq!(c, r, "decision {} differs when run in a chunk", r.id);
        }
        assert!(whole.iter().all(|r| r.log2_worlds.is_some()));
    }

    /// Worlds valued by another version of the code must never be resumed into a yardstick file:
    /// the fingerprint must say which version of the values a world holds.
    #[test]
    fn checkpoint_from_an_older_binary_is_refused() {
        let ds = set();
        let j = job(0, 3);
        let base = tmp("ck-base");
        clean(&base);
        let want = run(&base, &ds, &j).expect("fresh run");
        let rows: Vec<WorldRows> = read_jsonl(&base).expect("rows");
        clean(&base);
        // The old binary's values of world 0 of the first decision differ from ours (here: the
        // values of world 1, standing in for a different burn rule).
        let r = &rows[0];
        let d = ds.iter().find(|d| d.id == r.id).expect("decision");
        let p = tmp("ck-old");
        clean(&p);
        std::fs::write(ckpt_path(&p), old_line(d, &j, 0, &r.rows[1]) + "\n").expect("ckpt");
        let got = run(&p, &ds, &j);
        clean(&p);
        assert!(
            got.is_err(),
            "an older binary's checkpoint was resumed silently (file {} the fresh run's)",
            if got.as_ref().is_ok_and(|g| *g == want) {
                "equals"
            } else {
                "differs from"
            }
        );
    }

    /// A worlds record written by an older binary (fingerprint without a values version) must
    /// neither be resumed nor joined to worlds valued by the current code.
    #[test]
    fn records_from_an_older_binary_are_not_extended() {
        let ds = set();
        let base = tmp("rec-base");
        clean(&base);
        run(&base, &ds, &job(0, 3)).expect("fresh run");
        let rows: Vec<WorldRows> = read_jsonl(&base).expect("rows");
        clean(&base);
        // The same records as an older binary wrote them: the fingerprint has no version.
        let old: Vec<WorldRows> = rows
            .iter()
            .map(|r| {
                let mut v = serde_json::to_value(r).expect("json");
                v["fp"]
                    .as_object_mut()
                    .expect("fingerprint")
                    .retain(|k, _| ["seed", "decision", "source", "table"].contains(&k.as_str()));
                serde_json::from_value(v).expect("an old record still reads")
            })
            .collect();
        // Resuming into the old file (a longer range is refused anyway, so the same range).
        let p = tmp("rec-old");
        clean(&p);
        let text: String = old
            .iter()
            .map(|r| serde_json::to_string(r).expect("json") + "\n")
            .collect();
        std::fs::write(&p, text).expect("write");
        let resumed = run(&p, &ds, &job(0, 3));
        clean(&p);
        assert!(
            resumed.is_err(),
            "an older binary's records were taken as this run's own"
        );
        // Extending them with worlds 3..5 valued now, in another file, then joining the two.
        let e = tmp("rec-ext");
        clean(&e);
        run(&e, &ds, &job(3, 5)).expect("extension");
        let ext: Vec<WorldRows> = read_jsonl(&e).expect("rows");
        clean(&e);
        let joined = merge_rows(old.into_iter().chain(ext).collect());
        assert!(
            joined.is_err(),
            "an older binary's worlds 0..3 were joined to this binary's worlds 3..5"
        );
    }
}

/// The decision set's robber nodes: a seat that robs at its turn in trick 1 has two nodes at its
/// position (the joint burn-and-play decision, then the play after its burn); the bot searches
/// only the first (M3-SPEC §5.5: the play after a joint decision is not searched again).
#[test]
fn decision_set_robber_nodes() {
    use crate::measure::{build_decisions, collect_starts};
    let starts = collect_starts(2000, 1);
    let ds = build_decisions(1, 256, 1, &starts);
    let (mut joint, mut after_burn) = (0, 0);
    for d in &ds {
        let (_, v, _) = d.replay();
        let p = &v.public;
        let robbed_now = p.robber == Some(d.seat)
            && p.burn_order
                .iter()
                .any(|&(s, k)| s == d.seat && k == engine::view::BurnKind::RobAtTurn);
        match p.phase {
            Phase::AwaitBurn(_) => joint += 1,
            Phase::AwaitPlay(_) if robbed_now => after_burn += 1,
            _ => {}
        }
    }
    eprintln!(
        "trick-1 set of {}: {joint} joint decisions, {after_burn} plays right after the decider's own rob burn",
        ds.len()
    );
    assert_eq!(
        after_burn, 0,
        "{after_burn} decisions are the play right after the decider's own joint burn decision"
    );
}

/// The joint (burn, play) candidates of a rob at turn are exactly the engine's: every legal burn,
/// and after it every play the engine then allows; nothing else.
#[test]
fn joint_candidates_are_exactly_the_engines() {
    use engine::cards::{Suit, card};
    use engine::game::Action;
    let mut checked = 0;
    for seed in 0..400u64 {
        let pack = [card(Suit::ALL[(seed % 4) as usize], 2 + (seed % 9) as u8)];
        let mut m = deal_with(seed, &pack);
        let h = m.hand_index();
        while m.phase() != Phase::MatchOver && m.hand_index() == h && m.public().trick_no <= 1 {
            let seat = m.to_act().expect("to act");
            let v = m.view(seat);
            let legal = m.legal_actions();
            if matches!(v.public.phase, Phase::AwaitBurn(_)) && v.public.trick_no == 1 {
                let (_, cands) = candidates(&v, &legal);
                let mut want = Vec::new();
                for &b in &legal {
                    let mut mm = m.clone();
                    mm.apply(b).expect("legal burn");
                    for p in mm.legal_actions() {
                        let Action::Play(c) = p else {
                            panic!("a play follows the burn")
                        };
                        want.push((b, Some(c)));
                    }
                }
                assert_eq!(cands, want, "seed {seed}");
                checked += 1;
            }
            let a = bots::decide(&v, &Rules::HOUSE, &legal).0;
            m.apply(a).expect("legal");
        }
    }
    assert!(checked > 100, "{checked}");
}

/// The production world source (the belief sampler; the measure tests use the rejection oracle):
/// the worlds file is the same bytes on 1 and 4 threads, under the position-aware schedule, and
/// when the run is split into two chunks of decisions (`--ids`) and joined.
#[test]
fn belief_worlds_files_identical_under_threads_schedules_and_chunks() {
    stale::belief_identity();
}
