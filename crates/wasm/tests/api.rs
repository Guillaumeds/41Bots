//! The app's API (docs/WASM-API.md): the seat view rebuilt from the app's JSON is the engine's
//! own, the values are the native search's, shards merge to the same bytes, and every refusal
//! code. The networks here are seeded random ones over a synthetic leaf (the real files are not
//! in the repo): they test the plumbing, never a value's quality. `tools/wasm_check.sh` runs the
//! same requests with the real files natively and as WebAssembly.

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::game::{Event, Match, Phase, Rules};
use engine::team_of;
use engine::view::SeatView;
use equity::MoneyChance;
use nn::Oracle;
use nn::synthetic::{SyntheticLeaf, random_net};
use search::pimc::{
    Ctx, SearchConfig, WorldSolve, aggregate, candidates, exact_world, world_match,
};
use search::worlds::{BeliefSource, WorldSource, hand_events};
use serde_json::{Value, json};
use solver::{FrontierCut, Leaf, SolveMem};
use wasm::api::{Engine, merge};
use wasm::appjson::{Mask, app_match, app_request};
use wasm::appview::{AppTable, rebuild};
use wasm::cards::{app_card, engine_card};
use wasm::fixtures::{any_turned, app_actions, find, playing, referee_step};

fn leaf() -> Arc<dyn Leaf + Send + Sync> {
    Arc::new(SyntheticLeaf)
}

/// A seeded random network for trick `k` with the outcome head (as O_3 has), as a weights file.
fn net_bytes(k: u8) -> Vec<u8> {
    let net = random_net(u64::from(k) * 7 + 1, &[32, 16], nn::outcome::vocab(k), 100);
    let extra = BTreeMap::from([("head".to_string(), "outcome-v1".to_string())]);
    nn::oracle::write_net(&net, k, &extra)
}

/// An engine over the synthetic leaf with random O_3 (and O_2 when asked).
fn engine(o2: bool) -> Engine {
    let mut e = Engine::with_leaf(leaf(), "synthetic");
    e.load_net(&net_bytes(3)).expect("loads");
    if o2 {
        e.load_net(&net_bytes(2)).expect("loads");
    }
    e
}

fn parse(s: &str) -> Value {
    serde_json::from_str(s).expect("the API writes JSON")
}

fn code(s: &str) -> String {
    let v = parse(s);
    assert_eq!(v["status"], "refused", "{s}");
    v["code"].as_str().expect("a code").to_string()
}

/// Every decision of random matches from `seed`, with the app's referee steps about one chance
/// in `rate` at each opportunity (0: none): `f` sees each position before its action.
fn each_decision(seed: u64, matches: u64, rate: u64, mut f: impl FnMut(&Match)) {
    for i in 0..matches {
        let mut m = Match::new(seed + i, Rules::HOUSE);
        let mut rng = engine::rng::Rng::derive(seed, &[i]);
        while m.to_act().is_some() {
            f(&m);
            if rate > 0 && referee_step(&mut m, &mut rng, rate) {
                continue;
            }
            let mut legal = app_actions(&m);
            let mut ok = false;
            while !legal.is_empty() {
                let a = legal.remove(rng.bounded(legal.len() as u64) as usize);
                if m.apply(a).is_ok() {
                    ok = true;
                    break;
                }
            }
            if !ok {
                break;
            }
        }
    }
}

fn same_view(a: &SeatView, b: &SeatView) -> bool {
    a.seat == b.seat
        && a.hand == b.hand
        && a.own_burns == b.own_burns
        && a.public == b.public
        && hand_events(a) == hand_events(b)
}

/// Where two views differ, in words (for failures).
fn view_diff(a: &SeatView, b: &SeatView) -> String {
    if a.hand != b.hand || a.own_burns != b.own_burns {
        return format!(
            "hand {} / {}, burns {} / {}",
            engine::cards::names(a.hand),
            engine::cards::names(b.hand),
            engine::cards::names(a.own_burns),
            engine::cards::names(b.own_burns)
        );
    }
    if a.public != b.public {
        return format!("public\n{:?}\n{:?}", a.public, b.public);
    }
    let (x, y) = (hand_events(a), hand_events(b));
    for (i, (e, f)) in x.iter().zip(y).enumerate() {
        if e != f {
            return format!("event {i}: {} / {}", e.canonical(), f.canonical());
        }
    }
    format!("{} / {} events", x.len(), y.len())
}

/// Does the seat to act have a renege of its own nobody has caught (the app then shows it to
/// that seat alone, and the API refuses the view)?
fn own_hidden_renege(m: &Match) -> bool {
    let s = m.to_act().expect("a decision");
    let open = app_match(m, s, Mask::Open);
    open["hand"]["reneges"]
        .as_array()
        .expect("a list")
        .iter()
        .any(|r| r["seat"] == s && r["caughtAt"] == -1)
}

fn check_rebuild(m: &Match, counts: &mut [u32; 4]) {
    let seat = m.to_act().expect("a decision");
    let want = m.view(seat);
    let hidden_own = own_hidden_renege(m);
    for mask in [Mask::SeatOwnBurns, Mask::Open] {
        let j = app_match(m, seat, mask);
        let t = AppTable::read(&j, seat, None);
        let got = t.and_then(|t| rebuild(&t));
        match got {
            Ok(r) => {
                assert!(!hidden_own, "a seat's own hidden renege is refused");
                assert!(
                    same_view(&r.view, &want),
                    "{mask:?}: the rebuilt view differs: {}\n{}",
                    view_diff(&r.view, &want),
                    engine::game::transcript(m)
                );
                counts[0] += 1;
            }
            Err(e) if hidden_own && e.code == "inconsistent_view" => counts[3] += 1,
            Err(e) => panic!(
                "{mask:?}: refused {e:?}\nseat {seat}\n{}\n{}",
                engine::game::transcript(m),
                j
            ),
        }
    }
    // The masked view without the seat's own discards: refused exactly when it burned a card.
    let j = app_match(m, seat, Mask::Seat);
    let burned = want.own_burns != 0;
    match AppTable::read(&j, seat, None).and_then(|t| rebuild(&t)) {
        Ok(r) => {
            assert!(!burned);
            assert!(same_view(&r.view, &want));
        }
        Err(e) if burned && e.code == "own_burns_hidden" => counts[1] += 1,
        Err(e) if hidden_own && e.code == "inconsistent_view" => {}
        Err(e) => panic!("refused {e:?}"),
    }
    if any_turned(m) {
        counts[2] += 1;
    }
}

#[test]
fn the_rebuilt_view_is_the_engines_own() {
    let mut counts = [0u32; 4];
    each_decision(1, 12, 0, |m| check_rebuild(m, &mut counts));
    assert!(counts[0] > 1000 && counts[1] > 50, "{counts:?}");
}

#[test]
fn the_rebuilt_view_is_the_engines_own_after_the_referee() {
    let mut counts = [0u32; 4];
    each_decision(2, 40, 5, |m| check_rebuild(m, &mut counts));
    assert!(counts[2] > 300, "positions with a turned pair: {counts:?}");
}

/// The values the native search gives the seat's own view, as the API reports them: per
/// candidate (engine order) the mean over deals `0..n` of `exact_world` (with the frontier cut,
/// when given), drawn from the same seed.
fn native_means(
    m: &Match,
    n: u64,
    seed: u64,
    net: Option<&Oracle>,
) -> Vec<(engine::game::Action, Option<u8>, MoneyChance)> {
    let seat = m.to_act().expect("a decision");
    let view = m.view(seat);
    let (_, cands) = candidates(&view, &m.legal_actions());
    let leaf = leaf();
    let source = BeliefSource::new();
    let ctx = Ctx {
        leaf: &*leaf,
        source: &source,
        rules: Rules::HOUSE,
    };
    let cfg = SearchConfig {
        tt_mb: 16,
        ..SearchConfig::default()
    };
    let worlds = source.prepare(&view, Rules::HOUSE);
    let cut = net.map(|o| FrontierCut::new(view.public.trick_no + 1, o));
    let mut mem = SolveMem::new();
    let rows: Vec<Vec<MoneyChance>> = (0..n)
        .map(|w| {
            let wm = world_match(&view, &*worlds, w, seed, Rules::HOUSE);
            exact_world(
                &wm,
                &cands,
                &ctx,
                &cfg,
                None,
                &mut mem,
                &mut WorldSolve::default(),
                cut,
            )
            .expect("solved")
        })
        .collect();
    let (means, _, _) = aggregate(&rows, cands.len(), team_of(seat));
    cands
        .iter()
        .zip(means)
        .map(|(&(a, p), v)| (a, p, v))
        .collect()
}

/// The API's candidates, ungrouped, with their mean in fixed point.
fn api_means(result: &Value) -> Vec<(engine::game::Action, Option<u8>, MoneyChance)> {
    let mut out = Vec::new();
    for c in result["candidates"].as_array().expect("candidates") {
        let fp: Vec<u64> = c["mean_fp"]
            .as_array()
            .expect("means")
            .iter()
            .map(|x| x.as_str().expect("a string").parse().expect("a number"))
            .collect();
        let v = MoneyChance([fp[0], fp[1], fp[2], fp[3]]);
        for e in c["equivalent"].as_array().expect("a list") {
            let card = |x: &Value| engine_card(x.as_i64().expect("a card")).expect("a card");
            let a = &e["action"];
            let act = match a["type"].as_str() {
                Some("play") => engine::game::Action::Play(card(&a["card"])),
                _ => engine::game::Action::Burn(card(&a["discard"])),
            };
            let then = e.get("then").map(|t| card(&t["card"]));
            out.push((act, then, v));
        }
    }
    out.sort_by_key(|x| (x.0, x.1));
    out
}

#[test]
fn values_are_the_native_searchs() {
    let e = engine(true);
    let o3 = Oracle::from_bytes(&net_bytes(3), leaf()).expect("loads");
    let o2 = Oracle::from_bytes(&net_bytes(2), leaf()).expect("loads");
    let plain = |m: &Match| !any_turned(m) && m.reneges().is_empty();
    let cases: Vec<(Match, u32, Option<&Oracle>)> = vec![
        (find(31, 0, &|m| plain(m) && playing(m, 2)), 3, Some(&o3)),
        (find(32, 0, &|m| plain(m) && playing(m, 3)), 12, None),
        (find(33, 0, &|m| plain(m) && playing(m, 4)), 20, None),
        (
            find(34, 3, &|m| {
                any_turned(m) && playing(m, 3) && m.reneges().iter().all(|r| r.caught)
            }),
            8,
            None,
        ),
        (
            find(35, 0, &|m| {
                plain(m) && playing(m, 1) && m.public().trick_cards.len() >= 6
            }),
            2,
            Some(&o2),
        ),
        (
            find(36, 0, &|m| {
                let p = m.public();
                matches!(p.phase, Phase::AwaitBurn(_))
                    && p.trick_no == 1
                    && p.trick_cards.len() >= 6
            }),
            1,
            Some(&o2),
        ),
    ];
    for (k, (m, deals, net)) in cases.iter().enumerate() {
        let seed = 900 + k as u64;
        let req = app_request(m, *deals, seed, Mask::SeatOwnBurns);
        let out = e.evaluate(&req.to_string());
        let v = parse(&out);
        assert_eq!(v["status"], "ok", "case {k}: {out}");
        let mut native = native_means(m, u64::from(*deals), seed, *net);
        // A burn of the card just taken is never listed.
        let p = m.public();
        if matches!(p.phase, Phase::AwaitBurn(_)) {
            let taken = p.turned.last().map(|&(c, _)| c);
            native.retain(
                |&(a, _, _)| !(matches!(a, engine::game::Action::Burn(c) if Some(c) == taken)),
            );
        }
        native.sort_by_key(|x| (x.0, x.1));
        assert_eq!(api_means(&v), native, "case {k}");
        // The best is the native search's choice: the first best by the acting pair's value.
        let team = team_of(m.to_act().expect("a decision"));
        let best = native
            .iter()
            .max_by(|a, b| a.2.get(team).cmp(&b.2.get(team)).then(b.0.cmp(&a.0)))
            .expect("candidates");
        let first = &v["candidates"][0]["action"];
        let card = first["card"]
            .as_i64()
            .or(first["discard"].as_i64())
            .expect("a card");
        assert_eq!(engine_card(card), Some(best.0.card()), "case {k}: the best");
    }
}

#[test]
fn exact_values_are_pimc_evaluates() {
    // From trick 3 the native entry point itself: `search::pimc::evaluate`, exact PIMC.
    let e = engine(false);
    for (k, trick) in [(41u64, 3u8), (42, 4), (43, 3)] {
        let m = find(k, 0, &|m| !any_turned(m) && playing(m, trick));
        let seat = m.to_act().expect("a decision");
        let view = m.view(seat);
        let leaf = leaf();
        let source = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*leaf,
            source: &source,
            rules: Rules::HOUSE,
        };
        let cfg = SearchConfig {
            w34: 10,
            ..SearchConfig::default()
        };
        let dv = search::pimc::evaluate(&view, &m.legal_actions(), 77, &ctx, &cfg);
        let out = e.evaluate(&app_request(&m, 10, 77, Mask::SeatOwnBurns).to_string());
        let mut native: Vec<_> = dv.candidates.clone();
        native.sort_by_key(|x| (x.0, x.1));
        assert_eq!(api_means(&parse(&out)), native, "trick {trick}");
    }
}

#[test]
fn shards_merge_to_the_same_bytes() {
    let e = engine(false);
    let plain = |m: &Match| !any_turned(m) && m.reneges().is_empty();
    for (m, deals) in [
        (find(51, 0, &|m| plain(m) && playing(m, 2)), 7u32),
        (find(52, 0, &|m| plain(m) && playing(m, 3)), 14),
        (
            find(53, 4, &|m| {
                any_turned(m) && playing(m, 3) && m.reneges().iter().all(|r| r.caught)
            }),
            9,
        ),
        (find(54, 0, &|m| m.public().trick_no == 5), 5),
    ] {
        let mut req = app_request(&m, deals, 4242, Mask::SeatOwnBurns);
        let whole = e.evaluate(&req.to_string());
        assert_eq!(parse(&whole)["status"], "ok", "{whole}");
        assert_eq!(merge(&format!("[{whole}]")), whole);
        for k in [1u32, 2, 3, 4, 7] {
            let k = k.min(deals);
            let mut parts = Vec::new();
            for i in (0..k).rev() {
                req["shard"] = json!(i);
                req["shards"] = json!(k);
                parts.push(e.evaluate(&req.to_string()));
            }
            assert_eq!(
                merge(&format!("[{}]", parts.join(","))),
                whole,
                "{k} shards"
            );
            // Results may also be passed as strings.
            let as_strings = serde_json::to_string(&parts).expect("json");
            assert_eq!(merge(&as_strings), whole, "{k} shards as strings");
            if k > 1 {
                assert_eq!(
                    code(&merge(&format!("[{}]", parts[1..].join(",")))),
                    "merge_mismatch"
                );
                let twice = format!("[{},{}]", parts.join(","), parts[0]);
                assert_eq!(code(&merge(&twice)), "merge_mismatch");
            }
        }
        req.as_object_mut().expect("an object").remove("shard");
        req.as_object_mut().expect("an object").remove("shards");
    }
    // Results of different searches never merge.
    let a = e.evaluate(
        &app_request(&find(52, 0, &|m| playing(m, 3)), 4, 1, Mask::SeatOwnBurns).to_string(),
    );
    let b = e.evaluate(
        &app_request(&find(52, 0, &|m| playing(m, 3)), 4, 2, Mask::SeatOwnBurns).to_string(),
    );
    assert_eq!(code(&merge(&format!("[{a},{b}]"))), "merge_mismatch");
}

#[test]
fn the_unmasked_match_gives_the_same_bytes() {
    let e = engine(false);
    for (seed, trick) in [(61u64, 2u8), (62, 3), (63, 4)] {
        let m = find(seed, 0, &|m| playing(m, trick));
        let a = e.evaluate(&app_request(&m, 3, 5, Mask::SeatOwnBurns).to_string());
        let b = e.evaluate(&app_request(&m, 3, 5, Mask::Open).to_string());
        assert_eq!(a, b);
    }
}

#[test]
fn actions_map_both_ways() {
    let e = engine(true);
    // A robber's rob at its turn in trick 1 (with O_2): every candidate is a rob from its five
    // cards, never of the card it is taking, with the card it then plays.
    let m = find(71, 0, &|m| {
        let p = m.public();
        matches!(p.phase, Phase::AwaitBurn(_)) && p.trick_no == 1 && p.trick_cards.len() >= 7
    });
    let seat = m.to_act().expect("a decision");
    let taken = m.public().turned.last().expect("turned").0;
    let req = app_request(&m, 1, 3, Mask::SeatOwnBurns);
    // The app's view: the turned card still on the table, five cards in hand.
    assert_eq!(req["match"]["hand"]["turned"], json!(app_card(taken)));
    assert_eq!(
        req["match"]["hand"]["hands"][usize::from(seat)]
            .as_array()
            .map(Vec::len),
        Some(5)
    );
    let v = parse(&e.evaluate(&req.to_string()));
    assert_eq!(v["status"], "ok", "{v}");
    let mut burns = std::collections::BTreeSet::new();
    for c in v["candidates"].as_array().expect("candidates") {
        for x in c["equivalent"].as_array().expect("a list") {
            assert_eq!(x["action"]["type"], "rob");
            assert_eq!(x["then"]["type"], "play");
            let d = x["action"]["discard"].as_i64().expect("a card");
            assert_ne!(
                engine_card(d),
                Some(taken),
                "the taken card is never burned"
            );
            burns.insert(d);
        }
    }
    let hand: Vec<i64> = req["match"]["hand"]["hands"][usize::from(seat)]
        .as_array()
        .expect("cards")
        .iter()
        .map(|c| c.as_i64().expect("a card"))
        .collect();
    let ace = engine::rules::ace_of_trumps(m.public().trump.expect("trumps"));
    let want: std::collections::BTreeSet<i64> = hand
        .iter()
        .copied()
        .filter(|&c| engine_card(c) != Some(ace))
        .collect();
    assert_eq!(burns, want, "a rob of each card but the ace of trumps");
    // Plays: the app's card numbers.
    let m = find(72, 0, &|m| playing(m, 3));
    let v = parse(&e.evaluate(&app_request(&m, 2, 3, Mask::SeatOwnBurns).to_string()));
    let mut got: Vec<i64> = v["candidates"]
        .as_array()
        .expect("candidates")
        .iter()
        .flat_map(|c| c["equivalent"].as_array().expect("a list").clone())
        .map(|x| {
            assert_eq!(x["action"]["type"], "play");
            x["action"]["card"].as_i64().expect("a card")
        })
        .collect();
    got.sort_unstable();
    let mut want: Vec<i64> = m
        .legal_actions()
        .iter()
        .map(|a| i64::from(app_card(a.card())))
        .collect();
    want.sort_unstable();
    assert_eq!(got, want);
}

#[test]
fn refusals_have_their_codes() {
    let e = engine(false);
    let m3 = find(81, 0, &|m| !any_turned(m) && playing(m, 3));
    let base = app_request(&m3, 2, 1, Mask::SeatOwnBurns);
    let ev = |r: &Value| code(&e.evaluate(&r.to_string()));
    let with = |path: &[&str], v: Value| {
        let mut r = base.clone();
        let mut x = &mut r;
        for p in &path[..path.len() - 1] {
            x = &mut x[*p];
        }
        x[path[path.len() - 1]] = v;
        r
    };
    assert_eq!(code(&e.evaluate("not json")), "bad_request");
    assert_eq!(code(&e.evaluate("[]")), "bad_request");
    assert_eq!(ev(&with(&["deals"], json!(0))), "bad_request");
    assert_eq!(ev(&with(&["deals"], json!(5000))), "bad_request");
    assert_eq!(ev(&with(&["seed"], json!("x"))), "bad_request");
    assert_eq!(
        ev(&with(&["seed"], json!(9_007_199_254_740_993u64))),
        "bad_request"
    );
    assert_eq!(ev(&with(&["v"], json!(2))), "bad_request");
    assert_eq!(
        ev(&with(&["shards"], json!(3))),
        "bad_request",
        "more shards than deals"
    );
    assert_eq!(
        ev(&with(&["match", "config", "n"], json!(6))),
        "not_8_seats"
    );
    assert_eq!(
        ev(&with(&["match", "config", "jokers"], json!(2))),
        "two_jokers"
    );
    assert_eq!(
        ev(&with(&["match", "config", "target"], json!(31))),
        "not_house_rules"
    );
    assert_eq!(
        ev(&with(
            &["match", "config", "mustTrumpWhenVoid"],
            json!(true)
        )),
        "not_house_rules"
    );
    assert_eq!(
        ev(&with(&["match", "status"], json!("gameOver"))),
        "not_playing"
    );
    assert_eq!(
        ev(&with(&["match", "hand", "phase"], json!("over"))),
        "not_playing"
    );
    let other = (base["seat"].as_u64().expect("a seat") + 1) % 8;
    assert_eq!(ev(&with(&["seat"], json!(other))), "not_to_act");
    assert_eq!(
        ev(&with(&["match", "hand", "toAct"], json!(other))),
        "not_to_act"
    );
    // A card the seat holds shown as played elsewhere, a hand of the wrong size.
    let own =
        base["match"]["hand"]["hands"][base["seat"].as_u64().expect("seat") as usize][0].clone();
    let mut r = base.clone();
    r["match"]["hand"]["tricks"][0]["plays"][0]["card"] = own;
    assert_eq!(ev(&r), "inconsistent_view");
    let mut r = base.clone();
    r["match"]["hand"]["hands"][usize::from((m3.to_act().expect("seat") + 1) % 8)] = json!([-1]);
    assert_eq!(ev(&r), "inconsistent_view");
    // The second Joker.
    let mut r = base.clone();
    r["match"]["hand"]["tricks"][0]["plays"][0]["card"] = json!(53);
    assert_eq!(ev(&r), "two_jokers");

    // No equity table, no network.
    let bare = Engine::new();
    assert_eq!(code(&bare.evaluate(&base.to_string())), "no_equity");
    let m2 = find(82, 0, &|m| !any_turned(m) && playing(m, 2));
    let no_net = Engine::with_leaf(leaf(), "synthetic");
    assert_eq!(
        code(&no_net.evaluate(&app_request(&m2, 2, 1, Mask::SeatOwnBurns).to_string())),
        "no_network_for_trick"
    );
    let m1 = find(83, 0, &|m| !any_turned(m) && playing(m, 1));
    assert_eq!(
        code(&e.evaluate(&app_request(&m1, 2, 1, Mask::SeatOwnBurns).to_string())),
        "no_network_for_trick"
    );
    // A forced card needs nothing loaded.
    let mf = find(84, 0, &|m| m.public().trick_no == 5);
    let v = parse(&bare.evaluate(&app_request(&mf, 2, 1, Mask::SeatOwnBurns).to_string()));
    assert_eq!(
        (
            v["status"].as_str(),
            v["method"].as_str(),
            v["worlds"].as_u64()
        ),
        (Some("ok"), Some("forced"), Some(0))
    );
    // The dealer's burn before trick 1.
    let m0 = find(85, 0, &|m| {
        let p = m.public();
        matches!(p.phase, Phase::AwaitBurn(_)) && p.trick_no == 0 && m.legal_actions().len() > 1
    });
    assert_eq!(
        code(&e.evaluate(&app_request(&m0, 2, 1, Mask::SeatOwnBurns).to_string())),
        "not_built"
    );
    // Trick 1 with six seats in play.
    let e2 = engine(true);
    let m6 = find(86, 3, &|m| {
        m.turned_teams().iter().filter(|&&t| t).count() == 1
            && playing(m, 1)
            && m.reneges().iter().all(|r| r.caught)
    });
    assert_eq!(
        code(&e2.evaluate(&app_request(&m6, 2, 1, Mask::SeatOwnBurns).to_string())),
        "too_slow"
    );
    // The seat's own burn hidden.
    let mb = find(87, 0, &|m| {
        m.view(m.to_act().expect("seat")).own_burns != 0 && playing(m, 2)
    });
    assert_eq!(
        code(&e.evaluate(&app_request(&mb, 2, 1, Mask::Seat).to_string())),
        "own_burns_hidden"
    );
    let mut r = app_request(&mb, 2, 1, Mask::Seat);
    let seat = mb.to_act().expect("seat");
    let burn = engine::cards::cards(mb.view(seat).own_burns)
        .next()
        .expect("a burn");
    r["ownBurns"] = json!([app_card(burn)]);
    assert_eq!(
        e.evaluate(&r.to_string()),
        e.evaluate(&app_request(&mb, 2, 1, Mask::SeatOwnBurns).to_string())
    );
}

#[test]
fn loading_checks_the_files() {
    let mut e = Engine::new();
    assert_eq!(
        e.load_net(&net_bytes(3)).map_err(|r| r.code),
        Err("no_equity")
    );
    assert_eq!(
        e.load_equity(b"nonsense").map_err(|r| r.code),
        Err("bad_file")
    );
    let mut e = Engine::with_leaf(leaf(), "synthetic");
    assert_eq!(e.load_net(b"nonsense").map_err(|r| r.code), Err("bad_file"));
    let (sha, trick) = e.load_net(&net_bytes(3)).expect("loads");
    assert_eq!((sha.len(), trick), (64, 3));
    assert_eq!(sha, equity::sha256::sha256_hex(&net_bytes(3)));
    let info = parse(&e.info(None));
    assert_eq!(info["nets"]["3"], json!(sha));
}

#[test]
fn the_app_example_reads() {
    // The trick-2 example of docs/WASM-API.md §3, as the app would send it.
    let doc = include_str!("../../../docs/WASM-API.md");
    let start = doc
        .find("```json\n{\n  \"v\": 1, \"seat\": 7")
        .expect("the example")
        + 8;
    let end = start + doc[start..].find("```").expect("its end");
    let req: Value = serde_json::from_str(&doc[start..end]).expect("the example is JSON");
    let t = AppTable::read(&req["match"], 7, None).expect("reads");
    let r = rebuild(&t).expect("rebuilds");
    assert_eq!(r.view.public.trick_no, 2);
    assert_eq!(r.m.to_act(), Some(7));
    let legal: Vec<u8> =
        r.m.legal_actions()
            .iter()
            .map(|a| app_card(a.card()))
            .collect();
    let mut legal = legal;
    legal.sort_unstable();
    assert_eq!(legal, vec![25, 31, 35], "K♦, 6♣ and 10♣");
    let v = parse(&engine(false).evaluate(&req.to_string()));
    assert_eq!(v["status"], "ok", "{v}");
    assert_eq!(v["method"], "hybrid");
    assert_eq!(v["meanHorizon"], 1);
    // The robber's turned card is known to be in its hand, not anywhere else.
    assert!(matches!(
        hand_events(&r.view)
            .iter()
            .find(|e| matches!(e, Event::Took { .. })),
        Some(Event::Took { seat: 2, .. })
    ));
}

#[test]
fn every_refusal_code_is_listed() {
    // Every code the sources can refuse with is in the documented list (docs/WASM-API.md §5).
    let src = [
        include_str!("../src/api.rs"),
        include_str!("../src/appview.rs"),
        include_str!("../src/ffi.rs"),
    ]
    .concat();
    let doc = include_str!("../../../docs/WASM-API.md");
    let mut found = 0;
    for (i, _) in src.match_indices("Refusal::new(") {
        let rest = src[i + "Refusal::new(".len()..].trim_start();
        let Some(rest) = rest.strip_prefix('"') else {
            continue;
        };
        let code = &rest[..rest.find('"').expect("a closing quote")];
        assert!(
            wasm::api::REFUSAL_CODES.contains(&code),
            "{code} is not listed"
        );
        assert!(
            doc.contains(&format!("`{code}`")),
            "{code} is not in docs/WASM-API.md"
        );
        found += 1;
    }
    assert!(found >= 15, "{found}");
}

#[test]
#[ignore = "long: WASM_API_MATCHES=400 cargo test --release -p wasm --test api -- --ignored"]
fn the_rebuilt_view_is_the_engines_own_long() {
    let n: u64 = std::env::var("WASM_API_MATCHES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    let mut counts = [0u32; 4];
    for (seed, rate) in [(100, 0), (200, 3), (300, 6), (400, 12)] {
        each_decision(seed, n, rate, |m| check_rebuild(m, &mut counts));
    }
    eprintln!(
        "rebuilt {} views; {} own burns hidden; {} with a pair turned; {} own lapses refused",
        counts[0], counts[1], counts[2], counts[3]
    );
}

/// Positions for the lines' tests: trick 2 (hybrid), 3 and 4 (exact), a turned pair, trick 1
/// with O_2 (a play and a robber's rob).
fn line_cases() -> Vec<(Match, u32)> {
    let plain = |m: &Match| !any_turned(m) && m.reneges().is_empty();
    vec![
        (find(91, 0, &|m| plain(m) && playing(m, 2)), 3),
        (find(92, 0, &|m| plain(m) && playing(m, 3)), 6),
        (find(93, 0, &|m| plain(m) && playing(m, 4)), 8),
        (
            find(94, 4, &|m| {
                any_turned(m) && playing(m, 3) && m.reneges().iter().all(|r| r.caught)
            }),
            5,
        ),
        (
            find(95, 0, &|m| {
                plain(m) && playing(m, 1) && m.public().trick_cards.len() >= 6
            }),
            2,
        ),
        (
            find(96, 0, &|m| {
                let p = m.public();
                matches!(p.phase, Phase::AwaitBurn(_))
                    && p.trick_no == 1
                    && p.trick_cards.len() >= 6
            }),
            1,
        ),
    ]
}

#[test]
fn each_lines_end_is_its_deals_value() {
    // The principal line ends where the search's value comes from: the leaf after the hand, or
    // the network's value at the frontier, equal to the candidate's value in that deal.
    let o3 = Oracle::from_bytes(&net_bytes(3), leaf()).expect("loads");
    let o2 = Oracle::from_bytes(&net_bytes(2), leaf()).expect("loads");
    let leaf = leaf();
    for (k, (m, deals)) in line_cases().iter().enumerate() {
        let seat = m.to_act().expect("a decision");
        let view = m.view(seat);
        let p = &view.public;
        let net = match p.trick_no {
            1 if !any_turned(m) => Some((2u8, &o2)),
            2 if !any_turned(m) => Some((3u8, &o3)),
            _ => None,
        };
        let (_, cands) = candidates(&view, &m.legal_actions());
        let source = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*leaf,
            source: &source,
            rules: Rules::HOUSE,
        };
        let cfg = SearchConfig::default();
        let worlds = source.prepare(&view, Rules::HOUSE);
        let walker = wasm::lines::Walker {
            leaf: &*leaf,
            cut: net,
            tt_mb: 16,
        };
        let mut mem = SolveMem::new();
        for w in 0..u64::from(*deals) {
            let wm = world_match(&view, &*worlds, w, 5 + k as u64, Rules::HOUSE);
            let cut = net.map(|(t, o)| FrontierCut::new(t, o));
            let row = exact_world(
                &wm,
                &cands,
                &ctx,
                &cfg,
                None,
                &mut mem,
                &mut WorldSolve::default(),
                cut,
            )
            .expect("solved");
            for (j, &cand) in cands.iter().enumerate() {
                let l = wasm::lines::principal_line(&wm, cand, &walker, &mut mem);
                assert_eq!(
                    l.end_value, row[j],
                    "case {k}, deal {w}, candidate {cand:?}"
                );
                // The candidate's own card is played in the current trick.
                let own = cand.1.unwrap_or(cand.0.card());
                let t0 = p.trick_no.max(1) - 1;
                assert_eq!(l.plays.first(), Some(&(own, t0)), "case {k}");
            }
        }
    }
}

#[test]
fn lines_merge_to_the_same_bytes_and_add_up() {
    let e = engine(true);
    for (k, (m, deals)) in line_cases().into_iter().enumerate() {
        let mut req = app_request(&m, deals, 31 + k as u64, Mask::SeatOwnBurns);
        req["lines"] = json!(true);
        let whole = e.evaluate(&req.to_string());
        let v = parse(&whole);
        assert_eq!(v["status"], "ok", "case {k}: {whole}");
        let seat = m.to_act().expect("a decision");
        let n = f64::from(deals);
        let hand: Vec<String> = engine::cards::cards(m.view(seat).hand)
            .map(|c| app_card(c).to_string())
            .collect();
        for c in v["candidates"].as_array().expect("candidates") {
            // One row per action with the lines.
            assert_eq!(c["equivalent"][0]["action"], c["action"], "case {k}");
            let winners: Vec<f64> = c["trickWinner"]
                .as_array()
                .expect("8")
                .iter()
                .map(|x| x.as_f64().expect("a share"))
                .collect();
            assert_eq!(winners.len(), 8);
            let total: f64 = winners.iter().sum();
            assert!(total <= 1.0 + 1e-9, "case {k}");
            let pair: f64 = (0..8)
                .filter(|&s| s % 4 == seat % 4)
                .map(|s| winners[s as usize])
                .sum();
            assert!((pair - c["trickWin"].as_f64().expect("a share")).abs() < 1e-9);
            for x in c["expectedPoints"].as_array().expect("4") {
                let x = x.as_f64().expect("points");
                assert!((0.0..=41.0).contains(&x), "case {k}: {x}");
            }
            let g = c["gameEnds"].as_f64().expect("a share");
            assert!((0.0..=1.0).contains(&g));
            // The plan: a row per card of the seat's hand; the candidate's card in this trick in
            // every deal; no card in two tricks more often than there are deals.
            let plan = c["plan"].as_object().expect("the plan");
            let mut keys: Vec<&String> = plan.keys().collect();
            keys.sort_by_key(|x| x.parse::<u8>().expect("a card"));
            let mut want = hand.clone();
            want.sort_by_key(|x| x.parse::<u8>().expect("a card"));
            assert_eq!(keys, want.iter().collect::<Vec<_>>(), "case {k}");
            let own = c.get("then").map_or(&c["action"]["card"], |t| &t["card"]);
            let t0 = (m.public().trick_no.max(1) - 1) as usize;
            assert_eq!(plan[&own.to_string()][t0].as_f64(), Some(1.0), "case {k}");
            for row in plan.values() {
                let s: f64 = row
                    .as_array()
                    .expect("5")
                    .iter()
                    .map(|x| x.as_f64().expect("a share"))
                    .sum();
                assert!(s <= 1.0 + 1e-9, "case {k}");
            }
            let _ = n;
        }
        for kk in [2u32, 3, 4, 7] {
            let kk = kk.min(deals);
            let mut parts = Vec::new();
            for i in (0..kk).rev() {
                req["shard"] = json!(i);
                req["shards"] = json!(kk);
                parts.push(e.evaluate(&req.to_string()));
            }
            assert_eq!(
                merge(&format!("[{}]", parts.join(","))),
                whole,
                "case {k}, {kk} shards"
            );
        }
        // Results with and without the lines never merge.
        req.as_object_mut().expect("an object").remove("shard");
        req.as_object_mut().expect("an object").remove("shards");
        req["lines"] = json!(false);
        let plain = e.evaluate(&req.to_string());
        assert_eq!(
            code(&merge(&format!("[{plain},{whole}]"))),
            "merge_mismatch"
        );
    }
}
