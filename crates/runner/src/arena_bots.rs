//! The bots of an `arena_blocks` stream (docs/STUDENT.md §6, "The app's player in the arena").
//!
//! `params.a` and `params.b` are each a bot name of the registry (`search::factory`: `pimc-v0[:…]`,
//! `rules-v1`, `random`) or an object describing the app's player (`search::bot::AppPlay`):
//!
//! ```json
//! { "kind": "app", "student1": "s1new", "student2": "s2new",
//!   "deals3": [64, 64, 128, 128, 256, 256, 256, 256], "deals45": 256, "base": "pimc-v0" }
//! ```
//!
//! `student1` and `student2` name the students' weights files among the manifest's `inputs`
//! (fetched by SHA-256 like every input), `deals3` the exact deals at trick 3 by place in the trick
//! (the leader first; the app's own table when absent), `deals45` the deals at tricks 4 and 5 (the
//! app's 1024 when absent) and `base` the `pimc-v0[:…]` name that plays what the students do not
//! (the dealer's burn, a turned pair at tricks 1-2) and sets the exact solves' table size (`tt_mb`,
//! 64 by default). `"inference": {"tau": 10, "floor": 50}` (permille of the money) weights the
//! exact searches' deals by the other seats' plays under the students (`search::inference`; absent:
//! uniform deals, as the app plays today). `student3` (an input name; docs/STUDENT.md §6.3) values
//! each trick-3 deal by a playout in which the other seats answer with the trick-3 student from
//! their own view, exact search from trick 4 (absent: the open-hand solve, as the app plays today).
//! The recorded bot name is the factory's (`app:s1=…,s2=…,deals3=…,deals45=…[,inf=tau…/floor…]
//! [,s3=…]`).

use std::collections::BTreeMap;

use bot::BotFactory;
use search::bot::{AppPlay, InferenceCfg};
use serde_json::Value;

/// One side of a block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum BotSpec {
    /// A registry name.
    Name(String),
    /// The app's player.
    App {
        /// The input names of the students' weights files.
        student1: String,
        student2: String,
        /// Exact deals at trick 3 by place.
        deals3: [u32; 8],
        /// Exact deals at tricks 4 and 5.
        deals45: u32,
        /// The `pimc-v0[:…]` name for what the students do not play.
        base: String,
        /// The deal weighting, if any.
        inference: Option<InferenceCfg>,
        /// The trick-3 student's input name, if any.
        student3: Option<String>,
    },
}

/// The keys an app bot object may carry.
const APP_KEYS: [&str; 8] = [
    "kind",
    "student1",
    "student2",
    "student3",
    "deals3",
    "deals45",
    "base",
    "inference",
];

/// `params.<side>` parsed and checked (the shape only; the inputs' presence is the stream's
/// check, `input_names`).
pub(crate) fn parse_bot(params: &Value, side: &str) -> Result<BotSpec, String> {
    let v = params
        .get(side)
        .ok_or_else(|| format!("arena_blocks needs params.{side}"))?;
    if let Some(name) = v.as_str() {
        if name.is_empty() {
            return Err(format!("params.{side}: a bot name is not empty"));
        }
        return Ok(BotSpec::Name(name.to_string()));
    }
    let o = v
        .as_object()
        .ok_or_else(|| format!("params.{side} is a bot name or an app bot object"))?;
    if o.get("kind").and_then(Value::as_str) != Some("app") {
        return Err(format!(
            "params.{side}: an object bot has \"kind\": \"app\""
        ));
    }
    for k in o.keys() {
        if !APP_KEYS.contains(&k.as_str()) {
            return Err(format!(
                "params.{side}: unknown key `{k}` (an app bot takes {})",
                APP_KEYS.join(", ")
            ));
        }
    }
    let input = |k: &str| -> Result<String, String> {
        o.get(k)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .ok_or_else(|| format!("params.{side}: an app bot names its {k} (an input name)"))
    };
    let student1 = input("student1")?;
    let student2 = input("student2")?;
    let student3 = match o.get("student3") {
        None | Some(Value::Null) => None,
        Some(_) => Some(input("student3")?),
    };
    let deals45 = match o.get("deals45") {
        None => AppPlay::APP_DEALS45,
        Some(d) => deal_count(d, &format!("params.{side}.deals45"))?,
    };
    let deals3 = match o.get("deals3") {
        None => AppPlay::APP_DEALS3,
        Some(Value::Array(a)) if a.len() == 8 => {
            let mut out = [0u32; 8];
            for (i, d) in a.iter().enumerate() {
                out[i] = deal_count(d, &format!("params.{side}.deals3[{i}]"))?;
            }
            out
        }
        Some(_) => {
            return Err(format!(
                "params.{side}.deals3 is a list of 8 deal counts, one per place in the trick"
            ));
        }
    };
    let base = match o.get("base") {
        None => "pimc-v0".to_string(),
        Some(b) => b
            .as_str()
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .ok_or_else(|| format!("params.{side}.base is a pimc-v0[:…] name"))?,
    };
    search::bot::PimcSpec::parse(&base).map_err(|e| format!("params.{side}.base: {e}"))?;
    let inference = match o.get("inference") {
        None | Some(Value::Null) => None,
        Some(Value::Object(i)) => {
            for k in i.keys() {
                if !["tau", "floor"].contains(&k.as_str()) {
                    return Err(format!(
                        "params.{side}.inference: unknown key `{k}` (tau and floor, permille of the money)"
                    ));
                }
            }
            let pm = |k: &str, default: u32| -> Result<u32, String> {
                match i.get(k) {
                    None => Ok(default),
                    Some(v) => v
                        .as_u64()
                        .filter(|&n| n <= 1000)
                        .map(|n| n as u32)
                        .ok_or_else(|| {
                            format!(
                                "params.{side}.inference.{k}: a whole number of permille, 0 to 1000"
                            )
                        }),
                }
            };
            Some(
                InferenceCfg::from_permille(pm("tau", 10)?, pm("floor", 50)?)
                    .map_err(|e| format!("params.{side}.inference: {e}"))?,
            )
        }
        Some(_) => {
            return Err(format!(
                "params.{side}.inference is an object {{\"tau\": permille, \"floor\": permille}}"
            ));
        }
    };
    Ok(BotSpec::App {
        student1,
        student2,
        deals3,
        deals45,
        base,
        inference,
        student3,
    })
}

fn deal_count(v: &Value, what: &str) -> Result<u32, String> {
    v.as_u64()
        .filter(|&n| (1..=1_000_000).contains(&n))
        .map(|n| n as u32)
        .ok_or_else(|| format!("{what}: a deal count is a whole number from 1 to 1000000"))
}

/// Both sides of the stream.
pub(crate) fn parse_bots(params: &Value) -> Result<(BotSpec, BotSpec), String> {
    Ok((parse_bot(params, "a")?, parse_bot(params, "b")?))
}

/// The inputs the stream's bots name (the students' weights files), in order, without repeats.
pub(crate) fn input_names(params: &Value) -> Result<Vec<String>, String> {
    let (a, b) = parse_bots(params)?;
    let mut out: Vec<String> = Vec::new();
    for spec in [a, b] {
        if let BotSpec::App {
            student1,
            student2,
            student3,
            ..
        } = spec
        {
            for n in [Some(student1), Some(student2), student3]
                .into_iter()
                .flatten()
            {
                if !out.contains(&n) {
                    out.push(n);
                }
            }
        }
    }
    Ok(out)
}

/// The factory of one side: the registry's for a name, the app's player built from the inputs'
/// files (`inputs`: input name to local path) for an app bot.
pub(crate) fn build(
    spec: &BotSpec,
    inputs: &BTreeMap<String, String>,
) -> Result<Box<dyn BotFactory + Send + Sync>, String> {
    match spec {
        BotSpec::Name(name) => {
            search::factory(name).ok_or_else(|| search::bot::factory_error(name))
        }
        BotSpec::App {
            student1,
            student2,
            deals3,
            deals45,
            base,
            inference,
            student3,
        } => {
            let read = |name: &str| -> Result<Vec<u8>, String> {
                let path = inputs
                    .get(name)
                    .ok_or_else(|| format!("app bot: input `{name}` is not among the inputs"))?;
                std::fs::read(path).map_err(|e| format!("app bot: input `{name}` ({path}): {e}"))
            };
            let (s1, s2) = (read(student1)?, read(student2)?);
            let s3 = student3.as_deref().map(read).transpose()?;
            let f = search::bot::app_factory(
                base,
                &s1,
                &s2,
                s3.as_deref(),
                *deals3,
                *deals45,
                *inference,
            )
            .map_err(|e| format!("app bot ({student1}, {student2}): {e}"))?;
            Ok(Box::new(f))
        }
    }
}

/// The bytes one chunk of the stream is planned to hold: the equity table, two students and the
/// exact solves' transposition table of an app bot (`base`'s `tt_mb`), plus a small process; a
/// chunk of registry bots keeps the old estimate of nothing (small).
pub(crate) fn chunk_bytes(params: &Value, equity_bytes: u64) -> u64 {
    const MIB: u64 = 1 << 20;
    let Ok((a, b)) = parse_bots(params) else {
        return 0;
    };
    let tt = |spec: &BotSpec| match spec {
        BotSpec::App { base, .. } => search::bot::PimcSpec::parse(base)
            .map(|s| s.cfg.tt_mb as u64)
            .unwrap_or(64),
        BotSpec::Name(_) => 0,
    };
    let apps = [&a, &b]
        .iter()
        .filter(|s| matches!(s, BotSpec::App { .. }))
        .count() as u64;
    if apps == 0 {
        return 0;
    }
    // The bots of one side share their students (one factory); the table is per exact solve,
    // one at a time in a chunk's single thread.
    equity_bytes + apps * 16 * MIB + tt(&a).max(tt(&b)) * MIB + 64 * MIB
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn app(extra: Value) -> Value {
        let mut o = json!({ "kind": "app", "student1": "s1", "student2": "s2" });
        if let (Some(a), Some(b)) = (o.as_object_mut(), extra.as_object()) {
            for (k, v) in b {
                a.insert(k.clone(), v.clone());
            }
        }
        o
    }

    #[test]
    fn names_and_app_bots_parse_with_the_apps_deals_as_defaults() {
        let p = json!({ "a": app(json!({})), "b": "pimc-v0" });
        let (a, b) = parse_bots(&p).expect("parses");
        assert_eq!(b, BotSpec::Name("pimc-v0".into()));
        assert_eq!(
            a,
            BotSpec::App {
                student1: "s1".into(),
                student2: "s2".into(),
                deals3: AppPlay::APP_DEALS3,
                deals45: AppPlay::APP_DEALS45,
                base: "pimc-v0".into(),
                inference: None,
                student3: None,
            }
        );
        let p = json!({ "a": app(json!({ "deals3": [64, 64, 128, 128, 256, 256, 256, 256],
                                         "deals45": 256, "base": "pimc-v0:tt_mb=32" })),
                        "b": app(json!({ "student1": "old1", "student2": "old2" })) });
        let (a, b) = parse_bots(&p).expect("parses");
        assert_eq!(
            a,
            BotSpec::App {
                student1: "s1".into(),
                student2: "s2".into(),
                deals3: [64, 64, 128, 128, 256, 256, 256, 256],
                deals45: 256,
                base: "pimc-v0:tt_mb=32".into(),
                inference: None,
                student3: None,
            }
        );
        assert!(matches!(b, BotSpec::App { ref student1, .. } if student1 == "old1"));
        assert_eq!(
            input_names(&p).expect("names"),
            ["s1", "s2", "old1", "old2"]
        );
        // The deal weighting: defaults tau 1%, floor 5%; its keys checked.
        let p = json!({ "a": app(json!({ "inference": {} })), "b": app(json!({ "inference": { "tau": 20, "floor": 100 } })) });
        let (a, b) = parse_bots(&p).expect("parses");
        let BotSpec::App {
            inference: Some(ia),
            ..
        } = a
        else {
            panic!("inference")
        };
        assert_eq!(ia, InferenceCfg::DEFAULT);
        let BotSpec::App {
            inference: Some(ib),
            ..
        } = b
        else {
            panic!("inference")
        };
        assert_eq!(ib, InferenceCfg::from_permille(20, 100).expect("ok"));
        // A trick-3 student is an input of its side, listed after the other two.
        let p = json!({ "a": app(json!({ "student3": "s3" })), "b": app(json!({})) });
        let (a, _) = parse_bots(&p).expect("parses");
        assert!(matches!(a, BotSpec::App { student3: Some(ref n), .. } if n == "s3"));
        assert_eq!(input_names(&p).expect("names"), ["s1", "s2", "s3"]);
        for bad in [json!(""), json!(3)] {
            let p = json!({ "a": app(json!({ "student3": bad })), "b": "pimc-v0" });
            assert!(parse_bots(&p).is_err());
        }
        let same = json!({ "a": app(json!({})), "b": app(json!({})) });
        assert_eq!(input_names(&same).expect("names"), ["s1", "s2"]);
        assert!(
            input_names(&json!({ "a": "pimc-v0", "b": "rules-v1" }))
                .expect("none")
                .is_empty()
        );
    }

    #[test]
    fn bad_bots_are_refused_with_the_key_named() {
        let bad = [
            (json!({ "b": "x" }), "needs params.a"),
            (json!({ "a": "", "b": "x" }), "not empty"),
            (
                json!({ "a": 3, "b": "x" }),
                "a bot name or an app bot object",
            ),
            (
                json!({ "a": { "student1": "s1", "student2": "s2" }, "b": "x" }),
                "\"kind\": \"app\"",
            ),
            (
                json!({ "a": app(json!({ "net": "o3" })), "b": "x" }),
                "unknown key `net`",
            ),
            (
                json!({ "a": { "kind": "app", "student1": "s1" }, "b": "x" }),
                "names its student2",
            ),
            (
                json!({ "a": app(json!({ "deals3": [1, 2] })), "b": "x" }),
                "list of 8",
            ),
            (
                json!({ "a": app(json!({ "deals3": [0, 1, 1, 1, 1, 1, 1, 1] })), "b": "x" }),
                "deals3[0]",
            ),
            (
                json!({ "a": app(json!({ "deals45": 0 })), "b": "x" }),
                "deals45",
            ),
            (
                json!({ "a": app(json!({ "deals45": "many" })), "b": "x" }),
                "deals45",
            ),
            (
                json!({ "a": app(json!({ "base": "rules-v1" })), "b": "x" }),
                "params.a.base: not a PIMC bot",
            ),
            (
                json!({ "a": app(json!({ "base": "pimc-v0:w12=0" })), "b": "x" }),
                "params.a.base",
            ),
            (
                json!({ "a": app(json!({ "inference": 1 })), "b": "x" }),
                "inference is an object",
            ),
            (
                json!({ "a": app(json!({ "inference": { "temp": 1 } })), "b": "x" }),
                "unknown key `temp`",
            ),
            (
                json!({ "a": app(json!({ "inference": { "tau": 0 } })), "b": "x" }),
                "inference",
            ),
            (
                json!({ "a": app(json!({ "inference": { "floor": 1001 } })), "b": "x" }),
                "inference.floor",
            ),
        ];
        for (p, msg) in bad {
            let e = parse_bots(&p).expect_err(&p.to_string());
            assert!(e.contains(msg), "{p}: {e}");
        }
    }

    #[test]
    fn the_memory_estimate_counts_the_apps_tables_only() {
        let eq = 26 << 20;
        assert_eq!(
            chunk_bytes(&json!({ "a": "pimc-v0", "b": "rules-v1" }), eq),
            0
        );
        let one = chunk_bytes(&json!({ "a": app(json!({})), "b": "pimc-v0" }), eq);
        assert_eq!(one, eq + (16 + 64 + 64) * (1 << 20));
        let two = chunk_bytes(
            &json!({ "a": app(json!({ "base": "pimc-v0:tt_mb=128" })), "b": app(json!({})) }),
            eq,
        );
        assert_eq!(two, eq + (32 + 128 + 64) * (1 << 20));
    }

    /// Two factories of the app's player from the same students and deals are identical bots: a
    /// 14-way block of them scores exactly 0 (the canary), with the students deciding tricks
    /// 1-2 and tiny exact searches after. Built from the weights files through the inputs map,
    /// over the synthetic leaf (the production table is not a test fixture).
    #[test]
    fn a_block_of_identical_app_players_scores_exactly_zero() {
        use std::sync::Arc;
        use student::net::{random_net, write_net};
        let dir = std::env::temp_dir().join(format!("arena-bots-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let (s1, s2) = (random_net(4, &[16, 8], 12), random_net(5, &[16, 8], 12));
        let p1 = dir.join("s1.safetensors");
        let p2 = dir.join("s2.safetensors");
        std::fs::write(&p1, write_net(s1.layers(), &BTreeMap::new())).expect("s1");
        std::fs::write(&p2, write_net(s2.layers(), &BTreeMap::new())).expect("s2");
        let make = || {
            let app = AppPlay::from_bytes(
                &std::fs::read(&p1).expect("s1"),
                &std::fs::read(&p2).expect("s2"),
                [1, 1, 2, 1, 1, 1, 1, 2],
                1,
            )
            .expect("students");
            let spec = search::bot::PimcSpec::parse("pimc-v0:w12=1,w34=1,tt_mb=1").expect("ok");
            let leaf: Arc<search::pimc::SharedLeaf> = Arc::new(nn::synthetic::SyntheticLeaf);
            search::bot::PimcFactory::app(&spec, app, leaf, search::bot::default_source())
        };
        let (a, b) = (make(), make());
        assert_eq!(a.name(), b.name());
        assert!(a.name().starts_with("app:s1="), "{}", a.name());
        let r = arena::play_block(
            arena::BlockKind::Block14,
            arena::block_seed(7, 0, 0),
            &a,
            &b,
            engine::game::Rules::HOUSE,
            false,
        )
        .expect("a block plays");
        assert_eq!((r.score, r.stake_score), (0, 0));
        assert_eq!(r.records.len(), 14);
        // Different students on one side: the block is not a tie in general (the names differ).
        let other = {
            let app = AppPlay::from_bytes(
                &write_net(random_net(9, &[16, 8], 12).layers(), &BTreeMap::new()),
                &std::fs::read(&p2).expect("s2"),
                [1, 1, 2, 1, 1, 1, 1, 2],
                1,
            )
            .expect("students");
            let spec = search::bot::PimcSpec::parse("pimc-v0:w12=1,w34=1,tt_mb=1").expect("ok");
            let leaf: Arc<search::pimc::SharedLeaf> = Arc::new(nn::synthetic::SyntheticLeaf);
            search::bot::PimcFactory::app(&spec, app, leaf, search::bot::default_source())
        };
        assert_ne!(other.name(), a.name());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_input_file_is_a_clear_error() {
        let spec = BotSpec::App {
            student1: "s1".into(),
            student2: "s2".into(),
            deals3: AppPlay::APP_DEALS3,
            deals45: 1024,
            base: "pimc-v0".into(),
            inference: None,
            student3: None,
        };
        let e = build(&spec, &BTreeMap::new()).err().expect("no inputs");
        assert!(e.contains("input `s1` is not among the inputs"), "{e}");
        let inputs = BTreeMap::from([
            ("s1".to_string(), "/nonexistent/s1".to_string()),
            ("s2".to_string(), "/nonexistent/s2".to_string()),
        ]);
        let e = build(&spec, &inputs).err().expect("no file");
        assert!(e.contains("input `s1` (/nonexistent/s1)"), "{e}");
        assert!(build(&BotSpec::Name("nobody".into()), &inputs).is_err());
        assert_eq!(
            build(&BotSpec::Name("rules-v1".into()), &inputs)
                .expect("known")
                .name(),
            "rules-v1"
        );
    }
}
