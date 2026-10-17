//! The app's API (docs/WASM-API.md): load the files, value a decision, merge shards.
//!
//! A thin adapter over the engine's own search: the seat's view is rebuilt in the engine
//! ([`crate::appview`]), the candidates are `search::pimc::candidates`, every deal is drawn and
//! rebuilt by `search::pimc::world_match` (the belief source, deal `w` from
//! `Rng::derive(seed, [WORLD, w])`) and valued by `search::pimc::exact_world` (exact, or the
//! hybrid H_k with the oracle network as its frontier). Nothing here computes a value of its own:
//! it adds the deals' fixed-point values exactly and writes them out.

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::cards::Card;
use engine::game::{Action, Rules};
use engine::team_of;
use engine::view::SeatView;
use equity::sha256::sha256_hex;
use equity::{EquityTable, MoneyChance, ONE};
use nn::Oracle;
use search::pimc::{
    Candidate, Ctx, SearchConfig, WorldSolve, candidates, exact_world, isqrt, world_match,
};
use search::worlds::{BeliefSource, WorldSource};
use serde_json::Value;
use solver::{FrontierCut, Leaf, SolveMem};

use crate::appview::{AppPhase, AppTable, Refusal, field, int, rebuild};
use crate::cards::{app_card, engine_card};

/// The request and result format version.
pub const API_VERSION: u32 = 1;

/// The most deals one search may have.
pub const MAX_DEALS: u32 = 4096;

/// The exact search's memory table, in MB, when the request names none.
pub const DEFAULT_TT_MB: usize = 16;

/// The largest memory table a request may ask for, in MB.
pub const MAX_TT_MB: usize = 1024;

/// The source commit the engine was built from (`ENGINE_COMMIT` at build time, set by
/// `tools/wasm_build.sh`), else `unversioned`.
pub fn engine_commit() -> &'static str {
    option_env!("ENGINE_COMMIT").unwrap_or("unversioned")
}

/// The engine version string of a result: `<commit>+<network SHA-256 or exact>+v<values
/// version>` (contract §6.4).
pub fn engine_version(net: Option<&str>) -> String {
    format!(
        "{}+{}+v{}",
        engine_commit(),
        net.unwrap_or("exact"),
        search::measure::VALUES_VERSION
    )
}

/// The loaded files.
pub struct Engine {
    /// The equity table (the solver's leaf and the networks' dense inputs) and its SHA-256.
    leaf: Option<(Arc<dyn Leaf + Send + Sync>, String)>,
    /// Oracle networks by the trick at whose start they take over.
    nets: BTreeMap<u8, Oracle>,
}

impl Default for Engine {
    fn default() -> Self {
        Engine::new()
    }
}

/// How a decision is valued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Exact search to the start of trick `cut`, then the network.
    Hybrid(u8),
    /// Exact search to the end of the hand.
    Exact,
    /// One legal action: nothing searched.
    Forced,
}

impl Method {
    fn name(self) -> &'static str {
        match self {
            Method::Hybrid(_) => "hybrid",
            Method::Exact => "exact",
            Method::Forced => "forced",
        }
    }
}

/// How the app names a burn: the dealer taking a turned Joker steals, everyone else robs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BurnVerb {
    Rob,
    Steal,
}

/// One candidate's totals over the deals of a result.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CandTotals {
    cand: Candidate,
    /// Σ of each team's fixed-point value.
    sums: [u64; 4],
    /// Σ of the square of the acting team's value.
    sumsq: u128,
}

/// Everything a result says, in the engine's terms: what [`write`] prints and [`merge`] adds.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Totals {
    engine: String,
    key: String,
    seat: u8,
    trick: u8,
    method: Method,
    deals: u32,
    shard: u32,
    shards: u32,
    from: u32,
    to: u32,
    world_count: Option<u128>,
    verb: BurnVerb,
    /// In the engine's candidate order.
    cands: Vec<CandTotals>,
    /// Per candidate: the first candidate whose value equalled its own in every deal.
    class: Vec<usize>,
}

/// A request, read.
struct Request {
    seat: u8,
    deals: u32,
    seed: u64,
    shard: u32,
    shards: u32,
    tt_mb: usize,
    table: AppTable,
}

fn bad(reason: impl Into<String>) -> Refusal {
    Refusal::new("bad_request", reason)
}

fn read_seed(v: &Value) -> Result<u64, Refusal> {
    match v {
        Value::String(s) => s
            .parse::<u64>()
            .map_err(|_| bad("seed must be a whole number 0..2^64 (a decimal string)")),
        Value::Number(n) => match n.as_u64() {
            Some(x) if x < 1 << 53 => Ok(x),
            _ => Err(bad(
                "seed as a number must be a whole number below 2^53; give larger seeds as a string",
            )),
        },
        _ => Err(bad("seed must be a decimal string or a number")),
    }
}

fn read_request(text: &str) -> Result<Request, Refusal> {
    let v: Value =
        serde_json::from_str(text).map_err(|e| bad(format!("the request is not JSON: {e}")))?;
    if !v.is_object() {
        return Err(bad("the request must be a JSON object"));
    }
    if int(&v, "v", "request", 0, 1000)? != i64::from(API_VERSION) {
        return Err(bad(format!("request format v must be {API_VERSION}")));
    }
    let seat = int(&v, "seat", "request", 0, 7)? as u8;
    let deals = int(&v, "deals", "request", 1, i64::from(MAX_DEALS))? as u32;
    let seed = read_seed(field(&v, "seed", "request")?)?;
    let shards = match v.get("shards") {
        None => 1,
        Some(_) => int(&v, "shards", "request", 1, i64::from(deals))? as u32,
    };
    let shard = match v.get("shard") {
        None => 0,
        Some(_) => int(&v, "shard", "request", 0, i64::from(shards) - 1)? as u32,
    };
    let tt_mb = match v.get("ttMb") {
        None => DEFAULT_TT_MB,
        Some(_) => int(&v, "ttMb", "request", 1, MAX_TT_MB as i64)? as usize,
    };
    let own_burns = match v.get("ownBurns") {
        None | Some(Value::Null) => None,
        Some(Value::Array(a)) => {
            let mut out = Vec::new();
            for (i, x) in a.iter().enumerate() {
                let n = x
                    .as_i64()
                    .ok_or_else(|| bad(format!("ownBurns[{i}] must be a card")))?;
                out.push(
                    engine_card(n).ok_or_else(|| bad(format!("ownBurns[{i}]: no such card")))?,
                );
            }
            Some(out)
        }
        Some(_) => return Err(bad("ownBurns must be an array of cards")),
    };
    let table = AppTable::read(field(&v, "match", "request")?, seat, own_burns)?;
    Ok(Request {
        seat,
        deals,
        seed,
        shard,
        shards,
        tt_mb,
        table,
    })
}

/// The deals of shard `i` of `k` out of `n`: `floor(i·n/k)..floor((i+1)·n/k)`.
pub fn shard_range(n: u32, i: u32, k: u32) -> (u32, u32) {
    let at = |j: u32| (u64::from(j) * u64::from(n) / u64::from(k)) as u32;
    (at(i), at(i + 1))
}

impl Engine {
    /// Nothing loaded.
    pub fn new() -> Engine {
        Engine {
            leaf: None,
            nets: BTreeMap::new(),
        }
    }

    /// An engine whose leaf is `leaf` (tests: a synthetic leaf in place of the equity table),
    /// named `name` in [`Engine::info`].
    pub fn with_leaf(leaf: Arc<dyn Leaf + Send + Sync>, name: &str) -> Engine {
        Engine {
            leaf: Some((leaf, name.to_string())),
            nets: BTreeMap::new(),
        }
    }

    /// Load the equity table (`.npy` bytes); returns its SHA-256. Networks loaded before are
    /// dropped (their inputs read the table).
    pub fn load_equity(&mut self, bytes: &[u8]) -> Result<String, Refusal> {
        let table = EquityTable::from_npy(bytes)
            .map_err(|e| Refusal::new("bad_file", format!("equity table: {e}")))?;
        if !table.is_complete() {
            return Err(Refusal::new(
                "bad_file",
                "equity table: not every state is solved",
            ));
        }
        let sha = sha256_hex(bytes);
        self.nets.clear();
        self.leaf = Some((Arc::new(table), sha.clone()));
        Ok(sha)
    }

    /// Load an oracle network (its safetensors bytes); returns its SHA-256 and the trick at whose
    /// start it takes over (from the file's metadata). A network for the same trick is replaced.
    pub fn load_net(&mut self, bytes: &[u8]) -> Result<(String, u8), Refusal> {
        let Some((leaf, _)) = &self.leaf else {
            return Err(Refusal::new(
                "no_equity",
                "load the equity table before a network",
            ));
        };
        let o = Oracle::from_bytes(bytes, leaf.clone())
            .map_err(|e| Refusal::new("bad_file", format!("network: {e}")))?;
        if !(2..=5).contains(&o.trick) {
            return Err(Refusal::new(
                "bad_file",
                format!(
                    "network for trick {}: a cut is at the start of trick 2..5",
                    o.trick
                ),
            ));
        }
        let out = (o.sha256.clone(), o.trick);
        self.nets.insert(o.trick, o);
        Ok(out)
    }

    /// The loaded files and the engine's version, as JSON.
    pub fn info(&self, memory_bytes: Option<u64>) -> String {
        let mut s = format!(
            "{{\"engine\":{},\"api\":{API_VERSION},\"valuesVersion\":{},\"equity\":{}",
            json_str(engine_commit()),
            search::measure::VALUES_VERSION,
            match &self.leaf {
                Some((_, sha)) => json_str(sha),
                None => "null".to_string(),
            }
        );
        s.push_str(",\"nets\":{");
        for (i, (t, o)) in self.nets.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("\"{t}\":{}", json_str(&o.sha256)));
        }
        s.push('}');
        if let Some(m) = memory_bytes {
            s.push_str(&format!(",\"memoryBytes\":{m}"));
        }
        s.push('}');
        s
    }

    /// Value a decision (docs/WASM-API.md §3–§5): the result JSON, or a refusal JSON.
    pub fn evaluate(&self, request: &str) -> String {
        match self.evaluate_totals(request) {
            Ok(t) => write(&t),
            Err(r) => refusal_json(&r),
        }
    }

    fn evaluate_totals(&self, request: &str) -> Result<Totals, Refusal> {
        let req = read_request(request)?;
        let rebuilt = rebuild(&req.table)?;
        let view = rebuilt.view;
        let m = rebuilt.m;
        let legal = m.legal_actions();
        let (_, all) = candidates(&view, &legal);
        // A burn of the card just taken has no app action (the app's rob burns from the hand
        // before the take): never listed, never searched.
        let taken = view
            .public
            .turned
            .iter()
            .rev()
            .find(|&&(_, by)| by == Some(req.seat))
            .map(|&(c, _)| c)
            .filter(|_| matches!(view.public.phase, engine::game::Phase::AwaitBurn(_)));
        let cands: Vec<Candidate> = all
            .into_iter()
            .filter(|&(a, _)| !(matches!(a, Action::Burn(_)) && Some(a.card()) == taken))
            .collect();
        let verb = if req.table.phase == AppPhase::JokerSteal {
            BurnVerb::Steal
        } else {
            BurnVerb::Rob
        };
        let p = &view.public;
        let trick = p.trick_no.max(1);
        let turned = p.turned_teams.iter().filter(|&&x| x).count();
        let method = if cands.len() <= 1 {
            Method::Forced
        } else if p.trick_no == 0 {
            return Err(Refusal::new(
                "not_built",
                "the dealer's burn before trick 1 is not valued yet",
            ));
        } else if turned > 0 {
            if p.trick_no == 1 && turned == 1 {
                return Err(Refusal::new(
                    "too_slow",
                    "a trick-1 decision with six seats in play takes seconds a deal: not searched",
                ));
            }
            Method::Exact
        } else if p.trick_no >= 3 {
            Method::Exact
        } else {
            let cut = p.trick_no + 1;
            if !self.nets.contains_key(&cut) {
                return Err(Refusal::new(
                    "no_network_for_trick",
                    format!(
                        "a trick-{} decision needs the trick-{cut} network (O_{cut}), not loaded",
                        p.trick_no
                    ),
                ));
            }
            Method::Hybrid(cut)
        };
        let net_sha = match method {
            Method::Hybrid(cut) => Some(self.nets[&cut].sha256.as_str()),
            _ => None,
        };
        let engine = engine_version(net_sha);
        let key = request_key(&engine, &view, req.seat, req.deals, req.seed);
        let (from, to) = shard_range(req.deals, req.shard, req.shards);
        let mut t = Totals {
            engine,
            key,
            seat: req.seat,
            trick,
            method,
            deals: req.deals,
            shard: req.shard,
            shards: req.shards,
            from,
            to,
            world_count: None,
            verb,
            cands: cands
                .iter()
                .map(|&cand| CandTotals {
                    cand,
                    sums: [0; 4],
                    sumsq: 0,
                })
                .collect(),
            class: (0..cands.len()).collect(),
        };
        if method == Method::Forced {
            return Ok(t);
        }
        let Some((leaf, _)) = &self.leaf else {
            return Err(Refusal::new(
                "no_equity",
                "load the equity table before evaluating",
            ));
        };
        let source = BeliefSource::new();
        let ctx = Ctx {
            leaf: &**leaf,
            source: &source,
            rules: Rules::HOUSE,
        };
        let cfg = SearchConfig {
            tt_mb: req.tt_mb,
            threads: 1,
            solve_threads: 1,
            ..SearchConfig::default()
        };
        let worlds = source.prepare(&view, Rules::HOUSE);
        t.world_count = worlds.count();
        let cut = match method {
            Method::Hybrid(k) => Some(FrontierCut::new(k, &self.nets[&k])),
            _ => None,
        };
        let team = usize::from(team_of(req.seat));
        let n = cands.len();
        let mut same = vec![vec![true; n]; n];
        let mut mem = SolveMem::new();
        for w in from..to {
            let world = world_match(&view, &*worlds, u64::from(w), req.seed, Rules::HOUSE);
            let mut run = WorldSolve::default();
            let row = exact_world(&world, &cands, &ctx, &cfg, None, &mut mem, &mut run, cut)
                .expect("nothing stops a solve here");
            for (j, v) in row.iter().enumerate() {
                let c = &mut t.cands[j];
                for (s, x) in c.sums.iter_mut().zip(v.0) {
                    *s += x;
                }
                let own = u128::from(v.0[team]);
                c.sumsq += own * own;
                for i in 0..j {
                    if row[i] != *v {
                        same[i][j] = false;
                    }
                }
            }
        }
        t.class = (0..n)
            .map(|j| (0..=j).find(|&i| i == j || same[i][j]).expect("itself"))
            .collect();
        Ok(t)
    }
}

/// The fingerprint of a search: the engine version, the seat's view as the engine rebuilt it,
/// the seat, the deals and the seed (equal for every shard of one search).
fn request_key(engine: &str, view: &SeatView, seat: u8, deals: u32, seed: u64) -> String {
    let v = serde_json::to_string(view).expect("a view serialises");
    sha256_hex(format!("{engine}\n{seat}\n{deals}\n{seed}\n{v}").as_bytes())
}

// ----- writing -----

/// A JSON string literal.
fn json_str(s: &str) -> String {
    serde_json::to_string(s).expect("a string serialises")
}

/// A fixed-point value (units of 2^-32) as a JSON number: exact (a double holds it exactly).
fn fp_number(x: u64) -> String {
    format!("{}", x as f64 / ONE as f64)
}

/// The refusal JSON.
pub fn refusal_json(r: &Refusal) -> String {
    format!(
        "{{\"v\":{API_VERSION},\"status\":\"refused\",\"code\":{},\"reason\":{},\"engine\":{}}}",
        json_str(r.code),
        json_str(&r.reason),
        json_str(engine_commit())
    )
}

fn action_json(cand: Candidate, verb: BurnVerb) -> String {
    match cand.0 {
        Action::Play(c) => format!("{{\"type\":\"play\",\"card\":{}}}", app_card(c)),
        Action::Burn(c) => format!(
            "{{\"type\":\"{}\",\"discard\":{}}}",
            match verb {
                BurnVerb::Rob => "rob",
                BurnVerb::Steal => "steal",
            },
            app_card(c)
        ),
    }
}

fn candidate_json(cand: Candidate, verb: BurnVerb) -> String {
    let mut s = format!("\"action\":{}", action_json(cand, verb));
    if let Some(c) = cand.1 {
        s.push_str(&format!(
            ",\"then\":{{\"type\":\"play\",\"card\":{}}}",
            app_card(c)
        ));
    }
    s
}

/// The result JSON of totals (one shard, or merged).
fn write(t: &Totals) -> String {
    let team = usize::from(team_of(t.seat));
    let n = u64::from(t.to - t.from);
    let forced = t.method == Method::Forced;
    // Groups: candidates equal in every deal, each led by its first member (engine order).
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for j in 0..t.cands.len() {
        if t.class[j] == j {
            groups.push((j, vec![j]));
        } else {
            let g = groups
                .iter_mut()
                .find(|(lead, _)| *lead == t.class[j])
                .expect("a class is led by an earlier candidate");
            g.1.push(j);
        }
    }
    let means: Vec<Option<MoneyChance>> = t
        .cands
        .iter()
        .map(|c| (!forced && n > 0).then(|| MoneyChance::mean_of_totals(c.sums, n)))
        .collect();
    // Best first: the acting pair's value, high first; ties to the engine's first candidate.
    let value = |j: usize| means[j].map_or(0, |m| m.0[team]);
    groups.sort_by(|a, b| value(b.0).cmp(&value(a.0)).then(a.0.cmp(&b.0)));

    let mut s = String::new();
    s.push_str(&format!(
        "{{\"v\":{API_VERSION},\"status\":\"ok\",\"engine\":{},\"key\":{},\"seat\":{},\"pair\":{},\"trick\":{},\"method\":\"{}\",\"meanHorizon\":{}",
        json_str(&t.engine),
        json_str(&t.key),
        t.seat,
        team,
        t.trick,
        t.method.name(),
        match t.method {
            Method::Hybrid(k) => (k - t.trick).to_string(),
            _ => "null".to_string(),
        }
    ));
    s.push_str(&format!(
        ",\"deals\":{},\"shard\":{},\"shards\":{},\"dealsFrom\":{},\"dealsTo\":{},\"worlds\":{},\"worldCount\":{},\"best\":0,\"candidates\":[",
        t.deals,
        t.shard,
        t.shards,
        t.from,
        t.to,
        if forced { 0 } else { n },
        match t.world_count {
            Some(c) => json_str(&c.to_string()),
            None => "null".to_string(),
        }
    ));
    for (gi, (lead, members)) in groups.iter().enumerate() {
        if gi > 0 {
            s.push(',');
        }
        let c = &t.cands[*lead];
        s.push('{');
        s.push_str(&candidate_json(c.cand, t.verb));
        s.push_str(",\"equivalent\":[");
        for (k, &j) in members.iter().enumerate() {
            if k > 0 {
                s.push(',');
            }
            s.push('{');
            s.push_str(&candidate_json(t.cands[j].cand, t.verb));
            s.push('}');
        }
        s.push(']');
        match means[*lead] {
            None => s.push_str(
                ",\"mean\":null,\"mean_fp\":null,\"value\":null,\"se\":null,\"n\":0,\"ess\":0,\"sum_fp\":null,\"sumsq_fp\":null",
            ),
            Some(m) => {
                let mean = m.0.map(fp_number).join(",");
                let mean_fp = m.0.map(|x| json_str(&x.to_string())).join(",");
                let sums = c.sums.map(|x| json_str(&x.to_string())).join(",");
                s.push_str(&format!(
                    ",\"mean\":[{mean}],\"mean_fp\":[{mean_fp}],\"value\":{},\"se\":{},\"n\":{n},\"ess\":{n},\"sum_fp\":[{sums}],\"sumsq_fp\":{}",
                    fp_number(m.0[team]),
                    fp_number(se_fp(c.sums[team], c.sumsq, n)),
                    json_str(&c.sumsq.to_string())
                ));
            }
        }
        s.push('}');
    }
    s.push_str("]}");
    s
}

/// The standard error of the mean of `n` values with sum `s` and sum of squares `s2`, in units of
/// 2^-32, in whole numbers (0 for fewer than two values).
fn se_fp(s: u64, s2: u128, n: u64) -> u64 {
    if n < 2 {
        return 0;
    }
    let (s, n) = (u128::from(s), u128::from(n));
    let num = (n * s2).saturating_sub(s * s);
    isqrt(num / (n * n * (n - 1))) as u64
}

// ----- merging -----

/// Merge shard results of one search (a JSON array of result JSON objects or strings) into the
/// result the whole search gives: the same bytes as one call with `shards` 1.
pub fn merge(results: &str) -> String {
    match merge_totals(results) {
        Ok(t) => write(&t),
        Err(r) => refusal_json(&r),
    }
}

/// Every refusal code (docs/WASM-API.md §5).
pub const REFUSAL_CODES: [&str; 17] = [
    "bad_request",
    "bad_file",
    "sha256_mismatch",
    "not_8_seats",
    "two_jokers",
    "not_house_rules",
    "not_playing",
    "not_to_act",
    "no_equity",
    "no_network_for_trick",
    "not_built",
    "too_slow",
    "led_trick_in_progress",
    "own_burns_hidden",
    "inconsistent_view",
    "merge_mismatch",
    "api_mismatch",
];

fn mismatch(reason: impl Into<String>) -> Refusal {
    Refusal::new("merge_mismatch", reason)
}

fn merge_totals(results: &str) -> Result<Totals, Refusal> {
    let v: Value = serde_json::from_str(results)
        .map_err(|e| bad(format!("merge takes a JSON array of results: {e}")))?;
    let Value::Array(items) = v else {
        return Err(bad("merge takes a JSON array of results"));
    };
    if items.is_empty() {
        return Err(bad("merge takes at least one result"));
    }
    let mut values = Vec::new();
    for (i, x) in items.iter().enumerate() {
        values.push(match x {
            Value::String(s) => serde_json::from_str::<Value>(s)
                .map_err(|e| bad(format!("result {i} is not JSON: {e}")))?,
            other => other.clone(),
        });
    }
    // Shards of a refused request are refused alike: the merge is that refusal.
    if values
        .iter()
        .all(|r| r.get("status") == Some(&Value::from("refused")))
    {
        let r = &values[0];
        if values.iter().all(|x| x == r) {
            let code = r["code"].as_str().unwrap_or("");
            if let Some(&code) = REFUSAL_CODES.iter().find(|&&c| c == code) {
                return Err(Refusal::new(code, r["reason"].as_str().unwrap_or("")));
            }
        }
        return Err(mismatch("the refusals differ"));
    }
    let mut parts = Vec::new();
    for (i, r) in values.iter().enumerate() {
        parts.push(
            read_totals(r)
                .map_err(|e| Refusal::new(e.code, format!("result {i}: {}", e.reason)))?,
        );
    }
    parts.sort_by_key(|t| t.from);
    let first = &parts[0];
    let mut out = first.clone();
    let mut at = 0;
    for p in &parts {
        if p.key != first.key
            || p.engine != first.engine
            || p.seat != first.seat
            || p.trick != first.trick
            || p.method != first.method
            || p.deals != first.deals
            || p.world_count != first.world_count
            || p.verb != first.verb
            || p.cands.len() != first.cands.len()
            || p.cands
                .iter()
                .zip(&first.cands)
                .any(|(a, b)| a.cand != b.cand)
        {
            return Err(mismatch("the results are not shards of one search"));
        }
        if p.from != at {
            return Err(mismatch(format!(
                "the shards do not cover the deals once each (deal {at} {})",
                if p.from > at {
                    "is missing"
                } else {
                    "is twice"
                }
            )));
        }
        at = p.to;
    }
    if at != first.deals {
        return Err(mismatch(format!("deals {at}..{} are missing", first.deals)));
    }
    let n = out.cands.len();
    for (j, c) in out.cands.iter_mut().enumerate() {
        c.sums = [0; 4];
        c.sumsq = 0;
        for p in &parts {
            for (s, x) in c.sums.iter_mut().zip(p.cands[j].sums) {
                *s += x;
            }
            c.sumsq += p.cands[j].sumsq;
        }
    }
    out.class = (0..n)
        .map(|j| {
            (0..=j)
                .find(|&i| i == j || parts.iter().all(|p| p.class[i] == p.class[j]))
                .expect("itself")
        })
        .collect();
    out.shard = 0;
    out.shards = 1;
    out.from = 0;
    out.to = out.deals;
    Ok(out)
}

fn u_str<T: std::str::FromStr>(v: &Value, what: &str) -> Result<T, Refusal> {
    v.as_str()
        .and_then(|s| s.parse::<T>().ok())
        .ok_or_else(|| bad(format!("{what} must be a decimal string")))
}

fn read_action(v: &Value, what: &str) -> Result<(Action, Option<BurnVerb>), Refusal> {
    let ty = field(v, "type", what)?
        .as_str()
        .ok_or_else(|| bad(format!("{what}.type must be a string")))?;
    let card = |name: &str| -> Result<Card, Refusal> {
        let n = int(v, name, what, 0, 52)?;
        engine_card(n).ok_or_else(|| bad(format!("{what}.{name}: no such card")))
    };
    match ty {
        "play" => Ok((Action::Play(card("card")?), None)),
        "rob" => Ok((Action::Burn(card("discard")?), Some(BurnVerb::Rob))),
        "steal" => Ok((Action::Burn(card("discard")?), Some(BurnVerb::Steal))),
        _ => Err(bad(format!("{what}.type {ty} is not an action"))),
    }
}

fn read_totals(r: &Value) -> Result<Totals, Refusal> {
    match field(r, "status", "result")?.as_str() {
        Some("ok") => {}
        _ => return Err(mismatch("a refused result cannot be merged")),
    }
    if int(r, "v", "result", 0, 1000)? != i64::from(API_VERSION) {
        return Err(bad(format!("result format v must be {API_VERSION}")));
    }
    let s = |name: &str| -> Result<String, Refusal> {
        field(r, name, "result")?
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| bad(format!("result.{name} must be a string")))
    };
    let seat = int(r, "seat", "result", 0, 7)? as u8;
    let trick = int(r, "trick", "result", 1, 5)? as u8;
    let method = match field(r, "method", "result")?.as_str() {
        Some("forced") => Method::Forced,
        Some("exact") => Method::Exact,
        Some("hybrid") => {
            let h = int(r, "meanHorizon", "result", 1, 4)? as u8;
            Method::Hybrid(trick + h)
        }
        _ => return Err(bad("result.method is unknown")),
    };
    let deals = int(r, "deals", "result", 1, i64::from(MAX_DEALS))? as u32;
    let shards = int(r, "shards", "result", 1, i64::from(deals))? as u32;
    let shard = int(r, "shard", "result", 0, i64::from(shards) - 1)? as u32;
    let from = int(r, "dealsFrom", "result", 0, i64::from(deals))? as u32;
    let to = int(r, "dealsTo", "result", 0, i64::from(deals))? as u32;
    if (from, to) != shard_range(deals, shard, shards) {
        return Err(bad("result.dealsFrom/dealsTo are not its shard's"));
    }
    let world_count = match field(r, "worldCount", "result")? {
        Value::Null => None,
        x => Some(u_str::<u128>(x, "result.worldCount")?),
    };
    let forced = method == Method::Forced;
    let mut verb = BurnVerb::Rob;
    let mut rows: Vec<(Candidate, [u64; 4], u128, usize)> = Vec::new();
    let list = field(r, "candidates", "result")?
        .as_array()
        .ok_or_else(|| bad("result.candidates must be an array"))?;
    for (g, c) in list.iter().enumerate() {
        let at = format!("result.candidates[{g}]");
        let (sums, sumsq) = if forced {
            ([0; 4], 0)
        } else {
            let a = field(c, "sum_fp", &at)?
                .as_array()
                .filter(|a| a.len() == 4)
                .ok_or_else(|| bad(format!("{at}.sum_fp must hold 4 sums")))?;
            let mut sums = [0u64; 4];
            for (k, x) in a.iter().enumerate() {
                sums[k] = u_str(x, &format!("{at}.sum_fp[{k}]"))?;
            }
            (
                sums,
                u_str(field(c, "sumsq_fp", &at)?, &format!("{at}.sumsq_fp"))?,
            )
        };
        let eq = field(c, "equivalent", &at)?
            .as_array()
            .ok_or_else(|| bad(format!("{at}.equivalent must be an array")))?;
        for (k, e) in eq.iter().enumerate() {
            let what = format!("{at}.equivalent[{k}]");
            let (a, vb) = read_action(field(e, "action", &what)?, &format!("{what}.action"))?;
            if let Some(vb) = vb {
                verb = vb;
            }
            let then = match e.get("then") {
                None => None,
                Some(t) => match read_action(t, &format!("{what}.then"))? {
                    (Action::Play(c), None) => Some(c),
                    _ => return Err(bad(format!("{what}.then must be a play"))),
                },
            };
            rows.push(((a, then), sums, sumsq, g));
        }
    }
    rows.sort_by_key(|r| r.0);
    if rows.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err(bad("a candidate is listed twice"));
    }
    // The class of each candidate: the first (in engine order) of its group.
    let mut lead_of_group: BTreeMap<usize, usize> = BTreeMap::new();
    let mut class = Vec::with_capacity(rows.len());
    for (j, row) in rows.iter().enumerate() {
        class.push(*lead_of_group.entry(row.3).or_insert(j));
    }
    Ok(Totals {
        engine: s("engine")?,
        key: s("key")?,
        seat,
        trick,
        method,
        deals,
        shard,
        shards,
        from,
        to,
        world_count,
        verb,
        cands: rows
            .iter()
            .map(|&(cand, sums, sumsq, _)| CandTotals { cand, sums, sumsq })
            .collect(),
        class,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shards_cover_the_deals_once() {
        for n in 1..40u32 {
            for k in 1..=n.min(9) {
                let mut at = 0;
                for i in 0..k {
                    let (a, b) = shard_range(n, i, k);
                    assert_eq!(a, at);
                    assert!(b > a, "every shard has a deal when k <= n");
                    at = b;
                }
                assert_eq!(at, n);
            }
        }
    }

    #[test]
    fn standard_errors_are_whole_numbers() {
        assert_eq!(se_fp(10, 100, 1), 0);
        // Values 0 and 2^32: mean 2^31, sample SD 2^31·√2, SE 2^31.
        let one = u128::from(ONE);
        assert_eq!(se_fp(ONE, one * one, 2), 1 << 31);
        // Equal values: no error.
        assert_eq!(se_fp(3 * 7, 3 * 49, 3), 0);
    }

    #[test]
    fn fixed_point_numbers_print_exactly() {
        assert_eq!(fp_number(0), "0");
        assert_eq!(fp_number(ONE), "1");
        assert_eq!(fp_number(ONE / 2), "0.5");
        assert_eq!(fp_number(ONE / 4), "0.25");
    }
}
