//! The app's API (docs/WASM-API.md): load the files, value a decision, merge shards.
//!
//! A thin adapter over the engine's own search: the seat's view is rebuilt in the engine
//! ([`crate::appview`]), the candidates are `search::pimc::candidates`, every deal is drawn and
//! rebuilt by `search::pimc::world_match` (the belief source, deal `w` from
//! `Rng::derive(seed, [WORLD, w])`) and valued by `search::pimc::exact_world` (exact, or the
//! hybrid H_k with the oracle network as its frontier). Nothing here computes a value of its own:
//! it adds the deals' fixed-point values exactly and writes them out. With `"method": "student"`
//! a decision at trick 1 or 2 with all eight seats in play is valued instead by the student
//! network (`crates/student`, docs/STUDENT.md: the bots' fast player), every other decision as
//! before.

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
use student::encode::{encode_cand, encode_view};
use student::net::StudentNet;

use crate::appview::{AppPhase, AppTable, Refusal, field, int, rebuild};
use crate::cards::{app_card, engine_card};
use crate::lines::{LineSums, Walker, principal_line};

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
    /// The default student network (`"method": "student"`): serves every trick that has no student
    /// of its own.
    student: Option<StudentNet>,
    /// Student networks by the trick (1 or 2) they were trained for; they take precedence over
    /// the default one.
    students: BTreeMap<u8, StudentNet>,
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
    /// The student network (tricks 1–2, all eight seats in play, `"method": "student"`).
    Student,
}

impl Method {
    fn name(self) -> &'static str {
        match self {
            Method::Hybrid(_) => "hybrid",
            Method::Exact => "exact",
            Method::Forced => "forced",
            Method::Student => "student",
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
    /// The principal lines' sums (when the request asked for `lines`).
    lines: LineSums,
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
    /// The coach's fields from each deal's principal line were asked for.
    lines: bool,
    /// The acting seat's cards (its hand at the decision; the plan's rows).
    hand: Vec<Card>,
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
    lines: bool,
    /// `"method": "student"`.
    student: bool,
    /// `"shortlist"`: the cards to search (engine cards, sorted, no repeats), when given.
    shortlist: Option<Vec<Card>>,
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
    let shortlist = match v.get("shortlist") {
        None | Some(Value::Null) => None,
        Some(Value::Array(a)) => {
            let mut out: Vec<Card> = Vec::new();
            for (i, x) in a.iter().enumerate() {
                let n = x
                    .as_i64()
                    .ok_or_else(|| bad(format!("shortlist[{i}] must be a card")))?;
                let c =
                    engine_card(n).ok_or_else(|| bad(format!("shortlist[{i}]: no such card")))?;
                if out.contains(&c) {
                    return Err(bad(format!("shortlist[{i}]: a card twice")));
                }
                out.push(c);
            }
            if out.len() < 2 {
                return Err(bad("a shortlist holds at least two cards"));
            }
            out.sort_unstable();
            Some(out)
        }
        Some(_) => return Err(bad("shortlist must be an array of cards")),
    };
    let lines = match v.get("lines") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(bad("lines must be true or false")),
    };
    let student = match v.get("method") {
        None | Some(Value::Null) => false,
        Some(Value::String(m)) if m == "search" => false,
        Some(Value::String(m)) if m == "student" => true,
        Some(_) => return Err(bad("method must be \"search\" or \"student\"")),
    };
    if student && shards != 1 {
        return Err(bad(
            "a student request is one call: shards must be 1 (nothing to split)",
        ));
    }
    if student && lines {
        return Err(bad(
            "lines come from a search's deals: not with method student",
        ));
    }
    if student && shortlist.is_some() {
        return Err(bad(
            "a shortlist belongs to a search: not with method student",
        ));
    }
    let table = AppTable::read(field(&v, "match", "request")?, seat, own_burns)?;
    Ok(Request {
        seat,
        deals,
        seed,
        shard,
        shards,
        tt_mb,
        lines,
        student,
        shortlist,
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
            student: None,
            students: BTreeMap::new(),
        }
    }

    /// An engine whose leaf is `leaf` (tests: a synthetic leaf in place of the equity table),
    /// named `name` in [`Engine::info`].
    pub fn with_leaf(leaf: Arc<dyn Leaf + Send + Sync>, name: &str) -> Engine {
        Engine {
            leaf: Some((leaf, name.to_string())),
            nets: BTreeMap::new(),
            student: None,
            students: BTreeMap::new(),
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

    /// Load the student network (its safetensors bytes); returns its SHA-256. Refused unless it
    /// was made for this build's encoder (`student-v1` and its hash). A second one replaces the
    /// first.
    pub fn load_student(&mut self, bytes: &[u8]) -> Result<String, Refusal> {
        let n = StudentNet::from_bytes(bytes)
            .map_err(|e| Refusal::new("bad_file", format!("student network: {e}")))?;
        let sha = n.sha256.clone();
        self.student = Some(n);
        Ok(sha)
    }

    /// Load a student network for one trick (1 or 2: the network was trained on that trick's
    /// decisions, S1 and S2); returns its SHA-256. It serves `"method": "student"` requests at
    /// that trick, ahead of the default network of [`Engine::load_student`]. A second one for the
    /// same trick replaces the first. Refused like `load_student`, and for any other trick.
    pub fn load_student_for(&mut self, bytes: &[u8], trick: u8) -> Result<String, Refusal> {
        if !(1..=2).contains(&trick) {
            return Err(Refusal::new(
                "bad_request",
                format!("a student serves trick 1 or 2, not trick {trick}"),
            ));
        }
        let n = StudentNet::from_bytes(bytes)
            .map_err(|e| Refusal::new("bad_file", format!("student network: {e}")))?;
        let sha = n.sha256.clone();
        self.students.insert(trick, n);
        Ok(sha)
    }

    /// The student that values a decision at `trick`: its own, else the default one.
    fn student_for(&self, trick: u8) -> Option<&StudentNet> {
        self.students.get(&trick).or(self.student.as_ref())
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
        if let Some(st) = &self.student {
            s.push_str(&format!(",\"student\":{}", json_str(&st.sha256)));
        }
        if !self.students.is_empty() {
            s.push_str(",\"students\":{");
            for (i, (t, n)) in self.students.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&format!("\"{t}\":{}", json_str(&n.sha256)));
            }
            s.push('}');
        }
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
        // A shortlist searches only its cards (the rest are neither valued nor listed); each
        // must be one of the plays on offer. A decision with a rob or a steal is not shortlisted.
        let cands: Vec<Candidate> = match &req.shortlist {
            None => cands,
            Some(list) => {
                if cands.iter().any(|c| !matches!(c.0, Action::Play(_))) {
                    return Err(bad("a shortlist is for plays: not for a rob or a steal"));
                }
                for &c in list {
                    if !cands.iter().any(|x| x.0 == Action::Play(c)) {
                        return Err(bad(format!(
                            "shortlist: card {} is not one of the plays on offer",
                            app_card(c)
                        )));
                    }
                }
                cands
                    .into_iter()
                    .filter(|x| list.contains(&x.0.card()))
                    .collect()
            }
        };
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
        } else if req.student && (1..=2).contains(&p.trick_no) && turned == 0 {
            if self.student_for(p.trick_no).is_none() {
                return Err(Refusal::new(
                    "no_student",
                    format!(
                        "method student at trick {} needs a student network, not loaded",
                        p.trick_no
                    ),
                ));
            }
            Method::Student
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
        let student_sha = self
            .student_for(p.trick_no)
            .map(|n| format!("student:{}", n.sha256));
        let net_sha = match method {
            Method::Hybrid(cut) => Some(self.nets[&cut].sha256.as_str()),
            Method::Student => student_sha.as_deref(),
            _ => None,
        };
        let engine = engine_version(net_sha);
        let key = request_key(
            &engine,
            &view,
            req.seat,
            req.deals,
            req.seed,
            req.lines,
            req.shortlist.as_deref(),
        );
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
            lines: req.lines,
            hand: engine::cards::cards(view.hand).collect(),
            cands: cands
                .iter()
                .map(|&cand| CandTotals {
                    cand,
                    sums: [0; 4],
                    sumsq: 0,
                    lines: LineSums::default(),
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
        if method == Method::Student {
            // The student values every candidate from the seat's view: no deal is drawn.
            let net = self
                .student_for(view.public.trick_no)
                .expect("checked above");
            let f =
                encode_view(&view, &**leaf).map_err(|e| Refusal::new("inconsistent_view", e))?;
            let cf: Vec<Vec<u16>> = cands.iter().map(|&c| encode_cand(&view, c)).collect();
            for (c, v) in t.cands.iter_mut().zip(net.evaluate(&f, &cf)) {
                c.sums = v.0;
            }
            return Ok(t);
        }
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
        let walker = Walker {
            leaf: &**leaf,
            cut: match method {
                Method::Hybrid(k) => Some((k, &self.nets[&k])),
                _ => None,
            },
            tt_mb: req.tt_mb,
        };
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
            if req.lines {
                for (j, &cand) in cands.iter().enumerate() {
                    let l = principal_line(&world, cand, &walker, &mut mem);
                    t.cands[j].lines.add(&l);
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
fn request_key(
    engine: &str,
    view: &SeatView,
    seat: u8,
    deals: u32,
    seed: u64,
    lines: bool,
    shortlist: Option<&[Card]>,
) -> String {
    let v = serde_json::to_string(view).expect("a view serialises");
    // A search without a shortlist keeps the key it always had.
    let short = shortlist.map_or(String::new(), |l| {
        format!(
            "\nshortlist:{}",
            l.iter()
                .map(|c| app_card(*c).to_string())
                .collect::<Vec<_>>()
                .join(",")
        )
    });
    sha256_hex(format!("{engine}\n{seat}\n{deals}\n{seed}\n{lines}\n{v}{short}").as_bytes())
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
    let student = t.method == Method::Student;
    // Groups: candidates equal in every deal, each led by its first member (engine order).
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for j in 0..t.cands.len() {
        if t.lines {
            // With the lines every candidate is its own row (its line is its own); `equivalent`
            // still lists the candidates of equal value in every deal, itself first.
            let mut members = vec![j];
            members.extend((0..t.cands.len()).filter(|&i| i != j && t.class[i] == t.class[j]));
            groups.push((j, members));
        } else if t.class[j] == j {
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
        .map(|c| {
            if student {
                Some(MoneyChance(c.sums))
            } else {
                (!forced && n > 0).then(|| MoneyChance::mean_of_totals(c.sums, n))
            }
        })
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
        if forced || student { 0 } else { n },
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
            Some(m) if student => {
                // The student's own value: no deals, so no standard error; `sum_fp` carries the
                // value for `merge` (a student result is one call).
                let mean = m.0.map(fp_number).join(",");
                let mean_fp = m.0.map(|x| json_str(&x.to_string())).join(",");
                s.push_str(&format!(
                    ",\"mean\":[{mean}],\"mean_fp\":[{mean_fp}],\"value\":{},\"se\":null,\"n\":0,\"ess\":0,\"sum_fp\":[{mean_fp}],\"sumsq_fp\":null",
                    fp_number(m.0[team]),
                ));
            }
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
                if t.lines {
                    s.push_str(&lines_json(&c.lines, &t.hand, t.seat, n));
                }
            }
        }
        s.push('}');
    }
    s.push_str("]}");
    s
}

/// A share of `n` deals as a JSON number (an exact quotient of whole numbers, correctly rounded).
fn share(count: u64, n: u64) -> String {
    format!("{}", count as f64 / n as f64)
}

/// The mean over `n` deals of a sum in units of 2^-32, as a JSON number (floor in whole numbers,
/// then exact).
fn mean_fp(sum: u128, n: u64) -> String {
    let q = sum / u128::from(n);
    format!("{}", q as f64 / ONE as f64)
}

/// The coach's fields of a candidate from its lines' sums (docs/WASM-API.md §4), then the sums
/// themselves (`lines_fp`, for `merge`).
fn lines_json(l: &LineSums, hand: &[Card], seat: u8, n: u64) -> String {
    let team = team_of(seat);
    let won: u64 = (0..8u8)
        .filter(|&s| team_of(s) == team)
        .map(|s| l.winner[usize::from(s)])
        .sum();
    let winner = l.winner.map(|c| share(c, n)).join(",");
    let points = l.points_fp.map(|x| mean_fp(x, n)).join(",");
    // The plan's rows: the acting seat's cards, in the app's numbering.
    let mut rows: Vec<(u8, [u64; 5])> = hand
        .iter()
        .map(|&c| (app_card(c), l.plan.get(&c).copied().unwrap_or_default()))
        .collect();
    rows.sort_by_key(|r| r.0);
    let plan = rows
        .iter()
        .map(|(c, v)| format!("\"{c}\":[{}]", v.map(|x| share(x, n)).join(",")))
        .collect::<Vec<_>>()
        .join(",");
    let raw_plan = rows
        .iter()
        .map(|(c, v)| format!("\"{c}\":[{}]", v.map(|x| x.to_string()).join(",")))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        ",\"trickWin\":{},\"trickWinner\":[{winner}],\"expectedPoints\":[{points}],\"gameEnds\":{},\"plan\":{{{plan}}},\"lines_fp\":{{\"winner\":[{}],\"points\":[{}],\"gameEnds\":{},\"plan\":{{{raw_plan}}}}}",
        share(won, n),
        mean_fp(l.game_end_fp, n),
        l.winner.map(|c| c.to_string()).join(","),
        l.points_fp.map(|x| json_str(&x.to_string())).join(","),
        json_str(&l.game_end_fp.to_string()),
    )
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
pub const REFUSAL_CODES: [&str; 18] = [
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
    "no_student",
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
            || p.lines != first.lines
            || p.hand != first.hand
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
        c.lines = LineSums::default();
        for p in &parts {
            for (s, x) in c.sums.iter_mut().zip(p.cands[j].sums) {
                *s += x;
            }
            c.sumsq += p.cands[j].sumsq;
            c.lines.merge(&p.cands[j].lines);
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
        Some("student") => Method::Student,
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
    let student = method == Method::Student;
    let mut verb = BurnVerb::Rob;
    let mut lines = false;
    let mut hand: Vec<Card> = Vec::new();
    // Per candidate: its sums, its lines' sums and its class (the least candidate of its
    // `equivalent` list).
    type Row = (Candidate, [u64; 4], u128, Candidate, LineSums);
    let mut rows: Vec<Row> = Vec::new();
    let list = field(r, "candidates", "result")?
        .as_array()
        .ok_or_else(|| bad("result.candidates must be an array"))?;
    let read_cand = |e: &Value, what: &str, verb: &mut BurnVerb| -> Result<Candidate, Refusal> {
        let (a, vb) = read_action(field(e, "action", what)?, &format!("{what}.action"))?;
        if let Some(vb) = vb {
            *verb = vb;
        }
        let then = match e.get("then") {
            None => None,
            Some(t) => match read_action(t, &format!("{what}.then"))? {
                (Action::Play(c), None) => Some(c),
                _ => return Err(bad(format!("{what}.then must be a play"))),
            },
        };
        Ok((a, then))
    };
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
            let sumsq = if student {
                0
            } else {
                u_str(field(c, "sumsq_fp", &at)?, &format!("{at}.sumsq_fp"))?
            };
            (sums, sumsq)
        };
        let own_lines = match c.get("lines_fp") {
            None => None,
            Some(l) => {
                lines = true;
                let (sums, cards) = read_lines(l, &format!("{at}.lines_fp"))?;
                hand = cards;
                Some(sums)
            }
        };
        let eq = field(c, "equivalent", &at)?
            .as_array()
            .ok_or_else(|| bad(format!("{at}.equivalent must be an array")))?;
        let mut members = Vec::new();
        for (k, e) in eq.iter().enumerate() {
            members.push(read_cand(e, &format!("{at}.equivalent[{k}]"), &mut verb)?);
        }
        let key = *members
            .iter()
            .min()
            .ok_or_else(|| bad(format!("{at}.equivalent is empty")))?;
        match own_lines {
            // With the lines every candidate is its own row.
            Some(l) => {
                let own = read_cand(c, &at, &mut verb)?;
                rows.push((own, sums, sumsq, key, l));
            }
            // Without, a row stands for every candidate of its class.
            None => {
                for m in members {
                    rows.push((m, sums, sumsq, key, LineSums::default()));
                }
            }
        }
    }
    rows.sort_by_key(|r| r.0);
    if rows.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err(bad("a candidate is listed twice"));
    }
    // The class of each candidate: the first (in engine order) of its group.
    let mut lead: BTreeMap<Candidate, usize> = BTreeMap::new();
    let mut class = Vec::with_capacity(rows.len());
    for (j, row) in rows.iter().enumerate() {
        class.push(*lead.entry(row.3).or_insert(j));
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
        lines,
        hand,
        cands: rows
            .into_iter()
            .map(|(cand, sums, sumsq, _, lines)| CandTotals {
                cand,
                sums,
                sumsq,
                lines,
            })
            .collect(),
        class,
    })
}

/// A candidate's lines' sums (`lines_fp`) and the acting seat's cards (the plan's rows).
fn read_lines(l: &Value, at: &str) -> Result<(LineSums, Vec<Card>), Refusal> {
    let arr = |name: &str, len: usize| -> Result<&Vec<Value>, Refusal> {
        field(l, name, at)?
            .as_array()
            .filter(|a| a.len() == len)
            .ok_or_else(|| bad(format!("{at}.{name} must hold {len} entries")))
    };
    let mut out = LineSums::default();
    for (k, x) in arr("winner", 8)?.iter().enumerate() {
        out.winner[k] = x
            .as_u64()
            .ok_or_else(|| bad(format!("{at}.winner[{k}] must be a count")))?;
    }
    for (k, x) in arr("points", 4)?.iter().enumerate() {
        out.points_fp[k] = u_str(x, &format!("{at}.points[{k}]"))?;
    }
    out.game_end_fp = u_str(field(l, "gameEnds", at)?, &format!("{at}.gameEnds"))?;
    let mut cards = Vec::new();
    let plan = field(l, "plan", at)?
        .as_object()
        .ok_or_else(|| bad(format!("{at}.plan must be an object")))?;
    for (card, v) in plan {
        let c = card
            .parse::<i64>()
            .ok()
            .and_then(engine_card)
            .ok_or_else(|| bad(format!("{at}.plan: {card} is not a card")))?;
        let a = v
            .as_array()
            .filter(|a| a.len() == 5)
            .ok_or_else(|| bad(format!("{at}.plan[{card}] must hold 5 counts")))?;
        let mut row = [0u64; 5];
        for (t, x) in a.iter().enumerate() {
            row[t] = x
                .as_u64()
                .ok_or_else(|| bad(format!("{at}.plan[{card}] must hold counts")))?;
        }
        if row.iter().any(|&x| x > 0) {
            out.plan.insert(c, row);
        }
        cards.push(c);
    }
    cards.sort_unstable();
    Ok((out, cards))
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
