//! The PIMC bots and the bot registry (docs/M3-SPEC.md §5.5, §8.1).
//!
//! Names: `pimc-v0[:key=value,…]` and `pimc-teacher[:key=value,…]`, keys `w12`, `w34`, `t1`,
//! `t1w`, `t2` (`rollout` | `exact-few`), `t2w`, `threads`, `solve_threads`, `tt_mb`. The canonical
//! form (keys sorted, defaults omitted) is the recorded bot name. Any other name falls back to
//! `bots::factory` (`random`, `rules-v1`).

use std::sync::{Arc, Mutex, OnceLock};

use bot::{Bot, BotFactory};
use engine::cards::Card;
use engine::game::{Action, Event, Phase, Rules};
use engine::rng::Rng;
use engine::team_of;
use engine::view::SeatView;
use equity::EquityTable;

use equity::MoneyChance;
use student::encode::{encode_cand, encode_view};
use student::net::{StudentNet, choose};

use crate::inference::Inference;
use crate::labels;
use crate::pimc::{
    BeliefMode, Ctx, DecisionValues, Early, Method, SearchConfig, SharedLeaf, Students, evaluate,
    evaluate_weighted,
};
use crate::student_teacher::app_candidates;
use crate::worlds::{BeliefSource, WorldSource};

/// The sha256 of the production equity table the bots must use.
pub const EQUITY_SHA256: &str = "64740dab7845f0502c93ce195d721e1ffb8aa6ab141739ef43f754690de16808";

/// The defaults of `pimc-v0` (provisional until the agreement curves of §5.8/§13.1 set them).
pub fn v0_defaults() -> SearchConfig {
    SearchConfig::default()
}

/// The defaults of `pimc-teacher` (provisional, as `pimc-v0`; one thread, teacher parallelism is
/// across hands).
pub fn teacher_defaults() -> SearchConfig {
    SearchConfig::default()
}

/// A parsed bot name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PimcSpec {
    /// `pimc-v0` or `pimc-teacher`.
    pub base: &'static str,
    /// The configuration.
    pub cfg: SearchConfig,
}

impl PimcSpec {
    /// Parse `pimc-v0[:k=v,…]` or `pimc-teacher[:k=v,…]`.
    pub fn parse(name: &str) -> Result<PimcSpec, String> {
        let (base, rest) = match name.split_once(':') {
            Some((b, r)) => (b, Some(r)),
            None => (name, None),
        };
        let (base, mut cfg) = match base {
            "pimc-v0" => ("pimc-v0", v0_defaults()),
            "pimc-teacher" => ("pimc-teacher", teacher_defaults()),
            _ => return Err(format!("not a PIMC bot: `{name}`")),
        };
        for kv in rest.into_iter().flat_map(|r| r.split(',')) {
            if kv.is_empty() {
                continue;
            }
            let (k, v) = kv
                .split_once('=')
                .ok_or_else(|| format!("`{kv}` is not key=value"))?;
            let num = || -> Result<u32, String> {
                v.parse::<u32>()
                    .ok()
                    .filter(|&x| x > 0)
                    .ok_or_else(|| format!("`{k}` needs a positive whole number, not `{v}`"))
            };
            let early = || match v {
                "rollout" => Ok(Early::Rollout),
                "exact" | "exact-few" => Ok(Early::Exact),
                _ => Err(format!("`{k}` is rollout or exact-few, not `{v}`")),
            };
            match k {
                "w12" => cfg.w12 = num()?,
                "w34" => cfg.w34 = num()?,
                "t1" => cfg.t1 = early()?,
                "t1w" => cfg.t1w = num()?,
                "t2" => cfg.t2 = early()?,
                "t2w" => cfg.t2w = num()?,
                "threads" => cfg.threads = num()? as usize,
                "solve_threads" => cfg.solve_threads = num()? as usize,
                "tt_mb" => cfg.tt_mb = num()? as usize,
                _ => return Err(format!("unknown key `{k}`")),
            }
        }
        Ok(PimcSpec { base, cfg })
    }

    /// The canonical name: keys sorted, defaults omitted. Thread counts never change a result
    /// and are left out.
    pub fn canonical(&self) -> String {
        let d = if self.base == "pimc-v0" {
            v0_defaults()
        } else {
            teacher_defaults()
        };
        let c = &self.cfg;
        let early = |e: Early| match e {
            Early::Rollout => "rollout",
            Early::Exact => "exact-few",
        };
        let mut kv: Vec<(&str, String)> = Vec::new();
        if c.t1 != d.t1 {
            kv.push(("t1", early(c.t1).to_string()));
        }
        if c.t1w != d.t1w {
            kv.push(("t1w", c.t1w.to_string()));
        }
        if c.t2 != d.t2 {
            kv.push(("t2", early(c.t2).to_string()));
        }
        if c.t2w != d.t2w {
            kv.push(("t2w", c.t2w.to_string()));
        }
        if c.tt_mb != d.tt_mb {
            kv.push(("tt_mb", c.tt_mb.to_string()));
        }
        if c.w12 != d.w12 {
            kv.push(("w12", c.w12.to_string()));
        }
        if c.w34 != d.w34 {
            kv.push(("w34", c.w34.to_string()));
        }
        kv.sort();
        if kv.is_empty() {
            return self.base.to_string();
        }
        let s: Vec<String> = kv.iter().map(|(k, v)| format!("{k}={v}")).collect();
        format!("{}:{}", self.base, s.join(","))
    }
}

/// The decision seed (§5.6): `Rng::derive(bot_seed, &[DECISION, hand_index, actions so far in
/// the match]).next_u64()`; actions are counted from the view's public events.
pub fn decision_seed(bot_seed: u64, view: &SeatView) -> u64 {
    let actions = view
        .events
        .iter()
        .filter(|e| matches!(e, Event::Played { .. } | Event::BurnDone { .. }))
        .count() as u64;
    Rng::derive(
        bot_seed,
        &[labels::DECISION, u64::from(view.public.hand_index), actions],
    )
    .next_u64()
}

/// A shared log of (view, values) for every decision of some bots.
pub type DecisionLog = Arc<Mutex<Vec<(SeatView, DecisionValues)>>>;

/// The app's player (fivefingers' engine bots; `docs/WASM-API.md`, `docs/STUDENT.md`): at tricks
/// 1 and 2 with all eight seats in play the students S1 and S2 value every candidate from the
/// seat's own view and the best is played (no deal is drawn; the robber's joint burn-then-play
/// candidates as the app lists them, `app_candidates`); at tricks 3-5 exact PIMC at a fixed deal
/// count by trick and place in the trick. The dealer's burn before trick 1 and a decision with a
/// pair turned at tricks 1-2 are played as the bot's `SearchConfig` says (the app hands those to
/// its old bot or the hybrid).
#[derive(Clone)]
pub struct AppPlay {
    /// The trick-1 student.
    pub s1: Arc<StudentNet>,
    /// The trick-2 student.
    pub s2: Arc<StudentNet>,
    /// Exact deals at trick 3 by place in the trick (0 = the leader).
    pub deals3: [u32; 8],
    /// Exact deals at tricks 4 and 5, any place.
    pub deals45: u32,
    /// Weight the exact searches' deals by the other seats' plays under the students
    /// (`crate::inference`; PLAN §6.4 point 3): `None` plays as the app does today.
    pub inference: Option<InferenceCfg>,
}

/// The world weighting's settings in a bot: the temperature and the floor of
/// [`crate::inference::Inference`] (units of 2^-32 of the money).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InferenceCfg {
    /// The softmax temperature in money units of 2^-32.
    pub tau_q32: u64,
    /// The floor in units of 2^-32.
    pub floor_q32: u64,
}

impl InferenceCfg {
    /// The first arena arm's settings (tau 1% of the money, floor 5%).
    pub const DEFAULT: InferenceCfg = InferenceCfg {
        tau_q32: Inference::TAU_DEFAULT_Q32,
        floor_q32: Inference::FLOOR_DEFAULT_Q32,
    };

    /// From permille of the money (the manifest's unit).
    pub fn from_permille(tau: u32, floor: u32) -> Result<InferenceCfg, String> {
        if tau == 0 || tau > 1000 {
            return Err(format!(
                "inference tau {tau}: 1 to 1000 permille of the money"
            ));
        }
        if floor > 1000 {
            return Err(format!("inference floor {floor}: 0 to 1000 permille"));
        }
        let q = |x: u32| (u64::from(x) << 32) / 1000;
        Ok(InferenceCfg {
            tau_q32: q(tau),
            floor_q32: q(floor),
        })
    }
}

impl AppPlay {
    /// The app's trick-3 deals by place (`ENGINE41_BOTS` in fivefingers' `config.ts`).
    pub const APP_DEALS3: [u32; 8] = [192, 224, 512, 640, 1024, 1024, 1024, 1024];
    /// The app's deals at tricks 4 and 5.
    pub const APP_DEALS45: u32 = 1024;

    /// The students from their weights files (checked as the app checks them), with the deals.
    pub fn from_bytes(
        s1: &[u8],
        s2: &[u8],
        deals3: [u32; 8],
        deals45: u32,
    ) -> Result<AppPlay, String> {
        if deals3.contains(&0) || deals45 == 0 {
            return Err("every deal count of the app's player is at least 1".to_string());
        }
        let s1 = StudentNet::from_bytes(s1).map_err(|e| format!("student1: {e}"))?;
        let s2 = StudentNet::from_bytes(s2).map_err(|e| format!("student2: {e}"))?;
        Ok(AppPlay {
            s1: Arc::new(s1),
            s2: Arc::new(s2),
            deals3,
            deals45,
            inference: None,
        })
    }

    /// The same player with its exact searches' deals weighted by the other seats' plays.
    pub fn with_inference(mut self, cfg: InferenceCfg) -> AppPlay {
        self.inference = Some(cfg);
        self
    }

    /// The weighting, with the students as the play model.
    fn inference(&self) -> Option<Inference<'_>> {
        self.inference.map(|c| Inference {
            students: Students {
                s1: &self.s1,
                s2: &self.s2,
            },
            tau_q32: c.tau_q32,
            floor_q32: c.floor_q32,
        })
    }

    /// The student that plays `trick` (1 or 2).
    pub fn student_for(&self, trick: u8) -> Option<&StudentNet> {
        match trick {
            1 => Some(&self.s1),
            2 => Some(&self.s2),
            _ => None,
        }
    }

    /// The exact deals of a decision at `trick` (3 to 5) in `place` (cards already in the trick).
    pub fn deals_for(&self, trick: u8, place: usize) -> u32 {
        if trick == 3 {
            self.deals3[place.min(7)]
        } else {
            self.deals45
        }
    }

    /// The name's part that says which students and deals play: the students by the first 12
    /// hex digits of their SHA-256.
    pub fn name(&self) -> String {
        let d3: Vec<String> = self.deals3.iter().map(u32::to_string).collect();
        let inf = self
            .inference()
            .map_or(String::new(), |i| format!(",{}", i.name()));
        format!(
            "app:s1={},s2={},deals3={},deals45={}{inf}",
            &self.s1.sha256[..12.min(self.s1.sha256.len())],
            &self.s2.sha256[..12.min(self.s2.sha256.len())],
            d3.join("/"),
            self.deals45
        )
    }

    /// Whether the students play this decision: trick 1 or 2, all eight seats in play.
    fn plays(&self, view: &SeatView) -> bool {
        let p = &view.public;
        (1..=2).contains(&p.trick_no) && p.turned_teams.iter().all(|&t| !t)
    }

    /// The students' decision from the seat's view: every candidate valued by the trick's
    /// student, the best for the seat's team chosen (`choose`, the app's rule).
    fn decide(
        &self,
        view: &SeatView,
        legal: &[Action],
        leaf: &SharedLeaf,
    ) -> Result<DecisionValues, String> {
        let (kind, cands) = app_candidates(view, legal);
        let net = self
            .student_for(view.public.trick_no)
            .ok_or_else(|| format!("no student for trick {}", view.public.trick_no))?;
        let values = if cands.len() <= 1 {
            vec![MoneyChance::even(); cands.len()]
        } else {
            let f = encode_view(view, leaf)?;
            let cf: Vec<Vec<u16>> = cands.iter().map(|&c| encode_cand(view, c)).collect();
            net.evaluate(&f, &cf)
        };
        let chosen = choose(&values, team_of(view.seat));
        Ok(DecisionValues {
            kind,
            method: if cands.len() <= 1 {
                Method::Forced
            } else {
                Method::Student
            },
            n_worlds: 0,
            world_count: None,
            belief: BeliefMode::Uniform,
            candidates: cands
                .iter()
                .zip(&values)
                .map(|(&(a, p), &v)| (a, p, v))
                .collect(),
            chosen,
            own_sd: vec![0; cands.len()],
        })
    }
}

/// The PIMC bot.
pub struct PimcBot {
    name: String,
    cfg: SearchConfig,
    leaf: Arc<SharedLeaf>,
    source: Arc<dyn WorldSource>,
    rules: Rules,
    seed: u64,
    /// The app's player, when this bot plays as the app does.
    app: Option<Arc<AppPlay>>,
    /// The play stored by a joint (burn, first card) decision: (hand index, card).
    pending_play: Option<(u32, Card)>,
    /// The last decision's values (for teacher records).
    pub last: Option<DecisionValues>,
    /// Every decision of this bot, when recording (for teacher records).
    pub log: Option<DecisionLog>,
}

impl PimcBot {
    /// A PIMC bot.
    pub fn new(
        name: String,
        cfg: SearchConfig,
        leaf: Arc<SharedLeaf>,
        source: Arc<dyn WorldSource>,
    ) -> PimcBot {
        PimcBot {
            name,
            cfg,
            leaf,
            source,
            rules: Rules::HOUSE,
            seed: 0,
            app: None,
            pending_play: None,
            last: None,
            log: None,
        }
    }

    /// The same bot playing as the app does ([`AppPlay`]); `cfg` plays what the students do not.
    pub fn with_app(mut self, app: Arc<AppPlay>) -> PimcBot {
        self.app = Some(app);
        self
    }

    /// The values of this decision: the students' at tricks 1-2 with eight seats in play when
    /// the bot plays as the app does, else the search at the configured budget (the app's exact
    /// deals by trick and place at tricks 3-5).
    fn decide(&self, view: &SeatView, legal: &[Action]) -> DecisionValues {
        let dseed = decision_seed(self.seed, view);
        let ctx = Ctx {
            leaf: &*self.leaf,
            source: &*self.source,
            rules: self.rules,
        };
        let Some(app) = &self.app else {
            return evaluate(view, legal, dseed, &ctx, &self.cfg);
        };
        if app.plays(view) {
            return app
                .decide(view, legal, &*self.leaf)
                .expect("a table's view is consistent and all eight seats are in play");
        }
        let p = &view.public;
        if p.trick_no >= 3 {
            let cfg = SearchConfig {
                w34: app.deals_for(p.trick_no, p.trick_cards.len()),
                ..self.cfg
            };
            return match app.inference() {
                Some(inf) => evaluate_weighted(view, legal, dseed, &ctx, &cfg, &inf),
                None => evaluate(view, legal, dseed, &ctx, &cfg),
            };
        }
        evaluate(view, legal, dseed, &ctx, &self.cfg)
    }
}

impl Bot for PimcBot {
    fn name(&self) -> &str {
        &self.name
    }

    fn start_match(&mut self, _seat: u8, rules: Rules, seed: u64) {
        self.rules = rules;
        self.seed = seed;
        self.pending_play = None;
    }

    fn act(&mut self, view: &SeatView, legal: &[Action]) -> Action {
        if let Some((h, c)) = self.pending_play.take()
            && h == view.public.hand_index
            && matches!(view.public.phase, Phase::AwaitPlay(_))
        {
            debug_assert!(
                legal.contains(&Action::Play(c)),
                "stored joint play is legal"
            );
            if legal.contains(&Action::Play(c)) {
                return Action::Play(c);
            }
        }
        let dv = self.decide(view, legal);
        let (a, p, _) = dv.candidates[dv.chosen];
        if let Some(c) = p {
            self.pending_play = Some((view.public.hand_index, c));
        }
        if let Some(log) = &self.log {
            log.lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((view.clone(), dv.clone()));
        }
        self.last = Some(dv);
        a
    }
}

/// Builds PIMC bots.
pub struct PimcFactory {
    name: String,
    cfg: SearchConfig,
    leaf: Arc<SharedLeaf>,
    source: Arc<dyn WorldSource>,
    app: Option<Arc<AppPlay>>,
}

impl PimcFactory {
    /// A factory for a parsed spec with this leaf and world source.
    pub fn new(spec: &PimcSpec, leaf: Arc<SharedLeaf>, source: Arc<dyn WorldSource>) -> Self {
        PimcFactory {
            name: spec.canonical(),
            cfg: spec.cfg,
            leaf,
            source,
            app: None,
        }
    }

    /// A factory of bots that play as the app does ([`AppPlay`]); `spec` plays what the students
    /// do not (the dealer's burn, a turned pair at tricks 1-2) and sets the threads and table
    /// size of the exact solves. Named `app:…` after the students and deals, with the spec's
    /// canonical name after a `|` unless it is the plain `pimc-v0`.
    pub fn app(
        spec: &PimcSpec,
        app: AppPlay,
        leaf: Arc<SharedLeaf>,
        source: Arc<dyn WorldSource>,
    ) -> Self {
        let base = spec.canonical();
        let name = if base == "pimc-v0" {
            app.name()
        } else {
            format!("{}|{base}", app.name())
        };
        PimcFactory {
            name,
            cfg: spec.cfg,
            leaf,
            source,
            app: Some(Arc::new(app)),
        }
    }

    /// Build a concrete bot (for callers that need the decision log).
    pub fn build_pimc(&self) -> PimcBot {
        let bot = PimcBot::new(
            self.name.clone(),
            self.cfg,
            self.leaf.clone(),
            self.source.clone(),
        );
        match &self.app {
            Some(app) => bot.with_app(app.clone()),
            None => bot,
        }
    }
}

/// A factory of the app's player with the production equity table: the students' weights files
/// and the deals, over `base` (a `pimc-v0[:…]` name for what the students do not play).
pub fn app_factory(
    base: &str,
    s1: &[u8],
    s2: &[u8],
    deals3: [u32; 8],
    deals45: u32,
    inference: Option<InferenceCfg>,
) -> Result<PimcFactory, String> {
    let spec = PimcSpec::parse(base)?;
    let mut app = AppPlay::from_bytes(s1, s2, deals3, deals45)?;
    if let Some(c) = inference {
        app = app.with_inference(c);
    }
    let table = equity()?;
    let leaf: Arc<SharedLeaf> = table;
    Ok(PimcFactory::app(&spec, app, leaf, default_source()))
}

impl BotFactory for PimcFactory {
    fn name(&self) -> &str {
        &self.name
    }
    fn build(&self) -> Box<dyn Bot> {
        Box::new(self.build_pimc())
    }
}

static EQUITY_PATH: OnceLock<String> = OnceLock::new();
static EQUITY: OnceLock<Result<Arc<EquityTable>, String>> = OnceLock::new();

/// Set the equity table's path (`arena --equity <path>`); otherwise `FORTYONE_EQUITY` is read.
/// Only the first call counts.
pub fn set_equity_path(path: &str) {
    let _ = EQUITY_PATH.set(path.to_string());
}

/// Load an equity table and check its sha256 against the production table.
pub fn load_equity(path: &str) -> Result<EquityTable, String> {
    load_equity_sha(path).map(|(t, _)| t)
}

/// [`load_equity`], also returning the table's SHA-256 (hex): the worlds files and checkpoints
/// record it, so a run is never resumed with another table.
pub fn load_equity_sha(path: &str) -> Result<(EquityTable, String), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let sha = equity::sha256::sha256_hex(&bytes);
    if sha != EQUITY_SHA256 {
        return Err(format!(
            "{path}: sha256 {sha} is not the production equity table {EQUITY_SHA256}"
        ));
    }
    let table = EquityTable::from_npy(&bytes).map_err(|e| format!("{path}: {e}"))?;
    Ok((table, sha))
}

/// The shared production equity table (loaded once).
pub fn equity() -> Result<Arc<EquityTable>, String> {
    EQUITY
        .get_or_init(|| {
            let path = EQUITY_PATH
                .get()
                .cloned()
                .or_else(|| std::env::var("FORTYONE_EQUITY").ok())
                .ok_or_else(|| {
                    "PIMC bots need the equity table: set FORTYONE_EQUITY or pass --equity"
                        .to_string()
                })?;
            load_equity(&path).map(Arc::new)
        })
        .clone()
}

/// The default world generator: `crates/belief`'s counted, exactly uniform sampler (M3-SPEC
/// §5.2, `belief = uniform`).
pub fn default_source() -> Arc<dyn WorldSource> {
    Arc::new(BeliefSource::new())
}

/// The bot registry: PIMC bots by name (with the production equity table), else `bots::factory`.
pub fn factory(name: &str) -> Option<Box<dyn BotFactory + Send + Sync>> {
    if name.starts_with("pimc-") {
        let spec = PimcSpec::parse(name).ok()?;
        let table = equity().ok()?;
        let leaf: Arc<SharedLeaf> = table;
        return Some(Box::new(PimcFactory::new(&spec, leaf, default_source())));
    }
    bots::factory(name)
        .map(|_| Box::new(Named(name.to_string())) as Box<dyn BotFactory + Send + Sync>)
}

/// Why `factory` refused a name (for error messages).
pub fn factory_error(name: &str) -> String {
    if name.starts_with("pimc-") {
        if let Err(e) = PimcSpec::parse(name) {
            return e;
        }
        if let Err(e) = equity() {
            return e;
        }
    }
    format!(
        "unknown bot `{name}` (known: {}, pimc-v0[:…], pimc-teacher[:…])",
        bots::names().join(", ")
    )
}

/// A `bots` registry factory by name, `Send + Sync`.
struct Named(String);

impl BotFactory for Named {
    fn name(&self) -> &str {
        &self.0
    }
    fn build(&self) -> Box<dyn Bot> {
        bots::factory(&self.0).expect("checked name").build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_parse_and_canonicalise() {
        let s = PimcSpec::parse("pimc-v0").expect("ok");
        assert_eq!(s.canonical(), "pimc-v0");
        let s = PimcSpec::parse("pimc-v0:w34=8,w12=10,threads=4").expect("ok");
        assert_eq!(s.cfg.w12, 10);
        assert_eq!(s.cfg.threads, 4);
        assert_eq!(s.canonical(), "pimc-v0:w12=10,w34=8");
        let s = PimcSpec::parse("pimc-teacher:t2=exact-few,t2w=4").expect("ok");
        assert_eq!(s.canonical(), "pimc-teacher:t2=exact-few,t2w=4");
        assert!(PimcSpec::parse("pimc-v0:w12=0").is_err());
        assert!(PimcSpec::parse("pimc-v0:x=1").is_err());
        assert!(PimcSpec::parse("pimc-v9").is_err());
    }

    #[test]
    fn registry_falls_back_to_bots() {
        assert_eq!(factory("rules-v1").expect("known").name(), "rules-v1");
        assert_eq!(factory("random").expect("known").build().name(), "random");
        assert!(factory("nobody").is_none());
    }

    use std::collections::BTreeMap;

    use engine::game::Match;
    use nn::synthetic::SyntheticLeaf;
    use student::net::{random_net, write_net};

    fn leaf() -> Arc<SharedLeaf> {
        Arc::new(SyntheticLeaf)
    }

    fn students() -> (StudentNet, StudentNet) {
        (random_net(4, &[16, 8], 12), random_net(5, &[16, 8], 12))
    }

    fn tiny_app() -> AppPlay {
        let (s1, s2) = students();
        AppPlay {
            s1: Arc::new(s1),
            s2: Arc::new(s2),
            deals3: [1, 1, 2, 1, 1, 1, 1, 2],
            deals45: 1,
            inference: None,
        }
    }

    fn tiny_cfg() -> SearchConfig {
        SearchConfig {
            w12: 1,
            w34: 1,
            t1w: 1,
            t2w: 1,
            tt_mb: 1,
            ..SearchConfig::default()
        }
    }

    /// Plays whole matches with one bot per seat from `make`, checking every action is legal;
    /// returns every decision's values by seat (the bots' `last`), in order.
    fn play_matches(
        make: &dyn Fn() -> PimcBot,
        seeds: std::ops::Range<u64>,
    ) -> Vec<(u8, DecisionValues)> {
        let mut out = Vec::new();
        for seed in seeds {
            let mut m = Match::new(seed, Rules::HOUSE);
            let mut bots: Vec<PimcBot> = (0..8).map(|_| make()).collect();
            for (s, b) in (0u8..).zip(bots.iter_mut()) {
                b.start_match(
                    s,
                    Rules::HOUSE,
                    Rng::derive(seed, &[0xB07, u64::from(s)]).next_u64(),
                );
            }
            let mut guard = 0;
            while let Some(seat) = m.to_act() {
                let view = m.view(seat);
                let legal = m.legal_actions();
                let a = bots[usize::from(seat)].act(&view, &legal);
                assert!(
                    legal.contains(&a),
                    "seat {seat} chose {a:?}, not among {legal:?}"
                );
                m.apply(a).expect("legal");
                if let Some(dv) = bots[usize::from(seat)].last.take() {
                    out.push((seat, dv));
                }
                guard += 1;
                assert!(guard < 10_000, "a match ends");
            }
            assert_eq!(m.phase(), Phase::MatchOver);
        }
        out
    }

    /// The app's player: legal over whole matches; tricks 1-2 with eight seats in play are the
    /// students' (`Method::Student`, no world, every candidate valued, the chosen one its team's
    /// best); tricks 3-5 are exact at the deals of the trick and place; a forced card is forced.
    #[test]
    fn the_app_player_plays_legal_whole_matches_the_students_deciding_tricks_one_and_two() {
        let app = Arc::new(tiny_app());
        let l = leaf();
        let source = default_source();
        let make = || {
            PimcBot::new("app".into(), tiny_cfg(), l.clone(), source.clone()).with_app(app.clone())
        };
        let decisions = play_matches(&make, 1..7);
        let (mut students, mut exact, mut forced, mut joint) = (0, 0, 0, 0);
        for (seat, dv) in &decisions {
            match dv.method {
                Method::Student => {
                    students += 1;
                    assert_eq!(dv.n_worlds, 0);
                    assert!(dv.candidates.len() > 1);
                    let team = team_of(*seat);
                    let best = dv
                        .candidates
                        .iter()
                        .map(|c| c.2.get(team))
                        .max()
                        .expect("some");
                    assert_eq!(dv.candidates[dv.chosen].2.get(team), best);
                    joint += usize::from(dv.kind == crate::pimc::DecisionKind::JointBurnPlay);
                }
                Method::ExactPimc => exact += 1,
                Method::Forced => forced += 1,
                Method::RolloutPimc => {} // the dealer's burn, or a turned pair at tricks 1-2
                Method::PlayoutPimc => panic!("a bot never values by the playout teacher"),
            }
        }
        assert!(students > 100, "{students} student decisions");
        assert!(joint > 0, "a robber's joint decision went to the student");
        assert!(exact > 50, "{exact} exact decisions");
        assert!(forced > 0);
    }

    /// With the deal weighting on, the exact decisions mix their deals by the other seats' plays
    /// (`BeliefMode::Policy`), the students' decisions are as before, every action is legal over
    /// whole matches, and the name says so.
    #[test]
    fn the_app_player_with_inference_weights_its_exact_decisions() {
        let app = Arc::new(tiny_app().with_inference(InferenceCfg::DEFAULT));
        assert!(app.name().ends_with(",inf=tau10/floor50"), "{}", app.name());
        let l = leaf();
        let source = default_source();
        let make = || {
            PimcBot::new("app".into(), tiny_cfg(), l.clone(), source.clone()).with_app(app.clone())
        };
        let decisions = play_matches(&make, 31..34);
        let (mut policy, mut students) = (0, 0);
        for (_, dv) in &decisions {
            match dv.method {
                Method::ExactPimc => {
                    assert_eq!(dv.belief, BeliefMode::Policy);
                    policy += 1;
                }
                Method::Student => {
                    assert_eq!(dv.belief, BeliefMode::Uniform);
                    students += 1;
                }
                _ => {}
            }
        }
        assert!(
            policy > 20 && students > 50,
            "{policy} weighted, {students} student decisions"
        );
        assert_eq!(
            InferenceCfg::from_permille(10, 50).expect("ok"),
            InferenceCfg::DEFAULT
        );
        assert!(InferenceCfg::from_permille(0, 50).is_err());
        assert!(InferenceCfg::from_permille(10, 1001).is_err());
    }

    /// With the app off the bot is byte-identical to the plain PIMC bot: the same actions and
    /// values over whole matches.
    #[test]
    fn without_the_app_the_bot_is_the_pimc_bot_of_before() {
        let l = leaf();
        let source = default_source();
        let plain = || PimcBot::new("p".into(), tiny_cfg(), l.clone(), source.clone());
        let spec = PimcSpec::parse("pimc-v0:w12=1,w34=1,t1w=1,t2w=1,tt_mb=1").expect("ok");
        let via_factory = PimcFactory::new(&spec, l.clone(), source.clone());
        let a = play_matches(&plain, 11..14);
        let b = play_matches(&|| via_factory.build_pimc(), 11..14);
        assert_eq!(a, b);
        assert!(a.iter().all(|(_, dv)| dv.method != Method::Student));
    }

    /// The deals of the app's trick-3 table are the place's; tricks 4-5 take the one count.
    #[test]
    fn the_exact_deals_follow_the_trick_and_the_place() {
        let app = tiny_app();
        assert_eq!(app.deals_for(3, 0), 1);
        assert_eq!(app.deals_for(3, 2), 2);
        assert_eq!(app.deals_for(3, 7), 2);
        assert_eq!(app.deals_for(4, 7), 1);
        assert_eq!(app.deals_for(5, 0), 1);
        assert_eq!(
            AppPlay::APP_DEALS3,
            [192, 224, 512, 640, 1024, 1024, 1024, 1024]
        );
        assert_eq!(AppPlay::APP_DEALS45, 1024);
        let decisions = {
            let app = Arc::new(AppPlay {
                deals3: [3, 3, 3, 3, 3, 3, 3, 3],
                deals45: 2,
                ..tiny_app()
            });
            let l = leaf();
            let source = default_source();
            play_matches(
                &|| {
                    PimcBot::new("app".into(), tiny_cfg(), l.clone(), source.clone())
                        .with_app(app.clone())
                },
                21..23,
            )
        };
        // Exact decisions carry the deals they searched: 3 at trick 3, 2 at tricks 4-5 (the
        // plain cfg's w34 of 1 never shows).
        let worlds: std::collections::BTreeSet<u32> = decisions
            .iter()
            .filter(|(_, dv)| dv.method == Method::ExactPimc)
            .map(|(_, dv)| dv.n_worlds)
            .collect();
        assert_eq!(worlds, [2u32, 3].into_iter().collect());
    }

    /// The factory builds from the weights files, names the students and deals, and refuses a
    /// file that is not a student network or a zero deal count.
    #[test]
    fn the_app_factory_builds_from_bytes_and_refuses_bad_ones() {
        let (s1, s2) = students();
        let b1 = write_net(s1.layers(), &BTreeMap::new());
        let b2 = write_net(s2.layers(), &BTreeMap::new());
        let app = AppPlay::from_bytes(&b1, &b2, AppPlay::APP_DEALS3, 1024).expect("builds");
        assert_eq!(app.s1.sha256, s1.sha256);
        assert_eq!(app.s2.sha256, s2.sha256);
        assert_eq!(
            app.name(),
            format!(
                "app:s1={},s2={},deals3=192/224/512/640/1024/1024/1024/1024,deals45=1024",
                &s1.sha256[..12],
                &s2.sha256[..12]
            )
        );
        assert!(AppPlay::from_bytes(&b1[..100], &b2, AppPlay::APP_DEALS3, 1024).is_err());
        assert!(AppPlay::from_bytes(&b1, b"not a net", AppPlay::APP_DEALS3, 1024).is_err());
        assert!(AppPlay::from_bytes(&b1, &b2, [0; 8], 1024).is_err());
        assert!(AppPlay::from_bytes(&b1, &b2, AppPlay::APP_DEALS3, 0).is_err());
        let spec = PimcSpec::parse("pimc-v0").expect("ok");
        let f = PimcFactory::app(&spec, app.clone(), leaf(), default_source());
        assert_eq!(f.name(), app.name());
        assert_eq!(f.build().name(), app.name());
        let spec = PimcSpec::parse("pimc-v0:w12=8,threads=3").expect("ok");
        let f = PimcFactory::app(&spec, app.clone(), leaf(), default_source());
        assert_eq!(f.name(), format!("{}|pimc-v0:w12=8", app.name()));
    }
}
