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
use engine::view::SeatView;
use equity::EquityTable;

use crate::labels;
use crate::pimc::{Ctx, DecisionValues, Early, SearchConfig, SharedLeaf, evaluate};
use crate::worlds::{RejectionSource, WorldSource};

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

/// The PIMC bot.
pub struct PimcBot {
    name: String,
    cfg: SearchConfig,
    leaf: Arc<SharedLeaf>,
    source: Arc<dyn WorldSource>,
    rules: Rules,
    seed: u64,
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
            pending_play: None,
            last: None,
            log: None,
        }
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
        let dseed = decision_seed(self.seed, view);
        let ctx = Ctx {
            leaf: &*self.leaf,
            source: &*self.source,
            rules: self.rules,
        };
        let dv = evaluate(view, legal, dseed, &ctx, &self.cfg);
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
}

impl PimcFactory {
    /// A factory for a parsed spec with this leaf and world source.
    pub fn new(spec: &PimcSpec, leaf: Arc<SharedLeaf>, source: Arc<dyn WorldSource>) -> Self {
        PimcFactory {
            name: spec.canonical(),
            cfg: spec.cfg,
            leaf,
            source,
        }
    }

    /// Build a concrete bot (for callers that need the decision log).
    pub fn build_pimc(&self) -> PimcBot {
        PimcBot::new(
            self.name.clone(),
            self.cfg,
            self.leaf.clone(),
            self.source.clone(),
        )
    }
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
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let sha = equity::sha256::sha256_hex(&bytes);
    if sha != EQUITY_SHA256 {
        return Err(format!(
            "{path}: sha256 {sha} is not the production equity table {EQUITY_SHA256}"
        ));
    }
    EquityTable::from_npy(&bytes).map_err(|e| format!("{path}: {e}"))
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

/// The default world generator.
pub fn default_source() -> Arc<dyn WorldSource> {
    Arc::new(RejectionSource::new())
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
}
