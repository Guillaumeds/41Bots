//! The O3G2 round (`hybrid_rung` with `round`; docs/m3-hybrid/o3-rounds.md): one round of
//! G2-guided training for the oracle network O_3 on one machine, from start to shutdown.
//!
//! ```json
//! { "round": { "id": "o3r-1", "seed": 93000,
//!     "risk_model": "inputs/sha256/<hex>", "risk_power": "1", "risk_floor": "0.2",
//!     "candidates": [ { "rows": 2000000 }, { "rows": 4000000 },
//!                     { "rows": 8000000, "mode": "scratch", "widths": "4096,2048,1024,256" } ],
//!     "label_check": { "shard": "inputs/sha256/<hex>", "sidecar": "inputs/sha256/<hex>" },
//!     "g2": { "splits": [ { "name": "train", "count": 400000, "seed": 93101 },
//!                         { "name": "cal", "count": 100000, "seed": 93102 },
//!                         { "name": "test", "count": 100000, "seed": 93103 } ] } } }
//! ```
//! beside `train_mode` `"additive"` (`init_net`, `init_net_float`: the incumbent O_3), the stored
//! training set (`label_inputs`, the previous round's dataset), the stored golden set
//! (`golden_decisions_input`, `golden_worlds_input`) and `upload_shards` true.
//!
//! **The method** (research and the owner's decisions: o3-rounds.md). G2, the leaf-risk model of
//! the incumbent (`nn::leafrisk`), chooses which new trick-3 positions get exact labels: the
//! harvest's own search (driven by the incumbent) reaches positions, G2 scores them, and each risk
//! band gets new rows in proportion to its frequency × risk^α, with a plain floor
//! (`search::risk_harvest`; Yoo & Kweon 2019, Schaul et al. 2016, Hesterberg 1995).
//!
//! **Steps** (each in the heartbeat; `out/hybrid/round/` holds every report):
//! 1. **G2 for the selection**: `risk_model` (stored), or, without it, built on the incumbent
//!    first (the `g2` settings; `leafrisk::build`).
//! 2. **Risk profile** (`search risk-profile`, seed `seed`): the bands and their acceptances,
//!    shared by every part (`profile.json`).
//! 3. **New labels in nested parts**: the candidates' `rows` are cumulative (each candidate trains
//!    on the rows of the ones before it and more), so part `k` holds `rows_k − rows_{k−1}` rows,
//!    seed `seed + k`, made once and stored (`upload_shards`) as soon as it is made. On a GPU the
//!    first candidate starts training as soon as part 1 is made, while the CPU makes the rest.
//! 4. **The incumbent's golden check** (H_3 plain against the exact golden set, the 1% recheck
//!    of the stored worlds first) and its **label check** (below).
//! 5. **Candidates, overlapped**: candidate `i` trains on the stored base and parts `1..=i`
//!    (`mode` `"additive"`, the default: continued from the incumbent for `additive_epochs` at
//!    `additive_lr`; `"scratch"`: a fresh network of `widths` for `scratch_epochs` at
//!    `scratch_lr`), the GPU training candidate `i + 1` while the CPU checks candidate `i`: its
//!    golden check, its paired comparison with the incumbent (`search hybrid-paired`) and its
//!    label check.
//! 6. **The winner**: every candidate is judged against the incumbent by the additive loop's
//!    adoption rule (`additive::adopt`, the 0.01% standard, `adopt_tolerance`); among the
//!    adopted, the lowest overall 95% bound wins (ties: the earlier); none adopted keeps the
//!    incumbent.
//! 7. **The next G2** on the winner (`g2` settings; skipped when the incumbent stays: its G2 is
//!    still the right one).
//! 8. **The round's record** `round.json` (and `report.txt`): the incumbent, every candidate, the
//!    winner, the dataset for the next round (`label_inputs`: the base and every new part, whoever
//!    won: labels are never made twice) and the next round's starting settings; stored at
//!    `rounds/<id>/round.json` and content-addressed.
//!
//! **Label check** (`label_check`, optional): O_2's training labels will be hybrid (exact through
//! trick 2, then O_3), so each network's hybrid labels of the stored exact trick-2 shard's own
//! positions (`search labels` with its seed, source and start hands, `--label-net`) are compared
//! with the exact labels (`fortyone_nn.labelcmp`): the error SD and the bias per team.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde_json::{Map, Value, json};

use super::audit::additive::{Golden, Next, adopt};
use super::{
    Feed, GOLDEN_SEED, Rung, label_progress, merge_beside, read_json, round1, secs, write_json,
};
use crate::remote::{input_hex, input_key, put_input};

/// The settings inside `round`; any other key is refused.
pub(crate) const KEYS: [&str; 15] = [
    "id",
    "seed",
    "risk_model",
    "risk_power",
    "risk_floor",
    "risk_bands",
    "risk_pilot",
    "risk_cap",
    "risk_tricks",
    "risk_positions",
    "candidates",
    "scratch_epochs",
    "scratch_lr",
    "label_check",
    "g2",
];

/// The default `risk_cap`: rows a decision may give. High enough that it rarely binds: risky
/// positions bunch in a few decisions, and a low cap (the plain harvest's 32) takes them out of the
/// mix the profile asks for (measured 2026-10-21: 56% of rows in the risky band at 32, 69% at 256,
/// 80% uncapped, against a target of 86%). At 2M+ rows a round still draws on tens of thousands of
/// decisions.
const RISK_CAP: u64 = 1000;

/// The standard the adoption rule judges passes by (percent of money chance).
const STANDARD_PCT: f64 = 0.01;

/// How a candidate trains.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Mode {
    /// Continued from the incumbent (`init_net_float`).
    Additive,
    /// A fresh network of these widths.
    Scratch(String),
}

/// One candidate network.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Candidate {
    /// New rows it trains on (cumulative: the earlier candidates' and more).
    pub rows: u64,
    pub mode: Mode,
}

/// The round's settings.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RoundSettings {
    pub id: String,
    pub seed: u64,
    /// The stored G2 of the incumbent (hex); `None`: built first.
    pub risk_model: Option<String>,
    pub power: String,
    pub floor: String,
    pub bands: u64,
    pub pilot: u64,
    pub cap: u64,
    pub tricks: String,
    pub positions: String,
    pub candidates: Vec<Candidate>,
    pub scratch_epochs: u64,
    pub scratch_lr: f64,
    /// The stored exact trick-2 shard and its sidecar (hex).
    pub label_check: Option<(String, String)>,
    /// The leaf-risk build's settings (`leafrisk` params without `trick` and `net_input`).
    pub g2: Option<Value>,
}

impl RoundSettings {
    /// Part `k`'s seed (1-based).
    pub fn part_seed(&self, k: u64) -> u64 {
        self.seed + k
    }

    /// Part `k`'s rows (1-based).
    pub fn part_rows(&self, k: usize) -> u64 {
        let prev = if k > 1 {
            self.candidates[k - 2].rows
        } else {
            0
        };
        self.candidates[k - 1].rows - prev
    }

    /// The leaf-risk build's settings for network trick `cut`.
    fn g2_params(&self, cut: u64) -> Option<Value> {
        self.g2.as_ref().map(|g| {
            let mut v = g.clone();
            v["trick"] = json!(cut);
            v["net_input"] = json!("round");
            v
        })
    }

    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id, "seed": self.seed,
            "risk_model": self.risk_model.as_deref().map(input_key),
            "risk_power": self.power, "risk_floor": self.floor, "risk_bands": self.bands,
            "risk_pilot": self.pilot, "risk_cap": self.cap, "risk_tricks": self.tricks,
            "risk_positions": self.positions,
            "candidates": self.candidates.iter().map(|c| match &c.mode {
                Mode::Additive => json!({ "rows": c.rows, "mode": "additive" }),
                Mode::Scratch(w) => json!({ "rows": c.rows, "mode": "scratch", "widths": w }),
            }).collect::<Vec<_>>(),
            "scratch_epochs": self.scratch_epochs, "scratch_lr": self.scratch_lr,
            "label_check": self.label_check.as_ref().map(|(s, c)| json!({
                "shard": input_key(s), "sidecar": input_key(c) })),
            "g2": self.g2,
        })
    }
}

/// Reads `round` (`None`: no round). `seeds`: every label seed of the rung (training,
/// validation, reused), which the round's seeds must avoid. The rung's other settings a round
/// needs are checked by the caller ([`check_rung`]).
pub(crate) fn parse(
    o: &Map<String, Value>,
    cut: u64,
    seeds: &[u64],
) -> Result<Option<RoundSettings>, String> {
    let Some(v) = o.get("round").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let r = v
        .as_object()
        .ok_or("hybrid_rung: round must be an object")?;
    if let Some(k) = r.keys().find(|k| !KEYS.contains(&k.as_str())) {
        return Err(format!("hybrid_rung: round: unknown setting `{k}`"));
    }
    if cut != 3 {
        return Err(format!(
            "hybrid_rung: round is built for O_3 (cut 3, the stored trick-2 golden set); cut {cut} \
             has no golden set to judge its candidates"
        ));
    }
    let u = |k: &str, d: u64| -> Result<u64, String> {
        match r.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(x) => x
                .as_u64()
                .ok_or_else(|| format!("hybrid_rung: round.{k} must be a whole number")),
        }
    };
    let t = |k: &str, d: &str| -> Result<String, String> {
        match r.get(k) {
            None | Some(Value::Null) => Ok(d.to_string()),
            Some(Value::String(s)) => Ok(s.clone()),
            Some(Value::Number(n)) => Ok(n.to_string()),
            Some(_) => Err(format!("hybrid_rung: round.{k} must be text")),
        }
    };
    let id = t("id", "")?;
    if id.is_empty()
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(
            "hybrid_rung: round.id: a name of letters, digits, - and _ (e.g. o3r-1)".into(),
        );
    }
    let seed = u("seed", 0)?;
    let risk_model =
        match r.get("risk_model") {
            None | Some(Value::Null) => None,
            Some(Value::String(k)) => Some(input_hex(k).ok_or_else(|| {
                format!("hybrid_rung: round.risk_model {k:?}: a stored input key")
            })?),
            Some(_) => return Err("hybrid_rung: round.risk_model: a stored input key".into()),
        };
    let power = t("risk_power", "1")?;
    search::risk_harvest::parse_power(&power).map_err(|e| format!("hybrid_rung: round.{e}"))?;
    let floor = t("risk_floor", "0.2")?;
    search::risk_harvest::parse_ppm(&floor)
        .map_err(|e| format!("hybrid_rung: round.risk_floor {e}"))?;
    let bands = u("risk_bands", search::risk_harvest::RISK_BANDS as u64)?;
    let pilot = u("risk_pilot", search::risk_harvest::RISK_PILOT_DECISIONS)?;
    let cap = u("risk_cap", RISK_CAP)?;
    if !(1..=255).contains(&bands) || pilot == 0 || cap == 0 {
        return Err(
            "hybrid_rung: round: risk_bands 1 to 255, risk_pilot and risk_cap at least 1".into(),
        );
    }
    let tricks = t("risk_tricks", "2")?;
    if tricks.split(',').any(|x| !matches!(x.trim(), "1" | "2")) {
        return Err(format!(
            "hybrid_rung: round.risk_tricks {tricks:?}: decision tricks 1 and/or 2 (\"2\", the \
             default; \"1,2\": half from each, but a trick-1 search reaches about 600,000 trick-3 \
             positions against about 700 for trick 2, so trick 1 costs hours of CPU a round)"
        ));
    }
    let positions = t("risk_positions", "0.4,0.3,0.3")?;
    search::oracle_labels::parse_band_shares(&positions)
        .map_err(|e| format!("hybrid_rung: round.risk_positions: {e}"))?;
    let mut candidates = Vec::new();
    let list = r
        .get("candidates")
        .and_then(Value::as_array)
        .filter(|a| (1..=3).contains(&a.len()))
        .ok_or("hybrid_rung: round.candidates: one to three, e.g. [{\"rows\": 2000000}]")?;
    for c in list {
        let o = c
            .as_object()
            .ok_or("hybrid_rung: round.candidates: objects {rows, mode, widths}")?;
        if let Some(k) = o
            .keys()
            .find(|k| !["rows", "mode", "widths"].contains(&k.as_str()))
        {
            return Err(format!(
                "hybrid_rung: round.candidates: unknown setting `{k}`"
            ));
        }
        let rows = o
            .get("rows")
            .and_then(Value::as_u64)
            .filter(|&n| n > 0)
            .ok_or("hybrid_rung: round.candidates: rows, a whole number above 0")?;
        if candidates
            .last()
            .is_some_and(|p: &Candidate| p.rows >= rows)
        {
            return Err(
                "hybrid_rung: round.candidates: the rows grow from one candidate to the next \
                 (they are nested: each trains on the earlier ones' new rows and more)"
                    .into(),
            );
        }
        let mode = match (o.get("mode").and_then(Value::as_str), o.get("widths")) {
            (None | Some("additive"), None) => Mode::Additive,
            (None | Some("additive"), Some(_)) => {
                return Err(
                    "hybrid_rung: round.candidates: widths are for mode \"scratch\" (an additive \
                     candidate keeps the incumbent's shape)"
                        .into(),
                );
            }
            (Some("scratch"), Some(Value::String(w))) => {
                let ws = super::parse_widths_text(w)
                    .map_err(|e| format!("hybrid_rung: round.candidates: widths: {e}"))?;
                Mode::Scratch(ws.iter().map(u64::to_string).collect::<Vec<_>>().join(","))
            }
            (Some("scratch"), _) => {
                return Err(
                    "hybrid_rung: round.candidates: mode \"scratch\" needs widths (e.g. \
                     \"4096,2048,1024,256\")"
                        .into(),
                );
            }
            (Some(m), _) => {
                return Err(format!(
                    "hybrid_rung: round.candidates: mode {m:?}: \"additive\" or \"scratch\""
                ));
            }
        };
        candidates.push(Candidate { rows, mode });
    }
    let scratch_epochs = u("scratch_epochs", 40)?;
    let scratch_lr = match r.get("scratch_lr") {
        None | Some(Value::Null) => 0.001,
        Some(x) => x
            .as_f64()
            .filter(|l| *l > 0.0 && *l < 1.0)
            .ok_or("hybrid_rung: round.scratch_lr: a learning rate between 0 and 1")?,
    };
    if scratch_epochs == 0 {
        return Err("hybrid_rung: round.scratch_epochs is at least 1".into());
    }
    let label_check = match r.get("label_check") {
        None | Some(Value::Null) => None,
        Some(Value::Object(lc)) => {
            let key = |k: &str| {
                lc.get(k)
                    .and_then(Value::as_str)
                    .and_then(input_hex)
                    .ok_or_else(|| {
                        format!("hybrid_rung: round.label_check.{k}: a stored input key")
                    })
            };
            if let Some(k) = lc
                .keys()
                .find(|k| !["shard", "sidecar"].contains(&k.as_str()))
            {
                return Err(format!(
                    "hybrid_rung: round.label_check: unknown setting `{k}`"
                ));
            }
            Some((key("shard")?, key("sidecar")?))
        }
        Some(_) => return Err("hybrid_rung: round.label_check: {shard, sidecar}".into()),
    };
    let g2 = r.get("g2").filter(|v| !v.is_null()).cloned();
    if g2.is_none() && risk_model.is_none() {
        return Err(
            "hybrid_rung: round: without risk_model the incumbent's G2 is built first, which \
             needs the g2 settings"
                .into(),
        );
    }
    let s = RoundSettings {
        id,
        seed,
        risk_model,
        power,
        floor,
        bands,
        pilot,
        cap,
        tricks,
        positions,
        candidates,
        scratch_epochs,
        scratch_lr,
        label_check,
        g2,
    };
    // Every seed the round uses is its own: never a label seed of the rung, never the golden one.
    let mut mine: Vec<u64> = (0..=s.candidates.len() as u64)
        .map(|k| s.part_seed(k))
        .collect();
    if let Some(g) = s.g2_params(cut) {
        let q = crate::leafrisk::parse(&g).map_err(|e| format!("hybrid_rung: round.g2: {e}"))?;
        mine.extend(q.splits.iter().map(|x| x.seed));
    }
    if s.seed < 2 {
        return Err("hybrid_rung: round.seed: the round's base seed (2 or more)".into());
    }
    for (i, a) in mine.iter().enumerate() {
        if *a == GOLDEN_SEED || seeds.contains(a) || mine[..i].contains(a) {
            return Err(format!(
                "hybrid_rung: round: seed {a} is used twice (the parts use seed + 1, seed + 2, …; \
                 the profile seed; the g2 splits' seeds; and the rung's label seeds must all \
                 differ)"
            ));
        }
    }
    Ok(Some(s))
}

/// The rung settings a round needs (with `p` already read).
pub(crate) fn check_rung(p: &super::Params) -> Result<(), String> {
    if p.round.is_none() {
        return Ok(());
    }
    let need = |ok: bool, what: &str| -> Result<(), String> {
        if ok {
            Ok(())
        } else {
            Err(format!("hybrid_rung: round needs {what}"))
        }
    };
    need(
        p.additive.is_some(),
        "train_mode \"additive\" with init_net and init_net_float (the incumbent)",
    )?;
    need(
        p.check && p.golden_inputs.is_some(),
        "the stored golden set (golden_decisions_input, golden_worlds_input)",
    )?;
    need(p.audit.is_none(), "no audit loop (audit_rounds 0)")?;
    need(
        p.shards().is_empty() && p.label_inputs.iter().any(|x| x.set == "val"),
        "its whole base stored (label_inputs with the validation shard, and labels_random, \
         labels_rules and the other generated sets 0): every round trains on the stored base and \
         is measured on the same validation shard",
    )?;
    need(p.label_parts == 1, "label_parts 1")?;
    need(
        p.upload_shards,
        "upload_shards true (its new labels are stored for the next round)",
    )?;
    need(
        p.hybrid_labels.is_none(),
        "exact labels (label_source \"exact\": O_3's training labels are exact)",
    )?;
    Ok(())
}

/// One judged network: its files, golden check and judgement.
struct Judged {
    name: String,
    int: PathBuf,
    hc: Value,
    rec: Value,
}

impl Rung<'_> {
    fn round_dir(&self) -> (PathBuf, PathBuf) {
        (self.out.join("round"), self.work.join("round"))
    }

    /// A foreground step of the round: `c` on the threads not taken by the background training,
    /// the label progress read from its log, the training's progress beside it.
    fn round_step(
        &mut self,
        step: &str,
        c: Command,
        log: &str,
        stdout: Option<&Path>,
    ) -> Result<(), String> {
        let t = Instant::now();
        let base = self.progress(step, self.round_hb());
        self.ctx.set_phase("chunk", base.clone());
        let beside = self.beside_progress();
        let log_path = self.ctx.logs.join(log);
        let labels = log.contains("labels");
        self.ctx.run_child_io(c, log, stdout, None, &mut |ctx| {
            let mut p = base.clone();
            if labels && let Some((done, total)) = label_progress(&log_path) {
                p["rows_done"] = json!(done);
                p["rows_total"] = json!(total);
            }
            merge_beside(&mut p, &beside);
            ctx.progress = p;
        })?;
        self.done(step, t)
    }

    /// The round's heartbeat fields.
    fn round_hb(&self) -> Value {
        let s = self.p.round.as_ref().expect("a round");
        json!({ "round": s.id, "candidates": s.candidates.len() })
    }

    /// The round (module notes). `base`: the stored training shards, then the validation shard.
    pub(super) fn round_chunk(&mut self, base: Vec<PathBuf>) -> Result<Value, String> {
        let s = self.p.round.clone().expect("a round");
        let (rdir, wdir) = self.round_dir();
        for d in [&rdir, &wdir] {
            fs::create_dir_all(d).map_err(|e| format!("mkdir {}: {e}", d.display()))?;
        }
        let input = |n: &str| -> Result<PathBuf, String> {
            self.inputs
                .get(n)
                .cloned()
                .ok_or_else(|| format!("hybrid_rung: round: input {n} was not fetched"))
        };
        let (init_int, init_float) = (input("init_net")?, input("init_net_float")?);
        let (decisions, worlds) = (input("golden_decisions")?, input("golden_worlds")?);
        let gpu = self.device == "cuda";
        let mut rec = Map::new();
        rec.insert("settings".into(), s.to_json());
        rec.insert("device".into(), json!(self.device));
        // 1. G2 for the selection.
        let model = match self.inputs.get("round_risk_model").cloned() {
            Some(m) => {
                rec.insert(
                    "selection_g2".into(),
                    json!({ "stored": s.risk_model.as_deref().map(input_key) }),
                );
                m
            }
            None => {
                let (m, r) = self.round_g2(&s, &init_int, "g2-incumbent")?;
                rec.insert("selection_g2".into(), r);
                m
            }
        };
        // 2. The risk profile.
        let profile = rdir.join("profile.json");
        let threads = self.step_threads();
        let mut c = Command::new(&self.tools.search);
        c.args(["risk-profile", "--trick", &self.p.cut.to_string()])
            .args(["--seed", &s.seed.to_string()])
            .arg("--harvest-net")
            .arg(&init_int)
            .arg("--harvest-risk")
            .arg(&model)
            .arg("--equity")
            .arg(&self.table)
            .args(["--decisions", &s.pilot.to_string()])
            .args(["--bands", &s.bands.to_string()])
            .args(["--power", &s.power, "--floor", &s.floor])
            .args(["--harvest-tricks", &s.tricks])
            .args(["--harvest-positions", &s.positions])
            .args(["--start-hands", &self.p.start_hands.to_string()])
            .args(["--threads", &threads.to_string()])
            .arg("--out")
            .arg(&profile);
        self.round_step("risk profile", c, "hybrid/round-profile.log", None)?;
        rec.insert(
            "profile".into(),
            read_json(&profile)
                .ok_or("hybrid_rung: round: search risk-profile wrote no profile")?,
        );
        self.round_save(&rec)?;
        // 3. The new labels, part by part; on a GPU the first candidate trains once part 1 is in.
        let (train_base, val) = base.split_at(base.len().saturating_sub(1));
        let val = val
            .first()
            .cloned()
            .ok_or("hybrid_rung: round: no validation shard")?;
        let mut parts: Vec<PathBuf> = Vec::new();
        let mut part_recs = Vec::new();
        let files_for = |parts: &[PathBuf], i: usize| -> Vec<PathBuf> {
            let mut f = train_base.to_vec();
            f.extend(parts[..=i].iter().cloned());
            f.push(val.clone());
            f
        };
        for k in 1..=s.candidates.len() {
            let (path, pr) = self.round_part(&s, k, &init_int, &model, &profile)?;
            parts.push(path);
            part_recs.push(pr);
            rec.insert("parts".into(), json!(part_recs));
            self.round_save(&rec)?;
            if k == 1 && gpu {
                self.round_start_train(&s, 0, files_for(&parts, 0), &init_int, &init_float)?;
            }
        }
        // 4. The incumbent: its golden check (the stored worlds rechecked first) and label check.
        let inc_hc = self.round_check(&init_int, "incumbent", &decisions, &worlds, true)?;
        let mut incumbent = Judged {
            name: "incumbent".into(),
            int: init_int.clone(),
            rec: json!({ "int_key": self.p.additive.as_ref().map(|a| input_key(&a.init_net)),
                         "float_key": self.p.additive.as_ref().map(|a| input_key(&a.init_float)),
                         "golden": golden_summary(&inc_hc) }),
            hc: inc_hc,
        };
        if let Some(lc) = self.round_label_check(&s, &init_int, "incumbent")? {
            incumbent.rec["label_check"] = lc;
        }
        rec.insert("incumbent".into(), incumbent.rec.clone());
        self.round_save(&rec)?;
        // 5. The candidates: train i + 1 on the GPU while the CPU checks i.
        let mut judged: Vec<Judged> = Vec::new();
        let tol = self.p.additive.as_ref().map_or(0.002, |a| a.tolerance);
        let inc_golden = Golden::of(&incumbent.hc);
        for i in 0..s.candidates.len() {
            let key = format!("net_rc{}", i + 1);
            let step = format!("train (candidate {})", i + 1);
            let dir = rdir.join(format!("c{}", i + 1)).join("net");
            if gpu {
                self.finish_train_as(&key, &step)?;
                if i + 1 < s.candidates.len() {
                    self.round_start_train(
                        &s,
                        i + 1,
                        files_for(&parts, i + 1),
                        &init_int,
                        &init_float,
                    )?;
                }
            } else {
                self.round_setup_train(&s, i, &init_int, &init_float);
                self.train(&files_for(&parts, i), &dir, "cpu", &step, &key)?;
            }
            let int = dir.join("int.safetensors");
            let name = format!("c{}", i + 1);
            let hc = self.round_check(&int, &name, &decisions, &worlds, false)?;
            let paired = self.round_paired(&name)?;
            let g = Golden::of(&hc);
            let (adopted, worse) = adopt(&g, &inc_golden, STANDARD_PCT / 100.0, tol / 100.0);
            let train = self.rec.get(&key).cloned().unwrap_or(Value::Null);
            let c = &s.candidates[i];
            let mut cr = json!({
                "candidate": i + 1, "new_rows": c.rows,
                "mode": match &c.mode { Mode::Additive => "additive".to_string(), Mode::Scratch(w) => format!("scratch {w}") },
                "int_key": train["int_key"], "float_key": train["float_key"],
                "int_sha256": train["int_sha256"], "best_epoch": train["best_epoch"],
                "train_seconds": train["seconds"],
                "golden": golden_summary(&hc), "paired_with_incumbent": paired,
                "adopted": adopted, "worse_than_incumbent": worse,
            });
            if let Some(lc) = self.round_label_check(&s, &int, &name)? {
                cr["label_check"] = lc;
            }
            judged.push(Judged {
                name,
                int,
                hc,
                rec: cr,
            });
            rec.insert(
                "candidates".into(),
                json!(judged.iter().map(|j| j.rec.clone()).collect::<Vec<_>>()),
            );
            self.round_save(&rec)?;
        }
        // 6. The winner.
        let winner = judged
            .iter()
            .filter(|j| j.rec["adopted"] == true)
            .min_by(|a, b| {
                Golden::of(&a.hc)
                    .overall
                    .partial_cmp(&Golden::of(&b.hc).overall)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        let (wname, wint, wrec) = match winner {
            Some(j) => (j.name.clone(), j.int.clone(), j.rec.clone()),
            None => (
                incumbent.name.clone(),
                incumbent.int.clone(),
                incumbent.rec.clone(),
            ),
        };
        rec.insert(
            "winner".into(),
            json!({ "name": wname, "int_key": wrec["int_key"], "float_key": wrec["float_key"],
                    "golden": wrec["golden"],
                    "rule": "adopted over the incumbent by the additive loop's rule (no pass lost, \
                             nothing worse beyond adopt_tolerance; standard 0.01%); among the \
                             adopted, the lowest overall 95% bound; none adopted keeps the \
                             incumbent" }),
        );
        self.round_save(&rec)?;
        // 7. The next G2, on the winner (the incumbent's G2 stays when it stays).
        let next_model = if wname == "incumbent" {
            match &s.risk_model {
                Some(h) => json!(input_key(h)),
                None => rec["selection_g2"]["model"]["key"].clone(),
            }
        } else if s.g2.is_some() {
            let (_, r) = self.round_g2(&s, &wint, "g2-winner")?;
            let key = r["model"]["key"].clone();
            rec.insert("next_g2".into(), r);
            key
        } else {
            self.warn(
                "round: no g2 settings: the winner's G2 was not built (give round.g2 to build it)"
                    .into(),
            );
            Value::Null
        };
        // 8. The record: the dataset and the next round's start.
        // The stored base, then the new parts.
        let mut dataset: Vec<Value> = self.p.label_inputs.iter().map(|x| x.to_json()).collect();
        let new_entries = read_json(&rdir.join("label_inputs.json"))
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        if new_entries.len() != parts.len() {
            self.warn(format!(
                "round: {} of {} new parts were stored; the dataset lists the stored ones",
                new_entries.len(),
                parts.len()
            ));
        }
        dataset.extend(new_entries);
        let dataset_path = rdir.join("dataset.json");
        write_json(&dataset_path, &json!(dataset))?;
        let dataset_key = self.round_share(&dataset_path);
        let golden = self
            .p
            .golden_inputs
            .as_ref()
            .map(|(d, w)| (input_key(d), input_key(w)));
        rec.insert(
            "next".into(),
            json!({ "init_net": wrec["int_key"], "init_net_float": wrec["float_key"],
                    "risk_model": next_model, "label_inputs": dataset, "dataset_key": dataset_key,
                    "golden_decisions_input": golden.as_ref().map(|g| g.0.clone()),
                    "golden_worlds_input": golden.as_ref().map(|g| g.1.clone()) }),
        );
        rec.insert("seconds".into(), json!(round1(secs(self.t0))));
        rec.insert("warnings".into(), json!(self.warnings));
        self.round_save(&rec)?;
        let text = round_text(&Value::Object(rec.clone()));
        fs::write(rdir.join("report.txt"), &text).map_err(|e| e.to_string())?;
        print!("{text}");
        self.round_publish(&s, &rdir);
        self.rec.insert("round".into(), Value::Object(rec.clone()));
        self.rec.insert("seconds".into(), json!(secs(self.t0)));
        self.save()?;
        let verdict = format!("round {}: winner {wname}", s.id);
        self.ctx.set_phase(
            "chunk",
            json!({ "kind": "hybrid_rung", "step": "done", "partial_cpu_ok": true,
                    "verdict": verdict, "warnings": self.warnings }),
        );
        Ok(json!({
            "kind": "hybrid_rung",
            "verdict": verdict,
            "device": self.device,
            "round": s.id,
            "winner": wname,
            "net_int_key": wrec["int_key"],
            "net_float_key": wrec["float_key"],
            "risk_model": next_model,
            "dataset_key": dataset_key,
            "steps": self.steps,
            "warnings": self.warnings,
        }))
    }

    /// `round.json` as it grows.
    fn round_save(&mut self, rec: &Map<String, Value>) -> Result<(), String> {
        let (rdir, _) = self.round_dir();
        write_json(&rdir.join("round.json"), &Value::Object(rec.clone()))?;
        self.rec.insert("round".into(), Value::Object(rec.clone()));
        self.save()
    }

    /// Stores a file content-addressed; its key, or `None` with a warning.
    fn round_share(&mut self, path: &Path) -> Option<String> {
        let remote = self.tools.remote.clone()?;
        let p2 = path.to_path_buf();
        match self.ctx.blocking(move || put_input(&*remote, &p2, None)) {
            Ok((sha, _, _)) => Some(input_key(&sha)),
            Err(e) => {
                self.warn(format!("round: {} was not stored: {e}", path.display()));
                None
            }
        }
    }

    /// `rounds/<id>/round.json` and `report.txt` (what the next round starts from) and the
    /// record content-addressed; a failure is a warning (the files are in the run's bundle).
    fn round_publish(&mut self, s: &RoundSettings, rdir: &Path) {
        let Some(remote) = self.tools.remote.clone() else {
            self.warn("round: no remote store: round.json is only in the run's bundle".into());
            return;
        };
        for f in ["round.json", "report.txt", "profile.json"] {
            let (r, local, key) = (remote.clone(), rdir.join(f), format!("rounds/{}/{f}", s.id));
            if let Err(e) = self.ctx.blocking(move || r.put(&local, &key, false)) {
                self.warn(format!("round: rounds/{}/{f} was not stored: {e}", s.id));
            }
        }
        let _ = self.round_share(&rdir.join("round.json"));
    }

    /// Builds a leaf-risk model (G2) for the network `int` (`leafrisk::build`, the round's `g2`
    /// settings) into `round/<tag>/`; its path and record (with `model.key`).
    fn round_g2(
        &mut self,
        s: &RoundSettings,
        int: &Path,
        tag: &str,
    ) -> Result<(PathBuf, Value), String> {
        let g = s
            .g2_params(self.p.cut)
            .ok_or("hybrid_rung: round: building a G2 needs the g2 settings")?;
        let q = crate::leafrisk::parse(&g)?;
        let (rdir, wdir) = self.round_dir();
        let (out, work) = (rdir.join(tag), wdir.join(tag));
        let t = Instant::now();
        let progress = self.progress(&format!("G2 ({tag})"), self.round_hb());
        self.ctx.set_phase("chunk", progress);
        let python = self.python.clone();
        let net = int.display().to_string();
        let mut r = crate::leafrisk::build(
            self.ctx,
            self.tools,
            &q,
            &net,
            &self.table.clone(),
            &work,
            &out,
            Some(&python),
        )?;
        let model = out.join("model.json");
        r["model"]["key"] = json!(self.round_share(&model));
        r["calibration"] = read_json(&out.join("calibration.json")).unwrap_or(Value::Null);
        self.done(&format!("G2 ({tag})"), t)?;
        Ok((model, r))
    }

    /// New part `k` (1-based): the risk-guided harvest, stored with its sidecar.
    fn round_part(
        &mut self,
        s: &RoundSettings,
        k: usize,
        init_int: &Path,
        model: &Path,
        profile: &Path,
    ) -> Result<(PathBuf, Value), String> {
        let (rdir, wdir) = self.round_dir();
        let seed = s.part_seed(k as u64);
        let rows = s.part_rows(k);
        let file = format!("t{}-risk-{}-p{k}-s{seed}.npy", self.p.cut, s.id);
        let path = wdir.join(&file);
        let threads = self.step_threads();
        let step = format!("new labels (part {k}, {rows} rows)");
        let (tt_args, tt_rec) = self.label_tt_args(threads, &step, "extra");
        let mut c = Command::new(&self.tools.search);
        c.args(["labels", "--trick", &self.p.cut.to_string()])
            .args(["--count", &rows.to_string(), "--seed", &seed.to_string()])
            .args(["--source", "harvest", "--out"])
            .arg(&path)
            .arg("--equity")
            .arg(&self.table)
            .args(["--threads", &threads.to_string()])
            .args(["--start-hands", &self.p.start_hands.to_string()])
            .args(["--harvest-cap", &s.cap.to_string()])
            .args(["--harvest-positions", &s.positions])
            .args(["--harvest-tricks", &s.tricks])
            .arg("--harvest-net")
            .arg(init_int)
            .arg("--harvest-risk")
            .arg(model)
            .arg("--risk-profile")
            .arg(profile)
            .args(&tt_args);
        self.round_step(&step, c, &format!("hybrid/labels-round-p{k}.log"), None)?;
        let side = read_json(&PathBuf::from(format!("{}.json", path.display())))
            .ok_or_else(|| format!("hybrid_rung: {} wrote no sidecar", path.display()))?;
        write_json(&rdir.join(format!("{file}.json")), &side)?;
        self.note_labels(&file, "extra", &side)?;
        self.share_targeted(&rdir, &path, seed);
        Ok((
            path,
            json!({ "part": k, "file": file, "rows": side["rows"], "seed": seed,
                    "shard_sha256": side["shard_sha256"], "label_tables": tt_rec,
                    "rows_by_band": side["risk"]["rows_by_band"],
                    "rows_by_race_class": side["rows_by_race_class"],
                    "decisions_played": side["decisions_played"] }),
        ))
    }

    /// Sets the next training's start, shape, epochs and rate for candidate `i` (0-based).
    fn round_setup_train(&mut self, s: &RoundSettings, i: usize, int: &Path, float: &Path) {
        let epochs = self.p.additive.as_ref().map_or(8, |a| a.epochs);
        match &s.candidates[i].mode {
            Mode::Additive => {
                // The incumbent's own shape (checked at step 0; `widths` may be left out).
                self.shape = self.additive.as_ref().map(|a| {
                    a.init_widths
                        .iter()
                        .map(u64::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                });
                self.train_override = None;
                if let Some(a) = self.additive.as_mut() {
                    a.next = Some(Next {
                        float: float.to_path_buf(),
                        int: int.to_path_buf(),
                        epochs,
                        round: i as u64 + 1,
                    });
                }
            }
            Mode::Scratch(w) => {
                self.shape = Some(w.clone());
                self.train_override = Some((s.scratch_epochs, s.scratch_lr));
                if let Some(a) = self.additive.as_mut() {
                    a.next = None;
                }
            }
        }
    }

    /// Candidate `i` (0-based) training in the background on `files` (training shards, then the
    /// validation shard).
    fn round_start_train(
        &mut self,
        s: &RoundSettings,
        i: usize,
        files: Vec<PathBuf>,
        int: &Path,
        float: &Path,
    ) -> Result<(), String> {
        self.round_setup_train(s, i, int, float);
        let (rdir, _) = self.round_dir();
        let dir = rdir.join(format!("c{}", i + 1)).join("net");
        let prog = self.work.join(format!("net_rc{}.progress.json", i + 1));
        let _ = fs::remove_file(&prog);
        let threads = self.train_threads();
        let c = self.train_cmd(
            &Feed::Files(files),
            &dir,
            &self.device.clone(),
            threads,
            &prog,
        );
        self.start_train_with(c, dir, prog, &format!("hybrid/train-c{}.log", i + 1))
    }

    /// The golden check of the integer network `int` (H_3 plain against the exact golden set)
    /// into `round/<tag>/hc.json` (and `.txt`); `verify`: recompute `verify_pct`% of the stored
    /// worlds first (once a run).
    fn round_check(
        &mut self,
        int: &Path,
        tag: &str,
        decisions: &Path,
        worlds: &Path,
        verify: bool,
    ) -> Result<Value, String> {
        let (rdir, _) = self.round_dir();
        let dir = rdir.join(tag);
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let (json_path, txt) = (dir.join("hc.json"), dir.join("hc.txt"));
        let threads = self.step_threads();
        let verify_pct = if verify { self.p.verify_pct } else { 0.0 };
        let mut c = Command::new(&self.tools.search);
        c.args([
            "hybrid-check",
            "--trick",
            &self.p.golden_trick.to_string(),
            "--cut",
            &self.p.cut.to_string(),
        ])
        .arg("--net")
        .arg(int)
        .arg("--equity")
        .arg(&self.table)
        .arg("--decisions")
        .arg(decisions)
        .arg("--worlds")
        .arg(worlds)
        .args(["--to", &self.p.golden_worlds.to_string()])
        .args(["--seed", &self.p.golden_seed.to_string()])
        .args(["--threads", &threads.to_string()])
        .args(["--tt-mb", &self.p.tt_mb.to_string()])
        .args(["--verify-pct", &verify_pct.to_string()])
        .args(["--standard", &STANDARD_PCT.to_string()])
        .arg("--json")
        .arg(&json_path);
        self.round_step(
            &format!("golden check ({tag})"),
            c,
            &format!("hybrid/round-check-{tag}.log"),
            Some(&txt),
        )?;
        read_json(&json_path)
            .ok_or_else(|| format!("hybrid_rung: hybrid-check wrote no {}", json_path.display()))
    }

    /// Candidate `tag`'s paired comparison with the incumbent on the same golden decisions
    /// (`search hybrid-paired`, B − A with A the incumbent).
    fn round_paired(&mut self, tag: &str) -> Result<Value, String> {
        let (rdir, _) = self.round_dir();
        let out = rdir.join(tag).join("paired.json");
        let mut c = Command::new(&self.tools.search);
        c.arg("hybrid-paired")
            .arg("--a")
            .arg(rdir.join("incumbent").join("hc.json"))
            .arg("--b")
            .arg(rdir.join(tag).join("hc.json"))
            .args(["--a-name", "incumbent", "--b-name", tag])
            .arg("--json")
            .arg(&out);
        self.round_step(
            &format!("paired comparison ({tag})"),
            c,
            &format!("hybrid/round-paired-{tag}.log"),
            Some(&rdir.join(tag).join("paired.txt")),
        )?;
        Ok(read_json(&out).unwrap_or(Value::Null))
    }

    /// The label check of network `int` (module notes); `None` without `label_check`.
    fn round_label_check(
        &mut self,
        s: &RoundSettings,
        int: &Path,
        tag: &str,
    ) -> Result<Option<Value>, String> {
        if s.label_check.is_none() {
            return Ok(None);
        }
        let shard = self
            .inputs
            .get("round_check_shard")
            .cloned()
            .ok_or("hybrid_rung: round: the label check's shard was not fetched")?;
        let side = self
            .inputs
            .get("round_check_sidecar")
            .and_then(|p| read_json(p))
            .ok_or("hybrid_rung: round: the label check's sidecar is not JSON")?;
        let trick = side["trick"].as_u64().unwrap_or(0);
        if trick + 1 != self.p.cut {
            return Err(format!(
                "hybrid_rung: round.label_check: the shard holds trick-{trick} positions; the \
                 hybrid labels of O_{} value trick-{} positions",
                self.p.cut,
                self.p.cut - 1
            ));
        }
        let (rdir, wdir) = self.round_dir();
        let out = wdir.join(format!("labelcheck-{tag}.npy"));
        let rows = side["rows"]
            .as_u64()
            .or_else(|| side["count"].as_u64())
            .unwrap_or(0);
        let threads = self.step_threads();
        let mut c = Command::new(&self.tools.search);
        c.args(["labels", "--trick", &trick.to_string()])
            .args(["--count", &rows.to_string()])
            .args(["--seed", &side["seed"].as_u64().unwrap_or(0).to_string()])
            .args(["--source", side["source"].as_str().unwrap_or("random")])
            .args([
                "--start-hands",
                &side["start_hands"].as_u64().unwrap_or(20_000).to_string(),
            ])
            .arg("--label-net")
            .arg(int)
            .arg("--equity")
            .arg(&self.table)
            .args(["--threads", &threads.to_string()])
            .arg("--out")
            .arg(&out);
        self.round_step(
            &format!("label check ({tag}): hybrid labels"),
            c,
            &format!("hybrid/labels-check-{tag}.log"),
            None,
        )?;
        let report = rdir.join(tag).join("labelcheck.json");
        fs::create_dir_all(rdir.join(tag)).map_err(|e| e.to_string())?;
        let mut c = self.python_cmd("fortyone_nn.labelcmp");
        c.arg("--exact")
            .arg(&shard)
            .arg("--hybrid")
            .arg(&out)
            .arg("--out")
            .arg(&report);
        self.round_step(
            &format!("label check ({tag}): compare"),
            c,
            &format!("hybrid/labelcmp-{tag}.log"),
            None,
        )?;
        let _ = fs::remove_file(&out);
        let _ = fs::remove_file(PathBuf::from(format!("{}.json", out.display())));
        Ok(read_json(&report))
    }
}

/// The golden check's summary for the record: the overall row and the seat bands, mean and 95%
/// bound, and each team's bias (percent).
fn golden_summary(hc: &Value) -> Value {
    let g = Golden::of(hc);
    let groups = hc["groups"].as_array().cloned().unwrap_or_default();
    let row = |name: &str| {
        groups
            .iter()
            .find(|x| x["group"] == name)
            .map(|x| x["summary"].clone())
            .unwrap_or(Value::Null)
    };
    let pct = |v: &Value, k: &str| v[k].as_f64().map(|x| x * 100.0);
    let all = row("all");
    json!({
        "pass": hc["pass"],
        "loss_mean_pct": pct(&all, "loss_mean"),
        "loss_upper95_pct": g.overall * 100.0,
        "bands_upper95_pct": g.bands.iter().map(|(b, x)| json!({ "band": b, "upper95_pct": x * 100.0 })).collect::<Vec<_>>(),
        "pos1": { "mean_pct": pct(&row("pos 1"), "loss_mean"), "upper95_pct": pct(&row("pos 1"), "loss_upper95") },
        "bias_pct": g.bias.iter().map(|b| b * 100.0).collect::<Vec<_>>(),
        "hybrid_ms_per_world": hc["time"]["hybrid_ms_per_world"],
    })
}

/// The round's plain-English summary.
fn round_text(r: &Value) -> String {
    let s = &r["settings"];
    let f4 = |v: &Value| v.as_f64().map_or("-".to_string(), |x| format!("{x:.4}%"));
    let line = |name: &str, g: &Value| {
        format!(
            "  {name:<10} mean {}  95% bound {}  seat 1 {} / {}\n",
            f4(&g["loss_mean_pct"]),
            f4(&g["loss_upper95_pct"]),
            f4(&g["pos1"]["mean_pct"]),
            f4(&g["pos1"]["upper95_pct"])
        )
    };
    let mut t = format!(
        "O3G2 round {}\n\nNew labels chosen by G2 (risk power {}, plain floor {}), nested parts:\n",
        s["id"].as_str().unwrap_or("?"),
        s["risk_power"].as_str().unwrap_or("?"),
        s["risk_floor"].as_str().unwrap_or("?")
    );
    for p in r["parts"].as_array().into_iter().flatten() {
        t += &format!(
            "  part {}: {} rows (seed {})\n",
            p["part"], p["rows"], p["seed"]
        );
    }
    // The mix the profile aimed at and what the parts hold, for the riskiest tenth of the bands.
    let table = r["profile"]["table"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let top = table.len().div_ceil(10).max(1);
    let target: f64 = table
        .iter()
        .rev()
        .take(top)
        .filter_map(|b| b["target_share"].as_f64())
        .sum();
    let (mut got, mut all) = (0u64, 0u64);
    for p in r["parts"].as_array().into_iter().flatten() {
        let rows: Vec<u64> = p["rows_by_band"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_u64).collect())
            .unwrap_or_default();
        all += rows.iter().sum::<u64>();
        got += rows.iter().rev().take(top).sum::<u64>();
    }
    if all > 0 && !table.is_empty() {
        t += &format!(
            "  new rows in the riskiest {top} of {} bands: {:.1}% (the profile aimed at {:.1}%)\n",
            table.len(),
            got as f64 * 100.0 / all as f64,
            target * 100.0
        );
    }
    t += "\nGolden check, H_3 without exact leaves (decision loss):\n";
    t += &line("incumbent", &r["incumbent"]["golden"]);
    for c in r["candidates"].as_array().into_iter().flatten() {
        t += &line(&format!("c{}", c["candidate"]), &c["golden"]);
        t += &format!(
            "    {} new rows, {}; {}\n",
            c["new_rows"],
            c["mode"].as_str().unwrap_or("?"),
            if c["adopted"] == true {
                "adopted over the incumbent".to_string()
            } else {
                format!(
                    "NOT adopted: {}",
                    c["worse_than_incumbent"]
                        .as_array()
                        .map(|a| a
                            .iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join("; "))
                        .unwrap_or_default()
                )
            }
        );
        if let Some(l) = c["label_check"]["all"].as_object() {
            t += &format!(
                "    label check: rms {:.4}%, largest |bias| {:.4}%\n",
                l.get("rms_pct").and_then(Value::as_f64).unwrap_or(0.0),
                l.get("largest_abs_bias_pct")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0)
            );
        }
    }
    if let Some(l) = r["incumbent"]["label_check"]["all"].as_object() {
        t += &format!(
            "  incumbent label check: rms {:.4}%, largest |bias| {:.4}%\n",
            l.get("rms_pct").and_then(Value::as_f64).unwrap_or(0.0),
            l.get("largest_abs_bias_pct")
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
        );
    }
    t += &format!(
        "\nWinner: {}\nNext round starts from: init_net {}, G2 {}, dataset {}\n",
        r["winner"]["name"].as_str().unwrap_or("?"),
        r["next"]["init_net"].as_str().unwrap_or("-"),
        r["next"]["risk_model"].as_str().unwrap_or("-"),
        r["next"]["dataset_key"].as_str().unwrap_or("-")
    );
    for w in r["warnings"].as_array().into_iter().flatten() {
        t += &format!("warning: {}\n", w.as_str().unwrap_or(""));
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(c: char) -> String {
        format!("inputs/sha256/{}", c.to_string().repeat(64))
    }

    fn o(v: Value) -> Map<String, Value> {
        v.as_object().cloned().expect("object")
    }

    fn base() -> Value {
        json!({ "round": { "id": "o3r-1", "seed": 93000, "risk_model": key('a'),
                           "candidates": [ { "rows": 2000000 }, { "rows": 4000000 },
                                           { "rows": 8000000, "mode": "scratch",
                                             "widths": "4096,2048,1024,256" } ] } })
    }

    #[test]
    fn settings_are_read_and_checked() {
        let s = parse(&o(base()), 3, &[321, 322])
            .expect("valid")
            .expect("some");
        assert_eq!(s.part_rows(1), 2_000_000);
        assert_eq!(s.part_rows(2), 2_000_000);
        assert_eq!(s.part_rows(3), 4_000_000);
        assert_eq!((s.part_seed(1), s.part_seed(3)), (93001, 93003));
        assert_eq!(s.power, "1");
        assert_eq!(s.floor, "0.2");
        assert_eq!(s.tricks, "2");
        assert_eq!(s.cap, 1000);
        assert_eq!(
            s.candidates[2].mode,
            Mode::Scratch("4096,2048,1024,256".into())
        );
        assert_eq!(parse(&o(json!({})), 3, &[]), Ok(None));
        let bad = |f: &dyn Fn(&mut Value)| {
            let mut v = base();
            f(&mut v);
            parse(&o(v), 3, &[321]).is_err()
        };
        assert!(bad(&|v| v["round"]["typo"] = json!(1)));
        assert!(bad(&|v| v["round"]["id"] = json!("bad id")));
        assert!(bad(
            &|v| v["round"]["candidates"][1]["rows"] = json!(2000000)
        ));
        assert!(bad(&|v| v["round"]["candidates"] = json!([])));
        assert!(bad(&|v| v["round"]["candidates"][0]["widths"] = json!("64")));
        assert!(bad(&|v| v["round"]["candidates"][2]["widths"] = json!(null)));
        assert!(bad(
            &|v| v["round"]["candidates"][2]["mode"] = json!("other")
        ));
        assert!(bad(&|v| v["round"]["risk_power"] = json!("0.6")));
        assert!(bad(&|v| v["round"]["risk_floor"] = json!("2")));
        assert!(bad(&|v| v["round"]["risk_tricks"] = json!("3")));
        assert!(bad(&|v| v["round"]["risk_model"] = json!(null)));
        assert!(bad(&|v| v["round"]["seed"] = json!(320)));
        assert!(bad(
            &|v| v["round"]["label_check"] = json!({ "shard": key('b') })
        ));
        // A part seed that is a rung label seed.
        let mut v = base();
        v["round"]["seed"] = json!(319);
        assert!(parse(&o(v), 3, &[321]).is_err());
        // Only O_3 (cut 3).
        assert!(parse(&o(base()), 2, &[]).is_err());
        // Without a stored G2 the g2 settings are needed, and their seeds count.
        let mut v = base();
        v["round"]["risk_model"] = json!(null);
        v["round"]["g2"] = json!({ "splits": [
            { "name": "train", "count": 100, "seed": 93101 },
            { "name": "cal", "count": 100, "seed": 93102 },
            { "name": "test", "count": 100, "seed": 93103 } ] });
        assert!(parse(&o(v.clone()), 3, &[]).expect("valid").is_some());
        v["round"]["g2"]["splits"][0]["seed"] = json!(93001);
        assert!(parse(&o(v), 3, &[]).is_err());
        let back = s.to_json();
        assert_eq!(back["candidates"][2]["widths"], "4096,2048,1024,256");
        assert_eq!(back["risk_model"], key('a'));
    }

    #[test]
    fn the_report_reads_the_record() {
        let r = json!({
            "settings": { "id": "o3r-1", "risk_power": "1", "risk_floor": "0.2" },
            "parts": [ { "part": 1, "rows": 10, "seed": 5 } ],
            "incumbent": { "golden": { "loss_mean_pct": 0.0056, "loss_upper95_pct": 0.0068,
                                       "pos1": { "mean_pct": 0.0154, "upper95_pct": 0.0201 } } },
            "candidates": [ { "candidate": 1, "new_rows": 10, "mode": "additive", "adopted": false,
                              "worse_than_incumbent": ["overall bound worse"],
                              "golden": { "loss_mean_pct": 0.006 } } ],
            "winner": { "name": "incumbent" },
            "next": { "init_net": key('a') },
            "warnings": [],
        });
        let t = round_text(&r);
        assert!(t.contains("O3G2 round o3r-1"));
        assert!(t.contains("incumbent  mean 0.0056%  95% bound 0.0068%  seat 1 0.0154% / 0.0201%"));
        assert!(t.contains("NOT adopted: overall bound worse"));
        assert!(t.contains("Winner: incumbent"));
    }
}
