//! The audit loop's training modes (docs/m3-hybrid/audit-loop.md §10; owner, 2026-10-17), and the
//! leader focus of its reports (plan item 2d).
//!
//! **`train_mode`** (`"scratch"`, the default: the loop exactly as before, a fresh network each
//! round and the capacity ladder; or `"additive"`). In **additive** mode:
//! - `init_net` (required; a stored integer network's key, `inputs/sha256/<hex>`, as `rung.json`'s
//!   `net.int_key` gives it) and `init_net_float` (required; the key of the same network's float
//!   weights, `float.safetensors`, `rung.json`'s `net.float_key` since this build; for the networks
//!   of earlier runs, see audit-loop.md §10) are fetched and checked at step 0: the integer network
//!   was trained for the rung's trick and head, its hidden widths are `widths` when that is given
//!   (else they become the rung's), and the float file has the same layer shapes and the same
//!   training record (metadata). Round 1's "training" then proves, byte for byte, that the float
//!   weights quantise to `init_net` (`train.py --init-int`).
//! - **Round 1's network is `init_net` as it is** (`train.py --init --epochs 0`: copied and
//!   measured on the validation shard, nothing trained): its audit shard, audit, golden check, mini
//!   golden sets and targeted harvest run as in scratch mode. It is the best so far.
//! - **Round r ≥ 2 continues the best network so far** (`train.py --init <best float>`) on every
//!   training shard so far, the base sets and every targeted shard (old and new data together:
//!   new data alone makes a network forget, Kirkpatrick et al. 2017), for `additive_epochs` (8)
//!   epochs at `additive_lr` (0.0003), with early stopping on the validation shard as before;
//!   `shrink_perturb` (λ, off by default; 0 < λ < 1): every weight λ·w + N(0, σ²) first, σ = (1 − λ)
//!   times the spread of the layer's fresh initialisation (`train.py`'s `SIGMA_RULE`), seeded with
//!   `train_seed × 1000 + r` (Ash & Adams, "On warm-starting neural network training", NeurIPS
//!   2020).
//! - **Never lose a won pass:** round r's network is adopted as the new best only when, on the
//!   golden set, its overall 95% upper bound of the decision loss, each seat band's (positions 1-2,
//!   3-4, 5-8) and every team's |bias| on the overall row are each no more than the best's plus
//!   `adopt_tolerance` (percentage points of money chance; 0 by default). Otherwise the best stays,
//!   the round is reported "not adopted" with every number that was worse, its targeted harvest
//!   still joins the next round's training, the groups the best still fails stay failing whatever
//!   the rejected network measured, and the round cannot end the loop as flat. The chunk's
//!   network is the adopted best (not `best_round`'s ranking).
//! - The capacity ladder is off: every round trains `init_net`'s shape (`audit_shapes` refused).
//!   The rounds run one after another (a training starts from the network adopted after the
//!   previous round's golden check, so it cannot start beside that check; `overlap` refused): the
//!   additive trainings are short, and the loop's CPU work is the long part.
//! - **Focused harvest:** the targeted harvest takes the groups still failing at the decision
//!   level, the groups confirmed this round that have not passed at the decision level (failing,
//!   not yet measured, or unmeasurable), and the groups confirmed for the first time this round;
//!   a known group that passes is not harvested again.
//! - Refused with it: `epochs` and `lr` (use `additive_epochs` and `additive_lr`), `label_parts`
//!   above 1, `audit_shapes`, `overlap`; it needs the audit loop (`audit_rounds` above 0). With it,
//!   `harvest_net` is not needed at trick 3 and earlier (round 1's audit shard is driven by
//!   `init_net`).
//!
//! **Leader focus** (both modes): the round reports give the golden check by the decider's seat
//! band and position (the leader's own team's bias first), and `harvest_seat_weights` (`"w12,w34,w58"`,
//! e.g. `"2,1,1"`; none by default) gives every harvested row (the rung's harvest and race shards,
//! the targeted harvests) a loss weight by its decision's seat band, through the per-row weights
//! (`NAME.npy.rowweights`, written by `search labels --seat-weights` before the shard is renamed
//! into place); each weight a multiple of 1/256 from 1/256 to 65535/256.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use super::super::Rung;
use super::{Loop, Round, RoundResult, UNFILLED, pct4};
use crate::remote::{input_hex, input_key};

/// The settings of this module; any other key is the rung's or the audit loop's.
pub(crate) const KEYS: [&str; 8] = [
    "train_mode",
    "init_net",
    "init_net_float",
    "additive_epochs",
    "additive_lr",
    "shrink_perturb",
    "adopt_tolerance",
    "harvest_seat_weights",
];

/// The settings that belong to `train_mode` `"additive"` only.
const ADDITIVE_ONLY: [&str; 6] = [
    "init_net",
    "init_net_float",
    "additive_epochs",
    "additive_lr",
    "shrink_perturb",
    "adopt_tolerance",
];

/// The seat bands of the golden check (`search hybrid-check`'s groups), the decider's position.
pub(crate) const BANDS: [&str; 3] = ["pos 1-2", "pos 3-4", "pos 5-8"];

/// `train_mode` `"additive"`'s settings (module notes).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Additive {
    /// `init_net`: the SHA-256 of the starting integer network.
    pub init_net: String,
    /// `init_net_float`: the SHA-256 of its float weights.
    pub init_float: String,
    /// `additive_epochs` (8).
    pub epochs: u64,
    /// `additive_lr` (0.0003).
    pub lr: f64,
    /// `shrink_perturb`: λ, or none.
    pub shrink_perturb: Option<f64>,
    /// `adopt_tolerance` (0): percentage points of money chance.
    pub tolerance: f64,
    /// Whether `widths` was given (it must then be `init_net`'s).
    pub widths_given: bool,
}

impl Additive {
    pub fn to_json(&self) -> Value {
        json!({ "train_mode": "additive", "init_net": input_key(&self.init_net),
                "init_net_float": input_key(&self.init_float), "additive_epochs": self.epochs,
                "additive_lr": self.lr, "shrink_perturb": self.shrink_perturb,
                "adopt_tolerance_pct": self.tolerance })
    }
}

/// Reads `train_mode` and its settings (`None`: scratch). `audit_rounds`, `label_parts`: the
/// rung's.
pub(crate) fn parse(
    o: &Map<String, Value>,
    audit_rounds: u64,
    label_parts: u64,
) -> Result<Option<Additive>, String> {
    let mode = match o.get("train_mode") {
        None | Some(Value::Null) => "scratch",
        Some(Value::String(s)) if s == "scratch" || s == "additive" => s.as_str(),
        Some(v) => {
            return Err(format!(
                "hybrid_rung: train_mode {v}: \"scratch\" (a fresh network each round, the \
                 default) or \"additive\" (continue a stored network)"
            ));
        }
    };
    if mode == "scratch" {
        if let Some(k) = ADDITIVE_ONLY.iter().find(|k| o.contains_key(**k)) {
            return Err(format!(
                "hybrid_rung: {k} is set but train_mode is \"scratch\": it belongs to train_mode \
                 \"additive\""
            ));
        }
        return Ok(None);
    }
    if audit_rounds == 0 {
        return Err(
            "hybrid_rung: train_mode \"additive\" needs the audit loop (audit_rounds above 0): \
             each round is kept only if the golden set says it is no worse"
                .into(),
        );
    }
    for (k, why) in [
        (
            "audit_shapes",
            "the shape is init_net's: the capacity ladder is off",
        ),
        ("epochs", "use additive_epochs"),
        ("lr", "use additive_lr"),
        (
            "overlap",
            "the rounds run one after another (each training starts from the network adopted \
             after the previous round's golden check)",
        ),
    ] {
        if o.contains_key(k) {
            return Err(format!(
                "hybrid_rung: {k} is not used with train_mode \"additive\" ({why}); remove it"
            ));
        }
    }
    if label_parts > 1 {
        return Err(
            "hybrid_rung: label_parts above 1 streams a training beside the labels, but with \
             train_mode \"additive\" round 1 does not train: use label_parts 1"
                .into(),
        );
    }
    let key = |k: &str| -> Result<String, String> {
        match o.get(k) {
            Some(Value::String(v)) => input_hex(v).ok_or_else(|| {
                format!(
                    "hybrid_rung: {k} {v:?} must be a stored input's key, \"inputs/sha256/<64 hex \
                     digits>\""
                )
            }),
            None | Some(Value::Null) => Err(format!(
                "hybrid_rung: train_mode \"additive\" needs {k} (a stored network's key, \
                 \"inputs/sha256/<64 hex digits>\")"
            )),
            Some(_) => Err(format!(
                "hybrid_rung: {k} must be a string, \"inputs/sha256/<64 hex digits>\""
            )),
        }
    };
    let (init_net, init_float) = (key("init_net")?, key("init_net_float")?);
    if init_net == init_float {
        return Err(
            "hybrid_rung: init_net (the integer network) and init_net_float (its float weights) \
             are different files"
                .into(),
        );
    }
    let num = |k: &str, d: f64, ok: &dyn Fn(f64) -> bool, rule: &str| -> Result<f64, String> {
        match o.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(v) => v
                .as_f64()
                .filter(|x| x.is_finite() && ok(*x))
                .ok_or_else(|| format!("hybrid_rung: {k} must be {rule}")),
        }
    };
    let epochs = match o.get("additive_epochs") {
        None | Some(Value::Null) => 8,
        Some(v) => v
            .as_u64()
            .filter(|e| (1..=1000).contains(e))
            .ok_or("hybrid_rung: additive_epochs must be a whole number from 1 to 1000")?,
    };
    let lr = num(
        "additive_lr",
        0.0003,
        &|x| x > 0.0 && x <= 1.0,
        "a number above 0 (at most 1)",
    )?;
    let shrink_perturb = match o.get("shrink_perturb") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_f64()
                .filter(|x| x.is_finite() && *x > 0.0 && *x < 1.0)
                .ok_or("hybrid_rung: shrink_perturb is the shrink factor λ: above 0 and below 1")?,
        ),
    };
    let tolerance = num(
        "adopt_tolerance",
        0.0,
        &|x| (0.0..=1.0).contains(&x),
        "percentage points of money chance, from 0 to 1",
    )?;
    Ok(Some(Additive {
        init_net,
        init_float,
        epochs,
        lr,
        shrink_perturb,
        tolerance,
        widths_given: o.contains_key("widths"),
    }))
}

/// `harvest_seat_weights` (`"w12,w34,w58"`): three loss weights by the decision's seat band, each a
/// multiple of 1/256 from 1/256 to 255 (`nn::rowweights::fixed`); `None` when absent. Returned
/// as written to `search labels --seat-weights`.
pub(crate) fn parse_seat_weights(o: &Map<String, Value>) -> Result<Option<String>, String> {
    let text = match o.get("harvest_seat_weights") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::String(s)) => s.clone(),
        Some(_) => {
            return Err(
                "hybrid_rung: harvest_seat_weights must be a string such as \"2,1,1\" (seats \
                 1-2, 3-4, 5-8)"
                    .into(),
            );
        }
    };
    search::oracle_labels::parse_seat_weights(&text)
        .map_err(|e| format!("hybrid_rung: harvest_seat_weights {text:?}: {e}"))?;
    Ok(Some(text))
}

/// A golden check's numbers the adoption rule compares: the overall 95% upper bound of the
/// decision loss, each seat band's, and every team's bias on the overall row.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Golden {
    pub overall: f64,
    pub bands: Vec<(String, f64)>,
    pub bias: Vec<f64>,
}

impl Golden {
    /// From a `search hybrid-check` JSON (`groups`: `all`, then the bands and positions).
    pub fn of(hc: &Value) -> Golden {
        let groups = hc["groups"].as_array().cloned().unwrap_or_default();
        let find = |name: &str| {
            groups
                .iter()
                .find(|g| g["group"] == name)
                .map(|g| g["summary"].clone())
        };
        let all = find("all")
            .or_else(|| groups.first().map(|g| g["summary"].clone()))
            .unwrap_or(Value::Null);
        Golden {
            overall: all["loss_upper95"].as_f64().unwrap_or(f64::INFINITY),
            bands: BANDS
                .iter()
                .filter_map(|b| {
                    find(b).map(|s| {
                        (
                            (*b).to_string(),
                            s["loss_upper95"].as_f64().unwrap_or(f64::INFINITY),
                        )
                    })
                })
                .collect(),
            bias: all["bias"]
                .as_array()
                .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect())
                .unwrap_or_default(),
        }
    }

    fn to_json(&self) -> Value {
        json!({ "loss_upper95": self.overall,
                "bands": self.bands.iter().map(|(b, u)| (b.clone(), json!(u))).collect::<Map<_, _>>(),
                "bias": self.bias })
    }
}

/// The adoption rule ("never lose a won pass"): `new` is adopted over `best` when its overall
/// bound, each of the best's seat bands' bounds and every team's |bias| are each at most the
/// best's plus `tol` (a fraction of money chance). Returns whether, and every number that was
/// worse (empty when adopted).
pub(crate) fn adopt(new: &Golden, best: &Golden, tol: f64) -> (bool, Vec<String>) {
    let mut worse = Vec::new();
    let p = |x: f64| pct4(Some(x));
    if new.overall.is_nan() || new.overall > best.overall + tol {
        worse.push(format!(
            "overall bound {} > the best's {}",
            p(new.overall),
            p(best.overall)
        ));
    }
    for (band, b) in &best.bands {
        let n = new
            .bands
            .iter()
            .find(|(x, _)| x == band)
            .map_or(f64::INFINITY, |x| x.1);
        if n.is_nan() || n > b + tol {
            worse.push(format!("{band} bound {} > the best's {}", p(n), p(*b)));
        }
    }
    for (t, b) in best.bias.iter().enumerate() {
        let n = new.bias.get(t).copied().unwrap_or(f64::INFINITY);
        if n.is_nan() || n.abs() > b.abs() + tol {
            worse.push(format!(
                "team {t}'s |bias| {} > the best's {}",
                p(n.abs()),
                p(b.abs())
            ));
        }
    }
    (worse.is_empty(), worse)
}

/// The golden check by the decider's seat (`check_summary`'s `bands`): each band and position
/// group's decisions, 95% upper bound of the decision loss, the leader's own team's bias (team 0)
/// and every team's.
pub(crate) fn bands_of(hc: &Value) -> Value {
    Value::Array(
        hc["groups"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter(|g| g["group"].as_str().is_some_and(|n| n.starts_with("pos ")))
                    .map(|g| {
                        let s = &g["summary"];
                        json!({ "group": g["group"], "decisions": s["decisions"],
                                "loss_upper95": s["loss_upper95"], "loss_pass": s["loss_pass"],
                                "bias": s["bias"], "bias_pass": s["bias_pass"] })
                    })
                    .collect()
            })
            .unwrap_or_default(),
    )
}

/// The round report's "leader focus" lines (both modes), from `check_summary`'s `bands`.
pub(crate) fn focus_text(golden: &Value) -> String {
    let Some(bands) = golden["bands"].as_array().filter(|b| !b.is_empty()) else {
        return String::new();
    };
    let one = |x: &Value| {
        format!(
            "{}: {} decisions, 95% upper {}, own team's bias {}",
            x["group"].as_str().unwrap_or("?").replace("pos ", "seat "),
            x["decisions"],
            pct4(x["loss_upper95"].as_f64()),
            x["bias"][0]
                .as_f64()
                .map_or("?".into(), |b| format!("{:+.4}%", 100.0 * b))
        )
    };
    let wide: Vec<String> = bands
        .iter()
        .filter(|x| x["group"].as_str().is_some_and(|g| g.contains('-')))
        .map(one)
        .collect();
    let lead: Vec<String> = bands
        .iter()
        .filter(|x| x["group"] == "pos 1" || x["group"] == "pos 2")
        .map(one)
        .collect();
    let mut t = format!(
        "Leader focus (the golden set by the decider's seat; seat 1 leads the trick): {}.\n",
        wide.join("; ")
    );
    if !lead.is_empty() {
        t += &format!("  The leader and the second seat: {}.\n", lead.join("; "));
    }
    t
}

/// The focused harvest's share of round `r`'s confirmed groups (`confirmed`, union indices): those
/// first confirmed in round `r` (`first`: each group's first round) or without a pass at the
/// decision level (`passed`: each group's latest verdict, `Some(true)` a pass).
pub(crate) fn focused(
    confirmed: &[usize],
    first: &[Option<u64>],
    passed: &[Option<bool>],
    r: u64,
) -> Vec<usize> {
    confirmed
        .iter()
        .copied()
        .filter(|&i| {
            first.get(i).copied().flatten() == Some(r)
                || passed.get(i).copied().flatten() != Some(true)
        })
        .collect()
}

/// The best network so far, adopted.
#[derive(Clone, Debug)]
pub(crate) struct Best {
    pub round: u64,
    pub golden: Golden,
    pub float: PathBuf,
    pub int: PathBuf,
}

/// The start of the next training.
#[derive(Clone, Debug)]
pub(crate) struct Next {
    pub float: PathBuf,
    pub int: PathBuf,
    pub epochs: u64,
    /// The round it trains (for the perturbation's seed).
    pub round: u64,
}

/// The additive loop's state (`Rung::additive`).
#[derive(Clone, Debug, Default)]
pub(crate) struct Run {
    /// `init_net`'s hidden widths (checked at step 0).
    pub init_widths: Vec<u64>,
    pub best: Option<Best>,
    /// Each judged round: adopted or not.
    pub decided: BTreeMap<u64, bool>,
    /// The failing groups before a not-adopted round's measurements (restored after them).
    pub before: Vec<Option<u64>>,
    /// Each union group's latest decision-level verdict (`Some(true)`: passed).
    pub passed: Vec<Option<bool>>,
    pub next: Option<Next>,
    /// Each judged round's adoption record.
    pub records: BTreeMap<u64, Value>,
}

/// A stored network file's facts: hidden widths (from its tensors' shapes), trick, head and its
/// metadata.
type NetFacts = (Vec<u64>, String, String, BTreeMap<String, String>);

/// The [`NetFacts`] of the network file at `path`.
fn net_facts(path: &Path) -> Result<NetFacts, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let f = nn::safetensors::read(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let layers = f.tensors.keys().filter(|k| k.starts_with('w')).count();
    let mut widths = Vec::new();
    for k in 0..layers {
        let w = f.get(&format!("w{k}"))?;
        if w.shape.len() != 2 {
            return Err(format!("{}: w{k} is not a matrix", path.display()));
        }
        if k + 1 < layers {
            widths.push(w.shape[1] as u64);
        }
    }
    Ok((
        widths,
        f.meta("trick").unwrap_or("?").to_string(),
        f.meta("head").unwrap_or("?").to_string(),
        f.meta,
    ))
}

impl Rung<'_> {
    /// Step 0 in additive mode, after the stored inputs are fetched: `init_net` is an integer
    /// network for the rung's trick and head, of `widths` when given (else its widths become the
    /// rung's), and `init_net_float` has its layer shapes and its training record. Records
    /// `rung.json`'s `additive`.
    pub(in crate::hybrid) fn additive_check_init(&mut self) -> Result<(), String> {
        let Some(a) = self.p.additive.clone() else {
            return Ok(());
        };
        let (int, float) = (
            self.inputs.get("init_net").cloned(),
            self.inputs.get("init_net_float").cloned(),
        );
        let (Some(int), Some(float)) = (int, float) else {
            return Err("hybrid_rung: init_net and init_net_float were not fetched".into());
        };
        let (iw, itrick, ihead, imeta) = net_facts(&int).map_err(|e| format!("init_net: {e}"))?;
        if imeta.get("format").map(String::as_str) != Some(nn::oracle::FORMAT) {
            return Err(format!(
                "hybrid_rung: init_net {} is not an integer network ({})",
                input_key(&a.init_net),
                nn::oracle::FORMAT
            ));
        }
        let head = if self.p.head == "outcome" {
            "outcome-v1"
        } else {
            "money4"
        };
        if itrick != self.p.cut.to_string() || ihead != head {
            return Err(format!(
                "hybrid_rung: init_net was trained for trick {itrick} with the {ihead} head, but \
                 the rung is cut {} with the {head} head",
                self.p.cut
            ));
        }
        let text = |w: &[u64]| w.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
        if a.widths_given && iw != self.p.widths {
            return Err(format!(
                "hybrid_rung: widths {} disagree with init_net's {}: in additive mode the network \
                 keeps its shape (leave widths out, or give init_net's)",
                text(&self.p.widths),
                text(&iw)
            ));
        }
        nn::mlp::check_hidden_widths(&iw.iter().map(|&w| w as usize).collect::<Vec<_>>())
            .map_err(|e| format!("hybrid_rung: init_net: {e}"))?;
        let (fw, _, _, fmeta) = net_facts(&float).map_err(|e| format!("init_net_float: {e}"))?;
        let same_record = fmeta
            .iter()
            .filter(|(k, _)| !["trick", "head"].contains(&k.as_str()))
            .all(|(k, v)| imeta.get(k) == Some(v));
        if fw != iw || !same_record || fmeta.contains_key("format") {
            return Err(format!(
                "hybrid_rung: init_net_float {} is not init_net's float weights (widths {} against \
                 {}, or another training's record)",
                input_key(&a.init_float),
                text(&fw),
                text(&iw)
            ));
        }
        self.rec.insert(
            "additive".into(),
            json!({ "init_net": input_key(&a.init_net), "init_net_float": input_key(&a.init_float),
                    "init_widths": text(&iw), "init_trick": itrick, "init_head": ihead,
                    "init_record": imeta.iter().filter(|(k, _)| k.as_str() != "train_sha256")
                        .map(|(k, v)| (k.clone(), json!(v))).collect::<Map<_, _>>(),
                    "check": "same layer shapes and training record; round 1 proves the float \
                              weights quantise to init_net byte for byte" }),
        );
        self.additive = Some(Run {
            init_widths: iw,
            next: Some(Next {
                float,
                int,
                epochs: 0,
                round: 1,
            }),
            ..Run::default()
        });
        Ok(())
    }

    /// The audit loop's start in additive mode: the ladder is the one shape, `init_net`'s.
    pub(super) fn additive_start(&mut self, st: &mut Loop) {
        if let Some(a) = &self.additive {
            st.s.shapes = vec![a.init_widths.clone()];
        }
    }

    /// The trainer's epochs and learning rate: in additive mode the next training's
    /// (`additive_epochs`, 0 for round 1) and `additive_lr`, else `epochs` and `lr`.
    pub(in crate::hybrid) fn epochs_lr(&self) -> (u64, f64) {
        match (
            &self.p.additive,
            self.additive.as_ref().and_then(|a| a.next.as_ref()),
        ) {
            (Some(s), Some(n)) => (n.epochs, s.lr),
            _ => (self.p.epochs, self.p.lr),
        }
    }

    /// The trainer's start in additive mode: `--init` (and `--init-int`, and the
    /// shrink-and-perturb when asked, never for round 1's network taken as it is).
    pub(in crate::hybrid) fn init_args(&self) -> Vec<String> {
        let (Some(s), Some(n)) = (
            &self.p.additive,
            self.additive.as_ref().and_then(|a| a.next.as_ref()),
        ) else {
            return Vec::new();
        };
        let mut v = vec![
            "--init".to_string(),
            n.float.display().to_string(),
            "--init-int".into(),
            n.int.display().to_string(),
        ];
        if let (Some(l), true) = (s.shrink_perturb, n.epochs > 0) {
            v.extend([
                "--shrink-perturb".into(),
                l.to_string(),
                "--perturb-seed".into(),
                (self
                    .p
                    .train_seed
                    .saturating_mul(1000)
                    .saturating_add(n.round))
                .to_string(),
            ]);
        }
        v
    }

    /// Round `p.r`'s adoption (additive mode), on its first golden check (`tag` empty): adopted
    /// when no worse than the best ([`adopt`]; round 1, the starting network, always). Sets the
    /// next training's start (the best), and notes the failing groups so that a round not adopted
    /// cannot clear them ([`Rung::additive_failing`]).
    pub(super) fn additive_judge(&mut self, st: &mut Loop, p: &Round, hc: &Value, tag: &str) {
        let Some(s) = self.p.additive.clone() else {
            return;
        };
        let Some(a) = self.additive.as_mut() else {
            return;
        };
        if !tag.is_empty() || a.decided.contains_key(&p.r) {
            return;
        }
        let g = Golden::of(hc);
        let (adopted, worse) = match &a.best {
            None => (true, Vec::new()),
            Some(b) => adopt(&g, &b.golden, s.tolerance / 100.0),
        };
        let before = a.best.as_ref().map(|b| (b.round, b.golden.clone()));
        a.before = st.groups.iter().map(|g| g.failing_since).collect();
        a.decided.insert(p.r, adopted);
        if adopted {
            a.best = Some(Best {
                round: p.r,
                golden: g.clone(),
                float: p.net.with_file_name("float.safetensors"),
                int: p.net.clone(),
            });
        }
        let best = a.best.clone().expect("a best network after round 1");
        a.next = Some(Next {
            float: best.float.clone(),
            int: best.int.clone(),
            epochs: s.epochs,
            round: p.r + 1,
        });
        let line = match (&before, adopted) {
            (None, _) => format!(
                "round {}: the starting network (init_net) taken as it is: the best so far",
                p.r
            ),
            (Some((b, _)), true) => format!(
                "round {}: adopted (no worse than round {b} on the golden set: overall, every \
                 seat band and every team's bias); round {} continues from it",
                p.r,
                p.r + 1
            ),
            (Some((b, _)), false) => format!(
                "round {}: NOT adopted ({}); the best stays round {b}, which round {} continues; \
                 this round's targeted data still joins the training",
                p.r,
                worse.join("; "),
                p.r + 1
            ),
        };
        println!("runner: hybrid_rung: {line}");
        st.lines.push(line);
        // The heartbeat: the best so far and the latest round's adoption.
        self.audit_hb.insert(
            "additive".into(),
            json!(format!(
                "best: round {}; round {} {}",
                best.round,
                p.r,
                if adopted { "adopted" } else { "not adopted" }
            )),
        );
        let tol_pct = s.tolerance;
        a.records.insert(
            p.r,
            json!({ "adopted": adopted, "best_round": best.round,
                    "compared_with": before.as_ref().map(|(b, g)| json!({ "round": b, "golden": g.to_json() })),
                    "golden": g.to_json(), "worse": worse, "tolerance_pct": tol_pct,
                    "next_training": { "round": p.r + 1, "from_round": best.round,
                                       "epochs": s.epochs, "lr": s.lr,
                                       "shrink_perturb": s.shrink_perturb } }),
        );
    }

    /// After a round's decision-level measurements (`update_failing`): each group's latest
    /// verdict is noted (the focused harvest), and when the round was not adopted the groups the
    /// best was failing stay failing.
    pub(super) fn additive_failing(&mut self, st: &mut Loop, judged: &[Value], r: u64) {
        let Some(a) = self.additive.as_mut() else {
            return;
        };
        a.passed.resize(st.groups.len(), None);
        for (i, x) in judged.iter().enumerate().take(st.groups.len()) {
            match x["pass"].as_bool() {
                Some(b) => a.passed[i] = Some(b),
                None if x["basis"] == UNFILLED => a.passed[i] = None,
                None => {}
            }
        }
        if a.decided.get(&r) == Some(&false) {
            for (g, before) in st.groups.iter_mut().zip(&a.before) {
                if g.failing_since.is_none() && before.is_some() {
                    g.failing_since = *before;
                }
            }
        }
    }

    /// Whether round `r` may end the loop as flat: in additive mode only an adopted round.
    pub(super) fn additive_may_be_flat(&self, r: u64) -> bool {
        self.additive
            .as_ref()
            .is_none_or(|a| a.decided.get(&r) != Some(&false))
    }

    /// The round's confirmed groups the targeted harvest takes: all of them in scratch mode; in
    /// additive mode those confirmed for the first time this round or not passing at the
    /// decision level (failing, not yet measured, unmeasurable).
    pub(super) fn additive_targets(&self, st: &Loop, p: &Round) -> Vec<usize> {
        let Some(a) = &self.additive else {
            return p.confirmed.clone();
        };
        let first: Vec<Option<u64>> = st
            .groups
            .iter()
            .map(|g| g.confirmed.first().copied())
            .collect();
        focused(&p.confirmed, &first, &a.passed, p.r)
    }

    /// The round whose network is the chunk's in additive mode: the adopted best.
    pub(super) fn additive_best<'a>(&self, st: &'a Loop) -> Option<&'a RoundResult> {
        let b = self.additive.as_ref()?.best.as_ref()?.round;
        st.results.iter().find(|x| x.r == b)
    }

    /// Round `r`'s report fields: its adoption (additive mode) and the leader focus.
    pub(super) fn additive_fields(&self, mut rec: Value, r: u64, golden: &Value) -> Value {
        rec["leader_focus"] = golden["bands"].clone();
        if let Some(a) = &self.additive {
            rec["additive"] = a.records.get(&r).cloned().unwrap_or(Value::Null);
        }
        rec
    }

    /// Round `r`'s report lines: its adoption (additive mode), then the leader focus.
    pub(super) fn additive_text(&self, r: u64, golden: &Value) -> String {
        let mut t = String::new();
        if let Some(x) = self.additive.as_ref().and_then(|a| a.records.get(&r)) {
            t += &match (x["adopted"].as_bool(), x["compared_with"].is_null()) {
                (_, true) => "Additive: the starting network (init_net), taken as it is, is the \
                              best so far.\n"
                    .to_string(),
                (Some(true), _) => format!(
                    "Additive: adopted as the best (no worse than round {} on the golden set: \
                     overall, every seat band and every team's bias, tolerance {}%).\n",
                    x["compared_with"]["round"], x["tolerance_pct"]
                ),
                _ => format!(
                    "Additive: NOT adopted, the best stays round {}: {}. Its targeted data still \
                     joins the next round's training.\n",
                    x["best_round"],
                    x["worse"]
                        .as_array()
                        .map(|w| w
                            .iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join("; "))
                        .unwrap_or_default()
                ),
            };
        }
        t + &focus_text(golden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn o(v: Value) -> Map<String, Value> {
        v.as_object().cloned().expect("object")
    }

    fn key(c: char) -> String {
        format!("inputs/sha256/{}", c.to_string().repeat(64))
    }

    #[test]
    fn modes_are_read_and_checked() {
        assert_eq!(parse(&o(json!({})), 3, 1), Ok(None));
        assert_eq!(
            parse(&o(json!({ "train_mode": "scratch" })), 3, 1),
            Ok(None)
        );
        let a = parse(
            &o(json!({ "train_mode": "additive", "init_net": key('a'),
                         "init_net_float": format!("sha256:{}", "b".repeat(64)) })),
            3,
            1,
        )
        .expect("additive")
        .expect("some");
        assert_eq!(
            (
                a.epochs,
                a.lr,
                a.shrink_perturb,
                a.tolerance,
                a.widths_given
            ),
            (8, 0.0003, None, 0.0, false)
        );
        assert_eq!(
            (a.init_net.as_str(), a.init_float.as_str()),
            ("a".repeat(64).as_str(), "b".repeat(64).as_str())
        );
        let a = parse(
            &o(
                json!({ "train_mode": "additive", "init_net": key('a'), "init_net_float": key('b'),
                         "additive_epochs": 4, "additive_lr": 0.0001, "shrink_perturb": 0.9,
                         "adopt_tolerance": 0.0005, "widths": "16,8" }),
            ),
            3,
            1,
        )
        .expect("additive")
        .expect("some");
        assert_eq!(
            (
                a.epochs,
                a.lr,
                a.shrink_perturb,
                a.tolerance,
                a.widths_given
            ),
            (4, 0.0001, Some(0.9), 0.0005, true)
        );
        assert_eq!(a.to_json()["init_net"], key('a'));
        let base =
            json!({ "train_mode": "additive", "init_net": key('a'), "init_net_float": key('b') });
        let refused = |extra: Value, rounds: u64, parts: u64| {
            let mut v = base.clone();
            for (k, x) in o(extra) {
                v[k] = x;
            }
            parse(&o(v), rounds, parts).expect_err("refused")
        };
        for (extra, rounds, parts, want) in [
            (json!({}), 0, 1, "needs the audit loop"),
            (json!({}), 3, 2, "label_parts above 1"),
            (json!({ "train_mode": "fine" }), 3, 1, "train_mode \"fine\""),
            (json!({ "init_net": Value::Null }), 3, 1, "needs init_net"),
            (
                json!({ "init_net_float": 7 }),
                3,
                1,
                "init_net_float must be a string",
            ),
            (
                json!({ "init_net": "f5766aa9" }),
                3,
                1,
                "stored input's key",
            ),
            (
                json!({ "init_net_float": key('a') }),
                3,
                1,
                "different files",
            ),
            (
                json!({ "audit_shapes": "<widths>" }),
                3,
                1,
                "audit_shapes is not used",
            ),
            (json!({ "epochs": 8 }), 3, 1, "use additive_epochs"),
            (json!({ "lr": 0.001 }), 3, 1, "use additive_lr"),
            (json!({ "overlap": true }), 3, 1, "one after another"),
            (json!({ "additive_epochs": 0 }), 3, 1, "additive_epochs"),
            (json!({ "additive_lr": 0 }), 3, 1, "additive_lr"),
            (json!({ "shrink_perturb": 1.0 }), 3, 1, "shrink_perturb"),
            (json!({ "shrink_perturb": 0 }), 3, 1, "shrink_perturb"),
            (json!({ "adopt_tolerance": -0.1 }), 3, 1, "adopt_tolerance"),
            (json!({ "adopt_tolerance": "0" }), 3, 1, "adopt_tolerance"),
        ] {
            let e = refused(extra.clone(), rounds, parts);
            assert!(e.contains(want), "{extra}: {e}");
        }
        for k in ADDITIVE_ONLY {
            let e = parse(&o(json!({ k: 1 })), 3, 1).expect_err(k);
            assert!(e.contains("train_mode is \"scratch\""), "{e}");
        }
    }

    #[test]
    fn seat_weights_are_read_and_checked() {
        assert_eq!(parse_seat_weights(&o(json!({}))), Ok(None));
        assert_eq!(
            parse_seat_weights(&o(json!({ "harvest_seat_weights": "2,1,1" }))),
            Ok(Some("2,1,1".into()))
        );
        assert!(parse_seat_weights(&o(json!({ "harvest_seat_weights": "1.5,1,0.5" }))).is_ok());
        for bad in [
            json!("2,1"),
            json!("2,1,1,1"),
            json!("0,1,1"),
            json!("1.001,1,1"),
            json!("256,1,1"),
            json!("x,1,1"),
            json!(2),
        ] {
            assert!(
                parse_seat_weights(&o(json!({ "harvest_seat_weights": bad.clone() }))).is_err(),
                "{bad}"
            );
        }
    }

    fn golden(overall: f64, bands: [f64; 3], bias: [f64; 4]) -> Golden {
        Golden {
            overall,
            bands: BANDS
                .iter()
                .zip(bands)
                .map(|(b, u)| ((*b).to_string(), u))
                .collect(),
            bias: bias.to_vec(),
        }
    }

    /// The adoption rule: no worse on the overall bound, every seat band and every team's |bias|
    /// (with the tolerance); the reasons name every number that was worse.
    #[test]
    fn adoption_rule() {
        let best = golden(
            0.000068,
            [0.000122, 0.000080, 0.000052],
            [-0.000085, 0.000054, 0.000036, -0.000005],
        );
        assert_eq!(adopt(&best, &best, 0.0), (true, vec![]), "the same numbers");
        let better = golden(
            0.000060,
            [0.000100, 0.000080, 0.000050],
            [0.000040, -0.000054, 0.0, 0.000005],
        );
        assert!(
            adopt(&better, &best, 0.0).0,
            "|bias| is compared, not the sign"
        );
        // Better overall, worse on one band: not adopted.
        let band = golden(
            0.000050,
            [0.000130, 0.000070, 0.000040],
            [0.0, 0.0, 0.0, 0.0],
        );
        let (ok, why) = adopt(&band, &best, 0.0);
        assert!(!ok);
        assert_eq!(why.len(), 1, "{why:?}");
        assert!(
            why[0].starts_with("pos 1-2 bound 0.0130% > the best's 0.0122%"),
            "{why:?}"
        );
        // Within the tolerance (0.001 percentage points = 0.00001): adopted.
        assert!(adopt(&band, &best, 0.00001).0);
        // One team's bias worse.
        let bias = golden(
            0.000060,
            [0.000100, 0.000070, 0.000050],
            [-0.000086, 0.0, 0.0, 0.0],
        );
        let (ok, why) = adopt(&bias, &best, 0.0);
        assert!(
            !ok && why.len() == 1 && why[0].starts_with("team 0's |bias|"),
            "{why:?}"
        );
        // Everything worse: every number named.
        let worse = golden(
            0.0001,
            [0.0002, 0.0002, 0.0002],
            [0.001, 0.001, 0.001, 0.001],
        );
        assert_eq!(adopt(&worse, &best, 0.0).1.len(), 1 + 3 + 4);
        // A missing number is never better.
        let missing = Golden {
            overall: 0.00001,
            bands: vec![],
            bias: vec![],
        };
        assert_eq!(adopt(&missing, &best, 0.0).1.len(), 3 + 4);
        assert!(!adopt(&golden(f64::NAN, [0.0; 3], [0.0; 4]), &best, 0.0).0);
    }

    /// The focused harvest: a group confirmed again that passes is left out; new ones, failing
    /// ones and ones not yet measured (or unmeasurable) are taken.
    #[test]
    fn focused_harvest() {
        let first = [Some(1), Some(1), Some(2), Some(1), Some(1)];
        let passed = [Some(true), Some(false), Some(true), None, Some(true)];
        assert_eq!(focused(&[0, 1, 2, 3], &first, &passed, 2), vec![1, 2, 3]);
        assert_eq!(
            focused(&[0, 4], &first, &passed, 1),
            vec![0, 4],
            "round 1: all new"
        );
        assert_eq!(focused(&[], &first, &passed, 2), Vec::<usize>::new());
    }

    #[test]
    fn golden_numbers_and_bands_from_a_check() {
        let hc = json!({ "groups": [
            { "group": "all", "summary": { "loss_upper95": 0.00007, "bias": [-0.00008, 0.00005, 0.00004, -0.00001], "decisions": 2000 } },
            { "group": "pos 1-2", "summary": { "loss_upper95": 0.00012, "bias": [-0.00024, 0.0, 0.0, 0.0], "decisions": 504, "loss_pass": false } },
            { "group": "pos 3-4", "summary": { "loss_upper95": 0.00008, "bias": [0.0, 0.0, 0.0, 0.0], "decisions": 504 } },
            { "group": "pos 5-8", "summary": { "loss_upper95": 0.00005, "bias": [0.0, 0.0, 0.0, 0.0], "decisions": 992 } },
            { "group": "pos 1", "summary": { "loss_upper95": 0.0002, "bias": [-0.00024, 0.0, 0.0, 0.0], "decisions": 252 } }
        ] });
        let g = Golden::of(&hc);
        assert_eq!(g.overall, 0.00007);
        assert_eq!(
            g.bands,
            vec![
                ("pos 1-2".to_string(), 0.00012),
                ("pos 3-4".to_string(), 0.00008),
                ("pos 5-8".to_string(), 0.00005)
            ]
        );
        assert_eq!(g.bias.len(), 4);
        let bands = bands_of(&hc);
        assert_eq!(bands.as_array().map(Vec::len), Some(4));
        let t = focus_text(&json!({ "bands": bands }));
        assert!(
            t.contains(
                "seat 1-2: 504 decisions, 95% upper 0.0120%, own team's bias -0.0240%; seat 3-4"
            ),
            "{t}"
        );
        assert!(
            t.contains("The leader and the second seat: seat 1: 252 decisions"),
            "{t}"
        );
        assert_eq!(focus_text(&json!({})), "");
    }
}
