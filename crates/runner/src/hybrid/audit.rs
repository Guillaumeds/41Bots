//! The audit loop of a `hybrid_rung` chunk (docs/m3-hybrid/audit-loop.md §4 and §6 B3): one
//! rental trains, audits, checks and retrains, round after round, until the network is "flat" or
//! a limit is reached; nobody restarts it between rounds.
//!
//! **Settings** (all refused unless `audit_rounds` is above 0, and the loop needs the golden set:
//! not with `"check": false`):
//! - `audit_rounds` (0: no loop, the chunk is exactly as before): R_max, at most 49;
//! - `audit_rows` (1,000,000): rows of each round's audit shard;
//! - `audit_seed` (301): round `r`'s audit shard has the seed `audit_seed × 100 + r`, its targeted
//!   harvest `audit_seed × 100 + 50 + r` (a pure function of the manifest; refused when one meets
//!   a training or validation seed, or the golden sets' seed 1);
//! - `audit_keep_cap` (256): in-group positions kept per decision by the targeted harvest
//!   (`search labels --harvest-keep … --keep-cap`; it keeps in-group positions only, so every row
//!   of a targeted shard is in a weak group);
//! - `audit_weight` (2): the loss weight of every targeted shard's rows (the trainer's per-shard
//!   weight, `NAME.npy.weight`; 1 writes no weight file);
//! - `audit_mini_decisions` (50): the size of a group's mini golden set, made when the group's
//!   golden slice holds fewer decisions;
//! - `audit_target_rows` (0: computed): rows of each targeted harvest. Computed, it lifts every
//!   confirmed group's share of the training set to at least 3× its natural share (plan §4 step
//!   5): with `T` training rows and a group of weight `w` (the auditor's share of the audit shard),
//!   `⌈2wT / (1 − 3w)⌉` rows for it (all of `T` when `w ≥ 1/3`), summed over the round's groups,
//!   at least 1,000 and at most `T`;
//! - `audit_predicates_input` (none; a stored input's key): a stand-in for the auditor (dry runs
//!   of the loop's machinery): every round's confirmed groups are this file's, so the loop runs to
//!   `audit_rounds`.
//!
//! **A round** `r` (step names carry the round: `train (round r)`, …):
//! 1. **Train** network `r` on the training shards (random, rules, harvest, race, then every
//!    targeted shard so far, each with its weight), into `out/hybrid/net/` for round 1 and
//!    `out/hybrid/audit/round-<r>/net/` after; the integer network is shared content-addressed.
//! 2. **Audit shard** (`search labels --source harvest`, `audit_rows` rows, the round's seed, the
//!    rung's `harvest_cap_per_decision` and `harvest_positions`): frontier positions as the hybrid
//!    reaches them, labelled exactly. Driven by exact values at trick 4 or later, by the previous
//!    round's network at trick 3 and earlier (round 1: `harvest_net`, else exact).
//! 3. **Audit** (`python -m fortyone_nn.audit --net NET --shard SHARD --out DIR --seed S`, the
//!    auditor of build B1, `S` the audit shard's seed): `predicates.json` (the confirmed groups),
//!    `report.json` (`audit.candidates`: groups found on half A; `audit.confirmed`), `report.txt`.
//! 4. **Golden slices** (`search hybrid-check --slices UNION`): network `r` on the golden set, with
//!    a slice for every group confirmed in this or an earlier round (the union, deduplicated by
//!    conditions, `out/hybrid/audit/round-<r>/union.json`); `verify_pct` in round 1, 0 after (the
//!    same worlds). **Mini golden sets**: a union group whose slice holds fewer than
//!    `audit_mini_decisions` decisions gets its own set, once (`search decisions --keep` with
//!    network `r`, first candidate `golden_count`, then `search worlds --method exact`;
//!    `out/hybrid/audit/mini/m<k>/`), and every mini set is checked with each round's network
//!    (its decision loss 95% upper bound ≤ 0.01%). A group whose mini set finds no decision in
//!    the candidates drawn is reported as unmeasured at the decision level, not failed.
//! 5. **Targeted harvest** (`search labels --source harvest --harvest-keep PRED --harvest-cap 0
//!    --keep-cap audit_keep_cap`, network `r` driving at trick 3 and earlier, exact at trick 4):
//!    the round's confirmed groups, `audit_target_rows` rows, weight `audit_weight`; the shard
//!    joins the next round's training.
//!
//! **Stop** (plan §4): *flat* when a round's audit confirms no group and its golden check passes
//! (overall, every band, every team's bias, every group's slice) with every mini set; when a
//! round's audit confirms no group but a check fails, the loop stops too (nothing is left to
//! target: the next lever is network capacity); *round limit* after `audit_rounds` rounds with
//! groups still confirmed (the last round's slices and mini sets are still checked, for the
//! report); *stopped* on the stop file or the launcher's STOP (after the current step, everything
//! so far written and uploaded), the reason naming a dollar cap when the stop does.
//!
//! **GPU and CPU together** (plan §4, owner 2026-10-16 21:15): on a GPU with `overlap`, round `r`'s
//! training runs in the background (`Rung::start_train_with`, one child beside the steps), and
//! the CPU steps that do not need network `r` run beside it on the threads it leaves: in round 1
//! the golden set (when generated), then for round `r − 1` its targeted harvest (written into the
//! directory the trainer watches, `--watch-dir … --expect`, so it joins between epochs; the
//! trainer's time limit counts again from when it lands), its golden slices, its mini golden
//! sets; then round `r`'s audit shard. The audit and the decision to go on follow the training.
//! On the CPU (or `overlap` false) the same steps run one after another, the targeted harvest
//! before the training it feeds. Either way every answer is the same: each child gets a fixed
//! thread count and the labels, checks and sets are byte-identical for any.
//!
//! **Reporting**: every heartbeat carries `round` (`r of R`), `step`, `groups_found`,
//! `groups_confirmed` (the latest audit's), `groups_remaining` (union groups still out of spec
//! after the latest round's checks: confirmed again, or failing their slice or mini set),
//! `worst_group` (`name: weight …, bias …, SD …`, the auditor's most severe), `slices_failing`
//! (`k of n`, with the worst) and `last_check` (the latest golden check's overall row). Each round
//! writes `out/hybrid/audit/round-<r>.txt` (plain English) and `round-<r>.json`; the loop ends with
//! `out/hybrid/audit/audit-summary.txt` (one line per round, then the stop reason). Each round's
//! network, predicate files and reports are shared content-addressed (`inputs/sha256/<hex>`) and
//! are in the bundle. `rung.json`'s `audit` holds every round's record; its `net` and `check` are
//! the last round's, as the chunk's result.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};

use super::{GOLDEN_SEED, Rung, label_progress, merge_beside, read_json, secs, write_json};
use crate::remote::{input_hex, input_key, put_input};

/// The audit loop's settings beside `audit_rounds`, refused when it is 0.
const AUDIT_KEYS: [&str; 7] = [
    "audit_rows",
    "audit_seed",
    "audit_keep_cap",
    "audit_weight",
    "audit_mini_decisions",
    "audit_target_rows",
    "audit_predicates_input",
];

/// A decision is in a group's golden slice (and a mini set's candidate is kept) when at least
/// this share of its reached frontier positions are in the group.
const SLICE_SHARE: &str = "0.5";

/// Worlds searched per candidate of a mini golden set.
const MINI_WORLDS: u64 = 8;

/// The auditor's time limit.
const AUDIT_LIMIT: Duration = Duration::from_secs(90 * 60);

/// The audit loop's settings (module notes).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AuditSettings {
    pub rounds: u64,
    pub rows: u64,
    pub seed: u64,
    pub keep_cap: u64,
    pub weight: f64,
    pub mini_decisions: u64,
    pub target_rows: u64,
    /// `audit_predicates_input`: the SHA-256 of a stored predicate file standing in for the
    /// auditor.
    pub predicates_input: Option<String>,
}

impl AuditSettings {
    /// Round `r`'s audit shard seed.
    pub fn audit_seed(&self, r: u64) -> u64 {
        self.seed * 100 + r
    }

    /// Round `r`'s targeted harvest seed.
    pub fn target_seed(&self, r: u64) -> u64 {
        self.seed * 100 + 50 + r
    }

    pub fn to_json(&self) -> Value {
        json!({ "rounds": self.rounds, "rows": self.rows, "seed": self.seed,
                "keep_cap": self.keep_cap, "weight": self.weight,
                "mini_decisions": self.mini_decisions, "target_rows": self.target_rows,
                "predicates_input": self.predicates_input.as_deref().map(input_key),
                "audit_seeds": (1..=self.rounds).map(|r| self.audit_seed(r)).collect::<Vec<_>>(),
                "target_seeds": (1..=self.rounds).map(|r| self.target_seed(r)).collect::<Vec<_>>() })
    }
}

/// Reads the audit settings of a `hybrid_rung` chunk (`o`; `check`: the chunk's, `seeds`: every
/// training and validation shard's seed). `None` when `audit_rounds` is absent or 0.
pub(crate) fn parse_settings(
    o: &Map<String, Value>,
    check: bool,
    seeds: &[u64],
) -> Result<Option<AuditSettings>, String> {
    let u = |k: &str, d: u64| -> Result<u64, String> {
        match o.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(v) => v
                .as_u64()
                .ok_or_else(|| format!("hybrid_rung: {k} must be a whole number")),
        }
    };
    let rounds = u("audit_rounds", 0)?;
    if rounds == 0 {
        if let Some(k) = AUDIT_KEYS.iter().find(|k| o.contains_key(**k)) {
            return Err(format!(
                "hybrid_rung: {k} is set but audit_rounds is 0: the audit loop's settings need \
                 audit_rounds above 0"
            ));
        }
        return Ok(None);
    }
    if rounds > 49 {
        return Err("hybrid_rung: audit_rounds is at most 49".into());
    }
    if !check {
        return Err(
            "hybrid_rung: the audit loop checks every round on the golden set: it needs \
             \"check\": true"
                .into(),
        );
    }
    if o.get("compare_cpu").and_then(Value::as_bool) == Some(true) {
        return Err(
            "hybrid_rung: compare_cpu is not run in the audit loop (one network a round); remove \
             it"
            .into(),
        );
    }
    let rows = u("audit_rows", 1_000_000)?;
    if rows == 0 {
        return Err("hybrid_rung: audit_rows must be at least 1".into());
    }
    let seed = u("audit_seed", 301)?;
    if seed == 0
        || seed
            .checked_mul(100)
            .and_then(|x| x.checked_add(99))
            .is_none()
    {
        return Err(
            "hybrid_rung: audit_seed must be a whole number from 1 (seeds of the rounds: \
                    audit_seed × 100 + r and audit_seed × 100 + 50 + r)"
                .into(),
        );
    }
    let keep_cap = u("audit_keep_cap", u64::from(search::oracle_labels::KEEP_CAP))?;
    if !(1..=u64::from(u32::MAX)).contains(&keep_cap) {
        return Err("hybrid_rung: audit_keep_cap must be at least 1".into());
    }
    let weight = match o.get("audit_weight") {
        None | Some(Value::Null) => 2.0,
        Some(v) => v
            .as_f64()
            .filter(|w| w.is_finite() && *w > 0.0 && *w <= 100.0)
            .ok_or("hybrid_rung: audit_weight must be a number above 0 (at most 100)")?,
    };
    let mini_decisions = u("audit_mini_decisions", 50)?;
    if !(1..=10_000).contains(&mini_decisions) {
        return Err("hybrid_rung: audit_mini_decisions must be 1 to 10000".into());
    }
    let target_rows = u("audit_target_rows", 0)?;
    let predicates_input = match o.get("audit_predicates_input") {
        None | Some(Value::Null) => None,
        Some(Value::String(v)) => Some(input_hex(v).ok_or_else(|| {
            format!(
                "hybrid_rung: audit_predicates_input {v:?} must be a stored input's key, \
                 \"inputs/sha256/<64 hex digits>\" (or \"sha256:<64 hex digits>\")"
            )
        })?),
        Some(_) => {
            return Err("hybrid_rung: audit_predicates_input must be a string, \
                        \"inputs/sha256/<64 hex digits>\""
                .into());
        }
    };
    let s = AuditSettings {
        rounds,
        rows,
        seed,
        keep_cap,
        weight,
        mini_decisions,
        target_rows,
        predicates_input,
    };
    for r in 1..=rounds {
        for (what, x) in [
            ("audit shard", s.audit_seed(r)),
            ("targeted harvest", s.target_seed(r)),
        ] {
            if x == GOLDEN_SEED || seeds.contains(&x) {
                return Err(format!(
                    "hybrid_rung: round {r}'s {what} would have seed {x} (audit_seed {seed}), \
                     which a training or validation shard (or the golden sets) uses; choose \
                     another audit_seed"
                ));
            }
        }
    }
    Ok(Some(s))
}

/// A group the loop has met (the union over its rounds).
struct Group {
    /// Its name in the union (unique).
    name: String,
    /// Its conditions as text (`nn::predicate`): its identity across rounds.
    key: String,
    /// The auditor's latest entry for it (conditions, weight, bias, SD, …), named `name`.
    entry: Value,
    /// The rounds whose audit confirmed it.
    confirmed: Vec<u64>,
    /// Its mini golden set, once made.
    mini: Option<Mini>,
}

/// A group's mini golden set.
struct Mini {
    /// `m<k>`.
    id: String,
    dir: PathBuf,
    /// The group alone, as a predicate file.
    pred: PathBuf,
    decisions: PathBuf,
    /// The exact worlds (`None`: no decision was found).
    worlds: Option<PathBuf>,
    kept: u64,
    drawn: u64,
    made_round: u64,
}

/// One round's network and audit.
struct Round {
    r: u64,
    /// The integer network.
    net: PathBuf,
    net_sha: String,
    net_key: Option<String>,
    /// The training's record (`rung.json`'s for the round).
    train: Value,
    /// The audit shard's facts.
    shard: Value,
    /// The auditor's counts and report.
    found: u64,
    auditor: Value,
    /// This round's confirmed groups (indices in the union) and their predicate file.
    confirmed: Vec<usize>,
    preds: Option<PathBuf>,
}

/// The loop's state.
struct Loop {
    s: AuditSettings,
    groups: Vec<Group>,
    /// Training shards in order (the base sets, then the targeted ones).
    train: Vec<PathBuf>,
    train_rows: u64,
    val: PathBuf,
    golden: Option<(PathBuf, PathBuf)>,
    /// `out/hybrid/audit`, `hybrid-work/audit`.
    dir: PathBuf,
    wdir: PathBuf,
    /// Summary lines, one per round.
    lines: Vec<String>,
    /// Every round's record.
    rounds: Vec<Value>,
    /// The last golden check (its JSON and text paths).
    last_check: Option<(Value, PathBuf, PathBuf)>,
    /// The last round's network record.
    last_net: Option<Value>,
    /// Whether the last round's checks passed (flat).
    flat: bool,
    /// Sharing to the remote store failed or is impossible (warned once).
    share_warned: bool,
}

/// The stop reason for a stop request: `dollar cap` when the request names one.
fn stop_reason(why: &str) -> String {
    let w = why.to_lowercase();
    if w.contains("dollar") || w.contains("cost cap") || w.contains("usd") {
        format!("dollar cap ({why})")
    } else {
        format!("stopped ({why})")
    }
}

/// A percentage of a fraction, 4 decimals.
fn pct4(x: Option<f64>) -> String {
    x.map_or("n/a".into(), |x| format!("{:.4}%", 100.0 * x))
}

/// The largest absolute bias of an auditor's entry (a list per team, or one number).
fn worst_bias(v: &Value) -> Option<f64> {
    match v {
        Value::Array(a) => a
            .iter()
            .filter_map(Value::as_f64)
            .max_by(|x, y| x.abs().total_cmp(&y.abs())),
        x => x.as_f64(),
    }
}

/// `name: weight …, bias …, SD …` of an auditor's entry.
fn describe_group(g: &Value) -> String {
    let p2 = |x: Option<f64>| x.map_or("?".into(), |x| format!("{:.2}%", 100.0 * x));
    format!(
        "{}: weight {}, bias {}, SD {}",
        g["name"].as_str().unwrap_or("?"),
        p2(g["weight"].as_f64()),
        worst_bias(&g["bias"]).map_or("?".into(), |b| format!("{:+.2}%", 100.0 * b)),
        p2(g["sd"].as_f64())
    )
}

/// The groups of a predicate file as written (a list, or `{"groups": [...]}`), checked with
/// `nn::predicate`, and each one's conditions as text.
fn read_groups(path: &Path) -> Result<Vec<(Value, String)>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let p = nn::predicate::Predicates::parse(&bytes)
        .map_err(|e| format!("hybrid_rung: {}: {e}", path.display()))?;
    let v: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let list = match v {
        Value::Array(a) => a,
        Value::Object(mut o) => match o.remove("groups") {
            Some(Value::Array(a)) => a,
            _ => Vec::new(),
        },
        _ => Vec::new(),
    };
    Ok(list
        .into_iter()
        .zip(p.groups.iter().map(nn::predicate::Group::conditions_text))
        .collect())
}

/// A golden check's summary from its JSON: pass, the overall row, the failing slices and the
/// worst of them.
fn check_summary(hc: &Value) -> Value {
    let all = &hc["groups"][0]["summary"];
    let slices: Vec<&Value> = hc["slices"]["groups"]
        .as_array()
        .map(|a| a.iter().filter(|x| !x["group"].is_null()).collect())
        .unwrap_or_default();
    let failing: Vec<&Value> = slices
        .iter()
        .copied()
        .filter(|x| x["pass"].as_bool() == Some(false))
        .collect();
    let worst = failing.iter().max_by(|a, b| {
        a["summary"]["loss_upper95"]
            .as_f64()
            .unwrap_or(0.0)
            .total_cmp(&b["summary"]["loss_upper95"].as_f64().unwrap_or(0.0))
    });
    json!({
        "pass": hc["pass"].as_bool().unwrap_or(false),
        "pass_without_slices": hc["pass_without_slices"].as_bool().or_else(|| hc["pass"].as_bool()).unwrap_or(false),
        "decisions": all["decisions"], "mean_loss": all["mean_loss"],
        "loss_upper95": all["loss_upper95"], "loss_pass": all["loss_pass"], "bias": all["bias"],
        "sd": all["sd"],
        "slices": slices.iter().map(|x| json!({
            "name": x["name"], "decisions": x["ids"].as_array().map_or(0, Vec::len),
            "loss_upper95": x["summary"]["loss_upper95"], "pass": x["pass"],
            "reached_share": x["reached_share"] })).collect::<Vec<_>>(),
        "slices_failing": failing.len(),
        "worst_slice": worst.map(|w| json!({ "name": w["name"], "loss_upper95": w["summary"]["loss_upper95"] })),
    })
}

impl Rung<'_> {
    /// The chunk's steps with the audit loop, after the labels (`shards`: the training shards,
    /// then the validation one): the rounds (module notes), the summary, and the chunk's result.
    pub(super) fn audit_chunk(&mut self, shards: Vec<PathBuf>) -> Result<Value, String> {
        let s = self.p.audit.clone().expect("the audit loop's settings");
        let (train, val) = shards.split_at(shards.len() - 1);
        let dir = self.out.join("audit");
        let wdir = self.work.join("audit");
        for d in [&dir, &wdir] {
            fs::create_dir_all(d).map_err(|e| format!("mkdir {}: {e}", d.display()))?;
        }
        let mut st = Loop {
            train_rows: self.p.train.iter().map(|x| x.count).sum(),
            s,
            groups: Vec::new(),
            train: train.to_vec(),
            val: val[0].clone(),
            golden: None,
            dir,
            wdir,
            lines: Vec::new(),
            rounds: Vec::new(),
            last_check: None,
            last_net: None,
            flat: false,
            share_warned: false,
        };
        let outcome = self.audit_rounds(&mut st);
        let stop = match &outcome {
            Ok(why) => why.clone(),
            Err(e) => match crate::stop::requested() {
                Some(why) => stop_reason(&why),
                None => format!("failed: {e}"),
            },
        };
        self.audit_hb.insert("audit_stop".into(), json!(stop));
        let summary = st.lines.join("\n")
            + if st.lines.is_empty() { "" } else { "\n" }
            + &format!(
                "stop: {stop}\nfinal network: {}\n",
                st.last_net.as_ref().map_or("none".into(), |n| format!(
                    "{} (sha256 {}, key {})",
                    n["dir"].as_str().unwrap_or("?"),
                    n["int_sha256"].as_str().unwrap_or("?"),
                    n["int_key"].as_str().unwrap_or("not shared")
                ))
            );
        let summary_path = st.dir.join("audit-summary.txt");
        fs::write(&summary_path, &summary)
            .map_err(|e| format!("write {}: {e}", summary_path.display()))?;
        print!("{summary}");
        let summary_key = self.share_file(&mut st, &summary_path);
        self.rec.insert(
            "audit".into(),
            json!({ "settings": st.s.to_json(), "rounds": st.rounds, "stop": stop,
                    "flat": st.flat, "summary": "audit/audit-summary.txt",
                    "summary_key": summary_key }),
        );
        self.save()?;
        outcome?;
        // The last round's network and check are the chunk's.
        if let Some(n) = st.last_net.clone() {
            self.rec.insert("net".into(), n);
        }
        let verdict = if st.flat { "PASS" } else { "FAIL" };
        if let Some((hc, json_path, txt_path)) = &st.last_check {
            let (g, k) = (self.p.golden_trick, self.p.cut);
            let (j, t) = (
                self.out.join(format!("hc-t{g}-o{k}.json")),
                self.out.join(format!("hc-t{g}-o{k}.txt")),
            );
            let _ = fs::copy(json_path, &j);
            let _ = fs::copy(txt_path, &t);
            self.rec.insert(
                "check".into(),
                json!({ "verdict": verdict, "json": j.file_name().map(|n| n.to_string_lossy().into_owned()),
                        "text": t.file_name().map(|n| n.to_string_lossy().into_owned()),
                        "values": hc["values"], "net_sha256": hc["net"]["sha256"],
                        "overall": hc["groups"].get(0).cloned().unwrap_or(Value::Null),
                        "slices": hc["slices"].clone(), "audit_stop": stop, "time": hc["time"] }),
            );
        }
        self.rec.insert("seconds".into(), json!(secs(self.t0)));
        self.save()?;
        let mut done = self.progress("done", json!({ "verdict": verdict }));
        done["audit_stop"] = json!(stop);
        self.ctx.set_phase("chunk", done);
        Ok(json!({
            "kind": "hybrid_rung",
            "verdict": verdict,
            "device": self.device,
            "gpu": self.rec.get("device").and_then(|d| d.get("gpu_name")).cloned(),
            "net_int_sha256": self.rec.get("net").and_then(|n| n.get("int_sha256")).cloned(),
            "net_int_key": self.rec.get("net").and_then(|n| n.get("int_key")).cloned(),
            "audit": { "stop": stop, "rounds": st.rounds.len(), "flat": st.flat,
                       "summary_key": summary_key },
            "steps": self.steps,
            "warnings": self.warnings,
        }))
    }

    /// The rounds; returns the stop reason.
    fn audit_rounds(&mut self, st: &mut Loop) -> Result<String, String> {
        let rmax = st.s.rounds;
        let mut prev: Option<Round> = None;
        for r in 1..=rmax {
            self.audit_hb
                .insert("round".into(), json!(format!("{r} of {rmax}")));
            let (net_dir, key) = if r == 1 {
                (self.out.join("net"), "net".to_string())
            } else {
                (
                    st.dir.join(format!("round-{r}")).join("net"),
                    format!("net_r{r}"),
                )
            };
            let step = format!("train (round {r})");
            // The targeted shard of the previous round joins this training.
            let pending = prev.as_ref().is_some_and(|p| !p.confirmed.is_empty());
            let bg = self.bg_train.is_some() || (self.device == "cuda" && self.p.overlap);
            let shard;
            if bg {
                let watch = st.wdir.join(format!("train-r{r}"));
                if self.bg_train.is_none() {
                    fs::create_dir_all(&watch).map_err(|e| e.to_string())?;
                    let prog = self.work.join(format!("{key}.progress.json"));
                    let _ = fs::remove_file(&prog);
                    let threads = self.train_threads();
                    let mut files = st.train.clone();
                    files.push(st.val.clone());
                    let mut c = self.train_cmd(
                        &super::Feed::Files(files),
                        &net_dir,
                        &self.device.clone(),
                        threads,
                        &prog,
                    );
                    if pending {
                        c.arg("--watch-dir")
                            .arg(&watch)
                            .args(["--expect", &(st.train.len() + 1).to_string()])
                            // Loading the new shard must not take the threads the CPU steps use.
                            .args(["--load-threads", &threads.to_string()]);
                        if self.p.partial_epochs > 0 {
                            c.args(["--partial-epochs", &self.p.partial_epochs.to_string()]);
                        }
                    }
                    self.start_train_with(c, net_dir.clone(), prog, &format!("hybrid/{key}.log"))?;
                }
                if r == 1 && self.p.check {
                    st.golden = Some(self.golden()?);
                }
                if let Some(p) = prev.take() {
                    self.after_round(st, &p, Some(&watch), false)?;
                }
                shard = self.audit_shard(st, r)?;
                self.finish_train_as(&key, &step)?;
            } else {
                if r == 1 && self.p.check {
                    st.golden = Some(self.golden()?);
                }
                if let Some(p) = prev.take() {
                    self.after_round(st, &p, None, false)?;
                }
                shard = self.audit_shard(st, r)?;
                let mut files = st.train.clone();
                files.push(st.val.clone());
                self.train(&files, &net_dir, &self.device.clone(), &step, &key)?;
            }
            let round = self.audit_round(st, r, &net_dir, &key, shard)?;
            if round.confirmed.is_empty() {
                let pass = self.after_round(st, &round, None, true)?;
                st.flat = pass;
                return Ok(if pass {
                    format!("flat after round {r}")
                } else {
                    format!(
                        "no weak group confirmed in round {r}, but the golden check or a mini set \
                         fails: nothing left to target (the next lever is network capacity)"
                    )
                });
            }
            if r == rmax {
                self.after_round(st, &round, None, true)?;
                let left = self
                    .audit_hb
                    .get("groups_remaining")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                return Ok(format!(
                    "round limit ({rmax} rounds; {left} groups remain: the next lever is network \
                     capacity)"
                ));
            }
            prev = Some(round);
        }
        Err("hybrid_rung: the audit loop ran no round".into())
    }

    /// Shares a file content-addressed (`inputs/sha256/<hex>`); its key, or `None` (warned once
    /// per loop) without a remote store or when the upload fails (the file is in the bundle).
    fn share_file(&mut self, st: &mut Loop, path: &Path) -> Option<String> {
        let Some(remote) = self.tools.remote.clone() else {
            if !st.share_warned {
                st.share_warned = true;
                self.warn(
                    "the audit loop's networks, predicates and reports were not shared as inputs: \
                     no remote store (R2, or RUNNER_REMOTE_DIR); they are in this run's bundle"
                        .into(),
                );
            }
            return None;
        };
        let p2 = path.to_path_buf();
        match self.ctx.blocking(move || put_input(&*remote, &p2, None)) {
            Ok((sha, _, _)) => Some(input_key(&sha)),
            Err(e) => {
                self.warn(format!(
                    "{} was not shared as an input: {e}; it is in this run's bundle",
                    path.display()
                ));
                None
            }
        }
    }

    /// Runs one `search` (or other) step in the foreground, beside whatever runs in the
    /// background (its threads already chosen), the heartbeat showing both; `labels_log`: read
    /// the rows done from this log; `worlds_prog`: read a worlds run's progress file.
    fn audit_step(
        &mut self,
        step: &str,
        c: Command,
        log: &str,
        stdout: Option<&Path>,
        limit: Option<Duration>,
        worlds_prog: Option<&Path>,
    ) -> Result<(), String> {
        let t = Instant::now();
        let base = self.progress(step, json!({}));
        self.ctx.set_phase("chunk", base.clone());
        let beside = self.beside_progress();
        let log_path = self.ctx.logs.join(log);
        let labels = log.contains("labels");
        self.ctx.run_child_io(c, log, stdout, limit, &mut |ctx| {
            let mut p = base.clone();
            if labels && let Some((done, total)) = label_progress(&log_path) {
                p["rows_done"] = json!(done);
                p["rows_total"] = json!(total);
            }
            if let Some(x) = worlds_prog.and_then(read_json) {
                for k in [
                    "decisions_done",
                    "decisions_total",
                    "worlds_done",
                    "worlds_total",
                ] {
                    p[k] = x.get(k).cloned().unwrap_or(Value::Null);
                }
            }
            merge_beside(&mut p, &beside);
            ctx.progress = p;
        })?;
        self.done(step, t)
    }

    /// The threads a foreground step gets now.
    fn fg_threads(&self) -> u64 {
        self.step_threads()
    }

    /// The network a harvest's search uses in round `r` (`None`: exact values): exact at trick 4
    /// and later; at trick 3 and earlier the latest network (`latest`), else `harvest_net`.
    fn harvest_driver(&self, latest: Option<&Path>) -> Option<PathBuf> {
        if self.p.cut >= 4 {
            return None;
        }
        latest
            .map(Path::to_path_buf)
            .or_else(|| self.inputs.get("harvest_net").cloned())
    }

    /// Step 2: round `r`'s audit shard. Returns its path and facts.
    fn audit_shard(&mut self, st: &mut Loop, r: u64) -> Result<(PathBuf, Value), String> {
        let seed = st.s.audit_seed(r);
        let file = format!("t{}-audit-r{r}-s{seed}.npy", self.p.cut);
        let path = st.wdir.join(&file);
        let latest = st
            .last_net
            .as_ref()
            .and_then(|n| n["int_path"].as_str())
            .map(PathBuf::from);
        let driver = self.harvest_driver(latest.as_deref());
        let threads = self.fg_threads();
        let mut c = Command::new(&self.tools.search);
        c.args(["labels", "--trick", &self.p.cut.to_string()])
            .args([
                "--count",
                &st.s.rows.to_string(),
                "--seed",
                &seed.to_string(),
            ])
            .args(["--source", "harvest", "--out"])
            .arg(&path)
            .arg("--equity")
            .arg(&self.table)
            .args(["--threads", &threads.to_string()])
            .args(["--start-hands", &self.p.start_hands.to_string()])
            .args(self.harvest_args());
        if let Some(n) = &driver {
            c.arg("--harvest-net").arg(n);
        }
        self.audit_step(
            &format!("audit shard (round {r})"),
            c,
            &format!("hybrid/labels-audit-r{r}.log"),
            None,
            None,
            None,
        )?;
        let side = read_json(&PathBuf::from(format!("{}.json", path.display())))
            .ok_or_else(|| format!("hybrid_rung: {} wrote no sidecar", path.display()))?;
        let rdir = st.dir.join(format!("round-{r}"));
        write_json(&rdir.join(format!("{file}.json")), &side)?;
        Ok((
            path,
            json!({ "file": file, "rows": st.s.rows, "seed": seed, "threads": threads,
                    "shard_sha256": side["shard_sha256"],
                    "driver": driver.as_ref().map_or("exact".to_string(), |d| d.display().to_string()),
                    "rows_by_race_class": side["rows_by_race_class"] }),
        ))
    }

    /// The harvest's settings (`harvest_cap_per_decision`, `harvest_positions`), as the rung's
    /// harvest shard takes them.
    fn harvest_args(&self) -> Vec<String> {
        let from = self.p.train.iter().find(|s| s.name == "harvest");
        match from {
            Some(h) => h.args.clone(),
            None => vec![
                "--harvest-cap".into(),
                search::oracle_labels::HARVEST_CAP.to_string(),
                "--harvest-positions".into(),
                search::oracle_labels::HARVEST_SHARES.into(),
            ],
        }
    }

    /// Step 3: the audit of network `r` (in `net_dir`, recorded as `key`) on its audit shard:
    /// the auditor (or the stand-in file), the union of groups, the heartbeat's counts.
    fn audit_round(
        &mut self,
        st: &mut Loop,
        r: u64,
        net_dir: &Path,
        key: &str,
        (shard, shard_facts): (PathBuf, Value),
    ) -> Result<Round, String> {
        let rdir = st.dir.join(format!("round-{r}"));
        let adir = rdir.join("auditor");
        fs::create_dir_all(&adir).map_err(|e| e.to_string())?;
        let net = net_dir.join("int.safetensors");
        let train = self.rec.get(key).cloned().unwrap_or(Value::Null);
        let mut last = train.clone();
        last["int_path"] = json!(net.display().to_string());
        st.last_net = Some(last);
        let step = format!("audit (round {r})");
        let report: Value = match self.inputs.get("audit_predicates").cloned() {
            Some(stub) => {
                let t = Instant::now();
                fs::copy(&stub, adir.join("predicates.json")).map_err(|e| e.to_string())?;
                let n = read_groups(&stub)?.len();
                let rep = json!({ "stub": true, "audit": { "candidates": n, "confirmed": n },
                                  "note": "audit_predicates_input: the stored predicate file stands \
                                           in for the auditor (every round confirms its groups)" });
                write_json(&adir.join("report.json"), &rep)?;
                fs::write(
                    adir.join("report.txt"),
                    format!("stand-in audit: the {n} groups of audit_predicates_input\n"),
                )
                .map_err(|e| e.to_string())?;
                self.done(&step, t)?;
                rep
            }
            None => {
                let mut c = self.python_cmd("fortyone_nn.audit");
                c.arg("--net")
                    .arg(&net)
                    .arg("--shard")
                    .arg(&shard)
                    .arg("--out")
                    .arg(&adir)
                    .args(["--seed", &st.s.audit_seed(r).to_string()]);
                self.audit_step(
                    &step,
                    c,
                    &format!("hybrid/audit-r{r}.log"),
                    None,
                    Some(AUDIT_LIMIT),
                    None,
                )?;
                read_json(&adir.join("report.json")).unwrap_or(Value::Null)
            }
        };
        let preds_path = adir.join("predicates.json");
        if !preds_path.is_file() {
            return Err(format!(
                "hybrid_rung: the auditor wrote no {}",
                preds_path.display()
            ));
        }
        let found_groups = read_groups(&preds_path)?;
        let found = report["audit"]["candidates"]
            .as_u64()
            .or_else(|| report["groups_found"].as_u64())
            .unwrap_or(found_groups.len() as u64);
        // Into the union (one group per set of conditions).
        let mut confirmed = Vec::new();
        for (entry, key) in found_groups {
            let i = match st.groups.iter().position(|g| g.key == key) {
                Some(i) => i,
                None => {
                    let base = entry["name"].as_str().unwrap_or("group").to_string();
                    let mut name = base.clone();
                    let mut k = 2;
                    while st.groups.iter().any(|g| g.name == name) {
                        name = format!("{base} #{k}");
                        k += 1;
                    }
                    st.groups.push(Group {
                        name,
                        key,
                        entry: Value::Null,
                        confirmed: Vec::new(),
                        mini: None,
                    });
                    st.groups.len() - 1
                }
            };
            let g = &mut st.groups[i];
            let mut e = entry;
            e["name"] = json!(g.name);
            g.entry = e;
            g.confirmed.push(r);
            confirmed.push(i);
        }
        let preds = if confirmed.is_empty() {
            None
        } else {
            let p = rdir.join("predicates.json");
            let list: Vec<Value> = confirmed
                .iter()
                .map(|&i| st.groups[i].entry.clone())
                .collect();
            write_json(&p, &Value::Array(list))?;
            Some(p)
        };
        self.audit_hb.insert("groups_found".into(), json!(found));
        self.audit_hb
            .insert("groups_confirmed".into(), json!(confirmed.len()));
        self.audit_hb.insert(
            "worst_group".into(),
            confirmed.first().map_or(json!("none"), |&i| {
                json!(describe_group(&st.groups[i].entry))
            }),
        );
        let net_sha = train["int_sha256"].as_str().unwrap_or_default().to_string();
        let net_key = train["int_key"].as_str().map(str::to_string);
        Ok(Round {
            r,
            net,
            net_sha,
            net_key,
            train,
            shard: shard_facts,
            found,
            auditor: json!({ "candidates": found, "confirmed": report["audit"]["confirmed"],
                             "confirmed_general": confirmed.len(), "stub": report["stub"],
                             "report": format!("audit/round-{r}/auditor/report.txt") }),
            confirmed,
            preds,
        })
    }

    /// Steps 4–5 of round `p.r` and its report: the targeted harvest (unless `last`; into
    /// `watch` when a training waits for it), the golden slices, the mini golden sets. Returns
    /// whether the golden check and every mini set pass.
    fn after_round(
        &mut self,
        st: &mut Loop,
        p: &Round,
        watch: Option<&Path>,
        last: bool,
    ) -> Result<bool, String> {
        let r = p.r;
        let rdir = st.dir.join(format!("round-{r}"));
        fs::create_dir_all(&rdir).map_err(|e| e.to_string())?;
        // 5. Targeted harvest first: a training may be waiting for it.
        let targeted = match (&p.preds, last) {
            (Some(preds), false) => Some(self.targeted_harvest(st, p, preds, watch)?),
            _ => None,
        };
        // 4. Golden slices (the union of every group so far).
        let (decisions, worlds) = st
            .golden
            .clone()
            .ok_or("hybrid_rung: the audit loop has no golden set")?;
        let union = if st.groups.is_empty() {
            None
        } else {
            let u = rdir.join("union.json");
            let list: Vec<Value> = st
                .groups
                .iter()
                .map(|g| {
                    let mut e = g.entry.clone();
                    e["rounds_confirmed"] = json!(g.confirmed);
                    e
                })
                .collect();
            write_json(&u, &Value::Array(list))?;
            Some(u)
        };
        let (g, k) = (self.p.golden_trick, self.p.cut);
        let hc_json = rdir.join(format!("hc-t{g}-o{k}.json"));
        let hc_txt = rdir.join(format!("hc-t{g}-o{k}.txt"));
        let verify = if r == 1 { self.p.verify_pct } else { 0.0 };
        let c = self.check_cmd(
            &p.net,
            &decisions,
            &worlds,
            union.as_deref(),
            &hc_json,
            verify,
        );
        self.audit_step(
            &format!("golden slices (round {r})"),
            c,
            &format!("hybrid/check-r{r}.log"),
            Some(&hc_txt),
            None,
            None,
        )?;
        let hc = read_json(&hc_json)
            .ok_or_else(|| format!("hybrid_rung: hybrid-check wrote no {}", hc_json.display()))?;
        let golden = check_summary(&hc);
        // Mini golden sets: made once for a group whose slice is too small, checked every round.
        let slice_n = |name: &str| -> u64 {
            golden["slices"]
                .as_array()
                .and_then(|a| a.iter().find(|x| x["name"] == name))
                .and_then(|x| x["decisions"].as_u64())
                .unwrap_or(0)
        };
        for i in 0..st.groups.len() {
            if st.groups[i].mini.is_none() && slice_n(&st.groups[i].name) < st.s.mini_decisions {
                let m = self.make_mini(st, i, &p.net, r)?;
                st.groups[i].mini = Some(m);
            }
        }
        let mut minis = Vec::new();
        for i in 0..st.groups.len() {
            let Some(m) = &st.groups[i].mini else {
                continue;
            };
            let (id, mpred, mdec, mworlds, kept, drawn) = (
                m.id.clone(),
                m.pred.clone(),
                m.decisions.clone(),
                m.worlds.clone(),
                m.kept,
                m.drawn,
            );
            let (made, mdir) = (m.made_round, m.dir.clone());
            let rel = mdir
                .strip_prefix(&self.out)
                .map_or_else(|_| mdir.display().to_string(), |x| x.display().to_string());
            let name = st.groups[i].name.clone();
            let Some(mworlds) = mworlds else {
                minis.push(json!({ "group": name, "mini": id, "decisions": 0, "drawn": drawn,
                                   "made_round": made, "dir": rel, "pass": Value::Null,
                                   "note": "no decision of the generator reaches this group at the share: \
                                            unmeasured at the decision level (the overall check covers it)" }));
                continue;
            };
            let mj = rdir.join(format!("mini-{id}.json"));
            let mt = rdir.join(format!("mini-{id}.txt"));
            let c = self.check_cmd(&p.net, &mdec, &mworlds, Some(&mpred), &mj, 0.0);
            self.audit_step(
                &format!("mini golden set {id} (round {r})"),
                c,
                &format!("hybrid/mini-{id}-r{r}.log"),
                Some(&mt),
                None,
                None,
            )?;
            let x = read_json(&mj)
                .ok_or_else(|| format!("hybrid_rung: hybrid-check wrote no {}", mj.display()))?;
            let all = &x["groups"][0]["summary"];
            minis.push(json!({ "group": name, "mini": id, "decisions": kept, "drawn": drawn,
                               "made_round": made, "dir": rel, "mean_loss": all["mean_loss"], "loss_upper95": all["loss_upper95"],
                               "pass": all["loss_pass"].as_bool().unwrap_or(false),
                               "report": format!("audit/round-{r}/mini-{id}.txt") }));
        }
        // What is still out of spec: confirmed this round, a failing slice, a failing mini set.
        let failing_slice = |name: &str| {
            golden["slices"].as_array().is_some_and(|a| {
                a.iter()
                    .any(|x| x["name"] == name && x["pass"].as_bool() == Some(false))
            })
        };
        let failing_mini = |name: &str| {
            minis
                .iter()
                .any(|x| x["group"] == name && x["pass"].as_bool() == Some(false))
        };
        let remaining: Vec<String> = st
            .groups
            .iter()
            .enumerate()
            .filter(|(i, g)| {
                p.confirmed.contains(i) || failing_slice(&g.name) || failing_mini(&g.name)
            })
            .map(|(_, g)| g.name.clone())
            .collect();
        let minis_pass = minis.iter().all(|x| x["pass"].as_bool() != Some(false));
        let pass = golden["pass"].as_bool().unwrap_or(false) && minis_pass;
        // The heartbeat.
        let n_slices = golden["slices"].as_array().map_or(0, Vec::len);
        self.audit_hb
            .insert("groups_remaining".into(), json!(remaining.len()));
        self.audit_hb.insert(
            "slices_failing".into(),
            json!(match golden["worst_slice"].as_object() {
                Some(w) => format!(
                    "{} of {n_slices} (worst: {}, 95% upper {})",
                    golden["slices_failing"],
                    w["name"].as_str().unwrap_or("?"),
                    pct4(w["loss_upper95"].as_f64())
                ),
                None => format!("0 of {n_slices}"),
            }),
        );
        self.audit_hb.insert(
            "last_check".into(),
            json!(format!(
                "round {r}: mean loss {}, 95% upper {} ({})",
                pct4(golden["mean_loss"].as_f64()),
                pct4(golden["loss_upper95"].as_f64()),
                if golden["pass"].as_bool() == Some(true) {
                    "PASS"
                } else {
                    "FAIL"
                }
            )),
        );
        st.last_check = Some((hc, hc_json, hc_txt));
        self.round_report(st, p, &golden, &minis, targeted.as_ref(), &remaining)?;
        Ok(pass)
    }

    /// `search hybrid-check` of `net` on a decision set and its exact worlds, with slices.
    fn check_cmd(
        &self,
        net: &Path,
        decisions: &Path,
        worlds: &Path,
        slices: Option<&Path>,
        json_out: &Path,
        verify: f64,
    ) -> Command {
        let mut c = Command::new(&self.tools.search);
        c.args([
            "hybrid-check",
            "--trick",
            &self.p.golden_trick.to_string(),
            "--cut",
            &self.p.cut.to_string(),
        ])
        .arg("--net")
        .arg(net)
        .arg("--equity")
        .arg(&self.table)
        .arg("--decisions")
        .arg(decisions)
        .arg("--worlds")
        .arg(worlds)
        .args(["--to", &self.p.golden_worlds.to_string()])
        .args(["--seed", &self.p.golden_seed.to_string()])
        .args(["--threads", &self.fg_threads().to_string()])
        .args(["--tt-mb", &self.p.tt_mb.to_string()])
        .args(["--verify-pct", &verify.to_string()]);
        if let Some(s) = slices {
            c.arg("--slices")
                .arg(s)
                .args(["--slice-share", SLICE_SHARE]);
        }
        c.arg("--json").arg(json_out);
        c
    }

    /// A mini golden set for group `i` (module notes, step 4), drawn with network `net`.
    fn make_mini(&mut self, st: &mut Loop, i: usize, net: &Path, r: u64) -> Result<Mini, String> {
        let id = format!("m{}", i + 1);
        let dir = st.dir.join("mini").join(&id);
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let pred = dir.join("group.json");
        write_json(&pred, &json!([st.groups[i].entry.clone()]))?;
        let decisions = dir.join("decisions.jsonl");
        let mut c = Command::new(&self.tools.search);
        c.args(["decisions", "--keep"])
            .arg(&pred)
            .args(["--keep-share", SLICE_SHARE])
            .args(["--keep-worlds", &MINI_WORLDS.to_string()])
            .args(["--trick", &self.p.golden_trick.to_string()])
            .args(["--cut", &self.p.cut.to_string()])
            .args(["--count", &st.s.mini_decisions.to_string()])
            .args(["--seed", &self.p.golden_seed.to_string()])
            .args(["--first-candidate", &self.p.golden_count.to_string()])
            .arg("--net")
            .arg(net)
            .arg("--equity")
            .arg(&self.table)
            .args(["--threads", &self.fg_threads().to_string()])
            .args(["--start-hands", &self.p.start_hands.to_string()])
            .args(["--tt-mb", &self.p.tt_mb.to_string(), "--out"])
            .arg(&decisions);
        self.audit_step(
            &format!("mini golden set {id} decisions (round {r})"),
            c,
            &format!("hybrid/mini-{id}.log"),
            None,
            None,
            None,
        )?;
        let side = read_json(&PathBuf::from(format!("{}.json", decisions.display())))
            .unwrap_or(Value::Null);
        let kept = side["kept"].as_u64().unwrap_or_else(|| {
            fs::read_to_string(&decisions)
                .map(|t| t.lines().filter(|l| !l.trim().is_empty()).count() as u64)
                .unwrap_or(0)
        });
        let drawn = side["candidates_drawn"].as_u64().unwrap_or(0);
        let worlds = if kept == 0 {
            None
        } else {
            let w = dir.join(format!(
                "exact-t{}-n{}.jsonl",
                self.p.golden_trick, self.p.golden_worlds
            ));
            let prog = self.work.join(format!("mini-{id}.progress.json"));
            let _ = fs::remove_file(&prog);
            let mut c = Command::new(&self.tools.search);
            c.args(["worlds", "--decisions"])
                .arg(&decisions)
                .args([
                    "--method",
                    "exact",
                    "--to",
                    &self.p.golden_worlds.to_string(),
                ])
                .args(["--seed", &self.p.golden_seed.to_string(), "--out"])
                .arg(&w)
                .arg("--equity")
                .arg(&self.table)
                .args(["--threads", &self.fg_threads().to_string()])
                .args(["--tt-mb", &self.p.tt_mb.to_string(), "--progress"])
                .arg(&prog);
            self.audit_step(
                &format!("mini golden set {id} worlds (round {r})"),
                c,
                &format!("hybrid/mini-{id}.log"),
                None,
                None,
                Some(&prog),
            )?;
            Some(w)
        };
        Ok(Mini {
            id,
            dir,
            pred,
            decisions,
            worlds,
            kept,
            drawn,
            made_round: r,
        })
    }

    /// Step 5: round `p.r`'s targeted harvest of its confirmed groups (`preds`), into `watch`
    /// (a training waits for it) or the loop's work directory; the shard joins the training set.
    fn targeted_harvest(
        &mut self,
        st: &mut Loop,
        p: &Round,
        preds: &Path,
        watch: Option<&Path>,
    ) -> Result<Value, String> {
        let r = p.r;
        let seed = st.s.target_seed(r);
        let rows = if st.s.target_rows > 0 {
            st.s.target_rows
        } else {
            let t = st.train_rows as f64;
            let x: f64 = p
                .confirmed
                .iter()
                .map(|&i| {
                    let w = st.groups[i].entry["weight"]
                        .as_f64()
                        .unwrap_or(0.01)
                        .clamp(0.0, 1.0);
                    if w < 1.0 / 3.0 {
                        (2.0 * w * t / (1.0 - 3.0 * w)).ceil()
                    } else {
                        t
                    }
                })
                .sum();
            (x as u64).clamp(1_000, st.train_rows.max(1_000))
        };
        let file = format!("t{}-target-r{r}-s{seed}.npy", self.p.cut);
        let dir = match watch {
            Some(w) => w.to_path_buf(),
            None => st.wdir.join("targets"),
        };
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join(&file);
        // The weight goes in place before the shard is renamed into a watched directory.
        if st.s.weight != 1.0 {
            fs::write(
                PathBuf::from(format!("{}.weight", path.display())),
                format!("{}\n", st.s.weight),
            )
            .map_err(|e| e.to_string())?;
        }
        let driver = self.harvest_driver(Some(&p.net));
        let shares = self
            .harvest_args()
            .windows(2)
            .find(|w| w[0] == "--harvest-positions")
            .map_or(search::oracle_labels::HARVEST_SHARES.to_string(), |w| {
                w[1].clone()
            });
        let mut c = Command::new(&self.tools.search);
        c.args(["labels", "--trick", &self.p.cut.to_string()])
            .args(["--count", &rows.to_string(), "--seed", &seed.to_string()])
            .args(["--source", "harvest", "--out"])
            .arg(&path)
            .arg("--equity")
            .arg(&self.table)
            .args(["--threads", &self.fg_threads().to_string()])
            .args(["--start-hands", &self.p.start_hands.to_string()])
            .args(["--harvest-cap", "0", "--harvest-positions", &shares])
            .arg("--harvest-keep")
            .arg(preds)
            .args(["--keep-cap", &st.s.keep_cap.to_string()]);
        if let Some(n) = &driver {
            c.arg("--harvest-net").arg(n);
        }
        self.audit_step(
            &format!("targeted harvest (round {r})"),
            c,
            &format!("hybrid/labels-target-r{r}.log"),
            None,
            None,
            None,
        )?;
        if watch.is_some() {
            // The training's time limit counts from its last shard.
            self.ctx.bg_limit_from_now("train");
        }
        let side = read_json(&PathBuf::from(format!("{}.json", path.display())))
            .ok_or_else(|| format!("hybrid_rung: {} wrote no sidecar", path.display()))?;
        write_json(
            &st.dir
                .join(format!("round-{r}"))
                .join(format!("{file}.json")),
            &side,
        )?;
        st.train.push(path);
        st.train_rows += rows;
        Ok(
            json!({ "file": file, "rows": rows, "seed": seed, "weight": st.s.weight,
                   "keep_cap": st.s.keep_cap, "shard_sha256": side["shard_sha256"],
                   "groups": side["harvest"]["keep"]["groups"].clone(),
                   "rows_in_any_group": side["harvest"]["keep"]["rows_in_any_group"].clone(),
                   "driver": driver.as_ref().map_or("exact".to_string(), |d| d.display().to_string()),
                   "joined": if watch.is_some() { "the next round's training, while it runs (watched directory)" } else { "the next round's training" } }),
        )
    }

    /// Writes round `p.r`'s report (`round-<r>.json` and `.txt`), shares it and the round's
    /// predicate files, and adds its summary line.
    fn round_report(
        &mut self,
        st: &mut Loop,
        p: &Round,
        golden: &Value,
        minis: &[Value],
        targeted: Option<&Value>,
        remaining: &[String],
    ) -> Result<(), String> {
        let r = p.r;
        let prev = st.rounds.last().cloned();
        let groups: Vec<Value> = p
            .confirmed
            .iter()
            .map(|&i| st.groups[i].entry.clone())
            .collect();
        let int_err = &p.train["int"]["error"];
        let mut keys = Map::new();
        if let Some(pp) = &p.preds {
            keys.insert("predicates".into(), json!(self.share_file(st, pp)));
        }
        let union = st.dir.join(format!("round-{r}")).join("union.json");
        if union.is_file() {
            keys.insert("union".into(), json!(self.share_file(st, &union)));
        }
        let rec = json!({
            "round": r,
            "net": { "sha256": p.net_sha, "key": p.net_key, "path": p.net.display().to_string() },
            "train": { "epochs_run": p.train["epochs_run"], "best_epoch": p.train["best_epoch"],
                       "int_error": int_err, "minutes": p.train["minutes"],
                       "shards": st.train.len(), "rows": st.train_rows },
            "audit_shard": p.shard,
            "auditor": p.auditor,
            "groups_found": p.found,
            "groups_confirmed": groups,
            "golden": golden,
            "mini_sets": minis,
            "targeted": targeted,
            "groups_remaining": remaining,
            "shared": keys,
        });
        let json_path = st.dir.join(format!("round-{r}.json"));
        write_json(&json_path, &rec)?;
        // Plain English.
        let mut t = format!("Audit loop, round {r} of {}\n\n", st.s.rounds);
        t += &format!(
            "Network: {} (int SD {} on the validation shard), trained {} epochs on {} shards ({} \
             rows).\n",
            p.net_sha.get(..12).unwrap_or("?"),
            pct4(int_err["sd"].as_f64()),
            p.train["epochs_run"],
            st.train.len(),
            st.train_rows
        );
        t += &format!(
            "Audit shard: {} positions as the hybrid reaches them (seed {}), labelled exactly.\n",
            p.shard["rows"], p.shard["seed"]
        );
        if groups.is_empty() {
            t += &format!(
                "Found: {} candidate groups; none confirmed on the held-out half.\n",
                p.found
            );
        } else {
            t += &format!(
                "Found: {} candidate groups; {} confirmed on the held-out half:\n",
                p.found,
                groups.len()
            );
            for g in &groups {
                t += &format!("  - {}\n", describe_group(g));
            }
        }
        t += &format!(
            "Golden set: mean decision loss {}, 95% upper bound {} ({}).\n",
            pct4(golden["mean_loss"].as_f64()),
            pct4(golden["loss_upper95"].as_f64()),
            if golden["pass"].as_bool() == Some(true) {
                "passes"
            } else {
                "fails"
            }
        );
        if let Some(sl) = golden["slices"].as_array().filter(|a| !a.is_empty()) {
            t += "Golden slices (a decision is in a group when half its reached positions are):\n";
            for x in sl {
                t += &format!(
                    "  - {}: {} decisions, 95% upper {}{}\n",
                    x["name"].as_str().unwrap_or("?"),
                    x["decisions"],
                    pct4(x["loss_upper95"].as_f64()),
                    match x["pass"].as_bool() {
                        Some(true) => " (passes)",
                        Some(false) => " (FAILS)",
                        None => " (no decision: see its mini set)",
                    }
                );
            }
        }
        for m in minis {
            t += &format!(
                "Mini golden set for {}: {} decisions, 95% upper {}{}\n",
                m["group"].as_str().unwrap_or("?"),
                m["decisions"],
                pct4(m["loss_upper95"].as_f64()),
                match m["pass"].as_bool() {
                    Some(true) => " (passes)",
                    Some(false) => " (FAILS)",
                    None => " (unmeasured: no decision reaches the group at the share)",
                }
            );
        }
        match targeted {
            Some(x) => {
                t += &format!(
                    "Harvested for the next round: {} positions inside the confirmed groups (seed {}, \
                     loss weight {}).\n",
                    x["rows"], x["seed"], x["weight"]
                )
            }
            None => t += "Nothing harvested (no next round).\n",
        }
        if let Some(pr) = &prev {
            let before = pr["golden"]["loss_upper95"].as_f64();
            t += &format!(
                "Since round {}: golden 95% upper {} -> {}; int SD {} -> {}; groups remaining {} \
                 -> {}.\n",
                pr["round"],
                pct4(before),
                pct4(golden["loss_upper95"].as_f64()),
                pct4(pr["train"]["int_error"]["sd"].as_f64()),
                pct4(int_err["sd"].as_f64()),
                pr["groups_remaining"].as_array().map_or(0, Vec::len),
                remaining.len()
            );
        }
        t += &format!(
            "Still out of spec: {}.\n",
            if remaining.is_empty() {
                "nothing".to_string()
            } else {
                remaining.join("; ")
            }
        );
        let txt_path = st.dir.join(format!("round-{r}.txt"));
        fs::write(&txt_path, &t).map_err(|e| format!("write {}: {e}", txt_path.display()))?;
        let jk = self.share_file(st, &json_path);
        let tk = self.share_file(st, &txt_path);
        let mut rec = rec;
        rec["shared"]["report_json"] = json!(jk);
        rec["shared"]["report_txt"] = json!(tk);
        write_json(&json_path, &rec)?;
        st.lines.push(format!(
            "round {r}: {} groups confirmed ({} found) -> {} remaining; golden 95% upper {} \
             ({}); worst slice {}; network {}",
            groups.len(),
            p.found,
            remaining.len(),
            pct4(golden["loss_upper95"].as_f64()),
            if golden["pass"].as_bool() == Some(true) {
                "PASS"
            } else {
                "FAIL"
            },
            golden["worst_slice"]
                .as_object()
                .map_or("none failing".to_string(), |w| format!(
                    "{} {}",
                    w["name"].as_str().unwrap_or("?"),
                    pct4(w["loss_upper95"].as_f64())
                )),
            p.net_sha.get(..12).unwrap_or("?")
        ));
        st.rounds.push(rec);
        self.rec.insert(
            "audit".into(),
            json!({ "settings": st.s.to_json(), "rounds": st.rounds }),
        );
        self.save()?;
        Ok(())
    }
}
