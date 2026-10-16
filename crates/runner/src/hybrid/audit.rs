//! The audit loop of a `hybrid_rung` chunk (docs/m3-hybrid/audit-loop.md §4 and §6 B3): one
//! rental trains, audits, checks and retrains, round after round, until the network is "flat" or
//! a limit is reached; nobody restarts it between rounds.
//!
//! **Settings** (all refused unless `audit_rounds` is above 0, and the loop needs the golden set:
//! not with `"check": false`):
//! - `audit_rounds` (0: no loop, the chunk is exactly as before): R_max, at most 49;
//! - `audit_rows` (1,000,000; 200,000 at trick 2): rows of each round's audit shard;
//! - `audit_seed` (301): round `r`'s audit shard has the seed `audit_seed × 100 + r`, its targeted
//!   harvest `audit_seed × 100 + 50 + r` (a pure function of the manifest; refused when one meets
//!   a training or validation seed, or the golden sets' seed 1);
//! - `audit_keep_cap` (256): in-group positions kept per decision by the targeted harvest
//!   (`search labels --harvest-keep … --keep-cap`; it keeps in-group positions only, so every row
//!   of a targeted shard is in a weak group);
//! - `audit_weight` (2): the loss weight of every targeted shard's rows (the trainer's per-shard
//!   weight, `NAME.npy.weight`; 1 writes no weight file);
//! - `audit_mini_decisions` (50): a group whose golden slice holds fewer decisions is judged on a
//!   mini golden set of this many decisions instead;
//! - `audit_mini_sets` (4): at most this many new mini golden sets a round, the groups with the
//!   largest weight × SD² (their expected share of the loss) first: a set at trick 3 costs about
//!   15–30 minutes of a 192-thread machine, and an auditor confirms tens of overlapping groups;
//!   the others wait for a later round (reported as not yet measured);
//! - `audit_shapes` (`"<widths>;2048,1024,256;2048,2048,1024,256;4096,2048,1024,256"`, the
//!   shapes over `nn::MAX_WIDTH` left out of the default until the limit is raised; given, each
//!   is checked against it) and `audit_min_gain` (0.15): the capacity ladder (plan §4, "Capacity
//!   is part of the loop"). After each round's audit the SD of the previous round's worst
//!   confirmed group (matched by its conditions, else this round's worst) is compared with what
//!   it was; when it fell by less than `audit_min_gain`, the next round retrains from scratch on
//!   all the shards so far with the next shape; at the top of the ladder the loop ends
//!   ("capacity ladder exhausted");
//! - `audit_target_rows` (10,000,000 at trick 4 and later, 4,000,000 at trick 3, 1,000,000 at
//!   trick 2): rows of each round's targeted harvest (the confirmed groups can cover most
//!   positions, so the targeted shard's size is a setting, not "every reached position");
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
//!    same worlds). **Mini golden sets** (`search decisions --keep` with network `r`, first
//!    candidate `golden_count`, then `search worlds --method exact`; `out/hybrid/audit/mini/m<k>/`)
//!    for the groups whose slice holds fewer than `audit_mini_decisions` decisions, made once
//!    (`audit_mini_sets` a round) and checked with every later network. **The decision level**
//!    (the standard; plan §4, decision of 2026-10-16): each group's 95% upper bound of the
//!    decision loss, on its golden slice when that holds `audit_mini_decisions` decisions, else
//!    on its mini golden set; a group over 0.01% is *remaining*; one whose mini set found no
//!    decision is unmeasurable (the overall row covers it); one waiting for its mini set is *not
//!    yet measured*. The auditor's thresholds only choose what to target and slice; each round's
//!    report pairs every group's SD with its decision-level bound (the recalibration table, report
//!    only).
//! 5. **Targeted harvest** (`search labels --source harvest --harvest-keep PRED --harvest-cap 0
//!    --keep-cap audit_keep_cap`, network `r` driving at trick 3 and earlier, exact at trick 4):
//!    the round's confirmed groups and the *failing groups* (every group whose decision-level
//!    bound failed in this or an earlier round and has not passed since: it stays a target until
//!    it passes; `round-<r>/targets.json`), `audit_keep_cap` in-group positions per decision over
//!    their union, `audit_target_rows` rows, weight `audit_weight`; the shard joins the next
//!    round's training. A round that confirms nothing new but has failing groups targets those
//!    ("re-targeting N groups still failing"). Not made when the round is flat.
//!
//! **Stop**: *flat* when the golden set's overall row passes (decision loss bound, every team's
//! bias; the bands are reported only), no group is failing and none is waiting to be measured,
//! confirmed groups or not (`flat after round r`); *nothing to target* when a round confirms
//! nothing, no group is still failing, and the overall row fails (every waiting mini set is made
//! first, so the decision is complete); *capacity ladder exhausted*; *round limit* after
//! `audit_rounds` rounds; *stopped* on the stop file or the launcher's STOP (the running step is
//! stopped, everything so far written and uploaded), the reason naming a dollar cap when the
//! stop does.
//!
//! **Trick 2** (cut 2; docs/M3-HYBRID-SPEC.md §3, owner decision of 2026-10-16: no trick-1
//! golden set is ever built): no golden set and no mini golden sets. The stop is position-level
//! only: *flat* when a round's audit confirms no group on its fresh trick-2 audit shard. Each
//! round, after the targeted harvest (only the round's confirmed groups: no group is ever
//! "failing" at trick 2), the *rough test* runs: `search hybrid-check --trick 1 --cut 2 --saved
//! …` on the saved exact trick-1 worlds (`t1_decisions_input`, `t1_saved_inputs`; the
//! `hybrid_t1check` chunk's command and table plan, its verification sample `t1_verify_*` in
//! round 1 only) with a slice per group of the union; it is reported in every round's report
//! (`rough_test`, each group's slice labelled "thin: N decisions", how few decisions and seats
//! the saved worlds cover), never decides the stop and never makes a group failing. Defaults:
//! `audit_rows` 200,000 (an exact trick-2 label costs seconds), `audit_target_rows` 1,000,000.
//! Round 1 without `harvest_net` makes its audit shard after its training, driven by its own
//! network (an exact-driven trick-2 harvest would solve every reached position); with
//! `upload_shards` the loop's audit and targeted shards are shared too (hours of labels each).
//! The settings: `t1_decisions_input` and `t1_saved_inputs` (`[{"run": LABEL, "input":
//! "inputs/sha256/<hex>"}, …]`, required at cut 2, refused elsewhere), `t1_verify_pct` (3),
//! `t1_verify_rule` (`rules-v1`), `t1_verify_solve_threads` (46), `t1_verify_tt_mb` (8192).
//!
//! **GPU and CPU together** (plan §4, owner 2026-10-16 21:15): on a GPU with `overlap`, round `r`'s
//! training starts at once in the background (`Rung::start_train_with`, one child beside the
//! steps) when round `r − 1` has groups to target (confirmed, or still failing), and the CPU steps
//! that do not need network `r` run beside it on the threads it leaves: in round 1 the golden set
//! (when generated); for round `r − 1` its golden slices, mini golden sets and targeted harvest
//! (written into the directory the trainer watches, `--watch-dir … --expect`, so it joins between
//! epochs; the trainer's time limit counts again from when it lands; when the checks of earlier
//! mini sets already rule flatness out the harvest runs before the new mini sets are made); then
//! round `r`'s audit shard. If round `r − 1` turns out flat (or has nothing to target), the
//! training of round `r` is stopped unused and the loop ends. A round with nothing confirmed and
//! nothing failing is decided before any further training starts. On the CPU (or `overlap` false) the same steps run one after
//! another, the targeted harvest before the training it feeds. Either way every answer is the
//! same: each child gets a fixed thread count and the labels, checks and sets are byte-identical
//! for any.
//!
//! **Reporting**: every heartbeat carries `round` (`r of R`), `step`, `groups_found` (the latest
//! audit's candidates on half A), `groups_confirmed` (its confirmed groups), `groups_remaining`
//! (groups measured over 0.01% at the decision level by the latest checks), `groups_failing` (the
//! failing groups carried across rounds, the targets still owed), `worst_group` (`name:
//! weight …, bias …, SD …`, the auditor's most severe), `shape` and `shape_step` (`k of n`: the
//! network's hidden widths and their step on the ladder), `thresholds` (the latest audit's gamma,
//! alpha_bias, alpha_sd, fdr), `slices_failing` (`k of n`, with the worst) and `last_check` (the
//! latest golden check's overall row, and how many groups are not yet measured); `audit_stop` at
//! the end. Each round writes `out/hybrid/audit/round-<r>.txt` (plain English) and
//! `round-<r>.json`; the loop ends with `out/hybrid/audit/audit-summary.txt` (one line per round,
//! then the stop reason). Each round's network, predicate files and reports are shared
//! content-addressed (`inputs/sha256/<hex>`) and are in the bundle. `rung.json`'s `audit` holds
//! every round's record; its `net` and `check` are the last round's, as the chunk's result.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};

use super::{GOLDEN_SEED, Rung, label_progress, merge_beside, read_json, secs, write_json};
use crate::remote::{input_hex, input_key, put_input};

/// The audit loop's settings beside `audit_rounds`, refused when it is 0.
const AUDIT_KEYS: [&str; 16] = [
    "audit_rows",
    "audit_seed",
    "audit_keep_cap",
    "audit_weight",
    "audit_mini_decisions",
    "audit_mini_sets",
    "audit_shapes",
    "audit_min_gain",
    "audit_target_rows",
    "audit_predicates_input",
    "t1_decisions_input",
    "t1_saved_inputs",
    "t1_verify_pct",
    "t1_verify_rule",
    "t1_verify_solve_threads",
    "t1_verify_tt_mb",
];

/// The trick-2 loop's rough test's settings (`t1_*`): at cut 2 only.
const T1_KEYS: [&str; 6] = [
    "t1_decisions_input",
    "t1_saved_inputs",
    "t1_verify_pct",
    "t1_verify_rule",
    "t1_verify_solve_threads",
    "t1_verify_tt_mb",
];

/// The golden set's settings, refused at cut 2: no trick-1 golden set is built (owner,
/// 2026-10-16), and the trick-2 loop has no mini golden sets.
const NO_GOLDEN_AT_2: [&str; 9] = [
    "golden_trick",
    "golden_count",
    "golden_seed",
    "golden_worlds",
    "golden_decisions_input",
    "golden_worlds_input",
    "verify_pct",
    "audit_mini_decisions",
    "audit_mini_sets",
];

/// The rough test's limit (it never gates the stop: a rough test over its limit is reported as
/// not run, and the loop goes on).
const ROUGH_LIMIT: Duration = Duration::from_secs(2 * 60 * 60);

/// The trick-2 loop's rough test (docs/M3-HYBRID-SPEC.md §3, owner decision of 2026-10-16: no
/// trick-1 golden set): H_2 against the exact trick-1 worlds earlier runs saved (the
/// `hybrid_t1check` chunk's check, `search hybrid-check --saved`), with a slice per group, every
/// round, reported only: it never decides the stop and never makes a group failing.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RoughTest {
    /// `t1_decisions_input`: the SHA-256 of the decisions file the worlds were valued for.
    pub decisions: String,
    /// `t1_saved_inputs`: each saved worlds file (or checkpoint) as (run label, SHA-256).
    pub saved: Vec<(String, String)>,
    /// `t1_verify_pct` (3), `t1_verify_rule` (`rules-v1`), `t1_verify_solve_threads` (46),
    /// `t1_verify_tt_mb` (8192): the verification sample of round 1 (as `hybrid_t1check`'s).
    pub verify_pct: f64,
    pub verify_rule: String,
    pub verify_solve_threads: u64,
    pub verify_tt_mb: u64,
}

impl RoughTest {
    pub fn to_json(&self) -> Value {
        json!({ "decisions_input": input_key(&self.decisions),
                "saved": self.saved.iter().map(|(run, h)| json!({ "run": run, "input": input_key(h) }))
                             .collect::<Vec<_>>(),
                "verify_pct": self.verify_pct, "verify_rule": self.verify_rule,
                "verify_solve_threads": self.verify_solve_threads,
                "verify_tt_mb": self.verify_tt_mb,
                "gates": "never: a rough test, reported only (owner, 2026-10-16)" })
    }

    /// The name under which the saved file of run `run` is fetched (`Rung::inputs`).
    pub fn input_name(run: &str) -> String {
        format!("t1_saved:{run}")
    }
}

/// The rough test's settings (`t1_*`): required by the audit loop at cut 2, refused elsewhere;
/// at cut 2 the golden set's and the mini sets' settings are refused.
fn parse_rough(o: &Map<String, Value>, cut: u64) -> Result<Option<RoughTest>, String> {
    if cut != 2 {
        if let Some(k) = T1_KEYS.iter().find(|k| o.contains_key(**k)) {
            return Err(format!(
                "hybrid_rung: {k} is for the trick-2 loop (cut 2): the saved trick-1 worlds are its \
                 rough test"
            ));
        }
        return Ok(None);
    }
    if let Some(k) = NO_GOLDEN_AT_2.iter().find(|k| o.contains_key(**k)) {
        return Err(format!(
            "hybrid_rung: {k} is set, but at cut 2 there is no golden set and no mini golden set: \
             no trick-1 golden set is built (owner, 2026-10-16); the trick-2 loop's rough test is \
             the saved trick-1 worlds (t1_decisions_input, t1_saved_inputs); remove it"
        ));
    }
    let key = |k: &str, v: &Value| -> Result<String, String> {
        v.as_str().and_then(input_hex).ok_or_else(|| {
            format!(
                "hybrid_rung: {k} {v} must be a stored input's key, \"inputs/sha256/<64 hex \
                 digits>\" (or \"sha256:<64 hex digits>\")"
            )
        })
    };
    let decisions = match o.get("t1_decisions_input") {
        None | Some(Value::Null) => {
            return Err(
                "hybrid_rung: the trick-2 loop (cut 2) needs t1_decisions_input and \
                 t1_saved_inputs: the saved exact trick-1 worlds are its rough test (no trick-1 \
                 golden set is built)"
                    .into(),
            );
        }
        Some(v) => key("t1_decisions_input", v)?,
    };
    let list = o
        .get("t1_saved_inputs")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or(
            "hybrid_rung: t1_saved_inputs must be a list of {\"run\": LABEL, \"input\": \
             \"inputs/sha256/<hex>\"} (the saved trick-1 worlds files)",
        )?;
    let mut saved: Vec<(String, String)> = Vec::new();
    for e in list {
        let obj = e
            .as_object()
            .ok_or("hybrid_rung: each t1_saved_inputs entry is {\"run\": …, \"input\": …}")?;
        if let Some(k) = obj.keys().find(|k| !["run", "input"].contains(&k.as_str())) {
            return Err(format!(
                "hybrid_rung: t1_saved_inputs: unknown key {k:?} (run, input)"
            ));
        }
        let run = obj
            .get("run")
            .and_then(Value::as_str)
            .filter(|r| !r.is_empty() && !r.contains('='))
            .ok_or("hybrid_rung: t1_saved_inputs: each entry needs a run label (no `=`)")?;
        let hex = key(
            "t1_saved_inputs input",
            obj.get("input").unwrap_or(&Value::Null),
        )?;
        if saved.iter().any(|(r, _)| r == run) {
            return Err(format!(
                "hybrid_rung: t1_saved_inputs: run label {run:?} is given twice"
            ));
        }
        saved.push((run.to_string(), hex));
    }
    let verify_pct = match o.get("t1_verify_pct") {
        None | Some(Value::Null) => 3.0,
        Some(v) => v
            .as_f64()
            .filter(|x| (0.0..=100.0).contains(x))
            .ok_or("hybrid_rung: t1_verify_pct must be a number from 0 to 100")?,
    };
    let verify_rule = match o.get("t1_verify_rule") {
        None | Some(Value::Null) => "rules-v1".to_string(),
        Some(v) => v
            .as_str()
            .filter(|r| ["all", "rules-v1", "searched"].contains(r))
            .ok_or("hybrid_rung: t1_verify_rule: all, rules-v1 or searched")?
            .to_string(),
    };
    let n = |k: &str, d: u64| -> Result<u64, String> {
        match o.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(v) => v
                .as_u64()
                .filter(|&x| x >= 1)
                .ok_or_else(|| format!("hybrid_rung: {k} must be a whole number, at least 1")),
        }
    };
    Ok(Some(RoughTest {
        decisions,
        saved,
        verify_pct,
        verify_rule,
        verify_solve_threads: n("t1_verify_solve_threads", 46)?,
        verify_tt_mb: n("t1_verify_tt_mb", 8192)?,
    }))
}

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
    /// `audit_mini_sets`: at most this many new mini golden sets a round.
    pub mini_sets: u64,
    /// `audit_shapes`: the capacity ladder (hidden widths), from the rung's `widths`.
    pub shapes: Vec<Vec<u64>>,
    /// `audit_min_gain`: the fall of the worst group's SD below which the next round steps up.
    pub min_gain: f64,
    pub target_rows: u64,
    /// `audit_predicates_input`: the SHA-256 of a stored predicate file standing in for the
    /// auditor.
    pub predicates_input: Option<String>,
    /// The trick-2 loop's rough test (cut 2 only, where it replaces the golden set's slices and
    /// the mini golden sets, and never gates).
    pub rough: Option<RoughTest>,
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
                "mini_decisions": self.mini_decisions, "mini_sets": self.mini_sets,
                "shapes": self.shapes.iter().map(|w| shape_text(w)).collect::<Vec<_>>(),
                "min_gain": self.min_gain,
                "target_rows": self.target_rows,
                "predicates_input": self.predicates_input.as_deref().map(input_key),
                "rough_test": self.rough.as_ref().map(RoughTest::to_json),
                "audit_seeds": (1..=self.rounds).map(|r| self.audit_seed(r)).collect::<Vec<_>>(),
                "target_seeds": (1..=self.rounds).map(|r| self.target_seed(r)).collect::<Vec<_>>() })
    }
}

/// Reads the audit settings of a `hybrid_rung` chunk (`o`; `check`: the chunk's, `seeds`: every
/// training and validation shard's seed). `None` when `audit_rounds` is absent or 0.
pub(crate) fn parse_settings(
    o: &Map<String, Value>,
    check: bool,
    cut: u64,
    widths: &[u64],
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
    // Rows of each round's audit shard: 1M from trick 3 (seconds to minutes of labels), 200,000
    // at trick 2 (an exact trick-2 label costs seconds: docs/m3-hybrid/vast.md).
    let rows = u("audit_rows", if cut <= 2 { 200_000 } else { 1_000_000 })?;
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
    let mini_sets = u("audit_mini_sets", 4)?;
    let shapes = parse_shapes(o.get("audit_shapes"), widths)?;
    let min_gain = match o.get("audit_min_gain") {
        None | Some(Value::Null) => 0.15,
        Some(v) => v
            .as_f64()
            .filter(|g| g.is_finite() && (0.0..1.0).contains(g))
            .ok_or("hybrid_rung: audit_min_gain must be a number from 0 to below 1")?,
    };
    // Rows of each targeted harvest (the targeted shard's share of the training set): 10M at
    // trick 4 and later, 4M at trick 3, 1M at trick 2.
    let target_rows = u(
        "audit_target_rows",
        match cut {
            0..=2 => 1_000_000,
            3 => 4_000_000,
            _ => 10_000_000,
        },
    )?;
    if target_rows == 0 {
        return Err("hybrid_rung: audit_target_rows must be at least 1".into());
    }
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
    let rough = parse_rough(o, cut)?;
    let s = AuditSettings {
        rounds,
        rows,
        seed,
        keep_cap,
        weight,
        mini_decisions,
        mini_sets,
        shapes,
        min_gain,
        target_rows,
        predicates_input,
        rough,
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

/// The default capacity ladder after the rung's `widths` (plan §4, "Capacity is part of the
/// loop"); a shape wider than the integer inference allows (`nn::MAX_WIDTH`) is left out until the
/// limit is raised.
const LADDER: [&[u64]; 3] = [
    &[2048, 1024, 256],
    &[2048, 2048, 1024, 256],
    &[4096, 2048, 1024, 256],
];

/// Hidden widths as written (`1024,512,128`).
fn shape_text(w: &[u64]) -> String {
    w.iter().map(u64::to_string).collect::<Vec<_>>().join(",")
}

/// The capacity ladder: `audit_shapes` (`"<widths>;2048,1024,256;…"`, lists of hidden widths
/// separated by `;`, starting with the rung's `widths`, written as such or as `<widths>`), each
/// shape checked against `nn::MAX_WIDTH`; by default `widths`, then the [`LADDER`] shapes within
/// the limit.
fn parse_shapes(v: Option<&Value>, widths: &[u64]) -> Result<Vec<Vec<u64>>, String> {
    let fits = |w: &[u64]| {
        let h: Vec<usize> = w.iter().map(|&x| x as usize).collect();
        nn::mlp::check_hidden_widths(&h)
    };
    let text = match v {
        None | Some(Value::Null) => {
            let mut out = vec![widths.to_vec()];
            out.extend(
                LADDER
                    .iter()
                    .map(|w| w.to_vec())
                    .filter(|w| fits(w).is_ok()),
            );
            return Ok(out);
        }
        Some(Value::String(t)) => t.clone(),
        Some(_) => {
            return Err("hybrid_rung: audit_shapes must be a string such as \
                               \"<widths>;2048,1024,256\""
                .into());
        }
    };
    let mut out: Vec<Vec<u64>> = Vec::new();
    for part in text.split(';').map(str::trim) {
        let w: Vec<u64> = if part == "<widths>" {
            widths.to_vec()
        } else {
            part.split(',')
                .map(|x| {
                    x.trim()
                        .parse::<u64>()
                        .ok()
                        .filter(|&x| (1..=65_536).contains(&x))
                })
                .collect::<Option<Vec<_>>>()
                .filter(|w| !w.is_empty())
                .ok_or_else(|| {
                    format!(
                        "hybrid_rung: audit_shapes {text:?}: {part:?} is not a list of hidden \
                         widths like \"2048,1024,256\""
                    )
                })?
        };
        fits(&w).map_err(|e| format!("hybrid_rung: audit_shapes {text:?}: {part:?}: {e}"))?;
        if out.contains(&w) {
            return Err(format!(
                "hybrid_rung: audit_shapes {text:?}: {part:?} is given twice"
            ));
        }
        out.push(w);
    }
    if out.first().map(Vec::as_slice) != Some(widths) {
        return Err(format!(
            "hybrid_rung: audit_shapes {text:?} must start with the rung's widths ({}, or \
             \"<widths>\")",
            shape_text(widths)
        ));
    }
    Ok(out)
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
    /// The round its decision-level bound failed, while it has not passed since: it stays a
    /// target of every targeted harvest until it passes (the failing groups).
    failing_since: Option<u64>,
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
    /// The training set the network was trained on: shards and rows.
    trained_on: (usize, u64),
    /// The auditor's thresholds, as text.
    thresholds: String,
    /// The auditor's report (`report.json`).
    report: Value,
    /// The shape the network was trained with, and its step on the ladder (0-based).
    shape: String,
    shape_step: usize,
    /// The capacity check (plan §4): the worst confirmed group's SD against the round before,
    /// and the shape of the next round when it steps up.
    capacity: Value,
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
    /// The last golden check (its JSON and text paths); at trick 2 the last rough test's.
    last_check: Option<(Value, PathBuf, PathBuf)>,
    /// At trick 2: the last rough test's summary ([`rough_summary`]).
    last_rough: Option<Value>,
    /// The last round's network record.
    last_net: Option<Value>,
    /// The current step on the capacity ladder (`s.shapes`).
    step: usize,
    /// The previous round's worst confirmed group: its conditions (identity), name and SD.
    prev_worst: Option<(String, String, f64)>,
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
        // The standard's overall row (decision loss bound, every team's bias); the bands are
        // reported only (docs/M3-HYBRID-SPEC.md §3 amendment), the slices judged per group.
        "overall_pass": all["loss_pass"].as_bool().unwrap_or(false)
            && all["bias_pass"].as_bool().unwrap_or(true),
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

/// The SD of the group with conditions `key` in an auditor's report: among its confirmed groups
/// (`audit.groups`, the half-B numbers), else `None`.
fn reported_sd(report: &Value, key: &str) -> Option<f64> {
    report["audit"]["groups"].as_array()?.iter().find_map(|g| {
        let one = json!([{ "name": "g", "conditions": g["conditions"] }]);
        let p = nn::predicate::Predicates::parse(one.to_string().as_bytes()).ok()?;
        (p.groups[0].conditions_text() == key)
            .then(|| g["half_b"]["sd"].as_f64().or_else(|| g["sd"].as_f64()))
            .flatten()
    })
}

/// The capacity check of round `p` (plan §4, "Capacity is part of the loop"): the SD now of the
/// previous round's worst confirmed group (matched by its conditions among this round's groups,
/// else this round's largest SD) against what it was. Records it in `p.capacity`; returns whether
/// it fell by less than `audit_min_gain` (the next round steps up the ladder).
fn capacity_check(st: &mut Loop, p: &mut Round) -> bool {
    let worst = p
        .confirmed
        .iter()
        .map(|&i| &st.groups[i])
        .filter_map(|g| {
            g.entry["sd"]
                .as_f64()
                .map(|sd| (g.key.clone(), g.name.clone(), sd))
        })
        .max_by(|a, b| a.2.total_cmp(&b.2));
    let mut no_gain = false;
    p.capacity = json!({ "shape": p.shape, "shape_step": p.shape_step + 1,
                         "shapes": st.s.shapes.len(), "min_gain": st.s.min_gain });
    if let (Some((key, name, before)), Some(now_worst)) = (st.prev_worst.clone(), worst.clone()) {
        let matched = p
            .confirmed
            .iter()
            .map(|&i| &st.groups[i])
            .find(|g| g.key == key)
            .and_then(|g| g.entry["sd"].as_f64())
            .or_else(|| reported_sd(&p.report, &key));
        let (now, basis) = match matched {
            Some(sd) => (sd, "the same group"),
            None => (
                now_worst.2,
                "this round's worst group (the previous one is not confirmed)",
            ),
        };
        let gain = if before > 0.0 {
            (before - now) / before
        } else {
            0.0
        };
        no_gain = gain < st.s.min_gain;
        p.capacity["compared"] = json!({ "group": name, "sd_before": before, "sd_now": now,
                                         "basis": basis, "gain": gain, "no_gain": no_gain });
    }
    if let Some((k, n, sd)) = worst {
        st.prev_worst = Some((k, n, sd));
    }
    no_gain
}

/// What a round's checks decided (`Rung::after_round`).
#[derive(Debug, PartialEq, Eq)]
enum After {
    /// No group failing or waiting to be measured, and the golden set's overall row passes.
    Flat,
    /// Not flat, and no group confirmed now or still failing: nothing to target.
    Nothing,
    /// Not flat; with `target`, the targeted harvest is made.
    Go,
}

/// Every group of the union at the decision level after round `r`'s checks, in the union's
/// order: its golden slice when that holds `big` decisions, else its mini golden set (`minis`:
/// this round's measurements), else not yet measured.
fn judge(st: &Loop, golden: &Value, minis: &[Value], r: u64) -> Vec<Value> {
    let big = st.s.mini_decisions;
    st.groups
        .iter()
        .map(|g| {
            let sl = golden["slices"]
                .as_array()
                .and_then(|a| a.iter().find(|x| x["name"] == g.name.as_str()).cloned());
            let n = sl
                .as_ref()
                .and_then(|x| x["decisions"].as_u64())
                .unwrap_or(0);
            let mini = minis.iter().find(|x| x["group"] == g.name.as_str());
            let (basis, bound, pass) = if n >= big {
                let x = sl.unwrap_or(Value::Null);
                ("golden slice", x["loss_upper95"].clone(), x["pass"].clone())
            } else {
                match mini {
                    Some(m) if m["pass"].is_null() => ("unmeasurable", Value::Null, Value::Null),
                    Some(m) => (
                        "mini golden set",
                        m["loss_upper95"].clone(),
                        m["pass"].clone(),
                    ),
                    None => ("not yet measured", Value::Null, Value::Null),
                }
            };
            json!({ "group": g.name, "weight": g.entry["weight"], "sd": g.entry["sd"],
                    "bias": g.entry["bias"], "golden_decisions": n, "basis": basis,
                    "loss_upper95": bound, "pass": pass,
                    "confirmed_this_round": g.confirmed.last() == Some(&r) })
        })
        .collect()
}

/// The failing groups after round `r`'s measurements (`judged`, in the union's order): a group
/// whose bound fails joins them (or stays), one whose bound passes leaves them, one not measured
/// keeps its state.
fn update_failing(st: &mut Loop, judged: &[Value], r: u64) {
    for (g, x) in st.groups.iter_mut().zip(judged) {
        match x["pass"].as_bool() {
            Some(false) => {
                g.failing_since.get_or_insert(r);
            }
            Some(true) => g.failing_since = None,
            None => {}
        }
    }
}

/// The union of every group met so far, as a predicate file in `rdir` (`union.json`; each with
/// the rounds that confirmed it), or `None` before any.
fn write_union(st: &Loop, rdir: &Path) -> Result<Option<PathBuf>, String> {
    if st.groups.is_empty() {
        return Ok(None);
    }
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
    Ok(Some(u))
}

/// Whole numbers as ranges (`1-2, 5`).
fn ranges(v: &[u64]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1] == v[j] + 1 {
            j += 1;
        }
        out.push(if j == i {
            v[i].to_string()
        } else {
            format!("{}-{}", v[i], v[j])
        });
        i = j + 1;
    }
    out.join(", ")
}

/// The rough test's summary (trick 2; owner decision of 2026-10-16) from `search hybrid-check
/// --saved --slices`' JSON, `file` being the trick-1 decisions in the decisions file: per set (one
/// per burn rule) its overall row, bands and verification; the decisions with saved worlds and
/// their seats; the slices merged over the sets (a group's decisions in every set, its worst
/// bound, passing when every set's slice with decisions does); and the plain-English statement
/// of how thin the evidence is. Reported only: nothing here gates the loop.
fn rough_summary(hc: &Value, file: u64) -> Value {
    use std::collections::BTreeSet;
    let mut sets = Vec::new();
    let mut ids: BTreeSet<u64> = BTreeSet::new();
    let mut seats: BTreeSet<u64> = BTreeSet::new();
    let mut merged: Vec<(String, Vec<Value>)> = Vec::new();
    let mut per_set_text = Vec::new();
    let mut not_compared = Vec::new();
    for s in hc["sets"].as_array().into_iter().flatten() {
        let compared = s["groups"].is_array();
        let all = &s["groups"][0]["summary"];
        let decs = s["decisions"].as_array().cloned().unwrap_or_default();
        for d in &decs {
            ids.extend(d["id"].as_u64());
            seats.extend(d["position"].as_u64());
        }
        let (n, w) = (
            s["plan"]["decisions"].as_u64().unwrap_or(decs.len() as u64),
            s["plan"]["worlds"].as_u64().unwrap_or(0),
        );
        let rule = s["burn_rule"].as_str().unwrap_or("?");
        per_set_text.push(format!("{rule}: {n} decisions, {w} worlds"));
        if !compared {
            not_compared.push(rule.to_string());
        }
        let bands: Vec<Value> = s["groups"]
            .as_array()
            .map(|g| {
                g.iter()
                    .skip(1)
                    .take(3)
                    .map(|b| {
                        json!({ "band": b["group"], "decisions": b["summary"]["decisions"],
                                "loss_upper95": b["summary"]["loss_upper95"] })
                    })
                    .collect()
            })
            .unwrap_or_default();
        let pass = compared
            && all["loss_pass"].as_bool().unwrap_or(false)
            && all["bias_pass"].as_bool().unwrap_or(false);
        sets.push(json!({
            "set": s["set"], "burn_rule": rule, "values_version": s["values_version"],
            "compared": compared, "exact_values": s["exact_values"],
            "verification": s["verification"], "decisions": n, "worlds": w,
            "ids": decs.iter().map(|d| d["id"].clone()).collect::<Vec<_>>(),
            "mean_loss": all["mean_loss"], "loss_upper95": all["loss_upper95"],
            "loss_pass": all["loss_pass"], "bias": all["bias"], "bias_pass": all["bias_pass"],
            "sd": all["sd"], "agreement": all["agreement"], "overall_pass": pass, "bands": bands }));
        for x in s["slices"]["groups"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|x| !x["group"].is_null())
        {
            let name = x["name"].as_str().unwrap_or("?").to_string();
            let e = json!({ "set": s["set"], "burn_rule": rule,
                            "decisions": x["ids"].as_array().map_or(0, Vec::len),
                            "ids": x["ids"], "loss_upper95": x["summary"]["loss_upper95"],
                            "pass": x["pass"], "reached_share": x["reached_share"] });
            match merged.iter_mut().find(|m| m.0 == name) {
                Some(m) => m.1.push(e),
                None => merged.push((name, vec![e])),
            }
        }
    }
    let compared: Vec<&Value> = sets
        .iter()
        .filter(|s| s["compared"].as_bool() == Some(true))
        .collect();
    let worst = compared.iter().copied().max_by(|a, b| {
        a["loss_upper95"]
            .as_f64()
            .unwrap_or(0.0)
            .total_cmp(&b["loss_upper95"].as_f64().unwrap_or(0.0))
    });
    let overall_pass = !compared.is_empty()
        && compared
            .iter()
            .all(|s| s["overall_pass"].as_bool() == Some(true));
    let slices: Vec<Value> = merged
        .into_iter()
        .map(|(name, per)| {
            let with: Vec<&Value> = per
                .iter()
                .filter(|e| e["decisions"].as_u64().unwrap_or(0) > 0)
                .collect();
            let n: u64 = with.iter().filter_map(|e| e["decisions"].as_u64()).sum();
            let pass = if with.is_empty() {
                Value::Null
            } else {
                json!(with.iter().all(|e| e["pass"].as_bool() == Some(true)))
            };
            let bound = with
                .iter()
                .filter_map(|e| e["loss_upper95"].as_f64())
                .fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.max(x))));
            let reached = per
                .iter()
                .filter_map(|e| e["reached_share"].as_f64())
                .fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.max(x))));
            json!({ "name": name, "decisions": n, "label": format!("thin: {n} decisions"),
                    "loss_upper95": bound, "pass": pass, "reached_share": reached,
                    "by_set": per })
        })
        .collect();
    let failing: Vec<&Value> = slices
        .iter()
        .filter(|x| x["pass"].as_bool() == Some(false))
        .collect();
    let worst_slice = failing.iter().max_by(|a, b| {
        a["loss_upper95"]
            .as_f64()
            .unwrap_or(0.0)
            .total_cmp(&b["loss_upper95"].as_f64().unwrap_or(0.0))
    });
    let ids: Vec<u64> = ids.into_iter().collect();
    let seats: Vec<u64> = seats.into_iter().collect();
    let missing: Vec<u64> = (1..=8).filter(|s| !seats.contains(s)).collect();
    let evidence = format!(
        "Rough test at trick 1 (reported only: it never decides the stop and never makes a group \
         failing; owner, 2026-10-16): H_2 against the saved exact trick-1 worlds. {file} decisions \
         in the saved trick-1 decisions file, {} of them with saved exact worlds ({}){}{}. That is \
         far weaker evidence than the 400 decisions x 320 worlds of the golden sets at tricks 3 \
         and 4: a 95% bound over a handful of decisions is coarse. No trick-1 golden set is built \
         and no mini golden sets are made at trick 2; trick 2's standard is the position-level \
         audit.",
        ids.len(),
        per_set_text.join("; "),
        if seats.is_empty() {
            String::new()
        } else {
            format!(
                ", at seats {} only{}",
                ranges(&seats),
                if missing.is_empty() {
                    String::new()
                } else {
                    format!(" (seats {} have no saved decision)", ranges(&missing))
                }
            )
        },
        if not_compared.is_empty() {
            String::new()
        } else {
            format!(
                "; not compared (stored values differ from this solver's in the verification \
                 sample): {}",
                not_compared.join(", ")
            )
        }
    );
    json!({
        "what": "rough test: H_2 against the saved exact trick-1 worlds (reported only)",
        "gates": false,
        "file_decisions": file,
        "decisions_with_worlds": ids.len(),
        "ids": ids,
        "seats": seats,
        "sets": sets,
        "overall_pass": overall_pass,
        "mean_loss": worst.map_or(Value::Null, |w| w["mean_loss"].clone()),
        "loss_upper95": worst.map_or(Value::Null, |w| w["loss_upper95"].clone()),
        "worst_set": worst.map_or(Value::Null, |w| w["burn_rule"].clone()),
        "slices": slices,
        "slices_failing": failing.len(),
        "worst_slice": worst_slice.map(|w| json!({ "name": w["name"], "loss_upper95": w["loss_upper95"] })),
        "evidence": evidence,
    })
}

/// The rough test in one line: each set's bound and size, and how many of the file's decisions
/// it covers.
fn rough_line(rough: &Value) -> String {
    if rough["ran"].as_bool() == Some(false) {
        return "did not run this round".into();
    }
    let sets: Vec<String> = rough["sets"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|s| {
            if s["compared"].as_bool() == Some(true) {
                format!(
                    "{} 95% upper {} ({} decisions, {})",
                    s["burn_rule"].as_str().unwrap_or("?"),
                    pct4(s["loss_upper95"].as_f64()),
                    s["decisions"],
                    if s["overall_pass"].as_bool() == Some(true) {
                        "would pass"
                    } else {
                        "would fail"
                    }
                )
            } else {
                format!(
                    "{} not compared (stored values differ)",
                    s["burn_rule"].as_str().unwrap_or("?")
                )
            }
        })
        .collect();
    format!(
        "{}; {} of {} trick-1 decisions have saved worlds",
        sets.join(", "),
        rough["decisions_with_worlds"],
        rough["file_decisions"]
    )
}

/// A trick-2 round's report lines on the rough test and the position level (`remaining`: the
/// groups this round's audit confirmed).
fn rough_text(rough: &Value, judged: &[Value], remaining: usize) -> String {
    let mut t = format!(
        "{}\n",
        rough["evidence"]
            .as_str()
            .unwrap_or("Rough test at trick 1: no summary.")
    );
    if rough["ran"].as_bool() != Some(false) {
        t += "Rough test by set (report only):\n";
        for s in rough["sets"].as_array().into_iter().flatten() {
            if s["compared"].as_bool() != Some(true) {
                t += &format!(
                    "  - {}: not compared ({})\n",
                    s["burn_rule"].as_str().unwrap_or("?"),
                    s["exact_values"].as_str().unwrap_or("?")
                );
                continue;
            }
            let bias = worst_bias(&s["bias"]);
            t += &format!(
                "  - {} (values version {}): {} decisions, {} worlds; decision loss mean {}, 95% \
                 upper {} ({} the 0.01% line); largest team |bias| {}; agreement {}; exact values: \
                 {}\n",
                s["burn_rule"].as_str().unwrap_or("?"),
                s["values_version"],
                s["decisions"],
                s["worlds"],
                pct4(s["mean_loss"].as_f64()),
                pct4(s["loss_upper95"].as_f64()),
                if s["overall_pass"].as_bool() == Some(true) {
                    "within"
                } else {
                    "over"
                },
                pct4(bias.map(f64::abs)),
                s["agreement"]
                    .as_f64()
                    .map_or("?".into(), |a| format!("{:.0}%", 100.0 * a)),
                s["exact_values"].as_str().unwrap_or("?")
            );
        }
    }
    if !judged.is_empty() {
        let mut lines: Vec<&Value> = judged.iter().collect();
        lines.sort_by_key(|x| u8::from(x["pass"].as_bool() == Some(true)));
        t += &format!(
            "Position level (trick 2's standard), {} groups met so far: {remaining} confirmed on \
             this round's audit shard (out of spec):\n",
            judged.len()
        );
        for x in lines.iter().take(20) {
            let r = &x["rough"];
            t += &format!(
                "  - {}: SD {} on the audit shard; {}; rough test: {}\n",
                x["group"].as_str().unwrap_or("?"),
                x["sd"]
                    .as_f64()
                    .map_or("?".into(), |v| format!("{:.2}%", 100.0 * v)),
                if x["pass"].as_bool() == Some(false) {
                    "confirmed this round (OUT OF SPEC)"
                } else {
                    "not confirmed this round (within)"
                },
                if r.is_object() {
                    format!(
                        "{}, 95% upper {} ({}; report only)",
                        r["label"].as_str().unwrap_or("?"),
                        pct4(r["loss_upper95"].as_f64()),
                        if r["pass"].as_bool() == Some(true) {
                            "within"
                        } else {
                            "over"
                        }
                    )
                } else {
                    "no saved decision reaches it".to_string()
                }
            );
        }
        if lines.len() > 20 {
            t += &format!("  ({} more in the round's JSON)\n", lines.len() - 20);
        }
    }
    t += "Mini golden sets: none at trick 2 (no trick-1 golden set is built: owner, 2026-10-16).\n";
    t
}

/// Every group of the union after round `p.r` at trick 2: judged at the position level only
/// (out of spec when this round's audit confirmed it), with its rough-test slice beside it for
/// the report (`rough`: its decisions, worst bound and verdict over the sets; null when no saved
/// decision reaches it).
fn judge_rough(st: &Loop, rough: &Value, p: &Round) -> Vec<Value> {
    st.groups
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let now = p.confirmed.contains(&i);
            let sl = rough["slices"]
                .as_array()
                .and_then(|a| a.iter().find(|x| x["name"] == g.name.as_str()));
            let n = sl.and_then(|x| x["decisions"].as_u64()).unwrap_or(0);
            let r = sl.filter(|_| n > 0).map_or(Value::Null, |x| {
                json!({ "decisions": n, "label": x["label"], "loss_upper95": x["loss_upper95"],
                        "pass": x["pass"] })
            });
            json!({ "group": g.name, "weight": g.entry["weight"], "sd": g.entry["sd"],
                    "bias": g.entry["bias"],
                    "basis": if now { "confirmed on this round's audit shard (position level)" }
                             else { "not confirmed on this round's audit shard (position level)" },
                    "pass": !now, "confirmed_this_round": now,
                    "golden_decisions": n,
                    "loss_upper95": r["loss_upper95"],
                    "rough": r })
        })
        .collect()
}

/// The failing groups (the carried set).
fn failing(st: &Loop) -> Vec<usize> {
    (0..st.groups.len())
        .filter(|&i| st.groups[i].failing_since.is_some())
        .collect()
}

/// The stop reason when nothing is left to target after round `r`.
fn nothing_reason(r: u64) -> String {
    format!(
        "nothing to target after round {r}: no group confirmed and none still failing, but the \
         golden set's overall row fails (the next lever is network capacity)"
    )
}

/// The stop reason of a flat network after round `r`.
fn flat_reason(st: &Loop, r: u64) -> String {
    let n = st.groups.len();
    if st.s.rough.is_some() {
        return format!(
            "flat after round {r}: no weak group confirmed at the position level on the fresh \
             trick-2 audit shard ({n} groups confirmed in earlier rounds; the saved trick-1 worlds \
             are a rough test, reported only)"
        );
    }
    if n == 0 {
        format!("flat after round {r}")
    } else {
        format!(
            "flat after round {r} ({n} groups confirmed by the auditor so far, every one within \
             0.01% at the decision level)"
        )
    }
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
            last_rough: None,
            last_net: None,
            step: 0,
            prev_worst: None,
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
        let note = match (&st.s.rough, &st.last_rough) {
            (Some(_), Some(rs)) => format!(
                "trick 2: the standard is the position-level audit; the rough test is the only \
                 trick-1 measurement and much weaker evidence than tricks 3-4 (reported only): {}\n",
                rough_line(rs)
            ),
            (Some(_), None) => "trick 2: the standard is the position-level audit; the rough test \
                                did not run\n"
                .to_string(),
            _ => String::new(),
        };
        let summary = st.lines.join("\n")
            + if st.lines.is_empty() { "" } else { "\n" }
            + &note
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
        if let (Some(rs), Some(c)) = (&st.last_rough, self.rec.get_mut("check")) {
            // Trick 2: the verdict is the position level's; the rough test is reported beside.
            c["standard"] = json!(
                "trick 2: flat when the last round's audit confirmed no weak group on its fresh \
                 trick-2 audit shard (position level); the rough test on the saved trick-1 worlds \
                 is reported only (owner, 2026-10-16)"
            );
            c["rough_test"] = rs.clone();
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
                       "summary_key": summary_key,
                       "rough_test": st.last_rough.as_ref().map(rough_line) },
            "steps": self.steps,
            "warnings": self.warnings,
        }))
    }

    /// The rounds; returns the stop reason.
    fn audit_rounds(&mut self, st: &mut Loop) -> Result<String, String> {
        let rmax = st.s.rounds;
        // The round whose checks and targeted harvest run beside the next training.
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
            let shape = shape_text(&st.s.shapes[st.step]);
            self.audit_hb.insert("shape".into(), json!(shape));
            self.audit_hb.insert(
                "shape_step".into(),
                json!(format!("{} of {}", st.step + 1, st.s.shapes.len())),
            );
            self.shape = Some(shape);
            let bg = self.bg_train.is_some() || (self.device == "cuda" && self.p.overlap);
            // The audit shard is made beside (or before) the training, driven by the previous
            // network; at trick 2 with no earlier network (round 1 without `harvest_net`) it is
            // made after the training, driven by this round's own network: an exact-driven
            // trick-2 harvest would solve every reached position (thousands a decision, seconds
            // each).
            let latest = st
                .last_net
                .as_ref()
                .and_then(|n| n["int_path"].as_str())
                .map(PathBuf::from);
            let early = self.p.cut >= 3 || self.harvest_driver(latest.as_deref()).is_some();
            let own = net_dir.join("int.safetensors");
            let rough = st.s.rough.is_some();
            let before;
            if bg {
                // Round r trains at once: round r - 1 has groups to target (confirmed, or still
                // failing), so its targeted shard is coming and the trainer watches for it; if
                // round r - 1 turns out flat (or has nothing to target after all) the training is
                // stopped unused.
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
                    if prev.is_some() {
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
                if r == 1 && !rough {
                    st.golden = Some(self.golden()?);
                }
                if let Some(p) = prev.take() {
                    match self.after_round(st, &p, Some(&watch), true)? {
                        After::Flat => {
                            self.stop_train(&format!("{step}: not needed, round {} is flat", p.r));
                            self.audit_hb
                                .insert("round".into(), json!(format!("{} of {rmax}", p.r)));
                            return Ok(flat_reason(st, p.r));
                        }
                        After::Nothing => {
                            self.stop_train(&format!(
                                "{step}: not needed, nothing to target after round {}",
                                p.r
                            ));
                            self.audit_hb
                                .insert("round".into(), json!(format!("{} of {rmax}", p.r)));
                            return Ok(nothing_reason(p.r));
                        }
                        After::Go => {}
                    }
                }
                before = if early {
                    Some(self.audit_shard(st, r, None)?)
                } else {
                    None
                };
                self.finish_train_as(&key, &step)?;
            } else {
                if r == 1 && !rough {
                    st.golden = Some(self.golden()?);
                }
                if let Some(p) = prev.take() {
                    match self.after_round(st, &p, None, true)? {
                        After::Flat => return Ok(flat_reason(st, p.r)),
                        After::Nothing => return Ok(nothing_reason(p.r)),
                        After::Go => {}
                    }
                }
                before = if early {
                    Some(self.audit_shard(st, r, None)?)
                } else {
                    None
                };
                let mut files = st.train.clone();
                files.push(st.val.clone());
                self.train(&files, &net_dir, &self.device.clone(), &step, &key)?;
            }
            // Trick 2 without an earlier network: the audit shard waits for this round's own.
            let shard = match before {
                Some(s) => s,
                None => self.audit_shard(st, r, Some(&own))?,
            };
            let mut round = self.audit_round(st, r, &net_dir, &key, shard)?;
            let no_gain = capacity_check(st, &mut round);
            if no_gain && r < rmax {
                if st.step + 1 < st.s.shapes.len() {
                    // The next round retrains from scratch, one step up, on all the data so far.
                    st.step += 1;
                    round.capacity["next_shape"] = json!(shape_text(&st.s.shapes[st.step]));
                } else {
                    if self.after_round(st, &round, None, false)? == After::Flat {
                        return Ok(flat_reason(st, r));
                    }
                    return Ok(format!(
                        "capacity ladder exhausted (round {r}: the worst group's SD fell by less \
                         than {:.0}% on the largest shape, {})",
                        100.0 * st.s.min_gain,
                        round.shape
                    ));
                }
            }
            if r == rmax {
                if self.after_round(st, &round, None, false)? == After::Flat {
                    return Ok(flat_reason(st, r));
                }
                if rough {
                    return Ok(format!(
                        "round limit ({rmax} rounds; {} groups still confirmed at the position \
                         level on round {r}'s audit shard: the next lever is network capacity)",
                        round.confirmed.len()
                    ));
                }
                let left = failing(st).len();
                return Ok(format!(
                    "round limit ({rmax} rounds; {left} groups still failing at the decision \
                     level: the next lever is network capacity)"
                ));
            }
            if round.confirmed.is_empty() && failing(st).is_empty() {
                // Nothing confirmed and nothing failing: no targeted shard is expected, so the
                // round is decided before another training starts (a group measured for the
                // first time may still fail, and is then targeted into the next training).
                match self.after_round(st, &round, None, true)? {
                    After::Flat => return Ok(flat_reason(st, r)),
                    After::Nothing => return Ok(nothing_reason(r)),
                    After::Go => {}
                }
            } else {
                prev = Some(round);
            }
        }
        Err("hybrid_rung: the audit loop ran no round".into())
    }

    /// Stops the background training unused (`why` in the log and `rung.json`'s steps).
    /// Its output (a partial network) is removed, so the bundle holds only networks that were
    /// used.
    fn stop_train(&mut self, why: &str) {
        self.ctx.kill_bg();
        if let Some(bg) = self.bg_train.take() {
            let _ = fs::remove_dir_all(&bg.dir);
            let _ = fs::remove_file(&bg.progress);
            // The round's directory, when the network was all it held.
            if let Some(parent) = bg.dir.parent() {
                let _ = fs::remove_dir(parent);
            }
        }
        self.beside.retain(|b| b.1 != "train");
        println!("runner: hybrid_rung: {why}");
        self.steps.push(json!({ "step": why, "seconds": 0.0 }));
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
                    "exact_done",
                    "exact_total",
                ] {
                    if let Some(v) = x.get(k) {
                        p[k] = v.clone();
                    }
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

    /// Step 2: round `r`'s audit shard, driven at trick 3 and earlier by the latest network
    /// before this round's (else `harvest_net`, else `own`: this round's network, at trick 2,
    /// where the shard then waits for it). Returns its path and facts.
    fn audit_shard(
        &mut self,
        st: &mut Loop,
        r: u64,
        own: Option<&Path>,
    ) -> Result<(PathBuf, Value), String> {
        let seed = st.s.audit_seed(r);
        let file = format!("t{}-audit-r{r}-s{seed}.npy", self.p.cut);
        let path = st.wdir.join(&file);
        let latest = st
            .last_net
            .as_ref()
            .and_then(|n| n["int_path"].as_str())
            .map(PathBuf::from);
        let earlier = self.harvest_driver(latest.as_deref());
        let own_driver = earlier.is_none() && own.is_some();
        let driver = earlier.or_else(|| own.map(Path::to_path_buf));
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
        let mut facts = json!({ "file": file, "rows": st.s.rows, "seed": seed, "threads": threads,
                    "shard_sha256": side["shard_sha256"],
                    "driver": driver.as_ref().map_or("exact".to_string(), |d| d.display().to_string()),
                    "rows_by_race_class": side["rows_by_race_class"] });
        if own_driver {
            facts["driver_note"] = json!(
                "no earlier network: the shard was made after this round's training, its search \
                 driven by this round's own network (labels exact either way)"
            );
        }
        if self.p.upload_shards && st.s.rough.is_some() {
            // At trick 2 (only) shared like the training shards (`upload_shards`): a shard is
            // hours of exact labels there.
            facts["key"] = json!(self.share_file(st, &path));
        }
        Ok((path, facts))
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
                        failing_since: None,
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
        // The auditor's thresholds (they choose what to target and slice; the standard is
        // decided at the decision level).
        let th = &report["thresholds"];
        let thresholds = if th.is_object() {
            let f = |k: &str| th[k].as_f64();
            format!(
                "gamma {}, alpha_bias {}, alpha_sd {}, fdr {}",
                f("gamma").map_or("?".into(), |x| format!("{:.2}%", 100.0 * x)),
                f("alpha_bias").map_or("?".into(), |x| format!("{:.2}%", 100.0 * x)),
                f("alpha_sd").map_or("?".into(), |x| format!("{:.2}%", 100.0 * x)),
                f("fdr").map_or("?".into(), |x| format!("{:.0}%", 100.0 * x)),
            )
        } else {
            "none (stand-in predicates)".to_string()
        };
        self.audit_hb.insert("thresholds".into(), json!(thresholds));
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
                             "thresholds": report["thresholds"], "overall": report["overall"],
                             "report": format!("audit/round-{r}/auditor/report.txt") }),
            confirmed,
            preds,
            trained_on: (st.train.len(), st.train_rows),
            thresholds,
            report,
            shape: shape_text(&st.s.shapes[st.step]),
            shape_step: st.step,
            capacity: Value::Null,
        })
    }

    /// Steps 4–5 of round `p.r` and its report: the golden slices, the mini golden sets, and
    /// (with `target`) the targeted harvest, into `watch` when a training waits for it. Decides
    /// at the decision level (plan §4, decision of 2026-10-16): a group is judged by its golden
    /// slice when that holds at least `audit_mini_decisions` decisions, else by its mini golden
    /// set; a group whose bound fails joins the *failing groups* and stays a target of every
    /// targeted harvest until its bound passes; one waiting for its mini set is *not yet
    /// measured* (at most `audit_mini_sets` new sets a round, in order of weight × SD²; every
    /// waiting one when the round confirmed nothing new, so its decision is complete).
    ///
    /// Order: the golden slices, then the mini sets made in earlier rounds (a check each); when
    /// that already rules flatness out (the overall row fails, or a group fails) the targeted
    /// harvest runs next (a training may be waiting for it), before the new mini sets are made;
    /// otherwise the new mini sets come first and a flat network harvests nothing. The targets
    /// are the round's confirmed groups and the failing groups, as known when the harvest runs.
    fn after_round(
        &mut self,
        st: &mut Loop,
        p: &Round,
        watch: Option<&Path>,
        target: bool,
    ) -> Result<After, String> {
        if st.s.rough.is_some() {
            return self.after_round_rough(st, p, watch, target);
        }
        let r = p.r;
        let rdir = st.dir.join(format!("round-{r}"));
        fs::create_dir_all(&rdir).map_err(|e| e.to_string())?;
        // 4. Golden slices (the union of every group so far).
        let (decisions, worlds) = st
            .golden
            .clone()
            .ok_or("hybrid_rung: the audit loop has no golden set")?;
        let union = write_union(st, &rdir)?;
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
        let overall_pass = golden["overall_pass"].as_bool().unwrap_or(false);
        let big = st.s.mini_decisions;
        // The mini golden sets of earlier rounds, re-measured with this network.
        let mut minis = Vec::new();
        for i in 0..st.groups.len() {
            if st.groups[i].mini.is_some() {
                minis.push(self.check_mini(st, i, &p.net, &rdir, r)?);
            }
        }
        let judged = judge(st, &golden, &minis, r);
        update_failing(st, &judged, r);
        let mut targeted = None;
        if target && (!overall_pass || !failing(st).is_empty()) {
            targeted = self.target(st, p, &rdir, watch)?;
        }
        // New mini golden sets: made once for a group whose slice is too small (the largest
        // expected share of the loss first: weight × SD²), checked with every later network.
        let slice_n = |name: &str| {
            golden["slices"]
                .as_array()
                .and_then(|a| a.iter().find(|x| x["name"] == name))
                .and_then(|x| x["decisions"].as_u64())
                .unwrap_or(0)
        };
        let mut want: Vec<usize> = (0..st.groups.len())
            .filter(|&i| st.groups[i].mini.is_none() && slice_n(&st.groups[i].name) < big)
            .collect();
        let severity = |e: &Value| {
            let (w, sd) = (
                e["weight"].as_f64().unwrap_or(0.0),
                e["sd"].as_f64().unwrap_or(0.0),
            );
            w * sd * sd
        };
        want.sort_by(|&a, &b| {
            severity(&st.groups[b].entry)
                .total_cmp(&severity(&st.groups[a].entry))
                .then(a.cmp(&b))
        });
        let cap = if p.confirmed.is_empty() {
            usize::MAX
        } else {
            st.s.mini_sets as usize
        };
        let waiting: Vec<String> = want
            .iter()
            .skip(cap)
            .map(|&i| st.groups[i].name.clone())
            .collect();
        for &i in want.iter().take(cap) {
            let m = self.make_mini(st, i, &p.net, r)?;
            st.groups[i].mini = Some(m);
            minis.push(self.check_mini(st, i, &p.net, &rdir, r)?);
        }
        // Every group at the decision level.
        let judged = judge(st, &golden, &minis, r);
        update_failing(st, &judged, r);
        let remaining = judged
            .iter()
            .filter(|x| x["pass"].as_bool() == Some(false))
            .count();
        let pending = judged
            .iter()
            .filter(|x| x["basis"] == "not yet measured")
            .count();
        let n_failing = failing(st).len();
        let flat = overall_pass && n_failing == 0 && pending == 0;
        let mut verdict = if flat { After::Flat } else { After::Go };
        if target && !flat && targeted.is_none() {
            targeted = self.target(st, p, &rdir, watch)?;
            if targeted.is_none() {
                verdict = After::Nothing;
            }
        }
        // The heartbeat.
        let n_slices = golden["slices"].as_array().map_or(0, Vec::len);
        self.audit_hb
            .insert("groups_remaining".into(), json!(remaining));
        self.audit_hb
            .insert("groups_failing".into(), json!(n_failing));
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
                "round {r}: mean loss {}, 95% upper {} ({}){}",
                pct4(golden["mean_loss"].as_f64()),
                pct4(golden["loss_upper95"].as_f64()),
                if overall_pass { "PASS" } else { "FAIL" },
                if pending > 0 {
                    format!("; {pending} groups not yet measured at the decision level")
                } else {
                    String::new()
                }
            )),
        );
        st.last_check = Some((hc, hc_json, hc_txt));
        if !waiting.is_empty() {
            minis.push(json!({ "waiting": waiting, "pass": Value::Null,
                               "note": format!("no mini golden set yet: at most {} new sets a round \
                                                (audit_mini_sets), the largest weight × SD² first",
                                               st.s.mini_sets) }));
        }
        st.flat = flat;
        self.round_report(st, p, &golden, &minis, &judged, targeted.as_ref())?;
        Ok(verdict)
    }

    /// Steps 4–5 of round `p.r` at trick 2 (cut 2; docs/M3-HYBRID-SPEC.md §3, owner decision of
    /// 2026-10-16: no trick-1 golden set): flat when the round's audit confirmed no group (the
    /// position level is trick 2's standard). Otherwise (with `target`) the targeted harvest of
    /// the round's confirmed groups runs first, into `watch` when a training waits for it. Then
    /// the rough test on the saved exact trick-1 worlds, with a slice per group of the union: it
    /// is reported only, never decides the stop and never makes a group failing. No mini golden
    /// sets.
    fn after_round_rough(
        &mut self,
        st: &mut Loop,
        p: &Round,
        watch: Option<&Path>,
        target: bool,
    ) -> Result<After, String> {
        let r = p.r;
        let rdir = st.dir.join(format!("round-{r}"));
        fs::create_dir_all(&rdir).map_err(|e| e.to_string())?;
        let flat = p.confirmed.is_empty();
        let targeted = if target && !flat {
            self.target(st, p, &rdir, watch)?
        } else {
            None
        };
        let union = write_union(st, &rdir)?;
        let rough = self.rough_test(st, p, union.as_deref(), &rdir)?;
        let judged = judge_rough(st, &rough, p);
        let n_slices = rough["slices"].as_array().map_or(0, Vec::len);
        self.audit_hb
            .insert("groups_remaining".into(), json!(p.confirmed.len()));
        self.audit_hb.insert("groups_failing".into(), json!(0));
        self.audit_hb.insert(
            "slices_failing".into(),
            json!(match rough["worst_slice"].as_object() {
                Some(w) => format!(
                    "{} of {n_slices} (rough test, report only; worst: {}, 95% upper {})",
                    rough["slices_failing"],
                    w["name"].as_str().unwrap_or("?"),
                    pct4(w["loss_upper95"].as_f64())
                ),
                None => format!("0 of {n_slices} (rough test, report only)"),
            }),
        );
        self.audit_hb.insert(
            "last_check".into(),
            json!(format!(
                "round {r}: {} groups confirmed at the position level ({}); rough test (report \
                 only): {}",
                p.confirmed.len(),
                if flat { "flat" } else { "not flat" },
                rough_line(&rough)
            )),
        );
        st.flat = flat;
        self.round_report_rough(st, p, &rough, &judged, targeted.as_ref())?;
        Ok(if flat { After::Flat } else { After::Go })
    }

    /// The rough test of network `p.net` (trick 2): `search hybrid-check --trick 1 --cut 2
    /// --saved …` as the `hybrid_t1check` chunk runs it (`t1check::command`, its tables planned
    /// by `t1check::plan_tables` on the threads the trainer leaves), with `--slices` over the
    /// union of groups; round 1 recomputes the verification sample (`t1_verify_*`), later rounds
    /// reuse those worlds. Writes `round-<r>/t1check.json` and `.txt`; returns its summary
    /// ([`rough_summary`]). A rough test that fails to run is a warning (it never gates the
    /// loop), unless the run is being stopped.
    fn rough_test(
        &mut self,
        st: &mut Loop,
        p: &Round,
        union: Option<&Path>,
        rdir: &Path,
    ) -> Result<Value, String> {
        let r = p.r;
        let rt =
            st.s.rough
                .clone()
                .ok_or("hybrid_rung: no rough test settings")?;
        let settings = json!({
            "net_input": "net", "decisions_input": "t1_decisions",
            "saved": rt.saved.iter().map(|(run, _)| json!({ "input": RoughTest::input_name(run), "run": run }))
                        .collect::<Vec<_>>(),
            "trick": 1, "cut": self.p.cut, "tt_mb": self.p.tt_mb,
            "verify_pct": if r == 1 { rt.verify_pct } else { 0.0 },
            "verify_rule": rt.verify_rule, "verify_solve_threads": rt.verify_solve_threads,
            "verify_tt_mb": rt.verify_tt_mb });
        let q = crate::t1check::parse(&settings)?;
        let mut paths: std::collections::BTreeMap<String, String> = self
            .inputs
            .iter()
            .map(|(k, v)| (k.clone(), v.display().to_string()))
            .collect();
        paths.insert("net".into(), p.net.display().to_string());
        let resolved = crate::t1check::resolve(&settings, &paths)?;
        let threads = self.fg_threads() as usize;
        let plan = crate::t1check::plan_tables(threads, &q, &self.table, "rough test");
        for w in plan.warnings {
            self.warn(format!("round {r}: rough test: {w}"));
        }
        let prog = self.work.join(format!("t1check-r{r}.progress.json"));
        let _ = fs::remove_file(&prog);
        let mut c = crate::t1check::command(
            &self.tools.search,
            &resolved,
            &q,
            &crate::t1check::Run {
                table: &self.table,
                threads,
                tt_mb: plan.tt_mb,
                verify_tt_mb: plan.verify_tt_mb,
                out: rdir,
                progress: &prog,
            },
        )?;
        if let Some(u) = union {
            c.arg("--slices")
                .arg(u)
                .args(["--slice-share", SLICE_SHARE]);
        }
        let (hc_json, hc_txt) = (rdir.join("t1check.json"), rdir.join("t1check.txt"));
        let file = self
            .inputs
            .get("t1_decisions")
            .map_or(Ok(0), |d| super::decisions_at(d, 1))?;
        let ran = self.audit_step(
            &format!("rough test (round {r})"),
            c,
            &format!("hybrid/t1check-r{r}.log"),
            Some(&hc_txt),
            Some(ROUGH_LIMIT),
            Some(&prog),
        );
        let hc = match ran.and_then(|()| {
            read_json(&hc_json)
                .ok_or_else(|| format!("hybrid-check wrote no {}", hc_json.display()))
        }) {
            Ok(hc) => hc,
            Err(e) => {
                if crate::stop::requested().is_some() {
                    return Err(e);
                }
                self.warn(format!(
                    "round {r}: the rough test on the saved trick-1 worlds did not finish ({e}); \
                     it never gates the stop, so the loop goes on"
                ));
                return Ok(json!({ "ran": false, "error": e, "file_decisions": file,
                                  "slices": [], "slices_failing": 0,
                                  "evidence": "Rough test at trick 1: it did not run to the end \
                                               this round (see the warnings); it is reported \
                                               only and never gates the stop." }));
            }
        };
        let mut summary = rough_summary(&hc, file);
        summary["ran"] = json!(true);
        summary["report"] = json!(format!("audit/round-{r}/t1check.txt"));
        st.last_check = Some((hc, hc_json, hc_txt));
        st.last_rough = Some(summary.clone());
        Ok(summary)
    }

    /// The targeted harvest of round `p.r`, when it has targets: the round's confirmed groups,
    /// then the failing groups not among them (`round-<r>/targets.json`). Its record, with the
    /// targets and why; `None` (nothing harvested) without targets.
    fn target(
        &mut self,
        st: &mut Loop,
        p: &Round,
        rdir: &Path,
        watch: Option<&Path>,
    ) -> Result<Option<Value>, String> {
        let carried: Vec<usize> = failing(st)
            .into_iter()
            .filter(|i| !p.confirmed.contains(i))
            .collect();
        if p.confirmed.is_empty() && carried.is_empty() {
            return Ok(None);
        }
        let list: Vec<Value> = p
            .confirmed
            .iter()
            .chain(&carried)
            .map(|&i| st.groups[i].entry.clone())
            .collect();
        let path = rdir.join("targets.json");
        write_json(&path, &Value::Array(list))?;
        let why = match (p.confirmed.len(), carried.len()) {
            (0, n) => format!("re-targeting {n} groups still failing"),
            (c, 0) => format!("{c} groups confirmed this round"),
            (c, n) => format!("{c} groups confirmed this round and {n} groups still failing"),
        };
        println!(
            "runner: hybrid_rung: round {}: targeted harvest: {why}",
            p.r
        );
        let mut t = self.targeted_harvest(st, p, &path, watch)?;
        t["why"] = json!(why);
        t["targets"] = json!({
            "file": format!("audit/round-{}/targets.json", p.r),
            "confirmed": p.confirmed.iter().map(|&i| st.groups[i].name.clone()).collect::<Vec<_>>(),
            "failing": carried.iter().map(|&i| json!({ "group": st.groups[i].name,
                                                        "failing_since": st.groups[i].failing_since }))
                              .collect::<Vec<_>>(),
        });
        Ok(Some(t))
    }

    /// Checks group `i`'s mini golden set with `net` (step `mini golden set m<k> (round r)`); its
    /// record (`pass` null when the set found no decision: unmeasurable).
    fn check_mini(
        &mut self,
        st: &mut Loop,
        i: usize,
        net: &Path,
        rdir: &Path,
        r: u64,
    ) -> Result<Value, String> {
        let m = st.groups[i].mini.as_ref().expect("a mini golden set");
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
            return Ok(
                json!({ "group": name, "mini": id, "decisions": 0, "drawn": drawn,
                              "made_round": made, "dir": rel, "pass": Value::Null,
                              "note": "no decision of the generator reaches this group at the share: \
                                       unmeasurable at the decision level (the overall check covers it)" }),
            );
        };
        let mj = rdir.join(format!("mini-{id}.json"));
        let mt = rdir.join(format!("mini-{id}.txt"));
        let c = self.check_cmd(net, &mdec, &mworlds, Some(&mpred), &mj, 0.0);
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
        Ok(
            json!({ "group": name, "mini": id, "decisions": kept, "drawn": drawn,
                   "made_round": made, "dir": rel, "mean_loss": all["mean_loss"],
                   "loss_upper95": all["loss_upper95"],
                   "pass": all["loss_pass"].as_bool().unwrap_or(false),
                   "report": format!("audit/round-{r}/mini-{id}.txt") }),
        )
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
        let rows = st.s.target_rows;
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
        // At trick 2 (only) shared with `upload_shards`: hours of exact labels there.
        let key = if self.p.upload_shards && st.s.rough.is_some() {
            self.share_file(st, &path)
        } else {
            None
        };
        st.train.push(path);
        st.train_rows += rows;
        Ok(
            json!({ "file": file, "rows": rows, "seed": seed, "weight": st.s.weight, "key": key,
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
        judged: &[Value],
        targeted: Option<&Value>,
    ) -> Result<(), String> {
        let r = p.r;
        let remaining: Vec<String> = judged
            .iter()
            .filter(|x| x["pass"].as_bool() == Some(false))
            .map(|x| x["group"].as_str().unwrap_or("?").to_string())
            .collect();
        let failing_now: Vec<Value> = failing(st)
            .into_iter()
            .map(|i| json!({ "group": st.groups[i].name, "failing_since": st.groups[i].failing_since }))
            .collect();
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
                       "shards": p.trained_on.0, "rows": p.trained_on.1 },
            "audit_shard": p.shard,
            "auditor": p.auditor,
            "thresholds": p.thresholds,
            "shape": p.shape,
            "shape_step": p.shape_step + 1,
            "capacity": p.capacity,
            "groups_found": p.found,
            "groups_confirmed": groups,
            "golden": golden,
            "mini_sets": minis,
            "decision_level": judged,
            "calibration": {
                "what": "each group's SD on the audit shard against its decision-level 95% upper \
                         bound (golden slice or mini golden set): the pairs the auditor's alpha_sd \
                         is recalibrated from (reported only; no automatic change)",
                "pairs": judged.iter().filter(|x| !x["loss_upper95"].is_null()).map(|x| json!({
                    "group": x["group"], "sd": x["sd"], "weight": x["weight"],
                    "loss_upper95": x["loss_upper95"], "basis": x["basis"] })).collect::<Vec<_>>(),
            },
            "targeted": targeted,
            "groups_remaining": remaining,
            "groups_failing": failing_now,
            "flat": st.flat,
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
            p.trained_on.0,
            p.trained_on.1
        );
        t += &format!(
            "Shape: {} (step {} of the capacity ladder of {}).\n",
            p.shape,
            p.shape_step + 1,
            st.s.shapes.len()
        );
        t += &format!(
            "Audit shard: {} positions as the hybrid reaches them (seed {}), labelled exactly.\n",
            p.shard["rows"], p.shard["seed"]
        );
        if let Some(c) = p.capacity["compared"].as_object() {
            t += &format!(
                "Capacity: the worst group ({}) went from SD {} to {} ({:+.0}%; at least {:.0}% \
                 wanted){}.\n",
                c["group"].as_str().unwrap_or("?"),
                c["sd_before"]
                    .as_f64()
                    .map_or("?".into(), |v| format!("{:.2}%", 100.0 * v)),
                c["sd_now"]
                    .as_f64()
                    .map_or("?".into(), |v| format!("{:.2}%", 100.0 * v)),
                -100.0 * c["gain"].as_f64().unwrap_or(0.0),
                100.0 * st.s.min_gain,
                match p.capacity["next_shape"].as_str() {
                    Some(n) => format!(": the next round trains the larger shape {n}"),
                    None if c["no_gain"].as_bool() == Some(true) => {
                        ": no larger shape is left".to_string()
                    }
                    None => String::new(),
                }
            );
        }
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
            for g in groups.iter().take(20) {
                t += &format!("  - {}\n", describe_group(g));
            }
            if groups.len() > 20 {
                t += &format!(
                    "  … and {} more (round-{r}.json lists them all)\n",
                    groups.len() - 20
                );
            }
        }
        t += &format!(
            "Auditor's thresholds (they choose what to target and slice): {}.\n",
            p.thresholds
        );
        t += &format!(
            "Golden set: mean decision loss {}, 95% upper bound {} ({}).\n",
            pct4(golden["mean_loss"].as_f64()),
            pct4(golden["loss_upper95"].as_f64()),
            if golden["overall_pass"].as_bool() == Some(true) {
                "passes"
            } else {
                "fails"
            }
        );
        if !judged.is_empty() {
            let mut lines: Vec<&Value> = judged.iter().collect();
            lines.sort_by_key(|x| match x["pass"].as_bool() {
                Some(false) => 0,
                None => 1,
                Some(true) => 2,
            });
            t += &format!(
                "Decision level (the standard), {} groups: {} over 0.01%, {} not yet measured:\n",
                judged.len(),
                remaining.len(),
                judged
                    .iter()
                    .filter(|x| x["basis"] == "not yet measured")
                    .count()
            );
            for x in lines.iter().take(20) {
                t += &format!(
                    "  - {}: SD {} on the audit shard; {} -> 95% upper {}{}\n",
                    x["group"].as_str().unwrap_or("?"),
                    x["sd"]
                        .as_f64()
                        .map_or("?".into(), |v| format!("{:.2}%", 100.0 * v)),
                    x["basis"].as_str().unwrap_or("?"),
                    pct4(x["loss_upper95"].as_f64()),
                    match x["pass"].as_bool() {
                        Some(true) => " (within)",
                        Some(false) => " (REMAINING)",
                        None => "",
                    }
                );
            }
            if lines.len() > 20 {
                t += &format!("  ({} more in round-{r}.json)\n", lines.len() - 20);
            }
        }
        if let Some(sl) = golden["slices"].as_array().filter(|a| !a.is_empty()) {
            // Failing slices first; at most 20 lines (round-<r>.json has every slice).
            let mut sl: Vec<&Value> = sl.iter().collect();
            sl.sort_by_key(|x| match x["pass"].as_bool() {
                Some(false) => 0,
                None => 1,
                Some(true) => 2,
            });
            t += &format!(
                "Golden slices ({} groups; a decision is in a group when half its reached \
                 positions are):\n",
                sl.len()
            );
            if sl.len() > 20 {
                t += &format!(
                    "  ({} of them not listed here: round-{r}.json has every one)\n",
                    sl.len() - 20
                );
            }
            for x in sl.into_iter().take(20) {
                t += &format!(
                    "  - {}: {} decisions, 95% upper {}{}\n",
                    x["name"].as_str().unwrap_or("?"),
                    x["decisions"],
                    pct4(x["loss_upper95"].as_f64()),
                    match (
                        x["pass"].as_bool(),
                        x["decisions"].as_u64().unwrap_or(0) >= st.s.mini_decisions
                    ) {
                        (_, false) => " (too few decisions: judged on its mini set)",
                        (Some(true), true) => " (passes)",
                        (Some(false), true) => " (FAILS)",
                        (None, true) => "",
                    }
                );
            }
        }
        for m in minis {
            if let Some(w) = m["waiting"].as_array() {
                t += &format!(
                    "No mini golden set yet for {} groups with too few golden decisions (at most {} \
                     new sets a round).\n",
                    w.len(),
                    st.s.mini_sets
                );
                continue;
            }
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
                    "Harvested for the next round: {} positions inside the targeted groups ({}; \
                     seed {}, loss weight {}).\n",
                    x["rows"],
                    x["why"].as_str().unwrap_or("?"),
                    x["seed"],
                    x["weight"]
                )
            }
            None => t += "Nothing harvested (flat, nothing to target, or no next round).\n",
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
        if !failing_now.is_empty() {
            t += &format!(
                "Failing groups (targets until their bound passes): {}.\n",
                failing_now
                    .iter()
                    .map(|x| format!(
                        "{} (since round {})",
                        x["group"].as_str().unwrap_or("?"),
                        x["failing_since"]
                    ))
                    .collect::<Vec<_>>()
                    .join("; ")
            );
        }
        let txt_path = st.dir.join(format!("round-{r}.txt"));
        fs::write(&txt_path, &t).map_err(|e| format!("write {}: {e}", txt_path.display()))?;
        let jk = self.share_file(st, &json_path);
        let tk = self.share_file(st, &txt_path);
        let mut rec = rec;
        rec["shared"]["report_json"] = json!(jk);
        rec["shared"]["report_txt"] = json!(tk);
        write_json(&json_path, &rec)?;
        st.lines.push(format!(
            "round {r}: {} groups confirmed ({} found) -> {} remaining, {} failing; golden 95% \
             upper {} ({}); worst slice {}; shape {}; network {}",
            groups.len(),
            p.found,
            remaining.len(),
            failing_now.len(),
            pct4(golden["loss_upper95"].as_f64()),
            if golden["overall_pass"].as_bool() == Some(true) {
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
            p.shape,
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

    /// Writes round `p.r`'s report at trick 2 (`round-<r>.json` and `.txt`): the position level
    /// (trick 2's standard: the groups this round's audit confirmed are out of spec) and the rough
    /// test beside it, reported only; shares it and the round's predicate files, and adds its
    /// summary line.
    fn round_report_rough(
        &mut self,
        st: &mut Loop,
        p: &Round,
        rough: &Value,
        judged: &[Value],
        targeted: Option<&Value>,
    ) -> Result<(), String> {
        let r = p.r;
        let remaining: Vec<String> = p
            .confirmed
            .iter()
            .map(|&i| st.groups[i].name.clone())
            .collect();
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
                       "shards": p.trained_on.0, "rows": p.trained_on.1 },
            "audit_shard": p.shard,
            "auditor": p.auditor,
            "thresholds": p.thresholds,
            "shape": p.shape,
            "shape_step": p.shape_step + 1,
            "capacity": p.capacity,
            "groups_found": p.found,
            "groups_confirmed": groups,
            "standard": "trick 2: flat when the round's audit confirms no weak group on its fresh \
                         trick-2 audit shard (position level); the rough test on the saved trick-1 \
                         worlds is reported only (owner, 2026-10-16)",
            "position_level": judged,
            "rough_test": rough,
            "mini_sets": "none at trick 2: no trick-1 golden set is built (owner, 2026-10-16)",
            "calibration": {
                "what": "each group's SD on the audit shard against its rough-test bound on the \
                         saved trick-1 worlds (thin: a handful of decisions; reported only, no \
                         automatic change)",
                "pairs": judged.iter().filter(|x| x["rough"].is_object()).map(|x| json!({
                    "group": x["group"], "sd": x["sd"], "weight": x["weight"],
                    "loss_upper95": x["rough"]["loss_upper95"], "basis": x["rough"]["label"] }))
                    .collect::<Vec<_>>(),
            },
            "targeted": targeted,
            "groups_remaining": remaining,
            "groups_failing": [],
            "flat": st.flat,
            "shared": keys,
        });
        let json_path = st.dir.join(format!("round-{r}.json"));
        write_json(&json_path, &rec)?;
        // Plain English.
        let mut t = format!("Audit loop, round {r} of {} (trick 2)\n\n", st.s.rounds);
        t += &format!(
            "Network: {} (int SD {} on the validation shard), trained {} epochs on {} shards ({} \
             rows).\n",
            p.net_sha.get(..12).unwrap_or("?"),
            pct4(int_err["sd"].as_f64()),
            p.train["epochs_run"],
            p.trained_on.0,
            p.trained_on.1
        );
        t += &format!(
            "Shape: {} (step {} of the capacity ladder of {}).\n",
            p.shape,
            p.shape_step + 1,
            st.s.shapes.len()
        );
        t += &format!(
            "Audit shard: {} trick-2 positions as the hybrid reaches them from trick-1 decisions \
             (seed {}), labelled exactly{}.\n",
            p.shard["rows"],
            p.shard["seed"],
            if p.shard["driver_note"].is_string() {
                "; made after this round's training, driven by this round's network (no earlier one)"
            } else {
                ""
            }
        );
        if let Some(c) = p.capacity["compared"].as_object() {
            t += &format!(
                "Capacity: the worst group ({}) went from SD {} to {} ({:+.0}%; at least {:.0}% \
                 wanted){}.\n",
                c["group"].as_str().unwrap_or("?"),
                c["sd_before"]
                    .as_f64()
                    .map_or("?".into(), |v| format!("{:.2}%", 100.0 * v)),
                c["sd_now"]
                    .as_f64()
                    .map_or("?".into(), |v| format!("{:.2}%", 100.0 * v)),
                -100.0 * c["gain"].as_f64().unwrap_or(0.0),
                100.0 * st.s.min_gain,
                match p.capacity["next_shape"].as_str() {
                    Some(n) => format!(": the next round trains the larger shape {n}"),
                    None if c["no_gain"].as_bool() == Some(true) => {
                        ": no larger shape is left".to_string()
                    }
                    None => String::new(),
                }
            );
        }
        if groups.is_empty() {
            t += &format!(
                "Found: {} candidate groups; none confirmed on the held-out half: flat at the \
                 position level.\n",
                p.found
            );
        } else {
            t += &format!(
                "Found: {} candidate groups; {} confirmed on the held-out half:\n",
                p.found,
                groups.len()
            );
            for g in groups.iter().take(20) {
                t += &format!("  - {}\n", describe_group(g));
            }
            if groups.len() > 20 {
                t += &format!(
                    "  … and {} more (round-{r}.json lists them all)\n",
                    groups.len() - 20
                );
            }
        }
        t += &format!(
            "Auditor's thresholds (they decide the stop at trick 2): {}.\n",
            p.thresholds
        );
        t += &rough_text(rough, judged, remaining.len());
        match targeted {
            Some(x) => {
                t += &format!(
                    "Harvested for the next round: {} positions inside the targeted groups ({}; \
                     seed {}, loss weight {}).\n",
                    x["rows"],
                    x["why"].as_str().unwrap_or("?"),
                    x["seed"],
                    x["weight"]
                )
            }
            None => t += "Nothing harvested (flat, or no next round).\n",
        }
        if let Some(pr) = &prev {
            t += &format!(
                "Since round {}: groups confirmed {} -> {}; int SD {} -> {}; rough test (worst set, \
                 report only) 95% upper {} -> {}.\n",
                pr["round"],
                pr["groups_remaining"].as_array().map_or(0, Vec::len),
                remaining.len(),
                pct4(pr["train"]["int_error"]["sd"].as_f64()),
                pct4(int_err["sd"].as_f64()),
                pct4(pr["rough_test"]["loss_upper95"].as_f64()),
                pct4(rough["loss_upper95"].as_f64())
            );
        }
        t += &format!(
            "Still out of spec at the position level: {}.\n",
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
            "round {r}: {} groups confirmed ({} found) at the position level{}; rough test (report \
             only): {}; shape {}; network {}",
            groups.len(),
            p.found,
            if groups.is_empty() { " (flat)" } else { "" },
            rough_line(rough),
            p.shape,
            p.net_sha.get(..12).unwrap_or("?")
        ));
        st.rounds.push(rec);
        self.rec.insert(
            "audit".into(),
            json!({ "settings": st.s.to_json(), "rounds": st.rounds }),
        );
        self.save()
    }
}
