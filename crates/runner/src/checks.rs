//! The `hybrid_checks` chunk kind: several `search hybrid-check` runs of stored networks on one
//! stored golden set, and paired comparisons between them, on one CPU machine with every usable
//! thread; no labels, no training, no GPU (the optimizer's-curse and ensemble test of
//! 2026-10-19, docs/m3-hybrid/curse-ensemble.md; manifest docs/vast/o3-curse-ensemble.manifest.json).
//!
//! ```json
//! { "name": "checks", "priority": 1, "kind": "hybrid_checks", "chunks": 1,
//!   "params": { "threads": "all", "trick": 2, "cut": 3, "worlds": 320, "seed": 1,
//!     "decisions_input": "golden-t2-decisions", "worlds_input": "golden-t2-worlds",
//!     "verify_pct": 1, "tt_mb": 64, "boot": 1000, "boot_seed": 1, "standard": "0.01",
//!     "race_groups": true,
//!     "checks": [
//!       { "name": "curse", "nets": ["o3-f5766a"], "truth": "net",
//!         "noise_sd_by_class": "0.978,0.918,2.010,1.519,4.050", "noise_seed": 1 },
//!       { "name": "single", "nets": ["o3-f5766a"] },
//!       { "name": "ensemble4", "nets": ["o3-f5766a", "o3-a099061", "o3-bfd1bec", "o3-a97ac17"] } ],
//!     "paired": [ { "name": "single-vs-ensemble4", "a": "single", "b": "ensemble4" } ] } }
//! ```
//! The files are format-2 manifest `inputs` (content-addressed, `inputs/sha256/<hex>`, fetched and
//! SHA-256 checked before the chunk starts; the launcher checks they exist before renting):
//! `decisions_input` (the golden decisions at trick `trick`), `worlds_input` (their exact worlds,
//! needed by any check against exact values) and each check's `nets` (integer networks trained
//! for trick `cut`; several make an ensemble, the integer mean of their money chances,
//! `nn::ensemble`). `v2.rs` turns the names into local paths ([`resolve`]).
//!
//! Each check (in the order given, each on every thread) is `search hybrid-check --trick T --cut
//! K --net A,B,… --decisions … --to W`:
//! - `truth` `"exact"` (the default): against the stored exact worlds (`--worlds`), the first
//!   such check recomputing `verify_pct`% of them first (spec §4: a stored golden set is reused
//!   only after a byte-identical recheck; the later checks read the same verified file and
//!   recompute none);
//! - `truth` `"net"`: the optimizer's-curse test (`search::curse`), the network itself as the
//!   truth and the player the same search with seeded zero-mean noise of SD `noise_sd_by_class`
//!   (five percentages, race classes 0-4; required) and seed `noise_seed` (1) at each leaf;
//!   `noise_calibrate` (0: none) above 0 first rescales the SDs on a sample of that many
//!   decisions and their first `noise_calibrate_worlds` (8) worlds until the realised SD per
//!   race class matches the given ones (`search hybrid-check --noise-calibrate`);
//! - `exact_leaves` (optional; not with `truth` `"net"`): the hybrid's frontier positions this
//!   policy picks are solved exactly instead of being valued by the network (`search hybrid-check
//!   --exact-leaves`, `search::exact_leaves`: `race:4`, `race:2,4`, `race:1-4`, `+band:LO-HI`,
//!   `all`; `none` or absent: none), each leaf solve with a table of `tt_mb` (a second table per
//!   thread in the memory plan) (fix G, docs/m3-hybrid/rung2.md "Exact leaves (G)"; manifest
//!   docs/vast/o3-exactleaf.manifest.json). A learned term `learned:NAME:t=T` (or
//!   `:budget=S`) names its leaf-risk model (`nn::leafrisk`) as a manifest input, whose path
//!   replaces the name in the command (rung2.md "Learned exact leaves"; manifest
//!   docs/vast/o3-learnedleaf.manifest.json).
//! - `defer_expert` (optional; with `exact_leaves`): who values the picked leaves (`search
//!   hybrid-check --defer-expert`, `search::exact_leaves::Expert`): `exact` (the default) or a
//!   deeper hybrid `hK:NET[,NET…][:SPEC]` (K = `cut + 1`), its networks and its own exact
//!   leaves' learned model named as manifest inputs (learned deferral at cut 2,
//!   docs/m3-hybrid/learned-leaves.md; manifest docs/vast/t1-defer2-check.manifest.json).
//!
//! A `reference` object `{ "cut": R, "net_input": NAME, "exact_leaves": SPEC, "defer_expert":
//! E }` (optional; R after `cut`) makes the checks against exact values checks against a silver
//! reference (`search hybrid-check --reference silver --reference-net … --reference-cut R
//! [--reference-exact-leaves SPEC] [--reference-defer-expert E]`, docs/m3-hybrid/silver.md):
//! `worlds_input` is then that reference's stored worlds (its records' fingerprint must name the
//! same network and exact leaves), and the verification sample is recomputed with it.
//!
//! with `--race-groups` when `race_groups` (true), into `out/hybrid/checks/<name>.json` and the
//! printed tables in `<name>.txt`. Then each `paired` entry is `search hybrid-paired --a
//! <a>.json --b <b>.json` (two checks of this chunk with the same truth): `paired-<name>.json`
//! and `.txt`. `result.json` gives each check's overall, band, seat-1 and seat-2 rows and noise
//! report, and each comparison's measures. Optional: `seed` (1, the golden set's), `worlds`
//! (320), `tt_mb` (64, lowered if the tables would not fit), `boot` (1000), `boot_seed` (1),
//! `standard` ("0.01"), `table` and `equity_input` (the equity table, as `search_measure`'s).
//! Any other key is refused ([`KEYS`]).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde_json::{Value, json};

use crate::bundle::sha256_file;
use crate::mem::{self, Requested, Tables};
use crate::spike::build_table_from;
use crate::{Ctx, sibling};

/// The settings a `hybrid_checks` chunk takes (after `v2.rs` added the inputs' paths); any other
/// key is refused.
pub(crate) const KEYS: [&str; 20] = [
    "trick",
    "cut",
    "worlds",
    "seed",
    "decisions_input",
    "worlds_input",
    "verify_pct",
    "tt_mb",
    "boot",
    "boot_seed",
    "standard",
    "race_groups",
    "checks",
    "paired",
    "table",
    "equity_input",
    "threads",
    "est_seconds",
    "chunk_sizes",
    "reference",
];

/// Keys `v2.rs` adds: the local paths of the named inputs.
const RESOLVED: [&str; 4] = ["decisions", "worlds_path", "net_paths", "model_paths"];

/// Keys of a check.
const CHECK_KEYS: [&str; 9] = [
    "name",
    "nets",
    "truth",
    "noise_sd_by_class",
    "noise_seed",
    "noise_calibrate",
    "noise_calibrate_worlds",
    "exact_leaves",
    "defer_expert",
];

/// One check.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Check {
    pub name: String,
    /// Input names of its networks (several: an ensemble).
    pub nets: Vec<String>,
    /// Against the network itself (the curse test) rather than exact values.
    pub truth_net: bool,
    pub noise_sd_by_class: Option<String>,
    pub noise_seed: u64,
    /// Decisions of the noise's calibration sample (0: none) and their worlds.
    pub noise_calibrate: u64,
    pub noise_calibrate_worlds: u64,
    /// The exact leaves' policy (`search hybrid-check --exact-leaves`, `search::exact_leaves`),
    /// in its normal form (as written when it has a learned term, whose model is an input
    /// name); `None`: none (the hybrid as before).
    pub exact_leaves: Option<String>,
    /// Input names of the leaf-risk models of a learned term (`learned:NAME:t=T`), the
    /// expert's own included.
    pub models: Vec<String>,
    /// The picked leaves' expert as written (`None`: exact): `hK:NETS[:SPEC]` with input names.
    pub defer_expert: Option<String>,
    /// Input names of the expert's networks.
    pub expert_nets: Vec<String>,
}

/// A silver reference of a `hybrid_checks` chunk (module notes).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RefSpec {
    pub cut: u64,
    pub net_input: String,
    /// Its exact leaves as written (learned models by input name; `None`: none).
    pub exact_leaves: Option<String>,
    /// Their expert as written (`None`: exact).
    pub defer_expert: Option<String>,
    /// Input names of its leaf-risk models and its expert's networks.
    pub models: Vec<String>,
    pub expert_nets: Vec<String>,
}

/// Exact leaves and their expert as a manifest writes them (`what` names the place for
/// messages): checked without their files, `exact_leaves` in its normal form unless it has a
/// learned term (then as written, its model an input name). Returns (the exact leaves, the
/// expert, the input names of the leaf-risk models, those of the expert's networks).
#[allow(clippy::type_complexity)]
pub(crate) fn leaf_settings(
    exact_leaves: Option<&str>,
    defer_expert: Option<&str>,
    what: &str,
) -> Result<(Option<String>, Option<String>, Vec<String>, Vec<String>), String> {
    let err = |e: String| format!("{what}: {e}");
    let mut models = Vec::new();
    let leaves = match exact_leaves {
        None => None,
        Some(text) if text.contains("learned:") => {
            models = search::exact_leaves::check_spec(text).map_err(err)?;
            Some(text.trim().to_string())
        }
        Some(text) => search::exact_leaves::LeafPolicy::parse(text, 1)
            .map_err(err)?
            .map(|p| p.describe()),
    };
    let mut nets = Vec::new();
    let expert = match defer_expert {
        None => None,
        Some(_) if leaves.is_none() => {
            return Err(err(
                "defer_expert defers the leaves exact_leaves picks: none are picked".into(),
            ));
        }
        Some(text) => match search::exact_leaves::expert_parts(text).map_err(err)? {
            search::exact_leaves::ExpertText::Exact => None,
            search::exact_leaves::ExpertText::Specialist { net } => {
                nets = vec![net.to_string()];
                Some(text.trim().to_string())
            }
            search::exact_leaves::ExpertText::Deeper { nets: n, spec, .. } => {
                nets = n.iter().map(|x| x.to_string()).collect();
                if let Some(sp) = spec {
                    for m in search::exact_leaves::check_spec(sp).map_err(err)? {
                        if !models.contains(&m) {
                            models.push(m);
                        }
                    }
                }
                Some(text.trim().to_string())
            }
        },
    };
    Ok((leaves, expert, models, nets))
}

/// Exact leaves as written with each learned model's input name turned into its path (`path`).
pub(crate) fn leaves_with_paths(
    text: &str,
    models: &[String],
    path: &dyn Fn(&str) -> Result<String, String>,
) -> Result<String, String> {
    let mut t = text.to_string();
    for m in models {
        if t.contains(&format!("learned:{m}:")) {
            t = t.replace(&format!("learned:{m}:"), &format!("learned:{}:", path(m)?));
        }
    }
    Ok(t)
}

/// An expert as written (`hK:NETS[:SPEC]`) with its networks' and its leaf-risk models' input
/// names turned into paths (`net_path`, `model_path`).
pub(crate) fn expert_with_paths(
    text: &str,
    net_path: &dyn Fn(&str) -> Result<String, String>,
    model_path: &dyn Fn(&str) -> Result<String, String>,
) -> Result<String, String> {
    match search::exact_leaves::expert_parts(text)? {
        search::exact_leaves::ExpertText::Exact => Ok("exact".into()),
        search::exact_leaves::ExpertText::Specialist { net } => Ok(format!("s:{}", net_path(net)?)),
        search::exact_leaves::ExpertText::Deeper { cut, nets, spec } => {
            let nets: Vec<String> = nets.iter().map(|n| net_path(n)).collect::<Result<_, _>>()?;
            let mut t = format!("h{cut}:{}", nets.join(","));
            if let Some(sp) = spec {
                let models = search::exact_leaves::check_spec(sp)?;
                t.push(':');
                t.push_str(&leaves_with_paths(sp, &models, model_path)?);
            }
            Ok(t)
        }
    }
}

/// The checked settings.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Params {
    pub trick: u64,
    pub cut: u64,
    pub worlds: u64,
    pub seed: u64,
    pub decisions_input: String,
    pub worlds_input: Option<String>,
    pub verify_pct: f64,
    pub tt_mb: u64,
    pub boot: u64,
    pub boot_seed: u64,
    pub standard: String,
    pub race_groups: bool,
    pub checks: Vec<Check>,
    /// (name, check a, check b).
    pub paired: Vec<(String, String, String)>,
    /// A silver reference for the checks against exact values (module notes).
    pub reference: Option<RefSpec>,
}

fn name_ok(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
}

/// Checks the settings (also when the manifest is read, so a typo stops the run before it
/// starts).
pub(crate) fn parse(p: &Value) -> Result<Params, String> {
    let o = p
        .as_object()
        .ok_or("hybrid_checks: params must be an object")?;
    for k in o.keys() {
        if !KEYS.contains(&k.as_str()) && !RESOLVED.contains(&k.as_str()) {
            return Err(format!("hybrid_checks: unknown setting `{k}`"));
        }
    }
    let n = |v: &Value, k: &str, d: u64| -> Result<u64, String> {
        match v.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(x) => x
                .as_u64()
                .ok_or_else(|| format!("hybrid_checks: {k} must be a whole number")),
        }
    };
    let s = |v: &Value, k: &str| -> Result<Option<String>, String> {
        match v.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(x)) => Ok(Some(x.clone())),
            Some(_) => Err(format!("hybrid_checks: {k} must be a string")),
        }
    };
    let trick = n(p, "trick", 2)?;
    let cut = n(p, "cut", trick + 1)?;
    if trick == 0 || trick >= cut || !(2..=5).contains(&cut) {
        return Err(format!(
            "hybrid_checks: cut {cut} must be after trick {trick} (1 ≤ trick < cut ≤ 5)"
        ));
    }
    let worlds = n(p, "worlds", 320)?;
    if worlds == 0 {
        return Err("hybrid_checks: worlds must be at least 1".into());
    }
    let decisions_input =
        s(p, "decisions_input")?.ok_or("hybrid_checks: needs decisions_input (an input name)")?;
    let worlds_input = s(p, "worlds_input")?;
    let verify_pct = match p.get("verify_pct") {
        None | Some(Value::Null) => 1.0,
        Some(v) => v
            .as_f64()
            .filter(|x| (0.0..=100.0).contains(x))
            .ok_or("hybrid_checks: verify_pct must be a number from 0 to 100")?,
    };
    let standard = s(p, "standard")?.unwrap_or_else(|| "0.01".to_string());
    search::hybrid::parse_standard(&standard).map_err(|e| format!("hybrid_checks: {e}"))?;
    let race_groups = match p.get("race_groups") {
        None | Some(Value::Null) => true,
        Some(v) => v
            .as_bool()
            .ok_or("hybrid_checks: race_groups must be true or false")?,
    };
    let list = p
        .get("checks")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or("hybrid_checks: needs checks: a list of {\"name\", \"nets\", …}")?;
    let mut checks: Vec<Check> = Vec::new();
    for c in list {
        for k in c.as_object().into_iter().flat_map(|o| o.keys()) {
            if !CHECK_KEYS.contains(&k.as_str()) {
                return Err(format!("hybrid_checks: a check: unknown key `{k}`"));
            }
        }
        let name = s(c, "name")?.ok_or("hybrid_checks: each check needs a name")?;
        if !name_ok(&name) {
            return Err(format!(
                "hybrid_checks: check name `{name}`: lower-case letters, digits, - and _ only"
            ));
        }
        if checks.iter().any(|x| x.name == name) {
            return Err(format!("hybrid_checks: two checks named `{name}`"));
        }
        let nets: Vec<String> = c
            .get("nets")
            .and_then(Value::as_array)
            .filter(|a| !a.is_empty())
            .ok_or_else(|| format!("hybrid_checks: check `{name}` needs nets: input names"))?
            .iter()
            .map(|x| {
                x.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| format!("hybrid_checks: check `{name}`: nets are input names"))
            })
            .collect::<Result<_, _>>()?;
        if nets.len() > 8 {
            return Err(format!(
                "hybrid_checks: check `{name}`: at most 8 networks in an ensemble"
            ));
        }
        let truth_net = match s(c, "truth")?.as_deref() {
            None | Some("exact") => false,
            Some("net") => true,
            Some(t) => {
                return Err(format!(
                    "hybrid_checks: check `{name}`: truth {t}: exact or net"
                ));
            }
        };
        let noise = s(c, "noise_sd_by_class")?;
        if truth_net {
            let text = noise.as_deref().ok_or_else(|| {
                format!(
                    "hybrid_checks: check `{name}` (truth net) needs noise_sd_by_class: five \
                     percentages, race classes 0-4"
                )
            })?;
            search::curse::NoiseSpec::parse(text, 1)
                .map_err(|e| format!("hybrid_checks: check `{name}`: {e}"))?;
        } else {
            if noise.is_some()
                || ["noise_seed", "noise_calibrate", "noise_calibrate_worlds"]
                    .iter()
                    .any(|k| c.get(*k).is_some())
            {
                return Err(format!(
                    "hybrid_checks: check `{name}`: the noise settings are for truth net"
                ));
            }
            if worlds_input.is_none() {
                return Err(format!(
                    "hybrid_checks: check `{name}` is against exact values: needs worlds_input"
                ));
            }
        }
        let (exact_leaves, defer_expert, models, expert_nets) = leaf_settings(
            s(c, "exact_leaves")?.as_deref(),
            s(c, "defer_expert")?.as_deref(),
            &format!("hybrid_checks: check `{name}`"),
        )?;
        if exact_leaves.is_some() && truth_net {
            return Err(format!(
                "hybrid_checks: check `{name}`: exact_leaves is for checks against exact \
                 values, not truth net"
            ));
        }
        checks.push(Check {
            name,
            nets,
            truth_net,
            noise_sd_by_class: noise,
            noise_seed: n(c, "noise_seed", 1)?,
            noise_calibrate: n(c, "noise_calibrate", 0)?,
            noise_calibrate_worlds: n(c, "noise_calibrate_worlds", 8)?.max(1),
            exact_leaves,
            models,
            defer_expert,
            expert_nets,
        });
    }
    let mut paired = Vec::new();
    for e in p
        .get("paired")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for k in e.as_object().into_iter().flat_map(|o| o.keys()) {
            if !["name", "a", "b"].contains(&k.as_str()) {
                return Err(format!("hybrid_checks: a paired entry: unknown key `{k}`"));
            }
        }
        let get = |k: &str| -> Result<String, String> {
            s(e, k)?.ok_or_else(|| format!("hybrid_checks: a paired entry needs {k}"))
        };
        let (name, a, b) = (get("name")?, get("a")?, get("b")?);
        if !name_ok(&name) || paired.iter().any(|(n, _, _)| *n == name) {
            return Err(format!(
                "hybrid_checks: paired name `{name}`: unique, lower-case letters, digits, - and _"
            ));
        }
        let find = |x: &str| {
            checks
                .iter()
                .find(|c| c.name == x)
                .ok_or_else(|| format!("hybrid_checks: paired `{name}` names no check `{x}`"))
        };
        let (ca, cb) = (find(&a)?, find(&b)?);
        if a == b || ca.truth_net != cb.truth_net {
            return Err(format!(
                "hybrid_checks: paired `{name}`: two different checks with the same truth"
            ));
        }
        paired.push((name, a, b));
    }
    if p.get("paired")
        .is_some_and(|v| !v.is_array() && !v.is_null())
    {
        return Err("hybrid_checks: paired must be a list".into());
    }
    let reference = match p.get("reference") {
        None | Some(Value::Null) => None,
        Some(r) => {
            for k in r
                .as_object()
                .ok_or("hybrid_checks: reference must be an object")?
                .keys()
            {
                if !["cut", "net_input", "exact_leaves", "defer_expert"].contains(&k.as_str()) {
                    return Err(format!("hybrid_checks: reference: unknown key `{k}`"));
                }
            }
            let rc = n(r, "cut", cut + 1)?;
            if rc <= cut || rc > 5 {
                return Err(format!(
                    "hybrid_checks: reference cut {rc} must be after the checks' cut {cut} (at \
                     most 5)"
                ));
            }
            let net_input = s(r, "net_input")?.ok_or("hybrid_checks: reference needs net_input")?;
            let (exact_leaves, defer_expert, models, expert_nets) = leaf_settings(
                s(r, "exact_leaves")?.as_deref(),
                s(r, "defer_expert")?.as_deref(),
                "hybrid_checks: reference",
            )?;
            if worlds_input.is_none() {
                return Err("hybrid_checks: a silver reference needs worlds_input (its \
                            stored worlds)"
                    .into());
            }
            Some(RefSpec {
                cut: rc,
                net_input,
                exact_leaves,
                defer_expert,
                models,
                expert_nets,
            })
        }
    };
    Ok(Params {
        reference,
        trick,
        cut,
        worlds,
        seed: n(p, "seed", 1)?,
        decisions_input,
        worlds_input,
        verify_pct,
        tt_mb: n(p, "tt_mb", 64)?.max(1),
        boot: n(p, "boot", 1000)?,
        boot_seed: n(p, "boot_seed", 1)?,
        standard,
        race_groups,
        checks,
        paired,
    })
}

/// The manifest input names a chunk's settings use (for `v2.rs` to turn into paths).
pub(crate) fn input_names(p: &Value) -> Result<Vec<String>, String> {
    let q = parse(p)?;
    let mut v = vec![q.decisions_input];
    v.extend(q.worlds_input);
    let mut add = |n: &String| {
        if !v.contains(n) {
            v.push(n.clone());
        }
    };
    for c in &q.checks {
        c.nets.iter().for_each(&mut add);
        c.expert_nets.iter().for_each(&mut add);
        c.models.iter().for_each(&mut add);
    }
    if let Some(r) = &q.reference {
        add(&r.net_input);
        r.expert_nets.iter().for_each(&mut add);
        r.models.iter().for_each(&mut add);
    }
    Ok(v)
}

/// Adds the local path of each named input to the settings (`decisions`, `worlds_path`,
/// `net_paths`: input name → path), from the worker's `inputs` (name → path). A name the
/// manifest does not list is an error.
pub(crate) fn resolve(p: &Value, inputs: &BTreeMap<String, String>) -> Result<Value, String> {
    let q = parse(p)?;
    let path = |name: &str| {
        inputs
            .get(name)
            .cloned()
            .ok_or_else(|| format!("hybrid_checks needs input `{name}` in the manifest's inputs"))
    };
    let mut out = p.clone();
    out["decisions"] = json!(path(&q.decisions_input)?);
    if let Some(w) = &q.worlds_input {
        out["worlds_path"] = json!(path(w)?);
    }
    let mut nets = serde_json::Map::new();
    let mut models = serde_json::Map::new();
    for c in &q.checks {
        for n in c.nets.iter().chain(&c.expert_nets) {
            nets.insert(n.clone(), json!(path(n)?));
        }
        for n in &c.models {
            models.insert(n.clone(), json!(path(n)?));
        }
    }
    if let Some(r) = &q.reference {
        for n in std::iter::once(&r.net_input).chain(&r.expert_nets) {
            nets.insert(n.clone(), json!(path(n)?));
        }
        for n in &r.models {
            models.insert(n.clone(), json!(path(n)?));
        }
    }
    out["net_paths"] = Value::Object(nets);
    out["model_paths"] = Value::Object(models);
    Ok(out)
}

/// What the commands run with on this machine.
pub(crate) struct Run<'a> {
    pub table: &'a Path,
    pub threads: usize,
    pub tt_mb: u64,
    pub out: &'a Path,
    pub progress: &'a Path,
}

fn local(p: &Value, k: &str) -> Result<String, String> {
    p.get(k)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("hybrid_checks: no local path for `{k}` (inputs not resolved)"))
}

/// The `search hybrid-check` command of check `i` (`verify`: whether it recomputes the
/// verification sample of the stored worlds).
pub(crate) fn check_command(
    search: &Path,
    p: &Value,
    q: &Params,
    i: usize,
    verify: bool,
    r: &Run,
) -> Result<Command, String> {
    let c = &q.checks[i];
    let p_in = p;
    let p_models =
        |p: &Value, m: &str| -> Result<String, String> {
            p["model_paths"][m].as_str().map(str::to_string).ok_or_else(|| {
            format!("hybrid_checks: no local path for leaf-risk model `{m}` (inputs not resolved)")
        })
        };
    let net_path = |n: &str| -> Result<String, String> {
        p["net_paths"][n]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| {
                format!("hybrid_checks: no local path for network `{n}` (inputs not resolved)")
            })
    };
    let model_path = |m: &str| p_models(p_in, m);
    let nets: Vec<String> = c
        .nets
        .iter()
        .map(|n| net_path(n))
        .collect::<Result<_, _>>()?;
    let mut cmd = Command::new(search);
    cmd.args([
        "hybrid-check",
        "--trick",
        &q.trick.to_string(),
        "--cut",
        &q.cut.to_string(),
        "--net",
        &nets.join(","),
    ])
    .arg("--equity")
    .arg(r.table)
    .args(["--decisions", &local(p, "decisions")?])
    .args(["--to", &q.worlds.to_string()])
    .args(["--seed", &q.seed.to_string()]);
    if c.truth_net {
        cmd.args(["--truth", "net"])
            .args([
                "--noise-sd-by-class",
                c.noise_sd_by_class.as_deref().unwrap_or(""),
            ])
            .args(["--noise-seed", &c.noise_seed.to_string()]);
        if c.noise_calibrate > 0 {
            cmd.args(["--noise-calibrate", &c.noise_calibrate.to_string()])
                .args([
                    "--noise-calibrate-worlds",
                    &c.noise_calibrate_worlds.to_string(),
                ]);
        }
    } else {
        cmd.args(["--worlds", &local(p, "worlds_path")?]).args([
            "--verify-pct",
            &if verify {
                q.verify_pct.to_string()
            } else {
                "0".to_string()
            },
        ]);
        if let Some(rf) = &q.reference {
            cmd.args(["--reference", "silver"])
                .args(["--reference-net", &net_path(&rf.net_input)?])
                .args(["--reference-cut", &rf.cut.to_string()]);
            if let Some(l) = &rf.exact_leaves {
                cmd.args([
                    "--reference-exact-leaves",
                    &leaves_with_paths(l, &rf.models, &model_path)?,
                ])
                .args(["--reference-exact-leaf-tt-mb", &r.tt_mb.to_string()]);
            }
            if let Some(e) = &rf.defer_expert {
                cmd.args([
                    "--reference-defer-expert",
                    &expert_with_paths(e, &net_path, &model_path)?,
                ]);
            }
        }
    }
    cmd.args(["--threads", &r.threads.to_string()])
        .args(["--tt-mb", &r.tt_mb.to_string()])
        .args(["--boot", &q.boot.to_string()])
        .args(["--boot-seed", &q.boot_seed.to_string()])
        .args(["--standard", &q.standard]);
    if q.race_groups {
        cmd.arg("--race-groups");
    }
    if let Some(l) = &c.exact_leaves {
        cmd.args([
            "--exact-leaves",
            &leaves_with_paths(l, &c.models, &model_path)?,
        ])
        .args(["--exact-leaf-tt-mb", &r.tt_mb.to_string()]);
    }
    if let Some(e) = &c.defer_expert {
        cmd.args([
            "--defer-expert",
            &expert_with_paths(e, &net_path, &model_path)?,
        ]);
    }
    cmd.arg("--json")
        .arg(r.out.join(format!("{}.json", c.name)))
        .arg("--progress")
        .arg(r.progress);
    Ok(cmd)
}

/// The `search hybrid-paired` command of comparison `k`.
pub(crate) fn paired_command(search: &Path, q: &Params, k: usize, out: &Path) -> Command {
    let (name, a, b) = &q.paired[k];
    let mut cmd = Command::new(search);
    cmd.arg("hybrid-paired")
        .arg("--a")
        .arg(out.join(format!("{a}.json")))
        .arg("--b")
        .arg(out.join(format!("{b}.json")))
        .args(["--a-name", a, "--b-name", b])
        .args(["--boot", &q.boot.to_string()])
        .args(["--boot-seed", &q.boot_seed.to_string()])
        .arg("--json")
        .arg(out.join(format!("paired-{name}.json")));
    cmd
}

fn read_json(p: &Path) -> Option<Value> {
    fs::read_to_string(p)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
}

/// A check's summary for `result.json`: verdict, the overall, band, seat-1 and seat-2 rows, the
/// noise report and what it valued.
fn summary(hc: &Value) -> Value {
    let rows: Vec<Value> = hc["groups"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|g| {
            ["all", "pos 1-2", "pos 3-4", "pos 5-8", "pos 1", "pos 2"]
                .contains(&g["group"].as_str().unwrap_or(""))
        })
        .cloned()
        .collect();
    json!({
        "pass": hc["pass"], "values": hc["values"], "reference": hc["reference"]["kind"],
        "net_sha256": hc["net"]["sha256"], "ensemble": hc["ensemble"]["id"],
        "verified_worlds": hc["verified_worlds"],
        "rows": rows, "noise": hc["noise"], "time": hc["time"],
        "exact_leaves": hc["exact_leaves"],
    })
}

/// Run one `hybrid_checks` chunk (its inputs resolved to paths by `v2.rs`).
pub(crate) fn chunk(ctx: &mut Ctx, p: &Value) -> Result<Value, String> {
    parse(p)?;
    let base =
        json!({ "kind": "hybrid_checks", "step": "setup: table v0", "partial_cpu_ok": true });
    ctx.set_phase("chunk", base);
    let (table, table_rec) = build_table_from(ctx, p.get("table").unwrap_or(&json!({})), "auto")?;
    chunk_with(ctx, p, &sibling("search"), &table, table_rec)
}

/// The chunk after the equity table, with the given `search` program.
pub(crate) fn chunk_with(
    ctx: &mut Ctx,
    p: &Value,
    search: &Path,
    table: &Path,
    table_rec: Value,
) -> Result<Value, String> {
    let q = parse(p)?;
    let t0 = Instant::now();
    let threads = ctx.threads.max(1);
    // A check with exact leaves holds a second table per thread, for the leaf solves.
    let per_thread = if q.checks.iter().any(|c| c.exact_leaves.is_some())
        || q.reference
            .as_ref()
            .is_some_and(|r| r.exact_leaves.is_some())
    {
        2
    } else {
        1
    };
    let plan = mem::plan_here(
        Requested::Mb(q.tt_mb),
        Tables::Own {
            count: per_thread * threads as u64,
        },
        table,
        "hybrid_checks (hybrid and reference tables)",
    );
    let warnings: Vec<String> = plan.warning.iter().cloned().collect();
    let out = ctx.root.join("out").join("hybrid").join("checks");
    fs::create_dir_all(&out).map_err(|e| format!("mkdir {}: {e}", out.display()))?;
    let progress: PathBuf = ctx.root.join("checks.progress.json");
    let run = Run {
        table,
        threads,
        tt_mb: plan.tt_mb,
        out: &out,
        progress: &progress,
    };
    let first_exact = q.checks.iter().position(|c| !c.truth_net);
    let mut checks = Vec::new();
    for (i, c) in q.checks.iter().enumerate() {
        let t = Instant::now();
        let _ = fs::remove_file(&progress);
        let cmd = check_command(search, p, &q, i, Some(i) == first_exact, &run)?;
        let step = json!({ "kind": "hybrid_checks", "step": format!("check {} of {}: {}", i + 1, q.checks.len(), c.name),
                           "partial_cpu_ok": false, "warnings": warnings });
        ctx.set_phase("chunk", step.clone());
        ctx.run_child_io(
            cmd,
            &format!("hybrid/checks-{}.log", c.name),
            Some(&out.join(format!("{}.txt", c.name))),
            None,
            &mut |ctx| {
                let mut s = step.clone();
                if let Some(x) = read_json(&progress) {
                    for k in ["worlds_done", "worlds_total", "exact_done", "exact_total"] {
                        s[k] = x[k].clone();
                    }
                    s["partial_cpu_ok"] = json!(x["tail"].as_bool().unwrap_or(false));
                }
                ctx.progress = s;
            },
        )?;
        let hc = read_json(&out.join(format!("{}.json", c.name)))
            .ok_or_else(|| format!("hybrid_checks: hybrid-check wrote no {}.json", c.name))?;
        let mut s = summary(&hc);
        s["name"] = json!(c.name);
        s["truth"] = json!(if c.truth_net {
            "net"
        } else if q.reference.is_some() {
            "silver"
        } else {
            "exact"
        });
        if let Some(e) = &c.defer_expert {
            s["defer_expert"] = json!(e);
        }
        s["nets"] = json!(c.nets);
        s["seconds"] = json!((t.elapsed().as_secs_f64() * 10.0).round() / 10.0);
        checks.push(s);
    }
    let mut paired = Vec::new();
    for (k, (name, a, b)) in q.paired.iter().enumerate() {
        ctx.set_phase(
            "chunk",
            json!({ "kind": "hybrid_checks", "step": format!("paired comparison: {name}"), "partial_cpu_ok": true }),
        );
        ctx.run_child_io(
            paired_command(search, &q, k, &out),
            &format!("hybrid/paired-{name}.log"),
            Some(&out.join(format!("paired-{name}.txt"))),
            None,
            &mut |_| {},
        )?;
        let pj = read_json(&out.join(format!("paired-{name}.json")))
            .ok_or_else(|| format!("hybrid_checks: hybrid-paired wrote no paired-{name}.json"))?;
        paired.push(json!({ "name": name, "a": a, "b": b, "measures": pj["measures"] }));
    }
    let mut inputs = vec![("decisions", p["decisions"].as_str())];
    inputs.push(("worlds", p["worlds_path"].as_str()));
    let mut input_rec: Vec<Value> = inputs
        .into_iter()
        .filter_map(|(what, path)| {
            let (sha, bytes) = sha256_file(Path::new(path?)).ok()?;
            Some(json!({ "what": what, "sha256": sha, "bytes": bytes }))
        })
        .collect();
    for (n, path) in p["net_paths"].as_object().into_iter().flatten() {
        if let Some((sha, bytes)) = path.as_str().and_then(|x| sha256_file(Path::new(x)).ok()) {
            input_rec.push(json!({ "what": "net", "name": n, "sha256": sha, "bytes": bytes }));
        }
    }
    ctx.set_phase(
        "chunk",
        json!({ "kind": "hybrid_checks", "step": "done", "partial_cpu_ok": true }),
    );
    Ok(json!({
        "kind": "hybrid_checks",
        "trick": q.trick, "cut": q.cut, "worlds": q.worlds,
        "threads": threads, "tt_mb": plan.tt_mb, "tables": plan.record,
        "checks": checks,
        "paired": paired,
        "inputs": input_rec,
        "table": table_rec,
        "warnings": warnings,
        "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_params() -> Value {
        json!({ "threads": "all", "trick": 2, "cut": 3, "worlds": 320,
                "decisions_input": "dec", "worlds_input": "worlds", "verify_pct": 1,
                "checks": [
                  { "name": "curse", "nets": ["a"], "truth": "net",
                    "noise_sd_by_class": "0.978,0.918,2.010,1.519,4.050", "noise_seed": 3,
                    "noise_calibrate": 200 },
                  { "name": "single", "nets": ["a"] },
                  { "name": "ens", "nets": ["a", "b", "c"] } ],
                "paired": [ { "name": "single-vs-ens", "a": "single", "b": "ens" } ] })
    }

    #[test]
    fn settings_are_checked() {
        let q = parse(&manifest_params()).expect("valid");
        assert_eq!((q.trick, q.cut, q.worlds, q.tt_mb), (2, 3, 320, 64));
        assert_eq!(q.checks.len(), 3);
        assert!(q.checks[0].truth_net && !q.checks[1].truth_net);
        assert_eq!(q.checks[0].noise_seed, 3);
        assert_eq!(q.paired[0].2, "ens");
        assert!(q.race_groups);
        assert_eq!(
            input_names(&manifest_params()).expect("names"),
            vec!["dec", "worlds", "a", "b", "c"]
        );
        let bad = |f: &dyn Fn(&mut Value)| {
            let mut p = manifest_params();
            f(&mut p);
            parse(&p).is_err()
        };
        assert!(bad(&|p| p["typo"] = json!(1)));
        assert!(bad(&|p| p["cut"] = json!(2)));
        assert!(bad(&|p| p["checks"] = json!([])));
        assert!(bad(&|p| p["checks"][0]["extra"] = json!(1)));
        assert!(bad(&|p| p["checks"][1]["name"] = json!("curse")));
        assert!(bad(&|p| p["checks"][1]["name"] = json!("Bad Name")));
        assert!(bad(&|p| p["checks"][1]["nets"] = json!([])));
        assert!(bad(
            &|p| p["checks"][0]["noise_sd_by_class"] = json!("1,2,3")
        ));
        assert!(bad(&|p| {
            p["checks"][0]
                .as_object_mut()
                .expect("object")
                .remove("noise_sd_by_class");
        }));
        assert!(bad(&|p| p["checks"][1]["noise_seed"] = json!(2)));
        assert!(bad(&|p| p["checks"][1]["noise_calibrate"] = json!(2)));
        assert!(bad(&|p| p["checks"][0]["truth"] = json!("silver")));
        assert!(bad(&|p| p["paired"][0]["b"] = json!("curse")));
        assert!(bad(&|p| p["paired"][0]["b"] = json!("nope")));
        assert!(bad(&|p| p["paired"][0]["b"] = json!("single")));
        assert!(bad(&|p| p["verify_pct"] = json!(101)));
        assert!(bad(&|p| p["standard"] = json!("2")));
        assert!(bad(&|p| {
            p.as_object_mut().expect("object").remove("worlds_input");
        }));
        // Exact leaves: a policy in its normal form; none is no policy; refused with truth net
        // and when it does not parse.
        let mut p = manifest_params();
        p["checks"][1]["exact_leaves"] = json!("race:4,2");
        p["checks"][2]["exact_leaves"] = json!("none");
        let q = parse(&p).expect("valid");
        assert_eq!(q.checks[1].exact_leaves.as_deref(), Some("race:2,4"));
        assert_eq!(q.checks[2].exact_leaves, None);
        assert_eq!(q.checks[0].exact_leaves, None);
        assert!(bad(&|p| p["checks"][0]["exact_leaves"] = json!("race:4")));
        assert!(bad(&|p| p["checks"][1]["exact_leaves"] = json!("race:9")));
        assert!(bad(&|p| p["checks"][1]["exact_leaves"] = json!(4)));
        // Only the curse test: no exact worlds needed.
        let mut p = manifest_params();
        p.as_object_mut().expect("object").remove("worlds_input");
        p["checks"] = json!([p["checks"][0].clone()]);
        p["paired"] = json!([]);
        assert!(parse(&p).is_ok());
    }

    /// The inputs become paths; the first exact check verifies the stored worlds, the later one
    /// does not; the curse test passes its noise; the paired command reads both JSONs.
    #[test]
    fn inputs_resolve_into_the_commands() {
        let p = manifest_params();
        let mut inputs: BTreeMap<String, String> = BTreeMap::new();
        for (n, path) in [
            ("dec", "/in/d"),
            ("worlds", "/in/w"),
            ("a", "/in/a"),
            ("b", "/in/b"),
        ] {
            inputs.insert(n.into(), path.into());
        }
        let e = resolve(&p, &inputs).expect_err("c is missing");
        assert!(e.contains("`c`"), "{e}");
        inputs.insert("c".into(), "/in/c".into());
        let r = resolve(&p, &inputs).expect("resolves");
        let q = parse(&r).expect("still valid");
        assert_eq!(q, parse(&p).expect("valid"));
        let run = Run {
            table: Path::new("/t/eq.npy"),
            threads: 128,
            tt_mb: 64,
            out: Path::new("/o"),
            progress: Path::new("/p.json"),
        };
        let args = |c: Command| -> String {
            c.get_args()
                .map(|a| a.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join(" ")
        };
        assert_eq!(
            args(check_command(Path::new("search"), &r, &q, 0, false, &run).expect("cmd")),
            "hybrid-check --trick 2 --cut 3 --net /in/a --equity /t/eq.npy --decisions /in/d \
             --to 320 --seed 1 --truth net --noise-sd-by-class 0.978,0.918,2.010,1.519,4.050 \
             --noise-seed 3 --noise-calibrate 200 --noise-calibrate-worlds 8 --threads 128 --tt-mb 64 --boot 1000 --boot-seed 1 --standard 0.01 \
             --race-groups --json /o/curse.json --progress /p.json"
        );
        assert_eq!(
            args(check_command(Path::new("search"), &r, &q, 1, true, &run).expect("cmd")),
            "hybrid-check --trick 2 --cut 3 --net /in/a --equity /t/eq.npy --decisions /in/d \
             --to 320 --seed 1 --worlds /in/w --verify-pct 1 --threads 128 --tt-mb 64 --boot \
             1000 --boot-seed 1 --standard 0.01 --race-groups --json /o/single.json --progress \
             /p.json"
        );
        let ens = args(check_command(Path::new("search"), &r, &q, 2, false, &run).expect("cmd"));
        assert!(
            ens.contains("--net /in/a,/in/b,/in/c ") && ens.contains("--verify-pct 0 "),
            "{ens}"
        );
        assert_eq!(
            args(paired_command(Path::new("search"), &q, 0, Path::new("/o"))),
            "hybrid-paired --a /o/single.json --b /o/ens.json --a-name single --b-name ens \
             --boot 1000 --boot-seed 1 --json /o/paired-single-vs-ens.json"
        );
        // A check with exact leaves passes its policy and the leaf solves' table.
        let mut px = p.clone();
        px["checks"][1]["exact_leaves"] = json!("race:1-4");
        let rx = resolve(&px, &inputs).expect("resolves");
        let qx = parse(&rx).expect("valid");
        let x = args(check_command(Path::new("search"), &rx, &qx, 1, false, &run).expect("cmd"));
        assert!(
            x.ends_with(
                "--race-groups --exact-leaves race:1,2,3,4 --exact-leaf-tt-mb 64 --json \
                 /o/single.json --progress /p.json"
            ),
            "{x}"
        );
        // A learned term names its model as an input: kept as written, its name resolved into
        // the model's path in the command; an input it names must be there.
        let mut pl = p.clone();
        pl["checks"][1]["exact_leaves"] = json!("race:4+learned:risk:t=123");
        let ql = parse(&pl).expect("valid");
        assert_eq!(
            ql.checks[1].exact_leaves.as_deref(),
            Some("race:4+learned:risk:t=123")
        );
        assert_eq!(ql.checks[1].models, ["risk"]);
        assert!(resolve(&pl, &inputs).is_err(), "the model input is missing");
        let mut inl = inputs.clone();
        inl.insert("risk".into(), "/in/risk.json".into());
        let rl = resolve(&pl, &inl).expect("resolves");
        let x = args(
            check_command(
                Path::new("search"),
                &rl,
                &parse(&rl).expect("valid"),
                1,
                false,
                &run,
            )
            .expect("cmd"),
        );
        assert!(
            x.contains("--exact-leaves race:4+learned:/in/risk.json:t=123 --exact-leaf-tt-mb 64"),
            "{x}"
        );
        for bad_l in [
            "learned:risk",
            "learned:risk:t=-2",
            "learned:a:t=1+learned:b:t=1",
            "race:7+learned:risk:t=1",
        ] {
            let mut pb = p.clone();
            pb["checks"][1]["exact_leaves"] = json!(bad_l);
            assert!(parse(&pb).is_err(), "{bad_l}");
        }
        let mut pb = p.clone();
        pb["checks"][0]["exact_leaves"] = json!("learned:risk:t=1");
        assert!(parse(&pb).is_err(), "not with truth net");
        // Unresolved settings cannot make a command.
        assert!(check_command(Path::new("search"), &p, &q, 0, false, &run).is_err());
    }

    /// The chunk with a fake `search`: each check's report and JSON, the paired comparison, and
    /// the summary in `result.json`.
    #[cfg(unix)]
    #[test]
    fn chunk_runs_the_checks_and_the_comparison() {
        use std::os::unix::fs::PermissionsExt;
        let base = std::env::temp_dir().join(format!("hchecks-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("in")).expect("mkdir");
        let mut inputs: BTreeMap<String, String> = BTreeMap::new();
        for n in ["dec", "worlds", "a", "b", "c"] {
            let f = base.join("in").join(n);
            fs::write(&f, n).expect("write");
            inputs.insert(n.into(), f.display().to_string());
        }
        let table = base.join("equity-v0.npy");
        fs::write(&table, "table").expect("write");
        let calls = base.join("calls.log");
        let search = base.join("search");
        let arg = r#"arg() { n=$1; shift; while [ $# -gt 0 ]; do [ "$1" = "$n" ] && { echo "$2"; return; }; shift; done; }"#;
        fs::write(
            &search,
            format!(
                r#"#!/bin/sh
{arg}
echo "search $*" >> '{calls}'
case "$1" in
  hybrid-check)
    echo '{{"worlds_done": 1, "worlds_total": 2, "exact_done": 0, "exact_total": 0, "tail": false}}' > "$(arg --progress "$@")"
    echo '{{"pass": true, "values": {{"hybrid": "2+O3:x"}}, "reference": {{"kind": "exact"}}, "net": {{"sha256": "abc"}},
      "groups": [{{"group": "all", "summary": {{"mean_loss": 0.00005}}}}, {{"group": "pos 1", "summary": {{}}}}, {{"group": "pos 7", "summary": {{}}}}],
      "noise": {{"seed": 3}}}}' > "$(arg --json "$@")"
    echo "the check" ;;
  hybrid-paired)
    echo '{{"measures": [{{"measure": "loss, all", "diff": -0.00001}}]}}' > "$(arg --json "$@")"
    echo "the comparison" ;;
esac
"#,
                calls = calls.display()
            ),
        )
        .expect("write");
        fs::set_permissions(&search, fs::Permissions::from_mode(0o755)).expect("chmod");
        let p = resolve(&manifest_params(), &inputs).expect("resolves");
        let mut ctx = Ctx::local("hchecks-test", &base.join("chunk"), 4);
        let r = chunk_with(&mut ctx, &p, &search, &table, json!({"sha256": "t"})).expect("runs");
        assert_eq!(r["kind"], "hybrid_checks");
        assert_eq!(r["checks"].as_array().map(Vec::len), Some(3));
        assert_eq!(r["checks"][0]["truth"], "net");
        assert_eq!(r["checks"][0]["noise"]["seed"], 3);
        assert_eq!(r["checks"][1]["rows"].as_array().map(Vec::len), Some(2));
        assert_eq!(r["paired"][0]["measures"][0]["measure"], "loss, all");
        assert_eq!(r["inputs"].as_array().map(Vec::len), Some(5));
        let dir = base.join("chunk/out/hybrid/checks");
        for f in [
            "curse.txt",
            "single.json",
            "ens.txt",
            "paired-single-vs-ens.json",
        ] {
            assert!(dir.join(f).exists(), "{f}");
        }
        assert_eq!(
            fs::read_to_string(dir.join("paired-single-vs-ens.txt")).expect("txt"),
            "the comparison\n"
        );
        let call = fs::read_to_string(&calls).expect("calls");
        assert_eq!(call.lines().count(), 4, "{call}");
        assert!(
            call.lines()
                .nth(1)
                .is_some_and(|l| l.contains("--verify-pct 1 "))
        );
        assert!(
            call.lines()
                .nth(2)
                .is_some_and(|l| l.contains("--verify-pct 0 "))
        );
        let _ = fs::remove_dir_all(&base);
    }

    /// The committed manifest (docs/vast/o3-curse-ensemble.manifest.json) is one the runner
    /// accepts: a CPU machine (no GPU asked for), one `hybrid_checks` chunk with the curse test,
    /// the single network, the 4-network ensemble and their paired comparison, every input named.
    #[test]
    fn committed_manifest_parses() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let text = fs::read_to_string(root.join("docs/vast/o3-curse-ensemble.manifest.json"))
            .expect("read");
        let v: Value = serde_json::from_str(&text).expect("json");
        assert_eq!(v["run_id"], "o3-curse-ensemble-1");
        assert!(v["machine"].get("gpus").is_none(), "CPU only");
        assert_eq!(
            (&v["machine"]["min_cores"], &v["machine"]["min_ram_gb"]),
            (&json!(128), &json!(60))
        );
        let m = crate::sched::parse_v2(&v).expect("runnable");
        assert_eq!(m.streams.len(), 1);
        let s = &m.streams[0];
        assert_eq!((s.kind.as_str(), s.threads_all), ("hybrid_checks", true));
        let q = parse(&s.params).expect("settings");
        assert_eq!((q.trick, q.cut, q.worlds), (2, 3, 320));
        let names: Vec<&str> = q.checks.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["curse", "single", "ensemble4"]);
        assert!(q.checks[0].truth_net && q.checks[2].nets.len() == 4);
        assert_eq!(q.checks[0].nets, q.checks[1].nets);
        assert_eq!(q.paired.len(), 1);
        assert!(v.get("bin_sha").is_none(), "the launcher sets bin_sha");
    }

    /// The committed exact-leaves manifest (docs/vast/o3-exactleaf.manifest.json, fix G) is one
    /// the runner accepts: a CPU machine, one `hybrid_checks` chunk on the stored trick-2 golden
    /// set (2,000 decisions x 320 worlds) with the adopted O_3, the baseline without exact leaves
    /// (verifying the stored worlds first), race class 4, classes 2 and 4, classes 1-4, and each
    /// against the baseline by the paired comparison.
    #[test]
    fn committed_exactleaf_manifest_parses() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let text =
            fs::read_to_string(root.join("docs/vast/o3-exactleaf.manifest.json")).expect("read");
        let v: Value = serde_json::from_str(&text).expect("json");
        assert_eq!(v["run_id"], "o3-exactleaf-1");
        assert!(v["machine"].get("gpus").is_none(), "CPU only");
        assert_eq!(
            (&v["machine"]["min_cores"], &v["machine"]["min_ram_gb"]),
            (&json!(128), &json!(60))
        );
        assert_eq!(
            v["inputs"]["golden-t2-decisions"],
            "sha256:635e9aef196c816380f3f15c05ede90b38b875fa4817061a57e6d39550065ab0"
        );
        assert_eq!(
            v["inputs"]["golden-t2-worlds"],
            "sha256:6ddc59b191e1fd534f260f9dfe103f742d6026768ca74acdbbb4fd430eb8265e"
        );
        assert_eq!(
            v["inputs"]["o3-f5766a"],
            "sha256:f5766aa98a1d769e31122c94cc4da9790c84508fab5c432d0c0358fe3124faad"
        );
        let m = crate::sched::parse_v2(&v).expect("runnable");
        assert_eq!(m.streams.len(), 1);
        let s = &m.streams[0];
        assert_eq!((s.kind.as_str(), s.threads_all), ("hybrid_checks", true));
        let q = parse(&s.params).expect("settings");
        assert_eq!((q.trick, q.cut, q.worlds, q.verify_pct), (2, 3, 320, 1.0));
        let checks: Vec<(&str, Option<&str>)> = q
            .checks
            .iter()
            .map(|c| (c.name.as_str(), c.exact_leaves.as_deref()))
            .collect();
        assert_eq!(
            checks,
            [
                ("none", None),
                ("race4", Some("race:4")),
                ("race24", Some("race:2,4")),
                ("race1to4", Some("race:1,2,3,4")),
            ]
        );
        assert!(
            q.checks
                .iter()
                .all(|c| !c.truth_net && c.nets == ["o3-f5766a"])
        );
        let paired: Vec<(&str, &str)> = q
            .paired
            .iter()
            .map(|(_, a, b)| (a.as_str(), b.as_str()))
            .collect();
        assert_eq!(
            paired,
            [("none", "race4"), ("none", "race24"), ("none", "race1to4")]
        );
        assert!(v.get("bin_sha").is_none(), "the launcher sets bin_sha");
    }

    /// The committed learned-leaves manifest (docs/vast/o3-learnedleaf.manifest.json): a CPU
    /// machine, one `hybrid_checks` chunk on the stored trick-2 golden set with the adopted O_3:
    /// no exact leaves, G v1's race classes 1-4, and the learned policy at two thresholds of the
    /// stored leaf-risk model (an input, committed as data/leafrisk/), each against the baseline
    /// and the learned one against G v1 by the paired comparison; the model input resolves into
    /// the commands, and the committed model is the one the manifest names.
    #[test]
    fn committed_learnedleaf_manifest_parses() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let text =
            fs::read_to_string(root.join("docs/vast/o3-learnedleaf.manifest.json")).expect("read");
        let v: Value = serde_json::from_str(&text).expect("json");
        assert_eq!(v["run_id"], "o3-learnedleaf-1");
        assert!(v["machine"].get("gpus").is_none(), "CPU only");
        let exactleaf: Value = serde_json::from_str(
            &fs::read_to_string(root.join("docs/vast/o3-exactleaf.manifest.json")).expect("read"),
        )
        .expect("json");
        for k in ["golden-t2-decisions", "golden-t2-worlds", "o3-f5766a"] {
            assert_eq!(
                v["inputs"][k], exactleaf["inputs"][k],
                "{k}: the same as fix G v1's"
            );
        }
        let model = root.join("data/leafrisk/leafrisk-o3-v1.json");
        let sha = crate::bundle::sha256_file(&model)
            .expect("the committed model")
            .0;
        assert_eq!(v["inputs"]["leafrisk-o3-v1"], format!("sha256:{sha}"));
        let m = crate::sched::parse_v2(&v).expect("runnable");
        assert_eq!(m.streams.len(), 1);
        let s = &m.streams[0];
        assert_eq!((s.kind.as_str(), s.threads_all), ("hybrid_checks", true));
        let q = parse(&s.params).expect("settings");
        let e = parse(&exactleaf["streams"][0]["params"]).expect("settings");
        assert_eq!(
            (q.trick, q.cut, q.worlds, q.verify_pct, q.seed, q.tt_mb),
            (e.trick, e.cut, e.worlds, e.verify_pct, e.seed, e.tt_mb)
        );
        let names: Vec<&str> = q.checks.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["none", "race1to4", "learned", "learnedhalf"]);
        assert_eq!(q.checks[0].exact_leaves, None);
        assert_eq!(q.checks[1].exact_leaves.as_deref(), Some("race:1,2,3,4"));
        for c in &q.checks[2..] {
            assert_eq!(c.models, ["leafrisk-o3-v1"]);
            let text = c.exact_leaves.as_deref().expect("learned");
            let (_, thr) = search::exact_leaves::learned_parts(text).expect("a learned term");
            assert!(matches!(thr, search::exact_leaves::LearnedThreshold::At(_)));
        }
        assert!(
            q.checks
                .iter()
                .all(|c| !c.truth_net && c.nets == ["o3-f5766a"])
        );
        let paired: Vec<(&str, &str)> = q
            .paired
            .iter()
            .map(|(_, a, b)| (a.as_str(), b.as_str()))
            .collect();
        assert_eq!(
            paired,
            [
                ("none", "race1to4"),
                ("none", "learned"),
                ("none", "learnedhalf"),
                ("race1to4", "learned")
            ]
        );
        // The model input becomes the model's path in the commands, and the search reads the
        // committed model at the manifest's thresholds.
        let mut inputs: BTreeMap<String, String> = BTreeMap::new();
        for k in v["inputs"].as_object().expect("inputs").keys() {
            inputs.insert(k.clone(), format!("/in/{k}"));
        }
        let r = resolve(&s.params, &inputs).expect("resolves");
        let run = Run {
            table: Path::new("/t/eq.npy"),
            threads: 8,
            tt_mb: 64,
            out: Path::new("/o"),
            progress: Path::new("/p.json"),
        };
        for i in 2..4 {
            let c = check_command(Path::new("search"), &r, &q, i, false, &run).expect("cmd");
            let a: Vec<String> = c
                .get_args()
                .map(|a| a.to_string_lossy().into_owned())
                .collect();
            let at = a
                .iter()
                .position(|x| x == "--exact-leaves")
                .expect("policy");
            assert!(
                a[at + 1].starts_with("learned:/in/leafrisk-o3-v1:t="),
                "{a:?}"
            );
            let local = a[at + 1].replace("/in/leafrisk-o3-v1", &model.display().to_string());
            let p = search::exact_leaves::LeafPolicy::parse(&local, 64)
                .expect("the committed model loads")
                .expect("a policy");
            let l = p.learned.expect("learned");
            assert_eq!(l.model.sha256, sha);
            assert!(l.t.is_some());
        }
        assert!(v.get("bin_sha").is_none(), "the launcher sets bin_sha");
    }
}
