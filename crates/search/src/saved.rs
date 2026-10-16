//! Saved exact worlds as stored exact values for the hybrid check (docs/M3-HYBRID-SPEC.md §3,
//! "Trick-1 checks for O_3 and O_2"): the per-world exact values that earlier measurement runs
//! saved, read back so H_k can be compared with them without solving them again (an exact
//! trick-1 world costs 10–80×10⁹ solver nodes).
//!
//! **Formats read** (both `crate::measure`'s own, one JSON object per line; a line is a
//! checkpoint line when it has `w`, a worlds record when it has `rows`; a torn line, from a run
//! killed mid-write, is skipped and counted):
//! - checkpoint lines ([`CkptRow`], `<worlds file>.ckpt`): one world each, with its decision's
//!   fingerprint, method, label, seed, world source, the equity table's SHA-256 (absent before
//!   2026-10-15), the values version (absent = 1), the microseconds and (from 2026-10-16) the
//!   solver nodes it took;
//! - worlds records ([`WorldRows`], the worlds file): one decision's worlds `from..`, with its
//!   candidates and, from 2026-10-15, a fingerprint ([`RowFp`]: seed, decision, world source,
//!   table, values version); no times.
//!
//! The same world found more than once (a checkpoint line and the record written from it, or two
//! runs that valued the same world) is kept once, with every place it was found and the time and
//! nodes each recorded; its values must be identical everywhere, else the files are refused.
//!
//! **Burn rule.** A world's values version says how another seat's rob burn later in trick 1 was
//! valued ([`VALUES_VERSION`]): version 1 (not recorded) with the `rules-v1` burn, the current
//! version searched inside the solve. Each saved set is checked with the matching
//! `SearchConfig::rules_robber_burn` (version 1: `true`), in the hybrid's solves and in any exact
//! recompute. Version 1 also predates two other fixes (the turned card of a pending rob kept
//! live until the burn, and the tie-order merge), so its values equal the current solver's with
//! the switch only where a recomputed sample shows it (`--verify-pct`); the report says whether
//! that was done. Any other version is refused: no switch reproduces it.
//!
//! **Checks against the decisions file** (`plan`): every saved world's decision must be in the
//! file, with the same fingerprint ([`decision_fingerprint`]) and, where recorded, the same
//! candidates; the world source must be this run's; the equity table must be this run's where it
//! is recorded (where it is not, the report says so). The world itself is regenerated from (seed,
//! label, decision id, world number) exactly as the yardstick draws it, never stored.

use std::collections::BTreeMap;
use std::path::Path;

use equity::MoneyChance;
use serde::Serialize;
use serde_json::Value;

use crate::hybrid::{DecisionWorlds, WorldTask};
use crate::measure::{
    CkptRow, DecisionSpec, VALUES_VERSION, WorldRows, decision_fingerprint, measure_seed,
};
use crate::pimc::{Candidate, candidates};

/// How another seat's rob burn later in trick 1 was valued.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum BurnRule {
    /// Searched inside the exact solve (values version 2 on).
    Searched,
    /// The `rules-v1` burn (values version 1): `SearchConfig::rules_robber_burn`.
    RulesV1,
}

impl BurnRule {
    /// The rule of values version `values`; `None` for a version no switch reproduces.
    pub fn of_version(values: u32) -> Option<BurnRule> {
        match values {
            1 => Some(BurnRule::RulesV1),
            v if v == VALUES_VERSION => Some(BurnRule::Searched),
            _ => None,
        }
    }

    /// `SearchConfig::rules_robber_burn` for this rule.
    pub fn rules_robber_burn(self) -> bool {
        self == BurnRule::RulesV1
    }

    /// A short name for reports.
    pub fn name(self) -> &'static str {
        match self {
            BurnRule::Searched => "searched burn",
            BurnRule::RulesV1 => "rules-v1 burn",
        }
    }
}

/// One place a saved world was found, with the time and nodes recorded there.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SavedTime {
    /// The file's label (normally the run id).
    pub from: String,
    /// Microseconds the exact world took there (checkpoint lines only).
    pub us: Option<u64>,
    /// Solver nodes it took there (checkpoint lines from 2026-10-16 only).
    pub nodes: Option<u64>,
}

/// One saved exact world.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SavedWorld {
    /// The decision's id.
    pub id: u64,
    /// World number.
    pub w: u64,
    /// Values version (1 where none is recorded).
    pub values: u32,
    /// `exact` (other methods are skipped when read).
    pub method: String,
    /// World label.
    pub label: u64,
    /// Master seed; `None` in a worlds record without a fingerprint.
    pub seed: Option<u64>,
    /// The world source's name, where recorded.
    pub source: Option<String>,
    /// The equity table's SHA-256, where recorded.
    pub table: Option<String>,
    /// The decision's fingerprint, where recorded.
    pub decision_fp: Option<u64>,
    /// The candidates, where recorded (worlds records).
    #[serde(skip)]
    pub cands: Option<Vec<Candidate>>,
    /// The candidates' values (four u64 each).
    pub v: Vec<[u64; 4]>,
    /// Every place it was found.
    pub found: Vec<SavedTime>,
}

impl SavedWorld {
    /// The burn rule its values were made with.
    pub fn burn_rule(&self) -> Option<BurnRule> {
        BurnRule::of_version(self.values)
    }

    /// Its expected cost for ordering the work (never a value): the most solver nodes any place
    /// recorded, else the most microseconds (lines made before 2026-10-16 have no nodes), else 0.
    pub fn cost(&self) -> u64 {
        let most = |f: fn(&SavedTime) -> Option<u64>| self.found.iter().filter_map(f).max();
        most(|t| t.nodes).or_else(|| most(|t| t.us)).unwrap_or(0)
    }
}

/// What one file held.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SavedFile {
    /// The file's label.
    pub from: String,
    /// Lines read (not counting blank ones).
    pub lines: u64,
    /// Torn lines skipped.
    pub torn: u64,
    /// Checkpoint lines.
    pub ckpt_lines: u64,
    /// Worlds records.
    pub records: u64,
    /// Exact worlds read (a record counts each of its worlds).
    pub worlds: u64,
    /// Worlds of other methods, skipped.
    pub other_method: u64,
    /// Exact worlds by values version.
    pub by_version: BTreeMap<u32, u64>,
}

/// Reads one file of checkpoint lines and/or worlds records, labelled `from`.
pub fn read_saved(path: &Path, from: &str) -> Result<(Vec<SavedWorld>, SavedFile), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_saved(&text, from).map_err(|e| format!("{}: {e}", path.display()))
}

/// [`read_saved`] on the text of a file.
pub fn parse_saved(text: &str, from: &str) -> Result<(Vec<SavedWorld>, SavedFile), String> {
    let mut out = Vec::new();
    let mut info = SavedFile {
        from: from.to_string(),
        ..SavedFile::default()
    };
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        info.lines += 1;
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            info.torn += 1;
            continue;
        };
        let line_no = n + 1;
        let bad = |e: serde_json::Error| format!("line {line_no}: {e}");
        if v.get("w").is_some() {
            let r: CkptRow = serde_json::from_value(v).map_err(bad)?;
            info.ckpt_lines += 1;
            if r.method != "exact" {
                info.other_method += 1;
                continue;
            }
            out.push(SavedWorld {
                id: r.id,
                w: r.w,
                values: r.values.unwrap_or(1),
                method: r.method,
                label: r.label,
                seed: Some(r.seed),
                source: Some(r.source),
                table: (!r.table.is_empty()).then_some(r.table),
                decision_fp: Some(r.fp),
                cands: None,
                v: r.v,
                found: vec![SavedTime {
                    from: from.to_string(),
                    us: Some(r.us),
                    nodes: (r.nodes > 0).then_some(r.nodes),
                }],
            });
        } else if v.get("rows").is_some() {
            let r: WorldRows = serde_json::from_value(v).map_err(bad)?;
            info.records += 1;
            if r.method != "exact" {
                info.other_method += r.rows.len() as u64;
                continue;
            }
            for (i, row) in r.rows.iter().enumerate() {
                out.push(SavedWorld {
                    id: r.id,
                    w: r.from + i as u64,
                    values: r.fp.as_ref().and_then(|f| f.values).unwrap_or(1),
                    method: r.method.clone(),
                    label: r.label,
                    seed: r.fp.as_ref().map(|f| f.seed),
                    source: r.fp.as_ref().map(|f| f.source.clone()),
                    table: r.fp.as_ref().map(|f| f.table.clone()),
                    decision_fp: r.fp.as_ref().map(|f| f.decision),
                    cands: Some(r.cands.clone()),
                    v: row.clone(),
                    found: vec![SavedTime {
                        from: from.to_string(),
                        us: None,
                        nodes: None,
                    }],
                });
            }
        } else {
            return Err(format!(
                "line {line_no}: neither a checkpoint line (`w`) nor a worlds record (`rows`)"
            ));
        }
    }
    for s in &out {
        info.worlds += 1;
        *info.by_version.entry(s.values).or_default() += 1;
    }
    Ok((out, info))
}

/// Joins saved worlds: the same world (decision, number, values version, label, seed, world
/// source) found more than once is kept once with every place it was found; its values, its
/// table and its decision fingerprint must agree everywhere. Sorted by (values version, decision,
/// world).
pub fn join_saved(all: Vec<SavedWorld>) -> Result<Vec<SavedWorld>, String> {
    type Key = (u32, u64, u64, u64, Option<u64>, Option<String>);
    let mut by: BTreeMap<Key, SavedWorld> = BTreeMap::new();
    for s in all {
        let k = (s.values, s.id, s.w, s.label, s.seed, s.source.clone());
        match by.get_mut(&k) {
            None => {
                by.insert(k, s);
            }
            Some(e) => {
                if e.v != s.v {
                    return Err(format!(
                        "decision {} world {} (values version {}) was saved twice with different \
                         values ({} and {}): the files cannot both be right",
                        s.id,
                        s.w,
                        s.values,
                        e.found.first().map_or("?", |f| f.from.as_str()),
                        s.found.first().map_or("?", |f| f.from.as_str()),
                    ));
                }
                for (a, b, what) in [
                    (&e.table, &s.table, "equity table"),
                    (
                        &e.decision_fp.map(|x| x.to_string()),
                        &s.decision_fp.map(|x| x.to_string()),
                        "decision fingerprint",
                    ),
                ] {
                    if let (Some(a), Some(b)) = (a, b)
                        && a != b
                    {
                        return Err(format!(
                            "decision {} world {}: two records name different {what}s",
                            s.id, s.w
                        ));
                    }
                }
                e.table = e.table.take().or(s.table);
                e.decision_fp = e.decision_fp.or(s.decision_fp);
                e.cands = e.cands.take().or(s.cands);
                e.found.extend(s.found);
            }
        }
    }
    Ok(by.into_values().collect())
}

/// The saved worlds valued under one burn rule.
#[derive(Clone, Debug)]
pub struct SavedSet {
    /// The rule.
    pub rule: BurnRule,
    /// The values version.
    pub values: u32,
    /// The worlds, by (decision, world).
    pub worlds: Vec<SavedWorld>,
}

/// Splits joined worlds into sets by values version (one burn rule each), current version first.
/// A version no switch reproduces is refused.
pub fn saved_sets(joined: Vec<SavedWorld>) -> Result<Vec<SavedSet>, String> {
    let mut by: BTreeMap<BurnRule, SavedSet> = BTreeMap::new();
    for s in joined {
        let rule = s.burn_rule().ok_or_else(|| {
            format!(
                "decision {} world {}: values version {} (this binary: {VALUES_VERSION}; 1 = the \
                 rules-v1 burn): no switch of this binary reproduces it",
                s.id, s.w, s.values
            )
        })?;
        by.entry(rule)
            .or_insert_with(|| SavedSet {
                rule,
                values: s.values,
                worlds: Vec::new(),
            })
            .worlds
            .push(s);
    }
    Ok(by.into_values().collect())
}

/// The worlds saved under both burn rules (same decision, world number, label and seed):
/// `(decision, world, values identical)`. Where no other seat robs later in trick 1 the rules
/// agree by construction, so a difference there points at the other changes since version 1.
pub fn both_rules(sets: &[SavedSet]) -> Vec<(u64, u64, bool)> {
    let of = |r: BurnRule| sets.iter().find(|s| s.rule == r);
    let (Some(a), Some(b)) = (of(BurnRule::Searched), of(BurnRule::RulesV1)) else {
        return Vec::new();
    };
    let key = |s: &SavedWorld| (s.id, s.w, s.label, s.seed);
    let old: BTreeMap<_, &SavedWorld> = b.worlds.iter().map(|s| (key(s), s)).collect();
    a.worlds
        .iter()
        .filter_map(|s| old.get(&key(s)).map(|o| (s.id, s.w, o.v == s.v)))
        .collect()
}

/// What [`plan`] checked, for the report.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PlanNotes {
    /// Decisions and worlds planned.
    pub decisions: usize,
    /// See `decisions`.
    pub worlds: usize,
    /// Worlds whose equity table is not recorded (made before 2026-10-15): not checked.
    pub table_unrecorded: usize,
    /// Worlds whose seed is not recorded: the run's seed was assumed.
    pub seed_unrecorded: usize,
    /// Worlds whose decision fingerprint is not recorded: not checked.
    pub fp_unrecorded: usize,
}

/// The check plan of a saved set against the decisions `ds` (each world regenerated from its
/// seed, label, decision id and number): the decisions in `ds` order, each with its saved worlds
/// that `keep(id, w)` accepts, in world order. `seed` is assumed where a world records none;
/// `source` is this run's world source and `table` its equity table's SHA-256.
pub fn plan<'a>(
    ds: &'a [DecisionSpec],
    set: &SavedSet,
    seed: u64,
    source: &str,
    table: &str,
    keep: &dyn Fn(u64, u64) -> bool,
) -> Result<(Vec<DecisionWorlds<'a>>, PlanNotes), String> {
    let mut notes = PlanNotes::default();
    let by_id: BTreeMap<u64, &DecisionSpec> = ds.iter().map(|d| (d.id, d)).collect();
    let mut tasks: BTreeMap<u64, Vec<WorldTask>> = BTreeMap::new();
    let mut cands_of: BTreeMap<u64, Vec<Candidate>> = BTreeMap::new();
    for s in set.worlds.iter().filter(|s| keep(s.id, s.w)) {
        let d = by_id.get(&s.id).ok_or_else(|| {
            format!(
                "decision {} (saved world {}) is not in the decisions file",
                s.id, s.w
            )
        })?;
        match s.decision_fp {
            Some(fp) if fp != decision_fingerprint(d) => {
                return Err(format!(
                    "decision {}: the saved worlds were made for another decision of that id \
                     (fingerprint {fp}, the decisions file's {}): another decision set?",
                    s.id,
                    decision_fingerprint(d)
                ));
            }
            Some(_) => {}
            None => notes.fp_unrecorded += 1,
        }
        if let Some(src) = &s.source
            && src != source
        {
            return Err(format!(
                "decision {} world {}: drawn by the {src} world source, this run uses {source} \
                 (--source)",
                s.id, s.w
            ));
        }
        match &s.table {
            Some(t) if t != table => {
                return Err(format!(
                    "decision {} world {}: valued with equity table {}, this run loads {}",
                    s.id,
                    s.w,
                    &t[..t.len().min(12)],
                    &table[..table.len().min(12)]
                ));
            }
            Some(_) => {}
            None => notes.table_unrecorded += 1,
        }
        let cands = cands_of.entry(s.id).or_insert_with(|| {
            let (_, view, legal) = d.replay();
            candidates(&view, &legal).1
        });
        if s.cands.as_ref().is_some_and(|c| c != cands) {
            return Err(format!("decision {}: saved candidates differ", s.id));
        }
        if s.v.len() != cands.len() {
            return Err(format!(
                "decision {} world {}: {} saved values for {} candidates",
                s.id,
                s.w,
                s.v.len(),
                cands.len()
            ));
        }
        if s.seed.is_none() {
            notes.seed_unrecorded += 1;
        }
        let stored =
            s.v.iter()
                .map(|&x| {
                    MoneyChance::from_raw(x).ok_or_else(|| {
                        format!(
                            "decision {} world {}: saved values do not sum to one",
                            s.id, s.w
                        )
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
        tasks.entry(s.id).or_default().push(WorldTask {
            w: s.w,
            dseed: measure_seed(s.seed.unwrap_or(seed), s.label, s.id),
            stored: Some(stored),
            cost: s.cost(),
        });
    }
    let mut out = Vec::new();
    for d in ds {
        if let Some(mut ws) = tasks.remove(&d.id) {
            ws.sort_by_key(|t| t.w);
            notes.worlds += ws.len();
            out.push(DecisionWorlds { d, worlds: ws });
        }
    }
    notes.decisions = out.len();
    Ok((out, notes))
}

/// A `D:W` list (`0:3,4:1`): decision and world pairs.
pub fn parse_pairs(arg: &str) -> Result<Vec<(u64, u64)>, String> {
    arg.split(',')
        .filter(|t| !t.trim().is_empty())
        .map(|t| {
            let (d, w) = t
                .trim()
                .split_once(':')
                .ok_or_else(|| format!("`{t}` is not DECISION:WORLD"))?;
            let n = |x: &str| x.parse::<u64>().map_err(|_| format!("`{t}`: not a number"));
            Ok((n(d)?, n(w)?))
        })
        .collect()
}

#[cfg(test)]
mod tests;
