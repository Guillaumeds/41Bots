//! The `hybrid_t1check` chunk kind (docs/M3-HYBRID-SPEC.md §3, "Trick-1 checks for O_3 and
//! O_2"; docs/m3-hybrid/rung2.md §4): the hybrid H_cut against the exact trick-1 worlds earlier
//! runs saved, on one machine with every usable thread, without any new exact trick-1 solve
//! (unless a verification sample is asked for).
//!
//! ```json
//! { "name": "t1check", "priority": 1, "kind": "hybrid_t1check", "chunks": 1,
//!   "params": { "threads": "all",
//!     "net_input": "o3", "decisions_input": "t1-decisions",
//!     "saved": [ { "input": "t1-tables-1", "run": "m3-t1-tables-1" },
//!                { "input": "t1-ab1", "run": "m3-agree-t1-ab1" } ],
//!     "trick": 1, "cut": 3, "hard": "0:3", "tt_mb": 64,
//!     "verify_pct": 0, "verify_rule": "all", "verify_solve_threads": 1, "verify_tt_mb": 64 } }
//! ```
//! The files are format-2 manifest `inputs` (content-addressed, `inputs/sha256/<hex>`, fetched by
//! `remote::fetch_input` and SHA-256 checked before the chunk starts; the launcher checks that
//! they exist before renting): `net_input` (an `int.safetensors` trained for trick `cut`),
//! `decisions_input` (the decisions file the worlds were valued for) and each `saved[].input` (a
//! worlds file or its `.ckpt` checkpoint; `run` labels it in the report). `v2.rs` turns each name
//! into its local path. Optional: `only` (`"D:W,…"`: just those worlds), `hard` (`"D:W,…"`: on
//! their own lines with the saved exact times; default `"0:3"`, the spec's known hard world),
//! `seed` (1), `boot` (1000), `boot_seed` (1), `table` and `equity_input` (the equity table, as
//! `search_measure`'s). Any other key is refused ([`KEYS`]).
//!
//! One step: `search hybrid-check --trick T --cut K --net … --decisions … --saved RUN=PATH …`
//! (`search::saved`): the saved worlds split by the burn rule that valued them, each set checked
//! with the matching `rules_robber_burn`, the sets at once with the threads shared out by worlds,
//! one world per thread (each with a `tt_mb` table: about `tt_mb` + 40 MB a thread), the largest
//! saved worlds first. `verify_pct` recomputes that share of the saved worlds exactly first
//! (`verify_rule`: `all`, `rules-v1` or `searched`), each solve on `verify_solve_threads`
//! threads with a `verify_tt_mb` table, and the set is compared only if every recomputed world is
//! byte-identical; 0 (the default) recomputes none, and the report says so.
//!
//! Outputs: `out/hybrid/t1check.txt` (the printed report: per world, per decision, the §3 tables
//! per set, the hard worlds, memory) and `out/hybrid/t1check.json` (everything, every world's
//! exact and hybrid values in full); the child's log (each world as it finishes) in the chunk's
//! logs. `result.json` gives each set's verdict and verification and the hard worlds.
//!
//! Memory: the hybrid tables are sized as `mem.rs` plans them (an explicit `tt_mb` is only
//! lowered, with a warning, if the threads' tables would not fit), and the verification tables
//! within what is left; the chunk counts as needing the whole machine (`mem::chunk_need`).
//!
//! CPU: worlds differ in cost by 1,000 times or more, so the end of the check runs the last
//! large worlds on a few threads while the rest are idle: `partial_cpu_ok` is set whenever fewer
//! worlds (or verification solves) are left than threads, the genuine tail, and only then.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde_json::{Value, json};

use crate::bundle::sha256_file;
use crate::mem::{self, Requested, Tables};
use crate::spike::build_table_from;
use crate::{Ctx, sibling};

/// The settings a `hybrid_t1check` chunk takes (after `v2.rs` added the inputs' paths: `net`,
/// `decisions`, `saved[].path`); any other key is refused.
pub(crate) const KEYS: [&str; 20] = [
    "net_input",
    "decisions_input",
    "saved",
    "trick",
    "cut",
    "only",
    "hard",
    "tt_mb",
    "verify_pct",
    "verify_rule",
    "verify_solve_threads",
    "verify_tt_mb",
    "seed",
    "boot",
    "boot_seed",
    "table",
    "equity_input",
    "threads",
    "est_seconds",
    "chunk_sizes",
];

/// Keys `v2.rs` adds: the local paths of the named inputs.
const RESOLVED: [&str; 2] = ["net", "decisions"];

/// The checked settings.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Params {
    pub trick: u64,
    pub cut: u64,
    pub net_input: String,
    pub decisions_input: String,
    /// (input name, run label).
    pub saved: Vec<(String, String)>,
    pub only: Option<String>,
    pub hard: String,
    pub tt_mb: u64,
    pub verify_pct: f64,
    pub verify_rule: String,
    pub verify_solve_threads: u64,
    pub verify_tt_mb: u64,
    pub seed: u64,
    pub boot: u64,
    pub boot_seed: u64,
}

fn pairs_ok(s: &str) -> bool {
    s.split(',').all(|t| {
        t.trim().split_once(':').is_some_and(|(d, w)| {
            d.trim().parse::<u64>().is_ok() && w.trim().parse::<u64>().is_ok()
        })
    })
}

/// Checks the settings (also when the manifest is read, so a typo stops the run before it
/// starts).
pub(crate) fn parse(p: &Value) -> Result<Params, String> {
    let o = p
        .as_object()
        .ok_or("hybrid_t1check: params must be an object")?;
    for k in o.keys() {
        if !KEYS.contains(&k.as_str()) && !RESOLVED.contains(&k.as_str()) {
            return Err(format!("hybrid_t1check: unknown setting `{k}`"));
        }
    }
    let n = |k: &str, d: u64| -> Result<u64, String> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(v) => v
                .as_u64()
                .ok_or_else(|| format!("hybrid_t1check: {k} must be a whole number")),
        }
    };
    let s = |k: &str| -> Result<Option<String>, String> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(x)) => Ok(Some(x.clone())),
            Some(_) => Err(format!("hybrid_t1check: {k} must be a string")),
        }
    };
    let trick = n("trick", 1)?;
    let cut = n("cut", 3)?;
    if trick == 0 || trick >= cut || cut > 5 {
        return Err(format!(
            "hybrid_t1check: cut {cut} must be after trick {trick} (1 ≤ trick < cut ≤ 5)"
        ));
    }
    let net_input = s("net_input")?.ok_or("hybrid_t1check: needs net_input (an input name)")?;
    let decisions_input =
        s("decisions_input")?.ok_or("hybrid_t1check: needs decisions_input (an input name)")?;
    let saved = p
        .get("saved")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or("hybrid_t1check: needs saved: a list of {\"input\": NAME, \"run\": LABEL}")?
        .iter()
        .map(|e| {
            let get = |k: &str| e.get(k).and_then(Value::as_str).map(str::to_string);
            let input = get("input").ok_or("hybrid_t1check: each saved entry needs input")?;
            let run = get("run").unwrap_or_else(|| input.clone());
            if run.contains('=') || run.is_empty() {
                return Err(format!(
                    "hybrid_t1check: run label `{run}`: no `=`, not empty"
                ));
            }
            for k in e.as_object().into_iter().flat_map(|o| o.keys()) {
                if !["input", "run", "path"].contains(&k.as_str()) {
                    return Err(format!("hybrid_t1check: saved entry: unknown key `{k}`"));
                }
            }
            Ok((input, run))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let only = s("only")?;
    let hard = s("hard")?.unwrap_or_else(|| "0:3".to_string());
    for (k, v) in [("only", only.as_deref()), ("hard", Some(hard.as_str()))] {
        if let Some(v) = v
            && !pairs_ok(v)
        {
            return Err(format!(
                "hybrid_t1check: {k} must be \"DECISION:WORLD,…\", not {v}"
            ));
        }
    }
    let verify_pct = match p.get("verify_pct") {
        None | Some(Value::Null) => 0.0,
        Some(v) => v
            .as_f64()
            .filter(|x| (0.0..=100.0).contains(x))
            .ok_or("hybrid_t1check: verify_pct must be a number from 0 to 100")?,
    };
    let verify_rule = s("verify_rule")?.unwrap_or_else(|| "all".to_string());
    if !["all", "rules-v1", "searched"].contains(&verify_rule.as_str()) {
        return Err(format!(
            "hybrid_t1check: verify_rule {verify_rule}: all, rules-v1 or searched"
        ));
    }
    let tt_mb = n("tt_mb", 64)?.max(1);
    Ok(Params {
        trick,
        cut,
        net_input,
        decisions_input,
        saved,
        only,
        hard,
        tt_mb,
        verify_pct,
        verify_rule,
        verify_solve_threads: n("verify_solve_threads", 1)?.max(1),
        verify_tt_mb: n("verify_tt_mb", tt_mb)?.max(1),
        seed: n("seed", 1)?,
        boot: n("boot", 1000)?,
        boot_seed: n("boot_seed", 1)?,
    })
}

/// The manifest input names a chunk's settings use (for `v2.rs` to turn into paths).
pub(crate) fn input_names(p: &Value) -> Result<Vec<String>, String> {
    let q = parse(p)?;
    let mut v = vec![q.net_input, q.decisions_input];
    v.extend(q.saved.into_iter().map(|(i, _)| i));
    Ok(v)
}

/// Adds the local path of each named input to the settings (`net`, `decisions`,
/// `saved[].path`), from the worker's `inputs` (name → path). A name the manifest does not
/// list is an error.
pub(crate) fn resolve(
    p: &Value,
    inputs: &std::collections::BTreeMap<String, String>,
) -> Result<Value, String> {
    let q = parse(p)?;
    let path = |name: &str| {
        inputs
            .get(name)
            .cloned()
            .ok_or_else(|| format!("hybrid_t1check needs input `{name}` in the manifest's inputs"))
    };
    let mut out = p.clone();
    out["net"] = json!(path(&q.net_input)?);
    out["decisions"] = json!(path(&q.decisions_input)?);
    let saved = out["saved"]
        .as_array_mut()
        .ok_or("hybrid_t1check: saved must be a list")?;
    for (e, (input, _)) in saved.iter_mut().zip(&q.saved) {
        e["path"] = json!(path(input)?);
    }
    Ok(out)
}

/// What a chunk's command runs with on this machine.
pub(crate) struct Run<'a> {
    pub table: &'a Path,
    pub threads: usize,
    /// The hybrid's and the verification's table sizes, as planned here.
    pub tt_mb: u64,
    pub verify_tt_mb: u64,
    /// The output directory and the child's progress file.
    pub out: &'a Path,
    pub progress: &'a Path,
}

/// The `search hybrid-check` command of a chunk.
pub(crate) fn command(search: &Path, p: &Value, q: &Params, r: &Run) -> Result<Command, String> {
    let Run {
        table,
        threads,
        tt_mb,
        verify_tt_mb,
        out,
        progress,
    } = *r;
    let get = |k: &str| {
        p.get(k)
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .ok_or_else(|| format!("hybrid_t1check: no local path for `{k}` (inputs not resolved)"))
    };
    let mut c = Command::new(search);
    c.args([
        "hybrid-check",
        "--trick",
        &q.trick.to_string(),
        "--cut",
        &q.cut.to_string(),
    ])
    .arg("--net")
    .arg(get("net")?)
    .arg("--equity")
    .arg(table)
    .arg("--decisions")
    .arg(get("decisions")?);
    let saved = p["saved"].as_array().cloned().unwrap_or_default();
    for (e, (_, run)) in saved.iter().zip(&q.saved) {
        let path = e["path"]
            .as_str()
            .ok_or("hybrid_t1check: a saved entry has no local path (inputs not resolved)")?;
        c.arg("--saved").arg(format!("{run}={path}"));
    }
    if let Some(o) = &q.only {
        c.args(["--only", o]);
    }
    c.args(["--hard", &q.hard])
        .args(["--seed", &q.seed.to_string()])
        .args(["--threads", &threads.to_string()])
        .args(["--tt-mb", &tt_mb.to_string()])
        .args(["--verify-pct", &q.verify_pct.to_string()])
        .args(["--verify-rule", &q.verify_rule])
        .args([
            "--verify-solve-threads",
            &q.verify_solve_threads.to_string(),
        ])
        .args(["--verify-tt-mb", &verify_tt_mb.to_string()])
        .args(["--boot", &q.boot.to_string()])
        .args(["--boot-seed", &q.boot_seed.to_string()])
        .arg("--json")
        .arg(out.join("t1check.json"))
        .arg("--progress")
        .arg(progress);
    Ok(c)
}

fn read_json(p: &Path) -> Option<Value> {
    fs::read_to_string(p)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
}

/// The tables of a saved-worlds check on this machine ([`plan_tables`]).
pub(crate) struct TablePlan {
    /// The hybrid's table per thread, and its plan's record.
    pub tt_mb: u64,
    pub record: Value,
    /// The verification's table per solve, and its plan's record (null without verification).
    pub verify_tt_mb: u64,
    pub verify_record: Value,
    pub warnings: Vec<String>,
}

/// The tables of a saved-worlds check on `threads` threads: the hybrid's one per thread, as
/// `mem.rs` plans them (an explicit `tt_mb` only lowered, with a warning, if they would not fit),
/// and the verification's within what is left. Shared by the `hybrid_t1check` chunk and the
/// audit loop's rough test at trick 2 (`hybrid/audit.rs`).
pub(crate) fn plan_tables(threads: usize, q: &Params, table: &Path, what: &str) -> TablePlan {
    let hyb = mem::plan_here(
        Requested::Mb(q.tt_mb),
        Tables::Own {
            count: threads as u64,
        },
        table,
        &format!("{what} (hybrid tables)"),
    );
    let mut warnings: Vec<String> = hyb.warning.iter().cloned().collect();
    let mut verify_tt = q.verify_tt_mb;
    let mut verify_rec = Value::Null;
    if q.verify_pct > 0.0 {
        let workers = (threads as u64 / q.verify_solve_threads).max(1);
        let tables = Tables::Shared {
            count: workers,
            solve_threads: q.verify_solve_threads,
        };
        let limit = mem::Sys::host().limit();
        let other = mem::other_bytes(table, threads as u64)
            + hyb.record["tables_peak_mb"].as_u64().unwrap_or(0) * (1 << 20);
        let plan = mem::plan(Requested::Mb(q.verify_tt_mb), tables, other, &limit);
        if let Some(w) = &plan.warning {
            eprintln!("runner: WARNING: {what} (verification tables): {w}");
            warnings.push(w.clone());
        }
        verify_tt = plan.tt_mb;
        verify_rec = plan.record;
    }
    TablePlan {
        tt_mb: hyb.tt_mb,
        record: hyb.record,
        verify_tt_mb: verify_tt,
        verify_record: verify_rec,
        warnings,
    }
}

/// Run one `hybrid_t1check` chunk (its inputs resolved to paths by `v2.rs`).
pub(crate) fn chunk(ctx: &mut Ctx, p: &Value) -> Result<Value, String> {
    parse(p)?;
    let base =
        json!({ "kind": "hybrid_t1check", "step": "setup: table v0", "partial_cpu_ok": true });
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
    let table = table.to_path_buf();
    let threads = ctx.threads.max(1);
    // Tables: the hybrid's one per thread; the verification's within what is left.
    let TablePlan {
        tt_mb: hyb_tt,
        record: hyb_rec,
        verify_tt_mb: verify_tt,
        verify_record: verify_rec,
        warnings,
    } = plan_tables(threads, &q, &table, "hybrid_t1check");
    let out = ctx.root.join("out").join("hybrid");
    fs::create_dir_all(&out).map_err(|e| format!("mkdir {}: {e}", out.display()))?;
    let progress = ctx.root.join("t1check.progress.json");
    let _ = fs::remove_file(&progress);
    let c = command(
        search,
        p,
        &q,
        &Run {
            table: &table,
            threads,
            tt_mb: hyb_tt,
            verify_tt_mb: verify_tt,
            out: &out,
            progress: &progress,
        },
    )?;
    let step = json!({ "kind": "hybrid_t1check", "step": "hybrid check (saved worlds)",
                       "partial_cpu_ok": false, "warnings": warnings });
    ctx.set_phase("chunk", step.clone());
    ctx.run_child_io(
        c,
        "hybrid/t1check.log",
        Some(&out.join("t1check.txt")),
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
    let hc = read_json(&out.join("t1check.json"))
        .ok_or("hybrid_t1check: hybrid-check wrote no t1check.json")?;
    let sets: Vec<Value> = hc["sets"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|s| {
            json!({
                "set": s["set"], "burn_rule": s["burn_rule"], "values_version": s["values_version"],
                "rules_robber_burn": s["rules_robber_burn"], "threads": s["threads"],
                "worlds": s["plan"]["worlds"], "decisions": s["plan"]["decisions"],
                "verification": s["verification"], "exact_values": s["exact_values"],
                "pass": s["pass"], "overall": s["groups"].get(0).cloned().unwrap_or(Value::Null),
                "time": s["time"],
            })
        })
        .collect();
    let inputs: Vec<Value> = std::iter::once(("net", p["net"].as_str()))
        .chain(std::iter::once(("decisions", p["decisions"].as_str())))
        .chain(
            p["saved"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|e| ("saved", e["path"].as_str())),
        )
        .filter_map(|(what, path)| {
            let path = path?;
            let (sha, bytes) = sha256_file(Path::new(path)).ok()?;
            Some(json!({ "what": what, "sha256": sha, "bytes": bytes }))
        })
        .collect();
    ctx.set_phase(
        "chunk",
        json!({ "kind": "hybrid_t1check", "step": "done", "partial_cpu_ok": true }),
    );
    Ok(json!({
        "kind": "hybrid_t1check",
        "net_sha256": hc["net"]["sha256"],
        "trick": q.trick, "cut": q.cut,
        "threads": threads,
        "tt_mb": hyb_tt, "tables": hyb_rec,
        "verify_tt_mb": verify_tt, "verify_tables": verify_rec,
        "sets": sets,
        "hard": hc["hard"],
        "both_rules": hc["both_rules"],
        "peak_rss_mb": hc["peak_rss_mb"],
        "inputs": inputs,
        "table": table_rec,
        "warnings": warnings,
        "seconds": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn manifest_params() -> Value {
        json!({ "threads": "all", "net_input": "o3", "decisions_input": "dec",
                "saved": [ { "input": "tables", "run": "m3-t1-tables-1" },
                           { "input": "ab1", "run": "m3-agree-t1-ab1" } ],
                "verify_pct": 2.5, "verify_rule": "rules-v1", "verify_solve_threads": 46,
                "verify_tt_mb": 8192 })
    }

    #[test]
    fn settings_are_checked() {
        let q = parse(&manifest_params()).expect("valid");
        assert_eq!(
            (q.trick, q.cut, q.tt_mb, q.hard.as_str()),
            (1, 3, 64, "0:3")
        );
        assert_eq!(
            q.saved,
            vec![
                ("tables".to_string(), "m3-t1-tables-1".to_string()),
                ("ab1".to_string(), "m3-agree-t1-ab1".to_string())
            ]
        );
        assert_eq!(
            (q.verify_pct, q.verify_solve_threads, q.verify_tt_mb),
            (2.5, 46, 8192)
        );
        let bad = |f: &dyn Fn(&mut Value)| {
            let mut p = manifest_params();
            f(&mut p);
            parse(&p).is_err()
        };
        assert!(bad(&|p| p["typo"] = json!(1)));
        assert!(bad(&|p| p["cut"] = json!(1)));
        assert!(bad(&|p| p["saved"] = json!([])));
        assert!(bad(&|p| p["saved"][0]["extra"] = json!(1)));
        assert!(bad(&|p| p["saved"][0]["run"] = json!("a=b")));
        assert!(bad(&|p| p["hard"] = json!("0-3")));
        assert!(bad(&|p| p["verify_pct"] = json!(101)));
        assert!(bad(&|p| p["verify_rule"] = json!("some")));
        assert!(bad(&|p| {
            p.as_object_mut().expect("object").remove("net_input");
        }));
        assert_eq!(
            input_names(&manifest_params()).expect("names"),
            vec!["o3", "dec", "tables", "ab1"]
        );
    }

    /// The inputs become paths, and the command passes every setting to `search hybrid-check`.
    #[test]
    fn inputs_resolve_into_the_command() {
        let p = manifest_params();
        let mut inputs: BTreeMap<String, String> = BTreeMap::new();
        for (n, path) in [
            ("o3", "/in/aa"),
            ("dec", "/in/bb"),
            ("tables", "/in/cc"),
            ("equity", "/in/dd"),
        ] {
            inputs.insert(n.into(), path.into());
        }
        let e = resolve(&p, &inputs).expect_err("ab1 is missing");
        assert!(e.contains("`ab1`"), "{e}");
        inputs.insert("ab1".into(), "/in/ee".into());
        let r = resolve(&p, &inputs).expect("resolves");
        assert_eq!(parse(&r).expect("still valid"), parse(&p).expect("valid"));
        let q = parse(&r).expect("valid");
        let run = Run {
            table: Path::new("/t/equity.npy"),
            threads: 184,
            tt_mb: 64,
            verify_tt_mb: 4096,
            out: Path::new("/o"),
            progress: Path::new("/p.json"),
        };
        let c = command(Path::new("search"), &r, &q, &run).expect("command");
        let args: Vec<String> = c
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            // Windows joins paths with a backslash; the expected text uses `/`.
            args.join(" ").replace('\\', "/"),
            "hybrid-check --trick 1 --cut 3 --net /in/aa --equity /t/equity.npy --decisions /in/bb \
             --saved m3-t1-tables-1=/in/cc --saved m3-agree-t1-ab1=/in/ee --hard 0:3 --seed 1 \
             --threads 184 --tt-mb 64 --verify-pct 2.5 --verify-rule rules-v1 \
             --verify-solve-threads 46 --verify-tt-mb 4096 --boot 1000 --boot-seed 1 \
             --json /o/t1check.json --progress /p.json"
        );
        // Unresolved settings cannot make a command.
        assert!(command(Path::new("search"), &p, &q, &run).is_err());
    }

    /// The committed manifest (docs/vast/m3-hybrid-t1check.manifest.json) is refused until its
    /// network is filled in; filled in, the runner accepts it, and the inputs it names that the
    /// tests keep a copy of (`crates/search/testdata/t1-saved`) are those very files.
    #[test]
    fn committed_manifest_parses() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let text = fs::read_to_string(root.join("docs/vast/m3-hybrid-t1check.manifest.json"))
            .expect("read");
        let v: Value = serde_json::from_str(&text).expect("json");
        let e = crate::sched::parse_v2(&v).expect_err("placeholder");
        assert!(e.contains("`o3`"), "{e}");
        let filled = text.replace(
            "REPLACE_WITH_THE_O3_INT_SAFETENSORS_SHA256",
            &"a".repeat(64),
        );
        let v: Value = serde_json::from_str(&filled).expect("json");
        let m = crate::sched::parse_v2(&v).expect("runnable");
        let s = &m.streams[0];
        assert_eq!((s.kind.as_str(), s.threads_all), ("hybrid_t1check", true));
        let q = parse(&s.params).expect("settings");
        assert_eq!((q.trick, q.cut, q.hard.as_str()), (1, 3, "0:3"));
        assert_eq!(q.saved.len(), 5);
        assert!(v.get("bin_sha").is_none(), "the launcher sets bin_sha");
        let fixtures = root.join("crates/search/testdata/t1-saved");
        for (input, file) in [
            ("t1-decisions", "decisions-t1.jsonl"),
            ("t1-tables-1", "m3-t1-tables-1.ckpt"),
            ("t1-layouts-records", "m3-t1-leader-layouts-1.jsonl"),
        ] {
            let (sha, _) = sha256_file(&fixtures.join(file)).expect("fixture");
            let want = m
                .inputs
                .iter()
                .find(|(n, _)| n == input)
                .map(|(_, h)| h.clone());
            assert_eq!(want, Some(sha), "{input}");
        }
    }

    /// The chunk with a fake `search`: the report goes to `out/hybrid/t1check.txt`, the JSON is
    /// read back into `result.json`'s summary (each set's verdict and verification, the hard
    /// worlds), and every input's SHA-256 is recorded.
    #[cfg(unix)]
    #[test]
    fn chunk_runs_the_check_and_summarises_it() {
        use std::os::unix::fs::PermissionsExt;
        let base = std::env::temp_dir().join(format!("t1check-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("in")).expect("mkdir");
        let mut inputs: BTreeMap<String, String> = BTreeMap::new();
        for n in ["o3", "dec", "tables", "ab1"] {
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
echo '{{"worlds_done": 3, "worlds_total": 5, "exact_done": 0, "exact_total": 0, "tail": true}}' > "$(arg --progress "$@")"
echo '{{"net": {{"sha256": "abc"}}, "hard": [{{"set": 1, "world": {{"id": 0, "w": 3, "hybrid_us": 5}}}}], "both_rules": [], "peak_rss_mb": 90,
  "sets": [{{"set": 1, "burn_rule": "searched burn", "values_version": 2, "rules_robber_burn": false, "threads": 4,
             "plan": {{"worlds": 31, "decisions": 8}}, "verification": {{"done": false}}, "exact_values": "stored, NOT re-verified",
             "pass": false, "groups": [{{"group": "all"}}], "time": {{}}}}]}}' > "$(arg --json "$@")"
echo "the report"
echo "set 1 d0:w3: H_3 1.00 s" >&2
"#,
                calls = calls.display()
            ),
        )
        .expect("write");
        fs::set_permissions(&search, fs::Permissions::from_mode(0o755)).expect("chmod");
        let p = resolve(&manifest_params(), &inputs).expect("resolves");
        let mut ctx = Ctx::local("t1check-test", &base.join("chunk"), 4);
        let r = chunk_with(&mut ctx, &p, &search, &table, json!({"sha256": "t"})).expect("runs");
        assert_eq!(r["kind"], "hybrid_t1check");
        assert_eq!(r["net_sha256"], "abc");
        assert_eq!(r["sets"][0]["worlds"], 31);
        assert_eq!(r["sets"][0]["overall"]["group"], "all");
        assert_eq!(r["hard"][0]["world"]["w"], 3);
        assert_eq!(r["inputs"].as_array().map(Vec::len), Some(4));
        assert_eq!(r["tt_mb"], 64);
        assert_eq!(
            fs::read_to_string(base.join("chunk/out/hybrid/t1check.txt")).expect("report"),
            "the report\n"
        );
        let call = fs::read_to_string(&calls).expect("calls");
        assert!(call.contains("--saved m3-agree-t1-ab1="), "{call}");
        assert!(
            call.contains("--threads 4 --tt-mb 64 --verify-pct 2.5"),
            "{call}"
        );
        let log = fs::read_to_string(base.join("chunk/logs/hybrid/t1check.log")).expect("log");
        assert!(log.contains("d0:w3"), "{log}");
        let _ = fs::remove_dir_all(&base);
    }
}
