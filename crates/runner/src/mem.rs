//! The memory budget for the exact solver's transposition tables [its memory of positions
//! already solved] (run-ops fix after run m3-agree-t1-a3 was killed by the kernel for lack of
//! memory on a 251 GB host: 4 worlds at once × a 45,000 MB table each, plus the growth copies).
//!
//! The table size (`tt_mb`) changes only the speed of a solve, never a value (tested in
//! `crates/solver`), so the runner may choose or lower it freely; it never raises an explicit one.
//!
//! - **Limit**: the container's memory limit (cgroup v2 `memory.max`, else cgroup v1
//!   `memory.limit_in_bytes`), capped by `MemTotal` from `/proc/meminfo` (a missing or
//!   unlimited cgroup leaves `MemTotal`). vast.ai gives each instance RAM in proportion to its
//!   GPU share and may not enforce it as a hard limit (docs.vast.ai, Docker environment); the
//!   launcher rents whole machines (`gpu_frac=1`), so this is the whole machine's RAM.
//! - **Budget**: the limit, minus a safety margin (a tenth of it, at least 2 GiB: the host's own
//!   use, allocator overhead), minus the other memory the solving process holds (the equity
//!   table's size, [`PROCESS_BASE`], and a full stack per thread).
//! - **Tables**: how many tables live at once (one per world solved at the same time; each kept
//!   between worlds) times the peak of one table ([`solver::table_memory`],
//!   [`solver::shared_table_memory`]). The peak is the table's cap: the solver reserves the
//!   whole cap when it makes a table (pages are touched only as the table grows into them) and
//!   never reallocates, so no block but the final one ever exists. (Before that change, growth
//!   reallocated at each doubling and the old blocks could stay resident, so the peak was
//!   between about 2 and 3 times the cap; the `peak_factor` in the record now reads 1.)
//! - **Choice**: `tt_mb` absent or `"auto"` takes the largest size whose tables fit the budget.
//!   An explicit size that does not fit is lowered to that size, with a warning in the heartbeat,
//!   in `result.json` and on stderr; one that fits is used as given.
//!
//! The budget assumes the chunk has the machine's memory to itself, as the whole-machine
//! measurements that use it (`search_measure`, `solver_spike`) do.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// Memory a solving process holds besides its tables and the equity table: code, decisions,
/// world generators, results (measured in the low hundreds of MB; 1 GiB allowed).
pub(crate) const PROCESS_BASE: u64 = GIB;

/// Stack memory allowed per thread (Rust's default thread stack, all of it counted).
pub(crate) const STACK_PER_THREAD: u64 = 2 * MIB;

/// Where the memory files are read from: `/` on a machine, a directory of fake files in tests.
pub(crate) struct Sys {
    root: PathBuf,
}

impl Sys {
    pub(crate) fn host() -> Sys {
        Sys {
            root: PathBuf::from("/"),
        }
    }

    #[cfg(test)]
    pub(crate) fn at(root: &Path) -> Sys {
        Sys {
            root: root.to_path_buf(),
        }
    }

    fn read(&self, rel: &str) -> Option<String> {
        fs::read_to_string(self.root.join(rel))
            .ok()
            .map(|s| s.trim().to_string())
    }

    /// A `/proc/meminfo` field in bytes.
    fn meminfo(&self, field: &str) -> Option<u64> {
        let text = self.read("proc/meminfo")?;
        let line = text.lines().find(|l| {
            l.strip_prefix(field)
                .is_some_and(|rest| rest.starts_with(':'))
        })?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
    }

    /// The container's memory limit.
    pub(crate) fn limit(&self) -> Limit {
        let v2 = self.read("sys/fs/cgroup/memory.max");
        let v1 = self.read("sys/fs/cgroup/memory/memory.limit_in_bytes");
        // "max" (v2) means no limit; v1 shows no limit as a huge number, which MemTotal caps.
        let (cgroup, cgroup_source) = match (&v2, &v1) {
            (Some(s), _) => (s.parse::<u64>().ok(), "cgroup v2 memory.max"),
            (None, Some(s)) => (s.parse::<u64>().ok(), "cgroup v1 memory.limit_in_bytes"),
            (None, None) => (None, "none"),
        };
        let mem_total = self.meminfo("MemTotal");
        let (bytes, source) = match (cgroup, mem_total) {
            (Some(c), Some(t)) if c < t => (c, cgroup_source),
            (_, Some(t)) => (t, "MemTotal"),
            (Some(c), None) => (c, cgroup_source),
            // Nothing readable (not Linux): assume a small machine.
            (None, None) => (16 * GIB, "default (16 GiB)"),
        };
        Limit {
            bytes,
            source,
            cgroup,
            mem_total,
            mem_available: self.meminfo("MemAvailable"),
        }
    }
}

/// The memory this container may use.
#[derive(Clone, Debug)]
pub(crate) struct Limit {
    pub bytes: u64,
    /// Which file set it.
    pub source: &'static str,
    pub cgroup: Option<u64>,
    pub mem_total: Option<u64>,
    /// Free and reclaimable memory at the time it was read (diagnostic only).
    pub mem_available: Option<u64>,
}

impl Limit {
    pub(crate) fn mb(&self) -> u64 {
        self.bytes / MIB
    }

    /// The safety margin: a tenth of the limit, at least 2 GiB (but never more than half).
    fn margin(&self) -> u64 {
        (self.bytes / 10).max(2 * GIB).min(self.bytes / 2)
    }
}

/// The tables a solving process keeps at once.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Tables {
    /// `count` sequential solvers, each with its own table (`solve_threads` = 1).
    Own { count: u64 },
    /// `count` parallel solves at once, each with one table shared by `solve_threads` threads.
    Shared { count: u64, solve_threads: u64 },
}

impl Tables {
    fn count(self) -> u64 {
        match self {
            Tables::Own { count } | Tables::Shared { count, .. } => count.max(1),
        }
    }

    /// Solver threads of the process.
    pub(crate) fn threads(self) -> u64 {
        match self {
            Tables::Own { count } => count.max(1),
            Tables::Shared {
                count,
                solve_threads,
            } => count.max(1) * solve_threads.max(1),
        }
    }

    fn one(self, mb: u64) -> solver::TableMemory {
        match self {
            Tables::Own { .. } => solver::table_memory((mb * MIB) as usize),
            Tables::Shared { solve_threads, .. } => {
                solver::shared_table_memory(mb as usize, solve_threads as usize)
            }
        }
    }

    /// Peak bytes of all the tables at `mb` megabytes each.
    fn peak(self, mb: u64) -> u64 {
        self.count().saturating_mul(self.one(mb).peak as u64)
    }
}

/// The manifest's `tt_mb`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Requested {
    Auto,
    Mb(u64),
}

impl Requested {
    /// `params[key]`: absent, null or `"auto"` is [`Requested::Auto`]; a whole number of MB.
    pub(crate) fn from_param(p: &Value, key: &str) -> Result<Requested, String> {
        match p.get(key) {
            None | Some(Value::Null) => Ok(Requested::Auto),
            Some(Value::String(s)) if s == "auto" => Ok(Requested::Auto),
            Some(v) => match v.as_u64() {
                Some(n) if n > 0 => Ok(Requested::Mb(n)),
                _ => Err(format!(
                    "params.{key} must be a whole number of MB (at least 1) or \"auto\""
                )),
            },
        }
    }
}

/// The chosen table size and how it was chosen.
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    pub tt_mb: u64,
    pub warning: Option<String>,
    /// For `result.json` and the heartbeat.
    pub record: Value,
}

/// Chooses `tt_mb` for `tables` within `limit`, the process also holding `other` bytes.
pub(crate) fn plan(requested: Requested, tables: Tables, other: u64, limit: &Limit) -> Plan {
    let margin = limit.margin();
    let (budget, _) = budget(other, limit);
    let fits = |mb: u64| tables.peak(mb) <= budget;
    // The largest size that fits, up to `top` (the peak never falls as the size grows, so a
    // binary search finds it; the result is checked to fit either way).
    let largest = |top: u64| -> Option<u64> {
        if !fits(1) {
            return None;
        }
        let (mut lo, mut hi) = (1u64, top.max(1));
        if fits(hi) {
            return Some(hi);
        }
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if fits(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(lo)
    };
    let mb_of = |b: u64| b / MIB;
    let why = |mb: u64| {
        format!(
            "{} table(s) of {mb} MB peak at {} MB ({:.2} x their size), over the {} MB budget = limit {} MB ({}) - margin {} MB - other {} MB",
            tables.count(),
            mb_of(tables.peak(mb)),
            tables.one(mb).peak_factor(),
            mb_of(budget),
            limit.mb(),
            limit.source,
            mb_of(margin),
            mb_of(other)
        )
    };
    // An explicit size is only ever lowered, so its search stops at it.
    let top = match requested {
        Requested::Mb(m) => m,
        Requested::Auto => limit.mb(),
    };
    let (tt_mb, warning) = match (requested, largest(top)) {
        (Requested::Mb(m), Some(safe)) if safe < m => (
            safe,
            Some(format!(
                "tt_mb {m} lowered to {safe}: {}; the table size changes speed only, never a value",
                why(m)
            )),
        ),
        (_, Some(safe)) => (safe, None),
        (_, None) => (
            1,
            Some(format!(
                "even 1 MB tables do not fit: {}; running with 1 MB tables, slowly",
                why(1)
            )),
        ),
    };
    let one = tables.one(tt_mb);
    let (kind, count, solve_threads) = match tables {
        Tables::Own { count } => ("own", count, 1),
        Tables::Shared {
            count,
            solve_threads,
        } => ("shared", count, solve_threads),
    };
    let record = json!({
        "tt_mb": tt_mb,
        "requested": match requested { Requested::Auto => json!("auto"), Requested::Mb(m) => json!(m) },
        "clamped": matches!(requested, Requested::Mb(m) if m != tt_mb),
        "tables": count, "table_kind": kind, "solve_threads": solve_threads,
        "table_cap_mb": mb_of(one.cap as u64),
        "table_peak_mb": mb_of(one.peak as u64),
        "peak_factor": (one.peak_factor() * 1000.0).round() / 1000.0,
        "tables_peak_mb": mb_of(tables.peak(tt_mb)),
        "budget_mb": mb_of(budget),
        "limit_mb": limit.mb(), "limit_source": limit.source,
        "cgroup_limit_mb": limit.cgroup.map(mb_of),
        "mem_total_mb": limit.mem_total.map(mb_of),
        "mem_available_mb": limit.mem_available.map(mb_of),
        "margin_mb": mb_of(margin),
        "other_mb": mb_of(other),
        "warning": warning,
    });
    Plan {
        tt_mb,
        warning,
        record,
    }
}

/// The memory budget for all tables of a process (bytes): the limit minus the safety margin
/// minus the process's `other` memory, as [`plan`] computes it; with its record. The
/// position-aware schedule shares it out per thread (`search::plan`).
pub(crate) fn budget(other: u64, limit: &Limit) -> (u64, Value) {
    let margin = limit.margin();
    let budget = limit.bytes.saturating_sub(margin).saturating_sub(other);
    let mb_of = |b: u64| b / MIB;
    (
        budget,
        json!({
            "tt_mb": "per world (position-aware)",
            "budget_mb": mb_of(budget),
            "limit_mb": limit.mb(), "limit_source": limit.source,
            "cgroup_limit_mb": limit.cgroup.map(mb_of),
            "mem_total_mb": limit.mem_total.map(mb_of),
            "mem_available_mb": limit.mem_available.map(mb_of),
            "margin_mb": mb_of(margin),
            "other_mb": mb_of(other),
        }),
    )
}

/// [`budget`] on this machine for a process that loads `equity` and runs `threads` threads, as a
/// [`Plan`] (`tt_mb` 0: each world's table is chosen by the schedule) and the budget in bytes.
pub(crate) fn budget_here(equity: &Path, threads: u64, what: &str) -> (Plan, u64) {
    let limit = Sys::host().limit();
    let (b, record) = budget(other_bytes(equity, threads), &limit);
    println!(
        "runner: {what}: position-aware tables within {} MB of {} MB ({})",
        b / MIB,
        limit.mb(),
        limit.source
    );
    (
        Plan {
            tt_mb: 0,
            warning: None,
            record,
        },
        b,
    )
}

/// The other memory of a process that loads the equity table at `equity` and runs `threads`
/// solver threads.
pub(crate) fn other_bytes(equity: &Path, threads: u64) -> u64 {
    let table = fs::metadata(equity).map(|m| m.len()).unwrap_or(128 * MIB);
    table + PROCESS_BASE + threads * STACK_PER_THREAD
}

/// The peak memory (bytes) one format-2 chunk will hold, for the orchestrator's check that the
/// chunks it runs at the same time fit in memory together (`v2.rs`; review E, 2026-10-16). Each
/// chunk sizes its tables as if it had the machine to itself (this module's plan), so without that
/// check three `search_measure` chunks of 32 one-thread worlds at 12, 3 and 1 GB tables would all
/// start at once on a 500 GB machine and need 512 GB.
/// - `search_measure` with explicit numbers (`worlds_at_once`/`solve_threads`, a numeric `tt_mb`):
///   its tables at the size its own plan will choose on this machine, plus its other memory;
/// - `search_measure` with `"auto"` anywhere, `solver_spike` and `t1bench`: the whole budget
///   (they size their tables to the machine), so they never run beside another such chunk; and
///   `hybrid_rung` (the trainer's arrays of millions of label rows; the label shards themselves
///   are streamed to disk) and `hybrid_t1check` (a table per thread);
/// - every other kind: 0 (small, as before).
pub(crate) fn chunk_need(
    kind: &str,
    params: &Value,
    threads: u64,
    equity_bytes: u64,
    limit: &Limit,
) -> u64 {
    let whole = budget(0, limit).0;
    match kind {
        "search_measure" => {
            let fixed = crate::measure::spread(params, threads);
            let requested = Requested::from_param(params, "tt_mb");
            match (fixed, requested) {
                (
                    Ok(crate::measure::Spread::Fixed {
                        workers,
                        solve_threads,
                    }),
                    Ok(req @ Requested::Mb(_)),
                ) => {
                    let tables = if solve_threads > 1 {
                        Tables::Shared {
                            count: workers,
                            solve_threads,
                        }
                    } else {
                        Tables::Own { count: workers }
                    };
                    let other = equity_bytes + PROCESS_BASE + tables.threads() * STACK_PER_THREAD;
                    let p = plan(req, tables, other, limit);
                    (tables.peak(p.tt_mb) + other).min(whole)
                }
                _ => whole,
            }
        }
        "solver_spike" | "t1bench" | "hybrid_rung" | "hybrid_t1check" => whole,
        _ => 0,
    }
}

/// Plans `tt_mb` on this machine and reports a warning on stderr.
pub(crate) fn plan_here(requested: Requested, tables: Tables, equity: &Path, what: &str) -> Plan {
    let limit = Sys::host().limit();
    let plan = plan(
        requested,
        tables,
        other_bytes(equity, tables.threads()),
        &limit,
    );
    match &plan.warning {
        Some(w) => eprintln!("runner: WARNING: {what}: {w}"),
        None => println!(
            "runner: {what}: tt_mb {} ({} tables, budget {} MB of {} MB from {})",
            plan.tt_mb,
            tables.count(),
            plan.record["budget_mb"],
            limit.mb(),
            limit.source
        ),
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake `/` holding the given files.
    fn fake(name: &str, files: &[(&str, &str)]) -> PathBuf {
        let d = std::env::temp_dir().join(format!("runner-mem-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        for (rel, text) in files {
            let p = d.join(rel);
            fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
            fs::write(&p, text).expect("write");
        }
        d
    }

    fn meminfo(total_kb: u64) -> String {
        format!(
            "MemTotal:       {total_kb} kB\nMemFree:        1000 kB\nMemAvailable:   {} kB\n",
            total_kb / 2
        )
    }

    const HOST_KB: u64 = 263_000_000; // a 251 GiB host

    #[test]
    fn limit_from_cgroup_files() {
        let mi = meminfo(HOST_KB);
        // cgroup v2 with a limit below RAM.
        let d = fake(
            "v2",
            &[
                ("sys/fs/cgroup/memory.max", "107374182400\n"),
                ("proc/meminfo", &mi),
            ],
        );
        let l = Sys::at(&d).limit();
        assert_eq!((l.bytes, l.source), (100 * GIB, "cgroup v2 memory.max"));
        assert_eq!(l.mem_total, Some(HOST_KB * 1024));
        assert_eq!(l.mem_available, Some(HOST_KB / 2 * 1024));
        // cgroup v2 unlimited: MemTotal.
        let d = fake(
            "v2max",
            &[("sys/fs/cgroup/memory.max", "max\n"), ("proc/meminfo", &mi)],
        );
        let l = Sys::at(&d).limit();
        assert_eq!(
            (l.bytes, l.source, l.cgroup),
            (HOST_KB * 1024, "MemTotal", None)
        );
        // cgroup v1: a limit, and the "unlimited" huge number.
        let d = fake(
            "v1",
            &[
                ("sys/fs/cgroup/memory/memory.limit_in_bytes", "34359738368"),
                ("proc/meminfo", &mi),
            ],
        );
        let l = Sys::at(&d).limit();
        assert_eq!(
            (l.bytes, l.source),
            (32 * GIB, "cgroup v1 memory.limit_in_bytes")
        );
        let d = fake(
            "v1max",
            &[
                (
                    "sys/fs/cgroup/memory/memory.limit_in_bytes",
                    "9223372036854771712",
                ),
                ("proc/meminfo", &mi),
            ],
        );
        assert_eq!(Sys::at(&d).limit().source, "MemTotal");
        // A cgroup above RAM: RAM.
        let d = fake(
            "v2big",
            &[
                ("sys/fs/cgroup/memory.max", "999999999999999"),
                ("proc/meminfo", &mi),
            ],
        );
        assert_eq!(Sys::at(&d).limit().bytes, HOST_KB * 1024);
        // No meminfo: the cgroup alone; nothing at all: the default.
        let d = fake("cgonly", &[("sys/fs/cgroup/memory.max", "8589934592")]);
        assert_eq!(Sys::at(&d).limit().bytes, 8 * GIB);
        let d = fake("nothing", &[]);
        assert_eq!(Sys::at(&d).limit().bytes, 16 * GIB);
    }

    fn limit(bytes: u64) -> Limit {
        Limit {
            bytes,
            source: "test",
            cgroup: None,
            mem_total: Some(bytes),
            mem_available: None,
        }
    }

    /// The run that was killed: 4 trick-1 worlds at once, 96 threads each, 45,000 MB tables on
    /// a 251 GiB host. At the time, growth copies made its tables reach far more than the host
    /// has; now that a table never holds more than its cap, that setting fits (180,000 MB
    /// within the 224 GiB budget), and a request of 70,000 MB tables (280,000 MB) is what the
    /// plan lowers to a size whose peak fits, saying so.
    #[test]
    fn the_killed_run_is_clamped_with_a_warning() {
        let host = limit(HOST_KB * 1024);
        let tables = Tables::Shared {
            count: 4,
            solve_threads: 96,
        };
        let other = 76 * MIB + PROCESS_BASE + tables.threads() * STACK_PER_THREAD;
        let budget = host.bytes - host.margin() - other;
        assert_eq!(tables.peak(45_000), tables.one(45_000).cap as u64 * 4);
        assert!(tables.peak(45_000) <= budget, "the old setting now fits");
        assert!(tables.peak(70_000) > host.bytes, "this setting cannot fit");
        let p = plan(Requested::Mb(70_000), tables, other, &host);
        assert!(p.tt_mb < 70_000 && p.tt_mb > 45_000, "{}", p.tt_mb);
        assert!(tables.peak(p.tt_mb) <= budget);
        assert!(tables.peak(p.tt_mb + 1) > budget, "the largest that fits");
        let w = p.warning.as_deref().expect("a warning");
        assert!(w.contains("tt_mb 70000 lowered to"), "{w}");
        assert_eq!(p.record["clamped"], json!(true));
        assert_eq!(p.record["warning"], json!(w));
        assert_eq!(p.record["tables"], json!(4));
        // Auto gives the same size, silently.
        let a = plan(Requested::Auto, tables, other, &host);
        assert_eq!((a.tt_mb, a.warning.is_none()), (p.tt_mb, true));
        assert_eq!(a.record["requested"], json!("auto"));
        // A size that fits is kept as given.
        let k = plan(Requested::Mb(4096), tables, other, &host);
        assert_eq!((k.tt_mb, k.warning), (4096, None));
        assert_eq!(k.record["clamped"], json!(false));
    }

    #[test]
    fn budget_arithmetic() {
        // 64 GiB, 8 own tables: margin 6.4 GiB, other 1 GiB + stacks.
        let l = limit(64 * GIB);
        assert_eq!(l.margin(), 64 * GIB / 10);
        let t = Tables::Own { count: 8 };
        let other = PROCESS_BASE + 8 * STACK_PER_THREAD;
        let p = plan(Requested::Auto, t, other, &l);
        let budget = 64 * GIB - 64 * GIB / 10 - other;
        assert_eq!(p.record["budget_mb"], json!(budget / MIB));
        assert!(t.peak(p.tt_mb) <= budget && t.peak(p.tt_mb + 1) > budget);
        // The peak per table is its size: the tables use the whole budget, slot for slot.
        let f = p.record["peak_factor"].as_f64().expect("factor");
        assert_eq!(f, 1.0);
        assert_eq!(p.record["table_peak_mb"], p.record["table_cap_mb"]);
        assert!(p.tt_mb * 8 * MIB <= budget, "never more than the budget");
        assert!((p.tt_mb + 1) * 8 * MIB > budget, "and all of it");
        // Small machines: the margin is at least 2 GiB, at most half the limit.
        assert_eq!(limit(8 * GIB).margin(), 2 * GIB);
        assert_eq!(limit(GIB).margin(), GIB / 2);
        // Too little memory even for 1 MB tables: 1 MB with a warning.
        let tiny = plan(Requested::Auto, Tables::Own { count: 4096 }, 0, &limit(GIB));
        assert_eq!(tiny.tt_mb, 1);
        assert!(tiny.warning.expect("warning").contains("even 1 MB"));
        // More tables, smaller tables.
        let a = plan(Requested::Auto, Tables::Own { count: 4 }, other, &l).tt_mb;
        let b = plan(Requested::Auto, Tables::Own { count: 16 }, other, &l).tt_mb;
        assert!(a > b);
    }

    #[test]
    fn requested_parses() {
        assert_eq!(
            Requested::from_param(&json!({}), "tt_mb"),
            Ok(Requested::Auto)
        );
        assert_eq!(
            Requested::from_param(&json!({ "tt_mb": "auto" }), "tt_mb"),
            Ok(Requested::Auto)
        );
        assert_eq!(
            Requested::from_param(&json!({ "tt_mb": 512 }), "tt_mb"),
            Ok(Requested::Mb(512))
        );
        assert!(Requested::from_param(&json!({ "tt_mb": 0 }), "tt_mb").is_err());
        assert!(Requested::from_param(&json!({ "tt_mb": "big" }), "tt_mb").is_err());
        assert!(Requested::from_param(&json!({ "tt_mb": 1.5 }), "tt_mb").is_err());
    }
}
