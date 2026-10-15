//! Machine-cost records (owner request, 2026-10-16): what every rented machine really costs per
//! solver node, measured on the run itself, so future rentals pick the cheapest-fastest machines
//! from data rather than estimates (docs/VAST-MACHINES.md, "Measured on our runs").
//!
//! The [`Tally`] follows the heartbeat's `progress` and counts the solver nodes and worlds that
//! this machine finished: a `search_measure` chunk reports `nodes_valued` and `worlds_valued`
//! (from `search worlds`' progress file; restored worlds are not counted), a `t1bench` window
//! reports `nodes` and `finished_worlds`. Each step's live count is folded into the total when
//! the step changes (the step is identified by phase, kind, step and window). Every heartbeat
//! then carries `progress.nodes_total`, `worlds_finished`, `usd_so_far` (elapsed × the machine's
//! price, `DPH`), `usd_per_1e12_nodes` (null until a node was counted) and
//! `nodes_per_s_per_thread` (over the time spent in node-producing steps, per worker thread).
//!
//! Every 10 minutes and at the end of the run, the record ([`record`]) is written to
//! `machines/<run_id>.json` under the run root and to the remote store (the bucket's own
//! `machines/<run_id>.json`, mutable, plus an immutable copy under
//! `machines/history/<run_id>/<utc>.json`), on the background network lane (`net.rs`): the
//! network never blocks the runner. The final record carries `state: "final"` and the outcome;
//! `python/launcher/cleanup.py` finalises the record of a handed-over run whose runner could not,
//! and `python/tools/machines.py` reads them (`--live`, `--by-cpu`, `--rank`).
//!
//! Early figures read high: the work queue runs the hardest deals first, so the first worlds
//! cost the most per node; the record says `settling: true` while fewer than 10% of the worlds
//! to do are finished.

use std::time::{Duration, Instant};

use serde_json::{Value, json};

/// How often the running record is written.
pub const WRITE_EVERY: Duration = Duration::from_secs(600);

/// Below this share of worlds finished, the figures are marked `settling`.
const SETTLED_SHARE: f64 = 0.10;

/// Running totals of this machine's solver work, from the heartbeat's `progress`.
#[derive(Debug, Default)]
pub struct Tally {
    /// Nodes and worlds of the steps already over.
    done_nodes: u64,
    done_worlds: u64,
    /// The step under way: its identity and its live counts (cumulative within the step).
    live_key: String,
    live_nodes: u64,
    live_worlds: u64,
    /// Worlds to do on this machine in the current (or last) worlds step (total minus resumed).
    worlds_total: Option<u64>,
    /// Seconds spent in node-producing steps, and the last moment nodes were seen in this step.
    solve_s: f64,
    last_seen: Option<Instant>,
    /// What the last steps said about themselves.
    schedule: Option<String>,
    tt_mb: Option<u64>,
    tt_budget_mb: Option<u64>,
    trick: Option<u64>,
    chunk: Option<String>,
    /// Whether any node count was ever seen (the heartbeat fields appear from then on).
    seen: bool,
    /// When the running record was last written.
    last_write: Option<Instant>,
    /// From `hardware.json`, kept for the record.
    cpu_model: Option<String>,
    ram_gb: Option<f64>,
    usable_cores: Option<f64>,
}

impl Tally {
    /// Keeps the hardware facts the record needs.
    pub fn hardware(&mut self, hw: &Value) {
        self.cpu_model = hw["cpu_model"].as_str().map(str::to_string);
        self.ram_gb = hw["mem_total_kb"]
            .as_u64()
            .map(|kb| (kb as f64 / 1_048_576.0 * 10.0).round() / 10.0);
        self.usable_cores = hw["usable_cores"].as_f64();
    }

    /// Reads a heartbeat's phase and progress.
    pub fn observe(&mut self, phase: &str, progress: &Value) {
        let key = format!(
            "{phase}|{}|{}|{}",
            progress["kind"], progress["step"], progress["window"]
        );
        if key != self.live_key {
            self.done_nodes += self.live_nodes;
            self.done_worlds += self.live_worlds;
            self.live_nodes = 0;
            self.live_worlds = 0;
            self.live_key = key;
            self.last_seen = None;
        }
        if let Some(s) = progress["kind"].as_str() {
            self.chunk = Some(s.to_string());
        }
        let nodes = progress["nodes_valued"]
            .as_u64()
            .or_else(|| progress["nodes"].as_u64());
        let Some(n) = nodes else {
            return;
        };
        self.seen = true;
        self.live_nodes = n;
        let now = Instant::now();
        if let Some(t) = self.last_seen {
            self.solve_s += now.duration_since(t).as_secs_f64();
        }
        self.last_seen = Some(now);
        if let Some(w) = progress["worlds_valued"]
            .as_u64()
            .or_else(|| progress["finished_worlds"].as_u64())
        {
            self.live_worlds = w;
        }
        if let Some(t) = progress["worlds_total"].as_u64() {
            self.worlds_total =
                Some(t.saturating_sub(progress["worlds_resumed"].as_u64().unwrap_or(0)));
        }
        if let Some(s) = progress["schedule"]
            .as_str()
            .or_else(|| progress["window"].as_str())
        {
            self.schedule = Some(s.to_string());
        }
        if let Some(t) = progress["tt_mb"].as_u64() {
            self.tt_mb = Some(t);
        }
        if let Some(t) = progress["tt_budget_mb"].as_u64() {
            self.tt_budget_mb = Some(t);
        }
        if let Some(t) = progress["trick"].as_u64() {
            self.trick = Some(t);
        }
    }

    /// Whether a node count was ever seen.
    pub fn active(&self) -> bool {
        self.seen
    }

    pub fn nodes_total(&self) -> u64 {
        self.done_nodes + self.live_nodes
    }

    pub fn worlds_finished(&self) -> u64 {
        self.done_worlds + self.live_worlds
    }

    /// Dollars per 10^12 nodes at `usd` spent, if any node was counted.
    pub fn usd_per_1e12(&self, usd: f64) -> Option<f64> {
        let n = self.nodes_total();
        (n > 0).then(|| (usd / n as f64 * 1e12 * 1000.0).round() / 1000.0)
    }

    /// Nodes per second per worker thread over the node-producing steps.
    pub fn nodes_per_s_per_thread(&self, threads: usize) -> Option<f64> {
        (self.solve_s > 0.0 && threads > 0)
            .then(|| (self.nodes_total() as f64 / self.solve_s / threads as f64).round())
    }

    /// Fewer than 10% of the worlds to do are finished (the hard deals run first, so the cost per
    /// node still reads high). Without a known total (a bench's fixed-time windows) never.
    pub fn settling(&self) -> bool {
        match self.worlds_total {
            Some(t) if t > 0 => (self.worlds_finished() as f64) < SETTLED_SHARE * t as f64,
            _ => false,
        }
    }

    /// The heartbeat's cost fields (merged into `progress`).
    pub fn fields(
        &self,
        elapsed_s: f64,
        dph: f64,
        threads: usize,
    ) -> serde_json::Map<String, Value> {
        let usd = usd(elapsed_s, dph);
        let v = json!({
            "nodes_total": self.nodes_total(),
            "worlds_finished": self.worlds_finished(),
            "worlds_to_do": self.worlds_total,
            "usd_so_far": usd,
            "usd_per_1e12_nodes": self.usd_per_1e12(usd),
            "nodes_per_s_per_thread": self.nodes_per_s_per_thread(threads),
            "settling": self.settling(),
        });
        match v {
            Value::Object(m) => m,
            _ => serde_json::Map::new(),
        }
    }

    /// Whether the running record is due (never written, or `WRITE_EVERY` ago).
    pub fn write_due(&self) -> bool {
        self.seen && self.last_write.is_none_or(|t| t.elapsed() >= WRITE_EVERY)
    }

    pub fn written(&mut self) {
        self.last_write = Some(Instant::now());
    }
}

/// Dollars spent after `elapsed_s` at `dph` dollars per hour.
pub fn usd(elapsed_s: f64, dph: f64) -> f64 {
    (elapsed_s / 3600.0 * dph * 10_000.0).round() / 10_000.0
}

/// What the record needs from the run besides the tally.
pub struct RunFacts<'a> {
    pub run_id: &'a str,
    pub utc: String,
    pub elapsed_s: f64,
    pub threads: usize,
    pub dph: f64,
}

/// The machine record (`machines/<run_id>.json`): `state` is `running` or `final`; `outcome`
/// (`DONE`, `FAILED`, `STOPPED`) is given for a final record only.
pub fn record(t: &Tally, f: &RunFacts, state: &str, outcome: Option<&str>) -> Value {
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let usd = usd(f.elapsed_s, f.dph);
    json!({
        "format": "41bots-machine-record",
        "version": 1,
        "run_id": f.run_id,
        "state": state,
        "outcome": outcome,
        "utc": f.utc,
        "cpu_model": t.cpu_model,
        "threads": f.threads,
        "usable_cores": t.usable_cores,
        "ram_gb": t.ram_gb,
        "dph": f.dph,
        "minutes": (f.elapsed_s / 60.0 * 10.0).round() / 10.0,
        "usd": usd,
        "nodes_total": t.nodes_total(),
        "nodes_basis": "measured",
        "worlds_finished": t.worlds_finished(),
        "worlds_total": t.worlds_total,
        "usd_per_1e12_nodes": t.usd_per_1e12(usd),
        "nodes_per_s_per_thread": t.nodes_per_s_per_thread(f.threads),
        "settling": t.settling(),
        "schedule": t.schedule,
        "tt_mb": t.tt_mb,
        "tt_budget_mb": t.tt_budget_mb,
        "trick": t.trick,
        "chunk": t.chunk,
        "bin_sha": env("BIN_SHA"),
        "image": env("IMAGE"),
        "offer_id": env("OFFER_ID"),
        "host_id": env("HOST_ID"),
        "machine_id": env("MACHINE_ID"),
        "instance_id": env("CONTAINER_ID"),
        "machine_label": env("CONTAINER_ID").map(|_| format!("41bots-{}", f.run_id)),
    })
}

/// The remote keys of a record: the mutable one and its immutable history copy.
pub fn keys(run_id: &str, utc: &str) -> [(String, bool); 2] {
    [
        (format!("machines/{run_id}.json"), false),
        (format!("machines/history/{run_id}/{utc}.json"), true),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tally_folds_steps_and_counts_only_this_machine() {
        let mut t = Tally::default();
        assert!(!t.active());
        // A search_measure step: nodes of the worlds valued here, restored worlds left out.
        let p = |nodes: u64, valued: u64| {
            json!({ "kind": "search_measure", "step": "exact worlds", "nodes_valued": nodes,
                    "worlds_valued": valued, "worlds_total": 100, "worlds_resumed": 20,
                    "schedule": "plan", "tt_mb": 448, "trick": 1 })
        };
        t.observe(
            "chunk",
            &json!({ "kind": "search_measure", "step": "decisions" }),
        );
        assert!(!t.active());
        t.observe("chunk", &p(1_000, 3));
        t.observe("chunk", &p(5_000, 7));
        assert!(t.active());
        assert_eq!(t.nodes_total(), 5_000);
        assert_eq!(t.worlds_finished(), 7);
        assert!(t.settling(), "7 of 80 to do");
        t.observe("chunk", &p(9_000, 9));
        assert!(!t.settling(), "9 of 80 to do is over 10%");
        // The next step starts its own count; the first one is kept.
        t.observe(
            "chunk",
            &json!({ "kind": "search_measure", "step": "method 0 (rollout) worlds",
                     "nodes_valued": 0, "worlds_valued": 2, "worlds_total": 50 }),
        );
        assert_eq!(t.nodes_total(), 9_000);
        assert_eq!(t.worlds_finished(), 11);
        // The end: progress is gone, the totals stay.
        t.observe("uploading", &Value::Null);
        assert_eq!(t.nodes_total(), 9_000);
        assert_eq!(t.usd_per_1e12(0.009), Some(1_000_000.0));
        assert_eq!(t.schedule.as_deref(), Some("plan"));
        assert_eq!(t.trick, Some(1));
        let r = record(
            &t,
            &RunFacts {
                run_id: "r1",
                utc: "2026-10-16T00:00:00Z".into(),
                elapsed_s: 3600.0,
                threads: 4,
                dph: 1.5,
            },
            "final",
            Some("DONE"),
        );
        assert_eq!(r["usd"], 1.5);
        assert_eq!(r["nodes_total"], 9_000);
        assert_eq!(r["worlds_finished"], 11);
        assert_eq!(r["state"], "final");
        assert_eq!(r["outcome"], "DONE");
        assert_eq!(
            keys("r1", "2026-10-16T00:00:00Z")[1].0,
            "machines/history/r1/2026-10-16T00:00:00Z.json"
        );
    }

    #[test]
    fn tally_reads_t1bench_windows() {
        let mut t = Tally::default();
        let w = |window: &str, nodes: u64, done: u64| {
            json!({ "kind": "t1bench", "window": window, "started": true, "nodes": nodes,
                    "finished_worlds": done, "threads": 8 })
        };
        t.observe("chunk", &w("fill", 100, 1));
        t.observe("chunk", &w("fill", 300, 4));
        t.observe("chunk", &w("prod", 50, 0));
        assert_eq!(t.nodes_total(), 350);
        assert_eq!(t.worlds_finished(), 4);
        assert_eq!(t.schedule.as_deref(), Some("prod"));
        assert!(!t.settling(), "a bench has no world total: never settling");
        let f = t.fields(60.0, 3.6, 8);
        assert_eq!(f["usd_so_far"], 0.06);
        assert!(f["usd_per_1e12_nodes"].as_f64().expect("cost") > 0.0);
        let mut none = Tally::default();
        assert_eq!(none.fields(60.0, 3.6, 8)["usd_per_1e12_nodes"], Value::Null);
        assert!(!none.write_due());
        none.observe("chunk", &w("fill", 1, 0));
        assert!(none.write_due());
        none.written();
        assert!(!none.write_due());
    }
}
