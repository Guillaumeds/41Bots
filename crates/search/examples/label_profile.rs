//! Profile of the exact label solves (docs/ENGINEERING-2026-10-17.md item A): the solver's own
//! counters and timers over the label solves of one source, on one thread.
//!
//! ```text
//! cargo run --release -p search --example label_profile -- --equity TABLE --trick K
//!     --source random|rules|harvest|race [--count N] [--seed S] [--net INT.safetensors]
//!     [--start-hands H]
//! ```
//! `random` / `rules`: rows `0..N` of the label job, each solved as `search labels` solves it.
//! `harvest` / `race`: harvest decisions `0..N` (exact driver, or the network's with `--net`),
//! then the label solves of each decision again: with the exact driver every reached position,
//! with a network the sampled ones (as `search labels` labels them).

use std::cell::Cell;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use equity::{MatchState, MoneyChance};
use search::BeliefSource;
use search::measure::collect_starts_with;
use search::oracle_labels::{
    Driver, HARVEST_SHARES, HarvestJob, LabelJob, RaceKeep, Source, harvest_decision, harvest_root,
    parse_band_shares, position_schedule, row_position, tt_mb_for,
};
use search::pimc::SharedLeaf;
use solver::{
    FastOptions, Leaf, MatchCtx, Position, SolveFamily, SolveMem, Stats, solve_fast_with,
};

struct Counting<'a> {
    inner: &'a SharedLeaf,
    calls: Cell<u64>,
}

impl Leaf for Counting<'_> {
    fn value(&self, s: &MatchState) -> MoneyChance {
        self.calls.set(self.calls.get() + 1);
        self.inner.value(s)
    }
}

#[derive(Default)]
struct Tot {
    solves: u64,
    secs: f64,
    leaf_calls: u64,
    st: Stats,
    max_nodes: u64,
}

impl Tot {
    fn add(&mut self, s: &Stats, secs: f64, calls: u64) {
        self.solves += 1;
        self.secs += secs;
        self.leaf_calls += calls;
        self.max_nodes = self.max_nodes.max(s.nodes);
        let t = &mut self.st;
        t.nodes += s.nodes;
        t.leaves += s.leaves;
        t.tt_probes += s.tt_probes;
        t.tt_hits += s.tt_hits;
        t.tt_entries += s.tt_entries;
        t.tt_replaced += s.tt_replaced;
        t.cutoffs += s.cutoffs;
        t.equiv_skipped += s.equiv_skipped;
        t.forced += s.forced;
        t.tt_bytes = t.tt_bytes.max(s.tt_bytes);
        for (a, b) in t.nodes_by_trick.iter_mut().zip(&s.nodes_by_trick) {
            *a += b;
        }
    }

    fn print(&self, what: &str) {
        let n = self.solves.max(1) as f64;
        let t = &self.st;
        println!(
            "{what}: {} solves, {:.3} ms a solve ({:.2} s per 1,000), {:.0} nodes a solve (max {}), \
             {:.1} ns a node; table {:.0} probes a solve, hit rate {:.1}%, {:.0} entries a solve, \
             {} replaced, peak {} KB; leaves {:.0} a solve, leaf-cache hit rate {:.1}%; forced \
             {:.0} a solve, cutoffs {:.0}, equiv skipped {:.0}; nodes by trick {:?}",
            self.solves,
            self.secs * 1e3 / n,
            self.secs * 1e3 / n,
            t.nodes as f64 / n,
            self.max_nodes,
            self.secs * 1e9 / t.nodes.max(1) as f64,
            t.tt_probes as f64 / n,
            100.0 * t.tt_hits as f64 / t.tt_probes.max(1) as f64,
            t.tt_entries as f64 / n,
            t.tt_replaced,
            t.tt_bytes / 1024,
            t.leaves as f64 / n,
            100.0 * (1.0 - self.leaf_calls as f64 / t.leaves.max(1) as f64),
            t.forced as f64 / n,
            t.cutoffs as f64 / n,
            t.equiv_skipped as f64 / n,
            t.nodes_by_trick,
        );
    }
}

fn solve_one(
    pos: &Position,
    ctx: MatchCtx,
    leaf: &SharedLeaf,
    opts: FastOptions,
    mem: &mut SolveMem,
    tot: &mut Tot,
) -> MoneyChance {
    let c = Counting {
        inner: leaf,
        calls: Cell::new(0),
    };
    let t = Instant::now();
    let sol = solve_fast_with(pos, ctx, &c, opts, mem);
    let secs = t.elapsed().as_secs_f64();
    tot.add(&sol.stats, secs, c.calls.get());
    sol.value.expect("no cap")
}

fn arg<'a>(a: &'a [String], k: &str) -> Option<&'a str> {
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1))
        .map(String::as_str)
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let num = |k: &str, d: u64| -> u64 { arg(&a, k).map_or(d, |x| x.parse().expect(k)) };
    let trick = num("--trick", 3) as u8;
    let count = num("--count", 100);
    let seed = num("--seed", 7);
    let src = arg(&a, "--source").unwrap_or("random");
    let table = search::bot::load_equity(arg(&a, "--equity").ok_or("--equity")?)?;
    let leaf: Arc<SharedLeaf> = Arc::new(table);
    let starts = collect_starts_with(num("--start-hands", 20_000), seed, 4);
    let source = BeliefSource::new();
    let opts = FastOptions {
        tt_mb: tt_mb_for(trick),
        ..FastOptions::default()
    };
    let mut mem = SolveMem::new();
    let mut tot = Tot::default();
    match src {
        "random" | "rules" => {
            let job = LabelJob {
                trick,
                count,
                seed,
                source: Source::parse(src)?,
                threads: 1,
            };
            for i in 0..count {
                let (pos, ctx, _) = row_position(&job, i, &starts, &source);
                let _ = solve_one(&pos, ctx, &*leaf, opts, &mut mem, &mut tot);
            }
            tot.print(&format!("trick {trick} {src}"));
        }
        "harvest" | "race" => {
            let net = arg(&a, "--net")
                .map(|p| nn::Oracle::load(Path::new(p), leaf.clone()))
                .transpose()?;
            let driver = net.as_ref().map_or(Driver::Exact, |o| Driver::Net(o));
            let job = HarvestJob {
                trick,
                count: u64::MAX,
                seed,
                threads: 1,
                cap: 16,
                tricks: vec![trick - 1],
                schedule: position_schedule(parse_band_shares(HARVEST_SHARES)?),
                strata: vec![1, 2, 3, 4],
                race: (src == "race").then_some(RaceKeep {
                    classes: 0b10100,
                    cap: 256,
                }),
                keep: None,
            };
            let mut search_mem = SolveMem::new();
            let mut shared = Tot::default();
            let mut admitted = 0u64;
            let (mut search_s, mut label_s, mut reached, mut rows) = (0.0, 0.0, 0u64, 0u64);
            for j in 0..count {
                let d =
                    harvest_decision(&job, j, &starts, &*leaf, &source, driver, &mut search_mem);
                search_s += d.search_time.as_secs_f64();
                label_s += d.label_time.as_secs_f64();
                reached += d.reached.len() as u64;
                rows += d.rows.len() as u64;
                let list: Vec<(Position, MatchCtx)> = match driver {
                    Driver::Exact => d.reached.iter().map(|r| (r.pos, r.ctx)).collect(),
                    Driver::Net(_) => d.positions.clone(),
                };
                let mut fresh = Vec::with_capacity(list.len());
                for (p, c) in &list {
                    fresh.push(solve_one(p, *c, &*leaf, opts, &mut mem, &mut tot));
                }
                let (_, _, m, _) = harvest_root(&job, j, &starts, &source);
                let mut smem = SolveMem::new();
                smem.share(SolveFamily::of_match(&m).expect("a family"));
                for ((p, c), v) in list.iter().zip(&fresh) {
                    let fam = SolveFamily::of_match(&m).expect("a family");
                    admitted += u64::from(fam.admits(p, *c));
                    let w = solve_one(p, *c, &*leaf, opts, &mut smem, &mut shared);
                    assert_eq!(*v, w, "shared-table value differs");
                }
            }
            println!(
                "trick {trick} {src} ({}): {count} decisions, {reached} reached, {rows} rows; \
                 harvest's own search {search_s:.2} s, its labelling {label_s:.2} s",
                if net.is_some() { "net" } else { "exact" }
            );
            tot.print(&format!(
                "trick {trick} {src} label solves (fresh table each)"
            ));
            shared.print(&format!(
                "trick {trick} {src} label solves (one table a decision; {admitted} admitted)"
            ));
        }
        _ => return Err(format!("unknown --source {src}")),
    }
    Ok(())
}
