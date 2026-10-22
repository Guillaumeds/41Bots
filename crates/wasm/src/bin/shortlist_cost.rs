//! `shortlist-cost`: where the root-restricted solve's saving goes (docs/EXPERIMENTS.md E11).
//!
//! ```text
//! shortlist-cost EQUITY.npy O3.safetensors [DEALS_PER_POSITION [POSITIONS_PER_PLACE]]
//! ```
//! For the API's bench positions of tricks 2 (hybrid, O_3 at the frontier) and 3 (exact), with at
//! least 4 candidates: the setup no root card changes (belief preparation, drawing the deals), then
//! the exact solve of the first deals with the 1, 2, 3 best cards kept (best by mean over those
//! deals) and with all of them, each as a request of its own (cold network memo, fresh table),
//! counting solver nodes, leaves and frontier (network) calls and the time. Totals per trick and
//! place, as ratios to the full solve.

use std::sync::Arc;
use std::time::Instant;

use engine::game::{Action, Rules};
use equity::EquityTable;
use nn::Oracle;
use search::pimc::{Ctx, SearchConfig, WorldSolve, candidates, exact_world, world_match};
use search::worlds::{BeliefSource, WorldSource};
use solver::{FrontierCut, Leaf, SolveMem};
use wasm::appview::{AppTable, rebuild};

const KEEPS: [usize; 4] = [1, 2, 3, usize::MAX];

#[derive(Default, Clone, Copy)]
struct Sum {
    ms: f64,
    nodes: u64,
    leaves: u64,
    frontier: u64,
    n: u64,
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 2 {
        return Err("usage: shortlist-cost EQUITY.npy O3 [DEALS [PER_PLACE]]".into());
    }
    let deals: u64 = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(8);
    let per: u64 = a.get(3).and_then(|x| x.parse().ok()).unwrap_or(4);
    let rd = |p: &str| std::fs::read(p).map_err(|e| format!("{p}: {e}"));
    let table = EquityTable::from_npy(&rd(&a[0])?).map_err(|e| e.to_string())?;
    let leaf: Arc<dyn Leaf + Send + Sync> = Arc::new(table);
    let o3_bytes = rd(&a[1])?;
    // [trick][place][setup + keeps]
    let mut cands: std::collections::BTreeMap<(u8, usize), Vec<usize>> = Default::default();
    let mut tot: std::collections::BTreeMap<(u8, usize), [Sum; 5]> = Default::default();
    for (name, req) in wasm::fixtures::bench_requests(1, per) {
        let trick: u8 = name[7..8].parse().map_err(|_| "name")?;
        if trick > 3 {
            continue;
        }
        let place: usize = name
            .split("-p")
            .nth(1)
            .and_then(|s| s[..1].parse().ok())
            .ok_or("place")?;
        let v: serde_json::Value = serde_json::from_str(&req).map_err(|e| e.to_string())?;
        let seat = v["seat"].as_u64().ok_or("seat")? as u8;
        let seed = v["seed"]
            .as_str()
            .and_then(|s| s.parse().ok())
            .or(v["seed"].as_u64())
            .ok_or("seed")?;
        let table = AppTable::read(&v["match"], seat, None).map_err(|e| e.reason)?;
        let r = rebuild(&table).map_err(|e| e.reason)?;
        let (_, all) = candidates(&r.view, &r.m.legal_actions());
        cands.entry((trick, place)).or_default().push(all.len());
        if all.len() < 4 || all.iter().any(|c| !matches!(c.0, Action::Play(_))) {
            continue; // the keeps 1, 2, 3 and all must differ
        }
        let source = BeliefSource::new();
        let ctx = Ctx {
            leaf: &*leaf,
            source: &source,
            rules: Rules::HOUSE,
        };
        let cfg = SearchConfig {
            tt_mb: 16,
            threads: 1,
            solve_threads: 1,
            ..SearchConfig::default()
        };
        // Setup that no root card changes: the belief preparation once, then each deal's draw.
        let t = Instant::now();
        let worlds = source.prepare(&r.view, Rules::HOUSE);
        let prepare = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let ws: Vec<_> = (0..deals)
            .map(|w| world_match(&r.view, &*worlds, w, seed, Rules::HOUSE))
            .collect();
        let draws = t.elapsed().as_secs_f64() * 1e3;
        // Rank the cards by their mean over these deals (a fixed shortlist, as the API has one).
        let team = engine::team_of(seat);
        let load = || Oracle::from_bytes(&o3_bytes, leaf.clone()).map_err(|e| e.to_string());
        let mut sums = vec![0u128; all.len()];
        {
            let o = load()?;
            let cut = (trick == 2).then(|| FrontierCut::new(3, &o));
            let mut mem = SolveMem::new();
            for w in &ws {
                let mut run = WorldSolve::default();
                let vals = exact_world(w, &all, &ctx, &cfg, None, &mut mem, &mut run, cut)
                    .ok_or("stopped")?;
                for (s, v) in sums.iter_mut().zip(&vals) {
                    *s += u128::from(v.get(team));
                }
            }
        }
        let mut order: Vec<usize> = (0..all.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(sums[i]));
        let row = tot.entry((trick, place)).or_insert([Sum::default(); 5]);
        row[0].ms += prepare + draws;
        row[0].n += deals;
        for (ki, &k) in KEEPS.iter().enumerate() {
            let keep: Vec<_> = order.iter().take(k).map(|&i| all[i]).collect();
            // A request of its own: cold network memo, fresh table, the deals one after another.
            let o = load()?;
            let cut = (trick == 2).then(|| FrontierCut::new(3, &o));
            let mut mem = SolveMem::new();
            let mut run = WorldSolve::default();
            let t = Instant::now();
            for w in &ws {
                exact_world(w, &keep, &ctx, &cfg, None, &mut mem, &mut run, cut)
                    .ok_or("stopped")?;
            }
            let s = &mut row[ki + 1];
            s.ms += t.elapsed().as_secs_f64() * 1e3;
            s.nodes += run.stats.nodes;
            s.leaves += run.stats.leaves;
            s.frontier += run.stats.frontier;
            s.n += deals;
        }
    }
    for ((t, p), c) in &cands {
        eprintln!("trick {t} place {p}: candidates per position {c:?}");
    }
    println!(
        "trick place positions*deals | setup ms/deal | solve ms: keep 1, 2, 3 as ratio to all (all in ms) | nodes ratio | frontier ratio"
    );
    let mut grand: std::collections::BTreeMap<u8, [Sum; 5]> = Default::default();
    for (&(t, p), r) in &tot {
        let g = grand.entry(t).or_insert([Sum::default(); 5]);
        for i in 0..5 {
            g[i].ms += r[i].ms;
            g[i].nodes += r[i].nodes;
            g[i].leaves += r[i].leaves;
            g[i].frontier += r[i].frontier;
            g[i].n += r[i].n;
        }
        println!("{}", line(t, &p.to_string(), r));
    }
    for (t, g) in &grand {
        println!("{}", line(*t, "all", g));
    }
    Ok(())
}

fn line(t: u8, place: &str, r: &[Sum; 5]) -> String {
    let all = r[4];
    let ratio = |f: fn(&Sum) -> f64| {
        (1..4)
            .map(|i| format!("{:.2}", f(&r[i]) / f(&all).max(1e-9)))
            .collect::<Vec<_>>()
            .join("/")
    };
    format!(
        "{t} {place:>5} {:>5} | setup {:8.3} | solve {} (all {:.3}) | nodes {} | frontier {} | setup/(setup+all) {:.2}",
        all.n,
        r[0].ms / r[0].n.max(1) as f64,
        ratio(|s| s.ms),
        all.ms / all.n.max(1) as f64,
        ratio(|s| s.nodes as f64),
        ratio(|s| s.frontier as f64),
        r[0].ms / (r[0].ms + all.ms).max(1e-9)
    )
}
