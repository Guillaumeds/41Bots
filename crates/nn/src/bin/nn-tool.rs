//! `nn-tool`: small tools around the oracle networks.
//!
//! ```text
//! nn-tool fixture --trick K --count N --seed S --out FILE.npy
//! nn-tool eval    --net WEIGHTS.safetensors --shard FILE.npy --out OUT.npy [--reps R]
//! nn-tool golden  --out FILE.json [--seed 41] [--count 200]
//! ```
//! - `fixture`: a label shard of random positions at the start of trick `K` (random deals and
//!   random legal play, `solver::worlds::sample_world`), valued exactly with the synthetic leaf
//!   (`nn::synthetic`). For tests and the Python parity check only: it needs no equity table.
//! - `eval`: the integer network's money chance of every record of a shard, in the shard's
//!   (rotated) team order, as a `uint64` `.npy` of shape `[N, 4]`; prints the time per
//!   evaluation (forward pass and softmax, best of `R` passes over the shard).
//! - `golden`: the golden network's outputs on the golden inputs (`nn::synthetic`), as JSON.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use equity::npy::to_npy_u64;
use nn::encode::encode;
use nn::money::money_from_logits;
use nn::oracle::{Head, parse_net};
use nn::outcome::{Outcomes, money_from_outcomes, outcomes, vocab};
use nn::shard::{Row, SOURCE_FIXTURE, from_npy, to_npy};
use nn::synthetic::{SyntheticLeaf, golden_features, golden_net, golden_net_out, golden_outcomes};
use solver::{FastOptions, canonical, solve_fast};

struct Args(Vec<String>);

impl Args {
    fn get(&self, k: &str) -> Option<&str> {
        self.0
            .iter()
            .position(|a| a == k)
            .and_then(|i| self.0.get(i + 1))
            .map(String::as_str)
    }
    fn req(&self, k: &str) -> Result<&str, String> {
        self.get(k).ok_or_else(|| format!("missing {k}"))
    }
    fn num(&self, k: &str, d: u64) -> Result<u64, String> {
        self.get(k).map_or(Ok(d), |v| {
            v.parse().map_err(|_| format!("{k}: not a number: {v}"))
        })
    }
}

fn main() -> ExitCode {
    let a = Args(std::env::args().skip(1).collect());
    let r = match a.0.first().map(String::as_str) {
        Some("fixture") => fixture(&a),
        Some("eval") => eval(&a),
        Some("golden") => golden(&a),
        _ => Err("usage: nn-tool fixture|eval|golden … (see the source docs)".into()),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("nn-tool: {e}");
            ExitCode::from(2)
        }
    }
}

fn fixture(a: &Args) -> Result<(), String> {
    let trick = a.num("--trick", 4)? as u8;
    if !(2..=5).contains(&trick) {
        return Err("--trick is 2..=5".into());
    }
    let count = a.num("--count", 1000)?;
    let seed = a.num("--seed", 7)?;
    let leaf = SyntheticLeaf;
    let mut rows = Vec::new();
    for i in 0..count {
        let (pos, ctx, _) = solver::worlds::sample_world(seed, trick, i, 0);
        let v = solve_fast(&pos, ctx, &leaf, FastOptions::default())
            .value
            .expect("no time cap");
        let c = canonical(&pos);
        let (features, r) = encode(&c, &ctx, &leaf);
        let value = v.rotate(r % 4);
        let o = outcomes(&c, &ctx, &leaf);
        let label = o
            .label(&value)
            .ok_or_else(|| format!("position {i}: no outcome has the exact value"))?;
        rows.push(Row {
            features,
            value,
            source: SOURCE_FIXTURE,
            trick,
            label,
            mask: o.mask,
            outcomes: o.vecs,
        });
    }
    std::fs::write(a.req("--out")?, to_npy(&rows, vocab(trick))).map_err(|e| e.to_string())
}

/// The money chance of a shard row under a network with this head (rotated order, as stored).
fn row_money(net: &nn::IntMlp, head: Head, r: &Row) -> equity::MoneyChance {
    let z = net.forward(&r.features);
    match head {
        Head::Money4 => money_from_logits(&z),
        Head::Outcome => money_from_outcomes(
            &z,
            &Outcomes {
                n: r.outcomes.len(),
                mask: r.mask,
                vecs: r.outcomes.clone(),
            },
        ),
    }
}

fn eval(a: &Args) -> Result<(), String> {
    let wpath = a.req("--net")?;
    let bytes = std::fs::read(wpath).map_err(|e| format!("{wpath}: {e}"))?;
    let file = nn::safetensors::read(&bytes)?;
    let net = parse_net(&file)?;
    let head = Head::of(&file)?;
    let spath = a.req("--shard")?;
    let rows = from_npy(&std::fs::read(spath).map_err(|e| format!("{spath}: {e}"))?)?;
    let mut out = Vec::with_capacity(rows.len() * 4);
    for r in &rows {
        out.extend_from_slice(&row_money(&net, head, r).0);
    }
    let reps = a.num("--reps", 3)?.max(1);
    let mut best = f64::INFINITY;
    let mut check = 0u64;
    for _ in 0..reps {
        let t = Instant::now();
        for r in &rows {
            check ^= row_money(&net, head, r).0[0];
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    eprintln!(
        "nn-tool eval: {} records, widths {:?}, {:.0} ns per evaluation (forward + softmax; check {check:x})",
        rows.len(),
        net.widths(),
        best * 1e9 / rows.len().max(1) as f64
    );
    std::fs::write(a.req("--out")?, to_npy_u64(&[rows.len(), 4], &out)).map_err(|e| e.to_string())
}

fn golden(a: &Args) -> Result<(), String> {
    let seed = a.num("--seed", 41)?;
    let count = a.num("--count", 200)? as usize;
    let net = golden_net(seed);
    let outs: Vec<[u64; 4]> = golden_features(seed + 1, count)
        .iter()
        .map(|f| money_from_logits(&net.forward(f)).0)
        .collect();
    // The outcome head: a 14-outcome network (trick 4's vocabulary) on the same inputs.
    let onet = golden_net_out(seed, 14);
    let tables = golden_outcomes(seed + 2, count, 14);
    let oouts: Vec<[u64; 4]> = golden_features(seed + 1, count)
        .iter()
        .zip(&tables)
        .map(|(f, o)| money_from_outcomes(&onet.forward(f), o).0)
        .collect();
    let v = serde_json::json!({
        "seed": seed, "count": count, "outputs": outs,
        "outcome": { "n_out": 14, "outputs": oouts },
    });
    std::fs::write(Path::new(a.req("--out")?), v.to_string() + "\n").map_err(|e| e.to_string())
}
