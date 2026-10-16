//! `nn-tool`: small tools around the oracle networks.
//!
//! ```text
//! nn-tool fixture --trick K --count N --seed S --out FILE.npy
//! nn-tool eval    --net WEIGHTS.safetensors --shard FILE.npy --out OUT.npy [--reps R]
//! nn-tool classes --shard FILE.npy --out OUT.npy
//! nn-tool groups  --predicates PRED.json --shard FILE.npy --out OUT.npy
//! nn-tool golden  --out FILE.json [--seed 41] [--count 200]
//! nn-tool random  --trick K --widths 512,256,64 --out FILE.safetensors [--seed 17] [--head outcome-v1|money4]
//! nn-tool bench   --equity TABLE --trick K [--worlds 4] [--max 20000] [--seed 5]
//!                 [--widths 512,256,64 --widths 1024,512,128 …] [--net WEIGHTS …] [--reps 3]
//! nn-tool limits                      (prints `max_width=N`: the audited width limit, `nn::MAX_WIDTH`)
//! nn-tool row-weights --shard FILE.npy [--weight W --rows ROWS.npy]
//! ```
//! - `fixture`: a label shard of random positions at the start of trick `K` (random deals and
//!   random legal play, `solver::worlds::sample_world`), valued exactly with the synthetic leaf
//!   (`nn::synthetic`). For tests and the Python parity check only: it needs no equity table.
//! - `eval`: the integer network's money chance of every record of a shard, in the shard's
//!   (rotated) team order, as a `uint64` `.npy` of shape `[N, 4]`; prints the time per
//!   evaluation (forward pass and softmax, best of `R` passes over the shard).
//! - `classes`: the race class (`nn::outcome::race_class_of`) of every record of a shard, from
//!   its position decoded from the binary features (`nn::encode::decode`), not from the record's
//!   mask: a `uint64` `.npy` of shape `[N]`, and the count per class on stderr. For the Python
//!   parity check of the trainer's per-class report (`python/nn`).
//! - `groups`: the membership of every record of a shard in every group of a predicate file
//!   (`nn::predicate`), from the record's stored features: a `uint64` `.npy` of shape `[N, G]`
//!   (1: in the group), and the rows per group on stderr. For the Python parity check of the
//!   auditor's predicate evaluator (`python/nn`).
//! - `golden`: the golden network's outputs on the golden inputs (`nn::synthetic`), as JSON.
//! - `random`: a random integer network for frontier trick `K` (`nn::synthetic::random_net`,
//!   the outcome head by default), for benchmarks only: never for a yardstick value.
//! - `bench`: time one frontier evaluation, split into its parts, on real reached frontier
//!   positions at the start of trick `K` (`nn::synthetic::reached_frontier`: every position the
//!   solver asks for from `--worlds` random roots at trick `K − 1`, in its order, at most
//!   `--max`), with the production equity table, for random outcome-head networks of the given
//!   hidden widths (`nn::synthetic::random_net`) and any trained weights files. Prints
//!   microseconds per evaluation (best of `--reps` passes).
//! - `row-weights`: a shard's per-row loss weights (`nn::rowweights`, the sidecar
//!   `FILE.npy.rowweights` the trainer reads). With `--weight W --rows ROWS.npy` it writes the
//!   sidecar (replacing any): the rows listed in `ROWS.npy` (a 1-D `<u8` array of row indices, as
//!   `numpy.save` writes it) weigh `W` (a multiple of 1/256, from 1/256 to 65535/256) and the
//!   others weigh 1. Either way it prints the sidecar's summary as JSON (`rows`, `rows_not_1`,
//!   `sum`, `min` and `max` in units of 1/256, `sha256`), or `"sidecar": null` without one.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use equity::npy::to_npy_u64;
use nn::encode::encode;
use nn::money::money_from_logits;
use nn::oracle::{Head, parse_net};
use nn::outcome::{mix_outcomes, money_from_outcomes, outcomes, vocab};
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
    fn all(&self, k: &str) -> Vec<&str> {
        self.0
            .iter()
            .enumerate()
            .filter(|(_, a)| *a == k)
            .filter_map(|(i, _)| self.0.get(i + 1).map(String::as_str))
            .collect()
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
        Some("classes") => classes(&a),
        Some("groups") => groups(&a),
        Some("golden") => golden(&a),
        Some("bench") => bench(&a),
        Some("random") => random(&a),
        Some("limits") => {
            println!("max_width={}", nn::MAX_WIDTH);
            Ok(())
        }
        Some("row-weights") => row_weights(&a),
        _ => Err(
            "usage: nn-tool fixture|eval|classes|groups|golden|random|bench|limits|row-weights … \
             (see the source docs)"
                .into(),
        ),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("nn-tool: {e}");
            ExitCode::from(2)
        }
    }
}

fn row_weights(a: &Args) -> Result<(), String> {
    use nn::rowweights;
    let shard = Path::new(a.req("--shard")?);
    let rows = rowweights::shard_rows(shard)?;
    if let Some(w) = a.get("--weight") {
        let weight = w
            .parse::<f64>()
            .ok()
            .and_then(rowweights::fixed)
            .ok_or_else(|| format!("--weight {w}: a multiple of 1/256 from 1/256 to 65535/256"))?;
        let f = a.req("--rows")?;
        let b = std::fs::read(f).map_err(|e| format!("{f}: {e}"))?;
        let (shape, chosen) = equity::npy::from_npy_u64(&b).map_err(|e| format!("{f}: {e}"))?;
        if shape.len() != 1 {
            return Err(format!(
                "{f}: row indices must be one-dimensional, not {shape:?}"
            ));
        }
        let weights = rowweights::chosen_rows(rows, &chosen, weight)?;
        rowweights::write(shard, &weights)?;
    } else if a.get("--rows").is_some() {
        return Err("--rows needs --weight".into());
    }
    let path = rowweights::path_for(shard);
    let out = match rowweights::read(shard)? {
        None => serde_json::json!({ "shard": shard, "rows": rows, "sidecar": null }),
        Some(w) => {
            if w.len() != rows {
                return Err(format!(
                    "{}: {} row weights for {rows} rows",
                    path.display(),
                    w.len()
                ));
            }
            let s = rowweights::summary(&w);
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            serde_json::json!({ "shard": shard, "rows": rows, "sidecar": path,
                                "sha256": equity::sha256::sha256_hex(&bytes),
                                "rows_not_1": s.rows_not_1, "sum": s.sum,
                                "min": s.min, "max": s.max })
        }
    };
    println!("{out}");
    Ok(())
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

/// The money chance of a shard row from its logits under a network with this head (rotated
/// order, as stored).
fn row_money(z: &[i64], head: Head, r: &Row) -> equity::MoneyChance {
    match head {
        Head::Money4 => money_from_logits(z),
        Head::Outcome => mix_outcomes(z, r.outcomes.len(), r.mask, &r.outcomes),
    }
}

fn classes(a: &Args) -> Result<(), String> {
    let spath = a.req("--shard")?;
    let rows = from_npy(&std::fs::read(spath).map_err(|e| format!("{spath}: {e}"))?)?;
    let mut out = Vec::with_capacity(rows.len());
    let mut counts = [0u64; nn::outcome::N_RACE_CLASSES];
    for (i, r) in rows.iter().enumerate() {
        let (c, ctx) = nn::encode::decode(&r.features)
            .ok_or_else(|| format!("row {i}: the features are not an encoding"))?;
        let k = nn::outcome::race_class_of(&c, &ctx, &SyntheticLeaf);
        counts[usize::from(k)] += 1;
        out.push(u64::from(k));
    }
    let opath = a.req("--out")?;
    std::fs::write(opath, to_npy_u64(&[out.len()], &out)).map_err(|e| format!("{opath}: {e}"))?;
    eprintln!(
        "nn-tool classes: {} records, by race class 0..4: {counts:?}",
        rows.len()
    );
    Ok(())
}

fn groups(a: &Args) -> Result<(), String> {
    let p = nn::predicate::Predicates::load(Path::new(a.req("--predicates")?))?;
    let spath = a.req("--shard")?;
    let rows = from_npy(&std::fs::read(spath).map_err(|e| format!("{spath}: {e}"))?)?;
    let g = p.len();
    let mut out = Vec::with_capacity(rows.len() * g);
    let mut counts = vec![0u64; g];
    for r in &rows {
        for (k, c) in counts.iter_mut().enumerate() {
            let x = u64::from(p.holds(k, &r.features));
            *c += x;
            out.push(x);
        }
    }
    for (name, c) in p.names().iter().zip(&counts) {
        eprintln!("{name}: {c} of {} rows", rows.len());
    }
    std::fs::write(a.req("--out")?, to_npy_u64(&[rows.len(), g], &out)).map_err(|e| e.to_string())
}

fn eval(a: &Args) -> Result<(), String> {
    let wpath = a.req("--net")?;
    let bytes = std::fs::read(wpath).map_err(|e| format!("{wpath}: {e}"))?;
    let file = nn::safetensors::read(&bytes)?;
    let net = parse_net(&file)?;
    let fast = nn::fast::FastMlp::new(net.clone());
    let head = Head::of(&file)?;
    let spath = a.req("--shard")?;
    let rows = from_npy(&std::fs::read(spath).map_err(|e| format!("{spath}: {e}"))?)?;
    // The fast forward pass, checked against the plain reference on every row.
    let mut s = nn::fast::Scratch::new();
    let mut out = Vec::with_capacity(rows.len() * 4);
    for (i, r) in rows.iter().enumerate() {
        let z = fast.forward(&r.features, &mut s);
        if z != net.forward(&r.features) {
            return Err(format!(
                "row {i}: the fast forward pass differs from the plain one"
            ));
        }
        out.extend_from_slice(&row_money(&z, head, r).0);
    }
    let reps = a.num("--reps", 3)?.max(1);
    let (mut best, mut plain) = (f64::INFINITY, f64::INFINITY);
    let mut check = 0u64;
    for _ in 0..reps {
        let t = Instant::now();
        for r in &rows {
            check ^= row_money(&fast.forward(&r.features, &mut s), head, r).0[0];
        }
        best = best.min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        let mut c = 0u64;
        for r in &rows {
            c ^= row_money(&net.forward(&r.features), head, r).0[0];
        }
        plain = plain.min(t.elapsed().as_secs_f64());
        std::hint::black_box(c);
    }
    let per = |x: f64| x * 1e9 / rows.len().max(1) as f64;
    eprintln!(
        "nn-tool eval: {} records, widths {:?}, {:.0} ns per evaluation (forward + softmax; the plain forward pass: {:.0} ns; check {check:x}; kernels {})",
        rows.len(),
        net.widths(),
        per(best),
        per(plain),
        nn::kernel::level().name()
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

fn random(a: &Args) -> Result<(), String> {
    let trick = a.num("--trick", 4)? as u8;
    if !(2..=5).contains(&trick) {
        return Err("--trick is 2..=5".into());
    }
    let w = a.get("--widths").unwrap_or("512,256,64");
    let widths: Vec<usize> = w
        .split(',')
        .map(|x| x.parse().map_err(|_| format!("--widths {w}")))
        .collect::<Result<_, _>>()?;
    nn::mlp::check_hidden_widths(&widths).map_err(|e| format!("--widths {w}: {e}"))?;
    let head = a.get("--head").unwrap_or("outcome-v1");
    let n_out = match head {
        "outcome-v1" => vocab(trick),
        "money4" => 4,
        h => return Err(format!("--head {h}")),
    };
    let net = nn::synthetic::random_net(a.num("--seed", 17)?, &widths, n_out, 100);
    let mut meta = std::collections::BTreeMap::new();
    meta.insert("head".to_string(), head.to_string());
    meta.insert(
        "note".to_string(),
        "random network for benchmarks".to_string(),
    );
    std::fs::write(a.req("--out")?, nn::oracle::write_net(&net, trick, &meta))
        .map_err(|e| e.to_string())
}

/// Best of `reps` timed passes of `f` over `n` items, in microseconds per item (`f` gets the
/// pass number).
fn time_us(reps: u64, n: usize, mut f: impl FnMut(u64) -> u64) -> (f64, u64) {
    let mut best = f64::INFINITY;
    let mut check = 0;
    for rep in 0..reps.max(1) {
        let t = Instant::now();
        check ^= f(rep);
        best = best.min(t.elapsed().as_secs_f64());
    }
    (best * 1e6 / n.max(1) as f64, check)
}

fn bench(a: &Args) -> Result<(), String> {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    use nn::Oracle;
    use nn::encode::{DenseKey, binary_features, dense_features};
    use nn::fast::Scratch;
    use nn::kernel::{Level, detected};
    use nn::oracle::write_net;
    use nn::outcome::{mix_outcomes, outcomes_of, top_team};
    use nn::synthetic::{random_net, reached_frontier};

    let trick = a.num("--trick", 4)? as u8;
    if !(2..=5).contains(&trick) {
        return Err("--trick is 2..=5".into());
    }
    let epath = a.req("--equity")?;
    let table =
        equity::EquityTable::from_npy(&std::fs::read(epath).map_err(|e| format!("{epath}: {e}"))?)
            .map_err(|e| format!("{epath}: {e}"))?;
    let leaf: Arc<equity::EquityTable> = Arc::new(table);
    let reps = a.num("--reps", 3)?;
    let t = Instant::now();
    let pos = reached_frontier(
        a.num("--seed", 5)?,
        trick,
        a.num("--worlds", 4)?,
        a.num("--max", 20_000)? as usize,
        &*leaf,
    );
    let n = pos.len();
    let distinct = pos
        .iter()
        .map(|(c, x)| (*c, x.games, x.dealer_team))
        .collect::<BTreeSet<_>>()
        .len();
    let keys: BTreeSet<DenseKey> = pos.iter().map(|(c, x)| DenseKey::of(c, x)).collect();
    let okeys: BTreeSet<(DenseKey, Option<u8>)> = pos
        .iter()
        .map(|(c, x)| (DenseKey::of(c, x), top_team(c)))
        .collect();
    println!(
        "trick {trick}: {n} frontier positions as the solver asks for them ({distinct} distinct canonical forms, {} dense keys, {} outcome keys), collected in {:.1} s; levels up to {}",
        keys.len(),
        okeys.len(),
        t.elapsed().as_secs_f64(),
        detected().name()
    );
    let mut nets: Vec<(String, Vec<u8>)> = Vec::new();
    let widths = a.all("--widths");
    let widths = if widths.is_empty() && !a.0.iter().any(|x| x == "--net") {
        vec!["512,256,64", "1024,512,128"]
    } else {
        widths
    };
    for w in widths {
        let ws: Vec<usize> = w
            .split(',')
            .map(|x| x.parse().map_err(|_| format!("--widths {w}")))
            .collect::<Result<_, _>>()?;
        let net = random_net(17, &ws, vocab(trick), 100);
        let mut meta = BTreeMap::new();
        meta.insert("head".to_string(), "outcome-v1".to_string());
        nets.push((format!("random {w}"), write_net(&net, trick, &meta)));
    }
    for p in a.all("--net") {
        nets.push((
            p.to_string(),
            std::fs::read(p).map_err(|e| format!("{p}: {e}"))?,
        ));
    }
    for (name, bytes) in &nets {
        let load = || Oracle::from_bytes(bytes, leaf.clone());
        let o = load()?;
        if o.trick != trick {
            return Err(format!("{name}: a network for trick {}", o.trick));
        }
        println!("{name} {:?}", o.net().widths());
        // The plain path, split into its parts.
        let (enc, _) = time_us(reps, n, |_| {
            pos.iter()
                .map(|(c, x)| u64::from(nn::encode(c, x, &*leaf).0.n_active))
                .sum()
        });
        let feats: Vec<_> = pos
            .iter()
            .map(|(c, x)| nn::encode(c, x, &*leaf).0)
            .collect();
        let (fwd, _) = time_us(reps, n, |_| {
            feats.iter().map(|f| o.net().forward(f)[0] as u64).sum()
        });
        let zs: Vec<_> = feats.iter().map(|f| o.net().forward(f)).collect();
        let (out, _) = time_us(reps, n, |_| {
            pos.iter()
                .zip(&zs)
                .map(|((c, x), z)| money_from_outcomes(z, &outcomes(c, x, &*leaf)).0[0])
                .sum()
        });
        let (tot, check) = time_us(reps, n, |_| {
            pos.iter().map(|(c, x)| o.eval_reference(c, x).0[0]).sum()
        });
        println!(
            "  plain: {tot:.2} µs per evaluation = encode {enc:.2} + forward {fwd:.2} + outcomes {out:.2} (check {check:x})"
        );
        // The fast path at each level: its parts in the order an evaluation runs them, with
        // the caches an evaluation keeps (dense parts and outcome tables per key, computed on
        // first use), then the whole evaluation without and with the memo.
        for l in Level::ALL.into_iter().filter(|&l| l <= detected()) {
            nn::kernel::set_cap(Some(l));
            let (a1, _) = time_us(reps, n, |_| {
                let mut d: BTreeMap<DenseKey, [i32; 100]> = BTreeMap::new();
                pos.iter()
                    .map(|(c, x)| {
                        let k = DenseKey::of(c, x);
                        let f = binary_features(c, x);
                        let dd = d.entry(k).or_insert_with(|| dense_features(&k, &*leaf));
                        u64::from(f.n_active) ^ dd[0] as u64
                    })
                    .sum()
            });
            let dense: BTreeMap<DenseKey, [i32; 100]> = keys
                .iter()
                .map(|k| (*k, dense_features(k, &*leaf)))
                .collect();
            let bins: Vec<_> = pos.iter().map(|(c, x)| binary_features(c, x)).collect();
            let fast = o.fast();
            let (b1, _) = time_us(reps, n, |_| {
                let mut s = Scratch::new();
                let mut bases: BTreeMap<DenseKey, Vec<i64>> = BTreeMap::new();
                pos.iter()
                    .zip(&bins)
                    .map(|((c, x), f)| {
                        let k = DenseKey::of(c, x);
                        let b = bases.entry(k).or_insert_with(|| {
                            let mut b = vec![0; fast.base_len()];
                            fast.base(&dense[&k], &mut s, &mut b);
                            b
                        });
                        fast.forward_base(f.active(), b, &mut s)[0] as u64
                    })
                    .sum()
            });
            let (c1, _) = time_us(reps, n, |_| {
                let mut tabs = BTreeMap::new();
                pos.iter()
                    .zip(&zs)
                    .map(|((c, x), z)| {
                        let k = (DenseKey::of(c, x), top_team(c));
                        let t = tabs
                            .entry(k)
                            .or_insert_with(|| outcomes_of(&k.0, k.1, &*leaf));
                        mix_outcomes(z, t.n, t.mask, &t.vecs).0[0]
                    })
                    .sum()
            });
            // A fresh oracle (empty caches) for every pass, made before the clock starts.
            let mut fresh: Vec<Oracle> =
                (0..reps.max(1)).map(|_| load()).collect::<Result<_, _>>()?;
            for o in &mut fresh {
                o.set_memo(false);
            }
            let (e0, c0) = time_us(reps, n, |rep| {
                let o = &fresh[rep as usize];
                pos.iter().map(|(c, x)| o.eval(c, x).0[0]).sum()
            });
            let fresh: Vec<Oracle> = (0..reps.max(1)).map(|_| load()).collect::<Result<_, _>>()?;
            let (e1, c2) = time_us(reps, n, |rep| {
                let o = &fresh[rep as usize];
                pos.iter().map(|(c, x)| o.eval(c, x).0[0]).sum()
            });
            let stats = fresh[fresh.len() - 1].thread_stats();
            assert_eq!(
                (c0, c2),
                (check, check),
                "the fast path gives the plain values"
            );
            println!(
                "  fast at {}: {e0:.2} µs per evaluation (= binary features and keyed dense features {a1:.2} + forward {b1:.2} + outcomes {c1:.2}); {e1:.2} µs per request with the memo ({} of {} answered by it; {} forward passes, {} dense parts, {} outcome tables); first layer in i32: {}",
                l.name(),
                stats.memo_hits,
                stats.requests,
                stats.forwards,
                stats.dense_misses,
                stats.outcome_misses,
                fast.bin_i32()
            );
        }
        nn::kernel::set_cap(None);
    }
    Ok(())
}
