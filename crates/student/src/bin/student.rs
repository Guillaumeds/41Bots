//! `student`: the student network's tools (docs/STUDENT.md §5).
//!
//! ```text
//! student encode  --teacher A.npy [--teacher B.npy …] --equity TABLE --out F.npy
//!                 [--allow-stamp KEY …] [--deal-weights W.npy]
//! student eval    --net INT.safetensors --features F.npy --out LOGITS.npy
//! student fixture --count N --seed S --out F.npy [--role fixture|test]
//! student hash
//! ```
//! - `encode`: teacher shards (`student::record`) → one feature shard (`student::features`) and
//!   its sidecar `F.npy.json`: the records of one decision merged across shards by deal index, the
//!   view and candidates encoded by this build's encoder (`student-v1`, its hash in the sidecar),
//!   the targets the teacher's means (or with `--deal-weights` the weighted means) in the rotated
//!   pair order. The shards' stamps must agree (`--allow-stamp KEY` lets one differ); the equity
//!   table must be the one the teacher used (`equity_sha256`). A test-set shard gives a test-set
//!   feature shard (`role: "test"`), which the trainer never trains on. Byte-identical for the same
//!   shards in the same order.
//! - `eval`: the integer network's logits (`<i8`, `[N, 32, 4]`, rotated pair order, 0 past a
//!   decision's candidates) for every record of a feature shard: the Python trainer's integer
//!   forward pass is checked against it bit for bit (`python/nn/tests/test_student.py`).
//! - `fixture`: a feature shard of `rules-v1` self-play decisions at tricks 1 and 2 encoded with the
//!   synthetic leaf (`nn::synthetic`) and synthetic targets: for tests only (no equity table, no
//!   teacher).
//! - `hash`: prints the encoder's name, hash and sizes.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use equity::{EquityTable, MoneyChance};
use serde_json::{Value, json};
use student::data::{DealWeights, Shards, Uniform, WeightsFile, feature_row};
use student::encode::{
    ENCODER, K_MAX, N_BINARY, N_VIEW, V_MAX, encode_cand, encode_view, encoder_sha256,
};
use student::features::{self, FeatureRow};
use student::net::StudentNet;
use student::record::MAX_CANDS;

struct Args(Vec<String>);

impl Args {
    fn get(&self, k: &str) -> Option<&str> {
        self.0
            .iter()
            .position(|a| a == k)
            .and_then(|i| self.0.get(i + 1))
            .map(String::as_str)
    }
    fn all(&self, k: &str) -> Vec<&str> {
        self.0
            .iter()
            .enumerate()
            .filter(|(_, a)| *a == k)
            .filter_map(|(i, _)| self.0.get(i + 1).map(String::as_str))
            .collect()
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
        Some("encode") => encode(&a),
        Some("eval") => eval(&a),
        Some("fixture") => fixture(&a),
        Some("hash") => {
            println!("{}", sizes());
            Ok(())
        }
        _ => Err("usage: student encode|eval|fixture|hash … (see the source docs)".into()),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("student: {e}");
            ExitCode::from(2)
        }
    }
}

/// The encoder's facts, as in every feature shard's sidecar.
fn sizes() -> Value {
    json!({
        "format": features::FORMAT,
        "encoder": ENCODER,
        "encoder_sha256": encoder_sha256(),
        "layout": student::encode::layout(),
        "n_view": N_VIEW,
        "n_binary": N_BINARY,
        "n_dense": nn::encode::N_DENSE,
        "v_max": V_MAX,
        "k_max": K_MAX,
        "c_max": MAX_CANDS,
    })
}

fn sha256_hex(b: &[u8]) -> String {
    equity::sha256::sha256_hex(b)
}

/// Write a feature shard and its sidecar (`extra` merged into the sizes): `.part` files renamed
/// into place, the shard last.
fn write_shard(out: &Path, rows: &[FeatureRow], extra: Value) -> Result<String, String> {
    let mut b = features::header(rows.len() as u64);
    for r in rows {
        features::row_bytes(r, &mut b);
    }
    let sha = sha256_hex(&b);
    let mut side = sizes();
    if let (Some(s), Value::Object(e)) = (side.as_object_mut(), extra) {
        s.extend(e);
        s.insert("records".into(), json!(rows.len()));
        s.insert("sha256".into(), json!(sha));
        s.insert("bytes".into(), json!(b.len()));
    }
    let part = |p: &Path| PathBuf::from(format!("{}.part", p.display()));
    let sp = PathBuf::from(format!("{}.json", out.display()));
    std::fs::write(
        part(&sp),
        serde_json::to_string_pretty(&side).expect("JSON") + "\n",
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(part(out), &b).map_err(|e| e.to_string())?;
    std::fs::rename(part(&sp), &sp).map_err(|e| e.to_string())?;
    std::fs::rename(part(out), out).map_err(|e| e.to_string())?;
    Ok(sha)
}

fn encode(a: &Args) -> Result<(), String> {
    let paths: Vec<PathBuf> = a.all("--teacher").into_iter().map(PathBuf::from).collect();
    if paths.is_empty() {
        return Err("give at least one --teacher shard".into());
    }
    let allow: Vec<String> = a
        .all("--allow-stamp")
        .into_iter()
        .map(String::from)
        .collect();
    let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
    let mut shards = Shards::open(&refs, &allow)?;
    let eq_path = a.req("--equity")?;
    let bytes = std::fs::read(eq_path).map_err(|e| format!("{eq_path}: {e}"))?;
    let eq_sha = sha256_hex(&bytes);
    for r in &shards.readers {
        if r.sidecar["equity_sha256"] != eq_sha.as_str() {
            return Err(format!(
                "{}: made with equity table {}, --equity is {eq_sha}",
                r.path.display(),
                r.sidecar["equity_sha256"]
            ));
        }
    }
    let table = EquityTable::from_npy(&bytes).map_err(|e| format!("{eq_path}: {e}"))?;
    let weights: Box<dyn DealWeights> = match a.get("--deal-weights") {
        None => Box::new(Uniform),
        Some(p) => Box::new(WeightsFile::read(Path::new(p))?),
    };
    let weights_sha = a
        .get("--deal-weights")
        .map(|p| std::fs::read(p).map(|b| sha256_hex(&b)))
        .transpose()
        .map_err(|e| e.to_string())?;
    let mut rows = Vec::with_capacity(shards.groups.len());
    for g in 0..shards.groups.len() {
        let m = shards.merged(g)?;
        rows.push(feature_row(&m, &table, &*weights).map_err(|e| format!("decision {g}: {e}"))?);
    }
    let out = PathBuf::from(a.req("--out")?);
    let teacher: Vec<Value> = shards
        .readers
        .iter()
        .map(|r| {
            json!({ "file": r.path.file_name().map(|f| f.to_string_lossy().to_string()),
                    "key": r.key, "values_sha256": r.sidecar["values_sha256"],
                    "records": r.records.len() })
        })
        .collect();
    let s0 = &shards.readers[0].sidecar;
    let stamps: Value = student::record::STAMPS
        .iter()
        .map(|&k| (k.to_string(), s0[k].clone()))
        .collect::<serde_json::Map<_, _>>()
        .into();
    let sha = write_shard(
        &out,
        &rows,
        json!({
            "role": shards.role(),
            "teacher_shards": teacher,
            "teacher_stamps": stamps,
            "allowed_stamps": allow,
            "equity_sha256": eq_sha,
            "deal_weights_sha256": weights_sha,
            "decisions_merged": shards.groups.iter().filter(|g| g.len() > 1).count(),
        }),
    )?;
    println!(
        "{} decisions from {} shards ({}) -> {} ({sha})",
        rows.len(),
        shards.readers.len(),
        shards.role(),
        out.display()
    );
    Ok(())
}

fn eval(a: &Args) -> Result<(), String> {
    let net_path = a.req("--net")?;
    let net =
        StudentNet::from_bytes(&std::fs::read(net_path).map_err(|e| format!("{net_path}: {e}"))?)?;
    let fp = a.req("--features")?;
    let rows = features::read(&std::fs::read(fp).map_err(|e| format!("{fp}: {e}"))?)?;
    let mut out: Vec<i64> = Vec::with_capacity(rows.len() * MAX_CANDS * 4);
    for r in &rows {
        let z = net.logits(&r.view, &r.cands);
        for c in 0..MAX_CANDS {
            out.extend(z.get(c).copied().unwrap_or([0; 4]));
        }
    }
    let mut b = npy_i64_header(&[rows.len(), MAX_CANDS, 4]);
    for x in out {
        b.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(a.req("--out")?, b).map_err(|e| e.to_string())?;
    Ok(())
}

fn npy_i64_header(shape: &[usize]) -> Vec<u8> {
    let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
    let mut h = format!(
        "{{'descr': '<i8', 'fortran_order': False, 'shape': ({},), }}",
        dims.join(", ")
    );
    let unpadded = 10 + h.len() + 1;
    h.push_str(&" ".repeat((64 - unpadded % 64) % 64));
    h.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    out
}

fn fixture(a: &Args) -> Result<(), String> {
    use engine::game::{Match, Rules};
    let count = a.num("--count", 64)? as usize;
    let seed = a.num("--seed", 1)?;
    let leaf = nn::synthetic::SyntheticLeaf;
    let mut rng = engine::rng::Rng::new(seed);
    let mut rows = Vec::with_capacity(count);
    let mut k = 0u64;
    while rows.len() < count {
        let mut m = Match::new(seed.wrapping_mul(1000) + k, Rules::HOUSE);
        k += 1;
        while let Some(seat) = m.to_act() {
            let v = m.view(seat);
            let legal = m.legal_actions();
            if (1..=2).contains(&v.public.trick_no)
                && v.public.trump.is_some()
                && legal.len() >= 2
                && matches!(v.public.phase, engine::game::Phase::AwaitPlay(_))
                && rng.bounded(3) == 0
                && rows.len() < count
            {
                let f = encode_view(&v, &leaf)?;
                let cands: Vec<Vec<u16>> =
                    legal.iter().map(|&x| encode_cand(&v, (x, None))).collect();
                let targets = legal
                    .iter()
                    .map(|_| {
                        MoneyChance::from_weights(core::array::from_fn(|_| {
                            u128::from(rng.next_u64() >> 40) + 1
                        }))
                    })
                    .collect();
                rows.push(FeatureRow {
                    view: f,
                    cands,
                    targets,
                    sd: vec![0; legal.len()],
                    n_deals: 1,
                    trick: v.public.trick_no,
                    place: v.public.trick_cards.len() as u8,
                    stratum: 1,
                    source: 9,
                    kind: 0,
                    pos_key: rng.next_u64(),
                });
            }
            let act = bots::decide(&v, &Rules::HOUSE, &legal).0;
            m.apply(act).expect("legal");
        }
    }
    let out = PathBuf::from(a.req("--out")?);
    let role = a.get("--role").unwrap_or("fixture");
    write_shard(&out, &rows, json!({ "role": role, "seed": seed }))?;
    Ok(())
}
